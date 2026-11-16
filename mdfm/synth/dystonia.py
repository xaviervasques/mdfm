"""Sustained abnormal posture, dystonic or spastic not separable (docs/proposals/SYNTH_PATIENT_DRAFT.md,
section 6; clinical note 4.5; design section 6: not called dystonia until a dystonia-specific test passes).

Inverse recipe (draft section 6; DR-14.2, DR-14.3 of docs/rules/DESCRIPTOR_RULES_DRAFT.md): a rotation offset of
one region about one axis, built up over a ramp, held for the requested share of time, released between
bouts. The end segment of the region deviates from the level-0 body by exactly the offset (the D(t) of DR-14.2
with the level-0 body as the norm), so TON-T and TON-DW recover the level and the share by construction.
Phasic components (jerks, mobile dystonia, dystonic tremor) are not in version 1 (no sourced band). Task
dependence (POST-TD) and overflow (OVF-K) are schedules: a caller applies the operator with a schedule per
exam segment, or a second time on a segment the task does not use.

Regions (REGIONS): a joint chain, an axis and a sign per direction, and the frame of the offset.
- neck (TWSTRS): head on trunk over the neck (12) and head (15) joints; rotation_left (+y) and rotation_right,
  laterocollis tilt_left (-z) and tilt_right, anterocollis (+x), retrocollis (-x). Chain offsets in each
  segment's own frame, the last joint taking the exact remainder so that the head deviates by the offset.
- shoulder_elevation (TWSTRS): the collar (13 left, 14 right) about z (+z raises the left, -z the right).
- trunk (DIS): spine1, spine2, spine3 (3, 6, 9); flexion (+x), extension, lateral_left (-z), lateral_right,
  rotation_left (+y), rotation_right.
- proximal_arm (DIS): the shoulder (16, 17): abduction (left +z, right -z), adduction.
- distal_arm (DIS): the elbow hinge (18, 19; flexion left -y, right +y) and the wrist (20, 21; flexion left -z,
  right +z).
- proximal_leg (DIS): the hip (1, 2): flexion (-x), extension, abduction (left +z, right -z), adduction.
- distal_leg (DIS): the knee (4, 5; flexion +x) and the ankle (7, 8; plantarflexion +x, dorsiflexion -x).
Single joints take the offset in the parent's frame (a hinge fixed in the proximal segment); the deviation
angle of the moved segment is the offset either way.

Levels (grades): TWSTRS rotation and laterocollis in degrees at the centre of their bands, the open band of
laterocollis grade 3 closed by the possible range; anterocollis, retrocollis and shoulder elevation by thirds
of the possible range; DIS amplitude as a share of the possible range, at the centre of the DIS band. The
possible range of a direction is the room the SP1 joint-limit table leaves from the base posture to the
table limit in that direction, summed over the region's joints (decision D3, the table as the denominator;
reading the DIS share against that directional range, not against the whole hi - lo span, is a module
reading: against the whole span a joint with its neutral mid-range could never reach the upper DIS bands in
one direction, and the share of the whole span is reported beside). A chain splits its offset in proportion
to each joint's room. Duration: the share of time with the posture held (TWSTRS
duration bands crossed with maximal or submaximal excursion, submaximal at half the level, a module value;
DIS duration bands), as raised-cosine bouts (common.schedule, ramp_s 1 s by default, a module value).

Bounds: an offset larger than the room the table leaves in that direction is clipped to that room
(reachable False in the truth); at each knot the offset is scaled down where a joint would leave the table
or the base's own excursion (limit_scale in the truth), so that no new violation is introduced.

Feet (leg regions, contact.py; version sp4-v3): a leg posture on a body whose foot of that side stands on the
floor keeps it there (foot 'plant', the default): an ankle posture rolls the foot on the floor and re-solves the
leg above it with the hip fixed (contact.plant_ankle; the ankle's flexion is the level, the hip and knee
compensate); a knee or hip posture keeps the operator's leg down to the ankle and flexes the ankle so that the
foot's lowest sole point stays on the floor (contact.plant_compensate). Where the foot cannot stay planted within
the joint-limit table, the posture's level is scaled down uniformly over the clip, to the largest scale found by
bisection at which every knot is planted and inside the table (plant_scale_min in the truth, reachable False). Hip flexion lifts the leg by design (foot 'raise', its default): the foot may leave the floor,
never go below it. foot 'free' is the version-2 operator (no planting, contacts relabelled). The truth's
'contact' check (contact.sole_report) reads the lowest sole point of both feet against the base at every knot:
within 5 mm for a planted foot and for the other foot, never more than 5 mm below for a raised one. For an
ankle posture the deviation reported is the ankle joint's (its local rotation against the base's), since the
foot's global orientation also carries the shank's compensation.

What planting costs is in the truth (review of 2026-10-05, contact.py): params 'foot_pose' (heel and toe lift,
foot pitch, hip, knee and ankle turned from the base) for every leg posture, and the validity 'continuity' check
(no leg joint faster than contact.MAX_LEG_SPEED_DEG_S unless the base is). A planted knee or hip posture on a
standing leg with the pelvis fixed is held on the toes (the compensating ankle flexion lifts the heel): when its
heel rises more than contact.TOL_M, params 'toe_stance' is True and 'grading_note' says that the clip is kept out of
DIS grading until the neurologist decides how such a posture is graded (a DIS distal-leg rating would read the
compensating equinus too). A standing ankle posture that ramps jumps at its first planted knot (contact.py): its
continuity check fails.

The seated hands (plan phase E, the gallery on the open body): a planted leg posture on a seated base (the knee of that
side flexed past SEATED_KNEE_DEG) raises the thigh under the hand of the 'sit', which lies over it 46 mm above the open
body's skin: the plantarflexed ankle of DIS grade 4 lifts the heel and the knee and ran the thigh 6.5 mm into the left
hand on the mesh (the page's rigid thigh read the thumb tip 13.6 mm inside). Each hand whose forearm or hand comes
nearer than reach.CLEAR_M to the thighs at the knot where the posture peaks is lifted by flexing its shoulder (about the
parent's x axis, the arm forward) by the least angle on the ARM_LIFT_STEP_DEG grid that keeps reach.CLEAR_M there
(arm_lift, read on the skinned mesh by postures.arm_part_gap), on the posture's own time course, so that the hand rises
with the thigh; params 'arm_lift' (only when a hand is lifted: a posture that keeps the hands clear leaves the arms
and the truth as they were).

realized_peak_deg: the largest offset the clip holds, peak x max over knots of (envelope x knot scale), so that a
posture scaled down at the table's limits or by the planted foot reports the level it reaches.

Joint-limit table (SH2, limits_ext.py; version sp5-v6 plus SH2): the operator reads the dystonia table, the healthy SP1
table with three targeted extensions (ankle inversion, toe flexion, the shoulder's adduction coupled to the collar's
rise over the base), for its rooms, its per-knot scaling and its validity; table='healthy' gives the SP1 table of the
versions before SH2. params 'limits_table' records which table and, for the dystonia one, the widened bounds and their
sources. The regions of REGIONS do not move the ankle about z nor the toes, and the shoulder's coupled bound only opens
under a collar raised over the base (the shoulder elevation's own room is the collar's bound, kept), so the rooms and
levels of every region are those of the healthy table on any base.

Shoulder elevation and the arm (SH2): with arm 'hang' (the default) the shoulder of the raised side turns back by the
collar's offset, so that the whole arm keeps the global rotation it has in the base: the shoulder joint rises with
the clavicle and the arm hangs under it (TWSTRS D, a shrug), the adduction this needs read against the coupled bound;
arm 'ride' is the operator before SH2 (the arm turns with the collar and swings out).

Trunk and the rest of the body (wave SH1d, review M6 of 2026-11-03): a trunk posture bends the spine (3, 6, 9) only,
and with arm 'ride' everything above it turns with it as one block: the hanging arms swing out on one side and into the
thighs on the other, the head tilts by the whole bend, and the pelvis stays put while the centre of mass moves over
one foot. With arm 'hang' (the default) the body answers the bend as a relaxed standing body does, each answer a
passive or postural response that a DIS trunk rating does not read (the trunk's deviation, the level and the share
are those of 'ride' exactly):
- the arms hang (hang_from_trunk): at each knot each shoulder (16, 17) turns its humerus by the minimal rotation back
  onto the world direction it has in the base (the shoulder turned back by the bend's change of the trunk's global
  rotation), the largest share of the way that keeps the shoulder inside the table; where that share falls below 1 at
  some knot, the clip's smallest share is taken at every knot, so that the shoulder's turn grows and shrinks with the
  bend instead of filling the table's room early in a ramp and stopping there (a kink of the wrist's path at every
  onset and release, review of SH1d). In a lateral bend the arm on the side of the bend abducts at the shoulder and
  hangs as in the base, clear of the thighs; the other arm cannot adduct at the shoulder (the table leaves the hanging
  shoulder almost no adduction: shoulder_share_min in the truth), so its shoulder girdle drops under it
  (girdle_drop): at the knot of the largest bend the collar (13, 14) is lowered about the axis of its elevation, on
  steps of DROP_STEP_DEG, until the first of four bounds: the forearm or the hand comes to rest against the trunk or
  a thigh (rest_gap on the skinned mesh down to REST_GAP_M), the upper arm presses into the chest deeper than the
  validity allows (UPPER_ENTRY, below), the humerus comes nearest its world direction in the base, or the girdle's
  depression range (GIRDLE_DEPRESSION_DEG, jarm2022
  of limits_ext.SOURCES) is used up; at every knot the collar takes that drop times the knot's share of the largest
  bend. The elbows, wrists and hands keep their local rotations. In the larger bends that arm stays off the vertical:
  the forearm rests against the lower chest just below the elbow and the rest of the forearm and the hand stay off
  the body, the angle between the humerus and the vertical recorded in the truth (humerus_to_vertical_deg, beside
  the base's). The side of the chest holds that arm there: it cannot come nearer the vertical without entering the
  trunk (2 deg more shoulder adduction puts the forearm inside it, tested), as the arm on the convex side of a
  lateral trunk flexion lies against the chest wall. Where the base is not standing (STAND_THIGH_DEG) the arms ride
  (a seated forearm rests on the chair or the thighs);
- the head rights itself in the bends that postures.py rights (RIGHTING_DIRS: flexion and the lateral bends; an
  axial rotation does not tilt the head, and postures.py has no righting of a backward bend): the neck (12, 15) turns
  back about the bend's axis by the share head_righting(direction) of the trunk's offset at each knot, read in
  postures.py so that the two modules keep one value (a lateral bend: PISA_HEAD_COMP, the share of the Pisa postures;
  a flexion: the head's extension in HEAD_COMP of the spinal bends without the hip, 'lumbar' and 'thoracic', which
  agree; module values), split by the neck's rooms, the share lowered to the neck's room where that is smaller
  and the knots scaled down where a neck joint would leave the table;
- standing, the pelvis moves so that the centre of mass keeps its place over the ankles (postures.plant_stand with
  com_keep, the solve of the parkinsonian axial postures): the ankles and the feet's global rotations of the base are
  kept and the legs re-solved, the more extended knee held at the base's own flexion and the other bending as the
  pelvis shifts. Where the posture is absent (envelope times knot
  scale 0) the clip is its base bit for bit; plant_stand's own solve of the base differs from it by under 0.1 mm,
  the step at the start of a ramp.
Validity adds the planted feet and the leg continuity of postures.stand_validity (standing; stand_checks reads them
once per distinct knot) and, on the skinned mesh at every CLEAR_EVERY-th knot where the posture is present, the least
signed distance of each forearm and hand to the thighs ('hand_thigh') and to the trunk ('hand_trunk') (rest_gap):
'ok' when nothing enters deeper than the base's own entry by more than CLEAR_TOL_M, and of the distal half of each upper
arm (UPPER_FROM) to the trunk (UPPER_ENTRY): the arm and the chest wall are soft tissue pressed together, which the
rigid mesh passes through (the base already reads its hanging upper arm about 2 cm inside the chest), so 'ok' when it
enters no deeper than legcontact.SOFT_TOL_M, the depth the inner thighs may press into each other, or than the base's
own entry by CLEAR_TOL_M where that is deeper; the girdle's drop stops there too. rest_gap reads the arm's vertices
against the part's triangles only, a vertex negative only where the winding number of the body without that arm puts
it inside (WIND_IN): reach.clearance also reads the part's vertices against the arm's triangles, open at the elbow,
and that reading turns negative beyond the open edge where nothing enters. Each step of
the answer works knot by knot, so it is solved once per distinct knot of the clip (a held posture on a still stand:
a few knots) and copied to the knots that repeat it (the collar's drop, read at one knot, is passed with it). arm
'ride' is the trunk operator before SH1d, the operator of version sp5-v6: the probe stimuli ask for it
(stimuli.V6_ARM), so that they keep the bodies the SP3 runs were measured on.
"""
import functools
from dataclasses import replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d
from ..body.joints import fk
from . import contact as CT
from . import grades as G
from . import limits_ext as LX
from .body import (LIMIT_TOL_DEG, F64, Body, axis_rotation, compose, components_deg, contact_labels, deviation,
                   room_deg, rotation_angle_deg)
from .common import as_rng, envelope, realized_share, schedule, truth_record
from .ik import rmin

RAMP_S = 1.0               # [module] build-up and release of the posture
TABLE = LX.DYSTONIA        # the operator's joint-limit table (module docstring)
ARM_MODES = ("hang", "ride")
SHOULDER_OF_COLLAR = {13: 16, 14: 17}
ARM_REGIONS = ("shoulder_elevation", "trunk")      # the regions that take arm 'hang' or 'ride'
# the trunk's answer (module docstring, Trunk and the rest of the body)
TRUNK_ARMS = {"left": (13, 16, 18), "right": (14, 17, 19)}   # collar, shoulder, elbow of each side
HANG_BISECT = 20           # [module] bisection steps of the hanging share at each knot
GIRDLE_DEPRESSION_DEG = 10.0   # shoulder girdle depression 0-10 deg (limits_ext.SOURCES['jarm2022']): the most the
#                                collar of an arm the shoulder cannot bring back may drop (module docstring)
DROP_STEP_DEG = 0.25       # [module] the steps on which the collar's drop is read on the mesh
REST_GAP_M = 0.001         # [module] the least gap the dropped forearm and hand keep to the trunk and the thighs at the
#                            knot of the largest bend (they come to rest on them)
NECK = (12, 15)
RIGHTING_DIRS = ("flexion", "lateral_left", "lateral_right")   # the bends the head rights (head_righting)
STAND_THIGH_DEG = 45.0     # [module] a base stands when both feet touch the floor and both thighs lie within this of
#                            the vertical at every knot (a seated thigh lies near the horizontal)
SEATED_KNEE_DEG = 45.0     # [module] a base is seated on a side when that knee is flexed past this at its first knot
#                            (postures.limb_posture's seated leg)
ARM_LIFT_STEP_DEG = 1.0    # [module] the grid of the seated hand's lift (arm_lift; module docstring, The seated hands)
ARM_LIFT_MAX_DEG = 30.0    # [module] the largest lift arm_lift tries (ValueError beyond)
CLEAR_EVERY = 10           # [module] knots between two mesh clearance readings (10 Hz at 100 Hz)
CLEAR_TOL_M = 0.001        # [module] the depth a forearm or hand may take inside the thighs or the trunk beyond the
#                            base's own entry (a rounding allowance of the mesh reading)
CLEAR_PARTS = {"hand_thigh": "thighs", "hand_trunk": "trunk"}   # validity entry: reach.PARTS surface
WIND_IN = 0.5              # [module] the winding number above which a point lies inside the body (about 1 inside, 0
#                            outside; module docstring, rest_gap)
UPPER_FROM = 0.5           # [module] the upper arm read against the trunk: the shoulder's vertices farther along the
#                            humerus than this share of it on the rest mesh (its distal half, clear of the armpit)
UPPER_ENTRY = "upper_arm_trunk"   # validity entry of the upper arm against the trunk (module docstring)


def head_righting(direction="lateral_left"):
    """The share of a trunk bend in direction (RIGHTING_DIRS) the neck turns back, read in postures.py so that its
    trunk postures and the dystonic trunk keep one value: a lateral bend, PISA_HEAD_COMP (the Pisa postures); a
    flexion, the head's extension of HEAD_COMP's spinal bends without the hip ('lumbar' and 'thoracic', which agree:
    the gaze kept up)."""
    from .postures import HEAD_COMP, PISA_HEAD_COMP
    if direction in ("lateral_left", "lateral_right"):
        return float(PISA_HEAD_COMP)
    if direction == "flexion":
        shares = {-(nk + hd) for nk, hd in (HEAD_COMP["lumbar"], HEAD_COMP["thoracic"])}
        assert len(shares) == 1, shares
        return float(shares.pop())
    raise ValueError(f"no head righting for a trunk bend in {direction!r} (RIGHTING_DIRS: {RIGHTING_DIRS})")


REGIONS = {
    "neck": {"chain": (12, 15), "frame": "child", "scale": "TWSTRS",
             "dirs": {"rotation_left": ("y", 1.0, "rotation"), "rotation_right": ("y", -1.0, "rotation"),
                      "tilt_left": ("z", -1.0, "laterocollis"), "tilt_right": ("z", 1.0, "laterocollis"),
                      "anterocollis": ("x", 1.0, "anterocollis"), "retrocollis": ("x", -1.0, "retrocollis")}},
    "shoulder_elevation": {"chain": {"left": (13,), "right": (14,)}, "frame": "parent", "scale": "TWSTRS",
                           "dirs": {"elevation": ("z", {"left": 1.0, "right": -1.0}, "shoulder_elevation")}},
    "trunk": {"chain": (3, 6, 9), "frame": "child", "scale": "DIS",
              "dirs": {"flexion": ("x", 1.0, None), "extension": ("x", -1.0, None),
                       "lateral_left": ("z", -1.0, None), "lateral_right": ("z", 1.0, None),
                       "rotation_left": ("y", 1.0, None), "rotation_right": ("y", -1.0, None)}},
    "proximal_arm": {"chain": {"left": (16,), "right": (17,)}, "frame": "parent", "scale": "DIS",
                     "dirs": {"abduction": ("z", {"left": 1.0, "right": -1.0}, None),
                              "adduction": ("z", {"left": -1.0, "right": 1.0}, None)}},
    "distal_arm": {"chain": {"left": (18,), "right": (19,)}, "frame": "parent", "scale": "DIS",
                   "dirs": {"elbow_flexion": ("y", {"left": -1.0, "right": 1.0}, None),
                            "elbow_extension": ("y", {"left": 1.0, "right": -1.0}, None)}},
    "wrist": {"chain": {"left": (20,), "right": (21,)}, "frame": "parent", "scale": "DIS",
              "dirs": {"flexion": ("z", {"left": -1.0, "right": 1.0}, None),
                       "extension": ("z", {"left": 1.0, "right": -1.0}, None)}},
    "proximal_leg": {"chain": {"left": (1,), "right": (2,)}, "frame": "parent", "scale": "DIS",
                     "dirs": {"flexion": ("x", -1.0, None), "extension": ("x", 1.0, None),
                              "abduction": ("z", {"left": 1.0, "right": -1.0}, None),
                              "adduction": ("z", {"left": -1.0, "right": 1.0}, None)}},
    "distal_leg": {"chain": {"left": (4,), "right": (5,)}, "frame": "parent", "scale": "DIS",
                   "dirs": {"knee_flexion": ("x", 1.0, None), "plantarflexion_ankle": ("x", 1.0, None),
                            "dorsiflexion_ankle": ("x", -1.0, None)}},
}
ANKLE = {"left": (7,), "right": (8,)}
LEG_REGIONS = ("proximal_leg", "distal_leg")
FOOT_MODES = ("plant", "raise", "free")
PLANT_BISECT = 14           # bisection steps on the uniform scale of a leg posture that cannot stay planted
GRADING_NOTE = ("held on the toes: with the pelvis fixed the planted foot compensates by ankle flexion and the heel "
                "leaves the floor, which a DIS distal-leg rating would also read; kept out of DIS grading until the "
                "neurologist decides how such a posture is graded")


def default_foot(region, direction):
    """The foot rule of a leg posture (module docstring): 'raise' for hip flexion, 'plant' otherwise; None for
    a region that does not move a leg."""
    if region not in LEG_REGIONS:
        return None
    return "raise" if (region == "proximal_leg" and direction == "flexion") else "plant"



def _resolve(region, direction, side):
    reg = REGIONS[region]
    if direction not in reg["dirs"]:
        raise ValueError(f"{region}: direction {direction!r}, expected one of {sorted(reg['dirs'])}")
    axis, sign, kind = reg["dirs"][direction]
    sign = sign[side] if isinstance(sign, dict) else sign
    chain = reg["chain"][side] if isinstance(reg["chain"], dict) else reg["chain"]
    if region == "distal_leg" and direction.endswith("_ankle"):
        chain = ANKLE[side]
    return chain, axis, float(sign), kind, reg["frame"], reg["scale"]


def rooms(body, chain, axis, sign, frame="child", k=0, table=TABLE):
    """Degrees of offset each joint of the chain can take from knot k of the base in the direction before a
    component leaves the table (body.room_deg; table 'dystonia' or 'healthy', limits_ext, read from the base itself,
    where no collar has risen over the base: the shoulders keep their healthy adduction bounds): a chain's joints take
    their split in their own frame, a single joint in the region's frame."""
    R = body.local_R()[k]
    box = LX.box_at(table)
    fr = "child" if len(chain) > 1 else frame
    return [room_deg(R[j], j, axis, sign, fr, limits=box) for j in chain]


def rom(chain, axis, table=TABLE):
    """Range of motion (deg) of a chain on an axis: the sum of hi - lo over its joints (the DIS denominator,
    D3), in table at rest collars."""
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    lim = LX.box_at(table)
    return float(sum(lim.rom[j, ax] for j in chain))


def hang_from_collar(R0, R, chain):
    """(T, 22, 3, 3): R with the shoulder under each collar of chain (13, 14) turned back by the collar's change from
    R0, so that the shoulder keeps its rotation in the collar's parent frame (R0's): the arm keeps hanging as in R0
    while the collar rises (module docstring)."""
    Rn = R.clone()
    for c in chain:
        s = SHOULDER_OF_COLLAR.get(c)
        if s is not None:
            Rn[:, s] = R[:, c].transpose(-1, -2) @ R0[:, c] @ R0[:, s]
    return Rn


def stands(body):
    """True when the body stands at every knot: both feet on the floor (its contact labels) and both thighs within
    STAND_THIGH_DEG of the vertical, the knee below the hip."""
    if not bool(body.contact.all()):
        return False
    J = body.joints()
    for h, k in ((1, 4), (2, 5)):
        u = J[:, k] - J[:, h]
        if bool((-u[:, 1] / u.norm(dim=-1) < np.cos(np.radians(STAND_THIGH_DEG))).any()):
            return False
    return True


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _inside(M, lo, hi):
    """(T,) bool: every axis-angle component (deg) of the rotations M (T, 3, 3) inside [lo, hi] (T, 3)."""
    c = np.degrees(matrix_to_axis_angle(M).numpy())
    return ((c >= lo - LIMIT_TOL_DEG) & (c <= hi + LIMIT_TOL_DEG)).all(-1)


def _largest_share(make, ok, T, steps=HANG_BISECT):
    """(T,) in [0, 1]: at each knot the largest share u whose rotations make(u) pass ok (1 where the whole way does;
    bisection otherwise, share 0 being the rotation before the step, which passes)."""
    full = ok(make(np.ones(T)))
    a, b = np.zeros(T), np.ones(T)
    for _ in range(steps):
        m = 0.5 * (a + b)
        f = ok(make(m))
        a, b = np.where(f, m, a), np.where(f, b, m)
    return np.where(full, 1.0, a)


def _turned(aa, u, R):
    """(T, 3, 3): the rotations axis-angle aa (T, 3) scaled by u (T,), applied before R (T, 3, 3)."""
    return axis_angle_to_matrix(aa * torch.as_tensor(np.asarray(u, dtype=np.float64))[:, None]) @ R


def hang_from_trunk(base, R, table=TABLE, sides=("left", "right"), shares=None):
    """(R, info): R (T, 22, 3, 3) with each arm hanging as in base while the trunk bends (module docstring, Trunk and
    the rest of the body): the shoulder turns the humerus by the minimal rotation back onto its world direction in
    base, the largest share of the way inside the table at each knot (limits_ext.knot_bounds, the base's own excursion
    allowed) and, where that falls below 1 at some knot, the smallest of them at every knot; shares ({side: share})
    gives that share instead (a solve on some knots of a clip). The collar, the elbow, the wrist and the hand keep
    their local rotations. info per side: the share and the largest angle (deg) left between the humerus and its base
    direction."""
    rest = base.rest()
    T = R.shape[0]
    _, Gb = base.fk()
    cb = np.degrees(matrix_to_axis_angle(base.local_R()).numpy())
    Rn = R.clone()
    lo, hi = LX.knot_bounds(table, np.degrees(matrix_to_axis_angle(Rn).numpy()), cb)
    with torch.no_grad():
        _, G = fk(Rn, base.transl, rest)
    info = {}
    for sd in sides:
        c, s, e = TRUNK_ARMS[sd]
        hx = _unit(rest[e] - rest[s])
        tw = _unit(torch.einsum("tij,j->ti", Gb[:, s], hx))                 # the humerus of the base, in the world
        h0 = torch.einsum("tij,j->ti", Rn[:, s], hx)                        # the riding humerus, in the collar's frame
        tg = torch.einsum("tji,tj->ti", G[:, c], tw)                        # its target, in the collar's frame
        aa = matrix_to_axis_angle(rmin(_unit(h0), _unit(tg)))
        Rs = Rn[:, s].clone()
        if shares is not None and sd in shares:
            u = np.full(T, float(shares[sd]))
        else:
            u = _largest_share(lambda x: _turned(aa, x, Rs), lambda M: _inside(M, lo[:, s], hi[:, s]), T)
            u = np.full(T, u.min())                       # one share over the clip: the turn follows the bend
        Rn[:, s] = _turned(aa, u, Rs)
        hn = torch.einsum("tij,j->ti", G[:, c] @ Rn[:, s], hx)
        res = torch.rad2deg(torch.acos((_unit(hn) * tw).sum(-1).clamp(-1.0, 1.0)))
        info[sd] = {"shoulder_share_min": float(u.min()), "residual_deg_max": float(res.max())}
    return Rn, info


def _lowered(R, side, deg):
    """(..., 3, 3): the collar rotations R of side lowered by deg (...) about the axis of its elevation, in the
    parent's frame (REGIONS['shoulder_elevation'], the direction reversed)."""
    up = REGIONS["shoulder_elevation"]["dirs"]["elevation"][1][side]
    return compose(R, axis_rotation("z", torch.as_tensor(-up * np.radians(deg), dtype=F64)), "parent")


@functools.lru_cache(maxsize=2)
def _arm_free_faces(side):
    """(F, 3): the faces of the selected body model's mesh with no vertex skinned mostly by side's shoulder, elbow,
    wrist or hand joints (the body without that arm, the surface the winding number of rest_gap reads); the open
    body's inner parts (dominant joint -1, design section 12.3) left out, as physio's body surface leaves them (D35).
    Cached per side: one body model per process (design D18)."""
    from .hands import FIRST, WRIST
    from .viewer import dominant_joint, faces
    f = faces()
    dom = dominant_joint("neutral")
    _, s, e = TRUNK_ARMS[side]
    arm = np.isin(dom, (s, e, WRIST[side]) + tuple(range(FIRST[side], FIRST[side] + 15))) | (dom < 0)
    return f[~arm[f].any(1)]


def _winding(P, V, F, chunk=64):
    """(N,): the winding number of the triangle mesh (V, F) at the points P (N, 3): the solid angle its triangles
    subtend at each point (signed, the triangle's outward side positive) over 4 pi."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk, None]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = (a * np.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        out[i:i + chunk] = (2.0 * np.arctan2(det, den)).sum(1) / (4.0 * np.pi)
    return out


@functools.lru_cache(maxsize=2)
def _upper_ids(side):
    """The vertices of side's upper arm read against the trunk: skinned mostly by the shoulder and farther along the
    humerus than UPPER_FROM of it on the selected body model's rest mesh."""
    from .body import rest_joints
    from .viewer import dominant_joint, rest_vertices
    v = rest_vertices()
    r = rest_joints().numpy()
    _, s, e = TRUNK_ARMS[side]
    ax = r[e] - r[s]
    return np.flatnonzero((dominant_joint("neutral") == s) & (((v - r[s]) @ ax / (ax @ ax)) > UPPER_FROM))


def rest_gap(body, side, knots=None, parts=("trunk", "thighs"), chunk=16, upper=False):
    """{part: {'hand': (K,), 'fore': (K,)}} m: at the knots (default all), the least signed distance of the vertices
    of side's hand (wrist and fingers) and forearm to each part's triangles (reach.PARTS, the skinned mesh), read one
    way: a vertex reads negative only when it lies inside the body without that arm (winding number above WIND_IN),
    its distance otherwise; upper adds {'trunk': {'upper': (K,)}}, the distal upper arm (_upper_ids) against the trunk
    (review of SH1d). reach.clearance reads both ways, and its reverse reading (the part's vertices against the
    arm's triangles, open at the elbow and the wrist) turns negative beyond those open edges where nothing enters;
    the one-way reading needs the winding test because the part's own triangles are open too (review of SH1d)."""
    from .hands import body_points
    from .reach import _signed_to_mesh, _surface_sets
    from .viewer import knot_vertices
    sets = dict(_surface_sets(side), upper=(_upper_ids(side), None))
    k = np.arange(body.T) if knots is None else np.asarray(knots, dtype=np.int64).reshape(-1)
    upper = bool(upper and "trunk" in parts)
    names = ("hand", "fore") + (("upper",) if upper else ()) + tuple(parts)
    ids = np.concatenate([sets[n][0] for n in names])
    off = np.cumsum([0] + [len(sets[n][0]) for n in names])
    arms = {p: ("hand", "fore") + (("upper",) if upper and p == "trunk" else ()) for p in parts}
    out = {p: {a: np.empty(len(k)) for a in arms[p]} for p in parts}
    for a in range(0, len(k), chunk):
        X = body_points(body, ids, knots=k[a:a + chunk])
        for i in range(len(X)):
            S = {n: X[i, off[j]:off[j + 1]] for j, n in enumerate(names)}
            V = None
            for p in parts:
                for arm in arms[p]:
                    d = _signed_to_mesh(S[arm], S[p], sets[p][1])
                    neg = np.flatnonzero(d < 0.0)
                    if len(neg):
                        if V is None:
                            V = knot_vertices(body, [int(k[a + i])])[0]
                        w = _winding(V[sets[arm][0][neg]], V, _arm_free_faces(side))
                        d[neg] = np.where(w > WIND_IN, d[neg], -d[neg])
                    out[p][arm][a + i] = d.min()
    return out


def _gap(cl):
    """(K,) m: the least of the hand's and the forearm's readings over the parts of a rest_gap result."""
    return np.minimum.reduce([v for c in cl.values() for a, v in c.items() if a in ("hand", "fore")])


def _upper_allowed(base_m):
    """m (negative): the deepest reading of the upper arm against the trunk that passes, given the base's own reading
    base_m: the soft tissue of the arm and the chest wall pressed together (legcontact.SOFT_TOL_M, the depth the inner
    thighs may press into each other), or the base's own entry and CLEAR_TOL_M where that is deeper."""
    from .legcontact import SOFT_TOL_M
    return np.minimum(np.asarray(base_m) - CLEAR_TOL_M, -SOFT_TOL_M)


def girdle_drop(base, R, side, k):
    """(deg, info): the drop of side's collar at knot k of the bent rotations R (T, 22, 3, 3) on base (module
    docstring, Trunk and the rest of the body): the largest drop on the steps of DROP_STEP_DEG up to the one that
    brings the humerus nearest its world direction in base (at most GIRDLE_DEPRESSION_DEG) at which the forearm and
    the hand keep REST_GAP_M off the trunk and the thighs (rest_gap; at every step up to it, and never nearer than
    with no drop) and the upper arm enters the trunk no deeper than _upper_allowed of its entry with no drop."""
    c, s, e = TRUNK_ARMS[side]
    d = np.arange(0.0, GIRDLE_DEPRESSION_DEG + 1e-9, DROP_STEP_DEG)
    rest = base.rest()
    hx = _unit(rest[e] - rest[s])
    _, Gb = base.fk()
    tw = _unit(Gb[k, s] @ hx)
    Rk = R[k].expand(len(d), 22, 3, 3).clone()
    Rk[:, c] = _lowered(Rk[:, c], side, d)
    with torch.no_grad():
        _, G = fk(Rk, base.transl[[k] * len(d)], rest)
    hu = _unit(torch.einsum("tij,j->ti", G[:, s], hx))
    ang = torch.rad2deg(torch.acos((hu @ tw).clamp(-1.0, 1.0))).numpy()
    vert = np.degrees(np.arccos(np.clip(-hu[:, 1].numpy(), -1.0, 1.0)))       # the humerus from the downward vertical
    n = int(np.argmin(ang)) + 1                           # the steps up to the nearest direction
    info = {"humerus_to_base_deg_undropped": float(ang[0]),
            "humerus_to_vertical_deg_base": float(np.degrees(np.arccos(np.clip(-float(tw[1]), -1.0, 1.0))))}
    if n == 1:
        return 0.0, dict(info, girdle_drop_deg=0.0, bound="base direction", humerus_to_base_deg=float(ang[0]),
                         humerus_to_vertical_deg=float(vert[0]))
    kk = [k] * n
    trial = Body(matrix_to_rot6d(Rk[:n]), base.transl[kk], base.contact[kk], base.betas, base.gender, base.rate)
    cl = rest_gap(trial, side, upper=True)
    gap, up = _gap(cl), cl["trunk"]["upper"]
    ok = (gap >= min(gap[0], REST_GAP_M)) & (up >= _upper_allowed(up[0]))
    j = n - 1 if ok.all() else max(int(np.argmin(ok)) - 1, 0)
    bound = "contact" if j < n - 1 else "girdle range" if n == len(d) else "base direction"
    near = min(((p, a) for p in cl for a in cl[p] if a != "upper"), key=lambda x: cl[x[0]][x[1]][j])
    return float(d[j]), dict(info, girdle_drop_deg=float(d[j]), bound=bound, rest_gap_mm=float(1000 * gap[j]),
                             nearest=f"{'forearm' if near[1] == 'fore' else 'hand'} to {near[0]}",
                             upper_arm_in_trunk_mm=float(-1000 * up[j]), humerus_to_base_deg=float(ang[j]),
                             humerus_to_vertical_deg=float(vert[j]))


def _distinct(*arrays):
    """(idx, inv): the first knot of each distinct row of the arrays (T, ...) side by side and, for every knot, the
    index of its row among them (knots[idx][inv] reproduces every knot); at least two knots, so that a Body can be
    built on them."""
    key = np.concatenate([np.asarray(a, dtype=np.float64).reshape(len(a), -1) for a in arrays], 1)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    idx, inv = np.asarray(idx), np.asarray(inv).reshape(-1)
    return (np.r_[idx, idx] if len(idx) == 1 else idx), inv


def _knots(body, k):
    """A Body on the knots k of body (at least two; contacts kept)."""
    k = np.asarray(k, dtype=np.int64)
    k = np.r_[k, k] if len(k) == 1 else k
    return Body(body.rot6d[k], body.transl[k], body.contact[k], body.betas, body.gender, body.rate)


def stand_checks(body, out, table=TABLE):
    """postures.stand_validity read once per distinct knot where it reads knot by knot: limits_ext.check, the skinned
    soles of both feet against the base (contact.sole_report, within contact.TOL_M; its extremes over the distinct
    knots are those over the clip) and the leg continuity (contact.step_report, knot to knot, on the whole clip)."""
    v = LX.check(table, body, out)
    idx, inv = _distinct(body.rot6d.numpy(), body.transl.numpy(), out.rot6d.numpy(), out.transl.numpy())
    k = idx[:int(inv.max()) + 1]
    sb, so = _knots(body, k), _knots(out, k)
    rl, rr = CT.sole_report(sb, so, "left", True), CT.sole_report(sb, so, "right", True)
    v["contact"] = {"ok": bool(rl["ok"] and rr["ok"]), "left": rl, "right": rr, "distinct_knots": int(len(k))}
    sl, sr = CT.step_report(body, out, "left"), CT.step_report(body, out, "right")
    v["continuity"] = {"ok": bool(sl["ok"] and sr["ok"]), "left": sl, "right": sr}
    return v


def trunk_response(body, R, direction, axis, sign, th, table=TABLE, drops=None, shares=None):
    """(Body, info): the trunk posture R (T, 22, 3, 3; the spine rows bent, every other row the base's) with the body's
    answer (module docstring, Trunk and the rest of the body): the head righted (direction in RIGHTING_DIRS), the arms
    hanging and, standing, the pelvis moved by postures.plant_stand(com_keep=True); th (T,) rad the trunk's signed
    offset at each knot (peak x envelope x knot scale, about axis). Every row the answer leaves alone keeps the base's
    rot6d bit for bit, and so does every row at the knots where th is 0. Each step works knot by knot, so the answer
    is solved once per distinct knot (the base's pose and th) and copied to the knots that repeat it; the collars'
    drops at the largest bend (girdle_drop, {side: deg}) are read at the knot of the largest bend unless drops gives
    them, and the shoulders' shares (hang_from_trunk) over the clip unless shares gives them."""
    th = np.asarray(th, dtype=np.float64)
    idx, inv = _distinct(body.rot6d.numpy(), body.transl.numpy(), th)
    out_s, info = _trunk_response(_knots(body, idx), R[idx], direction, axis, sign, th[idx], float(np.abs(th).max()),
                                  stands(body), table, drops, shares)
    r6, transl = out_s.rot6d[inv], out_s.transl[inv]
    still = torch.from_numpy(th == 0)
    r6[still] = body.rot6d[still]
    transl[still] = body.transl[still]
    out = replace(body.copy(), rot6d=r6, transl=transl)
    out = replace(out, contact=contact_labels(out))
    info["distinct_knots"] = int(inv.max()) + 1
    if info["standing"]:
        Jb, Jo = body.joints().numpy(), out.joints().numpy()
        info["pelvis_shift_mm_max"] = float(1000 * np.linalg.norm((Jo[:, 0] - Jb[:, 0])[:, [0, 2]], axis=-1).max())
    return out, info


def _trunk_response(body, R, direction, axis, sign, th, peak, standing, table, drops=None, shares=None):
    """trunk_response on the distinct knots: peak (rad) the trunk's largest offset over the whole clip, standing
    whether the whole clip's base stands (stands), drops the collars' drops at that offset (None: girdle_drop),
    shares the shoulders' shares (None: hang_from_trunk's)."""
    from .postures import plant_stand
    Rn = R.clone()
    base_comp = components_deg(body.rot6d)
    info = {"standing": standing}
    rows = {3, 6, 9}
    if direction in RIGHTING_DIRS:
        rm = rooms(body, NECK, axis, -sign, "child", table=table)
        share = min(head_righting(direction), np.radians(sum(rm)) / peak) if peak > 0 else 0.0
        split = [r / sum(rm) for r in rm] if sum(rm) > 0 else [0.5, 0.5]
        s_k = np.ones(len(th))
        for i in range(31):                               # scale down knots where the neck would leave the table
            Rh = _apply(Rn, NECK, axis, [torch.from_numpy(-share * th * s_k * f) for f in split], "child")
            bad = (LX.excess(table, components_deg(matrix_to_rot6d(Rh)), base_comp, NECK) > 1e-6).any(-1).any(-1)
            if not bad.any() or i == 30:
                break
            s_k = np.where(bad, s_k * 0.95, s_k)
        Rn = Rh
        rows |= set(NECK)
        info["head_righting"] = {"share": float(share), "share_asked": head_righting(direction),
                                 "peak_deg": float(np.degrees(share * np.abs(th * s_k).max())),
                                 "neck_room_deg": float(sum(rm)), "limit_scale_min": float(s_k.min()),
                                 "knots_outside_table": int(bad.sum())}
    if standing:
        Rn, hinfo = hang_from_trunk(body, Rn, table, shares=shares)
        rows |= {16, 17}
        kp = int(np.argmax(np.abs(th)))
        u = np.abs(th) / peak if peak > 0 else np.zeros(len(th))
        held = {}
        for sd in TRUNK_ARMS:                             # the arm the shoulder cannot bring back drops its girdle
            if drops is not None:
                dd = float(drops.get(sd, 0.0))
                held[sd] = {"girdle_drop_deg": dd, "given": True}
            elif hinfo[sd]["shoulder_share_min"] < 1.0 and peak > 0:
                dd, held[sd] = girdle_drop(body, Rn, sd, kp)
            else:
                dd, held[sd] = 0.0, {"girdle_drop_deg": 0.0, "bound": "none needed"}
            if dd > 0:
                c = TRUNK_ARMS[sd][0]
                Rn[:, c] = _lowered(Rn[:, c], sd, dd * u)
                rows.add(c)
            hinfo[sd].update(held[sd])
        info["arms"] = dict(hinfo, rule="each humerus turned back onto its base direction at the shoulder, as far as "
                                        "the shoulder's range allows; an arm the shoulder cannot bring back drops "
                                        "with its shoulder girdle, within the girdle's depression range, until the "
                                        "arm comes to rest against the trunk or a thigh; in the larger bends the "
                                        "forearm rests against the lower chest just below the elbow and the rest of "
                                        "the forearm and the hand stay off the body: the arm on the side away from "
                                        "the bend lies against the side of the chest, which holds it off the "
                                        "vertical")
        knee = torch.minimum(rotation_angle_deg(Rn[:, 4]), rotation_angle_deg(Rn[:, 5])).numpy()
        st, sinfo = plant_stand(body, Rn, knee, com_keep=True)
        rows |= {1, 2, 4, 5, 7, 8}
        info["stand"] = sinfo
        transl = st.transl.clone()
        src6 = st.rot6d
    else:
        info["arms"] = {"rule": "the base does not stand: the arms ride with the trunk"}
        transl = body.transl.clone()
        src6 = matrix_to_rot6d(Rn)
    rows = sorted(rows)
    r6 = body.rot6d.clone()
    r6[:, rows] = src6[:, rows]
    info["rows"] = rows
    return replace(body.copy(), rot6d=r6, transl=transl), info


def _clearance_at(body, side, knots, parts):
    """rest_gap at the knots (the upper arm too), read once per distinct pose among them."""
    k = np.asarray(knots, dtype=np.int64)
    idx, inv = _distinct(body.rot6d[k].numpy(), body.transl[k].numpy())
    c = rest_gap(body, side, k[idx[:int(inv.max()) + 1]], parts=parts, upper=True)
    return {p: {a: v[inv] for a, v in c[p].items()} for p in parts}


def _clearance(body, out, knots, sides=("left", "right")):
    """{entry: {side: {'min_mm', 'at_s', 'base_min_mm', 'new_mm_max'}, 'ok'}} for the entries of CLEAR_PARTS: the
    least signed distance (rest_gap, the skinned mesh, negative inside) of each forearm and hand to the part's
    surface at the knots, the base's at the same knots, and how much deeper than the base's own entry the output goes;
    ok when that stays within CLEAR_TOL_M. UPPER_ENTRY: the distal upper arm against the trunk, ok within
    _upper_allowed of the base's reading at the same knot."""
    k = np.asarray(knots, dtype=np.int64)
    rep = {name: {} for name in tuple(CLEAR_PARTS) + (UPPER_ENTRY,)}
    oks = {name: True for name in rep}
    parts = tuple(CLEAR_PARTS.values())
    for sd in sides:
        co, cb = _clearance_at(out, sd, k, parts), _clearance_at(body, sd, k, parts)
        for name, part in CLEAR_PARTS.items():
            o = np.minimum(co[part]["hand"], co[part]["fore"])
            b = np.minimum(cb[part]["hand"], cb[part]["fore"])
            new = np.maximum(np.minimum(b, 0.0) - o, 0.0)
            i = int(np.argmin(o))
            rep[name][sd] = {"min_mm": float(1000 * o[i]), "at_s": float(k[i] / body.rate),
                             "base_min_mm": float(1000 * b.min()), "new_mm_max": float(1000 * new.max())}
            oks[name] = oks[name] and bool(new.max() <= CLEAR_TOL_M)
        o, b = co["trunk"]["upper"], cb["trunk"]["upper"]
        allow = _upper_allowed(b)
        i = int(np.argmin(o))
        rep[UPPER_ENTRY][sd] = {"min_mm": float(1000 * o[i]), "at_s": float(k[i] / body.rate),
                                "base_min_mm": float(1000 * b.min()), "allowed_mm_at_min": float(1000 * allow[i]),
                                "margin_mm_min": float(1000 * (o - allow).min())}
        oks[UPPER_ENTRY] = oks[UPPER_ENTRY] and bool((o >= allow).all())
    for name in rep:
        rep[name]["ok"] = oks[name]
    return rep


def seated(body, side):
    """True when the knee of side is flexed past SEATED_KNEE_DEG at the body's first knot."""
    kn = 4 if side == "left" else 5
    return float(np.degrees(matrix_to_axis_angle(body.local_R()[0, kn]).numpy())[0]) > SEATED_KNEE_DEG


def with_arm_flexion(body, side, deg):
    """A copy of body with the shoulder of side (16 or 17) flexed by deg (scalar or (T,)) about its parent's x axis
    (the arm forward and up; -x, as the hip's flexion), every other row bit for bit."""
    j = 16 if side == "left" else 17
    a = torch.as_tensor(-np.radians(np.asarray(deg, dtype=np.float64)) * np.ones(body.T), dtype=F64)
    R = body.local_R().clone()
    R[:, j] = compose(R[:, j], axis_rotation("x", a), "parent")
    return body.with_rotations(R, [j])


def arm_lift(out, weight, sides=("left", "right"), step=ARM_LIFT_STEP_DEG, max_deg=ARM_LIFT_MAX_DEG):
    """(Body, info): the seated hands lifted off the thighs (module docstring, The seated hands): at the knot k where
    weight (T,) peaks, each side whose hand or forearm lies nearer than reach.CLEAR_M to the thighs' skin
    (postures.arm_part_gap, the skinned mesh) takes the least shoulder flexion on the step grid that keeps reach.CLEAR_M
    there, on the time course weight / weight[k] (with_arm_flexion). info: the rule, the knot, and per side the lift
    (deg, 0 for a hand left as it was) with the gaps before and after (mm). ValueError past max_deg."""
    from .postures import arm_part_gap
    from .reach import CLEAR_M
    w = np.asarray(weight, dtype=np.float64)
    info = {"rule": "the least shoulder flexion on the grid keeping the hand and forearm reach.CLEAR_M from the thighs "
                    "on the mesh at the posture's peak, on the posture's time course",
            "clear_mm": 1000.0 * CLEAR_M, "step_deg": float(step), "sides": {}}
    if w.max() <= 0.0:
        return out, dict(info, knot_s=None)
    k = int(np.argmax(w))
    wn = w / w[k]
    info["knot_s"] = float(k / out.rate)
    for sd in sides:
        g0 = float(arm_part_gap(out, sd, [k], "thighs")[0])
        deg, g = 0.0, g0
        while g < CLEAR_M:
            deg += float(step)
            if deg > max_deg + 1e-9:
                raise ValueError(f"the {sd} hand stays within {1000 * CLEAR_M:g} mm of the thighs up to {max_deg:g} "
                                 "deg of shoulder flexion")
            g = float(arm_part_gap(with_arm_flexion(out, sd, deg), sd, [k], "thighs")[0])   # wn[k] = 1
        if deg:
            out = with_arm_flexion(out, sd, deg * wn)
        info["sides"][sd] = {"deg": deg, "gap_before_mm": 1000.0 * g0, "gap_after_mm": 1000.0 * g}
    return out, info


def level_deg(region, direction, side, amplitude_grade, base, table=TABLE):
    """(level deg, possible deg) of an amplitude grade: TWSTRS degrees or thirds, DIS share of the possible
    range (module docstring)."""
    chain, axis, sign, kind, frame, scale = _resolve(region, direction, side)
    possible = sum(rooms(base, chain, axis, sign, frame, table=table))
    if scale == "TWSTRS":
        return G.twstrs_level_deg(kind, amplitude_grade, possible), possible
    return G.dis_share(amplitude_grade) * possible, possible


def _apply(R, chain, axis, offsets_rad, frame):
    """Local rotations with a chain offset: joints before the last get their split offset about the axis in
    their own frame; the last takes the remainder so that the end segment deviates by the total exactly."""
    Rn = R.clone()
    total = sum(offsets_rad)
    if len(chain) == 1:
        Rn[:, chain[0]] = compose(R[:, chain[0]], axis_rotation(axis, torch.as_tensor(total)), frame)
        return Rn
    prod_base = torch.eye(3, dtype=F64).expand(R.shape[0], 3, 3)
    prod_new = torch.eye(3, dtype=F64).expand(R.shape[0], 3, 3)
    for j, off in zip(chain[:-1], offsets_rad[:-1]):
        Rn[:, j] = R[:, j] @ axis_rotation(axis, torch.as_tensor(off))
        prod_base = prod_base @ R[:, j]
        prod_new = prod_new @ Rn[:, j]
    last = chain[-1]
    want = prod_base @ R[:, last] @ axis_rotation(axis, torch.as_tensor(total))
    Rn[:, last] = prod_new.transpose(-1, -2) @ want
    return Rn


def dystonic_posture(body, region, direction, side="left", amplitude_grade=None, level=None, duration_grade=None,
                     share=None, excursion="maximal", n_bouts=None, ramp_s=RAMP_S, seed=0, check=True, foot=None,
                     arm=None, table=TABLE):
    """(Body, truth) of a sustained posture (module docstring). Amplitude: amplitude_grade (TWSTRS for neck
    and shoulder elevation, DIS otherwise) or level (deg). Time: duration_grade (TWSTRS 0 to 5 with excursion
    'maximal' or 'submaximal', DIS 0 to 4) or share (0 to 1, default 1). seed: int or numpy Generator (bout
    placement). foot (leg regions): 'plant', 'raise' or 'free' (module docstring, Feet; default
    default_foot). arm (shoulder elevation and trunk): 'hang' or 'ride' (module docstring; default
    'hang'). table: the joint-limit table, 'dystonia' (default) or 'healthy' (limits_ext)."""
    if region not in REGIONS:
        raise ValueError(f"region {region!r}, expected one of {sorted(REGIONS)}")
    chain, axis, sign, kind, frame, scale = _resolve(region, direction, side)
    if region in LEG_REGIONS:
        foot = default_foot(region, direction) if foot is None else foot
        if foot not in FOOT_MODES:
            raise ValueError(f"foot {foot!r}, expected one of {FOOT_MODES}")
    elif foot is not None:
        raise ValueError(f"foot applies to the leg regions {LEG_REGIONS}, not {region!r}")
    if region in ARM_REGIONS:
        if arm is None:
            arm = "hang"
        if arm not in ARM_MODES:
            raise ValueError(f"arm {arm!r}, expected one of {ARM_MODES}")
    elif arm is not None:
        raise ValueError(f"arm applies to the regions {ARM_REGIONS}, not {region!r}")
    seed_val = seed if isinstance(seed, (int, np.integer)) else None
    if (amplitude_grade is None) == (level is None):
        raise ValueError("give exactly one of amplitude_grade and level")
    possible = sum(rooms(body, chain, axis, sign, frame, table=table))
    if amplitude_grade is not None:
        lvl, _ = level_deg(region, direction, side, amplitude_grade, body, table)
    else:
        lvl = float(level)
    exc = "maximal"
    if duration_grade is not None:
        if share is not None:
            raise ValueError("give duration_grade or share, not both")
        if scale == "TWSTRS":
            share, exc = G.twstrs_duration(duration_grade, excursion)
        else:
            share = G.dis_share(duration_grade)
    share = 1.0 if share is None else float(share)
    params = {"region": region, "direction": direction, "side": side, "scale": scale, "chain": list(chain),
              "axis": axis, "sign": sign, "frame": frame, "kind": kind, "possible_deg": possible,
              "rom_deg": rom(chain, axis, table), "duration_grade": duration_grade, "excursion": exc,
              "ramp_s": ramp_s, "foot": foot, "arm": arm, "limits_table": LX.record(table)}
    if lvl is None:
        raise ValueError(f"{region} {direction}: grade {amplitude_grade} unreachable within the joint-limit table "
                         f"({possible:.1f} deg possible)")
    if lvl == 0 or share == 0:
        params.update(level_deg=0.0, share=0.0, intervals_s=[])
        return body.copy(), truth_record("dystonic_posture", seed_val, amplitude_grade, scale, params,
                                         (scale.lower(), "table"), None)
    rng = as_rng(seed)
    reachable = lvl <= possible
    applied = min(lvl, possible)
    peak = applied * (G.SUBMAXIMAL if exc == "submaximal" else 1.0)
    if n_bouts is None:
        n_bouts = 1 if share >= 0.9 else 2
    intervals = schedule(body.duration, share, n_bouts, ramp_s, rng, body.rate)
    env = envelope(body.times().numpy(), intervals, ramp_s, body.duration)
    rm = rooms(body, chain, axis, sign, frame, table=table)
    split = [r / sum(rm) for r in rm] if sum(rm) > 0 else [1.0 / len(chain)] * len(chain)
    R = body.local_R()
    base_comp = components_deg(body.rot6d)
    hang = region == "shoulder_elevation" and arm == "hang"
    trunk_hang = region == "trunk" and arm == "hang"

    def build(scale_k):
        th = np.radians(peak) * sign * env * scale_k
        offs = [torch.from_numpy(th * s) for s in split]
        Rn = _apply(R, chain, axis, offs, frame)
        return hang_from_collar(R, Rn, chain) if hang else Rn
    plant_mode = None
    if region in LEG_REGIONS and foot == "plant":
        plant_mode = "ankle" if chain[-1] in (7, 8) else "compensate"
        on_floor = CT.stance_knots(body, side)
    leg = list(CT.LEG[side][:3])
    rows = sorted(set(chain) | (set(leg) if plant_mode else set())
                  | ({SHOULDER_OF_COLLAR[c] for c in chain if c in SHOULDER_OF_COLLAR} if hang else set()))

    def planted(Rn):
        if plant_mode is None:
            return Rn, np.ones(body.T, bool), None
        Rn = Rn.numpy()
        if plant_mode == "ankle":
            Rp, ok, info = CT.plant_ankle(body, Rn, side, on_floor & (env > 0))
        else:
            Rp, ok, info, _ = CT.plant_compensate(body, Rn, side, on_floor & (env > 0))
        return torch.from_numpy(Rp), ok, info
    def trial(s_k):
        Rn, ok, info = planted(build(s_k))
        comp = components_deg(matrix_to_rot6d(Rn))
        return Rn, (LX.excess(table, comp, base_comp, rows) > 1e-6).any(-1).any(-1) | ~ok, info
    s_k = np.ones(body.T)
    pinfo = None
    plant_scale = 1.0
    if plant_mode:                                        # one scale over the clip: the posture keeps one level
        Rn, bad, pinfo = trial(s_k)
        if bad.any():
            a, b, best = 0.0, 1.0, (R.clone(), None)      # scale 0 is the base itself
            for _ in range(PLANT_BISECT):
                m = 0.5 * (a + b)
                Rm, bm, im = trial(np.full(body.T, m))
                if bm.any():
                    b = m
                else:
                    a, best = m, (Rm, im)
            plant_scale = a
            Rn, pinfo = best
            s_k = np.full(body.T, plant_scale)
    else:
        for _ in range(30):                               # scale down knots that would leave the table
            Rn, bad, _ = trial(s_k)
            if not bad.any():
                break
            s_k = np.where(bad, s_k * 0.95, s_k)
    Rall = R.clone()
    Rall[:, rows] = Rn[:, rows]
    response = None
    if trunk_hang:                                        # the body's answer to the bend (module docstring)
        out, response = trunk_response(body, Rall, direction, axis, sign, np.radians(peak) * sign * env * s_k, table)
    else:
        out = body.with_rotations(Rall, rows)
    feet = (0, 1)
    if region in LEG_REGIONS:                             # the foot moves with the leg: relabel, check the other
        out = replace(out, contact=contact_labels(out))
        feet = (1,) if side == "left" else (0,)
    lift = None
    if plant_mode is not None and seated(body, side):     # the seated hands off the raised thigh (module docstring)
        out, lift = arm_lift(out, env * s_k)
    _, Gb = body.fk()
    _, Go = out.fk()
    dev_seg, comps = deviation(Gb[:, chain[-1]], Go[:, chain[-1]])
    Lb, Lo = body.local_R()[:, chain[-1]], out.local_R()[:, chain[-1]]
    dev_joint = rotation_angle_deg(Lb.transpose(-1, -2) @ Lo)
    dev = dev_joint if plant_mode == "ankle" else dev_seg
    if plant_mode:
        reachable = reachable and plant_scale >= 1.0
    params.update(level_deg=float(lvl), applied_deg=float(applied), peak_deg=float(peak), reachable=bool(reachable),
                  split=[float(x) for x in split], share=realized_share(intervals, body.duration),
                  intervals_s=[list(x) for x in intervals], n_bouts=int(n_bouts),
                  limit_scale_min=float(s_k.min()), limit_scaled_share=float((s_k < 1).mean()),
                  max_deviation_deg=float(dev.max()), max_segment_deviation_deg=float(dev_seg.max()),
                  max_joint_deviation_deg=float(dev_joint.max()), plant=pinfo, plant_scale_min=plant_scale,
                  realized_peak_deg=float(peak * np.max(env * s_k)))
    if region in LEG_REGIONS:                             # what planting costs (module docstring)
        fp = CT.foot_pose_report(body, out, side)
        toe = bool(plant_mode == "compensate" and fp["heel_lift_max_m"] > CT.TOL_M)
        params.update(foot_pose=fp, toe_stance=toe, grading_note=GRADING_NOTE if toe else None)
    if scale == "TWSTRS" and kind in G.TWSTRS_BANDS_DEG:
        params["grade_of_level"] = next((g for g, (lo, hi) in enumerate(G.TWSTRS_BANDS_DEG[kind], 1)
                                         if lo <= round(applied) <= (hi if hi is not None else 1e9)), 0)
    elif scale == "DIS":
        params["grade_of_level"] = G.dis_grade_of(applied / possible) if possible > 0 else 0
        params["share_of_rom"] = float(applied / params["rom_deg"])
        params["dis_duration_grade"] = G.dis_grade_of(params["share"])
    if hang:                                              # the arm under the raised collar (module docstring)
        s_ = SHOULDER_OF_COLLAR[chain[-1]]
        e_ = s_ + 2
        Jb, Jo = body.joints(), out.joints()
        ub, uo = Jb[:, e_] - Jb[:, s_], Jo[:, e_] - Jo[:, s_]
        cos = ((ub * uo).sum(-1) / ub.norm(dim=-1) / uo.norm(dim=-1)).clamp(-1.0, 1.0)
        params.update(humerus_turn_deg_max=float(torch.rad2deg(torch.acos(cos)).max()),
                      shoulder_rise_cm_max=float(100 * (Jo[:, s_, 1] - Jb[:, s_, 1]).max()))
    if response is not None:
        params["response"] = response
    if lift is not None and any(v["deg"] for v in lift["sides"].values()):
        params["arm_lift"] = lift                         # a posture that keeps the hands clear: the truth as before
    if check and response is not None and response["standing"]:
        val = stand_checks(body, out, table)
    else:
        val = LX.check(table, body, out, feet=feet) if check else None
    if check and region in LEG_REGIONS:
        val["contact"] = CT.sole_report(body, out, side, planted=foot == "plant")
        val["continuity"] = CT.step_report(body, out, side)
    if check and response is not None:                    # the hands against the thighs and the trunk
        on = np.flatnonzero(env * s_k > 0)
        if len(on):
            ks = np.unique(np.r_[on[::CLEAR_EVERY], on[-1], int(np.argmax(env * s_k))])
            val.update(_clearance(body, out, ks))
    return out, truth_record("dystonic_posture", seed_val, amplitude_grade, scale, params,
                             (scale.lower(), "table", "module") + (("limits_ext",) if table == LX.DYSTONIA else ()),
                             val, arrays={"envelope": env, "deviation_deg": dev.numpy(), "limit_scale": s_k})
