"""Rest and postural tremor of the forearm (docs/proposals/SYNTH_PATIENT_DRAFT.md, section 3).

Inverse recipe (draft section 3; DR-11.4 and DR-11.5 of docs/rules/DESCRIPTOR_RULES_DRAFT.md): a sinusoid of
the requested frequency, amplitude envelope and on and off schedule added to one joint angle of one limb, per
limb independently (no shared frequency, design section 10).

Driver:
- 'elbow_flexion' (default): a rotation of the elbow (18 left, 19 right) about the elbow hinge, the y axis of
  the upper arm's frame (the parent frame: the hinge is fixed in the humerus), so that the wrist joint moves.
  The amplitude is the peak-to-peak wrist displacement in cm (decision D4 as first proposed: the meaning of the
  MDS-UPDRS 3.15 to 3.17 bands): the chord between the two extremes of the arc, cm / 100 = 2 r sin(theta), with r
  the distance of the wrist joint from the hinge axis in the base posture (the forearm length, 0.2494 m in the
  neutral left arm, when the forearm is perpendicular to the hinge) and theta the peak angle. Since wave SH1d
  (2026-11-03, the session's decision on D4 under the owner's delegation: every hand tremor is levelled at the
  hand, raters scoring the largest amplitude they see) no exam clip uses this driver: it stays, bit for bit, for
  the SP3 probe stimuli (stimuli.py) and the callers that read the wrist.
- 'rest' (wave SH1d, review M8: the rest tremor of version 2, an elbow see-saw under a still hand, read as a
  proximal or mechanical tremor): the parkinsonian rest tremor of the forearm and hand, driven by forearm
  pronation-supination (the axial turn of 'pronation_supination', below) and the hand flexing and extending at the
  wrist (tremors.drives_of 'wrist'), the elbow hinge joining as a small share from the lower edge of the
  MDS-UPDRS grade-3 band (REST_ELBOW_FROM_CM), every drive in phase on one waveform (tremors.oscillate). The angle
  shares (REST_SHARES, module values) give pronation-supination the larger share at every grade: in the inertial
  study of Chan et al. 2022 (SOURCES['chan2022']), on subclinical tremors (no clinical signs), pronation-supination
  is the predominant motion at rest, ahead of elbow flexion-extension, then wrist flexion-extension. The wrist's
  share ahead of the elbow's, the elbow joining only from grade 3, follows the clinical review of the page (wave
  SH1d: the elbow a small share at grades 3 and 4 only), not that source. The amplitude is the peak-to-peak chord
  of the fingertip that moves
  most (FINGERTIPS: the largest windowed chord of the five fingertip vertices of hands.tip_vertices on the skinned
  mesh), calibrated as the 'postural' driver's is (a rigid point, then a correction on the skinned fingertips,
  repeated while the reading is more than REST_CAL_TOL off), so that MDS-UPDRS 3.17 is levelled at the hand as 3.15
  is, at the point a rater sees move most. With the forearm's turn the thumb and index tips, off the forearm's axis,
  move more than the middle fingertip (each exam truth's point_readings records both: the middle fingertip, near
  the axis, takes mostly the wrist's share), so the middle fingertip, the 'postural' driver's
  point and the point the session's decision on D4 names, would leave the thumb and index a band higher at the
  above-floor grade 1; the middle fingertip's reading stays in every exam truth (point_readings). The forearm of
  the seated rest lies just above the chair's armrest, the hand past its end (not on it: the posture 'sit' leaves
  a gap), so the forearm can turn: with clear (the chair's boxes) the centre of the elbow drive would move toward
  flexion until the forearm and hand keep CLEAR_GAP_CM from every box at every phase (ChairRoom, an oscillate room
  like the face room of the reach, on the rule of the elbow driver's scene clearance below: the forearm lifted
  just enough for its turn), and the validity re-measures the skinned forearm and hand at every extremum. On the
  exam body no grade needs the lift (elbow_lift_deg 0 in every E1 truth); a forearm supported on the armrest is
  a matter of the seated posture and the scene, not of this driver.
- 'pronation_supination': an axial rotation of the forearm segment (the elbow joint, 18 or 19) about the
  forearm's long axis (the rest bone from the elbow to the wrist, in the forearm's own frame): the forearm and
  the hand turn, the wrist joint, on the axis, does not move. The amplitude is given in degrees peak to peak;
  no cm value exists for it (the hand carries no finger point).
- 'postural' (shared wave SH4, 2026-11-02: the drive of the tremor family's outstretched hand, tremors.py): the
  hand flexing and extending at the wrist with the shoulder's flexion joining from 3 cm (tremors.drives_of
  'postural', its angle weights tremors.POSTURAL_SHOULDER), so that the outstretched hand moves up and down; the
  amplitude is the peak-to-peak chord of the middle fingertip vertex on the skinned mesh. tremors.oscillate
  calibrates at every knot a rigid point (tremors.point_spec 'middle_tip': that vertex on the hand at rest, rigid
  on the wrist; forward kinematics at both extremes); one more pass rescales its target by the ratio of the skinned
  fingertip's chord, read as readback() reads it, to the level (mesh_calibration in the truth: the skinned
  fingertip lies a few percent farther out than the rigid point, which put a grade-1 level just under 1 cm above
  the band's edge on the mesh). On the arms-forward posture the elbow hinge is vertical, so the elbow driver swept
  the hands sideways; this driver replaces it for the arms-outstretched segment E2 of exam.py.

Envelope: on intervals of common.schedule, raised-cosine ramps of ramp_s (0.5 s by default, a module value: no
published on or off duration was sourced, draft section 11), constant amplitude inside a bout. The share of
the clip with tremor follows MDS-UPDRS 3.18 (grades.constancy_share) or is given, or the on intervals are given
(intervals: the rest tremor of the raise that stops with the raise, exam.py); re-emergent tremor is a
schedule that starts after latency_s (Jankovic et al. 1999: mean 9.37 s). Amplitude modulation and frequency
jitter are not modelled (unsourced, and stimulus variability waits for the neurologist's read, draft section
9). Phase drawn from the seed.

Joint limits (bound): the oscillation stays inside the SP1 joint-limit table. driver_rooms gives the room
the table leaves from the base posture in each direction about the driver's axis (body.room_deg, the smallest
over the knots with tremor). When the peak angle exceeds the room on one side but twice the peak fits the two
rooms together, bound 'shift' (default) moves the oscillation's centre toward the side with room by the
shortfall, ramped with the envelope, so that one extreme touches the table; amplitude and frequency stay
exact (the chord between the extremes of a hinge arc and the band amplitude of the wrist path do not depend
on the centre). The truth records the shift (centre_shift_deg, signed about the driver axis; shift_toward, the
anatomical direction). bound 'refuse' raises instead. When twice the peak exceeds the two rooms together the
amplitude does not fit the posture and the operator raises. Example: the straight elbow of sit_arms_forward
has less room toward extension than the grade-4 peak (reports/synth/stimuli/levels.md). The 'postural' driver
takes the table shifts of tremors.oscillate (the same rule on a 0.5 degree grid, per drive joint).

Scene clearance (clear, elbow_flexion only; shared wave, 2026-11-02): the forearm of the seated rest lies on the
chair's armrest (viewer.chair_mesh: the armrest 1 cm under the proximal third of the forearm), so an elbow tremor
centred on the rest posture drove the forearm into the armrest at its extension extreme (tremor_E1_rest_g4 of
exam.py). With clear (axis-aligned boxes, chair_boxes() for the exam chair) the centre moves toward flexion, in
steps of CLEAR_STEP_DEG from the table's own shift, until the forearm and hand keep CLEAR_GAP_CM from every box at
every angle of the oscillation (or the base's own clearance where the base lies closer); the table bounds the
move, and a tremor that fits neither raises. Amplitude and frequency stay exact, as for the table shift; the
forearm then lifts off the armrest and comes back to it once per cycle. The scan treats the forearm and hand
vertices (dominant joint the elbow, the wrist or a finger joint) as rigid on the forearm about the elbow joint,
skinned once per distinct base pose, at the angles of a CLEAR_STEP_DEG grid inside the oscillation's window and at
the window's two ends exactly (review of the shared wave: a grid alone stopped short of the ends); the truth's
validity re-measures the skinned output at every local extremum of the elbow angle (scene_clearance). Without
clear the operator is the one of version 2, bit for bit.

Reading points (point_readings, review of the shared wave): the wrist joint, where the elbow driver's level is
read, the middle fingertip on the skinned mesh, where the 'postural' driver's level is read, and the fingertip that
moves most of the five (FINGERTIPS), where the 'rest' driver's level is read (the hand, since wave SH1d, for every
exam hand tremor); exam.py records all three in every E1 and E2 truth.

The tremor is added on the 100 Hz clock after the body store's 12 Hz low-pass (plan SP1): the base body is
read as it is. Only the driving joints' rows change; the legs, root and contacts keep their values bit for
bit. A grade of 0 (amplitude 0) returns a copy of the input.
"""
import math

import numpy as np
import torch

from . import grades as G
from ..body.rotations import rot6d_to_matrix
from .body import (ELBOW_FLEXION_SIGN, F64, SUPINATION_SIGN, axis_rotation, bone_axis, compose, rest_joints,
                   room_deg, validity)
from .common import as_rng, envelope, realized_share, schedule, truth_record

ELBOW = {"left": 18, "right": 19}
WRIST = {"left": 20, "right": 21}
SHOULDER = {"left": 16, "right": 17}
RAMP_S = 0.5                     # [module] ramp of the on and off schedule
BOUNDS = ("shift", "refuse")
DRIVERS = ("elbow_flexion", "pronation_supination", "postural", "rest")
SOURCES = {
    "chan2022": "Chan PY, Ripin ZM, Abdul Halim S, Arifin WN, Yahya AS, Eow GB, Tan K, Hor JY, Wong CK 2022, Sci Rep "
                "12: 4021, doi 10.1038/s41598-022-07957-z (full text, PMC8901710): inertial sensors on the upper "
                "limb of 63 patients with Parkinson's disease and 62 controls; at rest and with the arms "
                "outstretched the decreasing order of severity of the patients' subclinical tremors (no clinical "
                "signs) is elbow pronation-supination, elbow flexion-extension, wrist flexion-extension, wrist "
                "abduction-adduction; read for the 'rest' driver's first share (pronation-supination), no number "
                "taken; the wrist's share ahead of the elbow's is not from this source (module docstring)"}
# the 'rest' driver (module docstring): angle shares of its drives, pronation-supination first (chan2022), the wrist
# ahead of the elbow (the clinical review of the page, not chan2022);
# the elbow hinge joins at and above REST_ELBOW_FROM_CM, the lower edge of the MDS-UPDRS grade-3 band
REST_SHARES = {"distal": {"pronation_supination": 0.6, "wrist": 0.4},                     # [module]
               "with_elbow": {"pronation_supination": 0.55, "wrist": 0.3, "elbow": 0.15}}  # [module]
REST_ELBOW_FROM_CM = G.TREMOR_BANDS_CM[2][0]
CHAIR_ROOM_MAX_DEG = 20.0        # [module] the largest chair-room shift searched along the drives, either way
CHAIR_ROOM_PHASES = 9            # [module] waveform values from -1 to 1 checked by the chair room
FINGERTIPS = "fingertips"       # the 'rest' driver's reading point: the fingertip vertex that moves most (docstring)
REST_CAL_TOL = 0.003             # [module] the 'rest' driver's calibration tolerance on the fingertip level (share)
CLEAR_GAP_CM = 0.3               # [module] least clearance of the forearm and hand from a scene box (clear)
CLEAR_STEP_DEG = 0.25            # [module] scan step of the clearance shift and of its angle grid
# the vertices that turn with the forearm about the elbow: dominant joint the elbow, the wrist or a finger joint
ARM_VERTEX_JOINTS = {"left": (18, 20) + tuple(range(22, 37)), "right": (19, 21) + tuple(range(37, 52))}
CLEAR_MAX_POSES = 64             # [module] distinct base poses scanned at most (a moving base is subsampled)


def driver_rooms(body, joint, axis, frame, knots, max_deg):
    """(room_plus, room_minus) deg: the smallest room (body.room_deg) over the selected knots (T,) bool of the
    base for an offset of joint about axis in frame, in the positive and the negative direction, each scanned up
    to max_deg (distinct base rotations computed once)."""
    rows = torch.unique(body.rot6d[torch.as_tensor(knots, dtype=torch.bool), joint], dim=0)
    plus = minus = float(max_deg)
    for r6 in rows:
        R = rot6d_to_matrix(r6)
        plus = min(plus, room_deg(R, joint, axis, 1.0, frame, max_deg=max_deg))
        minus = min(minus, room_deg(R, joint, axis, -1.0, frame, max_deg=max_deg))
    return plus, minus


def centre_shift(peak_deg, room_plus, room_minus, bound="shift"):
    """Signed centre (deg, about the driver axis) of an oscillation of peak_deg inside [-room_minus, room_plus]
    (module docstring, Joint limits): 0 when the peak fits both sides, else the shortfall toward the side with
    room under bound 'shift'; raises ValueError under 'refuse', or when 2 peak_deg exceeds the two rooms."""
    if bound not in BOUNDS:
        raise ValueError(f"bound {bound!r}, expected one of {BOUNDS}")
    if peak_deg <= room_plus and peak_deg <= room_minus:
        return 0.0
    if 2.0 * peak_deg > room_plus + room_minus:
        raise ValueError(f"a tremor of {2 * peak_deg:.2f} deg peak to peak does not fit the "
                         f"{room_plus + room_minus:.2f} deg the joint-limit table leaves this posture about the "
                         f"driver axis")
    if bound == "refuse":
        raise ValueError(f"a tremor peak of {peak_deg:.2f} deg leaves the joint-limit table (room {room_plus:.2f} "
                         f"deg positive, {room_minus:.2f} deg negative); bound='shift' moves its centre")
    return room_plus - peak_deg if peak_deg > room_plus else peak_deg - room_minus


def hinge_radius(body, side, knots=None):
    """(T,) m: distance of the wrist joint from the elbow hinge axis (the y axis of the upper arm frame through
    the elbow) at each knot of the base body."""
    J, Gl = body.fk()
    e, w = ELBOW[side], WRIST[side]
    axis = Gl[:, e - 2] @ torch.tensor([0.0, 1.0, 0.0], dtype=F64)          # parent of the elbow: the shoulder
    d = J[:, w] - J[:, e]
    along = (d * axis).sum(-1, keepdim=True) * axis
    return (d - along).norm(dim=-1)


# ---------------------------------------------------------------- scene clearance (module docstring)
def mesh_boxes(vertices):
    """[(lo (3,), hi (3,))] m: the axis-aligned boxes of a scene mesh made of viewer.box parts (8 corners each, in
    order); ValueError when the mesh is not such a union of boxes."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or len(v) % 8:
        raise ValueError("a scene mesh of axis-aligned boxes has 8 corners per part")
    out = []
    for b in v.reshape(-1, 8, 3):
        lo, hi = b.min(0), b.max(0)
        if not np.all(np.isclose(b, lo, rtol=0.0, atol=1e-9) | np.isclose(b, hi, rtol=0.0, atol=1e-9)):
            raise ValueError("a part of the scene mesh is not an axis-aligned box")
        out.append((lo, hi))
    return out


def chair_boxes(floor_y=None):
    """The boxes of the exam chair (viewer.chair_mesh: seat, backrest, armrests) standing on floor_y (default the
    neutral mesh floor, viewer.mesh_floor_y); their tops do not depend on the floor."""
    from .viewer import chair_mesh
    return mesh_boxes(chair_mesh(floor_y)[0])


def box_depth(P, boxes):
    """(...,) m: the depth of points P (..., 3) inside the union of axis-aligned boxes [(lo, hi)]; negative outside
    (minus the distance to the nearest box)."""
    P = np.asarray(P, dtype=np.float64)
    sdf = np.full(P.shape[:-1], np.inf)
    for lo, hi in boxes:
        c, h = (np.asarray(lo) + np.asarray(hi)) / 2.0, (np.asarray(hi) - np.asarray(lo)) / 2.0
        q = np.abs(P - c) - h
        sdf = np.minimum(sdf, np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0))
    return -sdf


def arm_vertex_ids(side, gender=2):
    """(M,) the mesh vertices that turn with the forearm about the elbow (ARM_VERTEX_JOINTS)."""
    from .viewer import GENDERS, dominant_joint
    return np.flatnonzero(np.isin(dominant_joint(GENDERS[int(gender)]), ARM_VERTEX_JOINTS[side]))


def _rodrigues(axis, angles_rad):
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    a = np.asarray(angles_rad, dtype=np.float64)[:, None, None]
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def _distinct_knots(body, knots):
    """Knot indices of the distinct base poses (rotations and translation) among the selected knots, at most
    CLEAR_MAX_POSES (evenly subsampled beyond)."""
    sel = np.flatnonzero(np.asarray(knots, dtype=bool))
    rows = torch.cat([body.rot6d[sel].reshape(len(sel), -1), body.transl[sel]], 1)
    _, inv = torch.unique(rows, dim=0, return_inverse=True)
    first = np.array(sorted({int(i): k for k, i in reversed(list(enumerate(inv.numpy())))}.values()))
    ks = sel[first]
    if len(ks) > CLEAR_MAX_POSES:
        ks = ks[np.round(np.linspace(0, len(ks) - 1, CLEAR_MAX_POSES)).astype(int)]
    return ks


def clearance_shift(body, side, knots, peak_deg, shift_deg, room_plus, room_minus, boxes, gap_cm=CLEAR_GAP_CM,
                    step_deg=CLEAR_STEP_DEG):
    """(centre deg, info) of an elbow-flexion oscillation of peak_deg (about the hinge y axis of the upper arm,
    signed as tremor()'s angle) moved from shift_deg toward flexion in steps of step_deg until the forearm and hand
    keep gap_cm from the boxes at every angle of the oscillation, or the base's own clearance where the base lies
    closer (module docstring, Scene clearance), inside [-room_minus, room_plus]; ValueError when no centre fits."""
    from . import hands as H
    e = ELBOW[side]
    sgn = ELBOW_FLEXION_SIGN[side]
    ids = arm_vertex_ids(side, body.gender)
    ks = _distinct_knots(body, knots)
    J, Gl = body.fk()
    V = H.body_points(body, ids, knots=ks, chunk=16)                              # (K, M, 3) skinned base
    lo_a, hi_a = -float(room_minus), float(room_plus)
    grid = np.r_[np.arange(lo_a, hi_a, step_deg), hi_a]
    if not np.any(np.isclose(grid, 0.0)):
        grid = np.sort(np.r_[grid, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    poses = [(J[k, e].numpy(), (Gl[k, e - 2] @ y).numpy(), V[i] - J[k, e].numpy()) for i, k in enumerate(ks)]

    def clear_at(angles_deg):                                                    # (A,) m, least over the poses
        out = np.full(len(angles_deg), np.inf)
        for je, axis, rel in poses:
            P = je + np.einsum("aij,mj->ami", _rodrigues(axis, np.radians(angles_deg)), rel)
            out = np.minimum(out, -box_depth(P, boxes).max(-1))
        return out
    clear = clear_at(grid)
    base_clear = float(clear[np.argmin(np.abs(grid))])
    need = min(gap_cm / 100.0, base_clear)
    c = float(shift_deg)
    n_max = int(math.ceil((room_plus + room_minus) / step_deg)) + 1
    for _ in range(n_max):
        a, b = c - peak_deg, c + peak_deg
        if b > room_plus + 1e-9 or a < -room_minus - 1e-9:
            break
        win = (grid > a) & (grid < b)
        got = float(clear_at(np.array([a, b])).min())                            # the window's ends, exactly
        if win.any():
            got = min(got, float(clear[win].min()))
        if got >= need - 1e-9:
            return c, {"gap_cm": float(gap_cm), "boxes": len(boxes), "poses": int(len(ks)),
                       "base_clearance_cm": 100.0 * base_clear, "clearance_cm_needed": 100.0 * need,
                       "clearance_cm_at_extremes": 100.0 * got, "centre_deg": c,
                       "added_shift_deg": c - float(shift_deg), "step_deg": float(step_deg),
                       "rule": "rigid forearm and hand about the elbow joint, skinned base"}
        c += sgn * step_deg
    raise ValueError(f"an elbow tremor of {2 * peak_deg:.2f} deg peak to peak does not keep {gap_cm:g} cm from the "
                     f"scene boxes inside the joint-limit table (room {room_plus:.2f} deg positive, "
                     f"{room_minus:.2f} deg negative)")


def clearance_check(base, out, side, boxes, angle, env, what="the elbow angle"):
    """{min_clearance_cm, max_depth_cm, base_min_clearance_cm, knots_checked}: the skinned forearm and hand of out
    against the boxes at every local extremum of a drive angle (rad, (T,); what names it) where env > 0, and of the
    base at the same knots."""
    from . import hands as H
    a = np.asarray(angle, dtype=np.float64)
    on = np.asarray(env) > 0
    ext = np.flatnonzero(on[1:-1] & (((a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]))
                                     | ((a[1:-1] <= a[:-2]) & (a[1:-1] <= a[2:])))) + 1
    if not len(ext):
        ext = np.flatnonzero(on)[:1]
    ids = arm_vertex_ids(side, out.gender)
    d_out = np.concatenate([box_depth(H.body_points(out, ids, knots=ext[i:i + 64], chunk=16), boxes).max(-1)
                            for i in range(0, len(ext), 64)])
    d_base = box_depth(H.body_points(base, ids, knots=_distinct_knots(base, on), chunk=16), boxes).max(-1)
    return {"min_clearance_cm": float(-100.0 * d_out.max()), "max_depth_cm": float(max(0.0, 100.0 * d_out.max())),
            "base_min_clearance_cm": float(-100.0 * d_base.max()), "knots_checked": int(len(ext)),
            "where": f"skinned forearm and hand at every local extremum of {what}"}


def postural_text(cm_pp):
    """The drive of the 'postural' driver for a chord of cm_pp, in clinical words (caption text; the shoulder's angle
    weight, tremors.postural_shoulder_weight, is in the truth)."""
    from .tremors import postural_shoulder_weight
    sh = postural_shoulder_weight(cm_pp)
    t = "the hand flexing and extending at the wrist"
    return t if not sh else f"{t}, the shoulder joining in"


def rest_shares(cm_pp):
    """{drive: angle share} of the 'rest' driver for a chord of cm_pp (REST_SHARES, REST_ELBOW_FROM_CM)."""
    return dict(REST_SHARES["with_elbow" if cm_pp is not None and cm_pp >= REST_ELBOW_FROM_CM else "distal"])


def rest_text(cm_pp):
    """The drive of the 'rest' driver for a chord of cm_pp, in clinical words (caption text)."""
    t = "forearm pronation-supination and wrist flexion-extension"
    # ', the elbow too' (', the elbow joining in' before the open-body port's wave 2: on the open body the level tip
    # of grade 4 is the index fingertip, not the thumb tip, and the longer caption left the viewer's band)
    return t + (", the elbow too" if "elbow" in rest_shares(cm_pp) else "")


def rest_drives(side, cm_pp, body=None):
    """[tremors.Drive] of the 'rest' driver (module docstring): forearm pronation-supination (the elbow joint about
    the forearm's rest bone in its own frame, positive toward supination, body.SUPINATION_SIGN), the hand's
    flexion-extension at the wrist (tremors.drives_of 'wrist', positive lifting the fingers) and the elbow hinge
    (about the upper arm's y axis, its weight positive toward flexion, body.ELBOW_FLEXION_SIGN), weighted by
    rest_shares (the elbow's weight 0 below REST_ELBOW_FROM_CM: it then carries only the chair room's lift), all on
    one waveform. body: its shape for the forearm axis (default the neutral one)."""
    from .tremors import Drive, drives_of
    sh = rest_shares(cm_pp)
    ax = bone_axis(ELBOW[side], WRIST[side], None if body is None else body.betas, 2 if body is None else body.gender)
    wr = drives_of("wrist", side)[0]
    return [Drive(ELBOW[side], tuple(float(x) for x in ax), "child",
                  SUPINATION_SIGN[side] * sh["pronation_supination"]),
            Drive(wr.joint, wr.axis, wr.frame, wr.weight * sh["wrist"]),
            Drive(ELBOW[side], "y", "parent", ELBOW_FLEXION_SIGN[side] * sh.get("elbow", 0.0))]


class ChairRoom:
    """The chair as a room of the 'rest' driver (module docstring; tremors.oscillate's room callable), the rule of
    the elbow driver's scene clearance (clearance_shift) on the rest driver's drives: at every distinct base pose
    among the knots with tremor (at most CLEAR_MAX_POSES), the least lift s (deg of elbow flexion added to the
    centre of the elbow drive, the last of rest_drives; a scan of CLEAR_STEP_DEG up to CHAIR_ROOM_MAX_DEG, then a
    bisection inside the step) after which the forearm and hand vertices (ARM_VERTEX_JOINTS, skinned on the base and
    rigid on the elbow, or on the wrist for the hand and fingers) keep gap_cm from the boxes at CHAIR_ROOM_PHASES
    values of the waveform from -1 to 1 at full peak, or the base's own clearance where the base lies closer, so
    that the forearm lifts off the armrest just enough for its turn. The lift follows the envelope (it scales with
    the peak at each knot) and is covered as the joint-limit shifts are. A body that needs no lift gets none, bit
    for bit. ValueError when no lift fits."""

    def __init__(self, body, side, boxes, gap_cm=CLEAR_GAP_CM):
        from . import hands as H
        from .viewer import GENDERS, dominant_joint
        self.side, self.boxes, self.gap = side, list(boxes), gap_cm / 100.0
        self.transl, self.rest, self.body, self.H = body.transl, body.rest(), body, H
        self.ids = arm_vertex_ids(side, body.gender)
        self.dom = dominant_joint(GENDERS[int(body.gender)])[self.ids]
        self.info = {"lift_deg": 0.0, "poses": 0, "base_clearance_cm": None, "gap_cm": float(gap_cm),
                     "boxes": len(self.boxes), "rule": "elbow flexion added to the centre until the forearm and hand "
                                                       "keep the gap at every phase (ChairRoom)"}

    def _locals(self, knots):
        """{joint: (K, M_j, 3)} the arm vertices of the base at knots in the frame of the joint they are rigid on:
        the elbow for the forearm's, the wrist for the hand's (its finger joints held at the base's pose)."""
        J, Gl = self.body.fk()
        V = self.H.body_points(self.body, self.ids, knots=knots, chunk=16)            # (K, M, 3)
        e, w = ELBOW[self.side], WRIST[self.side]
        fore = self.dom == e
        out = {}
        for j, sel in ((e, fore), (w, ~fore)):
            Jk, Gk = J[knots, j].numpy(), Gl[knots, j].numpy()
            out[j] = np.einsum("kba,kmb->kma", Gk, V[:, sel] - Jk[:, None])
        return out

    def clearance(self, R, transl, loc):
        """(N,) m: the least distance of the arm points (loc, rigid on their joints) to the boxes (negative inside)
        for local rotations R (N, 22, 3, 3) and translations (N, 3)."""
        from ..body.joints import fk as _fk
        J, Gl = _fk(torch.as_tensor(R, dtype=F64), torch.as_tensor(transl, dtype=F64), self.rest)
        pts = [J[:, j, None, :].numpy() + np.einsum("nab,nmb->nma", Gl[:, j].numpy(), L) for j, L in loc.items()]
        return -box_depth(np.concatenate(pts, 1), self.boxes).max(-1)

    def __call__(self, R0, drives, shifts, peak, rate, window_s):
        from .tremors import _apply, _cover
        T = len(peak)
        zero = [np.zeros(T) for _ in drives]
        on = peak > 0
        if not on.any():
            return zero
        lift = len(drives) - 1
        if drives[lift].joint != ELBOW[self.side] or drives[lift].axis != "y" or drives[lift].frame != "parent":
            raise ValueError("ChairRoom lifts the elbow drive, the last of rest_drives")
        sgn = ELBOW_FLEXION_SIGN[self.side]
        ks = _distinct_knots(self.body, on)
        loc_all = self._locals(ks)
        Y = np.linspace(-1.0, 1.0, CHAIR_ROOM_PHASES)
        req = np.zeros(T)
        best, base_min = 0.0, np.inf
        sel_on = np.flatnonzero(on)
        owner = ks[np.argmin(np.abs(sel_on[:, None] - ks[None]), axis=1)]     # each on-knot's nearest distinct pose
        for i, k in enumerate(ks):
            loc = {j: L[i:i + 1].repeat(len(Y), 0) for j, L in loc_all.items()}
            mine = sel_on[owner == k]
            pk = float(peak[mine].max())
            base = float(self.clearance(R0[[k]], self.transl[[k]], {j: L[i:i + 1] for j, L in loc_all.items()})[0])
            base_min = min(base_min, base)
            need = min(self.gap, base) - 1e-9

            def margin(s):
                ang = [np.radians(shifts[n][k] + d.weight * pk * Y + (sgn * s if n == lift else 0.0))
                       for n, d in enumerate(drives)]
                Rk = R0[[k]].expand(len(Y), 22, 3, 3)
                return float(self.clearance(_apply(Rk, drives, ang), self.transl[[k]].expand(len(Y), 3), loc).min()
                             - need)
            if margin(0.0) >= 0:
                continue
            s = None
            for sv in np.arange(CLEAR_STEP_DEG, CHAIR_ROOM_MAX_DEG + 1e-9, CLEAR_STEP_DEG):
                if margin(sv) >= 0:
                    lo_, hi_ = sv - CLEAR_STEP_DEG, sv
                    for _ in range(8):
                        mid = 0.5 * (lo_ + hi_)
                        lo_, hi_ = (lo_, mid) if margin(mid) >= 0 else (mid, hi_)
                    s = hi_
                    break
            if s is None:
                raise ValueError(f"the {self.side} forearm and hand cannot keep {100 * self.gap:g} cm from the chair "
                                 f"with {CHAIR_ROOM_MAX_DEG:g} deg of elbow flexion added to the rest tremor's centre")
            req[mine] = s * peak[mine] / pk
            best = max(best, s)
        self.info.update(lift_deg=float(best), poses=int(len(ks)), base_clearance_cm=float(100 * base_min))
        if not req.any():
            return zero
        n = max(1, int(round(window_s * rate)))
        out = [np.zeros(T) for _ in drives]
        out[lift] = sgn * _cover(req, n)
        return out


def tremor(body, side="left", grade=None, cm_pp=None, deg_pp=None, frequency_hz=G.TREMOR_DEFAULT_HZ,
           driver="elbow_flexion", condition="rest", tremor_type="parkinsonian_rest", constancy_grade=None,
           share=None, n_bouts=None, latency_s=0.0, ramp_s=RAMP_S, grade1="above_floor", bound="shift", seed=0,
           check=True, clear=None, intervals=None):
    """(Body, truth) of a forearm tremor on one limb (module docstring).

    Amplitude: grade (0 to 4, by grades.tremor_level_cm; grade1 picks the grade-1 level), or cm_pp
    (elbow_flexion: at the wrist; postural: at the middle fingertip; rest: at the fingertip that moves most), or
    deg_pp (elbow_flexion or pronation_supination); grade and cm_pp together give a level of the caller's inside
    the grade's band (the open top band of tremors.TOP_LEVEL_CM). Frequency within the 3 to 12 Hz tremor band of
    DR-11.3, flagged when outside the published band of tremor_type. Schedule: constancy_grade (3.18) or share of the
    clip from latency_s on (default 1, the whole clip), in n_bouts bouts (default 1 at a share of 0.9 or more, else 2),
    or intervals, the on intervals [(a, b)] s given (no share, constancy grade or latency then). condition ('rest',
    'postural', 'kinetic') is recorded, not inferred. bound: 'shift' or 'refuse' (module docstring, Joint limits).
    clear: None or [(lo, hi)] boxes the forearm and hand keep clear of (elbow_flexion: Scene clearance; rest:
    ChairRoom). seed: int or numpy Generator (phase and schedule)."""
    if side not in ELBOW:
        raise ValueError(f"side {side!r}")
    if not 3.0 <= frequency_hz <= 12.0:
        raise ValueError(f"tremor frequency {frequency_hz} Hz outside the 3 to 12 Hz tremor band (DR-11.3)")
    if driver not in DRIVERS:
        raise ValueError(f"driver {driver!r}")
    given = [x is not None for x in (grade, cm_pp, deg_pp)]
    if sum(given) != 1 and not (grade is not None and cm_pp is not None and deg_pp is None):
        raise ValueError("give exactly one of grade, cm_pp and deg_pp (or grade with its cm_pp level)")
    if bound not in BOUNDS:
        raise ValueError(f"bound {bound!r}, expected one of {BOUNDS}")
    if grade is not None and cm_pp is not None:
        if G.tremor_grade_of_cm(float(cm_pp)) != int(grade):
            raise ValueError(f"{cm_pp} cm peak to peak lies outside the band of grade {grade}")
    elif grade is not None:
        cm_pp = G.tremor_level_cm(grade, grade1)
    if cm_pp is not None and driver == "pronation_supination":
        raise ValueError("a cm amplitude needs the elbow_flexion, the postural or the rest driver (the wrist joint "
                         "does not move otherwise)")
    if driver in ("postural", "rest") and deg_pp is not None:
        point = "the middle fingertip" if driver == "postural" else "the fingertip that moves most"
        raise ValueError(f"the {driver} driver takes a cm amplitude at {point} (grade or cm_pp)")
    if clear is not None and driver not in ("elbow_flexion", "rest"):
        raise ValueError("clear applies to the elbow_flexion and the rest drivers")
    if intervals is not None:
        if share is not None or constancy_grade is not None or latency_s:
            raise ValueError("give intervals, or a share, a constancy grade and a latency, not both")
        intervals = [(float(a), float(b)) for a, b in intervals]
        share = realized_share(intervals, body.duration)
    if constancy_grade is not None:
        if share is not None:
            raise ValueError("give constancy_grade or share, not both")
        share = G.constancy_share(constancy_grade)
    share = 1.0 if share is None else float(share)
    amp_zero = (cm_pp is not None and cm_pp == 0) or (deg_pp is not None and deg_pp == 0) or share == 0
    rng = as_rng(seed)
    seed_val = seed if isinstance(seed, (int, np.integer)) else None
    params = {"side": side, "driver": driver, "condition": condition, "tremor_type": tremor_type,
              "frequency_hz": float(frequency_hz), "in_published_band": bool(
                  G.TREMOR_TYPES[tremor_type][0] <= frequency_hz <= G.TREMOR_TYPES[tremor_type][1]),
              "ramp_s": float(ramp_s), "latency_s": float(latency_s), "bound": bound}
    if driver in ("postural", "rest"):
        params.update(point="middle_tip" if driver == "postural" else FINGERTIPS)
    band_src = ("bhatia2018",) if tremor_type == "parkinsonian_rest" else ()     # the published band (grades)
    if amp_zero:
        params.update(cm_pp=0.0 if driver != "pronation_supination" else None, deg_pp=0.0, share=0.0,
                      intervals_s=[])
        return body.copy(), truth_record("tremor", seed_val, grade, "MDS-UPDRS 3.15-3.18", params,
                                         ("mds_updrs", "elble2006", "charles1999") + band_src, None)
    t = body.times().numpy()
    if n_bouts is None:
        n_bouts = 1 if share >= 0.9 else 2
    phase = float(rng.uniform(0.0, 2 * math.pi))
    if intervals is None:
        intervals = schedule(body.duration, share, n_bouts, ramp_s, rng, body.rate, start_s=latency_s)
    else:
        n_bouts = len(intervals)
    env = envelope(t, intervals, ramp_s, body.duration)
    if driver == "postural":
        return _postural(body, side, grade, cm_pp, frequency_hz, phase, t, env, intervals, n_bouts, latency_s,
                         bound, params, seed_val, check, band_src)
    if driver == "rest":
        return _rest(body, side, grade, cm_pp, frequency_hz, phase, t, env, intervals, n_bouts, latency_s, bound,
                     params, seed_val, check, band_src, clear)
    R = body.local_R()
    j = ELBOW[side]
    if driver == "elbow_flexion":
        r = hinge_radius(body, side)
        r_ref = float(r.median())
        if cm_pp is not None:
            theta = math.radians(G.elbow_angle_deg(cm_pp, r_ref))
        else:
            theta = math.radians(deg_pp / 2.0)
        axis, frame, toward = "y", "parent", ("flexion", "extension", ELBOW_FLEXION_SIGN[side])
    else:
        axis = bone_axis(ELBOW[side], WRIST[side], body.betas, body.gender)
        theta = math.radians(deg_pp / 2.0)
        frame, toward = "child", ("supination", "pronation", SUPINATION_SIGN[side])
    peak = math.degrees(theta)
    max_room = min(180.0, 2.0 * peak + 1.0)
    if clear is not None:                          # room for the clearance shift toward flexion
        max_room = min(180.0, 2.0 * peak + 30.0)
    room_plus, room_minus = driver_rooms(body, j, axis, frame, env > 0, max_room)
    shift = centre_shift(peak, room_plus, room_minus, bound)
    cinfo = None
    if clear is not None:
        c, cinfo = clearance_shift(body, side, env > 0, peak, shift, room_plus, room_minus, list(clear))
        if bound == "refuse" and c != shift:
            raise ValueError(f"the elbow tremor needs a centre {abs(c - shift):.2f} deg toward flexion to keep clear "
                             f"of the scene boxes; bound='shift' moves it")
        shift = c
    ang = theta * env * np.sin(2 * math.pi * frequency_hz * t + phase)
    if shift:                                      # the centre ramps with the envelope (module docstring)
        ang = ang + math.radians(shift) * env
    R_new = compose(R[:, j], axis_rotation(axis, torch.from_numpy(ang)), frame)
    params.update(room_plus_deg=room_plus, room_minus_deg=room_minus, centre_shift_deg=float(shift),
                  shift_toward=None if not shift else toward[0] if shift * toward[2] > 0 else toward[1])
    if cinfo is not None:
        params.update(scene_clearance=cinfo)
    if driver == "elbow_flexion":
        chord_cm = 100.0 * 2.0 * r.numpy() * math.sin(theta)
        params.update(cm_pp=float(100.0 * 2.0 * r_ref * math.sin(theta)), deg_pp=float(2 * math.degrees(theta)),
                      peak_deg=float(math.degrees(theta)), radius_m=r_ref,
                      radius_range_m=[float(r.min()), float(r.max())],
                      chord_cm_range=[float(chord_cm.min()), float(chord_cm.max())],
                      forearm_m=float((rest_joints(body.betas, body.gender)[WRIST[side]]
                                       - rest_joints(body.betas, body.gender)[j]).norm()),
                      grade_of_cm=G.tremor_grade_of_cm(100.0 * 2.0 * r_ref * math.sin(theta)))
    else:
        params.update(cm_pp=None, deg_pp=float(deg_pp), peak_deg=float(math.degrees(theta)))
    Rall = R.clone()
    Rall[:, j] = R_new
    out = body.with_rotations(Rall, [j])
    params.update(joint=int(j), phase_rad=phase, share=realized_share(intervals, body.duration, latency_s)
                  if latency_s < body.duration else 0.0, intervals_s=[list(x) for x in intervals],
                  constancy_grade=G.constancy_grade_of(realized_share(intervals, body.duration, latency_s)),
                  n_bouts=int(n_bouts))
    val = validity(body, out) if check else None
    if val is not None and clear is not None:
        val["scene_clearance"] = clearance_check(body, out, side, list(clear), ang, env)
    return out, truth_record("tremor", seed_val, grade, "MDS-UPDRS 3.15-3.18", params,
                             ("mds_updrs", "elble2006", "charles1999", "jankovic1999", "video_floor", "forearm")
                             + band_src + (("module",) if clear is not None else ()),
                             val, arrays={"angle_rad": ang, "envelope": env})


def _postural(body, side, grade, cm_pp, frequency_hz, phase, t, env, intervals, n_bouts, latency_s, bound, params,
              seed_val, check, band_src=()):
    """The 'postural' driver of tremor() (module docstring): tremors.oscillate with the drives of
    tremors.drives_of('postural') and the middle fingertip as the measured point."""
    from . import tremors as TR
    drives = TR.drives_of("postural", side, cm=cm_pp)
    point = TR.point_spec("middle_tip", side)
    y = np.sin(2 * math.pi * frequency_hz * t + phase)
    out, info, arr = TR.oscillate(body, drives, cm_pp, y, env, point, frequency_hz)
    first = _skinned_chord(out, body, side, env)                     # mesh calibration (module docstring)
    rigid = float(cm_pp) * float(cm_pp) / first
    out, info, arr = TR.oscillate(body, drives, rigid, y, env, point, frequency_hz)
    cal = {"rigid_target_cm": rigid, "first_pass_skinned_cm": first,
           "skinned_cm": _skinned_chord(out, body, side, env),
           "rule": "median windowed chord of the skinned middle fingertip against the base (tremor.readback)"}
    shifts = info["centre_shift_deg_max"]
    if bound == "refuse" and any(s > 0 for s in shifts):
        raise ValueError(f"the postural tremor leaves the joint-limit table (centre shifts {shifts} deg); "
                         f"bound='shift' moves its centre")
    th = info["peak_deg_range"][1]
    w = {d["joint"]: abs(d["weight"]) for d in info["drives"]}
    chord = info["chord_cm_at_full_envelope"]
    params.update(cm_pp=float(cm_pp), read_at="middle fingertip vertex, skinned mesh", mesh_calibration=cal,
                  drives=info["drives"], shoulder_weight=float(TR.postural_shoulder_weight(cm_pp)),
                  peak_deg=float(th), deg_pp=float(2 * th * w[WRIST[side]]),
                  wrist_deg_pp=float(2 * th * w[WRIST[side]]),
                  shoulder_deg_pp=float(2 * th * w.get(SHOULDER[side], 0.0)),
                  centre_shift_deg=float(max(shifts)), centre_shift_deg_per_drive=[float(s) for s in shifts],
                  shift_toward=None, chord_cm_range=chord,
                  grade_of_cm=G.tremor_grade_of_cm(float(cm_pp)), joint=int(WRIST[side]),
                  joints=sorted({int(d["joint"]) for d in info["drives"]}), phase_rad=phase,
                  share=realized_share(intervals, body.duration, latency_s) if latency_s < body.duration else 0.0,
                  intervals_s=[list(x) for x in intervals],
                  constancy_grade=G.constancy_grade_of(realized_share(intervals, body.duration, latency_s)),
                  n_bouts=int(n_bouts))
    val = validity(body, out) if check else None
    return out, truth_record("tremor", seed_val, grade, "MDS-UPDRS 3.15-3.18", params,
                             ("mds_updrs", "elble2006", "charles1999", "jankovic1999", "video_floor", "module")
                             + tuple(band_src), val,
                             arrays={"angle_rad": arr["angle_rad"], "centre_rad": arr["centre_rad"], "envelope": env})


def _rest(body, side, grade, cm_pp, frequency_hz, phase, t, env, intervals, n_bouts, latency_s, bound, params,
          seed_val, check, band_src=(), clear=None):
    """The 'rest' driver of tremor() (module docstring): tremors.oscillate with rest_drives and the middle fingertip
    (rigid on the wrist) as oscillate's measured point, the level then calibrated on the largest skinned fingertip
    chord (FINGERTIPS, _fingertips_chord: a rigid target scaled by the asked level over the first pass's reading, a
    second correction while the reading is more than REST_CAL_TOL off); with clear, the chair as a room (ChairRoom)
    and the skinned forearm and hand re-measured at every extremum (clearance_check)."""
    from . import tremors as TR
    drives = rest_drives(side, cm_pp, body)
    point = TR.point_spec("middle_tip", side)
    y = np.sin(2 * math.pi * frequency_hz * t + phase)
    room = ChairRoom(body, side, clear) if clear is not None else None
    rigid, passes = float(cm_pp), []
    for _ in range(3):                                               # mesh calibration (module docstring)
        out, info, arr = TR.oscillate(body, drives, rigid, y, env, point, frequency_hz, room=room)
        got, tip = _fingertips_chord(out, body, side, env)
        passes.append({"rigid_target_cm": rigid, "skinned_cm": got, "tip": tip})
        if abs(got / float(cm_pp) - 1.0) <= REST_CAL_TOL:
            break
        rigid *= float(cm_pp) / got
    cal = {"rigid_target_cm": passes[-1]["rigid_target_cm"], "first_pass_skinned_cm": passes[0]["skinned_cm"],
           "skinned_cm": passes[-1]["skinned_cm"], "tip": passes[-1]["tip"], "passes": passes,
           "rule": "median windowed chord of the skinned fingertip that moves most against the base (tremor.readback)"}
    if abs(cal["skinned_cm"] / float(cm_pp) - 1.0) > REST_CAL_TOL:
        raise ValueError(f"the rest tremor's fingertip level {cal['skinned_cm']:.3f} cm does not reach "
                         f"{float(cm_pp):.3f} cm within {REST_CAL_TOL:g} in {len(passes)} passes")
    shifts = info["centre_shift_deg_max"]
    if bound == "refuse" and any(s > 0 for s in shifts):
        raise ValueError(f"the rest tremor leaves the joint-limit table or meets the chair (centre shifts {shifts} "
                         f"deg); bound='shift' moves its centre")
    th = info["peak_deg_range"][1]
    sh = rest_shares(cm_pp)
    names = ["pronation_supination", "wrist"] + (["elbow"] if "elbow" in sh else [])
    pp = {f"{n}_deg_pp": float(2 * th * sh[n]) for n in names}
    params.update(cm_pp=float(cm_pp), read_at="the fingertip vertex that moves most (of the five), skinned mesh",
                  mesh_calibration=cal, level_tip=cal["tip"], drives=info["drives"], drive_names=names,
                  angle_shares=sh, peak_deg=float(th),
                  deg_pp=float(2 * th), **pp, centre_shift_deg=float(max(shifts)),
                  centre_shift_deg_per_drive=[float(s) for s in shifts],
                  room_shift_deg_per_drive=[float(s) for s in info["room_shift_deg_max"]], shift_toward=None,
                  chord_cm_range=info["chord_cm_at_full_envelope"],
                  grade_of_cm=G.tremor_grade_of_cm(float(cm_pp)), joint=int(ELBOW[side]),
                  joints=sorted({int(d["joint"]) for d in info["drives"]}), phase_rad=phase,
                  share=realized_share(intervals, body.duration, latency_s) if latency_s < body.duration else 0.0,
                  intervals_s=[list(x) for x in intervals],
                  constancy_grade=G.constancy_grade_of(realized_share(intervals, body.duration, latency_s)),
                  n_bouts=int(n_bouts))
    if room is not None:
        params.update(scene_clearance=dict(room.info, added_shift_deg=float(room.info["lift_deg"])),
                      elbow_lift_deg=float(room.info["lift_deg"]),
                      shift_toward="flexion" if room.info["lift_deg"] else None)
    val = validity(body, out) if check else None
    if val is not None and clear is not None:
        val["scene_clearance"] = clearance_check(body, out, side, list(clear), arr["angle_rad"][1], env,
                                                 "the wrist's angle (every drive on one waveform)")
    return out, truth_record("tremor", seed_val, grade, "MDS-UPDRS 3.15-3.18", params,
                             ("mds_updrs", "elble2006", "charles1999", "jankovic1999", "video_floor", "chan2022",
                              "module") + tuple(band_src), val,
                             arrays={"angle_rad": arr["angle_rad"], "centre_rad": arr["centre_rad"], "envelope": env})


def _skinned_chord(out, base, side, env):
    """cm: the median windowed peak-to-peak chord of the skinned middle fingertip of out against base over the knots
    at full envelope (above one half when none is), as readback() reads it."""
    from . import tremors as TR
    sel = env >= 1.0 - 1e-9
    if sel.sum() < 2:
        sel = env > 0.5
    tip, tip0 = TR.point_path(out, "middle_tip", side), TR.point_path(base, "middle_tip", side)
    return TR.oscillation_readback(tip, tip0, out.rate, mask=sel, fmask=env > 0)["pp_cm_median"]


def fingertip_readback(body, base, side, mask, fmask=None):
    """{'frequency_hz', 'pp_cm_max', 'pp_cm_median', 'tip', 'per_tip', 'path', 'base_path'} of the five fingertip
    vertices of a side (hands.tip_vertices, skinned mesh) of body against base (module docstring, the 'rest'
    driver's reading point FINGERTIPS): each tip by tremors.oscillation_readback over mask (spectrum over fmask), the
    reading that of the tip with the largest windowed chord (tip names it; per_tip gives every tip's largest chord,
    cm)."""
    from . import hands as H
    from . import tremors as TR
    tips = H.tip_vertices()[side]
    ids = list(tips.values())
    P, P0 = H.body_points(body, ids), H.body_points(base, ids)
    per = {f: TR.oscillation_readback(P[:, i], P0[:, i], body.rate, mask=mask, fmask=fmask)
           for i, f in enumerate(tips)}
    top = max(per, key=lambda f: per[f]["pp_cm_max"])
    i = list(tips).index(top)
    return dict(per[top], tip=top, per_tip={f: float(r["pp_cm_max"]) for f, r in per.items()}, path=P[:, i],
                base_path=P0[:, i])


def _fingertips_chord(out, base, side, env):
    """(cm, tip): the largest over the five fingertips of the median windowed chord against base over the knots at
    full envelope (above one half when none is), the calibration reading of the 'rest' driver (_skinned_chord's rule
    at every fingertip)."""
    from . import hands as H
    from . import tremors as TR
    sel = env >= 1.0 - 1e-9
    if sel.sum() < 2:
        sel = env > 0.5
    tips = H.tip_vertices()[side]
    P, P0 = H.body_points(out, list(tips.values())), H.body_points(base, list(tips.values()))
    med = {f: TR.oscillation_readback(P[:, i], P0[:, i], out.rate, mask=sel, fmask=env > 0)["pp_cm_median"]
           for i, f in enumerate(tips)}
    top = max(med, key=med.get)
    return float(med[top]), top


def readback(body, truth, base):
    """{'frequency_hz', 'pp_cm_max', 'pp_cm_median'} of a tremor() clip, re-measured on the body: the measured point
    (the middle fingertip on the skinned mesh for the 'postural' driver, the fingertip that moves most of the five
    for the 'rest' driver, plus 'tip' and 'per_tip' then, else the wrist joint) against the same point of base (the
    clip without its tremor, its level 0), by tremors.oscillation_readback over the knots where the envelope is above
    one half; plus 'axis' (the unit first principal axis of the displacement, x lateral, y up, z forward in the body
    frame)."""
    from . import tremors as TR
    p = truth["params"]
    name = p.get("point", "wrist")
    env = np.asarray(truth["arrays"]["envelope"])
    if name == FINGERTIPS:
        out = fingertip_readback(body, base, p["side"], env > 0.5, env > 0)
        path, bpath = out.pop("path"), out.pop("base_path")
    else:
        path = TR.point_path(body, name, p["side"])
        bpath = TR.point_path(base, name, p["side"])
        out = TR.oscillation_readback(path, bpath, body.rate, mask=env > 0.5, fmask=env > 0)
    D = (path - bpath)[env > 0.5]
    D = D - D.mean(0)
    ax = np.linalg.svd(D, full_matrices=False)[2][0]
    out["axis"] = [float(x) for x in ax * np.sign(ax[np.argmax(np.abs(ax))])]
    return out


READING_POINTS = ("wrist", "middle_tip", FINGERTIPS)


def point_readings(body, base, side, env):
    """{'<point>_pp_cm': max, 'readings': {point: {pp_cm_max, pp_cm_median, grade_of_max}, 'rule'}} of a tremor
    clip at its reading points: the wrist joint (the elbow driver's level), the middle fingertip on the skinned mesh
    (the 'postural' driver's level) and the fingertip that moves most of the five (FINGERTIPS, the 'rest' driver's
    level, with 'tip' and 'per_tip'), each against the same point of base (the clip without its tremor) over the
    knots where the envelope env is above one half (tremors.oscillation_readback, as readback() reads);
    grade_of_max by grades.tremor_grade_of_cm."""
    from . import tremors as TR
    env = np.asarray(env)
    out, rd = {}, {"rule": "peak-to-peak chord against the base, tremors.oscillation_readback over the knots where "
                           "the envelope is above one half (tremor.point_readings)"}
    for name in READING_POINTS:
        if name == FINGERTIPS:
            rb = fingertip_readback(body, base, side, env > 0.5, env > 0)
        else:
            rb = TR.oscillation_readback(TR.point_path(body, name, side), TR.point_path(base, name, side),
                                         body.rate, mask=env > 0.5, fmask=env > 0)
        rd[name] = {"pp_cm_max": rb["pp_cm_max"], "pp_cm_median": rb["pp_cm_median"],
                    "grade_of_max": G.tremor_grade_of_cm(rb["pp_cm_max"])}
        if name == FINGERTIPS:
            rd[name].update(tip=rb["tip"], per_tip=rb["per_tip"])
        out[f"{name}_pp_cm"] = rb["pp_cm_max"]
    out["readings"] = rd
    return out
