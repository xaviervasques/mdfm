"""The hands of the synthetic patient (task SP5): the 30 hand joints of the program's body, their poses and the
skinning of the mesh with them, so that the exam items that need fingers (MDS-UPDRS 3.4 finger tapping, 3.5 hand movements,
the fingers apart of 3.15) have a body (docs/proposals/SYNTH_PATIENT_DRAFT.md, sections 1 and 2: version 1 held
the hands at rest).

Joints. The program carries 15 joints per hand under the wrists (20 left, 21 right), three per finger in the order
index, middle, pinky, ring, thumb (joints 22 to 36 on the left, 37 to 51 on the right; openbody.PARENTS52, the same
tree on both body models, and the tests check the order on the rest geometry). For a finger the three joints are the knuckle (MCP), the middle joint
(PIP) and the distal joint (DIP); for the thumb, the base (CMC), the MCP and the IP joint.

Body. HandBody is a synth.body.Body with hand6d (T, 30, 6): the rot6d of the local rotations of the 30 hand
joints (rows 0 to 14 the left hand, joints 22 to 36; rows 15 to 29 the right, 37 to 51) on the body's clock. Its
copy() keeps the hands, so the operators of this package, which edit the 22 body joints through copy() and
with_rotations, keep the hand rows bit for bit. hands_of(body) gives a HandBody with every hand joint at rest
for a plain Body.

Skinning (skin_ids, body_points; plan phase C wave 1): the selected body model's one skinning call
(body.body_model().skin: the open body by default, the SMPL+H reference behind its flag) on chosen vertices; skin
is the SMPL+H arithmetic itself (smplh_reference.skin_smplh), kept for the reference behind its flag. The model arrays
stay in memory.

Anatomical axes (hand_frame, from the rest mesh of the selected body model, never typed):
- the palmar normal n of a hand: the unit normal of the plane through the wrist and the index and pinky
  knuckles, on the palm side (down in the rest pose, where the palms face the floor);
- the flexion axis of a finger joint: a = b x n, with b the rest direction of the segment the joint turns (to
  the next joint, or to the fingertip vertex for the distal joint), so that a positive rotation turns the
  segment toward the palm; the abduction axis of a knuckle: b x l, l the in-plane direction away from the
  middle finger, so that a positive rotation spreads the finger;
- the thumb: its flexion axis (CMC, MCP and IP) is b x m, m the unit direction from the thumb's joint toward
  the pinky knuckle made normal to b, so that flexion turns the thumb across the palm; the CMC's second axis
  (palmar abduction) is +-m, normal to the flexion axis and to b, signed so that a positive rotation turns the
  thumb toward the palm side (n).
Fingertips (tip_vertices): per finger, the vertex whose largest skinning weight lies on the distal joint and
that lies farthest along the distal segment's rest direction.

Hand joint limits (FLEX_LIMITS_DEG, ABDUCTION_LIMIT_DEG, CMC_LIMIT_DEG; the SP1 joint-limit table of body.py
has no hand joint): the flexion and extension of the finger MCP, PIP and DIP joints, means of B.K. et al. 2024
(390 hands of healthy adults, active range by goniometer; abstract), and of the thumb MCP and IP, means of the
left hand of Boike et al. 2025 (162 adults, digit goniometer); the thumb CMC and the knuckles' abduction have
module bounds (no source read gives them). A hand pose's components on its axes (hand_components) are checked against the table (hand_limit_report).
"""
import functools
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d, rot6d_to_matrix
from ..body.joints import N_BODY
from .body import F64, Body

N_HAND = 30
SIDES = ("left", "right")
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
FIRST = {"left": 22, "right": 37}                       # first hand joint of each side (left index MCP)
WRIST = {"left": 20, "right": 21}
LEVELS = ("mcp", "pip", "dip")                          # finger joints; thumb: cmc, mcp, ip
THUMB_LEVELS = ("cmc", "mcp", "ip")

# Flexion (positive) and extension (negative) limits in degrees: grades.SOURCES['bk2024'] means (MCP flexion:
# index 86.0, middle 86.6, ring 84.2, little 85.0; PIP flexion 97.2, 96.2, 96.0, 91.8; DIP flexion index 81.6 and
# little 84.6, the abstract placing middle and ring between them, so the lower is used; extension MCP 25.7, 24.7,
# 22.3, 26.3, PIP 13.7, 15.6, 16.2, 13.2, DIP 6.0, 8.4, 8.1, 6.8). Thumb: grades.SOURCES['boike2025'], left hand: MCP
# flexion 56.0, extension 7.4 (reported as -7.4), IP flexion 73.2, extension 21.5 (reported as -21.5).
FLEX_LIMITS_DEG = {
    "index": {"mcp": (-25.7, 86.0), "pip": (-13.7, 97.2), "dip": (-6.0, 81.6)},
    "middle": {"mcp": (-24.7, 86.6), "pip": (-15.6, 96.2), "dip": (-8.4, 81.6)},
    "ring": {"mcp": (-22.3, 84.2), "pip": (-16.2, 96.0), "dip": (-8.1, 81.6)},
    "pinky": {"mcp": (-26.3, 85.0), "pip": (-13.2, 91.8), "dip": (-6.8, 84.6)},
    "thumb": {"mcp": (-7.4, 56.0), "ip": (-21.5, 73.2)},
}
ABDUCTION_LIMIT_DEG = 20.0      # [module] knuckle abduction either way (no source read gives it)
CMC_LIMIT_DEG = 45.0            # [module] thumb CMC flexion and palmar abduction either way (no source read)
OFF_AXIS_TOL_DEG = 1e-6         # the hand poses of this module turn about their axes only


def hand_joint(side, finger, level):
    """Program joint index of a finger joint (level 0, 1, 2 or its name)."""
    lv = level if isinstance(level, int) else (THUMB_LEVELS if finger == "thumb" else LEVELS).index(level)
    return FIRST[side] + 3 * FINGERS.index(finger) + lv


def hand_row(joint):
    """Row of hand6d (0 to 29) of a hand joint (22 to 51)."""
    if not N_BODY <= int(joint) < N_BODY + N_HAND:
        raise ValueError(f"joint {joint} is not a hand joint")
    return int(joint) - N_BODY


# ---------------------------------------------------------------- the body with hands
def _eye6(T):
    return matrix_to_rot6d(torch.eye(3, dtype=F64)).expand(T, N_HAND, 6).clone()


@dataclass
class HandBody(Body):
    """A Body with the 30 hand joints (module docstring). hand6d (T, 30, 6) float64; None: every hand joint at
    rest."""
    hand6d: torch.Tensor = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.hand6d = _eye6(self.T) if self.hand6d is None else torch.as_tensor(self.hand6d).to(F64)
        if tuple(self.hand6d.shape) != (self.T, N_HAND, 6):
            raise ValueError(f"hand6d must be (T, {N_HAND}, 6) for T = {self.T}, got {tuple(self.hand6d.shape)}")

    def copy(self):
        return HandBody(self.rot6d.clone(), self.transl.clone(), self.contact.clone(), self.betas.clone(),
                        int(self.gender), float(self.rate), self.hand6d.clone())

    def hand_R(self):
        """(T, 30, 3, 3) local rotations of the hand joints."""
        return rot6d_to_matrix(self.hand6d)

    def with_hands(self, R_hand, rows=None):
        """A copy whose hand rows (default all 30) take the rotations R_hand (T, 30, 3, 3); the other rows keep
        their values bit for bit."""
        h6 = self.hand6d.clone()
        sel = list(range(N_HAND)) if rows is None else list(rows)
        h6[:, sel] = matrix_to_rot6d(torch.as_tensor(R_hand, dtype=F64)[:, sel])
        return replace(self.copy(), hand6d=h6)


def hands_of(body, hand6d=None):
    """A HandBody of a Body (its rows kept bit for bit), hands at rest unless hand6d is given."""
    if isinstance(body, HandBody) and hand6d is None:
        return body.copy()
    return HandBody(body.rot6d.clone(), body.transl.clone(), body.contact.clone(), body.betas.clone(),
                    int(body.gender), float(body.rate), hand6d)


def has_hands(body):
    return getattr(body, "hand6d", None) is not None


# ---------------------------------------------------------------- skinning
def skin(model, global_orient, body_rot, transl, betas, hand_rot=None, allow_reference=False):
    """(B, 6890, 3): the SMPL+H skinning of an SMPLHBody (smplh_reference.skin_smplh), for the reference only (behind
    its flag, openbody.require_reference; smplh_reference.py never ships)."""
    from .openbody import require_reference
    require_reference("hands.skin", allow_reference)
    from .smplh_reference import skin_smplh
    return skin_smplh(model, global_orient, body_rot, transl, betas, hand_rot)


def skin_ids(R_local, transl, betas, gender, ids, R_hand=None):
    """(N, M, 3) the vertices ids (M,) skinned for body rotations R_local (N, 22, 3, 3), translations (N, 3) and
    hand rotations R_hand (N, 30, 3, 3) (None: at rest): the selected body model's one skinning call (numpy,
    float64). betas and gender: the catalogue's one shape (body._check_open_shape on the open body)."""
    from .body import _check_open_shape, body_model, is_open
    if is_open():
        _check_open_shape(torch.as_tensor(np.asarray(betas, dtype=np.float64)), gender)
    return body_model().skin(np.asarray(R_local, dtype=np.float64), np.asarray(transl, dtype=np.float64),
                             None if R_hand is None else np.asarray(R_hand, dtype=np.float64), ids=ids)


def body_points(body, ids, knots=None, chunk=256):
    """(T, M, 3) the vertices ids of a body (HandBody or Body) at every knot (or the given knots)."""
    R, tr = body.local_R().numpy(), body.transl.numpy()
    Rh = body.hand_R().numpy() if has_hands(body) else None
    if knots is not None:
        k = np.asarray(knots, dtype=np.int64).reshape(-1)
        R, tr = R[k], tr[k]
        Rh = None if Rh is None else Rh[k]
    out = []
    for a in range(0, len(R), chunk):
        b = slice(a, a + chunk)
        out.append(skin_ids(R[b], tr[b], body.betas.numpy(), body.gender, ids, None if Rh is None else Rh[b]))
    return np.concatenate(out, 0)


# ---------------------------------------------------------------- rest geometry and axes
def _model():
    from .body import model_name
    return model_name()


def _rest(gender="neutral"):
    """(rest joints (52, 3), rest vertices (V, 3), dominant joint (V,)) of the selected body model's rest mesh."""
    return _rest_of(_model())


@functools.lru_cache(maxsize=2)
def _rest_of(name):
    from .body import body_model
    m = body_model()
    return m.rest_joints52().numpy().copy(), m.rest_vertices_np(), m.dominant_joint()


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _perp(v, b):
    return _unit(v - (v @ b) * b)


def tip_vertices(gender="neutral"):
    """{side: {finger: vertex id}}: the fingertip vertices (module docstring)."""
    return _tip_vertices(_model())


@functools.lru_cache(maxsize=2)
def _tip_vertices(name):
    J, v, dom = _rest_of(name)
    out = {}
    for side in SIDES:
        out[side] = {}
        for f in FINGERS:
            j2, j3 = hand_joint(side, f, 1), hand_joint(side, f, 2)
            ids = np.flatnonzero(dom == j3)
            b = _unit(J[j3] - J[j2])
            out[side][f] = int(ids[np.argmax((v[ids] - J[j3]) @ b)])
    return out


def hand_frame(gender="neutral"):
    """{side: {'palmar': n (3,), 'joints': {joint: {'b', 'flex', 'abd'}}}}: the anatomical axes of the module
    docstring in the rest frame (equal to each joint's parent frame at rest), unit vectors."""
    return _hand_frame(_model())


@functools.lru_cache(maxsize=2)
def _hand_frame(name):
    J, v, _ = _rest_of(name)
    tips = _tip_vertices(name)
    out = {}
    for side in SIDES:
        w, i1, p1, m1 = WRIST[side], hand_joint(side, "index", 0), hand_joint(side, "pinky", 0), \
            hand_joint(side, "middle", 0)
        n = _unit(np.cross(J[i1] - J[w], J[p1] - J[w]))
        if n[1] > 0:                                       # the palm faces the floor in the rest pose
            n = -n
        joints = {}
        for f in FINGERS:
            for lv in range(3):
                j = hand_joint(side, f, lv)
                child = v[tips[side][f]] if lv == 2 else J[j + 1]
                b = _unit(child - J[j])
                if f == "thumb":
                    m = _perp(J[p1] - J[j], b)              # toward the pinky knuckle, normal to b
                    flex = _unit(np.cross(b, m))
                    abd = None
                    if lv == 0:                             # palmar abduction: about m, toward the palm side
                        abd = m if np.cross(m, b) @ n > 0 else -m
                else:
                    nn = _perp(n, b)
                    flex = _unit(np.cross(b, nn))
                    abd = None
                    if lv == 0 and f != "middle":
                        lat = _perp(_perp(J[j] - J[m1], b) - (_perp(J[j] - J[m1], b) @ nn) * nn, b)
                        abd = _unit(np.cross(b, lat))
                joints[j] = {"finger": f, "level": lv, "b": b, "flex": flex, "abd": abd}
        out[side] = {"palmar": n, "joints": joints}
    return out


def _axis_rot(axis, deg):
    return axis_angle_to_matrix(torch.as_tensor(np.asarray(axis) * np.radians(deg), dtype=F64))


def hand_rotations(pose, gender="neutral"):
    """(30, 3, 3) local rotations of a hand pose: pose {(side, finger, level): (flex_deg, abd_deg)} (level 0 to 2
    or its name; abd_deg 0 where the joint has no abduction axis), each joint turned first about its abduction
    axis then about its flexion axis (both fixed in the parent frame); joints not given stay at rest."""
    fr = hand_frame(gender)
    R = torch.eye(3, dtype=F64).repeat(N_HAND, 1, 1)
    for (side, finger, level), val in pose.items():
        flex, abd = (val, 0.0) if np.isscalar(val) else val
        j = hand_joint(side, finger, level)
        ax = fr[side]["joints"][j]
        M = torch.eye(3, dtype=F64)
        if abd:
            if ax["abd"] is None:
                raise ValueError(f"joint {j} ({side} {finger} {level}) has no abduction axis")
            M = _axis_rot(ax["abd"], abd) @ M
        if flex:
            M = _axis_rot(ax["flex"], flex) @ M
        R[hand_row(j)] = M
    return R


def hand_components(R_hand, gender="neutral"):
    """{joint: (flex_deg (...), abd_deg (...), off_deg (...))}: the components of the axis-angle vectors of the
    local hand rotations R_hand (..., 30, 3, 3) along each joint's flexion and abduction axes, and the norm of the
    rest (deg)."""
    fr = hand_frame(gender)
    aa = np.degrees(matrix_to_axis_angle(torch.as_tensor(R_hand, dtype=F64)).numpy())
    out = {}
    for side in SIDES:
        for j, ax in fr[side]["joints"].items():
            a = aa[..., hand_row(j), :]
            fl = a @ ax["flex"]
            ab = a @ ax["abd"] if ax["abd"] is not None else np.zeros_like(fl)
            rest = a - fl[..., None] * ax["flex"] - (ab[..., None] * ax["abd"] if ax["abd"] is not None else 0.0)
            out[j] = (fl, ab, np.linalg.norm(rest, axis=-1))
    return out


def hand_limit_report(R_hand, gender="neutral", tol=1e-6):
    """JSON-ready check of hand rotations (..., 30, 3, 3) against HAND_LIMITS (module docstring): per joint the
    flexion range reached and its limits, the largest abduction and off-axis component; 'violations' counts the
    joints outside, 'ok' when none."""
    comp = hand_components(R_hand, gender)
    fr = hand_frame(gender)
    rows, bad = {}, 0
    for side in SIDES:
        for j, ax in fr[side]["joints"].items():
            f, lv = ax["finger"], ax["level"]
            fl, ab, off = comp[j]
            name = (THUMB_LEVELS if f == "thumb" else LEVELS)[lv]
            if f == "thumb" and name == "cmc":
                lo, hi = -CMC_LIMIT_DEG, CMC_LIMIT_DEG
            else:
                lo, hi = FLEX_LIMITS_DEG[f][name]
            alim = CMC_LIMIT_DEG if (f == "thumb" and name == "cmc") else ABDUCTION_LIMIT_DEG
            ok = (fl.min() >= lo - tol and fl.max() <= hi + tol and np.abs(ab).max() <= alim + tol)
            bad += int(not ok)
            rows[f"{side}_{f}_{name}"] = {"joint": int(j), "flex_min_deg": float(fl.min()),
                                          "flex_max_deg": float(fl.max()), "limits_deg": [lo, hi],
                                          "abd_max_deg": float(np.abs(ab).max()), "abd_limit_deg": alim,
                                          "off_axis_max_deg": float(off.max()), "ok": bool(ok)}
    return {"joints": rows, "violations": bad, "ok": bad == 0, "sources": ["bk2024", "boike2025", "module"]}
