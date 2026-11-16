"""The synthetic patient's body: one trajectory of the program's body representation on the 100 Hz clock, its
forward kinematics, the joint-limit table and the kinematic checks that every operator of this package reports.

Representation (the program's body, design section 1; mdfm/data/body_store.py): rot6d (T, 22, 6) with row 0
the root orientation and rows 1 to 21 the local joint rotations, transl (T, 3) m (the pelvis is transl plus the
body's rest pelvis), contact (T, 2) bool [left, right], betas (16,) and the gender code (0 female, 1 male, 2
neutral) of the model track's records, y up, floor at y = 0. Body holds float64 tensors on the CPU; a body read from
the float32 store converts exactly, and the rows an operator leaves alone keep their values bit for bit. Stimulus
bodies carry one shared shape (plan principle 9).

The body model (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md, wave 1): the rest skeleton, the mesh and the skinning
come from the selected backend (body_model): the open body (synth/openbody.py, Anny, code Apache-2.0 and model data
CC0) by default, in the catalogue's one shape (openbody.catalogue_phenotype, design D8: a neutral adult of 40 years);
the SMPL+H reference (synth/smplh_reference.py) only with MDFM_BODY_MODEL=smplh_reference and MDFM_SMPLH_REFERENCE=1,
for the clip-by-clip comparison of the port. On the open body a body's betas must be zero and its gender code the
neutral one (the catalogue's one shape; another value refuses, ValueError): the shape is the backend's, not the
record's. A module whose code still reads the SMPL+H mesh itself (viewer.mesh_model) refuses on the open body until
its wave ports it (the design's waves 2 to 7).

Joint-limit table (design section 5, decision D13; owner decision C8 of 2026-11-12: the AMASS-percentile table of
reports/body_store/priors.json is out of the synthetic patient, the model track keeps its own until its port): the
anatomical table of reports/openbody/joint_limits.json (scripts/openbody_limits.py: jarm2022 for every joint, typed
once with its source, mapped onto the program's axes and signs from the canonical T pose; regional ranges shared
over the joints that carry them, stated there), on either body. A component is the axis-angle vector of a local
joint rotation in degrees; the range of motion of a joint axis is hi - lo of that table. The root (row 0) is a world
orientation and is not checked against the table.

Kinematic checks (validity):
- joint limits: a component of the output outside [min(lo, base), max(hi, base)] at a knot is a new
  violation (an operator never pushes a joint beyond the table or beyond the base body's own excursion);
- foot skate in stance: the horizontal speed of each foot point (the mean of its ankle and foot joints, the
  contact point of mdfm/model/mfm.py) over the knots where the output's contact label holds, its 95th
  percentile against the base's;
- contact: the share of the base's contact knots that the output loses, and the lowest foot joint against
  the base's (penetration). An operator that moves a foot on purpose (a leg task or posture) recomputes the
  contact labels of its output by the body-store rule and checks the stance of the other foot.
"""
import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch

from .. import paths
from ..body import frames
from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d, rot6d_to_matrix
from ..body.joints import BODY_JOINTS, N_BODY, fk

RATE = 100.0
N_BETAS = 16                                  # the record's shape vector (the model track's records, mdfm/data/
#                                               body_store.py; mdfm/body/smplh.py's N_BETAS): zero on the open body
F64 = torch.float64
NEUTRAL = 2                                   # the gender code of the neutral shape (plan principle 9)
LIMITS_PATH = os.path.join(paths.TREE, "reports", "openbody", "joint_limits.json")   # scripts/openbody_limits.py
AXES = {"x": 0, "y": 1, "z": 2}
FEET = ((7, 10), (8, 11))                     # (ankle, foot) joints of the left and right foot
LIMIT_TOL_DEG = 1e-6                          # rounding allowance of the joint-limit check
# Anatomical signs that differ between the sides (a rotation about a fixed body axis x keeps its sign under the
# left-right mirror; about y or about a bone that points away from the midline it flips):
ELBOW_FLEXION_SIGN = {"left": -1.0, "right": 1.0}   # elbow flexion about the upper arm's y axis (base.py)
SUPINATION_SIGN = {"left": -1.0, "right": 1.0}      # supination about the distal forearm bone (elbow to wrist)


# ---------------------------------------------------------------- the body model (selector)
def model_name():
    """The selected body model: openbody.OPEN by default; openbody.SMPLH_REFERENCE when MDFM_BODY_MODEL names it
    and MDFM_SMPLH_REFERENCE=1 (openbody.OpenBodyRefused without the flag)."""
    from . import openbody as OB
    name = os.environ.get(OB.BACKEND_ENV, OB.OPEN) or OB.OPEN
    if name == OB.SMPLH_REFERENCE and os.environ.get(OB.REFERENCE_ENV) != "1":
        raise OB.OpenBodyRefused(f"{OB.BACKEND_ENV}={name} needs {OB.REFERENCE_ENV}=1: the SMPL+H reference is for the "
                                 "port's comparison only")
    if name not in (OB.OPEN, OB.SMPLH_REFERENCE):
        raise ValueError(f"body model {name!r}, expected {OB.OPEN!r} or {OB.SMPLH_REFERENCE!r}")
    return name


def is_open():
    from .openbody import OPEN
    return model_name() == OPEN


@lru_cache(maxsize=2)
def _backend(name):
    from . import openbody as OB
    if name == OB.OPEN:
        return OB.OpenBody(OB.catalogue_phenotype())
    return OB.backend(name, allow_reference=True)


def body_model():
    """The selected backend (openbody.OpenBody of the catalogue body, or smplh_reference.SMPLHReference): its
    rest_joints52, rest_vertices_np, dominant_joint, faces and skin, the core's one mesh interface."""
    return _backend(model_name())


def _check_open_shape(betas, gender):
    if betas is not None and bool(torch.as_tensor(betas).abs().max() > 0):
        raise ValueError("the open body carries the catalogue's one shape: betas must be zero (openbody.py)")
    if int(gender) != NEUTRAL:
        raise ValueError(f"the open body carries the catalogue's one shape: gender code {int(gender)}, expected "
                         f"{NEUTRAL} (openbody.py)")


# ---------------------------------------------------------------- skeleton
# The reference's SkeletonBank rest (smplh_reference.smplh_rest) is imported only in the reference branch of
# rest_joints, which model_name() reaches behind the flag alone (open-body port wave 8: the SMPL+H code of synth
# lives in smplh_reference.py, which never ships).
@lru_cache(maxsize=2)
def _open_rest(name):
    return _backend(name).rest_joints().to(F64).clone()


def rest_joints(betas=None, gender=NEUTRAL):
    """Rest joints (22, 3) m, float64, of the selected body model (module docstring): the open body's canonical
    rest (betas zero and the neutral code only), or the reference's SkeletonBank rest of a shape (betas (16,),
    default 0) and gender code."""
    if is_open():
        _check_open_shape(betas, gender)
        return _open_rest(model_name()).clone()
    from .smplh_reference import smplh_rest
    return smplh_rest(betas, gender)


def rest_joints52():
    """(52, 3) m: the selected body model's rest joints with the hands (the neutral shape)."""
    return body_model().rest_joints52().to(F64).clone()


def segment_length(a, b, betas=None, gender=NEUTRAL):
    """Distance (m) between rest joints a and b."""
    r = rest_joints(betas, gender)
    return float((r[b] - r[a]).norm())


# ---------------------------------------------------------------- the body
@dataclass
class Body:
    """One trajectory (module docstring)."""
    rot6d: torch.Tensor
    transl: torch.Tensor
    contact: torch.Tensor
    betas: torch.Tensor = field(default_factory=lambda: torch.zeros(N_BETAS, dtype=F64))
    gender: int = NEUTRAL
    rate: float = RATE

    def __post_init__(self):
        self.rot6d = torch.as_tensor(self.rot6d).to(F64)
        self.transl = torch.as_tensor(self.transl).to(F64)
        self.contact = torch.as_tensor(self.contact).to(torch.bool)
        self.betas = torch.as_tensor(self.betas).to(F64).reshape(-1)
        T = self.rot6d.shape[0]
        if self.rot6d.dim() != 3 or tuple(self.rot6d.shape[1:]) != (N_BODY, 6):
            raise ValueError(f"rot6d must be (T, 22, 6), got {tuple(self.rot6d.shape)}")
        if tuple(self.transl.shape) != (T, 3) or tuple(self.contact.shape) != (T, 2):
            raise ValueError(f"transl (T, 3) and contact (T, 2) for T = {T}, got {tuple(self.transl.shape)} and "
                             f"{tuple(self.contact.shape)}")
        if self.betas.numel() != N_BETAS:
            raise ValueError(f"betas must hold {N_BETAS} values")
        if T < 2:
            raise ValueError("a body needs at least two knots")

    @property
    def T(self):
        return int(self.rot6d.shape[0])

    @property
    def duration(self):
        """(T - 1) / rate s, the span of the knots."""
        return (self.T - 1) / self.rate

    def times(self):
        """(T,) float64 s, knot k at k / rate."""
        return torch.arange(self.T, dtype=F64) / self.rate

    def copy(self):
        return Body(self.rot6d.clone(), self.transl.clone(), self.contact.clone(), self.betas.clone(), int(self.gender),
                    float(self.rate))

    def rest(self):
        return rest_joints(self.betas, self.gender)

    def local_R(self):
        """(T, 22, 3, 3) rotations of rot6d (row 0 the root orientation)."""
        return rot6d_to_matrix(self.rot6d)

    def fk(self):
        """Joint positions (T, 22, 3) m and global rotations (T, 22, 3, 3), float64."""
        return fk(self.local_R(), self.transl, self.rest())

    def joints(self):
        return self.fk()[0]

    def with_rotations(self, R_local, joints=None):
        """A copy whose rows `joints` (default every row) take the rotations R_local (T, 22, 3, 3); the other
        rows keep their rot6d values bit for bit."""
        r6 = self.rot6d.clone()
        sel = list(range(N_BODY)) if joints is None else list(joints)
        r6[:, sel] = matrix_to_rot6d(R_local[:, sel])
        return replace(self.copy(), rot6d=r6)


def crop(body, a, b):
    """Knots [a, b) of a body (contacts kept)."""
    if not 0 <= a < b <= body.T or b - a < 2:
        raise ValueError(f"crop [{a}, {b}) of a body of {body.T} knots")
    return Body(body.rot6d[a:b].clone(), body.transl[a:b].clone(), body.contact[a:b].clone(), body.betas.clone(),
                int(body.gender), float(body.rate))


def stack_bodies(bodies):
    """(rot6d (n, K, 22, 6), transl (n, K, 3), contact (n, K, 2), lengths (n,)) of bodies padded to the longest
    by holding each one's last knot, float64."""
    K = max(b.T for b in bodies)
    r6 = torch.stack([torch.cat([b.rot6d, b.rot6d[-1:].expand(K - b.T, N_BODY, 6)]) for b in bodies])
    tr = torch.stack([torch.cat([b.transl, b.transl[-1:].expand(K - b.T, 3)]) for b in bodies])
    ct = torch.stack([torch.cat([b.contact, torch.zeros(K - b.T, 2, dtype=torch.bool)]) for b in bodies])
    return r6, tr, ct, torch.tensor([b.T for b in bodies], dtype=torch.int64)


def to_windows(bodies, person="synthetic:neutral", task=None):
    """data.body_store.BodyWindows (float32) of bodies, padded to the longest by holding each one's last knot
    (valid False after it), for factory.synth.SynthSampler.render. person: one id or one per body; task: task
    ids (default the vocabulary's unknown)."""
    from ..data.body_store import BodyWindows
    from ..model.prior import TASK_UNKNOWN
    bodies = list(bodies)
    n = len(bodies)
    r6, tr, ct, L = stack_bodies(bodies)
    K = r6.shape[1]
    persons = np.array([person] * n if isinstance(person, str) else list(person))
    tasks = torch.full((n,), TASK_UNKNOWN, dtype=torch.int64) if task is None else \
        torch.as_tensor(task, dtype=torch.int64).expand(n).clone()
    return BodyWindows(r6.float(), tr.float(), ct, torch.arange(K)[None] < L[:, None], L,
                       torch.stack([b.betas for b in bodies]).float(),
                       torch.tensor([int(b.gender) for b in bodies], dtype=torch.int64), persons, tasks,
                       torch.arange(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64))


def from_windows(bw, i):
    """Body of row i of a BodyWindows (its valid knots)."""
    n = int(bw.lengths[i])
    return Body(bw.rot6d[i, :n], bw.transl[i, :n], bw.contact[i, :n], bw.betas[i], int(bw.gender[i]))


def window_frame(body):
    """The body in its window frame (data/body_store.py): floor kept at y = 0, the pelvis heading on +z and the
    pelvis x and z at 0 at the centre knot T // 2 (frames.heading_frame and frames.frame_knots)."""
    rest = body.rest()
    J, _ = body.fk()
    R = body.local_R()
    c = body.T // 2
    R_yaw, origin = frames.heading_frame(J[None], R[None, :, 0], c)
    r6, tr = frames.frame_knots(body.rot6d[None], body.transl[None], rest[0][None], R_yaw, origin)
    return replace(body.copy(), rot6d=r6[0], transl=tr[0])


def floor_to_zero(body):
    """The body shifted vertically so that its lowest foot joint (10 or 11) over every knot lies at y = 0."""
    J = body.joints()
    low = float(torch.minimum(J[:, 10, 1], J[:, 11, 1]).min())
    tr = body.transl.clone()
    tr[:, 1] -= low
    return replace(body.copy(), transl=tr)


def contact_labels(body, joints=None):
    """(T, 2) bool: the body-store rule (frames.contact_labels, floor y = 0, central-difference velocities)."""
    J = body.joints() if joints is None else joints
    return frames.contact_labels(J[None], body.rate, floor=0.0)[0]


# ---------------------------------------------------------------- rotations
def axis_rotation(axis, angle_rad):
    """(..., 3, 3) rotations about a body axis ('x', 'y' or 'z', or a unit 3-vector) by angle_rad (...)."""
    a = torch.as_tensor(angle_rad, dtype=F64)
    if isinstance(axis, str):
        u = torch.zeros(3, dtype=F64)
        u[AXES[axis]] = 1.0
    else:
        u = torch.as_tensor(axis, dtype=F64)
        u = u / u.norm()
    return axis_angle_to_matrix(a[..., None] * u)


def compose(R_base, R_off, frame):
    """R_base (..., 3, 3) composed with the offset R_off: frame 'child' gives R_base R_off (about the moving
    segment's own axes: an axial rotation, a deviation in the segment frame), 'parent' gives R_off R_base
    (about the parent segment's axes: a hinge fixed in the parent)."""
    if frame == "child":
        return R_base @ R_off
    if frame == "parent":
        return R_off @ R_base
    raise ValueError(f"frame {frame!r}, expected 'child' or 'parent'")


def rotation_angle_deg(R):
    """(...) deg, the angle of rotations R (..., 3, 3)."""
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def deviation(G_base, G_new):
    """(angle deg (...), axis-angle components deg (..., 3)) of D = G_base^T G_new, the deviation of a segment
    in its own frame (the D(t) of DR-14.2 with the level-0 body as the norm)."""
    D = G_base.transpose(-1, -2) @ G_new
    return rotation_angle_deg(D), torch.rad2deg(matrix_to_axis_angle(D))


# ---------------------------------------------------------------- the joint-limit table
class JointLimits(NamedTuple):
    """lo, hi (22, 3) deg: the bounds of each axis-angle component per joint (module docstring); percentiles and
    n_sampled are those of a percentile table (empty and 0 for the anatomical one)."""
    lo: np.ndarray
    hi: np.ndarray
    percentiles: tuple
    path: str
    n_sampled: int

    @property
    def rom(self):
        """(22, 3) deg, hi - lo: the range of motion of each joint axis."""
        return self.hi - self.lo


@lru_cache(maxsize=4)
def joint_limits(path=LIMITS_PATH):
    """The anatomical joint-limit table (module docstring), from reports/openbody/joint_limits.json."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "body" not in d or d.get("script") != "scripts/openbody_limits.py":
        raise ValueError(f"{path}: not the anatomical table of scripts/openbody_limits.py")
    jl = d["body"]
    if list(jl["joints"]) != list(BODY_JOINTS):
        raise ValueError(f"{path}: joints differ from the program's 22 body joints")
    lo, hi = np.asarray(jl["lo_deg"], np.float64), np.asarray(jl["hi_deg"], np.float64)
    if lo.shape != (N_BODY, 3) or not (lo <= hi).all():
        raise ValueError(f"{path}: lo and hi must be (22, 3) with lo <= hi")
    lo.setflags(write=False)
    hi.setflags(write=False)
    return JointLimits(lo, hi, (), os.path.relpath(path, paths.TREE).replace(chr(92), "/"), 0)


def components_deg(rot6d):
    """(T, 22, 3) deg: axis-angle components of every row of rot6d (T, 22, 6)."""
    return torch.rad2deg(matrix_to_axis_angle(rot6d_to_matrix(torch.as_tensor(rot6d, dtype=F64)))).numpy()


def bone_axis(joint, child, betas=None, gender=NEUTRAL):
    """(3,) unit vector of the rest bone from joint to child, in joint's own frame (the long axis of the segment
    the joint rotates: the forearm axis for 18 to 20)."""
    r = rest_joints(betas, gender)
    v = r[child] - r[joint]
    return v / v.norm()


def room_deg(R_base, joint, axis, sign, frame, limits=None, max_deg=180.0, step_deg=0.5):
    """Largest angle (deg) of an offset about axis ('x', 'y', 'z' or a unit vector) in the direction sign,
    composed with the local rotation R_base (3, 3) of joint in frame ('child' or 'parent'), for which every
    axis-angle component of the result stays inside the SP1 table, or inside the base's own excursion where the
    base already lies outside it. Scanned in steps of step_deg, then refined by bisection to 1e-6 deg."""
    lim = joint_limits() if limits is None else limits
    c0 = torch.rad2deg(matrix_to_axis_angle(torch.as_tensor(R_base, dtype=F64))).numpy()
    lo = np.minimum(lim.lo[joint], c0) - LIMIT_TOL_DEG
    hi = np.maximum(lim.hi[joint], c0) + LIMIT_TOL_DEG

    def inside(deg):
        R = compose(torch.as_tensor(R_base, dtype=F64), axis_rotation(axis, torch.as_tensor(sign * np.radians(deg))),
                    frame)
        c = torch.rad2deg(matrix_to_axis_angle(R)).numpy()
        return ((c >= lo) & (c <= hi)).all(-1)
    grid = np.arange(step_deg, max_deg + step_deg / 2, step_deg)
    ok = inside(grid)
    if ok.all():
        return float(max_deg)
    k = int(np.argmin(ok))
    a, b = (grid[k - 1] if k else 0.0), grid[k]
    for _ in range(40):
        m = 0.5 * (a + b)
        a, b = (m, b) if bool(inside(np.array([m]))[0]) else (a, m)
    return float(a)


def limit_violations(rot6d, base=None, limits=None, joints=range(1, N_BODY), tol=LIMIT_TOL_DEG):
    """Counts of axis-angle components (joints 1 to 21 by default) outside the table: 'outside' against [lo,
    hi]; with a base rot6d of the same shape, 'new' against [min(lo, base), max(hi, base)] (module docstring).
    Returns {'outside', 'new' (or None), 'worst_deg' (largest new excess, or largest excess without base)}."""
    lim = joint_limits() if limits is None else limits
    j = list(joints)
    a = components_deg(rot6d)[:, j]
    lo, hi = lim.lo[j][None], lim.hi[j][None]
    outside = int(((a < lo - tol) | (a > hi + tol)).sum())
    if base is None:
        excess = np.maximum(lo - a, a - hi).max(initial=0.0)
        return {"outside": outside, "new": None, "worst_deg": float(max(excess, 0.0))}
    b = components_deg(base)[:, j]
    lo2, hi2 = np.minimum(lo, b), np.maximum(hi, b)
    excess = np.maximum(lo2 - a, a - hi2)
    return {"outside": outside, "new": int((excess > tol).sum()), "worst_deg": float(max(excess.max(initial=0.0), 0.0))}


# ---------------------------------------------------------------- kinematic checks
def foot_points(joints):
    """(T, 2, 3) m: the mean of the ankle and foot joints of each foot."""
    return torch.stack([joints[:, list(f)].mean(1) for f in FEET], 1)


def foot_speed(joints, rate=RATE):
    """(T, 2) m/s: horizontal speed of each foot point (central differences)."""
    v = torch.gradient(foot_points(joints), spacing=1.0 / rate, dim=0)[0]
    return torch.sqrt(v[..., 0] ** 2 + v[..., 2] ** 2)


def skate(joints, contact, rate=RATE, q=0.95, feet=(0, 1)):
    """{'p95', 'mean', 'n'}: the q quantile and mean (m/s) of the horizontal foot-point speed over the contact
    knots of the given feet (0 left, 1 right), pooled (0 with no contact knot)."""
    f = list(feet)
    s = foot_speed(joints, rate)[:, f][torch.as_tensor(contact, dtype=torch.bool)[:, f]]
    if not s.numel():
        return {"p95": 0.0, "mean": 0.0, "n": 0}
    return {"p95": float(torch.quantile(s, q)), "mean": float(s.mean()), "n": int(s.numel())}


def validity(base, out, limits=None, knots=None, feet=(0, 1)):
    """The kinematic checks of an operator output against its base body (module docstring), JSON-ready:
    joint limits, skate in stance (base on its own labels, output on its labels), contact knots lost, lowest
    foot joint. knots: a (T,) bool selection of the output's knots for the skate (default all); feet: the
    feet whose stance is checked (an operator that moves a foot on purpose checks the other one; the moved
    foot's lost contact knots are reported apart)."""
    Jb, Jo = base.joints(), out.joints()
    lim = limit_violations(out.rot6d, base.rot6d if base.T == out.T else None, limits)
    sel = torch.ones(out.T, dtype=torch.bool) if knots is None else torch.as_tensor(knots, dtype=torch.bool)
    sk_o = skate(Jo[sel], out.contact[sel], out.rate, feet=feet)
    sk_b = skate(Jb, base.contact, base.rate, feet=feet)
    lost = moved_lost = None
    f = list(feet)
    others = [i for i in (0, 1) if i not in f]
    if base.T == out.T:
        n = int(base.contact[:, f].sum())
        lost = float((base.contact[:, f] & ~out.contact[:, f]).sum()) / n if n else 0.0
        if others:
            m = int(base.contact[:, others].sum())
            moved_lost = float((base.contact[:, others] & ~out.contact[:, others]).sum()) / m if m else 0.0
    low = lambda J: float(torch.minimum(J[:, 10, 1], J[:, 11, 1]).min())  # noqa: E731
    return {"limits_outside": lim["outside"], "limits_new": lim["new"], "limits_worst_deg": lim["worst_deg"],
            "skate_feet": f, "skate_p95_m_s": sk_o["p95"], "skate_base_p95_m_s": sk_b["p95"],
            "skate_added_m_s": max(0.0, sk_o["p95"] - sk_b["p95"]), "contact_lost_share": lost,
            "moved_foot_contact_lost_share": moved_lost, "lowest_foot_m": low(Jo), "lowest_foot_base_m": low(Jb)}
