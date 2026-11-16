"""The skeleton of a synthetic-patient clip (task SKELV): the joint positions of a body on the viewer's clock, for
the skeleton catalogue (scripts/render_skeleton_catalogue.py), where a symptom is seen as bones and joints beside
its level 0.

Clock and pose: frame k sits at t_k = k / fps (viewer.frame_times, 30 fps by default); the 22 body rotations and the
translation are read at t_k by the body's Hermite interpolant and re-orthonormalized (viewer.pose_at), the 30 hand
rotations of a hands.HandBody likewise (viewer.hands_at), as for the mesh of viewer.vertices_at. Positions: the 22
body joints by the forward kinematics of mdfm.body.joints.fk on the body's rest joints (Body.rest(), the joints of
Body.joints()), so that at a frame on a knot they are Body.joints() at that knot; the 30 hand joints continue the
chain from the wrists' global rotations on the bone offsets of the rest joints of the selected body model
(body.rest_joints52: the joints viewer.vertices_at skins the mesh on; the open body by default, plan phase C wave 1)
with the program's parents (openbody.PARENTS52, the same tree on both models).

Hands and head (why more than the 22 joints): a rotation about a bone moves no joint of that bone, so the 22 body
joints show neither pronation-supination (3.6: the forearm turns about the elbow-to-wrist axis, the wrist joint
stays put) nor a neck rotation (TWSTRS: the head joint stays put). The skeleton therefore carries
- the 30 hand joints for every clip (hands=True): with the hand rotations of a hands.HandBody (hand_pose
  'animated'), else at rest in the wrist's frame (hand_pose 'rest'), the hands the mesh viewer skins for such a body;
- two head landmarks (head_landmarks), points fixed in the head joint's frame (joint 15, its global rotation from
  the forward kinematics): the head centre and the nose, offsets rounded to 1 cm from the rest mesh of the selected
  body model (on SMPL+H the mean of the vertices whose dominant joint is the head, on Anny the midpoint of their
  extent, the face being meshed more densely than the skull; the nose the most forward head vertex), measured at first
  use (HEAD_LANDMARKS keeps the SMPL+H values, the reference's); drawing landmarks of the head's
  orientation, not mesh vertices.

Output (Skeleton): positions (N, P, 3) m, y up, in the frame of the body (the procedural bodies and the retargeted
references face +z, viewer.heading_deg); P = 52 joints (mdfm.body.joints.BODY_JOINTS, then the hand joints
<side>_<finger><1 to 3> in the program order of hands.FINGERS) plus the landmarks; the parents (-1 for the root, the
head joint for the landmarks) and the bones (child, parent). Joint positions and the two landmarks only: no vertex,
face, shape coefficient or model array leaves this module (the model arrays stay in memory); record()
is the JSON-ready form, positions rounded to 1 mm.
"""
import functools
from dataclasses import dataclass

import numpy as np
import torch

from ..body.joints import BODY_JOINTS, N_BODY, PARENTS, fk
from . import viewer as V
from .hands import FINGERS, N_HAND, SIDES, has_hands

N_WITH_HANDS = N_BODY + N_HAND
HEAD = 15
HAND_JOINTS = tuple(f"{s}_{f}{k}" for s in SIDES for f in FINGERS for k in (1, 2, 3))
JOINT_NAMES = tuple(BODY_JOINTS) + HAND_JOINTS
# [module] m, in the head joint's frame (the rest frame at rest: y up, the face toward +z); rounded to 1 cm from the
# neutral rest mesh: the mean of the head vertices is 0.07 m above and 0.02 m in front of the head joint, the most
# forward head vertex (the nose) 0.07 m above and 0.10 m in front (task SKELV probe, 2026-11-02)
HEAD_LANDMARKS = {"head_centre": (0.0, 0.07, 0.02), "nose": (0.0, 0.07, 0.10)}
LANDMARK_ROUND_M = 0.01      # the landmarks' offsets rounded to 1 cm (a drawing landmark, not a vertex)


def head_landmarks():
    """{'head_centre', 'nose'}: offsets (m) in the head joint's frame on the selected body model: HEAD_LANDMARKS on
    the SMPL+H reference; on the open body measured on its canonical rest mesh (the centre: the midpoint of the head
    vertices' extent; the nose: the most forward head vertex), rounded to LANDMARK_ROUND_M."""
    from .body import model_name
    return _head_landmarks(model_name())


@functools.lru_cache(maxsize=2)
def _head_landmarks(name):
    from .openbody import OPEN
    if name != OPEN:
        return dict(HEAD_LANDMARKS)
    from .body import rest_joints52
    v, dom = V.rest_vertices(), V.dominant_joint()
    head = np.flatnonzero(dom == HEAD)
    j = rest_joints52().numpy()[HEAD]
    # the midpoint of the head vertices' extent: on Anny the face is meshed far more densely than the skull, so their
    # mean lies 6 cm in front of the head joint, next to the mouth's inner surface (reports/openbody/core_wave1.json)
    centre = 0.5 * (v[head].min(0) + v[head].max(0)) - j
    nose = v[head[np.argmax(v[head, 2])]] - j
    rnd = lambda x: tuple(float(round(c / LANDMARK_ROUND_M) * LANDMARK_ROUND_M) + 0.0 for c in x)  # noqa: E731
    return {"head_centre": rnd(centre), "nose": rnd(nose)}


@dataclass(frozen=True)
class Skeleton:
    """Points of one clip (module docstring): positions (N, P, 3) float64 m, fps, names (P,), parents (P,),
    n_joints (the program joints among the P points, 22 or 52; the landmarks follow them), hand_pose ('animated',
    'rest' or None without hand joints)."""
    positions: np.ndarray
    fps: float
    names: tuple
    parents: tuple
    n_joints: int
    hand_pose: object = None

    @property
    def n_frames(self):
        return int(self.positions.shape[0])

    @property
    def n_points(self):
        return int(self.positions.shape[1])

    @property
    def hands(self):
        return self.n_joints == N_WITH_HANDS

    @property
    def landmarks(self):
        return self.names[self.n_joints:]

    def times(self):
        """(N,) s: t_k = k / fps."""
        return np.arange(self.n_frames, dtype=np.float64) / self.fps

    def bones(self):
        """[(child, parent)] of every point but the root."""
        return bones(self.parents)


def bones(parents):
    """[(child, parent)] of a parent list (the root, parent -1, has no bone)."""
    return [(j, int(p)) for j, p in enumerate(parents) if int(p) >= 0]


def model_parents(gender=2):
    """(52,) parents of the program's 52 joints (openbody.PARENTS52, the tree of both body models; the first 22 are
    PARENTS)."""
    from .openbody import PARENTS52
    if tuple(PARENTS52[:N_BODY]) != tuple(PARENTS):
        raise RuntimeError("unexpected body tree")
    return tuple(PARENTS52)


def chain_at(body, t, hands=True):
    """(positions (N, J, 3), global rotations (N, J, 3, 3)) float64 of the joints of a body at times t (N,) s
    (module docstring); J = 52 with hands (the hand rotations of a hands.HandBody, at rest for any other body),
    else 22."""
    R, tr = V.pose_at(body, t)
    with torch.no_grad():
        P, G = fk(R, tr, body.rest())
        if not hands:
            return P.numpy(), G.numpy()
        Rh = V.hands_at(body, t)
        from .body import rest_joints52
        V._shape_of(body)
        J = rest_joints52()
        par = model_parents(body.gender)
        pos, rot = list(P.unbind(-2)), list(G.unbind(-3))
        for i in range(N_BODY, N_WITH_HANDS):
            p = par[i]
            pos.append(pos[p] + (rot[p] @ (J[i] - J[p])[:, None])[..., 0])
            rot.append(rot[p] if Rh is None else rot[p] @ Rh[:, i - N_BODY])
        return torch.stack(pos, -2).numpy(), torch.stack(rot, -3).numpy()


def joints_at(body, t, hands=True):
    """(N, J, 3) float64 m: the joint positions of chain_at."""
    return chain_at(body, t, hands)[0]


def landmarks_from(P, G, names=tuple(HEAD_LANDMARKS)):
    """(N, L, 3) m: the head landmarks (head_landmarks) of joint positions P (N, J, 3) and global rotations G
    (N, J, 3, 3): the head joint plus its global rotation times each offset."""
    lm = head_landmarks()
    off = np.array([lm[n] for n in names], dtype=np.float64)
    return P[:, HEAD, None, :] + np.einsum("nab,lb->nla", G[:, HEAD], off)


def skeleton(body, fps=V.FPS, hands=True, landmarks=True):
    """Skeleton of a body on the clock t_k = k / fps over its span (viewer.frame_times): the 52 joints (hands) or
    22, then the head landmarks (landmarks)."""
    t = V.frame_times(body.duration, fps)
    P, G = chain_at(body, t, hands)
    J = P.shape[1]
    par = model_parents(body.gender) if J == N_WITH_HANDS else tuple(PARENTS)
    names, parents = JOINT_NAMES[:J], par[:J]
    if landmarks:
        P = np.concatenate([P, landmarks_from(P, G)], 1)
        names = names + tuple(HEAD_LANDMARKS)
        parents = parents + (HEAD,) * len(HEAD_LANDMARKS)
    hp = None if J == N_BODY else ("animated" if has_hands(body) else "rest")
    return Skeleton(P, float(fps), names, tuple(parents), J, hp)


def quantize_mm(positions, mm=1):
    """(N, P, 3) int64: positions (m) in millimetres rounded to a multiple of mm."""
    p = np.asarray(positions, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite joint position")
    return np.rint(p * (1000.0 / mm)).astype(np.int64) * int(mm)


def record(skel, mm=1, delta=False):
    """JSON-ready dict of a Skeleton: fps, frame, joint and point counts, names, parents, bones, the landmarks, and
    the positions in millimetres rounded to a multiple of mm (quantize_mm), flat in the order frame, point, axis
    (x, y, z; y up): positions_mm, or with delta positions_mm_delta (frame 0 absolute, then each frame minus the
    previous one; the positions are its running sum over frames, decode_positions)."""
    q = quantize_mm(skel.positions, mm)
    out = {"unit": "mm", "rounding_mm": int(mm), "up": "y", "fps": float(skel.fps), "n_frames": skel.n_frames,
           "n_points": skel.n_points, "n_joints": skel.n_joints, "hands": skel.hands, "hand_pose": skel.hand_pose,
           "duration_s": (skel.n_frames - 1) / skel.fps, "names": list(skel.names), "parents": list(skel.parents),
           "bones": [list(b) for b in skel.bones()],
           "landmarks": {n: {"joint": HEAD, "offset_m": list(head_landmarks()[n])} for n in skel.landmarks},
           "layout": "frame, point, axis"}
    if delta:
        d = q.copy()
        d[1:] -= q[:-1]
        out["encoding"] = "delta over frames"
        out["positions_mm_delta"] = d.reshape(-1).tolist()
    else:
        out["positions_mm"] = q.reshape(-1).tolist()
    return out


def decode_positions(rec):
    """(N, P, 3) float64 m of a record (record()), either encoding."""
    shape = (rec["n_frames"], rec["n_points"], 3)
    if "positions_mm_delta" in rec:
        q = np.cumsum(np.asarray(rec["positions_mm_delta"], dtype=np.int64).reshape(shape), axis=0)
    else:
        q = np.asarray(rec["positions_mm"], dtype=np.int64).reshape(shape)
    return q / 1000.0
