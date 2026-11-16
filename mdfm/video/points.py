"""The drawn point set of the generator's pose video, from an SMPL+H body (proposal section 3.1).

The Wan2.2 pose drawing (draw_aapose_new, mdfm/video/draw.py) draws 20 body points and 21 points per hand.
Its preprocessing builds them from the 133 COCO-WholeBody keypoints of ViTPose (split_pose2d_kps_to_aa: body
point k is the mean of points A[k] and B[k], the neck the mean of the two shoulders, each toe the mean of the
big and small toe; the left hand is points 91 to 111, the right hand 112 to 132). The driver projects the same
set from the SMPL+H body, so that the pose video is exact and the truth of every measure is this set:

- shoulders, elbows, wrists, hips, knees and ankles: the SMPL joint centres;
- neck: the midpoint of the two shoulder points, as the preprocessing builds it;
- nose, eyes, ears, and each toe (the mean of the big-toe and small-toe vertices): mesh vertices;
- hands, COCO-WholeBody order (wrist, then thumb, index, middle, ring and little finger from base to tip): the
  wrist joint, the three SMPL+H joints of each finger and a fingertip vertex. Hands stay at their rest pose
  (the body store carries no hand pose), as SMPLHBody.vertices holds them, unless hand rotations are given (the
  opt-in path of task VCH: a synthetic body with posed hands, mdfm.synth.hands.HandBody): then the 30 hand joints
  turn by their local rotations and the fingertip vertices take the hands' pose blend shapes too, the arithmetic of
  mdfm.synth.hands.skin (SMPLHBody.vertices with the hand rotations); hand rotations between knots are the Hermite
  interpolant of their rot6d (mdfm.body.hermite.hermite, as mdfm.synth.viewer.hands_at).

Vertex ids are the SMPL+H landmark vertices of the SMPL family (nose 332, eyes 6260 and 2800, ears 4071 and
583, toes 3216, 3226, 6617, 6624, fingertips 2746, 2319, 2445, 2556, 2673 on the left and 6191, 5782, 5905,
6016, 6133 on the right); tests/test_video_driver.py checks them against the rest mesh (left-right mirror
pairs, left on +x, the nose on the midline ahead of the head joint, toes ahead of and below the ankles,
fingertips distal to the last finger joint, thumb in front). Points of a vertex are posed by linear blend
skinning of those vertices only (SMPLHPoints.posed), equal to SMPLHBody.vertices at them.
"""
import functools

import numpy as np
import torch

from ..body.kinematics import KnotBatch
from ..body.joints import N_BODY

BODY20 = ("Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
          "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LToe", "RToe")
HAND21 = ("Wrist", "Thumb1", "Thumb2", "Thumb3", "Thumb4", "Index1", "Index2", "Index3", "Index4", "Middle1",
          "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3", "Ring4", "Little1", "Little2", "Little3",
          "Little4")
N_BODY20, N_HAND = len(BODY20), len(HAND21)
BODY = slice(0, N_BODY20)
LHAND = slice(N_BODY20, N_BODY20 + N_HAND)
RHAND = slice(N_BODY20 + N_HAND, N_BODY20 + 2 * N_HAND)
N_POINTS = N_BODY20 + 2 * N_HAND
NAMES = BODY20 + tuple("LHand" + h for h in HAND21) + tuple("RHand" + h for h in HAND21)
IDX = {n: i for i, n in enumerate(BODY20)}

# Wan2.2 split_pose2d_kps_to_aa: body point k = (kps[A[k]] + kps[B[k]]) / 2 of the 133 COCO-WholeBody points
WHOLEBODY_A = (0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20)
WHOLEBODY_B = (0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21)
WHOLEBODY_LHAND = slice(91, 112)
WHOLEBODY_RHAND = slice(112, 133)

# body points that are SMPL joint centres (SMPL 22 indices)
BODY_JOINT = {"RShoulder": 17, "RElbow": 19, "RWrist": 21, "LShoulder": 16, "LElbow": 18, "LWrist": 20,
              "RHip": 2, "RKnee": 5, "RAnkle": 8, "LHip": 1, "LKnee": 4, "LAnkle": 7}
# body points that are mesh vertices (the mean of the vertices listed)
BODY_VERTICES = {"Nose": (332,), "REye": (6260,), "LEye": (2800,), "REar": (4071,), "LEar": (583,),
                 "LToe": (3216, 3226), "RToe": (6617, 6624)}
# SMPL+H hand joints: left 22-36, right 37-51, in the order index, middle, little (pinky), ring, thumb, three
# joints each from the base; the COCO-WholeBody hand order is wrist, thumb, index, middle, ring, little
_FINGER_OFFSET = {"Thumb": 12, "Index": 0, "Middle": 3, "Ring": 9, "Little": 6}
_HAND_BASE = {"L": 22, "R": 37}
_WRIST = {"L": 20, "R": 21}
TIP_VERTEX = {"L": {"Thumb": 2746, "Index": 2319, "Middle": 2445, "Ring": 2556, "Little": 2673},
              "R": {"Thumb": 6191, "Index": 5782, "Middle": 5905, "Ring": 6016, "Little": 6133}}
N_JOINTS52 = 52


def hand_sources(side):
    """[(kind, index)] of the 21 hand points of side 'L' or 'R': ('joint', SMPL+H joint 0-51) or ('vertex',
    vertex id)."""
    out = [("joint", _WRIST[side])]
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
        base = _HAND_BASE[side] + _FINGER_OFFSET[finger]
        out += [("joint", base), ("joint", base + 1), ("joint", base + 2), ("vertex", TIP_VERTEX[side][finger])]
    return out


def vertex_ids():
    """Sorted vertex ids the drawn set reads."""
    ids = {v for vs in BODY_VERTICES.values() for v in vs}
    ids |= {i for s in ("L", "R") for k, i in hand_sources(s) if k == "vertex"}
    return tuple(sorted(ids))


def wholebody133_to_drawn(kps, scores=None):
    """133 COCO-WholeBody keypoints (..., 133, 2) px and scores (..., 133) or None -> the drawn set (..., 62,
    2) and scores (..., 62), by the rule of Wan2.2 split_pose2d_kps_to_aa. A body point built from two
    keypoints takes the smaller of their two scores (a rule of this module: the scores are not part of the
    truth)."""
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape[-2:] != (133, 2):
        raise ValueError(f"keypoints must be (..., 133, 2), got {kps.shape}")
    a, b = list(WHOLEBODY_A), list(WHOLEBODY_B)
    body = (kps[..., a, :] + kps[..., b, :]) / 2
    out = np.concatenate([body, kps[..., WHOLEBODY_LHAND, :], kps[..., WHOLEBODY_RHAND, :]], -2)
    if scores is None:
        return out, np.ones(out.shape[:-1])
    s = np.asarray(scores, dtype=np.float64)
    sb = np.minimum(s[..., a], s[..., b])
    return out, np.concatenate([sb, s[..., WHOLEBODY_LHAND], s[..., WHOLEBODY_RHAND]], -1)


class SMPLHPoints:
    """The drawn set of one SMPL+H gender: subset skinning of the vertices it reads and the hand joints, all
    in float64 on the CPU. The model arrays stay in memory (MPI licence, mdfm/body/smplh.py)."""

    def __init__(self, gender="neutral"):
        from ..body.smplh import SMPLHBody      # the SMPL+H drawn set only: no SMPL+H module at the top
        body = SMPLHBody(gender, dtype=torch.float64)
        self.body = body
        self.gender = gender
        self.vids = vertex_ids()
        self._col = {v: k for k, v in enumerate(self.vids)}
        sel = torch.tensor(self.vids, dtype=torch.int64)
        self.parents = [int(p) for p in body.parents.tolist()]
        self.parents[0] = -1
        self._v_template = body.v_template[sel]
        self._shapedirs = body.shapedirs[sel]
        self._posedirs = body.posedirs[sel][..., :(N_BODY - 1) * 9]      # hands at rest add nothing
        self._posedirs_hands = body.posedirs[sel]                          # (S, 3, 459): body then hand joints
        self._weights = body.weights[sel]                                  # (S, 52)

    def rest(self, betas):
        """Rest joints (52, 3) by the vertex path and rest vertices (S, 3) of the drawn set, for betas (16,)."""
        betas = torch.as_tensor(betas, dtype=torch.float64).reshape(-1)
        v = self.body.v_template + torch.einsum("vcs,s->vc", self.body.shapedirs, betas)
        J = self.body.J_regressor @ v
        return J, self._v_template + torch.einsum("vcs,s->vc", self._shapedirs, betas)

    def rest_joints22(self, betas):
        """Rest joints (22, 3) of the joint basis, as KnotBatch takes them."""
        return self.body.rest_joints22(torch.as_tensor(betas, dtype=torch.float64).reshape(-1))

    def posed(self, p22, R22, betas, hand_R=None):
        """Joint positions p22 (N, 22, 3) m and global rotations R22 (N, 22, 3, 3) of the body (KnotBatch.pose)
        -> the drawn set (N, 62, 3) m, the 52 joint positions (N, 52, 3) and the vertex points (N, S, 3).
        hand_R (N, 30, 3, 3): local rotations of the hand joints 22 to 51 (module docstring); None, the hands at
        rest (the path of every driver clip before task VCH, unchanged)."""
        p22 = torch.as_tensor(p22, dtype=torch.float64)
        R22 = torch.as_tensor(R22, dtype=torch.float64)
        N = p22.shape[0]
        J, vS = self.rest(betas)
        par = self.parents
        eye = torch.eye(3, dtype=torch.float64)
        Rl = torch.empty_like(R22)
        Rl[:, 0] = R22[:, 0]
        for i in range(1, N_BODY):
            Rl[:, i] = R22[:, par[i]].transpose(-1, -2) @ R22[:, i]
        feat = (Rl[:, 1:] - eye).reshape(N, -1)                           # (N, 189)
        Rg = [R22[:, i] for i in range(N_BODY)]
        pg = [p22[:, i] for i in range(N_BODY)]
        if hand_R is None:
            for j in range(N_BODY, N_JOINTS52):                            # hands: local rotation identity
                p = par[j]
                Rg.append(Rg[p])
                pg.append(pg[p] + Rg[p] @ (J[j] - J[p]))
            posedirs = self._posedirs
        else:
            hand_R = torch.as_tensor(hand_R, dtype=torch.float64)
            if tuple(hand_R.shape) != (N, N_JOINTS52 - N_BODY, 3, 3):
                raise ValueError(f"hand_R must be ({N}, 30, 3, 3), got {tuple(hand_R.shape)}")
            for j in range(N_BODY, N_JOINTS52):                            # hands: their local rotations
                p = par[j]
                Rg.append(Rg[p] @ hand_R[:, j - N_BODY])
                pg.append(pg[p] + Rg[p] @ (J[j] - J[p]))
            feat = torch.cat([feat, (hand_R - eye).reshape(N, -1)], 1)    # (N, 459), SMPLHBody's order
            posedirs = self._posedirs_hands
        Rg = torch.stack(Rg, 1)                                            # (N, 52, 3, 3)
        pg = torch.stack(pg, 1)                                            # (N, 52, 3)
        v = vS[None] + torch.einsum("scp,np->nsc", posedirs, feat)        # (N, S, 3) posed rest vertices
        local = v[:, :, None, :] - J[None, None]                           # (N, S, 52, 3)
        moved = torch.einsum("njab,nsjb->nsja", Rg, local) + pg[:, None]   # (N, S, 52, 3)
        x = (self._weights[None, :, :, None] * moved).sum(2)               # (N, S, 3)
        out = torch.empty(N, N_POINTS, 3, dtype=torch.float64)
        for name, j in BODY_JOINT.items():
            out[:, IDX[name]] = p22[:, j]
        for name, vs in BODY_VERTICES.items():
            out[:, IDX[name]] = torch.stack([x[:, self._col[i]] for i in vs], 0).mean(0)
        out[:, IDX["Neck"]] = (out[:, IDX["RShoulder"]] + out[:, IDX["LShoulder"]]) / 2
        for side, sl in (("L", LHAND), ("R", RHAND)):
            for k, (kind, i) in enumerate(hand_sources(side)):
                out[:, sl.start + k] = pg[:, i] if kind == "joint" else x[:, self._col[i]]
        return out, pg, x


@functools.lru_cache(maxsize=3)
def points_model(gender):
    """SMPLHPoints of a gender ('female', 'male' or 'neutral'), built once per process."""
    return SMPLHPoints(gender)


def body_pose(rot6d, transl, rest22, t, rate=100.0):
    """KnotBatch.pose of one body at times t (N,) s: joint positions (N, 22, 3) and global rotations (N, 22, 3,
    3), float64 on the CPU. rot6d (K, 22, 6), transl (K, 3) m on the clock k / rate."""
    rot6d = torch.as_tensor(rot6d, dtype=torch.float64)
    transl = torch.as_tensor(transl, dtype=torch.float64)
    t = torch.as_tensor(t, dtype=torch.float64)
    kb = KnotBatch(rot6d[None], transl[None], torch.as_tensor(rest22, dtype=torch.float64)[None], rate=rate)
    with torch.no_grad():
        return kb.pose(torch.zeros(t.shape[0], dtype=torch.int64), t)


def hand_rotations(hand6d, t, rate=100.0):
    """(N, 30, 3, 3) local hand rotations at times t (N,) s of hand knots hand6d (K, 30, 6) on the clock k / rate:
    the Hermite interpolant of the rot6d, orthonormalized (mdfm.synth.viewer.hands_at)."""
    from ..body.hermite import hermite
    from ..body.rotations import rot6d_to_matrix
    with torch.no_grad():
        return rot6d_to_matrix(hermite(torch.as_tensor(hand6d, dtype=torch.float64),
                                       torch.as_tensor(t, dtype=torch.float64), rate))


def drawn_points(rot6d, transl, betas, gender, t, rate=100.0, hand6d=None):
    """The drawn set (N, 62, 3) m of a body (knots rot6d (K, 22, 6), transl (K, 3), betas (16,), gender) at
    times t (N,) s, float64; also the SMPL joint positions (N, 22, 3). hand6d (K, 30, 6): hand knots on the same
    clock (the opt-in path, module docstring); None, the hands at rest."""
    model = points_model(gender)
    with torch.no_grad():
        p22, R22 = body_pose(rot6d, transl, model.rest_joints22(betas), t, rate)
        hand_R = None if hand6d is None else hand_rotations(hand6d, t, rate)
        pts, _, _ = model.posed(p22, R22, betas, hand_R)
    return pts, p22
