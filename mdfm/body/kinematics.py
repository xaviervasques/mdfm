"""Batched body kinematics: every query point of a micro-step from one second-order forward pass.

KnotBatch holds the knots of W windows on the body clock (rot6d (W, T, 22, 6), transl (W, T,
3) m, y up, time 0 at each window's start) and the rest joints of each window's subject.
kinematics(w, t) evaluates the 22 joints at flat queries (window w, time t in s) with one
evaluation of Hermite, Gram-Schmidt and forward kinematics under two nested forward-mode jvp
(mdfm.body.trajectory.second_order): positions, velocities and accelerations, global rotations,
world angular velocities and accelerations of all joints at every query at once. point_kin()
carries them to points rigidly attached to a segment (sensor sites, markers) by rigid-body
transport, outside the transform: r = R offset, v = v_k + omega x r, a = a_k + alpha x r + omega
x (omega x r).

A query point is one (window, time) pair; the 22 joints are evaluated at each. Knots are read
through the Hermite supports only (knots i - 1 to i + 2 of the query's window,
mdfm.body.hermite), so the gradient reaches the decoded knots and the rest joints (hence betas)
and nothing else. Forward kinematics runs by tree depth, 7 levels below the root instead of 21
joints (fk_levels, equal to joints.fk). Products of 3 x 3 blocks are broadcast sums, not batched
GEMM, and the angular rates come from column cross products, omega = 1/2 sum_k R_k x dR_k and
alpha = 1/2 sum_k R_k x ddR_k (the vee of dR R^T and of its time derivative).

Precision: float32 or float64 only. Half-precision input or an active autocast on the tensors'
device raises, since the second derivative amplifies knot rounding by rate^2 (1e4 at 100 Hz);
callers under bf16 autocast wrap the call in torch.autocast(device, enabled=False).
"""
from typing import NamedTuple

import torch

from .hermite import (basis_matrix, check_queries, combine, gather_knots, knots_per_query, position,
                      segment, support_of, weights)
from .rotations import rot6d_to_matrix
from .joints import N_BODY, PARENTS
from .trajectory import second_order

FLOAT_TYPES = (torch.float32, torch.float64)


class BodyKin(NamedTuple):
    """Joint kinematics at N query points, world frame (y up): p (N, 22, 3) m, v (N, 22, 3) m/s,
    a (N, 22, 3) m/s^2, R (N, 22, 3, 3) global rotations, omega (N, 22, 3) rad/s, alpha (N, 22,
    3) rad/s^2."""
    p: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    R: torch.Tensor
    omega: torch.Tensor
    alpha: torch.Tensor


class PointKin(NamedTuple):
    """Kinematics of S attached points at N query points: p, v, a (N, S, 3) in m, m/s, m/s^2;
    R (N, S, 3, 3) the rotation of each point's segment; omega, alpha (N, S, 3) in rad/s and
    rad/s^2, world frame."""
    p: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    R: torch.Tensor
    omega: torch.Tensor
    alpha: torch.Tensor


def check_precision(*tensors):
    """TypeError on a tensor that is not float32 or float64 (half precision included);
    RuntimeError when autocast is active on a tensor's device type."""
    for x in tensors:
        if x.dtype not in FLOAT_TYPES:
            raise TypeError(f"body kinematics run in float32 or float64, got {x.dtype}: knot rounding is "
                            "amplified by rate^2 in accelerations")
        if torch.is_autocast_enabled(x.device.type):
            raise RuntimeError(f"autocast is active on {x.device.type}: call the body kinematics inside "
                               f"torch.autocast('{x.device.type}', enabled=False)")


def _levels(parents):
    """Forward-kinematics plan by tree depth: per level below the root, the joint range (a, b)
    and the parent selector in the previous level: ('slice', lo, hi) or ('index', ids), a
    single parent broadcasting as a slice of length 1."""
    depth = [0] * len(parents)
    for i in range(1, len(parents)):
        depth[i] = depth[parents[i]] + 1
    levels = [[j for j in range(len(parents)) if depth[j] == d] for d in range(max(depth) + 1)]
    if [j for lv in levels for j in lv] != list(range(len(parents))):
        raise ValueError("joints must be numbered level by level (true for SMPL+H)")
    plan = []
    for prev, cur in zip(levels, levels[1:]):
        local = [prev.index(parents[j]) for j in cur]
        if len(set(local)) == 1:
            sel = ("slice", local[0], local[0] + 1)
        elif local == list(range(local[0], local[0] + len(local))):
            sel = ("slice", local[0], local[0] + len(local))
        else:
            sel = ("index", tuple(local))
        plan.append(((cur[0], cur[-1] + 1), sel))
    return tuple(plan)


LEVELS = _levels(PARENTS[:N_BODY])


def _mm(A, B):
    """(..., i, k) @ (..., k, j) as a broadcast sum, for 3 x 3 blocks."""
    return (A[..., :, :, None] * B[..., None, :, :]).sum(-2)


def fk_levels(R_local, b):
    """Forward kinematics by tree depth. R_local (N, 22, 3, 3) local rotations with row 0 the
    root orientation; b (N, 22, 3) m: row 0 the root joint position (rest pelvis plus
    translation), rows 1 to 21 the rest bone vectors (joint minus parent). Returns positions
    (N, 22, 3) m and global rotations (N, 22, 3, 3), as smplh.fk."""
    top = torch.cat([R_local, b[..., None]], -1)          # (N, 22, 3, 4): [R_local | bone]
    Rs, ps = [top[:, 0:1, :, :3]], [top[:, 0:1, :, 3]]
    for (a, e), sel in LEVELS:
        if sel[0] == "slice":
            Rp, pp = Rs[-1][:, sel[1]:sel[2]], ps[-1][:, sel[1]:sel[2]]
        else:
            Rp, pp = Rs[-1][:, list(sel[1])], ps[-1][:, list(sel[1])]
        out = _mm(Rp, top[:, a:e])                         # [R_parent R_local | R_parent bone]
        Rs.append(out[..., :3])
        ps.append(pp + out[..., 3])
    return torch.cat(ps, 1), torch.cat(Rs, 1)


def angular_rates(R, dR, ddR):
    """World angular velocity and acceleration (..., 3) from a rotation (..., 3, 3) and its
    first and second time derivatives: omega = vee(dR R^T) = 1/2 sum_k R_k x dR_k over the
    columns k, alpha = vee(ddR R^T + dR dR^T) = 1/2 sum_k R_k x ddR_k."""
    return (0.5 * torch.cross(R, dR, dim=-2).sum(-1), 0.5 * torch.cross(R, ddR, dim=-2).sum(-1))


class KnotBatch:
    """Knots of W windows and their subjects' rest joints, evaluated at flat queries."""

    def __init__(self, rot6d, transl, rest, rate=100.0, lengths=None):
        """rot6d (W, T, 22, 6), transl (W, T, 3) m on the clock k / rate from each window's
        start, one dtype, float32 or float64; rest (W, 22, 3) m the rest joints of each window's
        subject (SkeletonBank.rest(betas, gender); the 22 parents are the same for the three
        genders), cast to that dtype; lengths (W,) int64 valid knots per window in a padded
        buffer (default T). ValueError on shapes, devices or lengths; TypeError on half
        precision."""
        if rot6d.dim() != 4 or tuple(rot6d.shape[2:]) != (N_BODY, 6):
            raise ValueError(f"rot6d must be (W, T, 22, 6), got {tuple(rot6d.shape)}")
        W, T = rot6d.shape[:2]
        if tuple(transl.shape) != (W, T, 3):
            raise ValueError(f"transl must be ({W}, {T}, 3), got {tuple(transl.shape)}")
        if tuple(rest.shape) != (W, N_BODY, 3):
            raise ValueError(f"rest must be ({W}, 22, 3), one row per window, got {tuple(rest.shape)}")
        check_precision(rot6d, transl, rest)
        if transl.dtype != rot6d.dtype:
            raise TypeError(f"rot6d and transl must share a dtype, got {rot6d.dtype} and {transl.dtype}")
        if transl.device != rot6d.device or rest.device != rot6d.device:
            raise ValueError("rot6d, transl and rest must be on one device")
        empty = torch.empty(0, dtype=torch.int64, device=rot6d.device)
        self.lengths = check_queries(W, T, empty, empty.to(rot6d.dtype), lengths)
        self.rot6d, self.transl, self.rest = rot6d, transl, rest.to(rot6d.dtype)
        self.rate = float(rate)
        self.W, self.T = W, T

    @property
    def dtype(self):
        return self.rot6d.dtype

    @property
    def device(self):
        return self.rot6d.device

    def _pose_fn(self, w, t):
        """The pose as a function of the query times, f(tt) -> positions (N, 22, 3) and global
        rotations (N, 22, 3, 3), with the support knots and rest bones gathered once outside
        it; and t in the knots' dtype."""
        check_queries(self.W, self.T, w, t)
        if w.device != self.device or t.device != self.device:
            raise ValueError(f"queries must be on {self.device}")
        check_precision(t, self.rot6d, self.transl, self.rest)
        t = t.to(self.dtype)
        rate = self.rate
        L = knots_per_query(self.T, w, self.lengths)
        if torch.is_tensor(L):
            L = L.to(self.device)
        i = segment(position(t, rate, L), L)
        k = support_of(i, L)
        B = basis_matrix(k, self.dtype)
        K6 = gather_knots(self.rot6d, w, k)                    # (N, 4, 22, 6)
        K3 = gather_knots(self.transl, w, k)                   # (N, 4, 3)
        # per window: rest root joint in row 0, rest bone vectors in rows 1 to 21; then per query
        rest = self.rest
        b_w = torch.cat([rest[:, :1], rest[:, 1:] - rest[:, list(PARENTS[1:N_BODY])]], 1)
        b_n = b_w.index_select(0, w)                           # (N, 22, 3)

        def f(tt):
            c = weights(position(tt, rate, L), i, B)
            tr = combine(c, K3)
            b = torch.cat([b_n[:, :1] + tr[:, None], b_n[:, 1:]], 1)
            return fk_levels(rot6d_to_matrix(combine(c, K6)), b)

        return f, t

    def kinematics(self, w, t):
        """BodyKin of the 22 joints at queries w (N,) int64 windows, t (N,) s from each window's
        start, from one evaluation under two nested forward-mode jvp. Queries outside a window's
        span are clamped to it: the end knot's pose, with the end segment's derivatives on the
        end knot and within 1e-3 knot of it, zero derivatives farther out
        (mdfm.body.hermite.position)."""
        f, t = self._pose_fn(w, t)
        (p, R), (v, dR), (a, ddR) = second_order(f, t)
        omega, alpha = angular_rates(R, dR, ddR)
        return BodyKin(p, v, a, R, omega, alpha)

    def pose(self, w, t):
        """Joint positions (N, 22, 3) m and global rotations (N, 22, 3, 3) at the queries,
        primal only (no jvp), for direct supervision."""
        f, t = self._pose_fn(w, t)
        return f(t)


def point_kin(kin, joint, offset):
    """PointKin of S points rigidly attached to segments: joint (S,) int64 the segment joint of
    each point, offset (S, 3) or (N, S, 3) m in that segment's frame (the offset of
    BodyTrajectory.point_at). Rigid-body transport of the joint kinematics kin (BodyKin):
    r = R offset, p = p_k + r, v = v_k + omega x r, a = a_k + alpha x r + omega x (omega x r);
    R, omega and alpha are the segment's. ValueError on shapes or on joint ids outside [0, 22);
    the ids of a GPU joint are not read back (no device synchronization), an id outside [0, 22)
    stops the gather with a device-side assertion. TypeError on half precision."""
    check_precision(offset, kin.p)
    joint = torch.as_tensor(joint, dtype=torch.int64)
    if joint.device.type == "cpu" and joint.numel() and bool(((joint < 0) | (joint >= N_BODY)).any()):
        raise ValueError(f"joint ids must lie in [0, {N_BODY}), got {joint.tolist()}")
    # a negative id would index from the end (joint -1 read as joint 21): sent out of range instead
    joint = torch.where(joint < 0, N_BODY, joint).to(kin.p.device)
    if joint.dim() != 1 or offset.dim() not in (2, 3) or tuple(offset.shape[-2:]) != (joint.shape[0], 3) \
            or (offset.dim() == 3 and offset.shape[0] != kin.p.shape[0]):
        raise ValueError(f"joint (S,) and offset (S, 3) or (N, S, 3), got {tuple(joint.shape)} and "
                         f"{tuple(offset.shape)} for N = {kin.p.shape[0]}")
    offset = offset.to(kin.p.dtype)
    R = kin.R[:, joint]
    omega, alpha = kin.omega[:, joint], kin.alpha[:, joint]
    r = (R * offset[..., None, :]).sum(-1)
    wxr = torch.cross(omega, r, dim=-1)
    p = kin.p[:, joint] + r
    v = kin.v[:, joint] + wxr
    a = kin.a[:, joint] + torch.cross(alpha, r, dim=-1) + torch.cross(omega, wxr, dim=-1)
    return PointKin(p, v, a, R, omega, alpha)
