"""A body trajectory q(t) on the 100 Hz clock, with exact time derivatives at any timestamp.

q(t) holds the root translation and the 22 joint rotations in 6D form (design section 1).
Rotations are interpolated in 6D and projected back by Gram-Schmidt, translations by cubic
Hermite. Joint and sensor-point positions, velocities and accelerations, global rotations,
angular velocities and angular accelerations come from forward-mode differentiation of the
forward kinematics with respect to time (torch.func.jvp), so every kernel sees the
derivatives of the same interpolant. The subject's rest joints are computed once from its
betas through the joint-space basis and shared by every timestamp and every kernel.
"""
import torch
from torch.func import jvp

from .hermite import hermite
from .rotations import rot6d_to_matrix, vee
from .joints import N_BODY, fk


def second_order(f, t):
    """Values, first and second time derivatives of f(t) (a tensor or a tuple of tensors) at
    times t (N,), from one evaluation of f under two nested forward-mode jvp."""
    ones = torch.ones_like(t)

    def df(tt):
        return jvp(f, (tt,), (ones,))

    (y, dy), (_, d2y) = jvp(df, (t,), (ones,))
    return y, dy, d2y


class BodyTrajectory:
    def __init__(self, body, rot6d, transl, betas, rate=100.0, t0=0.0, rest=None):
        """body: SMPLHBody; rot6d (T, 22, 6); transl (T, 3) m; betas (16,); rest (22, 3) m,
        the subject's rest joints when already computed (SkeletonBank.rest; cast to the dtype
        of rot6d, betas is then not read), else derived from betas through the body's
        joint-space basis. Other shapes raise ValueError."""
        self.body, self.rot6d, self.transl, self.betas = body, rot6d, transl, betas
        self.rate, self.t0 = rate, t0
        # one subject: a (1, 22, 3) bank row or (1, 16) betas would broadcast through fk and
        # break the per-subject site table
        if rest is not None:
            if tuple(rest.shape) != (N_BODY, 3):
                raise ValueError(f"rest joints must be (22, 3) for one subject, got {tuple(rest.shape)}")
            rest = rest.to(rot6d.dtype)
        else:
            from .smplh import N_BETAS     # the SMPL+H betas branch only (open-body port wave 8: no import at the top)
            if tuple(betas.shape) != (N_BETAS,):
                raise ValueError(f"betas must be (16,) for one subject, got {tuple(betas.shape)}")
            if not betas.requires_grad:
                # here, outside any jvp: a tensor cached from inside a transform would escape it
                rest = self._rest_from_betas()
        self._rest = rest

    def _rest_from_betas(self):
        b = self.betas.to(self.body.J_template.dtype)
        return self.body.rest_joints22(b).to(self.rot6d.dtype)

    @property
    def rest(self):
        """Rest joints (22, 3) m of the subject in the dtype of rot6d: given, or computed once
        at construction when betas carries no gradient; while it does, recomputed per call
        (one row, 1,056 multiply-adds) so that each backward pass owns its graph."""
        return self._rest if self._rest is not None else self._rest_from_betas()

    def pose_at(self, t):
        """t (N,) -> joint positions (N, 22, 3), global rotations (N, 22, 3, 3)."""
        r6 = hermite(self.rot6d, t, self.rate, self.t0)
        tr = hermite(self.transl, t, self.rate, self.t0)
        return fk(rot6d_to_matrix(r6), tr, self.rest)

    def point_at(self, t, joint, offset):
        """Position (N, 3) and rotation (N, 3, 3) of a point rigidly attached to a joint's
        segment, at offset (3,) in that segment's frame."""
        pos, rot = self.pose_at(t)
        return pos[:, joint] + (rot[:, joint] @ offset[:, None])[..., 0], rot[:, joint]

    def point_kinematics(self, t, joint, offset):
        """Position, velocity, acceleration (N, 3), rotation (N, 3, 3), world angular velocity
        and world angular acceleration (N, 3) of the attached point, at times t (N,)."""
        (p, R), (v, Rdot), (a, Rddot) = second_order(lambda tt: self.point_at(tt, joint, offset), t)
        # world angular velocity: skew(w) = Rdot R^T; angular acceleration from its derivative
        w = vee(Rdot @ R.transpose(-1, -2))
        wdot = vee(Rddot @ R.transpose(-1, -2) + Rdot @ Rdot.transpose(-1, -2))
        return p, v, a, R, w, wdot
