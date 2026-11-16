"""Rotation representations (torch, batched over leading dimensions, differentiable).

6D form (Zhou et al. 2019, CVPR): the first two columns of the rotation matrix, turned back
into a rotation by Gram-Schmidt. Axis-angle by the Rodrigues formula, stable near zero. Yaw is
the rotation about the world up axis +y (MD_FM world, y up).
"""
import torch

EPS = 1e-8
# floor on the squared column norm of the Gram-Schmidt step (a norm floor of 1e-6); a unit
# column is shortened by 5e-13, below float64 rounding of the tests and far below float32
GS_EPS = 1e-12


def skew(v):
    """(..., 3) -> (..., 3, 3) with skew(v) @ w = v x w."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def vee(m):
    """(..., 3, 3) skew-symmetric part -> (..., 3); inverse of skew on skew matrices."""
    return 0.5 * torch.stack([m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0],
                              m[..., 1, 0] - m[..., 0, 1]], -1)


def axis_angle_to_matrix(aa):
    """(..., 3) axis times angle -> (..., 3, 3), exact and smooth at zero (Taylor terms)."""
    theta2 = (aa * aa).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + EPS * EPS)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + EPS * EPS))
    k = skew(aa)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def matrix_to_axis_angle(r):
    """(..., 3, 3) -> (..., 3), angle in [0, pi]; used in tests and for export only."""
    cos = ((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]) - 1) / 2
    theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    v = vee(r)
    s = torch.sin(theta)
    scale = torch.where(s.abs() < 1e-6, torch.ones_like(s), theta / s)
    return v * scale[..., None]


def rot6d_to_matrix(x):
    """(..., 6) -> (..., 3, 3): columns b1, b2, b3 by Gram-Schmidt on the two 3-vectors.

    Each column is normalized as a * rsqrt(sum(a * a) + GS_EPS). The form a / (|a| + eps)
    breaks backward through two nested torch.func.jvp calls (an in-place error on the norm
    output, torch 2.11); this one passes gradcheck there in float64 (tests/test_skeleton.py)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 * torch.rsqrt((a1 * a1).sum(-1, keepdim=True) + GS_EPS)
    a2p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2p * torch.rsqrt((a2p * a2p).sum(-1, keepdim=True) + GS_EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


def matrix_to_rot6d(r):
    """(..., 3, 3) -> (..., 6): the first two columns."""
    return torch.cat([r[..., :, 0], r[..., :, 1]], -1)


def rot_y(theta):
    """theta (...) rad -> (..., 3, 3), the rotation about +y by theta (right-handed: +z turns
    toward +x)."""
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([c, z, s], -1),
                        torch.stack([z, o, z], -1),
                        torch.stack([-s, z, c], -1)], -2)


def yaw_angle(R):
    """(..., 3, 3) -> (...) rad in (-pi, pi]: the heading of R's +z axis projected on the floor
    (x, z), measured about +y from +z, so that yaw_angle(rot_y(theta)) = theta. Undefined when
    R's +z axis is vertical."""
    return torch.atan2(R[..., 0, 2], R[..., 2, 2])


def geodesic(R1, R2):
    """(..., 3, 3), (..., 3, 3) -> (...) rad in [0, pi]: the angle of R1^T R2, by atan2 of the
    sine (norm of the skew part) and the cosine (from the trace), accurate near 0 and pi; the
    gradient of its square is zero at R1 = R2."""
    M = R1.transpose(-1, -2) @ R2
    c = (M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2] - 1) / 2
    return torch.atan2(vee(M).norm(dim=-1), c)


# AMASS and SMPL+H use a z-up world; MoveAtlas canonical 3D and MD_FM use y-up:
# (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
