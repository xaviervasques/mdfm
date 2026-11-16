"""Inverse kinematics of a leg, shared by the procedural walk (base.walk) and the trembling of freezing
episodes (freezing.py).

- rmin: the minimal rotation taking one unit vector onto another.
- two_bone: the knee position K for a hip H and an ankle A, with thigh and shank lengths L1 and L2, on the
  circle of points at L1 from H and L2 from A, the one nearest a pole point P (the knee bends toward P).
- hinge_leg: the thigh's global rotation and a pure knee hinge angle (a rotation about the knee's x axis, the
  knee flexion axis of the rest skeleton) that put the ankle at A, the knee turned toward a pole point. With
  the rest bones p (hip to knee) and q (knee to ankle), the hip-to-ankle distance |p + R_x(k) q| fixes the
  hinge angle k in closed form; the thigh is then aligned on A - H and twisted about that axis so that the
  knee turns toward the pole, or so that the hinge axis lies along a given direction.
The ankle must lie closer to the hip than the leg reaches (ValueError otherwise).
"""
import torch

from ..body.rotations import axis_angle_to_matrix


def rmin(a, b):
    """(..., 3, 3) minimal rotations taking unit vectors a (..., 3) onto unit vectors b (..., 3)."""
    v = torch.cross(a, b, dim=-1)
    s = v.norm(dim=-1, keepdim=True)
    c = (a * b).sum(-1, keepdim=True)
    ang = torch.atan2(s, c)
    aa = torch.where(s > 1e-15, v / s.clamp_min(1e-300) * ang, torch.zeros_like(v))
    return axis_angle_to_matrix(aa)


def two_bone(H, A, P, L1, L2, slack=1e-9):
    """Knee positions (N, 3) for hips H (N, 3), ankles A (N, 3), pole points P (N, 3) (module docstring)."""
    d = A - H
    dist = d.norm(dim=-1, keepdim=True)
    if bool((dist >= L1 + L2 - slack).any()):
        raise ValueError(f"an ankle lies {float(dist.max()):.4f} m from its hip, the leg reaches {L1 + L2:.4f} m")
    u = d / dist
    along = (L1 ** 2 - L2 ** 2 + dist ** 2) / (2 * dist)
    h = torch.sqrt((L1 ** 2 - along ** 2).clamp_min(0.0))
    w = P - H
    w = w - (w * u).sum(-1, keepdim=True) * u
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return H + along * u + h * w


def hinge_leg(H, A, P, p, q, G_ref, hinge=None):
    """(G_thigh (N, 3, 3), knee angle (N,) rad) for hips H (N, 3), ankles A (N, 3), pole points P (N, 3), rest
    bones p (hip to knee) and q (knee to ankle) (3,) and reference frames G_ref (N, 3, 3) (the pelvis's, from
    which the thigh is aligned): the knee's local rotation is R_x(angle) (module docstring). hinge (N, 3):
    a direction for the knee hinge axis (the thigh's x axis) instead of the pole, well conditioned when the
    leg is nearly straight."""
    d = (A - H).norm(dim=-1)
    a = p[1] * q[1] + p[2] * q[2]
    b = p[2] * q[1] - p[1] * q[2]
    rho = torch.sqrt(a * a + b * b)
    delta = torch.atan2(b, a)
    base = (p * p).sum() + (q * q).sum() + 2 * p[0] * q[0]
    x = (d * d - base) / (2 * rho)
    if bool((x > 1 + 1e-12).any()):
        raise ValueError(f"an ankle lies {float(d.max()):.4f} m from its hip, beyond the leg's reach")
    k = delta + torch.acos(x.clamp(-1.0, 1.0))
    c, s = torch.cos(k), torch.sin(k)
    qk = torch.stack([q[0].expand_as(k), q[1] * c - q[2] * s, q[1] * s + q[2] * c], -1)    # R_x(k) q
    v = p + qk                                                    # hip to ankle, thigh frame
    u = (A - H) / d[:, None]
    G1 = rmin(torch.einsum("nij,nj->ni", G_ref, v) / v.norm(dim=-1, keepdim=True), u) @ G_ref
    if hinge is not None:                                # twist the knee hinge axis (thigh x) onto hinge, projected
        kp = G1[..., :, 0]
        pp = hinge
    else:                                                # twist the knee toward the pole point
        kp = torch.einsum("nij,j->ni", G1, p)
        pp = P - H
    kp = kp - (kp * u).sum(-1, keepdim=True) * u
    pp = pp - (pp * u).sum(-1, keepdim=True) * u
    kp = kp / kp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pp = pp / pp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gamma = torch.atan2((torch.cross(kp, pp, dim=-1) * u).sum(-1), (kp * pp).sum(-1))
    G = axis_angle_to_matrix(u * gamma[:, None]) @ G1
    return G, k
