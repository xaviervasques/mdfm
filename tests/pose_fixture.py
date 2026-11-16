"""Synthetic canonical-15 figures for the pose kernel and camera render tests (no MoveAtlas file).

stick(): a standing stick figure (camera.CANONICAL_15 order, m, window frame: y up, floor y = 0, the
subject facing +z, x toward the subject's left) whose arms and legs swing about the shoulders and hips
at f_hz, in the sagittal plane (about x) or the frontal plane (about z, mirrored left and right),
optionally walking along +z.
"""
import math

import torch

BASE = [[0.0, 0.95, 0.0], [0.0, 1.45, 0.0], [0.0, 1.65, 0.0], [0.18, 1.42, 0.0], [-0.18, 1.42, 0.0],
        [0.2, 1.15, 0.0], [-0.2, 1.15, 0.0], [0.2, 0.9, 0.0], [-0.2, 0.9, 0.0], [0.09, 0.9, 0.0],
        [-0.09, 0.9, 0.0], [0.1, 0.5, 0.0], [-0.1, 0.5, 0.0], [0.1, 0.08, 0.0], [-0.1, 0.08, 0.0]]
# (proximal joint, joints moved with it, phase)
CHAINS = [(3, (5, 7), 0.0), (4, (6, 8), math.pi), (9, (11, 13), math.pi), (10, (12, 14), 0.0)]
MOVING = (5, 6, 7, 8, 11, 12, 13, 14)


def stick(T, fps, f_hz, amp=0.5, plane="sagittal", speed=0.0, dtype=torch.float64):
    """(T, 15, 3) m at t = k / fps: limbs swing by amp sin(2 pi f_hz t + phase) rad; the whole figure
    moves along +z at speed m/s."""
    base = torch.tensor(BASE, dtype=dtype)
    t = torch.arange(T, dtype=dtype) / fps
    pos = base.expand(T, 15, 3).clone()
    for prox, joints, ph in CHAINS:
        a = amp * torch.sin(2 * math.pi * f_hz * t + ph)
        for j in joints:
            d = base[j] - base[prox]
            if plane == "sagittal":        # about x: y' = c y - s z, z' = s y + c z
                c, s = torch.cos(a), torch.sin(a)
                pos[:, j, 1] = base[prox, 1] + c * d[1] - s * d[2]
                pos[:, j, 2] = base[prox, 2] + s * d[1] + c * d[2]
            elif plane == "frontal":       # about z by +a on the left, -a on the right
                phi = a if float(base[prox, 0]) > 0 else -a
                c, s = torch.cos(phi), torch.sin(phi)
                pos[:, j, 0] = base[prox, 0] + c * d[0] - s * d[1]
                pos[:, j, 1] = base[prox, 1] + s * d[0] + c * d[1]
            else:
                raise ValueError(f"unknown plane {plane!r}")
    pos[..., 2] = pos[..., 2] + speed * t[:, None]
    return pos
