"""Virtual IMUs in the canonical layout of MoveAtlas (configs/canonical_imu.yaml): 14 sites,
9 channels (acc in g, gyro in deg/s, mag in uT, masked here), at a device rate.

Each site is attached to an SMPL segment with an offset expressed in that segment's rest
frame (SMPL rest pose: y up, z forward, x toward the subject's left), computed from the
subject's shaped rest joints, and a mounting rotation (identity by default: sensor axes
along the segment's rest axes). Placement and mounting are nuisances with priors (design
section 3); the placement defaults here are anatomical guesses, the mounting priors are
measured on weargait_pd (mdfm.kernels.nuisance.MountPriors, reports/mount_priors/). A device
model adds a constant bias, white noise and saturation. All sites are rendered from one body
evaluation under two nested forward-mode jvp (value, velocity and acceleration of the whole
pose at once), so the cost does not grow with the number of sites; the subject's rest joints
are computed once (BodyTrajectory.rest) and serve both the site table and the forward
kinematics. This is the reference renderer of the G1 harness and of the tests; the training
renderer is factory/synth.py, whose site placement is site_offsets (the same rule, batched).
"""
from dataclasses import dataclass

import torch

from ..body.rotations import vee
from ..body.trajectory import second_order
from ..kernels.imu import G

SITES = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R",
         "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R", "pocket"]
G0 = 9.80665  # the canonical contract converts m/s^2 to g with standard gravity
# site -> (segment joint, rest bone (a, b) or None, fraction of the bone b - a, surface offset m)
SITE_RULES = {
    "head": (15, None, 0.0, (0.0, 0.08, 0.08)),
    "trunk": (9, None, 0.0, (0.0, 0.05, 0.12)),
    "lower_back": (0, None, 0.0, (0.0, 0.05, -0.12)),
    "wrist_L": (18, (18, 20), 0.9, (0.0, 0.03, 0.0)),
    "wrist_R": (19, (19, 21), 0.9, (0.0, 0.03, 0.0)),
    "thigh_L": (1, (1, 4), 0.5, (0.08, 0.0, 0.0)),
    "thigh_R": (2, (2, 5), 0.5, (-0.08, 0.0, 0.0)),
    "shank_L": (4, (4, 7), 0.5, (0.06, 0.0, 0.0)),
    "shank_R": (5, (5, 8), 0.5, (-0.06, 0.0, 0.0)),
    "ankle_L": (4, (4, 7), 0.9, (0.05, 0.0, 0.0)),
    "ankle_R": (5, (5, 8), 0.9, (-0.05, 0.0, 0.0)),
    "foot_L": (7, (7, 10), 0.5, (0.0, 0.04, 0.0)),
    "foot_R": (8, (8, 11), 0.5, (0.0, 0.04, 0.0)),
    "pocket": (1, (1, 4), 0.3, (0.0, 0.0, 0.08)),
}
SITE_JOINTS = [SITE_RULES[s][0] for s in SITES]


def site_table(rest_joints):
    """rest_joints (22, 3) of one subject -> dict site -> (segment joint index, offset (3,)).
    Offsets: fractions of the rest bone vector plus a surface offset in metres (SITE_RULES)."""
    J = rest_joints
    out = {}
    for s in SITES:
        k, bone, frac, surface = SITE_RULES[s]
        t = torch.tensor(surface, dtype=J.dtype, device=J.device)
        out[s] = (k, t if bone is None else frac * (J[bone[1]] - J[bone[0]]) + t)
    return out


def site_offsets(rest, sites=SITES):
    """Batched site_table: rest (..., 22, 3) m of any number of subjects -> (joints (S,) int64,
    offsets (..., S, 3) m in the segment frames) for the sites in the given order; element for
    element the same operations as site_table, so each row equals it exactly."""
    rows, joints = [], []
    for s in sites:
        k, bone, frac, surface = SITE_RULES[s]
        t = torch.tensor(surface, dtype=rest.dtype, device=rest.device)
        joints.append(k)
        rows.append(t.expand(rest.shape[:-2] + (3,)) if bone is None
                    else frac * (rest[..., bone[1], :] - rest[..., bone[0], :]) + t)
    return torch.tensor(joints, dtype=torch.int64), torch.stack(rows, -2)


@dataclass
class Device:
    """A simple device model: constant biases, white noise, saturation (defaults are orders of
    magnitude of consumer MEMS; they are nuisances sampled per record by the factory)."""
    acc_noise_g: float = 0.005
    gyro_noise_dps: float = 0.3
    acc_bias_g: float = 0.02
    gyro_bias_dps: float = 1.0
    acc_range_g: float = 16.0
    gyro_range_dps: float = 2000.0


def site_kinematics(traj, t, sites, table):
    """Positions, accelerations (N, S, 3), rotations (N, S, 3, 3) and world angular
    velocities (N, S, 3) of the site points, from one pose evaluation under nested jvp."""
    idx = torch.tensor([table[s][0] for s in sites], device=t.device)
    off = torch.stack([table[s][1] for s in sites]).to(t.dtype)

    def f(tt):
        pos, rot = traj.pose_at(tt)
        R = rot[:, idx]
        p = pos[:, idx] + (R @ off[None, :, :, None])[..., 0]
        return p, R

    (p, R), (_, Rdot), (a, _) = second_order(f, t)
    w = vee(Rdot @ R.transpose(-1, -2))
    return p, a, R, w


def render(traj, t, sites=SITES, mount=None, device=None, generator=None, gravity_removed=False):
    """Canonical IMU block at times t (N,): signal (N, 14, 9) float32 with acc in g and gyro in
    deg/s, location_mask (14,), channel_mask (14, 9). Absent sites and magnetometer channels
    are zero and masked, as in the MoveAtlas contract. The site offsets come from the
    trajectory's rest joints (traj.rest, computed once per subject)."""
    table = site_table(traj.rest)
    _, a, R, w = site_kinematics(traj, t, sites, table)
    if mount is not None:
        R = R @ torch.stack([mount[s] for s in sites]).to(R.dtype)
    g = torch.zeros(3, dtype=a.dtype, device=a.device)
    g[1] = -G
    f = a if gravity_removed else a - g
    Rt = R.transpose(-1, -2)
    acc = (Rt @ f[..., None])[..., 0] / G0
    gyro = torch.rad2deg((Rt @ w[..., None])[..., 0])
    if device is not None:
        gen = generator
        n, s = acc.shape[:2]
        acc = acc + device.acc_bias_g * torch.randn(1, s, 3, generator=gen, dtype=acc.dtype) \
            + device.acc_noise_g * torch.randn(n, s, 3, generator=gen, dtype=acc.dtype)
        gyro = gyro + device.gyro_bias_dps * torch.randn(1, s, 3, generator=gen, dtype=acc.dtype) \
            + device.gyro_noise_dps * torch.randn(n, s, 3, generator=gen, dtype=acc.dtype)
        acc = acc.clamp(-device.acc_range_g, device.acc_range_g)
        gyro = gyro.clamp(-device.gyro_range_dps, device.gyro_range_dps)
    signal = torch.zeros(t.shape[0], len(SITES), 9, dtype=torch.float32)
    loc_mask = torch.zeros(len(SITES), dtype=torch.bool)
    ch_mask = torch.zeros(len(SITES), 9, dtype=torch.bool)
    for j, s in enumerate(sites):
        k = SITES.index(s)
        signal[:, k, 0:3] = acc[:, j].float()
        signal[:, k, 3:6] = gyro[:, j].float()
        loc_mask[k] = True
        ch_mask[k, :6] = True
    return signal, loc_mask, ch_mask
