"""Nuisances of the sensor kernels (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.3; design
section 3): IMU mounting, placement, biases, gains and noise scales per window and site; the camera
nuisance; per-source joint offsets; and the mounting priors per layout family.

IMU (IMUNuisance, W windows by S sites). Mounting, sensor frame to segment frame: M = R0 @ dR with
R0 the prior rotation of the window's family and site and dR = rot6d_to_matrix(E6 + mount_res6d),
so that a zero residual gives R0 (to the 1e-12 floor of the Gram-Schmidt step). Placement: offset
= site default (factory.imu_render.site_table, segment frame) + OFFSET_MAX_M tanh(|r|) r / |r|,
never beyond 3 cm. Biases in the contract units (g, deg/s). Gains exp(GAIN_LOG_MAX
tanh(gain_res)), within about 5 percent [cfg].
log_scale (W, S, 2): the Student-t scales of kernels.imu.imu_nll (acc, gyro) above their floors.

Mounting prior (MountPriors, reports/mount_priors/priors.json, scripts/build_mount_priors.py). The
rotation vector theta of dR, in the nominal sensor frame, splits into yaw, its component along
up_sensor (the sensor axis that points up when the body stands, R0^T times the segment's up in the
renderer's standing posture), and tilt, the rest: tilt ~ 2D Gaussian with sigma_tilt, yaw ~
Gaussian with sigma_yaw, per family and site (a Gaussian in the tangent space of SO(3), rad). The
angle comes from atan2 of the sine and cosine of dR, so that a flipped mount is penalized at pi; the
split into tilt and yaw (tilt_yaw2) reads the axis from the skew part of dR near the identity and from
its symmetric part near pi, so that value and gradient stay finite up to a flipped mount.
Biases carry zero-mean Gaussian priors with the device family's spreads; placement and gains have
their bounds only. Families: generic (every site, R0 identity, 10 degrees [cfg], the Device
defaults of factory.imu_render) and xsens13 (the 13 Xsens sites of weargait_pd, measured; pocket
falls back to generic).
"""
import json
import math
import os
from dataclasses import dataclass, fields

import torch

from .. import paths
from ..body.rotations import axis_angle_to_matrix, rot6d_to_matrix, vee
from ..factory.imu_render import SITES
from .imu import imu_predict

PRIORS_PATH = os.path.join(paths.TREE, "reports", "mount_priors", "priors.json")
OFFSET_MAX_M = 0.03   # [cfg] placement residual bound
GAIN_LOG_MAX = 0.05   # [cfg] log-gain bound
N_JOINTS15 = 15


def identity6d(dtype=torch.float32, device=None):
    """The 6D form of the identity rotation, (6,)."""
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)


def _bounded(r, bound):
    """bound tanh(|r|) r / |r| over the last axis: smooth, zero at zero, norm below bound."""
    n = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-24)
    return bound * torch.tanh(n) * r / n


def angle_axis_parts(R, u):
    """Rotation angle (...) rad in [0, pi] of R (..., 3, 3) and its signed yaw about the unit vector u
    (..., 3): yaw = angle x (axis . u), so that angle^2 - yaw^2 is the squared tilt. Finite
    gradients at the identity; the axis comes from vee(R) = sin(angle) axis, which loses it near pi
    (the prior uses tilt_yaw2)."""
    v = vee(R)
    s = torch.sqrt((v * v).sum(-1) + 1e-24)
    c = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    angle = torch.atan2(s, c)
    return angle, angle * (v * u).sum(-1) / s


def tilt_yaw2(R, u):
    """Squared tilt and yaw (...) rad^2 of R (..., 3, 3) about the unit vector u (..., 3): yaw^2 =
    angle^2 (axis . u)^2, tilt^2 = angle^2 - yaw^2. (axis . u)^2 is read from two identities,
    (vee(R) . u)^2 = sin^2 (axis . u)^2, exact near the identity, and u^T R u - cos = (1 - cos)
    (axis . u)^2, exact near pi: (axis . u)^2 = ((vee(R) . u)^2 + (1 - cos)(u^T R u - cos)) /
    (sin^2 + (1 - cos)^2), whose value and gradient stay finite from the identity to a flipped mount."""
    v = vee(R)
    s2 = (v * v).sum(-1)
    c = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    angle = torch.atan2(torch.sqrt(s2 + 1e-24), c)
    vu = (v * u).sum(-1)
    uRu = (u[..., :, None] * R * u[..., None, :]).sum((-1, -2))
    one_c = 1 - c
    au2 = ((vu * vu + one_c * (uRu - c)) / (s2 + one_c * one_c + 1e-30)).clamp(0.0, 1.0)
    angle2 = angle * angle
    yaw2 = angle2 * au2
    return angle2 - yaw2, yaw2


# ---------------------------------------------------------------- IMU nuisance

@dataclass
class IMUNuisance:
    """IMU nuisance of W windows and S sites (module docstring): mount_res6d (W, S, 6), offset_res
    (W, S, 3), bias_acc (W, S, 3) g, bias_gyro (W, S, 3) deg/s, log_scale (W, S, 2), gain_res (W,
    S, 2) or None (unit gains)."""
    mount_res6d: torch.Tensor
    offset_res: torch.Tensor
    bias_acc: torch.Tensor
    bias_gyro: torch.Tensor
    log_scale: torch.Tensor
    gain_res: torch.Tensor = None

    @classmethod
    def zeros(cls, W, S, dtype=torch.float32, device=None):
        """The nuisance at its prior centre: zero residuals, biases and log scales, unit gains."""
        z = lambda *s: torch.zeros(W, S, *s, dtype=dtype, device=device)  # noqa: E731
        return cls(z(6), z(3), z(3), z(3), z(2), z(2))

    @property
    def shape(self):
        return tuple(self.mount_res6d.shape[:2])

    def residual_rotation(self):
        """dR (W, S, 3, 3), the identity for a zero residual."""
        e6 = identity6d(self.mount_res6d.dtype, self.mount_res6d.device)
        return rot6d_to_matrix(e6 + self.mount_res6d)

    def mount(self, R0):
        """Sensor to segment rotations (W, S, 3, 3) = R0 @ dR; R0 (W, S, 3, 3) or (S, 3, 3)."""
        return R0.to(self.mount_res6d.dtype) @ self.residual_rotation()

    def offset(self):
        """Placement residual (W, S, 3) m in the segment frame, norm below OFFSET_MAX_M."""
        return _bounded(self.offset_res, OFFSET_MAX_M)

    def gain(self):
        """Gains (W, S, 2) for acc and gyro, within exp(+-GAIN_LOG_MAX)."""
        if self.gain_res is None:
            return torch.ones(*self.shape, 2, dtype=self.mount_res6d.dtype, device=self.mount_res6d.device)
        return torch.exp(GAIN_LOG_MAX * torch.tanh(self.gain_res))

    def predict(self, kin, w, site_joint, offset0, R0, fir=None, segment_len=None, gravity_removed=False):
        """kernels.imu.imu_predict at the N points of kin (BodyKin) with the nuisance of their
        windows w (N,) int64: offset0 (S, 3) or (W, S, 3) m the site defaults, R0 (S, 3, 3) or
        (W, S, 3, 3) the prior rotations. Returns acc (N, S, 3) g and gyro (N, S, 3) deg/s."""
        off = self.offset()
        off = (offset0.to(off.dtype) + off)[w]
        R = self.mount(R0)[w]
        gain = self.gain()[w]
        return imu_predict(kin, site_joint, off, R, self.bias_acc[w], self.bias_gyro[w], fir,
                           gain_acc=gain[..., 0:1], gain_gyro=gain[..., 1:2],
                           gravity_removed=gravity_removed, segment_len=segment_len)


# ---------------------------------------------------------------- camera and source offsets

@dataclass
class CameraNuisance:
    """Camera nuisance of W windows: rot6d (W, 6) the world-to-camera rotation (OpenCV axes of
    kernels.camera: x right, y down, z forward), log_distance (W,) log m from the camera to its
    target, log_focal (W,) log px, log_sigma_px (W,) log px of the keypoint noise scale."""
    rot6d: torch.Tensor
    log_distance: torch.Tensor
    log_focal: torch.Tensor
    log_sigma_px: torch.Tensor

    def rotation(self):
        """World-to-camera rotations (W, 3, 3)."""
        return rot6d_to_matrix(self.rot6d)

    def extrinsics(self, target):
        """R (W, 3, 3) and t (W, 3) m of kernels.camera.project for a camera on its optical axis at
        exp(log_distance) from target (W, 3) m (window-mean pelvis): centre c = target - d R^T e_z,
        t = -R c, so the target projects to the principal point at depth d."""
        R = self.rotation()
        c = target - torch.exp(self.log_distance)[:, None] * R[:, 2, :]
        return R, -(R * c[:, None, :]).sum(-1)

    def intrinsics(self, width=1280, height=720):
        """K (W, 3, 3): focal exp(log_focal) px on both axes, principal point at the image centre."""
        f = torch.exp(self.log_focal)
        K = torch.zeros(f.shape[0], 3, 3, dtype=f.dtype, device=f.device)
        K[:, 0, 0], K[:, 1, 1] = f, f
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = width / 2.0, height / 2.0, 1.0
        return K

    def sigma_px(self):
        return torch.exp(self.log_sigma_px)


class SourceOffsets(torch.nn.Module):
    """Per-source joint-definition offsets (n_sources, 15, 3) m of the 15 canonical joints, in each
    joint's parent segment frame (kernels.pose.to_canonical15); the row of synthetic truth is
    held at zero (no value, no gradient)."""

    def __init__(self, n_sources, synthetic_row=0):
        super().__init__()
        if not 0 <= synthetic_row < n_sources:
            raise ValueError(f"synthetic_row {synthetic_row} outside [0, {n_sources})")
        self.offsets = torch.nn.Parameter(torch.zeros(n_sources, N_JOINTS15, 3))
        mask = torch.ones(n_sources, 1, 1)
        mask[synthetic_row] = 0.0
        self.register_buffer("mask", mask, persistent=False)
        self.synthetic_row = synthetic_row

    def forward(self, source_id):
        """source_id (W,) int64 -> offsets (W, 15, 3) m."""
        return (self.offsets * self.mask)[source_id]


# ---------------------------------------------------------------- mounting priors

# MountPriors field -> key of a family in priors.json (units in the key)
JSON_KEYS = {"R0": "R0", "up_sensor": "up_sensor", "sigma_tilt": "sigma_tilt_deg", "sigma_yaw": "sigma_yaw_deg",
             "bias_acc_sd": "bias_acc_sd_g", "bias_gyro_sd": "bias_gyro_sd_dps", "noise_acc": "noise_acc_g",
             "noise_gyro": "noise_gyro_dps", "fitted": "fitted"}


@dataclass
class MountPriors:
    """Mounting and device priors per layout family (module docstring), float64 unless moved:
    R0 (F, 14, 3, 3) sensor to segment; up_sensor (F, 14, 3); sigma_tilt, sigma_yaw (F, 14) rad;
    bias_acc_sd (F, 14, 3) g, bias_gyro_sd (F, 14, 3) deg/s, noise_acc (F, 14, 3) g, noise_gyro
    (F, 14, 3) deg/s (white noise per sample); fitted (F, 14) bool, measured rather than generic.
    Sites in the order of factory.imu_render.SITES."""
    families: tuple
    sites: tuple
    R0: torch.Tensor
    up_sensor: torch.Tensor
    sigma_tilt: torch.Tensor
    sigma_yaw: torch.Tensor
    bias_acc_sd: torch.Tensor
    bias_gyro_sd: torch.Tensor
    noise_acc: torch.Tensor
    noise_gyro: torch.Tensor
    fitted: torch.Tensor

    @classmethod
    def load(cls, path=PRIORS_PATH, dtype=torch.float64, device=None):
        """Read priors.json (scripts/build_mount_priors.py)."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if tuple(d["sites"]) != tuple(SITES):
            raise ValueError(f"{path}: sites {d['sites']} differ from factory.imu_render.SITES")
        fams = tuple(d["family_order"])
        deg = math.pi / 180.0
        arr = {}
        for k, src in JSON_KEYS.items():
            x = torch.tensor([d["families"][f][src] for f in fams], dtype=torch.bool if k == "fitted" else dtype,
                             device=device)
            arr[k] = x * deg if k in ("sigma_tilt", "sigma_yaw") else x
        return cls(fams, tuple(d["sites"]), **arr)

    def to(self, dtype=None, device=None):
        out = {}
        for f in fields(self):
            x = getattr(self, f.name)
            if isinstance(x, torch.Tensor):
                x = x.to(device=device, dtype=None if x.dtype == torch.bool else dtype)
            out[f.name] = x
        return MountPriors(**out)

    def family_index(self, family):
        """A family name, an int or an int64 tensor of family ids -> an int64 tensor."""
        if isinstance(family, str):
            if family not in self.families:
                raise KeyError(f"unknown mount family {family!r}; known: {self.families}")
            family = self.families.index(family)
        return torch.as_tensor(family, dtype=torch.int64, device=self.R0.device)

    def site_index(self, sites):
        """Site names or ids -> (S,) int64 ids into SITES; None gives all 14."""
        if sites is None:
            return torch.arange(len(self.sites), device=self.R0.device)
        if isinstance(sites, torch.Tensor):
            return sites.to(torch.int64).to(self.R0.device)
        return torch.tensor([self.sites.index(s) if isinstance(s, str) else int(s) for s in sites],
                            dtype=torch.int64, device=self.R0.device)

    def rotation(self, family, sites=None):
        """R0 of a family (-> (S, 3, 3)) or of per-window families (W,) (-> (W, S, 3, 3))."""
        f, s = self.family_index(family), self.site_index(sites)
        return self.R0[f][..., s, :, :]

    def per_site(self, x, family, sites=None):
        """Rows of a prior array x (F, 14) or (F, 14, 3) at family and sites: (S,) or (S, 3), or
        (W, S) and (W, S, 3) for per-window families (W,)."""
        f, s = self.family_index(family), self.site_index(sites)
        return x[f][..., s] if x.dim() == 2 else x[f][..., s, :]

    def mount_log_prior(self, dR, family, sites=None):
        """Log density (..., S) of residual rotations dR (..., S, 3, 3) (M = R0 @ dR) under the
        tilt and yaw spreads of family (name, id or (W,) ids) at sites."""
        u = self.per_site(self.up_sensor, family, sites).to(dR.dtype)
        st = self.per_site(self.sigma_tilt, family, sites).to(dR.dtype)
        sy = self.per_site(self.sigma_yaw, family, sites).to(dR.dtype)
        tilt2, yaw2 = tilt_yaw2(dR, u)
        return (-0.5 * tilt2 / st ** 2 - 0.5 * yaw2 / sy ** 2 - 2 * torch.log(st) - torch.log(sy)
                - 1.5 * math.log(2 * math.pi))

    def bias_log_prior(self, bias_acc, bias_gyro, family, sites=None):
        """Gaussian log density (..., S) of biases (..., S, 3) in g and deg/s."""
        out = 0.0
        for b, sd in ((bias_acc, self.bias_acc_sd), (bias_gyro, self.bias_gyro_sd)):
            sd = self.per_site(sd, family, sites).to(b.dtype)
            out = out + (-0.5 * (b / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)).sum(-1)
        return out

    def log_prior(self, nuis, family, sites=None, mask=None):
        """-nuis_prior of the loss table (spec section 4) per window, (W,): mount tilt and yaw and
        the biases of IMUNuisance nuis under family (name, id or (W,) ids) at sites (S names or
        ids, None for the 14 sites), summed over the sites where mask (W, S) is True."""
        lp = self.mount_log_prior(nuis.residual_rotation(), family, sites) \
            + self.bias_log_prior(nuis.bias_acc, nuis.bias_gyro, family, sites)
        if mask is not None:
            lp = torch.where(mask, lp, torch.zeros_like(lp))
        return lp.sum(-1)

    def sample_mounts(self, generator, family, sites=None, n=1):
        """n draws (n, S, 3, 3) of sensor to segment rotations R0 @ Exp(theta) from the prior of one
        family: theta = sigma_tilt (z - (z . u) u) + sigma_yaw (z . u) u with z standard normal, drawn
        on the CPU from generator (a CPU torch.Generator) and moved to the priors' device."""
        u = self.per_site(self.up_sensor, family, sites)
        st = self.per_site(self.sigma_tilt, family, sites)[:, None]
        sy = self.per_site(self.sigma_yaw, family, sites)[:, None]
        z = torch.randn(n, u.shape[0], 3, generator=generator, dtype=u.dtype).to(u.device)
        zu = (z * u).sum(-1, keepdim=True)
        theta = st * (z - zu * u) + sy * zu * u
        return self.rotation(family, sites) @ axis_angle_to_matrix(theta)
