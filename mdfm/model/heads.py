"""Nuisance heads of the stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15;
design section 1: the record nuisance is consumed by the sensor kernels and by nothing else).

The heads read the encoder outputs at nuisance-token positions only (mfm.MFM.nuisance_from_hidden gathers
them; the typed mask of model/masks.py keeps every body position from attending to those tokens) and
return the nuisance parameterization of kernels/nuisance.py. Each head is LayerNorm, Linear(d, hidden),
GELU and a zero-initialized Linear(hidden, n_out), run in float32 under any autocast, so at
initialization every head returns its prior centre:
- imu(h (W, S, d), family (W,), sites) -> IMUNuisance: mount_res6d 0 (the mount is the prior rotation R0 of
  the window's family and site), offset_res 0 (the site default), biases 0, gains 1 (gain_res 0) and
  log_scale at LOG_SCALE0 (the Student-t scales of kernels.imu.imu_nll above their floors, 0.05 g and 5
  deg/s [cfg]). Bounds on any input: each 3-vector of the 6D mount residual has a norm below MOUNT_RES_MAX
  = 0.9, so each Gram-Schmidt input keeps a norm of at least 0.1 and never vanishes; the placement
  residual below 3 cm and the gains within exp(+-0.05) (IMUNuisance); the biases within BIAS_MAX_SD = 4
  standard deviations of the device prior of the family and site (MountPriors bias_acc_sd, bias_gyro_sd),
  output in g and deg/s; log_scale within LOG_SCALE_HALF = 4 of LOG_SCALE0.
- camera(h (W, d)) -> CameraNuisance: at initialization a camera in front of the subject (on +z of the
  window frame, looking along -z at the target, the pelvis heading faces it), at the centre of the
  generic family's distance range and field of view (factory/camera_render.py: 4.5 m and 70 degrees, a
  focal of 914 px at 1,280 px), keypoint noise 3 px (kernels/pose.SIGMA0_PX); log distance, log focal
  and log sigma bounded within +-1.5, +-1 and +-2 of their centres; the rotation residual is unbounded
  (the azimuth is free).
- pose3d(h (W, d)) -> log_sigma (W,) log m of the 3D pose likelihood above its 1 cm floor, at
  POSE3D_LOG_SIGMA0 = log 0.02 m [cfg], within +-3.
The mount priors (reports/mount_priors/priors.json) are held as persistent buffers (measured priors, not
SMPL+H arrays); priors() returns them as a kernels.nuisance.MountPriors on the module's device.
"""
import math

import torch
import torch.nn as nn

from ..factory.camera_render import FAMILIES as CAMERA_FAMILIES
from ..kernels.nuisance import CameraNuisance, IMUNuisance, MountPriors
from ..kernels.pose import SIGMA0_PX

MOUNT_RES_MAX = 0.9
BIAS_MAX_SD = 4.0
LOG_SCALE0 = (math.log(0.05), math.log(5.0))     # [cfg] acc g, gyro deg/s above the floors
LOG_SCALE_HALF = 4.0
POSE3D_LOG_SIGMA0 = math.log(0.02)               # [cfg] m above the 1 cm floor
POSE3D_HALF = 3.0
_CAM = CAMERA_FAMILIES["generic"]
CAM_LOG_DISTANCE0 = math.log(sum(_CAM.distance) / 2)
CAM_LOG_FOCAL0 = math.log((_CAM.width / 2) / math.tan(math.radians(sum(_CAM.hfov_deg) / 2) / 2))
CAM_LOG_SIGMA0 = math.log(SIGMA0_PX)
CAM_HALF = (1.5, 1.0, 2.0)                       # log distance, log focal, log sigma
# world to camera of a camera on +z looking along -z, image y down (kernels.camera.look_at): rows x, y, z
CAM_ROT6D0 = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
N_IMU_OUT = 6 + 3 + 3 + 3 + 2 + 2               # mount, offset, bias acc, bias gyro, log scale, gain
N_CAM_OUT = 6 + 3


def bounded(r, bound):
    """bound tanh(|r|) r / |r| over the last axis: zero at zero, about r for small r, norm below bound."""
    n = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-24)
    return bound * torch.tanh(n) * r / n


def centred(raw, centre, half):
    """centre + half tanh(raw / half): the centre at raw 0, slope 1 there, within +-half."""
    return centre + half * torch.tanh(raw / half)


class _Head(nn.Module):
    def __init__(self, d, hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.net(h.to(self.net[1].weight.dtype))


class NuisanceHeads(nn.Module):
    """NuisanceHeads(d=512, hidden=1024, priors=None): the IMU, camera and pose3d heads (module docstring);
    priors a kernels.nuisance.MountPriors (default reports/mount_priors/priors.json)."""

    PRIOR_FIELDS = ("R0", "up_sensor", "sigma_tilt", "sigma_yaw", "bias_acc_sd", "bias_gyro_sd", "noise_acc",
                    "noise_gyro", "fitted")

    def __init__(self, d=512, hidden=1024, priors=None):
        super().__init__()
        pr = MountPriors.load() if priors is None else priors
        self.families, self.sites = tuple(pr.families), tuple(pr.sites)
        for name in self.PRIOR_FIELDS:
            x = getattr(pr, name)
            self.register_buffer("mp_" + name, x if x.dtype == torch.bool else x.float())
        self.imu_head = _Head(d, hidden, N_IMU_OUT)
        self.camera_head = _Head(d, hidden, N_CAM_OUT)
        self.pose3d_head = _Head(d, hidden, 1)

    def priors(self):
        """The mount and device priors (MountPriors) on this module's device and dtype."""
        return MountPriors(self.families, self.sites, **{n: getattr(self, "mp_" + n) for n in self.PRIOR_FIELDS})

    def imu(self, h, family, sites=None):
        """IMUNuisance (W, S) from nuisance-token outputs h (W, S, d); family (W,) int64 mount-family ids of
        the windows; sites (S,) or (W, S) int64 site ids (default the 14 sites in order)."""
        W, S = h.shape[:2]
        y = self.imu_head(h)
        dev = y.device
        fam = torch.as_tensor(family, dtype=torch.int64, device=dev).reshape(W)
        site = torch.arange(S, device=dev) if sites is None else torch.as_tensor(sites, dtype=torch.int64, device=dev)
        site = site.expand(W, S)
        sd_a = self.mp_bias_acc_sd.to(y.dtype)[fam[:, None], site]              # (W, S, 3)
        sd_g = self.mp_bias_gyro_sd.to(y.dtype)[fam[:, None], site]
        mount = torch.cat([bounded(y[..., 0:3], MOUNT_RES_MAX), bounded(y[..., 3:6], MOUNT_RES_MAX)], -1)
        ls0 = torch.tensor(LOG_SCALE0, dtype=y.dtype, device=dev)
        return IMUNuisance(mount_res6d=mount, offset_res=y[..., 6:9],
                           bias_acc=sd_a * centred(y[..., 9:12], 0.0, BIAS_MAX_SD),
                           bias_gyro=sd_g * centred(y[..., 12:15], 0.0, BIAS_MAX_SD),
                           log_scale=centred(y[..., 15:17], ls0, LOG_SCALE_HALF), gain_res=y[..., 17:19])

    def camera(self, h):
        """CameraNuisance (W,) from nuisance-token outputs h (W, d) of pose2d streams."""
        y = self.camera_head(h)
        r0 = torch.tensor(CAM_ROT6D0, dtype=y.dtype, device=y.device)
        return CameraNuisance(rot6d=r0 + y[:, 0:6], log_distance=centred(y[:, 6], CAM_LOG_DISTANCE0, CAM_HALF[0]),
                              log_focal=centred(y[:, 7], CAM_LOG_FOCAL0, CAM_HALF[1]),
                              log_sigma_px=centred(y[:, 8], CAM_LOG_SIGMA0, CAM_HALF[2]))

    def pose3d(self, h):
        """log_sigma (W,) log m of the pose3d likelihood from nuisance-token outputs h (W, d)."""
        return centred(self.pose3d_head(h)[:, 0], POSE3D_LOG_SIGMA0, POSE3D_HALF)
