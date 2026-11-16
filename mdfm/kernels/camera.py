"""Camera kernel (design section 3, pose 2D): projection of the body into a camera, and the
likelihood of tracker keypoints.

A camera (intrinsics K in pixels, world-to-camera rotation R and translation t in metres,
optional radial distortion k1, k2) projects the 15 canonical joints; OpenCV convention: camera
x right, y down, z forward, image u right and v down. Camera holds one camera (K (3, 3), R (3,
3), t (3,)) or one per window (K (W, 3, 3), R (W, 3, 3), t (W, 3)); camera_from_params builds
the per-window camera of the nuisance parameterization (rotation, distance to a target point,
focal length). The keypoint likelihood is a bivariate Student-t with scale sigma0 / confidence
(tracker confidence in (0, 1], 1 when the stream has none), so that a confident detection
weighs more and an outlier does not dominate; missing joints are masked here and modelled by
the tracker process elsewhere (factory/camera_render.py). Depth stays in the posterior: this
kernel never lifts 2D to 3D.
"""
import math
from typing import NamedTuple

import torch

from ..body.rotations import rot6d_to_matrix

# canonical 15 (MoveAtlas configs/canonical_skeleton.yaml) <- SMPL 22 joint indices
CANONICAL_15 = ["pelvis", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow",
                "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                "right_knee", "left_ankle", "right_ankle"]
SMPL_TO_CANONICAL_15 = [0, 12, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]


class Camera(NamedTuple):
    """A pinhole camera, or one per window. K (..., 3, 3) px; R (..., 3, 3) world to camera; t
    (..., 3) m, so that a world point p sits at R p + t in the camera frame; k1, k2 radial
    distortion (float or (W,)); width and height px; fps Hz. Leading dimension: none for one
    camera, W for one camera per window."""
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    k1: object = 0.0
    k2: object = 0.0
    width: int = 1280
    height: int = 720
    fps: float = 30.0

    @property
    def batched(self):
        return self.R.dim() > 2

    def at(self, w):
        """The camera of each query point: fields gathered by the window index w (N,) of a
        per-window camera; a single camera is returned unchanged."""
        if not self.batched:
            return self
        pick = lambda v: v[w] if torch.is_tensor(v) and v.dim() > 0 else v  # noqa: E731
        return self._replace(K=self.K[w], R=self.R[w], t=self.t[w], k1=pick(self.k1), k2=pick(self.k2))

    def to(self, dtype=None, device=None):
        cast = lambda v: v.to(dtype=dtype, device=device) if torch.is_tensor(v) else v  # noqa: E731
        return self._replace(K=cast(self.K), R=cast(self.R), t=cast(self.t), k1=cast(self.k1), k2=cast(self.k2))


def stack_cameras(cams):
    """A list of W single cameras -> one Camera with (W, ...) fields (k1, k2 as (W,) tensors;
    width, height and fps of the first camera)."""
    K = torch.stack([c.K for c in cams])
    as_t = lambda vals: torch.as_tensor([float(v) for v in vals], dtype=K.dtype, device=K.device)  # noqa: E731
    return Camera(K, torch.stack([c.R for c in cams]), torch.stack([c.t for c in cams]),
                  as_t([c.k1 for c in cams]), as_t([c.k2 for c in cams]), cams[0].width, cams[0].height,
                  cams[0].fps)


def look_at(eye, target, up=(0.0, 1.0, 0.0), dtype=torch.float64):
    """World-to-camera rotation and translation for a camera at `eye` looking at `target`
    (OpenCV convention: x right, y down, z forward)."""
    eye = torch.as_tensor(eye, dtype=dtype)
    target = torch.as_tensor(target, dtype=dtype)
    up = torch.as_tensor(up, dtype=dtype)
    z = target - eye
    z = z / z.norm()
    x = torch.cross(z, up, dim=0)
    x = x / x.norm()
    y = torch.cross(z, x, dim=0)
    R = torch.stack([x, y, z], 0)
    return R, -R @ eye


def intrinsics(fx, fy, cx, cy, dtype=torch.float64):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=dtype)


def camera_from_params(rot6d, log_distance, log_focal, target, width=1280, height=720, fps=30.0):
    """Per-window camera of the nuisance parameterization: rot6d (W, 6) the world-to-camera
    rotation, log_distance (W,) log m from the camera centre to target (W, 3) m along the
    optical axis, log_focal (W,) log px (fx = fy), principal point at the image centre. The
    target projects to the principal point at depth exp(log_distance): t = (0, 0, d) - R target."""
    R = rot6d_to_matrix(rot6d)
    d = torch.exp(log_distance)
    f = torch.exp(log_focal)
    t = torch.stack([torch.zeros_like(d), torch.zeros_like(d), d], -1) - (R @ target[..., None])[..., 0]
    z, o = torch.zeros_like(f), torch.ones_like(f)
    K = torch.stack([torch.stack([f, z, z + width / 2], -1), torch.stack([z, f, z + height / 2], -1),
                     torch.stack([z, z, o], -1)], -2)
    return Camera(K, R, t, 0.0, 0.0, width, height, fps)


def _nonzero(k):
    return torch.is_tensor(k) or k != 0


def project(points, K, R, t, k1=0.0, k2=0.0):
    """points (..., 3) world -> pixels (..., 2) and depth (...). One camera: K (3, 3), R (3, 3),
    t (3,). One camera per group: K (B..., 3, 3), R (B..., 3, 3), t (B..., 3), k1 and k2 float or
    (B...,), with points (B..., P, 3). Radial distortion on the normalized image coordinates:
    r' = r (1 + k1 r^2 + k2 r^4)."""
    if R.dim() > 2:
        pc = (R[..., None, :, :] @ points[..., None])[..., 0] + t[..., None, :]
        Kp = K[..., None, :, :]
        per = lambda k: k[..., None, None] if torch.is_tensor(k) and k.dim() > 0 else k  # noqa: E731
    else:
        pc = points @ R.transpose(-1, -2) + t
        Kp = K
        per = lambda k: k  # noqa: E731
    z = pc[..., 2]
    xy = pc[..., :2] / z[..., None]
    if _nonzero(k1) or _nonzero(k2):
        r2 = (xy * xy).sum(-1, keepdim=True)
        xy = xy * (1 + per(k1) * r2 + per(k2) * r2 * r2)
    x, y = xy[..., 0], xy[..., 1]
    u = Kp[..., 0, 0] * x + Kp[..., 0, 1] * y + Kp[..., 0, 2]
    v = Kp[..., 1, 0] * x + Kp[..., 1, 1] * y + Kp[..., 1, 2]
    return torch.stack([u, v], -1), z


def keypoint_nll(pred_uv, obs_uv, conf, valid, sigma0=8.0, nu=4.0, reduce="sum"):
    """Negative log-likelihood of observed keypoints (..., J, 2) under a bivariate Student-t
    with scale sigma0 / conf pixels and nu degrees of freedom. conf (..., J) or None (1);
    sigma0 a float or a tensor broadcasting against (..., J). reduce 'sum' sums over valid
    joints; 'none' returns (..., J), zero where valid is 0."""
    sigma = torch.as_tensor(sigma0, dtype=pred_uv.dtype, device=pred_uv.device)
    if conf is not None:
        sigma = sigma / conf.clamp_min(1e-3)
    r2 = ((pred_uv - obs_uv) ** 2).sum(-1) / sigma ** 2
    # 2D Student-t: log p = log Gamma((nu+2)/2) - log Gamma(nu/2) - log(nu pi) - 2 log sigma
    #                       - (nu+2)/2 log(1 + r2 / nu)
    logp = (math.lgamma((nu + 2) / 2) - math.lgamma(nu / 2) - math.log(nu * math.pi)
            - 2 * torch.log(sigma) - (nu + 2) / 2 * torch.log1p(r2 / nu))
    nll = torch.where(valid != 0, -logp * valid, torch.zeros_like(logp))
    if reduce == "none":
        return nll
    if reduce != "sum":
        raise ValueError(f"reduce must be 'sum' or 'none', got {reduce!r}")
    return nll.sum()
