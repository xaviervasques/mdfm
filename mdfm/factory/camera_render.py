"""Synthetic pose streams (design sections 3 and 5; spec 2.4): camera sampler, tracker-failure
process and the renders of the canonical 2D and 3D contracts.

sample_camera draws a static camera for one window [cfg]: horizontal distance 3 to 6 m from the
target (the window-mean pelvis), eye height 0.8 to 1.8 m above the floor (y = 0), uniform azimuth,
horizontal field of view 60 to 80 degrees (fx = fy), principal point at the image centre, no
distortion, 1280 x 720 px, fps drawn from {25, 30, 50, 60}; it looks at the target with zero roll.

TrackerProcess turns the clean projection into tracker output [cfg]:
- frame-time jitter: each frame samples the body at k / fps + d_k, d_k ~ N(0, jitter_s), by linear
  interpolation between the rendered frames (clamped to the first and last frame; the truth keeps the
  applied shift);
- dropouts: a joint's detection is lost with entry probability sigmoid(logit(base_rate) + speed_gain x
  speed + occlusion_gain x occlusion), speed the joint's image speed in image torso lengths per second
  (central differences of the clean pixels; the torso length is the median pelvis-neck pixel
  distance), occlusion the depth-order proxy of occlusion(); a lost joint stays lost the next frame
  with probability max(persist, entry probability);
- joints outside the image or behind the camera are not detected;
- pixel noise: isotropic bivariate Student-t, scale noise_px, nu degrees of freedom (an integer: the
  chi-square is a sum of squared normals, so every draw goes through the torch generator).
The per-joint confidence (1 - entry probability) is nuisance truth only: the 2D contract has no
confidence field. The detections then pass through kernels/pose.normalize_2d (a frame whose pelvis is
lost is invalid as a whole, as in MoveAtlas).

render_pose3d maps body joints to the canonical 15 with the per-source offsets
(kernels/pose.to_canonical15), turns them by a lab heading about +y (uniform unless given; real
records carry an arbitrary heading), adds Student-t joint noise and applies normalize_3d.
Renders run without gradients on the CPU in float64 (inputs are detached and moved). Draws happen in
a fixed order from the caller's torch.Generator (a CPU generator), so a render is deterministic under
a seed.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from ..body.rotations import rot_y
from ..kernels import pose
from ..kernels.camera import Camera, intrinsics, look_at, project

WIDTH, HEIGHT = 1280, 720
# bones of the canonical 15 (camera.CANONICAL_15 order): trunk, head, arms, legs
CANONICAL_BONES = ((0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (5, 7), (4, 6), (6, 8), (0, 9), (0, 10), (9, 11),
                   (11, 13), (10, 12), (12, 14))


@dataclass(frozen=True)
class CameraFamily:
    """Ranges of a camera family [cfg]: distance m (horizontal, to the target), eye height m,
    horizontal field of view deg, frame rates Hz, image size px."""
    distance: tuple = (3.0, 6.0)
    height: tuple = (0.8, 1.8)
    hfov_deg: tuple = (60.0, 80.0)
    fps: tuple = (25.0, 30.0, 50.0, 60.0)
    width: int = WIDTH
    height_px: int = HEIGHT


FAMILIES = {"generic": CameraFamily()}


def _uniform(gen, lo, hi):
    return lo + (hi - lo) * float(torch.rand((), generator=gen, dtype=torch.float64))


def sample_camera(gen, target, family="generic", fps=None):
    """One static camera looking at target (3,) m (the window-mean pelvis, window frame, y up) ->
    camera.Camera (float64, single). fps None draws from the family's rates; draws in the order
    azimuth, distance, height, field of view, fps."""
    fam = FAMILIES.get(family)
    if fam is None:
        raise ValueError(f"unknown camera family {family!r}; known: {sorted(FAMILIES)}")
    target = torch.as_tensor(target).detach().to("cpu", torch.float64).reshape(3)
    az = _uniform(gen, -math.pi, math.pi)
    d = _uniform(gen, *fam.distance)
    h = _uniform(gen, *fam.height)
    hfov = math.radians(_uniform(gen, *fam.hfov_deg))
    k = int(torch.randint(len(fam.fps), (), generator=gen))
    rate = float(fam.fps[k]) if fps is None else float(fps)
    eye = torch.tensor([float(target[0]) + d * math.sin(az), h, float(target[2]) + d * math.cos(az)],
                       dtype=torch.float64)
    R, t = look_at(eye, target)
    f = (fam.width / 2) / math.tan(hfov / 2)
    K = intrinsics(f, f, fam.width / 2, fam.height_px / 2)
    return Camera(K, R, t, 0.0, 0.0, fam.width, fam.height_px, rate)


def occluders(n=15, bones=None):
    """(J, J) float: 1 where joint k may hide joint j, i.e. k is neither j nor a joint sharing a bone
    with j (the canonical bones by default; a joint's bone neighbours sit next to it in every view)."""
    m = 1 - torch.eye(n, dtype=torch.float64)
    for a, b in (CANONICAL_BONES if bones is None else bones):
        m[a, b] = m[b, a] = 0.0
    return m


def occlusion(uv, depth, torso_px, radius=0.3, margin=0.05):
    """Self-occlusion proxy (T, J) in [0, 1] from depth order: 1 - prod_k (1 - [depth_j > depth_k +
    margin] exp(-|uv_j - uv_k|^2 / (2 (radius torso_px)^2))) over the occluders k of j (occluders());
    uv (T, J, 2) px, depth (T, J) m along the optical axis."""
    behind = (depth[:, :, None] - depth[:, None, :]) > margin
    d2 = ((uv[:, :, None, :] - uv[:, None, :, :]) ** 2).sum(-1)
    k = torch.exp(-d2 / (2 * (radius * torso_px) ** 2)) * behind
    k = k * occluders(uv.shape[1]).to(dtype=k.dtype, device=k.device)
    return 1 - torch.prod(1 - k, dim=-1)


def image_speed(uv, fps):
    """(T, J, 2) px -> (T, J) px/s by central differences (one-sided at the ends; 0 for one frame)."""
    if uv.shape[0] < 2:
        return torch.zeros(uv.shape[:2], dtype=uv.dtype, device=uv.device)
    d = torch.empty_like(uv)
    d[1:-1] = (uv[2:] - uv[:-2]) / 2
    d[0] = uv[1] - uv[0]
    d[-1] = uv[-1] - uv[-2]
    return d.norm(dim=-1) * fps


def _torso_px(uv):
    return float(torch.nan_to_num((uv[:, pose.ROOT] - uv[:, pose.NECK]).norm(dim=-1), nan=0.0).median().clamp_min(1.0))


@dataclass(frozen=True)
class TrackerProcess:
    """Parametric tracker-failure process [cfg] (module docstring): noise_px Student-t scale px, nu
    degrees of freedom (integer), base_rate the entry probability of a still, unoccluded joint per
    frame, speed_gain logit per image torso length per second, occlusion_gain logit per unit of the
    occlusion proxy, persist the probability a lost joint stays lost, jitter_s frame-time jitter sd s,
    occ_radius (torso lengths) and occ_margin (m) of the occlusion proxy."""
    noise_px: float = 3.0
    nu: int = 4
    base_rate: float = 0.01
    speed_gain: float = 0.2
    occlusion_gain: float = 2.0
    persist: float = 0.6
    jitter_s: float = 0.002
    occ_radius: float = 0.3
    occ_margin: float = 0.05

    @classmethod
    def clean(cls):
        """No noise, no dropout, no jitter: the exact projection (joints outside the image stay
        undetected)."""
        return cls(noise_px=0.0, base_rate=0.0, speed_gain=0.0, occlusion_gain=0.0, persist=0.0, jitter_s=0.0)

    def entry_prob(self, speed, occ):
        """Dropout entry probability (T, J) from image speed (torso lengths per second) and the
        occlusion proxy."""
        if self.base_rate <= 0:
            return torch.zeros_like(speed)
        logit = math.log(self.base_rate / (1 - self.base_rate))
        return torch.sigmoid(logit + self.speed_gain * speed + self.occlusion_gain * occ)

    def dropouts(self, p, gen):
        """Lost-detection mask (T, ...) bool of the two-state chain along the first (frame) axis: lost at
        frame k with probability p_k when detected at k - 1, max(persist, p_k) when lost at k - 1."""
        u = torch.rand(p.shape, generator=gen, dtype=torch.float64).to(p.dtype)
        lost = torch.zeros(p.shape, dtype=torch.bool)
        prev = torch.zeros(p.shape[1:], dtype=torch.bool)
        for k in range(p.shape[0]):
            q = torch.where(prev, p[k].clamp_min(self.persist), p[k])
            prev = u[k] < q
            lost[k] = prev
        return lost

    def noise(self, shape, gen, dtype=torch.float64):
        """Isotropic bivariate Student-t pixel noise of shape (T, J, 2)."""
        z = torch.randn(shape, generator=gen, dtype=torch.float64)
        if self.noise_px <= 0:
            return torch.zeros(shape, dtype=dtype)
        if int(self.nu) != self.nu or self.nu < 1:
            raise ValueError(f"nu must be a positive integer, got {self.nu}")
        chi2 = (torch.randn(shape[:-1] + (int(self.nu),), generator=gen, dtype=torch.float64) ** 2).sum(-1)
        return (self.noise_px * z / torch.sqrt(chi2 / self.nu)[..., None]).to(dtype)


def jitter_frames(pos, fps, jitter_s, gen):
    """pos (T, J, 3) at k / fps -> the positions at k / fps + d_k, d_k ~ N(0, jitter_s) (T,), by
    linear interpolation between frames, clamped to the first and last frame; returns (pos, d) with d
    the shift s actually applied (a draw beyond an end frame is clamped there)."""
    T = pos.shape[0]
    d = torch.randn(T, generator=gen, dtype=torch.float64) * jitter_s
    if T < 2 or jitter_s <= 0:
        return pos, torch.zeros(T, dtype=torch.float64)
    k = torch.arange(T, dtype=torch.float64)
    f = (k + d * fps).clamp(0, T - 1)
    i0 = f.floor().long().clamp(max=T - 2)
    a = (f - i0).to(pos.dtype)[:, None, None]
    return pos[i0] * (1 - a) + pos[i0 + 1] * a, (f - k) / fps


def render_pose2d(pos15, cam, gen, tracker=None):
    """pos15 (T, 15, 3) m window frame, y up, at the camera's frame times k / cam.fps; cam a single
    camera.Camera -> (record, truth). record: the canonical 2D contract (coords (T, 15, 3) float32
    with z = 0, coord_mask, joint_mask, valid_mask (T, 15), dim 2, fps, modality 2d_pixel, n_frames,
    scale px per torso unit). truth: clean pixels uv (T, 15, 2) of the jittered frames, jitter_s (T,),
    entry probability, confidence and lost mask (T, 15), in_image (T, 15), the camera."""
    tr = TrackerProcess() if tracker is None else tracker
    if cam.batched:
        raise ValueError("render_pose2d takes one camera (K (3, 3), R (3, 3), t (3,))")
    pos = torch.as_tensor(pos15).detach().to("cpu", torch.float64)
    cam = cam.to(torch.float64, "cpu")
    pos, dt = jitter_frames(pos, float(cam.fps), tr.jitter_s, gen)
    uv, depth = project(pos, cam.K, cam.R, cam.t, cam.k1, cam.k2)
    in_image = (depth > 0.1) & (uv[..., 0] >= 0) & (uv[..., 0] < cam.width) & (uv[..., 1] >= 0) & \
        (uv[..., 1] < cam.height)
    torso = _torso_px(uv)
    speed = torch.nan_to_num(image_speed(uv, float(cam.fps)) / torso, nan=0.0, posinf=1e6)
    occ = torch.nan_to_num(occlusion(uv, depth, torso, tr.occ_radius, tr.occ_margin), nan=0.0)
    p = tr.entry_prob(speed, occ)
    lost = tr.dropouts(p, gen)
    obs = uv + tr.noise(uv.shape, gen)
    detected = in_image & ~lost
    coords, scale, valid = pose.normalize_2d(obs.numpy(), detected.numpy())
    record = {"coords": coords, "coord_mask": np.array([True, True, False]), "joint_mask": np.ones(15, bool),
              "valid_mask": valid, "dim": 2, "fps": float(cam.fps), "modality": "2d_pixel",
              "n_frames": int(coords.shape[0]), "scale": scale}
    truth = {"uv": uv.float().numpy(), "jitter_s": dt.numpy(), "entry_prob": p.float().numpy(),
             "confidence": (1 - p).float().numpy(), "lost": lost.numpy(), "in_image": in_image.numpy(),
             "camera": cam}
    return record, truth


def render_pose3d(pos, R_glob, gen, offsets=None, fps=100.0, noise_m=0.003, nu=4, heading=None):
    """pos (T, 22, 3) m and R_glob (T, 22, 3, 3) of the body, window frame, y up; offsets (15, 3) m of
    the source (kernels/pose.to_canonical15) or None -> (record, truth). The canonical joints are
    turned by heading rad about +y (None: uniform), given Student-t noise (scale noise_m, nu integer
    degrees of freedom) and normalized (normalize_3d). record: the canonical 3D contract (coords
    float32, coord_mask, joint_mask, valid_mask, dim 3, fps, modality 3d_meter, n_frames, scale m).
    truth: heading and the clean turned joints (T, 15, 3) m."""
    pos = torch.as_tensor(pos).detach().to("cpu", torch.float64)
    R_glob = torch.as_tensor(R_glob).detach().to("cpu", torch.float64)
    off = None if offsets is None else torch.as_tensor(offsets).detach().to("cpu", torch.float64)
    j15 = pose.to_canonical15(pos, R_glob, off)
    th = _uniform(gen, -math.pi, math.pi) if heading is None else float(heading)
    j15 = j15 @ rot_y(torch.tensor(th, dtype=torch.float64)).T
    z = torch.randn(j15.shape, generator=gen, dtype=torch.float64)
    if noise_m > 0:
        if int(nu) != nu or nu < 1:
            raise ValueError(f"nu must be a positive integer, got {nu}")
        chi2 = (torch.randn(j15.shape[:-1] + (int(nu),), generator=gen, dtype=torch.float64) ** 2).sum(-1)
        noisy = j15 + noise_m * z / torch.sqrt(chi2 / nu)[..., None]
    else:
        noisy = j15
    coords, scale, valid = pose.normalize_3d(noisy.numpy())
    record = {"coords": coords, "coord_mask": np.array([True, True, True]), "joint_mask": np.ones(15, bool),
              "valid_mask": valid, "dim": 3, "fps": float(fps), "modality": "3d_meter",
              "n_frames": int(coords.shape[0]), "scale": scale}
    return record, {"heading": th, "joints": j15.float().numpy()}
