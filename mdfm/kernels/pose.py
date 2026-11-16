"""Pose kernels (design section 3, pose 2D and 3D; spec 2.3): the MoveAtlas canonical
normalization, the body-to-canonical joint map with per-source joint-definition offsets, and the
likelihoods of 3D pose, 2D pose and the root track.

Canonical records (MoveAtlas configs/canonical_skeleton.yaml, build_canonical.py normalize and
build_canonical_amass.py normalize): 15 joints (camera.CANONICAL_15), root-centred per frame on the
pelvis, image y flipped up in 2D, divided by the per-record median pelvis-neck length over the frames
where both are finite and the length exceeds 1e-6 (scale, in the source unit: metres for 3d_meter,
pixels for 2d_pixel; 1.0 when no frame qualifies). A joint is valid when it is detected and its
frame's pelvis is detected; invalid entries hold 0. normalize_3d and normalize_2d replicate this in
float64 with numpy and return float32 coords, as the records store them.

Likelihoods take N flat query points with their window index w (N,) int64 and return per-window sums
(W,). Per-window inputs (scale, unit_to_si, log_sigma, z_up, yaw, sigma0, a per-window camera) are
scalars or (W,) tensors; per-point inputs are (N, ...). weight (N,) multiplies each point's term (the
n_total / n_sampled factor of spec section 4). Missing joints (valid 0) contribute nothing, to the
likelihood and to the yaw fit alike. The pelvis is zero on both sides of a root-centred comparison, so
it carries no likelihood term in pose3d and pose2d (a zero-residual term would only pull a learned
sigma down). Precision: float32 or float64, as the body kinematics.
- pose3d: the prediction (window frame, y up, m) is root-centred on its pelvis; the observation is
  obs x scale x unit_to_si (m, root-centred by the contract), rotated from z-up to y-up first for
  z-up sources, then turned by one yaw per window about +y fitted in closed form by least squares
  (yaw_fit); isotropic 3D Student-t per joint, scale sigma = sqrt(exp(log_sigma)^2 + floor^2) m with
  the floor 1 cm [cfg].
- pose2d: the prediction is projected with the camera (camera.project), root-centred per frame on
  the projected pelvis and flipped to image y up, then compared in pixels with obs x scale_px through
  camera.keypoint_nll (sigma0 3 px [cfg], confidence 1: the 2D contract has none).
- root: the pelvis track relative to its window mean (valid samples), the observation turned by the
  window's yaw (given, for example the pose3d yaw, or fitted on the track itself).
"""
import math

import numpy as np
import torch

from ..body.kinematics import check_precision
from ..body.rotations import Z_UP_TO_Y_UP
from ..body.joints import PARENTS
from . import camera

ROOT, NECK, HEAD = 0, 1, 2           # pelvis, neck and head in canonical order
SEL = camera.SMPL_TO_CANONICAL_15
# SMPL joint whose global rotation turns each canonical joint's offset: the segment that carries the
# joint (its SMPL parent), where the landmarks of the sources sit (ASIS on the pelvis, epicondyles on
# the thigh and upper arm, malleoli on the shank, acromion on the collar); the pelvis and the head in
# their own frame (the sources' head is a skull landmark: nose, forehead, head markers, and the skull
# turns with SMPL joint 15, not with the neck segment that carries the joint-15 position)
CANONICAL_PARENT = [k if (PARENTS[k] < 0 or c == HEAD) else PARENTS[k] for c, k in enumerate(SEL)]
SIGMA_FLOOR_M = 0.01
SIGMA0_PX = 3.0


# ---------------------------------------------------------------- canonical normalization
def _as_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _normalize(coords, valid, flip_y):
    """MoveAtlas build_canonical.normalize on (T, 15, C) coords with NaN for undetected joints."""
    c = _as_numpy(coords).astype(float).copy()
    if valid is not None:
        c[~_as_numpy(valid).astype(bool)] = np.nan
    ok = ~np.isnan(c).any(axis=-1)
    root_ok = np.isfinite(c[:, ROOT, :]).all(axis=-1)
    torso = np.linalg.norm(c[:, ROOT, :] - c[:, NECK, :], axis=-1)
    good = torso[np.isfinite(torso) & (torso > 1e-6)]
    scale = float(np.median(good)) if good.size else 1.0
    c = c - c[:, ROOT:ROOT + 1, :]
    if flip_y:
        c[:, :, 1] = -c[:, :, 1]
    c = c / scale
    ok = ok & root_ok[:, None]
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c, scale, ok


def normalize_3d(joints, valid=None):
    """joints (T, 15, 3) in the source unit (m), NaN or valid (T, 15) False for a missing joint ->
    coords (T, 15, 3) float32 canonical torso units, scale (float, source unit per torso unit),
    valid (T, 15) bool. Equals the MoveAtlas 3D normalization (module docstring)."""
    c, scale, ok = _normalize(joints, valid, flip_y=False)
    return c.astype(np.float32), scale, ok


def normalize_2d(uv, valid=None):
    """uv (T, 15, 2) px, image y down, NaN or valid (T, 15) False for a missing joint -> coords (T,
    15, 3) float32 with image y flipped up and z = 0, scale (float, px per torso unit), valid (T, 15)
    bool. Equals the MoveAtlas 2D normalization (module docstring)."""
    c, scale, ok = _normalize(uv, valid, flip_y=True)
    out = np.zeros(c.shape[:2] + (3,), dtype=np.float32)
    out[..., :2] = c
    return out, scale, ok


def to_canonical15(pos, R_glob=None, offsets=None):
    """pos (..., 22, 3) m -> (..., 15, 3) m: the SMPL joints of camera.SMPL_TO_CANONICAL_15 plus,
    when offsets (..., 15, 3) m is given, each joint's per-source offset in the frame of the segment
    that carries it (CANONICAL_PARENT, rotated by R_glob (..., 22, 3, 3); the pelvis and the head in
    their own frame). Offsets are in the SMPL rest frame of that segment (x toward the subject's left,
    y up, z forward) and say where a source defines the joint: gait120 hips are ASIS markers, its knees
    and ankles lateral markers, a video tracker's head is the nose."""
    p = pos[..., SEL, :]
    if offsets is None:
        return p
    if R_glob is None:
        raise ValueError("offsets need the global rotations R_glob (..., 22, 3, 3)")
    Rp = R_glob[..., CANONICAL_PARENT, :, :]
    return p + (Rp @ offsets.to(dtype=p.dtype, device=p.device)[..., None])[..., 0]


# ---------------------------------------------------------------- per-window helpers
def _n_windows(w, n_windows):
    return int(w.max()) + 1 if n_windows is None else int(n_windows)


def _per_point(x, w, like):
    """A scalar or a (W,) per-window value -> a 0-dim or (N,) tensor of like's dtype and device."""
    x = x.to(dtype=like.dtype, device=like.device) if torch.is_tensor(x) else \
        torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return x[w] if x.dim() > 0 else x


def _bc(x, n):
    """Append n singleton dimensions to an (N,) tensor; a 0-dim tensor stays as it is."""
    return x.reshape(x.shape + (1,) * n) if x.dim() > 0 else x


def _mask(valid, shape, like):
    if valid is None:
        return torch.ones(shape, dtype=like.dtype, device=like.device)
    return valid.to(dtype=like.dtype, device=like.device)


def _term_mask(valid, like):
    """(N, 15) weights of the per-joint likelihood terms: valid, with the pelvis (root-centred on both
    sides, residual zero by construction) left out."""
    m = _mask(valid, like.shape[:2], like)
    keep = torch.ones(m.shape[-1], dtype=m.dtype, device=m.device)
    keep[ROOT] = 0
    return m * keep


def window_sum(x, w, n_windows):
    """x (N, ...) -> (W, ...): the sum over the points of each window."""
    return torch.zeros((n_windows,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device).index_add(0, w, x)


def window_mean(x, m, w, n_windows):
    """x (N, D), weights m (N,) -> (W, D) weighted mean per window (0 for a window without weight)."""
    s = window_sum(x * m[:, None], w, n_windows)
    n = window_sum(m, w, n_windows)
    return s / torch.where(n > 0, n, torch.ones_like(n))[:, None]


def yaw_fit(p, o, m, w, n_windows):
    """Closed-form yaw per window: cos and sin (W,) of the theta about +y that minimizes sum m |rot_y(
    theta) o - p|^2 over the points p, o (N, J, 3) of each window with weights m (N, J):
    theta = atan2(B, A) with A = sum m (p_x o_x + p_z o_z) and B = sum m (p_x o_z - p_z o_x)
    (rot_y of mdfm.body.rotations). A window with A = B = 0 gets theta = 0."""
    A = window_sum((m * (p[..., 0] * o[..., 0] + p[..., 2] * o[..., 2])).sum(-1), w, n_windows)
    B = window_sum((m * (p[..., 0] * o[..., 2] - p[..., 2] * o[..., 0])).sum(-1), w, n_windows)
    n2 = A * A + B * B
    ok = n2 > 0
    n = torch.sqrt(torch.where(ok, n2, torch.ones_like(n2)))
    return torch.where(ok, A / n, torch.ones_like(A)), torch.where(ok, B / n, torch.zeros_like(B))


def _turn(o, c, s):
    """rot_y applied to o (N, J, 3) with cos and sin (N,) per point."""
    c, s = c[:, None], s[:, None]
    return torch.stack([c * o[..., 0] + s * o[..., 2], o[..., 1], -s * o[..., 0] + c * o[..., 2]], -1)


def _yaw_cs(yaw, like, n_windows):
    th = yaw.to(dtype=like.dtype, device=like.device) if torch.is_tensor(yaw) else \
        torch.as_tensor(yaw, dtype=like.dtype, device=like.device)
    th = th.expand(n_windows) if th.dim() == 0 else th
    return torch.cos(th), torch.sin(th)


def student_t_nll(r2, sigma, nu, dim):
    """-log p of an isotropic dim-variate Student-t with scale sigma and nu degrees of freedom at the
    squared distance r2 (same unit as sigma^2)."""
    const = math.lgamma((nu + dim) / 2) - math.lgamma(nu / 2) - dim / 2 * math.log(nu * math.pi)
    return -const + dim * torch.log(sigma) + (nu + dim) / 2 * torch.log1p(r2 / (nu * sigma * sigma))


def _sigma(log_sigma, floor, w, like):
    """sqrt(exp(log_sigma)^2 + floor^2) per point (N,) or 0-dim; the floor alone without log_sigma."""
    if log_sigma is None:
        return torch.as_tensor(floor, dtype=like.dtype, device=like.device)
    ls = _per_point(log_sigma, w, like)
    return torch.sqrt(torch.exp(2 * ls) + floor * floor)


# ---------------------------------------------------------------- pose 3D
def pose3d_residual(pred, obs, scale, w, n_windows=None, *, unit_to_si=1.0, valid=None, z_up=False,
                    yaw=None, weight=None):
    """pred (N, 15, 3) m window frame, obs (N, 15, 3) canonical torso units -> residual (N, 15, 3) m
    (root-centred prediction minus the aligned observation) and yaw (W,) rad, fitted in closed form
    (yaw_fit, on valid x weight) unless given. scale, unit_to_si: per window; z_up: bool or (W,) bool,
    obs z-up (rotated by Z_UP_TO_Y_UP before the yaw)."""
    check_precision(pred, obs)
    W = _n_windows(w, n_windows)
    o = obs * _bc(_per_point(scale, w, pred) * _per_point(unit_to_si, w, pred), 2)
    zu = torch.as_tensor(z_up, dtype=torch.bool, device=pred.device)
    zu = zu[w] if zu.dim() > 0 else zu
    if bool(zu.any()):
        o = torch.where(_bc(zu, 2), o @ Z_UP_TO_Y_UP.to(o.dtype).to(o.device).T, o)
    p = pred - pred[:, ROOT:ROOT + 1]
    if yaw is None:
        m = _mask(valid, pred.shape[:2], pred)
        if weight is not None:
            m = m * weight[:, None]
        c, s = yaw_fit(p, o, m, w, W)
    else:
        c, s = _yaw_cs(yaw, pred, W)
    return p - _turn(o, c[w], s[w]), torch.atan2(s, c)


def pose3d_nll(pred, obs, scale, unit_to_si, valid, log_sigma, w, n_windows=None, *, nu=4.0, z_up=False,
               weight=None, sigma_floor=SIGMA_FLOOR_M, yaw=None, return_yaw=False):
    """3D pose negative log-likelihood per window (W,) (module docstring): Student-t (nu) per valid
    joint other than the pelvis on the pose3d_residual, sigma from log_sigma (per window, log m; None:
    the floor). With return_yaw, also the per-window yaw (W,) rad."""
    r, th = pose3d_residual(pred, obs, scale, w, n_windows, unit_to_si=unit_to_si, valid=valid, z_up=z_up,
                            yaw=yaw, weight=weight)
    m = _term_mask(valid, pred)
    nll = student_t_nll((r * r).sum(-1), _bc(_sigma(log_sigma, sigma_floor, w, pred), 1), nu, 3)
    per = torch.where(m != 0, nll * m, torch.zeros_like(nll)).sum(-1)
    if weight is not None:
        per = per * weight
    out = window_sum(per, w, th.shape[0])
    return (out, th) if return_yaw else out


# ---------------------------------------------------------------- pose 2D
def _pose2d_pair(pred, cam, obs, scale_px, w):
    check_precision(pred, obs)
    c = cam.to(pred.dtype, pred.device).at(w)
    uv, _ = camera.project(pred, c.K, c.R, c.t, c.k1, c.k2)
    flip = torch.tensor([1.0, -1.0], dtype=pred.dtype, device=pred.device)
    rel = (uv - uv[:, ROOT:ROOT + 1]) * flip
    return rel, obs[..., :2] * _bc(_per_point(scale_px, w, pred), 2)


def pose2d_residual(pred, cam, obs, scale_px, w):
    """pred (N, 15, 3) m window frame, cam a camera.Camera (one, or one per window), obs (N, 15, 3)
    canonical torso units (z masked) -> residual (N, 15, 2) px: projected, pelvis-centred, image y up
    prediction minus obs x scale_px (per window)."""
    rel, target = _pose2d_pair(pred, cam, obs, scale_px, w)
    return rel - target


def pose2d_nll(pred, cam, obs, scale_px, valid, w, n_windows=None, *, conf=None, sigma0=SIGMA0_PX, nu=4.0,
               weight=None):
    """2D pose negative log-likelihood per window (W,) (module docstring): camera.keypoint_nll in
    pixels on valid joints other than the pelvis; conf (N, 15) or None (1, the 2D contract has no
    confidence field); sigma0 px, a float or per window (W,)."""
    W = _n_windows(w, n_windows)
    rel, target = _pose2d_pair(pred, cam, obs, scale_px, w)
    m = _term_mask(valid, pred)
    nll = camera.keypoint_nll(rel, target, conf, m, _bc(_per_point(sigma0, w, pred), 1), nu, reduce="none")
    per = nll.sum(-1)
    if weight is not None:
        per = per * weight
    return window_sum(per, w, W)


# ---------------------------------------------------------------- root track
def root_nll(pelvis_pred, root_obs, valid, w, n_windows=None, *, log_sigma=None, nu=4.0, yaw=None, weight=None,
             sigma_floor=SIGMA_FLOOR_M, return_yaw=False):
    """Root-track negative log-likelihood per window (W,): pelvis_pred (N, 3) m window frame and
    root_obs (N, 3) m y up, each relative to its window mean over the valid samples (valid (N,)), the
    observation turned by yaw (W,) rad (None: fitted in closed form on the track); Student-t (nu) in
    3D with sigma from log_sigma (per window, log m; None: the floor)."""
    check_precision(pelvis_pred, root_obs)
    W = _n_windows(w, n_windows)
    m = _mask(valid, pelvis_pred.shape[:1], pelvis_pred)
    mw = m if weight is None else m * weight
    p = pelvis_pred - window_mean(pelvis_pred, mw, w, W)[w]
    o = root_obs - window_mean(root_obs, mw, w, W)[w]
    if yaw is None:
        c, s = yaw_fit(p[:, None], o[:, None], mw[:, None], w, W)
    else:
        c, s = _yaw_cs(yaw, pelvis_pred, W)
    r = p - _turn(o[:, None], c[w], s[w])[:, 0]
    nll = student_t_nll((r * r).sum(-1), _sigma(log_sigma, sigma_floor, w, pelvis_pred), nu, 3)
    per = torch.where(m != 0, nll * m, torch.zeros_like(nll))
    if weight is not None:
        per = per * weight
    out = window_sum(per, w, W)
    return (out, torch.atan2(s, c)) if return_yaw else out
