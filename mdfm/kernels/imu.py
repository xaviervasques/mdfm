"""IMU kernel (design section 3): a sensor rigidly attached to a body segment.

For a sensor on the segment of joint k at offset r (segment frame) with mounting rotation R_s
(sensor frame to segment frame):
  acc  = (R_k R_s)^T (p_ddot_sensor - g) + bias_acc      (specific force, m/s^2)
  gyro = (R_k R_s)^T omega_world + bias_gyro             (rad/s)
where p_ddot_sensor is the second time derivative of the sensor point, which equals
p_ddot_k + omega_dot x (R_k r) + omega x (omega x (R_k r)). Here it is obtained exactly by
differentiating the point trajectory, so the formula holds by construction; the unit tests
check the decomposition against it. gravity_removed=True drops the -g term (sources that
ship gravity-free acceleration, such as pads). The world is y-up with g = (0, -9.81, 0).

Two forms. imu() reads one sensor on a BodyTrajectory in m/s^2 and rad/s. The batched form
reads S sites at the N query points of a micro-step from one kinematics evaluation
(mdfm.body.kinematics.KnotBatch.kinematics, then point_kin): sensor_readings() in m/s^2 and
rad/s; imu_predict() adds the optional per-site FIR (identity by default), gains and biases and
returns the units of the canonical contract (configs/canonical_imu.yaml), acc in g with G0 =
9.80665 m/s^2 per g and gyro in deg/s, as factory.imu_render.render does; a standing body reads
9.81 / 9.80665 g. imu_nll() is the Student-t likelihood of the observed channels, summed per
window. The batched functions refuse half precision and an active autocast (TypeError,
RuntimeError), like the kinematics.
"""
import math

import torch

from ..body.kinematics import check_precision, point_kin

G = 9.81
G0 = 9.80665  # the canonical contract converts m/s^2 to g with standard gravity (factory.imu_render.G0)
FLOOR_ACC_G = 0.01    # [cfg] floor of the acc noise scale of imu_nll
FLOOR_GYRO_DPS = 1.0  # [cfg] floor of the gyro noise scale of imu_nll


def gravity(dtype=torch.float32, device=None, up="y"):
    g = torch.zeros(3, dtype=dtype, device=device)
    g[{"x": 0, "y": 1, "z": 2}[up]] = -G
    return g


def imu(traj, t, joint, offset, mount=None, gravity_removed=False, up="y",
        bias_acc=None, bias_gyro=None):
    """Virtual IMU readings at times t (N,): acc (N, 3) in m/s^2 and gyro (N, 3) in rad/s,
    in the sensor frame. traj is a BodyTrajectory whose world uses the given up axis."""
    p, v, a, R, w, wdot = traj.point_kinematics(t, joint, offset)
    Rs = R if mount is None else R @ mount
    f = a if gravity_removed else a - gravity(a.dtype, a.device, up)
    acc = (Rs.transpose(-1, -2) @ f[..., None])[..., 0]
    gyro = (Rs.transpose(-1, -2) @ w[..., None])[..., 0]
    if bias_acc is not None:
        acc = acc + bias_acc
    if bias_gyro is not None:
        gyro = gyro + bias_gyro
    return acc, gyro


# ---------------------------------------------------------------- batched kernel

def _tmv(R, x):
    """R^T x for (..., 3, 3) and (..., 3), as a broadcast sum."""
    return (R * x[..., :, None]).sum(-2)


def sensor_readings(pk, mount=None, gravity_removed=False):
    """Specific force (N, S, 3) m/s^2 and angular rate (N, S, 3) rad/s in the sensor frames of S
    sites at N points. pk: PointKin of the sites (mdfm.body.kinematics.point_kin), world y up;
    mount: (S, 3, 3) or (N, S, 3, 3) sensor to segment, None for identity; gravity_removed: bool,
    or a bool tensor broadcastable to (N, S) that drops -g where True."""
    g = gravity(pk.a.dtype, pk.a.device)
    if isinstance(gravity_removed, torch.Tensor):
        keep = (~gravity_removed.to(device=pk.a.device, dtype=torch.bool)).to(pk.a.dtype)
        f = pk.a - keep[..., None] * g
    else:
        f = pk.a if gravity_removed else pk.a - g
    acc, gyro = _tmv(pk.R, f), _tmv(pk.R, pk.omega)
    if mount is not None:
        acc, gyro = _tmv(mount, acc), _tmv(mount, gyro)
    return acc, gyro


def fir_filter(x, taps, segment_len):
    """Causal FIR along consecutive samples: x (N, S, 3) holds N / segment_len segments of
    segment_len consecutive samples each (site axis second); taps (S, K), y[j] = sum_k taps[k]
    x[j - k] within a segment, the segment's first sample standing for the samples before it.
    taps [1] (or [1, 0, ...]) is the identity; the DC gain is the sum of the taps."""
    N, S = x.shape[:2]
    if taps.dim() != 2 or taps.shape[0] != S:
        raise ValueError(f"FIR taps must be (S, K) with S = {S}, got {tuple(taps.shape)}")
    if segment_len is None or segment_len < 1 or N % segment_len:
        raise ValueError(f"a FIR needs segment_len dividing N = {N}, got {segment_len}")
    K, L = taps.shape[1], int(segment_len)
    xs = x.reshape(N // L, L, S, x.shape[-1])
    xp = torch.cat([xs[:, :1].expand(-1, K - 1, -1, -1), xs], 1)
    taps = taps.to(x.dtype)
    y = sum(taps[None, None, :, k, None] * xp[:, K - 1 - k:K - 1 - k + L] for k in range(K))
    return y.reshape(x.shape)


def imu_predict(kin, site_joint, offset, mount=None, bias_acc=None, bias_gyro=None, fir=None, *,
                gain_acc=None, gain_gyro=None, gravity_removed=False, segment_len=None):
    """Predicted IMU channels of S sites at the N query points of kin (BodyKin): acc (N, S, 3) in
    g and gyro (N, S, 3) in deg/s, sensor frames.

    site_joint (S,) int64 segment joint of each site; offset (S, 3) or (N, S, 3) m in the segment
    frame (factory.imu_render.site_table plus the nuisance offset); mount (S, 3, 3) or (N, S, 3,
    3) sensor to segment, None for identity; bias_acc in g and bias_gyro in deg/s, broadcastable
    to (N, S, 3); gain_acc and gain_gyro multiplicative, broadcastable to (N, S, 3); fir None
    (identity) or causal taps (S, K), or (S, 2, K) with separate acc and gyro taps, applied to the
    physical signal before gain and bias along segments of segment_len consecutive samples
    (fir_filter); gravity_removed as in sensor_readings. Order: acc = gain x FIR(specific force)
    / G0 + bias, gyro = gain x FIR(rate) in deg/s + bias."""
    extras = [x for x in (offset, mount, bias_acc, bias_gyro, gain_acc, gain_gyro, fir)
              if isinstance(x, torch.Tensor)]
    check_precision(kin.a, *extras)
    pk = point_kin(kin, site_joint, offset)
    acc, gyro = sensor_readings(pk, mount, gravity_removed)
    if fir is not None:
        if not (fir.dim() == 2 or (fir.dim() == 3 and fir.shape[1] == 2)):
            raise ValueError(f"FIR taps must be (S, K) or (S, 2, K), got {tuple(fir.shape)}")
        taps_a, taps_g = (fir, fir) if fir.dim() == 2 else (fir[:, 0], fir[:, 1])
        acc, gyro = fir_filter(acc, taps_a, segment_len), fir_filter(gyro, taps_g, segment_len)
    acc = acc / G0
    gyro = torch.rad2deg(gyro)
    if gain_acc is not None:
        acc = acc * gain_acc
    if gain_gyro is not None:
        gyro = gyro * gain_gyro
    if bias_acc is not None:
        acc = acc + bias_acc
    if bias_gyro is not None:
        gyro = gyro + bias_gyro
    return acc, gyro


def student_t_nll(r, sigma, nu):
    """-log of the Student-t density with scale sigma and nu degrees of freedom at residual r."""
    c = math.lgamma(nu / 2) - math.lgamma((nu + 1) / 2) + 0.5 * math.log(nu * math.pi)
    return c + torch.log(sigma) + 0.5 * (nu + 1) * torch.log1p((r / sigma) ** 2 / nu)


def imu_nll(pred, obs, valid, log_scale, weight, w, n_windows, nu=4.0,
            floor_acc=FLOOR_ACC_G, floor_gyro=FLOOR_GYRO_DPS):
    """Student-t negative log-likelihood per window, (n_windows,).

    pred: (N, S, 6), or the (acc, gyro) pair of imu_predict, in g and deg/s; obs (N, S, 6) in the
    same units (any value, NaN included, where valid is False); valid (N, S, 6) bool, or
    broadcastable; log_scale (N, S, 2): the scale is floor + exp(log_scale), floors floor_acc g
    and floor_gyro deg/s; weight (N,) the point weights (n_valid / n_sampled of the point's
    stream); w (N,) int64 window of each point. Masked channels contribute zero, to the value and
    to the gradient."""
    if isinstance(pred, (tuple, list)):
        pred = torch.cat(pred, -1)
    check_precision(pred, log_scale)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=pred.device).expand_as(pred)
    obs = torch.where(valid, obs.to(pred.dtype), torch.zeros((), dtype=pred.dtype, device=pred.device))
    floor = torch.tensor([floor_acc, floor_gyro], dtype=pred.dtype, device=pred.device)
    sigma = (floor + torch.exp(log_scale.to(pred.dtype))).repeat_interleave(3, -1)
    nll = student_t_nll(torch.where(valid, pred - obs, torch.zeros_like(pred)), sigma, nu)
    per_point = torch.where(valid, nll, torch.zeros_like(nll)).sum((-1, -2)) * weight.to(pred.dtype)
    out = torch.zeros(n_windows, dtype=pred.dtype, device=pred.device)
    return out.index_add(0, w.to(pred.device), per_point)
