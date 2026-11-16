"""Foot-sensor strapdown integration with zero-velocity updates (ZUPT), version 2 of task SCALE (F2 lever 5,
docs/proposals/F2_LEVER5_DRAFT.md section 3; task L5).

The stride length of one foot from that foot sensor's specific force and angular rate alone: no model, no fitted
parameter, no walkway value. This is reports/bench/scale/code/zupt2.py (SHA-256 b0b399f3..., the version-2 method of
test Z, docs/notes/SCALE_2026-10-15.md section 2) moved into the package: the same constants and the same arithmetic,
the orientation increments and the rotation of the specific force computed in arrays (tests/test_zupt.py holds the
stride lengths and instants equal to zupt2.py's within 1e-9 m on analytic and noisy trajectories). Version 2 (decided
on the synthetic renders and analytic tests after the METHOD INVALID reading of version 1, amendment 1 of the SCALE
prediction): the tilt at the start of a stride from the specific force of the TILT_HALF samples on each side of m_a,
and no gyro bias (version 1 took both from the whole still run, which holds real rotation up to W_STILL_DPS). The
constants were fixed on analytic trajectories and the synthetic renders, never on a real walkway value.

Method (Foxlin 2005; Skog et al. 2010 in its simplest per-stride form):
- still samples: |w| < W_STILL_DPS and | |f| - 1 g | < F_STILL_G, both on a centred moving median of STILL_SMOOTH
  samples; a sample with a non-finite value is never still; still runs: maximal runs of still samples at least
  MIN_STILL_S long;
- one stride per pair of consecutive still runs (a, b) of one foot, from the sample m_a of minimum |w| inside run a
  to the sample m_b of minimum |w| inside run b, when m_b - m_a lies in [MIN_STRIDE_S, MAX_STRIDE_S];
- at m_a: the gravity direction from the mean specific force of the samples max(a0, m_a - TILT_HALF) to
  min(a1, m_a + TILT_HALF + 1) (tilt), yaw 0; the orientation integrated from m_a by the exact exponential of each
  sample interval's mean rate (body frame, right multiplication);
- world acceleration a = R f - g z; velocity by the trapezoid from 0 at m_a; the velocity at m_b removed linearly in
  time; position by the trapezoid; stride = horizontal norm of the position at m_b.
Per stride also: the peak horizontal speed, the swing time (the samples between the end of run a and the start of run
b), the duration m_b - m_a, the end velocity removed (a quality figure), the path length and, added here, turn_deg:
the heading change of the sensor between m_a and m_b (the yaw of R(m_b) R(m_a)^T about the vertical, degrees; near 0
on a straight stride), which zupt2.py did not report.
Units in: acc in g with gravity (specific force, +1 g up at rest), gyro in deg/s, any fixed sensor mounting.
"""
import math

import numpy as np

VERSION = ("zupt-v2 (reports/bench/scale/code/zupt2.py, SHA-256 "
           "b0b399f37a9a8558d2e25747bf0cc834a454050d74a53e06cfae27f22c19c5e5)")
G0 = 9.80665
W_STILL_DPS = 45.0
F_STILL_G = 0.12
STILL_SMOOTH = 5
MIN_STILL_S = 0.05
MAX_STRIDE_S = 2.5
MIN_STRIDE_S = 0.6
TILT_HALF = 2
CONSTANTS = {"W_STILL_DPS": W_STILL_DPS, "F_STILL_G": F_STILL_G, "STILL_SMOOTH": STILL_SMOOTH,
             "MIN_STILL_S": MIN_STILL_S, "MAX_STRIDE_S": MAX_STRIDE_S, "MIN_STRIDE_S": MIN_STRIDE_S,
             "TILT_HALF": TILT_HALF}


def _movmed(x, k):
    if k <= 1:
        return x
    n = len(x)
    h = k // 2
    pad = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
    idx = np.arange(n)[:, None] + np.arange(k)[None, :]
    return np.median(pad[idx], axis=1)


def still_runs(acc_g, gyro_dps, fps, w_still=W_STILL_DPS, f_still=F_STILL_G, smooth=STILL_SMOOTH,
               min_still_s=MIN_STILL_S):
    """([(start, end_exclusive)] of the still runs, |w| (n,) deg/s with inf at a non-finite sample)."""
    acc = np.asarray(acc_g, np.float64)
    gyr = np.asarray(gyro_dps, np.float64)
    if len(acc) == 0:
        return [], np.zeros(0)
    ok = np.isfinite(acc).all(1) & np.isfinite(gyr).all(1)
    wn = np.where(ok, np.linalg.norm(np.nan_to_num(gyr), axis=1), np.inf)
    fn = np.where(ok, np.abs(np.linalg.norm(np.nan_to_num(acc), axis=1) - 1.0), np.inf)
    wn_s = _movmed(np.where(np.isfinite(wn), wn, 1e9), smooth)
    fn_s = _movmed(np.where(np.isfinite(fn), fn, 1e9), smooth)
    still = (wn_s < w_still) & (fn_s < f_still) & ok
    m = max(1, int(round(min_still_s * fps)))
    d = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= m]
    return runs, wn


def _tilt(f_mean):
    """Rotation R (sensor to world, world z up) that maps the mean specific force to +z, yaw 0."""
    u = f_mean / np.linalg.norm(f_mean)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), float(np.dot(u, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _exp_so3_batch(phi):
    """Exact exponentials (n, 3, 3) of rotation vectors phi (n, 3) (Rodrigues; identity below 1e-12 rad)."""
    th = np.linalg.norm(phi, axis=1)
    out = np.tile(np.eye(3), (len(phi), 1, 1))
    big = th >= 1e-12
    if big.any():
        k = phi[big] / th[big, None]
        K = np.zeros((int(big.sum()), 3, 3))
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        s, c = np.sin(th[big])[:, None, None], np.cos(th[big])[:, None, None]
        out[big] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return out


def integrate(acc_g, gyro_dps, fps, run_a, run_b, wn):
    """One stride between still runs a and b (module docstring): dict, or None when a sample is not finite."""
    a0, a1 = run_a
    b0, b1 = run_b
    ma = a0 + int(np.argmin(wn[a0:a1]))
    mb = b0 + int(np.argmin(wn[b0:b1]))
    f = np.asarray(acc_g[ma:mb + 1], np.float64) * G0
    w = np.deg2rad(np.asarray(gyro_dps[ma:mb + 1], np.float64))
    if not (np.isfinite(f).all() and np.isfinite(w).all()):
        return None
    dt = 1.0 / fps
    lo, hi = max(a0, ma - TILT_HALF), min(a1, ma + TILT_HALF + 1)
    ft = np.asarray(acc_g[lo:hi], np.float64)
    if not np.isfinite(ft).all():
        return None
    R0 = _tilt(ft.mean(0))
    n = len(f)
    E = _exp_so3_batch(0.5 * (w[:-1] + w[1:]) * dt)
    Rs = np.empty((n, 3, 3))
    R = R0
    Rs[0] = R
    for i in range(n - 1):
        R = R @ E[i]
        Rs[i + 1] = R
    aw = np.einsum("nij,nj->ni", Rs, f)
    aw[:, 2] -= G0
    v = np.zeros((n, 3))
    v[1:] = np.cumsum(0.5 * (aw[1:] + aw[:-1]) * dt, axis=0)
    t = np.arange(n) * dt
    T = t[-1]
    v_end = v[-1].copy()
    v = v - (t / T)[:, None] * v_end[None, :]
    p = np.zeros((n, 3))
    p[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)
    vh = np.linalg.norm(v[:, :2], axis=1)
    Rr = Rs[-1] @ R0.T
    return {"m_a": ma, "m_b": mb, "t_a": ma / fps, "t_b": mb / fps, "duration_s": T,
            "length_m": float(np.linalg.norm(p[-1, :2])), "vertical_m": float(p[-1, 2]),
            "peak_speed_m_s": float(vh.max()), "swing_s": (b0 - a1) / fps, "v_end_m_s": float(np.linalg.norm(v_end)),
            "path_m": float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))),
            "turn_deg": float(math.degrees(math.atan2(Rr[1, 0], Rr[0, 0])))}


def strides(acc_g, gyro_dps, fps, **kw):
    """[dict] of every stride between consecutive still runs whose duration lies in [MIN_STRIDE_S, MAX_STRIDE_S]
    (integrate's keys, with run_a and run_b)."""
    runs, wn = still_runs(acc_g, gyro_dps, fps, **kw)
    out = []
    for ra, rb in zip(runs[:-1], runs[1:]):
        ma = ra[0] + int(np.argmin(wn[ra[0]:ra[1]]))
        mb = rb[0] + int(np.argmin(wn[rb[0]:rb[1]]))
        d = (mb - ma) / fps
        if d < MIN_STRIDE_S or d > MAX_STRIDE_S:
            continue
        r = integrate(acc_g, gyro_dps, fps, ra, rb, wn)
        if r is not None:
            r["run_a"], r["run_b"] = ra, rb
            out.append(r)
    return out


def match(zs, t0, t1, lead_s=(-0.10, 0.60), dur_tol=0.15):
    """The stride of zs for a reference stride with ICs at t0 and t1 (same foot): the one whose start lies in
    [t0 + lead_s[0], t0 + lead_s[1]] and whose end lies in [t1 + lead_s[0], t1 + lead_s[1]], its duration within
    dur_tol of t1 - t0 (relative); the one whose start is nearest t0 + 0.2 s when several qualify; None otherwise.
    (A readout's segmentation keyed by a reference timing: the inertial anchor of mdfm/data/zupt_anchor.py never
    calls it.)"""
    best, bd = None, None
    for z in zs:
        if not (t0 + lead_s[0] <= z["t_a"] <= t0 + lead_s[1]):
            continue
        if not (t1 + lead_s[0] <= z["t_b"] <= t1 + lead_s[1]):
            continue
        if abs(z["duration_s"] - (t1 - t0)) > dur_tol * (t1 - t0):
            continue
        d = abs(z["t_a"] - (t0 + 0.2))
        if bd is None or d < bd:
            best, bd = z, d
    return best
