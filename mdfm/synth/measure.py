"""Measurements that read an operator's quantity back from a body, the same way on a truth body and on a body
decoded by the model (tests/test_synth_pathology.py, scripts/sp3_decoder_prior_probe.py).

- dominant_frequency: the peak of the Hann-windowed periodogram of a (T,) or (T, d) signal (trace over the
  components, mean removed) inside a band, zero-padded 16 times and refined by a parabola on the logarithm of
  the power at the peak bin and its two neighbours (the Gaussian interpolation of DR-11.4).
- band_amplitude_pp: 2 sqrt(2) times the square root of the band power of the same periodogram in [f - delta,
  f + delta], the peak-to-peak amplitude of a sinusoid of the same RMS (DR-11.5); the periodogram is scaled so
  that its sum over all frequencies equals the mean square of the windowed signal over the window's mean
  square, so a whole-window sinusoid of amplitude A reads 2A.
- chord_pp: the peak-to-peak extent of a point path along its first principal axis (the chord between the
  extremes of a symmetric arc).
- band_power and freeze_ratio: P[3, 8] / P[0.5, 3] (Moore et al. 2008; DR-10.1 (b)) on the trapezoid integral
  of a Hann periodogram, of the vertical acceleration of a joint.
- cycles: movement cycles of a bradykinesia offset o(t) that leaves a constant rest extreme and returns to it
  (operator bradykinesia): a cycle is a run of |o| above 1e-6 of its maximum, its amplitude the largest |o|
  in the run, its period the run's duration; pauses are the runs at rest between cycles. This is DR-12.1
  with the sign of s chosen so that the constant rest extreme is the maximum.
- decrement_dec: DR-12.3, DEC = -100 b / a of the least-squares line A_c = a + b c over cycles c = 0 to C - 1.
- segment_deviation: the angle (deg) of G_base^T G_out per knot for one segment (DR-14.2 with the level-0 body
  as the norm).
"""
import math

import numpy as np
import torch

from .body import F64, rotation_angle_deg


def _detrend(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x - x.mean(0, keepdims=True)


def periodogram(x, rate, pad=16):
    """(f (F,), power (F,)) of the Hann-windowed (T,) or (T, d) signal, trace over components, one-sided, scaled
    so that sum(power) over all bins equals the windowed mean square over mean(w^2) (module docstring)."""
    x = _detrend(x)
    T = x.shape[0]
    w = np.hanning(T)
    n = int(2 ** math.ceil(math.log2(T * pad)))
    X = np.fft.rfft(x * w[:, None], n=n, axis=0)
    p = (np.abs(X) ** 2).sum(1)
    p[1:-1] *= 2.0
    p /= (T * (w ** 2).sum()) * (n / T)
    return np.fft.rfftfreq(n, 1.0 / rate), p


def dominant_frequency(x, rate, band=(3.0, 12.0)):
    """Peak frequency (Hz) inside band, Gaussian-interpolated (module docstring)."""
    f, p = periodogram(x, rate)
    sel = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    i = sel[np.argmax(p[sel])]
    if 0 < i < len(p) - 1 and min(p[i - 1], p[i], p[i + 1]) > 0:
        a, b, c = np.log(p[i - 1]), np.log(p[i]), np.log(p[i + 1])
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(f[i] + d * (f[1] - f[0]))
    return float(f[i])


def band_amplitude_pp(x, rate, f_peak, delta=1.0):
    """Peak-to-peak amplitude of a sinusoid with the band power of x in [f_peak - delta, f_peak + delta]."""
    f, p = periodogram(x, rate)
    sel = (f >= f_peak - delta) & (f <= f_peak + delta)
    return float(2.0 * math.sqrt(2.0) * math.sqrt(p[sel].sum()))


def chord_pp(points):
    """Peak-to-peak extent (same unit) of a (T, 3) path along its first principal axis."""
    x = _detrend(points)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    s = x @ vt[0]
    return float(s.max() - s.min())


def band_power(x, rate, band):
    """Trapezoid integral over band of a Hann periodogram density (unpadded, mean removed) of a (T,) signal."""
    x = _detrend(x)[:, 0]
    T = len(x)
    w = np.hanning(T)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(T, 1.0 / rate)
    psd = np.abs(X) ** 2 / (rate * (w ** 2).sum())
    psd[1:-1] *= 2.0
    sel = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(psd[sel], f[sel])) if sel.sum() > 1 else 0.0


def freeze_ratio(acc_vertical, rate, freeze=(3.0, 8.0), loco=(0.5, 3.0)):
    """P[freeze] / P[loco] of a vertical acceleration (DR-10.1 (b))."""
    lo = band_power(acc_vertical, rate, loco)
    return float(band_power(acc_vertical, rate, freeze) / lo) if lo > 0 else math.inf


def vertical_acceleration(joints, joint, rate):
    """(T,) m/s^2: second central difference of a joint's height."""
    y = joints[:, joint, 1].numpy() if torch.is_tensor(joints) else np.asarray(joints)[:, joint, 1]
    return np.gradient(np.gradient(y, 1.0 / rate), 1.0 / rate)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def cycles(o, rate, rel_tol=1e-6):
    """{'amplitude' (C,), 'period_s' (C,), 'speed' (C,) mean |do/dt| per cycle, 'start_s' (C,), 'pauses':
    [(start_s, duration_s)] between cycles} of a movement offset o (T,) that rests at 0 (module docstring)."""
    o = np.asarray(o, dtype=np.float64)
    a = np.abs(o)
    if not a.max() > 0:
        return {"amplitude": np.zeros(0), "period_s": np.zeros(0), "speed": np.zeros(0), "start_s": np.zeros(0),
                "pauses": []}
    runs = [(s, e) for s, e in _runs(a > rel_tol * a.max())]
    amp, per, spd, st = [], [], [], []
    v = np.abs(np.gradient(o, 1.0 / rate))
    for s, e in runs:
        amp.append(a[s:e].max())
        per.append((e - s + 1) / rate)            # from the last rest knot before to the first after
        spd.append(v[s:e].mean())
        st.append((s - 1) / rate)
    pauses = []
    for (s0, e0), (s1, e1) in zip(runs, runs[1:]):
        gap = (s1 - e0 - 1) / rate               # rest knots strictly between two cycles, as a duration
        if gap > 0:
            pauses.append(((e0) / rate, gap))
    return {"amplitude": np.array(amp), "period_s": np.array(per), "speed": np.array(spd), "start_s": np.array(st),
            "pauses": pauses}


def decrement_dec(amplitudes):
    """DR-12.3: DEC = -100 b / a (percent per cycle) of the least-squares line over cycles 0 to C - 1."""
    A = np.asarray(amplitudes, dtype=np.float64)
    if len(A) < 2:
        return math.nan
    c = np.arange(len(A), dtype=np.float64)
    b, a = np.polyfit(c, A, 1)
    return float(-100.0 * b / a) if a != 0 else math.nan


def slope(values, x=None):
    """Least-squares slope of values against x (default 0, 1, ...)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return math.nan
    x = np.arange(len(v), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    return float(np.polyfit(x, v, 1)[0])


def segment_deviation(G_base, G_out):
    """(T,) deg: the rotation angle of G_base^T G_out (one segment's global rotations (T, 3, 3))."""
    Gb = torch.as_tensor(G_base, dtype=F64)
    return rotation_angle_deg(Gb.transpose(-1, -2) @ torch.as_tensor(G_out, dtype=F64)).numpy()


def heading_relative_joints(joints, R_root):
    """(T, 22, 3) m: joints relative to the pelvis in each knot's heading frame (yaw of the root removed)."""
    from ..body.rotations import rot_y, yaw_angle
    J = torch.as_tensor(joints, dtype=F64)
    H = rot_y(-yaw_angle(torch.as_tensor(R_root, dtype=F64)))
    return torch.einsum("tij,tkj->tki", H, J - J[:, :1])
