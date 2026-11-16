"""Synthetic tests of mdfm/clinical/image_plane.py (S1 image-plane descriptors, DRAFT).

Every input is a synthetic stick figure in the keypoint order of the module (POINTS) with planted segment
angles, oscillation amplitudes and frequencies, near and far side asymmetries, dwell bouts, point paths,
likelihood dropouts, rotations and scale changes. No data of any source is read.
"""
import inspect
import math
import os
import re

import numpy as np
import pytest
from scipy.signal import welch

from mdfm.clinical import image_plane as IP

FS = 25.0
MODULE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mdfm", "clinical")


def d(alpha):
    """Image direction (y down) of an angle clockwise from image up."""
    a = np.radians(alpha)
    return np.stack([np.sin(a), -np.cos(a)], axis=-1)


def figure(T, fs=FS, scale=1.0, rot_deg=0.0, centre=(320.0, 240.0), angles=None, paths=None):
    """(T, 14, 2) px: a frontal stick figure, torso 100 px (pelvis to shoulder midpoint), segment angles in the
    image frame (constants or functions of t), points overridden by paths (offsets from the pelvis in torso
    lengths), then rotated by rot_deg about the centre and scaled."""
    t = np.arange(T) / fs
    ang = {"trunk": 0.0, "upper_arm_1": 190.0, "upper_arm_2": 170.0, "forearm_1": 185.0, "forearm_2": 175.0,
           "thigh_1": 185.0, "thigh_2": 175.0, "shank_1": 180.0, "shank_2": 180.0}
    ang.update(angles or {})
    A = {k: (np.asarray(v(t), float) if callable(v) else np.full(T, float(v))) for k, v in ang.items()}
    P = np.broadcast_to(np.asarray(centre, float), (T, 2)).copy()
    N = P + 100.0 * d(A["trunk"])
    side = d(A["trunk"] + 90.0)
    pts = {"shoulder1": N - 25 * side, "shoulder2": N + 25 * side, "hip1": P - 18 * side, "hip2": P + 18 * side,
           "forehead": N + 30 * d(A["trunk"]), "chin": N + 15 * d(A["trunk"])}
    for s in "12":
        pts["elbow" + s] = pts["shoulder" + s] + 45 * d(A["upper_arm_" + s])
        pts["wrist" + s] = pts["elbow" + s] + 40 * d(A["forearm_" + s])
        pts["knee" + s] = pts["hip" + s] + 55 * d(A["thigh_" + s])
        pts["ankle" + s] = pts["knee" + s] + 50 * d(A["shank_" + s])
    for k, f in (paths or {}).items():
        pts[k] = P + 100.0 * np.asarray(f(t), float)
    xy = np.stack([pts[n] for n in IP.POINTS], axis=1)
    c = np.asarray(centre, float)
    r = math.radians(rot_deg)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return (xy - c) @ R.T * scale + c


def get(rows, desc, qty, target, window):
    m = [r for r in rows if r["descriptor"] == desc and r["quantity"] == qty and r["target"] == target
         and r["window"] == window]
    assert len(m) == 1, (desc, qty, target, window, len(m))
    return m[0]


def sine(amp, f, offset=0.0):
    return lambda t: offset + amp * np.sin(2 * np.pi * f * t)


# ---------------------------------------------------------------- building blocks
def test_window_bounds_and_halves():
    assert IP.window_bounds(501, FS) == [(0, 250), (125, 375), (250, 500)]
    assert IP.window_bounds(501, FS, 10.0, 10.0) == [(0, 250), (250, 500)]   # the two 10 s halves of S1
    assert IP.window_bounds(100, FS) == []
    assert IP.descriptors(figure(100), FS, "sitting") == []
    rows = IP.descriptors(figure(501), FS, "sitting")       # R13: midpoint of the first and last sample times
    r0, r1 = get(rows, "TON-T", "orientation_deg", "trunk", 0), get(rows, "TON-T", "orientation_deg", "trunk", 1)
    assert r0["t_start_s"] == 0.0 and r0["t_center_s"] == pytest.approx(249 / 2 / FS)       # 4.98 s
    assert r1["t_start_s"] == pytest.approx(5.0) and r1["t_center_s"] == pytest.approx(499 / 2 / FS)   # 9.98 s


@pytest.mark.parametrize("f_hz,kind,order", [(0.3, "low", 2), (0.5, "high", 2), (3.0, "low", 2), (0.3, "low", 4)])
def test_filter_minus_3db_at_named_frequency(f_hz, kind, order):
    """DR-2.4: the combined forward-backward response is -3 dB (amplitude 1/sqrt 2) at the named frequency."""
    t = np.arange(int(600 * FS)) / FS
    y = IP.zero_phase(np.sin(2 * np.pi * f_hz * t), FS, f_hz, kind, filter_order=order)
    mid = slice(len(t) // 4, 3 * len(t) // 4)
    basis = np.stack([np.sin(2 * np.pi * f_hz * t[mid]), np.cos(2 * np.pi * f_hz * t[mid])], 1)
    amp = math.hypot(*np.linalg.lstsq(basis, y[mid], rcond=None)[0])
    assert amp == pytest.approx(1 / math.sqrt(2), abs=1e-3)


def test_filter_runs_over_valid_segments_only():
    x = np.sin(2 * np.pi * 0.2 * np.arange(500) / FS)
    x[200:205] = np.nan
    x[300:305] = np.nan
    x[400:] = np.nan
    y = IP.zero_phase(x, FS, 0.3, "low")
    assert np.isnan(y[200:205]).all() and np.isnan(y[300:305]).all() and np.isnan(y[400:]).all()
    assert np.isfinite(y[:200]).all() and np.isfinite(y[205:300]).all() and np.isfinite(y[305:400]).all()
    short = np.full(50, np.nan)
    short[10:16] = 1.0                      # 6 samples: shorter than the edge padding
    assert np.isnan(IP.zero_phase(short, FS, 0.3, "low")).all()


def test_welch_matches_scipy_and_averages_valid_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=250)
    f, p, n = IP.welch_psd(x, FS)
    f0, p0 = welch(x, fs=FS, nperseg=100)
    assert n == 4 and np.allclose(f, f0) and np.allclose(p, p0)
    y = x.copy()
    y[120:130] = np.nan
    f, p, n = IP.welch_psd(y, FS)
    pa = welch(x[:120], fs=FS, nperseg=100)[1]
    pb = welch(x[130:], fs=FS, nperseg=100)[1]
    assert n == 2 and np.allclose(p, (pa + pb) / 2)
    v = rng.normal(size=(250, 2))
    assert np.allclose(IP.welch_psd(v, FS)[1], welch(v[:, 0], fs=FS, nperseg=100)[1] + welch(v[:, 1], fs=FS,
                                                                                               nperseg=100)[1])
    assert IP.welch_psd(np.full(250, np.nan), FS) == (None, None, 0)


def test_band_power_of_a_planted_sinusoid():
    t = np.arange(250) / FS
    f, p, _ = IP.welch_psd(3.0 * np.sin(2 * np.pi * 1.5 * t), FS)
    assert IP.band_power(f, p, (0.5, 3.0)) == pytest.approx(3.0 ** 2 / 2, rel=1e-6)
    assert f[np.argmax(p)] == 1.5


@pytest.mark.parametrize("method", ["central5", "central3"])
def test_derivative_gain(method):
    f = 2.0
    t = np.arange(250) / FS
    v = IP.time_derivative(np.sin(2 * np.pi * f * t), FS, derivative=method)
    expect = 2 * np.pi * f * np.cos(2 * np.pi * f * t) * IP.derivative_gain(f, FS, method)
    assert np.allclose(v[5:-5], expect[5:-5], atol=1e-9)
    assert IP.derivative_gain(f, FS, "central5") > IP.derivative_gain(f, FS, "central3")
    x = np.random.default_rng(1).normal(size=(40, 2)).cumsum(axis=0)
    if method == "central3":
        assert np.allclose(IP.time_derivative(x, FS, method), np.gradient(x, axis=0, edge_order=2) * FS, atol=1e-9)


def test_dii_fixtures_2d():
    """DR-21.5, 2D form: one axis gives 0, a planar circle 1."""
    t = np.arange(250) / FS
    line = np.stack([np.cos(2 * np.pi * t), 0 * t], 1)
    circle = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], 1)
    assert IP.dii(line) == pytest.approx(0.0, abs=1e-12)
    assert IP.dii(circle) == pytest.approx(1.0, abs=0.01)
    assert math.isnan(IP.dii(np.zeros((10, 2))))


def test_dwell_run_rules():
    fs = 25.0
    theta = np.zeros(100)
    theta[10:12] = 30.0        # 2 samples
    theta[15:25] = 30.0        # 10 samples, 3-sample gap before
    theta[60:70] = 30.0        # 10 samples
    plain = IP.dwell(theta, 20.0, fs)
    assert plain["episodes"] == 3 and plain["fraction_pct"] == pytest.approx(22.0)
    dropped = IP.dwell(theta, 20.0, fs, dwell_min_run_s=0.2)
    assert dropped["episodes"] == 2 and dropped["fraction_pct"] == pytest.approx(20.0)
    merged = IP.dwell(theta, 20.0, fs, dwell_min_run_s=0.2, dwell_merge_gap_s=0.12)
    assert merged["episodes"] == 2 and merged["fraction_pct"] == pytest.approx(25.0)
    assert merged["episode_mean_s"] == pytest.approx((15 + 10) / 2 / fs)
    assert merged["episode_rate_per_min"] == pytest.approx(2 / (100 / fs / 60))
    assert IP.dwell(np.full(10, np.nan), 1.0, fs) is None
    # a run of exactly the minimum length stays (DR-5.4: no event shorter than d_min is counted)
    edge = np.zeros(100)
    edge[10:15] = 30.0
    kept = IP.dwell(edge, 20.0, fs, dwell_min_run_s=5 / fs)
    assert kept["episodes"] == 1 and kept["fraction_pct"] == pytest.approx(5.0)
    assert IP.dwell(edge, 20.0, fs, dwell_min_run_s=6 / fs)["episodes"] == 0
    # a threshold equal to theta is not dwelling (theta > theta_95, DR-14.3)
    assert IP.dwell(np.full(50, 20.0), 20.0, fs)["fraction_pct"] == 0.0


def test_dwell_with_invalid_samples():
    """DR-14.3 as written: fraction over the valid samples, time = fraction / 100 x window length; the episode rate
    per minute of valid samples (R8). 20 invalid samples of 100, 40 dwelling samples in one run."""
    fs = 25.0
    theta = np.zeros(100)
    theta[:20] = np.nan
    theta[40:80] = 30.0
    r = IP.dwell(theta, 20.0, fs)
    assert r["fraction_pct"] == pytest.approx(50.0)
    assert r["time_s"] == pytest.approx(0.5 * 100 / fs)                  # 2.0 s, not 40 / fs = 1.6 s
    assert r["episodes"] == 1 and r["episode_mean_s"] == pytest.approx(40 / fs)
    assert r["episode_rate_per_min"] == pytest.approx(1 / (80 / fs / 60))  # 18.75, not 15 per minute
    # through descriptors: forearm_1 20 deg off its reference, wrist1 invalid for the first 5 s
    lk = np.ones((501, len(IP.POINTS)))
    lk[:125, IP.POINTS.index("wrist1")] = 0.2
    rows = IP.descriptors(figure(501), FS, "sitting", likelihood=lk, reference_deg={"forearm_1": 165.0},
                          dwell_threshold_deg=10.0)
    assert get(rows, "TON-DW", "fraction_pct", "forearm_1", 0)["valid_fraction"] == pytest.approx(0.5)
    assert get(rows, "TON-DW", "fraction_pct", "forearm_1", 0)["value"] == pytest.approx(100.0)
    assert get(rows, "TON-DW", "time_s", "forearm_1", 0)["value"] == pytest.approx(10.0)
    assert get(rows, "TON-DW", "episode_rate_per_min", "forearm_1", 0)["value"] == pytest.approx(12.0)
    assert get(rows, "TON-DW", "episode_mean_s", "forearm_1", 0)["value"] == pytest.approx(5.0)


# ---------------------------------------------------------------- tonic
def test_tonic_constant_offset():
    """DR-21.6 on the image plane: a segment held 20 deg from its reference."""
    rows = IP.descriptors(figure(501, angles={"upper_arm_1": 190.0}), FS, "sitting",
                          reference_deg={"upper_arm_1": 170.0}, dwell_threshold_deg=15.0)
    for w in range(3):
        assert get(rows, "TON-T", "deviation_deg", "upper_arm_1", w)["value"] == pytest.approx(20.0, abs=0.1)
        assert get(rows, "TON-T", "in_plane_deg", "upper_arm_1", w)["value"] == pytest.approx(20.0, abs=0.1)
        assert get(rows, "TON-DW", "fraction_pct", "upper_arm_1", w)["value"] == pytest.approx(100.0)
        assert get(rows, "TON-DW", "time_s", "upper_arm_1", w)["value"] == pytest.approx(10.0)
    rows = IP.descriptors(figure(501, angles={"upper_arm_1": 190.0}), FS, "sitting",
                          reference_deg={"upper_arm_1": 210.0}, dwell_threshold_deg=25.0)
    assert get(rows, "TON-T", "in_plane_deg", "upper_arm_1", 1)["value"] == pytest.approx(-20.0, abs=0.1)
    assert get(rows, "TON-DW", "fraction_pct", "upper_arm_1", 1)["value"] == pytest.approx(0.0)
    r = get(rows, "TON-DW", "episode_mean_s", "upper_arm_1", 1)
    assert math.isnan(r["value"]) and r["reason"] == "no_episode"


def test_tonic_planted_dwell_bout():
    """A 20 deg trunk bout from 13 to 17 s: window 2 (10 to 20 s) dwells 40 percent, 4 s, one episode."""
    bout = lambda t: np.where((t >= 13) & (t < 17), 20.0, 0.0)  # noqa: E731
    rows = IP.descriptors(figure(501, angles={"trunk": bout}), FS, "sitting", reference_deg=0.0,
                          dwell_threshold_deg=10.0)
    assert get(rows, "TON-DW", "fraction_pct", "trunk", 2)["value"] == pytest.approx(40.0, abs=2.0)
    assert get(rows, "TON-DW", "time_s", "trunk", 2)["value"] == pytest.approx(4.0, abs=0.2)
    assert get(rows, "TON-DW", "episode_rate_per_min", "trunk", 2)["value"] == pytest.approx(6.0)
    assert get(rows, "TON-DW", "episode_mean_s", "trunk", 2)["value"] == pytest.approx(4.0, abs=0.2)
    assert get(rows, "TON-DW", "fraction_pct", "trunk", 0)["value"] == pytest.approx(0.0)


def _unfiltered_median(offset, amp):
    """Median of |offset + amp sin| (offset < amp < 3 offset): m with P(-m <= X <= m) = 1/2, by bisection."""
    lo, hi = offset, amp + offset
    for _ in range(80):
        m = (lo + hi) / 2
        p = (math.asin(min(1.0, (m - offset) / amp)) - math.asin(max(-1.0, (-m - offset) / amp))) / math.pi
        lo, hi = (m, hi) if p < 0.5 else (lo, m)
    return m


def test_tonic_lowpass_removes_the_phasic_oscillation():
    """DR-14.2: the 0.3 Hz low-pass separates the tonic offset from a 1.5 Hz oscillation. A segment 10 deg off its
    reference, oscillating 30 deg about it, has TON-T 10 deg; without the low-pass the median |deviation| would be
    18.7 deg (closed form), so the test fails if the filter is skipped."""
    xy = figure(501, angles={"upper_arm_1": sine(30.0, 1.5, 180.0)})
    rows = IP.descriptors(xy, FS, "sitting", reference_deg={"upper_arm_1": 170.0})
    assert _unfiltered_median(10.0, 30.0) == pytest.approx(18.708, abs=1e-3)
    for w in range(3):
        assert get(rows, "TON-T", "deviation_deg", "upper_arm_1", w)["value"] == pytest.approx(10.0, abs=0.1)
        assert get(rows, "TON-T", "in_plane_deg", "upper_arm_1", w)["value"] == pytest.approx(10.0, abs=0.1)
        o = get(rows, "TON-T", "orientation_deg", "upper_arm_1", w)["value"]
        assert float(IP.wrap_deg(o - 180.0)) == pytest.approx(0.0, abs=0.1)


def test_tonic_median_not_mean():
    """TON-T is a median over the window (DR-14.2): a 2.5 s bout at 40 deg in a 10 s window at 0 deg leaves the
    median deviation near 0 (low-pass edges only), while the mean would be near 10 deg."""
    bout = lambda t: np.where((t >= 2.0) & (t < 4.5), 40.0, 0.0)  # noqa: E731
    xy = figure(501, angles={"trunk": bout})
    rows = IP.descriptors(xy, FS, "sitting", reference_deg=0.0)
    lp = IP.wrap_deg(IP.lowpass_direction(IP.segment_angle(IP.points(xy), "trunk"), FS)[:250])
    assert np.mean(np.abs(lp)) > 9.0 and np.mean(lp) > 9.0              # the fixture separates median and mean
    dev = get(rows, "TON-T", "deviation_deg", "trunk", 0)["value"]
    inp = get(rows, "TON-T", "in_plane_deg", "trunk", 0)["value"]
    assert dev == pytest.approx(float(np.median(np.abs(lp))), abs=1e-9) and dev < 2.0
    assert inp == pytest.approx(float(np.median(lp)), abs=1e-9) and abs(inp) < 0.5


def test_tonic_without_reference_has_no_value():
    rows = IP.descriptors(figure(501), FS, "sitting")
    r = get(rows, "TON-T", "deviation_deg", "trunk", 0)
    assert math.isnan(r["value"]) and r["reason"] == "no_reference" and "no_norm" in r["flags"]
    r = get(rows, "TON-DW", "fraction_pct", "trunk", 0)
    assert math.isnan(r["value"]) and "no_norm" in r["flags"]
    rows = IP.descriptors(figure(501), FS, "sitting", reference_deg=0.0)
    r = get(rows, "TON-DW", "fraction_pct", "trunk", 0)
    assert math.isnan(r["value"]) and r["reason"] == "no_threshold"
    o = get(rows, "TON-T", "orientation_deg", "upper_arm_2", 0)
    assert o["value"] == pytest.approx(170.0, abs=1e-6) and "auxiliary" in o["flags"]


def test_frame_rotation_and_scale():
    """DR-21.17 on the image plane: an in-plane rotation of the whole figure moves the image-frame angles by the
    rotation and leaves the parent-frame angles, the phasic power and the flow descriptors unchanged; a scale
    change leaves everything unchanged (torso lengths)."""
    ang = {"forearm_1": sine(8.0, 1.5, 185.0)}
    paths = {"wrist2": lambda t: np.stack([-0.5 + 0.2 * np.cos(2 * np.pi * t), 0.3 + 0.1 * np.sin(2 * np.pi * t)], 1)}
    kw = dict(reference_deg=0.0, occ_threshold=0.5)
    base = {(r["descriptor"], r["quantity"], r["target"], r["window"]): r["value"]
            for r in IP.descriptors(figure(501, angles=ang, paths=paths), FS, "sitting", frame="parent", **kw)}
    for xy in (figure(501, angles=ang, paths=paths, rot_deg=30.0), figure(501, angles=ang, paths=paths, scale=2.5)):
        other = {(r["descriptor"], r["quantity"], r["target"], r["window"]): r["value"]
                 for r in IP.descriptors(xy, FS, "sitting", frame="parent", **kw)}
        for key, v in base.items():
            if key[0] == "TON-T" and key[2] == "trunk":
                continue                     # the trunk's parent is the image axis
            if math.isnan(v):
                assert math.isnan(other[key]), key
            else:
                assert other[key] == pytest.approx(v, rel=1e-6, abs=1e-6), key
    img = IP.descriptors(figure(501, rot_deg=30.0), FS, "sitting")
    assert get(img, "TON-T", "orientation_deg", "trunk", 0)["value"] == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------- phasic
def test_phasic_planted_amplitude_frequency_asymmetry():
    """PHA-P1: omega of a 1.5 Hz angle sinusoid of amplitude A has band power (2 pi f A g)^2 / 2, g the derivative
    gain; the far side at 2 A gives 4 times the power (6.02 dB); a 5 Hz oscillation lies outside the band."""
    A, f = 8.0, 1.5
    xy = figure(501, angles={"forearm_1": sine(A, f, 185.0), "forearm_2": sine(2 * A, f, 175.0),
                             "shank_1": sine(A, 5.0, 180.0)})
    norm = {("PHA-P1", "forearm_1"): 10.0, ("PHA-P1", "forearm_2"): 10.0}
    rows = IP.descriptors(xy, FS, "sitting", norm_median=norm)
    expect = (2 * np.pi * f * A * IP.derivative_gain(f, FS)) ** 2 / 2
    for w in range(3):
        p1 = get(rows, "PHA-P1", "power", "forearm_1", w)["value"]
        p2 = get(rows, "PHA-P1", "power", "forearm_2", w)["value"]
        assert p1 == pytest.approx(expect, rel=1e-3)
        assert p2 / p1 == pytest.approx(4.0, rel=1e-6)
        db1 = get(rows, "PHA-P1", "db_re_norm", "forearm_1", w)["value"]
        db2 = get(rows, "PHA-P1", "db_re_norm", "forearm_2", w)["value"]
        assert db1 == pytest.approx(10 * math.log10(p1 / 10.0))
        assert db2 - db1 == pytest.approx(10 * math.log10(4.0), abs=1e-6)
        assert get(rows, "PHA-P1", "power", "shank_1", w)["value"] < 1e-6 * expect
        assert get(rows, "PHA-P1", "power", "upper_arm_1", w)["value"] < 1e-9
    r = get(rows, "PHA-P1", "db_re_norm", "shank_1", 0)
    assert math.isnan(r["value"]) and r["reason"] == "no_norm"
    omega = IP.time_derivative(IP.unwrap_deg(IP.segment_angle(IP.points(xy), "forearm_1")), FS)
    fr, p, _ = IP.welch_psd(omega[:250], FS)
    assert fr[np.argmax(p)] == f


def test_phasic_uses_the_segment_own_angular_velocity():
    """R9: PHA-P1 differentiates the image-frame angle (the segment's own angular velocity, DR-2.2) in either frame.
    A trunk swinging 10 deg at 1 Hz under arms held still in the image: the arms have no PHA-P1 power although
    their parent-frame angle swings, and the trunk has the closed-form power."""
    xy = figure(501, angles={"trunk": sine(10.0, 1.0, 0.0)})
    expect = (2 * np.pi * 1.0 * 10.0 * IP.derivative_gain(1.0, FS)) ** 2 / 2
    for frame in ("image", "parent"):
        rows = IP.descriptors(xy, FS, "sitting", frame=frame)
        for w in range(3):
            assert get(rows, "PHA-P1", "power", "upper_arm_1", w)["value"] < 1e-9, (frame, w)
            assert get(rows, "PHA-P1", "power", "trunk", w)["value"] == pytest.approx(expect, rel=1e-3)
    parent = IP.segment_angle(IP.points(xy), "upper_arm_1", "parent")
    assert np.ptp(parent) == pytest.approx(20.0, abs=0.1)                  # the parent-frame angle does swing


# ---------------------------------------------------------------- flow
def _flow_record():
    a = 0.2
    paths = {"wrist1": lambda t: np.stack([0.5 + a * np.sin(2 * np.pi * t), 0.3 + 0 * t], 1),
             "wrist2": lambda t: np.stack([-0.5 + 2 * a * np.sin(2 * np.pi * t), 0.3 + 0 * t], 1),
             "ankle1": lambda t: np.stack([0.3 + 0.3 * np.cos(np.pi * t), 1.0 + 0.3 * np.sin(np.pi * t)], 1),
             "ankle2": lambda t: np.stack([-0.3 + 0.2 * np.cos(2 * np.pi * t), 1.0 + 0.2 * np.sin(2 * np.pi * t)],
                                          1)}
    return figure(int(30 * FS) + 1, paths=paths), a


def test_flow_planted_amplitudes_and_directions():
    xy, a = _flow_record()
    rows = IP.descriptors(xy, FS, "lying", norm_median={"FLOW-E": 0.01})
    expect = (2 * np.pi * 1.0 * a * IP.derivative_gain(1.0, FS)) ** 2 / 2
    for w in (1, 2, 3):                          # windows clear of the record's edges
        e1 = get(rows, "FLOW-E", "power", "wrist1", w)["value"]
        e2 = get(rows, "FLOW-E", "power", "wrist2", w)["value"]
        assert e1 == pytest.approx(expect, rel=1e-6)
        assert e2 / e1 == pytest.approx(4.0, rel=1e-6)
        db = get(rows, "FLOW-E", "db_re_norm", "wrist1", w)["value"]
        assert db == pytest.approx(10 * math.log10(e1 / 0.01))
        assert get(rows, "FLOW-DII", "dii", "wrist1", w)["value"] == pytest.approx(0.0, abs=1e-6)
        assert get(rows, "FLOW-DII", "dii", "ankle2", w)["value"] == pytest.approx(1.0, abs=0.01)
        assert get(rows, "FLOW-R", "roaming", "ankle1", w)["value"] == pytest.approx(0.3, rel=0.002)
        assert get(rows, "FLOW-R", "roaming", "ankle2", w)["value"] == pytest.approx(0.2, rel=0.01)
        assert get(rows, "FLOW-R", "roaming", "wrist1", w)["value"] == pytest.approx(a / math.sqrt(2), rel=0.01)


def test_static_point_has_no_direction():
    """A point that does not move has no FLOW-DII (reason no_motion) and zero flow energy, occupancy and roaming."""
    rows = IP.descriptors(figure(501), FS, "lying", occ_threshold=0.01)
    r = get(rows, "FLOW-DII", "dii", "ankle1", 1)
    assert math.isnan(r["value"]) and r["reason"] == "no_motion"
    assert get(rows, "FLOW-E", "power", "ankle1", 1)["value"] == 0.0
    assert get(rows, "FLOW-OCC", "occupancy_pct", "ankle1", 1)["value"] == 0.0
    # a speed equal to the threshold is not occupancy (strictly above, note 4.5): exact 0 against a threshold of 0
    zero = IP.descriptors(figure(501), FS, "lying", occ_threshold=0.0)
    assert get(zero, "FLOW-OCC", "occupancy_pct", "ankle1", 1)["value"] == 0.0
    assert get(rows, "FLOW-R", "roaming", "ankle1", 1)["value"] == pytest.approx(0.0, abs=1e-12)
    v = IP.time_derivative(np.full((50, 2), 123.456), FS)
    assert (v == 0.0).all()


def test_flow_occupancy_planted_bout():
    """FLOW-OCC: a circle of constant speed from 15 s on, still before; threshold at half its speed."""
    r, f = 0.2, 1.0
    start = np.array([r * math.cos(2 * math.pi * f * 15), r * math.sin(2 * math.pi * f * 15)])

    def path(t):
        circ = np.stack([r * np.cos(2 * np.pi * f * t), r * np.sin(2 * np.pi * f * t)], 1)
        return np.where((t >= 15)[:, None], circ, start[None, :]) + np.array([0.5, 0.3])

    xy = figure(int(30 * FS) + 1, paths={"wrist1": path})
    rows = IP.descriptors(xy, FS, "sitting", occ_threshold={"wrist1": 0.5 * 2 * math.pi * f * r})
    occ = [get(rows, "FLOW-OCC", "occupancy_pct", "wrist1", w)["value"] for w in range(5)]
    assert occ[0] == pytest.approx(0.0) and occ[1] == pytest.approx(0.0, abs=2.0)
    assert occ[2] == pytest.approx(50.0, abs=5.0)
    assert occ[3] == pytest.approx(100.0, abs=2.0) and occ[4] == pytest.approx(100.0, abs=2.0)
    r2 = get(rows, "FLOW-OCC", "occupancy_pct", "wrist2", 0)
    assert math.isnan(r2["value"]) and r2["reason"] == "no_threshold" and "proposed" in r2["flags"]


def test_flow_relative_to_a_reference_point():
    """flow_reference: a wrist carried by a swinging upper arm with a fixed forearm angle moves in the image but not
    against its elbow; against the elbow its flow is nil, and a forearm oscillation is seen alone."""
    ang = {"upper_arm_1": sine(10.0, 1.0, 190.0), "upper_arm_2": sine(10.0, 1.0, 170.0),
           "forearm_2": sine(6.0, 1.5, 175.0)}
    xy = figure(501, angles=ang)
    absolute = IP.descriptors(xy, FS, "sitting", flow_points=("wrist1", "wrist2"))
    relative = IP.descriptors(xy, FS, "sitting", flow_points=("wrist1", "wrist2"),
                              flow_reference={"wrist1": "elbow1", "wrist2": "elbow2"})
    assert get(absolute, "FLOW-E", "power", "wrist1", 1)["value"] > 1e-3
    r = get(relative, "FLOW-E", "power", "wrist1", 1)
    assert r["value"] == pytest.approx(0.0, abs=1e-20) and "relative_to_elbow1" in r["flags"]
    assert get(relative, "FLOW-DII", "dii", "wrist1", 1)["reason"] == "no_motion"
    # wrist2 against elbow2: the forearm (0.4 torso lengths) turning 6 deg at 1.5 Hz
    lin = 0.4 * math.radians(6.0) * 2 * math.pi * 1.5 * IP.derivative_gain(1.5, FS)
    assert get(relative, "FLOW-E", "power", "wrist2", 1)["value"] == pytest.approx(lin ** 2 / 2, rel=0.01)
    assert get(relative, "FLOW-R", "roaming", "wrist1", 1)["value"] == pytest.approx(
        get(absolute, "FLOW-R", "roaming", "wrist1", 1)["value"])          # FLOW-R stays against the pelvis
    with pytest.raises(ValueError):
        IP.descriptors(xy, FS, "sitting", flow_points=("chin",))


def test_flow_occupancy_on_excursion():
    """occ_signal 'excursion': a 1 Hz line of amplitude a, band-passed with gain g, exceeds a / 2 for the share
    1 - (2 / pi) asin(1 / (2 g)) of the time (two thirds at g = 1), and never exceeds 2 a."""
    xy, a = _flow_record()
    c = (math.sqrt(2) - 1) ** (1 / 4)
    tf = math.tan(math.pi * 1.0 / FS)
    hp = 1 / (1 + (c * math.tan(math.pi * 0.5 / FS) / tf) ** 4)
    lp = 1 / (1 + (tf * c / math.tan(math.pi * 3.0 / FS)) ** 4)
    expect = 100 * (1 - 2 / math.pi * math.asin(0.5 / (hp * lp)))
    half = IP.descriptors(xy, FS, "lying", occ_signal="excursion", occ_threshold={"wrist1": a / 2})
    never = IP.descriptors(xy, FS, "lying", occ_signal="excursion", occ_threshold={"wrist1": 2 * a})
    for w in (1, 2, 3):
        r = get(half, "FLOW-OCC", "occupancy_pct", "wrist1", w)
        assert r["value"] == pytest.approx(expect, abs=2.5) and "occ_excursion" in r["flags"]
        assert get(never, "FLOW-OCC", "occupancy_pct", "wrist1", w)["value"] == 0.0
    with pytest.raises(ValueError):
        IP.descriptors(xy, FS, "lying", occ_signal="displacement")


def test_half_windows_and_explicit_windows():
    assert IP.half_windows(501) == [(0, 250), (250, 500)]
    assert IP.half_windows(498) == [(0, 249), (249, 498)]
    assert IP.half_windows(1) == []
    xy = figure(498, angles={"forearm_1": sine(8.0, 1.5, 185.0)})
    assert {r["window"] for r in IP.descriptors(xy, FS, "sitting", hop_s=10.0)} == {0}   # 10 s tiling: one window
    rows = IP.descriptors(xy, FS, "sitting", windows=IP.half_windows(498))
    assert {r["window"] for r in rows} == {0, 1}
    assert get(rows, "TON-T", "orientation_deg", "trunk", 1)["t_start_s"] == pytest.approx(249 / FS)
    with pytest.raises(ValueError):
        IP.descriptors(xy, FS, "sitting", windows=[(0, 600)])


def test_no_torso_refuses_flow():
    lk = np.ones((501, len(IP.POINTS)))
    lk[:, IP.POINTS.index("hip1")] = 0.0
    rows = IP.descriptors(figure(501), FS, "lying", likelihood=lk)
    flow = [r for r in rows if r["descriptor"].startswith("FLOW-")]
    assert flow and all(r["reason"] == "no_torso" and math.isnan(r["value"]) for r in flow)
    t = IP.torso_length(IP.points(figure(501, scale=2.0)))
    assert t == pytest.approx(200.0)


# ---------------------------------------------------------------- validity, bandwidth, postures, pooling
def test_likelihood_masks_and_valid_fraction():
    lk = np.ones((501, len(IP.POINTS)))
    lk[:, IP.POINTS.index("chin")] = 0.05
    lk[:, IP.POINTS.index("forehead")] = 0.05          # as in every training frame (T10)
    lk[:125, IP.POINTS.index("wrist1")] = 0.2          # first 5 s of wrist1 below the screen
    lk[:, IP.POINTS.index("wrist2")] = 0.5             # exactly at the screen: valid ("at least", R3)
    rows = IP.descriptors(figure(501), FS, "sitting", likelihood=lk)
    head = [r for r in rows if r["target"] in ("head", "forehead")]
    assert head and all(math.isnan(r["value"]) and r["reason"] == "no_valid_samples" for r in head)
    assert get(rows, "FLOW-R", "roaming", "wrist1", 0)["valid_fraction"] == pytest.approx(0.5)
    assert get(rows, "TON-T", "orientation_deg", "forearm_1", 0)["valid_fraction"] == pytest.approx(0.5)
    assert get(rows, "FLOW-R", "roaming", "wrist2", 0)["valid_fraction"] == 1.0
    strict = IP.descriptors(figure(501), FS, "sitting", likelihood=lk, min_valid_fraction=0.8)
    r = get(strict, "FLOW-R", "roaming", "wrist1", 0)
    assert math.isnan(r["value"]) and r["reason"] == "valid_fraction_below_min"
    assert get(strict, "FLOW-R", "roaming", "wrist1", 2)["reason"] is None
    # a valid fraction equal to the minimum enters (DR-2.7 refuses below the minimum only)
    edge = IP.descriptors(figure(501), FS, "sitting", likelihood=lk, min_valid_fraction=0.5)
    r = get(edge, "FLOW-R", "roaming", "wrist1", 0)
    assert r["valid_fraction"] == 0.5 and r["reason"] is None and math.isfinite(r["value"])
    loose = IP.descriptors(figure(501), FS, "sitting", likelihood=lk, likelihood_min=0.1)
    assert get(loose, "FLOW-R", "roaming", "wrist1", 0)["valid_fraction"] == pytest.approx(1.0)
    # with no valid sample, a missing reference or threshold yields to no_valid_samples (R15)
    off = np.ones((501, len(IP.POINTS)))
    off[:250, IP.POINTS.index("elbow1")] = 0.0
    r = get(IP.descriptors(figure(501), FS, "sitting", likelihood=off), "TON-DW", "fraction_pct", "upper_arm_1", 0)
    assert r["reason"] == "no_valid_samples" and "no_norm" in r["flags"]


def test_bandwidth_admission():
    xy = figure(501)
    rows = IP.descriptors(xy, FS, "sitting")
    r = get(rows, "PHA-P1", "power", "trunk", 0)
    assert r["f_max_hz"] is None and "bandwidth_k_open" in r["flags"] and r["reason"] is None
    rows = IP.descriptors(xy, FS, "sitting", k=0.1)            # f_max 2.5 Hz under the 3 Hz band edge
    for desc, q, tgt in (("PHA-P1", "power", "trunk"), ("FLOW-DII", "dii", "wrist1"), ("FLOW-E", "power", "wrist1"),
                         ("FLOW-OCC", "occupancy_pct", "wrist1")):
        r = get(rows, desc, q, tgt, 0)
        assert r["reason"] == "band_above_f_max" and r["f_max_hz"] == pytest.approx(2.5)
        assert "bandwidth_k_open" not in r["flags"]
    assert get(rows, "FLOW-R", "roaming", "wrist1", 0)["reason"] is None
    rows = IP.descriptors(xy, FS, "sitting", k=0.38, f_lp_hz=2.0)
    assert get(rows, "FLOW-E", "power", "wrist1", 0)["reason"] == "band_above_f_max"
    assert get(rows, "FLOW-E", "power", "wrist1", 0)["f_max_hz"] == pytest.approx(2.0)
    # DR-5.3 admits a band whose upper edge equals f_max: k = 0.12 at 25 fps, and a 3 Hz device low-pass
    for kw in (dict(k=0.12), dict(k=0.38, f_lp_hz=3.0)):
        rows = IP.descriptors(xy, FS, "sitting", **kw)
        for desc, q, tgt in (("PHA-P1", "power", "trunk"), ("FLOW-E", "power", "wrist1")):
            r = get(rows, desc, q, tgt, 0)
            assert r["f_max_hz"] == pytest.approx(3.0) and r["reason"] is None, (kw, desc)
    rows = IP.descriptors(xy, FS, "sitting", k=0.38, f_lp_hz=2.999)
    assert get(rows, "PHA-P1", "power", "trunk", 0)["reason"] == "band_above_f_max"


def test_both_postures():
    xy = figure(501, angles={"forearm_1": sine(8.0, 1.5, 185.0)})
    lying = IP.descriptors(xy, FS, "lying", reference_deg=0.0)
    sitting = IP.descriptors(xy, FS, "sitting", reference_deg=0.0)
    assert len(lying) == len(sitting) > 0
    for a, b in zip(lying, sitting):
        assert a["descriptor"] == b["descriptor"] and a["target"] == b["target"]
        assert (math.isnan(a["value"]) and math.isnan(b["value"])) or a["value"] == b["value"]
        assert a["task"] == "rest_lying" and b["task"] == "rest_sitting"
        assert a["dis_item"] == b["dis_item"]
        expected = {"trunk": ("sitting",), "arms": ("sitting",), "legs": ("lying",)}.get(a["region"])
        if a["dis_item"] is None:                   # distal targets and the head: no item of the sheet scores them
            assert a["scored_in_posture"] is False and b["scored_in_posture"] is False
        else:
            assert a["scored_in_posture"] == ("lying" in expected)
            assert b["scored_in_posture"] == ("sitting" in expected)
    with pytest.raises(ValueError):
        IP.descriptors(xy, FS, "standing")


def test_dis_items_are_the_proximal_items_only():
    """R12: the held DIS sheet scores the trunk, the proximal arm and the proximal leg; the distal segments and
    points keep their region but carry no DIS item and are not scored in either posture."""
    rows = IP.descriptors(figure(501), FS, "sitting", flow_points=("wrist1", "elbow1", "knee2", "ankle2", "neck"))
    item = {r["target"]: (r["region"], r["dis_item"], r["scored_in_posture"]) for r in rows if r["window"] == 0}
    assert item["trunk"] == ("trunk", "trunk", True) and item["neck"] == ("trunk", "trunk", True)
    assert item["upper_arm_1"] == ("arms", "proximal_arm", True) and item["elbow1"] == ("arms", "proximal_arm", True)
    assert item["thigh_2"] == ("legs", "proximal_leg", False) and item["knee2"] == ("legs", "proximal_leg", False)
    for t, region in (("forearm_1", "arms"), ("wrist1", "arms"), ("shank_2", "legs"), ("ankle2", "legs"),
                      ("head", "head")):
        assert item[t] == (region, None, False), t
    assert set(IP.DIS_ITEMS) == {s.dis_item for s in IP.SEGMENTS.values()} - {None}


def test_rows_carry_flags_and_fields():
    rows = IP.descriptors(figure(501), FS, "sitting", record_flags=("paediatric", "within_source_norm"))
    descs = {r["descriptor"] for r in rows}
    assert descs == {"TON-T", "TON-DW", "PHA-P1", "FLOW-DII", "FLOW-E", "FLOW-R", "FLOW-OCC"}
    for r in rows:
        assert tuple(r) == IP.ROW_KEYS
        assert r["flags"][:3] == ("image_plane", "paediatric", "within_source_norm")
        assert ("laterality_unknown" in r["flags"]) == (r["side"] is not None)
        assert r["resolution_s"] == pytest.approx(1 / FS) and r["rule"].startswith("DRAFT")
        assert (r["reason"] is None) == math.isfinite(r["value"]), r
    n_targets = len(IP.PARAMS["segments"].default) * 9 + len(IP.PARAMS["flow_points"].default) * 5
    assert len(rows) == 3 * n_targets


def test_every_row_has_a_value_or_a_reason():
    """R15: a finite value and no reason, or NaN and a named reason, under the paths that give NaN: a still
    figure in dB re 1 unit (zero_power), a norm of 0 (norm_not_positive), a 0.5 s Welch segment with one bin in
    the band (band_too_few_bins), a one-sample window (too_few_samples), and the T26 settings."""
    still = figure(501)
    moving = figure(501, angles={"forearm_1": sine(8.0, 1.5, 185.0)},
                    paths={"wrist2": lambda t: np.stack([-0.5 + 0.2 * np.sin(2 * np.pi * t), 0.3 + 0 * t], 1)})
    cases = {
        "zero_power": (still, dict(norm_median={"PHA-P1": 1.0, "FLOW-E": 1.0})),
        "norm_not_positive": (moving, dict(norm_median={"PHA-P1": 0.0, "FLOW-E": -1.0})),
        "band_too_few_bins": (moving, dict(welch_seg_s=0.5)),
        "too_few_samples": (still, dict(windows=[(0, 1), (0, 250)])),
        "t26": (moving, dict(frame="parent", reference_deg=0.0, dwell_threshold_deg=5.0, occ_signal="excursion",
                             occ_threshold=0.02, min_valid_fraction=0.5, dwell_min_run_s=1.8 / FS,
                             windows=IP.half_windows(501), norm_median={"PHA-P1": 1.0, "FLOW-E": 1.0})),
    }
    seen = {}
    for name, (xy, kw) in cases.items():
        rows = IP.descriptors(xy, FS, "sitting", **kw)
        for r in rows:
            assert (r["reason"] is None) == math.isfinite(r["value"]), (name, r)
            assert r["reason"] != "not_finite", (name, r)
        seen[name] = {r["reason"] for r in rows}
    assert "zero_power" in seen["zero_power"]
    assert "norm_not_positive" in seen["norm_not_positive"]
    assert "band_too_few_bins" in seen["band_too_few_bins"]
    assert "too_few_samples" in seen["too_few_samples"]
    rows = IP.descriptors(still, FS, "sitting", norm_median={"PHA-P1": 1.0, "FLOW-E": 1.0})
    assert get(rows, "PHA-P1", "db_re_norm", "trunk", 0)["reason"] == "zero_power"
    assert get(rows, "FLOW-E", "db_re_norm", "wrist1", 0)["reason"] == "zero_power"
    rows = IP.descriptors(moving, FS, "sitting", welch_seg_s=0.5, norm_median={"PHA-P1": 1.0})
    assert get(rows, "PHA-P1", "power", "forearm_1", 0)["reason"] == "band_too_few_bins"
    assert get(rows, "PHA-P1", "db_re_norm", "forearm_1", 0)["reason"] == "band_too_few_bins"
    assert IP.db_re_norm(4.0, 1.0, None) == (pytest.approx(10 * math.log10(4.0)), None)
    assert IP.db_re_norm(4.0, None, None)[1] == "no_norm"
    assert IP.db_re_norm(4.0, 1.0, "no_welch_segment")[1] == "no_welch_segment"


def test_pool_sides():
    A, f = 8.0, 1.5
    xy = figure(501, angles={"forearm_1": sine(A, f, 185.0), "forearm_2": sine(2 * A, f, 175.0)})
    lk = np.ones((501, len(IP.POINTS)))
    lk[:, IP.POINTS.index("ankle2")] = 0.0
    lk[:125, IP.POINTS.index("wrist1")] = 0.2
    rows = IP.descriptors(xy, FS, "sitting", likelihood=lk)
    pooled = IP.pool_sides(rows)
    w = [r for r in pooled if r["descriptor"] == "FLOW-R" and r["window"] == 0 and r["target"] == "wrist_pooled"]
    assert get(rows, "FLOW-R", "roaming", "wrist1", 0)["valid_fraction"] == pytest.approx(0.5)
    assert get(rows, "FLOW-R", "roaming", "wrist2", 0)["valid_fraction"] == pytest.approx(1.0)
    assert w[0]["valid_fraction"] == pytest.approx(0.5) and w[0]["reason"] is None   # the smaller of the two sides
    p = [r for r in pooled if r["descriptor"] == "PHA-P1" and r["quantity"] == "power" and r["window"] == 0
         and r["target"] == "forearm_pooled"]
    assert len(p) == 1
    p1 = get(rows, "PHA-P1", "power", "forearm_1", 0)["value"]
    p2 = get(rows, "PHA-P1", "power", "forearm_2", 0)["value"]
    assert p[0]["value"] == pytest.approx((p1 + p2) / 2) and p[0]["side"] == "pooled"
    assert "side_pooled" in p[0]["flags"] and "laterality_unknown" in p[0]["flags"]
    mx = [r for r in IP.pool_sides(rows, pool="max") if r["descriptor"] == "PHA-P1" and r["quantity"] == "power"
          and r["window"] == 0 and r["target"] == "forearm_pooled"]
    assert mx[0]["value"] == pytest.approx(max(p1, p2))
    s = [r for r in pooled if r["descriptor"] == "FLOW-R" and r["window"] == 0 and r["target"] == "ankle_pooled"]
    assert math.isnan(s[0]["value"]) and s[0]["reason"] == "side_missing"
    assert not [r for r in pooled if r["target"] in ("trunk_pooled", "head_pooled")]


# ---------------------------------------------------------------- parameters and file access
def test_every_rule_parameter_is_explicit_and_registered():
    ids = re.compile(r"^(DR-\d+\.\d+|OPEN-D\d\d)(, (DR-\d+\.\d+|OPEN-D\d\d))*$")
    for name, p in IP.PARAMS.items():
        assert ids.match(p.ids), (name, p.ids)
        assert p.status in IP.STATUSES and p.note
    seen = set()
    for fname, fn in inspect.getmembers(IP, inspect.isfunction):
        if fname.startswith("_") or fn.__module__ != IP.__name__:
            continue
        for pname, par in inspect.signature(fn).parameters.items():
            if par.default is inspect.Parameter.empty:
                continue
            if pname in IP.NON_RULE_ARGS:
                continue
            assert pname in IP.PARAMS, (fname, pname)
            assert par.default == IP.PARAMS[pname].default, (fname, pname)
            seen.add(pname)
    assert seen == set(IP.PARAMS), set(IP.PARAMS) - seen
    sig = inspect.signature(IP.descriptors).parameters
    assert set(IP.PARAMS) - {"pool"} <= set(sig)
    table = IP.param_table()
    assert [r["name"] for r in table] == list(IP.PARAMS)


def test_module_opens_no_file():
    """A grep of mdfm/clinical: no file access in its code."""
    pattern = re.compile(r"\bopen\(|read_csv|read_parquet|read_text|read_bytes|write_text|write_bytes|np\.load|"
                         r"np\.save|loadtxt|savetxt|genfromtxt|pickle|\bjson\b|torch|h5py|zipfile|Path\(|"
                         r"\bimport os\b|os\.path|glob|urllib|requests|socket|subprocess|shutil|tofile|fromfile")
    files = [f for f in os.listdir(MODULE_DIR) if f.endswith(".py") and not f.startswith("._")]
    assert sorted(files) == ["__init__.py", "image_plane.py", "s1_formulas.py"]
    for f in files:
        with open(os.path.join(MODULE_DIR, f), encoding="utf-8") as fh:
            src = fh.read()
        assert not pattern.findall(src), (f, pattern.findall(src))
        assert chr(0x2014) not in src and chr(0x2013) not in src
