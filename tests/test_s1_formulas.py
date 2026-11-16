"""Synthetic fixtures of the S1 image-plane descriptors and statistics (mdfm/clinical/s1_formulas.py).

S1 section 6, step 2, and S1 draft section 7, first kill condition: the image-plane descriptors pass these fixtures
before any of their outputs is read (DR-0.10). Fixtures follow DR-21.5, DR-21.6, DR-21.13 and DR-21.17 in their
image-plane form, plus known values for the filters, spectra, flow, dwell, window admission, the agreement
statistics and the precision formula. Synthetic inputs only; no file of any source is read.
"""
import math

import numpy as np
import pytest

from mdfm.clinical import s1_formulas as F

FS = 25.0


def body(T, fs=FS, arm_deg=0.0, leg_deg=0.0, rot_deg=0.0, extra=None):
    """Keypoints (T, 14, 2) and likelihoods of a still frontal body: hips 100 px below the shoulders, arms and
    thighs at arm_deg and leg_deg from the trunk axis (positive toward image left of the trunk axis), the whole
    body rotated by rot_deg in the image plane. extra: {point: (T, 2) offset}."""
    H = np.array([0.0, 0.0])
    S = np.array([0.0, -100.0])
    p = {"hip1": H + [-20, 0], "hip2": H + [20, 0], "shoulder1": S + [-30, 0], "shoulder2": S + [30, 0]}

    def rot(v, deg):
        a = math.radians(deg)
        return np.array([math.cos(a) * v[0] - math.sin(a) * v[1], math.sin(a) * v[0] + math.cos(a) * v[1]])
    for s in ("1", "2"):
        p[f"elbow{s}"] = p[f"shoulder{s}"] + rot(np.array([0.0, 60.0]), arm_deg)
        p[f"wrist{s}"] = p[f"elbow{s}"] + [0, 50]
        p[f"knee{s}"] = p[f"hip{s}"] + rot(np.array([0.0, 80.0]), leg_deg)
        p[f"ankle{s}"] = p[f"knee{s}"] + [0, 70]
    p["chin"], p["forehead"] = S + [0, -20], S + [0, -50]
    xy = np.stack([np.tile(p[n], (T, 1)) for n in F.POINTS], axis=1).astype(float)
    if extra:
        for n, off in extra.items():
            xy[:, F.POINTS.index(n)] += off
    if rot_deg:
        a = math.radians(rot_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        xy = xy @ R.T + 300.0
    like = np.ones((T, len(F.POINTS)))
    like[:, F.POINTS.index("chin")] = like[:, F.POINTS.index("forehead")] = 0.0
    return xy, like


def rec(xy, like):
    return F.record_descriptors(xy, like, FS)


# ---------------------------------------------------------------- DR-21.6 and DR-21.17, tonic
def test_tonic_constant_20_deg_gives_ton_t_20_and_full_dwell():
    d = rec(*body(500, arm_deg=20.0))
    arm = d["regions"]["proximal_arm"]["record"]
    assert arm["TON-T"] == pytest.approx(20.0, abs=0.1)
    assert arm["TON-DW"] == pytest.approx(100.0)          # threshold 18.0 deg < 20 deg
    leg = d["regions"]["proximal_leg"]["record"]
    assert leg["TON-T"] == pytest.approx(0.0, abs=0.1)
    assert leg["TON-DW"] == pytest.approx(0.0)


def test_leg_dwell_threshold_is_4_5_deg():
    assert F.DWELL_THRESHOLD_DEG == {"proximal_arm": 18.0, "proximal_leg": 4.5}
    d5 = rec(*body(500, leg_deg=5.0))["regions"]["proximal_leg"]["record"]
    d4 = rec(*body(500, leg_deg=4.0))["regions"]["proximal_leg"]["record"]
    assert d5["TON-T"] == pytest.approx(5.0, abs=0.1) and d5["TON-DW"] == pytest.approx(100.0)
    assert d4["TON-T"] == pytest.approx(4.0, abs=0.1) and d4["TON-DW"] == pytest.approx(0.0)


def test_tonic_frame_whole_body_rotation_changes_nothing():
    a = rec(*body(500, arm_deg=12.0, leg_deg=3.0))
    b = rec(*body(500, arm_deg=12.0, leg_deg=3.0, rot_deg=90.0))
    for region in ("proximal_arm", "proximal_leg"):
        assert b["regions"][region]["record"]["TON-T"] == pytest.approx(a["regions"][region]["record"]["TON-T"],
                                                                        abs=0.1)
    assert a["regions"]["proximal_arm"]["record"]["TON-T"] == pytest.approx(12.0, abs=0.1)
    # the trunk angle is raw against the image vertical: 0 upright, 90 when the body is turned by 90 deg
    assert a["regions"]["trunk"]["record"]["TRUNK-ANGLE-RAW"] == pytest.approx(0.0, abs=0.1)
    assert b["regions"]["trunk"]["record"]["TRUNK-ANGLE-RAW"] == pytest.approx(90.0, abs=0.1)
    assert "TON-T" not in F.DESCRIPTORS["trunk"] and "TON-DW" not in F.DESCRIPTORS["trunk"]


def test_ton_t_is_the_absolute_deviation_of_either_sign():
    a = rec(*body(500, arm_deg=-25.0))["regions"]["proximal_arm"]["record"]
    assert a["TON-T"] == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------- DR-21.5, flow direction irregularity
def test_dii_formula_line_and_circle():
    t = np.arange(250) / 100.0
    assert F.dii(np.stack([np.cos(2 * np.pi * t), np.zeros_like(t)], 1)) == pytest.approx(0.0, abs=1e-9)
    t = np.arange(1000) / 100.0
    circ = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], 1)
    assert F.dii(circ) == pytest.approx(1.0, abs=0.01)


def test_flow_dii_through_the_pipeline():
    T = 1501        # 60 s: the median window lies away from the filter transients at the record ends
    t = np.arange(T) / FS
    line = np.stack([6 * np.sin(2 * np.pi * t), np.zeros(T)], 1)
    circ = np.stack([6 * np.cos(2 * np.pi * t), 6 * np.sin(2 * np.pi * t)], 1)
    a = rec(*body(T, extra={"elbow1": line, "elbow2": line}))["regions"]["proximal_arm"]["record"]
    b = rec(*body(T, extra={"elbow1": circ, "elbow2": circ}))["regions"]["proximal_arm"]["record"]
    assert a["FLOW-DII"] == pytest.approx(0.0, abs=0.01)
    assert b["FLOW-DII"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------- flow energy, roaming, occupancy
def test_flow_e_of_a_1_hz_oscillation():
    T, amp_px = 501, 10.0                     # torso 100 px: amplitude 0.1 torso lengths
    t = np.arange(T) / FS
    osc = np.stack([amp_px * np.sin(2 * np.pi * t), np.zeros(T)], 1)
    d = rec(*body(T, extra={"elbow1": osc, "elbow2": osc}))["regions"]["proximal_arm"]["record"]
    expected = 10 * math.log10((2 * math.pi * 0.1) ** 2 / 2)          # velocity power (TL/s)^2
    gain_db = 20 * math.log10(math.sin(2 * math.pi / FS) / (2 * math.pi / FS))   # central difference at 1 Hz
    assert d["FLOW-E"] == pytest.approx(expected + gain_db, abs=0.2)


def test_flow_r_of_a_circle_is_its_radius():
    T = 501
    t = np.arange(T) / FS
    circ = np.stack([8 * np.cos(2 * np.pi * t), 8 * np.sin(2 * np.pi * t)], 1)
    d = rec(*body(T, extra={"elbow1": circ, "elbow2": circ}))["regions"]["proximal_arm"]["record"]
    assert d["FLOW-R"] == pytest.approx(0.08, rel=0.03)   # 8 px over a 100 px torso
    still = rec(*body(T))["regions"]["proximal_arm"]["record"]
    assert still["FLOW-R"] == pytest.approx(0.0, abs=1e-9)


def test_flow_occ_threshold_is_10_percent_of_the_segment():
    T = 501
    t = np.arange(T) / FS
    seg_px = 60.0
    big = np.stack([0.2 * seg_px * np.sin(2 * np.pi * t), np.zeros(T)], 1)
    small = np.stack([0.05 * seg_px * np.sin(2 * np.pi * t), np.zeros(T)], 1)
    b = rec(*body(T, extra={"elbow1": big, "elbow2": big}))["regions"]["proximal_arm"]["record"]
    s = rec(*body(T, extra={"elbow1": small, "elbow2": small}))["regions"]["proximal_arm"]["record"]
    assert b["FLOW-OCC"] == pytest.approx(100 * 2 / 3, abs=2.5)     # |sin| > 0.5 two thirds of the time
    assert s["FLOW-OCC"] == pytest.approx(0.0, abs=1e-9)
    tr = rec(*body(T))["regions"]["trunk"]
    assert tr["seg_len_tl"] == pytest.approx(1.0)


# ---------------------------------------------------------------- phasic band power
def test_pha_p1_of_a_1_hz_angle_oscillation():
    T, A = 501, 5.0
    t = np.arange(T) / FS
    ang = np.radians(A * np.sin(2 * np.pi * t))
    off = np.stack([-60 * np.sin(ang), 60 * np.cos(ang) - 60], 1)     # the 60 px upper arm turned by ang
    d = rec(*body(T, extra={"elbow1": off, "elbow2": off}))["regions"]["proximal_arm"]["record"]
    expected = 10 * math.log10((2 * math.pi * A) ** 2 / 2)
    gain_db = 20 * math.log10(math.sin(2 * math.pi / FS) / (2 * math.pi / FS))
    assert d["PHA-P1"] == pytest.approx(expected + gain_db, abs=0.2)


# ---------------------------------------------------------------- dwell runs, windows, admission
def test_dwell_counts_no_run_shorter_than_d_min():
    th = np.zeros(250)
    th[10] = 30.0                       # one sample, 0.04 s < 0.072 s: not counted
    th[50:52] = 30.0                    # two samples, 0.08 s: counted
    th[100:150] = 30.0
    frac, rate, mean_s = F.dwell(th, 18.0, FS)
    assert frac == pytest.approx(100 * 52 / 250)
    assert rate == pytest.approx(2 / (250 / FS / 60))
    assert mean_s == pytest.approx((0.08 + 2.0) / 2)


def test_window_bounds_hop_and_halves():
    assert F.window_bounds(500, FS) == [(0, 250), (125, 375), (250, 500)]
    assert F.window_bounds(501, FS) == [(0, 250), (125, 375), (250, 500)]
    assert F.window_bounds(249, FS) == []
    assert F.half_bounds(501) == [(0, 250), (251, 501)]
    assert F.half_bounds(500) == [(0, 250), (250, 500)]


def test_window_enters_only_with_half_of_its_frames_valid():
    xy, like = body(500, arm_deg=20.0)
    lk = like.copy()
    e1 = F.POINTS.index("elbow1")
    lk[:, e1] = 0.0
    lk[0:100, e1] = lk[250:350, e1] = 1.0     # each window (0-250, 125-375, 250-500) 40 percent valid
    d = rec(xy, lk)["regions"]["proximal_arm"]
    assert d["sides"]["1"]["n_windows_entered"]["TON-T"] == 0
    assert math.isnan(d["sides"]["1"]["record"]["TON-T"])
    assert d["sides"]["2"]["record"]["TON-T"] == pytest.approx(20.0, abs=0.1)
    assert math.isnan(d["record"]["TON-T"])                 # side pooling needs both sides
    lk[:, e1] = 0.0
    lk[0:300, e1] = 1.0                       # windows 100, 70 and 20 percent valid: two enter
    d = rec(xy, lk)["regions"]["proximal_arm"]["sides"]["1"]
    assert d["n_windows_entered"]["TON-T"] == 2
    assert d["record"]["TON-T"] == pytest.approx(20.0, abs=0.1)


def test_likelihood_below_half_is_invalid():
    xy, like = body(500)
    like[:, F.POINTS.index("hip1")] = 0.49
    d = rec(xy, like)
    assert math.isnan(d["regions"]["trunk"]["record"]["FLOW-R"])
    assert d["usable_fraction"] == 0.0


# ---------------------------------------------------------------- filters, spectra, bandwidth
def test_forward_backward_minus_3_db_at_the_named_frequency():
    from scipy.signal import freqz
    for f, kind in ((0.3, "low"), (3.0, "low"), (0.5, "high")):
        b, a = F._ba(f, FS, kind)
        _, h = freqz(b, a, worN=[f], fs=FS)
        assert abs(h[0]) ** 2 == pytest.approx(1 / math.sqrt(2), abs=1e-6)


def test_welch_band_power_of_a_sinusoid():
    t = np.arange(1000) / FS
    x = 3.0 * np.sin(2 * np.pi * 1.5 * t)
    assert F.welch_band_power(x, FS) == pytest.approx(4.5, rel=0.02)
    assert math.isnan(F.welch_band_power(x[:99], FS))       # no 4 s segment


def test_bandwidth_admission_at_25_fps():
    assert F.f_max_hz(FS) == pytest.approx(7.5)
    assert F.band_admitted(FS, (0.5, 3.0))
    assert not F.band_admitted(FS, (3.0, 12.0))             # PHA-P2 refused (DR-14.13)


# ---------------------------------------------------------------- agreement statistics, DR-21.13
SF = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)


def test_icc_shrout_fleiss_table():
    assert F.icc_1_1(SF) == pytest.approx(0.17, abs=0.005)
    assert F.icc_2_1(SF) == pytest.approx(0.29, abs=0.005)
    same = np.array([[1, 1], [2, 2], [4, 4]], float)
    assert F.icc_1_1(same) == pytest.approx(1.0) and F.icc_2_1(same) == pytest.approx(1.0)


def test_qwk_against_the_confusion_matrix_form():
    a = np.array([0, 1, 2, 3, 4, 2, 1, 0, 3, 3])
    b = np.array([0, 2, 2, 3, 3, 1, 1, 1, 4, 2])
    K = 5
    O = np.zeros((K, K))
    for i, j in zip(a, b):
        O[i, j] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)], float)
    assert F.qwk(a, b) == pytest.approx(1 - (W * O).sum() / (W * E).sum())
    assert F.qwk(a, a) == pytest.approx(1.0)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=27), rng.integers(0, 4, 27).astype(float)
    assert F.spearman(x, y) == pytest.approx(spearmanr(x, y)[0])
    y[3] = np.nan
    ok = np.isfinite(y)
    assert F.spearman(x, y) == pytest.approx(spearmanr(x[ok], y[ok])[0])


def test_sem_and_bound_dr_21_13():
    s = 10.0 * math.sqrt(1 - 0.9)
    assert s == pytest.approx(3.162, abs=0.001)
    assert F.bound(s) == pytest.approx(8.77, abs=0.01)
    v = np.array([0.0, 20.0])
    assert F.sem(v, 0.9) == pytest.approx(np.std(v, ddof=1) * math.sqrt(0.1))


def test_srm_and_cluster_bootstrap():
    assert F.srm([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert math.isnan(F.srm([0.1 + 0.2, 0.3, 0.30000000000000004]))     # equal changes: no SD
    lo, hi, bad = F.cluster_bootstrap(range(10), lambda d: float(np.mean(d)), B=500, seed=3)
    assert lo < 4.5 < hi and bad == 0


# ---------------------------------------------------------------- the precision printout
def test_precision_formula_fieller_hartley_pearson():
    assert F.fisher_halfwidth(27) == pytest.approx(1.96 * 1.03 / math.sqrt(24))
    assert F.fisher_halfwidth(27) == pytest.approx(0.41209, abs=1e-5)


def test_side_pooling_needs_both_sides():
    assert F.pool_sides([{"a": 1.0}, {"a": 3.0}]) == {"a": 2.0}
    assert math.isnan(F.pool_sides([{"a": 1.0}, {"a": math.nan}])["a"])
