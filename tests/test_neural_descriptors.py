"""mdfm.neural.descriptors on synthetic signals only (task T29): planted aperiodic exponents and peaks recovered, the
phase-slope lag returns +d within 2 SE for a body series delayed by d, bursts, ERD, the bandwidth and stimulation
refusals, and the NR-3.5 output schema."""
import math

import numpy as np
import pytest

from mdfm.neural import descriptors as D


# --- aperiodic and periodic fit (NR-3.3) ---------------------------------------------------------------------------

F = np.arange(0.5, 60.01, 0.5)


def planted_spectrum(seed, offset=1.0, exponent=1.5, peaks=(10.0, 0.6, 1.0, 22.0, 0.35, 1.5), knee=None, sd=0.02):
    rng = np.random.default_rng(seed)
    y = D.aperiodic_model(F, offset, exponent, knee) + D.gaussian_sum(F, list(peaks)) + rng.normal(0, sd, F.size)
    return 10 ** y


def test_fit_recovers_planted_exponent_offset_and_peaks():
    fit = D.spectral_fit(F, planted_spectrum(0))
    assert fit.exponent == pytest.approx(1.5, abs=0.03)
    assert fit.offset == pytest.approx(1.0, abs=0.03)
    assert fit.joint_fit_used and fit.r2 > 0.99
    assert len(fit.peaks) == 2
    p1, p2 = fit.peaks
    assert p1.center_hz == pytest.approx(10.0, abs=0.2) and p2.center_hz == pytest.approx(22.0, abs=0.2)
    assert p1.power_log10 == pytest.approx(0.6, abs=0.05) and p2.power_log10 == pytest.approx(0.35, abs=0.05)
    assert p1.power_db == pytest.approx(10 * p1.power_log10)
    assert p1.bandwidth_hz == pytest.approx(2.0, rel=0.2) and p2.bandwidth_hz == pytest.approx(3.0, rel=0.2)
    assert "joint_fit (OPEN-17)" in fit.draft and "freq_range (OPEN-17)" in fit.draft


def test_stepwise_ending_recovers_the_same_model():
    fit = D.spectral_fit(F, planted_spectrum(0), joint_fit=False)
    assert not fit.joint_fit_used
    assert fit.exponent == pytest.approx(1.5, abs=0.03)
    assert [round(p.center_hz) for p in fit.peaks] == [10, 22]


def test_knee_mode_recovers_knee_and_exponent():
    fit = D.spectral_fit(F, planted_spectrum(0, offset=2.0, exponent=2.0, knee=10.0, peaks=(12.0, 0.5, 1.2)),
                         mode="knee")
    assert fit.exponent == pytest.approx(2.0, abs=0.1)
    assert fit.knee == pytest.approx(10.0, rel=0.3)
    assert len(fit.peaks) == 1 and fit.peaks[0].center_hz == pytest.approx(12.0, abs=0.2)


@pytest.mark.parametrize("seed", range(5))
def test_peak_free_spectrum_gives_no_peak(seed):
    fit = D.spectral_fit(F, planted_spectrum(seed, peaks=()))
    assert fit.peaks == ()
    assert fit.exponent == pytest.approx(1.5, abs=0.03)
    bp = D.band_periodic_power(fit, D.BANDS["low_beta"], summary="max_peak")
    assert bp.no_peak and bp.value_db == 0.0 and "no_peak_value (OPEN-17)" in bp.draft


def synth_series(seed, exponent, peaks, fs=500.0, seconds=120.0):
    """A series whose spectrum is the planted aperiodic plus peaks model (random phases)."""
    n = int(seconds * fs)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    f1 = fr.copy()
    f1[0] = fr[1]
    log_s = D.aperiodic_model(f1, 0.0, exponent) + (D.gaussian_sum(f1, list(peaks)) if peaks else 0.0)
    amp = np.sqrt(10 ** log_s)
    amp[0] = 0.0
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, fr.size)
    return np.fft.irfft(amp * np.exp(1j * ph), n)


@pytest.mark.parametrize("seed", range(3))
def test_fit_on_a_time_series_recovers_planted_values(seed):
    x = synth_series(seed, 1.8, (11.0, 0.8, 1.5, 24.0, 0.4, 2.0))
    psd = D.welch_psd(x, 500.0)
    assert psd.freq_resolution_hz == pytest.approx(0.5) and psd.seg_s == 2.0
    fit = D.spectral_fit(psd.freqs, psd.power, freq_range=(2.0, 45.0))
    assert fit.exponent == pytest.approx(1.8, abs=0.05)
    assert [round(p.center_hz) for p in fit.peaks] == [11, 24]
    assert fit.peaks[0].center_hz == pytest.approx(11.0, abs=0.3)
    assert fit.peaks[0].power_log10 == pytest.approx(0.8, abs=0.08)
    assert fit.peaks[0].bandwidth_hz == pytest.approx(3.0, rel=0.2)
    assert fit.aperiodic_se["exponent"] < 0.05
    flat = D.welch_psd(synth_series(seed + 10, 1.8, ()), 500.0)
    no_peak = D.spectral_fit(flat.freqs, flat.power, freq_range=(2.0, 45.0))
    assert no_peak.peaks == () and no_peak.exponent == pytest.approx(1.8, abs=0.05)


def test_band_periodic_power_mean_flat_matches_the_planted_mean():
    fit = D.spectral_fit(F, planted_spectrum(1, sd=0.005))
    band = D.BANDS["theta_alpha"]
    bp = D.band_periodic_power(fit, band)
    mask = (F >= 4) & (F <= 12)
    planted = 10 * np.mean(D.gaussian_sum(F[mask], [10.0, 0.6, 1.0, 22.0, 0.35, 1.5]))
    assert bp.value_db == pytest.approx(planted, abs=0.15)
    assert bp.n_bins == int(mask.sum()) and "summary (OPEN-17)" in bp.draft
    mp = D.band_periodic_power(fit, band, summary="max_peak")
    assert mp.peak is not None and mp.value_db == pytest.approx(6.0, abs=0.5)


@pytest.mark.parametrize("seed", range(3))
def test_minimum_peak_height_keeps_a_small_peak_and_drops_a_smaller_one(seed):
    """Peaks of 0.13 and 0.07 log10 around the 0.1 log10 (1 dB) draft minimum: the first is reported, the second
    is not."""
    y = D.aperiodic_model(F, 1.0, 1.5) + D.gaussian_sum(F, [10.0, 0.13, 1.0, 30.0, 0.07, 1.5])
    fit = D.spectral_fit(F, 10 ** (y + np.random.default_rng(seed).normal(0, 0.005, F.size)))
    assert len(fit.peaks) == 1 and fit.peaks[0].center_hz == pytest.approx(10.0, abs=0.1)
    assert fit.peaks[0].power_log10 == pytest.approx(0.13, abs=0.015)


def test_peak_width_lower_limit_holds():
    """A planted peak of bandwidth 0.5 Hz (SD 0.25) on a 0.1 Hz grid is fitted at the 1 Hz draft lower limit, with or
    without the final joint fit; a lower limit of 0.2 Hz lets the fit reach 0.5 Hz."""
    f = np.arange(0.5, 60.01, 0.1)
    y = D.aperiodic_model(f, 1.0, 1.5) + D.gaussian_sum(f, [15.0, 0.6, 0.25])
    p = 10 ** (y + np.random.default_rng(0).normal(0, 0.005, f.size))
    for joint in (True, False):
        fit = D.spectral_fit(f, p, joint_fit=joint)
        assert len(fit.peaks) == 1 and fit.peaks[0].bandwidth_hz == pytest.approx(1.0, abs=1e-6)
        free = D.spectral_fit(f, p, joint_fit=joint, peak_width=(0.2, 12.0))
        assert free.peaks[0].bandwidth_hz == pytest.approx(0.5, abs=0.05)


def test_band_outside_the_fit_range_is_refused():
    """The draft fit range (1 to 45 Hz) does not hold the gamma band (60 to 90 Hz): both summaries refuse it rather
    than average part of the band or report 'no peak' on a range never fitted."""
    f = np.arange(0.5, 120.01, 0.5)
    p = 10 ** (D.aperiodic_model(f, 1.0, 1.5) + D.gaussian_sum(f, [75.0, 0.5, 3.0]))
    fit = D.spectral_fit(f, p)
    assert fit.freq_range == (1.0, 45.0)
    for summary in ("mean_flat", "max_peak"):
        with pytest.raises(ValueError, match="fit range"):
            D.band_periodic_power(fit, D.BANDS["gamma"], summary=summary)
    with pytest.raises(ValueError, match="fit range"):
        D.band_periodic_power(fit, (40.0, 50.0))
    wide = D.spectral_fit(f, p, freq_range=(30.0, 110.0))
    assert D.band_periodic_power(wide, D.BANDS["gamma"], summary="max_peak").value_db == pytest.approx(5.0, abs=0.1)


def test_spectral_fit_refusals():
    with pytest.raises(ValueError):
        D.spectral_fit(F, -planted_spectrum(0))
    with pytest.raises(ValueError):
        D.spectral_fit(F, planted_spectrum(0), freq_range=(0.0, 40.0))
    with pytest.raises(ValueError):
        D.spectral_fit(F, planted_spectrum(0), mode="lorentz")


# --- phase-slope lag (NR-3.8, NR-5.2) ------------------------------------------------------------------------------

def delayed_pair(seed, d_samples, fs=1000.0, seconds=120.0, noise=1.0):
    """neural x and body y = x delayed by d_samples plus independent noise, on one clock."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    z = rng.normal(size=n + abs(d_samples))
    if d_samples >= 0:
        x, y = z[d_samples:], z[:n]
    else:
        x, y = z[:n], z[-d_samples:]
    return x, y + noise * rng.normal(size=n)


@pytest.mark.parametrize("d_samples", [25, 120])
@pytest.mark.parametrize("fit", ["origin", "intercept"])
def test_phase_slope_returns_plus_d_within_two_se(d_samples, fit):
    x, y = delayed_pair(1, d_samples)
    r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0), fit=fit)
    d_ms = 1000.0 * d_samples / 1000.0
    assert abs(r.lag_ms - d_ms) <= 2 * r.se_ms, (r.lag_ms, r.se_ms)
    assert r.se_ms > 0 and r.n_seg == 120 and r.freq_resolution_hz == pytest.approx(1.0)
    assert D.lag_direction(r.lag_ms, r.se_ms) == "neural leads"


def test_phase_slope_sign_when_the_neural_series_lags():
    x, y = delayed_pair(7, -30)
    r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0))
    assert abs(r.lag_ms + 30.0) <= 2 * r.se_ms
    assert D.lag_direction(r.lag_ms, r.se_ms) == "body leads"


def test_phase_slope_without_coupling_rarely_claims_a_direction():
    """The coherence floor keeps incoherent bins out of the fit; without it eq. 6.7 understates the phase variance
    of incoherent bins and a direction is claimed on most uncoupled pairs."""
    estimable = claims = claims_no_floor = 0
    for s in range(100):
        rng = np.random.default_rng(1000 + s)
        x, y = rng.normal(size=60_000), rng.normal(size=60_000)
        r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0), seg_s=0.5)
        estimable += r.estimable
        claims += D.lag_direction(r.lag_ms, r.se_ms) != "none"
        r0 = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0), seg_s=0.5, coherence_alpha=None)
        claims_no_floor += D.lag_direction(r0.lag_ms, r0.se_ms) != "none"
    assert r.n_seg == 120 and r.coherence_floor == pytest.approx(D.coherence_null_limit(120, 0.05))
    assert estimable <= 10 and claims <= 10
    assert claims_no_floor >= 50
    assert not r.estimable or r.n_bins >= 3
    nan = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0), seg_s=0.5, min_bins=1000)
    assert not nan.estimable and math.isnan(nan.lag_ms) and D.lag_direction(nan.lag_ms, nan.se_ms) == "none"


def gapped_pair(seed, d_samples, fs=1000.0, seconds=120.0):
    """neural x white; body y = x band-stopped at 10 to 22 Hz (zero phase), delayed by d_samples, plus noise: the
    coherence falls under its floor inside a 4 to 40 Hz band, so the fitted bins are not contiguous."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    z = rng.normal(size=n + d_samples)
    sos = signal.butter(8, [10.0, 22.0], btype="bandstop", fs=fs, output="sos")
    return z[d_samples:], signal.sosfiltfilt(sos, z)[:n] + 0.5 * rng.normal(size=n)


@pytest.mark.parametrize("d_samples", [40, 120])
@pytest.mark.parametrize("fit", ["origin", "intercept"])
def test_phase_slope_across_a_coherence_gap(d_samples, fit):
    """The bins above the coherence floor need not be contiguous; unwrapping along frequency across the gap gave lags
    of 9.7 ms for 40 ms and 58 to 89 ms for 120 ms with SEs under 0.1 ms (review of T29)."""
    within, gapped, errs = 0, 0, []
    for seed in range(20):
        x, y = gapped_pair(seed, d_samples)
        r = D.phase_slope_lag(x, y, 1000.0, (4.0, 40.0), fit=fit)
        assert r.estimable and r.freqs.min() <= 10.0 and r.freqs.max() >= 22.0 and r.n_bins < r.n_band_bins
        gapped += np.diff(r.freqs).max() >= 5.0  # leakage lets a stray stop-band bin pass the floor on some seeds
        errs.append(abs(r.lag_ms - d_samples))
        within += errs[-1] <= 2 * r.se_ms
    assert gapped >= 15
    assert max(errs) < 1.0, errs
    assert within >= (18 if fit == "origin" else 16)


@pytest.mark.parametrize("seed", range(3))
def test_sign_inverted_body_series(seed):
    """A body series of opposite sign (a constant phase offset of pi, as a sensor mounted the other way gives): the
    intercept fit returns +d within 2 SE with an intercept near pi; the origin fit keeps every bin on one branch but,
    the model lacking the offset, misses d by about 24 ms with a tiny SE, which only its chi2_dof flags (a known
    limit of the draft origin fit, a point for OPEN-21)."""
    x, y = delayed_pair(seed, 25)
    i = D.phase_slope_lag(x, -y, 1000.0, (4.0, 30.0), fit="intercept")
    assert abs(i.lag_ms - 25.0) <= 2 * i.se_ms and abs(abs(i.intercept_rad) - math.pi) < 0.1 and i.chi2_dof < 2
    o = D.phase_slope_lag(x, -y, 1000.0, (4.0, 30.0), fit="origin")
    assert np.max(np.abs(np.diff(o.phase_rad) - 2 * math.pi * 0.025)) < 1.0  # no bin moved by a whole turn
    assert abs(o.lag_ms - 25.0) > 10 * o.se_ms and o.chi2_dof > 50


def test_lag_is_identified_modulo_the_segment_length():
    x, y = delayed_pair(1, 600)
    r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0))
    assert r.alias_period_ms == pytest.approx(1000.0)
    assert abs(r.lag_ms - (600.0 - r.alias_period_ms)) <= 2 * r.se_ms


@pytest.mark.parametrize("lag,level,expected", [
    (1.9, None, "none"), (2.0, None, "neural leads"), (-1.9, None, "none"), (-2.0, None, "body leads"),
    (1.6, 0.9, "none"), (1.7, 0.9, "neural leads"), (-1.7, 0.9, "body leads"),
])
def test_lag_direction_uses_the_two_sided_z(lag, level, expected):
    """NR-5.2: z_(1 - a/2) is 1.960 at the draft level 0.95 and 1.645 at 0.9 (SE 1)."""
    kw = {} if level is None else {"level": level}
    assert D.lag_direction(lag, 1.0, **kw) == expected


def test_coherence_null_limit():
    assert D.coherence_null_limit(2, 0.05) == pytest.approx(0.95)
    assert D.coherence_null_limit(120, 0.05) == pytest.approx(1 - 0.05 ** (1 / 119))
    with pytest.raises(ValueError):
        D.coherence_null_limit(1, 0.05)


def test_se_runs_low_near_the_coherence_null():
    """A known limit, pinned: eq. 6.7 is a large-sample approximation, and with the body noise six times the shared
    signal (coherence near the floor) the lag errors spread wider than the SE says."""
    z = []
    for s in range(200):
        x, y = delayed_pair(100 + s, 25, noise=6.0)
        r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0))
        if r.estimable:
            z.append((r.lag_ms - 25.0) / r.se_ms)
            line = 2 * math.pi * r.lag_ms / 1000.0 * r.freqs
            assert np.max(np.abs(r.phase_rad - line)) < math.pi  # every bin on the branch nearest the fitted line
    assert len(z) >= 190 and np.std(z) > 1.2


def test_phase_slope_se_is_calibrated_under_the_draft_taper():
    z = []
    for s in range(200):
        x, y = delayed_pair(100 + s, 25)
        r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0))
        z.append((r.lag_ms - 25.0) / r.se_ms)
    z = np.asarray(z)
    assert np.mean(np.abs(z) <= 2) >= 0.9
    assert 0.85 <= np.std(z) <= 1.2
    assert "taper (OPEN-21)" in r.draft and "fit (OPEN-21)" in r.draft


def test_hann_taper_makes_the_se_run_low():
    """Why the draft taper is boxcar: Hann correlates adjacent bins that the fit treats as independent."""
    z = []
    for s in range(200):
        x, y = delayed_pair(100 + s, 25)
        r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0), taper="hann")
        z.append((r.lag_ms - 25.0) / r.se_ms)
    assert np.std(z) > 1.15


def test_phase_slope_weights_follow_halliday_eq_6_7():
    x, y = delayed_pair(2, 10)
    r = D.phase_slope_lag(x, y, 1000.0, (4.0, 30.0))
    var = (1.0 / (2.0 * r.n_seg)) * (1.0 / r.coherence - 1.0)
    assert np.allclose(r.weights, 1.0 / var)
    se_slope = 1.0 / math.sqrt(float(np.sum(r.weights * r.freqs ** 2)))
    assert r.se_ms == pytest.approx(1000.0 * se_slope / (2 * math.pi))


def test_cross_spectra_needs_one_clock():
    with pytest.raises(ValueError):
        D.cross_spectra(np.zeros(1000), np.zeros(999), 100.0)


# --- bursts (NR-3.6) and ERD (NR-3.7) ------------------------------------------------------------------------------

def burst_record(seed=3, fs=1000.0, seconds=60.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = 0.1 * rng.normal(size=n)
    durs = np.linspace(0.15, 0.6, 20)
    ons = 1.5 + np.arange(20) * 2.9
    for o, d in zip(ons, durs):
        m = (t >= o) & (t < o + d)
        x[m] += np.sin(2 * np.pi * 20.0 * t[m])
    return x, ons, durs


def test_planted_bursts_are_recovered():
    x, ons, durs = burst_record()
    b = D.detect_bursts(x, 1000.0, (13.0, 30.0), threshold=0.5)
    assert b.n_bursts == 20 and b.rate_per_min == pytest.approx(20.0)
    assert np.all(np.abs(b.durations_ms - 1000 * durs) <= 15.0)
    assert np.all(np.abs(b.onsets_s - ons) <= 0.01)
    assert b.threshold_source == "given" and "threshold_percentile (OPEN-19)" not in b.draft
    assert "min_duration_s (OPEN-19)" in b.draft


def edge_record(fs=1000.0):
    """10 s: a burst cut by the start (0 to 0.4 s), bursts of 0.11 s at 2 s and 0.09 s at 4 s, one of 0.3 s at 6 s,
    and one cut by the end (9.7 s to the last sample)."""
    t = np.arange(int(10 * fs)) / fs
    x = 0.05 * np.random.default_rng(0).normal(size=t.size)
    for o, d in ((0.0, 0.4), (2.0, 0.11), (4.0, 0.09), (6.0, 0.3), (9.7, 0.4)):
        m = (t >= o) & (t < o + d)
        x[m] += np.sin(2 * np.pi * 20.0 * t[m])
    return x


def test_short_and_edge_bursts_are_excluded():
    fs = 1000.0
    x = edge_record(fs)
    b = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5)
    assert b.n_bursts == 2 and b.n_short_excluded == 1 and b.n_edge_excluded == 2
    assert b.onsets_s == pytest.approx([2.0, 6.0], abs=0.01)  # the 0.11 s burst is kept, the 0.09 s one is not
    assert "edge_guard_cycles (OPEN-19)" in b.draft
    inc = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5, edge_policy="include")
    assert inc.n_bursts == 4 and inc.n_edge_excluded == 0
    assert "edge_guard_cycles (OPEN-19)" not in inc.draft
    # Why the guard: the edge artefacts pull the envelope of the burst cut by the end under the threshold before the
    # last sample, so a rule on the edge sample alone counts that burst.
    bare = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5, edge_guard_cycles=0.0)
    assert bare.n_bursts == 3 and bare.onsets_s[-1] == pytest.approx(9.7, abs=0.01)


def test_minimum_duration_is_inclusive():
    fs = 1000.0
    x = edge_record(fs)
    every = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5, min_duration_s=0.0)
    i = int(np.argmin(np.abs(every.onsets_s - 2.0)))
    n_samples = int(round(every.durations_ms[i] * fs / 1000.0))
    at = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5, min_duration_s=n_samples / fs)
    assert np.any(np.abs(at.onsets_s - 2.0) < 0.01)  # a run of exactly the minimum is kept
    above = D.detect_bursts(x, fs, (13.0, 30.0), threshold=0.5, min_duration_s=(n_samples + 1) / fs)
    assert not np.any(np.abs(above.onsets_s - 2.0) < 0.01)


@pytest.mark.parametrize("band,f0", [((13.0, 30.0), 20.0), ((13.0, 30.0), 14.0), ((4.0, 12.0), 6.0),
                                     ((60.0, 90.0), 75.0)])
def test_bursts_cut_by_either_record_edge_are_excluded(band, f0):
    """Bursts of 0.5 s cut by the start and by the end, at 10 phases and record lengths and two thresholds: none is
    counted under the draft guard."""
    fs = 1000.0
    for k in range(10):
        n = int((10.0 + k * 0.0137) * fs)
        t = np.arange(n) / fs
        x = 0.05 * np.random.default_rng(k).normal(size=n)
        for m, ph in ((t < 0.5, 4 * np.pi * k / 10), (t >= t[-1] - 0.5, 2 * np.pi * k / 10)):
            x[m] += np.sin(2 * np.pi * f0 * t[m] + ph)
        for thr in (0.5, 0.7):
            b = D.detect_bursts(x, fs, band, threshold=thr, min_duration_s=0.0)
            assert b.n_bursts == 0 and b.n_edge_excluded >= 2, (k, thr, b.onsets_s)


def test_percentile_threshold_path():
    x, _, _ = burst_record()
    b = D.detect_bursts(x, 1000.0, (13.0, 30.0))
    env = D.band_envelope(x, 1000.0, (13.0, 30.0))
    assert b.threshold == pytest.approx(np.percentile(env, 75.0))
    assert "threshold_percentile (OPEN-19)" in b.draft


def test_erd_formula_and_planted_desynchronization():
    assert D.erd_percent(0.25, 1.0) == pytest.approx(-75.0)
    assert D.erd_percent(1.5, 1.0) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        D.erd_percent(1.0, 0.0)
    fs, n_tr, tr = 250.0, 40, 6.0
    n = int(n_tr * tr * fs)
    t = np.arange(n) / fs
    amp = np.full(n, 2.0)
    ons = 3.0 + np.arange(n_tr) * tr
    for o in ons:
        amp[(t >= o) & (t < o + 2.0)] = 1.0
    x = amp * np.sin(2 * np.pi * 10.0 * t) + 0.05 * np.random.default_rng(4).normal(size=n)
    e = D.erd(x, fs, (8.0, 12.0), ons, event_window_s=(0.5, 1.5))
    assert e.erd_percent == pytest.approx(-75.0, abs=3.0)
    assert e.n_trials == 40 and e.n_dropped == 0
    assert set(e.draft) == {"baseline_window_s (OPEN-20)", "order (OPEN-20)"}
    e2 = D.erd(x, fs, (8.0, 12.0), np.concatenate([[0.5], ons]), event_window_s=(0.5, 1.5))
    assert e2.n_dropped == 1


# --- refusals (NR-3.14, NR-3.16) -----------------------------------------------------------------------------------

def test_bandwidth_refusal_percept_and_ear_eeg():
    assert not D.bandwidth_refusal((60.0, 90.0), 250.0, lowpass_hz=100.0).refused
    r = D.bandwidth_refusal((100.0, 125.0), 250.0, lowpass_hz=125.0)
    assert r.refused and "fs / 2" in r.reason  # the 250 Hz Percept stream carries nothing above 125 Hz
    assert D.bandwidth_refusal((100.0, 128.0), 256.0, lowpass_hz=128.0).refused
    assert not D.bandwidth_refusal((100.0, 127.0), 256.0, undeclared="nyquist_only").refused


def test_bandwidth_refusal_declared_and_undeclared_corner():
    assert D.bandwidth_refusal((80.0, 110.0), 1000.0, lowpass_hz=100.0).refused
    assert not D.bandwidth_refusal((60.0, 100.0), 1000.0, lowpass_hz=100.0).refused  # at the corner, not above
    r = D.bandwidth_refusal((13.0, 30.0), 1000.0)
    assert r.refused and "OPEN-23" in r.reason and r.draft == ("undeclared (OPEN-23)",)
    assert not D.bandwidth_refusal((13.0, 30.0), 1000.0, undeclared="nyquist_only").refused


def test_stimulation_exclusion_at_130_hz():
    s = D.stimulation_exclusion(D.BANDS["gamma"], 130.0)
    assert s.excluded == ((63.0, 67.0),)
    assert s.kept == ((60.0, 63.0), (67.0, 90.0))
    assert not s.refused and s.stim_freq_hz == 130.0
    assert set(s.draft) == {"mode (OPEN-24)", "half_width_hz (OPEN-24)", "multiples (OPEN-24)"}
    f = np.arange(55.0, 95.0, 1.0)
    m = D.frequency_mask(f, s.kept)
    assert not m[(f > 63) & (f < 67)].any() and m[(f >= 60) & (f <= 63)].all()
    off = D.stimulation_exclusion(D.BANDS["gamma"], 0.0)
    assert off.kept == ((60.0, 90.0),) and off.excluded == ()


def test_stimulation_exclusion_refusals():
    with pytest.raises(ValueError):
        D.stimulation_exclusion(D.BANDS["gamma"], None)
    with pytest.raises(NotImplementedError):
        D.stimulation_exclusion(D.BANDS["gamma"], 130.0, mode="model")
    whole = D.stimulation_exclusion((64.0, 66.0), 130.0)
    assert whole.refused and whole.kept == ()
    both = D.stimulation_exclusion((60.0, 140.0), 130.0, multiples=(0.5, 1.0))
    assert both.kept == ((60.0, 63.0), (67.0, 128.0), (132.0, 140.0))


# --- output schema (NR-3.5) ----------------------------------------------------------------------------------------

def test_band_descriptor_measured_row_has_every_field():
    d = D.band_descriptor("periodic_power_low_beta", "dB", D.BANDS["low_beta"], 1000.0, window_s=1.0,
                          freq_resolution_hz=1.0, value=3.2, interval=(2.1, 4.0), interval_level=0.95,
                          interval_method="patient-cluster bootstrap, enumerated", lowpass_hz=400.0)
    row = d.to_dict()
    assert tuple(row) == D.SCHEMA_FIELDS
    assert row["status"] == "measured" and row["value"] == 3.2 and row["interval_lo"] == 2.1
    assert "Nyquist 500 Hz" in row["bandwidth_statement"]


def test_band_descriptor_refused_band_carries_no_value():
    d = D.band_descriptor("periodic_power_gamma", "dB", D.BANDS["gamma"], 128.0, window_s=1.0,
                          freq_resolution_hz=1.0, value=1.0, interval=(0.0, 2.0), interval_level=0.95,
                          interval_method="x", stim_freq_hz=0.0, lowpass_hz=60.0)
    assert d.status == "refused" and d.value is None and d.interval is None
    assert "fs / 2" in d.refusal_reason
    row = D.descriptor_rows([d])[0]
    assert row["value"] is None and row["interval_lo"] is None


@pytest.mark.parametrize("change,needle", [
    (dict(interval=None), "interval"),
    (dict(status="estimated"), "status"),
    (dict(bandwidth_statement=""), "bandwidth"),
    (dict(window_s=0.0), "window_s"),
    (dict(freq_resolution_hz=None), "freq_resolution_hz"),
    (dict(interval_level=None), "level"),
    (dict(interval=(2.0, 1.0)), "lo <= hi"),
    (dict(status="refused"), "no value"),
    (dict(band_hz=(60.0, 90.0)), "stimulation frequency"),
    (dict(interval_method=None), "method"),
    (dict(refusal_reason="x"), "refusal"),
])
def test_schema_refuses_incomplete_values(change, needle):
    base = dict(name="x", unit="dB", status="measured", value=1.0, interval=(0.5, 1.5), interval_level=0.95,
                interval_method="m", band_hz=(13.0, 20.0), bandwidth_statement="fs 1000 Hz", window_s=1.0,
                freq_resolution_hz=1.0, fs_hz=1000.0)
    D.DescriptorValue(**base).validate()
    base.update(change)
    with pytest.raises(ValueError, match=needle):
        D.DescriptorValue(**base).to_dict()


def test_prior_only_status_is_written_with_its_interval():
    d = D.DescriptorValue("head_acc_regressor", "g", "prior-only", 0.0, (-0.1, 0.1), 0.95, "prior", None,
                          "no head sensor", 2.0, 0.5, 250.0)
    assert d.to_dict()["status"] == "prior-only"
