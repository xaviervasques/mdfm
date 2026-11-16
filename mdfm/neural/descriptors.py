"""Neural descriptors of the neural rule file (docs/rules/NEURAL_RULES_DRAFT.md, section 3, DRAFT).

Status: DRAFT. Every default that settles an OPEN item is a Draft value (mdfm.neural.stats.Draft) and every result
lists the draft defaults it used (field ``draft``).

Contents and rules:

- BANDS, HOP_S, MIN_WINDOW_S: the pre-registered bands of NR-3.1 and the windows of NR-3.4.
- welch_psd, spectral_fit, band_periodic_power: the aperiodic and periodic fit of NR-3.3 (spectral parametrization
  after Donoghue et al. 2020: aperiodic component plus Gaussian peaks in semilog space, dB = 10 x log10 units;
  settings OPEN-17), written from the method description with scipy, not a port of a reference package.
- band_envelope, detect_bursts: bursts of NR-3.6 (rate per minute, duration in ms, native-rate envelope; envelope,
  threshold, minimum duration and the edge rule OPEN-19).
- erd_percent, erd: event-related desynchronization of NR-3.7 (baseline window OPEN-20).
- cross_spectra, phase_slope_lag, lag_direction: the phase-slope lag of NR-3.8 with the weights 1 / var(phase) of
  Halliday et al. 1995 eq. 6.7 (segment length, slope frequencies and the origin or intercept fit OPEN-21); a
  positive lag means the neural series leads; the direction rule of NR-5.2.
- bandwidth_refusal, stimulation_exclusion, frequency_mask: the refusals of NR-3.14 (undeclared low-pass corner
  OPEN-23) and NR-3.16 (half-width and exclusion or entrainment model OPEN-24).
- DescriptorValue, bandwidth_statement, band_descriptor, descriptor_rows: the output schema of NR-3.5.

No function here reads data; every input is an array the caller gives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize, signal, stats as sstats

from .stats import LEVEL, Draft, resolve

# ---------------------------------------------------------------------------------------------------------------
# Bands and windows (NR-3.1, NR-3.4), fixed by the design
# ---------------------------------------------------------------------------------------------------------------

BANDS = {"theta_alpha": (4.0, 12.0), "low_beta": (13.0, 20.0), "high_beta": (20.0, 30.0), "gamma": (60.0, 90.0)}
HOP_S = 0.25
MIN_WINDOW_S = {"theta_alpha": 2.0, "aperiodic": 2.0, "beta": 1.0}

# ---------------------------------------------------------------------------------------------------------------
# Draft defaults
# ---------------------------------------------------------------------------------------------------------------

BAND_EDGES = Draft("closed", "OPEN-15", "a band [lo, hi] holds both edge bins")
PSD_SEG_S = Draft(2.0, "OPEN-17", "Welch segment length, the 2 s minimum of aperiodic fits (NR-3.4)")
PSD_OVERLAP = Draft(0.5, "OPEN-17", "Welch segment overlap")
PSD_WINDOW = Draft("hann", "OPEN-17", "Welch taper")
FIT_RANGE = Draft((1.0, 45.0), "OPEN-17", "fit range in Hz; it does not cover the gamma band (60 to 90 Hz), so "
                                          "a gamma descriptor needs a fit range of its own (OPEN-17, OPEN-18) and "
                                          "band_periodic_power refuses a band outside the fit range")
AP_MODE = Draft("fixed", "OPEN-17", "aperiodic mode: 'fixed' (offset, exponent) or 'knee'")
PEAK_WIDTH = Draft((1.0, 12.0), "OPEN-17", "peak bandwidth limits in Hz (bandwidth = 2 Gaussian SD)")
MAX_PEAKS = Draft(6, "OPEN-17", "maximum number of peaks")
MIN_PEAK_HEIGHT = Draft(0.1, "OPEN-17", "minimum peak height in log10 power above the aperiodic fit (1 dB); at 0, "
                                         "noise maxima of a peak-free spectrum enter as peaks")
PEAK_THRESHOLD = Draft(2.0, "OPEN-17", "a peak candidate must exceed this many SD of the flattened spectrum")
EDGE_SD = Draft(1.0, "OPEN-17", "a candidate within this many Gaussian SD of a range edge is dropped")
OVERLAP_SD = Draft(0.75, "OPEN-17", "of two candidates overlapping within this many SD, the lower is dropped")
JOINT_FIT = Draft(True, "OPEN-17", "final joint least-squares fit of the aperiodic part and the peaks; False ends "
                                   "with the aperiodic refit on the peak-removed spectrum")
BAND_SUMMARY = Draft("mean_flat", "OPEN-17", "mean over the band of log power minus the aperiodic fit, in dB")
NO_PEAK_DB = Draft(0.0, "OPEN-17", "value written by the 'max_peak' summary when no peak is fitted in the band")
BURST_METHOD = Draft("hilbert", "OPEN-19", "band-pass then Hilbert amplitude at native rate")
BURST_ORDER = Draft(4, "OPEN-19", "Butterworth order of the zero-phase band-pass")
BURST_SMOOTH_S = Draft(0.0, "OPEN-19", "moving-average smoothing of the envelope (the literature candidate uses 0.2 s)")
BURST_PERCENTILE = Draft(75.0, "OPEN-19", "threshold percentile of the record's envelope (Tinkhauser et al. 2017 used "
                                          "the 75th, per hemisphere over OFF and ON)")
BURST_MIN_S = Draft(0.1, "OPEN-19", "minimum burst duration in s")
BURST_EDGE = Draft("exclude", "OPEN-19", "a burst cut by the record start or end is not counted")
BURST_EDGE_GUARD = Draft(3.0, "OPEN-19", "under 'exclude', a run starting or ending within this many cycles of the "
                                         "band's lower edge of the record start or end counts as cut: the band-pass "
                                         "and Hilbert edge artefacts pull the envelope of a cut burst under threshold "
                                         "before the last sample, so touching the edge sample alone misses it")
ERD_BASELINE = Draft((-2.0, -1.0), "OPEN-20", "baseline window in s relative to each event onset")
ERD_ORDER = Draft(4, "OPEN-20", "Butterworth order of the zero-phase band-pass; the register does not name it")
PSL_SEG_S = Draft(1.0, "OPEN-21", "disjoint segment length in s (1 Hz resolution)")
PSL_TAPER = Draft("boxcar", "OPEN-21", "no taper: a Hann taper correlates adjacent bins, which the weighted fit "
                                       "treats as independent, so its SE runs low")
PSL_FIT = Draft("origin", "OPEN-21", "phase regressed through the origin as in Rosenberg et al. 1989")
PSL_COH_ALPHA = Draft(0.05, "OPEN-21", "bins enter the slope fit only where coherence exceeds its null limit at "
                                       "this level, 1 - alpha^(1 / (n_seg - 1)); None lets every band bin in")
PSL_MIN_BINS = Draft(3, "OPEN-21", "fewest bins a slope is fitted on; fewer leave the lag not estimable")
UNDECLARED_LOWPASS = Draft("refuse", "OPEN-23", "a record without a declared low-pass corner: every band refused; "
                                                "'nyquist_only' applies the fs / 2 test alone")
STIM_HALF_WIDTH = Draft(2.0, "OPEN-24", "half-width in Hz excluded around each stimulation multiple")
STIM_MULTIPLES = Draft((0.5,), "OPEN-24", "multiples of the stimulation frequency excluded (NR-3.16 names one half)")
STIM_MODE = Draft("exclude", "OPEN-24", "'exclude'; 'model' (entrainment model) is not implemented")


def _band(band) -> tuple[float, float]:
    lo, hi = float(band[0]), float(band[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or hi <= lo:
        raise ValueError(f"band must satisfy 0 <= lo < hi, got {band}")
    return lo, hi


def band_bins(freqs, band, *, edges=BAND_EDGES) -> np.ndarray:
    """Boolean mask of the bins of ``freqs`` inside ``band``: 'closed' [lo, hi] or 'half_open' [lo, hi)."""
    e = resolve(edges, "edges")
    lo, hi = _band(band)
    f = np.asarray(freqs, float)
    if e == "closed":
        return (f >= lo) & (f <= hi)
    if e == "half_open":
        return (f >= lo) & (f < hi)
    raise ValueError("edges must be 'closed' or 'half_open'")


def _overlaps(band, other) -> bool:
    return band[0] < other[1] and band[1] > other[0]


# ---------------------------------------------------------------------------------------------------------------
# Spectral estimate and the aperiodic plus periodic fit (NR-3.3)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PSD:
    freqs: np.ndarray
    power: np.ndarray
    fs: float
    seg_s: float
    freq_resolution_hz: float
    n_seg: int
    draft: tuple = ()


def welch_psd(x, fs: float, *, seg_s=PSD_SEG_S, overlap=PSD_OVERLAP, window=PSD_WINDOW) -> PSD:
    """Welch power spectral density (density scaling, mean removed per segment)."""
    used: list = []
    seg = float(resolve(seg_s, "seg_s", used))
    ov = float(resolve(overlap, "overlap", used))
    win = resolve(window, "window", used)
    a = np.asarray(x, float).ravel()
    nper = int(round(seg * fs))
    if nper < 8 or nper > a.size:
        raise ValueError("segment length out of range for this series")
    nover = int(round(ov * nper))
    f, p = signal.welch(a, fs=fs, window=win, nperseg=nper, noverlap=nover, detrend="constant", scaling="density")
    n_seg = 1 + (a.size - nper) // (nper - nover)
    return PSD(f, p, float(fs), seg, float(fs) / nper, int(n_seg), tuple(used))


def aperiodic_model(freqs, offset: float, exponent: float, knee: float | None = None) -> np.ndarray:
    """log10 aperiodic power: offset - exponent * log10(f) ('fixed') or offset - log10(knee + f^exponent) ('knee')."""
    f = np.asarray(freqs, float)
    if knee is None:
        return offset - exponent * np.log10(f)
    return offset - np.log10(knee + f ** exponent)


def gaussian_sum(freqs, params) -> np.ndarray:
    """Sum of Gaussians, params flat (center, height, sd) repeated."""
    f = np.asarray(freqs, float)
    out = np.zeros_like(f)
    p = np.asarray(params, float).reshape(-1, 3)
    for c, h, s in p:
        out += h * np.exp(-((f - c) ** 2) / (2.0 * s * s))
    return out


def _ap_func(mode):
    if mode == "fixed":
        return lambda f, off, exp_: aperiodic_model(f, off, exp_)
    return lambda f, off, knee, exp_: aperiodic_model(f, off, exp_, knee)


def _fit_aperiodic(f, y, mode, p0=None):
    """Least-squares aperiodic fit; returns (params, covariance or None). Fixed mode is linear in log10 f."""
    if mode == "fixed":
        a = np.column_stack([np.ones_like(f), -np.log10(f)])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        dof = max(1, f.size - 2)
        resid = y - a @ coef
        s2 = float(resid @ resid) / dof
        try:
            cov = s2 * np.linalg.inv(a.T @ a)
        except np.linalg.LinAlgError:
            cov = None
        return coef, cov
    if p0 is None:
        fixed, _ = _fit_aperiodic(f, y, "fixed")
        p0 = [fixed[0], 0.0, max(fixed[1], 1e-3)]
    p0 = [p0[0], max(p0[1], 0.0), p0[2]]
    popt, pcov = optimize.curve_fit(_ap_func("knee"), f, y, p0=p0,
                                    bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=20000)
    return popt, pcov


def _ap_eval(f, params, mode):
    if mode == "fixed":
        return aperiodic_model(f, params[0], params[1])
    return aperiodic_model(f, params[0], params[2], params[1])


@dataclass(frozen=True)
class Peak:
    """A fitted peak: center frequency, power above the aperiodic fit at the center (log10 and dB), bandwidth (two
    Gaussian SD, as the source reports it), and the Gaussian parameters."""
    center_hz: float
    power_log10: float
    power_db: float
    bandwidth_hz: float
    gauss_height: float
    gauss_sd: float


@dataclass(frozen=True)
class SpectralFit:
    """One aperiodic plus periodic fit of a spectrum on its fit range (log10 power)."""
    freqs: np.ndarray
    log_power: np.ndarray
    mode: str
    offset: float
    exponent: float
    knee: float | None
    aperiodic_se: dict
    peaks: tuple
    aperiodic_fit: np.ndarray
    model_fit: np.ndarray
    r2: float
    mean_abs_error: float
    freq_range: tuple
    freq_resolution_hz: float
    joint_fit_used: bool
    draft: tuple = ()

    def flattened_db(self) -> np.ndarray:
        """10 x (log10 power - aperiodic fit): the periodic part in dB above the aperiodic fit, per bin."""
        return 10.0 * (self.log_power - self.aperiodic_fit)


def spectral_fit(freqs, power, *, freq_range=FIT_RANGE, mode=AP_MODE, peak_width=PEAK_WIDTH, max_peaks=MAX_PEAKS,
                 min_peak_height=MIN_PEAK_HEIGHT, peak_threshold=PEAK_THRESHOLD, edge_sd=EDGE_SD,
                 overlap_sd=OVERLAP_SD, joint_fit=JOINT_FIT) -> SpectralFit:
    """Aperiodic plus periodic fit of a power spectrum (NR-3.3), after Donoghue et al. 2020.

    Steps: (1) aperiodic fit on the whole range, then refit on the bins at or below that first fit (robust start);
    (2) on the flattened spectrum (log10 power minus that fit), iterative peak candidates: the highest bin, stopped
    when it falls under peak_threshold x SD of the flattened spectrum or under min_peak_height, its SD guessed from
    the shorter half-height side and clipped to the width limits, then subtracted; (3) candidates within edge_sd SD
    of a range edge dropped, and of two candidates overlapping within overlap_sd SD the lower dropped; (4) a joint
    least-squares fit of the remaining Gaussians to the flattened spectrum (centers within 1.5 SD of the guess);
    (5) the aperiodic refit on the peak-removed spectrum; (6) with joint_fit, a final least-squares fit of the
    aperiodic part and every Gaussian together, from the step 4 and 5 values. Offset and exponent come from that one
    fit; peak power is the summed Gaussians at the peak center, in log10 and in dB (10 x log10).
    """
    used: list = []
    f_lo, f_hi = resolve(freq_range, "freq_range", used)
    md = resolve(mode, "mode", used)
    w_lo, w_hi = resolve(peak_width, "peak_width", used)
    n_max = int(resolve(max_peaks, "max_peaks", used))
    h_min = float(resolve(min_peak_height, "min_peak_height", used))
    thr = float(resolve(peak_threshold, "peak_threshold", used))
    e_sd = float(resolve(edge_sd, "edge_sd", used))
    o_sd = float(resolve(overlap_sd, "overlap_sd", used))
    joint = bool(resolve(joint_fit, "joint_fit", used))
    if md not in ("fixed", "knee"):
        raise ValueError("mode must be 'fixed' or 'knee'")
    if not 0 < f_lo < f_hi:
        raise ValueError("fit range must satisfy 0 < lo < hi")
    f_all = np.asarray(freqs, float).ravel()
    p_all = np.asarray(power, float).ravel()
    sel = (f_all >= f_lo) & (f_all <= f_hi)
    f, p = f_all[sel], p_all[sel]
    if f.size < 8:
        raise ValueError("fewer than 8 bins in the fit range")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("power must be finite and positive in the fit range")
    y = np.log10(p)
    df = float(np.median(np.diff(f)))
    sd_lo, sd_hi = w_lo / 2.0, w_hi / 2.0

    # (1) robust aperiodic start
    ap0, _ = _fit_aperiodic(f, y, md)
    below = (y - _ap_eval(f, ap0, md)) <= 0
    if below.sum() > len(ap0) + 2:
        ap0, _ = _fit_aperiodic(f[below], y[below], md, p0=ap0)
    flat = y - _ap_eval(f, ap0, md)

    # (2) iterative peak candidates
    it = flat.copy()
    it[it < 0] = 0.0
    guesses = []
    while len(guesses) < n_max:
        i = int(np.argmax(it))
        h = float(it[i])
        if h <= thr * float(np.std(it)) or h <= h_min:
            break
        half = 0.5 * h
        left = next((j for j in range(i - 1, -1, -1) if it[j] <= half), None)
        right = next((j for j in range(i + 1, it.size) if it[j] <= half), None)
        sides = [abs(j - i) for j in (left, right) if j is not None]
        sd = (2.0 * min(sides) * df) / (2.0 * math.sqrt(2.0 * math.log(2.0))) if sides else (sd_lo + sd_hi) / 2.0
        sd = float(np.clip(sd, sd_lo, sd_hi))
        guesses.append([float(f[i]), h, sd])
        it = it - gaussian_sum(f, [f[i], h, sd])

    # (3) edge and overlap drops
    guesses = [g for g in guesses if abs(g[0] - f[0]) > e_sd * g[2] and abs(g[0] - f[-1]) > e_sd * g[2]]
    guesses.sort(key=lambda g: g[0])
    drop = set()
    for a in range(len(guesses) - 1):
        g1, g2 = guesses[a], guesses[a + 1]
        if g1[0] + o_sd * g1[2] > g2[0] - o_sd * g2[2]:
            drop.add(a if g1[1] < g2[1] else a + 1)
    guesses = [g for k, g in enumerate(guesses) if k not in drop]

    # (4) joint Gaussian fit on the flattened spectrum
    gp = np.zeros(0)
    if guesses:
        p0, lo_b, hi_b = [], [], []
        for c, h, s in guesses:
            p0 += [c, h, s]
            lo_b += [max(f[0], c - 1.5 * s), 0.0, sd_lo]
            hi_b += [min(f[-1], c + 1.5 * s), np.inf, sd_hi]
        p0 = np.clip(p0, lo_b, hi_b)
        try:
            gp, _ = optimize.curve_fit(lambda ff, *q: gaussian_sum(ff, q), f, flat, p0=p0, bounds=(lo_b, hi_b),
                                       maxfev=20000)
        except (RuntimeError, ValueError):
            gp = np.asarray(p0, float)

    # (5) aperiodic refit on the peak-removed spectrum
    ap, ap_cov = _fit_aperiodic(f, y - gaussian_sum(f, gp), md, p0=ap0)
    joint_used = False

    # (6) final joint fit
    if joint and gp.size:
        n_ap = len(ap)
        ap_lo = [-np.inf, -np.inf] if md == "fixed" else [-np.inf, 0.0, -np.inf]
        p0 = list(ap) + list(gp)
        g3 = gp.reshape(-1, 3)
        lo_b = ap_lo + [v for c, _, s in g3 for v in (max(f[0], c - 1.5 * s), 0.0, sd_lo)]
        hi_b = [np.inf] * n_ap + [v for c, _, s in g3 for v in (min(f[-1], c + 1.5 * s), np.inf, sd_hi)]
        p0 = np.clip(p0, lo_b, hi_b)
        apf = _ap_func(md)

        def full(ff, *q):
            return apf(ff, *q[:n_ap]) + gaussian_sum(ff, q[n_ap:])
        try:
            popt, pcov = optimize.curve_fit(full, f, y, p0=p0, bounds=(lo_b, hi_b), maxfev=40000)
            ap, gp, ap_cov = popt[:n_ap], popt[n_ap:], pcov[:n_ap, :n_ap]
            joint_used = True
        except (RuntimeError, ValueError):
            pass

    ap_fit = _ap_eval(f, ap, md)
    model = ap_fit + gaussian_sum(f, gp)
    ss_res = float(np.sum((y - model) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    se = {}
    if ap_cov is not None and np.all(np.isfinite(ap_cov)):
        diag = np.sqrt(np.clip(np.diag(ap_cov), 0, None))
        se = {"offset": float(diag[0]), "exponent": float(diag[-1])}
        if md == "knee":
            se["knee"] = float(diag[1])
    peaks = []
    g = np.asarray(gp, float).reshape(-1, 3)
    for c, h, s in sorted(g.tolist(), key=lambda r: r[0]):
        pw = float(gaussian_sum([c], g.ravel())[0])
        peaks.append(Peak(float(c), pw, 10.0 * pw, 2.0 * float(s), float(h), float(s)))
    return SpectralFit(
        freqs=f, log_power=y, mode=md, offset=float(ap[0]), exponent=float(ap[-1]),
        knee=float(ap[1]) if md == "knee" else None, aperiodic_se=se, peaks=tuple(peaks), aperiodic_fit=ap_fit,
        model_fit=model, r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        mean_abs_error=float(np.mean(np.abs(y - model))), freq_range=(float(f_lo), float(f_hi)),
        freq_resolution_hz=df, joint_fit_used=joint_used, draft=tuple(used))


@dataclass(frozen=True)
class BandPower:
    value_db: float
    summary: str
    n_bins: int
    peak: Peak | None
    no_peak: bool
    draft: tuple = ()


def band_periodic_power(fit: SpectralFit, band, *, summary=BAND_SUMMARY, no_peak_value=NO_PEAK_DB,
                        edges=BAND_EDGES) -> BandPower:
    """Periodic power in a band, in dB above the aperiodic fit (NR-3.3, OPEN-17).

    'mean_flat': the mean over the band's bins of 10 x (log10 power - aperiodic fit). 'max_peak': the power of the
    largest fitted peak whose center lies in the band, or ``no_peak_value`` when none does.

    A band not wholly inside the fit range is refused (ValueError) under both summaries: a mean over part of the band
    would misreport it, and 'no peak' on a range never fitted would be false. The draft fit range (1 to 45 Hz) does
    not cover the gamma band of NR-3.1 (60 to 90 Hz): a gamma descriptor needs a fit range chosen for it (OPEN-17,
    with the gamma window of OPEN-18).
    """
    used: list = []
    summ = resolve(summary, "summary", used)
    e = resolve(edges, "edges", used)
    lo, hi = _band(band)
    r_lo, r_hi = fit.freq_range
    if lo < r_lo or hi > r_hi:
        raise ValueError(f"band {lo:g} to {hi:g} Hz is not inside the fit range {r_lo:g} to {r_hi:g} Hz (OPEN-17)")
    mask = band_bins(fit.freqs, band, edges=e)
    if summ == "mean_flat":
        if not mask.any():
            raise ValueError("no bin of the fit inside the band")
        return BandPower(float(np.mean(fit.flattened_db()[mask])), summ, int(mask.sum()), None, False, tuple(used))
    if summ == "max_peak":
        inside = [pk for pk in fit.peaks if (lo <= pk.center_hz <= hi if e == "closed" else lo <= pk.center_hz < hi)]
        if not inside:
            v = float(resolve(no_peak_value, "no_peak_value", used))
            return BandPower(v, summ, int(mask.sum()), None, True, tuple(used))
        best = max(inside, key=lambda pk: pk.power_db)
        return BandPower(best.power_db, summ, int(mask.sum()), best, False, tuple(used))
    raise ValueError("summary must be 'mean_flat' or 'max_peak'")


# ---------------------------------------------------------------------------------------------------------------
# Bursts (NR-3.6) and event-related desynchronization (NR-3.7)
# ---------------------------------------------------------------------------------------------------------------


def _bandpass(x, fs, band, order):
    lo, hi = _band(band)
    if hi >= fs / 2:
        raise ValueError("band upper edge at or above fs / 2")
    sos = signal.butter(int(order), [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, float))


def band_envelope(x, fs: float, band, *, method=BURST_METHOD, order=BURST_ORDER, smooth_s=BURST_SMOOTH_S,
                  _used: list | None = None) -> np.ndarray:
    """Native-rate amplitude envelope of a band: zero-phase Butterworth band-pass, Hilbert amplitude, optional
    centered moving average of smooth_s seconds."""
    used = _used if _used is not None else []
    meth = resolve(method, "method", used)
    od = int(resolve(order, "order", used))
    sm = float(resolve(smooth_s, "smooth_s", used))
    if meth != "hilbert":
        raise NotImplementedError("only the Hilbert envelope is implemented (OPEN-19)")
    env = np.abs(signal.hilbert(_bandpass(x, fs, band, od)))
    w = int(round(sm * fs))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return env


@dataclass(frozen=True)
class Bursts:
    """Bursts of one record: rate per minute over the whole record, durations in ms, onsets in s."""
    rate_per_min: float
    n_bursts: int
    durations_ms: np.ndarray
    onsets_s: np.ndarray
    threshold: float
    threshold_source: str
    n_edge_excluded: int
    n_short_excluded: int
    record_s: float
    fs: float
    draft: tuple = ()


def detect_bursts(x, fs: float, band, *, threshold: float | None = None, threshold_percentile=BURST_PERCENTILE,
                  min_duration_s=BURST_MIN_S, edge_policy=BURST_EDGE, edge_guard_cycles=BURST_EDGE_GUARD,
                  method=BURST_METHOD, order=BURST_ORDER, smooth_s=BURST_SMOOTH_S) -> Bursts:
    """Bursts of NR-3.6 on the native-rate envelope (never on the 0.25 s grid).

    A burst is a run of samples whose envelope is at or above the threshold, kept when it lasts at least
    min_duration_s (a run of exactly min_duration_s is kept). The threshold is ``threshold`` when given (for
    instance one set per hemisphere across records), else the threshold_percentile of this record's envelope.
    edge_policy 'exclude' drops, as cut by the record start or end, runs that start within g samples of the first
    sample or end within g samples of the last, g = ceil(edge_guard_cycles x fs / band lower edge) (g = 0: runs
    touching the first or last sample); 'include' keeps them. The rate divides the kept bursts by the whole record
    length in minutes.
    """
    used: list = []
    env = band_envelope(x, fs, band, method=method, order=order, smooth_s=smooth_s, _used=used)
    if threshold is None:
        pct = float(resolve(threshold_percentile, "threshold_percentile", used))
        thr = float(np.percentile(env, pct))
        source = f"percentile {pct:g} of the record envelope"
    else:
        thr = float(threshold)
        source = "given"
    mind = float(resolve(min_duration_s, "min_duration_s", used))
    edge = resolve(edge_policy, "edge_policy", used)
    if edge not in ("exclude", "include"):
        raise ValueError("edge_policy must be 'exclude' or 'include'")
    guard = 0
    if edge == "exclude":
        gc = float(resolve(edge_guard_cycles, "edge_guard_cycles", used))
        if not gc >= 0:
            raise ValueError("edge_guard_cycles must be >= 0")
        guard = int(math.ceil(gc * fs / _band(band)[0] - 1e-9)) if _band(band)[0] > 0 else 0
    above = (env >= thr).astype(np.int8)
    d = np.diff(np.concatenate([[0], above, [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    n = env.size
    keep_on, keep_dur = [], []
    n_edge = n_short = 0
    for s, e in zip(starts, ends):
        if edge == "exclude" and (s <= guard or e >= n - guard):
            n_edge += 1
            continue
        dur = (e - s) / fs
        if dur < mind:
            n_short += 1
            continue
        keep_on.append(s / fs)
        keep_dur.append(1000.0 * dur)
    record_s = n / fs
    return Bursts(len(keep_on) / (record_s / 60.0), len(keep_on), np.asarray(keep_dur), np.asarray(keep_on), thr,
                  source, n_edge, n_short, record_s, float(fs), tuple(used))


def erd_percent(p_event: float, p_baseline: float) -> float:
    """ERD = 100 x (P_event - P_baseline) / P_baseline (NR-3.7); negative is desynchronization, positive
    synchronization."""
    if not p_baseline > 0:
        raise ValueError("baseline power must be positive")
    return 100.0 * (float(p_event) - float(p_baseline)) / float(p_baseline)


@dataclass(frozen=True)
class ERD:
    erd_percent: float
    p_event: float
    p_baseline: float
    n_trials: int
    n_dropped: int
    event_window_s: tuple
    baseline_window_s: tuple
    draft: tuple = ()


def erd(x, fs: float, band, onsets_s: Sequence[float], *, event_window_s, baseline_window_s=ERD_BASELINE,
        order=ERD_ORDER) -> ERD:
    """Event-related (de)synchronization in percent per band (NR-3.7), the intertrial power method of Pfurtscheller
    and Lopes da Silva 1999: zero-phase band-pass, square, average over trials and over each window.

    Windows are (start, end) in s relative to each onset, end exclusive. ``event_window_s`` is fixed by the question
    and has no default; the baseline window is OPEN-20. Trials whose windows leave the record are dropped and
    counted.
    """
    used: list = []
    base = resolve(baseline_window_s, "baseline_window_s", used)
    od = int(resolve(order, "order", used))
    ev = (float(event_window_s[0]), float(event_window_s[1]))
    base = (float(base[0]), float(base[1]))
    if ev[1] <= ev[0] or base[1] <= base[0]:
        raise ValueError("windows must satisfy start < end")
    pw = _bandpass(x, fs, band, od) ** 2
    n = pw.size
    pe, pb, dropped = [], [], 0
    for t0 in onsets_s:
        idx = [(int(round((t0 + a) * fs)), int(round((t0 + b) * fs))) for a, b in (ev, base)]
        if any(i0 < 0 or i1 > n or i1 <= i0 for i0, i1 in idx):
            dropped += 1
            continue
        pe.append(pw[idx[0][0]:idx[0][1]].mean())
        pb.append(pw[idx[1][0]:idx[1][1]].mean())
    if not pe:
        raise ValueError("no trial inside the record")
    p_e, p_b = float(np.mean(pe)), float(np.mean(pb))
    return ERD(erd_percent(p_e, p_b), p_e, p_b, len(pe), dropped, ev, base, tuple(used))


# ---------------------------------------------------------------------------------------------------------------
# Coherence and the phase-slope lag (NR-3.8, NR-5.2)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSpectra:
    freqs: np.ndarray
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    coherence: np.ndarray
    n_seg: int
    seg_s: float
    freq_resolution_hz: float


def cross_spectra(x, y, fs: float, *, seg_s=PSL_SEG_S, taper=PSL_TAPER, _used: list | None = None) -> CrossSpectra:
    """Segment-averaged auto and cross spectra over disjoint segments (no overlap: the n_seg of eq. 6.7 counts
    disjoint segments), mean removed and tapered per segment. S_xy = mean of X conj(Y), so a body series y delayed
    by d behind x has phase +2 pi f d. Coherence is the magnitude-squared coherence |S_xy|^2 / (S_xx S_yy)."""
    used = _used if _used is not None else []
    seg = float(resolve(seg_s, "seg_s", used))
    tp = resolve(taper, "taper", used)
    a = np.asarray(x, float).ravel()
    b = np.asarray(y, float).ravel()
    if a.size != b.size:
        raise ValueError("the two series must share one clock and one length")
    nper = int(round(seg * fs))
    n_seg = a.size // nper
    if nper < 4 or n_seg < 2:
        raise ValueError("fewer than 2 disjoint segments")
    xa = a[: n_seg * nper].reshape(n_seg, nper)
    ya = b[: n_seg * nper].reshape(n_seg, nper)
    xa = xa - xa.mean(axis=1, keepdims=True)
    ya = ya - ya.mean(axis=1, keepdims=True)
    win = signal.get_window(tp, nper, fftbins=True) if tp != "boxcar" else np.ones(nper)
    fx = np.fft.rfft(xa * win, axis=1)
    fy = np.fft.rfft(ya * win, axis=1)
    sxx = np.mean(np.abs(fx) ** 2, axis=0)
    syy = np.mean(np.abs(fy) ** 2, axis=0)
    sxy = np.mean(fx * np.conj(fy), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(sxy) ** 2 / (sxx * syy)
    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    return CrossSpectra(freqs, sxx, syy, sxy, coh, int(n_seg), seg, float(fs) / nper)


@dataclass(frozen=True)
class PhaseSlopeLag:
    """Group delay from the phase slope: lag in ms (positive: the neural series leads) and its SE from the weighted
    fit with known weights 1 / var(phase), var(phase) = (1 / (2 n_seg)) (1 / |R|^2 - 1) (Halliday et al. 1995,
    eq. 6.7). chi2_dof is the weighted residual sum over its degrees of freedom (near 1 when the weights are right).
    ``estimable`` is False, and lag and SE are NaN, when fewer than min_bins bins of the band have a coherence above
    the floor; ``coherence_floor`` is that floor. On bins spaced freq_resolution_hz apart a lag is identified only
    modulo alias_period_ms (the segment length): the lag returned lies in [-alias_period_ms / 2, alias_period_ms / 2).
    ``phase_rad`` is the phase of every fitted bin on the branch the fit used; ``wrap_turns`` the whole turns added to
    the principal phase of the lowest fitted bin."""
    lag_ms: float
    se_ms: float
    estimable: bool
    n_bins: int
    n_band_bins: int
    coherence_floor: float
    fit: str
    intercept_rad: float | None
    wrap_turns: int
    chi2_dof: float
    n_seg: int
    seg_s: float
    freq_resolution_hz: float
    alias_period_ms: float
    band: tuple
    freqs: np.ndarray
    phase_rad: np.ndarray
    coherence: np.ndarray
    weights: np.ndarray
    draft: tuple = ()


def coherence_null_limit(n_seg: int, alpha: float) -> float:
    """Upper limit of the magnitude-squared coherence of two independent series at level alpha, for n_seg disjoint
    segments: 1 - alpha^(1 / (n_seg - 1)) (the confidence limit used by Rosenberg et al. 1989 and Halliday et al.
    1995 for coherence)."""
    if n_seg < 2 or not 0 < alpha < 1:
        raise ValueError("n_seg >= 2 and 0 < alpha < 1 required")
    return 1.0 - alpha ** (1.0 / (n_seg - 1))


def _wrap_phase(a):
    """Phase wrapped into [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


def _weighted_phase_fit(f, phi, w, ft):
    """Weighted least squares of phase on frequency with known weights: (slope, intercept or None, SE of the slope,
    weighted residual sum, degrees of freedom)."""
    if ft == "origin":
        sff = float(np.sum(w * f * f))
        slope = float(np.sum(w * f * phi)) / sff
        rss = float(np.sum(w * (phi - slope * f) ** 2))
        return slope, None, 1.0 / math.sqrt(sff), rss, f.size - 1
    sw = float(np.sum(w))
    fbar = float(np.sum(w * f)) / sw
    pbar = float(np.sum(w * phi)) / sw
    sxx = float(np.sum(w * (f - fbar) ** 2))
    slope = float(np.sum(w * (f - fbar) * (phi - pbar))) / sxx
    intercept = pbar - slope * fbar
    rss = float(np.sum(w * (phi - intercept - slope * f) ** 2))
    return slope, intercept, 1.0 / math.sqrt(sxx), rss, f.size - 2


PSL_GRID_PER_TURN = 16  # lag grid: at the highest fitted frequency, one grid step turns the phase by 1 / 16 turn


def _phase_branches(f, phi, w, ft, df):
    """Phase of every bin on the branch nearest a line found by a global search, for bins on the grid k * df that
    need not be contiguous (np.unwrap assumes adjacent bins and fails across a gap of the coherence floor).

    The search runs over one alias period of the lag, [-1 / (2 df), 1 / (2 df)), on which every bin's phase
    2 pi f tau repeats: the lag maximizing sum w cos(phi - 2 pi f tau) (origin fit) or |sum w exp(i (phi - 2 pi f
    tau))| (intercept fit, the intercept being the angle of that sum), on a grid evaluated by one zero-padded FFT
    over the bin indices. Each bin then goes on the branch within pi of that line. Returns the phases on their
    branches and the whole turns added at the lowest bin."""
    k = np.rint(f / df).astype(np.int64)
    if np.any(np.abs(k * df - f) > 1e-6 * df):
        raise ValueError("fitted bins are not on the frequency grid of the segments")
    n_fft = 1 << int(math.ceil(math.log2(max(16, PSL_GRID_PER_TURN * (int(k.max()) + 1)))))
    a = np.zeros(n_fft, dtype=complex)
    np.add.at(a, k, w * np.exp(1j * phi))
    s = np.fft.fft(a)  # s[q] = sum w exp(i (phi - 2 pi f tau_q)), tau_q = q / (n_fft df)
    score = s.real if ft == "origin" else np.abs(s)
    q = int(np.argmax(score))
    period = 1.0 / df
    tau = q * period / n_fft
    if tau >= period / 2.0:
        tau -= period
    c = 0.0 if ft == "origin" else float(np.angle(s[q]))
    two_pi = 2.0 * math.pi
    line = two_pi * f * tau + c
    turns = np.rint((line + _wrap_phase(phi - line) - phi) / two_pi).astype(np.int64)
    return phi + two_pi * turns, int(turns[0])


def phase_slope_lag(neural, body, fs: float, band, *, seg_s=PSL_SEG_S, taper=PSL_TAPER, fit=PSL_FIT,
                    coherence_alpha=PSL_COH_ALPHA, min_bins=PSL_MIN_BINS, edges=BAND_EDGES) -> PhaseSlopeLag:
    """Phase-slope lag of NR-3.8 between a neural and a body series on one native-rate clock.

    Bins enter the fit when they lie in the band and their coherence exceeds coherence_null_limit(n_seg,
    coherence_alpha) (coherence_alpha None: every band bin); the bins kept need not be contiguous. The phase of S_xy
    over those bins is put on its branches by a global lag search over one alias period (_phase_branches), not by
    unwrapping along frequency, and is regressed on frequency by weighted least squares
    with weights 1 / var(phase(f)). fit 'origin': through the origin (Rosenberg et al. 1989); fit 'intercept': with
    an intercept. lag = slope / (2 pi); SE = SE(slope) / (2 pi) with SE(slope) = 1 / sqrt(sum w f^2) (origin) or
    1 / sqrt(sum w (f - f_w)^2) (intercept). On bins spaced df apart a lag is identified modulo 1 / df (the segment
    length); the lag returned lies in [-1 / (2 df), 1 / (2 df)).

    Eq. 6.7 is a large-sample approximation that holds where the coherence is clearly above zero; at the null its
    phase variance is far below the uniform phase's pi^2 / 3, so the SE is only meaningful once coupling is
    established (by the null of NR-5.6), and the floor keeps incoherent bins out of the fit.
    """
    used: list = []
    cs = cross_spectra(neural, body, fs, seg_s=seg_s, taper=taper, _used=used)
    ft = resolve(fit, "fit", used)
    alpha = resolve(coherence_alpha, "coherence_alpha", used)
    nmin = int(resolve(min_bins, "min_bins", used))
    e = resolve(edges, "edges", used)
    if ft not in ("origin", "intercept"):
        raise ValueError("fit must be 'origin' or 'intercept'")
    floor = 0.0 if alpha is None else coherence_null_limit(cs.n_seg, float(alpha))
    in_band = band_bins(cs.freqs, band, edges=e) & (cs.freqs > 0) & np.isfinite(cs.coherence)
    m = in_band & (cs.coherence > floor)
    f = cs.freqs[m]
    coh = np.clip(cs.coherence[m], 1e-12, 1.0 - 1e-12)
    var = (1.0 / (2.0 * cs.n_seg)) * (1.0 / coh - 1.0)
    w = 1.0 / var
    phi = np.angle(cs.sxy[m])
    common = dict(n_bins=int(m.sum()), n_band_bins=int(in_band.sum()), coherence_floor=float(floor), fit=ft,
                  n_seg=cs.n_seg, seg_s=cs.seg_s, freq_resolution_hz=cs.freq_resolution_hz,
                  alias_period_ms=1000.0 / cs.freq_resolution_hz, band=_band(band), freqs=f, coherence=coh,
                  weights=w, draft=tuple(used))
    if m.sum() < max(nmin, 3 if ft == "intercept" else 2):
        return PhaseSlopeLag(lag_ms=float("nan"), se_ms=float("nan"), estimable=False, intercept_rad=None,
                             wrap_turns=0, chi2_dof=float("nan"), phase_rad=phi, **common)
    phi, turns = _phase_branches(f, phi, w, ft, cs.freq_resolution_hz)
    slope, intercept, se, rss, dof = _weighted_phase_fit(f, phi, w, ft)
    two_pi = 2.0 * math.pi
    return PhaseSlopeLag(lag_ms=1000.0 * slope / two_pi, se_ms=1000.0 * se / two_pi, estimable=True,
                         intercept_rad=intercept, wrap_turns=turns, chi2_dof=rss / max(dof, 1), phase_rad=phi,
                         **common)


def lag_direction(lag_ms: float, se_ms: float, *, level=LEVEL) -> str:
    """NR-5.2: a direction is claimed only when lag +/- z_(1 - a/2) SE excludes 0: 'neural leads', 'body leads' or
    'none'."""
    lev = float(resolve(level, "level"))
    z = float(sstats.norm.ppf(1.0 - (1.0 - lev) / 2.0))
    if lag_ms - z * se_ms > 0:
        return "neural leads"
    if lag_ms + z * se_ms < 0:
        return "body leads"
    return "none"


# ---------------------------------------------------------------------------------------------------------------
# Refusals (NR-3.14, NR-3.16)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    refused: bool
    reason: str | None
    statement: str
    draft: tuple = ()


def bandwidth_statement(fs: float, band, lowpass_hz: float | None) -> str:
    """The bandwidth statement of NR-3.5: sampling rate, Nyquist, declared low-pass corner, the band."""
    lo, hi = _band(band)
    lp = f"declared low-pass corner {lowpass_hz:g} Hz" if lowpass_hz is not None else "low-pass corner undeclared"
    return f"fs {fs:g} Hz, Nyquist {fs / 2:g} Hz, {lp}; band {lo:g} to {hi:g} Hz"


def bandwidth_refusal(band, fs: float, *, lowpass_hz: float | None = None, undeclared=UNDECLARED_LOWPASS) -> Refusal:
    """NR-3.14: a band descriptor whose upper edge f_hi satisfies f_hi >= fs / 2, or lies above the declared
    low-pass corner, is refused. A record without a declared corner follows ``undeclared`` (OPEN-23): 'refuse' or
    'nyquist_only'."""
    used: list = []
    lo, hi = _band(band)
    st = bandwidth_statement(fs, band, lowpass_hz)
    if hi >= fs / 2:
        return Refusal(True, f"upper edge {hi:g} Hz >= fs / 2 = {fs / 2:g} Hz (NR-3.14)", st, ())
    if lowpass_hz is not None:
        if hi > lowpass_hz:
            return Refusal(True, f"upper edge {hi:g} Hz above the declared low-pass corner {lowpass_hz:g} Hz "
                                 "(NR-3.14)", st, ())
        return Refusal(False, None, st, ())
    pol = resolve(undeclared, "undeclared", used)
    if pol == "refuse":
        return Refusal(True, "low-pass corner undeclared (OPEN-23 draft: refuse)", st, tuple(used))
    if pol == "nyquist_only":
        return Refusal(False, None, st, tuple(used))
    raise ValueError("undeclared must be 'refuse' or 'nyquist_only'")


@dataclass(frozen=True)
class StimExclusion:
    """The parts of a band kept for an estimate under stimulation at stim_freq_hz (0: stimulation off)."""
    band: tuple
    stim_freq_hz: float
    excluded: tuple
    kept: tuple
    refused: bool
    reason: str | None
    draft: tuple = ()


def stimulation_exclusion(band, stim_freq_hz: float | None, *, half_width_hz=STIM_HALF_WIDTH,
                          multiples=STIM_MULTIPLES, mode=STIM_MODE) -> StimExclusion:
    """NR-3.16: every gamma estimate carries the stimulation frequency of its block, and a band around half of it is
    excluded. stim_freq_hz must be given (0 when the block has no stimulation); None raises. Every interval
    [k f_stim - hw, k f_stim + hw] for k in ``multiples`` is removed from the band; when nothing of positive width
    remains the estimate is refused. mode 'model' (an explicit entrainment model) is not implemented."""
    used: list = []
    if stim_freq_hz is None:
        raise ValueError("the stimulation frequency of the block is required (0 when off), NR-3.16")
    md = resolve(mode, "mode", used)
    if md == "model":
        raise NotImplementedError("the entrainment model of NR-3.16 is not implemented (OPEN-24)")
    if md != "exclude":
        raise ValueError("mode must be 'exclude' or 'model'")
    lo, hi = _band(band)
    fst = float(stim_freq_hz)
    if fst < 0:
        raise ValueError("stimulation frequency must be >= 0")
    if fst == 0:
        return StimExclusion((lo, hi), 0.0, (), ((lo, hi),), False, None, ())
    hw = float(resolve(half_width_hz, "half_width_hz", used))
    mult = tuple(resolve(multiples, "multiples", used))
    cuts = sorted((k * fst - hw, k * fst + hw) for k in mult)
    excluded = tuple((max(lo, a), min(hi, b)) for a, b in cuts if a < hi and b > lo)
    kept, cur = [], lo
    for a, b in excluded:
        if a > cur:
            kept.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        kept.append((cur, hi))
    refused = not kept
    reason = "the excluded stimulation bands cover the whole band (NR-3.16)" if refused else None
    return StimExclusion((lo, hi), fst, excluded, tuple(kept), refused, reason, tuple(used))


def frequency_mask(freqs, kept) -> np.ndarray:
    """Bins of ``freqs`` inside any kept interval (closed intervals, so a bin on a cut edge is kept)."""
    f = np.asarray(freqs, float)
    m = np.zeros(f.shape, dtype=bool)
    for a, b in kept:
        m |= (f >= a) & (f <= b)
    return m


# ---------------------------------------------------------------------------------------------------------------
# Output schema (NR-3.5)
# ---------------------------------------------------------------------------------------------------------------

STATUSES = ("measured", "prior-only", "refused")
SCHEMA_FIELDS = ("name", "unit", "status", "value", "interval_lo", "interval_hi", "interval_level", "interval_method",
                 "band_lo_hz", "band_hi_hz", "bandwidth_statement", "window_s", "freq_resolution_hz", "fs_hz",
                 "stim_freq_hz", "refusal_reason", "draft")


@dataclass(frozen=True)
class DescriptorValue:
    """One descriptor value with the fields NR-3.5 requires: an interval, a status (measured, prior-only or refused),
    a bandwidth statement, and its time and frequency resolution (window length in s, frequency resolution in Hz).

    A refused value carries no value and no interval but a reason. A value whose band overlaps gamma (60 to 90 Hz)
    carries the stimulation frequency of its block (NR-3.16; 0 when off). ``to_dict`` validates first: a value
    without these fields is not written.
    """
    name: str
    unit: str
    status: str
    value: float | None
    interval: tuple | None
    interval_level: float | None
    interval_method: str | None
    band_hz: tuple | None
    bandwidth_statement: str
    window_s: float
    freq_resolution_hz: float
    fs_hz: float
    stim_freq_hz: float | None = None
    refusal_reason: str | None = None
    draft: tuple = ()

    def validate(self) -> None:
        def bad(msg):
            raise ValueError(f"descriptor {self.name!r}: {msg} (NR-3.5)")
        if not self.name or not self.unit:
            bad("name and unit are required")
        if self.status not in STATUSES:
            bad(f"status must be one of {STATUSES}")
        if not self.bandwidth_statement:
            bad("bandwidth statement missing")
        for fld in ("window_s", "freq_resolution_hz", "fs_hz"):
            v = getattr(self, fld)
            if v is None or not math.isfinite(v) or v <= 0:
                bad(f"{fld} must be a positive number")
        if self.band_hz is not None:
            _band(self.band_hz)
            if _overlaps(self.band_hz, BANDS["gamma"]) and self.stim_freq_hz is None:
                bad("a gamma estimate carries the stimulation frequency of its block (NR-3.16)")
        if self.status == "refused":
            if self.value is not None or self.interval is not None:
                bad("a refused descriptor carries no value and no interval")
            if not self.refusal_reason:
                bad("a refused descriptor carries its reason")
            return
        if self.refusal_reason:
            bad("only a refused descriptor carries a refusal reason")
        if self.value is None or not math.isfinite(self.value):
            bad("value must be finite")
        if self.interval is None or len(self.interval) != 2:
            bad("interval missing")
        lo, hi = self.interval
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
            bad("interval must be finite with lo <= hi")
        if self.interval_level is None or not 0 < self.interval_level < 1:
            bad("interval level must lie in (0, 1)")
        if not self.interval_method:
            bad("interval method missing")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "name": self.name, "unit": self.unit, "status": self.status, "value": self.value,
            "interval_lo": None if self.interval is None else float(self.interval[0]),
            "interval_hi": None if self.interval is None else float(self.interval[1]),
            "interval_level": self.interval_level, "interval_method": self.interval_method,
            "band_lo_hz": None if self.band_hz is None else float(self.band_hz[0]),
            "band_hi_hz": None if self.band_hz is None else float(self.band_hz[1]),
            "bandwidth_statement": self.bandwidth_statement, "window_s": float(self.window_s),
            "freq_resolution_hz": float(self.freq_resolution_hz), "fs_hz": float(self.fs_hz),
            "stim_freq_hz": self.stim_freq_hz, "refusal_reason": self.refusal_reason,
            "draft": "; ".join(self.draft),
        }


def band_descriptor(name: str, unit: str, band, fs: float, *, window_s: float, freq_resolution_hz: float,
                    value: float | None = None, interval: tuple | None = None, interval_level: float | None = None,
                    interval_method: str | None = None, status: str = "measured", lowpass_hz: float | None = None,
                    stim_freq_hz: float | None = None, undeclared=UNDECLARED_LOWPASS,
                    draft: Sequence[str] = ()) -> DescriptorValue:
    """A band descriptor that passes through the refusal of NR-3.14 first: when the band is refused, the value is
    emitted with status refused and no value, whatever value was given. Validated on return."""
    ref = bandwidth_refusal(band, fs, lowpass_hz=lowpass_hz, undeclared=undeclared)
    tags = tuple(draft) + tuple(t for t in ref.draft if t not in draft)
    if ref.refused:
        out = DescriptorValue(name, unit, "refused", None, None, None, None, _band(band), ref.statement, window_s,
                              freq_resolution_hz, fs, stim_freq_hz, ref.reason, tags)
    else:
        out = DescriptorValue(name, unit, status, value, interval, interval_level, interval_method, _band(band),
                              ref.statement, window_s, freq_resolution_hz, fs, stim_freq_hz, None, tags)
    out.validate()
    return out


def descriptor_rows(values: Sequence[DescriptorValue]) -> list:
    """Rows to write, one per descriptor, each validated (NR-3.5)."""
    return [v.to_dict() for v in values]


__all__ = [
    "BANDS", "HOP_S", "MIN_WINDOW_S", "band_bins", "PSD", "welch_psd", "aperiodic_model", "gaussian_sum", "Peak",
    "SpectralFit", "spectral_fit", "BandPower", "band_periodic_power", "band_envelope", "Bursts", "detect_bursts",
    "erd_percent", "ERD", "erd", "CrossSpectra", "cross_spectra", "PhaseSlopeLag", "coherence_null_limit",
    "phase_slope_lag",
    "lag_direction", "Refusal", "bandwidth_statement", "bandwidth_refusal", "StimExclusion", "stimulation_exclusion",
    "frequency_mask", "STATUSES", "SCHEMA_FIELDS", "DescriptorValue", "band_descriptor", "descriptor_rows",
]
