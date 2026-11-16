"""Fixed image-plane formulas and statistics of study S1 (docs/rules/STUDY_S1_2026-11-12.md), read by S1 and S2.

Registered rules: docs/rules/STUDY_FAMILY_2026-11-12.md, docs/rules/STUDY_S1_2026-11-12.md and
docs/rules/STUDY_S2_2026-11-12.md, with the drafts they adopt (docs/proposals/STUDY_S1_MODYS_MEASUREMENT_DRAFT.md,
docs/proposals/STUDY_S2_Q9_FACETS_DRAFT.md) and the DR rules those cite. Every value below is the registered value;
the id that fixes it is named beside it. The module reads no file and writes none: callers pass arrays.

Readings of this module, each the most literal one the rules allow (logged in reports/studies/s1/decisions.md):
- Points: the 14 DeepLabCut points of both zips; a point is valid when its coordinates are finite and its
  likelihood is at least 0.5 (S1 draft 3.1; OPEN-D56). Hip midpoint H and shoulder midpoint S need both points.
- Torso length L(t) = |S(t) - H(t)| in the same frame (S1 draft 3.1); a frame with L = 0 is not valid.
- Regions (OPEN-D53): trunk (H to S), proximal arm (shoulder to elbow of a side), proximal leg (hip to knee of a
  side). The keypoints a region reads: hip1, hip2, shoulder1, shoulder2, plus the elbow or knee of the side.
- Valid segment: the maximal runs of frames where every keypoint the region reads is valid; every filter and every
  derivative runs over each such run of the whole record before any window is cut (DR-2.4, OPEN-D07).
- Filters (DR-2.4, OPEN-D07): Butterworth order 2 per pass, forward and backward (scipy.signal.filtfilt, odd
  extension, its default pad length), the single-pass corner set so that the combined -3 dB point is at the named
  frequency, c = (sqrt(2) - 1)^(1 / (2N)); band-pass = high-pass then low-pass. A run not longer than the pad length
  gives no filtered sample.
- Derivative: numpy.gradient over each valid run (second-order central differences, first-order at the run ends);
  no DR rule fixes a derivative on raw keypoints. A run of one frame gives none.
- Spectra (DR-2.5, OPEN-D06): Welch, Hann, 4 s segments, 50 percent overlap, mean removal, one-sided density, trace
  over vector components, over the valid runs of the window that hold a segment, averaged with weights equal to
  their segment counts; band power = trapezoid over the bins with 0.5 <= f <= 3.0 Hz.
- Angles (S1 section 3, OPEN-D38): the signed image-plane angle of the segment vector against the parent axis (the
  trunk axis, shoulder midpoint to hip midpoint); its unit vector (cos, sin) is low-passed at 0.3 Hz and the angle
  read back (the 2D counterpart of DR-14.2); TON-T deviation theta(t) = |angle| against the reference 0 deg. The
  trunk angle is against the image vertical (down), reported raw, never as a deviation.
- TON-DW (DR-14.3, S1 section 3, OPEN-D39): percent of the window's valid theta samples beyond 18.0 deg (arm) or
  4.5 deg (leg), counted in maximal runs, a run shorter than d_min = 1.8 / fs not counted; the episode rate per
  minute of valid samples and the mean episode duration beside it. No dwell threshold exists for the trunk.
- PHA-P1 (DR-14.4, S1 draft 3.2): 10 log10 of the 0.5 to 3 Hz band power of the angular velocity (derivative of
  the unwrapped unfiltered angle) re 1 (deg/s)^2.
- Flow (DR-13, OPEN-D36, OPEN-D37): r(t) = (distal - proximal) / L(t) in torso lengths; v = dr/dt. FLOW-DII on v
  band-passed 0.5 to 3 Hz, 2 x (1 - lambda_1); FLOW-E = 10 log10 of the 0.5 to 3 Hz band power of v re 1 (torso
  length per second)^2; FLOW-R on y(t) = (distal - H) / L(t), low-passed at 3 Hz, sqrt(mean |y - mean y|^2);
  FLOW-OCC = percent of the window's valid samples whose |r band-passed 0.5 to 3 Hz| exceeds 10 percent of the
  segment's image-plane length, the record median of |distal - proximal| / L (torso lengths).
- Windows (DR-2.7, OPEN-D05, OPEN-D59): 10 s every 5 s from the record's first frame; a window enters a descriptor
  when every keypoint the descriptor reads is valid in at least half of its frames; the record value is the median
  of the window values. Within-clip halves: the first and the last floor(n / 2) frames, each read as one window.
- Side pooling (OPEN-D53): the region value is the mean of the side 1 and side 2 values, both needed; every pooled
  value carries laterality_unknown.
- Bandwidth (DR-5.1, OPEN-D02): k = 0.30, f_max = k x fs; a band whose upper edge exceeds f_max is refused.
"""
import math

import numpy as np

# ---------------------------------------------------------------- registered values
LIKELIHOOD_MIN = 0.5          # S1 draft 3.1, OPEN-D56 (S1 section 4, item 16)
WINDOW_S = 10.0               # DR-2.7, DR-14.1
HOP_S = 5.0                   # OPEN-D05 (S1 section 4, item 2)
FILTER_ORDER = 2              # OPEN-D07 (item 4)
PADTYPE = "odd"               # OPEN-D07 (item 4)
WELCH_SEG_S = 4.0             # OPEN-D06 (item 3)
K_BANDWIDTH = 0.30            # OPEN-D02 (item 1)
TONIC_LOWPASS_HZ = 0.3        # DR-14.2, S1 section 3
BAND_HZ = (0.5, 3.0)          # DR-13.1, DR-14.4
ROAM_LOWPASS_HZ = 3.0         # OPEN-D37 (item 8), DR-13.4
DWELL_THRESHOLD_DEG = {"proximal_arm": 18.0, "proximal_leg": 4.5}   # S1 section 3 (jarm2022, 10 percent of range)
OCC_FRACTION = 0.10           # S1 section 3, OPEN-D37
MIN_VALID_FRACTION = 0.5      # OPEN-D59 (item 6)
D_MIN_SAMPLES = 1.8           # DR-5.4, OPEN-D39: d_min = 1.8 / fs
REFERENCE_DEG = 0.0           # S1 section 3: anatomical position, arm hanging along the trunk, thigh along it
BOOT_B = 2000                 # OPEN-D69 (item 16), OPEN-04 (S2 item 7)
INTERVAL = (2.5, 97.5)        # OPEN-01 (family file section 2): 95 percent two-sided
READABLE_LINE = 0.5           # S1 draft section 7 (Koo and Li 2016, checked in S1 section 5)

POINTS = ("ankle1", "knee1", "hip1", "hip2", "knee2", "ankle2", "wrist1", "elbow1", "shoulder1", "shoulder2",
          "elbow2", "wrist2", "chin", "forehead")
TORSO_POINTS = ("hip1", "hip2", "shoulder1", "shoulder2")
SCORED_POINTS = ("hip1", "hip2", "shoulder1", "shoulder2", "elbow1", "elbow2", "knee1", "knee2")
REGIONS = ("trunk", "proximal_arm", "proximal_leg")
SIDES = {"trunk": (None,), "proximal_arm": ("1", "2"), "proximal_leg": ("1", "2")}

# descriptors per region; the trunk carries a raw angle and no deviation or dwell (S1 section 3)
TONIC = ("TON-T", "TON-DW", "TON-DW-RATE", "TON-DW-MEAN")
FLOW = ("FLOW-DII", "FLOW-E", "FLOW-R", "FLOW-OCC")
DESCRIPTORS = {
    "trunk": ("TRUNK-ANGLE-RAW", "PHA-P1") + FLOW,
    "proximal_arm": TONIC + ("PHA-P1",) + FLOW,
    "proximal_leg": TONIC + ("PHA-P1",) + FLOW,
}
UNITS = {"TON-T": "deg", "TRUNK-ANGLE-RAW": "deg", "TON-DW": "percent", "TON-DW-RATE": "per min",
         "TON-DW-MEAN": "s", "PHA-P1": "dB re 1 (deg/s)^2", "FLOW-DII": "1", "FLOW-E": "dB re 1 (TL/s)^2",
         "FLOW-R": "TL", "FLOW-OCC": "percent"}


def region_points(region, side):
    """(proximal, distal, keypoints read) of a region and side."""
    if region == "trunk":
        return "H", "S", TORSO_POINTS
    if region == "proximal_arm":
        return f"shoulder{side}", f"elbow{side}", TORSO_POINTS + (f"elbow{side}",)
    if region == "proximal_leg":
        return f"hip{side}", f"knee{side}", TORSO_POINTS + (f"knee{side}",)
    raise ValueError(region)


# ---------------------------------------------------------------- points
def valid_points(xy, like, names=POINTS, likelihood_min=LIKELIHOOD_MIN):
    """({name: (T, 2)} with NaN where not valid, {name: (T,) bool}) plus the midpoints H and S."""
    xy = np.asarray(xy, dtype=np.float64)
    like = np.asarray(like, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1] != len(names) or xy.shape[2] != 2 or like.shape != xy.shape[:2]:
        raise ValueError(f"xy {xy.shape} and likelihood {like.shape} for {len(names)} points")
    ok = np.isfinite(xy).all(axis=-1) & (np.nan_to_num(like, nan=-np.inf) >= likelihood_min)
    pts = {n: np.where(ok[:, i, None], xy[:, i], np.nan) for i, n in enumerate(names)}
    val = {n: ok[:, i].copy() for i, n in enumerate(names)}
    pts["H"] = (pts["hip1"] + pts["hip2"]) / 2.0
    pts["S"] = (pts["shoulder1"] + pts["shoulder2"]) / 2.0
    val["H"] = val["hip1"] & val["hip2"]
    val["S"] = val["shoulder1"] & val["shoulder2"]
    return pts, val


def runs(mask):
    """[(start, stop)] of the maximal runs of True."""
    m = np.concatenate([[0], np.asarray(mask, dtype=bool).astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def window_bounds(n, fs, window_s=WINDOW_S, hop_s=HOP_S):
    w, h = int(round(window_s * fs)), int(round(hop_s * fs))
    return [(s, s + w) for s in range(0, int(n) - w + 1, h)]


def half_bounds(n):
    h = int(n) // 2
    return [(0, h), (int(n) - h, int(n))] if h else []


# ---------------------------------------------------------------- filters, derivative, spectra
def corner_hz(f_hz, fs, kind, order=FILTER_ORDER):
    """Single-pass corner that puts the forward-backward -3 dB point at f_hz (DR-2.4)."""
    if not 0.0 < f_hz < fs / 2.0:
        raise ValueError(f"{f_hz} Hz outside (0, {fs / 2.0})")
    c = (math.sqrt(2.0) - 1.0) ** (1.0 / (2 * order))
    t = math.tan(math.pi * f_hz / fs)
    t = t / c if kind == "low" else t * c
    return fs / math.pi * math.atan(t)


def _ba(f_hz, fs, kind, order=FILTER_ORDER):
    from scipy.signal import butter
    return butter(order, corner_hz(f_hz, fs, kind, order), btype="lowpass" if kind == "low" else "highpass", fs=fs)


def zero_phase(x, mask, fs, f_hz, kind, order=FILTER_ORDER):
    """Filter x (T,) or (T, C) over each run of mask; NaN elsewhere and on runs not longer than the pad length."""
    from scipy.signal import filtfilt
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    b, a = _ba(f_hz, fs, kind, order)
    padlen = 3 * max(len(a), len(b))
    for s, e in runs(mask):
        if e - s > padlen:
            out[s:e] = filtfilt(b, a, x[s:e], axis=0, padtype=PADTYPE)
    return out


def bandpass(x, mask, fs, band=BAND_HZ):
    hp = zero_phase(x, mask, fs, band[0], "high")
    fin = np.isfinite(hp) if hp.ndim == 1 else np.isfinite(hp).all(axis=1)
    return zero_phase(hp, mask & fin, fs, band[1], "low")


def derivative(x, mask, fs):
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    for s, e in runs(mask):
        if e - s >= 2:
            out[s:e] = np.gradient(x[s:e], axis=0) * fs
    return out


def welch_band_power(x, fs, band=BAND_HZ, seg_s=WELCH_SEG_S):
    """Band power of x (T,) or (T, C) over the finite runs holding a segment; NaN when none holds one."""
    from scipy.signal import welch
    x = np.asarray(x, dtype=np.float64)
    x2 = x[:, None] if x.ndim == 1 else x
    fin = np.isfinite(x2).all(axis=1)
    nper = int(round(seg_s * fs))
    step = nper - nper // 2
    total, count, f = None, 0, None
    for s, e in runs(fin):
        if e - s < nper:
            continue
        f, p = welch(x2[s:e].T, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, detrend="constant",
                     scaling="density", return_onesided=True, axis=-1)
        m = (e - s - nper) // step + 1
        p = p.sum(axis=0) * m
        total = p if total is None else total + p
        count += m
    if not count:
        return math.nan
    psd = total / count
    sel = (f >= band[0] - 1e-9) & (f <= band[1] + 1e-9)
    return float(np.trapezoid(psd[sel], f[sel])) if sel.sum() >= 2 else math.nan


def db(p):
    return 10.0 * math.log10(p) if (p is not None and math.isfinite(p) and p > 0) else math.nan


def dii(v):
    """2 x (1 - lambda_1) of the squared-speed-weighted orientation tensor of 2D velocities (DR-13.2)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v).all(axis=1)]
    s = float((v ** 2).sum())
    if len(v) < 2 or not s > 1e-24:
        return math.nan
    lam1 = float(np.linalg.eigvalsh(v.T @ v / s)[-1])
    return float(min(1.0, max(0.0, 2.0 * (1.0 - lam1))))


def f_max_hz(fs, k=K_BANDWIDTH):
    return k * fs


def band_admitted(fs, band=BAND_HZ, k=K_BANDWIDTH):
    """DR-5.3: computed when the upper edge is at most f_max."""
    return band[1] <= f_max_hz(fs, k) + 1e-12


# ---------------------------------------------------------------- angles and dwell
def signed_angle_deg(axis, u):
    cross = axis[..., 0] * u[..., 1] - axis[..., 1] * u[..., 0]
    dot = (axis * u).sum(axis=-1)
    ok = (np.linalg.norm(axis, axis=-1) > 0) & (np.linalg.norm(u, axis=-1) > 0)
    return np.where(ok, np.degrees(np.arctan2(cross, dot)), np.nan)


def lowpass_angle_deg(angle_deg, mask, fs, f_hz=TONIC_LOWPASS_HZ):
    r = np.radians(np.asarray(angle_deg, dtype=np.float64))
    e = zero_phase(np.stack([np.cos(r), np.sin(r)], axis=1), mask, fs, f_hz, "low")
    n = np.linalg.norm(e, axis=1)
    return np.where(n > 1e-6, np.degrees(np.arctan2(e[:, 1], e[:, 0])), np.nan)


def unwrap_deg(angle_deg, mask):
    a = np.asarray(angle_deg, dtype=np.float64)
    out = np.full_like(a, np.nan)
    for s, e in runs(mask):
        out[s:e] = np.degrees(np.unwrap(np.radians(a[s:e])))
    return out


def dwell(theta, threshold, fs):
    """(fraction percent, episodes per minute, mean episode s) over the finite samples of theta (OPEN-D39)."""
    theta = np.asarray(theta, dtype=np.float64)
    fin = np.isfinite(theta)
    nv = int(fin.sum())
    if not nv:
        return math.nan, math.nan, math.nan
    on = fin & (np.where(fin, theta, -np.inf) > threshold)
    d_min = D_MIN_SAMPLES / fs
    kept = [(a, b) for a, b in runs(on) if (b - a) / fs >= d_min - 1e-12]
    n_on = sum(b - a for a, b in kept)
    frac = 100.0 * n_on / nv
    rate = len(kept) / (nv / fs / 60.0)
    mean_s = float(np.mean([(b - a) / fs for a, b in kept])) if kept else math.nan
    return frac, rate, mean_s


# ---------------------------------------------------------------- one record
def _admitted(val, read, a, b, min_frac=MIN_VALID_FRACTION):
    return all(val[p][a:b].mean() >= min_frac for p in read) if b > a else False


def side_signals(pts, val, region, side, fs):
    """Per-frame signals of one region and side, filtered over the valid runs of the whole record."""
    prox, dist, read = region_points(region, side)
    T = len(val["hip1"])
    L = np.linalg.norm(pts["S"] - pts["H"], axis=-1)
    mask = np.ones(T, dtype=bool)
    for p in read:
        mask &= val[p]
    mask &= np.isfinite(L) & (L > 0)
    Lm = np.where(mask, L, np.nan)
    P, D = pts[prox], pts[dist]
    seg = np.where(mask[:, None], D - P, np.nan)
    sig = {"mask": mask, "read": read}
    if region == "trunk":
        down = np.broadcast_to(np.array([0.0, 1.0]), seg.shape)
        raw = signed_angle_deg(down, -seg)                       # trunk axis S to H against the image vertical
    else:
        axis = np.where(mask[:, None], pts["H"] - pts["S"], np.nan)
        raw = signed_angle_deg(axis, seg)
    raw = np.where(mask, raw, np.nan)
    sig["angle_lp"] = lowpass_angle_deg(raw, mask, fs)
    sig["theta"] = np.abs(sig["angle_lp"] - REFERENCE_DEG) if region != "trunk" else np.abs(sig["angle_lp"])
    sig["omega"] = derivative(unwrap_deg(raw, mask), mask, fs)
    r = seg / Lm[:, None]
    v = derivative(r, mask, fs)
    sig["v"] = v
    fin_v = mask & np.isfinite(v).all(axis=1)
    sig["v_bp"] = bandpass(v, fin_v, fs)
    sig["r_bp"] = bandpass(r, mask, fs)
    y = np.where(mask[:, None], (D - pts["H"]) / Lm[:, None], np.nan)
    sig["y_lp"] = zero_phase(y, mask, fs, ROAM_LOWPASS_HZ, "low")
    seg_len = np.linalg.norm(r, axis=1)
    seg_len = seg_len[np.isfinite(seg_len)]
    sig["seg_len_tl"] = float(np.median(seg_len)) if seg_len.size else math.nan
    sig["torso_px"] = float(np.nanmedian(Lm)) if np.isfinite(Lm).any() else math.nan
    return sig


def window_values(sig, region, fs, a, b):
    """{descriptor: value} of one window [a, b) of a side's signals; NaN where the window does not enter."""
    names = DESCRIPTORS[region]
    out = {n: math.nan for n in names}
    if not _admitted(sig["val"], sig["read"], a, b):
        return out
    th = sig["theta"][a:b]
    if np.isfinite(th).any():
        if region == "trunk":
            out["TRUNK-ANGLE-RAW"] = float(np.nanmedian(th))
        else:
            out["TON-T"] = float(np.nanmedian(th))
            f, rt, mn = dwell(th, DWELL_THRESHOLD_DEG[region], fs)
            out["TON-DW"], out["TON-DW-RATE"], out["TON-DW-MEAN"] = f, rt, mn
    if band_admitted(fs):
        out["PHA-P1"] = db(welch_band_power(sig["omega"][a:b], fs))
        out["FLOW-E"] = db(welch_band_power(sig["v"][a:b], fs))
    out["FLOW-DII"] = dii(sig["v_bp"][a:b])
    y = sig["y_lp"][a:b]
    y = y[np.isfinite(y).all(axis=1)]
    if len(y):
        out["FLOW-R"] = float(np.sqrt(np.mean(np.sum((y - y.mean(axis=0)) ** 2, axis=1))))
    rb = sig["r_bp"][a:b]
    ex = np.linalg.norm(rb, axis=1)
    fin = np.isfinite(ex)
    thr = OCC_FRACTION * sig["seg_len_tl"]
    if fin.any() and math.isfinite(thr):
        out["FLOW-OCC"] = 100.0 * float((ex[fin] > thr).sum()) / float(fin.sum())
    return out


def record_descriptors(xy, like, fs, names=POINTS):
    """Per region: {'windows': n windows, 'record': {descriptor: value}, 'halves': [{...}, {...}], 'sides': ...,
    'torso_px', 'seg_len_tl', 'usable_fraction'}. Region values are side-pooled means (both sides needed)."""
    pts, val = valid_points(xy, like, names)
    T = len(val["hip1"])
    wins = window_bounds(T, fs)
    halves = half_bounds(T)
    allv = np.ones(T, dtype=bool)
    for p in SCORED_POINTS:
        allv &= val[p]
    out = {"n_frames": T, "n_windows": len(wins), "usable_fraction": float(allv.mean()) if T else math.nan,
           "window_usable_fractions": [float(allv[a:b].mean()) for a, b in wins], "regions": {}}
    for region in REGIONS:
        per_side = {}
        for side in SIDES[region]:
            sig = side_signals(pts, val, region, side, fs)
            sig["val"] = val
            wv = [window_values(sig, region, fs, a, b) for a, b in wins]
            hv = [window_values(sig, region, fs, a, b) for a, b in halves]
            rec = {}
            for n in DESCRIPTORS[region]:
                xs = [w[n] for w in wv if math.isfinite(w[n])]
                rec[n] = float(np.median(xs)) if xs else math.nan
            per_side[side] = {"record": rec, "halves": hv,
                              "n_windows_entered": {n: sum(math.isfinite(w[n]) for w in wv)
                                                    for n in DESCRIPTORS[region]},
                              "seg_len_tl": sig["seg_len_tl"], "torso_px": sig["torso_px"]}
        out["regions"][region] = {"sides": per_side, "record": pool_sides([s["record"] for s in per_side.values()]),
                                  "halves": [pool_sides([s["halves"][i] for s in per_side.values()])
                                             for i in range(len(halves))],
                                  "seg_len_tl": pool_sides([{"x": s["seg_len_tl"]} for s in per_side.values()])["x"]}
    return out


def pool_sides(dicts):
    """Mean over the sides of each key; NaN unless every side has a finite value (OPEN-D53)."""
    if len(dicts) == 1:
        return dict(dicts[0])
    keys = dicts[0].keys()
    return {k: (float(np.mean([d[k] for d in dicts])) if all(math.isfinite(d[k]) for d in dicts) else math.nan)
            for k in keys}


# ---------------------------------------------------------------- statistics
def rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return math.nan
    rx, ry = rank(x[ok]), rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def icc_1_1(r):
    """ICC(1,1), one-way random, single measure (Shrout and Fleiss 1979, case 1); r (n targets, k ratings)."""
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r).all(axis=1)]
    n, k = r.shape if r.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return math.nan
    gm = r.mean()
    msb = k * ((r.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((r - r.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    den = msb + (k - 1) * msw
    return float((msb - msw) / den) if den > 0 else math.nan


def icc_2_1(r):
    """ICC(2,1), two-way random, absolute agreement, single measure (Shrout and Fleiss 1979, case 2)."""
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r).all(axis=1)]
    n, k = r.shape if r.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return math.nan
    gm = r.mean()
    ssr = k * ((r.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((r.mean(axis=0) - gm) ** 2).sum()
    sse = ((r - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else math.nan


def qwk(a, b):
    """Quadratic weighted kappa of two numeric ratings of the same targets (Cohen 1968), weights (a - b)^2."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return math.nan
    obs = np.mean((a - b) ** 2)
    exp = np.mean((a[:, None] - b[None, :]) ** 2)
    return float(1.0 - obs / exp) if exp > 0 else math.nan


def sem(values, icc):
    """SEM = SD x sqrt(1 - ICC), SD of all values entering the ICC (OPEN-D58, DR-18.1)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if len(v) < 2 or not math.isfinite(icc):
        return math.nan
    return float(np.std(v, ddof=1) * math.sqrt(max(0.0, 1.0 - icc)))


def bound(s):
    """1.96 x sqrt(2) x SEM (DR-18.1 formula), a lower bound, not MDC."""
    return 1.96 * math.sqrt(2.0) * s if math.isfinite(s) else math.nan


def srm(change):
    c = np.asarray(change, dtype=np.float64)
    c = c[np.isfinite(c)]
    if len(c) < 2:
        return math.nan
    sd = np.std(c, ddof=1)
    m = float(np.mean(c))
    # a set of equal changes has no SD; rounding leaves one of order 1e-16, which is read as none
    return m / sd if sd > 1e-12 * max(1.0, abs(m)) else math.nan


def cluster_bootstrap(clusters, stat, B=BOOT_B, seed=0):
    """Percentile interval of stat(list of resampled cluster ids) over B child-clustered replicates (DR-19.1)."""
    rng = np.random.default_rng(seed)
    ids = list(clusters)
    n = len(ids)
    vals = np.empty(B)
    for i in range(B):
        draw = [ids[j] for j in rng.integers(0, n, n)]
        vals[i] = stat(draw)
    fin = vals[np.isfinite(vals)]
    if not fin.size:
        return math.nan, math.nan, B
    lo, hi = np.percentile(fin, INTERVAL)
    return float(lo), float(hi), int(B - fin.size)


def fisher_halfwidth(n, z=1.96):
    """Half-width in z units of the Fisher-z interval of a Spearman correlation, 1.03 / sqrt(n - 3) (Fieller,
    Hartley and Pearson 1957)."""
    return z * 1.03 / math.sqrt(n - 3)
