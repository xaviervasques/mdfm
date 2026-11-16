"""Gait readouts of a body on the 100 Hz clock, and the shank-gyroscope initial-contact baseline (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 2.7, task B4.12). Fixed formulas, not learned.

Conventions: world y up, metres, the horizontal plane is (x, z); joints are the 22 SMPL body joints
(pelvis 0, left ankle 7, right ankle 8); times are seconds on the trial clock, frame f at f / rate, a
time maps to the frame round(t x rate); feet are "L" and "R", contact columns [left, right].

- initial_contacts: an initial contact (IC) is a frame at or above thr whose previous frame is finite and
  below it (first frame of contact, the walkway convention of scripts/build_weargait_targets.py, which
  also merges contact runs under 50 ms apart; here only the refractory period guards against a flicker);
  contact at frame 0 and a rise next to a missing (NaN) frame are no event; an IC closer than refractory_s
  to the last kept IC of the same foot is dropped.
- strides: successive ICs of one foot; length is the horizontal distance of that foot's ankle joint between
  the two IC frames, read inside one window frame. Joints (T, 22, 3) are one frame. Joints (W, T, 22, 3)
  hold window w's body on the trial clock (on_clock: NaN outside the window) and window_of_frame (T,) comes
  from stitch; a stride is read in the window of its midpoint frame (with full 10 s windows and a 5 s hop
  that window holds the whole stride when the stride lasts under 5 s); when that window lacks one of the two
  ends (a short last window, whose centre claims frames near its start), in the window holding both ends
  that is nearest to it in window order (the earlier on a tie).
- pass_speed: horizontal pelvis displacement between two frames over their time difference. On a
  stitched body the displacement is read in the window of the midpoint frame when that window holds both
  frames (always for a pass under 5 s with 10 s windows and a 5 s hop); otherwise the path is cut into runs
  of equal window_of_frame and the chords of the runs are summed, each read in its run's window from the
  run's first frame to the next run's first frame (inside the window as long as windows overlap); on a
  straight pass the sum equals the single chord.
- cadence: the rule of scripts/build_weargait_targets.py: ICs of both feet sorted, split where two are
  more than 1.5 s apart, 60 x sum(n - 1) / sum(duration) over the passes with at least 3 ICs.
- stitch: each frame from the window where it is most central (distance to the window's centre sample;
  a tie keeps the earlier window).
- shank_gyro_ic (Salarian et al. 2004, fixed before any real read): the sagittal axis is the first
  principal axis of the centred gyroscope over the finite samples of the recording, signed so that the
  largest excursions (the mid-swing peaks) are positive (flipped when the 1st percentile lies farther from
  0 than the 99th); mid-swing peaks are local maxima of the sagittal rate above 50 deg/s, at least 0.5 s
  apart (the smaller of two closer peaks is dropped); the IC of a mid-swing peak is the next negative
  peak: the first sample k within the 0.4 s that follow the peak with x[k] < 0, x[k] <= x[k - 1] and
  x[k] < x[k + 1], searched only when those samples and the one after them are finite and inside the
  recording.
- heel_strikes (Zeni et al. 2008, on the ankle joint): a kinematic heel-strike reference of a body, the
  local maxima (at least 0.5 s apart) of the ankle's horizontal position relative to the pelvis along the
  direction of progression, that direction being the horizontal pelvis velocity averaged over a centred
  1 s window, kept where that averaged speed is at least 0.2 m/s. The contact readout (initial_contacts
  on contact labels of mdfm.body.frames: a foot joint within 5 cm of the floor and slower than 0.3 m/s)
  marks foot loading, later than the heel strike.
- stance_speed (reports/bench/root_motion_diagnosis.md, C4 and fix 7): the horizontal pelvis speed implied by
  stance-foot no-slip, read from the pelvis-relative joints alone (the root translation is not read): at each
  frame, minus the horizontal velocity (central differences) of the foot point (the mean of the ankle and the
  SMPL foot joint) of each foot in contact (contact at or above thr), averaged over the feet in contact; NaN
  where no foot is in contact or a read joint is missing.
"""
import math

import numpy as np

UP = 1
PELVIS = 0
ANKLES = {"L": 7, "R": 8}
FOOT_POINTS = ((7, 10), (8, 11))    # (ankle, SMPL foot joint) of the left and right foot
FEET = ("L", "R")
RATE = 100.0
CONTACT_THR = 0.5           # contact probability
REFRACTORY_S = 0.3          # s between two ICs of one foot, at least
PASS_GAP_S = 1.5            # s, the pass split of the cadence rule
SWING_DPS = 50.0            # deg/s, mid-swing peak height
SWING_DISTANCE_S = 0.5      # s between two mid-swing peaks of one shank, at least
IC_SEARCH_S = 0.4           # s after the mid-swing peak
HS_SMOOTH_S = 1.0           # s, window of the direction of progression (heel_strikes)
HS_MIN_SPEED = 0.2          # m/s, averaged pelvis speed where heel strikes are kept
HS_DISTANCE_S = 0.5         # s between two heel strikes of one foot, at least


def _np(x):
    """float64 numpy copy of an array or tensor."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float64)


def frame_of(t, rate=RATE):
    """Frame index (int64) of times t (s): round(t x rate)."""
    return np.rint(np.asarray(t, dtype=np.float64) * rate).astype(np.int64)


def _horizontal_norm(d):
    return float(math.hypot(d[0], d[2]))


# ---------------------------------------------------------------- contacts
def initial_contacts(contact_prob, rate=RATE, thr=CONTACT_THR, refractory_s=REFRACTORY_S):
    """contact_prob (T, 2) [left, right] in [0, 1] (NaN where unknown) -> {"L": times, "R": times}, s,
    float64, increasing (module docstring)."""
    p = _np(contact_prob)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"contact_prob must be (T, 2), got {p.shape}")
    gap = int(round(refractory_s * rate))
    out = {}
    for j, foot in enumerate(FEET):
        x = p[:, j]
        ok = np.isfinite(x)
        on = ok & (x >= thr)
        rise = np.flatnonzero(on[1:] & ok[:-1] & ~on[:-1]) + 1
        kept = []
        for f in rise:
            if kept and f - kept[-1] < gap:
                continue
            kept.append(int(f))
        out[foot] = np.asarray(kept, dtype=np.float64) / rate
    return out


# ---------------------------------------------------------------- strides, speed, cadence
def _stack(joints, window_of_frame):
    """joints -> (W, T, 22, 3) float64 and window_of_frame (T,) int64 (zeros for a single frame)."""
    J = _np(joints)
    if J.ndim == 3:
        if window_of_frame is not None:
            raise ValueError("window_of_frame needs joints (W, T, 22, 3) on the trial clock")
        return J[None], np.zeros(J.shape[0], np.int64)
    if J.ndim != 4 or window_of_frame is None:
        raise ValueError(f"joints must be (T, 22, 3), or (W, T, 22, 3) with window_of_frame; got {J.shape}")
    wof = np.asarray(window_of_frame, dtype=np.int64)
    if wof.shape != (J.shape[1],) or wof.min() < 0 or wof.max() >= J.shape[0]:
        raise ValueError(f"window_of_frame {wof.shape} does not index the {J.shape[0]} windows of {J.shape[1]} frames")
    return J, wof


def strides(joints, ics, window_of_frame=None, rate=RATE):
    """joints (T, 22, 3) m, or (W, T, 22, 3) m with window_of_frame (T,); ics {"L": times, "R": times} s
    -> DataFrame(foot, t0, t1, length_m, window) sorted by t0, one row per pair of successive ICs of one
    foot (module docstring). attrs["dropped"]: pairs outside the clock or that no window holds with a finite
    ankle joint at both ends."""
    import pandas as pd
    J, wof = _stack(joints, window_of_frame)
    T = J.shape[1]
    rows, dropped = [], 0
    for foot in FEET:
        t = np.sort(np.asarray(ics.get(foot, ()), dtype=np.float64))
        for t0, t1 in zip(t[:-1], t[1:]):
            f0, f1 = int(frame_of(t0, rate)), int(frame_of(t1, rate))
            if not 0 <= f0 < f1 < T:
                dropped += 1
                continue
            a = ANKLES[foot]
            w = int(wof[(f0 + f1) // 2])
            length = _horizontal_norm(J[w, f1, a] - J[w, f0, a])
            if not math.isfinite(length) and J.shape[0] > 1:
                both = np.flatnonzero(np.isfinite(J[:, f0, a]).all(axis=1) & np.isfinite(J[:, f1, a]).all(axis=1))
                if len(both):
                    w = int(both[np.argmin(np.abs(both - w))])
                    length = _horizontal_norm(J[w, f1, a] - J[w, f0, a])
            if not math.isfinite(length):
                dropped += 1
                continue
            rows.append((foot, float(t0), float(t1), length, w))
    df = pd.DataFrame(rows, columns=["foot", "t0", "t1", "length_m", "window"])
    df = df.sort_values(["t0", "foot"], kind="stable").reset_index(drop=True)
    df.attrs["dropped"] = dropped
    return df


def pass_speed(joints, t0, t1, window_of_frame=None, rate=RATE):
    """Horizontal pelvis speed (m/s) between times t0 < t1 (s): displacement over (f1 - f0) / rate, the
    chords of the window runs summed on a stitched body (module docstring). NaN when a read joint is
    missing."""
    J, wof = _stack(joints, window_of_frame)
    f0, f1 = int(frame_of(t0, rate)), int(frame_of(t1, rate))
    if not 0 <= f0 < f1 < J.shape[1]:
        raise ValueError(f"frames {f0} to {f1} outside the clock of {J.shape[1]} frames or not increasing")
    k = int(wof[(f0 + f1) // 2])
    one = _horizontal_norm(J[k, f1, PELVIS] - J[k, f0, PELVIS])
    if math.isfinite(one):
        return one / ((f1 - f0) / rate)
    w = wof[f0:f1 + 1]
    cuts = np.flatnonzero(np.diff(w)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(w) - 1]])
    dist = 0.0
    for s, e in zip(starts, ends):
        k = int(w[s])
        dist += _horizontal_norm(J[k, f0 + e, PELVIS] - J[k, f0 + s, PELVIS])
    return dist / ((f1 - f0) / rate)


def cadence(ics, gap_s=PASS_GAP_S):
    """steps/min from the ICs of both feet (module docstring); NaN without a pass of 3 ICs."""
    parts = [np.asarray(v, dtype=np.float64) for v in ics.values()]
    t = np.sort(np.concatenate(parts)) if parts else np.zeros(0)
    steps = dur = 0.0
    cuts = np.flatnonzero(np.diff(t) > gap_s) + 1
    for seg in np.split(t, cuts):
        if len(seg) >= 3 and seg[-1] > seg[0]:
            steps += len(seg) - 1
            dur += seg[-1] - seg[0]
    return 60.0 * steps / dur if dur > 0 else float("nan")


def stance_speed(rel, contact, rate=RATE, thr=CONTACT_THR):
    """rel (T, 22, 3) m: joint positions minus the pelvis in one frame (the root translation is not read);
    contact (T, 2) [left, right] probabilities or labels -> (T,) m/s, the horizontal pelvis speed implied by
    stance-foot no-slip (module docstring), NaN where no foot is in contact."""
    R = _np(rel)
    c = _np(contact)
    if R.ndim != 3 or R.shape[1] < 12 or R.shape[2] != 3:
        raise ValueError(f"rel must be (T, 22, 3), got {R.shape}")
    if c.shape != (R.shape[0], 2):
        raise ValueError(f"contact must be ({R.shape[0]}, 2), got {c.shape}")
    T = R.shape[0]
    acc, cnt = np.zeros((T, 2)), np.zeros(T)
    if T < 2:
        return np.full(T, np.nan)
    for j, (a, b) in enumerate(FOOT_POINTS):
        p = 0.5 * (R[:, a] + R[:, b])[:, [0, 2]]
        v = np.gradient(p, 1.0 / rate, axis=0)
        on = np.isfinite(c[:, j]) & (c[:, j] >= thr) & np.isfinite(v).all(axis=1)
        acc[on] -= v[on]
        cnt[on] += 1
    out = np.full(T, np.nan)
    ok = cnt > 0
    out[ok] = np.linalg.norm(acc[ok] / cnt[ok, None], axis=1)
    return out


# ---------------------------------------------------------------- windows
def _starts(windows, hop, starts):
    if starts is None:
        if hop is None:
            raise ValueError("give hop or starts")
        return [k * int(hop) for k in range(len(windows))]
    if len(starts) != len(windows):
        raise ValueError(f"{len(starts)} starts for {len(windows)} windows")
    return [int(s) for s in starts]


def stitch(windows, hop, starts=None):
    """windows: sequence of arrays (T_k, ...) with window k starting at frame k x hop (or starts[k]) ->
    (stitched (N, ...) float64, window_of_frame (N,) int64), each frame from the window where it is most
    central; frames no window covers are NaN with window -1."""
    arrs = [_np(w) for w in windows]
    s0 = _starts(arrs, hop, starts)
    n = max(s + len(a) for s, a in zip(s0, arrs))
    out = np.full((n,) + arrs[0].shape[1:], np.nan)
    wof = np.full(n, -1, np.int64)
    best = np.full(n, np.inf)
    for k, (s, a) in enumerate(zip(s0, arrs)):
        i = np.arange(len(a))
        c = np.abs(i - (len(a) - 1) / 2.0)
        f = s + i
        better = c < best[f]
        out[f[better]] = a[better]
        wof[f[better]] = k
        best[f[better]] = c[better]
    return out, wof


def on_clock(windows, hop, starts=None):
    """windows: sequence of arrays (T_k, ...) -> (W, N, ...) float64 on the trial clock, NaN outside each
    window (the joints argument of strides and pass_speed on a stitched body)."""
    arrs = [_np(w) for w in windows]
    s0 = _starts(arrs, hop, starts)
    n = max(s + len(a) for s, a in zip(s0, arrs))
    out = np.full((len(arrs), n) + arrs[0].shape[1:], np.nan)
    for k, (s, a) in enumerate(zip(s0, arrs)):
        out[k, s:s + len(a)] = a
    return out


# ---------------------------------------------------------------- shank-gyroscope baseline
def sagittal_rate(gyro):
    """gyro (T, 3) deg/s -> (rate (T,) deg/s along the signed first principal axis, NaN where a sample is
    missing; axis (3,)) (module docstring)."""
    g = _np(gyro)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gyro must be (T, 3), got {g.shape}")
    ok = np.isfinite(g).all(axis=1)
    if ok.sum() < 3:
        raise ValueError("fewer than 3 finite gyroscope samples")
    x = g[ok] - g[ok].mean(axis=0)
    _, vec = np.linalg.eigh(x.T @ x)
    axis = vec[:, -1]
    s = np.where(ok, np.nan_to_num(g) @ axis, np.nan)
    lo, hi = np.percentile(s[ok], [1, 99])
    if -lo > hi:
        axis, s = -axis, -s
    return s, axis


def shank_gyro_ic(gyro, rate=RATE, swing_dps=SWING_DPS, search_s=IC_SEARCH_S, distance_s=SWING_DISTANCE_S):
    """Initial contacts (s, float64, increasing) of one shank from its gyroscope (T, 3) in deg/s (module
    docstring)."""
    from scipy.signal import find_peaks
    s, _ = sagittal_rate(gyro)
    x = np.where(np.isfinite(s), s, 0.0)          # a missing sample holds no swing peak (0 < swing_dps)
    peaks, _ = find_peaks(x, height=swing_dps, distance=max(1, int(round(distance_s * rate))))
    n = int(round(search_s * rate))
    out = []
    for p in peaks:
        seg = s[p:p + n + 2]                      # the peak, the n samples after it and one more
        if len(seg) < n + 2 or not np.isfinite(seg).all():
            continue
        k = np.flatnonzero((seg[1:-1] < 0) & (seg[1:-1] <= seg[:-2]) & (seg[1:-1] < seg[2:]))
        if len(k):
            out.append((p + 1 + int(k[0])) / rate)
    return np.asarray(out, dtype=np.float64)


def heel_strikes(joints, rate=RATE, smooth_s=HS_SMOOTH_S, min_speed=HS_MIN_SPEED, distance_s=HS_DISTANCE_S):
    """joints (T, 22, 3) m, one frame -> {"L": times, "R": times} s of the kinematic heel strikes (module
    docstring)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    J = _np(joints)
    if J.ndim != 3:
        raise ValueError(f"joints must be (T, 22, 3), got {J.shape}")
    v = np.gradient(J[:, PELVIS][:, [0, 2]], 1.0 / rate, axis=0)
    v = uniform_filter1d(v, max(1, int(round(smooth_s * rate))), axis=0, mode="nearest")
    speed = np.linalg.norm(v, axis=1)
    u = v / np.maximum(speed, 1e-9)[:, None]
    out = {}
    for foot in FEET:
        rel = ((J[:, ANKLES[foot]][:, [0, 2]] - J[:, PELVIS][:, [0, 2]]) * u).sum(axis=1)
        pk, _ = find_peaks(rel, distance=max(1, int(round(distance_s * rate))))
        out[foot] = pk[speed[pk] >= min_speed].astype(np.float64) / rate
    return out
