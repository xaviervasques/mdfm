"""The measures of section 3.4 of docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md and their void check.

Every measure is computed the same way on the truth (the drawn point set of the driver clip, mdfm/video/points.py)
and on an estimate in the same order (wholebody133_to_drawn maps a ViTPose whole-body estimate to it), so that a
fixed bias of a rule cancels. Values marked [cfg] are proposed, like every threshold of the draft.

- Tremor (tremor_measure): the wrist minus the shoulder keypoint along the image direction of the driven motion
  (driven_direction: the first principal axis of the detrended truth difference), linear detrend, Hann window
  over the clip, dominant frequency in the band (3 to 7 Hz rest, 7 to 13 Hz action; inclusive edges), amplitude
  twice the windowed spectrum magnitude at that frequency over the window sum, divided by the clip's torso length
  in pixels (median neck to hip-midpoint distance), so that a scale change of the generated person cancels.
- Image-plane joint angles (joint_angles), degrees, per frame: trunk (hip midpoint to neck against the image
  vertical), neck (trunk line against neck to nose), shoulders and hips (trunk line downward against the upper
  arm, the thigh), elbows and knees (flexion, 180 minus the interior angle). Keypoints are low-passed first,
  zero-phase fourth-order Butterworth at LOWPASS_HZ [cfg] (lowpass).
- Initial contacts (initial_contacts), one fixed rule [cfg]: on the low-passed keypoints, the forward distance
  of each ankle from the hip midpoint along the direction of progression in the image (the sign of the median
  horizontal velocity of the hip midpoint); a contact is a peak of it above 0, peaks at least IC_MIN_SEP_S
  apart, none within EDGE_S of either end of the clip. The body's contact labels (contact_onsets) are reported
  beside.
- Jerk onsets (jerk_onsets) [cfg]: rising crossings of JERK_DPS by the angular speed of a joint angle low-passed
  at JERK_LOWPASS_HZ, merged within JERK_MIN_GAP_S.
- Events of truth and estimate are matched one to one, nearest first within a tolerance (match_events).
- Dropouts (dropout_inputs): image speed in torso lengths per second and the depth-order occlusion proxy of
  mdfm/factory/camera_render.py over the drawn set's own bones: inputs of the calibration, not criteria.
- Seams (seam_phase_jumps): the phase of a sinusoid at the clip's dominant frequency fitted to 1 s before and
  1 s after each seam, times on the clip's clock, so that a continuous oscillation jumps by 0.

Void check (section 3.4): the analysis runs on the truth plus the Student-t pixel noise of the default
mdfm.factory.camera_render.TrackerProcess (scale 3 px, 4 degrees of freedom), fixed seeds. tremor_void_check
follows the protocol of the V0 review's R6 (reports/weekend_2026-10-01/tasks/V0.md: a sinusoid of each driven
frequency and amplitude at the tremor framing, noise on the wrist and the shoulder, seed 1000 f + s, 200 seeds per
frequency) and admits to T1 and T2 the levels where, at every frequency of the band, the frequency lies within
0.2 Hz and the amplitude ratio within 0.5 to 2 in at least 95 percent of the seeds; T3's bound is the 95th
percentile of the noise-only in-band amplitude. walk_void_check scores H1 and H2 of a walk's truth under the same
noise on its 20 body points.
"""
import math

import numpy as np
import torch

from ..factory import camera_render as CR
from . import draw as D
from . import points as P

BANDS = {"rest": (3.0, 7.0), "action": (7.0, 13.0)}
FREQS = {"rest": (4.0, 5.0, 6.0), "action": (8.0, 10.0, 12.0)}
LEVELS_MM = (0.0, 2.0, 5.0, 10.0, 20.0)
F_TOL_HZ = 0.2              # T1
RATIO_BOUNDS = (0.5, 2.0)   # T2, per clip
ADMIT_SHARE = 0.95          # share of seeds per frequency
TOL_EPS = 1e-9              # comparisons of frequencies on the 0.1 Hz grid are inclusive
LOWPASS_HZ = 6.0            # [cfg] keypoint low-pass for angles and contacts
IC_MIN_SEP_S = 0.5          # [cfg]
EDGE_S = 0.25               # [cfg]
MATCH_TOL_S = 0.25          # [cfg] event matching tolerance
JERK_DPS = 20.0             # [cfg] angular speed threshold, deg/s
JERK_LOWPASS_HZ = 8.0       # [cfg]
JERK_MIN_GAP_S = 0.3        # [cfg]
H1_PASS_DEG, H1_FAIL_DEG = 5.0, 10.0
H2 = {"median_frames": 1.0, "p90_frames": 2.0, "unmatched": 0.05}
WALK_ANGLES = ("l_hip", "r_hip", "l_knee", "r_knee")
SIDES = {"l": {"shoulder": "LShoulder", "elbow": "LElbow", "wrist": "LWrist", "hip": "LHip", "knee": "LKnee",
               "ankle": "LAnkle"},
         "r": {"shoulder": "RShoulder", "elbow": "RElbow", "wrist": "RWrist", "hip": "RHip", "knee": "RKnee",
               "ankle": "RAnkle"}}
I = P.IDX


# ---------------------------------------------------------------- signal helpers
def lowpass(x, fps, cutoff_hz=LOWPASS_HZ, order=4):
    """Zero-phase Butterworth low-pass along axis 0 (scipy filtfilt); x unchanged when cutoff_hz is None or at
    or above the Nyquist frequency."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, dtype=np.float64)
    if cutoff_hz is None or cutoff_hz >= fps / 2:
        return x
    b, a = butter(order, cutoff_hz / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def spectral_peak(x, fps, band):
    """(dominant frequency Hz, amplitude) of a signal (T,): linear detrend, Hann window over the whole signal,
    the largest windowed spectrum magnitude in band (inclusive), amplitude 2 |X| / sum(window)."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    tt = np.arange(T) / fps
    x = x - np.polyval(np.polyfit(tt, x, 1), tt)
    win = np.hanning(T)
    X = np.fft.rfft(x * win)
    freq = np.fft.rfftfreq(T, 1 / fps)
    k = int(np.argmax(np.abs(X) * ((freq >= band[0]) & (freq <= band[1]))))
    return float(freq[k]), float(2 * np.abs(X[k]) / win.sum())


def torso_px(uv):
    """Median neck to hip-midpoint distance (px) of keypoints (T, >= 20, 2) in the drawn order."""
    uv = np.asarray(uv, dtype=np.float64)
    mid = (uv[:, I["RHip"]] + uv[:, I["LHip"]]) / 2
    return float(np.median(np.linalg.norm(uv[:, I["Neck"]] - mid, axis=-1)))


# ---------------------------------------------------------------- tremor
def tremor_difference(uv, side="l"):
    """Wrist minus shoulder (T, 2) px of one side."""
    s = SIDES[side]
    uv = np.asarray(uv, dtype=np.float64)
    return uv[:, I[s["wrist"]]] - uv[:, I[s["shoulder"]]]


def driven_direction(uv_truth, side="l"):
    """Unit image direction (2,) of the driven motion: the first principal axis of the linearly detrended truth
    wrist minus shoulder, its x component made non-negative."""
    d = tremor_difference(uv_truth, side)
    tt = np.arange(len(d), dtype=np.float64)
    d = d - np.stack([np.polyval(np.polyfit(tt, d[:, c], 1), tt) for c in range(2)], 1)
    v = np.linalg.svd(d, full_matrices=False)[2][0]
    return -v if v[0] < 0 or (v[0] == 0 and v[1] < 0) else v


def tremor_measure(uv, fps, band, direction, side="l", torso=None):
    """{freq_hz, amp_px, amp_torso}: the tremor measure of keypoints (T, >= 20, 2) along direction (2,);
    torso px (default torso_px(uv))."""
    x = tremor_difference(uv, side) @ np.asarray(direction, dtype=np.float64)
    f, a = spectral_peak(x, fps, band)
    tor = torso_px(uv) if torso is None else float(torso)
    return {"freq_hz": f, "amp_px": a, "amp_torso": a / tor}


# ---------------------------------------------------------------- angles
def signed_angle(u, v):
    """Signed angle (deg, (-180, 180]) from u to v, (..., 2) image vectors (y down)."""
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    dot = (u * v).sum(-1)
    return np.degrees(np.arctan2(cross, dot))


def flexion(a, b, c):
    """180 minus the interior angle at b of the chain a-b-c (deg): 0 for a straight limb."""
    return 180.0 - np.abs(signed_angle(a - b, c - b))


def joint_angles(uv):
    """{name: (T,) deg} of keypoints (T, >= 20, 2) px (module docstring)."""
    uv = np.asarray(uv, dtype=np.float64)
    neck = uv[:, I["Neck"]]
    mid = (uv[:, I["RHip"]] + uv[:, I["LHip"]]) / 2
    up = neck - mid
    out = {"trunk": signed_angle(np.broadcast_to(np.array([0.0, -1.0]), up.shape), up),
           "neck": signed_angle(up, uv[:, I["Nose"]] - neck)}
    for side, s in SIDES.items():
        S, E, W = uv[:, I[s["shoulder"]]], uv[:, I[s["elbow"]]], uv[:, I[s["wrist"]]]
        H, K, A = uv[:, I[s["hip"]]], uv[:, I[s["knee"]]], uv[:, I[s["ankle"]]]
        out[f"{side}_shoulder"] = signed_angle(-up, E - S)
        out[f"{side}_elbow"] = flexion(S, E, W)
        out[f"{side}_hip"] = signed_angle(-up, K - H)
        out[f"{side}_knee"] = flexion(H, K, A)
    return out


def angle_error(a, b):
    """a - b wrapped to [-180, 180) deg."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def rmse(a, b):
    return float(np.sqrt(np.mean(angle_error(a, b) ** 2)))


# ---------------------------------------------------------------- events
def initial_contacts(uv, fps, cutoff_hz=LOWPASS_HZ, min_sep_s=IC_MIN_SEP_S, edge_s=EDGE_S):
    """({'l': frames, 'r': frames} int arrays, direction +1 or -1) of initial contacts (module docstring)."""
    from scipy.signal import find_peaks
    x = lowpass(np.asarray(uv, dtype=np.float64)[:, :P.N_BODY20], fps, cutoff_hz)
    T = x.shape[0]
    mid = (x[:, I["RHip"]] + x[:, I["LHip"]]) / 2
    direction = 1.0 if np.median(np.gradient(mid[:, 0])) >= 0 else -1.0
    e = int(round(edge_s * fps))
    out = {}
    for side, s in SIDES.items():
        f = direction * (x[:, I[s["ankle"]], 0] - mid[:, 0])
        peaks, _ = find_peaks(f, distance=max(1, int(round(min_sep_s * fps))), height=0.0)
        out[side] = peaks[(peaks >= e) & (peaks <= T - 1 - e)].astype(np.int64)
    return out, direction


def contact_onsets(contact):
    """{'l': frames, 'r': frames} where the body's contact label (T, 2) [left, right] turns on (frame 0 left
    out)."""
    c = np.asarray(contact, dtype=bool)
    return {side: (np.flatnonzero(c[1:, j] & ~c[:-1, j]) + 1).astype(np.int64) for j, side in enumerate("lr")}


def match_events(true, est, tol):
    """One-to-one matching, nearest pairs first within tol (frames): (errors est - true of the matched pairs, in
    true order; unmatched true events; unmatched estimates)."""
    true, est = np.asarray(true, dtype=np.float64), np.asarray(est, dtype=np.float64)
    pairs = sorted((abs(e - t), i, j) for i, t in enumerate(true) for j, e in enumerate(est) if abs(e - t) <= tol)
    ut, ue, m = set(range(len(true))), set(range(len(est))), {}
    for _, i, j in pairs:
        if i in ut and j in ue:
            m[i] = j
            ut.discard(i)
            ue.discard(j)
    err = np.array([est[m[i]] - true[i] for i in sorted(m)], dtype=np.float64)
    return err, len(ut), len(ue)


def jerk_onsets(angle_deg, fps, threshold_dps=JERK_DPS, cutoff_hz=JERK_LOWPASS_HZ, min_gap_s=JERK_MIN_GAP_S):
    """Frames where the angular speed of a joint angle (T,) deg rises above threshold_dps (module docstring)."""
    a = lowpass(np.unwrap(np.radians(np.asarray(angle_deg, dtype=np.float64))), fps, cutoff_hz)
    w = np.abs(np.gradient(np.degrees(a)) * fps)
    above = w > threshold_dps
    rise = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above[0]:
        rise = np.concatenate([[0], rise])
    out, last = [], -np.inf
    gap = min_gap_s * fps
    for r in rise:
        if r - last >= gap:
            out.append(r)
        last = r
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------- seams and dropouts
def seam_phase_jumps(x, fps, freq_hz, seams, win_s=1.0):
    """[{seam, jump_deg, amp_before, amp_after}]: per seam frame, the phase jump of a sinusoid at freq_hz fitted
    (with an offset and a linear trend) to win_s before and win_s after it, on the clip's clock."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    t = np.arange(T) / fps
    n = int(round(win_s * fps))
    w = 2 * np.pi * freq_hz
    out = []
    for s in seams:
        fits = []
        for sl in (slice(max(0, s - n), s), slice(s, min(T, s + n))):
            tt = t[sl]
            A = np.stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt), tt - t[min(s, T - 1)]], 1)
            c = np.linalg.lstsq(A, x[sl], rcond=None)[0]
            fits.append((math.atan2(c[1], c[0]), math.hypot(c[0], c[1])))
        jump = (math.degrees(fits[1][0] - fits[0][0]) + 180.0) % 360.0 - 180.0
        out.append({"seam": int(s), "jump_deg": jump, "amp_before": fits[0][1], "amp_after": fits[1][1]})
    return out


def drawn_bones():
    """(a, b) index pairs of the drawn set's bones: the 19 limbs and the 20 edges of each hand."""
    bones = [(a - 1, b - 1) for a, b in D.LIMB_SEQ]
    for sl in (P.LHAND, P.RHAND):
        bones += [(sl.start + a, sl.start + b) for a, b in D.HAND_EDGES]
    return bones


def dropout_inputs(uv, depth, fps, radius=0.3, margin=0.05):
    """(speed (T, 62) torso lengths per second, occlusion proxy (T, 62)) of the truth: camera_render.image_speed
    over torso_px, and the proxy of camera_render.occlusion with the drawn set's bones as neighbours."""
    uv_t = torch.as_tensor(np.asarray(uv, dtype=np.float64))
    z = torch.as_tensor(np.asarray(depth, dtype=np.float64))
    tor = torso_px(uv)
    speed = CR.image_speed(uv_t, float(fps)) / tor
    n = uv_t.shape[1]
    occ_m = 1 - torch.eye(n, dtype=torch.float64)
    for a, b in drawn_bones():
        occ_m[a, b] = occ_m[b, a] = 0.0
    behind = (z[:, :, None] - z[:, None, :]) > margin
    d2 = ((uv_t[:, :, None, :] - uv_t[:, None, :, :]) ** 2).sum(-1)
    k = torch.exp(-d2 / (2 * (radius * tor) ** 2)) * behind * occ_m
    return speed.numpy(), (1 - torch.prod(1 - k, dim=-1)).numpy()


def dropout_rates(lost, speed, occ, speed_edges=(0.0, 1.0, 2.0, 5.0, np.inf), occ_edges=(0.0, 0.25, 0.5, 1.0)):
    """[{speed_bin, occ_bin, n, lost_share}] of a lost-detection mask (T, P) against the truth inputs."""
    lost, speed, occ = (np.asarray(a).ravel() for a in (lost, speed, occ))
    rows = []
    for i in range(len(speed_edges) - 1):
        for j in range(len(occ_edges) - 1):
            top = occ <= occ_edges[j + 1] if j == len(occ_edges) - 2 else occ < occ_edges[j + 1]
            m = (speed >= speed_edges[i]) & (speed < speed_edges[i + 1]) & (occ >= occ_edges[j]) & top
            rows.append({"speed_bin": [speed_edges[i], speed_edges[i + 1]], "occ_bin": [occ_edges[j], occ_edges[j + 1]],
                         "n": int(m.sum()), "lost_share": float(lost[m].mean()) if m.any() else None})
    return rows


# ---------------------------------------------------------------- scoring a walk
def score_walk(uv_true, uv_est, fps, cutoff_hz=LOWPASS_HZ, tol_s=MATCH_TOL_S):
    """H1 and H2 inputs of one clip: RMSE (deg) of each walk angle and pooled over WALK_ANGLES, and the matched
    initial-contact errors (frames) with the unmatched counts, the estimate against the truth (both by the same
    rules)."""
    at = joint_angles(lowpass(np.asarray(uv_true, dtype=np.float64)[:, :P.N_BODY20], fps, cutoff_hz))
    ae = joint_angles(lowpass(np.asarray(uv_est, dtype=np.float64)[:, :P.N_BODY20], fps, cutoff_hz))
    per = {k: rmse(ae[k], at[k]) for k in WALK_ANGLES}
    pooled = float(np.sqrt(np.mean([per[k] ** 2 for k in WALK_ANGLES])))
    ict, _ = initial_contacts(uv_true, fps, cutoff_hz)
    ice, _ = initial_contacts(uv_est, fps, cutoff_hz)
    errs, n_true, un_t, un_e = [], 0, 0, 0
    for side in "lr":
        e, ut, ue = match_events(ict[side], ice[side], tol_s * fps)
        errs.append(e)
        n_true += len(ict[side])
        un_t += ut
        un_e += ue
    errs = np.concatenate(errs) if errs else np.zeros(0)
    return {"rmse_deg": per, "rmse_pooled_deg": pooled, "ic_errors_frames": errs, "ic_true": n_true,
            "ic_unmatched": un_t, "ic_false": un_e}


def h2_summary(errs, n_true, unmatched):
    a = np.abs(np.asarray(errs, dtype=np.float64))
    return {"median_frames": float(np.median(a)) if len(a) else None,
            "p90_frames": float(np.percentile(a, 90)) if len(a) else None,
            "unmatched_share": unmatched / n_true if n_true else None}


def h2_pass(s):
    return (s["median_frames"] is not None and s["median_frames"] <= H2["median_frames"]
            and s["p90_frames"] <= H2["p90_frames"] and s["unmatched_share"] <= H2["unmatched"])


# ---------------------------------------------------------------- void checks
def _noise(tracker, shape, seed):
    return tracker.noise(shape, torch.Generator().manual_seed(int(seed))).numpy()


def tremor_void_check(px_per_mm, fps=30.0, duration_s=10.0, seeds=200, tracker=None, levels=LEVELS_MM,
                      freqs=None, bands=None, phase=0.3):
    """The tremor void check (module docstring) -> {band: {levels: [...], admitted_mm, t3_bound_mm}}. Per
    level and frequency: share of seeds with the frequency within F_TOL_HZ (inclusive; also the share under a
    plain floating-point comparison, as R6 computed it), share with the amplitude ratio inside RATIO_BOUNDS, and
    the ratio's median and 5th and 95th percentiles; for the zero level, the in-band amplitude (mm) of noise
    alone. Pooled over the band's frequencies as R6 printed them."""
    tr = CR.TrackerProcess() if tracker is None else tracker
    freqs = FREQS if freqs is None else freqs
    bands = BANDS if bands is None else bands
    T = int(round(duration_s * fps))
    tt = np.arange(T) / fps
    out = {}
    for band, edges in bands.items():
        rows = []
        for a_mm in levels:
            per_f, pooled = [], {"ferr": [], "ratio": [], "amp_mm": []}
            for f0 in freqs[band]:
                truth = a_mm * px_per_mm * np.sin(2 * np.pi * f0 * tt + phase)
                a_true = spectral_peak(truth, fps, edges)[1] if a_mm else px_per_mm
                ferr, ratio = [], []
                for s in range(seeds):
                    n = _noise(tr, (T, 2, 2), 1000 * f0 + s)
                    fd, a = spectral_peak(truth + n[:, 0, 0] - n[:, 1, 0], fps, edges)
                    ferr.append(abs(fd - f0))
                    ratio.append(a / a_true)
                ferr, ratio = np.array(ferr), np.array(ratio)
                row = {"freq_hz": f0}
                if a_mm:
                    row.update(share_f_within=float(np.mean(ferr <= F_TOL_HZ + TOL_EPS)),
                               share_f_within_float=float(np.mean(ferr <= F_TOL_HZ)),
                               share_ratio_in=float(np.mean((ratio >= RATIO_BOUNDS[0]) & (ratio <= RATIO_BOUNDS[1]))),
                               ratio_median=float(np.median(ratio)), ratio_p5=float(np.percentile(ratio, 5)),
                               ratio_p95=float(np.percentile(ratio, 95)))
                else:
                    row.update(amp_mm_median=float(np.median(ratio)), amp_mm_p95=float(np.percentile(ratio, 95)))
                per_f.append(row)
                pooled["ferr"].append(ferr)
                pooled["ratio"].append(ratio)
            fe, ra = np.concatenate(pooled["ferr"]), np.concatenate(pooled["ratio"])
            lvl = {"level_mm": a_mm, "level_px": a_mm * px_per_mm, "per_freq": per_f, "pooled": {}}
            if a_mm:
                lvl["pooled"].update(share_f_within=float(np.mean(fe <= F_TOL_HZ + TOL_EPS)),
                                     share_f_within_float=float(np.mean(fe <= F_TOL_HZ)),
                                     ratio_median=float(np.median(ra)), ratio_p5=float(np.percentile(ra, 5)),
                                     ratio_p95=float(np.percentile(ra, 95)),
                                     share_ratio_in=float(np.mean((ra >= RATIO_BOUNDS[0]) & (ra <= RATIO_BOUNDS[1]))))
                lvl["admitted"] = all(r["share_f_within"] >= ADMIT_SHARE and r["share_ratio_in"] >= ADMIT_SHARE
                                      for r in per_f)
                lvl["admitted_float"] = all(r["share_f_within_float"] >= ADMIT_SHARE
                                            and r["share_ratio_in"] >= ADMIT_SHARE for r in per_f)
            else:
                lvl["pooled"].update(amp_mm_median=float(np.median(ra)), amp_mm_p95=float(np.percentile(ra, 95)))
            rows.append(lvl)
        zero = [r for r in rows if not r["level_mm"]]
        out[band] = {"edges_hz": list(edges), "levels": rows,
                     "admitted_mm": [r["level_mm"] for r in rows if r.get("admitted")],
                     "admitted_mm_float": [r["level_mm"] for r in rows if r.get("admitted_float")],
                     "t3_bound_mm": zero[0]["pooled"]["amp_mm_p95"] if zero else None}
    return {"fps": fps, "duration_s": duration_s, "samples": T, "seeds": seeds, "px_per_mm": px_per_mm,
            "tracker": {"noise_px": tr.noise_px, "nu": tr.nu}, "phase_rad": phase, "bands": out}


def walk_void_check(uv_truth, fps, seeds=200, tracker=None, seed0=0, cutoff_hz=LOWPASS_HZ):
    """H1 and H2 of a walk's truth (T, >= 20, 2) under the tracker's pixel noise on its 20 body points, seeds
    seed0 .. seed0 + seeds - 1 -> per-seed values and their summary (share of seeds that pass H1 and H2)."""
    tr = CR.TrackerProcess() if tracker is None else tracker
    uv = np.asarray(uv_truth, dtype=np.float64)[:, :P.N_BODY20]
    T = uv.shape[0]
    clean = score_walk(uv, uv, fps, cutoff_hz)
    h1, h2, ok1, ok2 = [], [], [], []
    for s in range(seeds):
        est = uv + _noise(tr, (T, P.N_BODY20, 2), seed0 + s)
        sc = score_walk(uv, est, fps, cutoff_hz)
        s2 = h2_summary(sc["ic_errors_frames"], sc["ic_true"], sc["ic_unmatched"])
        h1.append(sc["rmse_pooled_deg"])
        h2.append(s2)
        ok1.append(sc["rmse_pooled_deg"] <= H1_PASS_DEG)
        ok2.append(h2_pass(s2))
    h1 = np.array(h1)
    med = np.array([s["median_frames"] if s["median_frames"] is not None else np.nan for s in h2])
    p90 = np.array([s["p90_frames"] if s["p90_frames"] is not None else np.nan for s in h2])
    un = np.array([s["unmatched_share"] if s["unmatched_share"] is not None else np.nan for s in h2])
    return {"fps": fps, "frames": T, "seeds": seeds, "lowpass_hz": cutoff_hz,
            "tracker": {"noise_px": tr.noise_px, "nu": tr.nu}, "torso_px": torso_px(uv),
            "clean": {"rmse_pooled_deg": clean["rmse_pooled_deg"], "ic_true": clean["ic_true"],
                      "ic_unmatched": clean["ic_unmatched"], "ic_false": clean["ic_false"]},
            "h1_rmse_deg": {"median": float(np.median(h1)), "p95": float(np.percentile(h1, 95)),
                            "max": float(h1.max()), "share_pass": float(np.mean(ok1))},
            "h2": {"median_abs_err_frames_median": float(np.nanmedian(med)),
                   "p90_abs_err_frames_median": float(np.nanmedian(p90)),
                   "unmatched_share_median": float(np.nanmedian(un)), "unmatched_share_max": float(np.nanmax(un)),
                   "share_pass": float(np.mean(ok2))}}
