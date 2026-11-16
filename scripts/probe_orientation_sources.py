"""Which real IMU sources carry a per-sensor orientation, how it is defined, and how stable its heading is
(weekend task ORI, the G1 orientation-input decision; docs/proposals/G1_ORIENTATION_INPUT_DRAFT.md).

Stages (--stage, default all):

records  The record index (mdfm.data.canonical.index(): manifest joined with the record index) for every
         source of the stage-1 pilot mixture (configs/stage1_pilot.yaml sources) and of the stage-2 table
         (reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/sources.csv). Rows kept: tier not EVAL or
         EXCLUDED_V1, not CODY, not lockbox, with a container. For IMU rows the container's meta.json and two
         small arrays per record (channel_mask, location_ids) are read, never the signal: records and persons
         per source and group, records whose magnetometer channels are present at one site or at every site,
         records with gyroscope channels at every site (what a six-axis attitude filter needs), and the union
         of record metadata keys (an orientation key would show there). Non-IMU rows are counted from the
         index only.

raw      Header lines (and, for sixmwt_lifespan, the field names of the MAT struct) of the raw files of the
         IMU sources, persons outside the lockbox only: a person in the lockbox (reports/lockbox/
         lockbox_persons.csv) or flagged lockbox in the index, or absent from the index, is skipped from the
         file name alone, before the file is opened. Per source: which orientation fields exist (Euler angles,
         quaternions, orientation increments, Noraxon course/pitch/roll, earth-frame free acceleration,
         magnetometer) and on how many non-lockbox trials and persons. weargait_pd: every per-trial CSV under
         'CSV files' folders; the canonical records are matched to their CSV by (subject_ns, key) as
         build_canonical_imu.py names them. clinical_gait_multipatho: the raw_data_<site>.txt and
         processed_data.txt headers and the 'sensor' field of meta.json. nonan_gaitprint(_young): the header of
         every trial CSV in each subject zip. sixmwt_lifespan: Data.RawData field names of each Data.mat.
         pads: the declared channels of each observation JSON.

heading  weargait_pd only, non-lockbox persons of both groups, trials SelfPace and Balance of the main CSV
         folders (default; --trials, --groups, --limit). The vendor orientation is the Xsens Euler triple per
         sensor, R = Rz(yaw) Ry(pitch) Rx(roll), sensor frame to the local earth frame (z up; verified by
         scripts/probe_weargait_sensor_frames.py against the accelerometer up vector). Two sensors sit on the
         same shank (L_LatShank and L_Ankle, R_LatShank and R_Ankle). For a rigid segment R_ankle(t) =
         R_shank(t) C with C constant; C is taken as the rotation nearest the mean of R_shank^T R_ankle over the
         trial, and D(t) = R_ankle(t) C^T R_shank(t)^T is the identity when both orientations agree.
         psi(t), the twist of D about the earth vertical (swing-twist decomposition), is the relative heading
         of the pair. It measures the heading error between two CO-LOCATED sensors only: a heading error
         common to both cancels in it, and two sensors a few cm apart on one shank see nearly the same
         magnetic disturbance, so psi is not a bound on the heading error between sensors on different
         segments (stage cross); soft-tissue motion between the two sensors adds to it. tilt(t) is the angle
         of D's vertical axis to the vertical. Per trial and side: RMS of psi about its median, 95th
         percentile of |psi - median|, least-squares slope of psi in deg/min; psi split into a slow part
         (centred 1 s moving average) and a fast part: RMS of the fast part, and over consecutive 10 s chunks
         (the G1 window length; a tail chunk under 5 s is not scored) the median and 90th percentile of the
         chunk range (max - min) of the slow part and the median of its absolute change from the chunk's
         first to its last sample; RMS of tilt; RMS of psi and of tilt over the samples labelled Walk and
         Standing in GeneralEvent (at least 100 samples); median magnetometer norm of each sensor
         (normalized to the local field, 1 when undisturbed). Row 0 of a trial is dropped (an accumulation
         artefact, probe_weargait_sensor_frames.py). Samples with a non-finite Euler value at either sensor
         are skipped.

cross    weargait_pd only, the same persons and trials as heading: the relative heading of sensors on
         DIFFERENT segments, the error a G1 window would carry between its sensors, measured in quiet stance
         only, where the segments hardly move relative to each other. Quiet bouts (quiet_bouts): the stance
         conditions of a Balance trial (GeneralEvent labels EO_* and EC_*, about 12 s each) and the
         Standing bouts of a SelfPace trial, each trimmed by TRIM_S at both ends. Pairs: the 15 pairs of the
         six G1 sites (Forehead, LowerBack, L_Wrist, R_Wrist, L_LatShank, R_LatShank) and, on the same
         samples, the two co-located shank pairs of stage heading. Per trial and pair, C is fitted over the
         trial's quiet samples and psi computed as in heading. Per bout: the slow part (centred 1 s moving
         average within the bout), its SD and range over chunks of at most 10 s (bouts or tails under 5 s
         not scored), the fast part's RMS over the quiet samples, and the bout median of psi. Between bouts
         (bout_pairs): the absolute change of the bout median from each EO_<stance> bout to the EC_<stance>
         bout after it, and from the first to the last Standing bout of a SelfPace trial, with the time
         between the bout centres. Posture changes (sway, head turns, arm and foot moves) add to every
         number, so they are upper bounds on the sensor error for those pairs; walking through a
         non-uniform field is not covered (quiet stance only). Per group and trial, site_decomposition fits
         per-site levels v_i >= 0 with v_i + v_j = y_ij (y the squared median level of each pair) by
         non-negative least squares, the model of independent per-sensor errors, with its relative
         residual as the test of that model; for all six sites and for the four sites without the wrists.

cost     The wall time of the phases of the reference stage-1 run from its training log (default
         cache/runs/phaseA_root_full/log.jsonl): phase A from the first start event to the phase A end event,
         phase B from the last resume event before the phase B end event to that event; hours per 1,000
         steps of each phase (steps from the phase end events), and the hours of a paired short check (two
         phase A runs of SHORT_A_STEPS steps each at the reference phase A rate). G1 dev decode: the wall
         time of v1's full G1 dev decode (G1_DECODE_JSON, timing_s.all: calibration and dev sets, 20
         refinement steps) times the decodes of the short check (SHORT_CHECK_DECODES) and of the full run
         (FULL_RUN_DECODES), at v1's token count (a decode with the stream reads more tokens: not measured).
         Tokens: the 10 s window tokens of the IMU layout families (mdfm.factory.layouts.token_count; for
         random_sites the expectation over its uniform site count), now and with an orientation stream of
         one channel group per site (+P tokens per site, P = n_patches(10)) and, as the upper count, one
         nuisance token per site as well; the synthetic mixture's token growth with the stream on
         ORI_FAMILIES (configs/stage1_pilot.yaml mixture.layouts weights, share present 1); the longest bucket
         row of that config, which a window may not exceed (data/sampler.py). Full-run hours: the reference
         total at the config's fixed token budget (same steps and tokens; fewer windows per step), and the
         reference total times the mixture growth if the windows per step were kept instead.

Writes <out> (default reports/orientation_sources/): records.csv, records_meta_keys.json, raw_sources.csv,
weargait_raw_trials.csv, heading_pairs.csv, heading_summary.csv, cross_pairs.csv, cross_summary.csv,
cross_sites.csv, cost.json, summary.json, summary.md, _provenance.json. Reads MoveAtlas read-only, writes
nothing there, no GPU, no network. Canonical records are admitted through mdfm.data.canonical.check, the
program's refusal path, before any of their arrays is read.
"""
import argparse
import csv
import datetime as dt
import glob
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LOCKBOX_CSV = os.path.join(TREE, "reports", "lockbox", "lockbox_persons.csv")
STAGE2_SOURCES = os.path.join(TREE, "reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06", "sources.csv")
STAGE1_CONFIG = os.path.join(TREE, "configs", "stage1_pilot.yaml")
RUN_LOG = os.path.join(TREE, "cache", "runs", "phaseA_root_full", "log.jsonl")

SITES14 = ["head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R",
           "shank_L", "shank_R", "ankle_L", "ankle_R", "foot_L", "foot_R", "pocket"]
G1_SIX = ("head", "lower_back", "wrist_L", "wrist_R", "shank_L", "shank_R")
WEARGAIT_LOCMAP = {"Forehead": "head", "Xiphoid": "trunk", "LowerBack": "lower_back",
                   "L_Wrist": "wrist_L", "R_Wrist": "wrist_R",
                   "L_MidLatThigh": "thigh_L", "R_MidLatThigh": "thigh_R",
                   "L_LatShank": "shank_L", "R_LatShank": "shank_R",
                   "L_Ankle": "ankle_L", "R_Ankle": "ankle_R",
                   "L_DorsalFoot": "foot_L", "R_DorsalFoot": "foot_R"}
PAIRS = {"L": ("L_LatShank", "L_Ankle"), "R": ("R_LatShank", "R_Ankle")}
CHUNK_S = 10.0           # the G1 window length
MIN_CHUNK_S = 5.0        # a chunk shorter than this (the trial's tail) is not scored
MIN_PAIR_SAMPLES = 1000  # 10 s at 100 Hz
SMOOTH_S = 1.0           # slow part of psi: centred moving average over 1 s
SHORT_A_STEPS = 4000     # [cfg] steps of each run of the paired short check (cost stage)
FS_WEARGAIT = 100.0
# stage cross: the six G1 sites by their Xsens prefix, canonical G1 order
CROSS_SITES = {"head": "Forehead", "lower_back": "LowerBack", "wrist_L": "L_Wrist", "wrist_R": "R_Wrist",
               "shank_L": "L_LatShank", "shank_R": "R_LatShank"}
COLOC_PAIRS = {"shank_L~ankle_L": ("L_LatShank", "L_Ankle"), "shank_R~ankle_R": ("R_LatShank", "R_Ankle")}
NO_WRISTS = ("head", "lower_back", "shank_L", "shank_R")
TRIM_S = 0.5             # [cfg] cut from both ends of a quiet bout (the transition into and out of the stance)
MIN_BETWEEN_S = 1.0      # [cfg] a bout shorter than this (after the trim) is not used for the between-bout change
CROSS_STATS = ("slow_sd_deg", "slow_range_deg", "fast_rms_deg", "between_abs_change_deg")
# stage cost
G1_DECODE_JSON = os.path.join(TREE, "reports", "gates", "g1", "dev_phaseA_root_full_ckpt_B_032044", "g1.json")
SHORT_CHECK_DECODES = ("model without the stream: G1a", "model with the stream: G1a",
                       "model with the stream: G1a-ori", "model with the stream: G1a-ori-exact")
FULL_RUN_DECODES = ("G1a", "G1a-ori", "G1a-ori-exact")
ORI_FAMILIES = ("xsens13", "xsens13_insoles", "g1_six", "wrist_L", "random_sites")   # [cfg] layouts with the stream

# orientation field patterns of a header (case-sensitive column names of the vendors)
ORI_PATTERNS = {
    "euler": re.compile(r"(^|_)(Roll|Pitch|Yaw)$"),
    "quaternion": re.compile(r"(?i)(quat|_q[0-3]$|^q[wxyz0-3]$)"),
    "orientation_increment": re.compile(r"OriInc_q[0-3]$"),
    "rotation_matrix": re.compile(r"(?i)(^|_)(Mat|RotMat|DCM)(_|$)"),
    "course_pitch_roll": re.compile(r" (course|pitch|roll) \(deg\)$"),
    "free_acc_earth": re.compile(r"FreeAcc_"),
    "magnetometer": re.compile(r"(^|_)Mag_[XYZ]$"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- pure helpers (tested)
def classify_columns(header):
    """{kind: sorted columns} of the orientation-related columns of a header (ORI_PATTERNS). The
    orientation increment columns are not counted as quaternions (they are a per-sample delta)."""
    out = {}
    for col in header:
        col = col.strip()
        for kind, pat in ORI_PATTERNS.items():
            if kind == "quaternion" and ORI_PATTERNS["orientation_increment"].search(col):
                continue
            if pat.search(col):
                out.setdefault(kind, []).append(col)
    return {k: sorted(v) for k, v in out.items()}


def weargait_sites(header, suffixes):
    """Canonical sites whose raw prefix carries every <prefix>_<suffix> column of the header."""
    hs = set(h.strip() for h in header)
    return [site for pre, site in WEARGAIT_LOCMAP.items() if all(f"{pre}_{s}" in hs for s in suffixes)]


def nonan_orientation_segments(header):
    """Segments of a NONAN GaitPrint header with all three '<seg> course/pitch/roll (deg)' columns."""
    segs = {}
    for col in header:
        m = re.match(r"^(.+?) (course|pitch|roll) \(deg\)$", col.strip())
        if m:
            segs.setdefault(m.group(1), set()).add(m.group(2))
    return sorted(s for s, k in segs.items() if k == {"course", "pitch", "roll"})


def euler_to_R(roll_deg, pitch_deg, yaw_deg):
    """Xsens Euler angles (deg) to sensor-to-earth rotations (N, 3, 3): R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (np.deg2rad(np.asarray(v, dtype=np.float64)) for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(r.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def Rz(theta_rad):
    t = np.asarray(theta_rad, dtype=np.float64)
    R = np.zeros(t.shape + (3, 3))
    R[..., 0, 0] = np.cos(t)
    R[..., 0, 1] = -np.sin(t)
    R[..., 1, 0] = np.sin(t)
    R[..., 1, 1] = np.cos(t)
    R[..., 2, 2] = 1.0
    return R


def nearest_rotation(M):
    """The rotation nearest a 3 x 3 matrix in the Frobenius norm (SVD projection, det +1)."""
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def twist_about_z(D):
    """Twist angle (rad) of rotations D (N, 3, 3) about the z axis (swing-twist decomposition) and the
    swing angle (rad) of D's z axis away from z."""
    tr = D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]
    # twist = 2 atan2(qz, qw), with qw = sqrt(1 + tr) / 2 and qz = (D10 - D01) / (4 qw), that is
    # 2 atan2(D10 - D01, 1 + tr) (qw > 0); a pure rotation about z by theta gives theta
    twist = 2.0 * np.arctan2(D[..., 1, 0] - D[..., 0, 1], 1.0 + tr)
    swing = np.arccos(np.clip(D[..., 2, 2], -1.0, 1.0))
    return twist, swing


def pair_discrepancy(R_ref, R_other):
    """psi (rad, unwrapped) and tilt (rad) of D(t) = R_other C^T R_ref^T for two sensors on one rigid
    segment, C the rotation nearest the mean of R_ref^T R_other (module docstring)."""
    M = np.einsum("nji,njk->nik", R_ref, R_other)
    C = nearest_rotation(M.mean(0))
    D = np.einsum("nij,kj,nlk->nil", R_other, C, R_ref)
    psi, tilt = twist_about_z(D)
    return np.unwrap(psi), tilt


def moving_average(x, n):
    """Centred moving average over n samples (at most len(x)), normalized at the edges."""
    n = max(1, min(int(n), len(x)))
    k = np.ones(n)
    return np.convolve(x, k, mode="same") / np.convolve(np.ones(len(x)), k, mode="same")


def heading_stats(t_s, psi_rad, tilt_rad, chunk_s=CHUNK_S, min_chunk_s=MIN_CHUNK_S, smooth_s=SMOOTH_S,
                  fs=FS_WEARGAIT):
    """Per-trial statistics of a pair (module docstring), in degrees and deg/min. psi is split into a
    slow part (centred moving average over smooth_s) and a fast part (the rest): hf_rms is the RMS of the
    fast part; per 10 s chunk, lf_range is the range of the slow part within the chunk and lf_net the
    absolute change of the slow part from the chunk's first to its last sample."""
    psi = np.rad2deg(psi_rad)
    med = float(np.median(psi))
    dev = psi - med
    slow = moving_average(psi, smooth_s * fs)
    tm = t_s / 60.0
    slope = float(np.polyfit(tm, psi, 1)[0]) if len(psi) > 2 and np.ptp(tm) > 0 else float("nan")
    ranges, nets = [], []
    t0 = float(t_s[0])
    k = np.floor((t_s - t0) / chunk_s).astype(np.int64)
    for c in np.unique(k):
        sel = k == c
        span = float(t_s[sel][-1] - t_s[sel][0])
        if span < min_chunk_s:
            continue
        ranges.append(float(np.ptp(slow[sel])))
        nets.append(float(abs(slow[sel][-1] - slow[sel][0])))
    return {"duration_min": float((t_s[-1] - t_s[0]) / 60.0), "n_samples": int(len(psi)),
            "psi_rms_deg": float(np.sqrt(np.mean(dev ** 2))),
            "psi_p95_abs_deg": float(np.percentile(np.abs(dev), 95)),
            "psi_hf_rms_deg": float(np.sqrt(np.mean((psi - slow) ** 2))),
            "psi_slope_deg_min": slope,
            "n_chunks": len(ranges),
            "chunk_lf_range_median_deg": float(np.median(ranges)) if ranges else float("nan"),
            "chunk_lf_range_p90_deg": float(np.percentile(ranges, 90)) if ranges else float("nan"),
            "chunk_lf_net_median_deg": float(np.median(nets)) if nets else float("nan"),
            "tilt_rms_deg": float(np.sqrt(np.mean(np.rad2deg(tilt_rad) ** 2)))}


def phase_hours(events):
    """Phase A and B wall hours from the training log events (cost stage rule)."""
    def ts(e):
        return dt.datetime.fromisoformat(e["time"])
    starts = [e for e in events if e.get("event") == "start"]
    ends = {e.get("phase"): e for e in events if e.get("event") == "phase_end"}
    out = {}
    if starts and "A" in ends:
        out["phase_A_h"] = (ts(ends["A"]) - ts(starts[0])).total_seconds() / 3600.0
        out["phase_A_start"], out["phase_A_end"] = starts[0]["time"], ends["A"]["time"]
    if "B" in ends:
        resumes = [e for e in events if e.get("event") == "resume" and ts(e) < ts(ends["B"])]
        if resumes:
            out["phase_B_h"] = (ts(ends["B"]) - ts(resumes[-1])).total_seconds() / 3600.0
            out["phase_B_start"], out["phase_B_end"] = resumes[-1]["time"], ends["B"]["time"]
    if "phase_A_h" in out and "phase_B_h" in out:
        out["total_h"] = out["phase_A_h"] + out["phase_B_h"]
        out["phase_A_share"] = out["phase_A_h"] / out["total_h"]
    return out


def is_quiet(label):
    """A quiet-stance label of GeneralEvent: Standing, or a Balance stance condition (EO_* or EC_*)."""
    return label == "Standing" or bool(re.match(r"^E[OC]_", label))


def quiet_bouts(labels, fs=FS_WEARGAIT, trim_s=TRIM_S):
    """[(label, start, stop)] of the quiet bouts of a trial's GeneralEvent labels: runs of one quiet label,
    sample indices (stop exclusive) trimmed by trim_s at both ends; a bout the trim empties is dropped."""
    lab = np.asarray(labels).astype(str)
    if not len(lab):
        return []
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    k = int(round(trim_s * fs))
    out = []
    for a, b in zip(np.r_[0, ch], np.r_[ch, len(lab)]):
        if is_quiet(lab[a]) and b - k > a + k:
            out.append((str(lab[a]), int(a + k), int(b - k)))
    return out


def bout_pairs(bouts, fs=FS_WEARGAIT, min_s=MIN_BETWEEN_S):
    """Index pairs (i, j) of bouts compared for the between-bout change: each EO_<stance> bout with the first
    EC_<stance> bout after it, and the first and the last Standing bout when there are two or more. A bout
    shorter than min_s is not paired."""
    ok = [(b[2] - b[1]) / fs >= min_s for b in bouts]
    pairs = []
    for i, (lab, _, _) in enumerate(bouts):
        if ok[i] and lab.startswith("EO_"):
            for j in range(i + 1, len(bouts)):
                if ok[j] and bouts[j][0] == "EC_" + lab[3:]:
                    pairs.append((i, j))
                    break
    st = [i for i, b in enumerate(bouts) if b[0] == "Standing" and ok[i]]
    if len(st) >= 2:
        pairs.append((st[0], st[-1]))
    return pairs


def quiet_pair_stats(psi_deg, spans, fs=FS_WEARGAIT, chunk_s=CHUNK_S, min_chunk_s=MIN_CHUNK_S,
                     smooth_s=SMOOTH_S):
    """Statistics of a pair's relative heading psi_deg over quiet samples concatenated bout after bout;
    spans [(start, stop)] of each bout in that array. Per bout the slow part is the centred moving average
    over smooth_s within the bout and the fast part the rest; chunks of at most chunk_s from the bout start
    (a chunk under min_chunk_s is not scored) give the SD and the range of the slow part. Returns the
    medians over chunks, the fast RMS over all quiet samples, and the bout medians of psi."""
    sds, rngs, fast, meds = [], [], [], []
    n = int(round(chunk_s * fs))
    m = int(round(min_chunk_s * fs))
    for a, b in spans:
        x = np.asarray(psi_deg[a:b], dtype=float)
        meds.append(float(np.median(x)))
        slow = moving_average(x, smooth_s * fs)
        fast.append(x - slow)
        for c in range(0, len(x), n):
            s = slow[c:c + n]
            if len(s) >= m:
                sds.append(float(np.std(s)))
                rngs.append(float(np.ptp(s)))
    f = np.concatenate(fast) if fast else np.zeros(0)
    return {"n_chunks": len(sds),
            "slow_sd_deg": float(np.median(sds)) if sds else float("nan"),
            "slow_range_deg": float(np.median(rngs)) if rngs else float("nan"),
            "fast_rms_deg": float(np.sqrt(np.mean(f ** 2))) if len(f) else float("nan"),
            "bout_medians_deg": meds}


def site_decomposition(levels, sites):
    """Per-site levels from pair levels under independent per-sensor errors: v_i >= 0 with v_i + v_j = y_ij,
    y_ij = levels[(i, j)] ** 2, by non-negative least squares. Returns ({site: sqrt(v_i)}, relative
    residual ||A v - y|| / ||y||, pairs used). Pairs with a non-finite level are left out."""
    from scipy.optimize import nnls
    sites = list(sites)
    rows, y = [], []
    for (a, b), lv in levels.items():
        if a in sites and b in sites and lv is not None and np.isfinite(lv):
            r = np.zeros(len(sites))
            r[sites.index(a)] = r[sites.index(b)] = 1.0
            rows.append(r)
            y.append(float(lv) ** 2)
    if len(rows) < len(sites):
        return {s: float("nan") for s in sites}, float("nan"), len(rows)
    A, y = np.array(rows), np.array(y)
    v, _ = nnls(A, y)
    res = float(np.linalg.norm(A @ v - y) / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else 0.0
    return {s: float(np.sqrt(x)) for s, x in zip(sites, v)}, res, len(rows)


def layout_tokens(layout, P, L):
    """(10 s window tokens, IMU sites) of a layout family: token_count for a concrete layout; for a layout
    that draws k sites uniformly (random_sites), the expectation over k of the same count."""
    imu = [s for s in layout.streams if s.role == "input" and s.modality == "imu"]
    if layout.k_sites is None:
        return float(L.token_count(layout)), float(sum(len(s.sites) for s in imu))
    lo, hi = layout.k_sites
    k = (lo + hi) / 2.0
    s = imu[0]
    g, nuis = L._groups("imu", s.chan_mask)
    per_site = (g * P + nuis) / float(len(s.sites))
    return float(P + 1 + k * per_site), k


def token_growth(tokens, mixture, families, P):
    """Window tokens with an orientation stream of one channel group per IMU site: per family (now,
    with +P per site, with +P+1 per site) and the mixture's growth over all its families, the stream on
    `families` only. tokens: {family: (tokens now, IMU sites)}; mixture: {family: weight}."""
    fam = {}
    num_lo = num_hi = den = 0.0
    for f, w in mixture.items():
        if f not in tokens:
            continue
        t, k = tokens[f]
        lo = t + (k * P if f in families else 0.0)
        hi = t + (k * (P + 1) if f in families else 0.0)
        fam[f] = {"tokens_now": round(t, 1), "sites": k, "tokens_ori": round(lo, 1), "tokens_ori_upper": round(hi, 1)}
        den += w * t
        num_lo += w * lo
        num_hi += w * hi
    return fam, (num_lo / den if den else float("nan")), (num_hi / den if den else float("nan"))


def quant(vals, ps=(10, 50, 90)):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return [None] * len(ps)
    return [round(float(np.percentile(v, p)), 3) for p in ps]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), int(len(x))

    def rank(a):
        o = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[o] = np.arange(len(a))
        # average ties
        for v in np.unique(a):
            m = a == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(x))


# ---------------------------------------------------------------- lockbox and index
def read_lockbox(path=LOCKBOX_CSV):
    """{source: set of person_id} of the lockbox file."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["source"], set()).add(r["person_id"].strip())
    return out


def base_id(pid):
    return re.sub(r"\s*\(control\)\s*$", "", pid).strip()


class PersonGate:
    """Decides from a subject id alone whether a raw file may be opened: the subject must be in the index
    for the source, flagged non-lockbox on every row, outside the lockbox file (with and without the
    ' (control)' suffix), and not CODY or EVAL."""

    def __init__(self, allowed, lockbox_ids):
        self.allowed = set(allowed)
        self.lockbox = set(lockbox_ids)
        self.refused = set()

    @classmethod
    def from_index(cls, idx, source, lockbox):
        rows = idx[idx["source"] == source]
        bad = set(rows.loc[rows["lockbox"] | rows["cody"] | (rows["tier"].isin(["EVAL", "EXCLUDED_V1"])),
                           "subject_ns"])
        good = set(rows["subject_ns"]) - bad
        lb = set(lockbox.get(source, set())) | bad
        return cls(good, lb)

    def ok(self, subject_ns):
        cands = {subject_ns, base_id(subject_ns)}
        if cands & self.lockbox or subject_ns not in self.allowed:
            self.refused.add(subject_ns)
            return False
        return True


# ---------------------------------------------------------------- stage: records
def stage1_sources(path=STAGE1_CONFIG):
    """The 'sources:' list of the stage-1 pilot config (read as text, no YAML dependency)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*sources:\s*\[(.*)\]", line)
            if m:
                return [s.strip() for s in m.group(1).split(",") if s.strip()]
    return []


def stage2_rows(path=STAGE2_SOURCES):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stage_records(out):
    from mdfm.data import canonical as C
    idx = C.index()
    s1 = stage1_sources()
    s2 = stage2_rows()
    s2_scope = {(r["source"], r["group"]): r["stage2_scope"] for r in s2}
    s2_sources = sorted({r["source"] for r in s2})
    wanted = sorted(set(s1) | set(s2_sources))
    keep = (idx["source"].isin(wanted) & (idx["container"] != "") & ~idx["lockbox"] & ~idx["cody"]
            & ~idx["tier"].isin(["EVAL", "EXCLUDED_V1"]))
    df = idx[keep].copy()
    log(f"records: {len(df)} non-lockbox rows of {len(wanted)} sources")
    rows, meta_keys = [], {}
    n_refused_by_check = 0
    for (src, mod), g in df.groupby(["source", "modality"], sort=True):
        for grp, gg in g.groupby("group", sort=True):
            rec = {"source": src, "modality": mod, "group": grp,
                   "stage1": int(src in s1), "stage2_scope": s2_scope.get((src, grp), ""),
                   "records": int(len(gg.drop_duplicates(["container", "record"]))),
                   "persons": int(gg["person_id"].nunique())}
            if mod == "imu":
                n_mag_any = n_mag_all = n_gyro_all = 0
                site_n, site_mag = {}, {}
                keys = meta_keys.setdefault(src, set())
                gg = gg.drop_duplicates(["container", "record"]).sort_values(["container", "record"])
                for cont, gc in gg.groupby("container", sort=False):
                    admitted = []
                    for row in gc.to_dict("records"):
                        try:
                            C.check(row)        # the program's refusal path, before anything is read
                        except C.Refused:
                            n_refused_by_check += 1
                            continue
                        admitted.append(row["record"])
                    if not admitted:
                        continue
                    with zipfile.ZipFile(C.container_path(cont)) as zf:
                        header = C.read_header(cont, zf)
                        for r in admitted:
                            meta = C.record_meta(header, r)
                            keys.update(meta.keys())
                            pre = C.member_prefix(r) if header.get("kind") != "record" else ""
                            def member(name):
                                v = meta.get(name)
                                return v[C.TOKEN] if isinstance(v, dict) and C.TOKEN in v else name
                            cm = C.read_npy(zf, f"{pre}{member('channel_mask')}.npy").astype(bool)
                            loc = C.read_npy(zf, f"{pre}{member('location_ids')}.npy").astype(int)
                            mag = cm[:, 6:9].all(1)
                            n_mag_any += int(mag.any())
                            n_mag_all += int(mag.all())
                            n_gyro_all += int(cm[:, 3:6].all(1).all())
                            for li, has in zip(loc.tolist(), mag.tolist()):
                                s = SITES14[li]
                                site_n[s] = site_n.get(s, 0) + 1
                                site_mag[s] = site_mag.get(s, 0) + int(has)
                rec.update(mag_records_any_site=n_mag_any, mag_records_every_site=n_mag_all,
                           gyro_records_every_site=n_gyro_all,
                           sites="|".join(f"{s}:{site_n[s]}" for s in SITES14 if s in site_n),
                           mag_sites="|".join(f"{s}:{site_mag[s]}" for s in SITES14 if site_mag.get(s)))
            rows.append(rec)
    cols = ["source", "modality", "group", "stage1", "stage2_scope", "records", "persons",
            "mag_records_any_site", "mag_records_every_site", "gyro_records_every_site", "sites", "mag_sites"]
    write_csv(os.path.join(out, "records.csv"), rows, cols)
    with open(os.path.join(out, "records_meta_keys.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: sorted(v) for k, v in sorted(meta_keys.items())}, f, indent=1)
    ori_keys = {k: sorted(x for x in v if re.search(r"(?i)orient|quat|euler|attitude|heading", x))
                for k, v in meta_keys.items()}
    return {"rows": rows, "orientation_meta_keys": ori_keys, "stage1_sources": s1,
            "stage2_sources": s2_sources, "imu_rows_refused_by_check": n_refused_by_check}, idx


# ---------------------------------------------------------------- stage: raw
def weargait_csv_files(root):
    pat = os.path.join(root, "datasets", "imu", "weargait_pd", "raw", "**", "CSV files", "*.csv")
    out = []
    for f in glob.glob(pat, recursive=True):
        b = os.path.basename(f)
        if b.startswith("._") or "Demographic" in b or "Metrics" in b or b == "manifest.csv":
            continue
        out.append(f.replace("\\", "/"))
    return sorted(out)


def weargait_subject_key(path):
    parts = os.path.basename(path)[:-4].split("_")
    return f"weargait:{parts[0]}", "_".join(parts[1:]) or "rec"


def read_header_line(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        return next(csv.reader([fh.readline()]))


def raw_weargait(root, gate, idx_keys):
    """Header census of the weargait_pd CSVs of non-lockbox persons. idx_keys: set of (subject_ns, key) of
    the non-lockbox canonical weargait records."""
    files = weargait_csv_files(root)
    trials, seen_keys = [], set()
    for f in files:
        subj, key = weargait_subject_key(f)
        if not gate.ok(subj):
            continue
        header = read_header_line(f)
        kinds = classify_columns(header)
        eul = weargait_sites(header, ("Roll", "Pitch", "Yaw"))
        mag = weargait_sites(header, ("Mag_X", "Mag_Y", "Mag_Z"))
        acc = weargait_sites(header, ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"))
        group = "control" if "CONTROL PARTICIPANTS" in f else ("pd" if "PD PARTICIPANTS" in f else "other")
        folder = os.path.relpath(os.path.dirname(f), os.path.join(root, "datasets", "imu", "weargait_pd", "raw"))
        matched = (subj, key) in idx_keys
        seen_keys.add((subj, key))
        trials.append({"group": group, "subject_ns": subj, "key": key, "folder": folder.replace("\\", "/"),
                       "imu_sites": len(acc), "euler_sites": len(eul), "mag_sites": len(mag),
                       "euler_every_imu_site": int(bool(acc) and set(acc) <= set(eul)),
                       "euler_g1_six": int(set(G1_SIX) <= set(eul)),
                       "oriinc_sites": len(weargait_sites(header, ("OriInc_q0", "OriInc_q1", "OriInc_q2", "OriInc_q3"))),
                       "freeacc_sites": len(weargait_sites(header, ("FreeAcc_E", "FreeAcc_N", "FreeAcc_U"))),
                       "quaternion_columns": len(kinds.get("quaternion", [])),
                       "matches_record": int(matched)})
    rec_with = sum(1 for k in idx_keys if any(t["subject_ns"] == k[0] and t["key"] == k[1]
                                                and t["euler_every_imu_site"] for t in trials))
    return trials, {"files_listed": len(files), "records_nonlockbox": len(idx_keys),
                    "records_with_euler_csv": rec_with,
                    "records_without_csv": len(set(idx_keys) - seen_keys)}


def raw_clinical_gait(root, gate):
    z = zipfile.ZipFile(os.path.join(root, "datasets", "imu", "clinical_gait_multipatho", "raw", "dataset.zip"))
    names = z.namelist()
    stems = {}
    for n in names:
        b = n.rsplit("/", 1)[-1]
        m = re.match(r"^([A-Za-z]+_\d+)_(\d+)_(raw_data_(HE|LB|LF|RF)\.txt|processed_data\.txt|meta\.json)$", b)
        if m:
            stems.setdefault(f"{m.group(1)}_{m.group(2)}", {})[m.group(3)] = n
    kinds_raw, kinds_proc, sensors = {}, {}, {}
    trials = persons = 0
    pset, mag_trials = set(), 0
    for stem, files in sorted(stems.items()):
        subj = "clinical_gait:" + "_".join(stem.split("_")[:2])
        if not gate.ok(subj):
            continue
        trials += 1
        pset.add(subj)
        all_mag = True
        for k, n in files.items():
            if k.startswith("raw_data_"):
                head = z.open(n).readline().decode("latin-1").rstrip("\r\n").split("\t")
                c = classify_columns(head)
                for kind in c:
                    kinds_raw[kind] = kinds_raw.get(kind, 0) + 1
                all_mag &= "magnetometer" in c
            elif k == "processed_data.txt":
                head = z.open(n).readline().decode("latin-1").rstrip("\r\n").split("\t")
                for kind in classify_columns(head):
                    kinds_proc[kind] = kinds_proc.get(kind, 0) + 1
            elif k == "meta.json":
                s = json.loads(z.read(n).decode("utf-8")).get("sensor", "")
                sensors[s] = sensors.get(s, 0) + 1
        mag_trials += int(all_mag and any(k.startswith("raw_data_") for k in files))
    persons = len(pset)
    return {"trials": trials, "persons": persons, "raw_file_kinds": kinds_raw,
            "processed_file_kinds": kinds_proc, "sensor_field": sensors,
            "trials_with_mag_every_raw_file": mag_trials}


def raw_nonan(root, source, gate):
    d = os.path.join(root, "datasets", "imu", source, "raw")
    zips = sorted(f for f in os.listdir(d) if re.fullmatch(r"S\d+\.zip", f))
    trials = 0
    pset, segs_count, kinds = set(), {}, {}
    trials_with = 0
    for zf_name in zips:
        subj = f"{source}:{zf_name[:-4]}"
        if not gate.ok(subj):
            continue
        with zipfile.ZipFile(os.path.join(d, zf_name)) as z:
            for n in sorted(x for x in z.namelist() if x.endswith(".csv") and "/._" not in "/" + x):
                with z.open(n) as fh:
                    header = io.TextIOWrapper(fh, encoding="utf-8", errors="replace").readline().rstrip().split(",")
                trials += 1
                pset.add(subj)
                segs = nonan_orientation_segments(header)
                for s in segs:
                    segs_count[s] = segs_count.get(s, 0) + 1
                trials_with += int(bool(segs))
                for k in classify_columns(header):
                    kinds[k] = kinds.get(k, 0) + 1
    return {"trials": trials, "persons": len(pset), "trials_with_course_pitch_roll": trials_with,
            "segments_with_course_pitch_roll": segs_count, "column_kinds": kinds}


def raw_sixmwt(root, gate):
    import scipy.io as sio
    zp = os.path.join(root, "datasets", "imu", "sixmwt_lifespan", "raw", "6MWT_IMU_Dataset.zip")
    fields, n, pset = {}, 0, set()
    with zipfile.ZipFile(zp) as z:
        for name in sorted(x for x in z.namelist() if x.endswith("Data.mat") and "/._" not in x):
            m = re.search(r"Subject[ _]*(\d+)", name)
            if not m:
                continue
            subj = f"sixmwt_lifespan:S{int(m.group(1)):02d}"
            if not gate.ok(subj):
                continue
            mat = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)
            raw = getattr(mat.get("Data"), "RawData", None)
            names = sorted(getattr(raw, "_fieldnames", []) or [])
            for f_ in names:
                fields[f_] = fields.get(f_, 0) + 1
            n += 1
            pset.add(subj)
    return {"subjects_read": n, "persons": len(pset), "rawdata_fields": fields}


def raw_pads(root, gate):
    d = os.path.join(root, "datasets", "imu", "pads", "raw", "movement")
    chans, devices, n = {}, {}, 0
    for f in sorted(glob.glob(os.path.join(d, "observation_*.json"))):
        b = os.path.basename(f)
        if b.startswith("._"):
            continue
        subj = "pads:" + b[len("observation_"):-5]
        if not gate.ok(subj):
            continue
        with open(f, encoding="utf-8") as fh:
            o = json.load(fh)
        n += 1
        devices[o.get("device_id", "")] = devices.get(o.get("device_id", ""), 0) + 1
        for sess in o.get("session", []):
            for r in sess.get("records", []):
                for c in r.get("channels", []):
                    chans[c] = chans.get(c, 0) + 1
    return {"subjects_read": n, "device_id": devices, "declared_channels": chans}


def stage_raw(out, root, idx, lockbox):
    res = {}
    wg_gate = PersonGate.from_index(idx, "weargait_pd", lockbox)
    wg = idx[(idx["source"] == "weargait_pd") & (idx["modality"] == "imu") & ~idx["lockbox"]
             & (idx["container"] != "")]
    idx_keys = set(zip(wg["subject_ns"], wg["key"]))
    log("raw: weargait_pd headers")
    trials, wsum = raw_weargait(root, wg_gate, idx_keys)
    write_csv(os.path.join(out, "weargait_raw_trials.csv"), trials, list(trials[0].keys()) if trials else ["group"])
    wsum["trials_read"] = len(trials)
    wsum["persons_read"] = len({t["subject_ns"] for t in trials})
    for g in ("control", "pd", "other"):
        tg = [t for t in trials if t["group"] == g]
        wsum[f"{g}_trials"] = len(tg)
        wsum[f"{g}_persons"] = len({t["subject_ns"] for t in tg})
        wsum[f"{g}_trials_euler_every_site"] = sum(t["euler_every_imu_site"] for t in tg)
        wsum[f"{g}_persons_euler_every_site"] = len({t["subject_ns"] for t in tg if t["euler_every_imu_site"]})
        wsum[f"{g}_trials_euler_g1_six"] = sum(t["euler_g1_six"] for t in tg)
    wsum["trials_with_quaternion_columns"] = sum(1 for t in trials if t["quaternion_columns"])
    wsum["persons_refused_by_gate"] = len(wg_gate.refused)
    res["weargait_pd"] = wsum
    log("raw: clinical_gait_multipatho headers")
    g = PersonGate.from_index(idx, "clinical_gait_multipatho", lockbox)
    res["clinical_gait_multipatho"] = raw_clinical_gait(root, g)
    res["clinical_gait_multipatho"]["persons_refused_by_gate"] = len(g.refused)
    for src in ("nonan_gaitprint", "nonan_gaitprint_young"):
        log(f"raw: {src} headers")
        g = PersonGate.from_index(idx, src, lockbox)
        res[src] = raw_nonan(root, src, g)
        res[src]["persons_refused_by_gate"] = len(g.refused)
    log("raw: sixmwt_lifespan MAT fields")
    g = PersonGate.from_index(idx, "sixmwt_lifespan", lockbox)
    res["sixmwt_lifespan"] = raw_sixmwt(root, g)
    res["sixmwt_lifespan"]["persons_refused_by_gate"] = len(g.refused)
    log("raw: pads observation channels")
    g = PersonGate.from_index(idx, "pads", lockbox)
    res["pads"] = raw_pads(root, g)
    res["pads"]["persons_refused_by_gate"] = len(g.refused)
    rows = []
    for src, d in res.items():
        rows.append({"source": src, "facts": json.dumps(d, sort_keys=True)})
    write_csv(os.path.join(out, "raw_sources.csv"), rows, ["source", "facts"])
    return res


# ---------------------------------------------------------------- stage: heading
def load_pair_trial(path):
    import pandas as pd
    header = read_header_line(path)
    hs = set(header)
    want = ["Time"] + (["GeneralEvent"] if "GeneralEvent" in hs else [])
    for a, b in PAIRS.values():
        for p in (a, b):
            want += [f"{p}_{s}" for s in ("Roll", "Pitch", "Yaw", "Mag_X", "Mag_Y", "Mag_Z")]
    use = [c for c in want if c in hs]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)     # row 0: accumulation artefact
    t = pd.to_numeric(df["Time"].astype(str).str.replace("sec", "", regex=False).str.strip(), errors="coerce")
    t = t.to_numpy(dtype=float)
    if not np.isfinite(t).any():
        t = np.arange(len(df)) / FS_WEARGAIT
    return df, t


def heading_trial(df, t):
    import pandas as pd
    rows = []
    ev = df["GeneralEvent"].astype(str).to_numpy() if "GeneralEvent" in df.columns else None
    for side, (a, b) in PAIRS.items():
        cols = [f"{p}_{s}" for p in (a, b) for s in ("Roll", "Pitch", "Yaw")]
        if not all(c in df.columns for c in cols):
            continue
        E = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(E).all(1) & np.isfinite(t)
        if ok.sum() < MIN_PAIR_SAMPLES:
            continue
        Rs = euler_to_R(E[ok, 0], E[ok, 1], E[ok, 2])
        Ra = euler_to_R(E[ok, 3], E[ok, 4], E[ok, 5])
        psi, tilt = pair_discrepancy(Rs, Ra)
        st = heading_stats(t[ok], psi, tilt)
        st["side"] = side
        if ev is not None:
            e = ev[ok]
            psi_d = np.rad2deg(psi - np.median(psi))
            tilt_d = np.rad2deg(tilt)
            for lab, key in (("Walk", "walk"), ("Standing", "standing")):
                m = e == lab
                st[f"psi_rms_{key}_deg"] = float(np.sqrt(np.mean(psi_d[m] ** 2))) if m.sum() >= 100 else float("nan")
                st[f"tilt_rms_{key}_deg"] = float(np.sqrt(np.mean(tilt_d[m] ** 2))) if m.sum() >= 100 else float("nan")
        for p, key in ((a, "shank"), (b, "ankle")):
            mc = [f"{p}_Mag_{x}" for x in "XYZ"]
            if all(c in df.columns for c in mc):
                Mv = df[mc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                nrm = np.linalg.norm(Mv, axis=1)
                nrm = nrm[np.isfinite(nrm)]
                st[f"mag_norm_{key}"] = float(np.median(nrm)) if len(nrm) else float("nan")
        rows.append(st)
    return rows


def stage_heading(out, root, idx, lockbox, trials=("SelfPace", "Balance"), groups=("control", "pd"), limit=0):
    gate = PersonGate.from_index(idx, "weargait_pd", lockbox)
    v1 = os.path.join(root, "datasets", "imu", "weargait_pd", "raw", "Version 1")
    dirs = {"control": os.path.join(v1, "CONTROL PARTICIPANTS", "CSV files"),
            "pd": os.path.join(v1, "PD PARTICIPANTS", "CSV files")}
    rows = []
    for grp in groups:
        files = sorted(f for f in os.listdir(dirs[grp]) if f.endswith(".csv") and not f.startswith("._"))
        files = [f for f in files if weargait_subject_key(f)[1] in trials]
        persons_done = []
        for f in files:
            subj, key = weargait_subject_key(f)
            if not gate.ok(subj):
                continue
            if limit and subj not in persons_done and len(persons_done) >= limit:
                continue
            if subj not in persons_done:
                persons_done.append(subj)
            df, t = load_pair_trial(os.path.join(dirs[grp], f))
            for st in heading_trial(df, t):
                st.update(group=grp, subject_ns=subj, trial=key)
                rows.append(st)
        log(f"heading: {grp} {len(persons_done)} persons, {sum(1 for r in rows if r['group'] == grp)} pair rows")
    cols = ["group", "subject_ns", "trial", "side", "duration_min", "n_samples", "psi_rms_deg", "psi_p95_abs_deg",
            "psi_hf_rms_deg", "psi_slope_deg_min", "n_chunks", "chunk_lf_range_median_deg",
            "chunk_lf_range_p90_deg", "chunk_lf_net_median_deg", "tilt_rms_deg", "psi_rms_walk_deg", "psi_rms_standing_deg",
            "tilt_rms_walk_deg", "tilt_rms_standing_deg", "mag_norm_shank", "mag_norm_ankle"]
    write_csv(os.path.join(out, "heading_pairs.csv"), rows, cols)
    summ = []
    keys = ["psi_rms_deg", "psi_p95_abs_deg", "psi_hf_rms_deg", "chunk_lf_range_median_deg", "chunk_lf_range_p90_deg",
            "chunk_lf_net_median_deg", "tilt_rms_deg", "psi_rms_walk_deg", "psi_rms_standing_deg",
            "tilt_rms_walk_deg", "tilt_rms_standing_deg"]
    for grp in groups:
        for tr in trials:
            rr = [r for r in rows if r["group"] == grp and r["trial"] == tr]
            if not rr:
                continue
            s = {"group": grp, "trial": tr, "pair_rows": len(rr), "persons": len({r["subject_ns"] for r in rr}),
                 "duration_min_p50": quant([r["duration_min"] for r in rr])[1]}
            for k in keys:
                s[f"{k}_p10_p50_p90"] = "|".join(str(x) for x in quant([r.get(k, float("nan")) for r in rr]))
            s["abs_slope_deg_min_p10_p50_p90"] = "|".join(
                str(x) for x in quant([abs(r["psi_slope_deg_min"]) for r in rr]))
            magdev = [np.nanmean([abs(r.get("mag_norm_shank", np.nan) - 1), abs(r.get("mag_norm_ankle", np.nan) - 1)])
                      for r in rr]
            rho, n = spearman(magdev, [r["psi_rms_deg"] for r in rr])
            s["spearman_magdev_psi_rms"] = round(rho, 3) if np.isfinite(rho) else None
            s["spearman_n"] = n
            summ.append(s)
    write_csv(os.path.join(out, "heading_summary.csv"), summ, list(summ[0].keys()) if summ else ["group"])
    return {"rows": len(rows), "summary": summ, "persons_refused_by_gate": len(gate.refused)}


# ---------------------------------------------------------------- stage: cross
def cross_pair_names():
    """[(name, prefix_a, prefix_b, kind)]: the 15 pairs of the six G1 sites, then the co-located pairs."""
    sites = list(CROSS_SITES)
    out = [(f"{a}~{b}", CROSS_SITES[a], CROSS_SITES[b], "cross")
           for i, a in enumerate(sites) for b in sites[i + 1:]]
    out += [(n, a, b, "colocated") for n, (a, b) in COLOC_PAIRS.items()]
    return out


def load_cross_trial(path):
    """Time, GeneralEvent and the Euler triples of the six G1 sensors and the two ankle sensors of a trial;
    row 0 dropped. Returns (labels, {prefix: (N, 3) Euler deg}) or (None, None) without GeneralEvent."""
    import pandas as pd
    header = read_header_line(path)
    hs = set(header)
    if "GeneralEvent" not in hs:
        return None, None
    prefixes = sorted(set(CROSS_SITES.values()) | {p for ab in COLOC_PAIRS.values() for p in ab})
    want = ["GeneralEvent"] + [f"{p}_{s}" for p in prefixes for s in ("Roll", "Pitch", "Yaw")]
    df = pd.read_csv(path, usecols=[c for c in want if c in hs], low_memory=False).iloc[1:].reset_index(drop=True)
    eul = {}
    for p in prefixes:
        cols = [f"{p}_{s}" for s in ("Roll", "Pitch", "Yaw")]
        if all(c in df.columns for c in cols):
            eul[p] = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return df["GeneralEvent"].astype(str).to_numpy(), eul


def cross_trial(labels, eul, fs=FS_WEARGAIT):
    """Rows of stage cross for one trial (module docstring): one per pair with both sensors present."""
    bouts = quiet_bouts(labels, fs)
    if not bouts:
        return []
    finite = np.ones(len(labels), dtype=bool)
    for E in eul.values():
        finite &= np.isfinite(E).all(1)
    idx_parts, spans, kept, n = [], [], [], 0
    for lab, a, b in bouts:
        ii = np.arange(a, b)[finite[a:b]]
        if not len(ii):
            continue
        idx_parts.append(ii)
        spans.append((n, n + len(ii)))
        kept.append((lab, a, b))
        n += len(ii)
    if not n:
        return []
    ii = np.concatenate(idx_parts)
    R = {p: euler_to_R(E[ii, 0], E[ii, 1], E[ii, 2]) for p, E in eul.items()}
    centres = [(a + b) / 2.0 / fs for _, a, b in kept]
    pairs_ij = bout_pairs(kept, fs)
    rows = []
    for name, pa, pb, kind in cross_pair_names():
        if pa not in R or pb not in R:
            continue
        psi, tilt = pair_discrepancy(R[pa], R[pb])
        st = quiet_pair_stats(np.rad2deg(psi), spans, fs)
        meds = st.pop("bout_medians_deg")
        ch = [abs(meds[j] - meds[i]) for i, j in pairs_ij]
        dts = [centres[j] - centres[i] for i, j in pairs_ij]
        st.update(pair=name, kind=kind, n_bouts=len(spans), quiet_s=round(n / fs, 2),
                  between_n=len(ch),
                  between_abs_change_deg=float(np.median(ch)) if ch else float("nan"),
                  between_dt_s=float(np.median(dts)) if dts else float("nan"))
        rows.append(st)
    return rows


def stage_cross(out, root, idx, lockbox, trials=("SelfPace", "Balance"), groups=("control", "pd"), limit=0):
    gate = PersonGate.from_index(idx, "weargait_pd", lockbox)
    v1 = os.path.join(root, "datasets", "imu", "weargait_pd", "raw", "Version 1")
    dirs = {"control": os.path.join(v1, "CONTROL PARTICIPANTS", "CSV files"),
            "pd": os.path.join(v1, "PD PARTICIPANTS", "CSV files")}
    rows, no_event, no_quiet = [], 0, 0
    for grp in groups:
        files = sorted(f for f in os.listdir(dirs[grp]) if f.endswith(".csv") and not f.startswith("._"))
        files = [f for f in files if weargait_subject_key(f)[1] in trials]
        persons_done = []
        for f in files:
            subj, key = weargait_subject_key(f)
            if not gate.ok(subj):
                continue
            if limit and subj not in persons_done and len(persons_done) >= limit:
                continue
            if subj not in persons_done:
                persons_done.append(subj)
            labels, eul = load_cross_trial(os.path.join(dirs[grp], f))
            if labels is None:
                no_event += 1
                continue
            rr = cross_trial(labels, eul)
            no_quiet += int(not rr)
            for st in rr:
                st.update(group=grp, subject_ns=subj, trial=key)
                rows.append(st)
        log(f"cross: {grp} {len(persons_done)} persons, {sum(1 for r in rows if r['group'] == grp)} pair rows")
    cols = ["group", "subject_ns", "trial", "pair", "kind", "n_bouts", "quiet_s", "n_chunks", "slow_sd_deg",
            "slow_range_deg", "fast_rms_deg", "between_n", "between_abs_change_deg", "between_dt_s"]
    write_csv(os.path.join(out, "cross_pairs.csv"), rows, cols)
    summ, sites_rows = [], []
    names = [p[0] for p in cross_pair_names()]
    for grp in groups:
        for tr in trials:
            sel = [r for r in rows if r["group"] == grp and r["trial"] == tr]
            if not sel:
                continue
            med = {}
            for pn in names:
                rr = [r for r in sel if r["pair"] == pn]
                if not rr:
                    continue
                s = {"group": grp, "trial": tr, "pair": pn, "kind": rr[0]["kind"], "pair_rows": len(rr),
                     "persons": len({r["subject_ns"] for r in rr}),
                     "quiet_s_p50": quant([r["quiet_s"] for r in rr])[1],
                     "between_dt_s_p50": quant([r["between_dt_s"] for r in rr])[1],
                     "between_rows": sum(1 for r in rr if r["between_n"])}
                for k in CROSS_STATS:
                    q = quant([r[k] for r in rr])
                    s[f"{k}_p10_p50_p90"] = "|".join(str(x) for x in q)
                    med[(pn, k)] = q[1]
                summ.append(s)
            for k in CROSS_STATS:
                for variant, vs in (("six_sites", tuple(CROSS_SITES)), ("no_wrists", NO_WRISTS)):
                    lv = {}
                    for pn in names:
                        a, _, b = pn.partition("~")
                        if a in CROSS_SITES and b in CROSS_SITES and (pn, k) in med:
                            lv[(a, b)] = med[(pn, k)] if med[(pn, k)] is not None else float("nan")
                    lev, res, npairs = site_decomposition(lv, vs)
                    r = {"group": grp, "trial": tr, "statistic": k, "variant": variant, "pairs": npairs,
                         "relative_residual": round(res, 3) if np.isfinite(res) else None}
                    for site in CROSS_SITES:
                        r[site] = round(lev[site], 3) if site in lev and np.isfinite(lev[site]) else None
                    sites_rows.append(r)
    write_csv(os.path.join(out, "cross_summary.csv"), summ, list(summ[0].keys()) if summ else ["group"])
    write_csv(os.path.join(out, "cross_sites.csv"), sites_rows,
              ["group", "trial", "statistic", "variant", "pairs", "relative_residual"] + list(CROSS_SITES))
    return {"rows": len(rows), "summary": summ, "sites": sites_rows, "trials_without_generalevent": no_event,
            "trials_without_quiet_bout": no_quiet, "persons_refused_by_gate": len(gate.refused)}


# ---------------------------------------------------------------- stage: cost
def stage1_mixture(path=STAGE1_CONFIG):
    """({family: weight} of mixture.layouts, [bucket row lengths] of mixture.rows) of the stage-1 config,
    read as text (no YAML dependency)."""
    mix, rows, inside = {}, [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*rows:\s*\[([^\]]*)\]", line)
            if m:
                rows = [int(x) for x in m.group(1).split(",") if x.strip()]
            if re.match(r"^\s*layouts:\s*(#.*)?$", line):
                inside = True
                continue
            if inside:
                m = re.match(r"^\s+([A-Za-z0-9_]+):\s*([0-9.eE+-]+)\s*(#.*)?$", line)
                if m:
                    mix[m.group(1)] = float(m.group(2))
                else:
                    inside = False
    return mix, rows


def stage_cost(out, run_log=RUN_LOG, decode_json=G1_DECODE_JSON):
    if not os.path.isfile(run_log):
        return {"run_log": run_log, "missing": True}
    events = []
    with open(run_log, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") in ("start", "resume", "phase_end"):
                events.append({k: d.get(k) for k in ("event", "phase", "step", "time")})
    res = phase_hours(events)
    steps = {e.get("phase"): e.get("step") for e in events if e.get("event") == "phase_end"}
    if steps.get("A") and "phase_A_h" in res:
        res["phase_A_steps"] = int(steps["A"])
        res["phase_A_h_per_1000_steps"] = res["phase_A_h"] / steps["A"] * 1000.0
        # a paired short check before a full run: two phase A runs of SHORT_A_STEPS (with and without the
        # orientation stream), at the reference phase A rate
        res["short_check_A_steps"] = SHORT_A_STEPS
        res["short_check_pair_h"] = 2 * SHORT_A_STEPS * res["phase_A_h"] / steps["A"]
    if steps.get("B") and "phase_B_h" in res:
        res["phase_B_steps"] = int(steps["B"])
        res["phase_B_h_per_1000_steps"] = res["phase_B_h"] / steps["B"] * 1000.0
    # G1 dev decodes (v1's full dev decode, calibration and dev sets)
    if os.path.isfile(decode_json):
        with open(decode_json, encoding="utf-8") as f:
            g1 = json.load(f)
        dec_h = float(g1["timing_s"]["all"]) / 3600.0
        res["g1_dev_decode_h"] = dec_h
        res["g1_dev_decode_source"] = os.path.relpath(decode_json, TREE).replace("\\", "/")
        res["g1_dev_decode_windows"] = {k: g1["sets"][k]["windows"] for k in ("dev", "calib")}
        res["short_check_decodes"] = list(SHORT_CHECK_DECODES)
        res["full_run_decodes"] = list(FULL_RUN_DECODES)
        if "short_check_pair_h" in res:
            res["short_check_decodes_h"] = len(SHORT_CHECK_DECODES) * dec_h
            res["short_check_total_h"] = res["short_check_pair_h"] + res["short_check_decodes_h"]
        if "total_h" in res:
            res["full_run_decodes_h"] = len(FULL_RUN_DECODES) * dec_h
    # tokens of a 10 s window with the orientation stream, and the full run's hours
    from mdfm.factory import layouts as L
    lays = L.load_layouts()
    P = L.n_patches(CHUNK_S)
    mixture, rows_cfg = stage1_mixture()
    tokens = {f: layout_tokens(lays[f], P, L) for f in mixture if f in lays}
    fam, g_lo, g_hi = token_growth(tokens, mixture, ORI_FAMILIES, P)
    longest = max(rows_cfg) if rows_cfg else None
    for f, d in fam.items():
        d["exceeds_longest_row_ori"] = bool(longest and d["tokens_ori"] > longest)
        d["exceeds_longest_row_ori_upper"] = bool(longest and d["tokens_ori_upper"] > longest)
    res["tokens"] = {"patches_per_window": P, "families": fam, "stream_on": list(ORI_FAMILIES),
                     "mixture_growth": g_lo, "mixture_growth_upper": g_hi, "longest_row": longest,
                     "rows_source": "configs/stage1_pilot.yaml mixture.rows; weights mixture.layouts"}
    if "total_h" in res:
        res["full_run_h_fixed_token_budget"] = res["total_h"]
        res["full_run_h_fixed_windows_upper"] = res["total_h"] * g_hi
        if "full_run_decodes_h" in res:
            res["full_run_total_h_range"] = [res["total_h"] + res["full_run_decodes_h"],
                                             res["total_h"] * g_hi + res["full_run_decodes_h"]]

    def rnd(v):
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, dict):
            return {k: rnd(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rnd(x) for x in v]
        return v
    res = {k: rnd(v) for k, v in res.items()}
    res["run_log"] = os.path.relpath(run_log, TREE).replace("\\", "/")
    with open(os.path.join(out, "cost.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    return res


# ---------------------------------------------------------------- output
def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def summary_md(s):
    L = ["# Orientation sources (task ORI)", "",
         "Generated by scripts/probe_orientation_sources.py; every number below is read from summary.json and the "
         "CSV and JSON files of this folder. MoveAtlas commit in _provenance.json. Definitions in the script "
         "docstring. Non-lockbox, non-EVAL, non-CODY persons only; lockbox persons' files were not opened.", ""]
    if "records" in s:
        L += ["## Canonical records (record index, non-lockbox)", "",
              "| source | modality | group | stage 1 | stage 2 scope | records | persons | magnetometer at one site | at every site | "
              "gyroscope at every site |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["records"]["rows"]:
            L.append(f"| {r['source']} | {r['modality']} | {r['group']} | {r['stage1']} | {r['stage2_scope']} | "
                     f"{r['records']} | {r['persons']} | {r.get('mag_records_any_site', '')} | "
                     f"{r.get('mag_records_every_site', '')} | {r.get('gyro_records_every_site', '')} |")
        L += ["", "Record metadata keys naming an orientation, per IMU source: "
              + json.dumps(s["records"]["orientation_meta_keys"], sort_keys=True), ""]
    if "raw" in s:
        L += ["## Raw files (headers only, non-lockbox persons)", ""]
        for src, d in s["raw"].items():
            L.append(f"- {src}: {json.dumps(d, sort_keys=True)}")
        L.append("")
    if "heading" in s:
        def cell(v):
            return str(v).replace("|", " / ")
        L += ["## weargait_pd: relative heading of two Xsens sensors on one shank", "",
              f"Pair rows: {s['heading']['rows']}.", "",
              "Each cell: p10 / p50 / p90 over the pair rows (one row per trial and side).", "",
              "| group | trial | pair rows | persons | duration min p50 | psi RMS deg | |psi| p95 deg | psi fast RMS deg | "
              "10 s chunk slow range median deg | 10 s chunk slow range p90 deg | 10 s chunk slow net change deg | "
              "|slope| deg/min | tilt RMS deg | psi RMS walk deg | psi RMS standing deg | tilt RMS walk deg | "
              "tilt RMS standing deg | Spearman(|mag-1|, psi RMS) (n) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["heading"]["summary"]:
            c = {k: cell(v) for k, v in r.items()}
            L.append(f"| {c['group']} | {c['trial']} | {c['pair_rows']} | {c['persons']} | {c['duration_min_p50']} | "
                     f"{c['psi_rms_deg_p10_p50_p90']} | {c['psi_p95_abs_deg_p10_p50_p90']} | "
                     f"{c['psi_hf_rms_deg_p10_p50_p90']} | "
                     f"{c['chunk_lf_range_median_deg_p10_p50_p90']} | {c['chunk_lf_range_p90_deg_p10_p50_p90']} | "
                     f"{c['chunk_lf_net_median_deg_p10_p50_p90']} | {c['abs_slope_deg_min_p10_p50_p90']} | {c['tilt_rms_deg_p10_p50_p90']} | "
                     f"{c['psi_rms_walk_deg_p10_p50_p90']} | {c['psi_rms_standing_deg_p10_p50_p90']} | "
                     f"{c.get('tilt_rms_walk_deg_p10_p50_p90', '')} | {c.get('tilt_rms_standing_deg_p10_p50_p90', '')} | "
                     f"{c['spearman_magdev_psi_rms']} ({c['spearman_n']}) |")
        L.append("")
    if "cross" in s:
        def cell(v):
            return str(v).replace("|", " / ")
        L += ["## weargait_pd: relative heading of sensors on different segments, quiet stance only", "",
              f"Pair rows: {s['cross']['rows']}. Each cell: p10 / p50 / p90 over the pair rows (one row per trial "
              "and pair). Upper bounds of the sensor error for these pairs (posture changes add to them).", "",
              "| group | trial | pair | kind | pair rows | persons | quiet s p50 | slow SD deg | slow range deg | "
              "fast RMS deg | between-bout change deg | rows with a bout pair | between dt s p50 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["cross"]["summary"]:
            c = {k: cell(v) for k, v in r.items()}
            L.append(f"| {c['group']} | {c['trial']} | {c['pair']} | {c['kind']} | {c['pair_rows']} | {c['persons']} | "
                     f"{c['quiet_s_p50']} | {c['slow_sd_deg_p10_p50_p90']} | {c['slow_range_deg_p10_p50_p90']} | "
                     f"{c['fast_rms_deg_p10_p50_p90']} | {c['between_abs_change_deg_p10_p50_p90']} | "
                     f"{c['between_rows']} | {c['between_dt_s_p50']} |")
        L += ["", "Per-site levels (deg) under independent per-sensor errors, fitted on the pair medians "
              "(site_decomposition; relative residual = the misfit of that model):", "",
              "| group | trial | statistic | variant | pairs | relative residual | " + " | ".join(CROSS_SITES) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(CROSS_SITES)]
        for r in s["cross"]["sites"]:
            L.append(f"| {r['group']} | {r['trial']} | {r['statistic']} | {r['variant']} | {r['pairs']} | "
                     f"{r['relative_residual']} | " + " | ".join(str(r[x]) for x in CROSS_SITES) + " |")
        L.append("")
    if "cost" in s:
        L += ["## Reference stage-1 run (training log)", "", "cost.json: " + json.dumps(s["cost"], sort_keys=True), ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "orientation_sources"))
    ap.add_argument("--stage", default="all", choices=["all", "records", "raw", "heading", "cross", "cost"])
    ap.add_argument("--trials", default="SelfPace,Balance")
    ap.add_argument("--groups", default="control,pd")
    ap.add_argument("--limit", type=int, default=0, help="heading: first N persons per group (tests)")
    a = ap.parse_args()
    root = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(root, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {root}")
    os.environ["MOVEATLAS_ROOT"] = root
    os.makedirs(a.out, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from _common import write_provenance
    t0 = time.time()
    lockbox = read_lockbox()
    summ_path = os.path.join(a.out, "summary.json")
    s = {}
    if a.stage != "all" and os.path.isfile(summ_path):
        with open(summ_path, encoding="utf-8") as f:
            s = json.load(f)
    idx = None
    if a.stage in ("all", "records"):
        s["records"], idx = stage_records(a.out)
    if a.stage in ("all", "raw", "heading", "cross") and idx is None:
        from mdfm.data import canonical as C
        idx = C.index()
    if a.stage in ("all", "raw"):
        s["raw"] = stage_raw(a.out, root, idx, lockbox)
    if a.stage in ("all", "heading"):
        s["heading"] = stage_heading(a.out, root, idx, lockbox, tuple(a.trials.split(",")),
                                     tuple(a.groups.split(",")), a.limit)
        s["heading"]["limit"] = a.limit
    if a.stage in ("all", "cross"):
        s["cross"] = stage_cross(a.out, root, idx, lockbox, tuple(a.trials.split(",")),
                                 tuple(a.groups.split(",")), a.limit)
        s["cross"]["limit"] = a.limit
    if a.stage in ("all", "cost"):
        s["cost"] = stage_cost(a.out)
    with open(summ_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1, default=str)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summary_md(s))

    class _A:
        pass
    pa = _A()
    pa.moveatlas, pa.out = root, a.out
    # one runtime, measured here, in both the provenance and the log (the provenance write itself, a git
    # call, comes after it)
    runtime_s = round(time.time() - t0, 1)
    write_provenance(pa, {"stage": a.stage, "trials": a.trials, "groups": a.groups, "limit": a.limit,
                          "runtime_s": runtime_s})
    log(f"done in {runtime_s} s (runtime_s of _provenance.json, before its write)")


if __name__ == "__main__":
    main()
