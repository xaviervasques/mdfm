"""Sensefog LFP to EEG synchronisation anchor (task N0.3b; addendum gate G15, sensefog part).

Descriptive only: no coupling statistic is computed.

1. Inventory of every sensefog session folder: EEG (BrainVision, same amplifier as the EMG and the two
   IPG channels), LFP (Medtronic Percept BrainSenseTimeDomain, from the raw JSON or from the JSON the
   authors embedded in their _lfpalg.mat), IMU (APDM h5), rates, clocks, stimulation trains of the 2 Hz
   BrainSenseLfp stream, and the authors' hand-read alignment samples ('signalpoint' in
   SenseFOG-Documentation sub-XX-datafile.m).
2. Rebuilds the Percept time base from TicksInMses and GlobalPacketSizes; lost ticks become masked
   samples; a cross-check against the MoveAtlas processed LFP (length and masked count) is recorded.
3. Per session, the LFP to EEG offset (theta = LFP time minus EEG time for one event):
   a. the authors' anchor (stimulation-offset transition artefact), and an automatic re-read of the same
      artefact in both streams (this is NOT an independent anchor: it checks the hand reading only; the LFP side
      counts only where the step near the hand-read sample is clear of noise, step z >= LFP_EDGE_MIN_Z);
   b. the ECG anchor: R peaks in the EEG-file channel with the strongest ECG, the R-locked LFP template
      in a window at session start and one at session end; their relative shift gives the drift in ppm,
      the lag of the LFP template against the EEG ECG template gives an ECG-based offset at start and end;
   c. a second stimulation train where one exists.
   Controls: the same ECG analysis on a scalp EEG channel (EEG clock, true drift 0), agreement of the two
   LFP hemispheres, and a within-subject comparison of the ECG lag at the anchor (lead geometry fixed).
   The admissible-lag floor (4 ms plus the worst-case drift over the session) is reported per session.
4. The Percept artefact edge: rise time and onset-to-peak delay against the EEG edge.
Also: the IMU to EEG check from the APDM start and stop sync markers.

Reads the MoveAtlas tree read-only; writes reports/sync_sensefog/ only. Resumable: per-session results are
cached in reports/sync_sensefog/cache/ (use --force to recompute). Deterministic (fixed seeds).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import zlib

import h5py
import numpy as np
import scipy.io as sio
from scipy import interpolate, signal

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _common import parse_args, write_provenance  # noqa: E402

# ------------------------------------------------------------------------------------------
# method parameters (all reported in _provenance.json)
# ------------------------------------------------------------------------------------------
SEED = 20260928
ECG_BAND_HZ = (5.0, 40.0)            # QRS band, applied zero-phase to both streams
LAGS_MS = np.arange(-150, 151)       # R-locked template lags, 1 ms grid
SHIFT_MAX_MS = 30                    # search range of the window-vs-reference shift
ELL_MAX_MS = 60                      # search range of the LFP-vs-EEG ECG template lag
WIN_MAX_S = 60.0                     # start and end window length (shorter when the usable span < 180 s)
SLIDE_W_S, SLIDE_STEP_S = 30.0, 15.0
N_BOOT, N_BOOT_SLIDE, N_NULL = 300, 100, 200
P_NULL_MAX, BOOT_SD_MAX_MS, R_START_END_MIN, MIN_BEATS_WIN = 0.01, 2.0, 0.5, 30
FLOOR_BASE_MS = 4.0                  # G15: 4 ms plus the worst-case drift over the session
STIM_PAD_S = (1.0, 3.0)              # beats excluded from 1 s before a stimulation train to 3 s after it
EDGE_THR_Z = 8.0                     # EEG pulse detector: a percentile across channels of normalized |diff|
EDGE_CH_PCT = 75                     # that percentile (the transition spike is strong in most, not all, channels)
EDGE_GAP_MS = 50                     # quiet gap that separates the transition spike from the pulse train
EDGE_TRAIN_MIN_SAMPLES = 20          # active samples required in the 300 ms before that gap (a train precedes)
LFP_EDGE_MIN_Z = 10.0                # a clear LFP step: |diff| over the noise scale; pure noise over the +-6 sample search
                                     # peaks near 2.5, the weakest device offset step of the cohort is near 24
TASK = {"sitting": "Sit", "standing": "Stand", "walk": "Walk", "walkws": "WalkWS", "walkint": "WalkINT",
        "walkint2": "WalkINT_new"}
MA_TASK = {"sitting": "sitting", "standing": "standing", "walk": "walk", "walkws": "walk_with_stops",
           "walkint": "walk_dualtask", "walkint2": "walk_dualtask_2"}


def get_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--force", action="store_true", help="recompute sessions already cached")
    own.add_argument("--only", default="", help="comma list like sub-09/walk (debug; summary still written)")
    mine, rest = own.parse_known_args()
    if "--out" not in rest:
        rest += ["--out", os.path.join(TREE, "reports", "sync_sensefog")]
    sys.argv = [sys.argv[0]] + rest
    a = parse_args(__doc__.splitlines()[0])
    a.force = mine.force
    a.only = [s for s in mine.only.split(",") if s]
    return a


def rnd(v, k=4):
    if v is None:
        return None
    try:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        f = float(v)
        if not np.isfinite(f):
            return None
        return round(f, k)
    except (TypeError, ValueError):
        return v


# ------------------------------------------------------------------------------------------
# readers
# ------------------------------------------------------------------------------------------
def read_signalpoints(path):
    """Authors' hand-read alignment samples; None where the file sets []."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            code = line.split("%")[0]
            m = re.search(r"subjectdata\.signalpoint\.(\w+)\.(EEG_signal|LFP_signal)\s*=\s*([^;]*);", code)
            if m:
                v = m.group(3).strip()
                out.setdefault(m.group(1), {})[m.group(2)] = int(v) if re.fullmatch(r"\d+", v) else None
    return out


def read_vhdr(path):
    txt = open(path, encoding="utf-8", errors="replace").read()

    def key(k):
        m = re.search(rf"^{k}=(.*)$", txt, re.M)
        return m.group(1).strip() if m else None
    chans = []
    for m in re.finditer(r"^Ch(\d+)=([^\n]*)$", txt, re.M):
        parts = m.group(2).split(",")
        res = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 1.0
        chans.append((int(m.group(1)), parts[0].strip(), res, parts[3].strip() if len(parts) > 3 else ""))
    chans.sort()
    lowcut = re.findall(r"^\d+\s+\S+\s+\d+\s+[\d.]+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)", txt, re.M)
    return {"nch": int(key("NumberOfChannels")), "fs": 1e6 / float(key("SamplingInterval")),
            "fmt": key("BinaryFormat"), "orient": key("DataOrientation"),
            "names": [c[1] for c in chans], "res": np.array([c[2] for c in chans]), "units": [c[3] for c in chans],
            "hw_filters": sorted(set(lowcut))}


def read_vmrk(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r"^Mk(\d+)=([^,\n]*),([^,\n]*),(\d+),", txt, re.M):
        out.append({"type": m.group(2), "desc": m.group(3), "pos": int(m.group(4))})
    return out


class RawEEG:
    """BrainVision INT_16 multiplexed, read channel by channel from a memmap (values in uV)."""

    def __init__(self, sd, stem):
        self.h = read_vhdr(f"{sd}/eeg/{stem}.vhdr")
        assert self.h["fmt"] == "INT_16" and self.h["orient"] == "MULTIPLEXED", self.h
        p = f"{sd}/eeg/{stem}.eeg"
        nbytes = os.path.getsize(p)
        self.nch = self.h["nch"]
        self.n = nbytes // (2 * self.nch)
        self.exact = nbytes == self.n * 2 * self.nch
        self.mm = np.memmap(p, dtype="<i2", mode="r", shape=(self.n, self.nch))
        self.names = self.h["names"]
        self.fs = self.h["fs"]
        self.marks = read_vmrk(f"{sd}/eeg/{stem}.vmrk")

    def chan(self, i, lo=0, hi=None):
        return np.asarray(self.mm[lo:hi, i], dtype=np.float64) * self.h["res"][i]


class MatEEG:
    """Authors' aligned EEG (_eegalg.mat): 58 x n in uV at 1000 Hz, already cut."""

    def __init__(self, path):
        e = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["EEG_File"]
        self.x = np.asarray(e.EEG_signal, dtype=np.float64)
        self.names = [str(s) for s in np.atleast_1d(e.vhdr.label)]
        self.fs = float(e.fs_eeg)
        self.nch, self.n = self.x.shape
        self.exact = True
        self.marks = [{"type": str(getattr(v, "type", "")), "desc": str(getattr(v, "value", "")),
                       "pos": int(getattr(v, "sample", 0))} for v in np.atleast_1d(e.vmrk)]
        self.h = {"fs": self.fs, "hw_filters": []}

    def chan(self, i, lo=0, hi=None):
        return self.x[i, lo:hi]


def _ints(v):
    if isinstance(v, str):
        return [int(t) for t in v.strip().strip(",").split(",") if t.strip()]
    return [int(t) for t in np.atleast_1d(v)]


def to_py(o):
    if isinstance(o, sio.matlab.mat_struct):
        return {k: to_py(getattr(o, k)) for k in o._fieldnames}
    if isinstance(o, np.ndarray) and o.dtype == object:
        return [to_py(v) for v in o.ravel()]
    return o


def percept_streams(d):
    """Only the fields needed for timing; dates, serials and patient fields are never read out."""
    out = {"td": [], "stim_ivs": [], "hp_hz": None, "blank_us": None, "stim_status": ""}
    td = d.get("BrainSenseTimeDomain") or []
    if isinstance(td, dict):
        td = [td]
    for s in td:
        out["td"].append({"channel": str(s["Channel"]), "fs": float(s["SampleRateInHz"]), "gain": s.get("Gain"),
                          "ticks": np.array(_ints(s["TicksInMses"]), dtype=np.int64),
                          "sizes": np.array(_ints(s["GlobalPacketSizes"]), dtype=np.int64),
                          "x": np.asarray(s["TimeDomainData"], dtype=np.float64).ravel()})
    st = d.get("Stimulation") or {}
    if isinstance(st, dict):
        out["stim_status"] = (str(st.get("InitialStimStatus", "")).replace("StimStatusDef.", "") + "->" +
                              str(st.get("FinalStimStatus", "")).replace("StimStatusDef.", ""))
    bsl = d.get("BrainSenseLfp") or []
    if isinstance(bsl, dict):
        bsl = [bsl]
    ivs = []
    for s in bsl:
        ts = s.get("TherapySnapshot") or {}
        if isinstance(ts, dict):
            out["hp_hz"] = ts.get("HighPassFilterInHertz", out["hp_hz"])
            for bk in ("SensingBlankingDurationInMicroseconds", "SensingBlankingDurationInMilliSeconds"):
                if ts.get(bk) is not None:
                    out["blank_us"] = f"{bk.replace('SensingBlankingDurationIn', '')}={ts.get(bk)}"   # verbatim key and value
        ld = s.get("LfpData") or []
        if isinstance(ld, dict):
            ld = [ld]
        t = np.array([int(e["TicksInMs"]) for e in ld], dtype=np.int64)
        mA = np.array([[float(e["Left"]["mA"]), float(e["Right"]["mA"])] for e in ld]) if len(ld) else np.zeros((0, 2))
        on = np.any(mA > 0, axis=1)
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                ivs.append((int(t[i]), int(t[j]), float(mA[i:j + 1, 0].max()), float(mA[i:j + 1, 1].max())))
                i = j + 1
            else:
                i += 1
    out["stim_ivs_ticks"] = ivs
    return out


def rebuild_timebase(tdst):
    """Place every packet by GlobalPacketSizes; a tick step of g > step ms inserts (g - step) * fs / 1000
    masked samples. Returns grid (NaN where masked), valid mask and facts."""
    ticks, sizes, x, fs = tdst["ticks"], tdst["sizes"], tdst["x"], tdst["fs"]
    f = {"n_samples": int(len(x)), "n_packets": int(len(ticks)), "sum_sizes": int(sizes.sum()),
         "sizes_match": bool(int(sizes.sum()) == len(x)), "fs": fs}
    dt = np.diff(ticks)
    step = int(np.median(dt)) if len(dt) else 0
    f["tick_step_ms"] = step
    insert = np.zeros(len(ticks), dtype=np.int64)
    lost, irregular = 0, 0
    for i in np.flatnonzero(dt != step):
        g = int(dt[i])
        if g > step and g % step == 0:
            lost += g // step - 1
        else:
            irregular += 1
        insert[i + 1] = max(0, int(round((g - step) * fs / 1000.0)))
    f["n_tick_gaps"] = int(np.sum(dt != step))
    f["lost_ticks"] = int(lost)
    f["irregular_tick_steps"] = int(irregular)
    f["gap_at_s"] = [rnd((ticks[i + 1] - ticks[0]) / 1000.0, 3) for i in np.flatnonzero(dt != step)]
    if not f["sizes_match"]:
        f["placement"] = "concatenated (sizes do not sum to the sample count)"
        grid, valid = x.copy(), np.ones(len(x), bool)
        f["masked_samples"] = 0
        f["grid_len"] = len(x)
        return grid, valid, f
    f["placement"] = "packets by GlobalPacketSizes, lost ticks masked"
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    off = np.cumsum(insert)
    grid_len = int(len(x) + insert.sum())
    grid = np.full(grid_len, np.nan)
    valid = np.zeros(grid_len, bool)
    for k in range(len(ticks)):
        a0 = int(starts[k] + off[k])
        grid[a0:a0 + sizes[k]] = x[starts[k]:starts[k] + sizes[k]]
        valid[a0:a0 + sizes[k]] = True
    f["masked_samples"] = int((~valid).sum())
    f["grid_len"] = grid_len
    # packet accounting against the tick clock (same device crystal: checks the packets, not the drift)
    cum_end = np.cumsum(sizes) + off
    dev = cum_end - (ticks - ticks[0]) * fs / 1000.0
    f["tick_dev_range_samples"] = rnd(dev.max() - dev.min(), 2)
    # samples in excess of the tick clock (a regular stream sits at 0 +- 0.5: one packet holds step*fs/1000 samples)
    f["count_minus_tick_samples"] = rnd(np.median(dev) - step * fs / 1000.0, 2)
    f["size_values"] = sorted(set(int(s) for s in sizes))
    # packets holding more samples than one tick step can carry: (packet index, tick time from the first tick s, size)
    cap = int(np.ceil(step * fs / 1000.0))
    f["oversize_packets"] = [(int(k), rnd((ticks[k] - ticks[0]) / 1000.0, 3), int(sizes[k])) for k in np.flatnonzero(sizes > cap)]
    f["duration_s"] = rnd(grid_len / fs, 3)
    return grid, valid, f


# ------------------------------------------------------------------------------------------
# stimulation-artefact edges
# ------------------------------------------------------------------------------------------
def crossing_times(x, i_from, i_peak, base, frac_list):
    """Linear-interpolated crossing times (in samples) of base + frac*(x[i_peak]-base) on the rising flank."""
    A = x[i_peak] - base
    out = {}
    for fr in frac_list:
        lev = base + fr * A
        t = None
        for j in range(i_peak, i_from, -1):
            a0, a1 = x[j - 1] - lev, x[j] - lev
            if (a0 <= 0 < a1) if A > 0 else (a0 >= 0 > a1):
                t = (j - 1) + a0 / (a0 - a1)
                break
        out[fr] = t
    return out


def parabolic(y, i):
    if 0 < i < len(y) - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d != 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / d
    return float(i)


def edge_shape(x, k_step, fs):
    """Rising flank around the steepest step between samples k_step and k_step+1 (sign-agnostic)."""
    d = np.diff(x)
    sgn = 1.0 if d[k_step] > 0 else -1.0
    y = sgn * x
    j0 = k_step
    while j0 > 0 and (y[j0] - y[j0 - 1]) > 0:
        j0 -= 1
    p = k_step + 1
    while p + 1 < len(y) and (y[p + 1] - y[p]) > 0:
        p += 1
    pre_lo, pre_hi = max(0, j0 - int(0.040 * fs)), max(1, j0 - 1)
    base = float(np.median(y[pre_lo:pre_hi])) if pre_hi > pre_lo else float(y[j0])
    pk = parabolic(y, p)
    cr = crossing_times(y, max(j0 - 2, 1), p, base, [0.1, 0.2, 0.5, 0.8, 0.9])
    res = {"sign": sgn, "amp": float(y[p] - base), "i_peak": pk, "i_onset_sample": j0}
    for fr, v in cr.items():
        res[f"i_{int(fr * 100)}"] = v
    if cr[0.2] is not None and cr[0.8] is not None and cr[0.8] > cr[0.2]:
        res["i_onset_extrap"] = cr[0.2] - 0.2 * (cr[0.8] - cr[0.2]) / 0.6
    else:
        res["i_onset_extrap"] = None
    return res


def lfp_step(grid, fs, s_lo, s_hi, pad=0.3):
    """Steepest sample-to-sample step whose left sample lies in [s_lo, s_hi] (s); its shape is measured on a
    context of +-pad s. Returns times in LFP seconds."""
    c0, c1 = max(1, int((s_lo - pad) * fs)), min(len(grid) - 2, int((s_hi + pad) * fs))
    x = grid[c0:c1]
    if len(x) < 20 or np.any(~np.isfinite(x)):
        return None
    d = np.abs(np.diff(x))
    k0, k1 = max(0, int(np.floor(s_lo * fs)) - c0), min(len(d), int(np.ceil(s_hi * fs)) - c0 + 1)
    if k1 <= k0:
        return None
    k = k0 + int(np.argmax(d[k0:k1]))
    es = edge_shape(x, k, fs)
    ref = d[:max(5, k - int(0.1 * fs))]
    noise = 1.4826 * np.median(ref) + 1e-9
    out = {"t_peak": (c0 + es["i_peak"]) / fs, "t_step": (c0 + k + 0.5) / fs, "step_z": float(d[k] / noise),
           "amp": es["amp"], "sign": es["sign"]}
    for key in ("i_10", "i_20", "i_50", "i_80", "i_90", "i_onset_extrap"):
        v = es.get(key)
        out["t" + key[1:]] = None if v is None else (c0 + v) / fs
    return out


MATCH_HALF_S = 0.024     # the LFP feature matched to a reading is searched within +-6 samples at 250 Hz


def eeg_transition_spike(E, i_pred, fs, prefer=("IPG_1", "IPG_2")):
    """The isolated transient that follows the stimulation pulse train (the authors' EEG anchor)."""
    pre, post = int(0.9 * fs), int(0.6 * fs)
    lo, hi = max(1, i_pred - pre), min(E.n - 1, i_pred + post)
    rlo, rhi = min(E.n - 1, hi + int(0.5 * fs)), min(E.n, hi + int(2.5 * fs))
    if hi - lo < 100 or rhi - rlo < 200:
        return None
    Z = []
    for c in range(E.nch):
        x = E.chan(c, lo - 1, hi)
        r = E.chan(c, rlo, rhi)
        sc = 1.4826 * np.median(np.abs(np.diff(r))) + 1e-6
        Z.append(np.abs(np.diff(x)) / sc)
    Z = np.array(Z)
    a = np.percentile(Z, EDGE_CH_PCT, axis=0)
    idx = np.flatnonzero(a > EDGE_THR_Z)
    if len(idx) == 0:
        return None
    cl = []
    s0 = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v - prev > 3:
            cl.append((s0, prev))
            s0 = v
        prev = v
    cl.append((s0, prev))
    gap = int(EDGE_GAP_MS * fs / 1000)
    above = np.zeros(len(a), bool)
    above[idx] = True
    cands = []
    for j in range(1, len(cl)):
        g = cl[j][0] - cl[j - 1][1]
        e_prev = cl[j - 1][1]
        n_before = int(above[max(0, e_prev - int(0.3 * fs)):e_prev + 1].sum())   # active samples of the train
        if g >= gap and n_before >= EDGE_TRAIN_MIN_SAMPLES:
            cands.append((j, g, n_before))
    if not cands:
        return {"found": False, "n_clusters": len(cl)}
    # candidate nearest to the prediction; within its first 150 ms the dominant cluster is the spike (a small
    # transient may precede it inside the quiet gap)
    j, g, nb = min(cands, key=lambda c: abs((lo + cl[c[0]][0]) - i_pred))
    grp = [jj for jj in range(j, len(cl)) if cl[jj][0] - cl[j][0] <= int(0.15 * fs)]
    jm = max(grp, key=lambda jj: a[cl[jj][0]:cl[jj][1] + 1].max())
    if jm != j:
        g = cl[jm][0] - cl[jm - 1][1]
    c0, c1 = cl[jm]
    # channel: preferred IPG channel if it carries the spike, else the channel with the largest z
    zc = Z[:, max(0, c0 - 2):c1 + 3].max(axis=1)
    ch = None
    for nm in prefer:
        if nm in E.names and zc[E.names.index(nm)] > 2 * EDGE_THR_Z:
            ch = E.names.index(nm)
            break
    if ch is None:
        ch = int(np.argmax(zc))
    w0, w1 = max(0, c0 - int(0.02 * fs)), min(hi - lo, c1 + int(0.01 * fs))
    x = E.chan(ch, lo + w0, lo + w1 + 1)
    base = float(np.median(E.chan(ch, lo + max(0, c0 - int(0.03 * fs)), lo + max(1, c0 - 2))))
    k = int(np.argmax(np.abs(x - base)))
    y = np.sign(x[k] - base) * (x - base)
    pk = parabolic(y, k)
    cr = crossing_times(y, 1, k, 0.0, [0.1, 0.2, 0.5, 0.8, 0.9])
    onset = None
    if cr[0.2] is not None and cr[0.8] is not None and cr[0.8] > cr[0.2]:
        onset = cr[0.2] - 0.2 * (cr[0.8] - cr[0.2]) / 0.6
    t0 = (lo + w0) / fs
    out = {"found": True, "t_peak": t0 + pk / fs, "channel": E.names[ch], "sign": float(np.sign(x[k] - base)),
           "z": float(zc[ch]), "gap_ms": g * 1000.0 / fs, "train_clusters_300ms": nb,
           "t_onset_extrap": None if onset is None else t0 + onset / fs}
    for fr in (0.1, 0.5, 0.9):
        out[f"t_{int(fr * 100)}"] = None if cr[fr] is None else t0 + cr[fr] / fs
    return out


# ------------------------------------------------------------------------------------------
# ECG anchor
# ------------------------------------------------------------------------------------------
def bandpass(x, fs):
    b, a = signal.butter(3, ECG_BAND_HZ, btype="band", fs=fs)
    return signal.filtfilt(b, a, x)


EMG_RE = re.compile(r"^(TA|GA)_[LR]\d$")


def score_ecg_channels(E, i0, i1):
    """Score = regularity**4 x peak amplitude over noise. Regularity (share of intervals within 15 percent of the
    median) is what separates the heart from gait and electrode artefacts, so it dominates. Leg EMG channels are
    not candidates: their gait bursts are regular too. Sharpness is reported, not scored."""
    fs = E.fs
    res = []
    half = int(0.2 * fs)
    far = np.abs(np.arange(-half, half + 1)) > int(0.06 * fs)
    for c, nm in enumerate(E.names):
        if EMG_RE.match(nm):
            continue
        x = bandpass(E.chan(c, i0, i1), fs)
        for pol in (1, -1):
            y = pol * x
            h = 0.4 * np.percentile(y, 99.5)
            if h <= 0:
                continue
            pk, _ = signal.find_peaks(y, height=h, distance=int(0.33 * fs))
            pk = pk[(pk > half) & (pk < len(y) - half - 1)]
            if len(pk) < 10:
                continue
            ibi = np.diff(pk) / fs
            med = float(np.median(ibi))
            if not (0.33 <= med <= 1.6):
                continue
            reg = float(np.mean(np.abs(ibi - med) < 0.15 * med))
            noise = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9
            amp = float(np.median(y[pk]) / noise)
            tm = np.mean([y[p - half:p + half + 1] for p in pk], axis=0)
            sharp = float(tm[half] / (np.sqrt(np.mean(tm[far] ** 2)) + 1e-9))
            res.append({"score": reg ** 4 * amp, "reg": reg, "amp": amp, "sharp": sharp, "ibi": med,
                        "channel": nm, "pol": pol})
    res.sort(key=lambda r: -r["score"])
    return res


def detect_r(E, ch, pol, excl, lo_i, hi_i):
    """R peaks (sub-sample, EEG seconds) in [lo_i, hi_i) of channel ch, outside excluded index intervals."""
    fs = E.fs
    y = pol * bandpass(E.chan(ch), fs)
    ok = np.zeros(len(y), bool)
    ok[lo_i:hi_i] = True
    for a0, a1 in excl:
        ok[max(0, a0):max(0, a1)] = False
    yz = np.where(ok, y, 0.0)
    thr = np.zeros(len(y))
    blk = int(10 * fs)
    gthr = 0.4 * np.percentile(y[ok], 99.5) if ok.any() else np.inf
    for b0 in range(0, len(y), blk):
        seg = y[b0:b0 + blk][ok[b0:b0 + blk]]
        t = 0.4 * np.percentile(seg, 99.5) if len(seg) > fs else gthr
        thr[b0:b0 + blk] = max(t, 0.5 * gthr)
    pk, _ = signal.find_peaks(yz, height=thr, distance=int(0.33 * fs))
    half = int(0.06 * fs)
    pk = pk[(pk > half) & (pk < len(y) - half - 1)]
    n_raw = len(pk)
    if len(pk) < 10:
        return np.array([]), {"n_raw": n_raw, "n_kept": 0}
    segs = np.array([y[p - half:p + half + 1] for p in pk])
    tm = np.median(segs, axis=0)
    segc = segs - segs.mean(1, keepdims=True)
    tmc = tm - tm.mean()
    r = segc @ tmc / (np.linalg.norm(segc, axis=1) * np.linalg.norm(tmc) + 1e-12)
    pk = pk[r >= 0.8]
    # double detections: keep the larger of two peaks closer than half the median interval
    if len(pk) > 2:
        med = np.median(np.diff(pk))
        keep = np.ones(len(pk), bool)
        for i in range(1, len(pk)):
            if pk[i] - pk[i - 1] < 0.5 * med:
                if y[pk[i]] > y[pk[i - 1]]:
                    keep[i - 1] = False
                else:
                    keep[i] = False
        pk = pk[keep]
    t = np.array([parabolic(y, int(p)) for p in pk]) / fs
    ibi = np.diff(t)
    return t, {"n_raw": int(n_raw), "n_kept": int(len(t)), "median_ibi_s": float(np.median(ibi)) if len(ibi) else None,
               "frac_ibi_within_15pct": float(np.mean(np.abs(ibi - np.median(ibi)) < 0.15 * np.median(ibi))) if len(ibi) else None}


def xshift(T, R, maxlag, absolute=False):
    """Lag (in template samples, here ms) maximizing sum T[i+k]*R[i]; positive = T later than R."""
    n = len(R)
    cc = np.empty(2 * maxlag + 1)
    for j, k in enumerate(range(-maxlag, maxlag + 1)):
        cc[j] = np.dot(T[k:], R[:n - k]) if k >= 0 else np.dot(T[:k], R[-k:])
    c = np.abs(cc) if absolute else cc
    i = int(np.argmax(c))
    f = parabolic(c, i)
    norm = np.sqrt(np.dot(T, T) * np.dot(R, R)) + 1e-12
    return f - maxlag, float(cc[i] / norm)


class LfpEcg:
    def __init__(self, grid, valid, fs, excl_s):
        self.fs = fs
        x = np.where(valid, grid - np.nanmedian(grid), 0.0)
        self.xf = bandpass(x, fs)
        self.t = np.arange(len(x)) / fs
        self.cs = interpolate.CubicSpline(self.t, self.xf)
        inv = (~valid).astype(np.int64)
        self.cinv = np.concatenate([[0], np.cumsum(inv)])
        self.excl = excl_s
        self.T = len(x) / fs

    def usable(self, tb):
        pad = 0.15 + 0.5
        ok = (tb - pad > 0) & (tb + pad < self.T)
        i0 = np.clip(((tb - pad) * self.fs).astype(int), 0, len(self.cinv) - 1)
        i1 = np.clip(((tb + pad) * self.fs).astype(int) + 1, 0, len(self.cinv) - 1)
        ok &= (self.cinv[i1] - self.cinv[i0]) == 0
        for a0, a1 in self.excl:
            ok &= ~((tb > a0) & (tb < a1))
        return ok

    def mat(self, tb):
        return self.cs(tb[:, None] + LAGS_MS[None, :] / 1000.0)


def ecg_channel_analysis(le, tR_lfp, tR_eeg, eeg_tmpl_fn, rng, ipg1_mat_fn=None):
    ok = le.usable(tR_lfp)
    tb, te = tR_lfp[ok], tR_eeg[ok]
    out = {"n_beats_usable": int(len(tb))}
    if len(tb) < 3 * MIN_BEATS_WIN:
        out["status"] = "too_few_beats"
        return out, []
    M = le.mat(tb)
    amp = np.max(np.abs(M), axis=1)
    keep = amp < 6 * np.median(amp)
    out["n_beats_outlier"] = int((~keep).sum())
    M, tb, te = M[keep], tb[keep], te[keep]
    ref = M.mean(0)
    jr = int(np.argmax(np.abs(ref)))
    obs = float(np.abs(ref[jr]))
    out["template_peak_lag_ms"] = int(LAGS_MS[jr])
    out["template_peak_amp"] = float(ref[jr])
    # null: circular shifts of the beat train inside the usable span
    lo, hi = tb[0] - 0.5, tb[-1] + 0.5
    nulls = []
    for _ in range(N_NULL):
        sh = rng.uniform(1.0, hi - lo - 1.0)
        tn = lo + ((tb - lo + sh) % (hi - lo))
        tn = tn[le.usable(tn)]
        if len(tn) < 10:
            continue
        nulls.append(float(np.max(np.abs(le.mat(tn).mean(0)))))
    nulls = np.array(nulls)
    out["p_null"] = float((1 + np.sum(nulls >= obs)) / (len(nulls) + 1))
    out["peak_over_null95"] = float(obs / np.percentile(nulls, 95)) if len(nulls) else None
    ME = None
    if ipg1_mat_fn is not None:
        # whole-session lag of the LFP template against the IPG_1 ECG template (same beats); compared across the
        # sessions of one subject, where lead geometry is fixed, it checks the hand reading without the stim artefact
        ME = ipg1_mat_fn(te)
        ell, rr = xshift(ref, ME.mean(0), ELL_MAX_MS, absolute=True)
        bo = []
        for _ in range(N_BOOT):
            ii = rng.integers(0, len(te), len(te))
            bo.append(xshift(M[ii].mean(0), ME[ii].mean(0), ELL_MAX_MS, absolute=True)[0])
        out["lag_all_vs_IPG_1_ms"] = float(ell)
        out["lag_all_vs_IPG_1_sd_ms"] = float(np.std(bo))
        out["lag_all_vs_IPG_1_r"] = rr
        out["all_center_eeg_s"] = float(te.mean())
    T0, T1 = tb[0], tb[-1]
    W = min(WIN_MAX_S, (T1 - T0) / 3.0)
    out["win_s"] = W
    wins = {}
    for name, sel in (("start", tb < T0 + W), ("end", tb > T1 - W)):
        idx = np.flatnonzero(sel)
        Tw = M[idx].mean(0)
        s, r = xshift(Tw, ref, SHIFT_MAX_MS)
        boots = [xshift(M[rng.choice(idx, len(idx), replace=True)].mean(0), ref, SHIFT_MAX_MS)[0] for _ in range(N_BOOT)]
        ell, rell = xshift(Tw, eeg_tmpl_fn(te[idx]), ELL_MAX_MS, absolute=True)
        wins[name] = {"n": int(len(idx)), "center_lfp_s": float(tb[idx].mean()), "center_eeg_s": float(te[idx].mean()),
                      "shift_ms": float(s), "shift_sd_ms": float(np.std(boots)), "r_ref": r,
                      "ecg_lag_ms": float(ell), "ecg_lag_r": rell, "tmpl": Tw}
        if ME is not None:
            e1, r1 = xshift(Tw, ME[idx].mean(0), ELL_MAX_MS, absolute=True)
            b1 = []
            for _ in range(N_BOOT):
                ii = rng.choice(idx, len(idx), replace=True)
                b1.append(xshift(M[ii].mean(0), ME[ii].mean(0), ELL_MAX_MS, absolute=True)[0])
            wins[name].update({"lag_vs_IPG_1_ms": float(e1), "lag_vs_IPG_1_sd_ms": float(np.std(b1)), "lag_vs_IPG_1_r": r1})
    for name in ("start", "end"):
        for k, v in wins[name].items():
            if k != "tmpl":
                out[f"{name}_{k}"] = v
    ts, tend = wins["start"]["tmpl"], wins["end"]["tmpl"]
    out["r_start_end_zero_lag"] = float(np.corrcoef(ts, tend)[0, 1])
    out["r_start_end"] = xshift(tend, ts, SHIFT_MAX_MS)[1]      # at the best lag: a drift must not fail the shape test
    dt = wins["end"]["center_eeg_s"] - wins["start"]["center_eeg_s"]
    dms = wins["end"]["shift_ms"] - wins["start"]["shift_ms"]
    sems = float(np.hypot(wins["start"]["shift_sd_ms"], wins["end"]["shift_sd_ms"]))
    out["baseline_s"] = dt
    out["drift_ms"] = dms
    out["drift_se_ms"] = sems
    out["drift_ppm"] = dms / dt * 1000.0 if dt > 0 else None
    out["drift_se_ppm"] = sems / dt * 1000.0 if dt > 0 else None
    # sliding windows, OLS slope with residual-based standard error
    rows = []
    for c0 in np.arange(T0, T1 - SLIDE_W_S + 1e-9, SLIDE_STEP_S):
        idx = np.flatnonzero((tb >= c0) & (tb < c0 + SLIDE_W_S))
        if len(idx) < 20:
            continue
        s, r = xshift(M[idx].mean(0), ref, SHIFT_MAX_MS)
        bo = [xshift(M[rng.choice(idx, len(idx), replace=True)].mean(0), ref, SHIFT_MAX_MS)[0] for _ in range(N_BOOT_SLIDE)]
        rows.append({"center_eeg_s": float(te[idx].mean()), "n": int(len(idx)), "shift_ms": float(s), "sd_ms": float(np.std(bo)), "r_ref": r})
    if len(rows) >= 4:
        c = np.array([r_["center_eeg_s"] for r_ in rows])
        s = np.array([r_["shift_ms"] for r_ in rows])
        p, cov = np.polyfit(c, s, 1, cov=True)
        out["slide_ppm"] = float(p[0] * 1000.0)
        out["slide_se_ppm"] = float(np.sqrt(cov[0, 0]) * 1000.0)
        out["slide_n"] = len(rows)
        out["slide_resid_sd_ms"] = float(np.std(s - np.polyval(p, c), ddof=2))
    ok_q = (out["p_null"] <= P_NULL_MAX and out["start_shift_sd_ms"] <= BOOT_SD_MAX_MS and out["end_shift_sd_ms"] <= BOOT_SD_MAX_MS
            and out["r_start_end"] >= R_START_END_MIN and out["start_n"] >= MIN_BEATS_WIN and out["end_n"] >= MIN_BEATS_WIN)
    out["status"] = "pass" if ok_q else "fail"
    out["_ref"] = ref
    out["_M_tb"] = (M, tb, te)
    return out, rows


# ------------------------------------------------------------------------------------------
# one session
# ------------------------------------------------------------------------------------------
def session_seed(sub, ses):
    return SEED + zlib.crc32(f"{sub}/{ses}".encode("utf-8")) % 100000


def load_session_sources(sd, sub, ses):
    stem = f"{sub}-ses-{ses}"
    p = {"eeg_raw": f"{sd}/eeg/{stem}.vhdr", "eeg_alg": f"{sd}/eeg/{stem}_eegalg.mat",
         "json": f"{sd}/ieeg/{stem}.json", "lfp_alg": f"{sd}/ieeg/{stem}_lfpalg.mat",
         "h5": f"{sd}/motion/{stem}.h5"}
    return stem, {k: (v if os.path.exists(v) else None) for k, v in p.items()}


def process_session(a, sd, sub, ses, sp):
    t_start = time.time()
    stem, src = load_session_sources(sd, sub, ses)
    rng = np.random.default_rng(session_seed(sub, ses))
    R = {"sub": sub, "ses": ses, "task_key": TASK.get(ses, ""), "files": {k: bool(v) for k, v in src.items()}}
    notes = []
    # ---------------- EEG
    E = None
    if src["eeg_raw"]:
        E = RawEEG(sd, stem)
        R["eeg_source"] = "raw_brainvision"
        R["eeg_hw_filters"] = E.h["hw_filters"]
    elif src["eeg_alg"]:
        E = MatEEG(src["eeg_alg"])
        R["eeg_source"] = "authors_eegalg_mat (already cut at the anchor)"
    else:
        R["eeg_source"] = "none"
    if E is not None:
        R.update({"eeg_fs": E.fs, "eeg_nch": E.nch, "eeg_n": int(E.n), "eeg_dur_s": E.n / E.fs, "eeg_size_exact": E.exact,
                  "eeg_n_segments": sum(1 for m in E.marks if m["type"] == "New Segment"),
                  "eeg_n_toggle": sum(1 for m in E.marks if m["type"] == "Toggle"),
                  "has_ipg_channels": all(n in E.names for n in ("IPG_1", "IPG_2")),
                  "n_emg_channels": sum(1 for n in E.names if re.match(r"(TA|GA)_[LR]\d", n))})
        m1 = [m["pos"] for m in E.marks if m["desc"] == "M  1"]
        R["eeg_m1_positions"] = m1
    # ---------------- LFP
    P = None
    if src["json"]:
        with open(src["json"], encoding="utf-8") as f:
            P = percept_streams(json.load(f))
        R["lfp_source"] = "raw_json"
    elif src["lfp_alg"]:
        L = sio.loadmat(src["lfp_alg"], squeeze_me=True, struct_as_record=False)["LFP"]
        if "rawfile" in L._fieldnames:
            P = percept_streams(to_py(L.rawfile))
            R["lfp_source"] = "json_embedded_in_authors_lfpalg_mat"
        else:
            R["lfp_source"] = "authors_lfpalg_mat_1000Hz_no_ticks"
            P = {"td": [], "stim_ivs_ticks": [], "hp_hz": None, "blank_us": None, "stim_status": "",
                 "alg": {"L": np.asarray(L.LFP_signal_L, float), "R": np.asarray(L.LFP_signal_R, float)}}
    else:
        R["lfp_source"] = "none"
    streams = []
    if P is not None:
        R["lfp_hp_hz"] = P.get("hp_hz")
        R["lfp_blanking_us"] = P.get("blank_us")
        R["stim_status"] = P.get("stim_status")
        if P["td"]:
            t0tick = int(P["td"][0]["ticks"][0])
            R["lfp_n_streams"] = len(P["td"])
            R["lfp_channels"] = [s["channel"] for s in P["td"]]
            R["lfp_gains"] = [s["gain"] for s in P["td"]]
            R["lfp_tick0_equal"] = all(int(s["ticks"][0]) == t0tick for s in P["td"])
            for s in P["td"]:
                grid, valid, facts = rebuild_timebase(s)
                streams.append({"channel": s["channel"], "grid": grid, "valid": valid, "facts": facts, "fs": s["fs"],
                                "x_concat": s["x"]})
            R["stim_trains_2hz"] = [(rnd((i0 - t0tick) / 1000.0, 3), rnd((i1 - t0tick) / 1000.0, 3), l_, r_)
                                    for i0, i1, l_, r_ in P["stim_ivs_ticks"]]
        elif "alg" in P:
            for nm in ("L", "R"):
                x = P["alg"][nm]
                streams.append({"channel": f"authors_lfpalg_{nm}", "grid": x, "valid": np.ones(len(x), bool), "fs": 1000.0,
                                "facts": {"n_samples": len(x), "placement": "authors' 1000 Hz resample, no ticks",
                                          "masked_samples": 0, "grid_len": len(x), "duration_s": len(x) / 1000.0}, "x_concat": x})
            R["lfp_n_streams"] = 2
            R["lfp_channels"] = [s["channel"] for s in streams]
            R["stim_trains_2hz"] = []
    if streams:
        R["lfp_fs"] = streams[0]["fs"]
        R["lfp_dur_s"] = streams[0]["facts"]["grid_len"] / streams[0]["fs"]
        R["lfp_masked_samples"] = [s["facts"]["masked_samples"] for s in streams]
        R["lfp_lost_ticks"] = [s["facts"].get("lost_ticks") for s in streams]
        R["lfp_timebase"] = [{k: v for k, v in s["facts"].items()} for s in streams]
        # MoveAtlas processed cross-check (lengths and masked counts only)
        mp = os.path.join(a.moveatlas, "datasets", "eeg", "sensefog", "processed", "aux",
                          f"sensefog__sensefog_{sub}__{MA_TASK.get(ses, ses)}_lfp.npz")
        if os.path.exists(mp):
            z = np.load(mp, allow_pickle=False)
            ma = []
            k = 0
            while f"streams.{k}.valid" in z.files:
                v = z[f"streams.{k}.valid"]
                ma.append((int(len(v)), int((~v).sum())))
                k += 1
            mine = sorted((s["facts"]["grid_len"], s["facts"]["masked_samples"]) for s in streams)
            R["moveatlas_lfp_len_masked"] = ma
            R["moveatlas_lfp_agrees"] = sorted(ma) == mine
    # ---------------- authors' anchor
    ev, lv = sp.get("EEG_signal"), sp.get("LFP_signal")
    R["auth_EEG_signal"], R["auth_LFP_signal"] = ev, lv
    has_anchor = ev is not None and lv is not None and E is not None and bool(streams)
    if ev is not None and lv is not None:
        R["auth_branch"] = "LFP_cut" if lv > ev else "EEG_cut"
        R["auth_delay_samples"] = abs(lv - ev)
    if has_anchor:
        if R["eeg_source"] == "raw_brainvision":
            use_m1 = bool(R["eeg_m1_positions"]) and bool(src["h5"])
            M1 = R["eeg_m1_positions"][0] if use_m1 else 1
            e0 = ev + M1 - 2                                     # 0-based raw EEG sample of the anchor
            R["auth_M1_used"] = M1
        else:
            # aligned EEG from _eegalg.mat: the cut stream keeps index delay:end (1.5_Data_Alignment.m)
            e0 = (ev - (ev - lv)) if ev > lv else (ev - 1)
            R["auth_M1_used"] = "n/a (aligned file)"
        l0 = (lv - 1) / 1000.0                                  # LFP time on the concatenated 1000 Hz resample
        theta = l0 - e0 / E.fs
        R["theta_auth_s"] = theta
        R["auth_eeg_anchor_s"] = e0 / E.fs
        R["auth_lfp_anchor_s"] = l0
        # overlap on the EEG clock
        T_L = R["lfp_dur_s"]
        ov0, ov1 = max(0.0, -theta), min(E.n / E.fs, T_L - theta)
        R["overlap_eeg_s"] = (ov0, ov1)
        R["duration_overlap_s"] = max(0.0, ov1 - ov0)
    else:
        notes.append("no usable authors' anchor (missing values, EEG or LFP)")
        R["notes"] = notes
        R["seconds"] = time.time() - t_start
        return R, [], []
    fsL = streams[0]["fs"]
    # ---------------- stimulation trains in LFP time
    trains = [(t0, t1) for t0, t1, _, _ in R["stim_trains_2hz"]]
    if not trains:
        # no 2 Hz stream (1000 Hz authors' file): the anchor is the offset of the only train
        trains = [(l0 - 10.0, l0)]
        notes.append("no BrainSenseLfp stim intervals: train assumed to end at the authors' anchor")
    # ---------------- stimulation edges (OFF), each train
    # The authors' train is the one whose span holds the authors' LFP sample (else the nearest). For it, the EEG
    # transition spike is searched around the authors' EEG sample and the LFP feature within +-MATCH_HALF_S of the
    # authors' LFP sample (this re-reads the hand reading). For any other train, the EEG spike is searched around
    # the major LFP step and the LFP feature around (EEG spike + re-read theta of the authors' train).
    def train_dist(tr):
        return 0.0 if (tr[0] - 1.0 <= l0 <= tr[1] + 2.0) else min(abs(l0 - tr[0]), abs(l0 - tr[1]))
    k_auth = min(range(len(trains)), key=lambda k: train_dist(trains[k]))
    R["auth_train_index"] = k_auth
    edges = []
    theta_re = None
    for k in [k_auth] + [k for k in range(len(trains)) if k != k_auth]:
        t_on, t_off = trains[k]
        has2 = k < len(R["stim_trains_2hz"])
        lo_w = max(t_on + 0.3, t_off - 0.5) if has2 else l0 - 0.5
        hi_w = t_off + 1.5 if has2 else l0 + 0.5
        row = {"train": k, "is_authors_train": k == k_auth, "train_on_2hz_s": t_on, "train_off_2hz_s": t_off}
        majors = []
        for s in streams:
            m_ = lfp_step(s["grid"], s["fs"], lo_w, hi_w)
            if m_:
                m_["channel"] = s["channel"]
                majors.append(m_)
        if not majors:
            row["found_lfp"] = False
            edges.append(row)
            continue
        maj = max(majors, key=lambda m_: m_["step_z"])
        row.update({"major_step_channel": maj["channel"], "major_step_t_s": maj["t_peak"], "major_step_z": maj["step_z"],
                    "major_step_clear": maj["step_z"] >= LFP_EDGE_MIN_Z})
        i_pred = int(round(e0)) if k == k_auth else int(round((maj["t_peak"] - theta) * E.fs))
        sp_ = eeg_transition_spike(E, i_pred, E.fs) if 0 < i_pred < E.n else None
        if not (sp_ and sp_.get("found")):
            row["found_eeg"] = False
            edges.append(row)
            continue
        row.update({"found_eeg": True, "eeg_channel": sp_["channel"], "eeg_t_peak_s": sp_["t_peak"], "eeg_z": sp_["z"],
                    "eeg_gap_after_train_ms": sp_["gap_ms"],
                    "eeg_onset_to_peak_ms": None if sp_["t_onset_extrap"] is None else (sp_["t_peak"] - sp_["t_onset_extrap"]) * 1000,
                    "major_step_minus_eeg_spike_on_auth_ms": ((maj["t_peak"] - theta) - sp_["t_peak"]) * 1000})
        if k == k_auth:
            centre = l0
        else:
            centre = sp_["t_peak"] + (theta_re if theta_re is not None else theta)
        feats = []
        for s in streams:
            f_ = lfp_step(s["grid"], s["fs"], centre - MATCH_HALF_S, centre + MATCH_HALF_S)
            if f_:
                f_["channel"] = s["channel"]
                feats.append(f_)
        if not feats:
            row["found_lfp"] = False
            edges.append(row)
            continue
        # the feature whose peak is nearest to the reading (in samples), the steeper one on a tie
        ft = min(feats, key=lambda f_: (round(abs(f_["t_peak"] - centre) * fsL), -f_["step_z"]))
        row.update({"found_lfp": True, "lfp_channel": ft["channel"], "lfp_t_peak_s": ft["t_peak"], "lfp_step_z": ft["step_z"],
                    "lfp_edge_clear": ft["step_z"] >= LFP_EDGE_MIN_Z, "lfp_amp": ft["amp"],
                    "lfp_rise_10_90_ms": None if (ft["t_10"] is None or ft["t_90"] is None) else (ft["t_90"] - ft["t_10"]) * 1000,
                    "lfp_onset_to_peak_ms": None if ft["t_onset_extrap"] is None else (ft["t_peak"] - ft["t_onset_extrap"]) * 1000,
                    "lfp_peak_channel_spread_ms": (max(f_["t_peak"] for f_ in feats) - min(f_["t_peak"] for f_ in feats)) * 1000,
                    "lfp_feature_is_major_step": abs(ft["t_peak"] - maj["t_peak"]) < 1.5 / fsL,
                    "theta_reread_s": ft["t_peak"] - sp_["t_peak"]})
        if row["lfp_onset_to_peak_ms"] is not None and row["eeg_onset_to_peak_ms"] is not None:
            row["percept_edge_delay_ms"] = row["lfp_onset_to_peak_ms"] - row["eeg_onset_to_peak_ms"]
        if ft["t_onset_extrap"] is not None and sp_["t_onset_extrap"] is not None:
            row["lfp_onset_minus_eeg_onset_on_auth_ms"] = ((ft["t_onset_extrap"] - theta) - sp_["t_onset_extrap"]) * 1000
        if k == k_auth:
            # a feature below the clear-edge gate is noise or a minor wiggle: it neither checks the reading nor guides the
            # search on another train (the hand-read theta does then)
            theta_re = row["theta_reread_s"] if row["lfp_edge_clear"] else None
            row["reread_minus_auth_ms"] = (row["theta_reread_s"] - theta) * 1000
            row["lfp_part_ms"] = (ft["t_peak"] - l0) * 1000
            row["eeg_part_ms"] = (sp_["t_peak"] - e0 / E.fs) * 1000
        edges.append(row)
    R["n_stim_trains"] = len(R["stim_trains_2hz"])
    ea = next((e_ for e_ in edges if e_["is_authors_train"]), {})
    for k_ in ("reread_minus_auth_ms", "lfp_part_ms", "eeg_part_ms", "eeg_channel", "lfp_rise_10_90_ms", "lfp_onset_to_peak_ms",
               "eeg_onset_to_peak_ms", "percept_edge_delay_ms", "lfp_onset_minus_eeg_onset_on_auth_ms", "found_eeg", "found_lfp",
               "theta_reread_s", "major_step_minus_eeg_spike_on_auth_ms", "lfp_feature_is_major_step", "lfp_step_z", "lfp_edge_clear",
               "major_step_z", "major_step_clear"):
        R[f"anchor_{k_}"] = ea.get(k_)
    others = [e_ for e_ in edges if not e_["is_authors_train"] and e_.get("theta_reread_s") is not None and e_.get("lfp_edge_clear")]
    if others and ea.get("theta_reread_s") is not None and ea.get("lfp_edge_clear"):
        e2 = max(others, key=lambda e_: abs(e_["eeg_t_peak_s"] - ea["eeg_t_peak_s"]))
        dt = e2["eeg_t_peak_s"] - ea["eeg_t_peak_s"]
        R["stim2_theta_minus_anchor_ms"] = (e2["theta_reread_s"] - ea["theta_reread_s"]) * 1000
        R["stim2_baseline_s"] = dt
        R["stim2_drift_ppm"] = (e2["theta_reread_s"] - ea["theta_reread_s"]) / dt * 1e6 if dt != 0 else None
        R["stim2_lfp_feature_is_major_step"] = e2.get("lfp_feature_is_major_step")
    # ---------------- ECG anchor
    fsE = E.fs
    ov0, ov1 = R["overlap_eeg_s"]
    excl_L = [(t0 - STIM_PAD_S[0], t1 + STIM_PAD_S[1]) for t0, t1 in trains]
    excl_E_idx = [(int((t0 - theta) * fsE), int((t1 - theta) * fsE)) for t0, t1 in excl_L]
    # scoring stretch: the longest clean stretch inside the overlap, at most 120 s
    last_excl_eeg = max(t1 - theta for _, t1 in excl_L)
    s0 = max(ov0, last_excl_eeg) + 1.0
    s1 = min(ov1 - 1.0, s0 + 120.0)
    ch_scores = score_ecg_channels(E, int(s0 * fsE), int(s1 * fsE)) if s1 - s0 > 20 else []
    ecg_rows, win_rows = [], []
    if not ch_scores:
        R["ecg_status"] = "no_ecg_channel"
        notes.append("no EEG-file channel with a regular ECG in the scoring stretch")
    else:
        best = ch_scores[0]
        R["ecg_eeg_channel"] = best["channel"]
        R["ecg_eeg_polarity"] = best["pol"]
        R["ecg_eeg_score"] = best["score"]
        R["ecg_eeg_runner_up"] = ch_scores[1]["channel"] if len(ch_scores) > 1 else None
        R["ecg_eeg_top5"] = [(c["channel"], c["pol"], rnd(c["score"], 2)) for c in ch_scores[:5]]
        ch = E.names.index(best["channel"])
        tR, rstats = detect_r(E, ch, best["pol"], excl_E_idx, int((ov0 + 0.7) * fsE), int((ov1 - 0.7) * fsE))
        R.update({f"r_{k}": v for k, v in rstats.items()})
        yE = best["pol"] * bandpass(E.chan(ch), fsE)
        csE = interpolate.CubicSpline(np.arange(len(yE)) / fsE, yE)

        def eeg_tmpl(te):
            return csE(te[:, None] + LAGS_MS[None, :] / 1000.0).mean(0)
        ipg1_fn = None
        if "IPG_1" in E.names:
            y1 = bandpass(E.chan(E.names.index("IPG_1")), fsE)
            cs1 = interpolate.CubicSpline(np.arange(len(y1)) / fsE, y1)

            def ipg1_fn(te):
                return cs1(te[:, None] + LAGS_MS[None, :] / 1000.0)
        if len(tR) < 3 * MIN_BEATS_WIN:
            R["ecg_status"] = "too_few_r_peaks"
        else:
            per = []
            for s in streams:
                le = LfpEcg(s["grid"], s["valid"], s["fs"], excl_L)
                res, rows = ecg_channel_analysis(le, tR + theta, tR, eeg_tmpl, rng, ipg1_fn)
                res["channel"] = s["channel"]
                # time-base check: the same end window on the concatenated axis (no gap handling)
                if s["facts"].get("masked_samples", 0) > 0 and res.get("status") in ("pass", "fail"):
                    xc = s["x_concat"]
                    lc = LfpEcg(xc, np.ones(len(xc), bool), s["fs"], excl_L)
                    M, tb, te = res["_M_tb"]
                    Wn = res["win_s"]
                    sel_s = tb < tb[0] + Wn
                    sel_e = tb > tb[-1] - Wn
                    ts_ = lc.mat(tb[sel_s]).mean(0)
                    tbe = tb[sel_e]
                    tbe = tbe[tbe + 0.8 < lc.T]
                    Mc = lc.cs(tbe[:, None] + np.arange(-700, 701)[None, :] / 1000.0).mean(0)
                    k_ = np.arange(-550, 551)          # Mc index m holds lag m - 700 ms
                    cc = np.array([np.dot(Mc[550 + k:550 + k + 301], ts_) for k in k_])
                    res["concat_axis_end_shift_ms"] = float(k_[int(np.argmax(cc))])
                    res["rebuilt_axis_end_shift_ms"] = res["end_shift_ms"] - res["start_shift_ms"]
                for r_ in rows:
                    r_.update({"sub": sub, "ses": ses, "channel": s["channel"]})
                win_rows += rows
                per.append(res)
            # negative control on the EEG clock: the same analysis on the scalp channel with the strongest R-locked
            # template (not IPG, not EMG); both clocks are the same, so its drift must be 0 within its SE
            cands = [c for c, nm in enumerate(E.names) if not EMG_RE.match(nm) and not nm.startswith("IPG")]
            ri = np.round(tR * fsE).astype(int)
            ri = ri[(ri > 200) & (ri < E.n - 200)]
            best_c, best_v = None, -1.0
            for c in cands:
                yc = bandpass(E.chan(c), fsE)
                tm = np.mean([yc[ri + k] for k in range(-150, 151, 2)], axis=1)
                v = float(np.max(np.abs(tm)) / (np.std(yc) / np.sqrt(len(ri)) + 1e-12))
                if v > best_v:
                    best_c, best_v = c, v
            if best_c is not None:
                xc = E.chan(best_c)
                lc = LfpEcg(xc, np.ones(len(xc), bool), fsE, [(t0 - theta, t1 - theta) for t0, t1 in excl_L])
                cres, _ = ecg_channel_analysis(lc, tR, tR, eeg_tmpl, rng, None)
                R["ctrl_channel"] = E.names[best_c]
                R["ctrl_strength"] = best_v
                for k_ in ("status", "p_null", "drift_ppm", "drift_se_ppm", "slide_ppm", "slide_se_ppm", "start_shift_sd_ms", "end_shift_sd_ms"):
                    R[f"ctrl_{k_}"] = cres.get(k_)
            # secondary: sliding-window slope of the best significant template, whatever the start/end gate says
            sig = [p for p in per if p.get("p_null") is not None and p["p_null"] <= P_NULL_MAX and p.get("slide_se_ppm")]
            if sig:
                bs = min(sig, key=lambda p: p["slide_se_ppm"])
                R["slide_any_channel"] = bs["channel"]
                R["slide_any_ppm"] = bs["slide_ppm"]
                R["slide_any_se_ppm"] = bs["slide_se_ppm"]
            passing = [p for p in per if p.get("status") == "pass"]
            R["ecg_channels_pass"] = [p["channel"] for p in passing]
            if len(passing) == 2:
                a_, b_ = passing
                R["ecg_two_channel_drift_diff_ppm"] = a_["drift_ppm"] - b_["drift_ppm"]
                R["ecg_two_channel_drift_diff_se_ppm"] = float(np.hypot(a_["drift_se_ppm"], b_["drift_se_ppm"]))
            R["ecg_status"] = "pass" if passing else "fail"
            if passing:
                w = np.array([1.0 / p["drift_se_ppm"] ** 2 for p in passing])
                d = np.array([p["drift_ppm"] for p in passing])
                R["ecg_drift_ppm"] = float(np.sum(w * d) / np.sum(w))
                R["ecg_drift_se_ppm"] = float(1.0 / np.sqrt(np.sum(w)))
                bestp = min(passing, key=lambda p: p["drift_se_ppm"])
                R["ecg_best_channel"] = bestp["channel"]
                for k_ in ("start_center_eeg_s", "end_center_eeg_s", "baseline_s", "start_ecg_lag_ms", "end_ecg_lag_ms",
                           "start_ecg_lag_r", "drift_ms", "drift_se_ms", "slide_ppm", "slide_se_ppm", "p_null", "start_n", "end_n", "win_s"):
                    R[f"ecg_best_{k_}"] = bestp.get(k_)
                R["ecg_offset_start_s"] = theta + bestp["start_ecg_lag_ms"] / 1000.0
                R["ecg_offset_end_s"] = theta + bestp["end_ecg_lag_ms"] / 1000.0
            for p in per:
                row = {k: v for k, v in p.items() if not k.startswith("_")}
                row.update({"sub": sub, "ses": ses})
                ecg_rows.append(row)
    R["edges"] = edges
    R["notes"] = notes
    R["seconds"] = time.time() - t_start
    return R, ecg_rows, win_rows


# ------------------------------------------------------------------------------------------
# IMU markers (APDM start and stop sync events written as 'M  1' into the EEG file)
# ------------------------------------------------------------------------------------------
def imu_marker_rows(sessions_meta):
    rows = []
    for sd, sub, ses in sessions_meta:
        stem, src = load_session_sources(sd, sub, ses)
        if not src["h5"]:
            continue
        marks = None
        if src["eeg_raw"]:
            marks = read_vmrk(f"{sd}/eeg/{stem}.vmrk")
        elif src["eeg_alg"]:
            e = sio.loadmat(src["eeg_alg"], squeeze_me=True, struct_as_record=False, variable_names=["EEG_File"])["EEG_File"]
            marks = [{"type": str(getattr(v, "type", "")), "desc": str(getattr(v, "value", "")), "pos": int(getattr(v, "sample", 0))}
                     for v in np.atleast_1d(e.vmrk)]
        m1 = [m["pos"] for m in (marks or []) if m["desc"] == "M  1"]
        with h5py.File(src["h5"], "r") as f:
            groups = [g for g in f.keys() if g != "Annotations"]
            g = sorted(groups)[0]
            t = f[g]["Time"][:].astype(np.int64)
            fs_imu = float(f[g].attrs["SampleRate"])
            ann = f["Annotations"][:] if "Annotations" in f else []
            ns = sorted({len(f[x]["Time"]) for x in groups})
        stops = [int(x[0]) for x in ann if b"stop" in bytes(x[2]).lower()]
        idx = np.arange(len(t))
        slope = np.polyfit(idx, t, 1)[0]
        row = {"sub": sub, "ses": ses, "n_m1": len(m1), "imu_fs": fs_imu, "imu_n": int(len(t)), "imu_n_per_sensor_set": ns,
               "imu_time_step_us": slope, "n_stop_annotations": len(stops)}
        if len(m1) >= 2 and stops:
            span = (m1[-1] - m1[0]) / 1000.0
            stop_s = (stops[0] - t[0]) / 1e6
            row.update({"eeg_m1_span_s": span, "imu_stop_after_first_sample_s": stop_s,
                        "imu_samples_after_stop_ms": (t[-1] - stops[0]) / 1000.0,
                        "eeg_minus_imu_ms": (span - stop_s) * 1000.0})
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------
# output helpers
# ------------------------------------------------------------------------------------------
def write_rows(path, rows, cols=None):
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) else rnd(v)) for k, v in r.items() if k in cols})
    return cols


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o


def main():
    a = get_args()
    t_all = time.time()
    raw_root = os.path.join(a.moveatlas, "datasets", "eeg", "sensefog", "raw", "SenseFOG-main").replace("\\", "/")
    doc_root = os.path.join(a.moveatlas, "datasets", "eeg", "sensefog", "raw", "_doc_repo", "Matlab_scripts", "A) Subject Files")
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    sessions = []
    for sd in sorted(glob.glob(raw_root + "/sub-*/ses-*")):
        sd = sd.replace("\\", "/")
        if os.path.basename(sd).startswith("._") or not os.path.isdir(sd):
            continue
        sub, ses = sd.split("/")[-2], sd.split("/")[-1][4:]
        sessions.append((sd, sub, ses))
    print(f"{len(sessions)} session folders under {raw_root}", flush=True)
    sps = {}
    results, ecg_rows, win_rows = [], [], []
    for i, (sd, sub, ses) in enumerate(sessions):
        if a.only and f"{sub}/{ses}" not in a.only:
            continue
        if sub not in sps:
            fp = os.path.join(doc_root, f"{sub}-datafile.m")
            sps[sub] = read_signalpoints(fp) if os.path.exists(fp) else {}
        sp = sps[sub].get(TASK.get(ses, ""), {})
        cp = os.path.join(cache, f"{sub}_{ses}.json")
        if os.path.exists(cp) and not a.force:
            with open(cp, encoding="utf-8") as f:
                c = json.load(f)
            R, er, wr = c["session"], c["ecg"], c["windows"]
            print(f"[{i + 1}/{len(sessions)}] {sub} {ses}: cached", flush=True)
        else:
            R, er, wr = process_session(a, sd, sub, ses, sp)
            R, er, wr = jsonable(R), jsonable(er), jsonable(wr)
            with open(cp, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"session": R, "ecg": er, "windows": wr}, f, indent=0)
            print(f"[{i + 1}/{len(sessions)}] {sub} {ses}: ecg {R.get('ecg_status')} "
                  f"drift {rnd(R.get('ecg_drift_ppm'), 1)} +- {rnd(R.get('ecg_drift_se_ppm'), 1)} ppm; "
                  f"reread-auth {rnd(R.get('anchor_reread_minus_auth_ms'), 2)} ms; {R.get('seconds', 0):.1f} s", flush=True)
        results.append(R)
        ecg_rows += er
        win_rows += wr
    imu = imu_marker_rows([s for s in sessions if not a.only or f"{s[1]}/{s[2]}" in a.only])
    summarize(a, results, ecg_rows, win_rows, imu, time.time() - t_all)


# ------------------------------------------------------------------------------------------
# cohort summary
# ------------------------------------------------------------------------------------------
def fmt(v, k=1):
    if v is None:
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{k}f}"


def summarize(a, results, ecg_rows, win_rows, imu, seconds):
    out = a.out
    anchored = [r for r in results if r.get("theta_auth_s") is not None]
    passing = [r for r in anchored if r.get("ecg_status") == "pass"]

    def own_bound(r):
        """Larger of the start/end bound and the sliding-window regression bound (the latter absorbs slow wander)."""
        b = abs(r["ecg_drift_ppm"]) + 1.96 * r["ecg_drift_se_ppm"]
        if r.get("ecg_best_slide_ppm") is not None and r.get("ecg_best_slide_se_ppm") is not None:
            b = max(b, abs(r["ecg_best_slide_ppm"]) + 1.96 * r["ecg_best_slide_se_ppm"])
        return b
    # per-subject pooled drift: one Percept device per patient against one EEG amplifier, so the drift is a
    # property of the device pair; heterogeneity inflates the SE (overdispersion factor sqrt(chi2/df) when > 1)
    subj_rows = []
    for s_ in sorted({r["sub"] for r in passing}):
        rs = [r for r in passing if r["sub"] == s_]
        w = np.array([1.0 / r["ecg_drift_se_ppm"] ** 2 for r in rs])
        d = np.array([r["ecg_drift_ppm"] for r in rs])
        m = float(np.sum(w * d) / np.sum(w))
        se = float(1.0 / np.sqrt(np.sum(w)))
        chi2 = float(np.sum(w * (d - m) ** 2))
        df = len(rs) - 1
        infl = float(np.sqrt(chi2 / df)) if df > 0 and chi2 > df else 1.0
        subj_rows.append({"sub": s_, "n_sessions_pass": len(rs), "sessions": [r["ses"] for r in rs], "pooled_drift_ppm": m,
                          "pooled_se_ppm": se, "chi2": chi2, "df": df, "se_inflation": infl, "pooled_se_inflated_ppm": se * infl,
                          "bound_ppm": abs(m) + 1.96 * se * infl, "min_session_ppm": float(d.min()), "max_session_ppm": float(d.max())})
    write_rows(os.path.join(out, "subject_drift.csv"), subj_rows)
    worst_sub = {s["sub"]: s["bound_ppm"] for s in subj_rows}
    worst_cohort = max(worst_sub.values(), default=None)
    rows = []
    for r in results:
        dur = r.get("duration_overlap_s")
        n_anchor = 0
        types = []
        if r.get("theta_auth_s") is not None:
            n_anchor = 1
            types.append("stim_offset_authors")
        if r.get("ecg_status") == "pass":
            n_anchor += 1
            types.append("ecg_start_end")
        if r.get("stim2_drift_ppm") is not None:
            n_anchor += 1
            types.append("second_stim_train")
        row = {
            "sub": r["sub"], "ses": r["ses"], "eeg_source": r.get("eeg_source"), "lfp_source": r.get("lfp_source"),
            "eeg_fs_hz": r.get("eeg_fs"), "eeg_n_channels": r.get("eeg_nch"), "eeg_duration_s": r.get("eeg_dur_s"),
            "eeg_n_segments": r.get("eeg_n_segments"), "eeg_n_M1": len(r.get("eeg_m1_positions") or []),
            "has_imu_h5": (r.get("files") or {}).get("h5"), "lfp_channels": r.get("lfp_channels"), "lfp_fs_hz": r.get("lfp_fs"),
            "lfp_duration_s": r.get("lfp_dur_s"), "lfp_lost_ticks": r.get("lfp_lost_ticks"), "lfp_masked_samples": r.get("lfp_masked_samples"),
            "lfp_hp_filter_hz": r.get("lfp_hp_hz"), "lfp_blanking_setting": r.get("lfp_blanking_us"), "stim_status": r.get("stim_status"),
            "n_stim_trains_2hz": r.get("n_stim_trains"), "moveatlas_lfp_agrees": r.get("moveatlas_lfp_agrees"),
            "auth_EEG_signal": r.get("auth_EEG_signal"), "auth_LFP_signal": r.get("auth_LFP_signal"), "auth_branch": r.get("auth_branch"),
            "auth_M1_used": r.get("auth_M1_used"), "theta_auth_s": r.get("theta_auth_s"), "duration_overlap_s": dur,
            "reread_eeg_channel": r.get("anchor_eeg_channel"), "reread_minus_auth_ms": r.get("anchor_reread_minus_auth_ms"),
            "reread_lfp_part_ms": r.get("anchor_lfp_part_ms"),
            "reread_eeg_part_ms": r.get("anchor_eeg_part_ms"),
            "reread_lfp_step_z": r.get("anchor_lfp_step_z"), "reread_lfp_edge_clear": is_clear(r.get("anchor_lfp_step_z")),
            "major_lfp_step_z": r.get("anchor_major_step_z"),
            "major_lfp_step_minus_eeg_spike_ms": r.get("anchor_major_step_minus_eeg_spike_on_auth_ms"),
            "reread_lfp_feature_is_major_step": r.get("anchor_lfp_feature_is_major_step"),
            "lfp_edge_rise_10_90_ms": r.get("anchor_lfp_rise_10_90_ms"), "lfp_edge_onset_to_peak_ms": r.get("anchor_lfp_onset_to_peak_ms"),
            "eeg_edge_onset_to_peak_ms": r.get("anchor_eeg_onset_to_peak_ms"), "percept_edge_delay_ms": r.get("anchor_percept_edge_delay_ms"),
            "lfp_onset_minus_eeg_onset_on_auth_ms": r.get("anchor_lfp_onset_minus_eeg_onset_on_auth_ms"),
            "ecg_eeg_channel": r.get("ecg_eeg_channel"), "ecg_r_peaks": r.get("r_n_kept"), "ecg_median_ibi_s": r.get("r_median_ibi_s"),
            "ecg_status": r.get("ecg_status"), "ecg_lfp_channels_pass": r.get("ecg_channels_pass"), "ecg_best_lfp_channel": r.get("ecg_best_channel"),
            "ecg_win_s": r.get("ecg_best_win_s"), "ecg_start_center_s": r.get("ecg_best_start_center_eeg_s"),
            "ecg_end_center_s": r.get("ecg_best_end_center_eeg_s"),
            "ecg_offset_start_s": r.get("ecg_offset_start_s"), "ecg_offset_end_s": r.get("ecg_offset_end_s"),
            "ecg_lag_start_ms": r.get("ecg_best_start_ecg_lag_ms"), "ecg_lag_end_ms": r.get("ecg_best_end_ecg_lag_ms"),
            "ecg_drift_end_minus_start_ms": r.get("ecg_best_drift_ms"), "ecg_drift_se_ms": r.get("ecg_best_drift_se_ms"),
            "ecg_drift_ppm": r.get("ecg_drift_ppm"), "ecg_drift_se_ppm": r.get("ecg_drift_se_ppm"),
            "ecg_slide_ppm": r.get("ecg_best_slide_ppm"), "ecg_slide_se_ppm": r.get("ecg_best_slide_se_ppm"),
            "slide_any_channel": r.get("slide_any_channel"), "slide_any_ppm": r.get("slide_any_ppm"), "slide_any_se_ppm": r.get("slide_any_se_ppm"),
            "ctrl_eeg_channel": r.get("ctrl_channel"), "ctrl_status": r.get("ctrl_status"), "ctrl_drift_ppm": r.get("ctrl_drift_ppm"),
            "ctrl_drift_se_ppm": r.get("ctrl_drift_se_ppm"), "ctrl_slide_ppm": r.get("ctrl_slide_ppm"),
            "ctrl_slide_se_ppm": r.get("ctrl_slide_se_ppm"),
            "ecg_two_channel_drift_diff_ppm": r.get("ecg_two_channel_drift_diff_ppm"),
            "ecg_two_channel_drift_diff_se_ppm": r.get("ecg_two_channel_drift_diff_se_ppm"),
            "stim2_baseline_s": r.get("stim2_baseline_s"), "stim2_theta_minus_anchor_ms": r.get("stim2_theta_minus_anchor_ms"),
            "stim2_drift_ppm": r.get("stim2_drift_ppm"),
            "n_anchors": n_anchor, "anchor_types": "+".join(types)}
        if dur is not None and r.get("theta_auth_s") is not None:
            if r.get("ecg_status") == "pass":
                bppm = own_bound(r)
                row["floor_basis"] = "own ECG drift bound"
            elif r["sub"] in worst_sub:
                bppm = worst_sub[r["sub"]]
                row["floor_basis"] = "pooled ECG drift bound of the same subject (same Percept device)"
            else:
                bppm = worst_cohort
                row["floor_basis"] = "largest subject-pooled ECG drift bound of the cohort"
            row["drift_bound_ppm"] = bppm
            row["admissible_lag_floor_ms"] = None if bppm is None else FLOOR_BASE_MS + bppm * dur / 1000.0
            if r.get("ecg_status") == "pass":
                # with the measured drift removed, what is left is the uncertainty of the correction at the session end
                se_c = r["ecg_drift_se_ppm"]
                if r.get("ecg_best_slide_se_ppm") is not None:
                    se_c = max(se_c, r["ecg_best_slide_se_ppm"])
                t_end_after_anchor = r["overlap_eeg_s"][1] - r["auth_eeg_anchor_s"]
                row["drift_at_session_end_ms"] = r["ecg_drift_ppm"] * t_end_after_anchor / 1000.0
                row["theta_at_session_end_s"] = r["theta_auth_s"] + r["ecg_drift_ppm"] * 1e-6 * t_end_after_anchor
                row["floor_if_drift_corrected_ms"] = FLOOR_BASE_MS + 1.96 * se_c * dur / 1000.0
            row["admissible_lag_floor_cohort_ms"] = None if worst_cohort is None else FLOOR_BASE_MS + worst_cohort * dur / 1000.0
        if r.get("theta_auth_s") is None:
            row["g15_status"] = "no_anchor"
        elif r.get("anchor_eeg_part_ms") is not None and abs(r["anchor_eeg_part_ms"]) > 4.0:
            row["g15_status"] = "hand_reading_inconsistent"
        elif r.get("ecg_status") == "pass":
            row["g15_status"] = "two_anchors_drift_measured"
        else:
            row["g15_status"] = "one_anchor_floor_applies"
        rr_ = r.get("anchor_reread_minus_auth_ms")
        if r.get("theta_auth_s") is not None and rr_ is None:
            row["reading_note"] = "anchor not re-read"
        elif rr_ is not None and not is_clear(r.get("anchor_lfp_step_z")):
            row["reading_note"] = f"no clear LFP edge near the hand-read LFP sample (step z below {LFP_EDGE_MIN_Z:g}): LFP side not checked"
        elif rr_ is not None and abs(rr_) > 4.0:
            row["reading_note"] = "re-read differs from the hand reading by more than 4 ms"
        else:
            row["reading_note"] = ""
        rows.append(row)
    write_rows(os.path.join(out, "sessions.csv"), rows)
    # per-stream time base
    tb_rows = []
    for r in results:
        for k, f_ in enumerate(r.get("lfp_timebase") or []):
            d = {"sub": r["sub"], "ses": r["ses"], "channel": (r.get("lfp_channels") or [None] * 9)[k], "lfp_source": r.get("lfp_source")}
            d.update(f_)
            tb_rows.append(d)
    write_rows(os.path.join(out, "lfp_timebase.csv"), tb_rows)
    # edges
    ed_rows = []
    for r in results:
        for e_ in r.get("edges") or []:
            d = {"sub": r["sub"], "ses": r["ses"]}
            d.update(e_)
            ed_rows.append(d)
    write_rows(os.path.join(out, "stim_edges.csv"), ed_rows)
    write_rows(os.path.join(out, "ecg_channels.csv"), ecg_rows)
    write_rows(os.path.join(out, "ecg_sliding_windows.csv"), win_rows)

    # within-subject consistency of the whole-session ECG lag (LFP template against the IPG_1 template), per side:
    # the lead geometry is fixed within a subject, so a session whose lag departs from the subject's median points
    # at an anchor that departs from the others (independent of the stimulation artefact)
    def side(ch):
        c = str(ch).upper()
        if c.endswith("_L") or "LEFT" in c:
            return "LEFT"
        if c.endswith("_R") or "RIGHT" in c:
            return "RIGHT"
        return "?"
    # the whole-session lag sits at the mean beat time; it is carried back to the anchor with the subject's pooled
    # ECG drift (0 where the subject has none), so that a real drift does not pass for an anchor error
    cons_in = [e for e in ecg_rows if e.get("lag_all_vs_IPG_1_ms") is not None and e.get("p_null") is not None
               and e["p_null"] <= P_NULL_MAX and (e.get("lag_all_vs_IPG_1_sd_ms") or 99.0) <= BOOT_SD_MAX_MS]
    pooled = {x["sub"]: x["pooled_drift_ppm"] for x in subj_rows}
    groups = {}
    for e in cons_in:
        anc = next((r.get("auth_eeg_anchor_s") for r in results if r["sub"] == e["sub"] and r["ses"] == e["ses"]), None)
        dppm = pooled.get(e["sub"], 0.0)
        e["_dt_anchor_s"] = None if anc is None else e["all_center_eeg_s"] - anc
        e["_lag_at_anchor_ms"] = e["lag_all_vs_IPG_1_ms"] - (0.0 if anc is None else dppm * e["_dt_anchor_s"] / 1000.0)
        e["_drift_used_ppm"] = dppm
        groups.setdefault((e["sub"], side(e["channel"])), []).append(e)
    cons_rows = []
    for (sub_, sd_), es in sorted(groups.items()):
        med = float(np.median([e["_lag_at_anchor_ms"] for e in es]))
        med_raw = float(np.median([e["lag_all_vs_IPG_1_ms"] for e in es]))
        for e in sorted(es, key=lambda e: e["ses"]):
            cons_rows.append({"sub": sub_, "side": sd_, "ses": e["ses"], "channel": e["channel"], "n_sessions_in_group": len(es),
                              "mean_beat_after_anchor_s": e["_dt_anchor_s"], "drift_used_ppm": e["_drift_used_ppm"],
                              "lag_whole_session_ms": e["lag_all_vs_IPG_1_ms"], "lag_sd_ms": e["lag_all_vs_IPG_1_sd_ms"],
                              "lag_r": e.get("lag_all_vs_IPG_1_r"), "lag_at_anchor_ms": e["_lag_at_anchor_ms"],
                              "group_median_ms": med, "deviation_ms": e["_lag_at_anchor_ms"] - med,
                              "deviation_uncorrected_ms": e["lag_all_vs_IPG_1_ms"] - med_raw,
                              "start_window_lag_ms": e.get("start_lag_vs_IPG_1_ms"), "start_window_lag_sd_ms": e.get("start_lag_vs_IPG_1_sd_ms")})
    write_rows(os.path.join(out, "ecg_lag_consistency.csv"), cons_rows)
    # IMU regression
    im = [r for r in imu if r.get("eeg_minus_imu_ms") is not None]
    imu_fit = {}
    if len(im) >= 3:
        x = np.array([r["eeg_m1_span_s"] for r in im])
        y = np.array([r["eeg_minus_imu_ms"] for r in im])
        p, cov = np.polyfit(x, y, 1, cov=True)
        res = y - np.polyval(p, x)
        imu_fit = {"n": len(im), "slope_ppm": p[0] * 1000.0, "slope_se_ppm": np.sqrt(cov[0, 0]) * 1000.0, "intercept_ms": p[1],
                   "intercept_se_ms": np.sqrt(cov[1, 1]), "resid_sd_ms": float(np.std(res, ddof=2)),
                   "diff_min_ms": float(y.min()), "diff_max_ms": float(y.max())}
        for r in im:
            r["eeg_minus_imu_minus_fit_ms"] = r["eeg_minus_imu_ms"] - float(np.polyval(p, r["eeg_m1_span_s"]))
    write_rows(os.path.join(out, "imu_markers.csv"), imu)
    facts = cohort_facts(results, rows, ed_rows, ecg_rows, imu, imu_fit, worst_cohort, worst_sub)
    facts["subject_drift"] = [(x["sub"], x["n_sessions_pass"], rnd(x["pooled_drift_ppm"], 1), rnd(x["pooled_se_ppm"], 1),
                               rnd(x["chi2"] / x["df"], 2) if x["df"] > 0 else None) for x in subj_rows]
    facts["floor_basis_counts"] = {k: sum(1 for r in rows if r.get("floor_basis") == k) for k in sorted({str(r.get("floor_basis")) for r in rows if r.get("floor_basis")})}
    multi = [c for c in cons_rows if c["n_sessions_in_group"] >= 2]
    facts["lag_gate_channels"] = len(cons_in)
    facts["lag_groups_multi"] = len({(c["sub"], c["side"]) for c in multi})
    facts["lag_sessions_in_multi"] = len(multi)
    facts["lag_dev_abs_q"] = [q([abs(c["deviation_ms"]) for c in multi], p) for p in (50, 90, 100)]
    facts["lag_dev_uncorr_abs_q"] = [q([abs(c["deviation_uncorrected_ms"]) for c in multi], p) for p in (50, 90, 100)]
    facts["lag_dev_gt4"] = [(c["sub"], c["side"], c["ses"], rnd(c["deviation_ms"], 1), rnd(c["lag_sd_ms"], 1)) for c in multi
                            if abs(c["deviation_ms"]) > 4]
    facts["lag_single_session_groups"] = [(c["sub"], c["side"], c["ses"]) for c in cons_rows if c["n_sessions_in_group"] == 1]
    facts["lag_subjects_checked"] = sorted({c["sub"] for c in multi})
    write_summary(a, facts, rows)
    write_provenance(a, {"task": "N0.3b", "gate": "G15 (sensefog part)", "seconds": round(seconds, 1),
                         "parameters": {"seed": SEED, "ecg_band_hz": ECG_BAND_HZ, "lags_ms": [int(LAGS_MS[0]), int(LAGS_MS[-1])],
                                        "shift_max_ms": SHIFT_MAX_MS, "ecg_lag_max_ms": ELL_MAX_MS, "win_max_s": WIN_MAX_S,
                                        "slide_w_s": SLIDE_W_S, "slide_step_s": SLIDE_STEP_S, "n_boot": N_BOOT, "n_boot_slide": N_BOOT_SLIDE,
                                        "n_null": N_NULL, "p_null_max": P_NULL_MAX, "boot_sd_max_ms": BOOT_SD_MAX_MS,
                                        "r_start_end_min": R_START_END_MIN, "min_beats_win": MIN_BEATS_WIN, "floor_base_ms": FLOOR_BASE_MS,
                                        "stim_pad_s": STIM_PAD_S, "edge_thr_z": EDGE_THR_Z, "edge_ch_pct": EDGE_CH_PCT,
                                        "edge_gap_ms": EDGE_GAP_MS, "edge_train_min_samples": EDGE_TRAIN_MIN_SAMPLES,
                                        "match_half_s": MATCH_HALF_S, "lfp_edge_min_z": LFP_EDGE_MIN_Z},
                         "outputs": sorted(f for f in os.listdir(out) if f.endswith((".csv", ".md"))),
                         "facts": jsonable(facts)})


def q(vals, p):
    v = [x for x in vals if x is not None]
    return float(np.percentile(v, p)) if v else None


def is_clear(z):
    """LFP step clear of noise (computed from the stored z, so cached sessions are gated the same way)."""
    return None if z is None else bool(z >= LFP_EDGE_MIN_Z)


def cohort_facts(results, rows, ed_rows, ecg_rows, imu, imu_fit, worst_cohort, worst_sub):
    F = {}
    F["n_session_folders"] = len(results)
    F["n_subjects"] = len({r["sub"] for r in results})
    F["n_with_authors_anchor"] = sum(1 for r in rows if r.get("theta_auth_s") is not None)
    F["n_no_anchor"] = [f"{r['sub']}/{r['ses']}" for r in rows if r.get("theta_auth_s") is None]
    F["lfp_sources"] = {k: sum(1 for r in rows if r.get("lfp_source") == k) for k in sorted({str(r.get("lfp_source")) for r in rows})}
    F["eeg_sources"] = {k: sum(1 for r in rows if r.get("eeg_source") == k) for k in sorted({str(r.get("eeg_source")) for r in rows})}
    F["eeg_fs_values"] = sorted({r.get("eeg_fs_hz") for r in rows if r.get("eeg_fs_hz")})
    F["eeg_segments_values"] = sorted({r.get("eeg_n_segments") for r in rows if r.get("eeg_n_segments") is not None})
    F["lfp_fs_values"] = sorted({r.get("lfp_fs_hz") for r in rows if r.get("lfp_fs_hz")})
    tb = [f_ for r in results for f_ in (r.get("lfp_timebase") or []) if "lost_ticks" in f_]
    F["n_streams_with_ticks"] = len(tb)
    def chan_of(r, k):
        return ((r.get("lfp_channels") or []) + [None] * 9)[k]
    F["streams_with_lost_ticks"] = [(r["sub"], r["ses"], chan_of(r, k), f_.get("lost_ticks"), f_.get("masked_samples"), f_.get("gap_at_s"))
                                    for r in results for k, f_ in enumerate(r.get("lfp_timebase") or []) if f_.get("lost_ticks")]
    F["total_lost_ticks"] = sum(f_.get("lost_ticks", 0) for f_ in tb)
    F["total_masked_samples"] = sum(f_.get("masked_samples", 0) for f_ in tb)
    F["streams_sizes_mismatch"] = sum(1 for f_ in tb if not f_.get("sizes_match"))
    F["streams_irregular_ticks"] = sum(1 for f_ in tb if f_.get("irregular_tick_steps"))
    F["streams_tick_dev_gt1"] = [(r["sub"], r["ses"], chan_of(r, k), f_.get("tick_dev_range_samples"), f_.get("count_minus_tick_samples"))
                                 for r in results for k, f_ in enumerate(r.get("lfp_timebase") or [])
                                 if (f_.get("tick_dev_range_samples") or 0) > 1.01]
    F["oversize_packets"] = [(r["sub"], r["ses"], chan_of(r, k), f_.get("oversize_packets"), rnd(r.get("auth_lfp_anchor_s"), 3))
                             for r in results for k, f_ in enumerate(r.get("lfp_timebase") or []) if f_.get("oversize_packets")]
    # the authors' cut keeps the 1-based index delay:end, so in their aligned output the anchor of the cut stream lands one sample
    # after the anchor of the other stream; the aligned EEG files of the EEG_cut branch carry it as theta
    al = [r for r in results if str(r.get("eeg_source", "")).startswith("authors_eegalg") and r.get("auth_branch") == "EEG_cut"
          and r.get("theta_auth_s") is not None]
    F["aligned_eeg_cut_theta_ms"] = sorted({rnd(r["theta_auth_s"] * 1000.0, 3) for r in al})
    F["aligned_eeg_cut_sessions"] = [f"{r['sub']}/{r['ses']}" for r in al]
    F["moveatlas_agree"] = sum(1 for r in rows if r.get("moveatlas_lfp_agrees") is True)
    F["moveatlas_disagree"] = [f"{r['sub']}/{r['ses']}" for r in rows if r.get("moveatlas_lfp_agrees") is False]
    F["stim_trains_count"] = {str(k): sum(1 for r in rows if r.get("n_stim_trains_2hz") == k) for k in sorted({r.get("n_stim_trains_2hz") for r in rows if r.get("n_stim_trains_2hz") is not None})}
    F["sessions_with_2_trains"] = [f"{r['sub']}/{r['ses']}" for r in rows if (r.get("n_stim_trains_2hz") or 0) >= 2]
    F["hp_filter_values"] = sorted({str(r.get("lfp_hp_filter_hz")) for r in rows if r.get("lfp_hp_filter_hz") is not None})
    F["blanking_values_us"] = sorted({str(r.get("lfp_blanking_setting")) for r in rows if r.get("lfp_blanking_setting") is not None})
    F["auth_branch_counts"] = {k: sum(1 for r in rows if r.get("auth_branch") == k) for k in ("EEG_cut", "LFP_cut")}
    F["overlap_s_min_median_max"] = [q([r.get("duration_overlap_s") for r in rows], p) for p in (0, 50, 100)]
    # re-read of the authors' anchor; the LFP side counts only where the feature near the hand-read sample is a clear step
    auth_found = [e for e in ed_rows if e.get("is_authors_train") and e.get("reread_minus_auth_ms") is not None]
    auth_edges = [e for e in auth_found if is_clear(e.get("lfp_step_z"))]
    F["n_anchor_reread_found"] = len(auth_edges)
    F["n_anchor_reread_missing"] = [f"{r['sub']}/{r['ses']}" for r in rows if r.get("theta_auth_s") is not None and r.get("reread_minus_auth_ms") is None]
    F["reread_no_clear_lfp_edge"] = [(e["sub"], e["ses"], rnd(e.get("lfp_step_z"), 1), rnd(e.get("lfp_amp"), 1), rnd(e.get("major_step_z"), 1),
                                      rnd(e.get("major_step_minus_eeg_spike_on_auth_ms"), 1), rnd(e.get("eeg_part_ms"), 1))
                                     for e in auth_found if not is_clear(e.get("lfp_step_z"))]
    F["lfp_step_z_clear_min"] = q([e.get("lfp_step_z") for e in auth_edges], 0)
    F["lfp_step_z_unclear_max"] = q([e.get("lfp_step_z") for e in auth_found if not is_clear(e.get("lfp_step_z"))], 100)
    dd = [e["reread_minus_auth_ms"] for e in auth_edges]
    F["reread_minus_auth_ms_q"] = [q(dd, p) for p in (0, 25, 50, 75, 100)]
    F["reread_abs_gt_4ms"] = [(e["sub"], e["ses"], rnd(e["reread_minus_auth_ms"], 1), rnd(e.get("lfp_part_ms"), 1),
                               rnd(e.get("eeg_part_ms"), 1)) for e in auth_edges if abs(e["reread_minus_auth_ms"]) > 4]
    F["reread_abs_q"] = [q([abs(v) for v in dd], p) for p in (50, 90, 100)]
    F["reread_within_2ms"] = sum(1 for v in dd if abs(v) <= 2)
    ms = [e.get("major_step_minus_eeg_spike_on_auth_ms") for e in auth_edges if is_clear(e.get("major_step_z"))]
    F["major_step_minus_spike_ms_q"] = [q(ms, p) for p in (0, 25, 50, 75, 100)]
    F["n_feature_is_major_step"] = sum(1 for e in auth_edges if e.get("lfp_feature_is_major_step"))
    F["n_feature_not_major_step"] = [(e["sub"], e["ses"], rnd(e.get("major_step_minus_eeg_spike_on_auth_ms"), 1)) for e in auth_edges if not e.get("lfp_feature_is_major_step")]
    F["reread_eeg_channels"] = {k: sum(1 for e in auth_found if e.get("eeg_channel") == k) for k in sorted({e.get("eeg_channel") for e in auth_found})}
    raw_edges = [e for e in auth_edges if next((r for r in rows if r["sub"] == e["sub"] and r["ses"] == e["ses"]), {}).get("lfp_fs_hz") == 250.0]
    F["n_edges_250hz"] = len(raw_edges)
    F["lfp_rise_10_90_ms_q"] = [q([e.get("lfp_rise_10_90_ms") for e in raw_edges], p) for p in (0, 25, 50, 75, 100)]
    F["lfp_onset_to_peak_ms_q"] = [q([e.get("lfp_onset_to_peak_ms") for e in raw_edges], p) for p in (0, 25, 50, 75, 100)]
    F["eeg_onset_to_peak_ms_q"] = [q([e.get("eeg_onset_to_peak_ms") for e in raw_edges], p) for p in (0, 25, 50, 75, 100)]
    F["percept_edge_delay_ms_q"] = [q([e.get("percept_edge_delay_ms") for e in raw_edges], p) for p in (0, 25, 50, 75, 100)]
    F["lfp_onset_minus_eeg_onset_on_auth_ms_q"] = [q([e.get("lfp_onset_minus_eeg_onset_on_auth_ms") for e in raw_edges], p) for p in (0, 25, 50, 75, 100)]
    # ECG
    anchored = [r for r in rows if r.get("theta_auth_s") is not None]
    F["ecg_status_counts"] = {k: sum(1 for r in anchored if r.get("ecg_status") == k) for k in sorted({str(r.get("ecg_status")) for r in anchored})}
    F["ecg_eeg_channels"] = {k: sum(1 for r in anchored if r.get("ecg_eeg_channel") == k) for k in sorted({str(r.get("ecg_eeg_channel")) for r in anchored})}
    ps = [r for r in anchored if r.get("ecg_status") == "pass"]
    F["ecg_pass_sessions"] = [f"{r['sub']}/{r['ses']}" for r in ps]
    F["ecg_pass_subjects"] = sorted({r["sub"] for r in ps})
    F["ecg_fail_sessions"] = [f"{r['sub']}/{r['ses']}" for r in anchored if r.get("ecg_status") != "pass"]
    F["ecg_drift_ppm_q"] = [q([r["ecg_drift_ppm"] for r in ps], p) for p in (0, 25, 50, 75, 100)]
    F["ecg_drift_se_ppm_q"] = [q([r["ecg_drift_se_ppm"] for r in ps], p) for p in (0, 50, 100)]
    F["ecg_drift_abs_z_gt2"] = [(r["sub"], r["ses"], rnd(r["ecg_drift_ppm"], 1), rnd(r["ecg_drift_se_ppm"], 1)) for r in ps
                                if abs(r["ecg_drift_ppm"]) > 2 * r["ecg_drift_se_ppm"]]
    F["ecg_drift_over_session_ms_q"] = [q([r["ecg_drift_ppm"] * r["duration_overlap_s"] / 1000 for r in ps], p) for p in (0, 50, 100)]
    if ps:
        w = np.array([1 / r["ecg_drift_se_ppm"] ** 2 for r in ps])
        d = np.array([r["ecg_drift_ppm"] for r in ps])
        F["ecg_drift_pooled_ppm"] = float(np.sum(w * d) / np.sum(w))
        F["ecg_drift_pooled_se_ppm"] = float(1 / np.sqrt(np.sum(w)))
        F["ecg_drift_chi2_df"] = [float(np.sum(w * (d - F["ecg_drift_pooled_ppm"]) ** 2)), len(ps) - 1]
    F["slide_only"] = [(r["sub"], r["ses"], r.get("slide_any_channel"), rnd(r.get("slide_any_ppm"), 1), rnd(r.get("slide_any_se_ppm"), 1))
                       for r in anchored if r.get("ecg_status") != "pass" and r.get("slide_any_ppm") is not None]
    F["ecg_lag_start_ms_q"] = [q([r.get("ecg_lag_start_ms") for r in ps], p) for p in (0, 25, 50, 75, 100)]
    ctl = [r for r in anchored if r.get("ctrl_drift_ppm") is not None and r.get("ctrl_drift_se_ppm")]
    ctlp = [r for r in ctl if r.get("ctrl_status") == "pass"]
    F["ctrl_n"] = len(ctl)
    F["ctrl_n_pass"] = len(ctlp)
    F["ctrl_channels"] = {k: sum(1 for r in ctl if r.get("ctrl_eeg_channel") == k) for k in sorted({str(r.get("ctrl_eeg_channel")) for r in ctl})}
    zc = [r["ctrl_drift_ppm"] / r["ctrl_drift_se_ppm"] for r in ctlp]
    F["ctrl_pass_z_q"] = [q(zc, p) for p in (0, 25, 50, 75, 100)]
    # calibration of the two standard errors on the control (true drift 0): mean z^2 near 1 and about 5 percent beyond |z| = 2
    zsl = [r["ctrl_slide_ppm"] / r["ctrl_slide_se_ppm"] for r in ctlp if r.get("ctrl_slide_ppm") is not None and r.get("ctrl_slide_se_ppm")]
    F["ctrl_calib_start_end"] = [len(zc), float(np.mean(np.square(zc))) if zc else None, float(np.mean(np.abs(zc) > 2)) if zc else None]
    F["ctrl_calib_slide"] = [len(zsl), float(np.mean(np.square(zsl))) if zsl else None, float(np.mean(np.abs(zsl) > 2)) if zsl else None]
    F["ctrl_pass_abs_z_gt2"] = [(r["sub"], r["ses"], rnd(r["ctrl_drift_ppm"], 1), rnd(r["ctrl_drift_se_ppm"], 1)) for r in ctlp
                                if abs(r["ctrl_drift_ppm"]) > 2 * r["ctrl_drift_se_ppm"]]
    F["ctrl_pass_drift_ppm_q"] = [q([r["ctrl_drift_ppm"] for r in ctlp], p) for p in (0, 50, 100)]
    if ctlp:
        w = np.array([1 / r["ctrl_drift_se_ppm"] ** 2 for r in ctlp])
        d = np.array([r["ctrl_drift_ppm"] for r in ctlp])
        F["ctrl_pooled_ppm"] = float(np.sum(w * d) / np.sum(w))
        F["ctrl_pooled_se_ppm"] = float(1 / np.sqrt(np.sum(w)))
        F["ctrl_chi2_df"] = [float(np.sum(w * (d - F["ctrl_pooled_ppm"]) ** 2)), len(ctlp) - 1]
    zl = [r["ecg_drift_ppm"] / r["ecg_drift_se_ppm"] for r in ps]
    F["lfp_pass_z_q"] = [q(zl, p) for p in (0, 25, 50, 75, 100)]
    F["two_channel"] = [(r["sub"], r["ses"], rnd(r["ecg_two_channel_drift_diff_ppm"], 1), rnd(r["ecg_two_channel_drift_diff_se_ppm"], 1))
                        for r in anchored if r.get("ecg_two_channel_drift_diff_ppm") is not None]
    F["ecg_slide_ppm_q"] = [q([r.get("ecg_slide_ppm") for r in ps], p) for p in (0, 50, 100)]
    F["worst_drift_bound_ppm_cohort"] = worst_cohort
    F["worst_drift_bound_ppm_by_subject"] = worst_sub
    F["floor_ms_q"] = [q([r.get("admissible_lag_floor_ms") for r in anchored], p) for p in (0, 50, 100)]
    F["floor_cohort_ms_q"] = [q([r.get("admissible_lag_floor_cohort_ms") for r in anchored], p) for p in (0, 50, 100)]
    F["floor_corrected_ms_q"] = [q([r.get("floor_if_drift_corrected_ms") for r in anchored], p) for p in (0, 50, 100)]
    F["drift_at_end_ms_q"] = [q([r.get("drift_at_session_end_ms") for r in anchored], p) for p in (0, 50, 100)]
    F["n_anchor_counts"] = {str(k): sum(1 for r in anchored if r.get("n_anchors") == k) for k in (1, 2, 3)}
    F["g15_status_counts"] = {k: sum(1 for r in rows if r.get("g15_status") == k) for k in sorted({str(r.get("g15_status")) for r in rows})}
    F["g15_inconsistent"] = [(r["sub"], r["ses"], rnd(r.get("reread_eeg_part_ms"), 1), rnd(r.get("reread_minus_auth_ms"), 1),
                              rnd(r.get("major_lfp_step_minus_eeg_spike_ms"), 1)) for r in rows if r.get("g15_status") == "hand_reading_inconsistent"]
    subs_inc = {x[0] for x in F["g15_inconsistent"]}
    F["g15_inconsistent_subject_ecg"] = [(r["sub"], r["ses"], r.get("ecg_status")) for r in rows if r["sub"] in subs_inc]
    F["n_eeg_part_within_2ms"] = sum(1 for r in rows if r.get("reread_eeg_part_ms") is not None and abs(r["reread_eeg_part_ms"]) <= 2.0)
    F["n_eeg_part"] = sum(1 for r in rows if r.get("reread_eeg_part_ms") is not None)
    # concatenated-axis check
    F["concat_checks"] = [(e["sub"], e["ses"], e["channel"], rnd(e.get("concat_axis_end_shift_ms"), 1), rnd(e.get("rebuilt_axis_end_shift_ms"), 2))
                          for e in ecg_rows if e.get("concat_axis_end_shift_ms") is not None and e.get("status") == "pass"]
    # second stimulation train
    F["stim2"] = [(r["sub"], r["ses"], rnd(r.get("stim2_baseline_s"), 2), rnd(r.get("stim2_theta_minus_anchor_ms"), 2), rnd(r.get("stim2_drift_ppm"), 1))
                  for r in rows if r.get("stim2_drift_ppm") is not None]
    # IMU
    F["imu_sessions"] = len(imu)
    F["imu_with_start_stop"] = sum(1 for r in imu if r.get("eeg_minus_imu_ms") is not None)
    F["imu_samples_after_stop_ms_q"] = [q([r.get("imu_samples_after_stop_ms") for r in imu], p) for p in (0, 50, 100)]
    F["imu_time_step_us_q"] = [q([r.get("imu_time_step_us") for r in imu], p) for p in (0, 50, 100)]
    F["imu_fit"] = imu_fit
    return F


def write_summary(a, F, rows):
    L = []
    L.append("# sensefog synchronisation anchor (task N0.3b, gate G15, sensefog part)")
    L.append("")
    L.append("Written by scripts/probe_sync_sensefog.py; every number below is read from the files of this directory "
             "(sessions.csv, lfp_timebase.csv, stim_edges.csv, ecg_channels.csv, ecg_sliding_windows.csv, ecg_lag_consistency.csv, "
             "subject_drift.csv, imu_markers.csv, _provenance.json). Descriptive only; no coupling statistic. theta is LFP time minus EEG time for one event; "
             "LFP time is the rebuilt Percept sample index over 250 Hz, EEG time the BrainAmp sample index over 1000 Hz.")
    L.append("")
    L.append("## 1. Inventory")
    L.append("")
    L.append(f"- Session folders: {F['n_session_folders']} over {F['n_subjects']} subjects. EEG sources: "
             + ", ".join(f"{k} {v}" for k, v in F["eeg_sources"].items()) + ". LFP sources: "
             + ", ".join(f"{k} {v}" for k, v in F["lfp_sources"].items()) + ".")
    L.append(f"- EEG sampling rates (from the header): {F['eeg_fs_values']} Hz; New Segment markers per raw file: {F['eeg_segments_values']} "
             "(one continuous segment, so the EEG clock is the sample counter). EMG and the two IPG channels are channels of the same "
             "BrainVision file: they share the EEG clock by construction.")
    L.append(f"- LFP rates: {F['lfp_fs_values']} Hz; Percept high-pass setting (TherapySnapshot) values: {F['hp_filter_values']} Hz; sensing blanking settings as shipped "
             f"(key=value): {F['blanking_values_us']}. The JSON declares no low-pass corner.")
    L.append(f"- Stimulation trains in the 2 Hz BrainSenseLfp stream per session: {F['stim_trains_count']}; sessions with two trains: "
             f"{F['sessions_with_2_trains']}.")
    L.append(f"- Authors' alignment: sessions with both hand-read samples and data to apply them: {F['n_with_authors_anchor']}; branch counts "
             f"{F['auth_branch_counts']}; no anchor: {F['n_no_anchor']}. The EEG sample is counted from the first 'M  1' marker when the session "
             "has an IMU file (1.5_Data_Alignment.m), the LFP sample on the authors' 1000 Hz resample of the concatenated Percept samples.")
    L.append(f"- The authors' cut keeps the 1-based index delay:end (1.5_Data_Alignment.m), so in their aligned output the anchor of the cut "
             f"stream lands one sample after the anchor of the other stream. theta here is taken from the two hand-read samples, not from "
             f"that cut; the aligned EEG files of the EEG_cut branch ({F['aligned_eeg_cut_sessions']}) carry the one sample as theta (ms) "
             f"{F['aligned_eeg_cut_theta_ms']}.")
    L.append(f"- LFP to EEG overlap per session (s): min {fmt(F['overlap_s_min_median_max'][0])}, median {fmt(F['overlap_s_min_median_max'][1])}, "
             f"max {fmt(F['overlap_s_min_median_max'][2])}.")
    L.append("")
    L.append("## 2. Percept time base")
    L.append("")
    L.append(f"- Streams with TicksInMses: {F['n_streams_with_ticks']}. Lost ticks in total: {F['total_lost_ticks']}; masked samples: "
             f"{F['total_masked_samples']}; streams affected (sub, ses, channel, lost ticks, masked samples, gap at s): {F['streams_with_lost_ticks']}.")
    L.append(f"- Streams whose packet sizes do not sum to the sample count: {F['streams_sizes_mismatch']}; streams with a tick step that is not a "
             f"multiple of the nominal step: {F['streams_irregular_ticks']}; streams whose packet accounting departs from the tick clock by more than "
             f"one sample (sub, ses, channel, range in samples, samples in excess of the tick clock): {F['streams_tick_dev_gt1']}.")
    L.append(f"- Packets holding more samples than one tick step can carry (sub, ses, channel, [(packet index, tick time s, size)], authors' "
             f"LFP anchor s): {F['oversize_packets']}. Where these packets precede the anchor, the samples after them keep a constant offset to "
             "the tick clock, which the anchor absorbs; only the tick time of the samples inside them is ambiguous.")
    L.append(f"- Agreement with the MoveAtlas processed LFP (length and masked count): {F['moveatlas_agree']} sessions agree; disagree: {F['moveatlas_disagree']}.")
    L.append("- The ticks and the samples come from the same device clock, so this rebuild checks packet accounting only; the drift against "
             "the EEG needs the anchors of section 3.")
    if F["concat_checks"]:
        L.append(f"- Effect of the rebuild where ticks were lost (sub, ses, LFP channel with a passing ECG template, end-window ECG shift "
                 f"against the start window on the concatenated axis in ms, on the rebuilt axis in ms): {F['concat_checks']}. The authors' pipeline concatenates the samples, so their timeline after a gap is shifted "
                 "by the masked duration.")
    L.append("")
    L.append("## 3. Anchors per session")
    L.append("")
    L.append(f"- Re-read of the authors' anchor (the stimulation-offset transition artefact, located automatically in both streams): EEG "
             f"transition spike and a clear LFP step (step z at least {LFP_EDGE_MIN_Z:g}) found in {F['n_anchor_reread_found']} sessions; "
             f"not re-read: {F['n_anchor_reread_missing']}. Re-read minus hand-read theta (ms) "
             f"quantiles 0/25/50/75/100: {[fmt(v, 2) for v in F['reread_minus_auth_ms_q']]}. EEG channel carrying the spike (every session "
             f"where it was found): {F['reread_eeg_channels']}. This re-reads the same artefact, so it checks the hand reading and is not an "
             "independent anchor.")
    L.append(f"- No clear LFP step within +-{MATCH_HALF_S * 1000:g} ms of the hand-read LFP sample (the strongest step there is at noise level "
             f"or a minor wiggle, so the LFP side of the reading is not checked; sub, ses, step z, step amplitude, major step z in the "
             f"offset window, major step minus EEG spike on the authors' alignment ms, EEG part ms): {F['reread_no_clear_lfp_edge']}. "
             f"Largest step z among these: {fmt(F['lfp_step_z_unclear_max'], 1)}; smallest among the clear ones: {fmt(F['lfp_step_z_clear_min'], 1)}.")
    L.append(f"- |re-read minus hand-read| (ms) median, 90th percentile, max: {[fmt(v, 2) for v in F['reread_abs_q']]}; within 2 ms: "
             f"{F['reread_within_2ms']} of {F['n_anchor_reread_found']} sessions.")
    L.append(f"- Sessions where the re-read differs from the hand reading by more than 4 ms (sub, ses, difference, LFP part, EEG part, ms): "
             f"{F['reread_abs_gt_4ms']}. An EEG part far from 0 means the hand-read EEG sample is not the isolated transition spike "
             "that follows the pulse train.")
    L.append(f"- The LFP offset artefact is composite: besides the feature at the hand-read sample there can be a larger device step. "
             f"Among the sessions with a clear LFP step at the hand-read sample, that step is the major step of the offset window in "
             f"{F['n_feature_is_major_step']}; elsewhere (sub, ses, major step minus EEG spike on the authors' alignment, ms): "
             f"{F['n_feature_not_major_step']}. Major step minus EEG spike (ms, clear steps only), quantiles "
             f"0/25/50/75/100: {[fmt(v, 1) for v in F['major_step_minus_spike_ms_q']]}. So the LFP counterpart of the EEG spike is not "
             "fixed by the artefact shape alone.")
    L.append(f"- ECG channel chosen in the EEG file (strongest regular ECG): {F['ecg_eeg_channels']}.")
    L.append(f"- ECG anchor status: {F['ecg_status_counts']}. Pass rule: R-locked LFP template peak beyond a circular-shift null of the beat "
             f"train ({N_NULL} shifts, p <= {P_NULL_MAX}), bootstrap SD of the start and end shifts at most {BOOT_SD_MAX_MS} ms, start and end "
             f"templates correlated at least {R_START_END_MIN}, at least {MIN_BEATS_WIN} beats per window.")
    L.append(f"- Passing sessions ({len(F['ecg_pass_sessions'])}, subjects {F['ecg_pass_subjects']}): {F['ecg_pass_sessions']}.")
    L.append(f"- Failing or not measurable: {F['ecg_fail_sessions']}.")
    L.append(f"- ECG drift (ppm) over passing sessions, quantiles 0/25/50/75/100: {[fmt(v) for v in F['ecg_drift_ppm_q']]}; standard error min/median/max "
             f"{[fmt(v) for v in F['ecg_drift_se_ppm_q']]}; drift accumulated over the session (ms) min/median/max "
             f"{[fmt(v, 2) for v in F['ecg_drift_over_session_ms_q']]}.")
    if "ecg_drift_pooled_ppm" in F:
        L.append(f"- Inverse-variance pooled drift: {fmt(F['ecg_drift_pooled_ppm'])} ppm (SE {fmt(F['ecg_drift_pooled_se_ppm'])}); heterogeneity chi-square "
                 f"{fmt(F['ecg_drift_chi2_df'][0])} on {F['ecg_drift_chi2_df'][1]} df. Sessions with |drift| above twice its SE: {F['ecg_drift_abs_z_gt2']}.")
    L.append(f"- Sliding-window slope (ppm) min/median/max over passing sessions: {[fmt(v) for v in F['ecg_slide_ppm_q']]}.")
    cs, cl = F["ctrl_calib_start_end"], F["ctrl_calib_slide"]
    L.append(f"- Calibration of the standard errors on the negative control below (true drift 0; expected mean z^2 1 and about 0.05 "
             f"beyond |z| = 2): start/end drift over {cs[0]} passing controls, mean z^2 {fmt(cs[1], 2)}, share beyond |z| = 2 {fmt(cs[2], 2)}; "
             f"sliding-window slope over {cl[0]}, mean z^2 {fmt(cl[1], 2)}, share {fmt(cl[2], 2)}. The OLS standard error of the sliding "
             f"slope treats overlapping {SLIDE_W_S:g} s windows (step {SLIDE_STEP_S:g} s) and slow template wander as independent, so it is "
             "too small: the sliding slope is a second point estimate, its SE is not used for inference, and it enters the drift bound "
             "only as the larger of the two bounds, which is never below the calibrated start/end bound.")
    L.append(f"- Sessions failing the start/end gate but with a significant LFP ECG template (p_null <= {P_NULL_MAX}): sliding-window slope of "
             f"the best such channel (sub, ses, channel, ppm, OLS SE, too small per the calibration above), secondary information not "
             f"used for the floor: {F['slide_only']}.")
    L.append(f"- ECG lag of the LFP template against the EEG ECG template at session start, on the authors' alignment (ms), quantiles 0/25/50/75/100: "
             f"{[fmt(v) for v in F['ecg_lag_start_ms_q']]}. This lag mixes the lead geometry (the QRS peaks at different times in different "
             "leads) with the filter delays, so it is not a clock offset; only its change from start to end is.")
    L.append(f"- Negative control on the EEG clock (the same start/end analysis on the scalp EEG channel with the strongest R-locked template, "
             f"whose true drift is 0): computed in {F['ctrl_n']} sessions, passing the same gate in {F['ctrl_n_pass']}; channels {F['ctrl_channels']}. "
             f"Passing controls: drift/SE quantiles 0/25/50/75/100 {[fmt(v, 2) for v in F['ctrl_pass_z_q']]}, drift (ppm) min/median/max "
             f"{[fmt(v) for v in F['ctrl_pass_drift_ppm_q']]}, pooled {fmt(F.get('ctrl_pooled_ppm'))} ppm (SE {fmt(F.get('ctrl_pooled_se_ppm'))}), "
             f"chi-square {fmt((F.get('ctrl_chi2_df') or [None])[0])} on {(F.get('ctrl_chi2_df') or [None, None])[1]} df; controls with |drift| above twice "
             f"the SE: {F['ctrl_pass_abs_z_gt2']}. For comparison, LFP drift/SE over passing sessions: {[fmt(v, 2) for v in F['lfp_pass_z_q']]}.")
    L.append(f"- Two LFP hemispheres share the Percept clock; where both pass, drift difference (sub, ses, ppm, SE): {F['two_channel']}.")
    L.append(f"- Within-subject check of the hand reading by the ECG (whole-session lag of each LFP channel's ECG template against the IPG_1 "
             f"ECG template on the same beats, carried back to the anchor time with the subject's pooled ECG drift; channels "
             f"with p_null <= {P_NULL_MAX} and bootstrap SD <= {BOOT_SD_MAX_MS} ms: {F['lag_gate_channels']}; subject-side groups with two or more "
             f"sessions: {F['lag_groups_multi']}, holding {F['lag_sessions_in_multi']} session-channels, subjects {F['lag_subjects_checked']}): "
             f"|deviation from the group median| (ms) median, 90th percentile, max {[fmt(v, 2) for v in F['lag_dev_abs_q']]} "
             f"(without the drift correction {[fmt(v, 2) for v in F['lag_dev_uncorr_abs_q']]}); deviations above "
             f"4 ms (sub, side, ses, deviation ms, SD ms): {F['lag_dev_gt4']}. Single-session groups (no check possible): "
             f"{F['lag_single_session_groups']}.")
    L.append(f"- Second stimulation train (sub, ses, baseline s, theta change ms, drift ppm): {F['stim2']}.")
    L.append(f"- Drift pooled per subject (one Percept device per patient; inverse-variance over passing sessions; sub, n, ppm, SE, chi2/df): "
             f"{F['subject_drift']}. Bound per subject = |pooled| + 1.96 SE, the SE inflated by sqrt(chi2/df) when chi2 > df: "
             f"{ {k: rnd(v, 1) for k, v in F['worst_drift_bound_ppm_by_subject'].items()} }; largest (used where the subject has no "
             f"passing session): {fmt(F['worst_drift_bound_ppm_cohort'])} ppm.")
    L.append(f"- Admissible-lag floor, 4 ms plus the drift bound times the overlap duration (ms), min/median/max: {[fmt(v, 2) for v in F['floor_ms_q']]} "
             f"(bound: the session's own where its ECG passes, taken as the larger of |start/end drift| + 1.96 SE and |sliding slope| + 1.96 SE; "
             f"else the subject's pooled bound; else the largest subject bound); with the largest subject bound for every session: "
             f"{[fmt(v, 2) for v in F['floor_cohort_ms_q']]}. Floor basis counts: {F['floor_basis_counts']}.")
    L.append(f"- Where the ECG passes, the drift accumulated from the anchor to the session end (ms) min/median/max "
             f"{[fmt(v, 2) for v in F['drift_at_end_ms_q']]}; if that drift is removed, the floor becomes 4 ms plus 1.96 SE of the correction "
             f"over the session (ms) min/median/max {[fmt(v, 2) for v in F['floor_corrected_ms_q']]}.")
    L.append(f"- Anchors per anchored session: {F['n_anchor_counts']}.")
    L.append("")
    L.append("## 4. Percept artefact edge")
    L.append("")
    L.append(f"- Edges measured on 250 Hz raw LFP (authors' train, clear LFP step only): {F['n_edges_250hz']}. LFP rise 10 to 90 percent (ms), quantiles 0/25/50/75/100: "
             f"{[fmt(v, 2) for v in F['lfp_rise_10_90_ms_q']]}; LFP onset (20 to 80 percent line extrapolated to baseline) to peak: "
             f"{[fmt(v, 2) for v in F['lfp_onset_to_peak_ms_q']]}; EEG spike onset to peak: {[fmt(v, 2) for v in F['eeg_onset_to_peak_ms_q']]}.")
    L.append(f"- Edge delay of the Percept chain relative to the EEG chain (LFP onset-to-peak minus EEG onset-to-peak, ms): "
             f"{[fmt(v, 2) for v in F['percept_edge_delay_ms_q']]}. On the authors' alignment, LFP onset minus EEG onset (ms): "
             f"{[fmt(v, 2) for v in F['lfp_onset_minus_eeg_onset_on_auth_ms_q']]}; a negative value means the LFP response starts before the "
             "EEG event, which a causal chain cannot do: the peak-to-peak alignment absorbs the Percept rise into the offset.")
    L.append("- A pure transport latency of either chain is indistinguishable from a clock offset with electrical anchors; the delay above is "
             "the part visible in the edge shape, a lower bound under a causal response.")
    L.append("")
    L.append("## 5. IMU to EEG (APDM start and stop markers)")
    L.append("")
    fi = F["imu_fit"]
    L.append(f"- IMU files: {F['imu_sessions']}; with two 'M  1' markers in the EEG and a stop annotation in the h5: {F['imu_with_start_stop']}. "
             f"IMU samples recorded after the stop event (ms) min/median/max: {[fmt(v, 1) for v in F['imu_samples_after_stop_ms_q']]}; "
             f"h5 Time column step (us) min/median/max: {[fmt(v, 3) for v in F['imu_time_step_us_q']]} (nominal, derived from the sample count).")
    if fi:
        L.append(f"- EEG marker span minus IMU stop time (ms): min {fmt(fi['diff_min_ms'], 1)}, max {fmt(fi['diff_max_ms'], 1)}. Linear fit on the span "
                 f"over {fi['n']} sessions: slope {fmt(fi['slope_ppm'])} ppm (SE {fmt(fi['slope_se_ppm'])}), intercept {fmt(fi['intercept_ms'], 1)} ms "
                 f"(SE {fmt(fi['intercept_se_ms'], 1)}), residual SD {fmt(fi['resid_sd_ms'], 1)} ms. The slope is the APDM to BrainAmp drift if it is "
                 "common to all sessions; the intercept is the difference between the stop and start marker latencies.")
    L.append("")
    L.append("## Reading")
    L.append("")
    L.append(f"- Coverage: all {F['n_session_folders']} session folders were processed in full (every sample of every stream); no subset.")
    L.append(f"- Descriptive G15 status per session (sessions.csv, g15_status): {F['g15_status_counts']}.")
    L.append(f"- The hand-read EEG sample sits within 2 ms of the re-read transition spike in {F['n_eeg_part_within_2ms']} of {F['n_eeg_part']} sessions. "
             f"Inconsistent hand reading (sub, ses, EEG part ms, re-read minus hand-read ms, major LFP step minus EEG spike ms): "
             f"{F['g15_inconsistent']}. There the hand-read EEG sample lies inside the pulse train; pairing the transition spike with the "
             "hand-read LFP feature moves theta by the second number, pairing it with the major LFP step (the pattern of most sessions) by the "
             f"third. ECG status of the same subjects' sessions (sub, ses, status): {F['g15_inconsistent_subject_ecg']}.")
    L.append("- The ECG anchor is independent of the stimulation artefact: it uses the heart, seen by the EEG-file channels and, in some leads, "
             "by the Percept. Where it passes, the drift is measured; elsewhere only the authors' single anchor exists and the floor applies.")
    L.append("- Files: sessions.csv (one row per session folder), lfp_timebase.csv, stim_edges.csv, ecg_channels.csv, ecg_sliding_windows.csv, "
             "imu_markers.csv, ecg_lag_consistency.csv, subject_drift.csv.")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
