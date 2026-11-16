"""oxford_dystonia_lfp: readers, inventory, ramp step table and the oxford part of gate G15.

Tasks N0.1 and N0.2 of docs/EXECUTION_PLAN.md (addendum 1, stage N0 and gate G15).

Source (read-only): MoveAtlas datasets/eeg/oxford_dystonia_lfp/raw/, three zips
(P1,P2.zip, P3,P4.zip, P5,P6,P7.zip) holding one MATLAB .mat per recording. The zip members
are decompressed in memory, never extracted to disk. Each .mat holds a struct SmrData with
Fs (one rate per file), WvTits (channel labels), WvData (samples x channels) and, in most
files, EvTits and EvData (keyboard marks, in seconds). The fields FileName and FilePath are
never read.

Reader: read_mat() opens a v7.3 file (HDF5) with h5py and any older version with
scipy.io.loadmat, and returns labels, rate, data, marks and the MAT header.

Outputs (reports/oxford_lfp/ by default):
  inventory.csv            one row per .mat: patient, hemisphere, kind, channels, rate, duration
  channels.csv             one row per channel: category, stored rate, unit, amplitude summary;
                           for a repeated label, identical_to_index compares all samples, while
                           r_with_earlier_same_label is a Pearson r on every 16th sample (descriptive)
  counts.csv               files and hours per kind, per patient, per hemisphere (hours summed from
                           samples and rate, not from the rounded per-file duration_h)
  ramp_blocks.csv          every command segment (ON with amplitude, OFF) of every ramp file,
                           with the stimulation artefact measured on the LFP
  ramp_file_summary.csv    one row per ramp file: step pattern, block durations, G15 result
  nonramp_stim_blocks.csv  the same segment table for non-ramp files with a stimulation channel
  g15_transitions.csv      each ON block: command edges against the first and last artefact pulse
  g15_files.csv            per ramp file: pass or fail of the one-sample rule, lag range
  movement_channels.csv    accelerometer, EMG and auxiliary channels per file, with a
                           spectral profile over stimulation-free segments
  movement_hemispheres.csv movement channels per patient and hemisphere
  addendum_checks.csv      each oxford statement of the addendum, measured value, verdict
  summary.md               the same in prose
  cache/                   one JSON per .mat (resume; gitignored)
  _provenance.json

Artefact method (descriptive, fixed parameters, no randomness): on each LFP channel the
absolute first difference |dx| is computed. The artefact channel is the LFP channel with the
largest ratio of the 99.5th percentile of |dx| inside stimulation (command ON, 1 s trimmed at
both ends) to the median of |dx| outside stimulation (command OFF, 1 s trimmed). The pulse
period comes from the autocorrelation of the thresholded |dx| inside the longest ON block.
Per ON block the pulse threshold is max(0.25 x the 99th percentile of |dx| in the block core,
20 x the OFF median); blocks shorter than 2.5 s take the lowest threshold of the long blocks.
Pulses closer than 0.7 period are merged. The first artefact pulse of a block is the first
pulse followed by 3 intervals within 0.75 to 1.3 periods, moved back over at most 5
preceding samples that exceed 10 x the OFF median; the last pulse is defined symmetrically.
G15 (oxford): an ON transition passes when the first artefact pulse lies within one sample of
the command rise; a file passes when every ON transition passes.

Descriptive only: no statistic of questions Q1 or Q2 is computed.

Run: python scripts/probe_oxford_lfp.py [--moveatlas <moveatlas>] [--out DIR]
         [--force] [--only SUBSTRING]
"""
import csv
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_args, write_provenance  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_REL = "datasets/eeg/oxford_dystonia_lfp/raw"
CACHE_VERSION = "probe_oxford_lfp v6"

# Fixed parameters of the artefact method.
TRIM_S = 1.0            # trimmed at both ends of a segment to define cores
CORE_MIN_S = 0.5        # minimum core length for a block-specific threshold
THR_FRAC = 0.25         # fraction of the block-core 99th percentile of |dx|
THR_MAD = 20.0          # floor, in multiples of the OFF median of |dx|
REFINE_MAD = 10.0       # onset refinement level, in multiples of the OFF median
REFINE_MAX = 5          # at most this many samples moved back
REFR_FRAC = 0.7         # pulses closer than this fraction of the period are merged
ITI_LO, ITI_HI = 0.75, 1.3
N_REGULAR = 3
PERIOD_LAG_HZ = (5.0, 500.0)
OFF_MARGIN_S = 0.5      # stray-pulse search starts this long after a command fall
GAP_MIN_S = 0.1         # a pause in the pulse train longer than this is counted
EDGE_PULSES = 10        # first and last pulses compared with the block median (ramp-up, ramp-down)
INTERRUPT_S = 0.5       # an ON block is interrupted when its pulse train stops this long before
                        # the command fall or pauses this long inside the block
SHORT_BLOCK_S = 5.0
SPEC_NPERSEG = 4096
BANDS = [("frac_0p5_20hz", 0.5, 20.0), ("frac_20_500hz", 20.0, 500.0), ("frac_500_2048hz", 500.0, 2049.0)]

EEG_LABELS = {"C3", "Cz", "C4", "CP3", "CPz", "CPZ", "CP4", "FCz", "FC3", "FC4", "Pz", "F3", "F4", "P3", "P4"}
EMG_LABELS = {
    "EMG1": ("muscle not named in label", ""), "EMG2": ("muscle not named in label", ""),
    "RSpc": ("splenius capitis (inferred from label)", "R"), "LSpc": ("splenius capitis (inferred from label)", "L"),
    "RSpC": ("splenius capitis (inferred from label)", "R"), "LSpC": ("splenius capitis (inferred from label)", "L"),
    "SplL": ("splenius (label; README example)", "L"), "SCML": ("sternocleidomastoid (label; README example)", "L"),
    "LSCM": ("sternocleidomastoid (inferred from label)", "L"), "RSCM": ("sternocleidomastoid (inferred from label)", "R"),
    "RTr": ("trapezius (inferred from label)", "R"),
}
ACC_GROUPS = {
    "Acc": ("location not in label; README: accelerometers on the forehead or the vertex", True),
    "AcH": ("head (label H)", True),
    "AcV": ("vertex (label V)", True),
    "AcS": ("location not documented (label S)", False),
    "Ac1": ("location not documented (label 1)", False),
}


# ----------------------------------------------------------------------------- reader

HDF5_SIG = b"\x89HDF\r\n\x1a\n"


def _h5_char(ds):
    a = np.asarray(ds[()]).ravel()
    return "".join(chr(int(c)) for c in a if int(c) != 0)


def _h5_value(obj):
    if int(obj.attrs.get("MATLAB_empty", 0)):
        return np.zeros(0)
    if obj.attrs.get("MATLAB_class", b"") == b"char":
        return _h5_char(obj)
    return np.asarray(obj[()])


def _read_h5(raw, header):
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as f:
        S = f["SmrData"]
        fields = sorted(S.keys())
        fs = float(np.asarray(S["Fs"][()]).ravel()[0])
        labels = [str(_h5_value(f[r])) for r in np.asarray(S["WvTits"][()]).ravel()]
        data = np.asarray(S["WvData"][()], dtype=np.float64)
        marks = {}
        if "EvTits" in S and "EvData" in S:
            for rt, rd in zip(np.asarray(S["EvTits"][()]).ravel(), np.asarray(S["EvData"][()]).ravel()):
                name = str(_h5_value(f[rt]))
                val = _h5_value(f[rd])
                marks[name] = [float(v) for v in np.asarray(val, dtype=np.float64).ravel()]
    return fs, labels, data, marks, fields, "h5py"


def _read_v5(raw, header):
    from scipy.io import loadmat
    m = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False, variable_names=["SmrData"])
    S = m["SmrData"]
    fields = sorted(S._fieldnames)
    fs = float(np.ravel(S.Fs)[0])
    labels = [str(x).strip() for x in np.atleast_1d(S.WvTits).ravel()]
    data = np.asarray(S.WvData, dtype=np.float64)
    marks = {}
    if "EvTits" in fields and "EvData" in fields:
        names = [str(x) for x in np.atleast_1d(S.EvTits).ravel()]
        vals = S.EvData
        if len(names) == 1:
            vals = [vals]
        for name, v in zip(names, vals):
            marks[name] = [float(x) for x in np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel()]
    return fs, labels, data, marks, fields, "scipy.io.loadmat"


def read_mat(raw):
    """Open a .mat given as bytes. Returns a dict with labels, fs, data (samples x channels),
    marks, header, reader. v7.3 goes through h5py, older versions through scipy.io.loadmat."""
    header = raw[:116].decode("latin-1", "replace").rstrip(" \x00")
    if header.startswith("MATLAB 7.3") or raw[512:520] == HDF5_SIG:
        fs, labels, data, marks, fields, reader = _read_h5(raw, header)
    else:
        fs, labels, data, marks, fields, reader = _read_v5(raw, header)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] != len(labels) and data.shape[0] == len(labels):
        data = data.T
    if data.shape[1] != len(labels):
        raise ValueError(f"WvData shape {data.shape} does not match {len(labels)} labels")
    m = re.match(r"MATLAB ([0-9.]+) MAT-file", header)
    return {"labels": labels, "fs": fs, "data": data, "marks": marks, "fields": fields,
            "reader": reader, "mat_version": m.group(1) if m else "unknown",
            "created": (re.search(r"Created on: (.*?)(?: HDF5|$)", header).group(1).strip()
                        if "Created on:" in header else "")}


# ----------------------------------------------------------------------------- naming

def classify(label):
    if re.fullmatch(r"[LR][1-8]", label):
        return "lfp_unipolar"
    if re.fullmatch(r"[LR][1-8][1-8]", label):
        return "lfp_bipolar"
    if label in EEG_LABELS:
        return "eeg"
    if label in EMG_LABELS:
        return "emg"
    if re.fullmatch(r"(Acc|AcH|AcS|AcV|Ac1)[XYZxyz]", label):
        return "accelerometer"
    if re.fullmatch(r"AUX\d*", label) or label == "Mic":
        return "aux"
    if re.fullmatch(r"(Am|Cur)\d", label):
        return "stim_command"
    if re.fullmatch(r"(Rs|Res)\d", label):
        return "stim_telemetry"
    if re.fullmatch(r"[FS]-[LR][1-8]", label) or re.fullmatch(r"f[LR][1-8]{1,2}", label):
        return "derived"
    return "other"


def unit_of(label, cat):
    if cat == "stim_command":
        return "uA (inferred: values 500 to 4500 in steps of 500 match the README 0.5 mA steps to 4.5 mA)"
    return "not stored in file"


def parse_name(member):
    base = member.split("/")[-1]
    folder = member.split("/")[-2] if "/" in member else ""
    m = re.match(r"(?:Participant|Patient)(\d+)_(Left|Right)", folder)
    patient = f"P{m.group(1)}" if m else ""
    hemi = {"Left": "L", "Right": "R"}.get(m.group(2), "") if m else ""
    low = base.lower()
    if "_single" in low:
        kind = "single-pulse"
    elif "_paired" in low:
        kind = "paired-pulse"
    elif "_testing" in low or "_dbs" in low:
        kind = "ramp"
    else:
        kind = "other"
    lv = re.match(r"[A-Z]\d+_([LR])(\d)_", base)
    level = int(lv.group(2)) if lv else None
    prefix = base.split("_")[0]
    note = ""
    if patient and prefix != patient:
        note = f"file prefix {prefix} differs from folder patient {patient}"
    return base, patient, hemi, kind, level, note


# ----------------------------------------------------------------------------- helpers

def runs_of(x):
    d = np.flatnonzero(x[1:] != x[:-1]) + 1
    st = np.concatenate(([0], d)).astype(np.int64)
    en = np.concatenate((d, [len(x)])).astype(np.int64)
    return st, en, x[st]


def absdiff(x):
    dx = np.empty_like(x)
    dx[0] = 0.0
    np.abs(np.diff(x), out=dx[1:])
    return dx


def estimate_period(dx, lo, hi, thr, fs):
    ind = (dx[lo:hi] > thr).astype(np.float64)
    if ind.sum() < 20:
        return None
    dil = ind.copy()
    dil[1:] = np.maximum(dil[1:], ind[:-1])
    dil[:-1] = np.maximum(dil[:-1], ind[1:])
    n = len(ind)
    L = 1 << int(np.ceil(np.log2(2 * n)))
    ac = np.fft.irfft(np.conj(np.fft.rfft(ind, L)) * np.fft.rfft(dil, L), L)[:n]
    lags = np.arange(int(fs / PERIOD_LAG_HZ[1]), min(int(fs / PERIOD_LAG_HZ[0]), n - 1))
    v = ac[lags]
    if v.max() <= 0:
        return None
    return float(lags[np.flatnonzero(v >= 0.8 * v.max())[0]])


def pulse_train(dx, lo, hi, thr, refr):
    lo, hi = max(0, int(lo)), min(len(dx), int(hi))
    idx = np.flatnonzero(dx[lo:hi] > thr) + lo
    if len(idx) == 0:
        return idx
    keep = [int(idx[0])]
    for i in idx[1:]:
        if i - keep[-1] > refr:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def regular_mask(p, P):
    """ok[i] is True when the interval p[i+1]-p[i] lies within ITI_LO..ITI_HI periods."""
    if len(p) < 2:
        return np.zeros(0, bool)
    iti = np.diff(p)
    return (iti >= ITI_LO * P) & (iti <= ITI_HI * P)


def first_regular(p, P):
    ok = regular_mask(p, P)
    for i in range(len(ok) - N_REGULAR + 1):
        if ok[i:i + N_REGULAR].all():
            return i
    return None


def last_regular(p, P):
    ok = regular_mask(p, P)
    for i in range(len(ok) - 1, N_REGULAR - 2, -1):
        if ok[i - N_REGULAR + 1:i + 1].all():
            return i + 1
    return None


def periodic_count(p, P):
    """Number of pulses that belong to a run of at least N_REGULAR consecutive regular intervals
    (the same train criterion as the ON-block onset)."""
    if len(p) < N_REGULAR + 1:
        return 0
    ok = regular_mask(p, P)
    has = np.zeros(len(p), bool)
    i = 0
    while i < len(ok):
        if ok[i]:
            k = i
            while k + 1 < len(ok) and ok[k + 1]:
                k += 1
            if k - i + 1 >= N_REGULAR:
                has[i:k + 2] = True
            i = k + 1
        else:
            i += 1
    return int(has.sum())


def pulse_peaks(dx, pulses):
    idx = np.clip(pulses[:, None] + np.arange(6)[None, :], 0, len(dx) - 1)
    return dx[idx].max(axis=1)


def peak_amp(dx, pulses):
    if len(pulses) == 0:
        return None
    return float(np.median(pulse_peaks(dx, pulses)))


def spectral_profile(x, segs, fs):
    from scipy.signal import welch
    acc, used, f = None, 0, None
    for a, b in segs:
        if b - a < SPEC_NPERSEG:
            continue
        f, p = welch(x[a:b], fs=fs, nperseg=SPEC_NPERSEG, detrend="constant")
        acc = p * (b - a) if acc is None else acc + p * (b - a)
        used += b - a
    if acc is None:
        return None
    sel = f >= BANDS[0][1]
    tot = acc[sel].sum()
    out = {"spectral_seconds": round(used / fs, 1)}
    for name, lo, hi in BANDS:
        out[name] = round(float(acc[(f >= lo) & (f < hi)].sum() / tot), 4) if tot > 0 else None
    out["peak_hz"] = round(float(f[sel][np.argmax(acc[sel])]), 1) if tot > 0 else None
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    r = spearmanr(a, b).correlation
    return None if r is None or np.isnan(r) else round(float(r), 3)


def rnd(v, k=3):
    return None if v is None else round(float(v), k)


# ----------------------------------------------------------------------------- stimulation

def analyze_stim(data, labels, cats, fs, file_row):
    n = data.shape[0]
    cmd_idx = [j for j, c in enumerate(cats) if c == "stim_command"]
    active = [j for j in cmd_idx if np.any(data[:, j] != 0)]
    if not active:
        return None
    j_cmd = active[0]
    cmd = data[:, j_cmd]
    tel_name = {"Am": "Rs", "Cur": "Res"}[re.match(r"[A-Za-z]+", labels[j_cmd]).group(0)] + labels[j_cmd][-1]
    j_tel = labels.index(tel_name) if tel_name in labels else None
    st, en, val = runs_of(cmd)
    on = val > 0
    segs = [{"a": int(a), "b": int(b), "val": float(v), "on": bool(o)} for a, b, v, o in zip(st, en, val, on)]
    # ON blocks: maximal runs of consecutive ON segments
    blocks = []
    i = 0
    while i < len(segs):
        if segs[i]["on"]:
            k = i
            while k + 1 < len(segs) and segs[k + 1]["on"]:
                k += 1
            prev_off = segs[i - 1]["b"] - segs[i - 1]["a"] if i > 0 else 0
            next_off = segs[k + 1]["b"] - segs[k + 1]["a"] if k + 1 < len(segs) else 0
            blocks.append({"i0": i, "i1": k, "r": segs[i]["a"], "e": segs[k]["b"], "prev_off": prev_off, "next_off": next_off})
            i = k + 1
        else:
            i += 1
    trim = int(TRIM_S * fs)
    on_core = np.zeros(n, bool)
    off_core = np.zeros(n, bool)
    for s in segs:
        if s["b"] - s["a"] > 2 * trim:
            (on_core if s["on"] else off_core)[s["a"] + trim:s["b"] - trim] = True
    # artefact channel: largest ON/OFF ratio of |dx| among LFP channels
    cand = []
    for j, c in enumerate(cats):
        if c not in ("lfp_unipolar", "lfp_bipolar"):
            continue
        x = data[:, j]
        if x.max() == x.min():
            continue
        dx = absdiff(x)
        base = float(np.median(dx[off_core])) if off_core.any() else float(np.median(dx))
        top = float(np.percentile(dx[on_core], 99.5)) if on_core.any() else float(np.percentile(dx[cmd > 0], 99.5))
        cand.append((top / (base + 1e-12), j, base))
    cand.sort(key=lambda t: -t[0])
    res = {"cmd_label": labels[j_cmd], "tel_label": labels[j_tel] if j_tel is not None else "",
           "segs": segs, "blocks": blocks, "channels": []}
    period = None
    for rank, (ratio, j, base) in enumerate(cand[:2]):
        dx = absdiff(data[:, j])
        thr_b = []
        for b in blocks:
            c0, c1 = b["r"] + trim, b["e"] - trim
            if c1 - c0 >= int(CORE_MIN_S * fs):
                thr_b.append(max(THR_FRAC * float(np.percentile(dx[c0:c1], 99)), THR_MAD * base))
            else:
                thr_b.append(None)
        long_thr = [t for t in thr_b if t is not None]
        fallback = min(long_thr) if long_thr else THR_MAD * base
        thr_src = ["block_core" if t is not None else "lowest_long_block" for t in thr_b]
        thr_b = [t if t is not None else fallback for t in thr_b]
        if rank == 0:
            lens = [(b["e"] - b["r"], k) for k, b in enumerate(blocks)]
            L, kb = max(lens)
            b = blocks[kb]
            c0 = b["r"] + trim
            c1 = min(b["e"] - trim, c0 + int(10 * fs))
            if c1 - c0 < int(CORE_MIN_S * fs):
                c0, c1 = b["r"], b["e"]
            lag0 = estimate_period(dx, c0, c1, thr_b[kb], fs)
            if lag0 is not None:
                p = pulse_train(dx, c0, c1, thr_b[kb], REFR_FRAC * lag0)
                iti = np.diff(p)
                iti = iti[(iti >= ITI_LO * lag0) & (iti <= ITI_HI * lag0)]
                period = float(iti.mean()) if len(iti) else lag0
            res["period_samples"] = period
        chan = {"label": labels[j], "index": j, "ratio": ratio, "off_median_absdiff": base, "blocks": []}
        if period is None:
            res["channels"].append(chan)
            continue
        refr = REFR_FRAC * period
        for k, b in enumerate(blocks):
            thr = thr_b[k]
            w0 = b["r"] - min(trim, b["prev_off"] // 2)
            w1 = b["e"] + min(trim, b["next_off"] // 2)
            p = pulse_train(dx, w0, w1, thr, refr)
            i0 = first_regular(p, period)
            i1 = last_regular(p, period)
            out = {"thr": thr, "thr_source": thr_src[k], "detected": i0 is not None and i1 is not None and i1 >= i0}
            if out["detected"]:
                onset = int(p[i0])
                lo = onset
                while lo - 1 >= onset - REFINE_MAX and lo - 1 >= 0 and dx[lo - 1] > REFINE_MAD * base:
                    lo -= 1
                pulses = p[i0:i1 + 1]
                gaps = np.diff(pulses)
                out.update({"longest_gap": int(gaps.max()) if len(gaps) else 0,
                            "n_gaps_over_gapmin": int(np.count_nonzero(gaps > GAP_MIN_S * fs))})
                pk = pulse_peaks(dx, pulses)
                med = float(np.median(pk))
                out["first10_peak_ratio"] = float(np.median(pk[:EDGE_PULSES]) / med) if med > 0 else None
                out["last10_peak_ratio"] = float(np.median(pk[-EDGE_PULSES:]) / med) if med > 0 else None
                # largest |dx| between the command rise and the first detected pulse, and in the
                # same-length window just before the rise, in multiples of the OFF median
                gap_len = lo - 2 - b["r"]
                if gap_len > 0:
                    out["rise_to_onset_max_ratio"] = float(dx[b["r"]:lo - 2].max() / base)
                    over = np.flatnonzero(dx[b["r"]:lo - 2] > REFINE_MAD * base)
                    out["rise_to_onset_n_over"] = int(len(over))
                    out["rise_to_onset_first_over_lag"] = int(over[0]) if len(over) else None
                    w = min(gap_len, b["prev_off"] // 2)
                    out["pre_rise_max_ratio"] = float(dx[b["r"] - w:b["r"]].max() / base) if w > 0 else None
                tail0 = int(p[i1]) + int(period / 2)
                tail1 = min(len(dx), b["e"] + int(OFF_MARGIN_S * fs))
                out["last_to_offmargin_max_ratio"] = float(dx[tail0:tail1].max() / base) if tail1 > tail0 else None
                out.update({"onset": lo, "onset_peak_sample": onset, "last": int(p[i1]), "n_pulses": int(len(pulses)),
                            "freq_hz": fs * (len(pulses) - 1) / (pulses[-1] - pulses[0]) if len(pulses) > 1 else None,
                            "seg_pulses": [], "seg_peak": []})
                for si in range(b["i0"], b["i1"] + 1):
                    s = segs[si]
                    sp = pulses[(pulses >= s["a"]) & (pulses < s["b"])]
                    out["seg_pulses"].append(int(len(sp)))
                    out["seg_peak"].append(peak_amp(dx, sp))
            chan["blocks"].append(out)
        if rank == 0:
            thr_min = min(thr_b) if thr_b else THR_MAD * base
            stray = []
            for si, s in enumerate(segs):
                if s["on"]:
                    stray.append(None)
                    continue
                a = s["a"] + (int(OFF_MARGIN_S * fs) if si > 0 else 0)
                p = pulse_train(dx, a, s["b"], thr_min, refr) if s["b"] > a else np.zeros(0, np.int64)
                stray.append(periodic_count(p, period))
            res["stray"] = stray
        res["channels"].append(chan)
        del dx
    # telemetry per ON segment
    tel = []
    for s in segs:
        if not s["on"] or j_tel is None:
            tel.append((None, None))
            continue
        t = data[s["a"]:s["b"], j_tel]
        nz = np.flatnonzero(t != 0)
        tel.append((float(np.median(t[nz])) if len(nz) else None, int(s["a"] + nz[0]) if len(nz) else None))
    res["tel"] = tel
    res["tel_nonzero_while_cmd_zero"] = int(np.count_nonzero((data[:, j_tel] != 0) & (cmd == 0))) if j_tel is not None else None
    res["tel_zero_while_cmd_on"] = int(np.count_nonzero((data[:, j_tel] == 0) & (cmd != 0))) if j_tel is not None else None
    edges = np.concatenate([st[1:]])
    res["edge_gcd"] = int(np.gcd.reduce(np.diff(edges))) if len(edges) > 2 else None
    return res


def stim_rows(res, fs, file_row):
    """Segment rows, G15 rows and the per-file summary from analyze_stim output."""
    seg_rows, g15_rows = [], []
    segs, blocks = res["segs"], res["blocks"]
    ch1 = res["channels"][0] if res["channels"] else None
    ch2 = res["channels"][1] if len(res["channels"]) > 1 else None
    block_of_seg = {}
    for k, b in enumerate(blocks):
        for si in range(b["i0"], b["i1"] + 1):
            block_of_seg[si] = k
    prev_on_val = None
    on_vals, on_peaks = [], []
    for si, s in enumerate(segs):
        dur = (s["b"] - s["a"]) / fs
        row = {k: file_row[k] for k in ("file", "patient", "hemisphere", "kind", "contact_level")}
        row.update({"segment": si, "state": "ON" if s["on"] else "OFF",
                    "amplitude_uA": s["val"] if s["on"] else 0.0, "amplitude_mA": s["val"] / 1000.0 if s["on"] else 0.0,
                    "start_sample": s["a"], "end_sample": s["b"], "start_s": round(s["a"] / fs, 4),
                    "end_s": round(s["b"] / fs, 4), "duration_s": round(dur, 4)})
        if s["on"]:
            prev = segs[si - 1] if si > 0 else None
            entry = "file_start" if prev is None else ("from_off" if not prev["on"] else
                                                       ("step_up" if s["val"] > prev["val"] else "step_down"))
            row["entry"] = entry
            row["down_step"] = entry == "step_down"
            row["lower_than_previous_on_block"] = bool(entry == "from_off" and prev_on_val is not None and s["val"] < prev_on_val)
            row["repeat_of_previous_on_amplitude"] = bool(prev_on_val is not None and s["val"] == prev_on_val)
            row["short_block"] = dur < SHORT_BLOCK_S
            k = block_of_seg[si]
            b = blocks[k]
            if ch1 is not None and ch1["blocks"]:
                d1 = ch1["blocks"][k]
                row["artefact_channel"] = ch1["label"]
                row["artefact_detected"] = d1["detected"]
                if d1["detected"]:
                    pos = si - b["i0"]
                    if si == b["i0"]:
                        row["onset_lag_samples"] = d1["onset"] - b["r"]
                        row["onset_lag_ms"] = round(1000.0 * (d1["onset"] - b["r"]) / fs, 2)
                    if si == b["i1"]:
                        row["offset_lag_samples"] = d1["last"] - b["e"]
                        row["offset_lag_ms"] = round(1000.0 * (d1["last"] - b["e"]) / fs, 2)
                    row["artefact_on_duration_s"] = round((d1["last"] - d1["onset"]) / fs, 4) if b["i0"] == b["i1"] else None
                    row["n_pulses"] = d1["seg_pulses"][pos]
                    row["stim_freq_hz"] = rnd(d1["freq_hz"], 2)
                    row["artefact_peak_absdiff_median"] = rnd(d1["seg_peak"][pos], 1)
                    row["artefact_peak_per_mA"] = rnd(d1["seg_peak"][pos] / (s["val"] / 1000.0), 1) if d1["seg_peak"][pos] else None
                    if d1["seg_peak"][pos] is not None and not row["short_block"]:
                        on_vals.append(s["val"])
                        on_peaks.append(d1["seg_peak"][pos])
            tm, tfirst = res["tel"][si]
            row["telemetry_channel"] = res["tel_label"]
            row["telemetry_median_nonzero"] = tm
            if tfirst is not None:
                row["telemetry_first_nonzero_lag_samples"] = tfirst - s["a"]
            prev_on_val = s["val"]
        else:
            if si == 0:
                row["position"] = "leading"
            elif si == len(segs) - 1:
                row["position"] = "trailing"
            else:
                row["position"] = "interior"
            row["stray_periodic_pulses"] = res.get("stray", [None] * len(segs))[si]
        seg_rows.append(row)
    for k, b in enumerate(blocks):
        g = {k2: file_row[k2] for k2 in ("file", "patient", "hemisphere", "kind", "contact_level")}
        g.update({"on_block": k, "amplitude_uA": segs[b["i0"]]["val"], "command_rise_sample": b["r"],
                  "command_fall_sample": b["e"], "command_on_s": round((b["e"] - b["r"]) / fs, 4)})
        for tag, ch in (("", ch1), ("ch2_", ch2)):
            if ch is None or not ch["blocks"]:
                continue
            d = ch["blocks"][k]
            g[f"{tag}artefact_channel"] = ch["label"]
            g[f"{tag}detected"] = d["detected"]
            g[f"{tag}threshold_source"] = d["thr_source"]
            if d["detected"]:
                g[f"{tag}artefact_first_sample"] = d["onset"]
                g[f"{tag}onset_lag_samples"] = d["onset"] - b["r"]
                g[f"{tag}onset_lag_ms"] = round(1000.0 * (d["onset"] - b["r"]) / fs, 2)
                g[f"{tag}artefact_last_sample"] = d["last"]
                g[f"{tag}offset_lag_samples"] = d["last"] - b["e"]
                g[f"{tag}offset_lag_ms"] = round(1000.0 * (d["last"] - b["e"]) / fs, 2)
                if tag == "":
                    g["n_pulses"] = d["n_pulses"]
                    g["artefact_train_s"] = round((d["last"] - d["onset"]) / fs, 4)
                    g["delivered_fraction"] = round((d["last"] - d["onset"]) / (b["e"] - b["r"]), 4)
                    g["longest_pulse_gap_s"] = round(d["longest_gap"] / fs, 4)
                    g["n_pulse_gaps_over_0p1s"] = d["n_gaps_over_gapmin"]
                    g["first10_peak_ratio"] = rnd(d["first10_peak_ratio"], 3)
                    g["last10_peak_ratio"] = rnd(d["last10_peak_ratio"], 3)
                    g["rise_to_onset_max_ratio"] = rnd(d.get("rise_to_onset_max_ratio"), 1)
                    g["pre_rise_max_ratio"] = rnd(d.get("pre_rise_max_ratio"), 1)
                    g["rise_to_onset_n_samples_over_10x"] = d.get("rise_to_onset_n_over")
                    g["rise_to_onset_first_over_10x_lag_samples"] = d.get("rise_to_onset_first_over_lag")
                    g["last_to_offmargin_max_ratio"] = rnd(d.get("last_to_offmargin_max_ratio"), 1)
        g["within_1_sample"] = bool(g.get("detected") and abs(g["onset_lag_samples"]) <= 1)
        if not g.get("detected"):
            g["delivery"] = "no_artefact"
        elif (b["e"] - g["artefact_last_sample"]) > INTERRUPT_S * fs or g["longest_pulse_gap_s"] > INTERRUPT_S:
            g["delivery"] = "interrupted"
        else:
            g["delivery"] = "complete"
        g15_rows.append(g)
    # per-file summary
    ons = [r for r in seg_rows if r["state"] == "ON"]
    offs = [r for r in seg_rows if r["state"] == "OFF"]
    amps = [r["amplitude_mA"] for r in ons]
    on_d = [r["duration_s"] for r in ons]
    int_off = [r["duration_s"] for r in offs if r.get("position") == "interior"]
    distinct = sorted(set(amps))
    nondec = all(amps[i + 1] >= amps[i] for i in range(len(amps) - 1))
    if not ons:
        pattern = "no_stimulation"
    elif len(ons) == 1:
        pattern = "single_block"
    elif len(distinct) == 1:
        pattern = "repeated_single_amplitude"
    elif nondec:
        pattern = "ascending_ramp"
    else:
        pattern = "irregular"
    for r in seg_rows:
        if r["state"] == "ON" and r["segment"] in [blocks[k]["i0"] for k in range(len(blocks))]:
            k = [blocks[q]["i0"] for q in range(len(blocks))].index(r["segment"])
            r["delivery"] = g15_rows[k]["delivery"]
            r["delivered_fraction"] = g15_rows[k].get("delivered_fraction")
            r["longest_pulse_gap_s"] = g15_rows[k].get("longest_pulse_gap_s")
    lags = [g["onset_lag_samples"] for g in g15_rows if g.get("detected")]
    offl = [g["offset_lag_samples"] for g in g15_rows if g.get("delivery") == "complete"]
    lag2 = [g["ch2_onset_lag_samples"] - g["onset_lag_samples"] for g in g15_rows
            if g.get("detected") and g.get("ch2_detected")]
    n_tr = len(g15_rows)
    n_det = len(lags)
    n_in = sum(1 for g in g15_rows if g["within_1_sample"])
    summ = {k: file_row[k] for k in ("file", "patient", "hemisphere", "kind", "contact_level", "bipolar_channel",
                                     "level_consistent", "duration_s")}
    freqs = [r.get("stim_freq_hz") for r in ons if r.get("stim_freq_hz")]
    summ.update({
        "command_channel": res["cmd_label"], "telemetry_channel": res["tel_label"],
        "n_on_blocks": len(blocks), "n_on_segments": len(ons), "n_off_segments": len(offs),
        "amplitudes_mA": " ".join(f"{a:g}" for a in amps), "max_mA": max(amps) if amps else None,
        "n_distinct_amplitudes": len(distinct), "step_pattern": pattern, "nondecreasing": nondec,
        "n_down_steps": sum(1 for r in ons if r.get("down_step")),
        "n_lower_than_previous_on_block": sum(1 for r in ons if r.get("lower_than_previous_on_block")),
        "n_repeat_amplitude": sum(1 for r in ons if r.get("repeat_of_previous_on_amplitude")),
        "n_short_on_blocks": sum(1 for r in ons if r.get("short_block")),
        "on_s_min": rnd(min(on_d), 1) if on_d else None, "on_s_median": rnd(np.median(on_d), 1) if on_d else None,
        "on_s_max": rnd(max(on_d), 1) if on_d else None, "on_s_total": rnd(sum(on_d), 1),
        "interior_off_s_min": rnd(min(int_off), 1) if int_off else None,
        "interior_off_s_median": rnd(np.median(int_off), 1) if int_off else None,
        "interior_off_s_max": rnd(max(int_off), 1) if int_off else None,
        "leading_off_s": rnd(offs[0]["duration_s"], 1) if offs and offs[0].get("position") == "leading" else 0.0,
        "trailing_off_s": rnd(offs[-1]["duration_s"], 1) if offs and offs[-1].get("position") == "trailing" else 0.0,
        "stim_period_samples": rnd(res.get("period_samples"), 3),
        "stim_freq_hz_median": rnd(np.median(freqs), 2) if freqs else None,
        "artefact_channel": ch1["label"] if ch1 else "", "artefact_channel_ratio": rnd(ch1["ratio"], 1) if ch1 else None,
        "artefact_channel2": ch2["label"] if ch2 else "",
        "n_on_transitions": n_tr, "n_artefact_detected": n_det, "n_within_1_sample": n_in,
        "onset_lag_min_samples": min(lags) if lags else None,
        "onset_lag_median_samples": rnd(np.median(lags), 1) if lags else None,
        "onset_lag_max_samples": max(lags) if lags else None,
        "onset_lag_min_ms": rnd(1000.0 * min(lags) / fs, 1) if lags else None,
        "onset_lag_max_ms": rnd(1000.0 * max(lags) / fs, 1) if lags else None,
        "offset_lag_min_samples": min(offl) if offl else None,
        "offset_lag_max_samples": max(offl) if offl else None,
        "n_blocks_complete": sum(1 for g in g15_rows if g["delivery"] == "complete"),
        "n_blocks_interrupted": sum(1 for g in g15_rows if g["delivery"] == "interrupted"),
        "n_blocks_no_artefact": sum(1 for g in g15_rows if g["delivery"] == "no_artefact"),
        "ch2_minus_ch1_onset_max_abs_samples": max(abs(v) for v in lag2) if lag2 else None,
        "stray_periodic_pulses_in_off": sum(v for v in res.get("stray", []) if v),
        "command_edge_gcd_samples": res["edge_gcd"],
        "telemetry_nonzero_samples_while_command_zero": res["tel_nonzero_while_cmd_zero"],
        "telemetry_zero_samples_while_command_on": res["tel_zero_while_cmd_on"],
        "artefact_peak_vs_amplitude_spearman": spearman(on_vals, on_peaks),
        "g15_pass": bool(n_tr > 0 and n_in == n_tr),
    })
    reasons = []
    if n_det < n_tr:
        reasons.append(f"artefact not detected in {n_tr - n_det} of {n_tr} ON blocks")
    if n_det and n_in < n_det:
        reasons.append(f"first artefact pulse more than 1 sample from the command rise in {n_det - n_in} of {n_det} detected ON blocks")
    summ["g15_reason"] = "; ".join(reasons) if reasons else "all ON transitions within 1 sample"
    return seg_rows, g15_rows, summ


# ----------------------------------------------------------------------------- per file

def process(zpath, zinfo):
    t0 = time.time()
    raw = zipfile.ZipFile(zpath).read(zinfo)
    t_read = time.time() - t0
    M = read_mat(raw)
    del raw
    base, patient, hemi, kind, level, note = parse_name(zinfo.filename)
    fs, labels = M["fs"], M["labels"]
    X = np.asfortranarray(M.pop("data"))
    n, C = X.shape
    cats = [classify(l) for l in labels]
    bip = [l for l, c in zip(labels, cats) if c == "lfp_bipolar"]
    bip_level = int(bip[0][2]) - 1 if bip and int(bip[0][2]) - int(bip[0][1]) == 2 else None
    # channels
    ch_rows = []
    flat = []
    by_label = {}
    for j, (l, c) in enumerate(zip(labels, cats)):
        x = X[:, j]
        mn, mx = float(x.min()), float(x.max())
        nchg = int(np.count_nonzero(x[1:] != x[:-1]))
        if nchg:
            d = np.flatnonzero(x[1:] != x[:-1]) + 1
            rl = np.diff(np.concatenate(([0], d, [n])))
            medrun = float(np.median(rl))
        else:
            medrun = float(n)
        is_flat = mx == mn
        if is_flat:
            flat.append(l)
        dup_of, rmax = None, None
        if l in by_label:
            for jj in by_label[l]:
                if np.array_equal(X[:, jj], x):
                    dup_of = jj
                    break
            if not is_flat:
                rs = []
                for jj in by_label[l]:
                    y = X[::16, jj]
                    if y.std() > 0:
                        rs.append(float(np.corrcoef(y, x[::16])[0, 1]))
                rmax = rnd(max(rs, key=abs), 4) if rs else None
        by_label.setdefault(l, []).append(j)
        ch_rows.append({"file": base, "patient": patient, "hemisphere": hemi, "kind": kind, "index": j, "label": l,
                        "category": c, "fs_hz": fs, "unit": unit_of(l, c), "n_samples": n,
                        "min": rnd(mn, 4), "max": rnd(mx, 4), "mean": rnd(x.mean(), 4), "std": rnd(x.std(), 4),
                        "frac_zero": rnd(np.mean(x == 0), 4), "flat": is_flat, "n_value_changes": nchg,
                        "median_run_samples": medrun, "duplicate_label": len(by_label[l]) > 1,
                        "identical_to_index": dup_of, "r_with_earlier_same_label": rmax})
    dup_labels = sorted({l for l in labels if labels.count(l) > 1})
    file_row = {"file": base, "zip": os.path.basename(zpath), "member": zinfo.filename, "crc32": f"{zinfo.CRC:08x}",
                "uncompressed_bytes": zinfo.file_size, "patient": patient, "hemisphere": hemi, "kind": kind,
                "contact_level": level, "bipolar_channel": " ".join(bip),
                "level_from_bipolar": bip_level,
                "level_consistent": (level == bip_level) if (level is not None and bip_level is not None) else None,
                "mat_version": M["mat_version"], "created": M["created"], "reader": M["reader"],
                "struct_fields": " ".join(M["fields"]), "fs_hz": fs, "per_channel_rates": "one Fs per file",
                "n_channels": C, "n_samples": n, "duration_s": round(n / fs, 4), "duration_h": round(n / fs / 3600, 5),
                "channel_list": " ".join(labels)}
    for cat in ("lfp_unipolar", "lfp_bipolar", "eeg", "emg", "accelerometer", "aux", "stim_command",
                "stim_telemetry", "derived", "other"):
        file_row[cat] = " ".join(l for l, c in zip(labels, cats) if c == cat)
    file_row["flat_channels"] = " ".join(flat)
    file_row["duplicate_labels"] = " ".join(dup_labels)
    marks = M["marks"]
    file_row["n_marks"] = sum(len(v) for v in marks.values())
    file_row["marks_s"] = ";".join(f"{k}:" + ",".join(f"{v:.3f}" for v in vs) for k, vs in marks.items())
    head = [l for l, c in zip(labels, cats) if c == "accelerometer" and ACC_GROUPS[l[:3]][1] and l not in flat]
    second = [l for l, c in zip(labels, cats) if c == "accelerometer" and not ACC_GROUPS[l[:3]][1] and l not in flat]
    emg = [l for l, c in zip(labels, cats) if c == "emg" and l not in flat]
    aux = [l for l, c in zip(labels, cats) if c == "aux" and l not in flat]
    file_row["head_accelerometer"] = " ".join(head)
    file_row["other_accelerometer"] = " ".join(second)
    file_row["neck_emg"] = " ".join(emg)
    file_row["aux_unidentified"] = " ".join(aux)
    file_row["has_movement_channel"] = bool(head or second or emg)
    file_row["note"] = note
    # stimulation
    res = analyze_stim(X, labels, cats, fs, file_row)
    seg_rows, g15_rows, summ = ([], [], None)
    if res is not None:
        seg_rows, g15_rows, summ = stim_rows(res, fs, file_row)
    active_cmd = [l for j, (l, c) in enumerate(zip(labels, cats)) if c == "stim_command" and np.any(X[:, j] != 0)]
    file_row["stim_command_active"] = " ".join(active_cmd)
    # movement channels: spectra over stimulation-free cores, else whole file
    if res is not None:
        trim = int(TRIM_S * fs)
        spec_segs = [(s["a"] + trim, s["b"] - trim) for s in res["segs"] if not s["on"] and s["b"] - s["a"] > 2 * trim]
        spec_src = "command_off_cores"
    else:
        spec_segs = [(0, n)]
        spec_src = "whole_file"
    mv_rows = []
    for j, (l, c) in enumerate(zip(labels, cats)):
        if c not in ("emg", "accelerometer", "aux"):
            continue
        x = X[:, j]
        if c == "emg":
            group, what, side = "EMG", EMG_LABELS[l][0], EMG_LABELS[l][1]
            role = "neck_emg"
        elif c == "accelerometer":
            group, what, side = l[:3], ACC_GROUPS[l[:3]][0], ""
            role = "head_accelerometer" if ACC_GROUPS[l[:3]][1] else "other_accelerometer"
        else:
            group, what, side = l, "not documented", ""
            role = "aux_unidentified"
        row = {"file": base, "patient": patient, "hemisphere": hemi, "kind": kind, "index": j, "label": l,
               "category": c, "role": role, "group": group, "muscle_or_location": what, "side": side,
               "fs_hz": fs, "flat": bool(x.max() == x.min()), "mean": rnd(x.mean(), 3), "std": rnd(x.std(), 3),
               "spectral_source": spec_src}
        prof = spectral_profile(x, spec_segs, fs) if x.max() > x.min() else None
        if prof:
            row.update(prof)
        mv_rows.append(row)
    del X
    return {"version": CACHE_VERSION, "file_row": file_row, "channels": ch_rows, "segments": seg_rows,
            "g15": g15_rows, "summary": summ, "movement": mv_rows,
            "timing_s": {"zip_read": round(t_read, 2), "total": round(time.time() - t0, 2)}}


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- outputs

def shown(path):
    """Path relative to this tree for the progress print; absolute when --out lies on another drive."""
    try:
        return os.path.relpath(path, TREE)
    except ValueError:
        return path


def write_csv(path, rows, first=None):
    keys = list(first or [])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    print("wrote", shown(path), len(rows), "rows")


def fmt_list(xs):
    return ", ".join(xs) if xs else "none"


def hours_of(rows):
    """Exact hours of a set of inventory rows, from samples and rate (never from the rounded duration_h,
    whose rounding errors add up across files)."""
    if not rows:
        return 0
    return sum(r["n_samples"] / r["fs_hz"] for r in rows) / 3600.0


def main():
    force = "--force" in sys.argv
    only = None
    argv = [a for a in sys.argv[1:] if a != "--force"]
    if "--only" in argv:
        k = argv.index("--only")
        only = argv[k + 1]
        del argv[k:k + 2]
    if "--out" not in argv:
        argv += ["--out", os.path.join(TREE, "reports", "oxford_lfp")]
    sys.argv = [sys.argv[0]] + argv
    a = parse_args(__doc__.splitlines()[0])
    raw_dir = os.path.join(a.moveatlas, RAW_REL)
    cache = os.path.join(a.out, "cache")
    os.makedirs(cache, exist_ok=True)
    zips = sorted(os.path.join(raw_dir, z) for z in os.listdir(raw_dir) if z.endswith(".zip") and not z.startswith("._"))
    extracted = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith("._"))
    members = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for zi in zf.infolist():
                name = zi.filename.split("/")[-1]
                if zi.filename.lower().endswith(".mat") and not name.startswith("._") and "__MACOSX" not in zi.filename:
                    members.append((z, zi))
    print(f"{len(zips)} zips, {len(members)} .mat members, extracted folders: {extracted or 'none'}")
    results = []
    t_all = time.time()
    for i, (z, zi) in enumerate(members):
        base = zi.filename.split("/")[-1]
        if only and not any(o in base for o in only.split(",")):
            continue
        cpath = os.path.join(cache, base + ".json")
        r = None
        if os.path.isfile(cpath) and not force:
            with open(cpath, encoding="utf-8") as f:
                r = json.load(f)
            if r.get("version") != CACHE_VERSION or r["file_row"].get("crc32") != f"{zi.CRC:08x}":
                r = None
        if r is None:
            r = to_jsonable(process(z, zi))
            with open(cpath, "w", encoding="utf-8", newline="\n") as f:
                json.dump(r, f, indent=0)
            tag = f"{r['timing_s']['total']:.1f} s"
        else:
            tag = "cached"
        s = r["summary"]
        extra = ""
        if s:
            extra = (f" | {s['step_pattern']} on={s['n_on_blocks']} lag={s['onset_lag_min_samples']}..{s['onset_lag_max_samples']}"
                     f" g15={'pass' if s['g15_pass'] else 'fail'}")
        print(f"[{i + 1:2d}/{len(members)}] {base:24s} {r['file_row']['kind']:12s} {tag}{extra}", flush=True)
        results.append(r)
    print(f"files done in {time.time() - t_all:.1f} s")
    if only:
        print("--only given: aggregate outputs not written")
        return

    inv = [r["file_row"] for r in results]
    chans = [c for r in results for c in r["channels"]]
    ramp_seg = [s for r in results if r["file_row"]["kind"] == "ramp" for s in r["segments"]]
    other_seg = [s for r in results if r["file_row"]["kind"] != "ramp" for s in r["segments"]]
    ramp_sum = [r["summary"] for r in results if r["file_row"]["kind"] == "ramp" and r["summary"]]
    other_sum = [r["summary"] for r in results if r["file_row"]["kind"] != "ramp" and r["summary"]]
    g15 = [g for r in results if r["file_row"]["kind"] == "ramp" for g in r["g15"]]
    mv = [m for r in results for m in r["movement"]]
    out = a.out
    write_csv(os.path.join(out, "inventory.csv"), inv)
    write_csv(os.path.join(out, "channels.csv"), chans)
    write_csv(os.path.join(out, "ramp_blocks.csv"), ramp_seg,
              first=["file", "patient", "hemisphere", "kind", "contact_level", "segment", "state", "amplitude_uA",
                     "amplitude_mA", "start_sample", "end_sample", "start_s", "end_s", "duration_s", "position", "entry"])
    write_csv(os.path.join(out, "nonramp_stim_blocks.csv"), other_seg,
              first=["file", "patient", "hemisphere", "kind", "contact_level", "segment", "state", "amplitude_uA",
                     "amplitude_mA", "start_sample", "end_sample", "start_s", "end_s", "duration_s", "position", "entry"])
    write_csv(os.path.join(out, "ramp_file_summary.csv"), ramp_sum)
    write_csv(os.path.join(out, "nonramp_stim_summary.csv"), other_sum)
    write_csv(os.path.join(out, "g15_transitions.csv"), g15)
    g15_files = [{k: s[k] for k in ("file", "patient", "hemisphere", "contact_level", "command_channel", "artefact_channel",
                                    "n_on_transitions", "n_artefact_detected", "n_within_1_sample",
                                    "onset_lag_min_samples", "onset_lag_median_samples", "onset_lag_max_samples",
                                    "onset_lag_min_ms", "onset_lag_max_ms", "offset_lag_min_samples",
                                    "offset_lag_max_samples", "n_blocks_complete", "n_blocks_interrupted",
                                    "n_blocks_no_artefact", "ch2_minus_ch1_onset_max_abs_samples",
                                    "stray_periodic_pulses_in_off", "g15_pass", "g15_reason")} for s in ramp_sum]
    write_csv(os.path.join(out, "g15_files.csv"), g15_files)
    write_csv(os.path.join(out, "movement_channels.csv"), mv)

    # counts
    counts = []
    kinds = ["ramp", "single-pulse", "paired-pulse", "other"]
    pats = sorted({r["patient"] for r in inv}, key=lambda p: int(p[1:]))
    for scope, key in (("all", None), ("patient", "patient"), ("hemisphere", "hemi")):
        groups = {"all": inv} if key is None else {}
        if key == "patient":
            for p in pats:
                groups[p] = [r for r in inv if r["patient"] == p]
        if key == "hemi":
            for p in pats:
                for h in ("L", "R"):
                    g = [r for r in inv if r["patient"] == p and r["hemisphere"] == h]
                    if g:
                        groups[f"{p}_{h}"] = g
        for gname, g in groups.items():
            row = {"scope": scope, "group": gname, "n_files": len(g), "hours_all": round(hours_of(g), 4)}
            for k in kinds:
                gk = [r for r in g if r["kind"] == k]
                row[f"n_{k}"] = len(gk)
                row[f"hours_{k}"] = round(hours_of(gk), 4)
            ramps = [r for r in g if r["kind"] == "ramp"]
            row["ramp_duration_s_min"] = rnd(min(r["duration_s"] for r in ramps), 1) if ramps else None
            row["ramp_duration_s_max"] = rnd(max(r["duration_s"] for r in ramps), 1) if ramps else None
            counts.append(row)
    write_csv(os.path.join(out, "counts.csv"), counts)

    # movement per hemisphere
    mh = []
    for p in pats:
        for h in ("L", "R"):
            g = [r for r in inv if r["patient"] == p and r["hemisphere"] == h]
            if not g:
                continue
            ramps = [r for r in g if r["kind"] == "ramp"]
            row = {"patient": p, "hemisphere": h, "n_files": len(g), "n_ramp": len(ramps),
                   "ramp_hours": round(hours_of(ramps), 4),
                   "n_ramp_with_head_accelerometer": sum(1 for r in ramps if r["head_accelerometer"]),
                   "n_ramp_with_neck_emg": sum(1 for r in ramps if r["neck_emg"]),
                   "n_ramp_with_other_accelerometer": sum(1 for r in ramps if r["other_accelerometer"]),
                   "n_ramp_with_aux": sum(1 for r in ramps if r["aux_unidentified"]),
                   "head_accelerometer_labels": " ".join(sorted({x for r in ramps for x in r["head_accelerometer"].split()})),
                   "other_accelerometer_labels": " ".join(sorted({x for r in ramps for x in r["other_accelerometer"].split()})),
                   "neck_emg_labels": " ".join(sorted({x for r in ramps for x in r["neck_emg"].split()})),
                   "aux_labels": " ".join(sorted({x for r in ramps for x in r["aux_unidentified"].split()})),
                   "non_ramp_files": " ".join(f"{r['file']}({r['kind']})" for r in g if r["kind"] != "ramp"),
                   "non_ramp_movement_labels": " ".join(sorted({x for r in g if r["kind"] != "ramp" for x in
                                                              (r["head_accelerometer"] + " " + r["other_accelerometer"] + " " +
                                                               r["neck_emg"] + " " + r["aux_unidentified"]).split()})),
                   "n_files_with_movement_channel": sum(1 for r in g if r["has_movement_channel"])}
            mh.append(row)
    write_csv(os.path.join(out, "movement_hemispheres.csv"), mh)

    # addendum checks
    ramp_inv = [r for r in inv if r["kind"] == "ramp"]
    ramp_h = hours_of(ramp_inv)
    acc_hemis = [m for m in mh if m["n_ramp"] and m["n_ramp_with_head_accelerometer"] == m["n_ramp"]]
    acc_any = [m for m in mh if m["n_ramp_with_head_accelerometer"]]
    emg_hemis = [m for m in mh if m["n_ramp"] and m["n_ramp_with_neck_emg"] == m["n_ramp"]]
    emg_any = [m for m in mh if m["n_ramp_with_neck_emg"]]
    acc_keys = {(m["patient"], m["hemisphere"]) for m in acc_hemis}
    acc_h = hours_of([r for r in ramp_inv if (r["patient"], r["hemisphere"]) in acc_keys])
    hemi_str = lambda ms: " ".join(f"{m['patient']}{m['hemisphere']}" for m in ms)
    n_pat = lambda ms: len({m["patient"] for m in ms})
    checks = []

    def chk(claim, measured, ok, source="addendum 1"):
        checks.append({"claim": claim, "source": source, "measured": measured, "verdict": "confirmed" if ok else "corrected"})

    nk = {k: sum(1 for r in inv if r["kind"] == k) for k in kinds}
    chk("oxford_dystonia_lfp holds 68 .mat files, 62 of them ramp files; 5 single-pulse files; one paired-pulse file",
        f"{len(inv)} .mat files: {nk['ramp']} ramp, {nk['single-pulse']} single-pulse, {nk['paired-pulse']} paired-pulse, {nk['other']} other",
        len(inv) == 68 and nk["ramp"] == 62 and nk["single-pulse"] == 5 and nk["paired-pulse"] == 1)
    chk("Ramp files with a head accelerometer: 8 hemispheres of 4 patients (P2, P3, P6, P7), 6.69 h of the 8.38 h of ramps",
        f"{len(acc_hemis)} hemispheres of {n_pat(acc_hemis)} patients with a head accelerometer in every ramp file ({hemi_str(acc_hemis)}); "
        f"{len(acc_any)} with one in at least one ramp file; {acc_h:.2f} h of {ramp_h:.2f} h of ramps",
        len(acc_hemis) == 8 and n_pat(acc_hemis) == 4 and f"{acc_h:.2f}" == "6.69" and f"{ramp_h:.2f}" == "8.38")
    chk("6 hemispheres of 3 patients (P2, P3, P6) also carry neck EMG",
        f"{len(emg_hemis)} hemispheres of {n_pat(emg_hemis)} patients with neck EMG in every ramp file ({hemi_str(emg_hemis)}); "
        f"{len(emg_any)} with neck EMG in at least one ramp file; "
        f"labels per hemisphere: " + "; ".join(f"{m['patient']}{m['hemisphere']}: {m['neck_emg_labels']}" for m in emg_hemis),
        len(emg_hemis) == 6 and n_pat(emg_hemis) == 3)
    p4 = [r for r in inv if r["patient"] == "P4"]
    chk("P4 has no movement channel in any of its 12 files",
        f"P4: {len(p4)} files, {sum(1 for r in p4 if r['has_movement_channel'])} with an accelerometer or EMG channel, "
        f"{sum(1 for r in p4 if r['aux_unidentified'])} with an aux channel",
        len(p4) == 12 and not any(r["has_movement_channel"] or r["aux_unidentified"] for r in p4))
    p1 = [r for r in inv if r["patient"] == "P1"]
    p1r = [r for r in p1 if r["kind"] == "ramp"]
    p1_aux = [c for c in chans if c["patient"] == "P1" and c["category"] == "aux"]
    p1_ident = sum(1 for c in p1_aux if c["identical_to_index"] not in (None, ""))
    p1_mv = [m for m in mv if m["patient"] == "P1"]
    lowfrac = [m.get("frac_0p5_20hz") for m in p1_mv if m.get("frac_0p5_20hz") is not None]
    highfrac = [m.get("frac_500_2048hz") for m in p1_mv if m.get("frac_500_2048hz") is not None]
    chk("P1 has 2 ramp files (0.28 h) whose only aux channels are three identically named AUX1 traces, unidentified",
        f"P1: {len(p1r)} ramp files, {hours_of(p1r):.2f} h; aux labels per file: "
        + "; ".join(f"{r['file']}: {r['aux_unidentified']}" for r in p1)
        + f"; {p1_ident} of {len(p1_aux)} AUX traces identical to an earlier trace; accelerometer or EMG channels: "
        + fmt_list(sorted({x for r in p1 for x in (r['head_accelerometer'] + ' ' + r['neck_emg'] + ' ' + r['other_accelerometer']).split()}))
        + (f"; power fraction 0.5 to 20 Hz over command-OFF cores: {min(lowfrac):.3f} to {max(lowfrac):.3f}, "
           f"500 to 2048 Hz: {min(highfrac):.3f} to {max(highfrac):.3f} (other groups in movement_channels.csv)" if lowfrac else ""),
        len(p1r) == 2 and f"{hours_of(p1r):.2f}" == "0.28"
        and all(r["aux_unidentified"].split() == ["AUX1"] * 3 for r in p1))
    p5 = [r for r in inv if r["patient"] == "P5"]
    chk("P5 has EMG and two triaxial accelerometers in its single-pulse file only (0.19 h), and no ramp",
        "; ".join(f"{r['file']} ({r['kind']}, {r['duration_h']:.2f} h): EMG {r['neck_emg'] or 'none'}; accelerometers "
                  f"{(r['head_accelerometer'] + ' ' + r['other_accelerometer']).strip() or 'none'}" for r in p5),
        len(p5) == 1 and p5[0]["kind"] == "single-pulse" and f"{p5[0]['duration_h']:.2f}" == "0.19"
        and len((p5[0]["head_accelerometer"] + " " + p5[0]["other_accelerometer"]).split()) == 6 and p5[0]["neck_emg"] != "")
    sp = [r for r in inv if r["kind"] == "single-pulse"]
    pp = [r for r in inv if r["kind"] == "paired-pulse"]
    chk("Single-pulse files exist for P5 left, P6 left and right, P7 left and right; one paired-pulse file for P7 left",
        "single-pulse: " + " ".join(f"{r['patient']}{r['hemisphere']}" for r in sp) + "; paired-pulse: "
        + " ".join(f"{r['patient']}{r['hemisphere']}" for r in pp),
        sorted(f"{r['patient']}{r['hemisphere']}" for r in sp) == ["P5L", "P6L", "P6R", "P7L", "P7R"]
        and [f"{r['patient']}{r['hemisphere']}" for r in pp] == ["P7L"])
    dur = {}
    for p in ("P2", "P3", "P6", "P7"):
        ds = [r["duration_s"] for r in ramp_inv if r["patient"] == p]
        dur[p] = (min(ds), max(ds))
    p67 = (min(dur["P6"][0], dur["P7"][0]), max(dur["P6"][1], dur["P7"][1]))
    chk("Ramp file durations: 115 to 506 s for P6 and P7, 453 to 883 s for P3, 768 to 1,018 s per file for P2",
        f"P6 and P7 {p67[0]:.0f} to {p67[1]:.0f} s; P3 {dur['P3'][0]:.0f} to {dur['P3'][1]:.0f} s; "
        f"P2 {dur['P2'][0]:.0f} to {dur['P2'][1]:.0f} s",
        (round(p67[0]), round(p67[1]), round(dur["P3"][0]), round(dur["P3"][1]), round(dur["P2"][0]), round(dur["P2"][1]))
        == (115, 506, 453, 883, 768, 1018))
    by_pat = {}
    for s in ramp_sum:
        by_pat.setdefault(s["patient"], []).append(s)
    on_by_pat = {p: [r["duration_s"] for r in ramp_seg if r["patient"] == p and r["state"] == "ON" and not r.get("short_block")]
                 for p in by_pat}
    off_by_pat = {p: [r["duration_s"] for r in ramp_seg if r["patient"] == p and r["state"] == "OFF" and r.get("position") == "interior"]
                  for p in by_pat}
    pat_list = sorted(by_pat, key=lambda p: int(p[1:]))
    patterns = {}
    for s in ramp_sum:
        patterns[s["step_pattern"]] = patterns.get(s["step_pattern"], 0) + 1
    single = [s for s in ramp_sum if s["step_pattern"] == "single_block"]
    p3l5 = [s for s in ramp_sum if s["file"] == "P3_L5_Testing.mat"]
    if p3l5:
        s = p3l5[0]
        steps = [float(x) for x in s["amplitudes_mA"].split()]
        ok = (s["step_pattern"] == "ascending_ramp" and all(abs(steps[i + 1] - steps[i] - 0.5) < 1e-9 for i in range(len(steps) - 1))
              and 25 <= s["on_s_min"] and s["on_s_max"] <= 35 and round(s["interior_off_s_min"]) >= 15 and round(s["interior_off_s_max"]) <= 25)
        chk("On P3_L5_Testing.mat stimulation alternates about 30 s ON blocks per 0.5 mA step with 15 to 25 s OFF blocks",
            f"P3_L5_Testing.mat: {s['step_pattern']}, amplitudes {s['amplitudes_mA']} mA, ON {s['on_s_min']} to {s['on_s_max']} s, "
            f"interior OFF {s['interior_off_s_min']} to {s['interior_off_s_max']} s", ok)
    gen_ok = (set(patterns) == {"ascending_ramp"}
              and all(25 <= np.median(on_by_pat[p]) <= 35 for p in pat_list if on_by_pat[p]))
    chk("Stimulation during oxford ramps alternates about 30 s ON blocks per 0.5 mA step with 15 to 25 s OFF blocks (generalized from one file)",
        "step patterns over ramp files: " + ", ".join(f"{k} {v}" for k, v in sorted(patterns.items()))
        + "; ON blocks of 5 s or more, median (min to max) s per patient: "
        + "; ".join(f"{p} {np.median(on_by_pat[p]):.1f} ({min(on_by_pat[p]):.1f} to {max(on_by_pat[p]):.1f})" for p in pat_list if on_by_pat[p])
        + "; interior OFF blocks: "
        + "; ".join(f"{p} {np.median(off_by_pat[p]):.1f} ({min(off_by_pat[p]):.1f} to {max(off_by_pat[p]):.1f})" for p in pat_list if off_by_pat[p])
        + "; single-block files: " + fmt_list([s["file"] for s in single]),
        gen_ok)
    p6r3 = [s for s in ramp_sum if s["file"] == "P6_R3_Testing.mat"]
    if p6r3:
        s = p6r3[0]
        chk("P6_R3_Testing.mat shows 29.4 s OFF then 83.7 s at 2.0 mA",
            f"leading OFF {s['leading_off_s']} s, ON blocks {s['n_on_blocks']} at {s['amplitudes_mA']} mA lasting {s['on_s_max']} s, trailing OFF {s['trailing_off_s']} s",
            s["leading_off_s"] == 29.4 and s["on_s_max"] == 83.7 and s["amplitudes_mA"] == "2")
    cmd_names = {}
    for r in inv:
        cmd_names.setdefault(r["patient"], set()).update(r["stim_command_active"].split())
    no_cmd = [r["file"] for r in inv if not r["stim_command"]]
    zero_cmd = [r["file"] for r in inv if r["stim_command"] and not r["stim_command_active"]]
    chk("The time stamps and intensity of the applied stimulation currents are saved in channel Cur1/Am1",
        "active command channel per patient: " + "; ".join(f"{p} {' '.join(sorted(v)) or 'none'}" for p, v in sorted(cmd_names.items(), key=lambda t: int(t[0][1:])))
        + "; files with no command channel: " + fmt_list(no_cmd) + "; files whose command channels are all zero: " + fmt_list(zero_cmd),
        not no_cmd and not zero_cmd, source="dataset README")
    n_pass = sum(1 for s in ramp_sum if s["g15_pass"])
    det = [g for g in g15 if g.get("detected")]
    all_l = [g["onset_lag_samples"] for g in det]
    comp = [g for g in det if g["delivery"] == "complete"]
    all_o = [g["offset_lag_samples"] for g in comp]
    interrupted = [g for g in g15 if g["delivery"] == "interrupted"]
    all_lms = [g["onset_lag_ms"] for g in det]
    chk("G15 (oxford): the stimulation current channel and the LFP stimulation artefact coincide within one sample in every ramp file",
        f"{n_pass} of {len(ramp_sum)} ramp files pass; {sum(1 for g in g15 if g['within_1_sample'])} of {len(g15)} ON transitions within 1 sample; "
        f"artefact detected in {len(all_l)} of {len(g15)}; first pulse {min(all_l)} to {max(all_l)} samples after the command rise "
        f"({min(all_lms):.1f} to {max(all_lms):.1f} ms); in the {len(comp)} blocks with a complete train the last pulse lies {min(all_o)} to {max(all_o)} samples after the command fall; "
        f"{len(interrupted)} blocks interrupted, {len(g15) - len(det)} without artefact"
        if all_l else f"{n_pass} of {len(ramp_sum)} ramp files pass; no artefact detected",
        n_pass == len(ramp_sum))
    write_csv(os.path.join(out, "addendum_checks.csv"), checks)

    # summary.md
    L = []
    L.append("# oxford_dystonia_lfp: inventory, ramp steps and G15 (scripts/probe_oxford_lfp.py)\n")
    L.append("Generated by the script; every number below comes from the CSV files beside this one.\n")
    L.append("## Files\n")
    L.append(f"- {len(inv)} .mat files in {len(zips)} zips, read in memory; MAT versions: "
             + ", ".join(f"{v} ({sum(1 for r in inv if r['mat_version'] == v)})" for v in sorted({r['mat_version'] for r in inv}))
             + "; readers used: " + ", ".join(sorted({r['reader'] for r in inv})) + ".")
    L.append(f"- Kinds: {nk['ramp']} ramp, {nk['single-pulse']} single-pulse, {nk['paired-pulse']} paired-pulse, {nk['other']} other; "
             f"{hours_of(inv):.2f} h in total, {ramp_h:.2f} h of ramps.")
    L.append("- Per patient (files, ramp files, hours): " + "; ".join(
        f"{c['group']} {c['n_files']}, {c['n_ramp']}, {c['hours_all']:.2f} h" for c in counts if c["scope"] == "patient") + ".")
    fss = sorted({r["fs_hz"] for r in inv})
    L.append(f"- Sampling rate: one Fs per file, values {', '.join(f'{v:g}' for v in fss)} Hz; every channel is stored at that rate. "
             "No unit is stored in the files. The command channel values (500 to 4500) are read as uA.")
    notes = [r for r in inv if r["note"]]
    if notes:
        L.append("- Naming: " + "; ".join(f"{r['file']}: {r['note']}" for r in notes) + ".")
    incons = [r["file"] for r in inv if r["level_consistent"] is False]
    L.append(f"- Contact level from the file name against the bipolar label: {sum(1 for r in inv if r['level_consistent'])} consistent, "
             f"{len(incons)} inconsistent ({fmt_list(incons)}).")
    L.append("\n## Ramp step structure\n")
    L.append("- Step patterns: " + ", ".join(f"{k} {v}" for k, v in sorted(patterns.items())) + ".")
    L.append(f"- Down-steps (ON to lower ON without OFF): {sum(s['n_down_steps'] for s in ramp_sum)}; ON blocks lower than the previous ON block: "
             f"{sum(s['n_lower_than_previous_on_block'] for s in ramp_sum)}; repeated amplitudes: {sum(s['n_repeat_amplitude'] for s in ramp_sum)} "
             f"(files: {fmt_list([s['file'] for s in ramp_sum if s['n_repeat_amplitude']])}); ON blocks under {SHORT_BLOCK_S:g} s: "
             f"{sum(s['n_short_on_blocks'] for s in ramp_sum)} (files: {fmt_list([s['file'] for s in ramp_sum if s['n_short_on_blocks']])}).")
    mx = {}
    for s in ramp_sum:
        mx.setdefault(s["max_mA"], []).append(s["file"])
    L.append("- Highest amplitude per file (mA: n files): " + ", ".join(f"{k:g}: {len(v)}" for k, v in sorted(mx.items())) + ".")
    for p in pat_list:
        if on_by_pat[p]:
            L.append(f"- {p}: ON blocks of {SHORT_BLOCK_S:g} s or more, median {np.median(on_by_pat[p]):.1f} s (range {min(on_by_pat[p]):.1f} to {max(on_by_pat[p]):.1f}); "
                     + (f"interior OFF median {np.median(off_by_pat[p]):.1f} s (range {min(off_by_pat[p]):.1f} to {max(off_by_pat[p]):.1f})." if off_by_pat[p] else "no interior OFF block."))
    fr = [s["stim_freq_hz_median"] for s in ramp_sum if s["stim_freq_hz_median"]]
    if fr:
        L.append(f"- Stimulation frequency from the artefact pulse train: {min(fr):.2f} to {max(fr):.2f} Hz across ramp files.")
    rho = [s["artefact_peak_vs_amplitude_spearman"] for s in ramp_sum if s["artefact_peak_vs_amplitude_spearman"] is not None]
    if rho:
        L.append(f"- Artefact peak against command amplitude within file (Spearman, data-quality check): {min(rho):.3f} to {max(rho):.3f} over {len(rho)} files.")
    L.append("\n## G15, oxford part\n")
    L.append(f"- Rule: the first artefact pulse of each ON block lies within one sample of the command rise. {n_pass} of {len(ramp_sum)} ramp files pass.")
    if all_l:
        by_cmd = {}
        for s in ramp_sum:
            by_cmd.setdefault(s["command_channel"], set()).add(s["file"])
        parts = []
        for c, files in sorted(by_cmd.items()):
            v = [g["onset_lag_samples"] for g in det if g["file"] in files]
            vm = [g["onset_lag_ms"] for g in det if g["file"] in files]
            if v:
                parts.append(f"{c} ({len(files)} files) {min(v)} to {max(v)} samples ({min(vm):.1f} to {max(vm):.1f} ms)")
        L.append(f"- Over {len(all_l)} detected ON transitions (of {len(g15)}), the first pulse follows the command rise by {min(all_l)} to {max(all_l)} samples "
                 f"(median {np.median(all_l):.0f}); per command channel: " + "; ".join(parts) + ".")
        spread = [s["onset_lag_max_samples"] - s["onset_lag_min_samples"] for s in ramp_sum if s["onset_lag_min_samples"] is not None]
        L.append(f"- Within-file spread of the onset lag (max minus min): {min(spread)} to {max(spread)} samples.")
        L.append(f"- In the {len(comp)} ON blocks whose pulse train runs to the command fall, the last pulse lies {min(all_o)} to {max(all_o)} samples "
                 f"after the fall (median {np.median(all_o):.0f}).")
        if interrupted:
            L.append(f"- Interrupted ON blocks (train stops more than {INTERRUPT_S:g} s before the command fall, or pauses longer than {INTERRUPT_S:g} s): "
                     + "; ".join(f"{g['file']} block {g['on_block']} ({g['amplitude_uA']:g} uA, command {g['command_on_s']:.1f} s, train {g['artefact_train_s']:.1f} s, "
                                 f"longest pause {g['longest_pulse_gap_s']:.2f} s)" for g in interrupted) + ".")
        c2 = [s["ch2_minus_ch1_onset_max_abs_samples"] for s in ramp_sum if s["ch2_minus_ch1_onset_max_abs_samples"] is not None]
        if c2:
            L.append(f"- Second artefact channel against the first: onset difference at most {max(c2)} samples in any file.")
    und = [g for g in g15 if not g.get("detected")]
    if und:
        L.append("- ON blocks without a detected artefact: " + "; ".join(f"{g['file']} block {g['on_block']} ({g['command_on_s']} s at {g['amplitude_uA']:g} uA)" for g in und) + ".")
    f10 = [g["first10_peak_ratio"] for g in comp if g.get("first10_peak_ratio") is not None]
    l10 = [g["last10_peak_ratio"] for g in comp if g.get("last10_peak_ratio") is not None]
    if f10:
        L.append(f"- Delivered ramp-up and ramp-down: median artefact peak of the first {EDGE_PULSES} pulses over the block median, {min(f10):.3f} to {max(f10):.3f}; "
                 f"of the last {EDGE_PULSES} pulses, {min(l10):.3f} to {max(l10):.3f} ({len(f10)} complete blocks).")
    ro = [g["rise_to_onset_max_ratio"] for g in det if g.get("rise_to_onset_max_ratio") is not None]
    pr = [g["pre_rise_max_ratio"] for g in det if g.get("pre_rise_max_ratio") is not None]
    lo_ = [g["last_to_offmargin_max_ratio"] for g in comp if g.get("last_to_offmargin_max_ratio") is not None]
    if ro:
        L.append(f"- Hidden low-amplitude stimulation check: between the command rise and the first detected pulse, the largest |dx| on the artefact channel is "
                 f"{min(ro):.1f} to {max(ro):.1f} times the OFF median (median {np.median(ro):.1f}, {len(ro)} blocks); in a window of the same length before the rise, "
                 f"{min(pr):.1f} to {max(pr):.1f} (median {np.median(pr):.1f}); from the last pulse to {OFF_MARGIN_S:g} s after the command fall, "
                 f"{min(lo_):.1f} to {max(lo_):.1f} (median {np.median(lo_):.1f}). Detected pulses sit above {THR_MAD:g} times the OFF median.")
        by_cmd_t = {}
        for g in det:
            if g.get("rise_to_onset_first_over_10x_lag_samples") is not None:
                c = [s["command_channel"] for s in ramp_sum if s["file"] == g["file"]][0]
                by_cmd_t.setdefault(c, []).append((g["rise_to_onset_first_over_10x_lag_samples"], g["rise_to_onset_n_samples_over_10x"],
                                                   g["onset_lag_samples"]))
        for c, v in sorted(by_cmd_t.items()):
            n_c = sum(1 for g in det if [s["command_channel"] for s in ramp_sum if s["file"] == g["file"]][0] == c)
            L.append(f"- {c}: in {len(v)} of {n_c} detected blocks at least one sample between the command rise and the first pulse exceeds "
                     f"{REFINE_MAD:g} times the OFF median ({min(x[1] for x in v)} to {max(x[1] for x in v)} such samples per block); the first of them "
                     f"lies {min(x[0] for x in v)} to {max(x[0] for x in v)} samples after the rise and {min(x[2] - x[0] for x in v)} to "
                     f"{max(x[2] - x[0] for x in v)} samples before the first pulse.")
    stray = [s for s in ramp_sum if s["stray_periodic_pulses_in_off"]]
    L.append(f"- Periodic artefact pulses inside command-OFF segments (from {OFF_MARGIN_S:g} s after a fall): "
             + (", ".join(f"{s['file']} {s['stray_periodic_pulses_in_off']}" for s in stray) if stray else "none") + ".")
    gcds = sorted({s["command_edge_gcd_samples"] for s in ramp_sum if s["command_edge_gcd_samples"] is not None})
    L.append(f"- Greatest common divisor of command edge positions, per file: values {', '.join(str(v) for v in gcds)} samples.")
    L.append("- Failing files: " + fmt_list([s["file"] for s in ramp_sum if not s["g15_pass"]]) + ".")
    L.append("\n## Movement channels\n")
    for m in mh:
        L.append(f"- {m['patient']}{m['hemisphere']}: {m['n_ramp']} ramp files ({m['ramp_hours']:.2f} h); head accelerometer in {m['n_ramp_with_head_accelerometer']} "
                 f"({m['head_accelerometer_labels'] or 'none'}); other accelerometer in {m['n_ramp_with_other_accelerometer']} ({m['other_accelerometer_labels'] or 'none'}); "
                 f"neck EMG in {m['n_ramp_with_neck_emg']} ({m['neck_emg_labels'] or 'none'}); aux in {m['n_ramp_with_aux']} ({m['aux_labels'] or 'none'})"
                 + (f"; non-ramp: {m['non_ramp_files']} with {m['non_ramp_movement_labels'] or 'no movement channel'}" if m["non_ramp_files"] else "") + ".")
    grp = {}
    for m in mv:
        if m.get("frac_0p5_20hz") is not None:
            grp.setdefault((m["role"], m["group"]), []).append(m)
    L.append("- Spectral profile of movement and aux channels (median fraction of power in 0.5 to 20 Hz, 20 to 500 Hz, 500 to 2048 Hz; "
             "command-OFF cores, or the whole file when there is no active command channel): "
             + "; ".join(f"{g} ({r}, {len(v)} traces) {np.median([x['frac_0p5_20hz'] for x in v]):.3f}, "
                         f"{np.median([x['frac_20_500hz'] for x in v]):.3f}, {np.median([x['frac_500_2048hz'] for x in v]):.3f}"
                         for (r, g), v in sorted(grp.items())) + ".")
    flat_mv = [f"{m['file']} {m['label']}" for m in mv if m["flat"]]
    L.append("- Flat movement or aux channels: " + fmt_list(flat_mv) + ".")
    tl = {}
    for r in ramp_seg:
        if r["state"] == "ON" and r.get("telemetry_first_nonzero_lag_samples") not in (None, "") and r.get("onset_lag_samples") not in (None, ""):
            tl.setdefault(r["telemetry_channel"], []).append((r["telemetry_first_nonzero_lag_samples"], r["onset_lag_samples"]))
    L.append("\n## Telemetry channel\n")
    for c, v in sorted(tl.items()):
        a_ = [x[0] for x in v]
        d_ = [x[1] - x[0] for x in v]
        fs_c = [s for s in ramp_sum if s["telemetry_channel"] == c]
        nz_off = sum(s["telemetry_nonzero_samples_while_command_zero"] or 0 for s in fs_c)
        z_on = sum(s["telemetry_zero_samples_while_command_on"] or 0 for s in fs_c)
        L.append(f"- {c} (undocumented; its values lie in the range of a lead impedance in ohm, a reading, not a documented fact): "
                 f"{nz_off} nonzero samples while the command is 0 and {z_on} zero samples while the command is ON, over {len(fs_c)} ramp files; "
                 f"first nonzero sample {min(a_)} to {max(a_)} samples after the command rise; the first artefact pulse follows it by "
                 f"{min(d_)} to {max(d_)} samples ({len(v)} ON blocks).")
    L.append("\n## Addendum statements\n")
    for c in checks:
        L.append(f"- {c['verdict']}: {c['claim']}. Measured: {c['measured']}.")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", shown(os.path.join(out, "summary.md")))
    zinfo = []
    for z in zips:
        st = os.stat(z)
        zinfo.append({"zip": os.path.basename(z), "bytes": st.st_size})
    write_provenance(a, {"source": RAW_REL, "zips": zinfo, "n_mat": len(inv), "extracted_folders": extracted,
                         "cache_version": CACHE_VERSION,
                         "parameters": {"TRIM_S": TRIM_S, "CORE_MIN_S": CORE_MIN_S, "THR_FRAC": THR_FRAC, "THR_MAD": THR_MAD,
                                        "REFINE_MAD": REFINE_MAD, "REFINE_MAX": REFINE_MAX, "REFR_FRAC": REFR_FRAC,
                                        "ITI_LO": ITI_LO, "ITI_HI": ITI_HI, "N_REGULAR": N_REGULAR,
                                        "OFF_MARGIN_S": OFF_MARGIN_S, "SHORT_BLOCK_S": SHORT_BLOCK_S,
                                        "SPEC_NPERSEG": SPEC_NPERSEG, "BANDS": BANDS},
                         "note": "descriptive only; no statistic of Q1 or Q2"})


if __name__ == "__main__":
    main()
