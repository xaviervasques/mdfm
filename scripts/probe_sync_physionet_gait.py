"""physionet_multimodal_gait: synchronisation of EEG, IMU and force (task N0.3c, addendum gate G15).

G15 for this set: the lags between EEG, IMU and force, measured from shared events, must be
constant within 20 ms across the 3 treadmill speeds in at least 90 percent of subjects;
otherwise the set serves PRE only. This script measures those lags without any label.

Reads (read-only) the raw release under
<moveatlas>/datasets/eeg/physionet_multimodal_gait/raw/: EEG_Data/EEG.csv/*.csv (DSI-24 export,
header lines, Trigger, Time_Offset, ADC_Sequence columns), EEG_Data/EEG.edf/*.edf (header, Trigger
channel and EDF+ annotation texts only), FP_Data/*.csv (Bertec plates),
EMG_Data/*.csv (Delsys MAV envelope, timing facts only), IMU_Data/IMU.tar.gz streamed in memory
(Sensor_Free_Acceleration.csv, General_Information.csv, Markers.csv of every MVN folder),
Project_Content.pdf (synchronisation paragraph, through pdftotext), and the four pooled canonical
manifests (record membership only). subject-info.csv is never opened (it carries dates of birth).

Streams and clocks. Every file starts at its own time zero (EEG Time column, IMU Frame / frame
rate, force sample index / 1000 Hz). A lag is estimated on those file clocks.
Lag convention: lag_A_B = tau that maximises sum_t a(t + tau) b(t); a shared event seen at time
t_B on the B clock is seen at t_B + lag_A_B on the A clock.

Shared events, no label used:
- IF, IMU against force (primary): Newton's second law. The total vertical ground reaction force
  (left plate Fz + right plate Fz) and the vertical free acceleration of the pelvis sensor (MVN
  free acceleration is gravity-free, global frame, z up) carry the same vertical centre-of-mass
  dynamics. Both low-passed at 12 Hz (zero phase). The basin of the true alignment is found on the
  sum of this correlation and of the side-matched impact correlation below, over +-10 s, then
  refined on the Newton correlation alone within +-100 ms.
- IF_side (sensitivity): heel-strike impact energy, left plate Fz against left foot sensor and
  right against right (energy of the 8-25 Hz band of Fz, and of the 6-25 Hz band of the 3 foot
  acceleration axes, each smoothed at 15 Hz).
- EI, EEG against IMU (primary): the gait motion artefact. EEG: 20-45 Hz power (Hilbert) of every
  scalp channel that is not railed, each normalised by its median, averaged, log, smoothed at
  15 Hz. IMU: 6-25 Hz acceleration energy of the head sensor, smoothed at 15 Hz.
- EI_alt (sensitivity): EEG 8-45 Hz power against the head acceleration magnitude.
- EF, EEG against force (primary): the same EEG artefact envelope against the impact energy of
  both plates.
All filters are zero phase (filtfilt). Signals are interpolated to a common 1 kHz grid.

Uncertainty: each record is cut into 6 blocks of the reference stream; the lag is re-estimated
leaving each block out (jackknife SE) and within each block (block lags, from which a linear
drift in ms per minute is fitted). The alignment basin is called identified when the global
maximum over +-10 s falls in the same +-100 ms basin in all 6 leave-one-block-out replicates.
A record lag is 'measured' when it is identified, not at the edge of its window, and its
jackknife SE is at most 5 ms.

Per subject and pair: the spread (max - min) of the measured lags over the 3 speeds, a Monte
Carlo interval of that spread from the record SEs, and the class within 20 ms (point estimate).
G15 passes for a pair when at least 90 percent of the evaluable subjects are within 20 ms. The
addendum's three-outcome label is reported beside it: pass on the point estimate, absent when the
subject bootstrap interval of the fraction lies entirely below 0.90, undetermined otherwise. A second
reading of the tolerance (every lag within +-20 ms of one value per subject, a range up to 40 ms) is
reported as a sensitivity with its own random generator; the gate keeps the range up to 20 ms.

Which stream moves: on records with the three primary lags, the lags are centred per subject and their
pooled within-subject covariances (records minus subjects degrees of freedom) give the implied SD of
each file start, under independent start offsets.

Duplicates: the signal arrays of every record are hashed (header lines excluded). Records whose
EEG data are identical to another record's are left out of the EEG pairs for the gate (both
members, since the files do not say which subject a copy belongs to); the gate is also reported
as released, keeping them.

Writes under reports/sync_physionet_gait/: records.csv, subjects.csv, summary.json, summary.md,
sync_lags.png, _provenance.json. Descriptive only: no coupling statistic is computed.

Run: <conda>/python.exe scripts/probe_sync_physionet_gait.py
     [--moveatlas <moveatlas>] [--out DIR] [--subjects S1,S4]  (subset = debug only)
Runtime on the PC: about 5 minutes for the full set (the IMU archive is streamed once), so the
script is not resumable; it rewrites all its outputs on each run.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_args, write_provenance, read_csv  # noqa: E402

SRC = "physionet_multimodal_gait"
REL = "datasets/eeg/physionet_multimodal_gait/raw"
SPEEDS = ["0.5", "0.75", "1"]
TASK = {"0.5": "walk_0.50ms", "0.75": "walk_0.75ms", "1": "walk_1.00ms"}
SITES = ["P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Fp1", "Fp2", "T3", "T5", "O1", "O2",
         "F7", "F8", "T6", "T4"]
IMU_SENSORS = ["Head", "Pelvis", "Left Foot", "Right Foot"]
RAIL_UV = 9830.0          # DSI-24 rail value (+-9830.1 uV), datacard
RAIL_MAX_FRAC = 0.01      # a channel with more railed samples is left out of the envelope
MIN_EEG_CH = 6
FS_GRID = 1000.0          # common grid, Hz
WIDE_MS = 10000           # basin search range
BASIN_MS = 100            # refinement window around the basin
ALIAS_EXCL_MS = 250       # margin: best correlation outside +-250 ms of the basin
K_BLOCKS = 6
SE_MAX_MS = 5.0
TOL_MS = 20.0
PASS_FRAC = 0.90
N_MC = 4000
N_BOOT = 2000
SEED = 20260928
PAIRS_PRIMARY = ["EI", "IF", "EF"]
PAIRS_SENS = ["IF_side", "EI_alt"]
PAIR_TEXT = {
    "EI": "EEG against IMU: EEG 20-45 Hz artefact envelope vs head 6-25 Hz acceleration energy",
    "IF": "IMU against force: pelvis vertical free acceleration vs total vertical force (Newton)",
    "EF": "EEG against force: EEG 20-45 Hz artefact envelope vs impact energy of both plates",
    "IF_side": "IMU against force (sensitivity): foot 6-25 Hz energy vs same-side plate 8-25 Hz energy",
    "EI_alt": "EEG against IMU (sensitivity): EEG 8-45 Hz envelope vs head acceleration magnitude",
}


# ------------------------------------------------------------------------------ arguments
def get_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--subjects", default="")
    known, rest = pre.parse_known_args()
    if not any(x == "--out" or x.startswith("--out=") for x in rest):
        rest += ["--out", os.path.join(here, "reports", "sync_physionet_gait")]
    sys.argv = [sys.argv[0]] + rest
    a = parse_args(__doc__.splitlines()[0])
    a.subjects = [s.strip() for s in known.subjects.split(",") if s.strip()]
    return a


# ------------------------------------------------------------------------------ readers
def read_eeg(path):
    header, hdr_idx = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("Time,"):
                hdr_idx = i
                break
            parts = line.rstrip("\r\n").split(",")
            key = parts[0].lstrip("#").strip()
            key = re.sub(r"\s*=\s*$", "", key)
            header[key] = parts[1].strip() if len(parts) > 1 else ""
    if hdr_idx is None:
        raise ValueError("no Time column header")
    df = pd.read_csv(path, skiprows=hdr_idx, header=0, encoding="utf-8", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return header, df


def edf_facts(path):
    """The EDF+ export of the same EEG record: duration, Trigger channel changes and the texts of the
    annotation channel (timekeeping entries, which carry no text, are skipped)."""
    out = {}
    with open(path, "rb") as f:
        h = f.read(256)
        ns, nrec, dur = int(h[252:256]), int(h[236:244]), float(h[244:252])
        hdr = f.read(256 * ns)
        lab = [hdr[i * 16:(i + 1) * 16].decode("latin-1").strip() for i in range(ns)]
        b = ns * (16 + 80 + 8 * 5 + 80)
        nsamp = [int(hdr[b + i * 8:b + i * 8 + 8].decode("latin-1").strip()) for i in range(ns)]
        data = f.read()
    recsz = 2 * sum(nsamp)
    out["edf_dur_s"] = nrec * dur
    texts, trig = [], []
    ai = lab.index("EDF Annotations") if "EDF Annotations" in lab else None
    ti = lab.index("Trigger") if "Trigger" in lab else None
    for r in range(nrec):
        blk = data[r * recsz:(r + 1) * recsz]
        if ai is not None:
            o = 2 * sum(nsamp[:ai])
            for tal in blk[o:o + 2 * nsamp[ai]].split(b"\x00"):
                texts += [t.decode("latin-1").strip() for t in tal.split(b"\x14")[1:] if t.strip()]
        if ti is not None:
            o = 2 * sum(nsamp[:ti])
            trig.append(np.frombuffer(blk[o:o + 2 * nsamp[ti]], dtype="<i2"))
    out["edf_annotations_n"] = len(texts)
    out["edf_annotation_texts"] = ";".join(sorted(set(texts)))
    if trig:
        tr = np.concatenate(trig)
        out["edf_trigger_changes"] = int(np.sum(np.diff(tr) != 0))
    return out


def read_force(path):
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    has_header = "Fz" in first
    df = pd.read_csv(path, header=0 if has_header else None, encoding="utf-8")
    arr = df.values.astype(float)
    if arr.shape[1] != 22:
        raise ValueError(f"force file with {arr.shape[1]} columns")
    if has_header:
        cols = [str(c) for c in df.columns]
        assert cols[5] == "1:Fz" and cols[16] == "2:Fz", cols
    return arr, has_header


def read_emg(path):
    df = pd.read_csv(path, encoding="utf-8")
    return df.values.astype(float)


def stream_imu(tar_path, wanted):
    """Returns folder -> dict(frame, acc{sensor: n x 3}, info{}, markers, empty)."""
    out = {}
    t = tarfile.open(tar_path, "r|gz")
    for m in t:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        if len(parts) < 2:
            continue
        folder, base = parts[-2], parts[-1]
        if wanted and folder not in wanted:
            continue
        rec = out.setdefault(folder, {"empty": None, "info": {}, "markers": None})
        if base == "Sensor_Free_Acceleration.csv":
            raw = t.extractfile(m).read()
            if len(raw.strip()) == 0 or raw.count(b"\n") < 3:
                rec["empty"] = True
                continue
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
            rec["empty"] = False
            rec["frame"] = df["Frame"].values.astype(np.int64)
            rec["acc"] = {s: df[[f"{s} x", f"{s} y", f"{s} z"]].values.astype(float) for s in IMU_SENSORS}
        elif base == "General_Information.csv":
            txt = t.extractfile(m).read().decode("utf-8", "replace")
            for line in txt.splitlines():
                if "," in line:
                    k, v = line.split(",", 1)
                    rec["info"][k.strip()] = v.strip()
        elif base == "Markers.csv":
            txt = t.extractfile(m).read().decode("utf-8", "replace")
            rec["markers"] = max(0, len([x for x in txt.splitlines() if x.strip()]) - 1)
    return out


# ------------------------------------------------------------------------------ signals
def butter_filt(x, fs, lo=None, hi=None, order=4):
    if lo is not None and hi is not None:
        b, a = signal.butter(order, [lo, hi], btype="band", fs=fs)
    elif hi is not None:
        b, a = signal.butter(order, hi, btype="low", fs=fs)
    else:
        b, a = signal.butter(order, lo, btype="high", fs=fs)
    return signal.filtfilt(b, a, x, axis=0)


def band_energy(x, fs, lo, hi, smooth_hz=15.0):
    y = butter_filt(x, fs, lo, hi)
    e = (y ** 2).sum(axis=1) if y.ndim > 1 else y ** 2
    return np.clip(butter_filt(e, fs, hi=smooth_hz, order=2), 0, None)


def eeg_envelope(data, fs, lo, hi, smooth_hz=15.0):
    acc = 0.0
    for k in range(data.shape[1]):
        y = np.abs(signal.hilbert(butter_filt(data[:, k], fs, lo, hi))) ** 2
        acc = acc + y / max(np.median(y), 1e-12)
    acc = acc / data.shape[1]
    return butter_filt(np.log(np.clip(acc, 1e-6, None)), fs, hi=smooth_hz, order=2)


def heel_strikes(fz, fs=1000.0):
    """Stance onsets: upward crossing of 10 percent of the 5-95 percentile range, preceded by
    200 ms below and followed by 200 ms above. QC only (counts, cadence)."""
    lo, hi = np.percentile(fz, 5), np.percentile(fz, 95)
    if hi - lo <= 0:
        return np.array([])
    thr = lo + 0.1 * (hi - lo)
    above = fz > thr
    on = np.flatnonzero(~above[:-1] & above[1:]) + 1
    keep = [i for i in on if 200 <= i <= len(fz) - 250 and not above[i - 200:i].any() and above[i:i + 200].all()]
    return np.array(keep) / fs


def on_grid(t, x, T):
    return np.interp(np.arange(0.0, T, 1.0 / FS_GRID), t, x)


def zs(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


# ------------------------------------------------------------------------------ lag estimation
def xc_parts(pairs):
    """pairs: list of (a, b) on the 1 kHz grid, equal length. Block k keeps b only in block k.
    Returns lags (ms), parts (K x L, averaged over pairs), overlap counts, block centres (s)."""
    n = min(min(len(a), len(b)) for a, b in pairs)
    edges = np.linspace(0, n, K_BLOCKS + 1).astype(int)
    lags = np.arange(-(n - 1), n)
    sel = np.abs(lags) <= WIDE_MS
    parts = np.zeros((K_BLOCKS, int(sel.sum())))
    for a, b in pairs:
        a = zs(a[:n]); b = zs(b[:n])
        for k in range(K_BLOCKS):
            bk = np.zeros(n)
            bk[edges[k]:edges[k + 1]] = b[edges[k]:edges[k + 1]]
            parts[k] += signal.correlate(a, bk, mode="full", method="fft")[sel]
    parts /= len(pairs)
    return lags[sel].astype(float), parts, n - np.abs(lags[sel]), (edges[:-1] + edges[1:]) / 2.0 / FS_GRID


def parabolic(lags, c):
    j = int(np.argmax(c))
    d = 0.0
    if 0 < j < len(c) - 1:
        y0, y1, y2 = c[j - 1], c[j], c[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            d = 0.5 * (y0 - y2) / den
    return float(lags[j] + d * (lags[1] - lags[0])), float(c[j]), j


def estimate(coarse, fine=None):
    lags, parts, norm, centres = coarse
    C = parts.sum(0) / norm
    j0 = int(np.argmax(C))
    tau0 = lags[j0]
    out = {"basin_ms": float(tau0), "wide_edge": bool(j0 == 0 or j0 == len(C) - 1)}
    far = np.abs(lags - tau0) > ALIAS_EXCL_MS
    out["alias_ratio"] = float(C[far].max() / C[j0]) if far.any() and C[j0] > 0 else float("nan")
    stable = 0
    for k in range(K_BLOCKS):
        Ck = (parts.sum(0) - parts[k]) / norm
        if abs(lags[int(np.argmax(Ck))] - tau0) <= BASIN_MS:
            stable += 1
    out["basin_stable"] = stable
    fl, fp, fn, _ = fine if fine is not None else coarse
    w = np.abs(fl - tau0) <= BASIN_MS
    CF = fp.sum(0) / fn
    lag, r, j = parabolic(fl[w], CF[w])
    out["lag_ms"] = lag
    out["r"] = r
    out["at_edge"] = bool(j == 0 or j == int(w.sum()) - 1)
    loo = np.array([parabolic(fl[w], ((fp.sum(0) - fp[k]) / fn)[w])[0] for k in range(K_BLOCKS)])
    out["se_ms"] = float(np.sqrt((K_BLOCKS - 1) / K_BLOCKS * np.sum((loo - loo.mean()) ** 2)))
    blk = np.array([parabolic(fl[w], fp[k][w])[0] for k in range(K_BLOCKS)])
    x = centres / 60.0
    A = np.column_stack([x, np.ones_like(x)])
    coef, res, _, _ = np.linalg.lstsq(A, blk, rcond=None)
    resid = blk - A @ coef
    s2 = float(resid @ resid) / max(K_BLOCKS - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    out["drift_ms_per_min"] = float(coef[0])
    out["drift_se"] = float(np.sqrt(cov[0, 0]))
    out["block_lags_ms"] = ";".join(f"{v:.1f}" for v in blk)
    out["measured"] = bool(stable == K_BLOCKS and not out["at_edge"] and not out["wide_edge"]
                           and out["se_ms"] <= SE_MAX_MS)
    return out


# ------------------------------------------------------------------------------ per record
def eeg_facts(header, df, imu_info):
    f = {}
    t = df["Time"].values.astype(float)
    fs = float(header.get("Sample_Frequency_(Hz)", "nan") or "nan")
    f["eeg_n"] = len(t)
    f["eeg_fs_header"] = fs
    f["eeg_dur_s"] = float(t[-1] - t[0] + 1.0 / fs) if fs == fs else float("nan")
    f["eeg_t0_s"] = float(t[0])
    f["eeg_time_steps_off"] = int(np.sum(np.abs(np.diff(t) - 1.0 / fs) > 0.5 / fs)) if fs == fs else -1
    f["eeg_filter_delay_ms"] = header.get("Filter_Delay_(ms)", "")
    f["eeg_mains_hz"] = header.get("Mains_Frequency_(Hz)", "")
    f["eeg_trigger_source"] = header.get("Trigger Source", "")
    f["eeg_logger"] = header.get("Data_Logger", "")
    trig = pd.to_numeric(df["Trigger"], errors="coerce").values if "Trigger" in df else np.array([])
    u = np.unique(trig[~np.isnan(trig)]) if len(trig) else np.array([])
    f["eeg_trigger_values"] = ";".join(str(int(v)) for v in u[:6]) + (";..." if len(u) > 6 else "")
    f["eeg_trigger_changes"] = int(np.sum(np.diff(trig) != 0)) if len(trig) else -1
    if f["eeg_trigger_changes"] > 0:
        i = int(np.flatnonzero(np.diff(trig) != 0)[0])
        f["eeg_trigger_first_change_sample"] = i + 1
        f["eeg_trigger_first_transition"] = f"{int(trig[i])}>{int(trig[i + 1])}"
    if "Event" in df:
        ev = pd.to_numeric(df["Event"], errors="coerce").values
        f["eeg_event_nonzero"] = int(np.nansum(ev != 0))
    to = pd.to_numeric(df["Time_Offset"], errors="coerce").values if "Time_Offset" in df else None
    if to is not None and len(to) > 1:
        f["eeg_timeoffset_t0"] = float(to[0])
        f["eeg_timeoffset_rate_hz"] = float((to[-1] - to[0]) / (t[-1] - t[0]))
        f["eeg_timeoffset_nonmonotonic"] = int(np.sum(np.diff(to) <= 0))
    seq = pd.to_numeric(df["ADC_Sequence"], errors="coerce").values if "ADC_Sequence" in df else None
    if seq is not None and len(seq) > 1:
        d = np.diff(seq)
        wraps = (d < 0) & (seq[1:] <= 1)
        f["eeg_adcseq_gaps"] = int(np.sum((d != 1) & ~wraps))
        f["eeg_adcseq_max"] = int(np.nanmax(seq))
    # start-time agreement with the IMU header, in seconds, whole hours (time zone) removed;
    # the dates themselves are never written
    try:
        e_dt = dt.datetime.strptime(header.get("Date", "") + " " + header.get("Time", ""), "%d/%m/%Y %H:%M:%S")
        i_dt = dt.datetime.strptime(imu_info.get("Recorded Date UTC", ""), "%Y-%m-%d %H:%M:%S")
        d = (e_dt - i_dt).total_seconds()
        h = int(round(d / 3600.0))
        f["eeg_minus_imu_start_hours"] = h
        f["eeg_minus_imu_start_resid_s"] = float(d - 3600 * h)
    except Exception:  # noqa: BLE001
        pass
    return f


def process_record(root, subj, spd, imu_rec, manifests):
    rid = f"{subj}_{spd}"
    row = {"record": rid, "subject": subj, "speed": spd, "task": TASK[spd]}
    for fam in ("eeg", "imu", "force", "emg"):
        row[f"in_manifest_{fam}"] = int((f"{SRC}:{subj}", TASK[spd]) in manifests.get(fam, set()))
    sig = {}
    # ---- EEG
    p = os.path.join(root, "Dataset", "EEG_Data", "EEG.csv", f"{subj}_{spd}_raw.csv")
    row["has_eeg"] = int(os.path.isfile(p))
    eeg_ok = False
    if row["has_eeg"]:
        header, df = read_eeg(p)
        row.update(eeg_facts(header, df, imu_rec["info"] if imu_rec else {}))
        missing = [s for s in SITES if s not in df.columns]
        row["eeg_montage_ok"] = int(not missing)
        # content hash of the 24 signal columns (header lines excluded), for duplicate detection
        row["eeg_data_sha1"] = hashlib.sha1(np.ascontiguousarray(df.iloc[:, 1:25].values.astype(float)).tobytes()).hexdigest()
        if not missing:
            fs = row["eeg_fs_header"]
            X = df[SITES].values.astype(float)
            railed = np.mean(np.abs(X) >= RAIL_UV, axis=0)
            good = [k for k in range(len(SITES)) if railed[k] <= RAIL_MAX_FRAC]
            row["eeg_channels_used"] = len(good)
            row["eeg_channels_railed"] = ";".join(SITES[k] for k in range(len(SITES)) if railed[k] > RAIL_MAX_FRAC)
            if len(good) >= MIN_EEG_CH:
                te = df["Time"].values.astype(float)
                sig["eeg_t"] = te
                sig["E"] = eeg_envelope(X[:, good], fs, 20.0, 45.0)
                sig["E_alt"] = eeg_envelope(X[:, good], fs, 8.0, 45.0)
                eeg_ok = True
    row["eeg_usable"] = int(eeg_ok)
    p = os.path.join(root, "Dataset", "EEG_Data", "EEG.edf", f"{subj}_{spd}_raw.edf")
    row["has_eeg_edf"] = int(os.path.isfile(p))
    if row["has_eeg_edf"]:
        row.update(edf_facts(p))
    # ---- IMU
    row["has_imu"] = int(imu_rec is not None)
    imu_ok = False
    if imu_rec is not None:
        row["imu_empty_export"] = int(bool(imu_rec.get("empty")))
        row["imu_markers_rows"] = imu_rec.get("markers")
        row["imu_frame_rate_header"] = imu_rec["info"].get("Frame Rate", "")
        row["imu_processing_quality"] = imu_rec["info"].get("Processing Quality", "")
        if not imu_rec.get("empty") and "frame" in imu_rec:
            fr = imu_rec["frame"]
            fsi = float(row["imu_frame_rate_header"] or 60)
            row["imu_data_sha1"] = hashlib.sha1(np.ascontiguousarray(np.hstack([imu_rec["acc"][s] for s in IMU_SENSORS])).tobytes()).hexdigest()
            row["imu_frames"] = len(fr)
            row["imu_frame0"] = int(fr[0])
            row["imu_frame_gaps"] = int(np.sum(np.diff(fr) != 1))
            row["imu_dur_s"] = float((fr[-1] - fr[0] + 1) / fsi)
            ti = fr / fsi
            acc = imu_rec["acc"]
            sig["imu_t"] = ti
            sig["H_hf"] = band_energy(acc["Head"], fsi, 6.0, 25.0)
            sig["H_abs"] = np.linalg.norm(acc["Head"], axis=1)
            sig["P_z"] = butter_filt(acc["Pelvis"][:, 2], fsi, hi=12.0)
            sig["FootL_hf"] = band_energy(acc["Left Foot"], fsi, 6.0, 25.0)
            sig["FootR_hf"] = band_energy(acc["Right Foot"], fsi, 6.0, 25.0)
            imu_ok = True
    row["imu_usable"] = int(imu_ok)
    # ---- force
    p = os.path.join(root, "Dataset", "FP_Data", f"FP_{subj}_{spd}.csv")
    row["has_force"] = int(os.path.isfile(p))
    force_ok = False
    if row["has_force"]:
        arr, has_header = read_force(p)
        row["force_header_line"] = int(has_header)
        row["force_n"] = len(arr)
        row["force_dur_s"] = len(arr) / 1000.0
        row["force_data_sha1"] = hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()
        row["force_timestamp_all_zero"] = int(np.all(arr[:, [0, 11]] == 0))
        row["force_sync_all_zero"] = int(np.all(arr[:, [1, 12]] == 0))
        row["force_aux_all_zero"] = int(np.all(arr[:, [2, 13]] == 0))
        fzL, fzR = arr[:, 5], arr[:, 16]
        row["force_fz_sum_median"] = float(np.median(fzL + fzR))
        hsL, hsR = heel_strikes(fzL), heel_strikes(fzR)
        row["force_hs_L"] = len(hsL)
        row["force_hs_R"] = len(hsR)
        if len(hsL) > 3:
            row["stride_s_median"] = float(np.median(np.diff(hsL)))
        tf = np.arange(len(arr)) / 1000.0
        sig["force_t"] = tf
        sig["F_sum"] = butter_filt(fzL + fzR, 1000.0, hi=12.0)
        sig["FL_hf"] = band_energy(fzL, 1000.0, 8.0, 25.0)
        sig["FR_hf"] = band_energy(fzR, 1000.0, 8.0, 25.0)
        force_ok = True
    row["force_usable"] = int(force_ok)
    # ---- EMG (timing facts only)
    p = os.path.join(root, "Dataset", "EMG_Data", f"EMG_{subj}_{spd}.csv")
    row["has_emg"] = int(os.path.isfile(p))
    if row["has_emg"]:
        e = read_emg(p)
        row["emg_data_sha1"] = hashlib.sha1(np.ascontiguousarray(e).tobytes()).hexdigest()
        x = e[:, 0]
        dx = np.diff(x)
        row["emg_n"] = len(x)
        row["emg_dt_ms"] = float(np.median(dx) * 1000)
        row["emg_first_x_s"] = float(x[0])
        nz = np.flatnonzero(np.abs(e[:, 1:]).sum(1) > 0)
        row["emg_trailing_zero_rows"] = int(len(x) - 1 - nz[-1]) if len(nz) else len(x)
        row["emg_nonzero_end_s"] = float(x[nz[-1]]) if len(nz) else float("nan")
        row["emg_dur_s"] = float(x[-1] - x[0] + np.median(dx))
    # ---- durations relative to EEG
    if "eeg_dur_s" in row:
        for k in ("imu_dur_s", "force_dur_s", "emg_dur_s"):
            if k in row:
                row[k.replace("_dur_s", "_minus_eeg_dur_s")] = row[k] - row["eeg_dur_s"]
    # ---- lags
    def common(ta, tb):
        return min(ta[-1], tb[-1])

    if imu_ok and force_ok:
        T = common(sig["imu_t"], sig["force_t"])
        newton = xc_parts([(on_grid(sig["imu_t"], sig["P_z"], T), on_grid(sig["force_t"], sig["F_sum"], T))])
        side = xc_parts([(on_grid(sig["imu_t"], sig["FootL_hf"], T), on_grid(sig["force_t"], sig["FL_hf"], T)),
                         (on_grid(sig["imu_t"], sig["FootR_hf"], T), on_grid(sig["force_t"], sig["FR_hf"], T))])
        comb = (newton[0], (newton[1] + side[1]) / 2.0, newton[2], newton[3])
        put(row, "IF", estimate(comb, fine=newton))
        put(row, "IF_side", estimate(side))
    if eeg_ok and imu_ok:
        T = common(sig["eeg_t"], sig["imu_t"])
        put(row, "EI", estimate(xc_parts([(on_grid(sig["eeg_t"], sig["E"], T), on_grid(sig["imu_t"], sig["H_hf"], T))])))
        put(row, "EI_alt", estimate(xc_parts([(on_grid(sig["eeg_t"], sig["E_alt"], T), on_grid(sig["imu_t"], sig["H_abs"], T))])))
    if eeg_ok and force_ok:
        T = common(sig["eeg_t"], sig["force_t"])
        fh = sig["FL_hf"] + sig["FR_hf"]
        put(row, "EF", estimate(xc_parts([(on_grid(sig["eeg_t"], sig["E"], T), on_grid(sig["force_t"], fh, T))])))
    if all(row.get(f"{p}_measured") == 1 for p in PAIRS_PRIMARY):
        row["closure_ms"] = row["EF_lag_ms"] - row["EI_lag_ms"] - row["IF_lag_ms"]
    return row


def put(row, pair, est):
    for k, v in est.items():
        row[f"{pair}_{k}"] = int(v) if isinstance(v, bool) else v


# ------------------------------------------------------------------------------ subjects
EEG_PAIRS = {"EI", "EF", "EI_alt"}


def usable(r, pair, drop_eeg):
    """A record lag enters the gate when measured and, for an EEG pair, when the EEG data are not
    duplicated in another record (drop_eeg)."""
    return r.get(f"{pair}_measured") == 1 and not (pair in EEG_PAIRS and r["record"] in drop_eeg)


def subject_table(recs, rng, drop_eeg):
    rows = []
    subjects = sorted({r["subject"] for r in recs}, key=lambda s: int(s[1:]))
    by = {(r["subject"], r["speed"]): r for r in recs}
    pop_med = {}
    for pair in PAIRS_PRIMARY + PAIRS_SENS:
        for spd in SPEEDS:
            v = [r[f"{pair}_lag_ms"] for r in recs if r["speed"] == spd and usable(r, pair, drop_eeg)]
            pop_med[(pair, spd)] = float(np.median(v)) if v else float("nan")
    for s in subjects:
        for pair in PAIRS_PRIMARY + PAIRS_SENS:
            row = {"subject": s, "pair": pair}
            lags, ses, why = [], [], []
            for spd in SPEEDS:
                r = by.get((s, spd))
                if r is None or f"{pair}_lag_ms" not in r:
                    row[f"lag_{spd}"] = ""
                    why.append(f"{spd}:stream_missing")
                    continue
                row[f"lag_{spd}"] = round(r[f"{pair}_lag_ms"], 2)
                row[f"se_{spd}"] = round(r[f"{pair}_se_ms"], 2)
                if usable(r, pair, drop_eeg):
                    lags.append(r[f"{pair}_lag_ms"]); ses.append(r[f"{pair}_se_ms"])
                elif pair in EEG_PAIRS and r["record"] in drop_eeg:
                    why.append(f"{spd}:eeg_duplicated_in_another_record")
                else:
                    why.append(f"{spd}:not_measured")
            row["n_speeds_measured"] = len(lags)
            if len(lags) == 3:
                lags = np.array(lags); ses = np.array(ses)
                sp = float(lags.max() - lags.min())
                draws = rng.normal(lags[:, None], ses[:, None], size=(3, N_MC))
                sd = draws.max(0) - draws.min(0)
                noise = rng.normal(0.0, ses[:, None], size=(3, N_MC))
                sn = noise.max(0) - noise.min(0)
                row["spread_ms"] = round(sp, 2)
                row["spread_lo"] = round(float(np.percentile(sd, 2.5)), 2)
                row["spread_hi"] = round(float(np.percentile(sd, 97.5)), 2)
                row["p_noise_over_tol"] = round(float(np.mean(sn > TOL_MS)), 4)
                row["within_tol"] = int(sp <= TOL_MS)
                row["class"] = ("within" if row["spread_hi"] <= TOL_MS else
                                "outside" if row["spread_lo"] > TOL_MS else "undetermined")
                cen = np.array([row[f"lag_{spd}"] - pop_med[(pair, spd)] for spd in SPEEDS])
                row["spread_speed_centred_ms"] = round(float(cen.max() - cen.min()), 2)
                row["within_tol_speed_centred"] = int(row["spread_speed_centred_ms"] <= TOL_MS)
            else:
                row["class"] = "not_evaluable"
                row["reason"] = ";".join(why)
            rows.append(row)
    return rows, pop_med


def frac_ci(flags, rng):
    flags = np.asarray(flags, float)
    if len(flags) == 0:
        return float("nan"), float("nan"), float("nan")
    b = rng.choice(flags, size=(N_BOOT, len(flags)), replace=True).mean(1)
    return float(flags.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def read_manifests(ma):
    out = {}
    for fam, fn in (("eeg", "_canonical_eeg_manifest.csv"), ("imu", "_canonical_imu_manifest.csv"),
                    ("force", "_canonical_force_manifest.csv"), ("emg", "_canonical_emg_manifest.csv")):
        p = os.path.join(ma, "datasets", fn)
        if os.path.isfile(p):
            df = read_csv(p)
            df = df[df["source"] == SRC]
            out[fam] = set(zip(df["subject_ns"], df["task"]))
    return out


def doc_sync_text(root):
    """The synchronisation paragraph of the record description, through pdftotext."""
    p = os.path.join(root, "Project_Content.pdf")
    try:
        txt = subprocess.run(["pdftotext", "-layout", p, "-"], capture_output=True, timeout=120).stdout.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return f"(pdftotext failed: {e})"
    flat = re.sub(r"\s+", " ", txt)
    m = re.search(r"(Accurate synchronization across modalities.*?near-synchronous acquisition across all systems\.)", flat)
    return m.group(1) if m else "(paragraph not found)"


# ------------------------------------------------------------------------------ main
def main():
    a = get_args()
    root = os.path.join(a.moveatlas, REL)
    manifests = read_manifests(a.moveatlas)
    subjects = sorted({m.group(1) for f in os.listdir(os.path.join(root, "Dataset", "EEG_Data", "EEG.csv"))
                       if not f.startswith("._") for m in [re.match(r"(S\d+)_", f)] if m}, key=lambda s: int(s[1:]))
    if a.subjects:
        subjects = [s for s in subjects if s in a.subjects]
        print("DEBUG SUBSET:", subjects)
    wanted = {f"IMU_{s}_{v}" for s in subjects for v in SPEEDS}
    print(f"streaming IMU archive for {len(wanted)} candidate folders ...", flush=True)
    imu = stream_imu(os.path.join(root, "Dataset", "IMU_Data", "IMU.tar.gz"), wanted)
    print(f"IMU folders found: {len(imu)}", flush=True)
    recs = []
    for s in subjects:
        for v in SPEEDS:
            ids = [os.path.join(root, "Dataset", "EEG_Data", "EEG.csv", f"{s}_{v}_raw.csv"),
                   os.path.join(root, "Dataset", "FP_Data", f"FP_{s}_{v}.csv"),
                   os.path.join(root, "Dataset", "EMG_Data", f"EMG_{s}_{v}.csv")]
            ir = imu.get(f"IMU_{s}_{v}")
            if not any(os.path.isfile(p) for p in ids) and ir is None:
                continue
            r = process_record(root, s, v, ir, manifests)
            recs.append(r)
            print(r["record"], {p: (round(r[f"{p}_lag_ms"], 1), round(r[f"{p}_se_ms"], 1), r[f"{p}_measured"])
                                for p in PAIRS_PRIMARY if f"{p}_lag_ms" in r}, flush=True)
    # ---- duplicated content across records (hashes of the signal arrays, headers excluded)
    dups = {}
    for fam in ("eeg", "imu", "force", "emg"):
        groups = {}
        for r in recs:
            h = r.get(f"{fam}_data_sha1")
            if h:
                groups.setdefault(h, []).append(r["record"])
        dups[fam] = [g for g in groups.values() if len(g) > 1]
        for g in dups[fam]:
            for rid in g:
                next(r for r in recs if r["record"] == rid)[f"{fam}_same_data_as"] = ";".join(x for x in g if x != rid)
    drop_eeg = {rid for g in dups["eeg"] for rid in g}
    rng = np.random.default_rng(SEED)
    subj_rows, pop_med = subject_table(recs, rng, drop_eeg)
    subj_rows_rel, _ = subject_table(recs, rng, set())

    # ---- records.csv
    lead = ["record", "subject", "speed", "task"]
    keys = lead + sorted({k for r in recs for k in r} - set(lead))
    with open(os.path.join(a.out, "records.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in recs:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    lead = ["subject", "pair", "lag_0.5", "lag_0.75", "lag_1", "se_0.5", "se_0.75", "se_1", "n_speeds_measured",
            "spread_ms", "spread_lo", "spread_hi", "within_tol", "class", "p_noise_over_tol",
            "spread_speed_centred_ms", "within_tol_speed_centred", "reason"]
    with open(os.path.join(a.out, "subjects.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=lead, lineterminator="\n")
        w.writeheader()
        w.writerows(subj_rows)

    # ---- summary numbers
    S = {"moveatlas_rel": REL, "subset": a.subjects or "all", "tolerance_ms": TOL_MS, "pass_fraction": PASS_FRAC,
         "se_max_ms": SE_MAX_MS, "k_blocks": K_BLOCKS, "wide_ms": WIDE_MS, "basin_ms": BASIN_MS, "seed": SEED,
         "pair_definitions": PAIR_TEXT}
    S["documentation_sync_paragraph"] = doc_sync_text(root)
    S["n_records"] = len(recs)
    S["n_subjects_any_file"] = len({r["subject"] for r in recs})
    for fam in ("eeg", "imu", "force", "emg"):
        S[f"n_records_has_{fam}"] = sum(r.get(f"has_{fam}", 0) for r in recs)
        S[f"n_records_in_manifest_{fam}"] = sum(r.get(f"in_manifest_{fam}", 0) for r in recs)
    for fam in ("eeg", "imu", "force"):
        S[f"n_records_{fam}_usable"] = sum(r.get(f"{fam}_usable", 0) for r in recs)
    S["n_records_manifest_usable_mismatch"] = {fam: [r["record"] for r in recs if r.get(f"in_manifest_{fam}", 0) != r.get(f"{fam}_usable", r.get(f"has_{fam}", 0))]
                                               for fam in ("eeg", "imu", "force")}
    S["subjects_eeg_and_imu_usable"] = len({r["subject"] for r in recs if r.get("eeg_usable") and r.get("imu_usable")})
    S["subjects_eeg_imu_all_3_speeds"] = sum(1 for s in {r["subject"] for r in recs}
                                             if sum(1 for r in recs if r["subject"] == s and r.get("eeg_usable") and r.get("imu_usable")) == 3)
    S["subjects_eeg_imu_force_all_3_speeds"] = sum(1 for s in {r["subject"] for r in recs}
                                                   if sum(1 for r in recs if r["subject"] == s and r.get("eeg_usable") and r.get("imu_usable") and r.get("force_usable")) == 3)

    def vals(key, cond=lambda r: True):
        return [r[key] for r in recs if key in r and cond(r) and r[key] == r[key]]

    def uniq(key):
        v = vals(key)
        return {str(k): int(c) for k, c in zip(*np.unique(np.array([str(x) for x in v]), return_counts=True))} if v else {}

    def rng_(key):
        v = vals(key)
        return [float(np.min(v)), float(np.median(v)), float(np.max(v))] if v else []

    S["timing"] = {
        "eeg_n_samples": uniq("eeg_n"), "eeg_fs_header": uniq("eeg_fs_header"), "eeg_filter_delay_ms": uniq("eeg_filter_delay_ms"),
        "eeg_mains_hz": uniq("eeg_mains_hz"), "eeg_trigger_source": uniq("eeg_trigger_source"), "eeg_logger": uniq("eeg_logger"),
        "eeg_trigger_values": uniq("eeg_trigger_values"),
        "eeg_records_trigger_changes": {r["record"]: [r["eeg_trigger_changes"], r.get("eeg_trigger_first_change_sample"),
                                                      r.get("eeg_trigger_first_transition")]
                                        for r in recs if r.get("eeg_trigger_changes", 0) > 0},
        "eeg_trigger_constant_records": sum(1 for r in recs if r.get("eeg_trigger_changes", -1) == 0),
        "eeg_event_nonzero_records": sum(1 for r in recs if r.get("eeg_event_nonzero", 0) > 0),
        "eeg_time_steps_off_records": [r["record"] for r in recs if r.get("eeg_time_steps_off", 0) > 0],
        "eeg_t0_s": rng_("eeg_t0_s"), "eeg_dur_s": rng_("eeg_dur_s"),
        "eeg_timeoffset_t0": uniq("eeg_timeoffset_t0"), "eeg_timeoffset_rate_hz": rng_("eeg_timeoffset_rate_hz"),
        "eeg_timeoffset_nonmonotonic_records": sum(1 for r in recs if r.get("eeg_timeoffset_nonmonotonic", 0) > 0),
        "eeg_adcseq_gap_records": [r["record"] for r in recs if r.get("eeg_adcseq_gaps", 0) > 0],
        "eeg_adcseq_max": uniq("eeg_adcseq_max"),
        "eeg_minus_imu_start_hours": uniq("eeg_minus_imu_start_hours"),
        "eeg_minus_imu_start_resid_s": rng_("eeg_minus_imu_start_resid_s"),
        "eeg_minus_imu_start_resid_s_q10_q90": [float(np.percentile(vals("eeg_minus_imu_start_resid_s"), q)) for q in (10, 90)]
        if vals("eeg_minus_imu_start_resid_s") else [],
        "eeg_minus_imu_start_n": len(vals("eeg_minus_imu_start_resid_s")),
        "eeg_minus_imu_start_within_5s_records": [r["record"] for r in recs if abs(r.get("eeg_minus_imu_start_resid_s", 1e9)) <= 5],
        "imu_frame_rate_header": uniq("imu_frame_rate_header"), "imu_processing_quality": uniq("imu_processing_quality"),
        "imu_frame0": uniq("imu_frame0"), "imu_frame_gap_records": sum(1 for r in recs if r.get("imu_frame_gaps", 0) > 0),
        "imu_markers_rows": uniq("imu_markers_rows"), "imu_empty_export_records": [r["record"] for r in recs if r.get("imu_empty_export") == 1],
        "imu_dur_s": rng_("imu_dur_s"), "imu_minus_eeg_dur_s": rng_("imu_minus_eeg_dur_s"),
        "force_dur_s": rng_("force_dur_s"), "force_minus_eeg_dur_s": rng_("force_minus_eeg_dur_s"),
        "force_timestamp_all_zero": uniq("force_timestamp_all_zero"), "force_sync_all_zero": uniq("force_sync_all_zero"),
        "force_aux_all_zero": uniq("force_aux_all_zero"), "force_no_header_records": [r["record"] for r in recs if r.get("force_header_line") == 0],
        "emg_dt_ms": rng_("emg_dt_ms"), "emg_first_x_s": rng_("emg_first_x_s"), "emg_trailing_zero_rows": rng_("emg_trailing_zero_rows"),
        "emg_dur_s": rng_("emg_dur_s"), "emg_minus_eeg_dur_s": rng_("emg_minus_eeg_dur_s"),
        "emg_rate_hz_from_median_step": (1000.0 / float(np.median(vals("emg_dt_ms")))) if vals("emg_dt_ms") else float("nan"),
        "edf_records": sum(r.get("has_eeg_edf", 0) for r in recs), "edf_dur_s": rng_("edf_dur_s"),
        "edf_annotation_records": {r["record"]: [r["edf_annotations_n"], r["edf_annotation_texts"]]
                                   for r in recs if r.get("edf_annotations_n", 0) > 0},
        "edf_trigger_change_records": [r["record"] for r in recs if r.get("edf_trigger_changes", 0) > 0],
        "edf_trigger_change_records_equal_csv": sorted(r["record"] for r in recs if r.get("edf_trigger_changes", 0) > 0)
        == sorted(r["record"] for r in recs if r.get("eeg_trigger_changes", 0) > 0),
    }
    S["duplicates"] = {fam: dups[fam] for fam in dups}
    S["duplicates_eeg_evidence"] = [
        {r["record"]: {"has_imu": r.get("imu_usable", 0), "has_force": r.get("force_usable", 0),
                       "EI_measured": r.get("EI_measured", ""), "EI_r": r.get("EI_r", ""),
                       "EF_measured": r.get("EF_measured", ""), "EF_r": r.get("EF_r", ""),
                       "eeg_minus_imu_start_hours": r.get("eeg_minus_imu_start_hours", "")}
         for r in recs if r["record"] in g} for g in dups["eeg"]]
    S["records_eeg_dropped_as_duplicated"] = sorted(drop_eeg)
    hours_mode = pd.Series(vals("eeg_minus_imu_start_hours")).mode()
    S["records_header_hours_not_modal"] = [r["record"] for r in recs if "eeg_minus_imu_start_hours" in r and len(hours_mode)
                                           and r["eeg_minus_imu_start_hours"] != hours_mode.iloc[0]]
    S["records_header_hours_not_modal_that_are_eeg_duplicates"] = [x for x in S["records_header_hours_not_modal"] if x in drop_eeg]
    # gate statistics: primary (duplicated EEG dropped) and as released (sensitivity)
    G = gate_stats(recs, subj_rows, drop_eeg, rng, S)
    S.update(G)
    G_rel = gate_stats(recs, subj_rows_rel, set(), rng, S)
    S["as_released"] = {"pairs": {p: {k: G_rel["pairs"][p][k] for k in ("subjects_evaluable", "subjects_within", "fraction_within",
                                                                        "fraction_within_ci95", "pass", "outcome")} for p in G_rel["pairs"]},
                        "all_primary_pairs": G_rel["all_primary_pairs"], "g15_verdict": G_rel["g15_verdict"]}
    S["sensitivity_range_40ms"] = range_sensitivity(subj_rows, SEED + 1)
    S["subjects_with_eeg_imu_evaluable_of_eeg_imu_subjects"] = [
        sum(1 for x in subj_rows if x["pair"] == "EI" and x["class"] != "not_evaluable"), S["subjects_eeg_and_imu_usable"]]
    # which stream moves: within-subject centred lags, records with all three primary lags usable
    tri = [r for r in recs if all(usable(r, p, drop_eeg) for p in PAIRS_PRIMARY)]
    dfp = pd.DataFrame([{"subject": r["subject"], **{p: r[f"{p}_lag_ms"] for p in PAIRS_PRIMARY}, "closure": r["closure_ms"]} for r in tri])
    if len(dfp):
        cnt = dfp.groupby("subject")["EI"].transform("count")
        dfp = dfp[cnt >= 2].copy()
        for p in PAIRS_PRIMARY + ["closure"]:
            dfp[p + "_c"] = dfp[p] - dfp.groupby("subject")[p].transform("mean")
        cm = dfp[[p + "_c" for p in PAIRS_PRIMARY]].corr()
        # pooled within-subject covariance: one mean is removed per subject, so the degrees of freedom
        # are records minus subjects (the plain N - 1 of DataFrame.cov would bias every SD low)
        n_s = dfp.groupby("subject")["EI"].transform("count").values.astype(float)
        dof = int(len(dfp) - dfp["subject"].nunique())
        Cm = dfp[[p + "_c" for p in PAIRS_PRIMARY]]
        cv = (Cm.T @ Cm) / dof
        # EI = oI - oE, IF = oF - oI, EF = oF - oE for independent start offsets oE, oI, oF:
        # var(oE) = cov(EI, EF), var(oI) = -cov(EI, IF), var(oF) = cov(IF, EF)
        imp = {"EEG": float(cv.loc["EI_c", "EF_c"]), "IMU": float(-cv.loc["EI_c", "IF_c"]), "force": float(cv.loc["IF_c", "EF_c"])}
        # a residual from a mean over n records has variance (n - 1) / n of the record variance
        scale = np.sqrt(n_s / (n_s - 1.0))
        S["within_subject"] = {
            "records": int(len(dfp)), "subjects": int(dfp["subject"].nunique()), "dof_records_minus_subjects": dof,
            "corr_IF_EF": float(cm.loc["IF_c", "EF_c"]), "corr_EI_IF": float(cm.loc["EI_c", "IF_c"]), "corr_EI_EF": float(cm.loc["EI_c", "EF_c"]),
            "robust_sd_ms": {p: float(1.4826 * np.median(np.abs(dfp[p + "_c"].values * scale))) for p in PAIRS_PRIMARY + ["closure"]},
            "sd_ms": {p: float(np.sqrt(np.sum(dfp[p + "_c"].values ** 2) / dof)) for p in PAIRS_PRIMARY + ["closure"]},
            "implied_start_offset_variance_ms2": imp,
            "implied_start_offset_sd_ms": {k: (float(np.sqrt(v)) if v > 0 else 0.0) for k, v in imp.items()},
        }
        S["closure_ms_by_speed"] = {spd: [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)), len(v)]
                                    for spd in SPEEDS for v in [[r["closure_ms"] for r in tri if r["speed"] == spd]] if v}
    # lags by MVN processing mode (Live or HD)
    S["lag_ms_by_imu_processing"] = {}
    for q in sorted({r.get("imu_processing_quality") for r in recs if r.get("imu_processing_quality")}):
        S["lag_ms_by_imu_processing"][q] = {
            p: [float(np.median(v)), len(v), sorted({r["subject"] for r in recs if r.get("imu_processing_quality") == q})]
            for p in ("EI", "IF") for v in [[r[f"{p}_lag_ms"] for r in recs if r.get("imu_processing_quality") == q and usable(r, p, drop_eeg)]] if v}
    S["population_median_lag_ms"] = {f"{p}|{spd}": v for (p, spd), v in pop_med.items()}
    # the 0.5 m/s record against the two faster ones, per evaluable subject:
    # excess = lag(0.5) - mean(lag(0.75), lag(1))
    S["speed_0.5_excess_ms"] = {}
    for p in PAIRS_PRIMARY:
        ex = {x["subject"]: x["lag_0.5"] - (x["lag_0.75"] + x["lag_1"]) / 2.0
              for x in subj_rows if x["pair"] == p and x["class"] != "not_evaluable"}
        v = np.array(list(ex.values()), float)
        big = sorted([s for s, e in ex.items() if e > 40.0], key=lambda s: int(s[1:]))
        S["speed_0.5_excess_ms"][p] = {
            "subjects": int(len(v)),
            "quantiles_p10_p25_p50_p75_p90": [float(np.percentile(v, q)) for q in (10, 25, 50, 75, 90)] if len(v) else [],
            "subjects_over_40ms": len(big),
            "excess_over_40ms_min_median_max": [float(np.min([ex[s] for s in big])), float(np.median([ex[s] for s in big])),
                                                float(np.max([ex[s] for s in big]))] if big else [],
            "subjects_over_40ms_list": big}
    both = set(S["speed_0.5_excess_ms"]["IF"]["subjects_over_40ms_list"]) & set(S["speed_0.5_excess_ms"]["EF"]["subjects_over_40ms_list"])
    ef_ev = {x["subject"] for x in subj_rows if x["pair"] == "EF" and x["class"] != "not_evaluable"}
    S["speed_0.5_excess_ms"]["IF_over_40ms_also_EF_evaluable"] = len(set(S["speed_0.5_excess_ms"]["IF"]["subjects_over_40ms_list"]) & ef_ev)
    S["speed_0.5_excess_ms"]["IF_and_EF_over_40ms"] = len(both)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(S, f, indent=1, default=float)
    write_summary_md(a.out, S)
    plot(a.out, subj_rows)
    write_provenance(a, {"subset": a.subjects or "all", "n_records": len(recs), "seed": SEED})
    print("verdict:", S["g15_verdict"], {p: round(S["pairs"][p]["fraction_within"], 3) for p in S["pairs"]},
          "as released:", S["as_released"]["g15_verdict"])


def gate_stats(recs, subj_rows, drop, rng, S):
    P = {}
    for pair in PAIRS_PRIMARY + PAIRS_SENS:
        have = [r for r in recs if f"{pair}_lag_ms" in r]
        meas = [r for r in have if usable(r, pair, drop)]
        srows = [x for x in subj_rows if x["pair"] == pair]
        ev = [x for x in srows if x["class"] != "not_evaluable"]
        f, lo, hi = frac_ci([x["within_tol"] for x in ev], rng)
        fc, _, _ = frac_ci([x["within_tol_speed_centred"] for x in ev], rng)
        # subjects where both streams of the pair exist at all 3 speeds (EEG duplicates dropped)
        need = {"EI": ("eeg", "imu"), "EI_alt": ("eeg", "imu"), "EF": ("eeg", "force"), "IF": ("imu", "force"), "IF_side": ("imu", "force")}[pair]
        present = [s for s in {r["subject"] for r in recs}
                   if sum(1 for r in recs if r["subject"] == s and all(r.get(f"{fam}_usable") for fam in need)
                          and not (pair in EEG_PAIRS and r["record"] in drop)) == 3]
        n_out = sum(1 for x in ev if x["class"] == "outside")
        d = {"definition": PAIR_TEXT[pair], "records_estimated": len(have), "records_measured": sum(1 for r in have if r[f"{pair}_measured"] == 1),
             "records_used": len(meas),
             "records_not_identified": sum(1 for r in have if r[f"{pair}_basin_stable"] < K_BLOCKS),
             "records_se_over_max": sum(1 for r in have if r[f"{pair}_se_ms"] > SE_MAX_MS),
             "records_at_edge": sum(1 for r in have if r[f"{pair}_at_edge"] == 1 or r[f"{pair}_wide_edge"] == 1),
             "subjects_evaluable": len(ev), "subjects_within": int(sum(x["within_tol"] for x in ev)),
             "fraction_within": f, "fraction_within_ci95": [lo, hi],
             "subjects_streams_present_3_speeds": len(present),
             "fraction_within_upper_bound": ((len(present) - n_out) / len(present)) if present else float("nan"),
             "fraction_within_counting_unevaluable_as_fail": (sum(x["within_tol"] for x in ev) / len(present)) if present else float("nan"),
             "class_counts": {c: sum(1 for x in srows if x["class"] == c) for c in ("within", "outside", "undetermined", "not_evaluable")},
             "noise_only_expected_fraction_over_tol": float(np.mean([x["p_noise_over_tol"] for x in ev])) if ev else float("nan"),
             "spread_ms_quantiles": [float(np.percentile([x["spread_ms"] for x in ev], q)) for q in (10, 25, 50, 75, 90)] if ev else [],
             "fraction_within_speed_centred_diagnostic": fc,
             "pass": bool(f >= PASS_FRAC) if f == f else False,
             "outcome": outcome(f, hi)}
        for spd in SPEEDS:
            v = [r[f"{pair}_lag_ms"] for r in meas if r["speed"] == spd]
            d[f"lag_ms_speed_{spd}"] = [float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)), len(v)] if v else []
        d["se_ms_quantiles"] = [float(np.percentile([r[f"{pair}_se_ms"] for r in meas], q)) for q in (50, 90)] if meas else []
        d["r_quantiles"] = [float(np.percentile([r[f"{pair}_r"] for r in meas], q)) for q in (10, 50, 90)] if meas else []
        dr = np.array([r[f"{pair}_drift_ms_per_min"] for r in meas]); ds = np.array([r[f"{pair}_drift_se"] for r in meas])
        if len(dr):
            d["drift_ms_per_min_quantiles"] = [float(np.percentile(dr, q)) for q in (10, 50, 90)]
            d["drift_ppm_median"] = float(np.median(dr) / 60000.0 * 1e6)
            d["drift_records_abs_over_2se"] = int(np.sum(np.abs(dr) > 2 * ds))
            d["drift_records_abs_over_5ms_per_min"] = int(np.sum(np.abs(dr) > 5.0))
        d["records_abs_lag_over_1s"] = [r["record"] for r in meas if abs(r[f"{pair}_lag_ms"]) > 1000]
        P[pair] = d
    by_s = {}
    for x in subj_rows:
        if x["pair"] in PAIRS_PRIMARY:
            by_s.setdefault(x["subject"], {})[x["pair"]] = x
    ev_all = [s for s, d in by_s.items() if all(d[p]["class"] != "not_evaluable" for p in PAIRS_PRIMARY)]
    within_all = [int(all(by_s[s][p]["within_tol"] for p in PAIRS_PRIMARY)) for s in ev_all]
    f, lo, hi = frac_ci(within_all, rng)
    out = {"pairs": P, "all_primary_pairs": {"subjects_evaluable": len(ev_all), "subjects_within": int(sum(within_all)),
                                             "fraction_within": f, "fraction_within_ci95": [lo, hi],
                                             "outcome": outcome(f, hi)}}
    ok_pairs = [p for p in PAIRS_PRIMARY if P[p]["pass"]]
    out["g15_pass_pairs"] = ok_pairs
    out["g15_verdict"] = ("PASS" if len(ok_pairs) == len(PAIRS_PRIMARY) and f >= PASS_FRAC else "FAIL: PRE only")
    out["g15_absent_pairs"] = [p for p in PAIRS_PRIMARY if P[p]["outcome"] == "absent"]
    return out


def range_sensitivity(subj_rows, seed):
    """Second reading of 'constant within 20 ms': every lag within +-20 ms of one value per subject, so a
    range (max - min) of at most 40 ms. Sensitivity only, on the same evaluable subjects; the gate reads
    a range of at most 20 ms. Own random generator, so the gate numbers do not move."""
    rng = np.random.default_rng(seed)
    lim = 2 * TOL_MS
    out = {"max_range_ms": lim, "seed": seed, "pairs": {}}
    for pair in PAIRS_PRIMARY + PAIRS_SENS:
        ev = [x for x in subj_rows if x["pair"] == pair and x["class"] != "not_evaluable"]
        flags = [int(x["spread_ms"] <= lim) for x in ev]
        f, lo, hi = frac_ci(flags, rng)
        out["pairs"][pair] = {"subjects_evaluable": len(ev), "subjects_within": int(sum(flags)), "fraction_within": f,
                              "fraction_within_ci95": [lo, hi], "pass": bool(f >= PASS_FRAC) if f == f else False,
                              "outcome": outcome(f, hi)}
    by_s = {}
    for x in subj_rows:
        if x["pair"] in PAIRS_PRIMARY and x["class"] != "not_evaluable":
            by_s.setdefault(x["subject"], {})[x["pair"]] = x
    ev_all = [s for s, d in by_s.items() if len(d) == len(PAIRS_PRIMARY)]
    flags = [int(all(by_s[s][p]["spread_ms"] <= lim for p in PAIRS_PRIMARY)) for s in ev_all]
    f, lo, hi = frac_ci(flags, rng)
    out["all_primary_pairs"] = {"subjects_evaluable": len(ev_all), "subjects_within": int(sum(flags)), "fraction_within": f,
                                "fraction_within_ci95": [lo, hi], "outcome": outcome(f, hi)}
    ok = [p for p in PAIRS_PRIMARY if out["pairs"][p]["pass"]]
    out["pass_pairs"] = ok
    out["g15_verdict"] = "PASS" if len(ok) == len(PAIRS_PRIMARY) and f >= PASS_FRAC else "FAIL: PRE only"
    return out


def outcome(f, hi):
    """Three-outcome convention of the addendum: pass on the point estimate, absent when the subject
    bootstrap interval lies entirely below the pass fraction, undetermined otherwise."""
    if f != f:
        return "undetermined"
    if f >= PASS_FRAC:
        return "pass"
    return "absent" if hi < PASS_FRAC else "undetermined"


def fmt(x, nd=1):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def fmt_d(d, nd=1):
    return "; ".join(f"{k}: {fmt(v, nd)}" for k, v in d.items())


def write_summary_md(out, S):
    P = S["pairs"]
    T = S["timing"]
    L = []
    L.append("# physionet_multimodal_gait: synchronisation of EEG, IMU and force (G15)")
    L.append("")
    L.append("Written by scripts/probe_sync_physionet_gait.py; every number below is in summary.json, records.csv "
             "or subjects.csv beside this file. Descriptive only: no coupling statistic.")
    if S["subset"] != "all":
        L.append("")
        L.append(f"DEBUG SUBSET, not a full pass: {S['subset']}.")
    L.append("")
    L.append(f"## Verdict: {S['g15_verdict']}")
    L.append("")
    L.append(f"G15 asks for lags constant within {fmt(S['tolerance_ms'], 0)} ms across the 3 speeds in at least "
             f"{fmt(100 * S['pass_fraction'], 0)} percent of subjects, per pair, from label-free shared events.")
    L.append("")
    L.append("| pair | subjects evaluable | within 20 ms | fraction (95% subject bootstrap) | outside / undetermined by record precision | expected fraction over 20 ms from noise alone | pass | outcome |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in PAIRS_PRIMARY + PAIRS_SENS:
        d = P[p]
        cc = d["class_counts"]
        L.append(f"| {p} | {d['subjects_evaluable']} | {d['subjects_within']} | {fmt(d['fraction_within'], 3)} "
                 f"{fmt(d['fraction_within_ci95'], 3)} | {cc['outside']} / {cc['undetermined']} | "
                 f"{fmt(d['noise_only_expected_fraction_over_tol'], 3)} | {'yes' if d['pass'] else 'no'} | {d['outcome']} |")
    ap = S["all_primary_pairs"]
    L.append("")
    L.append("Outcome follows the addendum's three-outcome convention: pass on the point estimate, absent when the subject "
             f"bootstrap interval lies entirely below {fmt(S['pass_fraction'], 2)}, undetermined otherwise; only absent demotes. "
             f"Primary pairs absent: {S['g15_absent_pairs']}.")
    L.append("")
    L.append(f"All three primary pairs within 20 ms: {ap['subjects_within']} of {ap['subjects_evaluable']} evaluable subjects "
             f"({fmt(ap['fraction_within'], 3)}, interval {fmt(ap['fraction_within_ci95'], 3)}, {ap['outcome']}). "
             f"IF_side and EI_alt are sensitivity estimators, not part of the verdict. A subject is evaluable for a pair when "
             f"its lag is measured at the 3 speeds; records whose EEG data are duplicated in another record (section 4) are "
             f"left out of the EEG pairs.")
    L.append("")
    L.append("Robustness of the verdict. Upper bound: every subject with both streams at the 3 speeds counted as within 20 ms "
             "unless its spread interval lies entirely above 20 ms. As released: the same computation keeping the duplicated EEG records.")
    L.append("")
    L.append("| pair | subjects with both streams at 3 speeds | upper-bound fraction within | fraction within, as released (evaluable) | outcome as released |")
    L.append("|---|---|---|---|---|")
    AR = S["as_released"]["pairs"]
    for p in PAIRS_PRIMARY + PAIRS_SENS:
        d = P[p]
        L.append(f"| {p} | {d['subjects_streams_present_3_speeds']} | {fmt(d['fraction_within_upper_bound'], 3)} | "
                 f"{fmt(AR[p]['fraction_within'], 3)} {fmt(AR[p]['fraction_within_ci95'], 3)} ({AR[p]['subjects_evaluable']}) | "
                 f"{AR[p]['outcome']} |")
    L.append("")
    L.append(f"As released, all three primary pairs: {S['as_released']['all_primary_pairs']['subjects_within']} of "
             f"{S['as_released']['all_primary_pairs']['subjects_evaluable']} ({fmt(S['as_released']['all_primary_pairs']['fraction_within'], 3)}, "
             f"interval {fmt(S['as_released']['all_primary_pairs']['fraction_within_ci95'], 3)}, {S['as_released']['all_primary_pairs']['outcome']}); "
             f"verdict as released: {S['as_released']['g15_verdict']}.")
    L.append("")
    RS = S["sensitivity_range_40ms"]
    L.append("Reading of the tolerance. The gate reads 'constant within 20 ms' as a range (max - min) of at most 20 ms across "
             "the 3 speeds. Read instead as every lag within +-20 ms of one value per subject (a range of at most "
             f"{fmt(RS['max_range_ms'], 0)} ms), on the same evaluable subjects: "
             + "; ".join(f"{p} {RS['pairs'][p]['subjects_within']} of {RS['pairs'][p]['subjects_evaluable']} "
                         f"({fmt(RS['pairs'][p]['fraction_within'], 3)}, interval {fmt(RS['pairs'][p]['fraction_within_ci95'], 3)}, "
                         f"{RS['pairs'][p]['outcome']})" for p in PAIRS_PRIMARY + PAIRS_SENS)
             + f"; all three primary pairs {RS['all_primary_pairs']['subjects_within']} of {RS['all_primary_pairs']['subjects_evaluable']} "
             f"({fmt(RS['all_primary_pairs']['fraction_within'], 3)}, interval {fmt(RS['all_primary_pairs']['fraction_within_ci95'], 3)}, "
             f"{RS['all_primary_pairs']['outcome']}). Pairs passing under this reading: {RS['pass_pairs']}; verdict under this "
             f"reading: {RS['g15_verdict']}.")
    L.append("")
    L.append("Pair definitions:")
    for p in PAIRS_PRIMARY + PAIRS_SENS:
        L.append(f"- {p}: {P[p]['definition']}.")
    L.append("")
    L.append("## 1. How the streams are timestamped (files and documentation)")
    L.append("")
    L.append("Documentation (record description, Project_Content.pdf, synchronisation paragraph, extracted into "
             "summary.json): a MATLAB trigger protocol sent a digital pulse through a trigger interface box to the EEG, "
             "EMG and IMU systems over a wireless hub, with fixed wireless delays compensated; the Bertec treadmill had no "
             "hardware synchronisation, so its recording was started by hand, prompted by the same script, 'near-synchronous' "
             "in the authors' word.")
    L.append("")
    L.append("Files (counts over all records):")
    L.append(f"- EEG: sample counts {T['eeg_n_samples']}; header rate {T['eeg_fs_header']}; Time column starts at "
             f"{fmt(T['eeg_t0_s'], 4)} s and spans {fmt(T['eeg_dur_s'], 2)} s (min, median, max); records with a Time step off by more than half a sample: "
             f"{T['eeg_time_steps_off_records']}. Trigger source {T['eeg_trigger_source']}. Trigger column values per record "
             f"{T['eeg_trigger_values']}; records where the trigger changes inside the file: {T['eeg_records_trigger_changes']}. "
             f"Event column non-zero in {T['eeg_event_nonzero_records']} records. Header filter delay (ms) {T['eeg_filter_delay_ms']}; "
             f"mains {T['eeg_mains_hz']}. Time_Offset starts at {T['eeg_timeoffset_t0']} and runs at {fmt(T['eeg_timeoffset_rate_hz'], 2)} ticks per s "
             f"(min, median, max); ADC_Sequence gaps (dropped packets) in {T['eeg_adcseq_gap_records']}.")
    L.append(f"- EEG EDF+ export of the same records ({T['edf_records']} files): duration {fmt(T['edf_dur_s'], 2)} s (min, median, max); "
             f"records whose annotation channel carries any text (record: count, texts): {T['edf_annotation_records']}; records whose "
             f"EDF Trigger channel changes: {T['edf_trigger_change_records']} (the same records as in the CSV: "
             f"{T['edf_trigger_change_records_equal_csv']}). The EDF adds no event beyond the CSV Trigger column.")
    L.append(f"- IMU (MVN export): frame rate header {T['imu_frame_rate_header']}, processing {T['imu_processing_quality']}; first "
             f"frame index {T['imu_frame0']}; records with a frame gap: {T['imu_frame_gap_records']}; Markers.csv data rows "
             f"{T['imu_markers_rows']}; empty free-acceleration exports: {T['imu_empty_export_records']}. Duration "
             f"{fmt(T['imu_dur_s'], 2)} s, minus the EEG duration {fmt(T['imu_minus_eeg_dur_s'], 2)} s.")
    L.append(f"- Start-time headers ({T['eeg_minus_imu_start_n']} records with both): EEG header start minus MVN 'Recorded Date UTC', "
             f"whole hours {T['eeg_minus_imu_start_hours']}; residual seconds min, median, max {fmt(T['eeg_minus_imu_start_resid_s'], 0)}, "
             f"p10 and p90 {fmt(T['eeg_minus_imu_start_resid_s_q10_q90'], 0)}; records within 5 s: "
             f"{T['eeg_minus_imu_start_within_5s_records']} (1 s resolution; the dates themselves are not written anywhere).")
    L.append(f"- Force: Timestamp columns all zero {T['force_timestamp_all_zero']}, Sync all zero {T['force_sync_all_zero']}, "
             f"Aux all zero {T['force_aux_all_zero']} (key 1 = true); files without a header line {T['force_no_header_records']}. "
             f"Duration {fmt(T['force_dur_s'], 2)} s, minus EEG {fmt(T['force_minus_eeg_dur_s'], 2)} s. The rate is only declared (1000 Hz, record page).")
    L.append(f"- EMG: X step {fmt(T['emg_dt_ms'], 4)} ms ({fmt(T['emg_rate_hz_from_median_step'], 2)} Hz); first X value {fmt(T['emg_first_x_s'], 4)} s; trailing all-zero rows "
             f"{fmt(T['emg_trailing_zero_rows'], 0)}; duration {fmt(T['emg_dur_s'], 2)} s, minus EEG {fmt(T['emg_minus_eeg_dur_s'], 2)} s. "
             f"No label-free shared event with the EMG envelope is used here (16 Hz MAV; G15 names EEG, IMU and force).")
    L.append("")
    n_const = T["eeg_trigger_constant_records"]
    L.append(f"Reading: the EEG trigger column is constant in {n_const} of {S['n_records_has_eeg']} EEG records; where it changes "
             "(listed above: record, number of changes, first change sample, transition) there is no counterpart in the other "
             "streams, because the MVN Markers files are empty, the force Timestamp and Sync columns are zero and the EMG file "
             "has only its X column. No shared clock or trigger edge is written in the files. Each file starts at its own zero "
             "and the streams have different lengths, so any alignment is by file start plus whatever the acquisition did; "
             "the lags below measure it.")
    L.append("")
    L.append("## 2. Lags from shared events")
    L.append("")
    L.append("Lag convention: lag_A_B > 0 means the event appears later on the A file clock than on the B file clock. "
             f"Search over +-{S['wide_ms'] // 1000} s for the alignment basin, refinement within +-{S['basin_ms']} ms, "
             f"jackknife over {S['k_blocks']} blocks; a record is measured when its basin is stable in all leave-one-block-out "
             f"replicates, not at a window edge, and its SE is at most {fmt(S['se_max_ms'], 0)} ms.")
    L.append("")
    L.append("| pair | records estimated | measured | used (duplicated EEG left out) | basin not stable | SE over max | at edge | SE median, p90 (ms) | r p10, p50, p90 | lag 0.5 m/s: q25, median, q75, n | lag 0.75 | lag 1.0 | drift ms/min p10, p50, p90 | drift beyond 2 SE | median drift (ppm) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PAIRS_PRIMARY + PAIRS_SENS:
        d = P[p]
        L.append(f"| {p} | {d['records_estimated']} | {d['records_measured']} | {d['records_used']} | {d['records_not_identified']} | "
                 f"{d['records_se_over_max']} | {d['records_at_edge']} | {fmt(d.get('se_ms_quantiles', []), 2)} | "
                 f"{fmt(d.get('r_quantiles', []), 2)} | {fmt(d.get('lag_ms_speed_0.5', []), 1)} | {fmt(d.get('lag_ms_speed_0.75', []), 1)} | "
                 f"{fmt(d.get('lag_ms_speed_1', []), 1)} | {fmt(d.get('drift_ms_per_min_quantiles', []), 2)} | {d.get('drift_records_abs_over_2se', '')} | "
                 f"{fmt(d.get('drift_ppm_median', float('nan')), 0)} |")
    L.append("")
    for p in PAIRS_PRIMARY:
        if P[p].get("records_abs_lag_over_1s"):
            L.append(f"- {p}: records with a used lag beyond 1 s: {P[p]['records_abs_lag_over_1s']}.")
    L.append("")
    L.append("Drift is the slope of the 6 block lags against time; a positive IF drift means the IMU file clock runs ahead "
             "of the force file clock as the record goes on.")
    L.append("")
    if "within_subject" in S:
        W = S["within_subject"]
        L.append("Which stream moves. On the records where the three primary lags are used "
                 f"({W['records']} records, {W['subjects']} subjects with at least 2 such records), lags centred on each subject's mean: "
                 f"corr(IF, EF) {fmt(W['corr_IF_EF'], 2)}, corr(EI, IF) {fmt(W['corr_EI_IF'], 2)}, corr(EI, EF) {fmt(W['corr_EI_EF'], 2)}. "
                 f"Variances and covariances are pooled within subject with {W['dof_records_minus_subjects']} degrees of freedom "
                 "(records minus subjects); the robust SD rescales each residual by the square root of n / (n - 1) for a subject with n records. "
                 f"Within-subject robust SD (1.4826 MAD, ms): {', '.join(f'{k} {fmt(v, 1)}' for k, v in W['robust_sd_ms'].items())}; "
                 f"plain SD: {', '.join(f'{k} {fmt(v, 1)}' for k, v in W['sd_ms'].items())}. "
                 "With EI = oI - oE, IF = oF - oI and EF = oF - oE for independent file-start offsets, a moving force start shifts "
                 "IF and EF together, a moving IMU start shifts IF and EI in opposite directions, a moving EEG start shifts EI and "
                 "EF together. The implied within-subject SD of each file start (ms, from those covariances; 0 where the implied "
                 f"variance is not positive): {', '.join(f'{k} {fmt(v, 1)}' for k, v in W['implied_start_offset_sd_ms'].items())}.")
        L.append("")
        L.append(f"Closure EF - EI - IF (ms; q25, median, q75, n) by speed: {fmt_d(S.get('closure_ms_by_speed', {}))}. The EEG "
                 "envelope cancels in EF - EI, so the closure compares two independent IMU-to-force estimators (head impact "
                 "energy against plate impact energy, pelvis acceleration against total vertical force); its within-subject "
                 "SD above is the agreement of the two.")
    L.append("")
    if S.get("speed_0.5_excess_ms"):
        X = S["speed_0.5_excess_ms"]
        L.append("The slowest speed against the two faster ones, per evaluable subject, excess = lag(0.5) - mean(lag(0.75), lag(1)) "
                 "(ms; quantiles p10, p25, p50, p75, p90; subjects with an excess over 40 ms, and their min, median, max): "
                 + "; ".join(f"{p}: {fmt(X[p]['quantiles_p10_p25_p50_p75_p90'], 1)}, {X[p]['subjects_over_40ms']} of {X[p]['subjects']} "
                             f"over 40 ms {fmt(X[p]['excess_over_40ms_min_median_max'], 1)}" for p in PAIRS_PRIMARY)
                 + f". Of the IF subjects over 40 ms, {X['IF_over_40ms_also_EF_evaluable']} are evaluable for EF and "
                 f"{X['IF_and_EF_over_40ms']} are over 40 ms in EF as well.")
        L.append("")
    if S.get("lag_ms_by_imu_processing"):
        L.append("MVN processing mode of the IMU export (median lag in ms, records, subjects): "
                 + "; ".join(f"{q}: " + ", ".join(f"{p} {fmt(v[0], 1)} ({v[1]} records)" for p, v in d.items())
                             + f", subjects {next(iter(d.values()))[2] if d else []}"
                             for q, d in S["lag_ms_by_imu_processing"].items()) + ".")
        L.append("")
    L.append("## 3. Spread across speeds, per subject")
    L.append("")
    L.append("| pair | spread quantiles p10, p25, p50, p75, p90 (ms) | fraction within 20 ms after removing the population speed median (diagnostic, not the gate) | fraction within when unevaluable subjects count as failures (denominator: subjects with both streams at 3 speeds) |")
    L.append("|---|---|---|---|")
    for p in PAIRS_PRIMARY + PAIRS_SENS:
        d = P[p]
        L.append(f"| {p} | {fmt(d['spread_ms_quantiles'], 1)} | {fmt(d['fraction_within_speed_centred_diagnostic'], 3)} | "
                 f"{fmt(d['fraction_within_counting_unevaluable_as_fail'], 3)} |")
    L.append("")
    L.append(f"Subjects with usable EEG and IMU: {S['subjects_eeg_and_imu_usable']}; with both at all 3 speeds: "
             f"{S['subjects_eeg_imu_all_3_speeds']}; with EEG, IMU and force at all 3 speeds: {S['subjects_eeg_imu_force_all_3_speeds']}; "
             f"EI evaluable (3 measured speeds): {S['subjects_with_eeg_imu_evaluable_of_eeg_imu_subjects'][0]}.")
    L.append("")
    L.append("Record counts against the MoveAtlas canonical manifests (source rows): "
             + ", ".join(f"{fam} {S[f'n_records_in_manifest_{fam}']} in manifest, {S[f'n_records_has_{fam}']} raw files"
                         for fam in ("eeg", "imu", "force", "emg"))
             + f"; manifest membership differing from usability here: {S['n_records_manifest_usable_mismatch']}.")
    L.append("")
    L.append("## 4. Duplicated recordings")
    L.append("")
    L.append("Signal arrays hashed per record (header lines excluded). Groups of records holding identical data:")
    for fam in ("eeg", "imu", "force", "emg"):
        L.append(f"- {fam}: {S['duplicates'][fam] if S['duplicates'][fam] else 'none'}.")
    L.append("")
    if S["duplicates"]["eeg"]:
        L.append("Evidence per duplicated EEG record (does the EEG artefact align with this record's own IMU or force?):")
        L.append("")
        L.append("| record | IMU usable | force usable | EI measured (r) | EF measured (r) | EEG minus MVN header start, whole hours |")
        L.append("|---|---|---|---|---|---|")
        for g in S["duplicates_eeg_evidence"]:
            for rid, e in g.items():
                ei = f"{e['EI_measured']} ({fmt(e['EI_r'], 2)})" if e["EI_measured"] != "" else "n/a"
                ef = f"{e['EF_measured']} ({fmt(e['EF_r'], 2)})" if e["EF_measured"] != "" else "n/a"
                hh = e["eeg_minus_imu_start_hours"] if e["eeg_minus_imu_start_hours"] != "" else "n/a"
                L.append(f"| {rid} | {e['has_imu']} | {e['has_force']} | {ei} | {ef} | {hh} |")
        L.append("")
        L.append(f"Records whose EEG header start differs from the MVN start by a number of whole hours other than the modal "
                 f"one: {S['records_header_hours_not_modal']}; of these, members of a duplicated EEG group: "
                 f"{S['records_header_hours_not_modal_that_are_eeg_duplicates']}. The gate leaves every member of a duplicated "
                 "group out of the EEG pairs; the as-released line in the verdict section keeps them.")
        L.append("")
    L.append("Files: records.csv (one row per subject and speed, timing facts and every lag with SE, r, drift, flags), "
             "subjects.csv (one row per subject and pair), summary.json, sync_lags.png.")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def plot(out, subj_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    x = np.arange(3)
    for i, p in enumerate(PAIRS_PRIMARY):
        rows = [r for r in subj_rows if r["pair"] == p and r["class"] != "not_evaluable"]
        allv = np.array([r[f"lag_{s}"] for r in rows for s in SPEEDS], float)
        lo, hi = (np.percentile(allv, 2) - 20, np.percentile(allv, 98) + 20) if len(allv) else (-1, 1)
        off = 0
        for r in rows:
            y = [r[f"lag_{s}"] for s in SPEEDS]
            off += int(any(v < lo or v > hi for v in y))
            ax[i].plot(x, y, "-o", ms=3, lw=0.8, color="#2a7ab0" if r["within_tol"] else "#c0392b", alpha=0.7)
        ax[i].set_ylim(lo, hi)
        ax[i].set_xticks(x, ["0.5", "0.75", "1.0"])
        ax[i].set_xlabel("treadmill speed (m/s)")
        ax[i].set_ylabel(f"lag {p} (ms)")
        ax[i].set_title(f"{p}: blue within 20 ms, red outside ({off} subjects partly off scale)", fontsize=9)
        ax[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "sync_lags.png"), dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
