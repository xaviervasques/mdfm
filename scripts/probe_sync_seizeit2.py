"""seizeit2 cross-module synchronisation check, addendum gate G15 (task N0.3a), read-only.

SeizeIT2 (datasets/eeg/seizeit2/raw, BIDS, OpenNeuro ds005873) records each run with two
wearable modules: the EEG module (upper back, behind-the-ear electrodes) and the ECG and EMG
module (left chest, EMG on the left deltoid). Each module has a tri-axial accelerometer. G15:
cross-correlate the accelerometer magnitudes of the two modules in 10 min windows at the start
and at the end of every run; cross-module use of a run is void if the offset exceeds 40 ms or
the drift across the run exceeds 40 ms.

What the script does, in order.
1. Layout, from headers only: every EDF header of every run (eeg, ecg, emg, mov) and every mov
   JSON sidecar is read; the channel label sets are counted (channel_layout.csv). The events.tsv
   files (seizure annotations) are never opened, the "EDF Annotations" signal is never decoded
   and processed/eeg_ingest_audit.csv is not read.
2. Main pass, windows only: for every run whose mov EDF carries both accelerometer triplets,
   the six accelerometer channels are read for the candidate windows only (plus 20 s of filter
   padding). Signal per module: zero-filled samples (all three axes at physical 0, the
   converter's padding) are interpolated, and gaps longer than 2 samples are masked (1 s
   dilation) after filtering; an axis saturated (digital extreme) in more than 1 percent of
   the window is dropped; each remaining axis is high-passed (4th-order Butterworth, 0.5 Hz,
   zero phase) and the Euclidean norm of the high-passed axes is the magnitude (the dynamic
   acceleration magnitude, insensitive to the per-axis offsets and scale errors of these
   files). The same zero-phase filter on both modules adds no lag.
   Estimator per window: Pearson correlation of the central part of the EEG-module magnitude
   (window minus 60 s at each end) against the ECG-EMG-module magnitude at every lag within
   +-60 s (full overlap at every lag); integer peak, then sub-sample lag by parabolic
   interpolation of the three samples around the peak; sharpness is the relative curvature
   -(r[k-1] - 2 r[k] + r[k+1]) / r[k]; also the peak-to-second-peak ratio (outside +-1 s) and
   a robust z of the peak against lags farther than 2 s. A windowed-sinc interpolation of the
   correlation serves as a check of the parabolic sub-sample estimate.
   Acceptance: the central part is split into 8 sub-windows, each correlated on its own
   within +-60 s; the window is accepted when at least 4 sub-window lags lie within 80 ms
   (2 samples) of the window lag and the peak is more than 1 s inside the search range. The
   spread of the agreeing sub-window lags gives the standard error of the window lag.
   Anchors: the start anchor is the first accepted 10 min window counted from the start of
   the run, the end anchor the first accepted counted back from the end. Windows with less
   than 90 percent valid samples in both modules are skipped; at most 6 windows with enough
   valid data are tried per side, and a side never crosses the run midpoint. Runs shorter
   than 1200 s use two half-run windows (lag range min(60 s, window / 5)); runs shorter than
   240 s are not checkable.
   Sign: a positive lag means a movement appears later in the ECG-EMG-module channels than in
   the EEG-module channels of the same file.
   Verdict (strict, used for the counts): the line through the two anchor lags is
   extrapolated to the first and last second of the run; offset = the larger absolute
   extrapolated lag, drift = the extrapolated change across the whole run; void when either
   exceeds 40 ms, pass otherwise. The literal reading (offset = start-window lag, drift = end
   minus start window lag) is reported beside it. A verdict is robust when it holds with the
   decisive quantities moved by 2 standard errors toward the threshold.
3. Trace pass (secondary, whole files for a subset): for the longest run of every subject
   (duration at least 4 h), every non-overlapping 10 min window is estimated the same way,
   to test whether the two-anchor line describes the middle of the run.
4. Duplicated data: the first and the last 30 data records of every mov EDF (the EDF
   Annotations signal left out) are fingerprinted; for every pair of runs sharing a
   fingerprint, each modality (eeg, ecg, emg, mov) of the shorter run is compared record by
   record with the head or the tail of the longer run. A run contained in the middle of
   another is not found by this scan.

Outputs in reports/sync_seizeit2/: runs.csv (one row per run), windows.csv (every candidate
window tried), per_subject.csv, channel_layout.csv, acc_header_ranges.csv, traces.csv,
trace_fit.csv, summary.json, summary.md, known_delay_test.csv, consecutive_runs.csv,
contained_runs.csv, mov_sidecar_facts.csv, _provenance.json.
Resumable: per-run results are cached in _cache/ during a run and the cache is removed at the
end of a complete run (--keep-cache keeps it).
Run: python scripts/probe_sync_seizeit2.py --moveatlas <moveatlas>
Options (parsed before the common ones): --limit N (first N runs only, for calibration),
--no-trace, --trace-min-s S, --fresh (ignore and reset caches), --keep-cache.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
from scipy import signal
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import parse_args, write_provenance  # noqa: E402

# ---------------------------------------------------------------- parameters (fixed)
WIN_S = 600            # anchor window length (s), G15
LAG_S = 60             # lag search range (s) in full mode
N_SUB = 8              # sub-windows for the agreement test
AGREE_MS = 80.0        # sub-window agreement tolerance (2 samples at 25 Hz)
MIN_AGREE = 4          # sub-windows that must agree
MAX_ATTEMPTS = 6       # windows with enough valid data tried per side
MIN_VALID = 0.90       # minimum fraction of samples valid in both modules
SAT_MAX = 0.01         # an axis saturated in more than this fraction is dropped
HP_HZ = 0.5            # per-axis high-pass cut-off (Hz)
PAD_S = 20             # filter padding read on each side of a window (s)
SHORT_GAP = 2          # zero-filled gaps up to this many samples are only interpolated
THRESH_MS = 40.0       # G15 threshold for offset and drift
SHORT_RUN_S = 2 * WIN_S
MIN_HALF_S = 120       # half-window floor for short runs
TRACE_MIN_S = 4 * 3600
KD_WINDOWS = 200       # windows in the known-delay test
KD_SHIFTS = (0.25, 0.5, 1.0, 1.5)  # applied delays (samples)
SENS_MIN_AGREE = (3,)  # looser acceptance rule reported as a sensitivity check
DUP_REC = 30           # data records fingerprinted at each end of a mov EDF
SENTINEL_NORM = 10.0   # a module's median raw norm above this: an axis sits near the +-32.77 sentinel
SENTINEL2_NORM = 40.0  # above this: two or more axes sit near the sentinel

EEG_ACC = ["EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z"]
ECG_ACC = ["ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z"]
NAME_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_task-([^_]+)_run-([^_]+)_(eeg|ecg|emg|mov)\.edf$")

# ---------------------------------------------------------------- arguments
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--limit", type=int, default=0)
pre.add_argument("--no-trace", action="store_true")
pre.add_argument("--trace-min-s", type=int, default=TRACE_MIN_S)
pre.add_argument("--fresh", action="store_true")
pre.add_argument("--keep-cache", action="store_true")
OPT, rest = pre.parse_known_args()
if "--out" not in rest:
    rest += ["--out", os.path.join(os.path.dirname(HERE), "reports", "sync_seizeit2")]
sys.argv = [sys.argv[0]] + rest
ARGS = parse_args(__doc__.splitlines()[0])
RAW = f"{ARGS.moveatlas}/datasets/eeg/seizeit2/raw"
OUT = ARGS.out
CACHE = os.path.join(OUT, "_cache")
os.makedirs(CACHE, exist_ok=True)
if OPT.fresh:
    for _n in ("runs.jsonl", "traces.jsonl"):
        if os.path.isfile(os.path.join(CACHE, _n)):
            os.remove(os.path.join(CACHE, _n))


# ---------------------------------------------------------------- EDF header
def edf_header(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256].decode("ascii").strip())
        sh = f.read(256 * ns)
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names = ["label", "transducer", "physdim", "physmin", "physmax", "digmin", "digmax", "prefilt", "nsamp", "res"]
    sig, pos = {}, 0
    for n, w in zip(names, widths):
        sig[n] = [sh[pos + i * w: pos + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]
        pos += w * ns
    hd = {"nbytes": int(h[184:192].decode("ascii").strip()), "nrec": int(h[236:244].decode("ascii").strip()),
          "recdur": float(h[244:252].decode("ascii").strip()), "ns": ns,
          "reserved": h[192:236].decode("latin-1").strip(), "sig": sig, "size": os.path.getsize(path)}
    nsamp = [int(x) for x in sig["nsamp"]]
    hd["nsamp"] = nsamp
    hd["rec_samples"] = sum(nsamp)
    hd["offsets"] = [sum(nsamp[:i]) for i in range(ns)]
    hd["size_ok"] = hd["nbytes"] + hd["nrec"] * hd["rec_samples"] * 2 == hd["size"]
    return hd


def chan_info(hd, label):
    i = hd["sig"]["label"].index(label)
    pmin, pmax = float(hd["sig"]["physmin"][i]), float(hd["sig"]["physmax"][i])
    dmin, dmax = float(hd["sig"]["digmin"][i]), float(hd["sig"]["digmax"][i])
    g = (pmax - pmin) / (dmax - dmin) if dmax != dmin else float("nan")
    d0 = dmin + (0.0 - pmin) / g if g and math.isfinite(g) and g != 0 else float("nan")
    return {"i": i, "off": hd["offsets"][i], "n": hd["nsamp"][i], "pmin": pmin, "pmax": pmax,
            "dmin": dmin, "dmax": dmax, "g": g, "d0": d0}


def read_block(path, hd, infos, r0, r1):
    """Records r0..r1-1 of the six accelerometer channels; the annotation signal is never decoded."""
    rb = hd["rec_samples"] * 2
    with open(path, "rb") as f:
        f.seek(hd["nbytes"] + r0 * rb)
        buf = f.read((r1 - r0) * rb)
    a = np.frombuffer(buf, dtype="<i2").reshape(r1 - r0, hd["rec_samples"])
    out = []
    for ci in infos:
        out.append(a[:, ci["off"]:ci["off"] + ci["n"]].astype(np.float64).ravel())
    return out, len(buf)


# ---------------------------------------------------------------- signal processing
SOS = None


def sos_for(fs):
    global SOS
    if SOS is None or SOS[0] != fs:
        SOS = (fs, signal.butter(4, HP_HZ, "high", fs=fs, output="sos"))
    return SOS[1]


def runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def module_signal(digs, infos, core, fs):
    """digs: 3 digital arrays for one module over the padded block; core: slice of the window.
    Returns dict with magnitude over the core, zero mask (block), long-gap mask (block), axes used."""
    zero = np.ones_like(digs[0], dtype=bool)
    for d, ci in zip(digs, infos):
        zero &= np.abs(d - ci["d0"]) <= 1.0 if math.isfinite(ci["d0"]) else np.zeros_like(d, dtype=bool)
    good = ~zero
    used, sat_fracs, hps = [], [], []
    phys = [(d - ci["dmin"]) * ci["g"] + ci["pmin"] for d, ci in zip(digs, infos)]
    gc = good[core]
    raw_norm = float(np.median(np.sqrt(sum(x[core][gc] ** 2 for x in phys)))) if gc.any() else float("nan")
    for d, ci in zip(digs, infos):
        sat = ((d >= ci["dmax"]) | (d <= ci["dmin"])) & good
        sf = float(sat[core].sum()) / max(1, int(good[core].sum()))
        sat_fracs.append(sf)
        if sf > SAT_MAX:
            continue
        x = (d - ci["dmin"]) * ci["g"] + ci["pmin"]
        if zero.any() and good.sum() >= 2:
            idx = np.arange(len(x))
            x = x.copy()
            x[zero] = np.interp(idx[zero], idx[good], x[good])
        hps.append(signal.sosfiltfilt(sos_for(fs), x - x.mean()))
        used.append(ci["label"])
    st, en = runs_of(zero)
    long_gap = np.zeros_like(zero)
    for s, e in zip(st, en):
        if e - s > SHORT_GAP:
            long_gap[max(0, s - int(fs)):e + int(fs)] = True
    mag = np.sqrt(sum(h ** 2 for h in hps))[core] if len(hps) >= 2 else None
    return {"mag": mag, "zero": zero, "long_gap": long_gap, "used": used, "sat": sat_fracs, "raw_norm": raw_norm}


def ncc(a, b, L):
    """Pearson r of a[L:len-L] against b shifted by k in [-L, L] (index k+L); positive k: b lags a."""
    seg = a[L:len(a) - L]
    seg = seg - seg.mean()
    n = len(seg)
    ss = float((seg ** 2).sum())
    if n < 4 or ss <= 0:
        return None
    cs = np.concatenate([[0.0], np.cumsum(b)])
    cs2 = np.concatenate([[0.0], np.cumsum(b * b)])
    s1 = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    varb = s2 - s1 * s1 / n
    cc = signal.correlate(b, seg, mode="valid", method="fft")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cc / np.sqrt(varb * ss)
    r[~np.isfinite(r) | (varb <= 1e-20)] = -1.0
    return r


def parabolic(r, k):
    if 0 < k < len(r) - 1:
        c0, c1, c2 = r[k - 1], r[k], r[k + 1]
        den = c0 - 2.0 * c1 + c2
        if den < 0:
            return 0.5 * (c0 - c2) / den, (-den / c1) if c1 > 0 else float("nan")
    return 0.0, float("nan")


def sinc_peak(r, k, half=16, step=0.01):
    n = np.arange(max(0, k - half), min(len(r), k + half + 1))
    t = np.arange(k - 1.0, k + 1.0 + 1e-9, step)
    x = t[:, None] - n[None, :]
    wts = np.sinc(x) * np.sinc(x / half)
    v = (wts * r[n][None, :]).sum(1)
    return float(t[int(np.argmax(v))] - k)


def estimate(a, b, L, fs):
    """Window estimate. a: EEG-module magnitude, b: ECG-EMG-module magnitude (same length)."""
    r = ncc(a, b, L)
    if r is None:
        return {"status": "flat"}
    k = int(np.argmax(r))
    rk = float(r[k])
    dlt, sharp = parabolic(r, k)
    lag_ms = (k - L + dlt) / fs * 1000.0
    lag_sinc_ms = (k - L + sinc_peak(r, k)) / fs * 1000.0
    idx = np.arange(len(r))
    far1 = np.abs(idx - k) > fs
    far2 = np.abs(idx - k) > 2 * fs
    psr = rk / max(float(r[far1].max()), 1e-9) if far1.any() else float("nan")
    med = float(np.median(r[far2])) if far2.any() else float("nan")
    mad = 1.4826 * float(np.median(np.abs(r[far2] - med))) if far2.any() else float("nan")
    z = (rk - med) / mad if mad and mad > 0 else float("nan")
    m = (len(a) - 2 * L) // N_SUB
    subs = []
    for j in range(N_SUB):
        rs = ncc(a[j * m: j * m + m + 2 * L], b[j * m: j * m + m + 2 * L], L)
        if rs is None:
            subs.append(float("nan"))
            continue
        kk = int(np.argmax(rs))
        d2, _ = parabolic(rs, kk)
        subs.append((kk - L + d2) / fs * 1000.0)
    subs = np.array(subs)
    agree = np.isfinite(subs) & (np.abs(subs - lag_ms) <= AGREE_MS)
    na = int(agree.sum())
    sd = float(np.std(subs[agree], ddof=1)) if na >= 2 else float("nan")
    se = sd / math.sqrt(na) if na >= 2 else float("nan")
    edge = abs(k - L) >= L - fs
    ok = (na >= MIN_AGREE) and (not edge) and rk > 0
    status = "ok" if ok else ("edge_peak" if edge else "no_agreement")
    return {"status": status, "lag_ms": lag_ms, "lag_sinc_ms": lag_sinc_ms, "frac_samples": dlt,
            "r_peak": rk, "sharpness": sharp, "psr": psr, "z": z, "n_agree": na,
            "sub_median_ms": float(np.median(subs[agree])) if na else float("nan"), "sub_sd_ms": sd, "se_ms": se}


def eval_window(path, hd, einf, cinf, r0, r1, lag_samples, fs, stats, want_signals=False):
    """Evaluate the window of records [r0, r1). Returns a dict with status and metrics
    (and the two prepared magnitudes when want_signals is set and the window is estimable)."""
    rr0 = max(0, r0 - PAD_S)
    rr1 = min(hd["nrec"], r1 + PAD_S)
    spr = int(round(fs * hd["recdur"]))
    digs, nb = read_block(path, hd, einf + cinf, rr0, rr1)
    stats["windows_read"] += 1
    stats["bytes_read"] += nb
    core = slice((r0 - rr0) * spr, (r1 - rr0) * spr)
    e = module_signal(digs[:3], [dict(ci, label=l) for ci, l in zip(einf, EEG_ACC)], core, fs)
    c = module_signal(digs[3:], [dict(ci, label=l) for ci, l in zip(cinf, ECG_ACC)], core, fs)
    res = {"win_start_s": r0 * hd["recdur"], "win_len_s": (r1 - r0) * hd["recdur"],
           "axes_eeg": len(e["used"]), "axes_ecgemg": len(c["used"]),
           "sat_eeg": ";".join("%.4f" % x for x in e["sat"]), "sat_ecgemg": ";".join("%.4f" % x for x in c["sat"]),
           "raw_norm_eeg": e["raw_norm"], "raw_norm_ecgemg": c["raw_norm"]}
    valid = ~(e["zero"][core] | c["zero"][core])
    res["valid_frac"] = float(valid.mean())
    if res["valid_frac"] < MIN_VALID:
        res["status"] = "gap"
        return res
    if e["mag"] is None or c["mag"] is None:
        res["status"] = "saturated"
        return res
    mask = (e["long_gap"] | c["long_gap"])[core]
    res["masked_frac"] = float(mask.mean())
    if res["masked_frac"] > 2 * (1.0 - MIN_VALID):
        res["status"] = "gap"
        return res
    a = e["mag"].copy()
    b = c["mag"].copy()
    a -= a[~mask].mean()
    b -= b[~mask].mean()
    a[mask] = 0.0
    b[mask] = 0.0
    res["std_eeg"] = float(a[~mask].std())
    res["std_ecgemg"] = float(b[~mask].std())
    if res["std_eeg"] <= 1e-9 or res["std_ecgemg"] <= 1e-9:
        res["status"] = "flat"
        return res
    res.update(estimate(a, b, lag_samples, fs))
    if want_signals:
        res["_a"], res["_b"] = a, b
    return res


def data_cols(hd):
    """Sample columns of one record that hold data signals (the EDF Annotations signal left out)."""
    cols = []
    for i, lab in enumerate(hd["sig"]["label"]):
        if lab != "EDF Annotations":
            cols.extend(range(hd["offsets"][i], hd["offsets"][i] + hd["nsamp"][i]))
    return np.array(cols, dtype=np.int64)


def read_data_records(path, hd, r0, r1, stats):
    """Records r0..r1-1, data signals only (annotation bytes are loaded with the records, dropped, never decoded)."""
    rb = hd["rec_samples"] * 2
    with open(path, "rb") as f:
        f.seek(hd["nbytes"] + r0 * rb)
        buf = f.read((r1 - r0) * rb)
    stats["dup_bytes"] = stats.get("dup_bytes", 0) + len(buf)
    return np.frombuffer(buf, dtype="<i2").reshape(r1 - r0, hd["rec_samples"])[:, data_cols(hd)]


def header_sig(hd):
    s = hd["sig"]
    return (tuple(s["label"]), tuple(s["physmin"]), tuple(s["physmax"]), tuple(s["digmin"]), tuple(s["digmax"]),
            tuple(hd["nsamp"]), hd["recdur"])


def duplicate_scan(files, keys, stats):
    """Pairs of runs where the shorter run's data records equal the head or the tail of the longer run."""
    import hashlib
    fps, nrec = {}, {}
    for k in keys:
        if "mov" not in files[k]:
            continue
        p = files[k]["mov"]
        hd = edf_header(p)
        n = min(DUP_REC, hd["nrec"])
        if n <= 0:
            continue
        nrec[k] = hd["nrec"]
        for end, (r0, r1) in (("head", (0, n)), ("tail", (hd["nrec"] - n, hd["nrec"]))):
            h = hashlib.sha1(read_data_records(p, hd, r0, r1, stats).tobytes()).hexdigest()
            fps.setdefault((end, h), []).append(k)
    pairs = {}
    for (end, _), ks in fps.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs.setdefault(tuple(sorted((ks[i], ks[j]))), set()).add(end)
    out = []
    for (ka, kb), ends in sorted(pairs.items()):
        if nrec[ka] > nrec[kb]:
            ka, kb = kb, ka
        row = {"run_a": run_key_str(ka), "run_b": run_key_str(kb), "records_a": nrec[ka], "records_b": nrec[kb],
               "fingerprint_match": ";".join(sorted(ends))}
        offs = sorted({0 if e == "head" else nrec[kb] - nrec[ka] for e in ends})
        found = set()
        for m in ("eeg", "ecg", "emg", "mov"):
            if m not in files[ka] or m not in files[kb]:
                row[f"{m}_contained"] = None
                continue
            ha, hb = edf_header(files[ka][m]), edf_header(files[kb][m])
            row[f"{m}_headers_equal"] = header_sig(ha) == header_sig(hb)
            ok = False
            if ha["nrec"] <= hb["nrec"] and data_cols(ha).size == data_cols(hb).size:
                A = read_data_records(files[ka][m], ha, 0, ha["nrec"], stats)
                for off in offs:
                    if off + ha["nrec"] <= hb["nrec"] and np.array_equal(
                            A, read_data_records(files[kb][m], hb, off, off + ha["nrec"], stats)):
                        ok = True
                        found.add(off)
                        break
            row[f"{m}_contained"] = ok
        row["offset_records"] = ";".join(str(o) for o in sorted(found))
        row["contained_all_modalities"] = all(row.get(f"{m}_contained") is not False for m in ("eeg", "ecg", "emg", "mov"))
        out.append(row)
    return out, len(nrec)


def frac_delay(x, d):
    """Delay x by d samples (band-limited, circular; d is at most a few samples here)."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d), n)


# ---------------------------------------------------------------- inventory
def scan_inventory():
    files = {}
    for path in glob.glob(f"{RAW}/sub-*/ses-*/*/*.edf"):
        base = os.path.basename(path)
        if base.startswith("._"):
            continue
        m = NAME_RE.match(base)
        if not m:
            continue
        sub, ses, task, run, mod = m.groups()
        files.setdefault((sub, ses, task, run), {})[mod] = path.replace("\\", "/")
    return files


def run_key_str(k):
    return f"{k[0]}_{k[1]}_task-{k[2]}_run-{k[3]}"


def cache_load(name):
    p = os.path.join(CACHE, name)
    out = {}
    if OPT.fresh or not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["key"]] = d
    return out


def cache_append(name, d):
    with open(os.path.join(CACHE, name), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(d) + "\n")


def jnum(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def clean(d):
    return {k: jnum(v) for k, v in d.items()}


# ---------------------------------------------------------------- per-run analysis
def verdict_from(ls, le, ses_, see, ts, te, T):
    """G15 verdict from the two anchor lags (ms), their standard errors, centres (s) and run length (s)."""
    span = te - ts
    slope = (le - ls) / span  # ms per s
    lag0 = ls - slope * ts
    lagT = ls + slope * (T - ts)
    se0 = math.sqrt((1 + ts / span) ** 2 * ses_ ** 2 + (ts / span) ** 2 * see ** 2)
    seT = math.sqrt(((T - ts) / span) ** 2 * see ** 2 + (1 - (T - ts) / span) ** 2 * ses_ ** 2)
    drift_run = slope * T
    se_drift_run = math.sqrt(ses_ ** 2 + see ** 2) * T / span
    out = {"span_s": span, "drift_ms": le - ls, "drift_ppm": slope * 1000.0, "lag_run_start_ms": lag0,
           "lag_run_end_ms": lagT, "drift_run_ms": drift_run, "se_drift_run_ms": se_drift_run}
    off, se_off = (abs(lag0), se0) if abs(lag0) >= abs(lagT) else (abs(lagT), seT)
    out["offset_ms"] = off
    out["se_offset_ms"] = se_off
    fail_off = off > THRESH_MS
    fail_drift = abs(drift_run) > THRESH_MS
    out["verdict"] = "void" if (fail_off or fail_drift) else "pass"
    out["reason"] = ("offset_and_drift" if (fail_off and fail_drift) else "offset" if fail_off
                     else "drift" if fail_drift else "")
    if out["verdict"] == "pass":
        robust = (THRESH_MS - off >= 2 * se_off) and (THRESH_MS - abs(drift_run) >= 2 * se_drift_run)
    else:
        robust = (fail_off and off - THRESH_MS >= 2 * se_off) or (fail_drift and abs(drift_run) - THRESH_MS >= 2 * se_drift_run)
    out["verdict_robust"] = bool(robust)
    out["verdict_literal"] = "void" if (abs(ls) > THRESH_MS or abs(le - ls) > THRESH_MS) else "pass"
    return out


def candidate_windows(nrec, W, side, short):
    """Windows [r0, r1) in records, from the run edge inward, never crossing the midpoint."""
    if short:
        return [(0, W)] if side == "start" else [(nrec - W, nrec)]
    half = nrec / 2.0
    out, k = [], 0
    while True:
        r0, r1 = (k * W, (k + 1) * W) if side == "start" else (nrec - (k + 1) * W, nrec - k * W)
        if (side == "start" and r1 > half) or (side == "end" and r0 < half):
            break
        out.append((r0, r1))
        k += 1
    return out


def find_anchor(path, hd, einf, cinf, side, W, Ls, fs, short, stats):
    tried = []
    attempts = 0
    for k, (r0, r1) in enumerate(candidate_windows(hd["nrec"], W, side, short)):
        res = eval_window(path, hd, einf, cinf, r0, r1, Ls, fs, stats)
        res["side"] = side
        res["attempt"] = k
        tried.append(res)
        if res["status"] != "gap":
            attempts += 1
        if res["status"] == "ok":
            return res, tried
        if attempts >= MAX_ATTEMPTS:
            break
    return None, tried


def analyse_run(key, mods, stats):
    row = {"key": run_key_str(key), "subject": key[0], "session": key[1], "run": key[3],
           "has_eeg_edf": "eeg" in mods, "has_ecg_edf": "ecg" in mods, "has_emg_edf": "emg" in mods,
           "has_mov_edf": "mov" in mods}
    wins = []
    if "eeg" in mods:
        he = edf_header(mods["eeg"])
        row["eeg_duration_s"] = he["nrec"] * he["recdur"]
    if "mov" not in mods:
        row.update(verdict="not_checkable", reason="no_mov_edf")
        return row, wins
    hd = edf_header(mods["mov"])
    labels = hd["sig"]["label"]
    row["mov_edf_type"] = hd["reserved"][:5]
    row["mov_size_ok"] = hd["size_ok"]
    row["mov_nrec"] = hd["nrec"]
    row["mov_bytes"] = hd["size"]
    row["mov_duration_s"] = hd["nrec"] * hd["recdur"]
    row["durations_equal"] = (row.get("eeg_duration_s") == row["mov_duration_s"])
    row["acc_eeg_module"] = all(l in labels for l in EEG_ACC)
    row["acc_ecgemg_module"] = all(l in labels for l in ECG_ACC)
    if not (row["acc_eeg_module"] and row["acc_ecgemg_module"]):
        row.update(verdict="not_checkable", reason="missing_acc_channels")
        return row, wins
    einf = [chan_info(hd, l) for l in EEG_ACC]
    cinf = [chan_info(hd, l) for l in ECG_ACC]
    fss = {ci["n"] / hd["recdur"] for ci in einf + cinf}
    if len(fss) != 1 or not hd["size_ok"] or hd["reserved"].startswith("EDF+D"):
        row.update(verdict="not_checkable", reason="unexpected_edf_layout")
        return row, wins
    fs = fss.pop()
    row["acc_fs_hz"] = fs
    nrec = hd["nrec"]
    rd = hd["recdur"]
    short = row["mov_duration_s"] < SHORT_RUN_S
    if not short:
        W, Ls = int(round(WIN_S / rd)), int(LAG_S * fs)
    else:
        W = nrec // 2
        Ls = int(min(LAG_S * fs, (W * rd * fs) // 5))
    row["mode"] = "short" if short else "full"
    row["window_s"] = W * rd
    row["lag_range_s"] = Ls / fs
    if W * rd < MIN_HALF_S:
        row.update(verdict="not_checkable", reason="run_too_short")
        return row, wins
    anchors = {}
    for side in ("start", "end"):
        res, tried = find_anchor(mods["mov"], hd, einf, cinf, side, W, Ls, fs, short, stats)
        for t in tried:
            wins.append(dict(t, key=row["key"], subject=key[0]))
        row[f"{side}_tried"] = ",".join(t["status"] for t in tried)
        if res is not None:
            anchors[side] = res
            ctr = res["win_start_s"] + res["win_len_s"] / 2.0
            row[f"{side}_win_start_s"] = res["win_start_s"]
            row[f"{side}_centre_s"] = ctr
            row[f"{side}_shift_s"] = (res["win_start_s"] if side == "start"
                                      else row["mov_duration_s"] - (res["win_start_s"] + res["win_len_s"]))
            for f_ in ("lag_ms", "lag_sinc_ms", "r_peak", "sharpness", "psr", "z", "n_agree", "se_ms", "sub_sd_ms",
                       "axes_eeg", "axes_ecgemg", "valid_frac"):
                row[f"{side}_{f_}"] = res.get(f_)
    if len(anchors) < 2:
        statuses = [w["status"] for w in wins]
        if statuses and all(s == "gap" for s in statuses):
            reason = "no_valid_overlap"
        elif "start" not in anchors and "end" not in anchors:
            reason = "no_accepted_window"
        else:
            reason = "one_anchor_only"
        row.update(verdict="not_checkable", reason=reason)
        return row, wins
    row.update(verdict_from(anchors["start"]["lag_ms"], anchors["end"]["lag_ms"], anchors["start"]["se_ms"],
                            anchors["end"]["se_ms"], row["start_centre_s"], row["end_centre_s"], row["mov_duration_s"]))
    row["anchors_primary"] = (row["start_shift_s"] == 0 and row["end_shift_s"] == 0)
    return row, wins


# ---------------------------------------------------------------- trace pass
def trace_run(key, path, stats):
    hd = edf_header(path)
    einf = [chan_info(hd, l) for l in EEG_ACC]
    cinf = [chan_info(hd, l) for l in ECG_ACC]
    fs = einf[0]["n"] / hd["recdur"]
    Ls = int(LAG_S * fs)
    out = []
    for k in range(hd["nrec"] // WIN_S):
        r0, r1 = k * WIN_S, (k + 1) * WIN_S
        res = eval_window(path, hd, einf, cinf, r0, r1, Ls, fs, stats)
        out.append({"key": run_key_str(key), "subject": key[0], "win_start_s": r0, "status": res["status"],
                    "lag_ms": res.get("lag_ms"), "r_peak": res.get("r_peak"), "n_agree": res.get("n_agree"),
                    "se_ms": res.get("se_ms")})
    return out


# ---------------------------------------------------------------- output helpers
def write_csv(name, rows, fields=None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})
    return fields


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6g}" if abs(v) < 1e5 else f"{v:.1f}"
    return v


def q(vals, p):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(np.percentile(v, p)) if v else None


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    stats = {"windows_read": 0, "bytes_read": 0}
    files = scan_inventory()
    keys = sorted(files)
    print(f"runs found: {len(keys)}", flush=True)

    # 1. layout from headers and sidecars
    layout = {}
    json_facts = {}
    acc_ranges = {}
    for k in keys:
        for mod, path in files[k].items():
            hd = edf_header(path)
            labs = tuple(l for l in hd["sig"]["label"])
            key = (mod, "; ".join(labs), ";".join(f"{n / hd['recdur']:g}" for n in hd["nsamp"]), hd["reserved"][:5])
            layout[key] = layout.get(key, 0) + 1
            if mod == "mov":
                for i, l in enumerate(labs):
                    if " ACC " in l:
                        rk = (l, hd["sig"]["physdim"][i], hd["sig"]["physmin"][i], hd["sig"]["physmax"][i],
                              hd["sig"]["digmin"][i], hd["sig"]["digmax"][i])
                        acc_ranges[rk] = acc_ranges.get(rk, 0) + 1
        if "mov" in files[k]:
            js = files[k]["mov"][:-4] + ".json"
            if os.path.isfile(js):
                with open(js, encoding="utf-8") as f:
                    d = json.load(f)
                for fld in ("MOVPlacementScheme", "MOVChannelCount", "SamplingFrequency", "RecordingType"):
                    kk = (fld, str(d.get(fld)))
                    json_facts[kk] = json_facts.get(kk, 0) + 1
    lay_rows = [{"modality": m, "labels": l, "fs_hz": f, "edf_type": t, "n_files": n}
                for (m, l, f, t), n in sorted(layout.items(), key=lambda x: (x[0][0], -x[1]))]
    write_csv("channel_layout.csv", lay_rows, ["modality", "labels", "fs_hz", "edf_type", "n_files"])
    json_rows = [{"field": a, "value": b, "n_files": n} for (a, b), n in sorted(json_facts.items())]
    write_csv("mov_sidecar_facts.csv", json_rows, ["field", "value", "n_files"])
    range_rows = [{"label": a, "physdim": b, "physmin": c, "physmax": d, "digmin": e, "digmax": f, "n_channels": n}
                  for (a, b, c, d, e, f), n in sorted(acc_ranges.items(), key=lambda x: (-x[1], x[0]))]
    write_csv("acc_header_ranges.csv", range_rows, ["label", "physdim", "physmin", "physmax", "digmin", "digmax", "n_channels"])
    print(f"layout: {len(lay_rows)} distinct channel sets", flush=True)

    # 2. main pass
    if OPT.limit:
        keys_main = keys[:OPT.limit]
    else:
        keys_main = keys
    cache = cache_load("runs.jsonl")
    rows, wins = [], []
    for i, k in enumerate(keys_main):
        ks = run_key_str(k)
        if ks in cache:
            rows.append(cache[ks]["row"])
            wins.extend(cache[ks]["wins"])
            continue
        w0, b0 = stats["windows_read"], stats["bytes_read"]
        row, ws = analyse_run(k, files[k], stats)
        row["windows_read"] = stats["windows_read"] - w0
        row["bytes_read"] = stats["bytes_read"] - b0
        row = clean(row)
        ws = [clean(w) for w in ws]
        cache_append("runs.jsonl", {"key": ks, "row": row, "wins": ws})
        rows.append(row)
        wins.extend(ws)
        if (i + 1) % 50 == 0 or i == len(keys_main) - 1:
            print(f"  {i + 1}/{len(keys_main)} runs, {stats['windows_read']} windows read, "
                  f"{stats['bytes_read'] / 1e9:.2f} GB, {time.time() - t_start:.0f} s", flush=True)

    run_fields = ["key", "subject", "session", "run", "has_eeg_edf", "has_ecg_edf", "has_emg_edf", "has_mov_edf",
                  "mov_edf_type", "mov_size_ok", "eeg_duration_s", "mov_duration_s", "durations_equal",
                  "acc_eeg_module", "acc_ecgemg_module", "acc_fs_hz", "mode", "window_s", "lag_range_s",
                  "start_tried", "start_win_start_s", "start_centre_s", "start_shift_s", "start_lag_ms", "start_lag_sinc_ms",
                  "start_r_peak", "start_sharpness", "start_psr", "start_z", "start_n_agree", "start_se_ms",
                  "start_sub_sd_ms", "start_axes_eeg", "start_axes_ecgemg", "start_valid_frac",
                  "end_tried", "end_win_start_s", "end_centre_s", "end_shift_s", "end_lag_ms", "end_lag_sinc_ms",
                  "end_r_peak", "end_sharpness", "end_psr", "end_z", "end_n_agree", "end_se_ms",
                  "end_sub_sd_ms", "end_axes_eeg", "end_axes_ecgemg", "end_valid_frac",
                  "span_s", "drift_ms", "drift_ppm", "lag_run_start_ms", "lag_run_end_ms", "drift_run_ms",
                  "se_drift_run_ms", "offset_ms", "se_offset_ms", "anchors_primary", "verdict", "reason",
                  "verdict_robust", "verdict_literal", "mov_bytes", "windows_read", "bytes_read"]
    write_csv("runs.csv", rows, run_fields)
    win_fields = ["key", "subject", "side", "attempt", "win_start_s", "win_len_s", "status", "valid_frac",
                  "axes_eeg", "axes_ecgemg", "sat_eeg", "sat_ecgemg", "raw_norm_eeg", "raw_norm_ecgemg", "masked_frac", "std_eeg", "std_ecgemg", "lag_ms", "lag_sinc_ms",
                  "frac_samples", "r_peak", "sharpness", "psr", "z", "n_agree", "sub_median_ms", "sub_sd_ms", "se_ms"]
    write_csv("windows.csv", wins, win_fields)

    # per subject
    subj = {}
    for r in rows:
        s = subj.setdefault(r["subject"], {"subject": r["subject"], "runs": 0, "runs_with_mov": 0, "runs_both_acc": 0,
                                           "checked": 0, "pass": 0, "void": 0, "not_checkable": 0,
                                           "hours_total": 0.0, "hours_both_acc": 0.0, "hours_pass": 0.0,
                                           "_ppm": [], "_off": []})
        s["runs"] += 1
        dur = r.get("eeg_duration_s") or r.get("mov_duration_s") or 0.0
        s["hours_total"] += dur / 3600.0
        if r.get("has_mov_edf"):
            s["runs_with_mov"] += 1
        if r.get("acc_eeg_module") and r.get("acc_ecgemg_module"):
            s["runs_both_acc"] += 1
            s["hours_both_acc"] += (r.get("mov_duration_s") or 0.0) / 3600.0
        v = r.get("verdict")
        if v in ("pass", "void"):
            s["checked"] += 1
            s[v] += 1
            s["_ppm"].append(r["drift_ppm"])
            s["_off"].append(r["offset_ms"])
            if v == "pass":
                s["hours_pass"] += (r.get("mov_duration_s") or 0.0) / 3600.0
        else:
            s["not_checkable"] += 1
    per = []
    for s in sorted(subj.values(), key=lambda x: x["subject"]):
        s["median_abs_drift_ppm"] = q([abs(x) for x in s["_ppm"]], 50)
        s["median_offset_ms"] = q(s["_off"], 50)
        per.append({k: v for k, v in s.items() if not k.startswith("_")})
    write_csv("per_subject.csv", per)

    # 3. trace pass
    trows, tfit = [], []
    if not OPT.no_trace and not OPT.limit:
        cands = {}
        for r in rows:
            if (r.get("acc_eeg_module") and r.get("acc_ecgemg_module") and (r.get("mov_duration_s") or 0) >= OPT.trace_min_s
                    and r.get("reason") != "unexpected_edf_layout"):
                best = cands.get(r["subject"])
                if best is None or (r["mov_duration_s"], r["key"]) > (best["mov_duration_s"], best["key"]):
                    cands[r["subject"]] = r
        tcache = cache_load("traces.jsonl")
        by_key = {run_key_str(k): k for k in keys}
        sel = sorted(cands.values(), key=lambda x: x["key"])
        print(f"trace pass: {len(sel)} runs (longest run per subject, at least {OPT.trace_min_s} s)", flush=True)
        trace_bytes = 0
        for j, r in enumerate(sel):
            if r["key"] in tcache:
                tw = tcache[r["key"]]["wins"]
            else:
                k = by_key[r["key"]]
                b0 = stats["bytes_read"]
                tw = [clean(x) for x in trace_run(k, files[k]["mov"], stats)]
                cache_append("traces.jsonl", {"key": r["key"], "wins": tw, "bytes_read": stats["bytes_read"] - b0})
                tcache[r["key"]] = {"bytes_read": stats["bytes_read"] - b0}
            trace_bytes += tcache[r["key"]].get("bytes_read", 0)
            trows.extend(tw)
            acc = [x for x in tw if x["status"] == "ok"]
            fit = {"key": r["key"], "subject": r["subject"], "duration_s": r["mov_duration_s"], "windows": len(tw),
                   "windows_ok": len(acc), "anchor_verdict": r.get("verdict")}
            if len(acc) >= 4:
                t = np.array([x["win_start_s"] + WIN_S / 2 for x in acc], dtype=float)
                lg = np.array([x["lag_ms"] for x in acc], dtype=float)
                sl, ic, _, _ = theilslopes(lg, t)
                res = lg - (ic + sl * t)
                fit.update(theilsen_ppm=sl * 1000.0, theilsen_lag0_ms=ic, resid_mad_ms=1.4826 * float(np.median(np.abs(res))),
                           frac_windows_resid_gt40=float(np.mean(np.abs(res) > THRESH_MS)))
                # local deviation: each accepted window against the median of its accepted neighbours within 30 min
                devs = []
                for i in range(len(t)):
                    nb = (np.abs(t - t[i]) <= 1800) & (np.arange(len(t)) != i)
                    if nb.sum() >= 2:
                        devs.append(lg[i] - float(np.median(lg[nb] - sl * (t[nb] - t[i]))))
                if devs:
                    devs = np.array(devs)
                    fit.update(local_windows=len(devs), local_dev_mad_ms=1.4826 * float(np.median(np.abs(devs))),
                               frac_windows_local_dev_gt40=float(np.mean(np.abs(devs) > THRESH_MS)))
                # level steps: median of detrended lags in the hour after minus the hour before, at least 3 each side
                steps = []
                for i in range(len(t)):
                    bef = (t < t[i]) & (t >= t[i] - 3600)
                    aft = (t >= t[i]) & (t < t[i] + 3600)
                    if bef.sum() >= 3 and aft.sum() >= 3:
                        steps.append(float(np.median(res[aft]) - np.median(res[bef])))
                if steps:
                    fit["max_abs_step_ms"] = float(np.max(np.abs(steps)))
                if r.get("verdict") in ("pass", "void"):
                    line = r["start_lag_ms"] + (t - r["start_centre_s"]) * r["drift_ppm"] / 1000.0
                    dev = lg - line
                    fit.update(anchor_ppm=r["drift_ppm"], max_abs_dev_from_anchor_line_ms=float(np.max(np.abs(dev))),
                               frac_windows_dev_gt40=float(np.mean(np.abs(dev) > THRESH_MS)),
                               frac_windows_abs_lag_gt40=float(np.mean(np.abs(lg) > THRESH_MS)))
            tfit.append(clean(fit))
            if (j + 1) % 10 == 0 or j == len(sel) - 1:
                print(f"  traced {j + 1}/{len(sel)}, {time.time() - t_start:.0f} s", flush=True)
        write_csv("traces.csv", trows, ["key", "subject", "win_start_s", "status", "lag_ms", "r_peak", "n_agree", "se_ms"])
        write_csv("trace_fit.csv", tfit)
        stats["trace_bytes"] = trace_bytes

    # 4. known-delay test of the sub-sample estimator on real accepted windows
    kd_rows = []
    if not OPT.limit:
        acc = [r for r in rows if r["verdict"] in ("pass", "void") and r.get("mode") == "full"]
        acc = sorted(acc, key=lambda x: x["key"])
        step = max(1, len(acc) // KD_WINDOWS)
        sel = acc[::step][:KD_WINDOWS]
        by_key = {run_key_str(k): k for k in keys}
        print(f"known-delay test on {len(sel)} start anchors, shifts {KD_SHIFTS} samples", flush=True)
        for r in sel:
            k = by_key[r["key"]]
            path = files[k]["mov"]
            hd = edf_header(path)
            einf = [chan_info(hd, l) for l in EEG_ACC]
            cinf = [chan_info(hd, l) for l in ECG_ACC]
            fs = einf[0]["n"] / hd["recdur"]
            r0 = int(round(r["start_win_start_s"] / hd["recdur"]))
            base = eval_window(path, hd, einf, cinf, r0, r0 + int(round(WIN_S / hd["recdur"])), int(LAG_S * fs), fs, stats, True)
            if base.get("status") != "ok" or "_a" not in base:
                continue
            for d in KD_SHIFTS:
                est = estimate(base["_a"], frac_delay(base["_b"], d), int(LAG_S * fs), fs)
                exp_ms = d / fs * 1000.0
                kd_rows.append(clean({"key": r["key"], "shift_samples": d, "expected_change_ms": exp_ms,
                                      "base_lag_ms": base["lag_ms"], "lag_ms": est.get("lag_ms"),
                                      "lag_sinc_ms": est.get("lag_sinc_ms"),
                                      "error_ms": (est["lag_ms"] - base["lag_ms"] - exp_ms) if est.get("lag_ms") is not None else None,
                                      "error_sinc_ms": (est["lag_sinc_ms"] - base["lag_sinc_ms"] - exp_ms) if est.get("lag_sinc_ms") is not None else None,
                                      "status": est.get("status")}))
        write_csv("known_delay_test.csv", kd_rows)

    # 4b. duplicated data (runs whose records are contained in another run)
    dups, n_fp = duplicate_scan(files, keys_main, stats)
    write_csv("contained_runs.csv", dups, ["run_a", "run_b", "records_a", "records_b", "fingerprint_match", "offset_records",
                                           "eeg_contained", "ecg_contained", "emg_contained", "mov_contained",
                                           "eeg_headers_equal", "ecg_headers_equal", "emg_headers_equal", "mov_headers_equal",
                                           "contained_all_modalities"])
    stats["mov_files_fingerprinted"] = n_fp
    print(f"duplicate scan: {n_fp} mov EDFs fingerprinted, {len(dups)} candidate pairs", flush=True)

    # 5. summary
    summary = build_summary(rows, wins, per, lay_rows, json_rows, tfit, stats, time.time() - t_start, kd_rows,
                            dups, range_rows)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    write_summary_md(summary)
    write_provenance(ARGS, {"task": "N0.3a", "gate": "G15 (seizeit2 part)", "raw": RAW,
                            "parameters": {"WIN_S": WIN_S, "LAG_S": LAG_S, "N_SUB": N_SUB, "AGREE_MS": AGREE_MS,
                                           "MIN_AGREE": MIN_AGREE, "MAX_ATTEMPTS": MAX_ATTEMPTS, "MIN_VALID": MIN_VALID,
                                           "SAT_MAX": SAT_MAX, "HP_HZ": HP_HZ, "PAD_S": PAD_S, "SHORT_GAP": SHORT_GAP,
                                           "THRESH_MS": THRESH_MS, "SHORT_RUN_S": SHORT_RUN_S, "MIN_HALF_S": MIN_HALF_S,
                                           "TRACE_MIN_S": OPT.trace_min_s, "KD_WINDOWS": KD_WINDOWS, "KD_SHIFTS": list(KD_SHIFTS)},
                            "limit": OPT.limit, "trace": not OPT.no_trace and not OPT.limit,
                            "never_read": ["events.tsv", "EDF Annotations signal (not decoded)", "processed/eeg_ingest_audit.csv"]})
    if not OPT.keep_cache and not OPT.limit:
        # the caches only serve an interrupted run; a complete run leaves the CSV outputs alone
        for n_ in ("runs.jsonl", "traces.jsonl"):
            if os.path.isfile(os.path.join(CACHE, n_)):
                os.remove(os.path.join(CACHE, n_))
        if not os.listdir(CACHE):
            os.rmdir(CACHE)
    print(f"done in {time.time() - t_start:.0f} s", flush=True)


def build_summary(rows, wins, per, lay_rows, json_rows, tfit, stats, elapsed, kd_rows=(), dups=(), range_rows=()):
    S = {}
    S["mov_files_fingerprinted"] = stats.get("mov_files_fingerprinted")
    S["contained_runs"] = list(dups)
    S["contained_runs_all_modalities"] = sum(1 for d in dups if d.get("contained_all_modalities"))
    S["acc_header_ranges_distinct"] = len(range_rows)
    S["acc_header_ranges_top"] = list(range_rows[:8])
    S["acc_channels_range_pm1"] = sum(x["n_channels"] for x in range_rows if float(x["physmin"]) == -1.0 and float(x["physmax"]) == 1.0)
    S["acc_channels_total"] = sum(x["n_channels"] for x in range_rows)
    if kd_rows:
        kd = {}
        for d in KD_SHIFTS:
            e = [x["error_ms"] for x in kd_rows if x["shift_samples"] == d and x.get("error_ms") is not None]
            es = [x["error_sinc_ms"] for x in kd_rows if x["shift_samples"] == d and x.get("error_sinc_ms") is not None]
            kd[str(d)] = {"windows": len(e), "median_error_ms": q(e, 50), "p90_abs_error_ms": q([abs(x) for x in e], 90),
                          "median_error_sinc_ms": q(es, 50), "p90_abs_error_sinc_ms": q([abs(x) for x in es], 90)}
        S["known_delay_test"] = kd
    S["runs"] = len(rows)
    S["subjects"] = len({r["subject"] for r in rows})
    S["runs_with_mov_edf"] = sum(1 for r in rows if r.get("has_mov_edf"))
    S["runs_both_acc"] = sum(1 for r in rows if r.get("acc_eeg_module") and r.get("acc_ecgemg_module"))
    S["runs_with_modality"] = {m: sum(1 for r in rows if r.get(f"has_{m}_edf")) for m in ("eeg", "ecg", "emg", "mov")}
    S["not_checkable_both_acc"] = sum(1 for r in rows if r.get("acc_eeg_module") and r.get("acc_ecgemg_module")
                                      and r["verdict"] == "not_checkable")
    S["runs_mov_eeg_duration_equal"] = sum(1 for r in rows if r.get("durations_equal"))
    S["mov_edf_types"] = {}
    for r in rows:
        if r.get("mov_edf_type"):
            S["mov_edf_types"][r["mov_edf_type"]] = S["mov_edf_types"].get(r["mov_edf_type"], 0) + 1
    ver = {}
    for r in rows:
        ver[r["verdict"]] = ver.get(r["verdict"], 0) + 1
    S["verdicts"] = ver
    S["checked"] = ver.get("pass", 0) + ver.get("void", 0)
    reasons = {}
    for r in rows:
        key = f"{r['verdict']}:{r.get('reason') or ''}"
        reasons[key] = reasons.get(key, 0) + 1
    S["verdict_reasons"] = dict(sorted(reasons.items()))
    chk = [r for r in rows if r["verdict"] in ("pass", "void")]
    S["checked_both_acc_share"] = S["checked"] / S["runs_both_acc"] if S["runs_both_acc"] else None
    S["literal_verdicts"] = {}
    for r in chk:
        S["literal_verdicts"][r["verdict_literal"]] = S["literal_verdicts"].get(r["verdict_literal"], 0) + 1
    S["strict_vs_literal_disagree"] = sum(1 for r in chk if r["verdict"] != r["verdict_literal"])
    S["robust"] = {v: sum(1 for r in chk if r["verdict"] == v and r["verdict_robust"]) for v in ("pass", "void")}
    S["anchors_primary"] = sum(1 for r in chk if r.get("anchors_primary"))
    S["primary_anchor_verdicts"] = {v: sum(1 for r in chk if r.get("anchors_primary") and r["verdict"] == v) for v in ("pass", "void")}
    side_status = {}
    for w in wins:
        side_status.setdefault((w["key"], w["side"]), []).append(w["status"])
    S["anchor_sides_failed"] = sum(1 for st_ in side_status.values() if "ok" not in st_)
    S["anchor_sides_failed_at_attempt_cap"] = sum(1 for st_ in side_status.values() if "ok" not in st_
                                                  and sum(1 for s_ in st_ if s_ != "gap") >= MAX_ATTEMPTS)
    S["mode_short_checked"] = sum(1 for r in chk if r.get("mode") == "short")
    S["hours"] = {"all_runs": sum((r.get("eeg_duration_s") or 0) for r in rows) / 3600.0,
                  "both_acc": sum((r.get("mov_duration_s") or 0) for r in rows if r.get("acc_eeg_module") and r.get("acc_ecgemg_module")) / 3600.0,
                  "checked": sum(r["mov_duration_s"] for r in chk) / 3600.0,
                  "pass": sum(r["mov_duration_s"] for r in chk if r["verdict"] == "pass") / 3600.0}
    S["subjects_with_any_pass"] = sum(1 for p in per if p["pass"] > 0)
    S["subjects_with_any_checked"] = sum(1 for p in per if p["checked"] > 0)
    S["subjects_all_checked_pass"] = sum(1 for p in per if p["checked"] > 0 and p["void"] == 0)
    for nm, fld in (("abs_start_lag_ms", "start_lag_ms"), ("abs_end_lag_ms", "end_lag_ms"), ("offset_ms", "offset_ms"),
                    ("abs_drift_run_ms", "drift_run_ms"), ("abs_drift_ppm", "drift_ppm"), ("drift_ppm", "drift_ppm"),
                    ("mov_duration_s", "mov_duration_s"), ("start_se_ms", "start_se_ms"), ("end_se_ms", "end_se_ms"),
                    ("start_r_peak", "start_r_peak"), ("start_sharpness", "start_sharpness")):
        vals = [r.get(fld) for r in chk]
        if nm.startswith("abs_"):
            vals = [abs(v) for v in vals if v is not None]
        S[f"q_{nm}"] = {p: q(vals, p) for p in (5, 25, 50, 75, 95)}
    med_ppm = S["q_abs_drift_ppm"][50]
    S["hours_to_threshold_at_median_abs_drift"] = (THRESH_MS / (med_ppm / 1000.0) / 3600.0) if med_ppm else None
    bins =[(0, 7200, "under 2 h"), (7200, 28800, "2 h to 8 h"), (28800, 1e9, "8 h and more")]
    S["by_duration"] = []
    for lo, hi, nm in bins:
        rr = [r for r in rows if r.get("acc_eeg_module") and r.get("acc_ecgemg_module") and lo <= (r.get("mov_duration_s") or 0) < hi]
        S["by_duration"].append({"bin": nm, "runs_both_acc": len(rr), "pass": sum(1 for r in rr if r["verdict"] == "pass"),
                                 "void": sum(1 for r in rr if r["verdict"] == "void"),
                                 "not_checkable": sum(1 for r in rr if r["verdict"] == "not_checkable"),
                                 "checked": sum(1 for r in rr if r["verdict"] in ("pass", "void")),
                                 "drift_gt40": sum(1 for r in rr if r["verdict"] in ("pass", "void") and abs(r["drift_run_ms"]) > THRESH_MS),
                                 "offset_gt40": sum(1 for r in rr if r["verdict"] in ("pass", "void") and r["offset_ms"] > THRESH_MS),
                                 "median_abs_drift_ppm": q([abs(r["drift_ppm"]) for r in rr if r["verdict"] in ("pass", "void")], 50)})
    S["checked_start_lag_within_40"] = sum(1 for r in chk if abs(r["start_lag_ms"]) <= THRESH_MS)
    S["checked_abs_drift_ppm_gt_10"] = sum(1 for r in chk if abs(r["drift_ppm"]) > 10)
    # sub-sample checks on accepted windows
    ok = [w for w in wins if w.get("status") == "ok"]
    fr = np.array([w["frac_samples"] for w in ok if w.get("frac_samples") is not None])
    S["accepted_windows"] = len(ok)
    S["windows_tried"] = len(wins)
    st = {}
    for w in wins:
        st[w["status"]] = st.get(w["status"], 0) + 1
    S["window_status"] = st
    if len(fr):
        hist, edges = np.histogram(fr, bins=10, range=(-0.5, 0.5))
        S["parabolic_fraction_hist"] = {"edges": [round(float(e), 2) for e in edges], "counts": [int(h) for h in hist]}
        d = np.array([abs(w["lag_ms"] - w["lag_sinc_ms"]) for w in ok])
        S["parabolic_minus_sinc_abs_ms"] = {p: float(np.percentile(d, p)) for p in (50, 90, 99)}
        vc = [r for r in chk]
        S["verdict_change_with_sinc_lags"] = sum(1 for r in vc if sinc_verdict(r) != r["verdict"])
    # runs of one subject whose anchor windows give bit-identical lags (identical data in two runs)
    dup = []
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    for sj, rs in by_subj.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                for side_i in ("start", "end"):
                    for side_j in ("start", "end"):
                        a, b = rs[i].get(f"{side_i}_lag_ms"), rs[j].get(f"{side_j}_lag_ms")
                        if a is not None and b is not None and a == b:
                            dup.append(f"{rs[i]['key']}:{side_i}={rs[j]['key']}:{side_j}")
    S["identical_anchor_lag_pairs"] = dup
    # consecutive runs of a subject: start lag of run n+1 minus extrapolated end lag of run n
    cons = []
    for sj, rs in by_subj.items():
        rs = sorted(rs, key=lambda x: (x["session"], int(x["run"]) if str(x["run"]).isdigit() else x["run"]))
        for a, b in zip(rs, rs[1:]):
            if a["verdict"] in ("pass", "void") and b["verdict"] in ("pass", "void") and a["session"] == b["session"]:
                cons.append({"subject": sj, "run_a": a["key"], "run_b": b["key"],
                             "end_lag_a_ms": a["lag_run_end_ms"], "start_lag_b_ms": b["lag_run_start_ms"],
                             "jump_ms": b["lag_run_start_ms"] - a["lag_run_end_ms"],
                             "abs_start_lag_b_ms": abs(b["lag_run_start_ms"])})
    write_csv("consecutive_runs.csv", cons, ["subject", "run_a", "run_b", "end_lag_a_ms", "start_lag_b_ms", "jump_ms", "abs_start_lag_b_ms"])
    S["consecutive_pairs"] = len(cons)
    S["consecutive_abs_jump_le_100ms"] = sum(1 for c in cons if abs(c["jump_ms"]) <= 100)
    S["consecutive_abs_start_b_le_40ms"] = sum(1 for c in cons if c["abs_start_lag_b_ms"] <= THRESH_MS)
    S["q_consecutive_abs_jump_ms"] = {p: q([abs(c["jump_ms"]) for c in cons], p) for p in (25, 50, 75)}
    S["acc_labels_outside_mov"] = sum(x["n_files"] for x in lay_rows if x["modality"] != "mov" and "ACC" in x["labels"])
    S["mov_files_with_both_triplets"] = sum(x["n_files"] for x in lay_rows if x["modality"] == "mov"
                                            and all(l in x["labels"].split("; ") for l in EEG_ACC + ECG_ACC))
    okw = [w for w in wins if w.get("status") == "ok"]
    S["accepted_windows_axes_lt3"] = {"eeg": sum(1 for w in okw if (w.get("axes_eeg") or 0) < 3),
                                      "ecgemg": sum(1 for w in okw if (w.get("axes_ecgemg") or 0) < 3)}
    allw = [w for w in wins if w.get("raw_norm_eeg") is not None and w.get("status") != "gap"]
    S["q_raw_norm_eeg"] = {p: q([w["raw_norm_eeg"] for w in allw], p) for p in (5, 25, 50, 75, 95)}
    S["q_raw_norm_ecgemg"] = {p: q([w["raw_norm_ecgemg"] for w in allw if w.get("raw_norm_ecgemg") is not None], p) for p in (5, 25, 50, 75, 95)}
    S["windows_raw_norm"] = len(allw)

    def over(w, thr):
        return any((w.get(f) or 0) > thr for f in ("raw_norm_eeg", "raw_norm_ecgemg"))
    S["sentinel_windows"] = {"tried_norm_gt10": sum(1 for w in allw if over(w, SENTINEL_NORM)),
                             "accepted_norm_gt10": sum(1 for w in okw if over(w, SENTINEL_NORM)),
                             "tried_norm_gt40": sum(1 for w in allw if over(w, SENTINEL2_NORM)),
                             "accepted_norm_gt40": sum(1 for w in okw if over(w, SENTINEL2_NORM))}
    S["io"] = {"main_windows_read": sum((r.get("windows_read") or 0) for r in rows),
               "main_bytes_read_gb": sum((r.get("bytes_read") or 0) for r in rows) / 1e9,
               "trace_bytes_read_gb": stats.get("trace_bytes", 0) / 1e9,
               "duplicate_scan_bytes_read_gb": stats.get("dup_bytes", 0) / 1e9,
               "mov_bytes_total_gb": sum((r.get("mov_bytes") or 0) for r in rows) / 1e9,
               "elapsed_s_this_invocation": elapsed}
    # sensitivity: anchors re-selected from the logged windows with a looser agreement rule
    S["sensitivity_min_agree"] = {}
    for m in SENS_MIN_AGREE:
        S["sensitivity_min_agree"][str(m)] = sensitivity(rows, wins, m)
    S["per_subject"] = per
    S["layout"] = lay_rows
    S["mov_sidecar_facts"] = json_rows
    if tfit:
        f_ok = [f for f in tfit if f.get("resid_mad_ms") is not None]
        S["trace"] = {"runs_traced": len(tfit), "runs_with_4plus_windows": len(f_ok),
                      "q_resid_mad_ms": {p: q([f["resid_mad_ms"] for f in f_ok], p) for p in (25, 50, 75)},
                      "q_frac_windows_dev_gt40": {p: q([f.get("frac_windows_dev_gt40") for f in f_ok if f.get("frac_windows_dev_gt40") is not None], p) for p in (25, 50, 75, 95)},
                      "anchor_pass_runs_traced": sum(1 for f in f_ok if f.get("anchor_verdict") == "pass"),
                      "anchor_pass_runs_with_any_window_abs_lag_gt40": sum(1 for f in f_ok if f.get("anchor_verdict") == "pass" and (f.get("frac_windows_abs_lag_gt40") or 0) > 0),
                      "q_abs_ppm_diff_theilsen_vs_anchor": {p: q([abs(f["theilsen_ppm"] - f["anchor_ppm"]) for f in f_ok if f.get("anchor_ppm") is not None], p) for p in (50, 90)},
                      "q_local_dev_mad_ms": {p: q([f.get("local_dev_mad_ms") for f in f_ok if f.get("local_dev_mad_ms") is not None], p) for p in (25, 50, 75)},
                      "q_frac_windows_local_dev_gt40": {p: q([f.get("frac_windows_local_dev_gt40") for f in f_ok if f.get("frac_windows_local_dev_gt40") is not None], p) for p in (25, 50, 75, 95)},
                      "pooled_local": pooled_local(tfit),
                      "runs_with_step_data": sum(1 for f in f_ok if f.get("max_abs_step_ms") is not None),
                      "runs_max_step_gt100": sum(1 for f in f_ok if (f.get("max_abs_step_ms") or 0) > 100),
                      "q_max_abs_step_ms": {p: q([f.get("max_abs_step_ms") for f in f_ok if f.get("max_abs_step_ms") is not None], p) for p in (25, 50, 75)},
                      "windows_traced": sum(f["windows"] for f in tfit), "windows_traced_ok": sum(f["windows_ok"] for f in tfit)}
    return S


def pooled_local(tfit):
    """Pooled over traced runs: accepted windows with a local reference, and those deviating by more than 40 ms."""
    n = sum(f.get("local_windows") or 0 for f in tfit)
    k = sum(round((f.get("local_windows") or 0) * (f.get("frac_windows_local_dev_gt40") or 0)) for f in tfit)
    return {"windows": n, "dev_gt40": k, "frac": (k / n) if n else None}


def sensitivity(rows, wins, min_agree):
    """Verdict counts when a window is accepted with at least min_agree agreeing sub-windows (min_agree below
    MIN_AGREE): every window such a rule could pick was tried and logged by the main pass."""
    by = {}
    for w in wins:
        by.setdefault((w["key"], w["side"]), []).append(w)
    cnt = {"pass": 0, "void": 0, "not_checkable": 0}
    for r in rows:
        if r.get("mode") is None or r.get("reason") == "run_too_short":
            cnt["not_checkable"] += 1
            continue
        anc = {}
        for side in ("start", "end"):
            for w in sorted(by.get((r["key"], side), []), key=lambda x: x["attempt"]):
                if w["status"] in ("ok", "no_agreement") and (w.get("n_agree") or 0) >= min_agree and (w.get("r_peak") or 0) > 0:
                    anc[side] = w
                    break
        if len(anc) < 2:
            cnt["not_checkable"] += 1
            continue
        s_, e_ = anc["start"], anc["end"]
        v = verdict_from(s_["lag_ms"], e_["lag_ms"], s_["se_ms"], e_["se_ms"], s_["win_start_s"] + s_["win_len_s"] / 2.0,
                         e_["win_start_s"] + e_["win_len_s"] / 2.0, r["mov_duration_s"])
        cnt[v["verdict"]] += 1
    return cnt


def sinc_verdict(r):
    ts, te, T = r["start_centre_s"], r["end_centre_s"], r["mov_duration_s"]
    ls, le = r["start_lag_sinc_ms"], r["end_lag_sinc_ms"]
    slope = (le - ls) / (te - ts)
    lag0, lagT = ls - slope * ts, ls + slope * (T - ts)
    return "void" if max(abs(lag0), abs(lagT)) > THRESH_MS or abs(slope * T) > THRESH_MS else "pass"


def r1(x, nd=1):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def write_summary_md(S):
    v = S["verdicts"]
    L = []
    L.append("# seizeit2 cross-module synchronisation (G15, task N0.3a)")
    L.append("")
    L.append("Generated by scripts/probe_sync_seizeit2.py; every number below is read from summary.json and runs.csv in this directory. MoveAtlas commit in _provenance.json.")
    L.append("")
    L.append("## Where the accelerometers are")
    L.append("")
    L.append(f"The accelerometers of both modules sit in one file per run, the BIDS mov EDF (sub-*/ses-01/mov/*_mov.edf): {S['mov_files_with_both_triplets']} mov EDFs carry both triplets. eeg, ecg and emg EDFs with an accelerometer label: {S['acc_labels_outside_mov']}. Channel sets found (channel_layout.csv):")
    L.append("")
    L.append("| modality | EDF type | files | channels | fs (Hz) |")
    L.append("|---|---|---|---|---|")
    for x in S["layout"]:
        L.append(f"| {x['modality']} | {x['edf_type']} | {x['n_files']} | {x['labels']} | {x['fs_hz']} |")
    L.append("")
    L.append("EEG-module accelerometer: EEG SD ACC X, Y, Z. ECG-EMG-module accelerometer: ECGEMG SD ACC X, Y, Z. Both sit at the same rate in the same EDF records, so a misalignment between them comes from the two device clocks and from how the two recordings were merged into one file. The mov sidecars give the placement as " + "; ".join(f"\"{x['value']}\" ({x['n_files']} files)" for x in S["mov_sidecar_facts"] if x["field"] == "MOVPlacementScheme") + " (mov_sidecar_facts.csv); the dataset README places the EEG module on the upper back.")
    L.append("")
    med_n = S["q_raw_norm_eeg"][50]
    unit_word = "gives about 1 g" if (med_n is not None and 0.8 <= med_n <= 1.2) else "does not give 1 g"
    L.append(f"Units: the header scaling {unit_word} at rest. Median raw norm of the three axes per window (all non-gap windows tried, {S['windows_raw_norm']}): EEG module p5 {r1(S['q_raw_norm_eeg'][5], 3)}, p50 {r1(S['q_raw_norm_eeg'][50], 3)}, p95 {r1(S['q_raw_norm_eeg'][95], 3)}; ECG-EMG module p5 {r1(S['q_raw_norm_ecgemg'][5], 3)}, p50 {r1(S['q_raw_norm_ecgemg'][50], 3)}, p95 {r1(S['q_raw_norm_ecgemg'][95], 3)} (header unit \"g\"). The check here needs no calibration (per-axis high-pass, then norm), but any fixed-threshold formula on these channels does. Accepted windows that dropped a saturated axis: EEG module {S['accepted_windows_axes_lt3']['eeg']}, ECG-EMG module {S['accepted_windows_axes_lt3']['ecgemg']}.")
    L.append("")
    sw = S["sentinel_windows"]
    L.append(f"Header ranges: the physical range of the accelerometer channels changes from file to file and from axis to axis ({S['acc_header_ranges_distinct']} distinct label and range combinations over {S['acc_channels_total']} channels; -1 to 1 in {S['acc_channels_range_pm1']}; acc_header_ranges.csv), so a digital extreme marks the end of a file's range rather than a clipped sample. Some axes sit at a value near +-32.77: windows (not gap) with a module's median raw norm above {SENTINEL_NORM:g} (one axis at that value for most of the window): {sw['tried_norm_gt10']} tried, {sw['accepted_norm_gt10']} accepted; above {SENTINEL2_NORM:g} (two or more such axes, which the digital-extreme test can miss when the header range is wider than the value): {sw['tried_norm_gt40']} tried, {sw['accepted_norm_gt40']} accepted.")
    L.append("")
    L.append("## Counts")
    L.append("")
    L.append(f"- Runs (BIDS run keys over all modalities): {S['runs']}, subjects: {S['subjects']}.")
    L.append(f"- Runs with a mov EDF: {S['runs_with_mov_edf']}; with both accelerometer triplets: {S['runs_both_acc']}; mov and eeg EDF of equal duration: {S['runs_mov_eeg_duration_equal']}.")
    rm = S["runs_with_modality"]
    L.append(f"- Runs with an eeg EDF: {rm['eeg']}; ecg: {rm['ecg']}; emg: {rm['emg']}; mov: {rm['mov']} (each modality is its own EDF).")
    L.append(f"- Checked (both anchors accepted): {S['checked']}; pass: {v.get('pass', 0)}; void: {v.get('void', 0)}; not checkable: {v.get('not_checkable', 0)}, of which {S['not_checkable_both_acc']} among the {S['runs_both_acc']} runs with both accelerometer triplets and {v.get('not_checkable', 0) - S['not_checkable_both_acc']} without them.")
    L.append(f"- Anchor sides without an accepted window (runs long enough to check): {S['anchor_sides_failed']}; of these, {S['anchor_sides_failed_at_attempt_cap']} stopped at the cap of {MAX_ATTEMPTS} windows with enough valid data, with candidate windows closer to the run midpoint left untried.")
    L.append(f"- Robust verdicts (unchanged at 2 standard errors toward the threshold): pass {S['robust']['pass']}, void {S['robust']['void']}.")
    L.append(f"- Literal reading of G15 (offset = start-window lag, drift = end minus start window lag): {json.dumps(S['literal_verdicts'])}; runs where it differs from the strict verdict: {S['strict_vs_literal_disagree']}.")
    L.append(f"- Checked runs whose two anchors are the first and last 10 min exactly: {S['anchors_primary']} (pass {S['primary_anchor_verdicts']['pass']}, void {S['primary_anchor_verdicts']['void']}); checked in short mode (run under 1200 s): {S['mode_short_checked']}.")
    L.append(f"- Hours: all runs {r1(S['hours']['all_runs'])}, both accelerometers {r1(S['hours']['both_acc'])}, checked {r1(S['hours']['checked'])}, pass {r1(S['hours']['pass'])}.")
    L.append(f"- Subjects with at least one checked run: {S['subjects_with_any_checked']}; with at least one passing run: {S['subjects_with_any_pass']}; with every checked run passing: {S['subjects_all_checked_pass']}.")
    for m, c in S["sensitivity_min_agree"].items():
        L.append(f"- Sensitivity, acceptance with at least {m} of {N_SUB} agreeing sub-windows instead of {MIN_AGREE} (anchors re-selected from the logged windows): pass {c['pass']}, void {c['void']}, not checkable {c['not_checkable']}.")
    L.append("")
    L.append("Verdict and reason counts:")
    L.append("")
    L.append("| verdict:reason | runs |")
    L.append("|---|---|")
    for k, n in S["verdict_reasons"].items():
        L.append(f"| {k} | {n} |")
    L.append("")
    L.append("## Lags and drift on checked runs")
    L.append("")
    L.append("| quantity | p5 | p25 | p50 | p75 | p95 |")
    L.append("|---|---|---|---|---|---|")
    for nm in ("abs_start_lag_ms", "abs_end_lag_ms", "offset_ms", "abs_drift_run_ms", "abs_drift_ppm", "drift_ppm",
               "start_se_ms", "end_se_ms", "start_r_peak", "start_sharpness", "mov_duration_s"):
        qq = S[f"q_{nm}"]
        L.append(f"| {nm} | " + " | ".join(r1(qq[p], 2 if 'r_peak' in nm or 'sharp' in nm else 1) for p in (5, 25, 50, 75, 95)) + " |")
    L.append("")
    L.append(f"Checked runs with the start-window lag within 40 ms: {S['checked_start_lag_within_40']} of {S['checked']}; with an absolute drift above 10 ppm: {S['checked_abs_drift_ppm_gt_10']}. At the median absolute drift of the checked runs ({r1(S['q_abs_drift_ppm'][50])} ppm) the lag changes by {THRESH_MS:g} ms in {r1(S['hours_to_threshold_at_median_abs_drift'], 2)} h.")
    L.append("")
    L.append("By run duration (runs with both accelerometers; the drift and offset columns count checked runs whose strict drift or offset exceeds 40 ms, alone or together):")
    L.append("")
    L.append("| duration | runs | pass | void | not checkable | checked | drift over 40 ms | offset over 40 ms | median abs drift (ppm, checked) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in S["by_duration"]:
        L.append(f"| {x['bin']} | {x['runs_both_acc']} | {x['pass']} | {x['void']} | {x['not_checkable']} | {x['checked']} | {x['drift_gt40']} | {x['offset_gt40']} | {r1(x['median_abs_drift_ppm'])} |")
    L.append("")
    L.append("## Sub-sample estimate")
    L.append("")
    if "parabolic_fraction_hist" in S:
        h = S["parabolic_fraction_hist"]
        L.append(f"Accepted windows: {S['accepted_windows']} of {S['windows_tried']} tried (status counts {json.dumps(S['window_status'])}). Parabolic sub-sample fraction histogram, 10 bins from -0.5 to 0.5 sample: {h['counts']}. Absolute difference between the parabolic and the windowed-sinc peak (ms): median {r1(S['parabolic_minus_sinc_abs_ms'][50], 2)}, p90 {r1(S['parabolic_minus_sinc_abs_ms'][90], 2)}, p99 {r1(S['parabolic_minus_sinc_abs_ms'][99], 2)}. Checked runs whose strict verdict changes with the sinc lags: {S['verdict_change_with_sinc_lags']}.")
    L.append("")
    if "trace" in S:
        t = S["trace"]
        L.append("## Trace pass (longest run of each subject, whole file)")
        L.append("")
        L.append(f"Runs traced: {t['runs_traced']}; with at least 4 accepted windows: {t['runs_with_4plus_windows']}. Residual scatter of the accepted 10 min window lags around a Theil-Sen line, robust SD (ms): p25 {r1(t['q_resid_mad_ms'][25])}, p50 {r1(t['q_resid_mad_ms'][50])}, p75 {r1(t['q_resid_mad_ms'][75])}. Fraction of accepted windows farther than 40 ms from the two-anchor line, per run: p50 {r1(t['q_frac_windows_dev_gt40'][50], 3)}, p75 {r1(t['q_frac_windows_dev_gt40'][75], 3)}, p95 {r1(t['q_frac_windows_dev_gt40'][95], 3)}. Absolute difference between the Theil-Sen drift and the two-anchor drift (ppm): p50 {r1(t['q_abs_ppm_diff_theilsen_vs_anchor'][50], 2)}, p90 {r1(t['q_abs_ppm_diff_theilsen_vs_anchor'][90], 2)}. Traced runs with an anchor pass: {t['anchor_pass_runs_traced']}, of which with at least one accepted window whose absolute lag exceeds 40 ms: {t['anchor_pass_runs_with_any_window_abs_lag_gt40']}.")
        L.append("")
        pl = t["pooled_local"]
        L.append(f"Error of a single accepted window, from its accepted neighbours within 30 min (drift removed): robust SD per run p25 {r1(t['q_local_dev_mad_ms'][25])}, p50 {r1(t['q_local_dev_mad_ms'][50])}, p75 {r1(t['q_local_dev_mad_ms'][75])} ms; pooled, {pl['dev_gt40']} of {pl['windows']} accepted windows ({r1(100 * pl['frac'] if pl['frac'] is not None else None)} percent) deviate by more than 40 ms from their neighbours. This is the rate at which one anchor alone can flip a verdict. The median anchor standard error (start {r1(S['q_start_se_ms'][50])} ms, end {r1(S['q_end_se_ms'][50])} ms) is well below this per-run robust SD (p50 {r1(t['q_local_dev_mad_ms'][50])} ms): the sub-window standard error understates the error of an anchor, so the robust verdict counts above are optimistic.")
        L.append("")
        L.append(f"Level steps (median of the detrended lags in the hour after a window minus the hour before, at least 3 accepted windows each side): runs with step data {t['runs_with_step_data']}, runs whose largest step exceeds 100 ms {t['runs_max_step_gt100']}; largest step per run p25 {r1(t['q_max_abs_step_ms'][25])}, p50 {r1(t['q_max_abs_step_ms'][50])}, p75 {r1(t['q_max_abs_step_ms'][75])} ms. Traced windows: {t['windows_traced']}, accepted {t['windows_traced_ok']}.")
        L.append("")
    if "known_delay_test" in S:
        L.append("## Known-delay test of the estimator")
        L.append("")
        L.append("Accepted start windows (evenly spaced over the checked runs in key order) were re-estimated after delaying the ECG-EMG-module magnitude by a known fraction of a sample (band-limited delay); error = measured lag change minus applied delay (known_delay_test.csv).")
        L.append("")
        L.append("| delay (samples) | windows | median error, parabolic (ms) | p90 abs error, parabolic (ms) | median error, sinc (ms) | p90 abs error, sinc (ms) |")
        L.append("|---|---|---|---|---|---|")
        for d, x in S["known_delay_test"].items():
            L.append(f"| {d} | {x['windows']} | {r1(x['median_error_ms'], 2)} | {r1(x['p90_abs_error_ms'], 2)} | {r1(x['median_error_sinc_ms'], 2)} | {r1(x['p90_abs_error_sinc_ms'], 2)} |")
        L.append("")
        L.append("The delay is circular and band-limited, so a whole-sample delay moves the correlation by exactly one lag: its zero error checks the sign and the indexing, not the sub-sample interpolation.")
        L.append("")
    L.append("## Runs as pieces of longer recordings")
    L.append("")
    L.append(f"Consecutive run pairs of one subject with both runs checked: {S['consecutive_pairs']}. Pairs where the next run starts within 100 ms of the lag extrapolated to the end of the previous run: {S['consecutive_abs_jump_le_100ms']}; pairs where the next run starts with an absolute lag within 40 ms: {S['consecutive_abs_start_b_le_40ms']}. Absolute jump (ms): p25 {r1(S['q_consecutive_abs_jump_ms'][25])}, p50 {r1(S['q_consecutive_abs_jump_ms'][50])}, p75 {r1(S['q_consecutive_abs_jump_ms'][75])} (consecutive_runs.csv). The gap between two runs is not known (the EDF start dates are anonymized), so a small jump is consistent with, not proof of, one continuous device recording cut into runs.")
    L.append("")
    L.append(f"Anchor windows with bit-identical lags in two runs of one subject (identical data in two runs): {len(S['identical_anchor_lag_pairs'])} pairs: {'; '.join(S['identical_anchor_lag_pairs']) if S['identical_anchor_lag_pairs'] else 'none'}.")
    L.append("")
    cr = S["contained_runs"]
    desc = "; ".join(f"{d['run_a']} ({d['records_a']} records) equals records {d['offset_records'] or 'none'} onward of {d['run_b']} ({d['records_b']} records), "
                     f"contained in {', '.join(m for m in ('eeg', 'ecg', 'emg', 'mov') if d.get(f'{m}_contained'))}"
                     for d in cr) or "none"
    L.append(f"Record-level duplicate scan ({S['mov_files_fingerprinted']} mov EDFs, first and last {DUP_REC} data records fingerprinted, then every modality compared record by record, annotation signal left out; contained_runs.csv): {len(cr)} candidate pairs, {S['contained_runs_all_modalities']} contained in every modality: {desc}. Both runs of such a pair are counted in every total above.")
    L.append("")
    L.append("## Reading")
    L.append("")
    L.append("- Sign: a positive lag means a movement appears later in the ECG-EMG-module channels than in the EEG-module channels of the same mov EDF.")
    L.append("- The verdict fires on the point estimate; the standard error comes from the spread of the agreeing 60 s sub-window lags and does not include a systematic lag between the two body sites (upper back and left chest), which this design cannot separate from a clock offset.")
    L.append("- Not checkable means no accepted window on one side or both (quiet windows, gaps, a module absent or zero-filled, saturated axes, or a lag outside +-60 s). Under G15 a run that is not checked is not usable across modules.")
    L.append(f"- I/O of the main pass: {S['io']['main_windows_read']} windows read, {r1(S['io']['main_bytes_read_gb'], 2)} GB of {r1(S['io']['mov_bytes_total_gb'], 2)} GB of mov EDF (windows with 20 s padding, read as whole EDF records and sliced to the six accelerometer channels); trace pass: {r1(S['io']['trace_bytes_read_gb'], 2)} GB; duplicate scan: {r1(S['io']['duplicate_scan_bytes_read_gb'], 2)} GB.")
    L.append("- Never read: events.tsv (seizure annotations) and processed/eeg_ingest_audit.csv are not opened; the EDF Annotations signal is loaded as part of whole EDF records and dropped, never decoded.")
    L.append("")
    L.append("## Per subject (per_subject.csv)")
    L.append("")
    L.append("| subject | runs | runs with both accelerometers | checked | pass | void | not checkable | hours pass | median abs drift (ppm) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for x in S["per_subject"]:
        L.append(f"| {x['subject']} | {x['runs']} | {x['runs_both_acc']} | {x['checked']} | {x['pass']} | {x['void']} | {x['not_checkable']} | {r1(x['hours_pass'])} | {r1(x['median_abs_drift_ppm'])} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
