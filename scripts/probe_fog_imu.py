"""fog_turning_filho IMU pre-checks of the corrected gate G14 (plan task T0.8, IMU part).

Reads READ-ONLY, from the MoveAtlas tree:
- datasets/pose/fog_turning_filho/raw/IMU.zip: the SUBnn_k.txt turning trials (tab-separated,
  128 Hz, one Physilog 5 on the shank of the more affected side, ACC ML/AP/SI [g],
  GYR ML/AP/SI [deg/s], Freezing event [flag]) and the SUBnn_standing.txt files (no flag);
- datasets/pose/fog_turning_filho/raw/PDFEinfo.csv (semicolon-separated, latin-1);
- datasets/imu/daphnet/raw/daphnet.zip, member dataset_fog_release/scripts/do_test.m ONLY,
  to read the freeze-index thresholds TH.freeze and TH.power of the Daphnet reference
  implementation (fixed on another data set; no Daphnet signal is read).
No video is opened or decoded. Writes only under reports/fog_imu/ (or --out).

The turn-segmentation rule is the block between the TURN RULE markers below. At start the
script extracts that block from its own source, hashes it (SHA-256 of the UTF-8 text with LF
line ends), writes turn_rule.py.txt, turn_rule.sha256 and turn_rule_freeze.json when they do
not exist yet, and refuses to run when they exist with another hash. The functions that run
are compiled from the extracted text, so what is hashed is what runs. Run once with
--freeze-rule-only to freeze the rule before any turn statistic is computed.

Outputs (reports/fog_imu/):
- turn_rule.py.txt, turn_rule.sha256, turn_rule_freeze.json: the frozen rule;
- records.csv: one row per IMU record (sample grid, columns, flag values, standing means);
- flag_runs.csv: one row per flagged run, matched to the PDFEinfo interval list;
- flag_structure.csv: one row per turning trial (runs, shortest run and gap, first/last flag);
- subjects.csv, subjects_freezing.csv: per subject, and per session plus overall counts of
  subjects with at least one flagged episode (flags and PDFEinfo);
- fi_windows.csv: one row per 2 s window (turning and standing records) with the freeze index;
- fi_auroc.csv, fi_auroc_per_subject.csv, fi_thresholds.csv: threshold-free AUROC with
  subject-cluster bootstrap intervals, and alarms at the Daphnet reference thresholds;
- sensor_frame.csv: per turning trial, the gyro SD per axis and the turns the frozen rule finds
  when ML or AP plays the heading axis (axis check, computed after the turn statistics);
- turns.csv, sweeps.csv, turn_trials.csv: the output of the frozen turn rule;
- fog_imu_summary.md, _provenance.json.
"""
import csv
import datetime as _dt
import hashlib
import io
import json
import os
import re
import sys
import zipfile

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

FREEZE_ONLY = "--freeze-rule-only" in sys.argv
if FREEZE_ONLY:
    sys.argv.remove("--freeze-rule-only")
if "--out" not in sys.argv:
    sys.argv += ["--out", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "fog_imu")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_args, write_provenance  # noqa: E402

args = parse_args("fog_turning_filho IMU pre-checks of G14 (T0.8)")
OUT = args.out
RAW = f"{args.moveatlas}/datasets/pose/fog_turning_filho/raw"
DAPHNET_ZIP = f"{args.moveatlas}/datasets/imu/daphnet/raw/daphnet.zip"
DAPHNET_TEST = "dataset_fog_release/scripts/do_test.m"

FS = 128.0                 # nominal rate, verified on every record below
SCORE_WIN = 256            # 2 s scoring window at 128 Hz
FI_WIN = 512               # 4 s freeze-index analysis window centred on the scoring window
SEED = 1414
N_BOOT = 2000


def log(*a):
    print(*a, flush=True)


# >>> TURN RULE BEGIN (frozen: extracted, hashed and written to turn_rule.py.txt before any turn statistic)
import numpy as _np

TURN_RULE_PARAMS = {
    "still_window_s": 1.0,           # bias: non-overlapping windows of this length
    "still_sd_max_dps": 2.0,         # a window is still when the SD of every gyro axis is below this
    "still_min_windows": 2,          # at least this many still windows, else the fallback bias
    "reversal_hysteresis_deg": 90.0,  # a heading extreme is confirmed once heading moves back this far
    "turn_nominal_deg": 360.0,       # one turn
    "turn_min_sweep_deg": 270.0,     # a sweep shorter than this is a partial sweep, not a turn
    "plateau_tol_deg": 10.0,         # onset/offset: last/first sample within this of the start/end heading
    "commit_deg": 45.0,              # ... searched before/after the heading commits this far into the sweep
    "edge_s": 0.5,                   # a first (last) turn starting (ending) this close to the record edge is truncated
}


def estimate_gyro_bias(gyr, fs, p=TURN_RULE_PARAMS):
    """Bias of GYR SI in deg/s: median over still windows of the window mean of GYR SI.

    gyr: array (n, 3) with columns GYR ML, GYR AP, GYR SI in deg/s. A window is still when the
    standard deviation of each of the three axes is below still_sd_max_dps. Returns
    (bias or None, number of still windows); None when fewer than still_min_windows are still.
    """
    gyr = _np.asarray(gyr, dtype=float)
    w = int(round(p["still_window_s"] * fs))
    n = gyr.shape[0] // w
    if n == 0:
        return None, 0
    seg = gyr[: n * w].reshape(n, w, 3)
    still = (seg.std(axis=1) < p["still_sd_max_dps"]).all(axis=1)
    k = int(still.sum())
    if k < p["still_min_windows"]:
        return None, k
    return float(_np.median(seg[still, :, 2].mean(axis=1))), k


def segment_turns(gyr, fs, fallback_bias=None, p=TURN_RULE_PARAMS):
    """Fixed turn segmentation from the shank gyroscope of one record.

    1. Bias: estimate_gyro_bias on the record; when it returns None, fallback_bias (the caller
       passes the bias that estimate_gyro_bias gives on the same subject's standing file), else 0.
    2. Heading: trapezoidal time integral of (GYR SI - bias), in degrees, 0 at the first sample.
       GYR SI is the rotation about the shank long axis, vertical in stance.
    3. Sweeps: zigzag on the heading; an extreme is confirmed when the heading has moved
       reversal_hysteresis_deg back from it. A sweep runs from one confirmed extreme to the next;
       the last sweep ends at the running extreme at the end of the record (provisional end).
       The fragment before the first confirmed extreme is ignored (smaller than the hysteresis).
    4. Turns: a sweep of amplitude A >= turn_min_sweep_deg holds n = max(1, floor(A/360 + 0.5))
       turns in its direction. Onset: the last sample within plateau_tol_deg of the start heading
       before the heading first reaches commit_deg into the sweep. Offset: the first sample within
       plateau_tol_deg of the end heading after the heading last lies commit_deg or more before the
       end. With n > 1 the sweep is cut where the heading first reaches m*A/n (m = 1..n-1).
    5. Direction: +1 for a positive rotation about the SI axis, -1 for a negative one.
       Consecutive sweeps alternate by construction; consecutive turns alternate only when no
       sweep holds two turns and no partial sweep separates two turns of the same direction.
    6. Truncation: the first turn of the first sweep is truncated when its onset lies within
       edge_s of the record start; the last turn of the last sweep is truncated when its offset
       lies within edge_s of the record end.
    Indices are 0-based sample indices; times are index / fs.
    """
    gyr = _np.asarray(gyr, dtype=float)
    n_samp = gyr.shape[0]
    bias, n_still = estimate_gyro_bias(gyr, fs, p)
    src = "record_still_windows"
    if bias is None:
        if fallback_bias is not None:
            bias, src = float(fallback_bias), "fallback"
        else:
            bias, src = 0.0, "none"
    out = {"bias_dps": bias, "bias_source": src, "n_still_windows": n_still, "n_samples": n_samp,
           "sweeps": [], "turns": [], "params": dict(p)}
    if n_samp < 2:
        return out
    rate = gyr[:, 2] - bias
    h = _np.concatenate([[0.0], _np.cumsum((rate[1:] + rate[:-1]) * 0.5 / fs)])
    hl = h.tolist()
    hyst = p["reversal_hysteresis_deg"]
    ext = []
    trend, hi, lo = 0, 0, 0
    for i in range(1, n_samp):
        v = hl[i]
        if trend == 0:
            if v > hl[hi]:
                hi = i
            if v < hl[lo]:
                lo = i
            if v - hl[lo] >= hyst:
                ext.append(lo)
                trend, hi = 1, i
            elif hl[hi] - v >= hyst:
                ext.append(hi)
                trend, lo = -1, i
        elif trend == 1:
            if v > hl[hi]:
                hi = i
            elif hl[hi] - v >= hyst:
                ext.append(hi)
                trend, lo = -1, i
        else:
            if v < hl[lo]:
                lo = i
            elif v - hl[lo] >= hyst:
                ext.append(lo)
                trend, hi = 1, i
    if trend == 0:
        return out
    bounds = ext + [hi if trend == 1 else lo]
    n_sw = len(bounds) - 1
    for j in range(n_sw):
        s, e = bounds[j], bounds[j + 1]
        d = 1 if hl[e] > hl[s] else -1
        amp = abs(hl[e] - hl[s])
        seg = h[s:e + 1]
        prog = d * (seg - hl[s])
        rem = d * (hl[e] - seg)
        first, last = j == 0, j == n_sw - 1
        n_turns = 0
        if amp >= p["turn_min_sweep_deg"]:
            n_turns = max(1, int(_np.floor(amp / p["turn_nominal_deg"] + 0.5)))
            jc = int(_np.argmax(prog >= p["commit_deg"]))
            on = int(_np.nonzero(prog[:jc + 1] <= p["plateau_tol_deg"])[0][-1])
            kc = int(_np.nonzero(rem >= p["commit_deg"])[0][-1])
            off = kc + int(_np.argmax(rem[kc:] <= p["plateau_tol_deg"]))
            cuts = [on] + [int(_np.argmax(prog >= m * amp / n_turns)) for m in range(1, n_turns)] + [off]
            trunc_start = first and (s + on) / fs < p["edge_s"]
            trunc_end = last and (n_samp - 1 - (s + off)) / fs < p["edge_s"]
            for m in range(n_turns):
                a, b = s + cuts[m], s + cuts[m + 1]
                out["turns"].append({
                    "sweep": j, "k_in_sweep": m, "n_in_sweep": n_turns, "onset_idx": a, "offset_idx": b,
                    "onset_s": a / fs, "offset_s": b / fs, "duration_s": (b - a) / fs, "direction": d,
                    "sweep_amplitude_deg": amp,
                    "truncated": bool((m == 0 and trunc_start) or (m == n_turns - 1 and trunc_end))})
        out["sweeps"].append({"sweep": j, "start_idx": s, "end_idx": e, "start_s": s / fs, "end_s": e / fs,
                              "direction": d, "amplitude_deg": amp, "n_turns": n_turns,
                              "first": first, "provisional_end": last})
    return out
# <<< TURN RULE END


# ---------------------------------------------------------------- freeze the rule
def extract_rule():
    with open(os.path.abspath(__file__), "rb") as f:
        src = f.read().decode("utf-8").replace("\r\n", "\n")
    m = re.search(r"^# >>> TURN RULE BEGIN.*?^# <<< TURN RULE END[^\n]*\n", src, re.S | re.M)
    if not m:
        sys.exit("turn rule markers not found")
    return m.group(0)


RULE_TXT = os.path.join(OUT, "turn_rule.py.txt")
RULE_SHA = os.path.join(OUT, "turn_rule.sha256")
RULE_FREEZE = os.path.join(OUT, "turn_rule_freeze.json")
TURN_OUTPUTS = ["turns.csv", "sweeps.csv", "turn_trials.csv"]
rule_text = extract_rule()
rule_sha = hashlib.sha256(rule_text.encode("utf-8")).hexdigest()
if os.path.exists(RULE_SHA):
    with open(RULE_SHA, encoding="utf-8") as f:
        frozen_sha = f.read().split()[0]
    with open(RULE_TXT, "rb") as f:
        txt_sha = hashlib.sha256(f.read()).hexdigest()
    if frozen_sha != rule_sha or txt_sha != rule_sha:
        sys.exit(f"turn rule differs from the frozen one (frozen {frozen_sha}, file text {txt_sha}, "
                 f"script {rule_sha}): the rule may not change after freezing; stop")
    with open(RULE_FREEZE, encoding="utf-8") as f:
        freeze_info = json.load(f)
    log("turn rule verified against the frozen hash", rule_sha)
else:
    present = [n for n in TURN_OUTPUTS if os.path.exists(os.path.join(OUT, n))]
    with open(RULE_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write(rule_text)
    with open(RULE_SHA, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{rule_sha}  turn_rule.py.txt\n")
    freeze_info = {"sha256": rule_sha, "frozen_at": _dt.datetime.now().isoformat(timespec="seconds"),
                   "turn_statistic_files_present_at_freeze": present,
                   "note": "written by probe_fog_imu.py before any turn statistic was computed; "
                           "later runs verify the hash and stop on a mismatch"}
    with open(RULE_FREEZE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(freeze_info, f, indent=1)
    log("turn rule frozen:", rule_sha)
if FREEZE_ONLY:
    log("--freeze-rule-only: stop before reading any data")
    sys.exit(0)
RULE = {}
exec(compile(rule_text, "turn_rule.py.txt", "exec"), RULE)  # noqa: S102 (the frozen text is what runs)


# ---------------------------------------------------------------- helpers
def write_csv(name, rows, fields=None):
    path = os.path.join(OUT, name)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in fields})
    log("wrote", name, len(rows), "rows")


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "" if np.isnan(v) else str(v)
        return f"{float(v):.6g}" if abs(v) < 1e6 else f"{float(v):.1f}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return v


def pct(x, q):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, q)) if x.size else float("nan")


def dist(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "min": float(x.min()), "p10": pct(x, 10), "p25": pct(x, 25), "median": pct(x, 50),
            "p75": pct(x, 75), "p90": pct(x, 90), "max": float(x.max())}


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def runs_of(mask):
    f = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(f)
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------- IMU records
EXPECTED = ["Frame #", "Time [s]", "ACC ML [g]", "ACC AP [g]", "ACC SI [g]",
            "GYR ML [deg/s]", "GYR AP [deg/s]", "GYR SI [deg/s]"]
FLAG = "Freezing event [flag]"
NAME_RE = re.compile(r"^IMU/SUB(\d+)_(standing|\d+)\.txt$")
recs = {}
zip_other, csv_members, csv_ole2 = [], 0, 0
with zipfile.ZipFile(f"{RAW}/IMU.zip") as z:
    names = sorted(n for n in z.namelist() if not n.rsplit("/", 1)[-1].startswith("._"))
    for n in names:
        if n.endswith("/"):
            continue
        if n.lower().endswith(".csv"):
            csv_members += 1
            with z.open(n) as fh:
                csv_ole2 += fh.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
            continue
        m = NAME_RE.match(n)
        if not m:
            zip_other.append(n)
            continue
        df = pd.read_csv(io.BytesIO(z.read(n)), sep="\t", encoding="utf-8")
        cols = [c.strip() for c in df.columns]
        subj, trial = int(m.group(1)), m.group(2)
        t = df.iloc[:, 1].to_numpy(dtype=float)
        frame = df.iloc[:, 0].to_numpy(dtype=float)
        acc = df.iloc[:, 2:5].to_numpy(dtype=float)
        gyr = df.iloc[:, 5:8].to_numpy(dtype=float)
        has_flag = FLAG in cols
        flag = df[df.columns[cols.index(FLAG)]].to_numpy(dtype=float) if has_flag else None
        dt = np.diff(t)
        recs[(subj, trial)] = {"file": n, "subject": subj, "trial": trial, "t": t, "acc": acc, "gyr": gyr,
                               "flag": None if flag is None else (flag > 0.5).astype(np.int8),
                               "cols_ok": cols[:8] == EXPECTED, "n_cols": len(cols), "has_flag": has_flag,
                               "flag_values": "" if flag is None else " ".join(str(v) for v in sorted(set(flag.tolist()))),
                               "nan_values": int(np.isnan(np.column_stack([acc, gyr])).sum()),
                               "frames_consecutive": bool(np.all(np.diff(frame) == 1)),
                               "dt_min": float(dt.min()), "dt_max": float(dt.max()),
                               "grid_exact": bool(np.allclose(dt, 1.0 / FS, rtol=0, atol=1e-9)),
                               "t0": float(t[0])}
        log("read", n, len(t), "samples")
turning = sorted(k for k in recs if k[1] != "standing")
standing = sorted(k for k in recs if k[1] == "standing")
subjects = sorted({k[0] for k in recs})
log(f"records: {len(recs)} ({len(turning)} turning, {len(standing)} standing), subjects {len(subjects)}")

# ---------------------------------------------------------------- PDFEinfo
with open(f"{RAW}/PDFEinfo.csv", encoding="latin-1", newline="") as f:
    info_rows = list(csv.reader(f, delimiter=";"))
hdr = [re.sub(r"\s+", " ", h).strip() for h in info_rows[0]]


def col(r, label):
    return r[hdr.index(label)].strip() if label in hdr and hdr.index(label) < len(r) else ""


def num(x):
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None


def parse_intervals(cell):
    c = cell.strip()
    if c in ("", "-"):
        return None
    if num(c) == 0:
        return []
    iv = [(float(a), float(b)) for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", c)]
    return iv if iv else "unparsed"


pdfe = {}
for r in info_rows[1:]:
    if not r or not r[0].strip().upper().startswith("PDFE"):
        continue
    sid = int(re.sub(r"\D", "", r[0]))
    pdfe[sid] = {"side": col(r, "More affected side"), "sessions": col(r, "sessions #"), "s": {}}
    for s in (1, 2, 3):
        pdfe[sid]["s"][s] = {"episodes": num(col(r, f"Session {s} - numbers of FoG episodes (n)")),
                             "total_s": num(col(r, f"Session {s} - total time in FoG (s)")),
                             "intervals": parse_intervals(col(r, f"Session {s} - time of FoG (s)"))}
log("PDFEinfo subjects:", len(pdfe))

# ---------------------------------------------------------------- records.csv and sensor facts
rec_rows, frame_rows = [], []
for k in sorted(recs):
    R = recs[k]
    row = {"file": R["file"], "subject": R["subject"], "trial": R["trial"], "samples": len(R["t"]),
           "duration_s": float(R["t"][-1] - R["t"][0] + 1.0 / FS), "t0_s": R["t0"], "columns_as_expected": R["cols_ok"],
           "n_columns": R["n_cols"], "flag_column": R["has_flag"], "flag_values": R["flag_values"],
           "nan_values": R["nan_values"], "frames_consecutive": R["frames_consecutive"],
           "dt_min_s": R["dt_min"], "dt_max_s": R["dt_max"], "grid_exact_1_128": R["grid_exact"],
           "side": pdfe.get(R["subject"], {}).get("side", ""),
           "acc_ml_mean_g": float(R["acc"][:, 0].mean()), "acc_ap_mean_g": float(R["acc"][:, 1].mean()),
           "acc_si_mean_g": float(R["acc"][:, 2].mean()),
           "gyr_ml_median_dps": float(np.median(R["gyr"][:, 0])), "gyr_ap_median_dps": float(np.median(R["gyr"][:, 1])),
           "gyr_si_median_dps": float(np.median(R["gyr"][:, 2]))}
    rec_rows.append(row)
    if R["trial"] != "standing":
        sdg = R["gyr"].std(axis=0)
        frame_rows.append({"file": R["file"], "subject": R["subject"], "trial": R["trial"], "side": row["side"],
                           "flagged_fraction": float(R["flag"].mean()) if R["flag"] is not None else "",
                           "gyr_ml_sd_dps": float(sdg[0]), "gyr_ap_sd_dps": float(sdg[1]), "gyr_si_sd_dps": float(sdg[2])})
write_csv("records.csv", rec_rows)

# ---------------------------------------------------------------- (1) flag structure
flag_rows, run_rows = [], []
all_gaps, all_runs = [], []
for k in turning:
    R = recs[k]
    t, fl = R["t"], R["flag"]
    rs = runs_of(fl)
    dur = [(e - s + 1) / FS for s, e in rs]
    gaps = [(rs[i + 1][0] - rs[i][1] - 1) / FS for i in range(len(rs) - 1)]
    all_runs += dur
    all_gaps += gaps
    sess = int(k[1])
    iv = pdfe.get(k[0], {}).get("s", {}).get(sess, {}).get("intervals")
    iv_ok = isinstance(iv, list) and len(iv) == len(rs)
    sd, ed = [], []
    for i, (s, e) in enumerate(rs):
        rr = {"file": R["file"], "subject": k[0], "session": sess, "run": i + 1, "start_idx": s, "end_idx": e,
              "start_s": float(t[s]), "end_s": float(t[e]), "duration_s": dur[i],
              "gap_before_s": gaps[i - 1] if i > 0 else "", "pdfe_start_s": "", "pdfe_end_s": "",
              "start_minus_pdfe_s": "", "end_minus_pdfe_s": ""}
        if iv_ok:
            a, b = iv[i]
            rr.update({"pdfe_start_s": a, "pdfe_end_s": b, "start_minus_pdfe_s": float(t[s]) - a,
                       "end_minus_pdfe_s": float(t[e]) - b})
            sd.append(float(t[s]) - a)
            ed.append(float(t[e]) - b)
        run_rows.append(rr)
    flag_rows.append({
        "file": R["file"], "subject": k[0], "session": sess, "samples": len(t), "runs": len(rs),
        "flagged_s": float(fl.sum() / FS), "flagged_fraction": float(fl.mean()),
        "min_run_s": min(dur) if dur else "", "max_run_s": max(dur) if dur else "",
        "min_gap_s": min(gaps) if gaps else "", "first_flag_s": float(t[rs[0][0]]) if rs else "",
        "last_flag_s": float(t[rs[-1][1]]) if rs else "", "flag_at_first_sample": bool(fl[0]) if len(fl) else "",
        "flag_at_last_sample": bool(fl[-1]) if len(fl) else "",
        "pdfe_episodes": pdfe.get(k[0], {}).get("s", {}).get(sess, {}).get("episodes", ""),
        "pdfe_total_s": pdfe.get(k[0], {}).get("s", {}).get(sess, {}).get("total_s", ""),
        "pdfe_intervals": "unparsed" if iv == "unparsed" else ("" if iv is None else len(iv)),
        "runs_match_pdfe_intervals": iv_ok,
        "max_abs_start_diff_s": max(abs(x) for x in sd) if sd else "",
        "max_abs_end_diff_s": max(abs(x) for x in ed) if ed else ""})
write_csv("flag_structure.csv", flag_rows)
write_csv("flag_runs.csv", run_rows)
all_sd = [r["start_minus_pdfe_s"] for r in run_rows if r["start_minus_pdfe_s"] != ""]
all_ed = [r["end_minus_pdfe_s"] for r in run_rows if r["end_minus_pdfe_s"] != ""]
gaps_sorted = sorted(all_gaps)
flag_facts = {
    "turning_trials": len(turning),
    "trials_with_runs": sum(1 for r in flag_rows if r["runs"] > 0),
    "runs_total": len(all_runs),
    "run_s": dist(all_runs), "gap_s": dist(all_gaps),
    "shortest_10_gaps_s": gaps_sorted[:10],
    "gaps_below_s": {str(x): int(sum(g < x for g in all_gaps)) for x in (0.5, 1.0, 2.0, 3.0, 5.0)},
    "runs_below_s": {str(x): int(sum(g < x for g in all_runs)) for x in (0.5, 1.0, 2.0)},
    "trials_flag_at_first_sample": sum(1 for r in flag_rows if r["flag_at_first_sample"] is True),
    "trials_flag_at_last_sample": sum(1 for r in flag_rows if r["flag_at_last_sample"] is True),
    "trials_runs_match_pdfe_interval_count": sum(1 for r in flag_rows if r["runs_match_pdfe_intervals"]),
    "trials_with_pdfe_intervals_parsed": sum(1 for r in flag_rows if r["pdfe_intervals"] not in ("", "unparsed")),
    "start_minus_pdfe_s": dist(all_sd), "end_minus_pdfe_s": dist(all_ed),
    "max_abs_boundary_diff_s": max([abs(x) for x in all_sd + all_ed]) if all_sd else None,
    "max_abs_boundary_diff_s_runs_within_0.05s": max(
        [m for m in (max(abs(r["start_minus_pdfe_s"]), abs(r["end_minus_pdfe_s"])) for r in run_rows
                     if r["start_minus_pdfe_s"] != "") if m <= 0.05], default=None),
    "trials_runs_match_pdfe_episode_count": sum(
        1 for r in flag_rows if isinstance(r["pdfe_episodes"], float) and r["runs"] == r["pdfe_episodes"]),
    "trials_with_pdfe_episode_count": sum(1 for r in flag_rows if isinstance(r["pdfe_episodes"], float)),
    "subjects_mean_flagged_fraction_ge_0.9": {
        str(sid): float(np.mean([r["flagged_fraction"] for r in flag_rows if r["subject"] == sid]))
        for sid in sorted({r["subject"] for r in flag_rows})
        if np.mean([r["flagged_fraction"] for r in flag_rows if r["subject"] == sid]) >= 0.9},
    "runs_boundary_diff_above_0.05s": [
        {k2: r[k2] for k2 in ("file", "run", "start_s", "end_s", "pdfe_start_s", "pdfe_end_s")}
        for r in run_rows if r["start_minus_pdfe_s"] != "" and
        max(abs(r["start_minus_pdfe_s"]), abs(r["end_minus_pdfe_s"])) > 0.05],
    "pdfe_intervals_end_before_start": [
        {"subject": sid, "session": s, "interval": list(iv)} for sid in sorted(pdfe) for s in (1, 2, 3)
        if isinstance(pdfe[sid]["s"][s]["intervals"], list) for iv in pdfe[sid]["s"][s]["intervals"] if iv[1] < iv[0]],
}
log("flag facts:", json.dumps(r3(flag_facts))[:600])

# ---------------------------------------------------------------- (4) subjects with episodes
subj_rows, sess_rows = [], []
freezer = {}
for sid in subjects:
    row = {"subject": sid, "side": pdfe.get(sid, {}).get("side", ""), "pdfe_sessions": pdfe.get(sid, {}).get("sessions", "")}
    any_flag = False
    for s in (1, 2, 3):
        R = recs.get((sid, str(s)))
        nr = len(runs_of(R["flag"])) if R is not None else ""
        row[f"s{s}_imu_trial"] = R is not None
        row[f"s{s}_flag_runs"] = nr
        row[f"s{s}_pdfe_episodes"] = pdfe.get(sid, {}).get("s", {}).get(s, {}).get("episodes", "")
        any_flag = any_flag or (R is not None and nr > 0)
    row["standing_file"] = (sid, "standing") in recs
    row["freezer_by_flags"] = any_flag
    freezer[sid] = any_flag
    subj_rows.append(row)
for s in (1, 2, 3):
    with_trial = [r for r in subj_rows if r[f"s{s}_imu_trial"]]
    pd_rows = [sid for sid in pdfe if pdfe[sid]["s"][s]["episodes"] is not None]
    sess_rows.append({"scope": f"session {s}", "subjects_with_imu_trial": len(with_trial),
                      "subjects_with_flagged_episode": sum(1 for r in with_trial if r[f"s{s}_flag_runs"] > 0),
                      "subjects_in_pdfe_with_session": len(pd_rows),
                      "subjects_with_pdfe_episode": sum(1 for sid in pd_rows if pdfe[sid]["s"][s]["episodes"] > 0)})
sess_rows.append({"scope": "overall", "subjects_with_imu_trial": sum(1 for r in subj_rows if any(r[f"s{s}_imu_trial"] for s in (1, 2, 3))),
                  "subjects_with_flagged_episode": sum(1 for r in subj_rows if r["freezer_by_flags"]),
                  "subjects_in_pdfe_with_session": len(pdfe),
                  "subjects_with_pdfe_episode": sum(1 for sid in pdfe if any((pdfe[sid]["s"][s]["episodes"] or 0) > 0 for s in (1, 2, 3)))})
write_csv("subjects.csv", subj_rows)
write_csv("subjects_freezing.csv", sess_rows)
nonfreezers = [sid for sid in subjects if not freezer[sid]]
log("non-freezers by flags:", nonfreezers)

# ---------------------------------------------------------------- (3) freeze index
with zipfile.ZipFile(DAPHNET_ZIP) as z:
    do_test_bytes = z.read(DAPHNET_TEST)
do_test = do_test_bytes.decode("latin-1")
live = "\n".join(line.split("%")[0] for line in do_test.splitlines())
m_fr = re.search(r"TH\.freeze\s*=\s*\[([^\]]*)\]", live)
m_pw = re.search(r"TH\.power\s*=\s*2\.\^\s*([\d.]+)", live)
TH_FREEZE = sorted({float(v) for v in m_fr.group(1).split()})
TH_POWER_EXP = float(m_pw.group(1))
TH_POWER = 2.0 ** TH_POWER_EXP
log("Daphnet do_test.m thresholds: TH.freeze", TH_FREEZE, "TH.power 2^", TH_POWER_EXP)

FRES = FS / FI_WIN


def mround(v):
    return int(np.floor(v + 0.5))


BANDS = {  # python (0-based) bin index ranges, inclusive
    "nominal": ((mround(0.5 / FRES), mround(3.0 / FRES)), (mround(3.0 / FRES), mround(8.0 / FRES))),
    "daphnet_index": ((mround(0.5 / FRES) - 1, mround(3.0 / FRES) - 1), (mround(3.0 / FRES) - 1, mround(8.0 / FRES) - 1)),
}


def fi_block(x_mg, starts):
    seg = np.stack([x_mg[s:s + FI_WIN] for s in starts])
    seg = seg - seg.mean(axis=1, keepdims=True)
    Y = np.fft.fft(seg, n=FI_WIN, axis=1)
    P = (Y * np.conj(Y)).real / FI_WIN
    res = {}
    for name, ((l0, l1), (f0, f1)) in BANDS.items():
        pl, pf = P[:, l0:l1 + 1], P[:, f0:f1 + 1]
        loco = (pl[:, 1:].sum(1) / FS + pl[:, :-1].sum(1) / FS) / 2.0
        frz = (pf[:, 1:].sum(1) / FS + pf[:, :-1].sum(1) / FS) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            res[name] = (np.where(loco > 0, frz / loco, np.nan), loco + frz)
    return res


win_rows = []
skipped_short = []
for k in sorted(recs):
    R = recs[k]
    n = len(R["t"])
    if n < FI_WIN:
        skipped_short.append(R["file"])
        continue
    nw = n // SCORE_WIN
    starts = [min(max(w * SCORE_WIN + SCORE_WIN // 2 - FI_WIN // 2, 0), n - FI_WIN) for w in range(nw)]
    feats = {ax: fi_block(R["acc"][:, ai] * 1000.0, starts) for ax, ai in (("si", 2), ("ap", 1))}
    for w in range(nw):
        a = w * SCORE_WIN
        fr = float(R["flag"][a:a + SCORE_WIN].mean()) if R["flag"] is not None else float("nan")
        row = {"file": R["file"], "subject": k[0], "trial": k[1], "window": w, "t_start_s": float(R["t"][a]),
               "fi_start_idx": starts[w], "flag_fraction": fr,
               "label": "" if R["flag"] is None else int(fr >= 0.5)}
        for ax in ("si", "ap"):
            for name in BANDS:
                fi, pw = feats[ax][name]
                row[f"fi_{name}_{ax}"] = float(fi[w])
                row[f"power_{name}_{ax}"] = float(pw[w])
        win_rows.append(row)
write_csv("fi_windows.csv", win_rows)
W = pd.DataFrame(win_rows)
TW = W[W["trial"] != "standing"].copy()
SW = W[W["trial"] == "standing"].copy()
TW["label"] = TW["label"].astype(int)

VARIANTS = {  # name: (score column, gated, description)
    "nominal_si": ("fi_nominal_si", False, "vertical (SI) acceleration, bands 0.5-3 and 3-8 Hz, 4 s window, ungated (PRIMARY)"),
    "nominal_ap": ("fi_nominal_ap", False, "AP acceleration, bands 0.5-3 and 3-8 Hz, 4 s window, ungated"),
    "daphnet_index_si": ("fi_daphnet_index_si", False, "SI, Daphnet x_fi.m bin indexing (0.25-2.75 and 2.75-7.75 Hz), ungated"),
    "nominal_si_gated": ("fi_nominal_si", True, "SI nominal, set to 0 when loco+freeze power < TH.power (do_test.m)"),
    "daphnet_index_si_gated": ("fi_daphnet_index_si", True, "SI Daphnet indexing, gated as do_test.m"),
    "neg_power_si_posthoc": ("power_nominal_si", False, "POST HOC, not a freeze index: minus the 0.5-8 Hz SI power of "
                             "the same 4 s window; added after the gated variants ranked below 0.5"),
}


def score(df, variant):
    c, gated, _ = VARIANTS[variant]
    s = df[c].to_numpy(dtype=float)
    if variant.startswith("neg_"):
        return -s
    if gated:
        pw = df[c.replace("fi_", "power_")].to_numpy(dtype=float)
        s = np.where(pw < TH_POWER, 0.0, s)
    return s


fi_nan = {v: int(np.isnan(score(TW, v)).sum()) for v in VARIANTS}
_ypos = TW["label"].to_numpy(dtype=int) == 1
power_gate = {"th_power": TH_POWER}
for _b in BANDS:
    _pt = TW[f"power_{_b}_si"].to_numpy(dtype=float) < TH_POWER
    _ps = SW[f"power_{_b}_si"].to_numpy(dtype=float) < TH_POWER
    power_gate[_b] = {"positive_windows_below_th_power": float(_pt[_ypos].mean()),
                      "negative_windows_below_th_power": float(_pt[~_ypos].mean()),
                      "standing_windows_below_th_power": float(_ps.mean()) if _ps.size else None}
subj_t = sorted(TW["subject"].unique().tolist())
rng = np.random.default_rng(SEED)
draws = rng.integers(0, len(subj_t), size=(N_BOOT, len(subj_t)))
auc_rows, auc_subj_rows = [], []
for v in VARIANTS:
    s_all = score(TW, v)
    ok = ~np.isnan(s_all)
    y_all = TW["label"].to_numpy(dtype=int)
    sub_all = TW["subject"].to_numpy()
    per = {sid: (y_all[(sub_all == sid) & ok], s_all[(sub_all == sid) & ok]) for sid in subj_t}
    point = auroc(y_all[ok], s_all[ok])
    boots, invalid = [], 0
    for d in draws:
        yy = np.concatenate([per[subj_t[i]][0] for i in d])
        ss = np.concatenate([per[subj_t[i]][1] for i in d])
        a = auroc(yy, ss)
        if np.isnan(a):
            invalid += 1
        else:
            boots.append(a)
    per_auc = {sid: auroc(*per[sid]) for sid in subj_t}
    valid_sub = [a for a in per_auc.values() if not np.isnan(a)]
    auc_rows.append({"variant": v, "description": VARIANTS[v][2], "windows": int(ok.sum()),
                     "positive_windows": int(y_all[ok].sum()), "subjects": len(subj_t),
                     "subjects_with_positive": sum(1 for sid in subj_t if per[sid][0].sum() > 0),
                     "auroc": point, "ci95_low": pct(boots, 2.5), "ci95_high": pct(boots, 97.5),
                     "boot_sd": float(np.std(boots)), "boot_draws": N_BOOT, "boot_invalid": invalid, "seed": SEED,
                     "per_subject_n": len(valid_sub), "per_subject_median": pct(valid_sub, 50),
                     "per_subject_min": min(valid_sub) if valid_sub else "", "per_subject_max": max(valid_sub) if valid_sub else ""})
    for sid in subj_t:
        auc_subj_rows.append({"variant": v, "subject": sid, "windows": len(per[sid][0]),
                              "positive_windows": int(per[sid][0].sum()), "auroc": per_auc[sid]})
    log(f"AUROC {v}: {point:.4f} [{pct(boots, 2.5):.4f}, {pct(boots, 97.5):.4f}]")
write_csv("fi_auroc.csv", auc_rows)
write_csv("fi_auroc_per_subject.csv", auc_subj_rows)


def events(df, det):
    n = 0
    for _, g in df.assign(det=det).groupby("file", sort=False):
        n += len(runs_of(g.sort_values("window")["det"].to_numpy()))
    return n


thr_rows = []
NF = TW[TW["subject"].isin(nonfreezers)]
zero_flag_files = [r["file"] for r in flag_rows if r["runs"] == 0]
ZF = TW[TW["file"].isin(zero_flag_files)]
for v in ("nominal_si", "daphnet_index_si"):
    for gated in (True, False):
        vv = v + ("_gated" if gated else "")
        for th in TH_FREEZE:
            dT = score(TW, vv) > th
            dS = score(SW, vv) > th
            dN = score(NF, vv) > th
            dZ = score(ZF, vv) > th
            y = TW["label"].to_numpy(dtype=int) == 1
            thr_rows.append({
                "variant": vv, "th_freeze": th, "th_power": TH_POWER if gated else "",
                "sensitivity_windows": float(dT[y].mean()) if y.any() else "",
                "specificity_windows": float(1 - dT[~y].mean()),
                "turning_negative_windows": int((~y).sum()), "turning_negative_alarms": int(dT[~y].sum()),
                "standing_windows": len(SW), "standing_minutes": len(SW) * SCORE_WIN / FS / 60.0,
                "standing_alarm_windows": int(dS.sum()),
                "standing_alarms_per_min": dS.sum() / (len(SW) * SCORE_WIN / FS / 60.0) if len(SW) else "",
                "standing_alarm_events": events(SW, dS),
                "standing_events_per_min": events(SW, dS) / (len(SW) * SCORE_WIN / FS / 60.0) if len(SW) else "",
                "nonfreezer_subjects": len(nonfreezers), "nonfreezer_windows": len(NF),
                "nonfreezer_minutes": len(NF) * SCORE_WIN / FS / 60.0,
                "nonfreezer_alarm_windows": int(dN.sum()),
                "nonfreezer_alarms_per_min": dN.sum() / (len(NF) * SCORE_WIN / FS / 60.0) if len(NF) else "",
                "nonfreezer_alarm_events": events(NF, dN),
                "nonfreezer_events_per_min": events(NF, dN) / (len(NF) * SCORE_WIN / FS / 60.0) if len(NF) else "",
                "zero_flag_trials": len(zero_flag_files), "zero_flag_windows": len(ZF),
                "zero_flag_alarms_per_min": dZ.sum() / (len(ZF) * SCORE_WIN / FS / 60.0) if len(ZF) else ""})
write_csv("fi_thresholds.csv", thr_rows)

# ---------------------------------------------------------------- (2) turns under the frozen rule
log("turn statistics start (rule sha256 %s)" % rule_sha)
standing_bias = {}
for k in standing:
    b, nst = RULE["estimate_gyro_bias"](recs[k]["gyr"], FS)
    standing_bias[k[0]] = b
turn_rows, sweep_rows, tt_rows = [], [], []
for k in sorted(recs):
    R = recs[k]
    fb = standing_bias.get(k[0]) if k[1] != "standing" else None
    res = RULE["segment_turns"](R["gyr"], FS, fallback_bias=fb)
    t = R["t"]
    fl = R["flag"]
    trs = res["turns"]
    for tr in trs:
        a, b = tr["onset_idx"], tr["offset_idx"]
        nfl = int(fl[a:b + 1].sum()) if fl is not None else ""
        turn_rows.append({"file": R["file"], "subject": k[0], "trial": k[1], "sweep": tr["sweep"],
                          "k_in_sweep": tr["k_in_sweep"], "n_in_sweep": tr["n_in_sweep"],
                          "onset_s": float(t[a]), "offset_s": float(t[b]), "duration_s": tr["duration_s"],
                          "direction": tr["direction"], "sweep_amplitude_deg": tr["sweep_amplitude_deg"],
                          "truncated": tr["truncated"], "flagged_samples": nfl,
                          "freezing_free": (nfl == 0) if fl is not None else ""})
    for sw in res["sweeps"]:
        sweep_rows.append({"file": R["file"], "subject": k[0], "trial": k[1], "sweep": sw["sweep"],
                           "start_s": float(t[sw["start_idx"]]), "end_s": float(t[sw["end_idx"]]),
                           "direction": sw["direction"], "amplitude_deg": sw["amplitude_deg"],
                           "n_turns": sw["n_turns"], "first": sw["first"], "provisional_end": sw["provisional_end"]})
    dirs = [tr["direction"] for tr in trs]
    pairs = len(dirs) - 1 if len(dirs) > 1 else 0
    alt = sum(1 for i in range(pairs) if dirs[i] != dirs[i + 1])
    comp = [tr for tr in trs if not tr["truncated"]]
    inside = 0
    if fl is not None and fl.sum() > 0:
        cover = np.zeros(len(fl), dtype=bool)
        for tr in trs:
            cover[tr["onset_idx"]:tr["offset_idx"] + 1] = True
        inside = float(fl[cover].sum() / fl.sum())
    cfree = [tr for tr in comp if fl is not None and fl[tr["onset_idx"]:tr["offset_idx"] + 1].sum() == 0]
    tt_rows.append({"file": R["file"], "subject": k[0], "trial": k[1], "side": pdfe.get(k[0], {}).get("side", ""),
                    "bias_dps": res["bias_dps"], "bias_source": res["bias_source"], "still_windows": res["n_still_windows"],
                    "sweeps": len(res["sweeps"]),
                    "partial_sweeps": sum(1 for sw in res["sweeps"] if sw["n_turns"] == 0),
                    "multi_turn_sweeps": sum(1 for sw in res["sweeps"] if sw["n_turns"] > 1),
                    "turns": len(trs), "complete_turns": len(comp), "truncated_turns": len(trs) - len(comp),
                    "turn_pairs": pairs, "alternating_pairs": alt,
                    "first_turn_direction": dirs[0] if dirs else "",
                    "complete_turn_median_s": pct([tr["duration_s"] for tr in comp], 50) if comp else "",
                    "complete_freezing_free_turns": len(cfree) if fl is not None else "",
                    "flagged_fraction_inside_turns": inside if (fl is not None and fl.sum() > 0) else ""})
    log("turns", R["file"], len(trs), "bias", round(res["bias_dps"], 3), res["bias_source"])
write_csv("turns.csv", turn_rows)
write_csv("sweeps.csv", sweep_rows)
write_csv("turn_trials.csv", tt_rows)

# axis check (diagnostic, after the turn statistics): the frozen rule applied with the gyro columns
# permuted so that ML or AP plays the heading axis; the fallback bias is the standing-file bias of
# that axis from the same frozen estimator.
turns_si = {(r["subject"], r["trial"]): r["turns"] for r in tt_rows}
for fr in frame_rows:
    k = (fr["subject"], fr["trial"])
    fr["turns_heading_si"] = turns_si[k]
    for ax, name in ((0, "ml"), (1, "ap")):
        perm = [c for c in (0, 1, 2) if c != ax] + [ax]
        st = recs.get((k[0], "standing"))
        fb = RULE["estimate_gyro_bias"](st["gyr"][:, perm], FS)[0] if st is not None else None
        fr[f"turns_heading_{name}"] = len(RULE["segment_turns"](recs[k]["gyr"][:, perm], FS, fallback_bias=fb)["turns"])
write_csv("sensor_frame.csv", frame_rows)

TT = [r for r in tt_rows if r["trial"] != "standing"]
TS = [r for r in tt_rows if r["trial"] == "standing"]
TR = [r for r in turn_rows if r["trial"] != "standing"]
SWt = [r for r in sweep_rows if r["trial"] != "standing"]
comp = [r for r in TR if not r["truncated"]]
comp_free = [r for r in comp if r["freezing_free"] is True]
comp_frz = [r for r in comp if r["freezing_free"] is False]
turn_facts = {
    "rule_sha256": rule_sha, "rule_frozen_at": freeze_info.get("frozen_at"),
    "turning_trials": len(TT),
    "bias_source_counts": {s: sum(1 for r in TT if r["bias_source"] == s) for s in ("record_still_windows", "fallback", "none")},
    "bias_dps": dist([r["bias_dps"] for r in TT]),
    "standing_bias_dps": dist([v for v in standing_bias.values() if v is not None]),
    "standing_bias_missing": sum(1 for v in standing_bias.values() if v is None),
    "sweeps": len(SWt), "partial_sweeps": sum(1 for r in SWt if r["n_turns"] == 0),
    "partial_sweeps_180_or_more": sum(1 for r in SWt if r["n_turns"] == 0 and r["amplitude_deg"] >= 180),
    "multi_turn_sweeps": sum(1 for r in SWt if r["n_turns"] > 1),
    "sweep_amplitude_deg_turn_sweeps": dist([r["amplitude_deg"] for r in SWt if r["n_turns"] > 0]),
    "sweep_amplitude_deg_single_turn_sweeps": dist([r["amplitude_deg"] for r in SWt if r["n_turns"] == 1]),
    "sweep_amplitude_deg_partial_sweeps": dist([r["amplitude_deg"] for r in SWt if r["n_turns"] == 0]),
    "turns": len(TR), "complete_turns": len(comp), "truncated_turns": len(TR) - len(comp),
    "turns_per_trial": dist([r["turns"] for r in TT]),
    "complete_turns_per_trial": dist([r["complete_turns"] for r in TT]),
    "trials_without_turn": sum(1 for r in TT if r["turns"] == 0),
    "turn_pairs": sum(r["turn_pairs"] for r in TT), "alternating_pairs": sum(r["alternating_pairs"] for r in TT),
    "trials_fully_alternating": sum(1 for r in TT if r["turn_pairs"] > 0 and r["alternating_pairs"] == r["turn_pairs"]),
    "trials_with_pairs": sum(1 for r in TT if r["turn_pairs"] > 0),
    "direction_counts": {"+1": sum(1 for r in TR if r["direction"] == 1), "-1": sum(1 for r in TR if r["direction"] == -1)},
    "first_turn_direction_by_side": {f"{side} {d:+d}": sum(1 for r in TT if r["side"] == side and r["first_turn_direction"] == d)
                                     for side in sorted({r["side"] for r in TT}) for d in (1, -1)},
    "complete_turn_duration_s": dist([r["duration_s"] for r in comp]),
    "complete_freezing_free_turn_duration_s": dist([r["duration_s"] for r in comp_free]),
    "complete_turn_with_freezing_duration_s": dist([r["duration_s"] for r in comp_frz]),
    "complete_freezing_free_turns": len(comp_free), "complete_turns_with_freezing": len(comp_frz),
    "subjects_with_freezing_free_complete_turn": len({r["subject"] for r in comp_free}),
    "trials_with_freezing_free_complete_turn": len({r["file"] for r in comp_free}),
    "flagged_fraction_inside_turns_pooled": (
        float(sum(r["flagged_fraction_inside_turns"] * recs[(r["subject"], r["trial"])]["flag"].sum() for r in TT
                  if r["flagged_fraction_inside_turns"] != "")
              / sum(recs[(r["subject"], r["trial"])]["flag"].sum() for r in TT if r["flagged_fraction_inside_turns"] != ""))
        if any(r["flagged_fraction_inside_turns"] != "" for r in TT) else None),
    "standing_files": len(TS), "standing_turns": sum(r["turns"] for r in TS),
    "standing_sweeps": sum(r["sweeps"] for r in TS),
    "trials_without_turn_list": [{"file": r["file"], "flagged_fraction": float(recs[(r["subject"], r["trial"])]["flag"].mean())}
                                 for r in TT if r["turns"] == 0],
    "trials_bias_none": [r["file"] for r in TT if r["bias_source"] == "none"],
    "trials_bias_from_record": [r["file"] for r in TT if r["bias_source"] == "record_still_windows"],
    "freezing_free_complete_turns_per_subject": dist([sum(1 for r in comp_free if r["subject"] == sid) for sid in subjects]),
    "subjects_with_at_least_5_freezing_free_complete_turns": sum(
        1 for sid in subjects if sum(1 for r in comp_free if r["subject"] == sid) >= 5),
}
frame_facts = {
    "gyr_sd_dps": {ax: dist([r[f"gyr_{ax}_sd_dps"] for r in frame_rows]) for ax in ("ml", "ap", "si")},
    "trials_si_sd_largest": sum(1 for r in frame_rows if r["gyr_si_sd_dps"] > max(r["gyr_ml_sd_dps"], r["gyr_ap_sd_dps"])),
    "turns_if_heading": {ax: dist([r[f"turns_heading_{ax}"] for r in frame_rows]) for ax in ("si", "ml", "ap")},
    "trials_si_most_turns": sum(1 for r in frame_rows
                                if r["turns_heading_si"] > max(r["turns_heading_ml"], r["turns_heading_ap"])),
}
log("turn facts:", json.dumps(r3(turn_facts))[:800])

# ---------------------------------------------------------------- summary
rec_facts = {
    "zip_txt_records": len(recs), "turning": len(turning), "standing": len(standing), "subjects": len(subjects),
    "zip_csv_members": csv_members, "zip_csv_members_ole2": int(csv_ole2), "zip_other_members": zip_other,
    "records_columns_as_expected": sum(1 for r in rec_rows if r["columns_as_expected"]),
    "records_grid_exact": sum(1 for r in rec_rows if r["grid_exact_1_128"]),
    "records_frames_consecutive": sum(1 for r in rec_rows if r["frames_consecutive"]),
    "records_with_nan": sum(1 for r in rec_rows if r["nan_values"] > 0),
    "flag_value_sets": sorted({r["flag_values"] for r in rec_rows if r["flag_column"]}),
    "turning_with_flag_column": sum(1 for r in rec_rows if r["trial"] != "standing" and r["flag_column"]),
    "standing_with_flag_column": sum(1 for r in rec_rows if r["trial"] == "standing" and r["flag_column"]),
    "turning_duration_s": dist([r["duration_s"] for r in rec_rows if r["trial"] != "standing"]),
    "standing_duration_s": dist([r["duration_s"] for r in rec_rows if r["trial"] == "standing"]),
    "standing_acc_si_mean_g": dist([r["acc_si_mean_g"] for r in rec_rows if r["trial"] == "standing"]),
    "standing_gyr_si_median_dps": dist([r["gyr_si_median_dps"] for r in rec_rows if r["trial"] == "standing"]),
    "fi_skipped_short_records": skipped_short,
    "fi_nan_windows": fi_nan,
}
facts = {"records": rec_facts, "frame": frame_facts, "flags": flag_facts, "subjects": sess_rows, "nonfreezers": nonfreezers,
         "fi_auroc": auc_rows, "fi_power_gate": power_gate, "fi_thresholds": thr_rows, "turns": turn_facts,
         "daphnet_do_test": {"th_freeze": TH_FREEZE, "th_power_exp2": TH_POWER_EXP,
                             "sha256": hashlib.sha256(do_test_bytes).hexdigest()}}
with open(os.path.join(OUT, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(r3(facts), f, indent=1)


def f3(x):
    return "" if x is None or x == "" else (f"{x:.3f}" if isinstance(x, float) else str(x))


def dline(d, unit=""):
    if not d or d.get("n", 0) == 0:
        return "n 0"
    return (f"n {d['n']}, min {d['min']:.2f}, p25 {d['p25']:.2f}, median {d['median']:.2f}, "
            f"p75 {d['p75']:.2f}, p90 {d['p90']:.2f}, max {d['max']:.2f}{unit}")


A = {r["variant"]: r for r in auc_rows}
md = ["# fog_turning_filho IMU pre-checks for G14 (task T0.8, IMU part)", "",
      "Measured by scripts/probe_fog_imu.py (provenance in _provenance.json, every figure below is also in",
      "facts.json and the CSVs of this folder). IMU only: no video was opened. Flags are specialist video",
      "labels transferred to the IMU clock by the data authors, never IMU-derived.", "",
      "## Records", "",
      f"- IMU.zip holds {rec_facts['zip_txt_records']} .txt records ({rec_facts['turning']} turning trials, "
      f"{rec_facts['standing']} standing files) of {rec_facts['subjects']} subjects; its {rec_facts['zip_csv_members']} "
      f".csv members are OLE2 compound files (legacy Excel) in {rec_facts['zip_csv_members_ole2']} cases and were not read.",
      f"- Columns as expected in {rec_facts['records_columns_as_expected']} records; exact 1/128 s grid in "
      f"{rec_facts['records_grid_exact']}; consecutive frame numbers in {rec_facts['records_frames_consecutive']}; "
      f"records with NaN: {rec_facts['records_with_nan']}; flag value sets: {rec_facts['flag_value_sets']}.",
      f"- Turning trial duration (s): {dline(rec_facts['turning_duration_s'])}. Standing file duration (s): "
      f"{dline(rec_facts['standing_duration_s'])}.",
      f"- Standing ACC SI mean (g): {dline(rec_facts['standing_acc_si_mean_g'])}: the SI axis points up. "
      f"Standing GYR SI median (deg/s): {dline(rec_facts['standing_gyr_si_median_dps'])}.",
      f"- Turning axis: GYR SI has the largest standard deviation of the three gyro axes in "
      f"{frame_facts['trials_si_sd_largest']} of {len(frame_rows)} turning trials (SD, deg/s: ML "
      f"{dline(frame_facts['gyr_sd_dps']['ml'])}; AP {dline(frame_facts['gyr_sd_dps']['ap'])}; SI "
      f"{dline(frame_facts['gyr_sd_dps']['si'])}). The frozen turn rule run with the columns permuted (diagnostic, "
      f"after the turn statistics) finds per trial: SI as heading {dline(frame_facts['turns_if_heading']['si'])}; ML as "
      f"heading {dline(frame_facts['turns_if_heading']['ml'])}; AP as heading "
      f"{dline(frame_facts['turns_if_heading']['ap'])} turns; SI gives the most turns in "
      f"{frame_facts['trials_si_most_turns']} trials (the others have no turn on any axis or a tie).", "",
      "## 1. Flag structure", "",
      f"- {flag_facts['trials_with_runs']} of {flag_facts['turning_trials']} turning trials carry at least one flagged "
      f"run; {flag_facts['runs_total']} runs in total.",
      f"- Run duration (s): {dline(flag_facts['run_s'])}.",
      f"- Gap between consecutive runs of a trial (s, unflagged time): {dline(flag_facts['gap_s'])}.",
      f"- Ten shortest gaps (s): {', '.join(f'{g:.3f}' for g in flag_facts['shortest_10_gaps_s'])}. Gaps below "
      f"0.5/1/2/3/5 s: {', '.join(str(v) for v in flag_facts['gaps_below_s'].values())}. Runs below 0.5/1/2 s: "
      f"{', '.join(str(v) for v in flag_facts['runs_below_s'].values())}.",
      f"- Trials flagged at their first sample: {flag_facts['trials_flag_at_first_sample']}; at their last sample: "
      f"{flag_facts['trials_flag_at_last_sample']} (episodes censored by the record edges).",
      f"- Run count equals the PDFEinfo interval count in {flag_facts['trials_runs_match_pdfe_interval_count']} of "
      f"{flag_facts['turning_trials']} trials; run start minus PDFEinfo start (s): {dline(flag_facts['start_minus_pdfe_s'])}; "
      f"run end minus PDFEinfo end (s): {dline(flag_facts['end_minus_pdfe_s'])}; largest absolute boundary difference "
      f"{f3(flag_facts['max_abs_boundary_diff_s'])} s; largest absolute boundary difference over the runs within 0.05 s "
      f"of PDFEinfo {f3(flag_facts['max_abs_boundary_diff_s_runs_within_0.05s'])} s. Run count equals the PDFEinfo "
      f"episode-count column in {flag_facts['trials_runs_match_pdfe_episode_count']} of "
      f"{flag_facts['trials_with_pdfe_episode_count']} trials.",
      f"- Subjects whose turning trials are flagged for 90 percent of their samples or more (mean flagged fraction): "
      f"{json.dumps(r3(flag_facts['subjects_mean_flagged_fraction_ge_0.9']))}.",
      f"- Runs whose boundaries differ from PDFEinfo by more than 0.05 s: "
      f"{json.dumps(r3(flag_facts['runs_boundary_diff_above_0.05s']))}; PDFEinfo intervals whose end precedes their "
      f"start: {json.dumps(r3(flag_facts['pdfe_intervals_end_before_start']))}.",
      f"- Merge rule: the shortest gap between two runs is {flag_facts['gap_s']['min']:.3f} s and "
      f"{flag_facts['gaps_below_s']['1.0']} gaps are shorter than 1 s, so a rule merging episodes closer than a fixed "
      f"gap, if one was applied, used a gap no longer than {flag_facts['gap_s']['min']:.3f} s; the flags cannot show "
      f"whether such a rule exists below that value. The shortest run is {flag_facts['run_s']['min']:.3f} s, so no "
      "minimum episode duration above it was applied.", "",
      "## 4. Subjects with at least one flagged episode", "",
      "| scope | subjects with IMU trial | with a flagged episode | subjects in PDFEinfo with the session | with a PDFEinfo episode |",
      "|---|---|---|---|---|"]
for r in sess_rows:
    md.append(f"| {r['scope']} | {r['subjects_with_imu_trial']} | {r['subjects_with_flagged_episode']} | "
              f"{r['subjects_in_pdfe_with_session']} | {r['subjects_with_pdfe_episode']} |")
md += ["", f"Subjects without any flagged sample in any turning trial (non-freezers here): {nonfreezers}.", "",
       "## 3. Freeze-index baseline", "",
       "Choice, fixed before scoring: the vertical (SI) shank acceleration in mg, as in Moore 2008 and the Daphnet "
       "reference do_test.m (axis 1, vertical); power in 3-8 Hz over power in 0.5-3 Hz; a 4 s rectangular window "
       f"(512 samples, {FRES} Hz bins, mean removed, |FFT|^2/N, trapezoid sums with step 1/fs as in x_fi.m) centred on "
       "each 2 s scoring window and shifted inside the record at its edges. Scoring windows tile each record from its "
       "first sample (2 s, 256 samples); a window is positive when at least half of its samples are flagged. AUROC is "
       f"threshold-free (rank statistic, ties averaged); the interval is the 2.5 and 97.5 percentiles of {N_BOOT} "
       f"subject-cluster bootstrap draws (subjects resampled with replacement, seed {SEED}).", "",
       "| variant | windows | positive | AUROC | 95% interval | per-subject AUROC median (n) |", "|---|---|---|---|---|---|"]
for r in auc_rows:
    md.append(f"| {r['variant']} | {r['windows']} | {r['positive_windows']} | {r['auroc']:.3f} | "
              f"{r['ci95_low']:.3f} to {r['ci95_high']:.3f} | {r['per_subject_median']:.3f} ({r['per_subject_n']}) |")
md += ["", "Primary: nominal_si (ungated). neg_power_si_posthoc is not a freeze index and not a baseline of the gate: it "
       "was added after the gated variants ranked below 0.5, to show how far low motion alone separates the flagged "
       "windows. Variant descriptions are in fi_auroc.csv.", "",
       f"Share of windows whose SI loco+freeze power lies below TH.power (the gate sets their index to 0), nominal bands: "
       f"positive turning windows {power_gate['nominal']['positive_windows_below_th_power']:.3f}, negative turning "
       f"windows {power_gate['nominal']['negative_windows_below_th_power']:.3f}, standing windows "
       f"{f3(power_gate['nominal']['standing_windows_below_th_power'])}; Daphnet indexing: positive "
       f"{power_gate['daphnet_index']['positive_windows_below_th_power']:.3f}, negative "
       f"{power_gate['daphnet_index']['negative_windows_below_th_power']:.3f}, standing "
       f"{f3(power_gate['daphnet_index']['standing_windows_below_th_power'])}.", "",
       f"Thresholds for alarms come from "
       f"the Daphnet reference script do_test.m read inside daphnet.zip (fixed on another data set): TH.freeze values "
       f"{TH_FREEZE}, TH.power 2^{TH_POWER_EXP} on the summed band power in mg^2 units of x_fi.m.", "",
       "| variant | TH.freeze | sensitivity | specificity | standing alarms/min (events/min) | non-freezer alarms/min (events/min) |",
       "|---|---|---|---|---|---|"]
for r in thr_rows:
    md.append(f"| {r['variant']} | {r['th_freeze']} | {r['sensitivity_windows']:.3f} | {r['specificity_windows']:.3f} | "
              f"{r['standing_alarms_per_min']:.3f} ({r['standing_events_per_min']:.3f}) | "
              f"{r['nonfreezer_alarms_per_min']:.3f} ({r['nonfreezer_events_per_min']:.3f}) |")
md += ["", f"Standing: {len(SW)} windows ({len(SW) * SCORE_WIN / FS / 60.0:.2f} min); non-freezer turning trials: "
       f"{len(NF)} windows ({len(NF) * SCORE_WIN / FS / 60.0:.2f} min).", "",
       "## 2. Turns under the frozen rule", "",
       f"Rule: turn_rule.py.txt, SHA-256 {rule_sha}, frozen at {freeze_info.get('frozen_at')} before any turn statistic "
       f"was computed (files present at freeze: {freeze_info.get('turn_statistic_files_present_at_freeze')}). Direction +1 "
       "is a positive rotation about the SI axis; with SI up and a right-handed frame that is counterclockwise seen from "
       "above, a turn to the left. The handedness of the exported frame is not verified by this task, so the left or "
       "right label of a sign is an assumption; the alternation counts do not depend on it.", "",
       f"- Bias source per turning trial: {json.dumps(turn_facts['bias_source_counts'])}; bias (deg/s): "
       f"{dline(turn_facts['bias_dps'])}.",
       f"- Sweeps: {turn_facts['sweeps']}, of which partial (below 270 deg) {turn_facts['partial_sweeps']} and holding "
       f"two turns or more {turn_facts['multi_turn_sweeps']}. Amplitude of single-turn sweeps (deg): "
       f"{dline(turn_facts['sweep_amplitude_deg_single_turn_sweeps'])}. Amplitude of partial sweeps (deg): "
       f"{dline(turn_facts['sweep_amplitude_deg_partial_sweeps'])}; partial sweeps of 180 deg or more: "
       f"{turn_facts['partial_sweeps_180_or_more']} (turns the rule does not count; single-turn amplitudes below 360 deg "
       "are expected where the shank tilt projects the heading rotation onto SI with a cosine factor).",
       f"- Trials whose bias came from their own still windows: {turn_facts['trials_bias_from_record']}; trials with no "
       f"bias (no still windows in the record nor in the standing file): {turn_facts['trials_bias_none']}.",
       f"- Turns: {turn_facts['turns']} ({turn_facts['complete_turns']} complete, {turn_facts['truncated_turns']} "
       f"truncated by the record edges); trials without any turn: {turn_facts['trials_without_turn']} "
       f"({json.dumps(r3(turn_facts['trials_without_turn_list']))}, flagged fraction given).",
       f"- Turns per trial: {dline(turn_facts['turns_per_trial'])}. Complete turns per trial: "
       f"{dline(turn_facts['complete_turns_per_trial'])}.",
       f"- Direction alternation: {turn_facts['alternating_pairs']} of {turn_facts['turn_pairs']} consecutive turn pairs "
       f"alternate; {turn_facts['trials_fully_alternating']} of {turn_facts['trials_with_pairs']} trials with a pair "
       f"alternate throughout. Direction counts {json.dumps(turn_facts['direction_counts'])}; first turn direction by "
       f"more affected side {json.dumps(turn_facts['first_turn_direction_by_side'])}.",
       f"- Complete turn duration (s): {dline(turn_facts['complete_turn_duration_s'])}.",
       f"- Freezing-free complete turns (no flagged sample between onset and offset): "
       f"{turn_facts['complete_freezing_free_turns']} in {turn_facts['trials_with_freezing_free_complete_turn']} trials of "
       f"{turn_facts['subjects_with_freezing_free_complete_turn']} subjects; duration (s): "
       f"{dline(turn_facts['complete_freezing_free_turn_duration_s'])}. Per subject: "
       f"{dline(turn_facts['freezing_free_complete_turns_per_subject'])}; subjects with at least 5: "
       f"{turn_facts['subjects_with_at_least_5_freezing_free_complete_turns']}.",
       f"- Complete turns holding freezing: {turn_facts['complete_turns_with_freezing']}; duration (s): "
       f"{dline(turn_facts['complete_turn_with_freezing_duration_s'])}.",
       f"- Share of flagged samples that fall inside a detected turn (pooled): "
       f"{f3(turn_facts['flagged_fraction_inside_turns_pooled'])}.",
       f"- Standing files: {turn_facts['standing_files']}, sweeps detected {turn_facts['standing_sweeps']}, turns "
       f"detected {turn_facts['standing_turns']}.", ""]
text = "\n".join(md) + "\n"
assert "\u2014" not in text
with open(os.path.join(OUT, "fog_imu_summary.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
log("wrote fog_imu_summary.md")

write_provenance(args, {"imu_zip": f"{RAW}/IMU.zip", "imu_zip_bytes": os.path.getsize(f"{RAW}/IMU.zip"),
                        "pdfeinfo": f"{RAW}/PDFEinfo.csv", "daphnet_member": f"{DAPHNET_ZIP}::{DAPHNET_TEST}",
                        "daphnet_do_test_sha256": hashlib.sha256(do_test_bytes).hexdigest(),
                        "turn_rule_sha256": rule_sha, "turn_rule_frozen_at": freeze_info.get("frozen_at"),
                        "seed": SEED, "bootstrap_draws": N_BOOT, "videos_opened": False})
log("done")
