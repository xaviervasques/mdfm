"""oxford_dystonia_lfp structure probe: side-effect steps as OPEN-36 evidence, the content of the
five Single files and the Paired file, and the Q1 admissible-null counts of NR-6.1.7.

Task T11 of the weekend plan 2026-10-01. Rule: reports/oxford_nonramp/rule.txt, written and hashed
(rule.sha256) before the first data read. The script stops before reading any input when the SHA-256
of rule.txt differs from RULE_SHA256 or from rule.sha256.

Structure only. The script reads the README.txt members of the three zips, the unipolar LFP channels
(L1 to L8, R1 to R8) and the command channels (Am1, Am2, Cur1, Cur2) of the six non-ramp .mat members
(decompressed in memory, never extracted; the other channels are never read), and the tables that
scripts/probe_oxford_lfp.py wrote in reports/oxford_lfp/. It computes no accelerometer, EMG, LFP-power
or body metric, per block, per command state or otherwise. Nothing it writes fills a value of
docs/rules/NEURAL_RULES_DRAFT.md (NR-0.5).

Parts (letters of rule.txt):
  A  side-effect step of each of the 62 ramp files, inferred from the README stopping rule and the
     command steps; evidence for OPEN-36 only.
  B  artefact pulses of the 5 Single files and L7_L2_Paired.mat: pulse count, inter-pulse intervals,
     isolated pulses, pairs and trains with their rates, command channel, NR-6.2.1 anchor verdict,
     contact-level reading (P5) from flat channels and the artefact geometry. Reporting only, outside
     rule.txt and used by no verdict: pulse widths and strict-threshold splits per pair, train and series,
     and the pairs at most SHORT_PAIR_S apart (intervals, runs at 1 ms resolution, blank channels in
     their span).
  C  Q1 admissible-null k = prod_h (n_h! / prod_l m_{h,l}!) - 1 per scope and variant, with the
     smallest attainable p = 1 / (k + 1).

Outputs:
  reports/oxford_nonramp/files.csv      one row per non-ramp file (part B)
  reports/oxford_nonramp/trains.csv     one row per pair or train (bursts of 2 or more pulses)
  reports/oxford_nonramp/series.csv     series of isolated pulses and peak-channel runs
  reports/oxford_nonramp/channels.csv   one row per non-ramp file and LFP channel
  reports/oxford_nonramp/summary.md     the same in prose, every number from the rows above
  reports/oxford_nonramp/_provenance.json
  reports/oxford_lfp/side_effect_steps.csv   (part A) with _provenance_side_effect_steps.json
  reports/admissible_nulls/oxford_q1.csv     (part C) with _provenance_oxford_q1.json
Every provenance file carries the stamp of scripts/moveatlas_stamp.py, the rule hash and the SHA-256
of every output; no other output carries a time stamp, so two runs give identical bytes.

Run (one process, BelowNormal priority set by the script; it waits while cache/runs/queue/SLOW exists):
  python scripts/probe_oxford_structure.py [--moveatlas <moveatlas>]
"""
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_args  # noqa: E402
from moveatlas_stamp import stamp  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_REL = "datasets/eeg/oxford_dystonia_lfp/raw"
OUT = os.path.join(TREE, "reports", "oxford_nonramp")
OXF = os.path.join(TREE, "reports", "oxford_lfp")
NULLS = os.path.join(TREE, "reports", "admissible_nulls")
RULE = os.path.join(OUT, "rule.txt")
RULE_HASH_FILE = os.path.join(OUT, "rule.sha256")
RULE_SHA256 = "70b6288f57bfd5de45433604d4ca35a4ad5abe5ab4878629dcc8d71f111e7e68"
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")

# Fixed parameters (rule.txt D3).
THR_MAD = 20.0          # B2: supra-threshold when |dx| > THR_MAD x median |dx| of the channel
THR_STRICT = 100.0      # B6: sensitivity count only
N_COINC = 2             # B3: channels supra-threshold together
COINC_TOL = 1           # B3: samples
MERGE = 8               # B4: samples between pulse samples of one event
PEAK_PAD = 2            # B5: samples around an event for its peak
G_ISO_S = 0.5           # B8: isolated-pulse gap
G_SENS_S = (0.1, 1.0)   # B8: sensitivity gaps
G_SERIES_S = 10.0       # B10
BLANK_FRAC = 0.99       # B10
N_ANCHOR = 25           # B12, README: 25 pulses per contact level
ENUM_MAX = 10 ** 6      # C5: probe convention, not a rule value
SHORT_PAIR_S = 0.010    # reporting only (added after the first run, outside rule.txt): pairs at most 10 ms apart
IPI_BINS = [("ipi_le_5ms", 0.0, 0.005), ("ipi_5_10ms", 0.005, 0.010), ("ipi_10_100ms", 0.010, 0.100),
            ("ipi_100_500ms", 0.100, 0.500), ("ipi_0p5_2s", 0.5, 2.0), ("ipi_2_10s", 2.0, 10.0),
            ("ipi_over_10s", 10.0, math.inf)]
PARAMS = {"THR_MAD": THR_MAD, "THR_STRICT": THR_STRICT, "N_COINC": N_COINC, "COINC_TOL": COINC_TOL,
          "MERGE": MERGE, "PEAK_PAD": PEAK_PAD, "G_ISO_S": G_ISO_S, "G_SENS_S": list(G_SENS_S),
          "G_SERIES_S": G_SERIES_S, "BLANK_FRAC": BLANK_FRAC, "N_ANCHOR": N_ANCHOR, "ENUM_MAX": ENUM_MAX,
          "SHORT_PAIR_S (reporting only, not in rule.txt)": SHORT_PAIR_S}

LFP_RE = re.compile(r"[LR][1-8]")
CMD_RE = re.compile(r"(Am|Cur)[12]")
OPEN06 = ("interrupted", "no_artefact")
UNDET = "not reached or ambiguous"
EVIDENCE = "evidence for OPEN-36 only; not a filled value (NR-0.5)"


# ----------------------------------------------------------------------------- guards

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_rule_hash():
    """Stop before any input is read when rule.txt is not the hashed version."""
    got = sha256_file(RULE)
    with open(RULE_HASH_FILE, encoding="utf-8") as f:
        side = f.read().split()[0]
    if got != RULE_SHA256 or side != RULE_SHA256:
        raise SystemExit(f"rule.txt hash {got} differs from RULE_SHA256 {RULE_SHA256} or rule.sha256 {side}: stop")
    return got


def below_normal():
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


def wait_if_slow():
    told = False
    while os.path.exists(SLOW):
        if not told:
            print("paused while cache/runs/queue/SLOW exists", flush=True)
            told = True
        time.sleep(30)


# ----------------------------------------------------------------------------- io helpers

def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    print("wrote", os.path.relpath(path, TREE), len(rows), "rows", flush=True)


def rnd(v, k=3):
    return None if v is None else round(float(v), k)


def fmt(v, k=1):
    return "none" if v is None else f"{v:.{k}f}"


def rel(path):
    return os.path.relpath(path, TREE).replace("\\", "/")


# ----------------------------------------------------------------------------- README (A0)

def read_readmes(zips):
    """README.txt of each zip: member, SHA-256, lines; the stopping-rule and single-pulse lines."""
    out = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                base = n.split("/")[-1]
                if base == "README.txt" and "__MACOSX" not in n:
                    b = zf.read(n)
                    out.append({"zip": os.path.basename(z), "member": n, "sha256": hashlib.sha256(b).hexdigest(),
                                "lines": b.decode("utf-8", "replace").splitlines()})
    if len(out) != len(zips) or len({r["sha256"] for r in out}) != 1:
        raise SystemExit("README.txt copies missing or not identical: stop")
    lines = out[0]["lines"]
    stop_no = [i + 1 for i, line in enumerate(lines) if "side effect threshold" in line.lower()]
    single_no = [i + 1 for i, line in enumerate(lines) if "single pulse stimulation" in line.lower()]
    if len(stop_no) != 1 or len(single_no) != 1:
        raise SystemExit("README stopping-rule or single-pulse line not found once: stop")
    stop_line, single_line = lines[stop_no[0] - 1], lines[single_no[0] - 1]
    m = re.search(r"steps of ([0-9.]+) mA until either the side effect threshold or ([0-9.]+) mA", stop_line)
    p = re.search(r"\((\d+) pulses\)", single_line)
    if not m or not p:
        raise SystemExit("README step, cap or pulse count not parsed: stop")
    step, cap, n_pulses = float(m.group(1)), float(m.group(2)), int(p.group(1))
    if n_pulses != N_ANCHOR:
        raise SystemExit(f"README pulse count {n_pulses} differs from N_ANCHOR {N_ANCHOR}: stop")
    frag = re.search(r"until .*? reached", stop_line)
    cite = "; ".join(f"{r['zip']}:{r['member']} line {stop_no[0]}" for r in out)
    cite_single = "; ".join(f"{r['zip']}:{r['member']} line {single_no[0]}" for r in out)
    return {"copies": [{k: r[k] for k in ("zip", "member", "sha256")} for r in out], "sha256": out[0]["sha256"],
            "stop_line": stop_no[0], "single_line": single_no[0], "step_mA": step, "cap_mA": cap,
            "pulses_per_level": n_pulses, "fragment": frag.group(0) if frag else "", "cite": cite,
            "cite_single": cite_single}


# ----------------------------------------------------------------------------- part A

def side_effect_rows(summ, g15, readme):
    step, cap = readme["step_mA"], readme["cap_mA"]
    blocks = {}
    for g in g15:
        blocks.setdefault(g["file"], []).append((int(g["on_block"]), float(g["command_on_s"]), g["delivery"]))
    rows = []
    for s in summ:
        f = s["file"]
        bl = sorted(blocks[f])
        amps = [float(a) for a in s["amplitudes_mA"].split()]
        hi = max(amps)
        nondec = all(amps[i + 1] >= amps[i] - 1e-9 for i in range(len(amps) - 1))
        distinct = sorted({round(a, 6) for a in amps})
        expected = [round(step * (i + 1), 6) for i in range(int(round(hi / step)))]
        contiguous = distinct == expected
        last_del = bl[-1][2]
        others = [b[1] for b in bl[:-1]]
        n06 = sum(1 for b in bl if b[2] in OPEN06)
        row = {"file": f, "patient": s["patient"], "hemisphere": s["hemisphere"], "contact_level": s["contact_level"],
               "step_pattern": s["step_pattern"], "n_on_blocks": len(bl), "amplitudes_mA": s["amplitudes_mA"],
               "highest_mA": hi, "nondecreasing": nondec, "steps_contiguous_0p5mA": contiguous,
               "last_block_delivery": last_del, "last_block_on_s": rnd(bl[-1][1], 1),
               "other_blocks_on_s_median": rnd(np.median(others), 1) if others else None,
               "n_repeat_amplitude": int(s["n_repeat_amplitude"]), "n_open06_blocks": n06}
        inferred = below = None
        if len(bl) == 1:
            status, reason = UNDET, "single_block"
        elif hi >= cap - 1e-9:
            status, reason = UNDET, "reached_4p5_cap"
        else:
            fails = []
            if not nondec:
                fails.append("amplitudes_not_nondecreasing")
            if not contiguous:
                fails.append("steps_not_contiguous_0p5mA")
            if last_del != "complete":
                fails.append(f"last_block_{last_del}")
            if fails:
                status, reason = UNDET, " ".join(fails)
            else:
                status, reason = "inferred", "ramp stopped below the cap"
                inferred, below = hi, round(hi - step, 6)
        row.update({"status": status, "inferred_side_effect_step_mA": inferred, "step_below_mA": below,
                    "reason": reason, "readme_citation": readme["cite"], "readme_sha256": readme["sha256"],
                    "use": EVIDENCE})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- part C

def q1_rows(summ, g15, mh):
    head = {(m["patient"], m["hemisphere"]) for m in mh
            if int(m["n_ramp"]) > 0 and m["n_ramp_with_head_accelerometer"] == m["n_ramp"]}
    deliveries = {}
    for g in g15:
        deliveries.setdefault(g["file"], []).append(g["delivery"])
    rows = []
    for scope in ("q1", "all_ramp"):
        for files in ("all", "ramps_only"):
            for open06 in ("with_open06", "without_open06"):
                per_h, files_out = {}, []
                n_adm = n_out = 0
                for s in summ:
                    hk = (s["patient"], s["hemisphere"])
                    lvl = int(s["contact_level"])
                    if scope == "q1" and (hk not in head or not 2 <= lvl <= 7):
                        continue
                    if files == "ramps_only" and s["step_pattern"] != "ascending_ramp":
                        continue
                    d = deliveries[s["file"]]
                    adm = [x for x in d if open06 == "with_open06" or x not in OPEN06]
                    n_adm += len(adm)
                    n_out += len(d) - len(adm)
                    if not adm:
                        files_out.append(s["file"])
                        continue
                    per_h.setdefault(hk, []).append(lvl)
                total = 1
                parts = []
                for hk in sorted(per_h, key=lambda t: (int(t[0][1:]), t[1])):
                    levels = per_h[hk]
                    factor = math.factorial(len(levels))
                    for lv in set(levels):
                        factor //= math.factorial(levels.count(lv))
                    total *= factor
                    parts.append(f"{hk[0]}{hk[1]}: n={len(levels)} levels {' '.join(str(v) for v in sorted(levels))} factor {factor}")
                k = total - 1
                rows.append({"scope": scope, "files": files, "open06": open06,
                             "n_patients": len({h[0] for h in per_h}), "n_hemispheres": len(per_h),
                             "n_files": sum(len(v) for v in per_h.values()), "n_on_blocks_admitted": n_adm,
                             "n_on_blocks_left_out": n_out, "files_left_out": " ".join(files_out),
                             "hemispheres": "; ".join(parts), "k": str(k),
                             "log10_k": rnd(math.log10(k), 3) if k > 0 else None,
                             "smallest_p": f"{1.0 / (k + 1):.6g}", "pass_decision_possible_k_ge_20": k >= 20,
                             "enumeration_feasible_k_le_1e6": k <= ENUM_MAX,
                             "draws_needed_open33": k > ENUM_MAX})
    return rows


# ----------------------------------------------------------------------------- part B: reading

def _h5_text(obj):
    a = np.asarray(obj[()]).ravel()
    return "".join(chr(int(c)) for c in a if int(c) != 0)


def read_lfp_and_command(raw):
    """Fs, all labels, and {index: samples} for the unipolar LFP and command channels only."""
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as f:
        S = f["SmrData"]
        fs = float(np.asarray(S["Fs"][()]).ravel()[0])
        labels = [_h5_text(f[r]) for r in np.asarray(S["WvTits"][()]).ravel()]
        ds = S["WvData"]
        C = len(labels)
        if ds.shape[0] == C and ds.shape[1] != C:
            by_row = True
        elif len(ds.shape) == 2 and ds.shape[1] == C:
            by_row = False
        else:
            raise ValueError(f"WvData shape {ds.shape} does not match {C} labels")
        want = [j for j, lab in enumerate(labels) if LFP_RE.fullmatch(lab) or CMD_RE.fullmatch(lab)]
        X = {j: np.asarray(ds[j, :] if by_row else ds[:, j], dtype=np.float64) for j in want}
    return fs, labels, X


def absdiff(x):
    dx = np.empty_like(x)
    dx[0] = 0.0
    np.abs(np.diff(x), out=dx[1:])
    return dx


def detect(det, thr):
    """Event starts and ends (B2 to B4) on the detection channels at THR x s_c."""
    n = len(det[0]["dx"])
    count = np.zeros(n, np.int16)
    for d in det:
        m = d["dx"] > thr * d["s"]
        md = m.copy()
        for sh in range(1, COINC_TOL + 1):
            md[sh:] |= m[:-sh]
            md[:-sh] |= m[sh:]
        count += md
    ps = np.flatnonzero(count >= N_COINC)
    if len(ps) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    br = np.flatnonzero(np.diff(ps) > MERGE) + 1
    starts = ps[np.concatenate(([0], br))].astype(np.int64)
    ends = ps[np.concatenate((br - 1, [len(ps) - 1]))].astype(np.int64)
    return starts, ends


def isolated_mask(t, gap_samples):
    n = len(t)
    if n == 0:
        return np.zeros(0, bool)
    g = np.diff(t)
    prev_ok = np.ones(n, bool)
    next_ok = np.ones(n, bool)
    prev_ok[1:] = g > gap_samples
    next_ok[:-1] = g > gap_samples
    return prev_ok & next_ok


def runs(flags_or_keys):
    """Maximal runs of equal consecutive values: list of (i0, i1) inclusive."""
    out = []
    i = 0
    n = len(flags_or_keys)
    while i < n:
        k = i
        while k + 1 < n and flags_or_keys[k + 1] == flags_or_keys[i]:
            k += 1
        out.append((i, k))
        i = k + 1
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    r = spearmanr(a, b).correlation
    return None if r is None or np.isnan(r) else round(float(r), 3)


def modal(labels):
    if not labels:
        return "", None
    vals, cnt = np.unique(np.asarray(labels), return_counts=True)
    i = int(np.argmax(cnt))
    return str(vals[i]), round(float(cnt[i]) / len(labels), 3)


# ----------------------------------------------------------------------------- part B: one file

def analyze_file(zpath, zinfo, inv):
    t0 = time.time()
    raw = zipfile.ZipFile(zpath).read(zinfo)
    fs, labels, X = read_lfp_and_command(raw)
    del raw
    base = zinfo.filename.split("/")[-1]
    hemi = inv["hemisphere"]
    n = len(next(iter(X.values())))
    # channels (B1, B2)
    chans = []
    for j in sorted(X):
        lab = labels[j]
        if not LFP_RE.fullmatch(lab):
            continue
        x = X[j]
        d = {"label": lab, "side": lab[0], "level": int(lab[1]), "flat": bool(x.max() == x.min())}
        if not d["flat"]:
            d["dx"] = absdiff(x)
            d["s"] = float(np.median(d["dx"]))
            d["quiet"] = d["s"] == 0.0
        else:
            d["s"] = 0.0
            d["quiet"] = False
        chans.append(d)
    det = [d for d in chans if not d["flat"] and not d["quiet"]]
    # command channels (B11)
    cmd_labels = [labels[j] for j in sorted(X) if CMD_RE.fullmatch(labels[j])]
    cmd_active = [labels[j] for j in sorted(X) if CMD_RE.fullmatch(labels[j]) and np.any(X[j] != 0)]
    cmd = X[labels.index(cmd_active[0])] if cmd_active else None
    # events (B2 to B6)
    starts, ends = detect(det, THR_MAD) if len(det) >= N_COINC else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    s_strict, _ = detect(det, THR_STRICT) if len(det) >= N_COINC else (np.zeros(0, np.int64), None)
    t = starts
    npulse = len(t)
    P = np.zeros((npulse, len(det)))
    if npulse:
        lo = np.maximum(starts - PEAK_PAD, 0)
        hi = np.minimum(ends + PEAK_PAD + 1, n)
        idx = np.empty(2 * npulse, np.int64)
        idx[0::2] = lo
        idx[1::2] = hi
        for c, d in enumerate(det):
            P[:, c] = np.maximum.reduceat(np.append(d["dx"], 0.0), idx)[0::2]
    peak_c = np.argmax(P, axis=1) if npulse else np.zeros(0, int)
    peak_lab = [det[c]["label"] for c in peak_c]
    s_vec = np.array([d["s"] for d in det])
    peak_ratio = P[np.arange(npulse), peak_c] / s_vec[peak_c] if npulse else np.zeros(0)
    widths = ends - starts
    ipi = np.diff(t) / fs
    iso = isolated_mask(t, G_ISO_S * fs)
    iso_sens = {g: int(isolated_mask(t, g * fs).sum()) for g in G_SENS_S}
    iso_strict = int(isolated_mask(s_strict, G_ISO_S * fs).sum())
    # diagnostic of the sensitivity count: strict events that start inside a primary event span; a pulse
    # holding two or more strict starts may be two stimuli closer than the merge window (B4)
    nstrict = np.zeros(npulse, np.int64)
    if npulse and len(s_strict):
        pos = np.searchsorted(starts, s_strict, side="right") - 1
        inside = (pos >= 0) & (s_strict <= ends[np.maximum(pos, 0)])
        strict_inside = int(inside.sum())
        nstrict = np.bincount(pos[inside], minlength=npulse)
        holding2 = int(np.count_nonzero(nstrict >= 2))
    else:
        strict_inside = holding2 = 0
    split = nstrict >= 2
    # bursts (B9)
    br = np.flatnonzero(np.diff(t) > G_ISO_S * fs) + 1 if npulse else np.zeros(0, int)
    b0 = np.concatenate(([0], br)) if npulse else np.zeros(0, int)
    b1 = np.concatenate((br - 1, [npulse - 1])) if npulse else np.zeros(0, int)
    trains = []
    short_pairs = []   # (first pulse sample, interval in samples) of pairs at most SHORT_PAIR_S apart
    sp_widths = []     # widths of their pulses
    for i0, i1 in zip(b0, b1):
        m = int(i1 - i0 + 1)
        if m < 2:
            continue
        tt = t[i0:i1 + 1]
        g = np.diff(tt) / fs
        lab, share = modal(peak_lab[i0:i1 + 1])
        if m == 2 and tt[1] - tt[0] <= SHORT_PAIR_S * fs:
            short_pairs.append((int(tt[0]), int(tt[1] - tt[0])))
            sp_widths.extend(int(w) for w in widths[i0:i1 + 1])
        row = {"file": base, "burst": len(trains), "kind": "pair" if m == 2 else "train", "n_pulses": m,
               "start_s": rnd(tt[0] / fs, 4), "end_s": rnd(tt[-1] / fs, 4), "duration_s": rnd((tt[-1] - tt[0]) / fs, 4),
               "rate_hz": rnd((m - 1) * fs / (tt[-1] - tt[0]), 3), "ipi_median_ms": rnd(1000 * np.median(g), 3),
               "ipi_min_ms": rnd(1000 * g.min(), 3), "ipi_max_ms": rnd(1000 * g.max(), 3),
               "modal_peak_channel": lab, "modal_share": share,
               "width_samples_median": rnd(np.median(widths[i0:i1 + 1]), 1),
               "width_samples_max": int(widths[i0:i1 + 1].max()), "n_split_at_strict": int(split[i0:i1 + 1].sum())}
        if cmd is not None:
            cv = cmd[tt]
            row["command_values_uA"] = " ".join(f"{v:g}" for v in np.unique(cv))
        trains.append(row)
    # series and peak-channel runs of isolated pulses (B10); consecutive = adjacent in the pulse sequence
    series = []
    i = 0
    while i < npulse:
        if not iso[i]:
            i += 1
            continue
        k = i
        while k + 1 < npulse and iso[k + 1] and (t[k + 1] - t[k]) <= G_SERIES_S * fs:
            k += 1
        tt = t[i:k + 1]
        g = np.diff(tt) / fs
        lab, share = modal(peak_lab[i:k + 1])
        blank = []
        if k > i:
            for d in chans:
                if d["side"] != hemi:
                    continue
                if d["flat"]:
                    blank.append(d["label"])
                    continue
                seg = d["dx"][tt[0]:tt[-1] + 1]
                if np.mean(seg == 0) >= BLANK_FRAC:
                    blank.append(d["label"])
        series.append({"file": base, "row_kind": "series", "index": len(series), "n_pulses": int(k - i + 1),
                       "start_s": rnd(tt[0] / fs, 4), "end_s": rnd(tt[-1] / fs, 4),
                       "ipi_median_s": rnd(np.median(g), 4) if len(g) else None,
                       "ipi_min_s": rnd(g.min(), 4) if len(g) else None, "ipi_max_s": rnd(g.max(), 4) if len(g) else None,
                       "rate_hz": rnd(1.0 / np.median(g), 4) if len(g) else None,
                       "modal_peak_channel": lab, "modal_share": share,
                       "peak_ratio_median": rnd(np.median(peak_ratio[i:k + 1]), 1),
                       "blank_channels": " ".join(blank) if k > i else None,
                       "width_samples_median": rnd(np.median(widths[i:k + 1]), 1),
                       "width_samples_max": int(widths[i:k + 1].max()), "n_split_at_strict": int(split[i:k + 1].sum())})
        i = k + 1
    n_series = len(series)
    # short pairs (reporting only, no verdict uses them): intervals in samples, runs of consecutive short pairs
    # with the same interval rounded to 1 ms (other pulses between two short pairs do not break a run), and
    # the blank channels of the isolated-pulse series that overlap their span
    sp_tab, sp_runs, sp_blank = {}, [], []
    for s0, d0 in short_pairs:
        sp_tab[d0] = sp_tab.get(d0, 0) + 1
        ms = int(round(1000.0 * d0 / fs))
        if sp_runs and sp_runs[-1]["ms"] == ms:
            sp_runs[-1]["n"] += 1
            sp_runs[-1]["end"] = s0
        else:
            sp_runs.append({"ms": ms, "n": 1, "start": s0, "end": s0})
    if short_pairs:
        a0, a1 = short_pairs[0][0] / fs, short_pairs[-1][0] / fs
        for s_ in series:
            if s_["blank_channels"] and s_["start_s"] <= a1 and s_["end_s"] >= a0 and s_["blank_channels"] not in sp_blank:
                sp_blank.append(s_["blank_channels"])
    # isolated pulses tabulated by the blank channel of their series (B10, B13: the blank channel is the
    # contact that is not recorded while the series runs)
    by_blank, blank_seq = {}, []
    for s_ in series:
        if s_["blank_channels"]:
            b = s_["blank_channels"].replace(" ", "+")
            by_blank[b] = by_blank.get(b, 0) + s_["n_pulses"]
            if not blank_seq or blank_seq[-1] != b:
                blank_seq.append(b)
    keys = [(peak_lab[q] if iso[q] else None) for q in range(npulse)]
    pruns = []
    for i0, i1 in runs(keys):
        if keys[i0] is None:
            continue
        tt = t[i0:i1 + 1]
        pruns.append({"file": base, "row_kind": "peak_channel_run", "index": len(pruns), "n_pulses": int(i1 - i0 + 1),
                      "start_s": rnd(tt[0] / fs, 4), "end_s": rnd(tt[-1] / fs, 4), "modal_peak_channel": keys[i0],
                      "modal_share": 1.0})
    # channel table (B13 b)
    ch_rows = []
    med_peak = {}
    det_index = {d["label"]: c for c, d in enumerate(det)}
    for d in chans:
        r = {"file": base, "channel": d["label"], "side": d["side"],
             "recorded_hemisphere": d["side"] == hemi, "level": d["level"], "flat": d["flat"], "quiet": d["quiet"],
             "median_absdiff": rnd(d["s"], 4)}
        if d["label"] in det_index:
            c = det_index[d["label"]]
            r["n_supra_threshold_samples"] = int(np.count_nonzero(d["dx"] > THR_MAD * d["s"]))
            if npulse:
                med_peak[d["label"]] = float(np.median(P[:, c]))
                r["median_peak_absdiff_at_pulses"] = rnd(med_peak[d["label"]], 2)
                r["median_peak_ratio_at_pulses"] = rnd(np.median(P[:, c] / d["s"]), 1)
                r["n_pulses_as_peak_channel"] = int(np.count_nonzero(peak_c == c))
                r["share_as_peak_channel"] = rnd(np.mean(peak_c == c), 3)
        ch_rows.append(r)
    rec = [d for d in det if d["side"] == hemi and d["label"] in med_peak]
    ranked = sorted(rec, key=lambda d: (-med_peak[d["label"]], d["label"]))
    for r in ch_rows:
        lab = r["channel"]
        if lab in [d["label"] for d in ranked]:
            r["rank_in_recorded_hemisphere"] = [d["label"] for d in ranked].index(lab) + 1
    # contact-level reading (B13 c, d)
    fq = sorted(d["level"] for d in chans if d["side"] == hemi and (d["flat"] or d["quiet"]))
    top2 = [d["label"] for d in ranked[:2]]
    reading, rho, kk = "no reading", None, None
    if len(fq) == 1:
        kk = fq[0]
        nearest = sorted((lv for lv in range(1, 9) if lv != kk), key=lambda lv: (abs(lv - kk), lv))[:2]
        if sorted(int(lab[1]) for lab in top2) == sorted(nearest):
            reading = f"consistent with stimulation at level {kk}"
        rho = spearman([abs(d["level"] - kk) for d in ranked], [med_peak[d["label"]] for d in ranked])
    # command (B11)
    cmd_vals = cmd_segs = in_on = in_off = None
    if cmd is not None:
        cmd_vals = " ".join(f"{v:g}" for v in np.unique(cmd[cmd != 0]))
        cmd_segs = int(np.count_nonzero(cmd[1:] != cmd[:-1]) + 1)
        in_on = int(np.count_nonzero(cmd[t] != 0)) if npulse else 0
        in_off = npulse - in_on
    # verdict (B12)
    n_iso = int(iso.sum())
    emg = inv.get("neck_emg", "").strip()
    if n_iso == 0:
        verdict = "no: no isolated pulse"
    elif not emg:
        verdict = "no: no EMG channel"
    elif n_iso >= N_ANCHOR:
        verdict = "anchor possible"
    else:
        verdict = "insufficient"
    q = lambda p: rnd(1000 * np.percentile(ipi, p), 3) if len(ipi) else None  # noqa: E731
    pairs = [r for r in trains if r["kind"] == "pair"]
    trs = [r for r in trains if r["kind"] == "train"]
    lab_all, share_all = modal(peak_lab)
    row = {"file": base, "member": zinfo.filename, "crc32": f"{zinfo.CRC:08x}", "patient": inv["patient"],
           "hemisphere": hemi, "kind": inv["kind"], "contact_level_from_name": inv["contact_level"],
           "fs_hz": fs, "n_samples": n, "duration_s": rnd(n / fs, 3),
           "lfp_flat": " ".join(d["label"] for d in chans if d["flat"]),
           "lfp_quiet": " ".join(d["label"] for d in chans if d["quiet"]), "n_detection_channels": len(det),
           "command_labels": " ".join(cmd_labels), "command_present": bool(cmd_labels),
           "command_active": " ".join(cmd_active), "command_values_uA": cmd_vals, "command_segments": cmd_segs,
           "n_pulses": npulse, "n_pulses_strict": int(len(s_strict)),
           "n_strict_starting_inside_a_pulse": strict_inside,
           "n_pulses_holding_2plus_strict": holding2,
           "pulses_at_command_nonzero": in_on, "pulses_at_command_zero": in_off,
           "width_samples_median": rnd(np.median(widths), 1) if npulse else None,
           "width_samples_max": int(widths.max()) if npulse else None,
           "peak_ratio_p1": rnd(np.percentile(peak_ratio, 1), 1) if npulse else None,
           "peak_ratio_median": rnd(np.median(peak_ratio), 1) if npulse else None,
           "ipi_min_ms": rnd(1000 * ipi.min(), 3) if len(ipi) else None, "ipi_p5_ms": q(5), "ipi_median_ms": q(50),
           "ipi_p95_ms": q(95), "ipi_max_ms": rnd(1000 * ipi.max(), 3) if len(ipi) else None}
    for name, a, b in IPI_BINS:
        row[name] = int(np.count_nonzero((ipi > a) & (ipi <= b)))
    row.update({"n_isolated": n_iso, "n_isolated_gap_0p1s": iso_sens[G_SENS_S[0]],
                "n_isolated_gap_1p0s": iso_sens[G_SENS_S[1]], "n_isolated_strict": iso_strict,
                "n_isolated_split_at_strict": int(np.count_nonzero(iso & split)),
                "n_bursts": int(len(b0)), "n_pairs": len(pairs),
                "pair_interval_ms_min": min(r["ipi_min_ms"] for r in pairs) if pairs else None,
                "pair_interval_ms_median": rnd(np.median([r["ipi_median_ms"] for r in pairs]), 3) if pairs else None,
                "pair_interval_ms_max": max(r["ipi_max_ms"] for r in pairs) if pairs else None,
                "n_short_pairs": len(short_pairs),
                "short_pairs_first_s": rnd(short_pairs[0][0] / fs, 3) if short_pairs else None,
                "short_pairs_last_s": rnd(short_pairs[-1][0] / fs, 3) if short_pairs else None,
                "short_pair_interval_samples": " ".join(f"{k}:{v}" for k, v in sorted(sp_tab.items())),
                "short_pair_pulse_width_samples_median": rnd(np.median(sp_widths), 1) if sp_widths else None,
                "short_pair_pulse_width_samples_max": max(sp_widths) if sp_widths else None,
                "short_pair_runs_1ms": " ".join(f"{r['ms']}ms:{r['n']}@{r['start'] / fs:.1f}-{r['end'] / fs:.1f}s"
                                                for r in sp_runs),
                "short_pairs_blank_channels": " ".join(sp_blank),
                "n_trains": len(trs), "n_pulses_in_trains": sum(r["n_pulses"] for r in trs),
                "train_rate_hz_min": min(r["rate_hz"] for r in trs) if trs else None,
                "train_rate_hz_median": rnd(np.median([r["rate_hz"] for r in trs]), 3) if trs else None,
                "train_rate_hz_max": max(r["rate_hz"] for r in trs) if trs else None,
                "train_duration_s_total": rnd(sum(r["duration_s"] for r in trs), 3) if trs else 0.0,
                "n_series": n_series, "series_sizes": " ".join(str(r["n_pulses"]) for r in series),
                "n_isolated_in_series_with_blank_channel": sum(by_blank.values()),
                "isolated_by_blank_channel": " ".join(f"{k}:{v}" for k, v in sorted(by_blank.items())),
                "blank_channel_sequence": " ".join(blank_seq),
                "n_peak_channel_runs": len(pruns), "modal_peak_channel": lab_all, "modal_peak_share": share_all,
                "neck_emg": emg, "anchor_verdict": verdict,
                "flat_or_quiet_levels_recorded_hemisphere": " ".join(str(v) for v in fq),
                "top2_channels_recorded_hemisphere": " ".join(top2), "contact_reading": reading,
                "spearman_distance_vs_peak": rho, "reading_level": kk})
    print(f"  {base}: {npulse} pulses, {n_iso} isolated, {len(pairs)} pairs, {len(trs)} trains, "
          f"{time.time() - t0:.1f} s", flush=True)
    return row, trains, series + pruns, ch_rows, t


# ----------------------------------------------------------------------------- summary

def summary_md(readme, se_rows, q1, files, trains, series, conv, p5_seg, p5_art, rule_hash):
    L = ["# oxford_dystonia_lfp structure: side-effect steps, Single and Paired files, Q1 null counts "
         "(scripts/probe_oxford_structure.py)\n",
         f"Generated by the script under reports/oxford_nonramp/rule.txt (SHA-256 {rule_hash}), written and hashed "
         "before the first data read. Every number below comes from files.csv, trains.csv, series.csv and channels.csv "
         "beside this file, reports/oxford_lfp/side_effect_steps.csv and reports/admissible_nulls/oxford_q1.csv. "
         "Structure only: no accelerometer, EMG, LFP-power or body metric is computed, and no value here fills a "
         "value of the neural rule file (NR-0.5).\n"]
    # A
    L.append("## Side-effect steps (OPEN-36 evidence, part A)\n")
    L.append(f"- Source: README.txt, identical in the {len(readme['copies'])} zips (SHA-256 {readme['sha256'][:12]}), "
             f"line {readme['stop_line']}: \"{readme['fragment']}\" (steps of {readme['step_mA']:g} mA, cap "
             f"{readme['cap_mA']:g} mA). Line {readme['single_line']} gives {readme['pulses_per_level']} pulses per "
             "contact level for the Single files.")
    by = {}
    for r in se_rows:
        by.setdefault((r["status"], r["reason"]), []).append(r)
    L.append(f"- {len(se_rows)} ramp files: " + "; ".join(f"{st} ({rs}) {len(v)}" for (st, rs), v in sorted(by.items())) + ".")
    inf = [r for r in se_rows if r["status"] == "inferred"]
    hemis = {}
    for r in inf:
        hemis.setdefault(f"{r['patient']}{r['hemisphere']}", []).append(r)
    for h, v in sorted(hemis.items()):
        L.append(f"- {h}: inferred side-effect step per contact level: "
                 + ", ".join(f"L{r['contact_level']} {r['inferred_side_effect_step_mA']:g} mA" for r in
                             sorted(v, key=lambda r: int(r["contact_level"]))) + ".")
    multi = [h for h, v in hemis.items() if len({r["inferred_side_effect_step_mA"] for r in v}) > 1]
    L.append(f"- Hemispheres whose inferred steps differ across contact levels: {len(multi)} of {len(hemis)} "
             f"({', '.join(sorted(multi)) or 'none'})"
             + ("; the stopping step is a per-file (per contact level) quantity in these data, so a single threshold "
                "per hemisphere, as NR-6.1.2 words it, is not what the files show." if multi else "."))
    mixed = []
    for h in sorted({f"{r['patient']}{r['hemisphere']}" for r in se_rows}):
        st_h = {r["status"] for r in se_rows if f"{r['patient']}{r['hemisphere']}" == h}
        if len(st_h) > 1:
            mixed.append(h)
    L.append(f"- Hemispheres with both inferred and not reached or ambiguous files: {len(mixed)} "
             f"({', '.join(mixed) or 'none'}).")
    rep = [r for r in inf if r["n_repeat_amplitude"] or r["n_open06_blocks"]]
    if rep:
        L.append("- Inferred files with repeated amplitudes or OPEN-06 blocks (reported, not used by the rule): "
                 + "; ".join(f"{r['file']} repeats {r['n_repeat_amplitude']}, OPEN-06 blocks {r['n_open06_blocks']}" for r in rep) + ".")
    short = [r for r in inf if r["other_blocks_on_s_median"] and r["last_block_on_s"] < 0.5 * r["other_blocks_on_s_median"]]
    L.append(f"- Inferred files whose last ON block lasts less than half the median of their other ON blocks: {len(short)}"
             + (f" ({', '.join(r['file'] for r in short)})" if short else "") + ".")
    L.append("- Status: evidence for OPEN-36 only. NR-0.5 forbids filling a threshold from a stage N0 report. The README "
             "gives the stopping rule, not a threshold per hemisphere or per contact level; the steps above are inferred "
             "from where each ramp stopped.\n")
    # B
    L.append("## Single and Paired files (part B)\n")
    L.append(f"- Detection: unipolar LFP channels, |dx| above {THR_MAD:g} times the channel median on at least {N_COINC} "
             f"channels within {COINC_TOL} sample, samples within {MERGE} samples merged into one pulse; isolated pulse: "
             f"no other pulse within {G_ISO_S:g} s on either side; bursts split at gaps above {G_ISO_S:g} s.")
    for f in files:
        L.append(f"- {f['file']} ({f['kind']}, {f['patient']}{f['hemisphere']}, {f['duration_s']:.1f} s): command "
                 + (f"channels {f['command_labels']}, active {f['command_active'] or 'none'}"
                    + (f" (values {f['command_values_uA']} uA, {f['command_segments']} segments)" if f["command_active"] else "")
                    if f["command_present"] else "absent")
                 + f"; flat LFP {f['lfp_flat'] or 'none'}; {f['n_pulses']} artefact pulses ({f['n_pulses_strict']} at "
                 f"{THR_STRICT:g} times the median, {f['n_strict_starting_inside_a_pulse']} of them inside a pulse span, "
                 f"{f['n_pulses_holding_2plus_strict']} pulses split in two or more there, "
                 f"{f['n_isolated_split_at_strict']} of them isolated)"
                 + (f", IPI median {fmt(f['ipi_median_ms'], 2)} ms (5th to 95th percentile {fmt(f['ipi_p5_ms'], 2)} to "
                    f"{fmt(f['ipi_p95_ms'], 2)} ms, range {fmt(f['ipi_min_ms'], 2)} ms to {fmt(f['ipi_max_ms'], 1)} ms)"
                    if f["ipi_median_ms"] is not None else "")
                 + f"; isolated {f['n_isolated']} (gap 0.1 s: {f['n_isolated_gap_0p1s']}, 1.0 s: {f['n_isolated_gap_1p0s']}, "
                 f"strict threshold: {f['n_isolated_strict']}); pairs {f['n_pairs']}"
                 + (f" (interval {fmt(f['pair_interval_ms_min'], 2)} to {fmt(f['pair_interval_ms_max'], 2)} ms, median "
                    f"{fmt(f['pair_interval_ms_median'], 2)})" if f["n_pairs"] else "")
                 + (f", {f['n_short_pairs']} of them at most {1000 * SHORT_PAIR_S:g} ms apart from "
                    f"{f['short_pairs_first_s']:.1f} to {f['short_pairs_last_s']:.1f} s (interval in samples:count "
                    f"{f['short_pair_interval_samples']}; runs at 1 ms resolution {f['short_pair_runs_1ms']}; width of their "
                    f"pulses median {fmt(f['short_pair_pulse_width_samples_median'], 1)} samples, max "
                    f"{f['short_pair_pulse_width_samples_max']}; blank channels "
                    f"of the isolated-pulse series in that span: {f['short_pairs_blank_channels'] or 'none'})"
                    if f["n_short_pairs"] else "")
                 + f"; trains {f['n_trains']} holding {f['n_pulses_in_trains']} pulses"
                 + (f" (rate {fmt(f['train_rate_hz_min'], 2)} to {fmt(f['train_rate_hz_max'], 2)} Hz, median "
                    f"{fmt(f['train_rate_hz_median'], 2)}; {f['train_duration_s_total']:.1f} s in total)" if f["n_trains"] else "")
                 + (f"; pulses at nonzero command {f['pulses_at_command_nonzero']}, at zero command {f['pulses_at_command_zero']}"
                    if f["command_active"] else "")
                 + f"; pulse width median {fmt(f['width_samples_median'], 1)} samples, max {f['width_samples_max']}; "
                 f"peak over channel median, 1st percentile {fmt(f['peak_ratio_p1'], 1)}, median {fmt(f['peak_ratio_median'], 1)}.")
        sers = [s for s in series if s["file"] == f["file"] and s["row_kind"] == "series"]
        if sers:
            big = [s for s in sers if s["n_pulses"] >= 2]
            L.append(f"  - {len(sers)} series of isolated pulses (sizes {f['series_sizes']}); "
                     + "; ".join(f"{s['start_s']:.1f} to {s['end_s']:.1f} s, {s['n_pulses']} pulses at "
                                 f"{fmt(s['rate_hz'], 3)} Hz, peak {s['modal_peak_channel']} ({s['modal_share']:.2f}), blank "
                                 f"{s['blank_channels'] or 'none'}, width median {fmt(s['width_samples_median'], 1)} samples"
                                 + (f", {s['n_split_at_strict']} split at the strict threshold" if s["n_split_at_strict"] else "")
                                 for s in big[:12])
                     + (f"; {len(big) - 12} more in series.csv" if len(big) > 12 else "") + ".")
        if f["n_isolated"]:
            L.append(f"  - Isolated pulses inside a series with a blank channel: {f['n_isolated_in_series_with_blank_channel']} "
                     f"of {f['n_isolated']}; by blank channel: {f['isolated_by_blank_channel'] or 'none'}; blank channels in "
                     f"time order: {f['blank_channel_sequence'] or 'none'}.")
    for f in files:
        if f["kind"] != "paired-pulse" or not f["n_short_pairs"]:
            continue
        ws = [s for s in series if s["file"] == f["file"] and s["row_kind"] == "series" and s["n_pulses"] >= 2]
        L.append(f"- {f['file']}: the pulses of its {f['n_short_pairs']} short pairs have width median "
                 f"{fmt(f['short_pair_pulse_width_samples_median'], 1)} samples (max {f['short_pair_pulse_width_samples_max']}); "
                 f"its {len(ws)} isolated-pulse series have width medians "
                 f"{', '.join(fmt(s['width_samples_median'], 1) for s in ws)} samples in time order, and "
                 f"{sum(s['n_split_at_strict'] for s in ws)} of their pulses split at the strict threshold. Two stimuli "
                 f"closer than the merge window (B4, {MERGE} samples between pulse samples) count as one pulse, so an "
                 "isolated pulse wider than a single artefact may be a pair at a shorter interval; structure alone does "
                 "not settle it.")
    if p5_seg:
        prev_tot = sum(s["prev"] for s in p5_seg)
        mine_tot = sum(s["mine"] for s in p5_seg)
        diff = [s for s in p5_seg if s["mine"] != s["prev"]]
        worst = max(p5_seg, key=lambda s: (s["mine"] - s["prev"]) / max(s["s"], 1e-9))
        p5 = next(f for f in files if f["file"] == "P5_Left_Single.mat")
        L.append(f"- Cross-check on P5_Left_Single.mat against reports/oxford_lfp/nonramp_stim_blocks.csv "
                 f"(probe_oxford_lfp.py, threshold per command ON block): over its {len(p5_seg)} command ON segments that "
                 f"table counts {prev_tot} pulses and this probe {mine_tot}; {len(diff)} segments differ; the largest "
                 f"shortfall per second is in segment {worst['segment']} ({worst['uA']:g} uA, {worst['s']:.1f} s): "
                 f"{worst['prev']} against {worst['mine']} here, with a train rate of {worst['prev_hz']} Hz there and "
                 f"{fmt(p5['train_rate_hz_min'], 2)} to {fmt(p5['train_rate_hz_max'], 2)} Hz in every train here.")
    L.append("\n## NR-6.2.1 anchor verdict\n")
    for f in files:
        L.append(f"- {f['file']}: {f['anchor_verdict']} ({f['n_isolated']} isolated pulses, "
                 f"{f['n_isolated_in_series_with_blank_channel']} of them in series with a blank channel; neck EMG "
                 f"{f['neck_emg'] or 'none'}).")
    anyok = [f["file"] for f in files if f["anchor_verdict"] == "anchor possible"]
    L.append(f"- Files that can anchor NR-6.2.1 on structure: {', '.join(anyok) if anyok else 'none'}. The verdict reads "
             "structure only (isolated pulses present, EMG channel present); no evoked response was computed, and the "
             "response windows and the latency definition stay OPEN-22.")
    for f in files:
        if f["anchor_verdict"] != "anchor possible":
            continue
        lv = [kv.split(":") for kv in f["isolated_by_blank_channel"].split()]
        low = [f"{k} {v}" for k, v in lv if int(v) < N_ANCHOR]
        L.append(f"- {f['file']}: blank channels with fewer than {N_ANCHOR} isolated pulses in their series: "
                 f"{', '.join(low) or 'none'}"
                 + (f"; the {f['n_short_pairs']} pairs at most {1000 * SHORT_PAIR_S:g} ms apart lie from "
                    f"{f['short_pairs_first_s']:.1f} to {f['short_pairs_last_s']:.1f} s, where the isolated-pulse series "
                    f"have blank channel {f['short_pairs_blank_channels'] or 'none'}" if f["n_short_pairs"] else "") + ".")
    p5 = next((f for f in files if f["file"] == "P5_Left_Single.mat"), None)
    if p5 is not None and p5["n_isolated"] == 0:
        L.append(f"- NR-6.2.1 names the single-pulse files of P5 and P6 as the causal anchor. P5_Left_Single.mat holds no "
                 f"isolated pulse: its {p5['n_pulses']} pulses lie in {p5['n_trains']} trains at "
                 f"{fmt(p5['train_rate_hz_median'], 2)} Hz (median rate) under the active command "
                 f"{p5['command_active'] or 'none'} ({p5['command_values_uA'] or 'no'} uA), so on structure it is a "
                 "high-frequency stimulation file, not a single-pulse file.")
    L.append("\n## Contact level of P5 (artefact geometry)\n")
    for f in files:
        if f["file"] in ("P5_Left_Single.mat", "L7_L2_Paired.mat"):
            L.append(f"- {f['file']}: flat or quiet LFP levels of the recorded hemisphere: "
                     f"{f['flat_or_quiet_levels_recorded_hemisphere'] or 'none'}; two largest median artefact peaks: "
                     f"{f['top2_channels_recorded_hemisphere'] or 'none'}; reading: {f['contact_reading']}; Spearman of "
                     f"distance from level {f['reading_level']} against median peak: {f['spearman_distance_vs_peak']}"
                     + (f"; the file name gives level {f['contact_level_from_name']}" if f["contact_level_from_name"] else "") + ".")
    L.append(f"- Convention check on the ramp files (inventory.csv): the unipolar channel at the named contact level is flat "
             f"or absent in {conv[0]} of {conv[1]} files.")
    p5 = next((f for f in files if f["file"] == "P5_Left_Single.mat"), None)
    if p5 is not None and p5_art:
        lv = p5["reading_level"]
        near = (sorted((x for x in range(1, 9) if x != lv), key=lambda x: (abs(x - lv), x))[:2] if lv else [])
        art_lv = sorted(int(c[1]) for c in p5_art)
        L.append(f"- probe_oxford_lfp.py named {' and '.join(p5_art)} as the P5 artefact channels (nonramp_stim_summary.csv). "
                 + (f"They are the two contacts nearest to level {lv}, the only flat LFP level of the left lead"
                    if near and art_lv == sorted(near) else "They are not the two contacts nearest to the flat level")
                 + f"; this probe ranks {p5['top2_channels_recorded_hemisphere'] or 'none'} first.")
    L.append("- The reading describes where the artefact is largest and which contact is not recorded; it is not a "
             "documented contact level.")
    # C
    L.append("\n## Q1 admissible-null k (NR-6.1.7, part C)\n")
    for r in q1:
        L.append(f"- {r['scope']}, {r['files']}, {r['open06']}: {r['n_patients']} patients, {r['n_hemispheres']} hemispheres, "
                 f"{r['n_files']} files, k = {r['k']} (log10 {fmt(r['log10_k'], 3)}), smallest p {r['smallest_p']}; "
                 f"ON blocks left out {r['n_on_blocks_left_out']}; files left out: {r['files_left_out'] or 'none'}.")
    qq = [r for r in q1 if r["scope"] == "q1"]
    best = min(qq, key=lambda r: float(r["smallest_p"]))
    worst = max(qq, key=lambda r: float(r["smallest_p"]))
    L.append(f"- Smallest attainable p in the Q1 scope: {best['smallest_p']} ({best['files']}, {best['open06']}); the largest "
             f"of these minima is {worst['smallest_p']} ({worst['files']}, {worst['open06']}).")
    n20 = sum(1 for r in q1 if int(r["k"]) >= 20)
    ndraw = sum(1 for r in q1 if r["draws_needed_open33"])
    L.append(f"- Variants with k >= 20 (NR-1.8): {n20} of {len(q1)}. Variants with k above {ENUM_MAX:.0e} (enumeration "
             f"counted infeasible, nulls drawn without replacement per NR-1.7, draw count OPEN-33): {ndraw} of {len(q1)}.")
    same = all(r["k"] == next(x["k"] for x in q1 if x["scope"] == r["scope"] and x["files"] == r["files"]
                              and x["open06"] == "with_open06") for r in q1)
    o6_files = sorted({r["file"] for r in se_rows if r["n_open06_blocks"]})
    o6_pat = sorted({r["patient"] for r in se_rows if r["n_open06_blocks"]})
    q1_pat = sorted({h.split(":")[0][:-1] for r in qq for h in r["hemispheres"].split("; ") if h})
    L.append(f"- Leaving the OPEN-06 blocks out {'changes no k' if same else 'changes k in some variants'}: they lie in "
             f"{', '.join(o6_files) or 'no file'} (patients {', '.join(o6_pat) or 'none'}; Q1 scope patients "
             f"{', '.join(q1_pat)}), and a file leaves the count only when it loses every ON block.")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main

def write_prov(path, rule_hash, st, extra, outputs):
    prov = {"script": "scripts/probe_oxford_structure.py",
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "rule": "reports/oxford_nonramp/rule.txt", "rule_sha256": rule_hash,
            "moveatlas_stamp": st, "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "outputs": {rel(p): sha256_file(p) for p in outputs}}
    prov.update(extra)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    print("wrote", rel(path), flush=True)


def main():
    rule_hash = assert_rule_hash()          # before any input is read
    below_normal()
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv += ["--out", OUT]
    sys.argv = [sys.argv[0]] + argv
    a = parse_args(__doc__.splitlines()[0])
    os.makedirs(NULLS, exist_ok=True)
    st = stamp(a.moveatlas)
    raw_dir = os.path.join(a.moveatlas, RAW_REL)
    zips = sorted(os.path.join(raw_dir, z) for z in os.listdir(raw_dir) if z.endswith(".zip") and not z.startswith("._"))
    readme = read_readmes(zips)
    summ = read_rows(os.path.join(OXF, "ramp_file_summary.csv"))
    g15 = read_rows(os.path.join(OXF, "g15_transitions.csv"))
    inv = read_rows(os.path.join(OXF, "inventory.csv"))
    mh = read_rows(os.path.join(OXF, "movement_hemispheres.csv"))
    nonramp_blocks = read_rows(os.path.join(OXF, "nonramp_stim_blocks.csv"))
    nonramp_summ = read_rows(os.path.join(OXF, "nonramp_stim_summary.csv"))
    p5_art = [c for s in nonramp_summ if s["file"] == "P5_Left_Single.mat"
              for c in (s["artefact_channel"], s["artefact_channel2"]) if c]
    inputs = {rel(os.path.join(OXF, n)): sha256_file(os.path.join(OXF, n)) for n in
              ("ramp_file_summary.csv", "g15_transitions.csv", "inventory.csv", "movement_hemispheres.csv",
               "nonramp_stim_blocks.csv", "nonramp_stim_summary.csv")}
    if len(summ) != 62:
        raise SystemExit(f"ramp_file_summary.csv holds {len(summ)} files, expected 62: stop")

    # part A
    se_rows = side_effect_rows(summ, g15, readme)
    se_path = os.path.join(OXF, "side_effect_steps.csv")
    write_csv(se_path, se_rows)
    write_prov(os.path.join(OXF, "_provenance_side_effect_steps.json"), rule_hash, st,
               {"part": "A (rule.txt section A)", "readme": {k: readme[k] for k in ("copies", "stop_line", "single_line",
                                                                                  "step_mA", "cap_mA")},
                "inputs": inputs, "note": EVIDENCE}, [se_path])

    # part C
    q1 = q1_rows(summ, g15, mh)
    q1_path = os.path.join(NULLS, "oxford_q1.csv")
    write_csv(q1_path, q1)
    write_prov(os.path.join(NULLS, "_provenance_oxford_q1.json"), rule_hash, st,
               {"part": "C (rule.txt section C)", "inputs": inputs, "parameters": {"ENUM_MAX": ENUM_MAX},
                "note": "counts from file structure only (NR-0.5); k per NR-6.1.7, p per NR-1.7, draw count OPEN-33"},
               [q1_path])

    # part B
    inv_by = {r["file"]: r for r in inv}
    members = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for zi in zf.infolist():
                base = zi.filename.split("/")[-1]
                if zi.filename.lower().endswith(".mat") and not base.startswith("._") and "__MACOSX" not in zi.filename \
                        and inv_by.get(base, {}).get("kind") in ("single-pulse", "paired-pulse"):
                    members.append((z, zi))
    if len(members) != 6:
        raise SystemExit(f"{len(members)} non-ramp members found, expected 6: stop")
    files, trains, series, chans, p5_seg = [], [], [], [], []
    for z, zi in members:
        wait_if_slow()
        base = zi.filename.split("/")[-1]
        if inv_by[base]["crc32"] != f"{zi.CRC:08x}":
            raise SystemExit(f"{base}: CRC differs from inventory.csv: stop")
        r, tr, se, ch, t_p = analyze_file(z, zi, inv_by[base])
        if base == "P5_Left_Single.mat":
            p5_seg = []
            for b in nonramp_blocks:
                if b["file"] == base and b["state"] == "ON":
                    a_, b_ = int(b["start_sample"]), int(b["end_sample"])
                    mine = int(np.count_nonzero((t_p >= a_) & (t_p < b_)))
                    p5_seg.append({"segment": int(b["segment"]), "uA": float(b["amplitude_uA"]),
                                   "s": float(b["duration_s"]), "prev": int(float(b["n_pulses"] or 0)),
                                   "prev_hz": b["stim_freq_hz"], "mine": mine})
        files.append(r)
        trains += tr
        series += se
        chans += ch
    ramp_inv = [r for r in inv if r["kind"] == "ramp"]
    conv_n = 0
    for r in ramp_inv:
        lab = f"{r['hemisphere']}{r['contact_level']}"
        if lab not in r["lfp_unipolar"].split() or lab in r["flat_channels"].split():
            conv_n += 1
    paths ={n: os.path.join(OUT, n) for n in ("files.csv", "trains.csv", "series.csv", "channels.csv", "summary.md")}
    write_csv(paths["files.csv"], files)
    write_csv(paths["trains.csv"], trains)
    write_csv(paths["series.csv"], series, first=["file", "row_kind", "index", "n_pulses", "start_s", "end_s"])
    write_csv(paths["channels.csv"], chans)
    with open(paths["summary.md"], "w", encoding="utf-8", newline="\n") as f:
        f.write(summary_md(readme, se_rows, q1, files, trains, series, (conv_n, len(ramp_inv)), p5_seg, p5_art,
                           rule_hash))
    print("wrote", rel(paths["summary.md"]), flush=True)
    write_prov(os.path.join(OUT, "_provenance.json"), rule_hash, st,
               {"part": "B (rule.txt section B) and the summary of A, B, C", "parameters": PARAMS, "inputs": inputs,
                "readme_sha256": readme["sha256"], "source": RAW_REL,
                "members": [{"zip": os.path.basename(z), "member": zi.filename, "crc32": f"{zi.CRC:08x}",
                             "bytes": zi.file_size} for z, zi in members],
                "read_channels": "unipolar LFP L1-L8 R1-R8 and command Am1 Am2 Cur1 Cur2 only",
                "related_outputs": [rel(se_path), rel(q1_path)]},
               list(paths.values()))


if __name__ == "__main__":
    main()
