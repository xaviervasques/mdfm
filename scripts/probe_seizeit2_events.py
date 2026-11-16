"""seizeit2 seizure event census and Q8 feasibility on the non-lockbox persons only (task T28), read-only.

SeizeIT2 (datasets/eeg/seizeit2/raw, BIDS, OpenNeuro ds005873) annotates each run in one
events.tsv beside its EEG EDF. Q8 (addendum 1, section 3; docs/rules/NEURAL_RULES_DRAFT.md
NR-6.8) contrasts hyperkinetic against explicitly non-motor seizures from fixed body descriptors
read on the runs that G15 admits (NR-2.6, NR-2.7), less the contained runs (NR-2.15).

Lockbox. The 25 seizeit2 persons of the encoder lockbox (reports/lockbox/lockbox_persons.csv,
whose SHA-256 is checked against reports/lockbox/lockbox.sha256) are dropped from the subject
list before any path inside their folders is built (NR-12.2). Every file of the raw tree is opened
through one guard that refuses a path naming a lockbox subject; their rows of
reports/sync_seizeit2/runs.csv are discarded as the file is parsed. The script asserts at the end
that no lockbox subject was opened and that all 25 were listed and dropped (lockbox_check.json).
Full-cohort figures (the addendum's 883 seizures over 125 persons) need the owner's permission to
read the lockbox persons' labels; nothing here subtracts the non-lockbox counts from them.

What the script does, in order.
1. Census: every events.tsv of the non-lockbox persons; each raw eventType label is mapped to a
   draft class (label_class_map_draft.csv, a draft for OPEN-48, every level of events.json plus
   any label met on disk): hyperkinetic, explicitly_non_motor, tonic, automatisms,
   focal_to_bilateral, other_motor, motor_unspecified, unknown_or_unspecified, not_seizure.
2. Class counts (seizures, patients, runs) in five scopes: all runs; all runs minus the contained
   runs of reports/sync_seizeit2/contained_runs.csv; verdict = pass runs of
   reports/sync_seizeit2/runs.csv minus the contained runs (the admitted set of NR-2.7 and
   NR-2.15); the same with a deltoid EMG EDF; and, for OPEN-08, runs of any verdict with the
   ECG-EMG-module accelerometer and a deltoid EMG EDF (both on the chest module), minus the
   contained runs. A patient is a seizeit2 subject (identity map, reports/person_ids/person_map.csv).
3. Feasibility: whether each primary class (hyperkinetic, explicitly_non_motor) and the tonic class
   keep at least 10 patients (NR-6.8.2, D16.5) in each scope. Printed and written to
   feasibility.csv.
4. Accelerometer scale (input to OPEN-09), label-free: every mov EDF of the non-lockbox persons is
   read whole, in 10 s blocks per module (EEG module, ECG-EMG module). Blocks with a zero-filled
   sample (three axes at physical 0), a digital extreme or a sample beyond +-4 header units (the
   +-32.77 sentinel and the integer-valued axes of some files) are left out. A block is at rest when every axis SD is below 0.013 times the
   file's median block-mean norm (the 13 mg rest criterion of the GGIR autocalibration, read
   relative to the file's own scale). Per file and module: the median rest norm (header units,
   nominally g), its inverse as the scale to g, its spread, its first-half and second-half values,
   the orientation coverage of the rest blocks, and, when every axis is covered beyond +-0.3 on
   both sides, a per-axis gain and offset fit to the unit sphere (GGIR iterative closest point)
   with the rest-norm error before and after. No seizure annotation enters this pass, and no body
   descriptor is computed per seizure class. Cross-check: the file's median rest norm against the
   median raw norm of the sync probe's anchor windows of the same file
   (reports/sync_seizeit2/windows.csv, non-lockbox rows only).

Outputs in reports/seizeit2_events/: label_class_map_draft.csv, events_census.csv, seizures.csv,
class_counts.csv, class_vigilance.csv, feasibility.csv, per_subject.csv, contained_events.csv,
acc_scale_per_file.csv, acc_scale_per_subject.csv, lockbox_check.json, summary.json, summary.md,
_provenance.json (scripts/moveatlas_stamp.py).
Resumable: per-file scale results are cached in _cache/ and the cache is removed at the end of a
complete run (--keep-cache keeps it). The scale pass waits while cache/runs/queue/SLOW exists.
Run: python scripts/probe_seizeit2_events.py --moveatlas <moveatlas>
Options (parsed before the common ones): --limit N (first N mov files of the scale pass, for a
pilot), --no-scale, --fresh, --keep-cache.
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _common import parse_args  # noqa: E402

# ---------------------------------------------------------------- parameters (fixed)
MIN_PATIENTS = 10          # NR-6.8.2 and D16.5: no Q8 number below 10 patients per class
PRIMARY = ("hyperkinetic", "explicitly_non_motor")
SECONDARY = ("tonic",)     # the tonic against hyperkinetic contrast, NR-6.8.2
BLOCK_S = 10               # block length of the scale pass (s)
REST_SD_REL = 0.013        # rest: every axis SD below this fraction of the file's median block norm
IMPLAUSIBLE_ABS = 4.0      # a sample beyond this (header units) is no acceleration: the +-32.77 sentinel and the
                           # integer-valued axes (5 to 29) of some files; most header ranges stop at 1 or 2
COVER = 0.3                # orientation coverage needed on both sides of every axis for the fit
FIT_MIN_BLOCKS = 50        # rest blocks needed for the per-axis fit
FIT_MAX_BLOCKS = 3000      # rest blocks used by the fit (evenly spaced subsample)
FIT_MAX_ITER = 1000
FIT_TOL = 1e-10
CHUNK_REC = 3600           # EDF data records read at a time (one record is 1 s here)

EEG_ACC = ["EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z"]
ECG_ACC = ["ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z"]
MODULES = (("eeg_module", EEG_ACC), ("ecgemg_module", ECG_ACC))
EV_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_task-([^_]+)_run-([^_]+)_events\.tsv$")
MOV_RE = re.compile(r"^(sub-[^_]+)_(ses-[^_]+)_task-([^_]+)_run-([^_]+)_mov\.edf$")
SUB_RE = re.compile(r"sub-[A-Za-z0-9]+")

CLASSES = ("hyperkinetic", "explicitly_non_motor", "tonic", "automatisms", "focal_to_bilateral",
           "other_motor", "motor_unspecified", "unknown_or_unspecified", "not_seizure", "unlisted")
SEIZURE_CLASSES = tuple(c for c in CLASSES if c not in ("not_seizure",))
SCOPES = (("all_runs", "every events file of the non-lockbox persons"),
          ("all_minus_contained", "all runs minus the contained runs (NR-2.15)"),
          ("pass_minus_contained", "verdict = pass runs minus the contained runs (NR-2.7, NR-2.15)"),
          ("pass_minus_contained_emg", "the same, runs with a deltoid EMG EDF"),
          ("chest_module_emg_minus_contained", "any verdict, runs with the ECG-EMG-module accelerometer and a deltoid "
                                               "EMG EDF, minus the contained runs (a single-module reading, OPEN-08)"))
NORM_BINS = (0.0, 0.15, 0.20, 0.30, 0.45, 0.55, 0.70, 1.0, float("inf"))  # rest-norm histogram edges (header units)

# ---------------------------------------------------------------- arguments
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--limit", type=int, default=0)
pre.add_argument("--no-scale", action="store_true")
pre.add_argument("--fresh", action="store_true")
pre.add_argument("--keep-cache", action="store_true")
OPT, rest = pre.parse_known_args()
if "--out" not in rest:
    rest += ["--out", os.path.join(TREE, "reports", "seizeit2_events")]
sys.argv = [sys.argv[0]] + rest
ARGS = parse_args(__doc__.splitlines()[0])
RAW = f"{ARGS.moveatlas}/datasets/eeg/seizeit2/raw"
OUT = ARGS.out
CACHE = os.path.join(OUT, "_cache")
LOCKBOX_CSV = os.path.join(TREE, "reports", "lockbox", "lockbox_persons.csv")
LOCKBOX_SHA = os.path.join(TREE, "reports", "lockbox", "lockbox.sha256")
SYNC = os.path.join(TREE, "reports", "sync_seizeit2")
PERSON_MAP = os.path.join(TREE, "reports", "person_ids", "person_map.csv")
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")


# ---------------------------------------------------------------- process discipline
def below_normal():
    try:
        import psutil
        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            psutil.Process().nice(10)
        return True
    except Exception:  # noqa: BLE001
        return False


def wait_slow():
    told = False
    while os.path.exists(SLOW):
        if not told:
            print(f"paused: {SLOW} exists", flush=True)
            told = True
        time.sleep(30)
    if told:
        print("resumed", flush=True)


# ---------------------------------------------------------------- lockbox guard
def require(cond, msg):
    """A check that python -O does not strip: with an assert statement the guard would vanish under -O."""
    if not cond:
        raise AssertionError(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lockbox():
    """seizeit2 lockbox person ids as BIDS subject folder names; ids only."""
    sha = sha256_file(LOCKBOX_CSV)
    with open(LOCKBOX_SHA, encoding="utf-8") as f:
        expected = f.read().split()[0].strip().lower()
    require(sha == expected, f"lockbox_persons.csv SHA-256 {sha} differs from lockbox.sha256 {expected}")
    subs = set()
    with open(LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == "seizeit2":
                subs.add("sub-" + r["person_id"].split(":", 1)[1])
    require(len(subs) == 25, f"expected 25 seizeit2 lockbox persons, found {len(subs)}")
    return frozenset(subs), sha


class Guard:
    """Every raw-tree file is opened here; a path naming a lockbox subject is refused."""

    def __init__(self, lockbox):
        self.lockbox = lockbox
        self.opened = Counter()          # subject -> files opened
        self.by_kind = Counter()         # file kind -> files opened
        self.meta = []                   # dataset-level files opened

    def check(self, path):
        subs = set(SUB_RE.findall(path.replace("\\", "/")))
        require(not (subs & self.lockbox), f"refused: lockbox subject in {path}")
        return subs

    def open(self, path, mode="r", kind="other", **kw):
        subs = self.check(path)
        require(len(subs) == 1, f"a subject file must name one subject: {path}")
        self.opened[next(iter(subs))] += 1
        self.by_kind[kind] += 1
        return open(path, mode, **kw)

    def open_meta(self, path, **kw):
        subs = self.check(path)
        require(not subs, f"not a dataset-level file: {path}")
        self.meta.append(os.path.basename(path))
        return open(path, **kw)


# ---------------------------------------------------------------- label mapping (draft for OPEN-48)
UNSPECIFIED = {"sz", "sz_foc", "sz_foc_a", "sz_foc_ia", "sz_foc_ua", "sz_uo", "sz_gen"}
NM_MOVEMENT_NOTE = {
    "sz_gen_nm_atypical": "non-motor by ILAE, but its description names changes in tone",
    "sz_gen_nm_myoclonic": "non-motor by ILAE, but its description names rhythmic myoclonic movements",
    "sz_gen_nm_eyelidMyio": "non-motor by ILAE, but its description names eyelid myoclonia",
    "sz_uo_nm_behavior": "its description says behavioural activity, against the behaviour arrest of the focal labels",
}


def class_of(label):
    """Draft mapping of a raw eventType label; returns (class, rule, note)."""
    if label in ("bckg", "impd"):
        return "not_seizure", "bckg or impd", ""
    if not label.startswith("sz"):
        return "unlisted", "not a seizure prefix", "label outside the events.json pattern"
    parts = label.split("_")
    if parts[-1] == "hyperkinetic" and "m" in parts:
        return "hyperkinetic", "motor label ending in hyperkinetic", ""
    if "nm" in parts:
        return "explicitly_non_motor", "label carrying nm (non-motor)", NM_MOVEMENT_NOTE.get(label, "")
    if label == "sz_foc_f2b":
        return "focal_to_bilateral", "sz_foc_f2b", "bilateral tonic-clonic after a focal onset; not a primary class"
    if parts[-1] == "automatisms":
        return "automatisms", "motor label ending in automatisms", ""
    if parts[-1] == "tonic" and "m" in parts:
        note = "generalized tonic, grouped with focal tonic in this draft" if label.startswith("sz_gen") else ""
        return "tonic", "motor label ending in tonic", note
    if "um" in parts:
        return "unknown_or_unspecified", "label carrying um (motor features unknown or unreported)", ""
    if label in UNSPECIFIED:
        return "unknown_or_unspecified", "onset or awareness only, no motor field", ""
    if parts[-1] == "m":
        return "motor_unspecified", "motor label without a subtype", "motor, subtype not given; never a negative"
    if "m" in parts:
        return "other_motor", "motor label with another subtype (clonic, myoclonic, spasms, atonic, tonic-clonic)", ""
    return "unlisted", "no rule", "check by hand"


# ---------------------------------------------------------------- small helpers
def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def q(vals, p):
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(np.percentile(v, p)) if v else None


def r(x, nd=4):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return round(float(x), nd)


BIN_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in zip(NORM_BINS[:-1], NORM_BINS[1:])]


def bin_of(v):
    for lab, lo, hi in zip(BIN_LABELS, NORM_BINS[:-1], NORM_BINS[1:]):
        if lo <= v < hi:
            return lab
    return "n/a"


def subject_runs(pairs):
    """Consecutive subjects (in id order, lockbox ids absent) sharing a bin: 'sub-a..sub-b (n): bin'."""
    out, cur = [], None
    for s, b in pairs:
        if cur and cur[2] == b:
            cur[1], cur[3] = s, cur[3] + 1
        else:
            if cur:
                out.append(cur)
            cur = [s, s, b, 1]
    if cur:
        out.append(cur)
    return [f"{a}..{z} ({n}): {b}" if n > 1 else f"{a} (1): {b}" for a, z, b, n in out]


def write_csv(name, rows, fields):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ---------------------------------------------------------------- EDF header and blocks
def edf_header(path, guard):
    with guard.open(path, "rb", kind="mov_edf") as f:
        h = f.read(256)
        ns = int(h[252:256].decode("ascii").strip())
        sh = f.read(256 * ns)
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names = ["label", "transducer", "physdim", "physmin", "physmax", "digmin", "digmax", "prefilt", "nsamp", "res"]
    sig, pos = {}, 0
    for n, w in zip(names, widths):
        sig[n] = [sh[pos + i * w: pos + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]
        pos += w * ns
    nsamp = [int(x) for x in sig["nsamp"]]
    hd = {"nbytes": int(h[184:192].decode("ascii").strip()), "nrec": int(h[236:244].decode("ascii").strip()),
          "recdur": float(h[244:252].decode("ascii").strip()), "ns": ns, "sig": sig, "nsamp": nsamp,
          "rec_samples": sum(nsamp), "offsets": [sum(nsamp[:i]) for i in range(ns)],
          "size": os.path.getsize(path)}
    hd["size_ok"] = hd["nbytes"] + hd["nrec"] * hd["rec_samples"] * 2 == hd["size"]
    return hd


def chan_info(hd, label):
    i = hd["sig"]["label"].index(label)
    pmin, pmax = float(hd["sig"]["physmin"][i]), float(hd["sig"]["physmax"][i])
    dmin, dmax = float(hd["sig"]["digmin"][i]), float(hd["sig"]["digmax"][i])
    g = (pmax - pmin) / (dmax - dmin) if dmax != dmin else float("nan")
    d0 = dmin + (0.0 - pmin) / g if g and math.isfinite(g) and g != 0 else float("nan")
    return {"label": label, "off": hd["offsets"][i], "n": hd["nsamp"][i], "pmin": pmin, "pmax": pmax,
            "dmin": dmin, "dmax": dmax, "g": g, "d0": d0, "physdim": hd["sig"]["physdim"][i]}


def ggir_fit(u):
    """Per-axis offset and gain that bring rest-block mean vectors u (n x 3) onto the unit sphere
    (iterative closest point with weighted least squares per axis, as in GGIR g.calibrate)."""
    offset = np.zeros(3)
    scale = np.ones(3)
    w = np.ones(len(u))
    res_prev = None
    it = 0
    converged = False
    for it in range(1, FIT_MAX_ITER + 1):
        curr = (u + offset) * scale
        nrm = np.linalg.norm(curr, axis=1)
        nrm[nrm == 0] = 1.0
        closest = curr / nrm[:, None]
        och, sch = np.zeros(3), np.ones(3)
        fitted = np.empty_like(curr)
        sw = w.sum()
        for k in range(3):
            x, y = curr[:, k], closest[:, k]
            xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
            sxx = (w * (x - xm) ** 2).sum()
            b = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 0 else 1.0
            a = ym - b * xm
            och[k], sch[k] = a, b
            fitted[:, k] = a + b * x
        offset = offset + och / (scale * sch)
        scale = scale * sch
        d2 = ((fitted - closest) ** 2).sum(axis=1)
        res = 3 * float((w * d2).mean() / w.sum())
        w = np.minimum(1 / np.sqrt(np.maximum(d2, 1e-30)), 1 / 0.01)
        if res_prev is not None and abs(res - res_prev) < FIT_TOL:
            converged = True
            break
        res_prev = res
    return offset, scale, it, converged


def scale_file(path, hd, mods, guard):
    """Rest-norm statistics per module of one mov EDF (label-free)."""
    out = {}
    rec_bytes = hd["rec_samples"] * 2
    acc_n = {ci["n"] for infos in mods.values() for ci in infos}
    if len(acc_n) != 1 or hd["recdur"] <= 0:
        return {m: {"status": "irregular_layout"} for m in mods}
    n_per_rec = acc_n.pop()
    fs = n_per_rec / hd["recdur"]
    rpb = int(round(BLOCK_S / hd["recdur"]))
    blk = rpb * n_per_rec
    acc = {m: {"mean": [], "sd": [], "idx": [], "n_blocks": 0, "zero": 0, "sat": 0, "impl": 0,
               "impl_axis": np.zeros(3), "samples": 0} for m in mods}
    nrec = hd["nrec"]
    nb_total = nrec // rpb
    chunk = max(rpb, (CHUNK_REC // rpb) * rpb)
    bytes_read = 0
    with guard.open(path, "rb", kind="mov_edf") as f:
        b0 = 0
        for r0 in range(0, nb_total * rpb, chunk):
            r1 = min(nb_total * rpb, r0 + chunk)
            nb = (r1 - r0) // rpb
            f.seek(hd["nbytes"] + r0 * rec_bytes)
            buf = f.read((r1 - r0) * rec_bytes)
            bytes_read += len(buf)
            if len(buf) < (r1 - r0) * rec_bytes:
                nb = len(buf) // (rec_bytes * rpb)
                buf = buf[:nb * rpb * rec_bytes]
                r1 = r0 + nb * rpb
            if nb == 0:
                break
            a = np.frombuffer(buf, dtype="<i2").reshape(r1 - r0, hd["rec_samples"])
            for m, infos in mods.items():
                D = np.stack([a[:, ci["off"]:ci["off"] + ci["n"]].reshape(-1) for ci in infos], axis=1).astype(np.float64)
                P = np.empty_like(D)
                zero = np.ones(len(D), dtype=bool)
                sat = np.zeros(len(D), dtype=bool)
                for k, ci in enumerate(infos):
                    P[:, k] = (D[:, k] - ci["dmin"]) * ci["g"] + ci["pmin"]
                    zero &= (np.abs(D[:, k] - ci["d0"]) <= 1.0) if math.isfinite(ci["d0"]) else False
                    sat |= (D[:, k] <= ci["dmin"]) | (D[:, k] >= ci["dmax"])
                implaus = np.abs(P) > IMPLAUSIBLE_ABS
                sent = implaus.any(axis=1)
                zb = zero.reshape(nb, blk).any(axis=1)
                sb = sat.reshape(nb, blk).any(axis=1)
                eb = sent.reshape(nb, blk).any(axis=1)
                bad = zb | sb | eb
                Pb = P.reshape(nb, blk, 3)
                good = ~bad
                st = acc[m]
                st["n_blocks"] += nb
                st["zero"] += int(zb.sum())
                st["sat"] += int((sb & ~zb).sum())
                st["impl"] += int((eb & ~zb & ~sb).sum())
                st["impl_axis"] += implaus.sum(axis=0)
                st["samples"] += len(P)
                if good.any():
                    st["mean"].append(Pb[good].mean(axis=1))
                    st["sd"].append(Pb[good].std(axis=1))
                    st["idx"].append(np.flatnonzero(good) + b0)
            b0 += nb
    for m, st in acc.items():
        row = {"status": "ok", "fs_hz": fs, "n_blocks": st["n_blocks"], "blocks_zero": st["zero"],
               "blocks_saturated": st["sat"], "blocks_implausible": st["impl"],
               "implausible_frac_xyz": ";".join(f"{v / max(1, st['samples']):.4f}" for v in st["impl_axis"]),
               "physdim": ";".join(sorted({ci["physdim"] for ci in mods[m]})),
               "header_ranges": ";".join(f"{ci['pmin']:g}..{ci['pmax']:g}" for ci in mods[m])}
        if not st["mean"]:
            row["status"] = "no_valid_block"
            out[m] = row
            continue
        M = np.concatenate(st["mean"])
        S = np.concatenate(st["sd"])
        I = np.concatenate(st["idx"])
        norms = np.linalg.norm(M, axis=1)
        ref = float(np.median(norms))
        rest = S.max(axis=1) < REST_SD_REL * ref
        row.update({"blocks_valid": int(len(M)), "median_block_norm": ref, "blocks_rest": int(rest.sum()),
                    "rest_hours": float(rest.sum()) * BLOCK_S / 3600.0})
        if rest.sum() == 0 or ref <= 0:
            row["status"] = "no_rest_block"
            out[m] = row
            continue
        rn = norms[rest]
        p50 = float(np.median(rn))
        half = I[rest] < (nb_total / 2.0)
        first = float(np.median(rn[half])) if half.any() else None
        second = float(np.median(rn[~half])) if (~half).any() else None
        u = M[rest] / p50
        umin, umax = u.min(axis=0), u.max(axis=0)
        covered = bool(np.all(umin < -COVER) and np.all(umax > COVER))
        row.update({"rest_norm_p5": float(np.percentile(rn, 5)), "rest_norm_p25": float(np.percentile(rn, 25)),
                    "rest_norm_p50": p50, "rest_norm_p75": float(np.percentile(rn, 75)),
                    "rest_norm_p95": float(np.percentile(rn, 95)),
                    "rest_norm_rel_iqr": (float(np.percentile(rn, 75)) - float(np.percentile(rn, 25))) / p50,
                    "scale_to_g": 1.0 / p50, "rest_norm_first_half": first, "rest_norm_second_half": second,
                    "half_change_rel": (second - first) / p50 if first is not None and second is not None else None,
                    "cover_x": f"{umin[0]:.3f}..{umax[0]:.3f}", "cover_y": f"{umin[1]:.3f}..{umax[1]:.3f}",
                    "cover_z": f"{umin[2]:.3f}..{umax[2]:.3f}", "covered": covered,
                    "err_scalar_mean_abs": float(np.mean(np.abs(np.linalg.norm(u, axis=1) - 1.0)))})
        if covered and len(u) >= FIT_MIN_BLOCKS:
            sel = np.unique(np.linspace(0, len(u) - 1, min(len(u), FIT_MAX_BLOCKS)).astype(int))
            off, sc, it, conv = ggir_fit(u[sel])
            cal = (u + off) * sc
            row.update({"fit": True, "fit_blocks": int(len(sel)), "fit_iter": it, "fit_converged": conv,
                        "gain_x": float(sc[0]), "gain_y": float(sc[1]), "gain_z": float(sc[2]),
                        "offset_x": float(off[0]), "offset_y": float(off[1]), "offset_z": float(off[2]),
                        "max_abs_gain_dev": float(np.max(np.abs(sc - 1.0))), "max_abs_offset": float(np.max(np.abs(off))),
                        "err_fit_mean_abs": float(np.mean(np.abs(np.linalg.norm(cal, axis=1) - 1.0)))})
        else:
            row["fit"] = False
        out[m] = row
    out["_bytes_read"] = bytes_read
    return out


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    prio = below_normal()
    wait_slow()
    try:
        from moveatlas_stamp import stamp
    except ImportError:
        sys.exit("scripts/moveatlas_stamp.py is missing: every _provenance.json of this tree uses it")
    stamp_start = stamp(ARGS.moveatlas)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, "scale.jsonl")
    if OPT.fresh and os.path.isfile(cache_path):
        os.remove(cache_path)

    # 1. lockbox, then the subject list with the lockbox dropped before any path is built
    lockbox, lockbox_sha = load_lockbox()
    guard = Guard(lockbox)
    listed = sorted(n for n in os.listdir(RAW) if n.startswith("sub-") and os.path.isdir(os.path.join(RAW, n)))
    missing_lb = sorted(lockbox - set(listed))
    require(not missing_lb, f"lockbox subjects absent from the raw tree: {missing_lb}")
    kept = [s for s in listed if s not in lockbox]
    dropped = [s for s in listed if s in lockbox]
    require(len(dropped) == len(lockbox) == 25 and len(kept) == len(listed) - 25,
            f"lockbox drop: {len(dropped)} dropped, {len(kept)} kept of {len(listed)} listed")
    require(not (set(kept) & lockbox), "a lockbox subject is in the kept list")
    print(f"subjects listed {len(listed)}, lockbox dropped {len(dropped)}, kept {len(kept)}; priority below normal: {prio}", flush=True)

    # person map (identity expected)
    pmap = {}
    with open(PERSON_MAP, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source"] == "seizeit2":
                pmap["sub-" + row["subject_ns"].split(":", 1)[1]] = row["person_id"]
    non_identity = [s for s in kept if pmap.get(s) != "seizeit2:" + s[4:]]
    require(not non_identity, f"seizeit2 person map is not the identity for {non_identity[:5]}")

    # events.json levels (dataset-level file)
    with guard.open_meta(f"{RAW}/events.json", encoding="utf-8") as f:
        levels = json.load(f)["eventType"]["Levels"]

    # 2. runs.csv and contained_runs.csv, lockbox rows dropped as parsed
    runs = {}
    runs_lockbox_rows = 0
    with open(os.path.join(SYNC, "runs.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["subject"] in lockbox:
                runs_lockbox_rows += 1
                continue
            runs[row["key"]] = {"subject": row["subject"], "verdict": row["verdict"],
                                "has_emg_edf": row["has_emg_edf"] == "true", "has_mov_edf": row["has_mov_edf"] == "true",
                                "acc_ecgemg_module": row["acc_ecgemg_module"] == "true",
                                "eeg_duration_s": fnum(row["eeg_duration_s"]), "mov_duration_s": fnum(row["mov_duration_s"])}
    contained, contained_partial = {}, []
    with open(os.path.join(SYNC, "contained_runs.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subs = set(SUB_RE.findall(row["run_a"] + " " + row["run_b"]))
            if subs & lockbox:
                continue
            if row["contained_all_modalities"] == "true":
                contained[row["run_a"]] = {"run_b": row["run_b"], "offset_records": int(row["offset_records"])}
            else:                        # contained in some modalities only: kept, and named in the summary
                contained_partial.append(row["run_a"])
    require(all(k.split("_")[0] not in lockbox for k in runs), "a lockbox row of runs.csv was kept")

    # 3. events files of the kept subjects only
    ev_paths, other_folders = [], Counter()
    for s in kept:
        for p in sorted(glob.glob(f"{RAW}/{s}/ses-*/*/*_events.tsv")):
            if os.path.basename(p).startswith("._"):
                continue
            folder = os.path.basename(os.path.dirname(p))
            if folder != "eeg":
                other_folders[folder] += 1
            ev_paths.append(p)
    events, ev_files, ev_keys = [], 0, set()
    cols_seen = Counter()
    for p in ev_paths:
        m = EV_RE.match(os.path.basename(p))
        require(m, f"events file name not parsed: {p}")
        sub, ses, task, run = m.groups()
        key = f"{sub}_{ses}_task-{task}_run-{run}"
        require(sub not in lockbox, f"lockbox events file listed: {p}")
        ev_keys.add(key)
        ev_files += 1
        with guard.open(p, encoding="utf-8", newline="", kind="events_tsv") as f:
            rd = csv.DictReader(f, delimiter="\t")
            cols_seen[tuple(rd.fieldnames or ())] += 1
            for row in rd:
                lab = (row.get("eventType") or "").strip()
                cls, _, _ = class_of(lab)
                events.append({"key": key, "subject": sub, "onset_s": fnum(row.get("onset")),
                               "duration_s": fnum(row.get("duration")), "label": lab, "class": cls,
                               "lateralization": row.get("lateralization", ""), "localization": row.get("localization", ""),
                               "vigilance": row.get("vigilance", ""), "confidence": row.get("confidence", ""),
                               "recording_duration_s": fnum(row.get("recordingDuration"))})
    print(f"events files read {ev_files}; rows {len(events)}", flush=True)
    keys_not_in_runs = sorted(ev_keys - set(runs))
    runs_without_events = sorted(set(runs) - ev_keys)

    # 4. mapping draft
    labels_seen = Counter(e["label"] for e in events)
    map_rows = []
    for lab in sorted(set(levels) | set(labels_seen)):
        cls, rule, note = class_of(lab)
        map_rows.append({"raw_label": lab, "class_draft": cls, "rule": rule, "note": note,
                         "in_events_json": lab in levels, "description": levels.get(lab, ""),
                         "events_nonlockbox": labels_seen.get(lab, 0)})
    write_csv("label_class_map_draft.csv", map_rows,
              ["raw_label", "class_draft", "rule", "note", "in_events_json", "events_nonlockbox", "description"])

    # 5. per-seizure table and scopes
    def in_scope(e, scope):
        rr = runs.get(e["key"])
        if scope == "all_runs":
            return True
        if e["key"] in contained:
            return False
        if scope == "all_minus_contained":
            return True
        if scope == "chest_module_emg_minus_contained":
            return rr is not None and rr["acc_ecgemg_module"] and rr["has_emg_edf"]
        if rr is None or rr["verdict"] != "pass":
            return False
        if scope == "pass_minus_contained":
            return True
        return rr["has_emg_edf"]

    seizures = [e for e in events if e["class"] != "not_seizure"]
    for e in seizures:
        rr = runs.get(e["key"], {})
        e["verdict"] = rr.get("verdict", "")
        e["contained_run"] = e["key"] in contained
        e["has_emg_edf"] = rr.get("has_emg_edf", "")
        md = rr.get("mov_duration_s")
        end = (e["onset_s"] or 0) + (e["duration_s"] or 0)
        e["mov_duration_s"] = md
        e["inside_mov"] = (md is not None and e["onset_s"] is not None and end <= md + 1e-6) if rr.get("has_mov_edf") else False
        for sc, _ in SCOPES:
            e["in_" + sc] = in_scope(e, sc)
    write_csv("seizures.csv", seizures,
              ["key", "subject", "onset_s", "duration_s", "label", "class", "lateralization", "localization", "vigilance",
               "confidence", "recording_duration_s", "verdict", "contained_run", "has_emg_edf", "mov_duration_s", "inside_mov"]
              + ["in_" + sc for sc, _ in SCOPES])

    # census over all runs, one row per raw label
    cen = defaultdict(lambda: {"events": 0, "subjects": set(), "runs": set(), "dur": []})
    for e in events:
        c = cen[e["label"]]
        c["events"] += 1
        c["subjects"].add(e["subject"])
        c["runs"].add(e["key"])
        if e["duration_s"] is not None:
            c["dur"].append(e["duration_s"])
    cen_rows = []
    for lab in sorted(cen, key=lambda x: (CLASSES.index(class_of(x)[0]), x)):
        c = cen[lab]
        cen_rows.append({"raw_label": lab, "class_draft": class_of(lab)[0], "events": c["events"],
                         "subjects": len(c["subjects"]), "runs": len(c["runs"]),
                         "total_duration_s": r(sum(c["dur"]), 1) if c["dur"] else None,
                         "median_duration_s": r(float(np.median(c["dur"])), 1) if c["dur"] else None,
                         "in_events_json": lab in levels})
    write_csv("events_census.csv", cen_rows,
              ["raw_label", "class_draft", "events", "subjects", "runs", "total_duration_s", "median_duration_s", "in_events_json"])

    # class counts per scope
    count_rows, feas_rows, vig_rows = [], [], []
    scope_summary = {}
    for sc, desc in SCOPES:
        sel = [e for e in seizures if e["in_" + sc]]
        rk = [k for k in ev_keys if in_scope({"key": k}, sc)]
        nruns = len(rk)
        hours = sum((runs[k]["eeg_duration_s"] or 0) for k in rk if k in runs) / 3600.0
        per_class = {}
        for cls in SEIZURE_CLASSES:
            cs = [e for e in sel if e["class"] == cls]
            subj = Counter(e["subject"] for e in cs)
            top4 = sum(n for _, n in subj.most_common(4))
            row = {"scope": sc, "class": cls, "seizures": len(cs), "patients": len(subj),
                   "runs": len({e["key"] for e in cs}),
                   "patients_2plus": sum(1 for n in subj.values() if n >= 2),
                   "top4_patients_seizures": top4, "top4_share": r(top4 / len(cs), 3) if cs else None,
                   "inside_mov": sum(1 for e in cs if e["inside_mov"])}
            count_rows.append(row)
            per_class[cls] = row
            vc = Counter(e["vigilance"] or "n/a" for e in cs)
            for v in sorted(vc):
                vig_rows.append({"scope": sc, "class": cls, "vigilance": v, "seizures": vc[v],
                                 "patients": len({e["subject"] for e in cs if (e["vigilance"] or "n/a") == v})})
        hk = {e["subject"] for e in sel if e["class"] == "hyperkinetic"}
        nm = {e["subject"] for e in sel if e["class"] == "explicitly_non_motor"}
        feas = {}
        for cls in PRIMARY + SECONDARY:
            p = per_class[cls]["patients"]
            feas[cls] = p >= MIN_PATIENTS
            feas_rows.append({"scope": sc, "class": cls, "role": "primary" if cls in PRIMARY else "secondary (tonic against hyperkinetic)",
                              "patients": p, "seizures": per_class[cls]["seizures"], "min_patients": MIN_PATIENTS,
                              "feasible": p >= MIN_PATIENTS, "patients_short": max(0, MIN_PATIENTS - p)})
        scope_summary[sc] = {"description": desc, "runs": nruns, "hours": r(hours, 1), "seizures": len(sel),
                             "patients_with_seizure": len({e["subject"] for e in sel}),
                             "patients_hyperkinetic_and_non_motor": len(hk & nm),
                             "primary_feasible": all(feas[c] for c in PRIMARY),
                             "tonic_contrast_feasible": feas["tonic"] and feas["hyperkinetic"],
                             "classes": {c: {k: per_class[c][k] for k in ("seizures", "patients", "runs", "top4_share", "inside_mov")}
                                         for c in SEIZURE_CLASSES}}
    write_csv("class_counts.csv", count_rows,
              ["scope", "class", "seizures", "patients", "runs", "patients_2plus", "top4_patients_seizures", "top4_share", "inside_mov"])
    write_csv("class_vigilance.csv", vig_rows, ["scope", "class", "vigilance", "seizures", "patients"])
    write_csv("feasibility.csv", feas_rows,
              ["scope", "class", "role", "patients", "seizures", "min_patients", "feasible", "patients_short"])

    # per subject
    ps_rows = []
    for s in kept:
        rk = [k for k in runs if runs[k]["subject"] == s]
        row = {"subject": s, "runs": len(rk), "events_files": sum(1 for k in ev_keys if k.startswith(s + "_")),
               "pass_runs": sum(1 for k in rk if runs[k]["verdict"] == "pass" and k not in contained)}
        for cls in PRIMARY + SECONDARY + ("unknown_or_unspecified",):
            row[cls + "_all"] = sum(1 for e in seizures if e["subject"] == s and e["class"] == cls and e["in_all_minus_contained"])
            row[cls + "_pass"] = sum(1 for e in seizures if e["subject"] == s and e["class"] == cls and e["in_pass_minus_contained"])
        row["seizures_all"] = sum(1 for e in seizures if e["subject"] == s and e["in_all_minus_contained"])
        row["seizures_pass"] = sum(1 for e in seizures if e["subject"] == s and e["in_pass_minus_contained"])
        ps_rows.append(row)
    ps_fields = ["subject", "runs", "events_files", "pass_runs", "seizures_all", "seizures_pass"]
    for cls in PRIMARY + SECONDARY + ("unknown_or_unspecified",):
        ps_fields += [cls + "_all", cls + "_pass"]
    write_csv("per_subject.csv", ps_rows, ps_fields)

    # contained runs: are their events duplicated in the containing run?
    cont_rows = []
    for ka, info in sorted(contained.items()):
        ea = [e for e in seizures if e["key"] == ka]
        eb = [e for e in seizures if e["key"] == info["run_b"]]
        for e in ea:
            match = [x for x in eb if x["label"] == e["label"] and x["duration_s"] == e["duration_s"]]
            shift = [r((x["onset_s"] or 0) - (e["onset_s"] or 0), 2) for x in match]
            cont_rows.append({"run_a": ka, "run_b": info["run_b"], "offset_records": info["offset_records"],
                              "onset_s": e["onset_s"], "label": e["label"], "duration_s": e["duration_s"],
                              "matches_in_run_b": len(match), "onset_shift_s": ";".join(str(x) for x in shift)})
    write_csv("contained_events.csv", cont_rows,
              ["run_a", "run_b", "offset_records", "onset_s", "label", "duration_s", "matches_in_run_b", "onset_shift_s"])

    # print feasibility
    print("\nQ8 feasibility on the non-lockbox persons (at least %d patients per class):" % MIN_PATIENTS)
    wsc = max(len(sc) for sc, _ in SCOPES)
    print(f"{'scope':<{wsc}} {'class':<22} {'patients':>8} {'seizures':>8}  feasible")
    for fr in feas_rows:
        print(f"{fr['scope']:<{wsc}} {fr['class']:<22} {fr['patients']:>8} {fr['seizures']:>8}  {'yes' if fr['feasible'] else 'no'}")
    print(flush=True)

    # 6. scale pass
    scale_rows, bytes_read = [], 0
    if not OPT.no_scale:
        done = {}
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        done[d["key"]] = d
        mov_paths = []
        for s in kept:
            for p in sorted(glob.glob(f"{RAW}/{s}/ses-*/mov/*_mov.edf")):
                if not os.path.basename(p).startswith("._"):
                    mov_paths.append(p)
        if OPT.limit:
            mov_paths = mov_paths[:OPT.limit]
        print(f"scale pass: {len(mov_paths)} mov EDFs ({len(done)} cached)", flush=True)
        t1 = time.time()
        with open(cache_path, "a", encoding="utf-8", newline="\n") as cf:
            for i, p in enumerate(mov_paths):
                m = MOV_RE.match(os.path.basename(p))
                require(m, f"mov file name not parsed: {p}")
                sub, ses, task, run = m.groups()
                require(sub not in lockbox, f"lockbox mov file listed: {p}")
                key = f"{sub}_{ses}_task-{task}_run-{run}"
                if key in done:
                    continue
                wait_slow()
                hd = edf_header(p, guard)
                mods = {}
                for mname, labels in MODULES:
                    if all(lab in hd["sig"]["label"] for lab in labels):
                        mods[mname] = [chan_info(hd, lab) for lab in labels]
                res = {"key": key, "subject": sub, "size_ok": hd["size_ok"], "duration_s": hd["nrec"] * hd["recdur"],
                       "modules": {}}
                if mods and hd["size_ok"]:
                    sres = scale_file(p, hd, mods, guard)
                    res["bytes_read"] = sres.pop("_bytes_read", 0)
                    res["modules"] = sres
                cf.write(json.dumps(res) + "\n")
                cf.flush()
                done[key] = res
                if (i + 1) % 100 == 0:
                    el = time.time() - t1
                    print(f"  {i + 1}/{len(mov_paths)} files, {el / 60:.1f} min", flush=True)
        keys_order = [MOV_RE.match(os.path.basename(p)) for p in mov_paths]
        for mm in keys_order:
            sub, ses, task, run = mm.groups()
            key = f"{sub}_{ses}_task-{task}_run-{run}"
            res = done[key]
            bytes_read += res.get("bytes_read", 0)
            rr = runs.get(key, {})
            for mname, _ in MODULES:
                mres = res["modules"].get(mname)
                row = {"key": key, "subject": sub, "module": mname, "verdict": rr.get("verdict", ""),
                       "duration_s": res["duration_s"], "size_ok": res["size_ok"]}
                if mres is None:
                    row["status"] = "module_absent" if res["size_ok"] else "size_mismatch"
                else:
                    row.update(mres)
                scale_rows.append(row)
    sfields = ["key", "subject", "module", "verdict", "status", "duration_s", "size_ok", "fs_hz", "physdim", "header_ranges",
               "n_blocks", "blocks_zero", "blocks_saturated", "blocks_implausible", "implausible_frac_xyz", "blocks_valid", "median_block_norm",
               "blocks_rest", "rest_hours", "rest_norm_p5", "rest_norm_p25", "rest_norm_p50", "rest_norm_p75", "rest_norm_p95",
               "rest_norm_rel_iqr", "scale_to_g", "rest_norm_first_half", "rest_norm_second_half", "half_change_rel",
               "cover_x", "cover_y", "cover_z", "covered", "err_scalar_mean_abs", "fit", "fit_blocks", "fit_iter",
               "fit_converged", "gain_x", "gain_y", "gain_z", "offset_x", "offset_y", "offset_z", "max_abs_gain_dev",
               "max_abs_offset", "err_fit_mean_abs", "sync_window_raw_norm_p50", "ratio_to_sync"]

    # 7. scale summary
    scale_summary = {}
    sync_lb_rows = 0
    if scale_rows:
        # cross-check: median raw norm of the sync probe's anchor windows (non-lockbox rows, gap and flat left out)
        sync_norm = defaultdict(lambda: {"eeg_module": [], "ecgemg_module": []})
        with open(os.path.join(SYNC, "windows.csv"), encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["subject"] in lockbox:
                    sync_lb_rows += 1
                    continue
                if row["status"] in ("gap", "flat"):
                    continue
                for mname, col in (("eeg_module", "raw_norm_eeg"), ("ecgemg_module", "raw_norm_ecgemg")):
                    v = fnum(row[col])
                    if v is not None and v <= IMPLAUSIBLE_ABS:
                        sync_norm[row["key"]][mname].append(v)
        for x in scale_rows:
            v = sync_norm.get(x["key"], {}).get(x["module"]) if x["key"] in sync_norm else None
            x["sync_window_raw_norm_p50"] = float(np.median(v)) if v else None
            x["ratio_to_sync"] = (x["rest_norm_p50"] / x["sync_window_raw_norm_p50"]
                                  if v and x.get("status") == "ok" else None)
        write_csv("acc_scale_per_file.csv",
                  [{k: (r(v, 6) if isinstance(v, float) else v) for k, v in row.items()} for row in scale_rows], sfields)
        # per subject
        subj_rows = []
        by = defaultdict(list)
        for x in scale_rows:
            if x.get("status") == "ok":
                by[(x["subject"], x["module"])].append(x["rest_norm_p50"])
        for (s, mname), v in sorted(by.items()):
            fb = Counter(bin_of(x) for x in v)
            subj_rows.append({"subject": s, "module": mname, "files_ok": len(v), "rest_norm_p50_median": r(float(np.median(v)), 5),
                              "rest_norm_p50_min": r(min(v), 5), "rest_norm_p50_max": r(max(v), 5),
                              "max_over_min": r(max(v) / min(v), 4), "scale_to_g_median": r(1 / float(np.median(v)), 4),
                              "bin_of_median": bin_of(float(np.median(v))),
                              "files_per_bin": ";".join(f"{b} {n}" for b, n in sorted(fb.items(), key=lambda t: BIN_LABELS.index(t[0])))})
        write_csv("acc_scale_per_subject.csv", subj_rows,
                  ["subject", "module", "files_ok", "rest_norm_p50_median", "rest_norm_p50_min", "rest_norm_p50_max",
                   "max_over_min", "scale_to_g_median", "bin_of_median", "files_per_bin"])
        for mname, _ in MODULES:
            rows = [x for x in scale_rows if x["module"] == mname]
            ok = [x for x in rows if x.get("status") == "ok"]
            st = Counter(x.get("status") for x in rows)
            sc = [x["scale_to_g"] for x in ok]
            rn = [x["rest_norm_p50"] for x in ok]
            pooled = float(np.median(rn)) if rn else None
            dev = [abs(x["rest_norm_p50"] / pooled - 1) for x in ok] if pooled else []
            by_sub = defaultdict(list)
            for x in ok:
                by_sub[x["subject"]].append(x["rest_norm_p50"])
            sub_ratio = [max(v) / min(v) for v in by_sub.values() if len(v) >= 2]
            fit = [x for x in ok if x.get("fit")]
            scale_summary[mname] = {
                "files": len(rows), "status": dict(st), "files_ok": len(ok), "subjects_ok": len(by_sub),
                "rest_hours_total": r(sum(x["rest_hours"] for x in ok), 1),
                "rest_blocks_p50": q([x["blocks_rest"] for x in ok], 50),
                "rest_norm_p50": {p: r(q(rn, p), 4) for p in (5, 25, 50, 75, 95)},
                "scale_to_g": {p: r(q(sc, p), 3) for p in (5, 25, 50, 75, 95)},
                "within_file_rel_iqr": {p: r(q([x["rest_norm_rel_iqr"] for x in ok], p), 4) for p in (50, 90)},
                "half_change_rel_abs": {p: r(q([abs(x["half_change_rel"]) for x in ok if x.get("half_change_rel") is not None], p), 4) for p in (50, 90)},
                "files_off_pooled_by_5pct": sum(1 for d in dev if d > 0.05),
                "files_off_pooled_by_10pct": sum(1 for d in dev if d > 0.10),
                "within_subject_max_min_ratio": {p: r(q(sub_ratio, p), 3) for p in (50, 90, 100)},
                "files_covered": sum(1 for x in ok if x.get("covered")),
                "files_fit": len(fit), "fit_converged": sum(1 for x in fit if x.get("fit_converged")),
                "fit_max_abs_gain_dev": {p: r(q([x["max_abs_gain_dev"] for x in fit], p), 4) for p in (50, 90)},
                "fit_max_abs_offset": {p: r(q([x["max_abs_offset"] for x in fit], p), 4) for p in (50, 90)},
                "err_scalar_mean_abs_fit_files": r(q([x["err_scalar_mean_abs"] for x in fit], 50), 4),
                "err_fit_mean_abs": r(q([x["err_fit_mean_abs"] for x in fit], 50), 4),
                "err_scalar_mean_abs_all": {p: r(q([x["err_scalar_mean_abs"] for x in ok], p), 4) for p in (50, 90)},
                "rest_norm_bins_files": {b: sum(1 for v in rn if bin_of(v) == b) for b in BIN_LABELS},
                "rest_norm_bins_subjects": {b: sum(1 for v in by_sub.values() if bin_of(float(np.median(v))) == b)
                                            for b in BIN_LABELS},
                "subject_median_rest_norm": {p: r(q([float(np.median(v)) for v in by_sub.values()], p), 4) for p in (5, 50, 95)},
                "sync_crosscheck_files": sum(1 for x in ok if x.get("ratio_to_sync") is not None),
                "sync_crosscheck_ratio": {p: r(q([x["ratio_to_sync"] for x in ok if x.get("ratio_to_sync") is not None], p), 4)
                                          for p in (5, 50, 95)},
                "sync_crosscheck_within_5pct": sum(1 for x in ok if x.get("ratio_to_sync") is not None
                                                   and abs(x["ratio_to_sync"] - 1) <= 0.05),
                "files_axis_implausible_over_half": sum(1 for x in rows if x.get("implausible_frac_xyz")
                                                        and max(float(t) for t in x["implausible_frac_xyz"].split(";")) > 0.5),
                "subjects_files_in_several_bins": {s: dict(Counter(bin_of(x) for x in v))
                                                   for s, v in sorted(by_sub.items()) if max(v) / min(v) > 2},
                "subject_runs_by_bin": subject_runs([(s, bin_of(float(np.median(v)))) for s, v in sorted(by_sub.items())]),
            }
        both = defaultdict(dict)
        for x in scale_rows:
            if x.get("status") == "ok":
                both[x["key"]][x["module"]] = x["rest_norm_p50"]
        ratios = [v["eeg_module"] / v["ecgemg_module"] for v in both.values() if len(v) == 2]
        scale_summary["module_ratio_eeg_over_ecgemg"] = {"files": len(ratios),
                                                         **{f"p{p}": r(q(ratios, p), 4) for p in (5, 50, 95)}}
        pass_ok = [x for x in scale_rows if x.get("status") == "ok" and x["verdict"] == "pass"]
        scale_summary["pass_runs"] = {"files_ok_module_rows": len(pass_ok),
                                      "scale_to_g_p50": r(q([x["scale_to_g"] for x in pass_ok], 50), 3)}
        scale_summary["mov_bytes_read"] = bytes_read

    # 8. lockbox assertion
    opened_subjects = set(guard.opened)
    require(not (opened_subjects & lockbox), "a lockbox subject was opened")
    require(opened_subjects <= set(kept), "a subject outside the kept list was opened")
    lb_check = {"lockbox_csv_sha256": lockbox_sha, "seizeit2_lockbox_persons": len(lockbox),
                "lockbox_ids_sha256": hashlib.sha256(",".join(sorted(lockbox)).encode()).hexdigest(),
                "subjects_listed": len(listed), "lockbox_dropped_before_reading": len(dropped),
                "subjects_kept": len(kept), "subjects_opened": len(opened_subjects),
                "lockbox_subjects_opened": len(opened_subjects & lockbox),
                "files_opened_by_kind": dict(guard.by_kind), "dataset_level_files_opened": guard.meta,
                "runs_csv_lockbox_rows_discarded": runs_lockbox_rows,
                "windows_csv_lockbox_rows_discarded": sync_lb_rows,
                "assertions": "passed: every raw file was opened through the guard; no path named a lockbox subject"}
    with open(os.path.join(OUT, "lockbox_check.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(lb_check, f, indent=1)
    print(f"lockbox check: {len(dropped)} lockbox subjects dropped before reading, 0 opened "
          f"({sum(guard.opened.values())} files opened over {len(opened_subjects)} subjects)", flush=True)

    # 9. summary
    pass_runs_nl = [k for k, v in runs.items() if v["verdict"] == "pass"]
    S = {"task": "T28", "elapsed_s": r(time.time() - t0, 1), "subjects_kept": len(kept), "lockbox_dropped": len(dropped),
         "events_files": ev_files, "events_rows": len(events), "seizure_rows": len(seizures),
         "events_column_sets": {"|".join(k): v for k, v in cols_seen.items()},
         "events_files_outside_eeg_folder": dict(other_folders),
         "events_keys_not_in_runs_csv": keys_not_in_runs, "runs_csv_keys_without_events_file": len(runs_without_events),
         "runs_nonlockbox": len(runs), "pass_runs_nonlockbox": len(pass_runs_nl),
         "pass_runs_nonlockbox_subjects": len({runs[k]["subject"] for k in pass_runs_nl}),
         "contained_runs_nonlockbox": sorted(contained),
         "contained_runs_partial_nonlockbox_kept": sorted(contained_partial), "labels_seen": len(labels_seen),
         "labels_not_in_events_json": sorted(x for x in labels_seen if x not in levels),
         "class_totals_all_runs": dict(Counter(e["class"] for e in events)),
         "seizures_without_duration": sum(1 for e in seizures if e["duration_s"] is None),
         "events_negative_duration_by_label": dict(Counter(e["label"] for e in events
                                                           if e["duration_s"] is not None and e["duration_s"] < 0)),
         "seizures_beyond_recording": sum(1 for e in seizures if e["recording_duration_s"] is not None and e["onset_s"] is not None
                                          and e["onset_s"] + (e["duration_s"] or 0) > e["recording_duration_s"] + 1e-6),
         "scopes": scope_summary, "contained_events": cont_rows, "scale": scale_summary,
         "vigilance": {sc: {cls: {x["vigilance"]: x["seizures"] for x in vig_rows if x["scope"] == sc and x["class"] == cls}
                            for cls in PRIMARY + SECONDARY} for sc, _ in SCOPES},
         "parameters": {"MIN_PATIENTS": MIN_PATIENTS, "BLOCK_S": BLOCK_S, "REST_SD_REL": REST_SD_REL,
                        "IMPLAUSIBLE_ABS": IMPLAUSIBLE_ABS, "COVER": COVER, "FIT_MIN_BLOCKS": FIT_MIN_BLOCKS,
                        "FIT_MAX_BLOCKS": FIT_MAX_BLOCKS, "limit": OPT.limit, "scale_pass": not OPT.no_scale},
         "lockbox_check": lb_check}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(S, f, indent=1)
    write_summary_md(S, cen_rows, feas_rows, map_rows)

    stamp_end = stamp(ARGS.moveatlas)
    same = all(stamp_start[k] == stamp_end[k] for k in ("head", "diff_sha256", "untracked"))
    prov = {"script": os.path.basename(__file__), "script_sha256": sha256_file(os.path.abspath(__file__)),
            "task": "T28", "moveatlas_root": ARGS.moveatlas,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "moveatlas_stamp": stamp_start, "moveatlas_unchanged_during_run": same, "raw": RAW,
            "inputs": {"lockbox_persons.csv": lockbox_sha,
                       "sync_seizeit2/runs.csv": sha256_file(os.path.join(SYNC, "runs.csv")),
                       "sync_seizeit2/contained_runs.csv": sha256_file(os.path.join(SYNC, "contained_runs.csv")),
                       "person_ids/person_map.csv": sha256_file(PERSON_MAP),
                       **({"sync_seizeit2/windows.csv": sha256_file(os.path.join(SYNC, "windows.csv"))}
                          if scale_rows else {})},
            "parameters": S["parameters"],
            "never_read": ["files of the 25 seizeit2 encoder-lockbox persons (NR-12.2)", "participants.tsv",
                           "EDF Annotations signal (not decoded)", "eeg, ecg and emg EDF signals"]}
    with open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1, default=str)
    if not OPT.keep_cache and not OPT.limit and not OPT.no_scale and os.path.isfile(cache_path):
        os.remove(cache_path)
    try:
        os.rmdir(CACHE)                  # removed only when empty (a partial scale cache stays)
    except OSError:
        pass
    print(f"done in {(time.time() - t0) / 60:.1f} min; outputs in {OUT}", flush=True)


def write_summary_md(S, cen_rows, feas_rows, map_rows):
    L = []
    a = L.append
    a("# seizeit2 seizure events and Q8 feasibility, non-lockbox persons only (task T28)")
    a("")
    a("Generated by scripts/probe_seizeit2_events.py; every number below is read from summary.json and the CSV files of "
      "this directory. MoveAtlas state in _provenance.json (scripts/moveatlas_stamp.py).")
    a("")
    a("## Scope and lockbox")
    a("")
    lb = S["lockbox_check"]
    a(f"- Subjects in the raw tree: {lb['subjects_listed']}. The {lb['lockbox_dropped_before_reading']} seizeit2 persons of the "
      f"encoder lockbox (NR-12.2) were dropped from the subject list before any path inside their folders was built; "
      f"lockbox subjects opened: {lb['lockbox_subjects_opened']} (asserted). Files opened: "
      + ", ".join(f"{k} {v}" for k, v in sorted(lb["files_opened_by_kind"].items()))
      + f", over {lb['subjects_opened']} subjects. Lockbox rows discarded as the sync files were parsed: runs.csv "
      f"{lb['runs_csv_lockbox_rows_discarded']}, windows.csv {lb['windows_csv_lockbox_rows_discarded']}.")
    a(f"- Every count below covers the {S['subjects_kept']} non-lockbox persons only. The full-cohort census of the addendum "
      "(1.5: 883 seizures over all 2,850 events files of the 125 persons) includes the 25 lockbox persons; "
      "reproducing or updating it needs the owner's permission to read the lockbox persons' labels. No difference between "
      "that figure and the counts here is computed, since it would describe the lockbox persons' labels.")
    a(f"- Because the lockbox persons can only add patients, a class that reaches {MIN_PATIENTS} patients here reaches it on the "
      "full cohort; a class short of it here could reach it only through lockbox persons, at most 25 more.")
    a("")
    a("## Census")
    a("")
    a(f"- Events files read: {S['events_files']} (all in the eeg folder: "
      f"{'yes' if not S['events_files_outside_eeg_folder'] else S['events_files_outside_eeg_folder']}); rows {S['events_rows']}; "
      f"seizure rows (every label except bckg and impd) {S['seizure_rows']}.")
    a(f"- Column sets: " + "; ".join(f"{k} ({v} files)" for k, v in S["events_column_sets"].items()) + ".")
    a(f"- Distinct labels met: {S['labels_seen']}; labels absent from events.json: "
      f"{', '.join(S['labels_not_in_events_json']) if S['labels_not_in_events_json'] else 'none'}.")
    a(f"- Events files without a runs.csv row: {len(S['events_keys_not_in_runs_csv'])}; non-lockbox runs.csv rows without "
      f"an events file: {S['runs_csv_keys_without_events_file']}.")
    a(f"- Seizures without a duration: {S['seizures_without_duration']}; ending after the recording duration of their "
      f"events file: {S['seizures_beyond_recording']}. Events with a negative duration: "
      + (", ".join(f"{k} {v}" for k, v in sorted(S["events_negative_duration_by_label"].items())) or "none") + ".")
    a("")
    a("| raw label | draft class | seizures or events | subjects | runs | median duration (s) |")
    a("|---|---|---|---|---|---|")
    for x in cen_rows:
        a(f"| {x['raw_label']} | {x['class_draft']} | {x['events']} | {x['subjects']} | {x['runs']} | {fmt(x['median_duration_s'])} |")
    a("")
    a("## Label to class mapping (draft for OPEN-48)")
    a("")
    a("label_class_map_draft.csv lists every level of events.json and every label met on disk, with the rule that "
      "placed it and a note where the placement is debatable. Rules, in order: bckg and impd are not seizures; a motor "
      "label ending in hyperkinetic is hyperkinetic; any label carrying nm is explicitly non-motor; sz_foc_f2b is focal "
      "to bilateral; a motor label ending in automatisms is automatisms; a motor label ending in tonic is tonic; a label "
      "carrying um, or one giving onset or awareness only (sz, sz_foc, sz_foc_a, sz_foc_ia, sz_uo, sz_gen), is unknown or "
      "unspecified; a motor label without a subtype is motor unspecified; any other motor label is other motor.")
    a("")
    notes = [x for x in map_rows if x["note"]]
    a("Placements to confirm:")
    a("")
    for x in notes:
        a(f"- {x['raw_label']} ({x['class_draft']}, {x['events_nonlockbox']} events here): {x['note']}.")
    a("")
    a("Unknown or unspecified and motor unspecified are never negatives (NR-6.8.2); only explicitly non-motor is.")
    a("")
    a("## Class counts by scope")
    a("")
    a("| scope | runs | hours | seizures | patients with a seizure | patients with hyperkinetic and non-motor |")
    a("|---|---|---|---|---|---|")
    for sc, _ in SCOPES:
        s = S["scopes"][sc]
        a(f"| {sc} | {s['runs']} | {fmt(s['hours'])} | {s['seizures']} | {s['patients_with_seizure']} | {s['patients_hyperkinetic_and_non_motor']} |")
    a("")
    a("Scopes: " + "; ".join(f"{sc}, {d}" for sc, d in SCOPES) + ". Non-lockbox runs in runs.csv: "
      f"{S['runs_nonlockbox']}; with verdict = pass: {S['pass_runs_nonlockbox']} of {S['pass_runs_nonlockbox_subjects']} subjects. "
      f"Contained runs removed: {', '.join(S['contained_runs_nonlockbox']) or 'none'}; runs of contained_runs.csv "
      f"contained in some modalities only, kept: {', '.join(S['contained_runs_partial_nonlockbox_kept']) or 'none'}.")
    a("")
    a("Seizures (patients) per class:")
    a("")
    a("| class | " + " | ".join(sc for sc, _ in SCOPES) + " |")
    a("|---|" + "---|" * len(SCOPES))
    for cls in SEIZURE_CLASSES:
        cells = []
        for sc, _ in SCOPES:
            c = S["scopes"][sc]["classes"][cls]
            cells.append(f"{c['seizures']} ({c['patients']})")
        if all(cell == "0 (0)" for cell in cells):
            continue
        a(f"| {cls} | " + " | ".join(cells) + " |")
    a("")
    hk = S["scopes"]["all_runs"]["classes"]["hyperkinetic"]
    a(f"Hyperkinetic concentration, all runs: the top 4 patients hold {fmt(hk['top4_share'])} of the hyperkinetic seizures "
      "(class_counts.csv, top4_share, for every scope). Vigilance per class and scope: class_vigilance.csv.")
    a("")
    for sc in ("all_minus_contained", "chest_module_emg_minus_contained"):
        parts = []
        for cls in PRIMARY + SECONDARY:
            v = S["vigilance"][sc][cls]
            tot = sum(v.values())
            parts.append(f"{cls} asleep {v.get('asleep', 0)}, awake {v.get('awake', 0)}, unknown {v.get('un', 0)} of {tot}")
        a(f"Vigilance, {sc}: " + "; ".join(parts) + ".")
    ce = S["contained_events"]
    a("")
    if ce:
        a(f"Contained runs: {len(ce)} seizure rows, of which {sum(1 for x in ce if x['matches_in_run_b'])} "
          "repeat in the containing run with the same label and duration (onset shifts in contained_events.csv), so "
          "all_runs counts them twice and all_minus_contained once.")
    else:
        a(f"Contained runs ({', '.join(S['contained_runs_nonlockbox']) or 'none'}): no seizure row, so removing them "
          "changes no seizure count.")
    a("")
    a("## Feasibility (at least 10 patients per class, NR-6.8.2)")
    a("")
    a("| scope | class | role | patients | seizures | feasible |")
    a("|---|---|---|---|---|---|")
    for fr in feas_rows:
        a(f"| {fr['scope']} | {fr['class']} | {fr['role']} | {fr['patients']} | {fr['seizures']} | {'yes' if fr['feasible'] else 'no, short by ' + str(fr['patients_short'])} |")
    a("")
    for sc, _ in SCOPES:
        s = S["scopes"][sc]
        a(f"- {sc}: primary contrast {'feasible' if s['primary_feasible'] else 'not feasible'} on the non-lockbox persons; "
          f"tonic against hyperkinetic {'feasible' if s['tonic_contrast_feasible'] else 'not feasible'}.")
    a("")
    pm = S["scopes"]["pass_minus_contained"]
    a(f"Under the draft default of NR-2.7 (verdict = pass only) and NR-2.15, the admitted non-lockbox runs give "
      f"{pm['classes']['hyperkinetic']['patients']} hyperkinetic and {pm['classes']['explicitly_non_motor']['patients']} "
      f"explicitly non-motor patients. Seizures of the admitted scope that end inside the mov EDF: hyperkinetic "
      f"{pm['classes']['hyperkinetic']['inside_mov']} of {pm['classes']['hyperkinetic']['seizures']}, explicitly non-motor "
      f"{pm['classes']['explicitly_non_motor']['inside_mov']} of {pm['classes']['explicitly_non_motor']['seizures']}.")
    a("")
    a("## Accelerometer scale per file (input to OPEN-09)")
    a("")
    sc = S["scale"]
    if not sc:
        a("Scale pass not run.")
    else:
        a(f"Label-free: no seizure annotation enters this pass and no descriptor is computed per seizure class. mov EDF bytes "
          f"read: {sc['mov_bytes_read'] / 1e9:.2f} GB. Rest block: 10 s with every axis SD below 0.013 times the file's "
          "median block-mean norm; the scale to g is the inverse of the file's median rest norm (header unit g).")
        a("")
        a("| quantity | EEG module | ECG-EMG module |")
        a("|---|---|---|")
        e, c = sc["eeg_module"], sc["ecgemg_module"]

        def row(label, fe, fc):
            a(f"| {label} | {fe} | {fc} |")
        row("files (status ok)", f"{e['files']} ({e['files_ok']})", f"{c['files']} ({c['files_ok']})")
        row("status counts", ", ".join(f"{k} {v}" for k, v in sorted(e["status"].items())),
            ", ".join(f"{k} {v}" for k, v in sorted(c["status"].items())))
        row("files with an axis beyond +-4 header units in more than half of its samples",
            e["files_axis_implausible_over_half"], c["files_axis_implausible_over_half"])
        row("subjects with a scale", e["subjects_ok"], c["subjects_ok"])
        row("rest hours", fmt(e["rest_hours_total"]), fmt(c["rest_hours_total"]))
        row("median rest norm p5, p25, p50, p75, p95", ", ".join(fmt(e["rest_norm_p50"][p]) for p in (5, 25, 50, 75, 95)),
            ", ".join(fmt(c["rest_norm_p50"][p]) for p in (5, 25, 50, 75, 95)))
        row("scale to g p5, p50, p95", ", ".join(fmt(e["scale_to_g"][p]) for p in (5, 50, 95)),
            ", ".join(fmt(c["scale_to_g"][p]) for p in (5, 50, 95)))
        row("files off the pooled median by more than 5 / 10 percent", f"{e['files_off_pooled_by_5pct']} / {e['files_off_pooled_by_10pct']}",
            f"{c['files_off_pooled_by_5pct']} / {c['files_off_pooled_by_10pct']}")
        row("within-file relative IQR of the rest norm, p50 and p90", f"{fmt(e['within_file_rel_iqr'][50])}, {fmt(e['within_file_rel_iqr'][90])}",
            f"{fmt(c['within_file_rel_iqr'][50])}, {fmt(c['within_file_rel_iqr'][90])}")
        row("first against second half, absolute relative change p50 and p90",
            f"{fmt(e['half_change_rel_abs'][50])}, {fmt(e['half_change_rel_abs'][90])}",
            f"{fmt(c['half_change_rel_abs'][50])}, {fmt(c['half_change_rel_abs'][90])}")
        row("within-subject max over min file scale, p50, p90, max",
            ", ".join(fmt(e["within_subject_max_min_ratio"][p]) for p in (50, 90, 100)),
            ", ".join(fmt(c["within_subject_max_min_ratio"][p]) for p in (50, 90, 100)))
        row("files with orientation coverage (fit run, converged)", f"{e['files_covered']} ({e['files_fit']}, {e['fit_converged']})",
            f"{c['files_covered']} ({c['files_fit']}, {c['fit_converged']})")
        row("per-axis fit: max gain deviation p50, p90", f"{fmt(e['fit_max_abs_gain_dev'][50])}, {fmt(e['fit_max_abs_gain_dev'][90])}",
            f"{fmt(c['fit_max_abs_gain_dev'][50])}, {fmt(c['fit_max_abs_gain_dev'][90])}")
        row("per-axis fit: max offset (g after scaling) p50, p90", f"{fmt(e['fit_max_abs_offset'][50])}, {fmt(e['fit_max_abs_offset'][90])}",
            f"{fmt(c['fit_max_abs_offset'][50])}, {fmt(c['fit_max_abs_offset'][90])}")
        row("rest-norm error, scalar scale against per-axis fit (fit files, p50)",
            f"{fmt(e['err_scalar_mean_abs_fit_files'])} against {fmt(e['err_fit_mean_abs'])}",
            f"{fmt(c['err_scalar_mean_abs_fit_files'])} against {fmt(c['err_fit_mean_abs'])}")
        row("rest-norm error with the scalar scale, all files, p50 and p90",
            f"{fmt(e['err_scalar_mean_abs_all'][50])}, {fmt(e['err_scalar_mean_abs_all'][90])}",
            f"{fmt(c['err_scalar_mean_abs_all'][50])}, {fmt(c['err_scalar_mean_abs_all'][90])}")
        row("subject median rest norm p5, p50, p95", ", ".join(fmt(e["subject_median_rest_norm"][p]) for p in (5, 50, 95)),
            ", ".join(fmt(c["subject_median_rest_norm"][p]) for p in (5, 50, 95)))
        row("cross-check: this rest norm over the sync probe's window raw norm, p5, p50, p95 (files; within 5 percent)",
            ", ".join(fmt(e["sync_crosscheck_ratio"][p]) for p in (5, 50, 95)) + f" ({e['sync_crosscheck_files']}; {e['sync_crosscheck_within_5pct']})",
            ", ".join(fmt(c["sync_crosscheck_ratio"][p]) for p in (5, 50, 95)) + f" ({c['sync_crosscheck_files']}; {c['sync_crosscheck_within_5pct']})")
        a("")
        a("Median rest norm per file and per subject, by bin (header units):")
        a("")
        a("| bin | EEG module files | EEG module subjects | ECG-EMG module files | ECG-EMG module subjects |")
        a("|---|---|---|---|---|")
        for b in e["rest_norm_bins_files"]:
            a(f"| {b} | {e['rest_norm_bins_files'][b]} | {e['rest_norm_bins_subjects'][b]} | "
              f"{c['rest_norm_bins_files'][b]} | {c['rest_norm_bins_subjects'][b]} |")
        a("")
        for mname, lab in (("eeg_module", "EEG module"), ("ecgemg_module", "ECG-EMG module")):
            mm = sc[mname]
            several = mm["subjects_files_in_several_bins"]
            a(f"{lab}: subjects whose file rest norms differ by more than a factor 2 (files per bin): {len(several)}"
              + (" (" + "; ".join(f"{s}: " + ", ".join(f"{b} {n}" for b, n in d.items()) for s, d in several.items()) + ")"
                 if several else "") + ".")
            a("")
            a(f"{lab}, consecutive subject ids (lockbox ids absent) by the bin of the subject median: "
              + "; ".join(mm["subject_runs_by_bin"]) + ".")
            a("")
        a("Per subject: acc_scale_per_subject.csv.")
        a("")
        mr = sc["module_ratio_eeg_over_ecgemg"]
        a(f"EEG-module over ECG-EMG-module median rest norm in the same file: p5 {fmt(mr['p5'])}, p50 {fmt(mr['p50'])}, "
          f"p95 {fmt(mr['p95'])} ({mr['files']} files). Verdict = pass runs: {sc['pass_runs']['files_ok_module_rows']} module "
          f"rows, scale to g p50 {fmt(sc['pass_runs']['scale_to_g_p50'])}. Per file and module: acc_scale_per_file.csv.")
    a("")
    a("## Reading")
    a("")
    a("- The counts are label counts, not a test: no body descriptor, score or threshold is computed per seizure class, and "
      "seizeit2 labels train nothing (NR-6.8.6). Epileptic hyperkinetic seizures are not hyperkinetic movement disorders.")
    a("- The class mapping is a draft for OPEN-48; the owner fixes it before the rule file is hashed.")
    a("- The admitted scope follows the draft default of NR-2.7 (OPEN-08) and the draft NR-2.15 (OPEN-07). The scope "
      "chest_module_emg_minus_contained shows what a single-module reading would admit if OPEN-08 let the chest-module "
      "accelerometer and the deltoid EMG (one device) read void and not checkable runs; it assumes that the annotations "
      "can be placed on that module's clock, which is the second half of OPEN-08.")
    a("- The scale estimates use the non-lockbox persons only, so no lockbox person enters a normalization statistic "
      "(NR-12.2); a per-file scale for a lockbox file would need its own read.")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
