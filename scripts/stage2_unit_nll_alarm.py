"""S2CA: the unit_nll_rise alarm that stopped run S2c at phase C step 8000, measured (task S2CA, 2026-10-15).

The alarm (mdfm/train/loop.py ALARMS, "unit_nll_rise": ("dev", "above", 0.0, 0); unit_nll_rising, UNIT_NLL_READS 3)
reads the dev hook of scripts/train_stage1.py: at every dev read (the start, every dev_every steps, the phase end) the
hook evaluates the raw (not EMA) weights on UNIT_DEV_WINDOWS fixed windows of each stage-2 unit's val fold and as
many of its training persons (unit_dev_sets: drawn once when the hook is built, from numpy default_rng([DEV_SEED, unit
index, part index]); loop.evaluate with one posterior draw seeded DEV_SEED + micro-step index), and logs per unit the
kernel NLL (the sum of losses.KERNEL_TERMS) of each part. A unit is rising when, over its last three reads, each
held-out NLL is above the one before and each train NLL below the one before; the alarm fires on one read with at
least one rising unit. This script measures how noisy those reads are and how often that pattern occurs by chance.

Parts (each number of summary.md comes from results.json of this folder):
1. Logs (always): the dev reads of cache/runs/stage2_s2a, stage2_s2b and stage2_s2c (log.jsonl, the last event of
   each step, as train_stage1.unit_history reads them); the checks that the windows are fixed (windows and knots per
   unit and part equal at every read of every run; the start reads of the three runs, whose weights are the same);
   the alarm recounted with loop.unit_nll_rising; per unit and run, from step 1000 on: the linear trend, the residual
   spread, the successive differences, the lag-1 autocorrelation, the gap (held-out minus train) and its trend; S2b
   against S2c, two runs of one recipe, seed and plan stream, read by read (the replicate spread); every three-read
   window of the logs counted (held-out rising, train falling, both); the log records of S2b and S2c compared step by
   step (data of the plan: windows, buckets, fill, token shares; weights: the health record and the loss) and S2c's
   two calls compared where they overlap (6050 to 6400); the speed of the runs.
2. Model: the reads of a unit as independent draws around a level (no trend, or the measured trend) with the
   measured held-out and train spreads and their correlation, three units independent; per reading of the alarm
   (the loop's, k reads in a row, alarm_consecutive k on the loop's, a magnitude floor) the share of phases C that it
   stops by step 8000 and by the phase end (Monte Carlo, SIM_N phases, seed SIM_SEED), and the reads it takes to fire
   on a steady overfit (held-out up and train down by a drift per read) and on an S2a-like overfit (imu's held-out
   NLL rising at S2a's measured slope); each reading also run on the three logs, with what each log's stop censors
   (the reads in a row at its last read) and what not stopping S2a at its last read costs.
3. Plausibility: the share of decoded knots outside the joint limits per unit, part and read (the hook's
   outside_limits_share).
4. Recompute (--recompute, CPU): the hook's windows rebuilt from the stage-2 store (unit_dev_sets' calls), checked
   against the logged knots; per window the kernel NLL under the raw and the EMA weights of the S2b and S2c
   checkpoints, with evaluate's seeds on the CPU generator (the GPU's generator draws another stream, so a logged
   value is matched up to its draw); the spread over other draws at fixed weights; the paired per-window changes with
   person-cluster bootstrap intervals; the pre-registered predictions P1 to P3 of predictions.txt.
5. The healthy lines of S2c's stop checkpoint (--diagnostic, when the folder exists): the read that
   scripts/stage2_read.py wrote, and prediction P4.

Reads (read only): log.jsonl, drift and checkpoints of the run folders (torch.load through eval.infer.load_checkpoint),
configs/stage2_s2c.yaml, the stage-2 window store (the hook's unit windows: val folds and training persons of the
stage-2 units), the joint-limit table, predictions.txt and predictions.sha256 of this folder. No lockbox, EVAL, CODY
or care_pd record, no GPU.

Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python scripts/stage2_unit_nll_alarm.py [--recompute]
      [--diagnostic reports/bench/stage2_read/stage2_s2c_c8000_diagnostic] [--write]
Writes reports/bench/stage2_unit_nll_alarm/results.json, summary.md and _provenance.json with --write (never
predictions.txt or predictions.sha256).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

TASK = "S2CA"
OUT = os.path.join("reports", "bench", "stage2_unit_nll_alarm")
RUNS = ("stage2_s2a", "stage2_s2b", "stage2_s2c")
RUNS_DIR = os.path.join("cache", "runs")
CONFIG = os.path.join("configs", "stage2_s2c.yaml")
PARTS = ("held_out", "train")
FIRST_READ = 1000                    # reads from this step on (the start read sits far above every later one)
PHASE_C_STEPS = 35754                # stage2.steps of the S2a, S2b and S2c configs
DEV_EVERY = 1000                     # schedule.dev_every of the same configs
ALARM = ("stage2_s2c", 8000, "pads:all")
REPLICATE = ("stage2_s2b", "stage2_s2c")   # same recipe, seed and plan stream (part 1)
SIM_N, SIM_SEED = 200000, 20261015
DRIFTS = (0.5, 1.0)                  # steady overfit, in units of the read spread per read (part 2)
MAG_Z = (2.0, 3.0)                   # magnitude floors of the readings (part 2)
OUTSIDE_LINE = 0.10                  # a plausibility reading beside the NLL readings (part 3)
CKPTS = (2000, 4000, 6000, 8000)     # the checkpoints that S2b and S2c both hold at their dev reads
RECOMPUTE_UNITS = ("pads:all", "imu_inthewild_iprognosis:all", "clinical_gait_multipatho:all")
DRAWS = 8                            # other posterior draws at fixed weights (part 4)
DRAW_SEED_STEP = 7919
BOOT_N, BOOT_SEED = 10000, 20261015
PREDICTIONS = "predictions.txt"
S2B_DIAGNOSTIC = os.path.join("reports", "bench", "stage2_read", "stage2_s2b_c8550_diagnostic")   # S2BA's read


def rel(p):
    return os.path.relpath(os.path.abspath(p), TREE).replace("\\", "/")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    out, call = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "start":
                call += 1
            r["_call"] = call
            out.append(r)
    return out


def fnum(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def ols(x, y):
    """(slope, standard error) of y on x by least squares; (None, None) under 3 points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None, None
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    res = y - y.mean() - b * xc
    return b, float(math.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))


def binom_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return float(sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp
                              + (n - i) * lq) for i in range(0, k + 1)))


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    low = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < a)
    high = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > a)
    return [low, high]


# ---------------------------------------------------------------- part 1: the logs
def dev_reads(recs):
    """{step: {trigger, call, unit_nll_rise, units {unit: {part: {nll, out, knots, windows} or None}}}} of the 'dev'
    events of phase C, the last event of each step (train_stage1.unit_history)."""
    out = {}
    for r in recs:
        if r.get("event") != "dev" or r.get("phase") != "C":
            continue
        m = (r.get("dev") or {}).get("model") or {}
        us = m.get("units")
        if not us:
            continue
        units = {}
        for u, e in us.items():
            units[u] = {}
            for p in PARTS:
                x = e.get(p)
                units[u][p] = None if x is None else {"nll": x.get("nll"), "out": x.get("outside_limits_share"),
                                                      "knots": x.get("outside_limits_knots"),
                                                      "windows": x.get("windows")}
        out[int(r["step"])] = {"trigger": r.get("trigger"), "call": r["_call"], "unit_nll_rise": m.get("unit_nll_rise"),
                               "rising": m.get("unit_nll_rise_units"), "units": units}
    return dict(sorted(out.items()))


def unit_series(reads, unit, part, first=FIRST_READ):
    xs = [(s, r["units"][unit][part]["nll"]) for s, r in reads.items()
          if s >= first and unit in r["units"] and r["units"][unit][part] is not None]
    return [s for s, _ in xs], [float(v) for _, v in xs]


def series_stats(steps, vals):
    s, v = np.asarray(steps, float), np.asarray(vals, float)
    if len(v) < 3:
        return None
    b, se = ols(s, v)
    res = v - v.mean() - b * (s - s.mean())
    d = np.diff(v)
    ac = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 3 else None
    return {"reads": int(len(v)), "steps": [int(s[0]), int(s[-1])], "mean": float(v.mean()), "first": float(v[0]),
            "last": float(v[-1]), "min": float(v.min()), "max": float(v.max()),
            "slope_per_1000": 1000.0 * b, "slope_se_per_1000": 1000.0 * se,
            "residual_sd": float(math.sqrt((res ** 2).sum() / (len(v) - 2))),
            "diff_sd_over_sqrt2": float(d.std(ddof=1) / math.sqrt(2)) if len(d) > 1 else None,
            "diff_mean": float(d.mean()), "residual_lag1_autocorrelation": ac}


def rising_history(reads, units, loop):
    """Per read: the units loop.unit_nll_rising lists over the history up to that read (all reads, the start's
    included, as the hook keeps them)."""
    hist = {u: [] for u in units}
    out = {}
    for s, r in reads.items():
        for u in units:
            e = r["units"].get(u)
            if e is None:
                continue
            hist[u].append(tuple(None if e[p] is None else e[p]["nll"] for p in PARTS))
        out[s] = loop.unit_nll_rising(hist)
    return out


def triples(reads, units, first=0):
    """Every window of three consecutive reads (from `first`) of each unit with both parts: counts of held-out
    rising twice, train falling twice, both (the loop's pattern)."""
    n = up = down = both = 0
    hits = []
    for u in units:
        xs = [(s, r["units"][u]) for s, r in reads.items() if s >= first and u in r["units"]]
        xs = [(s, e["held_out"]["nll"], e["train"]["nll"]) for s, e in xs
              if e["held_out"] is not None and e["train"] is not None]
        for i in range(2, len(xs)):
            a, b, c = xs[i - 2], xs[i - 1], xs[i]
            n += 1
            r1 = b[1] > a[1] and c[1] > b[1]
            f1 = b[2] < a[2] and c[2] < b[2]
            up += r1
            down += f1
            both += r1 and f1
            if r1 and f1:
                hits.append({"unit": u, "steps": [a[0], b[0], c[0]]})
    return {"windows": n, "held_out_rising": up, "train_falling": down, "both": both, "hits": hits}


def compare_logs(rb, rc):
    """S2b against S2c, 'log' records step by step (module docstring, part 1)."""
    def data(r):
        return (r.get("windows"), json.dumps(r.get("buckets"), sort_keys=True), round(float(r.get("fill") or 0), 12),
                json.dumps({k: round(v, 12) for k, v in (r.get("token_share") or {}).items()}, sort_keys=True),
                json.dumps({k: round(v, 12) for k, v in (r.get("window_share_phase") or {}).items()}, sort_keys=True))
    lb = {r["step"]: r for r in rb if r.get("event") == "log"}
    calls = sorted({r["_call"] for r in rc if r.get("event") == "log"})
    lc = {k: {r["step"]: r for r in rc if r.get("event") == "log" and r["_call"] == k} for k in calls}
    rows = []
    for k in calls:
        for s in sorted(set(lb) & set(lc[k])):
            a, b = lb[s], lc[k][s]
            ha, hb = (a.get("health") or {}).get("beta_error"), (b.get("health") or {}).get("beta_error")
            la, lc_ = a.get("loss"), b.get("loss")
            rows.append({"step": s, "call": k, "data_equal": data(a) == data(b),
                         "beta_b": ha, "beta_c": hb,
                         "beta_abs_diff": None if ha is None or hb is None else abs(ha - hb),
                         "loss_rel_diff": None if la is None or lc_ is None else (lc_ - la) / abs(la),
                         "lr_equal": a.get("lr") == b.get("lr")})
    # S2c's two calls where they overlap
    overlap = []
    if len(calls) > 1:
        c1, c2 = lc[calls[0]], lc[calls[1]]
        for s in sorted(set(c1) & set(c2)):
            a, b = c1[s], c2[s]
            ha, hb = (a.get("health") or {}).get("beta_error"), (b.get("health") or {}).get("beta_error")
            overlap.append({"step": s, "data_equal": data(a) == data(b), "beta_1": ha, "beta_2": hb,
                            "beta_abs_diff": None if ha is None or hb is None else abs(ha - hb),
                            "loss_1": a.get("loss"), "loss_2": b.get("loss"),
                            "loss_rel_diff": None if a.get("loss") is None or b.get("loss") is None
                            else (b["loss"] - a["loss"]) / abs(a["loss"]),
                            "lr_equal": a.get("lr") == b.get("lr")})
    first = rows[0] if rows else None
    diverge = next((r["step"] for r in rows if r["beta_abs_diff"] is not None and r["beta_abs_diff"] > 1e-4), None)
    bands = {}
    for lo, hi in ((50, 50), (100, 500), (550, 1000), (1050, 3000), (3050, 6000), (6050, 8000)):
        xs = [abs(r["loss_rel_diff"]) for r in rows if lo <= r["step"] <= hi and r["loss_rel_diff"] is not None
              and r["call"] == (calls[-1] if r["step"] > 6000 else calls[0])]
        bs = [r["beta_abs_diff"] for r in rows if lo <= r["step"] <= hi and r["beta_abs_diff"] is not None
              and r["call"] == (calls[-1] if r["step"] > 6000 else calls[0])]
        bands[f"{lo}-{hi}"] = {"records": len(xs), "loss_rel_abs_median": float(np.median(xs)) if xs else None,
                               "beta_abs_median": float(np.median(bs)) if bs else None}
    return {"records": len(rows), "steps": len({r["step"] for r in rows}),
            "records_per_call": {str(k): sum(1 for r in rows if r["call"] == k) for k in calls},
            "data_equal_all": all(r["data_equal"] for r in rows),
            "data_unequal_steps": [r["step"] for r in rows if not r["data_equal"]],
            "lr_equal_all": all(r["lr_equal"] for r in rows),
            "first_record": first, "first_beta_divergence_step": diverge, "bands": bands,
            "overlap_calls": overlap, "calls": calls}


def speed(recs):
    """Seconds per optimizer step of each call from the 'log' records' times (dev reads and checkpoints in)."""
    out = {}
    for k in sorted({r["_call"] for r in recs if r.get("event") == "log"}):
        xs = [(r["step"], r.get("time")) for r in recs if r.get("event") == "log" and r["_call"] == k and r.get("time")]
        if len(xs) < 2:
            continue
        t = [time.mktime(time.strptime(x[1], "%Y-%m-%dT%H:%M:%S")) for x in xs]
        out[str(k)] = {"first_step": xs[0][0], "last_step": xs[-1][0],
                       "s_per_step": (t[-1] - t[0]) / (xs[-1][0] - xs[0][0])}
    return out


def logs_part(tree=TREE):
    from mdfm.train import loop as LP
    recs = {run: read_jsonl(os.path.join(tree, RUNS_DIR, run, "log.jsonl")) for run in RUNS}
    reads = {run: dev_reads(recs[run]) for run in RUNS}
    units = list(next(iter(reads["stage2_s2c"].values()))["units"])
    val_units = [u for u in units if next(iter(reads["stage2_s2c"].values()))["units"][u]["held_out"] is not None]
    # the windows are fixed: windows and knots per unit and part, every read of every run
    fixed = {}
    for u in units:
        for p in PARTS:
            ks = {(r["units"][u][p] or {}).get("knots") for run in RUNS for r in reads[run].values()
                  if u in r["units"]}
            ws = {(r["units"][u][p] or {}).get("windows") for run in RUNS for r in reads[run].values()
                  if u in r["units"]}
            fixed[f"{u}|{p}"] = {"knots": sorted(k for k in ks if k is not None), "windows": sorted(w for w in ws if w)}
    start = {}
    for u in units:
        for p in PARTS:
            vs = [(reads[run].get(0) or {}).get("units", {}).get(u, {}).get(p) for run in RUNS]
            vs = [v["nll"] for v in vs if v]
            if vs:
                start[f"{u}|{p}"] = {"values": vs, "range": max(vs) - min(vs)}
    # the alarm recounted
    recount = {}
    for run in RUNS:
        rh = rising_history(reads[run], units, LP)
        recount[run] = {str(s): {"recount": v, "logged": reads[run][s]["rising"]} for s, v in rh.items()}
    recount_ok = all(v["recount"] == (v["logged"] or []) for run in RUNS for v in recount[run].values())
    # per unit and run
    per = {}
    for run in RUNS:
        per[run] = {}
        for u in units:
            per[run][u] = {}
            for p in PARTS:
                st, vs = unit_series(reads[run], u, p)
                per[run][u][p] = {"series": dict(zip(map(str, st), vs)), "stats": series_stats(st, vs)}
            if u in val_units:
                st, ho = unit_series(reads[run], u, "held_out")
                _, tr = unit_series(reads[run], u, "train")
                gap = [a - b for a, b in zip(ho, tr)]
                dh, dt = np.diff(ho), np.diff(tr)
                x = np.asarray(st, float)
                rh = np.asarray(ho) - np.polyval(np.polyfit(x, ho, 1), x)
                rt = np.asarray(tr) - np.polyval(np.polyfit(x, tr, 1), x)
                per[run][u]["gap"] = {"series": dict(zip(map(str, st), gap)), "stats": series_stats(st, gap),
                                      "diff_correlation_held_out_train": float(np.corrcoef(dh, dt)[0, 1])
                                      if len(dh) > 2 else None,
                                      "residual_correlation_held_out_train": float(np.corrcoef(rh, rt)[0, 1])
                                      if len(rh) > 3 else None}
    # replicate: S2b minus S2c read by read
    rep = {}
    ra, rb_ = REPLICATE
    for u in units:
        rep[u] = {}
        for p in PARTS + ("gap",):
            if p == "gap" and u not in val_units:
                continue
            sa = per[ra][u][p]["series"] if p in per[ra][u] else None
            sb = per[rb_][u][p]["series"] if p in per[rb_][u] else None
            if not sa or not sb:
                continue
            common = sorted(set(sa) & set(sb), key=int)
            d = {s: sa[s] - sb[s] for s in common}
            d2 = [d[s] for s in common if int(s) >= 2 * FIRST_READ]
            rep[u][p] = {"differences": d, "sd_over_sqrt2_from_2000": float(np.std(d2, ddof=1) / math.sqrt(2))
                         if len(d2) > 1 else None,
                         "rms_over_sqrt2_from_2000": float(math.sqrt(np.mean(np.square(d2)) / 2)) if d2 else None,
                         "reads_from_2000": len(d2)}
        if u in val_units:
            common = sorted(set(rep[u]["held_out"]["differences"]) & set(rep[u]["train"]["differences"]), key=int)
            common = [s for s in common if int(s) >= 2 * FIRST_READ]
            x = [rep[u]["held_out"]["differences"][s] for s in common]
            y = [rep[u]["train"]["differences"][s] for s in common]
            rep[u]["held_out_train_correlation_from_2000"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None
    # the windows of three reads
    tri = {run: {"all_reads": triples(reads[run], val_units, 0), "from_1000": triples(reads[run], val_units, FIRST_READ)}
           for run in RUNS}
    pooled = {}
    for key in ("all_reads", "from_1000"):
        n = sum(tri[run][key]["windows"] for run in RUNS)
        k = sum(tri[run][key]["both"] for run in RUNS)
        up = sum(tri[run][key]["held_out_rising"] for run in RUNS)
        dn = sum(tri[run][key]["train_falling"] for run in RUNS)
        pooled[key] = {"windows": n, "both": k, "share": k / n if n else None,
                       "interval": clopper_pearson(k, n) if n else None,
                       "held_out_rising": up, "held_out_rising_share": up / n if n else None,
                       "train_falling": dn, "train_falling_share": dn / n if n else None,
                       "expected_iid_independent": n / 36.0}
    # the alarms of the logs
    alarms = {run: [{"step": r["step"], "call": r["_call"], "alarms": r.get("alarms")} for r in recs[run]
                    if r.get("event") == "alarm"] for run in RUNS}
    nonfinite = {run: [r["step"] for r in recs[run] if r.get("event") == "nonfinite"] for run in RUNS}
    resumes = {run: [{"step": r.get("step"), "path": rel(r.get("path", "")) if r.get("path") else None}
                     for r in recs[run] if r.get("event") == "resume"] for run in RUNS}
    snaps = sorted(n for n in os.listdir(os.path.join(tree, RUNS_DIR, "stage2_s2c"))
                   if n.startswith("code_snapshot_") and n.endswith(".zip"))
    snap_sha = {n: sha256_file(os.path.join(tree, RUNS_DIR, "stage2_s2c", n)) for n in snaps}
    comp = compare_logs(recs["stage2_s2b"], recs["stage2_s2c"])
    shares = {}
    for run in RUNS:
        last = [r for r in recs[run] if r.get("event") == "log"][-1]
        shares[run] = {"step": last["step"], "window_share_phase": last.get("window_share_phase"),
                       "token_share_interval": last.get("token_share")}
    # the dev reads of S2c's two calls: the call of each read
    calls_of_reads = {str(s): r["call"] for s, r in reads["stage2_s2c"].items()}
    return {"units": units, "val_units": val_units, "fixed_windows": fixed, "start_reads": start,
            "recount_matches_log": recount_ok, "recount": recount, "per_run": per, "replicate": rep,
            "triples": tri, "triples_pooled": pooled, "alarms": alarms, "nonfinite": nonfinite, "resumes": resumes,
            "s2c_code_snapshots": snap_sha, "compare_s2b_s2c": comp, "s2c_read_calls": calls_of_reads,
            "speed": {run: speed(recs[run]) for run in RUNS}, "shares_last_record": shares,
            "dev_read_steps": {run: list(reads[run]) for run in RUNS}}


# ---------------------------------------------------------------- part 2: the model
def phase_reads(steps=PHASE_C_STEPS, every=DEV_EVERY, first=FIRST_READ):
    """The dev reads of a phase C from `first` on: every `every` steps before the phase end (the loop's interval
    reads, step < steps) and the phase end."""
    return list(range(first, steps, every)) + [steps]


def rising_k(ho, tr, k):
    """(N, R) bool: at read t, the last k reads of held-out rising and train falling, each step (loop's pattern for
    k = 3)."""
    up = ho[:, 1:] > ho[:, :-1]
    dn = tr[:, 1:] < tr[:, :-1]
    ok = up & dn
    n, r = ho.shape
    out = np.zeros((n, r), bool)
    m = k - 1
    if r <= m:
        return out
    acc = np.ones((n, r - m), bool)
    for j in range(m):
        acc &= ok[:, j:j + r - m]
    out[:, m:] = acc
    return out


def fire_step(count, consecutive=1):
    """(N,) index of the first read where `count` > 0 holds `consecutive` reads in a row, -1 if never."""
    n, r = count.shape
    streak = np.zeros(n, int)
    first = np.full(n, -1)
    for t in range(r):
        hit = count[:, t] > 0
        streak = np.where(hit, streak + 1, 0)
        new = (streak >= consecutive) & (first < 0)
        first[new] = t
    return first


# readings of the alarm (part 2): (name, kind, k reads, consecutive reads, z). kind 'pattern': the loop's test over k
# reads (held-out rising and train falling at each step), with z the held-out rise over the k reads above z x sqrt(2)
# x the unit's held-out spread; kind 'gap': the gap (held-out minus train) risen over the k reads by more than z x
# sqrt(2) x the unit's gap spread while the held-out NLL is above its value k - 1 reads before; kind 'baseline': from
# the (k + 1)-th read on, the held-out NLL above the mean of its first k reads (steps 1000 to 3000 for k = 3) by more
# than z x the unit's held-out spread x sqrt(1 + 1/k). consecutive: the reading true at that many dev reads in a row
# (schedule.alarm_consecutive's count, any unit).
READINGS = (("loop_3_reads", "pattern", 3, 1, None), ("consecutive_2", "pattern", 3, 2, None),
            ("consecutive_3", "pattern", 3, 3, None), ("4_reads", "pattern", 4, 1, None),
            ("magnitude_2", "pattern", 3, 1, 2.0), ("gap_z3", "gap", 3, 1, 3.0),
            ("gap_z3_consecutive_2", "gap", 3, 2, 3.0), ("baseline_z3", "baseline", 3, 1, 3.0),
            ("baseline_z3_consecutive_2", "baseline", 3, 2, 3.0))
SCENARIOS = ("A", "B")      # steady overfit: A held-out up and train down by drift x sd per read; B held-out up only
RHO_SENSITIVITY = (0.95, 0.8, 0.6, 0.0)


def unit_hits(kind, ho, tr, k, z, sig_ho, sig_gap):
    """(N, R) bool of one unit under one reading (READINGS)."""
    lag = k - 1
    if kind == "pattern":
        r = rising_k(ho, tr, k)
        if z is not None:
            big = np.zeros_like(r)
            big[:, lag:] = (ho[:, lag:] - ho[:, :-lag]) > z * math.sqrt(2) * sig_ho
            r &= big
        return r
    if kind == "baseline":
        r = np.zeros(ho.shape, bool)
        base = ho[:, :k].mean(1, keepdims=True)
        r[:, k:] = (ho[:, k:] - base) > z * sig_ho * math.sqrt(1.0 + 1.0 / k)
        return r
    g = ho - tr
    r = np.zeros(ho.shape, bool)
    r[:, lag:] = ((g[:, lag:] - g[:, :-lag]) > z * math.sqrt(2) * sig_gap) & (ho[:, lag:] > ho[:, :-lag])
    return r


def reading_counts(series, kind, k, z, sigmas, per_unit=None):
    """(N, R) count of units raised by a reading: series {unit: (ho (N, R), tr (N, R))}, sigmas {unit: (held-out
    spread, gap spread)}; per_unit, a dict, receives each unit's hits."""
    tot = None
    for u, (ho, tr) in series.items():
        r = unit_hits(kind, ho, tr, k, z, *sigmas[u])
        if per_unit is not None:
            per_unit[u] = r
        tot = r.astype(int) if tot is None else tot + r
    return tot


def gap_sd(p):
    return math.sqrt(max(p["sd_ho"] ** 2 + p["sd_tr"] ** 2 - 2 * p["rho"] * p["sd_ho"] * p["sd_tr"], 1e-12))


def simulate(params, n_sim=None, seed=SIM_SEED, drift=None, drift_unit=None, scenario="A", sigmas=None):
    """Monte Carlo of the readings over one phase C (module docstring, part 2). params {unit: {sd_ho, sd_tr, rho,
    trend_ho, trend_tr}} in NLL units per read, the trend applied up to step 8000 (the range it was measured on) and
    held after; drift: a steady overfit of drift x sd per read from the first read in drift_unit (scenario A held-out
    up and train down, B held-out up only); sigmas {unit: (held-out spread, gap spread)} of the magnitude and gap
    readings (default the params')."""
    n_sim = SIM_N if n_sim is None else int(n_sim)
    rng = np.random.default_rng(seed)
    reads = phase_reads()
    R = len(reads)
    by8000 = sum(1 for s in reads if s <= 8000)
    sig = sigmas or {u: (p["sd_ho"], gap_sd(p)) for u, p in params.items()}
    out = {name: {"fire_by_8000": 0, "fire_phase": 0, "first_reads": [],
                  "unit_phase": dict.fromkeys(params, 0), "unit_reads": dict.fromkeys(params, 0),
                  "unit_reads8": dict.fromkeys(params, 0)}
           for name, *_ in READINGS}
    chunk = 20000
    done = 0
    t = np.minimum(np.arange(R), by8000 - 1)[None, :].astype(float)
    tt = np.arange(R)[None, :].astype(float)
    while done < n_sim:
        n = min(chunk, n_sim - done)
        series = {}
        for u, p in params.items():
            z1 = rng.standard_normal((n, R))
            z2 = rng.standard_normal((n, R))
            rho = max(-0.999, min(0.999, p["rho"]))
            ho = p["sd_ho"] * z1 + p.get("trend_ho", 0.0) * t
            tr = p["sd_tr"] * (rho * z1 + math.sqrt(1 - rho * rho) * z2) + p.get("trend_tr", 0.0) * t
            if drift is not None and u == drift_unit:
                ho = ho + drift * p["sd_ho"] * tt
                if scenario == "A":
                    tr = tr - drift * p["sd_tr"] * tt
            series[u] = (ho, tr)
        for name, kind, k, cons, z in READINGS:
            pu = {}
            f = fire_step(reading_counts(series, kind, k, z, sig, pu), cons)
            out[name]["fire_by_8000"] += int(((f >= 0) & (f < by8000)).sum())
            out[name]["fire_phase"] += int((f >= 0).sum())
            for u, h in pu.items():
                out[name]["unit_phase"][u] += int((fire_step(h.astype(int), cons) >= 0).sum())
                out[name]["unit_reads"][u] += int(h[:, k - 1:].sum())
                out[name]["unit_reads8"][u] += int(h[:, k - 1:by8000].sum())
            if drift is not None:
                out[name]["first_reads"].append(np.where(f >= 0, f + 1, R + 1))
        done += n
    res = {}
    for name, kind, k, cons, z in READINGS:
        o = out[name]
        r = {"fire_by_8000": o["fire_by_8000"] / n_sim, "fire_phase": o["fire_phase"] / n_sim,
             "unit_fire_phase": {u: v / n_sim for u, v in o["unit_phase"].items()},
             "unit_per_read": {u: v / (n_sim * (R - (k if kind == "baseline" else k - 1)))
                               for u, v in o["unit_reads"].items()},
             "unit_per_read_by_8000": {u: v / (n_sim * (by8000 - (k if kind == "baseline" else k - 1)))
                                       for u, v in o["unit_reads8"].items()}}
        if drift is not None:
            fr = np.concatenate(o["first_reads"])
            r.update(reads_to_fire_median=float(np.median(fr)), reads_to_fire_p90=float(np.quantile(fr, 0.9)),
                     never_share=float((fr > R).mean()))
        res[name] = r
    return {"reads_per_phase": R, "reads_by_8000": by8000, "readings": res}


def log_sigmas(L):
    """{unit: (held-out spread, gap spread)} of the replicate (part 1), the floors of the magnitude and gap
    readings."""
    rep = L["replicate"]
    return {u: (rep[u]["held_out"]["rms_over_sqrt2_from_2000"], rep[u]["gap"]["rms_over_sqrt2_from_2000"])
            for u in L["val_units"]}


def readings_on_logs(L):
    """Each reading of READINGS on the logged reads of each run (from step 1000), the units with a val fold; the
    floors at the replicate spreads of part 1 (in-sample: they are measured on the same logs)."""
    sig = log_sigmas(L)
    out = {}
    for run in RUNS:
        series, steps = {}, None
        for u in L["val_units"]:
            ho = L["per_run"][run][u]["held_out"]["series"]
            tr = L["per_run"][run][u]["train"]["series"]
            st = sorted(set(ho) & set(tr), key=int)
            steps = [int(s) for s in st]
            series[u] = (np.array([[ho[s] for s in st]]), np.array([[tr[s] for s in st]]))
        out[run] = {}
        for name, kind, k, cons, z in READINGS:
            pu = {}
            f = int(fire_step(reading_counts(series, kind, k, z, sig, pu), cons)[0])
            out[run][name] = None if f < 0 else {"step": steps[f],
                                                 "units": sorted(u for u, h in pu.items() if h[0, f])}
    return out


def censoring_on_logs(L, runs=RUNS):
    """Each log ends at its run's stop, so a reading that needs more reads than the stop left is censored there. Per
    run and reading of READINGS: the step it fires at (None if it does not within the log), the number of reads in a
    row at the log's last read with a raised unit (the streak), the units raised there, and 'open': not fired and the
    streak above 0, i.e. the next read, which the log does not hold, could complete it. Same floors as
    readings_on_logs."""
    sig = log_sigmas(L)
    out = {}
    for run in runs:
        series, steps = {}, None
        for u in L["val_units"]:
            ho = L["per_run"][run][u]["held_out"]["series"]
            tr = L["per_run"][run][u]["train"]["series"]
            st = sorted(set(ho) & set(tr), key=int)
            steps = [int(s) for s in st]
            series[u] = (np.array([[ho[s] for s in st]]), np.array([[tr[s] for s in st]]))
        out[run] = {"last_read": steps[-1], "readings": {}}
        for name, kind, k, cons, z in READINGS:
            pu = {}
            cnt = reading_counts(series, kind, k, z, sig, pu)
            f = int(fire_step(cnt, cons)[0])
            streak = 0
            for t in range(cnt.shape[1] - 1, -1, -1):
                if cnt[0, t] > 0:
                    streak += 1
                else:
                    break
            out[run]["readings"][name] = {
                "fires": None if f < 0 else steps[f], "consecutive": cons, "streak_at_last_read": streak,
                "units_at_last_read": sorted(u for u, h in pu.items() if h[0, -1]),
                "open": f < 0 and streak > 0}
    return out


def missed_stop_cost(L, run="stage2_s2a"):
    """What a reading that does not stop `run` at its last read costs at least (one dev read more) and at most (the
    rest of phase C), in steps and in GPU-h at S2b's rate and at the run's own rate."""
    last = L["dev_read_steps"][run][-1]
    sb = L["speed"]["stage2_s2b"]["1"]["s_per_step"]
    own = L["speed"][run]["1"]["s_per_step"]
    rest = PHASE_C_STEPS - last
    return {"run": run, "last_read": last, "one_read_steps": DEV_EVERY,
            "one_read_gpu_h": {"s2b_rate": DEV_EVERY * sb / 3600, "own_rate": DEV_EVERY * own / 3600},
            "rest_steps": rest, "rest_gpu_h": {"s2b_rate": rest * sb / 3600, "own_rate": rest * own / 3600}}


S2A_UNIT = "imu_inthewild_iprognosis:all"    # S2a's stop (part 2, the S2a-like overfit)


def model_part(L):
    rep = L["replicate"]
    params, params_trend, check = {}, {}, {}
    for u in L["val_units"]:
        sd_ho = rep[u]["held_out"]["rms_over_sqrt2_from_2000"]
        sd_tr = rep[u]["train"]["rms_over_sqrt2_from_2000"]
        rho = rep[u]["held_out_train_correlation_from_2000"]
        params[u] = {"sd_ho": sd_ho, "sd_tr": sd_tr, "rho": rho}
        params[u]["sd_gap_implied"] = gap_sd(params[u])
        # the trend of S2b and S2c from step 2000 (per read of 1000 steps), averaged
        tr_ho, tr_tr = [], []
        for run in REPLICATE:
            for p, acc in (("held_out", tr_ho), ("train", tr_tr)):
                s = L["per_run"][run][u][p]["series"]
                st = [int(x) for x in s if int(x) >= 2 * FIRST_READ]
                b, _ = ols(st, [s[str(x)] for x in st])
                acc.append(1000.0 * b)
        params_trend[u] = dict(params[u], trend_ho=float(np.mean(tr_ho)), trend_tr=float(np.mean(tr_tr)))
        # within-run successive differences, pooled over the three runs from 2000, beside the replicate spread
        d_ho, d_tr = [], []
        for run in RUNS:
            for p, acc in (("held_out", d_ho), ("train", d_tr)):
                s = L["per_run"][run][u][p]["series"]
                st = [x for x in sorted(s, key=int) if int(x) >= 2 * FIRST_READ]
                acc += list(np.diff([s[x] for x in st]))
        check[u] = {"diff_rms_over_sqrt2_held_out": float(math.sqrt(np.mean(np.square(d_ho)) / 2)),
                    "diff_rms_over_sqrt2_train": float(math.sqrt(np.mean(np.square(d_tr)) / 2)),
                    "diff_correlation": float(np.corrcoef(d_ho, d_tr)[0, 1]), "diffs": len(d_ho)}
    sig = log_sigmas(L)
    # a second estimate: the detrended residuals of S2b's and S2c's reads from 1000 (pooled sd, mean correlation)
    params_resid = {}
    for u in L["val_units"]:
        sh = [L["per_run"][run][u]["held_out"]["stats"]["residual_sd"] for run in REPLICATE]
        st_ = [L["per_run"][run][u]["train"]["stats"]["residual_sd"] for run in REPLICATE]
        rr = [L["per_run"][run][u]["gap"]["residual_correlation_held_out_train"] for run in REPLICATE]
        params_resid[u] = {"sd_ho": float(math.sqrt(np.mean(np.square(sh)))),
                           "sd_tr": float(math.sqrt(np.mean(np.square(st_)))), "rho": float(np.mean(rr)),
                           "rho_runs": rr}
    # exact per-evaluation shares under exchangeable i.i.d. reads, held-out and train independent
    exact = {f"{k}_reads": (1.0 / math.factorial(k)) ** 2 for k in (3, 4, 5)}
    out = {"params": params, "params_trend": params_trend, "within_run_check": check, "exact_per_read": exact,
           "sigmas": {u: list(v) for u, v in sig.items()},
           "null": simulate(params, sigmas=sig),
           "null_rho0": simulate({u: dict(p, rho=0.0) for u, p in params.items()}, sigmas=sig),
           "null_trend": simulate(params_trend, sigmas=sig),
           "params_resid": params_resid,
           "null_resid": simulate(params_resid, sigmas={u: (p["sd_ho"], gap_sd(p)) for u, p in params_resid.items()}),
           "null_resid_replicate_floors": simulate(params_resid, sigmas=sig),
           "logs": readings_on_logs(L)}
    # the pads pattern per read under other correlations (the replicate's value rests on 7 differences)
    a = ALARM[2]
    sens = {}
    for rho in RHO_SENSITIVITY:
        pp = {a: dict(params[a], rho=rho)}
        r = simulate(pp, n_sim=SIM_N // 4, seed=SIM_SEED + 2, sigmas={a: sig[a]})["readings"]["loop_3_reads"]
        sens[str(rho)] = {"per_read": r["unit_per_read"][a], "fire_by_8000": r["fire_by_8000"],
                          "fire_phase": r["fire_phase"]}
    out["pads_rho_sensitivity"] = sens
    # the logs' three-read windows from 1000 against the model with trend (expected count of the loop's pattern)
    exp = {"trend": 0.0, "residual": 0.0, "replicate": 0.0}
    for u in L["val_units"]:
        n_u = sum(max(0, len(L["per_run"][run][u]["held_out"]["series"]) - 2) for run in RUNS)
        for key, var in (("trend", "null_trend"), ("residual", "null_resid"), ("replicate", "null")):
            exp[key] += n_u * out[var]["readings"]["loop_3_reads"]["unit_per_read_by_8000"][u]
    out["expected_on_logs_trend"] = exp["trend"]
    out["expected_on_logs"] = exp
    out["overfit"] = {}
    for sc in SCENARIOS:
        for d in DRIFTS:
            out["overfit"][f"{sc}{d}"] = simulate(params, n_sim=SIM_N // 10, seed=SIM_SEED + 1, drift=d,
                                                  drift_unit=a, scenario=sc, sigmas=sig)
    # the logs end at each run's stop: what each reading would still need there; the cost of not stopping S2a
    out["logs_censoring"] = censoring_on_logs(L)
    out["missed_stop_cost_s2a"] = missed_stop_cost(L, "stage2_s2a")
    # an S2a-like overfit: imu's held-out NLL rising at S2a's measured slope (from 1000 to its stop), in units of
    # imu's replicate held-out spread per read, from the first read; B held-out only (S2a's imu train slope was not
    # negative), A with the train NLL falling at the same multiple of its spread
    s2a_ho = L["per_run"]["stage2_s2a"][S2A_UNIT]["held_out"]["stats"]
    s2a_tr = L["per_run"]["stage2_s2a"][S2A_UNIT]["train"]["stats"]
    d_s2a = s2a_ho["slope_per_1000"] / params[S2A_UNIT]["sd_ho"]
    out["overfit_s2a_like"] = {"unit": S2A_UNIT, "held_out_slope_per_1000": s2a_ho["slope_per_1000"],
                               "held_out_slope_se": s2a_ho["slope_se_per_1000"],
                               "train_slope_per_1000": s2a_tr["slope_per_1000"],
                               "train_slope_se": s2a_tr["slope_se_per_1000"], "sd_ho": params[S2A_UNIT]["sd_ho"],
                               "drift_sd_per_read": d_s2a}
    for sc in SCENARIOS:
        out["overfit_s2a_like"][sc] = simulate(params, n_sim=SIM_N // 10, seed=SIM_SEED + 3, drift=d_s2a,
                                               drift_unit=S2A_UNIT, scenario=sc, sigmas=sig)
    return out


# ---------------------------------------------------------------- the stop against the noise
def event_part(L, M, C=None):
    """The stop's three reads (ALARM) against the read-to-read spreads of part 1: the two-read rise of the held-out
    NLL and of the gap, in units of sqrt(2) x the spread (replicate and within-run estimates); S2b's reads at the
    same steps; with the recompute, its gap change in the same units."""
    run, step, u = ALARM
    steps = [str(step - 2 * DEV_EVERY), str(step - DEV_EVERY), str(step)]
    pr = L["per_run"]
    rep, chk = L["replicate"][u], M["within_run_check"][u]
    out = {"steps": [int(x) for x in steps]}
    for r in REPLICATE:
        out[r] = {p: [pr[r][u][p]["series"][x] for x in steps] for p in ("held_out", "train", "gap")}
    ho, gap = out[run]["held_out"], out[run]["gap"]
    sd_ho, sd_ho_w = rep["held_out"]["rms_over_sqrt2_from_2000"], chk["diff_rms_over_sqrt2_held_out"]
    sd_gap = rep["gap"]["rms_over_sqrt2_from_2000"]
    out["held_out_rise"] = ho[-1] - ho[0]
    out["held_out_rise_in_sd"] = {"replicate": out["held_out_rise"] / (math.sqrt(2) * sd_ho),
                                  "within_runs": out["held_out_rise"] / (math.sqrt(2) * sd_ho_w)}
    out["gap_rise"] = gap[-1] - gap[0]
    out["gap_rise_in_sd"] = out["gap_rise"] / (math.sqrt(2) * sd_gap)
    out["two_read_sd"] = {"held_out_replicate": math.sqrt(2) * sd_ho, "held_out_within_runs": math.sqrt(2) * sd_ho_w,
                          "gap_replicate": math.sqrt(2) * sd_gap}
    if C:
        g = C["paired"][f"{run}|model|{u}"]["gap"]["mean"]
        out["recompute_gap_change"] = g
        out["recompute_gap_change_in_sd"] = g / (math.sqrt(2) * sd_gap)
    return out


# ---------------------------------------------------------------- part 3: plausibility
def plausibility_part(L, recs_reads=None):
    rows = []
    for run in RUNS:
        for s, r in recs_reads[run].items():
            for u, e in r["units"].items():
                for p in PARTS:
                    if e[p] is not None and e[p]["out"] is not None:
                        rows.append((run, s, u, p, float(e[p]["out"])))
    top = sorted(rows, key=lambda x: -x[4])[:8]
    above = [x for x in rows if x[4] > OUTSIDE_LINE]
    rest = [x for x in rows if x[4] <= OUTSIDE_LINE and x[1] >= FIRST_READ]
    pads = {run: {str(s): {p: (r["units"].get(ALARM[2], {}).get(p) or {}).get("out") for p in PARTS}
                  for s, r in recs_reads[run].items()} for run in RUNS}
    return {"line": OUTSIDE_LINE, "reads": len(rows), "top": [list(x) for x in top],
            "above_line": [list(x) for x in above],
            "max_below_line_from_1000": max((x[4] for x in rest), default=None), "pads": pads}


# ---------------------------------------------------------------- part 4: recompute
def _train_stage1():
    spec = importlib.util.spec_from_file_location("_s2ca_train_stage1", os.path.join(HERE, "train_stage1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit_sets(TS, cfg, store, kw, units):
    """{unit: {part: (micro-steps, person ids per window)}}: the calls of train_stage1.unit_dev_sets (same seeds,
    same order), with the windows' persons kept."""
    import torch
    out = {}
    off = cfg.losses.plausibility_windows == "synthetic_replay"
    for k, u in enumerate(cfg.stage2.units):
        if u not in units:
            continue
        src = str(u).split(":", 1)[0]
        out[u] = {}
        for i, part in enumerate(TS.UNIT_PARTS):
            wins, _ = TS.unit_windows(store, src, part == "held_out", TS.UNIT_DEV_WINDOWS, [TS.DEV_SEED, k, i],
                                      cfg.mixture.win_s, cfg.mixture.min_s)
            mbs = TS.micro_batches(cfg, wins, TS.DEV_SEED + 1000 * (k + 1) + 100 * i, kw) if wins else []
            if off:
                for mb in mbs:
                    mb.targets.plausibility_on = torch.zeros(mb.batch.n_windows, dtype=torch.bool)
            out[u][part] = (mbs, [w.person_id for w in wins])
    return out


def fold_persons(TS, cfg, store, units):
    """{unit: {part: persons of the unit's fold whose recordings unit_windows may draw}} (its filter: the source, the
    fold, a span of at least mixture.min_s)."""
    recs = store.recordings
    out = {}
    for u in units:
        src = str(u).split(":", 1)[0]
        out[u] = {}
        for part in TS.UNIT_PARTS:
            fold = recs["fold"] == (TS.UNIT_HELD_OUT_FOLD if part == "held_out" else "train")
            g = recs[(recs["source"] == src) & fold & (recs["span_s"] >= float(cfg.mixture.min_s) - 1e-9)]
            out[u][part] = int(len(set(g["person_id"])))
    return out


def per_window_nll(model, mbs, loss_cfg, limits, seed):
    """Per window, in order, the kernel NLL of loop.evaluate's draw (one posterior draw per micro-step from a
    generator seeded seed + micro-step index, the CPU generator), and the knots read by the outside-limits share."""
    import torch
    from mdfm.train import loop as LP
    from mdfm.train import losses as LS
    dev = torch.device("cpu")
    was = model.training
    model.eval()
    vals, n_lim, n_out = [], 0, 0
    try:
        with torch.no_grad():
            for i, mb in enumerate(mbs):
                batch, targets = mb.batch.to(dev), mb.targets.to(dev)
                gen = torch.Generator(device=dev).manual_seed((int(seed) + i) % 2 ** 63)
                post = LP.encode(model, batch)
                body, pred = LP.forward_draw(model, batch, post, gen, truth_windows=LS.truth_mask(batch, targets, dev))
                out = LS.stage1_losses(model, batch, post, body, pred, LP.loss_phase("C"), targets=targets,
                                       progress=1.0, cfg=loss_cfg)
                w = sum(out[k].float() for k in LS.KERNEL_TERMS)
                assert w.shape == (batch.n_windows,), w.shape
                vals += [float(x) for x in w]
                if limits is not None:
                    from mdfm.model.posterior import subject_pool
                    z = post.mu_z * post.mask[..., None]
                    mu_b, _ = subject_pool(post.mu_b, post.logvar_b, batch.person.to(dev))
                    bo = model.decode(z, mu_b, batch.knots, batch.tokens.n_patches.to(dev), batch.lengths.to(dev))
                    off = LS.outside_limits(bo.rot6d.float(), limits[0].to(dev), limits[1].to(dev))
                    n_out += int(off.sum())
                    n_lim += int(off.numel())
    finally:
        model.train(was)
    return vals, n_lim, n_out


def cluster_boot(changes, persons, n=BOOT_N, rng=None):
    """Mean over windows of `changes` and its percentile 95 percent interval over person resamples."""
    g = {}
    for c, p in zip(changes, persons):
        g.setdefault(p, []).append(c)
    keys = sorted(g)
    sums = np.array([sum(g[k]) for k in keys])
    cnts = np.array([len(g[k]) for k in keys])
    idx = rng.integers(len(keys), size=(n, len(keys)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.mean(changes)), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], means


def recompute_part(tree, L, units=RECOMPUTE_UNITS):
    import torch
    from mdfm.eval import infer as I
    from mdfm.train import config as CF
    from mdfm.train import losses as LS
    from mdfm.train import names as NM
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2") or 2))
    t0 = time.time()
    TS = _train_stage1()
    cfg = CF.load(os.path.join(tree, CONFIG))
    stores = TS.stage2_stores(cfg)
    kw = TS.collate_kw(cfg)
    sets = unit_sets(TS, cfg, stores.stage2, kw, units)
    limits = LS.joint_limits(path=NM.resolve(str(cfg.losses.plausibility_table)))
    loss_cfg = cfg.losses.loss_config()
    t_sets = time.time() - t0
    persons = {u: {p: sets[u][p][1] for p in sets[u]} for u in sets}
    vals, knots, timing = {}, {}, {}
    weights = [(run, st, w) for run in REPLICATE for st in CKPTS for w in ("model", "ema")]
    for run, st, w in weights:
        path = os.path.join(tree, RUNS_DIR, run, f"ckpt_C_{st:06d}.pt")
        t1 = time.time()
        model = I.load_checkpoint(path, weights=w, device="cpu").model
        key = f"{run}:{st}:{w}"
        vals[key] = {}
        for u in sets:
            vals[key][u] = {}
            for p, (mbs, _) in sets[u].items():
                v, nl, _ = per_window_nll(model, mbs, loss_cfg, limits if (w == "model" and st == CKPTS[0]) else None,
                                          TS.DEV_SEED)
                vals[key][u][p] = v
                if nl:
                    knots[f"{u}|{p}"] = nl
        # other draws at fixed weights: the raw weights of S2c's 6000 and 8000 (pads only)
        if run == ALARM[0] and w == "model" and st in (6000, 8000):
            for d in range(1, DRAWS + 1):
                for p, (mbs, _) in sets[ALARM[2]].items():
                    v, _, _ = per_window_nll(model, mbs, loss_cfg, None, TS.DEV_SEED + DRAW_SEED_STEP * d)
                    vals.setdefault(f"{key}:draw{d}", {}).setdefault(ALARM[2], {})[p] = v
        del model
        timing[key] = time.time() - t1
        print(f"recompute {key}: {timing[key]:.0f} s", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    mean = lambda xs: float(np.mean(xs))  # noqa: E731
    # validation against the logged reads (raw weights)
    valid = []
    for run in REPLICATE:
        for st in CKPTS:
            for u in sets:
                for p in sets[u]:
                    logged = L["per_run"][run][u][p]["series"].get(str(st))
                    rec = mean(vals[f"{run}:{st}:model"][u][p])
                    valid.append({"run": run, "step": st, "unit": u, "part": p, "logged": logged, "recomputed": rec,
                                  "difference": None if logged is None else rec - logged})
    knots_check = {k: {"recomputed": v, "logged": L["fixed_windows"][k]["knots"]} for k, v in knots.items()}
    # the draws at fixed weights
    draws = {}
    for st in (6000, 8000):
        base = f"{ALARM[0]}:{st}:model"
        for p in PARTS:
            ms = [mean(vals[base][ALARM[2]][p])] + [mean(vals[f"{base}:draw{d}"][ALARM[2]][p])
                                                   for d in range(1, DRAWS + 1)]
            draws[f"{st}|{p}"] = {"means": ms, "sd": float(np.std(ms, ddof=1))}
    # the means per weight, raw and EMA
    means = {k: {u: {p: mean(v) for p, v in d.items()} for u, d in vals[k].items()} for k in vals if ":draw" not in k}
    # paired per-window changes 6000 to 8000 (and S2b's), person-cluster intervals
    paired = {}
    for run in REPLICATE:
        for w in ("model", "ema"):
            for u in sets:
                ch = {}
                for p in PARTS:
                    a, b = vals[f"{run}:6000:{w}"][u][p], vals[f"{run}:8000:{w}"][u][p]
                    c = [y - x for x, y in zip(a, b)]
                    m, ci, boots = cluster_boot(c, persons[u][p], rng=rng)
                    ch[p] = {"mean": m, "interval": ci, "boots": boots,
                             "persons": len(set(persons[u][p])), "windows": len(c),
                             "top": sorted(((float(x), persons[u][p][i]) for i, x in enumerate(c)),
                                           key=lambda t: -abs(t[0]))[:3]}
                gap = ch["held_out"]["boots"] - ch["train"]["boots"]
                paired[f"{run}|{w}|{u}"] = {
                    p: {k: v for k, v in ch[p].items() if k != "boots"} for p in PARTS}
                paired[f"{run}|{w}|{u}"]["gap"] = {"mean": ch["held_out"]["mean"] - ch["train"]["mean"],
                                                   "interval": [float(np.quantile(gap, 0.025)),
                                                                float(np.quantile(gap, 0.975))]}
    # successive differences of the means over the checkpoints, raw against EMA
    rms = {}
    for u in sets:
        for p in PARTS:
            for w in ("model", "ema"):
                d = []
                for run in REPLICATE:
                    xs = [means[f"{run}:{st}:{w}"][u][p] for st in CKPTS]
                    d += list(np.diff(xs))
                rms[f"{u}|{p}|{w}"] = {"diffs": [float(x) for x in d], "rms": float(math.sqrt(np.mean(np.square(d))))}
    avail = fold_persons(TS, cfg, stores.stage2, list(sets))
    persons_n = {u: {p: {"windows": len(persons[u][p]), "persons": len(set(persons[u][p])),
                         "fold_persons": avail[u][p]} for p in PARTS} for u in persons}
    a = ALARM[2]
    pred = {
        "P1": {"statistic": paired[f"{ALARM[0]}|model|{a}"]["held_out"]["mean"],
               "interval": paired[f"{ALARM[0]}|model|{a}"]["held_out"]["interval"]},
        "P2": {"statistic": paired[f"{ALARM[0]}|model|{a}"]["gap"]["mean"],
               "interval": paired[f"{ALARM[0]}|model|{a}"]["gap"]["interval"]},
        "P3": {"rms_raw": rms[f"{a}|held_out|model"]["rms"], "rms_ema": rms[f"{a}|held_out|ema"]["rms"]}}
    pred["P1"]["holds"] = pred["P1"]["interval"][0] <= 0.0 <= pred["P1"]["interval"][1]
    pred["P2"]["holds"] = pred["P2"]["interval"][0] <= 0.0 <= pred["P2"]["interval"][1]
    pred["P3"]["holds"] = pred["P3"]["rms_ema"] < pred["P3"]["rms_raw"]
    return {"seconds": time.time() - t0, "sets_seconds": t_sets, "timing": timing, "persons": persons_n,
            "knots_check": knots_check, "validation": valid, "draws": draws, "means": means, "paired": paired,
            "rms_successive": rms, "predictions": pred, "units": list(sets)}


# ---------------------------------------------------------------- part 5: the stop checkpoint's lines
def diagnostic_part(tree, folder):
    p = os.path.join(tree, folder, "results.json")
    if not os.path.exists(p):
        return {"folder": folder, "exists": False}
    with open(p, encoding="utf-8") as f:
        R = json.load(f)
    lines = {}
    for k, v in (R.get("lines") or {}).items():
        lines[k] = {x: v.get(x) for x in ("read", "start", "end", "difference", "interval", "margin", "holds",
                                          "unit", "decides", "reason")}
    deciding = [k for k, v in lines.items() if v.get("decides")]
    failed = [k for k in deciding if lines[k].get("holds") is False]
    unread = [k for k in deciding if lines[k].get("holds") is None]
    stopped = (R.get("verdict") or {}).get("verdict") == "STOPPED"
    status = "fails" if failed or not stopped else ("untested" if unread else "holds")
    return {"folder": folder, "exists": True, "verdict": R.get("verdict"), "lines": lines, "deciding": deciding,
            "end": R.get("end"), "P4": {"status": status, "holds": status == "holds", "forgets": failed,
                                       "not_read": unread}}


# ---------------------------------------------------------------- summary
def summary_md(R):
    L, M, P = R["logs"], R["model"], R["plausibility"]
    a_run, a_step, a_unit = ALARM
    out = ["# The unit_nll_rise alarm of run S2c (task S2CA)", "",
           "Dev diagnostic, not a gate read. Every number below is in results.json of this folder, written by "
           "scripts/stage2_unit_nll_alarm.py; the predictions tested are predictions.txt "
           f"(SHA-256 {R['predictions_file']['sha256'][:16]}..., written {R['predictions_file']['written']}).", ""]
    out += ["## 1. The reads", ""]
    fw = L["fixed_windows"]
    ok_fixed = all(len(v["knots"]) <= 1 and len(v["windows"]) <= 1 for v in fw.values())
    out.append(f"- Windows fixed: every unit and part reads one window count and one knot count at every read of the "
               f"three runs: {ok_fixed} (pads held-out {fw[a_unit + '|held_out']['windows']} windows, "
               f"{fw[a_unit + '|held_out']['knots']} knots; train {fw[a_unit + '|train']['knots']} knots).")
    sr = L["start_reads"]
    out.append("- Start reads (same weights in the three runs): largest range over runs "
               + ", ".join(f"{k} {v['range']:.4f}" for k, v in sr.items() if v["range"] is not None) + ".")
    out.append(f"- The alarm recounted with loop.unit_nll_rising at every read matches the logged list: "
               f"{L['recount_matches_log']}.")
    out += ["", "| run | unit | part | reads | first | last | slope per 1000 (se) | residual sd | diff sd/sqrt2 | "
            "lag-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        for u in L["val_units"]:
            for p in PARTS + ("gap",):
                s = L["per_run"][run][u][p]["stats"]
                if not s:
                    continue
                out.append(f"| {run} | {u} | {p} | {s['reads']} | {s['first']:.0f} | {s['last']:.0f} | "
                           f"{s['slope_per_1000']:+.0f} ({s['slope_se_per_1000']:.0f}) | {s['residual_sd']:.0f} | "
                           f"{fnum(s['diff_sd_over_sqrt2'], 0)} | {fnum(s['residual_lag1_autocorrelation'], 2)} |")
    out += ["", "Replicate S2b minus S2c (same plans; reads from 2000):", ""]
    for u in L["val_units"]:
        r = L["replicate"][u]
        out.append(f"- {u}: held-out rms/sqrt2 {r['held_out']['rms_over_sqrt2_from_2000']:.0f}, train "
                   f"{r['train']['rms_over_sqrt2_from_2000']:.0f}, gap {r['gap']['rms_over_sqrt2_from_2000']:.0f}; "
                   f"correlation of the held-out and train differences "
                   f"{fnum(r['held_out_train_correlation_from_2000'], 2)}; differences "
                   + ", ".join(f"{s} {v:+.0f}" for s, v in r["held_out"]["differences"].items()))
    tp = L["triples_pooled"]
    out += ["", f"Three-read windows of the logs (units with a val fold): all reads {tp['all_reads']['windows']}, "
            f"both {tp['all_reads']['both']} ({tp['all_reads']['share']:.4f} "
            f"[{tp['all_reads']['interval'][0]:.4f}, {tp['all_reads']['interval'][1]:.4f}]); from 1000 "
            f"{tp['from_1000']['windows']}, held-out rising {tp['from_1000']['held_out_rising']}, train falling "
            f"{tp['from_1000']['train_falling']}, both {tp['from_1000']['both']} "
            f"({tp['from_1000']['share']:.4f} [{tp['from_1000']['interval'][0]:.4f}, "
            f"{tp['from_1000']['interval'][1]:.4f}]); expected under exchangeable independent reads "
            f"{tp['from_1000']['expected_iid_independent']:.2f}."]
    for run in RUNS:
        h = L["triples"][run]["from_1000"]["hits"]
        out.append(f"- {run}: " + ("; ".join(f"{x['unit']} at {x['steps']}" for x in h) if h else "none"))
    cp = L["compare_s2b_s2c"]
    out += ["", "## 2. S2b and S2c, step by step", "",
            f"- 'log' records compared: {cp['records']} at {cp['steps']} shared steps (per S2c call: "
            + ", ".join(f"{k} {v}" for k, v in cp["records_per_call"].items())
            + f"; S2c's resumed steps counted once per call); plan data equal at every one: {cp['data_equal_all']}; "
            f"learning rate equal: {cp['lr_equal_all']}.",
            f"- First record (step {cp['first_record']['step']}): beta_error {cp['first_record']['beta_b']:.6f} and "
            f"{cp['first_record']['beta_c']:.6f}, loss relative difference {cp['first_record']['loss_rel_diff']:+.2e}; "
            f"first record with health values apart by more than 1e-4: {cp['first_beta_divergence_step']}.",
            "- Median absolute loss relative difference and beta difference by band: "
            + "; ".join(f"{k}: {fnum(v['loss_rel_abs_median'], 4)}, {fnum(v['beta_abs_median'], 4)}"
                        for k, v in cp["bands"].items()) + "."]
    ov = cp["overlap_calls"]
    if ov:
        out.append(f"- S2c's two calls at {ov[0]['step']} to {ov[-1]['step']}: plan data equal "
                   f"{all(x['data_equal'] for x in ov)}; health at {ov[0]['step']} "
                   f"{ov[0]['beta_1']:.6f} and {ov[0]['beta_2']:.6f}; loss relative difference "
                   + ", ".join(f"{x['step']} {x['loss_rel_diff']:+.1e}" for x in ov) + ".")
    out.append(f"- Code snapshots of S2c's calls: " + ", ".join(f"{k} {v[:12]}..." for k, v in
                                                               L["s2c_code_snapshots"].items()) + ".")
    out.append("- Non-finite steps: " + "; ".join(f"{r} {L['nonfinite'][r]}" for r in RUNS) + ".")
    sh = L["shares_last_record"]
    out.append("- pads at the last log record (window share of the phase so far, token share of the interval): "
               + "; ".join(f"{r} {sh[r]['step']}: {fnum((sh[r]['window_share_phase'] or {}).get(a_unit), 4)}, "
                           f"{fnum((sh[r]['token_share_interval'] or {}).get(a_unit), 4)}" for r in RUNS)
               + " (n/a: S2a's records carry no window share).")
    sb = L["speed"]["stage2_s2b"]["1"]["s_per_step"]
    sc1 = L["speed"]["stage2_s2c"]["1"]["s_per_step"]
    ovs = cp["overlap_calls"]
    lost = (ovs[-1]["step"] - ovs[0]["step"] + 50) if ovs else 0
    out.append(f"- Cost: a phase C of {PHASE_C_STEPS} steps at S2b's rate {PHASE_C_STEPS * sb / 3600:.2f} GPU-h (at S2c's "
               f"first call's {PHASE_C_STEPS * sc1 / 3600:.2f}); S2c's crash repeated at least {lost} steps "
               f"({lost * sc1 / 3600:.2f} GPU-h); S2c ran {a_step} steps.")
    out.append("- Seconds per step: " + "; ".join(
        f"{r} " + ", ".join(f"call {k} {v['s_per_step']:.4f} ({v['first_step']} to {v['last_step']})"
                            for k, v in L["speed"][r].items()) for r in RUNS) + ".")
    out += ["", "## 3. The model", ""]
    for u, p in M["params"].items():
        c = M["within_run_check"][u]
        t = M["params_trend"][u]
        out.append(f"- {u}: held-out sd {p['sd_ho']:.0f}, train sd {p['sd_tr']:.0f}, correlation {p['rho']:+.2f}, "
                   f"gap sd {M['sigmas'][u][1]:.0f} (implied {p['sd_gap_implied']:.0f}) (within runs: "
                   f"{c['diff_rms_over_sqrt2_held_out']:.0f}, {c['diff_rms_over_sqrt2_train']:.0f}, "
                   f"{c['diff_correlation']:+.2f} over {c['diffs']} differences); trend per read to 8000 "
                   f"{t['trend_ho']:+.0f} and {t['trend_tr']:+.0f}.")
    out.append("- Exact share per read of a k-read pattern, exchangeable independent reads: "
               + ", ".join(f"{k} {v:.5f}" for k, v in M["exact_per_read"].items()) + ".")
    lr = M["null"]["readings"]["loop_3_reads"]
    lt = M["null_trend"]["readings"]["loop_3_reads"]
    out.append("- The loop's reading per unit, per read and per phase C (no trend; with the trend to 8000): " + "; ".join(
        f"{u} {lr['unit_per_read'][u]:.5f}, {lr['unit_fire_phase'][u]:.4f} ({lt['unit_per_read'][u]:.5f}, "
        f"{lt['unit_fire_phase'][u]:.4f})" for u in M["params"]) + ".")
    out.append("- Expected loop patterns on the logs' three-read windows from 1000: " + ", ".join(
        f"{k} model {v:.2f}" for k, v in M["expected_on_logs"].items())
        + f" (observed {L['triples_pooled']['from_1000']['both']}).")
    out.append("- pads under other correlations (per read, by 8000, per phase C): " + "; ".join(
        f"rho {k}: {v['per_read']:.5f}, {v['fire_by_8000']:.4f}, {v['fire_phase']:.4f}"
        for k, v in M["pads_rho_sensitivity"].items()) + ".")
    keys = [f"{sc}{d}" for sc in SCENARIOS for d in DRIFTS]
    out.append("- Residual estimate (S2b and S2c detrended, from 1000): " + "; ".join(
        f"{u} sd {p['sd_ho']:.0f} and {p['sd_tr']:.0f}, correlation {p['rho']:+.2f} (runs "
        + ", ".join(f"{x:+.2f}" for x in p["rho_runs"]) + ")" for u, p in M["params_resid"].items()) + ".")
    out += ["", f"| reading | stops by 8000 (replicate; residual) | stops a phase C ({M['null']['reads_per_phase']} reads) | residual "
            f"estimate (own floors) | rho 0 | with trend | S2a | S2b | S2c | " + " | ".join(f"reads to fire {k}" for k in keys)
            + " |", "|---|---|---|---|---|---|---|---|---|" + "---|" * len(keys)]

    def lg_cell(x):
        return "none" if x is None else f"{x['step']} ({', '.join(u.split(':')[0] for u in x['units'])})"
    for name, *_ in READINGS:
        n0, nr, nt = M["null"]["readings"][name], M["null_rho0"]["readings"][name], M["null_trend"]["readings"][name]
        ns = M["null_resid"]["readings"][name]
        lg = M["logs"]
        cells = []
        for k in keys:
            o = M["overfit"][k]["readings"][name]
            cells.append(f"{o['reads_to_fire_median']:.0f} (never {o['never_share']:.2f})")
        out.append(f"| {name} | {n0['fire_by_8000']:.4f}; {ns['fire_by_8000']:.4f} | {n0['fire_phase']:.4f} | "
                   f"{ns['fire_phase']:.4f} | "
                   f"{nr['fire_phase']:.4f} | "
                   f"{nt['fire_phase']:.4f} | {lg_cell(lg['stage2_s2a'][name])} | {lg_cell(lg['stage2_s2b'][name])} | "
                   f"{lg_cell(lg['stage2_s2c'][name])} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("The residual estimate with the replicate's floors (the noise larger than the floors the readings "
               "were set at), per phase C: " + "; ".join(
                   f"{name} {M['null_resid_replicate_floors']['readings'][name]['fire_phase']:.4f}"
                   for name, kind, *_ in READINGS if kind != "pattern" or name == "magnitude_2") + ".")
    out.append("")
    out.append(f"Scenarios: A held-out up and train down by drift x sd per read, B held-out up only; drift {DRIFTS} "
               f"on pads from the first read; reads to fire: the median read index (1 is step 1000), "
               f"{M['null']['reads_per_phase'] + 1} when it never fires; never: the share of phases it never fires.")
    cz = M.get("logs_censoring")
    if cz:
        out += ["", "The logs end at each run's stop (a reading needing more reads is censored there): per run, the "
                "last read, and per reading the step it fires at, else the reads in a row at the last read with a "
                "raised unit (open: the next read, absent from the log, could complete it).", ""]
        for run in RUNS:
            c = cz[run]
            out.append(f"- {run} (last read {c['last_read']}): " + "; ".join(
                f"{name} " + (f"fires {v['fires']}" if v["fires"] is not None else
                              f"streak {v['streak_at_last_read']} of {v['consecutive']}"
                              + (" (open: " + ", ".join(u.split(':')[0] for u in v["units_at_last_read"]) + ")"
                                 if v["open"] else ""))
                for name, v in c["readings"].items()) + ".")
    mc = M.get("missed_stop_cost_s2a")
    if mc:
        out.append(f"- Not stopping {mc['run']} at {mc['last_read']} costs at least one dev read "
                   f"({mc['one_read_steps']} steps, {mc['one_read_gpu_h']['s2b_rate']:.2f} GPU-h at S2b's rate, "
                   f"{mc['one_read_gpu_h']['own_rate']:.2f} at its own) and at most the rest of phase C "
                   f"({mc['rest_steps']} steps, {mc['rest_gpu_h']['s2b_rate']:.2f} and "
                   f"{mc['rest_gpu_h']['own_rate']:.2f} GPU-h).")
    oa = M.get("overfit_s2a_like")
    if oa:
        out += ["", f"An S2a-like overfit: {oa['unit']} held-out NLL rising at S2a's slope "
                f"{oa['held_out_slope_per_1000']:+.0f} ({oa['held_out_slope_se']:.0f}) per 1000 steps (train slope "
                f"{oa['train_slope_per_1000']:+.0f} ({oa['train_slope_se']:.0f})), {oa['drift_sd_per_read']:.2f} of its "
                f"replicate held-out sd ({oa['sd_ho']:.0f}) per read from the first read; reads to fire, median and 90th "
                "percentile (1 is step 1000), the share fired by 8000, never:", ""]
        out += ["| reading | B: held-out only | A: train falling too |", "|---|---|---|"]
        for name, *_ in READINGS:
            cells = []
            for sc in ("B", "A"):
                o = oa[sc]["readings"][name]
                cells.append(f"{o['reads_to_fire_median']:.0f}, {o['reads_to_fire_p90']:.0f}; by 8000 "
                             f"{o['fire_by_8000']:.2f}; never {o['never_share']:.2f}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    E = R.get("event")
    if E:
        out += ["", "## 3b. The stop against the noise", "",
                f"- {a_run} {a_unit} at {E['steps']}: held-out " + ", ".join(f"{x:.0f}" for x in E[a_run]["held_out"])
                + ", train " + ", ".join(f"{x:.0f}" for x in E[a_run]["train"]) + ", gap "
                + ", ".join(f"{x:.0f}" for x in E[a_run]["gap"]) + "; S2b at the same steps: held-out "
                + ", ".join(f"{x:.0f}" for x in E["stage2_s2b"]["held_out"]) + ", gap "
                + ", ".join(f"{x:.0f}" for x in E["stage2_s2b"]["gap"]) + ".",
                f"- Held-out rise over the two reads {E['held_out_rise']:+.0f}: "
                f"{E['held_out_rise_in_sd']['replicate']:.2f} (replicate) and {E['held_out_rise_in_sd']['within_runs']:.2f}"
                f" (within runs) two-read sd ({E['two_read_sd']['held_out_replicate']:.0f} and "
                f"{E['two_read_sd']['held_out_within_runs']:.0f}); gap rise {E['gap_rise']:+.0f}: "
                f"{E['gap_rise_in_sd']:.2f} two-read gap sd ({E['two_read_sd']['gap_replicate']:.0f})"
                + (f"; the recompute's raw gap change {E['recompute_gap_change']:+.0f}: "
                   f"{E['recompute_gap_change_in_sd']:.2f} of it" if "recompute_gap_change" in E else "") + "."]
    out += ["", "## 4. Plausibility", "",
            f"- Reads above {P['line']}: " + "; ".join(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]:.4f}"
                                                       for x in P["above_line"]) + "; the largest other read from "
            f"1000: {fnum(P['max_below_line_from_1000'], 4)}.",
            "- pads outside-limits share, S2c: " + ", ".join(
                f"{s} {fnum(v['held_out'], 4)}/{fnum(v['train'], 4)}" for s, v in P["pads"]["stage2_s2c"].items())
            + " (held-out/train)."]
    if R.get("recompute"):
        C = R["recompute"]
        rf = R.get("recompute_from")
        out += ["", "## 5. Recompute on the CPU", ""]
        if rf:
            out.append(f"- Taken from {rf['path']} (SHA-256 {rf['sha256'][:16]}..., run at {rf['run_at']}, code "
                       f"{str(rf['code_sha256'])[:16]}..., argv {rf['argv']}).")
        out += [f"- {C['seconds']:.0f} s; windows and persons: " + "; ".join(
                    f"{u} " + ", ".join(f"{p} {v['windows']} windows of {v['persons']} persons (of "
                                        f"{v.get('fold_persons')} in the fold)" for p, v in d.items())
                    for u, d in C["persons"].items()) + ".",
                "- Knots, recomputed against logged: " + "; ".join(
                    f"{k} {v['recomputed']} {v['logged']}" for k, v in C["knots_check"].items()) + "."]
        vd = [v for v in C["validation"] if v["unit"] == a_unit]
        out.append("- pads, raw, recomputed minus logged: " + "; ".join(
            f"{v['run'][-3:]} {v['step']} {v['part']} {fnum(v['difference'], 0)}" for v in vd) + ".")
        out.append("- pads, raw, the 32-window mean over other draws at fixed weights (sd): " + "; ".join(
            f"{k} {v['sd']:.0f}" for k, v in C["draws"].items()) + ".")
        out += ["", "| weights | unit | part | 2000 | 4000 | 6000 | 8000 | change 6000 to 8000 [95 percent] |",
                "|---|---|---|---|---|---|---|---|"]
        for run in REPLICATE:
            for w in ("model", "ema"):
                for u in C["units"]:
                    for p in PARTS:
                        ms = [C["means"][f"{run}:{st}:{w}"][u][p] for st in CKPTS]
                        ch = C["paired"][f"{run}|{w}|{u}"][p]
                        out.append(f"| {run} {w} | {u} | {p} | " + " | ".join(f"{m:.0f}" for m in ms)
                                   + f" | {ch['mean']:+.0f} [{ch['interval'][0]:+.0f}, {ch['interval'][1]:+.0f}] |")
                    g = C["paired"][f"{run}|{w}|{u}"]["gap"]
                    out.append(f"| {run} {w} | {u} | gap | | | | | {g['mean']:+.0f} [{g['interval'][0]:+.0f}, "
                               f"{g['interval'][1]:+.0f}] |")
        out.append("")
        out.append("- Root mean square of the successive differences over the checkpoints (S2b and S2c, 6 "
                   "differences), raw and EMA: " + "; ".join(
                       f"{u} {p} {C['rms_successive'][f'{u}|{p}|model']['rms']:.0f} and "
                       f"{C['rms_successive'][f'{u}|{p}|ema']['rms']:.0f}" for u in C["units"] for p in PARTS) + ".")
        pr = C["predictions"]
        out.append(f"- P1 {'holds' if pr['P1']['holds'] else 'fails'}: {pr['P1']['statistic']:+.0f} "
                   f"[{pr['P1']['interval'][0]:+.0f}, {pr['P1']['interval'][1]:+.0f}]. P2 "
                   f"{'holds' if pr['P2']['holds'] else 'fails'}: {pr['P2']['statistic']:+.0f} "
                   f"[{pr['P2']['interval'][0]:+.0f}, {pr['P2']['interval'][1]:+.0f}]. P3 "
                   f"{'holds' if pr['P3']['holds'] else 'fails'}: raw {pr['P3']['rms_raw']:.0f}, EMA "
                   f"{pr['P3']['rms_ema']:.0f}.")
    D = R.get("diagnostic")
    if D:
        out += ["", "## 6. The lines at S2c's stop checkpoint (the read of the folder named below)", ""]
        if not D["exists"]:
            out.append(f"- {D['folder']}: not written yet.")
        else:
            out.append(f"- {D['folder']}: verdict {D['verdict'].get('verdict')} (case {D['verdict'].get('case')}).")
            B = R.get("diagnostic_s2b") or {}
            out += ["", "| line | start | S2c end | end minus start | interval | margin | holds | S2b at 8550, end minus "
                    "start |", "|---|---|---|---|---|---|---|---|"]
            for k, v in D["lines"].items():
                iv = v.get("interval")
                iv = (iv or {}).get("interval") if isinstance(iv, dict) else iv
                b = (B.get("lines") or {}).get(k) or {}
                out.append(f"| {k} | {fnum(v.get('start'), 4)} | {fnum(v.get('end'), 4)} | "
                           f"{fnum(v.get('difference'), 4)} | "
                           f"{'n/a' if not iv else f'[{iv[0]:+.4f}, {iv[1]:+.4f}]'} | {fnum(v.get('margin'), 4)} | "
                           f"{v.get('holds')} | {fnum(b.get('difference'), 4)} |")
            p4 = D["P4"]
            out.append(f"- P4 {p4['status']}" + (f" (deciding lines not read: {', '.join(p4['not_read'])})"
                                                 if p4["not_read"] else "")
                       + (f" (FORGETS: {', '.join(p4['forgets'])})" if p4["forgets"] else "") + ".")
    return "\n".join(out) + "\n"


def predictions_file(out_dir):
    p = os.path.join(out_dir, PREDICTIONS)
    s = os.path.join(out_dir, PREDICTIONS.replace(".txt", ".sha256"))
    if not os.path.exists(p) or not os.path.exists(s):
        return {"exists": False}
    with open(s, encoding="utf-8") as f:
        parts = f.read().split()
    sha = sha256_file(p)
    return {"exists": True, "sha256": sha, "recorded": parts[0], "matches": sha == parts[0],
            "written": parts[2] if len(parts) > 2 else None}


def write(R, out_dir):
    from mdfm.gates import common
    text = summary_md(R)
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(os.path.join(out_dir, "results.json"), json.dumps(R, indent=1, allow_nan=False, default=str) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    try:
        spec = importlib.util.spec_from_file_location("_s2ca_stamp", os.path.join(HERE, "moveatlas_stamp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stamp = {"head": mod.stamp().get("head")}
    except Exception as e:  # noqa: BLE001  (reported; the measurement reads no MoveAtlas file)
        stamp = {"error": f"{type(e).__name__}: {e}"}
    prov = {"script": "scripts/stage2_unit_nll_alarm.py", "task": TASK, "label": "dev diagnostic, not a gate read",
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "moveatlas_stamp": stamp,
            "code_sha256": common.code_sha256([os.path.abspath(__file__)]), "argv": R.get("argv"),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    return text


def set_sim_n(n):
    """The number of simulated phases of part 2 (--n-sim)."""
    global SIM_N
    SIM_N = int(n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recompute", action="store_true", help="part 4: per-window NLL on the CPU")
    p.add_argument("--diagnostic", default=None, help="part 5: the read folder of S2c's stop checkpoint")
    p.add_argument("--recompute-from", default=None,
                   help="part 4 taken from an earlier results.json of this script (its 'recompute' entry and the code "
                        "SHA-256 of its _provenance.json, recorded) instead of computed again")
    p.add_argument("--n-sim", type=int, default=SIM_N)
    p.add_argument("--out", default=OUT)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    out_dir = os.path.join(TREE, a.out)
    R = {"task": TASK, "label": "dev diagnostic, not a gate read", "argv": sys.argv[1:] if argv is None else argv,
         "alarm": {"run": ALARM[0], "step": ALARM[1], "unit": ALARM[2]},
         "predictions_file": predictions_file(os.path.join(TREE, OUT))}
    if a.recompute and not R["predictions_file"].get("matches"):
        raise SystemExit("predictions.txt does not match predictions.sha256: the tests run only after the hash")
    R["logs"] = logs_part(TREE)
    set_sim_n(a.n_sim)
    R["model"] = model_part(R["logs"])
    reads = {run: dev_reads(read_jsonl(os.path.join(TREE, RUNS_DIR, run, "log.jsonl"))) for run in RUNS}
    R["plausibility"] = plausibility_part(R["logs"], reads)
    if a.recompute and a.recompute_from:
        raise SystemExit("--recompute and --recompute-from exclude each other")
    if a.recompute:
        R["recompute"] = recompute_part(TREE, R["logs"])
    elif a.recompute_from:
        src = os.path.join(TREE, a.recompute_from)
        with open(src, encoding="utf-8") as f:
            prev = json.load(f)
        if "recompute" not in prev:
            raise SystemExit(f"{a.recompute_from} holds no recompute entry")
        prov_p = os.path.join(os.path.dirname(src), "_provenance.json")
        prov = json.load(open(prov_p, encoding="utf-8")) if os.path.exists(prov_p) else {}
        R["recompute"] = prev["recompute"]
        R["recompute_from"] = {"path": rel(src), "sha256": sha256_file(src), "run_at": prov.get("run_at"),
                               "code_sha256": prov.get("code_sha256"), "argv": prov.get("argv")}
    R["event"] = event_part(R["logs"], R["model"], R.get("recompute"))
    if a.diagnostic:
        R["diagnostic"] = diagnostic_part(TREE, a.diagnostic)
        R["diagnostic_s2b"] = diagnostic_part(TREE, S2B_DIAGNOSTIC)
    text = summary_md(R)
    print(text)
    if a.write:
        write(R, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
