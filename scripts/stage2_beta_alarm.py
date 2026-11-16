"""S2BA: the beta_error alarm that stopped run S2b at phase C step 8550, measured (task S2BA, 2026-10-13).

The alarm (mdfm/train/loop.py ALARMS, "beta_error": ("log", "above", 2.0, 1000)) reads one number per 'log' record:
loop.health() on the first micro-step of the record's interval (log_every optimizer steps), the raw (not EMA)
weights in eval mode, the mean over the micro-step's windows with a body truth (synthetic AMASS windows,
losses.truth_mask) and over the 16 shape dimensions of |mu_b - beta_true|, mu_b the window's own posterior mean (not
pooled per subject). This script measures how noisy that number is, where its line comes from in the logs, what the
record that stopped S2b held, and what other readings of the same line would have done on every log that carries it.

Parts (each number of summary.md comes from results.json of this folder):
1. Logs (always): every cache/runs/*/log.jsonl; per run and phase the records that carry beta_error, their spread,
   the records above the line from the alarm's first step (1,000) on, the lag-1 autocorrelation, the noise of
   successive differences, the linear trend; pooled per phase kind with an exact 95 percent interval on the share of
   records above 2.0; the readings of READINGS on every log (fired or not, first step, the largest value of the
   reading's statistic); their false-alarm projection on a phase C of 35,754 steps (i.i.d. records: exact for k
   consecutive records, by resampling the pooled phase C records for the others).
2. Composition (--replay): the phase C sampler of S2a and S2b replayed plan by plan from its fresh state (plans only,
   no window read: data/sampler.MixtureSampler.plan), checked at every checkpoint of the run against the checkpoint's
   sampler state (generator, step, credits, window tally); for each 'log' record the plan of its health micro-step
   (index 2 x (step - log_every), micro_per_step 2): windows, truth windows, distinct bodies (take, first knot),
   render pairs, distinct AMASS persons; beside it the logged value. Also every plan's truth windows (their
   distribution), and with --stage1-plans N the plans of v1's phase A and phase B samplers (configs of
   cache/runs/phaseA_root_full/ckpt_B_032044.pt) for the same counts.
3. Recompute (--replay with --panel): the truth windows of chosen health plans rendered on the CPU exactly as
   sampler.materialize renders them (the plan's synthetic items, body_store.windows, the renderer with
   numpy default_rng([plan seed, 1])), collated apart (the encoder attends within a window: model/masks.py, so a
   window's mu_b does not depend on the others) and encoded under chosen weights: the start (v1's phase B end,
   migrated to the stage-2 names as Trainer.init_stage2 migrates it, raw weights) and the raw weights of the S2a and
   S2b checkpoints. Validation: a record whose health weights are a checkpoint's (the record 50 steps after it, the
   start for the record at step 50) is recomputed against its logged value. Every S2b health plan is also encoded
   under fixed weights (the start and S2b ckpt_C_008000): the spread of the record at fixed weights, against the
   logged spread.

Reads (read only): log.jsonl and checkpoints of the run folders (torch.load weights_only, mmap), configs/stage2_s2a.yaml
and configs/stage2_s2b.yaml, the stores' tables and, for --panel, the synthetic AMASS bodies of the training split
(body store) that the plans name. No lockbox, EVAL, CODY or care_pd record, no real window, no GPU.

Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python scripts/stage2_beta_alarm.py [--replay] [--panel]
      [--stage1-plans 500] [--write]
Writes reports/bench/stage2_beta_alarm/results.json, summary.md and _provenance.json with --write.
"""
import argparse
import glob
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

TASK = "S2BA"
OUT = os.path.join("reports", "bench", "stage2_beta_alarm")
RUNS_DIR = os.path.join("cache", "runs")
START = os.path.join("cache", "runs", "phaseA_root_full", "ckpt_B_032044.pt")
LINE, FIRST = 2.0, 1000                    # loop.ALARMS["beta_error"]: above 2.0 from step 1,000 of the phase
PHASE_C_STEPS = 35754                      # stage2.steps of configs/stage2_s2a.yaml and stage2_s2b.yaml
LOG_EVERY = 50
STAGE2 = {"stage2_s2a": "configs/stage2_s2a.yaml", "stage2_s2b": "configs/stage2_s2b.yaml"}
ALARM_RECORD = ("stage2_s2b", 8550)
# readings of the line on a series of records: (name, kind, parameter); 'single' is the loop's today
READINGS = (("single", "consecutive", 1), ("consecutive_2", "consecutive", 2), ("consecutive_3", "consecutive", 3),
            ("mean_3", "mean", 3), ("mean_5", "mean", 5), ("median_5", "median", 5))
# phase kinds pooled for the rates: the reference runs of the program in that phase
POOLS = {"A": ("phaseA_fix_full", "phaseA_root_1", "phaseA_root_full", "pilot_v1_s2"),
         "B": ("phaseA_root_full", "pilot_v1_s2", "pilot_f2l2", "pilot_f2l4", "pilot_e1_jepa"),
         "C": ("stage2_s2a", "stage2_s2b")}
SIM_N, SIM_SEED = 20000, 20261013          # resampled phase C series for the false-alarm projection
PANEL = {"stage2_s2b": [50, 2050, 4050, 6050, 8050] + list(range(8100, 8551, 50)),
         "stage2_s2a": [50, 2050, 4050] + list(range(5550, 6001, 50))}
PANEL_CKPTS = {"stage2_s2a": [2000, 4000, 6000], "stage2_s2b": [2000, 4000, 6000, 8000, 8550]}
FIXED = ("start", "stage2_s2b:8000")       # weights under which every S2b health plan is encoded
BOOT_N, BOOT_SEED = 10000, 20261013


def rel(p):
    return os.path.relpath(os.path.abspath(p), TREE).replace("\\", "/")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def fnum(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- part 1: the logs
def series(recs):
    """{phase: [(step, value)]} of the 'log' records' health.beta_error, the last record of each step (a resumed call
    logs a step again), in step order; the None values counted apart ({phase: n})."""
    by, none = {}, {}
    for r in recs:
        if r.get("event") != "log":
            continue
        ph, st = r.get("phase"), int(r.get("step", -1))
        b = (r.get("health") or {}).get("beta_error")
        if b is None:
            if "health" in r:
                none[ph] = none.get(ph, 0) + 1
            continue
        by.setdefault(ph, {})[st] = float(b)
    return {ph: sorted(d.items()) for ph, d in by.items()}, none


def reading_values(vals, kind, k):
    """The statistic of a reading at each record of a series of values (records before the alarm's first step left
    out by the caller): 'consecutive' the smallest of the last k values (the reading fires when it is above the
    line: k consecutive records above it), 'mean' the mean of the last k, 'median' their median; None until k
    records exist."""
    out = []
    for i in range(len(vals)):
        if i + 1 < k:
            out.append(None)
            continue
        w = vals[i + 1 - k:i + 1]
        out.append(float(min(w)) if kind == "consecutive" else float(np.mean(w)) if kind == "mean"
                   else float(np.median(w)))
    return out


def readings(steps, vals, line=LINE, first=FIRST):
    """{reading: {fired, first_step, max_statistic, margin}} of READINGS on one phase's series (records from `first`
    on)."""
    keep = [(s, v) for s, v in zip(steps, vals) if s >= first]
    st = [s for s, _ in keep]
    vs = [v for _, v in keep]
    out = {}
    for name, kind, k in READINGS:
        stat = reading_values(vs, kind, k)
        fired = [s for s, x in zip(st, stat) if x is not None and x > line]
        m = max((x for x in stat if x is not None), default=None)
        out[name] = {"fired": bool(fired), "first_step": fired[0] if fired else None, "n_fired": len(fired),
                     "max_statistic": m, "margin": None if m is None else line - m}
    return out


def ols(x, y):
    """(slope, its standard error) of y on x by least squares."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None, None
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    res = y - y.mean() - b * xc
    se = float(math.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))
    return b, se


def phase_stats(steps, vals, line=LINE, first=FIRST):
    s, v = np.asarray(steps), np.asarray(vals, float)
    m = s >= first
    out = {"records": int(len(v)), "steps": [int(s.min()), int(s.max())] if len(s) else None,
           "log_every": int(np.median(np.diff(s))) if len(s) > 1 else None, "from_first": int(m.sum())}
    if m.sum() >= 2:
        w = v[m]
        slope, se = ols(s[m], w)
        d = np.diff(w)
        ac = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if len(w) > 2 else None
        out.update(mean=float(w.mean()), sd=float(w.std(ddof=1)), median=float(np.median(w)),
                   p95=float(np.quantile(w, 0.95)), p99=float(np.quantile(w, 0.99)), min=float(w.min()),
                   max=float(w.max()), max_step=int(s[m][int(np.argmax(w))]), above=int((w > line).sum()),
                   lag1_autocorrelation=ac, diff_sd_over_sqrt2=float(d.std(ddof=1) / math.sqrt(2)),
                   slope_per_1000_steps=None if slope is None else 1000.0 * slope,
                   slope_se_per_1000_steps=None if se is None else 1000.0 * se,
                   z_of_max=float((w.max() - w.mean()) / w.std(ddof=1)))
    return out


def binom_cdf(k, n, p):
    """P(X <= k), X ~ Binomial(n, p)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return float(sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp
                              + (n - i) * lq) for i in range(0, k + 1)))


def clopper_pearson(k, n, level=0.95):
    """Exact interval of a binomial share k / n, by bisection on the binomial distribution."""
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


def p_no_run(n, k, p):
    """P(no k consecutive successes in n i.i.d. trials of success probability p), exact (dynamic programme)."""
    state = np.zeros(k)
    state[0] = 1.0
    for _ in range(int(n)):
        new = np.zeros(k)
        new[0] = state.sum() * (1 - p)
        new[1:] = state[:-1] * p
        state = new
    return float(state.sum())


def records_in_phase(steps=PHASE_C_STEPS, every=LOG_EVERY, first=FIRST):
    """The 'log' records of a phase of `steps` steps from the alarm's first step on."""
    return len([s for s in range(every, steps + 1, every) if s >= first])


def projection(pool, line=LINE, steps=PHASE_C_STEPS, n_sim=SIM_N, seed=SIM_SEED):
    """False alarms of each reading over one phase C of `steps` steps of i.i.d. records: per record the share p of
    the pooled phase C records above the line and its exact upper bound; P(the reading fires at least once) exact
    for k consecutive records at both, and by n_sim resampled series of the pooled records for every reading."""
    n = records_in_phase(steps)
    k_above = int((np.asarray(pool) > line).sum())
    p = k_above / len(pool)
    ci = clopper_pearson(k_above, len(pool))
    out = {"records_per_phase": n, "pool": len(pool), "above": k_above, "p": p, "p_interval": ci, "readings": {}}
    from numpy.lib.stride_tricks import sliding_window_view
    rng = np.random.default_rng(seed)
    pool = np.asarray(pool, float)
    fires = {name: 0 for name, _, _ in READINGS}
    for _ in range(int(n_sim)):
        s = pool[rng.integers(len(pool), size=n)]
        for name, kind, k in READINGS:
            if kind == "consecutive":
                hit = np.convolve((s > line).astype(int), np.ones(k, int), "valid").max() >= k
            elif kind == "mean":
                hit = (np.convolve(s, np.ones(k) / k, "valid") > line).any()
            else:
                hit = (np.median(sliding_window_view(s, k), axis=1) > line).any()
            fires[name] += int(bool(hit))
    for name, kind, k in READINGS:
        r = {"resampled_fire_share": fires[name] / n_sim}
        if kind == "consecutive":
            r.update(exact_fire_at_p=1 - p_no_run(n, k, p), exact_fire_at_p_high=1 - p_no_run(n, k, ci[1]))
        out["readings"][name] = r
    return out


def logs_part(tree=TREE):
    runs, empty = {}, []
    for path in sorted(glob.glob(os.path.join(tree, RUNS_DIR, "*", "log.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("._"):
            continue
        recs = read_jsonl(path)
        by, none = series(recs)
        if not by:
            empty.append(name)
            continue
        alarms = [{"step": r.get("step"), "phase": r.get("phase"), "alarms": r.get("alarms")}
                  for r in recs if r.get("event") == "alarm"]
        ent = {"log": rel(path), "sha256": sha256_file(path), "none_values": none, "alarm_events": alarms,
               "phases": {}}
        for ph, sv in by.items():
            st, vs = [s for s, _ in sv], [v for _, v in sv]
            ent["phases"][ph] = dict(phase_stats(st, vs), readings=readings(st, vs))
        runs[name] = ent
    pools = {}
    for kind, names in POOLS.items():
        vals = []
        for n in names:
            sv = None
            if n in runs:
                path = os.path.join(tree, RUNS_DIR, n, "log.jsonl")
                sv = series(read_jsonl(path))[0].get(kind)
            vals += [v for s, v in (sv or []) if s >= FIRST]
        a = int((np.asarray(vals) > LINE).sum()) if vals else 0
        pools[kind] = {"runs": list(names), "records": len(vals), "above": a,
                       "share": a / len(vals) if vals else None,
                       "interval": clopper_pearson(a, len(vals)) if vals else None,
                       "mean": float(np.mean(vals)) if vals else None,
                       "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    pool_c = []
    for n in POOLS["C"]:
        pool_c += [v for s, v in series(read_jsonl(os.path.join(tree, RUNS_DIR, n, "log.jsonl")))[0].get("C", [])
                   if s >= FIRST]
    s2b = dict(series(read_jsonl(os.path.join(tree, RUNS_DIR, "stage2_s2b", "log.jsonl")))[0]["C"])
    s2a = dict(series(read_jsonl(os.path.join(tree, RUNS_DIR, "stage2_s2a", "log.jsonl")))[0]["C"])
    common = sorted(s for s in set(s2a) & set(s2b) if s >= FIRST)
    compare = {"steps": [common[0], common[-1]] if common else None, "records": len(common),
               "s2a_mean": float(np.mean([s2a[s] for s in common])), "s2b_mean": float(np.mean([s2b[s] for s in common])),
               "s2a_sd": float(np.std([s2a[s] for s in common], ddof=1)),
               "s2b_sd": float(np.std([s2b[s] for s in common], ddof=1))}
    d = np.array([s2b[s] for s in common]) - np.mean([s2b[s] for s in common])
    e = np.array([s2a[s] for s in common]) - np.mean([s2a[s] for s in common])
    se = math.sqrt(d.var(ddof=1) / len(d) + e.var(ddof=1) / len(e))
    compare.update(difference=compare["s2b_mean"] - compare["s2a_mean"], difference_se=se)
    by_k = {}
    for name, sv in (("stage2_s2a", s2a), ("stage2_s2b", s2b)):
        rows = []
        for lo in range(1000, max(sv) + 1, 1000):
            w = [v for s, v in sv.items() if lo <= s < lo + 1000]
            if w:
                rows.append({"from": lo, "to": lo + 950, "records": len(w), "mean": float(np.mean(w)),
                             "max": float(np.max(w))})
        by_k[name] = rows
    early = {n: float(np.mean([v for s, v in sv.items() if s < FIRST])) for n, sv in (("stage2_s2a", s2a),
                                                                                     ("stage2_s2b", s2b))}
    alarm_rec = {"value": s2b.get(ALARM_RECORD[1]),
                 "rank_in_run": int(sum(1 for v in s2b.values() if v >= s2b.get(ALARM_RECORD[1], 0))),
                 "previous": [{"step": s, "value": s2b[s]} for s in sorted(s2b) if ALARM_RECORD[1] - 300 <= s
                              < ALARM_RECORD[1]]}
    return {"runs": runs, "no_beta_error": empty, "pools": pools, "projection_C": projection(pool_c),
            "s2a_vs_s2b": compare, "by_thousand": by_k, "before_first_mean": early, "alarm_record": alarm_rec,
            "records_per_phase_C": records_in_phase()}


# ---------------------------------------------------------------- part 2: the plans
def _stores(cfg):
    from mdfm.data import sampler as SM
    from mdfm.data import window_store as WS
    from mdfm.train import config as CF
    from mdfm.train import names as NM
    return SM.Stage2Stores(WS.WindowStore(WS.store_root(), tier="RELEASE", names=NM.STAGE1),
                           WS.WindowStore(WS.store_root(kind=WS.TIER_KIND[cfg.tier]), tier=cfg.tier,
                                          names=CF.name_tables(cfg)))


def composition(items):
    from mdfm.data import sampler as SM
    syn = [it for it in items if it.source == SM.SYNTHETIC]
    return {"windows": len(items), "truth": len(syn), "bodies": len({(it.take, it.start) for it in syn}),
            "pairs": sum(1 for it in syn if it.second), "persons": len({it.person for it in syn}),
            "person_ids": sorted({it.person for it in syn})}


def ckpt_steps(run_dir):
    out = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_C_*.pt")):
        b = os.path.basename(p)
        if b.startswith("._"):
            continue
        out.append(int(b[len("ckpt_C_"):-3]))
    return sorted(out)


def replay(tree, run, render_steps=(), max_step=None):
    """The phase C plans of `run` from its fresh sampler (module docstring, part 2), checked at every checkpoint and
    then set to the checkpoint's sampler state (sampler.materialize can drop a window that the plan holds, which moves
    the window tally of a capped run; the next segment then starts exact). A record's plan is verified when the check
    that ends its segment holds. Returns ({checks, records (each with 'verified'), plans_truth}, the plans of
    render_steps {record step: plan}, {sampler, bodies, cfg, sizes (each verified plan's cluster sizes)})."""
    import torch
    from mdfm.data import body_store as BS
    from mdfm.data import sampler as SM
    from mdfm.factory import layouts as LY
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    t0 = time.time()
    cfg = CF.load(os.path.join(tree, STAGE2[run]))
    run_dir = os.path.join(tree, RUNS_DIR, run)
    recs = read_jsonl(os.path.join(run_dir, "log.jsonl"))
    logged = dict(series(recs)[0].get("C", []))
    every = int(cfg.schedule.log_every)
    micro = int(cfg.schedule.micro_per_step)
    rec_steps = sorted(int(r["step"]) for r in recs if r.get("event") == "log" and r.get("phase") == "C"
                       and (max_step is None or int(r["step"]) <= max_step))
    health_plan = {micro * (s - every): s for s in rec_steps}
    cks = [c for c in ckpt_steps(run_dir) if max_step is None or c <= max_step]
    bs = BS.BodyStore(split="train")
    lays = LY.load_stage2_layouts()
    sm = SM.MixtureSampler(bs, _stores(cfg), None, "C", LP.phase_seed(cfg, "C"), cfg, layouts=lays,
                           redraws=int(cfg.stage2.redraws), tier=cfg.tier)
    want_render = {micro * (s - every): s for s in render_steps}
    last = max(max(health_plan), micro * max(cks) - 1)
    checks, records, rendered, seg = [], {}, {}, []
    rows = []                                # (plan index, truth, persons, cluster sizes) of every plan
    k = 0
    while k <= last:
        p = sm.plan()
        c = composition(p.items)
        cnt = {}
        for it in p.items:
            if it.source == SM.SYNTHETIC:
                cnt[it.person] = cnt.get(it.person, 0) + 1
        rows.append((k, c["truth"], c["persons"], tuple(sorted(cnt.values()))))
        if k in health_plan:
            s = health_plan[k]
            records[s] = dict(c, plan=k, seed=int(p.seed), logged=logged.get(s))
        if k in want_render:
            rendered[want_render[k]] = p
        k += 1
        if k % micro == 0 and (k // micro) in cks:
            st = int(k // micro)
            ck = torch.load(os.path.join(run_dir, f"ckpt_C_{st:06d}.pt"), map_location="cpu", weights_only=True,
                            mmap=True)["sampler"]
            ok = {"step": int(sm.step) == int(ck["step"]), "rng": sm.rng.bit_generator.state == ck["rng"],
                  "credit": sm.credit == {str(a): int(b) for a, b in ck["credit"].items()}}
            if sm.wtally is not None:
                ok["window_tally"] = sm.wtally == {str(a): int(b) for a, b in ck["window_tally"].items()}
                ok["tally_difference"] = {a: int(sm.wtally[a]) - int(b) for a, b in ck["window_tally"].items()
                                          if int(sm.wtally.get(a, 0)) != int(b)}
            good = all(v for a, v in ok.items() if a != "tally_difference")
            checks.append({"checkpoint": st, "plans": int(sm.step), **ok, "all": good})
            seg.append((micro * st, good))
            sm.load_state_dict(ck)               # the next segment starts from the training's own state
    def verified(idx):
        for end, good in seg:
            if idx < end:
                return good
        return False
    for s, r in records.items():
        r["verified"] = verified(r["plan"])
    for s in list(rendered):
        if not verified(micro * (s - every)):
            del rendered[s]
    keep = [r for r in rows if verified(r[0])]
    truth_counts = [r[1] for r in keep]
    persons_counts = [r[2] for r in keep]
    sizes = [r[3] for r in keep]
    tc, pc = np.asarray(truth_counts), np.asarray(persons_counts)
    return {"config": STAGE2[run], "config_sha256": CF.sha256(cfg), "plans": int(len(rows)),
            "plans_verified": int(len(tc)), "checks": checks, "records": records, "redrawn": int(sm.redrawn),
            "plans_truth": {"mean": float(tc.mean()), "median": float(np.median(tc)),
                            "share_le": {str(n): float((tc <= n).mean()) for n in (0, 2, 4, 6)},
                            "persons_share_le": {str(n): float((pc <= n).mean()) for n in (1, 2, 3)}},
            "seconds": time.time() - t0}, rendered, {"sampler": sm, "bodies": bs, "cfg": cfg, "sizes": sizes}


def stage1_plans(tree, n):
    """Truth windows per plan of v1's phase A and phase B samplers (the config in the start checkpoint), n plans
    each from the fresh samplers."""
    import torch
    from mdfm.data import body_store as BS
    from mdfm.data import sampler as SM
    from mdfm.data import window_store as WS
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    ck = torch.load(os.path.join(tree, START), map_location="cpu", weights_only=True, mmap=True)
    rc = CF.from_json(ck["config"])
    del ck
    bs = BS.BodyStore(split="train")
    out = {}
    for ph in ("A", "B"):
        ws = WS.WindowStore() if ph == "B" else None
        sm = SM.MixtureSampler(bs, ws, None, ph, LP.phase_seed(rc, ph), rc.mixture, tier=rc.tier)
        tr, pe, win = [], [], []
        for _ in range(int(n)):
            c = composition(sm.plan().items)
            tr.append(c["truth"])
            pe.append(c["persons"])
            win.append(c["windows"])
        tr, pe = np.asarray(tr), np.asarray(pe)
        out[ph] = {"plans": int(n), "windows_mean": float(np.mean(win)), "truth_mean": float(tr.mean()),
                   "truth_median": float(np.median(tr)), "persons_mean": float(pe.mean()),
                   "truth_share_le": {str(k): float((tr <= k).mean()) for k in (2, 4, 6)},
                   "persons_share_le": {str(k): float((pe <= k).mean()) for k in (1, 2, 3)}}
    return out


def association(records):
    """The logged value against the composition of the health plans (records from the alarm's first step on):
    Spearman correlations and the mean value per persons bin."""
    rows = [r for s, r in records.items() if s >= FIRST and r.get("logged") is not None and r.get("verified", True)]
    if len(rows) < 3:
        return None

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    v = np.array([r["logged"] for r in rows])
    out = {"records": len(rows), "spearman_truth": spearman([r["truth"] for r in rows], v),
           "spearman_persons": spearman([r["persons"] for r in rows], v), "by_persons": []}
    for lo, hi in ((0, 2), (3, 4), (5, 6), (7, 99)):
        w = [r["logged"] for r in rows if lo <= r["persons"] <= hi]
        if w:
            out["by_persons"].append({"persons": f"{lo} to {hi}" if hi < 99 else f"{lo} or more", "records": len(w),
                                      "mean": float(np.mean(w)), "sd": float(np.std(w, ddof=1)) if len(w) > 1 else None,
                                      "max": float(np.max(w))})
    tr = np.array([r["truth"] for r in rows])
    pe = np.array([r["persons"] for r in rows])
    out["truth_share_le_4"] = float((tr <= 4).mean())
    out["persons_share_le_2"] = float((pe <= 2).mean())
    return out


# ---------------------------------------------------------------- part 3: recompute on the CPU
def render_truth(plan, bs, synth, cfg, kw):
    """The collated truth windows of a plan (module docstring, part 3) and their plan items."""
    from mdfm.data import collate as CO
    from mdfm.data import sampler as SM
    its = [it for it in plan.items if it.source == SM.SYNTHETIC]
    if not its:
        return None, []
    bodies = bs.windows([it.take for it in its], [it.start for it in its], float(cfg.mixture.win_s))
    wins = synth.render(bodies, [it.layout for it in its], np.random.default_rng([plan.seed, 1]))
    return CO.collate(wins, cfg=CO.collate_config(cfg), gen=plan.seed, **kw), its


def start_model(tree, cfg, replay_store):
    """The start (rule.txt's Start) migrated to the run's names as Trainer.init_stage2 migrates it: raw weights, CPU,
    eval mode; with the migration plan's SHA-256."""
    import torch
    from mdfm.kernels.nuisance import MountPriors
    from mdfm.model import mfm as M
    from mdfm.train import config as CF
    from mdfm.train import migrate as MG
    ck = torch.load(os.path.join(tree, START), map_location="cpu", weights_only=True, mmap=True)
    old, new = MG.names_of(ck["config"]), CF.name_tables(cfg)
    plan, _ = MG.plan_for(old, new, cfg.stage2.task_rows, store=replay_store)
    priors, _ = MG.extend_priors(MountPriors.load(), new.mount_families)
    sd = MG.apply(ck["model"], plan, priors)
    model = M.MFM(CF.model_config(cfg))
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, plan.sha256()


def errors(model, mb):
    """Per-window mean |mu_b - beta_true| over the shape dimensions, and per window the mean |beta_true|."""
    import torch
    from mdfm.train import loop as LP
    from mdfm.train import losses as LS
    with torch.no_grad():
        post = LP.encode(model, mb.batch)
        has = LS.truth_mask(mb.batch, mb.targets, "cpu")
        bt = mb.batch.truth.beta.float()
        e = (post.mu_b.float() - bt).abs().mean(-1)[has]
        size = bt.abs().mean(-1)[has]
    return [float(x) for x in e], [float(x) for x in size]


def cluster_boot(diffs, groups, n=BOOT_N, seed=BOOT_SEED):
    """Percentile 95 percent interval of the mean of per-group means of diffs, groups resampled."""
    g = {}
    for d, k in zip(diffs, groups):
        g.setdefault(k, []).append(d)
    means = np.array([np.mean(v) for v in g.values()])
    rng = np.random.default_rng(seed)
    bs = means[rng.integers(len(means), size=(n, len(means)))].mean(1)
    return float(means.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], int(len(means))


TAIL_DRAWS, TAIL_BOOT = 20, 200


def tail_model(clusters, sizes_all, line=LINE, draws=TAIL_DRAWS, boot=TAIL_BOOT, seed=BOOT_SEED):
    """The share of records above the line under fixed weights, by a cluster model: a record is the window-weighted
    mean of its AMASS persons' errors (the windows of one person in one micro-step, a cluster); the clusters of the
    rendered health plans (size, mean error) form a pool per size; every replayed plan's cluster sizes (sizes_all)
    are filled `draws` times by draws from the pool of their size (the largest smaller size present when a size is
    absent); plans without a truth window give no record, as in the log. Interval: `boot` resamples of the pool's
    clusters within each size (fewer draws). Returns {p, interval, clusters, records}."""
    rng = np.random.default_rng(seed)
    pools = {}
    for s, e in clusters:
        pools.setdefault(int(s), []).append(float(e))
    have = sorted(pools)

    def pool_of(s):
        c = [h for h in have if h <= s]
        return c[-1] if c else have[0]
    plans = [z for z in sizes_all if z]
    flat_size = np.array([s for z in plans for s in z], int)
    flat_plan = np.array([i for i, z in enumerate(plans) for _ in z], int)
    flat_pool = np.array([pool_of(s) for s in flat_size], int)
    n_plans = len(plans)

    def share(pl, n_draw, g):
        above = 0
        for _ in range(n_draw):
            e = np.empty(len(flat_size))
            for s, arr in pl.items():
                m = flat_pool == s
                e[m] = np.asarray(arr)[g.integers(len(arr), size=int(m.sum()))]
            num = np.bincount(flat_plan, weights=e * flat_size, minlength=n_plans)
            den = np.bincount(flat_plan, weights=flat_size.astype(float), minlength=n_plans)
            above += int((num / den > line).sum())
        return above / (n_draw * n_plans)
    p = share(pools, draws, rng)
    bs = []
    for _ in range(int(boot)):
        pl = {s: list(np.asarray(v)[rng.integers(len(v), size=len(v))]) for s, v in pools.items()}
        bs.append(share(pl, 2, rng))
    return {"p": p, "interval": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "clusters": {str(s): len(v) for s, v in sorted(pools.items())}, "records": n_plans * draws,
            "plans": n_plans}


def panel_part(tree, rendered, ctx, plan_sha_logged, max_step=None):
    """Every rendered truth set encoded under the weights of FIXED (all of them) and of the panel checkpoints
    (the PANEL records): per record its value under each weight, the validation rows, the fixed-weight spread, the
    per-window paired differences against the start (person clusters)."""
    from mdfm.eval import infer as I
    from mdfm.factory import synth as SY
    from mdfm.factory import layouts as LY
    bs, cfg = ctx["stage2_s2b"]["bodies"], ctx["stage2_s2b"]["cfg"]
    t0 = time.time()
    stores = _stores(cfg)
    model0, plan_sha = start_model(tree, cfg, stores.replay)
    synth = SY.SynthSampler(bs, device="cpu", layouts=LY.load_stage2_layouts())
    kw = {"sources": tuple(model0.cfg.sources), "nuis_fields": model0.nuis_fields}
    batches = {}
    for run, plans in rendered.items():
        for s, p in plans.items():
            mb, its = render_truth(p, bs, synth, ctx[run]["cfg"], kw)
            if mb is not None:
                batches[(run, s)] = (mb, its)
    t_render = time.time() - t0
    weights = {"start": None}
    for run, steps in PANEL_CKPTS.items():
        for st in steps:
            if max_step is not None and st > max_step:
                continue
            weights[f"{run}:{st}"] = os.path.join(tree, RUNS_DIR, run, f"ckpt_C_{st:06d}.pt")
    panel_keys = {(run, s) for run, ss in PANEL.items() for s in ss}
    vals, per_window = {}, {}
    for wname, path in weights.items():
        model = model0 if path is None else I.load_checkpoint(path, weights="model", device="cpu").model
        for key, (mb, its) in batches.items():
            if wname not in FIXED and key not in panel_keys:
                continue
            e, size = errors(model, mb)
            vals.setdefault(key, {})[wname] = float(np.mean(e))
            per_window.setdefault(key, {})[wname] = e
            per_window[key]["_size"] = size
            per_window[key]["_persons"] = [it.person for it in its]
        if path is not None:
            del model
    # validation: the record whose health weights are those of a checkpoint (50 steps after it) or of the start
    valid = []
    for (run, s), v in sorted(vals.items()):
        own = "start" if s == LOG_EVERY else f"{run}:{s - LOG_EVERY}"
        if own in v:
            logged = ctx["records"][run][s]["logged"]
            valid.append({"run": run, "step": s, "weights": own, "recomputed": v[own], "logged": logged,
                          "difference": v[own] - logged})
    fixed = {}
    n_rec = records_in_phase()
    for w in FIXED:
        xs = [v[w] for (run, s), v in vals.items() if run == "stage2_s2b" and w in v and s >= FIRST]
        if not xs:
            continue
        fixed[w] = {"records": len(xs), "mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)),
                    "max": float(np.max(xs)), "above": int(sum(x > LINE for x in xs))}
        clusters = []
        for (run, s), pw in per_window.items():
            if run != "stage2_s2b" or w not in pw:
                continue
            g = {}
            for e, person in zip(pw[w], pw["_persons"]):
                g.setdefault(person, []).append(e)
            clusters += [(len(v), float(np.mean(v))) for v in g.values()]
        tm = tail_model(clusters, ctx["stage2_s2b"]["sizes"])
        tm["projection"] = {
            "records_per_phase": n_rec,
            **{f"consecutive_{k}": {"fire_at_p": 1 - p_no_run(n_rec, k, tm["p"]),
                                    "fire_at_p_high": 1 - p_no_run(n_rec, k, tm["interval"][1])} for k in (1, 2, 3)}}
        fixed[w]["tail_model"] = tm
    # paired per-window differences against the start on the panel's windows
    paired = {}
    for w in weights:
        if w == "start":
            continue
        d, g, lvl = [], [], []
        for key in panel_keys:
            pw = per_window.get(key)
            if not pw or w not in pw or "start" not in pw:
                continue
            d += [a - b for a, b in zip(pw[w], pw["start"])]
            lvl += pw[w]
            g += [f"{key[0]}:{key[1]}:{p}" for p in pw["_persons"]]
        if d:
            m, ci, ng = cluster_boot(d, g)
            paired[w] = {"windows": len(d), "clusters": ng, "mean_error": float(np.mean(lvl)),
                         "difference_vs_start": m, "interval": ci}
    start_mean = None
    lv = [x for key in panel_keys if key in per_window and "start" in per_window[key] for x in per_window[key]["start"]]
    if lv:
        start_mean = float(np.mean(lv))
    alarm = per_window.get(ALARM_RECORD)
    alarm_out = None
    if alarm:
        alarm_out = {"values": vals.get(ALARM_RECORD), "persons": alarm["_persons"],
                     "beta_size": alarm["_size"],
                     "per_window": {w: alarm[w] for w in alarm if not w.startswith("_")}}
    # the size of the true betas of the panel's windows (mean |beta|): where the alarm's persons sit
    sizes = [x for key in panel_keys if key in per_window for x in per_window[key]["_size"]]
    errs8000 = []
    for key in batches:
        pw = per_window.get(key, {})
        if "stage2_s2b:8000" in pw:
            errs8000 += list(zip(pw["stage2_s2b:8000"], pw["_size"], pw["_persons"]))
    corr = None
    if len(errs8000) > 3:
        a = np.array([x[0] for x in errs8000])
        b = np.array([x[1] for x in errs8000])
        corr = float(np.corrcoef(a, b)[0, 1])
    persons = {}
    for e, size, p in errs8000:
        persons.setdefault(p, []).append(e)
    top = sorted(((float(np.mean(v)), p, len(v)) for p, v in persons.items()), reverse=True)[:5]
    return {"plan_sha256": plan_sha, "plan_sha256_logged": plan_sha_logged, "plan_matches": plan_sha == plan_sha_logged,
            "rendered": len(batches), "render_seconds": t_render, "seconds": time.time() - t0,
            "values": {f"{k[0]}:{k[1]}": v for k, v in sorted(vals.items())}, "validation": valid,
            "fixed_weights": fixed, "paired_vs_start": paired, "panel_start_mean_error": start_mean,
            "alarm_plan": alarm_out,
            "beta_size": {"windows": len(sizes), "median": float(np.median(sizes)) if sizes else None,
                          "p90": float(np.quantile(sizes, 0.9)) if sizes else None,
                          "max": float(np.max(sizes)) if sizes else None},
            "error_vs_beta_size_corr_8000": corr,
            "top_persons_8000": [{"person": p, "mean_error": m, "windows": n} for m, p, n in top],
            "windows_8000": len(errs8000)}


# ---------------------------------------------------------------- summary
def summary_md(R):
    L = ["# The beta_error alarm of run S2b (task S2BA)", "",
         "Dev diagnostic, not a gate read. Every number below is in results.json of this folder, written by "
         "scripts/stage2_beta_alarm.py.", ""]
    lg = R["logs"]
    L += ["## 1. The logs", "",
          f"Line {LINE} from step {FIRST} of the phase (mdfm/train/loop.py ALARMS). Records from step {FIRST} on.", "",
          "| run | phase | records | mean | sd | max (step) | above 2.0 | lag-1 autocorr. | diff. sd / sqrt 2 | "
          "slope per 1000 steps (se) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, ent in sorted(lg["runs"].items()):
        for ph, st in sorted(ent["phases"].items()):
            if "mean" not in st:
                L.append(f"| {name} | {ph} | {st['from_first']} | | | | | | | |")
                continue
            L.append(f"| {name} | {ph} | {st['from_first']} | {st['mean']:.3f} | {st['sd']:.3f} | {st['max']:.3f} "
                     f"({st['max_step']}) | {st['above']} | {fnum(st['lag1_autocorrelation'])} | "
                     f"{st['diff_sd_over_sqrt2']:.3f} | {fnum(st['slope_per_1000_steps'], 4)} "
                     f"({fnum(st['slope_se_per_1000_steps'], 4)}) |")
    L += ["", "Logs without a beta_error record: " + ", ".join(lg["no_beta_error"]) + ".", "",
          "Pooled per phase kind (records from step 1,000 on; exact 95 percent interval of the share above 2.0):", ""]
    for k, p in lg["pools"].items():
        L.append(f"- phase {k} ({', '.join(p['runs'])}): {p['records']} records, {p['above']} above 2.0, share "
                 f"{fnum(p['share'], 4)} [{fnum(p['interval'][0], 4)}, {fnum(p['interval'][1], 4)}]; mean "
                 f"{fnum(p['mean'])}, sd {fnum(p['sd'])}")
    ar = lg["alarm_record"]
    L += ["", f"The record of the stop: S2b step {ALARM_RECORD[1]}, {ar['value']:.4f}, rank {ar['rank_in_run']} of "
              f"the run's records; the six before it: " + ", ".join(f"{x['step']} {x['value']:.3f}"
                                                                    for x in ar["previous"]) + "."]
    c = lg["s2a_vs_s2b"]
    L += ["", "## 2. S2a against S2b", "",
          f"Common steps {c['steps'][0]} to {c['steps'][1]} ({c['records']} records each): S2a mean {c['s2a_mean']:.3f} "
          f"(sd {c['s2a_sd']:.3f}), S2b mean {c['s2b_mean']:.3f} (sd {c['s2b_sd']:.3f}); S2b minus S2a "
          f"{c['difference']:+.3f} (se {c['difference_se']:.3f}). Mean before step 1,000: S2a "
          f"{lg['before_first_mean']['stage2_s2a']:.3f}, S2b {lg['before_first_mean']['stage2_s2b']:.3f}.", "",
          "| run | steps | records | mean | max |", "|---|---|---|---|---|"]
    for name, rows in lg["by_thousand"].items():
        for r in rows:
            L.append(f"| {name} | {r['from']} to {r['to']} | {r['records']} | {r['mean']:.3f} | {r['max']:.3f} |")
    L += ["", "## 3. Readings of the line on every log", "",
          "A reading fires when its statistic is above 2.0 (records from step 1,000 on): single, the loop's today; "
          "consecutive_k, k records in a row above the line; mean_k and median_k, the mean and median of the last k "
          "records.", "", "| run | phase | " + " | ".join(n for n, _, _ in READINGS) + " |",
          "|---|---|" + "---|" * len(READINGS)]
    for name, ent in sorted(lg["runs"].items()):
        for ph, st in sorted(ent["phases"].items()):
            cells = []
            for n, _, _ in READINGS:
                r = st["readings"][n]
                if r["max_statistic"] is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{'FIRES at ' + str(r['first_step']) if r['fired'] else 'no'} (max "
                                 f"{r['max_statistic']:.3f})")
            L.append(f"| {name} | {ph} | " + " | ".join(cells) + " |")
    pj = lg["projection_C"]
    L += ["", f"Projection on one phase C of {PHASE_C_STEPS:,} steps ({pj['records_per_phase']} records from step "
              f"1,000), records i.i.d. from the pooled phase C records ({pj['pool']}, {pj['above']} above 2.0; p "
              f"{pj['p']:.4f}, exact interval [{pj['p_interval'][0]:.4f}, {pj['p_interval'][1]:.4f}]): the share of "
              f"phases in which each reading fires at least once.", "",
          "| reading | resampled pooled records | exact at p | exact at p's upper bound |", "|---|---|---|---|"]
    for n, kind, k in READINGS:
        r = pj["readings"][n]
        L.append(f"| {n} | {r['resampled_fire_share']:.4f} | {fnum(r.get('exact_fire_at_p'), 4)} | "
                 f"{fnum(r.get('exact_fire_at_p_high'), 4)} |")
    if R.get("replay"):
        L += ["", "## 4. What each record held (plan replay)", ""]
        for run, rp in R["replay"].items():
            bad = [c for c in rp["checks"] if not c["all"]]
            how = "yes, every one" if not bad else ("not at " + ", ".join(
                f"{c['checkpoint']} (" + ", ".join(f"{a} {'equal' if c[a] else 'differs'}" for a in
                                                   ("step", "rng", "credit", "window_tally") if a in c)
                + (f"; tally difference {c['tally_difference']}" if c.get("tally_difference") else "") + ")"
                for c in bad) + ": the plans of the segment that ends there are left out (the plans alone do not "
                "reproduce the training's draws there; sampler.materialize, which the replay does not run, can drop a "
                "planned window and so move the window tally), and the next segment starts from the checkpoint's state")
            L.append(f"- {run}: {rp['plans']} plans replayed in {rp['seconds']:.0f} s, {rp['plans_verified']} verified; "
                     f"sampler state equal to the checkpoint's at {', '.join(str(c['checkpoint']) for c in rp['checks'])}: "
                     f"{how}. Every verified plan: truth windows mean "
                     f"{rp['plans_truth']['mean']:.2f}, median {rp['plans_truth']['median']:.0f}; share of plans with at "
                     f"most 2, 4 truth windows {rp['plans_truth']['share_le']['2']:.4f}, "
                     f"{rp['plans_truth']['share_le']['4']:.4f}; with at most 1, 2 AMASS persons "
                     f"{rp['plans_truth']['persons_share_le']['1']:.4f}, {rp['plans_truth']['persons_share_le']['2']:.4f}.")
            a = rp.get("association")
            if a:
                L.append(f"  Verified health records from step 1,000 ({a['records']}): Spearman of the logged value with the "
                         f"truth windows {a['spearman_truth']:+.3f}, with the persons {a['spearman_persons']:+.3f}; "
                         + "; ".join(f"{b['persons']} persons: {b['records']} records, mean {b['mean']:.3f}, max "
                                     f"{b['max']:.3f}" for b in a["by_persons"]) + ".")
        rec = R["replay"]["stage2_s2b"]["records"].get(ALARM_RECORD[1]) or \
            R["replay"]["stage2_s2b"]["records"].get(str(ALARM_RECORD[1]))
        if rec:
            L.append(f"- The record of the stop (S2b step {ALARM_RECORD[1]}, plan {rec['plan']}): {rec['windows']} "
                     f"windows, {rec['truth']} with a body truth, {rec['bodies']} bodies ({rec['pairs']} render pairs), "
                     f"{rec['persons']} AMASS persons ({', '.join(rec['person_ids'])}).")
        if R.get("stage1_plans"):
            for ph, s in R["stage1_plans"].items():
                L.append(f"- v1's phase {ph} sampler, {s['plans']} plans: windows mean {s['windows_mean']:.1f}, truth "
                         f"windows mean {s['truth_mean']:.2f} (median {s['truth_median']:.0f}), AMASS persons mean "
                         f"{s['persons_mean']:.2f}; share with at most 2 persons {s['persons_share_le']['2']:.4f}.")
    pn = R.get("panel")
    if pn:
        L += ["", "## 5. Recomputed on the CPU", "",
              f"Migration plan of the start {pn['plan_sha256'][:12]}..., the runs' init_stage2 plan "
              f"{pn['plan_sha256_logged'][:12]}...: {'equal' if pn['plan_matches'] else 'DIFFERENT'}. "
              f"{pn['rendered']} truth sets rendered.", "", "Validation (recomputed at the record's own weights):", ""]
        for v in pn["validation"]:
            L.append(f"- {v['run']} step {v['step']} under {v['weights']}: {v['recomputed']:.4f} against logged "
                     f"{v['logged']:.4f} (difference {v['difference']:+.5f})")
        L += ["", "Every S2b health record from step 1,000 under fixed weights:", ""]
        for w, f in pn["fixed_weights"].items():
            L.append(f"- {w}: {f['records']} records, mean {f['mean']:.3f}, sd {f['sd']:.3f}, max {f['max']:.3f}, "
                     f"above 2.0: {f['above']}")
            tm = f.get("tail_model")
            if tm:
                pj = tm["projection"]
                L.append(f"  Cluster model ({tm['plans']} replayed S2b plans x {TAIL_DRAWS} draws; clusters per size "
                         f"{tm['clusters']}): share of records above 2.0 {tm['p']:.5f} [{tm['interval'][0]:.5f}, "
                         f"{tm['interval'][1]:.5f}]; over {pj['records_per_phase']} records of a phase C, P(fires) "
                         + ", ".join(f"{k} {pj[k]['fire_at_p']:.4f} (at the upper bound {pj[k]['fire_at_p_high']:.4f})"
                                     for k in ("consecutive_1", "consecutive_2", "consecutive_3")) + ".")
        ap = pn.get("alarm_plan")
        if ap:
            L += ["", f"The stop's micro-step (persons {', '.join(ap['persons'])}; mean |beta_true| per window "
                      f"{', '.join(f'{x:.3f}' for x in ap['beta_size'])}; panel windows' mean |beta_true| median "
                      f"{pn['beta_size']['median']:.3f}, 90th percentile {pn['beta_size']['p90']:.3f}, max "
                      f"{pn['beta_size']['max']:.3f}):", ""]
            for w, v in ap["values"].items():
                L.append(f"- {w}: {v:.4f} (windows {', '.join(f'{x:.3f}' for x in ap['per_window'][w])})")
        where = "; ".join(f"{run} records {', '.join(str(s) for s in ss)}" for run, ss in PANEL.items())
        L += ["", f"Panel windows ({where}), mean per-window error and paired difference against the start (person "
                  f"clusters, 95 percent interval of {BOOT_N} resamples); start mean "
                  f"{fnum(pn['panel_start_mean_error'])}:", ""]
        for w, p in pn["paired_vs_start"].items():
            L.append(f"- {w}: mean {p['mean_error']:.3f}, minus start {p['difference_vs_start']:+.3f} "
                     f"[{p['interval'][0]:+.3f}, {p['interval'][1]:+.3f}] ({p['windows']} windows, {p['clusters']} "
                     f"clusters)")
        L += ["", f"Under S2b ckpt_C_008000, over {pn['windows_8000']} truth windows: correlation of a window's error "
                  f"with its mean |beta_true| {fnum(pn['error_vs_beta_size_corr_8000'])}; highest mean errors: "
                  + ", ".join(f"{t['person']} {t['mean_error']:.3f} ({t['windows']} windows)"
                              for t in pn["top_persons_8000"]) + "."]
    return "\n".join(L) + "\n"


def write(R, out_dir):
    from mdfm.gates import common
    text = summary_md(R)
    common.assert_no_em_dash(text, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(os.path.join(out_dir, "results.json"), json.dumps(R, indent=1, allow_nan=False, default=str) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    try:
        spec = importlib.util.spec_from_file_location("_s2ba_stamp", os.path.join(HERE, "moveatlas_stamp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stamp = {"head": mod.stamp().get("head")}
    except Exception as e:  # noqa: BLE001  (reported; the measurement reads no MoveAtlas file)
        stamp = {"error": f"{type(e).__name__}: {e}"}
    prov = {"script": "scripts/stage2_beta_alarm.py", "task": TASK, "label": "dev diagnostic, not a gate read",
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "moveatlas_stamp": stamp,
            "code_sha256": common.code_sha256([os.path.abspath(__file__)]), "argv": R.get("argv"),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--replay", action="store_true", help="part 2: the plans of S2a and S2b")
    p.add_argument("--panel", action="store_true", help="part 3 (with --replay): recompute on the CPU")
    p.add_argument("--stage1-plans", type=int, default=0, help="plans of v1's phase A and B samplers")
    p.add_argument("--max-step", type=int, default=None, help="a check of the script: replay to this step only")
    p.add_argument("--out", default=OUT)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.panel and not a.replay:
        raise SystemExit("--panel renders the plans that --replay draws")
    R = {"task": TASK, "label": "dev diagnostic, not a gate read", "argv": sys.argv[1:] if argv is None else argv,
         "line": LINE, "first_step": FIRST, "logs": logs_part(TREE)}
    if a.replay:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2") or 2))
        R["replay"], rendered, ctx = {}, {}, {"records": {}}
        for run in ("stage2_s2a", "stage2_s2b"):
            want = set(PANEL[run]) if a.panel else set()
            if a.panel and run == "stage2_s2b":
                want |= {s for s in range(LOG_EVERY, ALARM_RECORD[1] + 1, LOG_EVERY)}
            want = {s for s in want if a.max_step is None or s <= a.max_step}
            res, rend, c3 = replay(TREE, run, sorted(want), a.max_step)
            res["association"] = association(res["records"])
            R["replay"][run] = res
            rendered[run] = rend
            ctx[run] = c3
            ctx["records"][run] = res["records"]
            print(f"{run}: {res['plans']} plans, checks {[c['all'] for c in res['checks']]}, {res['seconds']:.0f} s",
                  flush=True)
        if a.stage1_plans:
            R["stage1_plans"] = stage1_plans(TREE, a.stage1_plans)
        if a.panel:
            logged_plan = next((r.get("plan_sha256") for r in read_jsonl(os.path.join(TREE, RUNS_DIR, "stage2_s2b",
                                                                                       "log.jsonl"))
                                if r.get("event") == "init_stage2"), None)
            R["panel"] = panel_part(TREE, rendered, ctx, logged_plan, a.max_step)
        for run in R["replay"]:
            R["replay"][run]["records"] = {str(k): v for k, v in sorted(R["replay"][run]["records"].items())}
    text = summary_md(R)
    print(text)
    if a.write:
        write(R, os.path.join(TREE, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
