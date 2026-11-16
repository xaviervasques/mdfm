"""S2.9: stage-2 bench and budget at 60M (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 4.6 and 6, the S2.9 row of
section 7). Writes reports/bench/stage2_budget.md and reports/bench/stage2_budget.json for a stage-2 configuration
(default configs/stage2_s2a.yaml) from the stage-1 root, the stage-2 root, the body store's takes table, the queue's
job log and, once it exists, the GPU bench of the configuration (reports/bench/<config name>_bench.json, written by
--bench). No number of the report is typed: each comes from these inputs.

The default call (CPU only; no window read, no render, no GPU):
1. Inputs: the configuration and its SHA-256, the MoveAtlas HEAD now (git log -1, read only) and the HEAD the stage-2
   store was built at (reports/window_store_s2/_provenance.json), the SHA-256 of each root's store.json with its
   per-source file hashes, and the data SHA-256 of the tables the phase C sampler draws from.
2. Subject-hours per person of each unit, from recording time: per recording the longest span among its records, so
   that a recording is counted once whatever the number of its single-site records; a pads recording is the pair of
   wrist records <task>_wrist_L and <task>_wrist_R of one row of S2.0's pairing table
   (reports/stage2_sources/items/pads_pairs.csv), every other record a recording of its own. The training-fold
   recordings of at least mixture.min_s are those the sampler draws; the sum over a unit's persons is compared with
   sampler.unit_hours (the sampler's weights), which must agree.
3. Square-root weights and shares: per unit the square root of its subject-hours, its proportional share of the
   pathological tokens (1 - replay_share) before the caps, the share the sampler keeps after the caps (unit_cap,
   capped_shares), and the share of all planned tokens over PLANS plans of the configuration's phase C sampler (plans
   only; a synthetic window's count is its layout's upper bound), with the planned tokens per optimizer step.
4. Tokens per pass per unit: the tokens of the sampler's windows (min(win_s, span) s each, sampler.real_tokens at the
   recording's start) whose durations add up to each training recording's span, summed over the unit; the steps of
   one pass of each unit at its planned tokens per optimizer step, and its passes in the 4,000-step screen.
5. Rate: the seconds per phase C optimizer step of the GPU bench of the configuration when its JSON exists (kept
   prefetch, the configuration's refinement cadence; the bench must have read the same sampler tables, its data
   SHA-256). Otherwise the stage-2 rate is NOT MEASURED and the step counts are given at the measured stage-1 phase B
   rates as references: the pilot bench (reports/bench/stage1_pilot.json), every whole phase B training job of the
   queue (reports/gpu_queue/job_log.csv: hours over the phase B steps of its configuration) and every whole phase B
   that ran outside the queue (OUT_OF_QUEUE_RUNS: v1, whose run log gives the phase B clock of J1's output [A]); dev
   reads and checkpoints are included in a whole run's rate and not in a bench's. Each whole run's rate over the pilot
   bench's is its whole-run ratio; with a stage-2 bench, a further row reads the bench's rate times the largest ratio,
   the bench rate with the dev reads and checkpoints of the slowest stage-1 run. The phase C rate differs from the
   stage-1 rates: its replay renders 0.30 x 0.45 of the tokens against 0.45 in phase B, and its real windows are read
   from the stage-2 store.
6. Step counts and GPU-hours under the rule of spec section 6: the token budget of a stage-2 run is the smaller of
   (i) the WALL_BUDGET_H GPU-hour cap at the stage-2 rate and (ii) E passes over the capped pathological mixture, E
   fixed in the config after the per-unit dev NLL of the 4,000-step screen. One pass over the capped mixture is the
   step count at which the planned pathological tokens equal the sum of the units' tokens per pass; per unit the
   passes at that count differ, as the caps set its share, not its size. Rows for E in E_GRID and for the E that the
   cap reaches at each rate.
7. Costs beside the run's: the 4,000-step screen (SCREEN_STEPS) at each rate, and the post-train jobs of a stage-2
   run (POST_TRAIN): those spec sections 4.12 and 10 name (basis "4.12"), and those the stage-1 post-train templates
   (cache/runs/queue/post_train/post_j1.json and post_diag.json, run patterns "*") would also start on a stage-2 run
   folder they do not exclude (basis "template"). Each costs the median measured duration of the queue jobs of its
   kind (classified by script and options, done with exit code 0), SP3 the median seconds of its summaries
   (reports/synth/sp3/*/summary.json); a kind with no measured job is named and not counted. The total is an upper
   bound: G4 and the G3 PD half are priced at stage-1 jobs that scored more persons than their stage-2 rows hold, and
   the template kinds are counted until task S2.10 fixes the stage-2 templates. The per-unit dev NLL and the
   plausibility share are logged by the run's own dev reads (section 4.7) and cost no job of their own.

--bench (a GPU queue job, never run by this task): the configuration's Trainer on the two stores, init_stage2 from
--source (default the v1 phase B end, whose recipe stage1_pilot the DRAFT carries), then timed phase C passes
(unrefined and every step refined: the time split) and untimed passes at prefetch 0 and PREFETCH at the
configuration's refinement cadence; the faster is kept. It writes reports/bench/<config name>_bench.json (bench_record:
the config, data and store.json SHA-256s, the source and its SHA-256) from a run folder of its own
(cache/runs/_bench_<config name>/<time>), then the budget above. The weights train during the bench and no checkpoint
is written.

Run with the mdfm environment:
  <conda>/envs/mdfm/python.exe scripts/stage2_budget.py
  <conda>/envs/mdfm/python.exe scripts/stage2_budget.py --bench --memory-gb 12   (GPU; queue job)
"""
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402

import bench_stage1 as BN  # noqa: E402  (plan helpers, the pass runner and moveatlas_commit; sets the allocator variable)
from mdfm import paths  # noqa: E402
from mdfm.data import sampler as SM  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.train import config as CF  # noqa: E402
from mdfm.train import loop as LP  # noqa: E402

CONFIG = os.path.join(TREE, "configs", "stage2_s2a.yaml")
OUT = os.path.join(TREE, "reports", "bench")
STEM = "stage2_budget"
PADS_PAIRS = os.path.join(TREE, "reports", "stage2_sources", "items", "pads_pairs.csv")
PAIRED_SOURCE = "pads"            # the source whose single-site records pair into recordings (S2.0's table)
S2_PROVENANCE = os.path.join(TREE, "reports", "window_store_s2", "_provenance.json")
JOB_LOG = os.path.join(TREE, "reports", "gpu_queue", "job_log.csv")
SP3_SUMMARIES = os.path.join(TREE, "reports", "synth", "sp3", "*", "summary.json")
PILOT_JSON = BN.PILOT_JSON
RUNS = os.path.join(paths.CACHE, "runs")
SOURCE = os.path.join(RUNS, "phaseA_root_full", "ckpt_B_032044.pt")   # v1's phase B end
OUT_OF_QUEUE_RUNS = ("phaseA_root_full",)   # v1: its phase B ran before the queue's job log (spec section 1, item 7)
WALL_BUDGET_H = 24.0             # spec section 6, rule (i): the pilot's cap of one run [R reports/bench/stage1_pilot.md]
SCREEN_STEPS = 4000              # spec section 6: the screen whose per-unit dev NLL fixes E
E_GRID = (1, 2, 5, 10, 20, 50, 100, 200)
PLANS = 2000                     # phase C plans read for the planned shares (S2.7's plan check read 2,000)
PREFETCH = BN.PREFETCH
BENCH_STEPS = {"c_timed": 20, "c_timed_refined": 10, "c_rate": 60}
BENCH_FORMAT = "mdfm-stage2-bench/2"   # /2: the bench records the sampler's data SHA-256, which the budget checks

# Post-train jobs of a stage-2 run: (kind, what it reads, count per run, basis). The kinds are those of job_kind.
# Basis "4.12": named by spec section 4.12, the attainability decodes by section 10 (the start decoded twice before the
# hash, on the read's lines: the G3 dev lines and G1a). Basis "template": not named by 4.12, started by the stage-1
# post-train templates (cache/runs/queue/post_train/post_j1.json, post_diag.json; run patterns "*") on any run folder
# they do not exclude, so counted until S2.10's stage-2 templates replace them. The 4.12 cohort probe (the
# clinical_gait_multipatho group from frozen tokens of the start and the end) has no script yet: it is priced at the
# measured frozen-token probe (probe_frozen_tokens.py), and twice, as its start reading on multipatho windows does not
# exist (the frozen-token reports of v1's B end probe weargait_pd walkway targets).
BASIS_SPEC, BASIS_TEMPLATE = "4.12", "template"
POST_TRAIN = (
    ("g3_dev", "G3 dev of the end: the read's S(12), V(12), S(14), V(14), stride_mape_pct, walkway IC (EMA)", 1,
     BASIS_SPEC),
    ("g1_dev", "G1 dev of the end: the read's G1a line", 1, BASIS_SPEC),
    ("g3_dev", "attainability of the read rule: the start's G3 dev lines decoded twice (section 10)", 2, BASIS_SPEC),
    ("g1_dev", "attainability of the read rule: the start's G1a decoded twice (section 10)", 2, BASIS_SPEC),
    ("stage2_read", "the read under the hashed rule (scripts/stage2_read.py, task S2.10)", 1, BASIS_SPEC),
    ("g3_pd_dev", "G3 PD half on the PD dev fold; upper bound: the measured jobs scored every non-lockbox PD row, a "
                  "stage-2 checkpoint's trained PD persons leave them", 1, BASIS_SPEC),
    ("g4_dev", "G4 on the stroke fold of 8; upper bound: the measured jobs scored the 40 non-lockbox stroke patients",
     1, BASIS_SPEC),
    ("sp3", "SP3 families on the end", 1, BASIS_SPEC),
    ("frozen_tokens", "cohort probe from frozen tokens of the start and the end, priced at the measured frozen-token "
                      "probe", 2, BASIS_SPEC),
    ("g3_dev_model", "G3 dev of the end, model weights (post_j1)", 1, BASIS_TEMPLATE),
    ("insim", "in-simulation gait of the end (post_j1)", 1, BASIS_TEMPLATE),
    ("diag", "G3 diagnostics of the end (post_diag)", 1, BASIS_TEMPLATE),
    ("diag_curve", "G3 learning-curve table, CPU (post_diag)", 1, BASIS_TEMPLATE),
)


# ---------------------------------------------------------------- inputs
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), TREE).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def stage2_stores(cfg):
    """sampler.Stage2Stores of a stage-2 configuration, as scripts/train_stage1.stage2_stores opens them: the stage-1
    root under names.STAGE1 at RELEASE and the stage-2 root of its tier under its name tables (with the strata table
    when the configuration names one). Tables only are read until a window is asked for."""
    from mdfm.data import strata as ST
    from mdfm.train import names as NM
    replay = WS.WindowStore(WS.store_root(), tier="RELEASE", names=NM.STAGE1)
    strata = ST.load(cfg.losses.hsic_strata, cfg.losses.hsic_strata_sha256) if str(cfg.losses.hsic_strata) else None
    stage2 = WS.WindowStore(WS.store_root(kind=WS.TIER_KIND[cfg.tier]), tier=cfg.tier, names=CF.name_tables(cfg),
                            strata=strata)
    return SM.Stage2Stores(replay, stage2)


def store_hashes(stores):
    """Per root: the SHA-256 of its store.json and, per source, the file hashes it records."""
    out = {}
    for name, st in (("replay", stores.replay), ("stage2", stores.stage2)):
        p = os.path.join(st.root, "store.json")
        d = {"root": rel(st.root), "store_json_sha256": sha256_file(p) if os.path.exists(p) else None, "sources": {}}
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            for s, v in sorted((meta.get("sources") or {}).items()):
                d["sources"][s] = dict(sorted((v.get("files_sha256") or {}).items()))
        out[name] = d
    return out


def store_head(path=S2_PROVENANCE):
    """The MoveAtlas HEAD the stage-2 store was built at (its _provenance.json), None without the file."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("moveatlas_head")


# ---------------------------------------------------------------- subject-hours
def pads_pairing(path=PADS_PAIRS):
    """S2.0's pairing table: ({(person_id, record key): recording}, rows). The two wrist records <task>_wrist_L and
    <task>_wrist_R of one row are the one recording (person, task)."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        for side in ("L", "R"):
            out[(str(r["person_id"]), f"{r['task']}_wrist_{side}")] = f"pair:{r['task']}"
    return out, rows


def person_hours(persons, keys, spans, pairing=None):
    """{person: subject-hours} from recording time: records grouped into recordings (pairing[(person, key)] when the
    pairing holds the record, the record itself otherwise), per recording the longest span (s) among its records,
    summed per person, in hours. A recording held by two single-site records counts once."""
    best = {}
    for p, k, s in zip(persons, keys, spans):
        p, k = str(p), str(k)
        r = (pairing or {}).get((p, k), f"record:{k}")
        best[(p, r)] = max(best.get((p, r), 0.0), float(s))
    out = {}
    for (p, _), s in best.items():
        out[p] = out.get(p, 0.0) + s / 3600.0
    return dict(sorted(out.items()))


def pairing_check(persons, keys, pairing, rows):
    """How the store's records of the paired source sit in the pairing table: records, records in the table, records
    outside it, recordings the table makes of them, recordings holding both wrists, and the table's pairs of unequal
    length (same_length False) among the store's persons."""
    keys = [(str(p), str(k)) for p, k in zip(persons, keys)]
    inside = [pk for pk in keys if pk in pairing]
    recs = {}
    for p, k in inside:
        recs.setdefault((p, pairing[(p, k)]), set()).add(k)
    held = {p for p, _ in keys}
    unequal = sum(1 for r in rows if str(r["person_id"]) in held and str(r.get("same_length", "")).lower() != "true")
    return {"records": len(keys), "in_table": len(inside), "outside_table": len(keys) - len(inside),
            "recordings": len(recs), "both_wrists": sum(1 for v in recs.values() if len(v) == 2),
            "table_rows": len(rows), "unequal_length_pairs": unequal}


def unit_persons(sm, key, stores):
    """(persons, keys, spans) of a key's training recordings of at least min_s in the sampler's table, one per
    recording row (the store's recordings table gives the keys)."""
    R = sm.real[key]
    store = sm.store_of.get(key, stores.stage2)
    recs = store.recordings
    persons = recs.loc[R["rows"], "person_id"].astype(str).tolist()
    keys = recs.loc[R["rows"], "key"].astype(str).tolist()
    return persons, keys, [float(x) for x in R["span"]]


# ---------------------------------------------------------------- tokens per pass
def pass_tokens(sm, key):
    """(tokens of one pass, recordings) of a key: per training recording j, span_j / d_j windows of d_j =
    min(win_s, span_j) s at sampler.real_tokens(key, j, span_t0_j, d_j) tokens, summed (module docstring, item 4)."""
    R = sm.real[key]
    win = float(sm.cfg.win_s)
    tot = 0.0
    for j in range(len(R["rows"])):
        span = float(R["span"][j])
        d = min(win, span)
        tot += span / d * sm.real_tokens(key, j, float(R["span_t0"][j]), d)
    return tot, len(R["rows"])


def short_recordings(stores, units, min_s):
    """{unit: training recordings shorter than min_s} of the stage-2 store (the sampler leaves them out)."""
    recs = stores.stage2.recordings
    out = {}
    for u in units:
        src = str(u).split(":", 1)[0]
        g = recs[(recs["source"] == src) & (recs["fold"] == "train")]
        out[u] = int((g["span_s"] < float(min_s) - 1e-9).sum())
    return out


# ---------------------------------------------------------------- plans
def planned(sm, n):
    """Planned tokens over n plans of a phase C sampler (no render): per key the mean tokens per micro-step, the mean
    and minimum micro-step, plans drawn, the first empty plan (None: none) and the draws made again."""
    tot, sizes, first = {}, [], None
    for k in range(int(n)):
        try:
            t = sm.plan().tokens()
        except SM.EmptyStep:
            first = k
            break
        sizes.append(sum(t.values()))
        for s, v in t.items():
            tot[s] = tot.get(s, 0) + int(v)
    m = len(sizes)
    return {"plans": m, "first_empty": first, "redrawn": int(sm.redrawn),
            "tokens_mean": float(np.mean(sizes)) if m else None, "tokens_min": int(min(sizes)) if m else None,
            "per_micro": {s: v / m for s, v in sorted(tot.items())} if m else {}}


# ---------------------------------------------------------------- rates
def job_hours(row):
    try:
        a = _dt.datetime.fromisoformat(row["start"])
        b = _dt.datetime.fromisoformat(row["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return (b - a).total_seconds() / 3600.0


def _argv(row):
    try:
        return [str(x) for x in json.loads(row.get("argv") or "[]")]
    except ValueError:
        return []


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def read_jobs(path=JOB_LOG):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _done(row):
    return row.get("state") == "done" and str(row.get("exit_code")) == "0"


def queue_phase_b_rates(rows):
    """[{job, config, steps, hours, s_per_step}] of the queue's whole phase B training jobs: train_stage1.py --phase B,
    done with exit code 0, not resumed from a phase B checkpoint, its output the phase B end of its configuration's
    step count (config.step_counts)."""
    out = []
    for r in rows:
        a = _argv(r)
        if not _done(r) or not any(x.endswith("train_stage1.py") for x in a) or _opt(a, "--phase") != "B":
            continue
        res = _opt(a, "--resume")
        if res and "ckpt_B_" in res:
            continue
        cfg_path = _opt(a, "--config")
        h = job_hours(r)
        if not cfg_path or h is None:
            continue
        p = cfg_path if os.path.isabs(cfg_path) else os.path.join(TREE, cfg_path)
        if not os.path.exists(p):
            continue
        steps = int(CF.step_counts(CF.load(p))[1])
        if f"ckpt_B_{steps:06d}.pt" not in str(r.get("outputs", "")):
            continue
        out.append({"job": r["name"], "config": cfg_path, "steps": steps, "hours": h, "s_per_step": h * 3600.0 / steps,
                    "start": r["start"]})
    return out


def queue_run_dirs(rows):
    """Run folder names (--run-dir) of the queue's phase B training jobs, whatever their state."""
    out = set()
    for r in rows:
        a = _argv(r)
        if any(x.endswith("train_stage1.py") for x in a) and _opt(a, "--phase") == "B" and _opt(a, "--run-dir"):
            out.add(os.path.basename(_opt(a, "--run-dir").replace("\\", "/").rstrip("/")))
    return out


def _events(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def run_log_phase_b(run_dir):
    """{job, config, steps, hours, s_per_step, start} of a run folder's whole phase B from its log.jsonl: the clock
    from the last 'resume' or 'init_from' event before the first phase B log line to the phase_end B event (as J1's
    output [A] reads it), over the phase B steps of the run's start event. None without the log or a phase_end B, when
    the run started again between them (a 'start' event: a crash and a resume), or when the phase B log lines do not
    add up to the phase B step count (not a whole phase B)."""
    p = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(p):
        return None
    ev = _events(p)
    first_b = next((i for i, r in enumerate(ev) if r.get("event") == "log" and r.get("phase") == "B"), None)
    end_b = next((i for i, r in enumerate(ev) if r.get("event") == "phase_end" and r.get("phase") == "B"), None)
    if first_b is None or end_b is None or end_b < first_b:
        return None
    starts = [i for i, r in enumerate(ev) if r.get("event") == "start"]
    if any(first_b < i < end_b for i in starts):
        return None
    s = max((i for i in starts if i < first_b), default=None)
    t0 = max((i for i, r in enumerate(ev[:first_b]) if r.get("event") in ("resume", "init_from")), default=None)
    if s is None or t0 is None or t0 < s:
        return None
    steps = int((ev[s].get("steps") or [0, 0])[1])
    logged = sum(int(r.get("optimizer_steps") or 0) for r in ev[first_b:end_b]
                 if r.get("event") == "log" and r.get("phase") == "B")
    if steps <= 0 or logged != steps:
        return None
    h = (_dt.datetime.fromisoformat(ev[end_b]["time"]) - _dt.datetime.fromisoformat(ev[t0]["time"])).total_seconds()
    h /= 3600.0
    return {"job": os.path.basename(os.path.normpath(run_dir)), "config": ev[s].get("name"), "steps": steps,
            "hours": h, "s_per_step": h * 3600.0 / steps, "start": ev[t0]["time"]}


def out_of_queue_rates(rows, runs=OUT_OF_QUEUE_RUNS, root=RUNS):
    """run_log_phase_b of each run folder of `runs` under root whose phase B is not a queue job (a queue job's run
    folder is counted by queue_phase_b_rates, or left out there as resumed or failed)."""
    queued = queue_run_dirs(rows)
    out = []
    for name in runs:
        if name in queued:
            continue
        r = run_log_phase_b(os.path.join(root, name))
        if r is not None:
            out.append(r)
    return out


def pilot_rate(path=PILOT_JSON):
    """The pilot bench's phase B rate (refine every N, kept prefetch) and its date, None without the file."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        R = json.load(f)
    return {"s_per_step": float(R["budget"]["s_b"]), "date": R["env"]["date"], "N": int(R["refine"]["N"]),
            "prefetch": int(R["prefetch"])}


def bench_path(cfg, out=OUT):
    return os.path.join(out, f"{cfg.name}_bench.json")


def read_bench(path, cfg):
    """The kept phase C rate of a bench JSON of this configuration (its config SHA-256 must match), None without the
    file. ValueError: a bench of another configuration or format. The budget also checks its data SHA-256."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        B = json.load(f)
    if B.get("format") != BENCH_FORMAT:
        raise ValueError(f"{path}: format {B.get('format')!r}, expected {BENCH_FORMAT}")
    if B.get("config_sha256") != CF.sha256(cfg):
        raise ValueError(f"{path} benched the configuration {B.get('config_sha256')}, not {CF.sha256(cfg)}: rerun "
                         f"the bench (--bench)")
    return B


def check_bench_data(bench, data_sha256):
    """ValueError when a bench read other sampler tables than the budget's (a store rebuilt since the bench)."""
    if bench is not None and bench.get("data_sha256") != data_sha256:
        raise ValueError(f"the bench read the sampler tables {bench.get('data_sha256')}, the budget reads "
                         f"{data_sha256}: the store changed since the bench, rerun it (--bench)")


def rates(cfg, bench, rows, pilot, runs=()):
    """The rates the step counts are read at: the stage-2 bench's kept rate when it exists (measured, stage 2), and the
    stage-1 references (pilot bench, whole phase B queue jobs, whole phase B runs outside the queue: `runs`, of
    out_of_queue_rates). A whole run's 'ratio' is its rate over the pilot bench's; with a stage-2 bench, a row
    'stage2_bench_whole_run' reads the bench's rate times the largest ratio (module docstring, item 5)."""
    out = []
    if bench is not None:
        k = bench["kept"]
        out.append({"label": f"stage-2 bench ({bench['env']['date']}, {k['label']})", "kind": "stage2_bench",
                    "s_per_step": float(k["s_per_step"])})
    p = None
    if pilot is not None:
        p = float(pilot["s_per_step"])
        out.append({"label": f"stage-1 pilot bench, phase B refine every {pilot['N']} ({pilot['date']})",
                    "kind": "stage1_bench", "s_per_step": p})
    whole = [("queue job", q) for q in queue_phase_b_rates(rows)] + [("run outside the queue", q) for q in runs]
    for how, q in whole:
        out.append({"label": f"stage-1 {how} {q['job']}, whole phase B of {q['config']} ({q['steps']:,} steps, "
                             f"{q['hours']:.3f} h)", "kind": "stage1_run", "s_per_step": q["s_per_step"],
                    "ratio": q["s_per_step"] / p if p else None})
    ratios = [(r["ratio"], r["label"]) for r in out if r.get("ratio") is not None]
    if bench is not None and ratios:
        f, lab = max(ratios)
        out.insert(1, {"label": f"stage-2 bench x {f:.4f}, the largest whole-run ratio of stage 1 ({lab})",
                       "kind": "stage2_bench_whole_run", "s_per_step": float(bench["kept"]["s_per_step"]) * f,
                       "ratio": f})
    return out


# ---------------------------------------------------------------- the rule of section 6
def cap_steps(rate, hours=WALL_BUDGET_H):
    """Optimizer steps within `hours` GPU-hours at `rate` s per step (rule (i))."""
    return int(math.floor(hours * 3600.0 / float(rate) + 1e-9))


def rule_steps(rate, e, pass_steps, hours=WALL_BUDGET_H):
    """(steps, binding) of rule section 6: the smaller of the cap's steps at `rate` and e passes of pass_steps
    (rounded up); binding 'cap' or 'passes'."""
    c = cap_steps(rate, hours)
    p = int(math.ceil(float(e) * float(pass_steps) - 1e-9))
    return (p, "passes") if p <= c else (c, "cap")


def gpu_hours(steps, rate):
    return float(steps) * float(rate) / 3600.0


# ---------------------------------------------------------------- post-train costs
def job_kind(argv):
    """The post-train kind of a queue job's argv (POST_TRAIN), None for any other job."""
    a = [str(x) for x in argv]
    script = next((os.path.basename(x) for x in a if x.endswith(".py")), "")
    if script == "eval_g3_dev.py":
        if _opt(a, "--weights") == "model":
            return "g3_dev_model"
        return "g3_curve" if "--no-secondary" in a else "g3_dev"
    if script == "eval_gait_insim.py":
        if "--sites" in a:
            return "insim_sites"
        return "insim" if "--real" in a else "insim_curve"
    if script == "eval_g1_dev.py":
        return "g1_lt2" if "--limit-takes" in a else "g1_dev"
    if script == "eval_g3_diag.py":
        if "--curve" in a:
            return "diag_curve"
        if "--ablations" in a:
            return "diag_ablations"
        return "diag" if "--checkpoint" in a else None
    if script == "eval_g4_dev.py":
        return "g4_dev"
    if script == "eval_g3_pd_dev.py":
        return "g3_pd_dev"
    if script == "probe_frozen_tokens.py":
        return "frozen_tokens"
    if script == "stage2_read.py":
        return "stage2_read"
    return None


def whole_minutes(rows, kinds):
    """(jobs, of them lasting a whole number of minutes) among the done queue jobs of `kinds`: the resolution of
    their logged hours."""
    secs = [int(round(h * 3600.0)) for r in rows if _done(r) and job_kind(_argv(r)) in kinds
            for h in (job_hours(r),) if h is not None]
    return len(secs), sum(1 for s in secs if s % 60 == 0)


def measured_costs(rows, sp3_glob=SP3_SUMMARIES):
    """{kind: {hours: median, n, jobs}} of the done queue jobs per kind (job_kind), and 'sp3' from the seconds of the
    SP3 summaries."""
    by = {}
    for r in rows:
        k = job_kind(_argv(r))
        h = job_hours(r)
        if k is None or h is None or not _done(r):
            continue
        by.setdefault(k, []).append((r["name"], h))
    out = {k: {"hours": float(np.median([h for _, h in v])), "n": len(v), "jobs": [n for n, _ in v]}
           for k, v in sorted(by.items())}
    sp3 = []
    for p in sorted(glob.glob(sp3_glob)):
        with open(p, encoding="utf-8") as f:
            s = json.load(f).get("seconds")
        if s is not None:
            sp3.append((rel(os.path.dirname(p)), float(s) / 3600.0))
    if sp3:
        out["sp3"] = {"hours": float(np.median([h for _, h in sp3])), "n": len(sp3), "jobs": [n for n, _ in sp3]}
    return out


def post_train_cost(costs, plan=POST_TRAIN):
    """Rows of the post-train set with their measured hours (count x median), the total of the measured ones (an upper
    bound, module docstring item 7), its part per basis and the kinds without a measured job."""
    rows, total, by_basis, missing = [], 0.0, {}, []
    for kind, what, n, basis in plan:
        c = costs.get(kind)
        h = None if c is None else n * c["hours"]
        rows.append({"kind": kind, "what": what, "count": n, "basis": basis,
                     "hours_each": None if c is None else c["hours"], "measured_jobs": 0 if c is None else c["n"],
                     "hours": h})
        by_basis.setdefault(basis, 0.0)
        if h is None:
            missing.append(kind)
        else:
            total += h
            by_basis[basis] += h
    return {"rows": rows, "total_hours": total, "by_basis": by_basis, "not_measured": sorted(set(missing))}


# ---------------------------------------------------------------- the budget
def phase_c_sampler(cfg, bs, stores):
    """The phase C sampler the budget reads (plans only, no renderer, no guard), at the run's phase C seed."""
    return SM.MixtureSampler(bs, stores, None, LP.STAGE2_PHASE, LP.phase_seed(cfg, LP.STAGE2_PHASE), cfg, guard=False,
                             redraws=int(cfg.stage2.redraws))


def budget(cfg, bs, stores, plans=PLANS, bench=None, jobs=None, pilot=None, pairs=PADS_PAIRS, head=None,
           s2_head=None, config_path=None, runs=(), sp3_glob=SP3_SUMMARIES):
    """The budget of a stage-2 configuration (module docstring, items 1 to 7) as a dict. `runs`: the whole phase B
    rates of stage-1 runs outside the queue (out_of_queue_rates); sp3_glob: the SP3 summaries priced. ValueError:
    hours other than the sampler's, or a bench that read other sampler tables."""
    if not CF.stage2_on(cfg):
        raise ValueError(f"{cfg.name} has no enabled stage2 section: the budget reads a stage-2 configuration")
    jobs = [] if jobs is None else jobs
    seed = LP.phase_seed(cfg, LP.STAGE2_PHASE)
    sm = phase_c_sampler(cfg, bs, stores)
    data_sha = sm.data_sha256()
    check_bench_data(bench, data_sha)
    st = cfg.stage2
    units = list(st.units)
    R = float(st.replay_share)
    mps = int(cfg.schedule.micro_per_step)
    tpm = int(cfg.mixture.tokens_per_micro)
    # 2. subject-hours per person
    pairing, pair_rows = pads_pairing(pairs) if pairs and os.path.exists(pairs) else ({}, [])
    hours, persons_out, pair_info = {}, {}, None
    for u in units:
        ps, ks, sp = unit_persons(sm, u, stores)
        src = u.split(":", 1)[0]
        ph = person_hours(ps, ks, sp, pairing if src == PAIRED_SOURCE else None)
        vals = np.array(list(ph.values()), dtype=np.float64)
        if abs(float(vals.sum()) - float(sm.hours[u])) > 1e-9 * max(1.0, float(sm.hours[u])):
            raise ValueError(f"{u}: subject-hours {float(vals.sum())} from the pairing table differ from the "
                             f"sampler's {float(sm.hours[u])} (sampler.unit_hours): the weights would not be these")
        hours[u] = {"persons": len(ph), "hours": float(vals.sum()), "sampler_hours": float(sm.hours[u]),
                    "per_record_hours": float(sum(sp)) / 3600.0,
                    "median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)), "min": float(vals.min()), "max": float(vals.max())}
        persons_out[u] = ph
        if src == PAIRED_SOURCE:
            pair_info = pairing_check(ps, ks, pairing, pair_rows)
    # 3. weights and shares
    w = {u: math.sqrt(hours[u]["hours"]) for u in units}
    tw = sum(w.values())
    P = planned(phase_c_sampler(cfg, bs, stores), plans)
    tot_planned = sum(P["per_micro"].values())
    replay_keys = [SM.SYNTHETIC] + [s for s in sm.order if s in sm.store_of and sm.store_of[s] is stores.replay]
    shares = {}
    for k in sm.order:
        shares[k] = {"weight": w.get(k, math.sqrt(sm.hours[k]) if k in sm.hours else None),
                     "before_caps": (1.0 - R) * w[k] / tw if k in w else None,
                     "after_caps": float(sm.shares[k]),
                     "cap": float(sm.caps[k]) if k in (sm.caps or {}) else None,
                     "planned": P["per_micro"].get(k, 0.0) / tot_planned if tot_planned else None,
                     "tokens_per_step": mps * P["per_micro"].get(k, 0.0),
                     "group": "unit" if k in units else "replay"}
    # 4. tokens per pass
    short = short_recordings(stores, units, cfg.mixture.min_s)
    passes = {}
    for u in units:
        T, n = pass_tokens(sm, u)
        per_step = mps * P["per_micro"].get(u, 0.0)
        passes[u] = {"tokens": T, "recordings": n, "short_left_out": short[u], "tokens_per_step": per_step,
                     "steps_per_pass": T / per_step if per_step else None,
                     "passes_in_screen": SCREEN_STEPS * per_step / T if T else None}
    path_tokens = sum(p["tokens"] for p in passes.values())
    path_per_step = sum(p["tokens_per_step"] for p in passes.values())
    pass_steps = path_tokens / path_per_step if path_per_step else None
    # 5. rates
    rate_rows = rates(cfg, bench, jobs, pilot, runs)
    # 6. steps and GPU-hours under the rule
    table = []
    for r in rate_rows:
        rr = {"label": r["label"], "kind": r["kind"], "s_per_step": r["s_per_step"],
              "cap_steps": cap_steps(r["s_per_step"]), "rows": []}
        rr["cap_passes"] = rr["cap_steps"] / pass_steps if pass_steps else None
        for e in E_GRID:
            n, bind = rule_steps(r["s_per_step"], e, pass_steps)
            rr["rows"].append({"E": e, "steps": n, "binding": bind, "gpu_hours": gpu_hours(n, r["s_per_step"]),
                               "unit_passes": {u: n * passes[u]["tokens_per_step"] / passes[u]["tokens"]
                                               for u in units if passes[u]["tokens"]}})
        rr["screen_gpu_hours"] = gpu_hours(SCREEN_STEPS, r["s_per_step"])
        table.append(rr)
    # 7. post-train
    costs = measured_costs(jobs, sp3_glob)
    post = post_train_cost(costs)
    return {
        "format": "mdfm-stage2-budget/1", "date": _dt.date.today().isoformat(),
        "config": rel(config_path) if config_path else cfg.name, "config_name": cfg.name,
        "config_sha256": CF.sha256(cfg), "config_status": cfg.status, "names_sha256": CF.name_tables(cfg).sha256(),
        "moveatlas_head": head, "store_moveatlas_head": s2_head, "stores": store_hashes(stores),
        "data_sha256": data_sha, "seed": seed,
        "replay_share": R, "unit_cap": float(st.unit_cap), "source_cap": float(cfg.mixture.source_cap),
        "synthetic_share_b": float(cfg.mixture.synthetic_share[1]), "units": units, "replay_keys": replay_keys,
        "tokens_per_micro": tpm, "micro_per_step": mps, "win_s": float(cfg.mixture.win_s),
        "min_s": float(cfg.mixture.min_s), "stage2_steps": int(st.steps), "redraws": int(st.redraws),
        "hours": hours, "persons": persons_out, "pairing": pair_info, "pairs_table": rel(pairs) if pairs else None,
        "shares": shares, "planned": {k: v for k, v in P.items() if k != "per_micro"},
        "planned_per_micro": P["per_micro"], "passes": passes, "pathological_tokens": path_tokens,
        "pathological_tokens_per_step": path_per_step, "mixture_pass_steps": pass_steps,
        "rates": rate_rows, "stage2_rate_measured": bench is not None, "bench": bench,
        "rule": {"wall_budget_h": WALL_BUDGET_H, "E_grid": list(E_GRID), "screen_steps": SCREEN_STEPS},
        "table": table, "post_train": post, "post_train_kinds": costs,
        "post_train_whole_minutes": whole_minutes(jobs, {k for k, _, _, _ in POST_TRAIN}),
    }


# ---------------------------------------------------------------- the report
def _f(x, n=3):
    return "n/a" if x is None else f"{x:,.{n}f}"


def write_md(B, path):
    L = []
    a = L.append
    units = B["units"]
    a("# Stage-2 budget at 60M (task S2.9)")
    a("")
    a(f"Written by scripts/stage2_budget.py on {B['date']} for {B['config']} (status: {B['config_status']}; config "
      f"SHA-256 {B['config_sha256']}; name tables SHA-256 {B['names_sha256']}). Spec "
      f"docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 4.6 and 6. Plans only on the CPU: no window read, no render, no "
      f"GPU.")
    a("")
    a("## 1. Inputs")
    a("")
    a(f"- MoveAtlas HEAD at this run (git log -1, read only): {B['moveatlas_head']}; the stage-2 store was built at "
      f"{B['store_moveatlas_head']} (reports/window_store_s2/_provenance.json).")
    for k, d in B["stores"].items():
        a(f"- {k} root {d['root']}: store.json SHA-256 {d['store_json_sha256']}; per source, records.csv SHA-256: "
          + "; ".join(f"{s} {f.get('records.csv', 'n/a')[:16]}" for s, f in d["sources"].items()) + ".")
    a(f"- Tables the phase C sampler draws from (sampler.data_sha256, the body-store takes and both roots' training "
      f"recordings): {B['data_sha256']}; sampler seed {B['seed']} (loop.phase_seed of phase C).")
    a(f"- Phase C: replay {B['replay_share']} of a micro-step's tokens (synthetic {B['replay_share']} x "
      f"{B['synthetic_share_b']}, replay sources capped at {B['source_cap']}), units capped at {B['unit_cap']} of all "
      f"tokens; {B['tokens_per_micro']:,} tokens per micro-step x {B['micro_per_step']} micro-steps per optimizer "
      f"step; windows of at most {B['win_s']} s, recordings under {B['min_s']} s left out; stage2.steps "
      f"{B['stage2_steps']:,} in the configuration (the screen's count); redraws {B['redraws']}.")
    a("")
    a("## 2. Subject-hours per person (training fold, recordings of at least min_s)")
    a("")
    a("Per recording the longest span among its records; a recording of two single-site records counts once. "
      "'Sampler' is sampler.unit_hours, from which the square-root weights are drawn; 'per record' sums every "
      "record's span, the count this budget does not use.")
    a("")
    a("| unit | persons | subject-hours | sampler | per record | per person median | p10 | p90 | min | max |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for u in units:
        h = B["hours"][u]
        a(f"| {u} | {h['persons']} | {h['hours']:.3f} | {h['sampler_hours']:.3f} | {h['per_record_hours']:.3f} | "
          f"{h['median']:.4f} | {h['p10']:.4f} | {h['p90']:.4f} | {h['min']:.4f} | {h['max']:.4f} |")
    a("")
    pi = B["pairing"]
    if pi is not None:
        a(f"pads under S2.0's pairing table ({B['pairs_table']}, {pi['table_rows']:,} rows): {pi['records']:,} "
          f"training records, {pi['in_table']:,} in the table and {pi['outside_table']:,} outside it; they make "
          f"{pi['recordings']:,} recordings, {pi['both_wrists']:,} of them with both wrists; pairs of unequal length "
          f"among the store's persons: {pi['unequal_length_pairs']}.")
        a("")
    a("## 3. Square-root weights and shares")
    a("")
    a(f"Shares of all tokens of a micro-step. Before the caps: (1 - {B['replay_share']}) x sqrt(hours) / sum of "
      f"sqrt(hours) over the units. After the caps: the sampler's budget shares (capped_shares). Planned: the share of "
      f"all planned tokens over {B['planned']['plans']:,} phase C plans (first empty plan: "
      f"{B['planned']['first_empty']}; draws made again: {B['planned']['redrawn']}; planned tokens per micro-step mean "
      f"{_f(B['planned']['tokens_mean'], 1)}, minimum {B['planned']['tokens_min']}).")
    a("")
    a("| key | group | sqrt(hours) | before the caps | cap | after the caps | planned | planned tokens per "
      "optimizer step |")
    a("|---|---|---|---|---|---|---|---|")
    for k, s in B["shares"].items():
        a(f"| {k} | {s['group']} | {_f(s['weight'], 4)} | {_f(s['before_caps'], 4)} | {_f(s['cap'], 2)} | "
          f"{s['after_caps']:.4f} | {_f(s['planned'], 4)} | {_f(s['tokens_per_step'], 1)} |")
    a("")
    rep = sum(B["shares"][k]["planned"] or 0.0 for k in B["replay_keys"])
    a(f"Replay share of all planned tokens: {rep:.4f} (configured {B['replay_share']}).")
    a("")
    a("## 4. Tokens per pass per unit")
    a("")
    a("One pass: the tokens of the sampler's windows whose durations add up to each training recording's span (both "
      "wrist records of a pads task are data, so both count here, while the subject-hours count the recording once).")
    a("")
    a(f"| unit | recordings | under min_s, left out | tokens per pass | planned tokens per optimizer step | optimizer "
      f"steps per pass | passes in the {SCREEN_STEPS:,}-step screen |")
    a("|---|---|---|---|---|---|---|")
    for u in units:
        p = B["passes"][u]
        a(f"| {u} | {p['recordings']:,} | {p['short_left_out']} | {p['tokens']:,.0f} | {p['tokens_per_step']:,.1f} | "
          f"{_f(p['steps_per_pass'], 1)} | {_f(p['passes_in_screen'], 1)} |")
    a("")
    a(f"Pathological mixture: {B['pathological_tokens']:,.0f} tokens per pass of every unit, "
      f"{B['pathological_tokens_per_step']:,.1f} planned pathological tokens per optimizer step, so one pass over the "
      f"capped mixture takes {_f(B['mixture_pass_steps'], 1)} optimizer steps (the planned pathological tokens equal "
      f"the summed tokens per pass).")
    ranked = sorted((u for u in units if B["passes"][u]["passes_in_screen"] is not None),
                    key=lambda u: B["passes"][u]["passes_in_screen"])
    if ranked:
        lo, hi = ranked[0], ranked[-1]
        a(f"Across units the caps, not the data, set the shares: the screen makes "
          f"{B['passes'][hi]['passes_in_screen']:,.1f} passes over the data of {hi} and "
          f"{B['passes'][lo]['passes_in_screen']:,.1f} over those of {lo} (spec risk 3; per-unit dev NLL, section 4.7).")
    a("")
    a("## 5. Rate")
    a("")
    if B["stage2_rate_measured"]:
        bn = B["bench"]
        k = bn["kept"]
        a(f"Stage-2 rate measured by the bench of this configuration ({bn['env']['date']}, "
          f"{bn['env'].get('gpu')}; started from {bn.get('source')}, SHA-256 {bn.get('source_sha256')}; sampler "
          f"tables {bn.get('data_sha256')}, those of this budget): {k['s_per_step']:.3f} s per phase C optimizer step "
          f"({k['label']}), {_f(k.get('tokens_per_s'), 0)} useful tokens per second; in its timed unrefined pass the "
          f"host parts (sampling, rendering, collate) took {_f(k.get('host_share_timed'), 3)} of a micro-step.")
    else:
        a("Stage-2 rate: NOT MEASURED. No GPU bench of this configuration exists (scripts/stage2_budget.py --bench, a "
          "queue job not queued by S2.9). The rows below read the measured stage-1 phase B rates as references "
          "only; the phase C rate differs from them (module docstring, item 5), and the budget is final only at the "
          "bench's rate.")
    a("")
    a("A bench rate leaves out the dev reads (every 1,000 steps) and the checkpoints of a run; a whole run's rate "
      "holds them. 'Whole run over bench' is a stage-1 whole run's rate over the pilot bench's (the bench's "
      "refinement cadence and prefetch, the run's own configuration). It holds run-to-run variation as well as those "
      "costs, so a ratio under 1 can occur.")
    a("")
    a("| rate | s per optimizer step | whole run over bench |")
    a("|---|---|---|")
    for r in B["rates"]:
        a(f"| {r['label']} | {r['s_per_step']:.4f} | {_f(r.get('ratio'), 4)} |")
    a("")
    rat = [r["ratio"] for r in B["rates"] if r["kind"] == "stage1_run" and r.get("ratio") is not None]
    if rat:
        a(f"Whole-run ratios of stage 1: {min(rat):.4f} to {max(rat):.4f}. "
          + ("The row 'stage-2 bench x' reads the stage-2 bench at the largest of them: rule (i) at the bench rate "
             "alone overstates the steps a 24 GPU-hour run can take." if B["stage2_rate_measured"] else
             "Once the stage-2 bench exists, a row reads it at the largest of them, since rule (i) at a bench rate "
             "alone overstates the steps a 24 GPU-hour run can take."))
        a("")
    a("## 6. Step counts and GPU-hours under the rule of section 6")
    a("")
    a(f"Steps = min(rule (i): the steps of {B['rule']['wall_budget_h']:.0f} GPU-hours at the rate; rule (ii): E passes "
      f"of {_f(B['mixture_pass_steps'], 1)} steps over the capped mixture). E is fixed after the screen's per-unit dev "
      f"NLL; the grid shows where each rule binds. A whole-run rate includes the dev reads and checkpoints of a "
      f"stage-1 run; a bench rate does not (section 5).")
    a("")
    for t in B["table"]:
        a(f"### At {t['s_per_step']:.4f} s per step ({t['label']})")
        a("")
        a(f"Rule (i) alone: {t['cap_steps']:,} steps, {_f(t['cap_passes'], 1)} passes over the capped mixture.")
        a("")
        a("| E | steps | binding | GPU-hours | " + " | ".join(f"passes of {u}" for u in units) + " |")
        a("|---|---|---|---|" + "---|" * len(units))
        for r in t["rows"]:
            a(f"| {r['E']} | {r['steps']:,} | {r['binding']} | {r['gpu_hours']:.2f} | "
              + " | ".join(_f(r["unit_passes"].get(u), 1) for u in units) + " |")
        a("")
    a("## 7. Costs beside the run's")
    a("")
    a(f"The {SCREEN_STEPS:,}-step screen (spec section 6), training only (its read is the per-unit dev NLL of its "
      f"log): " + "; ".join(f"{t['screen_gpu_hours']:.2f} GPU-hours at {t['s_per_step']:.4f} s per step"
                            for t in B["table"]) + ".")
    a("")
    a(f"Post-train jobs of a stage-2 run, each at the median measured duration of the done queue jobs of its kind "
      f"(reports/gpu_queue/job_log.csv; SP3 from reports/synth/sp3/*/summary.json). Basis {BASIS_SPEC}: named by spec "
      f"sections 4.12 and 10. Basis {BASIS_TEMPLATE}: not named by 4.12, started by the stage-1 post-train templates "
      f"(cache/runs/queue/post_train/post_j1.json and post_diag.json, run pattern \"*\") on any run folder they do "
      f"not exclude, until task S2.10's stage-2 templates replace them. The per-unit dev NLL and the plausibility "
      f"share come from the run's own dev reads (section 4.7) and cost no job.")
    a("")
    a("| kind | what | basis | count | hours each (median) | measured jobs | hours |")
    a("|---|---|---|---|---|---|---|")
    for r in B["post_train"]["rows"]:
        a(f"| {r['kind']} | {r['what']} | {r['basis']} | {r['count']} | {_f(r['hours_each'], 3)} | "
          f"{r['measured_jobs']} | {_f(r['hours'], 3)} |")
    a("")
    nm = B["post_train"]["not_measured"]
    bb = B["post_train"]["by_basis"]
    a(f"Post-train total of the measured kinds: {B['post_train']['total_hours']:.3f} hours, an upper bound (G4 and "
      f"the G3 PD half priced at stage-1 jobs that scored more persons; the template kinds counted): basis "
      f"{BASIS_SPEC} {bb.get(BASIS_SPEC, 0.0):.3f} h, basis {BASIS_TEMPLATE} {bb.get(BASIS_TEMPLATE, 0.0):.3f} h"
      + (f"; not measured, not counted: {', '.join(nm)}." if nm else "."))
    n, m = B["post_train_whole_minutes"]
    a(f"Resolution: {m} of the {n} done queue jobs of these kinds last a whole number of minutes in the log, so their "
      f"hours carry about a minute's resolution.")
    a("")
    a("| rate | screen GPU-hours | run under rule (i) alone, GPU-hours | post-train hours | total |")
    a("|---|---|---|---|---|")
    for t in B["table"]:
        run_h = gpu_hours(t["cap_steps"], t["s_per_step"])
        tot = t["screen_gpu_hours"] + run_h + B["post_train"]["total_hours"]
        a(f"| {t['s_per_step']:.4f} | {t['screen_gpu_hours']:.2f} | {run_h:.2f} | "
          f"{B['post_train']['total_hours']:.3f} | {tot:.2f} |")
    a("")
    a("The run's row is the upper bound of rule (i); under rule (ii) the run is shorter (section 6 table).")
    a("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def write_budget(cfg, config_path, out=OUT, stem=STEM, plans=PLANS, bench_json=None, stores=None, bs=None,
                 job_log=JOB_LOG, pilot_json=PILOT_JSON, provenance=S2_PROVENANCE, pairs=PADS_PAIRS,
                 runs=OUT_OF_QUEUE_RUNS, runs_root=RUNS, sp3_glob=SP3_SUMMARIES):
    """budget() on the configuration's stores (or `stores` and `bs`) with its inputs read from their files, written
    to <out>/<stem>.md and .json; returns the budget dict."""
    if stores is None:
        stores = stage2_stores(cfg)
    if bs is None:
        from mdfm.data import body_store as BS
        bs = BS.BodyStore(split="train")
    bench = read_bench(bench_json or bench_path(cfg, out), cfg)
    jobs = read_jobs(job_log)
    B = budget(cfg, bs, stores, plans=plans, bench=bench, jobs=jobs, pilot=pilot_rate(pilot_json), pairs=pairs,
               head=BN.moveatlas_commit(), s2_head=store_head(provenance), config_path=config_path,
               runs=out_of_queue_rates(jobs, runs, runs_root), sp3_glob=sp3_glob)
    os.makedirs(out, exist_ok=True)
    md, js = os.path.join(out, f"{stem}.md"), os.path.join(out, f"{stem}.json")
    write_md(B, md)
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(LP._finite(B), f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print("wrote", rel(md), "and", rel(js), flush=True)
    return B


# ---------------------------------------------------------------- the GPU bench (a queue job)
def summarize(res):
    """Throughput and time split of a loop.bench result (bench_stage1.summarize's keys; the peaks None on the CPU):
    steps, s_per_step, tokens, tokens_per_s, tokens_per_micro, fill, the peaks; for a timed pass split_ms (mean ms per
    micro-step of each part of loop.PARTS) and host_share (loop.HOST_PARTS over all parts)."""
    micro = res["micro"]
    steps = max(int(res["steps"]), 1)
    wall = float(res["wall_s"])
    out = {"phase": res["phase"], "steps": res["steps"], "timed": res["timed"], "prefetch": res["prefetch"],
           "refine_every": res["refine_every"], "micro_steps": len(micro), "wall_s": wall, "s_per_step": wall / steps,
           "tokens": res["tokens"], "padded": res["padded"], "tokens_per_s": res["tokens"] / wall if wall else None,
           "tokens_per_micro": res["tokens"] / max(len(micro), 1), "fill": res["tokens"] / max(res["padded"], 1),
           "peak_allocated_gb": res["peak_allocated_gb"], "peak_reserved_gb": res["peak_reserved_gb"],
           "refined_micro": sum(1 for m in micro if m["refined"])}
    if res["timed"] and micro:
        parts = {k: 1000.0 * float(np.mean([m["time"].get(k, 0.0) for m in micro])) for k in LP.PARTS}
        tot = sum(parts.values())
        out.update(split_ms=parts, micro_ms=tot,
                   host_share=sum(parts[k] for k in LP.HOST_PARTS) / tot if tot else None)
    return out


def run_pass(tr, phase, steps, timed, refine_every=None, label=""):
    """loop.bench of one pass, summarized (summarize), with its label."""
    t0 = time.perf_counter()
    s = summarize(LP.bench(tr, phase, steps=steps, timed=timed, refine_every=refine_every, warmup=BN.WARMUP))
    s["label"] = label
    print(f"  {label}: {s['tokens_per_s']:,.0f} useful tokens/s, {s['s_per_step']:.3f} s per optimizer step, peak "
          f"reserved {_f(s['peak_reserved_gb'], 2)} GB ({time.perf_counter() - t0:.0f} s)", flush=True)
    return s


def bench_passes(tr, steps=None, prefetch=PREFETCH):
    """Phase C passes of a Trainer after init_stage2 (module docstring, --bench): timed unrefined and refined (the time
    split), then untimed at prefetch 0 and `prefetch` at the configuration's refinement cadence; returns {passes,
    kept} with kept the faster untimed pass."""
    st = dict(BENCH_STEPS if steps is None else steps)
    N = int(tr.cfg.schedule.refine_every)
    tr.set_prefetch(0)
    passes = [run_pass(tr, LP.STAGE2_PHASE, st["c_timed"], True, refine_every=0, label="C timed, unrefined"),
              run_pass(tr, LP.STAGE2_PHASE, st["c_timed_refined"], True, refine_every=1, label="C timed, refined")]
    r0 = run_pass(tr, LP.STAGE2_PHASE, st["c_rate"], False, refine_every=N, label=f"C prefetch 0, refine every {N}")
    tr.set_prefetch(int(prefetch))
    r2 = run_pass(tr, LP.STAGE2_PHASE, st["c_rate"], False, refine_every=N,
                     label=f"C prefetch {int(prefetch)}, refine every {N}")
    tr.set_prefetch(0)
    passes += [r0, r2]
    kept = min((r0, r2), key=lambda p: p["s_per_step"])
    return {"passes": passes, "kept": {"label": kept["label"], "s_per_step": kept["s_per_step"],
                                       "tokens_per_s": kept["tokens_per_s"], "tokens_per_micro": kept["tokens_per_micro"],
                                       "refine_every": N, "host_share_timed": passes[0].get("host_share")}}


def bench_record(cfg, config_path, source, bs, stores, res, run_dir=None, steps=None, env=None):
    """The bench JSON of bench_passes' result `res`: format, config and its SHA-256, the data SHA-256 of the phase C
    sampler's tables (phase_c_sampler, as the budget reads them; the budget refuses a bench of other tables), the
    store.json SHA-256 of both roots, the source checkpoint and its SHA-256, the run folder, the steps and env (date
    added)."""
    B = {"format": BENCH_FORMAT, "config": rel(config_path) if config_path else cfg.name,
         "config_sha256": CF.sha256(cfg), "data_sha256": phase_c_sampler(cfg, bs, stores).data_sha256(),
         "store_json_sha256": {k: v["store_json_sha256"] for k, v in store_hashes(stores).items()},
         "source": rel(source), "source_sha256": sha256_file(source),
         "run_dir": rel(run_dir) if run_dir else None, "steps": steps,
         "env": dict({"date": _dt.date.today().isoformat()}, **(env or {}))}
    B.update(res)
    return B


def run_bench(cfg, config_path, source, out=OUT, memory_gb=None, scale=1.0):
    import torch
    from mdfm.data import body_store as BS
    from mdfm.factory import synth as SY
    if not torch.cuda.is_available():
        raise SystemExit("--bench times phase C on the GPU (a queue job); CUDA is not available here")
    dev = torch.device("cuda")
    cap = float(LP.MEMORY_CAP_GB if memory_gb is None else memory_gb)
    run_dir = os.path.join(paths.CACHE, "runs", f"_bench_{cfg.name}", time.strftime("%Y%m%d_%H%M%S"))
    stores = stage2_stores(cfg)
    bs = BS.BodyStore(split="train")
    synth = SY.SynthSampler(bs, device=dev)
    tr = LP.Trainer(cfg, bs, stores, synth, dev, run_dir, echo=None, memory_cap_gb=cap)
    tr.init_stage2(source)
    steps = {k: max(2, int(round(v * scale))) for k, v in BENCH_STEPS.items()}
    res = bench_passes(tr, steps)
    props = torch.cuda.get_device_properties(dev)
    B = bench_record(cfg, config_path, source, bs, stores, res, run_dir=run_dir, steps=steps,
                     env={"gpu": props.name, "torch": torch.__version__, "moveatlas": BN.moveatlas_commit(),
                          "memory_cap_gb": cap, "parameters": int(sum(p.numel() for p in tr.model.parameters()))})
    path = bench_path(cfg, out)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(LP._finite(B), f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print("wrote", rel(path), flush=True)
    return B


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--plans", type=int, default=PLANS)
    ap.add_argument("--bench-json", default=None, help="the bench JSON read (default <out>/<config name>_bench.json)")
    ap.add_argument("--bench", action="store_true", help="time phase C on the GPU first (a queue job)")
    ap.add_argument("--source", default=SOURCE, help="--bench: the stage-1 phase B end init_stage2 reads")
    ap.add_argument("--memory-gb", type=float, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="--bench: scales the steps of every pass")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    path = args.config if os.path.isabs(args.config) else os.path.join(TREE, args.config)
    cfg = CF.load(path)
    if not CF.stage2_on(cfg):
        raise SystemExit(f"{args.config}: no enabled stage2 section")
    if args.plans <= 0:
        raise SystemExit(f"--plans {args.plans}: a positive number")
    if args.bench:
        run_bench(cfg, path, args.source, out=args.out, memory_gb=args.memory_gb, scale=args.scale)
    write_budget(cfg, path, out=args.out, stem=args.stem, plans=args.plans, bench_json=args.bench_json)


if __name__ == "__main__":
    main()
