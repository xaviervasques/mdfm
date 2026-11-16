"""Training of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25; mdfm/baselines/): a masked multimodal
transformer over the body model's tokens at equal compute, arm mfr (masked-feature reconstruction) or jepa (the JEPA
arm of docs/proposals/JEPA_EXPLORATION_DRAFT.md section 4).

Modes:
- --plan (CPU, no store opened): the equal-compute plan of configs/g12_baseline.yaml: the base configuration's
  SHA-256 against the pinned one, the reference budget of each body run log (mdfm/baselines/compute.py: clock,
  optimizer steps, tokens, per phase the median interval seconds per step, file and config SHA-256) and their mean,
  each log's clock over the mean and the largest over the smallest, per arm the measured seconds per step and the
  optimizer steps they give (none while not measured), the GPU-hours of --seeds seeds of each arm at the budget and
  at the runaway stop (1 + tolerance), and the parameter counts of each arm at the base configuration's size
  (mdfm/baselines/model.G12Model.parameter_counts, built on the CPU). Prints JSON.
- --attainability (CPU, no model, no record): the inputs of the attainability section of docs/rules/G12_RULE.md,
  read from files that exist: per stand-in checkpoint (the body run ends of STAND_INS) and per G3 dev set (dev12,
  dev14) and line (stride, speed, walkway IC), the decoded body (the G3 diag base row), the frozen-token probe on
  the body encoder's own latent tokens (the stand-in for a no-body encoder's probe) and their person-paired
  difference with its interval, as reports/bench/frozen_tokens_<name>/results.json holds them (scripts/
  probe_frozen_tokens.py); the paired interval's half width, scaled to the 17 lockbox controls by sqrt(n / 17) (an
  approximation: the per-person spread is taken as unchanged); the supervised baseline of the same file; the seed
  differences of each line between the two stand-ins; the subject-id leakage hook; the false-positive rates of the
  separation rule over exchangeable seeds: against one arm, exact (1 / 20, any continuous distribution); against
  the better of two arms picked as the rule picks it (the lower three-seed mean of the values), seeded Monte Carlo
  per seed distribution (SEP_DISTRIBUTIONS, SEP_DRAWS draws, seed ATTAIN_SEED), their range and the union bound
  (2 / 20, any continuous distribution); the exact rate with the arm picked by rank sum, kept as a reference (not
  the rule's pick); its power at gaps of 1 to 4 seed standard deviations against one arm and against the better of
  two (seeded Monte Carlo, ATTAIN_DRAWS draws, seed ATTAIN_SEED). Prints JSON with the files' SHA-256 and
  MoveAtlas commits.
- --bench N: N optimizer steps of each phase after 3 unmeasured ones (train/loop.bench, prefetch as given) on the
  GPU, the arm's seconds per optimizer step share_a x s_A + (1 - share_a) x s_B; with --write-config, written into
  the configuration's budget.seconds_per_step for the arm (the value the gpu_seconds plan divides the budget by).
  Writes no checkpoint and no log line. --smoke (with --bench; the GPU smoke of task G12Q, docs/rules/
  G12_RULE_REVISION_2026-10-08.md): every optimizer step of the bench, the unmeasured ones included, is recorded
  (loss, gradient norm, skipped, empty) and the smoke passes when, in both phases, every step's loss and gradient
  norm and every micro-step's loss are finite, no step is skipped, at least one step holds a masked window, and the
  peak reserved memory is measured and at most the run's bound (schedule.max_reserved_gb); a failed smoke writes
  nothing into the configuration and exits with status 5. --out PATH also writes the bench's JSON to PATH, with the
  time, the script's and the configuration's SHA-256 (before and after the write) and the MoveAtlas commit.
- default: the run (mdfm/baselines/trainer.G12Trainer.run): phases A then B, --max-steps to shorten a call (a
  checkpoint at the stop), --resume to continue; then checkpoint.load of the last checkpoint, logged as 'g12_verify':
  the GPU queue's post-train templates (cache/runs/queue/post_train/, run globs '*') fire on a body run's phase B
  'verify' event, and their body evaluations must never start on a baseline checkpoint.
  Exit status 3 after an alarm stop (the JEPA collapse alarm) and 4 after a compute overrun.

--arm and --seed override the file's values; --steps N runs N optimizer steps in budget mode 'steps' (smokes).

Reads: the AMASS body store (split train), the window store, the SMPL+H model files of MoveAtlas (read-only, the
checkpoint audit), the reports train/provenance.py reads, the body run logs of budget.reference_logs. Writes the run
folder only (train/loop.default_run_dir unless --run-dir: MDFM_RUNS, else <store>/runs, else cache/runs), and, with
--bench --write-config, the configuration file. Never a lockbox, EVAL or CODY record: every micro-step passes the
sampler's guard and the token collate's CODY refusal. The GPU only through scripts/queue_job.py (the queue runner
counts a train_g12 process as a GPU user); a CPU run sets CUDA_VISIBLE_DEVICES=-1 and --device cpu.

Usage (mdfm environment, from the tree root; through scripts/queue_job.py, name --config so that the queue's lock
and green-marker checks cover the configuration file, which a bench rewrites):
  python scripts/train_g12.py --plan
  python scripts/train_g12.py --attainability
  python scripts/train_g12.py --config configs/g12_baseline.yaml --arm mfr --bench 100 --write-config --prefetch 2
  python scripts/train_g12.py --config configs/g12_baseline.yaml --arm mfr --bench 100 --write-config --smoke \
      --out reports/bench/g12/smoke_mfr.json --prefetch 2 --memory-cap-gb 14.5
  python scripts/train_g12.py --config configs/g12_baseline.yaml --arm mfr --seed 20260928 --prefetch 2
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm.train import loop as LP  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", LP.ALLOC_CONF)

from dataclasses import replace  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.baselines import compute as CP  # noqa: E402
from mdfm.baselines import config as GC  # noqa: E402
from mdfm.train import checkpoint as CK  # noqa: E402
from mdfm.train import config as CF  # noqa: E402

PREFETCH = 2               # as the body run (reports/bench/stage1_pilot.md)
BENCH_WARMUP = 3
SPS_LINE = re.compile(r"^(\s*seconds_per_step:\s*)\{([^}]*)\}(.*)$", re.M)
STAND_INS = ("phaseA_root_full_ckpt_B_032044", "pilot_v1_s2_ckpt_B_032044")   # pilot v1 and seed 2, phase B ends
G3_SETS = ("dev12", "dev14")
G3_LINES = ("stride_mae_cm", "speed_mae_m_s", "ic_median_ms")
LOCKBOX_CONTROLS = 17                  # docs/rules/G3_RULE.md section 2
ATTAIN_SEED = 20261005
ATTAIN_DRAWS = 200_000
SEP_DRAWS = 1_000_000                  # per distribution, separation_rates_by_mean
SEP_CHUNK = 200_000
SEP_DISTRIBUTIONS = ("normal", "uniform", "t3", "cauchy", "exponential_right", "exponential_left",
                     "lognormal_right", "lognormal_left")
SMOKE_EXIT = 5                         # a failed --smoke (module docstring)
SMOKE_EDGE = 10                        # steps averaged for the first and last loss of a smoke phase


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=GC.CONFIG)
    ap.add_argument("--arm", choices=GC.ARMS, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--plan", action="store_true", help="print the equal-compute plan (CPU, no store)")
    ap.add_argument("--attainability", action="store_true", help="print the attainability inputs of the G12 rule")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per arm in the --plan cost")
    ap.add_argument("--bench", type=int, default=0, help="measure N optimizer steps of each phase")
    ap.add_argument("--write-config", action="store_true", help="with --bench: write the seconds per step")
    ap.add_argument("--smoke", action="store_true", help="with --bench: the smoke check (module docstring)")
    ap.add_argument("--out", default=None, help="with --bench: also write the JSON to this file")
    ap.add_argument("--steps", type=int, default=None, help="budget mode 'steps' with N optimizer steps")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--prefetch", type=int, default=PREFETCH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-cap-gb", type=float, default=None)
    a = ap.parse_args(argv)
    if (a.write_config or a.smoke or a.out) and not a.bench:
        ap.error("--write-config, --smoke and --out go with --bench")
    if a.bench < 0 or (a.steps is not None and a.steps < 1):
        ap.error("--bench and --steps are positive")
    return a


def load_config(a):
    cfg = GC.load(a.config, arm=a.arm, seed=a.seed)
    if a.steps is not None:
        cfg = GC.check(replace(cfg, budget=replace(cfg.budget, mode="steps", steps=int(a.steps))))
    return cfg


# ---------------------------------------------------------------- plan
def plan(cfg, seeds=3):
    """The equal-compute plan (module docstring) as a dict."""
    base, sha = GC.base_config(cfg)
    budget = CP.reference_budget(cfg.budget.reference_logs)
    arms = {}
    for arm in GC.ARMS:
        c = replace(cfg, arm=arm, budget=replace(cfg.budget, mode="gpu_seconds"))
        try:
            n, how = CP.plan_steps(c, budget)
        except ValueError as e:
            n, how = None, {"not_measured": str(e)}
        arms[arm] = {"seconds_per_step": float(cfg.budget.seconds_per_step.get(arm, 0.0)), "steps": n, "plan": how,
                     "steps_A_B": list(CF.step_counts(GC.run_config(c, base, n))) if n else None}
    per_seed_h = budget["hours"]
    tol = float(cfg.budget.tolerance)
    from mdfm.baselines.model import G12Model
    counts = {}
    for arm in GC.ARMS:
        m = G12Model.of(arm, base.model.mfm_config(), cfg.head)
        counts[arm] = m.parameter_counts()
        del m
    return {"parameters": counts, "base_config": cfg.base_config, "base_config_sha256": sha, "pinned": cfg.base_config_sha256,
            "pin_matches": sha == cfg.base_config_sha256,
            "base_steps_A_B": list(CF.step_counts(base)), "tokens_per_step": CF.tokens_per_step(base),
            "share_a": float(base.schedule.share_a),
            "reference": {k: budget[k] for k in ("wall_s", "hours", "steps", "tokens", "spread", "ratios",
                                                 "max_over_min")},
            "tolerance": tol,
            "logs": [{"path": r["path"], "sha256": r["sha256"], "config_sha256": r["config_sha256"],
                      "wall_s": r["wall_s"], "hours": r["wall_s"] / 3600.0, "steps": r["steps"], "tokens": r["tokens"],
                      "phases": r["phases"]} for r in budget["logs"]],
            "arms": arms, "seeds": int(seeds),
            "gpu_hours": {"per_seed": per_seed_h, "per_arm": per_seed_h * seeds,
                          "both_arms": per_seed_h * seeds * len(GC.ARMS),
                          "per_seed_at_stop": per_seed_h * (1.0 + tol),
                          "both_arms_at_stop": per_seed_h * (1.0 + tol) * seeds * len(GC.ARMS)}}


# ---------------------------------------------------------------- attainability
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def separation_rates():
    """Exact false-positive rates of the separation over exchangeable seeds, by enumeration of the rank orders:
    (body against one arm of three seeds, every body seed below every arm seed: 1 / C(6, 3) for any continuous
    distribution of the seed values; body against the better of two arms of three seeds with the better arm picked
    by RANK SUM, the lower sum, ties to the second arm; the number of assignments enumerated). The rank-sum pick is
    not the rule's: section 8 of docs/rules/G12_RULE.md picks the arm with the better three-seed mean of the values,
    whose rate depends on the distribution of the seed values (separation_rates_by_mean)."""
    six = set(range(6))
    one = sum(1 for c in itertools.combinations(range(6), 3) if max(c) < min(six - set(c))) / 20.0
    hits, n = 0, 0
    for body in itertools.combinations(range(9), 3):
        rest = [i for i in range(9) if i not in body]
        for a in itertools.combinations(rest, 3):
            b = [i for i in rest if i not in a]
            n += 1
            better = a if sum(a) < sum(b) else b
            hits += max(body) < min(better)
    return one, hits / n, n


def _draw(rng, name, shape):
    """Seed values of one distribution of SEP_DISTRIBUTIONS (a *_left distribution is the mirror of *_right)."""
    if name == "normal":
        return rng.standard_normal(shape)
    if name == "uniform":
        return rng.random(shape)
    if name == "t3":
        return rng.standard_t(3, size=shape)
    if name == "cauchy":
        return rng.standard_cauchy(size=shape)
    sign = -1.0 if name.endswith("_left") else 1.0
    if name.startswith("exponential_"):
        return sign * rng.exponential(size=shape)
    if name.startswith("lognormal_"):
        return sign * rng.lognormal(0.0, 1.0, size=shape)
    raise ValueError(f"unknown seed distribution {name!r}")


def separation_rates_by_mean(dists=None, draws=SEP_DRAWS, seed=ATTAIN_SEED, chunk=SEP_CHUNK):
    """False-positive rates of the rule's separation (docs/rules/G12_RULE.md section 8, lower is better: the
    comparator is the arm with the lower three-seed mean of the values; the body separates when every body seed is
    below every comparator seed) over 9 exchangeable seed values, 3 for the body and 3 per arm, seeded Monte Carlo of
    draws draws per distribution: {name: {'one_arm', 'better_of_two'}}. A higher-is-better line is the mirror image:
    its rate under a right-skewed distribution is the lower-is-better rate under the left-skewed one, hence both
    skews of the exponential and the lognormal."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for name in (SEP_DISTRIBUTIONS if dists is None else dists):
        one = better = done = 0
        while done < draws:
            m = min(chunk, draws - done)
            x = _draw(rng, name, (m, 9))
            body, a, b = x[:, :3], x[:, 3:6], x[:, 6:]
            comp = np.where(a.mean(1) < b.mean(1), a.min(1), b.min(1))
            top = body.max(1)
            one += int((top < a.min(1)).sum())
            better += int((top < comp).sum())
            done += m
        out[name] = {"one_arm": one / draws, "better_of_two": better / draws}
    return out


def separation_power(gaps=(1.0, 2.0, 3.0, 4.0), seeds=3, draws=ATTAIN_DRAWS, seed=ATTAIN_SEED, arms=1):
    """{gap: P(the body separates)} with seed values Normal(0, 1) for the body and Normal(gap, 1) for each of arms
    baseline arms (gaps in seed standard deviations), seeded Monte Carlo; arms 2: against the arm with the lower
    three-seed mean, as the rule picks it."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for g in gaps:
        b = rng.standard_normal((draws, seeds))
        a = g + rng.standard_normal((draws, seeds))
        if arms == 1:
            comp = a.min(1)
        else:
            c = g + rng.standard_normal((draws, seeds))
            comp = np.where(a.mean(1) < c.mean(1), a.min(1), c.min(1))
        out[str(g)] = float((b.max(1) < comp).mean())
    return out


def attainability(tree=TREE, stand_ins=STAND_INS):
    """The attainability inputs (module docstring) as a dict."""
    import math
    from mdfm.train import provenance as PV
    res, srcs = {}, []
    for name in stand_ins:
        p = os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}", "results.json")
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        srcs.append({"path": os.path.relpath(p, tree).replace(os.sep, "/"), "sha256": _sha(p),
                     "moveatlas_head": r.get("moveatlas_head"), "checkpoint_sha256": r["checkpoint"]["sha256"]})
        rows = {}
        for st in G3_SETS:
            for line in G3_LINES:
                dec = r["comparison"]["decoded"][st][line]
                if line == "ic_median_ms":
                    pr = r["ic"]["variants"]["latent"]["lines"][st]
                else:
                    pr = r["passes"]["latent"]["lines"][st][line]
                pd = r["comparison"]["paired_primary"][st][line]
                lo, hi = pd["interval"]
                hw = (hi - lo) / 2.0
                n = int(pd["n_persons"])
                sup = (r["comparison"]["supervised"].get(st) or {}).get(line) or {}
                rows[f"{st}/{line}"] = {
                    "decoded": dec["point"], "probe": pr["point"], "paired_probe_minus_decoded": pd["mean"],
                    "paired_interval": [lo, hi], "n_persons": n, "reading": pd["reading"], "half_width": hw,
                    "half_width_at_lockbox": hw * math.sqrt(n / LOCKBOX_CONTROLS), "supervised": sup.get("point")}
        res[name] = {"rows": rows, "leakage_points": r["leakage"]["leakage_points"],
                     "leakage_balanced_accuracy": r["leakage"]["balanced_accuracy"]}
    a, b = stand_ins[0], stand_ins[1]
    seed_diff = {k: {"decoded": abs(res[a]["rows"][k]["decoded"] - res[b]["rows"][k]["decoded"]),
                     "probe": abs(res[a]["rows"][k]["probe"] - res[b]["rows"][k]["probe"])} for k in res[a]["rows"]}
    one, better, n = separation_rates()
    by_mean = separation_rates_by_mean()
    two = [v["better_of_two"] for v in by_mean.values()]
    return {"stand_ins": res, "seed_differences": seed_diff, "lockbox_controls": LOCKBOX_CONTROLS,
            "separation": {"false_positive_one_arm": one,
                           "false_positive_better_of_two_rank_sum": {"rate": better, "assignments_enumerated": n},
                           "false_positive_by_mean": {"by_distribution": by_mean,
                                                      "better_of_two_min": min(two), "better_of_two_max": max(two),
                                                      "draws": SEP_DRAWS, "seed": ATTAIN_SEED},
                           "false_positive_better_of_two_bound": 2.0 * one,
                           "power_by_gap_in_seed_sd": separation_power(),
                           "power_by_gap_in_seed_sd_better_of_two": separation_power(arms=2),
                           "draws": ATTAIN_DRAWS, "seed": ATTAIN_SEED},
            "sources": srcs, "moveatlas_head_now": PV.moveatlas_head()}


# ---------------------------------------------------------------- stores and trainer
def stores(device, phase_b=True):
    from mdfm.data import body_store as BS
    from mdfm.data import window_store as WS
    from mdfm.factory import synth as SY
    bs = BS.BodyStore(split="train")
    ws = WS.WindowStore() if phase_b else None
    return bs, ws, SY.SynthSampler(bs, device=device)


def trainer(cfg, a, device, run_dir, phase_b=True):
    from mdfm.baselines.trainer import G12Trainer
    bs, ws, synth = stores(device, phase_b)
    cap = LP.MEMORY_CAP_GB if a.memory_cap_gb is None else a.memory_cap_gb
    return G12Trainer(cfg, bs, ws, synth, device, run_dir, prefetch=a.prefetch, memory_cap_gb=cap,
                      echo=lambda *x: print(*x, flush=True))


def write_seconds(path, arm, value):
    """Set budget.seconds_per_step[arm] = value in the YAML text of path (the one line holding the mapping)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = SPS_LINE.search(text)
    if m is None:
        raise ValueError(f"{path}: no line 'seconds_per_step: {{...}}'")
    pairs = {}
    for part in m.group(2).split(","):
        if part.strip():
            k, v = part.split(":", 1)
            pairs[k.strip()] = float(v)
    pairs[arm] = float(value)
    inner = ", ".join(f"{k}: {pairs[k]:.6g}" for k in GC.ARMS if k in pairs)
    text = text[:m.start()] + f"{m.group(1)}{{{inner}}}{m.group(3)}" + text[m.end():]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    GC.load(path)                       # the file still parses and checks
    return pairs


def record_steps(tr):
    """Wrap tr.optimizer_step (an instance attribute, which train/loop.bench calls) so that every call appends
    {loss, grad_norm, skipped, empty} of the step (trainer.last and the acc counters before and after) to the
    returned list, warmup steps included."""
    out = []
    real = tr.optimizer_step

    def step(timed=False):
        s0, e0 = int(tr.acc["skipped"]), int(tr.acc["empty"])
        stats = real(timed=timed)
        last = tr.last or {}
        out.append({"loss": float(last.get("loss", float("nan"))), "grad_norm": float(last.get("grad_norm", float("nan"))),
                    "skipped": int(tr.acc["skipped"]) - s0, "empty": int(tr.acc["empty"]) - e0,
                    "micro_losses": [float(s.sums["per_window"]) / max(int(s.W), 1) for s in stats]})
        return stats
    tr.optimizer_step = step
    return out


def _finite(x):
    import math
    return x is not None and math.isfinite(float(x))


def smoke_check(phases, steps, bound_gb, edge=SMOKE_EDGE):
    """The smoke verdict (module docstring, --smoke). phases: {phase: train/loop.bench result (micro, peak_reserved_gb,
    steps)}; steps: {phase: record_steps entries of that phase}; bound_gb: the run's peak reserved bound. Returns
    {ok, reasons, bound_gb, phases: {phase: {steps_recorded, nonfinite_steps, nonfinite_micro, skipped, empty,
    loss_first, loss_last, peak_reserved_gb}}}."""
    out, reasons = {}, []
    for ph in LP.PHASES:
        r, st = phases.get(ph) or {}, steps.get(ph) or []
        micro = [x for s in st for x in s.get("micro_losses", [])] or [m.get("loss") for m in r.get("micro", [])]
        bad_steps = sum(1 for s in st if not (_finite(s["loss"]) and _finite(s["grad_norm"])))
        bad_micro = sum(1 for x in micro if not _finite(x))
        skipped, empty = sum(s["skipped"] for s in st), sum(s["empty"] for s in st)
        fin = [s["loss"] for s in st if _finite(s["loss"]) and not s["empty"]]
        peak = r.get("peak_reserved_gb")
        out[ph] = {"steps_recorded": len(st), "nonfinite_steps": bad_steps, "nonfinite_micro": bad_micro,
                   "skipped": skipped, "empty": empty,
                   "loss_first": sum(fin[:edge]) / len(fin[:edge]) if fin else None,
                   "loss_last": sum(fin[-edge:]) / len(fin[-edge:]) if fin else None, "peak_reserved_gb": peak}
        if not st:
            reasons.append(f"phase {ph}: no optimizer step recorded")
        if bad_steps or bad_micro:
            reasons.append(f"phase {ph}: {bad_steps} steps and {bad_micro} micro-steps with a non-finite loss or gradient")
        if skipped:
            reasons.append(f"phase {ph}: {skipped} steps skipped")
        if st and empty >= len(st):
            reasons.append(f"phase {ph}: no step holds a masked window")
        if peak is None:
            reasons.append(f"phase {ph}: peak reserved memory not measured (no CUDA device)")
        elif float(peak) > float(bound_gb):
            reasons.append(f"phase {ph}: peak reserved {float(peak):.3f} GB above the bound {float(bound_gb)} GB")
    return {"ok": not reasons, "reasons": reasons, "bound_gb": float(bound_gb), "phases": out}


def bench(cfg, a):
    """--bench, with --smoke and --out (module docstring)."""
    import datetime
    import torch
    from mdfm.train import provenance as PV
    dev = torch.device(a.device)
    sha_before = _sha(a.config)
    c = GC.check(replace(cfg, budget=replace(cfg.budget, mode="steps", steps=max(10 * a.bench, 1000))))
    run_dir = a.run_dir or os.path.join(paths.CACHE, "runs", f"_bench_g12_{cfg.arm}")
    tr = trainer(c, a, dev, run_dir)
    rec = record_steps(tr) if a.smoke else None
    out, raw, steps = {}, {}, {}
    for ph in LP.PHASES:
        n0 = len(rec) if rec is not None else 0
        r = LP.bench(tr, ph, steps=a.bench, timed=False, warmup=BENCH_WARMUP)
        raw[ph] = r
        if rec is not None:
            steps[ph] = rec[n0:]
        out[ph] = {"wall_s": r["wall_s"], "steps": r["steps"], "seconds_per_step": r["wall_s"] / max(r["steps"], 1),
                   "tokens": r["tokens"], "padded": r["padded"], "peak_reserved_gb": r["peak_reserved_gb"],
                   "peak_allocated_gb": r.get("peak_allocated_gb"), "prefetch": r["prefetch"]}
    sa = float(tr.cfg.schedule.share_a)
    sps = sa * out["A"]["seconds_per_step"] + (1.0 - sa) * out["B"]["seconds_per_step"]
    res = {"arm": cfg.arm, "phases": out, "share_a": sa, "seconds_per_step": sps, "device": str(dev),
           "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "torch": torch.__version__,
           "bench_steps": int(a.bench), "warmup": BENCH_WARMUP,
           "memory_cap_gb": LP.MEMORY_CAP_GB if a.memory_cap_gb is None else a.memory_cap_gb,
           "max_reserved_gb": float(tr.cfg.schedule.max_reserved_gb),
           "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script_sha256": _sha(os.path.abspath(__file__)), "config": a.config, "config_sha256_before": sha_before,
           "moveatlas_head": PV.moveatlas_head()}
    if a.smoke:
        res["smoke"] = smoke_check(raw, steps, float(tr.cfg.schedule.max_reserved_gb))
    if a.write_config and (not a.smoke or res["smoke"]["ok"]):
        res["written"] = write_seconds(a.config, cfg.arm, sps)
    res["config_sha256_after"] = _sha(a.config)
    return res


def write_json(path, obj):
    """obj as JSON to path (atomic: a temporary file in the same folder, then os.replace)."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)
    return path


def verify(tr, fallback=None):
    path = tr.checkpoints[-1] if tr.checkpoints else fallback
    if path is None:
        return None
    ck = CK.load(path, cfg=tr.config_dict)
    with open(path + ".sha256", encoding="utf-8") as f:
        sha = f.read().split()[0]
    out = {"path": path, "sha256": sha, "drift": ck["drift"], "step": ck["step"], "phase": ck["sched"]["in_phase"],
           "persons": {s: len(v) for s, v in ck["provenance"]["persons"].items()},
           "compute": ck["sched"].get("compute")}
    tr.log(dict(out, event="g12_verify"))
    return out


def main(argv=None):
    a = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config(a)
    if a.plan:
        print(json.dumps(plan(cfg, a.seeds), indent=1, sort_keys=True))
        return
    if a.attainability:
        print(json.dumps(attainability(), indent=1, sort_keys=True))
        return
    if a.bench:
        res = bench(cfg, a)
        if a.out:
            write_json(a.out, res)
        print(json.dumps(res, indent=1, sort_keys=True, default=str))
        if a.smoke and not res["smoke"]["ok"]:
            print("smoke failed: " + "; ".join(res["smoke"]["reasons"]), file=sys.stderr, flush=True)
            sys.exit(SMOKE_EXIT)
        return
    import torch
    from mdfm.baselines.trainer import ComputeBudgetError
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    run_dir = a.run_dir if a.run_dir is not None or a.resume is None else os.path.dirname(os.path.abspath(a.resume))
    tr = trainer(cfg, a, dev, run_dir)
    print(f"run folder {tr.run_dir}; arm {cfg.arm}; steps A {tr.sched.steps[0]:,}, B {tr.sched.steps[1]:,}; plan "
          f"{json.dumps(tr.plan)}; set up in {time.perf_counter() - t0:.1f} s", flush=True)
    status, stop = 0, None
    try:
        summary = tr.run(max_steps=a.max_steps, resume=a.resume)
    except LP.AlarmError as e:
        status, stop = 3, str(e)
        summary = {"event": "alarm_stop", "phase": tr.sched.in_phase, "step": tr.sched.step}
    except ComputeBudgetError as e:
        status, stop = 4, str(e)
        summary = {"event": "compute_overrun", "phase": tr.sched.in_phase, "step": tr.sched.step}
    res = verify(tr, fallback=a.resume)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "last"}, "verify": res, "stop": stop},
                     indent=1, default=str))
    paths.assert_not_moveatlas(tr.run_dir)
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
