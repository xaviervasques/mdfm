"""Equal compute of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25): the budget read from the body
run logs and the step plan of a baseline run.

reference_budget(paths): per log.jsonl of a body run (train/loop.py), the clock of its optimizer steps as the loop
logs it: the wall_s of its 'log' records (perf_counter between two log records: the steps of the interval with
their sampling, the checkpoints and dev numbers written inside it), one record kept per (phase, step), the last
written, so that the steps a crash made the run repeat after a resume count once; the optimizer steps (the
optimizer_steps of the same records); per phase the median over those records of wall_s / optimizer_steps (the
interval rate, robust to the intervals holding a checkpoint or a dev step); the tokens (the sum of the token
tallies of the last 'phase_end' of each phase); the config SHA-256 of its first 'start' record and the file's
SHA-256. A log without a 'phase_end' of both
phases A and B is refused (a run that did not finish is no reference). The budget is the mean over the logs of
wall_s, steps and tokens: the body runs compared are seeds of one configuration, and their clocks differ by about
as much as a 10 percent band around their mean (pilot v1 and seed 2: reports/weekend_2026-10-01/tasks/J1.md,
output [A]; scripts/train_g12.py --plan, reference.ratios), so one
log alone would pick a seed's clock as the budget.

plan_steps(cfg, base, budget): the optimizer steps of a baseline run by its budget mode (config.py): gpu_seconds,
floor(budget wall_s / seconds_per_step[arm]) (ValueError while seconds_per_step is not measured); reference_steps,
round(budget steps); steps, cfg.budget.steps. With the GPU's measured seconds per step the gpu_seconds plan makes
the run's projected clock the body's; the trainer logs the realized clock against the budget and stops a run whose
clock passes (1 + tolerance) times it (trainer.ComputeBudgetError). The tolerance is a runaway stop, not a validity
band: the realized clock of identical runs varies about as much as a 10 percent band (the two pilot seeds'
reference.ratios and their phases' median_s_per_step in the output of scripts/train_g12.py --plan), so a finished
run is read with its realized clock reported (docs/rules/G12_RULE.md section 5, review of 2026-10-05).
"""
import hashlib
import json
import math
import os
import statistics

from .config import tree_path

PHASES = ("A", "B")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_log(path):
    """The reference numbers of one body run log (module docstring)."""
    p = tree_path(path)
    logs, ends, config = {}, {}, None
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("event")
            if ev == "start" and config is None:
                config = r.get("config_sha256")
            elif ev == "log":
                logs[(r["phase"], int(r["step"]))] = r
            elif ev == "phase_end":
                ends[r["phase"]] = r
    missing = [ph for ph in PHASES if ph not in ends]
    if missing:
        raise ValueError(f"{path}: no phase_end of phase {missing}: the run did not finish")
    phases = {}
    for ph in PHASES:
        recs = [r for (q, _), r in logs.items() if q == ph]
        t = ends[ph]["tallies"]
        rates = [float(r["wall_s"]) / int(r["optimizer_steps"]) for r in recs if int(r["optimizer_steps"]) > 0]
        phases[ph] = {"wall_s": float(sum(float(r["wall_s"]) for r in recs)),
                      "steps": int(sum(int(r["optimizer_steps"]) for r in recs)),
                      "tokens": int(sum(int(v) for v in t["tokens"].values())), "log_records": len(recs),
                      "end_step": int(ends[ph]["step"]),
                      "median_s_per_step": float(statistics.median(rates)) if rates else None}
    return {"path": str(path).replace("\\", "/"), "sha256": sha256_file(p), "config_sha256": config,
            "phases": phases, "wall_s": sum(v["wall_s"] for v in phases.values()),
            "steps": sum(v["steps"] for v in phases.values()), "tokens": sum(v["tokens"] for v in phases.values())}


def reference_budget(paths):
    """{'logs': [read_log of each path], 'wall_s', 'steps', 'tokens': their means, 'hours': wall_s / 3600,
    'spread': (max - min) / mean of the logs' wall_s, 'ratios': each log's wall_s / the mean, 'max_over_min': the
    largest wall_s over the smallest} (module docstring)."""
    logs = [read_log(p) for p in paths]
    if not logs:
        raise ValueError("no reference log")
    n = len(logs)
    walls = [r["wall_s"] for r in logs]
    wall = sum(walls) / n
    out = {"logs": logs, "wall_s": wall, "hours": wall / 3600.0, "steps": sum(r["steps"] for r in logs) / n,
           "tokens": sum(r["tokens"] for r in logs) / n,
           "spread": (max(walls) - min(walls)) / wall if wall > 0 else 0.0,
           "ratios": [w / wall if wall > 0 else None for w in walls],
           "max_over_min": max(walls) / min(walls) if min(walls) > 0 else None}
    return out


def plan_steps(cfg, budget=None):
    """(optimizer steps, how) of a baseline run (module docstring); budget: reference_budget(...) (needed by the
    gpu_seconds and reference_steps modes)."""
    b = cfg.budget
    if b.mode == "steps":
        return int(b.steps), {"mode": "steps", "steps": int(b.steps)}
    if budget is None:
        raise ValueError(f"budget mode {b.mode!r} needs the reference budget")
    if b.mode == "reference_steps":
        n = int(round(float(budget["steps"])))
        return n, {"mode": b.mode, "steps": n, "reference_steps": budget["steps"]}
    s = float(b.seconds_per_step.get(cfg.arm, 0.0))
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"budget mode 'gpu_seconds': seconds_per_step of arm {cfg.arm!r} is not measured "
                         f"(scripts/train_g12.py --bench writes it)")
    n = int(math.floor(float(budget["wall_s"]) / s))
    return n, {"mode": b.mode, "steps": n, "seconds_per_step": s, "budget_wall_s": float(budget["wall_s"]),
               "projected_wall_s": n * s}


def exists(path):
    return os.path.exists(tree_path(path))
