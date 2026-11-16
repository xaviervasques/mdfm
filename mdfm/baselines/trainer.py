"""Training loop of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25): G12Trainer, a subclass of
train/loop.Trainer.

Kept from the stage-1 loop, unchanged: the phases (A synthetic only, then B the mixture, share_a of the optimizer
steps in phase A), each phase's data.sampler.MixtureSampler seeded seed + its index, with its guard (provenance.
assert_clean with the held-out persons: a lockbox, EVAL, EXCLUDED_V1, CODY, canary or held-out person refuses the
micro-step before any forward), the prefetch thread, the token tallies and the cumulative synthetic cap
(check_synthetic_cap), the learning-rate schedule (Schedule: warmup, re-warmup, cosine), AdamW with its parameter
groups, the gradient clip, the model EMA (make_ema and ema_update leave the JEPA module out), the memory bound, the
non-finite rule (MAX_NONFINITE), the checkpoint writer and reader (train/checkpoint.py: person lists and their
hashes, the lockbox and records hashes, the state_dict audit, a run folder only under cache/ or runs/), the
provenance record, the alarms (the JEPA collapse alarm of loop.ALARMS for the jepa arm), run() and bench().

Changed: the data of a micro-step is data.collate_tokens (the token half of collate, CODY refused, labels refused
by losses.assert_label_free) through data.produce; the masks of every micro-step of an optimizer step are drawn
first (model.micro_masks from the micro-step seed), then each micro-step (micro_step) runs model.forward of the
arm and the backward of sum_w term_w / n, n the windows with masked patches in the optimizer step (all
micro-steps sampled first, as the stage-1 loop does), so that an
optimizer step minimizes the mean term over its masked windows; after an applied step, the JEPA target follows the
online tokenizer and encoder (jepa.ema_decay over the whole run: the global step over the run's steps). An
optimizer step whose micro-steps hold no masked window applies nothing (logged as empty). No dev numbers (the G12
readouts are probes read after training, mdfm/eval/probes.py), no health record, no refinement.

Configuration stored and hashed by every checkpoint and the provenance: config.combined(g12, run) (the G12
configuration and the derived run configuration); the base configuration's path and SHA-256, the budget and the
step plan go to the 'start' record and to the provenance's extra.

Equal compute (compute.py). The run's clock is the sum of the wall_s of its 'log' records, the quantity
compute.reference_budget reads from the body logs; G12Schedule keeps it (with the tokens and optimizer steps)
in the checkpoints, so that a resumed run continues it (the interval a crash cut is not counted, as in the
reference). Each 'log' record carries compute (the clock, tokens and steps so far, the budget and their ratio);
in mode gpu_seconds a ratio above 1 + tolerance logs 'compute_overrun', writes a checkpoint and raises
ComputeBudgetError (the runaway stop). run() ends with a 'compute' record (the realized clock against the budget,
within_tolerance: whether the ratio lies within 1 +/- tolerance; reported, a finished run is not discarded for it).

Init-from is refused: a baseline starts from its own initialization (config.py: the body's initial tokenizer and
encoder under the same seed), never from a body checkpoint.
"""
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..model import jepa as JP
from ..train import checkpoint as CK
from ..train import config as CF
from ..train import loop as LP
from ..train import provenance as PV
from . import compute as CP
from . import config as GC
from . import data as BD
from . import model as BM

FORMAT = "mdfm-g12/1"
PARTS = ("sample", "render", "collate", "transfer", "forward", "backward", "optimizer")


class ComputeBudgetError(RuntimeError):
    """The run's clock above (1 + tolerance) times the budget (module docstring)."""


class G12Schedule(LP.Schedule):
    """train/loop.Schedule with the run's compute so far: {wall_s, tokens, steps} (module docstring)."""

    def __init__(self, sc, steps):
        super().__init__(sc, steps)
        self.compute = {"wall_s": 0.0, "tokens": 0, "steps": 0}

    def state_dict(self):
        st = super().state_dict()
        st["compute"] = dict(self.compute)
        return st

    def load_state_dict(self, state):
        super().load_state_dict(state)
        c = state.get("compute") or {}
        self.compute = {"wall_s": float(c.get("wall_s", 0.0)), "tokens": int(c.get("tokens", 0)),
                        "steps": int(c.get("steps", 0))}


class Clock:
    """Wall seconds per part (PARTS); sync: CUDA synchronized before each reading."""

    def __init__(self, device, sync=False):
        self.device = torch.device(device)
        self.sync = bool(sync) and self.device.type == "cuda"
        self.s = dict.fromkeys(PARTS, 0.0)
        self.t = None

    def _sync(self):
        if self.sync:
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.t = time.perf_counter()

    def lap(self, part):
        self._sync()
        t = time.perf_counter()
        if self.t is not None:
            self.s[part] += t - self.t
        self.t = t

    def add(self, part, seconds):
        self.s[part] += float(seconds)


@dataclass
class StepStats:
    """One micro-step: W windows, terms (windows with a term), loss_sum (sum of their terms), masked patches,
    patches of the masked windows, hidden tokens, scored queries, rows passed, useful and padded tokens, the
    bucket row length, finite, the time split, the JEPA spreads (None otherwise); sums {'per_window': loss_sum}
    (the field train/loop.bench reads)."""
    W: int
    terms: int
    loss_sum: float
    masked: int
    patches: int
    hidden: int
    queries: int
    rows: int
    useful: int
    padded: int
    row_len: int
    finite: bool = True
    time: dict = field(default_factory=dict)
    std_target: float = None
    std_pred: float = None

    @property
    def sums(self):
        return {"per_window": self.loss_sum}


def micro_step(model, ts, patches, *, device, amp=None, n_terms=None, recompute=False, backward=True, timer=None):
    """One micro-step of a TokenStep with its masked patches (module docstring); returns StepStats."""
    dev = torch.device(device)
    amp = LP.default_amp(dev) if amp is None else amp
    timer = Clock(dev) if timer is None else timer
    if timer.t is None:
        timer.start()
    tokens = ts.tokens.to(dev)
    timer.lap("transfer")
    with LP.amp_context(dev, amp):
        out = BM.forward(model, tokens, patches, recompute)
    timer.lap("forward")
    n = int(out.has.sum()) if n_terms is None else int(n_terms)
    loss_sum = out.term.sum()
    if backward and out.queries and loss_sum.requires_grad:
        (loss_sum / max(n, 1)).backward()
        timer.lap("backward")
    ls = float(loss_sum.detach())
    return StepStats(W=ts.n_windows, terms=int(out.has.sum()), loss_sum=ls, masked=out.masked, patches=out.patches,
                     hidden=out.hidden, queries=out.queries, rows=out.rows, useful=int(ts.useful), padded=int(ts.padded),
                     row_len=int(ts.row_len), finite=math.isfinite(ls), time=dict(timer.s),
                     std_target=out.std_target, std_pred=out.std_pred)


class G12Trainer(LP.Trainer):
    """The G12 training loop (module docstring). g12: a config.G12Config; body_store, window_store, synth, device,
    run_dir, amp, prefetch, echo, check_reports, layouts, memory_cap_gb: as train/loop.Trainer; base: the base
    RunConfig (default config.base_config(g12), pinned by its SHA-256); budget: compute.reference_budget(...) (default
    read from g12.budget.reference_logs unless the mode is 'steps')."""

    def __init__(self, g12, body_store, window_store=None, synth=None, device=None, run_dir=None, *, base=None,
                 budget=None, amp=None, prefetch=0, echo=print, check_reports=True, layouts=None,
                 memory_cap_gb=LP.MEMORY_CAP_GB):
        g12 = GC.check(g12)
        if base is None:
            base, base_sha = GC.base_config(g12)
        else:
            base_sha = CF.sha256(base)
        if budget is None and g12.budget.mode != "steps":
            budget = CP.reference_budget(g12.budget.reference_logs)
        total, how = CP.plan_steps(g12, budget)
        run = GC.run_config(g12, base, total)
        self.g12, self.base_sha, self.budget, self.plan = g12, base_sha, budget, how
        self.rule = g12.mask.rule()
        torch.manual_seed(int(run.seed))
        model = BM.G12Model.of(g12.arm, run.model.mfm_config(), g12.head)
        super().__init__(run, body_store, window_store, synth, device, run_dir, amp=amp, prefetch=prefetch,
                         dev_hook=None, echo=echo, check_reports=check_reports, layouts=layouts, model=model,
                         memory_cap_gb=memory_cap_gb, decoder_checkpoint=False)
        self.sched = G12Schedule(run.schedule, CF.step_counts(run))
        self.rows = tuple(int(r) for r in run.mixture.rows)
        self.config_dict = GC.combined(g12, run)
        self.config_sha256 = GC.sha256(self.config_dict)
        jm = self.model.jepa
        if jm is not None:
            jm.sync(self.model.tokenizer, self.model.encoder)

    # ------------------------------------------------------------ data
    def _start_sampler(self, phase, state=None):
        self._close_prefetch()
        self.sampler = self._make_sampler(phase)
        self.sampler_phase = phase
        if state is not None:
            self.sampler.load_state_dict(state)
        self.sampler_snap = LP.snapshot(self.sampler)
        if self.prefetch > 0:
            self._pre = LP.Prefetcher(lambda: BD.produce(self.sampler, self.rows), self.prefetch)

    def next_item(self):
        return self._pre.get() if self._pre is not None else BD.produce(self.sampler, self.rows)

    # ------------------------------------------------------------ steps
    @property
    def total_steps(self):
        return int(sum(self.sched.steps))

    def global_step(self):
        """Optimizer steps of the run before the current one (phase A's count when in phase B)."""
        return (self.sched.steps[0] if self.sched.in_phase == LP.PHASES[1] else 0) + int(self.sched.step)

    def masks(self, item):
        """Masked patches (W, P_max) of an Item's micro-step (model.micro_masks with the micro-step seed)."""
        return BM.micro_masks(item.mb, item.step.seed, self.rule, self.g12.mask.max_redraws)[1]

    def optimizer_step(self, timed=False):
        """One optimizer step (module docstring); returns the list of StepStats."""
        ph, sc = self.sched.in_phase, self.cfg.schedule
        items = [self.next_item() for _ in range(int(sc.micro_per_step))]
        pms = [self.masks(it) for it in items]
        n_terms = sum(int(pm.any(1).sum()) if pm.size else 0 for pm in pms)
        lr = self.sched.apply(self.opt)
        self.model.train()
        stats = []
        for it, pm in zip(items, pms):
            clock = Clock(self.device, sync=timed)
            for k in ("sample", "render", "collate"):
                clock.add(k, it.host[k])
            clock.start()
            stats.append(micro_step(self.model, it.mb, pm, device=self.device, amp=self.amp, n_terms=n_terms,
                                    recompute=bool(self.g12.recompute), timer=clock))
        jm = self.model.jepa
        if jm is not None:
            for st in stats:
                if jm.set_references(st.std_target, st.std_pred):
                    break
            if self.acc["first"] is None and stats:
                self.acc["first"] = {"std_target": stats[0].std_target, "std_pred": stats[0].std_pred}
        t0 = time.perf_counter()
        if timed and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
        params = [p for p in self.model.parameters() if p.grad is not None]
        empty = not params
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(sc.clip))) if params else 0.0
        ok = math.isfinite(gn) and all(s.finite for s in stats)
        if ok and not empty:
            self.opt.step()
            LP.ema_update(self.ema, self.model, float(sc.ema_decay))
            if jm is not None:
                m = JP.ema_decay(self.global_step(), self.total_steps, self.g12.jepa.ema_start, self.g12.jepa.ema_end)
                jm.update_target(self.model.tokenizer, self.model.encoder, m)
                self.acc["decay"] = m
            self.nonfinite = 0
        elif ok:
            self.acc["empty"] += 1
            self.log({"event": "empty", "phase": ph, "step": self.sched.step})
        else:
            self.nonfinite += 1
            self.acc["skipped"] += 1
            self.log({"event": "nonfinite", "phase": ph, "step": self.sched.step, "grad_norm": gn})
        self.opt.zero_grad(set_to_none=True)
        if timed and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_opt = time.perf_counter() - t0
        for st in stats:
            st.time["optimizer"] = st.time.get("optimizer", 0.0) + t_opt / len(stats)
        for it, st in zip(items, stats):
            self.sched.count(ph, it.step, it.mb)
            self._accumulate(st, it)
            if timed:
                self.acc["timed"] += 1
                for k in PARTS:
                    self.acc["time"][k] += st.time.get(k, 0.0)
        if ok and not empty:
            self.acc["grad"].append(gn)
        self.acc["steps"] += 1
        self.sampler_snap = items[-1].state
        self.sched.step += 1
        self.last = {"lr": lr, "grad_norm": gn, "loss": sum(s.loss_sum for s in stats) / max(n_terms, 1),
                     "terms": n_terms, "refined": False}
        self.check_memory()
        if self.nonfinite >= LP.MAX_NONFINITE:
            raise FloatingPointError(f"{self.nonfinite} optimizer steps in a row with a non-finite loss or gradient")
        return stats

    # ------------------------------------------------------------ logging
    def _reset_interval(self):
        self.acc = {"W": 0, "terms": 0, "loss_sum": 0.0, "masked": 0, "patches": 0, "hidden": 0, "queries": 0,
                    "rows": 0, "useful": 0, "padded": 0, "micro": 0, "steps": 0, "timed": 0,
                    "time": dict.fromkeys(PARTS, 0.0), "grad": [], "skipped": 0, "empty": 0, "tokens": {},
                    "buckets": {}, "first": None, "decay": None, "health": None, "t0": time.perf_counter()}

    def _accumulate(self, st, item):
        a = self.acc
        for k in ("W", "terms", "masked", "patches", "hidden", "queries", "rows", "useful", "padded"):
            a[k] += getattr(st, k)
        a["loss_sum"] += st.loss_sum
        a["micro"] += 1
        a["buckets"][st.row_len] = a["buckets"].get(st.row_len, 0) + 1
        for s, n in item.step.tokens.items():
            a["tokens"][s] = a["tokens"].get(s, 0) + int(n)

    def budget_summary(self):
        """The reference budget without the per-phase detail, or None."""
        b = self.budget
        if b is None:
            return None
        return {"wall_s": b["wall_s"], "hours": b["hours"], "steps": b["steps"], "tokens": b["tokens"],
                "spread": b["spread"], "logs": [{k: r[k] for k in ("path", "sha256", "config_sha256", "wall_s", "steps",
                                                                     "tokens")} for r in b["logs"]]}

    def compute_record(self):
        """The run's compute so far against the budget (module docstring)."""
        c = self.sched.compute
        b = None if self.budget is None else float(self.budget["wall_s"])
        tol = float(self.g12.budget.tolerance)
        ratio = c["wall_s"] / b if b else None
        return {"mode": self.g12.budget.mode, "wall_s": c["wall_s"], "tokens": c["tokens"], "steps": c["steps"],
                "planned_steps": self.total_steps, "budget_wall_s": b, "ratio": ratio, "tolerance": tol}

    def interval_record(self):
        """The 'log' record of the interval since the last one; adds the interval to the run's compute."""
        a = self.acc
        wall = time.perf_counter() - a["t0"]
        c = self.sched.compute
        c["wall_s"] += wall
        c["tokens"] += int(a["useful"])
        c["steps"] += int(a["steps"])
        tok = sum(a["tokens"].values())
        rec = {"event": "log", "phase": self.sched.in_phase, "step": self.sched.step, "arm": self.g12.arm,
               "lr": self.sched.lr(step=max(self.sched.step - 1, 0)),
               "loss": a["loss_sum"] / a["terms"] if a["terms"] else float("nan"),
               "windows": a["W"], "term_windows": a["terms"],
               "masked_fraction": a["masked"] / a["patches"] if a["patches"] else None,
               "hidden_tokens": a["hidden"], "queries": a["queries"], "rows": a["rows"], "micro_steps": a["micro"],
               "optimizer_steps": a["steps"], "wall_s": wall,
               "tokens_per_s": a["useful"] / wall if wall > 0 else None,
               "padded_tokens_per_s": a["padded"] / wall if wall > 0 else None,
               "tokens_per_micro": a["useful"] / max(a["micro"], 1), "fill": a["useful"] / max(a["padded"], 1),
               "buckets": {str(k): v for k, v in sorted(a["buckets"].items())},
               "time_split_s": {k: v / a["timed"] for k, v in a["time"].items()} if a["timed"] else None,
               "memory_gb": self.memory(),
               "grad_norm": {"mean": float(np.mean(a["grad"])), "max": float(np.max(a["grad"]))} if a["grad"] else None,
               "skipped": a["skipped"], "empty": a["empty"],
               "token_share": {s: n / tok for s, n in sorted(a["tokens"].items())} if tok else {},
               "cumulative_synthetic_share": self.sched.synthetic_share(),
               "projected_synthetic_share": self.projected_synthetic_share(), "health": None,
               "compute": self.compute_record()}
        jm = self.model.jepa
        if jm is not None:
            first = a["first"] or {}
            st, sp = first.get("std_target"), first.get("std_pred")
            rt, rp = jm.references()
            ratio = None
            if None not in (st, sp, rt, rp) and rt > 0 and rp > 0:
                ratio = min(st / rt, sp / rp)
            rec["jepa"] = {"std_target": st, "std_pred": sp, "ref_std_target": rt, "ref_std_pred": rp,
                           "collapse_ratio": ratio, "ema_decay": a["decay"]}
        return rec

    def check_alarms(self, rec, source):
        """train/loop.Trainer.check_alarms (the JEPA collapse alarm), then, at a log of a gpu_seconds run, the
        compute overrun (module docstring)."""
        hits = super().check_alarms(rec, source)
        if source != "log" or self.g12.budget.mode != "gpu_seconds":
            return hits
        c = (rec or {}).get("compute") or {}
        r = c.get("ratio")
        if r is not None and r > 1.0 + float(self.g12.budget.tolerance):
            self.log(dict(c, event="compute_overrun", phase=self.sched.in_phase, step=self.sched.step))
            path = os.path.join(self.run_dir, f"ckpt_{self.sched.in_phase}_{self.sched.step:06d}.pt")
            if not self.checkpoints or os.path.abspath(self.checkpoints[-1]) != os.path.abspath(path):
                self.checkpoint()
            raise ComputeBudgetError(f"the run's clock {c['wall_s']:.0f} s is {r:.3f} times the budget "
                                     f"{c['budget_wall_s']:.0f} s (tolerance {c['tolerance']})")
        return hits

    def start_record(self):
        rec = super().start_record()
        rec["g12"] = {"arm": self.g12.arm, "config_sha256": self.config_sha256, "base_config": self.g12.base_config,
                      "base_config_sha256": self.base_sha, "mask": CF.to_dict(self.g12.mask),
                      "head": CF.to_dict(self.g12.head), "budget": self.budget_summary(), "plan": dict(self.plan),
                      "steps": list(self.sched.steps), "parameters": self.model.parameter_counts()}
        return rec

    # ------------------------------------------------------------ checkpoints
    def provenance(self):
        ws = self.window_store if self.sched.in_phase == LP.PHASES[1] else None
        extra = {"loop": FORMAT, "arm": self.g12.arm, "optimizer": self.opt_kind, "phase": self.sched.in_phase,
                 "step": self.sched.step, "base_config": self.g12.base_config, "base_config_sha256": self.base_sha,
                 "g12_config_sha256": self.config_sha256, "budget": self.budget_summary(), "plan": dict(self.plan),
                 "compute": dict(self.sched.compute)}
        return PV.record(self.config_dict, self.body_store, ws, extra=extra)

    def checkpoint(self):
        """Write <run>/ckpt_<phase>_<step>.pt under the combined configuration; returns (path, SHA-256)."""
        ph, s = self.sched.in_phase, self.sched.step
        path = os.path.join(self.run_dir, f"ckpt_{ph}_{s:06d}.pt")
        sha = CK.save(path, self.model, self.ema, self.opt, self.sched, LP.sampler_state(self.sampler, self.sampler_snap),
                      CK.rng_state(), self.config_dict, self.provenance(), step=s)
        self.checkpoints.append(path)
        self.log({"event": "checkpoint", "phase": ph, "step": s, "path": path, "sha256": sha})
        self.echo(f"[{ph} {s:>6}] checkpoint {path}")
        return path, sha

    def resume(self, path):
        """Restore a checkpoint of this run's combined configuration; returns the checkpoint dict."""
        ck = CK.load(path, map_location="cpu", cfg=self.config_dict)
        if "config_sha256" in ck["drift"]:
            raise ValueError(f"{path} was written under another configuration")
        CK.restore(ck, self.model, self.ema, self.opt, self.sched, rng=True)
        if self.sched.step < self.sched.n():
            self._start_sampler(self.sched.in_phase, ck["sampler"])
        self.log({"event": "resume", "path": os.path.abspath(path), "phase": self.sched.in_phase,
                  "step": self.sched.step, "drift": ck["drift"], "compute": dict(self.sched.compute)})
        return ck

    def init_from(self, path):
        raise ValueError("a G12 baseline starts from its own initialization, never from a body checkpoint")

    def run(self, phases=LP.PHASES, max_steps=None, resume=None, dev_at_start=False, init_from=None):
        """train/loop.Trainer.run, then the 'compute' record (module docstring)."""
        if init_from is not None:
            self.init_from(init_from)
        out = super().run(phases, max_steps=max_steps, resume=resume, dev_at_start=False)
        rec = self.compute_record()
        done = self.sched.in_phase == LP.PHASES[-1] and self.sched.step >= self.sched.n(LP.PHASES[-1])
        within = None
        if rec["ratio"] is not None and done:
            within = abs(rec["ratio"] - 1.0) <= rec["tolerance"]
        rec.update(event="compute", finished=done, within_tolerance=within)
        self.log(rec)
        out["compute"] = rec
        return out
