"""S2B: the recipe of run S2b from run S2a, before any S2b number exists (recipe approved by the owner on 2026-10-12,
reports/gpu_queue/decisions.md, entry 2026-10-12 00:24: in phase C the decoder and the body heads frozen, each
stage-2 unit's share of the phase C windows capped): the value of the cap (stage2.window_cap_per_person of
mdfm/train/config.py, enforced by mdfm/data/sampler.py) and the evidence and counts of the freeze (stage2.freeze,
train/loop.py freeze_parameters).

Inputs (read only): the S2a run folder (default cache/runs/stage2_s2a): its log.jsonl (the start event's config
SHA-256, the phase C dev events with the per-unit held-out and train NLL, the alarm) and the schedule tallies of its
phase C checkpoints (ckpt_C_<step>.pt, read with torch.load(weights_only=True, mmap=True): windows and tokens per
budget key and micro-steps; nothing else of a checkpoint is used); the S2a config (configs/stage2_s2a.yaml, whose
phase C steps, micro-steps per step and tokens_per_micro S2b inherits); reports/bench/stage2_budget.json (the
persons of each unit as the phase C sampler counts them, the tokens of one pass over each unit, the stores' MoveAtlas
HEAD). No lockbox, EVAL, CODY or care_pd record, no window store and no GPU.

Derivation (the numbers in summary.md):
1. The alarm unit: the units that loop.unit_nll_rising lists on S2a's phase C dev history at its last read (the
   unit_nll_rise alarm). Its best read: the dev read of its lowest held-out NLL (the earliest on a tie).
2. Its dose at the best read, D: its windows drawn by that step (the schedule tallies of the checkpoints that bracket
   the step, linear between them; exact at a checkpoint step) over its persons. After that read its held-out NLL
   rose while its train NLL fell.
3. The rate that holds a person of any unit to D over the whole phase C of S2b: r = D / M, M = phase C steps x
   micro-steps per step (S2b inherits S2a's), in windows per person per micro-step.
4. The cap c (window_cap_per_person, a share of the phase's windows per person) with c x W(c) = r, W(c) the windows
   per micro-step under the cap, from S2a's per-micro-step windows and tokens at its last checkpoint: every replay
   key as in S2a; each unit min(its S2a windows, c x persons x W) (its token budget is unchanged, so it never draws
   more than in S2a); the tokens the cap removes refilled by the replay's real sources in proportion to the sampler's
   shares of them, at each source's S2a tokens per window; the synthetic windows as S2a's (the real tokens, from which
   the sampler sets the synthetic budget, are S2a's); solved by fixed point for W and bisection for c, rounded down
   to two significant digits: the first estimate (cap_model). The configuration's cap is settled by item 7.
5. At the first estimate: per key the expected windows and tokens per micro-step, the units the cap binds, each unit's expected
   windows per person over S2b's phase C against D, its expected passes over its tokens (stage2_budget.json, passes),
   the replay's share of tokens and windows, the tokens per micro-step and fill, and what fills the windows the cap
   removes: the replay's real sources, in proportion to the sampler's shares of them (stage2_budget.json, shares,
   group replay; data/sampler.py, _cap_and_refill), so that the real tokens, and the synthetic budget that follows
   them, stay S2a's; beside it, why not the other units: their token slack under unit_cap (unit_cap minus their S2a
   token share), the most they could take, printed.
Items 4 and 5 are expectations from S2a's tallies and item 7 a count over the sampler's plans, none a measurement
of S2b's training; the S2b run measures it (the 'log' records' window_share_phase, train/loop.py).
6. The freeze (with --freeze-config, the S2b config): the parameters of each top-level module of the pilot model under
   that config (mdfm.train.config.model_config, CPU, no skeleton) and of each freeze group (loop.freeze_parameters:
   frozen and trainable parameters and tensors), and, as evidence of where phase C moved the start in S2a, the
   relative change of each top-level module's weights between the start checkpoint (the rule's Start) and S2a's last
   checkpoint, ||end - start|| / ||start|| over the start's entries (the rows a migration added left out), for the
   EMA and the raw weights (torch.load(weights_only=True, mmap=True) of the two state dicts only).
7. The sampler's plans (with --simulate N and --freeze-config, the S2b config): N plans of its phase C sampler on the
   stage-1 root and the stage-2 root (plans only: no window read, no render, as scripts/bench_stage1.py --plan-check),
   an empty plan counted and the draws continued: tokens and fill, windows per plan, every key's share of the
   windows and tokens, the replay's and the synthetic shares, and each capped unit's share of the plans' windows
   against its cap (the last one and the largest after a burn-in), and each unit's windows per person over S2b's
   phase C at that rate against D. It reads the store tables, never a record of the lockbox, EVAL, CODY or care_pd.
   The configuration's cap (cap) is the fixed point of the plans: the config's window_cap_per_person equals r / W,
   W the windows per plan of the plans drawn at that cap, rounded down to two significant digits (found by
   iterating from the first estimate; cap is None while the config's value is not that fixed point).
8. The counterfactual of the refill (with --no-refill N and --freeze-config): the plans of item 7 with the sampler's
   refill disabled (the windows the cap removes not replaced), its empty plans counted, written apart from the
   derivation (iterations/0_no_refill.json and .md): why the removed windows must be replaced.

Run (mdfm environment, CPU):
  CUDA_VISIBLE_DEVICES=-1 python scripts/stage2_s2b_recipe.py [--run cache/runs/stage2_s2a]
      [--freeze-config configs/stage2_s2b.yaml] [--write]
Writes reports/bench/stage2_s2b_recipe/results.json, summary.md and _provenance.json with --write (never over an
existing results.json unless --replace).
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

import numpy as np

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

TASK = "S2B"
RUN = os.path.join("cache", "runs", "stage2_s2a")
CONFIG = os.path.join("configs", "stage2_s2a.yaml")
BUDGET = os.path.join("reports", "bench", "stage2_budget.json")
OUT = os.path.join("reports", "bench", "stage2_s2b_recipe")
START = os.path.join("cache", "runs", "phaseA_root_full", "ckpt_B_032044.pt")    # the read rule's Start
CKPT = re.compile(r"^ckpt_C_(\d{6})\.pt$")
DIGITS = 2                       # significant digits of the cap, rounded down
FIXED_ITER, BISECT_ITER = 200, 200


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
                out.append(json.loads(line))
    return out


def dev_history(recs, units):
    """{unit: [(step, held-out NLL or None, train NLL or None)]} of the phase C dev events (the last of each step)."""
    by = {}
    for r in recs:
        if r.get("event") != "dev" or r.get("phase") != "C":
            continue
        us = ((r.get("dev") or {}).get("model") or {}).get("units")
        if us:
            by[int(r["step"])] = us
    out = {u: [] for u in units}
    for s in sorted(by):
        for u in units:
            e = by[s].get(u) or {}
            ho, tr = e.get("held_out"), e.get("train")
            out[u].append((s, None if not ho else ho.get("nll"), None if not tr else tr.get("nll")))
    return out


def tallies(run_dir):
    """{step: {windows {key: n}, tokens {key: n}, micro, sha256 (the .sha256 sidecar's value)}} of the phase C
    checkpoints of a run folder (schedule tallies of phase C only)."""
    import torch
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "ckpt_C_*.pt"))):
        m = CKPT.match(os.path.basename(p))
        if not m:
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        t = ck["sched"]["tallies"]["C"]
        side = p + ".sha256"
        sha = None
        if os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                sha = f.read().split()[0]
        out[int(m.group(1))] = {"windows": {k: int(v) for k, v in t["windows"].items()},
                                "tokens": {k: int(v) for k, v in t["tokens"].items()}, "micro": int(t["micro"]),
                                "sched_step": int(ck["sched"]["step"]), "path": rel(p), "sha256_sidecar": sha}
        del ck
    return out


def windows_at(tal, key, step):
    """(windows of `key` drawn by optimizer step `step`, how): exact at a checkpoint step, else linear between the
    checkpoints that bracket it (ValueError outside them)."""
    steps = sorted(tal)
    if step in tal:
        return float(tal[step]["windows"].get(key, 0)), f"checkpoint {step}"
    lo = [s for s in steps if s < step]
    hi = [s for s in steps if s > step]
    if not hi:
        raise ValueError(f"step {step} after the last checkpoint {steps[-1] if steps else None}")
    a, b = (lo[-1] if lo else 0), hi[0]
    wa = float(tal[a]["windows"].get(key, 0)) if lo else 0.0
    wb = float(tal[b]["windows"].get(key, 0))
    return wa + (wb - wa) * (step - a) / (b - a), f"linear between steps {a} ({wa:.0f}) and {b} ({wb:.0f})"


def model_at(c, base, persons, units, refill):
    """The expected per-micro-step windows and tokens per key under the per-person cap c (module docstring, item 4):
    {windows {key}, tokens {key}, W, bound [units], freed, iterations}. base: {windows, tokens} per micro-step of S2a
    per key ('synthetic' among them); refill: {replay real source: weight}, the sampler's shares of the replay's real
    sources, which take the tokens the cap removes in proportion (data/sampler.py, _cap_and_refill); the real tokens
    are then S2a's, and so is the synthetic budget that follows them."""
    w0, t0 = base["windows"], base["tokens"]
    tpw = {k: t0[k] / w0[k] for k in w0 if w0[k] > 0}
    syn = "synthetic"
    tw = sum(refill.values())
    W = sum(w0.values())
    win, F = dict(w0), 0.0
    for it in range(FIXED_ITER):
        win = {k: w0[k] for k in w0 if k not in units}
        F = 0.0
        for u in units:
            win[u] = min(w0[u], c * persons[u] * W) if c > 0 else w0[u]
            F += (w0[u] - win[u]) * tpw[u]
        for s, wgt in refill.items():
            win[s] = w0[s] + F * wgt / tw / tpw[s]
        W2 = sum(win.values())
        if abs(W2 - W) < 1e-12:
            W = W2
            break
        W = W2
    tok = {k: win[k] * tpw.get(k, 0.0) for k in win}
    bound = [u for u in units if c > 0 and c * persons[u] * W < w0[u] - 1e-9]
    return {"windows": win, "tokens": tok, "W": W, "bound": bound, "freed": F, "iterations": it + 1}


def solve_cap(r, base, persons, units, alarm, refill):
    """The cap c with c x W(c) = r for the alarm units (bisection on (0, 1 / min persons]); the dose c x W is
    increasing in c."""
    lo, hi = 0.0, 1.0 / min(persons[u] for u in alarm)
    f = (lambda c: c * model_at(c, base, persons, units, refill)["W"] - r)
    if f(hi) < 0:
        raise ValueError("the target rate is not reached even without a cap")
    for _ in range(BISECT_ITER):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def round_down(x, digits=DIGITS):
    """x rounded down to `digits` significant digits, as the float of its decimal text (0.00089, not 8.9e-4 times a
    power of ten)."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - (digits - 1)
    return float(f"{math.floor(x / 10 ** e)}e{e}")


def derive(tree=TREE, run=RUN, config=CONFIG, budget=BUDGET):
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    run_dir = os.path.join(tree, run)
    cfg = CF.load(os.path.join(tree, config))
    cfg_sha = CF.sha256(cfg)
    units = tuple(cfg.stage2.units)
    log_path = os.path.join(run_dir, "log.jsonl")
    recs = read_jsonl(log_path)
    starts = [r for r in recs if r.get("event") == "start"]
    if not starts or any(s.get("config_sha256") != cfg_sha for s in starts):
        raise ValueError(f"{run}: the start events are not at the config SHA-256 of {config} ({cfg_sha})")
    with open(os.path.join(tree, budget), encoding="utf-8") as f:
        bud = json.load(f)
    persons = {u: int(bud["hours"][u]["persons"]) for u in units}
    pass_tokens = {u: float(bud["passes"][u]["tokens"]) for u in units}
    hist = dev_history(recs, units)
    rising = LP.unit_nll_rising({u: [(h, t) for _, h, t in v] for u, v in hist.items()})
    alarms = [r for r in recs if r.get("event") == "alarm"]
    if not rising:
        raise ValueError("no unit's dev NLL rises over the last reads: no alarm unit to derive the cap from")
    tal = tallies(run_dir)
    last = max(tal)
    per = {}
    for u in rising:
        reads = [(s, h) for s, h, _ in hist[u] if h is not None]
        best = min(reads, key=lambda x: (x[1], x[0]))
        w, how = windows_at(tal, u, best[0])
        per[u] = {"best_step": best[0], "best_held_out_nll": best[1], "reads": hist[u], "windows_at_best": w,
                  "how": how, "persons": persons[u], "dose": w / persons[u]}
    alarm = min(per, key=lambda u: per[u]["dose"])
    D = per[alarm]["dose"]
    M = int(cfg.stage2.steps) * int(cfg.schedule.micro_per_step)
    r = D / M
    t_last = tal[last]
    micro = t_last["micro"]
    base = {"windows": {k: v / micro for k, v in t_last["windows"].items()},
            "tokens": {k: v / micro for k, v in t_last["tokens"].items()}}
    refill = {k: float(v["after_caps"]) for k, v in bud["shares"].items()
              if v.get("group") == "replay" and k != "synthetic" and k in base["windows"]}
    c_star = solve_cap(r, base, persons, units, list(per), refill)
    c_cfg = round_down(c_star)
    at = model_at(c_cfg, base, persons, units, refill)
    s2a = model_at(0.0, base, persons, units, refill)
    N = int(cfg.mixture.tokens_per_micro)
    replay = [k for k in base["windows"] if k not in units]

    def shares(m):
        T, W = sum(m["tokens"].values()), m["W"]
        return {"tokens_per_micro": T, "fill": T / N, "windows_per_micro": W,
                "replay_token_share": sum(m["tokens"][k] for k in replay) / T,
                "replay_window_share": sum(m["windows"][k] for k in replay) / W,
                "synthetic_token_share": m["tokens"]["synthetic"] / T,
                "window_share": {k: m["windows"][k] / W for k in sorted(m["windows"])},
                "token_share": {k: m["tokens"][k] / T for k in sorted(m["tokens"])}}

    unit_rows = {}
    S2A_STEPS = t_last["sched_step"]
    for u in units:
        unit_rows[u] = {
            "persons": persons[u], "cap_share": min(1.0, c_cfg * persons[u]),
            "s2a_windows_per_micro": base["windows"][u], "s2b_windows_per_micro": at["windows"][u],
            "s2a_window_share": s2a["windows"][u] / s2a["W"], "s2b_window_share": at["windows"][u] / at["W"],
            "s2a_token_share": base["tokens"][u] / sum(base["tokens"].values()),
            "bound": u in at["bound"],
            "s2a_windows_per_person_by_last": t_last["windows"][u] / persons[u],
            "s2b_windows_per_person_whole": at["windows"][u] * M / persons[u],
            "s2b_dose_over_D": at["windows"][u] * M / persons[u] / D,
            "s2a_passes_by_last": t_last["tokens"][u] / pass_tokens[u],
            "s2b_passes_whole": at["tokens"][u] * M / pass_tokens[u]}
    unit_cap = float(cfg.stage2.unit_cap)
    slack = {u: max(0.0, unit_cap - unit_rows[u]["s2a_token_share"]) for u in units if u not in at["bound"]}
    removed_tokens = sum(base["tokens"][u] - at["tokens"][u] for u in units)
    return {
        "task": TASK, "label": "derivation of a pre-registered configuration value from run S2a; no S2b number",
        "run": run.replace("\\", "/"), "config": config.replace("\\", "/"), "config_sha256": cfg_sha,
        "budget": budget.replace("\\", "/"), "budget_data_sha256": bud.get("data_sha256"),
        "store_moveatlas_head": bud.get("store_moveatlas_head"), "units": list(units), "persons": persons,
        "checkpoints": {str(k): {kk: v for kk, v in x.items() if kk not in ("windows", "tokens")}
                        for k, x in sorted(tal.items())},
        "alarm_events": [{"step": a.get("step"), "alarms": a.get("alarms")} for a in alarms],
        "rising_units": rising, "alarm_unit": alarm, "per_alarm_unit": per,
        "dose_D": D, "phase_c_steps": int(cfg.stage2.steps), "micro_per_step": int(cfg.schedule.micro_per_step),
        "phase_c_micro_steps": M, "rate_r": r, "cap_solved": c_star, "cap_model": c_cfg, "cap": None, "digits": DIGITS,
        "s2a_last_step": S2A_STEPS, "s2a_micro_steps_at_last": micro, "tokens_per_micro_cfg": N,
        "unit_cap": unit_cap, "s2a": shares(s2a), "s2b": shares(at), "bound": at["bound"], "units_table": unit_rows,
        "fill_slack_of_free_units": slack, "fill_slack_total": sum(slack.values()), "refill_weights": refill,
        "refill_tokens_per_micro": at["freed"],
        "removed_tokens_per_micro": removed_tokens,
        "removed_token_share_of_s2a": removed_tokens / sum(base["tokens"].values()),
        "inputs": {rel(log_path): sha256_file(log_path), budget.replace("\\", "/"):
                   sha256_file(os.path.join(tree, budget))}}


def module_drift(start_path, end_path):
    """{'ema' | 'model': {top-level module: {entries, values, rel_change}}}: ||end - start|| / ||start|| per top-level
    module of the floating entries of the two state dicts, over the start's shape of each entry (a migrated table's
    added rows left out); an entry absent from the start is counted under added_values."""
    import torch
    a = torch.load(start_path, map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(end_path, map_location="cpu", weights_only=True, mmap=True)
    out = {}
    for key in ("ema", "model"):
        A, B = a[key], b[key]
        agg = {}
        for n, x in B.items():
            if not torch.is_floating_point(x):
                continue
            top = n.split(".")[0]
            g = agg.setdefault(top, {"entries": 0, "values": 0, "d2": 0.0, "s2": 0.0, "added_values": 0})
            if n not in A:
                g["added_values"] += int(x.numel())
                continue
            y = A[n]
            if tuple(y.shape) != tuple(x.shape):
                x = x[tuple(slice(0, s) for s in y.shape)]
            g["entries"] += 1
            g["values"] += int(y.numel())
            g["d2"] += float((x.double() - y.double()).pow(2).sum())
            g["s2"] += float(y.double().pow(2).sum())
        out[key] = {k: {"entries": v["entries"], "values": v["values"], "added_values": v["added_values"],
                        "rel_change": math.sqrt(v["d2"] / v["s2"]) if v["s2"] > 0 else None}
                    for k, v in agg.items()}
    del a, b
    return out


def freeze_evidence(tree, config, start=START, end=None):
    """The freeze of a stage-2 config (module docstring, item 6): {config, config_sha256, groups, counts (the record of
    loop.freeze_parameters without the names), modules {top-level module: parameters}, frozen_names, drift, start,
    end}."""
    from dataclasses import replace
    from mdfm.model import mfm as M
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    cfg = CF.load(os.path.join(tree, config))
    model = M.MFM(replace(CF.model_config(cfg), skeleton=False))
    mods = {}
    for n, p in model.named_parameters():
        mods[n.split(".")[0]] = mods.get(n.split(".")[0], 0) + int(p.numel())
    rec = LP.freeze_parameters(model, tuple(cfg.stage2.freeze)) if tuple(cfg.stage2.freeze) else None
    drift = module_drift(os.path.join(tree, start), os.path.join(tree, end)) if end else None
    return {"config": config.replace("\\", "/"), "config_sha256": CF.sha256(cfg), "groups": list(cfg.stage2.freeze),
            "group_prefixes": {g: list(CF.STAGE2_FREEZE_GROUPS[g]) for g in cfg.stage2.freeze},
            "counts": None if rec is None else {k: v for k, v in rec.items() if k != "names"},
            "frozen_names": None if rec is None else rec["names"], "modules": mods,
            "parameters": int(sum(mods.values())), "drift": drift, "start": start.replace("\\", "/"),
            "end": None if end is None else end.replace("\\", "/")}


def simulate(tree, config, n, refill=True):
    """The phase C plans of a stage-2 config on the stores (module docstring, item 7): n plans of its sampler (seed
    loop.phase_seed, its redraws; plans only, no window read and no render), an EmptyStep counted and the draws
    continued. Returns {plans, empty, redrawn, tokens_mean, tokens_min, fill, windows_per_plan, window_share,
    token_share, replay_token_share, replay_window_share, synthetic_token_share, caps, final_share (each capped
    unit's share of the plans' windows), max_share_after (its largest cumulative share after plan `burn`), burn,
    data_sha256, seconds}."""
    from mdfm.data import body_store as BS
    from mdfm.data import sampler as SM
    from mdfm.data import window_store as WS
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    from mdfm.train import names as NM
    t0 = time.time()
    cfg = CF.load(os.path.join(tree, config))
    bs = BS.BodyStore(split="train")
    ws = SM.Stage2Stores(WS.WindowStore(WS.store_root(), tier="RELEASE", names=NM.STAGE1),
                         WS.WindowStore(WS.store_root(kind=WS.TIER_KIND[cfg.tier]), tier=cfg.tier,
                                        names=CF.name_tables(cfg)))
    sm = SM.MixtureSampler(bs, ws, None, "C", LP.phase_seed(cfg, "C"), cfg, redraws=int(cfg.stage2.redraws))
    if not refill:                       # the counterfactual of item 8: the removed windows not replaced
        sm._fill_pool = lambda *a, **k: 0
    caps = dict(sm.window_caps or {})
    replay = {SM.SYNTHETIC} | set(sm.refill_keys or [k for k in sm.order if sm.store_of.get(k) is ws.replay])
    win, tok, sizes, nwin, empty = {}, {}, [], [], 0
    burn = min(100, max(n // 10, 1))
    max_after = dict.fromkeys(caps, 0.0)
    for k in range(int(n)):
        try:
            p = sm.plan()
        except SM.EmptyStep:
            empty += 1
            continue
        for it in p.items:
            win[it.source] = win.get(it.source, 0) + 1
            tok[it.source] = tok.get(it.source, 0) + int(it.tokens)
        sizes.append(sum(int(it.tokens) for it in p.items))
        nwin.append(len(p.items))
        if k >= burn:
            W = sum(win.values())
            for u in caps:
                max_after[u] = max(max_after[u], win.get(u, 0) / W)
    W, T = sum(win.values()), sum(tok.values())
    N = int(cfg.mixture.tokens_per_micro)
    return {"config": config.replace("\\", "/"), "config_sha256": CF.sha256(cfg), "refill": bool(refill),
            "plans": len(sizes),
            "requested": int(n), "empty": empty, "redraws": int(cfg.stage2.redraws), "redrawn": int(sm.redrawn),
            "tokens_mean": float(np.mean(sizes)) if sizes else None, "tokens_min": int(min(sizes)) if sizes else None,
            "fill": float(np.mean(sizes)) / N if sizes else None,
            "windows_per_plan": float(np.mean(nwin)) if nwin else None,
            "window_share": {k: v / W for k, v in sorted(win.items())} if W else {},
            "token_share": {k: v / T for k, v in sorted(tok.items())} if T else {},
            "replay_token_share": sum(v for k, v in tok.items() if k in replay) / T if T else None,
            "replay_window_share": sum(v for k, v in win.items() if k in replay) / W if W else None,
            "synthetic_token_share": tok.get(SM.SYNTHETIC, 0) / T if T else None,
            "caps": caps, "final_share": {u: win.get(u, 0) / W for u in caps} if W else {},
            "max_share_after": max_after, "burn": burn, "data_sha256": sm.data_sha256(),
            "seconds": time.time() - t0}


def simulate_md(Q):
    if Q is None:
        return []
    L = ["", "## 6. The sampler's plans under the S2b config (plans only, no window read, no render)", "",
         f"{Q['plans']} plans of {Q['requested']} drawn by the phase C sampler of {Q['config']} (configuration SHA-256 "
         f"{Q['config_sha256']}, redraws {Q['redraws']}; store tables of data SHA-256 {Q['data_sha256']}): "
         f"{Q['empty']} empty plans, {Q['redrawn']} draws made again; tokens per plan mean {fmt(Q['tokens_mean'], 1)}, "
         f"minimum {Q['tokens_min']}, fill {fmt(Q['fill'])}; windows per plan {fmt(Q['windows_per_plan'], 2)}; "
         f"replay share of the tokens {fmt(Q['replay_token_share'])}, of the windows {fmt(Q['replay_window_share'])}; "
         f"synthetic share of the tokens {fmt(Q['synthetic_token_share'])}.", "",
         "| unit | cap | share of the plans' windows | largest cumulative share after plan "
         f"{Q['burn']} |", "|---|---|---|---|"]
    for u, c in Q["caps"].items():
        L.append(f"| {u} | {fmt(c)} | {fmt(Q['final_share'].get(u))} | {fmt(Q['max_share_after'].get(u))} |")
    if Q.get("cap_plans_exact") is not None:
        L += ["", f"- The cap that holds a person to D at the plans' {fmt(Q['windows_per_plan'], 2)} windows per "
                  f"micro-step: r / W = {Q['cap_plans_exact']:.6e}, rounded down {Q['cap_plans']!r}; the configuration's "
                  f"cap {Q['cap_config']!r}: fixed point {'yes' if Q['fixed_point'] else 'no'}.",
              "", "| unit | persons | windows per person over S2b's phase C at the plans' rate | over D |", "|---|---|---|---|"]
        for u, x in Q["dose"].items():
            L.append(f"| {u} | {x['persons']} | {fmt(x['windows_per_person'], 1)} | {fmt(x['over_D'], 3)} |")
    return L


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def freeze_md(F):
    if F is None:
        return []
    c = F["counts"] or {}
    L = ["", "## 5. The freeze", "",
         f"Config {F['config']} (configuration SHA-256 {F['config_sha256']}); groups "
         + ", ".join(f"{g} ({', '.join(F['group_prefixes'][g])})" for g in F["groups"]) + ".", ""]
    if c:
        L += [f"- Frozen: {c['frozen_parameters']:,} parameters in {c['frozen_tensors']} tensors ("
              + "; ".join(f"{g} {v['parameters']:,} in {v['tensors']}" for g, v in c["frozen"].items())
              + f"); trainable: {c['trainable_parameters']:,} parameters in {c['trainable_tensors']} tensors, of "
              f"{F['parameters']:,}."]
    L += ["", "| top-level module | parameters | frozen | S2a EMA relative change | S2a raw relative change |",
          "|---|---|---|---|---|"]
    fz = set(n.split(".")[0] for n in (F["frozen_names"] or []))
    d = F.get("drift") or {}
    for m, n in F["modules"].items():
        e = ((d.get("ema") or {}).get(m) or {}).get("rel_change")
        r = ((d.get("model") or {}).get(m) or {}).get("rel_change")
        L.append(f"| {m} | {n:,} | {'yes' if m in fz else 'no'} | {fmt(e)} | {fmt(r)} |")
    if d:
        L += ["", f"Relative change: ||end - start|| / ||start|| per module over the start's entries, start {F['start']}, "
                  f"end {F['end']} (S2a's last checkpoint)."]
    return L


def summary_md(R):
    a = R["per_alarm_unit"][R["alarm_unit"]]
    L = [f"# The recipe of run S2b from run S2a: the window cap and the freeze ({R['label']})", "",
         f"Inputs: run {R['run']} (start events at the config SHA-256 {R['config_sha256']} of {R['config']}), its "
         f"phase C checkpoints' schedule tallies, {R['budget']} (data SHA-256 {R['budget_data_sha256']}; stores built "
         f"at MoveAtlas {R['store_moveatlas_head']}).", "",
         "## 1. The alarm unit and its dose", "",
         f"- Units whose dev NLL rises over the last reads while their train NLL falls (loop.unit_nll_rising): "
         f"{', '.join(R['rising_units'])}; alarm events: {len(R['alarm_events'])}.",
         f"- {R['alarm_unit']}: {a['persons']} persons; held-out NLL by dev read: "
         + ", ".join(f"{s} {fmt(h, 1)}" for s, h, _ in a["reads"] if h is not None)
         + f"; best read at step {a['best_step']}.",
         f"- Windows of the unit by step {a['best_step']}: {a['windows_at_best']:.1f} ({a['how']}); dose D = "
         f"{R['dose_D']:.2f} windows per person.", "",
         "## 2. The cap", "",
         f"- S2b's phase C: {R['phase_c_steps']} steps x {R['micro_per_step']} micro-steps = {R['phase_c_micro_steps']} "
         f"micro-steps; the rate that holds a person to D over it: r = D / {R['phase_c_micro_steps']} = "
         f"{R['rate_r']:.6f} windows per person per micro-step.",
         f"- First estimate, the model of item 4 (W from S2a's tallies): the per-person cap with c x W(c) = r is "
         f"{R['cap_solved']:.6e}, rounded down to {R['digits']} significant digits {R['cap_model']!r}.",
         "- The configuration's value: " + (f"window_cap_per_person = {R['cap']!r}, the fixed point of the sampler's "
                                             f"plans (section 6): rounded down, r / W of the plans drawn at this cap "
                                             f"gives this cap again." if R.get("cap") is not None else
                                             "not settled (no simulation, or no fixed point: section 6)."), "",
         "## 3. Expected effect at the first estimate (from S2a's tallies at step "
         f"{R['s2a_last_step']}, {R['s2a_micro_steps_at_last']} micro-steps; not a measurement of S2b)", "",
         "| unit | persons | cap share | S2a windows per micro-step | S2b windows per micro-step | S2a window share | "
         "S2b window share | bound | S2b windows per person, whole phase C | over D | S2a passes by its last step | "
         "S2b passes, whole phase C |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for u, x in R["units_table"].items():
        L.append(f"| {u} | {x['persons']} | {fmt(x['cap_share'])} | {fmt(x['s2a_windows_per_micro'], 2)} | "
                 f"{fmt(x['s2b_windows_per_micro'], 2)} | {fmt(x['s2a_window_share'])} | {fmt(x['s2b_window_share'])} | "
                 f"{'yes' if x['bound'] else 'no'} | {fmt(x['s2b_windows_per_person_whole'], 1)} | "
                 f"{fmt(x['s2b_dose_over_D'], 3)} | {fmt(x['s2a_passes_by_last'], 1)} | {fmt(x['s2b_passes_whole'], 1)} |")
    s, b = R["s2a"], R["s2b"]
    L += ["", "| per micro-step | S2a | S2b, expected |", "|---|---|---|",
          f"| windows | {fmt(s['windows_per_micro'], 2)} | {fmt(b['windows_per_micro'], 2)} |",
          f"| tokens | {fmt(s['tokens_per_micro'], 1)} | {fmt(b['tokens_per_micro'], 1)} |",
          f"| fill of tokens_per_micro {R['tokens_per_micro_cfg']} | {fmt(s['fill'])} | {fmt(b['fill'])} |",
          f"| replay share of the tokens | {fmt(s['replay_token_share'])} | {fmt(b['replay_token_share'])} |",
          f"| replay share of the windows | {fmt(s['replay_window_share'])} | {fmt(b['replay_window_share'])} |",
          f"| synthetic share of the tokens | {fmt(s['synthetic_token_share'])} | {fmt(b['synthetic_token_share'])} |",
          "", "## 4. What fills the windows the cap removes", "",
          f"- The replay's real sources: the cap removes {fmt(R['removed_tokens_per_micro'], 1)} tokens per micro-step "
          f"({fmt(R['removed_token_share_of_s2a'])} of S2a's), and the sampler draws them again from the replay's "
          "real sources, each draw's source picked in proportion to the sampler's shares ("
          + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(R["refill_weights"].items())) + "); the synthetic budget "
          "follows the real tokens as in S2a, so the synthetic share keeps its bound.",
          f"- Not the other units: they sit near unit_cap {R['unit_cap']} of the tokens, so they could take at most "
          f"{fmt(R['fill_slack_total'])} of a micro-step's tokens ("
          + ", ".join(f"{u} {fmt(v)}" for u, v in R["fill_slack_of_free_units"].items()) + ").",
          "- Not nothing: a micro-step short of the removed tokens leaves the other units above unit_cap of its "
          "smaller total, and the sampler's caps then drop windows in a cascade (data/sampler.py docstring, the window "
          "cap; the sampler without the refill: iterations/0_no_refill.md)."]
    L += freeze_md(R.get("freeze"))
    L += simulate_md(R.get("simulation"))
    return "\n".join(L) + "\n"


def write(R, out_dir, replace=False):
    from mdfm.gates import common
    if os.path.exists(os.path.join(out_dir, "results.json")) and not replace:
        raise FileExistsError(os.path.join(out_dir, "results.json"))
    text = summary_md(R)
    common.assert_no_em_dash(text, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    common.write_text(os.path.join(out_dir, "results.json"), json.dumps(R, indent=1, allow_nan=False) + "\n")
    common.write_text(os.path.join(out_dir, "summary.md"), text)
    stamp = None
    try:
        spec = importlib.util.spec_from_file_location("_s2b_recipe_stamp", os.path.join(HERE, "moveatlas_stamp.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stamp = {"head": mod.stamp().get("head")}
    except Exception as e:  # noqa: BLE001  (the stamp is reported; the derivation reads no MoveAtlas file)
        stamp = {"error": f"{type(e).__name__}: {e}"}
    prov = {"script": "scripts/stage2_s2b_recipe.py", "task": TASK, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "moveatlas_stamp": stamp,
            "store_moveatlas_head": R.get("store_moveatlas_head"),
            "code_sha256": common.code_sha256([os.path.abspath(__file__)]), "inputs_sha256": R["inputs"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out_dir, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default=RUN)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--budget", default=BUDGET)
    p.add_argument("--out", default=OUT)
    p.add_argument("--freeze-config", default=None, help="the S2b config whose freeze is counted (item 6)")
    p.add_argument("--start", default=START, help="the start checkpoint of the drift (the read rule's Start)")
    p.add_argument("--simulate", type=int, default=0, help="plans of the --freeze-config sampler on the stores (item 7)")
    p.add_argument("--no-refill", type=int, default=0,
                   help="item 8 alone: plans of the --freeze-config sampler without the refill; with --write, written "
                        "to iterations/0_no_refill.json and .md (results.json untouched)")
    p.add_argument("--write", action="store_true")
    p.add_argument("--replace", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.no_refill:
        if not a.freeze_config:
            raise SystemExit("--no-refill reads the --freeze-config sampler")
        Q = simulate(TREE, a.freeze_config, a.no_refill, refill=False)
        text = "\n".join(["# The S2b sampler without the refill (task S2B, item 8 of scripts/stage2_s2b_recipe.py)"]
                         + simulate_md(Q)[1:]) + "\n"
        print(text)
        if a.write:
            from mdfm.gates import common
            out = os.path.join(TREE, a.out, "iterations")
            common.write_text(os.path.join(out, "0_no_refill.json"), json.dumps(Q, indent=1, allow_nan=False) + "\n")
            common.write_text(os.path.join(out, "0_no_refill.md"), text)
        return 0
    R = derive(TREE, a.run, a.config, a.budget)
    if a.freeze_config:
        last = max(int(k) for k in R["checkpoints"])
        R["freeze"] = freeze_evidence(TREE, a.freeze_config, a.start, R["checkpoints"][str(last)]["path"])
        for p_ in (a.start, R["checkpoints"][str(last)]["path"]):
            side = os.path.join(TREE, p_) + ".sha256"
            if os.path.exists(side):
                with open(side, encoding="utf-8") as f:
                    R["inputs"][p_.replace("\\", "/") + " (sidecar)"] = f.read().split()[0]
    if a.simulate:
        if not a.freeze_config:
            raise SystemExit("--simulate reads the --freeze-config sampler")
        Q = simulate(TREE, a.freeze_config, a.simulate)
        from mdfm.train import config as CF
        c_cfg = float(CF.load(os.path.join(TREE, a.freeze_config)).stage2.window_cap_per_person)
        W, M, D = Q["windows_per_plan"], R["phase_c_micro_steps"], R["dose_D"]
        Q["cap_plans_exact"] = R["rate_r"] / W
        Q["cap_plans"] = round_down(Q["cap_plans_exact"])
        Q["cap_config"] = c_cfg
        Q["fixed_point"] = bool(math.isclose(Q["cap_plans"], c_cfg, rel_tol=1e-9))
        Q["dose"] = {u: {"persons": R["persons"][u],
                         "windows_per_person": Q["final_share"][u] * W * M / R["persons"][u],
                         "over_D": Q["final_share"][u] * W * M / R["persons"][u] / D} for u in Q["caps"]}
        R["simulation"] = Q
        R["cap"] = c_cfg if Q["fixed_point"] else None
    text = summary_md(R)
    print(text)
    if a.write:
        write(R, os.path.join(TREE, a.out), a.replace)
        print(f"written {a.out.replace(os.sep, '/')}/results.json, summary.md, _provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
