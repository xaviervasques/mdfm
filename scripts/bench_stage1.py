"""B4.18: end-to-end bench of the stage-1 pilot on the RTX 5080 (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md,
sections 5 and 7; the loop is mdfm/train/loop.py). Writes reports/bench/stage1_pilot.md and
reports/bench/stage1_pilot.json, and the measured values into configs/stage1_pilot.yaml (bucket rows, micro-batch,
token budget with its step counts, refinement cadence N, status line); --no-write-config leaves the file alone.

The pilot model (configs/stage1_pilot.yaml), the AMASS body store (train split) through the GPU renderer, the
window store of the real sources, bf16 autocast, the AdamW that loop.build_optimizer picks, the allocator cap of
loop.MEMORY_CAP_GB with PYTORCH_CUDA_ALLOC_CONF = loop.ALLOC_CONF. The weights train during the bench (the costs
do not depend on them); no checkpoint is written and nothing reaches a run folder but the bench's empty one.

1. Window-length histogram: the planned token counts (sampler plans, no render; a synthetic window's count is
   the layout's upper bound) of the windows of PLAN_STEPS micro-steps of each phase, and the longest window of
   each micro-step (the bucket of a micro-step is the smallest row holding its longest window).
2. Bucket rows: for each candidate largest row of ROW_CANDIDATES (all at least the longest window a layout can
   make), the first-fit decreasing packing of ENC_PLANS planned micro-steps of each phase and the forward and
   backward time of the pilot encoder (bf16, block-diagonal masks of the packed windows) on them; the candidate
   with the least encoder time per useful token becomes the largest row, the shorter rows SMALL_ROWS stay.
3. Micro-batch: (tokens per micro-step, micro-steps per optimizer step) of MICRO_CANDIDATES, 32,768 tokens per
   optimizer step each; timed phase A steps of each; the candidate with the most useful tokens per second whose
   steps fit under the allocator cap (no out-of-memory error) is kept.
4. Phase A: timed steps (CUDA synchronized between the parts: the time split per bucket, peak memory per
   bucket), then untimed steps without and with prefetch (throughput); the faster prefetch is kept.
5. Phase B: timed steps unrefined and refined (refine_every 1), untimed steps at refine_every 0 and 1 with the
   kept prefetch; N is the smallest integer with (t1 - t0) / N <= REFINE_OVERHEAD x t0, t0 and t1 the wall
   seconds of an unrefined and a refined optimizer step; then untimed steps at N.
6. Token budget: the largest tokens_total of TOKEN_GRID (1.0 to 1.5 billion, the range of spec section 5) whose
   run takes at most WALL_BUDGET_H hours at the measured seconds per optimizer step of phase A and of phase B at
   N (with prefetch), 1.0 billion when none does; the step counts are config.step_counts of that budget (at full
   micro-steps: phase B micro-steps hold fewer tokens, so the run consumes fewer tokens than tokens_total, as
   reported). The cumulative synthetic share weighed by the measured tokens per micro-step of both phases is
   checked against the 0.60 cap (OD9). The CUDA-graph rule of spec section 5 (kinematics and rendering above
   CUDA_GRAPH_SHARE of a step) is read on the timed passes.
7. Allocator cap: refined phase B steps once more with the cap lifted (the fraction set back to 1), peak reserved
   memory against the capped passes.
--report-only rewrites the markdown from reports/bench/stage1_pilot.json.

Options (weekend task T15; the default call, no option, is the bench above with the files and report text it wrote
before them):
- --config: the configuration benched (default configs/stage1_pilot.yaml); the model of its model section (the bucket
  rows of step 2 are timed on its encoder). --out and --out-stem: the report <out>/<stem>.md and .json (default
  reports/bench/stage1_pilot). Another configuration under the stem stage1_pilot is refused, so that the pilot bench
  cited by OD5 and the spec is never overwritten by another model's numbers; the stem is compared without case (a case
  variant names the same files on this tree's case-insensitive disk), and any call but the default one whose report
  is the same file as the pilot bench's is refused too. Run folder: cache/runs/_bench_stage1, or
  cache/runs/_bench_<stem> for another stem (no checkpoint, no log line lands there).
- --no-write-config: the configuration file is never written; the rewritten text is checked in the run folder and
  removed (without it, as before, the temporary file sits next to the configuration and replaces it).
- --memory-gb: the caching-allocator cap of every capped pass (default loop.MEMORY_CAP_GB). --skip-uncapped: step 7
  runs its capped pass only, so that no pass lifts the cap (the evaluation cap of the GPU queue).
- Any call but the default one (another configuration, folder or stem) also: runs refined phase B steps in each
  micro-batch candidate (the pass with the highest peak of the pilot bench), so that a candidate fits only when both
  phases do; measures FALLBACK_MICRO the same way when no candidate of MICRO_CANDIDATES fits under the cap; reports
  the parameters per component, the GPU-hours of the configuration's own token budget and of the design token budget
  (DESIGN_EPOCHS epochs of DESIGN_EPOCH_TOKENS), and its rates against the pilot bench
  (reports/bench/stage1_pilot.json). The default call does none of this; when no candidate fits it stops with a
  message, as it stopped before on the empty list of fitting candidates.
- --redraws K (task T15B): the samplers of the bench (the histogram plans and every pass) draw again a plan or a step
  that the caps leave empty, at most K times (the opt-in redraws of mdfm/data/sampler.py); the report counts the draws
  made again per pass. 0, the default, is the sampler of the training runs; refused for the default call (the pilot
  bench measures the pilot training's sampler). Why: at the fallback micro-batch of 8,192 tokens the phase B sampler
  meets plans that the caps leave empty, where the bench of 2026-10-05 stopped (reports/bench/stage1_200m_partial.md).
- --plan-check STEPS (task T15B): no bench and no GPU; STEPS phase B plans (no render: a synthetic window's count is
  its layout's upper bound) of the configuration's sampler at its tokens_per_micro and at each FALLBACK_MICRO size, by
  the default sampler (until its first empty plan) and by the sampler with the call's --redraws; writes
  <out>/<stem>_plan_check.md and .json. A stage-2 configuration (its stage2 section enabled, task S2.7) checks its
  phase C plans on the stage-1 root and the stage-2 root, with the replay share and each key's largest share against
  its cap in the report; the bench itself refuses a stage-2 configuration (its bench is task S2.9's). A size where the
  sampler with redraws gives no plan at all (its redraws run out at the first plan) is a row like the others: its
  share cells are empty, its replay share reads 'no plan' and its printed fill 'none' (task HK; a TypeError stopped
  the check there before).

Run with the mdfm environment (about 25 minutes on the RTX 5080):
  <conda>/envs/mdfm/python.exe scripts/bench_stage1.py
The 200M lean configuration (T15; through scripts/queue_job.py; --redraws 8 since T15B):
  <conda>/envs/mdfm/python.exe scripts/bench_stage1.py --config configs/stage1_200m.yaml
      --no-write-config --out reports/bench --out-stem stage1_200m --memory-gb 12 --skip-uncapped --redraws 8
Its plan check (CPU only; reports/bench/stage1_200m_plan_check.md):
  <conda>/envs/mdfm/python.exe scripts/bench_stage1.py --config configs/stage1_200m.yaml
      --no-write-config --out reports/bench --out-stem stage1_200m --plan-check 2000 --redraws 8
"""
import argparse
import datetime as _dt
import gc
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.data import sampler as S  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.factory import layouts as LY  # noqa: E402
from mdfm.factory import synth as SY  # noqa: E402
from mdfm.model import encoder as E  # noqa: E402
from mdfm.model.masks import build_mask  # noqa: E402
from mdfm.train import bodies as BD  # noqa: E402
from mdfm.train import config as CF  # noqa: E402
from mdfm.train import loop as LP  # noqa: E402

# read by the CUDA caching allocator when CUDA starts (after this line), not at import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", LP.ALLOC_CONF)

CONFIG = os.path.join(TREE, "configs", "stage1_pilot.yaml")
OUT = os.path.join(TREE, "reports", "bench")
DEFAULT_STEM = "stage1_pilot"
PILOT_MD = os.path.join(OUT, f"{DEFAULT_STEM}.md")
PILOT_JSON = os.path.join(OUT, f"{DEFAULT_STEM}.json")
RUN_DIR = os.path.join(paths.CACHE, "runs", "_bench_stage1")
STEM = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")
PLAN_STEPS = 300
ENC_PLANS = 12
ENC_REPS = 3
ROW_CANDIDATES = (1280, 1536, 2048)
SMALL_ROWS = (128, 640)
MICRO_CANDIDATES = ((16384, 2), (32768, 1))
FALLBACK_MICRO = ((8192, 4),)  # measured only when no candidate above fits under the cap (32,768 tokens per step)
PREFETCH = 2
REFINE_OVERHEAD = 0.25     # [cfg] amortized refinement cost, share of an unrefined phase B optimizer step
TOKEN_GRID = tuple(round(1.0e9 + 0.05e9 * k) for k in range(11))
# design token budget of a 200M run (any call but the default one): docs/MFM_DESIGN_NOTE_ARCHITECTURE_2026-09-24.md
# section 3 estimates 1.2 to 1.7 billion patch tokens per epoch and a compute-optimal 200M run at a few epochs
DESIGN_EPOCH_TOKENS = (1.2e9, 1.7e9)
DESIGN_EPOCHS = (1, 2, 3)
WALL_BUDGET_H = 24.0       # [cfg] GPU-hours of one pilot run (spec section 5 estimated 12 to 23)
TARGET_GB = 12.0           # spec section 7: peak memory target
CUDA_GRAPH_SHARE = 0.40    # spec section 5: CUDA graphs when kinematics and rendering exceed this share of a step
STEPS = {"micro_probe": 6, "a_timed": 40, "a_rate": 100, "b_timed": 30, "b_timed_refined": 15, "b_rate0": 60,
         "b_rate1": 30, "b_rateN": 100, "uncapped": 15}
WARMUP = 3


# ---------------------------------------------------------------- environment
def moveatlas_commit():
    """Read-only: `git log -1` of the MoveAtlas tree whose SMPL+H arrays the model reads."""
    root = paths.MOVEATLAS
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", root, "log", "-1"], capture_output=True,
                           text=True, timeout=60)
        lines = r.stdout.splitlines()
        sha = lines[0].split()[1][:7] if lines and lines[0].startswith("commit ") else "unknown"
        date = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Date:")), "")
        return f"{sha} ({date})" if date else sha
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def sync():
    torch.cuda.synchronize()


def tree_path(p):
    """An absolute path; a relative one is read from the tree root (the queue runs from there too)."""
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(TREE, p))


def rel(p):
    """Tree-relative path with forward slashes (the path itself outside the tree)."""
    try:
        r = os.path.relpath(p, TREE)
    except ValueError:                # another drive
        return str(p).replace("\\", "/")
    return (str(p) if r.startswith("..") else r).replace("\\", "/")


def same_path(a, b):
    """True when a and b name the same file or folder: equal after os.path.normcase, or os.path.samefile when both
    exist (a case-insensitive disk such as this tree's exFAT one, where normcase alone may not tell on the Mac)."""
    a, b = os.path.abspath(a), os.path.abspath(b)
    if os.path.normcase(a) == os.path.normcase(b):
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def bench_paths(config=CONFIG, out=OUT, stem=DEFAULT_STEM, write_config=True):
    """{config, config_rel, out, stem, md, json, run_dir, tmp, write_config, default} of a call (module docstring,
    options). The default call: configs/stage1_pilot.yaml, reports/bench/stage1_pilot.md and .json, the run folder
    cache/runs/_bench_stage1, the temporary YAML next to the configuration. Without write_config the temporary YAML
    lies in the run folder, never next to the configuration. A case variant of the pilot's stem is the pilot's stem
    (the same files on a case-insensitive disk). ValueError: a stem that is not a plain file stem, another
    configuration under the pilot's stem, or any call but the default one whose report is the same file as the pilot
    bench's (the pilot bench cited by OD5 is never overwritten by it)."""
    config, out, stem = tree_path(config), tree_path(out), str(stem)
    if not STEM.match(stem):
        raise ValueError(f"--out-stem {stem!r}: letters, digits, '_', '.' and '-' only")
    if stem.lower() == DEFAULT_STEM.lower():
        stem = DEFAULT_STEM
    pilot_config = same_path(config, CONFIG)
    if stem == DEFAULT_STEM and not pilot_config:
        raise ValueError(f"the stem {DEFAULT_STEM} holds the pilot bench (OD5); bench {rel(config)} under another "
                         f"--out-stem")
    run_dir = RUN_DIR if stem == DEFAULT_STEM else os.path.join(paths.CACHE, "runs", f"_bench_{stem}")
    tmp = config + ".tmp" if write_config else os.path.join(run_dir, os.path.basename(config) + ".tmp")
    md, js = os.path.join(out, f"{stem}.md"), os.path.join(out, f"{stem}.json")
    default = pilot_config and stem == DEFAULT_STEM and same_path(out, OUT)
    if not default and (same_path(md, PILOT_MD) or same_path(js, PILOT_JSON)):
        raise ValueError(f"{rel(md)} is the pilot bench (OD5); bench {rel(config)} under another --out-stem")
    return {"config": config, "config_rel": rel(config), "out": out, "stem": stem, "md": md, "json": js,
            "run_dir": run_dir, "tmp": tmp, "write_config": bool(write_config), "default": default}


def check_config_text(new_text, want, P):
    """Write the rewritten YAML text to P['tmp'] and check that it parses to the RunConfig `want` (RuntimeError, the
    temporary file removed, otherwise). Returns the config SHA-256 of the parsed text; finish_config settles the
    temporary file."""
    os.makedirs(os.path.dirname(P["tmp"]), exist_ok=True)
    with open(P["tmp"], "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    try:
        parsed = CF.load(P["tmp"])
        if CF.to_dict(parsed) != CF.to_dict(want):
            raise RuntimeError("the rewritten YAML does not parse to the measured configuration")
    except Exception:
        os.remove(P["tmp"])
        raise
    return CF.sha256(parsed)


def yaml_status_header(P, cfg0, date):
    """(status line, leading comment lines) of the rewritten YAML: the pilot's B4.18 text for the default call; for
    another configuration its own leading comments, and a status naming the bench report, then the former status."""
    if P["default"]:
        status = f"B4.18 measured values of reports/bench/stage1_pilot.md ({date}); the rest are the starting values of spec section 5"
        header = ["# Stage-1 60M pilot: run configuration (mdfm/train/config.py; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md,",
                  "# section 5). Starting values of section 5 and of the B4.x modules they come from: [cfg] values are changed on",
                  "# dev folds only; [B4.18] values were measured by scripts/bench_stage1.py (reports/bench/stage1_pilot.md), which",
                  "# writes them here: bucket rows, micro-batch, token budget with its step counts, refinement cadence. Pairs are",
                  "# (phase A, phase B). The config SHA-256 is that of the canonical JSON of the parsed values, not of this text."]
        return status, header
    status = f"measured values of {rel(P['md'])} ({date}); the status before them, {cfg0.status}"
    header = []
    with open(P["config"], encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if not line.startswith("#"):
                break
            header.append(line)
    return status, header


def finish_config(P):
    """Replace P['config'] with the temporary file when P['write_config'], else remove the temporary file."""
    if P["write_config"]:
        os.replace(P["tmp"], P["config"])
    else:
        os.remove(P["tmp"])


def components(model):
    """{submodule name: parameters} of an MFM's direct submodules (an absent one left out) and 'total'."""
    out = {name: int(sum(p.numel() for p in mod.parameters())) for name, mod in model.named_children()
           if mod is not None}
    out["direct parameters"] = int(sum(p.numel() for p in model.parameters(recurse=False)))
    out["total"] = int(sum(p.numel() for p in model.parameters()))
    return out


# ---------------------------------------------------------------- 1. histogram
def plan_counts(cfg, bs, ws, phase, n, seed, redraws=0):
    """[per micro-step: list of planned window token counts] of n plans of a phase (no render); redraws: the
    sampler's opt-in redraws (--redraws)."""
    sm = S.MixtureSampler(bs, ws if phase == "B" else None, None, phase, seed, cfg.mixture, redraws=redraws)
    out = []
    for _ in range(n):
        out.append([int(it.tokens) for it in sm.plan().items])
    return out


def max_layout_tokens(win_s=10.0):
    """The longest window a layout family can make (random_sites with all 14 sites)."""
    best = 0
    for name, lay in LY.load_layouts().items():
        full = replace(lay, k_sites=None) if lay.k_sites is not None else lay
        best = max(best, LY.token_count(full, win_s))
    return best


def histogram(counts, edges=(0, 128, 256, 512, 640, 768, 1024, 1280, 1536, 2048)):
    """Rows (lo, hi, windows) of the planned window token counts."""
    flat = np.array([c for step in counts for c in step])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows.append((lo, hi, int(((flat > lo) & (flat <= hi)).sum())))
    return rows, flat


# ---------------------------------------------------------------- 2. rows
def encoder_times(enc, counts_list, L, dev):
    """(median ms of forward and backward, useful tokens, padded tokens, rows) per micro-step of token counts
    packed into rows of L."""
    bb = S.BucketBatcher(rows=(L,), max_windows=512)
    out = []
    for counts in counts_list:
        pk = bb.pack(counts, row_len=L)
        seg = torch.full((pk.n_rows, L), -1, dtype=torch.int64)
        for i, (r, o) in enumerate(pk.placement):
            seg[r, o:o + counts[i]] = i
        seg = seg.to(dev)
        mask = build_mask(seg, torch.zeros_like(seg))
        g = torch.Generator(device=dev).manual_seed(len(counts))
        x = torch.randn(pk.n_rows, L, enc.blocks[0].n1.normalized_shape[0], device=dev, generator=g, requires_grad=True)
        t = torch.rand(pk.n_rows, L, device=dev, generator=g) * 10.0
        ms = []
        for rep in range(ENC_REPS + 1):
            enc.zero_grad(set_to_none=True)
            x.grad = None
            sync()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = enc(x, t, mask)
            y.float().square().mean().backward()
            sync()
            if rep:
                ms.append(1000 * (time.perf_counter() - t0))
        out.append((statistics.median(ms), pk.useful, pk.padded, pk.n_rows))
        del x, t, mask, seg, y
    return out


def encoder_kwargs(cfg):
    """Encoder(**kwargs) of a RunConfig's model (E.PILOT for the pilot)."""
    mc = cfg.model.mfm_config()
    return {"d": int(mc.d), "layers": int(mc.layers), "heads": int(mc.heads), "mlp_ratio": int(mc.mlp_ratio)}


def choose_rows(counts_a, counts_b, dev, longest, enc_kw=None):
    """The row table and the chosen largest row (module docstring, step 2), on the encoder Encoder(**enc_kw)
    (default E.PILOT)."""
    enc = E.Encoder(**(E.PILOT if enc_kw is None else enc_kw)).to(dev)
    sample = counts_a[:ENC_PLANS] + counts_b[:ENC_PLANS]
    table = []
    for L in ROW_CANDIDATES:
        if L < longest:
            continue
        res = encoder_times(enc, sample, L, dev)
        ms = sum(r[0] for r in res)
        useful, padded, rows = sum(r[1] for r in res), sum(r[2] for r in res), sum(r[3] for r in res)
        table.append({"row": L, "micro_steps": len(res), "rows_per_micro": rows / len(res),
                      "fill": useful / padded, "encoder_ms_per_micro": ms / len(res),
                      "encoder_ms_per_1k_useful": 1000 * ms / useful})
    del enc
    torch.cuda.empty_cache()
    best = min(table, key=lambda r: r["encoder_ms_per_1k_useful"])
    return table, int(best["row"])


# ---------------------------------------------------------------- passes
def with_redraws(tr, redraws):
    """tr, whose samplers (loop.Trainer._make_sampler, one per pass) draw again a plan or a step that the caps leave
    empty, at most `redraws` times (--redraws; mdfm/data/sampler.py); they are listed in tr.redraw_samplers. redraws
    0 returns tr untouched (the training's sampler)."""
    if not redraws:
        return tr
    make = tr._make_sampler
    tr.redraw_samplers = []

    def _make_sampler(phase):
        sm = make(phase)
        sm.redraws = int(redraws)
        tr.redraw_samplers.append(sm)
        return sm
    tr._make_sampler = _make_sampler
    return tr


def summarize(res):
    """Throughput, time split and memory of a loop.bench result, overall and per bucket."""
    micro = res["micro"]
    steps = max(res["steps"], 1)
    out = {"phase": res["phase"], "steps": res["steps"], "timed": res["timed"], "prefetch": res["prefetch"],
           "refine_every": res["refine_every"], "micro_steps": len(micro), "wall_s": res["wall_s"],
           "s_per_step": res["wall_s"] / steps, "tokens": res["tokens"], "padded": res["padded"],
           "tokens_per_s": res["tokens"] / res["wall_s"] if res["wall_s"] else None,
           "padded_per_s": res["padded"] / res["wall_s"] if res["wall_s"] else None,
           "tokens_per_micro": res["tokens"] / max(len(micro), 1), "fill": res["tokens"] / max(res["padded"], 1),
           "peak_allocated_gb": res["peak_allocated_gb"], "peak_reserved_gb": res["peak_reserved_gb"],
           "refined_micro": sum(1 for m in micro if m["refined"]), "buckets": {}}
    for L in sorted({m["row_len"] for m in micro}):
        ms = [m for m in micro if m["row_len"] == L]
        b = {"micro_steps": len(ms), "tokens_per_micro": float(np.mean([m["useful"] for m in ms])),
             "peak_allocated_gb": max(m["peak_allocated_gb"] for m in ms),
             "peak_reserved_gb": max(m["peak_reserved_gb"] for m in ms)}
        if res["timed"]:
            parts = {k: 1000 * float(np.mean([m["time"].get(k, 0.0) for m in ms])) for k in LP.PARTS}
            tot = sum(parts.values())
            b.update(split_ms=parts, micro_ms=tot, tokens_per_s_sync=b["tokens_per_micro"] / (tot / 1000))
            if any(m["refined"] for m in ms) and not all(m["refined"] for m in ms):
                raise ValueError("a timed pass mixes refined and unrefined micro-steps")
        out["buckets"][str(L)] = b
    return out


def run_pass(tr, phase, steps, timed, refine_every=None, label=""):
    t0 = time.perf_counter()
    made = getattr(tr, "redraw_samplers", None)
    n0 = len(made) if made is not None else 0
    res = LP.bench(tr, phase, steps=steps, timed=timed, refine_every=refine_every, warmup=WARMUP)
    s = summarize(res)
    s["label"] = label
    extra = ""
    if made is not None:                 # --redraws: the draws made again by the pass's sampler
        s["redrawn"] = int(sum(sm.redrawn for sm in made[n0:]))
        extra = f", {s['redrawn']} draws made again"
    print(f"  {label}: {s['tokens_per_s']:,.0f} useful tokens/s, {s['s_per_step']:.3f} s per optimizer step, "
          f"peak allocated {s['peak_allocated_gb']:.2f} GB, reserved {s['peak_reserved_gb']:.2f} GB "
          f"({time.perf_counter() - t0:.0f} s{extra})", flush=True)
    return s


# ---------------------------------------------------------------- 6. budget
def budget_row(cfg, s_a, s_b, tot):
    """{tokens_total, steps_a, steps_b, hours, tokens_consumed} of a token budget at the measured passes s_a, s_b."""
    c = replace(cfg, schedule=replace(cfg.schedule, tokens_total=float(tot)))
    a, b = CF.step_counts(c)
    hours = (a * s_a["s_per_step"] + b * s_b["s_per_step"]) / 3600.0
    mp = int(cfg.schedule.micro_per_step)
    consumed = a * mp * s_a["tokens_per_micro"] + b * mp * s_b["tokens_per_micro"]
    return {"tokens_total": tot, "steps_a": a, "steps_b": b, "hours": hours, "tokens_consumed": consumed}


def budget(cfg, s_a, s_b):
    """[(tokens_total, steps A, steps B, hours, tokens consumed)] of TOKEN_GRID and the chosen tokens_total."""
    rows = [budget_row(cfg, s_a, s_b, tot) for tot in TOKEN_GRID]
    fit = [r for r in rows if r["hours"] <= WALL_BUDGET_H]
    chosen = max(fit, key=lambda r: r["tokens_total"]) if fit else rows[0]
    return rows, chosen


def design_budget(cfg, s_a, s_b):
    """budget_row of each design token budget, epochs x tokens per epoch over DESIGN_EPOCHS and DESIGN_EPOCH_TOKENS
    (the design note's estimate for a 200M run), each row with its epochs and epoch_tokens."""
    out = []
    for ep in DESIGN_EPOCHS:
        for et in DESIGN_EPOCH_TOKENS:
            r = budget_row(cfg, s_a, s_b, float(round(ep * et)))
            r.update(epochs=int(ep), epoch_tokens=float(et))
            out.append(r)
    return out


def rates(R):
    """The kept rates of a bench result R (this run's, or reports/bench/stage1_pilot.json): date, parameters, kept
    prefetch, N, seconds per optimizer step and useful tokens per second of phase A (kept prefetch) and of phase B
    (refine every N), the micro-batch, the largest capped peaks, and {tokens_total: GPU-hours} of the budget grid."""
    pre, N = R["prefetch"], R["refine"]["N"]
    by = {p["label"]: p for p in R["passes"]}
    a, b = by[f"A prefetch {pre}"], by[f"B prefetch {pre}, refine every {N}"]
    return {"date": R["env"]["date"], "parameters": R["env"]["parameters"], "prefetch": pre, "N": N,
            "s_a": a["s_per_step"], "s_b": b["s_per_step"], "tokens_per_s_a": a["tokens_per_s"],
            "tokens_per_s_b": b["tokens_per_s"], "micro": list(R["micro"]["chosen"]),
            "peak_reserved_gb": R["peak"]["reserved_gb"], "peak_allocated_gb": R["peak"]["allocated_gb"],
            "hours": {str(int(r["tokens_total"])): r["hours"] for r in R["budget"]["rows"]}}


# ---------------------------------------------------------------- the YAML file
def _fmt(v):
    """YAML text of an int, a float (1.25e+9 style from 1e6 on, a dot in the mantissa so that YAML reads a
    float) or a list of them."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, float) and abs(v) >= 1e6:
        e = int(math.floor(math.log10(abs(v))))
        m = f"{v / 10 ** e:.6g}"
        return (m if "." in m else m + ".0") + f"e+{e}"
    return repr(v) if isinstance(v, float) else str(v)


def update_yaml(path, values, status, header):
    """Rewrite the values {(section, key): (value, comment)} and the status line of the YAML file in place,
    keeping every other line; header replaces the leading comment lines. Returns the new text."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out, section, seen = [], None, set()
    body = False
    for line in lines:
        if not body and line.startswith("#"):
            continue
        if not body:
            out.extend(header)
            body = True
        m = re.match(r"^([A-Za-z_]+):", line)
        if m:
            section = m.group(1)
            if section == "status":
                out.append(f"status: {status}")
                seen.add(("", "status"))
                continue
        m = re.match(r"^(  )([A-Za-z_]+):(\s*)(.*)$", line)
        if m and (section, m.group(2)) in values:
            val, comment = values[(section, m.group(2))]
            left = f"  {m.group(2)}: {_fmt(val)}"
            out.append(f"{left:<35}# {comment}" if comment else left)
            seen.add((section, m.group(2)))
            continue
        out.append(line)
    missing = (set(values) | {("", "status")}) - seen
    if missing:
        raise KeyError(f"{path}: keys not found {sorted(missing)}")
    return "\n".join(out)


# ---------------------------------------------------------------- report
def f1(x, n=1):
    return "" if x is None else f"{x:,.{n}f}"


def report_meta(R):
    """The labels of the report, from R['bench'] (bench_meta); a result without it (the pilot bench of 2026-09-29)
    gets the pilot's labels, so that its report reads as before."""
    b = R.get("bench") or {}
    return {"default": b.get("default", True), "config": b.get("config", "configs/stage1_pilot.yaml"),
            "write_config": b.get("write_config", True), "encoder": b.get("encoder", "Pilot encoder (12 x 512)"),
            "skip_uncapped": b.get("skip_uncapped", False), "redraws": int(b.get("redraws", 0))}


def bench_meta(P, enc_kw, skip_uncapped, redraws=0):
    """R['bench'] of a call: its paths and labels (report_meta); redraws (--redraws) only when it is not 0."""
    pilot = P["default"] or enc_kw == E.PILOT
    meta = {"default": P["default"], "config": P["config_rel"], "stem": P["stem"], "write_config": P["write_config"],
            "encoder": "Pilot encoder (12 x 512)" if pilot else
            f"Encoder of the configuration ({enc_kw['layers']} x {enc_kw['d']}, {enc_kw['heads']} heads)",
            "enc_kw": enc_kw, "skip_uncapped": bool(skip_uncapped)}
    if redraws:
        meta["redraws"] = int(redraws)
    return meta


def write_report(R, path):
    L = []
    a = L.append
    env = R["env"]
    meta = report_meta(R)
    cfg_name = meta["config"]
    if meta["default"]:
        a("# Stage-1 pilot, end-to-end bench on the RTX 5080 (B4.18)")
    else:
        a(f"# Stage-1 end-to-end bench of {cfg_name} on the RTX 5080 (T15)")
    a("")
    if meta["write_config"]:
        cfg_text = (f"{cfg_name} (SHA-256 {env['config_sha256_before'][:12]} before this bench, "
                    f"{env['config_sha256_after'][:12]} after it wrote its values)")
    else:
        cfg_text = (f"{cfg_name} (SHA-256 {env['config_sha256_before'][:12]}, left unchanged: --no-write-config; the "
                    f"measured values below would give {env['config_sha256_after'][:12]})")
    model_text = "the pilot MFM" if meta["default"] else f"the MFM of {cfg_name} (stage 1: no session aggregator)"
    a(f"Measured by scripts/bench_stage1.py on {env['date']} with torch {env['torch']} (CUDA {env['cuda']}) on "
      f"{env['gpu']} ({env['gpu_mib']:,} MiB), MoveAtlas {env['moveatlas']}. Configuration: {cfg_text}. Model: "
      f"{model_text}, {env['parameters']:,} parameters, bf16 autocast in the stems, "
      f"encoder, prior and decoder trunk, float32 knots, kinematics and kernels; AdamW {env['optimizer']} "
      f"(fused smoke test: {env['fused_smoke']}). Data: the AMASS body store (train split, {env['body_takes']:,} "
      f"takes) through the GPU renderer, the window store of the real sources. Caching allocator capped at "
      f"{env['memory_cap_gb']} GB (loop.cap_memory) with PYTORCH_CUDA_ALLOC_CONF={env['alloc_conf']}. GB are "
      f"1e9 bytes. Every pass starts with {WARMUP} unmeasured optimizer steps; the weights train during the bench.")
    if not meta["default"]:
        L[-1] += (" No checkpoint is written; the run folder of the bench stays empty.")
    a("")
    a("Timed passes synchronize CUDA between the parts of every micro-step (the time split, which slows the step); "
      "untimed passes synchronize only at the ends of each optimizer step and give the throughput. Useful tokens "
      "are window tokens, padded tokens the rows the encoder runs.")
    a("")
    if meta["write_config"]:
        a(f"## Values written into {cfg_name}")
    else:
        a(f"## Values measured for {cfg_name} (not written: --no-write-config)")
    a("")
    a("| value | before | after | rule |")
    a("|---|---|---|---|")
    for row in R["written"]:
        a(f"| {row['key']} | {row['before']} | {row['after']} | {row['rule']} |")
    a("")
    a(f"Step counts of the budget (config.step_counts, full micro-steps): phase A {R['budget']['chosen']['steps_a']:,}, "
      f"phase B {R['budget']['chosen']['steps_b']:,} optimizer steps; at the measured tokens per micro-step the run "
      f"consumes {R['budget']['chosen']['tokens_consumed'] / 1e9:.3f} billion tokens and takes "
      f"{R['budget']['chosen']['hours']:.1f} GPU-hours.")
    if not meta["default"] and R["budget"]["chosen"]["hours"] > WALL_BUDGET_H:
        L[-1] += (f" No budget of the grid fits in {WALL_BUDGET_H:.0f} GPU-hours at these rates: the smallest is "
                  f"kept (section 7).")
    cf = R["budget"].get("configured")
    if cf is not None:
        a("")
        a(f"At the token budget of {cfg_name} ({cf['tokens_total'] / 1e9:.2f} billion tokens) and the measured rates: "
          f"phase A {cf['steps_a']:,} and phase B {cf['steps_b']:,} optimizer steps, "
          f"{cf['tokens_consumed'] / 1e9:.3f} billion tokens consumed, {cf['hours']:.1f} GPU-hours.")
    a("")
    comp = R.get("components")
    if comp and not meta["default"]:
        a("## Parameters per component")
        a("")
        a("Direct submodules of the model the bench trained (the SkeletonBank holds no parameter).")
        a("")
        a("| component | parameters |")
        a("|---|---|")
        for k, v in comp.items():
            if k != "total":
                a(f"| {k} | {v:,} |")
        a(f"| total | {comp['total']:,} |")
        a("")
    a("## 1. Window-length histogram (planned tokens)")
    a("")
    a(f"{PLAN_STEPS} planned micro-steps per phase at the configuration's mixture (no render; a synthetic window's "
      f"count is its layout's upper bound). The longest window any layout makes: {R['hist']['longest_layout']:,} "
      f"tokens (random_sites with the 14 sites).")
    a("")
    a("| tokens | phase A windows | phase B windows |")
    a("|---|---|---|")
    for (lo, hi, na), (_, _, nb) in zip(R["hist"]["A"]["rows"], R["hist"]["B"]["rows"]):
        a(f"| {lo + 1} to {hi} | {na:,} | {nb:,} |")
    for ph in ("A", "B"):
        h = R["hist"][ph]
        a("")
        a(f"Phase {ph}: {h['windows']:,} windows, median {h['median']:,.0f} tokens, {h['per_micro']:.1f} windows per "
          f"micro-step; longest window of a micro-step: median {h['longest_median']:,.0f}, minimum "
          f"{h['longest_min']:,}; micro-steps whose longest window fits 128 tokens: {h['fit_128']}, 640 tokens: "
          f"{h['fit_640']} of {PLAN_STEPS}.")
    a("")
    a("## 2. Bucket rows")
    a("")
    a(f"{meta['encoder']} forward and backward in bf16 on the packing of {ENC_PLANS} planned micro-steps of "
      f"each phase, median of {ENC_REPS} runs each.")
    a("")
    a("| largest row | rows per micro-step | fill | encoder ms per micro-step | ms per 1,000 useful tokens |")
    a("|---|---|---|---|---|")
    for r in R["rows"]["table"]:
        a(f"| {r['row']:,} | {r['rows_per_micro']:.1f} | {r['fill']:.3f} | {r['encoder_ms_per_micro']:.1f} | "
          f"{r['encoder_ms_per_1k_useful']:.2f} |")
    a("")
    a(f"Chosen rows: {R['rows']['chosen']}.")
    a("")
    a("## 3. Micro-batch")
    a("")
    a("| tokens per micro-step | micro-steps per optimizer step | outcome | useful tokens/s (timed) | peak allocated GB | peak reserved GB |")
    a("|---|---|---|---|---|---|")
    for r in R["micro"]["table"]:
        a(f"| {r['tokens_per_micro']:,} | {r['micro_per_step']} | {r['outcome']} | {f1(r.get('tokens_per_s'), 0)} | "
          f"{f1(r.get('peak_allocated_gb'), 2)} | {f1(r.get('peak_reserved_gb'), 2)} |")
    a("")
    a(f"Chosen: {R['micro']['chosen'][0]:,} tokens per micro-step, {R['micro']['chosen'][1]} micro-steps per "
      f"optimizer step.")
    if R["micro"].get("fallback"):
        L[-1] += (" No candidate of " + ", ".join(f"{t:,} x {m}" for t, m in MICRO_CANDIDATES) + " fit under the cap; "
                  "the fallback " + ", ".join(f"{t:,} x {m}" for t, m in FALLBACK_MICRO) + " was measured the same way.")
    if R["micro"].get("probe_b"):
        L[-1] += (f" Each candidate also ran {R['steps']['micro_probe']} refined phase B optimizer steps (refine_every "
                  f"1, after {WARMUP} unmeasured ones; the pass with the highest peak of the pilot bench): a candidate "
                  f"fits when both passes ran under the cap; its peaks are the larger of the two passes, its tokens/s "
                  f"those of phase A.")
    a("")
    a("## 4. Throughput per phase")
    a("")
    a("| pass | prefetch | refine every | optimizer steps | useful tokens/s | padded tokens/s | s per optimizer step | tokens per micro-step | peak allocated GB | peak reserved GB |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for s in R["passes"]:
        a(f"| {s['label']} | {s['prefetch']} | {s['refine_every'] if s['phase'] == 'B' else '-'} | {s['steps']} | "
          f"{f1(s['tokens_per_s'], 0)} | {f1(s['padded_per_s'], 0)} | {s['s_per_step']:.3f} | "
          f"{s['tokens_per_micro']:,.0f} | {s['peak_allocated_gb']:.2f} | {s['peak_reserved_gb']:.2f} |")
    a("")
    if meta["redraws"]:
        made = [(s["label"], s["redrawn"]) for s in R["passes"] if "redrawn" in s]
        a(f"Redraws (--redraws {meta['redraws']}, mdfm/data/sampler.py, task T15B): the samplers of this bench draw "
          f"again a plan or a step that the caps leave empty, at most {meta['redraws']} times in a row; the caps hold "
          f"on every step they return. Draws made again per pass: "
          + "; ".join(f"{lab} {n:,}" for lab, n in made) + ".")
        a("")
    a("## 5. Time split and peak memory per bucket (timed passes)")
    a("")
    a("Milliseconds per micro-step, CUDA synchronized between parts. sample: plan, body and window reads, guard; "
      "render: the synthetic renderer (its GPU kinematics and host streams); backward covers every module. "
      "Peaks are those of the optimizer steps holding the bucket's micro-steps.")
    a("")
    timed = [s for s in R["passes"] if s["timed"]]
    head = "| pass | bucket | micro-steps | " + " | ".join(LP.PARTS) + " | total ms | tokens/s synchronized | peak allocated GB | peak reserved GB |"
    a(head)
    a("|" + "---|" * (len(LP.PARTS) + 7))
    for s in timed:
        for L_, b in s["buckets"].items():
            a(f"| {s['label']} | {int(L_):,} | {b['micro_steps']} | "
              + " | ".join(f"{b['split_ms'][k]:.0f}" for k in LP.PARTS)
              + f" | {b['micro_ms']:.0f} | {b['tokens_per_s_sync']:,.0f} | {b['peak_allocated_gb']:.2f} | "
                f"{b['peak_reserved_gb']:.2f} |")
    a("")
    worst = 0.0
    for s in timed:
        for L_, b in s["buckets"].items():
            sp = b["split_ms"]
            host = sp["sample"] + sp["render"] + sp["collate"]
            kin = sp["render"] + sp["kinematics"]
            worst = max(worst, kin / b["micro_ms"])
            a(f"- {s['label']}, bucket {int(L_):,}: host parts (sample, render, collate) {host:.0f} ms of "
              f"{b['micro_ms']:.0f} ({100 * host / b['micro_ms']:.0f} percent); rendering and kinematics "
              f"{100 * kin / b['micro_ms']:.0f} percent; refinement {100 * sp['refinement'] / b['micro_ms']:.0f} percent.")
    a("")
    a(f"Spec section 5 captures kinematics and rendering as CUDA graphs when they exceed {100 * CUDA_GRAPH_SHARE:.0f} "
      f"percent of the step: they take at most {100 * worst:.0f} percent here, so no CUDA graph is built.")
    a("")
    a("## 6. Refinement cadence")
    a("")
    rf = R["refine"]
    a(f"Wall seconds per phase B optimizer step with prefetch {R['prefetch']}: unrefined t0 = {rf['t0']:.3f}, every "
      f"step refined ({rf['steps']} steps of Adam on {rf['points']} points per stream-window, stopped gradients) "
      f"t1 = {rf['t1']:.3f}; refinement adds {rf['t1'] - rf['t0']:.3f} s to a refined step. Rule: the smallest N "
      f"with (t1 - t0) / N <= {REFINE_OVERHEAD} t0 gives N = {rf['N']}; measured at N: {rf['tN']:.3f} s per "
      f"optimizer step, {rf['tN_tokens_per_s']:,.0f} useful tokens/s.")
    a("")
    a("## 7. Token budget and step counts")
    a("")
    a(f"Seconds per optimizer step: phase A {R['budget']['s_a']:.3f} (prefetch {R['prefetch']}), phase B "
      f"{R['budget']['s_b']:.3f} (prefetch {R['prefetch']}, N = {rf['N']}). Rule: the largest budget with a run of at "
      f"most {WALL_BUDGET_H:.0f} GPU-hours.")
    a("")
    a("| tokens_total | phase A steps | phase B steps | GPU-hours | tokens consumed |")
    a("|---|---|---|---|---|")
    for r in R["budget"]["rows"]:
        marks = ["chosen"] if r["tokens_total"] == R["budget"]["chosen"]["tokens_total"] else []
        if cf is not None and r["tokens_total"] == cf["tokens_total"]:
            marks.append("configured")
        mark = f" ({', '.join(marks)})" if marks else ""
        a(f"| {r['tokens_total'] / 1e9:.2f} billion{mark} | {r['steps_a']:,} | {r['steps_b']:,} | {r['hours']:.1f} | "
          f"{r['tokens_consumed'] / 1e9:.3f} billion |")
    a("")
    design = R["budget"].get("design")
    if design and not meta["default"]:
        a("Design token budget: docs/MFM_DESIGN_NOTE_ARCHITECTURE_2026-09-24.md section 3 estimates "
          + " to ".join(f"{t / 1e9:.1f}" for t in DESIGN_EPOCH_TOKENS) + " billion patch tokens per epoch and a "
          "compute-optimal 200M run at a few epochs; the rows below read those tokens as the window tokens of "
          "this bench and take the configuration's phase split. GPU-hours at the measured seconds per optimizer step "
          "above; they scale linearly with tokens_total (config.step_counts).")
        a("")
        a("| epochs | tokens per epoch | tokens_total | phase A steps | phase B steps | GPU-hours | tokens consumed |")
        a("|---|---|---|---|---|---|---|")
        for r in design:
            a(f"| {r['epochs']} | {r['epoch_tokens'] / 1e9:.1f} billion | {r['tokens_total'] / 1e9:.2f} billion | "
              f"{r['steps_a']:,} | {r['steps_b']:,} | {r['hours']:.1f} | {r['tokens_consumed'] / 1e9:.3f} billion |")
        a("")
    cs = R["synthetic_share"]
    a(f"Cumulative synthetic share (OD9, read on tokens): phase A {cs['tokens_a']:,.0f} tokens per micro-step, "
      f"synthetic only; phase B {cs['tokens_b']:,.0f} tokens per micro-step, synthetic share {cs['share_b']:.4f}; "
      f"weighed by these tokens over share_a = {cs['share_a']}: {cs['cumulative']:.4f} against the cap "
      f"{cs['cap']} ({'under' if cs['cumulative'] <= cs['cap'] else 'ABOVE'} the cap).")
    a("")
    a("## 8. Allocator cap")
    a("")
    u = R["uncapped"]
    a(f"Refined phase B steps (refine_every 1, prefetch {R['prefetch']}): with the cap of {env['memory_cap_gb']} GB "
      f"the pool peaks at {u['capped_reserved_gb']:.2f} GB for {u['capped_allocated_gb']:.2f} GB allocated, at "
      f"{u['capped_tokens_per_s']:,.0f} useful tokens/s. ")
    if u["outcome"] == "skipped":
        L[-1] += ("The pass with the cap lifted was not run (--skip-uncapped): the cap held for the whole bench, as "
                  "the evaluation cap of the GPU queue asks.")
    else:
        L[-1] += ("With the fraction set back to 1 (the garbage-collection threshold then acts at 80 percent of the "
                  "whole card), after an empty_cache: ")
    if u["outcome"] == "skipped":
        pass
    elif u["outcome"] == "completed":
        L[-1] += (f"the pool peaks at {u['reserved_gb']:.2f} GB for {u['allocated_gb']:.2f} GB allocated, at "
                  f"{u['tokens_per_s']:,.0f} useful tokens/s.")
    else:
        L[-1] += (f"the pass stopped with {u['outcome']} after {u['stopped_after']} of its {u['steps']} optimizer "
                  f"steps, the pool at {u['reserved_gb']:.2f} GB for {u['allocated_gb']:.2f} GB allocated.")
    a("")
    a(f"The loop asserts torch.cuda.max_memory_reserved() under {R['max_reserved_gb']} GB (MemoryBudgetError).")
    a("")
    a("## Peak memory against the target")
    a("")
    pk = R["peak"]
    a(f"Largest peaks over every capped pass: {pk['allocated_gb']:.2f} GB allocated, {pk['reserved_gb']:.2f} GB "
      f"reserved, against the spec target of at most {TARGET_GB:.0f} GB and the loop bound of "
      f"{R['max_reserved_gb']} GB.")
    a("")
    ref = R.get("pilot_ref")
    if ref:
        me = rates(R)
        tot = str(int(cf["tokens_total"])) if cf is not None else None
        a("## Against the pilot bench")
        a("")
        a(f"The pilot bench is reports/bench/stage1_pilot.md and .json ({ref['date']}), measured by this script with the "
          f"code of that date: changes of the training loop since then enter the ratios together with the model size. "
          f"Each bench at its own kept prefetch, refinement cadence N and micro-batch.")
        a("")
        a("| quantity | this bench | pilot bench | ratio |")
        a("|---|---|---|---|")

        def row(name, x, y, nd):
            ratio = f"{x / y:.2f}" if x is not None and y else ""
            a(f"| {name} | {f1(x, nd)} | {f1(y, nd)} | {ratio} |")
        row("parameters", me["parameters"], ref["parameters"], 0)
        a(f"| micro-batch (tokens x micro-steps) | {me['micro'][0]:,} x {me['micro'][1]} | {ref['micro'][0]:,} x "
          f"{ref['micro'][1]} | |")
        row(f"phase A s per optimizer step (prefetch {me['prefetch']}; pilot {ref['prefetch']})", me["s_a"],
            ref["s_a"], 3)
        row(f"phase B s per optimizer step (refine every {me['N']}; pilot {ref['N']})", me["s_b"], ref["s_b"], 3)
        row("phase A useful tokens/s", me["tokens_per_s_a"], ref["tokens_per_s_a"], 0)
        row("phase B useful tokens/s", me["tokens_per_s_b"], ref["tokens_per_s_b"], 0)
        row("largest peak reserved GB, capped passes", me["peak_reserved_gb"], ref["peak_reserved_gb"], 2)
        if tot is not None and tot in ref["hours"]:
            row(f"GPU-hours at {int(tot) / 1e9:.2f} billion tokens", cf["hours"], ref["hours"][tot], 1)
        a("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------- plan check (--plan-check, T15B)
def plan_key(plan):
    """The draws of a plan: per item its source, person, tokens, unit, body, pair, recording, t0, take, start, layout."""
    return [(it.source, it.person, it.tokens, it.unit, it.body, it.pair, it.second, it.recording, it.t0, it.take,
             it.start, it.layout) for it in plan.items]


def plan_check(cfg, bs, ws, steps, redraws, sizes, phase="B"):
    """Rows, one per tokens_per_micro of sizes, of `steps` plans (no render) of cfg's sampler of `phase` (seed
    loop.phase_seed; phase B, or C for a stage-2 configuration, task S2.7, whose sampler reads the RunConfig and a
    sampler.Stage2Stores ws): the default sampler's first plan that the caps leave empty (None: none in `steps`),
    and over
    the plans of the sampler with redraws: draws made again, the plan where its redraws ran out (None: never), whether
    its plans equal the default sampler's before the latter's first empty plan, planned tokens per micro-step (mean,
    minimum, fill of tokens_per_micro), the largest real-source and synthetic shares of a micro-step's tokens (the caps
    source_cap and synthetic_share[1]), the shares of all planned tokens per source and the sampler's shares of
    tokens_per_micro; first_empty_trace: trace_empty_plan of the default sampler's first empty plan (None without
    one); in phase C also the replay share (the synthetic and replay sources' tokens over all planned tokens) and the
    largest share of a micro-step per budget key against its cap."""
    rows = []
    seed = LP.phase_seed(cfg, phase)
    for n in sizes:
        mix = replace(cfg.mixture, tokens_per_micro=int(n))
        if phase == LP.STAGE2_PHASE:
            mix = replace(cfg, mixture=mix)        # phase C reads the mixture and the stage2 section
        d = S.MixtureSampler(bs, ws, None, phase, seed, mix)
        ref, first, trace = [], None, None
        for k in range(steps):
            before = d.state_dict()
            try:
                ref.append(plan_key(d.plan()))
            except S.EmptyStep:
                first = k
                trace = trace_empty_plan(bs, ws, seed, mix, before, phase)
                break
        r = S.MixtureSampler(bs, ws, None, phase, seed, mix, redraws=redraws)
        per_key = {}
        totals, sizes_, max_real, max_syn, same, out_at = {}, [], 0.0, 0.0, True, None
        for k in range(steps):
            try:
                p = r.plan()
            except S.EmptyStep:
                out_at = k
                break
            if k < len(ref):
                same = same and plan_key(p) == ref[k]
            t = p.tokens()
            T = sum(t.values())
            sizes_.append(T)
            for s, v in t.items():
                totals[s] = totals.get(s, 0) + v
            max_real = max([max_real] + [v / T for s, v in t.items() if s != S.SYNTHETIC])
            max_syn = max(max_syn, t.get(S.SYNTHETIC, 0) / T)
            for s, v in t.items():
                per_key[s] = max(per_key.get(s, 0.0), v / T)
        tot = sum(totals.values())
        row = {"tokens_per_micro": int(n), "steps": int(steps), "default_first_empty": first,
               "redraws": int(redraws), "plans": len(sizes_), "redrawn": int(r.redrawn),
               "redraws_ran_out_at": out_at, "same_as_default_before_first_empty": bool(same),
               "compared": len(ref), "tokens_mean": float(np.mean(sizes_)) if sizes_ else None,
               "tokens_min": int(min(sizes_)) if sizes_ else None,
               "fill": float(np.mean(sizes_)) / n if sizes_ else None,
               "max_real_share": max_real, "max_synthetic_share": max_syn,
               "token_shares": {s: v / tot for s, v in sorted(totals.items())} if tot else {},
               "sampler_shares": {s: float(v) for s, v in r.shares.items()},
               "data_sha256": r.data_sha256(), "seed": seed, "first_empty_trace": trace}
        if phase == LP.STAGE2_PHASE:
            replay = [S.SYNTHETIC] + [s for s in r.order if s in r.store_of and r.store_of[s] is ws.replay]
            row.update(phase=phase, replay_keys=replay, units=list(r.stage2.units),
                       replay_share=sum(totals.get(s, 0) for s in replay) / tot if tot else None,
                       max_key_share={s: float(v) for s, v in sorted(per_key.items())},
                       caps={s: float(c) for s, c in sorted(r.caps.items())},
                       hours={s: float(h) for s, h in sorted(r.hours.items())},
                       persons={s: int(len(r.real[s]["persons"])) for s in sorted(r.real)})
        rows.append(row)
    return rows


def trace_empty_plan(bs, ws, seed, mix, state, phase="B"):
    """The cap passes of the default sampler's plan from `state` (one that the caps leave empty): per call of
    sampler.enforce_caps (the real sources against the total the synthetic budget will complete, then every cap on
    the micro-step) the cap (in phase C {key: cap}), the synthetic share bound, the tokens per source and the tokens of
    each body per source before the call, and the tokens left after it."""
    sm = S.MixtureSampler(bs, ws, None, phase, seed, mix).load_state_dict(state)
    calls, orig = [], S.enforce_caps

    def spy(items, cap, synthetic_share=1.0, synthetic=S.SYNTHETIC):
        bodies = {}
        for it in items:
            bodies.setdefault(it.source, {})
            bodies[it.source][it.body] = bodies[it.source].get(it.body, 0) + int(it.tokens)
        out, dropped = orig(items, cap, synthetic_share, synthetic)
        calls.append({"cap": {k: float(v) for k, v in cap.items()} if isinstance(cap, dict) else float(cap),
                      "synthetic_share": float(synthetic_share),
                      "tokens": S.source_tokens(items), "bodies": {s: sorted(b.values()) for s, b in bodies.items()},
                      "tokens_after": S.source_tokens(out), "dropped": int(dropped)})
        return out, dropped
    S.enforce_caps = spy
    try:
        sm.plan()
    except S.EmptyStep as e:
        return {"error": str(e), "calls": calls}
    finally:
        S.enforce_caps = orig
    raise RuntimeError("the traced plan is not empty")


def share_cell(r, v):
    """A share of plan check row r as the report prints it: '' for a size where no plan held a window (r['plans'] 0:
    there is no planned token to share), else the value to 4 decimals."""
    return "" if not r["plans"] else f"{v:.4f}"


def plan_check_line(r):
    """The line run_plan_check prints for row r of plan_check; a size without a plan has no fill."""
    fill = "none (no plan)" if r["fill"] is None else f"{r['fill']:.3f}"
    return (f"  {r['tokens_per_micro']}: default first empty plan {r['default_first_empty']}, {r['plans']} plans with "
            f"redraws, {r['redrawn']} draws made again, fill {fill}")


def write_plan_check(C, path):
    """The markdown of a plan check result C (main, --plan-check)."""
    L = []
    a = L.append
    env = C["env"]
    ph = C.get("phase", "B")
    a(f"# Phase {ph} plans of {C['config']} at each micro-step size (T15B)")
    a("")
    a(f"Measured by scripts/bench_stage1.py --plan-check {C['steps']} --redraws {C['redraws']} on {env['date']}, "
      f"MoveAtlas {env['moveatlas']}. Configuration: {C['config']} (SHA-256 {env['config_sha256'][:12]}); the "
      f"phase {ph} sampler of the training (mdfm/data/sampler.py, seed {C['rows'][0]['seed']} = loop.phase_seed), its "
      f"mixture with tokens_per_micro replaced; store tables of data SHA-256 {C['rows'][0]['data_sha256'][:12]}. Plans "
      f"only, on the CPU: no window read, no render (a synthetic window's count is its layout's upper bound). The "
      f"bench passes start a fresh sampler of the phase, so they draw these plans in this order.")
    a("")
    a(f"| tokens per micro-step | default sampler: first plan the caps leave empty | plans with redraws "
      f"{C['redraws']} | draws made again | same plans as the default before its first empty one | planned tokens per "
      f"micro-step, mean | minimum | fill | largest real-source share of a micro-step | largest synthetic share |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in C["rows"]:
        first = f"plan {r['default_first_empty']:,} (0-based)" if r["default_first_empty"] is not None else \
            f"none in {r['steps']:,}"
        plans = f"{r['plans']:,}" + (f" (redraws ran out at plan {r['redraws_ran_out_at']:,})"
                                     if r["redraws_ran_out_at"] is not None else "")
        a(f"| {r['tokens_per_micro']:,} | {first} | {plans} | {r['redrawn']:,} | "
          f"{'yes' if r['same_as_default_before_first_empty'] else 'NO'} ({r['compared']:,} plans) | "
          f"{f1(r['tokens_mean'], 0)} | {'' if r['tokens_min'] is None else format(r['tokens_min'], ',')} | "
          f"{f1(r['fill'], 3)} | "
          f"{share_cell(r, r['max_real_share'])} | {share_cell(r, r['max_synthetic_share'])} |")
    a("")
    if ph == "B":
        a(f"Caps: every real source at most {C['source_cap']} of a micro-step's tokens, the synthetic share at most "
          f"{C['synthetic_share']} in phase B.")
        a("")
    else:
        write_phase_c_rows(C, a)
    srcs = sorted({s for r in C["rows"] for s in r["token_shares"]} | {s for r in C["rows"] for s in r["sampler_shares"]})
    a("Shares of all planned tokens per source, with redraws, against the sampler's shares of tokens_per_micro (its "
      "budgets before the caps):")
    a("")
    a("| source | sampler share | " + " | ".join(f"planned share at {r['tokens_per_micro']:,}" for r in C["rows"]) + " |")
    a("|---|---|" + "---|" * len(C["rows"]))
    for s in srcs:
        a(f"| {s} | {C['rows'][0]['sampler_shares'].get(s, 0.0):.4f} | "
          + " | ".join(share_cell(r, r['token_shares'].get(s, 0.0)) for r in C["rows"]) + " |")
    a("")
    for r in C["rows"]:
        tr = r.get("first_empty_trace")
        if not tr:
            continue
        a(f"## The default sampler's first empty plan at {r['tokens_per_micro']:,} tokens per micro-step")
        a("")
        a(f"Plan {r['default_first_empty']:,} (0-based), replayed from the sampler state before it: \"{tr['error']}\". "
          f"Its passes of sampler.enforce_caps (the first holds the real sources against the total the synthetic "
          f"budget will complete, the second every cap of the micro-step), with the tokens per source before each "
          f"pass and their share of its total:")
        a("")
        a("| pass | real-source cap | synthetic share bound | tokens before | per source before (share) | tokens after | "
          "bodies dropped |")
        a("|---|---|---|---|---|---|---|")
        for k, c in enumerate(tr["calls"]):
            tot = sum(c["tokens"].values())
            per = "; ".join(f"{s} {v:,} ({v / tot:.4f})" for s, v in sorted(c["tokens"].items())) if tot else ""
            cap = (f"{c['cap']:.4f}" if not isinstance(c["cap"], dict) else
                   "; ".join(f"{s} {v:.4f}" for s, v in sorted(c["cap"].items())))
            a(f"| {k + 1} | {cap} | {c['synthetic_share']:.2f} | {tot:,} | {per} | "
              f"{sum(c['tokens_after'].values()):,} | {c['dropped']:,} |")
        a("")
        last = tr["calls"][-1]["bodies"] if tr["calls"] else {}
        a("Bodies before the last pass, tokens per body (a render pair counts as one body): "
          + "; ".join(f"{s} [{', '.join(str(x) for x in b)}]" for s, b in sorted(last.items())) + ".")
        a("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


def write_phase_c_rows(C, a):
    """The phase C part of a plan check's markdown (write_plan_check): the caps, the replay share and per budget key
    its subject-hours, persons, cap and largest share of a micro-step."""
    a(f"Caps (phase C, task S2.7): every replay source at most {C['source_cap']} and every unit at most "
      f"{C['unit_cap']} of a micro-step's tokens, the synthetic share at most {C['replay_share']} x "
      f"{C['synthetic_share']} = {C['replay_share'] * C['synthetic_share']:.4f}. Replay share of all planned tokens "
      f"(synthetic and replay sources; configured {C['replay_share']}): "
      + ", ".join(f"{'no plan' if r['replay_share'] is None else format(r['replay_share'], '.4f')} at "
                  f"{r['tokens_per_micro']:,}" for r in C["rows"]) + ".")
    a("")
    r0 = C["rows"][0]
    a("| budget key | store | subject-hours (train fold) | persons | cap | "
      + " | ".join(f"largest share of a micro-step at {r['tokens_per_micro']:,}" for r in C["rows"]) + " |")
    a("|---|---|---|---|---|" + "---|" * len(C["rows"]))
    for s in sorted(r0["caps"]):
        store = "stage-1 (replay)" if s in r0["replay_keys"] else "stage-2 (unit)"
        a(f"| {s} | {store} | {r0['hours'][s]:.3f} | {r0['persons'][s]} | {r0['caps'][s]:.2f} | "
          + " | ".join(share_cell(r, r['max_key_share'].get(s, 0.0)) for r in C["rows"]) + " |")
    a("")


def run_plan_check(args, P):
    """--plan-check: rows of plan_check at the configuration's tokens_per_micro and the FALLBACK_MICRO sizes, written to
    <out>/<stem>_plan_check.md and .json (no bench, no GPU). A stage-2 configuration's plans are those of phase C,
    on the stage-1 root (names.STAGE1, RELEASE) and the stage-2 root of its tier under its name tables."""
    cfg0 = CF.load(P["config"])
    sizes = [int(cfg0.mixture.tokens_per_micro)] + [int(t) for t, _ in FALLBACK_MICRO
                                                     if int(t) != int(cfg0.mixture.tokens_per_micro)]
    bs = BD.train_body_store(cfg0)
    phase = LP.STAGE2_PHASE if CF.stage2_on(cfg0) else "B"
    if phase == "B":
        ws = WS.WindowStore()
    else:
        from mdfm.train import names as NM
        ws = S.Stage2Stores(WS.WindowStore(WS.store_root(), tier="RELEASE", names=NM.STAGE1),
                            WS.WindowStore(WS.store_root(kind=WS.TIER_KIND[cfg0.tier]), tier=cfg0.tier,
                                           names=CF.name_tables(cfg0)))
    print(f"plan check of {P['config']}: {args.plan_check} phase {phase} plans at {sizes}, redraws {args.redraws}",
          flush=True)
    rows = plan_check(cfg0, bs, ws, int(args.plan_check), int(args.redraws), sizes, phase=phase)
    C = {"env": {"date": _dt.date.today().isoformat(), "moveatlas": moveatlas_commit(),
                 "config_sha256": CF.sha256(cfg0)},
         "config": P["config_rel"], "steps": int(args.plan_check), "redraws": int(args.redraws),
         "source_cap": float(cfg0.mixture.source_cap), "synthetic_share": float(cfg0.mixture.synthetic_share[1]),
         "rows": rows}
    if phase != "B":
        C.update(phase=phase, unit_cap=float(cfg0.stage2.unit_cap), replay_share=float(cfg0.stage2.replay_share),
                 stage2_names_sha256=CF.name_tables(cfg0).sha256())
    os.makedirs(P["out"], exist_ok=True)
    md = os.path.join(P["out"], f"{P['stem']}_plan_check.md")
    js = os.path.join(P["out"], f"{P['stem']}_plan_check.json")
    write_plan_check(C, md)
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(LP._finite(C), f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    for r in rows:
        print(plan_check_line(r), flush=True)
    print("wrote", md)
    return C


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=CONFIG, help="the configuration benched (default configs/stage1_pilot.yaml)")
    ap.add_argument("--out", default=OUT, help="folder of the report (default reports/bench)")
    ap.add_argument("--out-stem", default=DEFAULT_STEM,
                    help=f"file stem of the report, <out>/<stem>.md and .json (default {DEFAULT_STEM})")
    ap.add_argument("--no-write-config", action="store_true", help="never write the configuration file")
    ap.add_argument("--memory-gb", type=float, default=LP.MEMORY_CAP_GB,
                    help=f"caching-allocator cap of the capped passes, GB (default {LP.MEMORY_CAP_GB})")
    ap.add_argument("--skip-uncapped", action="store_true", help="no pass lifts the cap (step 7 runs capped only)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every pass's step count (quick checks)")
    ap.add_argument("--report-only", action="store_true",
                    help="rewrite <stem>.md from <stem>.json without measuring")
    ap.add_argument("--redraws", type=int, default=0,
                    help="draw again a plan or a step the caps leave empty, at most this many times (the opt-in "
                         "redraws of mdfm/data/sampler.py; default 0, the training's sampler; refused for the default "
                         "call)")
    ap.add_argument("--plan-check", type=int, default=0, metavar="STEPS",
                    help="no bench, no GPU: replay STEPS phase B plans at the configuration's tokens_per_micro and at "
                         "the fallback micro-batch sizes; writes <out>/<stem>_plan_check.md and .json")
    return ap.parse_args(argv)


def check_redraws(P, redraws):
    """The --redraws value of a call (ValueError: negative, or not 0 for the default call, whose pilot bench measures
    the pilot training's sampler)."""
    redraws = int(redraws)
    if redraws < 0:
        raise ValueError(f"--redraws {redraws}: a non-negative integer")
    if redraws and P["default"]:
        raise ValueError("--redraws is refused for the default call: the pilot bench (OD5) measures the sampler of "
                         "the pilot training")
    return redraws


def micro_probe(cfg1, tpm, mps, bs, ws, synth, dev, P, steps, cap_gb, probe_b=False, redraws=0):
    """One row of the micro-batch table (step 3): timed phase A steps at (tpm tokens, mps micro-steps), or the
    out-of-memory outcome under the cap. probe_b (any call but the default one): then also
    refined phase B steps (refine_every 1, the pass with the highest peak of the pilot bench); the row fits when both
    ran under the cap, its peaks are the larger of the two and its tokens/s those of phase A. redraws: --redraws."""
    c = CF.check(replace(cfg1, mixture=replace(cfg1.mixture, tokens_per_micro=int(tpm)),
                         schedule=replace(cfg1.schedule, micro_per_step=int(mps))))
    row = {"tokens_per_micro": tpm, "micro_per_step": mps}
    tr, where = None, "phase A"
    try:
        tr = with_redraws(LP.Trainer(c, bs, ws, synth, dev, P["run_dir"], echo=None, memory_cap_gb=cap_gb), redraws)
        s = run_pass(tr, "A", steps["micro_probe"], True, label=f"micro {tpm} x {mps}")
        row.update(outcome="fits", tokens_per_s=s["tokens_per_s"], peak_allocated_gb=s["peak_allocated_gb"],
                   peak_reserved_gb=s["peak_reserved_gb"])
        if probe_b:
            where = "phase B, refined"
            sb = run_pass(tr, "B", steps["micro_probe"], False, refine_every=1,
                          label=f"micro {tpm} x {mps}, phase B refined")
            row.update(peak_allocated_gb=max(s["peak_allocated_gb"], sb["peak_allocated_gb"]),
                       peak_reserved_gb=max(s["peak_reserved_gb"], sb["peak_reserved_gb"]),
                       peak_reserved_gb_b=sb["peak_reserved_gb"])
    except torch.OutOfMemoryError as e:
        row = {"tokens_per_micro": tpm, "micro_per_step": mps}
        row.update(outcome=f"out of memory under the {cap_gb:g} GB cap" + (f" ({where})" if probe_b else ""))
        print(f"  micro {tpm} x {mps}: out of memory in {where} ({str(e)[:80]})", flush=True)
    if tr is not None:
        tr._close_prefetch()
    tr = None
    gc.collect()
    torch.cuda.empty_cache()
    return row


def micro_table(probe, extended, cap_gb):
    """(table, kept row, fallback measured) of step 3: probe(tokens, micro-steps) gives the row of each candidate of
    MICRO_CANDIDATES; a row fits when it ran under the cap with its peak reserved memory within TARGET_GB, and the
    fitting row with the most useful tokens per second is kept. extended (any call but the default one): when no
    candidate fits, FALLBACK_MICRO is probed too (rows marked fallback). SystemExit when no row fits (the default
    call stopped there before too, on the empty list)."""
    table = [probe(tpm, mps) for tpm, mps in MICRO_CANDIDATES]

    def fitting():
        return [r for r in table if r["outcome"] == "fits" and r["peak_reserved_gb"] <= TARGET_GB + 1e-9]
    ok = fitting()
    fallback = bool(extended) and not ok
    if fallback:
        print("  no candidate fits under the cap: the fallback micro-batch", flush=True)
        for tpm, mps in FALLBACK_MICRO:
            row = probe(tpm, mps)
            row["fallback"] = True
            table.append(row)
        ok = fitting()
    if not ok:
        raise SystemExit(f"no micro-batch fits under the {cap_gb:g} GB cap and the {TARGET_GB:g} GB target")
    return table, max(ok, key=lambda r: r["tokens_per_s"]), fallback


def main(argv=None):
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        P = bench_paths(args.config, args.out, args.out_stem, write_config=not args.no_write_config)
        redraws = check_redraws(P, args.redraws)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.report_only:
        with open(P["json"], encoding="utf-8") as f:
            write_report(json.load(f), P["md"])
        print("wrote", P["md"], "from", os.path.basename(P["json"]))
        return
    if args.plan_check:
        if args.plan_check < 0:
            raise SystemExit(f"--plan-check {args.plan_check}: a positive number of plans")
        run_plan_check(args, P)
        return
    steps = {k: max(2, int(round(v * args.scale))) for k, v in STEPS.items()}
    cap_gb = float(args.memory_gb)
    extended = not P["default"]          # the additions of the options (module docstring); never in the default call
    cfg0 = CF.load(P["config"])
    if CF.stage2_on(cfg0):
        raise SystemExit(f"{P['config']} is a stage-2 configuration: this bench measures phases A and B; its plan "
                         "check runs (--plan-check), its bench is task S2.9's (scripts/stage2_budget.py)")
    enc_kw = encoder_kwargs(cfg0)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    R = {"env": {"date": _dt.date.today().isoformat(), "torch": torch.__version__, "cuda": torch.version.cuda,
                 "gpu": props.name, "gpu_mib": props.total_memory // 2 ** 20, "moveatlas": moveatlas_commit(),
                 "config_sha256_before": CF.sha256(cfg0), "memory_cap_gb": cap_gb,
                 "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")},
         "steps": steps, "max_reserved_gb": float(cfg0.schedule.max_reserved_gb)}
    if not (P["default"] and P["write_config"]):     # the default call's result keeps its former keys
        R["bench"] = bench_meta(P, enc_kw, args.skip_uncapped, redraws)
    print(f"bench of {P['config']} on {props.name}", flush=True)
    bs = BD.train_body_store(cfg0)
    ws = WS.WindowStore()
    synth = BD.synth_sampler(cfg0, bs, device=dev)
    R["env"]["body_takes"] = len(bs.takes)

    # 1. histogram
    print("1. window-length histogram", flush=True)
    counts = {ph: plan_counts(cfg0, bs, ws, ph, PLAN_STEPS, LP.phase_seed(cfg0, ph), redraws) for ph in ("A", "B")}
    longest = max_layout_tokens(float(cfg0.mixture.win_s))
    R["hist"] = {"longest_layout": longest}
    for ph in ("A", "B"):
        rows, flat = histogram(counts[ph])
        lng = [max(c) for c in counts[ph]]
        R["hist"][ph] = {"rows": rows, "windows": int(len(flat)), "median": float(np.median(flat)),
                         "per_micro": len(flat) / PLAN_STEPS, "longest_median": float(np.median(lng)),
                         "longest_min": int(min(lng)), "fit_128": int(sum(x <= 128 for x in lng)),
                         "fit_640": int(sum(x <= 640 for x in lng))}

    # 2. rows
    print("2. bucket rows", flush=True)
    table, big = choose_rows(counts["A"], counts["B"], dev, max(longest, max(max(c) for c in counts["A"] + counts["B"])),
                             enc_kw=enc_kw)
    rows = tuple(SMALL_ROWS) + (big,)
    R["rows"] = {"table": table, "chosen": list(rows)}
    cfg1 = replace(cfg0, mixture=replace(cfg0.mixture, rows=rows))

    # 3. micro-batch
    print("3. micro-batch", flush=True)

    def probe(tpm, mps):
        return micro_probe(cfg1, tpm, mps, bs, ws, synth, dev, P, steps, cap_gb, probe_b=extended, redraws=redraws)
    mtable, best, fallback = micro_table(probe, extended, cap_gb)
    R["micro"] = {"table": mtable, "chosen": [best["tokens_per_micro"], best["micro_per_step"]]}
    if fallback:
        R["micro"]["fallback"] = True
    if extended:
        R["micro"]["probe_b"] = True
    cfg2 = CF.check(replace(cfg1, mixture=replace(cfg1.mixture, tokens_per_micro=int(best["tokens_per_micro"])),
                            schedule=replace(cfg1.schedule, micro_per_step=int(best["micro_per_step"]))))

    # 4. phase A
    print("4. phase A", flush=True)
    tr = with_redraws(LP.Trainer(cfg2, bs, ws, synth, dev, P["run_dir"], echo=None, memory_cap_gb=cap_gb), redraws)
    R["env"].update(parameters=int(sum(p.numel() for p in tr.model.parameters())), optimizer=tr.opt_kind,
                    fused_smoke=LP.fused_smoke(dev))
    comp = components(tr.model)
    print("  parameters: " + ", ".join(f"{k} {v:,}" for k, v in comp.items()), flush=True)
    if extended:
        R["components"] = comp
    passes = [run_pass(tr, "A", steps["a_timed"], True, label="A timed")]
    a0 = run_pass(tr, "A", steps["a_rate"], False, label="A prefetch 0")
    tr.set_prefetch(PREFETCH)
    a2 = run_pass(tr, "A", steps["a_rate"], False, label=f"A prefetch {PREFETCH}")
    passes += [a0, a2]
    prefetch = PREFETCH if a2["tokens_per_s"] > a0["tokens_per_s"] else 0
    s_a = a2 if prefetch else a0
    R["prefetch"] = prefetch

    # 5. phase B
    print("5. phase B", flush=True)
    tr.set_prefetch(0)
    passes.append(run_pass(tr, "B", steps["b_timed"], True, refine_every=0, label="B timed, unrefined"))
    passes.append(run_pass(tr, "B", steps["b_timed_refined"], True, refine_every=1, label="B timed, refined"))
    tr.set_prefetch(prefetch)
    b0 = run_pass(tr, "B", steps["b_rate0"], False, refine_every=0, label=f"B prefetch {prefetch}, unrefined")
    b1 = run_pass(tr, "B", steps["b_rate1"], False, refine_every=1, label=f"B prefetch {prefetch}, every step refined")
    t0, t1 = b0["s_per_step"], b1["s_per_step"]
    N = max(1, math.ceil((t1 - t0) / (REFINE_OVERHEAD * t0) - 1e-9))
    nsteps = max(N, int(math.ceil(steps["b_rateN"] / N)) * N)
    bN = run_pass(tr, "B", nsteps, False, refine_every=N, label=f"B prefetch {prefetch}, refine every {N}")
    passes += [b0, b1, bN]
    R["passes"] = passes
    R["refine"] = {"t0": t0, "t1": t1, "N": N, "tN": bN["s_per_step"], "tN_tokens_per_s": bN["tokens_per_s"],
                   "steps": int(cfg2.schedule.refine_steps), "points": int(cfg2.schedule.refine_points),
                   "overhead_rule": REFINE_OVERHEAD}

    # 6. budget
    rows_b, chosen = budget(cfg2, s_a, bN)
    R["budget"] = {"rows": rows_b, "chosen": chosen, "s_a": s_a["s_per_step"], "s_b": bN["s_per_step"],
                   "wall_budget_h": WALL_BUDGET_H}
    if extended:
        R["budget"]["configured"] = budget_row(cfg2, s_a, bN, float(cfg0.schedule.tokens_total))
        R["budget"]["design"] = design_budget(cfg2, s_a, bN)
    tb = tr.sched.tallies["B"]
    tot_b = sum(tb["tokens"].values())
    share_b = tb["tokens"].get(S.SYNTHETIC, 0) / tot_b if tot_b else float("nan")
    cum = CF.cumulative_synthetic_share(cfg2, tokens=(s_a["tokens_per_micro"], bN["tokens_per_micro"]),
                                        shares=(1.0, share_b))
    R["synthetic_share"] = {"tokens_a": s_a["tokens_per_micro"], "tokens_b": bN["tokens_per_micro"],
                            "share_b": share_b, "share_a": float(cfg2.schedule.share_a), "cumulative": cum,
                            "cap": float(cfg2.mixture.synthetic_cap), "b_tallies": tb}

    # 7. the cap lifted
    print("7. allocator cap", flush=True)
    capped = run_pass(tr, "B", steps["uncapped"], False, refine_every=1, label="B refined, capped")
    R["passes"].append(capped)
    capped_passes = [p for p in R["passes"]]
    R["uncapped"] = {"capped_reserved_gb": capped["peak_reserved_gb"], "capped_allocated_gb": capped["peak_allocated_gb"],
                     "capped_tokens_per_s": capped["tokens_per_s"], "steps": steps["uncapped"] + WARMUP}
    if args.skip_uncapped:
        R["uncapped"].update(outcome="skipped")
    else:
        torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
        gc.collect()
        torch.cuda.empty_cache()
        try:
            unc = run_pass(tr, "B", steps["uncapped"], False, refine_every=1, label="B refined, cap lifted")
            R["passes"].append(unc)
            R["uncapped"].update(outcome="completed", reserved_gb=unc["peak_reserved_gb"],
                                 allocated_gb=unc["peak_allocated_gb"], tokens_per_s=unc["tokens_per_s"])
        except (LP.MemoryBudgetError, torch.OutOfMemoryError) as e:
            done = int(tr.sched.step)
            R["uncapped"].update(outcome=type(e).__name__, stopped_after=done,
                                 reserved_gb=torch.cuda.max_memory_reserved(dev) / LP.GB,
                                 allocated_gb=torch.cuda.max_memory_allocated(dev) / LP.GB, tokens_per_s=None)
            print(f"  B refined, cap lifted: {type(e).__name__} after {done} optimizer steps, peak reserved "
                  f"{R['uncapped']['reserved_gb']:.2f} GB", flush=True)
            tr._close_prefetch()
            tr.sampler = None
            tr.opt.zero_grad(set_to_none=True)
            gc.collect()
        LP.cap_memory(dev, cap_gb)
        torch.cuda.empty_cache()
    R["peak"] = {"allocated_gb": max(p["peak_allocated_gb"] for p in capped_passes),
                 "reserved_gb": max(p["peak_reserved_gb"] for p in capped_passes)}
    tr._close_prefetch()

    # the configuration
    final = replace(cfg2, schedule=replace(cfg2.schedule, tokens_total=float(chosen["tokens_total"]),
                                           refine_every=int(N)))
    final = CF.check(final)
    a_steps, b_steps = CF.step_counts(final)
    rep = rel(P["md"])
    tag = "[B4.18]" if P["default"] else "[T15 bench]"
    values = {
        ("mixture", "tokens_per_micro"): (int(final.mixture.tokens_per_micro),
                                          f"{tag} useful tokens per micro-step, {rep} section 3"),
        ("mixture", "rows"): (list(final.mixture.rows), f"{tag} bucket row lengths, tokens, {rep} section 2"),
        ("schedule", "micro_per_step"): (int(final.schedule.micro_per_step),
                                         f"{tag} micro-steps per optimizer step, {rep} section 3"),
        ("schedule", "tokens_total"): (float(final.schedule.tokens_total),
                                       f"{tag} {rep} section 7; step counts A {a_steps:,}, B {b_steps:,} "
                                       f"(config.step_counts)"),
        ("schedule", "refine_every"): (int(final.schedule.refine_every),
                                       f"{tag} phase B refinement every N-th optimizer step, {rep} section 6"),
    }
    status, header = yaml_status_header(P, cfg0, R["env"]["date"])
    new_text = update_yaml(P["config"], values, status, header)
    want = replace(final, status=status)
    R["env"]["config_sha256_after"] = check_config_text(new_text, want, P)
    R["written"] = [
        {"key": "mixture.rows", "before": _fmt(list(cfg0.mixture.rows)), "after": _fmt(list(final.mixture.rows)),
         "rule": "least encoder time per useful token (section 2)"},
        {"key": "mixture.tokens_per_micro", "before": f"{cfg0.mixture.tokens_per_micro:,}",
         "after": f"{final.mixture.tokens_per_micro:,}", "rule": "fastest candidate under the cap (section 3)"},
        {"key": "schedule.micro_per_step", "before": str(cfg0.schedule.micro_per_step),
         "after": str(final.schedule.micro_per_step), "rule": "32,768 tokens per optimizer step (section 3)"},
        {"key": "schedule.tokens_total", "before": f"{cfg0.schedule.tokens_total / 1e9:.2f} billion",
         "after": f"{final.schedule.tokens_total / 1e9:.2f} billion",
         "rule": f"largest budget within {WALL_BUDGET_H:.0f} GPU-hours (section 7)"},
        {"key": "schedule.refine_every", "before": str(cfg0.schedule.refine_every),
         "after": str(final.schedule.refine_every), "rule": f"refinement at most {REFINE_OVERHEAD} of a step (section 6)"},
    ]
    if extended and os.path.exists(PILOT_JSON):
        with open(PILOT_JSON, encoding="utf-8") as f:
            R["pilot_ref"] = rates(json.load(f))
    os.makedirs(P["out"], exist_ok=True)
    write_report(R, P["md"])
    with open(P["json"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(LP._finite(R), f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    finish_config(P)
    if not P["write_config"]:
        print("configuration left unchanged (--no-write-config)")
    else:
        print(f"wrote {P['config']}: rows {list(final.mixture.rows)}, {final.mixture.tokens_per_micro} x "
              f"{final.schedule.micro_per_step}, tokens_total {final.schedule.tokens_total:.3g}, steps A {a_steps}, "
              f"B {b_steps}, refine_every {N}")
    print("wrote", P["md"])


if __name__ == "__main__":
    main()
