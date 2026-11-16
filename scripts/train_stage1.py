"""B4.18: stage-1 training of the 60M pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 5; the loop
is mdfm/train/loop.py).

Phases: A, synthetic only (the utility test of OD9: its last checkpoint is the simulator-only model that the
G3 line reads), then B, the mixture of synthetic and real healthy windows. --phase A runs phase A alone,
--phase B continues from a checkpoint at the end of phase A (--resume; the loop refuses phase B otherwise),
--phase AB runs both. --max-steps shortens the call (a checkpoint at the stop; --resume continues it). With
--resume and no --run-dir the run continues in the checkpoint's folder.

Init-from (task J2b; loop.Trainer.init_from): --phase B --init-from <checkpoint> --run-dir <new folder> starts phase
B of a new run, under --config (any configuration whose model fits the checkpoint's weights), from a checkpoint
written at the end of phase A of another run: model weights, model EMA, the optimizer's states of the parameters
the checkpoint holds (new parameters, such as a JEPA predictor, start fresh, and a JEPA target copies the loaded
tokenizer and encoder), the RNG states and phase A's token tallies; the schedule at phase B step 0 of the config's
step counts, the phase B sampler fresh, no dev numbers at the start: what --phase B --resume of the same checkpoint
does under the checkpoint's own config (v1's phase B: handoff 4, section 2j), so that a run under that config, or
under one that differs by a disabled jepa section or the name, reproduces it bit for bit
(tests/test_train_stage1.py). The run folder must hold no checkpoint. The lineage (source path, SHA-256, step,
config SHA-256) goes to log.jsonl ('init_from') and to extra.init_from of the provenance of every checkpoint of
the run. --init-from with --resume (the form the GPU queue's crash resume gives an init-from job,
scripts/gpu_queue.py) resumes the checkpoint after checking that it descends from the --init-from checkpoint.
--jepa-grad-steps N (a config with the jepa section enabled): the first N optimizer steps of phase B log g_J and
g_R, the encoder gradient norms of the JEPA term and of the rest ('jepa_grad' events, loop.py), which
scripts/jepa_lambda.py reads.

Phase C (task S2.7; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.6; loop.Trainer.init_stage2): a config with
the stage2 section enabled trains a stage-2 run, phase C only. --phase C --init-stage2 <checkpoint> --run-dir <new
folder> starts it from a checkpoint written at the end of phase B of a stage-1 run, migrated to the config's name
tables in memory (train/migrate.py; the plan's lineage goes to log.jsonl, 'init_stage2', and to extra.migration of
the provenance of every checkpoint of the run); --phase C --resume <checkpoint> continues it, and with
--init-stage2 also checks that the resumed checkpoint descends from that source under the same plan (the form of a
crash resume). The run reads two window stores (sampler.Stage2Stores): the stage-1 root (window_store.store_root(),
MDFM_WINDOW_STORE, read under names.STAGE1 at tier RELEASE: the healthy replay) and the stage-2 root of the config's
tier (MDFM_WINDOW_STORE_S2 for RELEASE) under the config's name tables; the layouts are the stage-1 families and the
S2a families (layouts.load_stage2_layouts), so that the replay renders every family a stage-2 layout has; the dev
windows are collated under the config's name tables. A config whose status line starts with DRAFT is refused for
phase C (configs/stage2_s2a.yaml stays DRAFT until the start checkpoint and the budget of S2.9 are written into it).

Reads: the AMASS body store (split train; split dev for the dev numbers), the window store (phase B), the
SMPL+H model files of MoveAtlas (read-only) and the reports that train/provenance.py reads. Writes the run
folder only: log.jsonl, ckpt_<phase>_<step>.pt with its .sha256, provenance.json, config.json, config.sha256,
persons/<source>.txt. Default folder: loop.default_run_dir (MDFM_RUNS, else <store>/runs when <store> exists,
else cache/runs, gitignored); never under reports/, never the lockbox.

Dev numbers (the dev hook of the loop, event 'dev' of log.jsonl): DEV_WINDOWS windows of the AMASS dev archives
rendered once by the factory (seed DEV_SEED, the run's layout mixture), collated into micro-steps of at most
tokens_per_micro tokens; loop.evaluate of the model and of its EMA at the start, every dev_every steps, at each
phase end and at the stop of a shortened call, with the fields the dev alarms of the loop read (spread_ratio,
joint_error_z0_mm; loop.ALARMS) and the constant mean-pose prior on the same windows. The dev persons never
train: the samplers draw the train split, and the provenance lists the dev split as held out.

Per-unit dev numbers (task S2.8; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7): for a stage-2 config the dev
hook also reads, per unit of stage2.units, UNIT_DEV_WINDOWS fixed windows of the unit's val fold (UNIT_HELD_OUT_FOLD,
the spec's per-unit val fold) and as many of its training persons, from the stage-2 store (unit_windows: seeded
draws, a person uniform, a recording by span), collated under the config's tables; loop.evaluate of the model (not
the EMA) on each gives the kernel NLL per window (the sum of losses.KERNEL_TERMS, the only early warning where no body
truth exists), its terms, the loss and the share of decoded knots outside the plausibility table
(losses.plausibility_table; evaluate's limits). A unit without a val fold (weargait_pd:PD and the stroke patients,
whose held-out persons form the dev folds that G3 and G4 read) logs None as its held-out entry: no training decision
reads a gate's development persons, and the unit stays out of the alarm until the owner decides (a val fold carved
from its training persons in the stage-2 store is the other way). The 'model' entry of the 'dev' event holds them
under units, with unit_nll_rise, the number of units whose held-out NLL rises over loop.UNIT_NLL_READS dev reads of
the phase while their train NLL falls (loop.unit_nll_rising; a resumed run reads the earlier reads back from
log.jsonl), which the loop's alarm unit_nll_rise reads (loop.ALARMS). The hook reads the val windows for evaluation
only: they never train (the sampler draws the train fold) and the provenance lists them as held out. For a stage-2
config whose losses.hsic_strata names a table, the stage-2 store is opened with it (stage2_stores; data/strata.py),
so the windows carry their HSIC strata.

Alarms (reports/bench/phaseA_diagnosis.md, fix 8): the loop logs 'alarm' events (decoder attention entropy,
beta error, logvar_z at the log intervals; spread ratio and the z = 0 error at dev events) and, with
schedule.stop_on_alarm, writes a checkpoint and raises loop.AlarmError. The script prints every alarm of the
run, verifies the last checkpoint and exits with status 3 after an alarm stop.

After the run the last checkpoint is re-read with checkpoint.load (file, config, person-list, records and
lockbox hashes, the person guard, the state_dict audit) and the result is printed and logged ('verify'), with
the report mismatches of the provenance (the stores against their reports) and the loss of the first and last
log intervals.

Usage (mdfm environment; PREFETCH from reports/bench/stage1_pilot.md):
  <conda>/envs/mdfm/python.exe scripts/train_stage1.py --phase A --max-steps 1000
  <conda>/envs/mdfm/python.exe scripts/train_stage1.py --phase B
      --init-from cache/runs/phaseA_root_full/ckpt_A_010681.pt --config configs/stage1_pilot_jepa.yaml
      --run-dir cache/runs/jepa_smoke --max-steps 200 --prefetch 2 --memory-cap-gb 14.5 --jepa-grad-steps 50
"""
import argparse
import json
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.data import collate as CO  # noqa: E402
from mdfm.data import sampler as SM  # noqa: E402
from mdfm.data import strata as ST  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.factory import layouts as LY  # noqa: E402
from mdfm.factory import synth as SY  # noqa: E402
from mdfm.train import bodies as BD  # noqa: E402
from mdfm.train import checkpoint as CK  # noqa: E402
from mdfm.train import config as CF  # noqa: E402
from mdfm.train import loop as LP  # noqa: E402
from mdfm.train import losses as LS  # noqa: E402
from mdfm.train import names as NM  # noqa: E402

# read by the CUDA caching allocator when CUDA starts (after this line), not at import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", LP.ALLOC_CONF)

CONFIG = os.path.join(TREE, "configs", "stage1_pilot.yaml")
DEV_WINDOWS = 96
DEV_SEED = 20260929
PREFETCH = 2               # micro-steps sampled ahead (reports/bench/stage1_pilot.md: faster than 0)
UNIT_DEV_WINDOWS = 32      # [cfg] fixed windows per stage-2 unit and part (held out, train) of the dev hook (task S2.8)
UNIT_PARTS = ("held_out", "train")
UNIT_HELD_OUT_FOLD = "val"     # the fold of a unit's held-out windows (spec section 4.7: the unit's val fold)


def collate_kw(cfg):
    """The collate arguments of a config's name tables: none while its names section is at its defaults (the
    stage-1 call), else the section's sources and nuisance vocabularies (loop.Trainer.collate_kw)."""
    if cfg.names == CF.NamesSection():
        return {}
    nm = CF.name_tables(cfg)
    return {"sources": tuple(nm.sources), "nuis_fields": nm.nuis_fields()}


def stage2_stores(cfg):
    """sampler.Stage2Stores of a stage-2 config (module docstring, Phase C): the stage-1 root under names.STAGE1 at
    tier RELEASE and the stage-2 root of the config's tier under its name tables, with the HSIC strata table of
    losses.hsic_strata when the config names one (data/strata.load, its SHA-256 checked; task S2.8)."""
    replay = WS.WindowStore(WS.store_root(), tier="RELEASE", names=NM.STAGE1)
    strata = ST.load(cfg.losses.hsic_strata, cfg.losses.hsic_strata_sha256) if str(cfg.losses.hsic_strata) else None
    stage2 = WS.WindowStore(WS.store_root(kind=WS.TIER_KIND[cfg.tier]), tier=cfg.tier, names=CF.name_tables(cfg),
                            strata=strata)
    return SM.Stage2Stores(replay, stage2)


def micro_batches(cfg, wins, seed, kw=None):
    """The windows collated into micro-steps of at most tokens_per_micro tokens, in order (micro-step i seeded seed +
    i); kw: collate_kw(cfg)."""
    groups, cur, tok = [], [], 0
    for w in wins:
        t = LY.window_tokens(w)
        if cur and tok + t > int(cfg.mixture.tokens_per_micro):
            groups.append(cur)
            cur, tok = [], 0
        cur.append(w)
        tok += t
    if cur:
        groups.append(cur)
    ccfg = CO.collate_config(cfg)
    return [CO.collate(g, cfg=ccfg, gen=seed + i, **(kw or {})) for i, g in enumerate(groups)]


def dev_batches(cfg, synth, n=DEV_WINDOWS, seed=DEV_SEED, kw=None):
    """Collated micro-steps of n dev windows rendered once (module docstring); kw: collate_kw(cfg)."""
    gen = np.random.default_rng(seed)
    bodies, specs = synth.draw(gen, n, mixture=dict(cfg.mixture.layouts), win_s=float(cfg.mixture.win_s))
    wins = synth.render(bodies, specs, gen)
    return micro_batches(cfg, wins, seed, kw)


def unit_windows(store, source, held_out, n, seed, win_s, min_s):
    """n fixed windows of one stage-2 unit (module docstring, Per-unit dev numbers): of its val fold
    (UNIT_HELD_OUT_FOLD; never a dev fold, which a gate reads) or of its training persons, from the stage-2 store's
    recordings of `source` of at least min_s; a person drawn uniformly, one of the person's recordings with
    probability proportional to its span, a span of min(win_s, span) s at a uniform start, from numpy
    default_rng(seed). ([], fold names) when the unit holds no such recording."""
    recs = store.recordings
    fold = recs["fold"] == (UNIT_HELD_OUT_FOLD if held_out else "train")
    g = recs[(recs["source"] == source) & fold & (recs["span_s"] >= float(min_s) - 1e-9)]
    folds = sorted(set(g["fold"]))
    if not len(g):
        return [], folds
    rng = np.random.default_rng(seed)
    persons = sorted(set(g["person_id"]))
    out = []
    for _ in range(int(n)):
        p = persons[int(rng.integers(len(persons)))]
        rows = g[g["person_id"] == p]
        span = np.asarray(rows["span_s"], np.float64)
        j = int(rng.choice(len(rows), p=span / span.sum())) if len(rows) > 1 else 0
        sp, st0 = float(span[j]), float(rows["span_t0"].iloc[j])
        dur = min(float(win_s), sp)
        t0 = st0 + (float(rng.uniform(0.0, sp - dur)) if sp > dur else 0.0)
        out.append(store.window(int(rows.index[j]), t0, dur))
    return out, folds


class DevHook:
    """The dev hook (module docstring): loop.evaluate of the model and the EMA on fixed dev micro-steps; layouts:
    the families of the renderer (default layouts.load_layouts(); a stage-2 config's: load_stage2_layouts). stores:
    the run's sampler.Stage2Stores for a stage-2 config (module docstring, Per-unit dev numbers), None otherwise;
    unit_windows: the windows of each unit and part."""

    def __init__(self, cfg, device, layouts=None, stores=None, unit_windows_n=UNIT_DEV_WINDOWS):
        store = BD.dev_body_store(cfg)         # the open store's dev folds for an open run (step M4)
        synth = BD.synth_sampler(cfg, store, device=device, layouts_=layouts)
        kw = collate_kw(cfg)
        self.mbs = dev_batches(cfg, synth, kw=kw)
        self.windows = sum(mb.batch.n_windows for mb in self.mbs)
        del synth
        self.units, self.hist = None, None
        if stores is not None and CF.stage2_on(cfg):
            self.units = unit_dev_sets(cfg, stores.stage2, kw, unit_windows_n)
            self.limits = LS.joint_limits(path=NM.resolve(str(cfg.losses.plausibility_table)))

    def __call__(self, trainer, event):
        out = {"windows": self.windows, "micro_steps": len(self.mbs)}
        for name, model in (("model", trainer.model), ("ema", trainer.ema)):
            r = LP.evaluate(model, self.mbs, trainer.sched.in_phase, device=trainer.device, loss_cfg=trainer.loss_cfg,
                            amp=trainer.amp, seed=DEV_SEED)
            out[name] = {"loss": r["terms"]["per_window"], "joint_error_mm": r["joint_error_mm"],
                         **{k: r[k] for k in DEV_FIELDS},
                         "terms": {k: r["terms"][k] for k in ("sup_rot", "sup_pos", "sup_root", "sup_root_xz", "sup_root_y",
                                                              "sup_root_rate", "sup_root_rate_u", "sup_root_rate_yaw",
                                                              "sup_deriv", "imu_nll", "pose2d_nll", "pose3d_nll", "kl_z",
                                                              "plausibility")}}
        m, e = out["model"], out["ema"]
        trainer.echo(f"[{trainer.sched.in_phase} {trainer.sched.step:>6}] dev ({event}): loss {m['loss']:.4g}, "
                     f"joint error {_mm(m['joint_error_mm'])} (z = 0 {_mm(m['joint_error_z0_mm'])}, true skeleton "
                     f"{_mm(m['joint_error_true_skeleton_mm'])}, prior {_mm(m['prior_joint_error_mm'])}), root "
                     f"{_num(m['root_orient_deg'])} deg, spread ratio {_num(m['spread_ratio'], 3)}, root speed ratio "
                     f"{_num(m['root_speed_ratio'], 3)}, heading gain {_num(m['heading_gain'], 3)}; EMA "
                     f"{e['loss']:.4g}, {_mm(e['joint_error_mm'])}")
        if self.units is not None:
            self.unit_numbers(trainer, m)
        return out

    def unit_numbers(self, trainer, rec):
        """The per-unit dev numbers of the model into rec (the 'model' entry; module docstring, Per-unit dev
        numbers): units {unit: {held_out_folds, held_out, train}}, unit_nll_rise (the count of loop.unit_nll_rising
        over the dev reads of this run's phase, read back from log.jsonl at the first call) and unit_nll_rise_units."""
        if self.hist is None:
            self.hist = unit_history(trainer.log_path, trainer.sched.in_phase, trainer.sched.step, self.units)
        units = {}
        for u, sets in self.units.items():
            units[u] = {"held_out_folds": sets["folds"]}
            for part in UNIT_PARTS:
                mbs = sets[part]
                if not mbs:
                    units[u][part] = None
                    continue
                r = LP.evaluate(trainer.model, mbs, trainer.sched.in_phase, device=trainer.device,
                                loss_cfg=trainer.loss_cfg, amp=trainer.amp, seed=DEV_SEED, limits=self.limits)
                units[u][part] = {"windows": r["windows"], "nll": sum(r["terms"][k] for k in LS.KERNEL_TERMS),
                                  "terms": {k: r["terms"][k] for k in LS.KERNEL_TERMS},
                                  "loss": r["terms"]["per_window"],
                                  "outside_limits_share": r["outside_limits_share"],
                                  "outside_limits_knots": r["outside_limits_knots"]}
            nll = [None if units[u][p] is None else units[u][p]["nll"] for p in UNIT_PARTS]
            self.hist.setdefault(u, []).append(tuple(nll))
        rising = LP.unit_nll_rising(self.hist)
        rec.update(units=units, unit_nll_rise=len(rising), unit_nll_rise_units=rising)
        trainer.echo(f"[{trainer.sched.in_phase} {trainer.sched.step:>6}] units: " + "; ".join(
            f"{u} NLL {_num((units[u]['held_out'] or {}).get('nll'), 4)} held out, "
            f"{_num((units[u]['train'] or {}).get('nll'), 4)} train, outside limits "
            f"{_num((units[u]['held_out'] or {}).get('outside_limits_share'), 4)}" for u in units)
                     + (f"; rising {rising}" if rising else ""))
        return rec


def unit_dev_sets(cfg, store, kw=None, n=UNIT_DEV_WINDOWS, seed=DEV_SEED):
    """{unit: {folds, held_out, train}} of a stage-2 config (module docstring, Per-unit dev numbers): per unit of
    stage2.units, the collated micro-steps of n fixed windows (unit_windows) of its val fold ([] for a unit without
    one) and of its training persons from the stage-2 store, and the held-out fold names. Under
    losses.plausibility_windows 'synthetic_replay' their windows carry no plausibility term (Targets.plausibility_on
    False), as in training."""
    out = {}
    off = cfg.losses.plausibility_windows == "synthetic_replay"
    for k, u in enumerate(cfg.stage2.units):
        src = str(u).split(":", 1)[0]
        sets = {}
        for i, part in enumerate(UNIT_PARTS):
            wins, folds = unit_windows(store, src, part == "held_out", n, [seed, k, i], cfg.mixture.win_s,
                                       cfg.mixture.min_s)
            if part == "held_out":
                sets["folds"] = folds
            mbs = micro_batches(cfg, wins, seed + 1000 * (k + 1) + 100 * i, kw) if wins else []
            if off:
                for mb in mbs:
                    mb.targets.plausibility_on = torch.zeros(mb.batch.n_windows, dtype=torch.bool)
            sets[part] = mbs
        out[u] = sets
    return out


def unit_history(log_path, phase, step, units):
    """{unit: [(held-out NLL, train NLL), ...]} of the 'dev' events of a run folder's log.jsonl in `phase` before
    `step` (the last event of each step), the reads a resumed run continues (unit_nll_rising)."""
    hist = {u: [] for u in units}
    if not os.path.exists(log_path):
        return hist
    by_step = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("event") != "dev" or r.get("phase") != phase or int(r.get("step", -1)) >= int(step):
                continue
            us = ((r.get("dev") or {}).get("model") or {}).get("units")
            if us:
                by_step[int(r["step"])] = us
    for s in sorted(by_step):
        for u in units:
            e = by_step[s].get(u)
            if e is not None:
                hist[u].append(tuple(None if e.get(p) is None else e[p].get("nll") for p in UNIT_PARTS))
    return hist


DEV_FIELDS = ("joint_error_true_skeleton_mm", "joint_error_z0_mm", "prior_joint_error_mm", "root_orient_deg",
              "spread_ratio", "root_speed_ratio", "root_speed_ratio_far", "heading_gain", "heading_corr",
              "heading_err_deg", "heading_zero_deg", "lateral_slope", "root_points")


def _num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _mm(x):
    return "n/a" if x is None else f"{x:.1f} mm"


def run_folder(run_dir, resume):
    """The run folder: run_dir when given, else the resumed checkpoint's folder, else None (the loop's
    default_run_dir)."""
    if run_dir is not None or resume is None:
        return run_dir
    return os.path.dirname(os.path.abspath(resume))


def verify(trainer, cfg, fallback=None):
    """checkpoint.load of the last checkpoint of this call (fallback: the resumed one when the call wrote none);
    the provenance's report mismatches; loss of the first and last log intervals."""
    path = trainer.checkpoints[-1] if trainer.checkpoints else fallback
    if path is None:
        return None
    ck = CK.load(path, cfg=cfg)
    logs = [h for h in trainer.history if h.get("event") == "log"]
    with open(path + ".sha256", encoding="utf-8") as f:
        sha = f.read().split()[0]
    out = {"path": path, "sha256": sha, "drift": ck["drift"],
           "step": ck["step"], "phase": ck["sched"]["in_phase"],
           "report_mismatches": ck["provenance"].get("report_mismatches"),
           "persons": {s: len(v) for s, v in ck["provenance"]["persons"].items()},
           "first_log": {"step": logs[0]["step"], "loss": logs[0]["loss"]} if logs else None,
           "last_log": {"step": logs[-1]["step"], "loss": logs[-1]["loss"]} if logs else None}
    trainer.log(dict(out, event="verify"))
    return out


def parse_args(argv=None):
    """The command line (module docstring), checked: --init-from needs --phase B and --run-dir, a run folder
    other than the checkpoint's; --init-stage2 needs --phase C and, without --resume, --run-dir, a run folder other
    than the checkpoint's; --init-from and --init-stage2 exclude each other; --jepa-grad-steps is not negative.
    argparse's exit (status 2) otherwise."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("A", "B", "AB", "C"), default="AB")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--max-steps", type=int, default=None, help="optimizer steps in this call (default: to the end)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None, help="a checkpoint of this run")
    ap.add_argument("--init-from", default=None,
                    help="phase B of a new run (--run-dir) under --config from a checkpoint at the end of phase A of "
                         "another run (module docstring, Init-from)")
    ap.add_argument("--init-stage2", default=None,
                    help="phase C of a stage-2 run (--run-dir) under --config from a checkpoint at the end of phase B "
                         "of a stage-1 run (module docstring, Phase C)")
    ap.add_argument("--jepa-grad-steps", type=int, default=0,
                    help="with the config's jepa section enabled: g_J and g_R at the first N optimizer steps of "
                         "phase B")
    ap.add_argument("--prefetch", type=int, default=PREFETCH)
    ap.add_argument("--no-dev", action="store_true", help="no dev numbers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-cap-gb", type=float, default=None,
                    help="caching-allocator cap in GB (default loop.MEMORY_CAP_GB; the float32 decoder of the phase A fix "
                         "needs up to about 12 GB reserved at 16,384 tokens: reports/bench/phaseA_fix_check.md)")
    ap.add_argument("--no-decoder-checkpoint", action="store_true",
                    help="store the decoder blocks' activations instead of recomputing them in the backward pass "
                         "(reports/bench/phaseB_memory.md)")
    a = ap.parse_args(argv)
    if a.init_from is not None:
        if a.phase != "B":
            ap.error("--init-from starts phase B: give --phase B")
        if a.run_dir is None:
            ap.error("--init-from needs --run-dir, the new run's folder")
        same = os.path.normcase(os.path.abspath(a.run_dir)) == os.path.normcase(
            os.path.dirname(os.path.abspath(a.init_from)))
        if same:
            ap.error("--run-dir is the folder of the --init-from checkpoint: give a new run folder")
    if a.init_stage2 is not None:
        if a.init_from is not None:
            ap.error("--init-from (phase B) and --init-stage2 (phase C) exclude each other")
        if a.phase != "C":
            ap.error("--init-stage2 starts phase C: give --phase C")
        if a.run_dir is None and a.resume is None:
            ap.error("--init-stage2 needs --run-dir, the new run's folder")
        if a.run_dir is not None and os.path.normcase(os.path.abspath(a.run_dir)) == os.path.normcase(
                os.path.dirname(os.path.abspath(a.init_stage2))):
            ap.error("--run-dir is the folder of the --init-stage2 checkpoint: give a new run folder")
    if a.jepa_grad_steps < 0:
        ap.error("--jepa-grad-steps is not negative")
    return a


def main(argv=None):
    a = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cfg = CF.load(a.config)
    if a.jepa_grad_steps and not cfg.jepa.enabled:
        raise SystemExit(f"--jepa-grad-steps {a.jepa_grad_steps}: the jepa section of {a.config} is not enabled")
    check_stage2_call(a, cfg)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    run_dir = run_folder(a.run_dir, a.resume)
    bs = BD.train_body_store(cfg)
    layouts = None
    if CF.stage2_on(cfg):
        ws = stage2_stores(cfg)
        layouts = LY.load_stage2_layouts()
    else:
        ws = WS.WindowStore() if "B" in a.phase else None
    synth = BD.synth_sampler(cfg, bs, device=dev, layouts_=layouts)
    hook = None if a.no_dev else DevHook(cfg, dev, layouts=layouts, stores=ws if CF.stage2_on(cfg) else None)
    cap = LP.MEMORY_CAP_GB if a.memory_cap_gb is None else a.memory_cap_gb
    if cap > float(cfg.schedule.max_reserved_gb):
        raise SystemExit(f"--memory-cap-gb {cap} is above schedule.max_reserved_gb {cfg.schedule.max_reserved_gb}")
    tr = LP.Trainer(cfg, bs, ws, synth, dev, run_dir, prefetch=a.prefetch, dev_hook=hook,
                    echo=lambda *x: print(*x, flush=True), memory_cap_gb=cap,
                    decoder_checkpoint=not a.no_decoder_checkpoint, jepa_grad_steps=a.jepa_grad_steps, layouts=layouts)
    steps = ", ".join(f"{p} {n:,}" for p, n in zip(tr.sched.run_phases, tr.sched.steps))      # 'A 10,681, B 32,044'
    print(f"run folder {tr.run_dir}; steps {steps}; optimizer {tr.opt_kind}; set up in "
          f"{time.perf_counter() - t0:.1f} s")
    alarm = None
    try:
        summary = tr.run(phases=tuple(a.phase), max_steps=a.max_steps, resume=a.resume, init_from=a.init_from,
                         init_stage2=a.init_stage2)
    except LP.AlarmError as e:
        alarm = str(e)
        summary = {"event": "alarm_stop", "phase": tr.sched.in_phase, "step": tr.sched.step,
                   "checkpoints": list(tr.checkpoints)}
    alarms = alarm_events(tr.log_path)
    for ev in alarms:
        print(f"alarm at phase {ev['phase']} step {ev['step']} ({ev['source']}): "
              + "; ".join(f"{h['name']} {h['value']:.4g}" for h in ev["alarms"]))
    res = verify(tr, cfg, fallback=a.resume)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "last"}, "verify": res,
                      "alarms": len(alarms), "alarm_stop": alarm}, indent=1, default=str))
    paths.assert_not_moveatlas(tr.run_dir)
    if alarm is not None:
        sys.exit(3)


def check_stage2_call(a, cfg):
    """SystemExit unless the call fits the config (module docstring, Phase C): phase C with a stage-2 config only,
    from --init-stage2 or --resume, and never under a config whose status starts with DRAFT; a stage-2 config trains
    phase C only."""
    if CF.stage2_on(cfg):
        if a.phase != "C":
            raise SystemExit(f"{a.config} is a stage-2 configuration: it trains phase C only (--phase C)")
        if a.init_stage2 is None and a.resume is None:
            raise SystemExit("phase C starts from --init-stage2 (a phase B end) or --resume (a phase C checkpoint)")
        if str(cfg.status).lstrip().upper().startswith("DRAFT"):
            raise SystemExit(f"{a.config} is a DRAFT ({cfg.status!r}): no run reads it until its start checkpoint and "
                             "its phase C budget are written into it")
    elif a.phase == "C" or a.init_stage2 is not None:
        raise SystemExit(f"--phase C and --init-stage2 need a configuration with the stage2 section enabled, not "
                         f"{a.config}")


def alarm_events(log_path):
    """The 'alarm' events of a run's log.jsonl (every call of the run folder)."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return [r for r in recs if r.get("event") == "alarm"]


if __name__ == "__main__":
    main()
