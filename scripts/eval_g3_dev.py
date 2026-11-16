"""B4.19: the G3 dev estimate of a checkpoint, simulator-only for phase A, healthy half for phase B (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.7, 5, 7 and 9 question 1; OD9 decided 2026-09-28: synthetic data are
admitted by utility, the simulator-only G3 line scored on non-lockbox controls after phase A; harness
mdfm/gates/g3.py, rule docs/rules/G3_RULE.md DRAFT; inference mdfm/eval/infer.py).

The checkpoint is read through mdfm.eval.infer.load_checkpoint (checkpoint.load: file, config, person-list, records
and lockbox hashes, the person guard, the state_dict audit; then the audit of the loaded weights) and run on the IMU
records of real weargait_pd controls: 13 IMUs, acc and gyro, the magnetometer masked (g3.model_input), 10 s windows
every 5 s, 20 refinement steps with 256 points per stream per step, bodies stitched on the trial clock
(infer.G3Predictor); the readouts of mdfm/readouts/gait.py give ICs, strides and pass speeds.
- Primary row, dev: the 14 dev controls of mdfm.data.sources.folds (drawn from the G2 TEST half: no body, signal
  or renderer statistic of theirs entered phase A), the steps of g3.dev_run with the model as its callable: the
  walkway (the reference floor), the population baseline (the 54 training-fold controls), the shank-gyroscope
  detector and the model, one constant offset per detector and IC reference fitted on these 14 (g3.fit_offsets).
  A predictor with no IC within the match window of any reference IC gets g3.UNFITTED (NaN, n 0) there, as in
  g3.dev_run, and its IC line reads inf (a kill) in both rows.
- Secondary row, all68: the 68 non-lockbox controls (the 14 and the 54 of the training fold; 54 of them gave
  mount-prior statistics and some device-noise and residual statistics to the renderer, never a window), scored
  with the offsets of the dev fit; the population baseline is in-sample for the 54 and stands for reference only.
Both initial-contact references are scored (walkway, the primary, and insole with the per-trial offset removed). OD12
closed on 2026-10-05: the walkway row decides the IC line, the insole row is reported and decides nothing (G3_RULE.md
section 6; g3.gate_results marks it decides False), so its verdict cell in the markdown reads REPORTED, 'reported
(does not decide)', and its computed verdict stays in the JSON with unchanged keys. The simulator-only line is stride_mape_pct (pass 25, kill above 40; above 40 on a
checkpoint that closes phase A stops the program, spec B4.19; such a checkpoint with no matched stride reads
'undetermined', never 'no': stop_decision). A checkpoint that does not close phase A (a smoke or partial run) is
labelled so: its numbers describe that checkpoint, not the simulator-only model.

Labels by phase (labels, default_out, stop_reading). A checkpoint of phase A, or of an unknown phase, is the
simulator-only estimate: 'what' "simulator-only G3 dev estimate (OD9), not a gate read", files g3_sim.json and
g3_sim.md, default folder reports/gates/g3_sim/dev_<checkpoint name>/. A phase B checkpoint (Loaded.phase 'B')
trained on the real healthy windows too (spec section 5), so it is not the simulator-only model: 'what' and the
markdown heading read "G3 healthy-half dev estimate (phase B checkpoint), not a gate read", the files are
g3_dev.json and g3_dev.md, the default folder reports/gates/g3/dev_<checkpoint name>/. Its stride MAPE is reported
with the same pass and kill values, and the stop rule, which reads the checkpoint closing phase A only
(G3_RULE.md section 8), reads 'not applicable' (stops_program false; stop_decision itself is unchanged). A
checkpoint of the stage-2 phase C (Loaded.phase 'C', task S2.11; a phase C run starts from a phase B end) gets the
labels, files and folder of a phase B checkpoint, with 'phase C checkpoint, stage 2' in place of 'phase B
checkpoint' in its texts (WHAT_C, STOP_PHASE_C), so that scripts/stage2_read.py reads its g3_dev.json. The JSON
keys are the same in every phase (the block simulator_only keeps its name). With --weights model and no --out,
the default folder ends in _model, so the raw weights never overwrite the EMA report; an explicit --out keeps
the file names of the phase.

Never read: a lockbox row or record (the g3 reader keeps the rows of the requested non-lockbox persons and asserts
their lockbox column), a lockbox person, EVAL or CODY data. A dev control among the checkpoint's training persons
refuses the run; training persons among the 68 are counted and reported.

F2 lever 5a (task L5A; mdfm/eval/zupt_refine.py), off by default: --zupt-anchor adds the inertial anchor of lever 5 to
every chunk's refinement objective (targets: the version-2 ZUPT strides of each record's own foot sensors, no
reference value; constants fixed in the module, 20 refinement steps, so another --refine-steps is refused);
--zupt-report computes the targets and the foot report without touching the objective. Either needs --out, states
the option in the JSON (inference.zupt_anchor, and the block zupt_anchor of the dev trials and of every trial), adds
the option's columns to trials.csv and writes zupt_strides.csv; with --zupt-anchor 'what' and the heading add
'inertially anchored body (lever 5a, at inference)'.

Writes <out> (default: the folder of the phase, above): g3_sim.json (g3_dev.json for phase B; sets, checkpoint,
inference settings, lines of every predictor in both rows, GateResult rows of the dev row as dev estimates,
offsets, timing), g3_sim.md (g3_dev.md), per_person.csv (row, predictor, line, person, value), offsets.csv,
trials.csv (per trial: windows, refinement objective, seconds), _provenance.json (MoveAtlas HEAD read from its
.git files, no git command).

Run (mdfm environment; the GPU is shared with training, so the caching allocator is capped with --memory-gb):
  <conda>/envs/mdfm/python.exe scripts/eval_g3_dev.py --checkpoint <run>/ckpt_A_<step>.pt
  <conda>/envs/mdfm/python.exe scripts/eval_g3_dev.py --checkpoint <run>/ckpt_B_<step>.pt
"""
import argparse
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

ROWS = ("dev", "all68")
PREDICTORS = ("walkway", "population", "shank_gyro", "model")
LINE_ORDER = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "ic_median_ms_insole")
MAPE_STOP = 40.0            # spec B4.19: the simulator-only stride MAPE above this stops the program
N_DEV = 14                  # the dev controls of mdfm.data.sources.folds (spec 2.2)
WHAT_SIM = "simulator-only G3 dev estimate (OD9), not a gate read"
WHAT_B = "G3 healthy-half dev estimate (phase B checkpoint), not a gate read"
WHAT_C = "G3 healthy-half dev estimate (phase C checkpoint, stage 2), not a gate read"
STOP_PHASE_B = ("not applicable: the stop rule reads the checkpoint closing phase A only, the simulator-only model "
                "(docs/rules/G3_RULE.md section 8), and this is a phase B checkpoint")
STOP_PHASE_C = ("not applicable: the stop rule reads the checkpoint closing phase A only, the simulator-only model "
                "(docs/rules/G3_RULE.md section 8), and this is a phase C checkpoint of stage 2")
REPORTED = "reported (does not decide)"


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256, help="kernel points per stream-window and refinement step")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a trial")
    p.add_argument("--no-secondary", action="store_true", help="the 14 dev controls only")
    z = p.add_mutually_exclusive_group()
    z.add_argument("--zupt-anchor", action="store_true",
                   help="F2 lever 5a: the inertial anchor in the refinement (constants fixed; needs --out)")
    z.add_argument("--zupt-report", action="store_true",
                   help="F2 lever 5a report only: targets and foot report, objective unchanged (needs --out)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.zupt_anchor or a.zupt_report:
        from mdfm.eval import zupt_refine as ZR
        if not a.out:
            p.error("--zupt-anchor and --zupt-report need --out (never the folder of the run without the option)")
        if a.zupt_anchor and a.refine_steps != ZR.STEPS:
            p.error(f"--zupt-anchor runs {ZR.STEPS} refinement steps (a fixed constant of lever 5a)")
    return a


ZUPT_LABEL = "inertially anchored body (lever 5a, at inference)"
ZUPT_CODE_FILES = ("mdfm/eval/zupt_refine.py", "mdfm/readouts/zupt.py", "mdfm/data/zupt_anchor.py",
                   "mdfm/train/losses.py")


def zupt_mode_of(a):
    """The lever-5a mode of the parsed arguments: 'anchor' (--zupt-anchor), 'report' (--zupt-report) or 'off'."""
    return "anchor" if getattr(a, "zupt_anchor", False) else "report" if getattr(a, "zupt_report", False) else "off"


def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def lines_of(summ):
    return {name: {line: {k: v for k, v in st.items() if k != "per_person"} for line, st in s.items()}
            for name, s in summ.items()}


def fit_offsets(preds, targets):
    """({(predictor, reference): (offset_s, n matched)}, unfitted ['predictor|reference']): g3.fit_offsets (the
    steps of g3.dev_run), with the pairs left unfitted (g3.UNFITTED, NaN offset: the IC line reads inf in both
    rows, never matched with a guessed offset) listed. A person outside the dev controls raises."""
    from mdfm.gates import g3
    offsets = g3.fit_offsets(preds, targets)
    unfitted = [f"{name}|{ref}" for (name, ref), (off, _) in offsets.items() if not g3.offset_fitted(off)]
    return offsets, unfitted


def stop_decision(phase_a_complete, mape):
    """The B4.19 stop rule on the model's dev stride MAPE (percent): True above MAPE_STOP on a checkpoint that
    closes phase A (or a phase B one), False at or below it and on any checkpoint that does not close phase A,
    None (undetermined) when such a checkpoint matched no walkway stride (MAPE not evaluated): the rule does
    not settle that case, so it is reported for the owner and never read as a pass."""
    if not phase_a_complete:
        return False
    v = fin(mape)
    return None if v is None else bool(v > MAPE_STOP)


def stop_text(stop):
    return {True: "yes", False: "no", None: "undetermined: no model stride matched a walkway stride, a case the "
            "rule does not settle (owner decision)"}[stop]


def is_phase_b(loaded):
    """True for a checkpoint saved in phase B (loaded.phase 'B') or in the stage-2 phase C ('C', which starts from a
    phase B end and trains on real windows too); phase A and an unknown phase are False."""
    return loaded.phase in ("B", "C")


def what_of(loaded):
    """WHAT_C for a phase C checkpoint, else WHAT_B (a phase B checkpoint; module docstring, labels by phase)."""
    return WHAT_C if loaded.phase == "C" else WHAT_B


def stop_not_applicable(loaded):
    return STOP_PHASE_C if loaded.phase == "C" else STOP_PHASE_B


def labels(loaded):
    """The labels of the report of a loaded checkpoint (module docstring, labels by phase); only loaded.phase is
    read. {'stem': file stem of the JSON and markdown, 'folder': the default folder under reports/gates/, 'run',
    'what', 'heading': the markdown title, 'basis': the decision cited in the markdown intro, 'mape_section': the
    title of the markdown section of the stride MAPE line}."""
    if is_phase_b(loaded):
        what = what_of(loaded)
        return {"stem": "g3_dev", "folder": "g3", "run": "g3_dev", "what": what, "heading": f"# {what}",
                "basis": "", "mape_section": "Stride MAPE and the dev GateResult rows"}
    return {"stem": "g3_sim", "folder": "g3_sim", "run": "g3_sim_dev", "what": WHAT_SIM,
            "heading": "# G3 simulator-only dev estimate (plan task B4.19)", "basis": " (OD9)",
            "mape_section": "Simulator-only line and the dev GateResult rows"}


def default_out(loaded, tree=None):
    """The output folder without --out: <tree>/reports/gates/<labels folder>/dev_<checkpoint name> (tree: TREE by
    default), with the suffix _model for the raw weights (loaded.weights 'model'), so that they never overwrite the
    EMA report."""
    suffix = "_model" if loaded.weights == "model" else ""
    return os.path.join(TREE if tree is None else tree, "reports", "gates", labels(loaded)["folder"],
                        f"dev_{loaded.name}{suffix}")


def stop_reading(loaded, mape):
    """(stops_program, stop_rule text) of the report. Phase A or an unknown phase: stop_decision(
    loaded.phase_a_complete, mape) and its stop_text. Phase B: (False, STOP_PHASE_B), the stop rule reading the
    checkpoint closing phase A only."""
    if is_phase_b(loaded):
        return False, stop_not_applicable(loaded)
    stop = stop_decision(loaded.phase_a_complete, mape)
    return stop, stop_text(stop)


def verdict_text(verdict, extra):
    """The verdict cell of a dev GateResult row in the markdown: REPORTED for a row marked decides False (the insole IC
    row, the second IC reference: docs/rules/G3_RULE.md section 6, revision 2026-10-05), whose computed verdict stays
    in the JSON only (results_dev, unchanged keys); the row's verdict otherwise."""
    return REPORTED if (extra or {}).get("decides", True) is False else verdict


def reported_note(results):
    """The sentence under the dev GateResult table naming the rows that are reported and decide nothing ('' when
    none); results are GateResult dicts (to_dict)."""
    rows = [r["line"] for r in results if verdict_text(r["verdict"], r.get("extra")) == REPORTED]
    if not rows:
        return ""
    return (f"Rows marked '{REPORTED}' ({', '.join(rows)}): the second initial-contact reference, scored with the "
            "same statistic, interval and margin and reported beside the walkway row; it decides nothing "
            "(docs/rules/G3_RULE.md section 6, revision 2026-10-05). The JSON keeps its computed verdict in "
            "results_dev with the extra keys decides (false) and role.")


def mape_sentence(loaded, mape, stop):
    """The markdown sentence of the model's dev stride MAPE (percent) under the stop reading `stop`."""
    if is_phase_b(loaded):
        return (f"stride MAPE of the model on the dev controls: {fmt(mape)} percent (pass at most 25, kill above "
                f"{MAPE_STOP:g}; stops the program: {stop_not_applicable(loaded)}).")
    return (f"stride MAPE of the model on the dev controls: {fmt(mape)} percent (pass at most 25, kill above "
            f"{MAPE_STOP:g}; stops the program: {stop_text(stop)}"
            + ("" if loaded.phase_a_complete else ", the checkpoint does not close phase A") + ").")


def main():
    a = parse()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import pandas as pd
    import torch

    from mdfm import paths
    from mdfm.data import sources
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    code_names = ["mdfm/gates/g3.py", "mdfm/readouts/gait.py", "mdfm/eval/infer.py", "scripts/eval_g3_dev.py"]
    code_paths = [g3.__file__, g3.gait.__file__, I.__file__, os.path.abspath(__file__)]
    if zupt_mode_of(a) != "off":
        code_names += list(ZUPT_CODE_FILES)
        code_paths += [os.path.join(TREE, *f.split("/")) for f in ZUPT_CODE_FILES]
    code_sha = common.code_sha256(code_paths)
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    lab = labels(loaded)
    out = a.out or default_out(loaded)
    paths.assert_not_moveatlas(out)
    t_load = time.time() - t_start
    print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step} "
          f"of {loaded.phase_steps}; load {t_load:.1f} s", flush=True)

    dev_ids, train_ids = g3.controls("dev"), g3.controls("train")
    if len(dev_ids) != N_DEV or set(dev_ids) & set(train_ids):
        raise SystemExit(f"dev set of {len(dev_ids)} controls, overlap {sorted(set(dev_ids) & set(train_ids))}")
    all68 = sorted(set(dev_ids) | set(train_ids))
    locked = set(g3.lockbox_persons())
    if set(all68) & locked:
        raise SystemExit("a lockbox person among the non-lockbox controls")
    trained = loaded.trained_persons()
    if set(dev_ids) & trained:
        raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev_ids) & trained)}")
    trained68 = sorted(set(all68) & trained)

    zmode = zupt_mode_of(a)
    if zmode != "off":
        from mdfm.eval import zupt_refine as ZR
        if zmode == "anchor":
            lab = dict(lab, what=f"{lab['what']}; {ZUPT_LABEL}", heading=f"{lab['heading']}; {ZUPT_LABEL}")
        else:
            lab = dict(lab, what=f"{lab['what']}; F2 lever 5a report only (the objective unchanged)")
    pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                         seed=a.seed, pool=not a.no_pool, zupt_mode=zmode)

    def model(tid, rec):
        tb = pred(tid, rec)
        r = pred.log[-1]
        print(f"  {tid}: {r['windows']} windows, {r['seconds']:.1f} s", flush=True)
        return tb

    def model_predictions(persons, targets, skip=()):
        """(Predictions 'model' of the persons' trials outside skip, {trial_id: shank gyro}): one trial at a time
        through body_predictions (a stitched body is dropped once read)."""
        gyros, parts = {}, []
        for tid, rec in g3.imu_records(persons, targets.trials):
            gyros[tid] = g3.shank_gyro(rec)
            if tid not in skip:
                parts.append(g3.body_predictions([model(tid, g3.model_input(rec))], targets))
        cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
        return (g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds")) if parts else None), gyros

    # ---------------- primary row: the 14 dev controls (the steps of g3.dev_run; an offset that cannot be fitted,
    # no IC of the predictor within the match window, is recorded as 0 with n 0 and its IC line reads inf)
    t0 = time.time()
    tg, tr = g3.load_targets(dev_ids), g3.load_targets(train_ids)
    if tg.counts["lockbox_rows"] or tr.counts["lockbox_rows"]:
        raise SystemExit("a lockbox row was kept")
    base = g3.population_baseline(tr)
    model_dev, gyros_dev = model_predictions(dev_ids, tg)
    preds = [g3.walkway_predictions(tg), g3.population_predictions(base, tg), g3.shank_predictions(gyros_dev),
             model_dev]
    offsets, unfitted = fit_offsets(preds, tg)
    summ, tables = g3.evaluate(tg, preds, {k: v[0] for k, v in offsets.items()})
    t_dev = time.time() - t0
    rows = {"dev": {"summ": summ, "tables": tables, "targets": tg, "persons": list(tg.persons)}}

    # ---------------- secondary row: the 68 non-lockbox controls, offsets of the dev fit
    t_68 = 0.0
    if not a.no_secondary:
        t0 = time.time()
        tg68 = g3.load_targets(all68)
        if tg68.counts["lockbox_rows"]:
            raise SystemExit("a lockbox row was kept")
        rest, gyros = model_predictions(all68, tg68, skip=set(tg.trials["trial_id"]))
        parts = [model_dev] + ([rest] if rest is not None else [])
        cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
        model68 = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds"))
        preds68 = [g3.walkway_predictions(tg68), g3.population_predictions(base, tg68), g3.shank_predictions(gyros),
                   model68]
        summ68, tables68 = g3.evaluate(tg68, preds68, {k: v[0] for k, v in offsets.items()})
        rows["all68"] = {"summ": summ68, "tables": tables68, "targets": tg68, "persons": list(tg68.persons)}
        t_68 = time.time() - t0

    # ---------------- GateResult rows of the dev row (dev estimates, not gate reads)
    with open(g3.RULE_PATH, "rb") as f:
        rule_sha = common.sha256_bytes(f.read())
    with open(g3.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())
    results = g3.gate_results(summ, tables, "model", rule_sha, loaded.sha256, I.train_persons_sha256(loaded))
    mape = summ["model"].get("stride_mape_pct", {}).get("point")
    stop, stop_rule = stop_reading(loaded, mape)
    reads = [r for r in common.read_log() if r["gate"] == g3.GATE] if os.path.exists(common.READ_LOG) else []

    # ---------------- tables
    pp = []
    for row, d in rows.items():
        for name, lines in d["summ"].items():
            for line, st in lines.items():
                for pid, v in st["per_person"].items():
                    pp.append((row, name, line, pid, fin(v)))
    per_person = pd.DataFrame(pp, columns=["row", "predictor", "line", "person_id", "value"])
    off = pd.DataFrame([(n, r, 1000.0 * o, k) for (n, r), (o, k) in sorted(offsets.items())],
                       columns=["predictor", "reference", "offset_ms", "n_matched"])
    dev_trials = set(tg.trials["trial_id"])
    zrows = []
    for e in pred.log:
        zrows += [dict(r, row="dev" if e["trial_id"] in dev_trials else "all68") for r in e.pop("zupt_rows", None) or []]
    trials = pd.DataFrame(pred.log)
    trials["row"] = ["dev" if t in dev_trials else "all68" for t in trials["trial_id"]]
    common.write_csv(per_person, os.path.join(out, "per_person.csv"))
    common.write_csv(off, os.path.join(out, "offsets.csv"))
    common.write_csv(trials, os.path.join(out, "trials.csv"))
    if zmode != "off":
        common.write_csv(pd.DataFrame(zrows, columns=["row"] + list(ZR.ROW_FIELDS)),
                         os.path.join(out, "zupt_strides.csv"))

    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    head = PV.moveatlas_head()
    wall = time.time() - t_start
    n_a = loaded.phase_steps[0] if loaded.phase_steps else "?"
    kind = ("closes phase A: the simulator-only model" if loaded.phase_a_complete and loaded.phase == "A"
            else "phase B checkpoint" if loaded.phase == "B"
            else "phase C checkpoint (stage 2)" if loaded.phase == "C"
            else f"partial phase A (step {loaded.step} of {n_a}): not the simulator-only model")
    res = {
        "run": lab["run"], "gate": g3.GATE, "what": lab["what"],
        "checkpoint": loaded.describe(),
        "checkpoint_kind": kind,
        "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S,
                      "hop_s": I.HOP_S, "batch_windows": a.batch_windows, "seed": a.seed, "shape_pooling":
                      "per trial" if not a.no_pool else "none", "weights": a.weights, "device": str(dev),
                      "amp_bf16": dev.type == "cuda", "memory": mem},
        "rule": {"path": common.rel(g3.RULE_PATH), "sha256": rule_sha, "status": status[0],
                 "registered": common.registered(g3.GATE, rule_sha)},
        "lockbox_reads_logged_for_g3": len(reads),
        "sets": {"dev": dev_ids, "dev_sha256": sources.person_hash(dev_ids),
                 "train_fold_sha256": sources.person_hash(train_ids),
                 "all68_sha256": sources.person_hash(all68), "n_all68": len(all68), "tasks": list(g3.TASKS),
                 "trained_on_among_all68": trained68,
                 "dev_counts": tg.counts, "all68_counts": rows["all68"]["targets"].counts if "all68" in rows else None},
        "constants": {"stride_match_s": g3.STRIDE_MATCH_S, "ic_match_s": g3.IC_MATCH_S, "ic_margin": g3.IC_MARGIN,
                      "lines": g3.LINES, "primary_ic_reference": g3.PRIMARY_REFERENCE, "mape_stop": MAPE_STOP},
        "offsets_ms": {f"{n}|{r}": {"offset_ms": 1000.0 * o, "n_matched": k} for (n, r), (o, k) in offsets.items()},
        "offsets_unfitted": unfitted,
        "model_events_dev": {"ics": int(len(model_dev.ics)), "strides": int(len(model_dev.strides)),
                             "pass_speeds": int(len(model_dev.speeds))},
        "population_baseline": {t: {"stride_m": s, "speed_m_s": v} for t, (s, v) in base.items()},
        "lines": {row: lines_of(d["summ"]) for row, d in rows.items()},
        "results_dev": [r.to_dict() for r in results],
        "simulator_only": {"stride_mape_pct": mape, "kill_line": MAPE_STOP, "phase_a_complete": loaded.phase_a_complete,
                           "stops_program": stop, "stop_rule": stop_rule},
        "timing_s": {"load": round(t_load, 1), "dev": round(t_dev, 1), "all68": round(t_68, 1), "all": round(wall, 1)},
        "moveatlas_head": head,
    }
    if zmode != "off":
        res["inference"]["zupt_anchor"] = ZR.record_settings(zmode)
        res["zupt_anchor"] = {"dev": ZR.summary_block([e for e in pred.log if e["trial_id"] in dev_trials], zmode),
                              "all": ZR.summary_block(pred.log, zmode)}
        if zmode == "anchor":
            res["row_label"] = ZUPT_LABEL
    raw, res = res, clean(res)          # the JSON holds null for a non-finite value, the markdown prints inf

    # ---------------- markdown
    ck = res["checkpoint"]
    L = [lab["heading"], "",
         "Written by scripts/eval_g3_dev.py (harness mdfm/gates/g3.py, inference mdfm/eval/infer.py); every number "
         f"below is read from {lab['stem']}.json, per_person.csv, offsets.csv and trials.csv of this folder. A dev "
         f"estimate on non-lockbox controls{lab['basis']}, not a gate read: no lockbox row or record was read.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,} of {ck['phase_steps']}; {res['checkpoint_kind']}. Training persons: "
         + ", ".join(f"{s} {n}" for s, n in ck["persons"].items()) + f" (SHA-256 {ck['persons_sha256'].get('all')}). "
         f"Hashes that moved since the save: {', '.join(ck['drift']) or 'none'}.", "",
         f"Inference: 13 IMUs (acc and gyro, magnetometer masked), {I.WIN_S:g} s windows every {I.HOP_S:g} s, "
         f"{a.refine_steps} refinement steps of {a.points} points per stream, shape pooled "
         f"{res['inference']['shape_pooling']}, stitched on the trial clock; device {dev}"
         + (f", peak allocated {mem['max_allocated_gb']:.2f} GB, reserved {mem['max_reserved_gb']:.2f} GB under a "
            f"{cap} GB cap" if mem else "") + ".", "",
         f"Rule docs/rules/G3_RULE.md: status '{status[0]}', SHA-256 {rule_sha} (registered: "
         f"{res['rule']['registered']}). Lockbox reads logged for G3: {len(reads)}. MoveAtlas HEAD {head}.", ""]
    if zmode != "off":
        zd = res["zupt_anchor"]["dev"]
        L += [f"F2 lever 5a option, mode {zmode} ({zd['what']}): weight {zd['weight']:g}, sd {zd['sd_m_s']:g} m/s, "
              f"edge {zd['edge_s']:g} s, strides of at least {zd['min_length_m']:g} m, {zd['steps']} steps; targets "
              f"from {zd['source']}. Dev trials: {zd['trials']}, {zd['trials_with_targets']} with a target, "
              f"{zd['targets']} targets. Per target: zupt_strides.csv; the block zupt_anchor of {lab['stem']}.json.", ""]
    L += ["## Sets", "",
          f"- dev (primary): the {len(dev_ids)} dev controls (SHA-256 {res['sets']['dev_sha256']}), none among the "
          "checkpoint's training persons (asserted).",
          f"- all68 (secondary): the {len(all68)} non-lockbox controls (SHA-256 {res['sets']['all68_sha256']}); among "
          f"the checkpoint's training persons: {len(trained68)}. Offsets from the dev fit. The population baseline "
          "learns from the 54 training-fold controls, in-sample in this row.", ""]
    L += ["| row | trials | walkway ICs | insole ICs | walkway strides | walkway passes |", "|---|---|---|---|---|---|"]
    for row, d in rows.items():
        c = d["targets"].counts
        L.append(f"| {row} | {c['trials']} | {c['walkway_ic']} | {c['insole_ic']} | {c['strides']} | {c['passes']} |")
    L += ["", f"## {lab['mape_section']}", "", mape_sentence(loaded, mape, stop), "",
          "| line | unit | pass | kill | model | 95 % interval | matched share | trivial baseline | verdict (dev estimate) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in raw["results_dev"]:
        L.append(f"| {r['line']} | {r['unit']} | {fmt(r['pass_line'])} | {fmt(r['kill_line'])} | {fmt(r['point'])} | "
                 f"{fmt_iv(r['interval'])} | {fmt(r['extra'].get('matched_share'), 3)} | {fmt(r['trivial_baseline'])} | "
                 f"{verdict_text(r['verdict'], r['extra'])} |")
    note = reported_note(raw["results_dev"])
    L += ["", "Trivial baselines: the population baseline (stride, speed, MAPE) and the shank-gyroscope detector "
          f"(IC; the model must reach at most {g3.IC_MARGIN} times its value on the ICs both match). IC references: "
          "walkway (the primary, whose row decides the IC line; OD12, closed 2026-10-05) and insole with the per-trial "
          "offset removed." + (f" {note}" if note else ""), ""]
    for row, d in rows.items():
        L += [f"## Every predictor, row {row}", "",
              "| predictor | line | point | 95 % interval | matched share | reference events | persons | offset (ms) |",
              "|---|---|---|---|---|---|---|---|"]
        for name in PREDICTORS:
            for line in LINE_ORDER:
                st = raw["lines"][row].get(name, {}).get(line)
                if st is None:
                    continue
                L.append(f"| {name} | {line} | {fmt(st['point'])} | {fmt_iv(st['interval'])} | "
                         f"{fmt(st['matched_share'], 3)} | {st['n_events']} | {st['n_persons']} | "
                         f"{fmt(st.get('offset_ms'), 1) if 'offset_ms' in st else ''} |")
        L.append("")
    L += ["Offsets (dev fit, median of predicted minus reference over the nearest predictions within "
          f"{g3.IC_MATCH_S} s): " + "; ".join(f"{r.predictor} against {r.reference} {r.offset_ms:.1f} ms (n {r.n_matched})"
                                             for r in off.itertuples()) + ".",
          f"Offsets not fitted (no IC of the predictor within {g3.IC_MATCH_S} s of a reference IC; 0 ms used, the "
          f"IC line reads inf): {', '.join(unfitted) or 'none'}. Model events on the dev trials: "
          f"{res['model_events_dev']['ics']} ICs, {res['model_events_dev']['strides']} strides, "
          f"{res['model_events_dev']['pass_speeds']} pass speeds.", ""]
    rf = trials.dropna(subset=["refine_first"])
    L += [f"Trials run through the model: {len(trials)} ({int(trials['windows'].sum())} windows); refinement objective "
          f"(sum over a trial's windows, per step's point draw) before the first step {rf['refine_first'].median():.4g} and "
          f"before the last {rf['refine_last'].median():.4g} (medians over trials). Time: load {res['timing_s']['load']} s, "
          f"dev row {res['timing_s']['dev']} s, all68 row {res['timing_s']['all68']} s, all {res['timing_s']['all']} s."
          if len(rf) else f"Trials run through the model: {len(trials)}; no refinement.", ""]
    text = "\n".join(L) + "\n"
    common.write(res, out, lab["stem"], text)
    prov = {"script": "scripts/eval_g3_dev.py", "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256"],
            "code_sha256": code_sha, "code_files": code_names,
            **({"zupt_anchor": zmode} if zmode != "off" else {})}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(text)
    print(f"written {common.rel(out)}")


if __name__ == "__main__":
    main()
