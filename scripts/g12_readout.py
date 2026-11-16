"""G12 readout of one checkpoint on the G3 dev controls (task G12R; docs/rules/G12_RULE.md sections 6 and 7, DRAFT;
revision docs/rules/G12_RULE_REVISION_2026-10-08.md, hashed, item 8): frozen tokens, the section 6 probes fitted on the
54 training-fold weargait_pd controls, the variant of each line chosen by person cross-fitted error on those 54, and
the three G3 dev lines (stride_mae_cm, speed_mae_m_s, ic_median_ms) of the chosen variants on the 14 and the 12 dev
controls, with the G3 harness's own statistics, plus the per-row predictions with each row's person (for the section 8
interval, scripts/g12_compare.py). Every output carries the label LABEL. The method is mdfm/baselines/readouts.py.

Checkpoints:
- a G12 baseline checkpoint (mdfm/baselines/load.load_frozen: EMA weights by default, tokenizer and encoder only),
  given by --checkpoint, or by --run-dir: the checkpoint of the run's 'g12_verify' event, read only when
  scripts/read_g12.py reads the run FINISHED (a stopped seed is not read, rule section 5). The run's training block
  (verdict, realized clock, its ratio to the budget, optimizer steps, tokens, the verified checkpoint) is reported
  beside the numbers (rule section 5), and final_verified says whether the checkpoint read is the run's verified final
  checkpoint (rule section 6: the EMA of the final checkpoint; scripts/g12_compare.py reads only those);
- a body checkpoint (mdfm/eval/infer.load_checkpoint): the representation row of section 6, reported without verdict.

Sets: the 54 training-fold controls fit (g3.controls('train')); the 14 dev controls (g3.controls('dev')) and the 12 of
g3.DEV_SETS are scored. A dev control among the checkpoint's training persons stops the run; a lockbox person among
either set stops it; records come from g3.imu_records (reader flags False), each asserted lockbox False and inside its
set, and the model reads them as G3 does (g3.model_input: 13 IMUs, acc and gyro, magnetometer masked).

Never: a lockbox, EVAL, CODY or care_pd record, a decoder pass, a write to a checkpoint, a comparison. (Loading a
checkpoint may append a tolerated records drift to drift_log.jsonl beside it, as every checkpoint read does; the
checkpoint is loaded once, its kind told by its run folder's config.json.)
The GPU only through the queue (scripts/queue_job.py); a CPU run sets CUDA_VISIBLE_DEVICES=-1 and --device cpu.

Writes <out> (default reports/bench/g12/readouts/<name>, name as mdfm.eval.infer.checkpoint_name): results.json,
summary.md, per_row.csv (comparison.ROW_COLUMNS plus in_dev12: every walkway stride, pass and walkway reference IC of
the 14, for every variant, the chosen ones flagged), per_pass.csv, _provenance.json; the token features to <features>
(default cache/runs/g12_readouts/<name>/features.npz), so that --from-features refits on the CPU without the model.

Run (mdfm environment; the GPU through the queue):
  python scripts/g12_readout.py --checkpoint cache/runs/g12_mfr_s20260928/ckpt_B_056774.pt --memory-gb 12
  python scripts/g12_readout.py --run-dir cache/runs/g12_jepa_s20260928 --memory-gb 12
  python scripts/g12_readout.py --checkpoint cache/runs/phaseA_root_full/ckpt_B_032044.pt --memory-gb 12
  python scripts/g12_readout.py --checkpoint ... --from-features --device cpu
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import numpy as np  # noqa: E402

LABEL = "G12 readout, dev only, not a G12 comparison"
TASK = "G12R"
OUT_ROOT = os.path.join(TREE, "reports", "bench", "g12", "readouts")
FEATURE_ROOT = os.path.join(TREE, "cache", "runs", "g12_readouts")
LINE_TITLES = (("stride_mae_cm", "stride MAE (cm)", 2), ("speed_mae_m_s", "speed MAE (m/s)", 3),
               ("ic_median_ms", "IC median, walkway (ms)", 1))
FOLD_SIZES = {"train": 54, "dev": 14}          # g3.controls' non-lockbox weargait_pd controls; others stop the run


# ---------------------------------------------------------------- helpers
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if (isinstance(x, float) and math.isinf(x)) or x in ("inf", "-inf"):
        return str(float(x))
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    """JSON-ready: non-finite floats as null (infinite ones as the string 'inf'), numpy values as Python values."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else ("inf" if o > 0 else "-inf" if o < 0 else None)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def read_g12_module():
    """scripts/read_g12.py (beside this script) as a module (its run reading: FINISHED, the compute record, the
    verified checkpoint)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_g12.py")
    spec = importlib.util.spec_from_file_location("read_g12_for_readout", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_run(d, tree=TREE):
    """read_g12.read_train of a run folder, without the live process list. read_train names the folder relative to
    the tree, which fails for a folder on another drive (a store on F:, a test folder): such a folder is read against
    its own drive root and named by its absolute path."""
    RG = read_g12_module()
    try:
        os.path.relpath(d, tree)
        other = False
    except ValueError:
        tree, other = os.path.splitdrive(os.path.abspath(d))[0] + os.sep, True
    tr = RG.read_train(d, tree, cmdlines=None)
    if other:
        tr["run_dir"] = os.path.abspath(d).replace("\\", "/")
    return tr


def resolve_run_dir(run_dir, tree=TREE):
    """(checkpoint path, training block) of a FINISHED baseline run (module docstring); SystemExit otherwise."""
    d = run_dir if os.path.isabs(run_dir) else os.path.join(tree, run_dir)
    tr = read_run(d, tree)
    if tr.get("verdict") != "FINISHED" or not (tr.get("verify") or {}).get("path"):
        raise SystemExit(f"{run_dir}: run {tr.get('verdict')}, not FINISHED with a verified checkpoint; not read "
                         "(rule section 5)")
    p = str(tr["verify"]["path"]).replace("\\", "/")
    return (p if os.path.isabs(p) else os.path.join(tree, p)), training_block(tr)


def training_block(tr):
    """The parts of a read_g12.read_train result reported beside the numbers."""
    return {"verdict": tr.get("verdict"), "run_dir": tr.get("run_dir"), "log_sha256": tr.get("log_sha256"),
            "compute": tr.get("compute"), "verify": tr.get("verify"), "planned_steps_A_B": tr.get("planned_steps_A_B")}


def training_of(ckpt, tree=TREE):
    """The training block of the run folder holding a baseline checkpoint, None without its log."""
    d = os.path.dirname(os.path.abspath(ckpt))
    if not os.path.exists(os.path.join(d, "log.jsonl")):
        return None
    return training_block(read_run(d, tree))


def final_verified(training, sha256):
    """True when the run is FINISHED and its verified checkpoint is the one read (same SHA-256)."""
    if not training:
        return None
    v = training.get("verify") or {}
    return bool(training.get("verdict") == "FINISHED" and v.get("sha256") and v.get("sha256") == sha256)


def check_features_meta(meta, name, path=""):
    got = (meta.get("checkpoint") or {}).get("name")
    if got != name:
        raise SystemExit(f"{path or 'the feature file'} holds the features of {got!r}, not of {name!r}")


def population_lines(tg_train, tgs):
    """{set: lines} of g3's population baseline (training fold) on the dev sets (context, as J0 reports it)."""
    from mdfm.baselines import readouts as RO
    from mdfm.eval import probes
    from mdfm.gates import g3
    base = g3.population_baseline(tg_train)
    return {s: {ln: RO.strip(st) for ln, st in probes.score(tg, [g3.population_predictions(base, tg)])["population"]
                .items()} for s, tg in tgs.items()}


# ---------------------------------------------------------------- results and report
def build_results(res0, ck, training, kind, inference, features, sets, train_in_ckpt, counts, constants, lockbox,
                  population, rows_file, head, timing, run_started, partial):
    """The results dict (results.json) from run_probes' output and the run's provenance."""
    from mdfm.data import sources as SRC
    return {"what": LABEL, "task": TASK, "run_started": run_started, "partial": bool(partial), "kind": kind,
            "arm": ck.get("arm"), "seed": ck.get("seed"), "checkpoint": ck, "training": training,
            "final_verified": final_verified(training, ck.get("sha256")) if kind == "baseline" else None,
            "representation_row": kind == "body", "inference": inference, "features": features,
            "sets": {"train": sorted(sets["train"]), "train_sha256": SRC.person_hash(sorted(sets["train"])),
                     "dev14": res0["sets"]["dev14"], "dev14_sha256": SRC.person_hash(res0["sets"]["dev14"]),
                     "dev12": res0["sets"]["dev12"], "dev12_sha256": SRC.person_hash(res0["sets"]["dev12"]),
                     "train_among_checkpoint_persons": train_in_ckpt},
            "constants": constants, "chosen": res0["chosen"], "crossfit": res0["crossfit"], "lines": res0["lines"],
            "per_person": res0["per_person"], "variants": res0["variants"], "population": population,
            "rows": rows_file, "counts": counts, "lockbox": lockbox, "moveatlas_head": head, "timing_s": timing}


def report_md(res):
    """summary.md (every number read from the results dict as written to results.json)."""
    from mdfm.baselines import readouts as RO
    ck = res["checkpoint"]
    who = (f"arm {res['arm']}, seed {res['seed']}" if res["kind"] == "baseline"
           else f"body model, seed {res['seed']} (representation row, reported without verdict)")
    L = [f"# G12 readout, {ck['name']} ({LABEL})", "",
         "Written by scripts/g12_readout.py (task G12R; method mdfm/baselines/readouts.py; rule docs/rules/G12_RULE.md "
         "sections 6 and 7, revision 2026-10-08 item 8); every number below is read from results.json of this folder. "
         "Dev only: no lockbox, EVAL or CODY row or record was read, and no G12 comparison is made here.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {who}, {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,}. Tokens: tokenizer and encoder only, {res['inference']['window_s']:g} s windows every "
         f"{res['inference']['hop_s']:g} s, each 0.25 s patch read from its most central window; device "
         f"{res['inference']['device']}."]
    tr = res.get("training")
    if tr:
        c = tr.get("compute") or {}
        L.append(f"Training run {tr.get('run_dir')}: {tr.get('verdict')}; clock {fmt(c.get('wall_s'), 0)} s against the "
                 f"budget {fmt(c.get('budget_wall_s'), 0)} s, ratio {fmt(c.get('ratio'), 4)}, optimizer steps "
                 f"{c.get('steps')}, tokens {c.get('tokens')}; the checkpoint read is the run's verified final checkpoint: "
                 f"{res.get('final_verified')}.")
    L += [f"Fit: the {len(res['sets']['train'])} training-fold controls ({res['sets']['train_among_checkpoint_persons']} "
          "of them among the checkpoint's training persons, whose windows it saw without labels); score: the "
          f"{len(res['sets']['dev14'])} dev controls and the {len(res['sets']['dev12'])} of the harness's set dev12. "
          "Variant of each line: the candidate (latent, measurement) with the lower person cross-fitted line on the "
          "fitting set; ridge penalty by person folds inside the fitting set.", "",
          "## G3 dev lines of the chosen variants", "",
          "| line | variant | cross-fitted on the fitting set, latent / measurement | dev14 | 95 % interval | matched "
          "share | dev12 | 95 % interval | population dev14 / dev12 |", "|---|---|---|---|---|---|---|---|---|"]
    for ln, title, nd in LINE_TITLES:
        a, b = res["lines"]["dev14"][ln], res["lines"]["dev12"][ln]
        cf = res["crossfit"][ln]
        pop = [((res.get("population") or {}).get(s) or {}).get(ln, {}).get("point") for s in ("dev14", "dev12")]
        L.append(f"| {title} | {a['variant']} | {fmt(cf.get('latent'), nd)} / {fmt(cf.get('measurement'), nd)} | "
                 f"{fmt(a.get('point'), nd)} | {fmt_iv(a.get('interval'), nd)} | {fmt(a.get('matched_share'), 3)} | "
                 f"{fmt(b.get('point'), nd)} | {fmt_iv(b.get('interval'), nd)} | "
                 f"{' / '.join(fmt(p, nd) for p in pop) if ln != 'ic_median_ms' else 'n/a'} |")
    L += ["", "Intervals: person-clustered bootstrap of the G3 harness (2,000 resamples, seed 20260929). The IC line "
          "subtracts one offset per variant fitted on the 14 (the harness's rule), on both sets.", "",
          "## Every pass-probe variant", "",
          "| variant | features | stride, cross-fitted | stride dev14 | stride dev12 | speed, cross-fitted | speed dev14 | "
          "speed dev12 | alpha stride | alpha speed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in res["variants"]["pass"].items():
        g = lambda s, ln: (r["dev"][s][ln] or {}).get("point")  # noqa: E731
        L.append(f"| {v}{' (candidate)' if v in RO.CANDIDATES else ''} | {r['n_features']} | "
                 f"{fmt(r['crossfit']['stride_mae_cm']['point'])} | {fmt(g('dev14', 'stride_mae_cm'))} | "
                 f"{fmt(g('dev12', 'stride_mae_cm'))} | {fmt(r['crossfit']['speed_mae_m_s']['point'], 3)} | "
                 f"{fmt(g('dev14', 'speed_mae_m_s'), 3)} | {fmt(g('dev12', 'speed_mae_m_s'), 3)} | "
                 f"{fmt(r['selection']['stride_m']['alpha'], 2)} | {fmt(r['selection']['speed_m_s']['alpha'], 2)} |")
    L += ["", "## Every IC-probe variant (walkway reference)", "",
          "| variant | cross-fitted (ms) | offset cross-fitted (ms) | dev14 (ms) | matched share | dev12 (ms) | offset dev "
          "(ms) | ICs read | alpha |", "|---|---|---|---|---|---|---|---|---|"]
    ic = res["variants"]["ic"]
    for v, r in ic["variants"].items():
        a, b = r["dev"].get("dev14", {}), r["dev"].get("dev12", {})
        L.append(f"| {v}{' (candidate)' if v in RO.CANDIDATES else ''} | {fmt(r['crossfit']['line'].get('point'), 1)} | "
                 f"{fmt(r['crossfit'].get('offset_ms'), 1)} | {fmt(a.get('point'), 1)} | {fmt(a.get('matched_share'), 3)} | "
                 f"{fmt(b.get('point'), 1)} | {fmt(r.get('offset_ms'), 1)} | {r.get('n_ics')} | {fmt(r.get('alpha'), 2)} |")
    rw = res.get("rows") or {}
    L += ["", f"Labelled training rows {ic.get('labelled_rows')} of {ic.get('rows_train')}; dev rows {ic.get('rows_dev')}. "
          "A value reads inf when a person has more than half of its reference ICs unmatched (as in g3).", "",
          "## Per-row predictions", "",
          f"{rw.get('path')} (SHA-256 {rw.get('sha256')}): {rw.get('n_rows')} rows (every walkway stride, pass and walkway "
          "reference IC of the 14, for every variant, with its person; the chosen variants flagged). The lines "
          "recomputed from these rows equal the G3 harness's points on both sets for every variant and line "
          f"({(rw.get('check') or {}).get('checked')} checked, largest absolute difference "
          f"{(rw.get('check') or {}).get('max_abs_difference')}).", "",
          "## Counts", "", "| set | records | windows | passes | passes with rows | rows per pass (median) | token rows kept |",
          "|---|---|---|---|---|---|---|"]
    for s in ("train", "dev"):
        c = res["counts"][s]
        L.append(f"| {s} | {c['records']} | {c['windows']} | {c['passes']} | {c['passes_with_rows']} | "
                 f"{fmt(c.get('rows_per_pass_median'), 1)} | {c['token_rows_kept']} |")
    L += ["", f"Lockbox: persons of either set in the weargait_pd lockbox list {res['lockbox']['persons_in_lockbox']}; "
          f"target rows kept {res['lockbox']['target_lockbox_rows']} (asserted 0; records read with every reader flag "
          "False and checked one by one).",
          f"MoveAtlas: HEAD {res.get('moveatlas_head') or 'n/a'}. Time: tokens {fmt(res['timing_s'].get('tokens'), 1)} s, "
          f"probes {fmt(res['timing_s'].get('probes'), 1)} s, all {fmt(res['timing_s'].get('all'), 1)} s.", ""]
    return "\n".join(L) + "\n"


def write_outputs(out, res, res0, prov):
    """per_row.csv, per_pass.csv, results.json (with the rows file's SHA-256), summary.md, _provenance.json."""
    from mdfm.gates import common
    rows_path = common.write_csv(res0["rows"], os.path.join(out, "per_row.csv"))
    common.write_csv(res0["per_pass"], os.path.join(out, "per_pass.csv"))
    res["rows"] = {"path": common.rel(rows_path), "sha256": common.sha256_file(rows_path), "n_rows": int(len(res0["rows"])),
                   "check": res0["rows_check"]}
    res = clean(res)
    text = report_md(res)
    common.write_text(os.path.join(out, "results.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(out, "summary.md"), text)
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    return res, text


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="a G12 baseline checkpoint or a body checkpoint")
    g.add_argument("--run-dir", help="a G12 baseline run folder: its verified final checkpoint, when FINISHED")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=12.0, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--out", default=None, help="default reports/bench/g12/readouts/<name>")
    p.add_argument("--features", default=None, help="default cache/runs/g12_readouts/<name>/features.npz")
    p.add_argument("--from-features", action="store_true", help="refit from the feature file, no model")
    p.add_argument("--limit-persons", type=int, default=0, help="smoke: the first N persons of each set (needs --out)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import torch

    from mdfm import paths
    from mdfm.baselines import comparison as CMP
    from mdfm.baselines import load as BL
    from mdfm.baselines import readouts as RO
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.eval import probes
    from mdfm.gates import common, g3
    from mdfm.model import encoder as ENC
    from mdfm.model import tokenize as TK
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    training = None
    if a.run_dir:
        ckpt, training = resolve_run_dir(a.run_dir)
    else:
        ckpt = a.checkpoint if os.path.isabs(a.checkpoint) else os.path.join(TREE, a.checkpoint)
    name = I.checkpoint_name(ckpt)
    if a.limit_persons and not a.out:
        raise SystemExit("--limit-persons needs an explicit --out (a partial run never takes the default folder)")
    out = os.path.abspath(a.out or os.path.join(OUT_ROOT, name))
    paths.assert_not_moveatlas(out)
    features = os.path.abspath(a.features or os.path.join(
        FEATURE_ROOT, name + (f"_limit{a.limit_persons}" if a.limit_persons else ""), "features.npz"))
    code_files = [RO.__file__, CMP.__file__, BL.__file__, probes.__file__, os.path.abspath(__file__), I.__file__,
                  g3.__file__, TK.__file__, ENC.__file__]
    code_sha = common.code_sha256(code_files)

    train, dev = g3.controls("train"), g3.controls("dev")
    if len(train) != FOLD_SIZES["train"] or len(dev) != FOLD_SIZES["dev"] or set(train) & set(dev):
        raise SystemExit(f"folds: train {len(train)}, dev {len(dev)}, overlap {sorted(set(train) & set(dev))}")
    locked = set(g3.lockbox_persons())
    if (set(train) | set(dev)) & locked:
        raise SystemExit("a lockbox person among the non-lockbox controls")
    if a.limit_persons:
        train = train[:max(int(a.limit_persons), 2 * probes.FOLDS)]        # nested folds need enough persons
        dev = dev[:max(int(a.limit_persons), 2)]
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise SystemExit("a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    print(f"checkpoint {name}; targets: train {tg_train.counts['passes']} passes, dev {tg_dev.counts['passes']}",
          flush=True)

    if a.from_features:
        data, meta = RO.load_features(features)
        check_features_meta(meta, name, common.rel(features))
        for s, u in (("train", u_tr), ("dev", u_dev)):
            f = data["units"][s]
            if len(f) != len(u) or list(f["trial_id"]) != list(u["trial_id"]) or \
                    list(f["pass_id"].astype(str)) != list(u["pass_id"].astype(str)):
                raise SystemExit(f"{common.rel(features)}: {s} passes differ from the targets read now")
            data["units"][s] = u
        ck, inference, kind = meta["checkpoint"], meta["inference"], meta["kind"]
        train_in_ckpt, t_tok = meta["train_among_checkpoint_persons"], meta.get("tokens_s")
        training = training or meta.get("training")
        print(f"features read from {common.rel(features)} (written {meta.get('written_at')})", flush=True)
    else:
        dev_t = torch.device(a.device)
        if dev_t.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda without a visible GPU (the GPU runs through scripts/queue_job.py)")
        cap = LP.cap_memory(dev_t, a.memory_gb)
        if dev_t.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev_t)
        kind, model, info = RO.load_any(ckpt, weights=a.weights, device=dev_t)
        trained = info["trained_persons"]
        if set(dev) & trained:
            raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev) & trained)}")
        train_in_ckpt = len(set(train) & trained)
        ck = info["describe"]
        if kind == "baseline" and training is None:
            training = training_of(ckpt)
        print(f"loaded {name} ({ck['sha256'][:12]}), {kind} {ck.get('arm')} seed {ck.get('seed')}, {ck['weights']}, "
              f"phase {ck['phase']} step {ck['step']}; training-fold controls among its training persons: "
              f"{train_in_ckpt}", flush=True)

        def encode(wins):
            return RO.encode_windows(model, wins, a.batch_windows)

        def records(persons, tg):
            for tid, rec in g3.imu_records(persons, tg.trials):          # canonical.load, reader flags False
                yield tid, g3.model_input(rec)

        t0 = time.time()
        data = {"units": {"train": u_tr, "dev": u_dev}}
        data["train"] = RO.collect(encode, records(train, tg_train), u_tr, tg_train, train)
        data["dev"] = RO.collect(encode, records(dev, tg_dev), u_dev, tg_dev, dev)
        t_tok = round(time.time() - t0, 1)
        mem = {}
        if dev_t.type == "cuda":
            mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev_t) / 1e9,
                   "max_reserved_gb": torch.cuda.max_memory_reserved(dev_t) / 1e9, "cap_gb": cap}
        inference = {"window_s": I.WIN_S, "hop_s": I.HOP_S, "batch_windows": a.batch_windows, "weights": a.weights,
                     "device": str(dev_t), "amp_bf16": dev_t.type == "cuda", "refine_steps": 0, "decoder": False,
                     "parts": "tokenizer and encoder only", "memory": mem}
        meta = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": ck, "kind": kind, "inference": inference,
                "training": training, "train_among_checkpoint_persons": train_in_ckpt, "tokens_s": t_tok,
                "code_sha256": code_sha, "limit_persons": a.limit_persons}
        RO.save_features(features, data, meta)
        print(f"features written to {common.rel(features)} ({t_tok} s)", flush=True)
        del model

    t0 = time.time()
    res0 = RO.run_probes(data, tg_train, tg_dev)
    t_probe = round(time.time() - t0, 1)
    _, tgs = RO.dev_sets(tg_dev)
    constants = {"alphas": list(probes.ALPHAS), "folds": probes.FOLDS, "probe_seed": probes.SEED, "g3_seed": g3.SEED,
                 "g3_boot_n": g3.BOOT_N, "sub_bins": probes.SUB_BINS, "ic_threshold": probes.IC_THRESHOLD,
                 "ic_nms_s": probes.IC_NMS_S, "span_margin_s": probes.SPAN_MARGIN_S, "keep_pad_s": RO.KEEP_PAD_S,
                 "pool": list(probes.POOL), "candidates": list(RO.CANDIDATES), "variants": list(RO.VARIANTS),
                 "ic_variants": list(RO.IC_VARIANTS), "ic_match_s": g3.IC_MATCH_S}
    lockbox = {"persons_in_lockbox": len((set(train) | set(dev)) & locked),
               "target_lockbox_rows": int(tg_train.counts["lockbox_rows"]) + int(tg_dev.counts["lockbox_rows"]),
               "reader_flags": {"allow_eval": False, "allow_lockbox": False}}
    counts = {"train": data["train"]["counts"], "dev": data["dev"]["counts"],
              "targets": {"train": tg_train.counts, "dev": tg_dev.counts}}
    res = build_results(res0, ck, training, kind, inference,
                        {"path": common.rel(features), "sha256": common.sha256_file(features)},
                        {"train": train}, train_in_ckpt, counts, constants, lockbox, population_lines(tg_train, tgs),
                        None, PV.moveatlas_head(), {"tokens": t_tok, "probes": t_probe,
                                                    "all": round(time.time() - t_start, 1)},
                        run_started, a.limit_persons)
    prov = {"script": "scripts/g12_readout.py", "label": LABEL, "task": TASK, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": res["moveatlas_head"], "python": sys.version.split()[0],
            "numpy": np.__version__, "torch": torch.__version__, "checkpoint": ck.get("path"),
            "checkpoint_sha256": ck.get("sha256"), "features": res["features"], "code_sha256": code_sha,
            "code_files": [common.rel(f) for f in code_files],
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    res, text = write_outputs(out, res, res0, prov)
    print(text)
    print(f"written {common.rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
