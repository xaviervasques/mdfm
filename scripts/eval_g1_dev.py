"""B4.19: the G1 dev table of a checkpoint (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 6 and 7; harness
mdfm/gates/g1.py, rule docs/rules/G1_RULE.md DRAFT; inference mdfm/eval/infer.py).

Inputs: the G1 truth and inputs of scripts/build_g1_inputs.py on the dev archives (EyesJapanDataset, HDM05, SFU)
and the calib archives (DanceDB, EKUT, HUMAN4D, HumanEva), cache/g1/g1_v1 by default; never the lockbox. The
checkpoint is read through mdfm.eval.infer.load_checkpoint (checkpoint.load: file, config, person-list, records
and lockbox hashes, the person guard, the state_dict audit; then the audit of the loaded weights); its training
persons (the amass list) are asserted disjoint from the dev and calib persons.

Model (infer.G1Predictor, rule section 4): encode, 20 refinement steps with 256 points per stream per step, decode
every knot; the skeleton of the take's recorded gender; the shape pooled per take. G1a: the posterior-mean body of
the six-IMU input; G1c: that of the camera input; G1b: 16 latent samples x 4 decoder draws = 64 left-wrist
positions per knot (pelvis-relative, pelvis-heading frame) of the single-wrist input, and the same from the
six-IMU input for the band-width reference.
Conformal factors (rule section 6): the single-wrist and the six-IMU wrist predictives fitted on the calib
windows (infer.fit_conformal: every calib knot a Mahalanobis radius, split conformal at 0.90), per run by default.
B4.20, both opt-in:
- --save-conformal writes the factors of this run's fit to cache/runs/<run>/conformal_<sha12>.json beside the
  checkpoint (infer.conformal_record and save_conformal: the checkpoint SHA-256, the settings of the wrist
  predictives, the calib archives and their manifest SHA-256, both factors). The record is built right after the
  calib fit, and the file is written last, after every output of the run: g1.json names it with the SHA-256 of
  its bytes (infer.conformal_text, computed before the write and checked after it), and a run that fails before
  its end leaves no file, so the same job can run again. Refused with --limit-takes (the saved factor is fitted
  on every calib window), with --conformal-file, and when the file exists (checked before any inference).
- --conformal-file <path> skips the calib fit and takes both factors from that file (G1Predictor.use_conformal_file:
  a file of another checkpoint, of other predictive settings or of other calib inputs is refused). Never automatic:
  without the flag the factors are fitted in the run, so G1b stays comparable across checkpoints.
g1.json (conformal_source) and g1.md name the factor source: fitted in this run, or the file with its SHA-256.
Scoring: g1.score on the dev windows; g1.lines against the population prior of
reports/gates/g1/population_prior/prior.json scored on the same windows (the trivial baseline, with its own
calib factor; the reference of the energy-score bound); g1.secondary; per person and per archive. These are dev
numbers under a draft rule, not a gate read: lockbox_reads.csv is not touched.

--limit-takes K keeps the first K takes of each archive of both sets (a partial run, labelled so in the outputs).
A checkpoint that does not close phase A is labelled a partial run of phase A.

--secondary-rows (opt-in; without it not_evaluated lists the four rows, NOT_EVALUATED): the four model rows of
G1_RULE section 8 that the harness table does not hold, computed with the row functions of
scripts/probe_g1_decompose.py (task T05, imported) on the scored dev windows, with this run's settings, blocks and
seeds (secondary_model_rows):
- zero refinement steps (the amortization gap): T05 main_pass without best-of draws, its refine0 column (the
  amortized posterior of the same encoder pass), paired per window with the main pass's own G1a; the main pass's
  G1a is compared with the harness table (harness_check);
- truth re-rendered and re-scored: the same pass's floor column (true local rotations and betas through the
  model's skeleton, root yaw through the decoder's path) and floor_kinematic (without the yaw path); truth
  rotations rebuilt from the cached raw takes (T05 TruthBodies, checked against the stored truth);
- acc-only wrist: T05 acc_only_wrist, its own conformal factor on the calib windows of the run, coverage, band
  width and energy score on dev, beside the acc and gyro wrist of this run on the same windows;
- analysis by synthesis: T05 mean_pass, --abs-steps (200) refinement steps from the prior with the encoder
  bypassed, on the first --abs-limit-takes (2; 0 for every scored window) takes of each dev archive, labelled
  partial, paired with the encoder start at the run's step count in the same partial blocks and seeds.
The four leave not_evaluated; the per-window values go to secondary_windows.csv, acc_only_windows.csv and
abs_windows.csv. With or without the flag, not_evaluated also lists the rows that G1_RULE section 8 gained on
2026-10-05 and that this script does not compute (NOT_COMPUTED_HERE: G1b under a factor fitted on one score per
calib person, the valid camera detection share beside G1c, the lockbox treadmill count), so that no g1.json reads
as holding every row of section 8.

Writes <out> (default reports/gates/g1/dev_<checkpoint name>/): g1.json (lines, secondary rows, conformal fits,
the prior's lines on the same windows, sets, checkpoint, inference settings, timing), g1.md, per_person.csv,
per_archive.csv, windows.csv (every scored dev window), blocks.csv (per inference block: takes, windows, the
refinement objective of each input, seconds), _provenance.json (MoveAtlas HEAD read from its .git files).

Run (mdfm environment; the GPU is shared with training, so the caching allocator is capped with --memory-gb):
  <conda>/envs/mdfm/python.exe scripts/eval_g1_dev.py --checkpoint <run>/ckpt_A_<step>.pt
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

PRIOR = os.path.join(TREE, "reports", "gates", "g1", "population_prior", "prior.json")
MM = 1000.0
ABS_LIMIT_TAKES = 2             # takes per dev archive of the analysis-by-synthesis row (partial, as T05)
ABS_STEPS = 200                 # refinement steps of analysis by synthesis (rule section 7, T05 ABS_STEPS)
NOT_EVALUATED = ["zero refinement steps (a run with --refine-steps 0)", "acc-only wrist",
                 "truth re-rendered and re-scored", "analysis by synthesis"]
# rows of G1_RULE section 8 added on 2026-10-05 that this script computes in no mode (listed in every g1.json)
NOT_COMPUTED_HERE = [
    "G1b coverage, band width and energy score under a conformal factor fitted on one score per calib person "
    "(G1_RULE section 8, added 2026-10-05): not computed by scripts/eval_g1_dev.py",
    "share of valid camera detections beside every G1c number (G1_RULE section 8, added 2026-10-05): not computed by "
    "scripts/eval_g1_dev.py",
    "count of lockbox takes flagged as treadmill (G1_RULE section 8, added 2026-10-05): a lockbox row, not computed "
    "on dev"]


def not_evaluated(secondary_rows):
    """The not_evaluated list of g1.json: the four model rows of NOT_EVALUATED unless --secondary-rows computed
    them, then the rows of NOT_COMPUTED_HERE."""
    return ([] if secondary_rows else list(NOT_EVALUATED)) + list(NOT_COMPUTED_HERE)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--root", default=None, help="G1 input root (default mdfm.gates.g1.default_root)")
    p.add_argument("--prior", default=PRIOR)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256, help="kernel points per stream-window and refinement step")
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--block-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a take")
    p.add_argument("--limit-takes", type=int, default=None, help="the first K takes of each archive (partial run)")
    p.add_argument("--out", default=None)
    p.add_argument("--save-conformal", action="store_true",
                   help="write this run's factors to cache/runs/<run>/conformal_<sha12>.json (B4.20)")
    p.add_argument("--conformal-file", default=None, help="take both factors from this saved file (opt-in)")
    p.add_argument("--secondary-rows", action="store_true",
                   help="the four model rows of G1_RULE section 8 through scripts/probe_g1_decompose.py")
    p.add_argument("--abs-limit-takes", type=int, default=ABS_LIMIT_TAKES,
                   help="takes per dev archive of the analysis-by-synthesis row (0: every scored window)")
    p.add_argument("--abs-steps", type=int, default=ABS_STEPS, help="refinement steps of analysis by synthesis")
    return p.parse_args(argv)


def check_args(a):
    """None when the options agree, else the reason they do not (B4.20 rules of the module docstring)."""
    if a.save_conformal and a.limit_takes is not None:
        return "--save-conformal needs every calib window: not with --limit-takes"
    if a.save_conformal and a.conformal_file:
        return "--save-conformal fits the factors, --conformal-file reads them: give one"
    if a.abs_limit_takes is not None and a.abs_limit_takes < 0:
        return f"--abs-limit-takes {a.abs_limit_takes}: 0 or more"
    return None


def save_conformal_last(I, common, path, record, sha256):
    """--save-conformal, the last step of a run: write the record (I: mdfm.eval.infer, written once, refused when
    the file exists) and check, before and after the write, that the file's bytes have the SHA-256 that g1.json
    already states; returns the path."""
    want = common.sha256_bytes(I.conformal_text(record).encode("utf-8"))
    if want != sha256:
        raise RuntimeError(f"eval_g1_dev: the conformal record has SHA-256 {want}, g1.json states {sha256}")
    p = I.save_conformal(path, record)
    got = common.sha256_file(p)
    if got != sha256:
        raise RuntimeError(f"eval_g1_dev: {common.rel(p)} has SHA-256 {got}, g1.json states {sha256}")
    return p


def t05():
    """scripts/probe_g1_decompose.py (task T05), whose row functions compute the secondary rows."""
    here = os.path.join(TREE, "scripts")
    if here not in sys.path:
        sys.path.insert(0, here)
    import probe_g1_decompose
    return probe_g1_decompose


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}"


def subset(gset, k):
    """Window indices of the first k takes of each archive (index order); every window when k is None."""
    w = gset.windows
    if k is None:
        return list(range(len(w)))
    keep = set()
    for _, g in w.groupby("archive", sort=True):
        keep |= set(list(dict.fromkeys(g["take_id"].tolist()))[:int(k)])
    return [int(i) for i in range(len(w)) if w["take_id"].iat[i] in keep]


KEYS = ["window", "take_id", "archive", "person_id", "task", "n"]


def secondary_model_rows(T, loaded, calib, dset, idx_cal, idx_dev, table, a, log=None, truth_of=None, draws=None):
    """The four model rows of G1_RULE section 8 (module docstring, --secondary-rows) through T, the module
    scripts/probe_g1_decompose.py: (sec, tables). sec: {rows {zero_refinement_steps, truth_rerendered,
    truth_rerendered_kinematic, analysis_by_synthesis (with its reference), acc_only_wrist}, harness_check,
    truth_check, timing_s, code}; each G1a row is T.row_summary (point and interval in mm, change = the run's G1a
    minus the row, paired per window). tables: the per-window values {secondary_windows, acc_only_windows,
    abs_windows}. truth_of: the truth rotations of a dev window (default T.TruthBodies(dset)); draws: {samples,
    decoder_draws} of the acc-only predictive (default the G1 parameters)."""
    log = log or (lambda *m: None)
    kw = dict(batch_windows=a.batch_windows, block_windows=a.block_windows, seed=a.seed, pool=not a.no_pool)
    rows, tabs, timing = {}, {}, {}
    steps = int(a.refine_steps)

    # zero refinement steps and truth re-rendered: one main pass of T05 (no best-of draws) on the scored windows
    t0 = time.time()
    truth = T.TruthBodies(dset) if truth_of is None else truth_of
    mp, _ = T.main_pass(loaded, dset, idx_dev, truth, steps, a.points, best_of=0, log=log, **kw)
    timing["main_pass"] = round(time.time() - t0, 1)
    m = table[["window", "g1a_m"]].merge(mp[["window", "g1a_m"]], on="window", suffixes=("_harness", ""))
    harness = {"windows": int(len(m)),
               "g1a_mm_harness": MM * T.pwm(table, "g1a_m"), "g1a_mm_main_pass": MM * T.pwm(mp, "g1a_m"),
               "max_window_difference_mm": float(MM * (m["g1a_m"] - m["g1a_m_harness"]).abs().max()) if len(m)
               else None}
    harness["difference_mm"] = harness["g1a_mm_main_pass"] - harness["g1a_mm_harness"]
    rows["zero_refinement_steps"] = T.row_summary(
        mp, "refine0_m", "g1a_m", f"G1a of the amortized posterior (0 refinement steps), the encoder pass of the "
        f"{steps}-step row")
    rows["truth_rerendered"] = T.row_summary(
        mp, "floor_m", "g1a_m", "truth re-rendered (true local rotations and betas through the model's skeleton, "
        "root yaw through the decoder's path) and re-scored")
    rows["truth_rerendered_kinematic"] = T.row_summary(
        mp, "floor_kin_m", "g1a_m", "truth through the model's skeleton and kinematics only (no root yaw path)")
    tabs["secondary_windows"] = mp[KEYS + ["g1a_m", "refine0_m", "floor_m", "floor_kin_m"]]
    tc = None
    if hasattr(truth, "max_dev_m"):
        tc = {"takes": int(truth.n_takes), "windows": int(truth.n_windows), "max_deviation_mm": MM * truth.max_dev_m,
              "tolerance_mm": MM * truth.tol}

    # acc-only wrist: its own conformal factor on the calib windows of the run, scored on the dev windows
    t0 = time.time()
    dk = dict(draws or {})
    fit = T.acc_only_wrist(loaded, calib, idx_cal, steps, a.points, log=log, **dk, **kw)
    accw = T.acc_only_wrist(loaded, dset, idx_dev, steps, a.points, c=fit["c"], log=log, **dk, **kw)
    timing["acc_only"] = round(time.time() - t0, 1)
    lo, hi = T.boot(accw, "coverage")
    j = accw.merge(table[["window", "coverage", "width_m", "energy_m"]], on="window", suffixes=("", "_acc_gyro"))
    d = j["coverage"].to_numpy(dtype=float) - j["coverage_acc_gyro"].to_numpy(dtype=float)
    dl, dh = T.boot(j, "coverage", d)
    ns = dk.get("samples", T.g1.PARAMS["latent_samples"])
    nd = dk.get("decoder_draws", T.g1.PARAMS["decoder_draws"])
    rows["acc_only_wrist"] = {
        "what": f"single-wrist line with the wrist gyro masked (acc only): {ns} latent samples x {nd} decoder draws, "
                "its own conformal factor on the calib windows of the run",
        "conformal": fit, "calib_windows": len(idx_cal), "coverage": T.pwm(accw, "coverage"),
        "coverage_interval": [lo, hi], "width_mm": MM * T.pwm(accw, "width_m"),
        "energy_mm": MM * T.pwm(accw, "energy_m"),
        "degenerate": int(accw["degenerate"].sum()), "n_knots": int(accw["n"].sum()), "n_windows": int(len(accw)),
        "n_persons": int(accw["person_id"].nunique()),
        "acc_gyro_same_windows": {"coverage": T.pwm(j, "coverage_acc_gyro"),
                                  "width_mm": MM * T.pwm(j, "width_m_acc_gyro"),
                                  "energy_mm": MM * T.pwm(j, "energy_m_acc_gyro"), "windows": int(len(j))},
        "coverage_delta": T.pwm(j, "coverage") - T.pwm(j, "coverage_acc_gyro"), "coverage_delta_interval": [dl, dh],
        "width_delta_mm": MM * (T.pwm(j, "width_m") - T.pwm(j, "width_m_acc_gyro")),
        "energy_delta_mm": MM * (T.pwm(j, "energy_m") - T.pwm(j, "energy_m_acc_gyro"))}
    tabs["acc_only_windows"] = accw

    # analysis by synthesis: from the prior, encoder bypassed, on the partial windows, against the encoder start
    t0 = time.time()
    k = int(a.abs_limit_takes) if a.abs_limit_takes else None
    in_run = set(int(i) for i in idx_dev)
    pidx = [i for i in subset(dset, k) if i in in_run]
    label = None if k is None else f"first {k} takes of each dev archive ({len(pidx)} windows)"
    if pidx:
        ref, _ = T.mean_pass(loaded, dset, pidx, steps, "encoder", a.points, name="reference_m", log=log, **kw)
        ab, _ = T.mean_pass(loaded, dset, pidx, a.abs_steps, "prior", a.points, name="abs_m", log=log, **kw)
        part = ref.merge(ab[["window", "abs_m"]], on="window")
        r = T.row_summary(part, "abs_m", "reference_m", f"G1a, analysis by synthesis: {a.abs_steps} refinement steps "
                          "from the prior, encoder bypassed", label)
        r["reference"] = T.row_summary(part, "reference_m", None, f"G1a, {steps} refinement steps from the encoder, "
                                       "the same windows in the same partial blocks and seeds", label)
        rows["analysis_by_synthesis"] = r
        tabs["abs_windows"] = part
    else:
        rows["analysis_by_synthesis"] = {"what": "analysis by synthesis", "partial": label, "point_mm": None,
                                         "reason": "no scored window among the partial takes"}
    timing["analysis_by_synthesis"] = round(time.time() - t0, 1)
    sec = {"what": "the model rows of docs/rules/G1_RULE.md section 8 that the harness table does not hold, through "
                   "the row functions of scripts/probe_g1_decompose.py (task T05)",
           "rows": rows, "harness_check": harness, "truth_check": tc, "timing_s": timing,
           "code": "scripts/probe_g1_decompose.py", "abs_steps": int(a.abs_steps), "abs_limit_takes": k}
    return sec, tabs


def secondary_markdown(sec, steps, fmt):
    """The markdown lines of the secondary model rows (sec of secondary_model_rows)."""
    R = sec["rows"]
    L = ["## Secondary rows of the model (G1_RULE section 8, no kill)", "",
         f"Computed with the row functions of {sec['code']} (task T05) on the scored dev windows, with the blocks, "
         "seeds and settings of this run. Value: the G1 statistic in mm; interval: person-clustered bootstrap, 95 "
         f"percent; change: the {steps}-step G1a minus the row, paired per window (for analysis by synthesis: the "
         "encoder start in the same partial blocks minus the row).", "",
         "| row | what | value mm | 95% interval | change mm | change interval | windows | set |",
         "|---|---|---|---|---|---|---|---|"]
    iv = lambda x, nd=1: "n/a" if not x or x[0] is None else f"{fmt(x[0], nd)} to {fmt(x[1], nd)}"  # noqa: E731
    order = [("zero refinement steps", R.get("zero_refinement_steps")),
             ("truth re-rendered and re-scored", R.get("truth_rerendered")),
             ("truth through the model kinematics only", R.get("truth_rerendered_kinematic")),
             ("analysis by synthesis", R.get("analysis_by_synthesis")),
             ("encoder start, same partial windows", (R.get("analysis_by_synthesis") or {}).get("reference"))]
    for name, r in order:
        if not r:
            continue
        L.append(f"| {name} | {r['what']} | {fmt(r.get('point_mm'))} | {iv(r.get('interval_mm'))} | "
                 f"{fmt(r.get('delta_mm'))} | {iv(r.get('delta_interval_mm'))} | {r.get('n_windows', 0):,} | "
                 f"{'partial: ' + r['partial'] if r.get('partial') else 'every scored window'} |")
    acc = R["acc_only_wrist"]
    ref = acc["acc_gyro_same_windows"]
    L += ["", f"Acc-only wrist (gyro masked): conformal factor on {acc['calib_windows']:,} calib windows c = "
          f"{fmt(acc['conformal']['c'], 4)} over {acc['conformal']['n_frames']:,} knots; coverage "
          f"{fmt(acc['coverage'], 3)} ({iv(acc['coverage_interval'], 3)}), band width {fmt(acc['width_mm'])} mm, "
          f"energy score {fmt(acc['energy_mm'])} mm, degenerate knots {acc['degenerate']} of {acc['n_knots']:,}. "
          f"The acc and gyro wrist of this run on the same {ref['windows']:,} windows: coverage "
          f"{fmt(ref['coverage'], 3)}, band width {fmt(ref['width_mm'])} mm, energy score {fmt(ref['energy_mm'])} mm; "
          f"acc-only minus acc and gyro: coverage {fmt(acc['coverage_delta'], 3)} "
          f"({iv(acc['coverage_delta_interval'], 3)}), band width {fmt(acc['width_delta_mm'])} mm, energy score "
          f"{fmt(acc['energy_delta_mm'])} mm.", ""]
    h = sec["harness_check"]
    L.append(f"Harness check: the main pass of these rows gives G1a {fmt(h['g1a_mm_main_pass'], 2)} mm against "
             f"{fmt(h['g1a_mm_harness'], 2)} mm in the table above (difference {fmt(h['difference_mm'], 2)} mm, "
             f"largest per-window difference {fmt(h['max_window_difference_mm'], 3)} mm over {h['windows']:,} "
             "windows).")
    tc = sec.get("truth_check")
    if tc:
        L[-1] += (f" Truth rotations rebuilt from {tc['takes']:,} raw takes, largest deviation from the stored truth "
                  f"joints {fmt(tc['max_deviation_mm'], 4)} mm (tolerance {fmt(tc['tolerance_mm'], 1)} mm).")
    L += ["", "Time (s): " + ", ".join(f"{k} {v}" for k, v in sec["timing_s"].items()) + ".", ""]
    return L


def main():
    a = parse()
    why = check_args(a)
    if why:
        raise SystemExit(f"eval_g1_dev: {why}")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import pandas as pd
    import torch

    from mdfm import paths
    from mdfm.data import amass_index
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.gates import common, g1
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    code_files = [g1.__file__, common.__file__, os.path.join(TREE, "mdfm", "eval", "conformal.py"), I.__file__,
                  os.path.abspath(__file__)]
    if a.secondary_rows:
        code_files.append(os.path.join(TREE, "scripts", "probe_g1_decompose.py"))
    code_sha = common.code_sha256(code_files)
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    out = a.out or os.path.join(TREE, "reports", "gates", "g1", f"dev_{loaded.name}")
    paths.assert_not_moveatlas(out)
    save_to = I.conformal_path(loaded.path, loaded.sha256) if a.save_conformal else None
    if save_to and os.path.exists(save_to):
        raise SystemExit(f"eval_g1_dev: {common.rel(save_to)} exists; a saved conformal file is written once")
    t_load = time.time() - t_start
    print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step} "
          f"of {loaded.phase_steps}; load {t_load:.1f} s", flush=True)

    calib, dset = g1.G1Set("calib", a.root), g1.G1Set("dev", a.root)
    kw = dict(refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
              block_windows=a.block_windows, seed=a.seed, pool=not a.no_pool)
    pred = I.G1Predictor(loaded, configs=I.G1_CONFIGS, **kw)
    train = set(pred.train_persons)
    for gs in (calib, dset):
        both = sorted(set(gs.persons) & train)
        if both:
            raise Refused(f"persons of the {gs.split} set among the checkpoint's training persons: {both[:3]}")
        if set(gs.persons) & set(amass_index.LOCKBOX_PERSONS):
            raise Refused(f"a lockbox person in the {gs.split} inputs")
    idx_cal, idx_dev = subset(calib, a.limit_takes), subset(dset, a.limit_takes)
    log = lambda *m: print(*m, flush=True)  # noqa: E731

    # ---------------- conformal factors on calib (both wrist predictives in one pass), or a saved file (opt-in)
    t0 = time.time()
    cal_pred, save_rec = None, None
    if a.conformal_file:
        doc = pred.use_conformal_file(a.conformal_file, calib)
        fits = {k: dict(doc["factors"][k]) for k in I.CONFORMAL_KEYS}
        source = {"kind": "file", "path": common.rel(a.conformal_file), "file_sha256": doc["file_sha256"],
                  "created": doc.get("created"), "calib_windows": doc["calib"]["windows"],
                  "calib_archives": doc["calib"]["archives"]}
        source_text = (f"read from {source['path']} (SHA-256 {source['file_sha256']}, written {source['created']}, "
                       f"fitted on {source['calib_windows']:,} calib windows; no calib pass in this run)")
    else:
        cal_pred = I.G1Predictor(loaded, configs=("six", "wrist"), **kw)
        cal_pred.subset = idx_cal
        fits = I.fit_conformal(calib, cal_pred, windows=idx_cal, log=log)
        pred.c_wrist, pred.c_six = fits["wrist"]["c"], fits["wrist_six"]["c"]
        source = {"kind": "fitted in this run", "calib_windows": len(idx_cal),
                  "calib_archives": sorted(calib.windows.iloc[idx_cal]["archive"].unique().tolist()),
                  "partial": a.limit_takes is not None, "saved_to": None}
        source_text = (f"fitted in this run on {len(idx_cal):,} calib windows"
                       + (" (partial run)" if a.limit_takes is not None else ""))
        if save_to:                     # the record now, the file at the end of the run (module docstring)
            save_rec = I.conformal_record(cal_pred, fits, calib, idx_cal,
                                          extra={"script": "scripts/eval_g1_dev.py", "run_code_sha256": code_sha,
                                                 "report": common.rel(out)})
            source["saved_to"] = common.rel(save_to)
            source["saved_sha256"] = common.sha256_bytes(I.conformal_text(save_rec).encode("utf-8"))
            source["saved_when"] = "at the end of the run, after every other output"
            source_text += (f"; saved to {source['saved_to']} (SHA-256 {source['saved_sha256']}) at the end of the "
                            "run")
    t_cal = time.time() - t0
    log(f"conformal: wrist c {pred.c_wrist:.4f}, six-IMU wrist c {pred.c_six:.4f} ({t_cal:.0f} s), {source['kind']}")

    # ---------------- dev scoring, the model and the population prior on the same windows
    t0 = time.time()
    pred.subset = idx_dev
    table = g1.score(dset, pred, windows=idx_dev, log=log)
    t_dev = time.time() - t0
    prior = g1.PopulationPrior.load(a.prior)
    table0 = g1.score(dset, prior, windows=idx_dev)
    with open(g1.RULE_PATH, "rb") as f:
        rule_sha = common.sha256_bytes(f.read())
    with open(g1.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())[0]
    res = g1.lines(table, table0, pred, rule_sha)
    res0 = g1.lines(table0, table0, prior, rule_sha)
    sec = g1.secondary(table)
    pp, pa = g1.per_group(table, "person_id"), g1.per_group(table, "archive")
    model_rows, sec_tabs, nev = None, {}, not_evaluated(a.secondary_rows)
    if a.secondary_rows:
        model_rows, sec_tabs = secondary_model_rows(t05(), loaded, calib, dset, idx_cal, idx_dev, table, a, log)
    blocks = []
    for r in (cal_pred.log if cal_pred is not None else []) + pred.log:
        row = {k: r[k] for k in ("split", "first_take", "takes", "windows")}
        for c in I.G1_CONFIGS:
            if c in r:
                row.update({f"{c}_refine_first": r[c]["refine_first"], f"{c}_refine_last": r[c]["refine_last"],
                            f"{c}_seconds": r[c]["seconds"]})
        blocks.append(row)
    blocks = pd.DataFrame(blocks)

    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    head = PV.moveatlas_head()
    partial = a.limit_takes is not None
    wall = time.time() - t_start
    ck = loaded.describe()
    kind = ("closes phase A" if loaded.phase_a_complete and loaded.phase == "A" else "phase B checkpoint"
            if loaded.phase == "B" else "phase C checkpoint (stage 2)" if loaded.phase == "C"
            else f"partial phase A (step {loaded.step} of {loaded.phase_steps[0] if loaded.phase_steps else '?'})")
    results = {
        "gate": g1.GATE, "run": "dev", "what": "G1 dev table (B4.19) under the draft rule, not a gate read",
        "partial_run": {"limit_takes": a.limit_takes} if partial else None,
        "checkpoint": ck, "checkpoint_kind": kind, "predictor": pred.name,
        "rule_path": common.rel(g1.RULE_PATH), "rule_sha256": rule_sha, "rule_status": status,
        "rule_registered": common.registered(g1.GATE, rule_sha),
        "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "latent_samples": pred.samples,
                      "decoder_draws": pred.decoder_draws, "batch_windows": a.batch_windows,
                      "block_windows": a.block_windows, "seed": a.seed,
                      "shape_pooling": "per take" if not a.no_pool else "none", "weights": a.weights,
                      "device": str(dev), "amp_bf16": dev.type == "cuda", "memory": mem},
        "lines": [r.to_dict() for r in res], "secondary": sec,
        "population_prior_same_windows": [r.to_dict() for r in res0],
        "conformal": fits, "conformal_source": source,
        "sets": {"dev": {"windows": len(idx_dev), "knots": int(table["n"].sum()),
                         "persons": int(table["person_id"].nunique()),
                         "takes": int(table["take_id"].nunique()), "manifest_sha256": dset.manifest["manifest_sha256"],
                         "persons_sha256": dset.manifest["persons_sha256"]},
                 "calib": {"windows": len(idx_cal), "persons": int(calib.windows.iloc[idx_cal]["person_id"].nunique()),
                           "manifest_sha256": calib.manifest["manifest_sha256"],
                           "persons_sha256": calib.manifest["persons_sha256"]},
                 "config_sha256": dset.manifest["config_sha256"]},
        "hashes": {"checkpoint_sha256": loaded.sha256, "train_persons_sha256": pred.train_persons_sha256,
                   "scoring_code_sha256": g1.code_sha256(g1.SCORE_CODE),
                   "input_code_sha256": sorted({s["code_sha256"] for s in dset.shards}),
                   "params_sha256": common.sha256_json(g1.PARAMS), "run_code_sha256": code_sha,
                   "run_code_files": [common.rel(f) for f in code_files]},
        "not_evaluated": nev,
        "lockbox_reads_logged": len(common.read_log()) if os.path.exists(common.READ_LOG) else 0,
        "moveatlas_head": head,
        "timing_s": {"load": round(t_load, 1), "calib": round(t_cal, 1), "dev": round(t_dev, 1), "all": round(wall, 1)},
    }
    if model_rows is not None:
        results["secondary_model_rows"] = model_rows

    # ---------------- markdown
    L = ["# G1 dev table (plan task B4.19)", "",
         "Written by scripts/eval_g1_dev.py (harness mdfm/gates/g1.py, inference mdfm/eval/infer.py); every number below "
         "is read from g1.json, per_person.csv, per_archive.csv and windows.csv of this folder. Development numbers "
         "under the draft rule on the dev archives, not a gate read (no lockbox array opened, lockbox_reads.csv "
         "untouched).", ""]
    if partial:
        L += [f"Partial run: the first {a.limit_takes} takes of each archive of the dev and calib sets.", ""]
    L += [f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
          f"{ck['step']:,}; {kind}. Training persons (amass) {len(train)}, list SHA-256 {pred.train_persons_sha256}, "
          "disjoint from the dev and calib persons (asserted). Hashes that moved since the save: "
          f"{', '.join(ck['drift']) or 'none'}.", "",
          f"Rule docs/rules/G1_RULE.md: status {status!r}, SHA-256 {rule_sha}. MoveAtlas HEAD {head}.", "",
          f"Set: {len(idx_dev):,} dev windows, {int(table['n'].sum()):,} knots of {table['person_id'].nunique()} persons in "
          f"{table['archive'].nunique()} archives (manifest SHA-256 {dset.manifest['manifest_sha256']}); calib "
          f"{len(idx_cal):,} windows.", "",
          f"Inference: {a.refine_steps} refinement steps of {a.points} points per stream, posterior-mean bodies for G1a "
          f"and G1c, {pred.samples} latent samples x {pred.decoder_draws} decoder draws for G1b, shape pooled "
          f"{results['inference']['shape_pooling']}; device {dev}"
          + (f", peak allocated {mem['max_allocated_gb']:.2f} GB, reserved {mem['max_reserved_gb']:.2f} GB under a "
             f"{cap} GB cap" if mem else "") + ".", "",
          "## Lines", "",
          "Point: the mean over persons of the knot-weighted person means; interval: person-clustered bootstrap, "
          f"{g1.PARAMS['bootstrap_n']} resamples, seed {g1.PARAMS['bootstrap_seed']}, 95 percent. Trivial baseline: the "
          "population prior on the same windows.", "",
          "| line | point | 95% interval | trivial baseline | pass | kill | verdict (dev) |", "|---|---|---|---|---|---|---|"]
    for r in res:
        nd = 3 if r.unit == "" else 1
        unit = f" {r.unit}" if r.unit else ""
        L.append(f"| {r.line} | {fmt(r.point, nd)}{unit} | {fmt(r.interval[0], nd)} to {fmt(r.interval[1], nd)} | "
                 f"{fmt(r.trivial_baseline, nd)}{unit} | {'at least' if r.higher_is_better else 'at most'} {r.pass_line:g} | "
                 f"{'below' if r.higher_is_better else 'above'} {r.kill_line:g} | {r.verdict} |")
    b = res[1].extra
    L += ["", f"G1b details: band width {fmt(b['band_width_mm'])} mm, ratio to the six-IMU band "
          f"{fmt(b['width_ratio_to_six_imu'], 3)} (bound 1.5 met: {fmt(b['width_bound_met'])}); energy score "
          f"{fmt(b['energy_score_mm'])} mm against the prior's {fmt(b['energy_score_prior_mm'])} mm, ratio "
          f"{fmt(b['energy_ratio_to_prior'], 3)} (bound 0.8 met: {fmt(b['energy_bound_met'])}); degenerate knots "
          f"{b['degenerate_frames']} of {b['n_frames']:,}.", "",
          f"Conformal factors on calib, {source_text}: " + "; ".join(
              f"{k} c = {fmt(v['c'], 4)} over {v['n_frames']:,} knots, coverage at c {fmt(v['coverage_at_c'], 4)}, "
              f"degenerate {v['n_degenerate']}" for k, v in fits.items()) + ".", "",
          "## Secondary rows (no kill)", "",
          "| set | row | point | 95% interval | persons | windows |", "|---|---|---|---|---|---|"]
    for r in sec:
        nd = 3 if "coverage" in r["row"] else 1
        unit = "" if "coverage" in r["row"] else " mm"
        L.append(f"| {r['set']} | {r['row']} | {fmt(r['point'], nd)}{unit} | {fmt(r['lo'], nd)} to {fmt(r['hi'], nd)} | "
                 f"{r['n_persons']} | {r['n_windows']:,} |")
    if model_rows is not None:
        L += [""] + secondary_markdown(model_rows, a.refine_steps, fmt)[:-1]
    L += ["", "## Per archive", "",
          "| archive | persons | windows | G1a mm | G1a 15 joints mm | G1b coverage | G1b width mm | G1b energy mm | "
          "G1c mm | G1c 22 joints mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pa.iterrows():
        L.append(f"| {r['archive']} | {r['n_persons']} | {r['n_windows']:,} | {r['g1a_mpjpe_mm']:.1f} | "
                 f"{r['g1a_mpjpe15_mm']:.1f} | {r['g1b_coverage']:.3f} | {r['g1b_width_mm']:.1f} | {r['g1b_energy_mm']:.1f} | "
                 f"{r['g1c_pa_mpjpe_mm']:.1f} | {r['g1c_pa_mpjpe22_mm']:.1f} |")
    L += ["", "## Hashes and time", "",
          f"- scoring code SHA-256 ({', '.join(g1.SCORE_CODE)}): {results['hashes']['scoring_code_sha256']}",
          f"- run code SHA-256 ({', '.join(results['hashes']['run_code_files'])}): {code_sha}",
          f"- input code SHA-256 at build: {', '.join(results['hashes']['input_code_sha256'])}",
          f"- time: load {results['timing_s']['load']} s, calib {results['timing_s']['calib']} s, dev "
          f"{results['timing_s']['dev']} s, all {results['timing_s']['all']} s ({len(blocks)} inference blocks, "
          "blocks.csv).", ""]
    md = "\n".join(L) + "\n"
    common.write(results, out, "g1", md)
    common.write_csv(pp, os.path.join(out, "per_person.csv"))
    common.write_csv(pa, os.path.join(out, "per_archive.csv"))
    common.write_csv(table, os.path.join(out, "windows.csv"))
    common.write_csv(blocks, os.path.join(out, "blocks.csv"))
    for name, df in sec_tabs.items():
        common.write_csv(df, os.path.join(out, f"{name}.csv"))
    prov = {"script": "scripts/eval_g1_dev.py", "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "checkpoint": ck["path"], "checkpoint_sha256": ck["sha256"], "prior": common.rel(a.prior),
            "code_sha256": code_sha}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    if save_rec is not None:            # last: a run that failed before this line leaves no file (written once)
        save_conformal_last(I, common, save_to, save_rec, source["saved_sha256"])
        log(f"conformal factors saved to {source['saved_to']}")
    for r in res:
        print(f"{r.line}: {r.point:.4g} [{r.interval[0]:.4g}, {r.interval[1]:.4g}] {r.unit} -> {r.verdict}")
    print(f"written {common.rel(out)}; {wall:.0f} s")


if __name__ == "__main__":
    main()
