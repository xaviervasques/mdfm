"""B4.20 (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 7: "inference mode chosen on dev"; weekend task
T21): the inference mode of the G1 and G3 dev evaluations of one checkpoint, chosen on dev by a criterion that
this script writes before any run.

Modes: weights {ema, model} x refinement steps {0, 5, 20} x shape pooling {on, off}, 12 modes named
<weights>_s<steps:02d>_<pool|nopool>; ema_s20_pool is the default of both harnesses (scripts/eval_g1_dev.py and
scripts/eval_g3_dev.py) and the inference of docs/rules/G1_RULE.md section 4. Checkpoint (--checkpoint): the
reference, cache/runs/phaseA_root_full/ckpt_B_032044.pt (reports/gpu_queue/decisions.md, 2026-10-05 09:00).

Stages, one per call (outputs under --out, default reports/gates/inference_mode/):
- --criterion (CPU, no model, no record): writes criterion.json once (refused when it exists or when any mode
  folder exists) and summary.md with the criterion, the tie band, the attainability check of the band and the
  grid to run. The band and the check are read from reports that exist before the grid (CRITERION_INPUTS).
- --grid (GPU, a queue job): for each mode, scripts/eval_g1_dev.py --limit-takes 2 into <mode>/g1_lt2/ and
  scripts/eval_g3_dev.py --no-secondary (the 14 dev controls) into <mode>/g3_dev/, as child processes with the
  mode's --weights, --refine-steps and --no-pool and the settings of the J1 runs of the reference (G1: --memory-gb
  12, --batch-windows 64, --block-windows 64; G3: --memory-gb 12, --batch-windows 32); a cell whose report exists
  for the same checkpoint SHA-256 and settings is kept. A failed cell is recorded and the grid goes on. Then the
  table. Exit code 0 only when the grid is complete (every mode with both reports, APPLICATION) and at least two
  modes have a finite G1a, 1 otherwise, so the queued --full jobs (after the grid) do not start on a partial
  ranking; running --grid again runs the missing cells only.
- --full RANK (GPU, a queue job after the grid): the mode of rank RANK (1 or 2) under the criterion on the complete
  grid (refused, exit 1, while the grid is incomplete); a full eval_g1_dev.py run (same settings, no --limit-takes)
  into <mode>/g1_full/, unless a report of the same checkpoint and settings is there (the --save-conformal job
  writes ema_s20_pool/g1_full/). Then the table.
- --tabulate (CPU): the table only.
A failed cell stays in results.json ('failed') through the later stages until a report of that cell exists.

The table (summary.md, results.json, table.csv): per mode, G1a, G1b (coverage, band width, energy score) and G1c
of the --limit-takes 2 run; S and V (dev stride MAE in cm, speed MAE in m/s) on the 12 controls (the 14 dev controls
without WHC003 and WHC026) and on the 14, read from per_person.csv as scripts/weekend_branch.py reads them
(g3set_from_folder: the mean over persons of the per-person values); the walkway IC median of the harness (14
controls); the rank under the criterion; then the full G1 lines of the two best modes and the selected mode.
Every number is read from the reports of the cells, cited by path. No lockbox, EVAL or CODY input is opened by
this script or its children (the G1 dev and calib archives and the G3 dev controls only).
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(TREE, "scripts")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FORMAT = "mdfm-inference-mode-criterion/1"
OUT = os.path.join(TREE, "reports", "gates", "inference_mode")
CHECKPOINT = "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
WEIGHTS = ("ema", "model")
STEPS = (0, 5, 20)
POOLS = (True, False)
DEFAULT_MODE = "ema_s20_pool"
LIMIT_TAKES = 2
G1_ARGS = ["--memory-gb", "12", "--batch-windows", "64", "--block-windows", "64"]
G3_ARGS = ["--memory-gb", "12", "--batch-windows", "32"]
SUB12, SUB14 = "no_whc003_whc026", "all14"
# the reports the tie band and its attainability check are read from (all exist before the grid)
CRITERION_INPUTS = {
    "g1_v1_full": "reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/g1.json",
    "g1_s2_full": "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_032044/g1.json",
    "g1_v1_lt2_b4000": "reports/gates/g1/dev_phaseA_root_full_ckpt_B_004000_lt2/g1.json",
    "g1_s2_lt2_b4000": "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_004000_lt2/g1.json",
    "decompose_v1": "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.json",
    "g3_v1_ema": "reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044",
    "g3_v1_model": "reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044_model",
    "g3_s2_ema": "reports/gates/g3/dev_pilot_v1_s2_ckpt_B_032044",
}
BAND_MULTIPLE = 2.0             # the tie band: 2 x the G1a seed difference at the B end (the v2check convention)
CRITERION = [
    "1. Rank the modes by the G1a point (six-IMU MPJPE, mm) of their --limit-takes 2 G1 dev run, lower first.",
    "2. The modes whose G1a lies within the tie band of the best remaining G1a are tied with it. Band: {band:.2f} mm, "
    "{multiple:g} x the G1a seed difference at the B end ({seed:.2f} mm, the absolute difference of the G1a points "
    "of {a} and {b}, full dev runs). Among tied modes the lower G3 dev stride MAE on the 12 controls (S(12), cm; the "
    "14 dev controls without "
    "WHC003 and WHC026, the mean over persons of the per-person stride MAE, as scripts/weekend_branch.py) ranks "
    "first; then the lower G1a, then the mode name. The tied group takes the next ranks and the rule repeats on the "
    "modes left.",
    "3. A mode without a finite G1a is not ranked; a tied mode without S(12) ranks after the tied modes that have one.",
    "4. The two best modes get a full G1 dev run (no --limit-takes). The selected mode is the better of the two under "
    "rules 1 to 3 applied to the full G1a (same band) and S(12).",
    "5. G1b, G1c, the G3 speed lines and the IC line are reported beside the ranking and decide nothing.",
    "6. Every number of the selected mode is post-selection (chosen on these dev windows), optimistic; the choice "
    "is a proposal for the owner (docs/proposals/G1_INFERENCE_MODE_PROPOSAL.md), and no rule changes on it.",
]
# Dated amendments of how the criterion is applied. criterion.json (written once, with the SHA-256 of its text) is
# never edited; an amendment is added here with its date and printed in summary.md and results.json.
APPLICATION = [
    "2026-10-06 (review of task T21, written before any cell of the grid ran; criterion.json unchanged): the "
    "criterion is applied to a complete grid only, every mode of criterion.json with both cell reports (its G1 "
    "--limit-takes 2 report and its G3 dev report, of this checkpoint and mode). Until then the ranking is "
    "provisional: --grid exits 1, so the queued --full jobs (after the grid) do not start, --full refuses to run, "
    "and no mode is selected. A report present with a non-finite G1a or S(12) counts as present (rule 3 applies to "
    "it). A failed cell stays listed in results.json until its report exists; running --grid again runs the "
    "missing cells only. The grid's G1 set (the first 2 takes of each dev archive) is far smaller than the full dev "
    "set of the tie band; summary.md states both sizes beside the grid.",
]


# ---------------------------------------------------------------- modes
def mode_name(weights, steps, pool):
    return f"{weights}_s{int(steps):02d}_{'pool' if pool else 'nopool'}"


def modes():
    """[(name, weights, steps, pool)] in the fixed grid order (the default mode first)."""
    out = [(mode_name(w, s, p), w, s, p) for w in WEIGHTS for s in sorted(STEPS, reverse=True) for p in POOLS]
    return out


def mode_args(weights, steps, pool):
    return ["--weights", weights, "--refine-steps", str(int(steps))] + ([] if pool else ["--no-pool"])


# ---------------------------------------------------------------- small helpers
def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def num(x):
    return float(x) if finite(x) else None


def fmt(x, nd=2):
    return "n/a" if not finite(x) else f"{float(x):.{nd}f}"


def rel(path):
    p = os.path.abspath(path)
    try:
        r = os.path.relpath(p, TREE)
    except ValueError:
        return p.replace("\\", "/")
    return p.replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def tree_path(p):
    return p if os.path.isabs(p) else os.path.join(TREE, p)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_text(path, text):
    if "\u2014" in text or "\u2013" in text:
        raise ValueError(f"{rel(path)}: an em-dash or en-dash in the text")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_sha256(path):
    """The checkpoint's SHA-256 from its sidecar <path>.sha256 (as mdfm.eval.infer.load_checkpoint), else the file."""
    p = tree_path(path)
    side = p + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            return f.read().split()[0]
    return sha256_file(p)


# ---------------------------------------------------------------- reading the reports
def g1_line(doc, prefix):
    for r in doc.get("lines") or []:
        if str(r.get("line", "")).startswith(prefix):
            return r
    return {}


def set_size(doc):
    """{dev_windows, dev_persons, dev_takes, calib_windows} of a eval_g1_dev.py g1.json (sets), None when absent."""
    s = (doc or {}).get("sets") or {}
    dv, cv = s.get("dev") or {}, s.get("calib") or {}
    if not dv:
        return None
    return {"dev_windows": dv.get("windows"), "dev_persons": dv.get("persons"), "dev_takes": dv.get("takes"),
            "calib_windows": cv.get("windows")}


def set_text(s):
    if not s:
        return "set size not in the report"
    return (f"{s['dev_windows']} dev windows of {s['dev_persons']} persons in {s['dev_takes']} takes, calib "
            f"{s['calib_windows']} windows")


def read_g1(folder):
    """The G1 numbers of a eval_g1_dev.py folder, or None: lines, settings, checkpoint and conformal source."""
    js = os.path.join(folder, "g1.json")
    d = read_json(js)
    if not isinstance(d, dict):
        return None
    a, b, c = g1_line(d, "G1a"), g1_line(d, "G1b"), g1_line(d, "G1c")
    bx = b.get("extra") or {}
    inf = d.get("inference") or {}
    prov = read_json(os.path.join(folder, "_provenance.json")) or {}
    return {"json": rel(js), "checkpoint_sha256": (d.get("checkpoint") or {}).get("sha256"),
            "weights": inf.get("weights"), "refine_steps": inf.get("refine_steps"),
            "shape_pooling": inf.get("shape_pooling"), "limit_takes": (d.get("partial_run") or {}).get("limit_takes"),
            "set": set_size(d),
            "g1a_mm": num(a.get("point")), "g1a_interval": a.get("interval"), "g1b_coverage": num(b.get("point")),
            "g1b_width_mm": num(bx.get("band_width_mm")), "g1b_energy_mm": num(bx.get("energy_score_mm")),
            "g1b_width_ratio": num(bx.get("width_ratio_to_six_imu")),
            "g1b_energy_ratio": num(bx.get("energy_ratio_to_prior")), "g1c_mm": num(c.get("point")),
            "conformal_source": (d.get("conformal_source") or {}).get("kind", "fitted in this run"),
            "run_at": prov.get("run_at")}


def read_g3(folder):
    """The G3 numbers of a eval_g3_dev.py folder, or None: S and V on the 12 and the 14 (weekend_branch's reading of
    per_person.csv), the harness's stride, speed, MAPE and walkway IC points, settings and checkpoint."""
    import weekend_branch as WB
    s = WB.g3set_from_folder(folder)
    if s is None:
        return None
    d = read_json(s["json"]) or {}
    m = ((d.get("lines") or {}).get("dev") or {}).get("model") or {}
    inf = d.get("inference") or {}
    pt = lambda k: num((m.get(k) or {}).get("point"))  # noqa: E731
    sub = s["subsets"]
    prov = read_json(os.path.join(folder, "_provenance.json")) or {}
    return {"json": rel(s["json"]), "checkpoint_sha256": (d.get("checkpoint") or {}).get("sha256"),
            "weights": inf.get("weights"), "refine_steps": inf.get("refine_steps"),
            "shape_pooling": inf.get("shape_pooling"),
            "s12_cm": num(sub[SUB12]["S_point"]), "s14_cm": num(sub[SUB14]["S_point"]),
            "v12_m_s": num(sub[SUB12]["V_point"]), "v14_m_s": num(sub[SUB14]["V_point"]),
            "n12": sub[SUB12]["n_persons"], "stride_mae_cm": pt("stride_mae_cm"), "speed_mae_m_s": pt("speed_mae_m_s"),
            "stride_mape_pct": pt("stride_mape_pct"), "ic_walkway_ms": pt("ic_median_ms_walkway"),
            "run_at": prov.get("run_at")}


def matches(cell, sha, weights, steps, pool, kind, limit=None):
    """True when a cell's report is of this checkpoint and mode (and of the asked --limit-takes for G1)."""
    if cell is None or cell.get("checkpoint_sha256") != sha:
        return False
    want_pool = ("per take" if pool else "none") if kind == "g1" else ("per trial" if pool else "none")
    ok = (cell.get("weights") == weights and cell.get("refine_steps") == int(steps)
          and cell.get("shape_pooling") == want_pool)
    if kind == "g1":
        ok = ok and cell.get("limit_takes") == limit
    return ok


# ---------------------------------------------------------------- the criterion
def rank_modes(rows, band):
    """Mode names in the order of the criterion (CRITERION, rules 1 to 3). rows: [{mode, g1a_mm, s12_cm}]."""
    rest = sorted([r for r in rows if finite(r.get("g1a_mm"))], key=lambda r: (float(r["g1a_mm"]), r["mode"]))
    order = []
    while rest:
        best = float(rest[0]["g1a_mm"])
        tied = [r for r in rest if float(r["g1a_mm"]) <= best + band]
        tied.sort(key=lambda r: (not finite(r.get("s12_cm")), float(r["s12_cm"]) if finite(r.get("s12_cm")) else 0.0,
                                 float(r["g1a_mm"]), r["mode"]))
        order += [r["mode"] for r in tied]
        names = {r["mode"] for r in tied}
        rest = [r for r in rest if r["mode"] not in names]
    return order


def band_inputs():
    """{band_mm, seed_difference_mm, a, b, values} of the tie band from the full B-end G1 dev runs of both seeds."""
    a, b = CRITERION_INPUTS["g1_v1_full"], CRITERION_INPUTS["g1_s2_full"]
    va = num(g1_line(read_json(tree_path(a)) or {}, "G1a").get("point"))
    vb = num(g1_line(read_json(tree_path(b)) or {}, "G1a").get("point"))
    if va is None or vb is None:
        raise SystemExit(f"summarize_inference_mode: no G1a in {a} or {b}: the tie band cannot be computed")
    seed = abs(va - vb)
    return {"band_mm": BAND_MULTIPLE * seed, "seed_difference_mm": seed, "a": a, "b": b, "g1a_a_mm": va, "g1a_b_mm": vb,
            "multiple": BAND_MULTIPLE}


def attainability(band):
    """The measured mode differences available before the grid, each set against the band: whether a difference of
    the band's size has been seen between inference settings (rows; a missing input reads 'not available')."""
    rows = []
    dec = read_json(tree_path(CRITERION_INPUTS["decompose_v1"]))
    if isinstance(dec, dict) and isinstance((dec.get("rows") or {}).get("refine0"), dict):
        r = dec["rows"]["refine0"]
        d = num(r.get("delta_mm"))
        rows.append({"what": "G1a, 20 minus 0 refinement steps (EMA, pooled), full dev, paired per window",
                     "source": CRITERION_INPUTS["decompose_v1"], "difference_mm": d,
                     "interval_mm": r.get("delta_interval_mm"),
                     "within_band": None if d is None else bool(abs(d) <= band)})
    for key, label in (("refine60", "G1a, 20 minus 60 refinement steps (EMA, pooled), partial: first 2 takes of each "
                        "dev archive"),):
        r = (dec or {}).get("rows", {}).get(key) if isinstance(dec, dict) else None
        if isinstance(r, dict):
            d = num(r.get("delta_mm"))
            rows.append({"what": label, "source": CRITERION_INPUTS["decompose_v1"], "difference_mm": d,
                         "interval_mm": r.get("delta_interval_mm"),
                         "within_band": None if d is None else bool(abs(d) <= band)})
    a = num(g1_line(read_json(tree_path(CRITERION_INPUTS["g1_v1_lt2_b4000"])) or {}, "G1a").get("point"))
    b = num(g1_line(read_json(tree_path(CRITERION_INPUTS["g1_s2_lt2_b4000"])) or {}, "G1a").get("point"))
    d = None if a is None or b is None else abs(a - b)
    rows.append({"what": "G1a of a --limit-takes 2 run, seed 1 minus seed 2 at ckpt_B_004000 (the noise of the grid's "
                 "partial G1a under a training seed change)",
                 "source": f"{CRITERION_INPUTS['g1_v1_lt2_b4000']}, {CRITERION_INPUTS['g1_s2_lt2_b4000']}",
                 "difference_mm": d, "interval_mm": None, "within_band": None if d is None else bool(d <= band)})
    try:
        e, m, s2 = (read_g3(tree_path(CRITERION_INPUTS[k])) for k in ("g3_v1_ema", "g3_v1_model", "g3_s2_ema"))
    except Exception:  # noqa: BLE001
        e = m = s2 = None
    rows.append({"what": "S(12), EMA minus raw weights at 20 steps, pooled (G3 dev, the reference checkpoint)",
                 "source": f"{CRITERION_INPUTS['g3_v1_ema']}, {CRITERION_INPUTS['g3_v1_model']}",
                 "difference_cm": None if not (e and m) else e["s12_cm"] - m["s12_cm"], "within_band": None})
    rows.append({"what": "S(12), seed 1 minus seed 2 at the B end (G3 dev, EMA, 20 steps, pooled)",
                 "source": f"{CRITERION_INPUTS['g3_v1_ema']}, {CRITERION_INPUTS['g3_s2_ema']}",
                 "difference_cm": None if not (e and s2) else e["s12_cm"] - s2["s12_cm"], "within_band": None})
    seen = [r for r in rows if r.get("within_band") is not None and "steps" in r["what"]]
    if seen and all(r["within_band"] for r in seen):
        reading = ("every measured difference between refinement step counts lies within the band, so the step "
                   "count is expected to tie on G1a and to be decided by S(12); the band itself is reachable "
                   "(a seed change at ckpt_B_004000 moves the partial G1a by the amount in the table)")
    elif seen:
        reading = "at least one measured difference between step counts exceeds the band: G1a can separate modes"
    else:
        reading = "no measured difference between modes is available: the band is not checked against data"
    return {"rows": rows, "reading": reading}


def criterion_text(b):
    return [c.format(band=b["band_mm"], multiple=b["multiple"], seed=b["seed_difference_mm"], a=b["a"], b=b["b"])
            for c in CRITERION]


def stage_criterion(out, ckpt):
    """criterion.json (once) and summary.md, before any run of the grid."""
    path = os.path.join(out, "criterion.json")
    if os.path.exists(path):
        raise SystemExit(f"summarize_inference_mode: {rel(path)} exists; the criterion is written once")
    done = [n for n, *_ in modes() if os.path.isdir(os.path.join(out, n))]
    if done:
        raise SystemExit(f"summarize_inference_mode: mode folders exist ({', '.join(done)}): the criterion must "
                         "precede every run")
    b = band_inputs()
    att = attainability(b["band_mm"])
    text = criterion_text(b)
    doc = {"format": FORMAT, "written": datetime.datetime.now().isoformat(timespec="seconds"),
           "checkpoint": {"path": ckpt, "sha256": checkpoint_sha256(ckpt)},
           "criterion": text, "criterion_sha256": hashlib.sha256("\n".join(text).encode("utf-8")).hexdigest(),
           "band": b, "attainability": att, "inputs": CRITERION_INPUTS,
           "inputs_sha256": {k: (sha256_file(tree_path(v)) if os.path.isfile(tree_path(v)) else None)
                             for k, v in CRITERION_INPUTS.items()},
           "modes": [n for n, *_ in modes()], "default_mode": DEFAULT_MODE, "limit_takes": LIMIT_TAKES,
           "g1_args": G1_ARGS, "g3_args": G3_ARGS, "script_sha256": sha256_file(os.path.abspath(__file__))}
    os.makedirs(out, exist_ok=True)
    write_text(path, json.dumps(doc, indent=1) + "\n")
    tabulate(out, ckpt)
    return doc


# ---------------------------------------------------------------- running cells
def run_child(argv, log):
    """A child process in the tree root (stdout and stderr to this job's console); its exit code."""
    log(f"run: {' '.join(argv)}")
    t0 = time.time()
    rc = subprocess.run([sys.executable] + argv, cwd=TREE).returncode
    log(f"exit {rc} after {time.time() - t0:.0f} s")
    return rc


def stage_grid(out, ckpt, log):
    sha = checkpoint_sha256(ckpt)
    crit = read_json(os.path.join(out, "criterion.json"))
    if not isinstance(crit, dict):
        raise SystemExit("summarize_inference_mode: no criterion.json: run --criterion before the grid")
    if crit.get("checkpoint", {}).get("sha256") != sha:
        raise SystemExit("summarize_inference_mode: criterion.json names another checkpoint")
    failed = []
    for name, w, s, p in modes():
        g1_out, g3_out = os.path.join(out, name, "g1_lt2"), os.path.join(out, name, "g3_dev")
        if not matches(read_g1(g1_out), sha, w, s, p, "g1", LIMIT_TAKES):
            rc = run_child(["scripts/eval_g1_dev.py", "--checkpoint", ckpt, "--limit-takes", str(LIMIT_TAKES), "--out",
                            rel(g1_out)] + G1_ARGS + mode_args(w, s, p), log)
            if rc != 0:
                failed.append(failure(f"{name}/g1_lt2", rc, "grid"))
        if not matches(read_g3(g3_out), sha, w, s, p, "g3"):
            rc = run_child(["scripts/eval_g3_dev.py", "--checkpoint", ckpt, "--no-secondary", "--out", rel(g3_out)]
                           + G3_ARGS + mode_args(w, s, p), log)
            if rc != 0:
                failed.append(failure(f"{name}/g3_dev", rc, "grid"))
    res = tabulate(out, ckpt, failed=failed)
    return grid_exit(res, log)


def grid_exit(res, log):
    """The exit code of --grid (APPLICATION): 0 when the grid is complete and at least two modes are ranked."""
    if not res["grid_complete"]:
        log(f"grid incomplete: {len(res['grid_missing'])} cells without a report ({', '.join(res['grid_missing'])}); "
            "exit 1, so the --full jobs do not start")
        return 1
    if len(res["ranking"]) < 2:
        log(f"grid complete but {len(res['ranking'])} mode(s) with a finite G1a: no two best modes; exit 1")
        return 1
    return 0


def stage_full(out, ckpt, rank, log):
    sha = checkpoint_sha256(ckpt)
    res = tabulate(out, ckpt)
    if not res["grid_complete"]:
        log(f"grid incomplete ({len(res['grid_missing'])} cells without a report): no full run of a rank on a "
            "provisional ranking")
        return 1
    order = res["best_two"]
    if len(order) < rank:
        log(f"no mode of rank {rank}: {len(res['ranking'])} ranked")
        return 1
    name = order[rank - 1]
    w, s, p = next((w, s, p) for n, w, s, p in modes() if n == name)
    folder = os.path.join(out, name, "g1_full")
    if matches(read_g1(folder), sha, w, s, p, "g1", None):
        log(f"rank {rank}: {name}, full G1 report already in {rel(folder)}; kept")
    else:
        rc = run_child(["scripts/eval_g1_dev.py", "--checkpoint", ckpt, "--out", rel(folder)] + G1_ARGS
                       + mode_args(w, s, p), log)
        if rc != 0:
            tabulate(out, ckpt, failed=[failure(f"{name}/g1_full", rc, f"full {rank}")])
            return rc
    tabulate(out, ckpt)
    return 0


# ---------------------------------------------------------------- the table
def failure(cell, rc, stage):
    """A failed cell of results.json 'failed': {cell '<mode>/<g1_lt2|g3_dev|g1_full>', exit, stage, at}."""
    return {"cell": cell, "exit": int(rc), "stage": stage, "at": datetime.datetime.now().isoformat(timespec="seconds")}


def kept_failures(previous, new, present):
    """The failures to record: those of the previous results.json and of this stage whose cell still has no report
    (present: the cells with a report of this checkpoint and mode); one per cell, the later one."""
    keep = {}
    for f in list(previous or []) + list(new or []):
        if isinstance(f, dict) and f.get("cell") and f["cell"] not in present:
            keep[f["cell"]] = f
    return list(keep.values())


def reference_sets():
    """The G1 set sizes of the criterion's reports: the same --limit-takes 2 selection (at ckpt_B_004000) and the
    full dev runs of the tie band, each with its source."""
    out = {}
    for key, name in (("g1_v1_lt2_b4000", "limit_takes_2"), ("g1_v1_full", "full_dev")):
        s = set_size(read_json(tree_path(CRITERION_INPUTS[key])))
        out[name] = None if s is None else dict(s, source=CRITERION_INPUTS[key])
    return out


def tabulate(out, ckpt, failed=()):
    """results.json, table.csv and summary.md from criterion.json and the cells present; returns the results.
    failed: the failures of this stage (failure); those of earlier stages are kept while their cell is missing."""
    crit = read_json(os.path.join(out, "criterion.json"))
    if not isinstance(crit, dict):
        raise SystemExit("summarize_inference_mode: no criterion.json (run --criterion first)")
    sha = crit["checkpoint"]["sha256"]
    band = float(crit["band"]["band_mm"])
    grid, full, present = [], [], set()
    for name, w, s, p in modes():
        g1c = read_g1(os.path.join(out, name, "g1_lt2"))
        g3c = read_g3(os.path.join(out, name, "g3_dev"))
        g1c = g1c if matches(g1c, sha, w, s, p, "g1", LIMIT_TAKES) else None
        g3c = g3c if matches(g3c, sha, w, s, p, "g3") else None
        row = {"mode": name, "weights": w, "refine_steps": s, "pooling": "on" if p else "off", "g1": g1c, "g3": g3c}
        for k in ("g1a_mm", "g1b_coverage", "g1b_width_mm", "g1b_energy_mm", "g1c_mm"):
            row[k] = (g1c or {}).get(k)
        for k in ("s12_cm", "s14_cm", "v12_m_s", "v14_m_s", "ic_walkway_ms", "stride_mape_pct"):
            row[k] = (g3c or {}).get(k)
        grid.append(row)
        present |= {f"{name}/{k}" for k, c in (("g1_lt2", g1c), ("g3_dev", g3c)) if c is not None}
        fc = read_g1(os.path.join(out, name, "g1_full"))
        if matches(fc, sha, w, s, p, "g1", None):
            full.append({"mode": name, "g1": fc, "g1a_mm": fc["g1a_mm"], "s12_cm": row["s12_cm"]})
            present.add(f"{name}/g1_full")
    missing_cells = [f"{r['mode']}/{k}" for r in grid for k in ("g1_lt2", "g3_dev")
                     if f"{r['mode']}/{k}" not in present]
    complete = not missing_cells
    n_both = sum(1 for r in grid if r["g1"] is not None and r["g3"] is not None)
    ranking = rank_modes(grid, band)
    for r in grid:
        r["rank"] = ranking.index(r["mode"]) + 1 if r["mode"] in ranking else None
    best2 = ranking[:2] if complete else []
    full2 = [f for f in full if f["mode"] in best2]
    missing = [m for m in best2 if m not in {f["mode"] for f in full2}]
    sel = rank_modes(full2, band) if full2 else []
    if not complete:
        selected = {"mode": None, "status": f"pending: grid incomplete, {n_both} of {len(grid)} modes with both "
                                            "reports (application of the criterion, 2026-10-06)"}
    elif len(best2) < 2:
        selected = {"mode": None, "status": f"no selection: {len(best2)} mode(s) with a finite G1a on the complete "
                                            "grid (rule 3)"}
    elif missing:
        selected = {"mode": None, "status": f"pending: full G1 of {', '.join(missing)} missing"}
    elif not sel:
        selected = {"mode": None, "status": "no selection: no finite full G1a among the two best modes (rule 3)"}
    else:
        selected = {"mode": sel[0], "status": "selected under the criterion (rule 4), post-selection, optimistic"}
    prev = read_json(os.path.join(out, "results.json"))
    fails = kept_failures((prev or {}).get("failed") if isinstance(prev, dict) else None, failed, present)
    sets = [r["g1"]["set"] for r in grid if r["g1"] is not None and r["g1"].get("set")]
    grid_sets = [dict(t) for t in sorted({tuple(sorted(s.items())) for s in sets}, key=str)]
    written = crit.get("written")
    runs = [c.get("run_at") for r in grid for c in (r["g1"], r["g3"]) if c and c.get("run_at")] + \
           [f["g1"].get("run_at") for f in full if f["g1"].get("run_at")]
    before = None if not runs else all(str(written) < str(t) for t in runs)
    res = {"what": "inference mode on dev (plan task B4.20, task T21), dev numbers, not a gate read",
           "checkpoint": crit["checkpoint"], "criterion": crit["criterion"],
           "criterion_sha256": crit["criterion_sha256"],
           "criterion_written": written, "criterion_before_every_run": before, "band_mm": band,
           "criterion_text_matches_script": criterion_text(crit["band"]) == crit["criterion"],
           "application": APPLICATION,
           "grid": [{k: v for k, v in r.items() if k not in ("g1", "g3")} | {
               "g1_json": (r["g1"] or {}).get("json"), "g3_json": (r["g3"] or {}).get("json")} for r in grid],
           "grid_complete": complete, "grid_missing": missing_cells, "grid_modes_with_both_reports": n_both,
           "grid_g1_sets": grid_sets, "reference_sets": reference_sets(),
           "ranking": ranking, "ranking_status": "final" if complete else "provisional: grid incomplete",
           "best_two": best2,
           "full": [{"mode": f["mode"], **{k: f["g1"].get(k) for k in ("json", "g1a_mm", "g1a_interval", "g1b_coverage",
                                                                         "g1b_width_mm", "g1b_energy_mm", "g1c_mm",
                                                                         "conformal_source")}, "s12_cm": f["s12_cm"],
                     "of_the_best_two": f["mode"] in best2} for f in full],
           "selected": selected, "failed": fails,
           "tabulated": datetime.datetime.now().isoformat(timespec="seconds")}
    write_text(os.path.join(out, "results.json"), json.dumps(res, indent=1) + "\n")
    cols = ["mode", "weights", "refine_steps", "pooling", "g1a_mm", "g1b_coverage", "g1b_width_mm", "g1b_energy_mm",
            "g1c_mm", "s12_cm", "s14_cm", "v12_m_s", "v14_m_s", "ic_walkway_ms", "rank", "g1_json", "g3_json"]
    lines = [",".join(cols)]
    for r in res["grid"]:
        lines.append(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    write_text(os.path.join(out, "table.csv"), "\n".join(lines) + "\n")
    write_text(os.path.join(out, "summary.md"), markdown(res, crit))
    return res


def size_lines(res):
    """The sizes of the grid's G1 set and of the full dev set of the tie band, each read from a cited g1.json."""
    rs = res.get("reference_sets") or {}
    lt, fu = rs.get("limit_takes_2"), rs.get("full_dev")
    if res.get("grid_g1_sets"):
        t = "G1 set of the grid's cells (sets of their g1.json): " + "; ".join(set_text(s) for s in res["grid_g1_sets"])
    elif lt:
        t = (f"G1 set of the grid's cells: no cell yet; the same --limit-takes {LIMIT_TAKES} selection holds "
             f"{set_text(lt)} in {lt['source']}")
    else:
        t = "G1 set of the grid's cells: no cell yet and no reference report"
    if fu:
        t += (f". The tie band comes from full dev runs ({set_text(fu)}, {fu['source']}), so the top-2 cut on the grid "
              "rests on a much smaller set than the band; the two full runs then decide (rule 4)")
    return [t + "."]


def markdown(res, crit):
    b = crit["band"]
    done = res.get("grid_modes_with_both_reports", 0)
    sel = res.get("selected") or {}
    L = ["# Inference mode of the G1 and G3 dev evaluations (plan task B4.20, task T21)", "",
         "Written by scripts/summarize_inference_mode.py; every number below is read from the reports it cites "
         "(criterion.json, results.json and table.csv of this folder, and the cell folders). Development numbers, not "
         "a gate read: no lockbox, EVAL or CODY input is opened.", "",
         f"Checkpoint: {crit['checkpoint']['path']} (SHA-256 {crit['checkpoint']['sha256']}), the reference "
         "(reports/gpu_queue/decisions.md, 2026-10-05 09:00).", "",
         f"Status: criterion written {crit['written']} (SHA-256 {crit['criterion_sha256']}); grid: {done} of "
         f"{len(res['grid'])} modes with both reports ({'complete' if res['grid_complete'] else 'incomplete'}); full "
         f"G1 runs: {len(res['full'])}; selected mode: " + (sel.get("mode") or sel.get("status") or "none yet") + ".",
         "", "## Criterion (stated before any run of the grid)", ""]
    L += crit["criterion"] + [""]
    L += [f"Tie band inputs: G1a {fmt(b['g1a_a_mm'])} mm in {b['a']} and {fmt(b['g1a_b_mm'])} mm in {b['b']}; seed "
          f"difference {fmt(b['seed_difference_mm'])} mm; band {fmt(b['band_mm'])} mm.", "",
          "## Application of the criterion (dated amendments; criterion.json unchanged)", ""]
    L += [f"- {t}" for t in res.get("application") or []] + [""]
    L += ["## Attainability of the tie band (measured before the grid)", "",
          "| measured difference | value | interval | within the band | source |", "|---|---|---|---|---|"]
    for r in crit["attainability"]["rows"]:
        if "difference_mm" in r:
            v = f"{fmt(r['difference_mm'])} mm"
        else:
            v = f"{fmt(r.get('difference_cm'), 3)} cm"
        iv = r.get("interval_mm")
        ivs = "n/a" if not iv or iv[0] is None else f"{fmt(iv[0])} to {fmt(iv[1])} mm"
        wb = {True: "yes", False: "no", None: "n/a (another line)"}[r.get("within_band")]
        L.append(f"| {r['what']} | {v} | {ivs} | {wb} | {r['source']} |")
    L += ["", f"Reading: {crit['attainability']['reading']}.", "",
          f"## Grid: G1 --limit-takes {LIMIT_TAKES}, G3 dev (14 controls)", "",
          f"G1 settings: {' '.join(G1_ARGS)}; G3 settings: {' '.join(G3_ARGS)}; the mode's --weights, --refine-steps "
          "and --no-pool; the other options are the scripts' defaults. S and V: stride and speed MAE on the 12 and the "
          "14 controls; IC: the walkway IC median of the harness (14 controls).", ""] + size_lines(res) + [
          "", "| mode | G1a mm | G1b coverage | G1b width mm | G1b energy mm | G1c mm | S(12) cm | S(14) cm | "
          "V(12) m/s | V(14) m/s | IC ms | rank |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["grid"]:
        L.append(f"| {r['mode']} | {fmt(r['g1a_mm'], 1)} | {fmt(r['g1b_coverage'], 3)} | {fmt(r['g1b_width_mm'], 1)} | "
                 f"{fmt(r['g1b_energy_mm'], 1)} | {fmt(r['g1c_mm'], 1)} | {fmt(r['s12_cm'])} | {fmt(r['s14_cm'])} | "
                 f"{fmt(r['v12_m_s'], 3)} | {fmt(r['v14_m_s'], 3)} | {fmt(r['ic_walkway_ms'], 1)} | "
                 f"{r['rank'] if r['rank'] else 'n/a'} |")
    L += ["", "Reports: " + "; ".join(f"{r['mode']}: {r['g1_json'] or 'no G1 report'}, {r['g3_json'] or 'no G3 report'}"
                                      for r in res["grid"]) + ".", ""]
    if res["grid_complete"] and res["ranking"]:
        L += [f"Ranking under the criterion: {', '.join(res['ranking'])}. The two best modes: "
              f"{', '.join(res['best_two']) or 'none (fewer than two ranked)'}.", ""]
    elif res["ranking"]:
        L += [f"Provisional ranking (grid incomplete: {done} of {len(res['grid'])} modes with both reports; no full "
              f"run and no selection until it is complete): {', '.join(res['ranking'])}.", ""]
    if not res["grid_complete"] and len(res["grid_missing"]) < 2 * len(res["grid"]):
        L += ["Cells without a report of this checkpoint and mode: " + ", ".join(res["grid_missing"]) + ".", ""]
    L += ["## Full G1 runs", ""]
    if res["full"]:
        L += ["Every full G1 report present; only those of the two best modes of a complete grid decide (rule 4).", "",
              "| mode | of the two best | G1a mm | 95% interval | G1b coverage | G1b width mm | G1b energy mm | "
              "G1c mm | S(12) cm | conformal factors | report |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for f in res["full"]:
            iv = f.get("g1a_interval") or [None, None]
            L.append(f"| {f['mode']} | {'yes' if f.get('of_the_best_two') else 'no'} | {fmt(f['g1a_mm'], 1)} | "
                     f"{fmt(iv[0], 1)} to {fmt(iv[1], 1)} | {fmt(f['g1b_coverage'], 3)} | "
                     f"{fmt(f['g1b_width_mm'], 1)} | {fmt(f['g1b_energy_mm'], 1)} | {fmt(f['g1c_mm'], 1)} | "
                     f"{fmt(f['s12_cm'])} | {f['conformal_source']} | {f['json']} |")
        L.append("")
    else:
        L += ["No full G1 run yet.", ""]
    L += ["## Selected mode", ""]
    if sel.get("mode"):
        L += [f"{sel['mode']}: {sel['status']}. A proposal for the owner "
              "(docs/proposals/G1_INFERENCE_MODE_PROPOSAL.md); no rule changes on it.", ""]
    else:
        L += [(sel.get("status") or "None yet") + ".", ""]
    if res["failed"]:
        L += ["Failed cells still without a report (kept across stages): " + "; ".join(
            f"{f['cell']} ({f.get('stage')}, exit {f.get('exit')}, {f.get('at')})" for f in res["failed"]) + ".", ""]
    if res["criterion_before_every_run"] is not None:
        L += [f"Criterion written before every run of a cell (criterion.json 'written' against each report's "
              f"_provenance.json run_at): {'yes' if res['criterion_before_every_run'] else 'NO'}.", ""]
    L += [f"The criterion text of criterion.json equals the one this script states: "
          f"{'yes' if res['criterion_text_matches_script'] else 'NO'}. Tabulated {res['tabulated']}.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--criterion", action="store_true", help="write criterion.json and summary.md before any run")
    g.add_argument("--grid", action="store_true", help="run the 12 modes (G1 --limit-takes 2, G3 dev), then tabulate")
    g.add_argument("--full", type=int, choices=(1, 2), help="the full G1 run of the mode of this rank")
    g.add_argument("--tabulate", action="store_true", help="the table only")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--out", default=OUT)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    out = os.path.abspath(a.out)
    try:
        inside = os.path.normcase(os.path.commonpath([os.path.abspath(TREE), out])) == os.path.normcase(
            os.path.abspath(TREE))
    except ValueError:                  # another drive
        inside = False
    if not inside:
        raise SystemExit(f"summarize_inference_mode: --out {out} lies outside the tree")
    if a.criterion:
        doc = stage_criterion(out, a.checkpoint)
        log(f"criterion written: {rel(os.path.join(out, 'criterion.json'))}, band {doc['band']['band_mm']:.2f} mm")
        return 0
    if a.grid:
        return stage_grid(out, a.checkpoint, log)
    if a.full:
        return stage_full(out, a.checkpoint, a.full, log)
    tabulate(out, a.checkpoint)
    log(f"written {rel(os.path.join(out, 'summary.md'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
