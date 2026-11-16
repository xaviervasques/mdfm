"""F2-L1, lever 3 of docs/notes/F2_SCALE_DIAGNOSIS_2026-10-05.md (section 5, rank 3): inference precision. Does the
bf16 autocast of the evaluations, alone, shorten the decoded G3 stride or pass speed? Label of every output: dev
diagnostic, not a G3 read.

Pre-registration: reports/bench/f2_lever1/precision_prediction.txt (the same text in
reports/weekend_2026-10-01/tasks/F2L1.md, section Prediction), its SHA-256 in precision_prediction.sha256 beside it and
in PREDICTION_SHA256 here; check_prediction asserts that the three agree and that the report still holds the text,
before any target or decoded value is read (--decode and the analysis both call it first).

Where precision and the refinement seed are set (mdfm/eval/infer.py, read, not edited):
- infer(..., amp=None) runs the encoder, the refinement and the decoder under torch.autocast(bfloat16) when the model
  is on CUDA (_amp: on = device is cuda when amp is None), in float32 on the CPU; amp=False turns the autocast off on
  any device, amp=True turns it on (also on the CPU). The forward kinematics of the decoded knots (_pose) always run
  in float32 with the autocast off.
- Refinement draws: infer seeds one torch.Generator on the model's device per call with `seed`; refine draws each
  step's kernel points with mdfm.model.mfm.Batch.subsample, torch.rand(n) on that generator, so the draws depend on
  the seed and the point counts only, not on the arithmetic. G3Predictor calls infer once per trial with
  seed_of(seed, trial_id) and does not pass amp; decode_trial here repeats G3Predictor.__call__ (record_windows,
  infer, g3.stitched_body) with amp passed through, so the passes differ in arithmetic only.

--decode (GPU, one queued job; --print-job prints its scripts/queue_job.py command): v1_B by default, on the
SelfPace and HurriedPace records of dev12 (g3.load_targets and g3.imu_records with allow_lockbox False; each record
checked by g3_gain_check.check_record: lockbox False, no EVAL tier, a dev12 person), with the inference of the diag run
(20 refinement steps, 256 points, batch_windows of the diag run, seed 20260929, shape pooled per trial, EMA weights),
three passes (PASSES): bf16 (amp None, the evaluations' setting), fp32 (amp False, with TF32 off for matmul and cuDNN
during the pass, restored after: float32_backends), bf16_rerun (amp None again, the rerun noise floor). Per pass:
g3.body_predictions, match_strides, match_speeds and eval_g3_diag.events_frame with the pass name as row. Writes
<out>/events.csv, trials.csv (per trial and pass: the amp decode_trial received, whether the autocast was on, and the
TF32 flags at the call), decode.json (per pass: the same, as seen by decode_trial) and _provenance_decode.json, then
runs the analysis.

Analysis (default run, CPU; also at the end of --decode): <out>/events.csv, persons guarded first (dev12 only, none in
the lockbox list); walkway strides matched in every pass (key trial_id, foot, t0_s, t1_s) and walkway passes matched
in every pass (trial_id, pass_id); per person the mean and the mean absolute value of fp32 minus bf16 and of
bf16_rerun minus bf16 (D, A, D0, A0), their means over persons with the person bootstrap of
g3_gain_check.line_of (BOOT_N resamples, SEED); S(12) and V(12) of each pass and the paired bf16 minus fp32 difference;
the reading of the pre-registration (precision_reading) per line; beside, the bf16 pass against the diag run's
events.csv (row base, dev12) and the dev12 gains per pass. Writes <out>/results.json, summary.md, per_stride.csv,
per_person.csv (and _provenance.json in the default run).

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_precision.py            (analysis of an existing decode)
  <conda>/envs/mdfm/python.exe scripts/f2_precision.py --print-job
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__" and "--decode" not in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the analysis never sees the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
BENCH = os.path.join(TREE, "reports", "bench", "f2_lever1")
PREDICTION_PATH = os.path.join(BENCH, "precision_prediction.txt")
PREDICTION_SHA_PATH = os.path.join(BENCH, "precision_prediction.sha256")
PREDICTION_SHA256 = "543fe5bd1f53cb8731e3dd7ef713ad56a79b824ea63e15ca22e6a1aa38e526ab"
REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2L1.md")
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2L1"
CHECKPOINT = "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
NAME = "phaseA_root_full_ckpt_B_032044"
PASSES = (("bf16", None, False), ("fp32", False, True), ("bf16_rerun", None, False))   # (row, amp, TF32 off)
REF, TEST, RERUN = "bf16", "fp32", "bf16_rerun"
BOOT_N = 10000
SEED = 20261005
THRESHOLD = {"stride": 1.0, "speed": 0.01}          # cm, m/s: the pre-registered floors of T_S and T_V
NOISE_FACTOR = 2.0
NOISE_NOTE = {"stride": 1.0, "speed": 0.01}         # A0 above this: rerun noise alone exceeds the size looked for
STRIDE_KEY = ["trial_id", "foot", "t0_s", "t1_s"]
SPEED_KEY = ["trial_id", "pass_id"]
JOB = {"name": "f2l1_precision_v1_B", "cls": "1", "priority": "71", "est_hours": "0.2", "timeout_hours": "0.6",
       "memory_gb": "12", "group": "F2L1"}
EXIT_NO_DECODE = 3


def _sibling(name):
    key = f"_f2prec_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def G():
    """scripts/g3_gain_check.py (F1): guards, record check, lines, person bootstrap."""
    return _sibling("g3_gain_check")


def DIAG():
    """scripts/eval_g3_diag.py: events_frame."""
    return _sibling("eval_g3_diag")


def rel(path):
    try:
        r = os.path.relpath(os.path.abspath(path), TREE)
    except ValueError:                                # another drive (tests on a temporary folder)
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def default_out(name=NAME):
    return os.path.join(BENCH, f"precision_{name}")


def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------- pre-registration
def check_prediction(path=None, sha_path=None, report=None, expected=None):
    """AssertionError unless the prediction text hashes to `expected`, the hash file names it, and the report still
    holds the text verbatim (module values at call time)."""
    path = PREDICTION_PATH if path is None else path
    sha_path = PREDICTION_SHA_PATH if sha_path is None else sha_path
    report = REPORT if report is None else report
    expected = PREDICTION_SHA256 if expected is None else expected
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    with open(sha_path, encoding="utf-8") as f:
        words = f.read().split()
    if not (sha == (words[0] if words else "") == expected):
        raise AssertionError(f"prediction hash mismatch: text {sha}, file {words[0] if words else 'empty'}, "
                             f"script {expected}")
    with open(report, encoding="utf-8", newline="") as f:
        if raw.decode("utf-8") not in f.read():
            raise AssertionError(f"{rel(report)} no longer holds the registered prediction text")
    return expected


# ---------------------------------------------------------------- decoding
@contextlib.contextmanager
def float32_backends(torch):
    """TF32 off for CUDA matmul and cuDNN and the float32 matmul precision 'highest' inside the block; the previous
    flags restored after. Yields the flags in force."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield backend_flags(torch)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


def backend_flags(torch):
    return {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def autocast_on(device, amp):
    """Whether infer's autocast is on for this device and amp (the rule of mdfm.eval.infer._amp)."""
    import torch
    return (torch.device(device).type == "cuda") if amp is None else bool(amp)


def decode_trial(loaded, trial_id, record, amp, refine_steps=20, points=256, batch_windows=32, seed=20260929,
                 pool=True):
    """(g3.TrialBody, log dict) of one trial: G3Predictor.__call__ of mdfm/eval/infer.py with amp passed to infer
    (record_windows, infer with groups per trial and seed_of(seed, trial_id), g3.stitched_body). The log records the
    amp this call received, whether infer's autocast is on for it, and the TF32 flags in force at the call, so that
    trials.csv and decode.json show the precision each pass actually ran with."""
    import torch

    from mdfm.eval import infer as I
    from mdfm.gates import g3
    seen = {"amp": repr(amp), "autocast_bf16": autocast_on(next(loaded.model.parameters()).device, amp),
            **backend_flags(torch)}
    I.refuse_cody(trial_id, record.get("source"), record.get("subject_ns"), record.get("person_id"))
    wins, sp, fps = I.record_windows(record, I.WIN_S, I.HOP_S)
    if not wins:
        raise ValueError(f"{trial_id}: record too short for a window")
    st = {}
    bodies = I.infer(loaded.model, wins, refine_steps=refine_steps, points=points,
                     groups=[trial_id] * len(wins) if pool else None, batch_windows=batch_windows,
                     seed=I.seed_of(seed, trial_id), amp=amp, stats=st)
    rate = loaded.model.rate
    starts = [int(round(s / fps * rate)) for s, _ in sp]
    tb = g3.stitched_body(trial_id, [b.joints for b in bodies], [b.contact for b in bodies], None, starts, rate)
    return tb, {"trial_id": trial_id, "person_id": str(record.get("person_id")), "windows": len(wins),
                "refine_first": st.get("refine_first"), "refine_last": st.get("refine_last"),
                "seconds": st.get("seconds"), **seen}


def dev12_persons(dev):
    return sorted(p for p in dev if p not in G().EXCLUDED_FROM_12)


def decode(a, log=print):
    """--decode (module docstring): returns the output folder."""
    check_prediction()                                    # before any target or record is read
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    import pandas as pd
    import torch

    from mdfm.data import sources
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    g = G()
    t_start = time.time()
    dev, train = list(g3.controls("dev")), list(g3.controls("train"))
    persons = dev12_persons(dev)
    locked = g3.lockbox_persons()
    g.guard_persons(persons, dev, locked, "dev12")
    if set(persons) & set(train):
        raise AssertionError("dev and training fold overlap")
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    if set(persons) & loaded.trained_persons():
        raise Refused("dev12 persons among the checkpoint's training persons")
    out = a.out or default_out(loaded.name)
    paths.assert_not_moveatlas(out)
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")
    tg = g3.load_targets(persons)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg.counts['lockbox_rows']} lockbox target rows kept")
    records = []
    for tid, rec in g3.imu_records(persons, tg.trials):
        g.check_record(tid, rec, persons)
        records.append((tid, rec))
    if not records:
        raise ValueError("no dev12 trial to decode")
    events, tlog, passes = [], [], {}
    for name, amp, strict in PASSES:
        t0 = time.time()
        ctx = float32_backends(torch) if strict else contextlib.nullcontext(backend_flags(torch))
        parts = []
        with ctx as flags:
            for tid, rec in records:
                tb, r = decode_trial(loaded, tid, g3.model_input(rec), amp, a.refine_steps, a.points, a.batch_windows,
                                     a.seed, not a.no_pool)
                parts.append(g3.body_predictions([tb], tg))
                tlog.append({"pass": name, **r})
                del tb
        cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
        P = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds"))
        ms, mv = g3.match_strides(P, tg), g3.match_speeds(P, tg)
        events.append(DIAG().events_frame(name, {"strides": ms, "speeds": mv}, tg))
        mine = [t for t in tlog if t["pass"] == name]
        passes[name] = {"amp": amp, "autocast_bf16": autocast_on(device, amp), "backend_flags": flags,
                        "seen_by_decode_trial": {k: sorted({t[k] for t in mine}, key=str) for k in
                                                 ("amp", "autocast_bf16", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
                                                  "float32_matmul_precision")},
                        "seconds": round(time.time() - t0, 1), "matched_strides": int(ms["matched"].sum()),
                        "walkway_strides": int(len(ms)), "matched_passes": int(mv["matched"].sum())}
        log(f"pass {name}: {len(parts)} trials, {passes[name]['seconds']} s, autocast {passes[name]['autocast_bf16']}")
    os.makedirs(out, exist_ok=True)
    common.write_csv(pd.concat(events, ignore_index=True), os.path.join(out, "events.csv"))
    common.write_csv(pd.DataFrame(tlog), os.path.join(out, "trials.csv"))
    mem = {}
    if device.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9, "cap_gb": cap}
    inference = {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S, "hop_s": I.HOP_S,
                 "batch_windows": a.batch_windows, "seed": a.seed,
                 "shape_pooling": "per trial" if not a.no_pool else "none", "weights": a.weights,
                 "device": str(device), "memory": mem}
    doc = {"what": LABEL, "task": TASK, "mode": "decode", "checkpoint": loaded.describe(), "inference": inference,
           "passes": passes, "prediction": {"path": rel(PREDICTION_PATH), "sha256": PREDICTION_SHA256,
                                            "asserted_before_targets": True},
           "sets": {"dev12": persons, "dev12_sha256": sources.person_hash(persons)}, "target_counts": tg.counts,
           "trials": len(records), "timing_s": {"all": round(time.time() - t_start, 1)}}
    common.write_text(os.path.join(out, "decode.json"),
                      json.dumps(G().clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    prov = {"script": "scripts/f2_precision.py", "mode": "decode", "label": LABEL, "checkpoint": rel(loaded.path),
            "checkpoint_sha256": loaded.sha256, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "prediction_sha256": PREDICTION_SHA256,
            "moveatlas_stamp": g.stamp_moveatlas(),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__, I.__file__,
                                               DIAG().__file__, g.__file__]),
            "code_files": ["scripts/f2_precision.py", "mdfm/gates/g3.py", "mdfm/eval/infer.py",
                           "scripts/eval_g3_diag.py", "scripts/g3_gain_check.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance_decode.json"),
                      json.dumps(G().clean(prov), indent=1, ensure_ascii=False) + "\n")
    log(f"decoded {len(records)} trials x {len(PASSES)} passes; written {rel(out)}")
    return out


# ---------------------------------------------------------------- analysis
def read_passes(path, persons, locked):
    """{pass: (strides, speeds)} of the matched rows of an events file of --decode, after its person column was
    guarded (persons only, none in locked). Frames: person_id, task, key columns, x (walkway), y (decoded)."""
    import pandas as pd
    g = G()
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    g.guard_persons(set(who["person_id"].dropna()), persons, locked, os.path.basename(path))
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[ev["matched"].str.strip().str.lower() == "true"]
    out = {}
    for name, _, _ in PASSES:
        e = ev[ev["row"] == name]
        if not len(e):
            raise ValueError(f"{os.path.basename(path)}: no matched row of pass {name}")
        frames = []
        for kind, key in (("stride", STRIDE_KEY), ("speed", SPEED_KEY)):
            k = e[e["kind"] == kind]
            d = pd.DataFrame({"person_id": k["person_id"].to_numpy(), "task": k["task"].to_numpy(),
                              **{c: k[c].to_numpy() for c in key},
                              "x": pd.to_numeric(k["reference"]).to_numpy(np.float64),
                              "y": pd.to_numeric(k["prediction"]).to_numpy(np.float64)})
            if len(d) and not np.isfinite(d[["x", "y"]].to_numpy()).all():
                raise ValueError(f"pass {name}: a matched {kind} row without a finite value")
            if d.duplicated(key).any():
                raise ValueError(f"pass {name}: duplicated {kind} key")
            frames.append(d.reset_index(drop=True))
        out[name] = tuple(frames)
    return out


def common_units(frames, key):
    """One frame of the units matched in every pass: person_id, task, key, x and one decoded column per pass; the
    walkway value asserted equal across passes."""
    names = [n for n, _, _ in PASSES]
    base = frames[names[0]][["person_id", "task"] + key + ["x"]]
    m = base.merge(frames[names[0]][key + ["y"]].rename(columns={"y": names[0]}), on=key)
    for n in names[1:]:
        f = frames[n][key + ["x", "y"]].rename(columns={"x": f"x_{n}", "y": n})
        m = m.merge(f, on=key, how="inner")
        if not np.allclose(m["x"].to_numpy(), m[f"x_{n}"].to_numpy(), rtol=0, atol=1e-9):
            raise AssertionError(f"walkway values differ between passes {names[0]} and {n}")
        m = m.drop(columns=[f"x_{n}"])
    return m.sort_values(key).reset_index(drop=True)


def person_differences(units, a, b):
    """{person: (mean of a - b, mean of |a - b|, n)} over the units of each person."""
    d = units[a].to_numpy(np.float64) - units[b].to_numpy(np.float64)
    t = units.assign(_d=d, _a=np.abs(d)).groupby(units["person_id"].astype(str))
    return {str(p): (float(g["_d"].mean()), float(g["_a"].mean()), int(len(g))) for p, g in t}


def precision_reading(D, A0, floor):
    """The pre-registered reading of one line: T = max(floor, NOISE_FACTOR x A0); PRECISION LEVER when |D| >= T and
    the interval of D excludes 0; NOT A LEVER when the interval lies inside (-T, T); INCONCLUSIVE otherwise."""
    T = max(float(floor), NOISE_FACTOR * float(A0["point"]))
    d, iv = float(D["point"]), D.get("interval")
    if iv is None or not math.isfinite(d):
        return {"reading": "INCONCLUSIVE", "threshold": T, "reason": "no interval (fewer than 2 persons)"}
    lo, hi = float(iv[0]), float(iv[1])
    if abs(d) >= T and (lo > 0 or hi < 0):
        r = "PRECISION LEVER"
    elif -T < lo and hi < T:
        r = "NOT A LEVER"
    else:
        r = "INCONCLUSIVE"
    return {"reading": r, "threshold": T, "D": d, "interval": [lo, hi]}


def consequence(stride, speed):
    """The pre-registered consequence, as text."""
    s, v = stride["reading"], speed["reading"]
    if s == "PRECISION LEVER":
        return ("PRECISION LEVER on the stride line: every F1 and F2 number measured under bf16 is restated from fp32 "
                "decodes before any body lever of the note (ranks 5 to 9) is ranked or run; the F2 lever 1 head is "
                "refitted on fp32 decodes before any proposal under its rule.")
    if s == "NOT A LEVER":
        if v == "PRECISION LEVER":
            return ("NOT A LEVER on the stride line, PRECISION LEVER on the speed line: a speed-only effect; the speed "
                    "numbers are restated from fp32 decodes, the stride numbers stand.")
        return ("NOT A LEVER: the bf16 autocast stays and the F1 and F2 numbers stand; the H1 one-trial CPU difference "
                "is not a bf16 effect (refinement-draw randomness or device arithmetic), and no CPU decode is proposed "
                "as a lever.")
    return ("INCONCLUSIVE on the stride line: no restatement; a second job with more trials (dev14 and train54) is "
            "proposed with the same criterion.")


def diag_check(diag_events, bf16_strides, persons, locked):
    """Beside, no criterion: the person stride MAE of the bf16 pass against the diag run's events.csv (row base),
    dev12: per-person differences and the largest absolute one."""
    if diag_events is None or not os.path.exists(diag_events):
        return {"evaluated": False, "reading": "diag events.csv absent"}
    g = G()
    st, _, _ = g.read_events(diag_events, None, locked)
    st = g.subset(st, persons)
    ref, mine = g.person_abs_error(st), g.person_abs_error(bf16_strides)
    both = sorted(set(ref) & set(mine))
    diff = {p: mine[p] - ref[p] for p in both}
    return {"evaluated": True, "reference": rel(diag_events), "persons_compared": len(both),
            "persons_on_one_side": sorted(set(ref) ^ set(mine)), "difference_cm": diff,
            "max_abs_difference_cm": max((abs(x) for x in diff.values()), default=None),
            "S12_bf16_pass": float(np.mean(list(mine.values()))) if mine else None,
            "S12_diag": float(np.mean(list(ref.values()))) if ref else None}


def analyse(out, persons, locked, diag_events=None, n_boot=None, seed=None):
    """The analysis of the module docstring on <out>/events.csv; returns (results dict, per_stride frame,
    per_person frame). Nothing written."""
    import pandas as pd
    check_prediction()
    g = G()
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    seed = SEED if seed is None else seed
    frames = read_passes(os.path.join(out, "events.csv"), persons, locked)
    res = {"what": LABEL, "task": TASK, "prediction_sha256": PREDICTION_SHA256, "lines": {}, "differences": {},
           "readings": {}, "gains": {}, "counts": {}}
    tables = {}
    for kind, key, i in (("stride", STRIDE_KEY, 0), ("speed", SPEED_KEY, 1)):
        u = common_units({n: frames[n][i] for n in frames}, key)
        tables[kind] = u
        res["counts"][kind] = {"common_units": int(len(u)), "persons": int(u["person_id"].nunique()),
                               **{f"matched_{n}": int(len(frames[n][i])) for n in frames}}
        blk = {}
        for tag, a, b in (("test", TEST, REF), ("rerun", RERUN, REF)):
            pdiff = person_differences(u, a, b)
            blk[f"D_{tag}"] = g.line_of({p: v[0] for p, v in pdiff.items()}, n_boot, seed)
            blk[f"A_{tag}"] = g.line_of({p: v[1] for p, v in pdiff.items()}, n_boot, seed)
            blk[f"n_{tag}"] = {p: v[2] for p, v in pdiff.items()}
        res["differences"][kind] = blk
        rd = precision_reading(blk["D_test"], blk["A_rerun"], THRESHOLD[kind])
        rd["rerun_noise_above_note"] = bool(blk["A_rerun"]["point"] > NOISE_NOTE[kind])
        res["readings"][kind] = rd
        res["gains"][kind] = {n: g.median_ratio(frames[n][i]["x"], frames[n][i]["y"]) for n in frames}
    sets = {"dev12": persons}
    defs = {"S12": ("stride", "dev12"), "V12": ("speed", "dev12")}
    lines = {n: g.lines_for(frames[n][0], frames[n][1], sets, 1.0, 1.0, defs, n_boot, seed) for n in frames}
    res["lines"] = {n: {L: {k: v for k, v in x.items() if k != "per_person"} for L, x in ln.items()}
                    for n, ln in lines.items()}
    paired = {}
    for L in defs:
        a, b = lines[REF][L]["per_person"], lines[TEST][L]["per_person"]
        keep = sorted(set(a) & set(b))
        paired[L] = {k: v for k, v in g.paired_of({p: a[p] for p in keep}, {p: b[p] for p in keep}, n_boot,
                                                  seed).items() if k != "per_person"}
        paired[L]["persons_dropped"] = sorted(set(a) ^ set(b))
    res["paired_ref_minus_test"] = paired
    res["consequence"] = consequence(res["readings"]["stride"], res["readings"]["speed"])
    res["diag_check"] = diag_check(diag_events, frames[REF][0], persons, locked)
    ps = tables["stride"].assign(d_test=tables["stride"][TEST] - tables["stride"][REF],
                                 d_rerun=tables["stride"][RERUN] - tables["stride"][REF])
    rows = []
    for kind, unit in (("stride", "cm"), ("speed", "m/s")):
        blk = res["differences"][kind]
        for p in sorted(blk["D_test"]["per_person"]):
            rows.append({"line": kind, "unit": unit, "person_id": p,
                         "mean_test_minus_ref": blk["D_test"]["per_person"][p],
                         "mean_abs_test_minus_ref": blk["A_test"]["per_person"][p],
                         "mean_rerun_minus_ref": blk["D_rerun"]["per_person"].get(p),
                         "mean_abs_rerun_minus_ref": blk["A_rerun"]["per_person"].get(p),
                         "units": blk["n_test"][p]})
    for kind in ("stride", "speed"):
        for k in ("D_test", "A_test", "D_rerun", "A_rerun"):
            res["differences"][kind][k] = {a: b for a, b in res["differences"][kind][k].items() if a != "per_person"}
        res["differences"][kind].pop("n_test", None)
        res["differences"][kind].pop("n_rerun", None)
    return res, ps, pd.DataFrame(rows)


def summary_md(r):
    def f(x, nd):
        v = fin(x)
        return "n/a" if v is None else f"{v:.{nd}f}"

    def iv(x, nd):
        return "n/a" if not x else f"[{f(x[0], nd)}, {f(x[1], nd)}]"
    L = [f"# F2 lever 3: inference precision, bf16 against fp32 ({LABEL})", "",
         "Written by scripts/f2_precision.py (task F2L1) under the pre-registration "
         f"reports/bench/f2_lever1/precision_prediction.txt (SHA-256 {r['prediction_sha256']}, asserted before any "
         "value was read). Every number below is read from results.json of this folder. Persons: dev12 only; no "
         "lockbox, EVAL, CODY or care_pd record.", "",
         f"## Reading: stride **{r['readings']['stride']['reading']}**, speed **{r['readings']['speed']['reading']}**",
         "", r["consequence"], "",
         "| line | units in all passes | persons | D = fp32 minus bf16 | 95 % | A = mean abs | D0 = rerun minus bf16 | "
         "A0 | threshold T | reading |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, nd in (("stride", 3), ("speed", 4)):
        b, rd, c = r["differences"][kind], r["readings"][kind], r["counts"][kind]
        L.append(f"| {kind} ({'cm' if kind == 'stride' else 'm/s'}) | {c['common_units']} | {c['persons']} | "
                 f"{f(b['D_test']['point'], nd)} | {iv(b['D_test']['interval'], nd)} | {f(b['A_test']['point'], nd)} | "
                 f"{f(b['D_rerun']['point'], nd)} | {f(b['A_rerun']['point'], nd)} | {f(rd['threshold'], nd)} | "
                 f"{rd['reading']} |")
    for kind in ("stride", "speed"):
        if r["readings"][kind]["rerun_noise_above_note"]:
            L.append(f"\nGPU rerun noise alone exceeds the size this check looks for on the {kind} line (A0 above "
                     f"{NOISE_NOTE[kind]}): the noise floor is a finding of its own.")
    L += ["", "## Lines per pass (dev12)", "", "| pass | S(12) cm | 95 % | V(12) m/s | 95 % | stride gain | speed gain |",
          "|---|---|---|---|---|---|---|"]
    for n, ln in r["lines"].items():
        L.append(f"| {n} | {f(ln['S12']['point'], 2)} | {iv(ln['S12']['interval'], 2)} | {f(ln['V12']['point'], 3)} | "
                 f"{iv(ln['V12']['interval'], 3)} | {f(r['gains']['stride'][n], 4)} | {f(r['gains']['speed'][n], 4)} |")
    p = r["paired_ref_minus_test"]
    L += ["", f"Paired bf16 minus fp32: S(12) {f(p['S12']['point'], 3)} cm {iv(p['S12']['interval'], 3)}, V(12) "
              f"{f(p['V12']['point'], 4)} m/s {iv(p['V12']['interval'], 4)}.", "", "## Beside", ""]
    dc = r["diag_check"]
    if dc.get("evaluated"):
        L.append(f"- bf16 pass against {dc['reference']} (row base, dev12): {dc['persons_compared']} persons, largest "
                 f"per-person stride MAE difference {f(dc['max_abs_difference_cm'], 3)} cm; S(12) {f(dc['S12_bf16_pass'], 2)}"
                 f" against {f(dc['S12_diag'], 2)} cm.")
    else:
        L.append(f"- diag comparison: {dc.get('reading')}.")
    L += [f"- Matched units per pass: " + "; ".join(
        f"{k} " + ", ".join(f"{n} {v}" for n, v in c.items() if n.startswith("matched_")) for k, c in r["counts"].items())
          + ".", ""]
    return "\n".join(L) + "\n"


def write_analysis(out, res, per_stride, per_person):
    from mdfm.gates import common
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(out, "results.json"),
                      json.dumps(G().clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(out, "summary.md"), text)
    common.write_csv(per_stride, os.path.join(out, "per_stride.csv"))
    common.write_csv(per_person, os.path.join(out, "per_person.csv"))
    return text


def job_command(tree=None):
    """The scripts/queue_job.py argument list of the precision decode (cls 1, --memory-gb 12, group F2L1, explicit
    --out), with the batch_windows of the v1_B diag run's inference block."""
    g = G()
    with open(os.path.join(g.diag_dir(NAME, tree), "diag.json"), encoding="utf-8") as f:
        bw = int((json.load(f).get("inference") or {})["batch_windows"])
    out = f"reports/bench/f2_lever1/precision_{NAME}"
    return ["scripts/queue_job.py", "--name", JOB["name"], "--cls", JOB["cls"], "--priority", JOB["priority"],
            "--est-hours", JOB["est_hours"], "--timeout-hours", JOB["timeout_hours"], "--group", JOB["group"],
            "--requires", CHECKPOINT, "--outputs", f"{out}/events.csv", f"{out}/results.json",
            "--note", "F2-L1 lever 3: dev12 decodes of v1_B under bf16, fp32 and a bf16 rerun, identical refinement draws",
            "--", "scripts/f2_precision.py", "--decode", "--checkpoint", CHECKPOINT, "--batch-windows", str(bw),
            "--memory-gb", JOB["memory_gb"], "--out", out]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--decode", action="store_true", help="the three dev12 passes (GPU, queued), then the analysis")
    p.add_argument("--print-job", action="store_true", help="print the queue_job.py command of the decode")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--boot", type=int, default=BOOT_N)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_job:
        py = "<conda>/envs/mdfm/python.exe"
        print(" ".join([py] + [f'"{x}"' if " " in x else x for x in job_command()]))
        return 0
    from mdfm.gates import common, g3
    if a.decode:
        out = decode(a)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise SystemExit("the analysis runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
        out = a.out or default_out()
        if not os.path.exists(os.path.join(out, "events.csv")):
            print(f"no decode yet: {rel(os.path.join(out, 'events.csv'))} is absent (queue the job: --print-job)")
            return EXIT_NO_DECODE
    check_prediction()
    locked = g3.lockbox_persons()
    persons = dev12_persons(g3.controls("dev"))
    res, ps, pp = analyse(out, persons, locked, os.path.join(G().diag_dir(NAME), "events.csv"), a.boot)
    res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    text = write_analysis(out, res, ps, pp)
    prov = {"script": "scripts/f2_precision.py", "mode": "analysis", "label": LABEL, "run_at": res["run_at"],
            "moveatlas_stamp": G().stamp_moveatlas(), "prediction_sha256": PREDICTION_SHA256,
            "inputs": {rel(os.path.join(out, "events.csv")): G().sha256_file(os.path.join(out, "events.csv"))},
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "g3_gain_check.py")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(G().clean(prov), indent=1) + "\n")
    print(text)
    print(f"written {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
