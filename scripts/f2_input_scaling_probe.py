"""F2R (2026-10-05): the input-scaling probe of the F2 lever 2, the gate that docs/notes/F2_SCALE_DIAGNOSIS_2026-10-05.md
(section "Dumps, 2026-10-05", recommendation) asked for before the lever-2 retrain. Does the decoded stride of v1_B on
real input lengthen toward its in-sim gain when the real IMU channels of the lever's sites are given the amplitude
the renders have at equal speed? Label of every output: dev diagnostic, not a G3 read.

Pre-registration: reports/bench/f2_lever2/probe_prediction.txt (prediction, criterion, attainability), its SHA-256 in
probe_prediction.sha256 beside it and in PREDICTION_SHA256 here, with probe_prediction_amendment_1.txt (SHA-256 in
probe_prediction_amendment_1.sha256 and AMEND_SHA256; review of task F2R, hashed before any probe output);
check_prediction asserts that the six agree before any record, target or decoded value is read (--decode and the
analysis both call it first). The text was hashed before this script existed. Amendment 1: the probe decides neither
the verdict of the lever-2 rule nor its route, and each consequence is read beside that verdict (P1, consequence);
the lower_back exit of OTHER SCOPE departs from the note's gate, whose consequence is reported beside the registered
one (P2, consequence_note_gate).

--decode (GPU, one queued job; --print-job prints its scripts/queue_job.py command): v1_B by default, EMA weights, on
the SelfPace and HurriedPace records of train54 and dev12 (g3.load_targets and g3.imu_records with allow_lockbox
False; each record checked by g3_gain_check.check_record), with the inference of the diag run (20 refinement steps,
256 points, batch_windows of the diag run, seed 20260929 per trial through seed_of, shape pooled per trial, bf16
autocast on the GPU). Five passes (ARMS): base, lever (lower_back, thigh_L, thigh_R), lower_back, all13 (the 13 sites
with an H2 row) and base_rerun. decode_trial repeats mdfm.eval.infer.G3Predictor.__call__ (record_windows, infer,
g3.stitched_body) and scales the windows between record_windows and infer (scale_window): at each site of the arm,
every valid sample of the specific force (channels 0 to 2) and of the angular rate (channels 3 to 5) becomes
m + (x - m) / q, m the channel's mean over the site's valid samples in the window, q the site's H2 ratio (A_dyn for
the specific force, G_pc for the angular rate; RATIOS, asserted equal to reports/bench/f2_scale/H2sensor/per_site.csv,
set train54, band lp12, quantity q_all_sites). A pass without scaling hands the windows to infer untouched. Per pass:
g3.body_predictions, match_strides, match_speeds and eval_g3_diag.events_frame with the pass name as row. Writes
<out>/events.csv, trials.csv, decode.json and _provenance_decode.json (never over an existing events.csv), then runs
the analysis.

Analysis (default run, CPU; also at the end of --decode): <out>/events.csv, persons guarded first (train54 and dev12
only, none in the lockbox list). Per set and pass: the stride gain and the speed gain (g3_gain_check.median_ratio
over the matched units). Per set and arm: the rise over base (rise: both gains over the units matched in both passes,
a person-cluster bootstrap with the same draw for both, BOOT_N resamples, SEED), the noise floor D0 (base_rerun over
base), T = max(HALF x (c - c_now), NOISE_FACTOR x |D0|) with c and c_now of the H4 dump, and the reading MOVES or
DOES NOT MOVE (arm_reading). The primary reading is the lever arm on train54; the consequence (consequence) follows
section 4 of the text. Reported: the speed rises beside half the H4 pass-speed distance, the share of the distance to
the in-sim stride gain closed, S(12) and V(12) per pass on dev12 with the paired base minus arm, the base pass against
F1's train54 decode and the diag run's dev12 events. Writes <out>/results.json, summary.md, per_arm.csv and
_provenance.json (a rerun needs --results-out with a new folder: nothing is rewritten).

attainability() computes every number of section 6 of the text from files that existed before its hash
(--attainability prints them as JSON).

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_input_scaling_probe.py     (analysis of the decode)
  <conda>/envs/mdfm/python.exe scripts/f2_input_scaling_probe.py --print-job
  CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/f2_input_scaling_probe.py --attainability
"""
import argparse
import csv
import dataclasses
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
BENCH = os.path.join(TREE, "reports", "bench", "f2_lever2")
PREDICTION_PATH = os.path.join(BENCH, "probe_prediction.txt")
PREDICTION_SHA_PATH = os.path.join(BENCH, "probe_prediction.sha256")
PREDICTION_SHA256 = "740b8f2d2bed9a0bfe06b024e141dbf6a6d2d065c57448d5cd025c8a287622a4"
# amendment 1 (review of task F2R, hashed before any probe output): the consequences beside the lever-2 rule (P1),
# the lower-back exit as a departure from the note's gate (P2), the provenance of section 6 (P3)
AMEND_PATH = os.path.join(BENCH, "probe_prediction_amendment_1.txt")
AMEND_SHA_PATH = os.path.join(BENCH, "probe_prediction_amendment_1.sha256")
AMEND_SHA256 = "7aa8afaa2e13cc2e57b21ca1bca6f9175ee8d81fbdf74bb538ede799b8166892"
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2R"
CHECKPOINT = "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
NAME = "phaseA_root_full_ckpt_B_032044"
PER_SITE = os.path.join(TREE, "reports", "bench", "f2_scale", "H2sensor", "per_site.csv")
SYNTH_FIT = os.path.join(TREE, "reports", "bench", "f2_scale", "H2sensor", "synth_fit.csv")
H4_DUMP = os.path.join(TREE, "reports", "bench", "f2_scale", "H4root", "dumps_2026-10-05", "results.json")
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
F1_DECODE_EVENTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", f"decode_{NAME}", "train_events.csv")
PRECISION_RESULTS = os.path.join(TREE, "reports", "bench", "f2_lever1", f"precision_{NAME}", "results.json")
TRAIN54_SHA256 = "293f98d02adc48b0fa266c2125be40c23201066403fcb110543d8c882c9c77d6"
DEV12_SHA256 = "09f1173e34474ae9053a2e74efa1680c07ca38c664fbd30e7c0f01481355338a"
SITES13 = ("head", "trunk", "lower_back", "wrist_L", "wrist_R", "thigh_L", "thigh_R", "shank_L", "shank_R", "ankle_L",
           "ankle_R", "foot_L", "foot_R")
LEVER_SITES = ("lower_back", "thigh_L", "thigh_R")
ARMS = (("base", ()), ("lever", LEVER_SITES), ("lower_back", ("lower_back",)), ("all13", SITES13), ("base_rerun", ()))
REF, RERUN, PRIMARY_ARM = "base", "base_rerun", "lever"
DESCRIPTIVE_ARMS = ("lower_back", "all13")
SCALED_ARMS = (PRIMARY_ARM,) + DESCRIPTIVE_ARMS
SETS = ("train54", "dev12")
PRIMARY_SET = "train54"
# H2 ratios (G_pc, A_dyn), real over render at equal speed: per_site.csv, train54, lp12, q_all_sites (the text, 2)
RATIOS = {"head": (0.9613, 0.8966), "trunk": (1.0403, 0.7794), "lower_back": (0.7424, 0.8230),
          "wrist_L": (0.9534, 0.8175), "wrist_R": (0.8812, 0.7919), "thigh_L": (0.8362, 1.2146),
          "thigh_R": (0.8842, 1.1930), "shank_L": (0.9765, 1.1930), "shank_R": (1.0057, 1.1909),
          "ankle_L": (1.0487, 0.8880), "ankle_R": (1.0533, 0.9593), "foot_L": (1.0589, 1.0861),
          "foot_R": (1.0309, 1.1384)}
ACC, GYRO = (0, 3), (3, 6)                  # channel ranges of an IMU stream: specific force g, angular rate deg/s
BOOT_N = 10000
SEED = 20261005
LEVEL = 0.95
TOL = 1e-9
HALF = 0.5
NOISE_FACTOR = 2.0
MOVES, NOT_MOVES = "MOVES", "DOES NOT MOVE"
STRIDE_KEY = ["trial_id", "foot", "t0_s", "t1_s"]
SPEED_KEY = ["trial_id", "pass_id"]
INFERENCE = {"refine_steps": 20, "points": 256, "seed": 20260929}
JOB = {"name": "f2r_input_scaling_probe_v1_B", "cls": "1", "priority": "72", "est_hours": "0.6",
       "timeout_hours": "1.5", "memory_gb": "12", "group": "F2R"}
EXIT_OK, EXIT_NO_DECODE, EXIT_EXISTS = 0, 3, 5


def _sibling(name):
    key = f"_f2probe_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def G():
    """scripts/g3_gain_check.py (F1): guards, record check, events reader, median ratio, lines, MoveAtlas stamp."""
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
    return os.path.join(BENCH, f"probe_{name}")


def fin(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- pre-registration and inputs
def _check_hash(path, sha_path, expected, what):
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(sha_path, encoding="utf-8") as f:
        words = f.read().split()
    if not (sha == (words[0] if words else "") == expected):
        raise AssertionError(f"{what} hash mismatch: text {sha}, file {words[0] if words else 'empty'}, "
                             f"script {expected}")
    return expected


def check_amendment():
    """AssertionError unless probe_prediction_amendment_1.txt, its hash file and AMEND_SHA256 agree (module values
    at call time)."""
    return _check_hash(AMEND_PATH, AMEND_SHA_PATH, AMEND_SHA256, "prediction amendment")


def check_prediction(path=None, sha_path=None, expected=None):
    """AssertionError unless the prediction text hashes to `expected` and its hash file names the same SHA-256
    (module values at call time), then the same for amendment 1 (check_amendment). Returns the text's SHA-256."""
    path = PREDICTION_PATH if path is None else path
    sha_path = PREDICTION_SHA_PATH if sha_path is None else sha_path
    expected = PREDICTION_SHA256 if expected is None else expected
    _check_hash(path, sha_path, expected, "prediction")
    check_amendment()
    return expected


def load_ratios(path=None):
    """{site: (G_pc, A_dyn)} of the H2 rows (set train54, band lp12, quantity q_all_sites) for the 13 sites,
    asserted equal to RATIOS (the values the text prints)."""
    got = {}
    with open(PER_SITE if path is None else path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if (r["set"] == "train54" and r["band"] == "lp12" and r["quantity"] == "q_all_sites"
                    and r["feature"] in ("G_pc", "A_dyn")):
                got.setdefault(r["site"], {})[r["feature"]] = float(r["median_over_persons"])
    out = {s: (got.get(s, {}).get("G_pc"), got.get(s, {}).get("A_dyn")) for s in SITES13}
    if out != RATIOS:
        bad = [s for s in SITES13 if out[s] != RATIOS[s]]
        raise AssertionError(f"H2 ratios differ from the registered values at {bad}")
    return out


def arm_scales(sites, ratios=None):
    """{site id of imu_render.SITES: (specific-force factor 1/A_dyn, angular-rate factor 1/G_pc)} of an arm."""
    from mdfm.factory.imu_render import SITES
    ratios = RATIOS if ratios is None else ratios
    return {SITES.index(s): (1.0 / ratios[s][1], 1.0 / ratios[s][0]) for s in sites}


def check_sets(train, dev12, locked):
    """AssertionError unless train54 and dev12 hash to the SHA-256 the text names and are disjoint; Refused for a
    lockbox person."""
    from mdfm.data import sources
    g = G()
    g.guard_persons(train, None, locked, "train54")
    g.guard_persons(dev12, None, locked, "dev12")
    if sources.person_hash(sorted(train)) != TRAIN54_SHA256 or sources.person_hash(sorted(dev12)) != DEV12_SHA256:
        raise AssertionError("train54 or dev12 differs from the sets the text names (SHA-256)")
    if set(train) & set(dev12):
        raise AssertionError("dev and training fold overlap")
    return True


# ---------------------------------------------------------------- scaling
def scale_values(values, valid, chan_mask, site_ids, scales):
    """A new float32 array of an IMU stream's values (T, G, C) with, at each column whose site id is in `scales`
    (site id -> (acc factor, gyro factor)), every valid sample of each unmasked specific-force and angular-rate
    channel replaced by m + f (x - m), m the channel's mean over that column's valid samples (float64 arithmetic).
    A factor of exactly 1, an invalid sample, a masked channel and every other column are left bit for bit."""
    x = np.array(values, dtype=np.float32, copy=True)
    valid = np.asarray(valid, dtype=bool)
    cm = np.asarray(chan_mask, dtype=bool)
    for j, sid in enumerate(site_ids):
        f = scales.get(int(sid))
        if f is None:
            continue
        rows = np.flatnonzero(valid[:, j])
        if not len(rows):
            continue
        for (a, b), fac in ((ACC, f[0]), (GYRO, f[1])):
            ch = [c for c in range(a, b) if cm[j, c]]
            if not ch or float(fac) == 1.0:
                continue
            v = np.asarray(values, dtype=np.float64)[np.ix_(rows, [j], ch)]
            m = v.mean(axis=0, keepdims=True)
            x[np.ix_(rows, [j], ch)] = (m + float(fac) * (v - m)).astype(np.float32)
    return x


def scale_window(window, scales):
    """(window, number of its IMU columns scaled): every IMU stream of the window through scale_values; the window
    itself when `scales` is empty."""
    from mdfm.data import windows as W
    if not scales:
        return window, 0
    streams, n = [], 0
    for s in window.streams:
        if s.modality != "imu":
            streams.append(s)
            continue
        ids = [int(i) for i in (s.meta or {}).get("site_ids") or []]
        n += sum(1 for i in ids if i in scales)
        streams.append(dataclasses.replace(s, values=scale_values(s.values, s.valid, s.chan_mask, ids, scales)))
    return W.validate(dataclasses.replace(window, streams=streams)), n


def decode_trial(loaded, trial_id, record, scales, refine_steps=20, points=256, batch_windows=32, seed=20260929,
                 pool=True):
    """(g3.TrialBody, log dict) of one trial: G3Predictor.__call__ of mdfm/eval/infer.py (record_windows, infer with
    groups per trial and seed_of(seed, trial_id), g3.stitched_body) with the windows scaled (scale_window) between
    record_windows and infer. The log records the IMU columns scaled over the trial's windows."""
    from mdfm.eval import infer as I
    from mdfm.gates import g3
    I.refuse_cody(trial_id, record.get("source"), record.get("subject_ns"), record.get("person_id"))
    wins, sp, fps = I.record_windows(record, I.WIN_S, I.HOP_S)
    if not wins:
        raise ValueError(f"{trial_id}: record too short for a window")
    scaled = [scale_window(w, scales) for w in wins]
    wins = [w for w, _ in scaled]
    st = {}
    bodies = I.infer(loaded.model, wins, refine_steps=refine_steps, points=points,
                     groups=[trial_id] * len(wins) if pool else None, batch_windows=batch_windows,
                     seed=I.seed_of(seed, trial_id), stats=st)
    rate = loaded.model.rate
    starts = [int(round(s / fps * rate)) for s, _ in sp]
    tb = g3.stitched_body(trial_id, [b.joints for b in bodies], [b.contact for b in bodies], None, starts, rate)
    return tb, {"trial_id": trial_id, "person_id": str(record.get("person_id")), "windows": len(wins),
                "columns_scaled": int(sum(n for _, n in scaled)), "sites_in_arm": len(scales),
                "refine_first": st.get("refine_first"), "refine_last": st.get("refine_last"),
                "seconds": st.get("seconds")}


# ---------------------------------------------------------------- decode
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
    ratios = load_ratios()
    t_start = time.time()
    train = sorted(g3.controls("train"))
    dev12 = sorted(p for p in g3.controls("dev") if p not in g.EXCLUDED_FROM_12)
    locked = g3.lockbox_persons()
    check_sets(train, dev12, locked)
    persons = sorted(set(train) | set(dev12))
    out = a.out or default_out()
    if os.path.exists(os.path.join(out, "events.csv")):
        raise FileExistsError(f"{rel(os.path.join(out, 'events.csv'))} exists: a decode is never rewritten")
    paths.assert_not_moveatlas(out)
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    if set(dev12) & loaded.trained_persons():
        raise Refused("dev12 persons among the checkpoint's training persons")
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")
    tg = g3.load_targets(persons)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg.counts['lockbox_rows']} lockbox target rows kept")
    records = []
    for tid, rec in g3.imu_records(persons, tg.trials):
        g.check_record(tid, rec, persons)
        records.append((tid, g3.model_input(rec)))
    if not records:
        raise ValueError("no trial to decode")
    events, tlog, passes = [], [], {}
    for arm, sites in ARMS:
        t0 = time.time()
        scales = arm_scales(sites, ratios)
        parts = []
        for tid, rec in records:
            tb, r = decode_trial(loaded, tid, rec, scales, a.refine_steps, a.points, a.batch_windows, a.seed,
                                 not a.no_pool)
            parts.append(g3.body_predictions([tb], tg))
            tlog.append({"pass": arm, **r})
            del tb
        cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
        P = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds"))
        ms, mv = g3.match_strides(P, tg), g3.match_speeds(P, tg)
        events.append(DIAG().events_frame(arm, {"strides": ms, "speeds": mv}, tg))
        mine = [t for t in tlog if t["pass"] == arm]
        passes[arm] = {"sites": list(sites), "factors": {s: {"specific_force": 1.0 / ratios[s][1],
                                                             "angular_rate": 1.0 / ratios[s][0]} for s in sites},
                       "columns_scaled": int(sum(t["columns_scaled"] for t in mine)),
                       "seconds": round(time.time() - t0, 1), "matched_strides": int(ms["matched"].sum()),
                       "walkway_strides": int(len(ms)), "matched_passes": int(mv["matched"].sum())}
        log(f"pass {arm}: {len(parts)} trials, {passes[arm]['seconds']} s, {passes[arm]['columns_scaled']} columns "
            "scaled")
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
                 "device": str(device), "amp_bf16": device.type == "cuda", "memory": mem}
    doc = {"what": LABEL, "task": TASK, "mode": "decode", "checkpoint": loaded.describe(), "inference": inference,
           "passes": passes, "prediction": {"path": rel(PREDICTION_PATH), "sha256": PREDICTION_SHA256,
                                            "amendment_1": {"path": rel(AMEND_PATH), "sha256": AMEND_SHA256},
                                            "asserted_before_targets": True},
           "sets": {"train54_sha256": sources.person_hash(train), "dev12_sha256": sources.person_hash(dev12),
                    "n_train54": len(train), "n_dev12": len(dev12),
                    "trained_on_among_train54": len(set(train) & loaded.trained_persons())},
           "target_counts": tg.counts, "trials": len(records), "timing_s": {"all": round(time.time() - t_start, 1)}}
    common.write_text(os.path.join(out, "decode.json"),
                      json.dumps(G().clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    prov = {"script": "scripts/f2_input_scaling_probe.py", "mode": "decode", "label": LABEL,
            "checkpoint": rel(loaded.path), "checkpoint_sha256": loaded.sha256,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "prediction_sha256": PREDICTION_SHA256, "amendment_1_sha256": AMEND_SHA256,
            "moveatlas_stamp": g.stamp_moveatlas(),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__, I.__file__, DIAG().__file__,
                                               g.__file__]),
            "code_files": ["scripts/f2_input_scaling_probe.py", "mdfm/gates/g3.py", "mdfm/eval/infer.py",
                           "scripts/eval_g3_diag.py", "scripts/g3_gain_check.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance_decode.json"),
                      json.dumps(G().clean(prov), indent=1, ensure_ascii=False) + "\n")
    log(f"decoded {len(records)} trials x {len(ARMS)} passes; written {rel(out)}")
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
    for name, _ in ARMS:
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


def _interval(values, level=LEVEL):
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return [float(lo), float(hi)]


def rise(base, arm, key, n_boot=None, seed=None):
    """The gain of `arm` minus the gain of `base` (median of decoded over walkway) over the units matched in both
    (key; walkway values asserted equal), with the person-cluster percentile interval: persons drawn with replacement,
    the same draw for both passes, n_boot resamples from default_rng(seed)."""
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    seed = SEED if seed is None else seed
    m = base[["person_id"] + key + ["x", "y"]].merge(arm[key + ["x", "y"]], on=key, suffixes=("_b", "_a"))
    if len(m) and not np.allclose(m["x_b"].to_numpy(), m["x_a"].to_numpy(), rtol=0, atol=1e-9):
        raise AssertionError("walkway values differ between the two passes")
    x = m["x_b"].to_numpy(np.float64)
    ok = np.isfinite(x) & (x > 0) & np.isfinite(m["y_b"].to_numpy()) & np.isfinite(m["y_a"].to_numpy())
    rb, ra = (m["y_b"].to_numpy()[ok] / x[ok]), (m["y_a"].to_numpy()[ok] / x[ok])
    p = m["person_id"].astype(str).to_numpy()[ok]
    out = {"n_units": int(ok.sum()), "n_persons": int(len(set(p))), "point": None, "interval": None,
           "g_base": None, "g_arm": None}
    if not ok.any():
        return out
    out.update(g_base=float(np.median(rb)), g_arm=float(np.median(ra)))
    out["point"] = out["g_arm"] - out["g_base"]
    ids, inv = np.unique(p, return_inverse=True)
    if len(ids) < 2:
        return out
    B = [np.sort(rb[inv == i]) for i in range(len(ids))]
    A = [np.sort(ra[inv == i]) for i in range(len(ids))]
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    d = np.empty(n_boot)
    for k in range(n_boot):
        d[k] = (np.median(np.concatenate([A[i] for i in draws[k]]))
                - np.median(np.concatenate([B[i] for i in draws[k]])))
    out["interval"] = _interval(d)
    return out


def threshold(half_distance, d0):
    """T = max(half_distance, NOISE_FACTOR x |D0 point|) (D0 None: the half distance)."""
    v = fin((d0 or {}).get("point"))
    return max(float(half_distance), NOISE_FACTOR * abs(v)) if v is not None else float(half_distance)


def arm_reading(r, T):
    """MOVES when the rise point is at least T (within TOL) and its lower bound above 0 (beyond TOL); DOES NOT MOVE
    otherwise (also without a point or an interval)."""
    pt, iv = fin((r or {}).get("point")), (r or {}).get("interval")
    ok = pt is not None and iv is not None and pt >= T - TOL and iv[0] > TOL
    return {"reading": MOVES if ok else NOT_MOVES, "point": pt, "interval": iv, "T": T,
            "point_at_least_T": bool(pt is not None and pt >= T - TOL),
            "lower_above_0": bool(iv is not None and iv[0] > TOL)}


BESIDE_RULE = ("The probe decides neither the verdict of the lever-2 rule nor its route: the routes of rule.txt "
               "section 9 stand (probe_prediction_amendment_1.txt P1).")


def consequence(readings):
    """The consequence of section 4 of the text, read beside the lever-2 rule as amendment 1 (P1) says, from the
    train54 readings {arm: arm_reading}."""
    if readings[PRIMARY_ARM]["reading"] == MOVES:
        return {"key": "AMPLITUDE PATH", "arms": [PRIMARY_ARM],
                "text": ("AMPLITUDE PATH: scaling the real channels of the lower back and the thighs to the renders' "
                         "amplitude moves the train54 stride gain by at least T. The amplitude at these sites carries part "
                         "of v1_B's loss. " + BESIDE_RULE + " Under a NO-GO of that rule, the handling failed, not the "
                         "relevance; under a GO, the probe is consistent with the lever's hypothesis and does not name "
                         "the mechanism of the f2l2 end (rule_amendment_1.txt A3).")}
    others = [a for a in DESCRIPTIVE_ARMS if readings[a]["reading"] == MOVES]
    if others:
        why = {"all13": "all13: the note's exception, the lever would widen to every site",
               "lower_back": ("lower_back: the lower back carries part of the loss and the thighs' scaling cancels it "
                              "(an exit that departs from the note's gate, registered before any decode; the owner "
                              "decides whether it stands, amendment P2)")}
        return {"key": "OTHER SCOPE", "arms": others,
                "text": ("OTHER SCOPE: the lever arm does not move the gain, " + "; ".join(why[a] for a in others)
                         + ". The site scope of the lever is questioned; the owner decides. " + BESIDE_RULE)}
    return {"key": "NOT THE AMPLITUDE PATH", "arms": [],
            "text": ("NOT THE AMPLITUDE PATH: no arm moves the train54 stride gain by T. The amplitude at these sites is "
                     "not where v1_B's loss sits. " + BESIDE_RULE + " Under a NO-GO of that rule, the lever drops and the "
                     "note's next lever (root-to-stance coupling with the supervised root as the anchor) follows; under "
                     "a GO, the f2l2 B end is still proposed to the owner, and this reading names its mechanism as "
                     "another than the amplitude path at these sites (rule_amendment_1.txt A3); under PARTIAL or NOT "
                     "EVALUATED, the owner decides whether the lever continues.")}


def consequence_note_gate(readings):
    """The consequence under the note's gate, reported beside the registered one (amendment P2): AMPLITUDE PATH when
    the lever arm moves; OTHER SCOPE (all13) when it does not and the all13 arm does, whatever the lower_back arm
    reads; NOT THE AMPLITUDE PATH otherwise."""
    if readings[PRIMARY_ARM]["reading"] == MOVES:
        return {"key": "AMPLITUDE PATH", "arms": [PRIMARY_ARM]}
    if readings["all13"]["reading"] == MOVES:
        return {"key": "OTHER SCOPE", "arms": ["all13"]}
    return {"key": "NOT THE AMPLITUDE PATH", "arms": []}


def h4_half(h4=None):
    """{c, c_now, half, speed_c, speed_now, speed_half} of the H4 dump (root-specific ceiling of train54)."""
    d = read_json(H4_DUMP if h4 is None else h4)["ends"]["v1_B"]["dump"]["root_specific_ceiling_train54"]
    return {"c": d["stride"], "c_now": d["stride_now"], "half": HALF * (d["stride"] - d["stride_now"]),
            "speed_c": d["speed"], "speed_now": d["speed_now"], "speed_half": HALF * (d["speed"] - d["speed_now"])}


def person_mae_check(reference, strides, persons):
    """Per-person stride MAE of `strides` against a reference frame (both g3_gain_check units): differences, the
    largest, the persons on one side only."""
    g = G()
    ref, mine = g.person_abs_error(g.subset(reference, persons)), g.person_abs_error(g.subset(strides, persons))
    both = sorted(set(ref) & set(mine))
    diff = {p: mine[p] - ref[p] for p in both}
    return {"evaluated": True, "persons_compared": len(both), "persons_on_one_side": sorted(set(ref) ^ set(mine)),
            "max_abs_difference_cm": max((abs(x) for x in diff.values()), default=None),
            "mean_difference_cm": float(np.mean(list(diff.values()))) if diff else None}


def analyse(out, sets, locked, f1_events=None, diag_events=None, h4=None, f1_results=None, n_boot=None, seed=None):
    """The analysis of the module docstring on <out>/events.csv; returns (results dict, per-arm frame). Nothing
    written. sets: {train54, dev12} person lists."""
    import pandas as pd
    check_prediction()
    g = G()
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    seed = SEED if seed is None else seed
    persons = sorted(set(sets["train54"]) | set(sets["dev12"]))
    frames = read_passes(os.path.join(out, "events.csv"), persons, locked)
    hh = h4_half(h4)
    g_sim = fin(((read_json(F1_RESULTS if f1_results is None else f1_results).get("ends") or {}).get("v1_B") or {})
                .get("insim", {}).get("stride_gain"))
    res = {"what": LABEL, "task": TASK, "prediction_sha256": PREDICTION_SHA256, "amendment_1_sha256": AMEND_SHA256,
           "h4": hh, "insim_stride_gain_v1_B": g_sim, "sets": {}, "counts": {}}
    rows = []
    for sname in SETS:
        ps = sets[sname]
        sub = {a: (g.subset(frames[a][0], ps), g.subset(frames[a][1], ps)) for a, _ in ARMS}
        gains = {a: {"stride": fin(g.median_ratio(st["x"], st["y"])), "speed": fin(g.median_ratio(sp["x"], sp["y"])),
                     "strides": int(len(st)), "passes": int(len(sp)), "persons": int(st["person_id"].nunique())}
                 for a, (st, sp) in sub.items()}
        rises = {a: {"stride": rise(sub[REF][0], sub[a][0], STRIDE_KEY, n_boot, seed),
                     "speed": rise(sub[REF][1], sub[a][1], SPEED_KEY, n_boot, seed)} for a in SCALED_ARMS + (RERUN,)}
        d0 = rises[RERUN]["stride"]
        T = threshold(hh["half"], d0)
        readings = {a: arm_reading(rises[a]["stride"], T) for a in SCALED_ARMS}
        for a in SCALED_ARMS:
            r = rises[a]["stride"]
            share = (None if r["point"] is None or g_sim is None or r["g_base"] is None or g_sim == r["g_base"] else
                     r["point"] / (g_sim - r["g_base"]))
            readings[a]["share_of_distance_to_insim"] = share
            readings[a]["speed_rise"] = rises[a]["speed"]["point"]
            readings[a]["speed_rise_at_least_half_speed_distance"] = (
                None if rises[a]["speed"]["point"] is None else bool(rises[a]["speed"]["point"] >= hh["speed_half"] - TOL))
            readings[a]["label"] = "primary" if (sname == PRIMARY_SET and a == PRIMARY_ARM) else "descriptive"
            rows.append({"set": sname, "arm": a, "stride_gain_base": r["g_base"], "stride_gain_arm": r["g_arm"],
                         "rise": r["point"], "rise_lo": (r["interval"] or [None, None])[0],
                         "rise_hi": (r["interval"] or [None, None])[1], "T": T, "reading": readings[a]["reading"],
                         "label": readings[a]["label"], "speed_rise": rises[a]["speed"]["point"],
                         "units": r["n_units"], "persons": r["n_persons"]})
        res["sets"][sname] = {"gains": gains, "rises": rises, "noise_D0": d0, "T": T, "readings": readings}
    res["primary"] = res["sets"][PRIMARY_SET]["readings"][PRIMARY_ARM]
    res["consequence"] = consequence(res["sets"][PRIMARY_SET]["readings"])
    ng = consequence_note_gate(res["sets"][PRIMARY_SET]["readings"])
    res["consequence_note_gate"] = {**ng, "differs": ng["key"] != res["consequence"]["key"],
                                    "label": "the note's gate, reported beside the registered consequence (amendment P2)"}
    # dev12 lines per pass and the paired base minus arm
    defs = {"S12": ("stride", "dev12"), "V12": ("speed", "dev12")}
    lines = {a: g.lines_for(frames[a][0], frames[a][1], {"dev12": sets["dev12"]}, 1.0, 1.0, defs, n_boot, seed)
             for a, _ in ARMS}
    res["dev12_lines"] = {a: {L: {k: v for k, v in x.items() if k != "per_person"} for L, x in ln.items()}
                          for a, ln in lines.items()}
    paired = {}
    for a in SCALED_ARMS:
        paired[a] = {}
        for L in defs:
            b, c = lines[REF][L]["per_person"], lines[a][L]["per_person"]
            keep = sorted(set(b) & set(c))
            if len(keep) < 2:
                paired[a][L] = {"point": None, "interval": None, "n_persons": len(keep)}
                continue
            p = g.paired_of({k: b[k] for k in keep}, {k: c[k] for k in keep}, n_boot, seed)
            paired[a][L] = {k: v for k, v in p.items() if k != "per_person"}
    res["dev12_paired_base_minus_arm"] = paired
    # the base pass against the earlier decodes
    checks = {}
    fe = F1_DECODE_EVENTS if f1_events is None else f1_events
    if fe and os.path.exists(fe):
        st, _, _ = g.read_events(fe, sets["train54"], locked)
        checks["train54_against_F1_decode"] = {**person_mae_check(st, frames[REF][0], sets["train54"]),
                                               "reference": rel(fe)}
    else:
        checks["train54_against_F1_decode"] = {"evaluated": False, "reason": "F1's train54 decode absent"}
    if diag_events and os.path.exists(diag_events):
        st, _, _ = g.read_events(diag_events, None, locked)
        checks["dev12_against_diag"] = {**person_mae_check(st, frames[REF][0], sets["dev12"]),
                                        "reference": rel(diag_events)}
    else:
        checks["dev12_against_diag"] = {"evaluated": False, "reason": "the diag run's events.csv absent"}
    res["checks"] = checks
    for a, _ in ARMS:
        res["counts"][a] = {"matched_strides": int(len(frames[a][0])), "matched_passes": int(len(frames[a][1]))}
    return res, pd.DataFrame(rows)


# ---------------------------------------------------------------- attainability (text, section 6)
def attainability(locked=None, train=None):
    """Every number of section 6 of the text, from files that existed before its hash: the half distance T and its
    share of the distance from F1's train54 gain to the in-sim gain (a); the GPU rerun noise of the precision check
    (b); the rise and interval of a uniform factor on F1's train54 decode (c); the implied speed factors of each arm
    through H2's synthetic exponents, weighted on the scaled pairs only and equally over the 26 pairs (d)."""
    from mdfm.gates import g3
    g = G()
    hh = h4_half()
    f1 = read_json(F1_RESULTS)["ends"]["v1_B"]
    gS, g_sim = f1["gain"]["stride"]["gain"], f1["insim"]["stride_gain"]
    prec = read_json(PRECISION_RESULTS)["gains"]["stride"]
    noise = abs(prec["bf16_rerun"] - prec["bf16"])
    locked = g3.lockbox_persons() if locked is None else locked
    train = sorted(g3.controls("train")) if train is None else train
    st, _, counts = g.read_events(F1_DECODE_EVENTS, train, locked)
    st = st.assign(unit=np.arange(len(st)))
    g0 = g.median_ratio(st["x"], st["y"])
    fac = (g0 + hh["half"]) / g0
    r = rise(st, st.assign(y=st["y"] * fac), ["unit"])
    ratios = load_ratios()
    b = {}
    with open(SYNTH_FIT, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["band"] == "lp12" and row["model"] == "v" and row["feature"] in ("G_pc", "A_dyn"):
                b[(row["site"], row["feature"])] = float(row["b"])
    n_all = 2 * len(SITES13)
    implied = {}
    for arm, sites in ARMS:
        if not sites:
            continue
        logs = []
        for s in sites:
            logs.append(-math.log(ratios[s][0]) / b[(s, "G_pc")])
            logs.append(-math.log(ratios[s][1]) / b[(s, "A_dyn")])
        f_scaled, f_equal = math.exp(sum(logs) / len(logs)), math.exp(sum(logs) / n_all)
        implied[arm] = {"scaled_only": f_scaled, "rise_scaled_only": gS * (f_scaled - 1.0), "equal_weight": f_equal,
                        "rise_equal_weight": gS * (f_equal - 1.0), "pairs": len(logs)}
    return {"T": hh["half"], "c": hh["c"], "c_now": hh["c_now"], "speed_half": hh["speed_half"], "g_S_F1": gS,
            "g_sim": g_sim, "share_of_distance": hh["half"] / (g_sim - gS), "noise": noise, "two_noise": 2 * noise,
            "T_over_two_noise": hh["half"] / (2 * noise), "uniform_factor": fac, "uniform_rise": r["point"],
            "uniform_interval": r["interval"], "uniform_strides": int(counts["stride"]["matched"]),
            "uniform_persons": int(counts["stride"]["persons"]), "uniform_meets": bool(
                r["point"] >= hh["half"] - TOL and r["interval"][0] > TOL),
            "implied": implied, "needed_factor": 1.0 + hh["half"] / gS, "pairs_in_layout": n_all}


# ---------------------------------------------------------------- report
def summary_md(r):
    def f(x, nd):
        v = fin(x)
        return "n/a" if v is None else f"{v:.{nd}f}"

    def iv(x, nd):
        return "n/a" if not x else f"[{f(x[0], nd)}, {f(x[1], nd)}]"
    p = r["primary"]
    L = [f"# F2 input-scaling probe of v1_B: the lever's sites at the renders' amplitude ({LABEL})", "",
         "Written by scripts/f2_input_scaling_probe.py (task F2R) under reports/bench/f2_lever2/probe_prediction.txt "
         f"(SHA-256 {r['prediction_sha256']}) with probe_prediction_amendment_1.txt (SHA-256 "
         f"{r.get('amendment_1_sha256')}), both asserted before any value was read. Every number below is read from "
         "results.json of this folder. Persons: train54 and dev12; no lockbox, EVAL, CODY or care_pd record.", "",
         f"## Primary reading (lever arm, train54): **{p['reading']}**", "",
         f"Rise of the stride gain {f(p['point'], 4)} {iv(p['interval'], 4)} against T {f(p['T'], 4)}.", "",
         r["consequence"]["text"], ""]
    ng = r.get("consequence_note_gate")
    if ng:
        L += [f"Under the note's gate (the lower_back exit absent; amendment P2): {ng['key']}"
              + (f" ({', '.join(ng['arms'])})" if ng["arms"] else "")
              + ("; it differs from the registered consequence, and the owner decides whether the lower_back exit "
                 "stands." if ng["differs"] else "; the same as the registered consequence."), ""]
    L += ["| set | arm | stride gain base | arm | rise | 95 % | T | reading | label | share to in-sim | speed rise |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sname, s in r["sets"].items():
        for a, x in s["readings"].items():
            rr = s["rises"][a]["stride"]
            L.append(f"| {sname} | {a} | {f(rr['g_base'], 4)} | {f(rr['g_arm'], 4)} | {f(x['point'], 4)} | "
                     f"{iv(x['interval'], 4)} | {f(x['T'], 4)} | {x['reading']} | {x['label']} | "
                     f"{f(x['share_of_distance_to_insim'], 3)} | {f(x['speed_rise'], 4)} |")
    L += ["", "Noise floor D0 (base_rerun minus base, stride gain): " + "; ".join(
        f"{k} {f(s['noise_D0']['point'], 4)} {iv(s['noise_D0']['interval'], 4)}" for k, s in r["sets"].items())
        + f". Half distance of the H4 dump {f(r['h4']['half'], 4)} (stride, {f(r['h4']['c_now'], 4)} to "
        f"{f(r['h4']['c'], 4)}), {f(r['h4']['speed_half'], 4)} (pass speed); in-sim stride gain of v1_B "
        f"{f(r['insim_stride_gain_v1_B'], 4)}.", "", "## Gains per pass (all matched units)", "",
          "| set | pass | stride gain | speed gain | strides | passes | persons |", "|---|---|---|---|---|---|---|"]
    for sname, s in r["sets"].items():
        for a, x in s["gains"].items():
            L.append(f"| {sname} | {a} | {f(x['stride'], 4)} | {f(x['speed'], 4)} | {x['strides']} | {x['passes']} | "
                     f"{x['persons']} |")
    L += ["", "## dev12 lines per pass (reported)", "", "| pass | S(12) cm | 95 % | V(12) m/s | 95 % |",
          "|---|---|---|---|---|"]
    for a, ln in r["dev12_lines"].items():
        L.append(f"| {a} | {f(ln['S12']['point'], 2)} | {iv(ln['S12']['interval'], 2)} | {f(ln['V12']['point'], 3)} | "
                 f"{iv(ln['V12']['interval'], 3)} |")
    L += ["", "Paired base minus arm (dev12): " + "; ".join(
        f"{a} S(12) {f(x['S12']['point'], 2)} {iv(x['S12']['interval'], 2)} cm, V(12) {f(x['V12']['point'], 3)} "
        f"{iv(x['V12']['interval'], 3)} m/s" for a, x in r["dev12_paired_base_minus_arm"].items()) + ".", "",
          "## Checks", ""]
    for k, c in r["checks"].items():
        if c.get("evaluated"):
            L.append(f"- base pass {k}: {c['persons_compared']} persons, largest per-person stride MAE difference "
                     f"{f(c['max_abs_difference_cm'], 3)} cm, mean {f(c['mean_difference_cm'], 3)} cm "
                     f"({c['reference']}).")
        else:
            L.append(f"- {k}: {c.get('reason')}.")
    L.append("- Matched units per pass: " + "; ".join(
        f"{a} {c['matched_strides']} strides, {c['matched_passes']} passes" for a, c in r["counts"].items()) + ".")
    return "\n".join(L) + "\n"


def write_analysis(out, res, per_arm):
    from mdfm.gates import common
    text = summary_md(res)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(out, "results.json"),
                      json.dumps(G().clean(res), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(out, "summary.md"), text)
    common.write_csv(per_arm, os.path.join(out, "per_arm.csv"))
    return text


def job_command(tree=None):
    """The scripts/queue_job.py argument list of the probe decode (cls 1, priority 72, --memory-gb 12, group F2R,
    explicit --out), with the batch_windows of the v1_B diag run's inference block."""
    g = G()
    with open(os.path.join(g.diag_dir(NAME, tree), "diag.json"), encoding="utf-8") as f:
        bw = int((json.load(f).get("inference") or {})["batch_windows"])
    out = f"reports/bench/f2_lever2/probe_{NAME}"
    return ["scripts/queue_job.py", "--name", JOB["name"], "--cls", JOB["cls"], "--priority", JOB["priority"],
            "--est-hours", JOB["est_hours"], "--timeout-hours", JOB["timeout_hours"], "--group", JOB["group"],
            "--requires", CHECKPOINT, "--outputs", f"{out}/events.csv", f"{out}/results.json",
            "--note", "F2R: the input-scaling probe of the lever-2 note gate; train54 and dev12 decodes of v1_B with the "
                      "real channels of the lever's sites scaled by the inverse H2 ratios (five passes)",
            "--", "scripts/f2_input_scaling_probe.py", "--decode", "--checkpoint", CHECKPOINT, "--batch-windows",
            str(bw), "--memory-gb", JOB["memory_gb"], "--out", out]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--decode", action="store_true", help="the five passes (GPU, queued), then the analysis")
    p.add_argument("--print-job", action="store_true", help="print the queue_job.py command of the decode")
    p.add_argument("--attainability", action="store_true", help="print the numbers of section 6 of the text")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=INFERENCE["refine_steps"])
    p.add_argument("--points", type=int, default=INFERENCE["points"])
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=INFERENCE["seed"])
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--boot", type=int, default=BOOT_N)
    p.add_argument("--out", default=None, help="the decode folder (default reports/bench/f2_lever2/probe_<name>)")
    p.add_argument("--results-out", default=None, help="write the analysis here instead of the decode folder")
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
        return EXIT_OK
    if a.attainability:
        print(json.dumps(G().clean(attainability()), indent=1, sort_keys=True))
        return EXIT_OK
    from mdfm.gates import common, g3
    if a.decode:
        out = decode(a)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise SystemExit("the analysis runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
        out = a.out or default_out()
        if not os.path.exists(os.path.join(out, "events.csv")):
            print(f"no decode yet: {rel(os.path.join(out, 'events.csv'))} is absent (the queued job writes it)")
            return EXIT_NO_DECODE
    check_prediction()
    res_dir = a.results_out or out
    if os.path.exists(os.path.join(res_dir, "results.json")):
        print(f"{rel(os.path.join(res_dir, 'results.json'))} exists: nothing rewritten; pass --results-out")
        return EXIT_EXISTS
    locked = g3.lockbox_persons()
    train = sorted(g3.controls("train"))
    dev12 = sorted(p for p in g3.controls("dev") if p not in G().EXCLUDED_FROM_12)
    check_sets(train, dev12, locked)
    sets = {"train54": train, "dev12": dev12}
    res, per_arm = analyse(out, sets, locked, diag_events=os.path.join(G().diag_dir(NAME), "events.csv"),
                           n_boot=a.boot)
    res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    res["decode_folder"] = rel(out)
    text = write_analysis(res_dir, res, per_arm)
    prov = {"script": "scripts/f2_input_scaling_probe.py", "mode": "analysis", "label": LABEL, "run_at": res["run_at"],
            "moveatlas_stamp": G().stamp_moveatlas(), "prediction_sha256": PREDICTION_SHA256,
            "inputs": {rel(p): G().sha256_file(p) for p in (os.path.join(out, "events.csv"), PER_SITE, H4_DUMP,
                                                              F1_RESULTS, F1_DECODE_EVENTS) if os.path.exists(p)},
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "g3_gain_check.py")]),
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(res_dir, "_provenance.json"), json.dumps(G().clean(prov), indent=1) + "\n")
    print(text)
    print(f"written {rel(res_dir)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
