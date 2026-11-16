"""F1 (weekend 2026-10-04): gain check of the decoded G3 stride and speed. Is the real-sensor G3 failure a scale error
of the decoded outputs (a near-constant gain) rather than lost information? Every output is labelled 'dev
diagnostic, not a G3 read; the gain is fitted on training-fold controls'.

Rule: reports/bench/g3_gain_check/rule.txt (version 1, written 2026-10-04 before this task read any target value).
Its SHA-256 sits beside it in rule.sha256 and here in RULE_SHA256; check_rule asserts that the three agree before
any target value is read (read_events, read_decode, insim_gain, run_check and the targets of --decode call it first),
so a rule edited after the fact stops the run. The rule records that the dev pattern was seen in a scratch reading
during the planning critique, before it was written.

Default run (CPU: CUDA_VISIBLE_DEVICES=-1 is set before anything loads torch), on the ends of ENDS: v1_B
(cache/runs/phaseA_root_full/ckpt_B_032044.pt), s2_B (cache/runs/pilot_v1_s2/ckpt_B_032044.pt) and v1_A
(cache/runs/phaseA_root_full/ckpt_A_010681.pt):
- decoded dev values: the matched stride and speed rows of row base of reports/gates/g3/diag_<name>/events.csv
  (scripts/eval_g3_diag.py); S(14), S(12), V(14) and V(12) raw, recomputed from them, are asserted equal to the base
  row of diag.json within CONSISTENCY_TOL (consistency);
- decoded training-fold values: reports/bench/g3_gain_check/decode_<name>/train_events.csv of --decode (below),
  checked against the inference settings of the diag run and, per person, against the all68 row of
  reports/gates/g3/dev_<name>/per_person.csv within RERUN_PERSON_TOL_CM (cross_check, reported; not evaluated when
  that report holds another checkpoint file or other weights than EMA);
- per set (dev12, dev14, and train54 when decoded) and unit (stride per stride, stride per pass, speed per pass):
  the Pearson correlation of decoded with walkway, the gain (median of decoded over walkway) and the within-person
  slope, with person bootstrap intervals (describe: BOOT_N resamples, seed SEED);
- one gain per line (stride g_S, speed g_V) fitted on train54 only (fit_gain asserts no dev and no lockbox person);
- S(12), S(14), V(12), V(14) raw and recalibrated (decoded / g), the paired person bootstrap of raw minus
  recalibrated (paired_of);
- the reading (gain_reading): GAIN when, at both B ends, raw minus recalibrated S(12) is at least MARGIN_CM with its
  paired lower bound above 0; not GAIN otherwise; not evaluated when a B end has no training-fold decode;
- the label-free in-sim gain of reports/gates/g3_sim/insim_<name>/ (insim_gain) beside it, and the four lines after
  dividing by it (secondary, they read nothing; skipped, with the reason, when an in-sim gain is not a finite
  positive number, so that this side report never stops the check).
Writes reports/bench/g3_gain_check/results.json, summary.md and _provenance.json (scripts/moveatlas_stamp.py),
asserted newer than the rule files.

--decode (GPU, one queued job per B end; --print-jobs prints the scripts/queue_job.py commands): the inference of
scripts/eval_g3_dev.py reused by import (mdfm.eval.infer.load_checkpoint and G3Predictor with eval_g3_dev's
defaults, one trial at a time as its all68 row runs it) on the SelfPace and HurriedPace records of the 54
training-fold controls, then mdfm.gates.g3.body_predictions, match_strides and match_speeds, written in the columns
of events.csv by scripts/eval_g3_diag.py's events_frame: <out>/train_events.csv, trials.csv (the predictor log),
decode.json and _provenance.json. The rule hash is asserted before the targets are read. Never a dev, lockbox, EVAL
or CODY record.

Run (mdfm environment):
  CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/g3_gain_check.py
  <conda>/envs/mdfm/python.exe scripts/g3_gain_check.py --print-jobs
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass

if __name__ == "__main__" and "--decode" not in sys.argv[1:]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the CPU check never sees the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm import paths  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
OUT = os.path.join(TREE, "reports", "bench", "g3_gain_check")
RULE_PATH = os.path.join(OUT, "rule.txt")
RULE_SHA_PATH = os.path.join(OUT, "rule.sha256")
RULE_SHA256 = "70c1e1cfa6021d192fd0e623e61736b464aa5b6222e72c6f098c49c45ba263b5"
LABEL = "dev diagnostic, not a G3 read; the gain is fitted on training-fold controls"
TASK = "F1"
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")
N_DEV, N_TRAIN = 14, 54
BOOT_N = 10000
SEED = 20261001
LEVEL = 0.95
MARGIN_CM = 1.95                    # the E1 margin, docs/proposals/JEPA_EXPLORATION_DRAFT.md section 3
CONSISTENCY_TOL = 1e-6
RERUN_PERSON_TOL_CM = 1.0           # the rerun tolerance of scripts/eval_g3_diag.py (person stride MAE)
INFER_KEYS = ("refine_steps", "points_per_stream", "batch_windows", "seed", "shape_pooling", "weights")
UNITS = ("stride_per_stride", "stride_per_pass", "speed_per_pass")
DEV_SETS = ("dev12", "dev14")
LINE_DEF = {"S12": ("stride", "dev12"), "S14": ("stride", "dev14"), "V12": ("speed", "dev12"),
            "V14": ("speed", "dev14")}
TRAIN_LINE_DEF = {"S54": ("stride", "train54"), "V54": ("speed", "train54")}
DIAG_SUBSET = {"dev14": "all14", "dev12": "no_whc003_whc026"}
DIAG_KEY = {"stride": "stride_mae_cm", "speed": "speed_mae_ms"}
UNIT_OF = {"stride": "cm", "speed": "m/s"}
JOB_PRIORITY0 = 60                  # behind the J0 jobs (priorities 50 to 52) of the queue
JOB_EST_HOURS = 0.25
JOB_MEMORY_GB = "12"


@dataclass(frozen=True)
class End:
    """A checkpoint read by the check: key, report name, tree-relative path, whether the GAIN reading reads it, and
    the batch_windows of its diag run (the decode job uses the same)."""
    key: str
    name: str
    checkpoint: str
    gain_end: bool
    batch_windows: int


ENDS = (End("v1_B", "phaseA_root_full_ckpt_B_032044", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", True, 32),
        End("s2_B", "pilot_v1_s2_ckpt_B_032044", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", True, 16),
        End("v1_A", "phaseA_root_full_ckpt_A_010681", "cache/runs/phaseA_root_full/ckpt_A_010681.pt", False, 32))
GAIN_ENDS = tuple(e.key for e in ENDS if e.gain_end)


# ---------------------------------------------------------------- folders
def diag_dir(name, tree=None):
    return os.path.join(tree or TREE, "reports", "gates", "g3", f"diag_{name}")


def dev_report(name, tree=None):
    """The G3 dev json of a checkpoint (eval_g3_dev.py: phase B g3/dev_<name>/g3_dev.json, phase A
    g3_sim/dev_<name>/g3_sim.json), the first that exists, or None."""
    for folder, stem in (("g3", "g3_dev"), ("g3_sim", "g3_sim")):
        p = os.path.join(tree or TREE, "reports", "gates", folder, f"dev_{name}", f"{stem}.json")
        if os.path.exists(p):
            return p
    return None


def insim_dir(name, tree=None):
    return os.path.join(tree or TREE, "reports", "gates", "g3_sim", f"insim_{name}")


def decode_dir(name, out=None):
    return os.path.join(out or OUT, f"decode_{name}")


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


# ---------------------------------------------------------------- sibling scripts
def _sibling(name):
    """The module of scripts/<name>.py of the tree (loaded once, under a private name)."""
    key = f"_f1_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def EG():
    return _sibling("eval_g3_dev")


def DIAG():
    return _sibling("eval_g3_diag")


def stamp_moveatlas():
    """The read-only stamp of scripts/moveatlas_stamp.py."""
    return _sibling("moveatlas_stamp").stamp()


# ---------------------------------------------------------------- rule and guards
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_rule(rule_path=None, sha_path=None, expected=None):
    """The SHA-256 of the rule file, asserted equal to the first word of the hash file and to the script constant
    (AssertionError otherwise, raised explicitly so that python -O keeps it). Paths default to the module constants
    at call time."""
    rule_path = RULE_PATH if rule_path is None else rule_path
    sha_path = RULE_SHA_PATH if sha_path is None else sha_path
    expected = RULE_SHA256 if expected is None else expected
    sha = sha256_file(rule_path)
    with open(sha_path, encoding="utf-8") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if not (sha == stored == expected):
        raise AssertionError(f"rule hash mismatch: {os.path.basename(rule_path)} {sha}, "
                             f"{os.path.basename(sha_path)} {stored or 'empty'}, script {expected}")
    return sha


def guard_persons(persons, allowed, locked, what="persons"):
    """The set of person ids; Refused when one is a lockbox person, AssertionError when one lies outside `allowed`
    (None: no such check)."""
    ps = {str(p) for p in persons}
    hit = sorted(ps & {str(p) for p in locked})
    if hit:
        raise Refused(f"{what}: {len(hit)} lockbox persons requested")
    if allowed is not None:
        out = sorted(ps - {str(p) for p in allowed})
        if out:
            raise AssertionError(f"{what}: {len(out)} persons outside the expected set ({', '.join(out[:3])})")
    return ps


def check_sets(sets, locked, n_dev=None, n_train=None, excluded=None):
    """AssertionError unless dev14 holds n_dev persons, dev12 is dev14 without the excluded persons, train54 holds
    n_train persons and dev and train are disjoint; Refused when a set holds a lockbox person. Defaults: the module
    constants N_DEV, N_TRAIN and EXCLUDED_FROM_12 at call time."""
    n_dev = N_DEV if n_dev is None else n_dev
    n_train = N_TRAIN if n_train is None else n_train
    excluded = EXCLUDED_FROM_12 if excluded is None else tuple(excluded)
    dev, dev12, train = set(sets["dev14"]), set(sets["dev12"]), set(sets["train54"])
    for name in ("dev14", "dev12", "train54"):
        guard_persons(sets[name], None, locked, name)
    if len(dev) != n_dev or len(train) != n_train:
        raise AssertionError(f"sets of {len(dev)} dev and {len(train)} training-fold persons, expected {n_dev} and "
                             f"{n_train}")
    if dev & train:
        raise AssertionError(f"dev and training fold overlap: {len(dev & train)} persons")
    if not set(excluded) <= dev or dev12 != dev - set(excluded):
        raise AssertionError("dev12 is not dev14 without " + ", ".join(excluded))
    return True


def check_record(tid, rec, persons):
    """AssertionError unless the record carries lockbox False, no EVAL tier, and a person of `persons`."""
    lb = rec.get("lockbox")
    if lb is None or (lb is not False and str(lb).strip().lower() not in ("false", "0")):
        raise AssertionError(f"{tid}: record with lockbox {lb!r}")
    if str(rec.get("tier", "")) in ("EVAL", "EXCLUDED_V1"):
        raise AssertionError(f"{tid}: record of tier {rec.get('tier')}")
    if str(rec.get("person_id")) not in {str(p) for p in persons}:
        raise AssertionError(f"{tid}: person {rec.get('person_id')!r} outside the requested set")


# ---------------------------------------------------------------- events
def read_events(path, allowed, locked, row="base"):
    """(strides, speeds, counts) of an events.csv in the columns of scripts/eval_g3_diag.py, read after check_rule:
    the persons of the file are read first and guarded (guard_persons); then the matched rows of `row`, kind stride
    (cm) and speed (m/s), as frames person_id, trial_id, task, foot, pass_id, x (walkway), y (decoded)."""
    import pandas as pd
    check_rule()
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    guard_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(path))
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[ev["row"] == row]
    matched = ev["matched"].str.strip().str.lower() == "true"
    out, counts = {}, {}
    for kind in ("stride", "speed"):
        allk = ev[ev["kind"] == kind]
        if (allk["unit"] != UNIT_OF[kind]).any():
            raise ValueError(f"{os.path.basename(path)}: {kind} rows in another unit than {UNIT_OF[kind]}")
        k = allk[matched.loc[allk.index].to_numpy()]
        d = pd.DataFrame({"person_id": k["person_id"].to_numpy(), "trial_id": k["trial_id"].to_numpy(),
                          "task": k["task"].to_numpy(), "foot": k["foot"].to_numpy(), "pass_id": k["pass_id"].to_numpy(),
                          "x": pd.to_numeric(k["reference"]).to_numpy(np.float64),
                          "y": pd.to_numeric(k["prediction"]).to_numpy(np.float64)})
        if len(d) and not np.isfinite(d[["x", "y"]].to_numpy()).all():
            raise ValueError(f"{os.path.basename(path)}: a matched {kind} row without a finite value")
        out[kind] = d
        counts[kind] = {"events": int(len(allk)), "matched": int(len(d)), "persons": int(d["person_id"].nunique())}
    return out["stride"], out["speed"], counts


def pass_units(strides):
    """Per walkway pass (person_id, trial_id, pass_id) with a matched stride: x and y the means over its matched
    strides, n the number of them."""
    import pandas as pd
    if not len(strides):
        return pd.DataFrame(columns=["person_id", "trial_id", "pass_id", "x", "y", "n"])
    g = strides.groupby(["person_id", "trial_id", "pass_id"], sort=True)
    return g.agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size")).reset_index()


def units_of(strides, speeds):
    """{unit name: frame with person_id, x, y} of the three units of the rule (section 4)."""
    return {"stride_per_stride": strides, "stride_per_pass": pass_units(strides), "speed_per_pass": speeds}


def subset(df, persons):
    return df[df["person_id"].isin({str(p) for p in persons})].reset_index(drop=True)


# ---------------------------------------------------------------- statistics
def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2:
        return float("nan")
    dx, dy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((dx * dx).sum() * (dy * dy).sum()))
    return float((dx * dy).sum() / den) if den > 0 else float("nan")


def median_ratio(x, y):
    """Median of y / x over the units with a finite positive x and a finite y (NaN without one)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return float(np.median(y[ok] / x[ok])) if ok.any() else float("nan")


def within_slope(x, y, persons):
    """Least-squares slope of y on x after removing each person's means: sum over persons of sum (x - mean x)
    (y - mean y), over the same sum of (x - mean x)^2 (NaN when that is 0)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    _, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    n = np.bincount(inv)
    if not len(n):
        return float("nan")
    mx, my = np.bincount(inv, x) / n, np.bincount(inv, y) / n
    dx, dy = x - mx[inv], y - my[inv]
    sxx = float((dx * dx).sum())
    return float((dx * dy).sum() / sxx) if sxx > 0 else float("nan")


def _interval(values, level=None):
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    a = (1.0 - (LEVEL if level is None else level)) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return [float(lo), float(hi)]


def describe(units, n_boot=None, seed=None):
    """n units, n persons, and the Pearson correlation of y with x, the gain (median_ratio) and the within-person
    slope over the pooled units, each with a person-clustered percentile bootstrap interval (persons drawn with
    replacement n_boot times with seed; a statistic undefined in a resample leaves that resample out, counted)."""
    n_boot = BOOT_N if n_boot is None else int(n_boot)
    seed = SEED if seed is None else seed
    x = units["x"].to_numpy(np.float64)
    y = units["y"].to_numpy(np.float64)
    p = units["person_id"].astype(str).to_numpy()
    out = {"n_units": int(len(x)), "n_persons": int(len(set(p)))}
    point = {"pearson_r": pearson(x, y), "gain": median_ratio(x, y), "within_slope": within_slope(x, y, p)}
    ids, inv = np.unique(p, return_inverse=True)
    P = len(ids)
    if P < 2:
        return {**out, **{k: {"point": v, "interval": None, "resamples_used": 0} for k, v in point.items()}}
    x0, y0 = x - x.mean(), y - y.mean()                      # centred once: no cancellation in the sums
    n = np.bincount(inv, minlength=P).astype(np.float64)
    S = np.stack([n, np.bincount(inv, x0, P), np.bincount(inv, y0, P), np.bincount(inv, x0 * x0, P),
                  np.bincount(inv, y0 * y0, P), np.bincount(inv, x0 * y0, P)], axis=1)
    mx, my = S[:, 1] / n, S[:, 2] / n
    dx, dy = x0 - mx[inv], y0 - my[inv]
    W = np.stack([np.bincount(inv, dx * dx, P), np.bincount(inv, dx * dy, P)], axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, P, size=(n_boot, P))
    C = np.zeros((n_boot, P))
    np.add.at(C, (np.repeat(np.arange(n_boot), P), draws.ravel()), 1.0)
    T = C @ S
    N, SX, SY, SXX, SYY, SXY = T.T
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt((N * SXX - SX * SX) * (N * SYY - SY * SY))
        r_b = np.where(den > 0, (N * SXY - SX * SY) / den, np.nan)
        Wb = C @ W
        slope_b = np.where(Wb[:, 0] > 0, Wb[:, 1] / Wb[:, 0], np.nan)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ratio = [np.sort((y[ok] / x[ok])[inv[ok] == i]) for i in range(P)]
    gain_b = np.empty(n_boot)
    for b in range(n_boot):
        r = np.concatenate([ratio[i] for i in draws[b]])
        gain_b[b] = np.median(r) if len(r) else np.nan
    for key, boot in (("pearson_r", r_b), ("gain", gain_b), ("within_slope", slope_b)):
        out[key] = {"point": point[key], "interval": _interval(boot), "resamples_used": int(np.isfinite(boot).sum())}
    return out


# ---------------------------------------------------------------- the gain and the lines
def fit_gain(units, train, dev, locked):
    """The rule's gain of one line: the median of y / x over the pooled units. Fitted on training-fold persons only:
    Refused for a lockbox person, AssertionError for a dev person or a person outside `train`. Returns {gain,
    n_units, n_persons}."""
    persons = {str(p) for p in units["person_id"]}
    guard_persons(persons, None, locked, "gain fit")
    on_dev = sorted(persons & {str(p) for p in dev})
    if on_dev:
        raise AssertionError(f"the gain is never fitted on dev persons: {len(on_dev)} in the fit")
    guard_persons(persons, train, locked, "gain fit")
    x, y = units["x"].to_numpy(np.float64), units["y"].to_numpy(np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not ok.any():
        raise ValueError("no unit to fit a gain on")
    g = float(np.median(y[ok] / x[ok]))
    if not usable_gain(g):
        raise ValueError(f"fitted gain {g} is not a finite positive number: nothing can be divided by it")
    return {"gain": g, "n_units": int(ok.sum()),
            "n_persons": int(len(set(units["person_id"].astype(str).to_numpy()[ok])))}


def usable_gain(g):
    """True for a finite gain above 0 (one the decoded values can be divided by)."""
    try:
        v = float(g)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def person_abs_error(units, gain=1.0):
    """{person: mean |y / gain - x|} over the units of each person (sorted ids)."""
    if not len(units):
        return {}
    e = np.abs(units["y"].to_numpy(np.float64) / float(gain) - units["x"].to_numpy(np.float64))
    s = (units.assign(_e=e).groupby(units["person_id"].astype(str))["_e"].mean())
    return {str(k): float(v) for k, v in s.sort_index().items()}


def line_of(per_person, n_boot=None, seed=None):
    """{point: mean over persons, interval: mdfm.gates.common.cluster_bootstrap of it (None under 2 persons),
    n_persons, per_person}."""
    from mdfm.gates import common
    ids = sorted(per_person)
    v = np.array([per_person[i] for i in ids], np.float64)
    if not len(ids):
        return {"point": float("nan"), "interval": None, "n_persons": 0, "per_person": {}}
    iv = (list(common.cluster_bootstrap(v, np.array(ids), n=BOOT_N if n_boot is None else int(n_boot),
                                        seed=SEED if seed is None else seed)) if len(ids) >= 2 else None)
    return {"point": float(v.mean()), "interval": iv, "n_persons": int(len(ids)),
            "per_person": {i: float(x) for i, x in zip(ids, v)}}


def paired_of(raw_pp, rec_pp, n_boot=None, seed=None):
    """Raw minus recalibrated, paired by person: per person the difference, its mean over persons and the person
    bootstrap interval of that mean (line_of), with lower_above_0."""
    if set(raw_pp) != set(rec_pp):
        raise AssertionError("raw and recalibrated lines on different persons")
    d = line_of({p: raw_pp[p] - rec_pp[p] for p in raw_pp}, n_boot, seed)
    d["lower_above_0"] = bool(d["interval"] is not None and d["interval"][0] > 0)
    return d


def lines_for(strides, speeds, sets, g_s=1.0, g_v=1.0, defs=None, n_boot=None, seed=None):
    """{line: line_of} of the lines of `defs` (default LINE_DEF): stride lines on the per-stride units divided by
    g_s, speed lines on the passes divided by g_v, each on its person set."""
    out = {}
    for line, (kind, set_name) in (LINE_DEF if defs is None else defs).items():
        u = subset(strides if kind == "stride" else speeds, sets[set_name])
        out[line] = line_of(person_abs_error(u, g_s if kind == "stride" else g_v), n_boot, seed)
        out[line]["unit"] = UNIT_OF[kind]
        out[line]["set"] = set_name
    return out


def recalibration(strides, speeds, sets, g_s, g_v, defs=None, n_boot=None, seed=None):
    """{raw, recalibrated, paired} line blocks for the gains g_s and g_v (lines_for, paired_of per line)."""
    raw = lines_for(strides, speeds, sets, 1.0, 1.0, defs, n_boot, seed)
    rec = lines_for(strides, speeds, sets, g_s, g_v, defs, n_boot, seed)
    paired = {k: {**paired_of(raw[k]["per_person"], rec[k]["per_person"], n_boot, seed), "unit": raw[k]["unit"]}
              for k in raw}
    return {"gains": {"stride": g_s, "speed": g_v}, "raw": raw, "recalibrated": rec, "paired": paired}


def gain_reading(ends, gain_ends=None, margin=None):
    """The reading of the rule (section 7) on {end key: analyse_end result}: GAIN when every gain end has a paired S12
    whose point is at least `margin` and whose lower bound is above 0; 'not GAIN' otherwise; 'not evaluated' when a
    gain end has no training-fold recalibration (named in missing)."""
    gain_ends = GAIN_ENDS if gain_ends is None else tuple(gain_ends)
    margin = MARGIN_CM if margin is None else float(margin)
    rows, missing = {}, []
    for k in gain_ends:
        r = ends.get(k) or {}
        rc = r.get("recalibration")
        if not rc:
            missing.append(k)
            continue
        pr = rc["paired"]["S12"]
        d = pr["point"]
        lo, hi = (pr["interval"] or [float("nan"), float("nan")])
        meets = bool(np.isfinite(d) and d >= margin)
        above = bool(np.isfinite(lo) and lo > 0)
        rows[k] = {"raw_S12_cm": rc["raw"]["S12"]["point"], "recalibrated_S12_cm": rc["recalibrated"]["S12"]["point"],
                   "raw_minus_recalibrated_cm": d, "paired_interval": [lo, hi], "meets_margin": meets,
                   "lower_above_0": above, "both": meets and above}
    if missing:
        reading = "not evaluated"
    else:
        reading = "GAIN" if rows and all(v["both"] for v in rows.values()) else "not GAIN"
    return {"reading": reading, "ends": rows, "missing": missing, "margin_cm": margin, "gain_ends": list(gain_ends),
            "rule": (f"GAIN when, at every end of {', '.join(gain_ends)}, the recalibrated S(12) is lower than the raw "
                     f"S(12) by at least {margin:g} cm and the paired lower bound is above 0; not GAIN otherwise; not "
                     "evaluated when such an end has no training-fold decode")}


# ---------------------------------------------------------------- the other inputs
def insim_gain(folder, expected_sha256=None):
    """The label-free in-sim gains of an eval_gait_insim.py folder (rule section 8), read after check_rule: stride
    1 + b / (100 t) (b the forward row's model stride bias in cm, t its person-weighted mean scored true stride in m),
    speed the median over the forward takes of model_speed_m_s / true_speed_m_s, and the median window pelvis speed
    ratio. ValueError when the folder holds another checkpoint than expected_sha256."""
    import pandas as pd
    check_rule()
    with open(os.path.join(folder, "insim.json"), encoding="utf-8") as f:
        d = json.load(f)
    sha = (d.get("checkpoint") or {}).get("sha256")
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{rel(folder)} holds checkpoint {str(sha)[:12]}, expected {expected_sha256[:12]}")
    b = float(d["lines"]["forward"]["model"]["stride_bias_cm"]["point"])
    t = float(d["counts"]["forward"]["true_stride_m"])
    tk = pd.read_csv(os.path.join(folder, "takes.csv"), encoding="utf-8")
    fw = tk[tk["forward"].astype(str).str.strip().str.lower() == "true"]
    vm = pd.to_numeric(fw["model_speed_m_s"], errors="coerce").to_numpy(np.float64)
    vt = pd.to_numeric(fw["true_speed_m_s"], errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(vm) & np.isfinite(vt) & (vt > 0)
    win = ((d.get("windows") or {}).get("forward") or {})
    pr = win.get("pelvis_speed_ratio") or {}
    return {"path": rel(folder), "checkpoint_sha256": sha, "row": "forward",
            "stride_gain": 1.0 + b / (100.0 * t) if t > 0 else float("nan"), "stride_bias_cm": b, "true_stride_m": t,
            "speed_gain": float(np.median(vm[ok] / vt[ok])) if ok.any() else float("nan"),
            "n_takes": int(ok.sum()), "n_forward_takes": int(len(fw)),
            "window_pelvis_speed_ratio_median": pr.get("median"), "n_windows": win.get("windows")}


def read_decode(folder, expected_sha256, train, locked):
    """(strides, speeds, counts, decode.json) of a --decode folder, None when its train_events.csv is absent.
    ValueError when decode.json names another checkpoint file; the persons of the file must be training-fold
    persons (read_events, after check_rule)."""
    path = os.path.join(folder, "train_events.csv")
    if not os.path.exists(path):
        return None
    check_rule()
    with open(os.path.join(folder, "decode.json"), encoding="utf-8") as f:
        doc = json.load(f)
    sha = (doc.get("checkpoint") or {}).get("sha256")
    if sha != expected_sha256:
        raise ValueError(f"{rel(folder)}: decode of checkpoint {str(sha)[:12]}, expected {expected_sha256[:12]}")
    st, sp, counts = read_events(path, train, locked)
    return st, sp, counts, doc


def cross_check(train_strides, dev_json, persons, tol=None, expected_sha256=None):
    """Per training-fold person, the stride MAE of the decode against the all68 row of the per_person.csv beside the
    G3 dev json (eval_g3_dev.py): differences, the persons beyond tol (cm) or on one side only, passed. Not evaluated
    (and said so) when that report is absent, or holds another checkpoint file than expected_sha256 or other weights
    than EMA (the decode and the diag run use EMA weights)."""
    import pandas as pd
    tol = RERUN_PERSON_TOL_CM if tol is None else float(tol)
    if dev_json is None or not os.path.exists(os.path.join(os.path.dirname(dev_json), "per_person.csv")):
        return {"evaluated": False, "reading": "no per_person.csv of a G3 dev report of this checkpoint"}
    with open(dev_json, encoding="utf-8") as f:
        ck = json.load(f).get("checkpoint") or {}
    if expected_sha256 is not None and ck.get("sha256") != expected_sha256:
        return {"evaluated": False, "reading": f"{rel(dev_json)} is of another checkpoint file (SHA-256 "
                                               f"{str(ck.get('sha256'))[:12]}, expected {expected_sha256[:12]})"}
    if ck.get("weights") not in (None, "ema"):
        return {"evaluated": False, "reading": f"{rel(dev_json)} holds {ck.get('weights')} weights, not EMA"}
    pp = pd.read_csv(os.path.join(os.path.dirname(dev_json), "per_person.csv"), encoding="utf-8", dtype={"person_id": str})
    keep = {str(p) for p in persons}
    ref = pp[(pp["row"] == "all68") & (pp["predictor"] == "model") & (pp["line"] == "stride_mae_cm")
             & pp["person_id"].isin(keep)]
    ref = {r.person_id: float(r.value) for r in ref.itertuples() if np.isfinite(float(r.value))}
    mine = person_abs_error(train_strides)
    both = sorted(set(ref) & set(mine))
    diff = {p: mine[p] - ref[p] for p in both}
    one_side = sorted(set(ref) ^ set(mine))
    beyond = sorted(p for p, d in diff.items() if abs(d) > tol)
    return {"evaluated": True, "reference": rel(os.path.join(os.path.dirname(dev_json), "per_person.csv")),
            "persons_compared": len(both), "max_abs_difference_cm": max((abs(d) for d in diff.values()), default=None),
            "persons_beyond_tol": beyond, "persons_on_one_side": one_side, "tolerance_cm": tol,
            "passed": not beyond and not one_side, "difference_cm": diff}


def consistency(raw, diag_doc, tol=None):
    """Raw dev lines recomputed from events.csv against the base row of diag.json (subsets all14 and
    no_whc003_whc026): AssertionError beyond tol or on a different person count; returns the table."""
    tol = CONSISTENCY_TOL if tol is None else float(tol)
    base = diag_doc["rows"]["base"]
    out, bad = {}, []
    for line, (kind, set_name) in LINE_DEF.items():
        sub = base[DIAG_SUBSET[set_name]]
        ref = sub[DIAG_KEY[kind]]
        n_ref = sub["n_persons"] if kind == "stride" else sub.get("n_persons_speed", sub["n_persons"])
        mine = raw[line]["point"]
        d = None if ref is None or not np.isfinite(mine) else float(mine - float(ref))
        ok = d is not None and abs(d) <= tol and int(n_ref) == raw[line]["n_persons"]
        out[line] = {"recomputed": mine, "diag": ref, "difference": d, "n_persons": raw[line]["n_persons"],
                     "n_persons_diag": n_ref, "ok": ok}
        if not ok:
            bad.append(line)
    if bad:
        raise AssertionError(f"raw lines differ from diag.json beyond {tol}: {bad} ({out})")
    return {"tolerance": tol, "lines": out, "ok": True}


def inference_differences(a, b):
    """{key: [a value, b value]} of the inference keys that differ (INFER_KEYS)."""
    a, b = a or {}, b or {}
    return {k: [a.get(k), b.get(k)] for k in INFER_KEYS if a.get(k) != b.get(k)}


# ---------------------------------------------------------------- the check
def analyse_end(end, sets, locked, tree=None, out=None, n_boot=None, seed=None, log=print):
    """Everything of the rule for one end: dev units and descriptive statistics, raw lines (asserted against
    diag.json), the training-fold decode when it exists (gains, recalibration, in-sample lines, checks) and the in-sim
    gain with its secondary recalibration."""
    check_rule()
    dd = diag_dir(end.name, tree)
    with open(os.path.join(dd, "diag.json"), encoding="utf-8") as f:
        diag = json.load(f)
    ck = diag.get("checkpoint") or {}
    if ck.get("name") != end.name or ck.get("weights", "ema") != "ema":
        raise ValueError(f"{rel(dd, tree)}: checkpoint {ck.get('name')} ({ck.get('weights')}), expected {end.name} (ema)")
    sha = ck.get("sha256")
    st, sp, counts = read_events(os.path.join(dd, "events.csv"), sets["dev14"], locked)
    units = units_of(st, sp)
    desc = {s: {u: describe(subset(units[u], sets[s]), n_boot, seed) for u in UNITS} for s in DEV_SETS}
    raw = lines_for(st, sp, sets, n_boot=n_boot, seed=seed)
    res = {"key": end.key, "name": end.name, "gain_end": end.gain_end,
           "checkpoint": {"path": end.checkpoint, "sha256": sha, "phase": ck.get("phase"), "step": ck.get("step"),
                          "weights": ck.get("weights", "ema")},
           "diag": {"path": rel(dd, tree), "inference": {k: (diag.get("inference") or {}).get(k) for k in INFER_KEYS}},
           "dev_counts": counts, "descriptive": desc, "raw": raw, "consistency": consistency(raw, diag)}
    log(f"{end.key}: raw S(12) {raw['S12']['point']:.3f} cm, V(12) {raw['V12']['point']:.4f} m/s (consistency ok)")
    dec = read_decode(decode_dir(end.name, out), sha, sets["train54"], locked)
    res["train_decode"] = {"path": rel(decode_dir(end.name, out), tree), "present": dec is not None}
    res["recalibration"] = None
    if dec is not None:
        tst, tsp, tcounts, doc = dec
        tunits = units_of(tst, tsp)
        g_s = fit_gain(tunits["stride_per_stride"], sets["train54"], sets["dev14"], locked)
        g_v = fit_gain(tunits["speed_per_pass"], sets["train54"], sets["dev14"], locked)
        res["gain"] = {"stride": g_s, "speed": g_v, "fitted_on": "train54"}
        res["recalibration"] = recalibration(st, sp, sets, g_s["gain"], g_v["gain"], None, n_boot, seed)
        res["descriptive"]["train54"] = {u: describe(tunits[u], n_boot, seed) for u in UNITS}
        res["train54_in_sample"] = recalibration(tst, tsp, sets, g_s["gain"], g_v["gain"], TRAIN_LINE_DEF, n_boot, seed)
        diffs = inference_differences(res["diag"]["inference"], doc.get("inference"))
        res["train_decode"].update({
            "counts": tcounts, "inference": {k: (doc.get("inference") or {}).get(k) for k in INFER_KEYS},
            "inference_equal_to_diag": not diffs, "inference_differences": diffs,
            "persons": int(tst["person_id"].nunique()), "trained_on_among_train54":
                (doc.get("sets") or {}).get("trained_on_among_train54"),
            "cross_check": cross_check(tst, dev_report(end.name, tree), sets["train54"], expected_sha256=sha)})
        log(f"{end.key}: g_S {g_s['gain']:.4f}, g_V {g_v['gain']:.4f}; recalibrated S(12) "
            f"{res['recalibration']['recalibrated']['S12']['point']:.3f} cm")
    ins = insim_dir(end.name, tree)
    res["insim"] = None
    if os.path.exists(os.path.join(ins, "insim.json")):
        ig = insim_gain(ins, sha)
        res["insim"] = {**ig, "recalibration": None, "recalibration_skipped": None}
        if usable_gain(ig["stride_gain"]) and usable_gain(ig["speed_gain"]):
            res["insim"]["recalibration"] = recalibration(st, sp, sets, ig["stride_gain"], ig["speed_gain"], None,
                                                          n_boot, seed)
        else:                               # a side report: it reads nothing, so it never stops the check
            res["insim"]["recalibration_skipped"] = (f"in-sim gains stride {ig['stride_gain']}, speed "
                                                     f"{ig['speed_gain']}: not both finite and above 0")
            log(f"{end.key}: in-sim recalibration skipped ({res['insim']['recalibration_skipped']})")
    return res


def run_check(sets, locked, ends=None, tree=None, out=None, n_boot=None, seed=None, log=print):
    """The check of the rule on every end (analyse_end) and the reading (gain_reading); returns the results dict
    (no file written)."""
    rule_sha = check_rule()
    check_sets(sets, locked)
    t0 = time.time()
    res = {e.key: analyse_end(e, sets, locked, tree, out, n_boot, seed, log) for e in (ENDS if ends is None else ends)}
    reading = gain_reading(res, [e.key for e in (ENDS if ends is None else ends) if e.gain_end])
    return {"what": LABEL, "task": TASK, "rule_sha256": rule_sha, "reading": reading, "ends": res,
            "constants": {"boot_n": BOOT_N if n_boot is None else int(n_boot), "seed": SEED if seed is None else seed,
                          "level": LEVEL, "margin_cm": MARGIN_CM, "consistency_tol": CONSISTENCY_TOL,
                          "rerun_person_tol_cm": RERUN_PERSON_TOL_CM},
            "timing_s": {"check": round(time.time() - t0, 1)}}


# ---------------------------------------------------------------- report
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
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


def _nd(unit):
    return 2 if unit == "cm" else 3


def summary_md(r):
    """The markdown report of a results dict (as written to results.json)."""
    rd = r["reading"]
    L = [f"# G3 gain check ({LABEL})", "",
         "Written by scripts/g3_gain_check.py (task F1) under the rule reports/bench/g3_gain_check/rule.txt (SHA-256 "
         f"{r['rule']['sha256']}, asserted before any target value was read). The rule records that the dev pattern "
         "was seen in a scratch reading during the planning critique, before the rule was written. Every number below "
         "is read from results.json of this folder. No lockbox person, EVAL or CODY record was read (asserted).", "",
         "## Reading", "",
         f"**{rd['reading']}**. Rule: {rd['rule']}." + (f" Missing training-fold decode: {', '.join(rd['missing'])}."
                                                        if rd["missing"] else ""), "",
         "| end | raw S(12) (cm) | recalibrated S(12) (cm) | raw minus recalibrated (cm) | 95 % paired interval | "
         f"at least {fmt(rd['margin_cm'])} cm | lower bound above 0 |", "|---|---|---|---|---|---|---|"]
    for k, v in rd["ends"].items():
        L.append(f"| {k} | {fmt(v['raw_S12_cm'])} | {fmt(v['recalibrated_S12_cm'])} | "
                 f"{fmt(v['raw_minus_recalibrated_cm'])} | {fmt_iv(v['paired_interval'])} | {v['meets_margin']} | "
                 f"{v['lower_above_0']} |")
    L += ["", "## Lines on the dev controls, raw and recalibrated by the training-fold gain", "",
          "Recalibrated: the decoded stride divided by g_S and the decoded pass speed by g_V, both fitted on the "
          f"{r['sets']['n_train54']} training-fold controls only. Intervals: person bootstrap, "
          f"{r['constants']['boot_n']:,} resamples, seed {r['constants']['seed']}; the difference is paired by person.",
          "", "| end | line | unit | persons | raw | 95 % | recalibrated | 95 % | raw minus recalibrated | 95 % paired |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        rc = e.get("recalibration")
        for line in LINE_DEF:
            raw = e["raw"][line]
            nd = _nd(raw["unit"])
            if rc:
                c, p = rc["recalibrated"][line], rc["paired"][line]
                tail = (f"{fmt(c['point'], nd)} | {fmt_iv(c['interval'], nd)} | {fmt(p['point'], nd)} | "
                        f"{fmt_iv(p['interval'], nd)} |")
            else:
                tail = "not computed | | | |"
            L.append(f"| {k} | {line} | {raw['unit']} | {raw['n_persons']} | {fmt(raw['point'], nd)} | "
                     f"{fmt_iv(raw['interval'], nd)} | {tail}")
    L += ["", "## Gains", "",
          "| end | g_S (train54, per stride) | strides / persons | g_V (train54, per pass) | passes / persons | in-sim "
          "stride gain | in-sim speed gain (takes) | forward takes | in-sim window pelvis speed ratio (median) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        g, ins = e.get("gain"), e.get("insim")
        gs = (f"{fmt(g['stride']['gain'], 4)} | {g['stride']['n_units']} / {g['stride']['n_persons']} | "
              f"{fmt(g['speed']['gain'], 4)} | {g['speed']['n_units']} / {g['speed']['n_persons']}") if g else \
            "not fitted (no training-fold decode) | | |"
        isx = (f"{fmt(ins['stride_gain'], 4)} | {fmt(ins['speed_gain'], 4)} | {ins['n_takes']} | "
               f"{fmt(ins['window_pelvis_speed_ratio_median'], 4)}") if ins else "absent | | |"
        L.append(f"| {k} | {gs} | {isx} |")
    L += ["", "In-sim gains: label-free (AMASS dev walks rendered through the training renderer, forward row of "
          "reports/gates/g3_sim/insim_<name>/): stride 1 + bias / (100 x mean true stride), speed the median over the "
          "forward takes of decoded over true speed. Reported beside; they read nothing.", "",
          "## Decoded against walkway: correlation, gain, within-person slope", "",
          "Units: each matched stride; each walkway pass (mean of its matched strides); each matched pass speed. Gain: "
          "median of decoded over walkway. Within-person slope: least-squares slope after removing each person's means. "
          "train54 is the fold the gain is fitted on (in-sample there).", "",
          "| end | set | unit | units | persons | Pearson r | 95 % | gain | 95 % | within-person slope | 95 % |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        for s, by in e["descriptive"].items():
            for u, d in by.items():
                L.append(f"| {k} | {s} | {u} | {d['n_units']} | {d['n_persons']} | {fmt(d['pearson_r']['point'], 3)} | "
                         f"{fmt_iv(d['pearson_r']['interval'], 3)} | {fmt(d['gain']['point'], 3)} | "
                         f"{fmt_iv(d['gain']['interval'], 3)} | {fmt(d['within_slope']['point'], 3)} | "
                         f"{fmt_iv(d['within_slope']['interval'], 3)} |")
    L += ["", "## Secondary: the dev lines after dividing by the in-sim gains (read nothing)", "",
          "| end | line | raw | recalibrated (in-sim) | 95 % | raw minus recalibrated | 95 % paired |",
          "|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        ins = e.get("insim")
        if not ins:
            L.append(f"| {k} | all | in-sim folder absent | | | | |")
            continue
        rc = ins.get("recalibration")
        if not rc:
            L.append(f"| {k} | all | not computed: {ins.get('recalibration_skipped')} | | | | |")
            continue
        for line in LINE_DEF:
            nd = _nd(rc["raw"][line]["unit"])
            c, p = rc["recalibrated"][line], rc["paired"][line]
            L.append(f"| {k} | {line} | {fmt(rc['raw'][line]['point'], nd)} | {fmt(c['point'], nd)} | "
                     f"{fmt_iv(c['interval'], nd)} | {fmt(p['point'], nd)} | {fmt_iv(p['interval'], nd)} |")
    L += ["", "## In-sample: the training-fold lines (the gain was fitted there)", "",
          "| end | line | unit | persons | raw | recalibrated | raw minus recalibrated | 95 % paired |",
          "|---|---|---|---|---|---|---|---|"]
    for k, e in r["ends"].items():
        t = e.get("train54_in_sample")
        if not t:
            L.append(f"| {k} | n/a | | | no training-fold decode | | | |")
            continue
        for line in TRAIN_LINE_DEF:
            nd = _nd(t["raw"][line]["unit"])
            L.append(f"| {k} | {line} | {t['raw'][line]['unit']} | {t['raw'][line]['n_persons']} | "
                     f"{fmt(t['raw'][line]['point'], nd)} | {fmt(t['recalibrated'][line]['point'], nd)} | "
                     f"{fmt(t['paired'][line]['point'], nd)} | {fmt_iv(t['paired'][line]['interval'], nd)} |")
    L += ["", "## Checks", ""]
    for k, e in r["ends"].items():
        c = e["consistency"]
        td = e["train_decode"]
        txt = (f"- {k} ({e['checkpoint']['path']}, SHA-256 {e['checkpoint']['sha256']}): raw lines recomputed from "
               f"{e['diag']['path']}/events.csv equal diag.json within {c['tolerance']:g} (max difference "
               f"{max(abs(v['difference']) for v in c['lines'].values()):.2e}). Dev events: "
               f"{e['dev_counts']['stride']['matched']} of {e['dev_counts']['stride']['events']} walkway strides and "
               f"{e['dev_counts']['speed']['matched']} of {e['dev_counts']['speed']['events']} passes matched.")
        if td["present"]:
            cc = td["cross_check"]
            txt += (f" Training-fold decode {td['path']}: {td['counts']['stride']['matched']} of "
                    f"{td['counts']['stride']['events']} strides and {td['counts']['speed']['matched']} of "
                    f"{td['counts']['speed']['events']} passes matched, {td['persons']} persons; inference settings "
                    + ("equal to the diag run" if td["inference_equal_to_diag"] else
                       f"differ from the diag run: {td['inference_differences']}")
                    + "; per-person stride MAE against the all68 row of the G3 dev report: "
                    + (f"{cc['persons_compared']} persons, largest difference {fmt(cc['max_abs_difference_cm'], 3)} cm, "
                       f"{'within' if cc['passed'] else 'NOT within'} {cc['tolerance_cm']:g} cm"
                       + (f" (beyond: {', '.join(cc['persons_beyond_tol'])}; one side only: "
                          f"{', '.join(cc['persons_on_one_side'])})" if not cc["passed"] else "")
                       if cc.get("evaluated") else cc["reading"]) + ".")
        else:
            txt += f" No training-fold decode ({td['path']}/train_events.csv absent)."
        L.append(txt)
    lb = r["lockbox"]
    L += [f"- Lockbox: persons of the sets in the weargait_pd lockbox list {lb['persons_in_lockbox']} (asserted 0); "
          "events and decode files read only after their persons were checked against the dev or training-fold set "
          "and the lockbox list.",
          f"- Sets: dev14 SHA-256 {r['sets']['dev14_sha256']}, dev12 leaves out {', '.join(r['sets']['excluded_from_12'])}, "
          f"train54 SHA-256 {r['sets']['train54_sha256']}.",
          f"- Rule file times: rule.txt {r['rule']['rule_mtime']}, rule.sha256 {r['rule']['sha_mtime']}; this run "
          f"started {r['run_started']}.",
          f"- MoveAtlas: HEAD {r['moveatlas'].get('head') or 'n/a'} (full stamp in _provenance.json).",
          f"- Time: {r['timing_s']['check']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- --decode
def decode(a, log=print):
    """--decode (module docstring): returns the output folder."""
    check_rule()                                          # before any target or record is read
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    import pandas as pd
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import loop as LP
    eg = EG()                                             # the inference of scripts/eval_g3_dev.py, by import
    t_start = time.time()
    train, dev = list(g3.controls("train")), list(g3.controls("dev"))
    locked = g3.lockbox_persons()
    guard_persons(train, None, locked, "train54")
    if set(train) & set(dev):
        raise AssertionError("dev and training fold overlap")
    device = torch.device(a.device)
    cap = LP.cap_memory(device, a.memory_gb)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=device)
    out = a.out or decode_dir(loaded.name)
    paths.assert_not_moveatlas(out)
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")
    tg = g3.load_targets(train)
    if tg.counts.get("lockbox_rows", 0):
        raise AssertionError(f"{tg.counts['lockbox_rows']} lockbox target rows kept")
    pred = I.G3Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                         seed=a.seed, pool=not a.no_pool)
    parts = []
    for tid, rec in g3.imu_records(train, tg.trials):
        check_record(tid, rec, train)
        tb = pred(tid, g3.model_input(rec))
        parts.append(g3.body_predictions([tb], tg))
        r = pred.log[-1]
        log(f"  {tid}: {r['windows']} windows, {float(r.get('seconds') or 0.0):.1f} s")
        del tb, rec
    if not parts:
        raise ValueError("no training-fold trial was decoded")
    cat = lambda k: pd.concat([getattr(p, k) for p in parts], ignore_index=True)  # noqa: E731
    P = g3.Predictions("model", cat("ics"), cat("strides"), cat("speeds"))
    ms, mv = g3.match_strides(P, tg), g3.match_speeds(P, tg)
    ev = DIAG().events_frame("base", {"strides": ms, "speeds": mv}, tg)
    common.write_csv(ev, os.path.join(out, "train_events.csv"))
    common.write_csv(pd.DataFrame(pred.log), os.path.join(out, "trials.csv"))
    mem = {}
    if device.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9, "cap_gb": cap}
    inference = {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S, "hop_s": I.HOP_S,
                 "batch_windows": a.batch_windows, "seed": a.seed,
                 "shape_pooling": "per trial" if not a.no_pool else "none", "weights": a.weights,
                 "device": str(device), "amp_bf16": device.type == "cuda", "memory": mem}
    doc = {"what": LABEL, "task": TASK, "mode": "decode", "checkpoint": loaded.describe(), "inference": inference,
           "rule": {"path": rel(RULE_PATH), "sha256": RULE_SHA256, "asserted_before_targets": True},
           "sets": {"train54": train, "train54_sha256": sources.person_hash(train), "n_train54": len(train),
                    "trained_on_among_train54": len(set(train) & loaded.trained_persons())},
           "target_counts": tg.counts,
           "model_events": {"ics": int(len(P.ics)), "strides": int(len(P.strides)), "pass_speeds": int(len(P.speeds))},
           "matched": {"strides": int(ms["matched"].sum()), "walkway_strides": int(len(ms)),
                       "passes": int(mv["matched"].sum()), "walkway_passes": int(len(mv))},
           "trials": len(pred.log), "timing_s": {"all": round(time.time() - t_start, 1)}}
    common.write_text(os.path.join(out, "decode.json"),
                      json.dumps(eg.clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    prov = {"script": "scripts/g3_gain_check.py", "mode": "decode", "label": LABEL, "checkpoint": rel(loaded.path),
            "checkpoint_sha256": loaded.sha256, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "rule_sha256": RULE_SHA256,
            "moveatlas_stamp": stamp_moveatlas(),
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__, g3.gait.__file__, I.__file__,
                                               eg.__file__, DIAG().__file__]),
            "code_files": ["scripts/g3_gain_check.py", "mdfm/gates/g3.py", "mdfm/readouts/gait.py",
                           "mdfm/eval/infer.py", "scripts/eval_g3_dev.py", "scripts/eval_g3_diag.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    log(f"decoded {len(pred.log)} trials: {doc['matched']}; written {rel(out)}")
    return out


def job_commands():
    """The scripts/queue_job.py argument lists of the decode jobs, one per gain end (cls 1, --memory-gb 12, explicit
    --out under reports/bench/g3_gain_check/)."""
    cmds = []
    for i, e in enumerate(x for x in ENDS if x.gain_end):
        out = f"reports/bench/g3_gain_check/decode_{e.name}"
        cmds.append(["scripts/queue_job.py", "--name", f"f1_gain_decode_{e.key}", "--cls", "1", "--priority",
                     str(JOB_PRIORITY0 + i), "--est-hours", str(JOB_EST_HOURS), "--group", "F1",
                     "--requires", e.checkpoint, "--outputs", f"{out}/train_events.csv",
                     "--note", f"F1 gain check: decode of the 54 training-fold controls by {e.name}",
                     "--", "scripts/g3_gain_check.py", "--decode", "--checkpoint", e.checkpoint,
                     "--batch-windows", str(e.batch_windows), "--memory-gb", JOB_MEMORY_GB, "--out", out])
    return cmds


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--decode", action="store_true", help="decode the 54 training-fold controls (GPU, queued)")
    p.add_argument("--print-jobs", action="store_true", help="print the queue_job.py commands of the decode jobs")
    p.add_argument("--checkpoint", help="--decode: a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256, help="kernel points per stream-window and refinement step")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a trial")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.decode and not a.checkpoint:
        p.error("--decode needs --checkpoint")
    if a.checkpoint and not a.decode:
        p.error("--checkpoint is read by --decode only")
    return a


def _mtime(path):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))


def finalize(res, sets, locked, stamp, run_started):
    """The JSON-ready results (non-finite numbers as null) with the rule, sets, lockbox and MoveAtlas blocks."""
    from mdfm.data import sources
    out = dict(res)
    out["run_started"] = run_started
    out["rule"] = {"path": rel(RULE_PATH), "sha256": res["rule_sha256"], "sha_file": rel(RULE_SHA_PATH),
                   "asserted_before_targets": True, "rule_mtime": _mtime(RULE_PATH), "sha_mtime": _mtime(RULE_SHA_PATH)}
    out["sets"] = {"dev14": sorted(sets["dev14"]), "dev12": sorted(sets["dev12"]),
                   "excluded_from_12": list(EXCLUDED_FROM_12), "dev14_sha256": sources.person_hash(sets["dev14"]),
                   "dev12_sha256": sources.person_hash(sets["dev12"]),
                   "train54_sha256": sources.person_hash(sets["train54"]), "n_train54": len(sets["train54"])}
    n_lb = len((set(sets["dev14"]) | set(sets["train54"])) & set(locked))
    if n_lb:
        raise AssertionError(f"{n_lb} lockbox persons in the sets")
    out["lockbox"] = {"persons_in_lockbox": n_lb, "asserted": True,
                      "reader_flags": {"allow_lockbox": False, "allow_eval": False}}
    out["moveatlas"] = {"head": (stamp or {}).get("head")}
    return clean(out)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_jobs:
        py = "<conda>/envs/mdfm/python.exe"
        for c in job_commands():
            print(" ".join([py] + [f'"{x}"' if " " in x else x for x in c]))
        return 0
    if a.decode:
        decode(a)
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("the check runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    from mdfm.gates import common, g3
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    check_rule()
    train, dev = list(g3.controls("train")), list(g3.controls("dev"))
    locked = g3.lockbox_persons()
    sets = {"dev14": dev, "dev12": [p for p in dev if p not in EXCLUDED_FROM_12], "train54": train}
    res = run_check(sets, locked)
    stamp = stamp_moveatlas()
    out = finalize(res, sets, locked, stamp, run_started)
    text = summary_md(out)
    common.assert_no_em_dash(text, "summary.md")
    common.write_text(os.path.join(OUT, "results.json"),
                      json.dumps(out, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), text)
    prov = {"script": "scripts/g3_gain_check.py", "mode": "check", "label": LABEL, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "numpy": np.__version__, "rule_sha256": res["rule_sha256"],
            "inputs": {e.key: {"diag": rel(os.path.join(diag_dir(e.name), "events.csv")),
                               "decode": rel(os.path.join(decode_dir(e.name), "train_events.csv")),
                               "insim": rel(insim_dir(e.name))} for e in ENDS},
            "code_sha256": common.code_sha256([os.path.abspath(__file__), g3.__file__]),
            "code_files": ["scripts/g3_gain_check.py", "mdfm/gates/g3.py"],
            "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    rule_t = max(os.path.getmtime(RULE_PATH), os.path.getmtime(RULE_SHA_PATH))
    for f in ("results.json", "summary.md", "_provenance.json"):
        if not os.path.getmtime(os.path.join(OUT, f)) > rule_t:
            raise AssertionError(f"{f} is not newer than the rule files")
    print(text)
    print(f"written {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
