"""F2-CHAL (2026-10-05): adversarial check of the F2 scale diagnosis (docs/notes/F2_SCALE_DIAGNOSIS_2026-10-05.md and
docs/proposals/F2_LEVER1_RULE_DRAFT.md). Label of every output: dev diagnostic, not a G3 read.

Pre-registration: reports/weekend_2026-10-01/tasks/F2-CHAL.md, section Prediction, copied verbatim to
reports/bench/f2_scale/CHAL/prediction.txt; its SHA-256 is PREDICTION_SHA256 here and in prediction.sha256 beside it.
check_prediction asserts the three agree, and that the report still holds the text, before any value is read.

Tests (CPU only, existing files only):
- R, the speed-range confound: on train54 (F1's training-fold decodes), the gain of the units whose walkway value lies
  in the in-sim band (pass speed [0.80, 1.20] m/s, stride [95, 125] cm) against F1's pooled gain and its in-sim gain;
  h = ln(g_in / g_all) / ln(g_sim / g_all) with a person bootstrap. Descriptive beside: gains by speed bin and task,
  the in-sim forward takes' speed ratio against their speed.
- X, cross-fit of the lever-1 head: leave-one-person-out S54 and V54 against the in-sample lines of F1; a
  person-weighted head and its dev12 lines beside.
- M, matched-stride selection: share of unmatched train54 walkway strides per task.
- A (not pre-registered): H4's f with the in-sim pass-speed divisors printed by H3.
- Q (not pre-registered): every number with two or more decimals quoted by the note and the draft rule, looked up
  in the files the section cites.
- D: every person id of the F2 output files inside dev14 or train54 and none in the weargait_pd lockbox list.

Data discipline: sets dev14, dev12, train54 of reports/bench/g3_gain_check/results.json, SHA-256 recomputed and
asserted; none in mdfm.gates.g3.lockbox_persons() (asserted); the person column of every events file is read and
guarded before any other column. No lockbox, EVAL, CODY or care_pd record; nothing written outside
reports/bench/f2_scale/CHAL/.

Run:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/f2_CHAL.py
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))
from mdfm import paths  # noqa: E402

TREE = paths.TREE
SCRIPTS = os.path.join(TREE, "scripts")
OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "CHAL")
F1_DIR = os.path.join(TREE, "reports", "bench", "g3_gain_check")
F1_RESULTS = os.path.join(F1_DIR, "results.json")
F2_DIR = os.path.join(TREE, "reports", "bench", "f2_scale")
REPORT = os.path.join(TREE, "reports", "weekend_2026-10-01", "tasks", "F2-CHAL.md")
NOTE = os.path.join(TREE, "docs", "notes", "F2_SCALE_DIAGNOSIS_2026-10-05.md")
DRAFT = os.path.join(TREE, "docs", "proposals", "F2_LEVER1_RULE_DRAFT.md")
LABEL = "dev diagnostic, not a G3 read"
TASK = "F2-CHAL"

PREDICTION_SHA256 = "756e5a5445624fe17b3f79291fd0c5cc7c6a33cd67a5a7c60e0826a7a4f22a4e"
BOOT_N = 10000
SEED = 20261005
LEVEL = 0.95
SPEED_BAND = (0.80, 1.20)          # m/s, walkway pass speed
STRIDE_BAND = (95.0, 125.0)        # cm, walkway stride
H_MAIN = 0.5
MIN_PERSONS_IN_BAND = 20
SPEED_BINS = ((0.0, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 9.9))
X_REL, X_ABS_S, X_ABS_V = 0.05, 0.10, 0.002
M_MAX_UNMATCHED = 0.10
N_DEV, N_TRAIN = 14, 54
ENDS = (("v1_B", "phaseA_root_full_ckpt_B_032044"), ("s2_B", "pilot_v1_s2_ckpt_B_032044"))


def _sibling(name):
    key = f"_f2chal_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def F1():
    return _sibling("g3_gain_check")


def rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), TREE).replace("\\", "/")
    except ValueError:                                # another drive (tests on a temporary folder)
        return str(path).replace("\\", "/")


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- pre-registration and sets
def check_prediction(report=REPORT, out=OUT, expected=PREDICTION_SHA256):
    """AssertionError unless prediction.txt hashes to `expected`, prediction.sha256 names it, and the report still
    holds the text verbatim."""
    with open(os.path.join(out, "prediction.txt"), encoding="utf-8", newline="") as f:
        text = f.read()
    if sha256_text(text) != expected:
        raise AssertionError("prediction.txt differs from the registered prediction")
    with open(os.path.join(out, "prediction.sha256"), encoding="utf-8") as f:
        if f.read().split()[0] != expected:
            raise AssertionError("prediction.sha256 names another hash")
    with open(report, encoding="utf-8", newline="") as f:
        if text not in f.read():
            raise AssertionError(f"{rel(report)} no longer holds the registered Prediction section")
    return expected


def load_sets(path=None, locked=None, train=None):
    """({dev14, dev12, train54}, locked) of F1's results.json, each SHA-256 recomputed and asserted, dev and train
    disjoint, dev12 inside dev14; Refused (via g3_gain_check.guard_persons) for a lockbox person."""
    from mdfm.data import sources
    with open(path or F1_RESULTS, encoding="utf-8") as f:
        s = json.load(f)["sets"]
    if locked is None:
        from mdfm.gates import g3
        locked = g3.lockbox_persons()
    if train is None:
        from mdfm.gates import g3
        train = g3.controls("train")
    sets = {"dev14": sorted(s["dev14"]), "dev12": sorted(s["dev12"]), "train54": sorted(train)}
    for k in sets:
        if sources.person_hash(sets[k]) != s[f"{k}_sha256"]:
            raise AssertionError(f"{k}: SHA-256 differs from F1's results.json")
        F1().guard_persons(sets[k], None, locked, k)
    if len(sets["dev14"]) != N_DEV or len(sets["train54"]) != N_TRAIN:
        raise AssertionError("set sizes differ from 14 and 54")
    if set(sets["dev14"]) & set(sets["train54"]) or not set(sets["dev12"]) <= set(sets["dev14"]):
        raise AssertionError("sets not nested and disjoint as expected")
    return sets, sorted(locked)


def read_all_rows(path, allowed, locked, keep, row="base"):
    """Every row of `row` (matched or not) of an events file, after its person column was guarded (inside
    `allowed`, none in `locked`) and the frame filtered to `keep`."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    F1().guard_persons(set(who["person_id"].dropna()), allowed, locked, os.path.basename(path))
    F1().guard_persons(keep, allowed, locked, "kept persons")
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[ev["person_id"].isin(set(keep)) & (ev["row"] == row)].reset_index(drop=True)
    ev["is_matched"] = ev["matched"].str.strip().str.lower() == "true"
    return ev


def matched_units(ev):
    """(strides, speeds) frames of the matched rows: person_id, trial_id, task, pass_id, x (walkway), y."""
    import pandas as pd
    out = []
    for kind in ("stride", "speed"):
        k = ev[(ev["kind"] == kind) & ev["is_matched"]]
        d = pd.DataFrame({"person_id": k["person_id"].to_numpy(), "trial_id": k["trial_id"].to_numpy(),
                          "task": k["task"].to_numpy(), "pass_id": k["pass_id"].to_numpy(),
                          "x": pd.to_numeric(k["reference"]).to_numpy(np.float64),
                          "y": pd.to_numeric(k["prediction"]).to_numpy(np.float64)})
        out.append(d.reset_index(drop=True))
    return out[0], out[1]


# ---------------------------------------------------------------- statistics
def pooled_gain(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return float(np.median(y[ok] / x[ok])) if ok.any() else float("nan")


def share_h(g_in, g_all, g_sim):
    """ln(g_in / g_all) / ln(g_sim / g_all); NaN when undefined."""
    try:
        den = math.log(g_sim / g_all)
        return math.log(g_in / g_all) / den if den != 0 else float("nan")
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _interval(v, level=LEVEL):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


def band_test(units, band, g_sim, n_boot=BOOT_N, seed=SEED):
    """Test R on one line: g_all, g_in (units with x in `band`, closed), h, and their person bootstrap intervals
    (g_in and g_all recomputed per resample, g_sim fixed); persons with a unit in the band."""
    x = units["x"].to_numpy(np.float64)
    y = units["y"].to_numpy(np.float64)
    p = units["person_id"].astype(str).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y, p = x[ok], y[ok], p[ok]
    r = y / x
    inb = (x >= band[0]) & (x <= band[1])
    ids, inv = np.unique(p, return_inverse=True)
    per_r = [r[inv == i] for i in range(len(ids))]
    per_in = [r[(inv == i) & inb] for i in range(len(ids))]
    g_all, g_in = float(np.median(r)), (float(np.median(r[inb])) if inb.any() else float("nan"))
    out = {"band": list(band), "g_sim": g_sim, "g_all": g_all, "g_in": g_in, "h": share_h(g_in, g_all, g_sim),
           "units": int(len(r)), "units_in_band": int(inb.sum()), "persons": int(len(ids)),
           "persons_in_band": int(sum(len(a) > 0 for a in per_in))}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    ga, gi, hh = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        ra = np.concatenate([per_r[i] for i in draws[b]])
        ri = np.concatenate([per_in[i] for i in draws[b]])
        ga[b] = np.median(ra)
        gi[b] = np.median(ri) if len(ri) else np.nan
        hh[b] = share_h(gi[b], ga[b], g_sim)
    out.update({"g_all_interval": _interval(ga), "g_in_interval": _interval(gi), "h_interval": _interval(hh)})
    return out


def band_reading(t, h_main=H_MAIN, min_persons=MIN_PERSONS_IN_BAND):
    if t["persons_in_band"] < min_persons or t["h_interval"] is None:
        return "NOT EVALUATED"
    lo, hi = t["h_interval"]
    if t["h"] >= h_main and lo > 0:
        return "MAIN"
    if hi < h_main:
        return "MINOR"
    return "MIXED"


def range_reading(per):
    """per: {(end, line): reading}. RANGE MAIN / RANGE MINOR / MIXED / NOT EVALUATED."""
    vals = list(per.values())
    if any(v == "NOT EVALUATED" for v in vals):
        return "NOT EVALUATED"
    if all(v == "MAIN" for v in vals):
        return "RANGE MAIN"
    if all(v == "MINOR" for v in vals):
        return "RANGE MINOR"
    return "MIXED"


def person_mae(units, gain):
    if not len(units):
        return {}
    e = np.abs(units["y"].to_numpy(np.float64) / float(gain) - units["x"].to_numpy(np.float64))
    s = units.assign(_e=e).groupby(units["person_id"].astype(str))["_e"].mean()
    return {str(k): float(v) for k, v in s.sort_index().items()}


def lopo(units):
    """Leave-one-person-out: per person, the gain fitted on the others (pooled median) and the person's mean absolute
    error with it. Returns {line: mean over persons, per_person, gains}."""
    p = units["person_id"].astype(str).to_numpy()
    x = units["x"].to_numpy(np.float64)
    y = units["y"].to_numpy(np.float64)
    out, gains = {}, {}
    for pid in sorted(set(p)):
        g = pooled_gain(x[p != pid], y[p != pid])
        gains[pid] = g
        out[pid] = float(np.mean(np.abs(y[p == pid] / g - x[p == pid])))
    return {"line": float(np.mean(list(out.values()))), "per_person": out, "gain_min": float(min(gains.values())),
            "gain_max": float(max(gains.values()))}


def person_weighted_gain(units):
    g = units.assign(_r=units["y"] / units["x"]).groupby(units["person_id"].astype(str))["_r"].median()
    return float(np.median(g.to_numpy(np.float64)))


def x_reading(rows):
    ok = all(r["S_lopo_minus_in"] <= max(X_REL * r["S_in"], X_ABS_S) and
             r["V_lopo_minus_in"] <= max(X_REL * r["V_in"], X_ABS_V) for r in rows.values())
    return "TRANSFERS" if ok else "DOES NOT TRANSFER"


def f_share(d_legs_ratio, d_root_ratio):
    """1 - ln(legs) / ln(root), the H4 share computed on ratios real / sim."""
    return 1.0 - math.log(d_legs_ratio) / math.log(d_root_ratio)


# ---------------------------------------------------------------- reproduction scan
def json_numbers(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            json_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            json_numbers(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
        out.append(float(obj))
    return out


WEARGAIT_ID = re.compile(r"^W?HC\d+$")
NUM = re.compile(r"(?<![\w.])(-?\d+\.\d{2,})(?![\w.])")


def scan_numbers(text, pool):
    """Each decimal token of `text` (two or more decimals) and whether some value of `pool` rounds to it (also as a
    percentage, x 100). Returns (found, missing) lists of tokens."""
    pool = np.asarray(pool, np.float64)
    found, missing = [], []
    for tok in NUM.findall(text):
        q = float(tok)
        d = len(tok.split(".")[1])
        tol = 0.5 * 10 ** (-d) + 1e-9
        hit = bool(np.any(np.abs(pool - q) <= tol) or np.any(np.abs(pool * 100 - q) <= tol))
        (found if hit else missing).append(tok)
    return found, missing


def sections(text):
    """{heading: body} of the '## ' and '### ' sections of a markdown text (body up to the next heading)."""
    parts, cur, buf = {}, "_head", []
    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            parts[cur] = "\n".join(buf)
            cur, buf = line.strip("# ").strip(), []
        else:
            buf.append(line)
    parts[cur] = "\n".join(buf)
    return parts


def pools():
    def load(p):
        with open(p, encoding="utf-8") as f:
            return json_numbers(json.load(f))
    f1 = load(F1_RESULTS)
    h = {k: load(os.path.join(F2_DIR, k, "results.json")) for k in ("H1shape", "H2sensor", "H3angles", "H4root")}
    seeds = load(os.path.join(TREE, "reports", "gpu_queue", "branch_seeds_20261002_211426.json"))
    with open(os.path.join(F1_DIR, "decode_phaseA_root_full_ckpt_B_032044", "decode.json"), encoding="utf-8") as f:
        dec = json_numbers(json.load(f))
    chal = os.path.join(OUT, "results.json")       # this task's own numbers, quoted by the corrected sections
    own = load(chal) if os.path.exists(chal) else []
    allp = f1 + sum(h.values(), []) + seeds + dec + own
    return {"F1": f1 + dec, "H1": h["H1shape"] + f1, "H2": h["H2sensor"] + f1, "H3": h["H3angles"] + f1,
            "H4": h["H4root"] + f1, "all": allp}


def reproduction_scan():
    P = pools()
    out = {}
    with open(NOTE, encoding="utf-8") as f:
        note = sections(f.read())
    for head, body in note.items():
        if head.startswith("Challenge"):
            continue                                 # this task's own section is not a quoted claim
        key = next((k for k in ("H1", "H2", "H3", "H4") if head.startswith(k + ",")), None)
        found, missing = scan_numbers(body, P[key] if key else P["all"])
        out[f"note: {head}"] = {"pool": key or "all", "found": len(found), "missing": missing}
    with open(DRAFT, encoding="utf-8") as f:
        found, missing = scan_numbers(f.read(), P["all"])
    out["draft rule"] = {"pool": "all", "found": len(found), "missing": missing}
    return out


# ---------------------------------------------------------------- data discipline scan
def discipline_scan(sets, locked, root=F2_DIR):
    import pandas as pd
    allowed = set(sets["dev14"]) | set(sets["train54"])
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        if os.path.basename(path).startswith("._") or os.sep + "CHAL" + os.sep in path:
            continue
        cols = list(pd.read_csv(path, encoding="utf-8", nrows=0).columns)
        pc = [c for c in cols if c in ("person_id", "person")]
        if not pc:
            rows.append({"file": rel(path), "person_column": None, "persons": 0, "outside": 0, "lockbox": 0})
            continue
        raw = set(pd.read_csv(path, encoding="utf-8", usecols=pc, dtype=str)[pc[0]].dropna())
        wg = {w if w.startswith("weargait:") else f"weargait:{w}" for w in raw
              if w.startswith("weargait:") or WEARGAIT_ID.match(w)}
        rows.append({"file": rel(path), "person_column": pc[0], "persons": len(wg),
                     "other_ids": len(raw) - len(wg), "outside": len(wg - allowed), "lockbox": len(wg & set(locked))})
    return rows


# ---------------------------------------------------------------- in-sim takes (descriptive)
def insim_takes(name):
    import pandas as pd
    p = os.path.join(TREE, "reports", "gates", "g3_sim", f"insim_{name}", "takes.csv")
    t = pd.read_csv(p, encoding="utf-8")
    fw = t[t["forward"].astype(str).str.lower().isin(("true", "1"))]
    v, m = fw["true_speed_m_s"].to_numpy(np.float64), fw["model_speed_m_s"].to_numpy(np.float64)
    ok = np.isfinite(v) & np.isfinite(m) & (v > 0)
    v, m = v[ok], m[ok]
    r = m / v
    slope = float(np.polyfit(v, m, 1)[0]) if len(v) > 2 else float("nan")
    inb = (v >= SPEED_BAND[0]) & (v <= SPEED_BAND[1])
    return {"path": rel(p), "takes": int(len(v)), "true_speed_q10_q50_q90": [float(q) for q in np.quantile(v, [.1, .5, .9])],
            "true_speed_max": float(v.max()), "median_ratio": float(np.median(r)),
            "median_ratio_in_band": float(np.median(r[inb])) if inb.any() else None, "takes_in_band": int(inb.sum()),
            "slope_decoded_on_true": slope}


# ---------------------------------------------------------------- main run
def run(n_boot=BOOT_N, log=print):
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    pred = check_prediction()
    sets, locked = load_sets()
    log(f"sets ok, prediction {pred[:12]} asserted")
    with open(F1_RESULTS, encoding="utf-8") as f:
        f1 = json.load(f)
    res = {"what": LABEL, "task": TASK, "prediction_sha256": pred, "run_started": started,
           "constants": {"speed_band": SPEED_BAND, "stride_band": STRIDE_BAND, "h_main": H_MAIN,
                         "min_persons_in_band": MIN_PERSONS_IN_BAND, "boot_n": n_boot, "seed": SEED,
                         "x_rel": X_REL, "x_abs_s_cm": X_ABS_S, "x_abs_v_m_s": X_ABS_V,
                         "m_max_unmatched": M_MAX_UNMATCHED},
           "R": {}, "X": {}, "M": {}, "descriptive": {}}
    readings_r, rows_x, m_ok = {}, {}, True
    for key, name in ENDS:
        e = f1["ends"][key]
        tr = read_all_rows(os.path.join(F1_DIR, f"decode_{name}", "train_events.csv"), sets["train54"], locked,
                           sets["train54"])
        st, sp = matched_units(tr)
        # F1 consistency
        gS, gV = pooled_gain(st["x"], st["y"]), pooled_gain(sp["x"], sp["y"])
        assert abs(gS - e["gain"]["stride"]["gain"]) < 1e-6 and abs(gV - e["gain"]["speed"]["gain"]) < 1e-6, \
            "pooled gains differ from F1"
        assert len(st) == e["gain"]["stride"]["n_units"] and len(sp) == e["gain"]["speed"]["n_units"]
        # R
        tS = band_test(st, STRIDE_BAND, e["insim"]["stride_gain"], n_boot)
        tV = band_test(sp, SPEED_BAND, e["insim"]["speed_gain"], n_boot)
        tS["reading"], tV["reading"] = band_reading(tS), band_reading(tV)
        readings_r[(key, "stride")], readings_r[(key, "speed")] = tS["reading"], tV["reading"]
        res["R"][key] = {"stride": tS, "speed": tV}
        # descriptive: bins by pass speed and by task
        spx = sp.set_index(["person_id", "trial_id", "pass_id"])["x"]
        st_v = st.join(spx.rename("v_pass"), on=["person_id", "trial_id", "pass_id"])
        bins = []
        for lo, hi in SPEED_BINS:
            a = sp[(sp["x"] >= lo) & (sp["x"] < hi)]
            b = st_v[(st_v["v_pass"] >= lo) & (st_v["v_pass"] < hi)]
            bins.append({"bin_m_s": [lo, hi], "passes": int(len(a)), "persons_passes": int(a["person_id"].nunique()),
                         "speed_gain": pooled_gain(a["x"], a["y"]), "strides": int(len(b)),
                         "stride_gain": pooled_gain(b["x"], b["y"]),
                         "stride_mean_walkway_cm": float(b["x"].mean()) if len(b) else None})
        tasks = {}
        for task in sorted(set(sp["task"])):
            a, b = sp[sp["task"] == task], st[st["task"] == task]
            tasks[task] = {"passes": int(len(a)), "speed_gain": pooled_gain(a["x"], a["y"]),
                           "walkway_speed_median": float(a["x"].median()), "strides": int(len(b)),
                           "stride_gain": pooled_gain(b["x"], b["y"]),
                           "walkway_stride_median_cm": float(b["x"].median())}
        # estimator check (not pre-registered): mean-based gain as the in-sim definition
        mean_based = {"stride": float(1 + (st["y"] - st["x"]).mean() / st["x"].mean()),
                      "speed": float(1 + (sp["y"] - sp["x"]).mean() / sp["x"].mean())}
        res["descriptive"][key] = {"speed_bins": bins, "tasks": tasks, "mean_based_gain_train54": mean_based,
                                   "insim_takes": insim_takes(name)}
        # X
        S_in = float(np.mean(list(person_mae(st, gS).values())))
        V_in = float(np.mean(list(person_mae(sp, gV).values())))
        ins = e["train54_in_sample"]["recalibrated"]
        assert abs(S_in - ins["S54"]["point"]) < 1e-6 and abs(V_in - ins["V54"]["point"]) < 1e-6, \
            "in-sample lines differ from F1"
        lS, lV = lopo(st), lopo(sp)
        rows_x[key] = {"S_in": S_in, "S_lopo": lS["line"], "S_lopo_minus_in": lS["line"] - S_in,
                       "V_in": V_in, "V_lopo": lV["line"], "V_lopo_minus_in": lV["line"] - V_in,
                       "lopo_gain_range_stride": [lS["gain_min"], lS["gain_max"]],
                       "lopo_gain_range_speed": [lV["gain_min"], lV["gain_max"]]}
        # person-weighted head and dev12 lines
        dv = read_all_rows(os.path.join(TREE, "reports", "gates", "g3", f"diag_{name}", "events.csv"),
                           sets["dev14"], locked, sets["dev12"])
        dst, dsp = matched_units(dv)
        pwS, pwV = person_weighted_gain(st), person_weighted_gain(sp)
        S12_f1 = float(np.mean(list(person_mae(dst, gS).values())))
        V12_f1 = float(np.mean(list(person_mae(dsp, gV).values())))
        assert abs(S12_f1 - e["recalibration"]["recalibrated"]["S12"]["point"]) < 1e-6, "dev12 S differs from F1"
        rows_x[key].update({"person_weighted_gain_stride": pwS, "person_weighted_gain_speed": pwV,
                            "S12_pooled_head": S12_f1, "V12_pooled_head": V12_f1,
                            "S12_person_weighted_head": float(np.mean(list(person_mae(dst, pwS).values()))),
                            "V12_person_weighted_head": float(np.mean(list(person_mae(dsp, pwV).values())))})
        # M
        trs = tr[tr["kind"] == "stride"]
        per_task = {}
        for task in sorted(set(trs["task"])):
            a = trs[trs["task"] == task]
            per_task[task] = {"walkway_strides": int(len(a)), "unmatched": int((~a["is_matched"]).sum()),
                              "share_unmatched": float((~a["is_matched"]).mean())}
            m_ok &= per_task[task]["share_unmatched"] <= M_MAX_UNMATCHED
        res["M"][key] = per_task
        log(f"{key}: R stride {tS['reading']} speed {tV['reading']}; X {rows_x[key]['S_lopo_minus_in']:.3f} cm")
    res["R"]["per_line"] = {f"{k[0]} {k[1]}": v for k, v in readings_r.items()}
    res["R"]["reading"] = range_reading(readings_r)
    res["X"] = {"ends": rows_x, "reading": x_reading(rows_x)}
    res["M"]["reading"] = "NEGLIGIBLE" if m_ok else "SELECTED"
    # A: H4 f with the pass-speed divisors of H3
    with open(os.path.join(F2_DIR, "H3angles", "results.json"), encoding="utf-8") as f:
        h3 = json.load(f)
    a = {}
    for key in ("v1_B", "s2_B"):
        for s in ("dev12", "dev14"):
            g = h3["part_a"][key]["speed_gains"][s]
            a[f"{key} {s}"] = {"D_L_speed": g["D_L_speed"], "D_R_speed": g["D_R_speed"],
                               "f_speed": f_share(g["D_L_speed"], g["D_R_speed"])}
    res["A"] = {"source": "reports/bench/f2_scale/H3angles/results.json part_a.<end>.speed_gains", "f_speed": a,
                "note": "not pre-registered: inputs seen before the Prediction section"}
    res["Q"] = reproduction_scan()
    disc = discipline_scan(sets, locked)
    res["D"] = {"files": disc, "outside_total": sum(r["outside"] for r in disc),
                "lockbox_total": sum(r["lockbox"] for r in disc)}
    assert res["D"]["lockbox_total"] == 0, "a lockbox person in an F2 output file"
    from mdfm.data import sources
    res["sets"] = {k: sources.person_hash(v) for k, v in sets.items()}
    res["lockbox"] = {"persons_in_sets": len((set(sets["dev14"]) | set(sets["train54"])) & set(locked)),
                      "list_size": len(locked)}
    assert res["lockbox"]["persons_in_sets"] == 0
    res["seconds"] = round(time.time() - t0, 1)
    return res


def fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def iv(i, d=3):
    return "" if not i else f"[{fmt(i[0], d)}, {fmt(i[1], d)}]"


def summary(r):
    L = [f"# F2-CHAL: adversarial check of the F2 scale diagnosis ({LABEL})", "",
         "Written by scripts/f2_CHAL.py; every number below is read from results.json of this folder. Prediction: "
         f"reports/weekend_2026-10-01/tasks/F2-CHAL.md, SHA-256 {r['prediction_sha256']} (asserted before any value "
         f"was read). Sets equal to F1's (SHA-256); lockbox persons in the sets {r['lockbox']['persons_in_sets']} of a "
         f"list of {r['lockbox']['list_size']} (asserted).", "",
         f"## Test R, speed-range confound: **{r['R']['reading']}**", "",
         "| end | line | band | units in band / all | persons in band | g_all | 95 % | g_in | 95 % | g_sim | h | 95 % | reading |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _ in ENDS:
        for line in ("stride", "speed"):
            t = r["R"][key][line]
            L.append(f"| {key} | {line} | {t['band']} | {t['units_in_band']} / {t['units']} | {t['persons_in_band']} | "
                     f"{fmt(t['g_all'])} | {iv(t['g_all_interval'])} | {fmt(t['g_in'])} | {iv(t['g_in_interval'])} | "
                     f"{fmt(t['g_sim'])} | {fmt(t['h'])} | {iv(t['h_interval'])} | {t['reading']} |")
    L += ["", "Descriptive (no criterion): gains by walkway pass speed bin and by task, train54.", "",
          "| end | bin (m/s) | passes | persons | speed gain | strides | stride gain | mean walkway stride (cm) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, _ in ENDS:
        for b in r["descriptive"][key]["speed_bins"]:
            L.append(f"| {key} | {b['bin_m_s']} | {b['passes']} | {b['persons_passes']} | {fmt(b['speed_gain'])} | "
                     f"{b['strides']} | {fmt(b['stride_gain'])} | {fmt(b['stride_mean_walkway_cm'], 1)} |")
    L += ["", "| end | task | passes | walkway speed median | speed gain | strides | walkway stride median (cm) | stride gain |",
          "|---|---|---|---|---|---|---|---|"]
    for key, _ in ENDS:
        for task, t in r["descriptive"][key]["tasks"].items():
            L.append(f"| {key} | {task} | {t['passes']} | {fmt(t['walkway_speed_median'])} | {fmt(t['speed_gain'])} | "
                     f"{t['strides']} | {fmt(t['walkway_stride_median_cm'], 1)} | {fmt(t['stride_gain'])} |")
    L += ["", "In-sim forward takes (takes.csv of the in-sim folder): decoded over true speed per take.", "",
          "| end | takes | true speed q10, q50, q90 | max | median ratio | median ratio in band (takes) | slope decoded on true |",
          "|---|---|---|---|---|---|---|"]
    for key, _ in ENDS:
        t = r["descriptive"][key]["insim_takes"]
        q = ", ".join(fmt(v) for v in t["true_speed_q10_q50_q90"])
        L.append(f"| {key} | {t['takes']} | {q} | {fmt(t['true_speed_max'])} | {fmt(t['median_ratio'])} | "
                 f"{fmt(t['median_ratio_in_band'])} ({t['takes_in_band']}) | {fmt(t['slope_decoded_on_true'])} |")
    L += ["", "Estimator check (not pre-registered): train54 gain as the in-sim definition, 1 + mean(decoded - "
          "walkway) / mean(walkway): " + "; ".join(
              f"{k} stride {fmt(r['descriptive'][k]['mean_based_gain_train54']['stride'])}, speed "
              f"{fmt(r['descriptive'][k]['mean_based_gain_train54']['speed'])}" for k, _ in ENDS) + ".", "",
          f"## Test X, leave-one-person-out cross-fit of the lever-1 head: **{r['X']['reading']}**", "",
          "| end | S54 in-sample (cm) | S54 LOPO | difference | V54 in-sample (m/s) | V54 LOPO | difference | LOPO stride gains | LOPO speed gains |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, x in r["X"]["ends"].items():
        L.append(f"| {key} | {fmt(x['S_in'], 2)} | {fmt(x['S_lopo'], 2)} | {fmt(x['S_lopo_minus_in'], 3)} | "
                 f"{fmt(x['V_in'], 4)} | {fmt(x['V_lopo'], 4)} | {fmt(x['V_lopo_minus_in'], 4)} | "
                 f"{iv(x['lopo_gain_range_stride'], 4)} | {iv(x['lopo_gain_range_speed'], 4)} |")
    L += ["", "Person-weighted head (median over persons of the person median ratio), dev12 lines (descriptive):", "",
          "| end | pooled g_S | person-weighted g_S | pooled g_V | person-weighted g_V | S12 pooled head | S12 person-weighted | V12 pooled head | V12 person-weighted |",
          "|---|---|---|---|---|---|---|---|---|"]
    with open(F1_RESULTS, encoding="utf-8") as f:
        f1 = json.load(f)
    for key, x in r["X"]["ends"].items():
        g = f1["ends"][key]["gain"]
        L.append(f"| {key} | {fmt(g['stride']['gain'], 4)} | {fmt(x['person_weighted_gain_stride'], 4)} | "
                 f"{fmt(g['speed']['gain'], 4)} | {fmt(x['person_weighted_gain_speed'], 4)} | "
                 f"{fmt(x['S12_pooled_head'], 2)} | {fmt(x['S12_person_weighted_head'], 2)} | "
                 f"{fmt(x['V12_pooled_head'], 3)} | {fmt(x['V12_person_weighted_head'], 3)} |")
    L += ["", f"## Test M, matched-stride selection: **{r['M']['reading']}**", "",
          "| end | task | walkway strides | unmatched | share |", "|---|---|---|---|---|"]
    for key, _ in ENDS:
        for task, t in r["M"][key].items():
            L.append(f"| {key} | {task} | {t['walkway_strides']} | {t['unmatched']} | {fmt(t['share_unmatched'])} |")
    L += ["", "## A, H4 share f on the pass-speed divisors (not pre-registered)", "",
          f"Source: {r['A']['source']}. f_speed = 1 - ln(D_L_speed) / ln(D_R_speed).", "",
          "| end, set | D_L_speed | D_R_speed | f_speed |", "|---|---|---|---|"]
    for k, v in r["A"]["f_speed"].items():
        L.append(f"| {k} | {fmt(v['D_L_speed'])} | {fmt(v['D_R_speed'])} | {fmt(v['f_speed'])} |")
    L += ["", "## Q, reproduction scan of the quoted numbers", "",
          "Each number with two or more decimals in a section, looked up among the numbers of the cited results.json "
          "(and F1's); 'missing' lists those with no value rounding to them. A missing token is checked by hand in "
          "the report.", "", "| section | pool | found | missing |", "|---|---|---|---|"]
    for k, v in r["Q"].items():
        L.append(f"| {k} | {v['pool']} | {v['found']} | {', '.join(v['missing']) or 'none'} |")
    L += ["", "## D, person ids in the F2 output files", "",
          f"Files with a person column: {sum(1 for x in r['D']['files'] if x['person_column'])}; persons outside "
          f"dev14 and train54: {r['D']['outside_total']}; lockbox persons: {r['D']['lockbox_total']} (asserted 0).", "",
          "| file | column | weargait persons | other ids (AMASS) | outside | lockbox |", "|---|---|---|---|---|---|"]
    for x in r["D"]["files"]:
        if x["person_column"]:
            L.append(f"| {x['file']} | {x['person_column']} | {x['persons']} | {x['other_ids']} | {x['outside']} | "
                     f"{x['lockbox']} |")
    L += ["", f"Run started {r['run_started']}, {r['seconds']} s."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--boot", type=int, default=BOOT_N)
    a = ap.parse_args(argv)
    from mdfm.gates import common
    r = run(a.boot)
    common.write_text(os.path.join(OUT, "results.json"), json.dumps(r, indent=1, default=float) + "\n")
    common.write_text(os.path.join(OUT, "summary.md"), summary(r))
    stamp = F1().stamp_moveatlas()
    prov = {"task": TASK, "what": LABEL, "run_started": r["run_started"], "moveatlas": stamp,
            "code_sha256": common.code_sha256([os.path.abspath(__file__), os.path.join(SCRIPTS, "g3_gain_check.py")]),
            "inputs": {rel(p): F1().sha256_file(p) for p in [F1_RESULTS, NOTE, DRAFT] + sorted(
                glob.glob(os.path.join(F1_DIR, "decode_*", "train_events.csv"))) + [
                os.path.join(F2_DIR, "H3angles", "results.json")]}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(f"R {r['R']['reading']}; X {r['X']['reading']}; M {r['M']['reading']}; {r['seconds']} s")


if __name__ == "__main__":
    main()
