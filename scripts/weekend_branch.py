"""The branch rule of the weekend of 2026-10-01, written and frozen before any phase B real-sensor dev number
existed (docs/handoffs/MDFM_HANDOFF_5.md section 1 holds the SHA-256 of this file and of CONSTANTS).

Usage:
  python scripts/weekend_branch.py --stage branch|seeds|v2check|reference|extension [--write] [--now ISO]
  python scripts/weekend_branch.py --hashes

The script reads report files only (json, csv, log.jsonl), imports no torch and no mdfm module, opens no record,
uses no GPU. --root (default this tree) is the base of every default path. --write saves
reports/gpu_queue/branch_<stage>_<YYYYmmdd_HHMMSS>.md and .json (the extension stage also writes
cache/runs/queue/EXTEND); without --write it prints only. Every output carries the SHA-256 of this file, the
SHA-256 of the canonical JSON of CONSTANTS (json.dumps(sort_keys=True, separators=(",", ":"),
ensure_ascii=True)) and whether both match the values frozen in handoff 5 section 1.

Definitions.
- S and V: the dev stride MAE (cm) and speed MAE (m/s) on the 12 controls without WHC003 and WHC026 (subset
  no_whc003_whc026); the values on all 14 (subset all14) are printed beside them. G1a: the six-IMU MPJPE (mm).
- Lines (lower is better): G1a pass 45 mm, kill 70; S pass 5 cm, kill 10; V pass 0.08 m/s, kill 0.15. Every row
  prints its numbers, its interval and its distance to both lines (value minus line: positive is above it).
- Intervals: percentile bootstrap, 95 percent, 10,000 resamples drawn in blocks of 1,000 from
  numpy.random.default_rng(20261001), a fresh generator per interval. G3: paired person bootstrap (resampled
  persons, the mean over persons of the per-person differences, on the persons both sides hold). G1: take-level
  (resampled takes, each take weighted by its knots when the input gives them); the point of a G1 effect is the
  same weighted mean over the paired takes, the top-level values of the input are printed beside it.
- An interval 'excludes 0' when its lower bound is above 0 or its upper bound below 0. Fewer than 2 paired units:
  not evaluated.
- A row whose input is missing, unreadable or lacks a key prints 'not evaluated' with the reason.

Inputs of --stage branch (defaults from the run folder --run, cache/runs/phaseA_root_full; <B> is the B end
name, <run folder name>_<checkpoint stem> of the last phase B verify event of <run>/log.jsonl, or --name; without
a phase B verify event (a crash), the checkpoint the queue runner evaluated in its place (cache/runs/queue/
state.json phase_b.fired, when that checkpoint lies in <run>), else <run folder name>_ckpt_B_032044; the R0
row prints which; <A> the same for the last phase A verify event, or --name-a):
- the run log <run>/log.jsonl and reports/gpu_queue/job_log.csv (R0);
- the G3 dev json of each end, g3_dev.json or g3_sim.json (both names are read), with its per_person.csv, in
  the first of reports/gates/g3/dev_<X>/, reports/gates/g3/curve_<X>/, reports/gates/g3_sim/dev_<X>/, then any
  other folder of reports/gates/g3 or g3_sim named after <X> whose json holds EMA weights (--g3-b, --g3-a: a
  folder or a json path);
- reports/gates/g1/dev_<B>/g1.json (--g1);
- reports/gates/g3_sim/insim_<B>/insim.json (--insim);
- diag.json of T04 (--diag-b, --diag-a, repeatable; rows merged, the first file holding a row wins):
  reports/gates/g3/diag_<X>/diag.json, then reports/gates/g3/diag_<X>_<suffix>/diag.json in sorted order (for
  instance the ablations);
- decompose.json of T05: reports/bench/g1_decompose_<B>/decompose.json (--decompose);
- the warp json of T07: reports/bench/root_motion_<B>_warp15.json, else root_motion_<B>__warp15.json (the
  name scripts/probe_root_motion.py gives to --tag _warp15) (--warp).

diag.json, as read here. rows: base, lowpass12, sites6, swap_whc026, refine0, refine60, model_legs,
family_lower_back, family_wrists, family_legs, either under a top-level 'rows' or at the top level. Each row has
the subsets all14, no_whc003 and no_whc003_whc026, each {stride_mae_cm, speed_mae_ms, stride_mape_pct,
n_persons, per_person {id: {stride_cm, speed_ms}}}; speed_mae_ms and speed_ms are in m/s. Also
selfpace_stride_mae_cm per subset, hurried_speed_bias_ms (m/s), spearman_speed_error and heel_strike_ic
{median_ms, offset_ms, margin_ratio}: looked up in the row first, then at the top level; a value given per
subset ({subset: value}) is read on no_whc003_whc026; a value given as a mapping with a point (point, value or
rho) is read at that key. The paired interval of M1 needs per-person SelfPace values:
selfpace_stride_mae_cm[subset] as {point, per_person {id: cm}}, or a selfpace_stride_cm entry in the per_person
records of the subset; a bare number gives M1 'not evaluated'.
When no diag.json holds the base row, S and V come from the per_person.csv beside the G3 json (rows dev,
predictor model, lines stride_mae_cm and speed_mae_m_s), the subsets formed by person id.
decompose.json: g1a_mm, true_root_orient_mm, true_betas_mm, refine0_mm, refine20_mm, refine60_mm, abs_mm,
best_of_16_mm, floor_mm, by_task, by_archive, by_group, lockbox_mix_mm, per_take. per_take is a list of
records, or a mapping take id -> record; a record holds the take id (take, take_id or id), values named like the
top-level keys (g1a_mm; also read without the unit, g1a, or in metres, g1a_m) and optionally its knots (knots,
n_knots, n or weight).
Warp json: rows with a factor (factor or warp key) in a list under rows, warp_rows or warps, or a mapping under
warp keyed by the factor; the in-sim stride error of a row is its first key among stride_mae_cm,
stride_error_cm, stride_mape_pct and stride_error that both the 1.0 and the 1.5 rows hold.

Rows of --stage branch (the frozen text is handoff 5 section 1):
- R0 integrity: verify event present, report_mismatches empty, no alarm, exit 0. The verify event is the last
  'verify' with phase B; report_mismatches of it and of every 'start' event are empty; no 'alarm' event in the
  log; the exit code is --exit-code, else that of the last job_log.csv row of a train_stage1.py job on the run
  folder (the job's --run-dir, else the folder of its --resume checkpoint; a last row without an exit code is
  not an exit 0), else 0 inferred from the verify event (train_stage1.py exits non-zero after it only on an
  alarm).
- R1 'clear of the kill lines': G1a at most 70 mm, S at most 10 cm, V at most 0.15 m/s. Never written 'pass'.
  The fix track continues toward the pass lines: 45 mm, 5 cm and 0.08 m/s. G1a from g1.json (its harness
  interval); S and V from the first G3 set of the B end (diag base row, else the G3 dev folder) with a person
  bootstrap of the mean, the harness's 14-person values of the G3 dev json printed beside.
- R-IC: model IC median (walkway reference), the shank-gyroscope detector, and the margin (model over detector
  on the ICs both match) against 0.8; the heel-strike row of diag.json beside. Reported only; the fix is the
  owner's (contact readout, OD12).
- R-insim: if the in-sim forward stride MAE of the evaluated end (insim.json lines.forward.model) is above 5 cm,
  print 'G3 pass unreachable by sim-to-real fixes alone'.
- R-AB: paired differences from the A end to the B end (A minus B, positive is a B improvement) in S and V, on
  the 12 and the 14. Verdict on the 12: 'phase B improved' when S or V improves with an interval excluding 0
  and neither worsens with an interval excluding 0; 'phase B did not improve' when neither improves with an
  interval excluding 0 and either worsens with an interval excluding 0 or both point differences are at most 0;
  'inconclusive' otherwise. When it reads 'phase B did not improve', the G3 v2 item may only be M2 or M6 (M1 is
  barred). When R-AB is not evaluated, whether M1 is allowed is unknown: a firing M1 then reads 'not evaluated',
  is not named and blocks R3.
- R2 retrain menu. v2 changes one thing. An item fires when the 95 percent interval of its effect excludes 0
  and its point effect is at least 30 percent of the line's gap to its pass line (the gap being the B end value
  minus the pass line; no gap, no firing).
  - M1 speed-and-cadence matched walking (G3). Effect: S minus S on SelfPace trials. Also requires a hurried
    speed bias of at most -0.10 m/s, a Spearman(true speed, signed error) of at most -0.5, and a warp json
    showing the in-sim stride error at warp 1.5 above warp 1.0.
  - M2 12 Hz low-pass of the real IMU input (G3). Effect: S(base) minus S(lowpass12).
  - M4 g1_six layout share from 0.10 to 0.25, the other ten layouts scaled down in proportion (G1). Effect: G1a
    minus true_root_orient_mm (gap: decompose g1a_mm minus 45 mm).
  - M6 stance no-slip on real windows (G3). Effect: S(model) minus S(model_legs) (model is the base row).
  - Not on the menu: M3, M5 and the non-finite guard.
  - v2 takes the single firing item with the largest effect as a fraction of its gap; ties go to G3 (then the
    menu order M1, M2, M6).
- R2a inference proposal (no retrain): refine60 or abs at least 10 mm better on G1a (against refine20 on the
  same takes), or refine60 or lowpass12 at least 20 percent better on S (against base), with an interval
  excluding 0. It goes to T21's table. Each candidate's value is printed with its distance to both lines.
- R3: every menu row evaluated and none fires, so there is no v2. A row with a missing input prints 'not
  evaluated' and blocks R3. An item that fires but is barred by R-AB counts as evaluated. When an item fires,
  the v2 item is named even if another row is not evaluated, and the decision lists the rows not evaluated.
- R0 does not block the other rows; a failed R0 is repeated in the decision line.

Other stages.
- seeds: |seed 1 minus seed 2| per line (S, V on the 12 and the 14; G1a) at ckpt_B_004000 and at the B end,
  runs --run (seed 1) and --run2 (default cache/runs/pilot_v1_s2). G1a from the G1 dev folders of EMA weights
  named after each checkpoint (a folder's variant is read from g1.json partial_run): at ckpt_B_004000 the
  --limit-takes 2 runs (else the full ones), at the B end the full runs (else the --limit-takes 2 ones), always
  the same variant for both seeds. The same pairing applies to every G1 comparison below.
- v2check: go requires all of: the targeted line (S for M1, M2 and M6; G1a for M4; from the latest
  branch_branch_*.json or --branch-json) on the 12 improves on v1's ckpt_B_004000 by at least max(10 percent,
  2 x the seed difference at that step), with a paired interval excluding 0; no other line (S, V, G1a) worse by
  more than max(5 percent, 2 x its seed difference); no alarm, no non-finite stop (3 'nonfinite' events at
  consecutive steps, the loop's MAX_NONFINITE) and peak reserved memory at most 14.0 GB (the largest
  memory_gb.max_reserved of the v2 log). Without seed 2's step-4,000 numbers it uses 10 and 5 percent and says
  so. The seed difference includes phase A variance, so it is conservative. v2: --v2-run (cache/runs/pilot_v2)
  at ckpt_B_004000.
- reference (a proposal for the owner): v1's phase B end stays unless a candidate (--candidates, default
  cache/runs/pilot_v1_s2 and cache/runs/pilot_v2, each at its phase B end) beats it on S by more than 2 x the
  seed difference, with a paired interval excluding 0, and without G1a worse by more than 2 x its seed
  difference (seed differences at the B end, from the latest branch_seeds_*.json; without it no candidate can
  beat v1). It writes reference_checkpoint (default v1's B end) and labels every selected number
  'post-selection, optimistic'. Among several candidates that beat v1, the largest S improvement. Seed 2 itself
  can never beat v1 under this test (its difference from v1 is the seed difference), so in practice only a v2
  run can replace the reference.
- extension: the candidate is reference_checkpoint of the latest branch_reference_*.json (else v1's B end).
  Extend only when both hold: the candidate's S at its end is below its value at 75 percent of phase B (the last
  phase B checkpoint at or before 0.75 x its phase B steps), with a paired interval excluding 0; and its G1
  --limit-takes curve still falls (the --limit-takes 2 G1a of its end below that of its latest earlier phase B
  checkpoint with one, G1 folders found by name as above). Steps = (Mon 2026-10-05 07:00 minus now minus 1.5 h)
  divided by the measured seconds per step (the run's last 2,000 phase B steps, spans holding a dev or
  checkpoint event left out; else
  reports/gpu_queue/baseline_s_per_step.json); none when under 4 h remain. With --write it writes
  cache/runs/queue/EXTEND (json: ext_ckpt, ext_steps) on yes, and removes an EXTEND this script wrote on no.

Exit codes: 0 rows printed (and written); 2 bad arguments.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDOFF = os.path.join("docs", "handoffs", "MDFM_HANDOFF_5.md")
FORMAT = "mdfm-weekend-branch/1"

CONSTANTS = {
    "version": "weekend_branch/1",
    "bootstrap": {"resamples": 10000, "block": 1000, "seed": 20261001, "level": 0.95, "min_units": 2},
    "subsets": {"twelve": "no_whc003_whc026", "fourteen": "all14", "thirteen": "no_whc003"},
    "excluded": {"no_whc003_whc026": ["WHC003", "WHC026"], "no_whc003": ["WHC003"], "all14": []},
    "lines": {"g1a_mm": {"pass": 45.0, "kill": 70.0},
              "stride_mae_cm": {"pass": 5.0, "kill": 10.0},
              "speed_mae_m_s": {"pass": 0.08, "kill": 0.15},
              "ic_median_ms": {"pass": 30.0, "kill": 50.0}},
    "ic_margin": 0.8,
    "insim_stride_mae_max_cm": 5.0,
    "menu": {
        "M1": {"gate": "G3", "line": "stride_mae_cm", "what": "speed-and-cadence matched walking",
               "effect": "S minus S on SelfPace trials", "hurried_speed_bias_max_m_s": -0.10,
               "spearman_max": -0.5, "warp_high": 1.5, "warp_low": 1.0},
        "M2": {"gate": "G3", "line": "stride_mae_cm", "what": "12 Hz low-pass of the real IMU input",
               "effect": "S(base) minus S(lowpass12)"},
        "M4": {"gate": "G1", "line": "g1a_mm", "what": "g1_six layout share from 0.10 to 0.25, the other ten "
                                                      "layouts scaled down in proportion",
               "effect": "G1a minus true_root_orient_mm"},
        "M6": {"gate": "G3", "line": "stride_mae_cm", "what": "stance no-slip on real windows",
               "effect": "S(model) minus S(model_legs)"}},
    "menu_order": ["M1", "M2", "M6", "M4"],
    "not_on_menu": ["M3", "M5", "non-finite guard"],
    "fire_fraction_of_gap": 0.30,
    "ab_did_not_improve_allows": ["M2", "M6"],
    "r2a": {"g1a_better_mm": 10.0, "stride_better_fraction": 0.20},
    "v2check": {"target_fraction": 0.10, "other_fraction": 0.05, "seed_multiple": 2.0,
                "peak_reserved_gb_max": 14.0, "nonfinite_stop_run": 3, "step_ckpt": "ckpt_B_004000"},
    "reference": {"seed_multiple": 2.0},
    "extension": {"deadline": "2026-10-05T07:00:00", "evaluation_reserve_h": 1.5, "min_hours": 4.0,
                  "fraction_of_phase_b": 0.75, "rate_window_steps": 2000},
    "phase_b_steps_default": 32044,
}

B = CONSTANTS["bootstrap"]
LINES = CONSTANTS["lines"]
SUB12, SUB14, SUB13 = (CONSTANTS["subsets"][k] for k in ("twelve", "fourteen", "thirteen"))
SUBSETS = (SUB14, SUB13, SUB12)
DIAG_ROWS = ("base", "lowpass12", "sites6", "swap_whc026", "refine0", "refine60", "model_legs",
             "family_lower_back", "family_wrists", "family_legs")
G3_LINES = {"S": "stride_mae_cm", "V": "speed_mae_m_s"}
UNITS = {"g1a_mm": "mm", "stride_mae_cm": "cm", "speed_mae_m_s": "m/s", "ic_median_ms": "ms"}
POST_SELECTION = "post-selection, optimistic"
DEFAULT_RUN = "cache/runs/phaseA_root_full"
DEFAULT_RUN2 = "cache/runs/pilot_v1_s2"
DEFAULT_V2_RUN = "cache/runs/pilot_v2"


# ---------------------------------------------------------------- hashes
def canonical_constants():
    return json.dumps(CONSTANTS, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def constants_sha256():
    return hashlib.sha256(canonical_constants().encode("ascii")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_check(root):
    """{script, constants, frozen_script, frozen_constants, match}: this file's hashes against the values handoff
    5 section 1 states as 'weekend_branch.py SHA-256: <hex>' and 'constants SHA-256: <hex>'."""
    out = {"script": file_sha256(os.path.abspath(__file__)), "constants": constants_sha256(),
           "frozen_script": None, "frozen_constants": None, "handoff": HANDOFF.replace("\\", "/")}
    path = os.path.join(root, HANDOFF)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m1 = re.search(r"weekend_branch\.py SHA-256: ([0-9a-f]{64})", text)
        m2 = re.search(r"constants SHA-256: ([0-9a-f]{64})", text)
        out["frozen_script"] = m1.group(1) if m1 else None
        out["frozen_constants"] = m2.group(1) if m2 else None
    if out["frozen_script"] is None or out["frozen_constants"] is None:
        out["match"] = None
        out["text"] = "frozen hashes: not found in the handoff"
    else:
        out["match"] = out["frozen_script"] == out["script"] and out["frozen_constants"] == out["constants"]
        out["text"] = ("frozen hashes: match" if out["match"] else
                       "frozen hashes: MISMATCH (this file or its constants differ from handoff 5 section 1)")
    return out


# ---------------------------------------------------------------- small helpers
def num(x):
    """A finite float, else None (bools, strings, inf and nan are not numbers here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def scalar(x):
    """num(x), or num of the point of a mapping (point, value or rho), for a diag value written with its interval."""
    if isinstance(x, dict):
        for k in ("point", "value", "rho"):
            if k in x:
                return num(x[k])
        return None
    return num(x)


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def nd_of(line):
    return {"g1a_mm": 1, "stride_mae_cm": 2, "speed_mae_m_s": 3, "ic_median_ms": 1}.get(line, 2)


def distances(value, line):
    """'pass P u (x - pass +d), kill K u (x - kill +d)' for a lower-is-better line."""
    lp, u, nd = LINES[line], UNITS[line], nd_of(line)
    if value is None:
        return f"pass {lp['pass']:g} {u}, kill {lp['kill']:g} {u}: n/a"
    return (f"pass {lp['pass']:g} {u} (x - pass {value - lp['pass']:+.{nd}f}), "
            f"kill {lp['kill']:g} {u} (x - kill {value - lp['kill']:+.{nd}f})")


ROOT = [TREE]          # the --root of the current run, for the paths printed in the rows


def rel(root, path):
    if path is None:
        return None
    try:
        r = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        r = path
    return r.replace("\\", "/")


def show(path):
    """A path as printed: relative to the root of the run, forward slashes."""
    return rel(ROOT[0], path)


def short_id(pid):
    """weargait:WHC003 -> WHC003; 'weargait:NLS901 (control)' -> NLS901."""
    s = str(pid).split(":")[-1].strip()
    return s.split(" ")[0] if s else s


def in_subset(pid, subset):
    return short_id(pid) not in CONSTANTS["excluded"][subset]


def read_json(path):
    """(doc, error): doc None with a reason when the file is missing or unreadable."""
    if path is None:
        return None, "no path"
    if not os.path.exists(path):
        return None, f"missing: {show(path)}"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, f"unreadable: {show(path)} ({e})"


def read_obj(path):
    """read_json for a file that must hold a json object: (dict or None, error)."""
    doc, err = read_json(path)
    if doc is not None and not isinstance(doc, dict):
        return None, f"not a json object: {show(path)}"
    return doc, err


def read_jsonl(path):
    """(records, unparsed lines) of a log.jsonl; ([], 0) when missing."""
    recs, bad = [], 0
    if path is None or not os.path.exists(path):
        return None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                bad += 1
    return recs, bad


def ckpt_stem(path):
    return os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]


def run_name(run_dir):
    return os.path.basename(os.path.normpath(str(run_dir).replace("\\", "/")))


# ---------------------------------------------------------------- bootstrap
def boot_interval(d, w=None):
    """95 percent percentile interval of the (weighted) mean of d under resampling of its units."""
    d = np.asarray(d, dtype=float)
    m = len(d)
    w = np.ones(m) if w is None else np.asarray(w, dtype=float)
    rng = np.random.default_rng(B["seed"])
    stats = []
    left = B["resamples"]
    while left > 0:
        k = min(B["block"], left)
        idx = rng.integers(0, m, size=(k, m))
        ww = w[idx]
        stats.append((d[idx] * ww).sum(axis=1) / ww.sum(axis=1))
        left -= k
    stats = np.concatenate(stats)
    a = (1.0 - B["level"]) / 2.0
    lo, hi = np.quantile(stats, [a, 1.0 - a])
    return [float(lo), float(hi)]


def excludes_0(iv):
    return bool(iv) and (iv[0] > 0 or iv[1] < 0)


def single(values):
    """{point, interval, n} of the mean of a {unit: value} mapping (person bootstrap)."""
    vals = [v for v in values.values() if v is not None]
    if len(vals) < B["min_units"]:
        return {"point": float(np.mean(vals)) if vals else None, "interval": None, "n": len(vals)}
    return {"point": float(np.mean(vals)), "interval": boot_interval(vals), "n": len(vals)}


def paired(a, b, wa=None, wb=None):
    """Paired comparison of two {unit: value} mappings: point and interval of the (weighted) mean of a - b over
    the units both hold a finite value for; weights (G1 takes) are the mean of both sides' knots."""
    ids = sorted(k for k in set(a) & set(b) if a[k] is not None and b[k] is not None)
    if len(ids) < B["min_units"]:
        return {"n": len(ids), "point": None, "interval": None, "excludes_0": False,
                "reason": f"{len(ids)} paired units (at least {B['min_units']} needed)"}
    d = np.array([a[k] - b[k] for k in ids])
    if wa is not None or wb is not None:
        w = np.array([0.5 * ((wa or {}).get(k, 1.0) + (wb or {}).get(k, (wa or {}).get(k, 1.0))) for k in ids])
    else:
        w = np.ones(len(ids))
    point = float((d * w).sum() / w.sum())
    iv = boot_interval(d, w)
    return {"n": len(ids), "point": point, "interval": iv, "excludes_0": excludes_0(iv), "reason": None}


# ---------------------------------------------------------------- G3 sets
def _pp_value(rec, key):
    if isinstance(rec, dict):
        return num(rec.get(key))
    return None


def g3set_from_diag_row(row, source):
    """{source, kind, subsets {subset: {S, V, S_point, V_point}}} of one diag.json row, or None."""
    if not isinstance(row, dict):
        return None
    subs = {}
    for s in SUBSETS:
        sub = row.get(s)
        if not isinstance(sub, dict):
            continue
        pp = sub.get("per_person") if isinstance(sub.get("per_person"), dict) else {}
        subs[s] = {"S": {k: _pp_value(v, "stride_cm") for k, v in pp.items()},
                   "V": {k: _pp_value(v, "speed_ms") for k, v in pp.items()},
                   "S_point": num(sub.get("stride_mae_cm")), "V_point": num(sub.get("speed_mae_ms")),
                   "n_persons": sub.get("n_persons")}
    return {"source": show(source), "kind": "diag", "subsets": subs} if subs else None


def find_g3_json(path):
    """The G3 dev json of a folder (g3_dev.json, then g3_sim.json) or the path itself when it is a file."""
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    for name in ("g3_dev.json", "g3_sim.json"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return p
    return None


def g3set_from_folder(path):
    """The model's per-person S and V of a G3 dev folder (per_person.csv beside g3_dev.json or g3_sim.json)."""
    js = find_g3_json(path)
    if js is None:
        return None
    csv_path = os.path.join(os.path.dirname(js), "per_person.csv")
    if not os.path.exists(csv_path):
        return None
    vals = {"S": {}, "V": {}}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("row") != "dev" or r.get("predictor") != "model":
                continue
            for key, line in G3_LINES.items():
                if r.get("line") == line:
                    try:
                        v = float(r.get("value"))
                    except (TypeError, ValueError):
                        v = None
                    vals[key][r.get("person_id")] = v if v is not None and math.isfinite(v) else None
    subs = {}
    for s in SUBSETS:
        S = {k: v for k, v in vals["S"].items() if in_subset(k, s)}
        V = {k: v for k, v in vals["V"].items() if in_subset(k, s)}
        sp = [v for v in S.values() if v is not None]
        vp = [v for v in V.values() if v is not None]
        subs[s] = {"S": S, "V": V, "S_point": float(np.mean(sp)) if sp else None,
                   "V_point": float(np.mean(vp)) if vp else None, "n_persons": len(sp)}
    return {"source": show(js), "kind": "g3", "subsets": subs, "json": js}


class Diag:
    """Rows of one or more diag.json files merged (the first file holding a row wins)."""

    def __init__(self, paths):
        self.paths, self.rows, self.tops, self.errors = [], {}, [], []
        for p in paths:
            doc, err = read_json(p)
            if not isinstance(doc, dict):
                if doc is not None or (err and not err.startswith("missing")):
                    self.errors.append(err or f"not a json object: {show(p)}")
                continue
            self.paths.append(p)
            rows = doc.get("rows") if isinstance(doc.get("rows"), dict) else doc
            for name in DIAG_ROWS:
                if name in rows and name not in self.rows and isinstance(rows[name], dict):
                    self.rows[name] = (rows[name], p)
            self.tops.append(doc)

    def g3set(self, row):
        if row not in self.rows:
            return None
        r, p = self.rows[row]
        return g3set_from_diag_row(r, p)

    def extra(self, row, key, subset=SUB12):
        """A diag value looked up in the row, then at the top level (where it may be keyed by row); a per-subset
        mapping is read on `subset`."""
        rowdict = self.rows[row][0] if row in self.rows else None
        for h in ([rowdict] if rowdict is not None else []) + self.tops:
            if not isinstance(h, dict) or key not in h:
                continue
            v = h[key]
            if h is not rowdict and isinstance(v, dict) and row in v:
                v = v[row]
            if isinstance(v, dict) and subset in v:
                v = v[subset]
            return v
        return None

    def selfpace(self, row="base", subset=SUB12):
        """(point, per-person {id: cm} or None) of the SelfPace stride MAE."""
        v = self.extra(row, "selfpace_stride_mae_cm", subset)
        point, pp = None, None
        if isinstance(v, dict):
            point = num(v.get("point", v.get("stride_mae_cm")))
            if isinstance(v.get("per_person"), dict):
                pp = {k: num(x) if not isinstance(x, dict) else num(x.get("stride_cm")) for k, x in
                      v["per_person"].items()}
        else:
            point = num(v)
        if pp is None and row in self.rows:
            sub = self.rows[row][0].get(subset)
            if isinstance(sub, dict) and isinstance(sub.get("per_person"), dict):
                cand = {k: _pp_value(x, "selfpace_stride_cm") for k, x in sub["per_person"].items()}
                if any(x is not None for x in cand.values()):
                    pp = cand
        if pp is not None:
            pp = {k: x for k, x in pp.items() if in_subset(k, subset)}
        return point, pp


def g3_sources(root, name, diag_paths=None, folder=None):
    """The G3 sets of a checkpoint name, in the order diag base row, then G3 dev folder."""
    diag = Diag(diag_paths if diag_paths is not None else default_diag_paths(root, name))
    out = []
    s = diag.g3set("base")
    if s is not None:
        out.append(s)
    folders = [folder] if folder else default_g3_folders(root, name)
    for f in folders:
        s = g3set_from_folder(f)
        if s is not None:
            out.append(s)
            break
    return out


def _named(base, name):
    """Folders of `base` named <prefix><name> or <prefix><name>_<suffix> (AppleDouble sidecars left out)."""
    found = set(glob.glob(os.path.join(base, f"*{name}"))) | set(glob.glob(os.path.join(base, f"*{name}_*")))
    return sorted(f for f in found if os.path.isdir(f) and not os.path.basename(f).startswith("._"))


def default_diag_paths(root, name):
    """diag_<name>/diag.json, then diag_<name>_<suffix>/diag.json (for instance the ablations), sorted."""
    base = os.path.join(root, "reports", "gates", "g3")
    first = os.path.join(base, f"diag_{name}")
    rest = [f for f in _named(base, name) if os.path.basename(f).startswith(f"diag_{name}_")]
    return [os.path.join(f, "diag.json") for f in [first] + rest]


def _ema(js):
    doc, _ = read_json(js)
    ck = (doc or {}).get("checkpoint") if isinstance(doc, dict) else None
    w = (ck or {}).get("weights") if isinstance(ck, dict) else None
    return w in (None, "ema")


def default_g3_folders(root, name):
    """G3 dev folders of a name: dev_<name>, curve_<name> (g3), dev_<name> (g3_sim), then any other folder of
    reports/gates/g3 or g3_sim named after it that holds a G3 dev json of EMA weights."""
    g3, g3s = os.path.join(root, "reports", "gates", "g3"), os.path.join(root, "reports", "gates", "g3_sim")
    first = [os.path.join(g3, f"dev_{name}"), os.path.join(g3, f"curve_{name}"), os.path.join(g3s, f"dev_{name}")]
    rest = [f for f in _named(g3, name) + _named(g3s, name) if f not in first]
    return [f for f in first + rest if find_g3_json(f) is not None and _ema(find_g3_json(f))]


def pick_pair(srcs_a, srcs_b):
    """The first source kind both sides hold (diag before g3 folder), so that a pair compares like with like."""
    for kind in ("diag", "g3"):
        a = next((s for s in srcs_a if s["kind"] == kind), None)
        b = next((s for s in srcs_b if s["kind"] == kind), None)
        if a is not None and b is not None:
            return a, b
    return None, None


def g3_line(gset, key, subset):
    """(per-person mapping, point) of S or V on a subset of a G3 set."""
    if gset is None or subset not in gset["subsets"]:
        return None, None
    sub = gset["subsets"][subset]
    return sub[key], sub[f"{key}_point"]


# ---------------------------------------------------------------- G1 sources
def g1_reads(root, name):
    """{variant: read} of the G1 dev folders of a name with EMA weights: dev_<name> first, then the other folders
    of reports/gates/g1 named after it; the variant is 'full', or 'lt<K>' when g1.json partial_run holds
    limit_takes K. The first folder of a variant wins."""
    base = os.path.join(root, "reports", "gates", "g1")
    first = os.path.join(base, f"dev_{name}")
    out = {}
    for f in [first] + [x for x in _named(base, name) if x != first]:
        js = os.path.join(f, "g1.json")
        if not os.path.exists(js):
            continue
        g = g1_read(f)
        if g is None or g["weights"] not in (None, "ema"):
            continue
        out.setdefault(g["variant"], g)
    return out


def g1_read(folder):
    """{point, interval, n_persons, variant, weights, takes {take: mm}, knots {take: n}, json} of a G1 dev
    folder."""
    doc, err = read_json(os.path.join(folder, "g1.json"))
    if not isinstance(doc, dict):
        return None
    line = None
    for x in doc.get("lines", []) or []:
        if isinstance(x, dict) and str(x.get("line", "")).startswith("G1a"):
            line = x
            break
    part = doc.get("partial_run")
    k = part.get("limit_takes") if isinstance(part, dict) else None
    inf = doc.get("inference") if isinstance(doc.get("inference"), dict) else {}
    out = {"json": os.path.join(folder, "g1.json"), "point": num(line.get("point")) if line else None,
           "interval": line.get("interval") if line else None, "n_persons": line.get("n_persons") if line else None,
           "variant": "full" if k is None else f"lt{k}", "weights": inf.get("weights"), "takes": {}, "knots": {}}
    wcsv = os.path.join(folder, "windows.csv")
    if os.path.exists(wcsv):
        acc = {}
        with open(wcsv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    v, n = float(r["g1a_m"]), float(r["n"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(v) and math.isfinite(n)) or n <= 0:
                    continue
                s = acc.setdefault(r.get("take_id"), [0.0, 0.0])
                s[0] += v * 1000.0 * n
                s[1] += n
        out["takes"] = {k: s[0] / s[1] for k, s in acc.items()}
        out["knots"] = {k: s[1] for k, s in acc.items()}
    return out


def g1_pair(root, name_a, name_b, variants=("full", "lt2")):
    """G1 reads of two names in the same variant, the first of `variants` both hold."""
    ra, rb = g1_reads(root, name_a), g1_reads(root, name_b)
    for v in variants:
        if v in ra and v in rb:
            return ra[v], rb[v], v
    return None, None, None


def per_take(doc):
    """{take: record} of decompose.json per_take (list or mapping)."""
    pt = doc.get("per_take") if isinstance(doc, dict) else None
    out = {}
    if isinstance(pt, dict):
        for k, rec in pt.items():
            if isinstance(rec, dict):
                out[str(k)] = rec
    elif isinstance(pt, list):
        for rec in pt:
            if not isinstance(rec, dict):
                continue
            k = rec.get("take", rec.get("take_id", rec.get("id")))
            if k is not None:
                out[str(k)] = rec
    return out


def take_weight(rec):
    for k in ("knots", "n_knots", "n", "weight"):
        w = num(rec.get(k))
        if w is not None and w > 0:
            return w
    return 1.0


def take_value(rec, key):
    """A per_take value in mm: the key itself (g1a_mm), else without its unit (g1a), else in metres (g1a_m)."""
    base = key[:-3] if key.endswith("_mm") else key
    for k, scale in ((key, 1.0), (base, 1.0), (base + "_m", 1000.0)):
        v = num(rec.get(k))
        if v is not None:
            return v * scale
    return None


def take_values(doc, key):
    """({take: value}, {take: knots}) of one decompose key over per_take."""
    vals, w = {}, {}
    for t, rec in per_take(doc).items():
        v = take_value(rec, key)
        if v is not None:
            vals[t] = v
            w[t] = take_weight(rec)
    return vals, w


# ---------------------------------------------------------------- run logs
def log_events(recs, event, phase=None):
    return [r for r in recs or [] if r.get("event") == event and (phase is None or r.get("phase") == phase)]


def last_verify(recs, phase):
    v = log_events(recs, "verify", phase)
    return v[-1] if v else None


def phase_b_steps(recs):
    for r in reversed(log_events(recs, "start")):
        st = r.get("steps")
        if isinstance(st, list) and len(st) >= 2 and num(st[1]):
            return int(st[1])
    return CONSTANTS["phase_b_steps_default"]


def b_end_verify(recs):
    """The phase B verify event that closes phase B (step at least the phase B steps), else None."""
    n = phase_b_steps(recs)
    for r in reversed(log_events(recs, "verify", "B")):
        if num(r.get("step")) is not None and r["step"] >= n:
            return r
    return None


def nonfinite_stop(recs, run=CONSTANTS["v2check"]["nonfinite_stop_run"]):
    """True when `run` nonfinite events sit at consecutive optimizer steps (the loop's FloatingPointError)."""
    steps = sorted({int(r["step"]) for r in log_events(recs, "nonfinite") if num(r.get("step")) is not None})
    best, cur, prev = 0, 0, None
    for s in steps:
        cur = cur + 1 if prev is not None and s == prev + 1 else 1
        best, prev = max(best, cur), s
    return best >= run


def peak_reserved_gb(recs):
    vals = []
    for r in recs or []:
        if r.get("event") in ("log", "end") and isinstance(r.get("memory_gb"), dict):
            v = num(r["memory_gb"].get("max_reserved"))
            if v is not None:
                vals.append(v)
    return max(vals) if vals else None


def _time(s):
    try:
        return datetime.datetime.fromisoformat(str(s))
    except ValueError:
        return None


def seconds_per_step(recs, phase="B", window=CONSTANTS["extension"]["rate_window_steps"]):
    """Seconds per optimizer step over the last `window` logged steps of a phase, spans between two 'log'
    events that hold a dev or checkpoint event left out; None when no clean span exists."""
    logs = [r for r in log_events(recs, "log", phase) if num(r.get("step")) is not None and _time(r.get("time"))]
    if len(logs) < 2:
        return None
    marks = [int(r["step"]) for r in recs or [] if r.get("event") in ("dev", "checkpoint") and r.get("phase") == phase
             and num(r.get("step")) is not None]
    last = int(logs[-1]["step"])
    secs = steps = 0.0
    for a, b in zip(logs[:-1], logs[1:]):
        sa, sb = int(a["step"]), int(b["step"])
        if sb <= sa or sa < last - window:
            continue
        if any(sa <= m < sb for m in marks):       # the loop logs step s, then checkpoints and evaluates at s
            continue
        secs += (_time(b["time"]) - _time(a["time"])).total_seconds()
        steps += sb - sa
    return secs / steps if steps > 0 else None


def argv_list(text):
    """The argv of a job_log.csv row (a JSON list, else split on blanks)."""
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return [str(x) for x in v]
    except (TypeError, ValueError):
        pass
    return str(text or "").split()


def job_run_folder(argv):
    """The run folder of a train_stage1.py argv: --run-dir, else the folder of --resume (train_stage1.run_folder);
    None when neither is given."""
    val = {}
    for i, x in enumerate(argv[:-1]):
        if x in ("--run-dir", "--resume"):
            val[x] = argv[i + 1]
    if "--run-dir" in val:
        return run_name(val["--run-dir"])
    if "--resume" in val:
        return run_name(os.path.dirname(val["--resume"].replace("\\", "/")))
    return None


def job_exit(root, job_log, run_dir):
    """(exit code, path) of the last job_log.csv row of a train_stage1.py job on run_dir; (None, path) without
    such a row; a last row without an exit code gives its state as text (not an exit 0)."""
    path = job_log if job_log else os.path.join(root, "reports", "gpu_queue", "job_log.csv")
    if not os.path.exists(path):
        return None, None
    key = run_name(run_dir)
    code = None
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            argv = argv_list(r.get("argv", ""))
            if not any("train_stage1" in x for x in argv) or job_run_folder(argv) != key:
                continue
            raw = next((r[k] for k in ("exit_code", "exit") if r.get(k) not in (None, "")), None)
            if raw is None:
                code = f"none ({r.get('state') or 'no exit code'})"
                continue
            try:
                code = int(float(raw))
            except ValueError:
                code = raw
    return code, path


# ---------------------------------------------------------------- rows
def row(rid, title, status, verdict, lines, **data):
    return {"id": rid, "title": title, "status": status, "verdict": verdict, "lines": lines, "data": data}


def not_evaluated(rid, title, reasons, lines=None, **data):
    return row(rid, title, "not evaluated", "not evaluated: " + "; ".join(reasons), lines or [], **data)


def r0_integrity(root, run_dir, recs, bad, job_log=None, exit_code=None):
    title = "R0 integrity: verify event present, report_mismatches empty, no alarm, exit 0"
    if recs is None:
        return not_evaluated("R0", title, [f"no run log at {rel(root, os.path.join(run_dir, 'log.jsonl'))}"])
    v = last_verify(recs, "B")
    starts = log_events(recs, "start")
    alarms = log_events(recs, "alarm")
    nonfin = log_events(recs, "nonfinite")
    mism = (v or {}).get("report_mismatches")
    start_mism = [r.get("report_mismatches") for r in starts if r.get("report_mismatches")]
    if exit_code is not None:
        code, how = int(exit_code), "--exit-code"
    else:
        code, jpath = job_exit(root, job_log, run_dir)
        how = f"job log {rel(root, jpath)}" if code is not None else None
        if code is None and v is not None and not alarms:
            code, how = 0, "inferred: train_stage1.py exits non-zero after its verify event only on an alarm"
    checks = {"verify_present": v is not None, "report_mismatches_empty": v is not None and mism == [] and not start_mism,
              "no_alarm": not alarms, "exit_0": code == 0}
    ok = all(checks.values())
    lines = [f"verify (phase B): {'step ' + str(v.get('step')) + ', ' + str(v.get('path')).replace(chr(92), '/') + ', sha256 ' + str(v.get('sha256')) if v else 'absent'}",
             f"report_mismatches: verify {mism if v else 'n/a'}, start events {start_mism or '[]'}",
             f"alarm events: {len(alarms)}; nonfinite (skipped) steps: {len(nonfin)}; unparsed log lines: {bad}",
             f"exit: {code if code is not None else 'unknown'} ({how or 'no job log row, no verify event'})"]
    return row("R0", title, "evaluated", "R0 holds" if ok else
               "R0 fails: " + ", ".join(k for k, x in checks.items() if not x), lines, checks=checks,
               verify=v, exit_code=code)


def line_text(label, line, point, interval, beside=None):
    nd = nd_of(line)
    t = f"{label} {fmt(point, nd)} {UNITS[line]} {fmt_iv(interval, nd)}"
    if beside:
        t += f" ({beside})"
    return t + "; " + distances(point, line)


def r1_kill_lines(g1, gB, g3_json=None):
    """G1a from g1.json; S and V from the first G3 set of the B end (diag base row, else the G3 dev folder), with
    the harness's 14-person values of the G3 dev json printed beside as a cross-check."""
    title = "R1 'clear of the kill lines': G1a at most 70 mm, S at most 10 cm, V at most 0.15 m/s"
    reasons, lines, vals = [], [], {}
    hdoc, _ = read_json(g3_json) if g3_json else (None, None)
    harness = {}
    for key, line in G3_LINES.items():
        try:
            harness[key] = num(hdoc["lines"]["dev"]["model"][line]["point"])
        except (KeyError, TypeError):
            harness[key] = None
    if g1 is None or g1.get("point") is None:
        reasons.append("G1a: no g1.json of the B end")
    else:
        vals["g1a_mm"] = g1["point"]
        lines.append(line_text("G1a", "g1a_mm", g1["point"], g1.get("interval"),
                               f"harness interval, {g1.get('n_persons')} persons, {show(g1['json'])}"))
    for key in ("S", "V"):
        line = G3_LINES[key]
        pp12, p12 = g3_line(gB, key, SUB12)
        pp14, p14 = g3_line(gB, key, SUB14)
        if pp12 is None or p12 is None:
            reasons.append(f"{key}: no G3 values of the B end on the 12")
            continue
        st = single(pp12)
        vals[line] = p12
        lines.append(line_text(f"{key} (12)", line, p12, st["interval"],
                               f"14: {fmt(p14, nd_of(line))}; {gB['source']}; harness 14: "
                               f"{fmt(harness[key], nd_of(line))}"))
    clear = [k for k, v in vals.items() if v <= LINES[k]["kill"]]
    above = [k for k, v in vals.items() if v > LINES[k]["kill"]]
    lines.append("the fix track continues toward the pass lines: 45 mm, 5 cm and 0.08 m/s")
    if reasons:
        return not_evaluated("R1", title, reasons, lines, values=vals, above_kill=above)
    verdict = ("clear of the kill lines" if not above else
               "not clear of the kill lines: above " + ", ".join(above))
    return row("R1", title, "evaluated", verdict, lines, values=vals, clear=clear, above_kill=above)


def r_ic(g3json_path, diag):
    title = "R-IC: model IC median, detector, and margin against 0.8 (reported only; the fix is the owner's: contact readout, OD12)"
    doc, err = read_obj(g3json_path)
    if doc is None:
        return not_evaluated("R-IC", title, ["no G3 dev json of the B end" if g3json_path is None else err])
    model = det = mm = md = None
    for r in doc.get("results_dev", []) or []:
        if isinstance(r, dict) and r.get("line") == "ic_median_ms_walkway":
            model = num(r.get("point"))
            ex = r.get("extra") or {}
            mm, md = num(ex.get("margin_model_ms")), num(ex.get("margin_detector_ms"))
    try:
        det_line = doc["lines"]["dev"]["shank_gyro"]["ic_median_ms_walkway"]
        det = num(det_line.get("point"))
    except (KeyError, TypeError, AttributeError):
        det_line = None
    if model is None:
        try:
            model = num(doc["lines"]["dev"]["model"]["ic_median_ms_walkway"]["point"])
        except (KeyError, TypeError):
            model = None
    ratio = mm / md if mm is not None and md not in (None, 0.0) else None
    hs = diag.extra("base", "heel_strike_ic") if diag is not None else None
    hs = hs if isinstance(hs, dict) else {}
    margin = CONSTANTS["ic_margin"]
    lines = [line_text("model IC median (walkway)", "ic_median_ms", model, None, show(g3json_path)),
             f"shank-gyroscope detector: {fmt(det, 1)} ms" + ("" if det is not None else
                                                              " (not finite: a person with most ICs unmatched)"),
             f"margin on the ICs both match: model {fmt(mm, 2)} ms, detector {fmt(md, 2)} ms, ratio {fmt(ratio, 3)} "
             f"against {margin} ({'met' if ratio is not None and ratio <= margin else 'not met' if ratio is not None else 'n/a'})",
             f"heel-strike IC (diag base): median {fmt(num(hs.get('median_ms')), 1)} ms, offset "
             f"{fmt(num(hs.get('offset_ms')), 1)} ms, margin ratio {fmt(num(hs.get('margin_ratio')), 3)}"]
    if model is None:
        return not_evaluated("R-IC", title, ["no model IC line in the G3 json"], lines)
    verdict = "reported only: margin " + ("met" if ratio is not None and ratio <= margin else
                                          "not met" if ratio is not None else "not computable")
    return row("R-IC", title, "evaluated", verdict, lines, model_ms=model, detector_ms=det, margin_ratio=ratio,
               heel_strike=hs)


def r_insim(insim_path):
    title = "R-insim: in-sim forward stride MAE of the evaluated end against 5 cm"
    doc, err = read_json(insim_path)
    if doc is None:
        return not_evaluated("R-insim", title, [err or "no insim.json"])
    try:
        x = doc["lines"]["forward"]["model"]["stride_mae_cm"]
        v, iv = num(x.get("point")), x.get("interval")
    except (KeyError, TypeError, AttributeError):
        v, iv = None, None
    if v is None:
        return not_evaluated("R-insim", title, ["no lines.forward.model.stride_mae_cm in insim.json"])
    lines = [line_text("in-sim forward stride MAE", "stride_mae_cm", v, iv, show(insim_path))]
    lim = CONSTANTS["insim_stride_mae_max_cm"]
    if v > lim:
        lines.append("G3 pass unreachable by sim-to-real fixes alone")
        verdict = "G3 pass unreachable by sim-to-real fixes alone"
    else:
        verdict = f"in-sim forward stride MAE at most {lim:g} cm"
    return row("R-insim", title, "evaluated", verdict, lines, value=v, unreachable=v > lim)


def r_ab(srcA, srcB):
    """srcA, srcB: the G3 sets of each end (g3_sources); the pair is of one kind (pick_pair)."""
    title = "R-AB: paired differences from the A end to the B end in S and V, on the 12 and the 14"
    gA, gB = pick_pair(srcA, srcB)
    if gA is None or gB is None:
        miss = [e for e, s in (("A end", srcA), ("B end", srcB)) if not s]
        return not_evaluated("R-AB", title, ["no G3 values of the " + " and the ".join(miss) if miss else
                                             "no source of the same kind (diag or G3 dev folder) at both ends"])
    res, lines = {}, []
    for sub in (SUB12, SUB14):
        for key in ("S", "V"):
            line = G3_LINES[key]
            a, pa = g3_line(gA, key, sub)
            b, pb = g3_line(gB, key, sub)
            p = paired(a or {}, b or {})
            res[(key, sub)] = p
            nd = nd_of(line)
            lines.append(f"{key} {'12' if sub == SUB12 else '14'}: A {fmt(pa, nd)} -> B {fmt(pb, nd)} {UNITS[line]}; "
                         f"A minus B {fmt(p['point'], nd)} {fmt_iv(p['interval'], nd)}, n {p['n']}"
                         + (f" ({p['reason']})" if p["reason"] else "") + "; B: " + distances(pb, line))
    lines.append(f"sources: A {gA['source']}; B {gB['source']}")
    s, v = res[("S", SUB12)], res[("V", SUB12)]
    if s["point"] is None or v["point"] is None:
        return not_evaluated("R-AB", title, ["fewer than 2 paired persons on the 12"], lines)
    imp = [k for k, p in (("S", s), ("V", v)) if p["interval"][0] > 0]
    wor = [k for k, p in (("S", s), ("V", v)) if p["interval"][1] < 0]
    if imp and not wor:
        verdict = "phase B improved"
    elif not imp and (wor or (s["point"] <= 0 and v["point"] <= 0)):
        verdict = "phase B did not improve"
    else:
        verdict = "inconclusive"
    if verdict == "phase B did not improve":
        lines.append("the G3 v2 item may only be M2 or M6")
    data = {f"{k}_{'12' if sub == SUB12 else '14'}": p for (k, sub), p in res.items()}
    return row("R-AB", title, "evaluated", verdict, lines, improved=imp, worsened=wor, **data)


def warp_rows(doc):
    """{factor: row} of a warp json (module docstring)."""
    out = {}
    if not isinstance(doc, dict):
        return out
    for key in ("rows", "warp_rows", "warps"):
        if isinstance(doc.get(key), list):
            for r in doc[key]:
                if isinstance(r, dict):
                    f = num(r.get("factor", r.get("warp")))
                    if f is not None:
                        out[round(f, 6)] = r
            if out:
                return out
    w = doc.get("warp")
    if isinstance(w, dict):
        for k, r in w.items():
            try:
                f = float(k)
            except ValueError:
                continue
            if isinstance(r, dict):
                out[round(f, 6)] = r
    return out


def warp_check(path):
    """(holds, text, reason): the in-sim stride error at warp 1.5 above warp 1.0."""
    m1 = CONSTANTS["menu"]["M1"]
    doc, err = read_json(path)
    if doc is None:
        return None, None, err or "no warp json"
    rows = warp_rows(doc)
    hi, lo = rows.get(round(m1["warp_high"], 6)), rows.get(round(m1["warp_low"], 6))
    if hi is None or lo is None:
        return None, None, f"the warp json has no row at {m1['warp_low']} and {m1['warp_high']} ({show(path)})"
    for key in ("stride_mae_cm", "stride_error_cm", "stride_mape_pct", "stride_error"):
        a, b = num(hi.get(key)), num(lo.get(key))
        if a is not None and b is not None:
            return a > b, f"in-sim {key} at warp 1.5 {a:.3f} against {b:.3f} at 1.0", None
    return None, None, f"the warp rows carry no stride error ({show(path)})"


def menu_item(item, effect, gap_value, extra_ok=True, extra_lines=(), extra_reasons=(), line=None):
    """The firing test of one menu item (CONSTANTS fire_fraction_of_gap)."""
    spec = CONSTANTS["menu"][item]
    line = line or spec["line"]
    lp = LINES[line]["pass"]
    nd = nd_of(line)
    title = f"{item} {spec['what']} ({spec['gate']}). Effect: {spec['effect']}"
    reasons = list(extra_reasons)
    if effect is None or effect.get("point") is None:
        why = (effect or {}).get("reason")
        if why or not reasons:
            reasons.append(why or "no effect")
    if gap_value is None and not reasons:
        reasons.append("no B end value of the line")
    lines = []
    if effect is not None:
        lines.append(f"effect {fmt(effect.get('point'), nd)} {UNITS[line]} {fmt_iv(effect.get('interval'), nd)}, "
                     f"n {effect.get('n')}")
    if gap_value is not None:
        lines.append(f"line {fmt(gap_value, nd)} {UNITS[line]}; gap to pass {fmt(gap_value - lp, nd)}; "
                     + distances(gap_value, line))
    lines.extend(extra_lines)
    if reasons:
        return not_evaluated(item, title, reasons, lines, item=item, fires=False, fraction=None)
    gap = gap_value - lp
    frac = effect["point"] / gap if gap > 0 else None
    need = CONSTANTS["fire_fraction_of_gap"]
    fires = bool(gap > 0 and effect["excludes_0"] and effect["point"] >= need * gap and extra_ok)
    lines.append(f"fraction of gap {fmt(frac, 3)} (needs at least {need}); interval excludes 0: "
                 f"{'yes' if effect['excludes_0'] else 'no'}" + ("" if extra_ok else "; extra conditions not met"))
    verdict = "fires" if fires else ("does not fire: line at or below its pass line" if gap <= 0 else "does not fire")
    return row(item, title, "evaluated", verdict, lines, item=item, fires=fires, fraction=frac,
               effect=effect, gap=gap, line=line)


def menu_m1(diag, gB_diag, warp_path):
    m1 = CONSTANTS["menu"]["M1"]
    reasons, extra = [], []
    s12, sp = g3_line(gB_diag, "S", SUB12)
    point_sp, pp_sp = (diag.selfpace("base", SUB12) if diag is not None else (None, None))
    if s12 is None:
        reasons.append("no diag base row on the 12")
    if pp_sp is None:
        reasons.append("no per-person SelfPace stride values in diag.json")
    effect = paired(s12, pp_sp) if s12 is not None and pp_sp is not None else None
    bias = scalar(diag.extra("base", "hurried_speed_bias_ms")) if diag is not None else None
    rho = scalar(diag.extra("base", "spearman_speed_error")) if diag is not None else None
    if bias is None:
        reasons.append("no hurried_speed_bias_ms")
    if rho is None:
        reasons.append("no spearman_speed_error")
    w_ok, w_text, w_reason = warp_check(warp_path)
    if w_ok is None:
        reasons.append(w_reason)
    ok_bias = bias is not None and bias <= m1["hurried_speed_bias_max_m_s"]
    ok_rho = rho is not None and rho <= m1["spearman_max"]
    extra.append(f"S on SelfPace (12): {fmt(point_sp, 2)} cm; hurried speed bias {fmt(bias, 3)} m/s (at most "
                 f"{m1['hurried_speed_bias_max_m_s']}: {'yes' if ok_bias else 'no'}); Spearman(true speed, signed error) "
                 f"{fmt(rho, 3)} (at most {m1['spearman_max']}: {'yes' if ok_rho else 'no'})")
    extra.append("warp: " + (w_text + f" ({'above' if w_ok else 'not above'})" if w_text else "n/a"))
    return menu_item("M1", effect, sp, extra_ok=bool(ok_bias and ok_rho and w_ok), extra_lines=extra,
                     extra_reasons=reasons)


def menu_diag_pair(item, diag, row_a, row_b):
    a = diag.g3set(row_a) if diag is not None else None
    b = diag.g3set(row_b) if diag is not None else None
    reasons = [f"no diag row {r}" for r, s in ((row_a, a), (row_b, b)) if s is None]
    if reasons and diag is not None:
        reasons += diag.errors
    sa, pa = g3_line(a, "S", SUB12)
    sb, pb = g3_line(b, "S", SUB12)
    effect = paired(sa, sb) if sa is not None and sb is not None else None
    extra = [f"S(12) {row_a} {fmt(pa, 2)} cm, {row_b} {fmt(pb, 2)} cm; 14: {row_a} "
             f"{fmt(g3_line(a, 'S', SUB14)[1], 2)}, {row_b} {fmt(g3_line(b, 'S', SUB14)[1], 2)}"]
    return menu_item(item, effect, pa, extra_lines=extra, extra_reasons=reasons)


def menu_m4(dec, dec_path, g1, err=None):
    reasons = []
    if dec is None:
        reasons.append(f"no decompose.json ({dec_path})" if err is None or err.startswith("missing") else err)
        return menu_item("M4", None, None, extra_reasons=reasons)
    a, wa = take_values(dec, "g1a_mm")
    b, wb = take_values(dec, "true_root_orient_mm")
    effect = paired(a, b, wa, wb) if a and b else None
    if not a or not b:
        reasons.append("per_take lacks g1a_mm or true_root_orient_mm")
    top = num(dec.get("g1a_mm"))
    tro = num(dec.get("true_root_orient_mm"))
    if top is None:
        reasons.append("no g1a_mm in decompose.json")
    extra = [f"top level: g1a_mm {fmt(top, 1)}, true_root_orient_mm {fmt(tro, 1)}, difference "
             f"{fmt(None if top is None or tro is None else top - tro, 1)} mm; g1.json G1a "
             f"{fmt((g1 or {}).get('point'), 1)} mm"]
    return menu_item("M4", effect, top, extra_lines=extra, extra_reasons=reasons)


def r2a(diag, dec):
    title = "R2a inference proposal (no retrain): refine60 or abs at least 10 mm better on G1a, or refine60 or lowpass12 at least 20 percent better on S, with an interval excluding 0 (to T21's table)"
    lines, props, missing, done = [], [], [], 0
    c = CONSTANTS["r2a"]
    for key in ("refine60_mm", "abs_mm"):
        if dec is None:
            missing.append(f"G1a {key}: no decompose.json")
            continue
        a, wa = take_values(dec, "refine20_mm")
        b, wb = take_values(dec, key)
        p = paired(a, b, wa, wb)
        if p["point"] is None:
            missing.append(f"G1a {key}: {p['reason']} with refine20_mm")
            continue
        ok = p["point"] >= c["g1a_better_mm"] and p["interval"][0] > 0
        done += 1
        lines.append(f"G1a refine20 minus {key[:-3]}: {fmt(p['point'], 1)} mm {fmt_iv(p['interval'], 1)}, n {p['n']} "
                     f"takes (at least {c['g1a_better_mm']:g} mm: {'yes' if ok else 'no'}); top level refine20 "
                     f"{fmt(num(dec.get('refine20_mm')), 1)}, {key[:-3]} {fmt(num(dec.get(key)), 1)}; {key[:-3]}: "
                     + distances(num(dec.get(key)), "g1a_mm"))
        if ok:
            props.append(key[:-3] + " (G1a)")
    base = diag.g3set("base") if diag is not None else None
    for r in ("refine60", "lowpass12"):
        other = diag.g3set(r) if diag is not None else None
        sa, pa = g3_line(base, "S", SUB12)
        sb, pb = g3_line(other, "S", SUB12)
        if sa is None or sb is None or pa is None:
            missing.append(f"S {r}: no diag rows base and {r}")
            continue
        p = paired(sa, sb)
        if p["point"] is None:
            missing.append(f"S {r}: {p['reason']}")
            continue
        frac = p["point"] / pa if pa > 0 else None
        ok = frac is not None and frac >= c["stride_better_fraction"] and p["interval"][0] > 0
        done += 1
        lines.append(f"S(12) base minus {r}: {fmt(p['point'], 2)} cm {fmt_iv(p['interval'], 2)}, n {p['n']}, "
                     f"{fmt(None if frac is None else 100 * frac, 1)} percent of S {fmt(pa, 2)} cm (at least "
                     f"{100 * c['stride_better_fraction']:g} percent: {'yes' if ok else 'no'}); {r} S(12) "
                     f"{fmt(pb, 2)} cm: " + distances(pb, "stride_mae_cm"))
        if ok:
            props.append(r + " (S)")
    for m in missing:
        lines.append("not evaluated: " + m)
    not_eval = ", ".join(m.split(":")[0] for m in missing)
    if props:
        verdict = "proposal for T21's table: " + ", ".join(props)
    elif missing and not done:
        verdict = "not evaluated: " + not_eval
    elif missing:
        verdict = "no proposal among the evaluated candidates; not evaluated: " + not_eval
    else:
        verdict = "no proposal"
    status = "evaluated" if not missing else ("partly evaluated" if done else "not evaluated")
    return row("R2a", title, status, verdict, lines, proposals=props, missing=missing)


def select_v2(items, ab_verdict):
    """(v2 item or None, R3 text) from the menu rows (module docstring)."""
    allowed = CONSTANTS["ab_did_not_improve_allows"]
    firing, barred = [], []
    for it in items:
        if it["data"].get("fires"):
            restricted = CONSTANTS["menu"][it["id"]]["gate"] == "G3" and it["id"] not in allowed
            if restricted and ab_verdict == "phase B did not improve":
                barred.append(it["id"])
                it["verdict"] = "fires, barred by R-AB (the G3 v2 item may only be M2 or M6)"
            elif restricted and str(ab_verdict).startswith("not evaluated"):
                it["status"] = "not evaluated"
                it["verdict"] = ("not evaluated: fires, but R-AB is not evaluated, so whether the G3 v2 item may be "
                                 f"{it['id']} (and not only M2 or M6) is unknown")
            else:
                firing.append(it)
    missing = [it["id"] for it in items if it["status"] != "evaluated"]
    order = CONSTANTS["menu_order"]
    if firing:
        best = sorted(firing, key=lambda it: (-it["data"]["fraction"],
                                              0 if CONSTANTS["menu"][it["id"]]["gate"] == "G3" else 1,
                                              order.index(it["id"])))[0]
        r3 = "R3 does not apply: an item fires"
        if missing:
            r3 += " (not evaluated: " + ", ".join(missing) + ")"
        return best["id"], r3, missing, barred
    if missing:
        return None, "R3 blocked: not evaluated " + ", ".join(missing) + "; no v2 item is named", missing, barred
    return None, "R3: every menu row evaluated and none fires" + (
        " (firing items barred by R-AB: " + ", ".join(barred) + ")" if barred else "") + ", so there is no v2", \
        missing, barred


# ---------------------------------------------------------------- stages
class Ctx:
    def __init__(self, a):
        self.a = a
        self.root = os.path.abspath(a.root)
        ROOT[0] = self.root
        self.now = datetime.datetime.fromisoformat(a.now) if a.now else datetime.datetime.now()
        self.inputs = {}

    def p(self, path):
        """A path relative to the root (absolute paths pass)."""
        return path if path is None or os.path.isabs(path) else os.path.join(self.root, path)

    def note(self, name, path):
        exists = path is not None and os.path.exists(path)
        self.inputs[name] = {"path": rel(self.root, path), "exists": exists,
                             "sha256": file_sha256(path) if exists and os.path.isfile(path) else None}
        return path


def run_log(ctx, run_dir):
    path = os.path.join(ctx.p(run_dir), "log.jsonl")
    ctx.note(f"log {run_name(run_dir)}", path)
    return read_jsonl(path)


def end_name(recs, run_dir, phase="B", fallback=None):
    v = b_end_verify(recs) if phase == "B" else last_verify(recs, "A")
    if v is None and phase == "B":
        v = last_verify(recs, "B")
    if v is None:
        return fallback, None
    return f"{run_name(run_dir)}_{ckpt_stem(v.get('path'))}", v


def crash_end(root, run_dir):
    """(name, how) of the B end of a run without a phase B verify event: the checkpoint the queue runner evaluated
    in its place (cache/runs/queue/state.json phase_b.fired) when it lies in run_dir; (None, None) otherwise."""
    rn = run_name(run_dir)
    doc, _ = read_obj(os.path.join(root, "cache", "runs", "queue", "state.json"))
    pb = (doc or {}).get("phase_b")
    fired = pb.get("fired") if isinstance(pb, dict) else None
    if isinstance(fired, dict) and fired.get("ckpt"):
        ck = str(fired["ckpt"]).replace("\\", "/")
        if run_name(os.path.dirname(ck)) == rn:
            return (f"{rn}_{ckpt_stem(ck)}", f"no phase B verify event: the checkpoint the queue runner evaluated, "
                                             f"{ck} (cache/runs/queue/state.json)")
    return None, None


def stage_branch(ctx):
    a = ctx.a
    recs, bad = run_log(ctx, a.run)
    nameB, vB = end_name(recs or [], a.run, "B", None)
    if a.name:
        nameB, howB = a.name, "--name"
    elif nameB:
        howB = f"phase B verify event at step {vB.get('step')}"
    else:
        nameB, howB = crash_end(ctx.root, a.run)
        if nameB is None:
            nameB = f"{run_name(a.run)}_ckpt_B_{CONSTANTS['phase_b_steps_default']:06d}"
            howB = "default name: no phase B verify event and no checkpoint fired by the queue runner"
    nameA, _ = end_name(recs or [], a.run, "A", None)
    nameA = a.name_a or nameA or f"{run_name(a.run)}_ckpt_A_010681"
    rows = [r0_integrity(ctx.root, ctx.p(a.run), recs, bad, ctx.p(a.job_log) if a.job_log else None, a.exit_code)]
    rows[0]["lines"].append(f"B end evaluated: {nameB} ({howB}); A end: {nameA}")
    g1_path = ctx.p(a.g1) if a.g1 else os.path.join(ctx.root, "reports", "gates", "g1", f"dev_{nameB}", "g1.json")
    if os.path.isdir(g1_path):
        g1_path = os.path.join(g1_path, "g1.json")
    ctx.note("g1 B", g1_path)
    g1 = g1_read(os.path.dirname(g1_path)) if os.path.exists(g1_path) else None
    diagB_paths = [ctx.p(x) for x in a.diag_b] if a.diag_b else default_diag_paths(ctx.root, nameB)
    diagA_paths = [ctx.p(x) for x in a.diag_a] if a.diag_a else default_diag_paths(ctx.root, nameA)
    for i, x in enumerate(diagB_paths):
        ctx.note(f"diag B {i}", x)
    for i, x in enumerate(diagA_paths):
        ctx.note(f"diag A {i}", x)
    diagB = Diag(diagB_paths)
    g3B_folder = ctx.p(a.g3_b) if a.g3_b else None
    g3A_folder = ctx.p(a.g3_a) if a.g3_a else None
    srcB = g3_sources(ctx.root, nameB, diagB_paths, g3B_folder)
    srcA = g3_sources(ctx.root, nameA, diagA_paths, g3A_folder)
    g3B_json = find_g3_json(g3B_folder) if g3B_folder else next(
        (find_g3_json(f) for f in default_g3_folders(ctx.root, nameB) if find_g3_json(f)), None)
    ctx.note("g3 B json", g3B_json)
    g3A_json = find_g3_json(g3A_folder) if g3A_folder else next(
        (find_g3_json(f) for f in default_g3_folders(ctx.root, nameA) if find_g3_json(f)), None)
    ctx.note("g3 A json", g3A_json)
    gB = srcB[0] if srcB else None
    rows.append(r1_kill_lines(g1, gB, g3B_json))
    rows.append(r_ic(g3B_json, diagB))
    insim = ctx.p(a.insim) if a.insim else os.path.join(ctx.root, "reports", "gates", "g3_sim", f"insim_{nameB}",
                                                         "insim.json")
    if os.path.isdir(insim):
        insim = os.path.join(insim, "insim.json")
    ctx.note("insim B", insim)
    rows.append(r_insim(insim))
    rows.append(r_ab(srcA, srcB))
    ab = rows[-1]["verdict"]
    dec_path = ctx.p(a.decompose) if a.decompose else os.path.join(ctx.root, "reports", "bench",
                                                                   f"g1_decompose_{nameB}", "decompose.json")
    ctx.note("decompose B", dec_path)
    dec, dec_err = read_obj(dec_path)
    if a.warp:
        warp = ctx.p(a.warp)
    else:
        names = [os.path.join(ctx.root, "reports", "bench", f"root_motion_{nameB}{t}.json")
                 for t in ("_warp15", "__warp15")]
        warp = next((p for p in names if os.path.exists(p)), names[0])
    ctx.note("warp B", warp)
    gB_diag = diagB.g3set("base")
    menu = [menu_m1(diagB, gB_diag, warp),
            menu_diag_pair("M2", diagB, "base", "lowpass12"),
            menu_m4(dec, rel(ctx.root, dec_path), g1, dec_err),
            menu_diag_pair("M6", diagB, "base", "model_legs")]
    v2, r3, missing, barred = select_v2(menu, ab)
    rows.extend(menu)
    rows.append(r2a(diagB, dec))
    rows.append(row("R3", "R3: every menu row evaluated and none fires, so there is no v2",
                    "evaluated" if not missing else "blocked", r3, [], missing=missing))
    v2_line = CONSTANTS["menu"][v2]["line"] if v2 else None
    if v2:
        it = next(x for x in menu if x["id"] == v2)
        decision = (f"v2: {v2} ({CONSTANTS['menu'][v2]['what']}), effect {fmt(it['data']['fraction'], 3)} of its gap"
                    + (" (not evaluated: " + ", ".join(missing) + ")" if missing else ""))
    else:
        decision = "v2: none (" + r3 + ")"
    r0 = rows[0]
    if r0["status"] == "evaluated" and not r0["verdict"].startswith("R0 holds"):
        decision += "; R0 fails (row R0): the rows above are read on a run that did not close cleanly"
    return {"names": {"B": nameB, "A": nameA}, "rows": rows, "v2_item": v2, "v2_line": v2_line,
            "r3": r3, "decision": decision, "r_ab": ab}


def g3_compare(ctx, name1, name2):
    s1, s2 = g3_sources(ctx.root, name1), g3_sources(ctx.root, name2)
    return pick_pair(s1, s2)


def seed_values(ctx, name1, name2, g1_variants):
    """Lines of two checkpoints and their absolute differences."""
    out, lines = {}, []
    a, b = g3_compare(ctx, name1, name2)
    for key in ("S", "V"):
        for sub, tag in ((SUB12, "12"), (SUB14, "14")):
            line = G3_LINES[key]
            _, pa = g3_line(a, key, sub)
            _, pb = g3_line(b, key, sub)
            d = abs(pa - pb) if pa is not None and pb is not None else None
            out[f"{key}{tag}"] = {"seed1": pa, "seed2": pb, "abs_diff": d, "unit": UNITS[line]}
            lines.append(f"{key} ({tag}): seed 1 {fmt(pa, nd_of(line))}, seed 2 {fmt(pb, nd_of(line))}, |difference| "
                         f"{fmt(d, nd_of(line))} {UNITS[line]}")
    g1a, g1b, var = g1_pair(ctx.root, name1, name2, g1_variants)
    pa, pb = (g1a or {}).get("point"), (g1b or {}).get("point")
    d = abs(pa - pb) if pa is not None and pb is not None else None
    out["G1a"] = {"seed1": pa, "seed2": pb, "abs_diff": d, "unit": "mm", "variant": var}
    lines.append(f"G1a ({var or 'n/a'}): seed 1 {fmt(pa, 1)}, seed 2 {fmt(pb, 1)}, |difference| {fmt(d, 1)} mm")
    lines.append(f"sources: G3 {a['source'] if a else 'n/a'} and {b['source'] if b else 'n/a'}")
    return out, lines


def stage_seeds(ctx):
    a = ctx.a
    r1, _ = run_log(ctx, a.run)
    r2, _ = run_log(ctx, a.run2)
    step = CONSTANTS["v2check"]["step_ckpt"]
    rows, diff = [], {}
    n1, n2 = f"{run_name(a.run)}_{step}", f"{run_name(a.run2)}_{step}"
    vals, lines = seed_values(ctx, n1, n2, ("lt2", "full"))
    diff[step] = vals
    miss = [k for k, v in vals.items() if v["abs_diff"] is None]
    rows.append(row("seeds-4000", f"|seed 1 minus seed 2| at {step}", "evaluated" if not miss else "not evaluated",
                    ("seed differences at " + step) if not miss else "not evaluated: " + ", ".join(miss),
                    lines, values=vals))
    e1, _ = end_name(r1 or [], a.run, "B", None)
    e2, _ = end_name(r2 or [], a.run2, "B", None)
    if e1 and e2 and b_end_verify(r1 or []) and b_end_verify(r2 or []):
        vals, lines = seed_values(ctx, e1, e2, ("full", "lt2"))
        miss = [k for k, v in vals.items() if v["abs_diff"] is None]
        rows.append(row("seeds-end", "|seed 1 minus seed 2| at the B end", "evaluated" if not miss else "not evaluated",
                        f"seed differences at the B end ({e1}, {e2})" if not miss else
                        "not evaluated: " + ", ".join(miss),
                        lines, values=vals))
        diff["B_end"] = vals
    else:
        rows.append(not_evaluated("seeds-end", "|seed 1 minus seed 2| at the B end",
                                  ["no phase B end verify event of " + (run_name(a.run) if not (e1 and b_end_verify(r1 or [])) else run_name(a.run2))]))
    return {"rows": rows, "seed_diff": diff, "decision": "; ".join(r["verdict"] for r in rows)}


def latest(ctx, pattern, explicit=None):
    if explicit:
        return ctx.p(explicit)
    files = sorted(glob.glob(os.path.join(ctx.root, "reports", "gpu_queue", pattern)))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    return files[-1] if files else None


def seed_diff_of(seeds_doc, section, key):
    try:
        return num(seeds_doc["seed_diff"][section][key]["abs_diff"])
    except (KeyError, TypeError):
        return None


def stage_v2check(ctx):
    a = ctx.a
    c = CONSTANTS["v2check"]
    bpath = latest(ctx, "branch_branch_*.json", a.branch_json)
    ctx.note("branch json", bpath)
    bdoc, berr = read_obj(bpath)
    item = (bdoc or {}).get("v2_item")
    if not item:
        if bdoc is None:
            why = "no branch_branch_*.json in reports/gpu_queue" if bpath is None else berr
        else:
            why = "the branch output names no v2 item"
        r = not_evaluated("v2check", "v2 go or no-go", [why])
        return {"rows": [r], "go": False, "decision": "no check: " + r["verdict"]}
    target = {"stride_mae_cm": "S12", "g1a_mm": "G1a"}[CONSTANTS["menu"][item]["line"]]
    spath = latest(ctx, "branch_seeds_*.json", a.seeds_json)
    ctx.note("seeds json", spath)
    sdoc, _ = read_obj(spath)
    step = c["step_ckpt"]
    seeds = {k: seed_diff_of(sdoc, step, k) for k in ("S12", "V12", "G1a")}
    no_seed = [k for k, v in seeds.items() if v is None]
    v1, v2 = f"{run_name(a.run)}_{step}", f"{run_name(a.v2_run)}_{step}"
    lines, values, missing = [], {}, []
    if no_seed:
        lines.append("without seed 2's step-4,000 numbers for " + ", ".join(no_seed) + ": 10 and 5 percent only")
    ga, gb = g3_compare(ctx, v1, v2)
    for key in ("S12", "V12"):
        k = key[0]
        pa, xa = g3_line(ga, k, SUB12)
        pb, xb = g3_line(gb, k, SUB12)
        p = paired(pa or {}, pb or {})
        values[key] = {"v1": xa, "v2": xb, "improvement": p, "line": G3_LINES[k]}
    g1v1, g1v2, var = g1_pair(ctx.root, v1, v2, ("lt2", "full"))
    if g1v1 and g1v2:
        p = paired(g1v1["takes"], g1v2["takes"], g1v1["knots"], g1v2["knots"])
        values["G1a"] = {"v1": g1v1["point"], "v2": g1v2["point"], "improvement": p, "line": "g1a_mm",
                         "variant": var}
    else:
        values["G1a"] = {"v1": None, "v2": None, "improvement": {"point": None, "interval": None, "n": 0,
                                                                  "excludes_0": False, "reason": "no G1 pair"},
                         "line": "g1a_mm", "variant": None}
    checks = {}
    for key, v in values.items():
        line, nd = v["line"], nd_of(v["line"])
        p = v["improvement"]
        sd = seeds.get(key)
        if v["v1"] is None or v["v2"] is None:
            missing.append(key)
            lines.append(f"{key}: not evaluated (v1 {fmt(v['v1'], nd)}, v2 {fmt(v['v2'], nd)})")
            continue
        if key == target:
            thr = max(c["target_fraction"] * v["v1"], c["seed_multiple"] * sd if sd is not None else 0.0)
            ok = p["point"] is not None and p["point"] >= thr and p["interval"] is not None and p["interval"][0] > 0
            checks[f"target {key}"] = ok
            lines.append(f"target {key}: v1 {fmt(v['v1'], nd)} -> v2 {fmt(v['v2'], nd)} {UNITS[line]}; paired "
                         f"improvement {fmt(p['point'], nd)} {fmt_iv(p['interval'], nd)}, n {p['n']}; needs at least "
                         f"{fmt(thr, nd)} (max of 10 percent and 2 x seed difference {fmt(sd, nd)}) with an interval "
                         f"above 0: {'yes' if ok else 'no'}")
        else:
            worse = v["v2"] - v["v1"]
            lim = max(c["other_fraction"] * v["v1"], c["seed_multiple"] * sd if sd is not None else 0.0)
            ok = worse <= lim
            checks[f"other {key}"] = ok
            lines.append(f"other {key}: v1 {fmt(v['v1'], nd)} -> v2 {fmt(v['v2'], nd)} {UNITS[line]}; worse by "
                         f"{fmt(worse, nd)}, at most {fmt(lim, nd)} (max of 5 percent and 2 x seed difference "
                         f"{fmt(sd, nd)}): {'yes' if ok else 'no'}")
    recs, _ = run_log(ctx, a.v2_run)
    if recs is None:
        missing.append("v2 run log")
    else:
        alarms = log_events(recs, "alarm")
        nfs = nonfinite_stop(recs)
        peak = peak_reserved_gb(recs)
        checks["no alarm"] = not alarms
        checks["no non-finite stop"] = not nfs
        if peak is None:
            missing.append("peak reserved memory")
        else:
            checks["peak reserved"] = peak <= c["peak_reserved_gb_max"]
        lines.append(f"v2 run: alarms {len(alarms)}, non-finite stop {'yes' if nfs else 'no'}, peak reserved "
                     f"{fmt(peak, 2)} GB (at most {c['peak_reserved_gb_max']})")
    go = not missing and all(checks.values())
    verdict = "go" if go else ("not evaluated: " + ", ".join(missing) if missing else
                               "no-go: " + ", ".join(k for k, x in checks.items() if not x))
    r = row("v2check", f"v2 go or no-go ({item}, target {target})", "evaluated" if not missing else "not evaluated",
            verdict, lines, checks=checks, values=values, seed_diff=seeds)
    return {"rows": [r], "go": go, "item": item, "target": target, "decision": verdict}


def stage_reference(ctx):
    a = ctx.a
    k = CONSTANTS["reference"]["seed_multiple"]
    r1, _ = run_log(ctx, a.run)
    v1_name, last_b = end_name(r1 or [], a.run, "B", None)
    v1_ver = b_end_verify(r1 or [])
    spath = latest(ctx, "branch_seeds_*.json", a.seeds_json)
    ctx.note("seeds json", spath)
    sdoc, _ = read_obj(spath)
    sdS, sdG = seed_diff_of(sdoc, "B_end", "S12"), seed_diff_of(sdoc, "B_end", "G1a")
    default = (v1_ver or last_b or {}).get("path")
    default = default.replace(chr(92), "/") if default else default
    rows, best = [], None
    if v1_ver is None:
        rows.append(not_evaluated("reference", "v1's phase B end", ["no phase B end verify event of " + run_name(a.run)]))
    for cand in a.candidates:
        rc, _ = run_log(ctx, cand)
        ver = b_end_verify(rc or [])
        title = f"candidate {run_name(cand)}"
        if ver is None or v1_ver is None:
            rows.append(not_evaluated(title, title, ["no phase B end verify event" if ver is None else "no v1 B end"]))
            continue
        cname = f"{run_name(cand)}_{ckpt_stem(ver.get('path'))}"
        ga, gb = g3_compare(ctx, v1_name, cname)
        sa, xa = g3_line(ga, "S", SUB12)
        sb, xb = g3_line(gb, "S", SUB12)
        p = paired(sa or {}, sb or {})
        g1a, g1b, var = g1_pair(ctx.root, v1_name, cname)
        ga_pt, gb_pt = (g1a or {}).get("point"), (g1b or {}).get("point")
        lines = [f"S (12): v1 {fmt(xa, 2)} -> {fmt(xb, 2)} cm; v1 minus candidate {fmt(p['point'], 2)} "
                 f"{fmt_iv(p['interval'], 2)}, n {p['n']}; needs more than 2 x seed difference {fmt(sdS, 2)}",
                 f"G1a ({var or 'n/a'}): v1 {fmt(ga_pt, 1)} -> {fmt(gb_pt, 1)} mm; worse by at most 2 x seed "
                 f"difference {fmt(sdG, 1)}"]
        reasons = []
        if sdS is None or sdG is None:
            reasons.append("no seed difference at the B end")
        if p["point"] is None:
            reasons.append(p["reason"] or "no S pair")
        if ga_pt is None or gb_pt is None:
            reasons.append("no G1a pair")
        if reasons:
            rows.append(not_evaluated(title, title, reasons, lines))
            continue
        beats = p["point"] > k * sdS and p["interval"][0] > 0 and (gb_pt - ga_pt) <= k * sdG
        rows.append(row(title, title, "evaluated", "beats v1" if beats else "does not beat v1", lines,
                        checkpoint=ver.get("path"), name=cname, improvement=p, beats=beats))
        if beats and (best is None or p["point"] > best[1]):
            best = (str(ver.get("path")).replace(chr(92), "/"), p["point"], cname, xb, gb_pt)
    if best:
        ref, ref_name = best[0], best[2]
        decision = (f"reference_checkpoint {ref} ({ref_name}); its S {fmt(best[3], 2)} cm and G1a {fmt(best[4], 1)} "
                    f"mm are {POST_SELECTION}")
    else:
        ref, ref_name = default, v1_name
        decision = (f"reference_checkpoint {ref} (v1's phase B end, the default; no candidate beats it)" if ref else
                    "reference_checkpoint none: v1 has no verified phase B checkpoint")
    return {"rows": rows, "reference_checkpoint": ref, "ref_name": ref_name, "label": POST_SELECTION,
            "decision": decision}


def ckpt_step(name):
    m = re.search(r"ckpt_B_(\d+)$", name or "")
    return int(m.group(1)) if m else None


def stage_extension(ctx):
    a = ctx.a
    c = CONSTANTS["extension"]
    rpath = latest(ctx, "branch_reference_*.json", a.reference_json)
    ctx.note("reference json", rpath)
    rdoc, _ = read_obj(rpath)
    ckpt = (rdoc or {}).get("reference_checkpoint")
    if ckpt:
        run_dir = os.path.dirname(str(ckpt).replace("\\", "/"))
        run_dir = rel(ctx.root, ctx.p(run_dir)) if run_dir else a.run
    else:
        run_dir = a.run
    recs, _ = run_log(ctx, run_dir)
    if not ckpt:
        v = b_end_verify(recs or [])
        ckpt = v.get("path") if v else None
    rows, reasons, lines = [], [], []
    if not ckpt:
        rows.append(not_evaluated("extension", "extension", ["no candidate checkpoint"]))
        return {"rows": rows, "extend": False, "decision": rows[0]["verdict"]}
    ckpt = str(ckpt).replace(chr(92), "/")
    rn = run_name(run_dir)
    end = f"{rn}_{ckpt_stem(ckpt)}"
    nB = phase_b_steps(recs or [])
    lim = c["fraction_of_phase_b"] * nB
    steps = sorted({int(r["step"]) for r in log_events(recs or [], "checkpoint", "B")
                    if num(r.get("step")) is not None and r["step"] <= lim})
    s75 = steps[-1] if steps else int(lim // 2000) * 2000
    n75 = f"{rn}_ckpt_B_{s75:06d}"
    ga, gb = g3_compare(ctx, n75, end)
    sa, xa = g3_line(ga, "S", SUB12)
    sb, xb = g3_line(gb, "S", SUB12)
    p = paired(sa or {}, sb or {})
    s_ok = p["point"] is not None and p["point"] > 0 and p["interval"][0] > 0
    lines.append(f"S (12): {n75} {fmt(xa, 2)} -> {end} {fmt(xb, 2)} cm; 75 percent minus end {fmt(p['point'], 2)} "
                 f"{fmt_iv(p['interval'], 2)}, n {p['n']}: {'falls' if s_ok else 'does not fall'}")
    if p["point"] is None:
        reasons.append("S at the end or at 75 percent: " + (p["reason"] or "missing"))
    curve = []
    g1base = os.path.join(ctx.root, "reports", "gates", "g1")
    found = set()
    for f in glob.glob(os.path.join(g1base, f"*{rn}_ckpt_B_*")):
        m = re.search(re.escape(rn) + r"_ckpt_B_(\d+)", os.path.basename(f))
        if m:
            found.add(int(m.group(1)))
    for st in sorted(found):
        g = g1_reads(ctx.root, f"{rn}_ckpt_B_{st:06d}").get("lt2")
        if g and g.get("point") is not None:
            curve.append((st, g["point"]))
    end_step = ckpt_step(end)
    at_end = [x for x in curve if x[0] == end_step]
    before = [x for x in curve if end_step is not None and x[0] < end_step]
    if not at_end or not before:
        reasons.append("G1 --limit-takes curve: needs the end and an earlier phase B checkpoint")
        g_ok = False
    else:
        g_ok = at_end[0][1] < before[-1][1]
        lines.append("G1a lt2 curve: " + ", ".join(f"{s} {v:.1f}" for s, v in curve) +
                     f" mm; end {at_end[0][1]:.1f} against {before[-1][1]:.1f} at {before[-1][0]}: "
                     f"{'still falls' if g_ok else 'does not fall'}")
    rate = seconds_per_step(recs or [])
    rate_src = f"{rn}/log.jsonl"
    if a.s_per_step:
        rate, rate_src = float(a.s_per_step), "--s-per-step"
    if rate is None:
        bdoc, _ = read_obj(os.path.join(ctx.root, "reports", "gpu_queue", "baseline_s_per_step.json"))
        rate = num((bdoc or {}).get("s_per_step"))
        rate_src = "reports/gpu_queue/baseline_s_per_step.json"
    deadline = datetime.datetime.fromisoformat(c["deadline"])
    hours = (deadline - ctx.now).total_seconds() / 3600.0 - c["evaluation_reserve_h"]
    ext_steps = int(hours * 3600.0 // rate) if rate and hours >= c["min_hours"] else 0
    lines.append(f"time: {fmt(hours, 2)} h after the 1.5 h reserve to {c['deadline']} (at least {c['min_hours']:g} h "
                 f"needed); {fmt(rate, 4)} s per step ({rate_src}); steps {ext_steps}")
    if rate is None:
        reasons.append("no measured seconds per step")
    if reasons:
        extend = False
        r = not_evaluated("extension", "extension", reasons, lines)
    else:
        extend = bool(s_ok and g_ok and hours >= c["min_hours"] and ext_steps > 0)
        why = [] if extend else ([x for x, ok in (("S does not fall", s_ok), ("G1 curve does not fall", g_ok),
                                                  ("under 4 h remain", hours >= c["min_hours"])) if not ok])
        r = row("extension", "extension", "evaluated", "extend" if extend else "no extension: " + ", ".join(why),
                lines, s_falls=s_ok, g1_falls=g_ok, hours=hours)
    rows.append(r)
    return {"rows": rows, "extend": extend, "ext_ckpt": ckpt if extend else None,
            "ext_steps": ext_steps if extend else 0, "s_per_step": rate, "decision": r["verdict"]}


# ---------------------------------------------------------------- output
def render(ctx, stage, res, frozen):
    out = [f"weekend_branch --stage {stage} ({ctx.now.isoformat(timespec='seconds')})",
           f"weekend_branch.py SHA-256 {frozen['script']}; constants SHA-256 {frozen['constants']}; {frozen['text']}"]
    for r in res["rows"]:
        out.append("")
        out.append(f"{r['id']}: {r['verdict']}")
        out.append(f"  {r['title']}")
        for x in r["lines"]:
            out.append(f"  - {x}")
    out.append("")
    out.append("decision: " + res["decision"])
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return _jsonable(x.item())
    return x


def write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_outputs(ctx, stage, res, frozen, text):
    stamp = ctx.now.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(ctx.root, "reports", "gpu_queue", f"branch_{stage}_{stamp}")
    doc = {"format": FORMAT, "stage": stage, "time": ctx.now.isoformat(timespec="seconds"),
           "script": "scripts/weekend_branch.py", "script_sha256": frozen["script"],
           "constants_sha256": frozen["constants"], "frozen": frozen, "argv": sys.argv[1:], "inputs": ctx.inputs}
    doc.update({k: v for k, v in res.items()})
    write_atomic(base + ".json", json.dumps(_jsonable(doc), indent=1) + "\n")
    md = [f"# Branch rule output: stage {stage}", "",
          f"Written by scripts/weekend_branch.py at {doc['time']}; the rule is frozen in {HANDOFF.replace(chr(92), '/')} "
          f"section 1. Inputs and their SHA-256 are in {rel(ctx.root, base + '.json')}.", "", "```"] + text + ["```", ""]
    write_atomic(base + ".md", "\n".join(md))
    paths = [base + ".md", base + ".json"]
    if stage == "extension":
        ext = os.path.join(ctx.root, "cache", "runs", "queue", "EXTEND")
        if res.get("extend"):
            write_atomic(ext, json.dumps({"ext_ckpt": res["ext_ckpt"], "ext_steps": res["ext_steps"],
                                          "s_per_step": res.get("s_per_step"), "written": doc["time"],
                                          "writer": "scripts/weekend_branch.py",
                                          "branch_json": rel(ctx.root, base + ".json")}, indent=1) + "\n")
            paths.append(ext)
        elif os.path.exists(ext):
            prev, _ = read_json(ext)
            if isinstance(prev, dict) and prev.get("writer") == "scripts/weekend_branch.py":
                os.remove(ext)
                paths.append(ext + " (removed)")
    return paths


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("branch", "seeds", "v2check", "reference", "extension"))
    ap.add_argument("--hashes", action="store_true", help="print the SHA-256 of this file and of CONSTANTS")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=TREE)
    ap.add_argument("--now", default=None, help="ISO time used for the output names and the extension clock")
    ap.add_argument("--run", default=DEFAULT_RUN, help="seed 1 run folder (v1)")
    ap.add_argument("--run2", default=DEFAULT_RUN2, help="seed 2 run folder")
    ap.add_argument("--v2-run", default=DEFAULT_V2_RUN)
    ap.add_argument("--candidates", nargs="*", default=[DEFAULT_RUN2, DEFAULT_V2_RUN])
    ap.add_argument("--name", default=None, help="B end name (default: from the run's phase B verify event)")
    ap.add_argument("--name-a", default=None, help="A end name (default: from the run's phase A verify event)")
    ap.add_argument("--g3-b", default=None)
    ap.add_argument("--g3-a", default=None)
    ap.add_argument("--g1", default=None)
    ap.add_argument("--insim", default=None)
    ap.add_argument("--diag-b", action="append", default=None)
    ap.add_argument("--diag-a", action="append", default=None)
    ap.add_argument("--decompose", default=None)
    ap.add_argument("--warp", default=None)
    ap.add_argument("--job-log", default=None)
    ap.add_argument("--exit-code", type=int, default=None)
    ap.add_argument("--branch-json", default=None)
    ap.add_argument("--seeds-json", default=None)
    ap.add_argument("--reference-json", default=None)
    ap.add_argument("--s-per-step", type=float, default=None)
    a = ap.parse_args(argv)
    if not a.hashes and not a.stage:
        ap.error("--stage or --hashes is required")
    return a


STAGES = {"branch": stage_branch, "seeds": stage_seeds, "v2check": stage_v2check, "reference": stage_reference,
          "extension": stage_extension}


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    a = parse(argv)
    ctx = Ctx(a)
    frozen = frozen_check(ctx.root)
    if a.hashes:
        print(f"weekend_branch.py SHA-256: {frozen['script']}")
        print(f"constants SHA-256: {frozen['constants']}")
        print(f"canonical constants: {canonical_constants()}")
        print(frozen["text"])
        return 0
    res = STAGES[a.stage](ctx)
    text = render(ctx, a.stage, res, frozen)
    print("\n".join(text))
    if a.write:
        for p in write_outputs(ctx, a.stage, res, frozen, text):
            print(f"wrote {rel(ctx.root, p) if not p.endswith('(removed)') else p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
