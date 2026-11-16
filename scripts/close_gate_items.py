"""Measurements behind the closure of the OPEN items of the gate rules and their attainability checks (task GR2,
2026-10-05): docs/rules/G3_RULE.md section 10 items 4 to 6, docs/rules/G4_RULE.md section 8 items 9 and 12.

CPU only (CUDA_VISIBLE_DEVICES is set to -1 before any import that loads torch), development data and existing
reports only: no lockbox, EVAL, CODY or care_pd record is read. Every reader is called with its lockbox flag at
False on non-lockbox persons; the lockbox read log (reports/gates/lockbox_reads.csv) is hashed before and after
the run and must not change. Writes under reports/gate_rules/ only (--out), never in docs/ or in MoveAtlas.

Parts (--parts, default all, in this order):
- g3: the blind input-validity check of mdfm.gates.g3 (input_validity) on the G3 records of the 54 training-fold
  controls first, then on the 14 dev controls (the order of docs/proposals/G3_INPUT_VALIDITY_PROPOSAL.md section
  4); the dev rows of g3.dev_run(sets=True, validity=True) without a model on the sets dev14, dev12 and
  valid_inputs; the IC-margin probe: the pressure insoles (the insole ICs of the targets, each trial's own offset
  removed) scored as an IC predictor against the walkway reference with the margin of G3 rule section 6 against
  the shank-gyroscope detector, on the reference ICs both matched; the stride oracle (added 2026-10-05 on review,
  defined here before its first run): each walkway stride predicted by the mean length of the same person's
  walkway strides (both feet) in the other passes of the same trial (leave one pass out), else of the person's
  other task, else no prediction (unmatched), scored by g3.evaluate on the stride lines with their matched share.
  Outputs g3/validity_train.csv, g3/validity_dev.csv, g3/set_rows.csv, g3/ic_margin_probe.csv,
  g3/stride_oracle.csv, g3/results.json, g3/summary.md.
- g4emg: the attainability reference of the EMG margin (G4 rule, delta 0.03, written before this script ran):
  on the gait120 dev fold, trials of at least 1 s of pose (g4.EMG_MIN_S), the population template of the rule
  (built on every paired trial of the 77 train-fold persons) against a personal template, the same phase-bin
  template built on the person's other paired trials only (leave one trial out), scored on the same (trial,
  muscle) pairs and frames (g4.emg_trial_table, same_samples); per person the median r of each, the gain the
  personal median minus the template median, the median over persons with the person bootstrap of the rule.
  Printed beside, a ceiling that decides nothing: each trial's own phase template, in sample. Outputs
  g4_emg/per_trial.csv, g4_emg/per_person.csv, g4_emg/results.json, g4_emg/summary.md.
- g4grf: the attainability reference of the GRF RMSE line (pass at most 10 percent BW): on the 40 non-lockbox
  stroke patients, the plate truth of the rule (g4.grf_trials) and the scored samples of the rule (plate-valid
  samples where the Newton floor is defined, g4.grf_samples), predicted by an oracle that knows each stance's
  timing from the plates: a stance is a run of at least STANCE_MIN_S of plate-valid samples above the unloaded
  band (force.BAND_N over m g); each stance is predicted by the mean time-normalized curve (N_STANCE points) of
  the same patient's complete stances of the same foot in its other trials (the other foot's when the same foot
  has none), stretched to the stance's duration, and 0 BW outside stances. A patient with one trial has no
  leave-one-trial-out template and is listed. A second oracle, the same with the template of every other
  patient's complete stances (both feet, leave one patient out), runs on every patient. Per patient the RMSE in
  percent BW and Pearson r over its pooled samples, mean over patients with the rule's person bootstrap; the
  Newton floor on the same samples beside, and each oracle's grf_r_gain over it (per patient the oracle's r
  minus the floor's, mean over patients), since grf_rmse passes only with grf_r_gain passing. Rows: primary
  (the rule's samples), no_low_peak (the secondary row of G4 rule section 2 (b)) and complete_stances_only (a
  diagnostic that decides nothing and is not the rule's sample set: the samples inside incomplete stances, cut
  by the trial start, the trial end or a plate edge, left out, since an oracle stretches a full-stance template
  over such a truncated run). Outputs g4_grf/per_patient.csv, g4_grf/per_series.csv, g4_grf/results.json,
  g4_grf/summary.md.
- g4gain: the dev value of emg_r_gain of the two phase B ends, after delta was written: per person model_r minus
  template_r of reports/gates/g4/dev_phaseA_root_full_ckpt_B_032044/per_person_emg.csv and
  reports/gates/g4/dev_pilot_v1_s2_ckpt_B_032044/per_person_emg.csv (the pairs of scripts/eval_g4_dev.py),
  median over persons, person bootstrap. A dev estimate that decides nothing. Outputs g4_gain_dev/results.json,
  g4_gain_dev/summary.md.
- index: the attainability table of every success margin of G3 and G4 from the parts' results.json and the
  existing reports it names, and the provenance. Outputs attainability.csv, summary.md, _provenance.json.

Attainability, as computed here (part index; standard revised 2026-10-05 on review, one standard for G3 and G4).
Each reference is computed from files that exist and has a kind:
- predictor: a dev model row, a baseline or an instrument under the line's own rules, on the line's data;
- oracle: the line's data, with partial truth information the line does not give a predictor (the plates' stance
  timing or amplitude, a person's own EMG or own walkway strides from its other trials or passes, an instrument
  aligned per trial on the reference), never the line's target itself;
- simulation: a predictor (the model or a baseline) in the model's synthetic domain with the line's readout;
- reported: printed beside and deciding nothing: the line's readout applied to the truth itself (in simulation or
  on real data such a reference holds the target and so shows nothing), and every row of a set other than the
  rule's primary row (valid_inputs for G3 dev rows, equal to dev12; the primary row for G4 GRF).
A reference lands on the pass side of a compound line only when every condition of the line holds jointly with
the line's statistic: grf_rmse with grf_r_gain at least 0.03, emg_r with emg_r_gain at least 0.03, the G3 IC line
with the margin (at most 0.8 times the detector on the ICs both matched), and stride_mae_cm, speed_mae_m_s and
stride_mape_pct with a matched share of at least 0.9; a condition that cannot be evaluated blocks the pass. A
success margin is "shown" when a predictor lands on its pass side, "shown by an oracle" when only an oracle does,
"shown in simulation" when only a simulation predictor does, "not shown" otherwise; the check passes for a margin
unless it is not shown. An oracle shows a margin reachable in principle (the truth's own variability does not
forbid it), not reachable by a model.

Run from the tree root:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
  <conda>/envs/mdfm/python.exe scripts/close_gate_items.py [--parts g3,g4emg,g4grf,g4gain,index]
"""
import argparse
import collections
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only, before torch loads (mdfm.gates.g4 imports it)
sys.dont_write_bytecode = True

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.gates import common  # noqa: E402

OUT = os.path.join(TREE, "reports", "gate_rules")
PARTS = ("g3", "g4emg", "g4grf", "g4gain", "index")
RULES = {g: os.path.join(TREE, "docs", "rules", f"{g}_RULE.md") for g in ("G1", "G3", "G4")}
G4_EMG_GAIN_DELTA = 0.03                            # docs/rules/G4_RULE.md section 6 and section 8 item 9
STANCE_MIN_S = 0.1                                  # s: a loaded plate run shorter than this is not a stance
N_STANCE = 101                                      # points of a time-normalized stance curve
REPORTS = {
    "g3_dev_v1": "reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044/g3_dev.json",
    "g3_dev_s2": "reports/gates/g3/dev_pilot_v1_s2_ckpt_B_032044/g3_dev.json",
    "g3_sim_phase_a": "reports/gates/g3_sim/dev_phaseA_root_full_ckpt_A_010681/g3_sim.json",
    "g3_insim_v1": "reports/gates/g3_sim/insim_phaseA_root_full_ckpt_B_032044/insim.json",
    "g3_supervised": "reports/bench/g3_supervised_baseline/results.json",
    "g4_dev_v1": "reports/gates/g4/dev_phaseA_root_full_ckpt_B_032044/g4.json",
    "g4_dev_s2": "reports/gates/g4/dev_pilot_v1_s2_ckpt_B_032044/g4.json",
    "g4_pp_emg_v1": "reports/gates/g4/dev_phaseA_root_full_ckpt_B_032044/per_person_emg.csv",
    "g4_pp_emg_s2": "reports/gates/g4/dev_pilot_v1_s2_ckpt_B_032044/per_person_emg.csv",
}


# ---------------------------------------------------------------- helpers
def _abs(rel_path):
    return os.path.join(TREE, *rel_path.split("/"))


def _load_json(rel_path):
    with open(_abs(rel_path), encoding="utf-8") as f:
        return json.load(f)


def _git(args, cwd, safe=None):
    cmd = ["git"] + (["-c", f"safe.directory={safe}"] if safe else []) + ["-C", cwd] + args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip() if r.returncode == 0 else f"unavailable ({r.stderr.strip()[:80]})"


def _f(x, nd=4):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def _iv(iv, nd=4):
    if iv is None or any(v is None for v in iv):
        return "n/a"
    return f"[{_f(iv[0], nd)}, {_f(iv[1], nd)}]"


def _write_json(path, obj):
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=common._default, allow_nan=True) + "\n"
    common.write_text(path, text)


def _md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return out


def _person_hash(ids):
    """The person-list SHA-256 of mdfm.data.sources.person_hash, as the gate reports print it."""
    from mdfm.data import sources
    return sources.person_hash(sorted(ids))


def _person_stat(values, statistic):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    from mdfm.gates import g4
    lo, hi = g4.person_bootstrap(v, statistic=statistic)
    return {"point": float(statistic(v)) if len(v) else float("nan"), "interval": [lo, hi], "n_persons": int(len(v))}


# ---------------------------------------------------------------- part g3
def part_g3(out):
    from mdfm.gates import g3
    t0 = time.time()
    train, dev = g3.controls("train"), g3.controls("dev")
    if set(train) & set(dev):
        raise ValueError("training fold and dev controls overlap")
    # T: the training fold first (the order of the proposal), then D and S on dev
    tg_train = g3.load_targets(train)
    vt = g3.validity_table(train, tg_train.trials)
    summ, tables, offsets, tg, _, _, preds = g3.dev_run(sets=True, validity=True)
    vd = tg.validity
    names = list(g3.set_names(summ))
    set_trials = {n: summ["population"]["stride_mae_cm"]["sets"][n]["trials"] for n in names}
    rows = []
    for pred, lines in summ.items():
        for line, st in lines.items():
            for n in names:
                x = st["sets"][n]
                rows.append({"predictor": pred, "line": line, "set": n, "point": x["point"],
                             "lo": (x.get("interval") or [None, None])[0], "hi": (x.get("interval") or [None, None])[1],
                             "n_persons": x.get("n_persons"), "matched_share": x.get("matched_share"),
                             "n_events": x.get("n_events"), "n_uncovered": x.get("n_uncovered", ""),
                             "n_trials": x.get("n_trials")})
    set_rows = pd.DataFrame(rows)
    # IC-margin probe: the insoles as an IC predictor against the walkway reference
    # two instruments: the insole ICs with each trial's own offset removed (aligned on the walkway, favourable)
    # and the raw insole ICs (one global offset only, fitted like every detector's)
    raw = tg.insole_ic[["trial_id", "foot", "t_raw_s"]].rename(columns={"t_raw_s": "t_s"})
    instruments = [g3.Predictions("insole", tg.insole_ic[["trial_id", "foot", "t_s"]].copy()),
                   g3.Predictions("insole_raw", raw.copy())]
    gyro = next(p for p in preds if p.name == "shank_gyro")
    off_ins = g3.fit_offsets(instruments, tg)
    offs = {k: v[0] for k, v in off_ins.items()}
    offs.update({k: v[0] for k, v in offsets.items() if k[0] == "shank_gyro"})
    s2, t2 = g3.evaluate(tg, [*instruments, gyro], offs, sets={n: frozenset(v) for n, v in set_trials.items()})
    probe = []
    for inst in instruments:
        for n in names:
            vs, vt2 = g3.set_view(s2, t2, n)
            m, g, ok = g3.ic_margin(vt2[inst.name]["ic_walkway"], vt2["shank_gyro"]["ic_walkway"])
            line = vs[inst.name]["ic_median_ms_walkway"]
            probe.append({"instrument": inst.name, "set": n, "ic_median_ms_walkway": line["point"],
                          "matched_share": line["matched_share"], "offset_ms": line["offset_ms"],
                          "margin_instrument_ms": m, "margin_detector_ms": g,
                          "required_ms": None if g is None or not math.isfinite(g) else g3.IC_MARGIN * g,
                          "ratio": (m / g) if (g and math.isfinite(g) and g > 0) else None, "margin_met": ok,
                          "n_persons": line["n_persons"]})
    # stride oracle: the person's own walkway strides of the other passes (leave one pass out), defined before its run
    oracle, oracle_counts = _stride_oracle(tg.strides)
    s3, _ = g3.evaluate(tg, [oracle], {}, sets={n: frozenset(v) for n, v in set_trials.items()})
    stride_oracle = []
    for n in names:
        for line in ("stride_mae_cm", "stride_mape_pct"):
            x = s3[oracle.name][line]["sets"][n]
            stride_oracle.append({"line": line, "set": n, "point": x["point"],
                                  "lo": (x.get("interval") or [None, None])[0],
                                  "hi": (x.get("interval") or [None, None])[1], "n_persons": x["n_persons"],
                                  "matched_share": x["matched_share"], "n_events": x["n_events"],
                                  "n_trials": x["n_trials"]})
    flagged_dev = sorted(set(vd.loc[~vd["valid"], "person_id"]))
    flagged_train = vt.loc[~vt["valid"], ["trial_id", "reasons"]].to_dict("records")
    valid_eq_12 = sorted(set_trials.get(g3.SET_VALID, [])) == sorted(set_trials.get(g3.SET_12, []))
    res = {
        "what": "G3: blind input-validity check (training fold, then dev), dev rows on dev14, dev12 and valid_inputs "
                "without a model, and the IC-margin probe with the insoles as predictor",
        "constants": {"REQUIRED_SITES": list(g3.REQUIRED_SITES), "MIN_VALID_SHARE": g3.MIN_VALID_SHARE,
                      "SWING_AXIS": g3.SWING_AXIS, "ENVELOPE_S": g3.ENVELOPE_S, "MIN_TEST_S": g3.MIN_TEST_S,
                      "IC_MARGIN": g3.IC_MARGIN, "IC_MATCH_S": g3.IC_MATCH_S},
        "train": {"persons": len(train), "records": int(len(vt)), "valid": int(vt["valid"].sum()),
                  "flagged": flagged_train, "persons_sha256": _person_hash(train)},
        "dev": {"persons": len(dev), "records": int(len(vd)), "valid": int(vd["valid"].sum()),
                "flagged_persons": flagged_dev, "persons_sha256": _person_hash(dev),
                "valid_inputs_equals_dev12": bool(valid_eq_12)},
        "offsets_ms": {f"{k[0]}|{k[1]}": {"offset_ms": 1000.0 * v[0], "n_matched": v[1]} for k, v in
                       sorted({**offsets, **off_ins}.items())},
        "sets": {n: {"trials": len(v)} for n, v in set_trials.items()},
        "ic_margin_probe": probe,
        "stride_oracle": {"definition": "each walkway stride predicted by the mean length of the same person's "
                                        "walkway strides (both feet) in the other passes of the same trial, else of "
                                        "the person's other task, else unmatched",
                          "counts": oracle_counts, "rows": stride_oracle},
        "time_s": round(time.time() - t0, 1),
    }
    os.makedirs(out, exist_ok=True)
    common.write_csv(vt, os.path.join(out, "validity_train.csv"))
    common.write_csv(vd, os.path.join(out, "validity_dev.csv"))
    common.write_csv(set_rows, os.path.join(out, "set_rows.csv"))
    common.write_csv(pd.DataFrame(probe), os.path.join(out, "ic_margin_probe.csv"))
    common.write_csv(pd.DataFrame(stride_oracle), os.path.join(out, "stride_oracle.csv"))
    _write_json(os.path.join(out, "results.json"), res)
    md = ["# G3: input validity, dev rows on three sets, IC-margin probe", "",
          "Written by scripts/close_gate_items.py (part g3); every number below is read from results.json and the CSV "
          "files of this folder. Development data only (training fold and dev controls of mdfm.data.sources.folds); "
          "no lockbox record or row read; no model.", "",
          "## Input validity (mdfm.gates.g3.input_validity), training fold first, then dev", "",
          f"- Training fold: {res['train']['persons']} persons, {res['train']['records']} records, "
          f"{res['train']['valid']} valid; flagged: {flagged_train if flagged_train else 'none'} (validity_train.csv).",
          f"- Dev controls: {res['dev']['persons']} persons, {res['dev']['records']} records, {res['dev']['valid']} "
          f"valid; flagged persons: {', '.join(flagged_dev) if flagged_dev else 'none'} (validity_dev.csv).",
          f"- The valid dev trials equal the set dev12 (the 14 without {', '.join(g3.EXCLUDED_FROM_12)}): "
          f"{res['dev']['valid_inputs_equals_dev12']}.",
          f"- Constants: {json.dumps(res['constants'])}.", "",
          "## Dev rows without a model (g3.dev_run(sets=True, validity=True))", ""]
    keep = set_rows[set_rows["predictor"].isin(["population", "shank_gyro", "walkway"])
                    & ~((set_rows["predictor"] == "walkway") & (set_rows["line"] != "ic_median_ms_insole"))]
    md += _md_table([{**{k: r[k] for k in ("predictor", "line", "set", "n_persons")},
                      "point": _f(r["point"]), "interval": _iv([r["lo"], r["hi"]]),
                      "matched share": _f(r["matched_share"], 3), "uncovered": r["n_uncovered"]}
                     for r in keep.to_dict("records")],
                    ["predictor", "line", "set", "point", "interval", "n_persons", "matched share", "uncovered"])
    md += ["", "## IC-margin probe: the insoles as an IC predictor against the walkway reference", "",
           "The insole ICs of the targets are scored as ic_median_ms against the walkway contact, and the margin of "
           f"rule section 6 compares them with the shank-gyroscope detector on the walkway ICs both matched (pass at "
           f"most {g3.IC_MARGIN} times the detector's value). Two instruments: insole, each trial's own insole minus "
           "walkway offset removed (an alignment on the reference that favours the insoles); insole_raw, the raw "
           "insole times with one global offset fitted on the 14 like every detector's.", ""]
    md += _md_table([{"instrument": r["instrument"], "set": r["set"], "line (ms)": _f(r["ic_median_ms_walkway"], 2),
                      "instrument on both matched (ms)": _f(r["margin_instrument_ms"], 2),
                      "detector on both matched (ms)": _f(r["margin_detector_ms"], 2),
                      "required (ms)": _f(r["required_ms"], 2), "ratio": _f(r["ratio"], 3),
                      "margin met": r["margin_met"], "persons": r["n_persons"]} for r in probe],
                    ["instrument", "set", "line (ms)", "instrument on both matched (ms)",
                     "detector on both matched (ms)", "required (ms)", "ratio", "margin met", "persons"])
    md += ["", f"Offsets (ms, fitted on the 14): " + "; ".join(f"{k} {_f(v['offset_ms'], 1)} (n {v['n_matched']})"
                                                             for k, v in res["offsets_ms"].items()) + ".",
           "", "## Stride oracle: the person's own walkway strides of its other passes", "",
           "Added on review (2026-10-05), defined before its first run as the analogue on G3 of the patient and "
           "personal templates of the G4 oracles: each walkway stride is predicted by the mean length of the same "
           "person's walkway strides (both feet) in the other passes of the same trial (leave one pass out); a "
           "stride of a trial with one pass takes the mean of the person's strides in its other task; a stride "
           "with neither has no prediction and counts as unmatched. It knows the person's own stride lengths "
           "from the walkway, which no predictor is given: an oracle. Counts: "
           f"{json.dumps(oracle_counts)}. The rule's primary dev row is valid_inputs (equal to dev12).", ""]
    md += _md_table([{"line": r["line"], "set": r["set"], "point": _f(r["point"]), "interval": _iv([r["lo"], r["hi"]]),
                      "persons": r["n_persons"], "matched share": _f(r["matched_share"], 3),
                      "strides": r["n_events"]} for r in stride_oracle],
                    ["line", "set", "point", "interval", "persons", "matched share", "strides"])
    md += ["", f"Run time {res['time_s']} s."]
    common.write_text(os.path.join(out, "summary.md"), "\n".join(md) + "\n")
    print(f"[g3] train {res['train']['valid']}/{res['train']['records']} valid; dev flagged {flagged_dev}; "
          f"valid==dev12 {valid_eq_12}; probe "
          f"{[(r['instrument'], r['set'], _f(r['ratio'], 3), r['margin_met']) for r in probe]}; stride oracle "
          f"{[(r['line'], r['set'], _f(r['point'], 2), _f(r['matched_share'], 3)) for r in stride_oracle]}")
    return res


def _stride_oracle(strides):
    """Predictions of the stride oracle of part g3 (module docstring) on a Targets.strides frame, and its counts."""
    from mdfm.gates import g3
    rows = []
    counts = collections.Counter()
    pid = strides["pass_id"].astype(str)
    for r in strides.itertuples():
        same = strides[(strides["trial_id"] == r.trial_id) & (pid != str(r.pass_id))]
        if len(same):
            v = float(same["length_m"].mean())
            counts["other_passes_same_trial"] += 1
        else:
            other = strides[(strides["person_id"] == r.person_id) & (strides["trial_id"] != r.trial_id)]
            if not len(other):
                counts["no_prediction"] += 1
                continue
            v = float(other["length_m"].mean())
            counts["other_task"] += 1
        rows.append((r.trial_id, r.foot, r.t0, r.t1, v))
    counts["strides"] = int(len(strides))
    s = pd.DataFrame(rows, columns=["trial_id", "foot", "t0", "t1", "length_m"])
    return g3.Predictions("person_stride_oracle", None, s, None), dict(counts)


# ---------------------------------------------------------------- part g4emg
def part_g4emg(out):
    from mdfm.gates import g4
    t0 = time.time()
    dev, train = g4.gait120_persons("dev"), g4.gait120_persons("train")
    if set(dev) & set(train):
        raise ValueError("template persons overlap the scored persons")
    stats = collections.Counter()
    template = g4.build_template(g4.emg_trials(train, stats=collections.Counter()))
    trials = g4.emg_trials(dev, stats=stats)
    by = collections.defaultdict(list)
    for tr in trials:
        by[tr.person].append(tr)
    loto = {}
    for p, trs in by.items():
        for tr in trs:
            others = [o for o in trs if o.key != tr.key]
            loto[(p, tr.key)] = g4.build_template(others) if others else None

    def pop(tr):
        return g4.template_predict(template, tr.phase)

    def personal(tr):
        T = loto.get((tr.person, tr.key))
        return g4.template_predict(T, tr.phase) if T is not None else np.full(tr.obs.shape, np.nan)

    def own(tr):
        return g4.template_predict(g4.build_template([tr]), tr.phase)

    left = []
    table = g4.emg_trial_table(trials, {"template": pop, "personal": personal}, min_s=g4.EMG_MIN_S,
                               same_samples=True, left_out=left)
    ceil = g4.emg_trial_table(trials, {"template": pop, "own_trial": own}, min_s=g4.EMG_MIN_S, same_samples=True)
    per = g4.emg_person_table(table, ["template", "personal"])
    per_c = g4.emg_person_table(ceil, ["template", "own_trial"])
    per["gain"] = per["personal_r"] - per["template_r"]
    per = per.merge(per_c[["person_id", "own_trial_r"]], on="person_id", how="left")
    per["own_trial_gain"] = per["own_trial_r"] - per_c.set_index("person_id").loc[per["person_id"], "template_r"].values
    gain = _person_stat(per["gain"], np.median)
    res = {
        "what": "G4 EMG margin attainability: personal leave-one-trial-out template against the population template "
                "on the gait120 dev fold (trials of at least 1 s), same pairs and frames",
        "delta": G4_EMG_GAIN_DELTA,
        "persons_dev": len(dev), "persons_dev_sha256": _person_hash(dev),
        "persons_template": len(train), "persons_template_sha256": _person_hash(train),
        "trials_paired": len(trials), "trials_left_out_under_min_s": len(left),
        "pairs": int(len(table)), "pairs_scored": int(table["scored"].sum()),
        "persons_scored": int(per["gain"].notna().sum()),
        "template_r": _person_stat(per["template_r"], np.median),
        "personal_r": _person_stat(per["personal_r"], np.median),
        "gain_personal_minus_template": gain,
        "persons_gain_at_least_delta": int((per["gain"] >= G4_EMG_GAIN_DELTA).sum()),
        "ceiling_own_trial_r": _person_stat(per["own_trial_r"], np.median),
        "ceiling_own_trial_gain": _person_stat(per["own_trial_gain"], np.median),
        "attainable": bool(gain["point"] >= G4_EMG_GAIN_DELTA),
        "stats": dict(stats), "time_s": round(time.time() - t0, 1),
    }
    os.makedirs(out, exist_ok=True)
    common.write_csv(table, os.path.join(out, "per_trial.csv"))
    common.write_csv(per, os.path.join(out, "per_person.csv"))
    _write_json(os.path.join(out, "results.json"), res)
    md = ["# G4 EMG margin: attainability reference", "",
          "Written by scripts/close_gate_items.py (part g4emg); every number below is read from results.json and the "
          "CSV files of this folder. Development data only (gait120 dev fold for the scored persons, train fold for "
          "the population template); no lockbox record or row read; no model. delta = 0.03 was written into "
          "docs/rules/G4_RULE.md before this part ran.", "",
          f"- Persons: {res['persons_dev']} dev (SHA-256 {res['persons_dev_sha256'][:12]}...), template "
          f"{res['persons_template']} train-fold persons. Paired trials {res['trials_paired']}, left out under "
          f"{g4.EMG_MIN_S} s {res['trials_left_out_under_min_s']}; (trial, muscle) pairs {res['pairs']}, scored for "
          f"both predictors {res['pairs_scored']}; persons with a gain {res['persons_scored']}.",
          f"- Population template (the rule's), median over persons of per-person medians: "
          f"{_f(res['template_r']['point'])} {_iv(res['template_r']['interval'])} ({res['template_r']['n_persons']} "
          "persons).",
          f"- Personal template, leave one trial out (an oracle that knows the person's EMG from its other trials): "
          f"{_f(res['personal_r']['point'])} {_iv(res['personal_r']['interval'])}.",
          f"- Gain, per person the personal median minus the template median, median over persons: "
          f"{_f(gain['point'])} {_iv(gain['interval'])} ({gain['n_persons']} persons; "
          f"{res['persons_gain_at_least_delta']} persons at or above 0.03).",
          f"- Ceiling that decides nothing (each trial's own phase template, in sample): r {_f(res['ceiling_own_trial_r']['point'])}, "
          f"gain over the template {_f(res['ceiling_own_trial_gain']['point'])} {_iv(res['ceiling_own_trial_gain']['interval'])}.",
          f"- Attainability of emg_r_gain at least 0.03: {'shown (oracle)' if res['attainable'] else 'not shown'} by the "
          "personal template.", "", f"Run time {res['time_s']} s. Counters: {json.dumps(res['stats'])}."]
    common.write_text(os.path.join(out, "summary.md"), "\n".join(md) + "\n")
    print(f"[g4emg] template {_f(res['template_r']['point'])} personal {_f(res['personal_r']['point'])} gain "
          f"{_f(gain['point'])} {_iv(gain['interval'])} n {gain['n_persons']}")
    return res


# ---------------------------------------------------------------- part g4grf
def _stances(obs_f, ok_f, thr, min_len):
    """[(start, end, complete)] of the stances of one foot: runs of plate-valid samples above thr of at least
    min_len frames; complete when the frames just before and after are plate-valid (and so at or below thr)."""
    from mdfm.gates import g4
    loaded = ok_f & (obs_f > thr)
    out = []
    for s, e in g4._runs(loaded):
        if e - s < min_len:
            continue
        out.append((int(s), int(e), bool(s > 0 and e < len(obs_f) and ok_f[s - 1] and ok_f[e])))
    return out


def _curve(y):
    return np.interp(np.linspace(0.0, 1.0, N_STANCE), np.linspace(0.0, 1.0, len(y)), y)


def _stretch(curve, n):
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(curve)), curve)


def part_g4grf(out):
    from mdfm.gates import g4
    from mdfm.kernels import force
    t0 = time.time()
    persons = g4.stroke_patients(lockbox=False)
    stats = collections.Counter()
    trials = g4.grf_trials(persons, stats=stats)
    st, peak = {}, {}                                # (person, key, foot) -> [(s, e, complete)], scored peak BW
    curves = collections.defaultdict(list)           # (person, key, foot) -> complete stance curves
    newton = {}
    for tr in trials:
        thr = force.BAND_N / (tr.mass_kg * force.G)
        n_min = int(round(STANCE_MIN_S * tr.fps))
        nw = newton[(tr.person, tr.key)] = g4.newton_predict(tr)
        ok = tr.obs_valid & np.isfinite(nw)
        for j in range(2):
            runs = _stances(tr.obs[:, j], tr.obs_valid[:, j], thr, n_min)
            st[(tr.person, tr.key, j)] = runs
            curves[(tr.person, tr.key, j)] = [_curve(tr.obs[s:e, j]) for s, e, c in runs if c]
            peak[(tr.person, tr.key, j)] = float(tr.obs[ok[:, j], j].max()) if ok[:, j].any() else float("nan")

    def tmpl(keys):
        cs = [c for k in keys for c in curves[k]]
        return np.mean(cs, axis=0) if cs else None

    def predict(tr, which):
        """(T, 2) BW of an oracle, or None when the patient oracles have no leave-one-trial-out template."""
        pred = np.zeros_like(tr.obs)
        for j in range(2):
            runs = st[(tr.person, tr.key, j)]
            if not runs:
                continue
            if which == "population":
                T = tmpl([k for k in curves if k[0] != tr.person])
            else:
                T = tmpl([k for k in curves if k[0] == tr.person and k[1] != tr.key and k[2] == j])
                if T is None:
                    T = tmpl([k for k in curves if k[0] == tr.person and k[1] != tr.key and k[2] != j])
            if T is None:
                return None
            for s, e, _ in runs:
                c = _stretch(T, e - s)
                if which == "patient_scaled":
                    c = c * (float(tr.obs[s:e, j].max()) / max(float(T.max()), 1e-9))
                pred[s:e, j] = c
        return pred

    names = ("newton", "population_oracle", "patient_oracle", "patient_scaled_oracle")
    preds = {}
    for tr in trials:
        k = (tr.person, tr.key)
        preds[k] = {"newton": newton[k], "population_oracle": predict(tr, "population"),
                    "patient_oracle": predict(tr, "patient"), "patient_scaled_oracle": predict(tr, "patient_scaled")}
    by = collections.defaultdict(list)
    for tr in trials:
        by[tr.person].append(tr)
    no_tmpl = sorted(p for p, trs in by.items() if any(preds[(t.person, t.key)]["patient_oracle"] is None for t in trs))
    in_stance, in_incomplete = {}, {}                # (person, key) -> (T, 2) masks of stance and incomplete-stance samples
    for tr in trials:
        a, b = np.zeros(tr.obs.shape, bool), np.zeros(tr.obs.shape, bool)
        for j in range(2):
            for s, e, c in st[(tr.person, tr.key, j)]:
                a[s:e, j] = True
                if not c:
                    b[s:e, j] = True
        in_stance[(tr.person, tr.key)], in_incomplete[(tr.person, tr.key)] = a, b
    scored_counts = collections.Counter()
    for tr in trials:
        ok = tr.obs_valid & np.isfinite(newton[(tr.person, tr.key)])
        scored_counts["scored_samples"] += int(ok.sum())
        scored_counts["scored_in_stance"] += int((ok & in_stance[(tr.person, tr.key)]).sum())
        scored_counts["scored_in_incomplete_stance"] += int((ok & in_incomplete[(tr.person, tr.key)]).sum())
    rows, series = [], []
    for row_name, low_out, drop_incomplete in (("primary", False, False), ("no_low_peak", True, False),
                                               ("complete_stances_only", False, True)):
        for person, trs in sorted(by.items()):
            obs, acc = [], collections.defaultdict(list)
            for tr in trs:
                pk = preds[(tr.person, tr.key)]
                ok = tr.obs_valid & np.isfinite(pk["newton"])
                if low_out:
                    for j in range(2):
                        if not peak[(tr.person, tr.key, j)] >= 0.5:
                            ok[:, j] = False
                if drop_incomplete:
                    ok = ok & ~in_incomplete[(tr.person, tr.key)]
                obs.append(tr.obs[ok])
                for n in names:
                    acc[n].append(pk[n][ok] if pk[n] is not None else None)
                if row_name == "primary":
                    for j in range(2):
                        runs = st[(tr.person, tr.key, j)]
                        series.append({"person_id": person, "key": tr.key, "foot": "LR"[j],
                                       "samples": int(ok[:, j].sum()), "stances": len(runs),
                                       "complete": int(sum(c for _, _, c in runs)),
                                       "samples_in_incomplete_stances": int(
                                           (ok[:, j] & in_incomplete[(tr.person, tr.key)][:, j]).sum()),
                                       "peak_bw": peak[(tr.person, tr.key, j)],
                                       "low_peak": bool(not peak[(tr.person, tr.key, j)] >= 0.5)})
            o = np.concatenate(obs)
            row = {"row": row_name, "person_id": person, "trials": len(trs), "samples": int(o.size),
                   "mass_from": trs[0].mass_from}
            for n in names:
                if o.size == 0 or any(v is None for v in acc[n]):
                    row[f"{n}_rmse_pct_bw"], row[f"{n}_r"] = float("nan"), float("nan")
                    continue
                p_ = np.concatenate(acc[n])
                row[f"{n}_rmse_pct_bw"] = 100.0 * float(np.sqrt(np.mean((p_ - o) ** 2)))
                row[f"{n}_r"] = g4.pearson(p_, o)
            rows.append(row)
    per = pd.DataFrame(rows)
    res = {
        "what": "G4 GRF RMSE attainability: oracles with the plates' stance timing (population template, leave-one-"
                "trial-out patient template, and the patient template scaled to each stance's plate peak) on the "
                "rule's scored samples; rows primary (every series) and no_low_peak (the secondary row of G4 rule "
                "section 2 (b))",
        "pass_line_pct_bw": g4.GRF_PASS, "kill_line_pct_bw": g4.GRF_KILL,
        "stance_min_s": STANCE_MIN_S, "n_stance": N_STANCE, "band_n": force.BAND_N, "low_peak_bw": 0.5,
        "patients_dev": len(persons), "patients_dev_sha256": _person_hash(persons),
        "trials": len(trials), "series_scored": int(sum(r["samples"] > 0 for r in series)),
        "series_low_peak": int(sum(r["low_peak"] for r in series if r["samples"] > 0)),
        "samples_low_peak": int(sum(r["samples"] for r in series if r["low_peak"] and r["samples"] > 0)),
        "stances": int(sum(len(v) for v in st.values())),
        "stances_complete": int(sum(c for v in st.values() for _, _, c in v)),
        "scored_sample_counts": dict(scored_counts),
        "patients_without_patient_template": no_tmpl,
        "rows": {}, "stats": dict(stats),
    }
    for row_name in ("primary", "no_low_peak", "complete_stances_only"):
        d = per[(per["row"] == row_name) & (per["samples"] > 0)]
        sub = d[d["patient_oracle_rmse_pct_bw"].notna()]
        r = {"patients": int(len(d)), "samples": int(d["samples"].sum()),
             "patients_with_patient_template": int(len(sub))}
        for n in names:
            r[f"{n}_rmse"] = _person_stat(d[f"{n}_rmse_pct_bw"], np.mean)
            r[f"{n}_r"] = _person_stat(d[f"{n}_r"], np.mean)
            r[f"{n}_at_most_pass"] = int((d[f"{n}_rmse_pct_bw"] <= g4.GRF_PASS).sum())
            if n != "newton":                        # grf_r_gain of the oracle over the floor, same samples
                r[f"{n}_r_gain"] = _person_stat(d[f"{n}_r"] - d["newton_r"], np.mean)
        r["newton_rmse_patient_template_subset"] = _person_stat(sub["newton_rmse_pct_bw"], np.mean)
        r["best_oracle_rmse"] = min(r[f"{n}_rmse"]["point"] for n in names[1:])
        r["attainable_by_an_oracle"] = bool(r["best_oracle_rmse"] <= g4.GRF_PASS)
        res["rows"][row_name] = r
    res["time_s"] = round(time.time() - t0, 1)
    os.makedirs(out, exist_ok=True)
    common.write_csv(per, os.path.join(out, "per_patient.csv"))
    common.write_csv(pd.DataFrame(series), os.path.join(out, "per_series.csv"))
    _write_json(os.path.join(out, "results.json"), res)
    md = ["# G4 GRF RMSE line: attainability reference", "",
          "Written by scripts/close_gate_items.py (part g4grf); every number below is read from results.json and the "
          "CSV files of this folder. Development data only (the 40 non-lockbox stroke patients of "
          "stroke_gait_vancriekinge); no lockbox record or row read; no model.", "",
          f"Scored samples: the rule's (plate-valid samples where the Newton floor is defined). A stance: a run of at "
          f"least {STANCE_MIN_S} s of plate-valid samples above the unloaded band ({force.BAND_N} N over m g); "
          f"complete when bounded by plate-valid unloaded samples. Trials {res['trials']}, scored series (trial, foot) "
          f"{res['series_scored']}, of them with a scored peak below 0.5 BW {res['series_low_peak']} "
          f"({res['samples_low_peak']} samples); stances {res['stances']}, complete {res['stances_complete']}. Every "
          "oracle takes each stance's timing from the plates (truth timing) and predicts 0 BW outside stances: "
          "population, the mean complete stance curve of every other patient (both feet); patient, the mean "
          "complete stance curve of the same patient's other trials (same foot, else the other foot); patient "
          "scaled, the patient curve scaled to each stance's own plate peak (truth amplitude as well). The patient "
          "oracles need a complete stance in another trial of the patient; patients without one (left out of "
          f"those two rows): {', '.join(no_tmpl) or 'none'}.", ""]
    sc = res["scored_sample_counts"]
    md += [f"Templates are built from complete stances only (whatever their peak, so the low-peak series of complete "
           f"timing enter them); every stance, complete or not, is predicted by the template stretched over its "
           f"run, so an incomplete stance (cut by the trial start, the trial end or a plate edge) gets a full-stance "
           f"shape squeezed into its truncated run. Scored samples of the primary row {sc['scored_samples']}, of "
           f"them inside a stance {sc['scored_in_stance']} and inside an incomplete stance "
           f"{sc['scored_in_incomplete_stance']} (per series in per_series.csv). The row complete_stances_only "
           "leaves those out: a diagnostic, not the rule's samples, deciding nothing.", ""]
    tab = []
    for row_name, r in res["rows"].items():
        for n in names:
            x = r[f"{n}_rmse"]
            tab.append({"row": row_name, "predictor": n, "RMSE percent BW": _f(x["point"], 2),
                        "interval": _iv(x["interval"], 2), "r": _f(r[f"{n}_r"]["point"], 3),
                        "r gain over the floor": _f(r[f"{n}_r_gain"]["point"], 3) if n != "newton" else "",
                        "patients": x["n_persons"], f"patients at most {g4.GRF_PASS:g}": r[f"{n}_at_most_pass"]})
    md += _md_table(tab, ["row", "predictor", "RMSE percent BW", "interval", "r", "r gain over the floor", "patients",
                          f"patients at most {g4.GRF_PASS:g}"])
    md += ["", "Mean over patients of per-patient pooled values, person bootstrap of the rule (2,000 resamples, seed "
               f"{g4.SEED}). Newton floor on the patients with a patient template, primary row: "
               f"{_f(res['rows']['primary']['newton_rmse_patient_template_subset']['point'], 2)}.",
           f"- Attainability of grf_rmse at most {g4.GRF_PASS:g} percent BW by an oracle: primary row "
           f"{'shown' if res['rows']['primary']['attainable_by_an_oracle'] else 'not shown'} (best oracle "
           f"{_f(res['rows']['primary']['best_oracle_rmse'], 2)}), secondary row without the low-peak series "
           f"{'shown' if res['rows']['no_low_peak']['attainable_by_an_oracle'] else 'not shown'} (best oracle "
           f"{_f(res['rows']['no_low_peak']['best_oracle_rmse'], 2)}); diagnostic row without the incomplete "
           f"stances (decides nothing): best oracle {_f(res['rows']['complete_stances_only']['best_oracle_rmse'], 2)}.",
           "", f"Run time {res['time_s']} s. Counters: {json.dumps(res['stats'])}."]
    common.write_text(os.path.join(out, "summary.md"), "\n".join(md) + "\n")
    print("[g4grf] " + "; ".join(f"{rn} " + ", ".join(f"{n} {_f(r[f'{n}_rmse']['point'], 2)}" for n in names)
                                 for rn, r in res["rows"].items()))
    return res


# ---------------------------------------------------------------- part g4gain
def part_g4gain(out):
    res = {"what": "dev emg_r_gain of the two phase B ends from their per_person_emg.csv (pairs of "
                   "scripts/eval_g4_dev.py), computed after delta = 0.03 was written; decides nothing",
           "delta": G4_EMG_GAIN_DELTA, "rows": {}}
    for tag, key in (("v1_B_032044", "g4_pp_emg_v1"), ("s2_B_032044", "g4_pp_emg_s2")):
        df = pd.read_csv(_abs(REPORTS[key]), encoding="utf-8")
        g = df["model_r"] - df["template_r"]
        res["rows"][tag] = {"source": REPORTS[key], "source_sha256": common.sha256_file(_abs(REPORTS[key])),
                            "gain": _person_stat(g, np.median), "template_r": _person_stat(df["template_r"], np.median),
                            "model_r": _person_stat(df["model_r"], np.median),
                            "persons_at_least_delta": int((g >= G4_EMG_GAIN_DELTA).sum())}
    os.makedirs(out, exist_ok=True)
    _write_json(os.path.join(out, "results.json"), res)
    md = ["# G4 emg_r_gain on dev, the two phase B ends (dev estimate, decides nothing)", "",
          "Written by scripts/close_gate_items.py (part g4gain) after delta = 0.03 was written into "
          "docs/rules/G4_RULE.md; read from the per_person_emg.csv files named below (the pairs of "
          "scripts/eval_g4_dev.py: 84 trials, the same trials for the template and the model; the frame-level "
          "same-samples rule of the sealed read was not applied there).", ""]
    md += _md_table([{"checkpoint": t, "source": r["source"], "template r": _f(r["template_r"]["point"]),
                      "model r": _f(r["model_r"]["point"]), "emg_r_gain": _f(r["gain"]["point"]),
                      "interval": _iv(r["gain"]["interval"]), "persons": r["gain"]["n_persons"],
                      "persons at least 0.03": r["persons_at_least_delta"]} for t, r in res["rows"].items()],
                    ["checkpoint", "source", "template r", "model r", "emg_r_gain", "interval", "persons",
                     "persons at least 0.03"])
    common.write_text(os.path.join(out, "summary.md"), "\n".join(md) + "\n")
    print("[g4gain] " + "; ".join(f"{t} {_f(r['gain']['point'])} {_iv(r['gain']['interval'])}"
                                   for t, r in res["rows"].items()))
    return res


# ---------------------------------------------------------------- part index
KINDS_COUNTED = ("predictor", "oracle", "simulation")   # "reported" references decide nothing (module docstring)


def _verdict(value, pass_line, higher_is_better):
    """True when a finite value lies on the pass side of pass_line (a missing or non-finite value: False)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return v >= pass_line if higher_is_better else v <= pass_line


def _ref_pass(ref, pass_line, hib):
    """(alone, joint): the reference's value on the pass side of the line, and that together with every joint
    condition of the compound line (label, value, pass line, higher is better); a condition that cannot be
    evaluated (None or non-finite) blocks the joint pass."""
    alone = _verdict(ref["value"], pass_line, hib)
    joint = alone and all(_verdict(v, pl, h) for _, v, pl, h in ref.get("joint", ()))
    return alone, joint


def _line_verdict(refs, pass_line, hib):
    """The attainability verdict of a line over its references (module docstring); reported ones never count."""
    hit = {k: any(_ref_pass(r, pass_line, hib)[1] for r in refs if r.get("kind", "predictor") == k)
           for k in KINDS_COUNTED}
    return ("shown" if hit["predictor"] else "shown by an oracle" if hit["oracle"]
            else "shown in simulation" if hit["simulation"] else "not shown")


def _selfcheck_index():
    """Assertions on the verdict logic, run before every index build (the script has no test file of its own)."""
    assert _verdict(4.0, 5.0, False) and not _verdict(6.0, 5.0, False) and _verdict(0.9, 0.9, True)
    assert not _verdict(None, 5.0, False) and not _verdict(float("nan"), 5.0, False) and not _verdict("x", 1, True)
    share = ("matched share", 0.95, 0.9, True)
    bad = ("matched share", 0.5, 0.9, True)
    assert _ref_pass({"value": 4.0, "joint": [share]}, 5.0, False) == (True, True)
    assert _ref_pass({"value": 4.0, "joint": [bad]}, 5.0, False) == (True, False)
    assert _ref_pass({"value": 4.0, "joint": [("gain", None, 0.03, True)]}, 5.0, False) == (True, False)
    assert _line_verdict([{"value": 4.0, "joint": [bad]}], 5.0, False) == "not shown"
    assert _line_verdict([{"value": 4.0, "kind": "reported"}], 5.0, False) == "not shown"
    assert _line_verdict([{"value": 9.0}, {"value": 4.0, "kind": "simulation"}], 5.0, False) == "shown in simulation"
    assert _line_verdict([{"value": 4.0, "kind": "simulation"}, {"value": 4.0, "kind": "oracle", "joint": [share]}],
                         5.0, False) == "shown by an oracle"
    assert _line_verdict([{"value": 4.0, "kind": "oracle"}, {"value": 4.0, "joint": [share]}], 5.0, False) == "shown"


def part_index(out, prov):
    _selfcheck_index()

    def part(rel):                                       # a part's results.json under the output folder of this run
        p = os.path.join(out, *rel.split("/"))
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    g3r, emg, grf, gain = part("g3/results.json"), part("g4_emg/results.json"), part("g4_grf/results.json"), \
        part("g4_gain_dev/results.json")
    try:                                                 # cited path of this run's outputs (tree-relative when inside)
        rel_out = os.path.relpath(out, TREE).replace(os.sep, "/")
    except ValueError:                                   # another drive
        rel_out = out.replace(os.sep, "/")
    if rel_out.startswith(".."):
        rel_out = out.replace(os.sep, "/")
    v1 = _load_json(REPORTS["g3_dev_v1"])
    s2 = _load_json(REPORTS["g3_dev_s2"])
    pa = _load_json(REPORTS["g3_sim_phase_a"])
    ins = _load_json(REPORTS["g3_insim_v1"])
    sup = _load_json(REPORTS["g3_supervised"])
    g4v1 = _load_json(REPORTS["g4_dev_v1"])
    g4s2 = _load_json(REPORTS["g4_dev_s2"])

    def res_line(d, line, key="point"):
        for r in d.get("results_dev", []):
            if r["line"] == line:
                return r[key] if key in r else r["extra"].get(key)
        return None

    def g4_line(d, line, key="point"):
        for r in d.get("lines", []):
            if r.get("line") == line:
                return r.get(key)
        return None

    def ratio(d):
        m, g = res_line(d, "ic_median_ms_walkway", "margin_model_ms"), res_line(d, "ic_median_ms_walkway",
                                                                               "margin_detector_ms")
        return m / g if (m is not None and g) else None

    rows = []

    def add(rule, line, margin, pass_line, hib, refs):
        verdict = _line_verdict(refs, pass_line, hib)
        for r in refs:
            alone, joint = _ref_pass(r, pass_line, hib)
            rows.append({"rule": rule, "line": line, "margin": margin, "reference": r["name"],
                         "kind": r.get("kind", "predictor"), "value": r["value"],
                         "joint_conditions": "; ".join(f"{lab} {_f(v)} ({'>=' if h else '<='} {pl:g}: "
                                                       f"{_verdict(v, pl, h)})" for lab, v, pl, h in r.get("joint", ())),
                         "pass_side_alone": alone, "pass_side": joint, "source": r["source"],
                         "line_verdict": verdict, "check_passed": verdict != "not shown"})

    def share(v):
        return ("matched share", v, 0.9, True)

    # G3
    fwd = ins["lines"]["forward"]
    sup12 = sup["lines"]["dev12"]["supervised"]
    stride_refs = [
        {"name": "model v1 phase B end, dev14", "value": res_line(v1, "stride_mae_cm"),
         "joint": [share(res_line(v1, "stride_mae_cm", "matched_share"))], "source": REPORTS["g3_dev_v1"]},
        {"name": "supervised IMU baseline, dev12", "value": sup12["stride_mae_cm"]["point"],
         "joint": [share(sup12["stride_mae_cm"]["matched_share"])], "source": REPORTS["g3_supervised"]}]
    if g3r:
        for r in g3r["stride_oracle"]["rows"]:
            if r["line"] == "stride_mae_cm":
                stride_refs.append({"name": f"person stride oracle (own strides of the other passes), {r['set']}",
                                    "value": r["point"], "joint": [share(r["matched_share"])],
                                    "source": f"{rel_out}/g3/stride_oracle.csv",
                                    "kind": "oracle" if r["set"] == "valid_inputs" else "reported"})
    for key, name, kind in (("model", "model v1 phase B end, in-sim forward", "simulation"),
                            ("constant", "constant baseline, in-sim forward", "simulation"),
                            ("model_at_true_ics", "model v1 phase B end at the true ICs (truth timing), in-sim "
                                                  "forward", "reported"),
                            ("gyro_on_truth", "readout on the true body (gyro ICs), in-sim forward", "reported")):
        x = fwd[key]["stride_mae_cm"]
        stride_refs.append({"name": name, "value": x["point"], "joint": [share(x.get("matched_share"))],
                            "source": REPORTS["g3_insim_v1"], "kind": kind})
    add("G3", "stride_mae_cm", "at most 5 cm, matched share at least 0.9", 5.0, False, stride_refs)
    speed_refs = [
        {"name": "model v1 phase B end, dev14", "value": res_line(v1, "speed_mae_m_s"),
         "joint": [share(res_line(v1, "speed_mae_m_s", "matched_share"))], "source": REPORTS["g3_dev_v1"]},
        {"name": "supervised IMU baseline, dev12", "value": sup12["speed_mae_m_s"]["point"],
         "joint": [share(sup12["speed_mae_m_s"]["matched_share"])], "source": REPORTS["g3_supervised"]}]
    for key, name in (("model", "model v1 phase B end, in-sim forward"), ("constant", "constant baseline, in-sim forward")):
        x = fwd[key]["speed_mae_m_s"]
        speed_refs.append({"name": name, "value": x["point"], "joint": [share(x.get("matched_share"))],
                           "source": REPORTS["g3_insim_v1"], "kind": "simulation"})
    add("G3", "speed_mae_m_s", "at most 0.08 m/s, matched share at least 0.9", 0.08, False, speed_refs)
    ic_refs = [
        {"name": "model v1 phase B end, dev14", "value": res_line(v1, "ic_median_ms_walkway"),
         "joint": [("ratio to the detector", ratio(v1), 0.8, False)], "source": REPORTS["g3_dev_v1"]},
        {"name": "model seed 2 phase B end, dev14", "value": res_line(s2, "ic_median_ms_walkway"),
         "joint": [("ratio to the detector", ratio(s2), 0.8, False)], "source": REPORTS["g3_dev_s2"]}]
    if g3r:
        for p in g3r["ic_margin_probe"]:
            counted = "oracle" if p["instrument"] == "insole" else "predictor"
            ic_refs.append({"name": f"{p['instrument']} instrument ({'aligned per trial on the walkway' if counted == 'oracle' else 'raw, one global offset'}), {p['set']}",
                            "value": p["ic_median_ms_walkway"],
                            "joint": [("ratio to the detector", p["ratio"], 0.8, False)],
                            "source": f"{rel_out}/g3/ic_margin_probe.csv",
                            "kind": counted if p["set"] == "valid_inputs" else "reported"})
    add("G3", "ic_median_ms (walkway)", "at most 30 ms, and at most 0.8 times the detector on the ICs both matched",
        30.0, False, ic_refs)
    add("G3", "stride_mape_pct (simulator-only)", "at most 25 percent, matched share at least 0.9", 25.0, False, [
        {"name": "phase A end (simulator-only model), dev14", "value": res_line(pa, "stride_mape_pct"),
         "joint": [share(res_line(pa, "stride_mape_pct", "matched_share"))], "source": REPORTS["g3_sim_phase_a"]}])
    # G4
    grf_refs = [
        {"name": "model v1 phase B end, 38 dev patients", "value": g4_line(g4v1, "grf_rmse"),
         "joint": [("grf_r_gain", g4_line(g4v1, "grf_r_gain"), 0.03, True)], "source": REPORTS["g4_dev_v1"]},
        {"name": "model seed 2 phase B end, 38 dev patients", "value": g4_line(g4s2, "grf_rmse"),
         "joint": [("grf_r_gain", g4_line(g4s2, "grf_r_gain"), 0.03, True)], "source": REPORTS["g4_dev_s2"]}]
    if grf:
        labels = {"population_oracle": "population oracle (plate stance timing, leave one patient out)",
                  "patient_oracle": "patient oracle (plate stance timing, leave one trial out)",
                  "patient_scaled_oracle": "patient oracle scaled to each stance's plate peak"}
        for row_name in ("primary", "no_low_peak", "complete_stances_only"):
            rr = grf["rows"].get(row_name)
            if rr is None:
                continue
            for n, lab in labels.items():
                grf_refs.append({"name": f"{lab}, {row_name} row", "value": rr[f"{n}_rmse"]["point"],
                                 "joint": [("grf_r_gain", (rr.get(f"{n}_r_gain") or {}).get("point"), 0.03, True)],
                                 "source": f"{rel_out}/g4_grf/results.json",
                                 "kind": "oracle" if row_name == "primary" else "reported"})
    add("G4", "grf_rmse", "at most 10 percent BW, with grf_r_gain at least 0.03", 10.0, False, grf_refs)
    add("G4", "grf_r_gain", "at least 0.03", 0.03, True, [
        {"name": "model v1 phase B end, 38 dev patients", "value": g4_line(g4v1, "grf_r_gain"),
         "source": REPORTS["g4_dev_v1"]},
        {"name": "model seed 2 phase B end, 38 dev patients", "value": g4_line(g4s2, "grf_r_gain"),
         "source": REPORTS["g4_dev_s2"]}])
    gains = {t: (gain["rows"][t]["gain"]["point"] if gain and t in gain["rows"] else None)
             for t in ("v1_B_032044", "s2_B_032044")}
    gain_src = f"{rel_out}/g4_gain_dev/results.json"
    emg_refs = [
        {"name": "population template, 18 dev persons (its gain over itself is 0 by construction)",
         "value": g4_line(g4v1, "emg_r", "trivial_baseline"), "joint": [("emg_r_gain", 0.0, 0.03, True)],
         "source": REPORTS["g4_dev_v1"]},
        {"name": "model v1 phase B end, 18 dev persons", "value": g4_line(g4v1, "emg_r"),
         "joint": [("emg_r_gain", gains["v1_B_032044"], 0.03, True)], "source": f"{REPORTS['g4_dev_v1']}; {gain_src}"},
        {"name": "model seed 2 phase B end, 18 dev persons", "value": g4_line(g4s2, "emg_r"),
         "joint": [("emg_r_gain", gains["s2_B_032044"], 0.03, True)], "source": f"{REPORTS['g4_dev_s2']}; {gain_src}"}]
    if emg:
        emg_refs.append({"name": "personal template oracle (leave one trial out), 18 dev persons",
                         "value": emg["personal_r"]["point"],
                         "joint": [("emg_r_gain", emg["gain_personal_minus_template"]["point"], 0.03, True)],
                         "source": f"{rel_out}/g4_emg/results.json", "kind": "oracle"})
    add("G4", "emg_r", "at least 0.6, with emg_r_gain at least 0.03", 0.6, True, emg_refs)
    gain_refs = [{"name": "model v1 phase B end over the template, 18 dev persons", "value": gains["v1_B_032044"],
                  "source": gain_src},
                 {"name": "model seed 2 phase B end over the template, 18 dev persons", "value": gains["s2_B_032044"],
                  "source": gain_src}]
    if emg:
        gain_refs.append({"name": "personal template oracle (leave one trial out) over the population template",
                          "value": emg["gain_personal_minus_template"]["point"],
                          "source": f"{rel_out}/g4_emg/results.json", "kind": "oracle"})
    add("G4", "emg_r_gain", "at least 0.03 (delta)", 0.03, True, gain_refs)
    tab = pd.DataFrame(rows)
    common.write_csv(tab, os.path.join(out, "attainability.csv"))
    per_line = tab.drop_duplicates(["rule", "line"])[["rule", "line", "margin", "line_verdict"]].to_dict("records")
    _write_json(os.path.join(out, "_provenance.json"), prov)
    md = ["# Gate rules: measurements for the closure of OPEN items and the attainability check", "",
          "Written by scripts/close_gate_items.py (task GR2, 2026-10-05; attainability standard revised the same "
          "evening on review). Development data and existing reports only; no lockbox, EVAL, CODY or care_pd record "
          "read; no model run; CPU only. Provenance in _provenance.json "
          f"(MD_FM HEAD {prov['md_fm_head'][:7]}, MoveAtlas HEAD {prov['moveatlas_head'][:7]}, lockbox read log "
          f"unchanged: {prov.get('lockbox_reads_unchanged')}).", "",
          "Parts: g3/ (input validity, dev rows on three sets, IC-margin probe, stride oracle), g4_emg/ (EMG margin "
          "reference), g4_grf/ (GRF RMSE reference), g4_gain_dev/ (dev emg_r_gain of the two phase B ends).", "",
          "## Attainability of every success margin (G3, G4)", "",
          "Kinds of reference: predictor (a model, baseline or instrument under the line's own rules, on the line's "
          "data); oracle (the line's data with partial truth information the line does not give a predictor: the "
          "plates' stance timing or amplitude, a person's own EMG or own walkway strides from its other trials or "
          "passes, an instrument aligned per trial on the reference; never the target itself); simulation (a "
          "predictor, the model or a baseline, in the model's synthetic domain); reported (decides nothing: the "
          "line's readout on the truth itself, a model given the truth's timing in simulation, and every row of a "
          "set other than the rule's primary row). A reference lands on the pass side of a compound line only when "
          "every condition of the line holds jointly (grf_rmse with grf_r_gain, emg_r with emg_r_gain, the IC line "
          "with its margin, the stride, speed and MAPE lines with a matched share of at least 0.9); a condition that "
          "cannot be evaluated blocks it. A margin is shown when a predictor lands on its pass side, shown by an "
          "oracle when only an oracle does, shown in simulation when only a simulation predictor does, not shown "
          "otherwise; the check passes for a margin unless it is not shown. The matched-share floor of G3 is "
          "checked inside each line it conditions. G1: no check computed (G1 rule section 11 item 11).", ""]
    md += _md_table(per_line, ["rule", "line", "margin", "line_verdict"])
    md += ["", "Every reference (attainability.csv):", ""]
    md += _md_table([{**{k: r[k] for k in ("rule", "line", "reference", "kind", "source")},
                      "value": _f(r["value"]), "joint conditions": r["joint_conditions"],
                      "pass side alone": r["pass_side_alone"], "pass side": r["pass_side"]} for r in rows],
                    ["rule", "line", "reference", "kind", "value", "joint conditions", "pass side alone", "pass side",
                     "source"])
    common.write_text(os.path.join(out, "summary.md"), "\n".join(md) + "\n")
    print("[index] " + "; ".join(f"{r['rule']} {r['line']}: {r['line_verdict']}" for r in per_line))
    return per_line


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default=",".join(PARTS), help=f"comma-separated subset of {PARTS}")
    ap.add_argument("--out", default=OUT, help="output folder (default reports/gate_rules)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    bad = sorted(set(parts) - set(PARTS))
    if bad:
        raise SystemExit(f"unknown parts {bad}; choose among {PARTS}")
    out = os.path.abspath(a.out)
    paths.assert_not_moveatlas(out)
    try:
        under_docs = os.path.commonpath([out, os.path.join(TREE, "docs")]) == os.path.join(TREE, "docs")
    except ValueError:                                   # another drive: not under docs/
        under_docs = False
    if under_docs:
        raise SystemExit("this script never writes under docs/")
    log = common.READ_LOG
    before = common.sha256_file(log) if os.path.exists(log) else None
    prov_path = os.path.join(out, "_provenance.json")
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
    prov.setdefault("runs", [])
    t0 = time.time()
    done = {}
    for p in parts:
        if p == "g3":
            done[p] = part_g3(os.path.join(out, "g3"))
        elif p == "g4emg":
            done[p] = part_g4emg(os.path.join(out, "g4_emg"))
        elif p == "g4grf":
            done[p] = part_g4grf(os.path.join(out, "g4_grf"))
        elif p == "g4gain":
            done[p] = part_g4gain(os.path.join(out, "g4_gain_dev"))
    after = common.sha256_file(log) if os.path.exists(log) else None
    if before != after:
        raise SystemExit("reports/gates/lockbox_reads.csv changed during the run")
    files = ["mdfm/gates/g3.py", "mdfm/gates/g4.py", "mdfm/gates/common.py", "scripts/close_gate_items.py",
             *(f"docs/rules/{g}_RULE.md" for g in RULES), *REPORTS.values()]
    prov.update({
        "script": "scripts/close_gate_items.py", "md_fm_head": _git(["rev-parse", "HEAD"], TREE),
        "moveatlas_head": _git(["rev-parse", "HEAD"], paths.MOVEATLAS, safe=paths.MOVEATLAS),
        "moveatlas_status_lines": len([x for x in _git(["status", "--porcelain"], paths.MOVEATLAS,
                                                        safe=paths.MOVEATLAS).splitlines() if x.strip()]),
        "files_sha256": {f: common.sha256_file(_abs(f)) for f in files if os.path.exists(_abs(f))},
        "lockbox_reads_sha256": after, "lockbox_reads_unchanged": before == after,
        "python": platform.python_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    })
    prov["runs"].append({"at": datetime.datetime.now().isoformat(timespec="seconds"), "parts": parts,
                         "time_s": round(time.time() - t0, 1)})
    os.makedirs(out, exist_ok=True)
    if "index" in parts:
        part_index(out, prov)
    else:
        _write_json(prov_path, prov)
    print(f"done in {time.time() - t0:.1f} s; lockbox read log unchanged: {before == after}")


if __name__ == "__main__":
    main()
