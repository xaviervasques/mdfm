"""G4 harness dry run on development data, with the model-free baselines (plan task B4.13; spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.7 and 7; rule docs/rules/G4_RULE.md, DRAFT).

No model exists yet, so every gate line is 'not evaluated'; the run writes the numbers the model must
beat, on development data only (never a lockbox row: every reader call keeps its flags at False):
- GRF: the Newton floor of mdfm.gates.g4 (force.com15 on canonical pose x scale plus the raw root track of
  mdfm.data.mocap_raw) against the force plates of the 40 non-lockbox stroke patients, in percent body
  weight; RMSE and Pearson r per person, mean over persons, person-bootstrap 95 percent interval.
- EMG: the template baseline (mean log-envelope per muscle over the step cycle of the gait120 train fold)
  on the gait120 dev fold, level walking; Pearson r per trial and muscle, median per person, median over
  persons; the measured envelope scored as its own prediction (r = 1, a harness self-check).
- Checks: world joints (canonical coords x scale plus the root track) against the raw joints of the TRC
  and C3D files, over every trial read; the post-stroke MAT sub_char rows against the C3D PROCESSING
  parameters of the dev patients (mocap_raw.mat_char order); the origin of each patient's body mass; the
  third mass fallback (mass_from_steps) against the recorded mass of the same patients; the scored plate
  series whose peak stays below 0.5 BW; the persons of each set (dev, train and lockbox) with a recording that holds both streams of its line,
  counted from the index rows alone (g4.covered_persons: no record opens).
Outputs, reports/gates/g4/baselines_dev/: g4.json, g4.md, per_person_grf.csv, per_trial_grf.csv,
per_person_emg.csv, per_trial_emg.csv, template_train.csv, mass.csv, _provenance.json.

Run: <conda>/envs/mdfm/python.exe scripts/gate_g4.py [--quick] [--out DIR]
(--quick: 3 patients, 2 dev and 5 train gait120 persons, written to the scratch folder given by --out.)
Owner only, after approving the rule: scripts/gate_g4.py --register-rule --approved-on YYYY-MM-DD.
"""
import argparse
import collections
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm import paths  # noqa: E402
from mdfm.data import mocap_raw as M  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from mdfm.gates import common  # noqa: E402
from mdfm.gates import g4  # noqa: E402
from _common import moveatlas_head  # noqa: E402

OUT = os.path.join(TREE, "reports", "gates", "g4", "baselines_dev")


def args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=OUT)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--register-rule", action="store_true", help="owner only: register the approved G4 rule")
    p.add_argument("--approved-on", default=None)
    a = p.parse_args()
    paths.assert_not_moveatlas(a.out)
    if a.quick and os.path.abspath(a.out) == os.path.abspath(OUT):
        sys.exit("--quick writes to a scratch folder: pass --out")
    return a


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def root_check(world, raw):
    """Largest |world - raw| (mm) over the joint positions finite in both, and the positions compared."""
    ok = (np.isfinite(world) & np.isfinite(raw)).all(-1)
    return (1000.0 * float(np.abs(world - raw)[ok].max()) if ok.any() else float("nan")), int(ok.sum())


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def mat_check(persons):
    """Per dev patient: the C3D PROCESSING BODYMASS, HEIGHT and LLEGLENGTH of each trial against the MAT
    sub_char row mocap_raw.mat_char reads."""
    rows = []
    for ns in persons:
        mc = M.mat_char(ns)
        for r in M.subject_rows(M.STROKE, ns):
            c, _, _ = M._stroke_c3d(ns, str(r["key"]))
            val = {}
            for k in ("BODYMASS", "HEIGHT", "LLEGLENGTH"):
                v = float(np.ravel(c.param("PROCESSING", k, [np.nan]))[0])
                val[k] = v if v > 0 else np.nan
            rows.append({"person_id": ns, "key": r["key"], "c3d_mass_kg": val["BODYMASS"],
                         "mat_weight_kg": mc["Weight"], "c3d_height_mm": val["HEIGHT"], "mat_height_mm": mc["Height"],
                         "c3d_leg_mm": val["LLEGLENGTH"], "mat_leg_mm": mc["LegLength"]})
    d = pd.DataFrame(rows)

    def agree(a, b, tol):
        known = d[a].notna()
        return int((np.abs(d[a] - d[b]) <= tol)[known].sum()), int(known.sum())

    (m_ok, m_n), (h_ok, h_n), (l_ok, l_n) = (agree("c3d_mass_kg", "mat_weight_kg", 0.5),
                                             agree("c3d_height_mm", "mat_height_mm", 0.5),
                                             agree("c3d_leg_mm", "mat_leg_mm", 0.5))
    return d, {"trials": int(len(d)), "trials_with_c3d_mass": m_n, "mass_within_0p5_kg": m_ok,
               "height_equal": h_ok, "trials_with_c3d_height": h_n, "leg_equal": l_ok, "trials_with_c3d_leg": l_n}


def steps_mass_check(persons, per_person_grf):
    """The third mass fallback against the recorded mass: per dev patient, the median over its paired trials
    of g4.trial_steps_mass (mass_from_steps), and its ratio to the C3D or MAT mass the harness used.
    Returns (per-patient DataFrame, summary)."""
    rows = []
    for pr in g4.paired_records(g4.STROKE, persons):
        root, valid, _ = M.root_for_paired(pr)
        world = g4.world_joints(pr.streams["pose3d"], root, valid)
        rows.append({"person_id": pr.person_id, "key": pr.key,
                     "steps_mass_kg": g4.trial_steps_mass(pr, world, float(pr.streams["pose3d"]["fps"]))})
    per = pd.DataFrame(rows).groupby("person_id")["steps_mass_kg"].median().reset_index()
    rec = per.merge(per_person_grf[["person_id", "mass_kg", "mass_from"]], how="inner")
    rec = rec[rec["mass_from"].isin(["c3d", "mat"])]
    ratio = (rec["steps_mass_kg"] / rec["mass_kg"]).to_numpy(dtype=np.float64)
    ratio = ratio[np.isfinite(ratio)]
    summ = {"patients_with_recorded_mass": int(len(rec)), "patients_with_steps_mass": int(len(ratio)),
            "ratio_median": r4(np.median(ratio)) if len(ratio) else None,
            "ratio_min": r4(ratio.min()) if len(ratio) else None, "ratio_max": r4(ratio.max()) if len(ratio) else None,
            "patients_off_by_more_than_10_percent": int(((ratio < 0.9) | (ratio > 1.1)).sum())}
    return per, summ


def plate_peak_check(per_trial_grf, below=0.5):
    """Scored (trial, foot) plate series whose plate peak stays below `below` BW (a partial strike, or a
    load that is not the whole foot's), with their patients and scored samples."""
    s = per_trial_grf[per_trial_grf["samples"] > 0]
    low = s[s["obs_peak_bw"] < below]
    return {"scored_trial_feet": int(len(s)), "peak_below_bw": below, "trial_feet": int(len(low)),
            "patients": int(low["person_id"].nunique()), "samples": int(low["samples"].sum()),
            "samples_scored": int(s["samples"].sum())}


def grf_trial_table(trials):
    rows = []
    for tr in trials:
        pred = g4.newton_predict(tr)
        for j, foot in enumerate(("L", "R")):
            ok = tr.obs_valid[:, j] & np.isfinite(pred[:, j])
            o, p = tr.obs[ok, j], pred[ok, j]
            rows.append({"person_id": tr.person, "key": tr.key, "foot": foot, "samples": int(ok.sum()),
                         "obs_peak_bw": r4(o.max()) if ok.any() else None,
                         "newton_rmse_pct_bw": r4(100 * np.sqrt(np.mean((p - o) ** 2))) if ok.any() else None,
                         "newton_r": r4(g4.pearson(p, o))})
    return pd.DataFrame(rows)


def md_table(df, cols=None):
    cols = list(df.columns) if cols is None else cols
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else
                                     (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main():
    a = args()
    if a.register_rule:
        if not a.approved_on:
            sys.exit("--register-rule needs --approved-on YYYY-MM-DD")
        row = common.register_rule(g4.GATE, g4.RULE_PATH, a.approved_on)
        print(json.dumps(row, indent=1))
        return
    t0 = time.time()
    sys.stdout.reconfigure(encoding="utf-8")
    head = moveatlas_head(paths.MOVEATLAS)
    stroke = g4.stroke_patients()
    dev, train = g4.gait120_persons("dev"), g4.gait120_persons("train")
    if a.quick:
        stroke, dev, train = stroke[:3], dev[:2], train[:5]
    if set(dev) & set(train):
        raise SystemExit("gait120 dev and train folds overlap")
    stats = collections.Counter()

    # GRF: Newton floor on the dev patients
    gtr = g4.grf_trials(stroke, stats=stats)
    per_person_grf = g4.grf_person_table(gtr, {"newton": g4.newton_predict})
    per_trial_grf = grf_trial_table(gtr)
    stroke_root = [root_check(tr.world, M.raw_joints15(M.STROKE, tr.person, tr.key)) for tr in gtr]
    print(f"GRF: {len(gtr)} trials of {per_person_grf.shape[0]} patients, {time.time() - t0:.0f} s", flush=True)

    # EMG: template from the train fold, scored on the dev fold; root check on every trial read
    etr = {"dev": [], "train": []}
    g120_root = []
    emg_stats = collections.Counter()
    for fold, persons in (("dev", dev), ("train", train)):
        for pr in g4.paired_records(g4.GAIT120, persons, stats=emg_stats):
            etr[fold].append(g4.emg_trial(pr))
            root, valid, _ = M.root_for_paired(pr)
            world = g4.world_joints(pr.streams["pose3d"], root, valid)
            g120_root.append(root_check(world, M.raw_joints15(M.GAIT120, pr.subject_ns, pr.key)))
    template = g4.build_template(etr["train"])
    preds = {"template": lambda tr: g4.template_predict(template, tr.phase), "measured": lambda tr: tr.obs}
    per_trial_emg = g4.emg_trial_table(etr["dev"], preds)
    per_person_emg = g4.emg_person_table(per_trial_emg, list(preds))
    print(f"EMG: {len(etr['dev'])} dev and {len(etr['train'])} train trials, {time.time() - t0:.0f} s", flush=True)

    # set sizes from the index rows alone (no record opens): the persons each line can score
    unscored = sorted(set(stroke) - set(per_person_grf["person_id"]))
    coverage = {}
    sets = (("stroke_dev", g4.STROKE, stroke), ("stroke_lockbox", g4.STROKE, g4.stroke_patients(lockbox=True)),
            ("gait120_dev", g4.GAIT120, dev), ("gait120_train", g4.GAIT120, train),
            ("gait120_lockbox", g4.GAIT120, g4.gait120_persons("lockbox")))
    for name, src, persons in sets:
        coverage[name] = {"persons": len(persons), "with_both_streams": len(g4.covered_persons(src, persons))}

    summ = g4.summary(per_person_grf, per_person_emg, ["newton"], list(preds))
    rule_sha = common.sha256_file(g4.RULE_PATH)
    rule_status = common.rule_status(open(g4.RULE_PATH, encoding="utf-8").read())[0]
    train_hash = S.person_hash(train)
    results = g4.gate_results(summ, None, rule_sha, "", "", train_hash)
    mat_rows, mat_summary = mat_check(sorted(per_person_grf["person_id"]))
    steps_rows, steps_summary = steps_mass_check(stroke, per_person_grf)
    plate_peaks = plate_peak_check(per_trial_grf)
    by_muscle = per_trial_emg.groupby("muscle")["template_r"].median().reindex(list(g4.MUSCLES))
    mass = per_person_grf[["person_id", "mass_kg", "mass_from", "sex", "trials", "samples"]]
    checks = {
        "stroke_root_max_mm": r6(max(x[0] for x in stroke_root)) if stroke_root else None,
        "stroke_root_joint_positions": int(sum(x[1] for x in stroke_root)),
        "gait120_root_max_mm": r6(max(x[0] for x in g120_root)) if g120_root else None,
        "gait120_root_joint_positions": int(sum(x[1] for x in g120_root)),
        "measured_as_prediction_r_min": r4(per_trial_emg["measured_r"].min()),
        "mat_mapping": mat_summary,
        "mass_from": {k: int(v) for k, v in per_person_grf["mass_from"].value_counts().items()},
        "steps_mass_vs_recorded": steps_summary,
        "plate_peaks": plate_peaks,
    }
    out = {"gate": g4.GATE, "run_at": dt.datetime.now().isoformat(timespec="seconds"), "moveatlas_head": head,
           "rule": {"path": common.rel(g4.RULE_PATH), "sha256": rule_sha, "status": rule_status},
           "quick": bool(a.quick),
           "sets": {"stroke_dev_patients": len(stroke), "stroke_patients_scored": int(per_person_grf.shape[0]),
                    "stroke_trials_scored": len(gtr), "stroke_samples_scored": int(per_person_grf["samples"].sum()),
                    "gait120_dev_persons": len(dev), "gait120_dev_trials": len(etr["dev"]),
                    "gait120_train_persons": len(train), "gait120_train_trials": len(etr["train"]),
                    "gait120_train_persons_sha256": train_hash,
                    "gait120_dev_persons_sha256": S.person_hash(dev),
                    "stroke_dev_unscored": unscored, "index_coverage": coverage},
           "pairing": {"stroke": dict(stats), "gait120": dict(emg_stats)},
           "summary": summ, "template_r_by_muscle": {k: r4(v) for k, v in by_muscle.items()},
           "lines": results, "checks": checks,
           "config": {"com_cutoff_hz": g4.COM_CUTOFF_HZ, "min_run_frames": g4.MIN_RUN, "stance_pct": g4.STANCE_PCT,
                      "force_lowpass_hz": g4.LOWPASS_HZ, "n_phase": g4.N_PHASE, "min_emg_run_s": g4.MIN_EMG_RUN_S,
                      "seed": g4.SEED,
                      "boot_n": g4.BOOT_N, "muscles": list(g4.MUSCLES)},
           "runtime_s": round(time.time() - t0, 1)}

    def line(name, key, stat):
        s = summ[name][key]
        lo, hi = s["interval"]
        return f"{s['point']:.4f} [{lo:.4f}, {hi:.4f}] ({stat} over {s['n_persons']} persons)"

    lines_md = md_table(pd.DataFrame([{"line": r.line, "unit": r.unit, "verdict": r.verdict,
                                       "trivial baseline": r4(r.trivial_baseline), "pass": r.pass_line,
                                       "kill": r.kill_line, "persons": r.n_persons} for r in results]))
    md = [
        "# G4 dry run on development data: Newton floor and EMG template (no model)",
        "",
        f"Generated by scripts/gate_g4.py (plan task B4.13) on {out['run_at'][:10]}; every number below is read "
        "from g4.json and the CSV files of this folder. MoveAtlas commit: " + head + ". Rule: "
        f"{out['rule']['path']} (status: {rule_status.rstrip('.')}; SHA-256 {rule_sha[:16]}...). Harness: "
        "mdfm/gates/g4.py; root tracks and body mass: mdfm/data/mocap_raw.py with the in-tree C3D reader "
        "mdfm/data/c3d.py. Development data only: no lockbox row was read." + (" QUICK RUN." if a.quick else ""),
        "",
        "## Gate lines",
        "",
        "No model exists yet: each line is not evaluated, and the trivial baseline is the number a model must "
        "beat (GRF: the Newton floor; the r gain is measured against it, so its baseline is 0; EMG: the template).",
        "",
        lines_md,
        "",
        "## Baselines",
        "",
        "- Newton floor, per-foot vertical GRF RMSE in percent body weight: "
        f"{line('grf_rmse_pct_bw', 'newton', 'mean')}.",
        f"- Newton floor, Pearson r against the plates: {line('grf_r', 'newton', 'mean')}.",
        f"- EMG template, Pearson r of log-envelopes: {line('emg_r', 'template', 'median')}; median r per muscle "
        "over dev trials: " + ", ".join(f"{k} {v:.4f}" for k, v in by_muscle.items()) + ".",
        f"- Measured envelope as its own prediction: {line('emg_r', 'measured', 'median')}; smallest r over "
        f"trials and muscles {checks['measured_as_prediction_r_min']}.",
        "",
        "## Sets",
        "",
        f"- Stroke dev patients: {len(stroke)}; with at least one trial paired with plate force: "
        f"{per_person_grf.shape[0]}; trials {len(gtr)}; scored (frame, foot) samples "
        f"{int(per_person_grf['samples'].sum())}. Pairing counts: {dict(stats)}. Unscored (no recording with a "
        f"pose and a plate force record): {', '.join(unscored) or 'none'}.",
        f"- gait120 dev fold: {len(dev)} persons, {len(etr['dev'])} level-walking trials; template from the train "
        f"fold: {len(train)} persons (SHA-256 {train_hash[:16]}...), {len(etr['train'])} trials. Pairing counts: "
        f"{dict(emg_stats)}.",
        "- Persons with at least one recording holding both streams of their line (stroke pose and plate force; "
        "gait120 level-walking pose and EMG), counted from the index rows alone (no record opened): "
        + "; ".join(f"{k} {v['with_both_streams']} of {v['persons']}" for k, v in coverage.items()) + ".",
        "",
        "## Checks",
        "",
        "- World joints (canonical coords x scale plus the raw root track) against the raw joints of the files: "
        f"stroke C3D, largest difference {checks['stroke_root_max_mm']} mm over "
        f"{checks['stroke_root_joint_positions']} joint positions; gait120 TRC, "
        f"{checks['gait120_root_max_mm']} mm over {checks['gait120_root_joint_positions']} joint positions (the "
        "done-when bound of B4.13 is 1 mm).",
        "- Post-stroke MAT row order (mocap_raw.mat_char, TVC folders in number order) against the C3D "
        "PROCESSING parameters of the dev patients: body mass within 0.5 kg on "
        f"{mat_summary['mass_within_0p5_kg']} of {mat_summary['trials_with_c3d_mass']} trials that carry one; "
        f"height equal on {mat_summary['height_equal']} of {mat_summary['trials_with_c3d_height']}; leg length "
        f"equal on {mat_summary['leg_equal']} of {mat_summary['trials_with_c3d_leg']} (of {mat_summary['trials']} "
        "trials).",
        f"- Body mass origin per patient (spec question 7): {checks['mass_from']} (c3d: PROCESSING:BODYMASS; mat: "
        "sub_char Weight; steps: mass_from_steps).",
        "- The third mass fallback (g4.mass_from_steps, median over a patient's trials) against the recorded mass "
        f"of the same patients: computable on {steps_summary['patients_with_steps_mass']} of "
        f"{steps_summary['patients_with_recorded_mass']} patients with a recorded mass; ratio median "
        f"{steps_summary['ratio_median']}, range {steps_summary['ratio_min']} to {steps_summary['ratio_max']}; "
        f"{steps_summary['patients_off_by_more_than_10_percent']} patients off by more than 10 percent (column "
        "steps_mass_kg of mass.csv).",
        f"- Scored plate series (trial, foot) whose plate peak stays below {plate_peaks['peak_below_bw']} BW: "
        f"{plate_peaks['trial_feet']} of {plate_peaks['scored_trial_feet']}, from {plate_peaks['patients']} "
        f"patients, {plate_peaks['samples']} of {plate_peaks['samples_scored']} scored samples (per_trial_grf.csv, "
        "obs_peak_bw).",
        "",
        "## Newton floor per patient",
        "",
        md_table(per_person_grf.assign(person_id=per_person_grf["person_id"].str.replace("stroke_gait:", ""))),
        "",
        "## EMG template per dev person",
        "",
        md_table(per_person_emg),
        "",
        f"Runtime: {out['runtime_s']} s.",
        "",
    ]
    common.write(out, a.out, "g4", "\n".join(md))
    common.write_csv(per_person_grf, os.path.join(a.out, "per_person_grf.csv"))
    common.write_csv(per_trial_grf, os.path.join(a.out, "per_trial_grf.csv"))
    common.write_csv(per_person_emg, os.path.join(a.out, "per_person_emg.csv"))
    common.write_csv(per_trial_emg, os.path.join(a.out, "per_trial_emg.csv"))
    common.write_csv(pd.DataFrame(template, columns=list(g4.MUSCLES)).assign(phase=np.linspace(0, 1, g4.N_PHASE)),
                     os.path.join(a.out, "template_train.csv"))
    common.write_csv(mass.merge(mat_rows.groupby("person_id")[["mat_weight_kg"]].first().reset_index(), how="left")
                     .merge(steps_rows, how="left"), os.path.join(a.out, "mass.csv"))
    prov = {"script": "gate_g4.py", "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "run_at": out["run_at"], "python": sys.version.split()[0], "quick": bool(a.quick),
            "inputs": ["cache/manifest/manifest_records.csv.gz", "cache/manifest/record_index.csv.gz",
                       "MoveAtlas canonical pose, force and EMG records of stroke_gait_vancriekinge and gait120 "
                       "(mdfm.data.canonical, flags False)",
                       "datasets/pose/stroke_gait_vancriekinge/raw/50_StrokePiG.zip (C3D of the dev patients)",
                       "datasets/pose/stroke_gait_vancriekinge/raw/MAT_normalizedData_PostStrokeAdults_v27-02-23.mat "
                       "(sub_char of the dev patients)",
                       "datasets/emg/gait120/raw/Gait120_*.zip (TRC of the dev and train level-walking trials)",
                       "configs/canonical_skeleton.yaml"]}
    common.write_text(os.path.join(a.out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(f"written {a.out} in {time.time() - t0:.0f} s")
    print(json.dumps({"summary": summ, "checks": checks}, indent=1, default=float))


if __name__ == "__main__":
    main()
