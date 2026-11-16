"""Gate G3 harness, dev dry run (plan task B4.12; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md 2.7; harness
mdfm/gates/g3.py; rule docs/rules/G3_RULE.md, DRAFT).

Runs the G3 lines on the 14 weargait_pd dev controls of mdfm.data.sources.folds (tasks SelfPace and
HurriedPace) without a model: the walkway itself as a predictor (zero on the walkway lines; against the
insole reference, the floor the two references leave), the population baseline (per task the
person-weighted mean stride and pass speed of the 54 training-fold controls) and the shank-gyroscope
detector (mdfm.readouts.gait.shank_gyro_ic) with its constant offsets fitted on the same dev persons. The
model column stays 'not evaluated'. The insole IC row (the second reference, OD12 closed on 2026-10-05) is reported and
decides nothing (g3.gate_results marks it decides False): its verdict cell in g3.md reads REPORTED, 'reported (does not
decide)', and g3.json keeps its computed verdict with unchanged keys. No lockbox row or record is read: the target reader keeps the rows of
the dev and training persons only and asserts their lockbox column.

Also the readout self-check of mdfm.gates.g3.readout_check on the first 20 forward walks of the AMASS dev body
store (readout speed against factory.gait, the shank detector against the truth body's heel strikes and
contact labels, the lag of the contact labels after the heel strike); skipped and reported as not run when
the dev body store is not built. --walks open reads the forward walks of the open body store's dev folds instead
(mdfm.gates.g3.open_dev_walks; step M4 of docs/MODEL_OPEN_PORT_PLAN.md), with the open body's rest skeletons.

Reads, read-only: reports/weargait_targets/ (events.csv, cache/insole_events.csv, trials.csv, summary.json),
the canonical IMU records of the dev controls through mdfm.data.canonical (MoveAtlas, flags left at False),
reports/lockbox/lockbox_persons.csv, the AMASS dev body store and the SMPL+H model. Writes <out> (default
reports/gates/g3/dev_dry_run/): g3.json, g3.md, per_person.csv, offsets.csv, side_check.csv,
readout_check.csv, _provenance.json.

--register-rule --approved-on YYYY-MM-DD is the owner's act (mdfm.gates.common.register_rule): it records the
SHA-256 of docs/rules/G3_RULE.md once the owner has set its status line to approved. A session never runs it.

Run: <conda>/envs/mdfm/python.exe scripts/gate_g3.py
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)
from _common import write_provenance  # noqa: E402

LINE_ORDER = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "ic_median_ms_insole")
PREDICTORS = ("walkway", "population", "shank_gyro", "shank_gyro_side_flags_out")
REPORTED = "reported (does not decide)"


def verdict_text(verdict, extra):
    """The verdict cell of a GateResult row in g3.md: REPORTED for a row marked decides False (the insole IC row, the
    second IC reference: docs/rules/G3_RULE.md section 6, revision 2026-10-05), whose computed verdict stays in
    g3.json only; the row's verdict otherwise."""
    return REPORTED if (extra or {}).get("decides", True) is False else verdict


def gate_line_row(r):
    """The g3.md row of a GateResult of the dry run (no model): line, unit, pass, kill, model n/a, trivial baseline,
    its 95 % interval (the detector's for an IC row), baseline persons and verdict_text."""
    iv = r.extra.get("detector_interval") if r.line.startswith("ic_") else r.extra.get("baseline_interval")
    return (f"| {r.line} | {r.unit} | {fmt(r.pass_line)} | {fmt(r.kill_line)} | n/a | "
            f"{fmt(r.trivial_baseline)} | {fmt_iv(iv)} | {r.n_persons} | {verdict_text(r.verdict, r.extra)} |")


def reported_note(results):
    """The sentence under the gate-line table naming the rows that are reported and decide nothing ('' when none)."""
    rows = [r.line for r in results if verdict_text(r.verdict, r.extra) == REPORTED]
    if not rows:
        return ""
    return (f"Rows marked '{REPORTED}' ({', '.join(rows)}): the second initial-contact reference, scored with the "
            "same statistic, interval and margin and reported beside the walkway row; it decides nothing "
            "(docs/rules/G3_RULE.md section 6, revision 2026-10-05). g3.json keeps its computed verdict with the "
            "extra keys decides (false) and role.")


def gate_line_table(results, ic_margin, primary):
    """The g3.md lines of the section 'Gate lines (no model)': heading, the table of gate_line_row, and the paragraph
    under it (trivial baselines, the deciding IC reference `primary`, the margin ic_margin, reported_note)."""
    md = ["## Gate lines (no model)\n",
          "| line | unit | pass | kill | model | trivial baseline | baseline 95 % interval | baseline persons "
          "| verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [gate_line_row(r) for r in results]
    note = reported_note(results)
    md.append("\nTrivial baselines: the population baseline for stride, speed and stride MAPE (the simulator-only "
              "line); the shank-gyroscope detector for initial contact (its margin: a model must score at most "
              f"{ic_margin} times its value on the ICs both match). Initial-contact primary reference: "
              f"{primary}, whose row decides the IC line (OD12, closed 2026-10-05)."
              + (f" {note}" if note else "") + "\n")
    return md


def fin(x):
    """None for a missing or non-finite number, else a float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    v = fin(x)
    if v is None:
        return "inf" if x is not None and isinstance(x, float) and math.isinf(x) else "n/a"
    return f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    if not iv:
        return "n/a"
    return f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    """JSON-friendly copy: non-finite floats to None, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    p.add_argument("--out", default=os.path.join(TREE, "reports", "gates", "g3", "dev_dry_run"))
    p.add_argument("--walks", default="amass", choices=("amass", "open", "open_first"),
                   help="the dev walks of the readout self-check: the AMASS dev body store (v1) or the open body "
                        "store's dev folds (step M4, mdfm.gates.g3.open_dev_walks)")
    p.add_argument("--register-rule", action="store_true",
                   help="the owner's act: register docs/rules/G3_RULE.md once its status line reads approved")
    p.add_argument("--approved-on", default=None, help="YYYY-MM-DD of the owner's approval (with --register-rule)")
    a = p.parse_args()
    if a.register_rule:
        from mdfm.gates import common, g3
        print(common.register_rule(g3.GATE, g3.RULE_PATH, a.approved_on))
        return
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    os.environ["MOVEATLAS_ROOT"] = a.moveatlas
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import json

    import pandas as pd

    from mdfm import paths
    from mdfm.data import sources
    from mdfm.gates import common, g3
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    summ, tables, offsets, tg, tr, base, preds = g3.dev_run()
    dev, train = list(tg.persons), list(tr.persons)
    # side check of the shank detector, and a diagnostic row without the flagged trial feet
    shank = next(p for p in preds if p.name == "shank_gyro")
    side = g3.side_check(shank, g3.reference_ics(tg, "walkway"), offsets[("shank_gyro", "walkway")][0])
    flagged = side[side["flagged"]]
    diag_name = "shank_gyro_side_flags_out"
    diag = g3.Predictions(diag_name, shank.ics, covered=frozenset(set(shank.covered)
                                                                 - set(zip(flagged["trial_id"], flagged["foot"]))))
    diag_summ, _ = g3.evaluate(tg, [diag], {(diag_name, r): offsets[("shank_gyro", r)][0] for r in g3.REFERENCES})
    summ_all = {**summ, **diag_summ}
    if set(dev) != set(g3.controls("dev")) or len(dev) != 14:
        raise SystemExit(f"dev set is not the 14 dev controls: {len(dev)}")
    if tg.counts["lockbox_rows"] or tr.counts["lockbox_rows"]:
        raise SystemExit("a lockbox row was kept")
    results = g3.gate_results(summ, tables, model=None, train_persons_sha256=sources.person_hash(train))
    # readout self-check on the dev walks of the body store (no weargait person): AMASS's (v1) or the open store's
    try:
        walks = (g3.open_dev_walks(draw="seeded") if a.walks == "open" else
                 g3.open_dev_walks(draw="first") if a.walks == "open_first" else g3.amass_dev_walks())
        rc_table, rc = g3.readout_check(walks)
        rc["takes"] = [w["take"] for w in walks]
        rc["walks"] = a.walks
        common.write_csv(rc_table, os.path.join(a.out, "readout_check.csv"))
    except FileNotFoundError as e:
        rc_table, rc = None, {"not_run": str(e)}

    with open(g3.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())
    registered = [r for r in common.registry() if r["gate"] == g3.GATE]
    reads = [r for r in common.read_log() if r["gate"] == g3.GATE] if os.path.exists(common.READ_LOG) else []
    with open(os.path.join(g3.TARGETS_DIR, "_provenance.json"), encoding="utf-8") as f:
        targets_prov = json.load(f)
    cell_cm, rule_version = g3.target_meta()

    # per-person table and offsets
    rows = []
    for name, lines in summ_all.items():
        for line, st in lines.items():
            for pid, v in st["per_person"].items():
                rows.append((name, line, pid, fin(v)))
    per_person = pd.DataFrame(rows, columns=["predictor", "line", "person_id", "value"])
    off = pd.DataFrame([(n, r, 1000.0 * o, k) for (n, r), (o, k) in sorted(offsets.items())],
                       columns=["predictor", "reference", "offset_ms", "n_matched"])
    common.write_csv(per_person, os.path.join(a.out, "per_person.csv"))
    common.write_csv(off, os.path.join(a.out, "offsets.csv"))
    common.write_csv(side, os.path.join(a.out, "side_check.csv"))

    lines = {name: {line: {k: v for k, v in st.items() if k != "per_person"} for line, st in s.items()}
             for name, s in summ_all.items()}
    uncovered = summ["shank_gyro"]["ic_median_ms_walkway"]["persons_uncovered"]
    out = {
        "gate": g3.GATE, "run": "dev_dry_run", "model": None,
        "rule": {"path": common.rel(g3.RULE_PATH), "status": status[0], "registered_versions": len(registered)},
        "lockbox_reads_logged_for_g3": len(reads),
        "sets": {"dev": dev, "dev_sha256": sources.person_hash(dev), "train": train,
                 "train_sha256": sources.person_hash(train), "tasks": list(g3.TASKS),
                 "lockbox_rows_kept": tg.counts["lockbox_rows"] + tr.counts["lockbox_rows"]},
        "targets": {"rule_version": rule_version, "cell_cm": cell_cm,
                    "moveatlas_head": targets_prov.get("moveatlas_head"), "dev_counts": tg.counts,
                    "train_counts": tr.counts},
        "constants": {"stride_match_s": g3.STRIDE_MATCH_S, "ic_match_s": g3.IC_MATCH_S, "ic_margin": g3.IC_MARGIN,
                      "min_matched_for_offset": g3.MIN_MATCHED_FOR_OFFSET, "seed": g3.SEED, "boot_n": g3.BOOT_N,
                      "primary_ic_reference": g3.PRIMARY_REFERENCE, "lines": g3.LINES},
        "population_baseline": {t: {"stride_m": s, "speed_m_s": v} for t, (s, v) in base.items()},
        "offsets_ms": {f"{n}|{r}": {"offset_ms": 1000.0 * o, "n_matched": k} for (n, r), (o, k) in offsets.items()},
        "shank_gyro_persons_uncovered": uncovered,
        "shank_side_check": {"trial_feet": int(len(side)),
                             "flagged": flagged[["trial_id", "foot", "person_id", "n", "same",
                                                 "other"]].to_dict("records")},
        "lines": lines,
        "results": [r.to_dict() for r in results],
        "readout_check": rc,
        "runtime_s": round(time.time() - t_start, 1),
    }
    out = clean(out)

    # markdown
    md = []
    md.append("# G3 harness, dev dry run (plan task B4.12)\n")
    md.append("Generated by scripts/gate_g3.py from mdfm/gates/g3.py; every number below is read from g3.json, "
              "per_person.csv, offsets.csv and side_check.csv of this directory. MoveAtlas commit in "
              "_provenance.json; targets "
              f"from reports/weargait_targets/ (rule {rule_version}, built against "
              f"{targets_prov.get('moveatlas_head', 'n/a')[:7]}).\n")
    md.append(f"Rule docs/rules/G3_RULE.md: status '{status[0]}', registered versions {len(registered)}. "
              f"Lockbox reads logged for G3: {len(reads)}. No model: the model column is not evaluated.\n")
    md.append("## Sets\n")
    md.append(f"- Dev: {len(dev)} weargait_pd controls of mdfm.data.sources.folds (SHA-256 "
              f"{sources.person_hash(dev)}): {', '.join(d.split(':', 1)[1] for d in dev)}.")
    md.append(f"- Training fold (population baseline only): {len(train)} controls (SHA-256 "
              f"{sources.person_hash(train)}).")
    md.append(f"- Tasks {', '.join(g3.TASKS)}. Lockbox rows kept from events.csv, insole_events.csv and trials.csv: "
              f"{out['sets']['lockbox_rows_kept']}.\n")
    md.append("| set | trials | walkway ICs (Walk) | insole ICs (Walk, offset removed) | trials with an insole offset "
              "| walkway strides | walkway passes |")
    md.append("|---|---|---|---|---|---|---|")
    for nm, c in (("dev", tg.counts), ("training fold", tr.counts)):
        md.append(f"| {nm} | {c['trials']} | {c['walkway_ic']} | {c['insole_ic']} | {c['trials_with_offset']} | "
                  f"{c['strides']} | {c['passes']} |")
    md.append("")
    md += gate_line_table(results, g3.IC_MARGIN, g3.PRIMARY_REFERENCE)
    md.append("## Every predictor\n")
    md.append("| predictor | line | point | 95 % interval | matched share | reference events | persons | offset (ms) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name in PREDICTORS:
        for line in LINE_ORDER:
            st = summ_all.get(name, {}).get(line)
            if st is None:
                continue
            md.append(f"| {name} | {line} | {fmt(st['point'])} | {fmt_iv(st['interval'])} | "
                      f"{fmt(st['matched_share'], 3)} | {st['n_events']} | {st['n_persons']} | "
                      f"{fmt(st.get('offset_ms'), 1) if 'offset_ms' in st else ''} |")
    md.append("")
    md.append("Offsets (dev fit, median of predicted minus reference over the nearest predictions within "
              f"{g3.IC_MATCH_S} s): " + "; ".join(f"{r.predictor} against {r.reference} {r.offset_ms:.1f} ms "
                                                  f"(n {r.n_matched})" for r in off.itertuples()) + ".\n")
    ic = summ["shank_gyro"]["ic_median_ms_walkway"]
    md.append(f"Shank-gyroscope coverage: dev controls whose records hold no shank gyroscope: {len(uncovered)} "
              f"({', '.join(u.split(':', 1)[1] for u in uncovered) or 'none'}); their "
              f"{ic['n_uncovered']} walkway ICs are left out of the detector's line and scored for a model.\n")
    md.append("Side check of the shank detector (g3.side_check, walkway reference, after its offset): of "
              f"{len(side)} covered trial feet, {len(flagged)} have walkway ICs matched better by the other shank's "
              "detections than by their own shank's: " + ("; ".join(f"{r.trial_id} {r.foot} (own shank {r.same:.3f}, "
                                                                     f"other shank {r.other:.3f}, {r.n} ICs)"
                                    for r in flagged.itertuples()) or "none") + ". The row "
              f"{diag_name} leaves those trial feet out; it is a diagnostic, not a gate line.\n")
    md.append("Population baseline (training fold, person-weighted): " + "; ".join(
        f"{t} stride {s:.4f} m, pass speed {v:.4f} m/s" for t, (s, v) in sorted(base.items())) + ".\n")
    md.append("Walkway strides of the dev controls (m): " + "; ".join(
        f"{t} {len(x)} strides, min {x.min():.3f}, median {x.median():.3f}, max {x.max():.3f}"
        for t, x in ((t, tg.strides.loc[tg.strides['task'] == t, 'length_m']) for t in g3.TASKS)) + ".\n")
    src = {"open": "the open body store's dev folds (g3.open_dev_walks, the seeded draw of g3.open_walk_draw: one walk "
                   "per person across the sources, at least half of its frames on a straight path; the open body's rest "
                   "skeletons)",
           "open_first": "the open body store's dev folds (g3.open_dev_walks draw 'first': OPEN_WALK_RULES, the takes "
                         "sorted by id; the open body's rest skeletons)",
           "amass": "the AMASS dev body store (g3.amass_dev_walks)"}[a.walks]
    md.append(f"## Readout self-check on {'AMASS' if a.walks == 'amass' else 'the open'} dev walks\n")
    if rc_table is None:
        md.append(f"Not run: {rc['not_run']}.\n")
    else:
        hs, ci = rc["shank_gyro_agreement"]["heel_strike"], rc["shank_gyro_agreement"]["contact_ic"]
        lag = rc["contact_lag_ms"]
        md.append(f"{'The' if a.walks == 'open' else 'The first'} {rc['n_walks']} forward walks of {src} (no weargait "
                  "person), truth bodies, shank "
                  "gyroscopes rendered without device noise.\n")
        md.append(f"- Readout speed (median pass_speed over the strides between contact ICs, strides up to 2 s) "
                  f"against factory.gait.speed_and_cadence: median absolute difference "
                  f"{rc['speed_abs_diff_median_m_s']:.3f} m/s, {rc['speed_within_0_05']} of {rc['n_walks']} walks "
                  f"within 0.05 m/s, largest {rc['speed_abs_diff_max_m_s']:.3f} m/s.")
        md.append(f"- Shank-gyroscope detector within {rc['agree_tol_ms']:.0f} ms of the kinematic heel strike "
                  f"(readouts.gait.heel_strikes) after one pooled offset: {hs['share_within']:.3f} of "
                  f"{hs['n_steps']} steady straight steps (offset {hs['offset_ms']:+.0f} ms); of the contact-label "
                  f"ICs: {ci['share_within']:.3f} of {ci['n_steps']} (offset {ci['offset_ms']:+.0f} ms).")
        md.append(f"- Contact-label IC minus the nearest heel strike of the same foot, median per take over its "
                  f"steady straight steps: {lag['min']:.0f} to {lag['max']:.0f} ms over {lag['takes']} takes "
                  f"(median of the takes {lag['median']:.0f} ms); per take in readout_check.csv.\n")
    md.append("Files: g3.json (sets, counts, lines, GateResult rows, readout self-check), per_person.csv (one value "
              "per predictor, line and person), offsets.csv, side_check.csv (per covered trial foot: walkway ICs, "
              "matched share by the same shank's and by the other shank's detections), readout_check.csv (per "
              f"{'AMASS' if a.walks == 'amass' else 'open'} dev walk).\n")
    md.append(f"Runtime: {out['runtime_s']} s.")
    text = "\n".join(md) + "\n"
    common.write(out, a.out, "g3", text)
    write_provenance(a, {"harness": "mdfm/gates/g3.py", "rule_status": status[0],
                         "code_sha256": common.code_sha256([g3.__file__, g3.gait.__file__, os.path.abspath(__file__)])})
    print(text)


if __name__ == "__main__":
    main()
