"""G8 dev run, weargait_pd line (task G8R; rule docs/rules/G8_RULE.md, DRAFT, not registered; harness mdfm/gates/g8.py):
MDS-UPDRS items 3.10 to 3.13 of the non-lockbox weargait_pd PD persons, read by a person-folded ridge on frozen tokens,
rounded and clipped to the item scale, with the trivial baselines and the reported oracles; dev only. Every output
carries LABEL: a dev estimate, not a gate read, and no G12 comparison.

Sets (rule section 2): fit on pd_train (T19's proposed PD train fold), score pd_dev (its dev fold); the reported row
pd_pool, 5-fold person cross-fitted over both folds (token probes: both candidate variants, the one chosen on pd_train
flagged, since the pool also scores the pd_train persons that chose it); the reported PD-and-controls rows all_dev (fit
on pd_train and the 54 training-fold controls, score pd_dev and the 14 dev controls) and all_pool. A person enters an
item's line when the item holds a grade and both G3 tasks hold an IMU record (persons.csv lists every person with the
reason it is out).
Lockbox persons are counted from reports/lockbox/lockbox_persons.csv only (g8.lockbox_counts); every request goes through
g8.guard, which refuses them and any id outside weargait_pd before a file opens.

Runs (mdfm environment; CUDA_VISIBLE_DEVICES=-1 for the CPU, the GPU only through scripts/queue_job.py):
  python scripts/eval_g8_dev.py --no-model
      census, baselines (majority, walkway speed) and oracles (part2, others); reports/gates/g8/dev_baselines/
  python scripts/eval_g8_dev.py --checkpoint cache/runs/<run>/<ckpt>.pt --device cpu [--with-controls]
      frozen tokens (tokenizer and encoder only) of the scored persons, the probes and their rows;
      reports/gates/g8/dev_<name>/ and the features cache/runs/g8_features/<name>/features.npz
  python scripts/eval_g8_dev.py --checkpoint ... --from-features      refit from the feature file, no model
  python scripts/eval_g8_dev.py --attainability                       reports/gates/g8/attainability.{json,md}
  python scripts/eval_g8_dev.py --print-jobs                          the queue commands of the GPU extractions

A checkpoint trained on a pd_dev person (or, with --with-controls, a dev control) is refused before any record is read;
a checkpoint trained on a pd_train person keeps its pd_dev and all_dev rows (the probe's fitting persons were seen
without item labels, as the G12 readout's) and loses its pool rows, whose scored persons it trained on.
Writes <out>: g8_dev.json, g8_dev.md, per_row.csv (g8.ROW_COLUMNS), persons.csv, _provenance.json (MoveAtlas HEAD,
script SHA-256, code hash).
"""
import argparse
import json
import math
import os
import sys
import time

if __name__ == "__main__" and ("--no-model" in sys.argv[1:] or "--attainability" in sys.argv[1:]):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # the runs without a model never see the GPU

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

import numpy as np  # noqa: E402

TASK = "G8R"
LABEL = "G8 dev estimate (weargait_pd items 3.10 to 3.13), not a gate read, no G12 comparison"
OUT_ROOT = os.path.join(TREE, "reports", "gates", "g8")
FEATURE_ROOT = os.path.join(TREE, "cache", "runs", "g8_features")
STEM = "g8_dev"
PYTHON = "<conda>/envs/mdfm/python.exe"
BASELINES = ("majority", "speed")
ORACLES = ("part2", "others")
KINDS = {"majority": "trivial", "speed": "predictor (walkway instrument)", "part2": "oracle (patient report)",
         "others": "oracle (the other three items)"}
GPU_ENDS = (("v1_B", "cache/runs/phaseA_root_full/ckpt_B_032044.pt"),
            ("s2_B", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt"),
            ("mfr_s20260928", "cache/runs/g12_mfr_s20260928/ckpt_B_056774.pt"),
            ("mfr_s20261001", "cache/runs/g12_mfr_s20261001/ckpt_B_056774.pt"),
            ("jepa_s20260928", "cache/runs/g12_jepa_s20260928/ckpt_B_047617.pt"),
            ("jepa_s20261001", "cache/runs/g12_jepa_s20261001/ckpt_B_047617.pt"),
            ("s2a_C", "cache/runs/stage2_s2a/ckpt_C_006000.pt"),
            ("s2b_C", "cache/runs/stage2_s2b/ckpt_C_008550.pt"))


# ---------------------------------------------------------------- helpers
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=3):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=3):
    return "" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    if hasattr(o, "to_dict"):
        return clean(o.to_dict())
    return o


def checkpoint_name(path):
    p = os.path.abspath(str(path).replace("/", os.sep))
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"


# ---------------------------------------------------------------- persons and data
def load_sets(with_controls=True):
    """The persons, their items and records, the sets and the walkway speed features (module docstring)."""
    import pandas as pd

    from mdfm.gates import g3, g8
    pd_train, pd_dev, fold_info = g8.pd_folds()
    c_train, c_dev = g8.control_folds() if with_controls else ([], [])
    fold = {**{p: "pd_train" for p in pd_train}, **{p: "pd_dev" for p in pd_dev},
            **{p: "control_train" for p in c_train}, **{p: "control_dev" for p in c_dev}}
    everyone = sorted(fold)
    items = pd.concat([g8.read_items(pd_train + pd_dev, groups=(g8.GROUP_PD,)),
                       g8.read_items(c_train + c_dev, groups=(g8.GROUP_CONTROL,)) if with_controls else
                       pd.DataFrame(columns=["person_id"])], ignore_index=True)
    items = items.set_index("person_id").reindex(everyone)
    rec = g8.record_persons(everyone)
    scored = sorted(p for p in everyone if set(g8.TASKS) <= set(rec[p]))
    tg = g3.load_targets(scored)
    if tg.counts["lockbox_rows"]:
        raise SystemExit("a lockbox target row was kept")
    speed = np.full((len(everyone), 2), np.nan)
    pos = {p: i for i, p in enumerate(everyone)}
    for (p, task), g in tg.passes.groupby(["person_id", "task"]):
        speed[pos[p], g8.TASKS.index(task)] = float(g["speed_m_s"].mean())
    rows = []
    for p in everyone:
        r = items.loc[p]
        reason = ""
        if p not in scored:
            reason = f"no IMU record of {', '.join(sorted(set(g8.TASKS) - set(rec[p])))}"
        elif not np.isfinite([r[i] for i in g8.ITEMS]).any():
            reason = "no grade of items 3.10 to 3.13 in the clinical table"
        rows.append({"person_id": p, "group": "PD" if fold[p].startswith("pd") else "control", "fold": fold[p],
                     "scored": p in scored, "out_reason": reason, "record_tasks": ";".join(rec[p]),
                     **{f"item_{i}": r[i] for i in g8.ITEMS}, **{f"part2_{i}": r[i] for i in g8.PART2},
                     "days_since_part3": r["days_since_part3"], "med_state": g8.med_label(r["med_state"]),
                     "speed_selfpace": speed[pos[p], 0], "speed_hurriedpace": speed[pos[p], 1]})
    persons = pd.DataFrame(rows)
    return {"persons": persons, "fold_info": fold_info, "scored": scored, "targets": tg, "everyone": everyone,
            "pd_train": pd_train, "pd_dev": pd_dev, "c_train": c_train, "c_dev": c_dev}


def census(S):
    """Per set and item: persons graded (and with records), the grade counts; lockbox counts from the split file."""
    from mdfm.gates import g8
    P = S["persons"]
    out = {"lockbox_split_file": g8.lockbox_counts(), "lockbox_index": g8.lockbox_index_counts(), "sets": {}}
    groups = {"pd_train": ("pd_train",), "pd_dev": ("pd_dev",), "control_train": ("control_train",),
              "control_dev": ("control_dev",)}
    for name, folds in groups.items():
        sel = P[P["fold"].isin(folds)]
        d = {"persons": int(len(sel)), "with_records": int(sel["scored"].sum()), "items": {}}
        for i in g8.ITEMS:
            col = sel[f"item_{i}"]
            graded = sel[np.isfinite(col.astype(float))]
            both = graded[graded["scored"]]
            d["items"][i] = {"graded": int(len(graded)), "graded_with_records": int(len(both)),
                             "grades": {str(int(k)): int(v) for k, v in both[f"item_{i}"].value_counts().sort_index().items()}}
        d["days_since_part3_nonzero"] = int((sel["days_since_part3"].fillna(-1) > 0).sum())
        d["days_since_part3_unknown"] = int(sel["days_since_part3"].isna().sum())
        d["med_state"] = {str(k): int(v) for k, v in sel["med_state"].value_counts().items()}
        out["sets"][name] = d
    return out


def set_persons(S):
    """{set: (fit persons, score persons)} of the scored persons, and the pools."""
    sc = set(S["scored"])
    pt, pdv = sorted(set(S["pd_train"]) & sc), sorted(set(S["pd_dev"]) & sc)
    ct, cdv = sorted(set(S["c_train"]) & sc), sorted(set(S["c_dev"]) & sc)
    return {"pd_dev": (pt, pdv), "all_dev": (sorted(pt + ct), sorted(pdv + cdv)),
            "pd_pool": sorted(pt + pdv), "all_pool": sorted(pt + pdv + ct + cdv)}


def item_vectors(P, persons):
    from mdfm.gates import g8
    idx = P.set_index("person_id").loc[list(persons)]
    return {i: idx[f"item_{i}"].to_numpy(np.float64) for i in g8.ITEMS}, idx["group"].to_numpy().astype(str)


def baseline_features(P, persons):
    """{name: X} of the baselines and oracles over persons (rows of persons, in order)."""
    from mdfm.gates import g8
    idx = P.set_index("person_id").loc[list(persons)]
    X = {"speed": idx[["speed_selfpace", "speed_hurriedpace"]].to_numpy(np.float64),
         "part2": idx[[f"part2_{i}" for i in g8.PART2]].to_numpy(np.float64),
         "others": {i: idx[[f"item_{j}" for j in g8.ITEMS if j != i]].to_numpy(np.float64) for i in g8.ITEMS}}
    return X


def majority_rows(model, set_name, persons, groups, items, fit_persons, score_persons):
    import pandas as pd

    from mdfm.gates import g8
    ps = np.asarray(persons).astype(str)
    fi, si = np.isin(ps, list(fit_persons)), np.isin(ps, list(score_persons))
    parts = []
    for i in g8.ITEMS:
        y = np.asarray(items[i], dtype=np.float64)
        m = g8.majority(y[fi])
        n = int(si.sum())
        parts.append(g8.make_rows(model, set_name, "round", "", i, ps[si], groups[si], y[si], np.full(n, m),
                                  np.full(n, m), np.zeros(n, bool)))
    return pd.concat(parts, ignore_index=True)


def majority_pool(model, set_name, persons, groups, items, pool):
    import pandas as pd

    from mdfm.eval import probes
    pool = np.asarray(sorted(pool))
    folds = probes.person_folds(pool, 5, probes.SEED)
    return pd.concat([majority_rows(model, set_name, persons, groups, items, set(pool[folds != f]),
                                    set(pool[folds == f])) for f in range(5)], ignore_index=True)


# ---------------------------------------------------------------- lines
def lines_of(rows, trivial=None):
    """{key: line} per (model, set, readout, variant) of a rows frame, and the GateResult dicts."""
    from mdfm.gates import g8
    out, results = {}, []
    for (m, s, r, v), g in rows.groupby(["model", "set", "readout", "variant"], sort=True):
        line = g8.line_of(g)
        line["chosen"] = bool(g["chosen"].all()) if len(g) else False
        key = f"{m}|{s}|{r}|{v}"
        it = g8.item_keys(g["item"])
        line["spearman_raw"] = {i: g8.spearman(g.loc[it == i, "raw"], g.loc[it == i, "target"]) for i in g8.ITEMS}
        line["modal_share"] = {i: _modal_share(g.loc[it == i, "prediction"]) for i in g8.ITEMS}
        out[key] = line
        triv = (trivial or {}).get(s, 0.0)
        results.append(g8.gate_result(line, m, s, r, v, trivial=triv).to_dict())
    return out, results


def _modal_share(pred):
    p = np.asarray(pred, dtype=np.float64)
    if not len(p):
        return None
    _, c = np.unique(p, return_counts=True)
    return float(c.max() / len(p))


def run_baselines(S):
    import pandas as pd

    from mdfm.gates import g8
    P = S["persons"]
    sp = set_persons(S)
    ev = S["everyone"]
    items, groups = item_vectors(P, ev)
    X = baseline_features(P, ev)
    parts = []
    for s in ("pd_dev", "all_dev"):
        fit, score = sp[s]
        if not score:
            continue
        parts.append(majority_rows("majority", s, ev, groups, items, fit, score))
        for name in ("speed", "part2", "others"):
            r, _ = g8.run_readout(X[name], ev, groups, items, fit, score, name, s, "")
            parts.append(r)
    for s in ("pd_pool", "all_pool"):
        pool = sp[s]
        if len(pool) < 10:
            continue
        parts.append(majority_pool("majority", s, ev, groups, items, pool))
        for name in ("speed", "part2", "others"):
            parts.append(g8.crossfit_readout(X[name], ev, groups, items, pool, name, s, ""))
    return pd.concat(parts, ignore_index=True)


def run_tokens(S, feats, name, trained, with_controls):
    """Rows of the token probes of one checkpoint (every variant and readout, the chosen variant flagged), the choice
    and its cross-fitted scores, the diagnostics."""
    import pandas as pd

    from mdfm.gates import g8
    P = S["persons"]
    sp = set_persons(S)
    ps = feats["persons"]
    items, groups = item_vectors(P, ps)
    sets = {"pd_dev": sp["pd_dev"]}
    if with_controls:
        sets["all_dev"] = sp["all_dev"]
    parts, diags, choice = [], {}, {}
    for s, (fit, score) in sets.items():
        fit = [p for p in fit if p in set(ps)]
        score = [p for p in score if p in set(ps)]
        diags[s] = {}
        for v in g8.VARIANTS:
            r, d = g8.run_readout(feats["X"][v], ps, groups, items, fit, score, name, s, v)
            parts.append(r)
            diags[s][v] = d
        best, scores = g8.choose_variant(diags[s])
        choice[s] = {"chosen": best, "crossfit_mean_qwk": scores}
    pools, pool_of = {}, {"pd_pool": "pd_dev", "all_pool": "all_dev"}
    for s in ("pd_pool", "all_pool") if with_controls else ("pd_pool",):
        pool = [p for p in sp[s] if p in set(ps)]
        if set(pool) & set(trained):
            pools[s] = f"not computed: {len(set(pool) & set(trained))} pool persons among the checkpoint's training persons"
            continue
        dev_set = pool_of[s]
        v = choice[dev_set]["chosen"]
        # both candidates: the variant chosen on the fitting set of dev_set was chosen on persons this row also scores
        # (rule section 4), so the other candidate's pool row is reported beside it
        for cand in g8.CANDIDATES:
            parts.append(g8.crossfit_readout(feats["X"][cand], ps, groups, items, pool, name, s, cand))
        pools[s] = (f"computed for both candidates {', '.join(g8.CANDIDATES)}; flagged chosen: {v}, the variant chosen on "
                    f"the fitting set of {dev_set}, whose persons this row also scores")
    rows = pd.concat(parts, ignore_index=True)
    rows["chosen"] = [(s in choice and v == choice[s]["chosen"])
                      or (s in pool_of and v == choice[pool_of[s]]["chosen"])
                      for s, v in zip(rows["set"], rows["variant"])]
    diag_out = {s: {v: {i: {k: x for k, x in d.items() if k != "cf_rows"} for i, d in dv.items()}
                    for v, dv in by.items()} for s, by in diags.items()}
    return rows, choice, pools, diag_out


# ---------------------------------------------------------------- report
def report_md(res):
    from mdfm.gates import g8
    L = [f"# G8 dev, weargait_pd items 3.10 to 3.13: {res['model_name']} ({LABEL})", "",
         "Written by scripts/eval_g8_dev.py (task G8R; harness mdfm/gates/g8.py; rule docs/rules/G8_RULE.md, DRAFT). Every "
         "number below is read from g8_dev.json of this folder. No lockbox, EVAL or CODY row or record was read.", ""]
    ck = res.get("checkpoint")
    if ck:
        L.append(f"Checkpoint: {ck.get('path')} (SHA-256 {ck.get('sha256')}), kind {res.get('kind')}, arm {ck.get('arm')}, "
                 f"seed {ck.get('seed')}, {ck.get('weights')} weights, phase {ck.get('phase')} step {ck.get('step')}. "
                 "Frozen tokens: tokenizer and encoder only, the G3 inference windows; per person the mean and the "
                 "standard deviation of the token rows of its SelfPace and HurriedPace trials.")
        L.append("")
    L += ["Line: mean over items 3.10 to 3.13 of the quadratic weighted kappa; pass at least 0.50, kill below 0.40 "
          "(design section 9); person-clustered 95 percent bootstrap interval (2,000 resamples, seed "
          f"{g8.SEED}). Reading aid only: no rule is registered.", "",
          "'filled rows' counts the person-item rows given the fitting set's majority grade (no finite prediction), "
          "'filled persons' the persons holding at least one such row.", "",
          "| model | set | readout | variant | chosen | mean QWK | 95 % interval | verdict (aid) | 3.10 | 3.11 | 3.12 | 3.13 | "
          "persons | filled rows | filled persons |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["gate_results"]:
        e = r["extra"]
        pi = e["per_item"]
        fp = e.get("filled_persons")
        L.append(f"| {e['model']} | {e['set']} | {e['readout']} | {e['variant'] or '-'} | "
                 f"{'yes' if res['lines'][_key(e)].get('chosen') else ''} | {fmt(r['point'])} | {fmt_iv(r['interval'])} | "
                 f"{r['verdict']} | {fmt(pi['3.10'])} | {fmt(pi['3.11'])} | {fmt(pi['3.12'])} | {fmt(pi['3.13'])} | "
                 f"{r['n_persons']} | {e['filled']} | {'n/a' if fp is None else fp} |")
    if res.get("choice"):
        L += ["", "Variant choice (person cross-fitted mean QWK on the fitting set, rounded readout):"]
        for s, c in res["choice"].items():
            L.append(f"- {s}: chosen {c['chosen']}; " + ", ".join(f"{v} {fmt(x)}" for v, x in c["crossfit_mean_qwk"].items()))
        for s, t in (res.get("pools") or {}).items():
            L.append(f"- {s}: {t}")
    c = res["census"]
    idx_counts = ", ".join(f"{k} {v.get('with_both_tasks')} of {v.get('persons')}"
                           for k, v in sorted((c.get("lockbox_index") or {}).items()))
    L += ["", "## Census (persons graded and with both G3 task records, per item)", "",
          "| set | persons | with records | 3.10 | 3.11 | 3.12 | 3.13 | part III not same day | part III day unknown |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, d in c["sets"].items():
        L.append(f"| {s} | {d['persons']} | {d['with_records']} | " + " | ".join(
            f"{d['items'][i]['graded_with_records']} ({', '.join(f'{k}:{v}' for k, v in d['items'][i]['grades'].items())})"
            for i in g8.ITEMS) + f" | {d['days_since_part3_nonzero']} | {d['days_since_part3_unknown']} |")
    L += ["", f"Lockbox, from reports/lockbox/lockbox_persons.csv only (no clinical row read): "
          f"{', '.join(f'{k} {v}' for k, v in sorted(c['lockbox_split_file'].items()))}; holding an IMU record of both G3 "
          f"tasks in the canonical index's metadata (no record opened): {idx_counts or 'n/a'}.",
          f"MoveAtlas HEAD {res.get('moveatlas_head')}. Time {fmt(res['timing_s'].get('all'), 1)} s.", ""]
    return "\n".join(L) + "\n"


def _key(e):
    return f"{e['model']}|{e['set']}|{e['readout']}|{e['variant']}"


def write_outputs(out, res, rows, persons, prov):
    from mdfm.gates import common
    rows_path = common.write_csv(rows, os.path.join(out, "per_row.csv"))
    common.write_csv(persons, os.path.join(out, "persons.csv"))
    res["rows"] = {"path": common.rel(rows_path), "sha256": common.sha256_file(rows_path), "n_rows": int(len(rows))}
    res = clean(res)
    text = report_md(res)
    common.write_text(os.path.join(out, f"{STEM}.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(out, f"{STEM}.md"), text)
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    return res, text


# ---------------------------------------------------------------- attainability
def attainability(root=OUT_ROOT):
    """Every line of reports/gates/g8/dev_*/g8_dev.json in one table: its kind, the point on each set, whether it
    reaches the pass line or lies beyond the kill line; per set and line the best reference of each kind."""
    import glob

    from mdfm.gates import common
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "dev_*", f"{STEM}.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        kind_run = r.get("kind") or "baselines"
        for g in r["gate_results"]:
            e = g["extra"]
            line = r["lines"][_key(e)]
            kind = KINDS.get(e["model"], "representation (body tokens)" if kind_run == "body" else
                             "predictor (baseline tokens)" if kind_run == "baseline" else kind_run)
            rows.append({"source": common.rel(p), "model": e["model"], "kind": kind,
                         "set": e["set"], "readout": e["readout"], "variant": e["variant"],
                         "chosen": bool(line.get("chosen")), "point": g["point"], "interval": g["interval"],
                         "n_persons": g["n_persons"], "per_item": e["per_item"], "filled_rows": e["filled"],
                         "filled_persons": e.get("filled_persons"),
                         "modal_share": line.get("modal_share") or {},
                         "largest_modal_share": max([x for x in (line.get("modal_share") or {}).values()
                                                     if fin(x) is not None], default=None),
                         "reaches_pass": fin(g["point"]) is not None and g["point"] >= 0.50,
                         "beyond_kill": fin(g["point"]) is not None and g["point"] < 0.40,
                         "moveatlas_head": r.get("moveatlas_head")})
    return rows


def attainability_md(rows):
    heads = sorted({r["moveatlas_head"] for r in rows if r.get("moveatlas_head")})
    L = ["# G8 attainability on dev, weargait_pd items 3.10 to 3.13 (task G8R)", "",
         "Written by scripts/eval_g8_dev.py --attainability from every reports/gates/g8/dev_*/g8_dev.json; dev estimates, "
         "not gate reads. Pass at least 0.50, kill below 0.40 (mean QWK over the four items, design section 9). MoveAtlas: "
         f"{', '.join(heads) or 'n/a'}.", "",
         "Primary rows: the rounded readout ('round') of the chosen variant (token probes) or of the baseline, on pd_dev "
         "(fit on pd_train). The quantile readout ('qmap'), the pool rows and the PD-and-controls rows are reported.", "",
         "Largest modal share: over the four items, the largest share of the set's persons given one grade by the readout "
         "(1.000: one item's predictions all on one grade). Filled persons: the persons given the fitting set's majority "
         "grade on at least one item (no finite prediction).", ""]
    head = ["| set | model | kind | readout | variant | mean QWK | 95 % interval | 3.10 | 3.11 | 3.12 | 3.13 | persons | "
            "filled persons | largest modal share | reaches 0.50 | below 0.40 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"pd_dev": 0, "pd_pool": 1, "all_dev": 2, "all_pool": 3}

    def line(r):
        pi = r["per_item"]
        fp = r.get("filled_persons")
        return (f"| {r['set']} | {r['model']} | {r['kind']} | {r['readout']} | {r['variant'] or '-'} | {fmt(r['point'])} | "
                f"{fmt_iv(r['interval'])} | {fmt(pi.get('3.10'))} | {fmt(pi.get('3.11'))} | {fmt(pi.get('3.12'))} | "
                f"{fmt(pi.get('3.13'))} | {r['n_persons']} | {'n/a' if fp is None else fp} | "
                f"{fmt(r.get('largest_modal_share'))} | {'yes' if r['reaches_pass'] else 'no'} | "
                f"{'yes' if r['beyond_kill'] else 'no'} |")
    ordered = sorted(rows, key=lambda r: (order.get(r["set"], 9), r["readout"] != "round", r["kind"], r["model"],
                                          r["variant"]))
    L += head + [line(r) for r in ordered if r["variant"] in ("", None) or r["chosen"]]
    other = [r for r in ordered if r["set"] in ("pd_pool", "all_pool") and r["variant"] not in ("", None)
             and not r["chosen"]]
    if other:
        L += ["", "## Pool rows of the candidate variant not chosen", "",
              "The pool row above uses the variant chosen on the fitting set of pd_dev (pd_train, cross-fitted), and the "
              "pool also scores those persons: the variant was chosen on persons the row scores. The other candidate's "
              "pool row, reported beside it:", ""] + head + [line(r) for r in other]
    L += ["", "Rows of the other variants on pd_dev and all_dev are in attainability.json (every variant of every token "
          "run).", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- jobs
def job(key, ckpt):
    """The GPU job of one checkpoint: PD persons and controls (--with-controls), outputs and features under a _gpu name
    (bf16 autocast on CUDA, float32 on the CPU: a GPU run never replaces a CPU folder)."""
    name = checkpoint_name(ckpt)
    out = f"reports/gates/g8/dev_{name}_gpu"
    feats = f"cache/runs/g8_features/{name}_gpu/features.npz"
    return {"name": f"g8_dev_{key}", "cls": 3, "priority": 60, "gpu": True, "training": False, "run_dir": None,
            "argv": [PYTHON, "scripts/eval_g8_dev.py", "--checkpoint", ckpt, "--weights", "ema", "--device", "cuda",
                     "--memory-gb", "8", "--batch-windows", "16", "--with-controls", "--features", feats, "--out", out],
            "est_hours": 0.25, "env": {}, "requires": [ckpt], "outputs": [f"{out}/{STEM}.json"], "timeout_hours": 1.0,
            "group": TASK, "note": f"PREPARED BY G8R, NOT QUEUED: G8 dev frozen-token probes of {name} on the non-lockbox "
                                   "weargait_pd PD persons and controls (docs/rules/G8_RULE.md, DRAFT; "
                                   "reports/weekend_2026-10-01/tasks/G8R.md)",
            "uses": ["scripts/eval_g8_dev.py"], "after": [], "soft_after": [], "not_if": [], "not_if_any": []}


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--no-model", action="store_true")
    g.add_argument("--attainability", action="store_true")
    g.add_argument("--print-jobs", action="store_true")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-gb", type=float, default=8.0)
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--features", default=None)
    p.add_argument("--from-features", action="store_true")
    p.add_argument("--with-controls", action="store_true", help="also extract the controls (the PD-and-controls rows)")
    p.add_argument("--jobs-dir", default=None, help="--print-jobs: also write the job files there")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm import paths
    from mdfm.gates import common, g3, g8
    from mdfm.train import provenance as PV

    if a.print_jobs:
        for key, ck in GPU_ENDS:
            j = job(key, ck)
            print(" ".join(j["argv"][1:]))
            if a.jobs_dir:
                os.makedirs(a.jobs_dir, exist_ok=True)
                common.write_text(os.path.join(a.jobs_dir, f"{j['name']}.json"), json.dumps(j, indent=1) + "\n")
        return 0
    if a.attainability:
        rows = attainability()
        common.write_text(os.path.join(OUT_ROOT, "attainability.json"), json.dumps(clean(rows), indent=1) + "\n")
        text = attainability_md(rows)
        common.write_text(os.path.join(OUT_ROOT, "attainability.md"), text)
        print(text)
        return 0

    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    S = load_sets(with_controls=True)
    cen = census(S)
    print(f"persons {len(S['everyone'])}, scored {len(S['scored'])}; targets {S['targets'].counts['passes']} passes",
          flush=True)
    code_files = [g8.__file__, g3.__file__, os.path.abspath(__file__)]
    res = {"what": LABEL, "task": TASK, "run_started": run_started, "rule": "docs/rules/G8_RULE.md (DRAFT, not registered)",
           "fold_info": S["fold_info"], "census": cen, "constants": {"seed": g8.SEED, "boot_n": g8.BOOT_N,
                                                                     "pass": g8.PASS_LINE, "kill": g8.KILL_LINE,
                                                                     "cross_folds": g8.CROSS_FOLDS}}
    if a.no_model:
        name = "baselines"
        out = os.path.abspath(a.out or os.path.join(OUT_ROOT, "dev_baselines"))
        rows = run_baselines(S)
        res.update(model_name=name, kind="baselines")
    else:
        ckpt = a.checkpoint if os.path.isabs(a.checkpoint) else os.path.join(TREE, a.checkpoint)
        name = checkpoint_name(ckpt)
        out = os.path.abspath(a.out or os.path.join(OUT_ROOT, f"dev_{name}"))
        features = os.path.abspath(a.features or os.path.join(FEATURE_ROOT, name, "features.npz"))
        sp = set_persons(S)
        want = sorted(set(sp["pd_pool"]) | (set(sp["all_pool"]) if a.with_controls else set()))
        from mdfm.baselines import readouts as RO
        code_files += [RO.__file__]
        if a.from_features:
            feats, meta = g8.load_features(features)
            if (meta.get("checkpoint") or {}).get("name") != name:
                raise SystemExit(f"{features} holds the features of another checkpoint")
            ck, kind, trained = meta["checkpoint"], meta["kind"], set(meta.get("trained_scored") or [])
        else:
            import mdfm  # noqa: F401
            import torch

            from mdfm.train import loop as LP
            dev_t = torch.device(a.device)
            if dev_t.type == "cuda" and not torch.cuda.is_available():
                raise SystemExit("--device cuda without a visible GPU (the GPU runs through scripts/queue_job.py)")
            if dev_t.type == "cpu":
                torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
            LP.cap_memory(dev_t, a.memory_gb)
            kind, model, info = RO.load_any(ckpt, weights=a.weights, device=dev_t)
            trained_all = info["trained_persons"]
            bad = sorted(set(sp["pd_dev"][1]) & trained_all) + (sorted(set(sp["all_dev"][1]) & trained_all)
                                                                  if a.with_controls else [])
            if bad:
                raise g8.Refused(f"dev persons among the checkpoint's training persons: {bad[:5]}")
            trained = set(want) & trained_all
            ck = info["describe"]
            print(f"loaded {name} ({ck['sha256'][:12]}), {kind}; scored persons among its training persons: "
                  f"{len(trained)}", flush=True)
            tg = S["targets"]

            def records():
                for tid, rec in g3.imu_records(want, tg.trials):
                    yield tid, g3.model_input(rec)

            t0 = time.time()
            feats = g8.person_features(lambda w: RO.encode_windows(model, w, a.batch_windows), records(), want)
            meta = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": ck, "kind": kind,
                    "trained_scored": sorted(trained), "tokens_s": round(time.time() - t0, 1),
                    "device": str(dev_t), "batch_windows": a.batch_windows, "weights": a.weights,
                    "with_controls": bool(a.with_controls), "code_sha256": common.code_sha256(code_files)}
            g8.save_features(features, feats, meta)
            print(f"features written to {common.rel(features)} ({meta['tokens_s']} s)", flush=True)
            del model
        with_controls = bool(meta.get("with_controls"))
        rows, choice, pools, diag = run_tokens(S, feats, name, trained, with_controls)
        res.update(model_name=name, kind=kind, checkpoint=ck, choice=choice, pools=pools, diagnostics=diag,
                   features={"path": common.rel(features), "sha256": common.sha256_file(features),
                             "counts": feats["counts"], "persons_without_rows": feats["counts"]["persons_without_rows"]},
                   trained_scored=sorted(trained))
    paths.assert_not_moveatlas(out)
    lines, results = lines_of(rows)
    res.update(lines=lines, gate_results=results, moveatlas_head=PV.moveatlas_head(),
               timing_s={"all": round(time.time() - t_start, 1)},
               lockbox={"persons_requested_in_lockbox": len(set(S["everyone"]) & set(g8.lockbox_persons())),
                        "target_lockbox_rows": int(S["targets"].counts["lockbox_rows"])})
    script_sha = common.sha256_file(os.path.abspath(__file__))
    prov = {"script": "scripts/eval_g8_dev.py", "label": LABEL, "task": TASK, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": res["moveatlas_head"], "script_sha256": script_sha,
            "code_sha256": common.code_sha256(code_files), "code_files": [common.rel(f) for f in code_files],
            "python": sys.version.split()[0], "numpy": np.__version__,
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    res, text = write_outputs(out, res, rows, S["persons"], prov)
    print(text)
    print(f"written {common.rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
