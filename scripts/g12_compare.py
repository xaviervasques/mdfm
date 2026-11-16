"""The section 8 comparison of docs/rules/G12_RULE.md (DRAFT) on the G3 dev lines (task G12R; revision
docs/rules/G12_RULE_REVISION_2026-10-08.md, hashed): the body against the better baseline arm on stride_mae_cm,
speed_mae_m_s and ic_median_ms, on the 14 and the 12 dev controls. The method is mdfm/baselines/comparison.py; this
script finds the per-row predictions, checks them against the points their own reports state, and writes the result.
Reads files only (CPU, no model, no record, no lockbox).

Sources:
- body seeds (BODY_REPORTS, rule section 4: pilot v1 seed 20260928, seed 2 seed 20261001; the third seed is OPEN,
  revision item 5): the dev rows of the G3 harness (scripts/eval_g3_dev.py) for each seed's phase B end, the reports the
  gate table cites (reports/gates/gate_table.md, MoveAtlas ecf7bbf): per_person.csv, rows 'dev' of predictor 'model'
  (one entering row per person; stride_mae_cm, speed_mae_m_s, ic_median_ms_walkway), checked against g3_dev.json's
  points on the 14; the 12 are the same persons' values (the harness fits the IC offset on the 14 for both sets);
- baseline seeds: reports/bench/g12/readouts/*/results.json of kind 'baseline' (scripts/g12_readout.py), read only
  when final_verified (the run FINISHED and the checkpoint read is its verified final one, rule sections 5 and 6) and
  not partial; per_row.csv of the chosen variants, its SHA-256 checked against results.json and its lines against
  results.json's points; two readouts of one arm and seed are refused (listed as missing);
- the representation row (readouts of kind 'body'): its points are reported, without verdict (rule section 6).

Preconditions (comparison.preconditions): the owner's registration of docs/rules/G12_RULE.md (mdfm.gates.common.
registered, gate G12, the file's SHA-256 now), three seeds with rows for the body and for each usable arm, every
baseline seed a body seed of the same value (rule section 4 and section 11 item 5); and (comparison.source_checks) one
MoveAtlas commit stated by every compared source and each source's dev sets equal to the comparison's (a
representation row that differs gets a note). While one is missing the status is NOT EVALUATED with the list, and
only the points of the sources found are reported (MFR seeds alone are rows without a verdict, revision item 1).
--unusable ARM=REASON marks an arm unusable (a collapse, or an arm taken out by a dated revision); nothing marks one
by default. G8, G10 and G7: NOT EVALUATED with their reasons.

Writes, with --write, <out> (default reports/bench/g12/compare_g3_dev): g12.json, g12.md, per_person.csv (model, seed,
set, line, person_id, value of every source read), the file names of rule section 13 in a dev folder (the gate
harness's reports/gates/g12/ is for the registered rule). Prints the JSON.

Usage (mdfm or base environment, from the tree root):
  python scripts/g12_compare.py [--write] [--unusable jepa=collapsed]
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import math
import os
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "bench", "g12", "compare_g3_dev")
READOUTS = os.path.join(TREE, "reports", "bench", "g12", "readouts")
RULE = os.path.join(TREE, "docs", "rules", "G12_RULE.md")
RULE_HASH = os.path.join(TREE, "reports", "bench", "g12", "rule.sha256")
BODY_REPORTS = {20260928: "reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044_bw32_now",
                20261001: "reports/gates/g3/dev_pilot_v1_s2_ckpt_B_032044_bw32"}
BODY_LINES = {"stride_mae_cm": "stride_mae_cm", "speed_mae_m_s": "speed_mae_m_s", "ic_median_ms_walkway": "ic_median_ms"}
SEED_NOTES = {"body": "the third seed is OPEN (revision 2026-10-08 item 5)"}
BODY_DEV12_NOTE = (
    "The body's points are its decoded dev rows of the G3 dev harness (scripts/eval_g3_dev.py, per_person.csv rows "
    "'dev' of predictor 'model', the reports under Sources). Its dev12 points are the 14's per-person values of that "
    "harness restricted to the 12, the IC offset fitted on the 14 (the harness fits it on the 14 for both sets). They "
    "are not the dev12 rows that reports/gates/gate_table.md lists for the same checkpoints, which come from another "
    "harness (scripts/eval_g3_diag.py, diag.json reports), nor the decoded dev12 numbers of rule section 10 (T25's "
    "attainability, read from eval_g3_diag reports measured against MoveAtlas f2e6f99).")
LABEL = "G12 section 8 comparison on the G3 dev lines, dev only"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path, tree=TREE):
    return os.path.relpath(os.path.abspath(path), tree).replace(os.sep, "/")


def _same(a, b, tol=1e-9):
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _float(x):
    if x is None:
        return float("nan")
    return float(x)


# ---------------------------------------------------------------- sources
def body_rows(report_dir, tree=TREE, dev14=None):
    """(rows DataFrame, source dict) of one body seed's G3 dev report (module docstring); SystemExit when its rows do
    not give the points of its g3_dev.json on the 14."""
    import pandas as pd

    from mdfm.baselines import comparison as CMP
    d = os.path.join(tree, report_dir)
    pp, gj = os.path.join(d, "per_person.csv"), os.path.join(d, "g3_dev.json")
    t = pd.read_csv(pp, encoding="utf-8", dtype={"person_id": str})
    t = t[(t["row"] == "dev") & (t["predictor"] == "model") & t["line"].isin(set(BODY_LINES))]
    n = len(t)
    rows = pd.DataFrame({"line": t["line"].map(BODY_LINES).to_numpy(), "variant": "decoded", "chosen": True,
                         "person_id": t["person_id"].astype(str).to_numpy(), "trial_id": "", "task": "", "pass_id": "",
                         "foot": "", "t0": [float("nan")] * n, "t1": [float("nan")] * n, "t_s": [float("nan")] * n,
                         "target": [float("nan")] * n, "prediction": [float("nan")] * n, "matched": True,
                         "enters": True, "value": pd.to_numeric(t["value"], errors="coerce").to_numpy()})
    with open(gj, encoding="utf-8") as f:
        g = json.load(f)
    model = g["lines"]["dev"]["model"]
    persons = dev14 or sorted(set(g["sets"]["dev"]))
    check = {}
    for src, ln in BODY_LINES.items():
        want = _float((model.get(src) or {}).get("point"))
        got = CMP.line_point(rows[rows["line"] == ln], CMP.LINES[ln]["how"], persons)
        check[ln] = {"report": want, "rows": got, "same": _same(want, got)}
        if not check[ln]["same"]:
            raise SystemExit(f"{rel(pp, tree)}: {ln} from the rows {got}, g3_dev.json {want}")
    ck = g.get("checkpoint") or {}
    from mdfm.data import sources as SRC
    stated = {"dev14": sorted(map(str, g["sets"]["dev"]))}
    src = {"kind": "body", "report": rel(d, tree), "per_person_csv": {"path": rel(pp, tree), "sha256": sha256_file(pp)},
           "g3_dev_json": {"path": rel(gj, tree), "sha256": sha256_file(gj)}, "checkpoint": ck.get("path"),
           "checkpoint_sha256": ck.get("sha256"), "train_persons_sha256": (ck.get("persons_sha256") or {}).get("all"),
           "moveatlas_head": g.get("moveatlas_head"), "sets": stated,
           "sets_sha256": {s: SRC.person_hash(p) for s, p in stated.items()}, "points_check": check}
    return rows, src


def readout_rows(results_path, tree=TREE):
    """(rows DataFrame of the chosen variants, source dict) of one readout's results.json (module docstring);
    SystemExit when its per_row.csv changed or does not give its points."""
    import pandas as pd

    from mdfm.baselines import comparison as CMP
    with open(results_path, encoding="utf-8") as f:
        r = json.load(f)
    rp = os.path.join(tree, r["rows"]["path"])
    if sha256_file(rp) != r["rows"]["sha256"]:
        raise SystemExit(f"{rel(rp, tree)} changed since {rel(results_path, tree)} was written")
    t = pd.read_csv(rp, encoding="utf-8", dtype={"person_id": str, "trial_id": str, "pass_id": str, "task": str,
                                                 "foot": str})
    for c in ("chosen", "enters", "matched"):
        t[c] = t[c].map(lambda x: str(x).strip().lower() in ("true", "1"))
    rows = t[t["chosen"]].reset_index(drop=True)
    check = {}
    for s in ("dev14", "dev12"):
        for ln, spec in CMP.LINES.items():
            want = (r["lines"][s][ln] or {}).get("point")
            want = float(want) if want is not None else float("nan")
            got = CMP.line_point(rows[rows["line"] == ln], spec["how"], r["sets"][s])
            check[f"{s} {ln}"] = {"report": want, "rows": got, "same": _same(want, got)}
            if not check[f"{s} {ln}"]["same"]:
                raise SystemExit(f"{rel(rp, tree)}: {s} {ln} from the rows {got}, results.json {want}")
    ck = r.get("checkpoint") or {}
    tr = r.get("training") or {}
    from mdfm.data import sources as SRC
    stated = {s: sorted(map(str, r["sets"][s])) for s in ("dev14", "dev12")}
    src = {"kind": r.get("kind"), "arm": r.get("arm"), "seed": r.get("seed"), "results": rel(results_path, tree),
           "sets": stated, "sets_sha256": {s: SRC.person_hash(p) for s, p in stated.items()},
           "results_sha256": sha256_file(results_path), "per_row_csv": {"path": r["rows"]["path"],
                                                                         "sha256": r["rows"]["sha256"]},
           "checkpoint": ck.get("path"), "checkpoint_sha256": ck.get("sha256"),
           "train_persons_sha256": (ck.get("persons_sha256") or {}).get("all"), "config_sha256": ck.get("config_sha256"),
           "final_verified": r.get("final_verified"), "partial": r.get("partial"), "chosen": r.get("chosen"),
           "compute": tr.get("compute"), "training_verdict": tr.get("verdict"), "moveatlas_head": r.get("moveatlas_head"),
           "points_check": check}
    return rows, src


def find_readouts(root=READOUTS, tree=TREE):
    """({arm: {seed: (rows, source)}}, [representation sources], [notes]) of the readouts under root."""
    arms, reps, notes, seen = {}, [], [], {}
    for p in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            notes.append(f"{rel(p, tree)}: unreadable")
            continue
        if r.get("task") != "G12R":
            continue
        if r.get("partial"):
            notes.append(f"{rel(p, tree)}: a partial (smoke) readout, not read")
            continue
        if r.get("kind") == "body":
            rows, src = readout_rows(p, tree)
            reps.append((rows, src))
            continue
        if r.get("kind") != "baseline":
            continue
        if not r.get("final_verified"):
            notes.append(f"{rel(p, tree)}: not the verified final checkpoint of a FINISHED run, not read (rule "
                         "sections 5 and 6)")
            continue
        key = (r.get("arm"), int(r.get("seed")))
        seen.setdefault(key, []).append(p)
    for (arm, seed), ps in sorted(seen.items()):
        if len(ps) > 1:
            notes.append(f"{arm} seed {seed}: {len(ps)} readouts ({', '.join(rel(x, tree) for x in ps)}); none read")
            continue
        arms.setdefault(arm, {})[seed] = readout_rows(ps[0], tree)
    return arms, reps, notes


def rule_state(tree=TREE):
    """The draft's SHA-256 and registration (mdfm.gates.common.registered), and the hashed revision's MATCH."""
    from mdfm.gates import common
    path = os.path.join(tree, "docs", "rules", "G12_RULE.md")
    sha = sha256_file(path)
    out = {"draft": rel(path, tree), "draft_sha256": sha, "registered": bool(common.registered("G12", sha))}
    hp = os.path.join(tree, "reports", "bench", "g12", "rule.sha256")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            parts = f.read().split()
        target = os.path.join(tree, *parts[1].lstrip("*").split("/"))
        now = sha256_file(target) if os.path.exists(target) else None
        out["revision"] = {"file": parts[1].lstrip("*"), "sha256": parts[0],
                           "verdict": "MATCH" if now == parts[0] else "MISMATCH"}
    return out


# ---------------------------------------------------------------- main reading
def per_person_table(sources, sets):
    """[(model, seed, set, line, person_id, value)] of every source (model: body, an arm, or representation)."""
    from mdfm.baselines import comparison as CMP
    out = []
    for model, seed, rows in sources:
        for s, persons in sets.items():
            for ln, spec in CMP.LINES.items():
                ids, v = CMP.person_values(rows[rows["line"] == ln], spec["how"], persons)
                out += [(model, seed, s, ln, i, float(x)) for i, x in zip(ids.tolist(), v.tolist())]
    return out


def compare(tree=TREE, readouts=None, body_reports=None, unusable=None, sets=None, registered=None):
    """The comparison dict (module docstring). sets: {set: persons} (default the G3 harness's dev14 and dev12)."""
    from mdfm.baselines import comparison as CMP
    if sets is None:
        from mdfm.gates import g3
        d14 = sorted(g3.controls("dev"))
        sets = {name: sorted(set(d14) - set(out)) for name, out in g3.DEV_SETS.items()}
    rule = rule_state(tree)
    if registered is None:
        registered = rule["registered"]
    body, bsrc = {}, {}
    for seed, d in sorted((body_reports or BODY_REPORTS).items()):
        if os.path.exists(os.path.join(tree, d, "per_person.csv")):
            body[seed], bsrc[seed] = body_rows(d, tree, sets["dev14"])
    found, reps, notes = find_readouts(readouts or READOUTS, tree)
    arms = {a: {s: rs[0] for s, rs in found.get(a, {}).items()} for a in CMP.ARMS}
    asrc = {a: {s: rs[1] for s, rs in found.get(a, {}).items()} for a in CMP.ARMS}
    prov = {m: {s: {"moveatlas_head": src.get("moveatlas_head"), "sets": src.get("sets")} for s, src in by.items()}
            for m, by in (("body", bsrc),) + tuple(asrc.items())}
    res = CMP.evaluate(body, arms, sets, registered=registered, unusable=unusable, notes=SEED_NOTES, provenance=prov)
    heads = {p["moveatlas_head"] for by in prov.values() for p in by.values()}
    rep_points = []
    for rows, src in reps:
        if src.get("moveatlas_head") not in heads or len(heads) != 1:
            notes.append(f"representation seed {src['seed']} ({src['results']}): MoveAtlas "
                         f"{(src.get('moveatlas_head') or 'none')[:7]}, the compared sources "
                         f"{', '.join(sorted(str(h)[:7] for h in heads)) or 'none'}")
        for s, p in sets.items():
            if src["sets"].get(s) is not None and src["sets"][s] != sorted(map(str, p)):
                notes.append(f"representation seed {src['seed']} ({src['results']}): its {s} persons are not the "
                             f"comparison's")
        rep_points.append({**{k: src[k] for k in ("seed", "results", "checkpoint", "checkpoint_sha256", "chosen")},
                           "points": {s: {ln: CMP.line_point(rows[rows["line"] == ln], sp["how"], p)
                                          for ln, sp in CMP.LINES.items()} for s, p in sets.items()}})
    srcs = [("body", s, r) for s, r in sorted(body.items())] + \
           [(a, s, r) for a in CMP.ARMS for s, r in sorted(arms[a].items())] + \
           [("representation", src["seed"], rows) for rows, src in reps]
    out = {"what": LABEL, "task": "G12R", "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script_sha256": sha256_file(os.path.abspath(__file__)), "rule": rule, "sets": sets, **res,
           "sources": {"body": bsrc, "arms": asrc, "representation": [src for _, src in reps]},
           "representation_rows": rep_points, "notes": notes}
    return out, per_person_table(srcs, sets)


def _f(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return str(v) if math.isinf(v) else f"{v:.{nd}f}"


def markdown(res):
    from mdfm.baselines import comparison as CMP
    nd = {"stride_mae_cm": 2, "speed_mae_m_s": 3, "ic_median_ms": 1}
    r = res["rule"]
    L = [f"# G12 comparison on the G3 dev lines ({res['what']})", "",
         f"Written {res['time']} by scripts/g12_compare.py (SHA-256 {res['script_sha256']}; method "
         "mdfm/baselines/comparison.py; docs/rules/G12_RULE.md section 8, revision 2026-10-08). Every number below is read "
         "from the files listed under Sources.", "",
         f"Rule: {r['draft']} SHA-256 {r['draft_sha256']}, registered by the owner: {r['registered']}; revision "
         f"{(r.get('revision') or {}).get('file')} {(r.get('revision') or {}).get('verdict')}.", "",
         f"## Status: {res['status']}", ""]
    for m in res["missing"]:
        L.append(f"- missing: {m}")
    for n in res.get("notes") or []:
        L.append(f"- note: {n}")
    L += ["", "## Points per model and seed (rows, no verdict)", "", "| set | line | model | seed | point |",
          "|---|---|---|---|---|"]
    for s, by_line in res["points"].items():
        for ln, by_model in by_line.items():
            for m, by_seed in by_model.items():
                for seed, p in by_seed.items():
                    L.append(f"| {s} | {ln} | {m} | {seed} | {_f(p, nd[ln])} |")
    if res["sources"]["body"]:
        L += ["", BODY_DEV12_NOTE]
    if res.get("representation_rows"):
        L += ["", "## Representation row (the body's own frozen tokens probed; reported without verdict)", "",
              "| seed | set | " + " | ".join(CMP.LINES) + " |", "|---|---|" + "---|" * len(CMP.LINES)]
        for rp in res["representation_rows"]:
            for s, pts in rp["points"].items():
                L.append(f"| {rp['seed']} | {s} | " + " | ".join(_f(pts[ln], nd[ln]) for ln in CMP.LINES) + " |")
    if res["status"] == CMP.EVALUATED:
        L += ["", "## Section 8 per line", "", "| set | line | comparator | body mean | comparator mean | separation | "
              "difference | 95 % interval | reading |", "|---|---|---|---|---|---|---|---|---|"]
        for s, d in res["lines"].items():
            for ln, c in d.items():
                iv = c.get("interval")
                L.append(f"| {s} | {ln} | {c['comparator']} | {_f(c['body_mean'], nd[ln])} | "
                         f"{_f((c.get('arm_means') or {}).get(c['comparator']), nd[ln])} | {c.get('separation')} | "
                         f"{_f(c.get('difference'), nd[ln])} | "
                         f"{'n/a' if not iv else '[' + _f(iv[0], nd[ln]) + ', ' + _f(iv[1], nd[ln]) + ']'} | "
                         f"{c['reading']} |")
    L += ["", f"Verdict: {res['verdict']}", "", "## Other lines", ""]
    for k, v in res["other_lines"].items():
        L.append(f"- {k}: {v['status']}: {v['reason']}")
    L += ["", "## Sources", ""]
    for seed, s in sorted(res["sources"]["body"].items()):
        L.append(f"- body seed {seed}: {s['per_person_csv']['path']} (SHA-256 {s['per_person_csv']['sha256']}), "
                 f"{s['g3_dev_json']['path']} (SHA-256 {s['g3_dev_json']['sha256']}); checkpoint {s['checkpoint']} "
                 f"(SHA-256 {s['checkpoint_sha256']}); MoveAtlas {s['moveatlas_head']}; dev14 persons SHA-256 "
                 f"{(s.get('sets_sha256') or {}).get('dev14')}")
    for a, by in res["sources"]["arms"].items():
        for seed, s in sorted(by.items()):
            c = s.get("compute") or {}
            L.append(f"- {a} seed {seed}: {s['results']} (SHA-256 {s['results_sha256']}), {s['per_row_csv']['path']}; "
                     f"checkpoint {s['checkpoint']} (SHA-256 {s['checkpoint_sha256']}); clock {_f(c.get('wall_s'), 0)} s, "
                     f"ratio to the budget {_f(c.get('ratio'), 4)}, steps {c.get('steps')}, tokens {c.get('tokens')}; "
                     f"MoveAtlas {s.get('moveatlas_head')}; dev14 persons SHA-256 "
                     f"{(s.get('sets_sha256') or {}).get('dev14')}")
    for s in res["sources"]["representation"]:
        L.append(f"- representation seed {s['seed']}: {s['results']} (SHA-256 {s['results_sha256']})")
    if not any(res["sources"]["arms"].values()):
        L.append("- no baseline readout found under reports/bench/g12/readouts")
    return "\n".join(L) + "\n"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else ("inf" if o > 0 else "-inf" if o < 0 else None)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return _clean(o.item())
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    return o


def write(res, table, out=OUT):
    """g12.json, g12.md and per_person.csv under out; returns out."""
    import io

    from mdfm.gates import common
    os.makedirs(out, exist_ok=True)
    common.write_text(os.path.join(out, "g12.json"), json.dumps(_clean(res), indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(out, "g12.md"), markdown(res))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["model", "seed", "set", "line", "person_id", "value"])
    for row in table:
        w.writerow(row)
    common.write_text(os.path.join(out, "per_person.csv"), buf.getvalue())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--unusable", action="append", default=[], metavar="ARM=REASON")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    unusable = {}
    for x in a.unusable:
        arm, _, why = x.partition("=")
        unusable[arm.strip()] = why.strip() or "marked unusable"
    res, table = compare(unusable=unusable)
    if a.write:
        write(res, table, os.path.abspath(a.out))
    print(json.dumps(_clean(res), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
