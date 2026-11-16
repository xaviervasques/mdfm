"""Migrate a checkpoint to another set of name tables (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.2, task
S2.2; the module is mdfm/train/migrate.py).

  python scripts/migrate_checkpoint.py --plan-only [--old-names stage1|s2b_draft|PATH.json] [--codes PATH.json]
         [--names s2b_draft|PATH.json] [--report DIR]
  python scripts/migrate_checkpoint.py CHECKPOINT --out NEW_RUN/ckpt.pt [--drift-log PATH] [--names ...]
         [--codes PATH.json] [--task-rows copy_unknown|zero] [--check] [--report DIR]

The plan needs the nuisance codes of the windows the source trained on, counted under the source's names
(migrate.plan_for): for STAGE1 names the script counts them on the train fold of the window store and the synthetic
layouts (migrate.stage1_codes); for other names --codes gives them (a JSON file of stage1_codes output under those
names), and without it the script stops. --plan-only plans from --old-names (default stage1) to the new names;
nothing is loaded or written except the report. With a checkpoint, the old names are those of its config
(migrate.names_of): migrate.migrate reads it with checkpoint.load (the drift log outside its run folder, by default
drift_log.jsonl in the new run folder), writes the migrated checkpoint into the new run folder (absent or empty)
and, with --check, loads the migrated file, builds both models (EMA and model weights) and prints the largest
differences that migrate.equivalence measures on one fixed training window per stage-1 source (the first train
recording of each source holding an input stream, its first 4 s). --names: 's2b_draft' (migrate.S2B_DRAFT) or a
JSON file of NameTables.to_dict(); --old-names also takes 'stage1'. --report DIR writes codes.json, plan.json,
check.json (with --check) and summary.md there. The migrated file's SHA-256 holds only for the code it was written
with (its lineage records the code SHA-256); the plan SHA-256 and the equivalence are the reproducible record.
Runs on the CPU.
"""
import argparse
import json
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)

from mdfm.train import migrate as MG  # noqa: E402
from mdfm.train import names as NM  # noqa: E402

DURATION_S = 4.0


def _names(arg):
    if arg == "s2b_draft":
        return MG.S2B_DRAFT
    if arg == "stage1":
        return NM.STAGE1
    with open(arg, encoding="utf-8") as f:
        return NM.NameTables.from_dict(json.load(f))


def _label(names):
    return "STAGE1" if names == NM.STAGE1 else ("S2B_DRAFT" if names == MG.S2B_DRAFT else "custom")


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _rel(p):
    return MG._rel(p)


def summary_text(info):
    pl = info["plan"]
    codes = info["codes"]
    lines = ["# Checkpoint migration (task S2.2)", "",
             f"Written by scripts/migrate_checkpoint.py on {info['time']}. MoveAtlas HEAD {info['moveatlas_head']}; "
             f"window store {info['store']} (store.json SHA-256 {info['store_sha256']}); layouts "
             f"reports/layouts/layouts.json (SHA-256 {codes.get('layouts_sha256')}).", "",
             f"Old names ({info['old_from']}): {info['old_label']}, SHA-256 {pl['old_sha256']}. New names: "
             f"{info['names_arg']}, SHA-256 {pl['new_sha256']}. Task rows: {pl['task_rows']}. Plan SHA-256 "
             f"{info['plan_sha256']}.", "",
             "## Nuisance codes of the windows the source trained on", "",
             f"Codes {info['codes_from']}, under the old names (SHA-256 {codes.get('names_sha256')}). Store input "
             f"streams {codes['store_streams']} (folds {', '.join(codes.get('folds', []))}), layout streams (one per "
             f"family stream and candidate rate) {codes['layout_streams']}; counts are nuisance tokens of one window "
             "per stream.", "", "| field | store codes | layout codes | 'unknown' seen |", "|---|---|---|---|"]
    for f in NM.NUIS_TABLES:
        s, lay = codes["store"][f], codes["layouts"][f]
        seen = s.get("unknown", 0) + lay.get("unknown", 0)
        lines.append(f"| {f} | {', '.join(f'{k} {v}' for k, v in s.items())} | "
                     f"{', '.join(f'{k} {v}' for k, v in lay.items())} | {seen} |")
    tu = info["trained_unknown"]
    lines += ["", f"Fields whose 'unknown' row those windows trained: {', '.join(tu) if tu else 'none'}; the new rows "
              "of the other growing tables start at zero.", "", "## Plan", "",
              "| table | old rows | new rows | start of the new rows |", "|---|---|---|---|"]
    for k, t in pl["tables"].items():
        new_rows = [r for r in t["rows"] if r[0] != "old"]
        start = ", ".join(sorted({f"{r[0]} {r[1]}" if r[1] is not None else r[0] for r in new_rows})) or "none"
        lines.append(f"| {k} | {t['old_rows']} | {t['new_rows']} | {start} |")
    if info.get("migration"):
        m = info["migration"]
        lines += ["", "## Migration", "",
                  f"Source {m['source']} (SHA-256 {m['source_sha256']}). Migrated file {m['out_name']}, SHA-256 "
                  f"{m['sha256']} for this run only: its lineage records the code SHA-256 at migration "
                  f"({m['code_sha256']}), so the same migration on other code writes a file of another SHA-256; the "
                  "plan SHA-256 above and the equivalence below are the reproducible record. Hashes that moved since "
                  f"the source was written (checkpoint.load drift): {', '.join(m['drift']) or 'none'}; the records "
                  f"drift line of that load went to {m['drift_log_name']} outside the source's run folder.",
                  "Mount priors per family of the new names (measured: the rows of reports/mount_priors/priors.json; "
                  f"generic: a copy of the generic family): {', '.join(f'{k} {v}' for k, v in m['mount_origin'].items())}."]
    if info.get("check"):
        lines += ["", "## Equivalence on stage-1 training windows", "",
                  f"{info['check']['windows']} windows ({', '.join(info['check']['sources'])}), {DURATION_S:g} s each, "
                  f"{info['check']['ema_knots']} decoded knots; largest absolute difference, the old model on the old "
                  "codes against the migrated model on the new codes:",
                  "", "| quantity | ema | model |", "|---|---|---|"]
        e, mo = info["check"]["ema"], info["check"]["model"]
        for k in e:
            lines.append(f"| {k} | {e[k]:g} | {mo[k]:g} |")
    return "\n".join(lines) + "\n"


def _print_plan(pl, codes):
    print(f"old names SHA-256 {pl.old.sha256()}; new names SHA-256 {pl.new.sha256()}; plan SHA-256 {pl.sha256()}")
    for f in NM.NUIS_TABLES:
        print(f"codes {f}: store {codes['store'][f]}; layouts {codes['layouts'][f]}")
    print(f"trained 'unknown' rows: {list(pl.trained_unknown)}")
    for k, t in pl.tables.items():
        print(f"table {k}: {t.old_rows} -> {t.new_rows}; new rows {[r for r in t.rows if r[0] != 'old']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--drift-log")
    ap.add_argument("--names", default="s2b_draft")
    ap.add_argument("--old-names")
    ap.add_argument("--codes")
    ap.add_argument("--task-rows", default="copy_unknown", choices=MG.TASK_ROWS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if not args.plan_only and not (args.checkpoint and args.out):
        ap.error("give CHECKPOINT and --out, or --plan-only")
    if args.old_names and not args.plan_only:
        ap.error("--old-names goes with --plan-only; a checkpoint's names are those of its config")
    import torch
    from mdfm.data import window_store as WS
    from mdfm.train import provenance as PV
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    new = _names(args.names)
    store_root = WS.store_root()
    given = None
    if args.codes:
        with open(args.codes, encoding="utf-8") as f:
            given = json.load(f)
    info = {"time": time.strftime("%Y-%m-%d %H:%M"), "moveatlas_head": PV.moveatlas_head(), "store": _rel(store_root),
            "store_sha256": PV.sha256_file(os.path.join(store_root, "store.json")), "names_arg": args.names,
            "codes_from": f"given in {_rel(args.codes)}" if args.codes else
            "counted by migrate.stage1_codes on the window store and the synthetic layouts"}
    if args.plan_only:
        old = _names(args.old_names or "stage1")
        pl, codes = MG.plan_for(old, new, args.task_rows, given)
        info["old_from"] = f"--old-names {args.old_names or 'stage1'}"
    else:
        res = MG.migrate(args.checkpoint, args.out, drift_log=args.drift_log, new=new, task_rows=args.task_rows,
                         codes=given, keep_source=args.check)
        pl, codes = res["plan"], res["codes"]
        info["old_from"] = "the source checkpoint's config"
        info["migration"] = {"source": _rel(args.checkpoint), "source_sha256": res["source_sha256"],
                             "out_name": os.path.basename(res["out"]), "sha256": res["sha256"], "drift": res["drift"],
                             "drift_log_name": os.path.basename(res["drift_log"]), "mount_origin": res["mount_origin"],
                             "code_sha256": res["code_sha256"]}
    info.update(old_label=_label(pl.old), codes=codes, trained_unknown=list(pl.trained_unknown), plan=pl.to_dict(),
                plan_sha256=pl.sha256())
    _print_plan(pl, codes)
    if not args.plan_only:
        print(f"migrated {args.checkpoint} (SHA-256 {res['source_sha256']}) -> {res['out']} (SHA-256 {res['sha256']}, "
              f"code SHA-256 {res['code_sha256']}); source drift {res['drift']}; mount priors {res['mount_origin']}")
        if args.check:
            from mdfm.train import checkpoint as CK
            src = res.pop("source")
            log2 = os.path.join(os.path.dirname(res["out"]), "drift_log_check.jsonl")
            ck = CK.load(res["out"], drift_log=log2)
            ck["opt"] = None
            wins = MG.stage1_windows(store_root, per_source=1, duration=DURATION_S)
            check = {"windows": len(wins), "sources": [w.source for w in wins]}
            for weights in ("ema", "model"):
                eq = MG.equivalence(MG.model_of(src, weights), MG.model_of(ck, weights), wins, pl.old, new)
                check[weights] = {k: v for k, v in eq.items() if k not in ("n_windows", "n_knots")}
                check[f"{weights}_knots"] = eq["n_knots"]
                print(f"equivalence ({weights}, {len(wins)} windows, {eq['n_knots']} knots): "
                      + ", ".join(f"{k} {v:g}" for k, v in check[weights].items()))
            info["check"] = check
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        _write(os.path.join(args.report, "codes.json"), json.dumps(codes, indent=1, sort_keys=True) + "\n")
        _write(os.path.join(args.report, "plan.json"), json.dumps(info["plan"], indent=1, sort_keys=True) + "\n")
        if info.get("check"):
            _write(os.path.join(args.report, "check.json"),
                   json.dumps({k: info[k] for k in ("migration", "check")}, indent=1, sort_keys=True) + "\n")
        _write(os.path.join(args.report, "summary.md"), summary_text(info))
        print(f"report written to {args.report}")
    return info


if __name__ == "__main__":
    main()
