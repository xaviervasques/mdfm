"""T16: the tier hash history of the tiers records file, for the records drift policy of checkpoint loads.

Reads (read-only): cache/manifest/manifest_records.csv.gz (gitignored), reports/manifest_tiers/hashes.json,
reports/lockbox/lockbox_persons.csv and reports/lockbox/lockbox.sha256. Before anything is written: the
records file hashes to records_file_sha256 of hashes.json; records_sha256 recomputed from the file by the rule
of scripts/build_tiers_and_lockbox.py equals that of hashes.json, and so do the header (records_columns) and
the row count; the lockbox list hashes to lockbox.sha256 and to hashes.json
(mdfm.train.provenance.lockbox_sha256). The entry (mdfm.train.provenance.tier_hash_entry, plus the lockbox
SHA-256, the tiers version and the MoveAtlas commit hashes.json names) is added to
reports/provenance_history/tier_hash_history.json: an entry already held is kept when its content is equal,
and the run stops when it differs or when its lockbox SHA-256 differs (entries are never rewritten).
summary.md lists every entry and the changes between consecutive entries (per source: rows added or removed,
tier columns or row set, other columns with their value counts); _provenance.json stamps the MoveAtlas
commit read.

When to run: before every rebuild of the tiers (scripts/build_tiers_and_lockbox.py overwrites the records
file, and an entry can only be computed from the file it describes), then after it. A checkpoint whose records
state has no entry stays strict: its load raises after a rebuild.

--check CKPT [CKPT ...]: loads each checkpoint with mdfm.train.checkpoint.load on the CPU (a tolerated
records drift is appended to drift_log.jsonl beside the checkpoint) and writes
<out>/checkpoint_checks_<first 12 of the current records_sha256>.json: per checkpoint, its SHA-256, step,
outcome (loaded, refused with every problem, or error), drift and records drift evidence; a later check of
the same checkpoint in the same records state replaces its row. --no-history: only the checks. The process
runs at BelowNormal priority when psutil is present; the exit code is 1 when a check does not load.
Run: <conda>/envs/mdfm/python.exe scripts/build_tier_hash.py [--check PATH ...]
"""
import argparse
import datetime
import json
import os
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
               help="MoveAtlas working tree (read-only), for the provenance commit")
p.add_argument("--out", default=os.path.join(TREE, "reports", "provenance_history"),
               help="output folder (default reports/provenance_history, the history checkpoint loads read)")
p.add_argument("--check", nargs="+", default=[], metavar="CKPT", help="checkpoints to load after the history")
p.add_argument("--no-history", action="store_true", help="with --check: leave the history as it is")
ARGS = p.parse_args()
ARGS.moveatlas = ARGS.moveatlas.replace("\\", "/").rstrip("/")
os.environ["MOVEATLAS_ROOT"] = ARGS.moveatlas
sys.stdout.reconfigure(encoding="utf-8")

from _common import write_provenance  # noqa: E402
from mdfm import paths  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.train import provenance as PV  # noqa: E402

OUT = os.path.abspath(ARGS.out)
paths.assert_not_moveatlas(OUT)
HISTORY = os.path.join(OUT, "tier_hash_history.json")
PV.HISTORY_JSON = HISTORY            # a check reads the history this run writes


def rel(path):
    a = os.path.abspath(path)
    try:
        r = os.path.relpath(a, TREE)
    except ValueError:
        return a.replace("\\", "/")
    return a.replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path, obj):
    write_text(path, json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def below_normal():
    try:
        import psutil
    except ImportError:
        return
    proc = psutil.Process()
    proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)


# ---------------------------------------------------------------- the entry of the current records state
def current_entry():
    t = PV.tiers()
    lock = PV.lockbox_sha256()
    file_sha = PV.records_sha256()
    e = PV.tier_hash_entry()
    bad = []
    if e["records_file_sha256"] != file_sha:
        bad.append(f"records file read as {e['records_file_sha256']}, checked as {file_sha}")
    if e["records_sha256"] != t["records_sha256"]:
        bad.append(f"records_sha256 recomputed {e['records_sha256']}, hashes.json {t['records_sha256']}")
    if e["records_columns"] != list(t.get("records_columns") or []):
        bad.append(f"header {e['records_columns']}, hashes.json records_columns {t.get('records_columns')}")
    if e["records"] != t.get("records"):
        bad.append(f"{e['records']} rows, hashes.json records {t.get('records')}")
    if bad:
        sys.exit(f"the records file and reports/manifest_tiers/hashes.json disagree: {bad}")
    e.update({"lockbox_sha256": lock, "tiers_version": t.get("version"),
              "tiers_moveatlas_head": t.get("moveatlas_head"),
              "written": datetime.datetime.now().isoformat(timespec="seconds"), "script": "scripts/build_tier_hash.py"})
    return e


def add_entry(e):
    doc = None
    if os.path.exists(HISTORY):
        PV.tier_hash_history(HISTORY)                   # every entry hash-checked before the file is touched
        with open(HISTORY, encoding="utf-8") as f:
            doc = json.load(f)
    try:
        doc, added = PV.add_tier_entry(doc, e)
    except PV.ProvenanceError as err:
        sys.exit(str(err))
    held = doc["entries"][e["records_sha256"]]
    if held.get("lockbox_sha256") != e["lockbox_sha256"]:
        sys.exit(f"the history entry of {e['records_sha256']} holds lockbox_sha256 {held.get('lockbox_sha256')}, "
                 f"the reports {e['lockbox_sha256']}")
    doc["rule"] = ("Entries of mdfm.train.provenance.tier_hash_entry, keyed by records_sha256, written by "
                   "scripts/build_tier_hash.py before and after each rebuild of the tiers; never rewritten. Read "
                   "by mdfm.train.provenance.records_drift (the records drift policy of checkpoint loads).")
    if added:
        write_json(HISTORY, doc)
    PV.tier_hash_history(HISTORY)                       # the file written reads back and hash-checks
    return doc, added


# ---------------------------------------------------------------- summary
def fmt_values(v):
    if v is None:
        return ""
    return ", ".join(f"{k or '(empty)'} {n}" for k, n in sorted(v.items()))


def summary(doc):
    entries = sorted(doc["entries"].values(), key=lambda e: (e.get("written", ""), e["records_sha256"]))
    md = ["# Tier hash history (T16)", "",
          "Built by scripts/build_tier_hash.py; rules in its docstring and in mdfm/train/provenance.py (tier hash "
          "history and the records drift policy). History: reports/provenance_history/tier_hash_history.json. "
          f"Tolerated outside the sources of a run: {', '.join(PV.TOLERATED_COLUMNS)}; tier columns: "
          f"{', '.join(PV.TIER_COLUMNS)}; other columns: {', '.join(PV.OTHER_COLUMNS)}.", "",
          "| records_sha256 | records | sources | records file SHA-256 | lockbox SHA-256 | tiers version | MoveAtlas "
          "commit of the tiers | written |", "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        md.append(f"| {e['records_sha256']} | {e['records']} | {len(e['sources'])} | {e['records_file_sha256'][:16]} | "
                  f"{(e.get('lockbox_sha256') or '')[:16]} | {e.get('tiers_version', '')} | "
                  f"{(e.get('tiers_moveatlas_head') or '').split(' ')[0]} | {e.get('written', '')} |")
    md += ["", "## Changes between consecutive entries", ""]
    if len(entries) < 2:
        md.append("One entry: no change to report.")
    for a, b in zip(entries, entries[1:]):
        changes, problems = PV.compare_tier_entries(a, b)
        md += [f"### {a['records_sha256'][:12]} to {b['records_sha256'][:12]}", "",
               f"Lockbox SHA-256: {'unchanged' if a.get('lockbox_sha256') == b.get('lockbox_sha256') else 'CHANGED'}. "
               f"Sources that differ: {len(changes)}; changes outside the tolerated kind (any run): "
               f"{len([c for c in changes if not c['tolerated']])}.", ""]
        if changes:
            md += ["| source | kind | rows (before, after) | columns | values before | values after |",
                   "|---|---|---|---|---|---|"]
            for c in changes:
                vals = c.get("values") or {}
                md.append(f"| {c['source']} | {c['kind']} | {c['rows'][0]}, {c['rows'][1]} | "
                          f"{', '.join(c['columns'])} | "
                          + "; ".join(f"{k}: {fmt_values(v['before'])}" for k, v in vals.items()) + " | "
                          + "; ".join(f"{k}: {fmt_values(v['after'])}" for k, v in vals.items()) + " |")
            md.append("")
    text = "\n".join(md) + "\n"
    if chr(0x2014) in text or chr(0x2013) in text:
        sys.exit("summary.md would hold a dash character the tree does not use")
    return text


# ---------------------------------------------------------------- checks
def check(ckpts):
    from mdfm.train import checkpoint as CK
    now = PV.hashes()
    rows = []
    for c in ckpts:
        path = os.path.abspath(c)
        side = path + ".sha256"
        r = {"checkpoint": rel(path), "checked_at": datetime.datetime.now().isoformat(timespec="seconds")}
        if os.path.exists(side):
            with open(side, encoding="utf-8") as f:
                r["sha256"] = f.read().split()[0]
        try:
            ck = CK.load(path, map_location="cpu")
            r.update(outcome="loaded", step=int(ck["step"]), drift=list(ck["drift"]),
                     drift_evidence=ck["drift_evidence"],
                     saved_records_sha256=ck["provenance"]["hashes"].get("records_sha256"),
                     saved_lockbox_sha256=ck["provenance"]["hashes"].get("lockbox_sha256"),
                     run_sources=PV.run_sources(ck["provenance"]))
            del ck
        except (PV.ProvenanceError, Refused) as e:
            r.update(outcome="refused", error=f"{type(e).__name__}: {e}", problems=getattr(e, "problems", None))
        except Exception as e:  # noqa: BLE001
            r.update(outcome="error", error=f"{type(e).__name__}: {e}")
        print(f"{r['checkpoint']}: {r['outcome']}" + (f", drift {r['drift']}" if r["outcome"] == "loaded" else
                                                       f": {r.get('error', '')[:400]}"))
        rows.append(r)
    out = os.path.join(OUT, f"checkpoint_checks_{now['records_sha256'][:12]}.json")
    held = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            held = {r["checkpoint"]: r for r in json.load(f).get("checks", [])}
    held.update({r["checkpoint"]: r for r in rows})
    doc = {"format": "mdfm-checkpoint-checks/1", "script": "scripts/build_tier_hash.py --check",
           "records_sha256": now["records_sha256"], "records_file_sha256": now["records_file_sha256"],
           "lockbox_sha256": now["lockbox_sha256"], "moveatlas_head": now["moveatlas_head"],
           "history": rel(HISTORY), "checks": [held[k] for k in sorted(held)]}
    write_json(out, doc)
    print("wrote", rel(out))
    return all(r["outcome"] == "loaded" for r in rows)


def main():
    below_normal()
    os.makedirs(OUT, exist_ok=True)
    ARGS.out = OUT
    if not ARGS.no_history:
        write_provenance(ARGS, {"records_file": rel(PV.records_path())})
        e = current_entry()
        doc, added = add_entry(e)
        write_text(os.path.join(OUT, "summary.md"), summary(doc))
        print(f"records_sha256 {e['records_sha256']}: {'added' if added else 'already held'}; "
              f"{len(doc['entries'])} entries in {rel(HISTORY)}")
    elif not ARGS.check:
        sys.exit("--no-history without --check does nothing")
    ok = check(ARGS.check) if ARGS.check else True
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
