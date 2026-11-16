"""Archive the records file of an earlier tiers state, so that a run can be read under the state it was born on
(decision of 2026-11-12, docs/MODEL_OPEN_PORT_PLAN.md step M7; mdfm.train.provenance.archived_state).

For each --source file (a tiers records file, cache/manifest/manifest_records.csv.gz of an earlier build):
- its tier hash entry is computed (provenance.tier_hash_entry) and must equal, on every entry key, the entry the
  tier hash history (reports/provenance_history/tier_hash_history.json) holds for its records_sha256;
- with --sums, the file's SHA-256 must equal the line of that SHA256SUMS file naming --sums-name (a backup's own
  check);
- it is copied to <cache>/manifest/archive/records_<first 12 of records_sha256>.csv.gz (gitignored) and the copy's
  SHA-256 checked; an existing copy with the same SHA-256 is kept, another one stops the script;
- its record is added to reports/provenance_history/records_archive.json: records_sha256, records_file_sha256,
  records, cache_path, history_entry_sha256, the source path and its check, the time. A record already held with
  other content stops the script (records are never rewritten).
Only file hashes and the history entry's per-source hashes are compared; no row is printed.
Run: <conda>/envs/mdfm/python.exe scripts/archive_records_state.py --source <file>
[--sums <SHA256SUMS.txt> --sums-name <name in it>]
"""
import argparse
import datetime
import json
import os
import shutil
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
from mdfm import paths  # noqa: E402
from mdfm.train import provenance as PV  # noqa: E402

KEEP = ("records_sha256", "records_file_sha256", "records", "cache_path", "history_entry_sha256")


def sums_line(path, name):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                return parts[0]
    sys.exit(f"{path} has no line for {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--sums", default=None)
    ap.add_argument("--sums-name", default=None)
    a = ap.parse_args()
    src = os.path.abspath(a.source)
    sha = PV.sha256_file(src)
    check = "history entry"
    if a.sums:
        want = sums_line(a.sums, a.sums_name)
        if want != sha:
            sys.exit(f"{src} SHA-256 {sha}, {a.sums} says {want}")
        check = f"history entry and {a.sums.replace(os.sep, '/')} ({a.sums_name})"
    entry = PV.tier_hash_entry(src)
    state = entry["records_sha256"]
    held = PV.tier_hash_history().get(state)
    if held is None:
        sys.exit(f"the history holds no entry for {state}")
    moved = [k for k in PV.ENTRY_KEYS if held.get(k) != entry.get(k)]
    if moved:
        sys.exit(f"{src} differs from the history entry of {state} on {moved}")
    rel = f"manifest/archive/records_{state[:12]}.csv.gz"
    dst = os.path.join(paths.cache_dir("manifest", "archive"), f"records_{state[:12]}.csv.gz")
    paths.assert_not_moveatlas(dst)
    if os.path.exists(dst):
        if PV.sha256_file(dst) != sha:
            sys.exit(f"{dst} exists with another SHA-256")
    else:
        shutil.copyfile(src, dst + ".tmp")
        if PV.sha256_file(dst + ".tmp") != sha:
            sys.exit(f"the copy of {src} does not hash to {sha}")
        os.replace(dst + ".tmp", dst)
    rec = {"records_sha256": state, "records_file_sha256": sha, "records": entry["records"], "cache_path": rel,
           "history_entry_sha256": held["entry_sha256"], "lockbox_sha256": held.get("lockbox_sha256"),
           "copied_from": src.replace(os.sep, "/"), "checked_against": check,
           "written": datetime.datetime.now().isoformat(timespec="seconds")}
    doc = {"format": PV.ARCHIVE_FORMAT,
           "rule": ("Records files of earlier tiers states, copied to the gitignored cache by "
                    "scripts/archive_records_state.py; mdfm.train.provenance.archived_state reads a state by its "
                    "records_sha256 and checks the copy against records_file_sha256 and the tier hash history."),
           "entries": {}}
    if os.path.exists(PV.ARCHIVE_JSON):
        with open(PV.ARCHIVE_JSON, encoding="utf-8") as f:
            doc = json.load(f)
    old = doc["entries"].get(state)
    if old is not None:
        if any(old.get(k) != rec.get(k) for k in KEEP):
            sys.exit(f"{PV.ARCHIVE_JSON} holds another record for {state}: records are never rewritten")
        print(f"{state}: already archived ({rel})")
    else:
        doc["entries"][state] = rec
        doc["entries"] = dict(sorted(doc["entries"].items()))
        with open(PV.ARCHIVE_JSON, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print(f"{state}: archived to cache/{rel} ({entry['records']} records, file {sha})")
    arch = PV.archived_state(state)
    print(f"read back: {arch['records_sha256']} {arch['records']} records, lockbox {arch['lockbox_sha256']}")


if __name__ == "__main__":
    main()
