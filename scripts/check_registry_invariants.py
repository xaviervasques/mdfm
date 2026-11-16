"""Read-only invariants of a MoveAtlas registry, before or after the 2026-09-24 correction patch.

Re-implements, without importing any MoveAtlas code, the assertions of its
tests/test_registry.py (well-formed rows, unique ids, path equals datasets/<family>/...),
tests/test_snapshot_drift.py (docs/ and moveatlas/data/ copies equal as CSV rows, SNAPSHOT.json
counts) and the path rule of validate_registry.py, then prints the fields the patch corrects.
Writes nothing. Exit 1 if an invariant fails.

Run: python scripts/check_registry_invariants.py --moveatlas <moveatlas>
"""
import argparse
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
root = ap.parse_args().moveatlas


def rows(rel):
    with open(os.path.join(root, rel), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


ok = True
for name in ("datasets_index.csv", "license_matrix.csv"):
    a, b = rows("docs/" + name), rows("moveatlas/data/" + name)
    print(f"{name}: docs {len(a)} rows, bundled {len(b)} rows, equal as rows: {a == b}")
    ok &= a == b
idx = rows("docs/datasets_index.csv")
lm = rows("docs/license_matrix.csv")
ids = [r["id"] for r in idx]
dup = sorted({i for i in ids if ids.count(i) > 1})
print("duplicate ids:", dup or "none")
ok &= not dup
for r in idx:
    good = all((r.get(c) or "").strip() for c in ("id", "family", "path"))
    good &= r["path"].startswith("datasets/") and r["path"].split("/")[1] == r["family"]
    ok &= good
    if r["path"] != f"datasets/{r['family']}/{r['id']}":
        print("path differs from datasets/<family>/<id> (known before the patch for visionmd):", r["id"], r["path"])
snap = json.load(open(os.path.join(root, "moveatlas", "data", "SNAPSHOT.json"), encoding="utf-8"))
print("SNAPSHOT.json:", snap, "| files:", len(idx), "index rows,", len(lm), "licence rows")
ok &= snap.get("datasets_index_rows") == len(idx) and snap.get("license_matrix_rows") == len(lm)
by = {r["id"]: r for r in idx}
lmby = {r["id"]: r for r in lm}
print("modys_video n_subjects:", by["modys_video"]["n_subjects"], "(source: 34)")
print("fog_turning_filho n_subjects:", by["fog_turning_filho"]["n_subjects"], "(source: 35)")
print("mrc_lfp_dystonia status:", by["mrc_lfp_dystonia"]["status"][:70])
print("merk_neumann_ecog_lfp licence row:", "present" if "merk_neumann_ecog_lfp" in lmby else "absent")
print("ALL INVARIANTS HOLD" if ok else "AN INVARIANT FAILED")
sys.exit(0 if ok else 1)
