"""The inputs of G1 on the open sets (docs/rules/G1_RULE_OPEN_2026-11-13.md, DRAFT; mdfm/gates/g1_open.py).

--halves: the rule's seeded halves of the two development pools (gait120's dev fold, the stroke patients outside the
lockbox), drawn from the person lists only (no record opens), written to reports/gates/g1/open_halves/halves.json and
halves.md with their SHA-256 (mdfm.data.sources.person_hash) and a _provenance.json. Run before any input is built.
--split dev / calib: the truth and inputs of that half under cache/g1/g1_open_v1/<split>/ (one shard per source,
g1's format, and manifest.json), from the open body stores' dev and G1 splits (scripts/build_open_body_store.py --split
dev and --split g1). The lockbox is refused.

Run: PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_g1_open_inputs.py --halves
     then --split calib and --split dev.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401
from mdfm.gates import common  # noqa: E402
from mdfm.gates import g1_open as GO  # noqa: E402


def head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def write_halves():
    doc = GO.draw_halves()
    doc.update({"script": "scripts/build_g1_open_inputs.py --halves", "mdfm_head": head(),
                "run_at": datetime.datetime.now().isoformat(timespec="seconds")})
    os.makedirs(GO.HALVES_DIR, exist_ok=True)
    common.write_text(GO.HALVES_JSON, json.dumps(doc, indent=1) + "\n")
    L = ["# G1 on the open sets: the development and calibration halves", "",
         f"Drawn by scripts/build_g1_open_inputs.py --halves on {doc['run_at']} at MD_FM {doc['mdfm_head']}, by the rule "
         "of docs/rules/G1_RULE_OPEN_2026-11-13.md section 3 (seed SHA-256 of '" + doc["seed_form"] + "', numpy "
         "default_rng permutation of the sorted ids, development the first round-half-up half). Person lists only; no "
         "record was opened.", "", "| source | pool | development | calibration | development SHA-256 | calibration "
         "SHA-256 |", "|---|---|---|---|---|---|"]
    for s in GO.SOURCES:
        L.append(f"| {s} | {len(doc['pools'][s])} | {len(doc['halves']['dev'][s])} | {len(doc['halves']['calib'][s])} | "
                 f"{doc['sha256']['dev'][s]} | {doc['sha256']['calib'][s]} |")
    L += ["", "Every number above comes from reports/gates/g1/open_halves/halves.json.", ""]
    common.write_text(os.path.join(GO.HALVES_DIR, "halves.md"), "\n".join(L))
    prov = {"script": doc["script"], "mdfm_head": doc["mdfm_head"], "run_at": doc["run_at"],
            "outputs": {"halves.json": common.sha256_file(GO.HALVES_JSON)}}
    common.write_text(os.path.join(GO.HALVES_DIR, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--halves", action="store_true")
    p.add_argument("--split", choices=GO.SPLITS)
    p.add_argument("--root", default=None)
    p.add_argument("--no-render", action="store_true")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.halves:
        write_halves()
    elif a.split:
        man = GO.build(a.split, a.root, render=not a.no_render, log=lambda *x: print(*x, flush=True))
        print(json.dumps({k: man[k] for k in ("n_windows", "n_knots", "n_persons", "manifest_sha256") if k in man}))
    else:
        p.error("--halves or --split")


if __name__ == "__main__":
    main()
