"""G1 on the open sets, the dev dry run (docs/rules/G1_RULE_OPEN_2026-11-13.md, DRAFT; mdfm/gates/g1_open.py): the
harness run end to end with the trivial baseline before any open model exists, as scripts/gate_g1.py does for v1.

The population prior of the open train store (g1_open.open_population_prior: the mean pose and the wrist draws of
the 397 train persons on their P3 rest skeletons), its conformal factor on the calibration half, then the three lines
on the development half (g1.score, g1.lines unchanged), per source (gait120, stroke patients). Writes
reports/gates/g1/population_prior_open_dev/: prior.json, g1.json, g1.md, per_person.csv, _provenance.json. No lockbox
row, no AMASS row and no SMPL+H array is read; the lockbox read log is not written.

Run: PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/gate_g1_open.py
"""
import datetime
import json
import os
import subprocess
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401
import numpy as np  # noqa: E402

from mdfm.gates import common  # noqa: E402
from mdfm.gates import g1 as G  # noqa: E402
from mdfm.gates import g1_open as GO  # noqa: E402

OUT = os.path.join(TREE, "reports", "gates", "g1", "population_prior_open_dev")


def head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    from mdfm.data.open_body_store import OpenTrainStore
    prior = GO.open_population_prior(OpenTrainStore())
    calib = GO.OpenG1Set("calib", verify=True)
    fit = G.fit_conformal(calib, prior)
    prior.c_wrist = fit["c"]
    prior.meta.update({"conformal": fit, "calib_manifest_sha256": calib.manifest["manifest_sha256"]})
    dev = GO.OpenG1Set("dev", verify=True)
    overlap = set(dev.persons) & set(prior.train_persons)
    if overlap or set(calib.persons) & set(prior.train_persons):
        raise SystemExit(f"scored persons among the prior's training persons: {sorted(overlap)[:3]}")
    table = G.score(dev, prior)
    with open(GO.RULE_PATH, "rb") as f:
        rule_sha = common.sha256_bytes(f.read())
    res = G.lines(table, table, prior, rule_sha)
    per_source = {}
    for s in GO.SOURCES:
        t = table[table["archive"] == s] if "archive" in table else table[table["take_id"].str.startswith(s)]
        if len(t):
            per_source[s] = {"windows": int(len(t)), "persons": int(t["person_id"].nunique()),
                             "g1a_mm": 1000 * G._pwm(t, "g1a_m"), "g1b_coverage": G._pwm(t, "coverage"),
                             "g1b_energy_mm": 1000 * G._pwm(t, "energy_m"), "g1c_mm": 1000 * G._pwm(t, "g1c_m")}
    os.makedirs(OUT, exist_ok=True)
    prior.save(os.path.join(OUT, "prior.json"))
    doc = {"script": "scripts/gate_g1_open.py", "mdfm_head": head(), "run_at": datetime.datetime.now().isoformat(
        timespec="seconds"), "rule": common.rel(GO.RULE_PATH), "rule_sha256": rule_sha,
           "halves_sha256": common.sha256_file(GO.HALVES_JSON),
           "dev": {"windows": len(dev), "knots": int(dev.windows["n"].sum()), "persons": len(dev.persons),
                   "manifest_sha256": dev.manifest["manifest_sha256"]},
           "calib": {"windows": len(calib), "knots": int(calib.windows["n"].sum()), "persons": len(calib.persons),
                     "manifest_sha256": calib.manifest["manifest_sha256"]},
           "prior_sha256": prior.sha256, "conformal": fit,
           "lines": [{"line": r.line, "point": r.point, "interval": list(r.interval), "pass": r.pass_line,
                      "kill": r.kill_line, "verdict": r.verdict, "unit": r.unit, "extra": r.extra} for r in res],
           "per_source": per_source, "seconds": round(time.time() - t0, 1)}
    common.write_text(os.path.join(OUT, "g1.json"), json.dumps(doc, indent=1, default=float) + "\n")
    table.groupby("person_id")[["g1a_m", "coverage", "energy_m", "g1c_m"]].mean().to_csv(
        os.path.join(OUT, "per_person.csv"), lineterminator="\n")
    L = ["# G1 on the open sets: the dev dry run of the trivial baseline", "",
         f"Written by scripts/gate_g1_open.py on {doc['run_at']} at MD_FM {doc['mdfm_head']}. Rule "
         f"{doc['rule']} (DRAFT, SHA-256 {rule_sha[:16]}...), halves of reports/gates/g1/open_halves/halves.json. The "
         "population prior of the open train store, its conformal factor on the calibration half "
         f"(c = {fit['c']:.4f}), scored on the development half: {doc['dev']['windows']} windows, "
         f"{doc['dev']['knots']:,} knots of {doc['dev']['persons']} persons (calibration {doc['calib']['windows']} "
         f"windows of {doc['calib']['persons']} persons). No lockbox row was read.", "",
         "| line | point | 95% interval | verdict |", "|---|---|---|---|"]
    for r in res:
        nd = 3 if r.unit == "" else 1
        L.append(f"| {r.line} | {r.point:.{nd}f} {r.unit} | {r.interval[0]:.{nd}f} to {r.interval[1]:.{nd}f} | "
                 f"{r.verdict} |")
    L += ["", "| source | persons | windows | G1a mm | G1b coverage | G1b energy mm | G1c mm |", "|---|---|---|---|---|---|---|"]
    for s, v in per_source.items():
        L.append(f"| {s} | {v['persons']} | {v['windows']} | {v['g1a_mm']:.1f} | {v['g1b_coverage']:.3f} | "
                 f"{v['g1b_energy_mm']:.1f} | {v['g1c_mm']:.1f} |")
    L += ["", "Every number above comes from reports/gates/g1/population_prior_open_dev/g1.json.", ""]
    common.write_text(os.path.join(OUT, "g1.md"), "\n".join(L))
    prov = {"script": doc["script"], "mdfm_head": doc["mdfm_head"], "run_at": doc["run_at"],
            "inputs": {"halves.json": doc["halves_sha256"], "dev manifest": doc["dev"]["manifest_sha256"],
                       "calib manifest": doc["calib"]["manifest_sha256"]},
            "outputs": {n: common.sha256_file(os.path.join(OUT, n)) for n in ("g1.json", "prior.json")}}
    common.write_text(os.path.join(OUT, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
