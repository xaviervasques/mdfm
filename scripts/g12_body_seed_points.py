"""Descriptive (2026-10-18): the body's G3 dev points per seed, the way scripts/g12_compare.py reads seeds 1 and 2
(per_person.csv rows 'dev', predictor 'model'; dev12 = the 14 dev controls minus WHC003 and WHC026), including the
third seed 20261013 (the owner's decision of 2026-10-13), which g12_compare.py leaves out until a dated G12 revision
records it. Not a G12 comparison: no rule is registered. Writes reports/bench/g12/body_seed_points.json.

Run: <conda>/python scripts/g12_body_seed_points.py
"""
import csv
import hashlib
import json
import os
import statistics

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")   # mdfm/gates/g3.py EXCLUDED_FROM_12
REPORTS = {20260928: "reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044_bw32_now",
           20261001: "reports/gates/g3/dev_pilot_v1_s2_ckpt_B_032044_bw32",
           20261013: "reports/gates/g3/dev_pilot_v1_s3_ckpt_B_032044"}
LINES = ("stride_mae_cm", "speed_mae_m_s")


def main():
    out = {"what": "descriptive, not a G12 comparison", "seeds": {}, "inputs": {}}
    for seed, rel in REPORTS.items():
        p = os.path.join(TREE, rel, "per_person.csv")
        with open(p, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["row"] == "dev" and r["predictor"] == "model"]
        with open(p, "rb") as f:
            out["inputs"][rel] = hashlib.sha256(f.read()).hexdigest()
        s = {}
        for line in LINES:
            v14 = [float(r["value"]) for r in rows if r["line"] == line]
            v12 = [float(r["value"]) for r in rows if r["line"] == line and r["person_id"] not in EXCLUDED_FROM_12]
            s[line] = {"dev14": statistics.mean(v14), "n14": len(v14), "dev12": statistics.mean(v12), "n12": len(v12)}
        out["seeds"][str(seed)] = s
        print(seed, {k: round(v["dev12"], 3) for k, v in s.items()})
    with open(os.path.join(TREE, "reports", "bench", "g12", "body_seed_points.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
