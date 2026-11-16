"""A reading of the learnability curves of the open body against v1's (step M4 of docs/MODEL_OPEN_PORT_PLAN.md): the
gate of scripts/check_learnability.py reads its pass lines at the last step only (one sample of a run at a constant
learning rate after its warmup); this script reads the same lines on every recorded step from READ_FROM on.

For each report (reports/bench/learnability*.json of v1, reports/bench_open/ of the open body): per pass line the
values at the recorded steps from READ_FROM, how many of them pass, their median, the median of the last LAST
readings, and the steps where the line is passed and failed; per curve the train loss and the gradient norm at each
of those steps. Writes reports/bench_open/learnability_reading.json and .md; reads the reports only.

Run: PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/open_learnability_reading.py
"""
import datetime
import glob
import json
import os
import subprocess
import sys

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(TREE, "reports", "bench_open")
READ_FROM = 600
LAST = 6
REPORTS = ["reports/bench/learnability.json", "reports/bench/learnability_turns.json",
           "reports/bench_open/learnability.json", "reports/bench_open/learnability_turns.json",
           "reports/bench_open/learnability_turns_rest10.json", "reports/bench_open/learnability_turns_rest10_lr1e-4.json",
           "reports/bench_open/learnability_turns_walks_rest10.json"]


def head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def passes(v, op, line):
    return v is not None and ((v < line) if op == "below" else (v > line))


def read(path):
    with open(os.path.join(TREE, path), encoding="utf-8") as f:
        d = json.load(f)
    curve = [r for r in d["curve"] if r["step"] >= READ_FROM]
    out = {"report": path, "pass_at_last_step": d["pass"], "steps": [r["step"] for r in curve],
           "train_loss": [r["train_loss"] for r in curve], "grad_norm": [r["grad_norm"] for r in curve], "lines": {}}
    for field, op, line in d["pass_lines"]:
        vals = [r.get(field) for r in curve]
        vals = [None if isinstance(v, list) else v for v in vals]
        if field == "entropy_min_block":
            vals = [min(r["entropy_blocks"]) if r.get("entropy_blocks") else None for r in curve]
        ok = [passes(v, op, line) for v in vals]
        num = [v for v in vals if v is not None]
        out["lines"][field] = {"op": op, "line": line, "values": vals, "n_pass": int(sum(ok)), "n": len(vals),
                               "median": float(np.median(num)) if num else None,
                               "median_last": float(np.median(num[-LAST:])) if num else None,
                               "pass_steps": [s for s, o in zip(out["steps"], ok) if o],
                               "fail_steps": [s for s, o in zip(out["steps"], ok) if not o]}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    paths = REPORTS + sorted(os.path.relpath(p, TREE).replace("\\", "/") for p in
                             glob.glob(os.path.join(OUT, "rest_scan", "learnability_*.json")))
    rows = [read(p) for p in paths if os.path.exists(os.path.join(TREE, p))]
    doc = {"script": "scripts/open_learnability_reading.py", "mdfm_head": head(),
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "read_from_step": READ_FROM,
           "last": LAST, "reports": rows}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "learnability_reading.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    L = ["# The learnability curves read at every recorded step", "",
         f"Written by scripts/open_learnability_reading.py on {doc['run_at']} at MD_FM {doc['mdfm_head']}: each pass "
         f"line of scripts/check_learnability.py read at the recorded steps from {READ_FROM} on (every 100 steps), "
         f"not only at the last one; median_last is the median of the last {LAST} readings.", "",
         "| report | pass at the last step | line | readings passing | median | median of the last readings | "
         "failing steps |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for f, v in r["lines"].items():
            fmt = (lambda x: "n/a" if x is None else f"{x:.4g}")
            L.append(f"| {r['report']} | {'yes' if r['pass_at_last_step'] else 'no'} | {f} {v['op']} {v['line']:g} | "
                     f"{v['n_pass']} of {v['n']} | {fmt(v['median'])} | {fmt(v['median_last'])} | "
                     f"{', '.join(str(s) for s in v['fail_steps']) or 'none'} |")
    L += ["", "Every number above comes from reports/bench_open/learnability_reading.json.", ""]
    with open(os.path.join(OUT, "learnability_reading.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
