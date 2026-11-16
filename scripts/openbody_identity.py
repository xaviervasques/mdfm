"""Bit identity of the symptom catalogue between two commits on the open body (docs/OPEN_BODY_PORT_DESIGN.md section
19, open-body port wave 8): two caches of scripts/openbody_limits_catalogue.py --build --bodies, one built on a
worktree checkout of each commit, compared clip by clip by the SHA-256 of every stored array (the axis-angle
components of the body and hand joints, the raw body's rot6d, translation, contact, hand rotations and rate, as the
cache stores them) and by the build log's record (every field but the build time); optionally the pass, fail, error
and skip counts per test file of two folders of pytest junit XML files (one pytest process per file).

    python scripts/openbody_identity.py --before <store>/cache_before --after <store>/cache_after \
        --tests-before <folder> --tests-after <folder> --name wave8a

Writes reports/openbody/identity_<name>.json and .md. Reads program rotations only (no mesh, no model array);
nothing is written but the two report files.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "openbody")
LOG_SKIP = ("build_s",)


def _logs(cache):
    out = {}
    for p in sorted(glob.glob(os.path.join(cache, "log_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("ok") or not out.get(r["clip_id"], {}).get("ok"):
                    out[r["clip_id"]] = r
    return out


def _metas(cache):
    out = []
    for p in sorted(glob.glob(os.path.join(cache, "meta_*.json"))):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        out.append({k: m.get(k) for k in ("tree", "mdfm_commit", "moveatlas_commit", "n_specs", "bodies")})
    return out


def _digests(path):
    with np.load(path) as z:
        return {k: hashlib.sha256(np.ascontiguousarray(z[k]).tobytes() + str(z[k].dtype).encode()
                                  + str(z[k].shape).encode()).hexdigest() for k in sorted(z.files)}


def compare(before, after):
    lb, la = _logs(before), _logs(after)
    fb = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(before, "*.npz"))}
    fa = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(after, "*.npz"))}
    clips = sorted(fb | fa | set(lb) | set(la))
    rows, identical, differ, missing = [], [], [], []
    for c in clips:
        if c not in fb or c not in fa:
            missing.append({"clip_id": c, "before": c in fb, "after": c in fa,
                            "error_before": lb.get(c, {}).get("error"), "error_after": la.get(c, {}).get("error")})
            continue
        db, da = _digests(os.path.join(before, c + ".npz")), _digests(os.path.join(after, c + ".npz"))
        arrays = sorted(set(db) | set(da))
        bad = [k for k in arrays if db.get(k) != da.get(k)]
        rb = {k: v for k, v in lb.get(c, {}).items() if k not in LOG_SKIP}
        ra = {k: v for k, v in la.get(c, {}).items() if k not in LOG_SKIP}
        log_diff = sorted(k for k in set(rb) | set(ra) if rb.get(k) != ra.get(k))
        if bad or log_diff:
            differ.append({"clip_id": c, "arrays": bad, "log_fields": log_diff})
        else:
            identical.append(c)
        rows.append(len(arrays))
    return {"clips_before": len(fb), "clips_after": len(fa), "compared": len(identical) + len(differ),
            "identical": len(identical), "differing": differ, "missing": missing,
            "arrays_compared": int(sum(rows)),
            "errors_before": sorted(c for c, r in lb.items() if not r.get("ok")),
            "errors_after": sorted(c for c, r in la.items() if not r.get("ok")),
            "meta_before": _metas(before), "meta_after": _metas(after)}


def _junit(folder):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.xml"))):
        root = ET.parse(p).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        n = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        failed = []
        for s in suites:
            for k in n:
                n[k] += int(s.get(k, 0))
            for case in s.iter("testcase"):
                if case.find("failure") is not None or case.find("error") is not None:
                    failed.append(case.get("name"))
        n["passed"] = n["tests"] - n["failures"] - n["errors"] - n["skipped"]
        n["failed_tests"] = sorted(failed)
        out[os.path.basename(p)[:-4] + ".py"] = n
    return out


def tests(before, after):
    tb, ta = _junit(before), _junit(after)
    files = sorted(set(tb) | set(ta))
    per = {}
    for f in files:
        b, a = tb.get(f), ta.get(f)
        per[f] = {"before": b, "after": a,
                  "newly_failing": sorted(set(a["failed_tests"]) - set(b["failed_tests"])) if a and b else None,
                  "newly_passing": sorted(set(b["failed_tests"]) - set(a["failed_tests"])) if a and b else None}

    def tot(t):
        return {k: int(sum(v[k] for v in t.values())) for k in ("tests", "passed", "failures", "errors", "skipped")}
    return {"files": per, "total_before": tot(tb), "total_after": tot(ta)}


def markdown(rec):
    c = rec["catalogue"]
    L = ["# Bit identity of the catalogue between two commits (open body)", "",
         f"Before: {c['meta_before'][0]['mdfm_commit'] if c['meta_before'] else None}; after: "
         f"{c['meta_after'][0]['mdfm_commit'] if c['meta_after'] else None} (scripts/openbody_identity.py).", "",
         f"- clips stored: {c['clips_before']} before, {c['clips_after']} after; build errors {len(c['errors_before'])} "
         f"before, {len(c['errors_after'])} after",
         f"- compared: {c['compared']} clips, {c['arrays_compared']} arrays; identical {c['identical']}; differing "
         f"{len(c['differing'])}; missing on one side {len(c['missing'])}"]
    for d in c["differing"]:
        L.append(f"  - differs: {d['clip_id']} (arrays {', '.join(d['arrays']) or 'none'}; log fields "
                 f"{', '.join(d['log_fields']) or 'none'})")
    for m in c["missing"]:
        L.append(f"  - missing: {m['clip_id']} (before {m['before']}, after {m['after']})")
    if rec.get("tests"):
        t = rec["tests"]
        L += ["", "## Tests per file (one pytest process per file)", "",
              "| file | before passed | failed | errors | skipped | after passed | failed | errors | skipped | newly failing |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for f, v in t["files"].items():
            b, a = v["before"] or {}, v["after"] or {}
            L.append(f"| {f} | {b.get('passed')} | {b.get('failures')} | {b.get('errors')} | {b.get('skipped')} | "
                     f"{a.get('passed')} | {a.get('failures')} | {a.get('errors')} | {a.get('skipped')} | "
                     f"{', '.join(v['newly_failing'] or []) or '-'} |")
        tb, ta = t["total_before"], t["total_after"]
        L += ["", f"Totals: before {tb['passed']} passed, {tb['failures']} failed, {tb['errors']} errors, "
                  f"{tb['skipped']} skipped; after {ta['passed']} passed, {ta['failures']} failed, {ta['errors']} errors, "
                  f"{ta['skipped']} skipped."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--tests-before", default=None)
    ap.add_argument("--tests-after", default=None)
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rec = {"catalogue": compare(a.before, a.after)}
    if a.tests_before and a.tests_after:
        rec["tests"] = tests(a.tests_before, a.tests_after)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"identity_{a.name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    md = markdown(rec)
    with open(os.path.join(OUT, f"identity_{a.name}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
