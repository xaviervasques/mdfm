"""Data scope of the MD_FM program: MoveAtlas data only (read-only).

The owner's rule (2026-09-24): use only MoveAtlas data, either published by MoveAtlas
(hosted Zenodo records) or held on disk with a right to use but not published.

For each of the 170 registry rows this script records:
- hosted: the dataset has a MoveAtlas Zenodo record (docs/record_titles.csv)
- harmonized: the dataset is in the canonical manifests (docs/demographics.csv)
- on-disk data: files and bytes under its folder, datacards and sidecars excluded
- licence class, parsed from docs/license_matrix.csv, else docs/datasets_index.csv
- modalities (M codes) and cells from docs/registry_cells.csv
- the design tier (EVAL, INTERNAL, RELEASE) from docs/MFM_DESIGN_2026-09-24.md section 5
and assigns one scope category:
  published             hosted by MoveAtlas
  usable_not_published  on disk, not hosted, licence grants research use (or own data)
  rights_unclear        on disk, not hosted, no licence or restrictive or unverified terms
  not_held              no data file on disk (barrier, catalogued only, not fetched)
  excluded              on disk, but decided out of scope (no licence chain)
  tool                  software, not data

Decisions on rights_unclear rows are read from docs/decisions/licence_decisions_*.csv (in this
tree): a decided row takes the decision's scope, tier and licence class, and the decision
column records it. Rows without a decision keep the category computed from the registry.

Outputs: <out>/data_scope.csv, <out>/data_scope_summary.md, <out>/_provenance.json.
Run: python scripts/data_scope.py --moveatlas <moveatlas>
"""
import collections
import csv
import os
import re

from _common import parse_args, write_provenance

ARGS = parse_args(__doc__.splitlines()[0])
REPO = ARGS.moveatlas
DOCS = REPO + "/docs"
OUT = ARGS.out
write_provenance(ARGS)


def rd(name):
    with open(os.path.join(DOCS, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


idx = rd("datasets_index.csv")
lic = {r["id"]: r for r in rd("license_matrix.csv")}
hosted = {r["dataset"] for r in rd("record_titles.csv")}
harmonized = {r["dataset_id"] for r in rd("demographics.csv")}
cells = {r["id"]: r for r in rd("registry_cells.csv")}

TOOLS = {"visionmd"}
OWN_DATA = {"cody_1", "cody_2", "cody_3"}  # Institut du Neurone cohorts, IRB frame, evaluation only
# shida_pd_gait (2026-11-12): its patients are very likely care_pd's BMCLab persons, so EVAL like care_pd
# (scripts/build_tiers_and_lockbox.py docstring).
EVAL = {"cody_1", "cody_2", "cody_3", "care_pd", "pd_biostamprc21", "kaggle_fog", "tremordb", "hubu_fis_fingertap",
        "shida_pd_gait"}
INTERNAL = {"ntu_rgbd", "koa_pd_nm", "goodwin_smm"}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIONS = {}
_ddir = os.path.join(HERE, "docs", "decisions")
for _fn in sorted(os.listdir(_ddir)) if os.path.isdir(_ddir) else []:
    if _fn.startswith("licence_decisions_") and _fn.endswith(".csv"):
        with open(os.path.join(_ddir, _fn), encoding="utf-8", newline="") as _f:
            for _r in csv.DictReader(_f):
                DECISIONS[_r["id"]] = dict(_r, file=_fn)
SKIP_FILES = re.compile(r"^(\._.*|\.DS_Store|datacard\.md|README.*|readme.*|LICENSE.*)$")


LIC_PATTERNS = [
    ("CC BY-NC-SA", r"BY-NC-SA"), ("CC BY-NC-ND", r"BY-NC-ND"), ("CC BY-NC", r"BY-NC(?!-SA|-ND)"),
    ("CC BY-SA", r"BY-SA"), ("CC0", r"CC0"), ("CC BY", r"CC[ -]BY\b(?!-)"), ("ODC-By", r"ODC-BY"),
    ("public domain", r"PUBLIC DOMAIN|PUBLIC RESOURCE"), ("etalab", r"ETALAB"), ("MIT", r"\bMIT\b"),
    ("Apache", r"APACHE"), ("non-commercial research", r"NON-?COMMERCIAL|RESEARCH ONLY|ACADEMIC"),
    ("competition terms", r"COMPETITION"),
]


def lic_class(text):
    """The licence named first in the text wins (later mentions are often history)."""
    u = text.upper()
    hits = [(m.start(), name) for name, pat in LIC_PATTERNS for m in [re.search(pat, u)] if m]
    return min(hits)[1] if hits else "other or unverified"


USABLE = {"CC0", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA", "CC BY-NC-ND", "ODC-By",
          "public domain", "etalab", "MIT", "Apache", "non-commercial research"}


def disk(path):
    n, b = 0, 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith("._")]
        for fn in filenames:
            if SKIP_FILES.match(fn):
                continue
            try:
                b += os.path.getsize(os.path.join(dirpath, fn))
                n += 1
            except OSError:
                pass
    return n, b


rows = []
for r in idx:
    i = r["id"]
    folder = os.path.join(REPO, r["path"])
    n_files, n_bytes = disk(folder)
    lm = lic.get(i)
    ltext = lm["license"] if lm else r["license"]
    lclass = "own data (IRB)" if i in OWN_DATA else lic_class(ltext)
    status = r["status"].upper()
    index_only = "INDEX-ONLY" in status or "INDEX ONLY" in status
    if i in TOOLS:
        cat = "tool"
    elif i in hosted:
        cat = "published"
    elif n_files == 0 or index_only:
        cat = "not_held"
    elif i in OWN_DATA or lclass in USABLE:
        cat = "usable_not_published"
    else:
        cat = "rights_unclear"
    c = cells.get(i, {})
    tier = "EVAL" if i in EVAL else "INTERNAL" if i in INTERNAL else "RELEASE"
    decision = ""
    if cat == "rights_unclear" and i in DECISIONS:
        d = DECISIONS[i]
        cat, tier, lclass = d["scope_after"], d["tier"], d["licence_class"]
        decision = f"{d['decision']} ({d['decided']})"
    rows.append({
        "id": i, "family": r["family"], "scope": cat, "hosted": i in hosted, "harmonized": i in harmonized,
        "design_tier": tier, "data_files": n_files, "data_gb": round(n_bytes / 1e9, 3),
        "licence_class": lclass, "licence_text": ltext[:160], "redistribution": r["redistribution"][:80],
        "modalities": c.get("modalities", ""), "cells": c.get("cells", ""), "n_subjects": r["n_subjects"][:40],
        "index_only": index_only, "status": r["status"][:120], "decision": decision,
    })

# Folders of the MoveAtlas tree that the registry does not list (the inversion testbed).
listed = {os.path.normcase(os.path.normpath(os.path.join(REPO, r["path"]))) for r in idx}
for fam in sorted(os.listdir(os.path.join(REPO, "datasets"))):
    famdir = os.path.join(REPO, "datasets", fam)
    if fam.startswith(("_", ".")) or not os.path.isdir(famdir):
        continue
    for d in sorted(os.listdir(famdir)):
        full = os.path.join(famdir, d)
        if d.startswith(("_", ".")) or not os.path.isdir(full) or os.path.normcase(os.path.normpath(full)) in listed:
            continue
        n_files, n_bytes = disk(full)
        card = os.path.join(full, "datacard.md")
        ltext = ""
        if os.path.isfile(card):
            with open(card, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if re.search(r"licen[cs]e", line, re.I):
                        ltext = line.strip()[:160]
                        break
        lclass = lic_class(ltext)
        cat = "not_held" if n_files == 0 else ("usable_not_published" if lclass in USABLE else "rights_unclear")
        rows.append({
            "id": d, "family": fam + " (not in registry)", "scope": cat, "hosted": False, "harmonized": False,
            "design_tier": "EVAL" if fam == "inversion_testbed" else "RELEASE", "data_files": n_files,
            "data_gb": round(n_bytes / 1e9, 3), "licence_class": lclass, "licence_text": ltext,
            "redistribution": "", "modalities": "", "cells": "", "n_subjects": "", "index_only": False,
            "status": "folder in the MoveAtlas tree, not in docs/datasets_index.csv", "decision": "",
        })

# Licence text of the registry against the licence matrix (the matrix is older on some rows).
for r in rows:
    reg = next((x for x in idx if x["id"] == r["id"]), None)
    if reg and r["id"] in lic and r["id"] not in OWN_DATA:
        a, b = lic_class(lic[r["id"]]["license"]), lic_class(reg["license"])
        r["licence_disagreement"] = "" if a == b else f"matrix {a} / registry {b}"
    else:
        r["licence_disagreement"] = ""

with open(os.path.join(OUT, "data_scope.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

by_cat = collections.Counter(r["scope"] for r in rows)
fam_cat = collections.Counter((r["family"], r["scope"]) for r in rows)
fams = sorted({r["family"] for r in rows})
cats = ["published", "usable_not_published", "rights_unclear", "not_held", "excluded", "tool"]
in_scope = [r for r in rows if r["scope"] in ("published", "usable_not_published")]


def has_mod(r, codes):
    mods = set(re.findall(r"M\d+", r["modalities"]))
    return bool(mods & codes)


def has_cell(r, prefix):
    return any(cc.startswith(prefix) for cc in r["cells"].split("|") if cc)


lines = ["# Data scope: MoveAtlas only", "",
         "Generated by scripts/data_scope.py; provenance in _provenance.json. Categories: published (MoveAtlas "
         "Zenodo record), usable_not_published (on disk, not hosted, licence grants research use, or own data), "
         "rights_unclear (on disk, not hosted, no or restrictive or unverified terms), not_held, excluded "
         "(on disk, decided out of scope), tool. Decisions: docs/decisions/licence_decisions_*.csv.", "",
         "## Counts", "", "| category | datasets | on-disk GB |", "|---|---|---|"]
for ccat in cats:
    sel = [r for r in rows if r["scope"] == ccat]
    lines.append(f"| {ccat} | {len(sel)} | {round(sum(r['data_gb'] for r in sel), 1)} |")
lines += ["", "## By family", "", "| family | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
for fam in fams:
    lines.append(f"| {fam} | " + " | ".join(str(fam_cat[(fam, ccat)]) for ccat in cats) + " |")
lines += ["", f"In scope (published + usable_not_published): {len(in_scope)} datasets, "
          f"{sum(1 for r in in_scope if r['harmonized'])} of them harmonized.", ""]


def listing(title, sel):
    out = [f"## {title} ({len(sel)})", "", "| id | scope | harmonized | tier | licence | GB | modalities | cells |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(sel, key=lambda x: (x["scope"], x["id"])):
        out.append(f"| {r['id']} | {r['scope']} | {'yes' if r['harmonized'] else 'no'} | {r['design_tier']} | {r['licence_class']} | {r['data_gb']} | {r['modalities']} | {r['cells']} |")
    return out + [""]


lines += listing("Neural data (M6 EEG or M7 LFP/ECoG), all categories", [r for r in rows if has_mod(r, {"M6", "M7"})])
lines += listing("Hyperkinetic cells (A1 to A8), all categories", [r for r in rows if any(has_cell(r, f"A{k}") for k in range(1, 9))])
lines += listing("Rights unclear (owner decision)", [r for r in rows if r["scope"] == "rights_unclear"])
lines += listing("Decided after the licence research (docs/decisions/)", [r for r in rows if r["decision"]])
lines += listing("Folders of the MoveAtlas tree not in the registry", [r for r in rows if "not in registry" in r["family"]])
dis = [r for r in rows if r.get("licence_disagreement")]
lines += [f"## Licence matrix and registry disagree ({len(dis)})", ""] + [f"- {r['id']}: {r['licence_disagreement']}" for r in dis] + [""]
with open(os.path.join(OUT, "data_scope_summary.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

print("categories:", dict(by_cat))
print("in scope:", len(in_scope), "harmonized in scope:", sum(1 for r in in_scope if r["harmonized"]))
print("wrote", OUT)
