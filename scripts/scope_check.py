"""Scope check of a document: every MoveAtlas dataset id it names, with its scope category.

Reads reports/data_scope/data_scope.csv (scripts/data_scope.py) and one or more markdown
files. Prints, per file, the ids named by category (published, usable_not_published,
rights_unclear, not_held, tool). Every rights_unclear or not_held id must sit in a
paragraph, or under a section or subsection heading, that flags it (conditional, owner's
decision, admits, outside the scope, not held, rights_unclear, data scope, excluded). The
script lists the paragraphs that name such an id without any flag, and exits 1 if any.

Run: python scripts/scope_check.py docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md [more.md]
"""
import collections
import csv
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scope_csv = os.path.join(here, "reports", "data_scope", "data_scope.csv")
with open(scope_csv, encoding="utf-8", newline="") as f:
    scope = {r["id"]: r["scope"] for r in csv.DictReader(f)}
ids = sorted(scope, key=len, reverse=True)
pat = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(i) for i in ids) + r")(?![A-Za-z0-9_])")
FLAG = re.compile(r"conditional|owner'?s decision|owner decision|admit|outside the scope|out of scope|not[ _]held|"
                  r"rights[ _]unclear|data scope|exclu|suspend|dropped|no longer", re.I)

status = 0
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8").read()
    named = collections.defaultdict(set)
    unflagged = []
    headings = {}  # level -> heading text
    para, para_start = [], 0
    lines = text.splitlines() + [""]
    for n, line in enumerate(lines, 1):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = line
        if line.strip() == "" or line.startswith("#") or line.startswith("|"):
            # a table row is its own paragraph
            blocks = [(para, para_start)] if para else []
            if line.startswith("|"):
                blocks.append(([line], n))
            for blk, start in blocks:
                ptxt = " ".join(blk)
                ctx = ptxt + " " + " ".join(headings.values())
                for m in pat.finditer(ptxt):
                    i = m.group(1)
                    named[scope[i]].add(i)
                    if scope[i] in ("not_held", "rights_unclear", "excluded") and not FLAG.search(ctx):
                        unflagged.append((start, i, scope[i], ptxt[:170]))
            para, para_start = [], n + 1
        else:
            if not para:
                para_start = n
            para.append(line)
    print(f"== {path}")
    for cat in ("published", "usable_not_published", "rights_unclear", "not_held", "excluded", "tool"):
        print(f"  {cat} ({len(named[cat])}): {', '.join(sorted(named[cat]))}")
    print(f"  paragraphs naming a rights_unclear, not_held or excluded id with no scope flag: {len(unflagged)}")
    for n, i, cat, t in unflagged:
        print(f"    line {n}: {i} [{cat}] {t}")
    status |= bool(unflagged)
sys.exit(1 if status else 0)
