"""Register a dated rule file on the owner's yes (question C2 of docs/OWNER_DECISIONS_2026-11-12.md, answered yes by
the owner on 2026-11-12): the owner reads the file in chat, answers "oui <gate>", and the session runs this script once.

Steps, all or nothing:
1. The file is read as bytes. Its SHA-256 must start with --expect (the prefix the owner was given with the text, at
   least 12 hex characters), so the registered text is the text the owner read; its status line (the first line that
   starts with 'Status: ') must start with 'Status: DRAFT'.
2. That line, and nothing else, becomes 'Status: APPROVED by the owner on <date>.'.
3. The file is written back (bytes otherwise unchanged) and mdfm.gates.common.register_rule records its SHA-256 in
   docs/rules/gate_rule_hashes.csv; it refuses a gate that has a lockbox read logged. On any refusal or error the
   original bytes are written back and the error is raised.
4. The registry row, the SHA-256 before and after, and the line that changed are printed.

A second run after a success finds the file APPROVED with a registered SHA-256 and prints that row (nothing written).

Run (mdfm environment, from the tree root, PYTHONDONTWRITEBYTECODE=1):
  python scripts/register_dated_rule.py --gate G3 --path docs/rules/G3_RULE_2026-11-12.md --expect <prefix>
      --approved-on YYYY-MM-DD [--note TEXT]
"""
import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.gates import common  # noqa: E402

STATUS_LINE = re.compile(rb"^Status: [^\r\n]*", re.M)


def approve(gate, path, expect, approved_on, note="", registry_path=common.RULE_HASHES, log_path=common.READ_LOG):
    """Steps 1 to 4 of the module docstring. Returns {row, sha_before, sha_after, old_line, new_line}."""
    if not re.fullmatch(r"[0-9a-f]{12,64}", expect or ""):
        raise ValueError("--expect must be 12 to 64 lowercase hex characters of the SHA-256 the owner read")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approved_on or ""):
        raise ValueError(f"approved_on must be YYYY-MM-DD, got {approved_on!r}")
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    status, date = common.rule_status(raw.decode("utf-8"))
    if status == "approved":
        for r in common.registry(registry_path):
            if r["gate"] == gate and r["sha256"] == sha:
                return {"row": r, "sha_before": sha, "sha_after": sha, "old_line": "", "new_line": "",
                        "already": True}
        raise Refused(f"{common.rel(path)} already reads APPROVED (on {date}) but its SHA-256 {sha[:12]} is not "
                      f"registered for {gate}: nothing done")
    if not sha.startswith(expect):
        raise Refused(f"{common.rel(path)} SHA-256 {sha[:12]} is not the text the owner read ({expect[:12]}): "
                      "nothing done")
    m = STATUS_LINE.search(raw)
    if m is None or not m.group(0).startswith(b"Status: DRAFT"):
        raise Refused(f"{common.rel(path)}: no 'Status: DRAFT' line to approve")
    new_line = f"Status: APPROVED by the owner on {approved_on}.".encode("utf-8")
    new = raw[:m.start()] + new_line + raw[m.end():]
    with open(path, "wb") as f:
        f.write(new)
    try:
        row = common.register_rule(gate, path, approved_on, approved_by="owner", note=note,
                                   registry_path=registry_path, log_path=log_path)
    except BaseException:
        with open(path, "wb") as f:
            f.write(raw)
        raise
    return {"row": row, "sha_before": sha, "sha_after": hashlib.sha256(new).hexdigest(),
            "old_line": m.group(0).decode("utf-8"), "new_line": new_line.decode("utf-8"), "already": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--expect", required=True)
    ap.add_argument("--approved-on", required=True)
    ap.add_argument("--note", default="owner yes in chat; only the status line changed")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    res = approve(a.gate, a.path, a.expect, a.approved_on, note=a.note)
    if res["already"]:
        print(f"already registered: {res['row']}")
        return 0
    print(f"{a.path}: '{res['old_line']}' -> '{res['new_line']}'")
    print(f"SHA-256 read by the owner {res['sha_before']}; registered {res['sha_after']}")
    print(f"registry row: {res['row']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
