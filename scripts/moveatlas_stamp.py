"""Read-only stamp of the MoveAtlas working tree, for the reports and jobs of this tree that must cite what
they were measured against while the MoveAtlas corrections stay uncommitted.

Usage: python scripts/moveatlas_stamp.py [--moveatlas ROOT] [--out FILE.json]

stamp(root) runs three git commands in the MoveAtlas tree, each as
    git -c safe.directory=<root> -C <root> <command>
with GIT_OPTIONAL_LOCKS=0 (so that status does not refresh the index file) and GIT_TERMINAL_PROMPT=0:
    rev-parse HEAD                                       -> head
    diff                                                 -> diff_sha256 (SHA-256 of the output bytes), diff_bytes
    status --porcelain=v1 -z --untracked-files=all       -> untracked (sorted paths of the '??' entries)
and adds manifest_tiers, the build date of reports/manifest_tiers of this tree (run_at of its
_provenance.json) set against batch 0002B2, applied to MoveAtlas on 2026-09-29 and not committed
(docs/moveatlas_corrections/0002B2_log.md), with a note. This script writes nothing in MoveAtlas and --out
refuses a path inside it; git itself may: 'git diff' rewrites the stat cache of .git/index when a tracked
file is stat-dirty (same content, newer time), even with GIT_OPTIONAL_LOCKS=0 (git 2.51), which changes no
content, commit or status. A git command that fails raises RuntimeError: a stamp is never partial.

diff_sha256 is the hash of the porcelain 'git diff' bytes as this machine's git configuration renders them:
textconv drivers and diff.* settings change them. The system gitattributes of Git for Windows send Word,
RTF, PDF and OpenDocument files (*.doc, *.docm, *.dot*, *.rtf, *.odt, *.ods, ...) to its astextplain textconv;
MoveAtlas marks *.pdf and *.docx binary, which overrides it, but not the others. Compare diff_sha256 values
only between stamps made on the same machine.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = "<moveatlas>"
BATCH = "0002B2"
BATCH_APPLIED = "2026-09-29"            # docs/moveatlas_corrections/0002B2_log.md, title line
BATCH_LOG = "docs/moveatlas_corrections/0002B2_log.md"
TIERS_PROVENANCE = os.path.join("reports", "manifest_tiers", "_provenance.json")
GIT_ENV = {"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}


def default_root():
    return os.environ.get("MOVEATLAS_ROOT", DEFAULT_ROOT).replace("\\", "/").rstrip("/")


def git_argv(root, *command):
    """The argv of a read-only git command in the MoveAtlas tree."""
    return ["git", "-c", f"safe.directory={root}", "-C", root, *command]


def _git(root, *command):
    """stdout bytes of a git command; RuntimeError on a non-zero exit."""
    env = dict(os.environ)
    env.update(GIT_ENV)
    argv = git_argv(root, *command)
    r = subprocess.run(argv, capture_output=True, env=env, timeout=600)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{' '.join(argv)} exited {r.returncode}: {err[:500]}")
    return r.stdout


def untracked(status_z):
    """Sorted paths of the untracked entries of 'git status --porcelain=v1 -z' output (bytes)."""
    out = []
    parts = status_z.decode("utf-8", errors="replace").split("\0")
    i = 0
    while i < len(parts):
        entry = parts[i]
        i += 1
        if len(entry) < 4:
            continue
        xy, path = entry[:2], entry[3:]
        if xy == "??":
            out.append(path)
        elif "R" in xy or "C" in xy:                         # a rename or copy: the next field is the source
            i += 1
    return sorted(out)


def manifest_tiers_build(tree=TREE):
    """{built_on, run_at, moveatlas_head, before_batch, note} of reports/manifest_tiers against batch 0002B2."""
    path = os.path.join(tree, TIERS_PROVENANCE)
    try:
        with open(path, encoding="utf-8") as f:
            prov = json.load(f)
    except (OSError, ValueError):
        prov = None
    if not isinstance(prov, dict) or not str(prov.get("run_at", ""))[:10]:
        return {"built_on": None, "run_at": None, "moveatlas_head": None, "before_batch": None,
                "note": f"{TIERS_PROVENANCE.replace(os.sep, '/')} not readable: the build date of "
                        f"reports/manifest_tiers against batch {BATCH} (applied {BATCH_APPLIED}) is unknown."}
    run_at = str(prov["run_at"])
    built = run_at[:10]
    head = str(prov.get("moveatlas_head", "")).split(" ")[0] or None
    before = built < BATCH_APPLIED
    if before:
        note = (f"reports/manifest_tiers was built on {built} (run_at {run_at}, MoveAtlas {head}), before batch "
                f"{BATCH}, applied to MoveAtlas on {BATCH_APPLIED} and not committed ({BATCH_LOG}): it predates "
                f"the {BATCH} record changes.")
    else:
        note = (f"reports/manifest_tiers was built on {built} (run_at {run_at}, MoveAtlas {head}), on or after "
                f"the day batch {BATCH} was applied ({BATCH_APPLIED}, {BATCH_LOG}).")
    return {"built_on": built, "run_at": run_at, "moveatlas_head": head, "before_batch": before, "note": note}


def stamp(root=None, tree=TREE):
    """The stamp of the MoveAtlas tree at `root` (default MOVEATLAS_ROOT or <moveatlas>)."""
    root = (root or default_root()).replace("\\", "/").rstrip("/")
    head = _git(root, "rev-parse", "HEAD").decode("ascii", errors="replace").strip()
    diff = _git(root, "diff")
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tiers = manifest_tiers_build(tree)
    return {
        "root": root,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "head": head,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_bytes": len(diff),
        "untracked": untracked(status),
        "manifest_tiers": tiers,
        "note": tiers["note"],
    }


def _inside(path, root):
    p = os.path.abspath(path).replace("\\", "/").rstrip("/").lower()
    r = os.path.abspath(root).replace("\\", "/").rstrip("/").lower()
    return p == r or p.startswith(r + "/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moveatlas", default=default_root(), help="MoveAtlas tree (read-only)")
    p.add_argument("--out", default=None, help="also write the stamp to this JSON file (never inside MoveAtlas)")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    s = stamp(a.moveatlas)
    text = json.dumps(s, indent=1, ensure_ascii=False)
    if a.out:
        if _inside(a.out, s["root"]):
            raise SystemExit(f"refusing to write inside MoveAtlas: {a.out}")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
