"""Shared helpers for the measurement scripts of the MD_FM tree.

Every script reads the MoveAtlas tree READ-ONLY and writes only under the
reports directory of this tree. The MoveAtlas root comes from --moveatlas or
from the MOVEATLAS_ROOT environment variable. Each run stamps the MoveAtlas
commit it measured against in <out>/_provenance.json.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys

DEFAULT_ROOT = "<moveatlas>"


def parse_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", DEFAULT_ROOT),
                   help="MoveAtlas working tree (read-only). Default: MOVEATLAS_ROOT or <moveatlas>")
    p.add_argument("--out", default=None,
                   help="output directory (default: <this tree>/reports/<script name>)")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(a.moveatlas, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {a.moveatlas} (no INDEX.md)")
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if a.out is None:
        a.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", script)
    os.makedirs(a.out, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return a


def moveatlas_head(root):
    """Read-only: the commit the measurement is made against."""
    try:
        r = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", root, "log", "-1", "--format=%h %ad %s", "--date=short"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or f"(git failed: {r.stderr.strip()[:200]})"
    except Exception as e:  # noqa: BLE001
        return f"(git unavailable: {e})"


def write_provenance(a, extra=None):
    prov = {"script": os.path.basename(sys.argv[0]), "moveatlas_root": a.moveatlas,
            "moveatlas_head": moveatlas_head(a.moveatlas), "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0]}
    if extra:
        prov.update(extra)
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print("provenance:", prov["moveatlas_head"])
    return prov


def read_csv(path, **kw):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", **kw)
