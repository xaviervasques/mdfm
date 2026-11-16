"""Licence and network audit of the open body (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md section 2).

Loads Anny through mdfm.synth.openbody (the allow list and the guard) under a Python audit hook (sys.addaudithook)
that records every file opened (path and mode), every socket, urllib, subprocess and os.system event, and refuses
any network connection. With --fresh the load builds Anny's cache from nothing in a temporary folder (the first
use: every file Anny parses and every file it writes), which is removed at the end; without it the load reads the
pinned cache (mdfm.synth.openbody.CACHE_DIR). Files written by native code (safetensors) escape the hook: the
cache folder is listed before and after instead.

Then, statically: every line of the installed anny sources that names a download, the non-commercial archive, an
HTTP library or an URL (DOWNLOAD_WORDS), with its function, so that the reading of the code paths (the only download
sits in anny/paths.py and is reached only from the SMPL and SMPL-X topologies and wrappers, which the guard
blocks) is a script output; the refusals of the guard tried one by one; the integrity of every anny file the load
opened against the wheel's RECORD; the SHA-256 of the licence texts of what is used.

Output: reports/openbody/licence_audit[_fresh].json. No network, nothing written in this tree but the report.

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/openbody_audit.py [--fresh]
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "reports", "openbody")
FRESH_DIR = "<store>/anny_cache_audit_tmp"
DOWNLOAD_WORDS = ("download", "noncommercial", "requests", "urllib", "http://", "https://", "hf_hub", "huggingface",
                  "urlopen", "socket")
NET_EVENTS = ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request", "http.client.connect")

EVENTS = {"open": [], "network": [], "process": []}


def _origin():
    """The calling frames of an event outside the standard library's own modules: 'package/file.py:line'."""
    import traceback
    keep = []
    for fr in traceback.extract_stack()[:-2]:
        f = fr.filename.replace("\\", "/")
        if "site-packages/" in f or "/mdfm/" in f:
            keep.append(f"{f.split('site-packages/')[-1].split('/mdfm/mdfm/')[-1]}:{fr.lineno}")
    return keep[-12:]


def _hook(ev, args):
    if ev == "open":
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            mode = args[1] if len(args) > 1 and args[1] is not None else "r"
            EVENTS["open"].append((os.fsdecode(p), str(mode)))
    elif ev.startswith("socket.") or ev.startswith("urllib") or ev.startswith("http"):
        EVENTS["network"].append((ev, repr(args)[:300], _origin()))
        if ev in NET_EVENTS:
            raise RuntimeError(f"network refused by the audit: {ev}")
    elif ev in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.startfile"):
        EVENTS["process"].append((ev, repr(args)[:300], _origin()))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listing(root):
    out = {}
    if os.path.isdir(root):
        for d, _, files in os.walk(root):
            for n in files:
                p = os.path.join(d, n)
                out[os.path.relpath(p, root).replace("\\", "/")] = os.path.getsize(p)
    return out


def record_hashes(dist_info):
    """{relative path: urlsafe-b64 sha256} of the wheel's RECORD."""
    rec = {}
    with open(os.path.join(dist_info, "RECORD"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(",", 2)
            if len(parts) == 3 and parts[1].startswith("sha256="):
                rec[parts[0]] = parts[1][len("sha256="):]
    return rec


def b64sha(p):
    with open(p, "rb") as f:
        return base64.urlsafe_b64encode(hashlib.sha256(f.read()).digest()).rstrip(b"=").decode()


def static_scan(pkg):
    """[{file, line, function, text}] of the anny source lines that hold a DOWNLOAD_WORDS word (comments included)."""
    hits = []
    for d, _, files in os.walk(pkg):
        for n in sorted(files):
            if not n.endswith(".py"):
                continue
            p = os.path.join(d, n)
            func = None
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    s = line.strip()
                    if s.startswith("def ") or s.startswith("class "):
                        func = s.split("(")[0].split(":")[0]
                    if any(w in line.lower() for w in DOWNLOAD_WORDS):
                        hits.append({"file": os.path.relpath(p, pkg).replace("\\", "/"), "line": i,
                                     "function": func, "text": s[:160]})
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", action="store_true", help="build the cache from nothing in a temporary folder")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.fresh:
        if os.path.exists(FRESH_DIR):
            sys.exit(f"{FRESH_DIR} exists: remove it first (a fresh audit starts from an empty folder)")
        os.makedirs(FRESH_DIR)
        os.environ["MDFM_ANNY_CACHE"] = FRESH_DIR
    home_cache = os.path.join(os.path.expanduser("~"), ".cache", "anny")
    home_before = os.path.exists(home_cache)
    sys.addaudithook(_hook)
    import importlib.util
    spec = importlib.util.find_spec("anny")
    pkg = os.path.dirname(spec.origin)
    site = os.path.dirname(pkg)
    from mdfm.synth import openbody as OB
    cache = OB.CACHE_DIR
    before = listing(cache)
    t0 = time.time()
    m = OB.model()
    load_s = time.time() - t0
    after = listing(cache)
    import anny
    # refusals of the guard
    refusals = {}
    for label, fn in (("topology smplx", lambda: OB.check_names("anny", "smplx")),
                      ("topology smpl", lambda: OB.check_names("anny", "smpl")),
                      ("topology soma", lambda: OB.check_names("anny", "soma")),
                      ("rig soma", lambda: OB.check_names("soma", "anny")),
                      ("rig makehuman", lambda: OB.check_names("makehuman", "anny")),
                      ("import anny.models.smpl", lambda: __import__("anny.models.smpl")),
                      ("import anny.models.retopology", lambda: __import__("anny.models.retopology")),
                      ("import smplx", lambda: __import__("smplx")),
                      ("anny.paths.get_anny2smplx_data_path", lambda: anny.paths.get_anny2smplx_data_path()),
                      ("anny.paths.get_anny2smpl_data_path", lambda: anny.paths.get_anny2smpl_data_path()),
                      ("anny.paths.download_noncommercial_data", lambda: anny.paths.download_noncommercial_data())):
        try:
            fn()
            refusals[label] = "NOT REFUSED"
        except (OB.OpenBodyRefused, ImportError) as e:
            refusals[label] = f"refused: {type(e).__name__}: {str(e)[:140]}"
    # files opened
    pkg_n = os.path.normcase(os.path.abspath(pkg))
    cache_n = os.path.normcase(os.path.abspath(cache))
    opened_pkg, opened_cache, writes = {}, {}, []
    for p, mode in EVENTS["open"]:
        ap_ = os.path.normcase(os.path.abspath(p))
        if any(c in mode for c in "wax+"):
            writes.append({"path": p.replace("\\", "/"), "mode": mode})
        if ap_.startswith(pkg_n + os.sep) and "__pycache__" not in ap_:
            opened_pkg[os.path.relpath(p, pkg).replace("\\", "/")] = mode
        elif ap_.startswith(cache_n + os.sep) or ap_ == cache_n:
            opened_cache[os.path.relpath(p, cache).replace("\\", "/")] = mode
    rec = record_hashes(os.path.join(site, f"anny-{anny.__version__}.dist-info"))
    integrity = {"checked": 0, "match": 0, "mismatch": []}
    dirs = {}
    digest = hashlib.sha256()
    for rel in sorted(opened_pkg):
        key = f"anny/{rel}"
        top = "/".join(rel.split("/")[:2]) if rel.startswith("data/") else rel.split("/")[0]
        dirs[top] = dirs.get(top, 0) + 1
        p = os.path.join(pkg, rel)
        if os.path.isfile(p):
            digest.update(rel.encode() + b"\0" + sha256_file(p).encode() + b"\0")
            if key in rec:
                integrity["checked"] += 1
                if b64sha(p) == rec[key]:
                    integrity["match"] += 1
                else:
                    integrity["mismatch"].append(rel)
    meta = open(os.path.join(site, f"anny-{anny.__version__}.dist-info", "METADATA"), encoding="utf-8").read()
    lic_section = meta[meta.index("## License"):meta.index("## Citation")]
    licences = {
        "anny code (Apache-2.0)": {"file": f"anny-{anny.__version__}.dist-info/licenses/LICENSE",
                                   "sha256": sha256_file(os.path.join(site, f"anny-{anny.__version__}.dist-info",
                                                                      "licenses", "LICENSE"))},
        "anny data/mpfb2 (CC0 1.0)": {"file": "anny/data/mpfb2/LICENSE.md",
                                      "sha256": sha256_file(os.path.join(pkg, "data", "mpfb2", "LICENSE.md"))},
        "README licence section (data/faceunits01 CC0, data/soma Apache-2.0, smplx non-commercial download)": {
            "file": f"anny-{anny.__version__}.dist-info/METADATA, from '## License' to '## Citation'",
            "sha256": hashlib.sha256(lic_section.encode("utf-8")).hexdigest(), "text": lic_section.strip()},
    }
    scan = static_scan(pkg)
    scan_files = sorted({h["file"] for h in scan})
    code_download = [h for h in scan if "requests.get" in h["text"] or "urlopen" in h["text"]]
    nonfree_entries = []
    try:
        OB.cache_audit()
    except OB.OpenBodyRefused as e:
        nonfree_entries.append(str(e))
    out = {
        "script": "scripts/openbody_audit.py", "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": "fresh (first use, cache built from nothing)" if a.fresh else "pinned cache",
        "anny_version": anny.__version__, "anny_package": pkg.replace("\\", "/"), "cache_dir": cache,
        "load": {"rig": "anny", "topology": "anny", "skinning": "lbs", "seconds": round(load_s, 2),
                 "bones": m.bone_count, "vertices": int(m.template_vertices.shape[0]),
                 "faces": int(m.faces.shape[0])},
        "network_events": EVENTS["network"], "process_events": EVENTS["process"],
        "files_written_by_python": writes,
        "cache_before": before, "cache_after": after,
        "cache_new_files": sorted(set(after) - set(before)),
        "cache_files_opened": opened_cache,
        "nonfree_cache_entries": nonfree_entries,
        "default_cache_~/.cache/anny": {"existed_before": home_before, "exists_after": os.path.exists(home_cache)},
        "modules": {m_: (m_ in sys.modules) for m_ in OB.BLOCKED_MODULES + ("requests", "warp", "trimesh",
                                                                             "anny.models.soma", "anny.keypoints")},
        "nonfree_packages_installed": OB.nonfree_packages(),
        "refusals": refusals,
        "anny_files_opened": {"count": len(opened_pkg), "by_folder": dirs,
                              "sha256_of_(path, sha256)_list": digest.hexdigest(),
                              "record_integrity": integrity, "files": sorted(opened_pkg)},
        "licences": licences,
        "static_scan": {"words": list(DOWNLOAD_WORDS), "files": scan_files, "hits": scan,
                        "network_calls_in_code": code_download},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "licence_audit_fresh.json" if a.fresh else "licence_audit.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("network events:", len(EVENTS["network"]), "process events:", len(EVENTS["process"]))
    print("files written by python:", len(writes), "cache new files:", len(out["cache_new_files"]))
    print("anny files opened:", len(opened_pkg), dirs, "integrity", integrity["match"], "/", integrity["checked"])
    print("refusals:", sum(v.startswith("refused") for v in refusals.values()), "/", len(refusals))
    print("network calls in anny code:", [(h["file"], h["line"], h["function"]) for h in code_download])
    print("wrote", os.path.relpath(path, ROOT))
    if a.fresh:
        shutil.rmtree(FRESH_DIR)
        print("removed", FRESH_DIR)


if __name__ == "__main__":
    main()
