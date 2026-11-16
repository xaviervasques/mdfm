"""Inventory of the 15 open motion-capture sets fetched into MoveAtlas on 2026-11-12 (shortlist:
reports/open_mocap/README.md; fetcher: fetch_open_mocap.py at the MoveAtlas root).

Reads, per dataset, MoveAtlas datasets/pose/<id>/fetch_manifest.json and fetch_log.json, and the archives in raw/
(zip members by zipfile, tar members by tarfile headers, 7z members by the system tar.exe listing; nothing is
extracted, imported or executed). Counts, per dataset: the files fetched with bytes and checksum status, the archive
members by format, the video and image members, the subjects as counted from the member paths (one rule per
dataset, stated in the report), and the frame rates and frame counts read from the C3D headers (the 512-byte header
block plus the processor type of the parameter block: Intel, DEC or MIPS), the BVH "Frame Time" line and the
Qualisys TSV "FREQUENCY" and "NO_OF_FRAMES" lines. Header reads stream only the first bytes of each member.

Writes reports/open_mocap/fetched.md and reports/open_mocap/fetched.json, with the MoveAtlas commit it read.

Run: <conda>/python -I scripts/open_mocap_inventory.py [--moveatlas <moveatlas>]
"""
import argparse
import collections
import datetime
import json
import os
import re
import struct
import subprocess
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "reports", "open_mocap")

ORDER = ["cmu_mocap", "ceti_age_kinematics", "lex3d", "horst_gait_fullbody", "schreiber_moissenet_gait",
         "lencioni_gait_stairs", "push_recovery_perturbations", "moore_perturbed_gait", "shida_pd_gait",
         "deepdance", "umons_taichi", "notredame_noncyclic", "style100", "sign_language_mocap_archive",
         "vanderzee_walking"]

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".wmv", ".mpg", ".mpeg", ".m4v", ".flv", ".3gp", ".mts"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".heic"}

# subject rule per dataset: a regex on the member path (archive name + "/" + member) whose group 1 names the
# subject; None when the files carry no subject key (stated as such in the report)
SUBJECT_RULES = {
    "cmu_mocap": r"subjects/(\d+)/",
    "ceti_age_kinematics": r"/(sub-d\d+)/",
    "horst_gait_fullbody": r"/(S\d+)_\d+_",
    "schreiber_moissenet_gait": r"/(20\d{5})/",
    "lencioni_gait_stairs": r"(?:^|/)Subject(\d+)\.mat$",
    "push_recovery_perturbations": r"Push-Recovery-OpenSource/(S\d+)_",
    "shida_pd_gait": r"/(SUB\d+)_(?:on|off)/",
    "deepdance": r"/Subject0*(\d+)_",  # Subject001 and Subject1 name the same dancer number
    "notredame_noncyclic": r"^(AB\d\d)\.zip/",
    "umons_taichi": r"/(P\d\d)T\d\dC\d\d\.c3d$",
    "vanderzee_walking": r"/(p\d+)_c3dfiles/",
}
SUBJECT_NOTES = {
    "lex3d": "counted as the distinct signer names on the first line of each *_V3D_MarkerData.csv (names not "
             "copied here).",
    "moore_perturbed_gait": "counted as the distinct subject ids of the trials' meta-*.yml files (subject: id).",
    "style100": "one actor (dataset description); the BVH names carry the style, not a person.",
    "sign_language_mocap_archive": "one signer (dataset description); this snapshot is the catalogue and "
                                   "documentation, without motion files.",
}


def ext_of(name):
    low = name.lower()
    for e in (".tar.gz",):
        if low.endswith(e):
            return e
    return os.path.splitext(low)[1] or "(none)"


def c3d_header(b):
    """(points, rate_hz, first, last, processor) from the first bytes of a C3D file, or None."""
    if len(b) < 512 or b[1] != 0x50:
        return None
    pblock = b[0]
    proc = None
    off = (pblock - 1) * 512
    if pblock >= 1 and len(b) >= off + 4:
        proc = {84: "intel", 85: "dec", 86: "mips"}.get(b[off + 3])
    if proc is None:
        proc = "intel"
    end = ">" if proc == "mips" else "<"
    npts, _, first, last = struct.unpack(end + "hhHH", b[2:10])
    raw = b[20:24]
    if proc == "dec":
        rate = struct.unpack("<f", bytes([raw[2], raw[3], raw[0], raw[1]]))[0] / 4.0
    else:
        rate = struct.unpack(end + "f", raw)[0]
    return npts, rate, first, last, proc


def bvh_header(b):
    t = b.decode("latin-1", "replace")
    m = re.search(r"Frames:\s*(\d+)\s*[\r\n]+\s*Frame Time:\s*([0-9.eE+-]+)", t)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def tsv_header(b):
    t = b.decode("latin-1", "replace")
    f = re.search(r"(?m)^FREQUENCY\t([0-9.]+)", t)
    n = re.search(r"(?m)^NO_OF_FRAMES\t(\d+)", t)
    m = re.search(r"(?m)^NO_OF_MARKERS\t(\d+)", t)
    if not f:
        return None
    return float(f.group(1)), int(n.group(1)) if n else None, int(m.group(1)) if m else None


def head_bytes(z, info, n):
    with z.open(info) as f:
        return f.read(n)


def c3d_bytes(z, info):
    b = head_bytes(z, info, 1024)
    if len(b) >= 512 and b[1] == 0x50 and b[0] > 2:
        b = head_bytes(z, info, b[0] * 512)
    return b


def list_members(path):
    """[(name, size)] of an archive, and a reader (name -> first bytes) where cheap."""
    low = path.lower()
    if low.endswith(".zip"):
        z = zipfile.ZipFile(path)
        infos = {i.filename: i for i in z.infolist() if not i.is_dir()}
        return "zip", [(n, i.file_size) for n, i in infos.items()], (z, infos)
    if low.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as t:
            return "tar", [(m.name, m.size) for m in t.getmembers() if m.isfile()], None
    if low.endswith(".7z"):
        tar = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
        out = subprocess.run([tar, "-tvf", path], capture_output=True, check=True).stdout.decode("utf-8", "replace")
        mem = []
        for ln in out.splitlines():
            p = ln.split(None, 8)
            if len(p) == 9 and not p[0].startswith("d"):
                mem.append((p[8], int(p[4])))
        return "7z", mem, None
    return None, [], None


def summarise_rates(rows):
    c = collections.Counter(round(r, 2) for r in rows)
    return ", ".join(f"{k:g} Hz x {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def inventory(root, sid):
    d = os.path.join(root, "datasets", "pose", sid)
    man = json.load(open(os.path.join(d, "fetch_manifest.json"), encoding="utf-8"))
    log = json.load(open(os.path.join(d, "fetch_log.json"), encoding="utf-8"))
    res = dict(dataset_id=sid, title=man["title"], licence=man["licence"]["name"],
               host_licence=man.get("host_licence"), source=man.get("host_record"), doi=man.get("doi"),
               files=[], excluded=man["excluded"], archives=[], formats=collections.Counter(),
               format_bytes=collections.Counter(), video=[], image=[], subjects=None, c3d=None, bvh=None,
               tsv=None)
    paths = []
    custom_subjects = set()
    for f in man["files"]:
        st = log["files"].get(f["name"], {})
        res["files"].append(dict(name=f["name"], bytes=st.get("bytes"), expected=f["bytes"],
                                 status=st.get("status"), checksum=st.get("checksum_status"),
                                 checksum_type=f["checksum_type"], sha256=st.get("sha256")))
        p = os.path.join(d, "raw", *f["name"].split("/"))
        if st.get("status") != "complete" or not os.path.exists(p):
            continue
        kind, mem, reader = list_members(p)
        if kind is None:
            paths.append((f["name"], f["bytes"]))
            res["formats"][ext_of(f["name"])] += 1
            res["format_bytes"][ext_of(f["name"])] += f["bytes"]
            continue
        res["archives"].append(dict(name=f["name"], kind=kind, members=len(mem),
                                    member_bytes=sum(s for _, s in mem),
                                    action=log.get("unpack", {}).get(f["name"], {}).get("action")))
        c3d, bvh, tsv = [], [], []
        sidecars = [n for n, _ in mem if n.startswith("__MACOSX/") or n.split("/")[-1].startswith("._")]
        res["macos_sidecars"] = res.get("macos_sidecars", 0) + len(sidecars)
        mem = [(n, s) for n, s in mem if n not in set(sidecars)]
        res["archives"][-1]["members"] = len(mem)
        res["archives"][-1]["member_bytes"] = sum(s for _, s in mem)
        for n, s in mem:
            e = ext_of(n)
            res["formats"][e] += 1
            res["format_bytes"][e] += s
            full = f["name"] + "/" + n
            paths.append((full, s))
            if e in VIDEO_EXT:
                res["video"].append(full)
            if e in IMAGE_EXT:
                res["image"].append(full)
            if reader is None:
                continue
            z, infos = reader
            try:
                if sid == "lex3d" and n.endswith("_V3D_MarkerData.csv"):
                    first = head_bytes(z, infos[n], 256).decode("latin-1").splitlines()[0].strip()
                    custom_subjects.add(first.lower())
                elif e == ".c3d":
                    h = c3d_header(c3d_bytes(z, infos[n]))
                    if h:
                        c3d.append(h)
                elif e == ".bvh":
                    h = bvh_header(head_bytes(z, infos[n], 1 << 16))
                    if h:
                        bvh.append(h)
                elif e == ".tsv":
                    h = tsv_header(head_bytes(z, infos[n], 4096))
                    if h:
                        tsv.append(h)
            except Exception:
                pass
        if sid == "moore_perturbed_gait" and kind == "tar":
            # one streaming pass: the subject id of each meta file, the sample interval of each mocap file
            with tarfile.open(p, "r|gz") as t:
                for m in t:
                    if not m.isfile():
                        continue
                    if re.search(r"meta-\d+\.yml$", m.name):
                        y = t.extractfile(m).read().decode("utf-8", "replace")
                        sm = re.search(r"(?m)^subject:\s*\n\s+id:\s*(\d+)", y)
                        if sm:
                            custom_subjects.add(sm.group(1))
                    elif re.search(r"mocap-\d+\.txt$", m.name):
                        # the TimeStamp column jitters row to row: the mean step over the rows of the first 4 MB
                        lines = t.extractfile(m).read(1 << 22).decode("latin-1").splitlines()[1:-1]
                        try:
                            ts = [float(ln.split("\t", 1)[0]) for ln in lines]
                            res.setdefault("text_rates", []).append(round((len(ts) - 1) / (ts[-1] - ts[0])))
                        except (ValueError, IndexError, ZeroDivisionError):
                            pass
        for key, rows in (("c3d", c3d), ("bvh", bvh), ("tsv", tsv)):
            if rows:
                res.setdefault(key + "_rows", []).extend(rows)
    if res.get("c3d_rows"):
        rows = res.pop("c3d_rows")
        frames = [(r[3] - r[2] + 1, r[1]) for r in rows if r[1] > 0 and r[3] >= r[2]]
        res["c3d"] = dict(n=len(rows), rates=summarise_rates([r[1] for r in rows]),
                          points=f"{min(r[0] for r in rows)}-{max(r[0] for r in rows)}",
                          processors=dict(collections.Counter(r[4] for r in rows)),
                          hours=sum(n / r for n, r in frames) / 3600.0,
                          capped_65535=sum(1 for r in rows if r[3] == 65535))
    if res.get("bvh_rows"):
        rows = res.pop("bvh_rows")
        res["bvh"] = dict(n=len(rows), rates=summarise_rates([1.0 / r[1] for r in rows if r[1] > 0]),
                          hours=sum(r[0] * r[1] for r in rows) / 3600.0)
    if res.get("tsv_rows"):
        rows = res.pop("tsv_rows")
        res["tsv"] = dict(n=len(rows), rates=summarise_rates([r[0] for r in rows]),
                          markers=f"{min(r[2] or 0 for r in rows)}-{max(r[2] or 0 for r in rows)}",
                          hours=sum((r[1] or 0) / r[0] for r in rows if r[0]) / 3600.0)
    if res.get("text_rates"):
        res["text_rates"] = summarise_rates(res["text_rates"])
    if custom_subjects:
        res["subjects"] = len(custom_subjects)
        res["subject_rule"] = "see note"
    rule = SUBJECT_RULES.get(sid)
    if rule:
        subj = set()
        for p, _ in paths:
            m = re.search(rule, p)
            if m:
                subj.add(m.group(1).lower())
        res["subjects"] = len(subj)
        res["subject_rule"] = rule
    res["subject_note"] = SUBJECT_NOTES.get(sid)
    res["formats"] = dict(res["formats"].most_common())
    res["format_bytes"] = dict(res["format_bytes"])
    return res


def gb(x):
    return f"{(x or 0) / 1e9:.3f}"


def md(results, head, when):
    L = ["# Open motion capture fetched into MoveAtlas, 2026-11-12", "",
         f"Written by scripts/open_mocap_inventory.py on {when}, read from MoveAtlas commit {head} (datasets/pose/<id>/:",
         "fetch_manifest.json, fetch_log.json and the archives in raw/, which git ignores). Fetcher:",
         "fetch_open_mocap.py at the MoveAtlas root. Sizes are decimal GB. Subjects are counted from the member paths",
         "with the rule given per dataset; frame rates and frame counts come from the C3D headers (header block),",
         "the BVH Frame Time line or the Qualisys TSV header, read from the first bytes of each member. Hours are",
         "the sum of frames over rate, static and calibration trials included. No datacard, registry row or",
         "canonical record exists yet for these sets.", "",
         "| Dataset | Licence | Files | GB | Checksums | Archive members | Subjects | Rates | Hours |",
         "|---|---|---|---|---|---|---|---|---|"]
    tot = 0
    for r in results:
        got = sum(f["bytes"] or 0 for f in r["files"] if f["status"] == "complete")
        tot += got
        ok = sum(1 for f in r["files"] if f["status"] == "complete")
        ver = sum(1 for f in r["files"] if f["checksum"] == "verified")
        so = sum(1 for f in r["files"] if f["checksum"] and f["checksum"].startswith("size only"))
        cs = f"{ver} verified" + (f", {so} size only" if so else "")
        rates, hours = [], None
        for k in ("c3d", "bvh", "tsv"):
            if r[k]:
                rates.append(f"{k.upper()}: {r[k]['rates']}")
                hours = (hours or 0.0) + r[k]["hours"]
        if r.get("text_rates"):
            rates.append(f"TXT: {r['text_rates']}")
        subj = r["subjects"] if r["subjects"] is not None else "n/a"
        L.append(f"| {r['dataset_id']} | {r['licence']} | {ok}/{len(r['files'])} | {gb(got)} | {cs} | "
                 f"{sum(a['members'] for a in r['archives'])} | {subj} | {'; '.join(rates) or 'n/a'} | "
                 f"{'n/a' if hours is None else f'{hours:.2f}'} |")
    L += ["", f"Total fetched: {gb(tot)} GB.", ""]
    for r in results:
        L += [f"## {r['dataset_id']}: {r['title']}", "",
              f"Source: {r['source']} (DOI {r['doi']}). Licence: {r['licence']}; as the host states it: "
              f"{r['host_licence']}.", ""]
        L += ["| File | Bytes | Status | Checksum |", "|---|---|---|---|"]
        for f in r["files"]:
            L.append(f"| {f['name']} | {f['bytes'] or 0} | {f['status']} | "
                     f"{f['checksum'] or ''} ({f['checksum_type']}) |")
        if r["excluded"]:
            ex = collections.OrderedDict()
            for e in r["excluded"]:
                ex.setdefault(e["reason"], []).append(e)
            L.append("")
            for reason, es in ex.items():
                b = sum(e.get("bytes") or 0 for e in es)
                names = ", ".join(e["name"] for e in es[:6]) + (f" and {len(es) - 6} more" if len(es) > 6 else "")
                L.append(f"- Not fetched ({len(es)} files{', ' + gb(b) + ' GB' if b else ''}): {names}. Why: {reason}.")
        L.append("")
        for a in r["archives"]:
            L.append(f"- Archive {a['name']} ({a['kind']}): {a['members']} members, {gb(a['member_bytes'])} GB "
                     f"unpacked size; {a['action'] or 'not unpacked yet'}.")
        L.append("- Formats: " + ", ".join(f"{k} {v} ({gb(r['format_bytes'].get(k))} GB)"
                                          for k, v in r["formats"].items()) + ".")
        if r.get("macos_sidecars"):
            L.append(f"- macOS sidecar members (__MACOSX/, ._ files) left out of every count and never extracted: "
                     f"{r['macos_sidecars']}.")
        L.append(f"- Video members: {len(r['video'])}" + (": " + ", ".join(r["video"][:10]) if r["video"] else "") + ".")
        L.append(f"- Image members: {len(r['image'])}" +
                 (" (first: " + ", ".join(r["image"][:3]) + ")" if r["image"] else "") + ".")
        if r["subjects"] is not None and r.get("subject_rule") != "see note":
            L.append(f"- Subjects counted: {r['subjects']} (rule: `{r['subject_rule']}` on archive/member paths).")
        elif r["subjects"] is not None:
            L.append(f"- Subjects counted: {r['subjects']}.")
        if r.get("text_rates"):
            L.append(f"- Text mocap sample rates (TimeStamp step of the first rows): {r['text_rates']}.")
        if r["subject_note"]:
            L.append(f"- Subjects: {r['subject_note']}")
        if r["c3d"]:
            c = r["c3d"]
            L.append(f"- C3D headers: {c['n']} files; rates {c['rates']}; {c['points']} points per frame; "
                     f"processor {c['processors']}; {c['hours']:.2f} h"
                     + (f"; {c['capped_65535']} files at the 65535-frame header cap (hours undercounted)"
                        if c["capped_65535"] else "") + ".")
        if r["bvh"]:
            L.append(f"- BVH headers: {r['bvh']['n']} files; rates {r['bvh']['rates']}; {r['bvh']['hours']:.2f} h.")
        if r["tsv"]:
            L.append(f"- TSV headers: {r['tsv']['n']} files; rates {r['tsv']['rates']}; markers {r['tsv']['markers']}; "
                     f"{r['tsv']['hours']:.2f} h.")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    head = subprocess.run(["git", "-c", f"safe.directory={a.moveatlas}", "-C", a.moveatlas, "rev-parse", "--short",
                           "HEAD"], capture_output=True, text=True).stdout.strip()
    when = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
    results = []
    for sid in ORDER:
        r = inventory(a.moveatlas, sid)
        results.append(r)
        print(sid, r["formats"], r["subjects"], r["c3d"] and r["c3d"]["rates"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fetched.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(moveatlas_commit=head, written=when, datasets=results), f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(os.path.join(OUT, "fetched.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md(results, head, when))


if __name__ == "__main__":
    main()
