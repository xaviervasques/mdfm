"""Study S1: measurement of dyskinetic cerebral palsy on modys_video with fixed image-plane formulas.

Rules: docs/rules/STUDY_FAMILY_2026-11-12.md and docs/rules/STUDY_S1_2026-11-12.md (registered on 2026-11-12,
docs/rules/gate_rule_hashes.csv), with the drafts and DR rules they adopt. Formulas: mdfm/clinical/s1_formulas.py.
Run order (S1 section 6), one stage per call:
  probe        section 2, step 1: rule version 2 of reports/modys_structure/rule.txt; writes probe_v2.* there and
               prints no reading
  movie        section 2, step 2: one stick-figure movie per zip of one training child, from the keypoints alone,
               into the session scratch directory given by --scratch (never into a tree)
  posture      combines step 1 and reports/modys_structure/viewing.md (refuses while viewing.md is absent)
  precision    the precision printout on synthetic fixtures (no datum read beyond identity counts)
  descriptors  the image-plane descriptors of every training record (keypoints only)
  outputs      outputs 1 to 6 and the readability card (reads the DIS item columns of training rows only)

Reads (MoveAtlas tree, read-only): datasets/pose/modys_video/raw/data_stickfigure_coordinates_lying.zip,
data_stickfigure_coordinates_sitting.zip and data_clinical_scoring.xlsx. This tree: reports/lockbox/lockbox_persons.csv
(used only to exclude), reports/modys_subjects/modys_video_person_map.csv, reports/nuisance/frame_rates_facts.json.
Writes: reports/modys_structure/ (probe_v2.*, posture_v2.*), reports/studies/s1/.

Lockbox: the 7 modys_video children of reports/lockbox/lockbox_persons.csv are excluded from the identity columns
of the sheet before any coordinate member or any item cell is read; an assertion stops the run if one of their ids
reaches the analysis. The xlsx reader decodes an item cell only on a row whose ID is a training child.

Run: PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/study_s1.py <stage>
     --moveatlas <moveatlas>  (or MOVEATLAS_ROOT)
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import zipfile
import zlib
from xml.etree import ElementTree as ET

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from mdfm.clinical import s1_formulas as F  # noqa: E402

FAMILY = "docs/rules/STUDY_FAMILY_2026-11-12.md"
S1_RULE = "docs/rules/STUDY_S1_2026-11-12.md"
S2_RULE = "docs/rules/STUDY_S2_2026-11-12.md"
REGISTERED = {
    FAMILY: "221c872e61ee60443e8e1bfb8af8c795378b4b89f354aaf1bcd249c570f0f0c4",
    S1_RULE: "028d21d24947d9439562832408c435bfa9e8fabfd5c211dc09fdc22136ea098b",
    S2_RULE: "cbc577aa12efedd3a3902edc5f7ee9f3a8aa08c9c094c34201b7a3b3431efc01",
}
LOCKBOX_CSV = "reports/lockbox/lockbox_persons.csv"
LOCKBOX_SHA256 = "619dfdacef464387ab5447299534e98b152507dea9bca91c95f68e3324762162"
PERSON_MAP = "reports/modys_subjects/modys_video_person_map.csv"
FPS_FACTS = "reports/nuisance/frame_rates_facts.json"
RULE_TXT = "reports/modys_structure/rule.txt"
RULE_V2_SHA256 = "f922c91c640bb6ab820845aa81086c4ce244d2f36ffbc42e700987d82923f082"
RAW = "datasets/pose/modys_video/raw"
ZIPS = ("data_stickfigure_coordinates_lying.zip", "data_stickfigure_coordinates_sitting.zip")
SHEET = "data_clinical_scoring.xlsx"
IDENTITY = ("ID", "video", "time", "scorer")
TIMES = ("T0", "T3", "T12")
COMPONENTS = {"D": "dystonia", "CA": "choreoathetosis"}
FACETS = {"tD": "duration", "tA": "amplitude"}
SHEET_REGIONS = {"TRUNK": ("trunk", None), "RAP": ("proximal_arm", "R"), "LAP": ("proximal_arm", "L"),
                 "RLP": ("proximal_leg", "R"), "LLP": ("proximal_leg", "L")}
ITEM_COLUMNS = tuple(f"{c}_{r}_R_{f}" for c in COMPONENTS for r in SHEET_REGIONS for f in FACETS)
ITEM_PATTERN = re.compile(r"^(D|CA)_[A-Z]+_R_t[DA]$")
NOT_SCORED = 999
# S1 draft OPEN-D53 / S2 draft 3.2: the facet map (component, facet) -> descriptor
FACET_MAP = {("dystonia", "duration"): "TON-DW", ("dystonia", "amplitude"): "TON-T",
             ("choreoathetosis", "duration"): "FLOW-OCC", ("choreoathetosis", "amplitude"): "FLOW-R"}
SCORED_POSTURE = {"trunk": "sitting", "proximal_arm": "sitting", "proximal_leg": "lying"}   # README, DR-14.14

OUT_S1 = os.path.join(HERE, "reports", "studies", "s1")
OUT_STRUCT = os.path.join(HERE, "reports", "modys_structure")


# ---------------------------------------------------------------- provenance
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_rules(paths=(FAMILY, S1_RULE)):
    out = {}
    for p in paths:
        h = sha256_file(os.path.join(HERE, p))
        assert h == REGISTERED[p], f"{p} differs from its registered SHA-256"
        out[p] = h
    return out


def moveatlas_commit(root):
    def git(*args):
        return subprocess.run(["git", "-c", f"safe.directory={root.replace(os.sep, '/')}", "-C", root, *args],
                              capture_output=True, text=True, encoding="utf-8").stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "status_modys_video": git("status", "--short", "--", RAW) or "clean"}


def header_lines(root, rules):
    ma = moveatlas_commit(root)
    lines = [f"- {p}: SHA-256 {h}" for p, h in rules.items()]
    lines.append(f"- MoveAtlas commit read: {ma['head']} (git status of {RAW}: {ma['status_modys_video']})")
    return lines, ma


def seed_of(label):
    return zlib.crc32(label.encode("utf-8")) & 0x7FFFFFFF


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------- lockbox and identity
def lockbox_children(here=HERE):
    """The modys_video persons of lockbox v1.2, used only to exclude; their ids are never printed."""
    path = os.path.join(here, LOCKBOX_CSV)
    assert sha256_file(path) == LOCKBOX_SHA256, "reports/lockbox/lockbox_persons.csv differs from lockbox v1.2"
    with open(path, encoding="utf-8", newline="") as f:
        ids = frozenset(r["person_id"] for r in csv.DictReader(f) if r["source"] == "modys_video")
    assert len(ids) == 7, "lockbox v1.2 holds 7 modys_video children"
    return ids


class LockboxGuard:
    """Fails the run when a lockbox child reaches the analysis."""

    def __init__(self, ids):
        self.ids = ids

    def check(self, children, where):
        bad = set(children) & self.ids
        assert not bad, f"lockbox child reached the analysis at {where}"


def child_id(raw_id):
    v = float(raw_id)
    assert v == int(v) and v > 0, "sheet ID is not a positive integer"
    return f"modys:child{int(v):02d}"


# ---------------------------------------------------------------- xlsx reader (identity first, items per row)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _shared_strings(z):
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{_NS}t")) for si in root.iter(f"{_NS}si")]


def _cell_value(c, ss):
    t = c.get("t")
    if t == "inlineStr":
        return "".join(x.text or "" for x in c.iter(f"{_NS}t"))
    v = c.find(f"{_NS}v")
    if v is None or v.text is None:
        return None
    if t == "s":
        return ss[int(v.text)]
    if t in ("str", "e"):
        return v.text
    return float(v.text)


def _col(ref):
    return re.match(r"([A-Z]+)", ref).group(1)


def read_sheet(path, lockbox, items=False):
    """Rows of the scoring sheet: identity columns for every row; item columns only for rows whose ID is a training
    child and only when items is True. The header is checked before any item cell: the item columns must be the
    20 rest items of ITEM_COLUMNS, and ID must precede every other column read."""
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        header, rows, cols = None, [], {}
        with z.open("xl/worksheets/sheet1.xml") as fh:
            row_vals, row_child, rnum = {}, None, None
            for ev, el in ET.iterparse(fh, events=("start", "end")):
                if ev == "start" and el.tag == f"{_NS}row":
                    row_vals, row_child, rnum = {}, None, int(el.get("r"))
                elif ev == "end" and el.tag == f"{_NS}c":
                    col = _col(el.get("r"))
                    if header is None:
                        row_vals[col] = _cell_value(el, ss)
                    else:
                        name = cols.get(col)
                        if name == "ID":
                            val = _cell_value(el, ss)
                            row_vals["ID"] = val
                            row_child = child_id(val) if val is not None else None
                        elif name in IDENTITY:
                            row_vals[name] = _cell_value(el, ss)
                        elif items and name in ITEM_COLUMNS and row_child is not None and row_child not in lockbox:
                            row_vals[name] = _cell_value(el, ss)
                    el.clear()
                elif ev == "end" and el.tag == f"{_NS}row":
                    if header is None:
                        header = {c: v for c, v in row_vals.items() if v is not None}
                        cols = {c: str(v) for c, v in header.items()}
                        names = list(cols.values())
                        for n in IDENTITY:
                            assert names.count(n) == 1, f"sheet column {n} missing or repeated"
                        item_like = {n for n in names if ITEM_PATTERN.match(n)}
                        assert item_like == set(ITEM_COLUMNS), "sheet region columns differ from the 20 rest items"
                        id_col = [c for c, n in cols.items() if n == "ID"][0]
                        order = sorted(cols, key=lambda c: (len(c), c))
                        assert order.index(id_col) == 0, "ID is not the first column of the sheet"
                    elif row_vals.get("ID") is not None:
                        r = {"row": rnum, "child": row_child}
                        for n in IDENTITY[1:]:
                            r[n] = row_vals.get(n)
                        if items and row_child not in lockbox:
                            r["items"] = {n: row_vals.get(n) for n in ITEM_COLUMNS}
                        rows.append(r)
                    el.clear()
    out = []
    for r in rows:
        r["video"] = int(r["video"])
        r["scorer"] = int(r["scorer"])
        r["time"] = str(r["time"])
        assert r["time"] in TIMES, "time value outside T0, T3, T12"
        out.append(r)
    return out


def identity(root, here=HERE):
    """Identity of the sheet (no item cell): per video its child and visit, training and lockbox video sets."""
    lock = lockbox_children(here)
    rows = read_sheet(os.path.join(root, RAW, SHEET), lock, items=False)
    child_of, time_of, scorers = {}, {}, {}
    for r in rows:
        assert child_of.setdefault(r["video"], r["child"]) == r["child"], "a video maps to two children"
        assert time_of.setdefault(r["video"], r["time"]) == r["time"], "a video carries two visits"
        scorers.setdefault(r["video"], []).append(r["scorer"])
    with open(os.path.join(here, PERSON_MAP), encoding="utf-8", newline="") as f:
        for m in csv.DictReader(f):
            v = int(m["video"])
            if v in child_of:
                assert child_of[v] == m["person_ns"], "person map and sheet disagree on a video"
    lock_videos = {v for v, c in child_of.items() if c in lock}
    train_videos = {v for v, c in child_of.items() if c not in lock}
    children = sorted({child_of[v] for v in train_videos})
    LockboxGuard(lock).check(children, "identity")
    return {"lock": lock, "rows": rows, "child_of": child_of, "time_of": time_of, "scorers": scorers,
            "train_videos": train_videos, "lock_videos": lock_videos, "children": children}


# ---------------------------------------------------------------- coordinate reader
def video_of(member):
    return int(os.path.basename(member).split("_")[0])


def zip_members(zf):
    return sorted(n for n in zf.namelist() if n.lower().endswith(".csv") and "__MACOSX" not in n
                  and re.match(r"\d{3}_", os.path.basename(n)))


def parse_dlc(zf, member, train_videos, lock_videos):
    v = video_of(member)
    assert v in train_videos and v not in lock_videos, "refused: not a training video"
    raw = zf.read(member)       # read only after the guard
    rows = [r for r in csv.reader(io.StringIO(raw.decode("utf-8"))) if r]
    assert rows[0][0] == "scorer" and rows[1][0] == "bodyparts" and rows[2][0] == "coords", "not a DeepLabCut CSV"
    parts, coords = rows[1][1:], rows[2][1:]
    names = list(dict.fromkeys(parts))
    for i, bp in enumerate(names):
        assert parts[3 * i:3 * i + 3] == [bp] * 3 and coords[3 * i:3 * i + 3] == ["x", "y", "likelihood"]
    data = np.array([[float(x) if x != "" else np.nan for x in r[1:]] for r in rows[3:]], dtype=float)
    assert tuple(names) == F.POINTS, "the keypoint set differs from the one ordered set of both zips"
    xy = np.stack([data[:, 0::3], data[:, 1::3]], axis=-1)
    return {"names": tuple(names), "xy": xy, "like": data[:, 2::3], "scorers": sorted(set(rows[0][1:]))}


def load_records(root, ident, log=print):
    """Every training record of both zips; lockbox and childless members are skipped before any member is read."""
    recs = []
    for zname in ZIPS:
        with zipfile.ZipFile(os.path.join(root, RAW, zname)) as zf:
            members = zip_members(zf)
            keep = [m for m in members if video_of(m) in ident["train_videos"]]
            skipped = len(members) - len(keep)
            assert not {video_of(m) for m in keep} & ident["lock_videos"]
            assert len({video_of(m) for m in keep}) == len(keep), "two members share a video number"
            log(f"{zname}: {len(members)} coordinate CSVs, {skipped} skipped before reading (lockbox child or no "
                f"child on the sheet), {len(keep)} training records to parse")
            for m in keep:
                rec = parse_dlc(zf, m, ident["train_videos"], ident["lock_videos"])
                v = video_of(m)
                rec.update({"zip": zname, "member": os.path.basename(m), "video": v,
                            "child": ident["child_of"][v], "time": ident["time_of"][v], "rows": len(rec["xy"])})
                recs.append(rec)
    LockboxGuard(ident["lock"]).check({r["child"] for r in recs}, "records")
    return recs


def source_fps(here=HERE):
    with open(os.path.join(here, FPS_FACTS), encoding="utf-8") as f:
        facts = json.load(f)
    return float(facts["modys_source_level_fps"]), facts.get("modys_records_by_evidence", {})


def root_of(args):
    root = args.moveatlas or os.environ.get("MOVEATLAS_ROOT")
    assert root, "pass --moveatlas or set MOVEATLAS_ROOT"
    return os.path.abspath(root)


# ---------------------------------------------------------------- stage probe (section 2, step 1)
def probe_q(rec):
    names = rec["names"]
    idx = {n: i for i, n in enumerate(names)}
    xy, lk = rec["xy"], rec["like"]
    need = ["hip1", "hip2", "knee1", "knee2", "shoulder1", "shoulder2"]
    ok = np.ones(len(xy), dtype=bool)
    for n in need:
        ok &= np.isfinite(xy[:, idx[n]]).all(axis=1) & (np.nan_to_num(lk[:, idx[n]], nan=-1) >= F.LIKELIHOOD_MIN)
    H = (xy[:, idx["hip1"]] + xy[:, idx["hip2"]]) / 2
    K = (xy[:, idx["knee1"]] + xy[:, idx["knee2"]]) / 2
    S = (xy[:, idx["shoulder1"]] + xy[:, idx["shoulder2"]]) / 2
    L = np.linalg.norm(S - H, axis=1)
    ok &= np.isfinite(L) & (L > 0)
    if not ok.any():
        return math.nan, 0
    r = np.linalg.norm(K[ok] - H[ok], axis=1) / L[ok]
    return float(np.median(r)), int(ok.sum())


def probe_reading(per_zip_q):
    """Rule version 2, sections V2.3: per zip Q, reading, side counts and the settled flag."""
    Q = {z: (float(np.median([q for q in qs if math.isfinite(q)])) if any(math.isfinite(q) for q in qs)
             else math.nan) for z, qs in per_zip_q.items()}
    zs = list(per_zip_q)
    res = {"Q": Q, "reading": None, "midpoint": None, "on_side": {}, "n": {}, "fraction": {}, "settled": False}
    if any(not math.isfinite(Q[z]) for z in zs) or Q[zs[0]] == Q[zs[1]]:
        return res
    lying = max(zs, key=lambda z: Q[z])
    sitting = [z for z in zs if z != lying][0]
    M = (Q[lying] + Q[sitting]) / 2
    res["reading"] = {lying: "lying", sitting: "sitting"}
    res["midpoint"] = M
    for z in zs:
        qs = per_zip_q[z]
        on = sum(1 for q in qs if math.isfinite(q) and ((q > M) if z == lying else (q < M)))
        res["on_side"][z], res["n"][z] = on, len(qs)
        res["fraction"][z] = on / len(qs) if qs else math.nan
    res["settled"] = all(res["fraction"][z] >= 0.9 for z in zs)
    return res


def stage_probe(args):
    root = root_of(args)
    rules = check_rules()
    rule_hash = sha256_file(os.path.join(HERE, RULE_TXT))
    with open(os.path.join(HERE, RULE_TXT + ".sha256"), encoding="utf-8") as f:
        stored = f.readline().split()[0]
    assert rule_hash == RULE_V2_SHA256 == stored, "rule.txt differs from the version 2 hash held by the script"
    ident = identity(root)
    recs = load_records(root, ident, log=lambda m: None)
    per_zip, rows = {z: [] for z in ZIPS}, []
    for r in sorted(recs, key=lambda r: (r["zip"], r["video"])):
        q, nf = probe_q(r)
        per_zip[r["zip"]].append(q)
        rows.append({"zip": r["zip"], "video": f"{r['video']:03d}", "time": r["time"], "frames_read": nf,
                     "q": q})
    res = probe_reading(per_zip)
    hdr, ma = header_lines(root, rules)
    facts = {"rule": RULE_TXT, "rule_sha256": rule_hash, "rules": rules, "moveatlas_head": ma["head"],
             "records": {z: len(v) for z, v in per_zip.items()},
             "records_without_q": {z: sum(1 for q in v if not math.isfinite(q)) for z, v in per_zip.items()},
             "Q": res["Q"], "reading": res["reading"], "midpoint": res["midpoint"], "on_side": res["on_side"],
             "fraction_on_side": res["fraction"], "settled": res["settled"]}
    write_csv(os.path.join(OUT_STRUCT, "probe_v2_per_record.csv"), rows,
              ["zip", "video", "time", "frames_read", "q"])
    with open(os.path.join(OUT_STRUCT, "probe_v2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1, sort_keys=True, default=lambda x: None)
        f.write("\n")
    md = ["# modys_video structure probe, rule version 2 (S1 section 2, step 1)", "",
          f"Rule: {RULE_TXT}, SHA-256 {rule_hash} (version 2; version 1 is bytes 0 to 8268).",
          "Script: scripts/study_s1.py, stage probe. Training records only; no DIS item column read.", ""] + hdr + [
          "", "| zip | training records | records without q | Q (median of q) | reading | records on its side "
          "| fraction |", "|---|---|---|---|---|---|---|"]
    for z in ZIPS:
        md.append(f"| {z} | {facts['records'][z]} | {facts['records_without_q'][z]} | {fmt(res['Q'][z])} | "
                  f"{(res['reading'] or {}).get(z, 'none')} | {res['on_side'].get(z, 'n/a')} | "
                  f"{fmt(res['fraction'].get(z))} |")
    md += ["", f"Midpoint of the two Q: {fmt(res['midpoint'])}. Step 1 settled (each zip at least 0.90 on its side): "
           f"{'yes' if res['settled'] else 'no'}.", "",
           "The posture of a zip is settled only when this step and the viewing of step 2 "
           "(reports/modys_structure/viewing.md) agree (stage posture).", ""]
    write_text(os.path.join(OUT_STRUCT, "probe_v2.md"), "\n".join(md))
    print("probe written: reports/modys_structure/probe_v2.json, probe_v2_per_record.csv, probe_v2.md (not shown)")
    for o in ("probe_v2.json", "probe_v2_per_record.csv", "probe_v2.md"):
        print(sha256_file(os.path.join(OUT_STRUCT, o)), o)


# ---------------------------------------------------------------- stage movie (section 2, step 2)
BONES = [("ankle1", "knee1"), ("knee1", "hip1"), ("hip1", "hip2"), ("hip2", "knee2"), ("knee2", "ankle2"),
         ("wrist1", "elbow1"), ("elbow1", "shoulder1"), ("shoulder1", "shoulder2"), ("shoulder2", "elbow2"),
         ("elbow2", "wrist2"), ("shoulder1", "hip1"), ("shoulder2", "hip2"), ("chin", "forehead")]


def movie_video(ident, recs):
    held = {}
    for r in recs:
        held.setdefault(r["video"], set()).add(r["zip"])
    cands = sorted(v for v, zs in held.items() if len(zs) == 2 and ident["time_of"][v] == "T0")
    return cands[0]


def stage_movie(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    root = root_of(args)
    scratch = os.path.abspath(args.scratch)
    for tree in (HERE, root):
        assert not os.path.normcase(scratch).startswith(os.path.normcase(os.path.abspath(tree))), \
            "the movie never goes into a tree"
    os.makedirs(scratch, exist_ok=True)
    ident = identity(root)
    recs = load_records(root, ident, log=lambda m: None)
    v = movie_video(ident, recs)
    fps, _ = source_fps()
    for r in recs:
        if r["video"] != v:
            continue
        idx = {n: i for i, n in enumerate(r["names"])}
        ok = np.isfinite(r["xy"]).all(axis=-1) & (np.nan_to_num(r["like"], nan=-1) >= F.LIKELIHOOD_MIN)
        xs, ys = r["xy"][..., 0][ok], r["xy"][..., 1][ok]
        lim = (np.percentile(xs, 1) - 40, np.percentile(xs, 99) + 40, np.percentile(ys, 99) + 40,
               np.percentile(ys, 1) - 40)
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=80)

        def draw(t, ax=ax, r=r, idx=idx, ok=ok, lim=lim):
            ax.clear()
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])          # image y down
            ax.set_aspect("equal")
            for a, b in BONES:
                if ok[t, idx[a]] and ok[t, idx[b]]:
                    ax.plot(*zip(r["xy"][t, idx[a]], r["xy"][t, idx[b]]),
                            color="tab:red" if a.endswith("1") or b.endswith("1") else "tab:blue", lw=2)
            for n, i in idx.items():
                if ok[t, i]:
                    ax.plot(*r["xy"][t, i], "o", color="k", ms=3)
                    ax.text(*r["xy"][t, i], n, fontsize=6)
            ax.set_title(f"{r['zip']} t = {t / fps:.2f} s")
            return []
        stem = os.path.join(scratch, f"{r['zip'][:-4]}_stick")
        anim = FuncAnimation(fig, draw, frames=range(0, r["rows"], 2), blit=False)
        anim.save(stem + ".gif", writer=PillowWriter(fps=int(fps // 2)))
        plt.close(fig)
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=70)
        for ax2, t in zip(axes.ravel(), np.linspace(0, r["rows"] - 1, 8).astype(int)):
            draw(int(t), ax=ax2)
        fig.tight_layout()
        fig.savefig(stem + "_sheet.png")
        plt.close(fig)
        print("movie written:", stem + ".gif", stem + "_sheet.png")


# ---------------------------------------------------------------- stage posture
def read_viewing(path):
    """{zip name: posture} from the lines 'posture <zip name>: lying|sitting' of viewing.md."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    seen = dict(re.findall(r"^posture (data_stickfigure_coordinates_\w+\.zip): (lying|sitting)\s*$", text, re.M))
    assert set(seen) == set(ZIPS), "viewing.md names no posture for each zip"
    return seen


def stage_posture(args):
    root = root_of(args)
    rules = check_rules()
    vpath = os.path.join(OUT_STRUCT, "viewing.md")
    assert os.path.exists(vpath), "reports/modys_structure/viewing.md is absent: step 2 comes first"
    viewing = read_viewing(vpath)
    with open(os.path.join(OUT_STRUCT, "probe_v2.json"), encoding="utf-8") as f:
        probe = json.load(f)
    assert probe["rule_sha256"] == RULE_V2_SHA256
    agree = probe["settled"] and probe["reading"] == viewing
    posture = dict(viewing) if agree else None
    hdr, ma = header_lines(root, rules)
    res = {"step1_settled": probe["settled"], "step1_reading": probe["reading"], "step2_viewing": viewing,
           "settled": bool(agree), "posture": posture, "moveatlas_head": ma["head"],
           "viewing_md_sha256": sha256_file(vpath), "probe_v2_json_sha256": sha256_file(
               os.path.join(OUT_STRUCT, "probe_v2.json"))}
    with open(os.path.join(OUT_STRUCT, "posture_v2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
        f.write("\n")
    md = ["# Posture of the two modys_video zips (S1 section 2)", ""] + hdr + [
          f"- reports/modys_structure/viewing.md: SHA-256 {res['viewing_md_sha256']}",
          f"- reports/modys_structure/probe_v2.json: SHA-256 {res['probe_v2_json_sha256']}", "",
          "| zip | step 1 (rule version 2) | step 2 (viewing) |", "|---|---|---|"]
    for z in ZIPS:
        md.append(f"| {z} | {(probe['reading'] or {}).get(z, 'none')} | {viewing[z]} |")
    md += ["", f"Step 1 settled: {'yes' if probe['settled'] else 'no'}. The two steps agree: "
           f"{'yes' if agree else 'no'}. Posture settled: {'yes' if agree else 'no'}.", ""]
    if agree:
        contra = [z for z in ZIPS if z.split("_")[-1][:-4] != posture[z]]
        md.append("Zips whose posture differs from the posture word of their own file name: "
                  + (", ".join(contra) if contra else "none") + ".")
    else:
        md.append("The posture stays open: every output is labelled by zip name, the outputs that pair a descriptor "
                  "with a DIS item in its scored posture are withdrawn, and S2 does not run (S1 section 2).")
    write_text(os.path.join(OUT_STRUCT, "posture_v2.md"), "\n".join(md) + "\n")
    print("\n".join(md))


def load_posture():
    with open(os.path.join(OUT_STRUCT, "posture_v2.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- stage fixtures (S1 section 6, step 2)
FIXTURE_TESTS = ("tests/test_s1_formulas.py", "tests/test_study_s1.py")
FORMULAS = os.path.join("mdfm", "clinical", "s1_formulas.py")


def stage_fixtures(args):
    """Runs the synthetic fixtures of the descriptor library and the readers; the first kill condition of S1 draft
    section 7: when one fails, S1 does not run (the later stages refuse)."""
    rules = check_rules()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *FIXTURE_TESTS], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8", env=env)
    lines = [ln for ln in p.stdout.splitlines() if "::" in ln or ln.startswith("=")]
    passed = p.returncode == 0
    head = ["S1 fixture log (S1 section 6, step 2; DR-0.10; S1 draft section 7, first kill condition)"]
    head += [f"{k}: SHA-256 {v}" for k, v in rules.items()]
    head += [f"{FORMULAS.replace(os.sep, '/')}: SHA-256 {sha256_file(os.path.join(HERE, FORMULAS))}",
             "scripts/study_s1.py: SHA-256 " + sha256_file(os.path.join(HERE, "scripts", "study_s1.py"))]
    head += [f"{t}: SHA-256 {sha256_file(os.path.join(HERE, t))}" for t in FIXTURE_TESTS]
    head.append(f"status: {'PASSED' if passed else 'FAILED: S1 does not run'}")
    write_text(os.path.join(OUT_S1, "fixtures.log"), "\n".join(head + [""] + lines) + "\n")
    print("\n".join(head))
    if not passed:
        sys.exit(1)


def require_fixtures():
    path = os.path.join(OUT_S1, "fixtures.log")
    assert os.path.exists(path), "the fixtures have not run (stage fixtures)"
    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "status: PASSED" in text, "the descriptor fixtures failed: S1 does not run"
    assert sha256_file(os.path.join(HERE, FORMULAS)) in text, "s1_formulas.py changed since its fixtures ran"


# ---------------------------------------------------------------- stage precision (before any label)
def synthetic_icc_halfwidth(n_targets_per_cluster, icc, rng, B, stat):
    """Median half-width of the child-clustered interval of ICC(1,1) or ICC(2,1) on synthetic ratings."""
    clusters = []
    for m in n_targets_per_cluster:
        t = rng.normal(0, math.sqrt(icc), m)[:, None] + rng.normal(0, math.sqrt(1 - icc), (m, 2))
        clusters.append(t)
    point = stat(np.vstack(clusters))
    lo, hi, _ = F.cluster_bootstrap(range(len(clusters)), lambda d: stat(np.vstack([clusters[i] for i in d])), B=B,
                                    seed=int(rng.integers(1 << 30)))
    return point, (hi - lo) / 2


def stage_precision(args):
    """Interval half-widths of S1's statistics on synthetic fixtures with the training counts and planted values
    (counts only, NR-0.5); also the Fisher-z formula of S1 draft section 4."""
    root = root_of(args)
    rules = check_rules()
    require_fixtures()
    ident = identity(root)
    recs = load_records(root, ident, log=lambda m: None)
    fps, _ = source_fps()
    n_children = len(ident["children"])
    dbl = {}
    for v in ident["train_videos"]:
        if len(ident["scorers"][v]) == 2:
            dbl.setdefault(ident["child_of"][v], 0)
            dbl[ident["child_of"][v]] += 1
    per_child_dbl = sorted(dbl.values())
    span_ok = {}
    for r in recs:
        if round((r["rows"] - 1) / fps, 2) in (19.96, 20.0):
            span_ok.setdefault(r["zip"], {}).setdefault(r["child"], 0)
            span_ok[r["zip"]][r["child"]] += 1
    visits = {}
    for v in ident["train_videos"]:
        visits.setdefault(ident["child_of"][v], set()).add(ident["time_of"][v])
    n_t0t12 = sum(1 for s in visits.values() if {"T0", "T12"} <= s)
    n_t0t3 = sum(1 for s in visits.values() if {"T0", "T3"} <= s)
    rng = np.random.default_rng(seed_of("precision"))
    reps, B = 50, F.BOOT_B
    hdr, ma = header_lines(root, rules)
    md = ["# S1 precision printout on synthetic fixtures (S1 section 6, step 3)", "",
          "Synthetic data only, with the training counts recomputed from the identity columns of the sheet and the "
          "zip members (no DIS item and no descriptor value read). Planted values are inputs, not estimates.", ""] + hdr
    md += ["", f"Counts: {n_children} training children; {sum(per_child_dbl)} double-scored training videos in "
           f"{len(per_child_dbl)} children; children with T0 and T12 on the sheet {n_t0t12}, with T0 and T3 "
           f"{n_t0t3}; records spanning 19.96 or 20.0 s: "
           + ", ".join(f"{z} {sum(c.values())} in {len(c)} children" for z, c in sorted(span_ok.items())) + ".", ""]
    md += ["## Fisher-z half-width of a Spearman correlation (S1 draft section 4)", "",
           "| n | 1.96 x 1.03 / sqrt(n - 3), z units | interval at rho = 0 | at rho = 0.5 |", "|---|---|---|---|"]
    for n in (n_children, n_t0t12, n_t0t3):
        h = F.fisher_halfwidth(n)
        lo5, hi5 = math.tanh(math.atanh(0.5) - h), math.tanh(math.atanh(0.5) + h)
        md.append(f"| {n} | {fmt(h)} | [{fmt(-math.tanh(h))}, {fmt(math.tanh(h))}] | [{fmt(lo5)}, {fmt(hi5)}] |")
    md += ["", f"## Child-clustered bootstrap half-widths ({B} replicates, median over {reps} synthetic data sets)",
           "", "| statistic | units | planted value | median point | median half-width |", "|---|---|---|---|---|"]
    for rho in (0.0, 0.3, 0.5, 0.7):
        hw, pts = [], []
        for _ in range(reps):
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n_children)
            pts.append(F.spearman(z[:, 0], z[:, 1]))
            lo, hi, _ = F.cluster_bootstrap(range(n_children), lambda d, z=z: F.spearman(z[d, 0], z[d, 1]), B=B,
                                            seed=int(rng.integers(1 << 30)))
            hw.append((hi - lo) / 2)
        md.append(f"| Spearman | {n_children} children | {rho} | {fmt(float(np.median(pts)))} | "
                  f"{fmt(float(np.median(hw)))} |")
    for icc in (0.3, 0.5, 0.7, 0.9):
        res = [synthetic_icc_halfwidth(per_child_dbl, icc, rng, B, F.icc_1_1) for _ in range(reps)]
        md.append(f"| scorer ICC(1,1) | {sum(per_child_dbl)} videos, {len(per_child_dbl)} children | {icc} | "
                  f"{fmt(float(np.median([r[0] for r in res])))} | {fmt(float(np.median([r[1] for r in res])))} |")
    for z, c in sorted(span_ok.items()):
        for icc in (0.3, 0.5, 0.7, 0.9):
            res = [synthetic_icc_halfwidth(sorted(c.values()), icc, rng, B, F.icc_2_1) for _ in range(reps)]
            md.append(f"| within-clip ICC(2,1), {z} | {sum(c.values())} records, {len(c)} children | {icc} | "
                      f"{fmt(float(np.median([r[0] for r in res])))} | {fmt(float(np.median([r[1] for r in res])))} |")
    for n in (n_t0t12, n_t0t3):
        for d in (0.0, 0.5, 1.0):
            hw, pts = [], []
            for _ in range(reps):
                ch = rng.normal(d, 1.0, n)
                pts.append(F.srm(ch))
                lo, hi, _ = F.cluster_bootstrap(range(n), lambda dd, ch=ch: F.srm(ch[dd]), B=B,
                                                seed=int(rng.integers(1 << 30)))
                hw.append((hi - lo) / 2)
            md.append(f"| standardized response mean | {n} children | {d} | {fmt(float(np.median(pts)))} | "
                      f"{fmt(float(np.median(hw)))} |")
    md += ["", "A bound of DR-18.1 form, 1.96 x sqrt(2) x SD x sqrt(1 - ICC), on the synthetic scale SD = 1: "
           + ", ".join(f"ICC {i}: {fmt(F.bound(math.sqrt(1 - i)))}" for i in (0.3, 0.5, 0.7, 0.9)) + ".", ""]
    write_text(os.path.join(OUT_S1, "precision.md"), "\n".join(md))
    print("\n".join(md))


# ---------------------------------------------------------------- stage descriptors (keypoints only)
DESC_FIELDS = ["zip", "video", "child_index", "time", "rows", "span_s", "n_windows", "usable_fraction",
               "window_usable_median", "region", "descriptor", "unit", "record", "half1", "half2",
               "side1", "side2", "windows_entered_side1", "windows_entered_side2", "seg_len_tl", "torso_px", "flags"]


def compute_descriptors(recs, fps, children):
    cidx = {c: i for i, c in enumerate(children)}
    rows = []
    for r in sorted(recs, key=lambda r: (r["zip"], r["video"])):
        d = F.record_descriptors(r["xy"], r["like"], fps, r["names"])
        span = round((r["rows"] - 1) / fps, 2)
        for region in F.REGIONS:
            g = d["regions"][region]
            sides = list(g["sides"].values())
            for n in F.DESCRIPTORS[region]:
                halves = g["halves"] if span in (19.96, 20.0) else []
                flags = ["image_plane"] + (["laterality_unknown"] if region != "trunk" else [])
                if n == "TRUNK-ANGLE-RAW":
                    flags.append("raw, not a deviation")
                rows.append({
                    "zip": r["zip"], "video": f"{r['video']:03d}", "child_index": cidx[r["child"]], "time": r["time"],
                    "rows": r["rows"], "span_s": span, "n_windows": d["n_windows"],
                    "usable_fraction": d["usable_fraction"],
                    "window_usable_median": (float(np.median(d["window_usable_fractions"]))
                                             if d["window_usable_fractions"] else math.nan),
                    "region": region, "descriptor": n, "unit": F.UNITS[n], "record": g["record"][n],
                    "half1": halves[0][n] if halves else math.nan, "half2": halves[1][n] if halves else math.nan,
                    "side1": sides[0]["record"][n], "side2": sides[-1]["record"][n] if len(sides) > 1 else math.nan,
                    "windows_entered_side1": sides[0]["n_windows_entered"][n],
                    "windows_entered_side2": sides[-1]["n_windows_entered"][n] if len(sides) > 1 else "",
                    "seg_len_tl": g["seg_len_tl"], "torso_px": sides[0]["torso_px"], "flags": ";".join(flags)})
    return rows


def stage_descriptors(args):
    root = root_of(args)
    rules = check_rules()
    require_fixtures()
    ident = identity(root)
    recs = load_records(root, ident)
    fps, evidence = source_fps()
    rows = compute_descriptors(recs, fps, ident["children"])
    write_csv(os.path.join(OUT_S1, "record_descriptors.csv"), rows, DESC_FIELDS)
    hdr, ma = header_lines(root, rules)
    md = ["# S1 image-plane descriptors per training record (keypoints only)", ""] + hdr + [
          f"- rate: {fps} frames per second, the source-level rate of {FPS_FACTS} (evidence classes {evidence})",
          f"- records: {len(recs)} (" + ", ".join(f"{z} {sum(r['zip'] == z for r in recs)}" for z in ZIPS) + ")",
          "- formulas: mdfm/clinical/s1_formulas.py, SHA-256 "
          + sha256_file(os.path.join(HERE, "mdfm", "clinical", "s1_formulas.py")),
          "- table: reports/studies/s1/record_descriptors.csv (child_index is the rank of the child id among the "
          "27 training children)", ""]
    write_text(os.path.join(OUT_S1, "record_descriptors.md"), "\n".join(md))
    print("\n".join(md))


# ---------------------------------------------------------------- labels (training rows only)
def item_value(v):
    if v is None:
        return math.nan
    v = float(v)
    if v == NOT_SCORED:
        return math.nan
    assert v in (0.0, 1.0, 2.0, 3.0, 4.0), "an item value lies outside 0 to 4 and 999"
    return v


def region_items(items):
    """The 12 region items of one sheet row (component, facet, region), side-pooled as the mean of right and
    left (OPEN-D53; NaN unless both are scored), plus the 20 raw items."""
    raw = {n: item_value(items[n]) for n in ITEM_COLUMNS}
    out = {}
    for c, comp in COMPONENTS.items():
        for f, facet in FACETS.items():
            out[(comp, facet, "trunk")] = raw[f"{c}_TRUNK_R_{f}"]
            for region, (r_, l_) in (("proximal_arm", ("RAP", "LAP")), ("proximal_leg", ("RLP", "LLP"))):
                a, b = raw[f"{c}_{r_}_R_{f}"], raw[f"{c}_{l_}_R_{f}"]
                out[(comp, facet, region)] = (a + b) / 2 if math.isfinite(a) and math.isfinite(b) else math.nan
    for c, comp in COMPONENTS.items():
        vals = [raw[n] for n in ITEM_COLUMNS if n.startswith(c + "_")]
        out[(comp, "subscale", "all")] = float(sum(vals)) if all(math.isfinite(x) for x in vals) else math.nan
    return out, raw


def load_labels(root, ident):
    rows = read_sheet(os.path.join(root, RAW, SHEET), ident["lock"], items=True)
    lab = []
    for r in rows:
        if r["child"] in ident["lock"]:
            assert "items" not in r
            continue
        reg, raw = region_items(r["items"])
        lab.append({"video": r["video"], "child": r["child"], "time": r["time"], "scorer": r["scorer"],
                    "region": reg, "raw": raw})
    LockboxGuard(ident["lock"]).check({x["child"] for x in lab}, "labels")
    return lab


ITEM_KEYS = [(comp, facet, region) for comp in COMPONENTS.values() for facet in FACETS.values()
             for region in F.REGIONS]


def per_video_ratings(lab):
    out = {}
    for x in lab:
        out.setdefault(x["video"], {})[x["scorer"]] = x
    return out


def item_mean(ratings, key, raw=False):
    vals = [(r["raw"] if raw else r["region"])[key] for r in ratings.values()]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if len(vals) == len(ratings) and vals else math.nan


# ---------------------------------------------------------------- stage outputs
def _pairs(rv, key, raw, pair=None):
    """[(child, (a, b))] of double-scored videos for an item (pair: the two scorers, else any two)."""
    out = []
    for v, rs in rv.items():
        if len(rs) != 2:
            continue
        sc = sorted(rs)
        if pair is not None and sc != list(pair):
            continue
        a, b = ((rs[s]["raw"] if raw else rs[s]["region"])[key] for s in sc)
        out.append((rs[sc[0]]["child"], (a, b)))
    return out


def _by_child(pairs):
    d = {}
    for c, ab in pairs:
        d.setdefault(c, []).append(ab)
    return d


def _mat(byc, draw):
    rows = [ab for c in draw for ab in byc.get(c, [])]
    return np.array(rows, dtype=float).reshape(-1, 2)


def agreement_rows(rv, keys, raw):
    rows = []
    for key in keys:
        pairs = [(c, ab) for c, ab in _pairs(rv, key, raw) if all(math.isfinite(x) for x in ab)]
        byc = _by_child(pairs)
        ch = sorted(byc)
        m = _mat(byc, ch)
        label = key if raw else "/".join(key)
        icc = F.icc_1_1(m)
        s = F.sem(m, icc)
        ci = F.cluster_bootstrap(ch, lambda d: F.icc_1_1(_mat(byc, d)), seed=seed_of("icc11" + str(label)))
        bci = F.cluster_bootstrap(ch, lambda d: F.bound(F.sem(_mat(byc, d), F.icc_1_1(_mat(byc, d)))),
                                  seed=seed_of("bound" + str(label)))
        row = {"item": label, "videos": len(m), "children": len(ch), "icc11": icc, "icc11_lo": ci[0],
               "icc11_hi": ci[1], "sem": s, "bound": F.bound(s), "bound_lo": bci[0], "bound_hi": bci[1],
               "qwk": F.qwk(m[:, 0], m[:, 1]) if len(m) else math.nan}
        for pair in ((1, 2), (1, 3)):
            pp = [(c, ab) for c, ab in _pairs(rv, key, raw, pair) if all(math.isfinite(x) for x in ab)]
            pbc = _by_child(pp)
            pm = _mat(pbc, sorted(pbc))
            pci = F.cluster_bootstrap(sorted(pbc), lambda d, pbc=pbc: F.icc_2_1(_mat(pbc, d)),
                                      seed=seed_of(f"icc21{pair}{label}"))
            row[f"icc21_{pair[0]}{pair[1]}"] = F.icc_2_1(pm)
            row[f"icc21_{pair[0]}{pair[1]}_lo"], row[f"icc21_{pair[0]}{pair[1]}_hi"] = pci[0], pci[1]
            row[f"videos_{pair[0]}{pair[1]}"] = len(pm)
        row["scorer_agreement_poor"] = bool(math.isfinite(row["icc11_hi"]) and row["icc11_hi"] < F.READABLE_LINE)
        rows.append(row)
    return rows


def desc_table(rows):
    """{(zip, region, descriptor): {video: row}}."""
    t = {}
    for r in rows:
        t.setdefault((r["zip"], r["region"], r["descriptor"]), {})[int(r["video"])] = r
    return t


def within_clip_rows(dt, child_of_video):
    out = []
    for (z, region, n), byv in sorted(dt.items()):
        pairs = [(child_of_video[v], (_f(r["half1"]), _f(r["half2"]))) for v, r in sorted(byv.items())
                 if math.isfinite(_f(r["half1"])) and math.isfinite(_f(r["half2"]))]
        byc = _by_child(pairs)
        ch = sorted(byc)
        m = _mat(byc, ch)
        lab = f"{z}/{region}/{n}"
        icc = F.icc_2_1(m)
        s = F.sem(m, icc)
        ci = F.cluster_bootstrap(ch, lambda d, byc=byc: F.icc_2_1(_mat(byc, d)), seed=seed_of("wc" + lab))
        bci = F.cluster_bootstrap(ch, lambda d, byc=byc: F.bound(F.sem(_mat(byc, d), F.icc_2_1(_mat(byc, d)))),
                                  seed=seed_of("wcb" + lab))
        out.append({"zip": z, "region": region, "descriptor": n, "unit": F.UNITS[n], "records": len(m),
                    "children": len(ch), "icc21": icc, "icc21_lo": ci[0], "icc21_hi": ci[1], "sem": s,
                    "bound": F.bound(s), "bound_lo": bci[0], "bound_hi": bci[1], "window_s": 10.0,
                    "not_readable": bool(math.isfinite(ci[1]) and ci[1] < F.READABLE_LINE)})
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def stage_outputs(args):
    root = root_of(args)
    rules = check_rules()
    require_fixtures()
    posture = load_posture()
    settled = posture["settled"]
    ident = identity(root)
    with open(os.path.join(OUT_S1, "record_descriptors.csv"), encoding="utf-8", newline="") as f:
        drows = list(csv.DictReader(f))
    cidx = {i: c for i, c in enumerate(ident["children"])}
    child_of_video = {int(r["video"]): cidx[int(r["child_index"])] for r in drows}
    LockboxGuard(ident["lock"]).check(set(child_of_video.values()), "descriptor table")
    dt = desc_table(drows)
    lab = load_labels(root, ident)
    rv = per_video_ratings(lab)
    hdr, ma = header_lines(root, rules)
    name = (lambda z: f"{z} ({posture['posture'][z]})") if settled else (lambda z: z)

    # output 1: scorer agreement (labels only)
    agree_region = agreement_rows(rv, ITEM_KEYS + [(c, "subscale", "all") for c in COMPONENTS.values()], raw=False)
    agree_raw = agreement_rows(rv, list(ITEM_COLUMNS), raw=True)
    a_fields = ["item", "videos", "children", "icc11", "icc11_lo", "icc11_hi", "icc21_12", "icc21_12_lo",
                "icc21_12_hi", "videos_12", "icc21_13", "icc21_13_lo", "icc21_13_hi", "videos_13", "qwk", "sem",
                "bound", "bound_lo", "bound_hi", "scorer_agreement_poor"]
    write_csv(os.path.join(OUT_S1, "output1_scorer_agreement.csv"), agree_region + agree_raw, a_fields)

    # output 2: within-clip reliability
    wc = within_clip_rows(dt, child_of_video)
    w_fields = ["zip", "region", "descriptor", "unit", "records", "children", "icc21", "icc21_lo", "icc21_hi",
                "sem", "bound", "bound_lo", "bound_hi", "window_s", "not_readable"]
    write_csv(os.path.join(OUT_S1, "output2_within_clip.csv"), wc, w_fields)
    wci = {(r["zip"], r["region"], r["descriptor"]): r for r in wc}
    agi = {r["item"]: r for r in agree_region}

    # per child and visit: item means and descriptor values
    vid_of = {}
    for v, rs in rv.items():
        c = next(iter(rs.values()))["child"]
        t = next(iter(rs.values()))["time"]
        assert (c, t) not in vid_of, "a child carries two videos at one visit"
        vid_of[(c, t)] = v

    def item_at(c, t, key):
        v = vid_of.get((c, t))
        return item_mean(rv[v], key) if v is not None else math.nan

    def desc_at(c, t, z, region, n):
        v = vid_of.get((c, t))
        r = dt.get((z, region, n), {}).get(v) if v is not None else None
        return _f(r["record"]) if r else math.nan

    children = ident["children"]
    zip_of_posture = {p: z for z, p in (posture["posture"] or {}).items()}

    # output 4: standardized response mean
    srm_rows = []
    for t1 in ("T3", "T12"):
        for key in ITEM_KEYS:
            ch = [c for c in children if math.isfinite(item_at(c, "T0", key)) and math.isfinite(item_at(c, t1, key))]
            chg = {c: item_at(c, t1, key) - item_at(c, "T0", key) for c in ch}
            b = agi["/".join(key)]["bound"]
            ci = F.cluster_bootstrap(ch, lambda d, chg=chg: F.srm([chg[c] for c in d]),
                                     seed=seed_of(f"srm{t1}{key}"))
            srm_rows.append({"interval": f"T0 to {t1}", "kind": "DIS item", "zip": "", "region": key[2],
                             "quantity": f"{key[0]} {key[1]}", "children": len(ch),
                             "mean_change": float(np.mean(list(chg.values()))) if ch else math.nan,
                             "srm": F.srm(list(chg.values())), "srm_lo": ci[0], "srm_hi": ci[1], "bound": b,
                             "children_beyond_bound": sum(abs(x) > b for x in chg.values()) if math.isfinite(b)
                             else "", "label": "change of unknown cause"})
        for (z, region, n), byv in sorted(dt.items()):
            ch = [c for c in children if math.isfinite(desc_at(c, "T0", z, region, n))
                  and math.isfinite(desc_at(c, t1, z, region, n))]
            chg = {c: desc_at(c, t1, z, region, n) - desc_at(c, "T0", z, region, n) for c in ch}
            b = wci[(z, region, n)]["bound"]
            ci = F.cluster_bootstrap(ch, lambda d, chg=chg: F.srm([chg[c] for c in d]),
                                     seed=seed_of(f"srm{t1}{z}{region}{n}"))
            srm_rows.append({"interval": f"T0 to {t1}", "kind": "descriptor", "zip": name(z), "region": region,
                             "quantity": n, "children": len(ch),
                             "mean_change": float(np.mean(list(chg.values()))) if ch else math.nan,
                             "srm": F.srm(list(chg.values())), "srm_lo": ci[0], "srm_hi": ci[1], "bound": b,
                             "children_beyond_bound": sum(abs(x) > b for x in chg.values()) if math.isfinite(b)
                             else "", "label": "change of unknown cause"})
    s_fields = ["interval", "kind", "zip", "region", "quantity", "children", "mean_change", "srm", "srm_lo", "srm_hi",
                "bound", "children_beyond_bound", "label"]
    write_csv(os.path.join(OUT_S1, "output4_response.csv"), srm_rows, s_fields)

    # output 5: co-variation of change (descriptive)
    cov_rows = []

    def add_cov(t1, kind, a_lab, b_lab, fa, fb):
        ch = [c for c in children if math.isfinite(fa(c)) and math.isfinite(fb(c))]
        da, db_ = {c: fa(c) for c in ch}, {c: fb(c) for c in ch}
        ci = F.cluster_bootstrap(ch, lambda d: F.spearman([da[c] for c in d], [db_[c] for c in d]),
                                 seed=seed_of(f"cov{t1}{a_lab}{b_lab}"))
        cov_rows.append({"interval": f"T0 to {t1}", "kind": kind, "a": a_lab, "b": b_lab, "children": len(ch),
                         "spearman": F.spearman(list(da.values()), list(db_.values())), "lo": ci[0], "hi": ci[1],
                         "label": "change of unknown cause"})

    def ichg(key, t1):
        return lambda c: item_at(c, t1, key) - item_at(c, "T0", key)

    for t1 in ("T3", "T12"):
        for facet in FACETS.values():
            for region in F.REGIONS:
                add_cov(t1, "DIS, dystonia against choreoathetosis", f"dystonia {facet} {region}",
                        f"choreoathetosis {facet} {region}", ichg(("dystonia", facet, region), t1),
                        ichg(("choreoathetosis", facet, region), t1))
        for comp in COMPONENTS.values():
            for facet in FACETS.values():
                for i, ra in enumerate(F.REGIONS):
                    for rb in F.REGIONS[i + 1:]:
                        add_cov(t1, "DIS, across regions", f"{comp} {facet} {ra}", f"{comp} {facet} {rb}",
                                ichg((comp, facet, ra), t1), ichg((comp, facet, rb), t1))
        # descriptors: in the scored posture of each region when the posture is settled, else within each zip by
        # its name (no descriptor is set against a DIS item here)
        configs = ([("scored posture", {r: zip_of_posture[SCORED_POSTURE[r]] for r in F.REGIONS})] if settled
                   else [(z, {r: z for r in F.REGIONS}) for z in ZIPS])
        for lab_z, zsel in configs:
            def dchg(region, n, t1=t1, zsel=zsel):
                z = zsel[region]
                return lambda c: desc_at(c, t1, z, region, n) - desc_at(c, "T0", z, region, n)
            for facet in FACETS.values():
                dn, cn = FACET_MAP[("dystonia", facet)], FACET_MAP[("choreoathetosis", facet)]
                for region in ("proximal_arm", "proximal_leg"):
                    add_cov(t1, f"descriptors ({lab_z}), dystonia axis against choreoathetosis axis",
                            f"{dn} {region}", f"{cn} {region}", dchg(region, dn), dchg(region, cn))
            for n in ("TON-T", "TON-DW", "FLOW-OCC", "FLOW-R"):
                regs = [r for r in F.REGIONS if n in F.DESCRIPTORS[r]]
                for i, ra in enumerate(regs):
                    for rb in regs[i + 1:]:
                        add_cov(t1, f"descriptors ({lab_z}), across regions", f"{n} {ra}", f"{n} {rb}",
                                dchg(ra, n), dchg(rb, n))
    write_csv(os.path.join(OUT_S1, "output5_covariation.csv"), cov_rows,
              ["interval", "kind", "a", "b", "children", "spearman", "lo", "hi", "label"])

    # output 6: bandwidth and view audit
    fps, evidence = source_fps()
    audit = []
    for region in F.REGIONS:
        for n in F.DESCRIPTORS[region]:
            band = n in ("PHA-P1", "FLOW-E", "FLOW-DII", "FLOW-OCC")
            for z in ZIPS:
                byv = dt.get((z, region, n), {})
                vals = [r for r in byv.values()]
                n_rec = len(vals)
                n_val = sum(math.isfinite(_f(r["record"])) for r in vals)
                audit.append({"descriptor": n, "region": region, "zip": name(z), "fs_hz": fps,
                              "f_max_hz": F.f_max_hz(fps), "band_hz": "0.5 to 3.0" if band else "0 to 0.3 (tonic)"
                              if n.startswith("TON") or n == "TRUNK-ANGLE-RAW" else "0 to 3.0 (low-pass)",
                              "admitted": "yes", "resolution_s": 1 / fps, "d_min_s": F.D_MIN_SAMPLES / fps,
                              "records": n_rec, "records_with_value": n_val,
                              "measured_axis": "in-plane angle only; out-of-plane and axial rotation not measured "
                              "(one camera, DR-14.6, DR-7.7)" if n.startswith("TON") or n in ("PHA-P1",
                                                                                            "TRUNK-ANGLE-RAW")
                              else "image-plane components only (DR-13.5)",
                              "flags": "image_plane" + (";laterality_unknown" if region != "trunk" else "")})
    for z in ZIPS:
        audit.append({"descriptor": "PHA-P2", "region": "all", "zip": name(z), "fs_hz": fps,
                      "f_max_hz": F.f_max_hz(fps), "band_hz": "3 to 12", "admitted":
                      "no: refused, 12 Hz > f_max (DR-5.3, DR-14.13); not computed (S1 draft 3.2)",
                      "resolution_s": 1 / fps, "d_min_s": F.D_MIN_SAMPLES / fps, "records": "", "records_with_value":
                      "", "measured_axis": "", "flags": ""})
    usable = {}
    for r in drows:
        if r["region"] == "trunk" and r["descriptor"] == "FLOW-R":
            usable.setdefault(r["zip"], []).append((_f(r["usable_fraction"]), _f(r["window_usable_median"])))
    write_csv(os.path.join(OUT_S1, "output6_audit.csv"), audit,
              ["descriptor", "region", "zip", "fs_hz", "f_max_hz", "band_hz", "admitted", "resolution_s", "d_min_s",
               "records", "records_with_value", "measured_axis", "flags"])

    # the readability card and the primary outcome
    card = []
    for comp in COMPONENTS.values():
        for facet in FACETS.values():
            for region in F.REGIONS:
                key = (comp, facet, region)
                n = FACET_MAP[(comp, facet)]
                a = agi["/".join(key)]
                row = {"region": region, "component": comp, "facet": facet, "descriptor": n,
                       "scorer_icc11": a["icc11"], "scorer_icc11_lo": a["icc11_lo"], "scorer_icc11_hi": a["icc11_hi"],
                       "scorer_bound": a["bound"], "scorer_bound_lo": a["bound_lo"], "scorer_bound_hi": a["bound_hi"],
                       "scorer_agreement": "scorer agreement poor" if a["scorer_agreement_poor"] else "",
                       "item_srm_T0_T12": next(x["srm"] for x in srm_rows if x["interval"] == "T0 to T12"
                                               and x["kind"] == "DIS item" and x["region"] == region
                                               and x["quantity"] == f"{comp} {facet}")}
                if n not in F.DESCRIPTORS[region]:
                    row.update({"zip": "", "within_icc21": math.nan, "within_icc21_lo": math.nan,
                                "within_icc21_hi": math.nan, "within_bound": math.nan, "within_bound_lo": math.nan,
                                "within_bound_hi": math.nan, "readability": "no descriptor (trunk: raw angle only, "
                                "no deviation and no dwell threshold, S1 section 3)", "descriptor_srm_T0_T12": math.nan})
                elif settled:
                    z = zip_of_posture[SCORED_POSTURE[region]]
                    w = wci[(z, region, n)]
                    row.update({"zip": name(z), "within_icc21": w["icc21"], "within_icc21_lo": w["icc21_lo"],
                                "within_icc21_hi": w["icc21_hi"], "within_bound": w["bound"],
                                "within_bound_lo": w["bound_lo"], "within_bound_hi": w["bound_hi"],
                                "readability": "not readable from this clinic video" if w["not_readable"] else
                                "readable (within-clip interval not wholly below 0.5)",
                                "descriptor_srm_T0_T12": next(x["srm"] for x in srm_rows if x["interval"] ==
                                                              "T0 to T12" and x["kind"] == "descriptor" and
                                                              x["zip"] == name(z) and x["region"] == region and
                                                              x["quantity"] == n)})
                else:
                    row.update({"zip": "withdrawn (posture open)", "readability": "withdrawn (posture open)"})
                card.append(row)
    c_fields = ["region", "component", "facet", "descriptor", "zip", "scorer_icc11", "scorer_icc11_lo",
                "scorer_icc11_hi", "scorer_agreement", "scorer_bound", "scorer_bound_lo", "scorer_bound_hi",
                "within_icc21", "within_icc21_lo", "within_icc21_hi", "readability", "within_bound",
                "within_bound_lo", "within_bound_hi", "item_srm_T0_T12", "descriptor_srm_T0_T12"]
    write_csv(os.path.join(OUT_S1, "readability_card.csv"), card, c_fields)
    summary = {"posture_settled": settled, "posture": posture["posture"], "moveatlas_head": ma["head"],
               "rules": rules, "children": len(children),
               "poor_items": [r["item"] for r in agree_region if r["scorer_agreement_poor"]],
               "not_readable": [f"{r['zip']}/{r['region']}/{r['descriptor']}" for r in wc if r["not_readable"]],
               "card": card}
    with open(os.path.join(OUT_S1, "s1_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=lambda x: None if isinstance(x, float) else str(x))
        f.write("\n")
    write_report(hdr, posture, agree_region, agree_raw, wc, srm_rows, cov_rows, audit, usable, card, children,
                 rv, fps, evidence, name)


def _ci(lo, hi):
    return f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"


def write_report(hdr, posture, agree_region, agree_raw, wc, srm_rows, cov_rows, audit, usable, card, children, rv,
                 fps, evidence, name):
    settled = posture["settled"]
    n_dbl = sum(1 for rs in rv.values() if len(rs) == 2)
    md = ["# Study S1: outputs 1 to 6 and the readability card", "",
          "Estimation study, no Holm slot, no kill on an outcome (S1 draft section 3). Every interval is a "
          f"child-clustered percentile bootstrap, {F.BOOT_B} replicates, 95 percent. Image plane; every arm and leg "
          "value is side-pooled and carries laterality_unknown. Both noise bounds are a lower bound, not MDC; every "
          "change between visits is a change of unknown cause. The trunk angle is raw, never a deviation. The flow "
          "channel reads irregular flow, not choreoathetosis; the tonic channel reads sustained posture, dystonic or "
          "spastic not separable.", ""] + hdr + [
          f"- training children read: {len(children)}; training videos with a sheet row: {len(rv)}, "
          f"double-scored: {n_dbl}; lockbox children: 7, excluded before any read",
          f"- posture: {'settled, ' + ', '.join(f'{z} {p}' for z, p in posture['posture'].items()) if settled else 'open'}"
          " (reports/modys_structure/posture_v2.md)", f"- rate: {fps} frames per second ({evidence})", ""]
    md += ["## Primary outcome: the readability card", ""]
    if not settled:
        md += ["The posture of the two zips is open (S1 section 2): the descriptor half of every row, which pairs a "
               "descriptor with a DIS item in its scored posture, is withdrawn (S1 draft section 7). The scorer half "
               "is printed; the within-clip values of every descriptor are in output 2, by zip name.", ""]
    md += [
           "| region | component | facet | descriptor (zip) | scorer ICC(1,1) | scorer bound, DIS points, lower bound, "
           "not MDC | within-clip ICC(2,1) | within-clip bound, lower bound, not MDC | SRM T0 to T12, item | SRM T0 "
           "to T12, descriptor | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in card:
        md.append(f"| {r['region']} | {r['component']} | {r['facet']} | {r['descriptor']} ({r.get('zip', '')}) | "
                  f"{fmt(r['scorer_icc11'], 3)} {_ci(r['scorer_icc11_lo'], r['scorer_icc11_hi'])} | "
                  f"{fmt(r['scorer_bound'], 3)} {_ci(r['scorer_bound_lo'], r['scorer_bound_hi'])} | "
                  f"{fmt(r.get('within_icc21', math.nan), 3)} {_ci(r.get('within_icc21_lo', math.nan), r.get('within_icc21_hi', math.nan))} | "
                  f"{fmt(r.get('within_bound', math.nan), 3)} {_ci(r.get('within_bound_lo', math.nan), r.get('within_bound_hi', math.nan))} "
                  f"{F.UNITS.get(r['descriptor'], '')} | {fmt(r['item_srm_T0_T12'], 3)} | "
                  f"{fmt(r.get('descriptor_srm_T0_T12', math.nan), 3)} | "
                  f"{'; '.join(x for x in (r['scorer_agreement'], r.get('readability', '')) if x)} |")
    md += ["", "## Output 1: scorer agreement (double-scored training videos, T0 and T12)", "",
           "| item | videos (children) | ICC(1,1) | ICC(2,1) pair 1+2 (videos) | ICC(2,1) pair 1+3 (videos) | "
           "quadratic weighted kappa | scorer SEM | scorer bound, lower bound, not MDC |",
           "|---|---|---|---|---|---|---|---|"]
    for r in agree_region + agree_raw:
        md.append(f"| {r['item']} | {r['videos']} ({r['children']}) | {fmt(r['icc11'], 3)} "
                  f"{_ci(r['icc11_lo'], r['icc11_hi'])} | {fmt(r['icc21_12'], 3)} "
                  f"{_ci(r['icc21_12_lo'], r['icc21_12_hi'])} ({r['videos_12']}) | {fmt(r['icc21_13'], 3)} "
                  f"{_ci(r['icc21_13_lo'], r['icc21_13_hi'])} ({r['videos_13']}) | {fmt(r['qwk'], 3)} | "
                  f"{fmt(r['sem'], 3)} | {fmt(r['bound'], 3)} {_ci(r['bound_lo'], r['bound_hi'])}"
                  f"{' (scorer agreement poor)' if r['scorer_agreement_poor'] else ''} |")
    md += ["", "Region items of arm and leg are the mean of the right and left items; a subscale is the sum of the "
           "10 rest items of its component. An item whose ICC(1,1) interval lies wholly below 0.5 reads "
           "\"scorer agreement poor\".", "",
           "## Output 2: within-clip reliability (first and last floor(n/2) frames of 19.96 s and 20.0 s records, "
           "each half one 10 s window)", "",
           "| zip | region | descriptor | records (children) | ICC(2,1) | within-clip SEM | within-clip bound, lower "
           "bound, not MDC | reading |", "|---|---|---|---|---|---|---|---|"]
    for r in wc:
        md.append(f"| {name(r['zip'])} | {r['region']} | {r['descriptor']} | {r['records']} ({r['children']}) | "
                  f"{fmt(r['icc21'], 3)} {_ci(r['icc21_lo'], r['icc21_hi'])} | {fmt(r['sem'], 3)} {r['unit']} | "
                  f"{fmt(r['bound'], 3)} {_ci(r['bound_lo'], r['bound_hi'])} | "
                  f"{'not readable from this clinic video' if r['not_readable'] else ''} |")
    md += ["", "## Output 3: the two noise bounds", "",
           "The scorer bound (DIS points) is in the output 1 table, the within-clip bound (descriptor units) in the "
           "output 2 table, both 1.96 x sqrt(2) x SEM, each a lower bound, not MDC: no between-day repeat is held.",
           "", "## Output 4: standardized response mean, change of unknown cause", "",
           "| interval | kind | zip | region | quantity | children | mean change | SRM | bound | children beyond "
           "the bound |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in srm_rows:
        md.append(f"| {r['interval']} | {r['kind']} | {r['zip']} | {r['region']} | {r['quantity']} | "
                  f"{r['children']} | {fmt(r['mean_change'], 3)} | {fmt(r['srm'], 3)} {_ci(r['srm_lo'], r['srm_hi'])}"
                  f" | {fmt(r['bound'], 3)} | {r['children_beyond_bound']} |")
    md += ["", "Exceeding a lower bound is necessary, not sufficient, for a detectable change. T3 carries no "
           "double-scored video: the T0 to T3 change of an item is set against bounds estimated at T0 and T12.", "",
           "## Output 5: co-variation of within-child change (Spearman, descriptive)", "",
           "| interval | kind | a | b | children | Spearman |", "|---|---|---|---|---|---|"]
    for r in cov_rows:
        md.append(f"| {r['interval']} | {r['kind']} | {r['a']} | {r['b']} | {r['children']} | "
                  f"{fmt(r['spearman'], 3)} {_ci(r['lo'], r['hi'])} |")
    if not settled:
        md.append("\nThe posture is open: descriptor rows are within each zip, labelled by its name; no descriptor "
                  "is set against a DIS item.")
    md += ["", "## Output 6: bandwidth and view audit", "",
           f"k = {F.K_BANDWIDTH} (OPEN-D02), f_max = k x fs = {fmt(F.f_max_hz(fps), 2)} Hz; every S1 band ends at "
           "3 Hz and is admitted; PHA-P2 (3 to 12 Hz) is refused at this rate for every admissible k. Durations: "
           f"resolution {fmt(1 / fps, 3)} s, no run shorter than d_min = 1.8 / fs = {fmt(F.D_MIN_SAMPLES / fps, 3)} s "
           "counted.", "",
           "| descriptor | region | zip | band | admitted | records | records with a value | measured |",
           "|---|---|---|---|---|---|---|---|"]
    for r in audit:
        md.append(f"| {r['descriptor']} | {r['region']} | {r['zip']} | {r['band_hz']} | {r['admitted']} | "
                  f"{r['records']} | {r['records_with_value']} | {r['measured_axis']} |")
    md += ["", "Usable fraction (OPEN-D56: frames with hips, shoulders, elbows and knees all at likelihood 0.5 or "
           "above), per zip, quantiles 5/25/50/75/95 over records of the record fraction and of the median window "
           "fraction:", ""]
    for z, v in sorted(usable.items()):
        a = np.array(v, dtype=float)
        q = lambda x: " / ".join(fmt(float(np.nanpercentile(x, p)), 3) for p in (5, 25, 50, 75, 95))
        md.append(f"- {name(z)}: record {q(a[:, 0])}; window median {q(a[:, 1])} ({len(a)} records)")
    md.append("")
    write_text(os.path.join(OUT_S1, "s1_outputs.md"), "\n".join(md))
    print("\n".join(md[:80]))


# ---------------------------------------------------------------- main
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "movie", "posture", "fixtures", "precision", "descriptors",
                                      "outputs"])
    ap.add_argument("--moveatlas", default=None)
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args(argv)
    {"probe": stage_probe, "movie": stage_movie, "posture": stage_posture, "fixtures": stage_fixtures,
     "precision": stage_precision, "descriptors": stage_descriptors, "outputs": stage_outputs}[args.stage](args)


if __name__ == "__main__":
    main()
