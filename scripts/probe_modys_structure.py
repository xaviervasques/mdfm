"""modys_video: structural probe of the two coordinate zips on the 27 training children (T10).

Settles, under a decision rule hashed before the data read (reports/modys_structure/rule.txt and
rule.txt.sha256), the structural facts that the dyskinetic CP studies need before any rule file
is hashed: which zip holds the lying and which the sitting protocol (trunk-axis orientation),
the facing direction per record, whether keypoint laterality is established, whether both zips
share one DeepLabCut keypoint set and model tag, the scorer-by-video structure of the clinical
sheet, the visits held per child, the clip spans and the likelihood coverage.

Guards (asserted in the code):
- data_clinical_scoring.xlsx is read with usecols ID, video, time and scorer only: no DIS item
  column is read, and no statistic that involves a DIS item is computed.
- The 7 lockbox children come from the lockbox column of cache/manifest/manifest_records.csv.gz
  (read with the gzip module). Their video numbers, and every video number that does not belong
  to a training child, are dropped from the zip member lists before any coordinate CSV inside a
  zip is opened; the parser refuses any other video.
- The script stops when rule.txt does not match the hash it holds.

Reads (read-only): MoveAtlas datasets/pose/modys_video/raw (README.txt, data_clinical_scoring.xlsx,
data_stickfigure_coordinates_lying.zip, data_stickfigure_coordinates_sitting.zip), this tree's
cache/manifest/manifest_records.csv.gz, reports/lockbox/lockbox_persons.csv and
reports/nuisance/frame_rates_facts.json.
Writes: reports/modys_structure/ (summary.md, per_record.csv, scorer_by_video.csv, facts.json,
run.log, _provenance.json).

Run: CUDA_VISIBLE_DEVICES=-1 python scripts/probe_modys_structure.py --moveatlas <moveatlas>
CPU only, small I/O, BelowNormal priority; pauses while cache/runs/queue/SLOW exists.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RULE_SHA256 = "0b56780354f3e8af5cdc8b57188698a9e455b8d58acd4bd738302c8221fadc7b"
RULE_V1_BYTES = 8269   # version 1 is bytes 0 to 8268 of rule.txt (version 2 appended on 2026-11-12)
USECOLS = ["ID", "video", "time", "scorer"]
LIKE = 0.5
ZIPS = ["data_stickfigure_coordinates_lying.zip", "data_stickfigure_coordinates_sitting.zip"]
SIDE_PAIRS = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
GEOMETRY_PARTS = ["forehead", "chin", "shoulder1", "shoulder2", "hip1", "hip2", "knee1", "knee2"]
TIMES = ["T0", "T3", "T12"]

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "reports", "modys_structure")
SLOW = os.path.join(HERE, "cache", "runs", "queue", "SLOW")

assert USECOLS == ["ID", "video", "time", "scorer"], "the scoring sheet may only be read on these four columns"


# ---------------------------------------------------------------- process discipline

def below_normal():
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


def wait_if_slow():
    announced = False
    while os.path.exists(SLOW):
        if not announced:
            print("cache/runs/queue/SLOW exists: paused", flush=True)
            announced = True
        time.sleep(30)
    if announced:
        print("SLOW removed: resumed", flush=True)


class Log:
    """Deterministic run log (no clock): written to run.log and echoed to stdout."""

    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)
        print(msg, flush=True)


# ---------------------------------------------------------------- small helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rnd(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if math.isnan(x):
        return None
    v = round(x, nd)
    return 0.0 if v == 0 else v


def img_dir(vx, vy):
    """Image direction of a vector, x wins a tie (image y points down)."""
    if vx is None or vy is None or (vx == 0 and vy == 0) or math.isnan(vx) or math.isnan(vy):
        return "undefined"
    if abs(vx) >= abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"


def side_tokens(name):
    toks = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name)
    return [t.lower() for t in toks if t.lower() in ("left", "right", "l", "r")]


def video_of(member):
    return int(os.path.basename(member).split("_")[0])


def counts(values):
    out = {}
    for v in values:
        k = str(v)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- DLC parser (guarded)

def parse_dlc(zf, member, allowed, lockbox):
    video = video_of(member)
    assert video in allowed and video not in lockbox, f"refused: video {video:03d} is not a training video"
    raw = zf.read(member)  # the first byte of a member is read only after the guard above
    rows = [r for r in csv.reader(io.StringIO(raw.decode("utf-8"))) if r]
    assert rows[0][0] == "scorer" and rows[1][0] == "bodyparts" and rows[2][0] == "coords", "not a DeepLabCut CSV"
    scorers = sorted(set(rows[0][1:]))
    parts, coords = rows[1][1:], rows[2][1:]
    bodyparts = list(dict.fromkeys(parts))
    for i, bp in enumerate(bodyparts):
        assert parts[3 * i:3 * i + 3] == [bp] * 3, f"bodypart columns of {bp} are not a triplet"
        assert coords[3 * i:3 * i + 3] == ["x", "y", "likelihood"], f"coords of {bp} are not x, y, likelihood"
    assert len(parts) == 3 * len(bodyparts)
    data = np.array([[float(v) if v != "" else np.nan for v in r[1:]] for r in rows[3:]], dtype=float)
    assert data.shape[1] == 3 * len(bodyparts)
    return {"scorers": scorers, "bodyparts": bodyparts, "x": data[:, 0::3], "y": data[:, 1::3],
            "like": data[:, 2::3]}


# ---------------------------------------------------------------- rule quantities per record

def group_mean(x, y, valid, idx):
    v = valid[:, idx]
    n = v.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.where(v, x[:, idx], 0.0).sum(1) / n
        my = np.where(v, y[:, idx], 0.0).sum(1) / n
    ok = n > 0
    mx[~ok] = np.nan
    my[~ok] = np.nan
    return mx, my, ok


def record_geometry(rec):
    bp = rec["bodyparts"]
    x, y, like = rec["x"], rec["y"], rec["like"]
    n_frames = x.shape[0]
    valid = like >= LIKE
    out = {"rows": n_frames,
           "like_frac": float(valid.sum() / valid.size) if valid.size else float("nan")}
    one = [bp.index(f"{p}1") for p in SIDE_PAIRS if f"{p}1" in bp and f"{p}2" in bp]
    two = [bp.index(f"{p}2") for p in SIDE_PAIRS if f"{p}1" in bp and f"{p}2" in bp]
    out["d_like"] = (float(np.mean(like[:, one].mean(1) - like[:, two].mean(1)))
                     if len(one) == len(SIDE_PAIRS) and n_frames else float("nan"))
    missing = [p for p in GEOMETRY_PARTS if p not in bp]
    out["missing_parts"] = ";".join(missing)
    if missing or n_frames == 0:
        out.update(sufficient=False, cls="insufficient", facing="not settled", evidence="missing body parts",
                   view="", head_direction="", camera_side="", front_sign=None, tv_frac=0.0, e=np.array([]))
        return out
    I = {p: bp.index(p) for p in GEOMETRY_PARTS}
    sx, sy, s_ok = group_mean(x, y, valid, [I["shoulder1"], I["shoulder2"]])
    hx, hy, h_ok = group_mean(x, y, valid, [I["hip1"], I["hip2"]])
    kx, ky, k_ok = group_mean(x, y, valid, [I["knee1"], I["knee2"]])
    px, py, p_ok = group_mean(x, y, valid, [I["forehead"], I["chin"]])
    with np.errstate(invalid="ignore", divide="ignore"):
        tx, ty = sx - hx, sy - hy
        L = np.hypot(tx, ty)
        tv = s_ok & h_ok & (L >= 1.0)
        ux, uy = tx / L, ty / L
        nx, ny = -uy, ux
        e = np.degrees(np.arctan2(hy - sy, np.abs(sx - hx)))
        thx, thy = kx - hx, ky - hy
        r = np.hypot(thx, thy) / L
        ax, ay = hx - sx, hy - sy
        denom = np.hypot(ax, ay) * np.hypot(thx, thy)
        cosc = np.clip((ax * thx + ay * thy) / denom, -1.0, 1.0)
        c = np.where(denom > 0, np.degrees(np.arccos(cosc)), np.nan)
        f = ((px - sx) * nx + (py - sy) * ny) / L
        k = (thx * nx + thy * ny) / L
        i1, i2 = I["shoulder1"], I["shoulder2"]
        both_sh = valid[:, i1] & valid[:, i2]
        w = np.hypot(x[:, i1] - x[:, i2], y[:, i1] - y[:, i2]) / L
    n_tv = int(tv.sum())
    kv = tv & k_ok
    pv = tv & p_ok
    bv = tv & both_sh
    out["tv_frac"] = n_tv / n_frames
    out["sufficient"] = n_tv >= 0.5 * n_frames
    out["e"] = e[tv]
    out["leg_frac"] = kv.sum() / n_tv if n_tv else 0.0
    out["face_frac"] = pv.sum() / n_tv if n_tv else 0.0
    out["both_sh_frac"] = bv.sum() / n_tv if n_tv else 0.0
    if not out["sufficient"]:
        out.update(cls="insufficient", facing="not settled", evidence="trunk-valid frames below 50 percent",
                   view="", head_direction="", camera_side="", front_sign=None)
        return out
    me = float(np.median(e[tv]))
    out["me"] = me
    out["e_q25"], out["e_q75"] = (float(q) for q in np.quantile(e[tv], [0.25, 0.75]))
    leg_avail = kv.sum() >= 0.25 * n_tv
    out["leg_avail"] = bool(leg_avail)
    mc = float(np.nanmedian(c[kv])) if leg_avail and np.isfinite(c[kv]).any() else float("nan")
    mr = float(np.median(r[kv])) if leg_avail else float("nan")
    out["mc"], out["mr"] = mc, mr
    top = bool(leg_avail and mc < 45 and mr >= 0.5)
    out["top_view"] = top
    if abs(me) <= 30:
        cls = "flat"
    elif abs(me) >= 60 and top:
        cls = "vertical, legs in line"
    elif me >= 60 and leg_avail and not top:
        cls = "upright"
    else:
        cls = "other"
    out["cls"] = cls
    # facing
    umx, umy = float(np.median(ux[tv])), float(np.median(uy[tv]))
    nrm = math.hypot(umx, umy)
    umx, umy = (umx / nrm, umy / nrm) if nrm > 0 else (float("nan"), float("nan"))
    out["head_direction"] = img_dir(umx, umy)
    out["mw"] = float(np.median(w[bv])) if bv.any() else float("nan")
    out["mf"] = float(np.median(f[pv])) if pv.any() else float("nan")
    out["mk"] = float(np.median(k[kv])) if kv.any() else float("nan")
    out["front_sign"] = None
    out["camera_side"] = ""
    if bv.sum() >= 0.25 * n_tv and out["mw"] >= 0.5:
        out.update(view="frontal", facing="frontal", evidence="shoulder width")
        return out
    out["view"] = "side"
    face = int(np.sign(out["mf"])) if pv.sum() >= 0.25 * n_tv and abs(out["mf"]) >= 0.10 else 0
    knee = int(np.sign(out["mk"])) if kv.sum() >= 0.25 * n_tv and abs(out["mk"]) >= 0.20 else 0
    if face and knee and face == knee:
        s, ev = face, "face and knee"
    elif face and knee:
        s, ev = 0, "conflict"
    elif face:
        s, ev = face, "face"
    elif knee:
        s, ev = knee, "knee"
    else:
        s, ev = 0, "none"
    out["evidence"] = ev
    if s == 0 or nrm == 0:
        out["facing"] = "not settled"
        return out
    out["front_sign"] = s
    out["facing"] = img_dir(s * -umy, s * umx)
    out["camera_side"] = "right" if s > 0 else "left"
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    args = ap.parse_args()
    root = args.moveatlas.replace("\\", "/").rstrip("/")
    raw_dir = os.path.join(root, "datasets", "pose", "modys_video", "raw")
    sys.stdout.reconfigure(encoding="utf-8")
    below_normal()
    log = Log()

    # 1. the rule precedes every data read
    # rule.txt holds version 1 in its first RULE_V1_BYTES bytes; version 2 (2026-11-12, S1 section 2) follows it
    # and governs scripts/study_s1.py only. This script reads version 1 and checks its hash.
    rule_path = os.path.join(OUT, "rule.txt")
    with open(rule_path, "rb") as f:
        rule_hash = hashlib.sha256(f.read()[:RULE_V1_BYTES]).hexdigest()
    with open(os.path.join(OUT, "rule.txt.sha256"), encoding="utf-8") as f:
        stored = [ln.split()[0] for ln in f if ln.strip()]
    assert rule_hash == RULE_SHA256 and RULE_SHA256 in stored, "rule.txt differs from the hash held by the script"
    log(f"rule.txt sha256 {rule_hash}: matches the script and rule.txt.sha256")
    wait_if_slow()

    # 2. lockbox and training persons from the manifest lockbox column
    man_path = os.path.join(HERE, "cache", "manifest", "manifest_records.csv.gz")
    with gzip.open(man_path, "rt", encoding="utf-8", newline="") as f:
        man = [r for r in csv.DictReader(f) if r["source"] == "modys_video"]
    assert man and all(r["lockbox"] in ("True", "False") for r in man)
    for r in man:
        assert r["subject_ns"] == "modys:" + r["key"][:3]
    lock_videos_manifest = sorted({int(r["key"][:3]) for r in man if r["lockbox"] == "True"})
    lock_persons = sorted({r["person_id"] for r in man if r["lockbox"] == "True"})
    train_persons = sorted({r["person_id"] for r in man if r["lockbox"] == "False"})
    assert not set(lock_persons) & set(train_persons), "a child carries both lockbox flags"
    assert len(lock_persons) == 7 and len(train_persons) == 27, (len(lock_persons), len(train_persons))
    with open(os.path.join(HERE, "reports", "lockbox", "lockbox_persons.csv"), encoding="utf-8", newline="") as f:
        lb_report = sorted(r["person_id"] for r in csv.DictReader(f) if r["source"] == "modys_video")
    assert lb_report == lock_persons, "manifest lockbox column and reports/lockbox/lockbox_persons.csv differ"
    log(f"manifest: {len(man)} modys_video records, {sum(r['lockbox'] == 'True' for r in man)} with lockbox True; "
        f"lockbox children {len(lock_persons)} (equal to reports/lockbox/lockbox_persons.csv), "
        f"training children {len(train_persons)}")
    log("lockbox video numbers from the manifest lockbox column: "
        + " ".join(f"{v:03d}" for v in lock_videos_manifest))

    # 3. scoring sheet: four columns only
    sheet = pd.read_excel(os.path.join(raw_dir, "data_clinical_scoring.xlsx"), usecols=USECOLS)
    assert sorted(sheet.columns) == sorted(USECOLS) and len(sheet.columns) == 4, list(sheet.columns)
    columns_read = list(sheet.columns)
    assert not sheet[USECOLS].isna().any().any()
    sheet["video"] = sheet["video"].astype(int)
    sheet["ID"] = sheet["ID"].astype(int)
    sheet["scorer"] = sheet["scorer"].astype(int)
    sheet["time"] = sheet["time"].astype(str)
    assert set(sheet["time"]) <= set(TIMES), f"time values outside {TIMES}: {sorted(set(sheet['time']) - set(TIMES))}"
    sheet["person_ns"] = "modys:child" + sheet["ID"].astype(str).str.zfill(2)
    log(f"data_clinical_scoring.xlsx read with usecols {USECOLS}; columns returned by the read: {columns_read}")
    per_video = sheet.groupby("video").agg(n_child=("person_ns", "nunique"), person_ns=("person_ns", "first"),
                                           n_time=("time", "nunique"), time=("time", "first"))
    assert (per_video["n_child"] == 1).all() and (per_video["n_time"] == 1).all()
    child_of = per_video["person_ns"].to_dict()
    time_of = per_video["time"].to_dict()
    for r in man:
        v = int(r["key"][:3])
        assert child_of.get(v) == r["person_id"], f"manifest person of video {v:03d} differs from the sheet"
    lock_videos_sheet = sorted(v for v, p in child_of.items() if p in lock_persons)
    lock_all = set(lock_videos_manifest) | set(lock_videos_sheet)
    train_videos = {v for v, p in child_of.items() if p in train_persons} - lock_all
    assert not train_videos & lock_all
    extra = sorted(set(lock_videos_sheet) - set(lock_videos_manifest))
    log("lockbox video numbers of the same children on the sheet but not in the manifest: "
        +(" ".join(f"{v:03d}" for v in extra) if extra else "none"))
    log(f"training video numbers on the sheet: {len(train_videos)}")

    # 4. drop every non-training member before any CSV is opened
    zips, plan = {}, {}
    for zname in ZIPS:
        zf = zipfile.ZipFile(os.path.join(raw_dir, zname))
        zips[zname] = zf
        names = zf.namelist()
        members = sorted(n for n in names if n.lower().endswith(".csv") and "__MACOSX" not in n
                         and re.match(r"\d{3}_", os.path.basename(n)))
        others = sorted(n for n in names if n not in members and not n.endswith("/"))
        keep, skipped_lock, skipped_other = [], [], []
        for m in members:
            v = video_of(m)
            if v in lock_all:
                skipped_lock.append(m)
                log(f"skip before parsing (lockbox child): {zname}/{m}")
            elif v not in train_videos:
                skipped_other.append(m)
                log(f"skip before parsing (no training child on the sheet): {zname}/{m}")
            else:
                keep.append(m)
        assert not {video_of(m) for m in keep} & lock_all, "a lockbox video reached the parse list"
        assert len({video_of(m) for m in keep}) == len(keep), "two members share a video number in one zip"
        plan[zname] = {"members": members, "others": others, "keep": keep, "skipped_lock": skipped_lock,
                       "skipped_other": skipped_other}
        log(f"{zname}: {len(members)} coordinate CSVs, {len(skipped_lock)} lockbox skipped, "
            f"{len(skipped_other)} without a training child skipped, {len(keep)} to parse; other members {len(others)}")
    log("lockbox members skipped in both zips before the first CSV is opened: "
        + str(sum(len(p["skipped_lock"]) for p in plan.values())))

    # 5. parse the training CSVs
    with open(os.path.join(HERE, "reports", "nuisance", "frame_rates_facts.json"), encoding="utf-8") as f:
        fps_facts = json.load(f)
    fps = float(fps_facts["modys_source_level_fps"])
    man_frames = {r["key"]: int(r["n_frames"]) for r in man}
    records, frames_e, like_by_part = [], {z: [] for z in ZIPS}, {z: {} for z in ZIPS}
    for zname in ZIPS:
        zf = zips[zname]
        for m in plan[zname]["keep"]:
            wait_if_slow()
            v = video_of(m)
            info = zf.getinfo(m)
            rec = parse_dlc(zf, m, train_videos, lock_all)
            log(f"parsed: {zname}/{m}")
            g = record_geometry(rec)
            frames_e[zname].append(g["e"])
            for j, bp in enumerate(rec["bodyparts"]):
                a = like_by_part[zname].setdefault(bp, [0, 0, 0.0])
                a[0] += int((rec["like"][:, j] >= LIKE).sum())
                a[1] += int(rec["like"].shape[0])
                a[2] = max(a[2], float(np.nanmax(rec["like"][:, j])) if rec["like"].shape[0] else 0.0)
            base = os.path.basename(m)
            key = base[:-4]
            named = [t for bp in rec["bodyparts"] for t in side_tokens(bp)]
            records.append({
                "zip": zname, "file": base, "video": f"{v:03d}", "person_ns": child_of[v], "time": time_of[v],
                "member_crc32": f"{info.CRC:08x}", "header_scorer": ";".join(rec["scorers"]), "name_tag": key[4:],
                "bodyparts": ";".join(rec["bodyparts"]), "n_bodyparts": len(rec["bodyparts"]),
                "rows": g["rows"], "span_s": rnd((g["rows"] - 1) / fps),
                "manifest_n_frames_equal": ("" if key not in man_frames else man_frames[key] == g["rows"]),
                "like_frac_ge_0_5": rnd(g["like_frac"]), "trunk_valid_frac": rnd(g.get("tv_frac")),
                "sufficient": bool(g["sufficient"]), "median_e_deg": rnd(g.get("me")),
                "e_q25_deg": rnd(g.get("e_q25")), "e_q75_deg": rnd(g.get("e_q75")),
                "leg_frac": rnd(g.get("leg_frac")), "median_c_deg": rnd(g.get("mc")), "median_r": rnd(g.get("mr")),
                "top_view_signature": g.get("top_view", ""), "posture_class": g["cls"],
                "both_shoulders_frac": rnd(g.get("both_sh_frac")), "median_w": rnd(g.get("mw")),
                "face_frac": rnd(g.get("face_frac")), "median_f": rnd(g.get("mf")), "median_k": rnd(g.get("mk")),
                "view": g["view"], "facing_evidence": g["evidence"],
                "front_sign": "" if g.get("front_sign") is None else g["front_sign"],
                "facing": g["facing"], "head_direction": g["head_direction"], "camera_side": g["camera_side"],
                "laterality": "from keypoint names" if named else "unknown",
                "d_like_suffix1_minus_suffix2": rnd(g["d_like"]),
            })
    for zf in zips.values():
        zf.close()
    log(f"parsed {len(records)} training records; no lockbox member was opened")
    assert not {int(r["video"]) for r in records} & lock_all

    # 6. aggregates per zip
    facts = {"rule_sha256": rule_hash, "fps_source_level": fps, "likelihood_threshold": LIKE}
    facts["persons"] = {"training_children": len(train_persons), "lockbox_children": len(lock_persons),
                        "lockbox_videos_manifest_column": len(lock_videos_manifest),
                        "lockbox_videos_added_from_sheet": len(extra),
                        "training_videos_on_sheet": len(train_videos)}
    with open(os.path.join(raw_dir, "README.txt"), encoding="cp1252", errors="replace") as f:
        readme = f.read()
    readme_claims = re.findall(r"(\d{3}_[A-Za-z0-9_]+?\.csv)\s*\(in zipfile (data_stickfigure_coordinates_[a-z]+)\)",
                               readme)
    base_members = {z: {os.path.basename(m) for m in plan[z]["members"]} for z in ZIPS}
    zips_out = {}
    for zname in ZIPS:
        rz = [r for r in records if r["zip"] == zname]
        suff = [r for r in rz if r["sufficient"]]
        n = len(suff)
        cls = counts(r["posture_class"] for r in rz)
        n_flat = sum(r["posture_class"] == "flat" for r in suff)
        n_up = sum(r["posture_class"] == "upright" for r in suff)
        if n >= 10 and n_flat >= 0.8 * n:
            verdict = "lying"
        elif n >= 10 and n_up >= 0.8 * n:
            verdict = "sitting"
        else:
            verdict = "not settled"
        ev = np.concatenate(frames_e[zname]) if frames_e[zname] else np.array([])
        hist, edges = np.histogram(ev, bins=np.arange(-90, 91, 15))
        meds = np.array([r["median_e_deg"] for r in suff], dtype=float)
        q = np.quantile(meds, [0.05, 0.25, 0.5, 0.75, 0.95]) if n else [np.nan] * 5
        side = [r for r in suff if r["camera_side"]]
        dpos = [r["d_like_suffix1_minus_suffix2"] is not None and r["d_like_suffix1_minus_suffix2"] > 0 for r in side]
        nl = sum(r["camera_side"] == "left" for r in side)
        nr = sum(r["camera_side"] == "right" for r in side)
        conc = {}
        if side:
            conc["suffix 1 near"] = sum(dpos) / len(side)
            conc["suffix 1 anatomical left"] = sum(p == (r["camera_side"] == "left") for p, r in zip(dpos, side)) / len(side)
            conc["suffix 1 anatomical right"] = sum(p == (r["camera_side"] == "right") for p, r in zip(dpos, side)) / len(side)
        passing = [k for k, val in conc.items() if val >= 0.9] if min(nl, nr) >= 5 else []
        reading = passing[0] if len(passing) == 1 else "undetermined"
        lk = like_by_part[zname]
        tot = [sum(a[0] for a in lk.values()), sum(a[1] for a in lk.values())]
        spans = [r["span_s"] for r in rz]
        p = plan[zname]
        zips_out[zname] = {
            "coordinate_csvs": len(p["members"]), "other_members": p["others"],
            "skipped_lockbox": len(p["skipped_lock"]), "skipped_no_training_child": len(p["skipped_other"]),
            "parsed_training_records": len(rz),
            "training_children_held": len({r["person_ns"] for r in rz}),
            "bodypart_lists": counts(r["bodyparts"] for r in rz),
            "header_scorers": counts(r["header_scorer"] for r in rz),
            "name_tags": counts(r["name_tag"] for r in rz),
            "header_scorer_not_in_name_tag": sum(r["header_scorer"] not in r["name_tag"] for r in rz),
            "keypoint_side_names": sum(r["laterality"] != "unknown" for r in rz),
            "manifest_n_frames_equal": counts(r["manifest_n_frames_equal"] for r in rz),
            "likelihood_frac_ge_0_5": rnd(tot[0] / tot[1] if tot[1] else float("nan")),
            "likelihood_points": tot[1],
            "likelihood_frac_ge_0_5_by_part": {bp: rnd(a[0] / a[1]) for bp, a in lk.items()},
            "likelihood_max_by_part": {bp: rnd(a[2]) for bp, a in lk.items()},
            "descriptive_outside_rule": {
                col: {k: rnd(val) for k, val in zip(
                    ["q05", "q25", "q50", "q75", "q95"],
                    np.quantile([r[col] for r in suff if r[col] is not None], [0.05, 0.25, 0.5, 0.75, 0.95]))}
                for col in ["median_r", "median_c_deg", "median_w"]
                if any(r[col] is not None for r in suff)},
            "spans_s": counts(spans), "records_spanning_20_0_s": sum(s == 20.0 for s in spans),
            "span_min_s": rnd(min(spans)) if spans else None, "span_max_s": rnd(max(spans)) if spans else None,
            "posture": {
                "sufficient_records": n, "class_counts": cls, "flat": n_flat, "upright": n_up,
                "flat_share": rnd(n_flat / n if n else float("nan")),
                "upright_share": rnd(n_up / n if n else float("nan")),
                "verdict": verdict,
                "record_median_e_quantiles_deg": {k: rnd(val) for k, val in zip(["q05", "q25", "q50", "q75", "q95"], q)},
                "frame_e_histogram": {f"[{int(a)},{int(b)}{']' if b == 90 else ')'}": int(h)
                                      for a, b, h in zip(edges[:-1], edges[1:], hist)},
                "frame_e_count": int(ev.size),
            },
            "facing": {"facing": counts(r["facing"] for r in suff), "view": counts(r["view"] for r in suff),
                       "evidence": counts(r["facing_evidence"] for r in suff),
                       "head_direction": counts(r["head_direction"] for r in suff),
                       "camera_side": counts(r["camera_side"] or "none" for r in suff)},
            "laterality": counts(r["laterality"] for r in rz),
            "suffix_evidence": {"records_with_camera_side": len(side), "camera_side_left": nl, "camera_side_right": nr,
                                "d_positive": sum(dpos),
                                "concordance": {k: rnd(val) for k, val in conc.items()},
                                "conventions_passing": passing, "reading": reading},
            "moveatlas_manifest_keys_held": sum(r["key"] + ".csv" in base_members[zname] for r in man),
        }
    verdicts = {z: zips_out[z]["posture"]["verdict"] for z in ZIPS}
    assignment = ("settled" if sorted(verdicts.values()) == ["lying", "sitting"] else "not settled")
    facts["zips"] = zips_out
    facts["posture_assignment"] = {"verdicts": verdicts, "assignment": assignment}
    all_lists = {r["bodyparts"] for r in records}
    all_tags = {r["header_scorer"] for r in records}
    facts["shared"] = {"one_keypoint_set_across_zips": len(all_lists) == 1,
                       "one_model_tag_across_zips": len(all_tags) == 1,
                       "distinct_keypoint_lists": len(all_lists), "distinct_header_scorers": len(all_tags),
                       "keypoint_set_equal_as_sets": len({frozenset(s.split(";")) for s in all_lists}) == 1}
    facts["readme_zip_claims"] = [{"file": fn, "readme_zip": rz_name,
                                   "actual_zip": [z for z in ZIPS if fn in base_members[z]]}
                                  for fn, rz_name in readme_claims]
    facts["moveatlas_manifest_records"] = len(man)

    # pooled suffix evidence (descriptive, never laterality)
    side = [r for r in records if r["sufficient"] and r["camera_side"]]
    dpos = [r["d_like_suffix1_minus_suffix2"] is not None and r["d_like_suffix1_minus_suffix2"] > 0 for r in side]
    nl = sum(r["camera_side"] == "left" for r in side)
    nr = sum(r["camera_side"] == "right" for r in side)
    conc = {}
    if side:
        conc["suffix 1 near"] = sum(dpos) / len(side)
        conc["suffix 1 anatomical left"] = sum(p == (r["camera_side"] == "left") for p, r in zip(dpos, side)) / len(side)
        conc["suffix 1 anatomical right"] = sum(p == (r["camera_side"] == "right") for p, r in zip(dpos, side)) / len(side)
    passing = [k for k, val in conc.items() if val >= 0.9] if min(nl, nr) >= 5 else []
    facts["suffix_evidence_pooled"] = {"records_with_camera_side": len(side), "camera_side_left": nl,
                                       "camera_side_right": nr, "d_positive": sum(dpos),
                                       "concordance": {k: rnd(val) for k, val in conc.items()},
                                       "conventions_passing": passing,
                                       "reading": passing[0] if len(passing) == 1 else "undetermined"}
    facts["laterality"] = counts(r["laterality"] for r in records)

    # 7. scoring structure on the training videos (four columns only)
    held = {z: {int(r["video"]) for r in records if r["zip"] == z} for z in ZIPS}
    st = sheet[sheet["video"].isin(train_videos)]
    sbv = []
    for v, g in st.groupby("video"):
        sc = sorted(g["scorer"].unique().tolist())
        sbv.append({"video": f"{v:03d}", "person_ns": child_of[v], "time": time_of[v], "n_scoring_rows": len(g),
                    "n_scorers": len(sc), "scorers": "+".join(str(s) for s in sc),
                    "in_zip_named_lying": v in held[ZIPS[0]], "in_zip_named_sitting": v in held[ZIPS[1]]})
    sbv.sort(key=lambda r: r["video"])

    def scoring_counts(rows):
        return {"videos": len(rows), "double_scored": sum(r["n_scorers"] == 2 for r in rows),
                "single_scored": sum(r["n_scorers"] == 1 for r in rows),
                "other_scorer_count": sum(r["n_scorers"] not in (1, 2) for r in rows),
                "rows_differ_from_scorers": sum(r["n_scoring_rows"] != r["n_scorers"] for r in rows),
                "scorer_sets": counts(r["scorers"] for r in rows),
                "by_time": {t: {"videos": sum(r["time"] == t for r in rows),
                                "double_scored": sum(r["time"] == t and r["n_scorers"] == 2 for r in rows)}
                            for t in TIMES}}

    facts["scoring"] = {"sheet_rows_training": int(len(st)),
                        "on_sheet": scoring_counts(sbv),
                        "held_in_zip_named_lying": scoring_counts([r for r in sbv if r["in_zip_named_lying"]]),
                        "held_in_zip_named_sitting": scoring_counts([r for r in sbv if r["in_zip_named_sitting"]]),
                        "held_in_both_zips": scoring_counts([r for r in sbv if r["in_zip_named_lying"]
                                                             and r["in_zip_named_sitting"]]),
                        "videos_scored_per_scorer": {str(s): int(st[st["scorer"] == s]["video"].nunique())
                                                     for s in sorted(st["scorer"].unique())}}

    # 8. visits per training child
    def visit_counts(vids):
        per = {}
        for v in vids:
            per.setdefault(child_of[v], set()).add(time_of[v])
        return {"children_with_any": len(per),
                "children_T0_and_T12": sum({"T0", "T12"} <= s for s in per.values()),
                "children_T0_T3_T12": sum({"T0", "T3", "T12"} <= s for s in per.values()),
                "visits_per_child": counts(len(s) for s in per.values())}

    facts["visits"] = {"on_sheet": visit_counts(train_videos),
                       "held_in_zip_named_lying": visit_counts(held[ZIPS[0]]),
                       "held_in_zip_named_sitting": visit_counts(held[ZIPS[1]]),
                       "held_in_both_zips": visit_counts(held[ZIPS[0]] & held[ZIPS[1]])}
    facts["videos_held"] = {"zip_named_lying_only": sorted(f"{v:03d}" for v in held[ZIPS[0]] - held[ZIPS[1]]),
                            "zip_named_sitting_only": sorted(f"{v:03d}" for v in held[ZIPS[1]] - held[ZIPS[0]]),
                            "on_sheet_in_neither_zip": sorted(f"{v:03d}" for v in train_videos - held[ZIPS[0]]
                                                              - held[ZIPS[1]])}
    both = held[ZIPS[0]] & held[ZIPS[1]]
    rows_by = {(r["zip"], int(r["video"])): r["rows"] for r in records}
    facts["rows_equal_across_zips"] = {"videos_in_both": len(both),
                                       "equal_rows": sum(rows_by[(ZIPS[0], v)] == rows_by[(ZIPS[1], v)] for v in both)}

    # 9. write outputs
    os.makedirs(OUT, exist_ok=True)

    def write_csv(path, rows):
        cols = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cols)
            for r in rows:
                w.writerow(["" if r[c] is None else r[c] for c in cols])

    write_csv(os.path.join(OUT, "per_record.csv"), records)
    write_csv(os.path.join(OUT, "scorer_by_video.csv"), sbv)
    with open(os.path.join(OUT, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1)
        f.write("\n")
    write_summary(facts, rule_hash)
    with open(os.path.join(OUT, "run.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log.lines) + "\n")

    # 10. provenance
    outputs = ["summary.md", "per_record.csv", "scorer_by_video.csv", "facts.json", "run.log"]
    prov = {"script": "scripts/probe_modys_structure.py", "rule": "reports/modys_structure/rule.txt",
            "rule_sha256": rule_hash, "moveatlas_root": root, "moveatlas_stamp": stamp_moveatlas(root),
            "inputs": {}, "outputs_sha256": {o: sha256_file(os.path.join(OUT, o)) for o in outputs},
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "note": ("input hashes are taken over the files as stored (zip archives compressed, no member "
                     "decompressed for hashing); only the training members listed in run.log were parsed. "
                     "The outputs listed in outputs_sha256 are byte-identical across runs; this file differs "
                     "only by the stamp time.")}
    for rel in ["datasets/pose/modys_video/raw/README.txt", "datasets/pose/modys_video/raw/data_clinical_scoring.xlsx"] + \
            [f"datasets/pose/modys_video/raw/{z}" for z in ZIPS]:
        pth = os.path.join(root, rel)
        prov["inputs"][rel] = {"bytes": os.path.getsize(pth), "sha256": sha256_file(pth)}
    for rel in ["cache/manifest/manifest_records.csv.gz", "reports/lockbox/lockbox_persons.csv",
                "reports/nuisance/frame_rates_facts.json"]:
        pth = os.path.join(HERE, rel)
        prov["inputs"][rel] = {"bytes": os.path.getsize(pth), "sha256": sha256_file(pth)}
    with open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    for o in outputs + ["_provenance.json"]:
        print(f"{sha256_file(os.path.join(OUT, o))}  {o}")


def stamp_moveatlas(root):
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    import moveatlas_stamp  # noqa: E402  (read-only HEAD, diff hash, untracked files, manifest_tiers note)
    return moveatlas_stamp.stamp(root)


# ---------------------------------------------------------------- summary

def write_summary(facts, rule_hash):
    Z = ZIPS
    zo = facts["zips"]
    L = []
    add = L.append
    add("# modys_video structure on the 27 training children (T10)")
    add("")
    add("Measured by scripts/probe_modys_structure.py against the MoveAtlas tree (stamp in _provenance.json),")
    add(f"under the decision rule rule.txt (SHA-256 {rule_hash}), written and hashed before the data read.")
    add("Every number below is in facts.json, per_record.csv or scorer_by_video.csv. Lockbox children: no")
    add("coordinate file of theirs was opened (run.log lists the members skipped before the first parse).")
    add("No DIS item column was read and no statistic involving a DIS item was computed.")
    add("")
    p = facts["persons"]
    add("## Scope")
    add("")
    add(f"- Children: {p['training_children']} training, {p['lockbox_children']} lockbox (lockbox column of "
        f"cache/manifest/manifest_records.csv.gz, equal to reports/lockbox/lockbox_persons.csv).")
    add(f"- Lockbox video numbers: {p['lockbox_videos_manifest_column']} from the manifest column, "
        f"{p['lockbox_videos_added_from_sheet']} more of the same children from the scoring sheet; training video "
        f"numbers on the sheet: {p['training_videos_on_sheet']}.")
    add("")
    add("| zip | coordinate CSVs | lockbox skipped | other skipped | training parsed | training children held |")
    add("|---|---|---|---|---|---|")
    for z in Z:
        o = zo[z]
        add(f"| {z} | {o['coordinate_csvs']} | {o['skipped_lockbox']} | {o['skipped_no_training_child']} | "
            f"{o['parsed_training_records']} | {o['training_children_held']} |")
    add("")
    add("## Keypoint set and model tag")
    add("")
    for z in Z:
        o = zo[z]
        add(f"- {z}: body-part lists {o['bodypart_lists']}; header scorers {o['header_scorers']}; "
            f"file-name tags {o['name_tags']}; records whose header scorer is not the model named in the file name: "
            f"{o['header_scorer_not_in_name_tag']}.")
    s = facts["shared"]
    add(f"- One keypoint set across both zips (ordered list): {s['one_keypoint_set_across_zips']} "
        f"(as sets: {s['keypoint_set_equal_as_sets']}); one model tag across both zips: "
        f"{s['one_model_tag_across_zips']} ({s['distinct_header_scorers']} distinct header scorers).")
    add("")
    add("## Posture from the trunk axis (rule sections 3 and 4)")
    add("")
    add("| zip | sufficient records | flat | upright | class counts | median e quantiles q05/q25/q50/q75/q95 (deg) | verdict |")
    add("|---|---|---|---|---|---|---|")
    for z in Z:
        po = zo[z]["posture"]
        qq = po["record_median_e_quantiles_deg"]
        add(f"| {z} | {po['sufficient_records']} | {po['flat']} ({po['flat_share']}) | {po['upright']} "
            f"({po['upright_share']}) | {po['class_counts']} | {qq['q05']} / {qq['q25']} / {qq['q50']} / "
            f"{qq['q75']} / {qq['q95']} | {po['verdict']} |")
    add("")
    add("Trunk elevation e over all trunk-valid frames (frames per 15-degree bin):")
    add("")
    bins = list(zo[Z[0]]["posture"]["frame_e_histogram"].keys())
    add("| zip | " + " | ".join(bins) + " |")
    add("|---|" + "---|" * len(bins))
    for z in Z:
        h = zo[z]["posture"]["frame_e_histogram"]
        add(f"| {z} | " + " | ".join(str(h[b]) for b in bins) + " |")
    add("")
    pa = facts["posture_assignment"]
    add(f"- Zip-to-posture assignment: {pa['assignment']} (verdicts {pa['verdicts']}).")
    for c in facts["readme_zip_claims"]:
        add(f"- README.txt places {c['file']} in {c['readme_zip']}; the file is in {c['actual_zip']}.")
    if pa["assignment"] == "settled":
        agree_names = all((("lying" in z) == (pa["verdicts"][z] == "lying")) for z in Z)
        held_by = {z: zo[z]["moveatlas_manifest_keys_held"] for z in Z}
        canon_zip = [z for z in Z if held_by[z] == facts["moveatlas_manifest_records"]]
        if agree_names:
            add("- Reading: the geometry agrees with the zip names; the README sentence that places the files is the swapped one.")
        else:
            add("- Reading: the geometry contradicts the zip names and agrees with the README sentence that places the files:")
            add("  the zip named lying holds the sitting protocol and the zip named sitting the lying protocol.")
        if canon_zip:
            add(f"- The {facts['moveatlas_manifest_records']} MoveAtlas canonical records all sit in {canon_zip[0]}, whose "
                f"verdict is '{pa['verdicts'][canon_zip[0]]}' on the training records (configs/canonical_skeleton.yaml "
                "of MoveAtlas calls them side-view lying).")
    fr = {z: (zo[z]["facing"]["view"].get("frontal", 0), zo[z]["posture"]["sufficient_records"]) for z in Z}
    if pa["assignment"] != "settled" and all(n and f >= 0.5 * n for f, n in fr.values()):
        add("- Reading: in both zips the trunk is near vertical in the image and most sufficient records show a frontal")
        add("  view (both shoulders apart: " + ", ".join(f"{z} {f} of {n}" for z, (f, n) in fr.items()) + "). The trunk")
        add("  axis alone does not separate a supine child filmed from above or from the feet from a sitting child")
        add("  filmed from the front, so rule version 1 leaves the posture of both zips open.")
    add("")
    add("Descriptive, outside the rule (added after the first run, no verdict): quantiles q05/q25/q50/q75/q95 over the")
    add("sufficient records of the record medians of the thigh ratio r, the continuation angle c and the shoulder width w.")
    add("")
    add("| zip | r | c (deg) | w |")
    add("|---|---|---|---|")
    for z in Z:
        d = zo[z]["descriptive_outside_rule"]
        cells = [" / ".join(str(d[col][k]) for k in ["q05", "q25", "q50", "q75", "q95"]) if col in d else ""
                 for col in ["median_r", "median_c_deg", "median_w"]]
        add(f"| {z} | " + " | ".join(cells) + " |")
    add("")
    add("## Facing direction (rule section 5)")
    add("")
    add("| zip | facing | view | evidence | head direction | camera side |")
    add("|---|---|---|---|---|---|")
    for z in Z:
        fa = zo[z]["facing"]
        add(f"| {z} | {fa['facing']} | {fa['view']} | {fa['evidence']} | {fa['head_direction']} | {fa['camera_side']} |")
    add("")
    add("Per record in per_record.csv (facing, head_direction, camera_side, facing_evidence). Camera side is the")
    add("anatomical side toward the camera under the no-mirroring assumption; it is not a keypoint laterality.")
    face_max = {z: max(zo[z]["likelihood_max_by_part"].get(p, 0.0) or 0.0 for p in ("forehead", "chin")) for z in Z}
    if all(v < LIKE for v in face_max.values()):
        add("Forehead and chin never reach likelihood 0.5 in any training frame (largest value per zip: "
            + ", ".join(f"{z} {v}" for z, v in face_max.items()) + "), so the face cue is never available.")
    add("")
    add("## Laterality (rule section 6)")
    add("")
    add(f"- Laterality per record: {facts['laterality']}.")
    for z in Z:
        se = zo[z]["suffix_evidence"]
        add(f"- {z}: suffix evidence over {se['records_with_camera_side']} records with a camera side "
            f"(left {se['camera_side_left']}, right {se['camera_side_right']}; d > 0 in {se['d_positive']}): "
            f"concordance {se['concordance']}; reading: {se['reading']}.")
    sp = facts["suffix_evidence_pooled"]
    add(f"- Both zips pooled: {sp['records_with_camera_side']} records (left {sp['camera_side_left']}, right "
        f"{sp['camera_side_right']}; d > 0 in {sp['d_positive']}): concordance {sp['concordance']}; reading: "
        f"{sp['reading']}.")
    add("- The reading is evidence only; by the rule it never sets a laterality.")
    add("")
    add("## Scoring structure of the training videos (columns ID, video, time, scorer)")
    add("")
    add("| set | videos | double-scored | single-scored | scorer sets | T0 / T3 / T12 videos (double) |")
    add("|---|---|---|---|---|---|")
    sc = facts["scoring"]
    for key in ["on_sheet", "held_in_zip_named_lying", "held_in_zip_named_sitting", "held_in_both_zips"]:
        o = sc[key]
        bt = " / ".join(f"{o['by_time'][t]['videos']} ({o['by_time'][t]['double_scored']})" for t in TIMES)
        add(f"| {key} | {o['videos']} | {o['double_scored']} | {o['single_scored']} | {o['scorer_sets']} | {bt} |")
    add("")
    add(f"- Videos scored per scorer (training, on the sheet): {sc['videos_scored_per_scorer']}; videos whose row "
        f"count differs from their scorer count: {sc['on_sheet']['rows_differ_from_scorers']}.")
    add("")
    add("## Visits per training child")
    add("")
    add("| set | children with a record | T0 and T12 | T0, T3 and T12 | visits per child |")
    add("|---|---|---|---|---|")
    for key, o in facts["visits"].items():
        add(f"| {key} | {o['children_with_any']} | {o['children_T0_and_T12']} | {o['children_T0_T3_T12']} | "
            f"{o['visits_per_child']} |")
    add("")
    vh = facts["videos_held"]
    add(f"- Training videos held in the zip named lying only: {vh['zip_named_lying_only']}; in the zip named sitting "
        f"only: {vh['zip_named_sitting_only']}; on the sheet in neither zip: {vh['on_sheet_in_neither_zip']}.")
    re_ = facts["rows_equal_across_zips"]
    add(f"- Videos held in both zips: {re_['videos_in_both']}; same row count in both zips: {re_['equal_rows']}.")
    add("")
    add(f"## Clip spans at {facts['fps_source_level']} fps (reports/nuisance/frame_rates_facts.json)")
    add("")
    for z in Z:
        o = zo[z]
        add(f"- {z}: {o['records_spanning_20_0_s']} of {o['parsed_training_records']} training records span 20.0 s; "
            f"range {o['span_min_s']} to {o['span_max_s']} s; spans (s: records) {o['spans_s']}.")
        if o["manifest_n_frames_equal"] != {"": o["parsed_training_records"]}:
            add(f"  Rows equal to the MoveAtlas manifest n_frames (records in the manifest): {o['manifest_n_frames_equal']}.")
    add("")
    add("## DeepLabCut likelihood at or above 0.5")
    add("")
    add("| zip | points | fraction | " + " | ".join(zo[Z[0]]["likelihood_frac_ge_0_5_by_part"].keys()) + " |")
    add("|---|---|---|" + "---|" * len(zo[Z[0]]["likelihood_frac_ge_0_5_by_part"]))
    for z in Z:
        o = zo[z]
        bp = o["likelihood_frac_ge_0_5_by_part"]
        add(f"| {z} | {o['likelihood_points']} | {o['likelihood_frac_ge_0_5']} | "
            + " | ".join(str(bp.get(k, "")) for k in zo[Z[0]]["likelihood_frac_ge_0_5_by_part"].keys()) + " |")
    for z in Z:
        mx = zo[z]["likelihood_max_by_part"]
        add(f"| {z}, largest likelihood | | | "
            + " | ".join(str(mx.get(k, "")) for k in zo[Z[0]]["likelihood_frac_ge_0_5_by_part"].keys()) + " |")
    add("")
    add("## Files")
    add("")
    add("- rule.txt, rule.txt.sha256: the decision rule and its hash, written before the data read.")
    add("- per_record.csv: one row per parsed training record and zip.")
    add("- scorer_by_video.csv: one row per training video of the scoring sheet (four columns read).")
    add("- facts.json: every number of this summary. run.log: the skip and parse order. _provenance.json: stamp,")
    add("  input and output hashes.")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
