"""fog_turning_filho: inventory of the fetched videos, matched to the IMU trials held by MoveAtlas.

Reads, without extracting or writing anything outside reports/: Videos.zip in the local cache
written by scripts/fetch_fog_turning_video.py (the MP4 boxes of each member are parsed from its
decompressed stream: mvhd, tkhd, mdhd, hdlr, stsd, stts), and MoveAtlas
datasets/pose/fog_turning_filho/raw/IMU.zip and PDFEinfo.csv (read-only). Runs no git command on
MoveAtlas: the MoveAtlas commit is copied from reports/probe_cells/_provenance.json.
Writes tables only, never a frame or a media file, under reports/fog_videos/:
- video_inventory.csv: one row per video (name, subject, session, duration, frame rate,
  frames, constant frame rate or not, resolution, codec, audio tracks);
- imu_trials.csv: one row per IMU trial (samples, duration, rate, freezing-flag fraction,
  flagged runs, flagged seconds, first flagged time);
- pairs.csv: video and IMU trial side by side with the duration difference;
- session_fog.csv: flagged runs and seconds per IMU trial against the per-session episode
  count and time in freezing of PDFEinfo.csv;
- fog_videos_summary.md and _provenance.json.

Run: python scripts/probe_fog_videos.py --cache <workdir>/data_cache/fog_turning_filho --moveatlas <moveatlas>
"""
import argparse
import csv
import io
import json
import os
import re
import statistics
import struct
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="<workdir>/data_cache/fog_turning_filho")
ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
args = ap.parse_args()
out = os.path.join(here, "reports", "fog_videos")
os.makedirs(out, exist_ok=True)
raw = os.path.join(args.moveatlas, "datasets", "pose", "fog_turning_filho", "raw")
NAME_RE = re.compile(r"(?:SUB|PDFE)[_ -]?(\d+)[_ -]?(standing|\d+)", re.I)


def key(name):
    m = NAME_RE.search(os.path.basename(name))
    return (int(m.group(1)), m.group(2).lower()) if m else None


# ---------------------------------------------------------------- MP4 boxes
def boxes(buf, off=0, end=None):
    end = len(buf) if end is None else end
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", buf[off + 8:off + 16])[0], 16
        elif size == 0:
            size = end - off
        if size < hdr:
            break
        yield typ.decode("latin-1"), off + hdr, off + size
        off += size


def child(buf, a, b, name):
    return next(((s, e) for t, s, e in boxes(buf, a, b) if t == name), None)


def timescale_duration(buf, a):
    """mvhd and mdhd: version 1 has 64-bit times."""
    if buf[a] == 1:
        return struct.unpack(">I", buf[a + 20:a + 24])[0], struct.unpack(">Q", buf[a + 24:a + 32])[0]
    return struct.unpack(">I", buf[a + 12:a + 16])[0], struct.unpack(">I", buf[a + 16:a + 20])[0]


def read_moov(zf, info):
    """Stream one member and keep only its moov box; mdat is read and dropped."""
    with zf.open(info) as fh:
        while True:
            h = fh.read(8)
            if len(h) < 8:
                return None
            size, typ = struct.unpack(">I4s", h)
            hdr = 8
            if size == 1:
                size, hdr = struct.unpack(">Q", fh.read(8))[0], 16
            body = size - hdr
            if typ == b"moov":
                return fh.read(body)
            while body > 0:
                chunk = fh.read(min(body, 8 * 1024 * 1024))
                if not chunk:
                    return None
                body -= len(chunk)


def mp4_facts(moov):
    f = {"codec": "", "width": "", "height": "", "fps": "", "frames": "", "constant_frame_rate": "",
         "duration_s": 0.0, "audio_tracks": 0}
    mv = child(moov, 0, len(moov), "mvhd")
    if mv:
        ts, du = timescale_duration(moov, mv[0])
        f["duration_s"] = round(du / ts, 3) if ts else 0.0
    for t, a, b in boxes(moov):
        if t != "trak":
            continue
        md = child(moov, a, b, "mdia")
        hd = child(moov, md[0], md[1], "hdlr") if md else None
        handler = moov[hd[0] + 8:hd[0] + 12].decode("latin-1") if hd else ""
        if handler == "soun":
            f["audio_tracks"] += 1
            continue
        if handler != "vide":
            continue
        mh = child(moov, md[0], md[1], "mdhd")
        ts, du = timescale_duration(moov, mh[0]) if mh else (0, 0)
        mi = child(moov, md[0], md[1], "minf")
        st = child(moov, mi[0], mi[1], "stbl") if mi else None
        if not st:
            continue
        sd = child(moov, st[0], st[1], "stsd")
        if sd:
            e = sd[0] + 8
            f["codec"] = moov[e + 4:e + 8].decode("latin-1")
            f["width"], f["height"] = struct.unpack(">HH", moov[e + 32:e + 36])
        tt = child(moov, st[0], st[1], "stts")
        if tt:
            n = struct.unpack(">I", moov[tt[0] + 4:tt[0] + 8])[0]
            ent = [struct.unpack(">II", moov[tt[0] + 8 + 8 * k:tt[0] + 16 + 8 * k]) for k in range(n)]
            frames = sum(c for c, _ in ent)
            f["frames"] = frames
            f["constant_frame_rate"] = len({d for _, d in ent}) == 1
            if du and ts:
                f["fps"] = round(frames / (du / ts), 3)
    return f


# ---------------------------------------------------------------- videos
videos, members = [], []
with zipfile.ZipFile(os.path.join(args.cache, "Videos.zip")) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        members.append(info.filename)
        if not info.filename.lower().endswith(".mp4"):
            continue
        moov = read_moov(zf, info)
        f = mp4_facts(moov) if moov else {"codec": "unparsed"}
        k = key(info.filename)
        row = {"file": info.filename, "subject": k[0] if k else "", "trial": k[1] if k else "",
               "size_bytes": info.file_size}
        row.update(f)
        videos.append(row)
        print(info.filename, f.get("duration_s"), f.get("fps"), f.get("width"), f.get("height"), f.get("codec"),
              "audio" if f.get("audio_tracks") else "no audio", flush=True)

# ---------------------------------------------------------------- IMU trials
imu = []
with zipfile.ZipFile(os.path.join(raw, "IMU.zip")) as z:
    for n in sorted(z.namelist()):
        if not n.endswith(".txt"):
            continue
        rows = list(csv.reader(io.StringIO(z.read(n).decode("utf-8", "replace")), delimiter="\t"))
        head, data = rows[0], [r for r in rows[1:] if r]
        t = [float(r[1]) for r in data]
        row = {"file": n, "subject": key(n)[0], "trial": key(n)[1], "samples": len(data),
               "duration_s": t[-1] if t else 0, "rate_hz": round((len(t) - 1) / (t[-1] - t[0]), 3) if len(t) > 1 else "",
               "flag_column": "Freezing event [flag]" in head, "flagged_runs": "", "flagged_fraction": "",
               "flagged_s": "", "first_flag_s": ""}
        if row["flag_column"]:
            c = head.index("Freezing event [flag]")
            fl = [int(float(r[c])) for r in data]
            starts = [i for i in range(len(fl)) if fl[i] == 1 and (i == 0 or fl[i - 1] == 0)]
            row.update({"flagged_runs": len(starts), "flagged_fraction": round(sum(fl) / len(fl), 4),
                        "flagged_s": round(sum(fl) / row["rate_hz"], 2) if row["rate_hz"] else "",
                        "first_flag_s": t[starts[0]] if starts else ""})
        imu.append(row)

# ---------------------------------------------------------------- pairs
vk = {(v["subject"], v["trial"]): v for v in videos if v["subject"] != ""}
ik = {(i["subject"], i["trial"]): i for i in imu}
pairs = []
for k in sorted(set(vk) | set(ik)):
    v, i = vk.get(k), ik.get(k)
    pairs.append({"subject": k[0], "trial": k[1], "video": v["file"] if v else "", "imu": i["file"] if i else "",
                  "video_s": v["duration_s"] if v else "", "imu_s": round(i["duration_s"], 3) if i else "",
                  "diff_s": round(v["duration_s"] - i["duration_s"], 3) if v and i else ""})

# ---------------------------------------------------------------- PDFEinfo sessions
with open(os.path.join(raw, "PDFEinfo.csv"), encoding="latin-1", newline="") as f:
    info_rows = list(csv.reader(f, delimiter=";"))
hdr = [re.sub(r"\s+", " ", h).strip() for h in info_rows[0]]


def col(r, label):
    return r[hdr.index(label)].strip() if label in hdr and hdr.index(label) < len(r) else ""


sess = []
for r in info_rows[1:]:
    if not r or not r[0].strip().upper().startswith("PDFE"):
        continue
    sid = int(re.sub(r"\D", "", r[0]))
    for s in (1, 2, 3):
        i = ik.get((sid, str(s)))
        v = vk.get((sid, str(s)))
        sess.append({"subject": sid, "session": s,
                     "episodes_pdfe": col(r, f"Session {s} - numbers of FoG episodes (n)"),
                     "time_in_fog_pdfe_s": col(r, f"Session {s} - total time in FoG (s)"),
                     "imu_trial": i["file"] if i else "", "video": v["file"] if v else "",
                     "flagged_runs_imu": i["flagged_runs"] if i else "",
                     "flagged_s_imu": i["flagged_s"] if i else ""})


def write(name, rows):
    if rows:
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


write("video_inventory.csv", videos)
write("imu_trials.csv", imu)
write("pairs.csv", pairs)
write("session_fog.csv", sess)


def num(x):
    try:
        return float(str(x).replace(",", "."))
    except ValueError:
        return None


both = [p for p in pairs if p["video"] and p["imu"]]
diffs = [p["diff_s"] for p in both]
turning = [i for i in imu if i["trial"] != "standing"]
cmp_rows = [s for s in sess if s["imu_trial"] and num(s["episodes_pdfe"]) is not None]
cmp_time = [s for s in sess if s["imu_trial"] and num(s["time_in_fog_pdfe_s"]) is not None and s["flagged_s_imu"] != ""]
facts = {
    "zip_members": len(members), "videos": len(videos),
    "video_subjects": len({v["subject"] for v in videos if v["subject"] != ""}),
    "video_sessions": dict(sorted(((str(t), sum(1 for v in videos if v["trial"] == t)) for t in {v["trial"] for v in videos}))),
    "videos_unparsed_name": [v["file"] for v in videos if v["subject"] == ""],
    "video_gb": round(sum(v["size_bytes"] for v in videos) / 1e9, 2),
    "codecs": sorted({str(v.get("codec")) for v in videos}),
    "resolutions": sorted({f"{v.get('width')}x{v.get('height')}" for v in videos}),
    "fps_values": sorted({v.get("fps") for v in videos if v.get("fps") != ""}),
    "videos_constant_frame_rate": sum(1 for v in videos if v.get("constant_frame_rate") is True),
    "videos_with_audio": sum(1 for v in videos if v.get("audio_tracks")),
    "video_duration_s_min_median_max": [min(v["duration_s"] for v in videos), statistics.median(v["duration_s"] for v in videos),
                                        max(v["duration_s"] for v in videos)] if videos else None,
    "imu_trials": len(imu), "imu_turning_trials": len(turning), "imu_standing_trials": len(imu) - len(turning),
    "imu_turning_with_flag": sum(1 for i in turning if i["flag_column"]),
    "imu_turning_duration_s_min_median_max": [min(i["duration_s"] for i in turning), statistics.median(i["duration_s"] for i in turning),
                                              max(i["duration_s"] for i in turning)] if turning else None,
    "pairs_matched": len(both),
    "video_without_imu": [p["video"] for p in pairs if p["video"] and not p["imu"]],
    "imu_turning_without_video": [p["imu"] for p in pairs if p["imu"] and not p["video"] and p["trial"] != "standing"],
    "duration_diff_s_min_median_max": [min(diffs), statistics.median(diffs), max(diffs)] if diffs else None,
    "pairs_duration_within_1s": sum(1 for d in diffs if abs(d) <= 1.0),
    "session_rows_compared": len(cmp_rows),
    "session_rows_episode_count_equal": sum(1 for s in cmp_rows if s["flagged_runs_imu"] == num(s["episodes_pdfe"])),
    "session_rows_time_in_fog_within_1s": sum(1 for s in cmp_time if abs(num(s["time_in_fog_pdfe_s"]) - s["flagged_s_imu"]) <= 1.0),
    "session_rows_time_compared": len(cmp_time),
}
md = ["# fog_turning_filho videos against the IMU trials", "",
      "Measured by scripts/probe_fog_videos.py (provenance in _provenance.json). The videos stay in the local",
      "cache outside the tree and were read inside the archive; this folder holds tables only.", "",
      "| quantity | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in facts.items()]
open(os.path.join(out, "fog_videos_summary.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
head = ""
pc = os.path.join(here, "reports", "probe_cells", "_provenance.json")
if os.path.exists(pc):
    head = json.load(open(pc, encoding="utf-8")).get("moveatlas_head", "")
journal = {}
jp = os.path.join(args.cache, "fetch_journal.json")
if os.path.exists(jp):
    journal = json.load(open(jp, encoding="utf-8"))
json.dump({"script": "scripts/probe_fog_videos.py", "moveatlas_head_from_probe_cells": head, "video_fetch": journal,
           "facts": facts}, open(os.path.join(out, "_provenance.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(facts, indent=1))
