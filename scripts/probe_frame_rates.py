"""Frame rates of the pose records whose MoveAtlas manifest fps is blank (plan task T0.7a).

Scope: modys_video (manifest source label "modys", 94 records), li_taati (511 pooled
records plus 132 leg agility records of the per-source manifest) and remap_open (3,498
turning records; the 403 sit-to-stand records, which carry fps 30.0 in the manifest, are
checked as well because they come from the same camera platform).

For each record the script reads the manifest fps, then looks for the true rate in the raw
files of the source, and writes one row with an evidence level:

  measured_container  the rate is read from a movie container that holds the same pose
                      frames (modys_video: the source's own stick-figure movies, parsed in
                      memory inside the zip, never extracted and never decoded)
  measured_container_source_level
                      as above, but this record's movie frame count does not match its
                      current coordinate file, so the value is the source-level one
  stated_by_source    the rate is stated in the source's documentation on disk
  stated_in_file      a per-frame time column inside the data file gives the rate, and the
                      column is generated from a nominal rate (it is not a clock)
  unresolved          no file on disk states or measures the rate

Corroboration that does not by itself set a rate (remap_open: the frame counts of the
turning clips against the rater-labelled turn durations) is reported beside the rate.

Reads (read-only, MoveAtlas): datasets/_canonical_manifest.csv; datasets/pose/li_taati/
processed/canonical_manifest.csv; datasets/pose/modys_video/raw/*.zip and README.txt;
datasets/pose/li_taati/raw/UDysRS_UPDRS_Export/ (trajectory exports, notebook, licence);
datasets/pose/remap_open/raw/remap_open.zip (readme.txt, turning keypoint CSVs, turning
labels, sit-to-stand CSVs); the three datacards; docs/datasets_index.csv; docs/citations.csv;
docs/demographics_curated.csv (only to name the Europe PMC record of the li_taati data paper).
Clinical rating files are not read.

Writes (reports/nuisance/): frame_rates.csv, frame_rates_by_source.csv,
frame_rates_modys_movies.csv, frame_rates_remap_turning_fit.csv, frame_rates_facts.json,
frame_rates_summary.md, _provenance.json and _provenance_probe_frame_rates.json (the same
content; the second name survives sibling scripts that write into the same directory).

Run: <conda>/python.exe scripts/probe_frame_rates.py
"""
import collections
import io
import json
import os
import re
import shutil
import struct
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if "--out" not in sys.argv:
    sys.argv += ["--out", os.path.join(os.path.dirname(HERE), "reports", "nuisance")]
from _common import parse_args, read_csv, write_provenance  # noqa: E402

RATE_RE = re.compile(r"(?i)\b(fps|frame[ -]?rate|frames per second|frames/s|hz)\b")
EVIDENCE_LEVELS = ["measured_container", "measured_container_source_level", "stated_by_source",
                   "stated_in_file", "unresolved"]


def log(*x):
    print(*x, flush=True)


# ----------------------------------------------------------------------------- MP4 parsing
def _boxes(buf, off, end):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr:
            return
        yield typ.decode("latin-1"), off, hdr, size
        off += size


def parse_mp4(buf):
    """Container facts of the first video track: timescale, sample deltas, sample count."""
    out = {"tracks": 0}

    def rec(off, end):
        for t, o, h, s in _boxes(buf, off, end):
            if t in ("moov", "mdia", "minf", "stbl"):
                rec(o + h, o + s)
            elif t == "trak":
                out["tracks"] += 1
                if out["tracks"] == 1:
                    rec(o + h, o + s)
            elif t in ("mvhd", "mdhd"):
                v = buf[o + h]
                if v == 0:
                    ts, du = struct.unpack(">II", buf[o + h + 12:o + h + 20])
                else:
                    ts, du = struct.unpack(">IQ", buf[o + h + 20:o + h + 32])
                out[t + "_timescale"], out[t + "_duration"] = ts, du
            elif t == "hdlr" and "handler" not in out:
                out["handler"] = buf[o + h + 8:o + h + 12].decode("latin-1")
            elif t == "stts":
                n = struct.unpack(">I", buf[o + h + 4:o + h + 8])[0]
                out["stts"] = [struct.unpack(">II", buf[o + h + 8 + 8 * i:o + h + 16 + 8 * i]) for i in range(n)]
            elif t == "stsz":
                out["stsz_count"] = struct.unpack(">I", buf[o + h + 8:o + h + 12])[0]
            elif t == "stsd":
                out["codec"] = buf[o + h + 12:o + h + 16].decode("latin-1")
                out["width"], out["height"] = struct.unpack(">HH", buf[o + h + 8 + 8 + 24:o + h + 8 + 8 + 28])

    rec(0, len(buf))
    return out


# ----------------------------------------------------------------------------- manifests
def load_manifests(root):
    pooled = read_csv(os.path.join(root, "datasets", "_canonical_manifest.csv"))
    la = read_csv(os.path.join(root, "datasets", "pose", "li_taati", "processed", "canonical_manifest.csv"))
    fps_blank_other = pooled[(pooled.fps == "") & (~pooled.source.isin(["modys", "li_taati", "remap_open"]))]
    return pooled, la, fps_blank_other.source.value_counts().to_dict()


def scan_text_for_rate(text):
    return sorted(set(m.group(0).lower() for m in RATE_RE.finditer(text)))


CANDIDATE_RATES = [24, 25, 30, 50, 60]


def whole_second_counts(frame_counts):
    """For each candidate integer rate r: how many clips span a whole number of seconds,
    counting the n - 1 intervals between n frames, and counting n frames; with the count
    expected by chance (len / r). A source that cut its clips at whole seconds shows a
    count far above chance at its own rate (and at the rates dividing it)."""
    n = np.asarray(list(frame_counts), dtype=int)
    return {str(r): {"n_minus_1_divisible": int(np.sum((n - 1) % r == 0)), "n_divisible": int(np.sum(n % r == 0)),
                     "chance": round(len(n) / r, 2)} for r in CANDIDATE_RATES}


# ----------------------------------------------------------------------------- modys_video
def probe_modys(root, pooled, facts):
    log("modys_video: reading coordinate zips, movie zips and README")
    raw = os.path.join(root, "datasets", "pose", "modys_video", "raw")
    man = pooled[pooled.source == "modys"].copy()
    facts["modys_manifest_records"] = len(man)
    facts["modys_manifest_fps_values"] = dict(collections.Counter(man.fps))

    # coordinate files: data rows per file (3 header rows in DeepLabCut CSVs)
    coords = {}
    for zname in ["data_stickfigure_coordinates_lying.zip", "data_stickfigure_coordinates_sitting.zip"]:
        zf = zipfile.ZipFile(os.path.join(raw, zname))
        for info in zf.infolist():
            if info.is_dir() or os.path.basename(info.filename).startswith("._"):
                continue
            lines = [ln for ln in zf.open(info).read().decode("utf-8").splitlines() if ln.strip()]
            name = os.path.basename(info.filename)
            stem = re.sub(r"\s", "", name)[:-4]
            num = stem[:3]
            coords[(zname, stem)] = {"zip": zname, "file": name, "stem": stem, "video": num,
                                     "rows": len(lines) - 3, "header0": lines[0].split(",")[0],
                                     "zip_date": "%04d-%02d-%02d" % info.date_time[:3]}
    facts["modys_coordinate_files"] = dict(collections.Counter(v["zip"] for v in coords.values()))

    # movies: container facts, parsed in memory
    movies = []
    pair = {"lying_stickfigures_movies.zip": "data_stickfigure_coordinates_lying.zip",
            "sitting_stickfigure_movies.zip": "data_stickfigure_coordinates_sitting.zip"}
    for mz, cz in pair.items():
        zf = zipfile.ZipFile(os.path.join(raw, mz))
        for info in zf.infolist():
            if info.is_dir() or os.path.basename(info.filename).startswith("._"):
                continue
            buf = zf.open(info).read()
            c = parse_mp4(buf)
            name = os.path.basename(info.filename)
            stem = re.sub(r"\s", "", name)
            stem = stem[:-8] if stem.endswith(".csv.mp4") else stem[:-4]
            num = stem[:3]
            deltas = sorted(set(d for _, d in c.get("stts", [])))
            ts = c.get("mdhd_timescale")
            fps_movie = ts / deltas[0] if ts and len(deltas) == 1 else float("nan")
            # the coordinate file of the same video in the paired zip: same stem, else same number
            same = coords.get((cz, stem))
            match = "same_name"
            if same is None:
                cands = [v for (z, s), v in coords.items() if z == cz and v["video"] == num]
                same = cands[0] if len(cands) == 1 else None
                match = "same_video_number" if same else "none"
            rows = same["rows"] if same else None
            n = c.get("stsz_count")
            factor = (n - 1) / (rows - 1) if (rows and n and rows > 1) else float("nan")
            movies.append({"movie_zip": mz, "movie_file": name, "video": num, "codec": c.get("codec"),
                           "width": c.get("width"), "height": c.get("height"), "tracks": c["tracks"],
                           "timescale": ts, "sample_deltas": ";".join(str(d) for d in deltas),
                           "movie_samples": n, "movie_duration_s": (c.get("mdhd_duration") or 0) / ts if ts else None,
                           "movie_fps": fps_movie, "coord_zip": cz, "coord_file": same["file"] if same else "",
                           "coord_match": match, "coord_rows": rows, "coord_zip_date": same["zip_date"] if same else "",
                           "movie_zip_date": "%04d-%02d-%02d" % info.date_time[:3],
                           "movie_frames_per_pose_frame": factor})
    mv = pd.DataFrame(movies)
    mv["factor_is_integer"] = mv.movie_frames_per_pose_frame.apply(lambda f: bool(np.isfinite(f) and abs(f - round(f)) < 1e-9))
    facts["modys_movies"] = len(mv)
    facts["modys_movie_fps_values"] = {str(k): int(v) for k, v in mv.movie_fps.value_counts().items()}
    facts["modys_movie_codecs"] = dict(collections.Counter(mv.codec))
    facts["modys_movie_sizes"] = dict(collections.Counter(f"{w}x{h}" for w, h in zip(mv.width, mv.height)))
    facts["modys_movie_coord_match"] = dict(collections.Counter(mv.coord_match))
    diff = mv[mv.coord_match == "same_video_number"]
    facts["modys_movies_named_after_another_model_run"] = {
        "count": int(len(diff)),
        "movie_name_tags": dict(collections.Counter(re.sub(r"\s", "", m)[4:].split("shuffle")[0] for m in diff.movie_file)),
        "csv_name_tags": dict(collections.Counter(re.sub(r"\s", "", c)[4:].split("shuffle")[0] for c in diff.coord_file)),
        "integer_factor": int(diff.factor_is_integer.sum()) if "factor_is_integer" in diff else None}
    ints = mv[mv.factor_is_integer]
    facts["modys_movies_with_integer_factor"] = int(len(ints))
    facts["modys_integer_factor_values"] = {str(int(round(k))): int(v) for k, v in ints.movie_frames_per_pose_frame.value_counts().items()}
    non = mv[~mv.factor_is_integer & mv.coord_rows.notna()]
    facts["modys_movies_non_integer_factor"] = [
        {"movie": r.movie_file, "movie_samples": int(r.movie_samples), "coord_rows": int(r.coord_rows),
         "coord_zip_date": r.coord_zip_date, "movie_zip_date": r.movie_zip_date,
         "pose_frames_implied_by_movie_at_modal_factor": None} for r in non.itertuples()]
    modal = collections.Counter(int(round(f)) for f in ints.movie_frames_per_pose_frame).most_common(1)[0][0]
    for d in facts["modys_movies_non_integer_factor"]:
        d["pose_frames_implied_by_movie_at_modal_factor"] = (d["movie_samples"] - 1) / modal + 1
    facts["modys_modal_factor"] = modal
    movie_fps_modal = float(mv.movie_fps.mode().iloc[0])
    source_fps = movie_fps_modal / modal
    facts["modys_source_level_fps"] = source_fps
    facts["modys_coord_zip_dates"] = dict(collections.Counter(v["zip_date"] for v in coords.values()))

    # README statement of which zip holds which file family, against the zips themselves
    readme = open(os.path.join(raw, "README.txt"), encoding="utf-8", errors="replace").read()
    stated = re.findall(r"(\d{3}_\S+?\.csv) \(in zipfile (\S+?)\)", readme)
    readme_check = []
    for fname, zstated in stated:
        actual = [z for (z, s), v in coords.items() if v["file"] == fname]
        readme_check.append({"file": fname, "readme_zip": zstated, "actual_zip": actual})
    facts["modys_readme_zip_statements"] = readme_check
    facts["modys_readme_rate_terms"] = scan_text_for_rate(readme)
    facts["modys_readme_code_url"] = (re.findall(r"https?://github\.com/\S+", readme) or [""])[0]
    xl_terms = []
    for x in ["README_data_clinical_scoring.xlsx"]:
        t = pd.read_excel(os.path.join(raw, x), header=None, sheet_name=None)
        xl_terms += scan_text_for_rate(" ".join(" ".join(map(str, df.values.ravel())) for df in t.values()))
    facts["modys_variable_dictionary_rate_terms"] = xl_terms

    # records
    rows = []
    lying = mv[mv.movie_zip == "lying_stickfigures_movies.zip"].set_index("video")
    for r in man.itertuples():
        stem = r.take
        cz = [z for (z, s) in coords if s == stem]
        crow = coords[(cz[0], stem)] if len(cz) == 1 else None
        num = stem[:3]
        mzip = "lying_stickfigures_movies.zip" if cz and cz[0] == "data_stickfigure_coordinates_lying.zip" else "sitting_stickfigure_movies.zip"
        mrow = mv[(mv.movie_zip == mzip) & (mv.video == num)]
        mrow = mrow.iloc[0] if len(mrow) == 1 else None
        factor = mrow.movie_frames_per_pose_frame if mrow is not None else float("nan")
        if mrow is not None and mrow.factor_is_integer and int(round(factor)) == modal:
            fps, ev = mrow.movie_fps / factor, "measured_container"
        elif mrow is not None:
            fps, ev = source_fps, "measured_container_source_level"
        else:
            fps, ev = "", "unresolved"
        rows.append({"source": "modys_video", "subject_ns": r.subject_ns, "take": r.take, "fps": fps,
                     "evidence": ev, "manifest_source": r.source, "manifest_fps": r.fps, "dim": r.dim,
                     "n_frames": r.n_frames, "raw_frames": crow["rows"] if crow else "",
                     "raw_file": f"{cz[0]}/{crow['file']}" if crow else "",
                     "check_name": "movie_frames_per_pose_frame (movie samples - 1) / (coordinate rows - 1)",
                     "check_value": round(float(factor), 4) if np.isfinite(factor) else ""})
    rec = pd.DataFrame(rows)
    facts["modys_records_by_evidence"] = dict(collections.Counter(rec.evidence))
    facts["modys_records_coord_zip"] = dict(collections.Counter(r.split("/")[0] for r in rec.raw_file))
    facts["modys_records_nframes_equal_raw_rows"] = int((rec.n_frames.astype(str) == rec.raw_frames.astype(str)).sum())
    nf = rec.n_frames.astype(int)
    facts["modys_record_n_frames"] = {str(k): int(v) for k, v in nf.value_counts().sort_index().items()}
    facts["modys_record_interval_s_at_source_fps"] = {str(round((k - 1) / source_fps, 2)): int(v) for k, v in nf.value_counts().sort_index().items()}
    facts["modys_whole_second_records"] = whole_second_counts(nf)
    facts["modys_whole_second_all_coordinate_files"] = whole_second_counts(v["rows"] for v in coords.values())
    facts["modys_coordinate_files_count"] = len(coords)
    return rec, mv


# ----------------------------------------------------------------------------- li_taati
def probe_li_taati(root, pooled, la, facts):
    log("li_taati: scanning trajectory exports and documentation for any timing field")
    raw = os.path.join(root, "datasets", "pose", "li_taati", "raw", "UDysRS_UPDRS_Export")
    man = pd.concat([pooled[pooled.source == "li_taati"], la], ignore_index=True)
    facts["li_taati_manifest_records"] = {"pooled": int((pooled.source == "li_taati").sum()), "leg_agility_per_source": len(la)}
    facts["li_taati_manifest_fps_values"] = dict(collections.Counter(man.fps))
    facts["li_taati_subject_ids"] = {"pooled": int(pooled[pooled.source == "li_taati"].subject_ns.nunique()),
                                     "leg_agility_per_source": int(la.subject_ns.nunique())}
    keysets = {}
    timing_keys = []
    for f in ["Communication_all_export.txt", "Drinking_all_export.txt", "LA_split_all_export.txt", "TT_opt_flow_export.txt"]:
        j = json.load(open(os.path.join(raw, f), encoding="utf-8"))
        top = collections.Counter()
        for k, v in j.items():
            if isinstance(v, dict):
                top.update(v.keys())
                for kk, vv in v.items():
                    if re.search(r"(?i)time|fps|rate|stamp|hz|sec", kk):
                        timing_keys.append(f"{f}:{k}:{kk}")
                    if isinstance(vv, dict):
                        for k3 in vv:
                            if re.search(r"(?i)time|fps|rate|stamp|hz|sec", k3):
                                timing_keys.append(f"{f}:{k}:{kk}:{k3}")
        keysets[f] = {"records": len(j), "record_level_keys": dict(top)}
    facts["li_taati_export_keys"] = keysets
    facts["li_taati_timing_like_keys"] = timing_keys[:20]
    facts["li_taati_timing_like_key_count"] = len(timing_keys)
    nb = json.load(open(os.path.join(raw, "data_import_demo.ipynb"), encoding="utf-8"))
    text = []
    for c in nb.get("cells", []):
        text.append("".join(c.get("source", [])))
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain")
            if t:
                text.append("".join(t))
    facts["li_taati_notebook_rate_terms"] = scan_text_for_rate("\n".join(text))
    facts["li_taati_licence_rate_terms"] = scan_text_for_rate(open(os.path.join(raw, "LICENSE.txt"), encoding="utf-8").read())
    other = sorted(os.listdir(raw))
    facts["li_taati_raw_files"] = [x for x in other if not x.startswith("._")]
    cit = read_csv(os.path.join(root, "docs", "citations.csv"))
    paper = cit[(cit.iloc[:, 0] == "li_taati") & (cit.iloc[:, 1] == "paper")]
    facts["li_taati_data_paper"] = paper.iloc[0, 2] if len(paper) else ""
    demo = read_csv(os.path.join(root, "docs", "demographics_curated.csv"))
    drow = demo[demo.iloc[:, 0] == "li_taati"]
    facts["li_taati_europe_pmc_ids_in_demographics_curated"] = sorted(set(re.findall(r"PMC\d+", " ".join(" ".join(r) for r in drow.values.tolist()))))
    facts["li_taati_whole_second_records"] = whole_second_counts(man.n_frames.astype(int))
    facts["li_taati_record_count"] = len(man)
    rows = []
    for r in man.itertuples():
        rows.append({"source": "li_taati", "subject_ns": r.subject_ns, "take": r.take, "fps": "",
                     "evidence": "unresolved", "manifest_source": r.source, "manifest_fps": r.fps, "dim": r.dim,
                     "n_frames": r.n_frames, "raw_frames": "", "raw_file": "",
                     "check_name": "", "check_value": ""})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- remap_open
def probe_remap(root, pooled, facts):
    log("remap_open: reading readme, turning keypoints, turning labels and sit-to-stand files inside the zip")
    zpath = os.path.join(root, "datasets", "pose", "remap_open", "raw", "remap_open.zip")
    zf = zipfile.ZipFile(zpath)
    names = [n for n in zf.namelist() if not os.path.basename(n).startswith("._")]
    pre = names[0].split("/")[0] + "/"
    readme = zf.open(pre + "readme.txt").read().decode("utf-8", errors="replace")
    m = re.search(r"captured at (\d+(?:\.\d+)?) frames per second", readme)
    stated = float(m.group(1)) if m else None
    sent = [" ".join(s.split()) for s in re.split(r"(?<=\.)\s", readme) if "frames per second" in s or "uneven video timings" in s]
    facts["remap_readme_stated_fps"] = stated
    facts["remap_readme_rate_sentences"] = sent
    facts["remap_readme_rate_terms"] = scan_text_for_rate(readme)
    facts["remap_readme_sts_time_column_sentences"] = [" ".join(s.split()) for s in re.split(r"(?<=\.)\s", readme) if "starting timestamp" in s]

    man = pooled[pooled.source == "remap_open"].copy()
    facts["remap_manifest_records"] = len(man)
    facts["remap_manifest_fps_by_task"] = {f"{t}|fps={f or 'blank'}": int(n) for (t, f), n in
                                           man.groupby([man["take"].str.split("_").str[0], "fps"]).size().items()}

    # turning keypoint files
    kp = {}
    turn_pat = re.compile(r"Turning_coarsen_CSV/(Pt\d+)_(PD|C)_n_(\d+)/input_(2D|3D)/(keypoints|keypoints3d)\.csv$")
    tnames = [n for n in names if turn_pat.search(n)]
    log(f"  turning keypoint files: {len(tnames)}")
    for i, n in enumerate(tnames):
        pt, grp, tid, d, _ = turn_pat.search(n).groups()
        lines = [ln for ln in zf.open(n).read().decode("utf-8").splitlines() if ln.strip()]
        ncol = len(lines[0].split(",")) if lines else 0
        kp[(pt, int(tid), d)] = {"rows": len(lines), "cols": ncol, "group": grp}
        if (i + 1) % 1000 == 0:
            log(f"  {i + 1}/{len(tnames)}")
    facts["remap_turning_keypoint_files"] = len(kp)
    facts["remap_turning_columns"] = dict(collections.Counter(f"{d}:{v['cols']}" for (_, _, d), v in kp.items()))
    same_rows = sum(1 for (pt, tid, d), v in kp.items() if d == "2D" and (pt, tid, "3D") in kp and kp[(pt, tid, "3D")]["rows"] == v["rows"])
    facts["remap_turning_episodes_2d_rows_equal_3d_rows"] = same_rows

    # labels
    lab_name = [n for n in names if n.endswith("Turning_human_labels.xlsx")][0]
    lab = pd.read_excel(io.BytesIO(zf.open(lab_name).read()))
    lab = lab.rename(columns={"Turn ID": "turn_id", "Participant ID number": "participant"})
    facts["remap_turning_label_rows"] = len(lab)
    dur = dict(zip(lab.turn_id.astype(int), lab.turning_duration.astype(float)))
    clin = dict(zip(lab.turn_id.astype(int), lab.clinical_assessment.astype(str)))
    grp = dict(zip(lab.turn_id.astype(int), lab.PD_or_C.astype(str)))

    turn = man[man["take"].str.startswith("turn_")].copy()
    turn["tid"] = turn["take"].str.extract(r"turn_n_(\d+)_")[0].astype(int)
    turn["pt"] = turn.subject_ns.str.split(":").str[1]
    turn["d"] = turn["take"].str[-2:].str.upper()
    turn["raw_rows"] = [kp.get((p, t, d), {}).get("rows") for p, t, d in zip(turn.pt, turn.tid, turn.d)]
    facts["remap_turning_records_with_raw_file"] = int(turn.raw_rows.notna().sum())
    facts["remap_turning_records_nframes_equal_raw_rows"] = int((turn.raw_rows.astype("Int64").astype(str) == turn.n_frames).sum())
    turn["dur"] = turn.tid.map(dur)
    facts["remap_turning_records_with_label_duration"] = int(turn.dur.notna().sum())

    # fit n_frames = offset + rate * labelled duration, one point per episode (2D rows)
    ep = turn[turn.d == "2D"].dropna(subset=["dur"]).copy()
    ep["nf"] = ep.n_frames.astype(int)
    ep["clin"] = ep.tid.map(clin)
    ep["grp"] = ep.tid.map(grp)
    fits = []

    def fit(df, name, level):
        if len(df) < 3:
            return None
        ts = theilslopes(df.nf.values, df.dur.values)
        A = np.vstack([np.ones(len(df)), df.dur.values]).T
        (a, b), *_ = np.linalg.lstsq(A, df.nf.values.astype(float), rcond=None)
        res = df.nf.values - (a + b * df.dur.values)
        fits.append({"level": level, "group": name, "episodes": len(df), "theil_sen_rate": ts[0],
                     "theil_sen_rate_lo95": ts[2], "theil_sen_rate_hi95": ts[3], "theil_sen_offset_frames": ts[1],
                     "ols_rate": b, "ols_offset_frames": a, "ols_residual_sd_frames": float(np.std(res, ddof=2)),
                     "ols_residual_abs_max_frames": float(np.max(np.abs(res))),
                     "share_abs_residual_le_1_frame": float(np.mean(np.abs(res) <= 1.0))})
        return ts, a, b

    ts_all, a_all, b_all = fit(ep, "all", "all")
    for k, g in ep.groupby("clin"):
        fit(g, f"clinical_assessment={k}", "subgroup")
    for k, g in ep.groupby("grp"):
        fit(g, f"PD_or_C={k}", "subgroup")
    for k, g in ep.groupby("subject_ns"):
        fit(g, k, "subject")
    fits = pd.DataFrame(fits)
    subj = fits[fits.level == "subject"]
    facts["remap_turning_fit_all"] = fits[fits.level == "all"].iloc[0].to_dict()
    facts["remap_turning_fit_subjects"] = {"subjects": len(subj), "subjects_with_turning_records": int(ep.subject_ns.nunique()), "subjects_in_manifest": int(man.subject_ns.nunique()), "theil_sen_rate_min": float(subj.theil_sen_rate.min()),
                                           "theil_sen_rate_median": float(subj.theil_sen_rate.median()),
                                           "theil_sen_rate_max": float(subj.theil_sen_rate.max()),
                                           "subjects_ci_contains_stated": int(((subj.theil_sen_rate_lo95 <= stated) & (subj.theil_sen_rate_hi95 >= stated)).sum())}
    # residual per record against the stated rate with the fitted offset of all episodes
    off_at_stated = float(np.median(ep.nf - stated * ep.dur))
    facts["remap_turning_offset_frames_at_stated_rate_median"] = off_at_stated

    # sit-to-stand: the per-frame time column
    sts_pat = re.compile(r"STS_2D_skeletons_coarsened/(Pt\d+)_(PD|C)_n_(\d+)\.csv$")
    sts = {}
    for n in [n for n in names if sts_pat.search(n)]:
        pt, g, eid = sts_pat.search(n).groups()
        fr, tm, sig = [], [], 0
        for ln in zf.open(n).read().decode("utf-8").splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.split(",")
            fr.append(float(p[0]))
            tm.append(float(p[1]))
            digits = p[1].strip().lstrip("-").replace(".", "").lstrip("0")
            sig = max(sig, len(digits))
        fr, tm = np.array(fr), np.array(tm)
        dt = np.diff(tm)
        dev = float(np.max(np.abs(tm - fr / stated)))
        sts[(pt, int(eid))] = {"group": g, "rows": len(fr), "frames_consecutive": bool(np.all(np.diff(fr) == 1)),
                               "time_significant_digits": sig, "dt_median": float(np.median(dt)) if dt.size else None,
                               "dt_min": float(dt.min()) if dt.size else None, "dt_max": float(dt.max()) if dt.size else None,
                               "span_rate": float((fr[-1] - fr[0]) / (tm[-1] - tm[0])) if tm[-1] > tm[0] else None,
                               "max_abs_time_minus_frame_over_stated": dev}
    facts["remap_sts_files"] = len(sts)
    dev = np.array([v["max_abs_time_minus_frame_over_stated"] for v in sts.values()])
    facts["remap_sts_time_significant_digits"] = dict(collections.Counter(str(v["time_significant_digits"]) for v in sts.values()))
    facts["remap_sts_files_max_deviation_below_1e-4_s"] = int(np.sum(dev < 1e-4))
    facts["remap_sts_max_abs_time_minus_frame_over_stated_s"] = float(dev.max())
    facts["remap_sts_max_deviation_in_frame_periods"] = float(dev.max() * stated)
    spans = np.array([v["span_rate"] for v in sts.values() if v["span_rate"]])
    facts["remap_sts_span_rate_range"] = [float(spans.min()), float(spans.max())]
    facts["remap_sts_files_frames_consecutive"] = int(sum(v["frames_consecutive"] for v in sts.values()))
    facts["remap_sts_dt_range_s"] = [float(min(v["dt_min"] for v in sts.values())), float(max(v["dt_max"] for v in sts.values()))]

    rows = []
    for r in turn.itertuples():
        resid = (int(r.n_frames) - (off_at_stated + stated * r.dur)) if pd.notna(r.dur) else None
        rows.append({"source": "remap_open", "subject_ns": r.subject_ns, "take": r.take, "fps": stated,
                     "evidence": "stated_by_source", "manifest_source": r.source, "manifest_fps": r.fps, "dim": r.dim,
                     "n_frames": r.n_frames, "raw_frames": "" if pd.isna(r.raw_rows) else int(r.raw_rows),
                     "raw_file": f"remap_open.zip:Turning_coarsen_CSV/{r.pt}_{grp.get(r.tid, '?')}_n_{r.tid}/input_{r.d}",
                     "check_name": "n_frames minus (stated fps x labelled turn duration + median offset), frames",
                     "check_value": round(resid, 3) if resid is not None else ""})
    stsr = man[man["take"].str.startswith("sts_")]
    for r in stsr.itertuples():
        pt = r.subject_ns.split(":")[1]
        eid = int(r.take.split("_")[-1])
        v = sts.get((pt, eid))
        rows.append({"source": "remap_open", "subject_ns": r.subject_ns, "take": r.take,
                     "fps": (stated if v["max_abs_time_minus_frame_over_stated"] < 1e-4 else v["span_rate"]) if v else "",
                     "evidence": "stated_in_file" if v else "unresolved",
                     "manifest_source": r.source, "manifest_fps": r.fps, "dim": r.dim, "n_frames": r.n_frames,
                     "raw_frames": v["rows"] if v else "", "raw_file": f"remap_open.zip:STS_2D_skeletons_coarsened/{pt}_{v['group'] if v else '?'}_n_{eid}.csv",
                     "check_name": "max |time - frame / stated fps|, s",
                     "check_value": v["max_abs_time_minus_frame_over_stated"] if v else ""})
    rec = pd.DataFrame(rows)
    rec["fps"] = rec.fps.apply(lambda x: round(x, 4) if isinstance(x, float) else x)
    facts["remap_records_by_evidence"] = {f"{t}|{e}": int(n) for (t, e), n in
                                          rec.groupby([rec["take"].str.split("_").str[0], "evidence"]).size().items()}
    return rec, fits


# ----------------------------------------------------------------------------- documents
def probe_documents(root, facts):
    out = {}
    reg = read_csv(os.path.join(root, "docs", "datasets_index.csv"))
    for ds in ["modys_video", "li_taati", "remap_open"]:
        dc = open(os.path.join(root, "datasets", "pose", ds, "datacard.md"), encoding="utf-8").read()
        row = reg[reg.id == ds]
        regtext = " ".join(row.iloc[0].tolist()) if len(row) else ""
        out[ds] = {"datacard_rate_terms": scan_text_for_rate(dc), "registry_rate_terms": scan_text_for_rate(regtext),
                   "registry_rate_sentence": [s.strip() for s in re.split(r"(?<=[.;])\s", regtext) if RATE_RE.search(s)]}
    facts["documents"] = out


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return str(x)


def write_summary(a, facts, bysrc):
    f = facts
    L = []
    L.append("# Frame rates of modys_video, li_taati and remap_open (T0.7a)")
    L.append("")
    L.append("Measured by scripts/probe_frame_rates.py against the MoveAtlas tree (commit in _provenance_probe_frame_rates.json).")
    L.append("Every number below is in frame_rates_facts.json or in the CSVs of this directory; nothing is typed.")
    L.append("")
    L.append("## Result per source")
    L.append("")
    L.append("| source | record group | records | manifest fps | recovered fps | evidence |")
    L.append("|---|---|---|---|---|---|")
    for r in bysrc.itertuples():
        L.append(f"| {r.source} | {r.group} | {r.records} | {r.manifest_fps} | {r.fps} | {r.evidence} |")
    L.append("")
    L.append("Evidence levels: measured_container (rate read from a movie container that holds the same pose frames),")
    L.append("measured_container_source_level (same measurement, but this record's movie was rendered from another version")
    L.append("of its coordinate file, so the source-level value is used), stated_by_source (stated in the source's")
    L.append("documentation on disk), stated_in_file (a time column in the data file, generated from a nominal rate),")
    L.append("unresolved (no file on disk states or measures it).")
    L.append("")
    # modys
    fa = f["modys_modal_factor"]
    L.append("## modys_video")
    L.append("")
    L.append(f"- Manifest: {f['modys_manifest_records']} records under the source label modys, fps values {f['modys_manifest_fps_values']} (blank).")
    L.append(f"- The coordinate CSVs carry no time column (DeepLabCut output: frame index, then x, y, likelihood per point); README.txt and the variable dictionary state no rate (rate terms found: {f['modys_readme_rate_terms']} and {f['modys_variable_dictionary_rate_terms']}).")
    L.append(f"- The source's stick-figure movies ({f['modys_movies']} files in two zips) were parsed in memory: codec {f['modys_movie_codecs']}, size {f['modys_movie_sizes']}, movie rate {f['modys_movie_fps_values']} (timescale over the single sample delta).")
    L.append(f"- Frame arithmetic: in {f['modys_movies_with_integer_factor']} of {f['modys_movies']} movies, (movie samples - 1) / (coordinate rows - 1) is an integer, with values {f['modys_integer_factor_values']}: each interval between two pose frames spans {fa} movie frames. The pose data therefore play at {fmt(f['modys_source_level_fps'])} pose frames per second of movie time.")
    L.append(f"- Movies matched to their coordinate file by name or by video number: {f['modys_movie_coord_match']}. Movies whose count does not fit their current CSV: {len(f['modys_movies_non_integer_factor'])}:")
    for d in f["modys_movies_non_integer_factor"]:
        L.append(f"  - {d['movie']}: {d['movie_samples']} movie samples against {d['coord_rows']} coordinate rows (CSV zip date {d['coord_zip_date']}, movie zip date {d['movie_zip_date']}); at the modal factor the movie holds {fmt(d['pose_frames_implied_by_movie_at_modal_factor'])} pose frames, so it was rendered from another version of the CSV.")
    mr = f["modys_movies_named_after_another_model_run"]
    L.append(f"- {mr['count']} movies carry another DeepLabCut model tag in their name than their coordinate file (movie tags {mr['movie_name_tags']}, CSV tags {mr['csv_name_tags']}); their frame arithmetic still gives an integer factor in {mr['integer_factor']} of them, so the timing holds even if the rendered points came from another model run.")
    L.append(f"- Records: {f['modys_records_by_evidence']}; manifest n_frames equals the CSV row count in {f['modys_records_nframes_equal_raw_rows']} of {f['modys_manifest_records']}. Span of each clip, (n_frames - 1) / fps at the recovered rate (seconds: records): {f['modys_record_interval_s_at_source_fps']}.")
    ws, wa = f["modys_whole_second_records"], f["modys_whole_second_all_coordinate_files"]
    L.append(f"- Corroboration (inferred, independent of the movies): the clips were cut at whole seconds at this rate. Records whose n_frames - 1 is a multiple of the candidate rate (count, chance count): " + ", ".join(f"{r} fps {v['n_minus_1_divisible']} ({v['chance']})" for r, v in ws.items()) + f"; over all {f['modys_coordinate_files_count']} coordinate files of both zips: " + ", ".join(f"{r} fps {v['n_minus_1_divisible']} ({v['chance']})" for r, v in wa.items()) + ". A rate of 30 fps is excluded; 50 fps fits the modal 501-frame clip but fewer of the others, and the movie arithmetic gives the rate directly.")
    L.append(f"- The manifest takes (19DLC ... ESMAC files) sit in {f['modys_records_coord_zip']}. README.txt says otherwise: {f['modys_readme_zip_statements']}. Either the zip names or the README are swapped; the posture of the 94 canonical records is therefore not settled by the file names, which matters for the DIS region mapping (arms and trunk scored sitting, legs lying) and not for the frame rate, since both zips give the same rate.")
    L.append(f"- What the measurement is: the rate at which the source's own renderer plays its pose data. It equals the capture rate if the renderer played in real time; the original camera videos are not published. The rate was not checked against the modys-video code named in README.txt ({f['modys_readme_code_url']}), which is not on disk.")
    L.append("")
    # li_taati
    L.append("## li_taati")
    L.append("")
    L.append(f"- Manifest: {f['li_taati_manifest_records']} records, fps values {f['li_taati_manifest_fps_values']} (blank); subject ids {f['li_taati_subject_ids']}.")
    L.append(f"- Raw files: {f['li_taati_raw_files']}. The trajectory exports hold per record only these keys: " + "; ".join(f"{k}: {v['record_level_keys']}" for k, v in f["li_taati_export_keys"].items()) + f". Keys that look like timing fields: {f['li_taati_timing_like_key_count']}. The leg agility idx field is a frame index, not a time.")
    L.append(f"- The demo notebook (sources and text outputs) and the licence state no rate (terms found: {f['li_taati_notebook_rate_terms']}, {f['li_taati_licence_rate_terms']}); the datacard and the registry row neither ({f['documents']['li_taati']}).")
    wl = f["li_taati_whole_second_records"]
    L.append(f"- No whole-second structure to infer from: records whose n_frames - 1 is a multiple of the candidate rate (count, chance count over {f['li_taati_record_count']} records): " + ", ".join(f"{r} fps {v['n_minus_1_divisible']} ({v['chance']})" for r, v in wl.items()) + "; the task segments were not cut at whole seconds, so the frame counts carry no rate.")
    L.append(f"- Unresolved on disk. The document that settles it is the data paper, DOI {f['li_taati_data_paper']} (MoveAtlas docs/demographics_curated.csv took the li_taati demographics from the Europe PMC full text {', '.join(f['li_taati_europe_pmc_ids_in_demographics_curated'])}); reading it needs network access, which this run did not have. Until then every hertz-bearing readout on li_taati stays undefined, as the design says.")
    L.append("")
    # remap
    fall = f["remap_turning_fit_all"]
    fs = f["remap_turning_fit_subjects"]
    L.append("## remap_open")
    L.append("")
    L.append(f"- Manifest by task: {f['remap_manifest_fps_by_task']}.")
    L.append(f"- readme.txt (inside remap_open.zip) states the rate: " + " ".join(f'\"{s}\"' for s in f["remap_readme_rate_sentences"]) + f" Parsed value: {fmt(f['remap_readme_stated_fps'])}.")
    L.append(f"- Turning files: {f['remap_turning_keypoint_files']} keypoint CSVs, columns {f['remap_turning_columns']}, no time column; 2D and 3D row counts equal in {f['remap_turning_episodes_2d_rows_equal_3d_rows']} episodes; manifest n_frames equals the raw row count in {f['remap_turning_records_nframes_equal_raw_rows']} of {f['remap_turning_records_with_raw_file']} records.")
    L.append(f"- Corroboration from the rater labels ({f['remap_turning_label_rows']} rows of Turning_human_labels.xlsx, turning_duration in seconds; {f['remap_turning_records_with_label_duration']} records joined): regressing the clip frame count on the labelled turn duration over {fall['episodes']} episodes gives a Theil-Sen slope of {fmt(fall['theil_sen_rate'])} frames per second (95 percent interval {fmt(fall['theil_sen_rate_lo95'])} to {fmt(fall['theil_sen_rate_hi95'])}), OLS {fmt(fall['ols_rate'])}, with a constant offset of {fmt(fall['ols_offset_frames'], 2)} frames (OLS residual SD {fmt(fall['ols_residual_sd_frames'], 3)} frames, {fmt(100 * fall['share_abs_residual_le_1_frame'], 1)} percent of episodes within one frame). Per subject ({fs['subjects']} subjects with at least 3 episodes; {fs['subjects_with_turning_records']} of the {fs['subjects_in_manifest']} remap_open subjects have turning records) the Theil-Sen slope runs from {fmt(fs['theil_sen_rate_min'])} to {fmt(fs['theil_sen_rate_max'])} (median {fmt(fs['theil_sen_rate_median'])}); {fs['subjects_ci_contains_stated']} subject intervals contain the stated rate. Subgroup fits are in frame_rates_remap_turning_fit.csv. The clips are the labelled turn plus a fixed margin of about the fitted offset; the readme does not document the margin.")
    L.append(f"- Sit-to-stand: {f['remap_sts_files']} files, time printed with {f['remap_sts_time_significant_digits']} significant digits (digits: files). In {f['remap_sts_files_max_deviation_below_1e-4_s']} files every row satisfies |time - frame / {fmt(f['remap_readme_stated_fps'])}| < 1e-4 s; the largest deviation over all files is {f['remap_sts_max_abs_time_minus_frame_over_stated_s']:.2e} s, {f['remap_sts_max_deviation_in_frame_periods']:.4f} of a frame period (frame intervals from {f['remap_sts_dt_range_s'][0]:.4f} to {f['remap_sts_dt_range_s'][1]:.4f} s, which is the rounding of 1/{fmt(f['remap_readme_stated_fps'])} and nothing else; rate over the whole span {fmt(f['remap_sts_span_rate_range'][0])} to {fmt(f['remap_sts_span_rate_range'][1])}), frame numbers consecutive in {f['remap_sts_files_frames_consecutive']}. The readme describes the column as a time offset from the file's starting timestamp (" + " ".join(f'\"{s}\"' for s in f["remap_readme_sts_time_column_sentences"]) + f"), but the values are generated from the nominal rate, not read from a clock: the manifest's 30.0 for these records is the stated rate, not a measured one (the registry row calls it measured: {[s for s in f['documents']['remap_open']['registry_rate_sentence'] if 'fps' in s]}).")
    L.append("- The rate is nominal. The readme itself warns that Kinect frames were aggregated into the video by timestamp with possibly uneven timing; no per-frame clock survives in the open subset, so frame-interval jitter cannot be measured here.")
    L.append("")
    L.append("## Blank-fps records outside this task")
    L.append("")
    L.append(f"- Pooled pose manifest, other sources with blank fps: {f['blank_fps_other_sources']} (not in the scope of T0.7a).")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append("- frame_rates.csv: one row per record (source, subject_ns, take, fps, evidence, then manifest fields, the raw file and a per-record check).")
    L.append("- frame_rates_by_source.csv: the table above.")
    L.append(f"- frame_rates_modys_movies.csv: container facts of the {f['modys_movies']} stick-figure movies and their coordinate files.")
    L.append("- frame_rates_remap_turning_fit.csv: frame count against labelled duration, overall, per subgroup, per subject.")
    L.append("- frame_rates_facts.json: every number of this summary.")
    with open(os.path.join(a.out, "frame_rates_summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    a = parse_args("Frame rates of modys_video, li_taati and remap_open (T0.7a)")
    facts = {}
    pooled, la, other = load_manifests(a.moveatlas)
    facts["blank_fps_other_sources"] = other
    probe_documents(a.moveatlas, facts)
    modys, movies = probe_modys(a.moveatlas, pooled, facts)
    li = probe_li_taati(a.moveatlas, pooled, la, facts)
    remap, fits = probe_remap(a.moveatlas, pooled, facts)

    allrec = pd.concat([modys, li, remap], ignore_index=True)
    cols = ["source", "subject_ns", "take", "fps", "evidence", "manifest_source", "manifest_fps", "dim", "n_frames",
            "raw_frames", "raw_file", "check_name", "check_value"]
    allrec = allrec[cols]
    allrec.to_csv(os.path.join(a.out, "frame_rates.csv"), index=False, encoding="utf-8", lineterminator="\n")
    movies.to_csv(os.path.join(a.out, "frame_rates_modys_movies.csv"), index=False, encoding="utf-8", lineterminator="\n")
    fits.to_csv(os.path.join(a.out, "frame_rates_remap_turning_fit.csv"), index=False, encoding="utf-8", lineterminator="\n")

    allrec["group"] = np.where(allrec.source == "remap_open", allrec["take"].str.split("_").str[0],
                               np.where(allrec.source == "li_taati", allrec["take"].str.rsplit("_", n=1).str[0], "all"))
    by = (allrec.groupby(["source", "group", "evidence"], sort=False)
          .agg(records=("take", "size"),
               manifest_fps=("manifest_fps", lambda s: ";".join(sorted(set(x if x != "" else "blank" for x in s.astype(str))))),
               fps=("fps", lambda s: ";".join(sorted(set(fmt(x) if x != "" else "none" for x in s)))))
          .reset_index())[["source", "group", "records", "manifest_fps", "fps", "evidence"]]
    by.to_csv(os.path.join(a.out, "frame_rates_by_source.csv"), index=False, encoding="utf-8", lineterminator="\n")
    facts["records_total"] = len(allrec)
    facts["records_by_source_evidence"] = {f"{r.source}|{r.group}|{r.evidence}": int(r.records) for r in by.itertuples()}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    with open(os.path.join(a.out, "frame_rates_facts.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(clean(facts), fh, indent=1, ensure_ascii=False)
    write_summary(a, facts, by)
    write_provenance(a, {"task": "T0.7a", "outputs": ["frame_rates.csv", "frame_rates_by_source.csv",
                                                      "frame_rates_modys_movies.csv", "frame_rates_remap_turning_fit.csv",
                                                      "frame_rates_facts.json", "frame_rates_summary.md"]})
    shutil.copyfile(os.path.join(a.out, "_provenance.json"), os.path.join(a.out, "_provenance_probe_frame_rates.json"))
    log(by.to_string(index=False))
    log("records:", len(allrec), "->", a.out)


if __name__ == "__main__":
    main()
