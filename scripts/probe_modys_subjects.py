"""modys_video: persons versus videos in the MoveAtlas canonical manifest.

The MoveAtlas canonical manifest of modys_video (datasets/pose/modys_video/processed/
canonical_manifest.csv) sets subject_ns to the video number (modys:001 ... modys:095).
The source (Zenodo 5638470, raw/README.txt) says each child has several videos: lying and
sitting share a video number, and the time points T0, T3 and T12 have different video
numbers. raw/data_clinical_scoring.xlsx carries the column ID (the child) beside video and
time. This script joins the manifest to that table and writes the person map that every
MD_FM split must use for this source.

Reads (read-only): MoveAtlas datasets/pose/modys_video/processed/canonical_manifest.csv,
raw/data_clinical_scoring.xlsx, raw/README_data_clinical_scoring.xlsx.
Writes: reports/modys_subjects/modys_video_person_map.csv (one row per manifest record),
reports/modys_subjects/modys_subjects_summary.md, reports/modys_subjects/_provenance.json.

Run: python scripts/probe_modys_subjects.py --moveatlas <moveatlas>
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
args = ap.parse_args()
src = os.path.join(args.moveatlas, "datasets", "pose", "modys_video")
out = os.path.join(here, "reports", "modys_subjects")
os.makedirs(out, exist_ok=True)

man = pd.read_csv(os.path.join(src, "processed", "canonical_manifest.csv"), encoding="utf-8")
score = pd.read_excel(os.path.join(src, "raw", "data_clinical_scoring.xlsx"))
dictionary = pd.read_excel(os.path.join(src, "raw", "README_data_clinical_scoring.xlsx"))

man["video"] = man["take"].str.slice(0, 3).astype(int)
vid = score.drop_duplicates("video")[["video", "ID", "time"]]
conflicts = int((score.groupby("video")["ID"].nunique() > 1).sum())
m = man.merge(vid, on="video", how="left")
m["person_ns"] = "modys:child" + m["ID"].astype("Int64").astype(str).str.zfill(2)
m[["subject_ns", "take", "video", "ID", "time", "person_ns", "n_frames"]].to_csv(
    os.path.join(out, "modys_video_person_map.csv"), index=False, encoding="utf-8", lineterminator="\n")

n = {
    "manifest_records": len(man),
    "manifest_distinct_subject_ns": int(man["subject_ns"].nunique()),
    "subject_ns_is_video_number": bool((man["subject_ns"] == "modys:" + man["take"].str.slice(0, 3)).all()),
    "scoring_rows": len(score),
    "scoring_distinct_children": int(score["ID"].nunique()),
    "scoring_distinct_videos": int(score["video"].nunique()),
    "videos_with_more_than_one_child": conflicts,
    "manifest_records_matched_to_a_child": int(m["ID"].notna().sum()),
    "manifest_distinct_children": int(m["ID"].nunique()),
    "time_points_per_video": vid["time"].value_counts().to_dict(),
    "videos_per_child": {int(k): int(v) for k, v in vid.groupby("ID").size().value_counts().sort_index().items()},
    "variable_dictionary_rows": len(dictionary),
    "scorers": sorted(int(s) for s in score["scorer"].unique()),
}
lines = [
    "# modys_video: persons versus videos",
    "",
    "Measured by scripts/probe_modys_subjects.py against the MoveAtlas tree (commit in _provenance.json).",
    "",
    "| quantity | value |",
    "|---|---|",
]
for k, v in n.items():
    lines.append(f"| {k} | {v} |")
lines += [
    "",
    "Reading: the manifest has one subject_ns per video number, while the source identifies",
    f"{n['scoring_distinct_children']} children, most of them filmed at three time points. A split on",
    "subject_ns puts the same child on both sides. MD_FM splits this source on person_ns from",
    "modys_video_person_map.csv. The registry figure of 63 matches the number of rows of the",
    "variable dictionary README_data_clinical_scoring.xlsx, not a participant count.",
]
open(os.path.join(out, "modys_subjects_summary.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

try:
    head = subprocess.run(["git", "-c", f"safe.directory={args.moveatlas}", "-C", args.moveatlas, "log", "-1",
                           "--format=%h"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:  # noqa: BLE001
    head = "unknown"
json.dump({"script": "scripts/probe_modys_subjects.py", "moveatlas": args.moveatlas, "moveatlas_head": head,
           "counts": n}, open(os.path.join(out, "_provenance.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(n, indent=1))
