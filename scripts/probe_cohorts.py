"""Cohort splits recoverable from subject_ns prefixes and the silhouette cohort column,
plus the within-subject ON/OFF pairing of pd_rest_eeg_onoff_ds002778 (read-only).
Output: <out>/harmonized_side_subjects.csv, stdout log.
Run: python scripts/probe_cohorts.py --moveatlas <moveatlas>
"""
import os, re
import pandas as pd
from _common import parse_args, write_provenance

ARGS = parse_args(__doc__.splitlines()[0])
REPO = ARGS.moveatlas
DOCS = REPO + "/docs"
DS = REPO + "/datasets"
OUT = ARGS.out
write_provenance(ARGS)

rd = lambda fn: pd.read_csv(os.path.join(DS, fn), dtype=str, keep_default_na=False, encoding="utf-8")

sil = rd("_canonical_silhouette_manifest.csv")
print("=== silhouette manifest: subjects / records per (source, cohort, severity)")
print(sil.groupby(["source", "cohort", "severity"]).agg(subjects=("subject_ns", "nunique"), records=("subject_ns", "size")).to_string())

force = rd("_canonical_force_manifest.csv")
imu = rd("_canonical_imu_manifest.csv")
pose = rd("_canonical_manifest.csv")
eeg = rd("_canonical_eeg_manifest.csv")
emg = rd("_canonical_emg_manifest.csv")

def prefix_table(df, source, pat, label):
    d = df[df["source"] == source].copy()
    d["grp"] = d["subject_ns"].str.extract(pat, expand=False)
    g = d.groupby("grp").agg(subjects=("subject_ns", "nunique"), records=("subject_ns", "size"))
    print(f"--- {label} ({source}): groups from subject_ns pattern {pat!r}")
    print(g.to_string())
    return g

print("=== force manifest")
prefix_table(force, "gaitndd", r"gaitndd:([a-z]+)", "gaitndd groups")
prefix_table(force, "physionet_gaitpdb", r"gaitpdb:([A-Za-z]{2})", "gaitpdb study prefix (Ga/Ju/Si)")
d = force[force["source"] == "physionet_gaitpdb"].copy(); d["grp"] = d["subject_ns"].str.extract(r"gaitpdb:[A-Za-z]{2}([A-Za-z]{2})", expand=False)
print(d.groupby("grp").agg(subjects=("subject_ns", "nunique"), records=("subject_ns", "size")).to_string())
prefix_table(force, "weargait_pd", r"weargait:([A-Za-z]+)", "weargait HC/PD")
prefix_table(force, "gaitrec", r"gaitrec:([A-Za-z_]+)", "gaitrec")
prefix_table(force, "stroke_gait_vancriekinge", r"stroke_gait:([a-z_]+)", "stroke_gait")
print("=== imu manifest")
prefix_table(imu, "weargait_pd", r"weargait:([A-Za-z]+)", "weargait HC/PD")
for s in ["pads", "clinical_gait_multipatho", "hd_accel_jnet", "kaggle_fog", "imu_inthewild_iprognosis", "monipar", "pd_biostamprc21", "capture24", "ltmm", "goodwin_smm", "fog_turning_filho", "fog_star", "daphnet", "multimodal_fog", "sensefog", "physionet_multimodal_gait"]:
    d = imu[imu["source"] == s]
    print(f"{s:28s} subjects={d['subject_ns'].nunique():5d} sample={sorted(d['subject_ns'].unique())[:4]}")
prefix_table(imu, "ltmm", r"ltmm:([A-Za-z]+)", "ltmm CO/FL")
prefix_table(imu, "clinical_gait_multipatho", r"clinical_gait_multipatho:([A-Za-z]+)", "clinical_gait_multipatho prefix")
prefix_table(imu, "kaggle_fog", r"kaggle_fog:([a-z]+)", "kaggle_fog subset")
prefix_table(imu, "imu_inthewild_iprognosis", r"iprognosis:([A-Za-z]+)", "iprognosis prefix")
print("=== pose manifest")
for s in ["care_pd", "li_taati", "cody", "modys", "remap_open", "stroke_gait_vancriekinge", "infant_pose_chambers", "gait120", "vidimu"]:
    d = pose[pose["source"] == s]
    print(f"{s:28s} subjects={d['subject_ns'].nunique():5d} sample={sorted(d['subject_ns'].unique())[:4]}")
prefix_table(pose, "care_pd", r"care_pd:([A-Za-z0-9_]+):", "care_pd cohorts")
prefix_table(pose, "stroke_gait_vancriekinge", r"stroke_gait:([a-z_]+)", "stroke_gait")
prefix_table(pose, "cody", r"^(adult:[CP])", "cody controls (C) vs patients (P)")
prefix_table(pose, "remap_open", r"remap_open:([A-Za-z]+)", "remap_open prefix")
prefix_table(pose, "infant_pose_chambers", r"chambers:([A-Za-z]+)", "chambers prefix")
print("=== eeg manifest")
for s in sorted(eeg["source"].unique()):
    d = eeg[eeg["source"] == s]
    print(f"{s:28s} subjects={d['subject_ns'].nunique():5d} sample subj={sorted(d['subject_ns'].unique())[:3]} sample task={sorted(d['task'].unique())[:4]}")
d = eeg[eeg["source"] == "pd_rest_eeg_onoff_ds002778"].copy()
d["ses"] = d["task"].str.extract(r"(ses-[a-z]+)", expand=False)
print("ds002778 records per session:", d["ses"].value_counts().to_dict())
per = d.groupby("subject_ns")["ses"].agg(lambda s: tuple(sorted(set(s))))
print("ds002778 subjects by session set:", per.value_counts().to_dict())
print("ds002778 subjects with BOTH ses-on and ses-off:", int(per.map(lambda t: "ses-on" in t and "ses-off" in t).sum()))
prefix_table(eeg, "pd_rest_eeg_ds004584", r"ds004584:(sub-[a-zA-Z]*)", "ds004584 prefix")
prefix_table(eeg, "pd_walking_eeg_ds007526", r"ds007526:(sub-[a-zA-Z]*)", "ds007526 prefix")
prefix_table(eeg, "cap_sleep", r"cap_sleep:([a-z]+)", "cap_sleep group prefix")
print("=== emg manifest")
for s in sorted(emg["source"].unique()):
    d = emg[emg["source"] == s]
    print(f"{s:28s} subjects={d['subject_ns'].nunique():5d} records={len(d):5d} kinds={sorted(d['kinds'].unique())[:5]}")

# registry n_subjects (datasets_index.csv) for the harmonized ids, per side
idx = pd.read_csv(os.path.join(DOCS, "datasets_index.csv"), dtype=str, keep_default_na=False, encoding="utf-8").set_index("id")
cells = pd.read_csv(os.path.join(DOCS, "registry_cells.csv"), dtype=str, keep_default_na=False, encoding="utf-8").set_index("id")
demo = pd.read_csv(os.path.join(DOCS, "demographics.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
print("=== registry n_subjects (datasets_index.csv) vs canonical subjects, harmonized ids, by side")
rows = []
for i in demo["dataset_id"]:
    raw = idx.loc[i, "n_subjects"]
    m = re.search(r"\d[\d,]*", raw)
    n = int(m.group(0).replace(",", "")) if m else None
    cs = cells.loc[i, "cells"].split("|")
    a = any(c.startswith("A") for c in cs); b = any(c.startswith("B") for c in cs)
    side = "both" if a and b else "hyper" if a else "hypo" if b else "neither"
    rows.append({"id": i, "side": side, "registry_n_subjects_raw": raw[:40], "registry_n_parsed": n,
                 "canonical_subjects": int(demo.set_index("dataset_id").loc[i, "n_subjects_canonical"])})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
print(t.groupby("side")[["registry_n_parsed", "canonical_subjects"]].sum().to_string())
t.to_csv(os.path.join(OUT, "harmonized_side_subjects.csv"), index=False)
