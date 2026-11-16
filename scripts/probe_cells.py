"""Taxonomy-cell occupancy, hyper/hypo side split and paired-modality inventory of the
harmonized corpus, from the six canonical manifests (read-only) crossed with
docs/registry_cells.csv and docs/taxonomy_map.csv.
Outputs: <out>/harmonized_per_dataset.csv, cell_occupancy_registry.csv,
cell_occupancy_taxonomy_map.csv, stdout log.
Run: python scripts/probe_cells.py --moveatlas <moveatlas>
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


# record-level source name -> registry id (docs/MANIFEST_SCHEMA.md, mask_statistics.SOURCE_ALIAS)
ALIAS = {"semg_walking": "physionet_semg_walking", "cervical_motion": "cervical_motion_multiplanar",
         "modys": "modys_video", "ntu": "ntu_rgbd", "cody": "cody_1", "cody_external": "cody_3",
         "cap_sleep_chin": "cap_sleep", "seizeit2_emg": "seizeit2", "weargait": "weargait_pd"}

MANIFESTS = {
    "pose": "_canonical_manifest.csv",
    "imu": "_canonical_imu_manifest.csv",
    "eeg": "_canonical_eeg_manifest.csv",
    "emg": "_canonical_emg_manifest.csv",
    "force": "_canonical_force_manifest.csv",
    "silhouette": "_canonical_silhouette_manifest.csv",
}

frames = []
for fam, fn in MANIFESTS.items():
    m = pd.read_csv(os.path.join(DS, fn), dtype=str, keep_default_na=False, encoding="utf-8")
    m["family"] = fam
    m["id"] = m["source"].map(lambda s: ALIAS.get(s, s))
    m["fps_f"] = pd.to_numeric(m["fps"], errors="coerce")
    m["n_frames_i"] = pd.to_numeric(m["n_frames"], errors="coerce")
    m["hours"] = m["n_frames_i"] / m["fps_f"] / 3600.0
    frames.append(m[["family", "source", "id", "subject_ns", "fps_f", "n_frames_i", "hours"]])
    print(f"{fam:11s} rows={len(m):7d} distinct sources={m['source'].nunique():3d} fps blank={int((m['fps']=='').sum())}")
all_ = pd.concat(frames, ignore_index=True)

print("=" * 78)
print("TOTALS from the six manifests")
print("records:", len(all_))
print("distinct subject_ns (all families):", all_["subject_ns"].nunique())
print("hours (records with fps):", round(all_["hours"].sum(), 1), "| records without fps:", int(all_["fps_f"].isna().sum()))
print("distinct registry ids:", all_["id"].nunique())

per = all_.groupby("id").agg(records=("subject_ns", "size"), subjects=("subject_ns", "nunique"),
                             hours=("hours", "sum"), families=("family", lambda s: "|".join(sorted(set(s)))),
                             sources=("source", lambda s: "|".join(sorted(set(s))))).reset_index()
per["hours"] = per["hours"].round(2)
# subjects summed across datasets vs distinct (cody_1 and cody_3 share the cody namespace)
print("sum of per-dataset distinct subjects:", int(per["subjects"].sum()))
cody = all_[all_["id"].isin(["cody_1", "cody_3"])]
print("cody_1 + cody_3 distinct subject_ns:", cody["subject_ns"].nunique(), "| per dataset:", cody.groupby("id")["subject_ns"].nunique().to_dict())
print("cody subject_ns sample:", sorted(cody["subject_ns"].unique())[:8])

demo = pd.read_csv(os.path.join(DOCS, "demographics.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
d = demo.set_index("dataset_id")
per["demo_subjects"] = per["id"].map(d["n_subjects_canonical"])
per["demo_records"] = per["id"].map(d["n_records_canonical"])
per["subj_match"] = per["subjects"].astype(str) == per["demo_subjects"]
per["rec_match"] = per["records"].astype(str) == per["demo_records"]
print("per-dataset table (manifest-derived; demo_* = docs/demographics.csv):")
print(per.to_string(index=False))
print("mismatches vs demographics.csv:", per.loc[~(per["subj_match"] & per["rec_match"]), ["id", "subjects", "demo_subjects", "records", "demo_records"]].to_string(index=False))
per.to_csv(os.path.join(OUT, "harmonized_per_dataset.csv"), index=False, lineterminator="\n")

# ---------------------------------------------------------------- cells
cells = pd.read_csv(os.path.join(DOCS, "registry_cells.csv"), dtype=str, keep_default_na=False, encoding="utf-8").set_index("id")
tax = pd.read_csv(os.path.join(DOCS, "taxonomy_map.csv"), dtype=str, keep_default_na=False, encoding="utf-8").set_index("jeu")
ALL_CELLS = ([f"A{i}" for i in range(1, 9)] + [f"B{i}" for i in range(1, 6)] + [f"C{i}" for i in range(1, 4)]
             + [f"D{i}" for i in range(1, 8)] + [f"E{i}" for i in range(1, 8)])

def cells_of(i, src):
    if src == "registry":
        s = cells["cells"].get(i, "")
        return [c for c in s.split("|") if c]
    s = tax["cellules"].get(i, "")
    return [c for c in re.split(r"[;|]", s) if c]

def occupancy(src):
    rows = []
    for c in ALL_CELLS:
        ids = [i for i in per["id"] if c in cells_of(i, src)]
        sub = per[per["id"].isin(ids)]
        rows.append({"cell": c, "n_sources": len(ids), "subjects_sum": int(sub["subjects"].sum()),
                     "records": int(sub["records"].sum()), "hours": round(float(sub["hours"].sum()), 1),
                     "sources": "|".join(ids)})
    return pd.DataFrame(rows)

for src in ("registry", "taxonomy_map"):
    print("=" * 78)
    print(f"CELL OCCUPANCY of the {len(per)} harmonized datasets, cell mapping = docs/{'registry_cells.csv' if src=='registry' else 'taxonomy_map.csv'}")
    print("(subjects_sum = sum of each dataset's distinct subject_ns; a multi-cell dataset is counted whole in every cell it carries, because subject-level cell membership is not in the CSVs)")
    occ = occupancy(src)
    print(occ.to_string(index=False))
    print("cells with fewer than two harmonized sources:", occ.loc[occ["n_sources"] < 2, "cell"].tolist())
    print("cells with exactly one source:", occ.loc[occ["n_sources"] == 1, ["cell", "sources"]].to_string(index=False))
    print("empty cells:", occ.loc[occ["n_sources"] == 0, "cell"].tolist())
    occ.to_csv(os.path.join(OUT, f"cell_occupancy_{src}.csv"), index=False, lineterminator="\n")
    missing = [i for i in per["id"] if not cells_of(i, src)]
    print("harmonized ids with NO cell in this mapping:", missing)

# hypo vs hyper side (registry mapping)
print("=" * 78)
print("HYPER (A*) vs HYPO (B*) side, registry_cells mapping, harmonized datasets")
def side_of(i):
    cs = cells_of(i, "registry")
    a = any(c.startswith("A") for c in cs); b = any(c.startswith("B") for c in cs)
    return "both" if a and b else "hyper" if a else "hypo" if b else "neither"
per["side"] = per["id"].map(side_of)
for s in ("hyper", "hypo", "both", "neither"):
    sub = per[per["side"] == s]
    print(f"{s:8s} datasets={len(sub):2d} subjects_sum={int(sub['subjects'].sum()):6d} records={int(sub['records'].sum()):7d} hours={round(float(sub['hours'].sum()),1):9.1f} ids={sub['id'].tolist()}")
print("per-dataset side table:")
print(per[["id", "side", "subjects", "records", "hours", "families"]].sort_values(["side", "subjects"], ascending=[True, False]).to_string(index=False))

# ---------------------------------------------------------------- paired modalities
print("=" * 78)
print("PAIRED MODALITIES: subject_ns present in 2+ canonical families")
fam_per_subj = all_.groupby("subject_ns")["family"].agg(lambda s: tuple(sorted(set(s))))
multi = fam_per_subj[fam_per_subj.map(len) >= 2]
print("subjects with >=2 families:", len(multi))
combo = multi.value_counts()
print(combo.to_string())
ids_multi = all_[all_["subject_ns"].isin(multi.index)].groupby("id")["subject_ns"].nunique()
print("per dataset (distinct multi-family subjects):")
print(ids_multi.to_string())
# note: cross-dataset pairs share a namespace only for cody; check families per id for multi subjects
x = all_[all_["subject_ns"].isin(multi.index)].groupby(["id"])["family"].agg(lambda s: "|".join(sorted(set(s))))
print(x.to_string())
