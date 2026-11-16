"""Access-case counts (read-only), replicating moveatlas/registry.py::access_case
without importing the package. Output: <out>/datasets_index_access_case.csv.
Run: python scripts/probe_access.py --moveatlas <moveatlas>
"""
import os
import pandas as pd
from _common import parse_args, write_provenance

ARGS = parse_args(__doc__.splitlines()[0])
REPO = ARGS.moveatlas
DOCS = REPO + "/docs"
DS = REPO + "/datasets"
OUT = ARGS.out
write_provenance(ARGS)


idx = pd.read_csv(os.path.join(DOCS, "datasets_index.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
rec = pd.read_csv(os.path.join(DOCS, "record_titles.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
zen = pd.read_csv(os.path.join(DOCS, "zenodo_corpus_index.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
demo = pd.read_csv(os.path.join(DOCS, "demographics.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
TOOLS = {"visionmd"}
records = set(rec["dataset"])
print("record_titles.csv datasets:", len(records), "| zenodo_corpus_index.csv datasets:", zen["dataset"].nunique(),
      "| same set:", records == set(zen["dataset"]))

def access_case(row):
    if row["id"] in TOOLS:
        return "tool"
    redist = row["redistribution"].strip().lower()
    if row["license"].lower().startswith("private"):
        return "local_only"
    if redist.startswith("yes"):
        return "hosted" if row["id"] in records else "recipe"
    up = row["status"].upper()
    if "DOCUMENTED BARRIER" in up or "WRITTEN OFF" in up or "HUMAN QUEUE" in up:
        return "barrier"
    return "recipe"

idx["access_case"] = idx.apply(access_case, axis=1)
idx["species"] = idx["family"].map(lambda f: "animal" if f == "animal" else "human")
idx["harmonized"] = idx["id"].isin(set(demo["dataset_id"]))
print("access_case counts (170 rows):\n" + idx["access_case"].value_counts().to_string())
print("access_case x species:\n" + pd.crosstab(idx["access_case"], idx["species"]).to_string())
print("access_case x harmonized:\n" + pd.crosstab(idx["access_case"], idx["harmonized"]).to_string())
print("harmonized x species:\n" + pd.crosstab(idx["harmonized"], idx["species"]).to_string())
print("family x access_case:\n" + pd.crosstab(idx["family"], idx["access_case"]).to_string())
print("barrier ids:", idx.loc[idx["access_case"] == "barrier", "id"].tolist())
print("hosted ids not harmonized:", idx.loc[(idx["access_case"] == "hosted") & ~idx["harmonized"], "id"].tolist())
print("harmonized ids not hosted (served by recipe):", idx.loc[idx["harmonized"] & (idx["access_case"] != "hosted"), ["id", "access_case"]].to_string(index=False))
idx[["id", "family", "species", "access_case", "harmonized", "moveatlas_cells", "n_subjects", "license", "redistribution"]].to_csv(
    os.path.join(OUT, "datasets_index_access_case.csv"), index=False, lineterminator="\n")
print("wrote", os.path.join(OUT, "datasets_index_access_case.csv"))
