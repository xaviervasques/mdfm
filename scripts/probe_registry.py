"""Registry probe (read-only): columns and counts of docs/datasets_index.csv, the 12
intervention datasets crossed with the registry, licence classes, mask statistics.
Outputs: <out>/datasets_index_derived.csv, <out>/license_classes.csv, stdout log.
Run: python scripts/probe_registry.py --moveatlas <moveatlas>
"""
import os, re, sys
import pandas as pd
from _common import parse_args, write_provenance

ARGS = parse_args(__doc__.splitlines()[0])
REPO = ARGS.moveatlas
DOCS = REPO + "/docs"
DS = REPO + "/datasets"
OUT = ARGS.out
write_provenance(ARGS)


def rd(name, **kw):
    return pd.read_csv(os.path.join(DOCS, name), dtype=str, keep_default_na=False, encoding="utf-8", **kw)

idx = rd("datasets_index.csv")
interv = rd("interventions.csv")
tax = rd("taxonomy_map.csv")
lic = rd("license_matrix.csv")
mask = rd("mask_statistics.csv")
cells = rd("registry_cells.csv")
demo = rd("demographics.csv")

print("=" * 78)
print("(1) datasets_index.csv")
print("columns:", list(idx.columns))
print("rows:", len(idx))
print("distinct ids:", idx["id"].nunique())
print("family counts:\n" + idx["family"].value_counts().to_string())
species = idx["family"].map(lambda f: "animal" if f == "animal" else "human")
print("species (derived: family==animal -> animal, else human):\n" + species.value_counts().to_string())
print("visionmd present:", (idx["id"] == "visionmd").any(), "(tool, excluded from corpus counts)")
print("human minus visionmd:", int((species == "human").sum()) - int((idx["id"] == "visionmd").any()))

# Access case: no column named access_case. Derive from status/access/redistribution text.
def access_case(row):
    st = row["status"].upper()
    acc = row["access"].lower()
    if "BARRIER" in st or "REFUSED" in st:
        return "barrier (status says BARRIER/REFUSED)"
    if "WITHDRAWN" in st:
        return "withdrawn"
    return "not derivable from row text"
idx["access_case_guess"] = idx.apply(access_case, axis=1)
print("status prefixes containing BARRIER:", int(idx["status"].str.upper().str.contains("BARRIER").sum()))
print("status containing INGESTED or HARMONIZED:", int(idx["status"].str.upper().str.contains("INGESTED|HARMONIZED").sum()))
print("redistribution starts with 'yes':", int(idx["redistribution"].str.lower().str.startswith("yes").sum()))
print("redistribution starts with 'no':", int(idx["redistribution"].str.lower().str.startswith("no").sum()))
print("redistribution other:", idx.loc[~idx["redistribution"].str.lower().str.startswith(("yes", "no")), ["id", "redistribution"]].to_string(index=False))

# Harmonized flag: not a column. Derive from demographics.csv (one row per harmonized dataset)
harm_ids = set(demo["dataset_id"])
idx["harmonized_derived"] = idx["id"].isin(harm_ids)
print("harmonized (derived from docs/demographics.csv dataset_id):", int(idx["harmonized_derived"].sum()), "of", len(idx))
print("demographics.csv ids not in datasets_index:", sorted(harm_ids - set(idx["id"])))
# mask_statistics sources with alias
ALIAS = {"semg_walking": "physionet_semg_walking", "cervical_motion": "cervical_motion_multiplanar",
         "modys": "modys_video", "ntu": "ntu_rgbd", "cody": "cody_1", "cody_external": "cody_3",
         "cap_sleep_chin": "cap_sleep", "seizeit2_emg": "seizeit2", "weargait": "weargait_pd"}
mask["id"] = mask["source"].map(lambda s: ALIAS.get(s, s))
mask_ids = set(mask["id"])
print("mask_statistics distinct sources:", mask["source"].nunique(), "-> distinct registry ids:", len(mask_ids))
print("mask ids not in demographics:", sorted(mask_ids - harm_ids))
print("demographics ids not in mask:", sorted(harm_ids - mask_ids))
print("mask_statistics rows:", len(mask), "sum n_records_seen:", mask["n_records_seen"].astype(int).sum())
print("n_records_seen by family:\n" + mask.groupby("family")["n_records_seen"].apply(lambda s: s.astype(int).sum()).to_string())

print("=" * 78)
print("(2) interventions.csv, 12 rows, all columns")
print("columns:", list(interv.columns))
pd.set_option("display.max_colwidth", 400, "display.width", 400)
for _, r in interv.iterrows():
    print("-" * 70)
    for c in interv.columns:
        print(f"{c}: {r[c]}")

print("=" * 78)
print("(3) intervention datasets crossed with datasets_index.csv / registry_cells.csv / license_matrix.csv / demographics.csv")
cols = ["id", "family", "moveatlas_cells", "n_subjects", "license", "redistribution", "access", "status"]
for d in interv["dataset"]:
    print("-" * 70)
    r = idx.loc[idx["id"] == d]
    if r.empty:
        print(d, ": NOT IN datasets_index.csv"); continue
    r = r.iloc[0]
    for c in cols:
        print(f"{c}: {r[c]}")
    rc = cells.loc[cells["id"] == d]
    print("registry_cells modalities:", rc["modalities"].iloc[0] if not rc.empty else "n/a", "| flags:", rc["flags"].iloc[0] if not rc.empty else "n/a")
    lr = lic.loc[lic["id"] == d]
    print("license_matrix:", (lr["license"].iloc[0][:120] + " | " + lr["redistribution"].iloc[0][:80] + " | " + lr["portal_action"].iloc[0]) if not lr.empty else "NOT IN license_matrix.csv")
    dr = demo.loc[demo["dataset_id"] == d]
    if dr.empty:
        print("harmonized: NO (not in demographics.csv; no canonical manifest rows)")
    else:
        print("harmonized: YES canonical_families=%s n_subjects_canonical=%s n_records_canonical=%s" % (dr["canonical_families"].iloc[0], dr["n_subjects_canonical"].iloc[0], dr["n_records_canonical"].iloc[0]))
    mr = mask.loc[mask["id"] == d]
    print("mask_statistics rows:", mr[["family", "n_records_seen", "fully_usable_frac"]].to_dict("records") if not mr.empty else "none")
    tm = tax.loc[tax["jeu"] == d]
    print("taxonomy_map:", tm[["cellules", "modalites_m"]].to_dict("records") if not tm.empty else "NOT IN taxonomy_map.csv")

print("=" * 78)
print("(5) licence classes")
def lic_class(s):
    u = s.upper()
    if "CC0" in u: return "CC0"
    if "CC BY-NC-SA" in u or "CC-BY-NC-SA" in u: return "CC BY-NC-SA"
    if "CC BY-NC-ND" in u or "CC-BY-NC-ND" in u: return "CC BY-NC-ND"
    if "CC BY-SA" in u or "CC-BY-SA" in u: return "CC BY-SA"
    if "CC BY-NC" in u or "CC-BY-NC" in u: return "CC BY-NC"
    if "CC BY" in u or "CC-BY" in u: return "CC BY"
    if "ODC-BY" in u: return "ODC-By"
    if "PUBLIC DOMAIN" in u or "PUBLIC RESOURCE" in u: return "public domain"
    if "ETALAB" in u: return "etalab"
    if "MIT" in u: return "MIT"
    if "APACHE" in u: return "Apache"
    if "NON-COMMERCIAL" in u or "NONCOMMERCIAL" in u or "NON COMMERCIAL" in u: return "custom non-commercial"
    return "other/none/unverified"
lic["lic_class"] = lic["license"].map(lic_class)
print("license_matrix.csv rows:", len(lic), "columns:", list(lic.columns))
print("licence class (parsed from license text, first CC token wins):\n" + lic["lic_class"].value_counts().to_string())
print("portal_action:\n" + lic["portal_action"].value_counts().to_string())
print("redistribution yes/no (prefix):\n" + lic["redistribution"].str.lower().str[:3].value_counts().to_string())
print("-- NC or ND terms in license_matrix.csv (text contains NC, ND, non-commercial, noderiv):")
m = lic["license"].str.contains(r"BY-NC|NC-SA|NC-ND|non-?commercial|noncommercial|NoDeriv|\bND\b|research only|research use|academic", case=False, regex=True)
print(lic.loc[m, ["id", "lic_class", "portal_action"]].to_string(index=False))
print("-- registry_cells.csv license_token flag counts (all 170 rows):")
tok = cells["flags"].str.extractall(r"license_token:([A-Za-z0-9.\-]+)")[0]
print(tok.value_counts().to_string())
print("rows with license_no_spdx_token:", int(cells["flags"].str.contains("license_no_spdx_token").sum()))
print("-- registry rows with NC/ND tokens:")
print(cells.loc[cells["flags"].str.contains("CC-BY-NC"), ["id", "family", "cells", "flags"]].to_string(index=False))
print("-- datasets_index.csv rows whose license text has NC/ND wording (170 rows):")
m2 = idx["license"].str.contains(r"BY-NC|NC-SA|NC-ND|non-?commercial|noncommercial|NoDeriv|research only|research use|academic", case=False, regex=True)
print(idx.loc[m2, ["id", "family"]].assign(license=idx.loc[m2, "license"].str[:90]).to_string(index=False))
# license_matrix ids vs index ids
print("license_matrix ids not in index:", sorted(set(lic["id"]) - set(idx["id"])))
print("index ids not in license_matrix:", len(set(idx["id"]) - set(lic["id"])))

print("=" * 78)
print("(6) mask_statistics.csv")
print("columns:", list(mask.columns))
print("rows:", len(mask))
print("severity-related columns:", [c for c in mask.columns if re.search("sever|grade|score|updrs|label", c, re.I)] or "NONE")
print("rows with fully_usable_frac < 0.95:")
print(mask.loc[mask["fully_usable_frac"].astype(float) < 0.95].to_string(index=False))
print("silhouette manifest has a 'severity' column (docs/MANIFEST_SCHEMA.md section 6), the only severity field in the manifests")

# save
idx.to_csv(os.path.join(OUT, "datasets_index_derived.csv"), index=False, lineterminator="\n")
lic.to_csv(os.path.join(OUT, "license_classes.csv"), index=False, lineterminator="\n")
print("wrote", OUT)
