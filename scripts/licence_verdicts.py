"""Render the verified licence verdicts of 2026-09-24 and the proposed scope decision per dataset.

Input: reports/licence_research/verdicts_2026-09-24.json (web research by one search pass per group,
each verdict re-read at the source by a second, skeptical pass; no dataset downloaded, no
login, no terms accepted) and reports/data_scope/data_scope.csv (scripts/data_scope.py).
Output: reports/licence_research/licence_verdicts.csv and licence_verdicts.md.

The proposal column is this session's recommendation to the owner. It changes nothing by
itself: data_scope.py keeps classifying from the MoveAtlas registry until the owner decides
and the registry licence rows are corrected (docs/moveatlas_corrections/).

Run: python scripts/licence_verdicts.py
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rdir = os.path.join(here, "reports", "licence_research")
ver = json.load(open(os.path.join(rdir, "verdicts_2026-09-24.json"), encoding="utf-8"))
with open(os.path.join(here, "reports", "data_scope", "data_scope.csv"), encoding="utf-8", newline="") as f:
    scope = {r["id"]: r for r in csv.DictReader(f)}
decided = {}
ddir = os.path.join(here, "docs", "decisions")
for fn in sorted(os.listdir(ddir)) if os.path.isdir(ddir) else []:
    if fn.startswith("licence_decisions_") and fn.endswith(".csv"):
        with open(os.path.join(ddir, fn), encoding="utf-8", newline="") as f:
            decided.update({r["id"]: r for r in csv.DictReader(f)})

# Short labels of the licence found at the source (the full text is in the JSON).
LABEL = {
    "pd_biostamprc21": "CC BY 4.0 (DataCite; IEEE DataPort terms)",
    "kaggle_fog": "Kaggle rules 7.A any use, 7.B no redistribution; Zenodo copy CC BY 4.0",
    "ninapro": "CC0 1.0 (Dryad deposit); site states none",
    "goodwin_smm": "none found (citation request only)",
    "asdpose": "no data licence; data sharing statement 'for any purpose'; code MIT",
    "mobile_gaitlab": "data CC BY-NC (2.0 or 4.0); code Apache 2.0",
    "leuven_cp_gait": "CC0 1.0 (figshare)",
    "mini_rgbd": "CC BY-NC-SA 4.0; SMIL model not redistributable",
    "osf_fmd_eeg": "none (OSF node without licence)",
    "osf_fmd_kinematics": "none (OSF node without licence)",
    "ataxie_kinect_dib": "CC BY 4.0 (article licence covers the supplement, inferred)",
    "kaggle_spiral": "CC BY-NC-ND 4.0 set by the uploader, not the creator",
    "kamath_snpc_geo": "NCBI GEO: no NCBI restriction",
    "clinvar": "NCBI policy, attribution requested",
    "panelapp": "Genomics England terms: all rights reserved, non-commercial",
    "merk_neumann_ecog_lfp": "CC0 1.0 (Harvard Dataverse)",
    "modys_video": "CC BY 4.0 (Zenodo)",
    "mrc_lfp_dystonia": "CC BY-SA 4.0 (duplicate of oxford_dystonia_lfp)",
    "oxford_dystonia_lfp": "CC BY-SA 4.0 plus portal click-through",
    "fog_turning_filho": "CC BY 4.0 (figshare, all three files)",
}

# Proposal for the owner, with the reason in one line.
ADMIT = "admit: in scope, licence allows training and a weights release"
ADMIT_NC = "admit if MD_FM is non-commercial research"
INTERNAL = "admit as usable_not_published: internal research only"
REFERENCE = "reference only: never inside a released artefact"
NOTHING = "nothing usable on disk: treat as not_held"
KEEP = "no change of scope"
PROPOSAL = {
    "pd_biostamprc21": (ADMIT, "attribution only; the page shows no label, the DOI metadata does"),
    "kaggle_fog": (ADMIT, "use of the held Kaggle copy is any-purpose; redistribute only from the Zenodo copy"),
    "ninapro": (ADMIT, "CC0 on Dryad; the held zips come from the site, record the equivalence before a derived release"),
    "leuven_cp_gait": (ADMIT, "public domain dedication"),
    "ataxie_kinect_dib": (ADMIT, "attribution; supplement table only, no skeletons were ever published"),
    "kamath_snpc_geo": (ADMIT, "no restriction found; cite the paper and GSE178265"),
    "clinvar": (ADMIT, "attribution; never for diagnosis"),
    "mini_rgbd": (ADMIT_NC, "NC-SA: derived data and, on the safe reading, weights carry NC-SA; SMIL files never shipped"),
    "goodwin_smm": (INTERNAL, "no licence: default copyright, no redistribution, weights need the authors' written grant"),
    "asdpose": (INTERNAL, "'for any purpose' covers training; a weights release needs an explicit author statement"),
    "osf_fmd_eeg": (INTERNAL, "no licence, GDPR caution on patient data; release needs the authors' permission"),
    "osf_fmd_kinematics": (INTERNAL, "same as osf_fmd_eeg"),
    "panelapp": (REFERENCE, "non-commercial, all rights reserved, no commercialisation of outputs without consent"),
    "kaggle_spiral": (REFERENCE, "NC-ND label from a non-creator; no licence chain from the curators"),
    "mobile_gaitlab": (NOTHING, "the disk holds the code and a Gillette demo (testing only); the 1,026-patient set is not held"),
    "merk_neumann_ecog_lfp": (KEEP, "in scope; rule R1 keeps it an evaluation target"),
    "modys_video": (KEEP, "published; 34 children, split on the child (reports/modys_subjects/)"),
    "mrc_lfp_dystonia": (KEEP, "duplicate row of oxford_dystonia_lfp, which holds the data"),
    "oxford_dystonia_lfp": (KEEP, "usable_not_published; share-alike on any derived release"),
    "fog_turning_filho": (KEEP, "published; the unfetched Videos.zip is under the same CC BY 4.0"),
}

rows = []
for g in ver["groups"]:
    for v in g["verdicts"]:
        i = v["id"]
        prop, why = PROPOSAL[i]
        rows.append({
            "id": i,
            "group": g["group"],
            "current_scope": scope.get(i, {}).get("scope", "absent"),
            "decision": (decided[i]["decision"] + ", tier " + decided[i]["tier"]) if i in decided else "",
            "licence_found": LABEL[i],
            "training_and_weights": v["model_training_and_weights_release"],
            "redistribution": v["redistribution"],
            "confidence": v["confidence"],
            "proposal": prop,
            "reason": why,
            "sources": " ".join(v["source_urls"][:3]),
        })
cols = list(rows[0])
with open(os.path.join(rdir, "licence_verdicts.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

counts = {}
for r in rows:
    if r["decision"]:
        k = r["decision"].split(",")[0]
        counts[k] = counts.get(k, 0) + 1
md = ["# Licence verdicts, 2026-09-24", "",
      "Rendered by scripts/licence_verdicts.py from verdicts_2026-09-24.json. Method: " + ver["method"] + ".", "",
      "The fifteen datasets that were rights_unclear were decided on 2026-09-24 by the session on the owner's",
      "delegation (docs/decisions/licence_decisions_2026-09-24.csv); current scope is recomputed by",
      "scripts/data_scope.py with those decisions.", "",
      "## The fifteen decided datasets: decision counts", "", "| decision | datasets |", "|---|---|"]
for k, n in sorted(counts.items(), key=lambda x: -x[1]):
    md.append(f"| {k} | {n} |")
md += ["", "## Per dataset", "",
       "| id | current scope | decision | licence found at the source | training and weights | redistribution | confidence | reason |",
       "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append("| " + " | ".join(r[c] for c in ("id", "current_scope", "decision", "licence_found", "training_and_weights",
                                               "redistribution", "confidence", "reason")) + " |")
text = "\n".join(md) + "\n"
assert chr(0x2014) not in text
open(os.path.join(rdir, "licence_verdicts.md"), "w", encoding="utf-8", newline="\n").write(text)
print(text)
