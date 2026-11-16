"""ataxie_kinect_dib: counts of the diagnosis column, naming the genetic ataxias.

Weekend task T23 (3); item 4 of the appendix of docs/feedback/CLINICAL_QUESTIONS_CROSSWALK.md
(addendum 4.7 states that no genotyped body data is in scope) and addendum 1.4 (a
supplementary table only, no skeleton data).

Source (read-only): MoveAtlas datasets/pose/ataxie_kinect_dib/raw/gait_parameters_ataxia_kinect.xlsx,
the supplementary table of Summa et al. 2020, Data in Brief 33:106307
(doi:10.1016/j.dib.2020.106307), one row per participant: Participant, Group, Diagnosis, Sara
and nine spatio-temporal gait parameters computed by the authors from Kinect v2 skeletons.

Label legend. The file carries abbreviations only. Their expansions come from Table 1 of the
article ("Diagnosis acronym legend", read 2026-10-01 at PMC7498847), which lists FRDA, ARSACS,
AT, SCA, SCA2 and Joubert S., then "Others genetic ataxias: PMMA, ADCK3, ITPR1", and separately
UCA (undiagnosed cerebellar atrophy/hypoplasia) and secondary ataxia due to a posterior cranial
fossa tumour. The group codes PA, CA and H are progressive ataxia, chronic ataxia and healthy
participants (article, Methods). The class of each label is fixed in LEGEND below:
  genetic_gene_named      the label is a gene symbol (ADCK3, ITPR1)
  genetic_single_gene     the label names a disorder caused by one gene (FRDA, ARSACS, AT, SCA2);
                          the gene in LEGEND is reference knowledge, not a column of the file
  genetic_gene_not_named  a genetic class whose gene the label does not give (SCA, Joubert S.)
  cause_undetermined      UCA
  acquired                secondary ataxia
  missing                 no diagnosis in the row
The file holds labels, not genotypes: no variant, gene test or zygosity column.

Outputs (reports/ataxie_diagnoses/ by default): sheets.csv (every sheet with its shape),
diagnosis_counts.csv (label by group), class_counts.csv (class by group),
addendum_checks.csv, facts.json, summary.md, _provenance.json (MoveAtlas stamp from
scripts/moveatlas_stamp.py). Counts only: no participant row is copied.

Run: python scripts/probe_ataxie_diagnoses.py [--moveatlas <moveatlas>] [--out DIR]
"""
import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
RAW_REL = "datasets/pose/ataxie_kinect_dib/raw"
XLSX = "gait_parameters_ataxia_kinect.xlsx"

# label as it appears in the file (lower case, stripped) -> (article legend, class, gene)
LEGEND = {
    "frda": ("Friedreich's ataxia (FRDA)", "genetic_single_gene", "FXN"),
    "arsacs": ("Autosomal recessive spastic ataxia of Charlevoix-Saguenay (ARSACS)", "genetic_single_gene", "SACS"),
    "at": ("Ataxia-telangiectasia syndrome (AT)", "genetic_single_gene", "ATM"),
    "sca2": ("Spinocerebellar ataxia type 2 (SCA2)", "genetic_single_gene", "ATXN2"),
    "sca": ("Spinocerebellar ataxia (SCA), type not given", "genetic_gene_not_named", ""),
    "joubert s.": ("Joubert syndrome (Joubert S.)", "genetic_gene_not_named", ""),
    "adck3": ("Other genetic ataxia: ADCK3", "genetic_gene_named", "ADCK3"),
    "itpr1": ("Other genetic ataxia: ITPR1", "genetic_gene_named", "ITPR1"),
    "pmma": ("Other genetic ataxia: PMMA", "genetic_gene_named", "PMMA (as written in the legend)"),
    "uca": ("Undiagnosed cerebellar atrophy/hypoplasia (UCA)", "cause_undetermined", ""),
    "secondary a.": ("Secondary ataxia due to posterior cranial fossa tumour", "acquired", ""),
}
GROUPS = {"PA": "progressive ataxia", "CA": "chronic ataxia", "H": "healthy"}
GENETIC = ("genetic_gene_named", "genetic_single_gene", "genetic_gene_not_named")


def below_normal():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


def wait_if_slow():
    while os.path.exists(SLOW):
        time.sleep(30)


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "ataxie_diagnoses"))
    a = ap.parse_args()
    root = a.moveatlas.replace("\\", "/").rstrip("/")
    raw = os.path.join(root, RAW_REL)
    path = os.path.join(raw, XLSX)
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")
    os.makedirs(a.out, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    below_normal()
    wait_if_slow()

    raw_files = sorted(f for f in os.listdir(raw) if not f.startswith("._"))
    with open(path, "rb") as f:
        xsha = hashlib.sha256(f.read()).hexdigest()
    xl = pd.ExcelFile(path)
    sheets = []
    frames = {}
    for s in xl.sheet_names:
        d = pd.read_excel(xl, s)
        frames[s] = d
        sheets.append({"sheet": s, "rows": int(d.shape[0]), "columns": int(d.shape[1]),
                       "column_names": "|".join(str(c) for c in d.columns)})
    write_csv(os.path.join(a.out, "sheets.csv"), sheets, ["sheet", "rows", "columns", "column_names"])
    nonempty = [s for s in sheets if s["rows"] > 0]
    if len(nonempty) != 1:
        sys.exit(f"expected one non-empty sheet, found {len(nonempty)}")
    d = frames[nonempty[0]["sheet"]]
    cols = list(d.columns)
    need = {"Participant", "Group", "Diagnosis"}
    if not need <= set(cols):
        sys.exit(f"columns missing: {need - set(cols)}")
    gait_cols = [c for c in cols if c not in ("Participant", "Group", "Diagnosis", "Sara")]

    grp = d["Group"].astype(str).str.strip()
    lab = d["Diagnosis"].where(d["Diagnosis"].notna(), "").astype(str).str.strip().str.lower()
    unknown = sorted(set(lab) - set(LEGEND) - {""})

    def cls(x):
        return "missing" if x == "" else LEGEND[x][1] if x in LEGEND else "not_in_legend"

    klass = lab.map(cls)
    # label by group
    rows = []
    for (g, l), n in pd.crosstab(grp, lab).stack().items():
        if n == 0:
            continue
        rows.append({"group": g, "group_meaning": GROUPS.get(g, ""), "label": l or "(missing)",
                     "legend": LEGEND[l][0] if l in LEGEND else ("" if l == "" else "not in the article legend"),
                     "class": cls(l), "gene": LEGEND[l][2] if l in LEGEND else "", "n": int(n)})
    rows.sort(key=lambda r: (r["group"], -r["n"], r["label"]))
    write_csv(os.path.join(a.out, "diagnosis_counts.csv"), rows,
              ["group", "group_meaning", "label", "legend", "class", "gene", "n"])
    crows = []
    for (g, c), n in pd.crosstab(grp, klass).stack().items():
        if n:
            crows.append({"group": g, "class": c, "n": int(n)})
    write_csv(os.path.join(a.out, "class_counts.csv"), crows, ["group", "class", "n"])

    patients = grp.isin(["PA", "CA"])
    gen = patients & klass.isin(GENETIC)
    facts = {
        "raw_files": raw_files, "xlsx_sha256": xsha, "sheets": sheets,
        "participants": int(len(d)), "distinct_participant_ids": int(d["Participant"].nunique()),
        "groups": {g: int((grp == g).sum()) for g in sorted(grp.unique())},
        "patients": int(patients.sum()),
        "labels_in_file": sorted(x for x in set(lab) if x),
        "labels_not_in_legend": unknown,
        "legend_labels_absent_from_file": sorted(set(LEGEND) - set(lab)),
        "healthy_with_label": int(((grp == "H") & (lab != "")).sum()),
        "patients_missing_label": int((patients & (lab == "")).sum()),
        "patients_by_class": {c: int((patients & (klass == c)).sum()) for c in sorted(set(klass[patients]))},
        "genetic_patients": int(gen.sum()),
        "genetic_patients_by_group": {g: int((gen & (grp == g)).sum()) for g in ("PA", "CA")},
        "genetic_labels": {l: int((patients & (lab == l)).sum()) for l in sorted(set(lab[gen]))},
        "gene_named_or_single_gene": int((patients & klass.isin(("genetic_gene_named", "genetic_single_gene"))).sum()),
        "gait_parameter_columns": gait_cols,
        "genotype_like_columns": [c for c in cols if any(k in str(c).lower() for k in ("gene", "variant", "mutation", "genotype", "allele", "repeat"))],
    }
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1)

    F = facts
    checks = []

    def add(cid, claim, source, measured, verdict):
        checks.append({"check_id": cid, "claim": claim, "source": source, "measured": measured, "verdict": verdict})

    gl = "; ".join(f"{l} {n}" for l, n in sorted(F["genetic_labels"].items(), key=lambda kv: (-kv[1], kv[0])))
    add("ATX-01", "No genotyped body data is in scope", "addendum 4.7; crosswalk appendix item 4",
        f"ataxie_kinect_dib pairs, per participant, {len(gait_cols)} spatio-temporal gait parameters and SARA with a "
        f"diagnosis label; {F['genetic_patients']} of {F['patients']} patients carry a label of a genetic ataxia ({gl}), "
        f"{F['gene_named_or_single_gene']} of them a gene symbol or a single-gene disorder; columns resembling a "
        f"genotype: {len(F['genotype_like_columns'])}. The body data are per-participant summaries, not a signal",
        "qualified" if F["genetic_patients"] > 0 else "confirmed")
    add("ATX-02", "ataxie_kinect_dib holds a supplementary table only, no skeleton data", "addendum 1.4; datacard",
        f"raw folder: {len(raw_files)} file(s) ({', '.join(raw_files)}); {len(sheets)} sheets, {len(nonempty)} non-empty "
        f"({nonempty[0]['rows']} rows x {nonempty[0]['columns']} columns: {nonempty[0]['column_names']})",
        "confirmed" if (raw_files == [XLSX] and len(nonempty) == 1) else "corrected")
    g = F["groups"]
    add("ATX-03", "31 patients with ataxia (15 progressive, 16 chronic) and 20 healthy controls", "datacard; article",
        f"{F['participants']} rows, {F['distinct_participant_ids']} distinct participant ids; "
        + ", ".join(f"{k} ({GROUPS.get(k, '?')}) {v}" for k, v in g.items()),
        "confirmed" if (g.get("PA") == 15 and g.get("CA") == 16 and g.get("H") == 20) else "corrected")
    pc = F["patients_by_class"]
    add("ATX-04", "The diagnosis column names genetic ataxias; a script count of that column is owed",
        "crosswalk appendix item 4",
        "patients by class: " + "; ".join(f"{k} {v}" for k, v in sorted(pc.items()))
        + f"; genetic by group: PA {F['genetic_patients_by_group']['PA']}, CA {F['genetic_patients_by_group']['CA']}; "
        f"healthy rows with a label: {F['healthy_with_label']}; legend labels absent from the file: "
        f"{', '.join(F['legend_labels_absent_from_file']) or 'none'}; labels outside the legend: {', '.join(unknown) or 'none'}",
        "confirmed" if F["genetic_patients"] > 0 else "corrected")
    write_csv(os.path.join(a.out, "addendum_checks.csv"), checks, ["check_id", "claim", "source", "measured", "verdict"])

    import moveatlas_stamp
    prov = {"script": os.path.basename(__file__), "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "pandas": pd.__version__, "moveatlas_root": root,
            "moveatlas_stamp": moveatlas_stamp.stamp(root), "source": f"{RAW_REL}/{XLSX}", "xlsx_sha256": xsha,
            "legend_source": "Summa et al. 2020, Data in Brief 33:106307, Table 1 (PMC7498847), read 2026-10-01",
            "legend": LEGEND}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)

    L = ["# ataxie_kinect_dib: the diagnosis column", "",
         f"Generated by scripts/probe_ataxie_diagnoses.py on {prov['run_at'][:10]} against MoveAtlas "
         f"{prov['moveatlas_stamp'].get('head', '?')} (stamp in _provenance.json), from {RAW_REL}/{XLSX}. Counts only; "
         "every number is in facts.json or a CSV of this folder.", "",
         f"{F['participants']} participants: " + ", ".join(f"{GROUPS.get(k, k)} {v}" for k, v in g.items()) + ". "
         f"{F['genetic_patients']} of the {F['patients']} patients carry the label of a genetic ataxia "
         f"(PA {F['genetic_patients_by_group']['PA']}, CA {F['genetic_patients_by_group']['CA']}): {gl}. "
         f"{F['gene_named_or_single_gene']} of them name a gene or a single-gene disorder. The others: "
         + "; ".join(f"{k} {v}" for k, v in sorted(pc.items()) if k not in GENETIC) + ". "
         + (f"The legend's {', '.join(l.upper() for l in F['legend_labels_absent_from_file'])} does not occur in the file. "
            if F["legend_labels_absent_from_file"] else ""), "",
         "| group | label | legend (article Table 1) | class | n |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['group']} | {r['label']} | {r['legend']} | {r['class']} | {r['n']} |")
    L += ["", "## Verdicts", "", "| id | claim | source | measured | verdict |", "|---|---|---|---|---|"]
    for c in checks:
        L.append(f"| {c['check_id']} | {c['claim']} | {c['source']} | {c['measured']} | {c['verdict']} |")
    L += ["", "## Reading", "",
          "This file holds diagnosis labels, not variants, so it is not genotyped in the strict sense. Addendum 4.7 does "
          "not hold for body data in the wide sense: the table pairs genetic ataxia labels with per-participant gait "
          f"parameters and SARA (check ATX-01). The largest genetic label holds {max(F['genetic_labels'].values()) if F['genetic_labels'] else 0} "
          f"patients and {sum(1 for v in F['genetic_labels'].values() if v == 1)} labels hold one each; the article "
          "describes children and adolescents; the body side is a summary table computed by the authors. The set "
          "supports a descriptive label-to-gait table at most.", "",
          "Expansions and the genetic class come from the article legend and LEGEND in the script; the gene of a "
          "single-gene disorder is reference knowledge, not a column of the file."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps({c["check_id"]: c["verdict"] for c in checks}))


if __name__ == "__main__":
    main()
