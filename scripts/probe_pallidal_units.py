"""pallidal_units_dystonia identifier probe: patients per gene, units per patient, which of hemisphere,
trajectory and depth exist, and the differences from addendum 1.5 of the neural design.

Task T12 of the weekend plan 2026-10-01. Identifier columns only. The script reads the header line of
genetic_dystonia_pallidal_neuron_activity.csv to name the columns, then reads the rows with usecols
restricted to the identifier columns present (patient, gene, unit, hemisphere, trajectory, depth). The
feature column names come from the header only; their values are never read, and the script asserts
that the frame it holds carries identifier columns only (assert_identifier_frame). No firing statistic
is computed.

A unit is a row of the CSV (the file has no unit column when none is found). A patient is a distinct
value of the patient column; when one patient label occurs under two genes, the patient key becomes
gene/patient and the collision is reported.

Outputs (reports/pallidal_units/):
  columns.csv            every header column, its role (identifier or feature) and whether its values were read
  patients_per_gene.csv  per gene: patients, units, units per patient (min, median, max)
  units_per_patient.csv  per patient: gene, units, hemispheres, trajectories, depth range and missing count
  identifier_values.csv  per identifier column other than patient: each distinct value and its unit count
                         (depth: numeric summary only)
  addendum_diff.csv      each count of addendum 1.5 (and of the addendum table row of the source) against the file
  summary.md             the same in prose
  _provenance.json       stamp of scripts/moveatlas_stamp.py, input and output SHA-256
No output other than _provenance.json carries a time stamp, so two runs give identical bytes.

Run:
  CUDA_VISIBLE_DEVICES=-1 python scripts/probe_pallidal_units.py [--moveatlas <moveatlas>]
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moveatlas_stamp import default_root, stamp  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_REL = "datasets/bio/pallidal_units_dystonia/raw/genetic_dystonia_pallidal_neuron_activity.csv"
SHA_REL = "datasets/bio/pallidal_units_dystonia/downloads.sha256"
ADDENDUM_REL = "docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md"
OUT_DEFAULT = os.path.join(TREE, "reports", "pallidal_units")

IDENTIFIER_COLUMNS = ("patient", "gene", "unit", "hemisphere", "trajectory", "depth")
WHERE_COLUMNS = ("hemisphere", "trajectory", "depth")
# Header tokens of covariates that addendum 1.2 (table row) and A3 Q10 say are absent.
ABSENT_TOKENS = ("age", "anaesthesia", "anesthesia", "anaesthetic", "anesthetic", "centre", "center", "site",
                 "era", "year", "severity", "sex", "bfmdrs", "burke")
TITLE = "gene or its surgical correlates"


# ----------------------------------------------------------------------------- reading

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path):
    """Column names from the first line only."""
    with open(path, encoding="utf-8", newline="") as f:
        line = f.readline()
    return next(csv.reader([line]))


def identifier_usecols(header):
    """The identifier columns present in the header, in header order."""
    use = [c for c in header if c.strip().lower() in IDENTIFIER_COLUMNS]
    assert set(c.strip().lower() for c in use) <= set(IDENTIFIER_COLUMNS), use
    return use


def assert_identifier_frame(df, header):
    """The frame holds identifier columns only: no feature value was read."""
    cols = [str(c).strip().lower() for c in df.columns]
    assert set(cols) <= set(IDENTIFIER_COLUMNS), f"non-identifier column read: {cols}"
    features = [c for c in header if c.strip().lower() not in IDENTIFIER_COLUMNS]
    assert not (set(features) & set(df.columns)), "a feature column entered the frame"


def read_identifiers(path):
    """(header, usecols, frame of identifier columns as strings)."""
    import pandas as pd
    header = read_header(path)
    use = identifier_usecols(header)
    assert "patient" in [c.lower() for c in use] and "gene" in [c.lower() for c in use], use
    df = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False, encoding="utf-8")
    assert_identifier_frame(df, header)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return header, use, df


# ----------------------------------------------------------------------------- addendum 1.5

def _gene_counts(text):
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, n = part.rsplit(" ", 1)
        out[name.strip()] = int(n.replace(",", ""))
    return out


def parse_addendum(path):
    """Counts the addendum states for pallidal_units_dystonia (section 1.5 line and table row)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    ref = {"found_1_5": False, "found_table": False}
    re15 = re.compile(r"^- pallidal_units_dystonia: ([\d,]+) units, (\d+) patients; patients per gene (.+?); "
                      r"units per gene (.+?) \(S\)\.\s*$")
    retab = re.compile(r"([\d,]+) GPi units from (\d+) patients as (\d+) unit-feature columns plus ([a-z, ]+?);")
    for i, line in enumerate(lines, 1):
        m = re15.match(line)
        if m:
            ref.update(found_1_5=True, line_1_5=i, units=int(m.group(1).replace(",", "")),
                       patients=int(m.group(2)), patients_per_gene=_gene_counts(m.group(3)),
                       units_per_gene=_gene_counts(m.group(4)))
        if line.startswith("| pallidal_units_dystonia |"):
            t = retab.search(line)
            if t:
                ref.update(found_table=True, line_table=i, table_units=int(t.group(1).replace(",", "")),
                           table_patients=int(t.group(2)), n_features=int(t.group(3)),
                           identifier_names=[s.strip() for s in t.group(4).split(",") if s.strip()])
    return ref


# ----------------------------------------------------------------------------- counts

def patient_keys(df):
    """(key series, list of patient labels found under more than one gene)."""
    genes_per_label = df.groupby("patient")["gene"].nunique()
    colliding = sorted(genes_per_label[genes_per_label > 1].index.tolist())
    if colliding:
        key = df["gene"] + "/" + df["patient"]
    else:
        key = df["patient"].copy()
    return key, colliding


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _fmt(x):
    if x is None:
        return ""
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


def per_patient(df, key):
    rows = []
    for k in sorted(key.unique(), key=_natural):
        sub = df[key == k]
        genes = sorted(sub["gene"].unique())
        row = {"patient": k, "gene": " ".join(genes), "n_units": len(sub)}
        if "hemisphere" in sub:
            hv = sorted(v for v in sub["hemisphere"].unique())
            row["n_hemispheres"] = len([v for v in hv if v != ""])
            row["hemispheres"] = " ".join(v if v != "" else "(empty)" for v in hv)
        if "trajectory" in sub:
            tv = sorted(v for v in sub["trajectory"].unique())
            row["n_trajectories"] = len([v for v in tv if v != ""])
            row["trajectories"] = " ".join(v if v != "" else "(empty)" for v in tv)
        if "depth" in sub:
            vals = [_num(v) for v in sub["depth"]]
            num = [v for v in vals if v is not None]
            row["depth_n_numeric"] = len(num)
            row["depth_n_missing_or_text"] = len(vals) - len(num)
            row["depth_min"] = _fmt(min(num)) if num else ""
            row["depth_max"] = _fmt(max(num)) if num else ""
        rows.append(row)
    return rows


def per_gene(df, key):
    rows = []
    for g in sorted(df["gene"].unique()):
        sub = key[df["gene"] == g]
        counts = sub.value_counts()
        n = sorted(counts.tolist())
        rows.append({"gene": g, "n_patients": len(counts), "n_units": int(len(sub)),
                     "units_per_patient_min": n[0], "units_per_patient_median": _fmt(statistics.median(n)),
                     "units_per_patient_max": n[-1]})
    rows.sort(key=lambda r: (-r["n_patients"], -r["n_units"], r["gene"]))
    return rows


def identifier_values(df):
    rows = []
    for c in ("gene", "unit", "hemisphere", "trajectory"):
        if c not in df:
            continue
        vc = df[c].value_counts()
        for v in sorted(vc.index, key=_natural):
            rows.append({"column": c, "value": v if v != "" else "(empty)", "n_units": int(vc[v])})
    if "depth" in df:
        vals = [_num(v) for v in df["depth"]]
        num = [v for v in vals if v is not None]
        rows.append({"column": "depth", "value": "numeric", "n_units": len(num)})
        rows.append({"column": "depth", "value": "missing or text", "n_units": len(vals) - len(num)})
        if num:
            rows.append({"column": "depth", "value": f"min {_fmt(min(num))}", "n_units": ""})
            rows.append({"column": "depth", "value": f"median {_fmt(statistics.median(num))}", "n_units": ""})
            rows.append({"column": "depth", "value": f"max {_fmt(max(num))}", "n_units": ""})
            rows.append({"column": "depth", "value": f"distinct {len(set(num))}", "n_units": ""})
    return rows


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def absent_covariates(header):
    """Header columns whose name tokens include a covariate the addendum says is absent."""
    hits = []
    for c in header:
        tokens = set(re.split(r"[^a-z0-9]+", c.lower()))
        if tokens & set(ABSENT_TOKENS):
            hits.append(c)
    return hits


def addendum_diff(ref, n_units, n_patients, genes, header, use):
    rows = []

    def add(item, a, m, source):
        rows.append({"item": item, "addendum": a, "file": m, "equal": "yes" if str(a) == str(m) else "no",
                     "addendum_source": source})

    s15 = f"{ADDENDUM_REL} line {ref.get('line_1_5', '?')} (section 1.5)"
    stab = f"{ADDENDUM_REL} line {ref.get('line_table', '?')} (table row)"
    if ref["found_1_5"]:
        add("units", ref["units"], n_units, s15)
        add("patients", ref["patients"], n_patients, s15)
        g_all = sorted(set(ref["patients_per_gene"]) | set(genes), key=lambda g: g)
        for g in g_all:
            add(f"patients {g}", ref["patients_per_gene"].get(g, "absent"),
                genes[g]["n_patients"] if g in genes else "absent", s15)
        for g in g_all:
            add(f"units {g}", ref["units_per_gene"].get(g, "absent"),
                genes[g]["n_units"] if g in genes else "absent", s15)
    else:
        add("section 1.5 pallidal line", "not found", "", s15)
    features = [c for c in header if c.strip().lower() not in IDENTIFIER_COLUMNS]
    present = [c.lower() for c in use]
    named = []
    if ref["found_table"]:
        add("feature columns", ref["n_features"], len(features), stab)
        named = ref["identifier_names"]
        add("identifier columns named", " ".join(named), " ".join(present), stab)
    else:
        add("table row", "not found", "", stab)
    add("unit column", "present" if "unit" in named else "absent", "present" if "unit" in present else "absent",
        stab)
    hits = absent_covariates(header)
    add("age, anaesthesia, centre, era, severity columns", "absent", " ".join(hits) if hits else "absent",
        f"{ADDENDUM_REL} table row and A3 Q10")
    return rows


# ----------------------------------------------------------------------------- writing

def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0].keys()) if rows else [])
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def stamp_line(s):
    """The MoveAtlas commit line of a summary, from a stamp of scripts/moveatlas_stamp.py (no time field)."""
    return (f"- MoveAtlas measured against: HEAD {s['head']}, uncommitted diff SHA-256 {s['diff_sha256']} "
            f"({s.get('diff_bytes', '?')} bytes), from scripts/moveatlas_stamp.py; the full stamp is in "
            "_provenance.json.")


def summary_md(n_rows, n_patients, colliding, use, header, genes_rows, pat_rows, idv_rows, diff_rows, csv_sha,
               csv_sha_ok, ref, ma_stamp):
    features = [c for c in header if c.strip().lower() not in IDENTIFIER_COLUMNS]
    present = [c.lower() for c in use]
    L = []
    L.append(f"# pallidal_units_dystonia: identifier counts ({TITLE})")
    L.append("")
    L.append("Status: DRAFT counts for the Q10 rule file (NR-6.10, NR-7.5); no firing feature value was read.")
    L.append("Written by scripts/probe_pallidal_units.py; every number below comes from the CSV files of this "
             "folder.")
    L.append("")
    L.append("## Input")
    L.append("")
    L.append(stamp_line(ma_stamp))
    L.append(f"- {CSV_REL}, SHA-256 {csv_sha} ({'equal to' if csv_sha_ok else 'DIFFERENT from'} "
             f"{SHA_REL}).")
    L.append(f"- Header: {len(header)} columns, of which {len(present)} identifier columns read "
             f"({', '.join(present)}) and {len(features)} feature columns named from the header only "
             "(columns.csv). The script reads the rows with usecols set to the identifier columns and asserts "
             "that no feature column entered its frame.")
    L.append("")
    L.append("## Counts")
    L.append("")
    unit_note = "one row per unit, the file has no unit column" if "unit" not in present else "unit column present"
    L.append(f"- Units: {n_rows} ({unit_note}). Patients: {n_patients}.")
    if colliding:
        L.append(f"- Patient labels found under more than one gene: {', '.join(colliding)}; the patient key is "
                 "gene/patient.")
    else:
        L.append("- Every patient label occurs under one gene only; the patient label is the patient key.")
    L.append("")
    L.append("| gene | patients | units | units per patient (min, median, max) |")
    L.append("|---|---|---|---|")
    for r in genes_rows:
        L.append(f"| {r['gene']} | {r['n_patients']} | {r['n_units']} | {r['units_per_patient_min']}, "
                 f"{r['units_per_patient_median']}, {r['units_per_patient_max']} |")
    upp = sorted(r["n_units"] for r in pat_rows)
    L.append("")
    L.append(f"- Units per patient over all patients: min {upp[0]}, median {_fmt(statistics.median(upp))}, "
             f"max {upp[-1]} (units_per_patient.csv).")
    genes_ge5 = [r for r in genes_rows if r["n_patients"] >= 5]
    L.append(f"- Genes with at least 5 patients: {len(genes_ge5)} ("
             + ", ".join(f"{r['gene']} {r['n_patients']}" for r in genes_ge5)
             + f"), {sum(r['n_patients'] for r in genes_ge5)} patients.")
    L.append("")
    L.append("## Hemisphere, trajectory, depth")
    L.append("")
    for c in WHERE_COLUMNS:
        L.append(f"- {c}: {'present' if c in present else 'absent'}.")
    if "hemisphere" in present:
        vals = [r for r in idv_rows if r["column"] == "hemisphere"]
        L.append("- hemisphere values: " + ", ".join(f"{r['value']} {r['n_units']}" for r in vals) + " units.")
        bil = sum(1 for r in pat_rows if r.get("n_hemispheres", 0) >= 2)
        L.append(f"- Patients with units in two or more hemisphere values: {bil} of {n_patients}.")
    if "trajectory" in present:
        vals = [r for r in idv_rows if r["column"] == "trajectory"]
        L.append("- trajectory values: " + ", ".join(f"{r['value']} {r['n_units']}" for r in vals) + " units.")
        multi = sum(1 for r in pat_rows if r.get("n_trajectories", 0) >= 2)
        L.append(f"- Patients with units in two or more trajectory values: {multi} of {n_patients}.")
    if "depth" in present:
        vals = {r["value"].split(" ")[0]: r for r in idv_rows if r["column"] == "depth"}
        num = vals.get("numeric", {}).get("n_units", 0)
        miss = vals.get("missing", {}).get("n_units", 0)
        extra = ", ".join(r["value"] for k, r in vals.items() if k in ("min", "median", "max", "distinct"))
        L.append(f"- depth: {num} numeric values, {miss} missing or text" + (f"; {extra}." if extra else "."))
    L.append("")
    L.append("## Differences from addendum 1.5")
    L.append("")
    diffs = [r for r in diff_rows if r["equal"] == "no"]
    if not ref["found_1_5"]:
        L.append("- The section 1.5 line of pallidal_units_dystonia was not found in the addendum.")
    if diffs:
        for r in diffs:
            L.append(f"- {r['item']}: addendum {r['addendum']}, file {r['file']} ({r['addendum_source']}).")
    else:
        L.append("- None: every count of addendum_diff.csv is equal.")
    same = [r for r in diff_rows if r["equal"] == "yes"]
    L.append(f"- Equal: {len(same)} of {len(diff_rows)} items of addendum_diff.csv.")
    L.append("")
    L.append("## Limits")
    L.append("")
    L.append("- Counts of identifiers only. Age at surgery, anaesthesia, centre and era are not in the file; any "
             f"later contrast on these patients is titled \"{TITLE}\" (NR-6.10.5).")
    L.append("")
    return "\n".join(L)


def run(moveatlas, out):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(moveatlas, CSV_REL)
    csv_sha = sha256_file(path)
    expected = None
    sha_path = os.path.join(moveatlas, SHA_REL)
    if os.path.isfile(sha_path):
        with open(sha_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == os.path.basename(CSV_REL):
                    expected = parts[0]
    csv_sha_ok = expected == csv_sha
    header, use, df = read_identifiers(path)
    key, colliding = patient_keys(df)
    n_patients = int(key.nunique())
    pat_rows = per_patient(df, key)
    genes_rows = per_gene(df, key)
    genes = {r["gene"]: r for r in genes_rows}
    idv_rows = identifier_values(df)
    ref = parse_addendum(os.path.join(TREE, ADDENDUM_REL))
    diff_rows = addendum_diff(ref, len(df), n_patients, genes, header, use)
    col_rows = [{"position": i + 1, "column": c,
                 "role": "identifier" if c.strip().lower() in IDENTIFIER_COLUMNS else "feature",
                 "values_read": "yes" if c in use else "no"} for i, c in enumerate(header)]
    outputs = {
        "columns.csv": (col_rows, None),
        "patients_per_gene.csv": (genes_rows, None),
        "units_per_patient.csv": (pat_rows, None),
        "identifier_values.csv": (idv_rows, ["column", "value", "n_units"]),
        "addendum_diff.csv": (diff_rows, None),
    }
    hashes = {}
    for name, (rows, fields) in outputs.items():
        p = os.path.join(out, name)
        write_csv(p, rows, fields)
        hashes[name] = sha256_file(p)
    ma_stamp = stamp(moveatlas)
    text = summary_md(len(df), n_patients, colliding, use, header, genes_rows, pat_rows, idv_rows, diff_rows,
                      csv_sha, csv_sha_ok, ref, ma_stamp)
    assert "\u2014" not in text and "\u2013" not in text
    p = os.path.join(out, "summary.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    hashes["summary.md"] = sha256_file(p)
    prov = {
        "script": "scripts/probe_pallidal_units.py",
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "moveatlas_stamp": ma_stamp,
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "inputs": {CSV_REL: csv_sha, ADDENDUM_REL: sha256_file(os.path.join(TREE, ADDENDUM_REL))},
        "input_sha256_matches_downloads_sha256": csv_sha_ok,
        "identifier_columns_read": use,
        "feature_columns_read": [],
        "outputs": {f"reports/pallidal_units/{k}": v for k, v in hashes.items()},
        "note": "identifier columns only (usecols); feature names from the header line only",
    }
    with open(os.path.join(out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    return {"n_units": len(df), "n_patients": n_patients, "genes": genes_rows, "colliding": colliding,
            "usecols": use, "diff": diff_rows, "hashes": hashes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moveatlas", default=default_root())
    p.add_argument("--out", default=OUT_DEFAULT)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = run(a.moveatlas.replace("\\", "/").rstrip("/"), a.out)
    print(f"usecols: {r['usecols']}")
    print(f"units {r['n_units']}, patients {r['n_patients']}, colliding labels {r['colliding'] or 'none'}")
    for g in r["genes"]:
        print(f"  {g['gene']}: {g['n_patients']} patients, {g['n_units']} units")
    for d in r["diff"]:
        if d["equal"] == "no":
            print(f"  differs: {d['item']}: addendum {d['addendum']}, file {d['file']}")
    for k, v in r["hashes"].items():
        print(f"  {k} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
