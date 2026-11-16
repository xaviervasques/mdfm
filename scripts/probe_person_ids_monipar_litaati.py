"""Person-identity audit for monipar and li_taati (task T0.2a of docs/EXECUTION_PLAN.md).

Patient-level splits need one id per PERSON. In the MoveAtlas canonical manifests,
monipar carries 46 subject ids and li_taati 258 (pooled pose manifest) plus 132 leg
agility ids (per-source manifest), far more than the people of either source. This
script finds what each canonical subject_ns encodes, finds in the raw data and the
source documentation what identifies a person, and writes one person id per canonical
subject id.

monipar: subject_ns is monipar:sup_r<i>c<j>, a cell of the matrix SUPERVISED_RAWDATA in
raw/MONIPAR_PD_SUPERVISED.mat. The source README says rows are subjects and columns are
weeks; raw/MONIPAR SUBJECTS DATA.xlsx lists the subjects (SUPERVISED_k, REMOTE_k,
HEALTHY_CTRLk) with their number of weekly tests. The person is the matrix row.

li_taati: subject_ns is li_taati:<task>:<N>, where N is the first field of the key of
the source JSON export ('trial number - segment number (noise type)' per the source
notebook). The source ships raw/UDysRS_UPDRS_Export/sn_numbers.txt, a dictionary from
trial number to subject number, provided for unbiased cross-validation. The person is
sn_numbers[N].

Reads (read-only, nothing is executed from MoveAtlas):
  datasets/_canonical_*manifest*.csv (pooled manifests, .bak files skipped)
  datasets/imu/monipar/{datacard.md, raw/*.mat, raw/*.xlsx, raw/Monipar_README.txt,
    processed/canonical_imu.npz}
  datasets/pose/li_taati/{datacard.md, raw/UDysRS_UPDRS_Export/*, processed/manifest.csv,
    processed/canonical_manifest.csv, processed/ratings.json, processed/*.npz meta}
  docs/datasets_index.csv, docs/demographics.csv, docs/demographics_curated.csv
Writes (reports/person_ids/ by default):
  map_monipar.csv, map_li_taati.csv (columns source, subject_ns, person_id, rule,
    evidence, ambiguity; one row per canonical subject id)
  roster_monipar.csv (the 28 subjects of the source table, with the canonical coverage)
  trials_li_taati.csv (every trial number seen in the source files, with its person)
  persons_li_taati.csv (per person: trials, canonical ids, canonical records)
  checks_monipar_li_taati.json (every measured number), summary_monipar_li_taati.md
  _provenance.json and _provenance_probe_person_ids_monipar_litaati.json

Run: python scripts/probe_person_ids_monipar_litaati.py --moveatlas <moveatlas>
"""
import collections
import glob
import hashlib
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from _common import parse_args, read_csv, write_provenance  # noqa: E402

SCRIPT = "probe_person_ids_monipar_litaati"
if not any(x == "--out" or x.startswith("--out=") for x in sys.argv[1:]):
    sys.argv += ["--out", os.path.join(HERE, "reports", "person_ids")]
a = parse_args("Person-identity audit for monipar and li_taati")
MA = a.moveatlas
DS = os.path.join(MA, "datasets")
OUT = a.out
MAP_COLS = ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"]
G = 9.80665  # m/s^2 per g, the only scale assumed; the fit below checks it
LOCKBOX_MIN = 30  # design rule (docs/MFM_DESIGN_2026-09-24.md, Splits): lockbox for sources with at least 30 patients


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {p} ({len(df)} rows)")
    return p


def npz_meta(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("meta.json").decode("utf-8"))


def listdir_clean(d):
    return sorted(n for n in os.listdir(d) if not n.startswith("._"))


checks = {"monipar": {}, "li_taati": {}}
inputs = {}

# ---------------------------------------------------------------- pooled manifests
print("scanning pooled manifests")
pooled = {}
for p in sorted(glob.glob(os.path.join(DS, "_canonical_*manifest*.csv"))):
    n = os.path.basename(p)
    if n.startswith("._") or not n.endswith(".csv"):
        continue
    df = read_csv(p)
    if "source" not in df.columns:
        continue
    for src in ("monipar", "li_taati"):
        sub = df[df["source"] == src]
        if len(sub):
            pooled.setdefault(src, {})[n] = sub
for src in ("monipar", "li_taati"):
    checks[src]["pooled_manifests_with_rows"] = {k: len(v) for k, v in pooled.get(src, {}).items()}
    print(src, checks[src]["pooled_manifests_with_rows"])

idx = read_csv(os.path.join(MA, "docs", "datasets_index.csv")).set_index("id")
demo = read_csv(os.path.join(MA, "docs", "demographics.csv")).set_index("dataset_id")
cur = read_csv(os.path.join(MA, "docs", "demographics_curated.csv")).set_index("dataset")
for src in ("monipar", "li_taati"):
    checks[src]["registry_n_subjects"] = idx.loc[src, "n_subjects"]
    checks[src]["registry_status"] = idx.loc[src, "status"]
    checks[src]["demographics_n_subjects_canonical"] = demo.loc[src, "n_subjects_canonical"]
    checks[src]["demographics_n_records_canonical"] = demo.loc[src, "n_records_canonical"]
    checks[src]["curated_n_participants"] = cur.loc[src, "n_participants"]
    checks[src]["curated_sex_statement"] = cur.loc[src, "sex_statement"]
    checks[src]["curated_source_url"] = cur.loc[src, "source_url"]
    fam = "imu" if src == "monipar" else "pose"
    card = open(os.path.join(DS, fam, src, "datacard.md"), encoding="utf-8").read()
    for fld in ("population", "n_subjects", "ingestion"):
        mm_ = re.search(rf"^- {fld}: (.*)$", card, re.M)
        checks[src][f"datacard_{fld}"] = mm_.group(1).strip() if mm_ else None

# =================================================================== monipar
print("=== monipar")
MP = os.path.join(DS, "imu", "monipar")
mraw = os.path.join(MP, "raw")
C = checks["monipar"]
C["raw_files"] = listdir_clean(mraw)
man_m = pd.concat(pooled["monipar"].values(), ignore_index=True)
C["manifest_records"] = len(man_m)
C["manifest_distinct_subject_ns"] = int(man_m["subject_ns"].nunique())
pat = re.compile(r"^monipar:(sup|rem|hc)_r(\d+)c(\d+)$")
parsed = man_m["subject_ns"].str.extract(pat)
C["subject_ns_matching_pattern"] = int(parsed[0].notna().sum())
C["subject_ns_prefixes"] = parsed[0].value_counts().to_dict()
man_m["grp"] = parsed[0]
man_m["row"] = parsed[1].astype(int)
man_m["col"] = parsed[2].astype(int)
C["task_equals_rc"] = bool((man_m["task"] == "r" + parsed[1] + "c" + parsed[2]).all())

# README statements
readme = open(os.path.join(mraw, "Monipar_README.txt"), encoding="utf-8", errors="replace").read()
m = re.search(r"(\d+) patients with PD and (\d+) healthy control", readme)
C["readme_n_pd"] = int(m.group(1)) if m else None
C["readme_n_hc"] = int(m.group(2)) if m else None
C["readme_says_rows_subjects_columns_weeks"] = bool(
    re.search(r"rows represent a different subject and the columns represent a different week", readme))
C["readme_says_weekly_sessions"] = bool(re.search(r"complete set of 8 exercises once a week", readme))
inputs["monipar_readme_sha256"] = sha256(os.path.join(mraw, "Monipar_README.txt"))
pngs = [n for n in C["raw_files"] if n.lower().endswith(".png")]
C["organization_png"] = "; ".join(n.replace("\uf022", "<U+F022>") for n in pngs)

# raw matrices
mats = {"SUPERVISED": ("MONIPAR_PD_SUPERVISED.mat", "SUPERVISED_RAWDATA"),
        "REMOTE": ("MONIPAR_PD_REMOTE.mat", "REMOTE_RAWDATA"),
        "HEALTHY_CTRL": ("MONIPAR_HEALTHYCONTROL.mat", "HEALTHYCONTROL_RAWDATA")}
cells = {}
nonempty = {}
for grp, (fn, var) in mats.items():
    arr = sio.loadmat(os.path.join(mraw, fn))[var]
    cells[grp] = arr
    nonempty[grp] = [int(sum(arr[i, j].size > 0 for j in range(arr.shape[1]))) for i in range(arr.shape[0])]
    C[f"mat_shape_{grp}"] = list(arr.shape)
    C[f"mat_nonempty_cells_{grp}"] = int(sum(nonempty[grp]))
    inputs[f"{fn}_sha256"] = sha256(os.path.join(mraw, fn))
    print(grp, arr.shape, "non-empty cells", sum(nonempty[grp]))

# subject table
xl_path = os.path.join(mraw, "MONIPAR SUBJECTS DATA.xlsx")
inputs["MONIPAR SUBJECTS DATA.xlsx_sha256"] = sha256(xl_path)
xl = pd.read_excel(xl_path, sheet_name=None, header=None)
C["xlsx_sheets"] = list(xl.keys())
sd = xl["Subject data"]
hdr = [str(x).replace("\n", " ") for x in sd.iloc[0].tolist()]
sd = sd.iloc[1:].copy()
sd.columns = hdr
sd = sd[sd[hdr[0]].notna()].reset_index(drop=True)
sid_re = re.compile(r"^(SUPERVISED|REMOTE|HEALTHY_CTRL)_?(\d+)$")
rows = []
for k, r in sd.iterrows():
    sid = str(r[hdr[0]]).strip()
    mm = sid_re.match(sid)
    grp, num = mm.group(1), int(mm.group(2))
    rows.append({"xlsx_order": k + 1, "subject_id": sid, "group": grp, "group_index": num,
                 "sex": str(r[hdr[1]]).strip(), "age": r[hdr[2]], "onset": str(r[hdr[3]]).strip(),
                 "diagnosis": str(r[hdr[4]]).strip(), "weekly_tests": int(r[hdr[6]]),
                 "tremor_flag": str(r[hdr[7]]).strip()})
roster = pd.DataFrame(rows)
C["xlsx_subjects"] = len(roster)
C["xlsx_subjects_per_group"] = roster["group"].value_counts().to_dict()
C["xlsx_sex_per_group"] = {g: dict(collections.Counter(roster[roster.group == g]["sex"])) for g in mats}
# age mean and sample SD per group, to set against the source paper's Table 2 (curated age statement)
C["xlsx_age_mean_sd_per_group"] = {
    g: [round(float(pd.to_numeric(roster[roster.group == g]["age"]).mean()), 1),
        round(float(pd.to_numeric(roster[roster.group == g]["age"]).std(ddof=1)), 1)] for g in mats}
C["curated_age_statement"] = cur.loc["monipar", "age_statement"]
_age_paper = {g: [float(x), float(y)] for g, x, y in re.findall(r"(\w+) (\d+\.\d) \(\+/-(\d+\.\d)\)", C["curated_age_statement"])}
_age_key = {"SUPERVISED": "supervised", "REMOTE": "remote", "HEALTHY_CTRL": "controls"}
C["xlsx_age_mean_sd_equal_paper_per_group"] = {
    g: _age_paper.get(_age_key[g]) == C["xlsx_age_mean_sd_per_group"][g] for g in mats}
# weekly tests against non-empty matrix cells, row by row (row index = group_index - 1)
roster["mat_file"] = roster["group"].map(lambda g: mats[g][0])
roster["mat_row"] = roster["group_index"] - 1
roster["nonempty_cells_in_mat_row"] = [nonempty[g][i] if i < len(nonempty[g]) else None
                                       for g, i in zip(roster["group"], roster["mat_row"])]
roster["weekly_tests_match"] = roster["weekly_tests"] == roster["nonempty_cells_in_mat_row"]
C["rows_per_mat_equal_xlsx_subjects_per_group"] = {
    g: [int(cells[g].shape[0]), int((roster.group == g).sum())] for g in mats}
C["weekly_tests_match_nonempty_cells"] = {g: [int(roster[(roster.group == g)]["weekly_tests_match"].sum()),
                                              int((roster.group == g).sum())] for g in mats}
# second sheet: weekly MDS-UPDRS scores of the supervised group; which weeks are scored,
# against which columns of the subject's matrix row are non-empty
brady = [s for s in xl if s.lower().startswith("mds updrs")]
roster["brady_scored_weeks"] = ""
roster["brady_weeks_equal_nonempty_columns"] = ""
if brady:
    b = xl[brady[0]]
    scored = collections.defaultdict(set)
    for _, r in b.iterrows():
        sid = str(r[3]).strip()
        if sid_re.match(sid):
            vals = r.iloc[4:].tolist()
            pos = tuple(w for w, v in enumerate(vals) if isinstance(v, (int, float)) and not pd.isna(v))
            scored[sid].add(pos)
    C["brady_sheet"] = brady[0]
    C["brady_scored_week_positions_per_subject"] = {k: [list(p) for p in sorted(v)] for k, v in sorted(scored.items())}
    agree = 0
    for k2, r in roster.iterrows():
        if r["subject_id"] not in scored:
            continue
        arr = cells[r["group"]]
        cols = tuple(j for j in range(arr.shape[1]) if arr[r["mat_row"], j].size > 0)
        ok = scored[r["subject_id"]] == {cols}
        agree += ok
        roster.loc[k2, "brady_scored_weeks"] = "|".join(",".join(str(w + 1) for w in p) for p in sorted(scored[r["subject_id"]]))
        roster.loc[k2, "brady_weeks_equal_nonempty_columns"] = ok
    C["brady_scored_weeks_equal_nonempty_columns"] = [int(agree), len(scored)]
# the Tremor flag of the table against the tremor label matrices, row by row:
# Yes <-> some label >= 1, No <-> labelled and no label above 0, '-' <-> no labelled cell
tremor_mats = {"SUPERVISED": ("MONIPAR_PD_SUPERVISED_TREMOR_LABEL.mat", "SUPERVISED_TREMOR_LABEL"),
               "REMOTE": ("MONIPAR_PD_REMOTE_TREMOR_LABEL.mat", "REMOTE_TREMOR_LABEL")}
tmax = {}
for grp, (fn, var) in tremor_mats.items():
    arr = sio.loadmat(os.path.join(mraw, fn))[var]
    for i in range(arr.shape[0]):
        vals = [arr[i, j].ravel() for j in range(arr.shape[1]) if arr[i, j].size > 0]
        tmax[(grp, i)] = int(max(v.max() for v in vals)) if vals else None
roster["tremor_label_max"] = [tmax.get((g, i), "") if g in tremor_mats else ""
                              for g, i in zip(roster["group"], roster["mat_row"])]
roster["tremor_label_max"] = roster["tremor_label_max"].map(lambda v: "none" if v is None else v)


def tremor_agree(flag, mx):
    # README: -1 = no labelled signal, 0 = no tremor, 1 and 2 = tremor. A row whose
    # cells hold only -1 carries no tremor labelling, so it agrees with '-' and not with 'No'.
    if mx == "":
        return ""
    if flag == "Yes":
        return mx != "none" and mx >= 1
    if flag == "No":
        return mx != "none" and mx == 0
    return mx == "none" or mx == -1


roster["tremor_flag_agrees_with_labels"] = [tremor_agree(f, m) for f, m in zip(roster["tremor_flag"], roster["tremor_label_max"])]
C["tremor_flag_agrees_with_label_matrix_row"] = {
    g: [int((roster[roster.group == g]["tremor_flag_agrees_with_labels"] == True).sum()),  # noqa: E712
        int((roster.group == g).sum())] for g in tremor_mats}
sup_r = roster[roster.group == "SUPERVISED"]
sig = sup_r["weekly_tests"].astype(str) + "/" + sup_r["tremor_flag"]
C["supervised_rows_pinned_by_weekly_count_and_tremor_flag"] = int((sig.map(sig.value_counts()) == 1).sum())
C["supervised_row_signatures"] = dict(collections.Counter(sig))
# possible duplicate person across the PD groups (same sex, onset and diagnosis year, age within 2)
pd_rows = roster[roster.group.isin(["SUPERVISED", "REMOTE"])]
dups = []
for _, x in pd_rows[pd_rows.group == "SUPERVISED"].iterrows():
    for _, y in pd_rows[pd_rows.group == "REMOTE"].iterrows():
        if x["onset"] in ("-", "nan") or x["diagnosis"] in ("-", "nan"):
            continue
        if (x["sex"] == y["sex"] and x["onset"] == y["onset"] and x["diagnosis"] == y["diagnosis"]
                and abs(float(x["age"]) - float(y["age"])) <= 2):
            dups.append([x["subject_id"], y["subject_id"]])
C["supervised_remote_pairs_same_sex_onset_diagnosis_age_within_2"] = dups
roster["person_id"] = "monipar:person" + roster["xlsx_order"].astype(str)

# canonical records against raw cells (npz meta order = manifest records)
npz_path = os.path.join(MP, "processed", "canonical_imu.npz")
meta = npz_meta(npz_path)
z = np.load(npz_path, allow_pickle=False)
rec_check = {}
scales = []
for ri, rec in enumerate(meta["records"]):
    ns = rec["subject_ns"]
    mm = pat.match(ns)
    i, j = int(mm.group(2)), int(mm.group(3))
    raw = cells["SUPERVISED"][i, j]
    sig = z[f"r{ri:06d}/signal"][:, 0, :3].astype(np.float64)
    lab = z[f"r{ri:06d}/label"] if f"r{ri:06d}/label" in z.files else None
    same_len = raw.shape[0] == sig.shape[0]
    maxdiff = float(np.max(np.abs(sig * G - raw[:, 1:4]))) if same_len else None
    s = float(np.sum(sig * raw[:, 1:4]) / np.sum(sig * sig)) if same_len else None
    scales.append(s)
    lab_ok = bool(lab is not None and same_len and np.array_equal(np.asarray(lab).ravel(), raw[:, 4]))
    # uniqueness: does the record also equal any other cell of the three matrices?
    others = 0
    for g2, arr in cells.items():
        for i2 in range(arr.shape[0]):
            for j2 in range(arr.shape[1]):
                if (g2, i2, j2) == ("SUPERVISED", i, j) or arr[i2, j2].shape[0] != sig.shape[0]:
                    continue
                if np.max(np.abs(sig * G - arr[i2, j2][:, 1:4])) < 1e-3:
                    others += 1
    rec_check[ns] = {"row": i, "col": j, "n_frames_equal": bool(same_len), "max_abs_diff_ms2": maxdiff,
                     "label_equal": lab_ok, "other_cells_equal": others}
C["npz_records"] = len(meta["records"])
C["npz_records_equal_to_their_raw_cell"] = int(sum(v["n_frames_equal"] and v["max_abs_diff_ms2"] is not None
                                                   and v["max_abs_diff_ms2"] < 1e-3 for v in rec_check.values()))
C["npz_records_label_equal"] = int(sum(v["label_equal"] for v in rec_check.values()))
C["npz_records_equal_to_another_cell"] = int(sum(v["other_cells_equal"] > 0 for v in rec_check.values()))
C["npz_max_abs_diff_ms2"] = max(v["max_abs_diff_ms2"] for v in rec_check.values())
C["npz_fitted_scale_min_max"] = [min(scales), max(scales)]
C["manifest_n_frames_equal_raw"] = int(sum(
    int(r.n_frames) == cells["SUPERVISED"][r.row, r.col].shape[0] for r in man_m.itertuples()))

# map
sup = roster[roster.group == "SUPERVISED"].set_index("mat_row")
dup_subjects = {s for pair in dups for s in pair}
map_rows = []
for r in man_m.sort_values(["row", "col"]).itertuples():
    x = sup.loc[r.row]
    v = rec_check.get(r.subject_ns, {})
    ev = (f"MONIPAR_PD_SUPERVISED.mat SUPERVISED_RAWDATA cell (row {r.row}, column {r.col}) = subject "
          f"{x['subject_id']}, week {r.col + 1}; Monipar_README.txt: rows are subjects, columns are weeks; "
          f"MONIPAR SUBJECTS DATA.xlsx {x['subject_id']} weekly tests {x['weekly_tests']} = non-empty cells of "
          f"row {r.row} ({x['nonempty_cells_in_mat_row']}); its Tremor flag '{x['tremor_flag']}' agrees with the "
          f"tremor label matrix row (max label {x['tremor_label_max']}): {x['tremor_flag_agrees_with_labels']}; "
          f"canonical_imu.npz record equals the raw cell "
          f"(max abs diff {v.get('max_abs_diff_ms2', float('nan')):.2e} m/s2, labels equal {v.get('label_equal')}, "
          f"equal to no other cell {v.get('other_cells_equal') == 0})")
    amb = ""
    if x["subject_id"] in dup_subjects:
        other = [p for p in dups if x["subject_id"] in p][0]
        amb = (f"same sex, onset, diagnosis year and age within 2 as {other}; kept separate; "
               "the MoniPAR recruitment log (not shipped) would settle it")
    map_rows.append({"source": "monipar", "subject_ns": r.subject_ns, "person_id": x["person_id"],
                     "rule": "person = row i of the source matrix (monipar:sup_r<i>c<j> is subject "
                             "SUPERVISED_<i+1>, week j+1); person<k> numbers the rows of MONIPAR SUBJECTS DATA.xlsx",
                     "evidence": ev, "ambiguity": amb})
map_m = pd.DataFrame(map_rows, columns=MAP_COLS)
C["map_rows"] = len(map_m)
C["persons_in_canonical"] = int(map_m["person_id"].nunique())
C["canonical_ids_per_person"] = {p: int(n) for p, n in map_m.groupby("person_id").size().items()}
C["persons_in_source_table"] = int(roster["person_id"].nunique())
C["persons_on_disk_not_in_canonical"] = int(C["persons_in_source_table"] - C["persons_in_canonical"])
C["ambiguous_rows"] = int((map_m["ambiguity"] != "").sum())
write_csv(map_m, "map_monipar.csv")
roster["in_canonical"] = roster["person_id"].isin(set(map_m["person_id"]))
roster["canonical_subject_ids"] = roster["person_id"].map(map_m.groupby("person_id").size()).fillna(0).astype(int)
write_csv(roster[["person_id", "subject_id", "group", "mat_file", "mat_row", "weekly_tests",
                  "nonempty_cells_in_mat_row", "weekly_tests_match", "brady_scored_weeks",
                  "brady_weeks_equal_nonempty_columns", "tremor_flag", "tremor_label_max",
                  "tremor_flag_agrees_with_labels", "in_canonical", "canonical_subject_ids"]], "roster_monipar.csv")

# =================================================================== li_taati
print("=== li_taati")
LT = os.path.join(DS, "pose", "li_taati")
lraw = os.path.join(LT, "raw", "UDysRS_UPDRS_Export")
L = checks["li_taati"]
L["raw_files"] = listdir_clean(lraw)
man_p = pd.concat(pooled["li_taati"].values(), ignore_index=True)
man_p["manifest"] = "datasets/_canonical_manifest.csv"
man_l = read_csv(os.path.join(LT, "processed", "canonical_manifest.csv"))
man_l["manifest"] = "datasets/pose/li_taati/processed/canonical_manifest.csv"
L["pooled_records"] = len(man_p)
L["pooled_distinct_subject_ns"] = int(man_p["subject_ns"].nunique())
L["per_source_manifest_records"] = len(man_l)
L["per_source_manifest_distinct_subject_ns"] = int(man_l["subject_ns"].nunique())
L["per_source_manifest_tasks"] = man_l["subject_ns"].str.split(":").str[1].value_counts().to_dict()
man_all = pd.concat([man_p, man_l], ignore_index=True)
lpat = re.compile(r"^li_taati:([a-z_]+):(\d+)$")
lp = man_all["subject_ns"].str.extract(lpat)
L["subject_ns_matching_pattern"] = int(lp[0].notna().sum())
man_all["task"] = lp[0]
man_all["trial"] = lp[1].astype(int)
man_all["take_seg"] = [t[len(k) + 1:] if isinstance(t, str) and t.startswith(k + "_") else None
                       for t, k in zip(man_all["take"], man_all["task"])]
L["subject_ns_per_task"] = man_all.groupby("task")["subject_ns"].nunique().to_dict()
L["records_per_task"] = man_all["task"].value_counts().to_dict()
L["subject_ns_total_all_canonical_manifests"] = int(man_all["subject_ns"].nunique())
# distinct N among the pooled ids: the figure the design counts as patients ("about 134")
L["pooled_distinct_trial_numbers"] = int(man_all.loc[man_all["manifest"] == "datasets/_canonical_manifest.csv", "trial"].nunique())
# processed/manifest.csv: its column 'subject' against the first field of the source key
pm = read_csv(os.path.join(LT, "processed", "manifest.csv"))
L["processed_manifest_rows"] = len(pm)
L["processed_manifest_tasks"] = pm["task"].value_counts().to_dict()
_key_trial = pm["record_key"].str.extract(r"^\s*(\d+)")[0]
L["processed_manifest_subject_equals_key_trial_field"] = int((pm["subject"] == _key_trial).sum())

# source key format and subject numbers
nb = json.load(open(os.path.join(lraw, "data_import_demo.ipynb"), encoding="utf-8"))
nbtext = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
L["notebook_key_format_trial_segment_noise"] = bool(
    re.search(r"format 'trial number - segment number \(noise type\)'", nbtext))
L["notebook_sn_keys_are_trial_numbers"] = bool(
    re.search(r"Dictionary keys are the trial number and the value is the subject number", nbtext))
L["notebook_sn_for_cross_validation"] = bool(re.search(r"Subject numbers have also been provided to ensure unbiased cross-validation", nbtext))
L["notebook_ratings_indexed_by_trial"] = bool(re.search(r"Ratings are indexed by trial number", nbtext))
L["notebook_higher_is_max_across_tasks"] = bool(re.search(r"highest rating across all tasks", nbtext))
m = re.search(r"accompanying the paper:\s*\n?'([^']+)' - ([^\n]+)", nbtext)
L["notebook_paper"] = (m.group(1) + " - " + m.group(2).strip()) if m else None
sn = json.load(open(os.path.join(lraw, "sn_numbers.txt"), encoding="utf-8"))
sn = {int(k): int(v) for k, v in sn.items()}
inputs["sn_numbers.txt_sha256"] = sha256(os.path.join(lraw, "sn_numbers.txt"))
L["sn_trial_keys"] = len(sn)
L["sn_distinct_subject_numbers"] = len(set(sn.values()))
L["sn_subject_numbers"] = sorted(set(sn.values()))
L["sn_numbers_absent_between_1_and_max"] = sorted(set(range(1, max(sn.values()) + 1)) - set(sn.values()))
L["sn_trials_per_subject_number"] = {str(k): int(v) for k, v in sorted(collections.Counter(sn.values()).items())}
rj = json.load(open(os.path.join(LT, "processed", "ratings.json"), encoding="utf-8"))
L["processed_ratings_sn_equal_raw"] = ({int(k): int(v) for k, v in rj.get("sn_numbers", {}).items()} == sn)

KEY = re.compile(r"^\s*(\d+)(?:-(\d+))?\s*(?:\((.*)\))?\s*$")
exports = {"communication": "Communication_all_export.txt", "drinking": "Drinking_all_export.txt",
           "leg_agility": "LA_split_all_export.txt", "toe_tapping": "TT_opt_flow_export.txt"}
raw_frames = collections.defaultdict(list)  # (task, trial, seg) -> [n_frames]
trial_tasks = collections.defaultdict(dict)  # trial -> task -> n keys
for task, fn in exports.items():
    print("loading", fn)
    inputs[f"{fn}_sha256"] = sha256(os.path.join(lraw, fn))
    d = json.load(open(os.path.join(lraw, fn), encoding="utf-8"))
    bad = [k for k in d if not KEY.match(k)]
    L[f"raw_{task}_keys"] = len(d)
    L[f"raw_{task}_unparsed_keys"] = bad
    for k, v in d.items():
        mm = KEY.match(k)
        t = int(mm.group(1))
        seg = mm.group(2) if mm.group(2) else "None"
        trial_tasks[t][task] = trial_tasks[t].get(task, 0) + 1
        if task != "toe_tapping":
            nf = min(len(x) for x in v["position"].values())
            raw_frames[(task, t, seg)].append(nf)
    L[f"raw_{task}_distinct_trials"] = len({t for t in trial_tasks if task in trial_tasks[t]})
    L[f"raw_{task}_trials_not_in_sn"] = sorted(t for t in trial_tasks if task in trial_tasks[t] and t not in sn)
L["raw_trials_with_any_trajectory"] = len(trial_tasks)
L["sn_trials_without_any_trajectory"] = sorted(t for t in sn if t not in trial_tasks)
L["raw_duplicate_trial_segment_keys"] = int(sum(len(v) > 1 for v in raw_frames.values()))

# canonical records against raw keys: same task, trial, segment, frame count
matched = 0
unmatched = []
for r in man_all.itertuples():
    lst = raw_frames.get((r.task, r.trial, str(r.take_seg)), [])
    if int(r.n_frames) in lst:
        matched += 1
    else:
        unmatched.append([r.subject_ns, r.take, r.n_frames, lst])
L["canonical_records_matched_to_raw_key_and_frames"] = matched
L["canonical_records_total"] = len(man_all)
L["canonical_records_unmatched_examples"] = unmatched[:10]
# npz meta against manifests
for fn, man in (("canonical.npz", man_p), ("canonical_leg_agility.npz", man_l)):
    mt = npz_meta(os.path.join(LT, "processed", fn))
    a1 = sorted((x["subject_ns"], x["take"], int(x["n_frames"])) for x in mt["records"])
    a2 = sorted((x, t, int(n)) for x, t, n in zip(man["subject_ns"], man["take"], man["n_frames"]))
    L[f"npz_{fn}_records"] = len(mt["records"])
    L[f"npz_{fn}_equals_manifest"] = a1 == a2

# ratings keyed by trial
ud = json.load(open(os.path.join(lraw, "UDysRS.txt"), encoding="utf-8"))
up = json.load(open(os.path.join(lraw, "UPDRS.txt"), encoding="utf-8"))
ca = json.load(open(os.path.join(lraw, "CAPSIT.txt"), encoding="utf-8"))
ud_trials = {int(k) for part in ud.values() for k in part}
up_trials = {int(k) for item in up.values() for k in item}
ca_trials = {int(k) for item in ca.values() for k in item}
L["udysrs_parts"] = list(ud.keys())
L["udysrs_trials"] = len(ud_trials)
L["updrs_trials"] = len(up_trials)
L["capsit_trials"] = len(ca_trials)
L["rated_trials_without_sn"] = sorted((ud_trials | up_trials | ca_trials) - set(sn))
L["rated_trials_without_sn_with_trajectory"] = sorted(t for t in L["rated_trials_without_sn"] if t in trial_tasks)

# persons and the paper count
m = re.search(r"(\d+)\s+participants", checks["li_taati"]["curated_n_participants"])
L["paper_participants_curated"] = int(m.group(1)) if m else None
L["sn_count_equals_paper_participants"] = L["paper_participants_curated"] == L["sn_distinct_subject_numbers"]
m = re.search(r"(\d+)\s+men", checks["li_taati"]["curated_sex_statement"])
L["paper_men_curated"] = int(m.group(1)) if m else None
m = re.search(r"\d+", L["registry_n_subjects"])
L["registry_n_equals_sn_trial_keys"] = bool(m and int(m.group(0)) == L["sn_trial_keys"])

# map
trials_per_sn = collections.Counter(sn.values())
paper_ok = L["sn_count_equals_paper_participants"]
map_rows = []
for ns, g in man_all.groupby("subject_ns", sort=False):
    task, trial = g["task"].iloc[0], int(g["trial"].iloc[0])
    nrec = len(g)
    nmatch = sum(int(n) in raw_frames.get((task, trial, str(s)), []) for n, s in zip(g["n_frames"], g["take_seg"]))
    manifest = g["manifest"].iloc[0]
    if trial in sn:
        s = sn[trial]
        pid = f"li_taati:person{s}"
        other_tasks = sorted(k for k in trial_tasks.get(trial, {}) if k != task)
        ev = (f"{manifest}: {nrec} record(s) of trial {trial} in task {task}, {nmatch} matched to a raw "
              f"{exports[task]} key '{trial}[-segment]' with equal frame count; sn_numbers.txt maps trial {trial} "
              f"to subject number {s} ({trials_per_sn[s]} trials); data_import_demo.ipynb: export keys are "
              f"'trial number - segment number', sn_numbers keys are trial numbers and values subject numbers; "
              f"trial {trial} also has {', '.join(other_tasks) if other_tasks else 'no other task'}")
        amb = "" if paper_ok else (f"distinct subject numbers ({L['sn_distinct_subject_numbers']}) differ from the "
                                   f"paper count ({L['paper_participants_curated']}); the paper's participant table "
                                   "would settle it")
    else:
        pid = f"li_taati:trial{trial}_unmapped"
        ev = f"{manifest}: trial {trial} is absent from sn_numbers.txt"
        amb = "no subject number for this trial; kept as its own id; the authors' subject list would settle it"
    map_rows.append({"source": "li_taati", "subject_ns": ns, "person_id": pid,
                     "rule": "person = sn_numbers.txt[N] for subject_ns li_taati:<task>:<N>, N being the trial "
                             "number (first field of the source export key); person<k> keeps the source subject number k",
                     "evidence": ev, "ambiguity": amb})
map_l = pd.DataFrame(map_rows, columns=MAP_COLS)
map_l["_o"] = map_l["subject_ns"].map(lambda x: (x.split(":")[1], int(x.split(":")[2])))
map_l = map_l.sort_values("_o").drop(columns="_o").reset_index(drop=True)
L["map_rows"] = len(map_l)
L["map_rows_from_pooled_manifest"] = int(man_p["subject_ns"].nunique())
L["map_rows_from_per_source_manifest"] = int(man_l["subject_ns"].nunique())
L["persons_in_canonical"] = int(map_l["person_id"].nunique())
L["persons_in_pooled_manifest"] = int(map_l[map_l["subject_ns"].isin(set(man_p["subject_ns"]))]["person_id"].nunique())
L["unmapped_ids"] = int(map_l["person_id"].str.contains("_unmapped").sum())
L["ambiguous_rows"] = int((map_l["ambiguity"] != "").sum())
# communication:N and drinking:N for the same N
comm = {int(x.split(":")[2]) for x in man_all["subject_ns"] if x.startswith("li_taati:communication:")}
drink = {int(x.split(":")[2]) for x in man_all["subject_ns"] if x.startswith("li_taati:drinking:")}
leg = {int(x.split(":")[2]) for x in man_all["subject_ns"] if x.startswith("li_taati:leg_agility:")}
L["trials_in_both_communication_and_drinking"] = len(comm & drink)
L["trials_in_communication_only_among_comm_drink"] = len(comm - drink)
L["trials_in_drinking_only_among_comm_drink"] = len(drink - comm)
L["trials_in_leg_agility_also_in_communication_or_drinking"] = len(leg & (comm | drink))
write_csv(map_l, "map_li_taati.csv")

# trial table
all_trials = sorted(set(sn) | set(trial_tasks) | ud_trials | up_trials | ca_trials)
ids_by_trial = collections.defaultdict(list)
for x in sorted(man_all["subject_ns"].unique()):
    ids_by_trial[int(x.split(":")[2])].append(x)
trows = []
for t in all_trials:
    trows.append({"trial": t, "subject_number": sn.get(t, ""),
                  "person_id": f"li_taati:person{sn[t]}" if t in sn else "",
                  "communication_segments": trial_tasks[t].get("communication", 0) if t in trial_tasks else 0,
                  "drinking_keys": trial_tasks[t].get("drinking", 0) if t in trial_tasks else 0,
                  "leg_agility_keys": trial_tasks[t].get("leg_agility", 0) if t in trial_tasks else 0,
                  "toe_tapping_keys": trial_tasks[t].get("toe_tapping", 0) if t in trial_tasks else 0,
                  "udysrs_rated": t in ud_trials, "updrs_rated": t in up_trials, "capsit_rated": t in ca_trials,
                  "canonical_subject_ids": "|".join(ids_by_trial.get(t, []))})
tt = pd.DataFrame(trows)
L["trial_table_rows"] = len(tt)
write_csv(tt, "trials_li_taati.csv")

# person table
man_all["person_id"] = man_all["trial"].map(lambda t: f"li_taati:person{sn[t]}" if t in sn else f"li_taati:trial{t}_unmapped")
prow = []
for s in sorted(set(sn.values())):
    pid = f"li_taati:person{s}"
    g = man_all[man_all["person_id"] == pid]
    prow.append({"person_id": pid, "subject_number": s, "trials_in_sn_numbers": trials_per_sn[s],
                 "trials_with_trajectory": int(sum(1 for t, v in sn.items() if v == s and t in trial_tasks)),
                 "canonical_subject_ids": int(g["subject_ns"].nunique()),
                 "canonical_records_pooled": int((g["manifest"] == "datasets/_canonical_manifest.csv").sum()),
                 "canonical_records_leg_agility": int((g["manifest"] != "datasets/_canonical_manifest.csv").sum()),
                 "trial_range": f"{min(t for t, v in sn.items() if v == s)}-{max(t for t, v in sn.items() if v == s)}"})
pt = pd.DataFrame(prow)
write_csv(pt, "persons_li_taati.csv")
L["lockbox_rule_min_patients_from_design"] = LOCKBOX_MIN
L["persons_with_30_or_more_needed_for_lockbox_rule"] = L["persons_in_canonical"] >= LOCKBOX_MIN

# ---------------------------------------------------------------- json + summary
with open(os.path.join(OUT, "checks_monipar_li_taati.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(checks, f, indent=1, default=str)
print("wrote checks_monipar_li_taati.json")


def yn(b):
    return "yes" if b else "no"


Cm, Lt = checks["monipar"], checks["li_taati"]
wk = Cm["weekly_tests_match_nonempty_cells"]
lines = [
    "# Person ids of monipar and li_taati (T0.2a)",
    "",
    "Measured by scripts/probe_person_ids_monipar_litaati.py against the MoveAtlas tree (commit in",
    "_provenance_probe_person_ids_monipar_litaati.json). Every number below is in checks_monipar_li_taati.json.",
    "",
    "## Result",
    "",
    "| source | canonical subject ids | persons | rule |",
    "|---|---|---|---|",
    f"| monipar | {Cm['map_rows']} | {Cm['persons_in_canonical']} | person = row of the source matrix SUPERVISED_RAWDATA (subject_ns sup_r<i>c<j>: row i = subject, column j = week) |",
    f"| li_taati | {Lt['map_rows']} ({Lt['map_rows_from_pooled_manifest']} pooled, {Lt['map_rows_from_per_source_manifest']} leg agility) | {Lt['persons_in_canonical']} | person = sn_numbers.txt[N], N = trial number in subject_ns li_taati:<task>:<N> |",
    "",
    f"Ambiguous rows: monipar {Cm['ambiguous_rows']}, li_taati {Lt['ambiguous_rows']}. Unmapped li_taati ids: {Lt['unmapped_ids']}.",
    "",
    "## monipar",
    "",
    f"- Where the ids come from: the pooled IMU manifest holds {Cm['manifest_records']} monipar records with "
    f"{Cm['manifest_distinct_subject_ns']} distinct subject_ns, all of the form monipar:sup_r<i>c<j> "
    f"({Cm['subject_ns_matching_pattern']} match; prefixes {Cm['subject_ns_prefixes']}); the task column repeats r<i>c<j> "
    f"({yn(Cm['task_equals_rc'])}). Each id is one cell of SUPERVISED_RAWDATA in raw/MONIPAR_PD_SUPERVISED.mat, "
    f"shape {Cm['mat_shape_SUPERVISED']}, {Cm['mat_nonempty_cells_SUPERVISED']} non-empty cells. "
    f"{Cm['npz_records_equal_to_their_raw_cell']} of {Cm['npz_records']} records of processed/canonical_imu.npz equal "
    f"their raw cell (acceleration divided by g, max abs difference {Cm['npz_max_abs_diff_ms2']:.2e} m/s2), "
    f"{Cm['npz_records_label_equal']} carry the same exercise labels, {Cm['npz_records_equal_to_another_cell']} equal any other cell.",
    f"- What identifies a person: Monipar_README.txt states that rows are subjects and columns are weeks "
    f"({yn(Cm['readme_says_rows_subjects_columns_weeks'])}) and that each participant did the 8 exercises once a week "
    f"({yn(Cm['readme_says_weekly_sessions'])}); the raw folder figure {Cm['organization_png']} labels the matrix axes "
    "SUBJECT and WEEK (read visually). MONIPAR SUBJECTS DATA.xlsx lists "
    f"{Cm['xlsx_subjects']} subjects {Cm['xlsx_subjects_per_group']}; its weekly test count equals the non-empty cells of "
    f"the matching matrix row for {wk['SUPERVISED'][0]} of {wk['SUPERVISED'][1]} supervised, {wk['REMOTE'][0]} of "
    f"{wk['REMOTE'][1]} remote and {wk['HEALTHY_CTRL'][0]} of {wk['HEALTHY_CTRL'][1]} control subjects, so row i is "
    "subject <group>_<i+1>. Two further checks of that order: the table's Tremor flag agrees with the tremor label "
    f"matrices row by row (supervised {Cm['tremor_flag_agrees_with_label_matrix_row']['SUPERVISED'][0]} of "
    f"{Cm['tremor_flag_agrees_with_label_matrix_row']['SUPERVISED'][1]}, remote "
    f"{Cm['tremor_flag_agrees_with_label_matrix_row']['REMOTE'][0]} of {Cm['tremor_flag_agrees_with_label_matrix_row']['REMOTE'][1]}), "
    f"and the scored weeks of the sheet '{Cm.get('brady_sheet', '')}' fall on the non-empty columns of the subject's row "
    f"({Cm.get('brady_scored_weeks_equal_nonempty_columns', ['', ''])[0]} of "
    f"{Cm.get('brady_scored_weeks_equal_nonempty_columns', ['', ''])[1]} supervised subjects). The README counts "
    f"{Cm['readme_n_pd']} PD and {Cm['readme_n_hc']} controls; sex per group in the table {Cm['xlsx_sex_per_group']} "
    f"against the source paper as curated by MoveAtlas: {Cm['curated_sex_statement']}; age mean and sample SD per group "
    f"in the table {Cm['xlsx_age_mean_sd_per_group']} against the paper: {Cm['curated_age_statement']} (equal to one "
    f"decimal: {Cm['xlsx_age_mean_sd_equal_paper_per_group']}). The table is the paper's cohort, one row per participant.",
    f"- Weeks of one subject are the same person: {Cm['persons_in_canonical']} persons carry the {Cm['map_rows']} ids "
    f"({Cm['canonical_ids_per_person']}).",
    f"- Supervised and remote pairs with the same sex, onset and diagnosis year and age within 2 years: "
    f"{len(Cm['supervised_remote_pairs_same_sex_onset_diagnosis_age_within_2'])}.",
    f"- Coverage: the canonical manifest holds the supervised group only; {Cm['persons_on_disk_not_in_canonical']} of the "
    f"{Cm['persons_in_source_table']} subjects of the source table (remote PD and controls) are on disk and not ingested. "
    f"The registry figure {Cm['registry_n_subjects']} is the source table, not the canonical cohort. roster_monipar.csv "
    "numbers all of them (person<k> = row k of the table) so that a later ingestion keeps the same ids.",
    "",
    "## li_taati",
    "",
    f"- Where the ids come from: the pooled pose manifest holds {Lt['pooled_records']} li_taati records with "
    f"{Lt['pooled_distinct_subject_ns']} distinct subject_ns (communication and drinking); "
    f"processed/canonical_manifest.csv adds {Lt['per_source_manifest_records']} leg agility records with "
    f"{Lt['per_source_manifest_distinct_subject_ns']} ids. Ids per task: {Lt['subject_ns_per_task']}. The id is "
    "li_taati:<task>:<N> with N the first field of the key of the source JSON export. "
    f"{Lt['canonical_records_matched_to_raw_key_and_frames']} of {Lt['canonical_records_total']} canonical records "
    "match a raw key with the same task, trial number, segment and frame count. In processed/manifest.csv "
    f"({Lt['processed_manifest_rows']} rows, {Lt['processed_manifest_tasks']}), the column named subject equals the first "
    f"field of the source key in {Lt['processed_manifest_subject_equals_key_trial_field']} rows.",
    f"- What N is: the source notebook data_import_demo.ipynb describes each export key as trial number, segment "
    f"number and noise type ({yn(Lt['notebook_key_format_trial_segment_noise'])}), says ratings are indexed by trial "
    f"number ({yn(Lt['notebook_ratings_indexed_by_trial'])}) and that the UDysRS 'Higher' part is the highest rating "
    f"across the tasks of a trial ({yn(Lt['notebook_higher_is_max_across_tasks'])}). N is an assessment (trial), not a person.",
    f"- What identifies a person: sn_numbers.txt, which the notebook presents as subject numbers provided for unbiased "
    f"cross-validation, keyed by trial number ({yn(Lt['notebook_sn_keys_are_trial_numbers'])}, "
    f"{yn(Lt['notebook_sn_for_cross_validation'])}). It has {Lt['sn_trial_keys']} trial keys and "
    f"{Lt['sn_distinct_subject_numbers']} distinct subject numbers {Lt['sn_subject_numbers']}; trials per subject number "
    f"{Lt['sn_trials_per_subject_number']}. Every trial with a trajectory is in it (trials not in sn per task: "
    f"communication {Lt['raw_communication_trials_not_in_sn']}, drinking {Lt['raw_drinking_trials_not_in_sn']}, leg agility "
    f"{Lt['raw_leg_agility_trials_not_in_sn']}, toe tapping {Lt['raw_toe_tapping_trials_not_in_sn']}). "
    f"processed/ratings.json keeps the same dictionary ({yn(Lt['processed_ratings_sn_equal_raw'])}).",
    f"- Cross-check: the source paper as curated by MoveAtlas (docs/demographics_curated.csv) reports "
    f"{Lt['paper_participants_curated']} participants and {Lt['paper_men_curated']} men; the subject numbers count "
    f"{Lt['sn_distinct_subject_numbers']} (equal: {yn(Lt['sn_count_equals_paper_participants'])}). The registry figure "
    f"{Lt['registry_n_subjects']} equals the number of trial keys of sn_numbers.txt "
    f"({yn(Lt['registry_n_equals_sn_trial_keys'])}): it counts assessments, not patients.",
    f"- communication:N and drinking:N (OD7, handoff 1 question 3): the same N is the same trial, hence the same person; "
    f"{Lt['trials_in_both_communication_and_drinking']} trials carry both ids. More than that, all trials of one subject "
    f"number are one person: the {Lt['map_rows']} canonical ids collapse to {Lt['persons_in_canonical']} persons "
    "(persons_li_taati.csv).",
    f"- Ratings for trials without trajectory and without subject number: {Lt['rated_trials_without_sn']} "
    f"(with trajectory: {Lt['rated_trials_without_sn_with_trajectory']}); they carry no canonical record.",
    "",
    "## Consequences for MD_FM (for the owner and the next handoff)",
    "",
    f"- li_taati is a {Lt['persons_in_canonical']}-person source. The design counts its {Lt['pooled_distinct_subject_ns']} "
    f"subject ids as about {Lt['pooled_distinct_trial_numbers']} patients, which is the number of distinct trial numbers N "
    f"among them (registry {Lt['registry_n_subjects']}). A person-level split has at most "
    f"{Lt['persons_in_canonical']} groups, the lockbox rule for sources of at least {LOCKBOX_MIN} patients does not apply "
    f"(at least {LOCKBOX_MIN}: {yn(Lt['persons_with_30_or_more_needed_for_lockbox_rule'])}); by the design's own rule a source "
    f"under {LOCKBOX_MIN} patients stays out of the backbone and serves the probes by patient-level cross-fitting. Every "
    "li_taati number in the design that counts patients (G8 severity, the levodopa operator, the hyperkinetic backbone "
    "count) needs to be recomputed on persons.",
    f"- monipar in the canonical manifest is {Cm['persons_in_canonical']} supervised PD persons observed weekly, not "
    f"{Cm['persons_in_source_table']} people; the remote and control matrices are on disk and not ingested.",
    f"- MoveAtlas correction candidates (not applied, MoveAtlas is read-only): li_taati n_subjects {Lt['registry_n_subjects']} "
    f"(registry and datacard, population '{Lt['datacard_population']}') is the number of trials, the source paper reports "
    "the participant count above; the li_taati manifest field named subject is the trial number; docs/demographics.csv "
    f"n_subjects_canonical counts ids, not persons (li_taati {Lt['demographics_n_subjects_canonical']}, monipar "
    f"{Cm['demographics_n_subjects_canonical']}); the monipar canonical cohort covers the supervised subgroup only, and its "
    f"datacard population '{Cm['datacard_population']}' should read {Cm['readme_n_pd']} PD and {Cm['readme_n_hc']} healthy "
    "controls, as the source README does.",
    "",
    "## Uncertainty",
    "",
    "- li_taati: the identity rests on the authors' sn_numbers.txt. Its distinct count matches the paper's participant "
    "count, so no two subject numbers are taken to be one person. Subject numbers "
    f"{Lt['sn_numbers_absent_between_1_and_max']} are absent from the file and the source files do not say why; the "
    "participant table of the source paper would say whether they are excluded participants. This does not change the map.",
    "- monipar: the partition into persons (one matrix row, one person) is exact. Which named subject a row is matters "
    "only for joining the table's clinical fields (H&Y, weekly MDS-UPDRS, Tremor): weekly count and Tremor flag single "
    f"out {Cm['supervised_rows_pinned_by_weekly_count_and_tremor_flag']} of the "
    f"{Cm['xlsx_subjects_per_group']['SUPERVISED']} supervised rows (signatures {Cm['supervised_row_signatures']}); the "
    "others rest on the README's row order.",
]
open(os.path.join(OUT, "summary_monipar_li_taati.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("wrote summary_monipar_li_taati.md")

outputs = ["map_monipar.csv", "map_li_taati.csv", "roster_monipar.csv", "trials_li_taati.csv", "persons_li_taati.csv",
           "checks_monipar_li_taati.json", "summary_monipar_li_taati.md"]
prov = write_provenance(a, {"task": "T0.2a", "outputs": outputs, "input_sha256": inputs,
                            "persons": {"monipar": Cm["persons_in_canonical"], "li_taati": Lt["persons_in_canonical"]}})
with open(os.path.join(OUT, f"_provenance_{SCRIPT}.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(prov, f, indent=1)
print("persons: monipar", Cm["persons_in_canonical"], "li_taati", Lt["persons_in_canonical"])
