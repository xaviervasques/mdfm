"""Person-identity audit of six small sources (plan task T0.2c).

Sources: cervical_motion_multiplanar, hd_accel_jnet, pd_rest_eeg_onoff_ds002778, kaggle_fog,
cap_sleep, fog_turning_filho. For each source the script compares the canonical subject ids of
the MoveAtlas manifests with the registry count, finds the person key in the raw data and the
source tables, checks whether one person carries several ids (sessions, visits, ON and OFF,
nights), and runs a duplicate check on the recordings themselves (byte hashes, and for the
accelerometer sources a content-defined window hash that finds shared signal segments).

Reads (read-only): MoveAtlas datasets/_canonical_*manifest*.csv (not the .bak copies),
docs/datasets_index.csv, docs/demographics.csv, the raw and processed folders of the six
sources; reports/probe_cohorts/harmonized_side_subjects.csv of this tree.
Writes, all under reports/person_ids/ (a folder shared with the sibling tasks T0.2a, b, d):
  map_<source>.csv                 one row per canonical subject id: source, subject_ns,
                                   person_id, rule, evidence, ambiguity
  small_sources_counts.csv         registry count, canonical ids, persons per source
  small_sources_ambiguities.csv    every flagged pair or group, its evidence and what settles it
  small_sources_not_ingested.csv   source persons or ids that have no canonical record
  small_sources_overlap_pairs.csv  cross-id pairs that share recording content
  small_sources_cohort_checks.csv  cohort statistics recomputed against the source-paper text
  small_sources_facts.json         every number of the summary
  small_sources_summary.md         the report
  _provenance_probe_person_ids_small_sources.json  (named per script: the folder is shared)

Run: python scripts/probe_person_ids_small_sources.py --moveatlas <moveatlas>
Runtime: about 20 s with a warm file cache, a few minutes from a cold disk (reads every ingested kaggle_fog CSV and
hashes the ds002778 BDF files and the cap_sleep EDF candidates); provenance records the measured runtime.
"""
import collections
import datetime as dt
import hashlib
import io
import itertools
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
    sys.argv += ["--out", os.path.join(HERE, "reports", "person_ids")]
from _common import parse_args, read_csv, moveatlas_head  # noqa: E402

A = parse_args(__doc__.splitlines()[0])
ROOT = A.moveatlas
DS = ROOT + "/datasets"
OUT = A.out
T0 = time.time()
RNG_SEED = 0
N_PERM = 1000


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def write_csv(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8", lineterminator="\n")
    log("wrote", name, len(df), "rows")


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                     "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def num(tok):
    tok = tok.strip().lower()
    return WORDS[tok] if tok in WORDS else float(tok)


# ---------------------------------------------------------------- shared inputs
MAN_FILES = sorted(f for f in os.listdir(DS) if f.startswith("_canonical_") and f.endswith(".csv"))
MAN = []
for f in MAN_FILES:
    d = read_csv(os.path.join(DS, f))
    if "subject_ns" not in d.columns:
        continue
    d = d.assign(manifest_file=f)
    MAN.append(d)
MAN = pd.concat(MAN, ignore_index=True)
log("manifests read:", len(MAN_FILES), "files,", len(MAN), "rows")

IDX = read_csv(ROOT + "/docs/datasets_index.csv").set_index("id")
DEMO = read_csv(ROOT + "/docs/demographics.csv").set_index("dataset_id")
HSS = read_csv(os.path.join(HERE, "reports", "probe_cohorts", "harmonized_side_subjects.csv")).drop_duplicates("id").set_index("id")
HSS_PROV = json.load(open(os.path.join(HERE, "reports", "probe_cohorts", "_provenance.json"), encoding="utf-8"))

SOURCES = {
    "cervical_motion_multiplanar": "cervical_motion:",
    "hd_accel_jnet": "hd_accel:",
    "pd_rest_eeg_onoff_ds002778": "pd_rest_eeg_onoff_ds002778:",
    "kaggle_fog": "kaggle_fog:",
    "cap_sleep": "cap_sleep:",
    "fog_turning_filho": "fog_turning:",
}

FACTS = {}
MAPS = {}
AMB = []          # ambiguity rows
NOT_ING = []      # not-ingested rows
OVERLAP = []      # overlap pair rows
COHORT = []       # cohort check rows


def manifest_part(sid):
    pre = SOURCES[sid]
    d = MAN[MAN["subject_ns"].str.startswith(pre)].copy()
    pooled = d["manifest_file"].str.match(r"^_canonical_(?:[a-z]+_)?manifest\.csv$")
    f = {"manifest_rows": int(len(d)),
         "records_pooled_manifests": int(pooled.sum()),
         "rows_in_per_source_extra_files": int((~pooled).sum()),
         "manifest_rows_per_file": {k: int(v) for k, v in d.groupby("manifest_file").size().items()},
         "manifest_source_labels": sorted(d["source"].unique().tolist()),
         "canonical_subject_ids": int(d["subject_ns"].nunique())}
    # the same subject_ns set in every modality manifest of the source?
    sets = {k: set(v) for k, v in d.groupby("manifest_file")["subject_ns"]}
    allids = set(d["subject_ns"])
    f["ids_per_manifest_file"] = {k: len(v) for k, v in sets.items()}
    f["ids_missing_per_manifest_file"] = {k: len(allids - v) for k, v in sets.items()}
    return d, f


def add_amb(sid, amb_id, ids, in_manifest, kind, strength, evidence, decision, settle_by):
    AMB.append({"amb_id": amb_id, "source": sid, "ids": ";".join(ids), "in_manifest": in_manifest, "kind": kind,
                "strength": strength, "evidence": evidence, "decision": decision, "settle_by": settle_by})


def registry_facts(sid, f):
    f["registry_n_subjects_now"] = IDX.loc[sid, "n_subjects"]
    f["registry_n_subjects_probe_cohorts"] = HSS.loc[sid, "registry_n_subjects_raw"] if sid in HSS.index else ""
    f["canonical_subjects_probe_cohorts"] = HSS.loc[sid, "canonical_subjects"] if sid in HSS.index else ""
    f["demographics_n_subjects_registry"] = DEMO.loc[sid, "n_subjects_registry"]
    f["demographics_n_subjects_canonical"] = DEMO.loc[sid, "n_subjects_canonical"]
    f["paper_age_statement"] = DEMO.loc[sid, "age_statement_source_paper"]
    f["paper_sex_statement"] = DEMO.loc[sid, "sex_statement_source_paper"]
    f["paper_url"] = DEMO.loc[sid, "demographics_url"]


# ---------------------------------------------------------------- content-defined window hashing
W_ROWS, ANCHOR_MOD, MIN_DISTINCT = 64, 64, 16


def window_keys(arr):
    """Keys of 64-row windows that start at content-defined anchors (row hash mod 64 == 0).
    Two recordings that share a segment of a few hundred rows share keys whatever the offset.
    Windows with fewer than 16 distinct rows (sensor at rest, quantized) are skipped."""
    # rounded to 6 decimals so that two text serializations of one float (last-bit differences) hash alike
    a = np.ascontiguousarray(np.round(np.asarray(arr, dtype=np.float64), 6)) + 0.0
    if a.ndim == 1:
        a = a[:, None]
    u = a.view(np.uint64).reshape(a.shape)
    h = np.zeros(a.shape[0], dtype=np.uint64)
    for j in range(a.shape[1]):
        h = (h * np.uint64(1099511628211)) ^ u[:, j]
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    keys = set()
    for s in np.nonzero((h % np.uint64(ANCHOR_MOD)) == 0)[0]:
        if s + W_ROWS > len(h):
            break
        w = h[s:s + W_ROWS]
        if len(np.unique(w)) < MIN_DISTINCT:
            continue
        keys.add(hashlib.blake2b(w.tobytes(), digest_size=8).digest())
    return keys


def overlap_scan(sid, records):
    """records: list of (record_label, subject_ns, loader). Returns pair rows across records."""
    owner = collections.defaultdict(set)
    nkeys = {}
    for i, (lab, sub, load) in enumerate(records):
        ks = window_keys(load())
        nkeys[i] = len(ks)
        for k in ks:
            owner[k].add(i)
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            log(f"  {sid}: window keys {i + 1}/{len(records)}")
    pc = collections.Counter()
    for own in owner.values():
        if len(own) > 1:
            for p in itertools.combinations(sorted(own), 2):
                pc[p] += 1
    rows = []
    for (i, j), c in sorted(pc.items()):
        la, sa, _ = records[i]
        lb, sb, _ = records[j]
        rows.append({"source": sid, "record_a": la, "subject_a": sa, "record_b": lb, "subject_b": sb,
                     "shared_windows": c, "windows_a": nkeys[i], "windows_b": nkeys[j],
                     "same_subject_ns": sa == sb})
    stats = {"records_scanned": len(records), "windows_total": int(sum(nkeys.values())),
             "records_without_windows": int(sum(1 for v in nkeys.values() if v == 0)),
             "pairs_sharing_windows": len(rows),
             "cross_subject_pairs": int(sum(1 for r in rows if not r["same_subject_ns"])),
             "same_subject_pairs": int(sum(1 for r in rows if r["same_subject_ns"]))}
    OVERLAP.extend(rows)
    return rows, stats


def sha256_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def edf_header(path):
    with open(path, "rb") as f:
        h = f.read(256)
    return {"patient_field": h[8:88].decode("latin-1").strip(), "recording_field": h[88:168].decode("latin-1").strip(),
            "startdate": h[168:176].decode("latin-1").strip(), "starttime": h[176:184].decode("latin-1").strip(),
            "header_bytes": int(h[184:192].decode("latin-1").strip()),
            "n_records": int(h[236:244].decode("latin-1").strip()),
            "record_s": float(h[244:252].decode("latin-1").strip()), "n_signals": int(h[252:256].decode("latin-1").strip()),
            "size": os.path.getsize(path)}


def data_prefix_hash(path, header_bytes, nbytes=8 << 20):
    with open(path, "rb") as f:
        f.seek(header_bytes)
        return hashlib.sha1(f.read(nbytes)).hexdigest()


# ================================================================= 1. cervical_motion_multiplanar
def do_cervical():
    sid = "cervical_motion_multiplanar"
    log("==", sid)
    base = DS + "/imu/cervical_motion_multiplanar"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    z = zipfile.ZipFile(base + "/raw/Dataset.zip")
    rows = []
    for i in z.infolist():
        name = i.filename
        if not name.lower().endswith(".csv") or os.path.basename(name).startswith("._"):
            continue
        parts = name.split("/")
        folder = parts[1]
        fdate, code = folder.split("_", 1)
        m = re.match(r"([\d.]+)_(\d+)_(part[AB])_(.+)\.csv$", os.path.basename(name))
        rows.append({"file": name, "folder": folder, "folder_date": fdate, "code": code, "file_date": m.group(1),
                     "file_code": m.group(2), "part": m.group(3), "plane": m.group(4), "crc": i.CRC, "size": i.file_size})
    zf = pd.DataFrame(rows)
    f["zip_csv_files"] = int(len(zf))
    f["zip_subject_folders"] = int(zf["folder"].nunique())
    f["zip_codes"] = sorted(zf["code"].unique().tolist(), key=int)
    f["file_code_equals_folder_code"] = bool((zf["code"] == zf["file_code"]).all())
    f["folders_with_file_date_not_folder_date"] = sorted(zf.loc[zf["folder_date"] != zf["file_date"], "folder"].unique().tolist())
    per = zf.groupby("code").agg(nA=("part", lambda s: int((s == "partA").sum())), nB=("part", lambda s: int((s == "partB").sum())),
                                 folder=("folder", "first"), file_dates=("file_date", lambda s: sorted(set(s))))
    codes_A = sorted(per.index[per["nA"] > 0].tolist(), key=int)
    codes_B = sorted(per.index[per["nB"] > 0].tolist(), key=int)
    f["codes_with_partA"] = len(codes_A)
    f["codes_with_partB"] = len(codes_B)
    only_A = sorted(set(codes_A) - set(codes_B), key=int)
    only_B = sorted(set(codes_B) - set(codes_A), key=int)
    f["codes_partA_only"] = only_A
    f["codes_partB_only"] = only_B
    f["codes_both_parts"] = len(set(codes_A) & set(codes_B))
    dup = zf.groupby(["crc", "size"]).filter(lambda g: len(g) > 1)
    f["byte_identical_csv_groups"] = int(dup.groupby(["crc", "size"]).ngroups)
    # PartB spreadsheet
    xl = pd.read_excel(io.BytesIO(z.read("Dataset/PartB_angles_values.xlsx")), sheet_name=None)
    sheet_codes = {}
    for k, v in xl.items():
        sheet_codes[k] = sorted({str(int(float(x))) for x in v["Subject"].dropna().tolist() if str(x).strip()}, key=int)
    f["partB_sheet_codes"] = sheet_codes
    allsheet = set().union(*sheet_codes.values())
    f["partB_sheet_codes_equal_partB_folders"] = bool(allsheet == set(codes_B))
    # ingest audit
    aud = read_csv(base + "/processed/cervical_ingest_audit.csv")
    f["ingest_audit_rows"] = int(len(aud))
    f["ingest_audit_not_ok"] = int((~aud["result"].str.endswith("ok")).sum())
    man_codes = sorted({s.split(":", 1)[1] for s in man["subject_ns"]}, key=int)
    f["manifest_codes_equal_zip_codes"] = bool(man_codes == f["zip_codes"])
    # the source paper count
    sex = f["paper_sex_statement"]
    mm = re.findall(r"(\w+) males? and (\w+) females?", sex)
    f["paper_persons_from_sex_statement"] = int(num(mm[0][0]) + num(mm[0][1])) if mm else None
    # ambiguity: one partA-only code and one partB-only code
    amb_ids = {}
    if only_A and only_B:
        ids = [SOURCES[sid] + c for c in only_A + only_B]
        aid = "cervical-1"
        ev = (f"code(s) {', '.join(only_A)} hold partA files only and code(s) {', '.join(only_B)} hold partB files only; "
              f"{f['codes_with_partA']} codes have partA and {f['codes_with_partB']} have partB, {len(f['zip_codes'])} codes in all; "
              f"the source paper counts {f['paper_persons_from_sex_statement']} participants ({sex}); "
              f"the two counts agree only if these codes are one person, or if the paper counts each part separately; "
              f"the CSV files carry no participant metadata")
        add_amb(sid, aid, ids, "yes", "partA-only code and partB-only code", "moderate", ev,
                "kept as separate persons; assign both to the same split fold",
                "Keidan et al., Sci Data 2025, doi 10.1038/s41597-024-04351-4, Methods (participants per part) or the authors")
        for c in only_A + only_B:
            amb_ids[c] = f"{aid}: possibly the same person as {', '.join(SOURCES[sid] + x for x in only_A + only_B if x != c)} (partA-only and partB-only codes; paper counts {f['paper_persons_from_sex_statement']}); same fold advised"
    out = []
    for c in man_codes:
        r = per.loc[c]
        insheet = [k for k, v in sheet_codes.items() if c in v]
        ev = (f"zip folder Dataset/{r['folder']}: {r['nA']} partA and {r['nB']} partB files, file-name dates {'/'.join(r['file_dates'])}; "
              f"in PartB_angles_values.xlsx sheets: {len(insheet)} of {len(sheet_codes)}; "
              f"README: subject code is the unique identifier of each subject")
        out.append({"source": sid, "subject_ns": SOURCES[sid] + c, "person_id": f"{sid}:{c}", "rule": "zip_folder_subject_code",
                    "evidence": ev, "ambiguity": amb_ids.get(c, "")})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    f["persons_lower_bound_if_flags_merged"] = f["persons"] - (1 if only_A and only_B else 0)
    f["gap_explanation"] = (f"{len(f['zip_codes'])} subject codes on disk against {f['registry_n_subjects_now']} in the registry: "
                            f"{f['codes_both_parts']} codes did both parts, {len(only_A)} partA only, {len(only_B)} partB only")
    f["multi_session"] = "partA (validation) and partB Pre and Post (test and retest) sit under one code; one folder date per code"
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= 2. hd_accel_jnet
def do_hd():
    sid = "hd_accel_jnet"
    log("==", sid)
    base = DS + "/imu/hd_accel_jnet"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    meta = read_csv(base + "/raw/metadata.csv")
    meta = meta[meta["Subject"].str.strip() != ""].copy()
    meta["grp"] = meta["hd_or_healthy"].str.strip()
    meta["has_scores"] = meta[["TMS", "FA", "TFC"]].apply(lambda r: any(x.strip() not in ("", "NA") for x in r), axis=1)
    meta["age"] = meta["Age"].astype(float)
    meta["bd"] = pd.to_datetime(meta["bi_birthdate"].str.strip(), format="%m/%d/%Y")
    f["metadata_rows"] = int(len(meta))
    f["metadata_group_counts"] = {(k if k else "blank"): int(v) for k, v in meta["grp"].value_counts().items()}
    f["rows_with_clinical_scores_by_group"] = {(k if k else "blank"): int(v) for k, v in meta.groupby("grp")["has_scores"].sum().items()}
    f["blank_group_subjects"] = meta.loc[meta["grp"] == "", "Subject"].tolist()
    raw_files = sorted([x for x in os.listdir(base + "/raw") if re.match(r"^\d+\.csv$", x)], key=natkey)
    f["raw_signal_files"] = len(raw_files)
    man_ids = sorted({s.split(":", 1)[1] for s in man["subject_ns"]}, key=int)
    f["manifest_ids_equal_metadata_subjects"] = bool(man_ids == sorted(meta["Subject"].tolist(), key=int))
    f["manifest_task_equals_subject_number"] = bool((man["task"] == man["subject_ns"].str.split(":").str[1]).all())
    # the paper's cohort statistics against candidate group compositions
    ps = f["paper_age_statement"]
    ages = dict((g, (float(a), float(b))) for g, a, b in re.findall(r"(HD|HC) ([\d.]+) \+/- ([\d.]+)", ps))
    pct = dict((g, float(p)) for g, p in re.findall(r"(HD|HC) ([\d.]+)% women", f["paper_sex_statement"]))
    cands = {"HD = group 1": meta["grp"] == "1",
             "HD = group 1 plus blank": meta["grp"].isin(["1", ""]),
             "HC = group 2": meta["grp"] == "2"}
    for name, sel in cands.items():
        x = meta[sel]
        g = name.split(" ")[0]
        for fem in ["1", "2"]:
            p = 100.0 * (x["bi_sex"] == fem).mean()
            COHORT.append({"source": sid, "group_definition": name, "n": int(len(x)), "female_code": fem,
                           "female_pct": round(p, 1), "paper_female_pct": pct.get(g),
                           "female_pct_matches_paper": bool(round(p) == pct.get(g)),
                           "age_mean": round(x["age"].mean(), 2), "age_sd": round(x["age"].std(ddof=1), 2),
                           "paper_age_mean": ages.get(g, (None, None))[0], "paper_age_sd": ages.get(g, (None, None))[1],
                           "age_mean_matches_paper": bool(round(x["age"].mean(), 1) == ages.get(g, (None, None))[0]),
                           "age_sd_matches_paper": bool(round(x["age"].std(ddof=1), 1) == ages.get(g, (None, None))[1]),
                           "_raw": (x["age"].mean(), x["age"].std(ddof=1))})
    # which group sizes can give the paper's whole-number women percentage at all, whatever the sex code means
    sizes = [("HD", int((meta["grp"] == "1").sum())), ("HD", int(meta["grp"].isin(["1", ""]).sum())),
             ("HC", int((meta["grp"] == "2").sum()))]
    f["paper_pct_attainable_female_counts"] = {
        f"{g} n={n} ({pct[g]:g}%)": [k for k in range(n + 1) if int(np.floor(100.0 * k / n + 0.5)) == pct[g]]
        for g, n in sizes if g in pct}
    # birthdate coincidences
    pairs = []
    for (i, a), (j, b) in itertools.combinations(meta.iterrows(), 2):
        d = abs((a["bd"] - b["bd"]).days)
        if d <= 31 and a["bi_sex"] == b["bi_sex"]:
            pairs.append((a, b, d))
    f["same_sex_pairs_birthdate_within_31_days"] = len(pairs)
    f["pairs_identical_birthdate"] = int(sum(1 for p in pairs if p[2] == 0))
    # the chance of that many identical birthdates: birthdates permuted within sex is meaningless;
    # the null here is the uniform birthday problem over the observed birth-year range
    yrs = meta["bd"].dt.year
    days = (yrs.max() - yrs.min() + 1) * 365.25
    n = len(meta)
    f["expected_identical_birthdate_pairs_uniform"] = round(n * (n - 1) / 2 / days, 4)
    # signal overlap
    recs = [(fn, SOURCES[sid] + fn[:-4], (lambda p=base + "/raw/" + fn: pd.read_csv(p, usecols=["x", "y", "z"]).to_numpy()))
            for fn in raw_files]
    ov, st = overlap_scan(sid, recs)
    f["overlap_scan"] = st
    # synthetic positive control of the window scan: copy a segment of one real recording into another at random
    # offsets and count how often the scan sees the pair (seeded; the recordings on disk are not modified)
    A_, B_ = recs[0][2](), recs[1][2]()
    ka_ = window_keys(A_)
    rng_ = np.random.default_rng(RNG_SEED)
    inj = {}
    for L in (128, 256, 512, 1024):
        hit = 0
        for _ in range(20):
            s0, d0 = int(rng_.integers(0, len(A_) - L)), int(rng_.integers(0, len(B_) - L))
            B2 = B_.copy()
            B2[d0:d0 + L] = A_[s0:s0 + L]
            hit += int(len(ka_ & window_keys(B2)) > 0)
        inj[f"{L} rows"] = f"{hit} of 20"
    f["window_scan_injection_test"] = {"records": f"{recs[0][0]} into {recs[1][0]}", "detected": inj}
    log("  window-scan injection test:", inj)
    amb_ids = {}
    for k, (a, b, d) in enumerate(pairs, 1):
        aid = f"hd-{k}"
        ov_ab = [r for r in ov if {r["subject_a"], r["subject_b"]} == {SOURCES[sid] + a["Subject"], SOURCES[sid] + b["Subject"]}]
        shared = ov_ab[0]["shared_windows"] if ov_ab else 0
        fields = ["grp", "bi_sex", "Age", "bi_birthdate", "bi_height", "bi_weight", "TMS", "FA", "TFC"]
        ev = (f"Subject {a['Subject']} ({', '.join(f'{c}={a[c]}' for c in fields)}) against Subject {b['Subject']} "
              f"({', '.join(f'{c}={b[c]}' for c in fields)}); birthdates {d} days apart; shared signal windows {shared}")
        strength = "moderate" if d == 0 else "weak"
        kind = "identical birthdate and sex" if d == 0 else "same sex, birthdates within 31 days"
        add_amb(sid, aid, [SOURCES[sid] + a["Subject"], SOURCES[sid] + b["Subject"]], "yes", kind, strength, ev,
                "kept as separate persons; " + ("assign both to the same split fold" if d == 0 else "no fold constraint"),
                "the J-Net data paper (arXiv 2412.11286) or the Zenodo 14384260 authors: twins, repeat visit, or data-entry copy")
        for s, o in [(a, b), (b, a)]:
            amb_ids.setdefault(s["Subject"], []).append(
                f"{aid}: {kind} with {SOURCES[sid] + o['Subject']}" + ("; same fold advised" if d == 0 else ""))
    out = []
    for _, r in meta.sort_values("Subject", key=lambda s: s.map(lambda x: int(x))).iterrows():
        g = {"1": "1 (HD)", "2": "2 (healthy)", "": "blank"}[r["grp"]]
        ev = (f"raw/metadata.csv Subject={r['Subject']}: hd_or_healthy={g}, sex code={r['bi_sex']}, age={r['Age']}, "
              f"clinical scores {'present' if r['has_scores'] else 'absent'}; one wrist recording raw/{r['Subject']}.csv")
        out.append({"source": sid, "subject_ns": SOURCES[sid] + r["Subject"], "person_id": f"{sid}:{r['Subject']}",
                    "rule": "metadata_subject_row", "evidence": ev, "ambiguity": " | ".join(amb_ids.get(r["Subject"], []))})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    f["persons_lower_bound_if_flags_merged"] = f["persons"] - f["pairs_identical_birthdate"]
    f["gap_explanation"] = (f"{f['metadata_rows']} metadata rows: group 1 {f['metadata_group_counts'].get('1', 0)}, "
                            f"group 2 {f['metadata_group_counts'].get('2', 0)}, blank {f['metadata_group_counts'].get('blank', 0)} "
                            f"(Subject {', '.join(f['blank_group_subjects'])}, with clinical scores); the registry's "
                            f"{f['registry_n_subjects_now']} is group 1 plus group 2")
    f["multi_session"] = "one recording per subject id; no session or visit field"
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= 3. pd_rest_eeg_onoff_ds002778
def do_ds002778():
    sid = "pd_rest_eeg_onoff_ds002778"
    log("==", sid)
    base = DS + "/eeg/pd_rest_eeg_onoff_ds002778"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    part = pd.read_csv(base + "/raw/participants.tsv", sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    part["grp"] = part["participant_id"].str.extract(r"sub-([a-z]+)\d+", expand=False)
    part["num"] = part["participant_id"].str.extract(r"(\d+)$", expand=False).astype(int)
    f["participants_rows"] = int(len(part))
    f["participants_by_group"] = {k: int(v) for k, v in part["grp"].value_counts().items()}
    f["female_by_group"] = {k: int(v) for k, v in part.groupby("grp")["gender"].apply(lambda s: (s.str.lower() == "f").sum()).items()}
    f["id_numbers_unique_across_groups"] = bool(part["num"].is_unique)
    f["id_number_max"] = int(part["num"].max())
    f["id_numbers_absent"] = sorted(set(range(1, part["num"].max() + 1)) - set(part["num"]))
    # the paper statement
    mm = re.findall(r"(PD|controls) (\w+) female of (\d+)", f["paper_sex_statement"])
    paper = {g: (int(num(a)), int(b)) for g, a, b in mm}
    for g, key in [("PD", "pd"), ("controls", "hc")]:
        if g in paper:
            COHORT.append({"source": sid, "group_definition": f"{key} participants", "n": f["participants_by_group"].get(key, 0),
                           "female_code": "f", "female_pct": round(100.0 * f["female_by_group"].get(key, 0) / max(1, f["participants_by_group"].get(key, 0)), 1),
                           "paper_female_pct": round(100.0 * paper[g][0] / paper[g][1], 1),
                           "female_pct_matches_paper": bool(f["female_by_group"].get(key, 0) == paper[g][0] and f["participants_by_group"].get(key, 0) == paper[g][1]),
                           "age_mean": round(part.loc[part["grp"] == key, "age"].astype(float).mean(), 2),
                           "age_sd": round(part.loc[part["grp"] == key, "age"].astype(float).std(ddof=1), 2),
                           "paper_age_mean": None, "paper_age_sd": None, "age_mean_matches_paper": None, "age_sd_matches_paper": None,
                           "_raw": (part.loc[part["grp"] == key, "age"].astype(float).mean(),
                                    part.loc[part["grp"] == key, "age"].astype(float).std(ddof=1))})
    for g, pat in [("pd", r"PD ([\d.]+) \+/- ([\d.]+)"), ("hc", r"controls ([\d.]+) \+/- ([\d.]+)")]:
        mm = re.search(pat, f["paper_age_statement"])
        for row in COHORT:
            if row["source"] == sid and row["group_definition"] == f"{g} participants" and mm:
                row["paper_age_mean"], row["paper_age_sd"] = float(mm.group(1)), float(mm.group(2))
                row["age_mean_matches_paper"] = bool(round(row["age_mean"], 1) == float(mm.group(1)))
                row["age_sd_matches_paper"] = bool(round(row["age_sd"], 1) == float(mm.group(2)))
    # sessions on disk, acquisition times, BDF headers and hashes
    sess = []
    for p in part["participant_id"]:
        d = base + "/raw/" + p
        for s in sorted(x for x in os.listdir(d) if x.startswith("ses-")):
            scans = pd.read_csv(f"{d}/{s}/{p}_{s}_scans.tsv", sep="\t", dtype=str, encoding="utf-8")
            bdf = f"{d}/{s}/eeg/{p}_{s}_task-rest_eeg.bdf"
            h = edf_header(bdf)
            sess.append({"participant_id": p, "session": s, "acq_time": scans["acq_time"].iloc[0], "bdf": bdf,
                         "bdf_startdate": h["startdate"], "bdf_starttime": h["starttime"], "patient_field": h["patient_field"],
                         "size": h["size"], "sha256": sha256_file(bdf)})
    ss = pd.DataFrame(sess)
    ss["acq"] = pd.to_datetime(ss["acq_time"])
    ss["bdf_date_matches_scans"] = ss.apply(lambda r: r["acq"].strftime("%d.%m.%y") == r["bdf_startdate"], axis=1)
    f["session_files"] = int(len(ss))
    gs = ss.assign(grp=ss["participant_id"].str.extract(r"sub-([a-z]+)", expand=False)).groupby(["grp", "session"]).size()
    f["sessions_per_group"] = {f"{g}/{s}": int(v) for (g, s), v in gs.items()}
    f["bdf_patient_field_nonblank"] = int((ss["patient_field"] != "").sum())
    sex_of = dict(zip(part["participant_id"], part["gender"]))
    f["bdf_patient_field_nonblank_files"] = {f"{r['participant_id']}/{r['session']}": f"{r['patient_field']} (participants.tsv gender {sex_of[r['participant_id']]})"
                                             for _, r in ss[ss["patient_field"] != ""].iterrows()}
    f["bdf_startdate_equals_scans_acq_date"] = int(ss["bdf_date_matches_scans"].sum())
    f["byte_identical_bdf_groups"] = int((ss.groupby("sha256").size() > 1).sum())
    f["identical_acq_time_groups"] = int((ss.groupby("acq_time").size() > 1).sum())
    # acquisition years outside the modal year: the participants' notes say these ON sessions were rebuilt from a
    # preprocessed EEGLAB file, so the scans.tsv date is a conversion date, not a recording date
    modal_year = int(ss["acq"].dt.year.mode().iloc[0])
    odd = ss[ss["acq"].dt.year != modal_year]
    notes = dict(zip(part["participant_id"], part["notes"]))
    f["acq_modal_year"] = modal_year
    f["sessions_dated_outside_modal_year"] = [f"{r['participant_id']}/{r['session']} {r['acq_time']}" for _, r in odd.iterrows()]
    f["of_which_participant_note_says_preprocessed_file"] = int(sum(1 for p in odd["participant_id"] if "preprocessed" in notes.get(p, "").lower()))
    pdd = ss[ss["session"].isin(["ses-on", "ses-off"])].pivot(index="participant_id", columns="session", values="acq")
    pdd["days"] = (pdd["ses-on"] - pdd["ses-off"]).dt.total_seconds().abs() / 86400.0
    pdd["date_trusted"] = ~pdd.index.isin(odd["participant_id"])
    good = pdd.loc[pdd["date_trusted"], "days"]
    f["pd_with_on_and_off"] = int(pdd[["ses-on", "ses-off"]].notna().all(axis=1).sum())
    f["pd_on_off_same_calendar_day"] = int((pdd["ses-on"].dt.date == pdd["ses-off"].dt.date).sum())
    f["pd_on_off_pairs_with_trusted_dates"] = int(len(good))
    f["pd_on_off_days_apart_min_median_max_trusted"] = [round(float(good.min()), 2), round(float(good.median()), 2), round(float(good.max()), 2)]
    mt = man.groupby("subject_ns")["task"].apply(lambda s: tuple(sorted(s)))
    f["manifest_task_sets"] = {"|".join(k): int(v) for k, v in mt.value_counts().items()}
    f["manifest_ids_equal_participants"] = bool(sorted(mt.index) == sorted(SOURCES[sid] + p for p in part["participant_id"]))
    # demographic coincidences between participants
    cols = ["age", "gender", "hand", "MMSE", "NAART"]
    g = part.groupby(cols)["participant_id"].apply(list)
    dup = [v for v in g if len(v) > 1]
    f["participant_pairs_same_age_gender_hand_mmse_naart"] = len(dup)
    amb_ids = {}
    for k, ids in enumerate(dup, 1):
        aid = f"ds002778-{k}"
        add_amb(sid, aid, [SOURCES[sid] + x for x in ids], "yes", "identical age, gender, hand, MMSE, NAART", "weak",
                f"participants.tsv rows {', '.join(ids)} agree on {', '.join(cols)}", "kept as separate persons", "the dataset curators")
        for x in ids:
            amb_ids[x] = f"{aid}: identical demographics with {', '.join(y for y in ids if y != x)}"
    out = []
    for _, r in part.sort_values("participant_id", key=lambda s: s.map(natkey)).iterrows():
        p = r["participant_id"]
        s = ss[ss["participant_id"] == p].sort_values("session")
        sl = "; ".join(f"{x['session']} acquired {x['acq_time'][:10]}" for _, x in s.iterrows())
        extra = ""
        if p in pdd.index and pd.notna(pdd.loc[p, "days"]):
            if pdd.loc[p, "date_trusted"]:
                extra = f"; ON and OFF {pdd.loc[p, 'days']:.1f} days apart, both records under this subject_ns"
            else:
                extra = ("; the ses-on date is a file conversion date (participant note: ON data taken from a preprocessed "
                         "EEGLAB file); both records under this subject_ns")
        ev = f"participants.tsv group={r['grp']}, age={r['age']}, gender={r['gender']}; {sl}{extra}"
        out.append({"source": sid, "subject_ns": SOURCES[sid] + p, "person_id": f"{sid}:{p}", "rule": "bids_participant",
                    "evidence": ev, "ambiguity": amb_ids.get(p, "")})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    f["persons_lower_bound_if_flags_merged"] = f["persons"]
    f["gap_explanation"] = (f"{f['participants_rows']} participants in the release ({f['participants_by_group'].get('pd', 0)} PD, "
                            f"{f['participants_by_group'].get('hc', 0)} controls); one id sequence 1 to {f['id_number_max']} "
                            f"across both groups with numbers {f['id_numbers_absent']} absent; the registry's "
                            f"'{f['registry_n_subjects_now']}' is an approximation")
    f["multi_session"] = (f"each PD participant has ses-off and ses-on under one subject_ns ({f['pd_with_on_and_off']} of "
                          f"{f['participants_by_group'].get('pd', 0)}), recorded on different days")
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= 4. kaggle_fog
def do_kaggle():
    sid = "kaggle_fog"
    log("==", sid)
    R = DS + "/imu/kaggle_fog/raw/extracted"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    rd = lambda fn: read_csv(os.path.join(R, fn))
    subj, td, de, da = rd("subjects.csv"), rd("tdcsfog_metadata.csv"), rd("defog_metadata.csv"), rd("daily_metadata.csv")
    lsd = lambda d: sorted(x[:-4] for x in os.listdir(R + "/" + d) if x.endswith(".csv") and not x.startswith("._"))
    files = {k: lsd(k) for k in ["train/tdcsfog", "test/tdcsfog", "train/defog", "test/defog", "train/notype"]}
    f["raw_file_counts"] = {k: len(v) for k, v in files.items()}
    f["daily_parquet_files"] = len([x for x in os.listdir(R + "/unlabeled") if x.endswith(".parquet") and not x.startswith("._")])
    man = man.assign(kind=man["task"].str.split("_").str[0], Id=man["task"].str.split("_").str[1])
    f["manifest_records_by_kind"] = {k: int(v) for k, v in man["kind"].value_counts().items()}
    f["manifest_duplicate_tasks"] = {k: int(v) for k, v in man["task"].value_counts().items() if v > 1}
    id2subj = pd.concat([td[["Id", "Subject"]], de[["Id", "Subject"]]]).drop_duplicates()
    mm = man.merge(id2subj, on="Id", how="left")
    f["manifest_subject_equals_metadata_subject"] = int((mm["subject_ns"] == SOURCES[sid] + mm["Subject"]).sum())
    S_td, S_de, S_da = set(td["Subject"]), set(de["Subject"]), set(da["Subject"])
    S_all = set(subj["Subject"])
    S_man = set(man["subject_ns"].str.split(":").str[1])
    nt = set(files["train/notype"])
    S_de_labelled = set(de.loc[~de["Id"].isin(nt), "Subject"])
    S_de_testfile = set(de.loc[de["Id"].isin(files["test/defog"]), "Subject"])
    f["subjects_csv_rows"] = int(len(subj))
    f["subjects_csv_unique"] = len(S_all)
    f["study_sets"] = {"tdcsfog": len(S_td), "defog": len(S_de), "daily": len(S_da), "tdcsfog_and_defog": len(S_td & S_de),
                       "tdcsfog_and_daily": len(S_td & S_da), "defog_and_daily": len(S_de & S_da),
                       "union": len(S_td | S_de | S_da)}
    f["manifest_subjects"] = len(S_man)
    daily_only = S_da - S_td - S_de
    notype_only = S_de - S_de_labelled - S_td
    notype_only_but_testfile = notype_only & S_de_testfile
    f["not_ingested_daily_only"] = len(daily_only)
    f["defog_subjects_with_only_notype_recordings"] = len(notype_only)
    f["of_which_ingested_through_test_defog_file"] = len(notype_only_but_testfile)
    f["subjects_ingested_through_test_defog_file"] = {
        s: sorted(set(de.loc[(de["Subject"] == s) & de["Id"].isin(files["test/defog"]), "Id"])) for s in sorted(notype_only_but_testfile)}
    f["not_ingested_notype_only"] = len(notype_only - notype_only_but_testfile)
    f["subjects_csv_minus_manifest"] = len(S_all - S_man)
    f["gap_accounted"] = bool((S_all - S_man) == (daily_only | (notype_only - notype_only_but_testfile)))
    f["manifest_subjects_absent_from_subjects_csv"] = len(S_man - S_all)
    for s in sorted(daily_only):
        NOT_ING.append({"source": sid, "native_id": SOURCES[sid] + s, "reason": "daily-living recordings only (unlabeled/*.parquet), not ingested"})
    for s in sorted(notype_only - notype_only_but_testfile):
        NOT_ING.append({"source": sid, "native_id": SOURCES[sid] + s, "reason": "defog recordings only in train/notype, not ingested"})
    # the manifest's duplicate record and the notype copy in test/defog
    dupchk = {}
    for kind in ["tdcsfog", "defog"]:
        for i in files[f"test/{kind}"]:
            a = pd.read_csv(f"{R}/test/{kind}/{i}.csv", usecols=["AccV", "AccML", "AccAP"]).to_numpy()
            for other in [f"train/{kind}", "train/notype"]:
                if i in files[other]:
                    b = pd.read_csv(f"{R}/{other}/{i}.csv", usecols=["AccV", "AccML", "AccAP"]).to_numpy()
                    same = a.shape == b.shape and np.allclose(a, b, rtol=0.0, atol=1e-9, equal_nan=True)
                    dupchk[f"test/{kind}/{i} vs {other}/{i}"] = (
                        f"equal within 1e-9 ({a.shape[0]} rows; {int((a != b).sum()) if a.shape == b.shape else 'n/a'} values differ in the last bits)"
                        if same else "different")
    f["test_files_equal_train_copies"] = dupchk
    # visit structure
    sv = subj.groupby("Subject")["Visit"].apply(lambda v: tuple(sorted(v)))
    f["subjects_csv_visit_sets"] = {"|".join(k) if k != ("",) else "blank": int(v) for k, v in sv.value_counts().items()}
    # demographic and clinical coincidences, with a permutation null
    one = subj.drop_duplicates("Subject").copy()
    f["age_constant_within_subject"] = bool((subj.groupby("Subject")["Age"].nunique() == 1).all())

    def trip_pairs(df):
        c = df.groupby(["Age", "Sex", "YearsSinceDx"]).size()
        return int((c * (c - 1) // 2).sum())
    obs = trip_pairs(one)
    rng = np.random.default_rng(RNG_SEED)
    null = []
    for _ in range(N_PERM):
        p = one.copy()
        p["Age"] = rng.permutation(p["Age"].to_numpy())
        p["YearsSinceDx"] = rng.permutation(p["YearsSinceDx"].to_numpy())
        null.append(trip_pairs(p))
    f["pairs_same_age_sex_yearsdx_observed"] = obs
    f["pairs_same_age_sex_yearsdx_null_mean"] = round(float(np.mean(null)), 2)
    f["pairs_same_age_sex_yearsdx_null_p_ge_obs"] = round(float(np.mean(np.array(null) >= obs)), 3)
    f["permutation_null"] = f"{N_PERM} permutations of Age and YearsSinceDx across subjects, Sex fixed, seed {RNG_SEED}"
    full = ["Age", "Sex", "YearsSinceDx", "UPDRSIII_On", "UPDRSIII_Off", "NFOGQ"]
    fr = subj.groupby(full)["Subject"].apply(lambda s: sorted(set(s)))
    full_groups = [v for v in fr if len(v) > 1]
    f["subject_visit_rows_identical_on_six_fields_across_subjects"] = len(full_groups)
    nullfull = []
    for _ in range(N_PERM):
        p = subj.copy()
        for c in full:
            p[c] = rng.permutation(p[c].to_numpy())
        g = p.groupby(full)["Subject"].nunique()
        nullfull.append(int((g > 1).sum()))
    f["identical_six_field_groups_null_mean"] = round(float(np.mean(nullfull)), 3)
    f["identical_six_field_groups_null_p_ge_obs"] = round(float(np.mean(np.array(nullfull) >= len(full_groups))), 3)

    def study(s):
        return "+".join(x for x, S in [("tdcsfog", S_td), ("defog", S_de), ("daily", S_da)] if s in S)

    def desc(s):
        r = subj[subj["Subject"] == s]
        return f"{s} [{study(s)}; " + "; ".join(
            f"visit {x['Visit'] or 'blank'}: " + " ".join(f"{c}={x[c]}" for c in full) for _, x in r.iterrows()) + "]"
    amb_ids = collections.defaultdict(list)
    k = 0
    flagged_pairs = set()
    for ids in full_groups:
        k += 1
        aid = f"kaggle-{k}"
        inm = "yes" if all(x in S_man for x in ids) else ("partly" if any(x in S_man for x in ids) else "no")
        recs = {x: de.loc[de["Subject"] == x, ["Id", "Visit", "Medication"]].values.tolist() for x in ids}
        ev = ("identical subject-visit row on six fields: " + " vs ".join(desc(x) for x in ids)
              + f"; groups this large under the permutation null: mean {f['identical_six_field_groups_null_mean']}; "
              + "defog recordings: " + "; ".join(f"{x}: {recs[x]}" for x in ids) + "; the recordings differ (no shared signal windows)")
        add_amb(sid, aid, [SOURCES[sid] + x for x in ids], inm, "identical clinical row under two subject hashes", "strong", ev,
                "kept as separate persons; assign to the same split fold",
                "the competition hosts (tlvmc data page) or Salomon et al. 2024 (PMC11156937): duplicated person or copied row")
        for x in ids:
            amb_ids[x].append(f"{aid}: identical visit row with {', '.join(SOURCES[sid] + y for y in ids if y != x)}; same fold advised")
        flagged_pairs.add(tuple(sorted(ids)))
    # complementary visits under two hashes (one subject has visit 1 only, the other visit 2 only)
    v1 = {s for s, v in sv.items() if v == ("1",)}
    v2 = {s for s, v in sv.items() if v == ("2",)}
    comp = []
    for a in sorted(v2):
        ra = one[one["Subject"] == a].iloc[0]
        for b in sorted(v1):
            rb = one[one["Subject"] == b].iloc[0]
            if (ra["Age"], ra["Sex"], ra["YearsSinceDx"]) == (rb["Age"], rb["Sex"], rb["YearsSinceDx"]) and tuple(sorted([a, b])) not in flagged_pairs:
                comp.append((b, a))
    f["visit2_only_subjects"] = len(v2)
    f["visit1_only_subjects"] = len(v1)
    f["complementary_visit_pairs_same_age_sex_yearsdx"] = len(comp)
    # near misses: visit-2-only against visit-1-only hashes, same sex, age and years since diagnosis each within one year
    # (not exact); recorded, not flagged, because the age field never changes between the two visits of one hash
    f["two_visit_subjects"] = int(sum(1 for v in sv if len(v) == 2))
    near = []
    for a in sorted(v2):
        ra = one[one["Subject"] == a].iloc[0]
        for b in sorted(v1):
            rb = one[one["Subject"] == b].iloc[0]
            da_, dy_ = abs(float(ra["Age"]) - float(rb["Age"])), abs(float(ra["YearsSinceDx"]) - float(rb["YearsSinceDx"]))
            if ra["Sex"] == rb["Sex"] and da_ <= 1 and dy_ <= 1 and (da_, dy_) != (0.0, 0.0):
                near.append(f"{b} (visit 1 only, age {rb['Age']}, {rb['YearsSinceDx']} years, in manifest: {b in S_man}) vs "
                            f"{a} (visit 2 only, age {ra['Age']}, {ra['YearsSinceDx']} years, in manifest: {a in S_man})")
    f["complementary_visit_near_pairs_not_flagged"] = near
    for b, a in comp:
        k += 1
        aid = f"kaggle-{k}"
        ids = [b, a]
        inm = "yes" if all(x in S_man for x in ids) else ("partly" if any(x in S_man for x in ids) else "no")
        ev = ("one hash holds visit 1 only, the other visit 2 only, same age, sex and years since diagnosis: "
              + " vs ".join(desc(x) for x in ids) + f"; {len(v2)} subjects in subjects.csv have visit 2 only")
        add_amb(sid, aid, [SOURCES[sid] + x for x in ids], inm, "complementary visits under two subject hashes", "moderate", ev,
                "kept as separate persons; assign to the same split fold",
                "the competition hosts or Salomon et al. 2024 (PMC11156937)")
        for x in ids:
            amb_ids[x].append(f"{aid}: visit {'1' if x == b else '2'} only, same age/sex/years since diagnosis as {SOURCES[sid] + (a if x == b else b)}; same fold advised")
        flagged_pairs.add(tuple(sorted(ids)))
    # remaining age/sex/years-since-diagnosis coincidences: weak, listed once
    tg = one.groupby(["Age", "Sex", "YearsSinceDx"])["Subject"].apply(sorted)
    weak = []
    for ids in tg:
        if len(ids) < 2:
            continue
        for a, b in itertools.combinations(ids, 2):
            if tuple(sorted([a, b])) not in flagged_pairs:
                weak.append((a, b))
    f["weak_pairs_same_age_sex_yearsdx"] = len(weak)
    for a, b in weak:
        k += 1
        aid = f"kaggle-{k}"
        inm = "yes" if a in S_man and b in S_man else ("partly" if a in S_man or b in S_man else "no")
        ev = ("same age, sex and years since diagnosis, clinical scores differ: " + " vs ".join(desc(x) for x in [a, b])
              + f"; the permutation null expects {f['pairs_same_age_sex_yearsdx_null_mean']} such pairs among the "
              + f"{len(one)} subjects, {obs} observed")
        add_amb(sid, aid, [SOURCES[sid] + a, SOURCES[sid] + b], inm, "same age, sex, years since diagnosis", "weak", ev,
                "kept as separate persons; no fold constraint", "the competition hosts")
        for x, y in [(a, b), (b, a)]:
            amb_ids[x].append(f"{aid} (weak): same age/sex/years since diagnosis as {SOURCES[sid] + y}")
    # signal overlap across the 926 ingested recordings
    recs = []
    for folder, kind in [("train/tdcsfog", "tdcsfog"), ("test/tdcsfog", "tdcsfog"), ("train/defog", "defog"), ("test/defog", "defog")]:
        meta = td if kind == "tdcsfog" else de
        i2s = dict(zip(meta["Id"], meta["Subject"]))
        for i in files[folder]:
            recs.append((f"{folder}/{i}", SOURCES[sid] + i2s[i],
                         (lambda p=f"{R}/{folder}/{i}.csv": pd.read_csv(p, usecols=["AccV", "AccML", "AccAP"]).to_numpy())))
    f["overlap_records_equal_manifest_records"] = bool(len(recs) == len(man))
    ov, st = overlap_scan(sid, recs)
    f["overlap_scan"] = st
    for row in AMB:
        if row["source"] == sid and row["kind"].startswith("identical clinical"):
            ids = set(row["ids"].split(";"))
            sh = [r for r in ov if {r["subject_a"], r["subject_b"]} == ids]
            if sh:
                row["evidence"] = row["evidence"].replace("the recordings differ (no shared signal windows)",
                                                          f"shared signal windows between their recordings: {sum(r['shared_windows'] for r in sh)}")
    for r in ov:
        if not r["same_subject_ns"]:
            k += 1
            add_amb(sid, f"kaggle-{k}", [r["subject_a"], r["subject_b"]], "yes", "shared signal segment", "strong",
                    f"{r['record_a']} and {r['record_b']} share {r['shared_windows']} signal windows", "same fold", "raw files")
    # cohort statistics against the paper text
    pa = {g: (float(a), float(b), int(lo), int(hi)) for g, a, b, lo, hi in
          re.findall(r"(tDCS|DeFOG|combined) ([\d.]+) \+/- ([\d.]+) \[(\d+)-(\d+)\]", f["paper_age_statement"])}
    pf = {g: float(p) for g, p in re.findall(r"(tDCS|DeFOG|combined) ([\d.]+)% F", f["paper_sex_statement"])}
    groups = {"tDCS: tdcsfog subjects": ("tDCS", S_td), "DeFOG: defog or daily subjects": ("DeFOG", S_de | S_da),
              "DeFOG: defog subjects": ("DeFOG", S_de), "combined: all subjects.csv": ("combined", S_all),
              "combined: tdcsfog or defog": ("combined", S_td | S_de)}
    for name, (g, S) in groups.items():
        x = one[one["Subject"].isin(S)]
        age = x["Age"].astype(float)
        p = 100.0 * (x["Sex"] == "F").mean()
        COHORT.append({"source": sid, "group_definition": name, "n": int(len(x)), "female_code": "F", "female_pct": round(p, 1),
                       "paper_female_pct": pf.get(g), "female_pct_matches_paper": bool(round(p, 1) == pf.get(g)),
                       "age_mean": round(age.mean(), 2), "age_sd": round(age.std(ddof=1), 2),
                       "paper_age_mean": pa.get(g, (None,) * 4)[0], "paper_age_sd": pa.get(g, (None,) * 4)[1],
                       "age_mean_matches_paper": bool(round(age.mean(), 1) == pa.get(g, (None,) * 4)[0]),
                       "age_sd_matches_paper": bool(round(age.std(ddof=1), 1) == pa.get(g, (None,) * 4)[1]),
                       "_raw": (age.mean(), age.std(ddof=1))})
    # map
    out = []
    for s in sorted(S_man):
        mr = man[man["subject_ns"] == SOURCES[sid] + s]
        meds = sorted(set(td.loc[td["Subject"] == s, "Medication"]) | set(de.loc[de["Subject"] == s, "Medication"]))
        vis = sorted(set(td.loc[td["Subject"] == s, "Visit"]) | set(de.loc[de["Subject"] == s, "Visit"]), key=natkey)
        kinds = mr["kind"].value_counts().to_dict()
        ev = (f"subjects.csv study {study(s)}, visit rows {list(sv.get(s, ()))}; manifest records "
              + ", ".join(f"{kk} {vv}" for kk, vv in sorted(kinds.items()))
              + f"; recording visits {vis}; medication {meds}; all under one subject hash")
        out.append({"source": sid, "subject_ns": SOURCES[sid] + s, "person_id": f"{sid}:{s}", "rule": "kaggle_subject_hash",
                    "evidence": ev, "ambiguity": " | ".join(amb_ids.get(s, []))})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    strong_mod = [r for r in AMB if r["source"] == sid and r["strength"] in ("strong", "moderate") and r["in_manifest"] == "yes"]
    f["flagged_strong_or_moderate_in_manifest"] = len(strong_mod)
    f["persons_lower_bound_if_flags_merged"] = f["persons"] - len(strong_mod)
    f["gap_explanation"] = (f"{f['subjects_csv_unique']} subjects in subjects.csv, {len(S_man)} ingested: "
                            f"{len(daily_only)} have daily-living recordings only and {f['not_ingested_notype_only']} have defog "
                            f"recordings only in train/notype, neither folder ingested")
    f["multi_session"] = ("visits, medication ON and OFF and repeated tests all sit under one subject hash; "
                          "the hash is the person key of the competition data")
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= 5. cap_sleep
def read_hyp(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    patient = next((l.split("\t", 1)[1].strip() for l in lines if l.startswith("Patient:") and "\t" in l), "")
    start = next((i for i, l in enumerate(lines) if l.startswith("Sleep Stage")), None)
    if start is None:
        return patient, {}
    hdr = lines[start].split("\t")
    ti = next((i for i, c in enumerate(hdr) if c.startswith("Time")), None)
    ei = next((i for i, c in enumerate(hdr) if c.strip() == "Event"), None)
    if ti is None or ei is None:
        return patient, {}
    ep = {}
    for l in lines[start + 1:]:
        c = l.split("\t")
        if len(c) > max(ti, ei) and c[ei].startswith("SLEEP-"):
            ep[c[ti]] = c[ei]
    return patient, ep


def do_cap():
    sid = "cap_sleep"
    log("==", sid)
    base = DS + "/eeg/cap_sleep"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    raw = sorted((x for x in os.listdir(base + "/raw") if not x.startswith("._")), key=natkey)
    edfs = [x[:-4] for x in raw if x.endswith(".edf")]
    f["edf_on_disk"] = len(edfs)
    f["edf_on_disk_by_prefix"] = dict(collections.Counter(re.sub(r"\d+$", "", x) for x in edfs))
    f["st_annotation_files_by_prefix"] = dict(collections.Counter(re.sub(r"\d+\.edf\.st$", "", x) for x in raw if x.endswith(".edf.st")))
    aud = read_csv(base + "/processed/eeg_ingest_audit.csv")
    refused = aud[aud["routing"].str.startswith("REFUSED")]
    f["ingest_audit_rows"] = int(len(aud))
    f["refused"] = refused["subject_ns"].tolist()
    for _, r in refused.iterrows():
        NOT_ING.append({"source": sid, "native_id": r["subject_ns"], "reason": f"{r['routing']} ({r['file']})"})
    # what the MoveAtlas datacard states (read-only), against what the manifests hold
    dc = open(base + "/datacard.md", encoding="utf-8").read()
    rx = lambda p: (re.search(p, dc).group(1).strip() if re.search(p, dc) else "")
    f["datacard_population"] = rx(r"population: ([^\n]+)")
    f["datacard_fetch_subset"] = rx(r"--cap-subset ([\w,]+)")
    f["datacard_eeg_records_subjects_stated"] = rx(r"(\d+ canonical EEG records / \d+ subjects \([^)]*\))")
    f["datacard_chin_emg_records_stated"] = int(rx(r"plus (\d+) chin-EMG records") or -1)
    f["emg_manifest_records"] = int((man["source"] == "cap_sleep_chin").sum())
    f["emg_manifest_ids"] = sorted(man.loc[man["source"] == "cap_sleep_chin", "subject_ns"].tolist(), key=natkey)
    hd = {}
    for i, e in enumerate(edfs):
        h = edf_header(f"{base}/raw/{e}.edf")
        h["prefix_sha1"] = data_prefix_hash(f"{base}/raw/{e}.edf", h["header_bytes"])
        pat, ep = read_hyp(f"{base}/raw/{e}.txt")
        h["hyp_patient"], h["hyp_epochs"] = pat, ep
        hd[e] = h
    f["edf_patient_field_nonblank"] = int(sum(1 for h in hd.values() if h["patient_field"]))
    # the hypnogram's Patient field against the record name
    norm = lambda s: re.sub(r"\s+", "", s).lower()
    f["hypnogram_patient_equals_record_name"] = int(sum(1 for e, h in hd.items() if norm(h["hyp_patient"]) == e))
    f["hypnogram_patient_other"] = {e: h["hyp_patient"] for e, h in hd.items() if norm(h["hyp_patient"]) != e}
    # duplicates: equal data prefix or equal size, then full hash
    groups = collections.defaultdict(list)
    for e, h in hd.items():
        groups[("p", h["prefix_sha1"])].append(e)
        groups[("s", h["size"])].append(e)
    cand = sorted({tuple(sorted(v, key=natkey)) for v in groups.values() if len(v) > 1})
    full = {}
    for c in cand:
        for e in c:
            if e not in full:
                log("  full sha256", e)
                full[e] = sha256_file(f"{base}/raw/{e}.edf")
    f["candidate_groups_equal_size_or_prefix"] = [list(c) for c in cand]
    ident = collections.defaultdict(list)
    for e, hsh in full.items():
        ident[hsh].append(e)
    ident = [sorted(v, key=natkey) for v in ident.values() if len(v) > 1]
    f["byte_identical_edf_groups"] = ident
    same_start = collections.defaultdict(list)
    for e, h in hd.items():
        same_start[(h["startdate"], h["starttime"], h["n_records"])].append(e)
    f["groups_same_start_and_length"] = [sorted(v, key=natkey) for v in same_start.values() if len(v) > 1]
    person = {e: e for e in edfs}
    amb_ids = {}
    for k, g in enumerate(ident, 1):
        keep = g[0]
        comp = []
        for e in g[1:]:
            person[e] = keep
            a, b = hd[keep]["hyp_epochs"], hd[e]["hyp_epochs"]
            common = sorted(set(a) & set(b))
            agree = sum(1 for t in common if a[t] == b[t])
            comp.append(f"{e}: {len(b)} scored epochs, {len(common)} at times shared with {keep}, stage agreement {agree} of {len(common)}")
            f[f"hypnogram_{keep}_vs_{e}"] = {"epochs_a": len(a), "epochs_b": len(b), "common_times": len(common), "agree": agree,
                                             "first_epoch_a": next(iter(a), ""), "first_epoch_b": next(iter(b), "")}
        h = hd[keep]
        aid = f"cap-{k}"
        ev = (f"raw EDF files {', '.join(g)} are byte-identical ({h['size']} bytes, sha256 {full[keep][:16]}); "
              f"header start {h['startdate']} {h['starttime']}, {h['n_records']} records of {h['record_s']:g} s; "
              f"hypnograms: {keep} {len(h['hyp_epochs'])} scored epochs; " + "; ".join(comp)
              + f"; hypnogram Patient fields: " + ", ".join(repr(hd[e]['hyp_patient']) for e in g))
        add_amb(sid, aid, [SOURCES[sid] + e for e in g], "yes", "byte-identical recording under two record names", "resolved",
                ev, f"merged: one person ({sid}:{keep}); the second record adds no new signal and should not count as a second person",
                "PhysioNet capslpdb maintainers, for whether the patient named in the second hypnogram has a missing recording")
        for e in g:
            amb_ids[e] = (f"{aid}: EDF byte-identical to {', '.join(SOURCES[sid] + x for x in g if x != e)}; merged into {sid}:{keep}")
    man_ids = sorted({s.split(":", 1)[1] for s in man["subject_ns"]}, key=natkey)
    f["manifest_ids"] = len(man_ids)
    f["manifest_ids_equal_edf_minus_refused"] = bool(set(man_ids) == set(edfs) - {x.split(":", 1)[1] for x in f["refused"]})
    out = []
    for e in man_ids:
        h = hd[e]
        ev = (f"raw/{e}.edf: header start {h['startdate']} {h['starttime']} (dates anonymized to 1 January), {h['n_records']} "
              f"records of {h['record_s']:g} s, {h['n_signals']} signals, patient field blank; hypnogram Patient '{h['hyp_patient']}'; "
              f"one night per record name")
        out.append({"source": sid, "subject_ns": SOURCES[sid] + e, "person_id": f"{sid}:{person[e]}",
                    "rule": "identical_edf_merged" if person[e] != e or e in amb_ids else "record_name",
                    "evidence": ev, "ambiguity": amb_ids.get(e, "")})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    f["persons_lower_bound_if_flags_merged"] = f["persons"]
    f["gap_explanation"] = (f"the registry's {f['registry_n_subjects_now']} is the whole PhysioNet database; on disk "
                            f"{f['edf_on_disk']} EDF of the rbd and n groups ({f['edf_on_disk_by_prefix']}); "
                            f"{len(f['refused'])} refused at ingest ({', '.join(f['refused'])})")
    f["multi_session"] = "one night per record name; no record name repeats a night except the byte-identical pair"
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= 6. fog_turning_filho
def do_fog():
    sid = "fog_turning_filho"
    log("==", sid)
    base = DS + "/pose/fog_turning_filho"
    man, f = manifest_part(sid)
    registry_facts(sid, f)
    z = zipfile.ZipFile(base + "/raw/IMU.zip")
    txt = sorted(i.filename for i in z.infolist() if i.filename.endswith(".txt") and not os.path.basename(i.filename).startswith("._"))
    f["imu_txt_files"] = len(txt)
    dup = collections.Counter((i.CRC, i.file_size) for i in z.infolist() if not i.is_dir())
    f["byte_identical_zip_member_groups"] = int(sum(1 for v in dup.values() if v > 1))
    tri = []
    arrays = {}
    for n in txt:
        m = re.match(r"IMU/SUB(\d+)_(\w+)\.txt$", n)
        d = pd.read_csv(io.BytesIO(z.read(n)), sep="\t", encoding="utf-8")
        flag = d.columns[-1]
        arrays[n] = d.iloc[:, 2:5].to_numpy(dtype=np.float64)
        rate = 1.0 / float(np.median(np.diff(d.iloc[:, 1].to_numpy(dtype=float))))
        row = {"file": n, "sub": int(m.group(1)), "trial": m.group(2), "samples": len(d), "rate": rate}
        if "flag" in flag.lower() and m.group(2) != "standing":
            x = d[flag].to_numpy().astype(int)
            row["runs"] = int(((x[1:] == 1) & (x[:-1] == 0)).sum() + (x[0] == 1))
            row["fog_s"] = float(x.sum() / rate)
        tri.append(row)
    tr = pd.DataFrame(tri)
    turning = tr[tr["trial"] != "standing"].copy()
    turning["session"] = turning["trial"].astype(int)
    f["imu_subjects"] = int(tr["sub"].nunique())
    f["imu_turning_trials"] = int(len(turning))
    f["imu_standing_trials"] = int((tr["trial"] == "standing").sum())
    # PDFEinfo
    info = pd.read_csv(base + "/raw/PDFEinfo.csv", sep=";", dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")
    info.columns = [re.sub(r"\s+", " ", c.replace(chr(0xFEFF), "")).strip() for c in info.columns]
    info = info[info["ID"].str.startswith("PDFE")].copy()
    info["num"] = info["ID"].str[4:].astype(int)
    f["pdfe_rows"] = int(len(info))
    epc = {int(re.match(r"Session (\d)", c).group(1)): c for c in info.columns if re.match(r"Session \d - numbers of FoG episodes", c)}
    tmc = {int(re.match(r"Session (\d)", c).group(1)): c for c in info.columns if re.match(r"Session \d - total time in FoG", c)}
    sesc = next(c for c in info.columns if c.lower().startswith("sessions"))

    def fnum(x):
        try:
            return float(str(x).strip())
        except ValueError:
            return None
    prof = {}
    for _, r in info.iterrows():
        prof[r["num"]] = {k: (fnum(r[epc[k]]), fnum(r[tmc[k]])) for k in epc}
    info["sessions_declared"] = info[sesc].astype(int)
    info["sessions_with_data"] = info["num"].map(lambda n: sum(1 for v in prof[n].values() if v[0] is not None))
    imu_sessions = turning.groupby("sub").size()
    info["imu_turning_trials"] = info["num"].map(lambda n: int(imu_sessions.get(n, 0)))
    f["pdfe_ids_equal_imu_sub_numbers"] = bool(sorted(info["num"]) == sorted(tr["sub"].unique()))
    f["subjects_imu_trials_equal_sessions_with_data"] = int((info["imu_turning_trials"] == info["sessions_with_data"]).sum())
    f["subjects_imu_trials_equal_sessions_declared"] = int((info["imu_turning_trials"] == info["sessions_declared"]).sum())
    f["subjects_sessions_declared_not_equal_with_data"] = info.loc[info["sessions_declared"] != info["sessions_with_data"], "ID"].tolist()

    nwith = dict(zip(info["num"], info["sessions_with_data"]))

    def match(sub, n):
        """(all trials agree, trials that agree): FoG episode count equal and FoG time within 1 s, per session."""
        ok = 0
        for _, t in turning[turning["sub"] == sub].iterrows():
            ep, tm = prof[n].get(t["session"], (None, None))
            if ep is None or tm is None or int(ep) != t["runs"] or abs(tm - t["fog_s"]) > 1.0:
                return False, ok
            ok += 1
        return True, ok
    uniq, own_ok, ties, trials_ok = 0, 0, {}, 0
    for sub in sorted(turning["sub"].unique()):
        ntr = int((turning["sub"] == sub).sum())
        # a candidate row must also hold the same number of sessions with data as the subject has trials
        hits = [n for n in prof if nwith[n] == ntr and match(sub, n)[0]]
        own, k = match(sub, sub)
        trials_ok += k
        own_ok += int(own)
        if hits == [sub]:
            uniq += 1
        elif own:
            ties[sub] = [n for n in hits if n != sub]
    f["trials_episode_count_and_fog_time_agree"] = int(trials_ok)
    f["subjects_whose_own_pdfe_row_matches_all_trials"] = own_ok
    f["subjects_matched_uniquely"] = uniq
    f["subjects_matched_with_ties"] = len(ties)
    f["ties"] = {f"SUB{s:02d}": [f"PDFE{n:02d}" for n in v] for s, v in ties.items()}
    f["tie_profiles"] = {f"SUB{s:02d}": turning.loc[turning["sub"] == s, ["session", "runs", "fog_s"]].values.round(2).tolist() for s in ties}
    f["tied_subjects_fog_free"] = int(sum(1 for s in ties if turning.loc[turning["sub"] == s, "runs"].sum() == 0))
    # cohort against the paper text
    sexs = f["paper_sex_statement"]
    mm = re.search(r"(\d+) females? and (\d+) males?", sexs)
    ag = re.search(r"from (\d+) to (\d+)", f["paper_age_statement"])
    COHORT.append({"source": sid, "group_definition": "PDFEinfo rows", "n": int(len(info)), "female_code": "F",
                   "female_pct": round(100.0 * (info["Gender"] == "F").mean(), 1),
                   "paper_female_pct": round(100.0 * int(mm.group(1)) / (int(mm.group(1)) + int(mm.group(2))), 1) if mm else None,
                   "female_pct_matches_paper": bool(mm and (info["Gender"] == "F").sum() == int(mm.group(1)) and (info["Gender"] == "M").sum() == int(mm.group(2))),
                   "age_mean": round(info["Age"].astype(float).mean(), 2), "age_sd": round(info["Age"].astype(float).std(ddof=1), 2),
                   "paper_age_mean": None, "paper_age_sd": None,
                   "age_mean_matches_paper": bool(ag and int(info["Age"].astype(int).min()) == int(ag.group(1)) and int(info["Age"].astype(int).max()) == int(ag.group(2))),
                   "age_sd_matches_paper": None, "_raw": (None, None)})
    f["pdfe_age_min_max"] = [int(info["Age"].astype(int).min()), int(info["Age"].astype(int).max())]
    # duplicate persons in PDFEinfo
    dcols = ["Age", "Weight (kg)", "Height (m)", "Gender", "Disease duration (years)"]
    dg = info.groupby(dcols)["ID"].apply(list)
    f["pdfe_pairs_same_age_weight_height_gender_duration"] = int(sum(1 for v in dg if len(v) > 1))
    # overlap scan
    recs = [(n, SOURCES[sid] + "SUB%02d" % int(re.match(r"IMU/SUB(\d+)", n).group(1)), (lambda a=arrays[n]: a)) for n in txt]
    ov, st = overlap_scan(sid, recs)
    f["overlap_scan"] = st
    man_ids = sorted({s.split(":", 1)[1] for s in man["subject_ns"]}, key=natkey)
    f["manifest_ids_equal_imu_subjects"] = bool(man_ids == ["SUB%02d" % s for s in sorted(tr["sub"].unique())])
    out = []
    for s in man_ids:
        n = int(s[3:])
        r = info[info["num"] == n].iloc[0]
        own, k = match(n, n)
        ev = (f"IMU.zip {s}: {int(imu_sessions.get(n, 0))} turning trials plus standing; PDFEinfo PDFE{n:02d}: sessions #="
              f"{r['sessions_declared']}, {r['sessions_with_data']} sessions with FoG data; FoG episode count and time agree for "
              f"{k} of {int(imu_sessions.get(n, 0))} trials" + (f", {'unique match' if n not in ties else 'tied with ' + ', '.join(f'PDFE{x:02d}' for x in ties[n])}")
              + "; every session of the patient shares this id")
        out.append({"source": sid, "subject_ns": SOURCES[sid] + s, "person_id": f"{sid}:PDFE{n:02d}",
                    "rule": "sub_number_equals_pdfe_number", "evidence": ev, "ambiguity": ""})
    m = pd.DataFrame(out)
    f["persons"] = int(m["person_id"].nunique())
    f["persons_lower_bound_if_flags_merged"] = f["persons"]
    f["gap_explanation"] = (f"none now: registry {f['registry_n_subjects_now']}, {len(man_ids)} ids; probe_cohorts read "
                            f"{f['registry_n_subjects_probe_cohorts']} at MoveAtlas {HSS_PROV['moveatlas_head'][:7]}, before the registry correction")
    f["multi_session"] = "up to three sessions (days) per patient plus a standing trial, all under one SUBnn id"
    MAPS[sid] = m
    FACTS[sid] = f


# ================================================================= run
for fn in [do_cervical, do_hd, do_ds002778, do_cap, do_fog, do_kaggle]:
    fn()

for sid, m in MAPS.items():
    m = m[["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"]]
    assert m["subject_ns"].is_unique
    write_csv(m, f"map_{sid}.csv")

AMBDF = pd.DataFrame(AMB, columns=["amb_id", "source", "ids", "in_manifest", "kind", "strength", "evidence", "decision", "settle_by"])
write_csv(AMBDF, "small_sources_ambiguities.csv")
write_csv(pd.DataFrame(NOT_ING, columns=["source", "native_id", "reason"]), "small_sources_not_ingested.csv")
write_csv(pd.DataFrame(OVERLAP, columns=["source", "record_a", "subject_a", "record_b", "subject_b", "shared_windows",
                                         "windows_a", "windows_b", "same_subject_ns"]), "small_sources_overlap_pairs.csv")
# papers print one decimal, some by truncation: test that reading too, from the unrounded statistic
for r in COHORT:
    raw = r.pop("_raw")
    for k, v in zip(("mean", "sd"), raw):
        pv = r.get(f"paper_age_{k}")
        r[f"age_{k}_matches_paper_truncated"] = (None if pv is None or v is None
                                                 else bool(round(np.floor(v * 10 + 1e-9) / 10, 1) == pv))
write_csv(pd.DataFrame(COHORT), "small_sources_cohort_checks.csv")

counts = []
for sid in SOURCES:
    f = FACTS[sid]
    a = AMBDF[AMBDF["source"] == sid]
    counts.append({"source": sid, "registry_n_subjects_now": f["registry_n_subjects_now"],
                   "registry_n_subjects_probe_cohorts": f["registry_n_subjects_probe_cohorts"],
                   "canonical_subject_ids": f["canonical_subject_ids"], "canonical_records": f["records_pooled_manifests"],
                   "persons": f["persons"], "persons_lower_bound_if_flags_merged": f["persons_lower_bound_if_flags_merged"],
                   "ids_merged": f["canonical_subject_ids"] - f["persons"],
                   "flags_strong": int((a["strength"] == "strong").sum()), "flags_moderate": int((a["strength"] == "moderate").sum()),
                   "flags_weak": int((a["strength"] == "weak").sum()), "flags_resolved": int((a["strength"] == "resolved").sum()),
                   "rule": MAPS[sid]["rule"].value_counts().idxmax(), "gap_explanation": f["gap_explanation"],
                   "multi_session": f["multi_session"]})
CNT = pd.DataFrame(counts)
write_csv(CNT, "small_sources_counts.csv")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


with open(os.path.join(OUT, "small_sources_facts.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(jsonable(FACTS), fh, indent=1, ensure_ascii=False)
log("wrote small_sources_facts.json")

# ---------------------------------------------------------------- summary
HEAD = moveatlas_head(ROOT)
L = []
w = L.append
w("# Person ids of six small sources (plan task T0.2c)")
w("")
w(f"Measured by scripts/probe_person_ids_small_sources.py against MoveAtlas {HEAD}. Every number below is in "
  "small_sources_facts.json or one of the CSV files of this folder. One map per source: map_<source>.csv, one row per "
  "canonical subject id, columns source, subject_ns, person_id, rule, evidence, ambiguity.")
w("")
w("| source | registry now | registry in probe_cohorts | canonical ids | records | persons | lower bound if flags merged | strong | moderate | weak | resolved |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in CNT.iterrows():
    w(f"| {r['source']} | {r['registry_n_subjects_now']} | {r['registry_n_subjects_probe_cohorts']} | {r['canonical_subject_ids']} | "
      f"{r['canonical_records']} | {r['persons']} | {r['persons_lower_bound_if_flags_merged']} | {r['flags_strong']} | "
      f"{r['flags_moderate']} | {r['flags_weak']} | {r['flags_resolved']} |")
w("")
w(f"Persons in the six maps: {int(CNT['persons'].sum())} for {int(CNT['canonical_subject_ids'].sum())} canonical ids; "
  f"{int(CNT['persons_lower_bound_if_flags_merged'].sum())} if every strong or moderate flag were one person. "
  "The maps keep flagged ids as separate persons, except a byte-identical recording, which is merged; "
  "the ambiguity column says which ids should share a split fold.")
w("")
w("Duplicate checks on the signals: byte hashes (zip CRC for the zipped sources, SHA-256 for EDF and BDF, a hash of the first "
  "8 MB after the header for every EDF), and for the accelerometer sources a content-defined window hash (64-row windows at "
  "anchors where the row hash is 0 mod 64, windows with fewer than 16 distinct rows skipped) over every ingested recording, "
  "which finds a shared segment of a few hundred samples at any offset. The cervical EMG and quaternion CSVs and the EEG "
  "files were checked by byte hash only. Positive control: the scan finds the "
  f"{int(sum(1 for r in OVERLAP if r['same_subject_ns']))} known within-id copy (kaggle_fog test file of a train recording, "
  "small_sources_overlap_pairs.csv). Synthetic control (a segment of one hd_accel_jnet recording copied into another at "
  f"random offsets, seed {RNG_SEED}), detections by segment length: "
  f"{FACTS['hd_accel_jnet']['window_scan_injection_test']['detected']}.")
w("")
for sid in SOURCES:
    f = FACTS[sid]
    a = AMBDF[AMBDF["source"] == sid]
    w(f"## {sid}")
    w("")
    w(f"- Registry n_subjects '{f['registry_n_subjects_now']}' (docs/datasets_index.csv now), {f['canonical_subject_ids']} canonical ids in "
      f"{f['records_pooled_manifests']} records in the pooled manifests, {f['manifest_rows']} manifest rows in all "
      f"({', '.join(f'{k}: {v}' for k, v in f['manifest_rows_per_file'].items())}), "
      f"source labels {f['manifest_source_labels']}.")
    w(f"- Count difference: {f['gap_explanation']}.")
    w(f"- Several sessions per person: {f['multi_session']}.")
    w(f"- Persons: {f['persons']} (rule {MAPS[sid]['rule'].value_counts().idxmax()}); lower bound if every strong or moderate flag "
      f"were one person: {f['persons_lower_bound_if_flags_merged']}.")
    if "overlap_scan" in f:
        o = f["overlap_scan"]
        w(f"- Window scan: {o['records_scanned']} recordings, {o['windows_total']} windows, {o['cross_subject_pairs']} pairs of "
          f"different ids sharing windows, {o['same_subject_pairs']} pairs within one id.")
    if sid == "cervical_motion_multiplanar":
        w(f"- {f['zip_csv_files']} CSV files in {f['zip_subject_folders']} folders; partA codes {f['codes_with_partA']}, partB codes "
          f"{f['codes_with_partB']}, both parts {f['codes_both_parts']}; partA only {f['codes_partA_only']}, partB only "
          f"{f['codes_partB_only']}. The PartB spreadsheet lists the same codes as the partB folders: "
          f"{f['partB_sheet_codes_equal_partB_folders']}. Byte-identical CSV groups: {f['byte_identical_csv_groups']}. "
          f"Folders whose file names carry another date: {f['folders_with_file_date_not_folder_date']}.")
    if sid == "hd_accel_jnet":
        w(f"- Groups in metadata.csv: {f['metadata_group_counts']}; rows with clinical scores (TMS, FA or TFC) per group: "
          f"{f['rows_with_clinical_scores_by_group']}. The cohort checks (small_sources_cohort_checks.csv) test which HD "
          "group reproduces the paper's women percentage.")
        w(f"- Women counts that give the paper's whole-number percentage, per group size, whatever the sex code means: "
          f"{f['paper_pct_attainable_female_counts']}. An empty list means that group size cannot produce the paper's figure.")
        w(f"- Same-sex pairs with birthdates within 31 days: {f['same_sex_pairs_birthdate_within_31_days']}, of which identical "
          f"birthdate: {f['pairs_identical_birthdate']} (uniform-birthday expectation over the observed birth-year range: "
          f"{f['expected_identical_birthdate_pairs_uniform']} pairs).")
    if sid == "pd_rest_eeg_onoff_ds002778":
        w(f"- participants.tsv {f['participants_by_group']}, women {f['female_by_group']}; id numbers absent from the "
          f"release: {f['id_numbers_absent']}. PD with both ON and OFF: {f['pd_with_on_and_off']}; same calendar day: "
          f"{f['pd_on_off_same_calendar_day']}; days between ON and OFF (min, median, max) over the "
          f"{f['pd_on_off_pairs_with_trusted_dates']} pairs with recording dates: {f['pd_on_off_days_apart_min_median_max_trusted']}. "
          f"Sessions dated outside {f['acq_modal_year']}: {f['sessions_dated_outside_modal_year']}, "
          f"{f['of_which_participant_note_says_preprocessed_file']} of them noted in participants.tsv as rebuilt from a preprocessed "
          f"EEGLAB file (a conversion date, not a second visit). "
          f"BDF start date equals scans.tsv date in {f['bdf_startdate_equals_scans_acq_date']} of {f['session_files']} files; "
          f"byte-identical BDF groups {f['byte_identical_bdf_groups']}; identical acquisition times {f['identical_acq_time_groups']}.")
        w(f"- BDF patient fields that are not blank: {f['bdf_patient_field_nonblank_files']}. The same string in the two rebuilt "
          "ON files of participants of different gender is a converter placeholder, not a person key.")
    if sid == "kaggle_fog":
        w(f"- subjects.csv {f['subjects_csv_rows']} rows, {f['subjects_csv_unique']} subjects; study sets {f['study_sets']}. "
          f"Not ingested: {f['not_ingested_daily_only']} daily-only, {f['not_ingested_notype_only']} notype-only "
          f"({f['defog_subjects_with_only_notype_recordings']} defog subjects have only notype recordings, "
          f"{f['of_which_ingested_through_test_defog_file']} of them enters through the test/defog copy: "
          f"{f['subjects_ingested_through_test_defog_file']}); gap fully accounted: {f['gap_accounted']}.")
        w(f"- Manifest records by kind {f['manifest_records_by_kind']}; tasks present twice {f['manifest_duplicate_tasks']}; "
          f"test files equal to their train copies: {f['test_files_equal_train_copies']}.")
        w(f"- Visit sets in subjects.csv {f['subjects_csv_visit_sets']}. Pairs of subjects with the same age, sex and years since "
          f"diagnosis: {f['pairs_same_age_sex_yearsdx_observed']} observed, {f['pairs_same_age_sex_yearsdx_null_mean']} expected "
          f"under the permutation null (p = {f['pairs_same_age_sex_yearsdx_null_p_ge_obs']}; {f['permutation_null']}). "
          f"Subject-visit rows identical on six fields under two hashes: "
          f"{f['subject_visit_rows_identical_on_six_fields_across_subjects']} (null mean {f['identical_six_field_groups_null_mean']}). "
          f"Complementary-visit pairs: {f['complementary_visit_pairs_same_age_sex_yearsdx']}.")
        w(f"- Near complementary pairs (visit 1 only against visit 2 only, same sex, age and years since diagnosis each within one "
          f"year, not both equal), recorded and not flagged because the age field is constant within every one of the "
          f"{f['two_visit_subjects']} two-visit hashes (age constant within subject: {f['age_constant_within_subject']}): "
          f"{f['complementary_visit_near_pairs_not_flagged']}.")
        kc = [r for r in COHORT if r["source"] == sid]
        w(f"- Cohort statistics of the source paper reproduced by "
          f"{sum(1 for r in kc if r['female_pct_matches_paper'] and (r['age_mean_matches_paper'] or r['age_mean_matches_paper_truncated']))} "
          f"of the {len(kc)} groupings of subjects.csv tried, reading the paper's decimals as rounded or truncated "
          "(small_sources_cohort_checks.csv): the paper's analysis set is not the subjects.csv roster, so the paper cannot "
          "confirm the subject count here.")
    if sid == "cap_sleep":
        w(f"- EDF on disk {f['edf_on_disk_by_prefix']}; refused at ingest {f['refused']}; byte-identical EDF groups "
          f"{f['byte_identical_edf_groups']}; groups with the same header start and length {f['groups_same_start_and_length']}. "
          f"Hypnogram Patient field equals the record name in {f['hypnogram_patient_equals_record_name']} of {f['edf_on_disk']} "
          f"(the others, blank or without number: {f['hypnogram_patient_other']}).")
        w(f"- Annotation (.edf.st) files on disk by group: {f['st_annotation_files_by_prefix']}; only the rbd and n groups have EDF. "
          f"Datacard (read-only): population '{f['datacard_population']}', fetch subset '{f['datacard_fetch_subset']}', "
          f"'{f['datacard_eeg_records_subjects_stated']}', chin-EMG records stated {f['datacard_chin_emg_records_stated']} "
          f"against {f['emg_manifest_records']} in the EMG manifest ({f['emg_manifest_ids']}).")
        for k, v in f.items():
            if k.startswith("hypnogram_") and "_vs_" in k and isinstance(v, dict):
                w(f"- {k}: first scored epoch {v['first_epoch_a']} and {v['first_epoch_b']}, {v['common_times']} epoch times in "
                  f"common, stage agreement {v['agree']} of {v['common_times']}: both hypnograms score the same night.")
    if sid == "fog_turning_filho":
        w(f"- IMU.zip {f['imu_subjects']} subjects, {f['imu_turning_trials']} turning and {f['imu_standing_trials']} standing trials; "
          f"PDFEinfo {f['pdfe_rows']} rows. SUBnn is PDFEnn: the FoG episode count and time of every turning trial agree with "
          f"the PDFE row of the same number in {f['trials_episode_count_and_fog_time_agree']} of {f['imu_turning_trials']} trials; "
          f"{f['subjects_matched_uniquely']} subjects match only their own row among rows with the same number of sessions, "
          f"{f['subjects_matched_with_ties']} tie with other rows ({f['ties']}); tied subjects without any flagged FoG in any "
          f"trial: {f['tied_subjects_fog_free']} of {f['subjects_matched_with_ties']} (a FoG-free profile cannot tell patients apart). "
          f"IMU trial count equals the sessions with data for "
          f"{f['subjects_imu_trials_equal_sessions_with_data']} of {f['pdfe_rows']} subjects; declared session count differs from "
          f"the sessions with data for {f['subjects_sessions_declared_not_equal_with_data']}.")
    if len(a):
        w("")
        w("| flag | strength | ids | kind | decision |")
        w("|---|---|---|---|---|")
        for _, r in a.iterrows():
            w(f"| {r['amb_id']} | {r['strength']} | {r['ids'].replace(';', ', ')} | {r['kind']} | {r['decision']} |")
    w("")
w("## Unresolved")
w("")
un = AMBDF[AMBDF["strength"].isin(["strong", "moderate"])]
for _, r in un.iterrows():
    w(f"- {r['amb_id']} ({r['strength']}, in manifest: {r['in_manifest']}): {r['ids'].replace(';', ', ')}. Settled by: {r['settle_by']}.")
w(f"- Weak flags ({int((AMBDF['strength'] == 'weak').sum())}) are listed in small_sources_ambiguities.csv; they impose no fold constraint.")
w("")
w("## Cohort checks against the source-paper text")
w("")
w("| source | group | n | female % | paper female % | match | age mean | age sd | paper mean | paper sd | mean match | sd match | mean match (truncated) | sd match (truncated) |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in COHORT:
    w(f"| {r['source']} | {r['group_definition']} (female code {r['female_code']}) | {r['n']} | {r['female_pct']} | {r['paper_female_pct']} | "
      f"{r['female_pct_matches_paper']} | {r['age_mean']} | {r['age_sd']} | {r['paper_age_mean']} | {r['paper_age_sd']} | "
      f"{r['age_mean_matches_paper']} | {r['age_sd_matches_paper']} | {r['age_mean_matches_paper_truncated']} | "
      f"{r['age_sd_matches_paper_truncated']} |")
w("")
w("For fog_turning_filho the age match column tests the paper's age range (minimum and maximum), not a mean. The truncated "
  "columns read the paper's one-decimal figure as the statistic cut after one decimal instead of rounded.")
with open(os.path.join(OUT, "small_sources_summary.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L) + "\n")
log("wrote small_sources_summary.md")

prov = {"script": os.path.basename(sys.argv[0]), "moveatlas_root": ROOT, "moveatlas_head": HEAD,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
        "seed": RNG_SEED, "permutations": N_PERM, "manifest_files": MAN_FILES, "runtime_s": round(time.time() - T0, 1),
        "outputs": sorted(["map_%s.csv" % s for s in SOURCES] + ["small_sources_counts.csv", "small_sources_ambiguities.csv",
                           "small_sources_not_ingested.csv", "small_sources_overlap_pairs.csv", "small_sources_cohort_checks.csv",
                           "small_sources_facts.json", "small_sources_summary.md"])}
with open(os.path.join(OUT, "_provenance_probe_person_ids_small_sources.json"), "w", encoding="utf-8", newline="\n") as fh:
    json.dump(prov, fh, indent=1)
log("provenance:", HEAD, "| done")
