"""Person-identity audit of the harmonized in-scope sources and the MD_FM person map (plan task T0.2d).

Read-only on MoveAtlas; writes only under reports/person_ids/ of this tree.

1. Loads the pooled canonical manifests and the per-source extras (datasets/_canonical_*manifest*.csv),
   drops the CODY rows of those manifests on read (CODY is evaluation-only: no CODY data file,
   label or count is opened, used or written here), maps manifest source names to registry ids
   and checks them against reports/data_scope/data_scope.csv.
2. Audits every source that no sibling task owns (T0.2a to T0.2c own monipar, li_taati, amass,
   infant_pose_chambers, sensefog, cervical_motion_multiplanar, hd_accel_jnet,
   pd_rest_eeg_onoff_ds002778, kaggle_fog, cap_sleep, fog_turning_filho; modys_video has its own
   map): suffix tokens in subject_ns that could split a person, normalized-id collisions, the
   same person under the same subject_ns in every modality manifest, subject codes embedded in
   take or task names, and the participant table or file list of the raw deposit where one
   exists (the CARE-PD pickles are read by opcode disassembly with pickletools: nothing is
   unpickled). Each audited source gets a verdict and a person rule.
3. Cross-source identity: a scan of site and investigator names over the datacards, the registry
   rows and the documentation files of the raw deposits, plus curated candidate groups whose
   evidence lines are quoted from those files at run time; writes cross_source_overlap.md/.csv.
4. Combiner (re-runnable): every reports/person_ids/map_<source>.csv present, the modys_video
   child map (reports/modys_subjects/modys_video_person_map.csv), and the rules of step 2, into
   person_map.csv (source, subject_ns, person_id, rule, evidence, ambiguity), person_counts.csv
   (source, subject_ids, persons, rule) and person_map_summary.md. A sibling source without a
   map file stays pending with an empty person_id, so that no split silently uses a wrong
   identity. Re-run after the sibling tasks finish:

   <conda>/python.exe scripts/build_person_map.py --moveatlas <moveatlas>

Other outputs: audit_sources.csv, audit_subjects.csv, raw_checks.csv, cross_source_name_scan.csv,
_provenance.json (shared folder: the last script to run owns it) and
_provenance_build_person_map.json (this script's own copy).
"""
import csv
import glob
import html
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
    sys.argv += ["--out", os.path.join(TREE, "reports", "person_ids")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parse_args, write_provenance, read_csv  # noqa: E402

import pandas as pd  # noqa: E402

ARGS = parse_args("Person-identity audit and MD_FM person map (plan task T0.2d)")
MA = ARGS.moveatlas
DS = MA + "/datasets"
OUT = ARGS.out.replace("\\", "/")
PROV = write_provenance(ARGS, {"task": "T0.2d"})
with open(os.path.join(OUT, "_provenance_build_person_map.json"), "w", encoding="utf-8", newline="\n") as _f:
    json.dump(PROV, _f, indent=1)

# Manifest source names that differ from the registry ids.
ALIAS = {"cap_sleep_chin": "cap_sleep", "cervical_motion": "cervical_motion_multiplanar",
         "semg_walking": "physionet_semg_walking", "seizeit2_emg": "seizeit2", "ntu": "ntu_rgbd",
         "modys": "modys_video"}
# Sources owned by sibling tasks (their map files are combined, never re-derived here).
SIBLING = {"monipar": "T0.2a", "li_taati": "T0.2a", "amass": "T0.2b", "infant_pose_chambers": "T0.2b",
           "sensefog": "T0.2b", "cervical_motion_multiplanar": "T0.2c", "hd_accel_jnet": "T0.2c",
           "pd_rest_eeg_onoff_ds002778": "T0.2c", "kaggle_fog": "T0.2c", "cap_sleep": "T0.2c",
           "fog_turning_filho": "T0.2c", "modys_video": "probe_modys_subjects.py"}
CODY_NS = ("adult", "cody", "pedia")


def clean(s):
    """House style for every text written: no em-dash, no en-dash, single line."""
    s = str(s).replace("\u2014", ", ").replace("\u2013", "-").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def log(*a):
    print(*a, flush=True)


ID_COLS = {"source", "subject_ns", "person_id"}  # data values: written exactly as read, never restyled


def write_csv(name, header, rows):
    keep = [h in ID_COLS for h in header]
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([clean(x) if isinstance(x, str) and not k else x for x, k in zip(r, keep)])
    log(f"wrote {name}: {len(rows)} rows")


def cell(x):
    return str(x).replace("|", "/")


def write_md(name, lines):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(l.replace("\u2014", ", ").replace("\u2013", "-") for l in lines) + "\n")
    log(f"wrote {name}")


def zip_names(path):
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist() if not os.path.basename(n.rstrip("/")).startswith("._")
                and "__MACOSX" not in n]


def zip_read(path, member):
    with zipfile.ZipFile(path) as zf:
        return zf.read(member).decode("utf-8", "replace")


def listdir(path):
    return sorted(x for x in os.listdir(path) if not x.startswith("._"))


# ---------------------------------------------------------------- 1. manifests and scope
log("== 1. manifests")
MANIFEST_FILES = sorted(f.replace("\\", "/") for f in glob.glob(DS + "/_canonical_*manifest*.csv")
                        if not os.path.basename(f).startswith("._"))
frames, dropped_sources = [], set()
for f in MANIFEST_FILES:
    d = read_csv(f)
    cody = d["source"].str.lower().str.startswith("cody")
    dropped_sources |= set(d.loc[cody, "source"].unique())
    d = d[~cody]
    ns = d["subject_ns"].str.split(":").str[0].str.lower()
    bad = ns.isin(CODY_NS) | ns.str.startswith("ext_ds")
    if bad.any():
        sys.exit(f"{os.path.basename(f)}: a non-CODY source carries a CODY namespace; stop and report")
    cols = ["source", "subject_ns"] + [c for c in ("take", "task", "sequence") if c in d.columns]
    d = d[cols].copy()
    d["manifest"] = os.path.basename(f)
    frames.append(d)
    log(f"  {os.path.basename(f)}: {len(d)} non-CODY rows")
M = pd.concat(frames, ignore_index=True).fillna("")
for c in ("take", "task", "sequence"):
    if c not in M.columns:
        M[c] = ""
M["src"] = M["source"].map(lambda s: ALIAS.get(s, s))
M["local"] = M["subject_ns"].str.split(":", n=1).str[1].fillna("")
SOURCES = sorted(M["src"].unique())

scope = read_csv(os.path.join(TREE, "reports", "data_scope", "data_scope.csv"))
harm = scope[(scope["harmonized"] == "True") & scope["scope"].isin(["published", "usable_not_published"])]
HARM_IDS = set(harm["id"])
not_in_scope = [s for s in SOURCES if s not in HARM_IDS]
harm_absent = sorted(i for i in HARM_IDS if i not in SOURCES and not i.startswith("cody"))
MINE = [s for s in SOURCES if s not in SIBLING]
log(f"  sources (non-CODY): {len(SOURCES)}; audited here: {len(MINE)}; sibling-owned: {len(SOURCES) - len(MINE)}")
log(f"  manifest sources outside the harmonized in-scope list: {not_in_scope}; in-scope harmonized absent: {harm_absent}")

REG = {r["id"]: r for r in csv.DictReader(open(MA + "/docs/datasets_index.csv", encoding="utf-8"))}
SRC_DIR = {s: MA + "/" + REG[s]["path"] for s in SOURCES if s in REG}

# ---------------------------------------------------------------- 2. per-source audit
log("== 2. per-source audit")
SEPS = r"[:_\-\s/.()]"
TOKEN_RX = {
    "s_number": re.compile(r"(?:^|[:_\-\s])s\d+$", re.I),
    "session": re.compile(r"ses(?:sion)?[-_]?\d+", re.I),
    "visit": re.compile(r"(?:visit|(?:^|[:_\-\s])v\d+(?=$|[:_\-\s]))", re.I),
    "trial_run": re.compile(r"(?:trial|run[-_]?\d+|rep[-_]?\d+|(?:^|[:_\-\s])[tr]\d+(?=$|[:_\-\s]))", re.I),
    "task_word": re.compile(r"(?:walk|rest|task|tug|turn|drink|commun|balance)", re.I),
    "side": re.compile(r"(?:^|[:_\-\s])(?:l|r|left|right|lh|rh)(?=$|[:_\-\s])", re.I),
    "night_day": re.compile(r"(?:night|(?:^|[:_\-\s])(?:n\d+|day\d*|d\d+)(?=$|[:_\-\s]))", re.I),
    "med_state": re.compile(r"(?:^|[:_\-\s])(?:on|off|onmed|offmed|postmed|premed)(?=$|[:_\-\s])", re.I),
    "timepoint": re.compile(r"(?:baseline|follow|(?:^|[:_\-\s])(?:pre|post|base|fu\d*|m\d+|w\d+)(?=$|[:_\-\s]))", re.I),
    "clip_video": re.compile(r"(?:clip\d+|vid\d+|video)", re.I),
    "study_prefix": re.compile(r"(?:^|[:_\-\s])(?:db\d+|dne\d+|study\d+)(?=$|[:_\-\s])", re.I),
    "annotation": re.compile(r"[()]"),
}


def tokens_of(local):
    if not re.search(SEPS, local):  # a bare code (S01, GaPt03, 1001) has no suffix to carry
        return []
    return [k for k, rx in TOKEN_RX.items() if rx.search(local)]


def norm_id(local):
    s = re.sub(r"\(.*?\)", "", local).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return re.sub(r"(?<!\d)0+(?=\d)", "", s)


# Curated reading of each audited source: which tokens are expected and what they mean, the
# verdict when every automated check passes, and the within-source ambiguity (if any).
JUDGE = {
    "capture24": dict(verdict="identity"),
    "care_pd": dict(verdict="identity_flagged", expect={"s_number": "DNE ids are dne<k>_s<nnn>, the top-level anonymized_subject_id key of the CARE-PD pickle (README data structure), so s<nnn> names a subject, not a session",
                                                          "study_prefix": "dne1 and dne2 name the two DNE sub-studies"}),
    "clinical_gait_multipatho": dict(verdict="identity"),
    "daphnet": dict(verdict="identity"),
    "drummer_tremor_model": dict(verdict="identity"),
    "fog_star": dict(verdict="identity"),
    "gait120": dict(verdict="identity"),
    "gaitndd": dict(verdict="identity"),
    "gaitrec": dict(verdict="identity"),
    "goodwin_smm": dict(verdict="identity"),
    "imu_inthewild_iprognosis": dict(verdict="identity"),
    "koa_pd_nm": dict(verdict="identity"),
    "ltmm": dict(verdict="identity"),
    "motion_x": dict(verdict="video_stem", expect="*",
                     collisions="clip names that differ only in case or in an underscore (Thumbs_Up and Thumbs_up) are distinct clips of the deposit, each with its own take; performer identity is not recorded either way"),
    "multimodal_fog": dict(verdict="identity"),
    "ninapro": dict(verdict="identity_flagged", expect={"s_number": "ids are db<k>_s<n>: s<n> is the subject number inside Ninapro database <k>",
                                                          "study_prefix": "db1 and db2 name two Ninapro databases"}),
    "nonan_gaitprint": dict(verdict="identity"),
    "nonan_gaitprint_young": dict(verdict="identity"),
    "ntu_rgbd": dict(verdict="identity"),
    "pads": dict(verdict="identity"),
    "pd_biostamprc21": dict(verdict="identity"),
    "pd_rest_eeg_ds004584": dict(verdict="identity"),
    "pd_walking_eeg_ds007526": dict(verdict="identity"),
    "physionet_gaitpdb": dict(verdict="identity_flagged"),
    "physionet_multimodal_gait": dict(verdict="identity"),
    "physionet_semg_walking": dict(verdict="identity"),
    "remap_open": dict(verdict="identity"),
    "seizeit2": dict(verdict="identity"),
    "sixmwt_lifespan": dict(verdict="identity"),
    "stroke_gait_vancriekinge": dict(verdict="identity", expect={"s_number": "ids are <cohort>:S<nnn>: S<nnn> is the subject number inside the able_bodied or stroke cohort"}),
    "vidimu": dict(verdict="identity"),
    "weargait_pd": dict(verdict="identity", expect={"annotation": "some NLS ids carry the literal suffix ' (control)' (count in the raw-table note); no plain twin id exists (collision check), so it is a label inside the id, not a second id"}),
}


# --- subject codes embedded in take or task names ------------------------------------------
def care_pd_expected(take, sns):
    coh = sns.split(":")[1]
    if coh == "BMCLab":
        m = re.match(r"(?:Trimmed_)?(SUB\d+)_", take)
        return f"care_pd:BMCLab:{m.group(1)}" if m else "?"
    if coh == "E-LC":
        m = re.match(r"FOG-(\d+)-", take)
        return f"care_pd:E-LC:{int(m.group(1))}" if m else "?"
    if coh == "PD-GaM":
        m = re.match(r"(\d+)-", take)
        return f"care_pd:PD-GaM:{m.group(1)}" if m else "?"
    if coh in ("T-SDU", "T-SDU-PD", "T-LTC"):
        m = re.match(r"(P\d+)_day", take)
        return f"care_pd:{coh}:{m.group(1)}" if m else "?"
    if coh == "DNE":
        m = re.match(r"(dne\d)_", take)
        return sns if (m and sns.split(":")[2].startswith(m.group(1) + "_")) else f"care_pd:DNE:{m.group(1) if m else '?'}_?"
    return None


def _m(rx, s, fmt):
    m = re.search(rx, s)
    return fmt(m) if m else "?"


EMBED_EQ = {  # source -> (column, function(take_or_task, subject_ns) -> expected subject_ns or None)
    "care_pd": ("take", care_pd_expected),
    "physionet_gaitpdb": ("task", lambda t, s: "gaitpdb:" + re.sub(r"_\d+$", "", t)),
    "gaitndd": ("task", lambda t, s: "gaitndd:" + t),
    "clinical_gait_multipatho": ("task", lambda t, s: "clinical_gait:" + re.sub(r"_\d+$", "", t)),
    "daphnet": ("task", lambda t, s: _m(r"^(S\d+)R\d+$", t, lambda m: "daphnet:" + m.group(1))),
    "vidimu": ("take", lambda t, s: "vidimu:" + t.split("_")[0]),
    "ntu_rgbd": ("take", lambda t, s: _m(r"P(\d{3})", t, lambda m: "ntu:P" + m.group(1))),
    # study1 tasks carry URI-<nnn>-, study2 tasks <nnn>-<date>: both must land on goodwin_smm:<nnn>
    "goodwin_smm": ("task", lambda t, s: _m(r"(?:URI-|study2_)(\d{3})-", t, lambda m: "goodwin_smm:" + m.group(1))),
    "stroke_gait_vancriekinge": ("task", lambda t, s: _m(r"^gait_SUBJ(\d+)(?:_|$)", t, lambda m: "stroke_gait:able_bodied:S%03d" % int(m.group(1))) if t.startswith("gait_SUBJ") else None),
    "motion_x": ("take", lambda t, s: ":".join(s.split(":")[:2]) + ":" + t),
    "nonan_gaitprint": ("task", lambda t, s: s if t.lower().startswith("g03_") else "group code " + t[:3]),
    "nonan_gaitprint_young": ("task", lambda t, s: s if t.lower().startswith("g01_") else "group code " + t[:3]),
}
EMBED_NEST = {  # source -> (column, regex whose group 1 is a session or video code nested in one subject)
    "gaitrec": ("task", r"^s(\d+)_t"),
    "care_pd": ("take", r"^(vid\d+|dne\d_\d+)_"),
}


def embedded_check(src, g):
    out = {}
    if src in EMBED_EQ:
        col, fn = EMBED_EQ[src]
        rec = g.drop_duplicates(["subject_ns", col])
        exp = [fn(t, s) for t, s in zip(rec[col], rec["subject_ns"])]
        used = [(e, s) for e, s in zip(exp, rec["subject_ns"]) if e is not None]
        bad = [(e, s) for e, s in used if e != s]
        out["eq_pairs_checked"] = len(used)
        out["eq_pairs_skipped"] = len(exp) - len(used)
        out["eq_mismatch"] = len(bad)
        out["eq_mismatch_examples"] = "; ".join(f"{s} <- {e}" for e, s in bad[:5])
    if src in EMBED_NEST:
        col, rx = EMBED_NEST[src]
        codes = g[col].str.extract(rx, expand=False)
        h = pd.DataFrame({"code": codes, "s": g["subject_ns"]}).dropna().drop_duplicates()
        per = h.groupby("code")["s"].nunique()
        out["nested_codes"] = int(len(per))
        out["nested_codes_in_several_subjects"] = int((per > 1).sum())
    return out


# --- participant tables and file lists of the raw deposits ---------------------------------
def rc_gaitrec(g):
    p = SRC_DIR["gaitrec"] + "/raw/GRF_metadata.csv"
    d = read_csv(p, usecols=["SUBJECT_ID", "SESSION_ID"])
    ses = d.drop_duplicates("SESSION_ID").set_index("SESSION_ID")["SUBJECT_ID"]
    t = g.assign(ses=g["task"].str.extract(r"^s(\d+)_t", expand=False))
    t = t.dropna(subset=["ses"]).drop_duplicates(["subject_ns", "ses"])
    agree = int((("gaitrec:" + t["ses"].map(ses).fillna("?")) == t["subject_ns"]).sum())
    multi = int((d.groupby("SESSION_ID")["SUBJECT_ID"].nunique() > 1).sum())
    return ({"gaitrec:" + s for s in d["SUBJECT_ID"]}, "GRF_metadata.csv SUBJECT_ID",
            f"{d['SESSION_ID'].nunique()} sessions, {multi} of them under more than one SUBJECT_ID; "
            f"{agree} of {len(t)} manifest (subject, session) pairs agree with the metadata",
            "session to subject disagreement" if (multi or agree < len(t)) else "")


def rc_gaitpdb(g):
    z = SRC_DIR["physionet_gaitpdb"] + "/raw/gaitpdb.zip"
    files = {"gaitpdb:" + m.group(1) for n in zip_names(z) if (m := re.search(r"/((?:Ga|Ju|Si)(?:Co|Pt)\d+)_\d+\.txt$", n))}
    t = zip_read(z, "gait-in-parkinsons-disease-1.0.0/demographics.txt")
    demo = {"gaitpdb:" + l.split("\t")[0].strip() for l in t.splitlines()[1:] if l.split("\t")[0].strip()}
    st = Counter(i.split(":")[1][:4] for i in files)
    return files, "gaitpdb.zip gait-file codes <study><Co|Pt><nn>_<k>.txt",\
        ("file codes per study and group " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) +
         f"; demographics.txt rows {len(demo)}, rows without a gait file: {', '.join(sorted(demo - files)) or 'none'}"
         f", gait-file codes without a demographics row: {', '.join(sorted(files - demo)) or 'none'}")


def rc_gaitndd(g):
    z = SRC_DIR["gaitndd"] + "/raw/gaitndd.zip"
    t = zip_read(z, "gait-in-neurodegenerative-disease-database-1.0.0/subject-description.txt")
    ids = {"gaitndd:" + l.split("\t")[0].strip() for l in t.splitlines()[1:] if l.split("\t")[0].strip()}
    return ids, "gaitndd.zip subject-description.txt first column", ""


def rc_ltmm(g):
    base = SRC_DIR["ltmm"] + "/raw/long-term-movement-monitoring-database-1.0.0"
    names = [(n, "") for n in listdir(base) if n.lower().endswith(".hea")]
    names += [(n, "LabWalks") for n in listdir(base + "/LabWalks") if n.lower().endswith(".hea")]
    ids = set()
    for n, _ in names:
        m = re.match(r"^(co|fl)(\d{3})", n, re.I)
        if m:
            ids.add(f"ltmm:{m.group(1).upper()}{m.group(2)}")
    lab = {re.match(r"^(co|fl)(\d{3})", n, re.I).group(0).upper() for n, w in names if w and re.match(r"^(co|fl)(\d{3})", n, re.I)}
    top = {re.match(r"^(co|fl)(\d{3})", n, re.I).group(0).upper() for n, w in names if not w and re.match(r"^(co|fl)(\d{3})", n, re.I)}
    refused = []
    audit = SRC_DIR["ltmm"] + "/processed/ltmm_ingest_audit.csv"
    if os.path.isfile(audit):
        with open(audit, encoding="utf-8", newline="") as f:
            for r in csv.reader(f):
                if r and any("REFUSED" in x for x in r) and ("ltmm:" + r[0].upper()) not in set(g["subject_ns"]):
                    refused.append(r[0])
    return ids, "WFDB .hea names, top folder and LabWalks (case-folded)", \
        (f"{len(top)} ids with a 3-day record, {len(lab)} with a lab walk, {len(top & lab)} with both; "
         f"records refused by processed/ltmm_ingest_audit.csv for ids absent from the manifest: {', '.join(sorted(set(refused))) or 'none'}")


def rc_daphnet(g):
    z = SRC_DIR["daphnet"] + "/raw/daphnet.zip"
    names = zip_names(z)
    ids = {"daphnet:" + m.group(1) for n in names if (m := re.search(r"/(S\d+)R\d+\.txt$", n))}
    doc = next((n for n in names if n.endswith("doc/documentation.html")), None)
    m = re.search(r"all runs of one\s+subject\s+should\s+be\s+combined", re.sub(r"<[^>]+>", " ", zip_read(z, doc))) if doc else None
    note = f"{doc.split('/', 1)[-1]}: \"{clean(m.group(0))}\"" if m else "no statement on combining runs found in the deposit documentation"
    return ids, "daphnet.zip S<ss>R<rr>.txt names", note


def rc_capture24(g):
    ids = {"capture24:" + m.group(1) for n in zip_names(SRC_DIR["capture24"].replace("/socle/", "/imu/") + "/raw/capture24.zip")
           if (m := re.search(r"(P\d{3})\.csv\.gz$", n))}
    return ids, "capture24.zip P<nnn>.csv.gz names", ""


def rc_pads(g):
    ids = {"pads:" + m.group(1) for n in listdir(SRC_DIR["pads"] + "/raw/patients") if (m := re.match(r"patient_(\d+)\.json$", n))}
    return ids, "raw/patients/patient_<nnn>.json names", ""


def rc_biostamp(g):
    d = read_csv(SRC_DIR["pd_biostamprc21"] + "/raw/Clinic_DataPDBioStampRCStudy.csv")
    st = Counter(d["Status"])
    return ({"pd_biostamprc21:" + s.zfill(3) for s in d["ID"]}, "Clinic_DataPDBioStampRCStudy.csv ID",
            "Status column: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))


def rc_bids(src, ns_prefix, key_col=None, group_col=None):
    def f(g):
        d = pd.read_csv(SRC_DIR[src] + "/raw/participants.tsv", sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        ids = {ns_prefix + p.replace("sub-", "") for p in d["participant_id"]}
        note, issue = "", ""
        if key_col and key_col in d.columns:
            note = f"{key_col}: {d[key_col].nunique()} distinct values over {len(d)} rows"
            if d[key_col].nunique() < len(d):
                within = int(d.duplicated([group_col, key_col]).sum()) if group_col in d.columns else int(d.duplicated([key_col]).sum())
                if group_col in d.columns:
                    note += (f"; {key_col} repeats within one {group_col}: {within} (each group is numbered from its own start); "
                             f"({group_col}, {key_col}) is unique on {len(d.drop_duplicates([group_col, key_col]))} rows")
                if within:
                    issue = f"{within} repeated {key_col} values within a group"
        return ids, "participants.tsv participant_id", note, issue
    return f


def _table_ids(path, col="Subject ID"):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    h = next(i for i, r in enumerate(rows) if col in r)
    j = rows[h].index(col)
    return [r[j].strip() for r in rows[h + 1:] if len(r) > j and r[j].strip() and r[j].strip() != "-"]


def rc_weargait(g):
    base = SRC_DIR["weargait_pd"] + "/raw/Version 1/"
    pdi = _table_ids(base + "PD - Demographic+Clinical - datasetV1.csv")
    coi = _table_ids(base + "CONTROLS - Demographic+Clinical - datasetV1.csv")
    both = set(pdi) & set(coi)
    ann = int(g["subject_ns"].drop_duplicates().str.contains(r"\(").sum())
    return ({"weargait:" + s for s in pdi + coi}, "Version 1 PD and CONTROLS Demographic+Clinical tables, Subject ID",
            f"PD table {len(set(pdi))} ids, controls table {len(set(coi))} ids, {len(both)} ids in both; "
            f"{ann} manifest ids carry a parenthetical suffix, compared without it",
            f"{len(both)} ids in both tables" if both else "")


def rc_nonan_old(g):
    z = SRC_DIR["nonan_gaitprint"] + "/raw/subject_trial_characteristics.zip"
    d = pd.read_csv(io.StringIO(zip_read(z, "subject_trial_characteristics/Gaitprint_subject_characteristics.csv")), dtype=str, keep_default_na=False)
    return ({"nonan_gaitprint:" + s for s in d["id"]}, "Gaitprint_subject_characteristics.csv id",
            "group column: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(d["group"]).items())))


def rc_nonan_young(g):
    d = pd.read_excel(SRC_DIR["nonan_gaitprint_young"] + "/raw/GaitPrint_Subject_Characteristics.xlsx")
    return ({"nonan_gaitprint_young:S%03d" % int(x) for x in d["id"]}, "GaitPrint_Subject_Characteristics.xlsx id",
            "group column: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(d["group"].astype(str)).items())))


def rc_fog_star(g):
    d = read_csv(SRC_DIR["fog_star"] + "/raw/clinical_data.csv")
    return {"fog_star:" + s for s in d["subjectID"]}, "clinical_data.csv subjectID", ""


def rc_gait120(g):
    d = pd.read_excel(SRC_DIR["gait120"] + "/raw/Subject_Demographics.xlsx")
    return {"gait120:" + str(s) for s in d["Subject"]}, "Subject_Demographics.xlsx Subject", ""


def rc_sixmwt(g):
    ids = {"sixmwt_lifespan:S%02d" % int(m.group(1)) for n in zip_names(SRC_DIR["sixmwt_lifespan"] + "/raw/6MWT_IMU_Dataset.zip")
           if (m := re.search(r"/Subject (\d+)/$", n))}
    return ids, "6MWT_IMU_Dataset.zip 'Subject <n>' folders", ""


def rc_clinical_gait(g):
    z = SRC_DIR["clinical_gait_multipatho"] + "/raw/dataset.zip"
    names = zip_names(z)
    ids, meta_bad, sessions = set(), 0, defaultdict(set)
    with zipfile.ZipFile(z) as zf:
        for n in names:
            m = re.match(r"dataset/data/[^/]+/[^/]+/([^/]+)/[^/]+/[^/]+_meta\.json$", n)
            if not m:
                continue
            ids.add("clinical_gait:" + m.group(1))
            j = json.loads(zf.read(n).decode("utf-8", "replace"))
            if j.get("subject") != m.group(1):
                meta_bad += 1
            sessions[m.group(1)].add(j.get("session"))
    multi = sum(1 for v in sessions.values() if len(v) > 1)
    return ids, "dataset.zip subject folders and every *_meta.json",\
        f"{sum(1 for n in names if n.endswith('_meta.json'))} meta files, {meta_bad} whose 'subject' differs from the folder; {multi} subjects with more than one session, all under one id",\
        (f"{meta_bad} meta files disagree with their folder" if meta_bad else "")


def rc_ninapro(g):
    ids = {"ninapro:db%s_s%s" % (m.group(1), m.group(2)) for n in listdir(SRC_DIR["ninapro"] + "/raw") if (m := re.match(r"^DB(\d)_s(\d+)\.zip$", n))}
    return ids, "raw DB<k>_s<n>.zip names", ""


def rc_semg(g):
    base = [d for d in listdir(SRC_DIR["physionet_semg_walking"] + "/raw") if os.path.isdir(SRC_DIR["physionet_semg_walking"] + "/raw/" + d)][0]
    ids = {"semg_walking:" + m.group(1) for n in listdir(SRC_DIR["physionet_semg_walking"] + "/raw/" + base) if (m := re.match(r"^(S\d+)\.hea$", n))}
    return ids, "WFDB S<n>.hea names (the source mixes S01 with S2..S31)", ""


def rc_pmg(g):
    base = SRC_DIR["physionet_multimodal_gait"] + "/raw/Dataset"
    ids, per = set(), defaultdict(set)
    for sub in ("EEG_Data/EEG.csv", "EMG_Data", "FP_Data"):
        if os.path.isdir(base + "/" + sub):
            for n in listdir(base + "/" + sub):
                m = re.search(r"(?:^|_)S(\d+)_", n)
                if m:
                    ids.add("physionet_multimodal_gait:S" + str(int(m.group(1))))
                    per[sub].add(m.group(1))
    return ids, "EEG, EMG and force-plate file names (subject-info.csv not read: it holds dates of birth)",\
        ", ".join(f"{k} {len(v)}" for k, v in per.items())


def rc_drummer(g):
    ids = {"drummer:" + m.group(1) for n in listdir(SRC_DIR["drummer_tremor_model"] + "/raw") if (m := re.match(r"^(S\d+)_", n))}
    return ids, "raw S<nn>_*.c3d names", ""


def rc_remap(g):
    per = defaultdict(set)
    for n in zip_names(SRC_DIR["remap_open"] + "/raw/remap_open.zip"):
        m = re.search(r"(?:^|/)Pt(\d+)_(PD|C)_", n)
        if m:
            per["Pt" + m.group(1)].add(m.group(2))
    two = sum(1 for v in per.values() if len(v) > 1)
    grp = Counter(next(iter(v)) for v in per.values() if len(v) == 1)
    return ({"remap_open:" + k for k in per}, "remap_open.zip Pt<ID>_<PD|C>_ episode names",
            "group per id: " + ", ".join(f"{k} {v}" for k, v in sorted(grp.items())) + f"; {two} ids under both groups",
            f"{two} ids under both groups" if two else "")


def rc_koa(g):
    ids, per_folder = set(), defaultdict(set)
    for root, _, files in os.walk(SRC_DIR["koa_pd_nm"] + "/raw/KOA-PD-NM"):
        for n in files:
            m = re.match(r"^(\d+)_(KOA|PD|NM)_(\d+)(?:_([A-Z]+))?\.MOV$", n, re.I)
            if m and not n.startswith("._"):
                ids.add(f"koa_pd_nm:{m.group(2).upper()}_{(m.group(4) or 'NA').upper()}_{m.group(1)}")
                per_folder[os.path.basename(root)].add(m.group(1))
    per_coh = Counter(i.split(":")[1].split("_")[0] for i in ids)
    restart = sum(1 for v in per_folder.values() if min(v) == "001")
    t = open(SRC_DIR["koa_pd_nm"] + "/raw/README.txt", encoding="utf-8", errors="replace").read()
    m = re.search(r"comprises of (\d+) subjects which involves (\d+) Knee Osteoarthritis \(KOA\), (\d+) Parkinson.s Disease \(PD\) and (\d+) Normal", t)
    readme = (f"README.txt states {m.group(1)} subjects ({m.group(2)} KOA, {m.group(3)} PD, {m.group(4)} NM)" if m
              else "README.txt subject totals not found")
    return ids, "KOA-PD-NM video names <num>_<cohort>_<seq>[_<severity>].MOV",\
        (f"ids per cohort (cohort, severity folder, subject number): " + ", ".join(f"{k} {v}" for k, v in sorted(per_coh.items())) +
         f"; subject numbers restart at 001 in {restart} of {len(per_folder)} folders; {readme}")


def rc_vidimu(g):
    base = SRC_DIR["vidimu"] + "/raw/dataset"
    ids = set()
    for sub in ("videonly", "videoandimus"):
        if os.path.isdir(base + "/" + sub):
            ids |= {"vidimu:" + n for n in listdir(base + "/" + sub) if re.match(r"^S\d+$", n)}
    return ids, "raw/dataset/videonly and videoandimus subject folders", ""


def rc_multimodal_fog(g):
    d = read_csv(MA + "/configs/multimodal_fog_patient_index.csv")
    ids = {"multimodal_fog:p" + m.group(1) for f in d["folder"] if (m := re.search(r"Filtered Data/(\d{3})", f))}
    ses = sorted({f for f in d["folder"] if re.search(r"Filtered Data/\d{3}/", f)})
    return ids, "configs/multimodal_fog_patient_index.csv folders (MoveAtlas, read-only)", "sub-session folders: " + ", ".join(ses)


PKL_KEYOPS = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BININT1", "BININT2", "BININT", "SHORT_BINSTRING", "BINSTRING"}


def pickle_subject_walks(path):
    """(subject key, walk key) pairs of a CARE-PD pickle read by opcode disassembly (pickletools.genops):
    nothing is unpickled, no object is built, no code runs. A walk key is a key whose value is a dict that
    starts with 'pose'; a subject key is a key whose value is a dict that starts with a walk key (README data
    structure: {anonymized_subject_id: {anonymized_walk_id: {pose, trans, beta, ...}}})."""
    import pickletools
    toks, memo, n, last = [], {}, 0, None
    with open(path, "rb") as f:
        for op, arg, _ in pickletools.genops(f):
            nm = op.name
            if nm == "MEMOIZE":
                memo[n] = last
                n += 1
                continue
            if nm in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[arg] = last
                continue
            if nm == "MARK":
                continue
            if nm in PKL_KEYOPS:
                tok = ("k", str(arg))
            elif nm in ("BINGET", "LONG_BINGET", "GET"):
                v = memo.get(arg)
                tok = ("k", v) if isinstance(v, str) else ("o", "get")
            elif nm == "EMPTY_DICT":
                tok = ("d", None)
            else:
                tok = ("o", nm)
            toks.append(tok)
            last = tok[1] if tok[0] == "k" else None
    walk_idx = {i for i in range(len(toks) - 2) if toks[i][0] == "k" and toks[i + 1][0] == "d" and toks[i + 2] == ("k", "pose")}
    pairs, cur = [], None
    for i in range(len(toks) - 2):
        if toks[i][0] == "k" and toks[i + 1][0] == "d" and (i + 2) in walk_idx:
            cur = toks[i][1]
        elif i in walk_idx:
            pairs.append((cur, toks[i][1]))
    return pairs


def rc_care_pd(g):
    raw = SRC_DIR["care_pd"] + "/raw/"
    pairs, subj = set(), set()
    for coh in sorted(set(g["subject_ns"].str.split(":").str[1])):
        for p in (raw + coh + ".pkl", raw + "Canonicalized_SMPL_pickles/" + coh + "_canonical.pkl"):
            if os.path.isfile(p):
                for s, w in pickle_subject_walks(p):
                    pairs.add((f"care_pd:{coh}:{s}", w))
                    subj.add(f"care_pd:{coh}:{s}")
    man = set(zip(g["subject_ns"], g["take"]))
    walk_subj = defaultdict(set)
    for s, w in pairs:
        walk_subj[(s.split(":")[1], w)].add(s)
    elsewhere = sum(1 for s, w in man - pairs if walk_subj.get((s.split(":")[1], w)))
    return subj, "the nine cohort pickles and their Canonicalized_SMPL_pickles twins, keys read by opcode disassembly (nothing unpickled)",\
        (f"{len(man & pairs)} of {len(man)} manifest (subject, walk) pairs found under the same subject key; "
         f"{elsewhere} manifest walks found under another subject key; {len(man - pairs) - elsewhere} not found"),\
        (f"{elsewhere} walks under another subject key" if elsewhere else "")


def rc_iprognosis(g):
    """subject_id values of tremor_dataset.pickle (a list of one dict per subject, dataset_description.txt), read by
    streaming opcode disassembly from the zip: nothing is unpickled."""
    import pickletools
    z = SRC_DIR["imu_inthewild_iprognosis"] + "/raw/i-PROGNOSIS_DS2.10_Sub1_IMUdataPhonecalls.zip"
    ints = {"BININT1", "BININT2", "BININT", "LONG1", "INT", "LONG"}
    vals, memo, n, last, want = [], {}, 0, None, False
    with zipfile.ZipFile(z) as zf, zf.open("tremor_dataset.pickle") as f:
        for op, arg, _ in pickletools.genops(f):
            nm = op.name
            if nm == "MEMOIZE":
                memo[n] = last
                n += 1
                continue
            if nm in ("BINPUT", "LONG_BINPUT", "PUT"):
                memo[arg] = last
                continue
            if nm in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                v = last = arg
            elif nm in ("BINGET", "LONG_BINGET", "GET"):
                v = last = memo.get(arg)
            else:
                if want and nm in ints:
                    vals.append(arg)
                if nm != "MARK":
                    want = False
                last = None
                continue
            want = v == "subject_id"
    desc = zip_read(z, "dataset_description.txt")
    m = re.search(r"group of (\d+) PD patients and (\d+) Healthy controls", desc)
    stated = f"; dataset_description.txt states {m.group(1)} PD and {m.group(2)} controls" if m else ""
    dup = len(vals) - len(set(vals))
    return ({f"imu_inthewild:{v}" for v in vals}, "tremor_dataset.pickle subject_id fields, opcode disassembly (nothing unpickled)",
            f"{len(vals)} subject dictionaries, {len(set(vals))} distinct subject_id values{stated}",
            f"{dup} repeated subject_id values" if dup else "")


def rc_stroke(g):
    """Subject folder names of the two C3D zips (names only: the C3D labels, which carry birthdate-like
    prefixes per the datacard, are never read)."""
    base = SRC_DIR["stroke_gait_vancriekinge"] + "/raw/"
    ids, per = set(), Counter()
    for z, rx, coh in (("138_HealthyPiG.zip", r"^[^/]+/SUBJ(\d+)/", "able_bodied"), ("50_StrokePiG.zip", r"^[^/]+/TVC(\d+)/", "stroke")):
        for n in zip_names(base + z):
            m = re.match(rx, n)
            if m:
                ids.add(f"stroke_gait:{coh}:S%03d" % int(m.group(1)))
        per[coh] = sum(1 for i in ids if f":{coh}:" in i)
    return ids, "138_HealthyPiG.zip SUBJ<nn> and 50_StrokePiG.zip TVC<nn> folder names (C3D labels not read)",\
        "folders per cohort: " + ", ".join(f"{k} {v}" for k, v in sorted(per.items()))


def rc_ntu(g):
    ids, n = set(), 0
    for z in sorted(glob.glob(SRC_DIR["ntu_rgbd"] + "/raw/nturgbd_skeletons_*.zip")):
        for name in zip_names(z):
            m = re.search(r"S\d{3}C\d{3}P(\d{3})R\d{3}A\d{3}\.skeleton$", name)
            if m:
                ids.add("ntu:P" + m.group(1))
                n += 1
    return ids, "nturgbd_skeletons_*.zip file names SxxxCxxxPxxxRxxxAxxx.skeleton", f"{n} skeleton files"


RAW_CHECKS = {
    "care_pd": rc_care_pd, "imu_inthewild_iprognosis": rc_iprognosis, "stroke_gait_vancriekinge": rc_stroke, "ntu_rgbd": rc_ntu,
    "gaitrec": rc_gaitrec, "physionet_gaitpdb": rc_gaitpdb, "gaitndd": rc_gaitndd, "ltmm": rc_ltmm,
    "daphnet": rc_daphnet, "capture24": rc_capture24, "pads": rc_pads, "pd_biostamprc21": rc_biostamp,
    "pd_rest_eeg_ds004584": rc_bids("pd_rest_eeg_ds004584", "pd_rest_eeg_ds004584:sub-", "ID", "GROUP"),
    "pd_walking_eeg_ds007526": rc_bids("pd_walking_eeg_ds007526", "pd_walking_eeg_ds007526:sub-", "subject_id"),
    "seizeit2": rc_bids("seizeit2", "seizeit2:"), "weargait_pd": rc_weargait,
    "nonan_gaitprint": rc_nonan_old, "nonan_gaitprint_young": rc_nonan_young, "fog_star": rc_fog_star,
    "gait120": rc_gait120, "sixmwt_lifespan": rc_sixmwt, "clinical_gait_multipatho": rc_clinical_gait,
    "ninapro": rc_ninapro, "physionet_semg_walking": rc_semg, "physionet_multimodal_gait": rc_pmg,
    "drummer_tremor_model": rc_drummer, "remap_open": rc_remap, "koa_pd_nm": rc_koa, "vidimu": rc_vidimu,
    "multimodal_fog": rc_multimodal_fog,
}
NO_RAW_CHECK = {
    "motion_x": "clip-indexed deposit without performer ids",
    "goodwin_smm": "URI-<nnn> codes of both studies checked in the task names",
}


def mfix(src, ids):
    """Manifest ids as compared to a raw table (only weargait carries a suffix to drop)."""
    if src == "weargait_pd":
        return {re.sub(r"\s*\(.*?\)\s*$", "", s) for s in ids}
    return set(ids)


AUDIT, SUBJ_ROWS, RAW_ROWS = {}, [], []
for src in SOURCES:
    g = M[M["src"] == src]
    ids = sorted(g["subject_ns"].unique())
    rec_per = g.groupby("subject_ns").size()
    man_per = g.groupby("subject_ns")["manifest"].agg(lambda s: "|".join(sorted(set(s))))
    per_manifest = {mf: set(gg["subject_ns"]) for mf, gg in g.groupby("manifest")}
    union = set(ids)
    a = {"source": src, "owner": SIBLING.get(src, "T0.2d"), "subject_ids": len(ids),
         "manifests": "; ".join(f"{mf} {len(s)}" for mf, s in sorted(per_manifest.items())),
         "missing_by_manifest": "; ".join(f"{mf} lacks {', '.join(sorted(union - s)[:12])}{' ...' if len(union - s) > 12 else ''}"
                                         for mf, s in sorted(per_manifest.items()) if union - s),
         "namespaces": "|".join(sorted(set(i.split(":")[0] for i in ids))),
         "registry_n_subjects": REG.get(src, {}).get("n_subjects", "")}
    toks = {i: tokens_of(i.split(":", 1)[1]) for i in ids}
    tok_count = Counter(t for v in toks.values() for t in v)
    a["tokens"] = ", ".join(f"{k} {v}" for k, v in sorted(tok_count.items()))
    norm = defaultdict(list)
    for i in ids:
        norm[norm_id(i.split(":", 1)[1])].append(i)
    coll = [v for v in norm.values() if len(v) > 1]
    a["norm_collisions"] = len(coll)
    a["norm_collision_examples"] = "; ".join(" = ".join(v) for v in coll[:5])
    coll_of = {i: " = ".join(v) for v in coll for i in v}
    for i in ids:
        SUBJ_ROWS.append([src, i, int(rec_per[i]), man_per[i], "|".join(toks[i]), coll_of.get(i, "")])
    if src in SIBLING:
        AUDIT[src] = a
        continue
    j = JUDGE.get(src, {"verdict": "identity"})
    expect = j.get("expect", {})
    unexplained = [] if expect == "*" else sorted(set(tok_count) - set(expect))
    a["expected_tokens"] = "all (clip names are action descriptions)" if expect == "*" else "; ".join(f"{k}: {v}" for k, v in expect.items())
    a["unexplained_tokens"] = ", ".join(unexplained)
    a.update(embedded_check(src, g))
    raw_issue = ""
    if src in RAW_CHECKS:
        try:
            res = RAW_CHECKS[src](g)
            rids, table, note = res[:3]
            mids = mfix(src, ids)
            if len(res) > 3 and res[3]:
                raw_issue = res[3]
            only_raw, only_man = sorted(rids - mids), sorted(mids - rids)
            a["raw_table"] = table
            a["raw_ids"] = len(rids)
            a["raw_matched"] = len(rids & mids)
            a["raw_only_in_manifest"] = len(only_man)
            a["raw_only_in_source"] = len(only_raw)
            a["raw_note"] = note
            RAW_ROWS.append([src, table, len(rids), len(mids), len(rids & mids), len(only_raw), len(only_man),
                             " ".join(only_raw[:30]) + (" ..." if len(only_raw) > 30 else ""),
                             " ".join(only_man[:30]) + (" ..." if len(only_man) > 30 else ""), note])
            if only_man:
                raw_issue = "; ".join(x for x in (raw_issue, f"{len(only_man)} manifest ids absent from {table}") if x)
        except Exception as e:  # noqa: BLE001
            a["raw_table"] = f"check failed: {type(e).__name__}: {e}"
            raw_issue = "raw check failed"
            RAW_ROWS.append([src, a["raw_table"], "", len(ids), "", "", "", "", "", ""])
    else:
        a["raw_table"] = "not checked: " + NO_RAW_CHECK.get(src, "no participant table identified")
    issues = []
    if unexplained:
        issues.append("unexplained tokens " + ", ".join(unexplained))
    if a["norm_collisions"] and not j.get("collisions"):
        issues.append(f"{a['norm_collisions']} normalized-id collisions")
    a["collisions_reading"] = j.get("collisions", "") if a["norm_collisions"] else ""
    if a.get("eq_mismatch"):
        issues.append(f"{a['eq_mismatch']} take/task codes disagree with subject_ns")
    if a.get("nested_codes_in_several_subjects"):
        issues.append(f"{a['nested_codes_in_several_subjects']} session or video codes under several subjects")
    if raw_issue:
        issues.append(raw_issue)
    a["issues"] = "; ".join(issues)
    a["verdict"] = "check_failed" if issues else j["verdict"]
    AUDIT[src] = a
    log(f"  {src}: {len(ids)} ids, verdict {a['verdict']}{' (' + a['issues'] + ')' if issues else ''}")

write_csv("audit_subjects.csv", ["source", "subject_ns", "n_records", "manifests", "tokens", "normalized_collision"], SUBJ_ROWS)
write_csv("raw_checks.csv", ["source", "table", "source_ids", "manifest_ids", "matched", "only_in_source", "only_in_manifest",
                             "only_in_source_ids", "only_in_manifest_ids", "note"], RAW_ROWS)
AUD_COLS = ["source", "owner", "subject_ids", "registry_n_subjects", "namespaces", "manifests", "missing_by_manifest", "tokens",
            "expected_tokens", "unexplained_tokens", "norm_collisions", "norm_collision_examples", "collisions_reading", "eq_pairs_checked",
            "eq_pairs_skipped", "eq_mismatch", "eq_mismatch_examples", "nested_codes", "nested_codes_in_several_subjects",
            "raw_table", "raw_ids", "raw_matched", "raw_only_in_manifest", "raw_only_in_source", "raw_note", "issues", "verdict"]
write_csv("audit_sources.csv", AUD_COLS, [[AUDIT[s].get(c, "") for c in AUD_COLS] for s in SOURCES])

# ---------------------------------------------------------------- 3. cross-source identity
log("== 3. cross-source identity")
DOC_RX = re.compile(r"(readme|description|license|\.md$|documentation|\.info\.json$)", re.I)


def doc_corpus(src):
    """(label, text) pairs: registry row, datacard, and documentation files of the raw deposit."""
    out = []
    r = REG.get(src)
    if r:
        out.append(("docs/datasets_index.csv row", " | ".join(r[k] for k in ("title", "n_subjects", "status", "link", "license"))))
    d = SRC_DIR.get(src)
    if not d:
        return out
    for base in [d]:
        if os.path.isfile(base + "/datacard.md"):
            out.append((REG[src]["path"] + "/datacard.md", open(base + "/datacard.md", encoding="utf-8").read()))
        raw = base + "/raw"
        if not os.path.isdir(raw):
            continue
        for root, dnames, files in os.walk(raw):
            depth = root[len(raw):].count(os.sep) + root[len(raw):].count("/")
            dnames[:] = [x for x in dnames if not x.startswith(".") and not x.startswith("._")] if depth < 2 else []
            for n in files:
                p = os.path.join(root, n).replace("\\", "/")
                if n.startswith("._"):
                    continue
                if DOC_RX.search(n) and os.path.getsize(p) < 2_000_000:
                    out.append((p[len(MA) + 1:], open(p, encoding="utf-8", errors="replace").read()))
                elif n.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(p) as zf:
                            for zi in zf.infolist():
                                b = os.path.basename(zi.filename)
                                if b.startswith("._") or "__MACOSX" in zi.filename or zi.filename.count("/") > 3:
                                    continue
                                if DOC_RX.search(b) and zi.file_size < 2_000_000:
                                    out.append((p[len(MA) + 1:] + "::" + zi.filename, zf.read(zi).decode("utf-8", "replace")))
                    except (zipfile.BadZipFile, OSError):
                        pass
    return out


SCAN_TERMS = {  # distinctive site and investigator names; case-sensitive whole words
    "site": ["Tel Aviv", "Sourasky", "tlvmc", "Bristol", "Toronto", "Leuven", "KU Leuven", "Münster", "Iowa", "New Mexico",
             "San Diego", "Khalifa", "Abu Dhabi", "Nebraska", "Omaha", "Chicago", "Shirley Ryan", "Northwestern", "UFABC",
             "Brazil", "São Paulo", "Emory", "Antwerp", "Oxford", "Parma", "Ancona", "Zurich", "Strasbourg", "Illinois",
             "Korea", "Valladolid", "Thessaloniki", "Singapore", "Max Planck", "Philadelphia", "Amsterdam", "Nijmegen",
             "Radboud", "Boston", "FDA", "Rochester", "Vienna", "Austria", "Leipzig", "Newcastle", "Glasgow", "Madrid",
             "Tübingen", "Milan", "Gnocchi", "Nantes"],
    "investigator": ["Hausdorff", "Giladi", "Maidan", "Mirelman", "Plotnik", "Roggen", "Katzir", "Vered", "Cavanagh",
                     "Narayanan", "Espinoza", "Rockhill", "Swann", "Adeli", "Taati", "Iaboni", "Whone", "Mirmehdi",
                     "Dadashzadeh", "Filtjens", "Nieuwboer", "Spildooren", "Ginis", "Gilat", "McKay", "Coelho", "Shida",
                     "Hoang", "Morgan", "Tonkin", "Craddock", "Katmah", "Khalaf", "Atzori", "Criekinge", "Hallemans",
                     "Varghese", "Goodwin", "Stergiou", "Likens", "Terzano", "Fioretti", "Nardo", "Doherty"],
}
CORPUS_SOURCES = sorted(set(SOURCES))
CORPUS = {}
for s in CORPUS_SOURCES:
    CORPUS[s] = doc_corpus(s)
    log(f"  corpus {s}: {len(CORPUS[s])} documents")


EMAIL = re.compile(r"[\w.+-]+@((?:[\w-]+\.)+\w+)")


def first_line(text, rx):
    """First line matching rx, tags stripped, cut around the match; e-mail addresses reduced to their domain."""
    for line in text.splitlines():
        if rx.search(line):
            line = re.sub(r"<[^>]+>", " ", line)
            m = rx.search(line)
            if not m:
                continue
            a, b = max(0, m.start() - 110), min(len(line), m.end() + 110)
            q = ("..." if a else "") + line[a:b] + ("..." if b < len(line) else "")
            return clean(EMAIL.sub(lambda e: f"e-mail at {e.group(1)}", html.unescape(q)))
    return None


SCAN_ROWS, TERM_HITS = [], defaultdict(dict)
for kind, terms in SCAN_TERMS.items():
    for t in terms:
        rx = re.compile(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])")
        for s in CORPUS_SOURCES:
            for label, text in CORPUS[s]:
                q = first_line(text, rx)
                if q:
                    TERM_HITS[t][s] = (label, q)
                    SCAN_ROWS.append([kind, t, s, label, q])
                    break
# Institutional e-mail domains name an affiliation even where no site word is written (tlvmc.gov.il, uiowa.edu):
# every domain of the corpus except webmail and publisher domains counts as a stated institution. Only the domain
# is written, never the address.
EMAIL_SKIP = {"gmail.com", "googlemail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com", "ieee.org", "github.com"}
EMAIL_DOM = re.compile(r"[\w.+-]+@((?:[\w-]+\.)+[A-Za-z]{2,})(?![\w-]|\.[\w-])")
DOMAIN_HITS = defaultdict(dict)
for s in CORPUS_SOURCES:
    for label, text in CORPUS[s]:
        for m in EMAIL_DOM.finditer(text):
            dom = m.group(1).lower()
            if dom in EMAIL_SKIP or re.search(r"\.(png|jpe?g|gif|svg|webp)$", dom) or s in DOMAIN_HITS[dom]:
                continue
            q = first_line(text, re.compile(re.escape(m.group(0))))
            DOMAIN_HITS[dom][s] = (label, q)
for dom in sorted(DOMAIN_HITS):
    for s, (label, q) in sorted(DOMAIN_HITS[dom].items()):
        SCAN_ROWS.append(["email_domain", dom, s, label, q])
        TERM_HITS[dom][s] = (label, q)
write_csv("cross_source_name_scan.csv", ["kind", "term", "source", "document", "line"], SCAN_ROWS)
SITE_STATED = {s for t in SCAN_TERMS["site"] for s in TERM_HITS.get(t, {})} | {s for v in DOMAIN_HITS.values() for s in v}
DOMAIN_ONLY = sorted({s for v in DOMAIN_HITS.values() for s in v if not any(s in TERM_HITS.get(t, {}) for t in SCAN_TERMS["site"])})


def quote(src, rx):
    rx = re.compile(rx)
    for label, text in CORPUS.get(src, []):
        q = first_line(text, rx)
        if q:
            return f"{label}: \"{q}\""
    return f"NOT FOUND in the {src} documents: /{rx.pattern}/"


# Measurements used by the groups.
def nonan_measure():
    a = {i.split(":")[1] for i in M.loc[M["src"] == "nonan_gaitprint", "subject_ns"]}
    b = {i.split(":")[1] for i in M.loc[M["src"] == "nonan_gaitprint_young", "subject_ns"]}
    young_raw = {n[:-4] for n in listdir(SRC_DIR["nonan_gaitprint_young"] + "/raw") if re.match(r"^S\d{3}\.zip$", n)}
    old_below = sorted(x for x in a if int(x[1:]) < max(int(y[1:]) for y in b))
    claim = [x for x in ("S009", "S013", "S019") if x in young_raw]
    return (f"older ids {len(a)}, young ids {len(b)}, shared ids {len(a & b)}; older ids below the highest young id "
            f"({max(b)}): {', '.join(old_below)} (the gaps of the young list); young raw zips named S009, S013 or S019: "
            f"{len(claim)}. The two releases draw on one lab-wide numbering (group code g01 in every young trial name, "
            f"g03 in every older one, checked in audit_sources.csv) and share no id, hence no person.")


def motionx_measure():
    subs = Counter(i.split(":")[1] for i in M.loc[M["src"] == "motion_x", "subject_ns"])
    return "motion_x subsets in the manifest: " + ", ".join(f"{k} {v}" for k, v in sorted(subs.items())) + \
        "; none of them is an AMASS or GRAB subset"


def care_cohorts():
    c = Counter(i.split(":")[1] for i in M.loc[M["src"] == "care_pd", "subject_ns"].drop_duplicates())
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items()))


GROUPS = [
    dict(gid="tel_aviv", title="Tel Aviv Sourasky Medical Center PD and gait sets",
         members=[("daphnet", None), ("pd_walking_eeg_ds007526", None), ("kaggle_fog", None), ("physionet_gaitpdb", None), ("ltmm", None)],
         level="possible (same centre and laboratory family, no shared key)",
         evidence=[("daphnet", r"Tel Aviv Sourasky Medical Center in \d{4}"), ("daphnet", r"Maidan"), ("daphnet", r"Hausdorff"),
                   ("pd_walking_eeg_ds007526", r"Tel Aviv Sourasky Medical Center"), ("pd_walking_eeg_ds007526", r"Inbal Maidan"),
                   ("kaggle_fog", r"tlvmc"), ("kaggle_fog", r"unique PD subjects"), ("physionet_gaitpdb", r"Hausdorff"), ("ltmm", r"PhysioNet"),
                   ("ltmm", r"population: community-dwelling older adults")],
         reading="daphnet (recording year in its README) and ds007526 state the centre on disk and share an investigator "
                 "(Maidan); the Kaggle competition name carries tlvmc, the e-mail domain of the ds007526 authors; gaitpdb "
                 "is registered under Hausdorff, a daphnet author. The ltmm documents on disk state no site (its Tel Aviv "
                 "origin is background knowledge, not verified on disk). The PD sets draw on one clinic, their recording "
                 "years are not all stated on disk, and no set carries a key shared with another.",
         consequence="No merge is possible without a shared key. kaggle_fog is EVAL (never trained) and holds PD patients "
                     "only: any G10 or G11 line evaluated on kaggle_fog is also reported with daphnet, physionet_gaitpdb and "
                     "pd_walking_eeg_ds007526 removed from training, or its person-disjointness claim is limited to "
                     "within-source. ltmm (older adults, fallers and non-fallers) can share people only with the control "
                     "arms of physionet_gaitpdb and pd_walking_eeg_ds007526: no constraint unless a claim rests on healthy "
                     "older adults being person-disjoint across these sources. The lockbox of T0.4 needs no change (it is "
                     "drawn within each source)."),
    dict(gid="bristol", title="Bristol PD gait: CARE-PD PD-GaM and REMAP",
         members=[("care_pd", r"^care_pd:PD-GaM:"), ("remap_open", None)],
         level="possible (same institution per background knowledge; not documented as shared)",
         evidence=[("care_pd", r"Majid Mirmehdi, Alan Whone"), ("remap_open", r"University of Bristol study reference number")],
         reading="The CARE-PD README cites PD-GaM papers by Mirmehdi and Whone; the REMAP readme names the PD SENSORS study "
                 "of the University of Bristol. That the PD-GaM investigators work in Bristol is background knowledge; no "
                 "document on disk says whether the two cohorts share patients.",
         consequence="care_pd is EVAL and remap_open is trainable: a claim evaluated on the PD-GaM cohort is reported also "
                     "with remap_open out of training (or PD-GaM is dropped from that evaluation)."),
    dict(gid="toronto", title="Toronto: CARE-PD T-SDU, T-SDU-PD, T-LTC and li_taati",
         members=[("care_pd", r"^care_pd:T-(SDU|SDU-PD|LTC):"), ("li_taati", None)],
         level="within care_pd possible; against li_taati low",
         evidence=[("care_pd", r"T-SDU(&nbsp;| ).{1,3}Ambient walking recordings"), ("care_pd", r"T-SDU-PD(&nbsp;| ).{1,3}PD patient walking"),
                   ("care_pd", r"T-LTC(&nbsp;| ).{1,3}Ambient walking recordings"), ("care_pd", r"Iaboni A, Taati B"),
                   ("li_taati", r"Taati"), ("li_taati", r"population: .*UDysRS")],
         reading="The three Toronto cohorts of CARE-PD come from one group (Iaboni, Taati) and T-SDU-PD carries the T-SDU "
                 "name with a PD suffix; the README does not say whether T-SDU-PD participants also appear in T-SDU or "
                 "T-LTC, and the ids of each cohort restart at P001. li_taati shares the senior investigator (Taati) but "
                 "is a dyskinesia rating set (UDysRS, UPDRS, CAPSIT per its registry row and datacard) with seated and "
                 "leg-agility tasks, not walking.",
         consequence="Inside care_pd, keep T-SDU, T-SDU-PD and T-LTC in one fold for any person-disjoint split (flagged row by "
                     "row in person_map.csv). Against li_taati: no constraint; note the shared group when a care_pd Toronto "
                     "result is compared with a li_taati-trained head."),
    dict(gid="kul", title="KU Leuven FOG: CARE-PD KUL-DT-T and kaggle_fog",
         members=[("care_pd", r"^care_pd:KUL-DT-T:"), ("kaggle_fog", None)],
         level="low (not verifiable on disk)",
         evidence=[("care_pd", r"Spildooren J, Vercruysse S"), ("kaggle_fog", r"tlvmc"), ("seizeit2", r"European EMUs: University Hospital Leuven")],
         reading="KUL-DT-T is the KU Leuven dual-task turning cohort (Spildooren et al., per the CARE-PD README). Whether the "
                 "Kaggle defog recordings include Leuven patients is background knowledge that no file on disk states. "
                 "seizeit2 also names Leuven (epilepsy monitoring units): a focal-epilepsy cohort against PD cohorts, no "
                 "plausible shared person, no constraint.",
         consequence="Both are EVAL-only: no training leak is possible between them. No constraint beyond not fitting a head "
                     "on one and claiming person-disjoint evaluation on the other."),
    dict(gid="nonan", title="NONAN GaitPrint older and young releases",
         members=[("nonan_gaitprint", None), ("nonan_gaitprint_young", None)],
         level="checked: no shared person", measure=nonan_measure,
         evidence=[("nonan_gaitprint_young", r"subject IDs collide"), ("nonan_gaitprint", r"true sibling")],
         reading="The young datacard says S009, S013 and S019 exist in both releases; on disk the young release has no such "
                 "zip and no such manifest id (measurement above), so that sentence of the datacard is wrong and is left "
                 "to the owner as a MoveAtlas correction.",
         consequence="None: the two sources can sit on different sides of any split."),
    dict(gid="motion_x_amass", title="Motion-X++ against AMASS and NTU RGB+D",
         members=[("motion_x", None), ("amass", None), ("ntu_rgbd", None)],
         level="checked: no shared person in the harmonized records", measure=motionx_measure,
         evidence=[("motion_x", r"Mocap subsets \(AMASS/GRAB\)"), ("motion_x", r"IDEA400 is Motion-X's licensed replacement for NTU-RGBD120")],
         reading="Motion-X++ distributes AMASS- and GRAB-derived mocap subsets, but none of them was harmonized; IDEA400 is "
                 "self-recorded, not NTU footage.",
         consequence="None today. If a Motion-X mocap subset is ever ingested, it duplicates AMASS performers and must follow "
                     "the AMASS person map."),
    dict(gid="eeg_sets", title="PD EEG sets: ds004584, ds002778, ds007526",
         members=[("pd_rest_eeg_ds004584", None), ("pd_rest_eeg_onoff_ds002778", None), ("pd_walking_eeg_ds007526", None)],
         level="checked: different sites",
         evidence=[("pd_rest_eeg_ds004584", r"uiowa\.edu"), ("pd_rest_eeg_ds004584", r"unm\.edu"), ("pd_rest_eeg_ds004584", r"usd\.edu"),
                   ("pd_rest_eeg_onoff_ds002778", r"University of San Diego|UC San Diego"),
                   ("pd_walking_eeg_ds007526", r"Tel Aviv Sourasky Medical Center")],
         reading="Authors with Iowa, New Mexico and South Dakota e-mail domains for ds004584 (no recording site written); "
                 "San Diego for ds002778 (its README names the University of San Diego); Tel Aviv for ds007526.",
         consequence="None among them; ds007526 belongs to the Tel Aviv group above."),
    dict(gid="physionet", title="PhysioNet sets among themselves",
         members=[("physionet_gaitpdb", None), ("gaitndd", None), ("ltmm", None), ("physionet_semg_walking", None),
                  ("physionet_multimodal_gait", None), ("pads", None), ("cap_sleep", None)],
         level="checked: only gaitpdb and ltmm share a plausible recruitment pool (Tel Aviv group)",
         evidence=[("physionet_multimodal_gait", r"Katmah R, AlShehhi A"), ("physionet_semg_walking", r"31 healthy young adults"),
                   ("gaitndd", r"16 controls \+ 15 PD \+ 20 Huntington \+ 13 ALS"), ("pads", r"469 subjects")],
         reading="PhysioNet is a host, not a cohort: the sets come from different groups (young adults for semg and "
                 "multimodal gait, a neurodegenerative gait set, a smartwatch clinic set, a sleep set).",
         consequence="None beyond the Tel Aviv group."),
    dict(gid="stroke", title="stroke_gait_vancriekinge against the other gait sources",
         members=[("stroke_gait_vancriekinge", None), ("clinical_gait_multipatho", None), ("gait120", None)],
         level="checked: no shared deposit or investigator found",
         evidence=[("stroke_gait_vancriekinge", r"50 stroke survivors \+ 138 able-bodied"), ("clinical_gait_multipatho", r"IDENTITY SETTLED")],
         reading="No other harmonized source names the Van Criekinge deposit or its investigators (name scan); the other "
                 "stroke cohort (clinical_gait_multipatho CVA) is a different deposit, and clinical_gait_2025 is the same "
                 "deposit ingested once under clinical_gait_multipatho.",
         consequence="None."),
]
for gr in GROUPS:
    if gr.get("measure"):
        gr["measured"] = gr["measure"]()
    gr["quotes"] = [(s, quote(s, rx)) for s, rx in gr["evidence"]]

# name-scan terms shared by several sources
shared_terms = [(t, sorted(v)) for t, v in TERM_HITS.items() if len(v) > 1]
X_ROWS = [[g["gid"], " | ".join(s + (f" ({f})" if f else "") for s, f in g["members"]), g["level"], clean(g["consequence"])] for g in GROUPS]
write_csv("cross_source_overlap.csv", ["group", "sources", "level", "split_consequence"], X_ROWS)

L = ["# Cross-source person identity", "",
     f"Generated by scripts/build_person_map.py against MoveAtlas {PROV['moveatlas_head']} (see _provenance_build_person_map.json).",
     "Evidence lines are quoted at run time from the registry row, the datacard and the documentation files of each raw",
     "deposit; a missing quote prints NOT FOUND. Background knowledge that no file on disk states is labelled as such.",
     "Rule: two sources that share people must share a split. Where sharing is possible but no key links the records,",
     "a merge is impossible and the consequence is a sensitivity analysis or a grouping rule, stated per group.", "",
     "## Name scan", "",
     f"Site and investigator names searched in {sum(len(v) for v in CORPUS.values())} documents of {len(CORPUS)} sources "
     f"({sum(1 for v in CORPUS.values() for l, _ in v if l == 'docs/datasets_index.csv row')} registry rows, "
     f"{sum(1 for v in CORPUS.values() for l, _ in v if l.endswith('/datacard.md'))} datacards, "
     f"{sum(1 for v in CORPUS.values() for l, _ in v if l != 'docs/datasets_index.csv row' and not l.endswith('/datacard.md'))} "
     f"documentation files of the raw deposits, down to two folder levels and three levels inside zips; "
     f"cross_source_name_scan.csv lists every hit). Names found in more than one source:", "",
     "| term | sources |", "|---|---|"]
for t, srcs in sorted(shared_terms):
    L.append(f"| {t} | {', '.join(srcs)} |")
dom_list = "; ".join(d + " (" + ", ".join(sorted(v)) + ")" for d, v in sorted(DOMAIN_HITS.items()))
L += ["", f"Institutional e-mail domains (only the domain is kept; webmail and publisher domains skipped) count as a"
      f" stated institution: {dom_list or 'none'}."
      f" Sources whose only statement of an institution is an e-mail domain: {', '.join(DOMAIN_ONLY) or 'none'}.", "",
      f"Sources with no site name and no institutional e-mail domain in any scanned document ({len([s for s in CORPUS_SOURCES if s not in SITE_STATED])}): "
      + ", ".join(s for s in CORPUS_SOURCES if s not in SITE_STATED) + ". For these, an overlap can be neither shown nor"
      " excluded from disk; none of them shares an investigator name with another source in the scan unless listed above.", ""]
L += ["## Groups", ""]
for gr in GROUPS:
    L += [f"### {gr['title']} ({gr['gid']})", "",
          "- sources: " + ", ".join(s + (f" (subject_ns matching {f})" if f else "") for s, f in gr["members"]),
          f"- level: {gr['level']}"]
    if gr.get("measured"):
        L.append(f"- measured: {gr['measured']}")
    L.append("- evidence:")
    for s, q in gr["quotes"]:
        L.append(f"  - {s}: {q}")
    L += [f"- reading: {gr['reading']}", f"- consequence for splits: {gr['consequence']}", ""]
L += ["## Within-source groups found by the audit", "",
      f"- care_pd cohorts (subject ids per cohort): {care_cohorts()}. Ids are anonymized per cohort; DNE has two sub-studies"
      " (dne1, dne2) and the three Toronto cohorts may share participants: both flagged in person_map.csv.",
      "- ninapro db1 and db2, physionet_gaitpdb studies Ga, Ju and Si: one laboratory each, separate acquisitions, no"
      " cross-database or cross-study key on disk: flagged in person_map.csv.",
      "- goodwin_smm and multimodal_fog: MoveAtlas already merged a person's two studies or two sessions under one id"
      " (verified in audit_sources.csv from the task names and the patient index).", ""]
write_md("cross_source_overlap.md", L)

# per-row cross-source tags for the combiner
XTAG = defaultdict(list)  # (source, regex or None) -> tags
for gr in GROUPS:
    if gr["level"].startswith("checked"):
        continue
    for s, f in gr["members"]:
        XTAG[s].append((f, f"cross-source group {gr['gid']} ({gr['level']}); see cross_source_overlap.md"))


def xtags(src, sns):
    return [t for f, t in XTAG.get(src, []) if f is None or re.search(f, sns)]


# ---------------------------------------------------------------- 4. combiner
log("== 4. combiner")


def evidence_of(src):
    a = AUDIT[src]
    parts = []
    if a.get("raw_ids") not in (None, ""):
        parts.append(f"{a['raw_matched']} of {a['subject_ids']} ids found in {a['raw_table']}")
    if a.get("eq_pairs_checked"):
        parts.append(f"{a['eq_pairs_checked']} (subject, take/task) pairs carry a matching subject code")
    if a.get("nested_codes"):
        parts.append(f"{a['nested_codes']} session/video codes each under one subject")
    if len(a["manifests"].split(";")) > 1:
        parts.append("same subject_ns across " + a["manifests"])
    if a.get("raw_ids") in (None, "") and a.get("raw_table"):
        parts.append("raw table " + a["raw_table"])
    return "; ".join(parts)


AMB_CLASS = defaultdict(Counter)  # source -> within-source ambiguity class -> rows


def own_rows(src):
    a = AUDIT[src]
    g = M[M["src"] == src]
    ids = sorted(g["subject_ns"].unique())
    rows = []
    if src == "motion_x":
        for i in ids:
            sub, loc = i.split(":")[1], i.split(":", 2)[2]
            stem = re.sub(r"(?:_clip\d+)+$", "", loc)
            rows.append([src, i, f"motionx:{sub}:{stem}", "video_stem",
                         "clip-indexed source: clips cut from one source video share the stem (take = local id in every record)",
                         "performer identity is not recorded; several stems can show one performer; for a person-disjoint claim split motion_x by subset"])
            AMB_CLASS[src]["performer not recorded; stems may share a performer"] += 1
        return rows
    ev = evidence_of(src)
    dne1 = {i.split("_s")[-1]: i for i in ids if ":DNE:dne1_" in i}
    dne2 = {i.split("_s")[-1]: i for i in ids if ":DNE:dne2_" in i}
    for i in ids:
        amb = []
        if src == "care_pd" and ":DNE:" in i:
            num = i.split("_s")[-1]
            other = (dne2 if ":dne1_" in i else dne1).get(num)
            AMB_CLASS[src]["DNE sub-studies dne1 and dne2 may share participants" + (", same s-number in both" if other else "")] += 1
            amb.append("DNE sub-studies dne1 and dne2 may share participants" +
                       (f"; candidate partner {other} (same s-number)" if other else "; no same-number partner") +
                       "; not settleable on disk (settle with the CARE-PD authors or the DNE papers)")
        if src == "care_pd" and re.search(r":T-(SDU|SDU-PD|LTC):", i):
            AMB_CLASS[src]["Toronto cohorts T-SDU, T-SDU-PD and T-LTC may share participants"] += 1
            amb.append("T-SDU, T-SDU-PD and T-LTC (one Toronto group) may share participants; keep the three cohorts in one fold")
        if src == "ninapro":
            AMB_CLASS[src]["Ninapro DB1 and DB2 may share participants"] += 1
            amb.append("Ninapro DB1 and DB2 are separate acquisitions; a person in both would carry two ids; not stated on disk (settle with the Ninapro participant tables, Atzori et al.)")
        if src == "physionet_gaitpdb":
            AMB_CLASS[src]["studies Ga, Ju and Si of one laboratory may share participants"] += 1
            amb.append(f"study {i.split(':')[1][:2]}: the three studies (Ga, Ju, Si) come from one laboratory; a person in two studies would carry two ids; not stated on disk")
        amb += xtags(src, i)
        rule = "identity (flagged)" if a["verdict"] == "identity_flagged" else "identity" if a["verdict"] == "identity" else "identity (audit check failed: " + a["issues"] + ")"
        rows.append([src, i, i, rule, ev, "; ".join(amb)])
    return rows


def load_map_files():
    maps = {}
    for p in sorted(glob.glob(os.path.join(OUT, "map_*.csv"))):
        b = os.path.basename(p)
        if b.startswith("._") or b == "combiner_map_status.csv":
            continue
        try:
            d = read_csv(p)
        except Exception as e:  # noqa: BLE001
            maps[b] = {"error": f"{type(e).__name__}: {e}"}
            continue
        name = b[4:-4]
        maps[b] = {"df": d, "name": ALIAS.get(name, name)}
    return maps


MAPS = load_map_files()
MODYS = os.path.join(TREE, "reports", "modys_subjects", "modys_video_person_map.csv")
PM_ROWS, COUNTS, MAP_STATUS = [], [], []
by_source_map = {}
for b, info in MAPS.items():
    if "error" in info:
        MAP_STATUS.append([b, "", "unreadable: " + info["error"], "", "", "", ""])
        continue
    d = info["df"]
    need = {"subject_ns", "person_id"}
    if not need <= set(d.columns):
        MAP_STATUS.append([b, info["name"], "missing columns " + ", ".join(sorted(need - set(d.columns))), "", "", "", ""])
        continue
    if "source" in d.columns:
        for s, dd in d.groupby(d["source"].map(lambda x: ALIAS.get(x, x))):
            by_source_map.setdefault(s, []).append((b, dd))
    else:
        by_source_map.setdefault(info["name"], []).append((b, d))
for s in sorted(set(by_source_map) - set(SOURCES)):
    MAP_STATUS.append([", ".join(x[0] for x in by_source_map[s]), s, "source absent from the manifests: not combined",
                       0, 0, sum(len(x[1]) for x in by_source_map[s]), ""])

for src in SOURCES:
    g = M[M["src"] == src]
    ids = sorted(g["subject_ns"].unique())
    idset = set(ids)
    if src in by_source_map:
        parts = by_source_map[src]
        d = pd.concat([x[1] for x in parts], ignore_index=True)
        files = ", ".join(x[0] for x in parts)
        for c in ("rule", "evidence", "ambiguity"):
            if c not in d.columns:
                d[c] = ""
        dup = d.groupby("subject_ns")["person_id"].nunique()
        conflicts = int((dup > 1).sum())
        d = d.drop_duplicates("subject_ns")
        extra = sorted(set(d["subject_ns"]) - idset)
        uncovered = sorted(idset - set(d["subject_ns"]))
        dm = d.set_index("subject_ns")
        for i in ids:
            if i in dm.index and dm.at[i, "person_id"]:
                r = dm.loc[i]
                if r["ambiguity"]:
                    AMB_CLASS[src][f"ambiguity recorded in {files}"] += 1
                amb = "; ".join(x for x in [r["ambiguity"]] + xtags(src, i) if x)
                PM_ROWS.append([src, i, r["person_id"], r["rule"] or f"from {files}", r["evidence"] or f"from {files}", amb])
            else:
                PM_ROWS.append([src, i, "", f"uncovered by {files}", "", "subject_ns absent from the map file or without person_id"])
        MAP_STATUS.append([files, src, "combined", len(ids), len(ids) - len(uncovered), len(extra), conflicts])
        persons = d[d["subject_ns"].isin(idset) & (d["person_id"] != "")]["person_id"].nunique()
        rules = "|".join(sorted(set(r for r in d["rule"] if r))) or f"map file {files}"
        COUNTS.append([src, len(ids), persons if not uncovered else "", rules + ("" if not uncovered else f" (incomplete: {len(uncovered)} ids uncovered)")])
    elif src == "modys_video" and os.path.isfile(MODYS):
        d = read_csv(MODYS)
        modys_conflicts = int((d.groupby("subject_ns")["person_ns"].nunique() > 1).sum())
        d = d.drop_duplicates("subject_ns")
        dm = d.set_index("subject_ns")["person_ns"]
        for i in ids:
            pid = dm.get(i, "")
            PM_ROWS.append([src, i, pid, "modys child map" if pid else "uncovered by modys map",
                            "reports/modys_subjects/modys_video_person_map.csv: video number joined to the child ID of data_clinical_scoring.xlsx (probe_modys_subjects.py)",
                            "; ".join(xtags(src, i))])
        cov = sum(1 for i in ids if dm.get(i, ""))
        MAP_STATUS.append(["reports/modys_subjects/modys_video_person_map.csv", src, "combined", len(ids), cov, len(set(d["subject_ns"]) - idset), modys_conflicts])
        COUNTS.append([src, len(ids), d[d["subject_ns"].isin(idset)]["person_ns"].nunique() if cov == len(ids) else "", "modys child map"])
    elif src in SIBLING:
        for i in ids:
            PM_ROWS.append([src, i, "", f"pending: map_{src}.csv of task {SIBLING[src]} absent at this run", "",
                            "; ".join(["person identity not yet audited"] + xtags(src, i))])
        COUNTS.append([src, len(ids), "", f"pending ({SIBLING[src]} map absent)"])
    else:
        rows = own_rows(src)
        PM_ROWS.extend(rows)
        COUNTS.append([src, len(ids), len({r[2] for r in rows}), rows[0][3] if rows else ""])

# global checks on the combined map
pid_src = defaultdict(set)
for r in PM_ROWS:
    if r[2]:
        pid_src[r[2]].add(r[0])
cross_pid = {p: s for p, s in pid_src.items() if len(s) > 1}
write_csv("person_map.csv", ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"], PM_ROWS)
write_csv("person_counts.csv", ["source", "subject_ids", "persons", "rule"], COUNTS)
write_csv("combiner_map_status.csv", ["map_file", "source", "status", "manifest_ids", "covered", "extra_ids", "conflicting_ids"], MAP_STATUS)


# ---------------------------------------------------------------- summary
tot_ids = sum(c[1] for c in COUNTS)
done = [c for c in COUNTS if c[2] != ""]
pending = [c for c in COUNTS if c[2] == ""]
flag_rows = sum(1 for r in PM_ROWS if r[5])


def row(cells):
    return "| " + " | ".join(cell(x) for x in cells) + " |"


raw_ok = [r for r in RAW_ROWS if r[4] != ""]
raw_cfg = [r for r in raw_ok if "MoveAtlas, read-only" in r[1]]
raw_failed = [r for r in RAW_ROWS if r[4] == ""]
map_rows = [r for r in MAP_STATUS if r[2] == "combined"]
dne_pairs = sum(1 for r in PM_ROWS if r[0] == "care_pd" and "candidate partner" in r[5]) // 2
S = ["# Person map of the harmonized in-scope sources (T0.2d)", "",
     f"Generated by scripts/build_person_map.py at {PROV['run_at']} against MoveAtlas {PROV['moveatlas_head']}.",
     "Re-run the script after each sibling task (T0.2a to T0.2c) writes its map file; the combiner picks up every",
     "reports/person_ids/map_<source>.csv present at run time and leaves a source pending (empty person_id) otherwise.", "",
     "## Map files present at this run", ""]
if MAP_STATUS:
    S += [row(["map file", "source", "status", "manifest ids", "covered", "extra ids", "conflicting ids"]), "|---|---|---|---|---|---|---|"]
    S += [row(r) for r in MAP_STATUS]
else:
    S.append("None.")
S += ["", "## Scope", "",
      f"- manifests read: {len(MANIFEST_FILES)} files (pooled and per-source extras); CODY rows dropped on read, never used (sources {', '.join(sorted(dropped_sources))})",
      f"- non-CODY sources in the manifests: {len(SOURCES)}; audited here: {len(MINE)}; owned by sibling tasks or the modys map: {len(SOURCES) - len(MINE)}",
      f"- manifest sources outside the harmonized in-scope list of reports/data_scope: {', '.join(not_in_scope) or 'none'}",
      f"- harmonized in-scope ids (non-CODY) absent from the manifests: {', '.join(harm_absent) or 'none'}",
      f"- totals: {tot_ids} subject ids; {len(done)} sources with persons resolved; excluding motion_x (clip-indexed, "
      f"counted in video stems, not persons): {sum(c[2] for c in done if c[0] != 'motion_x')} persons over "
      f"{sum(c[1] for c in done if c[0] != 'motion_x')} subject ids; {len(pending)} sources pending "
      f"({sum(c[1] for c in pending)} subject ids); {flag_rows} person_map rows carry an ambiguity or cross-source note "
      f"({sum(1 for r in PM_ROWS if r[5] and r[0] != 'motion_x')} outside motion_x)",
      f"- person_id values shared by two sources: {len(cross_pid)}",
      f"- map files combined: {len(map_rows)} ({sum(1 for r in map_rows if r[0].startswith('map_'))} map_<source>.csv files and"
      f" {sum(1 for r in map_rows if not r[0].startswith('map_'))} child map of modys_video); manifest ids they leave uncovered:"
      f" {sum(int(r[3]) - int(r[4]) for r in map_rows)}; subject ids with conflicting person ids across map rows: {sum(int(r[6]) for r in map_rows)}",
      f"- identity checks against a participant table or file list (raw_checks.csv): {len(raw_ok)} audited sources"
      f" ({len(raw_ok) - len(raw_cfg)} against a file of the raw deposit, {len(raw_cfg)} against a MoveAtlas config index);"
      f" checks that failed to run: {len(raw_failed)}; manifest ids absent from their table: {sum(int(r[6]) for r in raw_ok)};"
      f" table ids absent from the manifest: {sum(int(r[5]) for r in raw_ok)}"
      f" ({', '.join(r[7] for r in raw_ok if int(r[5])) or 'none'}); audited sources without such a check: "
      f"{', '.join(s + ' (' + NO_RAW_CHECK.get(s, 'none identified') + ')' for s in MINE if s not in RAW_CHECKS) or 'none'}",
      f"- care_pd DNE: {dne_pairs} dne1/dne2 pairs share an s-number (candidate partners, flagged, not merged)", "",
      "## Audit of the sources owned by this task", "",
      "Checks per source (audit_sources.csv has every field): suffix tokens in subject_ns and whether each is explained,",
      "normalized-id collisions (case, zero padding, punctuation, parenthetical suffix), the same subject_ns in every",
      "modality manifest, subject codes embedded in take or task names, and the raw participant table or file list.", "",
      row(["source", "subject ids", "manifests", "tokens", "collisions", "take/task code check", "raw table: matched / manifest ids", "verdict", "persons"]),
      "|---|---|---|---|---|---|---|---|---|"]
cnt = {c[0]: c for c in COUNTS}
for s in MINE:
    a = AUDIT[s]
    emb = ""
    if a.get("eq_pairs_checked") not in (None, ""):
        emb = f"{a['eq_pairs_checked']} pairs, {a['eq_mismatch']} mismatches"
    if a.get("nested_codes") not in (None, ""):
        emb += ("; " if emb else "") + f"{a['nested_codes']} nested codes, {a['nested_codes_in_several_subjects']} shared"
    raw = f"{a['raw_matched']} / {a['subject_ids']} ({a['raw_table']})" if a.get("raw_ids") not in (None, "") else a.get("raw_table", "")
    S.append(row([s, a["subject_ids"], a["manifests"], a["tokens"] or "none", a["norm_collisions"], emb or "n/a", raw,
                  a["verdict"] + (f" ({a['issues']})" if a.get("issues") else ""), cnt[s][2]]))
S += ["", "Explained tokens and collisions:", ""]
for s in MINE:
    if AUDIT[s].get("expected_tokens"):
        S.append(f"- {s}: {AUDIT[s]['expected_tokens']}")
    if AUDIT[s].get("collisions_reading"):
        S.append(f"- {s}, {AUDIT[s]['norm_collisions']} normalized collisions: {AUDIT[s]['collisions_reading']} (examples: {AUDIT[s]['norm_collision_examples']})")
S += ["", "Raw-table notes (raw_checks.csv):", ""]
for s in MINE:
    a = AUDIT[s]
    if a.get("raw_note") or a.get("raw_only_in_source"):
        S.append(f"- {s}: {a.get('raw_note', '')}" + (f"; ids in the source table and not in the manifest: {a['raw_only_in_source']}" if a.get("raw_only_in_source") else ""))
S += ["", "Ids missing from one modality manifest of a multimodal source (no harmonized record in that modality: a recording"
      " absent from the deposit or refused at ingestion, under the same naming, not a second id; the datacards give the reason):", ""]
for s in SOURCES:
    if AUDIT[s].get("missing_by_manifest"):
        S.append(f"- {s}: {AUDIT[s]['missing_by_manifest']}")
S += ["", "## Person counts", "", row(["source", "subject ids", "persons", "rule"]), "|---|---|---|---|"]
S += [row([c[0], c[1], c[2] if c[2] != "" else "pending", c[3]]) for c in COUNTS]
S += ["", "## Within-source ambiguities kept separate (never merged without evidence)", ""]
for s in sorted(AMB_CLASS):
    for k, n in sorted(AMB_CLASS[s].items()):
        S.append(f"- {s}: {k} ({n} ids)")
S += ["", "## Cross-source", "", "See cross_source_overlap.md. Groups whose sharing is possible and that constrain evaluation:", ""]
for gr in GROUPS:
    if not gr["level"].startswith("checked"):
        S.append(f"- {gr['gid']}: {', '.join(s + (' (' + f + ')' if f else '') for s, f in gr['members'])}; {gr['level']}")
S += ["", "## For the owner", "",
      "- MoveAtlas correction candidate: the nonan_gaitprint_young datacard says ids S009, S013 and S019 exist in both NONAN"
      " releases; on disk they exist only in the older release (cross_source_overlap.md, group nonan).",
      f"- pd_biostamprc21: registry n_subjects \"{REG['pd_biostamprc21']['n_subjects']}\", datacard {quote('pd_biostamprc21', r'population: ')};"
      f" the clinic table on disk: {AUDIT['pd_biostamprc21'].get('raw_note', '') or '(not read)'} (plan task T0.7 owns the group table).",
      "- The sibling maps decide li_taati (OD7), monipar, amass, kaggle_fog and the others; T0.3 and T0.4 wait until"
      " no source is pending in person_counts.csv.", ""]
write_md("person_map_summary.md", S)
log("done")
