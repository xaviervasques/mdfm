"""T0.3 and T0.4: tiers of every canonical record, and the lockbox, frozen with hashes (version 1.2).

Inputs (read-only): the MoveAtlas pooled canonical manifests and their per-source extras
(datasets/_canonical_*manifest*.csv), the person map of T0.2 (reports/person_ids/person_map.csv,
one person_id per subject_ns), the scope of reports/data_scope/data_scope.csv, for group labels
GaitRec's GRF_metadata.csv, ds004584's participants.tsv and weargait_pd's CONTROLS table, and for
the change list the v1.1 lockbox (reports/lockbox/lockbox_persons_v1_1.csv, SHA-256 checked).

Records, one row per record of a MoveAtlas container (v1.2):
- key of a manifest row: take (pose), record_path (EEG and EMG: the repository path of the row's
  one-record container), task (IMU and force), sequence (silhouette).
- location: a source whose rows split one (subject_ns, key) of a manifest file by location
  (goodwin_smm: the trunk, wrist_L and wrist_R sensors of one session) takes the key
  <key>_<location> on every row of that family, the form of pads, whose manifest task already ends
  with its location.
- occurrence: in a list-container family (pose, IMU, force, silhouette), the n-th row (n >= 2) of a
  manifest file that repeats (source, subject_ns, key) takes the key <key>#<n>: the n-th record of that
  id in container order (kaggle_fog 4dc2f8 tdcsfog_003f117e14 and vidimu S25 S25_A02_T01 are listed,
  and held, twice).
- identity: (source, family, subject_ns, key), with the key of a record_path row taken without its
  .npz or .pkl extension. Rows of several files with one identity are one record (a per-source file
  repeats the pooled rows; cervical_motion_multiplanar lists each EMG record as .npz in the pooled file
  and as .pkl in its own); the .npz path is kept. Two such rows with different n_frames stop the build.
scripts/build_record_index.py reads the same keys (keyed_manifests) for its manifest-order rule.

Tiers (design docs/MFM_DESIGN_2026-09-24.md section 5; addendum section 2.4; licence decisions
docs/decisions/licence_decisions_2026-09-24.csv):
- EVAL, never trained in any form: cody_1, cody_2, cody_3, care_pd, tremordb,
  hubu_fis_fingertap, fog_turning_filho, pd_biostamprc21, kaggle_fog, pd_rest_eeg_onoff_ds002778,
  shida_pd_gait. shida_pd_gait (open motion capture of 2026-11-12, MoveAtlas c561fc2) joined on 2026-11-12:
  its 26 Parkinson's patients are very likely the BMCLab persons of care_pd (EVAL), whose investigators include
  Shida (reports/person_ids/cross_source_overlap.md, cross_source_name_scan.csv); no person map links the two
  sources without reading care_pd, so it is never trained on.
- INTERNAL, trained locally only, never released: ntu_rgbd, koa_pd_nm, goodwin_smm, asdpose,
  osf_fmd_eeg, osf_fmd_kinematics, mini_rgbd.
- EXCLUDED_V1: infant_pose_chambers (infants are excluded in version 1, design section 2).
- RELEASE: every other in-scope source.
Lanes: EEG records feed the neural lane, every other modality the body lane; seizeit2 enters
the body lane in no form (design section 16.2, outside the body model).

Groups (the lockbox strata, column group of the records file): weargait_pd controls are the Subject
IDs of the source CONTROLS table, looked up with the " (control)" suffix that three ids carry removed
(the person_id keeps it), every other weargait_pd person is PD; physionet_gaitpdb Co or Pt (characters
2 and 3 of the id); gaitndd the letters of the id; stroke_gait_vancriekinge the namespace part after the
source; koa_pd_nm and clinical_gait_multipatho the id prefix before "_"; gaitrec CLASS_LABEL; ds004584
GROUP; "all" elsewhere.

Lockbox (design section 5; addendum section 2.4): 20 percent of the persons, rounded half up,
of every labelled pathology source with at least 30 patients, stratified by the group the
data carries (drawn per group, then per source), plus the gate hold-outs the design names:
gait120 (G4, 24 of 120), amass (G1, 50 subjects held out by subset as well as by id: whole
subsets, in a fixed pseudo-random order, until at least 50 persons), and the neural sources of
addendum 2.4. Pools: stroke_gait_vancriekinge draws from its stroke group only, gaitrec from
its orthopaedic patients only (class not HC). li_taati has 9 persons (T0.2a) and gets no
lockbox. The draw is seeded by the SHA-256 of "mdfm-lockbox-v1:<source>", so it is
reproducible and independent of the order of sources. v1.2 keeps the seeds; the build stops when a
source other than weargait_pd changes membership or group against v1.1.

CODY exclusion is mechanical. A CODY row is a row whose source or manifest_source starts with "cody".
The denylist (cody_denylist.json) holds cody_1, cody_2, cody_3, the namespaces of the CODY rows (the
part of subject_ns before the first colon, read from the manifest rows only: no CODY container or label
is opened) and the design's names (adult, ext_ds, PEDIA), compared lowercased. A row whose source,
manifest_source, subject_ns or person_id starts with "cody", or whose subject_ns or person_id sits in a
denylisted namespace, is EVAL; the build stops when a non-CODY source uses a denylisted namespace, so the
name test of mdfm/data/canonical.py refuses every row of those namespaces. A canary id is defined for the
training loader tests.

Writes: cache/manifest/manifest_records.csv.gz (every record with tier, lane, person_id, group and
lockbox flag; gitignored), reports/manifest_tiers/ (per-source summary, hashes.json,
cody_denylist.json) and reports/lockbox/ (lockbox_persons.csv, summary.md, lockbox.sha256,
lockbox_persons_v1_1.csv copied once from the v1.1 file, v1_2_changes.csv).
Run: python scripts/build_tiers_and_lockbox.py --moveatlas <moveatlas>
"""
import collections
import csv
import glob
import gzip
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys

from _common import parse_args, write_provenance

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "1.2"
VERSION_DATE = "2026-09-28"
V1_1_SHA256 = "f94df2ffffac823cb8334d9f5c6ad0c75f6b4bb3d81d4a62c96cf5c63114da23"   # reports/lockbox v1.1
EVAL = {"cody_1", "cody_2", "cody_3", "care_pd", "tremordb", "hubu_fis_fingertap", "fog_turning_filho",
        "pd_biostamprc21", "kaggle_fog", "pd_rest_eeg_onoff_ds002778",
        "shida_pd_gait"}   # 2026-11-12: probable care_pd BMCLab persons (docstring)
INTERNAL = {"ntu_rgbd", "koa_pd_nm", "goodwin_smm", "asdpose", "osf_fmd_eeg", "osf_fmd_kinematics", "mini_rgbd"}
EXCLUDED_V1 = {"infant_pose_chambers"}
NO_BODY_LANE = {"seizeit2"}
CANARY = "cody_canary:mdfm-v1"
CODY_SOURCES = ("cody_1", "cody_2", "cody_3")
DESIGN_NAMESPACES = ("adult", "ext_ds", "PEDIA")     # design section 5
# Manifest source names that differ from the registry ids (same table as scripts/build_person_map.py).
ALIAS = {"cap_sleep_chin": "cap_sleep", "cervical_motion": "cervical_motion_multiplanar",
         "semg_walking": "physionet_semg_walking", "seizeit2_emg": "seizeit2", "ntu": "ntu_rgbd",
         "modys": "modys_video", "cody": "cody_1", "cody_external": "cody_3"}
MODALITY = {"_canonical_manifest": "pose", "_canonical_imu_manifest": "imu", "_canonical_force_manifest": "force",
            "_canonical_emg_manifest": "emg", "_canonical_eeg_manifest": "eeg",
            "_canonical_silhouette_manifest": "silhouette"}
CONTROL_SUFFIX = re.compile(r"\s*\(control\)\s*$")
RECORD_COLS = ["source", "manifest_source", "modality", "subject_ns", "person_id", "key", "fps", "n_frames", "tier",
               "lane", "scope", "lockbox", "group"]
LOCK_COLS = ["source", "lane", "person_id", "group", "rule"]


def rd(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def require(ok, message):
    """Stop the build with message when ok is false (a build check that python -O does not remove)."""
    if not ok:
        sys.exit(message)


# ---------------------------------------------------------------- names
def is_cody_name(*names):
    """True when a name (source, subject_ns, person_id) starts with 'cody', any case."""
    return any(str(n).strip().lower().startswith("cody") for n in names)


def namespace(name):
    """Part of an id before the first colon, lowercased ('adult' for 'adult:C1'); '' without a colon."""
    name = str(name)
    return name.split(":", 1)[0].strip().lower() if ":" in name else ""


def is_cody_source(source, manifest_source=""):
    return is_cody_name(source, manifest_source)


def measured_cody_namespaces(records):
    """Namespaces of the subject_ns of the CODY rows (source or manifest_source starting with 'cody'), from
    the manifest rows only; a CODY subject_ns without a namespace raises ValueError."""
    rows = [r for r in records if is_cody_source(r["source"], r["manifest_source"])]
    bare = sorted({r["subject_ns"] for r in rows if not namespace(r["subject_ns"])})
    if bare:
        raise ValueError(f"CODY subject_ns without a namespace: {bare[:5]}")
    return sorted({namespace(r["subject_ns"]) for r in rows})


def trespassers(records, deny):
    """(source, subject_ns, person_id) of the non-CODY rows whose subject_ns or person_id starts with 'cody'
    or sits in a namespace of deny: the build stops on any."""
    return sorted({(r["source"], r["subject_ns"], r.get("person_id", "")) for r in records
                   if not is_cody_source(r["source"], r["manifest_source"])
                   and (is_cody_name(r["subject_ns"], r.get("person_id", ""))
                        or namespace(r["subject_ns"]) in deny or namespace(r.get("person_id", "")) in deny)})


def tier_of(source, subject_ns, person_id="", cody_namespaces=()):
    if is_cody_name(source, subject_ns, person_id) or namespace(subject_ns) in cody_namespaces \
            or namespace(person_id) in cody_namespaces:
        return "EVAL"
    if source in EVAL:
        return "EVAL"
    if source in INTERNAL:
        return "INTERNAL"
    if source in EXCLUDED_V1:
        return "EXCLUDED_V1"
    return "RELEASE"


# ---------------------------------------------------------------- manifest keys
def family_of(file_name):
    """Canonical family of a pooled manifest file (_canonical_<family>_manifest[_<source>].csv); None if none."""
    stem = file_name[:-4]
    return next((m for k, m in MODALITY.items() if stem == k or stem.startswith(k + "_")), None)


def read_manifests(ma):
    """[(file name, family, rows)] of the pooled canonical manifests, sorted by name; AppleDouble sidecars,
    .bak copies and files of no family are skipped."""
    out = []
    for path in sorted(glob.glob(f"{ma}/datasets/_canonical_*manifest*.csv")):
        base = os.path.basename(path)
        if base.startswith("._") or not base.endswith(".csv"):
            continue
        fam = family_of(base)
        if fam is not None:
            out.append((base, fam, rd(path)))
    return out


def base_key(r, family):
    """(key, is_path): the id of a manifest row within its subject, and whether it is a record_path."""
    if family == "silhouette":
        return r.get("sequence", "") or "", False
    if family in ("imu", "force") and r.get("task"):
        return r["task"], False
    key = r.get("take") or r.get("record_path") or r.get("task") or r.get("sequence") or ""
    return key, bool(key) and key == r.get("record_path")


def location_of(r):
    """The locations field of a row as one token ('wrist_L'; several locations joined by '_')."""
    return "_".join(x for x in re.split(r"[,;| ]+", r.get("locations", "") or "") if x)


def location_sources(files):
    """{(manifest source, family)} whose rows split one (subject_ns, key) of a file by location."""
    out = set()
    for _, family, rows in files:
        locs = collections.defaultdict(set)
        for r in rows:
            key, is_path = base_key(r, family)
            if not is_path:
                locs[(r["source"], r["subject_ns"], key)].add(location_of(r))
        out |= {(s, family) for (s, _, _), v in locs.items() if len(v) > 1}
    return out


def file_keys(rows, family, loc_sources):
    """Keys of the rows of one manifest file, in row order: base key, then the location suffix for a
    source of loc_sources, then #<n> on the n-th repeat (n >= 2) of (source, subject_ns, key) in a
    list-container family. A record_path key is never suffixed."""
    keys, seen = [], collections.Counter()
    for r in rows:
        key, is_path = base_key(r, family)
        if not is_path:
            if (r["source"], family) in loc_sources:
                loc = location_of(r)
                if not loc:
                    raise ValueError(f"{r['source']} {r['subject_ns']} {key}: no location on a row of a source "
                                     "keyed by location")
                key = f"{key}_{loc}"
            seen[(r["source"], r["subject_ns"], key)] += 1
            n = seen[(r["source"], r["subject_ns"], key)]
            if n > 1:
                key = f"{key}#{n}"
        keys.append(key)
    return keys


def keyed_manifests(ma):
    """[(file name, family, rows, keys)]: read_manifests with the key of every row (file_keys), the
    location rule decided over all files."""
    files = read_manifests(ma)
    locs = location_sources(files)
    return [(base, fam, rows, file_keys(rows, fam, locs)) for base, fam, rows in files]


def record_uid(source, family, subject_ns, key, is_path):
    """Identity of a record: a record_path is taken without its .npz or .pkl extension."""
    return (source, family, subject_ns, re.sub(r"\.(npz|pkl)$", "", key) if is_path else key)


# ---------------------------------------------------------------- groups
def strip_control(tail):
    """A weargait_pd Subject ID without the ' (control)' suffix ('NLS901 (control)' -> 'NLS901')."""
    return CONTROL_SUFFIX.sub("", tail).strip()


def group_of(source, person, tables, strip_suffix=True):
    """Lockbox stratum of a person. tables: gaitrec (id -> class), ds004584 (id -> group), weargait_controls
    (Subject IDs). strip_suffix=False is the v1.1 weargait_pd lookup (kept to list what v1.2 changed)."""
    tail = person.split(":")[-1]
    if source == "weargait_pd":
        return "control" if (strip_control(tail) if strip_suffix else tail) in tables["weargait_controls"] else "PD"
    if source == "physionet_gaitpdb":
        return tail[2:4]
    if source == "gaitndd":
        return "".join(ch for ch in tail if ch.isalpha())
    if source == "stroke_gait_vancriekinge":
        return person.split(":")[1]
    if source == "koa_pd_nm":
        return tail.split("_")[0]
    if source == "clinical_gait_multipatho":
        return tail.split("_")[0]
    if source == "gaitrec":
        return tables["gaitrec"].get(person, "")
    if source == "pd_rest_eeg_ds004584":
        return tables["ds004584"].get(person, "")
    return "all"


def gaitrec_classes(ma):
    paths = glob.glob(f"{ma}/datasets/*/gaitrec/raw/GRF_metadata.csv")
    out = {}
    if paths:
        for r in rd(paths[0]):
            out[f"gaitrec:{r['SUBJECT_ID']}"] = r.get("CLASS_LABEL", "")
    return out


def ds004584_groups(ma):
    paths = [p for p in glob.glob(f"{ma}/datasets/*/pd_rest_eeg_ds004584/**/participants.tsv", recursive=True)
             if not os.path.basename(p).startswith("._")]
    out = {}
    if paths:
        with open(paths[0], encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                pid = r.get("participant_id", "")
                out[f"pd_rest_eeg_ds004584:{pid}"] = r.get("GROUP", r.get("group", ""))
    return out


def weargait_controls(ma):
    """Subject IDs of the weargait_pd controls, from the source's own CONTROLS table (the ids carry HC, WHC
    and NLS prefixes; NLS is used by both groups, so a prefix rule is wrong)."""
    paths = [p for p in glob.glob(f"{ma}/datasets/imu/weargait_pd/raw/*/CONTROLS - Demographic+Clinical - datasetV1.csv")
             if not os.path.basename(p).startswith("._")]
    if not paths:
        return set()
    with open(paths[0], encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    head = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Subject ID")
    return {r[0].strip() for r in rows[head + 1:] if r and r[0].strip()}


# ---------------------------------------------------------------- lockbox
LOCKBOX = {
    # source: (lane, pool filter on group or None, design reference)
    "pads": ("body", None, "design s5 stage 2 (375 of 469)"),
    "weargait_pd": ("body", None, "design s5 stage 2 (148 of 185), G3 lockbox 37; groups from the source CONTROLS "
                                  "table, id suffix ' (control)' removed before the lookup (v1.2)"),
    "physionet_gaitpdb": ("body", None, "design s5 stage 2 (132 of 165)"),
    "gaitndd": ("body", None, "design s5 stage 2 (51 of 64)"),
    "clinical_gait_multipatho": ("body", None, "design s5 stage 2 (208 of 260)"),
    "imu_inthewild_iprognosis": ("body", None, "design s5 stage 2 (36 of 45)"),
    "modys_video": ("body", None, "design s5: 7 children with all their visits"),
    "stroke_gait_vancriekinge": ("body", {"stroke"}, "design s5 stage 2 (40 of 50 stroke)"),
    "gaitrec": ("body", "not_HC", "design s5 stage 2 (1,667 of 2,084 orthopaedic)"),
    "koa_pd_nm": ("body", None, "design s5 INTERNAL (77 of 96)"),
    "gait120": ("body", None, "design G4 (24 subjects)"),
    "pd_rest_eeg_ds004584": ("neural", None, "addendum 2.4 (30)"),
    "pd_walking_eeg_ds007526": ("neural", None, "addendum 2.4 (29)"),
    "seizeit2": ("neural", None, "addendum 2.4 (25)"),
    "cap_sleep": ("neural", None, "addendum 2.4 (7)"),
    "physionet_multimodal_gait": ("neural", None, "addendum 2.4 (12)"),
}


def seed_of(source):
    return int(hashlib.sha256(f"mdfm-lockbox-v1:{source}".encode()).hexdigest()[:16], 16)


def round_half_up(x):
    return int(x + 0.5)


def pool_of(source, persons, gof):
    rule = LOCKBOX[source][1]
    if rule == "not_HC":
        return [p for p in persons if gof(source, p) not in ("HC", "")]
    if rule:
        return [p for p in persons if gof(source, p) in rule]
    return list(persons)


def draw_lockbox(persons_by_source, gof):
    """Lockbox rows (source, lane, person_id, group, rule): per source a seeded draw per group, plus the
    amass subsets."""
    rows = []
    for source, (lane, _, ref) in LOCKBOX.items():
        persons = pool_of(source, sorted(persons_by_source.get(source, set())), gof)
        target = round_half_up(0.2 * len(persons))
        groups = collections.defaultdict(list)
        for p in persons:
            groups[gof(source, p)].append(p)
        rng = random.Random(seed_of(source))
        # allocate per group by largest remainder so the group shares follow the pool
        quotas = {g: 0.2 * len(v) for g, v in groups.items()}
        alloc = {g: int(q) for g, q in quotas.items()}
        rest = target - sum(alloc.values())
        for g in sorted(quotas, key=lambda g: (-(quotas[g] - int(quotas[g])), g))[:max(rest, 0)]:
            alloc[g] += 1
        for g in sorted(groups):
            members = sorted(groups[g])
            rng.shuffle(members)
            for p in sorted(members[:alloc[g]]):
                rows.append({"source": source, "lane": lane, "person_id": p, "group": g, "rule": ref})
    # amass G1 hold-out: whole subsets in seeded order until at least 50 persons
    subsets = collections.defaultdict(set)
    for p in persons_by_source.get("amass", set()):
        parts = p.split(":")
        subsets[parts[2] if len(parts) > 2 else "unknown"].add(p)
    order = sorted(subsets)
    random.Random(seed_of("amass")).shuffle(order)
    held, taken = [], 0
    for sub in order:
        if taken >= 50:
            break
        held.append(sub)
        taken += len(subsets[sub])
    for sub in held:
        for p in sorted(subsets[sub]):
            rows.append({"source": "amass", "lane": "body", "person_id": p, "group": sub,
                         "rule": "design G1 (50 subjects held out by subset and id)"})
    rows.sort(key=lambda r: (r["source"], r["person_id"]))
    return rows


def group_counts(rows):
    """{source: 'n (group n, ...)'} of lockbox rows."""
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["source"]][r["group"]] += 1
    return {s: f"{sum(c.values())} ({', '.join(f'{g} {n}' for g, n in sorted(c.items()))})" for s, c in by.items()}


# ---------------------------------------------------------------- build
def main():
    args = parse_args(__doc__.splitlines()[0])
    ma = args.moveatlas
    out_t = os.path.join(TREE, "reports", "manifest_tiers")
    out_l = os.path.join(TREE, "reports", "lockbox")
    cache = os.path.join(TREE, "cache", "manifest")
    for d in (out_t, out_l, cache):
        os.makedirs(d, exist_ok=True)
    args.out = out_t
    prov = write_provenance(args)

    # ------------------------------------------------ records: one row per record
    files = keyed_manifests(ma)
    records, by_uid = [], {}
    rows_v1_1 = collections.defaultdict(set)      # (source, family) -> v1.1 identities, to report what changed
    rows_v1_2 = collections.defaultdict(set)
    fixed_by = collections.defaultdict(set)
    locs = location_sources([(b, f, r) for b, f, r, _ in files])
    for base, fam, rows, keys in files:
        for r, key in zip(rows, keys):
            source = ALIAS.get(r["source"], r["source"])
            bkey, is_path = base_key(r, fam)
            uid = record_uid(source, fam, r["subject_ns"], key, is_path)
            rows_v1_1[(source, fam)].add((r["subject_ns"], bkey))
            rows_v1_2[(source, fam)].add(uid)
            if (r["source"], fam) in locs:
                fixed_by[(source, fam)].add("location suffix")
            if "#" in key and not is_path:
                fixed_by[(source, fam)].add("occurrence suffix")
            if is_path and key != uid[3] + ".npz":
                fixed_by[(source, fam)].add("record path without extension")
            rec = by_uid.get(uid)
            if rec is None:
                rec = {"source": source, "manifest_source": r["source"], "modality": fam, "subject_ns": r["subject_ns"],
                       "key": key, "fps": r.get("fps", ""), "n_frames": r.get("n_frames", "")}
                by_uid[uid] = rec
                records.append(rec)
                continue
            if str(r.get("n_frames", "")) != str(rec["n_frames"]):
                sys.exit(f"{base}: {uid} is listed with n_frames {r.get('n_frames')} and {rec['n_frames']}: two records "
                         "share one identity")
            if rec["key"].endswith(".pkl") and key.endswith(".npz"):
                rec["key"] = key
    used_files = [b for b, _, _, _ in files]
    changed_rows = {k: (len(rows_v1_1[k]), len(rows_v1_2[k]), ", ".join(sorted(fixed_by[k])))
                    for k in sorted(rows_v1_2) if len(rows_v1_1[k]) != len(rows_v1_2[k])}
    joins = collections.Counter((r["source"], r["modality"], r["subject_ns"], r["key"]) for r in records)
    require(max(joins.values()) == 1, f"records sharing a join key: {[k for k, v in joins.items() if v > 1][:5]}")

    # ------------------------------------------------ CODY namespaces, from the manifest rows only
    cody_rows = [r for r in records if is_cody_source(r["source"], r["manifest_source"])]
    try:
        measured = measured_cody_namespaces(records)
    except ValueError as e:
        sys.exit(str(e))
    deny = sorted(set(CODY_SOURCES) | set(measured) | {n.lower() for n in DESIGN_NAMESPACES})
    trespass = trespassers(records, deny)          # subject_ns names, before the person map
    if trespass:
        sys.exit(f"non-CODY rows in a CODY namespace ({len(trespass)}): {trespass[:5]}")

    # ------------------------------------------------ persons, tiers, lanes
    pmap = {(r["source"], r["subject_ns"]): r["person_id"]
            for r in rd(os.path.join(TREE, "reports", "person_ids", "person_map.csv"))}
    scope = {r["id"]: r["scope"] for r in rd(os.path.join(TREE, "reports", "data_scope", "data_scope.csv"))}
    unmapped = collections.Counter()
    for rec in records:
        pid = pmap.get((rec["source"], rec["subject_ns"]))
        if pid is None:
            # CODY is outside the person map by design (evaluation only, no statistic computed on it)
            if not is_cody_source(rec["source"], rec["manifest_source"]):
                unmapped[rec["source"]] += 1
            pid = rec["subject_ns"]
        rec["person_id"] = pid
        rec["tier"] = tier_of(rec["source"], rec["subject_ns"], pid, deny)
        rec["lane"] = "neural" if rec["modality"] == "eeg" else ("none" if rec["source"] in NO_BODY_LANE else "body")
        rec["scope"] = scope.get(rec["source"], "absent")
    trespass = trespassers(records, deny)          # again with the person ids of the person map
    if trespass:
        sys.exit(f"non-CODY rows in a CODY namespace ({len(trespass)}): {trespass[:5]}")
    require(all(r["tier"] == "EVAL" for r in cody_rows), "a CODY row is outside tier EVAL")

    # ------------------------------------------------ groups
    tables = {"gaitrec": gaitrec_classes(ma), "ds004584": ds004584_groups(ma), "weargait_controls": weargait_controls(ma)}
    if not tables["weargait_controls"]:
        sys.exit("weargait_pd CONTROLS table not found: every weargait_pd person would be stratified as PD")

    def gof(source, person):
        return group_of(source, person, tables)

    for rec in records:
        rec["group"] = gof(rec["source"], rec["person_id"])
    persons_by_source = collections.defaultdict(set)
    for rec in records:
        persons_by_source[rec["source"]].add(rec["person_id"])
    suffixed = sorted(p for p in persons_by_source["weargait_pd"] if "(control)" in p)
    wrong = [p for p in suffixed if gof("weargait_pd", p) != "control"]
    require(not wrong, f"weargait_pd ids carrying '(control)' stratified as PD: {wrong}")

    # ------------------------------------------------ lockbox
    lock_rows = draw_lockbox(persons_by_source, gof)
    lockset = {(r["source"], r["person_id"]) for r in lock_rows}
    for rec in records:
        rec["lockbox"] = (rec["source"], rec["person_id"]) in lockset
    require(not any(r["lockbox"] for r in cody_rows), "a CODY row is in the lockbox")

    # ------------------------------------------------ changes against v1.1
    lp = os.path.join(out_l, "lockbox_persons.csv")
    prev_path = os.path.join(out_l, "lockbox_persons_v1_1.csv")
    if not os.path.exists(prev_path):
        if not (os.path.exists(lp) and sha256_file(lp) == V1_1_SHA256):
            sys.exit(f"{prev_path} is missing and {lp} is not the v1.1 file (sha256 {V1_1_SHA256})")
        shutil.copyfile(lp, prev_path)
    if sha256_file(prev_path) != V1_1_SHA256:
        sys.exit(f"{prev_path} does not hash to the v1.1 lockbox {V1_1_SHA256}")
    prev = rd(prev_path)
    old = {(r["source"], r["person_id"]): r["group"] for r in prev}
    new = {(r["source"], r["person_id"]): r["group"] for r in lock_rows}
    changes = []
    for source in sorted(set(LOCKBOX) | {s for s, _ in old}):
        persons = persons_by_source.get(source, set()) | {p for s, p in old if s == source}
        for p in sorted(persons):
            k = (source, p)
            if source == "amass":
                g11, g12 = old.get(k, ""), new.get(k, "")
            else:
                g11 = old.get(k) or group_of(source, p, tables, strip_suffix=False)
                g12 = new.get(k) or gof(source, p)
            change = "entered" if k in new and k not in old else ("left" if k in old and k not in new else "none")
            if change != "none" or g11 != g12:
                changes.append({"source": source, "person_id": p, "change": change, "lockbox_v1_1": k in old,
                                "lockbox_v1_2": k in new, "group_v1_1": g11, "group_v1_2": g12})
    moved = sorted({c["source"] for c in changes})
    require(set(moved) <= {"weargait_pd"}, f"sources other than weargait_pd changed against v1.1: {moved}")
    before, after = group_counts(prev), group_counts(lock_rows)

    # ------------------------------------------------ write
    rec_path = os.path.join(cache, "manifest_records.csv.gz")
    records.sort(key=lambda r: (r["source"], r["modality"], r["subject_ns"], r["key"]))
    # gzip header time fixed at 0: a rebuild from the same inputs gives the same bytes
    with open(rec_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_COLS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    h = hashlib.sha256()
    for r in records:
        h.update(("\t".join(str(r[c]) for c in RECORD_COLS) + "\n").encode())
    records_hash = h.hexdigest()

    with open(lp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOCK_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(lock_rows)
    lock_hash = sha256_file(lp)
    with open(os.path.join(out_l, "lockbox.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{lock_hash}  lockbox_persons.csv\n")
    change_cols = ["source", "person_id", "change", "lockbox_v1_1", "lockbox_v1_2", "group_v1_1", "group_v1_2"]
    with open(os.path.join(out_l, "v1_2_changes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=change_cols, lineterminator="\n")
        w.writeheader()
        w.writerows(changes)

    by = collections.defaultdict(lambda: {"records": 0, "persons": set(), "lock_records": 0, "lock_persons": set()})
    for rec in records:
        b = by[(rec["source"], rec["modality"], rec["tier"], rec["lane"])]
        b["records"] += 1
        b["persons"].add(rec["person_id"])
        if rec["lockbox"]:
            b["lock_records"] += 1
            b["lock_persons"].add(rec["person_id"])
    with open(os.path.join(out_t, "tiers_by_source.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "modality", "tier", "lane", "records", "persons", "lockbox_records", "lockbox_persons"])
        for k in sorted(by):
            b = by[k]
            w.writerow([*k, b["records"], len(b["persons"]), b["lock_records"], len(b["lock_persons"])])

    tier_persons = collections.defaultdict(set)
    for rec in records:
        tier_persons[rec["tier"]].add(f"{rec['source']}|{rec['person_id']}")
    hashes = {"version": VERSION, "records_file": "cache/manifest/manifest_records.csv.gz (gitignored)",
              "records_columns": RECORD_COLS, "records_sha256": records_hash, "records": len(records),
              "records_file_sha256": sha256_file(rec_path),
              "lockbox_sha256": lock_hash,
              "lockbox_v1_1": {"file": "reports/lockbox/lockbox_persons_v1_1.csv", "sha256": V1_1_SHA256},
              "tier_person_list_sha256": {t: hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest()
                                          for t, v in sorted(tier_persons.items())},
              "cody_namespaces": deny, "manifest_files": used_files, "moveatlas_head": prov["moveatlas_head"],
              "unmapped_records_by_source": dict(unmapped)}
    with open(os.path.join(out_t, "hashes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(hashes, f, indent=1)
    denylist = {"version": VERSION, "namespaces": deny,
                "namespaces_measured": measured,
                "measured_from": "subject_ns before the first colon of the CODY rows (source or manifest_source "
                                 "starting with 'cody') of the pooled manifests; no CODY container or label opened",
                "namespaces_design": list(DESIGN_NAMESPACES),
                "match": "an id whose part before the first colon, lowercased, is in namespaces",
                "prefix_rule": "any source, manifest_source, subject_ns or person_id starting with 'cody', any case",
                "non_cody_rows_in_namespaces": 0,
                "canary_subject_ns": CANARY,
                "rule": "a training batch that contains any denylisted or canary id aborts the run"}
    with open(os.path.join(out_t, "cody_denylist.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(denylist, f, indent=1)

    # ------------------------------------------------ summary
    tiers = collections.Counter(r["tier"] for r in records)
    tp = {t: len(v) for t, v in tier_persons.items()}
    md = [f"# Manifest tiers and lockbox (T0.3, T0.4), version {VERSION}", "",
          "Built by scripts/build_tiers_and_lockbox.py; rules in its docstring; hashes in "
          "reports/manifest_tiers/hashes.json and reports/lockbox/lockbox.sha256.", "",
          f"Records: {len(records)}; non-CODY records whose subject_ns has no person map row: {sum(unmapped.values())}.", "",
          "| tier | records | persons |", "|---|---|---|"]
    md += [f"| {t} | {tiers[t]} | {tp[t]} |" for t in sorted(tiers)]
    md += ["", "## One row per record (v1.2)", "",
           "Sources whose row count changed from the v1.1 rule (one row per source, family, subject_ns and key, "
           "with no location or occurrence suffix and a record_path with its extension) to the v1.2 identity:", "",
           "| source | family | rows v1.1 | rows v1.2 | rule that separates or merges the rows |", "|---|---|---|---|---|"]
    md += [f"| {s} | {f} | {a} | {b} | {why} |" for (s, f), (a, b, why) in changed_rows.items()]
    md += ["", "## CODY denylist", "",
           f"Namespaces of the CODY rows of the manifests (subject_ns before the first colon, {len(cody_rows)} rows, "
           f"no container or label opened): {', '.join(measured)}. Design names: {', '.join(DESIGN_NAMESPACES)}. "
           f"Denylist (reports/manifest_tiers/cody_denylist.json, compared lowercased): {', '.join(deny)}. "
           "Non-CODY rows in a denylisted namespace: 0 (the build stops otherwise); CODY rows outside EVAL or in the "
           "lockbox: 0.", "",
           "## Lockbox", "", "| source | lane | pool persons | lockbox persons | groups |", "|---|---|---|---|---|"]
    lb = collections.defaultdict(list)
    for r in lock_rows:
        lb[r["source"]].append(r)
    for s in sorted(lb):
        pool = len(persons_by_source["amass"]) if s == "amass" else len(pool_of(s, persons_by_source.get(s, set()), gof))
        g = collections.Counter(r["group"] for r in lb[s])
        gs = ", ".join(f"{k} {v}" for k, v in sorted(g.items())) if s != "amass" else "subsets " + ", ".join(sorted(g))
        md.append(f"| {s} | {lb[s][0]['lane']} | {pool} | {len(lb[s])} | {gs} |")
    wg = collections.Counter(gof("weargait_pd", p) for p in persons_by_source["weargait_pd"])
    md += ["", f"Lockbox persons: {len(lock_rows)}; lockbox records: {sum(1 for r in records if r['lockbox'])}.",
           f"Lockbox file SHA-256: {lock_hash}.", f"Records SHA-256: {records_hash}.", "",
           f"weargait_pd controls read from the source CONTROLS table: {len(tables['weargait_controls'])}; weargait_pd "
           f"persons by group: {', '.join(f'{k} {v}' for k, v in sorted(wg.items()))}; ids carrying ' (control)': "
           f"{len(suffixed)} ({', '.join(suffixed)}), all controls.", "",
           "## Changes from v1.1", "",
           f"v1.1 file: reports/lockbox/lockbox_persons_v1_1.csv (SHA-256 {V1_1_SHA256}). Every person whose "
           "membership or group changed: reports/lockbox/v1_2_changes.csv. Sources that changed: "
           f"{', '.join(moved) or 'none'} (the build stops if another source changes).", "",
           "| source | v1.1 persons (groups) | v1.2 persons (groups) | entered | left |", "|---|---|---|---|---|"]
    for s in sorted(set(before) | set(after)):
        ent = sum(1 for c in changes if c["source"] == s and c["change"] == "entered")
        lft = sum(1 for c in changes if c["source"] == s and c["change"] == "left")
        md.append(f"| {s} | {before.get(s, '0')} | {after.get(s, '0')} | {ent} | {lft} |")
    fixed = ", ".join(s for s, _ in changed_rows)
    md += ["", "Frozen on the day it was built. A change needs a new version and a written reason. v1.1 (2026-09-28, "
           "before any use): the weargait_pd group rule read controls from the id prefix HC and stratified the WHC "
           "and NLS controls as PD; it now reads the source CONTROLS table. Seeds are unchanged, so every other "
           f"source draws the same persons as in v1. v{VERSION} ({VERSION_DATE}, before any training): the "
           f"{len(suffixed)} weargait_pd controls with a suffixed id ({', '.join(suffixed)}) were stratified as PD, "
           "because the CONTROLS lookup kept the ' (control)' suffix; the suffix is now removed before the lookup. "
           f"Manifest rows fixed for {len(changed_rows)} sources ({fixed}): one row per record. The CODY denylist "
           "lists the namespaces the CODY rows use and the design's names. Seeds are unchanged; only weargait_pd "
           "changed membership."]
    text = "\n".join(md) + "\n"
    require(chr(0x2014) not in text, "summary.md would hold an em-dash")
    for d in (out_l, out_t):
        with open(os.path.join(d, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
