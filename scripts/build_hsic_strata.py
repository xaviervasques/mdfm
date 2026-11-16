"""S2.8: the HSIC strata table of stage 2 (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7 and section 9 item 10).

Writes cache/hsic_strata/strata.csv (person_id, stratum: one row per person of the stage-2 window store, non-lockbox
persons only, asserted), reports/hsic_strata/legend.csv (stratum, class and the source values each class gathers),
reports/hsic_strata/summary.md (persons per stratum and source, per fold, the table's SHA-256 and the inputs'),
reports/hsic_strata/strata.sha256 and _provenance.json (the MoveAtlas HEAD). No module of mdfm/ opens legend.csv
(tests/test_hsic_strata.py greps for it): the code reads the integers of strata.csv through mdfm/data/strata.py and
never a class name.

Classes [cfg names, section 9 item 10, the owner's decision of 2026-10-06 as proposed]: healthy, PD, other movement
disorder, neurological other, orthopaedic, and unknown for the persons whose diagnosis no record or source table
carries (imu_inthewild_iprognosis: the diagnosis sits only in a raw pickle that no probe has read). Integers in the
order of CLASSES; healthy is 0, the stratum of every window that carries none (the synthetic windows and the stage-1
replay, Window.hsic_stratum's default), so that the healthy persons of the stage-2 units share their strata with them.

Sources of each person's class, never typed by hand:
- weargait_pd (unit weargait_pd:PD): the index group PD; stroke_gait_vancriekinge (unit stroke): the index group
  stroke, a neurological condition.
- clinical_gait_multipatho: the index group, mapped through the cohort folder that holds it in the deposit
  (raw/dataset.zip, members dataset/data/<cohort>/<group>/...; only the member names are listed, no file is read):
  cohort healthy to healthy, ortho to orthopaedic, neuro to neurological other except the group PD, which is PD.
  The run checks that every index group sits in exactly one cohort folder.
- pads: the condition column of the source table datasets/imu/pads/processed/pads_subject_table.csv (the record
  field condition of ingest_pads.py, reports/stage2_sources/diagnosis.csv), rows of the store's persons only:
  Healthy to healthy, Parkinson's to PD, Atypical Parkinsonism, Essential Tremor and Other Movement Disorders to other
  movement disorder, Multiple Sclerosis to neurological other. Atypical parkinsonism is a movement disorder other than
  Parkinson's disease, so it is not PD here (a decision of this task, the owner's to overturn).
- imu_inthewild_iprognosis: unknown.
A value without a rule, a person with two classes or a store person without a class stops the run.

Persons: those of the stage-2 window store's records.csv tables (cache/window_store_s2, MDFM_WINDOW_STORE_S2; every
fold, so that the dev hook's held-out windows carry their stratum too), checked against the lockbox list
(reports/lockbox/lockbox_persons.csv with its SHA-256), the CODY names and the index tiers (every row RELEASE). The
group column of the index is read through mdfm.data.sources.rows (canonical selection, no record opened) and stays in
this script: it never reaches a window.

Usage (mdfm environment, CPU):
  <conda>/envs/mdfm/python.exe scripts/build_hsic_strata.py
"""
import collections
import csv
import datetime
import hashlib
import json
import os
import sys
import zipfile

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import moveatlas_head  # noqa: E402

CLASSES = ("healthy", "PD", "other_movement_disorder", "neurological_other", "orthopaedic", "unknown")
UNIT_GROUP_CLASS = {("weargait_pd", "PD"): "PD", ("stroke_gait_vancriekinge", "stroke"): "neurological_other"}
MULTIPATHO = "clinical_gait_multipatho"
MULTIPATHO_ZIP = "datasets/imu/clinical_gait_multipatho/raw/dataset.zip"
COHORT_CLASS = {"healthy": "healthy", "neuro": "neurological_other", "ortho": "orthopaedic"}
GROUP_OVERRIDE = {"PD": "PD"}         # a neuro group that has its own class
PADS = "pads"
PADS_TABLE = "datasets/imu/pads/processed/pads_subject_table.csv"
PADS_CLASS = {"Healthy": "healthy", "Parkinson's": "PD", "Atypical Parkinsonism": "other_movement_disorder",
              "Essential Tremor": "other_movement_disorder", "Other Movement Disorders": "other_movement_disorder",
              "Multiple Sclerosis": "neurological_other"}
UNKNOWN_SOURCES = ("imu_inthewild_iprognosis",)
OUT_TABLE = os.path.join(TREE, "cache", "hsic_strata", "strata.csv")
OUT_REPORT = os.path.join(TREE, "reports", "hsic_strata")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def cohort_groups(zip_path):
    """{group: set of cohort folders} of the deposit's member names dataset/data/<cohort>/<group>/... (names only)."""
    out = collections.defaultdict(set)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) >= 5 and parts[0] == "dataset" and parts[1] == "data" and parts[2] and parts[3]:
                out[parts[3]].add(parts[2])
    return dict(out)


def multipatho_classes(cohorts):
    """{group: class} of the multipatho groups from their cohort folders (module docstring); ValueError for a group
    in more or fewer than one cohort folder or a cohort without a rule."""
    out = {}
    for g, cs in sorted(cohorts.items()):
        if len(cs) != 1:
            raise ValueError(f"{MULTIPATHO} group {g} sits in the cohort folders {sorted(cs)}")
        c = next(iter(cs))
        if c not in COHORT_CLASS:
            raise ValueError(f"{MULTIPATHO} cohort folder {c!r} has no class rule")
        out[g] = GROUP_OVERRIDE.get(g, COHORT_CLASS[c])
    return out


def assign(persons, groups, pads_condition, multipatho):
    """{person_id: class} (module docstring). persons: {person_id: (source, subject_ns)} of the store; groups:
    {person_id: set of index group values}; pads_condition: {subject_ns: condition} of the pads table;
    multipatho: {group: class}. ValueError for a person without exactly one class."""
    out, bad = {}, []
    for pid, (src, sns) in sorted(persons.items()):
        gs = sorted(groups.get(pid, ()))
        if src in UNKNOWN_SOURCES:
            cls = {"unknown"}
        elif src == PADS:
            cond = pads_condition.get(sns)
            cls = {PADS_CLASS[cond]} if cond in PADS_CLASS else set()
            if cond is not None and cond not in PADS_CLASS:
                bad.append(f"{pid}: pads condition {cond!r} without a rule")
        elif src == MULTIPATHO:
            cls = {multipatho[g] for g in gs if g in multipatho}
            if any(g not in multipatho for g in gs):
                bad.append(f"{pid}: {MULTIPATHO} group(s) {[g for g in gs if g not in multipatho]} without a rule")
        else:
            cls = {UNIT_GROUP_CLASS[(src, g)] for g in gs if (src, g) in UNIT_GROUP_CLASS}
            if any((src, g) not in UNIT_GROUP_CLASS for g in gs):
                bad.append(f"{pid}: {src} group(s) {gs} without a rule")
        if len(cls) != 1:
            bad.append(f"{pid} ({src}): classes {sorted(cls)}")
            continue
        out[pid] = next(iter(cls))
    if bad:
        raise ValueError(f"{len(bad)} persons without exactly one class: {bad[:10]}")
    return out


def table_text(classes):
    """The strata.csv text (LF): header person_id,stratum, one row per person sorted by person id."""
    lines = ["person_id,stratum"] + [f"{p},{CLASSES.index(c)}" for p, c in sorted(classes.items())]
    for p in classes:
        if "," in p or "\n" in p or '"' in p:
            raise ValueError(f"person id {p!r} needs quoting")
    return "\n".join(lines) + "\n"


def store_persons(root):
    """{person_id: (source, subject_ns)} and {person_id: fold} of the stage-2 store's records.csv tables, and the
    SHA-256 of each table."""
    persons, folds, shas = {}, {}, {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "records.csv")
        if name.startswith("._") or not os.path.isfile(path):
            continue
        shas[name] = sha256_file(path)
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                pid = r["person_id"]
                if persons.setdefault(pid, (r["source"], r["subject_ns"])) != (r["source"], r["subject_ns"]):
                    raise ValueError(f"{pid}: two (source, subject_ns) pairs in the store")
                if folds.setdefault(pid, r["fold"]) != r["fold"]:
                    raise ValueError(f"{pid}: two folds in the store")
    return persons, folds, shas


def refuse_store_persons(persons, lock, tiers):
    """canonical.Refused for a lockbox person or a CODY name among the store's persons {person_id: (source,
    subject_ns)}; window_store.refuse_tier for a person whose index tier (tiers, window_store.person_tiers) is not
    RELEASE. lock: the lockbox person ids (window_store.lockbox_persons)."""
    from mdfm.data import canonical
    from mdfm.data import window_store as WS
    for pid, (src, sns) in sorted(persons.items()):
        if pid in lock:
            raise canonical.Refused(f"lockbox person in the stage-2 store: {pid}")
        if canonical.is_cody(src, sns, pid):
            raise canonical.Refused(f"a CODY name in the stage-2 store: {pid}")
        WS.refuse_tier(src, pid, "RELEASE", tiers)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm import paths
    from mdfm.data import canonical
    from mdfm.data import sources as SRC
    from mdfm.data import strata as ST
    from mdfm.data import window_store as WS
    t0 = datetime.datetime.now()
    ma = paths.MOVEATLAS
    root = WS.store_root(kind="s2")
    persons, folds, rec_shas = store_persons(root)
    lock = WS.lockbox_persons()
    refuse_store_persons(persons, lock, WS.person_tiers())
    rows = SRC.rows(sources=sorted({s for s, _ in persons.values()}), specs=SRC.STAGE2_S2A)
    groups = collections.defaultdict(set)
    for pid, g in zip(rows["person_id"].astype(str), rows["group"].astype(str)):
        if pid in persons:
            groups[pid].add(g)
    missing = sorted(set(persons) - set(groups))
    if missing:
        raise ValueError(f"{len(missing)} store persons without an admitted index row, first {missing[:5]}")
    cohorts = cohort_groups(os.path.join(ma, MULTIPATHO_ZIP))
    mp_groups = sorted({g for p, (s, _) in persons.items() if s == MULTIPATHO for g in groups[p]})
    mp = multipatho_classes({g: cohorts.get(g, set()) for g in mp_groups})
    pads_path = os.path.join(ma, PADS_TABLE)
    pads_sns = {sns for s, sns in persons.values() if s == PADS}
    pads_condition = {}
    with open(pads_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["subject_ns"] in pads_sns:          # the store's persons only (no lockbox row is kept)
                if pads_condition.setdefault(r["subject_ns"], r["condition"]) != r["condition"]:
                    raise ValueError(f"pads {r['subject_ns']}: two conditions in the source table")
    classes = assign(persons, groups, pads_condition, mp)
    if set(classes) & lock:
        raise canonical.Refused("a lockbox person in the strata table")
    text = table_text(classes)
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    with open(OUT_TABLE, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    sha = sha256_bytes(text.encode("utf-8"))
    back = ST.load(OUT_TABLE, sha)                    # the reader's checks (header, ids, lockbox, CODY)
    if back.strata != {p: CLASSES.index(c) for p, c in classes.items()}:
        raise RuntimeError("the strata table does not read back")
    os.makedirs(OUT_REPORT, exist_ok=True)
    members = collections.defaultdict(list)
    members["healthy"].append("windows without a table entry (synthetic, stage-1 replay): stratum 0")
    for (src, g), c in sorted(UNIT_GROUP_CLASS.items()):
        members[c].append(f"{src}:{g}")
    for g, c in sorted(mp.items()):
        members[c].append(f"{MULTIPATHO}:{g} (cohort {next(iter(cohorts[g]))})")
    for v, c in sorted(PADS_CLASS.items()):
        members[c].append(f"{PADS}:{v}")
    for s in UNKNOWN_SOURCES:
        members["unknown"].append(f"{s}:all")
    with open(os.path.join(OUT_REPORT, "legend.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["stratum", "class", "members"])
        for i, c in enumerate(CLASSES):
            w.writerow([i, c, " | ".join(members[c])])
    with open(os.path.join(OUT_REPORT, "strata.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha}  cache/hsic_strata/strata.csv\n")
    head = moveatlas_head(ma)
    count = collections.Counter((classes[p], persons[p][0], folds[p]) for p in classes)
    per_cs = collections.Counter((classes[p], persons[p][0]) for p in classes)
    by_class = collections.Counter(classes.values())
    sources = sorted({s for s, _ in persons.values()})
    prov = {"script": "scripts/build_hsic_strata.py", "run_at": t0.isoformat(timespec="seconds"),
            "moveatlas_root": ma, "moveatlas_head": head, "store_root": root.replace("\\", "/"),
            "store_records_sha256": rec_shas, "pads_table_sha256": sha256_file(pads_path),
            "multipatho_zip": MULTIPATHO_ZIP, "multipatho_cohorts": {g: sorted(cohorts[g]) for g in mp},
            "strata_csv": "cache/hsic_strata/strata.csv", "strata_sha256": sha, "persons": len(classes),
            "lockbox_csv_sha256": WS.sha256_file(WS.LOCKBOX_CSV), "lockbox_in_table": 0,
            "classes": list(CLASSES), "persons_per_class": {c: by_class.get(c, 0) for c in CLASSES}}
    with open(os.path.join(OUT_REPORT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    L = ["# HSIC strata of stage 2 (task S2.8)", "",
         "Generated by scripts/build_hsic_strata.py; every number below is read from the table it wrote and from its "
         "inputs. Spec: docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7 and section 9 item 10. MoveAtlas commit: "
         f"{head}. Run {t0.isoformat(timespec='seconds')}.", "",
         f"Table: cache/hsic_strata/strata.csv (gitignored; person_id, stratum), {len(classes)} persons, SHA-256 {sha} "
         "(strata.sha256). A run config names it in losses.hsic_strata with this SHA-256 in losses.hsic_strata_sha256; "
         "mdfm/data/strata.py reads it and the stage-2 window store attaches each window's stratum.", "",
         f"Persons: every person of the stage-2 store root {root.replace(chr(92), '/')} (records.csv of "
         f"{', '.join(sources)}), all folds. Checks: lockbox persons in the table 0 (reports/lockbox/lockbox_persons.csv, "
         f"SHA-256 {prov['lockbox_csv_sha256'][:16]}..., checked against lockbox.sha256), CODY names 0, index rows of "
         "a tier other than RELEASE 0; the table read back through mdfm/data/strata.load equal to the classes "
         "assigned.", "",
         "## Strata", "",
         "Integers in the order of the classes (legend.csv; no module of mdfm/ opens it). Stratum 0 (healthy) is also "
         "the stratum of every window without a table entry: the synthetic windows and the stage-1 replay.", "",
         "| stratum | class | persons | " + " | ".join(sources) + " |",
         "|---|---|---|" + "---|" * len(sources)]
    for i, c in enumerate(CLASSES):
        L.append(f"| {i} | {c} | {by_class.get(c, 0)} | " + " | ".join(str(per_cs.get((c, s), 0)) for s in sources)
                 + " |")
    L += ["", "## Persons per stratum, source and fold", "", "| stratum | source | fold | persons |", "|---|---|---|---|"]
    for (c, s, fo), n in sorted(count.items(), key=lambda kv: (CLASSES.index(kv[0][0]), kv[0][1], kv[0][2])):
        L.append(f"| {CLASSES.index(c)} | {s} | {fo} | {n} |")
    L += ["", "## Where each class comes from", "",
          f"- weargait_pd and stroke_gait_vancriekinge: the index group of the unit ({', '.join(f'{s}:{g} to {c}' for (s, g), c in sorted(UNIT_GROUP_CLASS.items()))}).",
          f"- {MULTIPATHO}: the index group through the cohort folder of {MULTIPATHO_ZIP} that holds it (member "
          "names only): " + ", ".join(f"{g} ({next(iter(cohorts[g]))}) to {c}" for g, c in sorted(mp.items())) + ".",
          f"- pads: the condition column of {PADS_TABLE} (SHA-256 {prov['pads_table_sha256'][:16]}...), rows of the "
          "store's persons: " + ", ".join(f"{v} to {c}" for v, c in sorted(PADS_CLASS.items())) + ".",
          f"- {', '.join(UNKNOWN_SOURCES)}: unknown (no per-person diagnosis on its records; spec section 9 item 10).",
          "", "Inputs: records.csv SHA-256 " + ", ".join(f"{k} {v[:16]}..." for k, v in sorted(rec_shas.items())) + "."]
    with open(os.path.join(OUT_REPORT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print(f"strata table {OUT_TABLE}: {len(classes)} persons, SHA-256 {sha}")
    print("persons per class:", {c: by_class.get(c, 0) for c in CLASSES})


if __name__ == "__main__":
    main()
