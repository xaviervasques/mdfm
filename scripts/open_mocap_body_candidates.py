"""Candidate train sources of the open body store (step M3 of docs/MODEL_OPEN_PORT_PLAN.md), measured on the tiers
of the A23 rerun: the open motion capture of 2026-11-12 beside gait120 and stroke_gait_vancriekinge able-bodied.

Reads (read-only): cache/manifest/manifest_records.csv.gz (scripts/build_tiers_and_lockbox.py; its SHA-256 checked
against reports/manifest_tiers/hashes.json before use), reports/open_mocap/cmu_duplicate_folders.csv
(scripts/open_mocap_scope_persons.py), and, for the open sources only, the meta.json and joint_mask member of each
record of MoveAtlas datasets/pose/<id>/processed/canonical/*.npz (15 booleans per record: which canonical joints
the source's marker set or skeleton gives). A source with any EVAL or lockbox row is never opened (shida_pd_gait is
EVAL); no coordinate array is read. gait120 and stroke able-bodied come from mdfm.data.sources.folds() and the
manifest's frame counts, no container opened.

Rules, in this order, per record of the open sources (the first that applies is the record's status):
1. eval: tier EVAL (shida_pd_gait, the probable care_pd persons); lockbox: a lockbox row (none expected; the run
   stops if one appears).
2. duplicate: a cmu_mocap folder listed as the byte-identical copy of a lower-numbered folder; possible_duplicate: a
   second or later record of one (subject_ns, take) (key suffix #n; deepdance's Subject001 and Subject1 files).
3. joint_absent: a canonical joint without a landmark in the record (joint_mask False): the body fit would leave that
   joint to the solver's prior, and the decoder would learn a prior's pose as truth. Whole sources fail here when
   their marker set lacks the joint: horst_gait_fullbody (knees and ankles: thigh and shank clusters only, the
   static calibration not built), schreiber_moissenet_gait and vanderzee_walking (head). Moore's lower-set trials
   and the cmu_mocap takes without head markers fail record by record.
4. dev: the person sits in the dev fold, drawn as mdfm.data.sources.folds() draws gait120's: 20 percent of the
   source's persons, rounded half up, numpy default_rng seeded by SHA-256 of "20260928:<source>", over the sorted
   person ids of the tiers file (cmu_mocap's folder groups and AMASS ids of the person map count as persons). A
   source of one person (style100) has no subject-level split: train only.
5. train: a candidate.
Hours: n_frames / fps of the tiers file, per record.

Writes reports/open_mocap/body_store_candidates.md, body_store_candidates.csv (one row per source and status),
body_store_folds.csv (source, person_id, fold), body_store_candidates.json (hours, person-list SHA-256 per source and
fold, inputs' SHA-256) and _provenance_body_candidates.json.
Run: <conda>/envs/mdfm/python.exe scripts/open_mocap_body_candidates.py --moveatlas <moveatlas>
"""
import argparse
import collections
import csv
import datetime
import glob
import gzip
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TREE)

OPEN = ("cmu_mocap", "ceti_age_kinematics", "horst_gait_fullbody", "schreiber_moissenet_gait", "lencioni_gait_stairs",
        "push_recovery_perturbations", "moore_perturbed_gait", "shida_pd_gait", "deepdance", "umons_taichi",
        "notredame_noncyclic", "style100", "vanderzee_walking")
# what each source records the body with (MoveAtlas datacards, c561fc2): the fit of step M3 differs for a skeleton
CAPTURE = {"ceti_age_kinematics": "inertial suit, BVH skeleton (joint centres and root translation are the suit's "
                                  "estimate)",
           "style100": "BVH skeleton (no markers)"}
CANONICAL_15 = ["pelvis", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
STATUSES = ("train", "dev", "joint_absent", "possible_duplicate", "duplicate", "eval")
FOLD_SHARE = 0.2
SEED = 20260928


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def person_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def hours(rows):
    return sum(int(r["n_frames"]) / float(r["fps"]) for r in rows) / 3600.0


def joint_masks(ma, source):
    """{(subject_ns, take): [joint_mask (15,) bool, ...]} in container record order, plus {..: [marker_set]}."""
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(f"{ma}/datasets/pose/{source}/processed/canonical/*.npz")):
        if os.path.basename(p).startswith("._"):
            continue
        with zipfile.ZipFile(p) as z:
            meta = json.loads(z.read("meta.json"))
            for i, rec in enumerate(meta["records"]):
                if rec.get("source") != source:
                    sys.exit(f"{p}: record {i} of source {rec.get('source')}")
                tok = rec["joint_mask"]["$npy"]
                member = f"r{i:06d}/{tok}.npy"
                m = np.load(io.BytesIO(z.read(member)), allow_pickle=False)
                if m.shape != (15,) or m.dtype != bool:
                    sys.exit(f"{p}: record {i} joint_mask {m.shape} {m.dtype}")
                out[(rec["subject_ns"], str(rec["take"]))].append(m)
    return out


def draw_dev(source, persons):
    """Dev persons of a source by the rule of mdfm.data.sources.folds() (module docstring)."""
    from mdfm.data import sources as S
    persons = sorted(persons)
    n_hold = S._round_half_up(FOLD_SHARE * len(persons))
    rng = np.random.default_rng(S._seed(SEED, source))
    return {persons[i] for i in rng.permutation(len(persons))[:n_hold]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    a = ap.parse_args()
    ma = a.moveatlas.replace("\\", "/").rstrip("/")
    os.environ["MOVEATLAS_ROOT"] = ma
    sys.stdout.reconfigure(encoding="utf-8")
    from _common import moveatlas_head
    from mdfm.data import sources as S

    out = os.path.join(TREE, "reports", "open_mocap")
    rec_path = os.path.join(TREE, "cache", "manifest", "manifest_records.csv.gz")
    with open(os.path.join(TREE, "reports", "manifest_tiers", "hashes.json"), encoding="utf-8") as f:
        hashes = json.load(f)
    rec_sha = sha256_file(rec_path)
    if rec_sha != hashes["records_file_sha256"]:
        sys.exit(f"{rec_path} does not hash to records_file_sha256 of hashes.json")
    with gzip.open(rec_path, "rt", encoding="utf-8", newline="") as f:
        records = [r for r in csv.DictReader(f)]
    dup_path = os.path.join(out, "cmu_duplicate_folders.csv")
    with open(dup_path, encoding="utf-8", newline="") as f:
        dup_ns = {r["subject_ns_duplicate"] for r in csv.DictReader(f)}

    # ---------------------------------------------------------------- the open sources
    by_source = collections.defaultdict(list)
    for r in records:
        if r["source"] in OPEN:
            by_source[r["source"]].append(r)
    missing = [s for s in OPEN if s not in by_source]
    if missing:
        sys.exit(f"no tiers rows for {missing}")
    status_rows, folds_out, per_source = [], [], collections.OrderedDict()
    for s in OPEN:
        rows = by_source[s]
        if any(r["lockbox"] == "True" for r in rows):
            sys.exit(f"{s}: a lockbox row; the open sources draw no lockbox person")
        tiers = {r["tier"] for r in rows}
        info = {"tier": "/".join(sorted(tiers)), "records": len(rows), "hours": hours(rows),
                "persons": len({r["person_id"] for r in rows}), "capture": CAPTURE.get(s, "optical markers"),
                "absent_joints": {}}
        status = {}
        if tiers != {"RELEASE"}:
            for r in rows:
                status[id(r)] = "eval"
            info.update(fold_dev_persons=0, fold_train_persons=0, fold_rule="none: never trained on")
        else:
            masks = joint_masks(ma, s)
            persons = sorted({r["person_id"] for r in rows})
            dev = draw_dev(s, persons) if len(persons) > 1 else set()
            for p in persons:
                folds_out.append({"source": s, "person_id": p, "fold": "dev" if p in dev else "train"})
            info.update(fold_dev_persons=len(dev), fold_train_persons=len(persons) - len(dev),
                        fold_rule="one person: no subject-level split, train only" if len(persons) == 1 else
                        f"dev {FOLD_SHARE:.0%} of persons, rounded half up, seed SHA-256('{SEED}:{s}')",
                        dev_sha256=person_hash(dev), train_sha256=person_hash(set(persons) - dev))
            absent = collections.Counter()
            for r in rows:
                key = r["key"]
                take, _, occ = key.partition("#")
                n = int(occ) if occ else 1
                found = masks.get((r["subject_ns"], take), [])
                if len(found) < n:
                    sys.exit(f"{s}: no container record for {r['subject_ns']} {key}")
                m = found[n - 1]
                if r["subject_ns"] in dup_ns:
                    st = "duplicate"
                elif occ:
                    st = "possible_duplicate"
                elif not m.all():
                    st = "joint_absent"
                    absent.update(CANONICAL_15[j] for j in np.flatnonzero(~m))
                elif r["person_id"] in dev:
                    st = "dev"
                else:
                    st = "train"
                status[id(r)] = st
            info["absent_joints"] = dict(sorted(absent.items()))
        for st in STATUSES:
            sel = [r for r in rows if status[id(r)] == st]
            info[f"{st}_records"] = len(sel)
            info[f"{st}_hours"] = hours(sel)
            info[f"{st}_persons"] = len({r["person_id"] for r in sel})
            if sel:
                status_rows.append({"source": s, "status": st, "records": len(sel),
                                    "persons": len({r["person_id"] for r in sel}), "hours": f"{hours(sel):.4f}"})
        per_source[s] = info

    # ---------------------------------------------------------------- gait120 and stroke able-bodied (the plan's 2.41 h)
    specs = {n: S.SOURCES[n] for n in ("gait120", "stroke_gait_vancriekinge")}
    fr = S.folds(specs=specs)
    pilot = collections.OrderedDict()
    for n, spec in specs.items():
        fold = dict(zip(fr.loc[fr["source"] == n, "person_id"], fr.loc[fr["source"] == n, "fold"]))
        rows = [r for r in records if r["source"] == n and r["modality"] == "pose" and r["tier"] == "RELEASE"
                and r["lockbox"] != "True" and (spec.groups is None or r["group"] in spec.groups)]
        tr = [r for r in rows if fold.get(r["person_id"]) == "train"]
        dv = [r for r in rows if fold.get(r["person_id"]) == "dev"]
        pilot[n] = {"unit": spec.unit, "records": len(rows), "hours": hours(rows), "train_records": len(tr),
                    "train_hours": hours(tr), "train_persons": len({r["person_id"] for r in tr}),
                    "dev_hours": hours(dv), "dev_persons": len({r["person_id"] for r in dv}),
                    "unfolded_records": len(rows) - len(tr) - len(dv)}

    # ---------------------------------------------------------------- write
    write = lambda name, cols, rows: _write_csv(os.path.join(out, name), cols, rows)  # noqa: E731
    write("body_store_candidates.csv", ["source", "status", "records", "persons", "hours"], status_rows)
    write("body_store_folds.csv", ["source", "person_id", "fold"], folds_out)
    open_train = sum(v["train_hours"] for v in per_source.values())
    pilot_train = sum(v["train_hours"] for v in pilot.values())
    pilot_all = sum(v["hours"] for v in pilot.values())
    open_dev = sum(v["dev_hours"] for v in per_source.values())
    pilot_dev = sum(v["dev_hours"] for v in pilot.values())
    open_persons = sum(v["train_persons"] for v in per_source.values())
    pilot_persons = sum(v["train_persons"] for v in pilot.values())
    biggest = max(per_source, key=lambda k: per_source[k]["train_hours"])
    share = per_source[biggest]["train_hours"] / (open_train + pilot_train)
    head = moveatlas_head(ma)
    summary = {"moveatlas_head": head, "records_sha256": hashes["records_sha256"], "records_file_sha256": rec_sha,
               "cmu_duplicate_folders_sha256": sha256_file(dup_path), "open": per_source, "pilot": pilot,
               "open_train_hours": open_train, "pilot_train_hours": pilot_train, "pilot_all_folds_hours": pilot_all,
               "total_train_hours": open_train + pilot_train, "open_dev_hours": open_dev, "pilot_dev_hours": pilot_dev,
               "open_train_persons": open_persons, "pilot_train_persons": pilot_persons,
               "largest_source": biggest, "largest_source_share": share}
    with open(os.path.join(out, "body_store_candidates.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)

    md = ["# Open body store: candidate train sources (step M3), measured after the A23 tiers rerun", "",
          f"Built by scripts/open_mocap_body_candidates.py at MoveAtlas {head.split(' ')[0]}, tiers records_sha256 "
          f"{hashes['records_sha256'][:12]}...; rules in its docstring. Hours are n_frames / fps of the tiers file.", "",
          "## Open motion capture (MoveAtlas c561fc2)", "",
          "| source | tier | capture | persons (train fold / dev fold) | all hours | train candidate hours (records) | dev h | "
          "joint absent h | duplicate h | joints absent (records) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in per_source.items():
        dup_h = v["duplicate_hours"] + v["possible_duplicate_hours"]
        ab = ", ".join(f"{k} {n}" for k, n in v["absent_joints"].items()) or "none"
        pt = (f"{v['persons']} (EVAL)" if v["tier"] != "RELEASE" else
              f"{v['persons']} ({v['fold_train_persons']} / {v['fold_dev_persons']})")
        md.append(f"| {s} | {v['tier']} | {v['capture']} | {pt} | {v['hours']:.2f} | {v['train_hours']:.2f} "
                  f"({v['train_records']}) | {v['dev_hours']:.2f} | {v['joint_absent_hours']:.2f} | {dup_h:.2f} | {ab} |")
    md += ["", f"Open train candidate hours: {open_train:.2f} (records with every canonical joint given, train fold, "
           "no duplicate).", "", "## Stage-1 marker sources (mdfm.data.sources.folds)", "",
           "| unit | pose records | all folds h | train h (persons) | dev h (persons) |", "|---|---|---|---|---|"]
    for n, v in pilot.items():
        md.append(f"| {v['unit']} | {v['records']} | {v['hours']:.2f} | {v['train_hours']:.2f} ({v['train_persons']}) "
                  f"| {v['dev_hours']:.2f} ({v['dev_persons']}) |")
    md += ["", f"Stage-1 marker sources: {pilot_all:.2f} h over all folds, {pilot_train:.2f} h in the train folds.",
           f"Body store candidate total, train folds: {open_train + pilot_train:.2f} h of {open_persons + pilot_persons} "
           f"persons ({open_persons} open, {pilot_persons} stage-1); dev folds: {open_dev + pilot_dev:.2f} h ({open_dev:.2f} "
           f"open, {pilot_dev:.2f} stage-1). Largest source: {biggest}, {share:.1%} of the train-fold total.", ""]
    text = "\n".join(md) + "\n"
    if "\u2014" in text:
        sys.exit("em-dash in body_store_candidates.md")
    with open(os.path.join(out, "body_store_candidates.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    prov = {"script": os.path.basename(__file__), "moveatlas_root": ma, "moveatlas_head": head,
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "records_file_sha256": rec_sha}
    with open(os.path.join(out, "_provenance_body_candidates.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(text)


def _write_csv(path, cols, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
