"""The open motion capture of 2026-11-12 into MD_FM's data scope and person map, before the tiers rerun (A23, step
M1 of docs/MODEL_OPEN_PORT_PLAN.md).

The lot: 15 registry rows added to MoveAtlas at c561fc2, 13 of them with canonical pose records (LOT, WITH_RECORDS).
scripts/build_tiers_and_lockbox.py reads the person map and the data scope; without rows for these sources it would
give each record person_id = subject_ns and scope "absent".

Reads (read-only): MoveAtlas datasets/_canonical_manifest.csv (source, subject_ns and take of the rows of the 13
sources; no record container is opened), the C3D files of cmu_mocap (raw/allc3d_*/subjects/<n>/*.c3d: their sizes,
and the SHA-256 of the files of folders whose size lists coincide), reports/person_ids/person_map.csv (its AMASS CMU
rows), reports/data_scope/data_scope.csv, and the output of scripts/data_scope.py run at the current MoveAtlas commit
into a temporary folder outside both trees.

1. Data scope: the rows of the 15 ids from that run replace or join theirs in reports/data_scope/data_scope.csv;
   every other row is kept as it is (a full rerun also moves rows decided before this lot, which is not this
   script's change; scope_persons.md lists them). cmu_mocap carries CMU's own terms, which data_scope.py's licence
   parser classes "other or unverified"; docs/decisions/licence_decisions_2026-11-12.csv records the decision.
2. Person map, one row per (source, subject_ns) of the 13 sources:
   - cmu_mocap. A CMU subject folder is a capture session, not a verified person. Folders are joined into one group
     when their C3D file sets are byte-identical (the same takes filed under both actors' numbers) or when AMASS's
     two-actor folders (amass:CMU:CMU_<a>_<b>_<name>) name one person under several numbers; a group of several
     folders takes person_id cmu_mocap:group:<numbers>, so that no split separates them. A folder alone takes the
     person_id of AMASS's row amass:CMU:CMU_<nn> when it exists (rule cmu_subject_number_to_amass_cmu: the same
     persons and takes, AMASS's fits of these very markers), else its own subject_ns. Records of a two-actor take
     under a single folder are noted: the person id applies to the folder.
   - shida_pd_gait: identity, with the probable care_pd identity in the ambiguity column (tier EVAL in MD_FM).
   - the other 11: identity (one subject folder of the deposit, one person), the datacard's caveats in the ambiguity
     column. Cross-source overlaps other than the two named here were not audited.
   The rows replace any earlier rows of the 13 sources in reports/person_ids/person_map.csv (other rows unchanged)
   and are written to reports/person_ids/map_open_mocap.csv, which the combiner of scripts/build_person_map.py reads
   at its next run.

Writes: reports/data_scope/data_scope.csv (merged), reports/person_ids/person_map.csv (merged),
reports/person_ids/map_open_mocap.csv, reports/open_mocap/cmu_duplicate_folders.csv, reports/open_mocap/
scope_persons.md and reports/open_mocap/_provenance_scope_persons.json.
Run: <conda>/python scripts/open_mocap_scope_persons.py --moveatlas <moveatlas>
"""
import argparse
import collections
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _common import moveatlas_head  # noqa: E402

LOT = ("cmu_mocap", "ceti_age_kinematics", "lex3d", "horst_gait_fullbody", "schreiber_moissenet_gait",
       "lencioni_gait_stairs", "push_recovery_perturbations", "moore_perturbed_gait", "shida_pd_gait", "deepdance",
       "umons_taichi", "notredame_noncyclic", "style100", "sign_language_mocap_archive", "vanderzee_walking")
WITHOUT_RECORDS = ("lex3d", "sign_language_mocap_archive")
WITH_RECORDS = tuple(s for s in LOT if s not in WITHOUT_RECORDS)
PM_COLS = ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"]
CMU_SESSION = ("cmu_session_number: a CMU subject folder is a capture session, not a verified person; no actor list "
               "is on disk")
# Caveats of the MoveAtlas datacards (c561fc2), carried into the ambiguity column.
NOTES = {
    "shida_pd_gait": "probable care_pd person: CARE-PD lists Shida among its investigators and holds a BMCLab cohort "
                     "(reports/person_ids/cross_source_overlap.md, cross_source_name_scan.csv); no id link without "
                     "reading care_pd, so the source is EVAL in MD_FM, never trained on",
    "deepdance": "Subject001 and Subject1 are one subject_ns (MoveAtlas ingestion)",
    "style100": "one performer: the source has a single person",
    "moore_perturbed_gait": "id 0 (unloaded trials) skipped at ingestion; subject 12 refused (unlabelled markers)",
}


def rd(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, cols, rows):
    """LF, as the tree stores both merged files (git holds data_scope.csv with LF line ends)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Groups:
    """Union-find over folder numbers."""

    def __init__(self):
        self.up = {}

    def find(self, a):
        self.up.setdefault(a, a)
        while self.up[a] != a:
            self.up[a] = self.up[self.up[a]]
            a = self.up[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.up[max(ra, rb)] = min(ra, rb)


def cmu_folders(ma):
    """{folder number: sorted [(basename, path)]} of the cmu_mocap C3D files, AppleDouble sidecars left out."""
    out = collections.defaultdict(list)
    for p in glob.glob(f"{ma}/datasets/pose/cmu_mocap/raw/allc3d_*/subjects/*/*.c3d"):
        b = os.path.basename(p)
        if b.startswith("._"):
            continue
        out[int(os.path.basename(os.path.dirname(p)))].append((b, p))
    return {k: sorted(v) for k, v in out.items()}


def duplicate_folders(folders):
    """[(a, b, files)] of folder pairs whose C3D files are byte-identical as sets (sizes first, then SHA-256)."""
    by_sizes = collections.defaultdict(list)
    for n, files in folders.items():
        by_sizes[tuple(sorted(os.path.getsize(p) for _, p in files))].append(n)
    pairs = []
    for sizes, nums in sorted(by_sizes.items()):
        if len(nums) < 2:
            continue
        digests = {n: sorted(sha256_file(p) for _, p in folders[n]) for n in nums}
        nums = sorted(nums)
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                if digests[a] == digests[b]:
                    pairs.append((a, b, len(folders[a])))
    return pairs


def cmu_rows(ns_takes, amass_rows, folders):
    """Person-map rows of cmu_mocap and the duplicate-folder table."""
    amass_single = {}                      # folder number -> AMASS person_id of amass:CMU:CMU_<nn>
    amass_multi = collections.defaultdict(set)    # AMASS person -> folder numbers of two-actor folders
    multi_names = collections.defaultdict(set)
    for r in amass_rows:
        tail = r["subject_ns"][len("amass:CMU:CMU_"):]
        if tail.isdigit():
            amass_single[int(tail)] = r["person_id"]
            continue
        m = re.match(r"(\d+)_(\d+)_", tail)
        if m:
            nums = {int(m.group(1)), int(m.group(2))}
            amass_multi[r["person_id"]] |= nums
            for n in nums:
                multi_names[n].add(f"CMU_{m.group(1)}_{m.group(2)}")
    g = Groups()
    for ns in ns_takes:
        g.find(int(ns.split(":S")[1]))
    dups = duplicate_folders(folders)
    for a, b, _ in dups:
        g.join(a, b)
    for nums in amass_multi.values():
        nums = sorted(nums)
        for n in nums[1:]:
            g.join(nums[0], n)
    members = collections.defaultdict(list)
    for ns in ns_takes:
        n = int(ns.split(":S")[1])
        members[g.find(n)].append(n)
    dup_of = {b: a for a, b, _ in dups}
    rows = []
    for ns in sorted(ns_takes):
        n = int(ns.split(":S")[1])
        group = sorted(members[g.find(n)])
        actor_takes = sorted({t.rsplit("_actor", 1)[0] for t in ns_takes[ns] if "_actor" in t})
        evidence = (f"folder subjects/{n:02d} of the CMU C3D zips: {len(ns_takes[ns])} canonical records "
                    f"(MoveAtlas datasets/_canonical_manifest.csv)")
        amb = [CMU_SESSION]
        if len(group) > 1:
            pid = "cmu_mocap:group:" + "_".join(f"{x:03d}" for x in group)
            rule = "cmu_folder_group"
            same = [f"{a:03d}={b:03d}" for a, b, _ in dups if a in group]
            amass_names = sorted({x for k in group for x in multi_names.get(k, ())})
            amass_ids = sorted({amass_single[k] for k in group if k in amass_single})
            evidence += (f"; folders {', '.join(f'{x:03d}' for x in group)} joined: byte-identical C3D file sets "
                         f"({', '.join(same) or 'none'})"
                         + (f", AMASS two-actor folders {', '.join(amass_names)} naming one person under several "
                            "numbers" if amass_names else "")
                         + (f", AMASS rows {', '.join(amass_ids)} for the same folders" if amass_ids else ""))
            two_actor = amass_names or any("_actor" in t for k in group for t in ns_takes[f"cmu_mocap:S{k:03d}"])
            amb.append("two-actor takes: the group holds more than one performer; one group id so that no split "
                       "separates them" if two_actor else
                       "the same takes filed under several subject numbers (AMASS fits each number as its own "
                       "person); one group id so that no split separates the copies")
            if n in dup_of:
                amb.append(f"duplicate of folder {dup_of[n]:03d}: the same C3D files, the same takes")
        elif n in amass_single:
            pid, rule = amass_single[n], "cmu_subject_number_to_amass_cmu"
            evidence += (f"; AMASS row amass:CMU:CMU_{n:02d} (person {pid}) holds AMASS's fits of the same takes "
                         "(MoveAtlas ingest_open_mocap.py same_person_as)")
        else:
            pid, rule = ns, "identity"
            evidence += "; no AMASS CMU row for this folder number"
        if actor_takes and len(group) == 1:
            amb.append(f"{len(actor_takes)} two-actor take(s) under this folder ({', '.join(actor_takes)}): the "
                       "actor of such a record is not identified, the person id applies to the folder")
        rows.append({"source": "cmu_mocap", "subject_ns": ns, "person_id": pid, "rule": rule, "evidence": evidence,
                     "ambiguity": "; ".join(amb)})
    dup_rows = [{"folder": f"{a:03d}", "duplicate_folder": f"{b:03d}", "c3d_files": k,
                 "subject_ns_kept": f"cmu_mocap:S{a:03d}", "subject_ns_duplicate": f"cmu_mocap:S{b:03d}"}
                for a, b, k in dups]
    return rows, dup_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    a = ap.parse_args()
    ma = a.moveatlas.replace("\\", "/").rstrip("/")
    sys.stdout.reconfigure(encoding="utf-8")
    out_o = os.path.join(TREE, "reports", "open_mocap")
    scope_path = os.path.join(TREE, "reports", "data_scope", "data_scope.csv")
    pm_path = os.path.join(TREE, "reports", "person_ids", "person_map.csv")

    # ---------------------------------------------------------------- data scope
    tmp = tempfile.mkdtemp(prefix="mdfm_scope_")
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, os.path.join(HERE, "data_scope.py"), "--moveatlas", ma, "--out", tmp],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        fresh = {r["id"]: r for r in rd(os.path.join(tmp, "data_scope.csv"))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(scope_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        old = list(reader)
    missing = [i for i in LOT if i not in fresh]
    if missing:
        sys.exit(f"data_scope.py gave no row for {missing}")
    if list(next(iter(fresh.values())).keys()) != cols:
        sys.exit("data_scope.py columns differ from reports/data_scope/data_scope.csv")
    moved = sorted(r["id"] for r in old if r["id"] not in LOT and r["id"] in fresh and any(
        fresh[r["id"]][c] != r[c] for c in ("scope", "design_tier", "licence_class", "harmonized")))
    kept = [r for r in old if r["id"] not in LOT]
    write_csv(scope_path, cols, kept + [fresh[i] for i in LOT])

    # ---------------------------------------------------------------- person map
    man = rd(f"{ma}/datasets/_canonical_manifest.csv")
    ns_takes = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in man:
        if r["source"] in WITH_RECORDS:
            ns_takes[r["source"]][r["subject_ns"]].append(r["take"])
    absent = [s for s in WITH_RECORDS if s not in ns_takes]
    if absent:
        sys.exit(f"no manifest rows for {absent}")
    pm = rd(pm_path)
    amass_cmu = [r for r in pm if r["source"] == "amass" and r["subject_ns"].startswith("amass:CMU:")]
    rows, dup_rows = cmu_rows(ns_takes["cmu_mocap"], amass_cmu, cmu_folders(ma))
    for s in WITH_RECORDS:
        if s == "cmu_mocap":
            continue
        for ns in sorted(ns_takes[s]):
            rows.append({"source": s, "subject_ns": ns, "person_id": ns, "rule": "identity",
                         "evidence": f"subject folder or id of the deposit: {len(ns_takes[s][ns])} canonical records "
                                     "(MoveAtlas datasets/_canonical_manifest.csv)",
                         "ambiguity": NOTES.get(s, "")})
    rows.sort(key=lambda r: (r["source"], r["subject_ns"]))
    write_csv(os.path.join(TREE, "reports", "person_ids", "map_open_mocap.csv"), PM_COLS, rows)
    write_csv(pm_path, PM_COLS, [r for r in pm if r["source"] not in WITH_RECORDS] + rows)
    os.makedirs(out_o, exist_ok=True)
    write_csv(os.path.join(out_o, "cmu_duplicate_folders.csv"),
              ["folder", "duplicate_folder", "c3d_files", "subject_ns_kept", "subject_ns_duplicate"], dup_rows)

    # ---------------------------------------------------------------- summary
    per = collections.OrderedDict()
    for s in WITH_RECORDS:
        rs = [r for r in rows if r["source"] == s]
        per[s] = (len(rs), len({r["person_id"] for r in rs}), collections.Counter(r["rule"] for r in rs))
    head = moveatlas_head(ma)
    md = ["# Open motion capture: data scope and person map rows (A23, step M1)", "",
          f"Built by scripts/open_mocap_scope_persons.py at MoveAtlas {head.split(' ')[0]}; rules in its docstring.", "",
          "## Data scope", "",
          "| id | scope | design tier | licence class | on-disk GB | harmonized |", "|---|---|---|---|---|---|"]
    md += [f"| {i} | {fresh[i]['scope']} | {fresh[i]['design_tier']} | {fresh[i]['licence_class']} | "
           f"{fresh[i]['data_gb']} | {fresh[i]['harmonized']} |" for i in LOT]
    md += ["", "Rows of reports/data_scope/data_scope.csv outside the lot whose scope, tier, licence class or harmonized "
           f"flag a full data_scope.py run would move (kept as they were here): {', '.join(moved) or 'none'}.", "",
           "## Person map", "", "| source | subject_ns | persons | rules |", "|---|---|---|---|"]
    md += [f"| {s} | {n} | {p} | {', '.join(f'{k} {v}' for k, v in sorted(c.items()))} |" for s, (n, p, c) in per.items()]
    md += ["", f"cmu_mocap folders whose C3D file sets are byte-identical (reports/open_mocap/cmu_duplicate_folders.csv): "
           f"{len(dup_rows)} pairs, " + ", ".join(f"{d['folder']}={d['duplicate_folder']} ({d['c3d_files']} files)"
                                                   for d in dup_rows) + ".", ""]
    text = "\n".join(md) + "\n"
    if "\u2014" in text:
        sys.exit("em-dash in scope_persons.md")
    with open(os.path.join(out_o, "scope_persons.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    prov = {"script": os.path.basename(__file__), "moveatlas_root": ma, "moveatlas_head": head,
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "person_map_sha256": sha256_file(pm_path), "data_scope_sha256": sha256_file(scope_path),
            "map_open_mocap_sha256": sha256_file(os.path.join(TREE, "reports", "person_ids", "map_open_mocap.csv"))}
    with open(os.path.join(out_o, "_provenance_scope_persons.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
