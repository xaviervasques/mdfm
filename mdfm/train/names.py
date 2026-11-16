"""Name tables of a run (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.1, task S2.1).

A model holds one row per name in five tables, and a strict load fails when a table grows (spec section 1,
item 5): the sources (SourceOffsets and the HSIC source codes, mdfm.model.mfm.SOURCES), the tasks (the task
embedding of the motion prior, read from a vocabulary file), and the layout, device and camera vocabularies of
the nuisance descriptor (mdfm.model.tokenize.NUIS_FIELDS), plus the mount families of the nuisance priors
(reports/mount_priors/priors.json, family_order). NameTables names all of them in one value, so that a run
config can carry them and a migration (task S2.2) can map one set onto another by name.

NameTables(sources, vocabulary_path, layouts, devices, cameras, mount_families):
- vocabulary_path names a task vocabulary file, relative to the tree (forward slashes) or absolute; tasks()
  reads it through mdfm.data.windows.task_vocabulary(path), and task_name / task_id call
  mdfm.data.windows.task_name(source, key, path) on the resolved path, the function the window covariates use
  (task S2.3 gave it the path argument), so the checks below measure the production path.
- sha256() hashes the canonical JSON of the names: every table, the task names in id order and the SHA-256 of
  the vocabulary file bytes (an edited rule changes the hash; the path does not enter it).
- nuis_fields() gives tokenize.NUIS_FIELDS with the layout, device and camera tuples of these tables.
- appended(old) gives, per table, the names added at the end of old's table, and refuses a table that drops
  or reorders an old name (the condition under which every old row keeps its index).
- to_dict() and from_dict() are the form a run config holds.

STAGE1 is the stage-1 set, written out here as literals so that it stays the stage-1 set whatever the live
module constants become: tests/test_names.py checks that it equals mfm.SOURCES, the version-1 vocabulary
docs/rules/task_vocabulary.yaml (SHA-256 VOCABULARY_V1_SHA256), tokenize.NUIS_FIELDS and the family order of
reports/mount_priors/priors.json. VOCABULARY_V2 is the version-2 file (DRAFT): the version-1 names and rules
unchanged, rest_lying and rest_sitting appended after 'unknown', and rules for the stage-2 sources.

This module imports no torch and reads no record. Run as a script it prints the tables and the version-2
coverage checks (python -m mdfm.train.names; --vocabulary to name another file).
"""
import hashlib
import json
import os
from dataclasses import dataclass, fields

from .. import paths
from ..data import windows as W

VOCABULARY_V1 = "docs/rules/task_vocabulary.yaml"
VOCABULARY_V1_SHA256 = "ba3f6e58d32dc8d58e9509678be86b06fcdbc20f4abe4cd8f361d6a3d3604265"
VOCABULARY_V2 = "docs/rules/task_vocabulary_2026-10-06.yaml"
TABLES = ("sources", "tasks", "layouts", "devices", "cameras", "mount_families")
_TUPLES = ("sources", "layouts", "devices", "cameras", "mount_families")
# fields of tokenize.NUIS_FIELDS that a NameTables carries
NUIS_TABLES = {"layout": "layouts", "device": "devices", "camera": "cameras"}


def resolve(path):
    """Absolute path of a vocabulary path: an absolute path as given, a relative one under the tree (joined
    part by part, so the version-1 file resolves to the string mdfm.data.windows.VOCABULARY)."""
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(paths.TREE, *[p for p in path.replace("\\", "/").split("/") if p])


def _names(table, values):
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{table}: a sequence of names, not a single string")
    out = tuple(values)
    if not out:
        raise ValueError(f"{table}: empty")
    for v in out:
        if not isinstance(v, str) or not v:
            raise ValueError(f"{table}: names are non-empty strings, got {v!r}")
    dup = sorted({v for v in out if out.count(v) > 1})
    if dup:
        raise ValueError(f"{table}: duplicate names {dup}")
    return out


@dataclass(frozen=True)
class NameTables:
    """The name tables of a run (module docstring)."""
    sources: tuple
    vocabulary_path: str
    layouts: tuple
    devices: tuple
    cameras: tuple
    mount_families: tuple

    def __post_init__(self):
        for name in _TUPLES:
            object.__setattr__(self, name, _names(name, getattr(self, name)))
        if not isinstance(self.vocabulary_path, str) or not self.vocabulary_path:
            raise ValueError("vocabulary_path: a path string")
        if self.sources[0] != "synthetic":
            raise ValueError(f"sources: 'synthetic' comes first (its SourceOffsets row is frozen), got "
                             f"{self.sources[0]!r}")
        if "generic" not in self.mount_families:
            raise ValueError("mount_families: 'generic' is the family of a layout without a measured prior")

    # ------------------------------------------------------------ the vocabulary
    def vocabulary(self):
        """mdfm.data.windows.task_vocabulary of the vocabulary file (tasks, ids, default, rules, sha256)."""
        vocab = W.task_vocabulary(resolve(self.vocabulary_path))
        _names("tasks", vocab["tasks"])
        if "unknown" not in vocab["ids"]:
            raise ValueError(f"{self.vocabulary_path}: no 'unknown' task (the condition-dropout row)")
        return vocab

    def tasks(self):
        """Task names in id order."""
        return tuple(self.vocabulary()["tasks"])

    def vocabulary_sha256(self):
        """SHA-256 of the vocabulary file bytes."""
        return self.vocabulary()["sha256"]

    def task_name(self, source, key):
        """Task name of a recording key under this vocabulary's rules of `source` (first match, the default
        otherwise; an amass key is matched on its take file name): mdfm.data.windows.task_name on this file,
        after the refusals of vocabulary()."""
        self.vocabulary()
        return W.task_name(source, key, resolve(self.vocabulary_path))

    def task_id(self, source, key):
        """Integer task id of a recording key under this vocabulary."""
        return self.vocabulary()["ids"][self.task_name(source, key)]

    # ------------------------------------------------------------ tables
    def table(self, name):
        """One table by its name in TABLES."""
        if name == "tasks":
            return self.tasks()
        if name not in _TUPLES:
            raise KeyError(f"unknown table {name!r}; known: {TABLES}")
        return getattr(self, name)

    def nuis_fields(self, base=None):
        """tokenize.NUIS_FIELDS (or `base`) with the layout, device and camera tuples of these tables, in the
        same field order."""
        if base is None:
            from ..model.tokenize import NUIS_FIELDS as base
        return {k: (getattr(self, NUIS_TABLES[k]) if k in NUIS_TABLES else v) for k, v in base.items()}

    def appended(self, old):
        """{table: names added at the end of old's table} for every table of TABLES; ValueError when a table
        does not start with old's names in old's order (a dropped, renamed or moved name)."""
        out = {}
        for name in TABLES:
            a, b = tuple(old.table(name)), tuple(self.table(name))
            if b[:len(a)] != a:
                lost = [n for n in a if n not in b]
                raise ValueError(f"{name}: the new table does not start with the old one "
                                 f"(old {list(a)}, new {list(b)}" + (f", dropped {lost})" if lost else ")"))
            out[name] = b[len(a):]
        return out

    # ------------------------------------------------------------ hashing and config form
    def canonical(self):
        """The hashed content: every table, the task names and the vocabulary file's SHA-256."""
        out = {name: list(getattr(self, name)) for name in _TUPLES}
        out["tasks"] = list(self.tasks())
        out["vocabulary_sha256"] = self.vocabulary_sha256()
        return out

    def sha256(self):
        """SHA-256 of the canonical JSON of canonical() (sorted keys, no spaces)."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self):
        """The config form: the five name tuples as lists and the vocabulary path as given."""
        return {f.name: (list(getattr(self, f.name)) if f.name in _TUPLES else getattr(self, f.name))
                for f in fields(self)}

    @classmethod
    def from_dict(cls, d):
        """Inverse of to_dict; an unknown or missing key raises."""
        want = {f.name for f in fields(cls)}
        extra, missing = sorted(set(d) - want), sorted(want - set(d))
        if extra or missing:
            raise ValueError(f"names: unknown keys {extra}, missing keys {missing}")
        return cls(**{k: d[k] for k in want})           # __post_init__ makes the name lists tuples


STAGE1 = NameTables(
    sources=("synthetic", "weargait_pd", "gait120", "stroke_gait_vancriekinge", "nonan_gaitprint",
             "nonan_gaitprint_young", "sixmwt_lifespan"),
    vocabulary_path=VOCABULARY_V1,
    layouts=("xsens13", "xsens13_insoles", "nonan3", "lower_back", "g1_six", "wrist_L", "wrist_acc",
             "random_sites", "camera1", "mocap3d", "mocap3d_plate"),
    devices=("generic", "xsens13"),
    cameras=("generic",),
    mount_families=("generic", "xsens13"),
)


# ---------------------------------------------------------------- coverage checks (script)
def pattern_keys(pattern):
    """Keys standing for a key pattern of the probe tables ('#' a digit run): one per digit 0 to 9."""
    return [pattern.replace("#", str(d)) for d in range(10)]


def stage1_rows(path=None):
    """Rows (source, task, key_pattern) of reports/stage1_sources/tasks.csv that carry a key pattern."""
    import csv
    path = path or os.path.join(paths.TREE, "reports", "stage1_sources", "tasks.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["key_pattern"]]


def stage2_rows(path=None):
    """Rows of reports/stage2_sources/items/recording_keys.csv (task S2.0)."""
    import csv
    path = path or os.path.join(paths.TREE, "reports", "stage2_sources", "items", "recording_keys.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def body_store_takes(root=None):
    """take_id of every row of every takes.csv of the body store (train, dev, calib shards), read with the
    csv module; no array is opened. [] when the store is absent."""
    import csv
    if root is None:
        root = os.environ.get("MDFM_BODY_STORE") or os.path.join(paths.CACHE, "body_store")
    out = []
    for split in ("train", "dev", "calib"):
        folder = os.path.join(root, split)
        if not os.path.isdir(folder):
            continue
        for a in sorted(os.listdir(folder)):
            p = os.path.join(folder, a, "takes.csv")
            if a.startswith(("._", ".")) or a.endswith(".partial") or not os.path.isfile(p):
                continue
            with open(p, encoding="utf-8", newline="") as f:
                out += [r["take_id"] for r in csv.DictReader(f)]
    return out


def coverage(new, old=STAGE1):
    """The checks of the S2.1 done-when, as counts: per key of the stage-1 task table and per body-store take,
    whether `new` gives the id `old` gives; per S2.0 key pattern, the task of its keys under old and new."""
    rep = {"old_sha256": old.sha256(), "new_sha256": new.sha256(), "old_vocabulary": old.vocabulary_sha256(),
           "new_vocabulary": new.vocabulary_sha256(), "appended": {k: list(v) for k, v in new.appended(old).items()}}
    s1 = stage1_rows()
    keys = [(r["source"], k) for r in s1 for k in pattern_keys(r["key_pattern"])]
    rep["stage1_rows"] = len(s1)
    rep["stage1_rows_with_v1_task"] = sum(any(old.task_name(r["source"], k) == r["task"]
                                              for k in pattern_keys(r["key_pattern"])) for r in s1)
    rep["stage1_keys"] = len(keys)
    rep["stage1_keys_same_id"] = sum(old.task_id(s, k) == new.task_id(s, k) for s, k in keys)
    takes = body_store_takes()
    rep["body_store_takes"] = len(takes)
    rep["body_store_takes_same_id"] = sum(old.task_id("amass", t) == new.task_id("amass", t) for t in takes)
    s2 = []
    for r in stage2_rows():
        ks = pattern_keys(r["key_pattern"]) + [e.strip() for e in r["examples"].split(";") if e.strip()]
        s2.append({"unit": r["unit"], "modality": r["modality"], "key_pattern": r["key_pattern"],
                   "records": int(r["records"]),
                   "old": sorted({old.task_name(r["source"], k) for k in ks}),
                   "new": sorted({new.task_name(r["source"], k) for k in ks})})
    rep["stage2_patterns"] = s2
    return rep


def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vocabulary", default=VOCABULARY_V2)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from dataclasses import replace
    new = replace(STAGE1, vocabulary_path=args.vocabulary)
    rep = coverage(new)
    print(f"STAGE1 name-table SHA-256 {rep['old_sha256']}; vocabulary {STAGE1.vocabulary_path} SHA-256 "
          f"{rep['old_vocabulary']}")
    print(f"new name-table SHA-256 {rep['new_sha256']}; vocabulary {args.vocabulary} SHA-256 {rep['new_vocabulary']}")
    print("tasks (id: name): " + ", ".join(f"{i}: {t}" for i, t in enumerate(new.tasks())))
    print(f"appended: {rep['appended']}")
    print(f"stage-1 task table rows with a key pattern: {rep['stage1_rows']}; rows whose v1 task the pattern "
          f"keys reproduce: {rep['stage1_rows_with_v1_task']}; pattern keys {rep['stage1_keys']}, same id under "
          f"both: {rep['stage1_keys_same_id']}")
    print(f"body-store takes: {rep['body_store_takes']}; same id under both: {rep['body_store_takes_same_id']}")
    print("S2.0 key patterns (unit, modality, pattern, records: old task -> new task):")
    for r in rep["stage2_patterns"]:
        print(f"  {r['unit']} | {r['modality']} | {r['key_pattern']} | {r['records']}: "
              f"{'/'.join(r['old'])} -> {'/'.join(r['new'])}")
    return rep


if __name__ == "__main__":
    main()
