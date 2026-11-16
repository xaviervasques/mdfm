"""B4.0: the record index, every row of the tiered manifest resolved to its MoveAtlas container and record.

Inputs (read-only): cache/manifest/manifest_records.csv.gz (scripts/build_tiers_and_lockbox.py), the
MoveAtlas pooled canonical manifests datasets/_canonical_*manifest*.csv (record order),
docs/datasets_index.csv (registry path of each source) and the moveatlas-record/1 containers
(docs/RECORD_FORMAT.md), of which only meta.json is read, plus one array per source and family.

Resolution rules (MoveAtlas docs/MANIFEST_SCHEMA.md, docs/RECORD_FORMAT.md, moveatlas/datasets/generic.py):
1. record_path (EEG and EMG): the manifest key is the repository path of a one-record container; a
   .pkl path resolves to its .npz sibling; a record with no .npz is unresolved (pickles are never
   opened). The record's subject_ns must equal the row's.
2. identity (pose, IMU, force, silhouette): the list containers under <registry path>/processed/,
   canonical.npz, canonical_imu.npz, canonical_force.npz and canonical_silhouette.npz by name and
   canonical/<shard>.npz with the family read from the keys of each record; record i sits under the
   member prefix r{i:06d}/. A row takes the record of its source with the same family, subject_ns and
   task (else take, else sequence); among several, the first whose n_frames equals the row's.
3. manifest_order: a row rule 2 leaves unmatched takes record i of its source and family when it is
   the i-th row of the first pooled manifest file listing that source (builders write records in
   manifest order), rows keyed as scripts/build_tiers_and_lockbox.py keys them (keyed_manifests);
   accepted only when record and row counts agree and subject_ns and n_frames agree for every record
   (pads: its manifest task is <task>_<location>, its record task is <task>; goodwin_smm keys
   <task>_<location> in tiers v1.2; kaggle_fog and vidimu key the second row of a repeated id <id>#2;
   tests/test_canonical.py checks each pads and goodwin_smm record's location_ids against the suffix).
4. occurrence: a row rules 2 and 3 leave unmatched whose key is <id>#<n> (n >= 2; the occurrence suffix of
   scripts/build_tiers_and_lockbox.py: the n-th row of a manifest file repeating its source, subject_ns and id)
   takes the n-th record of that id, subject_ns, source and family in container order, when that record is
   unclaimed and its n_frames equals the row's (deepdance S01 lists nine takes twice).
CODY rows are never opened: they stay unresolved. A CODY row is one whose source, manifest_source,
subject_ns or person_id starts with cody, or whose subject_ns or person_id sits in a namespace a CODY
source uses (cody_1 files its adults under adult:); a list container is read only for a source with a
non-CODY row.

Checks on every resolved row: n_frames of the record (meta.json) against the manifest (status ok or
n_frames_mismatch); fps where both declare it: equal (relative 1e-6), rounded (equal at the
manifest's printed precision) or mismatch; one array per source and family, the main array of the
shortest resolved non-lockbox record, loaded to check that its first dimension is n_frames.

Incremental mode (--incremental, decided 2026-11-12 for the tiers rerun A23): the existing index is extended, never
rebuilt. Every row of the existing index is kept: its resolution (container, record, prefix, rule, n_candidates,
rec_n_frames, rec_fps, reason) as written; its status and fps_check are recomputed from those stored record values
against the current manifest (no container opened; the count of changed cells is reported), and shared over all
rows. Only the sources with no row in the existing index are new; a new row of a source already indexed, or an
indexed row the manifest no longer holds, stops the build (a full rebuild is then needed). A new source holding an
EVAL, EXCLUDED_V1 or lockbox row (canonical.never_opened_source) is never opened, neither meta.json nor any
array: its rows enter the index from the manifest's own columns with status canonical.NOT_OPENED, no container and
the reason; by_source.csv says 'not opened' for its container columns. The containers of the other new sources are
read by the rules above. The per-source columns read from containers (records without a row, spot check) of the
kept sources are carried from the previous by_source.csv. A full build (no flag) keeps the rules above for every
source, EVAL ones included, as before.

Writes cache/manifest/record_index.csv.gz (gitignored; columns INDEX_COLS; gzip header without a time stamp, so
that the file's SHA-256 depends on its content only), reports/record_index/summary.md, by_source.csv and
_provenance.json (with the index file's SHA-256 and that of its decompressed text).
Run: <conda>/envs/mdfm/python.exe scripts/build_record_index.py --moveatlas <moveatlas>
[--incremental]
"""
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCREMENTAL = "--incremental" in sys.argv
sys.argv = [a for a in sys.argv if a != "--incremental"]
if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
    sys.argv += ["--out", os.path.join(TREE, "reports", "record_index")]
from _common import parse_args, write_provenance  # noqa: E402
import build_tiers_and_lockbox as T  # noqa: E402  (the manifest keys, one definition for both builders)

ARGS = parse_args(__doc__.splitlines()[0])
MA = ARGS.moveatlas
os.environ["MOVEATLAS_ROOT"] = MA
sys.path.insert(0, TREE)
from mdfm import paths  # noqa: E402
from mdfm.data import canonical as C  # noqa: E402

assert paths.MOVEATLAS == MA, (paths.MOVEATLAS, MA)
OUT = paths.assert_not_moveatlas(ARGS.out)
INDEX_OUT = paths.assert_not_moveatlas(os.path.join(paths.cache_dir("manifest"), "record_index.csv.gz"))
INDEX_COLS = C.JOIN + ["container", "record", "prefix", "rule", "n_candidates", "rec_n_frames", "rec_fps",
                       "status", "fps_check", "shared", "reason"]
t0 = time.time()


def rd(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def fps_check(man, rec):
    man = (man or "").strip()
    if rec is None and not man:
        return "undeclared"
    if rec is None:
        return "record_undeclared"
    if not man:
        return "manifest_undeclared"
    m, x = float(man), float(rec)
    if abs(m - x) <= 1e-6 * max(1.0, abs(x)):
        return "equal"
    dec = len(man.split(".", 1)[1]) if "." in man else 0
    if abs(round(x, dec) - m) <= 1e-9 * max(1.0, abs(m)):
        return "rounded"
    return "mismatch"


# ---------------------------------------------------------------- inputs
man_path = C.manifest_path()
man_sha = hashlib.sha256(open(man_path, "rb").read()).hexdigest()
with gzip.open(man_path, "rt", encoding="utf-8", newline="") as f:
    ROWS = list(csv.DictReader(f))
CODY_NS = {C.namespace(r[c]) for r in ROWS if C.is_cody(r["source"], r["manifest_source"])
           for c in ("subject_ns", "person_id")} - {""}


def cody_row(r):
    """The CODY test of mdfm.data.canonical: names starting with cody, or a CODY source's namespace."""
    return (C.is_cody(r["source"], r["manifest_source"], r["subject_ns"], r["person_id"])
            or bool({C.namespace(r["subject_ns"]), C.namespace(r["person_id"])} & CODY_NS))


def join_key(r):
    return tuple(r[c] for c in C.JOIN)


def sha256_path(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- incremental mode: what is kept, what is new
KEPT, BASE_BY, BASE = {}, {}, {}
SHUT = set()                                 # new sources never opened (canonical.never_opened_source)
if INCREMENTAL:
    if not os.path.exists(INDEX_OUT):
        sys.exit(f"--incremental: no record index at {INDEX_OUT} to extend (run a full build)")
    BASE = {"record_index_sha256": sha256_path(INDEX_OUT)}
    with gzip.open(INDEX_OUT, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != INDEX_COLS:
            sys.exit(f"--incremental: the existing index has columns {reader.fieldnames}, this builder {INDEX_COLS}")
        for r in reader:
            KEPT[join_key(r)] = r
    with open(os.path.join(OUT, "_provenance.json"), encoding="utf-8") as f:
        BASE["manifest_records_sha256"] = json.load(f).get("manifest_records_sha256")
    BASE_BY = {(r["source"], r["modality"]): r for r in rd(os.path.join(OUT, "by_source.csv"))}
    man_keys = {join_key(r) for r in ROWS}
    gone = [k for k in KEPT if k not in man_keys]
    if gone:
        sys.exit(f"--incremental: {len(gone)} indexed rows are no longer in the manifest (first {gone[:3]}): "
                 "run a full build")
    kept_sources = {k[0] for k in KEPT}
    grown = sorted({r["source"] for r in ROWS if r["source"] in kept_sources and join_key(r) not in KEPT})
    if grown:
        sys.exit(f"--incremental: new rows in sources already indexed {grown}: run a full build")
    by_new = collections.defaultdict(list)
    for r in ROWS:
        if r["source"] not in kept_sources:
            by_new[r["source"]].append(r)
    SHUT = {s for s, rows in by_new.items() if C.never_opened_source(rows)}
    WORK = [r for r in ROWS if r["source"] in by_new]
else:
    WORK = ROWS
FRESH = {r["source"] for r in WORK}          # sources whose rows this run resolves
print(f"{'incremental' if INCREMENTAL else 'full'} build: {len(KEPT)} rows kept, {len(WORK)} rows to resolve "
      f"({len(FRESH)} sources, never opened: {sorted(SHUT)})")

REG = {r["id"]: r["path"].replace("\\", "/").strip("/") for r in rd(f"{MA}/docs/datasets_index.csv")}

# pooled manifests in the order the tier script reads them, each row with the key that script gives it
# (T.keyed_manifests: location and occurrence suffixes); first file listing (manifest_source, family); the rows of
# a never-opened source are left out
FIRST = {}
for base, fam, rows, keys in T.keyed_manifests(MA):
    per = collections.defaultdict(list)
    for r, key in zip(rows, keys):
        if r["source"] not in SHUT:
            per[(r["source"], fam)].append(dict(r, _key=key))
    for k, v in per.items():
        FIRST.setdefault(k, (base, v))


# ---------------------------------------------------------------- list containers
def list_containers(source):
    """[(relpath, family hint)] of the list containers of a source and a note on what was not used."""
    p = REG.get(source)
    if not p:
        return [], f"no registry path for {source} in docs/datasets_index.csv"
    base = f"{p}/processed"
    full = f"{MA}/{base}"
    if not os.path.isdir(full):
        return [], f"no {base}"
    names = {n for n in os.listdir(full) if not n.startswith("._")}
    out, notes = [], []
    for fname, hint in C.NAMED.items():
        if fname in names:
            out.append((f"{base}/{fname}", hint))
        elif fname[:-4] + ".pkl" in names:
            notes.append(f"{base}/{fname[:-4]}.pkl is a legacy pickle (never opened)")
    if C.SHARDS in names and os.path.isdir(f"{full}/{C.SHARDS}"):
        files = sorted(n for n in os.listdir(f"{full}/{C.SHARDS}") if not n.startswith("._"))
        npz = [n for n in files if n.endswith(".npz")]
        if not npz and any(n.endswith(".pkl") for n in files):
            notes.append(f"{base}/{C.SHARDS}/ holds legacy pickles only (never opened)")
        out += [(f"{base}/{C.SHARDS}/{n}", None) for n in npz]
    if not out:
        notes.append(f"no list container under {base}")
    return out, "; ".join(notes)


RECS = collections.defaultdict(list)        # (source, family) -> record entries in container order
IDENT = collections.defaultdict(list)       # (source, family, subject_ns, key) -> record entries
NOTES = {}
list_sources = sorted({r["source"] for r in WORK if r["modality"] not in ("eeg", "emg")
                       and not cody_row(r)} - SHUT)
n_containers = 0
for source in list_sources:
    conts, NOTES[source] = list_containers(source)
    for relpath, hint in conts:
        try:
            header = C.read_header(relpath)
        except (zipfile.BadZipFile, ValueError, OSError) as e:
            NOTES[source] = (NOTES[source] + "; " if NOTES[source] else "") + f"{relpath} unreadable ({type(e).__name__})"
            continue
        n_containers += 1
        kind = header.get("kind")
        for i, meta in enumerate(C.header_records(header)):
            fam = C.modality_of(meta, hint)
            e = {"container": relpath, "record": i if kind == "list" else -1, "subject_ns": str(meta.get("subject_ns")),
                 "key": C.record_key(meta), "n_frames": as_int(meta.get("n_frames")), "fps": meta.get("fps"), "claimed": 0}
            RECS[(source, fam)].append(e)
            IDENT[(source, fam, e["subject_ns"], e["key"])].append(e)
print(f"list containers read: {n_containers} ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- resolution
def assign(res, r, container, record, rule, n_cand, n_frames, fps):
    res.update(container=container, record=record, prefix=C.member_prefix(record), rule=rule, n_candidates=n_cand,
               rec_n_frames="" if n_frames is None else n_frames, rec_fps="" if fps is None else repr(float(fps)),
               fps_check=fps_check(r["fps"], fps), reason="")
    mn = as_int(r["n_frames"])
    res["status"] = "ok" if n_frames is not None and mn == n_frames else "n_frames_mismatch"


def unresolved(res, reason):
    res.update(status="unresolved", reason=reason)


REFERENCED = set()


def resolve_record_path(r, res):
    rel = r["key"].replace("\\", "/")
    if not rel.startswith("datasets/"):
        return unresolved(res, "record_path is not a repository path")
    pkl = rel.endswith(".pkl")
    npz = rel[:-4] + ".npz" if pkl else rel
    if not os.path.isfile(f"{MA}/{npz}"):
        return unresolved(res, "legacy pickle only (never opened)" if pkl and os.path.isfile(f"{MA}/{rel}")
                          else "container missing")
    try:
        header = C.read_header(npz)
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        return unresolved(res, f"container unreadable ({type(e).__name__})")
    metas = C.header_records(header)
    if header.get("kind") != "record" or len(metas) != 1:
        return unresolved(res, "record_path is not a one-record container")
    meta = metas[0]
    if str(meta.get("subject_ns")) != r["subject_ns"]:
        return unresolved(res, "record_path holds another subject_ns")
    REFERENCED.add(npz)
    assign(res, r, npz, -1, "record_path (.pkl to .npz)" if pkl else "record_path", 1,
           as_int(meta.get("n_frames")), meta.get("fps"))


def choose(cands, n):
    for test in (lambda e: not e["claimed"] and e["n_frames"] == n, lambda e: e["n_frames"] == n,
                 lambda e: not e["claimed"], lambda e: True):
        for e in cands:
            if test(e):
                return e


OCCURRENCE = re.compile(r"^(.*)#(\d+)$")
OUT_ROWS = []
DEFERRED = collections.defaultdict(list)
for r in WORK:
    res = {k: r[k] for k in C.JOIN}
    res.update(container="", record="", prefix="", rule="", n_candidates=0, rec_n_frames="", rec_fps="",
               status="", fps_check="", shared=0, reason="")
    OUT_ROWS.append((r, res))
    fam = r["modality"]
    if cody_row(r):
        unresolved(res, "CODY namespace: evaluation-only, never opened by this builder")
    elif r["source"] in SHUT:
        res.update(status=C.NOT_OPENED, reason="source with an EVAL, EXCLUDED_V1 or lockbox row: never opened by "
                                               "the incremental build (no meta.json, no array)")
    elif fam in ("eeg", "emg"):
        resolve_record_path(r, res)
    else:
        cands = IDENT.get((r["source"], fam, r["subject_ns"], r["key"]), [])
        if cands:
            e = choose(cands, as_int(r["n_frames"]))
            e["claimed"] += 1
            assign(res, r, e["container"], e["record"], "identity", len(cands), e["n_frames"], e["fps"])
        else:
            DEFERRED[(r["source"], r["manifest_source"], fam)].append((r, res))

for (source, msrc, fam), items in sorted(DEFERRED.items()):
    recs = RECS.get((source, fam), [])
    base, pooled = FIRST.get((msrc, fam), ("", []))
    aligned = bool(recs) and len(pooled) == len(recs) and all(
        p["subject_ns"] == e["subject_ns"] and as_int(p["n_frames"]) == e["n_frames"] for p, e in zip(pooled, recs))
    if aligned:
        pos = {}
        for i, p in enumerate(pooled):
            pos.setdefault((p["subject_ns"], p["_key"]), i)
        for r, res in items:
            i = pos.get((r["subject_ns"], r["key"]))
            if i is None:
                unresolved(res, f"key absent from {base}")
                continue
            e = recs[i]
            e["claimed"] += 1
            assign(res, r, e["container"], e["record"], "manifest_order", 1, e["n_frames"], e["fps"])
        continue
    if not recs:
        reason = f"no {fam} record of this source ({NOTES.get(source) or 'no container'})"
    else:
        reason = (f"no {fam} record with this subject_ns and key; manifest order not aligned "
                  f"({len(pooled)} rows in {base or 'no pooled file'}, {len(recs)} records)")
    for r, res in items:
        m = OCCURRENCE.match(r["key"])
        if m and int(m.group(2)) >= 2:
            same = IDENT.get((r["source"], fam, r["subject_ns"], m.group(1)), [])
            n = int(m.group(2))
            e = same[n - 1] if len(same) >= n else None
            if e is not None and not e["claimed"] and e["n_frames"] == as_int(r["n_frames"]):
                e["claimed"] += 1
                assign(res, r, e["container"], e["record"], "occurrence", len(same), e["n_frames"], e["fps"])
                continue
        unresolved(res, reason)

# kept rows (incremental mode): resolution as written, status and fps_check recomputed from the stored record values
ROWS_ADDED = len(OUT_ROWS)
RECHECKED = collections.Counter()
if INCREMENTAL:
    for r in ROWS:
        k = join_key(r)
        if k not in KEPT:
            continue
        res = dict(KEPT[k])
        res["shared"] = as_int(res["shared"]) or 0
        if res["container"]:
            n = as_int(res["rec_n_frames"])
            status = "ok" if n is not None and as_int(r["n_frames"]) == n else "n_frames_mismatch"
            fc = fps_check(r["fps"], res["rec_fps"] if res["rec_fps"] != "" else None)
            for col, new in (("status", status), ("fps_check", fc)):
                if res[col] != new:
                    RECHECKED[f"{r['source']} {col} {res[col] or '(empty)'} -> {new}"] += 1
                    res[col] = new
        OUT_ROWS.append((r, res))
    print(f"kept rows: {len(OUT_ROWS) - ROWS_ADDED}; check cells recomputed: {dict(RECHECKED)}")

share = collections.Counter((res["container"], res["record"]) for _, res in OUT_ROWS if res["container"])
for _, res in OUT_ROWS:
    if res["container"]:
        res["shared"] = share[(res["container"], res["record"])]
print(f"rows resolved ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- orphans (records no row points at)
ORPHANS = collections.Counter()
ORPHAN_EXAMPLES = collections.defaultdict(list)
for (source, fam), recs in RECS.items():
    for e in recs:
        if not e["claimed"]:
            ORPHANS[(source, fam)] += 1
            ORPHAN_EXAMPLES[(source, fam)].append(e)
per_record_sources = sorted({r["source"] for r in WORK if r["modality"] in ("eeg", "emg")
                             and not cody_row(r)} - SHUT)
for source in per_record_sources:
    p = REG.get(source)
    for fam in ("eeg", "emg"):
        d = f"{p}/processed/canonical_{fam}"
        if p and os.path.isdir(f"{MA}/{d}"):
            files = sorted(f"{d}/{n}" for n in os.listdir(f"{MA}/{d}") if n.endswith(".npz") and not n.startswith("._"))
            for x in files:
                if x not in REFERENCED:
                    ORPHANS[(source, fam)] += 1
                    ORPHAN_EXAMPLES[(source, fam)].append({"container": x, "record": -1, "subject_ns": "", "key": "",
                                                           "n_frames": None})

# ---------------------------------------------------------------- spot check: one array per source and family
SPOT = {}
groups = collections.defaultdict(list)
for r, res in OUT_ROWS:
    if (res["container"] and r["source"] in FRESH and r["source"] not in SHUT and r["lockbox"] != "True"
            and as_int(res["rec_n_frames"])):
        groups[(r["source"], r["modality"])].append((as_int(res["rec_n_frames"]), r, res))
for (source, fam), cand in sorted(groups.items()):
    n, r, res = min(cand, key=lambda x: (x[0], x[2]["container"], int(x[2]["record"])))
    name = C.MAIN_ARRAY[fam]
    with zipfile.ZipFile(C.container_path(res["container"])) as zf:
        header = C.read_header(res["container"], zf)
        meta = C.header_records(header)[max(int(res["record"]), 0)]
        tok = meta.get(name)
        member = f"{res['prefix']}{tok[C.TOKEN] if isinstance(tok, dict) else name}.npy"
        arr = C.read_npy(zf, member)
    ok = arr.shape[0] == as_int(r["n_frames"])
    SPOT[(source, fam)] = f"{name} {tuple(arr.shape)} {arr.dtype} {'ok' if ok else 'FIRST DIM != n_frames'}"
print(f"spot checks: {len(SPOT)} ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- write
OUT_ROWS.sort(key=lambda x: tuple(x[1][k] for k in C.JOIN))
with gzip.GzipFile(INDEX_OUT, "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=INDEX_COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(res for _, res in OUT_ROWS)
INDEX_SHA = sha256_path(INDEX_OUT)
with gzip.open(INDEX_OUT, "rb") as f:
    INDEX_TEXT_SHA = hashlib.sha256(f.read()).hexdigest()


def carried(k, col, fresh):
    """A by_source column read from containers: 'not opened' for a never-opened source, the previous
    by_source.csv value for a kept source (incremental mode), this run's value otherwise."""
    if k[0] in SHUT:
        return "not opened"
    if k[0] not in FRESH:
        return BASE_BY.get(k, {}).get(col, "")
    return fresh

by = collections.OrderedDict()
reasons = collections.defaultdict(collections.Counter)
for r, res in OUT_ROWS:
    k = (r["source"], r["modality"])
    b = by.setdefault(k, collections.Counter())
    b["rows"] += 1
    b["resolved"] += bool(res["container"])
    b["n_frames_mismatch"] += res["status"] == "n_frames_mismatch"
    b["fps_mismatch"] += res["fps_check"] == "mismatch"
    b["fps_rounded"] += res["fps_check"] == "rounded"
    b["unresolved"] += not res["container"]
    b["shared"] += res["shared"] > 1
    b["manifest_order"] += res["rule"] == "manifest_order"
    b["lockbox"] += r["lockbox"] == "True"
    if not res["container"]:
        reasons[k][res["reason"]] += 1
keys = sorted(by)
cols = ["source", "modality", "rows", "resolved", "n_frames_mismatch", "fps_mismatch", "fps_rounded", "unresolved",
        "unresolved_reasons", "container_records_without_row", "rows_sharing_a_record", "rows_by_manifest_order",
        "spot_check"]
with open(os.path.join(OUT, "by_source.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(cols)
    for k in keys:
        b = by[k]
        w.writerow([k[0], k[1], b["rows"], b["resolved"], b["n_frames_mismatch"], b["fps_mismatch"], b["fps_rounded"],
                    b["unresolved"], "; ".join(f"{v} {s}" for s, v in reasons[k].most_common()),
                    carried(k, "container_records_without_row", ORPHANS.get(k, 0)), b["shared"], b["manifest_order"],
                    carried(k, "spot_check", SPOT.get(k, ""))])

tot = collections.Counter()
for r, res in OUT_ROWS:
    cody = cody_row(r)
    tot["rows"] += 1
    tot["cody"] += cody
    if not cody:
        tot["noncody"] += 1
        tot["resolved"] += bool(res["container"])
        tot["unresolved"] += not res["container"]
        tot["not_opened"] += res["status"] == C.NOT_OPENED
        tot["mismatch"] += res["status"] == "n_frames_mismatch"
    tot["fps_" + (res["fps_check"] or "none")] += 1
fam_tot = collections.OrderedDict((m, collections.Counter()) for m in C.MODALITIES)
for r, res in OUT_ROWS:
    if cody_row(r):
        continue
    ft = fam_tot[r["modality"]]
    ft["rows"] += 1
    ft["resolved"] += bool(res["container"])
    ft["mismatch"] += res["status"] == "n_frames_mismatch"
pct = 100.0 * tot["resolved"] / max(tot["noncody"], 1)
head = write_provenance(ARGS, {"manifest_records_sha256": man_sha, "record_index": "cache/manifest/record_index.csv.gz",
                               "rows": tot["rows"], "noncody_rows": tot["noncody"], "resolved_noncody": tot["resolved"],
                               "list_containers_read": n_containers, "seconds": round(time.time() - t0),
                               "record_index_sha256": INDEX_SHA, "record_index_text_sha256": INDEX_TEXT_SHA,
                               "mode": "incremental" if INCREMENTAL else "full",
                               "rows_resolved_this_run": ROWS_ADDED, "rows_kept": len(OUT_ROWS) - ROWS_ADDED,
                               "sources_this_run": sorted(FRESH - SHUT), "sources_never_opened": sorted(SHUT),
                               "rows_never_opened": tot["not_opened"],
                               "base": BASE, "kept_check_cells_recomputed": dict(sorted(RECHECKED.items()))}
                         )["moveatlas_head"]
if INCREMENTAL:
    built = (f"Incremental build (--incremental) on the index of SHA-256 {BASE['record_index_sha256'][:16]} (manifest "
             f"{(BASE.get('manifest_records_sha256') or '')[:16]}): {len(OUT_ROWS) - ROWS_ADDED} rows kept as written "
             f"(status and fps_check recomputed from their stored record values: "
             f"{sum(RECHECKED.values())} cells changed), {ROWS_ADDED} rows of {len(FRESH)} new sources; containers "
             f"opened only for {', '.join(sorted(FRESH - SHUT)) or 'none'} (meta.json, plus one array per source and "
             f"family); never opened: {', '.join(sorted(SHUT)) or 'none'} ({tot['not_opened']} rows, status "
             f"{C.NOT_OPENED}, no container). Container columns of the kept sources (records without a row, spot "
             "check) are those of the previous build.")
else:
    built = "Only meta.json was read, plus one array per source and family (spot check)."
md = ["# Record index (B4.0)", "",
      "Built by scripts/build_record_index.py (rules in its docstring) from cache/manifest/manifest_records.csv.gz "
      f"(sha256 {man_sha[:16]}) against MoveAtlas {head}. Index: cache/manifest/record_index.csv.gz (gitignored); "
      f"per-source table: by_source.csv. Index file SHA-256 {INDEX_SHA}, decompressed text SHA-256 {INDEX_TEXT_SHA}. "
      + built, "",
      f"Manifest rows: {tot['rows']}; CODY rows (never opened): {tot['cody']}; non-CODY rows: {tot['noncody']}; "
      f"resolved: {tot['resolved']} ({pct:.3f} percent); unresolved: {tot['unresolved']}; resolved with an n_frames "
      f"mismatch: {tot['mismatch']}.",
      f"fps against the manifest: equal {tot['fps_equal']}, equal at the manifest's printed precision "
      f"{tot['fps_rounded']}, mismatch {tot['fps_mismatch']}, declared by neither {tot['fps_undeclared']}, "
      f"by the record only {tot['fps_manifest_undeclared']}, by the manifest only {tot['fps_record_undeclared']}.", "",
      "| family | non-CODY rows | resolved | n_frames mismatches |", "|---|---|---|---|"]
md += [f"| {m} | {v['rows']} | {v['resolved']} | {v['mismatch']} |" for m, v in fam_tot.items()]
md += ["", "## Per source and family", "",
       "Columns: rows; resolved; n_frames mismatches; fps mismatches; unresolved with the reason; records in the "
       "source's containers that no row points at; rows that share their record with another row; rows resolved "
       "by manifest order (rule 3); spot check (main array of the shortest non-lockbox record).", "",
       "| source | family | rows | resolved | n_frames mismatch | fps mismatch | unresolved (reason) | records without row | rows sharing a record | by order | spot check |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in keys:
    b = by[k]
    rs = "; ".join(f"{v}: {s}" for s, v in reasons[k].most_common())
    un = f"{b['unresolved']} ({rs})" if b["unresolved"] else "0"
    md.append(f"| {k[0]} | {k[1]} | {b['rows']} | {b['resolved']} | {b['n_frames_mismatch']} | {b['fps_mismatch']} | "
              f"{un} | {carried(k, 'container_records_without_row', ORPHANS.get(k, 0))} | {b['shared']} | "
              f"{b['manifest_order']} | {carried(k, 'spot_check', SPOT.get(k, ''))} |")
mism = [(r, res) for r, res in OUT_ROWS if res["status"] == "n_frames_mismatch"]
if mism:
    md += ["", "## n_frames mismatches (first 20)", "", "| source | family | subject_ns | key | manifest | record |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['source']} | {r['modality']} | {r['subject_ns']} | {r['key']} | {r['n_frames']} | {res['rec_n_frames']} |"
           for r, res in mism[:20]]
if ORPHAN_EXAMPLES:
    md += ["", "## Container records no row points at (first 5 per source and family)", ""]
    if INCREMENTAL:
        md += ["Sources opened by this run only; the kept sources' examples are in the previous build's summary.", ""]
    md += ["| source | family | container | record | subject_ns | key | n_frames |", "|---|---|---|---|---|---|---|"]
    for k in sorted(ORPHAN_EXAMPLES):
        md += [f"| {k[0]} | {k[1]} | {e['container']} | {e['record']} | {e['subject_ns']} | {e['key']} | {e['n_frames']} |"
               for e in ORPHAN_EXAMPLES[k][:5]]
shared_rows = collections.defaultdict(list)
for r, res in OUT_ROWS:
    if res["shared"] > 1:
        shared_rows[(r["source"], r["modality"])].append((r, res))
if shared_rows:
    md += ["", "## Rows sharing their record with another row (first 4 per source and family)", "",
           "| source | family | subject_ns | key | container | record |", "|---|---|---|---|---|---|"]
    for k in sorted(shared_rows):
        md += [f"| {k[0]} | {k[1]} | {r['subject_ns']} | {r['key']} | {res['container']} | {res['record']} |"
               for r, res in shared_rows[k][:4]]
fpm = [(r, res) for r, res in OUT_ROWS if res["fps_check"] == "mismatch"]
if fpm:
    md += ["", "## fps mismatches (first 20)", "", "| source | family | subject_ns | key | manifest | record |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['source']} | {r['modality']} | {r['subject_ns']} | {r['key']} | {r['fps']} | {res['rec_fps']} |"
           for r, res in fpm[:20]]
text = "\n".join(md) + "\n"
assert chr(0x2014) not in text
with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print(text)
