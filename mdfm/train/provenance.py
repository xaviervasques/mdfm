"""Provenance of a stage-1 run (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.6 and section 5, task B4.17):
the training person lists, the hashes a checkpoint carries, the person guard and the state_dict audit.

Person lists (person_lists): {source: sorted person ids} of what trains: 'amass', the persons of the body
store (split train) whose bodies the factory renders, and each real source of the window store at its train
fold. Hashes of a list: person_hash (SHA-256 of the sorted distinct ids joined by newlines, the rule of
reports/window_store) and amass_index.person_list_sha256 (one id per line with a trailing newline, the rule
of reports/amass_split and reports/body_store). report_mismatches compares the lists with the hashes those
reports hold, read at run time.

Hashes (hashes), read at run time and never typed: the records and lockbox SHA-256 of
reports/manifest_tiers/hashes.json, the lockbox file SHA-256 checked against reports/lockbox/lockbox.sha256
and hashes.json (a mismatch raises ProvenanceError), the records file checked against records_file_sha256;
the MoveAtlas HEAD read from its .git files (read-only) beside the commit reports/moveatlas_state/state.json
checked; the index and person-list hashes of reports/body_store and the person and file hashes of
reports/window_store and, when the stores are given, their live values (the take index file and the train
list of the body store, the person lists and the source.json file hashes of the window store); the SHA-256 of
every file of docs/rules; the config SHA-256; a code SHA-256 over mdfm/**/*.py (code_sha256: relative path
and file SHA-256 per line, sorted; the tree has no commit).

Guard (deny_lists, problems, assert_clean). A person id is refused when it is a lockbox person
(reports/lockbox/lockbox_persons.csv, hash-checked, and the lockbox rows of the tiers records file), has an
EVAL or EXCLUDED_V1 row there, is CODY (a name starting with 'cody', an id in a namespace of
reports/manifest_tiers/cody_denylist.json, the id of a CODY row), is the canary cody_canary:mdfm-v1
(canary_subject_ns of the denylist), or sits in the held-out set given (dev, val and calibration persons).
assert_clean first checks that the record reader refuses the canary (canonical.check) and that the guard
flags it, then raises canonical.Refused naming every refused person. The records file is
cache/manifest/manifest_records.csv.gz of scripts/build_tiers_and_lockbox.py (gitignored); without it the
guard cannot run and raises FileNotFoundError. is_cody_id applies the two name rules of CODY (prefix and
namespace) without the records file; data/collate.py refuses with it.

Audit (audit_state_dict, audit; plan docs/MODEL_OPEN_PORT_PLAN.md step M2). The body of the audit comes from
mdfm.body.backend.model_body: the open body by default, SMPL+H only behind the reference flag. On the open body:
the shape and name screen of mdfm.model.mfm.audit_state_dict (a dimension of 6,890 or 52, an SMPL+H array name)
on the model, the EMA and the optimizer state (every tensor at any depth), and the open body's licence record
(mdfm.body.openbody.check_licence: the three licence SHA-256 of the installed Anny against
reports/openbody/licence_audit.json, its version, topology, rig and joint map); no SMPL+H file is read. On SMPL+H
(the v1 comparison, behind the flag) or when the caller passes the arrays: the screen above plus bytes equal to an
SMPL+H array as SMPLHBody stores it or to the derived J_template and J_dirs (per gender and stacked), and values
equal by allclose for tensors up to 100,000 values; the SMPL+H arrays are read once per process from the MoveAtlas
model files (read-only).

Tier hash history and the records drift policy (task T16; tier_hash_entry, tier_hash_history, add_tier_entry,
compare_tier_entries, records_drift). records_sha256 covers every column of the tiers records file, so a
correction of MoveAtlas that only fills the fps of a source no run trained on would make every checkpoint
unloadable after the tiers are rebuilt. The history (reports/provenance_history/tier_hash_history.json,
written by scripts/build_tier_hash.py before and after each rebuild) keeps, per records_sha256, an entry
computed from the records file alone: records_sha256 by the rule of scripts/build_tiers_and_lockbox.py
(SHA-256 of the tab-joined rows, one newline each, header left out), the file SHA-256, the row count, the
columns and, per source, the row count, tier_sha256 (SHA-256 of the lines 'identity + tier columns' of its
rows), columns_sha256 (per other column, SHA-256 of the lines 'identity + column') and the value counts of
manifest_source, modality and fps. Identity: source, modality, subject_ns, key (the build's sort key; the
rows must come in that order, unique). Tier columns: source, subject_ns, person_id, key, tier, lane, scope,
lockbox, group; other columns: manifest_source, modality, fps, n_frames; a column in neither group stops the
entry. Each entry carries entry_sha256 (canonical JSON of the rest) and is never rewritten.
records_drift tolerates a moved records_sha256 only when the history holds the checkpoint's records state
(same records file and lockbox SHA-256 as the checkpoint recorded), the live records file hashes to the
records_sha256 of hashes.json, and every changed row lies in a source outside the run (its training and
held-out person lists) and changes only fps (TOLERATED_COLUMNS): no row added or removed, no tier column
changed in any source, every source of the run present in the records. Anything else raises ProvenanceError
naming each problem (per-sample fps rule: an fps change in a source of the run stays fatal). The lockbox
hash is never relaxed (train/checkpoint.py).

Archived records states (decision of 2026-11-12, plan step M7; records_archive, archived_state, hashes_at). A tiers
rerun that adds sources or changes a run's source moves records_sha256 beyond the drift policy, and the policy is not
widened for it. Instead a run is read under a named archived records state: a copy of an earlier records file kept
in the gitignored cache (scripts/archive_records_state.py; record reports/provenance_history/records_archive.json),
checked against its SHA-256 and against the history entry of its state. records_drift(..., state=) compares the
run's records state with the named state by the unchanged policy: a run born on that state reads it exactly, a run
born on another state passes only what the policy tolerates between the two. The lockbox stays the live one, and
must equal the one the history recorded for the state.

Tier and licences (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.3, task S2.4). A run has a tier, RELEASE (the
default) or INTERNAL (TIERS; run_tier reads it from a RunConfig, a config dict or its JSON text, no key meaning
RELEASE, and from the run part of a G12 combined configuration); a provenance record carries it under 'tier', and a
record without the field (every checkpoint written before S2.4) reads as RELEASE (tier_of). DenyLists holds the
INTERNAL persons of the tiers records file (rows of tier INTERNAL; the 'source|person_id' list is checked against
tier_person_list_sha256['INTERNAL'] of reports/manifest_tiers/hashes.json) and the sources holding an INTERNAL row:
problems(..., tier=RELEASE) refuses them ('INTERNAL'), as it refuses the lockbox, so that a RELEASE save, init,
resume or migration meeting an INTERNAL person is refused; problems(..., tier=INTERNAL) admits them. tiered(record,
tier) gives the record a checkpoint stores: refused (ProvenanceError) for a RELEASE tier when the record says
INTERNAL (a provenance copied from an INTERNAL checkpoint, the migration case) or a lineage entry (extra.init_from,
extra.migration.migrated_from) records tier INTERNAL, whatever tier the record itself says; otherwise the record
with the run's tier and its licences, recomputed (a copied licence record is replaced). licences(sources) lists, for
each training source (the keys of the person lists), its row of MoveAtlas docs/license_matrix.csv (read-only; id,
license, redistribution, portal_action) with the file's SHA-256, and the inherited terms: each distinct licence text
with the sources that carry it. The texts are listed, not classified: the matrix holds free text, some of it
naming a licence it rejects. A training source without a row raises ProvenanceError. The matrix is read once per
process (licence_matrix: its rows and SHA-256 from the same bytes), as the deny lists are: every checkpoint of a run
cites the matrix its record() read, and a correction of the matrix during a run neither stops a save nor mixes two
versions in one run; the runs started after the correction read it.
"""
import collections
import csv
import datetime
import functools
import glob
import gzip
import hashlib
import importlib.util
import io
import json
import os
import types
from typing import NamedTuple

from .. import paths
from ..body import backend as BK
from ..data.canonical import Refused, check as reader_check, is_cody, namespace
from ..data.canonical import manifest_path as records_path
from ..data.window_store import person_hash
from ..model import mfm as M

FORMAT = "mdfm-provenance/1"
REPORTS = os.path.join(paths.TREE, "reports")
TIERS_JSON = os.path.join(REPORTS, "manifest_tiers", "hashes.json")
DENYLIST_JSON = os.path.join(REPORTS, "manifest_tiers", "cody_denylist.json")
LOCKBOX_CSV = os.path.join(REPORTS, "lockbox", "lockbox_persons.csv")
LOCKBOX_SHA = os.path.join(REPORTS, "lockbox", "lockbox.sha256")
BODY_JSON = os.path.join(REPORTS, "body_store", "hashes.json")
WINDOW_JSON = os.path.join(REPORTS, "window_store", "hashes.json")
AMASS_JSON = os.path.join(REPORTS, "amass_split", "hashes.json")
STATE_JSON = os.path.join(REPORTS, "moveatlas_state", "state.json")
RULES_DIR = os.path.join(paths.TREE, "docs", "rules")
CODE_DIR = os.path.join(paths.TREE, "mdfm")
AMASS = "amass"
OPEN_BODIES = "open_bodies"       # the person list of the open body store (plan section 3.3), in place of 'amass'


def is_open_store(body_store):
    """True for the open body store's training view (mdfm/data/open_body_store.OpenTrainStore)."""
    return type(body_store).__name__ == "OpenTrainStore"


def open_body_hashes(body_store):
    """The open body's provenance (plan section 3.3): the store's hashes and served person list (OpenTrainStore.hashes),
    the SHA-256 of every open run-time input (mdfm/train/bodies.OPEN_INPUTS) and the open body's licence record
    (mdfm/body/openbody.licence_record: Anny's version, data version, topology and rig, licence texts, joint map)."""
    from ..body import openbody as BO
    from . import bodies as BD
    out = {"store": body_store.hashes(),
           "inputs_sha256": {k: sha256_file(v) for k, v in sorted(BD.inputs(BD.OPEN).items())},
           "basis_sha256": sha256_file(BO.BASIS_VALUES)}
    try:
        out["licence"] = BO.licence_record()
    except Exception as e:  # noqa: BLE001  (an environment without anny: the record says so, audit() refuses)
        out["licence"] = {"error": f"{type(e).__name__}: {e}"}
    return out
HELD_OUT_SPLITS = ("dev", "calib", "excluded_smplx")
HISTORY_JSON = os.path.join(REPORTS, "provenance_history", "tier_hash_history.json")
HISTORY_FORMAT = "mdfm-tier-hash-history/1"
ARCHIVE_JSON = os.path.join(REPORTS, "provenance_history", "records_archive.json")
ARCHIVE_FORMAT = "mdfm-records-archive/1"
ID_COLUMNS = ("source", "modality", "subject_ns", "key")
TIER_COLUMNS = ("source", "subject_ns", "person_id", "key", "tier", "lane", "scope", "lockbox", "group")
OTHER_COLUMNS = ("manifest_source", "modality", "fps", "n_frames")
VALUE_COLUMNS = ("manifest_source", "modality", "fps")
TOLERATED_COLUMNS = ("fps",)
ENTRY_KEYS = ("records_sha256", "records_file_sha256", "records", "records_columns", "sources")
RELEASE = "RELEASE"
INTERNAL = "INTERNAL"
TIERS = (RELEASE, INTERNAL)                 # the tiers a run may have (a RELEASE run never meets an INTERNAL one)
LICENSE_MATRIX = os.path.join(paths.MOVEATLAS, "docs", "license_matrix.csv")
LICENSE_COLUMNS = ("id", "license", "redistribution", "portal_action")


class ProvenanceError(RuntimeError):
    """A hash that does not match its record, an audit that finds SMPL+H content, a guard that lets the canary
    through."""


# ---------------------------------------------------------------- hashes
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tiers():
    """reports/manifest_tiers/hashes.json."""
    return _json(TIERS_JSON)


def lockbox_sha256():
    """SHA-256 of reports/lockbox/lockbox_persons.csv, checked against lockbox.sha256 and against the
    lockbox_sha256 of reports/manifest_tiers/hashes.json (ProvenanceError otherwise)."""
    got = sha256_file(LOCKBOX_CSV)
    with open(LOCKBOX_SHA, encoding="utf-8") as f:
        want = f.read().split()[0]
    rec = tiers()["lockbox_sha256"]
    if not got == want == rec:
        raise ProvenanceError(f"lockbox list SHA-256 {got}, lockbox.sha256 {want}, hashes.json {rec}")
    return got


def records_sha256():
    """SHA-256 of the tiers records file, checked against records_file_sha256 of hashes.json."""
    got = sha256_file(records_path())
    want = tiers()["records_file_sha256"]
    if got != want:
        raise ProvenanceError(f"{records_path()} SHA-256 {got}, hashes.json records_file_sha256 {want}")
    return got


def moveatlas_head(root=None):
    """The commit of the MoveAtlas tree from its .git files (HEAD, the branch ref or packed-refs), read-only;
    '' when unreadable."""
    root = paths.MOVEATLAS if root is None else root
    git = os.path.join(root, ".git")
    try:
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        p = os.path.join(git, *ref.split("/"))
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        with open(os.path.join(git, "packed-refs"), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return ""
    return ""


def code_sha256(root=CODE_DIR):
    """SHA-256 over the .py files under root (AppleDouble sidecars and caches left out): one line
    '<relative path>\\t<file SHA-256>' per file, sorted by path."""
    lines = []
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if os.path.basename(p).startswith("._") or "__pycache__" in p:
            continue
        rel = os.path.relpath(p, root).replace("\\", "/")
        lines.append(f"{rel}\t{sha256_file(p)}\n")
    return sha256_bytes("".join(sorted(lines)).encode("utf-8"))


def rule_hashes(root=RULES_DIR):
    """{file name: SHA-256} of every file of docs/rules (sidecars left out)."""
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not name.startswith(("._", ".")) and os.path.isfile(p):
            out[name] = sha256_file(p)
    return out


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(cfg):
    """SHA-256 of a config: train.config.sha256 for a RunConfig, of its canonical JSON for a dict."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return sha256_bytes(_canonical(cfg).encode("utf-8"))
    from .config import sha256
    return sha256(cfg)


def hashes(cfg=None, body_store=None, window_store=None):
    """The hashes of a run (module docstring), as a JSON-ready dict."""
    t = tiers()
    state = _json(STATE_JSON) if os.path.exists(STATE_JSON) else {}
    body, win = _json(BODY_JSON), _json(WINDOW_JSON)
    head = moveatlas_head()
    out = {"tiers_version": t.get("version"), "records_sha256": t["records_sha256"],
           "records_file_sha256": records_sha256(), "lockbox_sha256": lockbox_sha256(),
           "moveatlas_head": head, "moveatlas_checked": state.get("head_sha", ""),
           "moveatlas_head_is_checked": bool(head) and head == state.get("head_sha"),
           "body_store_report": {"index_sha256": body.get("index_sha256"),
                                 "train_persons_sha256": body["person_lists"]["train"]["store_sha256"]},
           "window_store_report": {"person_sha256": win.get("person_sha256"), "files_sha256": win.get("files_sha256")},
           "rules_sha256": rule_hashes(), "config_sha256": config_sha256(cfg), "code_sha256": code_sha256()}
    if body_store is not None and is_open_store(body_store):
        out["open_body"] = open_body_hashes(body_store)
    elif body_store is not None:
        from ..data import amass_index      # the v1 AMASS body store only: no AMASS module at the top
        out["body_store"] = {"train_persons_sha256": amass_index.person_list_sha256(_body_persons(body_store))}
        if os.path.exists(amass_index.INDEX_PATH):
            out["body_store"]["index_sha256"] = sha256_file(amass_index.INDEX_PATH)
    if window_store is not None:
        out["window_store"] = {"person_sha256": window_store.person_hashes()}
        srcs = getattr(window_store, "sources", None)
        if isinstance(srcs, dict):
            out["window_store"]["files_sha256"] = {k: v.get("files_sha256") for k, v in sorted(srcs.items())}
    return out


# ---------------------------------------------------------------- person lists
def _amass_index():
    """mdfm.data.amass_index, or None in a tree that does not carry it (the public tree: plan decision P10). Imported
    here, not at the top, so that this module loads no AMASS module (plan phase E step 2)."""
    if importlib.util.find_spec("mdfm.data.amass_index") is None:
        return None
    from ..data import amass_index
    return amass_index


def _body_persons(body_store):
    return sorted(set(str(p) for p in body_store.takes["person_id"]))


def person_lists(body_store=None, window_store=None, fold="train"):
    """{source: sorted person ids} of what trains (module docstring): 'amass' from the body store, each real
    source of the window store at `fold`."""
    out = {}
    if body_store is not None:
        split = getattr(body_store, "split", "train")
        if split != "train":
            raise Refused(f"the body store of split {split!r} never trains")
        out[OPEN_BODIES if is_open_store(body_store) else AMASS] = _body_persons(body_store)
    if window_store is not None:
        for s, ids in window_store.person_lists(fold).items():
            out[s] = sorted(set(ids))
    return out


def list_hashes(lists):
    """{source: person_hash of its list} and 'all': the SHA-256 of the distinct '<source>\\t<id>' lines, sorted,
    joined by newlines."""
    out = {s: person_hash(ids) for s, ids in sorted(lists.items())}
    lines = sorted({f"{s}\t{p}" for s, ids in lists.items() for p in ids})
    out["all"] = sha256_bytes("\n".join(lines).encode("utf-8"))
    return out


def report_mismatches(lists, fold="train"):
    """Sources whose list does not hash to the value its report holds (list of str): 'amass' against the train
    list of reports/amass_split and reports/body_store, a real source against its fold in
    reports/window_store/hashes.json."""
    out = []
    win = _json(WINDOW_JSON).get("person_sha256", {})
    for s, ids in lists.items():
        if s == OPEN_BODIES:
            continue        # the open store's files are checked against its hashes.json when it opens (OpenBodyStore)
        if s == AMASS:
            from ..data import amass_index  # the v1 AMASS list only: no AMASS module at the top
            got = amass_index.person_list_sha256(ids)
            want = {_json(AMASS_JSON)["person_lists"]["train"]["sha256"],
                    _json(BODY_JSON)["person_lists"]["train"]["store_sha256"]}
            if want != {got}:
                out.append(f"{s}: {got} against the reports' {sorted(want)}")
        else:
            want = (win.get(s) or {}).get(fold)
            got = person_hash(ids)
            if want != got:
                out.append(f"{s}/{fold}: {got} against reports/window_store {want}")
    return out


def held_out_persons(window_store=None, fold="train"):
    """{source: sorted ids} of persons that never train: the window store's persons outside `fold` (dev, val)
    and the amass persons of the dev, calibration and excluded splits of the take index (when built)."""
    out = {}
    if window_store is not None:
        r = window_store.records
        d = r[r["fold"] != fold]
        for s, g in d.groupby("source", sort=True):
            out[s] = sorted(set(g["person_id"]))
    amass_index = _amass_index()
    if amass_index is not None and os.path.exists(amass_index.INDEX_PATH):
        ix = amass_index.index()
        out[AMASS] = sorted(set(ix.loc[ix["split"].isin(HELD_OUT_SPLITS), "person_id"]))
    return out


def record(cfg=None, body_store=None, window_store=None, fold="train", extra=None, tier=None):
    """The provenance of a run or checkpoint (JSON-ready): format, created, tier (the given one, else the
    config's, run_tier), hashes, persons (person_lists), persons_sha256 (list_hashes), held_out and its hashes,
    report_mismatches (empty when the stores are those the reports describe), licences (licences() of the
    training sources; ProvenanceError for a source without a row of the licence matrix), and extra. A RELEASE
    record of a window store whose tier attribute is INTERNAL raises ProvenanceError."""
    tier = run_tier(cfg) if tier is None else check_tier(tier)
    ws_tier = getattr(window_store, "tier", RELEASE) if window_store is not None else RELEASE
    if tier == RELEASE and check_tier(ws_tier) == INTERNAL:
        raise ProvenanceError("a RELEASE run never reads an INTERNAL window store")
    lists = person_lists(body_store, window_store, fold)
    held = held_out_persons(window_store, fold)
    out = {"format": FORMAT, "created": datetime.datetime.now().isoformat(timespec="seconds"), "tier": tier,
           "fold": fold, "hashes": hashes(cfg, body_store, window_store), "persons": lists,
           "persons_sha256": list_hashes(lists), "held_out": held, "held_out_sha256": list_hashes(held),
           "report_mismatches": report_mismatches(lists, fold)}
    if OPEN_BODIES in lists:
        out["open_body_sources"] = open_body_sources(body_store)
    out["licences"] = licences(licence_sources(lists, out.get("open_body_sources")))
    if extra:
        out["extra"] = dict(extra)
    return out


# ---------------------------------------------------------------- tier and licences (task S2.4)
def check_tier(tier):
    """The tier as given when it is one of TIERS; ValueError otherwise."""
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r}, expected one of {TIERS}")
    return tier


def run_tier(cfg):
    """The tier of a run config: a RunConfig (its tier field), a dict or its canonical JSON text (key 'tier', absent
    meaning RELEASE; the combined configuration of a G12 baseline, {'g12', 'run'} of baselines/config.combined,
    holds it in its run part), None (RELEASE). ValueError on a tier outside TIERS."""
    if cfg is None:
        return RELEASE
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    if isinstance(cfg, dict):
        if set(cfg) == {"g12", "run"} and isinstance(cfg["run"], dict):
            cfg = cfg["run"]
        return check_tier(cfg.get("tier", RELEASE))
    return check_tier(getattr(cfg, "tier", RELEASE))


def tier_of(prov):
    """The tier of a provenance record: its 'tier' field, RELEASE without one (every record written before task
    S2.4). ProvenanceError on a value outside TIERS."""
    t = prov.get("tier", RELEASE)
    if t not in TIERS:
        raise ProvenanceError(f"provenance tier {t!r}, expected one of {TIERS}")
    return t


def lineage_tiers(prov):
    """{'init_from' or 'migrated_from': tier} of the lineage entries of a record's extra that record a tier."""
    extra = prov.get("extra") or {}
    out = {}
    for name, entry in (("init_from", extra.get("init_from")),
                        ("migrated_from", (extra.get("migration") or {}).get("migrated_from"))):
        if isinstance(entry, dict) and "tier" in entry:
            out[name] = entry["tier"]
    return out


def tiered(prov, tier):
    """A copy of the provenance record `prov` for a checkpoint of a run of tier `tier` (module docstring): its
    'tier' set to tier and its 'licences' to licences() of its training sources (a licence record copied from another
    checkpoint, the migration case, is replaced). ProvenanceError for a RELEASE tier
    when the record or a lineage entry says INTERNAL (a RELEASE checkpoint never descends from an INTERNAL one), or
    for a training source without a row of the licence matrix."""
    tier = check_tier(tier)
    if tier == RELEASE:
        if tier_of(prov) == INTERNAL:
            raise ProvenanceError("a RELEASE checkpoint never carries the provenance of an INTERNAL run (init, "
                                  "resume or migration from an INTERNAL checkpoint)")
        bad = {k: v for k, v in lineage_tiers(prov).items() if v != RELEASE}
        if bad:
            raise ProvenanceError(f"a RELEASE checkpoint never descends from an INTERNAL one: lineage {bad}")
    out = json.loads(json.dumps(prov))
    out["tier"] = tier
    out["licences"] = licences(licence_sources(out.get("persons") or {}, out.get("open_body_sources")))
    return out


def _matrix_path(path=None):
    return os.path.abspath(LICENSE_MATRIX if path is None else str(path))


@functools.lru_cache(maxsize=None)
def licence_matrix(path):
    """(SHA-256, rows) of the licence matrix at the absolute path `path` (read-only), both from one read of its
    bytes, once per process and path (module docstring); rows: {id: (license, redistribution, portal_action)}.
    ProvenanceError on missing columns or a repeated id (a failed read is not kept)."""
    with open(path, "rb") as f:
        data = f.read()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    missing = [c for c in LICENSE_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ProvenanceError(f"{path}: columns {missing} missing")
    rows = {}
    for r in reader:
        k = r["id"]
        if k in rows:
            raise ProvenanceError(f"{path}: id {k!r} repeated")
        rows[k] = tuple(r[c] for c in LICENSE_COLUMNS[1:])
    return sha256_bytes(data), types.MappingProxyType(rows)


def licence_rows(path=None):
    """{id: {id, license, redistribution, portal_action}} of the licence matrix (default LICENSE_MATRIX, read-only),
    a fresh copy of licence_matrix's rows. ProvenanceError on missing columns or a repeated id."""
    _, rows = licence_matrix(_matrix_path(path))
    return {k: dict(zip(LICENSE_COLUMNS, (k,) + v)) for k, v in rows.items()}


def open_body_sources(body_store):
    """The sources of the open body store's served takes (sorted), which a record keeps as open_body_sources."""
    if body_store is None or not is_open_store(body_store):
        raise ProvenanceError("an open_bodies person list without the open body store it comes from")
    return sorted(str(s) for s in body_store.takes["source"].unique())


def licence_sources(lists, open_sources=None):
    """The source names whose licence rows a run inherits: the keys of its person lists, the open body store's key
    (OPEN_BODIES) replaced by the sources its takes come from (open_sources, a record's open_body_sources: the open
    motion capture and the two stage-1 marker sources, each with its own row of the licence matrix)."""
    out = set(lists) - {OPEN_BODIES}
    if OPEN_BODIES in lists:
        if not open_sources:
            raise ProvenanceError("an open_bodies person list without the sources of its takes (open_body_sources)")
        out |= set(open_sources)
    return sorted(out)


def licences(sources, path=None):
    """The licence record of the training sources (module docstring): matrix (path, forward slashes),
    matrix_sha256, rows {source: row} and inherited [{license, sources}] (each distinct licence text, sorted, with
    its sources), from licence_matrix (read once per process). ProvenanceError naming every source without a row."""
    path = _matrix_path(path)
    sha, _ = licence_matrix(path)
    rows = licence_rows(path)
    sources = sorted(set(sources))
    missing = [s for s in sources if s not in rows]
    if missing:
        raise ProvenanceError(f"training sources without a row of the licence matrix {path}: {missing}")
    by = collections.defaultdict(list)
    for s in sources:
        by[rows[s]["license"]].append(s)
    return {"matrix": path.replace("\\", "/"), "matrix_sha256": sha,
            "rows": {s: rows[s] for s in sources},
            "inherited": [{"license": k, "sources": v} for k, v in sorted(by.items())]}


# ---------------------------------------------------------------- guard
class DenyLists(NamedTuple):
    """Person ids that never train: lockbox, eval (tier EVAL), excluded (EXCLUDED_V1), cody (ids of CODY
    rows), namespaces (CODY namespaces, lower case), canary; and those that never train a RELEASE run: internal
    (ids of INTERNAL rows) and internal_sources (sources holding an INTERNAL row)."""
    lockbox: frozenset
    eval: frozenset
    excluded: frozenset
    cody: frozenset
    namespaces: frozenset
    canary: str
    internal: frozenset = frozenset()
    internal_sources: frozenset = frozenset()


@functools.lru_cache(maxsize=None)
def cody_namespaces():
    """frozenset of the CODY namespaces (lower case) of reports/manifest_tiers/cody_denylist.json and
    hashes.json, read once per process; needs neither the records file nor the index."""
    deny = _json(DENYLIST_JSON)
    return frozenset({str(n).strip().lower() for n in deny.get("namespaces", [])} |
                     {str(n).strip().lower() for n in tiers().get("cody_namespaces", [])})


def is_cody_id(*names):
    """True when a name (source, subject_ns or person id) is CODY by its prefix (canonical.is_cody) or by its
    namespace (cody_namespaces)."""
    ns = cody_namespaces()
    return is_cody(*names) or any(isinstance(n, str) and namespace(n) in ns for n in names)


@functools.lru_cache(maxsize=None)
def deny_lists():
    """DenyLists from the reports and the tiers records file (hash-checked), read once per process."""
    import pandas as pd
    lockbox_sha256()
    records_sha256()
    with open(LOCKBOX_CSV, encoding="utf-8", newline="") as f:
        lockbox = {r["person_id"] for r in csv.DictReader(f)}
    cols = ["source", "manifest_source", "subject_ns", "person_id", "tier", "lockbox"]
    df = pd.read_csv(records_path(), dtype=str, keep_default_na=False, encoding="utf-8", usecols=cols)
    df = df.rename(columns={"manifest_source": "origin"})
    lockbox |= set(df.loc[df["lockbox"].str.strip().str.lower().isin(("true", "1")), "person_id"])
    deny = _json(DENYLIST_JSON)
    ns = cody_namespaces()
    cody_src = df["source"].str.strip().str.lower().str.startswith("cody") | \
        df["origin"].str.strip().str.lower().str.startswith("cody")
    cody = set(df.loc[cody_src, "person_id"]) | set(df.loc[cody_src, "subject_ns"])
    internal = df[df["tier"] == INTERNAL]
    pairs = sorted({f"{s}|{p}" for s, p in zip(internal["source"], internal["person_id"])})
    got = sha256_bytes("\n".join(pairs).encode("utf-8"))
    want = (tiers().get("tier_person_list_sha256") or {}).get(INTERNAL)
    if got != want:
        raise ProvenanceError(f"INTERNAL persons of the records file hash to {got}, hashes.json "
                              f"tier_person_list_sha256 INTERNAL {want}")
    return DenyLists(frozenset(lockbox), frozenset(df.loc[df["tier"] == "EVAL", "person_id"]),
                     frozenset(df.loc[df["tier"] == "EXCLUDED_V1", "person_id"]), frozenset(cody), frozenset(ns),
                     str(deny["canary_subject_ns"]), frozenset(internal["person_id"]), frozenset(internal["source"]))


def _pairs(persons):
    """(source or None, person id) pairs of: a mapping {source: ids}, windows (objects with source and
    person_id), (source, id) pairs, or ids."""
    if isinstance(persons, dict):
        return [(s, str(p)) for s, ids in persons.items() for p in ids]
    out = []
    for x in persons:
        if hasattr(x, "person_id"):
            out.append((getattr(x, "source", None), str(x.person_id)))
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            out.append((x[0], str(x[1])))
        else:
            out.append((None, str(x)))
    return out


def _held_set(held_out):
    if not held_out:
        return frozenset()
    if isinstance(held_out, dict):
        return frozenset(str(p) for ids in held_out.values() for p in ids)
    return frozenset(str(p) for p in held_out)


def problems(persons, held_out=None, deny=None, tier=RELEASE):
    """Refused persons (list of str 'source id: reasons', module docstring); for a RELEASE tier also the
    INTERNAL persons and every person listed under a source holding an INTERNAL row ('INTERNAL')."""
    deny = deny_lists() if deny is None else deny
    release = check_tier(tier) == RELEASE
    held = _held_set(held_out)
    out = []
    for source, pid in sorted(set(_pairs(persons)), key=lambda x: (str(x[0]), x[1])):
        why = []
        if pid == deny.canary:
            why.append("canary")
        if is_cody(source, pid) or namespace(pid) in deny.namespaces or pid in deny.cody:
            why.append("CODY")
        if pid in deny.lockbox:
            why.append("lockbox")
        if pid in deny.eval:
            why.append("EVAL")
        if pid in deny.excluded:
            why.append("EXCLUDED_V1")
        if release and (pid in deny.internal or source in deny.internal_sources):
            why.append("INTERNAL")
        if pid in held:
            why.append("held out")
        if why:
            out.append(f"{source} {pid}: {', '.join(why)}")
    return out


def canary_check(deny=None):
    """Raise ProvenanceError unless the record reader refuses the canary (canonical.check) and problems()
    flags it."""
    deny = deny_lists() if deny is None else deny
    c = deny.canary
    try:
        reader_check({"source": c.split(":", 1)[0], "subject_ns": c, "person_id": c, "tier": "RELEASE",
                      "lockbox": False})
    except Refused:
        pass
    else:
        raise ProvenanceError(f"the record reader does not refuse the canary {c}")
    if not problems([(None, c)], deny=deny):
        raise ProvenanceError(f"the person guard lets the canary {c} through")


def assert_clean(persons, held_out=None, deny=None, tier=RELEASE):
    """Raise canonical.Refused naming every refused person of `persons` (a mapping {source: ids}, windows,
    (source, id) pairs or ids) for a run of tier `tier` (problems); canary_check runs first."""
    canary_check(deny)
    bad = problems(persons, held_out, deny, tier=tier)
    if bad:
        raise Refused(f"{len(bad)} persons may not train: {bad[:10]}")


# ---------------------------------------------------------------- audit
@functools.lru_cache(maxsize=1)
def _smplh_arrays():
    return M.smplh_arrays(allow_reference=True)


def smplh_arrays(allow_reference=False):
    """mfm.smplh_arrays(), read once per process, behind the reference flag (mdfm.body.backend)."""
    BK.require_reference("the SMPL+H array audit", allow_reference)
    return _smplh_arrays()


@functools.lru_cache(maxsize=1)
def open_licence_problems():
    """mdfm.body.openbody.check_licence() of the installed Anny, once per process."""
    from ..body import openbody as BO
    return tuple(BO.check_licence())


def audit_state_dict(sd, arrays=None, body=None):
    """Problems (list of str) of a state dict or any nested container of tensors (module docstring): with arrays
    given, mfm.audit_state_dict with them; else by the body of backend.model_body(body): the open body's screen
    and licence record, or the SMPL+H array audit behind the flag."""
    if arrays is not None:
        return M.audit_state_dict(sd, arrays)
    if BK.model_body(body) == BK.SMPLH:
        return M.audit_state_dict(sd, smplh_arrays())
    return M.audit_state_dict(sd, {}) + [f"open body licence record: {x}" for x in open_licence_problems()]


def state_of(x):
    """A state dict of a module, an optimizer or any object with state_dict(); a dict or None as given."""
    if x is None or isinstance(x, dict):
        return x
    return x.state_dict()


def audit(model=None, ema=None, opt=None, arrays=None, body=None):
    """Raise ProvenanceError (its problems attribute lists every finding, named model.<key>, ema.<key> or
    opt.<key>) when audit_state_dict finds a problem in the model, the EMA or the optimizer state (modules,
    optimizers or their state dicts), in one pass, or in the open body's licence record; return the three state
    dicts otherwise."""
    states = {"model": state_of(model), "ema": state_of(ema), "opt": state_of(opt)}
    bad = audit_state_dict({k: v for k, v in states.items() if v is not None}, arrays, body)
    if bad:
        err = ProvenanceError(f"state_dict audit: {len(bad)} problems: {bad[:10]}")
        err.problems = bad
        raise err
    return states


# ---------------------------------------------------------------- tier hash history and records drift
TIER_LINE = ID_COLUMNS + tuple(c for c in TIER_COLUMNS if c not in ID_COLUMNS)


def _open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _line(row, ix):
    return ("\t".join(row[i] for i in ix) + "\n").encode("utf-8")


def tier_hash_entry(path=None):
    """The tier hash entry of a tiers records file (module docstring; default the records file of this tree):
    records_sha256, records_file_sha256, records, records_columns and sources {source: rows, tier_sha256,
    columns_sha256, values}. Raises ProvenanceError when a column is in neither group, a row has another
    width than the header or the rows are not in the build's order (identity sorted, unique)."""
    path = records_path() if path is None else str(path)
    whole = hashlib.sha256()
    by = {}
    n = 0
    with _open_text(path) as f:
        reader = csv.reader(f)
        head = next(reader)
        known = set(TIER_COLUMNS) | set(OTHER_COLUMNS)
        extra, missing = sorted(set(head) - known), sorted(known - set(head))
        if extra or missing or len(set(head)) != len(head):
            raise ProvenanceError(f"{path}: columns outside the tier and other groups {extra}, missing {missing}")
        at = {c: i for i, c in enumerate(head)}
        id_ix = [at[c] for c in ID_COLUMNS]
        tier_ix = [at[c] for c in TIER_LINE]
        col_ix = {c: id_ix + [at[c]] for c in OTHER_COLUMNS}
        val_ix = {c: at[c] for c in VALUE_COLUMNS}
        src_ix = at["source"]
        prev = None
        for row in reader:
            if len(row) != len(head):
                raise ProvenanceError(f"{path}: row {n + 1} has {len(row)} fields, the header {len(head)}")
            whole.update(("\t".join(row) + "\n").encode("utf-8"))
            ident = tuple(row[i] for i in id_ix)
            if prev is not None and ident <= prev:
                raise ProvenanceError(f"{path}: row {n + 1} {ident} is not after {prev}: the rows are not in the "
                                      f"build's order ({', '.join(ID_COLUMNS)}, sorted, unique)")
            prev = ident
            b = by.get(row[src_ix])
            if b is None:
                b = by[row[src_ix]] = {"rows": 0, "tier": hashlib.sha256(),
                                       "cols": {c: hashlib.sha256() for c in OTHER_COLUMNS},
                                       "values": {c: collections.Counter() for c in VALUE_COLUMNS}}
            b["rows"] += 1
            b["tier"].update(_line(row, tier_ix))
            for c, ix in col_ix.items():
                b["cols"][c].update(_line(row, ix))
            for c, i in val_ix.items():
                b["values"][c][row[i]] += 1
            n += 1
    sources = {s: {"rows": b["rows"], "tier_sha256": b["tier"].hexdigest(),
                   "columns_sha256": {c: h.hexdigest() for c, h in b["cols"].items()},
                   "values": {c: dict(sorted(v.items())) for c, v in b["values"].items()}}
               for s, b in sorted(by.items())}
    return {"records_sha256": whole.hexdigest(), "records_file_sha256": sha256_file(path), "records": n,
            "records_columns": list(head), "sources": sources}


_ENTRIES = {}


def current_tier_entry():
    """tier_hash_entry of the records file of this tree, computed once per file SHA-256 and process."""
    path = records_path()
    sha = sha256_file(path)
    if sha not in _ENTRIES:
        _ENTRIES[sha] = tier_hash_entry(path)
    return _ENTRIES[sha]


def entry_sha256(entry):
    """SHA-256 of the canonical JSON of an entry without its entry_sha256."""
    return sha256_bytes(_canonical({k: v for k, v in entry.items() if k != "entry_sha256"}).encode("utf-8"))


def _groups():
    return {"id_columns": list(ID_COLUMNS), "tier_columns": list(TIER_COLUMNS), "other_columns": list(OTHER_COLUMNS),
            "value_columns": list(VALUE_COLUMNS)}


def tier_hash_history(path=None):
    """{records_sha256: entry} of the history file (default HISTORY_JSON; {} when absent). Raises
    ProvenanceError on another format, other column groups or an entry that does not hash to its
    entry_sha256 or sits under another key."""
    path = HISTORY_JSON if path is None else str(path)
    if not os.path.exists(path):
        return {}
    doc = _json(path)
    if doc.get("format") != HISTORY_FORMAT:
        raise ProvenanceError(f"{path}: format {doc.get('format')!r}, expected {HISTORY_FORMAT}")
    moved = [k for k, v in _groups().items() if doc.get(k) != v]
    if moved:
        raise ProvenanceError(f"{path}: column groups {moved} differ from those of this code")
    out = {}
    for k, e in (doc.get("entries") or {}).items():
        if e.get("records_sha256") != k or e.get("entry_sha256") != entry_sha256(e):
            raise ProvenanceError(f"{path}: the entry {k} does not hash to its entry_sha256 or sits under "
                                  "another records_sha256")
        out[k] = e
    return out


def add_tier_entry(doc, entry):
    """(doc, added): the history object doc (a new one when None) with entry stored under its records_sha256,
    entry_sha256 set. An entry already held is kept when its content (ENTRY_KEYS) is equal (added False); a
    different content raises ProvenanceError: entries are never rewritten."""
    doc = dict({"format": HISTORY_FORMAT}, **_groups(), entries={}) if doc is None else doc
    if doc.get("format") != HISTORY_FORMAT:
        raise ProvenanceError(f"history format {doc.get('format')!r}, expected {HISTORY_FORMAT}")
    moved = [k for k, v in _groups().items() if doc.get(k) != v]
    if moved:
        raise ProvenanceError(f"history column groups {moved} differ from those of this code")
    entries = doc.setdefault("entries", {})
    k = entry["records_sha256"]
    held = entries.get(k)
    if held is not None:
        if any(held.get(c) != entry.get(c) for c in ENTRY_KEYS):
            raise ProvenanceError(f"the history holds another entry for records_sha256 {k}: entries are never "
                                  "rewritten")
        return doc, False
    e = dict(entry)
    e["entry_sha256"] = entry_sha256(e)
    entries[k] = e
    return doc, True


def compare_tier_entries(old, new, run_sources=()):
    """(changes, problems) between two entries. changes: one dict per source that differs (source, kind
    'added', 'removed', 'tier' or 'columns', rows [old, new], columns that moved, their value counts before and
    after for VALUE_COLUMNS, their columns_sha256 [old, new], tolerated). A change is tolerated only when its
    kind is 'columns', its source is outside run_sources and its columns are within TOLERATED_COLUMNS;
    problems names every other change, a records_columns change and a source of the run with no row in old."""
    run = set(run_sources)
    problems = []
    if list(old.get("records_columns") or []) != list(new.get("records_columns") or []):
        problems.append(f"records_columns {old.get('records_columns')} -> {new.get('records_columns')}")
    absent = sorted(run - set(old["sources"]))
    if absent:
        problems.append(f"sources of the run with no row in the records, whose changes cannot be told apart: "
                        f"{absent}")
    changes = []
    for s in sorted(set(old["sources"]) | set(new["sources"])):
        o, n = old["sources"].get(s), new["sources"].get(s)
        rows = [o["rows"] if o else 0, n["rows"] if n else 0]
        if o is None or n is None:
            ch = {"source": s, "kind": "added" if o is None else "removed", "rows": rows, "columns": []}
        elif o["rows"] != n["rows"] or o["tier_sha256"] != n["tier_sha256"]:
            ch = {"source": s, "kind": "tier", "rows": rows, "columns": [],
                  "tier_sha256": [o["tier_sha256"], n["tier_sha256"]]}
        else:
            cols = [c for c in OTHER_COLUMNS if o["columns_sha256"][c] != n["columns_sha256"][c]]
            if not cols:
                continue
            ch = {"source": s, "kind": "columns", "rows": rows, "columns": cols,
                  "values": {c: {"before": o["values"].get(c), "after": n["values"].get(c)}
                             for c in cols if c in VALUE_COLUMNS},
                  "columns_sha256": {c: [o["columns_sha256"][c], n["columns_sha256"][c]] for c in cols}}
        if ch["kind"] != "columns":
            why = f"rows {ch['kind']}" if ch["kind"] in ("added", "removed") else "tier columns or row set changed"
        elif s in run:
            why = f"{ch['columns']} changed in a source of the run (per-sample fps rule)"
        elif not set(ch["columns"]) <= set(TOLERATED_COLUMNS):
            why = f"{ch['columns']} changed, only {list(TOLERATED_COLUMNS)} may change outside the run"
        else:
            why = ""
        ch["tolerated"] = not why
        if why:
            problems.append(f"{s}: {why}")
        changes.append(ch)
    return changes, problems


def _tree_path(path):
    """path relative to this tree, forward slashes; the absolute path when it lies outside."""
    a = os.path.abspath(str(path))
    if paths._inside(a, paths.TREE):
        return os.path.relpath(a, paths.TREE).replace("\\", "/")
    return a.replace("\\", "/")


def run_sources(prov):
    """Sorted sources of a provenance record: its training lists and its held-out lists."""
    return sorted(set(prov.get("persons") or {}) | set(prov.get("held_out") or {}))


def records_drift(prov, now=None, history=None, current=None, state=None, archive=None):
    """None when the records_sha256 of a provenance record (record() or a checkpoint's) equals the current
    one; otherwise the evidence (JSON-ready) that the narrow drift policy tolerates the change (module
    docstring), or ProvenanceError naming every problem (its problems attribute lists them). now: hashes()
    (default read now); history: tier_hash_history() (default HISTORY_JSON); current: the entry of the live
    records file (default current_tier_entry()). state: None (the live records state) or the records_sha256 of an
    archived records state (archived_state): the run's records state is then compared with that state instead of
    the live one, by the same policy (now and current default to the archived state's hashes and entry); archive:
    records_archive() (default ARCHIVE_JSON)."""
    then = prov["hashes"]
    if state is not None:
        named, arch = hashes_at(state, now, archive=archive, history=history)
        now = named
        current = arch["entry"] if current is None else current
    now = hashes() if now is None else now
    a, b = then.get("records_sha256"), now.get("records_sha256")
    if a == b:
        return None
    hist = tier_hash_history() if history is None else history
    where = _tree_path(HISTORY_JSON)
    problems = []
    old = hist.get(a)
    new = current_tier_entry() if current is None else current
    if old is None:
        problems.append(f"the history ({where}) holds no entry for the records_sha256 {a} of the checkpoint")
    else:
        for k in ("records_file_sha256", "lockbox_sha256"):
            if k in old and old[k] != then.get(k):
                problems.append(f"the history entry of {a} has {k} {old[k]}, the checkpoint {then.get(k)}")
    if new["records_sha256"] != b or new["records_file_sha256"] != now.get("records_file_sha256"):
        problems.append(f"the live records file hashes to {new['records_sha256']} (file "
                        f"{new['records_file_sha256']}), hashes.json says {b} (file {now.get('records_file_sha256')})")
    held = hist.get(b)
    if held is not None and any(held.get(k) != new.get(k) for k in ENTRY_KEYS):
        problems.append(f"the history entry of {b} differs from the live records file")
    run = run_sources(prov)
    changes = []
    if not problems:
        changes, problems = compare_tier_entries(old, new, run)
    if problems:
        err = ProvenanceError(f"records_sha256 {a} -> {b}, not tolerated: {problems}")
        err.problems = problems
        raise err
    return {"policy": f"tolerated: every changed row lies in a source outside the run and changes only "
                      f"{list(TOLERATED_COLUMNS)}; no row added or removed, no tier column changed, lockbox "
                      "hash unchanged",
            "records_sha256": [a, b], "records_file_sha256": [then.get("records_file_sha256"),
                                                               now.get("records_file_sha256")],
            "lockbox_sha256": now.get("lockbox_sha256"), "run_sources": run, "history": where,
            "history_entry_sha256": old.get("entry_sha256"), "changes": changes}


# ---------------------------------------------------------------- archived records states
def records_archive(path=None):
    """{records_sha256: record} of the archive record (default ARCHIVE_JSON, written by
    scripts/archive_records_state.py; {} when absent). A record names the archived copy of a records file of an
    earlier tiers state (cache_path, relative to the cache folder, gitignored), its records_file_sha256, records and
    the entry_sha256 of the history entry it was checked against. Raises ProvenanceError on another format or a
    record under another key."""
    path = ARCHIVE_JSON if path is None else str(path)
    if not os.path.exists(path):
        return {}
    doc = _json(path)
    if doc.get("format") != ARCHIVE_FORMAT:
        raise ProvenanceError(f"{path}: format {doc.get('format')!r}, expected {ARCHIVE_FORMAT}")
    out = {}
    for k, r in (doc.get("entries") or {}).items():
        if r.get("records_sha256") != k:
            raise ProvenanceError(f"{path}: the record under {k} names records_sha256 {r.get('records_sha256')}")
        out[k] = r
    return out


def archive_path(record):
    """Absolute path of the archived records file of a records_archive() record."""
    return os.path.join(paths.CACHE, *str(record["cache_path"]).split("/"))


def archived_state(state, archive=None, history=None):
    """The archived records state named by its records_sha256 (decision of 2026-11-12, plan step M7: a run is
    read under the records state it was born on, the drift policy is not widened): {records_sha256,
    records_file_sha256, records, path, lockbox_sha256 (of the history entry), history_entry_sha256, entry (the
    tier hash entry recomputed from the archived file)}. archive: records_archive() (default ARCHIVE_JSON);
    history: tier_hash_history() (default HISTORY_JSON). Raises FileNotFoundError when the archived file is absent
    from this machine, ProvenanceError when the state is not in the archive record, the file does not hash to
    its record, the history holds no entry for the state or one that differs from the file's entry."""
    arch = records_archive() if archive is None else archive
    rec = arch.get(state)
    if rec is None:
        raise ProvenanceError(f"records state {state} is not archived ({_tree_path(ARCHIVE_JSON)} holds "
                              f"{sorted(arch)}): run scripts/archive_records_state.py")
    p = archive_path(rec)
    if not os.path.exists(p):
        raise FileNotFoundError(f"the archived records file of state {state} is absent from this machine: {p}")
    sha = sha256_file(p)
    if sha != rec.get("records_file_sha256"):
        raise ProvenanceError(f"{p} SHA-256 {sha}, the archive record says {rec.get('records_file_sha256')}")
    if sha not in _ENTRIES:
        _ENTRIES[sha] = tier_hash_entry(p)
    entry = _ENTRIES[sha]
    hist = tier_hash_history() if history is None else history
    held = hist.get(state)
    problems = []
    if entry["records_sha256"] != state:
        problems.append(f"{p} hashes to records_sha256 {entry['records_sha256']}")
    if held is None:
        problems.append(f"the history ({_tree_path(HISTORY_JSON)}) holds no entry for {state}")
    elif any(held.get(k) != entry.get(k) for k in ENTRY_KEYS):
        problems.append(f"the history entry of {state} differs from the archived file")
    elif rec.get("history_entry_sha256") not in (None, held.get("entry_sha256")):
        problems.append(f"the archive record was checked against history entry {rec.get('history_entry_sha256')}, "
                        f"the history holds {held.get('entry_sha256')}")
    if problems:
        raise ProvenanceError(f"archived records state {state}: {problems}")
    return {"records_sha256": state, "records_file_sha256": sha, "records": entry["records"],
            "path": p.replace("\\", "/"), "lockbox_sha256": held.get("lockbox_sha256"),
            "history_entry_sha256": held.get("entry_sha256"), "entry": entry}


def hashes_at(state, now=None, archive=None, history=None):
    """(hashes, archived): hashes() (or the now given) with records_sha256 and records_file_sha256 replaced by
    those of the archived records state (archived_state), and that state. The lockbox is never relaxed: the
    lockbox_sha256 the history recorded for the state must equal the live one (ProvenanceError otherwise). The
    person guard keeps reading the live records file, which only adds refusals."""
    arch = archived_state(state, archive=archive, history=history)
    out = dict(hashes() if now is None else now)
    if arch["lockbox_sha256"] is not None and arch["lockbox_sha256"] != out.get("lockbox_sha256"):
        raise ProvenanceError(f"archived records state {state} was written with lockbox_sha256 "
                              f"{arch['lockbox_sha256']}, the live lockbox is {out.get('lockbox_sha256')}")
    out.update(records_sha256=arch["records_sha256"], records_file_sha256=arch["records_file_sha256"])
    return out, arch
