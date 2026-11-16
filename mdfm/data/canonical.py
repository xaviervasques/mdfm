"""Canonical record reader: every manifest row of the program, its MoveAtlas container and its arrays.

Rows: cache/manifest/manifest_records.csv.gz (scripts/build_tiers_and_lockbox.py: source, modality,
subject_ns, person_id, key, fps, n_frames, tier, lane, scope, lockbox) joined on (source, modality,
subject_ns, key) with cache/manifest/record_index.csv.gz (scripts/build_record_index.py: container
path relative to MoveAtlas, record number and member prefix inside it, checks). Containers are
moveatlas-record/1 files (MoveAtlas docs/RECORD_FORMAT.md): a ZIP whose meta.json holds the record
dicts, arrays as .npy members named by {"$npy": member}; record i of a list container has the member
prefix r{i:06d}/. They are read here with zipfile, json and numpy (allow_pickle=False), never with
MoveAtlas code, and never written.

Refusals, checked before any container is opened:
- tier EXCLUDED_V1, always;
- CODY and the rest of tier EVAL, unless allow_eval=True. A CODY row is one whose source,
  manifest_source, subject_ns or person_id starts with "cody" (cody_1, cody_2, cody_3, the canary and
  any other cody namespace), or whose subject_ns or person_id sits in a namespace that a CODY source
  uses (cody_1 files its adults under "adult:");
- lockbox rows, unless allow_lockbox=True.

A loaded record is the record dict under its contract names (MoveAtlas docs/MANIFEST_SCHEMA.md):
pose coords [T, 15, 3] float32 (z masked when dim is 2), imu signal [T, L, 9] (acc g, gyro deg/s,
mag uT), force [T, 2, 9] N, emg and eeg signal [T, C], silhouette contour [T, 64, 2], with T the
record's n_frames; plus fps (float, native rate of that record, NaN when neither the record nor the
manifest declares it), the manifest fields that no contract uses (manifest_source, person_id, key,
tier, lane, scope, lockbox, always the index values) and the full index row under "manifest". The
record's own source, modality (pose: 2d_pixel or 3d_meter), subject_ns and n_frames keep their
contract meaning.
"""
import collections
import io
import json
import math
import os
import zipfile

import numpy as np

from mdfm import paths

FORMAT = "moveatlas-record/1"
META = "meta.json"
TOKEN = "$npy"
JOIN = ["source", "modality", "subject_ns", "key"]
MODALITIES = ("pose", "imu", "force", "emg", "eeg", "silhouette")
MAIN_ARRAY = {"pose": "coords", "imu": "signal", "force": "force", "emg": "signal", "eeg": "signal",
              "silhouette": "contour"}
# per-dataset list containers under <registry path>/processed/ and the family each announces
NAMED = {"canonical.npz": "pose", "canonical_imu.npz": "imu", "canonical_force.npz": "force",
         "canonical_silhouette.npz": "silhouette"}
SHARDS = "canonical"          # canonical/<shard>.npz, family read from the record keys
CODY_NAMESPACES = ("cody_1", "cody_2", "cody_3")
# the incremental build of scripts/build_record_index.py never opens a container of a source it adds when the source
# holds a row of these tiers or a lockbox row: no meta.json, no array; its rows enter the index with this status and
# no container (never_opened)
NOT_OPENED = "not_opened"
NEVER_OPENED_TIERS = ("EVAL", "EXCLUDED_V1")
_ROW_FIELDS = ("manifest_source", "person_id", "key", "tier", "lane", "scope", "lockbox")
_HEADER_CACHE = 32


class Refused(PermissionError):
    """A row the caller's flags do not allow: EXCLUDED_V1, CODY or EVAL without allow_eval, lockbox
    without allow_lockbox."""


# ---------------------------------------------------------------- paths
def manifest_path():
    return os.path.join(paths.CACHE, "manifest", "manifest_records.csv.gz")


def index_path():
    return os.path.join(paths.CACHE, "manifest", "record_index.csv.gz")


def container_path(relpath):
    """Absolute path of a container given relative to the MoveAtlas root (read-only)."""
    return f"{paths.MOVEATLAS}/{relpath}"


def member_prefix(record):
    """Member prefix of record number `record` in a list container; '' for a kind-record container (-1)."""
    return "" if int(record) < 0 else f"r{int(record):06d}/"


# ---------------------------------------------------------------- refusals
def is_cody(*names):
    """True when any name (source, subject_ns, person_id) is a CODY namespace (starts with 'cody')."""
    return any(isinstance(n, str) and n.strip().lower().startswith("cody") for n in names)


def namespace(name):
    """The namespace of a namespaced id ('adult' for 'adult:P01'); '' for an id without a colon."""
    name = str(name)
    return name.split(":", 1)[0].strip().lower() if ":" in name else ""


_CODY_NS = None


def cody_namespaces():
    """Namespaces of the subject_ns and person_id of the index rows of a CODY source ('adult' for
    cody_1, 'cody_external' for cody_3): an id in one of them is CODY under any source."""
    _index()
    return _CODY_NS


def _is_cody_row(row):
    if _truthy(row.get("cody", False)) or is_cody(row.get("source"), row.get("manifest_source"),
                                                  row.get("subject_ns"), row.get("person_id")):
        return True
    ns = [n for n in (namespace(row.get(c, "")) for c in ("subject_ns", "person_id")) if n]
    return bool(ns) and not cody_namespaces().isdisjoint(ns)


def _prefix_mask(df, cols):
    m = np.zeros(len(df), dtype=bool)
    for c in cols:
        m |= df[c].str.strip().str.lower().str.startswith("cody").to_numpy(dtype=bool)
    return m


def _truthy(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes")


def _refuse_cody(row, allow_eval):
    """The name test alone (no index read): refuses a CODY-named row before anything is opened."""
    if not allow_eval and is_cody(row.get("source"), row.get("manifest_source"), row.get("subject_ns"),
                                  row.get("person_id")):
        raise Refused(f"CODY is evaluation-only: {row.get('source')} {row.get('subject_ns')} "
                      "(pass allow_eval=True in an evaluation stage)")


def never_opened_source(rows):
    """True when the incremental build of the record index never opens the containers of a source with these
    manifest rows: one of them has a tier of NEVER_OPENED_TIERS or is a lockbox row."""
    return any(str(r.get("tier", "")) in NEVER_OPENED_TIERS or _truthy(r.get("lockbox", False)) for r in rows)


def never_opened(df):
    """Boolean mask of the index() rows the incremental build never opened (status NOT_OPENED): they carry their
    manifest columns and no container, so load refuses them (EVAL without allow_eval) or raises LookupError."""
    return (df["status"] == NOT_OPENED).to_numpy(dtype=bool)


def check(row, allow_eval=False, allow_lockbox=False):
    """Raise Refused when the flags do not allow this row; return the row otherwise. The CODY namespace
    test reads the index (cody_namespaces) when the names alone do not settle it."""
    row = _as_dict(row)
    tier = str(row.get("tier", ""))
    if tier == "EXCLUDED_V1":
        raise Refused(f"tier EXCLUDED_V1 is never loaded: {row.get('source')} {row.get('subject_ns')}")
    _refuse_cody(row, allow_eval)
    if not allow_eval and _is_cody_row(row):
        raise Refused(f"CODY namespace is evaluation-only: {row.get('source')} {row.get('subject_ns')} "
                      "(pass allow_eval=True in an evaluation stage)")
    if not allow_eval and tier == "EVAL":
        raise Refused(f"tier EVAL is evaluation-only: {row.get('source')} {row.get('subject_ns')} "
                      "(pass allow_eval=True in an evaluation stage)")
    if not allow_lockbox and _truthy(row.get("lockbox", False)):
        raise Refused(f"lockbox row: {row.get('source')} {row.get('person_id')} (pass allow_lockbox=True "
                      "only for the sealed gate read)")
    return row


# ---------------------------------------------------------------- containers
_HEADERS = collections.OrderedDict()


def read_header(relpath, zf=None):
    """meta.json of a container (relative to MoveAtlas), parsed and cached (the last 32 containers)."""
    if relpath in _HEADERS:
        _HEADERS.move_to_end(relpath)
        return _HEADERS[relpath]
    if zf is None:
        with zipfile.ZipFile(container_path(relpath)) as z:
            return read_header(relpath, z)
    if META not in zf.namelist():
        raise ValueError(f"{relpath}: not a {FORMAT} container (no {META})")
    header = json.loads(zf.read(META).decode("utf-8"))
    if header.get("format") != FORMAT:
        raise ValueError(f"{relpath}: unknown format {header.get('format')!r}")
    _HEADERS[relpath] = header
    while len(_HEADERS) > _HEADER_CACHE:
        _HEADERS.popitem(last=False)
    return header


def header_records(header):
    """The record dicts of a header, as metadata (arrays still tokens): one for kind record."""
    return [header["meta"]] if header.get("kind") == "record" else list(header.get("records") or [])


def record_meta(header, record):
    """Metadata of record `record` of a header (arrays still tokens); -1 (or 0) for a kind-record
    container. A number outside a list container raises IndexError (no negative indexing)."""
    record = int(record)
    if header.get("kind") == "record":
        if record not in (-1, 0):
            raise IndexError(f"record {record} of a one-record container")
        return header["meta"]
    records = header.get("records") or []
    if not 0 <= record < len(records):
        raise IndexError(f"record {record} outside a list container of {len(records)} records")
    return records[record]


def modality_of(meta, hint=None):
    """Canonical family of a record from its keys (coords, location_ids, force, contour, channels)."""
    if "coords" in meta:
        return "pose"
    if "location_ids" in meta:
        return "imu"
    if "force" in meta:
        return "force"
    if "contour" in meta:
        return "silhouette"
    ch = meta.get("channels")
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        if "site" in ch[0]:
            return "eeg"
        if "muscle" in ch[0] or "kind" in ch[0]:
            return "emg"
    return hint


def record_key(meta):
    """The id of a record within its subject: task, else take, else sequence (as str)."""
    for k in ("task", "take", "sequence"):
        if meta.get(k) is not None:
            return str(meta[k])
    return ""


def read_npy(zf, member):
    with zf.open(member) as f:
        return np.lib.format.read_array(io.BytesIO(f.read()), allow_pickle=False)


def _detok(v, zf, prefix):
    if isinstance(v, dict):
        if set(v) == {TOKEN}:
            return read_npy(zf, f"{prefix}{v[TOKEN]}.npy")
        return {k: _detok(x, zf, prefix) for k, x in v.items()}
    if isinstance(v, list):
        return [_detok(x, zf, prefix) for x in v]
    return v


def read_record(zf, header, record):
    """Record `record` (-1 for kind record) of an open container, arrays as numpy, '_arrays' dropped."""
    meta = record_meta(header, record)
    prefix = "" if header.get("kind") == "record" else member_prefix(record)
    rec = {k: _detok(v, zf, prefix) for k, v in meta.items() if k != "_arrays"}
    # containers of the first writer listed top-level arrays in _arrays without a token
    for name in meta.get("_arrays", []):
        if name not in rec and "." not in name:
            rec[name] = read_npy(zf, f"{prefix}{name}.npy")
    return rec


# ---------------------------------------------------------------- index
_INDEX = None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _read_frame(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def index():
    """The manifest joined with the record index, one row per manifest row (a copy; cached).

    Types: lockbox bool, n_frames int, fps float (NaN when blank), record int (-1 for a kind-record
    container or an unresolved row), rec_n_frames int (-1 when unknown), rec_fps float; container is
    '' and status 'unresolved' (or 'unindexed' when the index lacks the row, NOT_OPENED for a row of a source
    the incremental build never opened: never_opened) for a row with no record; cody bool (the CODY test of the module docstring, names and namespaces)."""
    return _index().copy()


def _index():
    global _INDEX, _CODY_NS
    if _INDEX is None:
        import pandas as pd
        man = _read_frame(manifest_path())
        idx = _read_frame(index_path())
        df = man.merge(idx, on=JOIN, how="left", validate="one_to_one", indicator=True)
        missing = df["_merge"] == "left_only"
        df = df.drop(columns="_merge")
        for c in ("container", "prefix", "rule", "reason", "fps_check"):
            df[c] = df[c].fillna("")
        df["status"] = df["status"].fillna("").mask(missing, "unindexed")
        df["lockbox"] = df["lockbox"].map(_truthy).astype(bool)
        df["n_frames"] = pd.to_numeric(df["n_frames"], errors="coerce").fillna(-1).astype(int)
        for c in ("record", "rec_n_frames", "n_candidates", "shared"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
        for c in ("fps", "rec_fps"):     # Python float: exact round trip of the written repr
            df[c] = df[c].map(_float).astype(float)
        src = _prefix_mask(df, ("source", "manifest_source"))
        ns = frozenset({namespace(x) for c in ("subject_ns", "person_id") for x in df.loc[src, c]} - {""})
        cody = _prefix_mask(df, ("source", "manifest_source", "subject_ns", "person_id"))
        for c in ("subject_ns", "person_id"):
            cody |= df[c].map(lambda x: namespace(x) in ns).to_numpy(dtype=bool)
        df["cody"] = cody
        _CODY_NS, _INDEX = ns, df
    return _INDEX


def _as_dict(row):
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()}


_POS = None


def _row_of(row):
    """The index row with the join keys of `row`: tier, lockbox and container come from the index,
    never from the caller."""
    global _POS
    df = _index()
    if _POS is None:
        _POS = {k: i for i, k in enumerate(zip(*(df[c] for c in JOIN)))}
    i = _POS.get(tuple(str(row.get(c, "")) for c in JOIN))
    if i is None:
        raise LookupError(f"no index row for {[row.get(k) for k in JOIN]}")
    return _as_dict(df.iloc[i])


def _check_meta(meta, row):
    """Refuse a record whose metadata disagrees with what the index saw (a stale index): subject_ns,
    n_frames against rec_n_frames, and the record id for rows resolved by identity."""
    bad = []
    if meta.get("subject_ns") is not None and str(meta["subject_ns"]) != str(row.get("subject_ns")):
        bad.append(f"subject_ns {meta['subject_ns']} (index {row.get('subject_ns')})")
    n = int(row.get("rec_n_frames", -1))
    if n >= 0 and meta.get("n_frames") is not None and int(meta["n_frames"]) != n:
        bad.append(f"n_frames {meta['n_frames']} (index {n})")
    if row.get("rule") == "identity" and record_key(meta) != str(row.get("key")):
        bad.append(f"id {record_key(meta)} (index {row.get('key')})")
    if bad:
        raise ValueError(f"{row.get('container')} record {row.get('record')} holds {'; '.join(bad)}: "
                         "rebuild the record index (scripts/build_record_index.py)")


def _finish(rec, row):
    fps = rec.get("fps")
    if fps is None:
        fps = row.get("fps")
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = float("nan")
    rec["fps"] = fps if not math.isinf(fps) else float("nan")
    for k in _ROW_FIELDS:
        if k in row:
            rec[k] = row[k]
    rec["manifest"] = dict(row)
    return rec


def _read_row(zf, row):
    header = read_header(row["container"], zf)
    _check_meta(record_meta(header, row["record"]), row)
    return _finish(read_record(zf, header, row["record"]), row)


# ---------------------------------------------------------------- public reader
def load(row, allow_eval=False, allow_lockbox=False, opener=None):
    """One record as a dict (module docstring). `row` is an index() row (Series or dict) or any mapping
    with source, modality, subject_ns and key; the index row with those keys decides tier, lockbox and
    container. Raises Refused before any file is opened (a CODY name in `row` itself is refused before
    the index is read), LookupError for a row absent from the index or resolved to no record, and
    ValueError when the record's metadata disagrees with the index (a stale index). opener: None (the
    container is opened and closed here) or a callable taking the container's path and returning an open
    zipfile.ZipFile that the caller owns and closes (mdfm.data.sources._Containers keeps one open across the
    rows of a container); it is called after every refusal above, so the two paths refuse alike."""
    row = _as_dict(row)
    _refuse_cody(row, allow_eval)
    row = check(_row_of(row), allow_eval, allow_lockbox)
    if not row.get("container"):
        raise LookupError(f"unresolved row {[row.get(k) for k in JOIN]}: {row.get('reason') or row.get('status')}")
    if opener is not None:
        return _read_row(opener(container_path(row["container"])), row)
    with zipfile.ZipFile(container_path(row["container"])) as zf:
        return _read_row(zf, row)


def _as_set(v):
    return {v} if isinstance(v, str) else set(v)


def _select(modality=None, tier=None, lane=None, sources=None, allow_eval=False, allow_lockbox=False):
    """The index rows iterate() reads, grouped by container in record order, one row per record."""
    # refusals by name alone, before the index is read
    if tier is not None and "EXCLUDED_V1" in _as_set(tier):
        raise Refused("tier EXCLUDED_V1 is never loaded")
    if tier is not None and not allow_eval and "EVAL" in _as_set(tier):
        raise Refused("tier EVAL is evaluation-only (pass allow_eval=True in an evaluation stage)")
    if sources is not None and not allow_eval and any(is_cody(x) for x in _as_set(sources)):
        raise Refused("CODY is evaluation-only (pass allow_eval=True in an evaluation stage)")
    df = _index()
    wanted = {"modality": modality, "tier": tier, "lane": lane}
    for col, want in wanted.items():
        if want is not None:
            unknown = _as_set(want) - set(df[col].unique())
            if unknown:
                raise ValueError(f"unknown {col} {sorted(unknown)}; known: {sorted(df[col].unique())}")
    sel = df["container"] != ""
    if sources is not None:
        s = _as_set(sources)
        named = df["source"].isin(s) | df["manifest_source"].isin(s)
        unknown = s - set(df.loc[named, "source"]) - set(df.loc[named, "manifest_source"])
        if unknown:
            raise LookupError(f"no manifest row for source {sorted(unknown)}")
        tiers = set(df.loc[named, "tier"])
        if "EXCLUDED_V1" in tiers:
            raise Refused(f"a named source is tier EXCLUDED_V1, never loaded: {sorted(s)}")
        if not allow_eval and ("EVAL" in tiers or df.loc[named, "cody"].any()):
            raise Refused(f"a named source is evaluation-only: {sorted(s)} (pass allow_eval=True in an "
                          "evaluation stage)")
        sel &= named
    for col, want in wanted.items():
        if want is not None:
            sel &= df[col].isin(_as_set(want))
    sel &= df["tier"] != "EXCLUDED_V1"
    if not allow_eval:
        sel &= (df["tier"] != "EVAL") & ~df["cody"]
    if not allow_lockbox:
        sel &= ~df["lockbox"]
    rows = df[sel].sort_values(["container", "record"], kind="stable")
    # a record listed by two manifest rows (cervical_motion_multiplanar: its .npz and its .pkl path)
    # is read once, with the first of its rows in manifest order
    return rows[~rows.duplicated(["container", "record"])]


def iterate(modality=None, tier=None, lane=None, sources=None, allow_eval=False, allow_lockbox=False):
    """Generator of (row, record) pairs over the rows the filters select and the flags allow.

    Filters take a value or a collection (sources match source or manifest_source); a value absent from
    the index raises ValueError (LookupError for a source). Refused and unresolved rows are skipped, but
    naming what the flags refuse raises Refused at the call: tier EXCLUDED_V1 or a source of that tier
    always, tier EVAL or a CODY or EVAL source without allow_eval. A record that two manifest rows list
    is yielded once. Rows come grouped by container, in record order; each container is opened once, its
    meta.json parsed once, and records are read as they are consumed. lane=None includes lane 'none'
    (seizeit2 EMG, outside the body model): a training stage passes its lane."""
    rows = _select(modality, tier, lane, sources, allow_eval, allow_lockbox)
    return _generate(rows, allow_eval, allow_lockbox)


def _generate(rows, allow_eval, allow_lockbox):
    for container, group in rows.groupby("container", sort=False):
        with zipfile.ZipFile(container_path(container)) as zf:
            for row in group.to_dict("records"):
                row = check(row, allow_eval, allow_lockbox)
                yield row, _read_row(zf, row)
