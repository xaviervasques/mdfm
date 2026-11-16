"""Gate commons (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 2.7): statistics, verdicts, rule
registration, the lockbox read log and the result writers of every gate harness.

Statistics. A gate line is scored per window; windows are averaged per person with their weights
(frames), and the statistic is the unweighted mean over persons (person_weighted_mean), so a person
with 1,000 windows counts once. Intervals: person-clustered percentile bootstrap (cluster_bootstrap:
persons resampled with replacement, BOOT_N resamples, the seed of the gate's rule file). The verdict
fires on the point estimate (verdict): kill beyond the kill line, pass inside the pass line when every
extra condition holds, undetermined otherwise, not evaluated when there is no point.

Rules. A rule file (docs/rules/<GATE>_RULE.md) starts DRAFT. After reading it the owner changes its
status line to "Status: APPROVED by the owner on YYYY-MM-DD." and registers it (register_rule, which
refuses anything else and a gate already read on the lockbox); docs/rules/gate_rule_hashes.csv then
holds its SHA-256. rule() returns the text and hash of a registered file only, so an edited registered
rule is refused. A session never registers a rule.

Lockbox read log (reports/gates/lockbox_reads.csv). log_lockbox_read appends one row before any
lockbox array is read: the gate, the checkpoint SHA-256, the rule SHA-256 (registered for that gate),
the input-cache and code hashes. A second read of the same gate, checkpoint and rule is refused unless
reread=True comes with a non-empty reason, which is logged. Refusals raise mdfm.data.canonical.Refused.

EVAL read guard and log (reports/gates/eval_reads.csv). A source whose tier is EVAL in
reports/manifest_tiers/tiers_by_source.csv is read by a gate only under that gate's registered rule
(docs/rules/DESCRIPTOR_RULES_DRAFT.md DR-0.3; docs/MFM_DESIGN_NOTE_CLINICAL_QUESTIONS_2026-09-29.md,
priority 8). guard_eval_read refuses it while the rule file is not registered for the gate, was edited
since its registration, or is registered for another gate (rule); it also refuses a source the tier
table does not list, an EXCLUDED_V1 source, and any CODY source, whatever its tier: CODY is opened only
by the sealed script under its own read log (docs/rules/CODY_CONSENSUS_RULE_DRAFT.md R2.8, R9), never
through this guard. A source of another tier passes with no rule and is not logged here.
log_eval_read runs the guard, then appends one row per allowed EVAL read before any EVAL array opens
(gate, source, tier, rule path and SHA-256, the checkpoint, input and code hashes when given, the
SHA-256 of the tier table). The log file is created by the first allowed read, never by a refusal or a
guard call. An EVAL set is evaluation-only, not sealed: a second read is logged, not refused.

Clinical-question tags (docs/MFM_DESIGN_NOTE_CLINICAL_QUESTIONS_2026-09-29.md, priority 3 and section
4.7). A GateResult may carry extra["clinical_questions"] (CQ_FIELD), the list of the questions i to xiv
of docs/feedback/CLINICAL_QUESTIONS_NEUROLOGIST_2026-09-28.md that the line addresses, each with how far
(answered, partly, not addressable) and its hard-case coverage (cq_tag). The field is optional and off by
default: a result built without it is left as it was (no key is added, so the G1, G3 and G4 outputs do
not change) and reads as an empty tag list (question_tags), as does a result written before the field
existed, so no logged result needs rewriting. Tags passed in extra are checked when the result is built.
An absent key means untagged; a present empty list means tagged as addressing no question. No harness
passes tags until the owner and the neurologist approve a tag table (draft:
docs/proposals/CQ_TAGS_DRAFT.md, awaiting the neurologist); no code reads that draft.

Writers assert that no em-dash (U+2014) reaches a report.
"""
import csv
import datetime
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import paths
from ..data.canonical import Refused, is_cody

TREE = paths.TREE
GATES_DIR = os.path.join(TREE, "reports", "gates")
READ_LOG = os.path.join(GATES_DIR, "lockbox_reads.csv")
EVAL_LOG = os.path.join(GATES_DIR, "eval_reads.csv")
TIERS = os.path.join(TREE, "reports", "manifest_tiers", "tiers_by_source.csv")
RULE_HASHES = os.path.join(TREE, "docs", "rules", "gate_rule_hashes.csv")
READ_LOG_COLUMNS = ("read_at", "gate", "checkpoint_sha256", "rule_sha256", "inputs_sha256", "code_sha256",
                    "reread", "reason", "note")
EVAL_LOG_COLUMNS = ("read_at", "gate", "source", "tier", "rule_path", "rule_sha256", "checkpoint_sha256",
                    "inputs_sha256", "code_sha256", "tiers_sha256", "note")
CQ_FIELD = "clinical_questions"
CQ_IDS = ("i", "ii", "iii", "iv", "v", "v(a)", "v(b)", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv")
CQ_VERDICTS = ("answered", "partly", "not addressable")
CQ_TAG_KEYS = ("question", "verdict", "hard_case")
RULE_COLUMNS = ("gate", "version", "path", "sha256", "approved_by", "approved_on", "registered_at", "note")
APPROVED = re.compile(r"^Status: APPROVED by the owner on (\d{4}-\d{2}-\d{2})\.?\s*$", re.M)
STATUS = re.compile(r"^Status: (.*)$", re.M)
EM_DASH = "\u2014"
VERDICTS = ("pass", "kill", "undetermined", "not evaluated")
BOOT_N = 2000
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------- hashes
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    """JSON text with sorted keys and no spaces (the text every config hash is taken over)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def sha256_json(obj):
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def rel(path):
    """Path relative to the tree, forward slashes (absolute when outside the tree)."""
    a = os.path.abspath(path)
    try:
        r = os.path.relpath(a, TREE)
    except ValueError:                       # another drive
        return a.replace("\\", "/")
    return a.replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def code_sha256(files):
    """SHA-256 over the sorted (relative path, file SHA-256) pairs of `files`: the code hash of a harness."""
    rows = sorted((rel(p), sha256_file(p)) for p in files)
    return sha256_bytes("".join(f"{r}\t{h}\n" for r, h in rows).encode("utf-8"))


# ---------------------------------------------------------------- statistics
def person_means(values, persons, weights=None):
    """Per-person weighted means: values (N,) or (N, k), persons (N,), weights (N,) >= 0 (default 1) ->
    (ids (P,) sorted, means (P,) or (P, k), weight sums (P,)). Non-finite values or a person whose
    weights sum to 0 raise ValueError."""
    v = np.asarray(values, dtype=np.float64)
    p = np.asarray(persons).astype(str)
    if v.shape[0] != p.shape[0] or v.shape[0] == 0:
        raise ValueError(f"{v.shape[0]} values for {p.shape[0]} persons")
    if not np.isfinite(v).all():
        raise ValueError("non-finite value in a person mean")
    w = np.ones(v.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (v.shape[0],) or (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite, non-negative, one per value")
    ids, inv = np.unique(p, return_inverse=True)
    ws = np.bincount(inv, weights=w, minlength=len(ids))
    if (ws <= 0).any():
        raise ValueError(f"persons without weight: {ids[ws <= 0][:3].tolist()}")
    wv = v * w.reshape((-1,) + (1,) * (v.ndim - 1))
    sums = np.zeros((len(ids),) + v.shape[1:])
    np.add.at(sums, inv, wv)
    return ids, sums / ws.reshape((-1,) + (1,) * (v.ndim - 1)), ws


def person_weighted_mean(values, persons, weights=None):
    """Mean over persons of the per-person weighted means (person_means): a float for (N,) values, an
    array (k,) for (N, k)."""
    _, m, _ = person_means(values, persons, weights)
    out = m.mean(axis=0)
    return float(out) if np.ndim(out) == 0 else out


def cluster_bootstrap(values, persons, n=BOOT_N, seed=0, weights=None, level=0.95, stat=None):
    """Person-clustered percentile bootstrap: persons resampled with replacement n times, each
    resample's statistic the mean of its person means (person_means), mapped by stat ((n, ...) ->
    (n,), for example a ratio of two columns; default identity on (N,) values). Returns (lo, hi) at
    the level's central percentiles."""
    _, m, _ = person_means(values, persons, weights)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.shape[0], size=(int(n), m.shape[0]))
    boot = m[idx].mean(axis=1)
    if stat is not None:
        boot = np.asarray(stat(boot), dtype=np.float64)
    if boot.ndim != 1:
        raise ValueError(f"the bootstrap statistic must be one number per resample, got shape {boot.shape}")
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return float(lo), float(hi)


def verdict(point, pass_line, kill_line, higher_is_better=False, conditions=()):
    """'kill' beyond the kill line (above it, or below it when higher is better); 'pass' within the pass
    line when every extra condition is True (a None condition, not evaluable, blocks the pass);
    'undetermined' otherwise; 'not evaluated' when the point is None or not finite."""
    if point is None or not math.isfinite(float(point)):
        return "not evaluated"
    p = float(point)
    if (p < kill_line) if higher_is_better else (p > kill_line):
        return "kill"
    inside = (p >= pass_line) if higher_is_better else (p <= pass_line)
    if inside and all(c is True for c in conditions):
        return "pass"
    return "undetermined"


# ---------------------------------------------------------------- clinical-question tags
def check_cq_tags(tags):
    """The clinical-question tags of a result, checked (module docstring): a list or tuple of dicts with
    exactly the keys question (one of CQ_IDS), verdict (one of CQ_VERDICTS) and hard_case (non-empty
    text), no question twice. Returns new plain dicts in the given order; [] for None or an empty list.
    ValueError otherwise. The vocabulary is that of the questions file; which tag a line may carry is
    the tag table's business (docs/proposals/CQ_TAGS_DRAFT.md, a draft no code reads)."""
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)):
        raise ValueError(f"{CQ_FIELD} must be a list of tags, got {type(tags).__name__}")
    out, seen = [], set()
    for t in tags:
        if not isinstance(t, dict) or set(t) != set(CQ_TAG_KEYS):
            raise ValueError(f"a {CQ_FIELD} tag is a dict with the keys {list(CQ_TAG_KEYS)}, got {t!r}")
        q, v, h = (t[k] if isinstance(t[k], str) else None for k in CQ_TAG_KEYS)
        if q not in CQ_IDS:
            raise ValueError(f"clinical question {t['question']!r} not in {CQ_IDS}")
        if v not in CQ_VERDICTS:
            raise ValueError(f"verdict {t['verdict']!r} of question {q} not in {CQ_VERDICTS}")
        if h is None or not h.strip():
            raise ValueError(f"question {q}: the hard-case coverage is non-empty text")
        if q in seen:
            raise ValueError(f"question {q} tagged twice")
        assert_no_em_dash(h, f"hard-case coverage of question {q}")
        seen.add(q)
        out.append({"question": q, "verdict": v, "hard_case": h.strip()})
    return out


def cq_tag(question, verdict, hard_case):
    """One clinical-question tag {question, verdict, hard_case}, checked (check_cq_tags)."""
    return check_cq_tags([{"question": question, "verdict": verdict, "hard_case": hard_case}])[0]


def question_tags(result):
    """The clinical-question tags of a GateResult or of its dict (to_dict, or a result JSON row): [] when
    the extra dict or its CQ_FIELD key is absent (an untagged result, or one written before the field
    existed) and when the key holds an empty list. `CQ_FIELD in extra` tells the two apart."""
    extra = result.extra if isinstance(result, GateResult) else (result or {}).get("extra")
    return check_cq_tags((extra or {}).get(CQ_FIELD))


@dataclass
class GateResult:
    """One line of a gate (spec 2.7): point and person-clustered interval in `unit`, the pass and kill
    lines, the verdict on the point, the trivial baseline on the same windows, and the hashes that tie
    the number to its rule, checkpoint and training persons. When the caller's extra holds CQ_FIELD, the
    clinical-question tags, they are checked (check_cq_tags) into a copy of extra, so the caller's dict is
    never changed; without the key, extra is kept as given and no key is added (module docstring)."""
    gate: str
    line: str
    point: float
    interval: tuple
    pass_line: float
    kill_line: float
    verdict: str
    trivial_baseline: float
    rule_sha256: str
    checkpoint_sha256: str
    train_persons_sha256: str
    n_persons: int
    unit: str = ""
    higher_is_better: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not in {VERDICTS}")
        if isinstance(self.extra, dict) and CQ_FIELD in self.extra:     # off by default: no key, no change
            self.extra = dict(self.extra)
            self.extra[CQ_FIELD] = check_cq_tags(self.extra[CQ_FIELD])

    def to_dict(self):
        d = asdict(self)
        d["interval"] = list(self.interval) if self.interval is not None else None
        return d


# ---------------------------------------------------------------- rule files
def _read_rows(path, columns):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if tuple(r.fieldnames or ()) != tuple(columns):
            raise ValueError(f"{path}: header {r.fieldnames}, expected {list(columns)}")
        return list(r)


def _append_row(path, columns, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(paths.assert_not_moveatlas(path), "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in columns})


def ensure_csv(path, columns):
    """Create `path` with its header when absent; check the header when present. Returns the path."""
    if os.path.exists(path):
        _read_rows(path, columns)
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(paths.assert_not_moveatlas(path), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow(columns)
    return path


def registry(path=RULE_HASHES):
    """Rows of the rule-hash registry (RULE_COLUMNS)."""
    return _read_rows(path, RULE_COLUMNS)


def rule_status(text):
    """('approved', 'YYYY-MM-DD') when the status line (the first line that starts with 'Status: ') reads
    'Status: APPROVED by the owner on DATE.', else (the raw status or 'missing', None). A later line of
    the body never overrides the status line."""
    s = STATUS.search(text)
    if not s:
        return "missing", None
    m = APPROVED.match(s.group(0))
    if m:
        return "approved", m.group(1)
    return s.group(1).strip(), None


def registered(gate, sha256, path=RULE_HASHES):
    """True when `sha256` is a registered hash of `gate`."""
    return any(r["gate"] == gate and r["sha256"] == sha256 for r in registry(path))


def rule(path, gate=None, registry_path=RULE_HASHES):
    """(text, SHA-256) of a registered rule file; Refused when the file is not registered, was edited
    since its registration, or is registered for another gate."""
    with open(path, "rb") as f:
        raw = f.read()
    sha = sha256_bytes(raw)
    rows = [r for r in registry(registry_path) if r["path"] == rel(path)]
    if not rows:
        raise Refused(f"{rel(path)} is not registered in {rel(registry_path)}: the owner reads and approves a "
                      "rule before any lockbox read")
    hit = [r for r in rows if r["sha256"] == sha]
    if not hit:
        raise Refused(f"{rel(path)} changed since its registration (SHA-256 {sha[:12]}, registered "
                      f"{', '.join(r['sha256'][:12] for r in rows)})")
    if gate is not None and all(r["gate"] != gate for r in hit):
        raise Refused(f"{rel(path)} is registered for {sorted({r['gate'] for r in hit})}, not {gate}")
    return raw.decode("utf-8"), sha


def register_rule(gate, path, approved_on, approved_by="owner", note="", registry_path=RULE_HASHES,
                  log_path=READ_LOG):
    """Record the SHA-256 of an approved rule file (the owner's act): refused unless approved_by is
    'owner' and the file's status line reads 'Status: APPROVED by the owner on <approved_on>.', and
    refused for a gate that has a lockbox read logged (a gate is frozen after its first read). An
    already registered hash returns its row. Returns the registry row."""
    if approved_by != "owner":
        raise Refused(f"a rule is approved by the owner only, not {approved_by!r}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(approved_on)):
        raise ValueError(f"approved_on must be YYYY-MM-DD, got {approved_on!r}")
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    status, date = rule_status(text)
    if status != "approved" or date != approved_on:
        raise Refused(f"{rel(path)} status is {status!r} (date {date}); the owner sets 'Status: APPROVED by the owner "
                      f"on {approved_on}.' before registering it")
    if EM_DASH in text:
        raise ValueError(f"{rel(path)} holds an em-dash")
    sha = sha256_bytes(raw)
    rows = registry(registry_path)
    for r in rows:
        if r["gate"] == gate and r["sha256"] == sha:
            return r
    if any(r["gate"] == gate for r in read_log(log_path)):
        raise Refused(f"{gate} has a lockbox read logged: its rule is frozen")
    row = {"gate": gate, "version": str(1 + sum(r["gate"] == gate for r in rows)), "path": rel(path), "sha256": sha,
           "approved_by": approved_by, "approved_on": approved_on,
           "registered_at": datetime.datetime.now().isoformat(timespec="seconds"), "note": note}
    _append_row(registry_path, RULE_COLUMNS, row)
    return row


# ---------------------------------------------------------------- lockbox read log
def ensure_read_log(path=READ_LOG):
    """reports/gates/lockbox_reads.csv with its header (created when absent)."""
    return ensure_csv(path, READ_LOG_COLUMNS)


def read_log(path=READ_LOG):
    """Rows of the lockbox read log (READ_LOG_COLUMNS)."""
    return _read_rows(path, READ_LOG_COLUMNS)


def log_lockbox_read(gate, checkpoint_sha256, rule_sha256, reread=False, reason="", inputs_sha256="",
                     code_sha256="", note="", log_path=READ_LOG, registry_path=RULE_HASHES):
    """Append the row of one lockbox read, before the read (module docstring). Refused when the rule
    hash is not registered for the gate, and for a second read of the same gate, checkpoint and rule
    without reread=True and a non-empty reason. Returns the row."""
    for name, h in (("checkpoint_sha256", checkpoint_sha256), ("rule_sha256", rule_sha256)):
        if not _HEX64.match(str(h)):
            raise ValueError(f"{name} must be a lowercase hex SHA-256, got {h!r}")
    if not registered(gate, rule_sha256, registry_path):
        raise Refused(f"rule {rule_sha256[:12]} is not registered for {gate}: no lockbox read")
    ensure_read_log(log_path)
    prior = [r for r in read_log(log_path) if (r["gate"], r["checkpoint_sha256"], r["rule_sha256"])
             == (gate, checkpoint_sha256, rule_sha256)]
    if prior and not reread:
        raise Refused(f"{gate} was read on the lockbox with checkpoint {checkpoint_sha256[:12]} and rule "
                      f"{rule_sha256[:12]} on {prior[0]['read_at']}: a second read needs reread and a reason")
    if reread and not prior:
        raise ValueError("reread asked for a read that was never logged")
    if reread and not str(reason).strip():
        raise Refused("a lockbox reread needs a logged reason")
    row = {"read_at": datetime.datetime.now().isoformat(timespec="seconds"), "gate": gate,
           "checkpoint_sha256": checkpoint_sha256, "rule_sha256": rule_sha256, "inputs_sha256": inputs_sha256,
           "code_sha256": code_sha256, "reread": "1" if reread else "0", "reason": str(reason).strip(), "note": note}
    _append_row(log_path, READ_LOG_COLUMNS, row)
    return row


# ---------------------------------------------------------------- EVAL read guard and log
def source_tiers(source, tiers_path=TIERS):
    """The set of tiers the tier table gives `source` (one row per source and modality); empty when the
    table does not list it. ValueError when the table lacks a source or tier column."""
    with open(tiers_path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not {"source", "tier"} <= set(r.fieldnames or ()):
            raise ValueError(f"{rel(tiers_path)}: header {r.fieldnames}, expected source and tier columns")
        return {row["tier"].strip() for row in r if row["source"].strip() == source}


def guard_eval_read(gate, source, rule_path, registry_path=RULE_HASHES, tiers_path=TIERS):
    """Refused unless `gate` may read `source` (module docstring): a CODY, unlisted or EXCLUDED_V1 source
    always; an EVAL source unless rule_path is registered for the gate and unchanged (rule). Writes
    nothing. Returns {gate, source, tier, eval, rule_path, rule_sha256}; eval is False and rule_sha256
    empty for a source of another tier, which needs no rule."""
    gate, source = str(gate).strip(), str(source).strip()
    if not gate or not source:
        raise ValueError("a gate and a source are named for every read")
    if is_cody(source):
        raise Refused(f"{source} is CODY: it is opened only by the sealed script under its own read log "
                      "(docs/rules/CODY_CONSENSUS_RULE_DRAFT.md R2.8, R9), never through the EVAL read guard")
    tiers = source_tiers(source, tiers_path)
    if not tiers:
        raise Refused(f"{source} is not in {rel(tiers_path)}: its tier is unknown, no read")
    if "EXCLUDED_V1" in tiers:
        raise Refused(f"{source} is tier EXCLUDED_V1: never read")
    out = {"gate": gate, "source": source, "tier": "+".join(sorted(tiers)), "eval": "EVAL" in tiers,
           "rule_path": rel(rule_path) if rule_path else "", "rule_sha256": ""}
    if not out["eval"]:
        return out
    out["tier"] = "EVAL"                     # a source with an EVAL row is read as EVAL in every modality
    if not rule_path or not os.path.isfile(rule_path):
        raise Refused(f"{source} is tier EVAL: {gate} reads it only under its registered rule, and the rule file "
                      f"{rule_path!r} does not exist")
    try:
        _, out["rule_sha256"] = rule(rule_path, gate=gate, registry_path=registry_path)
    except Refused as e:
        raise Refused(f"{source} is tier EVAL: {gate} reads it only under its registered rule; {e}") from e
    return out


def eval_read_log(path=EVAL_LOG):
    """Rows of the EVAL read log (EVAL_LOG_COLUMNS); [] while no EVAL read was made."""
    return _read_rows(path, EVAL_LOG_COLUMNS)


def log_eval_read(gate, source, rule_path, checkpoint_sha256="", inputs_sha256="", code_sha256="", note="",
                  log_path=EVAL_LOG, registry_path=RULE_HASHES, tiers_path=TIERS):
    """Append the row of one EVAL read, before the read (module docstring): the guard first
    (guard_eval_read; a refusal writes nothing), then the header check of an existing log, then the row.
    The hashes are optional (a fixed-formula read has no checkpoint) and, when given, lowercase hex
    SHA-256. ValueError for a source that is not EVAL: this log records EVAL reads only. Returns the row."""
    for name, h in (("checkpoint_sha256", checkpoint_sha256), ("inputs_sha256", inputs_sha256),
                    ("code_sha256", code_sha256)):
        if h and not _HEX64.match(str(h)):
            raise ValueError(f"{name} must be empty or a lowercase hex SHA-256, got {h!r}")
    assert_no_em_dash(str(note), "EVAL read note")
    g = guard_eval_read(gate, source, rule_path, registry_path=registry_path, tiers_path=tiers_path)
    if not g["eval"]:
        raise ValueError(f"{g['source']} is tier {g['tier']}, not EVAL: the EVAL read log records EVAL reads only")
    eval_read_log(log_path)                  # an existing log with another header is refused before the append
    row = {"read_at": datetime.datetime.now().isoformat(timespec="seconds"), "gate": g["gate"],
           "source": g["source"], "tier": g["tier"], "rule_path": g["rule_path"], "rule_sha256": g["rule_sha256"],
           "checkpoint_sha256": checkpoint_sha256, "inputs_sha256": inputs_sha256, "code_sha256": code_sha256,
           "tiers_sha256": sha256_file(tiers_path), "note": note}
    _append_row(log_path, EVAL_LOG_COLUMNS, row)
    return row


# ---------------------------------------------------------------- writers
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, GateResult):
        return o.to_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def assert_no_em_dash(text, what="text"):
    if EM_DASH in text:
        raise ValueError(f"{what} holds an em-dash (U+2014) at offset {text.index(EM_DASH)}")
    return text


def write_text(path, text):
    """Write UTF-8 text with LF line ends after the em-dash check; returns the path."""
    assert_no_em_dash(text, rel(path))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(paths.assert_not_moveatlas(path), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def write_csv(df, path):
    """A DataFrame to CSV (LF, UTF-8) after the em-dash check; returns the path."""
    return write_text(path, df.to_csv(index=False, lineterminator="\n"))


def write(results, out_dir, name, markdown):
    """<out_dir>/<name>.json (results, indent 1) and <out_dir>/<name>.md (markdown), both checked for
    em-dashes before either is written. Returns the two paths."""
    text = json.dumps(results, indent=1, ensure_ascii=False, default=_default, allow_nan=True) + "\n"
    assert_no_em_dash(text, f"{name}.json")
    assert_no_em_dash(markdown, f"{name}.md")
    return (write_text(os.path.join(out_dir, f"{name}.json"), text),
            write_text(os.path.join(out_dir, f"{name}.md"), markdown))
