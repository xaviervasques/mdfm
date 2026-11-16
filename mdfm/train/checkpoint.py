"""Checkpoints of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.6 and section 5, task
B4.17).

save(path, model, ema, opt, sched, sampler_state, rng, cfg, provenance) writes one file, a torch.save of
plain containers and tensors only (loadable with weights_only=True): format, step, the state dicts of the
model, the EMA, the optimizer and the scheduler, the sampler state, the RNG states (rng_state), the config as
canonical JSON with its SHA-256, the provenance (train/provenance.record) and the SHA-256 of each person
list. Before anything is written:
- provenance.assert_clean on the training person lists and on the held-out lists being disjoint from them
  (a lockbox, EVAL, CODY, canary or held-out person refuses the write);
- the stored list hashes are recomputed from the lists;
- provenance.audit on the model, the EMA and the optimizer state (SMPL+H content refuses the write).
The file is written beside itself (<path>.tmp) and renamed, with <path>.sha256 holding its SHA-256; the run
folder (the file's folder) receives provenance.json, config.json, config.sha256 and persons/<source>.txt (the
sorted ids of each training list, one per line). A path inside MoveAtlas, inside the gates-only store or
inside this tree outside cache/ and runs/ (both gitignored) is refused.

load(path) checks <path>.sha256 when present, reads the file with torch.load(weights_only=True) and re-checks:
the config hash, the person-list hashes, the current records and lockbox hashes of the reports against those
the checkpoint recorded (a mismatch raises ProvenanceError: the lockbox or the tiers changed since the run),
assert_clean on the stored lists under the current deny lists, and the audit of the three state dicts. Hashes
that may move without invalidating a checkpoint (code, rules, MoveAtlas HEAD, the records file, the body-store
and window-store hashes their reports hold, config of the run against the current file) are compared and
listed under 'drift'; a rebuilt store also moves the sampler's data_sha256, which refuses the resume of its
state. restore() puts the states back into live objects.

Tier (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.3, task S2.4). save takes the run's tier from cfg
(provenance.run_tier: a RunConfig's tier field, a config dict's 'tier' key, RELEASE without one) and stores
provenance.tiered(provenance, tier): the record with its tier and the licence rows of its training sources (the
licence matrix and its SHA-256), which provenance.json repeats. A RELEASE save is refused when the record or a
lineage entry says INTERNAL (a migration of an INTERNAL checkpoint under a RELEASE config) and when a training
person is INTERNAL or listed under a source holding an INTERNAL row (assert_clean at the tier). load reads the
checkpoint's tier (provenance.tier_of: no field reads as RELEASE, so every checkpoint written before S2.4 loads as
RELEASE), checks its lists at that tier and returns it as ck['tier']; with tier='RELEASE', or a cfg whose tier is
RELEASE when tier is not given, an INTERNAL checkpoint is refused (ProvenanceError). train/loop.py resumes and
inits with load(path, cfg=self.cfg), so a RELEASE run refuses to resume or init from an INTERNAL checkpoint.

Records drift (task T16): the lockbox hash is never relaxed. A moved records_sha256 passes only under the
narrow policy of provenance.records_drift: the tier hash history holds the checkpoint's records state and
every changed row lies in a source outside the run's training and held-out lists and changes only fps (no
row added or removed, no tier column changed). It is then listed first under 'drift', its evidence sits in
the checkpoint dict under drift_evidence['records_sha256'], and the load is logged: a warning of the logger
mdfm.train.checkpoint and one JSON line appended to drift_log.jsonl beside the checkpoint (or the drift_log
path given). Any other records change raises ProvenanceError with every problem named.

Archived records state (decision of 2026-11-12, plan step M7): load(..., records_state=<records_sha256>) reads the
checkpoint under that archived records state (provenance.hashes_at, archived_state) instead of the live one: the
records hashes it compares with are those of the archived file, checked against its archive record and the tier
hash history; the lockbox and the person guard stay live. A checkpoint born on the named state loads with no records
drift; one born on another state passes only what records_drift tolerates between the two. The state is returned
under ck['records_state'] and the load is logged (a warning of the logger). The v1 checkpoints are read this way for
the port's comparison, each under the state it was born on; without records_state a load compares with the live
records, as before, and refuses a change the policy does not tolerate.
"""
import datetime
import json
import logging
import os
import random

import torch

from .. import paths
from . import provenance as PV

FORMAT = "mdfm-checkpoint/1"
STRICT = ("records_sha256", "lockbox_sha256")
NEVER_RELAXED = ("lockbox_sha256",)          # records_sha256 moves to drift only under PV.records_drift
DRIFT = ("code_sha256", "rules_sha256", "moveatlas_head", "records_file_sha256", "body_store_report",
         "window_store_report")
DRIFT_LOG = "drift_log.jsonl"
LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------- paths and RNG
def check_path(path):
    """The absolute path of a checkpoint file, refused (PermissionError) inside MoveAtlas, the gates-only store
    or this tree outside cache/ and runs/."""
    p = os.path.abspath(str(path))
    paths.assert_not_moveatlas(p)
    gated = (paths.GATES_STORE, paths.GATES_STORE_TARGET, os.path.join(paths.CACHE, "gates_only"))
    if any(paths._inside(p, g) for g in gated):
        raise PermissionError(f"a checkpoint may not sit inside the gates-only store: {path}")
    if paths._inside(p, paths.TREE) and not (paths._inside(p, paths.CACHE)
                                             or paths._inside(p, os.path.join(paths.TREE, "runs"))):
        raise PermissionError(f"a checkpoint may sit in this tree only under cache/ or runs/ (gitignored): {path}")
    return p


def rng_state():
    """RNG states: torch (CPU), every CUDA device, Python's random (the sampler and collate carry their own
    numpy generators)."""
    cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {"torch": torch.get_rng_state(), "cuda": cuda, "python": random.getstate()}


def set_rng_state(state):
    """Restore rng_state()."""
    torch.set_rng_state(state["torch"])
    for i, s in enumerate(state.get("cuda") or []):
        if i < torch.cuda.device_count():
            torch.cuda.set_rng_state(s, i)
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))


def _config_json(cfg):
    if isinstance(cfg, str):
        return cfg
    if isinstance(cfg, dict):
        return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    from .config import canonical_json
    return canonical_json(cfg)


# ---------------------------------------------------------------- save
def _check_lists(prov, tier=PV.RELEASE):
    persons = prov["persons"]
    got = PV.list_hashes(persons)
    if prov.get("persons_sha256") not in (None, got):
        raise PV.ProvenanceError("the stored person-list hashes differ from the lists")
    PV.assert_clean(persons, tier=tier)
    held = {p for ids in (prov.get("held_out") or {}).values() for p in ids}
    both = sorted({p for ids in persons.values() for p in ids} & held)
    if both:
        raise PV.ProvenanceError(f"{len(both)} training persons are held out: {both[:10]}")
    return got


def save(path, model, ema, opt, sched, sampler_state, rng, cfg, provenance, step=0):
    """Write a checkpoint (module docstring); returns its SHA-256. model, ema, opt and sched: modules,
    optimizers or schedulers (their state_dict() is taken) or state dicts (ema and sched may be None);
    sampler_state: MixtureSampler.state_dict() (or None); rng: rng_state(); cfg: a RunConfig, a dict or its
    canonical JSON; provenance: provenance.record(...), stored as provenance.tiered(provenance, the config's
    tier) (module docstring, Tier)."""
    p = check_path(path)
    cfg_json = _config_json(cfg)
    tier = PV.run_tier(cfg_json)
    provenance = PV.tiered(provenance, tier)
    list_sha = _check_lists(provenance, tier)
    states = PV.audit(model, ema, opt)
    payload = {"format": FORMAT, "step": int(step), "model": states["model"], "ema": states["ema"],
               "opt": states["opt"], "sched": PV.state_of(sched), "sampler": sampler_state, "rng": rng,
               "config": cfg_json, "config_sha256": PV.sha256_bytes(cfg_json.encode("utf-8")),
               "provenance": json.loads(json.dumps(provenance)), "persons_sha256": list_sha}
    folder = os.path.dirname(p)
    os.makedirs(folder, exist_ok=True)
    tmp = p + ".tmp"
    torch.save(payload, tmp)
    sha = PV.sha256_file(tmp)
    os.replace(tmp, p)
    _write(p + ".sha256", f"{sha}  {os.path.basename(p)}\n")
    _write(os.path.join(folder, "provenance.json"), json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    _write(os.path.join(folder, "config.json"), cfg_json + "\n")
    _write(os.path.join(folder, "config.sha256"), payload["config_sha256"] + "\n")
    os.makedirs(os.path.join(folder, "persons"), exist_ok=True)
    for s, ids in provenance["persons"].items():
        _write(os.path.join(folder, "persons", f"{s}.txt"), "".join(f"{x}\n" for x in ids))
    return sha


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- load
def load(path, map_location="cpu", cfg=None, drift_log=None, tier=None, records_state=None):
    """The checkpoint dict (module docstring) with 'drift', the list of hashes that moved since it was saved,
    'drift_evidence' ({} or {'records_sha256': evidence}) and 'tier' (provenance.tier_of: RELEASE without a
    field); cfg: the current RunConfig or dict, compared with the stored config (drift); drift_log: the JSON-lines
    file a tolerated records drift is appended to (default drift_log.jsonl beside the checkpoint); tier: the tier
    of the run that loads it (default: cfg's tier when cfg is given, else none asked), a RELEASE run refusing an
    INTERNAL checkpoint; records_state: None (the live records) or the records_sha256 of an archived records state
    the checkpoint is read under (module docstring), returned as ck['records_state'] ({} without one). Raises
    ProvenanceError on a file, config, person-list or lockbox hash that does not
    match, a records change the drift policy refuses or a tier refusal, canonical.Refused on a person the current
    deny lists refuse at the checkpoint's tier."""
    want_tier = PV.check_tier(tier) if tier is not None else (PV.run_tier(_config_json(cfg)) if cfg is not None
                                                             else None)
    p = os.path.abspath(str(path))
    side = p + ".sha256"
    got = None
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            want = f.read().split()[0]
        got = PV.sha256_file(p)
        if got != want:
            raise PV.ProvenanceError(f"{p}: SHA-256 {got}, {side} says {want}")
    ck = torch.load(p, map_location=map_location, weights_only=True)
    if ck.get("format") != FORMAT:
        raise PV.ProvenanceError(f"{p}: format {ck.get('format')!r}, expected {FORMAT}")
    if PV.sha256_bytes(ck["config"].encode("utf-8")) != ck["config_sha256"]:
        raise PV.ProvenanceError(f"{p}: the stored config does not hash to its config_sha256")
    prov = ck["provenance"]
    ck_tier = PV.tier_of(prov)
    if want_tier == PV.RELEASE and ck_tier != PV.RELEASE:
        raise PV.ProvenanceError(f"{p} is an {ck_tier} checkpoint: a RELEASE run never loads it (init, resume or "
                                 "migration)")
    if PV.list_hashes(prov["persons"]) != ck["persons_sha256"]:
        raise PV.ProvenanceError(f"{p}: the stored person lists do not hash to the stored hashes")
    then = prov["hashes"]
    now = PV.hashes()
    arch = None
    if records_state is not None:
        now, arch = PV.hashes_at(records_state, now)
    bad = [k for k in NEVER_RELAXED if then.get(k) != now.get(k)]
    if bad:
        raise PV.ProvenanceError(f"{p}: {bad} changed since the checkpoint was written "
                                 f"({[(then.get(k), now.get(k)) for k in bad]})")
    evidence = None
    if then.get("records_sha256") != now.get("records_sha256"):
        try:
            evidence = PV.records_drift(prov, now, current=None if arch is None else arch["entry"])
        except PV.ProvenanceError as e:
            err = PV.ProvenanceError(f"{p}: ['records_sha256'] changed since the checkpoint was written "
                                     f"({then.get('records_sha256')}, {now.get('records_sha256')}) and the drift "
                                     f"policy refuses it: {e}")
            err.problems = getattr(e, "problems", [str(e)])
            raise err from e
    _check_lists(prov, ck_tier)
    PV.audit(ck["model"], ck["ema"], ck["opt"])
    drift = ([] if evidence is None else ["records_sha256"]) + [k for k in DRIFT if then.get(k) != now.get(k)]
    if cfg is not None and PV.sha256_bytes(_config_json(cfg).encode("utf-8")) != ck["config_sha256"]:
        drift.append("config_sha256")
    ck["drift"] = drift
    ck["tier"] = ck_tier
    ck["drift_evidence"] = {} if evidence is None else {"records_sha256": evidence}
    ck["records_state"] = {} if arch is None else {k: v for k, v in arch.items() if k != "entry"}
    if arch is not None:
        LOG.warning("%s read under the archived records state %s (%s), not the live records", p,
                    arch["records_sha256"], arch["path"])
    if evidence is not None:
        _log_drift(p, got, ck, evidence, drift_log)
    return ck


def _log_drift(p, sha, ck, evidence, drift_log):
    """Log a tolerated records drift: a warning and one JSON line appended to drift_log (default
    drift_log.jsonl beside the checkpoint), never inside MoveAtlas."""
    target = os.path.join(os.path.dirname(p), DRIFT_LOG) if drift_log is None else os.path.abspath(str(drift_log))
    paths.assert_not_moveatlas(target)
    rec = {"event": "records_drift", "time": datetime.datetime.now().isoformat(timespec="seconds"),
           "checkpoint": p.replace("\\", "/"), "checkpoint_sha256": sha, "step": int(ck.get("step", 0)),
           "evidence": evidence}
    a, b = evidence["records_sha256"]
    what = " | ".join(_change_text(c) for c in evidence["changes"])
    LOG.warning("%s loaded under a tolerated records drift %s -> %s: %s (logged in %s)", p, a, b,
                what or "no row changed", target)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _counts(d):
    """Value counts as text, '(empty) 3498, 30.0 403'; 'none' when there are none."""
    return ", ".join(f"{'(empty)' if k == '' else k} {v}" for k, v in (d or {}).items()) or "none"


def _change_text(c):
    """One tolerated change as text: the source with its total row count, then per changed column its value
    counts before -> after. The history keeps value counts, not per-row digests, so the number of changed rows
    is not printed as such: the counts bound it."""
    vals = c.get("values") or {}
    cols = "; ".join(f"{col} {_counts(vals[col]['before'])} -> {_counts(vals[col]['after'])}" if col in vals
                     else col for col in c["columns"])
    return f"{c['source']} (source of {c['rows'][1]} rows): {cols}"


def restore(ck, model=None, ema=None, opt=None, sched=None, sampler=None, rng=True):
    """Load the states of a checkpoint dict into the objects given (each optional) and, with rng, the RNG
    states; returns the step."""
    for obj, key in ((model, "model"), (ema, "ema"), (opt, "opt"), (sched, "sched")):
        if obj is not None:
            if ck[key] is None:
                raise ValueError(f"the checkpoint holds no {key} state")
            obj.load_state_dict(ck[key])
    if sampler is not None:
        sampler.load_state_dict(ck["sampler"])
    if rng:
        set_rng_state(ck["rng"])
    return int(ck["step"])
