"""Checkpoint migration between name tables (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.2, task S2.2).

A model holds one row per name in five tables (mdfm/train/names.py): SourceOffsets (sources), the task embedding
of the motion prior (tasks), the layout, device and camera tables of the tokenizer's NuisanceEmbed (each with its
'none' row at len(vocabulary) and its 'unknown' row at len + 1, model/tokenize.py, _code) and the mount and device
prior buffers of NuisanceHeads (mount families). A strict load fails when a table grows. This module maps a state
written under one NameTables (old) onto another (new) by name.

plan(old, new, task_rows, trained_unknown) -> MigrationPlan, per table the source of every new row:
- ('old', i): old row i keeps its meaning. Every old name keeps its row; a new table that drops, renames or moves
  an old name is refused (NameTables.appended raises ValueError).
- Nuisance tables: the 'none' and 'unknown' rows move to the new len and len + 1, so the special codes keep
  their meaning; a row of a new name starts at zero, or as a copy of the table's 'unknown' row ('copy', i) when
  that field is in trained_unknown: the fields whose 'unknown' code some window the source trained on carried
  (stage1_codes counts them, under the source's names, on the stage-1 window store and the synthetic layouts).
  plan_for derives trained_unknown from those counts: it counts them itself for a STAGE1 source and refuses a
  source of other names without them (only the caller knows the windows such a source trained on, an S2a
  checkpoint migrated to the S2b names for one), or counts made under other names than the source's.
- Tasks: a new task row starts as a copy of the 'unknown' row (task_rows 'copy_unknown', the default) or at zero
  ('zero'). 'unknown' is the trained marginal condition (the condition dropout of prior.py replaces the task by it),
  so a copied row gives a window of a new task, at step 0, the condition the stage-1 model gives 'unknown'.
- Sources: a new row of SourceOffsets starts at zero (the synthetic row stays frozen by its mask).
- Mount families: ('prior', family): a new family's buffers come from the new MountPriors (measured, or the
  generic family's rows, extend_priors); the buffers of the old families in those priors must equal the stored
  ones, otherwise apply refuses.

apply(state_dict, plan, priors, moments=False) maps every name-sized entry of a model state dict (the model, its
EMA, the JEPA target's tokenizer tables jepa.target_tokenizer.*); apply_optimizer maps the per-parameter states
of an AdamW state dict through the parameter order of loop.param_groups (param_names): moments of old rows kept,
of new rows zero (copied rows included), the step state kept. On a window of old names the migrated model
computes what the old model computed, since the new rows are never read there (equivalence measures it).

The new model (build_model): MFM(cfg with its sources, priors with its mount families), with MFMConfig.names set
to the names when MFMConfig has that field (task S2.3, under way when this module was written); without it, the
task embedding and the layout, device and camera tables are resized to the names after construction (same module
order, so the same state-dict order and the same optimizer parameter order), and with it that resize finds every
size right and does nothing. batch_from_windows passes the vocabularies to mfm.batch_from_windows when it takes
them (nuis_fields, the same task). Its windows are coded by nuisance_descriptors(stream, fields), the rule of
tokenize.nuisance_descriptors with the name tables' vocabularies. When mfm.batch_from_windows coded the tokens
under the names itself (STAGE1, or nuis_fields given), its codes must equal those, otherwise batch_from_windows
raises; without nuis_fields (before S2.3) it replaces mfm's STAGE1 codes by them. recode maps codes of the old
tables to the new ones.

migrate(path, ...) reads a checkpoint with checkpoint.load and a drift_log path outside its run folder (a load under
a tolerated records drift appends one line to drift_log; the default drift_log.jsonl beside the checkpoint is never
used), takes the old names from its config (names_of), plans with plan_for, migrates the model, the EMA, the
JEPA target and the optimizer moments, checks a strict load of the new model and the EMA, and writes the migrated
checkpoint with checkpoint.save into a new run folder (absent or empty, outside the source's folder;
checkpoint.check_path). Lineage: the provenance extra holds migrated_from (path, SHA-256), the old and new
name-table SHA-256, the plan and its SHA-256, the code counts behind trained_unknown and the code SHA-256 of the
migration; the persons, held-out lists and hashes are the source's (the weights trained on them). Since the
lineage records the code SHA-256, the migrated file's own SHA-256 changes with the code; the plan SHA-256 and the
equivalence are the reproducible record of a migration. The stored config is the source's config with a 'names' section (NameTables.to_dict); RunConfig reads that
section from task S2.3 on, so until then a migrated checkpoint loads only through model_of.

S2B_DRAFT is the name set the load test migrates to: the stage-1 tables plus the S2a and S2b names of spec sections
4.5, 4.8 and 4.9 and vocabulary version 2. It is a test target, not a decision: the run config of S2.3 and S2.6
carries the names a run uses.
"""
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .. import paths
from ..factory import layouts as LY
from ..factory.imu_render import SITES
from ..kernels.nuisance import MountPriors
from ..model import tokenize as TK
from ..model.heads import NuisanceHeads
from . import names as NM

FORMAT = "mdfm-migration/1"
TASK_ROWS = ("copy_unknown", "zero")
NUIS_KEYS = tuple(f"nuis.{f}" for f in NM.NUIS_TABLES)          # nuis.layout, nuis.device, nuis.camera
TABLE_KEYS = ("sources", "tasks") + NUIS_KEYS + ("mount_families",)
_NUIS_RX = re.compile(r"(?:^|\.)(?:target_)?tokenizer\.nuisance\.tables\.(\d+)\.weight$")
_PRIOR_PREFIX = "heads.mp_"

S2B_DRAFT = NM.NameTables(
    sources=NM.STAGE1.sources + ("pads", "clinical_gait_multipatho", "imu_inthewild_iprognosis",
                                 "physionet_gaitpdb", "gaitndd", "gaitrec", "modys_video"),
    vocabulary_path=NM.VOCABULARY_V2,
    layouts=NM.STAGE1.layouts + ("pads_wrists", "multipatho4", "pocket_acc", "insole8", "insole_total", "plate"),
    devices=NM.STAGE1.devices + ("multipatho4", "pocket_acc"),
    cameras=NM.STAGE1.cameras + ("modys",),
    mount_families=NM.STAGE1.mount_families + ("multipatho4", "pocket_acc"),
)


# ---------------------------------------------------------------- the plan
@dataclass(frozen=True)
class TablePlan:
    """One table: old_rows and new_rows of its state entries along dim 0, rows the source of each new row,
    (kind, arg) with kind 'old' (old row arg), 'copy' (a copy of old row arg), 'zero' or 'prior' (the new
    MountPriors' rows of family arg)."""
    old_rows: int
    new_rows: int
    rows: tuple

    def to_dict(self):
        return {"old_rows": self.old_rows, "new_rows": self.new_rows, "rows": [list(r) for r in self.rows]}

    def code_map(self):
        """{old row: new row} of the rows that keep their meaning."""
        return {arg: j for j, (kind, arg) in enumerate(self.rows) if kind == "old"}

    def is_identity(self):
        return self.old_rows == self.new_rows and all(r == ("old", j) for j, r in enumerate(self.rows))


@dataclass(frozen=True)
class MigrationPlan:
    """The plan of a migration (module docstring): old and new NameTables, task_rows, trained_unknown (the
    nuisance fields whose 'unknown' row stage-1 windows trained) and tables {key: TablePlan} for the keys of
    TABLE_KEYS."""
    old: NM.NameTables
    new: NM.NameTables
    task_rows: str
    trained_unknown: tuple
    tables: dict

    def to_dict(self):
        return {"format": FORMAT, "old": self.old.to_dict(), "new": self.new.to_dict(),
                "old_sha256": self.old.sha256(), "new_sha256": self.new.sha256(), "task_rows": self.task_rows,
                "trained_unknown": list(self.trained_unknown),
                "tables": {k: self.tables[k].to_dict() for k in TABLE_KEYS}}

    def sha256(self):
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def plan(old, new, task_rows="copy_unknown", trained_unknown=()):
    """MigrationPlan of old -> new NameTables (module docstring); ValueError on a dropped, renamed or moved old
    name or an unknown task_rows."""
    if task_rows not in TASK_ROWS:
        raise ValueError(f"task_rows {task_rows!r}, expected one of {TASK_ROWS}")
    bad = sorted(set(trained_unknown) - set(NM.NUIS_TABLES))
    if bad:
        raise ValueError(f"trained_unknown names fields {bad}; the growing nuisance fields are {tuple(NM.NUIS_TABLES)}")
    new.appended(old)                                      # refuses a dropped, renamed or moved old name
    tables = {}
    n_old, n_new = len(old.sources), len(new.sources)
    tables["sources"] = TablePlan(n_old, n_new, tuple(("old", i) for i in range(n_old))
                                  + tuple(("zero", None) for _ in range(n_new - n_old)))
    to, tn = old.tasks(), new.tasks()
    u = to.index("unknown")
    if tn.index("unknown") != u:
        raise ValueError(f"'unknown' moves from task id {u} to {tn.index('unknown')}")
    start = ("copy", u) if task_rows == "copy_unknown" else ("zero", None)
    tables["tasks"] = TablePlan(len(to), len(tn), tuple(("old", i) for i in range(len(to)))
                                + tuple(start for _ in range(len(tn) - len(to))))
    fo, fn = old.nuis_fields(), new.nuis_fields()
    for f in NM.NUIS_TABLES:
        a, b = len(fo[f]), len(fn[f])
        start = ("copy", a + 1) if f in trained_unknown else ("zero", None)
        rows = tuple(("old", j) for j in range(a)) + tuple(start for _ in range(b - a)) + (("old", a), ("old", a + 1))
        tables[f"nuis.{f}"] = TablePlan(a + 2, b + 2, rows)
    mo, mn = old.mount_families, new.mount_families
    tables["mount_families"] = TablePlan(len(mo), len(mn), tuple(("old", i) for i in range(len(mo)))
                                         + tuple(("prior", m) for m in mn[len(mo):]))
    return MigrationPlan(old, new, task_rows, tuple(f for f in NM.NUIS_TABLES if f in trained_unknown), tables)


# ---------------------------------------------------------------- state dicts
def table_of(key):
    """The TABLE_KEYS key of a name-sized state-dict entry (model, EMA or JEPA target), None for the others."""
    if key == "source_offsets.offsets":
        return "sources"
    if key == "prior.task.weight":
        return "tasks"
    m = _NUIS_RX.search(key)
    if m:
        f = TK.FIELD_NAMES[int(m[1])]          # m[1], not m.group(1): the label grep of test_losses.py
        return f"nuis.{f}" if f in NM.NUIS_TABLES else None
    if key.startswith(_PRIOR_PREFIX) and key[len(_PRIOR_PREFIX):] in NuisanceHeads.PRIOR_FIELDS:
        return "mount_families"
    return None


def _rows(x, tp, key, moments=False):
    """x (old_rows, ...) -> (new_rows, ...) under TablePlan tp; moments: copied rows start at zero too."""
    if x.dim() == 0 or x.shape[0] != tp.old_rows:
        raise ValueError(f"{key}: shape {tuple(x.shape)}, the old names give {tp.old_rows} rows")
    out = x.new_zeros((tp.new_rows,) + tuple(x.shape[1:]))
    for j, (kind, arg) in enumerate(tp.rows):
        if kind == "old" or (kind == "copy" and not moments):
            out[j] = x[arg]
        elif kind == "prior":
            raise ValueError(f"{key}: a 'prior' row outside the mount buffers")
    return out


def prior_buffers(priors):
    """{field: tensor} of a MountPriors as NuisanceHeads stores them (float32, bool kept)."""
    return {n: (x if x.dtype == torch.bool else x.float()) for n in NuisanceHeads.PRIOR_FIELDS
            for x in [getattr(priors, n).detach().cpu()]}


def priors_of_state(state, families):
    """MountPriors of the mount buffers of a model state dict (heads.mp_*), families named."""
    kw = {n: state[_PRIOR_PREFIX + n] for n in NuisanceHeads.PRIOR_FIELDS}
    return MountPriors(tuple(families), tuple(SITES), **kw)


def extend_priors(base, families):
    """MountPriors of `families`: the rows of base for its families, the generic family's rows for the others.
    Returns (priors, {family: 'measured' or 'generic'})."""
    if "generic" not in base.families:
        raise ValueError("the base priors hold no generic family")
    rows = [base.families.index(f) if f in base.families else base.families.index("generic") for f in families]
    idx = torch.tensor(rows, dtype=torch.int64)
    kw = {f.name: getattr(base, f.name)[idx.to(getattr(base, f.name).device)] for f in fields(base)
          if isinstance(getattr(base, f.name), torch.Tensor)}
    origin = {f: ("measured" if f in base.families else "generic") for f in families}
    return MountPriors(tuple(families), tuple(base.sites), **kw), origin


def apply(state, plan_, priors=None, moments=False):
    """The state dict `state` (model, EMA, or any dict of name-sized tensors keyed as a model's) under plan_
    (module docstring); priors: the new MountPriors, required when the mount families change. ValueError when
    an entry does not fit the old names, or an old family's buffer in priors differs from the stored one."""
    mp = plan_.tables["mount_families"]
    new_prior = None
    if not mp.is_identity():
        if priors is None:
            raise ValueError("the mount families change: give the new MountPriors")
        if tuple(priors.families) != tuple(plan_.new.mount_families):
            raise ValueError(f"priors of families {priors.families}, the new names hold {plan_.new.mount_families}")
        new_prior = prior_buffers(priors)
    elif priors is not None:
        new_prior = prior_buffers(priors)
    out = type(state)()
    for key, x in state.items():
        t = table_of(key)
        if t is None:
            out[key] = x
        elif t == "mount_families":
            name = key[len(_PRIOR_PREFIX):]
            if x.shape[0] != mp.old_rows:
                raise ValueError(f"{key}: shape {tuple(x.shape)}, the old names give {mp.old_rows} families")
            if new_prior is None:
                out[key] = x
                continue
            nv = new_prior[name].to(x.device)
            if tuple(nv.shape[1:]) != tuple(x.shape[1:]) or nv.dtype != x.dtype:
                raise ValueError(f"{key}: the new priors give {tuple(nv.shape)} {nv.dtype}, stored {tuple(x.shape)} "
                                 f"{x.dtype}")
            if not torch.equal(nv[:mp.old_rows], x):
                bad = [plan_.old.mount_families[i] for i in range(mp.old_rows) if not torch.equal(nv[i], x[i])]
                raise ValueError(f"{key}: the new priors' buffers of the old families {bad} differ from the stored ones")
            out[key] = nv.clone()
        else:
            out[key] = _rows(x, plan_.tables[t], key, moments=moments)
    return out


def param_names(model):
    """[[names of the decaying parameters], [names of the others]] in the order of loop.param_groups (the
    optimizer's parameter indices): every module of model.modules(), its own parameters that require a
    gradient, the weight of an nn.Linear decaying."""
    names = {id(p): n for n, p in model.named_parameters()}
    decay, rest = [], []
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if p.requires_grad:
                (decay if isinstance(mod, nn.Linear) and name == "weight" else rest).append(names[id(p)])
    return [decay, rest]


def apply_optimizer(state, old_model, new_model, plan_):
    """An AdamW state dict written for old_model's parameters (loop.param_groups order) under plan_: per parameter
    its states with the rows of its table mapped (old rows kept, new rows zero), scalar states (step) kept; the
    indices those of new_model's parameters. ValueError when the groups or a state's shape do not fit."""
    on, nn_ = param_names(old_model), param_names(new_model)
    groups = state["param_groups"]
    if len(groups) != len(on) or len(on) != len(nn_):
        raise ValueError(f"optimizer state of {len(groups)} groups, the model gives {len(on)}")
    oshape = {n: tuple(p.shape) for n, p in old_model.named_parameters()}
    nshape = {n: tuple(p.shape) for n, p in new_model.named_parameters()}
    if [sorted(g) for g in on] != [sorted(g) for g in nn_]:
        raise ValueError("the old and new models do not hold the same parameters")
    new_index = {n: i for i, n in enumerate([n for g in nn_ for n in g])}
    moved, out_groups = {}, []
    for g, names in zip(groups, on):
        if len(g["params"]) != len(names):
            raise ValueError(f"a stored group of {len(g['params'])} parameters, the old model's holds {len(names)}")
        new_ids = []
        for idx, name in zip(g["params"], names):
            new_ids.append(new_index[name])
            st = state["state"].get(idx)
            if st is None:
                continue
            t = table_of(name)
            ns = {}
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0:
                    if tuple(v.shape) != oshape[name]:
                        raise ValueError(f"state {k} of {name}: shape {tuple(v.shape)}, the parameter {oshape[name]}")
                    v = _rows(v, plan_.tables[t], name, moments=True) if t is not None else v
                    if tuple(v.shape) != nshape[name]:
                        raise ValueError(f"state {k} of {name}: migrated shape {tuple(v.shape)}, the new parameter "
                                         f"{nshape[name]}")
                ns[k] = v
            moved[new_index[name]] = ns
        out_groups.append(dict(g, params=new_ids))
    order = [new_index[n] for g in nn_ for n in g]
    if order != list(range(len(order))):
        raise ValueError("the new parameter order is not contiguous")
    return {"state": moved, "param_groups": out_groups}


# ---------------------------------------------------------------- the model of a NameTables
def build_model(mfm_cfg, names, priors=None, jepa=None):
    """MFM of mfm_cfg under the NameTables `names` (module docstring); priors: MountPriors of names.mount_families
    (default: reports/mount_priors/priors.json extended by extend_priors); jepa: None or the kwargs of
    MFM.attach_jepa (width, depth, heads), attached after the resize so that its target copies the resized
    tokenizer."""
    from ..model import mfm as M
    if priors is None:
        priors, _ = extend_priors(MountPriors.load(), names.mount_families)
    if tuple(priors.families) != tuple(names.mount_families):
        raise ValueError(f"priors of families {priors.families}, the names hold {names.mount_families}")
    cfg = replace(mfm_cfg, sources=tuple(names.sources))
    if _has_names_field() and names != NM.STAGE1:
        cfg = replace(cfg, names=names)                    # task S2.3: the model reads its tables from the names
    model = M.MFM(cfg, priors=priors)
    n_tasks = len(names.tasks())
    pr = model.prior
    if pr.task.num_embeddings != n_tasks:
        pr.task = nn.Embedding(n_tasks, pr.d)
        nn.init.normal_(pr.task.weight, std=0.02)
        pr.n_tasks = n_tasks
    nf = names.nuis_fields()
    if tuple(nf) != TK.FIELD_NAMES:
        raise ValueError(f"nuisance fields {tuple(nf)}, the tokenizer reads {TK.FIELD_NAMES}")
    tabs = model.tokenizer.nuisance.tables
    for i, f in enumerate(TK.FIELD_NAMES):
        n = len(nf[f]) + 2
        if tabs[i].num_embeddings != n:
            tabs[i] = nn.Embedding(n, model.tokenizer.d)
            nn.init.normal_(tabs[i].weight, std=0.02)
    if jepa:
        model.attach_jepa(**jepa)
    return model


def _has_names_field():
    """True once MFMConfig carries the name tables (task S2.3)."""
    from ..model import mfm as M
    return "names" in {f.name for f in fields(M.MFMConfig)}


def _code(vocab, value):
    if value is None:
        return len(vocab)
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab) + 1


def nuisance_descriptors(stream, nuis_fields):
    """(n, F) int64 descriptor codes of an input stream under the vocabularies nuis_fields (NameTables.nuis_fields):
    the rule of tokenize.nuisance_descriptors, field by field."""
    fps = TK._check_stream(stream)
    m = stream.meta or {}
    layout = m.get("layout")
    rule = LY.FAMILY_RULES.get(layout, {})
    nf = nuis_fields
    row = {"layout": _code(nf["layout"], layout), "rate": _code(nf["rate"], TK.rate_bucket(fps))}
    if stream.modality == "imu":
        device = m.get("device") or rule.get("device")
        row.update(kind=_code(nf["kind"], "imu_site"), device=_code(nf["device"], device or "unknown"),
                   acc_content=_code(nf["acc_content"], m.get("acc_content") or "with_gravity"),
                   dim=_code(nf["dim"], None), camera=_code(nf["camera"], None))
        rows = [[dict(row, site=_code(nf["site"], SITES[int(s)]))[f] for f in TK.FIELD_NAMES] for s in m["site_ids"]]
        return np.asarray(rows, dtype=np.int64).reshape(-1, len(TK.FIELD_NAMES))
    dim = 2 if stream.modality == "pose2d" else 3
    camera = (m.get("camera") or "unknown") if dim == 2 else None
    row.update(kind=_code(nf["kind"], stream.modality), device=_code(nf["device"], None),
               acc_content=_code(nf["acc_content"], None), dim=_code(nf["dim"], dim), camera=_code(nf["camera"], camera),
               site=_code(nf["site"], None))
    return np.asarray([[row[f] for f in TK.FIELD_NAMES]], dtype=np.int64)


def recode(desc, plan_):
    """Descriptor codes (N, F) of the old tables -> the codes of the new tables (the layout, device and camera
    columns mapped by the plan; the other fields do not change)."""
    out = torch.as_tensor(desc).clone()
    for f in NM.NUIS_TABLES:
        col = TK.FIELD_NAMES.index(f)
        cmap = plan_.tables[f"nuis.{f}"].code_map()
        lut = torch.tensor([cmap[i] for i in range(plan_.tables[f"nuis.{f}"].old_rows)], dtype=out.dtype)
        out[:, col] = lut[out[:, col]]
    return out


def batch_from_windows(windows, names, **kw):
    """mfm.batch_from_windows under the NameTables `names`: its sources and mount families, and the nuisance
    descriptors coded by nuisance_descriptors with the names' vocabularies (same token order). When
    mfm.batch_from_windows coded the tokens under the names itself (names STAGE1, or its nuis_fields argument,
    task S2.3), ValueError unless its codes equal nuisance_descriptors'; otherwise its STAGE1 codes are replaced."""
    import inspect
    from ..model import mfm as M
    wins = list(windows)
    nf = names.nuis_fields()
    takes = "nuis_fields" in inspect.signature(M.batch_from_windows).parameters
    if takes and names != NM.STAGE1:
        kw = dict(kw, nuis_fields=nf)                      # task S2.3: the batch carries its vocabularies
    b = M.batch_from_windows(wins, sources=tuple(names.sources), families=tuple(names.mount_families), **kw)
    parts = [nuisance_descriptors(s, nf) for w in wins for s in w.streams if s.role == "input"]
    desc = (torch.from_numpy(np.concatenate(parts)) if parts
            else torch.zeros((0, len(TK.FIELD_NAMES)), dtype=torch.int64))
    got = b.tokens.nuis_desc
    if tuple(desc.shape) != tuple(got.shape):
        raise ValueError(f"descriptors {tuple(desc.shape)} for {tuple(got.shape)} nuisance tokens")
    if takes or names == NM.STAGE1:
        if not torch.equal(torch.as_tensor(got).to(torch.int64).cpu(), desc):
            bad = sorted({TK.FIELD_NAMES[j] for j in
                          torch.nonzero(torch.as_tensor(got).to(torch.int64).cpu() != desc)[:, 1].tolist()})
            raise ValueError(f"mfm.batch_from_windows codes the nuisance fields {bad} otherwise than "
                             "migrate.nuisance_descriptors")
        return b
    return replace(b, tokens=replace(b.tokens, nuis_desc=desc))


# ---------------------------------------------------------------- stage-1 codes
def _code_name(vocab, c):
    c = int(c)
    return str(vocab[c]) if c < len(vocab) else ("none" if c == len(vocab) else "unknown")


def _count(counts, desc, nuis_fields):
    for row in np.asarray(desc):
        for f, c in zip(TK.FIELD_NAMES, row):
            k = _code_name(nuis_fields[f], c)
            counts[f][k] = counts[f].get(k, 0) + 1


def store_streams(store=None, folds=("train",)):
    """Pseudo-streams (modality, fps, meta) of the input rows of the stage-1 window store in `folds`, read from its
    records tables through WindowStore (which refuses a lockbox or CODY person on open); no array is read."""
    from ..data import window_store as WS
    st = store if isinstance(store, WS.WindowStore) else WS.WindowStore(store)
    d = st.records[(st.records["role"] == "input") & st.records["fold"].isin(list(folds))]
    return [SimpleNamespace(modality=r.stream, fps=float(r.fps), meta=json.loads(r.meta), source=r.source)
            for r in d.itertuples(index=False)]


def layout_streams(path=LY.LAYOUTS_PATH):
    """Pseudo-streams of every input stream of the synthetic layout families (reports/layouts/layouts.json), one per
    candidate rate after the band copy, with the meta a synthetic stream carries (factory/synth.py: layout,
    site_ids (every site for a random-subset family), acc_content and device for IMU, camera for pose2d)."""
    from ..data import window_store as WS
    out = []
    for name, lay in LY.load_layouts(path).items():
        for s in lay.streams:
            if s.role != "input":
                continue
            for fps in s.fps:
                meta = {"layout": name}
                if s.modality == "imu":
                    meta.update(site_ids=list(range(len(SITES))) if lay.k_sites else s.site_ids,
                                acc_content=s.acc_content, device=lay.device)
                elif s.modality == "pose2d":
                    meta["camera"] = s.camera or "generic"
                out.append(SimpleNamespace(modality=s.modality, fps=float(WS.band_plan(s.modality, fps).fps_out),
                                           meta=meta, source="synthetic"))
    return out


def stage1_codes(names=NM.STAGE1, store=None, layouts_path=LY.LAYOUTS_PATH, folds=("train",), streams=None):
    """Descriptor codes of the stage-1 windows under the NameTables `names`, counted per nuisance field (code name:
    nuisance tokens of one window of each stream): {'store': {field: {name: n}}, 'layouts': {...}, 'store_streams':
    n, 'layout_streams': n, 'folds': [...], 'names_sha256': names.sha256(), 'layouts_sha256': ...}. streams: (store
    pseudo-streams, layout pseudo-streams) given instead of reading them (tests)."""
    nf = names.nuis_fields()
    ss, ls = streams if streams is not None else (store_streams(store, folds), layout_streams(layouts_path))
    rep = {"folds": list(folds), "store": {f: {} for f in TK.FIELD_NAMES}, "layouts": {f: {} for f in TK.FIELD_NAMES},
           "store_streams": len(ss), "layout_streams": len(ls), "names_sha256": names.sha256()}
    for part, items in (("store", ss), ("layouts", ls)):
        for s in items:
            _count(rep[part], nuisance_descriptors(s, nf), nf)
    if streams is None:
        from . import provenance as PV
        rep["layouts_sha256"] = PV.sha256_file(layouts_path)
    return rep


def trained_unknown(codes):
    """The growing nuisance fields whose 'unknown' code a stage-1 window carried (store or layouts)."""
    return tuple(f for f in NM.NUIS_TABLES
                 if codes["store"][f].get("unknown", 0) + codes["layouts"][f].get("unknown", 0) > 0)


def plan_for(old, new, task_rows="copy_unknown", codes=None, store=None):
    """(MigrationPlan, codes) of a migration from the NameTables old to new, trained_unknown read from codes: the
    nuisance codes of the windows the source trained on, stage1_codes output counted under old. codes None: counted
    by stage1_codes on the stage-1 window store (`store`) and the synthetic layouts when old is STAGE1; ValueError
    for other old names (the caller gives the codes of the windows such a source trained on) and for codes counted
    under other names than old (their 'names_sha256')."""
    if codes is None:
        if old != NM.STAGE1:
            raise ValueError(f"the source's names (SHA-256 {old.sha256()}) are not STAGE1: give the nuisance codes of "
                             "the windows it trained on (codes, counted under its names), so that the plan knows "
                             "which 'unknown' rows trained")
        codes = stage1_codes(old, store)
    if codes.get("names_sha256") != old.sha256():
        raise ValueError(f"codes counted under the names of SHA-256 {codes.get('names_sha256')}, the source's are "
                         f"{old.sha256()}")
    return plan(old, new, task_rows, trained_unknown(codes)), codes


# ---------------------------------------------------------------- equivalence
@torch.no_grad()
def equivalence(old_model, new_model, windows, old, new, gen=0):
    """{quantity: max abs difference} between old_model on windows coded under old and new_model under new: the
    posterior (mu_z, logvar_z, mu_b, logvar_b, every nuisance tensor), the prior log-density of the posterior means
    (dropout off), the nuisance log-prior, the decoded knots (rot6d, transl, contact_logit, pos_logvar) and the
    source offsets of the windows. Both models in eval mode, float32 on their device. Under STAGE1 the old batch
    is mfm.batch_from_windows itself (the stage-1 coding of tokenize.nuisance_descriptors)."""
    from ..model import mfm as M
    from ..model.prior import Cond
    windows = list(windows)
    if old == NM.STAGE1:
        bo = M.batch_from_windows(windows, sources=old.sources, families=old.mount_families, gen=gen)
    else:
        bo = batch_from_windows(windows, old, gen=gen)
    bn = batch_from_windows(windows, new, gen=gen)
    for a, b in ((bo.task, bn.task), (bo.sex, bn.sex), (bo.lengths, bn.lengths), (bo.source_id, bn.source_id)):
        if not torch.equal(a, b):
            raise ValueError("the two batches differ outside the nuisance codes")
    out = {}

    def put(k, a, b):
        out[k] = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0

    res = []
    for model, b in ((old_model, bo), (new_model, bn)):
        model.eval()
        dev = model.device
        b = b.to(dev)
        po = model.encode(b)
        lp = model.prior.log_prob(po.mu_z, Cond(b.sex, b.task, po.mu_b), po.mask, dropout=False)
        nl = model.nuis_log_prior(po.nuis, b)
        dec = model.decode(po.mu_z, po.mu_b, b.knots, b.tokens.n_patches, b.lengths)
        off = model.source_offsets(b.source_id)
        res.append((po, lp, nl, dec, off))
    (po, lpo, nlo, do, oo), (pn, lpn, nln, dn, on) = res
    for k in ("mu_z", "logvar_z", "mu_b", "logvar_b"):
        put(k, getattr(po, k), getattr(pn, k))
    tn = pn.nuis.tensors()
    for k, v in po.nuis.tensors().items():
        put(f"nuis.{k}", v, tn[k])
    put("prior_log_prob", lpo, lpn)
    put("nuis_log_prior", nlo, nln)
    for k in ("rot6d", "transl", "contact_logit", "pos_logvar"):
        put(f"knots.{k}", getattr(do, k), getattr(dn, k))
    put("source_offsets", oo, on)
    out["n_windows"] = len(windows)
    out["n_knots"] = int(do.rot6d.shape[0])
    return out


def stage1_windows(store=None, per_source=1, duration=4.0, fold="train"):
    """A fixed set of stage-1 training windows: per source of the store (pilot order), the first `per_source`
    recordings of `fold` that hold an input stream in their first `duration` s, each cut from its span start.
    WindowStore refuses a lockbox or CODY person on open."""
    from ..data import window_store as WS
    st = store if isinstance(store, WS.WindowStore) else WS.WindowStore(store)
    out = []
    rec = st.recordings
    for src in dict.fromkeys(rec["source"]):
        n = 0
        for i in rec.index[(rec["source"] == src) & (rec["fold"] == fold)]:
            g = rec.loc[i]
            w = st.window(int(i), float(g["span_t0"]), float(min(duration, g["span_s"])))
            if any(s.role == "input" for s in w.streams):
                out.append(w)
                n += 1
            if n >= per_source:
                break
    return out


# ---------------------------------------------------------------- checkpoints
def _rel(path):
    p = os.path.abspath(str(path))
    return os.path.relpath(p, paths.TREE).replace("\\", "/") if paths._inside(p, paths.TREE) else p.replace("\\", "/")


def _config_dict(text):
    return json.loads(text) if isinstance(text, str) else dict(text)


def _run_config(d):
    """RunConfig of a stored config dict, its 'names' section set aside (RunConfig reads it from S2.3 on)."""
    from . import config as CF
    return CF.from_dict({k: v for k, v in d.items() if k != "names"})


def names_of(config):
    """The NameTables of a stored config (its 'names' section), STAGE1 without one."""
    d = _config_dict(config)
    return NM.NameTables.from_dict(d["names"]) if "names" in d else NM.STAGE1


def _jepa_kwargs(state, rc):
    if not any(k.startswith("jepa.") for k in state):
        return None
    jp = rc.jepa
    return {"width": int(jp.pred_width), "depth": int(jp.pred_depth), "heads": int(jp.pred_heads)}


def model_of(ck, weights="ema", priors=None):
    """The model of a loaded checkpoint dict (checkpoint.load) under the names of its config, its `weights`
    ('model' or 'ema') loaded strictly; the mount priors are the stored buffers. Eval mode, CPU."""
    d = _config_dict(ck["config"])
    names = names_of(d)
    rc = _run_config(d)
    state = ck[weights]
    pr = priors_of_state(state, names.mount_families) if priors is None else priors
    jepa = _jepa_kwargs(state, rc) if weights == "model" else None
    model = build_model(replace(rc.model.mfm_config(), skeleton=False), names, pr, jepa)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _check_out(out, src):
    p = os.path.abspath(str(out))
    folder = os.path.dirname(p)
    if paths._norm(folder) == paths._norm(os.path.dirname(src)):
        raise ValueError(f"the migrated checkpoint goes into a new run folder, not the source's: {out}")
    if os.path.isdir(folder) and [n for n in os.listdir(folder) if not n.startswith("._")]:
        raise ValueError(f"the run folder of the migrated checkpoint must be new or empty: {folder}")
    from . import checkpoint as CK
    return CK.check_path(p)


def migrate(path, out, drift_log=None, new=S2B_DRAFT, cfg=None, task_rows="copy_unknown", codes=None, priors=None,
            store=None, keep_source=False, records_state=None):
    """Migrate the checkpoint `path` to the NameTables `new` and write it at `out` (module docstring); returns
    {out, sha256, source_sha256, plan, plan_sha256, codes, trained_unknown, mount_origin, drift, drift_log,
    code_sha256} (with keep_source, also 'source': the loaded source dict without its optimizer state). drift_log:
    the JSON-lines file a tolerated records drift of the source load is appended to, outside the source's run
    folder (default: drift_log.jsonl in the new run folder); cfg: the stored config of the migrated checkpoint
    (default: the source's with a 'names' section; a given 'names' section must equal new); codes: the nuisance
    codes of the windows the source trained on, counted under its names (plan_for: counted on the store and the
    layouts when None and the source's names are STAGE1, ValueError when None for other names); priors: the new
    MountPriors (default: reports/mount_priors/priors.json extended by extend_priors); records_state: the
    records_sha256 of an archived records state the source is read under (checkpoint.load), kept in the lineage as
    source_records_state; the migrated checkpoint keeps the source's records hashes, so it is read under the same
    state."""
    from . import checkpoint as CK
    from . import provenance as PV
    src = os.path.abspath(str(path))
    p_out = _check_out(out, src)
    log = os.path.join(os.path.dirname(p_out), CK.DRIFT_LOG) if drift_log is None else os.path.abspath(str(drift_log))
    if paths._inside(log, os.path.dirname(src)):
        raise ValueError(f"drift_log inside the source's run folder: {log}")
    ck = CK.load(src, map_location="cpu", drift_log=log, records_state=records_state)
    side = src + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            src_sha = f.read().split()[0]
    else:
        src_sha = PV.sha256_file(src)
    d = _config_dict(ck["config"])
    old = names_of(d)
    rc = _run_config(d)
    pl, codes = plan_for(old, new, task_rows, codes, store)
    tu = pl.trained_unknown
    old_pr = priors_of_state(ck["model"], old.mount_families)
    if priors is None:
        priors, origin = extend_priors(MountPriors.load(), new.mount_families)
    else:
        origin = {f: "given" for f in new.mount_families}
    jepa = _jepa_kwargs(ck["model"], rc)
    mcfg = replace(rc.model.mfm_config(), skeleton=False)
    old_model = build_model(mcfg, old, old_pr, jepa)
    old_model.load_state_dict(ck["model"], strict=True)          # the source fits its names
    model_sd = apply(ck["model"], pl, priors)
    ema_sd = None if ck.get("ema") is None else apply(ck["ema"], pl, priors)
    new_model = build_model(mcfg, new, priors, jepa)
    new_model.load_state_dict(model_sd, strict=True)
    if ema_sd is not None:
        build_model(mcfg, new, priors, None).load_state_dict(ema_sd, strict=True)
    opt_sd = None if ck.get("opt") is None else apply_optimizer(ck["opt"], old_model, new_model, pl)
    del old_model, new_model
    if cfg is None:
        cfg_d = dict(d, names=new.to_dict())
    else:
        cfg_d = _config_dict(cfg if isinstance(cfg, (str, dict)) else json.dumps(cfg))
        if "names" in cfg_d and NM.NameTables.from_dict(cfg_d["names"]) != new:
            raise ValueError("the given config's names differ from the new names")
        cfg_d.setdefault("names", new.to_dict())
    prov = copy.deepcopy(ck["provenance"])
    extra = dict(prov.get("extra") or {})
    lineage = {"migrated_from": {"path": _rel(src), "sha256": src_sha, "step": int(ck["step"])},
               "names_sha256": {"old": old.sha256(), "new": new.sha256()},
               "plan": pl.to_dict(), "plan_sha256": pl.sha256(), "codes": codes, "mount_origin": origin,
               "code_sha256": PV.code_sha256(), "source_drift": list(ck["drift"])}
    if ck.get("records_state"):
        lineage["source_records_state"] = ck["records_state"]["records_sha256"]
    extra["migration"] = lineage
    prov["extra"] = extra
    sha = CK.save(p_out, model_sd, ema_sd, opt_sd, ck.get("sched"), ck.get("sampler"), ck["rng"], cfg_d, prov,
                  step=int(ck["step"]))
    res = {"out": p_out, "sha256": sha, "source_sha256": src_sha, "plan": pl, "plan_sha256": pl.sha256(),
           "codes": codes, "trained_unknown": tu, "mount_origin": origin, "drift": list(ck["drift"]),
           "drift_log": log, "code_sha256": lineage["code_sha256"]}
    if keep_source:
        ck["opt"] = None
        res["source"] = ck
    return res


def lineage(ck):
    """The migration lineage of a loaded checkpoint dict (provenance extra 'migration'), None for a checkpoint that
    no migration wrote."""
    return ((ck.get("provenance") or {}).get("extra") or {}).get("migration")
