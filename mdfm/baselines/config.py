"""Configuration of the G12 baselines (docs/rules/G12_RULE.md, DRAFT; task T25): configs/g12_baseline.yaml.

The file names the body run's configuration (base_config, pinned by base_config_sha256, the SHA-256 of
train/config.sha256): the baseline takes from it, unchanged, the mixture (shares, caps, sources, layouts, window
lengths), the collate rows, the model size (the tokenizer and encoder of mfm.MFMConfig), the optimizer and its
learning-rate schedule (warmup, re-warmup, cosine, clip, weight decay, EMA decay), share_a (the share of optimizer
steps in phase A, synthetic only) and the memory bound. What the file adds:
- name, status, seed (the sampler of phase p is seeded seed + its index, train/loop.phase_seed, as the body's);
- arm: 'mfr' (masked-feature reconstruction) or 'jepa' (the JEPA arm), mdfm/baselines/model.py;
- mask: the mask rule of model/jepa.py (min_patches, mask_min, mask_max, block_min, block_max, max_blocks), applied
  to every window, synthetic or real (a no-body arm has no reason to prefer one); pose3d_anchor: a pose3d window
  carries masks drawn so that the patch of its heading sample stays visible (model.draw_masks), false: pose3d
  windows carry no term (E1's remedy); max_redraws: the draws allowed per window before it is left unmasked;
- head: width, depth and heads of the predictor of either arm (4 x 256 with 4 heads, the predictor of E1);
- jepa: the EMA decay of the JEPA arm's target, from ema_start at the first optimizer step to ema_end at the last
  (cosine, model/jepa.ema_decay over the whole run);
- budget (compute.py): mode 'gpu_seconds' (equal compute, the primary definition: the run's optimizer steps are
  floor(reference seconds / seconds_per_step[arm]), the reference seconds read from the body run logs
  reference_logs, seconds_per_step measured by scripts/train_g12.py --bench, 0 while not measured, which refuses
  the mode), 'reference_steps' (the mean optimizer steps of the reference logs: the body's data at the same
  sampler) or 'steps' (steps optimizer steps, for benches, smokes and tests); tolerance: the runaway stop, the
  realized clock of a gpu_seconds run may exceed the budget by this share at most (the run stops with a checkpoint
  beyond it); not a validity band: a finished run is read with its realized clock reported (compute.py);
- recompute: activation recomputation of the encoder pass (model/jepa.run_encoder); checkpoint_every and log_every
  (optimizer steps).

run_config(cfg, base, total) derives the train/config.RunConfig the loop runs: the base with this name and seed,
its schedule's token budget set so that config.step_counts gives total optimizer steps split by share_a (the body's
rule), no dev numbers, the jepa section of the base off (the auxiliary loss of E1 belongs to the body model), its
render_aug section kept (the baseline sees the renders the body saw). combined(cfg, run) is the configuration a
checkpoint stores and hashes: {'g12': ..., 'run': ...}, the g12 part through arm_view (only the run's own arm's
seconds_per_step, so that a bench of the other arm does not change the run's configuration SHA-256 and resume
keeps accepting its checkpoints).

from_dict refuses an unknown key at any level and casts each value to its field's type (train/config's rules).
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields, replace

from .. import paths
from ..model import jepa as JP
from ..train import config as CF

ARMS = ("mfr", "jepa")
MODES = ("gpu_seconds", "reference_steps", "steps")
CONFIG = os.path.join(paths.TREE, "configs", "g12_baseline.yaml")


@dataclass(frozen=True)
class MaskSection:
    """The mask rule (module docstring); the fields of model/jepa.MaskRule plus the pose3d remedy."""
    min_patches: int = 14
    mask_min: float = 0.15
    mask_max: float = 0.30
    block_min: int = 4
    block_max: int = 8
    max_blocks: int = 4
    pose3d_anchor: bool = True
    max_redraws: int = 16

    def rule(self):
        """The jepa.MaskRule of this section: every window eligible by length, synthetic or real; pose3d windows
        eligible when pose3d_anchor is true."""
        return JP.MaskRule(min_patches=int(self.min_patches), real_only=False, pose3d=bool(self.pose3d_anchor),
                           mask_min=float(self.mask_min), mask_max=float(self.mask_max),
                           block_min=int(self.block_min), block_max=int(self.block_max),
                           max_blocks=int(self.max_blocks))


@dataclass(frozen=True)
class HeadSection:
    """The predictor of either arm: depth pre-norm blocks of width with heads heads."""
    width: int = 256
    depth: int = 4
    heads: int = 4


@dataclass(frozen=True)
class JepaArmSection:
    """The EMA decay of the JEPA arm's target over the whole run (cosine from ema_start to ema_end)."""
    ema_start: float = 0.996
    ema_end: float = 1.0


@dataclass(frozen=True)
class BudgetSection:
    """The step budget (module docstring)."""
    mode: str = "gpu_seconds"
    reference_logs: tuple = ()
    seconds_per_step: dict = field(default_factory=lambda: {"mfr": 0.0, "jepa": 0.0})
    steps: int = 0
    tolerance: float = 0.25


@dataclass(frozen=True)
class G12Config:
    """A G12 baseline run (module docstring)."""
    name: str = "g12"
    status: str = ""
    base_config: str = "configs/stage1_pilot.yaml"
    base_config_sha256: str = ""
    arm: str = "mfr"
    seed: int = 20260928
    recompute: bool = False
    checkpoint_every: int = 5000
    log_every: int = 50
    mask: MaskSection = field(default_factory=MaskSection)
    head: HeadSection = field(default_factory=HeadSection)
    jepa: JepaArmSection = field(default_factory=JepaArmSection)
    budget: BudgetSection = field(default_factory=BudgetSection)


# ---------------------------------------------------------------- reading and checks
def from_dict(d):
    """G12Config of a plain dict: unknown keys refused, values cast (train/config.from_dict's rules)."""
    return CF._from(G12Config, dict(d), "")


def to_dict(cfg):
    """Plain dict of a G12Config (tuples as lists)."""
    return CF.to_dict(cfg)


def load(path=CONFIG, **overrides):
    """G12Config of a YAML file, with field overrides (arm, seed, name ...) applied, checked."""
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    cfg = from_dict(doc)
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    return check(cfg)


def problems(cfg):
    """Violations of a G12Config (list of str): the arm, the mask rule (train/config.jepa_problems on the same
    fields), max_redraws at least 1, the head (at least one block, a width split into heads of an even dimension),
    the EMA decays, the budget mode, its steps in 'steps' mode, its tolerance in (0, 1), the intervals."""
    out = []
    if cfg.arm not in ARMS:
        out.append(f"arm {cfg.arm!r}: one of {ARMS}")
    m, h, j, b = cfg.mask, cfg.head, cfg.jepa, cfg.budget
    probe = CF.JepaSection(enabled=True, min_patches=m.min_patches, mask_min=m.mask_min, mask_max=m.mask_max,
                           block_min=m.block_min, block_max=m.block_max, max_blocks=m.max_blocks,
                           pred_depth=h.depth, pred_width=h.width, pred_heads=h.heads, ema_start=j.ema_start,
                           ema_end=j.ema_end)
    out.extend(p.replace("jepa", "g12") for p in CF.jepa_problems(probe))
    if int(m.max_redraws) < 1:
        out.append(f"mask.max_redraws {m.max_redraws}: at least 1")
    if b.mode not in MODES:
        out.append(f"budget.mode {b.mode!r}: one of {MODES}")
    if b.mode == "steps" and int(b.steps) < 1:
        out.append(f"budget.steps {b.steps}: at least 1 in mode 'steps'")
    if b.mode in ("gpu_seconds", "reference_steps") and not tuple(b.reference_logs):
        out.append(f"budget.reference_logs: at least one body run log in mode {b.mode!r}")
    if not 0.0 < float(b.tolerance) < 1.0:
        out.append(f"budget.tolerance {b.tolerance}: within (0, 1)")
    bad = sorted(set(b.seconds_per_step) - set(ARMS))
    if bad:
        out.append(f"budget.seconds_per_step for unknown arms {bad}")
    if any(not (math.isfinite(float(v)) and float(v) >= 0.0) for v in b.seconds_per_step.values()):
        out.append("budget.seconds_per_step: finite, non-negative (0: not measured)")
    if int(cfg.checkpoint_every) < 1 or int(cfg.log_every) < 1:
        out.append(f"checkpoint_every {cfg.checkpoint_every} and log_every {cfg.log_every}: at least 1")
    return out


def check(cfg):
    """Raise ValueError listing problems(cfg); return cfg otherwise."""
    bad = problems(cfg)
    if bad:
        raise ValueError("G12 configuration refused: " + "; ".join(bad))
    return cfg


def tree_path(p):
    """An absolute path of a path relative to this tree (absolute paths unchanged)."""
    return p if os.path.isabs(str(p)) else os.path.join(paths.TREE, str(p))


def base_config(cfg):
    """(the base RunConfig, its SHA-256) of cfg.base_config; ValueError when base_config_sha256 is set and differs,
    or when the base enables the jepa section (the auxiliary loss of E1 is a body-model run)."""
    base = CF.load(tree_path(cfg.base_config))
    sha = CF.sha256(base)
    if cfg.base_config_sha256 and sha != cfg.base_config_sha256:
        raise ValueError(f"{cfg.base_config} hashes to {sha}, the G12 configuration pins {cfg.base_config_sha256}")
    if base.jepa.enabled:
        raise ValueError(f"{cfg.base_config} enables the jepa section: the base of a G12 baseline is a body run "
                         f"without the auxiliary loss of E1")
    return base, sha


# ---------------------------------------------------------------- the derived run configuration
def run_config(cfg, base, total_steps):
    """The train/config.RunConfig of the run (module docstring): total_steps optimizer steps, phase A holding
    round(share_a x total_steps) of them (config.step_counts)."""
    total = int(total_steps)
    if total < 1:
        raise ValueError(f"{total} optimizer steps")
    tps = CF.tokens_per_step(base)
    sched = replace(base.schedule, tokens_total=float(total * tps), checkpoint_every=int(cfg.checkpoint_every),
                    log_every=int(cfg.log_every), dev_every=0)
    run = replace(base, name=f"{cfg.name}_{cfg.arm}", seed=int(cfg.seed),
                  status=f"G12 baseline {cfg.arm} derived from {cfg.base_config} (mdfm/baselines/config.py)",
                  schedule=sched, jepa=CF.JepaSection())
    run = CF.check(run)
    a, b = CF.step_counts(run)
    if a + b != total:
        raise ValueError(f"step counts {a} + {b} differ from {total} (token budget {total * tps})")
    return run


def arm_view(cfg):
    """cfg with budget.seconds_per_step reduced to the entry of its own arm: what one run depends on. The file holds
    both arms' measured rates on one line, and a bench of one arm rewrites it (scripts/train_g12.py --bench
    --write-config); the other arm's runs keep their configuration and its SHA-256 (review of 2026-10-05)."""
    own = {cfg.arm: float(dict(cfg.budget.seconds_per_step).get(cfg.arm, 0.0))}
    return replace(cfg, budget=replace(cfg.budget, seconds_per_step=own))


def combined(cfg, run):
    """The configuration a checkpoint stores: {'g12': to_dict(arm_view(cfg)), 'run': train/config.to_dict(run)}."""
    return {"g12": to_dict(arm_view(cfg)), "run": CF.to_dict(run)}


def sha256(obj):
    """SHA-256 of the canonical JSON of a combined configuration (or any JSON-ready dict)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def field_names(cls):
    """Field names of a section class (for the docs and tests)."""
    return tuple(f.name for f in fields(cls))
