"""Run configuration of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.6 and section 5,
task B4.17): dataclasses for the model size, the mixture, the collate budgets, the schedule and the loss
weights, read from configs/stage1_pilot.yaml and hashed as canonical JSON.

Sections (every value is [cfg] or [est]; B4.18 writes measured values into the YAML file):
- model: size 'pilot' (mdfm.model.mfm.PILOT) or 'tiny' (TINY), plus field overrides of MFMConfig.
- mixture (data/sampler.py): tokens per micro-step, windows per micro-step, bucket row lengths (tokens),
  the synthetic share of a micro-step's tokens in phase A and phase B, the cumulative synthetic cap, the
  per-source cap (share of all tokens of a micro-step), the share of synthetic body windows rendered twice,
  windows per person, window length and minimum (s), the layout-family weights of the factory, the real
  sources of phase B and the misses that close a source's budget in a micro-step.
- collate (data/collate.py): kernel segments per stream-window, samples per segment, points and decoded
  knots per micro-step, supervision knots per window with a body truth.
- schedule: micro-steps per optimizer step, the token budget, the share of optimizer steps in phase A
  (share_a), AdamW and learning-rate settings, gradient clip, EMA decay, the refinement cadence and sizes,
  the checkpoint, log and dev intervals (optimizer steps), the reserved-memory bound (GB), whether a
  training alarm (train/loop.py, ALARMS) stops the run (stop_on_alarm), and the alarms read on k reads in a row
  (alarm_consecutive, task S2BA, ALARM_CONSECUTIVE_DOC; {} by default, left out of the canonical JSON there).
- losses: weights {term: [phase A, phase B]} of mdfm.train.losses, KL annealing share of phase A, free bits,
  whether synthetic windows enter HSIC with their own code, the fixed scales of the direct supervision
  (sigmas {name: value}, the keys of losses.SIGMAS, each positive). Stage-2 fields (task S2.8; spec
  docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7; STAGE2_LOSS_FIELDS, left out of the canonical JSON at their
  defaults, so that every config written before them keeps its SHA-256): plausibility_table and
  plausibility_table_sha256 (the joint-limit table of the plausibility term, default reports/body_store/priors.json,
  and the SHA-256 of its bytes, required for a table other than the default), plausibility_windows ('all', or
  'synthetic_replay', stage 2 only), hsic_strata and hsic_strata_sha256 (the HSIC strata table of the stage-2 store
  and its SHA-256, data/strata.py; stage 2 only; empty: one stratum). loss_problems checks them.
- jepa (optional, off by default; docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2): the JEPA auxiliary loss on
  the encoder in phase B (model/jepa.py, train/loop.py): enabled, weight (lambda), the eligibility (min_patches,
  real_only, pose3d), the mask rule (mask_min and mask_max shares of a window's patches, block_min and block_max
  patches per block, max_blocks), the predictor (pred_depth blocks of pred_width with pred_heads heads), the EMA
  decay of the target from ema_start to ema_end over phase B (cosine), and the activation recomputation of the
  context pass (recompute). canonical_json leaves the section out while it equals its defaults, so that a config
  without it (configs/stage1_pilot.yaml, configs/stage1_pilot_s2.yaml) keeps its SHA-256. in_phases (task S2.7, spec
  docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.6): the phases whose micro-steps carry the term, ('B',) by default,
  'C' for a stage-2 run that adopts it (the spec's 'phases'; an attribute named phase* under mdfm/train is refused
  by the label grep of tests/test_losses.py); to_dict leaves the field out while it is ('B',), so that every jepa
  section written before it keeps its canonical JSON.
- render_aug (optional, off by default; docs/proposals/F2_LEVER2_RULE.md, task F2L2): the amplitude randomization of
  the synthetic IMU renders of training (factory/synth.py, AmplitudeAug; train/loop.py hands it to the Trainer's
  renderer): enabled, the sites (names of factory.imu_render.SITES) and the half-widths gyro_log_max and
  acc_log_max of the log-uniform gains on the angular rate and on the specific force. Left out of the canonical
  JSON while it equals its defaults, as the jepa section, so that every config written before it keeps its SHA-256.
- walkway_anchor (optional, off by default; F2 lever 4, docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md, task L4C):
  the walkway anchor of phase B (data/walkway_anchor.py, losses.walkway_anchor_parts, train/loop.py): enabled, weight
  (phase B; phase A carries the term as a diagnostic only), sd (m/s, the scale of the mean-speed error), tasks (the
  weargait_pd recording keys whose straight walkway passes anchor a window) and persons_sha256 (the SHA-256 of the
  sorted training-fold persons of weargait_pd that the anchor table must hold, data.sources.person_hash). Left out
  of the canonical JSON while it equals its defaults, as jepa and render_aug.
- zupt_anchor (optional, off by default; F2 lever 5, the inertial anchor, docs/proposals/F2_LEVER5_DRAFT.md section 2,
  task L5): the label-free anchor of phase B (data/zupt_anchor.py, losses.zupt_anchor_parts, train/loop.py): enabled,
  weight (phase B; phase A carries the term as a diagnostic only), sd (m/s), tasks (the weargait_pd recording keys
  whose foot-sensor ZUPT strides anchor a window), persons_sha256 (the SHA-256 of the sorted training-fold persons of
  weargait_pd that the table must hold), edge_s (a stride anchors a window when both its still instants lie at least
  this far inside it) and min_length_m (the shortest ZUPT stride the table keeps). Left out of the canonical JSON while
  it equals its defaults, as jepa, render_aug and walkway_anchor, so that every config written before it keeps its
  SHA-256.
- names (optional, at names.STAGE1 by default; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.1, task S2.3): the
  name tables of the run (train/names.NameTables): sources (synthetic first, then the real sources the mixture may
  draw), vocabulary_path (the task vocabulary file, relative to the tree), layouts, devices and cameras (the nuisance
  descriptor vocabularies) and mount_families (those of the mount priors). name_tables(cfg) gives the NameTables,
  real_sources(cfg) the real sources, model_config(cfg) the MFMConfig with the section's sources and names (at the
  defaults: cfg.model.mfm_config() itself, the stage-1 model). Left out of the canonical JSON while it equals its
  defaults, as jepa, render_aug and walkway_anchor, so that every stage-1 config keeps its SHA-256. Task S2.7 wired
  the section (NAMES_WIRED): train/loop.py builds the model of model_config(cfg) and collates its batches with the
  section's sources and nuisance vocabularies, and scripts/train_stage1.py opens the stage-2 window store under the
  section's tables; before it, problems() refused a section away from its defaults.
- tier (optional, RELEASE by default; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.3, task S2.4): the run's
  tier, one of train/provenance.TIERS (RELEASE, INTERNAL). Left out of the canonical JSON while it is RELEASE, so
  that every config written before it keeps its SHA-256 and a config without the key reads as RELEASE.
  train/checkpoint.py stores it in the provenance of every checkpoint, refuses a RELEASE save meeting an INTERNAL
  person or provenance and a RELEASE load of an INTERNAL checkpoint (train/provenance.py, module docstring).
- stage2 (optional, off by default; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.6, task S2.7): phase C, the
  stage-2 phase of the training loop, entered from a stage-1 phase B end through a migration (train/loop.py,
  Trainer.init_stage2): enabled; units, the sampling units of the stage-2 window store ('<source>:<groups>', the
  unit names of data/sources.py, departure 5); replay_share, the share of a micro-step's tokens drawn as this
  config's phase B mixture (its synthetic share and its real healthy sources, their rules unchanged) [D§5];
  unit_cap, the cap of each unit on all tokens of a micro-step [D§5; S1 2.2]; steps, the optimizer steps of phase C
  (scripts/stage2_budget.py writes them, task S2.9); task_rows, the start of the new task rows of the migration
  (train/migrate.TASK_ROWS); redraws, the phase C sampler's draws again of a plan or a step the caps leave empty
  (opt-in, data/sampler.py, task T15B; set from the plan check of scripts/bench_stage1.py); freeze (task S2B, the
  recipe of run S2b: reports/gpu_queue/decisions.md, entry 2026-10-12 00:24), the parameter groups of
  STAGE2_FREEZE_GROUPS that phase C does not train (requires_grad False, left out of the optimizer, their EMA kept at
  the start's; train/loop.py, freeze_parameters), () by default; window_cap_per_person (task S2B), the cap of each
  unit's share of the phase C windows per training person of the unit (a unit of n persons holds at most n x the
  value of the windows the phase has drawn, data/sampler.py, enforce_window_caps), 0.0 (off) by default. Left out
  of the canonical JSON while it equals its defaults, and each of freeze and window_cap_per_person while it equals
  its own default (OPTIONAL_SECTION_FIELDS), so that every stage-1 config and configs/stage2_s2a.yaml keep their
  SHA-256. With the section
  enabled, run_phases(cfg) is (A, B, C) and run_steps(cfg) gives the three step counts (A and B those of
  step_counts, the start recipe's); problems() refuses a unit outside the names section's sources, a replay the caps
  cannot absorb, a cumulative synthetic share over the three phases above synthetic_cap and the walkway anchor (its
  table reads phase B's weargait_pd items, not phase C's).

Phase-indexed values are pairs (phase A, phase B), as losses.WEIGHTS; phase C reads the phase B values (loss
weights, synthetic share of the replay, the re-warmup and cosine of the learning rate, the refinement cadence). The
60 percent synthetic cap of design section 5 is read cumulatively over the run's tokens (OD9, decided 2026-09-28).
At full micro-steps the share is share_a x synthetic_share[0] + (1 - share_a) x synthetic_share[1] (0.25 x 1.0 +
0.75 x 0.45 = 0.5875 with the section 5 values), which problems() checks. The sampler fills a phase A micro-step
closer to tokens_per_micro than a phase B one (whose real windows leave budget unused), so phase A holds more than
share_a of the tokens: cumulative_synthetic_share(cfg, tokens, shares) weighs the phases by their measured tokens
per micro-step (the sampler's totals), which the training loop checks against the cap.

Hash: SHA-256 of the canonical JSON of to_dict() (sorted keys, no spaces, tuples as lists; the jepa, render_aug,
walkway_anchor, zupt_anchor, names and stage2 sections omitted at their defaults, tier while it is RELEASE and
jepa.in_phases while it is ('B',), the stage-2 fields of the losses section at their defaults, stage2.freeze and
stage2.window_cap_per_person at theirs, schedule.alarm_consecutive at {}). from_dict refuses an unknown key at any
level and casts each value to its field's type. The hash
covers the names section's vocabulary_path, not the file's content: names.NameTables.sha256() covers the content,
and a phase C run records it (the migration lineage of train/loop.py, Trainer.init_stage2).
"""
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace

from ..factory.layouts import DEFAULT_MIXTURE, SITES
from ..model import mfm as M
from ..model.prior import TASK_UNKNOWN
from . import losses as LS
from . import names as NM
from .provenance import RELEASE, TIERS

REAL_SOURCES = tuple(M.SOURCES[1:])          # the stage-1 real sources (names.STAGE1); a run's: real_sources(cfg)
SIZES = {"pilot": M.PILOT, "tiny": M.TINY}
# True since task S2.7: the trainer builds its model from model_config(cfg), codes its batches with the model's
# sources and nuisance vocabularies and reads the stage-2 window store under the section's tables; while it was
# False, problems() refused a names section away from its defaults (NAMES_NOT_WIRED)
NAMES_WIRED = True
NAMES_NOT_WIRED = "names: a non-default names section is not wired into train/loop.py (task S2.7)"


@dataclass(frozen=True)
class ModelSection:
    """size: 'pilot' or 'tiny'; overrides: MFMConfig fields replaced (sources, skeleton and names excluded: the
    sources and names of a run come from its names section, model_config)."""
    size: str = "pilot"
    overrides: dict = field(default_factory=dict)

    def mfm_config(self):
        """The MFMConfig of this section (the stage-1 name tables; model_config(cfg) applies a run's)."""
        if self.size not in SIZES:
            raise ValueError(f"model size {self.size!r}, expected one of {tuple(SIZES)}")
        names = {f.name for f in fields(M.MFMConfig)} - {"sources", "skeleton", "names"}
        bad = sorted(set(self.overrides) - names)
        if bad:
            raise ValueError(f"model overrides of unknown fields {bad}")
        return replace(SIZES[self.size], **self.overrides)


@dataclass(frozen=True)
class MixtureSection:
    """Sampler settings (module docstring); shares of a micro-step's tokens, lengths in s."""
    tokens_per_micro: int = 16384
    max_windows: int = 512
    rows: tuple = (128, 640, 1536)
    synthetic_share: tuple = (1.0, 0.45)
    synthetic_cap: float = 0.60
    source_cap: float = 0.15
    pair_share: float = 0.25
    windows_per_person: int = 2
    win_s: float = 10.0
    min_s: float = 1.0
    layouts: dict = field(default_factory=lambda: dict(DEFAULT_MIXTURE))
    sources: tuple = REAL_SOURCES
    max_misses: int = 8


@dataclass(frozen=True)
class CollateSection:
    """Kernel points and decoded knots of a micro-step (module docstring)."""
    segments: int = 4
    segment: int = 16
    max_points: int = 32768
    max_knots: int = 32768
    supervision_knots: int = 64


@dataclass(frozen=True)
class ScheduleSection:
    """Optimizer steps and their settings (module docstring); learning rates per step, GB for memory."""
    micro_per_step: int = 2
    tokens_total: float = 1.25e9
    share_a: float = 0.25
    lr: float = 3.0e-4
    lr_min: float = 3.0e-5
    warmup_steps: int = 2000
    rewarmup_steps: int = 500
    betas: tuple = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.05
    fused: str = "auto"
    clip: float = 1.0
    ema_decay: float = 0.999
    refine_every: int = 4
    refine_steps: int = 5
    refine_points: int = 64
    eval_refine_steps: int = 20
    eval_refine_points: int = 256
    checkpoint_every: int = 2000
    log_every: int = 50
    dev_every: int = 5000
    max_reserved_gb: float = 14.5
    stop_on_alarm: bool = True
    # task S2BA: {alarm name: k}, an alarm of train/loop.py ALARMS raised only when its line is crossed at k of its
    # reads in a row (ALARM_CONSECUTIVE_DOC); {} reads every alarm on one read, as before (left out of to_dict there)
    alarm_consecutive: dict = field(default_factory=dict)


# the names of train/loop.py ALARMS (tests/test_alarm_consecutive.py checks the copy); schedule.alarm_consecutive
# names only these
ALARM_NAMES = ("attention_entropy", "beta_error", "logvar_z", "spread_ratio", "z0_margin_mm", "root_speed_ratio",
               "heading_gain", "jepa_collapse", "unit_nll_rise")
ALARM_CONSECUTIVE_DOC = """schedule.alarm_consecutive (task S2BA; reports/bench/stage2_beta_alarm/summary.md): {alarm
name: k}, k an integer of at least 1. An alarm named there is raised (logged 'alarm', and with stop_on_alarm a
checkpoint and AlarmError) only when its value crosses its line at k of its reads in a row: its source's records of the
phase ('log' records or 'dev' events), from the alarm's first step on, a read without a value (None) counting neither
for nor against the run; a crossing before the k-th logs 'alarm_watch' (the alarm, its value, the reads in a row and k)
and stops nothing. The run of reads survives a resume: the loop counts it again from the run folder's log.jsonl (the
records of the phase before the current step, the last record of each step). An alarm not named, and every alarm at the
default {}, is read on one read as before."""


def alarm_problems(sc):
    """Violations of schedule.alarm_consecutive (ALARM_CONSECUTIVE_DOC): names of ALARM_NAMES, values integers of at
    least 1 (a bool is not an integer here)."""
    ac = sc.alarm_consecutive
    if not isinstance(ac, dict):
        return [f"schedule.alarm_consecutive {ac!r}: a mapping of alarm names to integers"]
    out = []
    bad = sorted(str(k) for k in ac if k not in ALARM_NAMES)
    if bad:
        out.append(f"schedule.alarm_consecutive names {bad}, not alarms of train/loop.py ({list(ALARM_NAMES)})")
    bad_k = sorted(str(k) for k, v in ac.items() if isinstance(v, bool) or not isinstance(v, int) or v < 1)
    if bad_k:
        out.append(f"schedule.alarm_consecutive {bad_k}: each value an integer of at least 1")
    return out


PLAUSIBILITY_TABLE = "reports/body_store/priors.json"     # losses.plausibility_table at its default (tree-relative)


@dataclass(frozen=True)
class LossSection:
    """weights {term: (phase A, phase B)} of mdfm.train.losses.TERMS, kl_anneal, free_bits, hsic_synthetic,
    sigmas {name: value} of losses.SIGMAS; the stage-2 fields (task S2.8, STAGE2_LOSS_FIELDS, left out of to_dict at
    their defaults): plausibility_table, the joint-limit table of the plausibility term (a path, relative to the tree;
    default PLAUSIBILITY_TABLE), and plausibility_table_sha256, the SHA-256 of its bytes (required for a table other
    than the default, so that a table edited in place fails the config check); plausibility_windows, the windows it
    weighs (losses.PLAUSIBILITY_WINDOWS: 'all', or 'synthetic_replay' in a stage-2 run); hsic_strata and
    hsic_strata_sha256, the HSIC strata table of the stage-2 store (data/strata.py; a path relative to the tree and
    its SHA-256; empty: one stratum, the stage-1 HSIC)."""
    weights: dict = field(default_factory=lambda: {k: tuple(v) for k, v in LS.WEIGHTS.items()})
    kl_anneal: float = LS.KL_ANNEAL
    free_bits: float = LS.FREE_BITS
    hsic_synthetic: bool = True
    sigmas: dict = field(default_factory=lambda: dict(LS.SIGMAS))
    plausibility_table: str = PLAUSIBILITY_TABLE
    plausibility_windows: str = "all"
    hsic_strata: str = ""
    hsic_strata_sha256: str = ""
    plausibility_table_sha256: str = ""
    sup_rest: tuple = (0.0, 0.0)

    def loss_config(self):
        """The losses.LossConfig of this section (the plausibility table resolved under the tree)."""
        return LS.LossConfig(weights={k: tuple(float(x) for x in v) for k, v in self.weights.items()},
                             kl_anneal=float(self.kl_anneal), free_bits=float(self.free_bits),
                             hsic_synthetic=bool(self.hsic_synthetic),
                             sigmas={k: float(v) for k, v in self.sigmas.items()},
                             plausibility_table=NM.resolve(str(self.plausibility_table)),
                             plausibility_windows=str(self.plausibility_windows),
                             sup_rest=tuple(float(x) for x in self.sup_rest))


# the stage-2 fields of LossSection (task S2.8) at their defaults, left out of to_dict there; train/loop.stage2_recipe
# logs a difference in them as drift, not as a departure from the start recipe
STAGE2_LOSS_FIELDS = {"plausibility_table": PLAUSIBILITY_TABLE, "plausibility_windows": "all", "hsic_strata": "",
                      "hsic_strata_sha256": "", "plausibility_table_sha256": ""}


@dataclass(frozen=True)
class JepaSection:
    """The JEPA auxiliary loss of phase B (module docstring; docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2):
    enabled; weight lambda of the per-window term; eligible windows: at least min_patches patches, real only
    (real_only), pose3d windows only when pose3d is true (the pose3d heading leak, section 2.3); masks: a target
    share drawn between mask_min and mask_max of the patches, blocks of block_min to block_max patches, at most
    max_blocks; predictor of pred_depth blocks of width pred_width with pred_heads heads; target EMA decay from
    ema_start at the first phase B step to ema_end at its end (cosine); recompute: the context pass recomputes its
    activations in the backward pass; in_phases: the phases whose micro-steps carry the term (('B',), with 'C' for a
    stage-2 run that adopts JEPA; left out of to_dict while it is ('B',), JEPA_PHASES)."""
    enabled: bool = False
    weight: float = 1.0
    min_patches: int = 14
    real_only: bool = True
    pose3d: bool = False
    mask_min: float = 0.15
    mask_max: float = 0.30
    block_min: int = 4
    block_max: int = 8
    max_blocks: int = 4
    pred_depth: int = 4
    pred_width: int = 256
    pred_heads: int = 4
    ema_start: float = 0.996
    ema_end: float = 1.0
    recompute: bool = True
    in_phases: tuple = ("B",)


JEPA_PHASES = ("B",)        # JepaSection.in_phases at its default, left out of to_dict (module docstring)


@dataclass(frozen=True)
class RenderAugSection:
    """The amplitude randomization of the synthetic IMU renders of training (module docstring; F2 lever 2,
    docs/proposals/F2_LEVER2_RULE.md): with enabled, at each site of `sites` that a rendered IMU stream holds, the
    deviation of the angular rate and of the specific force from their window means is multiplied by gains drawn
    per window, site and feature, log-uniform within exp(+-gyro_log_max) and exp(+-acc_log_max)
    (factory.synth.AmplitudeAug). The defaults are the lever's values: the lower back and both thighs, half-width
    0.5 on both features (gains 0.607 to 1.649); enabled is off."""
    enabled: bool = False
    sites: tuple = ("lower_back", "thigh_L", "thigh_R")
    gyro_log_max: float = 0.5
    acc_log_max: float = 0.5


WALKWAY_TASKS = ("SelfPace", "HurriedPace", "SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN")


@dataclass(frozen=True)
class WalkwayAnchorSection:
    """The walkway anchor of phase B (module docstring; F2 lever 4, docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md,
    section A): with enabled, a real weargait_pd window of a training-fold person that holds at least one stride of a
    straight walkway pass of a recording whose key is in `tasks` carries the term losses.walkway_anchor_parts, the
    squared mean-speed error over 2 sd^2 counted at each knot of the segment, weighted by `weight` in phase B only;
    persons_sha256: the SHA-256 (data.sources.person_hash) of the window store's weargait_pd training fold, which
    data.walkway_anchor.build_anchor_table asserts before any target file opens. The defaults are off: weight 0.0,
    the scale of sup_root_rate (0.1 m/s), the five straight-walkway tasks, no persons."""
    enabled: bool = False
    weight: float = 0.0
    sd: float = 0.1
    tasks: tuple = WALKWAY_TASKS
    persons_sha256: str = ""


# the weargait_pd tasks with foot motion of the lever-5 draft (section 2; option O1 adds TandemGait and Balance)
ZUPT_TASKS = ("SelfPace", "HurriedPace", "SelfPace_mat", "HurriedPace_mat", "SelfPace_matTURN", "TUG", "FreeWalk",
              "SelfPace_doorpat")


@dataclass(frozen=True)
class ZuptAnchorSection:
    """The inertial anchor of phase B (module docstring; F2 lever 5, docs/proposals/F2_LEVER5_DRAFT.md, section 2):
    with enabled, a real weargait_pd window of a training-fold person whose recording key is in `tasks` and that
    holds a version-2 ZUPT stride of the table (data/zupt_anchor.py: its two still instants at least edge_s inside
    the window, its length at least min_length_m) carries the term losses.zupt_anchor_parts (distance and stillness
    of the decoded ankle of that foot), weighted by `weight` in phase B only; persons_sha256: the SHA-256 of the
    window store's weargait_pd training fold, which data.zupt_anchor.build_table asserts before any value is read.
    The defaults are off: weight 0.0, sd 0.1 m/s (the walkway anchor's and sup_root_rate's), the eight tasks of
    ZUPT_TASKS, no persons, edge 0.1 s and the 0.3 m floor of the draft's coverage count (section 6 (c))."""
    enabled: bool = False
    weight: float = 0.0
    sd: float = 0.1
    tasks: tuple = ZUPT_TASKS
    persons_sha256: str = ""
    edge_s: float = 0.1
    min_length_m: float = 0.3


@dataclass(frozen=True)
class NamesSection:
    """The name tables of the run (module docstring; train/names.NameTables, its fields in its order): the defaults
    are names.STAGE1, today's tables (mfm.SOURCES, docs/rules/task_vocabulary.yaml, the layout, device and camera
    vocabularies of tokenize.NUIS_FIELDS, the family order of reports/mount_priors/priors.json)."""
    sources: tuple = NM.STAGE1.sources
    vocabulary_path: str = NM.STAGE1.vocabulary_path
    layouts: tuple = NM.STAGE1.layouts
    devices: tuple = NM.STAGE1.devices
    cameras: tuple = NM.STAGE1.cameras
    mount_families: tuple = NM.STAGE1.mount_families

    def tables(self):
        """The names.NameTables of this section (its refusals as ValueError or TypeError)."""
        return NM.NameTables(**{f.name: getattr(self, f.name) for f in fields(self)})


STAGE2_PHASE = "C"          # the phase of a stage-2 run (module docstring, stage2)
TASK_ROWS = ("copy_unknown", "zero")       # train/migrate.TASK_ROWS (tests/test_phase_c.py checks the copy)
# the parameter groups a stage-2 config may freeze in phase C (task S2B; stage2.freeze): group -> the prefixes of the
# model's parameter names (mdfm.model.mfm.MFM) it holds. 'decoder' is the BodyDecoder (the map from z, beta and the
# knot times to the body: rotations, root path, contacts, pos_logvar; 12,728,447 parameters in the pilot); 'body_heads'
# the posterior's two body heads, latent_head (mu_z, logvar_z, the motion latent) and shape_head (mu_b, logvar_b, the
# shape), as against its nuisance heads (heads.*); 49,248 parameters. train/loop.freeze_parameters applies them.
STAGE2_FREEZE_GROUPS = {"decoder": ("decoder.",), "body_heads": ("latent_head.", "shape_head.")}


@dataclass(frozen=True)
class Stage2Section:
    """Phase C of a stage-2 run (module docstring; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.6): enabled;
    units ('<source>:<groups>' of the stage-2 window store); replay_share and unit_cap, shares of all tokens of a
    micro-step; steps, optimizer steps of phase C; task_rows (TASK_ROWS); redraws (data/sampler.py); freeze, the groups
    of STAGE2_FREEZE_GROUPS phase C leaves untrained, and window_cap_per_person, the per-person cap of a unit's share
    of the phase C windows (task S2B; 0.0 off). Off by default."""
    enabled: bool = False
    units: tuple = ()
    replay_share: float = 0.30
    unit_cap: float = 0.15
    steps: int = 0
    task_rows: str = "copy_unknown"
    redraws: int = 0
    freeze: tuple = ()
    window_cap_per_person: float = 0.0


# the S2B fields of Stage2Section at their defaults, left out of to_dict there (OPTIONAL_SECTION_FIELDS)
STAGE2_S2B_FIELDS = {"freeze": (), "window_cap_per_person": 0.0}


@dataclass(frozen=True)
class RunConfig:
    """A stage-1 run: name, seed, status (a line on where the values come from), the five sections and the
    optional jepa, render_aug, walkway_anchor and names sections (JepaSection, RenderAugSection, WalkwayAnchorSection,
    off by default; NamesSection, the stage-1 tables by default), the tier (RELEASE by default), the stage2
    section (Stage2Section, off by default: a stage-2 run, phase C) and the zupt_anchor section (ZuptAnchorSection,
    off by default; last, so that no positional construction moves) (module docstring)."""
    name: str = "stage1_pilot"
    seed: int = 20260928
    status: str = ""
    model: ModelSection = field(default_factory=ModelSection)
    mixture: MixtureSection = field(default_factory=MixtureSection)
    collate: CollateSection = field(default_factory=CollateSection)
    schedule: ScheduleSection = field(default_factory=ScheduleSection)
    losses: LossSection = field(default_factory=LossSection)
    jepa: JepaSection = field(default_factory=JepaSection)
    render_aug: RenderAugSection = field(default_factory=RenderAugSection)
    walkway_anchor: WalkwayAnchorSection = field(default_factory=WalkwayAnchorSection)
    names: NamesSection = field(default_factory=NamesSection)
    tier: str = RELEASE
    stage2: Stage2Section = field(default_factory=Stage2Section)
    zupt_anchor: ZuptAnchorSection = field(default_factory=ZuptAnchorSection)


# sections left out of to_dict while they equal their defaults (a config written before they existed keeps its
# canonical JSON and its SHA-256)
OPTIONAL_SECTIONS = {"jepa": JepaSection, "render_aug": RenderAugSection, "walkway_anchor": WalkwayAnchorSection,
                     "names": NamesSection, "stage2": Stage2Section, "zupt_anchor": ZuptAnchorSection}
# scalar fields of RunConfig left out of to_dict while they equal these values, for the same reason
OPTIONAL_FIELDS = {"tier": RELEASE}
# fields of a section left out of to_dict while they equal these values, for the same reason
# the rest-joint supervision of the open port (losses.sup_rest, step M4), off and left out of to_dict by default
OPEN_LOSS_FIELDS = {"sup_rest": (0.0, 0.0)}
OPTIONAL_SECTION_FIELDS = {JepaSection: {"in_phases": JEPA_PHASES}, LossSection: {**STAGE2_LOSS_FIELDS, **OPEN_LOSS_FIELDS},
                           Stage2Section: dict(STAGE2_S2B_FIELDS), ScheduleSection: {"alarm_consecutive": {}}}


def _at_default(v, d):
    """True when a field value v equals its optional default d (a tuple default compares as a tuple)."""
    if isinstance(d, tuple):
        return isinstance(v, (list, tuple)) and tuple(v) == d
    return v == d


# ---------------------------------------------------------------- dict and JSON
def _plain(x):
    if is_dataclass(x):
        out = {}
        skip = OPTIONAL_SECTION_FIELDS.get(type(x), {})
        for f in fields(x):
            v = getattr(x, f.name)
            if isinstance(x, RunConfig) and f.name in OPTIONAL_SECTIONS and v == OPTIONAL_SECTIONS[f.name]():
                continue
            if isinstance(x, RunConfig) and f.name in OPTIONAL_FIELDS and v == OPTIONAL_FIELDS[f.name]:
                continue
            if f.name in skip and _at_default(v, skip[f.name]):
                continue
            out[f.name] = _plain(v)
        return out
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def to_dict(cfg):
    """Plain dict of a RunConfig (or of any section): tuples as lists; an optional section (OPTIONAL_SECTIONS)
    left out while it equals its defaults."""
    return _plain(cfg)


def canonical_json(cfg):
    """JSON text of to_dict(cfg) (a dict is taken as is) with sorted keys and no spaces."""
    obj = cfg if isinstance(cfg, dict) else to_dict(cfg)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(cfg):
    """SHA-256 of canonical_json(cfg)."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def _cast(value, default, where):
    if is_dataclass(default):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _from(type(default), value, where)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true or false, got {value!r}")
        return value
    if isinstance(default, int):
        v = float(value)
        if not v.is_integer():
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return int(v)
    if isinstance(default, float):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{where}: not finite: {value!r}")
        return v
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string, got {value!r}")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        if default and all(type(x) is type(default[0]) for x in default):
            return tuple(_cast(v, default[0], f"{where}[{i}]") for i, v in enumerate(value))
        return tuple(value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {value!r}")
        proto = next(iter(default.values()), None)
        if proto is None:
            return {str(k): (tuple(v) if isinstance(v, list) else v) for k, v in value.items()}
        return {str(k): _cast(v, proto, f"{where}.{k}") for k, v in value.items()}
    return value


def _from(cls, d, where):
    names = {f.name: f for f in fields(cls)}
    bad = sorted(set(d) - set(names))
    if bad:
        raise ValueError(f"{where or 'config'}: unknown keys {bad}")
    base = cls()
    kw = {}
    for k, v in d.items():
        kw[k] = _cast(v, getattr(base, k), f"{where}.{k}" if where else k)
    return cls(**kw)


def from_dict(d):
    """RunConfig of a plain dict (to_dict's form): unknown keys refused, values cast to the field types."""
    return _from(RunConfig, dict(d), "")


def load(path):
    """RunConfig of a YAML file (configs/stage1_pilot.yaml), checked (check())."""
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    cfg = from_dict(doc or {})
    check(cfg)
    return cfg


def from_json(text):
    """RunConfig of canonical_json's text."""
    return from_dict(json.loads(text))


# ---------------------------------------------------------------- derived values and checks
def name_tables(cfg):
    """The names.NameTables of a run: names.STAGE1 itself while the names section is at its defaults, else the
    section's tables."""
    return NM.STAGE1 if cfg.names == NamesSection() else cfg.names.tables()


def real_sources(cfg):
    """The real sources a run's mixture may draw: its names section's sources after 'synthetic' (REAL_SOURCES at
    the defaults)."""
    return tuple(cfg.names.sources[1:])


def model_config(cfg):
    """The mfm.MFMConfig of a run: cfg.model.mfm_config() itself while the names section is at its defaults (the
    stage-1 model, names None), else with the section's sources and its NameTables as names."""
    mc = cfg.model.mfm_config()
    if cfg.names == NamesSection():
        return mc
    tables = cfg.names.tables()
    return replace(mc, sources=tables.sources, names=tables)


def cumulative_synthetic_share(cfg, tokens=None, shares=None):
    """Synthetic share of all tokens of the run. Phase A holds share_a of the optimizer steps and phase B the
    rest; tokens: (mean tokens of a micro-step in phase A, in phase B), default tokens_per_micro in both, so
    that the step share is the token share; shares: (synthetic share of the tokens in phase A, in phase B),
    default synthetic_share. With the defaults: share_a x synthetic_share[0] + (1 - share_a) x
    synthetic_share[1]."""
    a = float(cfg.schedule.share_a)
    n = (float(cfg.mixture.tokens_per_micro),) * 2 if tokens is None else tuple(float(x) for x in tokens)
    s = tuple(float(x) for x in (cfg.mixture.synthetic_share if shares is None else shares))
    wa, wb = a * n[0], (1.0 - a) * n[1]
    return (wa * s[0] + wb * s[1]) / (wa + wb)


def tokens_per_step(cfg):
    """Tokens of an optimizer step at full micro-steps: micro_per_step x tokens_per_micro (the sampler fills a
    micro-step to at most tokens_per_micro)."""
    return int(cfg.schedule.micro_per_step) * int(cfg.mixture.tokens_per_micro)


def step_counts(cfg):
    """(optimizer steps in phase A, in phase B) of the token budget at full micro-steps, rounded to whole
    steps."""
    total = int(round(float(cfg.schedule.tokens_total) / tokens_per_step(cfg)))
    a = int(round(float(cfg.schedule.share_a) * total))
    return a, total - a


def stage2_on(cfg):
    """True when the run is a stage-2 run (its stage2 section enabled)."""
    return bool(cfg.stage2.enabled)


def run_phases(cfg):
    """The phases of a run: ('A', 'B'), and 'C' after them for a stage-2 run (stage2_on)."""
    return ("A", "B", STAGE2_PHASE) if stage2_on(cfg) else ("A", "B")


def run_steps(cfg):
    """Optimizer steps of each phase of run_phases(cfg): step_counts(cfg), and stage2.steps for phase C."""
    return step_counts(cfg) + ((int(cfg.stage2.steps),) if stage2_on(cfg) else ())


def stage2_synthetic_share(cfg):
    """Synthetic share of all tokens of a stage-2 run at full micro-steps over its three phases (the cap is read
    over stages 1 and 2 together, spec section 4.6 and section 9 item 7): phase A at synthetic_share[0], phase B at
    synthetic_share[1], phase C at replay_share x synthetic_share[1], weighed by run_steps(cfg)."""
    a, b, c = run_steps(cfg)
    s = tuple(float(x) for x in cfg.mixture.synthetic_share)
    tot = a + b + c
    return (a * s[0] + b * s[1] + c * float(cfg.stage2.replay_share) * s[1]) / tot if tot else float("nan")


def problems(cfg):
    """Violations of the run's rules (list of str, empty when the config holds): shares in [0, 1] with phase A
    synthetic only, the cumulative synthetic cap (OD9), the per-source cap feasible for the real share of
    phase B, bucket rows increasing, one kernel segment per stream-window of max_windows windows within the
    point budget, the loss terms of mdfm.train.losses, the supervision scales (the keys of losses.SIGMAS,
    each positive and finite), the model size, the jepa section (jepa_problems), the render_aug section
    (render_aug_problems), the walkway_anchor section (walkway_anchor_problems), the zupt_anchor section
    (zupt_anchor_problems) and the names section
    (names_problems, and NAMES_NOT_WIRED for a section away from its defaults while NAMES_WIRED is False), the
    tier (one of provenance.TIERS), the stage2 section and jepa.in_phases (stage2_problems), the stage-2 fields of the
    losses section (loss_problems), schedule.alarm_consecutive (alarm_problems); the mixture's sources
    are checked against the names section's real sources."""
    out = []
    mx, co, sc = cfg.mixture, cfg.collate, cfg.schedule
    s = tuple(float(x) for x in mx.synthetic_share)
    if len(s) != 2 or not all(0.0 <= x <= 1.0 for x in s):
        out.append(f"synthetic_share {mx.synthetic_share}: two shares in [0, 1] (phase A, phase B)")
    elif s[0] != 1.0:
        out.append(f"synthetic_share of phase A is {s[0]}: phase A is synthetic only (OD9)")
    else:
        cum = cumulative_synthetic_share(cfg)
        if cum > float(mx.synthetic_cap) + 1e-12:
            out.append(f"cumulative synthetic share {cum:.4f} above the cap {mx.synthetic_cap}")
        real = 1.0 - s[1]
        if real > len(mx.sources) * float(mx.source_cap) + 1e-12:
            out.append(f"real share {real:.4f} of phase B above {len(mx.sources)} sources x cap {mx.source_cap}")
    if not 0.0 < float(mx.source_cap) <= 1.0:
        out.append(f"source_cap {mx.source_cap} outside (0, 1]")
    if not 0.0 <= float(sc.share_a) <= 1.0:
        out.append(f"share_a {sc.share_a} outside [0, 1]")
    if not 0.0 <= float(mx.pair_share) <= 1.0:
        out.append(f"pair_share {mx.pair_share} outside [0, 1]")
    rows = tuple(int(r) for r in mx.rows)
    if not rows or any(r <= 0 for r in rows) or any(b <= a for a, b in zip(rows, rows[1:])):
        out.append(f"rows {mx.rows}: positive and increasing")
    if int(mx.max_windows) * int(co.segment) > int(co.max_points):
        out.append(f"{mx.max_windows} windows x {co.segment} samples exceed max_points {co.max_points}")
    if not 0.0 < float(mx.min_s) <= float(mx.win_s):
        out.append(f"min_s {mx.min_s} and win_s {mx.win_s}")
    if int(mx.windows_per_person) < 1:
        out.append("windows_per_person below 1")
    bad_src = sorted(set(mx.sources) - set(real_sources(cfg)))
    if bad_src:
        out.append(f"sources outside the pilot's real sources: {bad_src}" if cfg.names == NamesSection() else
                   f"sources outside the real sources of the names section: {bad_src}")
    if set(cfg.losses.weights) != set(LS.TERMS):
        out.append(f"loss weights for {sorted(cfg.losses.weights)}, expected {sorted(LS.TERMS)}")
    elif any(len(tuple(v)) != 2 for v in cfg.losses.weights.values()):
        out.append("every loss weight is a pair (phase A, phase B)")
    sig = dict(cfg.losses.sigmas)
    if set(sig) != set(LS.SIGMAS):
        out.append(f"loss sigmas for {sorted(sig)}, expected {sorted(LS.SIGMAS)}")
    bad_sig = sorted(k for k, v in sig.items() if not (isinstance(v, (int, float)) and math.isfinite(float(v))
                                                        and float(v) > 0))
    if bad_sig:
        out.append(f"loss sigmas {bad_sig} are not positive and finite")
    try:
        cfg.model.mfm_config()
    except ValueError as e:
        out.append(str(e))
    out.extend(jepa_problems(cfg.jepa))
    out.extend(render_aug_problems(cfg.render_aug))
    out.extend(walkway_anchor_problems(cfg.walkway_anchor, cfg.mixture))
    out.extend(zupt_anchor_problems(cfg.zupt_anchor, cfg.mixture))
    out.extend(names_problems(cfg.names))
    if not NAMES_WIRED and cfg.names != NamesSection():
        out.append(NAMES_NOT_WIRED)
    if cfg.tier not in TIERS:
        out.append(f"tier {cfg.tier!r}: one of {list(TIERS)}")
    out.extend(stage2_problems(cfg))
    out.extend(loss_problems(cfg))
    out.extend(alarm_problems(sc))
    out.extend(body_problems(cfg))
    return out


def body_problems(cfg):
    """Violations of the body of a run (list of str; plan docs/MODEL_OPEN_PORT_PLAN.md, step M4): model.overrides.body
    'smplh' (the default, v1) or 'open'; an open run's plausibility table the open one (train/bodies.OPEN_INPUTS), a v1
    run's never the open one; an open run is a stage-1 run (stage 2 stopped, decision A4)."""
    from . import bodies as BD
    body = BD.body_of(cfg)
    out = []
    if body not in (BD.OPEN, BD.SMPLH):
        return [f"model.overrides.body {body!r}: 'open' or 'smplh'"]
    table = str(cfg.losses.plausibility_table)
    rest = tuple(cfg.losses.sup_rest)
    if len(rest) != 2 or any(not float(x) >= 0 for x in rest):
        out.append(f"losses.sup_rest {list(rest)}: two weights (phase A, phase B), each at least 0")
    elif any(float(x) for x in rest) and body != BD.OPEN:
        out.append("losses.sup_rest is the open body's rest-joint term (step M4); a v1 run keeps it off")
    if body == BD.OPEN:
        if table != BD.OPEN_INPUTS["plausibility_table"]:
            out.append(f"losses.plausibility_table {table!r}: an open run reads {BD.OPEN_INPUTS['plausibility_table']}")
        if stage2_on(cfg):
            out.append("an open run is a stage-1 run (the stage-2 store was built on the SMPL+H up axes)")
    elif table == BD.OPEN_INPUTS["plausibility_table"]:
        out.append(f"losses.plausibility_table {table!r} is the open body's table, the run's body is smplh")
    return out


def loss_problems(cfg):
    """Violations of the stage-2 fields of the losses section (list of str, each starting with 'losses'; task S2.8):
    none at their defaults. plausibility_table a file whose joint_limits block reads (losses.joint_limits); a table
    other than the default, or any table named with a SHA-256, with plausibility_table_sha256 a lowercase hex SHA-256
    equal to the SHA-256 of the file's bytes; plausibility_windows one of losses.PLAUSIBILITY_WINDOWS,
    'synthetic_replay' only with the stage2 section enabled; hsic_strata only with the stage2 section enabled and then
    with hsic_strata_sha256 a lowercase hex SHA-256 (the table itself is read and checked by data/strata.load when the
    stage-2 store opens), and no SHA-256 without a table."""
    ls = cfg.losses
    out = []
    tsha = str(ls.plausibility_table_sha256)
    if str(ls.plausibility_table) != PLAUSIBILITY_TABLE or tsha:
        path = NM.resolve(str(ls.plausibility_table))
        try:
            LS.joint_limits(path=path)
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        except (OSError, KeyError, ValueError, TypeError) as e:
            out.append(f"losses.plausibility_table {ls.plausibility_table!r}: {type(e).__name__}: {e}")
        else:
            if not _hex64(tsha):
                out.append(f"losses.plausibility_table_sha256 {tsha!r}: the lowercase hex SHA-256 of the table "
                           f"{ls.plausibility_table!r} (needed for a table other than the default)")
            elif got != tsha:
                out.append(f"losses.plausibility_table_sha256 {tsha}: the table {ls.plausibility_table!r} has SHA-256 "
                           f"{got} (edited since the configuration named it)")
    if ls.plausibility_windows not in LS.PLAUSIBILITY_WINDOWS:
        out.append(f"losses.plausibility_windows {ls.plausibility_windows!r}: one of {list(LS.PLAUSIBILITY_WINDOWS)}")
    elif ls.plausibility_windows != "all" and not stage2_on(cfg):
        out.append(f"losses.plausibility_windows {ls.plausibility_windows!r} needs the stage2 section (a stage-1 run "
                   "holds synthetic and replay windows only)")
    sha = str(ls.hsic_strata_sha256)
    if str(ls.hsic_strata):
        if not stage2_on(cfg):
            out.append("losses.hsic_strata names a strata table while the stage2 section is off (the stage-1 store "
                       "carries one stratum)")
        if not _hex64(sha):
            out.append(f"losses.hsic_strata_sha256 {sha!r}: the lowercase hex SHA-256 of the strata table")
    elif sha:
        out.append("losses.hsic_strata_sha256 without a strata table (losses.hsic_strata)")
    return out


def _hex64(sha):
    """True for a lowercase hex SHA-256 (64 characters)."""
    return len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


def stage2_problems(cfg):
    """Violations of the stage2 section and of jepa.in_phases (list of str, each starting with 'stage2' or 'jepa'):
    none at the defaults. jepa.in_phases a non-empty subset of ('B', 'C') without repeats, 'C' only with the stage2
    section enabled. A section that is off must equal its defaults. Enabled: at least one unit, each
    '<source>:<groups>' with its source among the names section's real sources (weargait_pd:PD reads the stage-2
    store's weargait_pd folder while the replay's weargait_pd controls read the stage-1 store), one unit per source;
    0 < replay_share < 1; 0 < unit_cap <= 1; the
    pathological share 1 - replay_share at most units x unit_cap (the caps absorb it); steps at least 1; task_rows of
    TASK_ROWS; redraws a non-negative integer; freeze groups of STAGE2_FREEZE_GROUPS without repeats;
    window_cap_per_person finite within [0, 1]; the cumulative synthetic share over the three phases at full
    micro-steps (stage2_synthetic_share) at most synthetic_cap; the walkway anchor and the inertial anchor off (module
    docstring)."""
    out = []
    ph = tuple(cfg.jepa.in_phases)
    if not ph or len(set(ph)) != len(ph) or any(p not in ("B", STAGE2_PHASE) for p in ph):
        out.append(f"jepa.in_phases {list(ph)}: a non-empty subset of ['B', '{STAGE2_PHASE}'] without repeats")
    elif STAGE2_PHASE in ph and not stage2_on(cfg):
        out.append(f"jepa.in_phases names phase {STAGE2_PHASE} while the stage2 section is off")
    st = cfg.stage2
    if not stage2_on(cfg):
        if st != Stage2Section():
            out.append("stage2: the section is off (enabled false) but holds values other than its defaults")
        return out
    units = tuple(st.units)
    if not units:
        out.append("stage2.units: at least one sampling unit")
    srcs = []
    for u in units:
        src, sep, groups = str(u).partition(":")
        if not sep or not src or not groups:
            out.append(f"stage2.units {u!r}: '<source>:<groups>'")
            continue
        srcs.append(src)
        if src not in real_sources(cfg):
            out.append(f"stage2.units {u!r}: source {src!r} is not a real source of the names section")
    if len(set(srcs)) != len(srcs):
        out.append(f"stage2.units {list(units)}: one unit per source (the stage-2 store holds one folder per source)")
    r, c = float(st.replay_share), float(st.unit_cap)
    if not 0.0 < r < 1.0:
        out.append(f"stage2.replay_share {r}: within (0, 1)")
    if not 0.0 < c <= 1.0:
        out.append(f"stage2.unit_cap {c}: within (0, 1]")
    elif units and (1.0 - r) > len(units) * c + 1e-12:
        out.append(f"stage2: pathological share {1.0 - r:.4f} above {len(units)} units x unit_cap {c}")
    if int(st.steps) < 1:
        out.append(f"stage2.steps {st.steps}: at least 1 optimizer step of phase {STAGE2_PHASE}")
    if st.task_rows not in TASK_ROWS:
        out.append(f"stage2.task_rows {st.task_rows!r}: one of {list(TASK_ROWS)}")
    if int(st.redraws) < 0:
        out.append(f"stage2.redraws {st.redraws}: a non-negative integer")
    fz = tuple(st.freeze)
    if any(not isinstance(g, str) or g not in STAGE2_FREEZE_GROUPS for g in fz) or len(set(fz)) != len(fz):
        out.append(f"stage2.freeze {list(fz)}: groups of {sorted(STAGE2_FREEZE_GROUPS)} without repeats")
    wc = float(st.window_cap_per_person)
    if not (math.isfinite(wc) and 0.0 <= wc <= 1.0):
        out.append(f"stage2.window_cap_per_person {wc}: finite, within [0, 1] (0: off)")
    s = tuple(float(x) for x in cfg.mixture.synthetic_share)
    if len(s) == 2 and int(st.steps) >= 1:
        cum = stage2_synthetic_share(cfg)
        if cum > float(cfg.mixture.synthetic_cap) + 1e-12:
            out.append(f"stage2: cumulative synthetic share {cum:.4f} over phases A, B and {STAGE2_PHASE} above the "
                       f"cap {cfg.mixture.synthetic_cap}")
    if bool(cfg.walkway_anchor.enabled):
        out.append(f"stage2: the walkway anchor is not wired into phase {STAGE2_PHASE} (its table reads phase B's "
                   "weargait_pd items of the stage-1 store)")
    if bool(cfg.zupt_anchor.enabled):
        out.append(f"stage2: the inertial anchor (zupt_anchor) is not wired into phase {STAGE2_PHASE} (its table reads "
                   "phase B's weargait_pd items of the stage-1 store)")
    return out


def names_problems(nm):
    """Violations of a NamesSection (list of str, each starting with 'names'): none at the defaults (names.STAGE1);
    otherwise the refusals of names.NameTables (an empty table, a repeated or empty name, 'synthetic' not first, no
    'generic' mount family), a vocabulary file that does not read or holds no 'unknown' task, and an 'unknown'
    task away from the motion prior's condition-dropout row (prior.TASK_UNKNOWN)."""
    if nm == NamesSection():
        return []
    import yaml
    try:
        tasks = nm.tables().tasks()
    except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as e:
        return [f"names: {type(e).__name__}: {e}"]
    if tasks.index("unknown") != TASK_UNKNOWN:
        return [f"names: task 'unknown' of {nm.vocabulary_path} at id {tasks.index('unknown')}, not at "
                f"{TASK_UNKNOWN}, the motion prior's condition-dropout row"]
    return []


def jepa_problems(jp):
    """Violations of a JepaSection (list of str): weight finite and non-negative, 0 < mask_min <= mask_max < 1,
    1 <= block_min <= block_max, max_blocks and min_patches at least 1, 0 <= ema_start <= ema_end <= 1, a
    predictor of at least one block whose width splits into heads of an even dimension (rotary pairs)."""
    out = []
    if not (math.isfinite(float(jp.weight)) and float(jp.weight) >= 0.0):
        out.append(f"jepa.weight {jp.weight}: finite and non-negative")
    if not 0.0 < float(jp.mask_min) <= float(jp.mask_max) < 1.0:
        out.append(f"jepa mask shares {jp.mask_min}, {jp.mask_max}: 0 < mask_min <= mask_max < 1")
    if not 1 <= int(jp.block_min) <= int(jp.block_max):
        out.append(f"jepa blocks of {jp.block_min} to {jp.block_max} patches: 1 <= block_min <= block_max")
    if int(jp.max_blocks) < 1 or int(jp.min_patches) < 1:
        out.append(f"jepa max_blocks {jp.max_blocks} and min_patches {jp.min_patches}: at least 1")
    if not 0.0 <= float(jp.ema_start) <= float(jp.ema_end) <= 1.0:
        out.append(f"jepa EMA decay {jp.ema_start} to {jp.ema_end}: 0 <= ema_start <= ema_end <= 1")
    w, h = int(jp.pred_width), int(jp.pred_heads)
    if int(jp.pred_depth) < 1 or h < 1 or w % h or (w // h) % 2:
        out.append(f"jepa predictor of {jp.pred_depth} blocks, width {w}, {h} heads: at least one block, the width "
                   f"a multiple of the heads with an even head dimension")
    return out


RENDER_AUG_LOG_MAX = 1.0    # bound of a render_aug half-width: gains within exp(+-1)


def render_aug_problems(ra):
    """Violations of a RenderAugSection (list of str): at least one site, each a name of factory.imu_render.SITES and
    none repeated; gyro_log_max and acc_log_max finite, within [0, RENDER_AUG_LOG_MAX]."""
    out = []
    sites = tuple(ra.sites)
    if not sites:
        out.append("render_aug.sites: at least one site")
    bad = sorted({str(s) for s in sites} - set(SITES))
    if bad:
        out.append(f"render_aug.sites {bad} are not sites of factory.imu_render.SITES")
    if len(set(sites)) != len(sites):
        out.append(f"render_aug.sites {list(sites)} repeat a site")
    for k in ("gyro_log_max", "acc_log_max"):
        v = float(getattr(ra, k))
        if not (math.isfinite(v) and 0.0 <= v <= RENDER_AUG_LOG_MAX):
            out.append(f"render_aug.{k} {v}: finite, within [0, {RENDER_AUG_LOG_MAX}]")
    return out


WALKWAY_SOURCE = "weargait_pd"     # the source whose walkway the anchor reads (data/walkway_anchor.py)


def walkway_anchor_problems(wa, mixture=None):
    """Violations of a WalkwayAnchorSection (list of str): weight finite and non-negative, sd positive and finite,
    tasks names without repeats; with enabled also at least one task, persons_sha256 a lowercase hex SHA-256 and,
    given the mixture section, weargait_pd among its sources."""
    out = []
    if not (math.isfinite(float(wa.weight)) and float(wa.weight) >= 0.0):
        out.append(f"walkway_anchor.weight {wa.weight}: finite and non-negative")
    if not (math.isfinite(float(wa.sd)) and float(wa.sd) > 0.0):
        out.append(f"walkway_anchor.sd {wa.sd}: positive and finite")
    tasks = tuple(wa.tasks)
    if any(not isinstance(t, str) or not t for t in tasks) or len(set(tasks)) != len(tasks):
        out.append(f"walkway_anchor.tasks {list(tasks)}: non-empty names without repeats")
    if bool(wa.enabled):
        if not tasks:
            out.append("walkway_anchor.tasks: at least one task when enabled")
        sha = str(wa.persons_sha256)
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            out.append(f"walkway_anchor.persons_sha256 {sha!r}: a lowercase hex SHA-256 when enabled")
        if mixture is not None and WALKWAY_SOURCE not in tuple(mixture.sources):
            out.append(f"walkway_anchor enabled while {WALKWAY_SOURCE} is not a source of the mixture")
    return out


ZUPT_SOURCE = "weargait_pd"        # the source whose foot sensors the inertial anchor reads (data/zupt_anchor.py)


def zupt_anchor_problems(za, mixture=None):
    """Violations of a ZuptAnchorSection (list of str): weight finite and non-negative, sd positive and finite, edge_s
    and min_length_m finite and non-negative, tasks names without repeats; with enabled also at least one task,
    persons_sha256 a lowercase hex SHA-256 and, given the mixture section, weargait_pd among its sources."""
    out = []
    if not (math.isfinite(float(za.weight)) and float(za.weight) >= 0.0):
        out.append(f"zupt_anchor.weight {za.weight}: finite and non-negative")
    if not (math.isfinite(float(za.sd)) and float(za.sd) > 0.0):
        out.append(f"zupt_anchor.sd {za.sd}: positive and finite")
    for k in ("edge_s", "min_length_m"):
        v = float(getattr(za, k))
        if not (math.isfinite(v) and v >= 0.0):
            out.append(f"zupt_anchor.{k} {v}: finite and non-negative")
    tasks = tuple(za.tasks)
    if any(not isinstance(t, str) or not t for t in tasks) or len(set(tasks)) != len(tasks):
        out.append(f"zupt_anchor.tasks {list(tasks)}: non-empty names without repeats")
    if bool(za.enabled):
        if not tasks:
            out.append("zupt_anchor.tasks: at least one task when enabled")
        if not _hex64(str(za.persons_sha256)):
            out.append(f"zupt_anchor.persons_sha256 {za.persons_sha256!r}: a lowercase hex SHA-256 when enabled")
        if mixture is not None and ZUPT_SOURCE not in tuple(mixture.sources):
            out.append(f"zupt_anchor enabled while {ZUPT_SOURCE} is not a source of the mixture")
    return out


def check(cfg):
    """Raise ValueError listing problems(cfg); return cfg otherwise."""
    bad = problems(cfg)
    if bad:
        raise ValueError("configuration refused: " + "; ".join(bad))
    return cfg


def as_json_dict(cfg):
    """to_dict(cfg) as JSON would return it (tuples as lists), for comparisons after a round trip."""
    return json.loads(canonical_json(cfg))
