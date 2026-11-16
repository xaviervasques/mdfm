"""Training loop of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.6 and section 5, task
B4.18): phase A (synthetic only) then phase B (the mixture), the micro-step, the optimizer, the learning-rate
schedule, the EMA, logging, the memory bound, checkpoints and the bench of a phase.

Phases and steps. config.step_counts(cfg) gives the optimizer steps of each phase at full micro-steps: phase A
holds share_a of them, synthetic only (the utility test of OD9: its last checkpoint is the simulator-only
model of G3); phase B the rest, synthetic_share[1] of each micro-step's tokens synthetic and the rest from the
real sources. Each phase draws from its own data.sampler.MixtureSampler seeded cfg.seed + its index in PHASES
(A 0, B 1). A call may stop earlier (max_steps optimizer steps in that call, a checkpoint at the stop); the
schedule stays that of the full phase and a resume continues it. Phase B starts from the end of phase A only: a
call whose phases leave an unfinished phase A behind (phase B alone on a fresh trainer, or on a checkpoint
written inside phase A) raises ValueError.

Micro-step (micro_step). The sampler's Step is collated (data/collate.py, seeded by the step's seed) on the
host and moved to the device. Under a bf16 autocast (default on CUDA; float32 on the CPU): the stems (the
tokenizer) and the encoder, then the float32 posterior heads (encode, the computation of MFM.encode); on
refined steps model/refine.refine (refine_steps Adam steps on refine_points points per stream-window,
stopped gradients, the objective logged) and refine.straight_through, the amortized posterior kept for the
amortization term; one posterior draw decoded at the batch knots, one KnotBatch.kinematics call and the
kernels (forward_draw, the computation of losses.forward, the windows with a body truth posed on the drawn
betas with their gradient stopped: losses.kinematic_beta); losses.stage1_losses. The whole decoder, Hermite,
kinematics, kernels and loss terms run in float32 (the model and the losses disable autocast there; a bf16
decoder trunk turned its rounding into kinematic noise, reports/bench/phaseA_diagnosis.md, C1).
Backward of the per-window losses summed and divided by the windows of the optimizer step, so that an
optimizer step minimizes the mean over its windows (spec section 4). One torch.Generator on the device, seeded
by the step's seed, draws the refinement points and the posterior sample.

Optimizer step. micro_per_step micro-steps, all sampled before the first forward (their window count scales
the losses); gradient clip at schedule.clip (the norm before clipping is logged); AdamW over param_groups: the
weights of nn.Linear layers decay, and norms, biases, embeddings, the learned token vectors, source offsets,
log scales and the EMG decoder's constrained coefficients do not (spec section 5 names norms, biases,
embeddings and source offsets); fused AdamW when schedule.fused is 'fused', or 'auto' and fused_smoke()
passes on the device, foreach otherwise; then the EMA (decay ema_decay, torch._foreach_lerp_). A step whose
gradient norm is not finite is skipped and logged; MAX_NONFINITE skipped steps in a row raise
FloatingPointError.

Learning rate (Schedule): phase A rises linearly from 0 to lr over warmup_steps, phase B from lr_min to lr
over rewarmup_steps (the re-warmup of section 5); then a cosine from lr to lr_min over the rest of the phase.

Refinement cadence (refined): in phase B, the optimizer steps whose index within the phase is a multiple of
refine_every, every micro-step of such a step; never in phase A.

Memory (CUDA). The caching allocator is capped at MEMORY_CAP_GB (cap_memory; the scripts also set
PYTORCH_CUDA_ALLOC_CONF to ALLOC_CONF before CUDA starts); torch.cuda.mem_get_info at start (log.jsonl, event
'start'); allocated, reserved and their peaks every log_every steps; MemoryBudgetError as soon as
torch.cuda.max_memory_reserved() exceeds max_reserved_gb (1e9 bytes per GB, the unit torch reports in).

Logging (the run folder's log.jsonl, one JSON object per line; non-finite floats written as null): 'start'
(device, versions, optimizer kind, memory, config SHA-256, persons per source), 'log' every log_every optimizer
steps (each loss term and diagnostic as its window-weighted mean over the interval; KL of z per valid patch and
latent dimension; active latent dimensions, those whose posterior mean varies over the valid patches of a
micro-step with a variance above ACTIVE_VAR, averaged over the micro-steps; useful and padded tokens per
second of wall time; the time split; memory; gradient norm; learning rate; token shares per source; the
cumulative synthetic share of the run so far and its projection to the end of phase B at the rates of the
interval; nuis_aux on synthetic windows; the refinement objective drop; HSIC and its active share; health,
computed without gradient on the first micro-step of the interval by health(): the decoder's attention per
block (BodyDecoder.attention_stats: entropy, logit gap, mass near the query, residual norm), the RMS of the
raw rotation outputs (root tilt, joints; rad), the RMS of the root rates at the grid points (yaw rate rad/s,
heading-frame pelvis velocity m/s), logvar_z, the spread of mu_z between windows (of the window means) and
within windows, and on windows with a body truth the mean absolute error of mu_b against the true betas and
the ratio of the predicted rest height to the true one), 'dev' (the dev hook's dict), 'alarm', 'checkpoint',
'phase_end', 'end'. The console gets one line per log event.

Alarms (check_alarms, ALARMS; reports/bench/phaseA_diagnosis.md, fix 8, and root_motion_diagnosis.md, fix 6):
at a log, an attention entropy below 0.5 nat in any decoder block from step 500 on, a mean absolute beta error
above 2 or a mean logvar_z below -9 from step 1,000 on; at a dev event (the 'model' entry of the dev hook's
dict), a spread_ratio below 0.3 or a joint_error_z0_mm within 1 mm of joint_error_mm from step 2,000 on, a
root_speed_ratio below 0.5 or a heading_gain below 0.3 from step 5,000 on (steps within the phase). Each logs
'alarm'; with schedule.stop_on_alarm it also writes a checkpoint and raises AlarmError, as
check_synthetic_cap does. An alarm that schedule.alarm_consecutive names with k > 1 (task S2BA;
config.ALARM_CONSECUTIVE_DOC) is raised only at the k-th crossing of its reads in a row, a crossing before it
logging 'alarm_watch'; the default {} reads every alarm on one read. Phase A of 2026-09-29 would have stopped
at step 2,000, the fixed phase A of 2026-09-30 (phaseA_fix_full) at step 5,000.

Synthetic cap (check_synthetic_cap): the 60 percent cap of design section 5 holds on the run's tokens (OD9). At
each phase B log, once phase B has drawn the micro-steps of one log interval, the projection of the whole
run's synthetic share (phase A's tallies plus phase B at its tokens per micro-step so far) is compared with
mixture.synthetic_cap; above it the loop logs 'synthetic_cap', writes a checkpoint and raises
SyntheticCapError.

Checkpoints (train/checkpoint.save) every checkpoint_every optimizer steps of a phase, at each phase end and
at the stop of a shortened call, as <run>/ckpt_<phase>_<step>.pt: model, EMA, optimizer, the Schedule state
(in_phase, step, step counts, the token tallies of both phases), the sampler state of the last micro-step
consumed, the RNG states, the config and provenance.record(cfg, body store, and the window store in phase B).
Trainer.resume(path) restores all of them and refuses a checkpoint of another configuration.

Init-from (Trainer.init_from, run(init_from=path); task J2b): phase B of a new run, under this trainer's
configuration, from a checkpoint written at the end of phase A of another run (its schedule in phase A at its own
phase A step count), into a run folder that holds no checkpoint. checkpoint.load reads it with its checks (file,
config, person-list, records and lockbox hashes, the person guard, the audit; a configuration other than the
checkpoint's is allowed and listed under drift), and its training persons are checked against this run's held-out
persons. Loaded: the model weights (strict: a JEPA module of this run, absent from the checkpoint, keeps its
predictor's initialization and its target copies the loaded tokenizer and encoder, mfm.MFM.attach_jepa), the model
EMA, the optimizer's per-parameter states (init_optimizer_state: parameters new to this run start without a state,
each group's hyperparameters are this run's), the RNG states and the token tallies of phase A; the schedule is set
to phase B step 0 of this run's step counts, the sampler of phase B starts fresh (seed cfg.seed + 1) and no dev
hook runs at the start, so that with an unchanged configuration (or one that differs by a disabled jepa section or
the name) the run reproduces, bit for bit, a resume of the same checkpoint followed by phase B. The lineage, the
checkpoint's path, SHA-256 (its .sha256 file, which checkpoint.load verifies, or the file's hash), step and config
SHA-256, is logged ('init_from') and recorded under extra.init_from of the provenance of every checkpoint the run
writes; a resume of such a checkpoint keeps it, and run(resume=..., init_from=...), the form a crash resume of an
init-from job takes, resumes and refuses a checkpoint whose lineage names another source.

Prefetch (prefetch > 0): a thread samples and collates up to `prefetch` micro-steps ahead of the model (the
sampler renders on the GPU in that thread; the default CUDA stream is shared); the draws are those of the
synchronous loop and the checkpoint keeps the sampler state of the last micro-step consumed. Forward-mode AD
levels are process-global, so the renderer's render() and the main thread's forward (encoder to losses) and dev
hook take turns under one lock (LockedRenderer, FWD_AD_LOCK_DOC); backward, sampling and collate overlap.

Dev numbers (evaluate): loss terms and a heading-frame joint error on fixed micro-batches, with the root
orientation error, the joint error on the true skeleton and with z = 0, the within-window spread ratio, the
error of the constant mean pose on each window's prior-mean skeleton, and the root motion at the root-grid
points of the heading-observable truth windows (speed ratio, heading gain, lateral slope), for the dev hook.

Time split (Timer): wall seconds per part of PARTS, CUDA synchronized before each reading on timed
micro-steps (the first optimizer step of each log interval in training, every micro-step in bench): sample
(the plan, body and window reads, the guard), render (the synthetic renderer, factory/synth.py), collate,
transfer, stems, encoder, refinement, decoder, kinematics, kernels, losses, backward (the micro-step's
backward, all modules), optimizer (clip, AdamW, EMA; per optimizer step). With prefetch, sample, render and
collate overlap the rest, and the parts under the lock include the wait for it.

JEPA (optional; config section jepa, off by default; docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2; model/
jepa.py). With jepa.enabled the Trainer attaches the JEPA module to the model (mfm.MFM.attach_jepa: the EMA target
copy of the tokenizer and encoder, and the predictor; the model EMA of the checkpoints leaves it out, make_ema,
ema_update) and the optimizer holds the predictor's parameters. In phase B, micro_step (argument jepa, a JepaArgs)
runs the JEPA part between the stems and the encoder of the main pass (_jepa_part): masks from the micro-step seed,
the context pass on the tokenizer output (detached into a leaf), the predictor and the target; the JEPA
contribution weight x sum_w term_w / n_windows is differentiated at once (torch.autograd.grad: the encoder and the
predictor receive their gradient, the context graph is freed before the main pass, and the gradient at the context
input joins the main backward through the tokenizer output), so that the context activations never coexist with
the decoder's; the term enters per_window with the weight (its value; its gradient is already taken). Phase A
carries no JEPA term. After every applied phase B optimizer step the target follows the online tokenizer and
encoder with the decay jepa.ema_decay(step, phase B steps, ema_start, ema_end); a skipped step updates nothing.
The target starts from the online weights at the first phase B step (JEPA.sync, unless a loaded state dict already
did). The JEPA module rides in the model's state dict, so the checkpoints save and restore it (target, predictor,
the collapse references); a checkpoint written without it loads into a model with it (mfm.MFM.attach_jepa, State
dicts), and load_optimizer_state loads an optimizer state whose groups lack the predictor's parameters. Logged:
the 'log' record's jepa entry (jepa_loss over the windows carrying a term, their count, the masked fraction of
their patches, the masked tokens, the last EMA decay, std_target and std_pred measured without gradient on the
first micro-step of the interval, null when it holds no masked patch, their step-0 references and the collapse
ratio, the JEPA seconds per timed micro-step), and with jepa_grad_steps > 0 (Trainer) a 'jepa_grad' event at each
of the first jepa_grad_steps optimizer steps of phase B: g_J, the norm over the encoder's parameters (its 12 blocks
and final norm) of the gradient of the step's JEPA contribution, and g_R, the same for every other term, both
before clipping. Alarm jepa_collapse (ALARMS): the collapse ratio, the smaller of std_target and std_pred over
their step-0 references, below 0.01; not evaluated on an interval whose first micro-step holds no masked patch.

Render amplitude randomization (optional; config section render_aug, off by default; F2 lever 2, task F2L2,
docs/proposals/F2_LEVER2_RULE.md). With render_aug.enabled the Trainer sets its renderer's amp_aug
(factory.synth.AmplitudeAug), so that every synthetic IMU stream the samplers render scales the deviation of the
angular rate and of the specific force from their window means at the section's sites by gains drawn per window,
site and feature; the dev windows of scripts/train_stage1.py come from a renderer of their own and stay unscaled.
The 'start' record carries the section. Nothing else of the loop changes.

Walkway anchor (optional; config section walkway_anchor, off by default; F2 lever 4, task L4C,
docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md). With walkway_anchor.enabled and a window store, the Trainer builds
the anchor table (data/walkway_anchor.build_anchor_table: the walkway targets of the store's weargait_pd training
fold, whose SHA-256 must be the section's persons_sha256; the cell size refit on those persons; the persons and the
cell checked by assert_anchor_clean) or checks a given one (argument anchor_table). For each micro-step,
optimizer_step reads the segments of its real weargait_pd windows from the sampler's Step
(walkway_anchor.targets_for_step: the plan items' recording indices and the windows' t0 and duration; rows keyed by
the window's index, never a person) and micro_step (argument anchor, an AnchorArgs) computes
losses.walkway_anchor_parts on the decoded root grid of the draw: in phase B the term enters per_window with
walkway_anchor.weight; in phase A it is computed and logged only. The sampler, the collate, the renders, the
refinement and every random draw are those of a run without the section. Logged: the 'start' record's
walkway_anchor entry (the section and the table check: persons, their SHA-256, the cell size, its fit count and
persons' SHA-256, the trials left out on the clock, the input files' SHA-256) and the 'log' record's walkway_anchor
entry (the phase's weight; loss, the term's sum over the interval's windows divided by their number as every term;
weighted, the same times the weight; the anchored windows and segments; the term per anchored window; gain, the
decoded forward distance over the walkway distance summed over the segments; chord_ratio, the decoded chord over the
decoded forward distance; the anchored seconds).

Inertial anchor (optional; config section zupt_anchor, off by default; F2 lever 5, task L5,
docs/proposals/F2_LEVER5_DRAFT.md). With zupt_anchor.enabled and a window store, the Trainer builds the anchor table
(data/zupt_anchor.build_table: the version-2 ZUPT strides of the foot sensors of the store's weargait_pd training-fold
recordings of the section's tasks, read from their imu rows alone; the training fold's SHA-256 must be the section's
persons_sha256) or checks a given one (argument zupt_table). For each micro-step, optimizer_step reads the strides of
its real weargait_pd windows from the sampler's Step (zupt_anchor.targets_for_step; rows keyed by the window's index,
never a person) and micro_step (argument zupt, a ZuptArgs) computes losses.zupt_anchor_parts on the draw's latents and betas
(one extra decoder call on the anchored windows): in phase B the term enters per_window with zupt_anchor.weight; in
phase A it is computed and logged only. The sampler, the collate, the renders, the refinement, the main decode and
every random draw are those of a run without the section. Logged: the 'start' record's zupt_anchor entry (the section
and the table check: persons, their SHA-256, tasks, the minimum length, the streams read, the recordings left out, the
input files' SHA-256, the counts, the ZUPT version and constants, the table's content SHA-256) and the 'log' record's
zupt_anchor entry (the phase's weight; loss, the term's sum over the interval's windows divided by their number as
every term, and its distance and still parts the same way; weighted, the loss times the weight; the anchored windows
and strides; the term per anchored window; gain, the decoded ankle displacement over the ZUPT stride summed over the
strides; speed_at_instants, the mean decoded horizontal ankle speed at the still instants (m/s); the anchored
seconds).

Name tables (task S2.7; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.1). The Trainer builds its model from
config.model_config(cfg) (the stage-1 model itself while the config's names section is at its defaults) and, for a
config whose names section is not at its defaults, collates its batches with the section's sources and nuisance
vocabularies (collate kwargs sources and nuis_fields); at the defaults the collate call is the stage-1 one.

Phase C (task S2.7; spec section 4.6): a config with the stage2 section enabled trains a stage-2 run, its phases
(A, B, C) (config.run_phases) with the step counts config.run_steps. Phase C is phase B continued on another
mixture: its loss weights, refinement cadence, JEPA eligibility (when jepa.in_phases holds 'C') and learning rate
(re-warmup from lr_min over rewarmup_steps, then the cosine) are phase B's; its sampler (data/sampler.py, phase C,
seed cfg.seed + 2) reads a sampler.Stage2Stores (the stage-1 root as the healthy replay, the stage-2 root for the
units). A stage-2 run never trains phase A or B: it starts from Trainer.init_stage2(source, plan) and continues
with resume. init_stage2 reads a checkpoint written at the end of phase B of a stage-1 run (its schedule in phase B
at its own phase B step count; refused otherwise, and a phase C checkpoint too) with checkpoint.load under this
run's configuration, so that a RELEASE run refuses an INTERNAL source (provenance.ProvenanceError) and the source's
training persons are checked against this run's held-out persons at its tier; it refuses a configuration that departs
from the source's recipe (stage2_recipe: the model, collate, losses, render_aug and jepa sections, jepa.in_phases
apart, the schedule outside its operational fields, the replay rules of the mixture and the A and B step counts) and
logs the other differences as recipe_drift; it migrates the model, the EMA, the
JEPA target and the optimizer moments from the source's name tables to the run's (train/migrate.py: the plan of
migrate.plan_for, the codes of the stage-1 replay store and layouts deciding which nuisance rows start as a copy of
'unknown', the new task rows by stage2.task_rows; a given plan must name the source's and the run's tables), loads
them strictly, sets the optimizer's per-parameter states (each group's hyperparameters this run's), carries the
token tallies of phases A and B (the 60 percent cap is read over stages 1 and 2 together, check_synthetic_cap) and
the RNG states, and sets the schedule at phase C step 0. Its lineage, in the form of migrate.migrate's (extra
'migration' of the provenance of every checkpoint the run writes: migrated_from {path, SHA-256, step, tier,
config_sha256, name, steps}, both name-table SHA-256, the plan and its SHA-256, the codes, the code SHA-256), is
logged ('init_stage2'); run(resume=..., init_stage2=...) refuses a resumed checkpoint whose lineage names another
source SHA-256 or another plan SHA-256 than the plan of that source under this configuration (check_stage2_lineage).

Stage 2 without body truth (task S2.8; spec section 4.7). HSIC strata: a stage-2 run whose losses.hsic_strata names a
strata table reads a stage-2 store opened with that table (data/strata.py; _check_stores refuses a store attaching
another table, or none, and a replay store attaching one), whose windows carry their person's hsic_stratum into
Targets and losses.hsic. Plausibility: losses.plausibility_table names the joint-limit table of the term; under
losses.plausibility_windows 'synthetic_replay' the items of phase C mark the windows that are not of a stage-2 unit
(plausibility_marks, Targets.plausibility_on) and the term weighs on those only. Dev numbers: evaluate(limits=...)
adds the share of decoded knots outside a table; the dev hook of scripts/train_stage1.py logs, per stage-2 unit, the
kernel NLL and that share on fixed windows of its val fold (None for a unit without one) and of its training persons,
and the alarm unit_nll_rise (ALARMS) counts the units whose dev NLL rises over UNIT_NLL_READS dev reads while their
train NLL falls (unit_nll_rising).

Phase C freeze and window cap (task S2B, the recipe of run S2b approved on 2026-10-12, reports/gpu_queue/decisions.md;
both off by default, so that every config and checkpoint written before keeps its behaviour). stage2.freeze names
groups of config.STAGE2_FREEZE_GROUPS ('decoder', 'body_heads'): the Trainer of such a config sets requires_grad
False on their parameters before it builds the EMA and the optimizer (freeze_parameters), so that param_groups leaves
them out of AdamW and of the gradient clip and no gradient is computed for them; the EMA of a frozen parameter is not
updated (ema_update with skip): it keeps the value loaded from the start checkpoint, so that the model the readings
decode (the EMA) holds the start's EMA decoder and body heads bit for bit; init_stage2 drops the frozen parameters'
optimizer states from the migrated optimizer state (freeze_optimizer_state) and a resume loads a checkpoint written
under the same freeze as it is. The 'start' record's stage2 entry and the 'init_stage2' record carry the trainable and
frozen parameter and tensor counts per group. stage2.window_cap_per_person caps each unit's share of the phase C
windows (data/sampler.py, enforce_window_caps); the 'log' records of such a run carry window_share_phase, every
budget key's share of the windows phase C has drawn so far (the schedule's tallies).

The loop reads no lockbox and writes nothing under reports/: the run folder may not sit in this tree outside
cache/ and runs/ (checkpoint.check_path). Every window reaches it through the sampler's guard
(provenance.assert_clean with the held-out persons) and collate's CODY refusal. mdfm.train never imports
mdfm.gates: dev evaluation is a hook the caller passes (scripts/train_stage1.py).
"""
import contextlib
import copy
import datetime
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

from .. import paths
from ..data import collate as CO
from ..data import sampler as S
from ..data import walkway_anchor as WA
from ..data import zupt_anchor as ZA
from ..model import jepa as JP
from ..model import mfm as M
from ..model.posterior import sample, subject_pool
from ..model.refine import refine, straight_through
from . import checkpoint as CK
from . import config as CF
from . import losses as LS
from . import provenance as PV

PHASES = ("A", "B")
STAGE2_PHASE = CF.STAGE2_PHASE             # phase C, a stage-2 run's (module docstring)
LOG_PHASE_KEY = "phase"                  # a log record's training phase (A, B or C), not a data label
ALL_PHASES = PHASES + (STAGE2_PHASE,)
PARTS = ("sample", "render", "collate", "transfer", "stems", "encoder", "refinement", "decoder", "kinematics",
         "kernels", "losses", "backward", "optimizer")
HOST_PARTS = ("sample", "render", "collate")
ACTIVE_VAR = 0.01          # variance over patches of a posterior mean for an active latent dimension
MAX_NONFINITE = 3          # skipped optimizer steps in a row before FloatingPointError
GB = 1e9                   # bytes per GB (torch and the bench report in 1e9-byte units)
RUNS_ENV = "MDFM_RUNS"
RUNS_TARGET = "<store>/runs"
FORMAT = "mdfm-loop/1"
# the diagnostics of losses.stage1_losses (tests/test_loop_smoke.py checks the copy)
SIDE_TERMS = ("sup_pos_band", "sup_root_xz", "sup_root_y", "sup_root_rate_u", "sup_root_rate_yaw", "plaus_limits",
              "plaus_penetration", "plaus_skate", "bone_spread", "hsic_active")
TERM_KEYS = tuple(LS.TERMS) + SIDE_TERMS + ("per_window",)
# training alarms (module docstring): name -> (source, 'below' or 'above', line, first step within the phase)
ALARMS = {"attention_entropy": ("log", "below", 0.5, 500),      # nats, the lowest decoder block
          "beta_error": ("log", "above", 2.0, 1000),            # mean |mu_b - beta_true| on truth windows
          "logvar_z": ("log", "below", -9.0, 1000),             # mean logvar_z over valid patches
          "spread_ratio": ("dev", "below", 0.3, 2000),          # decoded over true within-window spread
          "z0_margin_mm": ("dev", "below", 1.0, 2000),          # joint_error_z0_mm - joint_error_mm
          "root_speed_ratio": ("dev", "below", 0.5, 5000),      # decoded over true pelvis speed, moving grid points
          "heading_gain": ("dev", "below", 0.3, 5000),          # slope of decoded on true yaw about the centre
          "jepa_collapse": ("log", "below", 0.01, 0),           # JEPA spreads over their step-0 values (jepa runs)
          "unit_nll_rise": ("dev", "above", 0.0, 0)}            # stage-2 units whose dev NLL rises (unit_nll_rising)
UNIT_NLL_READS = 3         # [cfg] dev reads over which a unit's dev NLL rises while its train NLL falls (task S2.8)
ROOT_MOVING = 0.3          # m/s, true pelvis speed of a moving grid point (evaluate: root_speed_ratio)
ROOT_FAR_S = 3.0           # s from the centre knot, beyond which root_speed_ratio_far reads
HEADING_WITHIN = math.radians(120.0)   # true yaw about the centre within which heading_gain and heading_corr read
HEALTH_QUERIES = 256       # knots per micro-step read by the attention statistics of health()


MEMORY_CAP_GB = 12.0       # caching-allocator cap (cap_memory), the peak target of the B4.18 bench
ALLOC_CONF = "garbage_collection_threshold:0.8"   # PYTORCH_CUDA_ALLOC_CONF the scripts set before CUDA starts


class MemoryBudgetError(RuntimeError):
    """torch.cuda.max_memory_reserved() above schedule.max_reserved_gb."""


class SyntheticCapError(RuntimeError):
    """The projected synthetic share of the run above mixture.synthetic_cap (check_synthetic_cap)."""


class AlarmError(RuntimeError):
    """A training alarm (ALARMS, check_alarms) with schedule.stop_on_alarm."""


def unit_nll_rising(history, reads=UNIT_NLL_READS):
    """Units whose dev NLL rises over the last `reads` dev reads while their train NLL falls (task S2.8; spec
    docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7): history {unit: [(dev NLL, train NLL), ...]} in read order; a unit
    is listed when it holds at least `reads` reads, finite, and over the last `reads` each dev NLL is above the one
    before and each train NLL below the one before. Sorted unit names."""
    out = []
    for unit, h in history.items():
        last = [(float(d), float(t)) for d, t in list(h)[-int(reads):] if d is not None and t is not None]
        if len(last) < int(reads) or not all(math.isfinite(d) and math.isfinite(t) for d, t in last):
            continue
        if all(b[0] > a[0] and b[1] < a[1] for a, b in zip(last, last[1:])):
            out.append(str(unit))
    return sorted(out)


def alarm_values(rec, source):
    """{alarm name: value or None} that check_alarms reads: source 'log', rec a 'log' record (its health and jepa
    entries); source 'dev', rec the dev hook's dict (its 'model' entry when present); {} for another source."""
    if source == "log":
        h = (rec or {}).get("health") or {}
        ents = [b["entropy"] for b in h.get("decoder") or []]
        return {"attention_entropy": min(ents) if ents else None, "beta_error": h.get("beta_error"),
                "logvar_z": h.get("logvar_z"),
                "jepa_collapse": ((rec or {}).get("jepa") or {}).get("collapse_ratio")}
    if source == "dev":
        d = (rec or {}).get("model", rec or {})
        e, e0 = d.get("joint_error_mm"), d.get("joint_error_z0_mm")
        return {"spread_ratio": d.get("spread_ratio"),
                "z0_margin_mm": None if e is None or e0 is None else float(e0) - float(e),
                "root_speed_ratio": d.get("root_speed_ratio"), "heading_gain": d.get("heading_gain"),
                "unit_nll_rise": d.get("unit_nll_rise")}
    return {}


def alarm_streaks(log_path, phase, step, names):
    """{name: crossings in a row} for each alarm of `names` (schedule.alarm_consecutive, task S2BA): its reads in the
    run folder's log.jsonl ('log' records, or the 'dev' entry of 'dev' events, by the alarm's source) of `phase` before
    `step`, the last record of each step, in step order, from the alarm's first step on; a read without a value is
    skipped; the count of the last reads in a row that cross the line (0 without a log)."""
    out = {str(n): 0 for n in names}
    if not out or not log_path or not os.path.exists(log_path):
        return out
    reads = {"log": {}, "dev": {}}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = r.get("event")
            if ev not in reads or r.get(LOG_PHASE_KEY) != phase or int(r.get("step", -1)) >= int(step):
                continue
            reads[ev][int(r["step"])] = alarm_values(r if ev == "log" else r.get("dev"), ev)
    for name in out:
        src, side, line, first = ALARMS[name]
        n = 0
        for s in sorted(reads.get(src, {})):
            v = reads[src][s].get(name)
            if s < first or v is None:
                continue
            n = n + 1 if (side == "below" and v < line) or (side == "above" and v > line) else 0
        out[name] = n
    return out


def cap_memory(device, gb):
    """Cap the caching allocator of a CUDA device at gb (1e9 bytes) with
    torch.cuda.set_per_process_memory_fraction: past the cap it frees its unused cached blocks before asking the
    driver for more, and with ALLOC_CONF (garbage_collection_threshold) it reclaims them from 80 percent of the
    cap on. Variable shapes from one micro-step to the next otherwise leave the pool well above the allocated
    peak (reports/bench/stage1_pilot.md). Returns gb, or None on a CPU device or for gb None."""
    dev = torch.device(device)
    if gb is None or dev.type != "cuda":
        return None
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(gb) * GB / total), idx)
    return float(gb)


# ---------------------------------------------------------------- small pieces
def default_amp(device):
    """bf16 autocast on CUDA, float32 elsewhere."""
    return torch.device(device).type == "cuda"


def amp_context(device, amp):
    """The bf16 autocast of the stems, encoder and prior (enabled when amp is true); the decoder disables it
    and runs whole in float32 (model/decoder.py; reports/bench/phaseA_diagnosis.md, C1)."""
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bool(amp))


def phase_seed(cfg, phase):
    """Seed of the sampler of a phase: cfg.seed plus the phase's index in ALL_PHASES (A 0, B 1, C 2)."""
    return int(cfg.seed) + ALL_PHASES.index(phase)


def loss_phase(phase):
    """The phase whose loss weights a phase reads: phase B's for phase C (module docstring), else itself."""
    return "B" if phase == STAGE2_PHASE else phase


def refined(phase, step, every):
    """True for the optimizer steps that refine: phase B or C, step (0-based within the phase) a multiple of every
    (every <= 0: never)."""
    return phase in ("B", STAGE2_PHASE) and int(every) > 0 and int(step) % int(every) == 0


def default_run_dir(cfg, now=None):
    """<root>/<name>_<YYYYMMDD>_s<seed>: root MDFM_RUNS when set, else <store>/runs when <store> exists, else
    cache/runs of this tree (gitignored)."""
    root = os.environ.get(RUNS_ENV) or (RUNS_TARGET if os.path.isdir(os.path.dirname(RUNS_TARGET))
                                        else os.path.join(paths.CACHE, "runs"))
    day = (now or datetime.date.today()).strftime("%Y%m%d")
    return os.path.join(root, f"{cfg.name}_{day}_s{cfg.seed}")


def _finite(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return _finite(x.item())
    return x


def jsonl(path, obj):
    """Append obj as one JSON line (non-finite floats as null) to path."""
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(_finite(obj), sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- timing
class Timer:
    """Wall seconds per part (PARTS). sync: CUDA synchronize before each reading (device must be CUDA)."""

    def __init__(self, device=None, sync=False):
        dev = torch.device(device) if device is not None else None
        self.sync = bool(sync) and dev is not None and dev.type == "cuda"
        self.device = dev
        self.s = dict.fromkeys(PARTS, 0.0)
        self.t = None

    def _sync(self):
        if self.sync:
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.t = time.perf_counter()

    def lap(self, part):
        """Charge the time since the last reading to part."""
        self._sync()
        t = time.perf_counter()
        if self.t is not None:
            self.s[part] += t - self.t
        self.t = t

    def add(self, part, seconds):
        self.s[part] += float(seconds)


def _lap(timer):
    return timer.lap if timer is not None else (lambda part: None)


# ---------------------------------------------------------------- optimizer, schedule, EMA
def param_groups(model, weight_decay):
    """[{params: decaying, weight_decay}, {params: the rest, weight_decay: 0}] (module docstring): a parameter
    decays when it is the weight of an nn.Linear."""
    decay, rest = [], []
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if p.requires_grad:
                (decay if isinstance(mod, nn.Linear) and name == "weight" else rest).append(p)
    return [{"params": decay, "weight_decay": float(weight_decay)}, {"params": rest, "weight_decay": 0.0}]


def frozen_names(model, groups):
    """{group: [names]} of the model's parameters (named_parameters order, a JEPA module's left out) that the freeze
    groups hold (config.STAGE2_FREEZE_GROUPS: name prefixes). ValueError for an unknown group or a group that holds no
    parameter of the model."""
    jm = _jepa_of(model)
    skip = set() if jm is None else {id(p) for p in jm.parameters()}
    names = [n for n, p in model.named_parameters() if id(p) not in skip]
    out = {}
    for g in groups:
        if g not in CF.STAGE2_FREEZE_GROUPS:
            raise ValueError(f"freeze group {g!r}: one of {sorted(CF.STAGE2_FREEZE_GROUPS)}")
        pre = CF.STAGE2_FREEZE_GROUPS[g]
        out[g] = [n for n in names if n.startswith(pre)]
        if not out[g]:
            raise ValueError(f"freeze group {g!r} ({', '.join(pre)}) holds no parameter of the model")
    return out


def freeze_parameters(model, groups):
    """Set requires_grad False on the parameters of the freeze groups (frozen_names; module docstring, Phase C freeze
    and window cap); returns {groups, frozen {group: {parameters, tensors}}, frozen_parameters, frozen_tensors,
    trainable_parameters, trainable_tensors, names (the frozen parameter names, sorted)}. The counts leave a JEPA
    module out."""
    by = frozen_names(model, groups)
    names = {n for v in by.values() for n in v}
    params = dict(model.named_parameters())
    for n in names:
        params[n].requires_grad_(False)
    jm = _jepa_of(model)
    skip = set() if jm is None else {id(p) for p in jm.parameters()}
    own = [(n, p) for n, p in params.items() if id(p) not in skip]
    return {"groups": list(groups),
            "frozen": {g: {"parameters": int(sum(params[n].numel() for n in v)), "tensors": len(v)}
                       for g, v in by.items()},
            "frozen_parameters": int(sum(params[n].numel() for n in names)), "frozen_tensors": len(names),
            "trainable_parameters": int(sum(p.numel() for n, p in own if p.requires_grad)),
            "trainable_tensors": int(sum(1 for n, p in own if p.requires_grad)), "names": sorted(names)}


def freeze_optimizer_state(state, names, frozen, want):
    """The optimizer state dict `state` (its groups holding the parameters `names`, [[names of group 0], ...] in the
    order of its indices, migrate.param_names of the model it was written for, every parameter trainable) restricted
    to the parameters outside `frozen` and renumbered, so that it fits an optimizer whose groups hold the parameters
    `want` (migrate.param_names of the frozen model; as in load_optimizer_state, parameters added at the end of a group,
    such as a JEPA predictor's, start without a state); returns (state, parameters dropped). ValueError when the
    groups do not hold `names` or the kept parameters of a group are not the first ones of its `want` group."""
    groups = state["param_groups"]
    flat = [n for g in names for n in g]
    if len(groups) != len(names) or any(len(g["params"]) != len(nm) for g, nm in zip(groups, names)):
        raise ValueError(f"optimizer state of groups {[len(g['params']) for g in groups]}, the names give "
                         f"{[len(nm) for nm in names]}")
    kept = [[flat[int(i)] for i in g["params"] if flat[int(i)] not in frozen] for g in groups]
    if len(kept) != len(want) or any(list(w)[:len(k)] != k for k, w in zip(kept, want)):
        raise ValueError("the optimizer state's parameters outside the frozen groups differ from the optimizer's")
    pos = {n: i for i, n in enumerate(n for g in kept for n in g)}
    out_groups = [dict(g, params=[pos[n] for n in k]) for g, k in zip(groups, kept)]
    st = {pos[flat[int(i)]]: v for i, v in state["state"].items() if flat[int(i)] in pos}
    return {"state": st, "param_groups": out_groups}, len(flat) - len(pos)


def fused_smoke(device, steps=3):
    """True when fused AdamW runs on the device and equals foreach AdamW after `steps` steps on a small problem
    with two parameter groups (rtol 1e-5, atol 1e-6); False on a CPU device or on any error."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    g = torch.Generator().manual_seed(0)
    w0 = [torch.randn(64, 32, generator=g), torch.randn(32, generator=g)]
    grads = [[torch.randn(w.shape, generator=g) for w in w0] for _ in range(int(steps))]
    out = []
    try:
        for fused in (True, False):
            ps = [w.clone().to(dev).requires_grad_(True) for w in w0]
            kw = {"fused": True} if fused else {"foreach": True}
            opt = torch.optim.AdamW([{"params": ps[:1], "weight_decay": 0.05}, {"params": ps[1:], "weight_decay": 0.0}],
                                    lr=1e-3, betas=(0.9, 0.95), eps=1e-8, **kw)
            for gr in grads:
                for p, x in zip(ps, gr):
                    p.grad = x.to(dev)
                opt.step()
            out.append([p.detach().cpu() for p in ps])
    except (RuntimeError, TypeError, ValueError):
        return False
    return all(torch.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(*out))


def build_optimizer(model, sc, device):
    """(AdamW, kind 'fused' or 'foreach') of schedule section sc (module docstring)."""
    kind = str(sc.fused).strip().lower()
    dev = torch.device(device)
    if kind == "auto":
        use = fused_smoke(dev)
    elif kind in ("fused", "true"):
        if dev.type != "cuda":
            raise ValueError("fused AdamW needs a CUDA device")
        use = True
    elif kind in ("foreach", "false"):
        use = False
    else:
        raise ValueError(f"schedule.fused {sc.fused!r}: auto, fused or foreach")
    kw = {"fused": True} if use else {"foreach": True}
    opt = torch.optim.AdamW(param_groups(model, sc.weight_decay), lr=float(sc.lr), betas=tuple(sc.betas),
                            eps=float(sc.eps), **kw)
    return opt, ("fused" if use else "foreach")


class Schedule:
    """Learning rate and position of the loop (module docstring): in_phase (the current phase), step (optimizer
    steps done in it), steps (optimizer steps of each phase of run_phases, default PHASES: A and B; a stage-2 run's
    ALL_PHASES), tallies {phase: {tokens: {source: n}, windows: {source: n}, micro: n, padded: n}}. Phase C's
    learning rate is phase B's form (re-warmup from lr_min, then the cosine)."""

    def __init__(self, sc, steps, phases=PHASES):
        self.sc = sc
        self.steps = tuple(int(s) for s in steps)
        self.run_phases = tuple(phases)
        if len(self.steps) != len(self.run_phases):
            raise ValueError(f"{len(self.steps)} step counts for the phases {self.run_phases}")
        self.in_phase = self.run_phases[0]
        self.step = 0
        self.tallies = {p: {"tokens": {}, "windows": {}, "micro": 0, "padded": 0} for p in self.run_phases}

    def n(self, phase=None):
        return self.steps[self.run_phases.index(self.in_phase if phase is None else phase)]

    def lr(self, phase=None, step=None):
        """Learning rate of optimizer step `step` (0-based) of `phase` (default the current ones)."""
        phase = self.in_phase if phase is None else phase
        s = self.step if step is None else int(step)
        sc, n = self.sc, self.n(phase)
        lo, hi = float(sc.lr_min), float(sc.lr)
        warm, start = (int(sc.warmup_steps), 0.0) if phase == "A" else (int(sc.rewarmup_steps), lo)
        warm = max(0, min(warm, n))
        if s < warm:
            return start + (hi - start) * (s + 1) / warm
        u = min(1.0, (s - warm) / max(n - warm, 1))
        return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * u))

    def apply(self, opt):
        lr = self.lr()
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    def count(self, phase, step, mb):
        """Add a materialized sampler Step and its MicroBatch to the tallies of phase."""
        t = self.tallies[phase]
        for s, n in step.tokens.items():
            t["tokens"][s] = t["tokens"].get(s, 0) + int(n)
        for it in step.plan.items:
            t["windows"][it.source] = t["windows"].get(it.source, 0) + 1
        t["micro"] += 1
        t["padded"] += int(mb.padded)

    def synthetic_share(self):
        """Synthetic share of every token the run has consumed (both phases)."""
        syn = sum(t["tokens"].get(S.SYNTHETIC, 0) for t in self.tallies.values())
        tot = sum(sum(t["tokens"].values()) for t in self.tallies.values())
        return syn / tot if tot else float("nan")

    def state_dict(self):
        return {"format": FORMAT, "in_phase": self.in_phase, "step": int(self.step), "steps": list(self.steps),
                "tallies": copy.deepcopy(self.tallies)}

    def load_state_dict(self, state):
        if list(state["steps"]) != list(self.steps):
            raise ValueError(f"schedule of {state['steps']} steps, this run has {list(self.steps)}")
        if str(state["in_phase"]) not in self.run_phases:
            raise ValueError(f"schedule in phase {state['in_phase']}, this run has the phases {self.run_phases}")
        self.in_phase, self.step = str(state["in_phase"]), int(state["step"])
        self.tallies = copy.deepcopy(state["tallies"])


def _jepa_of(model):
    """The model's JEPA module (mfm.MFM.attach_jepa), or None."""
    jm = model._modules.get("jepa") if isinstance(model, nn.Module) else None
    return jm if isinstance(jm, nn.Module) else None


def make_ema(model):
    """A copy of the model for the EMA of its weights, without gradients; a JEPA module (mfm.MFM.attach_jepa) is
    left out of the copy (its jepa is None): the EMA is the model the evaluations read."""
    jm = _jepa_of(model)
    if jm is None:
        ema = copy.deepcopy(model)
    else:
        model.jepa = None
        try:
            ema = copy.deepcopy(model)
        finally:
            model.jepa = jm
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def _ema_followed(model):
    """(parameters, buffers) of the model that its EMA follows: all of them, the JEPA module's excepted."""
    jm = _jepa_of(model)
    if jm is None:
        return [p.detach() for p in model.parameters()], list(model.buffers())
    skip = {id(x) for x in jm.parameters()} | {id(x) for x in jm.buffers()}
    return ([p.detach() for p in model.parameters() if id(p) not in skip],
            [b for b in model.buffers() if id(b) not in skip])


@torch.no_grad()
def ema_update(ema, model, decay, skip=None):
    """ema <- decay ema + (1 - decay) model for every parameter; buffers copied (the JEPA module of the model,
    which the EMA does not hold, left out). skip: parameter names whose EMA is not updated (the frozen parameters
    of a phase C freeze keep their loaded EMA value; module docstring); None or empty: every parameter."""
    params, buffers = _ema_followed(model)
    if skip:
        pairs = [(e, m) for (n, e), m in zip(ema.named_parameters(), params) if n not in skip]
        torch._foreach_lerp_([e for e, _ in pairs], [m for _, m in pairs], 1.0 - float(decay))
    else:
        torch._foreach_lerp_(list(ema.parameters()), params, 1.0 - float(decay))
    for b_e, b_m in zip(ema.buffers(), buffers):
        b_e.copy_(b_m)


def load_optimizer_state(opt, state):
    """Load an optimizer state_dict whose parameter groups are prefixes of opt's groups (the same parameters first,
    in the same order; parameters added at the end of a group, such as the JEPA predictor's, start without a
    state): the stored per-parameter states are moved to the current indices and each group's hyperparameters
    taken from the state. ValueError when the group counts differ, a stored group is longer than the current
    one, or a stored tensor state does not fit its parameter's shape."""
    cur = opt.state_dict()
    old_groups, new_groups = state["param_groups"], cur["param_groups"]
    if len(old_groups) != len(new_groups):
        raise ValueError(f"optimizer state of {len(old_groups)} groups, the optimizer has {len(new_groups)}")
    params = [p for g in opt.param_groups for p in g["params"]]
    index = {i: p for i, p in zip([i for g in new_groups for i in g["params"]], params)}
    moved, groups = {}, []
    for go, gn in zip(old_groups, new_groups):
        if len(go["params"]) > len(gn["params"]):
            raise ValueError(f"a stored group of {len(go['params'])} parameters, the optimizer's holds "
                             f"{len(gn['params'])}")
        for io, inew in zip(go["params"], gn["params"]):
            st = state["state"].get(io)
            if st is None:
                continue
            p = index[inew]
            for k, v in st.items():
                if torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != tuple(p.shape):
                    raise ValueError(f"state {k} of shape {tuple(v.shape)} for a parameter of {tuple(p.shape)}")
            moved[inew] = st
        groups.append(dict(go, params=list(gn["params"])))
    opt.load_state_dict({"state": moved, "param_groups": groups})


def init_optimizer_state(opt, state):
    """The per-parameter states of an optimizer state_dict loaded into opt as load_optimizer_state loads them (the
    stored groups prefixes of opt's groups; parameters added at the end of a group start without a state), each
    group's hyperparameters kept as opt was built (the run's configuration) instead of taken from the state; returns
    (parameters with a loaded state, parameters of opt). ValueError as load_optimizer_state."""
    cur = opt.state_dict()["param_groups"]
    old = state["param_groups"]
    if len(old) != len(cur):
        raise ValueError(f"optimizer state of {len(old)} groups, the optimizer has {len(cur)}")
    groups = [dict({k: v for k, v in gn.items() if k != "params"}, params=list(go["params"]))
              for go, gn in zip(old, cur)]
    load_optimizer_state(opt, {"state": state["state"], "param_groups": groups})
    return len(opt.state), sum(len(g["params"]) for g in opt.param_groups)


def checkpoint_sha256(path):
    """(SHA-256 of a checkpoint file, 'sidecar' or 'file'): the value of its .sha256 file when present
    (checkpoint.load verifies it), else the hash of the file."""
    p = os.path.abspath(str(path))
    side = p + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            return f.read().split()[0], "sidecar"
    return PV.sha256_file(p), "file"


# ---------------------------------------------------------------- the start recipe of a stage-2 run
RECIPE_SECTIONS = ("model", "collate", "losses", "render_aug", "jepa")   # inherited unchanged (jepa: in_phases apart)
SCHEDULE_FREE = ("checkpoint_every", "log_every", "dev_every", "max_reserved_gb", "stop_on_alarm", "fused",
                 "tokens_total", "share_a",   # operational, or compared through the A and B step counts
                 "alarm_consecutive")         # how alarms are read, like stop_on_alarm (task S2BA, amendment 3 of S2c)
MIXTURE_RULES = ("synthetic_share", "synthetic_cap", "source_cap", "sources")   # the replay's rules


def _section_diff(a, b, skip=()):
    da, db = CF.to_dict(a), CF.to_dict(b)
    return sorted(k for k in set(da) | set(db) if k not in skip and da.get(k) != db.get(k))


def stage2_recipe(src, cfg, src_steps):
    """(departures, drift) of the stage-2 RunConfig cfg from the RunConfig src of its start checkpoint, whose phases A
    and B ran src_steps optimizer steps (module docstring, Phase C; spec section 4.6: a stage-2 config inherits the
    start recipe's sections unchanged). departures, refused by Trainer.init_stage2: a field that differs in a section
    of RECIPE_SECTIONS (jepa.in_phases apart), in schedule outside SCHEDULE_FREE, in the replay rules of the mixture
    (MIXTURE_RULES), and A and B step counts (config.run_steps) other than src_steps. drift, logged: the other fields
    that differ, schedule's SCHEDULE_FREE, the mixture's layouts and batching, walkway_anchor, zupt_anchor, seed and
    tier, the stage-2 fields of the losses section (config.STAGE2_LOSS_FIELDS: the plausibility table and windows, the
    HSIC strata; task S2.8), as 'section.field' (name, status, names and stage2 differ by construction)."""
    out, drift = [], []
    for sec in RECIPE_SECTIONS:
        skip = ("in_phases",) if sec == "jepa" else tuple(CF.STAGE2_LOSS_FIELDS) if sec == "losses" else ()
        out += [f"{sec}.{k}" for k in _section_diff(getattr(src, sec), getattr(cfg, sec), skip)]
    drift += [f"losses.{k}" for k in _section_diff(src.losses, cfg.losses) if k in CF.STAGE2_LOSS_FIELDS]
    sched = _section_diff(src.schedule, cfg.schedule)
    out += [f"schedule.{k}" for k in sched if k not in SCHEDULE_FREE]
    drift += [f"schedule.{k}" for k in sched if k in SCHEDULE_FREE]
    mix = _section_diff(src.mixture, cfg.mixture)
    out += [f"mixture.{k}" for k in mix if k in MIXTURE_RULES]
    drift += [f"mixture.{k}" for k in mix if k not in MIXTURE_RULES]
    drift += [f"walkway_anchor.{k}" for k in _section_diff(src.walkway_anchor, cfg.walkway_anchor)]
    drift += [f"zupt_anchor.{k}" for k in _section_diff(src.zupt_anchor, cfg.zupt_anchor)]
    drift += [k for k in ("seed", "tier") if getattr(src, k) != getattr(cfg, k)]
    want = [int(x) for x in CF.run_steps(cfg)[:len(PHASES)]]
    if [int(x) for x in src_steps] != want:
        out.append(f"steps of phases A and B {[int(x) for x in src_steps]}, this configuration's {want}")
    return out, drift


# ---------------------------------------------------------------- the forward pass, timed
def encode(model, batch, timer=None, hook=None):
    """MFM.encode(batch), timed: stems (the tokenizer), encoder (the encoder and the posterior heads); hook(x),
    when given, is called with the tokenizer output between the two (the JEPA part of micro_step)."""
    lap = _lap(timer)
    model._check(batch)
    tb = batch.tokens
    x = model.tokenizer(tb)
    lap("stems")
    if hook is not None:
        hook(x)
    h = model.encoder(x, tb.t.to(x.device), tb.mask().to(x.device))
    post = model.posterior_from_hidden(h, batch)
    lap("encoder")
    return post


def forward_draw(model, batch, post, generator=None, timer=None, truth_windows=None):
    """(losses.BodyDraw, pred) of losses.forward(model, batch, post, generator, truth_windows=truth_windows),
    timed: decoder (the draw of z and the pooled beta, the decoder, the dense knot buffers), kinematics,
    kernels. truth_windows (W,) bool: windows with a body truth, whose kinematics use the drawn betas with
    their gradient stopped (losses.kinematic_beta)."""
    lap = _lap(timer)
    z = sample(post.mu_z, post.logvar_z, generator=generator) * post.mask[..., None]
    mu_b, lv_b = subject_pool(post.mu_b, post.logvar_b, batch.person.to(post.mu_b.device))
    beta = sample(mu_b, lv_b, generator=generator)
    out = model.decode(z, beta, batch.knots, batch.tokens.n_patches.to(z.device), batch.lengths.to(z.device))
    dense = model.dense(out, LS.kinematic_beta(beta, truth_windows), batch)
    lap("decoder")
    kin = model.kinematics(dense, batch)
    lap("kinematics")
    pred = model.render(dense, batch, post.nuis, kin)
    lap("kernels")
    return LS.BodyDraw(z, beta, out, dense, kin), pred


def _active_dims(post):
    """Latent dimensions whose posterior mean has a variance above ACTIVE_VAR over the valid patches."""
    mu = post.mu_z.detach().float()[post.mask]
    if mu.shape[0] < 2:
        return 0
    return int((mu.var(0) > ACTIVE_VAR).sum())


@dataclass
class MicroStats:
    """One micro-step: windows W, synthetic windows, sums over windows of every TERM_KEYS entry, nuis_aux summed
    over synthetic windows, valid latent patches, active latent dimensions, useful and padded tokens, bucket
    row length, the refinement objective (first and last value, None unrefined), the time split; jepa: the JEPA
    part's numbers (_jepa_part), None without it; anchor: the walkway anchor's sums (anchor_sums), None without
    it; zupt: the inertial anchor's sums (zupt_sums), None without it."""
    W: int
    synthetic: int
    sums: dict
    nuis_aux_synthetic: float
    patches: int
    active: int
    useful: int
    padded: int
    row_len: int
    refine: tuple = None
    time: dict = field(default_factory=dict)
    finite: bool = True
    jepa: dict = None
    anchor: dict = None
    zupt: dict = None


@dataclass
class AnchorArgs:
    """The walkway anchor of a micro-step (micro_step, module docstring): targets (walkway_anchor.AnchorTargets),
    weight (the phase's: walkway_anchor.weight in phase B, 0 in phase A) and sd (m/s)."""
    targets: object
    weight: float = 0.0
    sd: float = LS.WALKWAY_SD


ANCHOR_SUMS = ("windows", "segments", "term_sum", "weighted_sum", "S_sum", "D_sum", "T_sum", "chord_sum")


@torch.no_grad()
def anchor_sums(parts, targets, weight):
    """{ANCHOR_SUMS: float} of losses.walkway_anchor_parts on a micro-step: anchored windows and segments, the
    term's sum over the windows (unweighted and times the weight), and the sums over the segments of the decoded
    forward distance, the walkway distance, the seconds and the decoded chord."""
    n = int(targets.n) if targets is not None else 0
    if not n:
        return dict.fromkeys(ANCHOR_SUMS, 0.0) | {"windows": 0, "segments": 0}
    term = parts["term"].detach().float()
    v = torch.stack([term.sum(), parts["S"].detach().float().sum(), parts["D"].detach().float().sum(),
                     parts["T"].detach().float().sum(), parts["chord"].detach().float().sum()]).cpu().double().numpy()
    return {"windows": int(torch.unique(targets.window).numel()), "segments": n, "term_sum": float(v[0]),
            "weighted_sum": float(weight) * float(v[0]), "S_sum": float(v[1]), "D_sum": float(v[2]),
            "T_sum": float(v[3]), "chord_sum": float(v[4])}


@dataclass
class ZuptArgs:
    """The inertial anchor of a micro-step (micro_step, module docstring): targets (zupt_anchor.ZuptTargets), weight
    (the phase's: zupt_anchor.weight in phase B, 0 in phase A) and sd (m/s)."""
    targets: object
    weight: float = 0.0
    sd: float = LS.ZUPT_SD


ZUPT_SUMS = ("windows", "strides", "term_sum", "weighted_sum", "distance_sum", "still_sum", "S_sum", "D_sum", "T_sum",
             "speed_sum")


@torch.no_grad()
def zupt_sums(parts, targets, weight):
    """{ZUPT_SUMS: float} of losses.zupt_anchor_parts on a micro-step: anchored windows and strides, the term's sum over
    the windows (unweighted and times the weight) and its distance and still parts, and the sums over the strides of the
    ZUPT stride S, the decoded displacement D, the seconds T and the decoded ankle speeds at both instants."""
    n = int(targets.n) if targets is not None else 0
    if not n:
        return dict.fromkeys(ZUPT_SUMS, 0.0) | {"windows": 0, "strides": 0}
    v = torch.stack([parts["term"].detach().float().sum(), parts["distance"].detach().float().sum(),
                     parts["still"].detach().float().sum(), parts["S"].detach().float().sum(),
                     parts["D"].detach().float().sum(), parts["T"].detach().float().sum(),
                     (parts["v_a"] + parts["v_b"]).detach().float().sum()]).cpu().double().numpy()
    return {"windows": int(torch.unique(targets.window).numel()), "strides": n, "term_sum": float(v[0]),
            "weighted_sum": float(weight) * float(v[0]), "distance_sum": float(v[1]), "still_sum": float(v[2]),
            "S_sum": float(v[3]), "D_sum": float(v[4]), "T_sum": float(v[5]), "speed_sum": float(v[6])}


@dataclass
class JepaArgs:
    """The JEPA part of a phase B micro-step (micro_step, module docstring): rule (jepa.MaskRule), weight lambda,
    recompute (activation recomputation in the context pass), stats (measure std_target and std_pred, and set the
    step-0 references while unset), grad (a list over the encoder's parameters, model.encoder.parameters() order,
    into which the gradient of the micro-step's JEPA contribution is summed; None: not measured)."""
    rule: object
    weight: float
    recompute: bool = True
    stats: bool = False
    grad: list = None

    @classmethod
    def of(cls, section, **kw):
        """JepaArgs of a config's jepa section (config.JepaSection)."""
        return cls(JP.MaskRule.of(section), float(section.weight), bool(section.recompute), **kw)


@torch.no_grad()
def jepa_grad_norms(encoder, jepa_grad):
    """(g_J, g_R) floats over the encoder's parameters (module docstring): the norm of the summed JEPA gradient
    jepa_grad (a list in encoder.parameters() order, None where nothing was summed) and the norm of the rest,
    the parameters' .grad minus it; call before clipping."""
    sj = sr = 0.0
    for p, gj in zip(encoder.parameters(), jepa_grad):
        g = p.grad
        if gj is not None:
            sj += float((gj.double() ** 2).sum())
        if g is not None:
            r = g.double() - gj.double() if gj is not None else g.double()
            sr += float((r ** 2).sum())
    return math.sqrt(sj), math.sqrt(sr)


def _jepa_part(model, batch, x, jp, seed, n_windows, backward, timer):
    """The JEPA part of a micro-step on the tokenizer output x (module docstring): {term (W,) detached, gx and
    x_sel (the gradient at the context input and the rows of x it belongs to, None without), stats}. With
    backward, the gradient of jp.weight x sum_w term_w / n_windows reaches the encoder and the predictor here
    (summed into their .grad) and jp.grad (encoder only); the context graph is freed on return."""
    timer._sync()
    t0 = time.perf_counter()
    jm = model.jepa
    dev = x.device
    if not jm.is_ready():
        jm.sync(model.tokenizer, model.encoder)
    grad = bool(backward) and torch.is_grad_enabled()
    f = jm.forward_pass(model.encoder, batch.tokens, x, LS.is_synthetic(batch, dev), seed, jp.rule,
                        recompute=jp.recompute, grad=grad)
    W = batch.n_windows
    term = LS.jepa_per_window(f.pred, f.target, f.window, W) if f.n_masked else torch.zeros(W, device=dev)
    gx = x_sel = None
    if f.n_masked and grad:
        enc_all = list(model.encoder.parameters())
        enc = [i for i, p in enumerate(enc_all) if p.requires_grad]
        prd = [p for p in jm.predictor.parameters() if p.requires_grad]
        params = [enc_all[i] for i in enc] + prd
        gs = torch.autograd.grad((float(jp.weight) * term).sum() / n_windows, [f.x_in] + params, allow_unused=True)
        gx = gs[0]
        for p, g in zip(params, gs[1:]):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad.add_(g)
        if jp.grad is not None:
            for i, g in zip(enc, gs[1:1 + len(enc)]):
                if g is not None:
                    jp.grad[i] = g.detach().clone() if jp.grad[i] is None else jp.grad[i].add_(g)
        x_sel = x[f.rows]
    has = f.patches.any(1) if f.patches.size else np.zeros(W, bool)
    n_p = batch.tokens.n_patches.cpu().numpy()
    st = {"windows": int(has.sum()), "eligible": int(f.eligible.sum()), "patches": int(n_p[has].sum()),
          "masked": int(f.patches.sum()), "tokens": 0 if f.hidden is None else int(f.hidden.sum()),
          "rows": 0 if f.rows is None else int(len(f.rows)),
          "loss_sum": float(term.detach()[torch.as_tensor(has, device=dev)].sum()) if has.any() else 0.0,
          "stats": bool(jp.stats), "std_target": None, "std_pred": None}
    if jp.stats:
        st["std_target"], st["std_pred"] = JP.spread(f.target), JP.spread(f.pred)
        jm.set_references(st["std_target"], st["std_pred"])
    del f
    timer._sync()
    t1 = time.perf_counter()
    st["time_s"] = t1 - t0
    timer.t = t1
    return {"term": term.detach(), "gx": gx, "x_sel": x_sel, "stats": st}


def micro_step(model, mb, phase, *, device, n_windows=None, loss_cfg=None, progress=1.0, refine_steps=0,
               refine_points=None, seed=0, amp=None, timer=None, backward=True, lock=None, jepa=None, anchor=None,
               zupt=None):
    """One micro-step of a collate.MicroBatch (module docstring): forward, losses and (backward=True) the
    backward of sum_w per_window / n_windows (default the micro-step's windows); lock: held from the encoder to
    the losses (the forward-mode AD of the kinematics, see FWD_AD_LOCK_DOC); jepa: a JepaArgs, the JEPA part of a
    phase B or C micro-step of a model with a JEPA module (ignored in phase A or without the module; the trainer
    passes it in the phases of jepa.in_phases); phase C's losses are those of phase B (loss_phase); anchor: an
    AnchorArgs, the walkway anchor of the micro-step's windows (its term enters per_window times anchor.weight when
    the weight is not 0, and is logged in MicroStats.anchor); zupt: a ZuptArgs, the inertial anchor of the
    micro-step's windows (the same way, logged in MicroStats.zupt). Returns MicroStats."""
    dev = torch.device(device)
    amp = default_amp(dev) if amp is None else amp
    timer = Timer(dev) if timer is None else timer
    if timer.t is None:
        timer.start()
    batch = mb.batch.to(dev)
    targets = mb.targets.to(dev)
    timer.lap("transfer")
    W = batch.n_windows
    n_windows = W if n_windows is None else int(n_windows)
    gen = torch.Generator(device=dev).manual_seed(int(seed) % 2 ** 63)
    hist = None
    jstate, hook = {}, None
    if jepa is not None and phase in ("B", STAGE2_PHASE) and _jepa_of(model) is not None:
        def hook(x):
            jstate.update(_jepa_part(model, batch, x, jepa, int(seed), n_windows, backward, timer))
    with (lock if lock is not None else contextlib.nullcontext()), amp_context(dev, amp):
        post = encode(model, batch, timer, hook=hook)
        amortized = None
        if refine_steps:
            hist = []
            ref = refine(model, post, batch, int(refine_steps), points_per_stream=refine_points, generator=gen,
                         history=hist)
            amortized, post = post, straight_through(post, ref)
            timer.lap("refinement")
        body, pred = forward_draw(model, batch, post, gen, timer, truth_windows=LS.truth_mask(batch, targets, dev))
        out = LS.stage1_losses(model, batch, post, body, pred, loss_phase(phase), targets=targets, amortized=amortized,
                               progress=progress, cfg=loss_cfg)
        if jstate:
            out["per_window"] = out["per_window"] + float(jepa.weight) * jstate["term"]
            out["total"] = out["per_window"].mean()
        astats = None
        if anchor is not None:
            ap = LS.walkway_anchor_parts(body.out.root, batch.lengths, anchor.targets, batch.rate, float(anchor.sd))
            if float(anchor.weight) != 0.0:
                out["per_window"] = out["per_window"] + float(anchor.weight) * ap["term"]
                out["total"] = out["per_window"].mean()
            astats = anchor_sums(ap, anchor.targets, float(anchor.weight))
        zstats = None
        if zupt is not None:
            zp = LS.zupt_anchor_parts(model, body.z, body.beta, batch, zupt.targets, float(zupt.sd))
            if float(zupt.weight) != 0.0:
                out["per_window"] = out["per_window"] + float(zupt.weight) * zp["term"]
                out["total"] = out["per_window"].mean()
            zstats = zupt_sums(zp, zupt.targets, float(zupt.weight))
    timer.lap("losses")
    if backward:
        if jstate.get("gx") is not None:
            torch.autograd.backward([out["per_window"].sum() / n_windows, jstate["x_sel"]], [None, jstate["gx"]])
        else:
            (out["per_window"].sum() / n_windows).backward()
        timer.lap("backward")
    syn = LS.is_synthetic(batch, dev)
    parts = [out[k].detach().float().sum() for k in TERM_KEYS]
    parts.append((out["nuis_aux"].detach().float() * syn.float()).sum())
    vals = torch.stack(parts).cpu().numpy().astype(np.float64)
    sums = {k: float(v) for k, v in zip(TERM_KEYS, vals[:-1])}
    return MicroStats(W=W, synthetic=int(syn.sum()), sums=sums, nuis_aux_synthetic=float(vals[-1]),
                      patches=int(post.mask.sum()), active=_active_dims(post), useful=int(mb.useful),
                      padded=int(mb.padded), row_len=int(mb.row_len),
                      refine=None if not hist else (hist[0], hist[-1]), time=dict(timer.s),
                      finite=bool(np.isfinite(vals).all()), jepa=jstate["stats"] if jstate else None, anchor=astats,
                      zupt=zstats)


def _window_spread(rel, w, W):
    """(W,) m: per window, the mean over joints of the RMS distance of each heading-relative joint position
    rel (M, J, 3) from its window mean over the window's knots w (M,); 0 for a window without knots."""
    from ..kernels.pose import window_sum
    n = window_sum(torch.ones_like(w, dtype=rel.dtype), w, W).clamp_min(1.0)
    mean = window_sum(rel.reshape(len(w), -1), w, W).reshape(W, *rel.shape[1:]) / n[:, None, None]
    d2 = ((rel - mean[w]) ** 2).sum(-1)                                                  # (M, J)
    return torch.sqrt(window_sum(d2, w, W) / n[:, None]).mean(-1)


class RootMotion:
    """Sums of the root-motion dev numbers over micro-batches (evaluate). add(root, truth, windows, rate,
    lengths): the decoded RootGrid and the truth's (losses.truth_root_grid, the same grid points) at the valid
    points of the windows (W,) bool of lengths (W,) knots. numbers(): root_speed_ratio, the sum of the decoded
    horizontal pelvis speed |u| over the sum of the true one at the points where the true speed exceeds
    ROOT_MOVING m/s; root_speed_ratio_far, the same beyond ROOT_FAR_S from the centre knot; heading_gain and
    heading_corr, the least-squares slope through 0 and the correlation of the decoded on the true yaw about the
    centre, both wrapped to [-pi, pi), at the points where the true one lies within HEADING_WITHIN;
    heading_err_deg, the mean absolute wrapped yaw error, and heading_zero_deg, the same for a body that never
    turns; lateral_slope, the slope through 0 of the decoded on the true lateral displacement X_x from the centre
    knot; root_points, the points read. None where nothing was read (and heading_corr for a constant decoded
    yaw)."""

    KEYS = ("speed", "speed_t", "far", "far_t", "sxy", "sxx", "syy", "sx", "sy", "n_h", "err", "zero", "n", "lxy",
            "lxx")

    def __init__(self):
        self.s = dict.fromkeys(self.KEYS, 0.0)

    def add(self, root, truth, windows, rate, lengths):
        if root is None:
            return
        if not torch.equal(root.k.to(truth.k.device), truth.k) or not torch.equal(root.w.to(truth.w.device), truth.w):
            raise ValueError("the decoded root grid and the truth's differ")
        dev = truth.k.device
        keep = truth.valid & windows.to(dev)[truth.w]
        if not bool(keep.any()):
            return
        f = lambda x: x.to(dev).double()[keep]                                      # noqa: E731
        u, u_t, X, X_t = f(root.u), f(truth.u), f(root.X), f(truth.X)
        psi, psi_t = LS._wrap(f(root.psi)), f(truth.psi)
        sp, sp_t = u.norm(dim=-1), u_t.norm(dim=-1)
        mov = sp_t > ROOT_MOVING
        c = torch.div(torch.as_tensor(lengths, device=dev).long(), 2, rounding_mode="floor")[truth.w[keep]]
        far = mov & ((truth.k[keep] - c).abs().double() / rate > ROOT_FAR_S)
        m = psi_t.abs() <= HEADING_WITHIN
        s = self.s
        s["speed"] += float(sp[mov].sum())
        s["speed_t"] += float(sp_t[mov].sum())
        s["far"] += float(sp[far].sum())
        s["far_t"] += float(sp_t[far].sum())
        x, y = psi_t[m], psi[m]
        s["sxy"] += float((x * y).sum())
        s["sxx"] += float((x * x).sum())
        s["syy"] += float((y * y).sum())
        s["sx"] += float(x.sum())
        s["sy"] += float(y.sum())
        s["n_h"] += int(m.sum())
        s["err"] += float(LS._wrap(psi - psi_t).abs().sum())
        s["zero"] += float(psi_t.abs().sum())
        s["n"] += int(keep.sum())
        s["lxy"] += float((X[:, 0] * X_t[:, 0]).sum())
        s["lxx"] += float((X_t[:, 0] ** 2).sum())

    def numbers(self):
        s = self.s
        r = lambda a, b: a / b if b > 0 else None                                  # noqa: E731
        corr = None
        n = s["n_h"]
        if n > 1:
            vx, vy = n * s["sxx"] - s["sx"] ** 2, n * s["syy"] - s["sy"] ** 2
            if vx > 0 and vy > 0:
                corr = (n * s["sxy"] - s["sx"] * s["sy"]) / math.sqrt(vx * vy)
        return {"root_speed_ratio": r(s["speed"], s["speed_t"]), "root_speed_ratio_far": r(s["far"], s["far_t"]),
                "heading_gain": r(s["sxy"], s["sxx"]), "heading_corr": corr,
                "heading_err_deg": math.degrees(s["err"] / s["n"]) if s["n"] else None,
                "heading_zero_deg": math.degrees(s["zero"] / s["n"]) if s["n"] else None,
                "lateral_slope": r(s["lxy"], s["lxx"]), "root_points": int(s["n"])}


@torch.no_grad()
def evaluate(model, mbs, phase, *, device, loss_cfg=None, amp=None, seed=0, limits=None):
    """Dev numbers of fixed collate.MicroBatch objects, the model in eval mode (restored after), no gradient,
    no refinement: {windows, terms: the window-weighted mean of every TERM_KEYS entry (one posterior draw, KL
    at full weight), joint_error_mm: the mean distance between decoded and true pelvis-relative joints 1 to
    21, each body in its own heading frame (losses.heading_relative), at the decoded knots of the windows with
    a body truth, the posterior means decoded (z = mu_z, beta = the person-pooled mu_b), joint_knots;
    root_orient_deg: the mean geodesic of the decoded root orientation (window frame) against the truth;
    joint_error_true_skeleton_mm: the same decoded rotations posed on the true skeleton (the pose error
    alone); joint_error_z0_mm: decoded with z = 0 (a decoder that ignores the latent scores as with z = mu);
    spread_ratio: the within-window spread of the decoded heading-frame joints (_window_spread) summed over
    windows, over the truth's; prior_joint_error_mm: the constant mean pose (decoder.mean_R) posed on the
    prior-mean skeleton of each window's gender, on the same knots; the root motion (root_motion_numbers) of
    the decoded posterior mean at the valid root-grid points of the windows with a body truth whose heading
    the input observes (losses.heading_observable): root_speed_ratio, root_speed_ratio_far, heading_gain,
    heading_corr, heading_err_deg, heading_zero_deg, lateral_slope, root_points}. limits (task S2.8): (lo, hi) of a
    plausibility table (losses.joint_limits), or None; given, the dict also holds outside_limits_share, the share of
    the posterior-mean decoded knots of every window (batch.knots) with an angle outside the table
    (losses.outside_limits), and outside_limits_knots, the knots read."""
    from ..body.rotations import geodesic, rot6d_to_matrix
    from ..body.joints import fk
    dev = torch.device(device)
    amp = default_amp(dev) if amp is None else amp
    was = model.training
    model.eval()
    sums, W, knots = dict.fromkeys(TERM_KEYS, 0.0), 0, 0
    acc = dict.fromkeys(("err", "err_ts", "err_z0", "err_prior", "root", "spread", "spread_t"), 0.0)
    n_err, n_root = 0, 0
    n_out, n_lim = 0, 0
    motion = RootMotion()
    try:
        for i, mb in enumerate(mbs):
            batch, targets = mb.batch.to(dev), mb.targets.to(dev)
            gen = torch.Generator(device=dev).manual_seed((int(seed) + i) % 2 ** 63)
            with amp_context(dev, amp):
                post = encode(model, batch)
                body, pred = forward_draw(model, batch, post, gen, truth_windows=LS.truth_mask(batch, targets, dev))
                out = LS.stage1_losses(model, batch, post, body, pred, loss_phase(phase), targets=targets,
                                       progress=1.0, cfg=loss_cfg)
                z = post.mu_z * post.mask[..., None]
                mu_b, _ = subject_pool(post.mu_b, post.logvar_b, batch.person.to(dev))
                npw = batch.tokens.n_patches.to(dev)
                lengths = batch.lengths.to(dev)
                bo = model.decode(z, mu_b, batch.knots, npw, lengths)
                bo0 = model.decode(torch.zeros_like(z), mu_b, batch.knots, npw, lengths)
            for k in TERM_KEYS:
                sums[k] += float(out[k].float().sum())
            W += batch.n_windows
            if limits is not None:
                with torch.autocast(device_type=dev.type, enabled=False):
                    off = LS.outside_limits(bo.rot6d.float(), limits[0].to(dev), limits[1].to(dev))
                n_out += int(off.sum())
                n_lim += int(off.numel())
            has = targets.has_truth.to(dev)
            keep = has[bo.w]
            if batch.truth is None or not bool(keep.any()):
                continue
            with torch.autocast(device_type=dev.type, enabled=False):
                motion.add(bo.root, LS.truth_root_grid(batch.truth, batch.rate, lengths),
                           has & LS.heading_observable(batch, dev), batch.rate, lengths)
                w, kk = bo.w[keep], bo.k[keep]
                g = batch.gender.to(dev)
                rest = model.skeleton.rest(mu_b.float(), g)
                tr = batch.truth
                rest_t = model.skeleton.rest(tr.beta.float(), g)
                R = rot6d_to_matrix(bo.rot6d[keep].float())
                Rt = rot6d_to_matrix(tr.rot6d[w, kk].float())
                pos_t, Rg_t = fk(Rt, tr.transl[w, kk].float(), rest_t[w])
                rel_t = LS.heading_relative(pos_t, Rg_t)
                rel = LS.heading_relative(*fk(R, bo.transl[keep].float(), rest[w]))
                rel_ts = LS.heading_relative(*fk(R, bo.transl[keep].float(), rest_t[w]))
                rel_0 = LS.heading_relative(*fk(rot6d_to_matrix(bo0.rot6d[keep].float()), bo0.transl[keep].float(),
                                                rest[w]))
                rest_p = model.skeleton.rest(model.shape_prior.mean[g].float(), g)
                R_p = model.decoder.mean_R.float().expand(len(w), -1, -1, -1)
                rel_p = LS.heading_relative(*fk(R_p, torch.zeros(len(w), 3, device=dev), rest_p[w]))
                for key, r in (("err", rel), ("err_ts", rel_ts), ("err_z0", rel_0), ("err_prior", rel_p)):
                    acc[key] += float((r - rel_t).norm(dim=-1).sum())
                acc["root"] += float(geodesic(R[:, 0], Rt[:, 0]).sum())
                Wb = batch.n_windows
                acc["spread"] += float(_window_spread(rel, w, Wb).sum())
                acc["spread_t"] += float(_window_spread(rel_t, w, Wb).sum())
            n_err += int(rel.shape[0] * rel.shape[1])
            n_root += int(keep.sum())
            knots += int(keep.sum())
    finally:
        model.train(was)
    mm = lambda key: 1000.0 * acc[key] / n_err if n_err else None  # noqa: E731
    res = {"windows": W, "terms": {k: v / max(W, 1) for k, v in sums.items()},
           "joint_error_mm": mm("err"), "joint_knots": knots,
           "joint_error_true_skeleton_mm": mm("err_ts"), "joint_error_z0_mm": mm("err_z0"),
           "prior_joint_error_mm": mm("err_prior"),
           "root_orient_deg": math.degrees(acc["root"] / n_root) if n_root else None,
           "spread_ratio": acc["spread"] / acc["spread_t"] if n_root and acc["spread_t"] > 0 else None,
           **motion.numbers()}
    if limits is not None:
        res.update(outside_limits_share=n_out / n_lim if n_lim else None, outside_limits_knots=n_lim)
    return res


@torch.no_grad()
def health(model, mb, *, device, amp=None, max_queries=HEALTH_QUERIES):
    """The health fields of a log record (module docstring) on one collate.MicroBatch, the model in eval mode
    (restored after), no gradient, the posterior means decoded: {decoder: [BodyDecoder.attention_stats per
    block], rotation_rms_rad: {root_tilt, joints}, root_rates_rms: {omega rad/s, u m/s} at the valid root-grid
    points, logvar_z, mu_z_sd_between, mu_z_sd_within, beta_error, rest_height_ratio, rest_height_error_m} (the
    last three None without a body truth)."""
    from ..model.decoder import N_ROT
    dev = torch.device(device)
    amp = default_amp(dev) if amp is None else amp
    was = model.training
    model.eval()
    try:
        batch, targets = mb.batch.to(dev), mb.targets.to(dev)
        with amp_context(dev, amp):
            post = encode(model, batch)
        m = post.mask
        z = post.mu_z * m[..., None]
        mu_b, _ = subject_pool(post.mu_b, post.logvar_b, batch.person.to(dev))
        npw = batch.tokens.n_patches.to(dev)
        att = model.decoder.attention_stats(z, mu_b, batch.knots, npw, max_queries=max_queries)
        y, _, _ = model.decoder.raw(z, mu_b, batch.knots, npw)
        rg = model.decode(z, mu_b, batch.knots, npw, batch.lengths.to(dev)).root
        rms = lambda x: float(torch.sqrt((x.float() ** 2).mean())) if x.numel() else None  # noqa: E731
        mf, mu = m.float()[..., None], post.mu_z.float()
        n = mf.sum(1).clamp_min(1.0)
        mw = (mu * mf).sum(1) / n                                                              # (W, 32)
        within = torch.sqrt((((mu - mw[:, None]) ** 2) * mf).sum(1) / n)                       # (W, 32)
        rec = {"decoder": att,
               "rotation_rms_rad": {"root_tilt": rms(y[:, 0:2]), "joints": rms(y[:, 2:N_ROT])},
               "root_rates_rms": {"omega": rms(rg.omega[rg.valid]), "u": rms(rg.u[rg.valid])},
               "logvar_z": float(post.logvar_z.float()[m].mean()),
               "mu_z_sd_between": float(mw.std(0).mean()) if mw.shape[0] > 1 else 0.0,
               "mu_z_sd_within": float(within.mean()),
               "beta_error": None, "rest_height_ratio": None, "rest_height_error_m": None}
        has = LS.truth_mask(batch, targets, dev)
        if batch.truth is not None and bool(has.any()):
            with torch.autocast(device_type=dev.type, enabled=False):
                bt = batch.truth.beta.to(dev).float()
                g = batch.gender.to(dev)
                height = lambda r: r[..., 1].amax(-1) - r[..., 1].amin(-1)  # noqa: E731
                hp = height(model.skeleton.rest(mu_b.float(), g))[has]
                ht = height(model.skeleton.rest(bt, g))[has]
                rec.update(beta_error=float((post.mu_b.float() - bt)[has].abs().mean()),
                           rest_height_ratio=float((hp / ht).mean()), rest_height_error_m=float((hp - ht).abs().mean()))
        return rec
    finally:
        model.train(was)


# ---------------------------------------------------------------- data: items and prefetch
@dataclass
class Item:
    """A sampled and collated micro-step: step (sampler.Step), mb (collate.MicroBatch), state (the sampler's
    resumable fields after it), host seconds {sample, render, collate}."""
    step: object
    mb: object
    state: dict
    host: dict


def snapshot(sampler):
    """The fields of MixtureSampler.state_dict() that move with each micro-step (step, generator, credits,
    totals, and the window tally of a phase C window cap, task S2B), copied."""
    snap = {"step": int(sampler.step), "rng": copy.deepcopy(sampler.rng.bit_generator.state),
            "credit": dict(sampler.credit), "totals": copy.deepcopy(sampler.totals)}
    if getattr(sampler, "wtally", None) is not None:
        snap["window_tally"] = dict(sampler.wtally)
    return snap


def sampler_state(sampler, snap):
    """MixtureSampler.state_dict() with the moving fields of snapshot snap."""
    st = sampler.state_dict()
    st.update(copy.deepcopy(snap))
    return st


def _render_seconds(synth):
    t = getattr(synth, "timing", None)
    return float(sum(t.values())) if isinstance(t, dict) else 0.0


def produce(sampler, collate_cfg, collate_kw=None):
    """Item of the sampler's next micro-step, collated with the step's seed; collate_kw: further collate arguments
    (the run's sources and nuisance vocabularies, Trainer.collate_kw; none for a stage-1 config)."""
    t0 = time.perf_counter()
    r0 = _render_seconds(sampler.synth)
    step = sampler.next()
    t1 = time.perf_counter()
    render = _render_seconds(sampler.synth) - r0
    mb = CO.collate(step.windows, cfg=collate_cfg, gen=step.seed, **(collate_kw or {}))
    t2 = time.perf_counter()
    return Item(step, mb, snapshot(sampler), {"sample": t1 - t0 - render, "render": render, "collate": t2 - t1})


def plausibility_marks(step, units):
    """(W,) bool of a sampler Step: True for the windows whose plan item is not one of the stage-2 units (a synthetic
    or replay window), in the order of step.windows (task S2.8; losses, plausibility)."""
    items = step.plan.items
    if len(items) != len(step.windows):
        raise ValueError(f"{len(items)} plan items for {len(step.windows)} windows")
    return torch.tensor([it.source not in units for it in items], dtype=torch.bool)


FWD_AD_LOCK_DOC = """Forward-mode AD levels (torch.func.jvp, used by body.kinematics) are process-global: the
renderer's kinematics in the prefetch thread and the model's in the main thread may not run at once. With
prefetch, the renderer's render() and the main thread's forward (encoder to losses, refinement included) and
dev hook hold one lock; backward and the host work of sampling and collate overlap."""


class LockedRenderer:
    """The renderer with every render() call under a lock (FWD_AD_LOCK_DOC); other attributes pass through."""

    def __init__(self, synth, lock):
        self._synth, self._lock = synth, lock

    def render(self, *args, **kw):
        with self._lock:
            return self._synth.render(*args, **kw)

    def __getattr__(self, name):
        return getattr(self._synth, name)


class Prefetcher:
    """A thread calling fn() ahead of the consumer, at most depth items queued; get() returns them in order
    and re-raises the thread's exception."""

    def __init__(self, fn, depth):
        self.fn, self.q = fn, queue.Queue(maxsize=max(1, int(depth)))
        self.stop, self.err = threading.Event(), None
        self.thread = threading.Thread(target=self._run, name="mdfm-prefetch", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                item = self.fn()
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # noqa: BLE001 (re-raised in the consumer)
            self.err = e

    def get(self):
        while True:
            try:
                return self.q.get(timeout=0.2)
            except queue.Empty:
                if self.err is not None:
                    raise self.err
                if not self.thread.is_alive():
                    raise RuntimeError("the prefetch thread stopped")

    def close(self):
        self.stop.set()
        while self.thread.is_alive():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=0.1)


# ---------------------------------------------------------------- the trainer
class Trainer:
    """The training loop (module docstring). cfg: a checked RunConfig; body_store (split train) and
    window_store (phase B) as data.sampler.MixtureSampler reads them; synth: the renderer
    (factory.synth.SynthSampler); device; run_dir (default default_run_dir(cfg)); amp (default bf16 on CUDA);
    prefetch: micro-steps sampled ahead in a thread (0: synchronous); dev_hook(trainer, event) -> dict or None,
    called at 'start', every dev_every steps ('interval'), at each phase end ('phase_end') and at the stop of
    a shortened call ('stop'); echo: a print function; check_reports: refuse stores whose person lists differ
    from their reports (provenance.report_mismatches); layouts: {name: Layout} for the samplers (default
    factory/layouts.load_layouts()); model: an MFM to train instead of MFM(cfg.model) (initialized with
    torch.manual_seed(cfg.seed) otherwise); memory_cap_gb: the caching allocator's cap on a CUDA device
    (cap_memory; None leaves the allocator uncapped); decoder_checkpoint: the decoder's blocks recompute their
    activations in the backward pass (model/decoder.BodyDecoder.checkpoint_blocks; same values and gradients,
    about 3.5 GB less at a phase B micro-step, reports/bench/phaseB_memory.md); jepa_grad_steps: with the config's
    jepa section enabled, the first jepa_grad_steps optimizer steps of phase B log g_J and g_R ('jepa_grad', module
    docstring; 0: never). With the jepa section enabled, the model receives its JEPA module (mfm.MFM.attach_jepa)
    unless it holds one, its predictor initialized under a forked CPU generator seeded cfg.seed + jepa.MASK_SALT,
    so that the global generator (which the motion prior's condition dropout draws from) moves as without JEPA.
    With the config's render_aug section enabled, the renderer's amp_aug is set to its factory.synth.AmplitudeAug
    (the amplitude randomization of every training render, both phases; the 'start' record carries the section); a
    renderer that carries an amp_aug under a config whose section is off is refused (ValueError). With the config's
    walkway_anchor section enabled and a window store, the anchor table is built (walkway_anchor.build_anchor_table)
    or, given as anchor_table, checked (walkway_anchor.assert_anchor_clean against the store's weargait_pd training
    fold, its held-out persons, the lockbox list, the proposed PD folds and the deny lists); an anchor_table under a
    config whose section is off is refused (ValueError). With the config's zupt_anchor section enabled and a window
    store, the inertial anchor's table is built (zupt_anchor.build_table) or, given as zupt_table, checked
    (zupt_anchor.assert_table_clean against the same persons and refused sets); a zupt_table under a config whose
    section is off is refused (ValueError). With the config's stage2 section enabled (module docstring,
    Phase C): window_store is a sampler.Stage2Stores whose stage-2 store was opened under the run's name tables and
    tier and whose replay store is the stage-1 root (ValueError otherwise); a Stage2Stores under a config whose
    section is off is refused."""

    def __init__(self, cfg, body_store, window_store=None, synth=None, device=None, run_dir=None, *, amp=None,
                 prefetch=0, dev_hook=None, echo=print, check_reports=True, layouts=None, model=None,
                 memory_cap_gb=MEMORY_CAP_GB, decoder_checkpoint=True, jepa_grad_steps=0, anchor_table=None,
                 zupt_table=None):
        self.cfg = CF.check(cfg)
        self.anchor_on = bool(cfg.walkway_anchor.enabled)
        self.anchor_table, self.anchor_check = None, None
        if anchor_table is not None and not self.anchor_on:
            raise ValueError("an anchor table under a configuration whose walkway_anchor section is off")
        if self.anchor_on and window_store is not None:
            wa = cfg.walkway_anchor
            table = WA.build_anchor_table(window_store, wa) if anchor_table is None else anchor_table
            train = window_store.person_lists("train").get(WA.SOURCE, [])
            self.anchor_check = WA.assert_anchor_clean(table, train, WA.refused_persons(window_store),
                                                       wa.persons_sha256)
            self.anchor_table = table
        self.zupt_on = bool(cfg.zupt_anchor.enabled)
        self.zupt_table, self.zupt_check = None, None
        if zupt_table is not None and not self.zupt_on:
            raise ValueError("a zupt anchor table under a configuration whose zupt_anchor section is off")
        if self.zupt_on and window_store is not None:
            za = cfg.zupt_anchor
            table = ZA.build_table(window_store, za) if zupt_table is None else zupt_table
            train = window_store.person_lists("train").get(ZA.SOURCE, [])
            self.zupt_check = ZA.assert_table_clean(table, train, WA.refused_persons(window_store), za.persons_sha256)
            self.zupt_table = table
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.memory_cap_gb = cap_memory(self.device, memory_cap_gb)
        self.amp = default_amp(self.device) if amp is None else bool(amp)
        self.body_store, self.window_store, self.synth = body_store, window_store, synth
        self.render_aug = None
        if cfg.render_aug.enabled:
            from ..factory.synth import AmplitudeAug
            self.render_aug = AmplitudeAug.of(cfg.render_aug)
            if synth is not None:
                synth.amp_aug = self.render_aug
        elif getattr(synth, "amp_aug", None) is not None:
            raise ValueError("the renderer carries an amplitude randomization (amp_aug) while the configuration's "
                             "render_aug section is off")
        if layouts is None and not CF.stage2_on(cfg):
            from . import bodies as BD
            if BD.body_of(cfg) == BD.OPEN:
                layouts = BD.layouts(BD.OPEN)       # the open layouts (reports/open_layouts; the families equal v1's)
        self.layouts = layouts
        self.run_dir = default_run_dir(cfg) if run_dir is None else str(run_dir)
        CK.check_path(os.path.join(self.run_dir, "ckpt.pt"))
        self.prefetch, self.dev_hook, self.echo = int(prefetch), dev_hook, echo or (lambda *a, **k: None)
        self.jepa_on = bool(cfg.jepa.enabled)
        self.jepa_in = tuple(cfg.jepa.in_phases)
        self.jepa_grad_steps = int(jepa_grad_steps)
        self.stage2_on = CF.stage2_on(cfg)
        self.run_phases = CF.run_phases(cfg)
        self.stage2_source = None           # the migration lineage of a stage-2 run (init_stage2; module docstring)
        self._check_stores(window_store)
        if model is None:
            torch.manual_seed(int(cfg.seed))
            model = M.MFM(CF.model_config(cfg))
        if self.jepa_on and _jepa_of(model) is None:
            jp = cfg.jepa
            with torch.random.fork_rng(devices=[]):    # the global generator moves as without JEPA
                torch.manual_seed((int(cfg.seed) + JP.MASK_SALT) % 2 ** 63)
                model.attach_jepa(width=int(jp.pred_width), depth=int(jp.pred_depth), heads=int(jp.pred_heads))
        self.model = model.to(self.device)
        dec = getattr(self.model, "decoder", None)
        self.decoder_checkpoint = bool(decoder_checkpoint) and hasattr(dec, "checkpoint_blocks")
        if hasattr(dec, "checkpoint_blocks"):
            dec.checkpoint_blocks = self.decoder_checkpoint
        # the phase C freeze (task S2B; module docstring): a stage-2 run trains phase C only, so the groups are frozen
        # before the EMA and the optimizer exist; None and an empty skip set while stage2.freeze is empty
        self.freeze = freeze_parameters(self.model, tuple(cfg.stage2.freeze)) \
            if self.stage2_on and tuple(cfg.stage2.freeze) else None
        self.ema_skip = frozenset(self.freeze["names"]) if self.freeze else frozenset()
        self.window_cap_on = self.stage2_on and float(cfg.stage2.window_cap_per_person) > 0.0
        self.ema = make_ema(self.model)
        self.opt, self.opt_kind = build_optimizer(self.model, cfg.schedule, self.device)
        self.sched = Schedule(cfg.schedule, CF.run_steps(cfg), self.run_phases)
        self.loss_cfg = cfg.losses.loss_config()
        # the stage-2 units whose windows carry no plausibility term (losses.plausibility_windows, task S2.8)
        self.plausibility_units = frozenset(cfg.stage2.units) if (
            CF.stage2_on(cfg) and cfg.losses.plausibility_windows == "synthetic_replay") else None
        self.collate_cfg = CO.collate_config(cfg)
        # a config's names section away from its defaults: the batches carry the model's sources and vocabularies
        self.collate_kw = {} if cfg.names == CF.NamesSection() else {
            "sources": tuple(self.model.cfg.sources), "nuis_fields": self.model.nuis_fields}
        self.sampler, self.sampler_snap, self._pre = None, None, None
        self.sampler_phase = None           # the phase of the running sampler
        self.fwd_lock = threading.Lock() if self.prefetch > 0 else None
        self.refine_every = None            # overrides schedule.refine_every when set (bench)
        self.nonfinite = 0
        self.alarm_streak, self.alarm_streak_phase = None, None   # schedule.alarm_consecutive
        self.last = None
        self.history = []
        self.checkpoints = []
        self.init_source = None             # the lineage of an init-from run (init_from; module docstring)
        self.log_path = os.path.join(self.run_dir, "log.jsonl")
        self._reset_interval()
        if isinstance(window_store, S.Stage2Stores):
            self.report_mismatches = window_store.report_mismatches(body_store)
        else:
            lists = PV.person_lists(body_store, window_store)
            self.report_mismatches = PV.report_mismatches(lists)
        if check_reports and self.report_mismatches:
            raise PV.ProvenanceError(f"the stores differ from their reports: {self.report_mismatches}")

    def _check_stores(self, window_store):
        """ValueError unless the window store fits the configuration (class docstring): a Stage2Stores, its stage-2
        store under the run's name tables and tier and its replay store the stage-1 root, for a stage-2 run; no
        Stage2Stores for a stage-1 run."""
        two = isinstance(window_store, S.Stage2Stores)
        if not self.stage2_on:
            if two:
                raise ValueError("a Stage2Stores under a configuration whose stage2 section is off")
            return
        if window_store is None:
            return
        if not two:
            raise ValueError("a stage-2 run reads a sampler.Stage2Stores (the stage-1 replay root and the stage-2 "
                             "root)")
        want = CF.name_tables(self.cfg).sha256()
        got = getattr(window_store.stage2, "names_sha256", None)
        if got != want:
            raise ValueError(f"the stage-2 store was opened under the name tables {got}, this run's are {want}")
        if window_store.tier != self.cfg.tier:
            raise ValueError(f"a stage-2 store of tier {window_store.tier} for a run of tier {self.cfg.tier}")
        want_strata = str(self.cfg.losses.hsic_strata_sha256) or None
        got_strata = getattr(window_store.stage2, "strata_sha256", None)
        if got_strata != want_strata:
            raise ValueError(f"the stage-2 store attaches the HSIC strata table {got_strata}, this run's "
                             f"losses.hsic_strata_sha256 is {want_strata} (task S2.8)")
        if getattr(window_store.replay, "strata_sha256", None) is not None:
            raise ValueError("the replay store attaches HSIC strata: the stage-1 root is read as it is (stratum 0)")
        from . import names as NM
        rep = getattr(window_store.replay, "names_sha256", None)
        if rep not in (None, NM.STAGE1.sha256()):
            raise ValueError(f"the replay store was opened under the name tables {rep}, not names.STAGE1 (the stage-1 "
                             f"root is read as it is)")

    # ------------------------------------------------------------ samplers
    def _make_sampler(self, phase):
        ws = self.window_store if phase != PHASES[0] else None
        if phase != PHASES[0] and ws is None:
            raise ValueError(f"phase {phase} needs the window store")
        synth = self.synth if self.fwd_lock is None or self.synth is None else LockedRenderer(self.synth, self.fwd_lock)
        if phase == STAGE2_PHASE:
            return S.MixtureSampler(self.body_store, ws, synth, phase, phase_seed(self.cfg, phase), self.cfg,
                                    layouts=self.layouts, redraws=int(self.cfg.stage2.redraws), tier=self.cfg.tier)
        return S.MixtureSampler(self.body_store, ws, synth, phase, phase_seed(self.cfg, phase), self.cfg.mixture,
                                layouts=self.layouts, tier=self.cfg.tier)

    def _start_sampler(self, phase, state=None):
        self._close_prefetch()
        self.sampler = self._make_sampler(phase)
        self.sampler_phase = phase
        if state is not None:
            self.sampler.load_state_dict(state)
        self.sampler_snap = snapshot(self.sampler)
        if self.prefetch > 0:
            self._pre = Prefetcher(self._produce, self.prefetch)

    def _close_prefetch(self):
        if self._pre is not None:
            self._pre.close()
            self._pre = None

    def set_prefetch(self, n):
        """Change the prefetch depth (the next sampler started uses it); 0 stops the thread and the lock."""
        self._close_prefetch()
        self.prefetch = int(n)
        self.fwd_lock = threading.Lock() if self.prefetch > 0 else None
        self.sampler, self.sampler_phase = None, None

    def next_item(self):
        """The next Item of the current phase's sampler (from the prefetch thread when running)."""
        return self._pre.get() if self._pre is not None else self._produce()

    def _produce(self):
        """produce() of the running sampler: the stage-1 call (two arguments) without collate_kw, the run's sources and
        vocabularies otherwise. In phase C under losses.plausibility_windows 'synthetic_replay' (task S2.8), the
        item's Targets.plausibility_on marks the windows whose plan item is not a stage-2 unit (losses, plausibility)."""
        item = produce(self.sampler, self.collate_cfg) if not self.collate_kw else \
            produce(self.sampler, self.collate_cfg, self.collate_kw)
        units = getattr(self, "plausibility_units", None)
        if units is not None and self.sampler_phase == STAGE2_PHASE:
            item.mb.targets.plausibility_on = plausibility_marks(item.step, units)
        return item

    # ------------------------------------------------------------ logging
    def log(self, obj):
        os.makedirs(self.run_dir, exist_ok=True)
        rec = dict(obj, time=datetime.datetime.now().isoformat(timespec="seconds"))
        jsonl(self.log_path, rec)
        return rec

    def memory(self):
        """Allocated, reserved and their peaks (GB) on a CUDA device, {} elsewhere."""
        if self.device.type != "cuda":
            return {}
        d = self.device
        return {"allocated": torch.cuda.memory_allocated(d) / GB, "reserved": torch.cuda.memory_reserved(d) / GB,
                "max_allocated": torch.cuda.max_memory_allocated(d) / GB,
                "max_reserved": torch.cuda.max_memory_reserved(d) / GB}

    def check_memory(self):
        """MemoryBudgetError when the peak reserved memory exceeds schedule.max_reserved_gb."""
        if self.device.type != "cuda":
            return
        peak = torch.cuda.max_memory_reserved(self.device) / GB
        bound = float(self.cfg.schedule.max_reserved_gb)
        if peak > bound:
            self.log({"event": "memory_exceeded", "phase": self.sched.in_phase, "step": self.sched.step,
                      "memory_gb": self.memory(), "bound_gb": bound})
            raise MemoryBudgetError(f"torch.cuda.max_memory_reserved {peak:.2f} GB above {bound} GB")

    def _reset_interval(self):
        self.acc = {"sums": dict.fromkeys(TERM_KEYS, 0.0), "W": 0, "synthetic": 0, "nuis_syn": 0.0, "patches": 0,
                    "active": [], "useful": 0, "padded": 0, "micro": 0, "steps": 0, "timed": 0,
                    "time": dict.fromkeys(PARTS, 0.0), "grad": [], "skipped": 0, "refine": [], "tokens": {},
                    "buckets": {}, "health": None, "t0": time.perf_counter(),
                    "jepa": {"micro": 0, "windows": 0, "eligible": 0, "patches": 0, "masked": 0, "tokens": 0,
                             "loss_sum": 0.0, "time": 0.0, "timed": 0, "first": None, "decay": None,
                             "grad_steps": 0},
                    "anchor": dict.fromkeys(ANCHOR_SUMS, 0.0) | {"micro": 0},
                    "zupt": dict.fromkeys(ZUPT_SUMS, 0.0) | {"micro": 0}}

    def _accumulate(self, st, item):
        a = self.acc
        for k in TERM_KEYS:
            a["sums"][k] += st.sums[k]
        a["W"] += st.W
        a["synthetic"] += st.synthetic
        a["nuis_syn"] += st.nuis_aux_synthetic
        a["patches"] += st.patches
        a["active"].append(st.active)
        a["useful"] += st.useful
        a["padded"] += st.padded
        a["micro"] += 1
        a["buckets"][st.row_len] = a["buckets"].get(st.row_len, 0) + 1
        if st.refine is not None:
            a["refine"].append(st.refine)
        for s, n in item.step.tokens.items():
            a["tokens"][s] = a["tokens"].get(s, 0) + int(n)
        if st.jepa is not None:
            j = a["jepa"]
            j["micro"] += 1
            for k in ("windows", "eligible", "patches", "masked", "tokens", "loss_sum"):
                j[k] += st.jepa[k]
        if st.anchor is not None:
            an = a["anchor"]
            an["micro"] += 1
            for k in ANCHOR_SUMS:
                an[k] += st.anchor[k]
        if st.zupt is not None:
            zs = a["zupt"]
            zs["micro"] += 1
            for k in ZUPT_SUMS:
                zs[k] += st.zupt[k]

    def zupt_record(self):
        """The zupt_anchor entry of a 'log' record (module docstring), from the interval's accumulators."""
        zs, W = self.acc["zupt"], max(self.acc["W"], 1)
        za = self.cfg.zupt_anchor
        return {"weight": float(za.weight) if self.sched.in_phase == "B" else 0.0, "sd": float(za.sd),
                "loss": zs["term_sum"] / W, "distance": zs["distance_sum"] / W, "still": zs["still_sum"] / W,
                "weighted": zs["weighted_sum"] / W, "windows": int(zs["windows"]), "strides": int(zs["strides"]),
                "micro_steps": int(zs["micro"]),
                "per_anchored_window": zs["term_sum"] / zs["windows"] if zs["windows"] else None,
                "gain": zs["D_sum"] / zs["S_sum"] if zs["S_sum"] > 0 else None,
                "speed_at_instants": zs["speed_sum"] / (2 * zs["strides"]) if zs["strides"] else None,
                "seconds": zs["T_sum"]}

    def zupt_args(self, item, phase):
        """The ZuptArgs of an Item (module docstring), or None when the section is off."""
        if not self.zupt_on:
            return None
        za = self.cfg.zupt_anchor
        tg = ZA.ZuptTargets.empty() if self.zupt_table is None else \
            ZA.targets_for_step(self.zupt_table, item.step, float(za.edge_s))
        return ZuptArgs(tg, float(za.weight) if phase == "B" else 0.0, float(za.sd))

    def anchor_record(self):
        """The walkway_anchor entry of a 'log' record (module docstring), from the interval's accumulators."""
        an, W = self.acc["anchor"], max(self.acc["W"], 1)
        wa = self.cfg.walkway_anchor
        return {"weight": float(wa.weight) if self.sched.in_phase == "B" else 0.0, "sd": float(wa.sd),
                "loss": an["term_sum"] / W, "weighted": an["weighted_sum"] / W, "windows": int(an["windows"]),
                "segments": int(an["segments"]), "micro_steps": int(an["micro"]),
                "per_anchored_window": an["term_sum"] / an["windows"] if an["windows"] else None,
                "gain": an["S_sum"] / an["D_sum"] if an["D_sum"] > 0 else None,
                "chord_ratio": an["chord_sum"] / an["S_sum"] if an["S_sum"] > 0 else None,
                "seconds": an["T_sum"]}

    def anchor_args(self, item, phase):
        """The AnchorArgs of an Item (module docstring), or None when the section is off."""
        if not self.anchor_on:
            return None
        wa = self.cfg.walkway_anchor
        tg = WA.AnchorTargets.empty() if self.anchor_table is None else \
            WA.targets_for_step(self.anchor_table, item.step)
        return AnchorArgs(tg, float(wa.weight) if phase == "B" else 0.0, float(wa.sd))

    def jepa_record(self):
        """The jepa entry of a 'log' record (module docstring), from the interval's accumulators and the JEPA
        module's step-0 references."""
        j = self.acc["jepa"]
        jm = _jepa_of(self.model)
        first = j["first"] or {}
        st, sp = first.get("std_target"), first.get("std_pred")
        rt, rp = jm.references() if jm is not None else (None, None)
        ratio = None
        if None not in (st, sp, rt, rp) and rt > 0 and rp > 0:
            ratio = min(st / rt, sp / rp)
        return {"weight": float(self.cfg.jepa.weight),
                "loss": j["loss_sum"] / j["windows"] if j["windows"] else None,
                "windows": j["windows"], "eligible": j["eligible"],
                "masked_fraction": j["masked"] / j["patches"] if j["patches"] else None,
                "masked_tokens": j["tokens"], "micro_steps": j["micro"], "ema_decay": j["decay"],
                "std_target": st, "std_pred": sp, "ref_std_target": rt, "ref_std_pred": rp, "collapse_ratio": ratio,
                "grad_steps": j["grad_steps"], "time_s": j["time"] / j["timed"] if j["timed"] else None}

    def interval_record(self):
        """The 'log' record of the interval since the last one (module docstring)."""
        a, sc = self.acc, self.cfg.schedule
        wall = time.perf_counter() - a["t0"]
        W = max(a["W"], 1)
        terms = {k: a["sums"][k] / W for k in TERM_KEYS}
        tok = sum(a["tokens"].values())
        d_z = int(self.model.cfg.d_z)
        rec = {"event": "log", "phase": self.sched.in_phase, "step": self.sched.step, "lr": self.sched.lr(
            step=max(self.sched.step - 1, 0)), "loss": terms["per_window"], "terms": {k: terms[k] for k in LS.TERMS},
            "diagnostics": {k: terms[k] for k in SIDE_TERMS},
            "kl_z_per_dim": a["sums"]["kl_z"] / max(a["patches"] * d_z, 1),
            "active_dims": float(np.mean(a["active"])) if a["active"] else None,
            "windows": a["W"], "micro_steps": a["micro"], "optimizer_steps": a["steps"], "wall_s": wall,
            "tokens_per_s": a["useful"] / wall if wall > 0 else None,
            "padded_tokens_per_s": a["padded"] / wall if wall > 0 else None,
            "tokens_per_micro": a["useful"] / max(a["micro"], 1), "fill": a["useful"] / max(a["padded"], 1),
            "buckets": {str(k): v for k, v in sorted(a["buckets"].items())},
            "time_split_s": {k: v / a["timed"] for k, v in a["time"].items()} if a["timed"] else None,
            "memory_gb": self.memory(),
            "grad_norm": {"mean": float(np.mean(a["grad"])), "max": float(np.max(a["grad"]))} if a["grad"] else None,
            "skipped": a["skipped"], "token_share": {s: n / tok for s, n in sorted(a["tokens"].items())} if tok else {},
            "cumulative_synthetic_share": self.sched.synthetic_share(),
            "projected_synthetic_share": self.projected_synthetic_share(),
            "nuis_aux_synthetic": a["nuis_syn"] / a["synthetic"] if a["synthetic"] else None,
            "hsic": terms["hsic"], "hsic_active": terms["hsic_active"],
            "refine_objective": ({"first": float(np.mean([r[0] for r in a["refine"]])),
                                  "last": float(np.mean([r[1] for r in a["refine"]])),
                                  "drop": float(np.mean([r[0] - r[1] for r in a["refine"]])), "n": len(a["refine"])}
                                 if a["refine"] else None)}
        rec["refine_every"] = self.cadence()
        rec["micro_per_step"] = int(sc.micro_per_step)
        rec["health"] = a["health"]
        if self.jepa_on:
            rec["jepa"] = self.jepa_record()
        if self.anchor_on:
            rec["walkway_anchor"] = self.anchor_record()
        if self.zupt_on:
            rec["zupt_anchor"] = self.zupt_record()
        if self.window_cap_on and self.sched.in_phase == STAGE2_PHASE:     # task S2B: the capped quantity, phase C
            w = self.sched.tallies[STAGE2_PHASE]["windows"]
            n = sum(w.values())
            rec["window_share_phase"] = {s: v / n for s, v in sorted(w.items())} if n else {}
        return rec

    def check_alarms(self, rec, source):
        """The ALARMS of `source` ('log': rec a log record, its health; 'dev': rec the dev hook's dict, its
        'model' entry when present) crossed at the current step (module docstring): logs 'alarm' and, with
        schedule.stop_on_alarm, writes a checkpoint (unless this step's is written) and raises AlarmError.
        An alarm that schedule.alarm_consecutive names with k > 1 (config.ALARM_CONSECUTIVE_DOC, task S2BA) is
        raised only at the k-th crossing of its reads in a row; a crossing before it logs 'alarm_watch' and stops
        nothing (the run of reads counted again from log.jsonl at the first call of a phase, alarm_streaks).
        Returns the list of alarms."""
        step = int(self.sched.step)
        vals = alarm_values(rec, source)
        k_of = {str(n): int(k) for n, k in dict(getattr(self.cfg.schedule, "alarm_consecutive", None) or {}).items()
                if int(k) > 1}
        if k_of and (getattr(self, "alarm_streak", None) is None
                     or getattr(self, "alarm_streak_phase", None) != self.sched.in_phase):
            self.alarm_streak = alarm_streaks(self.log_path, self.sched.in_phase, step, k_of)
            self.alarm_streak_phase = self.sched.in_phase
        hits, watch = [], []
        for name, value in vals.items():
            src, side, line, first = ALARMS[name]
            if src != source or value is None or step < first:
                continue
            crossed = (side == "below" and value < line) or (side == "above" and value > line)
            if name not in k_of:
                if crossed:
                    hits.append({"name": name, "value": float(value), side: line, "from_step": first})
                continue
            n = self.alarm_streak.get(name, 0) + 1 if crossed else 0
            self.alarm_streak[name] = n
            if crossed:
                entry = {"name": name, "value": float(value), side: line, "from_step": first, "consecutive": n,
                         "of": k_of[name]}
                (hits if n >= k_of[name] else watch).append(entry)
        if watch:
            self.log({"event": "alarm_watch", "phase": self.sched.in_phase, "step": step, "source": source,
                      "alarms": watch})
            self.echo(f"[{self.sched.in_phase} {step:>6}] alarm watch: " + "; ".join(
                f"{w['name']} {w['value']:.4g} ({'below' if 'below' in w else 'above'} "
                f"{w.get('below', w.get('above'))}), {w['consecutive']} of {w['of']} in a row" for w in watch))
        if not hits:
            return hits
        self.log({"event": "alarm", "phase": self.sched.in_phase, "step": step, "source": source, "alarms": hits})
        self.echo(f"[{self.sched.in_phase} {step:>6}] ALARM " + "; ".join(
            f"{h['name']} {h['value']:.4g} ({'below' if 'below' in h else 'above'} "
            f"{h.get('below', h.get('above'))})" for h in hits))
        if bool(self.cfg.schedule.stop_on_alarm):
            path = os.path.join(self.run_dir, f"ckpt_{self.sched.in_phase}_{step:06d}.pt")
            if not self.checkpoints or os.path.abspath(self.checkpoints[-1]) != os.path.abspath(path):
                self.checkpoint()
            raise AlarmError(f"training alarm at phase {self.sched.in_phase} step {step}: "
                             + ", ".join(f"{h['name']} = {h['value']:.4g}" for h in hits))
        return hits

    def projected_synthetic_share(self):
        """The synthetic share of the whole run if the rest of its last phase (phase B; phase C for a stage-2 run,
        phases A and B being the source's tallies) draws as its micro-steps so far did (None before that phase has
        drawn)."""
        phases = self.sched.run_phases
        tb = self.sched.tallies[phases[-1]]
        if not tb["micro"]:
            return None
        done = [self.sched.tallies[p] for p in phases[:-1]] + [tb]
        per_micro = {s: n / tb["micro"] for s, n in tb["tokens"].items()}
        left = max(self.sched.steps[-1] * int(self.cfg.schedule.micro_per_step) - tb["micro"], 0)
        syn = sum(t["tokens"].get(S.SYNTHETIC, 0) for t in done) + left * per_micro.get(S.SYNTHETIC, 0.0)
        tot = sum(sum(t["tokens"].values()) for t in done) + left * sum(per_micro.values())
        return syn / tot if tot else None

    def check_synthetic_cap(self):
        """Log 'synthetic_cap', write a checkpoint and raise SyntheticCapError when projected_synthetic_share()
        exceeds mixture.synthetic_cap (module docstring). The projection is read once the last phase (B, or C for a
        stage-2 run) has drawn the micro-steps of one log interval (log_every x micro_per_step; fewer micro-steps
        vary too much in tokens) or has ended; nothing happens before."""
        sc, last = self.cfg.schedule, self.sched.run_phases[-1]
        drawn = self.sched.tallies[last]["micro"]
        ended = self.sched.in_phase == last and self.sched.step >= self.sched.n(last)
        if drawn < int(sc.log_every) * int(sc.micro_per_step) and not ended:
            return
        proj = self.projected_synthetic_share()
        cap = float(self.cfg.mixture.synthetic_cap)
        if proj is None or proj <= cap + 1e-12:
            return
        self.log({"event": "synthetic_cap", "phase": self.sched.in_phase, "step": self.sched.step,
                  "projected_synthetic_share": proj, "cumulative_synthetic_share": self.sched.synthetic_share(),
                  "cap": cap})
        self.checkpoint()
        raise SyntheticCapError(f"projected synthetic share of the run {proj:.4f} above the cap {cap} "
                                f"(phase {self.sched.in_phase}, step {self.sched.step})")

    def _console(self, rec, ph):
        mem = rec.get("memory_gb") or {}
        self.echo(f"[{ph} {rec['step']:>6}] loss {rec['loss']:.4g} lr {rec['lr']:.3g} "
                  f"tok/s {rec['tokens_per_s'] or 0:,.0f} grad {(rec['grad_norm'] or {}).get('mean', float('nan')):.3g} "
                  f"reserved {mem.get('max_reserved', 0):.2f} GB")

    # ------------------------------------------------------------ steps
    def cadence(self):
        """The refinement cadence N in force (schedule.refine_every unless overridden)."""
        return int(self.cfg.schedule.refine_every if self.refine_every is None else self.refine_every)

    def optimizer_step(self, timed=False):
        """One optimizer step of the current phase (module docstring); returns the list of MicroStats."""
        ph, sc = self.sched.in_phase, self.cfg.schedule
        items = [self.next_item() for _ in range(int(sc.micro_per_step))]
        n_windows = sum(it.mb.batch.n_windows for it in items)
        if self.acc["health"] is None:              # the first micro-step of the interval, before the update
            with self.fwd_lock if self.fwd_lock is not None else contextlib.nullcontext():
                self.acc["health"] = health(self.model, items[0].mb, device=self.device, amp=self.amp)
        jm = _jepa_of(self.model) if self.jepa_on and ph in self.jepa_in else None
        jgrad = None
        if jm is not None:
            if not jm.is_ready():                   # the target starts from the online weights of phase B's start
                jm.sync(self.model.tokenizer, self.model.encoder)
            if self.sched.step < self.jepa_grad_steps:
                jgrad = [None] * len(list(self.model.encoder.parameters()))
        lr = self.sched.apply(self.opt)
        ref = refined(ph, self.sched.step, self.cadence())
        progress = self.sched.step / max(self.sched.n("A"), 1) if ph == "A" else 1.0
        self.model.train()
        stats = []
        for i, it in enumerate(items):
            timer = Timer(self.device, sync=timed)
            for k in HOST_PARTS:
                timer.add(k, it.host[k])
            timer.start()
            jp = None
            if jm is not None:
                first = i == 0 and self.acc["jepa"]["first"] is None
                jp = JepaArgs.of(self.cfg.jepa, stats=first or jm.references()[0] is None, grad=jgrad)
            st = micro_step(self.model, it.mb, ph, device=self.device, n_windows=n_windows, loss_cfg=self.loss_cfg,
                            progress=progress, refine_steps=int(sc.refine_steps) if ref else 0,
                            refine_points=int(sc.refine_points), seed=it.step.seed, amp=self.amp, timer=timer,
                            lock=self.fwd_lock, jepa=jp, anchor=self.anchor_args(it, ph), zupt=self.zupt_args(it, ph))
            if jm is not None and first:
                self.acc["jepa"]["first"] = {"std_target": st.jepa["std_target"], "std_pred": st.jepa["std_pred"]}
            stats.append(st)
        t0 = time.perf_counter()
        if timed and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
        if jgrad is not None:
            g_j, g_r = jepa_grad_norms(self.model.encoder, jgrad)
            self.log({"event": "jepa_grad", "phase": ph, "step": self.sched.step, "g_J": g_j, "g_R": g_r})
            self.acc["jepa"]["grad_steps"] += 1
            del jgrad
        params = [p for p in self.model.parameters() if p.grad is not None]
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(sc.clip)))
        ok = math.isfinite(gn) and all(s.finite for s in stats)
        if ok:
            self.opt.step()
            ema_update(self.ema, self.model, float(sc.ema_decay), skip=self.ema_skip)
            if jm is not None:
                jc = self.cfg.jepa
                m = JP.ema_decay(self.sched.step, self.sched.n(ph), jc.ema_start, jc.ema_end)
                jm.update_target(self.model.tokenizer, self.model.encoder, m)
                self.acc["jepa"]["decay"] = m
            self.nonfinite = 0
        else:
            self.nonfinite += 1
            self.acc["skipped"] += 1
            self.log({"event": "nonfinite", "phase": ph, "step": self.sched.step, "grad_norm": gn})
        self.opt.zero_grad(set_to_none=True)
        if timed and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_opt = time.perf_counter() - t0
        for st in stats:
            st.time["optimizer"] = st.time.get("optimizer", 0.0) + t_opt / len(stats)
        for it, st in zip(items, stats):
            self.sched.count(ph, it.step, it.mb)
            self._accumulate(st, it)
            if timed:
                self.acc["timed"] += 1
                for k in PARTS:
                    self.acc["time"][k] += st.time.get(k, 0.0)
                if st.jepa is not None:
                    self.acc["jepa"]["timed"] += 1
                    self.acc["jepa"]["time"] += st.jepa["time_s"]
        if ok:
            self.acc["grad"].append(gn)
        self.acc["steps"] += 1
        self.sampler_snap = items[-1].state
        self.sched.step += 1
        self.last = {"lr": lr, "grad_norm": gn, "loss": sum(s.sums["per_window"] for s in stats) / max(n_windows, 1),
                     "refined": ref}
        self.check_memory()
        if self.nonfinite >= MAX_NONFINITE:
            raise FloatingPointError(f"{self.nonfinite} optimizer steps in a row with a non-finite loss or gradient")
        return stats

    # ------------------------------------------------------------ checkpoints and dev
    def provenance(self):
        """provenance.record of the run at its position: the window store from phase B on (both stores in phase C,
        their report mismatches each against its own report), the lineage of an init-from or a stage-2 run."""
        ws = self.window_store if self.sched.in_phase != PHASES[0] else None
        extra = {"loop": FORMAT, "optimizer": self.opt_kind, "phase": self.sched.in_phase, "step": self.sched.step}
        if self.init_source is not None:
            extra["init_from"] = dict(self.init_source)
        if self.stage2_source is not None:
            extra["migration"] = copy.deepcopy(self.stage2_source)
        rec = PV.record(self.cfg, self.body_store, ws, extra=extra)
        if isinstance(ws, S.Stage2Stores):
            rec["report_mismatches"] = ws.report_mismatches(self.body_store)
        return rec

    def checkpoint(self):
        """Write <run>/ckpt_<phase>_<step>.pt (module docstring); returns (path, SHA-256)."""
        ph, s = self.sched.in_phase, self.sched.step
        path = os.path.join(self.run_dir, f"ckpt_{ph}_{s:06d}.pt")
        sha = CK.save(path, self.model, self.ema, self.opt, self.sched, sampler_state(self.sampler, self.sampler_snap),
                      CK.rng_state(), self.cfg, self.provenance(), step=s)
        self.checkpoints.append(path)
        self.log({"event": "checkpoint", "phase": ph, "step": s, "path": path, "sha256": sha})
        self.echo(f"[{ph} {s:>6}] checkpoint {path}")
        return path, sha

    def dev(self, event):
        if self.dev_hook is None:
            return None
        was = self.model.training
        with self.fwd_lock if self.fwd_lock is not None else contextlib.nullcontext():
            out = self.dev_hook(self, event)
        self.model.train(was)
        if out is not None:
            self.log({"event": "dev", "trigger": event, "phase": self.sched.in_phase, "step": self.sched.step,
                      "dev": out})
            self.check_alarms(out, "dev")
        return out

    def resume(self, path):
        """Restore a checkpoint of this run's configuration (module docstring); returns the checkpoint dict."""
        ck = CK.load(path, map_location="cpu", cfg=self.cfg)
        if "config_sha256" in ck["drift"]:
            raise ValueError(f"{path} was written under another configuration")
        CK.restore(ck, self.model, self.ema, self.opt, self.sched, rng=True)
        src = (ck["provenance"].get("extra") or {}).get("init_from")
        if src is not None:                          # an init-from run keeps its lineage (module docstring)
            self.init_source = dict(src)
        mig = (ck["provenance"].get("extra") or {}).get("migration")
        if mig is not None and self.stage2_on:       # a stage-2 run keeps its migration lineage (Phase C)
            self.stage2_source = copy.deepcopy(mig)
        if self.sched.step < self.sched.n():
            self._start_sampler(self.sched.in_phase, ck["sampler"])
        self.log({"event": "resume", "path": os.path.abspath(path), "phase": self.sched.in_phase,
                  "step": self.sched.step, "drift": ck["drift"]})
        return ck

    def init_from(self, path):
        """Start phase B of this run from a checkpoint written at the end of phase A of another run (module
        docstring, Init-from); logs and returns the lineage record ('init_from'). ValueError: a trainer that has
        trained or holds checkpoints, a run folder holding a checkpoint, a checkpoint not written at the end of
        phase A, a stored optimizer state that does not fit; RuntimeError: model weights that do not fit (strict
        load); what checkpoint.load raises."""
        p = os.path.abspath(str(path))
        first, second = PHASES
        if self.stage2_on:
            raise ValueError("a stage-2 run starts phase C from a phase B end (init_stage2), not phase B (init_from)")
        if self.sched.in_phase != first or self.sched.step or self.checkpoints or self.sched.tallies[first]["micro"]:
            raise ValueError("init_from starts a fresh trainer")
        held = sorted(n for n in os.listdir(self.run_dir) if n.startswith("ckpt_") and n.endswith(".pt")) \
            if os.path.isdir(self.run_dir) else []
        if held:
            raise ValueError(f"the run folder {self.run_dir} holds checkpoints ({held[0]}, ...): init_from starts a "
                             f"new run; resume one of them instead")
        ck = CK.load(p, map_location="cpu", cfg=self.cfg)
        sc = ck.get("sched") or {}
        steps = [int(x) for x in sc.get("steps", ())]
        if sc.get("in_phase") != first or not steps or int(sc.get("step", -1)) != steps[0]:
            raise ValueError(f"{p} is at phase {sc.get('in_phase')} step {sc.get('step')} of {steps}: init_from "
                             f"reads a checkpoint written at the end of phase {first}")
        PV.assert_clean(ck["provenance"]["persons"], held_out=PV.held_out_persons(self.window_store),
                        tier=self.cfg.tier)
        have = self.model.state_dict()
        absent = sorted(set(have) - set(ck["model"]))
        self.model.load_state_dict(ck["model"], strict=True)
        self.ema.load_state_dict(ck["ema"])
        n_loaded, n_params = init_optimizer_state(self.opt, ck["opt"])
        self.sched.in_phase, self.sched.step = second, 0
        self.sched.tallies[first] = copy.deepcopy(sc["tallies"][first])
        CK.set_rng_state(ck["rng"])
        sha, how = checkpoint_sha256(p)
        self.init_source = {"path": p.replace("\\", "/"), "sha256": sha, "step": int(ck["step"]),
                            "config_sha256": ck["config_sha256"], "name": json.loads(ck["config"]).get("name")}
        jm = _jepa_of(self.model)
        rec = {"event": "init_from", "phase": self.sched.in_phase, "step": self.sched.step,
               "source": dict(self.init_source, sha256_from=how, steps=steps), "steps": list(self.sched.steps),
               "config_sha256": CF.sha256(self.cfg), "drift": list(ck["drift"]),
               "loaded": ["model", "ema", "opt", "rng", f"tallies {first}"],
               "optimizer_states": {"loaded": int(n_loaded), "parameters": int(n_params)},
               "model_entries_not_in_checkpoint": {"n": len(absent),
                                                   "values": int(sum(have[k].numel() for k in absent))},
               "jepa_target_ready": None if jm is None else bool(jm.is_ready())}
        del ck
        self.log(rec)
        self.echo(f"[{self.sched.in_phase} {self.sched.step:>6}] init from {p} (SHA-256 {sha[:12]}, step "
                  f"{self.init_source['step']}, config {self.init_source['name']})")
        return rec

    def check_lineage(self, ck, init_from):
        """ValueError unless the resumed checkpoint dict ck descends from the checkpoint init_from (the SHA-256
        recorded under extra.init_from of its provenance, checkpoint_sha256)."""
        src = (ck["provenance"].get("extra") or {}).get("init_from") or {}
        want, _ = checkpoint_sha256(init_from)
        if src.get("sha256") != want:
            raise ValueError(f"the resumed checkpoint descends from {src.get('path')} ({src.get('sha256')}), not "
                             f"from {init_from} ({want})")

    # ------------------------------------------------------------ the stage-2 entry (phase C)
    def stage2_plan(self, old, codes=None):
        """(migrate.MigrationPlan, codes) of the source's NameTables `old` to this run's (migrate.plan_for with
        stage2.task_rows; codes None: counted on the stage-1 replay store and the synthetic layouts, for a STAGE1
        source)."""
        from . import migrate as MG
        ws = self.window_store
        store = ws.replay if isinstance(ws, S.Stage2Stores) else None
        return MG.plan_for(old, CF.name_tables(self.cfg), self.cfg.stage2.task_rows, codes=codes, store=store)

    def _source_checkpoint(self, path):
        """checkpoint.load of a stage-2 source under this run's configuration (its tier refusing an INTERNAL source
        for a RELEASE run), a tolerated records drift logged in this run's folder, never beside the source."""
        return CK.load(path, map_location="cpu", cfg=self.cfg, drift_log=os.path.join(self.run_dir, CK.DRIFT_LOG))

    def init_stage2(self, path, plan=None):
        """Start phase C of this stage-2 run from a checkpoint written at the end of phase B of a stage-1 run (module
        docstring, Phase C); plan: the migrate.MigrationPlan of the source's name tables to this run's (default
        stage2_plan). Logs and returns the 'init_stage2' record. ValueError: a configuration without the stage2
        section, a trainer that has trained or holds checkpoints, a run folder holding a checkpoint, a source not at
        the end of its own phase B, a configuration that departs from the source's recipe (stage2_recipe; its drift
        is logged as recipe_drift), a plan of other name tables or task rows, an optimizer state that does not fit;
        provenance.ProvenanceError: an INTERNAL source for a RELEASE run; canonical.Refused: a source training person
        this run holds out or refuses at its tier; RuntimeError: weights that do not fit (strict load); what
        checkpoint.load raises."""
        from . import migrate as MG
        from ..kernels.nuisance import MountPriors
        if not self.stage2_on:
            raise ValueError("init_stage2 starts phase C of a configuration whose stage2 section is enabled")
        p = os.path.abspath(str(path))
        if (self.sched.in_phase != self.run_phases[0] or self.sched.step or self.checkpoints
                or any(t["micro"] for t in self.sched.tallies.values())):
            raise ValueError("init_stage2 starts a fresh trainer")
        held = sorted(n for n in os.listdir(self.run_dir) if n.startswith("ckpt_") and n.endswith(".pt")) \
            if os.path.isdir(self.run_dir) else []
        if held:
            raise ValueError(f"the run folder {self.run_dir} holds checkpoints ({held[0]}, ...): init_stage2 starts a "
                             f"new run; resume one of them instead")
        ck = self._source_checkpoint(p)
        if ck["tier"] != PV.RELEASE and self.cfg.tier == PV.RELEASE:
            raise PV.ProvenanceError(f"{p} is an {ck['tier']} checkpoint: a RELEASE run never starts from it")
        sc = ck.get("sched") or {}
        steps = [int(x) for x in sc.get("steps", ())]
        if sc.get("in_phase") != PHASES[1] or len(steps) != len(PHASES) or int(sc.get("step", -1)) != steps[1]:
            raise ValueError(f"{p} is at phase {sc.get('in_phase')} step {sc.get('step')} of {steps}: init_stage2 "
                             f"reads a checkpoint written at the end of phase {PHASES[1]} of a stage-1 run")
        PV.assert_clean(ck["provenance"]["persons"], held_out=PV.held_out_persons(self.window_store),
                        tier=self.cfg.tier)
        src_rc = MG._run_config(MG._config_dict(ck["config"]))
        departures, recipe_drift = stage2_recipe(src_rc, self.cfg, steps)
        if departures:
            raise ValueError(f"this stage-2 configuration departs from the start recipe of {p} in {departures} "
                             f"(spec section 4.6: the start recipe's sections are inherited unchanged)")
        old, new = MG.names_of(ck["config"]), CF.name_tables(self.cfg)
        codes = None
        if plan is None:
            plan, codes = self.stage2_plan(old)
        elif plan.old != old or plan.new != new or plan.task_rows != self.cfg.stage2.task_rows:
            raise ValueError(f"the plan maps {plan.old.sha256()} to {plan.new.sha256()} with task rows "
                             f"{plan.task_rows!r}; this run needs {old.sha256()} to {new.sha256()} with "
                             f"{self.cfg.stage2.task_rows!r}")
        priors, origin = MG.extend_priors(MountPriors.load(), new.mount_families)
        model_sd = MG.apply(ck["model"], plan, priors)
        ema_sd = MG.apply(ck["ema"], plan, priors)
        have = self.model.state_dict()
        absent = sorted(set(have) - set(model_sd))
        self.model.load_state_dict(model_sd, strict=True)
        self.ema.load_state_dict(ema_sd)
        mcfg = replace(self.cfg.model.mfm_config(), skeleton=False)
        src_jepa = MG._jepa_kwargs(ck["model"], src_rc)
        old_model = MG.build_model(mcfg, old, MG.priors_of_state(ck["model"], old.mount_families), src_jepa)
        new_model = MG.build_model(mcfg, new, priors, src_jepa)
        opt_sd = MG.apply_optimizer(ck["opt"], old_model, new_model, plan)
        n_dropped = 0
        if self.freeze:                          # the frozen parameters' states leave the optimizer (task S2B)
            opt_sd, n_dropped = freeze_optimizer_state(opt_sd, MG.param_names(new_model), self.ema_skip,
                                                       MG.param_names(self.model))
        del old_model, new_model
        n_loaded, n_params = init_optimizer_state(self.opt, opt_sd)
        self.sched.in_phase, self.sched.step = STAGE2_PHASE, 0
        for ph in PHASES:
            self.sched.tallies[ph] = copy.deepcopy(sc["tallies"][ph])
        CK.set_rng_state(ck["rng"])
        sha, how = checkpoint_sha256(p)
        extra = ck["provenance"].get("extra") or {}
        self.stage2_source = {
            "entry": "init_stage2",
            "migrated_from": {"path": p.replace("\\", "/"), "sha256": sha, "step": int(ck["step"]), "tier": ck["tier"],
                              "config_sha256": ck["config_sha256"], "name": json.loads(ck["config"]).get("name"),
                              "steps": steps},
            "names_sha256": {"old": old.sha256(), "new": new.sha256()}, "plan": plan.to_dict(),
            "plan_sha256": plan.sha256(), "codes": codes, "mount_origin": origin, "code_sha256": PV.code_sha256(),
            "source_drift": list(ck["drift"]),
            "source_lineage": {k: extra[k] for k in ("init_from", "migration") if k in extra}}
        jm = _jepa_of(self.model)
        rec = {"event": "init_stage2", "phase": self.sched.in_phase, "step": self.sched.step,
               "source": dict(self.stage2_source["migrated_from"], sha256_from=how), "steps": list(self.sched.steps),
               "config_sha256": CF.sha256(self.cfg), "drift": list(ck["drift"]),
               "names_sha256": dict(self.stage2_source["names_sha256"]), "plan_sha256": plan.sha256(),
               "trained_unknown": list(plan.trained_unknown), "task_rows": plan.task_rows,
               "loaded": ["model", "ema", "opt", "rng", "tallies A", "tallies B"],
               "optimizer_states": {"loaded": int(n_loaded), "parameters": int(n_params)},
               "model_entries_not_in_checkpoint": {"n": len(absent),
                                                   "values": int(sum(have[k].numel() for k in absent))},
               "cumulative_synthetic_share": self.sched.synthetic_share(),
               "recipe_drift": recipe_drift,
               "jepa_target_ready": None if jm is None else bool(jm.is_ready())}
        if self.freeze:
            rec["freeze"] = dict(self.freeze_counts(), optimizer_states_dropped=int(n_dropped))
        del ck
        self.log(rec)
        self.echo(f"[{self.sched.in_phase} {self.sched.step:>6}] stage 2 from {p} (SHA-256 {sha[:12]}, step "
                  f"{int(self.stage2_source['migrated_from']['step'])}, plan {plan.sha256()[:12]})")
        return rec

    def check_stage2_lineage(self, ck, source, plan=None):
        """ValueError unless the resumed checkpoint dict ck descends from the stage-2 source checkpoint `source` (the
        SHA-256 recorded under extra.migration.migrated_from, checkpoint_sha256) through the plan of that source under
        this configuration (its SHA-256; plan: given, else stage2_plan of the source's names)."""
        from . import migrate as MG
        mig = (ck["provenance"].get("extra") or {}).get("migration") or {}
        frm = mig.get("migrated_from") or {}
        want, _ = checkpoint_sha256(source)
        if frm.get("sha256") != want:
            raise ValueError(f"the resumed checkpoint descends from {frm.get('path')} ({frm.get('sha256')}), not from "
                             f"{source} ({want})")
        if plan is None:
            src = self._source_checkpoint(source)
            plan, _ = self.stage2_plan(MG.names_of(src["config"]))
            del src
        if mig.get("plan_sha256") != plan.sha256():
            raise ValueError(f"the resumed checkpoint was migrated under the plan {mig.get('plan_sha256')}, this run's "
                             f"plan of {source} is {plan.sha256()}")

    # ------------------------------------------------------------ the run
    def freeze_counts(self):
        """The phase C freeze of the run (freeze_parameters) without the name list, with the optimizer's parameters
        and tensors (module docstring); None without a freeze."""
        if not self.freeze:
            return None
        ps = [p for g in self.opt.param_groups for p in g["params"]]
        return dict({k: v for k, v in self.freeze.items() if k != "names"},
                    optimizer_parameters=int(sum(p.numel() for p in ps)), optimizer_tensors=len(ps))

    def start_record(self):
        rec = {"event": "start", "device": str(self.device), "torch": torch.__version__, "amp_bf16": self.amp,
               "optimizer": self.opt_kind, "config_sha256": CF.sha256(self.cfg), "name": self.cfg.name,
               "seed": int(self.cfg.seed), "steps": list(self.sched.steps), "prefetch": self.prefetch,
               "parameters": int(sum(p.numel() for p in self.model.parameters())),
               "persons": {s: len(v) for s, v in PV.person_lists(self.body_store, self.window_store).items()},
               "report_mismatches": self.report_mismatches, "run_dir": self.run_dir,
               "memory_cap_gb": self.memory_cap_gb, "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
               "decoder_checkpoint": self.decoder_checkpoint}
        jm = _jepa_of(self.model)
        if self.jepa_on and jm is not None:
            rec["jepa"] = {"section": CF.to_dict(self.cfg.jepa), "grad_steps": self.jepa_grad_steps,
                           "predictor_parameters": int(sum(p.numel() for p in jm.predictor.parameters())),
                           "target_parameters": int(sum(p.numel() for p in jm.target_parameters())),
                           "target_ready": jm.is_ready(), "ref_std": list(jm.references())}
        if self.render_aug is not None:
            rec["render_aug"] = CF.to_dict(self.cfg.render_aug)
        if self.anchor_on:
            rec["walkway_anchor"] = {"section": CF.to_dict(self.cfg.walkway_anchor), "table": self.anchor_check}
        if self.zupt_on:
            rec["zupt_anchor"] = {"section": CF.to_dict(self.cfg.zupt_anchor), "table": self.zupt_check}
        if self.collate_kw:
            nm = CF.name_tables(self.cfg)
            rec["names"] = {"sha256": nm.sha256(), "vocabulary_sha256": nm.vocabulary_sha256()}
        if self.stage2_on:
            rec["stage2"] = {"section": CF.to_dict(self.cfg.stage2), "phases": list(self.run_phases),
                             "tier": self.cfg.tier}
            if self.freeze:
                rec["stage2"]["freeze"] = self.freeze_counts()
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            rec.update(gpu=torch.cuda.get_device_name(self.device), mem_get_info_gb={"free": free / GB,
                                                                                     "total": total / GB})
        return rec

    def check_order(self, phases):
        """ValueError when `phases` would start phase B while phase A is unfinished (module docstring); for a stage-2
        run, unless `phases` is phase C alone and the schedule is in phase C (after init_stage2 or a resume)."""
        if self.stage2_on:
            if tuple(phases) != (STAGE2_PHASE,):
                raise ValueError(f"a stage-2 run trains phase {STAGE2_PHASE} only, not {tuple(phases)}")
            if self.sched.in_phase != STAGE2_PHASE:
                raise ValueError(f"phase {STAGE2_PHASE} starts from a phase B end (init_stage2) or a checkpoint of "
                                 f"phase {STAGE2_PHASE} (resume)")
            return
        first = PHASES[0]
        if first in phases or not phases:
            return
        if self.sched.in_phase == first and self.sched.step < self.sched.n(first):
            raise ValueError(f"phases {tuple(phases)} leave phase {first} at step {self.sched.step} of "
                             f"{self.sched.n(first)}: phase B starts from a checkpoint written at the end of phase "
                             f"{first} (resume), or run phases {PHASES}")

    def run(self, phases=None, max_steps=None, resume=None, dev_at_start=True, init_from=None, init_stage2=None,
            stage2_plan=None):
        """Train through `phases` (module docstring; None: A and B, or C alone for a stage-2 run): at most max_steps
        optimizer steps in this call (None: to the end); resume: a checkpoint path; init_from: a checkpoint at the end
        of phase A of another run, phase B of this run starting from it (init_from; with resume, the resumed
        checkpoint must descend from it, check_lineage, and the resume is made); init_stage2: a checkpoint at the end
        of phase B of a stage-1 run, phase C of this stage-2 run starting from it (init_stage2, with stage2_plan as
        its plan; with resume, check_stage2_lineage, and the resume is made); check_order refuses phases that skip an
        unfinished phase A, and for a stage-2 run any phase but C. Returns a summary dict (phase, step, checkpoints,
        last log)."""
        if phases is None:
            phases = (STAGE2_PHASE,) if self.stage2_on else PHASES
        phases = tuple(phases)
        if any(p not in self.run_phases for p in phases):
            raise ValueError(f"phases {phases}, expected a subset of {self.run_phases}")
        if init_from is not None and init_stage2 is not None:
            raise ValueError("init_from (phase B of a stage-1 run) and init_stage2 (phase C) exclude each other")
        os.makedirs(self.run_dir, exist_ok=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.log(self.start_record())
        sc = self.cfg.schedule
        done, last = 0, None
        budget = float("inf") if max_steps is None else int(max_steps)
        try:
            if resume is not None:
                ck = self.resume(resume)
                if init_from is not None:
                    self.check_lineage(ck, init_from)
                if init_stage2 is not None:
                    self.check_stage2_lineage(ck, init_stage2, stage2_plan)
                del ck
            elif init_from is not None:
                self.init_from(init_from)
            elif init_stage2 is not None:
                self.init_stage2(init_stage2, stage2_plan)
            self.check_order(phases)
            if resume is None and init_from is None and dev_at_start:
                self.dev("start")
            for ph in self.run_phases:
                if self.run_phases.index(ph) < self.run_phases.index(self.sched.in_phase) or ph not in phases:
                    continue
                if done >= budget:
                    break
                if ph != self.sched.in_phase:
                    self.sched.in_phase, self.sched.step = ph, 0
                n = self.sched.n(ph)
                if self.sched.step >= n:
                    continue
                if self.sampler is None or self.sampler_phase != ph:
                    self._start_sampler(ph)
                self._reset_interval()
                while self.sched.step < n and done < budget:
                    self.optimizer_step(timed=self.sched.step % int(sc.log_every) == 0)
                    done += 1
                    s = self.sched.step
                    if s % int(sc.log_every) == 0 or s == n or done >= budget:
                        last = self.log(self.interval_record())
                        self._console(last, ph)
                        self.history.append(last)
                        self._reset_interval()
                        self.check_synthetic_cap()
                        self.check_alarms(last, "log")
                    if s < n and done < budget and s % int(sc.checkpoint_every) == 0:
                        self.checkpoint()
                    if s < n and done < budget and int(sc.dev_every) > 0 and s % int(sc.dev_every) == 0:
                        self.dev("interval")
                if self.sched.step < n:          # the call's step budget ends inside the phase
                    self.checkpoint()
                    self.dev("stop")
                    break
                self.log({"event": "phase_end", "phase": ph, "step": self.sched.step,
                          "tallies": self.sched.tallies[ph], "cumulative_synthetic_share": self.sched.synthetic_share()})
                self.checkpoint()
                self.dev("phase_end")
        finally:
            self._close_prefetch()
        summary = {"event": "end", "phase": self.sched.in_phase, "step": self.sched.step, "optimizer_steps": done,
                   "checkpoints": list(self.checkpoints), "cumulative_synthetic_share": self.sched.synthetic_share(),
                   "synthetic_cap": float(self.cfg.mixture.synthetic_cap), "memory_gb": self.memory()}
        self.log(summary)
        return dict(summary, last=last)


def train(cfg, body_store, window_store=None, synth=None, device=None, run_dir=None, *, phases=None,
          max_steps=None, resume=None, **kw):
    """Trainer(cfg, ...).run(phases, max_steps, resume) (module docstring); returns (trainer, summary)."""
    tr = Trainer(cfg, body_store, window_store, synth, device, run_dir, **kw)
    return tr, tr.run(phases, max_steps=max_steps, resume=resume)


# ---------------------------------------------------------------- bench
def bench(trainer, phase, steps=200, timed=True, refine_every=None, warmup=3):
    """Measure `steps` optimizer steps of `phase` on a Trainer after `warmup` unmeasured ones (the weights,
    the optimizer, the schedule, set to step 0 of the phase, and a fresh sampler of the phase move; no
    checkpoint, no log line; a bench trainer does not train afterwards). timed: CUDA synchronized
    between the parts of every micro-step (the time split), otherwise the wall time alone (throughput, with the
    trainer's prefetch). refine_every overrides the cadence (phase B: 1 refines every step, 0 never). Peaks are
    reset before each optimizer step. Returns {phase, steps, timed, prefetch, refine_every, optimizer, wall_s
    (measured steps, CUDA synchronized at both ends), tokens, padded, peak_reserved_gb and peak_allocated_gb
    (max over the measured steps), micro: [per micro-step {row_len, useful, padded, W, time, peak_allocated_gb
    and peak_reserved_gb of its optimizer step, refined, loss}]}."""
    old = trainer.refine_every
    trainer.refine_every = None if refine_every is None else int(refine_every)
    dev = trainer.device
    cuda = dev.type == "cuda"
    try:
        trainer._close_prefetch()
        trainer.sched.in_phase, trainer.sched.step = phase, 0
        trainer._start_sampler(phase)
        micro, tokens, padded, wall, peak_a, peak_r = [], 0, 0, 0.0, 0.0, 0.0
        for k in range(int(warmup) + int(steps)):
            if cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            ref = refined(phase, trainer.sched.step, trainer.cadence())
            t0 = time.perf_counter()
            stats = trainer.optimizer_step(timed=timed)
            if cuda:
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            if k < int(warmup):
                continue
            wall += t1 - t0
            pa = torch.cuda.max_memory_allocated(dev) / GB if cuda else None
            pr = torch.cuda.max_memory_reserved(dev) / GB if cuda else None
            if cuda:
                peak_a, peak_r = max(peak_a, pa), max(peak_r, pr)
            for st in stats:
                tokens += st.useful
                padded += st.padded
                micro.append({"row_len": st.row_len, "useful": st.useful, "padded": st.padded, "W": st.W,
                              "time": dict(st.time), "peak_allocated_gb": pa, "peak_reserved_gb": pr, "refined": ref,
                              "loss": st.sums["per_window"] / max(st.W, 1)})
        return {"phase": phase, "steps": int(steps), "timed": bool(timed), "prefetch": trainer.prefetch,
                "refine_every": trainer.cadence(), "optimizer": trainer.opt_kind, "wall_s": wall, "tokens": tokens,
                "padded": padded, "peak_reserved_gb": peak_r if cuda else None,
                "peak_allocated_gb": peak_a if cuda else None, "micro": micro}
    finally:
        trainer.refine_every = old
        trainer._close_prefetch()
        trainer.sampler = None
        trainer._reset_interval()
