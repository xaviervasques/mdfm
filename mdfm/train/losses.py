"""Loss terms of the stage-1 pilot and the label-free check (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md,
section 4 and 2.6, task B4.16).

stage1_losses(model, batch, post, body, pred, phase) returns every term of section 4 as a (W,) tensor, the
diagnostics, the weighted per-window sum and its mean over the windows of the micro-step (the total); each
term is also a function of its own. forward(model, batch, post) draws z and beta from the posterior, decodes
the knots and renders the kernel points (body, pred) as the terms expect them.

Scale: every term is a per-window sum over the window's samples estimated from the sampled ones, so that
likelihood, supervision and KL keep ELBO proportions. The kernel likelihoods carry the point weights
(mfm.Points.weight, n_total / n_sampled of each stream). The knot-level terms are sums over the 100 Hz body
clock: L_w (the window's knots) times the mean over its decoded knots (knot_estimate), or times the weighted
mean over its kernel points (point_estimate). The total is the mean over the windows.

Terms, units and truth ([cfg] weights in WEIGHTS, phase A and phase B):
- imu_nll, pose3d_nll, pose2d_nll, root_nll, force_nll, emg_nll: the kernel negative log-likelihoods of
  mdfm/kernels at the band-copy samples of every observed stream, exactly as MFM.log_lik sums them, with the
  root track split from pose3d (kernel_terms). root_nll and emg_nll weigh in phase B only, force_nll on
  synthetic windows only in phase A. EMG is scored on the detached body (MFM.render).
- sup_rot, sup_pos, sup_root, sup_contact at every decoded knot of a window with a body truth (synthetic
  windows rendered from AMASS bodies; truth from Batch.truth; targets marking a truth that the batch lacks
  raise, truth_mask): squared geodesic angle of the 22 joint rotations (root in the window frame) over
  2 sigma_rot^2; pelvis-relative positions of joints 1 to 21 in each body's own heading frame (the root's
  yaw removed per knot), the decoded and the true knots both posed on the true skeleton (the true betas are
  given on these windows, so the term scores the pose and never the shape; reports/bench/phaseA_diagnosis.md,
  C4), over 2 sigma_pos^2, plus the Gaussian NLL of the decoder's pos_logvar against the detached mean (the
  band calibrates without pulling the mean); root translation, the horizontal error over 2 sigma_root_xz^2
  and the vertical over 2 sigma_root_y^2 (sup_root_parts; diagnosis C5b: a 2 cm horizontal scale on a root
  that is off by 0.4 m took 59 percent of the gradient; the horizontal scale is 0.50 m since
  reports/bench/root_motion_diagnosis.md, fix 5: a weak window-frame consistency, which under a heading error
  no longer rewards a shorter path), both parts logged as the diagnostics sup_root_xz and sup_root_y; binary
  cross-entropy of the contact logits against the truth contacts (Targets.contact,
  the body store's labels; without them the store rule on the truth kinematics at the knots: ankle or foot
  joint within frames.H0 of the y = 0 floor and slower than frames.V0).
- sup_root_rate at the root-grid points of those windows (decoder.RootGrid, BodyOut.root): the heading-frame
  pelvis velocity (lateral, forward) and the root yaw rate that the decoder integrates into the root path,
  against the truth's (truth_root_grid: the mean rates over the grid interval centred on each point, which the
  decoder's trapezoid integrates back into the truth's path without drift), over 2 sigma_root_vel^2 and
  2 sigma_root_yaw_rate^2, logged apart as sup_root_rate_u and sup_root_rate_yaw:
  local quantities, which a heading error elsewhere in the window does not enter (root-motion diagnosis C2 and
  C3, fix 3); only at the points where the truth's heading is defined (heading_defined: the root's forward axis
  more than 26 deg from the vertical, over the grid interval), since a lying, crawling or swimming root has no
  heading and its yaw rate is atan2 noise (up to 30 rad/s on a CMU swimming take of the G-b turning batch).
- sup_deriv at the kernel points of those windows: joint velocities and accelerations of the decoded body
  (the KnotBatch.kinematics of the draw) against those of the truth knots through the same Hermite and
  kinematics (Targets.kin when the batch carries the render's, else one no-grad kinematics call on
  Batch.truth), each body's vectors in its own heading frame at the point (the root's yaw removed: a constant
  heading error leaves the term unchanged, and the root velocity is scored as a local quantity; root-motion
  diagnosis C2, fix 4), and both in the window frame where the truth's heading is undefined
  (heading_defined), over 2 sigma_vel^2 and 2 sigma_acc^2.
- sup_beta: Gaussian NLL of the true betas under each window's shape posterior.
- The fixed scales SIGMAS [cfg] (rot 0.1 rad, pos 0.02 m, root_xz 0.50 m, root_y 0.02 m, vel 0.1 m/s, acc
  2 m/s^2, root_vel 0.1 m/s, root_yaw_rate 0.05 rad/s) come from LossConfig.sigmas (configs/stage1_pilot.yaml,
  losses.sigmas).
- heading_observable(batch): windows whose heading the input can observe (a valid gyroscope channel at an IMU
  kernel point, or any pose2d or pose3d point), read from the input channel masks, never a label; the dev
  numbers of train/loop.evaluate score the root motion there.
- Shape on windows with a body truth: forward(..., truth_windows) poses the kinematics of those windows on
  the drawn betas with their gradient stopped, the decoder's FiLM reads the betas detached, and sup_pos
  uses the true skeleton, so there the pose terms and the kernels never move the shape posterior: it learns
  from sup_beta, kl_beta, subject and the motion prior's condition (diagnosis C4). On windows without a
  body truth the kernel likelihood reaches the betas through the skeleton.
- kl_z: MotionPrior.kl of q(z) against p(z | sex, task, beta) given the sampled past, free bits 0.02 nats per
  dimension and valid patch; kl_beta: ShapePrior.kl against the shape prior of the window's gender. Both
  rise from 0 to 1 over the first KL_ANNEAL = 10 percent of phase A (progress, the share of phase A done).
- nuis_prior: -MFM.nuis_log_prior (mount tilt and yaw, IMU biases at the present sites; camera and pose3d
  scale flat).
- nuis_aux (synthetic windows, phase A): the rendered nuisance (Targets): mount error tilt^2 / sigma_tilt^2
  + yaw^2 / sigma_yaw^2 with the family's prior spreads, bias errors squared in units of the device prior's
  standard deviations, at the rendered sites and channels; camera pose error: squared geodesic of the
  rotation over CAMERA_AUX_SD[0]^2, squared distance of the camera centres over CAMERA_AUX_SD[1]^2, squared
  log focal ratio over CAMERA_AUX_SD[2]^2 [cfg] (the camera carries a flat prior, so no prior sd exists).
- consistency: windows of one render pair (Targets.pair, one body under two nuisance draws) share the
  symmetric KL between their q(z) over the patches valid in both, half to each window.
- plausibility: squared hinge outside the joint-limit percentiles of reports/body_store/priors.json
  (axis-angle per joint, root in the heading frame, at the decoded knots) over 2 LIMIT_SD^2; squared
  relu(-y) of the ankle and foot joints (floor y = 0) over 2 PENETRATION_SD^2; contact probability times the
  squared horizontal speed of each foot (mean of its ankle and foot joints) over 2 SKATE_SD^2, both at the
  kernel points. The contact probability enters detached: the term moves the feet and never lowers the
  contact of a stance on a real window, where nothing else would hold it up (the G3 contacts read it).
  Bone-length constancy needs no term (one beta per window through SkeletonBank, spec section 4): the
  diagnostic bone_spread (m, largest spread of a bone length over the window's points) shows it is 0 to
  rounding. Stage 2 (task S2.8; spec section 4.7 and section 9 item 9): LossConfig.plausibility_table names the
  joint-limit table (default reports/body_store/priors.json, the AMASS percentiles of the train persons; another
  table must hold the same joint_limits block) and LossConfig.plausibility_windows the windows the term weighs:
  'all' (default), or 'synthetic_replay', the synthetic and replay windows of a stage-2 micro-step only, which the
  training loop marks in Targets.plausibility_on (W,) bool (train/loop.py, phase C); a window marked False carries
  a plausibility term of 0 while its three diagnostics are still logged. Without the mark every window carries the
  term. outside_limits counts the decoded knots outside the table (the dev numbers per stage-2 unit).
- subject: persons with at least two windows in the micro-step: KL of q(beta | window) against the pooled
  q(beta | person), the normalized geometric mean of the windows' Gaussians (posterior.subject_pool's mean,
  its precision divided by the window count), so that the term is 0 when the windows agree. Against the
  product of Gaussians itself the KL at agreement is 8 (n - 1 - log n) nats per window for n windows and
  pulls every window's variance toward the product's.
- hsic: Gaussian-kernel HSIC between mean_p(mu_z) (W, 32) and the source code, per task stratum of
  docs/rules/task_vocabulary.yaml holding at least two sources in the micro-step (synthetic windows carry
  the code of the source 'synthetic'; LossConfig.hsic_synthetic False leaves them out); every window of an
  active stratum carries its stratum's value, 0 elsewhere; logged in both phases, weighted in B. Stage 2 (task
  S2.8; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7): with Targets.hsic_stratum (the opaque HSIC stratum of
  each window, data/strata.py, carried by Window.hsic_stratum and passed by collate through targets_from_windows)
  the strata are the (task, hsic stratum) pairs instead of the tasks; targets_from_windows leaves the field None
  when every window carries stratum 0, the default, so that a stage-1 batch takes the stage-1 computation itself.
- amortization (refined steps): sum over valid patches of (mu_z - sg(z_K))^2 / sg(sigma_z^2), mu_z the
  amortized mean (argument amortized) and z_K the refined one (post, straight-through or refined).

JEPA (optional; docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2.6): jepa_per_window turns the predictions and
target vectors of model/jepa.py into a (W,) term, the smooth L1 (beta JEPA_BETA = 1) between prediction and
target per masked patch averaged over the features, then averaged over the window's masked patches; 0 for a
window without masked patches. It is not one of TERMS (whose weights the configs list): train/loop.micro_step
adds it to per_window with the config's jepa.weight, in phase B only, when the config's jepa section is enabled.

Walkway anchor (optional; F2 lever 4, docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md, section A; task L4C):
walkway_anchor_parts turns the decoder's root grid and the segments of data/walkway_anchor.AnchorTargets (real
weargait_pd windows of the training-fold controls that hold a stride of a straight walkway pass) into a (W,) term: per
segment, the decoded forward distance (the integral of the forward heading-frame pelvis velocity between the
segment's knots) against the walkway distance of the same heels, as the squared mean-speed error over 2 sd^2 counted
at each of the segment's knots; 0 on every window without a segment. Like the JEPA term it is not one of TERMS:
train/loop.micro_step adds it to per_window with the config's walkway_anchor.weight in phase B only, when the
section is enabled (logged in both phases). The walkway distance is a body measurement of the window's own
recording (no diagnosis, group or person key); its targets are keyed by the window's index in the micro-step and
pass assert_label_free.

Inertial anchor (optional; F2 lever 5, docs/proposals/F2_LEVER5_DRAFT.md, section 2; task L5): zupt_anchor_parts turns
the draw's latents and betas and the strides of data/zupt_anchor.ZuptTargets (real weargait_pd windows of the
training-fold persons that hold a version-2 ZUPT stride of their own foot sensor) into a (W,) term with two parts per
stride: distance, rate (S - D)^2 / T / (2 sd^2) (that is n ((S - D) / T)^2 / (2 sd^2) with n = rate T the stride's
knots), D the horizontal displacement (window-frame x and z) of the decoded ankle joint of that foot between the two
still instants t_a and t_b, S the ZUPT stride and T = t_b - t_a; stillness, (v_a^2 + v_b^2) / (2 sd^2), v the decoded
ankle's horizontal speed at each instant; 0 on every window without a stride. The decoded ankle comes from one extra
decoder call on the anchored windows only, at the Hermite supports of the instants (no attention among queries and a
grid that depends on z, beta and the lengths only: the knots equal those of the main decode, model/decoder.py), and
the kinematics of those knots on the drawn betas with their gradient stopped (as the walkway anchor, its rule D4), so
the term reaches the decoder, the latent head and the encoder and never the shape posterior. Like the walkway anchor
it is not one of TERMS: train/loop.micro_step adds it to per_window with the config's zupt_anchor.weight in phase B
only, when the section is enabled (logged in both phases). Its targets come from the input's own foot sensors (no
reference instrument), are keyed by the window's index in the micro-step and pass assert_label_free.

No label enters any term: stage1_losses runs assert_label_free on the batch and the targets, and
targets_from_windows on the windows. assert_label_free raises LabelLeak on any key whose name holds label,
cohort, severity, diagnos, manifest or lockbox anywhere, or has a word (split at non-alphanumerics, camelCase
and acronym boundaries: subjectGroup, HCGroup, OFFphase) starting with a stem of data.windows.FORBIDDEN
(label, group, cohort, severity, diagnosis, phase, manifest, tier, lockbox, scope, lane, analog_labels,
notes, aux) or diagnos, at any depth: dict keys, dataclass and namedtuple fields, attributes of plain
objects, DataFrame columns, Series index labels, structured array fields, inside lists, tuples, sets and
object arrays. One structural field is admitted after a check: mfm.Points.group, the stream-window index of
the kernel points, when it is an int64 tensor whose every id holds the points of one window. CODY never
trains: a batch naming a CODY source and a CODY window given to targets_from_windows (without allow_eval)
are refused.
"""
import json
import math
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from ..body import frames
from ..body.hermite import hermite_gather, support
from ..body.kinematics import BodyKin, KnotBatch
from ..body.rotations import geodesic, matrix_to_rot6d, rot6d_to_matrix, rot_y, vee, yaw_angle
from ..body.joints import BODY_JOINTS, N_BODY, PARENTS, fk
from ..data.canonical import Refused, is_cody
from ..data.windows import FORBIDDEN
from ..kernels.emg import emg_nll
from ..kernels.force import contact_prob, force_nll
from ..kernels.imu import imu_nll
from ..kernels.nuisance import tilt_yaw2
from ..kernels.pose import pose2d_nll, pose3d_nll, root_nll, window_sum
from ..model.decoder import GRID_KNOTS, RootGrid, cumtrap_from_centre, grid_of, scatter_dense, yaw_rotate
from ..model.mfm import MODALITIES, N_SITES, Points
from ..model.posterior import group_ids, sample, subject_pool
from ..model.prior import FREE_BITS, LOG_2PI, PRIORS_PATH, gaussian_kl

PHASES = ("A", "B")
# [cfg] fixed scales of the direct supervision: rad, m, m, m, m/s, m/s^2, m/s, rad/s (reports/bench/
# phaseA_diagnosis.md, C5b; reports/bench/root_motion_diagnosis.md, fixes 3 and 5)
SIGMAS = {"rot": 0.1, "pos": 0.02, "root_xz": 0.50, "root_y": 0.02, "vel": 0.1, "acc": 2.0, "root_vel": 0.1,
          "root_yaw_rate": 0.05}
SIGMA_ROT = SIGMAS["rot"]           # rad, sup_rot
SIGMA_POS = SIGMAS["pos"]           # m, sup_pos
SIGMA_ROOT_XZ = SIGMAS["root_xz"]   # m, sup_root, horizontal root in the window frame (0.02, then 0.20 before)
SIGMA_ROOT_Y = SIGMAS["root_y"]     # m, sup_root, vertical root
SIGMA_VEL = SIGMAS["vel"]           # m/s, sup_deriv
SIGMA_ACC = SIGMAS["acc"]           # m/s^2, sup_deriv
SIGMA_ROOT_VEL = SIGMAS["root_vel"]             # m/s, sup_root_rate, heading-frame pelvis velocity at the grid
SIGMA_ROOT_YAW_RATE = SIGMAS["root_yaw_rate"]   # rad/s, sup_root_rate, root yaw rate at the grid
KL_ANNEAL = 0.1                     # share of phase A over which the KL weights rise from 0 to 1
LIMIT_SD = 0.05                     # [cfg] rad, joint-limit hinge
PENETRATION_SD = 0.01               # [cfg] m, foot joints below the floor
SKATE_SD = 0.1                      # [cfg] m/s, horizontal foot speed in contact
HEADING_DEFINED = 0.9               # [cfg] |y of the root's forward axis| under which the heading is defined
CAMERA_AUX_SD = (0.1, 0.25, 0.1)    # [cfg] rad, m, log focal: nuis_aux camera pose
HSIC_CODE_SD = 1.0                  # width of the Gaussian kernel on one-hot source codes
FLOOR_JOINTS = (7, 8, 10, 11)       # ankles and SMPL foot joints
FOOT_POINTS = ((7, 10), (8, 11))    # (ankle, foot) of the left and right foot
SYNTHETIC = "synthetic"

# term -> (weight in phase A, weight in phase B) [cfg]
WEIGHTS = {"imu_nll": (1.0, 1.0), "pose3d_nll": (1.0, 1.0), "pose2d_nll": (1.0, 1.0), "root_nll": (0.0, 1.0),
           "force_nll": (1.0, 1.0), "emg_nll": (0.0, 1.0),
           "sup_rot": (1.0, 1.0), "sup_pos": (1.0, 1.0), "sup_root": (1.0, 1.0), "sup_root_rate": (1.0, 1.0),
           "sup_deriv": (0.1, 0.1),
           "sup_contact": (1.0, 1.0), "sup_beta": (1.0, 1.0),
           "kl_z": (1.0, 1.0), "kl_beta": (1.0, 1.0),
           "nuis_prior": (1.0, 1.0), "nuis_aux": (0.1, 0.0), "consistency": (0.5, 0.5),
           "plausibility": (0.1, 0.1), "subject": (0.1, 0.1), "hsic": (0.0, 0.1), "amortization": (0.0, 0.1)}
TERMS = tuple(WEIGHTS)
KERNEL_TERMS = ("imu_nll", "pose3d_nll", "pose2d_nll", "root_nll", "force_nll", "emg_nll")
ANNEALED = ("kl_z", "kl_beta")
SYNTHETIC_ONLY_IN_A = ("force_nll",)
DIAGNOSTICS = ("sup_pos_band", "sup_root_xz", "sup_root_y", "sup_root_rate_u", "sup_root_rate_yaw", "plaus_limits",
               "plaus_penetration", "plaus_skate", "bone_spread", "hsic_active")


PLAUSIBILITY_WINDOWS = ("all", "synthetic_replay")    # LossConfig.plausibility_windows (module docstring)


@dataclass
class LossConfig:
    """Weights {term: (phase A, phase B)}, the KL annealing share of phase A, free bits (nats per latent
    dimension and patch), whether synthetic windows enter HSIC with their own code, the supervision
    scales {name: value} of SIGMAS [cfg], the joint-limit table of the plausibility term (a path, default
    prior.PRIORS_PATH) and the windows it weighs (PLAUSIBILITY_WINDOWS; module docstring, plausibility)."""
    weights: dict = field(default_factory=lambda: dict(WEIGHTS))
    kl_anneal: float = KL_ANNEAL
    free_bits: float = FREE_BITS
    hsic_synthetic: bool = True
    sigmas: dict = field(default_factory=lambda: dict(SIGMAS))
    plausibility_table: str = PRIORS_PATH
    plausibility_windows: str = "all"
    sup_rest: tuple = (0.0, 0.0)


LOSS = LossConfig()


# ---------------------------------------------------------------- label-free check
class LabelLeak(ValueError):
    """A key named like a label or a record field that windows.cut drops (module docstring)."""


# word stems (data.windows.FORBIDDEN plus 'diagnos' for diagnoses, diagnostic, diagnosed), and the stems that
# name a label wherever they sit inside a key
LABEL_STEMS = tuple(FORBIDDEN) + ("diagnos",)
LABEL_ANYWHERE = ("label", "cohort", "severity", "diagnos", "manifest", "lockbox")


def _word_starts(s):
    """Indices of s where a word may start: the first alphanumeric character, one after a non-alphanumeric
    character, and the case changes of camelCase (subject|Group), of an acronym before a capitalized word
    (PD|Label, HC|Group) and of an acronym before a lowercase word (OFF|phase)."""
    out = []
    for i, c in enumerate(s):
        if not c.isalnum():
            continue
        prev = s[i - 1] if i else ""
        prev2 = s[i - 2] if i > 1 else ""
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if (not prev.isalnum() or (c.isupper() and (prev.islower() or prev.isdigit()))
                or (c.isupper() and prev.isupper() and nxt.islower())
                or (c.islower() and prev.isupper() and prev2.isupper())):
            out.append(i)
    return out


def label_key(name):
    """True when the key name holds a stem of LABEL_ANYWHERE anywhere (sublabel, UPDRSLabel, diagnoses), or
    when a word of it (_word_starts: non-alphanumerics, camelCase and acronym boundaries) starts with a stem
    of data.windows.FORBIDDEN or LABEL_STEMS, case-insensitive."""
    s = str(name)
    low = s.lower()
    if any(st in low for st in LABEL_ANYWHERE):
        return True
    return any(low.startswith(LABEL_STEMS, i) for i in _word_starts(s))


def _stream_window_ids(points, g):
    """The field g of mfm.Points (its stream-window index) is one: an int64 tensor shaped like points.w,
    non-negative, every id's points in one window."""
    w = points.w
    if not (torch.is_tensor(g) and torch.is_tensor(w) and g.dtype == torch.int64 and g.shape == w.shape):
        return False
    if not g.numel():
        return True
    if bool((g < 0).any()):
        return False
    pairs = torch.unique(torch.stack([g, w.to(g.device, torch.int64)], 1), dim=0)
    return pairs.shape[0] == torch.unique(g).numel()


# (class, field) -> check; the only fields whose names match a stem and that are admitted
STRUCTURAL = {(Points, "group"): _stream_window_ids}
_LEAVES = (str, bytes, int, float, complex, bool, type(None), torch.Tensor, torch.dtype, torch.device)


def _is_frame(obj):
    return type(obj).__module__.startswith("pandas") and hasattr(obj, "columns") and hasattr(obj, "iloc")


def _is_series(obj):
    return type(obj).__module__.startswith("pandas") and hasattr(obj, "index") and hasattr(obj, "iloc") \
        and not hasattr(obj, "columns")


def label_paths(obj):
    """Paths (list of str) of every key inside obj that label_key flags (module docstring), and of every
    structural field that fails its check."""
    out, seen = [], set()

    def key(name, path):
        p = f"{path}.{name}" if path else str(name)
        if label_key(name):
            out.append(p)
        return p

    def walk(x, path):
        if isinstance(x, _LEAVES) or isinstance(x, (np.generic, types.ModuleType, type)) or callable(x):
            return
        if isinstance(x, np.ndarray):
            for name in x.dtype.names or ():
                key(name, path)
            if x.dtype == object:
                for i, v in enumerate(x.reshape(-1)):
                    walk(v, f"{path}[{i}]")
            return
        if id(x) in seen:
            return
        seen.add(id(x))
        if is_dataclass(x):
            for f in fields(x):
                check = STRUCTURAL.get((type(x), f.name))
                if check is not None:
                    if not check(x, getattr(x, f.name)):
                        out.append(f"{path}.{f.name}" if path else f.name)
                    continue
                walk(getattr(x, f.name), key(f.name, path))
        elif isinstance(x, tuple) and hasattr(x, "_fields"):
            for name, v in zip(x._fields, x):
                walk(v, key(name, path))
        elif isinstance(x, Mapping):
            for k, v in x.items():
                walk(v, key(k, path))
        elif isinstance(x, (list, tuple, set, frozenset)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif _is_frame(x):
            for c in list(x.columns) + [n for n in x.index.names if n is not None]:
                key(c, path)
        elif _is_series(x):
            for c in list(x.index) + ([x.name] if x.name is not None else []):
                key(c, path)
        elif isinstance(x, torch.nn.Module):
            return
        elif hasattr(x, "__dict__"):
            for k, v in vars(x).items():
                walk(v, key(k, path))

    walk(obj, "")
    return out


def assert_label_free(*objs):
    """Raise LabelLeak naming every path of a label key inside any of objs (a Batch, Targets, Windows, a
    record dict, nested containers; module docstring). None objects are skipped."""
    bad = [p for o in objs if o is not None for p in label_paths(o)]
    if bad:
        raise LabelLeak(f"{len(bad)} label key(s) reach the losses: {bad[:20]}")


# ---------------------------------------------------------------- targets
@dataclass
class Targets:
    """Truth of a micro-step beyond Batch.truth, W windows: has_truth (W,) bool (a body truth, synthetic
    windows); contact (W, T, 2) float32 truth contacts [left, right] on the knot clock, or None; kin BodyKin
    of the truth at every kernel point in MFM.kinematics order, or None; imu_mount (W, 14, 3, 3) sensor to
    segment, imu_bias_acc (W, 14, 3) g, imu_bias_gyro (W, 14, 3) deg/s, imu_known (W, 14, 2) bool (acc, gyro
    rendered at the site); cam_R (W, 3, 3) world to camera, cam_t (W, 3) m, cam_focal (W,) px, cam_known
    (W,) bool; pair (W,) int64 render-pair id, -1 unpaired; hsic_stratum (W,) int64 the windows' HSIC strata, None
    when every window carries stratum 0 (task S2.8, module docstring, hsic); plausibility_on (W,) bool the windows
    whose plausibility term weighs, None for every window (module docstring, plausibility)."""
    has_truth: torch.Tensor
    contact: torch.Tensor = None
    kin: BodyKin = None
    imu_mount: torch.Tensor = None
    imu_bias_acc: torch.Tensor = None
    imu_bias_gyro: torch.Tensor = None
    imu_known: torch.Tensor = None
    cam_R: torch.Tensor = None
    cam_t: torch.Tensor = None
    cam_focal: torch.Tensor = None
    cam_known: torch.Tensor = None
    pair: torch.Tensor = None
    hsic_stratum: torch.Tensor = None
    plausibility_on: torch.Tensor = None

    def to(self, device):
        def mv(x):
            if x is None:
                return None
            return BodyKin(*(y.to(device) for y in x)) if isinstance(x, BodyKin) else x.to(device)
        return Targets(**{f.name: mv(getattr(self, f.name)) for f in fields(self)})


def _pad_rows(x, T):
    x = np.asarray(x)
    if len(x) >= T:
        return x[:T]
    fill = x[-1:] if len(x) else np.zeros((1,) + x.shape[1:], x.dtype)
    return np.concatenate([x, np.repeat(fill, T - len(x), 0)])


def targets_from_windows(windows, T=None, allow_eval=False):
    """Targets of windows.Window objects in batch order: body truth (Window.body: contact over its lengths,
    padded with the last row to T knots, default the longest truth; a body without contact raises
    ValueError); IMU mounts and biases, and the camera,
    from the rendered streams' truth (Stream.truth: mount, bias_acc, bias_gyro at meta site_ids, channels of
    chan_mask; camera K, R, t); render pairs: synthetic windows with a body sharing key and length; hsic_stratum
    the windows' Window.hsic_stratum as int64, None when every one is 0 (module docstring, hsic). Runs
    assert_label_free on the windows; refuses a CODY source or person id (canonical.Refused) unless
    allow_eval=True (CODY never trains)."""
    wins = list(windows)
    assert_label_free(wins)
    for i, wd in enumerate(wins):
        if not allow_eval and is_cody(wd.source, wd.person_id):
            raise Refused(f"window {i}: CODY is evaluation-only ({wd.source} {wd.person_id})")
    W = len(wins)
    lengths = [int(wd.body["lengths"]) if wd.body is not None else 0 for wd in wins]
    T = max(lengths + [1]) if T is None else int(T)
    has = np.array([wd.body is not None for wd in wins], bool)
    contact = np.zeros((W, T, 2), np.float32)
    mount = np.tile(np.eye(3, dtype=np.float32), (W, N_SITES, 1, 1))
    ba, bg = np.zeros((W, N_SITES, 3), np.float32), np.zeros((W, N_SITES, 3), np.float32)
    known = np.zeros((W, N_SITES, 2), bool)
    cam_R, cam_t = np.tile(np.eye(3, dtype=np.float32), (W, 1, 1)), np.zeros((W, 3), np.float32)
    cam_f, cam_k = np.ones(W, np.float32), np.zeros(W, bool)
    pair = np.full(W, -1, np.int64)
    groups = {}
    for i, wd in enumerate(wins):
        if wd.body is not None:
            if wd.body.get("contact") is None:
                raise ValueError(f"window {i}: a body truth without contact labels (sup_contact would read "
                                 "every knot as flight)")
            contact[i] = _pad_rows(np.asarray(wd.body["contact"], np.float32)[:lengths[i]], T)
            if wd.synthetic:
                groups.setdefault((str(wd.key), lengths[i]), []).append(i)
        for s in wd.streams:
            tr = s.truth or {}
            if s.modality == "imu" and tr.get("mount") is not None:
                ids = np.asarray((s.meta or {})["site_ids"], np.int64)
                cm = np.asarray(s.chan_mask, bool)
                mount[i, ids] = np.asarray(tr["mount"], np.float32)
                ba[i, ids] = np.asarray(tr.get("bias_acc", np.zeros((len(ids), 3))), np.float32)
                bg[i, ids] = np.asarray(tr.get("bias_gyro", np.zeros((len(ids), 3))), np.float32)
                known[i, ids, 0] = cm[:, 0:3].any(-1)
                known[i, ids, 1] = cm[:, 3:6].any(-1)
            elif s.modality == "pose2d" and tr.get("camera") is not None:
                cam = tr["camera"]
                cam_R[i] = np.asarray(cam["R"], np.float32)
                cam_t[i] = np.asarray(cam["t"], np.float32).reshape(3)
                cam_f[i] = float(np.asarray(cam["K"])[0, 0])
                cam_k[i] = True
    for n, idx in enumerate(g for g in groups.values() if len(g) > 1):
        pair[idx] = n
    strata = np.array([int(getattr(wd, "hsic_stratum", 0)) for wd in wins], np.int64)
    if (strata < 0).any():
        raise ValueError(f"negative HSIC strata {sorted(set(strata[strata < 0].tolist()))}")
    t = torch.from_numpy
    return Targets(t(has), t(contact), None, t(mount), t(ba), t(bg), t(known), t(cam_R), t(cam_t), t(cam_f),
                   t(cam_k), t(pair), t(strata) if strata.any() else None)


# ---------------------------------------------------------------- the forward pass the terms read
class BodyDraw(NamedTuple):
    """One posterior draw of a micro-step: z (W, P, 32) and beta (W, 16); out the decoder's BodyOut at
    batch.knots; dense the KnotBody of the decoded knots; kin the output of MFM.kinematics(dense, batch)
    (BodyKin at every kernel point in MODALITIES order, {modality: slice}, rest joints (W, 22, 3))."""
    z: torch.Tensor
    beta: torch.Tensor
    out: object
    dense: object
    kin: tuple


def kinematic_beta(beta, truth_windows=None):
    """The betas the kinematics pose the draw on: beta (W, 16) with its gradient stopped on the windows of
    truth_windows (W,) bool (a body truth, whose shape sup_beta supervises; module docstring), beta itself
    elsewhere or when truth_windows is None."""
    if truth_windows is None:
        return beta
    return torch.where(truth_windows.to(beta.device)[:, None], beta.detach(), beta)


def forward(model, batch, post, generator=None, pool=True, truth_windows=None):
    """(BodyDraw, pred): z drawn from q(z) (padding patches 0), beta from the shape posterior pooled per
    person (posterior.subject_pool; pool=False: each window's own), decoded at batch.knots, one kinematics
    call at every kernel point and MFM.render with post's nuisance. generator: a torch.Generator on the
    posterior's device. truth_windows (W,) bool or None: windows with a body truth, whose kinematics use the
    drawn betas with their gradient stopped (kinematic_beta); BodyDraw.beta keeps the drawn betas (the
    motion prior's condition)."""
    z = sample(post.mu_z, post.logvar_z, generator=generator) * post.mask[..., None]
    mu_b, lv_b = subject_pool(post.mu_b, post.logvar_b, batch.person.to(post.mu_b.device)) if pool else \
        (post.mu_b, post.logvar_b)
    beta = sample(mu_b, lv_b, generator=generator)
    out = model.decode(z, beta, batch.knots, batch.tokens.n_patches.to(z.device), batch.lengths.to(z.device))
    dense = model.dense(out, kinematic_beta(beta, truth_windows), batch)
    kin = model.kinematics(dense, batch)
    return BodyDraw(z, beta, out, dense, kin), model.render(dense, batch, post.nuis, kin)


# ---------------------------------------------------------------- helpers
def _zeros(W, device, dtype=torch.float32):
    return torch.zeros(W, dtype=dtype, device=device)


def knot_estimate(x, w, lengths, n_windows):
    """(W,) per-window sum over the L_w knots of the clock from values x (M,) at M sampled knots of windows w
    (M,): L_w times their mean (0 for a window without a sample)."""
    s = window_sum(x, w, n_windows)
    n = window_sum(torch.ones_like(x), w, n_windows)
    return s / n.clamp_min(1.0) * lengths.to(x)


def point_estimate(x, w, weight, lengths, n_windows):
    """(W,) per-window sum over the L_w knots from values x (N,) at kernel points of windows w with weights
    (N,): L_w times their weighted mean (0 for a window without a point)."""
    wt = weight.to(x)
    s = window_sum(x * wt, w, n_windows)
    n = window_sum(wt, w, n_windows)
    return s / torch.where(n > 0, n, torch.ones_like(n)) * lengths.to(x)


def point_index(batch, device=None):
    """(w, t, weight) (N,) of every kernel point in MFM.kinematics order (MODALITIES, streams with points)."""
    mods = [m for m in MODALITIES if m in batch.points and batch.points[m].w.numel()]
    if not mods:
        e = torch.zeros(0, device=device)
        return e.long(), e, e
    cat = lambda name: torch.cat([getattr(batch.points[m], name) for m in mods]).to(device)  # noqa: E731
    return cat("w"), cat("t"), cat("weight")


def is_synthetic(batch, device=None):
    """(W,) bool: the window's source is 'synthetic'."""
    sources = tuple(batch.sources)
    if SYNTHETIC not in sources:
        return torch.zeros(batch.n_windows, dtype=torch.bool, device=device)
    return batch.source_id.to(device) == sources.index(SYNTHETIC)


def heading_observable(batch, device=None):
    """(W,) bool: windows whose heading the input can observe, from the input channel masks of the kernel points
    (never a label): a valid gyroscope channel at an IMU point of the window, or any valid pose2d or pose3d
    point. Accelerometer-only layouts (nonan3, wrist_acc) leave it False (root-motion diagnosis C5)."""
    W = batch.n_windows
    out = torch.zeros(W, dtype=torch.bool, device=device)
    for m, p in batch.points.items():
        if not p.w.numel() or m not in ("imu", "pose2d", "pose3d"):
            continue
        ok = p.valid.to(device)
        ok = ok[..., 3:6] if m == "imu" else ok
        ok = ok.reshape(ok.shape[0], -1).any(-1)
        out[p.w.to(device)[ok]] = True
    return out


def truth_mask(batch, targets=None, device=None):
    """(W,) bool: windows with a body truth (Targets.has_truth, default every window when Batch.truth
    exists). Targets marking a truth that the batch does not carry raise ValueError: batch_from_windows keeps
    Batch.truth only when every window has a body, and a mixed batch would lose its direct supervision
    without a word."""
    W = batch.n_windows
    has_t = None if targets is None else targets.has_truth
    if has_t is not None and tuple(has_t.shape) != (W,):
        raise ValueError(f"Targets.has_truth has shape {tuple(has_t.shape)}, the batch {W} windows")
    if batch.truth is None:
        if has_t is not None and bool(has_t.any()):
            raise ValueError(f"targets mark {int(has_t.sum())} of {W} windows with a body truth, but the batch "
                             "carries none (Batch.truth None): direct supervision would be off")
        return torch.zeros(W, dtype=torch.bool, device=device)
    if has_t is not None:
        return has_t.to(device, torch.bool)
    return torch.ones(W, dtype=torch.bool, device=device)


def log_map(R):
    """Axis-angle (..., 3) rad of rotations R (..., 3, 3): the angle atan2(|vee(R)|, (tr R - 1) / 2) in
    [0, pi] along vee(R), as body/rotations.matrix_to_axis_angle, with finite gradients at the identity."""
    v = vee(R)
    s = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-24)
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2)[..., None]
    return v * (torch.atan2(s, c) / s)


@lru_cache(maxsize=None)
def _limits(path):
    with open(path, encoding="utf-8") as f:
        jl = json.load(f)["joint_limits"]
    if list(jl["joints"]) != list(BODY_JOINTS):
        raise ValueError(f"{path}: joint_limits joints differ from smplh.BODY_JOINTS")
    lo, hi = np.asarray(jl["lo"], np.float64), np.asarray(jl["hi"], np.float64)
    if lo.shape != (N_BODY, 3) or hi.shape != (N_BODY, 3) or not (lo <= hi).all():
        raise ValueError(f"{path}: joint_limits lo and hi must be (22, 3) with lo <= hi")
    return lo, hi


def joint_limits(dtype=torch.float32, device=None, path=PRIORS_PATH):
    """(lo, hi) (22, 3) rad: the 0.1 and 99.9 percentiles of each axis-angle component per joint over the
    train knots (root in the heading frame), reports/body_store/priors.json."""
    lo, hi = _limits(path)
    return torch.as_tensor(lo, dtype=dtype, device=device), torch.as_tensor(hi, dtype=dtype, device=device)


def heading_relative(pos, R_glob):
    """Positions of joints 1 to 21 relative to the pelvis (M, 21, 3) m in each knot's heading frame (the
    root's yaw removed) from joint positions (M, 22, 3) and global rotations (M, 22, 3, 3)."""
    H = rot_y(-yaw_angle(R_glob[:, 0]))
    return torch.einsum("mij,mkj->mki", H, pos[:, 1:] - pos[:, :1])


# ---------------------------------------------------------------- kernel likelihoods
def kernel_terms(model, pred, batch):
    """{imu_nll, pose3d_nll, pose2d_nll, root_nll, force_nll, emg_nll} (W,) of the predictions of
    MFM.render: the kernels exactly as MFM.nll_terms calls them, the root track apart from pose3d; 0 for a
    modality without points. Their sum is -MFM.log_lik(pred, batch)."""
    W = batch.n_windows
    dev = next((v.device for pr in pred.values() for v in pr.values() if torch.is_tensor(v)), model.device)
    out = {k: _zeros(W, dev) for k in KERNEL_TERMS}
    with torch.autocast(device_type=dev.type, enabled=False):
        win = {k: v.to(dev) for k, v in batch.win.items()}
        for m, pr in pred.items():
            p = batch.points[m]
            w, wt = p.w.to(dev), p.weight.to(dev)
            if m == "imu":
                out["imu_nll"] = imu_nll((pr["acc"], pr["gyro"]), p.obs.to(dev), p.valid.to(dev), pr["log_scale"],
                                         wt, w, W)
            elif m == "pose3d":
                j = pr["joints"]
                out["pose3d_nll"], yaw = pose3d_nll(j, p.obs.to(dev, j.dtype), win["p3_scale"], win["p3_unit"],
                                                    p.valid.to(dev), pr["log_sigma"], w, W, z_up=win["p3_zup"],
                                                    weight=wt, return_yaw=True)
                if "root" in p.extra:
                    out["root_nll"] = root_nll(pr["pelvis"], p.extra["root"].to(dev, j.dtype),
                                               p.extra["root_valid"].to(dev), w, W, yaw=yaw.detach(), weight=wt)
            elif m == "pose2d":
                j = pr["joints"]
                out["pose2d_nll"] = pose2d_nll(j, pr["camera"], p.obs.to(dev, j.dtype), win["p2_scale"],
                                               p.valid.to(dev), w, W, sigma0=pr["sigma_px"], weight=wt)
            elif m == "force":
                out["force_nll"] = force_nll(pr["bw"], p.obs.to(dev), w, win["f_mask"], p.valid.to(dev),
                                             model.force_log_sigma, weight=wt, offset=win["f_offset"],
                                             flight=pr["flight"])
            elif m == "emg":
                out["emg_nll"] = emg_nll(pr["log_env"], p.obs.to(dev), p.valid.to(dev) & pr["mask"], w,
                                         model.emg_log_sigma, weight=wt, n_windows=W)
            else:
                raise ValueError(f"no kernel term for modality {m!r}")
    return {k: v.float() for k, v in out.items()}


# ---------------------------------------------------------------- direct supervision
def _truth_knots(out, has):
    keep = has.to(out.w.device)[out.w]
    return keep, out.w[keep], out.k[keep]


def sup_rot(out, truth, lengths, has, sigma=SIGMA_ROT):
    """(W,) sum over knots of the squared geodesic angles of the 22 joint rotations (root in the window
    frame) between the decoded knots out (BodyOut) and truth (KnotBody) over 2 sigma^2, windows with truth."""
    W = len(lengths)
    keep, w, k = _truth_knots(out, has)
    R = rot6d_to_matrix(out.rot6d[keep])
    Rt = rot6d_to_matrix(truth.rot6d.to(R)[w, k])
    e = (geodesic(R, Rt) ** 2).sum(-1) / (2 * sigma ** 2)
    return knot_estimate(e, w, lengths, W)


def sup_pos_parts(out, truth, gender, skeleton, lengths, has, sigma=SIGMA_POS):
    """(squared error, band NLL) (W,) each, summed over knots: pelvis-relative joint positions (joints 1 to
    21, each body in its own heading frame) of the decoded knots against the truth knots, both posed on the
    true skeleton (truth.beta, which is given on these windows), so that the term scores the pose and never
    the shape (reports/bench/phaseA_diagnosis.md, C4: posed on the drawn betas, a skeleton 5 percent short
    paid off 30 times its sup_beta cost), over 2 sigma^2; the Gaussian NLL of the detached error under the
    decoder's isotropic pos_logvar (log m^2)."""
    W = len(lengths)
    keep, w, k = _truth_knots(out, has)
    gender = gender.to(out.rot6d.device)
    rest_t = skeleton.rest(truth.beta.to(out.rot6d), gender)
    pos, Rg = fk(rot6d_to_matrix(out.rot6d[keep]), out.transl[keep], rest_t[w])
    pos_t, Rg_t = fk(rot6d_to_matrix(truth.rot6d.to(out.rot6d)[w, k]), truth.transl.to(out.rot6d)[w, k], rest_t[w])
    rel, rel_t = heading_relative(pos, Rg), heading_relative(pos_t, Rg_t)
    se = ((rel - rel_t) ** 2).sum((-1, -2)) / (2 * sigma ** 2)
    d2 = ((rel.detach() - rel_t) ** 2).sum(-1)
    lv = out.pos_logvar[keep][:, 1:]
    band = 0.5 * (d2 * torch.exp(-lv) + 3 * lv + 3 * LOG_2PI).sum(-1)
    return knot_estimate(se, w, lengths, W), knot_estimate(band, w, lengths, W)


def sup_root_parts(out, truth, lengths, has, sigma_xz=SIGMA_ROOT_XZ, sigma_y=SIGMA_ROOT_Y):
    """(horizontal, vertical) (W,) sums over knots of the squared root translation errors (m, window frame)
    over 2 sigma_xz^2 and 2 sigma_y^2 (reports/bench/phaseA_diagnosis.md, C5b: 2 cm on a horizontal position
    that is off by 0.4 m took 59 percent of the gradient)."""
    W = len(lengths)
    keep, w, k = _truth_knots(out, has)
    d = out.transl[keep] - truth.transl.to(out.transl)[w, k]
    xz = (d[:, 0] ** 2 + d[:, 2] ** 2) / (2 * sigma_xz ** 2)
    return knot_estimate(xz, w, lengths, W), knot_estimate(d[:, 1] ** 2 / (2 * sigma_y ** 2), w, lengths, W)


def sup_root(out, truth, lengths, has, sigma_xz=SIGMA_ROOT_XZ, sigma_y=SIGMA_ROOT_Y):
    """(W,) the sum of the two parts of sup_root_parts."""
    xz, y = sup_root_parts(out, truth, lengths, has, sigma_xz, sigma_y)
    return xz + y


def _wrap(a):
    """Angles (rad) wrapped to [-pi, pi)."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def heading_defined(R):
    """(...) bool: root rotations R (..., 3, 3) whose forward axis (+z) lies more than acos(HEADING_DEFINED) = 26
    deg from the vertical, so that its projection on the floor, the heading of body/rotations.yaw_angle, is
    defined; a lying, crawling or swimming root has none."""
    return R[..., 1, 2].abs() < HEADING_DEFINED


def _interp(truth, w, t, L):
    """(yaw (N,) rad, pelvis xz (N, 2) m) of the truth knots at fractional knots t (N,) of windows w, clamped to
    [0, L - 1], linear between knots (the yaw unwrapped between its two knots)."""
    t = torch.minimum(torch.maximum(t, torch.zeros_like(t)), (L - 1).to(t.dtype))
    k0 = torch.floor(t).long()
    k1 = torch.minimum(k0 + 1, L - 1)
    f = t - k0.to(t.dtype)
    yaw = lambda kk: yaw_angle(rot6d_to_matrix(truth.rot6d[w, kk, 0]))          # noqa: E731
    y0 = yaw(k0)
    y = y0 + f.to(y0.dtype) * _wrap(yaw(k1) - y0)
    p = truth.transl[w, k0][:, [0, 2]] * (1 - f)[:, None].to(y0.dtype) + truth.transl[w, k1][:, [0, 2]] * f[:, None].to(
        y0.dtype)
    return y, p


def truth_root_rates(truth, w, k, rate, step=GRID_KNOTS):
    """(psi (G,) rad, omega (G,) rad/s, V (G, 2) m/s) of the truth knots (KnotBody) at knots (w, k) (G,) inside
    their windows: the root yaw in the window frame (-pi to pi) at the knot, and the mean yaw rate and mean
    window-frame pelvis velocity (x, z) over the grid interval centred on the knot, [k - step / 2, k + step / 2]
    clamped to the window (linear between knots; the pelvis moves as the root translation, the SMPL root joint
    being the translation plus a constant rest offset). Interval means, not instantaneous rates: under the
    decoder's cumulative trapezoid they telescope into the local mean of the path, where central differences
    alias the within-step motion and the mocap jitter into a drift (reports/bench/root_grid.md)."""
    w, k = w.to(truth.transl.device), k.to(truth.transl.device)
    L = truth.lengths.to(w.device).long()[w]
    dt = truth.transl.dtype
    tk = k.to(dt)
    a = torch.minimum(torch.maximum(tk + step / 2, torch.zeros_like(tk)), (L - 1).to(dt))
    b = torch.minimum(torch.maximum(tk - step / 2, torch.zeros_like(tk)), (L - 1).to(dt))
    ya, pa = _interp(truth, w, a, L)
    yb, pb = _interp(truth, w, b, L)
    span = ((a - b) / rate).clamp_min(1.0 / rate)
    psi = yaw_angle(rot6d_to_matrix(truth.rot6d[w, k, 0]))
    return psi, _wrap(ya - yb) / span, (pa - pb) / span[:, None]


def truth_root_grid(truth, rate, lengths=None, step=GRID_KNOTS):
    """RootGrid (decoder.RootGrid) of the truth knots (KnotBody) at the root grid of the windows
    (decoder.grid_of of lengths, default truth.lengths), in the order of the decoder's BodyOut.root: omega the
    interval-mean yaw rate of truth_root_rates; u its interval-mean pelvis velocity in the heading frame of the
    truth's own integrated yaw (decoder.cumtrap_from_centre of omega), so that decoder.integrate_root of (omega,
    u) gives back the truth's path to its local mean (the decoder's targets); psi the true yaw about the centre
    knot's (wrapped to -pi to pi; 0 at the centre in the window frame) and X the pelvis displacement (x, z) from
    the centre knot, both at the grid knot (the dev numbers read them); valid the points inside their window whose
    heading is defined (heading_defined at the knot and at both ends of its grid interval; points outside read the
    rates and the pose of the nearest end knot, an extrapolation never supervised)."""
    lengths = truth.lengths if lengths is None else lengths
    dev = truth.transl.device
    g = grid_of(torch.as_tensor(lengths, device=dev), step)
    L = torch.as_tensor(lengths, device=dev).long()[g.w]
    kc = torch.minimum(torch.maximum(g.k, torch.zeros_like(g.k)), L - 1)
    ends = [torch.minimum(torch.maximum(kc + d, torch.zeros_like(kc)), L - 1) for d in (-(step // 2), 0, step // 2)]
    defined = torch.stack([heading_defined(rot6d_to_matrix(truth.rot6d[g.w, e, 0])) for e in ends]).all(0)
    valid = (g.k >= 0) & (g.k < L) & defined
    psi, omega, V = truth_root_rates(truth, g.w, kc, rate, step)
    u = yaw_rotate(V, -cumtrap_from_centre(omega, g, step / rate))
    c = g.c[g.w]
    psi_c = yaw_angle(rot6d_to_matrix(truth.rot6d[g.w, c, 0]))
    X = (truth.transl[g.w, kc] - truth.transl[g.w, c])[:, [0, 2]]
    return RootGrid(g.w, g.k, _wrap(psi - psi_c), omega, u, X, valid)


def sup_root_rate_parts(out, truth, lengths, has, rate, sigma_u=SIGMA_ROOT_VEL, sigma_w=SIGMA_ROOT_YAW_RATE):
    """(velocity, yaw rate) (W,) sums over knots, estimated at the root-grid points inside each window
    (BodyOut.root, decoder.RootGrid), of the squared error of the heading-frame pelvis velocity over
    2 sigma_u^2 and of the root yaw rate over 2 sigma_w^2 against the truth's (truth_root_grid), windows with
    truth, at the points where the truth's heading is defined (truth_root_grid's valid): local quantities, which
    a heading error elsewhere in the window does not enter (reports/bench/root_motion_diagnosis.md, C2 and C3,
    fix 3). A BodyOut without a root grid, or with another grid than the lengths give, raises ValueError."""
    W = len(lengths)
    g = out.root
    if g is None:
        raise ValueError("sup_root_rate needs the root grid of the decoded body (BodyOut.root): decode with the "
                         "window lengths")
    tg = truth_root_grid(truth, rate, lengths)
    if not torch.equal(g.k, tg.k.to(g.k.device)) or not torch.equal(g.w, tg.w.to(g.w.device)):
        raise ValueError("the decoded root grid is not the grid of the window lengths")
    keep = has.to(g.w.device)[g.w] & g.valid & tg.valid.to(g.w.device)
    w = g.w[keep]
    eu = ((g.u[keep] - tg.u.to(g.u)[keep]) ** 2).sum(-1) / (2 * sigma_u ** 2)
    ew = (g.omega[keep] - tg.omega.to(g.omega)[keep]) ** 2 / (2 * sigma_w ** 2)
    return knot_estimate(eu, w, lengths, W), knot_estimate(ew, w, lengths, W)


def sup_contact(out, contact, lengths, has):
    """(W,) sum over knots of the binary cross-entropy of the decoded contact logits (M, 2) against the
    truth contacts (W, T, 2) in [0, 1], both feet."""
    W = len(lengths)
    keep, w, k = _truth_knots(out, has)
    logit = out.contact_logit[keep]
    e = F.binary_cross_entropy_with_logits(logit, contact.to(logit)[w, k], reduction="none").sum(-1)
    return knot_estimate(e, w, lengths, W)


def sup_deriv(kin, kin_true, w, weight, lengths, has, sigma_v=SIGMA_VEL, sigma_a=SIGMA_ACC):
    """(W,) sum over knots, estimated at the kernel points (w, weight (N,)), of the squared joint velocity
    and acceleration errors of the decoded body (BodyKin kin) against the truth (kin_true), each body's vectors
    in its own heading frame at the point (the root yaw removed, so that a heading error never enters the term
    and the root velocity is scored as a local quantity; reports/bench/root_motion_diagnosis.md, C2, fix 4),
    over 2 sigma_v^2 and 2 sigma_a^2, windows with truth. Where the truth's heading is undefined
    (heading_defined) both bodies take the truth's yaw: the window-frame errors. An infinite sigma drops its
    part."""
    W = len(lengths)
    keep = has.to(w.device)[w]
    Rt = kin_true.R.to(kin.v)[keep][:, 0]
    yt = yaw_angle(Rt)
    Ht = rot_y(-yt)
    H = rot_y(-torch.where(heading_defined(Rt), yaw_angle(kin.R[keep][:, 0]), yt))
    loc = lambda Hm, x: torch.einsum("nij,nkj->nki", Hm, x)                         # noqa: E731
    ev = ((loc(H, kin.v[keep]) - loc(Ht, kin_true.v.to(kin.v)[keep])) ** 2).sum((-1, -2)) / (2 * sigma_v ** 2)
    ea = ((loc(H, kin.a[keep]) - loc(Ht, kin_true.a.to(kin.a)[keep])) ** 2).sum((-1, -2)) / (2 * sigma_a ** 2)
    return point_estimate(ev + ea, w[keep], weight[keep], lengths, W)


def sup_rest(mu_b, beta_true, gender, skeleton, has, sigma=SIGMA_POS):
    """(W,) the rest-joint supervision of the open port (docs/MODEL_OPEN_PORT_PLAN.md, step M4; LossConfig.sup_rest,
    off by default): the 22 rest joints of the posterior mean shape mu_b against those of the true shape, both through
    the model's skeleton (m, the rest frame), squared and summed over the joints and axes over 2 sigma^2 (sigma the
    scale of sup_pos), windows with truth. sup_pos poses the truth on the true skeleton and never scores the shape
    (sup_pos_parts); on the open body one unit of the phenotype latent moves the rest height three times as far as
    one unit of beta on SMPL+H (reports/bench_open/shape_sensitivity.md), so the shape NLL alone left the rest height
    4.2 cm off on the learnability batch (reports/bench_open/learnability.md)."""
    g = gender.to(mu_b.device)
    rest_p = skeleton.rest(mu_b.float(), g)
    with torch.no_grad():
        rest_t = skeleton.rest(beta_true.to(mu_b.device).float(), g)
    se = ((rest_p - rest_t) ** 2).sum((-1, -2)) / (2 * sigma ** 2)
    return torch.where(has.to(mu_b.device), se, torch.zeros_like(se))


def sup_beta(mu_b, logvar_b, beta_true, has):
    """(W,) Gaussian NLL of the true betas (W, 16) under the shape posterior N(mu_b, exp(logvar_b)), windows
    with truth."""
    nll = 0.5 * (((beta_true.to(mu_b) - mu_b) ** 2) * torch.exp(-logvar_b) + logvar_b + LOG_2PI).sum(-1)
    return torch.where(has.to(mu_b.device), nll, torch.zeros_like(nll))


def truth_kinematics(model, batch):
    """BodyKin of Batch.truth at every kernel point (MFM.kinematics order), without gradient."""
    with torch.no_grad():
        return model.kinematics(batch.truth.to(model.device), batch)[0]


def truth_contacts(model, batch, w, k):
    """(M, 2) float32 truth contacts at knots (w, k) by the body store's rule on the truth kinematics: ankle
    or foot joint within frames.H0 of the y = 0 floor and slower than frames.V0 (3D speed)."""
    dev = model.device
    tr = batch.truth.to(dev)
    with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
        rest = model.skeleton.rest(tr.beta.float(), tr.gender)
        kb = KnotBatch(tr.rot6d.float(), tr.transl.float(), rest, batch.rate, tr.lengths)
        kin = kb.kinematics(w.to(dev), k.to(dev).float() / batch.rate)
        near = (kin.p[..., 1] < frames.H0) & (kin.v.norm(dim=-1) < frames.V0)
        return (near[:, list(frames.ANKLES)] | near[:, list(frames.FEET)]).float()


# ---------------------------------------------------------------- nuisance terms
def nuis_prior(model, nuis, batch):
    """(W,) -log p(nuisance | layout family): MFM.nuis_log_prior negated."""
    return -model.nuis_log_prior(nuis, batch).float()


def nuis_aux(model, nuis, batch, targets, pred=None, camera_sd=CAMERA_AUX_SD):
    """(W,) error of the posterior nuisance against the rendered one (module docstring), windows with
    truth: IMU mounts and biases at the rendered sites and channels; the camera of pred['pose2d'] (the
    Camera MFM.render built from the camera nuisance) against Targets' camera."""
    W = batch.n_windows
    dev = nuis.imu.mount_res6d.device if nuis.imu is not None else model.device
    out = _zeros(W, dev)
    if targets is None:
        return out
    has = targets.has_truth.to(dev)
    if nuis.imu is not None and targets.imu_known is not None and bool(targets.imu_known.any()):
        pr = model.heads.priors()
        fam = batch.win["imu_family"].to(dev)
        M = nuis.imu.mount(pr.rotation(fam))
        dt = M.dtype
        E = targets.imu_mount.to(dev, dt).transpose(-1, -2) @ M
        tilt2, yaw2 = tilt_yaw2(E, pr.per_site(pr.up_sensor, fam).to(dt))
        known = targets.imu_known.to(dev)
        mount = (tilt2 / pr.per_site(pr.sigma_tilt, fam).to(dt) ** 2
                 + yaw2 / pr.per_site(pr.sigma_yaw, fam).to(dt) ** 2) * known.any(-1)
        ea = (((nuis.imu.bias_acc - targets.imu_bias_acc.to(dev, dt)) / pr.per_site(pr.bias_acc_sd, fam).to(dt))
              ** 2).sum(-1) * known[..., 0]
        eg = (((nuis.imu.bias_gyro - targets.imu_bias_gyro.to(dev, dt)) / pr.per_site(pr.bias_gyro_sd, fam).to(dt))
              ** 2).sum(-1) * known[..., 1]
        out = out + (mount + ea + eg).sum(-1).float()
    cam = (pred or {}).get("pose2d", {}).get("camera")
    if cam is not None and targets.cam_known is not None and bool(targets.cam_known.any()):
        R, t = cam.R, cam.t
        Rt, tt = targets.cam_R.to(dev, R.dtype), targets.cam_t.to(dev, R.dtype)
        c = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
        ct = -(Rt.transpose(-1, -2) @ tt[..., None])[..., 0]
        e = (geodesic(R, Rt) ** 2 / camera_sd[0] ** 2 + ((c - ct) ** 2).sum(-1) / camera_sd[1] ** 2
             + (torch.log(cam.K[:, 0, 0]) - torch.log(targets.cam_focal.to(dev, R.dtype))) ** 2 / camera_sd[2] ** 2)
        out = out + torch.where(targets.cam_known.to(dev), e, torch.zeros_like(e)).float()
    return torch.where(has, out, torch.zeros_like(out))


# ---------------------------------------------------------------- latent terms
def consistency(mu_z, logvar_z, mask, pair):
    """(W,) symmetric KL between the q(z) (W, P, 32) of the windows of each render pair (pair (W,) int64, -1
    unpaired) over the patches valid in both, half of each pair's value on each of its windows."""
    W = mu_z.shape[0]
    out = _zeros(W, mu_z.device, mu_z.dtype)
    if pair is None:
        return out
    pair = pair.to(mu_z.device)
    same = (pair[:, None] == pair[None]) & (pair[:, None] >= 0)
    a, b = torch.nonzero(torch.triu(same, diagonal=1), as_tuple=True)
    if not a.numel():
        return out
    m = (mask[a] & mask[b]).to(mu_z.dtype)[..., None]
    ab = gaussian_kl(mu_z[a], logvar_z[a], mu_z[b], 0.5 * logvar_z[b])
    ba = gaussian_kl(mu_z[b], logvar_z[b], mu_z[a], 0.5 * logvar_z[a])
    skl = ((ab + ba) * m).sum((1, 2))
    return out.index_add(0, a, 0.5 * skl).index_add(0, b, 0.5 * skl)


def subject(mu_b, logvar_b, person):
    """(W,) KL of each window's q(beta) (W, 16) against the normalized geometric mean of the Gaussians of
    its person's windows in the micro-step (module docstring), 0 for a person with one window."""
    gid = group_ids(person).to(mu_b.device)
    n = torch.bincount(gid).to(mu_b.dtype)[gid]
    mu_p, lv_p = subject_pool(mu_b, logvar_b, gid)
    lv_c = lv_p + torch.log(n)[:, None]
    kl = gaussian_kl(mu_b, logvar_b, mu_p, 0.5 * lv_c).sum(-1)
    return torch.where(n >= 2, kl, torch.zeros_like(kl))


def hsic_gaussian(x, codes, code_sd=HSIC_CODE_SD):
    """Biased HSIC (1 / n^2) tr(K H L H) between rows of x (n, d) and integer codes (n,): K Gaussian on x
    with exp(-d^2 / m), m the median of the off-diagonal squared distances (detached; 1 when 0); L Gaussian
    on the one-hot codes with width code_sd."""
    n = x.shape[0]
    sq = (x * x).sum(-1)
    d2 = (sq[:, None] + sq[None] - 2 * x @ x.T).clamp_min(0.0)
    off = d2.detach()[~torch.eye(n, dtype=torch.bool, device=x.device)]
    med = off.median() if off.numel() else torch.ones((), dtype=x.dtype, device=x.device)
    med = torch.where(med > 0, med, torch.ones_like(med))
    K = torch.exp(-d2 / med)
    diff = (codes[:, None] != codes[None]).to(x.dtype)
    L = torch.exp(-2.0 * diff / (2 * code_sd ** 2))
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    return (Kc * L).sum() / (n * n)


def hsic(mu_z, mask, source_id, task, synthetic=None, include_synthetic=True, stratum=None):
    """((W,) HSIC, (W,) bool active): per task stratum with at least two source codes among its windows
    (synthetic windows (W,) bool left out when include_synthetic is False), the HSIC between the window
    summaries mean_p(mu_z) over valid patches and the source codes; every window of an active stratum carries
    its stratum's value. stratum (W,) int64 (task S2.8): the windows' HSIC strata, the strata then being the
    (task, stratum) pairs; None, the stage-1 computation (module docstring, hsic)."""
    W = mu_z.shape[0]
    dev = mu_z.device
    m = mask.to(mu_z.dtype)[..., None]
    summ = (mu_z * m).sum(1) / m.sum(1).clamp_min(1.0)
    src, task = source_id.to(dev), task.to(dev)
    if stratum is not None:
        st = stratum.to(dev, torch.int64)
        if st.shape != task.shape or bool((st < 0).any()) or bool((task < 0).any()):
            raise ValueError("hsic strata: one non-negative stratum and task per window")
        task = task.to(torch.int64) * (int(st.max()) + 1) + st if st.numel() else task
    elig = torch.ones(W, dtype=torch.bool, device=dev)
    if synthetic is not None and not include_synthetic:
        elig = ~synthetic.to(dev)
    out = _zeros(W, dev, mu_z.dtype)
    active = torch.zeros(W, dtype=torch.bool, device=dev)
    for s in torch.unique(task[elig]).tolist():
        sel = elig & (task == s)
        if torch.unique(src[sel]).numel() < 2:
            continue
        out = torch.where(sel, hsic_gaussian(summ[sel], src[sel]), out)
        active = active | sel
    return out, active


def amortization(amortized, refined_mu_z):
    """(W,) sum over valid patches of (mu_z - sg(z_K))^2 / sg(exp(logvar_z)) of the amortized posterior
    (mu_z, logvar_z, mask) against the refined means z_K (W, P, 32)."""
    m = amortized.mask.to(amortized.mu_z.dtype)[..., None]
    d = amortized.mu_z - refined_mu_z.detach()
    return (d * d * torch.exp(-amortized.logvar_z.detach()) * m).sum((1, 2))


# ---------------------------------------------------------------- plausibility
def limit_hinge(rot6d, lo, hi, sd=LIMIT_SD):
    """(M,) sum over joints and axis-angle components of the squared excess outside [lo, hi] (22, 3) rad
    over 2 sd^2, of knots rot6d (M, 22, 6) (local rotations, root in the window frame, taken in its heading
    frame)."""
    R = rot6d_to_matrix(rot6d)
    root = rot_y(-yaw_angle(R[:, 0])) @ R[:, 0]
    aa = log_map(torch.cat([root[:, None], R[:, 1:]], 1))
    ex = F.relu(lo.to(aa) - aa) + F.relu(aa - hi.to(aa))
    return (ex * ex).sum((-1, -2)) / (2 * sd ** 2)


def outside_limits(rot6d, lo, hi):
    """(M,) bool: knots rot6d (M, 22, 6) with any axis-angle component outside [lo, hi] (22, 3) rad, the angles of
    limit_hinge (root in its heading frame); the decoded knots outside the plausibility table (task S2.8)."""
    R = rot6d_to_matrix(rot6d)
    root = rot_y(-yaw_angle(R[:, 0])) @ R[:, 0]
    aa = log_map(torch.cat([root[:, None], R[:, 1:]], 1))
    return ((aa < lo.to(aa)) | (aa > hi.to(aa))).flatten(1).any(-1)


def plausibility_limits(cfg, dtype=torch.float32, device=None):
    """(lo, hi) of LossConfig cfg's plausibility table (joint_limits of its path); None for the default table
    (prior.PRIORS_PATH), which plausibility reads itself (module docstring, plausibility)."""
    path = getattr(cfg, "plausibility_table", PRIORS_PATH)
    if os.path.normcase(os.path.normpath(str(path))) == os.path.normcase(os.path.normpath(PRIORS_PATH)):
        return None
    return joint_limits(dtype, device, path=str(path))


def penetration(p, sd=PENETRATION_SD):
    """(N,) squared depth below the y = 0 floor of the ankle and foot joints of positions p (N, 22, 3) m over
    2 sd^2."""
    d = F.relu(-p[:, list(FLOOR_JOINTS), 1])
    return (d * d).sum(-1) / (2 * sd ** 2)


def skate(v, contact, sd=SKATE_SD):
    """(N,) contact probability (N, 2) (detached) times the squared horizontal speed of each foot (mean of its
    ankle and foot joints, velocities v (N, 22, 3) m/s) over 2 sd^2, summed over the feet."""
    vf = v[:, torch.tensor(FOOT_POINTS, device=v.device)].mean(2)
    return (contact.detach().to(v) * (vf[..., 0] ** 2 + vf[..., 2] ** 2)).sum(-1) / (2 * sd ** 2)


def contact_at_points(body, kin, w, t, rate):
    """(N, 2) contact probability at the kernel points: the sigmoid of the body's contact logits through
    Hermite, or kernels.force.contact_prob of the kinematics when the body has none."""
    if body.contact_logit is not None:
        return torch.sigmoid(hermite_gather(body.contact_logit.to(kin.p), w, t.to(kin.p), rate, body.lengths))
    return contact_prob(kin.p, kin.v)


def plausibility(out, body, kin, w, t, weight, lengths, rate, limits=None):
    """{plausibility, plaus_limits, plaus_penetration, plaus_skate} (W,) each (module docstring): the joint
    limit hinge at the decoded knots out (BodyOut), penetration and skate at the kernel points (kin BodyKin,
    w, t, weight (N,)) of the dense body (KnotBody)."""
    W = len(lengths)
    lo, hi = joint_limits(out.rot6d.dtype, out.rot6d.device) if limits is None else limits
    lim = knot_estimate(limit_hinge(out.rot6d, lo, hi), out.w, lengths, W)
    if kin is None or not w.numel():
        z = torch.zeros_like(lim)
        return {"plausibility": lim, "plaus_limits": lim, "plaus_penetration": z, "plaus_skate": z}
    pen = point_estimate(penetration(kin.p), w, weight, lengths, W)
    sk = point_estimate(skate(kin.v, contact_at_points(body, kin, w, t, rate)), w, weight, lengths, W)
    return {"plausibility": lim + pen + sk, "plaus_limits": lim, "plaus_penetration": pen, "plaus_skate": sk}


def bone_spread(p, w, n_windows):
    """(W,) m: the largest spread (max minus min) of a bone length |p_j - p_parent(j)| over the points of
    each window, positions p (N, 22, 3); 0 without points. Zero to rounding when one rest skeleton poses
    every point of a window."""
    L = (p[:, 1:] - p[:, list(PARENTS[1:N_BODY])]).norm(dim=-1).detach()
    idx = w[:, None].expand_as(L)
    z = torch.zeros(n_windows, L.shape[1], dtype=L.dtype, device=L.device)
    hi = z.scatter_reduce(0, idx, L, reduce="amax", include_self=False)
    lo = z.scatter_reduce(0, idx, L, reduce="amin", include_self=False)
    return (hi - lo).amax(-1)


# ---------------------------------------------------------------- weights and the total
def term_weights(phase, progress=1.0, cfg=None):
    """{term: weight} of a phase ('A' or 'B'); progress in [0, 1] the share of phase A done (the KL terms
    rise linearly to their weight over the first cfg.kl_anneal of phase A)."""
    cfg = LOSS if cfg is None else cfg
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    i = PHASES.index(phase)
    out = {t: float(cfg.weights[t][i]) for t in TERMS}
    if phase == "A":
        ramp = 1.0 if cfg.kl_anneal <= 0 else min(1.0, max(0.0, float(progress)) / cfg.kl_anneal)
        for t in ANNEALED:
            out[t] *= ramp
    return out


JEPA_BETA = 1.0                     # smooth L1 transition of the JEPA term (proposal section 2.6)


def jepa_per_window(pred, target, window, n_windows, beta=JEPA_BETA):
    """(W,) float32 JEPA term (module docstring): smooth L1 (beta) between pred and target (N, d) per masked
    patch, mean over the d features, then the mean over the masked patches of each window (window (N,) int64);
    0 for a window without masked patches. Runs in float32, autocast off."""
    dev = pred.device
    W = int(n_windows)
    if not window.numel():
        return torch.zeros(W, dtype=torch.float32, device=dev)
    with torch.autocast(device_type=dev.type, enabled=False):
        e = F.smooth_l1_loss(pred.float(), target.float().to(dev), beta=float(beta), reduction="none").mean(-1)
        w = window.to(dev)
        s = torch.zeros(W, dtype=torch.float32, device=dev).index_add(0, w, e)
        n = torch.zeros(W, dtype=torch.float32, device=dev).index_add(0, w, torch.ones_like(e))
        return s / n.clamp_min(1.0)


WALKWAY_SD = 0.1                    # m/s, default scale of the walkway anchor (config walkway_anchor.sd)


def _grid_at(g, x, w, k, step):
    """(index a (N,), fraction s (N,)) of fractional knots k (N,) of windows w on the root grid g (decoder.grid_of):
    the grid interval [a, a + 1] of the window holding k (clamped to the window's first and last interval) and
    s = (k - knot(a)) / step."""
    d = k - g.c[w].to(k.dtype)
    j = torch.floor(d / step).long()
    j = torch.minimum(torch.maximum(j, g.lo[w]), g.lo[w] + g.n[w] - 2)
    s = d / step - j.to(k.dtype)
    return g.start[w] + (j - g.lo[w]), s.to(x.dtype)


def walkway_anchor_parts(root, lengths, anchor, rate, sd=WALKWAY_SD, step=GRID_KNOTS):
    """The walkway anchor of F2 lever 4 (docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md, section A): {term (W,),
    S, D, T, chord (N,) per segment}. root: the decoder's RootGrid (BodyOut.root) of windows of `lengths` knots;
    anchor: data.walkway_anchor.AnchorTargets (N segments; None or empty: term 0 everywhere). Per segment, S is the
    decoded forward distance (m), the integral of the forward component of the heading-frame pelvis velocity
    (root.u[:, 1]) between the fractional knots k_a and k_b, the velocity linear between grid points (the trapezoid
    of the grid, exact on the linear interpolant); D the walkway distance (m); T = (k_b - k_a) / rate (s); chord the
    window-frame horizontal pelvis displacement between the two knots (root.X linear between grid points; a
    diagnostic, never weighted). term_w is the sum over the window's segments of n ((S - D) / T)^2 / (2 sd^2) with
    n = rate T the segment's knots: sup_root_rate's unit, each knot counting the segment's mean-speed error. Float32,
    autocast off. Runs assert_label_free on the targets; a root grid other than the lengths' raises ValueError."""
    W = len(lengths)
    dev = root.u.device
    if anchor is None or anchor.n == 0:
        z = torch.zeros(0, device=dev)
        return {"term": torch.zeros(W, dtype=torch.float32, device=dev), "S": z, "D": z, "T": z, "chord": z}
    assert_label_free(anchor)
    g = grid_of(torch.as_tensor(lengths, device=dev), step)
    if not torch.equal(g.k, root.k.to(g.k.device)) or not torch.equal(g.w, root.w.to(g.w.device)):
        raise ValueError("the decoded root grid is not the grid of the window lengths")
    h = float(step) / float(rate)
    with torch.autocast(device_type=dev.type, enabled=False):
        uf = root.u[:, 1].float()
        X = root.X.float()
        cum = cumtrap_from_centre(uf, g, h)
        w = anchor.window.to(dev)
        ka, kb = anchor.k_a.to(dev, torch.float32), anchor.k_b.to(dev, torch.float32)

        def at(k):
            a, s = _grid_at(g, uf, w, k, float(step))
            u0, u1 = uf[a], uf[a + 1]
            F = cum[a] + h * s * (u0 + 0.5 * s * (u1 - u0))
            P = X[a] + s[:, None] * (X[a + 1] - X[a])
            return F, P

        Fa, Pa = at(ka)
        Fb, Pb = at(kb)
        S = Fb - Fa
        T = ((kb - ka) / float(rate)).clamp_min(1.0 / float(rate))
        D = anchor.dist_m.to(dev, torch.float32)
        e = float(rate) * (S - D) ** 2 / T / (2.0 * float(sd) ** 2)
        term = torch.zeros(W, dtype=torch.float32, device=dev).index_add(0, w, e)
        chord = torch.linalg.vector_norm(Pb - Pa, dim=-1)
    return {"term": term, "S": S, "D": D, "T": T, "chord": chord}


ZUPT_SD = 0.1                       # m/s, default scale of the inertial anchor (config zupt_anchor.sd)
ANKLE_JOINTS = (7, 8)               # SMPL joints of the left and right ankle (FOOT_POINTS), by the targets' foot 0, 1
ZUPT_PART_KEYS = ("term", "distance", "still", "S", "D", "T", "v_a", "v_b")


def _zupt_zeros(W, dev):
    z = torch.zeros(0, device=dev)
    zw = torch.zeros(W, dtype=torch.float32, device=dev)
    return {"term": zw, "distance": zw, "still": zw, "S": z, "D": z, "T": z, "v_a": z, "v_b": z}


def zupt_anchor_terms(pa, pb, va, vb, anchor, n_windows, sd=ZUPT_SD, rate=100.0):
    """The inertial anchor's parts from the decoded ankle (module docstring): pa, pb (N, 3) the ankle positions at the
    targets' t_a and t_b (window frame, y up), va, vb (N, 3) its velocities there; anchor: ZuptTargets (N strides of
    windows < n_windows). {term, distance, still (W,), and per stride S (the ZUPT stride), D (the decoded horizontal
    displacement), T (s), v_a and v_b (the decoded horizontal speeds, m/s)}; float32, autocast off."""
    dev = pa.device
    W = int(n_windows)
    with torch.autocast(device_type=dev.type, enabled=False):
        d = (pb.float() - pa.float())[:, [0, 2]]
        D = torch.sqrt((d * d).sum(-1) + 1e-12)
        S = anchor.dist_m.to(dev, torch.float32)
        T = (anchor.t_b.to(dev, torch.float32) - anchor.t_a.to(dev, torch.float32)).clamp_min(1.0 / float(rate))
        sa = (va.float()[:, [0, 2]] ** 2).sum(-1)
        sb = (vb.float()[:, [0, 2]] ** 2).sum(-1)
        dist = float(rate) * (S - D) ** 2 / T / (2.0 * float(sd) ** 2)
        still = (sa + sb) / (2.0 * float(sd) ** 2)
        w = anchor.window.to(dev)
        zw = torch.zeros(W, dtype=torch.float32, device=dev)
        distance, stillness = zw.index_add(0, w, dist), zw.index_add(0, w, still)
    return {"term": distance + stillness, "distance": distance, "still": stillness, "S": S, "D": D, "T": T,
            "v_a": torch.sqrt(sa.detach()), "v_b": torch.sqrt(sb.detach())}


def zupt_anchor_kinematics(kb, local, anchor):
    """(pa, pb, va, vb) (N, 3) each: the ankle of each target's foot at its t_a and t_b, from kb (body.kinematics
    KnotBatch of the anchored windows) at the local window ids `local` (N,), one kinematics call."""
    n = anchor.n
    dev = kb.device
    t = torch.cat([anchor.t_a, anchor.t_b]).to(dev, kb.dtype)
    kin = kb.kinematics(torch.cat([local, local]).to(dev), t)
    j = torch.as_tensor(ANKLE_JOINTS, device=dev)[anchor.foot.to(dev)]
    i = torch.arange(n, device=dev)
    return kin.p[i, j], kin.p[n + i, j], kin.v[i, j], kin.v[n + i, j]


def zupt_anchor_parts(model, z, beta, batch, anchor, sd=ZUPT_SD):
    """The inertial anchor of F2 lever 5 (module docstring): zupt_anchor_terms of the decoded ankle of the draw (z
    (W, P, 32), beta (W, 16), the latents and pooled betas of BodyDraw) at the targets of `anchor`
    (data.zupt_anchor.ZuptTargets; None or empty: term 0 everywhere). One decoder call on the anchored windows at the
    Hermite supports of the instants (float32 and float64 alike, as mfm.support_knots), the dense buffers filled as
    MFM.dense fills them, the rest skeleton of the betas with their gradient stopped, one kinematics call. Runs
    assert_label_free on the targets."""
    W = batch.n_windows
    dev = z.device
    if anchor is None or anchor.n == 0:
        return _zupt_zeros(W, dev)
    assert_label_free(anchor)
    rate = float(model.rate)
    with torch.autocast(device_type=dev.type, enabled=False):
        w = anchor.window.to(dev)
        sel, local = torch.unique(w, sorted=True, return_inverse=True)
        lengths = batch.lengths.to(dev)[sel]
        t = torch.cat([anchor.t_a, anchor.t_b]).to(dev)
        wl = torch.cat([local, local])
        L = lengths[wl]
        k = torch.cat([support(t.float(), L, rate), support(t.double(), L, rate)], 1)
        T = int(lengths.max())
        key = torch.unique((wl[:, None] * T + k).reshape(-1))
        knots = (key // T, key % T)
        out = model.decode(z[sel], beta[sel], knots, batch.tokens.n_patches.to(dev)[sel], lengths)
        dt = out.rot6d.dtype
        n_sel = int(sel.numel())
        r6 = scatter_dense(out.rot6d, out.w, out.k, n_sel, T, matrix_to_rot6d(model.decoder.mean_R).to(dt))
        tr = scatter_dense(out.transl, out.w, out.k, n_sel, T, model.decoder.transl0.to(dt))
        rest = model.skeleton.rest(beta[sel].detach().to(dt), batch.gender.to(dev)[sel])
        kb = KnotBatch(r6, tr, rest, rate, lengths)
        pa, pb, va, vb = zupt_anchor_kinematics(kb, local, anchor.to(dev))
    return zupt_anchor_terms(pa, pb, va, vb, anchor.to(dev), W, sd, rate)


def stage1_losses(model, batch, post, body, pred, phase, *, targets=None, amortized=None, progress=1.0, cfg=None):
    """Every term (W,) of section 4 (module docstring), the DIAGNOSTICS (W,), per_window (W,) the weighted
    sum and total its mean. model: MFM; batch: mfm.Batch; post: the posterior decoded from (the
    straight-through posterior on refined steps); body: BodyDraw and pred: MFM.render of it (forward());
    phase 'A' or 'B'; targets: Targets or None; amortized: the encoder's Posterior on refined steps (the
    amortization term), else None; progress: share of phase A done (KL annealing); cfg: LossConfig (its
    sigmas scale the supervision terms). Runs assert_label_free on the batch and the targets; refuses a batch
    naming a CODY source."""
    cfg = LOSS if cfg is None else cfg
    sg = {**SIGMAS, **(cfg.sigmas or {})}
    weights = term_weights(phase, progress, cfg)
    assert_label_free(batch, targets)
    if any(is_cody(s) for s in batch.sources):
        raise Refused(f"batch sources {batch.sources} name CODY, which is evaluation-only")
    W = batch.n_windows
    dev = post.mu_z.device
    lengths = batch.lengths.to(dev)
    targets = None if targets is None else targets.to(dev)
    with torch.autocast(device_type=dev.type, enabled=False):
        terms = kernel_terms(model, pred, batch)
        has = truth_mask(batch, targets, dev)
        kin = body.kin[0] if body.kin is not None else None
        w_pt, t_pt, wt_pt = point_index(batch, dev)
        zeros = _zeros(W, dev)
        sup = {k: zeros for k in ("sup_rot", "sup_pos", "sup_root", "sup_root_rate", "sup_deriv", "sup_contact",
                                  "sup_beta")}
        band, root_xz, root_y, rate_u, rate_w = zeros, zeros, zeros, zeros, zeros
        if bool(has.any()):
            truth, out = batch.truth.to(dev), body.out
            sup["sup_rot"] = sup_rot(out, truth, lengths, has, sigma=sg["rot"])
            se, band = sup_pos_parts(out, truth, batch.gender, model.skeleton, lengths, has, sigma=sg["pos"])
            sup["sup_pos"] = se + band
            root_xz, root_y = sup_root_parts(out, truth, lengths, has, sg["root_xz"], sg["root_y"])
            sup["sup_root"] = root_xz + root_y
            rate_u, rate_w = sup_root_rate_parts(out, truth, lengths, has, batch.rate, sg["root_vel"],
                                                 sg["root_yaw_rate"])
            sup["sup_root_rate"] = rate_u + rate_w
            if kin is not None and w_pt.numel():
                kin_t = targets.kin if targets is not None and targets.kin is not None else \
                    truth_kinematics(model, batch)
                sup["sup_deriv"] = sup_deriv(kin, kin_t, w_pt, wt_pt, lengths, has, sigma_v=sg["vel"],
                                             sigma_a=sg["acc"])
            if targets is not None and targets.contact is not None:
                contact = targets.contact
                if contact.shape[1] < batch.T:
                    contact = torch.cat([contact, contact[:, -1:].expand(-1, batch.T - contact.shape[1], -1)], 1)
                sup["sup_contact"] = sup_contact(out, contact, lengths, has)
            else:
                _, w_k, k_k = _truth_knots(out, has)
                c = torch.zeros(W, batch.T, 2, device=dev)
                c[w_k, k_k] = truth_contacts(model, batch, w_k, k_k).to(dev)
                sup["sup_contact"] = sup_contact(out, c, lengths, has)
            sup["sup_beta"] = sup_beta(post.mu_b, post.logvar_b, truth.beta, has)
        rest_w = float(tuple(getattr(cfg, "sup_rest", (0.0, 0.0)))[PHASES.index(phase)])
        rest_on = any(float(x) for x in tuple(getattr(cfg, "sup_rest", (0.0, 0.0))))
        rest_term = zeros
        if rest_on and bool(has.any()):
            rest_term = sup_rest(post.mu_b, batch.truth.beta, batch.gender, model.skeleton, has, sigma=sg["pos"])
        terms.update(sup)
        terms["kl_z"] = model.prior.kl(post.mu_z, post.logvar_z, body.z, batch.cond(body.beta),
                                       free_bits=cfg.free_bits, mask=post.mask).float()
        terms["kl_beta"] = model.shape_prior.kl(post.mu_b, post.logvar_b, batch.gender.to(dev)).float()
        terms["nuis_prior"] = nuis_prior(model, post.nuis, batch)
        terms["nuis_aux"] = nuis_aux(model, post.nuis, batch, targets, pred)
        terms["consistency"] = consistency(post.mu_z, post.logvar_z, post.mask,
                                           None if targets is None else targets.pair)
        pl = plausibility(body.out, body.dense, kin, w_pt, t_pt, wt_pt, lengths, batch.rate,
                          limits=plausibility_limits(cfg, body.out.rot6d.dtype, body.out.rot6d.device))
        terms["plausibility"] = pl["plausibility"]
        on = None if targets is None else targets.plausibility_on
        if on is not None:
            if tuple(on.shape) != (W,):
                raise ValueError(f"Targets.plausibility_on has shape {tuple(on.shape)}, the batch {W} windows")
            terms["plausibility"] = torch.where(on.to(dev, torch.bool), pl["plausibility"],
                                                torch.zeros_like(pl["plausibility"]))
        terms["subject"] = subject(post.mu_b, post.logvar_b, batch.person)
        syn = is_synthetic(batch, dev)
        terms["hsic"], active = hsic(post.mu_z, post.mask, batch.source_id, batch.task, syn, cfg.hsic_synthetic,
                                     stratum=None if targets is None else targets.hsic_stratum)
        terms["amortization"] = zeros if amortized is None else amortization(amortized, post.mu_z)
        diag = {"sup_pos_band": band, "sup_root_xz": root_xz, "sup_root_y": root_y, "sup_root_rate_u": rate_u,
                "sup_root_rate_yaw": rate_w, "plaus_limits": pl["plaus_limits"],
                "plaus_penetration": pl["plaus_penetration"], "plaus_skate": pl["plaus_skate"],
                "bone_spread": zeros if kin is None or not w_pt.numel() else bone_spread(kin.p, w_pt, W),
                "hsic_active": active.float()}
        per_window = zeros
        for t in TERMS:
            if weights[t] == 0.0:
                continue
            x = terms[t].float()
            if phase == "A" and t in SYNTHETIC_ONLY_IN_A:
                x = torch.where(syn, x, torch.zeros_like(x))
            per_window = per_window + weights[t] * x
        if rest_on:
            diag["sup_rest"] = rest_term.float()
            if rest_w:
                per_window = per_window + rest_w * rest_term.float()
    return {**{t: terms[t].float() for t in TERMS}, **diag, "per_window": per_window, "total": per_window.mean()}
