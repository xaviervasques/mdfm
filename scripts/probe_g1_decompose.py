"""G1a gap breakdown on the dev windows (weekend task T05): where the six-IMU error of a checkpoint comes from.

Every row is scored on the windows of the default G1 dev root (scripts/build_g1_inputs.py, mdfm.gates.g1.G1Set;
3,358 windows of 23 persons), with the harness's own pieces (g1.mpjpe_yaw and g1.yaw_fit for G1a,
g1.procrustes_errors, g1.heading_points, g1.predictive_scores) and the model through mdfm.eval.infer (g1_windows,
the blocks of G1Predictor.block, seed_of, the forward kinematics of infer._pose). No lockbox, calib-only or CODY
input is opened for scoring: the dev archives are scored, the calib archives serve the conformal factor of the
acc-only wrist row only, and the lockbox mix is an estimate from the archive counts of reports/amass_split.

Rows (G1a values: the G1 statistic, the mean over persons of the knot-weighted person means, in mm):
- g1a: the harness reproduced in this process: six IMUs, encode, 20 refinement steps of 256 points per stream,
  shape pooled per take, posterior-mean body decoded at every knot, scored by g1.mpjpe_yaw. Its blocks and seeds
  are those of scripts/eval_g1_dev.py (G1Predictor.block with --block-windows, seed_of(seed, split, first take,
  'six')), so its value equals the J1 g1.json of the same checkpoint up to GPU nondeterminism; the difference is
  written beside it (j1: overall and on the windows both scored, with whether J1 scored the same checkpoint and
  which of its inference settings differ).
- heading: per-knot heading-frame MPJPE: each body pelvis-relative in its own pelvis-heading frame at the same
  knot (g1.heading_points with each body's root yaw), the 22 joints (the pelvis contributes 0); no window yaw fit.
- pa, pa15: six-IMU PA-MPJPE (per-knot similarity Procrustes) on the 22 joints and on the 15 canonical joints.
- Substitutions in the same 20-step pass, with the pooled shape of the pass. Global reading (the primary rows):
  true_root_orient, sub_legs, sub_spine_head, sub_arms set the global orientation of the root segment (the
  pelvis) or of the group's segments (GROUPS) to the truth's, turned into the prediction's frame by the window's
  G1a yaw, every other segment keeping its decoded global orientation (substitute_global): the error carried by
  those segments' own orientation. Local reading (the _local rows): the true local rotations of those joints with
  the decoded local rotations of the others (for the root: the whole decoded body turned onto the true root). The
  six IMUs measure global segment orientations, so decoded local rotations can compensate a wrong root, and the
  local reading can then raise the error. true_betas: the decoder run again on the refined latents with the take's
  true betas, and the skeleton of the true betas; true_betas_skeleton: the decoded rotations kept, only the
  skeleton of the true betas. Truth rotations and betas come from the raw take through the G1 truth path
  (g1.take_body, g1.window_frame on the cached dev npz of mdfm.data.amass), checked against the stored truth
  joints of the G1 set (truth_check: largest deviation, refused above TRUTH_TOL_M).
- refine0: the amortized posterior (no refinement step) of the same encoder pass, full set.
- refine60 and abs, partial: on the first --partial-takes takes of each dev archive (default 2, as eval_g1_dev
  --limit-takes 2): 60 refinement steps from the encoder; analysis by synthesis, 200 refinement steps from the
  prior with the encoder bypassed (z the mean path of the motion prior, patch by patch with no noise; beta the
  shape prior mean of the take's gender; the nuisance heads at a zero encoder state). refine20_partial, the
  reference of both, is the 20-step value of the same windows in the same partial blocks and seeds (those of
  eval_g1_dev --limit-takes), so that the pairing isolates the step count or the start: in the blocks of the
  full pass the refinement draws differ, which moves a take's 20-step G1a by up to a few mm.
  refine20_full_blocks is that full-pass value of the same windows, written for the comparison.
- best_of_16: 16 draws of the refined posterior (z and the pooled beta), decoded and scored; per window the
  lowest G1a (an oracle: ambiguity against bias); single_draw is the mean G1a of one draw.
- floor: the truth re-rendered and re-scored: the true local rotations and betas through the model's skeleton
  and forward kinematics (float32, infer._pose), with the root yaw taken through the decoder's root path (the
  0.1 s grid of interval-mean truth yaw rates, train.losses.truth_root_grid, the cumulative trapezoid and cubic
  Hermite of model.decoder), scored against the stored truth; floor_kinematic is the same without the yaw path.
- acc_only_wrist: the single-wrist line with the wrist gyro masked (acc only): 16 latent samples x 4 decoder
  draws, its own conformal factor on the calib windows (every calib knot a radius, as infer.fit_conformal),
  coverage, band width and energy score on dev; beside it the acc and gyro wrist of the J1 run (windows.csv of
  reports/gates/g1/dev_<name>/), paired per window.
- by_task and by_archive: every row per task (mdfm.data.windows vocabulary, the window table's task id) and per
  archive, knot-weighted over the windows (the 'mm' values), with the person-weighted G1 statistic beside it.
- by_group: per joint group and for the root, the G1a after the substitution (global and local readings), its
  gain over g1a, and the group's joints' share of g1a (their per-joint errors over 22); per_joint_mm.
- lockbox_mix_mm, an estimate: no lockbox array is opened. Weights: each lockbox archive's share of the 53 lockbox
  persons from the per-person archive lists of reports/amass_split/persons.csv, a person split equally over its
  archives (the G1 statistic is a mean over persons). Values: the G1 statistic of each dev group (its
  <row>_person_mm, the scale of g1a_mm and of the kill line); the same estimate on the knot-weighted group values
  is written beside it (knot_weighted). by task: each lockbox archive carries the task of LOCKBOX_ARCHIVE_TASK (a
  fixed reading of the archives' published content, an assumption, not a measurement) and the dev by-task value
  of that task (all windows when the task has none; the size of the dev group is written beside it); by archive:
  each lockbox archive carries the dev archive of the nearest mean take duration (hours per motion of
  archives.csv, log scale; the overall dev value when the archive has no hours). lockbox_mix_mm is the by-task
  estimate; both are written.
- per_take: one record per dev take (take_id, archive, person_id, task, windows, knots) with every row in mm,
  knot-weighted over its windows (refine60, abs, refine20_partial and refine20_full_blocks on the partial takes
  only), for a take-level bootstrap (scripts/weekend_branch.py pairs them by take id); refine20_mm is g1a_mm,
  and refine20_partial_mm on a partial take (the reference that weekend_branch pairs with refine60 and abs).
- oracle6 rows, opt-in (--oracle6, task ORI6; absent from the default run and from its outputs): the decoded body
  with the true global orientation of the six instrumented segments only, every other segment as decoded
  (oracle6_globals). The six segments are those the G1a sites are mounted on (ORACLE6_SEGMENTS, the segment joints
  of mdfm.factory.imu_render.SITE_RULES for g1.SITES): head, pelvis (lower_back), the forearms (wrist_L and wrist_R
  ride on left_elbow and right_elbow) and the shanks (left_knee, right_knee). The truth is turned into the
  prediction's frame by the window's G1a yaw, as in the global reading. oracle6_exact: exact truth (variant a).
  oracle6_noisy: the truth turned about +y per sensor and knot by the heading error model of
  docs/proposals/G1_ORIENTATION_INPUT_DRAFT.md (section 3) at its level set L1 (variant b): offset, slow wander and
  fast noise per sensor (oracle6_unit_draws, seeded per window by ORACLE6_SEED and the window index), re-referenced
  to the lower-back sensor at the window centre (oracle6_psi); no tilt noise. Levels from
  reports/orientation_sources (oracle6_levels). Descriptive: oracle6_noisy_L2 (level set L2), oracle6_tilt (true
  tilt, decoded heading) and oracle6_heading (decoded tilt, true heading), the heading split from the tilt by the
  closest turn about +y (closest_yaw). decompose.json 'oracle6' holds the rows, their by-task and by-archive values
  and the reading of reports/bench/g1_diag/oracle6/prediction.txt (oracle6_reading); decompose.md a section.
  --oracle6-prediction checks that file against its prediction.sha256 before any decode.

Intervals: person-clustered percentile bootstrap of the G1 rule (2,000 resamples, seed 20260929, 95 percent), and
the same on the paired per-window differences from g1a ('delta').

Row functions for other scripts (scripts/eval_g1_dev.py may import them): g1a_parts, heading_errors,
window_metrics, substitute, truth_root_yaw_path, TruthBodies, posterior_pass, prior_start, main_pass,
mean_pass (refine0 or any step count, encoder or prior start), acc_only_wrist, aggregate_rows, lockbox_mix.

Outputs in --out (default reports/bench/g1_decompose_<checkpoint name>/): decompose.json (top-level floats
g1a_mm, true_root_orient_mm, true_betas_mm, refine0_mm, refine20_mm, refine60_mm, abs_mm, best_of_16_mm, floor_mm,
heading_mm, pa_mm, lockbox_mix_mm; by_task, by_archive, by_group, per_take, rows with intervals and labels,
acc_only_wrist, lockbox_mix, j1, sets, checkpoint, hashes, timing), decompose.md (the attribution table written
here), windows.csv (every scored dev window), _provenance.json.

Run (mdfm environment; evaluation contract of scripts/queue_job.py):
  <conda>/envs/mdfm/python.exe scripts/probe_g1_decompose.py --checkpoint <run>/ckpt_B_<step>.pt
Attainability of the oracle6 reading (CPU, no model, no oracle6 value: the rows of an existing decompose folder and
the stored dev truth joints; oracle6_attain), written to <folder>/attain.json:
  <conda>/envs/mdfm/python.exe scripts/probe_g1_decompose.py --oracle6-attain <decompose folder>
      --out <folder>
"""
import argparse
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from mdfm.body.rotations import matrix_to_rot6d, rot6d_to_matrix, rot_y, yaw_angle  # noqa: E402
from mdfm.body.joints import BODY_JOINTS, N_BODY, PARENTS  # noqa: E402
from mdfm.body.sex import gender_code  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.eval import infer as I  # noqa: E402
from mdfm.factory import imu_render  # noqa: E402
from mdfm.gates import common, g1  # noqa: E402
from mdfm.model import mfm as M  # noqa: E402
from mdfm.model.decoder import GRID_KNOTS, all_knots, cumtrap_from_centre, grid_of, hermite_on_grid  # noqa: E402
from mdfm.model.posterior import sample, subject_pool  # noqa: E402
from mdfm.model.refine import refine  # noqa: E402

SPLIT = "dev"
SEED = 20260929                 # the --seed default of scripts/eval_g1_dev.py
RATE = g1.RATE
MM = 1000.0
TRUTH_TOL_M = 1e-3              # largest deviation of the rebuilt truth joints from the stored ones
CONFIG = "six"
PARTIAL_TAKES = 2
REFINE_LONG = 60
ABS_STEPS = 200
BEST_OF = 16
GAP_PASS_MM = g1.PARAMS["g1a_pass_mm"]


def _group(*words):
    return tuple(j for j, name in enumerate(BODY_JOINTS[:N_BODY]) if any(w in name for w in words))


# joint groups of the substitution rows (SMPL body joints; the pelvis is the root orientation row)
GROUPS = {"legs": _group("hip", "knee", "ankle", "foot"),
          "spine_head": _group("spine", "neck", "head"),
          "arms": _group("collar", "shoulder", "elbow", "wrist")}

# lockbox archive -> task of the vocabulary, a fixed reading of each archive's published content (an
# assumption used by the lockbox-mix estimate only; mixed-content archives read 'unknown', the task of dev takes
# whose names match no rule)
LOCKBOX_ARCHIVE_TASK = {"ACCAD": "unknown", "DFaust": "sport_dance", "GRAB": "adl", "MoSh": "unknown",
                        "PosePrior": "unknown", "SOMA": "unknown", "SSM": "unknown", "TCDHands": "adl",
                        "TotalCapture": "unknown", "Transitions": "transfer"}
ARCHIVES_CSV = os.path.join(TREE, "reports", "amass_split", "archives.csv")
PERSONS_CSV = os.path.join(TREE, "reports", "amass_split", "persons.csv")

# per-window columns (metres) of the full pass; each maps to a row name and its top-level key when it has one
FULL_ROWS = (
    ("g1a", "g1a_m", "G1a, 20 refinement steps, posterior mean (the harness)"),
    ("heading", "heading_m", "per-knot heading-frame MPJPE, 22 joints, no window yaw fit"),
    ("pa", "pa_m", "PA-MPJPE of the six-IMU body, 22 joints"),
    ("pa15", "pa15_m", "PA-MPJPE of the six-IMU body, 15 canonical joints"),
    ("refine0", "refine0_m", "G1a, amortized posterior (0 refinement steps)"),
    ("true_root_orient", "true_root_m", "true global orientation of the root segment, other segments as decoded"),
    ("true_root_orient_local", "true_root_local_m", "true root rotation, decoded local rotations kept"),
    ("true_betas", "true_betas_m", "true betas (decoder and skeleton)"),
    ("true_betas_skeleton", "true_betas_skel_m", "skeleton of the true betas, decoded rotations kept"),
    ("sub_legs", "sub_legs_m", "true global orientations of the leg segments"),
    ("sub_spine_head", "sub_spine_head_m", "true global orientations of the spine and head segments"),
    ("sub_arms", "sub_arms_m", "true global orientations of the arm segments"),
    ("sub_legs_local", "sub_legs_local_m", "true local rotations of the leg joints"),
    ("sub_spine_head_local", "sub_spine_head_local_m", "true local rotations of the spine and head joints"),
    ("sub_arms_local", "sub_arms_local_m", "true local rotations of the arm joints"),
    ("best_of_16", "best_m", "G1a of the best of 16 posterior draws per window (oracle)"),
    ("single_draw", "draw_mean_m", "G1a of one posterior draw (mean over the 16)"),
    ("floor", "floor_m", "truth re-rendered (model skeleton, decoder root yaw path) and re-scored"),
    ("floor_kinematic", "floor_kin_m", "truth through the model skeleton and kinematics only"),
)
PARTIAL_ROWS = (
    ("refine20_partial", "refine20_m", "G1a, 20 refinement steps, on the partial windows in their own blocks"),
    ("refine20_full_blocks", "g1a_m", "G1a, 20 refinement steps, the same windows in the blocks of the full pass"),
    ("refine60", "refine60_m", "G1a, 60 refinement steps from the encoder"),
    ("abs", "abs_m", "G1a, analysis by synthesis: 200 steps from the prior, encoder bypassed"),
)
PARTIAL_REF = "refine20_m"      # the reference of the partial rows (g1a_m when a partial table lacks it)
TOP_KEYS = {"g1a": "g1a_mm", "true_root_orient": "true_root_orient_mm", "true_betas": "true_betas_mm",
            "refine0": "refine0_mm", "best_of_16": "best_of_16_mm", "floor": "floor_mm", "heading": "heading_mm",
            "pa": "pa_mm", "refine60": "refine60_mm", "abs": "abs_mm", "refine20_partial": "refine20_partial_mm"}
JOINT_COLS = tuple(f"j{j:02d}_m" for j in range(N_BODY))

# six-segment orientation oracle (task ORI6, opt-in; module docstring)
ORACLE6_SEED = 20261006         # heading error draws per window: numpy default_rng((ORACLE6_SEED, window index))
ORACLE6_REF = "lower_back"      # the reference sensor of the draft's re-referencing (section 3)
ORACLE6_SMOOTH_S = 1.0          # the moving average of the slow part (scripts/probe_orientation_sources.py)
ORACLE6_GROUP = "control"       # the fit set of the draft's levels (non-lockbox weargait_pd controls)
ORACLE6_LEVELS_DIR = os.path.join(TREE, "reports", "orientation_sources")
ORACLE6_LEVEL_KEYS = ("offset", "slow", "fast")
# the segment joint of each G1a site (the joint whose global rotation carries the sensor), in the order of g1.SITES
ORACLE6_SEGMENTS = tuple(int(imu_render.SITE_RULES[s][0]) for s in g1.SITES)
# (row, column, mode of oracle6_globals, level set, what)
ORACLE6_ROWS = (
    ("oracle6_exact", "o6_exact_m", "exact", None,
     "true global orientation of the six instrumented segments only, every other segment as decoded"),
    ("oracle6_noisy", "o6_noisy_m", "noisy", "L1",
     "the same with the draft's heading error model at level set L1 (offset, slow wander, fast noise; seeded)"),
    ("oracle6_noisy_L2", "o6_noisy_L2_m", "noisy", "L2",
     "the same at level set L2 (L1 times the walking-to-standing ratio), descriptive"),
    ("oracle6_tilt", "o6_tilt_m", "tilt", None,
     "true tilt of the six segments with their decoded heading (closest yaw), descriptive"),
    ("oracle6_heading", "o6_heading_m", "heading", None,
     "decoded tilt of the six segments with their true heading (closest yaw), descriptive"),
)


# ---------------------------------------------------------------- scoring rows (numpy)
def g1a_parts(pred, truth):
    """(per-knot (n,), per-joint (22,)) errors in m of the G1a scorer: both bodies pelvis-relative, one yaw per
    window fitted over every joint and knot (g1.yaw_fit), as g1.mpjpe_yaw (whose per-knot values equal the
    first)."""
    t = np.asarray(truth, dtype=np.float64)
    p = np.broadcast_to(np.asarray(pred, dtype=np.float64), t.shape)
    t = t - t[:, g1.PELVIS:g1.PELVIS + 1]
    p = p - p[:, g1.PELVIS:g1.PELVIS + 1]
    th = g1.yaw_fit(t, p)
    e = np.linalg.norm(g1.turn_y(p, th) - t, axis=-1)
    return e.mean(-1), e.mean(0)


def heading_errors(pred, truth, pred_yaw, truth_yaw):
    """Per-knot heading-frame MPJPE (n,) m: each body pelvis-relative in its own pelvis-heading frame at the same
    knot (g1.heading_points with its root yaw), mean over the 22 joints."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    hp = g1.heading_points(p, p[:, g1.PELVIS:g1.PELVIS + 1], np.asarray(pred_yaw, dtype=np.float64)[:, None])
    ht = g1.heading_points(t, t[:, g1.PELVIS:g1.PELVIS + 1], np.asarray(truth_yaw, dtype=np.float64)[:, None])
    return np.linalg.norm(hp - ht, axis=-1).mean(-1)


def window_metrics(pred, truth, truth_yaw, pred_yaw=None):
    """Window means (m) of one predicted body (n, 22, 3) against the truth: g1a_m, pa_m, pa15_m, heading_m (when
    pred_yaw (n,) is given) and the per-joint G1a errors j00_m to j21_m."""
    e_knot, e_joint = g1a_parts(pred, truth)
    P, J = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    out = {"g1a_m": float(e_knot.mean()), "pa_m": float(g1.procrustes_errors(P, J).mean()),
           "pa15_m": float(g1.procrustes_errors(P[:, g1.CANON15], J[:, g1.CANON15]).mean())}
    if pred_yaw is not None:
        out["heading_m"] = float(heading_errors(P, J, pred_yaw, truth_yaw).mean())
    out.update({c: float(v) for c, v in zip(JOINT_COLS, e_joint)})
    return out


def g1a_of(pred, truth):
    """The window's G1a (m), g1.mpjpe_yaw averaged over its knots."""
    return float(g1a_parts(pred, truth)[0].mean())


def substitute(R_pred, R_true, joints):
    """Rotations (..., 22, 3, 3): R_pred with the rows `joints` taken from R_true (a torch copy)."""
    out = R_pred.clone()
    idx = list(joints)
    out[..., idx, :, :] = R_true[..., idx, :, :].to(out.dtype)
    return out


def local_from_global(Rg):
    """Local rotations (..., 22, 3, 3) of global ones: the root's own, R_parent^T R_joint for the others."""
    out = Rg.clone()
    out[..., 1:, :, :] = Rg[..., list(PARENTS[1:N_BODY]), :, :].transpose(-1, -2) @ Rg[..., 1:, :, :]
    return out


def substitute_global(Rg_pred, Rg_true, joints, theta):
    """Local rotations (..., 22, 3, 3) of a body whose segments `joints` take the true global orientation, turned
    into the prediction's frame by rot_y(-theta) (theta (...,) the window's G1a yaw, which turns the prediction
    onto the truth), every other segment keeping its decoded global orientation."""
    Ry = rot_y(-theta.to(Rg_true.dtype))[..., None, :, :]
    return local_from_global(substitute(Rg_pred, Ry @ Rg_true, joints))


def truth_root_yaw_path(R_root):
    """The root yaw (n,) rad of a window's true root rotations R_root (n, 3, 3) (window frame, float64) as the
    decoder carries it: the interval-mean yaw rates of the truth on the 0.1 s root grid
    (train.losses.truth_root_grid), the cumulative trapezoid from the centre knot n // 2 and the cubic Hermite at
    every knot (model.decoder), plus the true yaw of the centre knot."""
    from mdfm.model.mfm import KnotBody
    from mdfm.train.losses import truth_root_grid
    R = torch.as_tensor(R_root, dtype=torch.float64)
    n = R.shape[0]
    r6 = matrix_to_rot6d(R)[None, :, None, :].expand(1, n, N_BODY, 6).contiguous()
    L = torch.tensor([n])
    body = KnotBody(r6, torch.zeros(1, n, 3, dtype=torch.float64), None, torch.zeros(1, 16, dtype=torch.float64),
                    torch.zeros(1, dtype=torch.int64), L)
    tg = truth_root_grid(body, RATE, L)
    g = grid_of(L)
    h = GRID_KNOTS / RATE
    psi_g = cumtrap_from_centre(tg.omega, g, h)
    w, k = all_knots(L)
    psi = hermite_on_grid(psi_g, tg.omega, g, w, k, h)
    return (psi + yaw_angle(R[n // 2])).numpy()


def with_root_yaw(R_root, psi):
    """Root rotations (n, 3, 3) with their own yaw replaced by psi (n,): rot_y(psi) rot_y(-yaw(R)) R (the tilt
    kept)."""
    R = torch.as_tensor(R_root, dtype=torch.float64)
    p = torch.as_tensor(psi, dtype=torch.float64)
    return rot_y(p) @ rot_y(-yaw_angle(R)) @ R


# ---------------------------------------------------------------- six-segment orientation oracle (task ORI6)
def closest_yaw(A, B):
    """phi (...) rad such that rot_y(phi) A is the turn of A about +y closest to B (Frobenius norm), A and B
    (..., 3, 3): atan2(M[2, 0] - M[0, 2], M[0, 0] + M[2, 2]) with M = A B^T. rot_y(phi) A keeps A's tilt (the
    direction of +y in its frame) and takes the heading nearest to B's; it is B when B is a turn of A about +y."""
    M = A @ B.transpose(-1, -2)
    return torch.atan2(M[..., 2, 0] - M[..., 0, 2], M[..., 0, 0] + M[..., 2, 2])


def oracle6_globals(Rg_pred, Rg_true, theta, mode="exact", psi=None, segments=ORACLE6_SEGMENTS):
    """Global rotations (..., 22, 3, 3): Rg_pred with the rows `segments` (the six instrumented segments, in the
    order of g1.SITES) replaced, every other row Rg_pred's. T is the true global orientation turned into the
    prediction's frame by rot_y(-theta) (theta (...,) the window's G1a yaw, as substitute_global), P the decoded
    one. mode 'exact': T; 'noisy': rot_y(psi) T, psi (..., 6) rad the heading errors of the six sensors
    (oracle6_psi); 'tilt': T turned about +y onto P (the true tilt with the decoded heading); 'heading': P turned
    about +y onto T (the decoded tilt with the true heading)."""
    idx = list(segments)
    Ry = rot_y(-theta.to(Rg_true.dtype))[..., None, :, :]
    T = (Ry @ Rg_true[..., idx, :, :]).to(Rg_pred.dtype)
    P = Rg_pred[..., idx, :, :]
    if mode == "exact":
        S = T
    elif mode == "noisy":
        if psi is None:
            raise ValueError("mode 'noisy' needs the heading errors psi")
        S = rot_y(torch.as_tensor(psi).to(device=T.device, dtype=T.dtype)) @ T
    elif mode == "tilt":
        S = rot_y(closest_yaw(T, P)) @ T
    elif mode == "heading":
        S = rot_y(-closest_yaw(T, P)) @ P
    else:
        raise ValueError(f"oracle6 mode {mode!r}, expected exact, noisy, tilt or heading")
    out = Rg_pred.clone()
    out[..., idx, :, :] = S
    return out


def _moving_average(x, m):
    """Centred moving average over m samples along axis 0 (at most len(x)), normalized at the edges: the
    moving_average of scripts/probe_orientation_sources.py applied to every column."""
    x = np.asarray(x, dtype=np.float64)
    m = max(1, min(int(m), x.shape[0]))
    k = np.ones(m)
    norm = np.convolve(np.ones(x.shape[0]), k, mode="same")
    cols = x.reshape(x.shape[0], -1)
    out = np.stack([np.convolve(cols[:, i], k, mode="same") / norm for i in range(cols.shape[1])], 1)
    return out.reshape(x.shape)


def oracle6_unit_draws(window, n, rate=RATE, seed=ORACLE6_SEED, sensors=len(g1.SITES)):
    """Standardized heading error components of one window, a function of (seed, window index, n) only (so of no
    block or batch), drawn from numpy default_rng((seed, window)) in this order: offset (sensors,) N(0, 1);
    the steps of a random walk (n, sensors) N(0, 1); fast (n, sensors) N(0, 1). wander (n, sensors): the walk
    smoothed by the centred ORACLE6_SMOOTH_S moving average, demeaned and scaled so that the SD over the window of
    its own moving average is 1 (0 for a window too short to wander)."""
    rng = np.random.default_rng([int(seed), int(window)])
    n = int(n)
    off = rng.standard_normal(int(sensors))
    walk = np.cumsum(rng.standard_normal((n, int(sensors))), axis=0)
    fast = rng.standard_normal((n, int(sensors)))
    m = int(round(ORACLE6_SMOOTH_S * float(rate)))
    w = _moving_average(walk, m)
    w = w - w.mean(0)
    sd = _moving_average(w, m).std(0)
    w = np.where(sd > 1e-12, w / np.where(sd > 1e-12, sd, 1.0), 0.0)
    return {"offset": off, "wander": w, "fast": fast}


def oracle6_psi(units, levels, sites=g1.SITES, ref=ORACLE6_REF):
    """Heading errors (n, 6) rad of the six sensors of one window: per site, in degrees, offset x level 'offset' +
    wander x level 'slow' + fast x level 'fast' (units of oracle6_unit_draws, levels {site: {offset, slow, fast}}
    deg), then re-referenced as the draft's stream (section 3): minus the reference sensor's error at the window
    centre knot n // 2, so that the reference reads heading 0 there."""
    lv = {k: np.array([float(levels[s][k]) for s in sites], dtype=np.float64) for k in ORACLE6_LEVEL_KEYS}
    deg = units["offset"][None] * lv["offset"] + units["wander"] * lv["slow"] + units["fast"] * lv["fast"]
    deg = deg - deg[deg.shape[0] // 2, list(sites).index(ref)]
    return np.deg2rad(deg)


def oracle6_levels(folder=ORACLE6_LEVELS_DIR, group=ORACLE6_GROUP, sites=g1.SITES):
    """The draft's per-site heading error levels in degrees (docs/proposals/G1_ORIENTATION_INPUT_DRAFT.md,
    sections 2.1 and 3) for the sites of g1.SITES, read from folder (reports/orientation_sources). L1: the
    Balance per-site fit of the group (cross_sites.csv; head, lower back and shanks from the four-site fit
    'no_wrists', the wrists from the six-site fit 'six_sites'), offset = between_abs_change_deg, slow =
    slow_sd_deg, fast = fast_rms_deg. L2: L1 times the walking-to-standing ratio of the co-located psi RMS medians
    of the group's SelfPace trials (heading_summary.csv). Returns {L1, L2: {site: {offset, slow, fast}}, ratio_L2,
    psi_rms_walk_deg, psi_rms_standing_deg, group, files: {name: sha256}}."""
    import csv
    stat = {"offset": "between_abs_change_deg", "slow": "slow_sd_deg", "fast": "fast_rms_deg"}
    p_sites = os.path.join(folder, "cross_sites.csv")
    p_head = os.path.join(folder, "heading_summary.csv")
    fit = {}
    with open(p_sites, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == group and r["trial"] == "Balance":
                fit[(r["statistic"], r["variant"])] = r
    L1 = {}
    for s in sites:
        variant = "six_sites" if s.startswith("wrist") else "no_wrists"
        L1[s] = {}
        for k, col in stat.items():
            r = fit.get((col, variant))
            v = float(r[s]) if r is not None and str(r.get(s, "")).strip() else math.nan
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{p_sites}: no {col} level for {s} ({group}, Balance, {variant})")
            L1[s][k] = v
    walk = stand = None
    with open(p_head, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["group"] == group and r["trial"] == "SelfPace":
                walk = float(r["psi_rms_walk_deg_p10_p50_p90"].split("|")[1])
                stand = float(r["psi_rms_standing_deg_p10_p50_p90"].split("|")[1])
    if walk is None or not (stand and stand > 0):
        raise ValueError(f"{p_head}: no SelfPace walking and standing psi RMS medians for {group}")
    ratio = walk / stand
    L2 = {s: {k: v * ratio for k, v in d.items()} for s, d in L1.items()}
    return {"L1": L1, "L2": L2, "ratio_L2": ratio, "psi_rms_walk_deg": walk, "psi_rms_standing_deg": stand,
            "group": group, "files": {common.rel(p): common.sha256_file(p) for p in (p_sites, p_head)}}


def oracle6_reading(exact_mm, noisy_mm, kill_mm=g1.PARAMS["g1a_kill_mm"]):
    """The reading of reports/bench/g1_diag/oracle6/prediction.txt, in its order: CROSSES when the noisy row
    (variant b) reads at most the kill line; else NEAR when the exact row (variant a) reads under it; else CANNOT.
    None when a value is missing."""
    if exact_mm is None or noisy_mm is None:
        return None
    if noisy_mm <= kill_mm:
        return "CROSSES"
    if exact_mm < kill_mm:
        return "NEAR"
    return "CANNOT"


def oracle6_prediction_check(path):
    """{path, sha256, recorded_sha256, hashed_at, matches} of a prediction file against the first token of the
    prediction.sha256 beside it (its line: '<sha256>  <file name>  hashed <time>'). ValueError when the hash file
    is missing, names another file or holds another hash."""
    hp = os.path.join(os.path.dirname(os.path.abspath(path)), "prediction.sha256")
    if not os.path.exists(path) or not os.path.exists(hp):
        raise ValueError(f"oracle6 prediction: {path} or its prediction.sha256 not found")
    with open(hp, encoding="utf-8") as f:
        tokens = f.read().split()
    sha = common.sha256_file(path)
    out = {"path": common.rel(path), "sha256": sha, "recorded_sha256": tokens[0] if tokens else None,
           "hashed_at": tokens[-1] if len(tokens) > 3 and tokens[-2] == "hashed" else None,
           "matches": bool(tokens and tokens[0] == sha and len(tokens) > 1
                           and tokens[1] == os.path.basename(path))}
    if not out["matches"]:
        raise ValueError(f"oracle6 prediction: {path} does not match {hp} ({sha} against {tokens[:2]})")
    return out


# ---------------------------------------------------------------- truth bodies
class TruthBodies:
    """truth(i) of a dev or calib G1Set extended with the local rotations: {R_local (n, 22, 3, 3) float64 with
    row 0 the root orientation in the window frame, betas (16,) float64, gender (SkeletonBank code)}, rebuilt
    from the cached raw take (mdfm.data.amass.load_many, g1.take_body, g1.window_frame) and checked against the
    stored truth joints (max_dev_m; ValueError above tol). One take is kept in memory at a time."""

    def __init__(self, gset, tol=TRUTH_TOL_M):
        from mdfm.data import amass_index
        if gset.split not in ("dev", "calib"):
            raise Refused(f"the decompose probe reads dev and calib takes only, not {gset.split!r}")
        self.gset, self.tol = gset, float(tol)
        idx = amass_index.takes(gset.split)
        self.index = idx.set_index("take_id", drop=False)
        self.bodies, self.cache = {}, {}
        self.max_dev_m, self.n_windows, self.n_takes = 0.0, 0, 0

    def _take(self, archive, take):
        from mdfm.body.smplh import SMPLHBody
        from mdfm.data import amass
        row = self.index.loc[take]
        seq = amass.load_many(archive, [take])[take]
        g = str(seq["gender"]).strip().lower()
        if g not in self.bodies:
            self.bodies[g] = SMPLHBody(g, dtype=torch.float64)
        tk = g1.take_body(seq, row, self.bodies[g])
        if tk is None:
            raise ValueError(f"{take}: shorter than a G1 window, yet it has windows in the set")
        self.n_takes += 1
        return tk

    def __call__(self, i):
        r = self.gset.windows.iloc[int(i)]
        take = str(r["take_id"])
        if take not in self.cache:
            self.cache = {take: self._take(str(r["archive"]), take)}
        tk = self.cache[take]
        s, n = int(r["start"]), int(r["n"])
        joints, yaw, R_yaw, origin = g1.window_frame(tk["pos"], tk["R"], s, n)
        R_loc = rot6d_to_matrix(tk["r6"][s:s + n]).clone()
        R_loc[:, 0] = R_yaw @ tk["R"][s:s + n, 0]
        J, _ = self.gset.truth(i)
        dev = float(np.abs(joints.numpy() - J).max())
        if not dev <= self.tol:
            raise ValueError(f"window {i} ({take} k {int(r['k'])}): the rebuilt truth deviates by {dev:.3g} m from "
                             f"the stored truth (tolerance {self.tol:g} m)")
        self.max_dev_m = max(self.max_dev_m, dev)
        self.n_windows += 1
        return {"R_local": R_loc.numpy(), "betas": tk["betas"].numpy()[:16].astype(np.float64),
                "gender": gender_code(r["gender"])}


# ---------------------------------------------------------------- model passes
def prior_mean_path(prior, cond, mask):
    """The mean path (W, P, d_z) of the autoregressive motion prior: patch by patch, each patch the prior mean
    given the patches already set (no noise), padding patches 0."""
    W, P = mask.shape
    z = torch.zeros(W, P, prior.d_z, device=mask.device, dtype=prior.out.weight.dtype)
    for p in range(P):
        mu, _ = prior.params(z, cond, mask, dropout=False)
        z[:, p] = mu[:, p] * mask[:, p, None].to(mu.dtype)
    return z


def prior_start(model, batch):
    """The encoder-bypassed start of analysis by synthesis (module docstring): z the motion prior's mean path,
    beta the shape prior mean of each window's gender (log variances its diagonal), the nuisance of the heads at
    a zero encoder state, the patch mask of the tokens."""
    dev = model.device
    mask = batch.tokens.lat_idx.to(dev) >= 0
    W = mask.shape[0]
    gender = batch.gender.to(dev)
    sp = model.shape_prior
    mu_b = sp.mean[gender].float()
    lv_b = torch.log(torch.diagonal(sp.cov[gender], dim1=-2, dim2=-1)).float()
    mu_z = prior_mean_path(model.prior, batch.cond(mu_b), mask).float()
    d = model.cfg.d
    zero = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
    nuis = M.Nuisance(model.heads.imu(zero(W, M.N_SITES, d), batch.win["imu_family"].to(dev)),
                      model.heads.camera(zero(W, d)), model.heads.pose3d(zero(W, d)))
    return M.Posterior(mu_z, torch.zeros_like(mu_z), mu_b, lv_b, nuis, mask)


@torch.no_grad()
def posterior_pass(model, windows, refine_steps, points, groups, seed, batch_windows, start="encoder",
                   keep_amortized=False, amp=None):
    """infer.infer up to the pooled posterior, in its order (every chunk encoded and refined, then the shape
    pooled per group), so that the same seed gives the same refinement draws: {chunks: [{post, post0, lengths,
    npw, gender, n}], mu_p, lv_p (pooled), mu_p0, lv_p0 (the amortized posterior pooled, keep_amortized), gen
    (the generator, to draw from next), refine (first and last objective sums)}. start: 'encoder', or 'prior' for
    prior_start (the encoder is not run)."""
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(int(seed) % (1 << 63))
    B = max(1, int(batch_windows))
    wins = list(windows)
    chunks, first, last = [], 0.0, 0.0
    for a in range(0, len(wins), B):
        ch = wins[a:a + B]
        batch = M.batch_from_windows(ch, sources=model.cfg.sources, families=model.heads.families, rate=model.rate,
                                     gen=0).to(dev)
        post0 = prior_start(model, batch) if start == "prior" else None
        with I._amp(dev, amp):
            post = model.encode(batch) if start == "encoder" else post0
            amort = post.detach() if keep_amortized else None
            if refine_steps:
                hist = []
                post = refine(model, post, batch, int(refine_steps), points_per_stream=points, generator=gen,
                              history=hist)
                first += hist[0]
                last += hist[-1]
        chunks.append({"post": post.detach(), "post0": amort, "lengths": batch.lengths.to(dev),
                       "npw": batch.tokens.n_patches.to(dev), "gender": batch.gender.to(dev),
                       "n": [min(int(round(wd.duration * model.rate)), int(L)) for wd, L in
                             zip(ch, batch.lengths.tolist())]})
        del batch
    gid = I._group_ids(groups, len(wins)).to(dev)
    out = {"chunks": chunks, "gen": gen, "refine": {"first": first if refine_steps else None,
                                                    "last": last if refine_steps else None}}
    out["mu_p"], out["lv_p"] = subject_pool(torch.cat([c["post"].mu_b for c in chunks]),
                                            torch.cat([c["post"].logvar_b for c in chunks]), gid)
    if keep_amortized:
        out["mu_p0"], out["lv_p0"] = subject_pool(torch.cat([c["post0"].mu_b for c in chunks]),
                                                  torch.cat([c["post0"].logvar_b for c in chunks]), gid)
    return out


def _decode(model, post, beta, c, amp, z=None):
    """(rot6d (M, 22, 6), joints (M, 22, 3), global rotations (M, 22, 3, 3), transl (M, 3), window ids (M,)) of
    a chunk decoded at every knot, the forward kinematics of infer._pose; z default the posterior mean."""
    L, npw = c["lengths"], c["npw"]
    w, k = all_knots(L)
    mask = post.mask[..., None].to(post.mu_z.dtype)
    with I._amp(model.device, amp):
        bo = model.decode(post.mu_z * mask if z is None else z, beta, (w, k), npw, L)
    pos, R = I._pose(model, bo.rot6d, bo.transl, beta, c["gender"], w)
    return bo.rot6d.float(), pos, R, bo.transl.float(), w


def _cuts(c):
    L = c["lengths"].tolist()
    starts = np.concatenate([[0], np.cumsum(L)[:-1]]).astype(int)
    return [(int(s), int(n)) for s, n in zip(starts, c["n"])]


def _pad_rows(arrs, lengths):
    """Concatenate per-window arrays (n_j, ...) padded to lengths[j] rows by repeating the last row."""
    out = []
    for a, L in zip(arrs, lengths):
        a = np.asarray(a)
        if a.shape[0] < L:
            a = np.concatenate([a, np.repeat(a[-1:], L - a.shape[0], 0)], 0)
        out.append(a[:L])
    return np.concatenate(out, 0)


@torch.no_grad()
def main_pass(loaded, gset, windows, truth_of, refine_steps=g1.PARAMS["refine_steps"],
              points=g1.PARAMS["refine_points"], batch_windows=32, block_windows=32, seed=SEED, pool=True,
              best_of=BEST_OF, amp=None, log=None, oracle6=None):
    """The full pass (module docstring): one row per window with its keys (window, take_id, archive, person_id,
    task, n) and the columns of FULL_ROWS and JOINT_COLS in m, plus motion_m (mean over the joints of the std
    over the knots of the pelvis-relative truth, a stillness measure). truth_of(i) -> {R_local, betas, gender}
    (TruthBodies). oracle6 (opt-in, task ORI6): None, or the level sets of oracle6_levels ({L1, L2: {site:
    {offset, slow, fast}}}), which add the columns of ORACLE6_ROWS; the draws of every other column are those of
    the default pass. Returns (DataFrame, blocks log)."""
    log = log or (lambda *a: None)
    model = loaded.model
    dev = model.device
    blk = I.G1Predictor(loaded, refine_steps=refine_steps, points=points, batch_windows=batch_windows,
                        block_windows=block_windows, seed=seed, pool=pool, configs=(CONFIG,))
    windows = [int(i) for i in windows]
    blk.subset = windows
    rows, blocks, done = [], [], set()
    t0 = time.time()
    for i in windows:
        if i in done:
            continue
        idx = blk.block(gset, i)
        done |= set(idx)
        tb = time.time()
        takes = [str(gset.windows["take_id"].iat[j]) for j in idx]
        wins = [I.g1_windows(gset, j, (CONFIG,))[CONFIG] for j in idx]
        st = posterior_pass(model, wins, refine_steps, points, takes if pool else None,
                            I.seed_of(seed, gset.split, takes[0], CONFIG), batch_windows, keep_amortized=True, amp=amp)
        truths = [truth_of(j) for j in idx]
        a = 0
        for c in st["chunks"]:
            Wc = c["post"].mu_z.shape[0]
            jj = idx[a:a + Wc]
            tr = truths[a:a + Wc]
            mb, lb = st["mu_p"][a:a + Wc], st["lv_p"][a:a + Wc]
            mb0 = st["mu_p0"][a:a + Wc]
            a += Wc
            Lc = c["lengths"].tolist()
            rot6d, pos, Rg, transl, w = _decode(model, c["post"], mb, c, amp)
            Rloc = rot6d_to_matrix(rot6d)
            Rt = torch.as_tensor(_pad_rows([t["R_local"] for t in tr], Lc), dtype=torch.float32, device=dev)
            bt = torch.as_tensor(np.stack([t["betas"] for t in tr]), dtype=torch.float32, device=dev)
            zt = torch.zeros_like(transl)
            pos_t, Rg_t = I._pose(model, matrix_to_rot6d(Rt), zt, bt, c["gender"], w)
            # the window's G1a yaw of the posterior mean (it turns the prediction onto the truth), per decoded row
            pos_np = pos.cpu().numpy()
            th = np.zeros(pos_np.shape[0])
            for (s0, n), L, j in zip(_cuts(c), Lc, jj):
                J, _ = gset.truth(j)
                p = pos_np[s0:s0 + n]
                th[s0:s0 + L] = g1.yaw_fit(J[:n] - J[:n, :1], p - p[:, :1])
            theta = torch.as_tensor(th, dtype=torch.float32, device=dev)
            alt = {}
            for name, joints in (("true_root", (0,)), ("sub_legs", GROUPS["legs"]),
                                 ("sub_spine_head", GROUPS["spine_head"]), ("sub_arms", GROUPS["arms"])):
                Rs = substitute_global(Rg, Rg_t, joints, theta)
                alt[name + "_m"] = I._pose(model, matrix_to_rot6d(Rs), transl, mb, c["gender"], w)[0]
                Rs = substitute(Rloc, Rt, joints)
                alt[name + "_local_m"] = I._pose(model, matrix_to_rot6d(Rs), transl, mb, c["gender"], w)[0]
            alt["true_betas_skel_m"] = I._pose(model, rot6d, transl, bt, c["gender"], w)[0]
            alt["true_betas_m"] = _decode(model, c["post"], bt, c, amp)[1]
            alt["refine0_m"] = _decode(model, c["post0"], mb0, c, amp)[1]
            alt["floor_kin_m"] = pos_t
            Rf = []
            for t in tr:
                R = torch.as_tensor(t["R_local"], dtype=torch.float64).clone()
                R[:, 0] = with_root_yaw(R[:, 0], truth_root_yaw_path(R[:, 0]))
                Rf.append(R.numpy())
            Rf = torch.as_tensor(_pad_rows(Rf, Lc), dtype=torch.float32, device=dev)
            alt["floor_m"] = I._pose(model, matrix_to_rot6d(Rf), zt, bt, c["gender"], w)[0]
            if oracle6 is not None:
                units = [oracle6_unit_draws(j, n) for (_, n), j in zip(_cuts(c), jj)]
                psi = {lvl: torch.as_tensor(_pad_rows([oracle6_psi(u, oracle6[lvl]) for u in units], Lc),
                                            dtype=torch.float32, device=dev)
                       for lvl in sorted({r[3] for r in ORACLE6_ROWS if r[3]})}
                for _, col, mode, lvl, _ in ORACLE6_ROWS:
                    Rs = local_from_global(oracle6_globals(Rg, Rg_t, theta, mode, psi.get(lvl)))
                    alt[col] = I._pose(model, matrix_to_rot6d(Rs), transl, mb, c["gender"], w)[0]
            mask = c["post"].mask[..., None].to(c["post"].mu_z.dtype)
            draws = []
            for _ in range(int(best_of)):
                z = sample(c["post"].mu_z, c["post"].logvar_z, generator=st["gen"]) * mask
                beta = sample(mb, lb, generator=st["gen"])
                draws.append(_decode(model, c["post"], beta, c, amp, z=z)[1].cpu().numpy())
            pos, yaw_p = pos_np, yaw_angle(Rg[:, 0].float()).cpu().numpy()
            alt = {k: v.cpu().numpy() for k, v in alt.items()}
            for (s0, n), j, t in zip(_cuts(c), jj, tr):
                J, yaw = gset.truth(j)
                J, yaw = J[:n], yaw[:n]
                sl = slice(s0, s0 + n)
                r = gset.windows.iloc[int(j)]
                row = {"window": int(j), "take_id": r["take_id"], "archive": r["archive"], "person_id": r["person_id"],
                       "task": int(r["task"]), "n": int(n)}
                row.update(window_metrics(pos[sl], J, yaw, yaw_p[sl]))
                for k, v in alt.items():
                    row[k] = g1a_of(v[sl], J)
                e = np.array([g1a_of(d[sl], J) for d in draws]) if draws else np.array([math.nan])
                row["best_m"], row["draw_mean_m"] = float(e.min()), float(e.mean())
                rel = J - J[:, :1]
                row["motion_m"] = float(rel.std(0).mean())
                rows.append(row)
        blocks.append({"first_take": takes[0], "takes": len(set(takes)), "windows": len(idx),
                       "refine_first": st["refine"]["first"], "refine_last": st["refine"]["last"],
                       "seconds": round(time.time() - tb, 2)})
        if len(rows) and len(blocks) % 10 == 0:
            log(f"main pass: {len(rows)} windows, {time.time() - t0:.0f} s")
    return pd.DataFrame(rows), blocks


@torch.no_grad()
def mean_pass(loaded, gset, windows, refine_steps, start="encoder", points=g1.PARAMS["refine_points"],
              batch_windows=32, block_windows=32, seed=SEED, pool=True, name="g1a_m", amp=None, log=None):
    """G1a (m) of the posterior-mean body per window after refine_steps refinement steps from the encoder or from
    prior_start (start 'prior': analysis by synthesis), with the blocks and seeds of the harness: DataFrame of
    the window keys and the column `name`; and the blocks log."""
    log = log or (lambda *a: None)
    model = loaded.model
    blk = I.G1Predictor(loaded, block_windows=block_windows, batch_windows=batch_windows, configs=(CONFIG,))
    windows = [int(i) for i in windows]
    blk.subset = windows
    rows, blocks, done = [], [], set()
    for i in windows:
        if i in done:
            continue
        idx = blk.block(gset, i)
        done |= set(idx)
        tb = time.time()
        takes = [str(gset.windows["take_id"].iat[j]) for j in idx]
        wins = [I.g1_windows(gset, j, (CONFIG,))[CONFIG] for j in idx]
        st = posterior_pass(model, wins, refine_steps, points, takes if pool else None,
                            I.seed_of(seed, gset.split, takes[0], CONFIG), batch_windows, start=start, amp=amp)
        a = 0
        for c in st["chunks"]:
            Wc = c["post"].mu_z.shape[0]
            jj = idx[a:a + Wc]
            mb = st["mu_p"][a:a + Wc]
            a += Wc
            pos = _decode(model, c["post"], mb, c, amp)[1].cpu().numpy()
            for (s0, n), j in zip(_cuts(c), jj):
                J, _ = gset.truth(j)
                r = gset.windows.iloc[int(j)]
                rows.append({"window": int(j), "take_id": r["take_id"], "archive": r["archive"],
                             "person_id": r["person_id"], "task": int(r["task"]), "n": int(n),
                             name: g1a_of(pos[s0:s0 + n], J[:n])})
        blocks.append({"first_take": takes[0], "windows": len(idx), "start": start, "steps": int(refine_steps),
                       "refine_first": st["refine"]["first"], "refine_last": st["refine"]["last"],
                       "seconds": round(time.time() - tb, 2)})
        log(f"{name}: {len(rows)} windows")
    return pd.DataFrame(rows), blocks


def acc_only_windows(gset, j):
    """The single-wrist window of G1 window j with its gyro masked (values 0, channel mask off): acc only."""
    wd = I.g1_windows(gset, j, ("wrist",))["wrist"]
    s = wd.streams[0]
    s.values[..., 3:6] = 0.0
    s.chan_mask[:, 3:6] = False
    return wd


@torch.no_grad()
def acc_only_wrist(loaded, gset, windows, refine_steps=g1.PARAMS["refine_steps"], points=g1.PARAMS["refine_points"],
                   samples=g1.PARAMS["latent_samples"], decoder_draws=g1.PARAMS["decoder_draws"], batch_windows=32,
                   block_windows=32, seed=SEED, pool=True, c=None, log=None):
    """The acc-only single-wrist predictive (64 left-wrist draws per knot, infer.infer) on the windows of gset.
    On calib (c None): {'c', 'n_frames', 'n_degenerate', 'coverage_at_c'} of the conformal fit, every knot a
    radius. On dev (c given): DataFrame of the window keys with coverage, degenerate, width_m and energy_m of
    g1.predictive_scores."""
    from mdfm.eval import conformal
    log = log or (lambda *a: None)
    blk = I.G1Predictor(loaded, block_windows=block_windows, batch_windows=batch_windows, configs=("wrist",))
    windows = [int(i) for i in windows]
    blk.subset = windows
    rows, radii, bad, done = [], [], 0, set()
    for i in windows:
        if i in done:
            continue
        idx = blk.block(gset, i)
        done |= set(idx)
        takes = [str(gset.windows["take_id"].iat[j]) for j in idx]
        wins = [acc_only_windows(gset, j) for j in idx]
        bodies = I.infer(loaded.model, wins, refine_steps=refine_steps, points=points,
                         groups=takes if pool else None, samples=samples, decoder_draws=decoder_draws,
                         batch_windows=batch_windows, seed=I.seed_of(seed, gset.split, takes[0], "wrist_acc"))
        for j, b in zip(idx, bodies):
            J, yaw = gset.truth(j)
            y = g1.wrist_truth(J, yaw)
            D = np.asarray(b.draws, dtype=np.float64)
            if c is None:
                mean, cov = conformal.moment_match(D, ddof=g1.PARAMS["draws_ddof"])
                r, bd = conformal.mahalanobis(y, mean, cov)
                radii.append(r)
                bad += int(np.broadcast_to(bd, r.shape).sum())
            else:
                ps = g1.predictive_scores(D, y, c)
                r = gset.windows.iloc[int(j)]
                rows.append({"window": int(j), "take_id": r["take_id"], "archive": r["archive"],
                             "person_id": r["person_id"], "task": int(r["task"]), "n": int(len(y)),
                             "coverage": float(ps["covered"].mean()), "degenerate": int(ps["degenerate"].sum()),
                             "width_m": float(ps["width"].mean()), "energy_m": float(ps["energy"].mean())})
        log(f"acc-only wrist ({gset.split}): {len(done)} windows")
    if c is None:
        r = np.concatenate(radii)
        dof, level = g1.PARAMS["ellipsoid_dof"], g1.PARAMS["conformal_level"]
        cc = conformal.ellipsoid_factor(r, dof, level)
        cov = float(np.mean(conformal.covered(r, cc, dof, level))) if math.isfinite(cc) else 0.0
        return {"c": float(cc), "n_frames": int(r.size), "n_degenerate": bad, "coverage_at_c": cov}
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- aggregation
def pwm(t, col):
    """The G1 statistic of a column: mean over persons of the knot-weighted person means."""
    return common.person_weighted_mean(t[col].to_numpy(dtype=np.float64), t["person_id"].to_numpy(), t["n"].to_numpy())


def kwm(t, col):
    """Knot-weighted mean of a column over windows."""
    w = t["n"].to_numpy(dtype=np.float64)
    return float(np.sum(t[col].to_numpy(dtype=np.float64) * w) / np.sum(w))


def boot(t, col, values=None):
    """Person-clustered 95 percent interval (the G1 rule's bootstrap) of a column, or of `values` (N,)."""
    v = t[col].to_numpy(dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
    return common.cluster_bootstrap(v, t["person_id"].to_numpy(), g1.PARAMS["bootstrap_n"],
                                    g1.PARAMS["bootstrap_seed"], t["n"].to_numpy(), g1.PARAMS["bootstrap_level"])


def row_summary(t, col, ref_col="g1a_m", what="", partial=None):
    """{point_mm, interval_mm, knot_weighted_mm, delta_mm (ref minus row), delta_interval_mm, n_windows, n_knots,
    n_persons, n_takes, what, partial} of one column of a window table."""
    ok = t[col].notna()
    s = t[ok]
    out = {"what": what, "partial": partial, "n_windows": int(len(s)), "n_knots": int(s["n"].sum()),
           "n_persons": int(s["person_id"].nunique()), "n_takes": int(s["take_id"].nunique())}
    if s.empty:
        return dict(out, point_mm=None, interval_mm=None, knot_weighted_mm=None, delta_mm=None, delta_interval_mm=None)
    lo, hi = boot(s, col)
    out.update(point_mm=MM * pwm(s, col), interval_mm=[MM * lo, MM * hi], knot_weighted_mm=MM * kwm(s, col))
    if ref_col and ref_col != col and ref_col in s:
        d = s[ref_col].to_numpy(dtype=np.float64) - s[col].to_numpy(dtype=np.float64)
        dl, dh = boot(s, col, d)
        out.update(delta_mm=MM * common.person_weighted_mean(d, s["person_id"].to_numpy(), s["n"].to_numpy()),
                   delta_interval_mm=[MM * dl, MM * dh], delta_of=ref_col)
    else:
        out.update(delta_mm=None, delta_interval_mm=None)
    return out


def task_names():
    from mdfm.data.windows import task_vocabulary
    return list(task_vocabulary()["tasks"])


def aggregate_rows(table, cols, key):
    """{group: {n_windows, n_knots, n_persons, <row>_mm (knot-weighted), <row>_person_mm (G1 statistic)}} of the
    columns cols {row name: column} per value of key ('task' as vocabulary names, or 'archive')."""
    names = task_names()
    out = {}
    for g, sub in table.groupby(key, sort=True):
        label = names[int(g)] if key == "task" else str(g)
        r = {"n_windows": int(len(sub)), "n_knots": int(sub["n"].sum()), "n_persons": int(sub["person_id"].nunique())}
        for name, col in cols.items():
            s = sub[sub[col].notna()] if col in sub else sub.iloc[:0]
            r[f"{name}_mm"] = MM * kwm(s, col) if len(s) else None
            r[f"{name}_person_mm"] = MM * pwm(s, col) if len(s) else None
        out[label] = r
    return out


def read_archives(path=ARCHIVES_CSV):
    """{archive: {split, persons, motions, hours}} of reports/amass_split/archives.csv."""
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return {r["archive"]: {"split": r["split"], "persons": int(r["persons"]), "motions": int(r["motions"]),
                               "hours": float(r["hours"])} for r in csv.DictReader(f)}


def lockbox_person_weights(path=PERSONS_CSV):
    """({lockbox archive: share of the lockbox persons}, number of persons) from the per-person archive lists of
    reports/amass_split/persons.csv: a person counts once, split equally over the archives that hold it (the
    DFaust and MoSh subjects are the same persons). No id is returned."""
    import csv
    w, n = {}, 0
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != "lockbox":
                continue
            arcs = [a for a in r["archives"].split(";") if a]
            n += 1
            for a in arcs:
                w[a] = w.get(a, 0.0) + 1.0 / len(arcs)
    return {a: v / n for a, v in sorted(w.items())}, n


def lockbox_mix(by_task, by_archive, overall_mm, row="g1a", path=ARCHIVES_CSV, persons_path=PERSONS_CSV,
                weighting="person"):
    """The lockbox-mix estimate of one row (module docstring): {estimate_by_task_mm, estimate_by_archive_mm,
    weights {archive: persons share}, task_of, dev_archive_of, label}; from the counts of reports/amass_split
    only (lockbox_person_weights, archives.csv). weighting 'person' reads the groups' G1 statistic
    (<row>_person_mm, the scale of g1a_mm and of the kill line; overall_mm should then be the G1 statistic of
    every window), 'knot' their knot-weighted values (<row>_mm)."""
    if weighting not in ("person", "knot"):
        raise ValueError(f"weighting {weighting!r}, expected 'person' or 'knot'")
    arch = read_archives(path)
    lock = {a: v for a, v in arch.items() if v["split"] == "lockbox"}
    dev = {a: v for a, v in arch.items() if v["split"] == "dev"}
    weights, total = lockbox_person_weights(persons_path)
    if set(weights) != set(lock):
        raise ValueError(f"lockbox archives of persons.csv {sorted(weights)} differ from archives.csv {sorted(lock)}")
    key = f"{row}_person_mm" if weighting == "person" else f"{row}_mm"

    def dur(v):
        return v["hours"] * 3600.0 / v["motions"] if v["hours"] > 0 and v["motions"] > 0 else None

    def size(g):
        return {"windows": (g or {}).get("n_windows"), "persons": (g or {}).get("n_persons")}

    task_of, val_t, dev_of, val_a, n_t, n_a = {}, {}, {}, {}, {}, {}
    for a in weights:
        t = LOCKBOX_ARCHIVE_TASK.get(a, "unknown")
        v = (by_task.get(t) or {}).get(key)
        task_of[a] = t if v is not None else f"{t} (no dev window: all windows)"
        val_t[a] = v if v is not None else overall_mm
        n_t[a] = size(by_task.get(t)) if v is not None else "all windows"
        d = dur(lock[a])
        cands = [(abs(math.log(d / dur(dv))), da) for da, dv in dev.items() if d is not None and dur(dv)
                 and (by_archive.get(da) or {}).get(key) is not None]
        if cands:
            da = min(cands)[1]
            dev_of[a] = da
            val_a[a] = by_archive[da][key]
            n_a[a] = size(by_archive.get(da))
        else:
            dev_of[a] = "no take duration in the counts: all windows"
            val_a[a] = overall_mm
            n_a[a] = "all windows"
    est_t = sum(weights[a] * val_t[a] for a in weights) if all(v is not None for v in val_t.values()) else None
    est_a = sum(weights[a] * val_a[a] for a in weights) if all(v is not None for v in val_a.values()) else None
    scale = ("the G1 statistic of each dev group (mean over its persons of the knot-weighted person means), the "
             "scale of g1a_mm" if weighting == "person" else "the knot-weighted value of each dev group")
    return {"label": "estimate: no lockbox array opened; weights are each lockbox archive's share of the lockbox "
                     "persons (reports/amass_split/persons.csv, a person split over its archives); take durations "
                     "from reports/amass_split/archives.csv; the archive-to-task table is an assumption; values: "
                     + scale,
            "row": row, "weighting": weighting, "estimate_by_task_mm": est_t, "estimate_by_archive_mm": est_a,
            "weights": weights, "task_of": task_of, "dev_archive_of": dev_of, "values_by_task_mm": val_t,
            "values_by_archive_mm": val_a, "dev_group_size_by_task": n_t, "dev_group_size_by_archive": n_a,
            "lockbox_persons": total, "archives_csv_sha256": common.sha256_file(path),
            "persons_csv_sha256": common.sha256_file(persons_path)}


def per_take(table, part, cols, pcols):
    """per_take records (module docstring): knot-weighted means in mm of every row, partial rows on the partial
    takes only; refine20_mm is g1a_mm, and on a partial take its 20-step value in the partial blocks
    (refine20_partial_mm), the reference that scripts/weekend_branch.py pairs with refine60_mm and abs_mm."""
    names = task_names()
    out = []
    pt = {t: s for t, s in part.groupby("take_id", sort=False)} if part is not None and len(part) else {}
    for t, s in table.groupby("take_id", sort=False):
        rec = {"take_id": t, "archive": s["archive"].iat[0], "person_id": s["person_id"].iat[0],
               "task": names[int(s["task"].iat[0])], "windows": int(len(s)), "knots": int(s["n"].sum())}
        for name, col in cols.items():
            rec[f"{name}_mm"] = MM * kwm(s, col) if s[col].notna().all() else None
        rec["refine20_mm"] = rec["g1a_mm"]
        if t in pt:
            p = pt[t]
            for name, col in pcols.items():
                if col in p and p[col].notna().all():
                    rec[f"{name}_mm"] = MM * kwm(p, col)
            # the reference of refine60 and abs: the 20-step value in the same blocks as theirs
            if rec.get("refine20_partial_mm") is not None:
                rec["refine20_mm"] = rec["refine20_partial_mm"]
        out.append(rec)
    return out


# ---------------------------------------------------------------- report
def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_iv(iv, nd=1):
    return "n/a" if not iv or iv[0] is None else f"{fmt(iv[0], nd)} to {fmt(iv[1], nd)}"


def markdown(res):
    """decompose.md: the attribution table and the breakdowns, every number read from res."""
    R = res["rows"]
    g = res["g1a_mm"]
    ck = res["checkpoint"]
    L = ["# G1a gap breakdown on the dev windows (task T05)", "",
         "Written by scripts/probe_g1_decompose.py; every number below is read from decompose.json of this folder. "
         "Development numbers under the draft G1 rule on the dev archives, not a gate read: no lockbox array was "
         "opened and lockbox_reads.csv is untouched.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,}. Rule docs/rules/G1_RULE.md SHA-256 {res['rule_sha256']} (status: "
         f"{str(res['rule_status']).rstrip('.')}). MoveAtlas "
         f"HEAD {res['moveatlas_head']}.", "",
         f"Set: {res['sets']['dev']['windows']:,} dev windows, {res['sets']['dev']['knots']:,} knots, "
         f"{res['sets']['dev']['persons']} persons, {res['sets']['dev']['takes']:,} takes (manifest SHA-256 "
         f"{res['sets']['dev']['manifest_sha256']})."
         + (f" Whole run limited to the first {res['limit_takes']} takes of each archive." if res.get("limit_takes")
            else ""), ""]
    j1 = res["j1"]
    if j1.get("g1a_mm") is not None:
        cm = j1.get("checkpoint_matches")
        sd = j1.get("settings_differ")
        L += [f"Harness check: G1a {fmt(g, 2)} mm here, {fmt(j1['g1a_mm'], 2)} mm in {j1['path']}, difference "
              f"{fmt(j1['difference_mm'], 2)} mm (within 2 mm: {'yes' if j1['within_2mm'] else 'no'}); on the "
              f"{j1.get('windows_compared', 0):,} windows both scored, difference "
              f"{fmt(j1.get('same_windows_difference_mm'), 2)} mm, largest per-window difference "
              f"{fmt(j1.get('max_window_difference_mm'), 3)} mm. Same checkpoint: "
              f"{'n/a' if cm is None else ('yes' if cm else 'NO')}; inference settings that differ from J1's: "
              + ("n/a" if sd is None else (", ".join(f"{k} {v[0]} there, {v[1]} here" for k, v in sd.items())
                                          or "none"))
              + (f"; J1 was a partial run ({j1['partial_run']})" if j1.get("partial_run") else "") + ".", ""]
    else:
        L += [f"Harness check: no J1 g1.json at {j1['path']}; G1a was computed in this process.", ""]
    tc = res["truth_check"]
    L += [f"Truth rotations rebuilt from {tc['takes']:,} raw takes; largest deviation from the stored truth joints "
          f"{fmt(tc['max_deviation_mm'], 4)} mm.", "",
          "## Attribution", "",
          f"Point: the G1 statistic (mean over persons of the knot-weighted person means); interval: person-clustered "
          f"bootstrap, {g1.PARAMS['bootstrap_n']} resamples, seed {g1.PARAMS['bootstrap_seed']}, 95 percent. Change: "
          f"the 20-step G1a (refine20_partial for the partial rows) minus the row, paired per window. Share of the "
          f"gap: the change over the 20-step G1a's distance to the {GAP_PASS_MM:g} mm pass line.", "",
          "| row | what | value mm | 95% interval | change mm | change interval | share of the gap | windows | set |",
          "|---|---|---|---|---|---|---|---|---|"]
    gap = None if g is None else g - GAP_PASS_MM
    for name in [r[0] for r in FULL_ROWS] + [r[0] for r in PARTIAL_ROWS]:
        r = R.get(name)
        if r is None:
            continue
        share = (r["delta_mm"] / gap) if (r.get("delta_mm") is not None and gap and gap > 0) else None
        L.append(f"| {name} | {r['what']} | {fmt(r['point_mm'])} | {fmt_iv(r['interval_mm'])} | {fmt(r['delta_mm'])} | "
                 f"{fmt_iv(r['delta_interval_mm'])} | {fmt(share, 2)} | {r['n_windows']:,} | "
                 f"{'partial: ' + r['partial'] if r['partial'] else 'full'} |")
    L += ["", "## By joint group", "",
          "Global: the segments' true global orientations, every other segment as decoded. Local: the joints' true "
          "local rotations, every other joint's decoded local rotation kept. Share: the group's joints' per-joint "
          "errors over 22, summing to G1a.", "",
          "| group | joints | global mm | gain mm | gain interval | local mm | gain mm | share of G1a mm |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in res["by_group"].items():
        if not isinstance(v, dict) or "joints" not in v:
            continue
        L.append(f"| {k} | {', '.join(BODY_JOINTS[j] for j in v['joints'])} | {fmt(v['substituted_mm'])} | "
                 f"{fmt(v['gain_mm'])} | {fmt_iv(v['gain_interval_mm'])} | {fmt(v['substituted_local_mm'])} | "
                 f"{fmt(v['gain_local_mm'])} | {fmt(v['share_of_g1a_mm'])} |")
    cols = ("g1a", "refine0", "true_root_orient", "true_betas", "best_of_16", "floor", "heading", "pa")
    for title, key in (("By task (knot-weighted)", "by_task"), ("By archive (knot-weighted)", "by_archive")):
        L += ["", f"## {title}", "", "| group | windows | persons | " + " | ".join(f"{c} mm" for c in cols) + " |",
              "|---|---|---|" + "---|" * len(cols)]
        for k, v in res[key].items():
            L.append(f"| {k} | {v['n_windows']:,} | {v['n_persons']} | "
                     + " | ".join(fmt(v.get(f"{c}_mm")) for c in cols) + " |")
    bs = res.get("best_of_16_by_motion") or []
    if bs:
        L += ["", "## Best of 16 by stillness of the truth (terciles of motion_m)", "",
              "| tercile | motion mm | windows | G1a mm | best of 16 mm | single draw mm |", "|---|---|---|---|---|---|"]
        for r in bs:
            L.append(f"| {r['tercile']} | {fmt(r['motion_lo_mm'])} to {fmt(r['motion_hi_mm'])} | {r['n_windows']:,} | "
                     f"{fmt(r['g1a_mm'])} | {fmt(r['best_of_16_mm'])} | {fmt(r['single_draw_mm'])} |")
    lm = res["lockbox_mix"]
    kn = lm.get("knot_weighted") or {}

    def gsize(s):
        if isinstance(s, str):
            return s
        p = s.get("persons")
        return f"{s.get('windows')} windows, {p} {'person' if p == 1 else 'persons'}"

    L += ["", "## Lockbox mix (estimate)", "",
          f"{lm['label']}. G1a by task {fmt(lm['estimate_by_task_mm'])} mm (lockbox_mix_mm), by archive "
          f"{fmt(lm['estimate_by_archive_mm'])} mm, over {lm['lockbox_persons']} lockbox persons; the dev G1a on "
          f"the same scale is {fmt(g)} mm. On knot-weighted group values (dev G1a knot-weighted "
          f"{fmt(kn.get('g1a_knot_weighted_mm'))} mm): by task {fmt(kn.get('estimate_by_task_mm'))} mm, by archive "
          f"{fmt(kn.get('estimate_by_archive_mm'))} mm.", "",
          "| lockbox archive | person share | task (assumed) | value mm | dev group | nearest dev archive by take "
          "duration | value mm |", "|---|---|---|---|---|---|---|"]
    for a, wgt in lm["weights"].items():
        L.append(f"| {a} | {fmt(wgt, 3)} | {lm['task_of'][a]} | {fmt(lm['values_by_task_mm'][a])} | "
                 f"{gsize(lm.get('dev_group_size_by_task', {}).get(a, 'n/a'))} | {lm['dev_archive_of'][a]} | "
                 f"{fmt(lm['values_by_archive_mm'][a])} |")
    acc = res.get("acc_only_wrist") or {}
    L += ["", "## Acc-only wrist (single-wrist line)", ""]
    if acc.get("coverage") is not None:
        ref = acc.get("acc_gyro_j1") or {}
        L += [f"Wrist gyro masked; conformal factor on calib c = {fmt(acc['conformal']['c'], 4)} over "
              f"{acc['conformal']['n_frames']:,} knots. Coverage {fmt(acc['coverage'], 3)} "
              f"({fmt_iv(acc['coverage_interval'], 3)}), band width {fmt(acc['width_mm'])} mm, energy score "
              f"{fmt(acc['energy_mm'])} mm, degenerate knots {acc['degenerate']} of {acc['n_knots']:,}.",
              "", "Acc and gyro wrist of the J1 run on the same windows: "
              + (f"coverage {fmt(ref['coverage'], 3)}, band width {fmt(ref['width_mm'])} mm, energy score "
                 f"{fmt(ref['energy_mm'])} mm; acc-only minus acc and gyro: coverage {fmt(ref['coverage_delta'], 3)} "
                 f"({fmt_iv(ref['coverage_delta_interval'], 3)}), band width {fmt(ref['width_delta_mm'])} mm, energy "
                 f"score {fmt(ref['energy_delta_mm'])} mm." if ref else "not available (no J1 windows.csv).")]
    else:
        L += [f"Not evaluated: {acc.get('reason', 'skipped')}."]
    o6 = res.get("oracle6")
    if o6:
        pr = o6.get("prediction") or {}
        L += ["", "## Six-segment orientation oracle (task ORI6, opt-in)", "",
              "The decoded body with the true global orientation of the six instrumented segments only ("
              + ", ".join(f"{s}: {j}" for s, j in o6["segments"].items()) + "), every other segment as decoded; "
              f"heading error draws seeded by {o6.get('seed', 'n/a')} and the window index, re-referenced to the "
              f"{o6.get('reference_sensor', 'n/a')} sensor at the window centre. Change: the run's G1a minus the "
              "row, paired per window.", "",
              "| row | what | value mm | 95% interval | change mm | change interval | windows |",
              "|---|---|---|---|---|---|---|"]
        for name, r in o6["rows"].items():
            L.append(f"| {name} | {r['what']} | {fmt(r['point_mm'])} | {fmt_iv(r['interval_mm'])} | "
                     f"{fmt(r['delta_mm'])} | {fmt_iv(r['delta_interval_mm'])} | {r['n_windows']:,} |")
        L += ["", f"Reading: {o6['reading'] or 'n/a'} (variant a {fmt(o6['exact_mm'], 2)} mm, variant b "
                  f"{fmt(o6['noisy_mm'], 2)} mm, its person-bootstrap upper bound {fmt(o6['noisy_upper_mm'], 2)} mm, "
                  f"kill line {o6['kill_mm']:g} mm). Prediction "
                  + (f"{pr.get('path')} (SHA-256 {pr.get('sha256')}, hashed {pr.get('hashed_at')})" if pr
                     else "not checked") + "."]
    L += ["", "## Time and hashes", "",
          f"- run code SHA-256 ({', '.join(res['hashes']['run_code_files'])}): {res['hashes']['run_code_sha256']}",
          f"- scoring code SHA-256 ({', '.join(g1.SCORE_CODE)}): {res['hashes']['scoring_code_sha256']}",
          "- time (s): " + ", ".join(f"{k} {v}" for k, v in res["timing_s"].items()), ""]
    return "\n".join(L) + "\n"


def best_by_motion(table):
    """Terciles of motion_m: G1a, best of 16 and single draw (knot-weighted, mm)."""
    t = table[table["best_m"].notna()]
    if len(t) < 3:
        return []
    q = np.quantile(t["motion_m"].to_numpy(), [0, 1 / 3, 2 / 3, 1])
    out = []
    for i, name in enumerate(("stillest", "middle", "most moving")):
        s = t[(t["motion_m"] >= q[i]) & ((t["motion_m"] <= q[i + 1]) if i == 2 else (t["motion_m"] < q[i + 1]))]
        if len(s):
            out.append({"tercile": name, "motion_lo_mm": MM * q[i], "motion_hi_mm": MM * q[i + 1],
                        "n_windows": int(len(s)), "g1a_mm": MM * kwm(s, "g1a_m"), "best_of_16_mm": MM * kwm(s, "best_m"),
                        "single_draw_mm": MM * kwm(s, "draw_mean_m")})
    return out


J1_SETTINGS = ("refine_steps", "points_per_stream", "batch_windows", "block_windows", "seed", "shape_pooling",
               "weights")


def j1_check(path, table, checkpoint_sha256=None, inference=None):
    """{path, g1a_mm, difference_mm, within_2mm, max_window_difference_mm, windows_compared,
    same_windows_difference_mm, checkpoint_matches, settings_differ} against J1's g1.json and windows.csv of the
    same checkpoint (None values when absent). checkpoint_matches: J1's checkpoint SHA-256 equals
    checkpoint_sha256 (None when not given); settings_differ: the J1_SETTINGS of J1's inference block that differ
    from `inference` ({key: [J1, here]}), which change the blocks or the draws and so the per-window values."""
    out = {"path": common.rel(path), "g1a_mm": None}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    pt = next((r["point"] for r in d.get("lines", []) if str(r.get("line", "")).startswith("G1a")), None)
    if pt is None:
        return out
    mine = MM * pwm(table, "g1a_m")
    out.update(g1a_mm=float(pt), difference_mm=mine - float(pt), within_2mm=bool(abs(mine - float(pt)) <= 2.0),
               partial_run=d.get("partial_run"))
    ck = (d.get("checkpoint") or {}).get("sha256")
    out["checkpoint_matches"] = None if checkpoint_sha256 is None or ck is None else bool(ck == checkpoint_sha256)
    if inference is not None:
        ji = d.get("inference") or {}
        out["settings_differ"] = {k: [ji.get(k), inference.get(k)] for k in J1_SETTINGS
                                  if k in ji and k in inference and ji.get(k) != inference.get(k)}
    wp = os.path.join(os.path.dirname(path), "windows.csv")
    if os.path.exists(wp):
        w = pd.read_csv(wp, encoding="utf-8")
        m = table[["window", "person_id", "n", "g1a_m"]].merge(w[["window", "g1a_m"]], on="window",
                                                                 suffixes=("", "_j1"))
        if len(m):
            out["max_window_difference_mm"] = float(MM * np.abs(m["g1a_m"] - m["g1a_m_j1"]).max())
            out["windows_compared"] = int(len(m))
            out["same_windows_difference_mm"] = MM * (pwm(m, "g1a_m") - pwm(m, "g1a_m_j1"))
    return out


def acc_reference(acc, j1_windows):
    """The acc and gyro wrist of the J1 windows.csv on the acc-only windows, and the paired differences."""
    if not os.path.exists(j1_windows):
        return None
    w = pd.read_csv(j1_windows, encoding="utf-8")
    m = acc.merge(w[["window", "coverage", "width_m", "energy_m"]], on="window", suffixes=("", "_j1"))
    if not len(m):
        return None
    d = m["coverage"].to_numpy(dtype=np.float64) - m["coverage_j1"].to_numpy(dtype=np.float64)
    lo, hi = boot(m, "coverage", d)
    return {"coverage": pwm(m, "coverage_j1"), "width_mm": MM * pwm(m, "width_m_j1"),
            "energy_mm": MM * pwm(m, "energy_m_j1"), "coverage_delta": pwm(m, "coverage") - pwm(m, "coverage_j1"),
            "coverage_delta_interval": [lo, hi], "width_delta_mm": MM * (pwm(m, "width_m") - pwm(m, "width_m_j1")),
            "energy_delta_mm": MM * (pwm(m, "energy_m") - pwm(m, "energy_m_j1")), "windows": int(len(m))}


def assemble(table, part, acc, meta):
    """decompose.json (module docstring) from the full window table, the partial table (refine60, abs and the
    20-step g1a of its windows) and the acc-only dict; meta: checkpoint, sets, hashes, j1, timing and the rest."""
    full_cols = {name: col for name, col, _ in FULL_ROWS}
    part_cols = {name: col for name, col, _ in PARTIAL_ROWS}
    plabel = meta.get("partial_label")
    rows = {name: row_summary(table, col, "g1a_m", what) for name, col, what in FULL_ROWS}
    if part is not None and len(part):
        if PARTIAL_REF not in part:
            part = part.assign(**{PARTIAL_REF: part["g1a_m"]})
        for name, col, what in PARTIAL_ROWS:
            if col in part:
                rows[name] = row_summary(part, col, None if col == PARTIAL_REF else PARTIAL_REF, what, plabel)
    res = {"what": "G1a gap breakdown on the dev windows (task T05), development numbers under the draft G1 rule, "
                   "not a gate read",
           "partial_label": plabel}
    for name, key in TOP_KEYS.items():
        r = rows.get(name)
        res[key] = r["point_mm"] if r else None
    res["refine20_mm"] = res["g1a_mm"]
    by_task = aggregate_rows(table, full_cols, "task")
    by_arch = aggregate_rows(table, full_cols, "archive")
    if part is not None and len(part):
        pc = {k: v for k, v in part_cols.items() if v in part}
        for key, agg in (("task", by_task), ("archive", by_arch)):
            for label, r in aggregate_rows(part, pc, key).items():
                agg.setdefault(label, {}).update({f"{k}_mm": r.get(f"{k}_mm") for k in pc},
                                                 partial_windows=r["n_windows"])
    res["by_task"], res["by_archive"] = by_task, by_arch
    g = res["g1a_mm"]
    by_group = {}
    for k, joints, row in [(k, j, f"sub_{k}") for k, j in GROUPS.items()] + [("root_orientation", (0,), "true_root_orient")]:
        sub, loc = rows[row]["point_mm"], rows[row + "_local"]["point_mm"]
        share = MM * float(np.sum([pwm(table, JOINT_COLS[j]) for j in joints])) / N_BODY
        by_group[k] = {"joints": list(joints), "substituted_mm": sub, "gain_mm": None if sub is None else g - sub,
                       "gain_interval_mm": rows[row]["delta_interval_mm"], "substituted_local_mm": loc,
                       "gain_local_mm": None if loc is None else g - loc,
                       "gain_local_interval_mm": rows[row + "_local"]["delta_interval_mm"], "share_of_g1a_mm": share}
    by_group["per_joint_mm"] = {BODY_JOINTS[j]: MM * pwm(table, JOINT_COLS[j]) for j in range(N_BODY)}
    res["by_group"] = by_group
    arch_csv = meta.get("archives_csv", ARCHIVES_CSV)
    lm = lockbox_mix(by_task, by_arch, MM * pwm(table, "g1a_m"), "g1a", arch_csv, weighting="person")
    lk = lockbox_mix(by_task, by_arch, MM * kwm(table, "g1a_m"), "g1a", arch_csv, weighting="knot")
    lm["knot_weighted"] = {k: lk[k] for k in ("estimate_by_task_mm", "estimate_by_archive_mm", "values_by_task_mm",
                                              "values_by_archive_mm")}
    lm["knot_weighted"]["g1a_knot_weighted_mm"] = MM * kwm(table, "g1a_m")
    res["lockbox_mix"] = lm
    res["lockbox_mix_mm"] = lm["estimate_by_task_mm"]
    res["per_take"] = per_take(table, part, full_cols, part_cols)
    res["rows"] = rows
    res["best_of_16_by_motion"] = best_by_motion(table)
    res["acc_only_wrist"] = acc
    for k in ("checkpoint", "rule_sha256", "rule_status", "sets", "hashes", "j1", "truth_check", "inference",
              "moveatlas_head", "timing_s", "limit_takes", "blocks"):
        res[k] = meta.get(k)
    o6 = oracle6_block(table, meta.get("oracle6"))
    if o6 is not None:
        res["oracle6"] = o6
    return res


def oracle6_block(table, info=None):
    """decompose.json 'oracle6' (task ORI6) when the table holds the columns of ORACLE6_ROWS, else None: rows
    (row_summary of each, change from g1a paired per window), exact_mm and noisy_mm (variants a and b), the noisy
    row's bootstrap interval and upper bound, the reading (oracle6_reading), by_task and by_archive of the rows
    (aggregate_rows), the segments, and info (levels, seed, prediction check)."""
    present = [r for r in ORACLE6_ROWS if r[1] in table]
    if not present:
        return None
    rows = {name: row_summary(table, col, "g1a_m", what) for name, col, _, _, what in present}
    ex = (rows.get("oracle6_exact") or {}).get("point_mm")
    no = (rows.get("oracle6_noisy") or {}).get("point_mm")
    iv = (rows.get("oracle6_noisy") or {}).get("interval_mm")
    kill = g1.PARAMS["g1a_kill_mm"]
    cols = {name: col for name, col, _, _, _ in present}
    cols["g1a"] = "g1a_m"
    out = {"what": "six-segment orientation oracle (task ORI6): v1's decoded body with the true global orientation "
                   "of the six instrumented segments only; dev diagnostic under the draft G1 rule, not a gate read",
           "kill_mm": kill, "pass_mm": GAP_PASS_MM, "g1a_mm": MM * pwm(table, "g1a_m"),
           "exact_mm": ex, "noisy_mm": no, "noisy_interval_mm": iv, "noisy_upper_mm": iv[1] if iv else None,
           "reading": oracle6_reading(ex, no, kill),
           "noisy_below_exact": None if ex is None or no is None else bool(no < ex),
           "segments": {s: BODY_JOINTS[j] for s, j in zip(g1.SITES, ORACLE6_SEGMENTS)},
           "rows": rows, "by_task": aggregate_rows(table, cols, "task"),
           "by_archive": aggregate_rows(table, cols, "archive")}
    out.update(info or {})
    return out


def transfer_target(j, segments):
    """The joint whose decoded error joint j takes when the global orientations of `segments` are exact (the
    direct transfer of oracle6_attain): from j up the tree while the segment that places the current joint (its
    parent) is in segments; the pelvis (error 0, pelvis-relative) ends the walk."""
    k, seg = int(j), set(int(s) for s in segments)
    while k != 0 and PARENTS[k] in seg:
        k = PARENTS[k]
    return k


def transfer_gain(table, segments, descendants=False):
    """Per-window direct-transfer gain (m) of a substitution of `segments`: sum over the 22 joints of
    (e_j - e_k(j)) / 22, e the per-joint G1a errors of the window (JOINT_COLS), k(j) = transfer_target(j), e of
    the pelvis 0. Joints whose placing segment keeps its decoded orientation (the descendants of a transferred
    joint among them) keep their error: their translation is not counted. descendants True (the alternative
    rule, a sensitivity): such a joint gains the reduction e_a - e_k(a) of its nearest transferred ancestor a."""
    e = table[list(JOINT_COLS)].to_numpy(dtype=np.float64).copy()
    e[:, 0] = 0.0
    g = np.zeros(len(table))
    for j in range(1, N_BODY):
        k = transfer_target(j, segments)
        if k != j:
            g += e[:, j] - e[:, k]
        elif descendants:
            a = PARENTS[j]
            while a > 0 and transfer_target(a, segments) == a:
                a = PARENTS[a]
            if a > 0:
                g += e[:, a] - e[:, transfer_target(a, segments)]
    return g / N_BODY


def oracle6_attain(decompose_dir, levels, gset, log=None, windows=None):
    """Attainability of the oracle6 reading (task ORI6) from an existing decompose folder (decompose.json and
    windows.csv of the default run) and the stored truth joints of gset (the G1 dev set); no model and no oracle6
    value. Returns the dict written to attain.json:
    - needed_gain_mm: the decompose's g1a_mm minus the kill line.
    - calibration: per group of the decompose (root_orientation, legs, spine_head, arms) the direct-transfer
      estimate of its global substitution (transfer_gain, the G1 statistic in mm), the measured gain (its row's
      paired change) and their ratio; the same under the alternative descendant rule (descendants True).
    - six: the direct-transfer gain of ORACLE6_SEGMENTS by part (root, forearms, shanks, head), the calibrated
      gain (each part times the ratio of its group: root_orientation, arms, legs), feet_shift (the alternative
      descendant rule on the forearm, shank and head parts: the feet gain their ankles' reduction) and an
      optimistic gain (the foot segments also exact, so the feet take the knees' error); exact_estimate_mm =
      g1a_mm minus each gain.
    - noise: per level set the heading error SD of each sensor after re-referencing (the sum of the variances of
      offset, slow and fast of the sensor and of the reference; for the reference twice its slow and fast
      variances), and the G1a cost of that error: a joint placed by an instrumented segment s (and every scored
      descendant, which it translates) moves by 2 |sin(delta / 2)| |v_h|, v_h the horizontal part of the true
      vector from s's joint to it, so E[d^2] = 2 (1 - exp(-sigma^2 / 2)) mean_t |v_h|^2, summed over the
      instrumented segments on its chain; quadrature_mm = sum_j (sqrt(e_j^2 + E[d_j^2]) - e_j) / 22 with e_j the
      direct-transfer error, bound_mm = sum_j sqrt(E[d_j^2]) / 22 (triangle inequality, at the exact row's yaw);
      both the G1 statistic over the windows.
    - noisy_estimate_mm and expected_reading: exact estimates plus the noise cost, read by oracle6_reading."""
    log = log or (lambda *a: None)
    with open(os.path.join(decompose_dir, "decompose.json"), encoding="utf-8") as f:
        dec = json.load(f)
    wpath = os.path.join(decompose_dir, "windows.csv")
    t = pd.read_csv(wpath, encoding="utf-8")
    if windows is not None:
        t = t[t["window"].isin(set(int(w) for w in windows))].reset_index(drop=True)
    pw = lambda v: MM * common.person_weighted_mean(np.asarray(v, dtype=np.float64), t["person_id"].to_numpy(),  # noqa: E731
                                                    t["n"].to_numpy())
    g = float(dec["g1a_mm"])
    g_win = pw(t["g1a_m"])
    kill = g1.PARAMS["g1a_kill_mm"]
    rows = dec["rows"]
    cal = {}
    for name, segs, row in (("root_orientation", (0,), "true_root_orient"), ("legs", GROUPS["legs"], "sub_legs"),
                            ("spine_head", GROUPS["spine_head"], "sub_spine_head"), ("arms", GROUPS["arms"], "sub_arms")):
        est = pw(transfer_gain(t, segs))
        est_d = pw(transfer_gain(t, segs, descendants=True))
        meas = rows[row]["delta_mm"]
        cal[name] = {"segments": [BODY_JOINTS[s] for s in segs], "estimate_mm": est, "measured_mm": meas,
                     "measured_interval_mm": rows[row]["delta_interval_mm"],
                     "ratio": meas / est if est else None, "estimate_descendants_mm": est_d,
                     "ratio_descendants": meas / est_d if est_d else None}
    seg_of = dict(zip(g1.SITES, ORACLE6_SEGMENTS))
    parts = {"root": (seg_of["lower_back"],), "forearms": (seg_of["wrist_L"], seg_of["wrist_R"]),
             "shanks": (seg_of["shank_L"], seg_of["shank_R"]), "head": (seg_of["head"],)}
    part_cal = {"root": "root_orientation", "forearms": "arms", "shanks": "legs", "head": None}
    part_mm = {k: pw(transfer_gain(t, v)) for k, v in parts.items()}
    direct = pw(transfer_gain(t, ORACLE6_SEGMENTS))
    feet = tuple(ORACLE6_SEGMENTS) + (BODY_JOINTS.index("left_ankle"), BODY_JOINTS.index("right_ankle"))
    optimistic = pw(transfer_gain(t, feet))
    calibrated = sum(v * (cal[part_cal[k]]["ratio"] if part_cal[k] else 1.0) for k, v in part_mm.items())
    limbs = parts["forearms"] + parts["shanks"] + parts["head"]
    feet_shift = part_mm["root"] + pw(transfer_gain(t, limbs, descendants=True))
    gains = {"direct": direct, "calibrated": calibrated, "feet_shift": feet_shift, "optimistic": optimistic}
    exact_est = {k: g - v for k, v in gains.items()}

    # the noise cost: chains of (instrumented segment, the joint it places) above every scored joint
    six = set(ORACLE6_SEGMENTS)
    chains = {}
    for j in range(1, N_BODY):
        k, ch = j, []
        while k != 0:
            if PARENTS[k] in six:
                ch.append((PARENTS[k], k))
            k = PARENTS[k]
        chains[j] = ch
    pairs = sorted({p for ch in chains.values() for p in ch})
    h2 = np.zeros((len(t), len(pairs)))
    for r, i in enumerate(t["window"].to_numpy()):
        J, _ = gset.truth(int(i))
        J = J[:int(t["n"].iat[r])]
        for c, (s, a) in enumerate(pairs):
            v = J[:, a] - J[:, s]
            h2[r, c] = float(np.mean(v[:, 0] ** 2 + v[:, 2] ** 2))
        if (r + 1) % 500 == 0:
            log(f"oracle6 attain: truth geometry of {r + 1} windows")
    site_of = {s: site for site, s in seg_of.items()}
    e = t[list(JOINT_COLS)].to_numpy(dtype=np.float64).copy()
    e[:, 0] = 0.0
    e_a = e.copy()
    for j in range(1, N_BODY):
        e_a[:, j] = e[:, transfer_target(j, ORACLE6_SEGMENTS)]
    noise = {}
    for lvl in ("L1", "L2"):
        L = levels[lvl]
        ref = L[ORACLE6_REF]
        var_ref = sum(ref[k] ** 2 for k in ORACLE6_LEVEL_KEYS)
        sig = {}
        for site in g1.SITES:
            v = (2.0 * (ref["slow"] ** 2 + ref["fast"] ** 2) if site == ORACLE6_REF
                 else sum(L[site][k] ** 2 for k in ORACLE6_LEVEL_KEYS) + var_ref)
            sig[site] = math.sqrt(v)
        cpair = np.array([2.0 * (1.0 - math.exp(-math.radians(sig[site_of[s]]) ** 2 / 2.0)) for s, _ in pairs])
        d2 = np.zeros((len(t), N_BODY))
        for j, ch in chains.items():
            for p in ch:
                c = pairs.index(p)
                d2[:, j] += cpair[c] * h2[:, c]
        quad = (np.sqrt(e_a ** 2 + d2) - e_a).sum(1) / N_BODY
        bound = np.sqrt(d2).sum(1) / N_BODY
        noise[lvl] = {"sigma_deg": sig, "quadrature_mm": pw(quad), "bound_mm": pw(bound),
                      "rms_displacement_mm": {BODY_JOINTS[j]: MM * float(np.sqrt(np.average(
                          d2[:, j], weights=t["n"].to_numpy(dtype=np.float64)))) for j in range(1, N_BODY)
                          if chains[j]}}
    noisy_est = {lvl: {k: v + noise[lvl]["quadrature_mm"] for k, v in exact_est.items()} for lvl in noise}
    noisy_bound = {lvl: {k: v + noise[lvl]["bound_mm"] for k, v in exact_est.items()} for lvl in noise}
    reading = {k: oracle6_reading(exact_est[k], noisy_est["L1"][k], kill) for k in exact_est}
    return {"what": "attainability of the oracle6 reading (task ORI6): direct-transfer estimates from the rows "
                    "of an existing decompose folder and the stored dev truth joints; no model, no oracle6 value",
            "decompose": {"folder": common.rel(decompose_dir), "g1a_mm": g, "g1a_from_windows_mm": g_win,
                          "decompose_json_sha256": common.sha256_file(os.path.join(decompose_dir, "decompose.json")),
                          "windows_csv_sha256": common.sha256_file(wpath),
                          "checkpoint_sha256": (dec.get("hashes") or {}).get("checkpoint_sha256"),
                          "moveatlas_head": dec.get("moveatlas_head")},
            "windows": int(len(t)), "persons": int(t["person_id"].nunique()), "knots": int(t["n"].sum()),
            "kill_mm": kill, "pass_mm": GAP_PASS_MM, "needed_gain_mm": g - kill,
            "segments": {s: BODY_JOINTS[j] for s, j in seg_of.items()},
            "calibration": cal,
            "six": {"parts_mm": part_mm, "part_calibration_group": part_cal, "gain_mm": gains,
                    "exact_estimate_mm": exact_est},
            "noise": noise, "noisy_estimate_mm": noisy_est, "noisy_bound_mm": noisy_bound,
            "expected_reading": reading,
            "levels": levels, "seed": ORACLE6_SEED, "reference_sensor": ORACLE6_REF}


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=8.0, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--out", default=None, help="default reports/bench/g1_decompose_<checkpoint name>")
    p.add_argument("--root", default=None, help="G1 input root (default mdfm.gates.g1.default_root)")
    p.add_argument("--refine-steps", type=int, default=g1.PARAMS["refine_steps"])
    p.add_argument("--points", type=int, default=g1.PARAMS["refine_points"])
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--block-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--limit-takes", type=int, default=None, help="limit the whole run (a test run, labelled)")
    p.add_argument("--partial-takes", type=int, default=PARTIAL_TAKES, help="takes per archive of refine60 and abs")
    p.add_argument("--refine-long", type=int, default=REFINE_LONG)
    p.add_argument("--abs-steps", type=int, default=ABS_STEPS)
    p.add_argument("--best-of", type=int, default=BEST_OF)
    p.add_argument("--j1", default=None, help="J1's g1.json (default reports/gates/g1/dev_<name>/g1.json)")
    p.add_argument("--skip-acc-only", action="store_true")
    p.add_argument("--acc-only-limit-takes", type=int, default=None, help="limit the acc-only row (dev and calib)")
    p.add_argument("--oracle6", action="store_true",
                   help="opt-in: add the six-segment orientation oracle rows (task ORI6, module docstring)")
    p.add_argument("--oracle6-levels", default=ORACLE6_LEVELS_DIR,
                   help="folder of cross_sites.csv and heading_summary.csv (default reports/orientation_sources)")
    p.add_argument("--oracle6-prediction", default=None,
                   help="prediction.txt that must match the prediction.sha256 beside it before any decode")
    return p.parse_args(argv)


def parse_attain(argv):
    p = argparse.ArgumentParser(description="attainability of the oracle6 reading (task ORI6; oracle6_attain)")
    p.add_argument("--oracle6-attain", required=True, metavar="DECOMPOSE_DIR",
                   help="a decompose folder of the default run (decompose.json, windows.csv)")
    p.add_argument("--out", required=True, help="folder of attain.json")
    p.add_argument("--root", default=None, help="G1 input root (default mdfm.gates.g1.default_root)")
    p.add_argument("--oracle6-levels", default=ORACLE6_LEVELS_DIR)
    return p.parse_args(argv)


def main_attain(argv):
    """--oracle6-attain: attain.json in --out (oracle6_attain), on the CPU, no model."""
    a = parse_attain(argv)
    from mdfm import paths
    from mdfm.data import amass_index
    from mdfm.train import provenance as PV
    t0 = time.time()
    out = os.path.abspath(a.out)
    paths.assert_not_moveatlas(out)
    dset = g1.G1Set(SPLIT, a.root)
    if set(dset.persons) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person in the dev inputs")
    levels = oracle6_levels(a.oracle6_levels)
    res = oracle6_attain(a.oracle6_attain, levels, dset, log=lambda *m: print(*m, flush=True))
    res["dev_set"] = {"manifest_sha256": dset.manifest["manifest_sha256"],
                      "persons_sha256": dset.manifest["persons_sha256"]}
    res["provenance"] = {"script": "scripts/probe_g1_decompose.py", "argv": list(argv),
                         "code_sha256": common.code_sha256([os.path.abspath(__file__)]),
                         "moveatlas_head": PV.moveatlas_head(), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "seconds": round(time.time() - t0, 1)}
    os.makedirs(out, exist_ok=True)
    common.write_text(os.path.join(out, "attain.json"), json.dumps(res, indent=1) + "\n")
    print(f"attain: needed gain {res['needed_gain_mm']:.2f} mm; exact estimates "
          + ", ".join(f"{k} {v:.2f}" for k, v in res["six"]["exact_estimate_mm"].items())
          + f" mm; expected readings {res['expected_reading']}; written {common.rel(out)}", flush=True)
    return res


def subset(gset, k):
    """Window indices of the first k takes of each archive (eval_g1_dev.subset); all when k is None."""
    w = gset.windows
    if k is None:
        return list(range(len(w)))
    keep = set()
    for _, g in w.groupby("archive", sort=True):
        keep |= set(list(dict.fromkeys(g["take_id"].tolist()))[:int(k)])
    return [int(i) for i in range(len(w)) if w["take_id"].iat[i] in keep]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--oracle6-attain" in argv:
        return main_attain(argv)
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    o6_levels = oracle6_levels(a.oracle6_levels) if a.oracle6 else None
    o6_pred = oracle6_prediction_check(a.oracle6_prediction) if (a.oracle6 and a.oracle6_prediction) else None
    from mdfm import paths
    from mdfm.data import amass_index
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    code_files = [g1.__file__, common.__file__, os.path.join(TREE, "mdfm", "eval", "conformal.py"), I.__file__,
                  os.path.abspath(__file__)]
    code_sha = common.code_sha256(code_files)
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    out = a.out or os.path.join(TREE, "reports", "bench", f"g1_decompose_{loaded.name}")
    paths.assert_not_moveatlas(out)
    j1_path = a.j1 or os.path.join(TREE, "reports", "gates", "g1", f"dev_{loaded.name}", "g1.json")
    log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}")

    dset = g1.G1Set(SPLIT, a.root)
    calib = None if a.skip_acc_only else g1.G1Set("calib", a.root)
    train = set(loaded.persons.get("amass", ()))
    for gs in [x for x in (dset, calib) if x is not None]:
        both = sorted(set(gs.persons) & train)
        if both:
            raise Refused(f"persons of the {gs.split} set among the checkpoint's training persons: {both[:3]}")
        if set(gs.persons) & set(amass_index.LOCKBOX_PERSONS):
            raise Refused(f"a lockbox person in the {gs.split} inputs")
    kw = dict(batch_windows=a.batch_windows, block_windows=a.block_windows, seed=a.seed, pool=not a.no_pool)
    idx = subset(dset, a.limit_takes)
    timing = {"load": round(time.time() - t_start, 1)}

    # full pass: harness, substitutions, refine0, best of 16, floor
    t0 = time.time()
    truth = TruthBodies(dset)
    table, blocks = main_pass(loaded, dset, idx, truth, a.refine_steps, a.points, best_of=a.best_of, log=log,
                              oracle6=o6_levels, **kw)
    timing["main_pass"] = round(time.time() - t0, 1)
    log(f"main pass: G1a {MM * pwm(table, 'g1a_m'):.2f} mm on {len(table)} windows ({timing['main_pass']} s)")

    # partial rows: 60 steps from the encoder, analysis by synthesis
    t0 = time.time()
    in_run = set(idx)
    pidx = [i for i in subset(dset, a.partial_takes) if i in in_run]
    p20, b20 = mean_pass(loaded, dset, pidx, a.refine_steps, "encoder", a.points, name=PARTIAL_REF, log=log, **kw)
    p60, b60 = mean_pass(loaded, dset, pidx, a.refine_long, "encoder", a.points, name="refine60_m", log=log, **kw)
    pab, bab = mean_pass(loaded, dset, pidx, a.abs_steps, "prior", a.points, name="abs_m", log=log, **kw)
    part = table[table["window"].isin(pidx)][["window", "take_id", "archive", "person_id", "task", "n", "g1a_m"]]
    part = part.merge(p20[["window", PARTIAL_REF]], on="window").merge(
        p60[["window", "refine60_m"]], on="window").merge(pab[["window", "abs_m"]], on="window")
    timing["partial"] = round(time.time() - t0, 1)
    plabel = f"first {a.partial_takes} takes of each dev archive ({len(pidx)} windows)"

    # acc-only wrist
    acc = {"reason": "skipped (--skip-acc-only)"}
    if not a.skip_acc_only:
        t0 = time.time()
        cidx = subset(calib, a.acc_only_limit_takes)
        didx = [i for i in subset(dset, a.acc_only_limit_takes) if i in in_run]
        fit = acc_only_wrist(loaded, calib, cidx, a.refine_steps, a.points, log=log, **kw)
        accw = acc_only_wrist(loaded, dset, didx, a.refine_steps, a.points, c=fit["c"], log=log, **kw)
        lo, hi = boot(accw, "coverage")
        acc = {"conformal": fit, "coverage": pwm(accw, "coverage"), "coverage_interval": [lo, hi],
               "width_mm": MM * pwm(accw, "width_m"), "energy_mm": MM * pwm(accw, "energy_m"),
               "degenerate": int(accw["degenerate"].sum()), "n_knots": int(accw["n"].sum()),
               "n_windows": int(len(accw)), "calib_windows": len(cidx),
               "partial": None if a.acc_only_limit_takes is None else f"first {a.acc_only_limit_takes} takes",
               "acc_gyro_j1": acc_reference(accw, os.path.join(os.path.dirname(j1_path), "windows.csv"))}
        timing["acc_only"] = round(time.time() - t0, 1)

    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    with open(g1.RULE_PATH, "rb") as f:
        rule_sha = common.sha256_bytes(f.read())
    with open(g1.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())[0]
    timing["all"] = round(time.time() - t_start, 1)
    inference = {"refine_steps": a.refine_steps, "points_per_stream": a.points, "best_of": a.best_of,
                 "refine_long": a.refine_long, "abs_steps": a.abs_steps, "partial_takes": a.partial_takes,
                 "batch_windows": a.batch_windows, "block_windows": a.block_windows, "seed": a.seed,
                 "shape_pooling": "none" if a.no_pool else "per take", "weights": a.weights,
                 "device": str(dev), "amp_bf16": dev.type == "cuda", "memory": mem}
    meta = {"checkpoint": loaded.describe(), "rule_sha256": rule_sha, "rule_status": status,
            "partial_label": plabel, "limit_takes": a.limit_takes,
            "sets": {"dev": {"windows": len(table), "knots": int(table["n"].sum()),
                             "persons": int(table["person_id"].nunique()), "takes": int(table["take_id"].nunique()),
                             "manifest_sha256": dset.manifest["manifest_sha256"],
                             "persons_sha256": dset.manifest["persons_sha256"]},
                     "calib": None if calib is None else {"manifest_sha256": calib.manifest["manifest_sha256"]},
                     "config_sha256": dset.manifest["config_sha256"]},
            "hashes": {"checkpoint_sha256": loaded.sha256, "run_code_sha256": code_sha,
                       "run_code_files": [common.rel(f) for f in code_files],
                       "scoring_code_sha256": g1.code_sha256(g1.SCORE_CODE),
                       "params_sha256": common.sha256_json(g1.PARAMS)},
            "j1": j1_check(j1_path, table, loaded.sha256, inference),
            "truth_check": {"takes": truth.n_takes, "windows": truth.n_windows,
                            "max_deviation_mm": MM * truth.max_dev_m, "tolerance_mm": MM * TRUTH_TOL_M},
            "inference": inference,
            "moveatlas_head": PV.moveatlas_head(), "timing_s": timing,
            "blocks": {"main": blocks, "refine20_partial": b20, "refine60": b60, "abs": bab}}
    if a.oracle6:
        meta["oracle6"] = {"levels": o6_levels, "seed": ORACLE6_SEED, "reference_sensor": ORACLE6_REF,
                           "smooth_s": ORACLE6_SMOOTH_S, "prediction": o6_pred}
    res = assemble(table, part, acc, meta)
    common.write(res, out, "decompose", markdown(res))
    common.write_csv(table, os.path.join(out, "windows.csv"))
    prov = {"script": "scripts/probe_g1_decompose.py", "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_head": meta["moveatlas_head"], "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "checkpoint": meta["checkpoint"]["path"],
            "checkpoint_sha256": loaded.sha256, "code_sha256": code_sha, "argv": sys.argv[1:]}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    for k in ("g1a_mm", "true_root_orient_mm", "true_betas_mm", "refine0_mm", "refine20_mm", "refine60_mm", "abs_mm",
              "best_of_16_mm", "floor_mm", "lockbox_mix_mm"):
        log(f"{k}: {fmt(res[k], 2)}")
    if res.get("oracle6"):
        o6 = res["oracle6"]
        log(f"oracle6: exact {fmt(o6['exact_mm'], 2)}, noisy {fmt(o6['noisy_mm'], 2)} (upper "
            f"{fmt(o6['noisy_upper_mm'], 2)}) mm: {o6['reading']}")
    log(f"written {common.rel(out)}; {timing['all']:.0f} s")
    return res


if __name__ == "__main__":
    main()
