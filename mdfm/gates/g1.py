"""G1, the kernel round trip on held-out AMASS persons (design section 9; spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 6): inputs and truth, the scoring of the three lines and
the population-prior trivial baseline.

Rule: docs/rules/G1_RULE.md, a DRAFT until the owner approves and registers it (common.register_rule).
PARAMS mirrors the parameter table of the rule file (a test compares the two).

Sets (mdfm.data.amass_index): dev and calib archives for development and conformal calibration, the
lockbox archives for the gate read. This module is the only one that passes allow_lockbox=True (to
amass_index.takes, amass.load_many and G1Set), and only after common.rule has returned the registered
G1 rule; building lockbox inputs needs that rule, scoring the lockbox also a logged read
(common.log_lockbox_read) before any array opens. Any other split is refused, and so is a CODY name.

Truth, per take: amass.load_many (cached npz), prep.amass_to_body_clock (RATE, CUTOFF_HZ zero-phase
low-pass, y up, the take's gendered SMPLHBody), forward kinematics of the 22 body joints with the take's
betas (float64, CPU), the floor moved to y = 0 (frames.floor_height), the belt frame for a take that
frames.treadmill_stats flags (joint velocities by central differences). Windows: consecutive,
non-overlapping, WIN_S long from t = 0; the last is kept when it holds at least MIN_KNOTS knots; a take
with fewer knots is counted and dropped. Each window in its own window frame (frames.heading_frame at
knot n // 2: floor y = 0, pelvis heading +z and pelvis x, z at 0 there). Stored per knot: joints (22, 3)
m and the root yaw (rad, the pelvis heading in the window frame), float32.

Inputs, per window, from one torch CPU generator seeded by window_seed (SHA-256 of the take id, the
window index, the input configuration hash and SEED_TAG); draws in this order: mount jitter (6 x 3), acc
biases (6 x 3), gyro biases (6 x 3), acc noise (n x 6 x 3), gyro noise (n x 6 x 3), then
camera_render.sample_camera, then the tracker process of camera_render.render_pose2d.
- IMU, six sites SITES (G1a) whose WRIST_SITE column is the single-wrist input (G1b): the reference
  renderer factory.imu_render.render on the take's trajectory (float64 on the CPU or the GPU, whose
  float32 outputs agree; identity mounts: segment frame),
  turned per window by M = R0[mount family, site] Exp(theta), theta ~ N(0, MOUNT_JITTER_DEG^2 I_3) (an
  IMU signal does not depend on the world frame, so rendering the whole take equals rendering each
  window); biases and white noise per site and axis at the levels of the device (xsens13: the per-site
  levels of reports/mount_priors/priors.json; generic: imu_render.Device); clipped at the device ranges;
  magnetometer masked; then data.window_store.band_copy of the window. Stored as imu (n, 6, 6) float32,
  acc in g and gyro in deg/s, and imu_valid (n, 6).
- camera (G1c): camera_render.sample_camera (family generic, CAMERA_FPS) aimed at the window-mean
  pelvis; the 15 canonical joints at k / fps (Hermite of the take's knots, then the window frame);
  render_pose2d with the default TrackerProcess and the canonical 2D normalization. Stored as cam (n_c,
  15, 2) float32 torso units and cam_valid (n_c, 15); per window the scale (px per torso unit), the fps
  and the camera (K, R, t).

Scoring, on every knot of a window (100 Hz); windows averaged per person with their frame counts, the
statistic the mean over persons, intervals from the person-clustered bootstrap (common):
- G1a: MPJPE (m) over the 22 joints, pelvis-relative, after one yaw per window about +y fitted in closed
  form over all scored joints and frames (yaw_fit).
- G1b: the left wrist (SMPL joint WRIST_JOINT) relative to the pelvis in the pelvis-heading frame of the
  same body at the same knot (heading_points), the truth in its own frame and each draw in its own;
  coverage of the chi-square 3-dof ellipsoid of the moment-matched predictive scaled by the conformal
  factor fitted on calib (eval.conformal; a degenerate predictive is counted and never covers); band
  width; energy score.
- G1c: PA-MPJPE (m) on the 15 canonical joints, similarity Procrustes per frame (procrustes_errors).
A predictor gives, per window, joints_six and joints_camera ((n, 22, 3) or a constant (22, 3), any
frame), wrist ((n, m, 3) draws or one shared (m, 3) set, pelvis-heading frame) and wrist_six (the same
for the six-IMU predictive, or None); conformal holds its factors. PopulationPrior (from the train body
store only) predicts a constant mean pose and one shared set of PRIOR_DRAWS wrist positions.
"""
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import paths
from ..data.canonical import Refused, is_cody
from ..eval import conformal
from . import common

GATE = "G1"
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G1_RULE.md")
# the parameter table of docs/rules/G1_RULE.md (a test compares them)
PARAMS = {
    "version": "g1_v1",
    "seed_tag": "G1v1",
    "rate_hz": 100.0,
    "cutoff_hz": 12.0,
    "window_s": 10.0,
    "min_window_s": 2.0,
    "sites_six": ["head", "lower_back", "wrist_L", "wrist_R", "shank_L", "shank_R"],
    "site_single": "wrist_L",
    "imu_channels": ["acc", "gyro"],
    "mount_family": "xsens13",
    "mount_jitter_deg": 5.0,
    "device": "xsens13",
    "acc_range_g": 16.0,
    "gyro_range_dps": 2000.0,
    "camera_family": "generic",
    "camera_fps": 30.0,
    "tracker": "default",
    "g1a_joints": 22,
    "g1c_joints": 15,
    "wrist_joint": 20,
    "g1a_pass_mm": 45.0,
    "g1a_kill_mm": 70.0,
    "g1b_coverage_pass": 0.85,
    "g1b_coverage_kill": 0.70,
    "g1b_width_ratio_max": 1.5,
    "g1b_energy_ratio_max": 0.8,
    "g1c_pass_mm": 60.0,
    "g1c_kill_mm": 90.0,
    "conformal_level": 0.9,
    "ellipsoid_dof": 3,
    "draws_ddof": 1,
    "predictive_draws": 64,
    "latent_samples": 16,
    "decoder_draws": 4,
    "refine_steps": 20,
    "refine_points": 256,
    "bootstrap_n": 2000,
    "bootstrap_level": 0.95,
    "bootstrap_seed": 20260929,
    "prior_draw_seed": 1,
    "prior_mean_stride": 10,
}
VERSION = PARAMS["version"]
SEED_TAG = PARAMS["seed_tag"]
RATE = PARAMS["rate_hz"]
CUTOFF_HZ = PARAMS["cutoff_hz"]
WIN_S = PARAMS["window_s"]
KNOTS = int(round(WIN_S * RATE))
MIN_KNOTS = int(round(PARAMS["min_window_s"] * RATE))
SITES = tuple(PARAMS["sites_six"])
WRIST_SITE = PARAMS["site_single"]
WRIST_COL = SITES.index(WRIST_SITE)
MOUNT_JITTER_DEG = PARAMS["mount_jitter_deg"]
CAMERA_FPS = PARAMS["camera_fps"]
WRIST_JOINT = PARAMS["wrist_joint"]
PRIOR_DRAWS = PARAMS["predictive_draws"]
PELVIS = 0
SPLITS = ("dev", "calib", "lockbox")
N_CH = 6                                   # acc x, y, z (g), gyro x, y, z (deg/s)
# files of a shard: name -> (dtype, trailing shape, per) with per 'knot', 'cam' or 'window'
FILES = {"joints": ("<f4", (22, 3), "knot"), "root_yaw": ("<f4", (), "knot"),
         "imu": ("<f4", (len(SITES), N_CH), "knot"), "imu_valid": ("u1", (len(SITES),), "knot"),
         "cam": ("<f4", (15, 2), "cam"), "cam_valid": ("u1", (15,), "cam"),
         "cam_params": ("<f8", (21,), "window"), "mounts": ("<f4", (len(SITES), 3, 3), "window"),
         "biases": ("<f4", (len(SITES), N_CH), "window")}
TRUTH_FILES = ("joints", "root_yaw")
WINDOW_COLUMNS = ("window", "take_id", "archive", "subject_ns", "person_id", "gender", "task", "k", "start", "n",
                  "offset", "cam_offset", "cam_n", "cam_fps", "cam_scale", "seed")
TAKE_COLUMNS = ("take_id", "archive", "subject_ns", "person_id", "gender", "fps_native", "fps_manifest",
                "n_frames", "n_knots", "n_windows", "dropped", "treadmill", "belt_x", "belt_z", "floor_m",
                "betas")
# code the inputs depend on (their hash goes into every shard)
INPUT_CODE = ("mdfm/gates/g1.py", "mdfm/data/amass.py", "mdfm/data/amass_index.py", "mdfm/data/prep.py",
              "mdfm/data/window_store.py", "mdfm/body/frames.py", "mdfm/body/hermite.py", "mdfm/body/rotations.py",
              "mdfm/body/smplh.py", "mdfm/body/trajectory.py", "mdfm/factory/imu_render.py",
              "mdfm/factory/camera_render.py", "mdfm/kernels/camera.py", "mdfm/kernels/pose.py",
              "mdfm/kernels/nuisance.py")
SCORE_CODE = ("mdfm/gates/g1.py", "mdfm/gates/common.py", "mdfm/eval/conformal.py")


def _canon15():
    from ..kernels.camera import SMPL_TO_CANONICAL_15
    return list(SMPL_TO_CANONICAL_15)


CANON15 = _canon15()


def code_sha256(files):
    return common.code_sha256([os.path.join(paths.TREE, f) for f in files])


def input_code_sha256():
    """code_sha256(INPUT_CODE) after importing every module of INPUT_CODE, so that the hash describes the
    code this process runs (a module is read once, at its first import); build checks it again at its
    end."""
    import importlib
    for f in INPUT_CODE:
        importlib.import_module(f[:-3].replace("/", "."))
    return code_sha256(INPUT_CODE)


# ---------------------------------------------------------------- configuration and seeds
def input_config():
    """The input configuration (dict) whose hash seeds every window: the input parameters, the device
    levels and mount priors (their file hash), the camera family, the tracker and the band copy."""
    from dataclasses import asdict
    from ..data import window_store
    from ..factory.camera_render import FAMILIES, TrackerProcess
    from ..kernels.nuisance import PRIORS_PATH
    keys = ("version", "seed_tag", "rate_hz", "cutoff_hz", "window_s", "min_window_s", "sites_six", "site_single",
            "imu_channels", "mount_family", "mount_jitter_deg", "device", "acc_range_g", "gyro_range_dps",
            "camera_family", "camera_fps", "tracker")
    cfg = {k: PARAMS[k] for k in keys}
    cfg["mount_priors_sha256"] = common.sha256_file(PRIORS_PATH)
    cfg["camera"] = asdict(FAMILIES[PARAMS["camera_family"]])
    cfg["tracker_process"] = asdict(TrackerProcess())
    cfg["band"] = dict(window_store.BAND)
    return cfg


def window_seed(take_id, k, config_sha256):
    """Seed of window k of a take: the first 63 bits of SHA-256(take_id, k, config hash, SEED_TAG)."""
    h = common.sha256_bytes(f"{take_id}\n{int(k)}\n{config_sha256}\n{SEED_TAG}".encode("utf-8"))
    return int(h[:16], 16) & ((1 << 63) - 1)


def window_starts(n_knots):
    """(start, n) of the windows of a take of n_knots knots: from 0 every KNOTS knots, the last kept when
    it holds at least MIN_KNOTS knots; [] for a take under MIN_KNOTS knots."""
    out = []
    for s in range(0, int(n_knots), KNOTS):
        n = min(KNOTS, int(n_knots) - s)
        if n >= MIN_KNOTS:
            out.append((s, n))
    return out


def device_levels(priors, device=PARAMS["device"]):
    """Per-site levels (S, 3) float64 of SITES: bias_acc_sd, noise_acc (g), bias_gyro_sd, noise_gyro
    (deg/s); xsens13 from the mount priors, generic from factory.imu_render.Device."""
    if device == "xsens13":
        f = "xsens13"
        get = lambda x: priors.per_site(x, f, list(SITES)).double()  # noqa: E731
        return {"bias_acc": get(priors.bias_acc_sd), "noise_acc": get(priors.noise_acc),
                "bias_gyro": get(priors.bias_gyro_sd), "noise_gyro": get(priors.noise_gyro)}
    if device == "generic":
        from ..factory.imu_render import Device
        d = Device()
        full = lambda v: torch.full((len(SITES), 3), float(v), dtype=torch.float64)  # noqa: E731
        return {"bias_acc": full(d.acc_bias_g), "noise_acc": full(d.acc_noise_g), "bias_gyro": full(d.gyro_bias_dps),
                "noise_gyro": full(d.gyro_noise_dps)}
    raise ValueError(f"unknown device {device!r}")


# ---------------------------------------------------------------- roots and guards
def default_root(split):
    """Root of the G1 inputs of a split: the gates-only store for the lockbox, cache/g1/ otherwise."""
    if split == "lockbox":
        return paths.gates_store_dir(VERSION).replace("\\", "/")
    return os.path.join(paths.CACHE, "g1", VERSION).replace("\\", "/")


def _check_root(split, root):
    paths.assert_not_moveatlas(root)
    if split == "lockbox":
        if not paths.in_gates_store(root):
            raise Refused(f"lockbox inputs live in the gates-only store only, not {root}")
    elif paths.in_gates_store(root):
        raise Refused(f"{split} inputs never go to the gates-only store: {root}")
    return root


def _check_split(split):
    if split not in SPLITS:
        raise Refused(f"G1 reads the dev, calib and lockbox archives only, not {split!r}")
    return split


def _rows(split, take_ids=None, archives=None, limit=None):
    """Index rows of a split (the lockbox needs the registered G1 rule), restricted to take ids, archives
    and the first `limit` takes of each archive (index order)."""
    from ..data import amass_index
    _check_split(split)
    if split == "lockbox":
        common.rule(RULE_PATH, GATE)                  # Refused until the owner registers the G1 rule
        rows = amass_index.takes("lockbox", allow_lockbox=True)
        persons = set(rows["person_id"])
        if persons != set(amass_index.LOCKBOX_PERSONS):
            raise ValueError(f"lockbox rows hold {len(persons)} persons, the lockbox file "
                             f"{len(amass_index.LOCKBOX_PERSONS)}")
    else:
        rows = amass_index.takes(split)
    names = rows[["archive", "take_id", "subject_ns", "person_id"]].to_numpy().ravel().tolist()
    if is_cody(*names):
        raise Refused("a CODY name in the AMASS index: CODY is evaluation-only and never a G1 set")
    if archives is not None:
        for a in archives:
            if is_cody(a):
                raise Refused(f"CODY is evaluation-only and never a G1 archive: {a!r}")
            amass_index.check(a, split == "lockbox")
            if amass_index.archive_split(a) != split:
                raise Refused(f"{a} is not a {split} archive")
        rows = rows[rows["archive"].isin(list(archives))]
    if take_ids is not None:
        take_ids = list(take_ids)
        if is_cody(*take_ids):
            raise Refused("CODY is evaluation-only and never a G1 take")
        missing = sorted(set(take_ids) - set(rows["take_id"]))
        if missing:
            raise Refused(f"takes outside the {split} split: {missing[:3]}")
        rows = rows[rows["take_id"].isin(take_ids)]
    if limit is not None:
        rows = rows.groupby("archive", sort=False).head(int(limit))
    return rows.reset_index(drop=True)


# ---------------------------------------------------------------- truth
def _decimals(x):
    r = repr(float(x))
    return len(r.split(".")[1]) if "." in r and "e" not in r else 17


def take_body(seq, row, body):
    """One loaded AMASS sequence and its index row -> dict of the take on the body clock in the lab frame
    (floor y = 0, belt frame when treadmill): r6 (T, 22, 6), tr (T, 3) m, betas (16,), rest (22, 3) m,
    pos (T, 22, 3) m, R (T, 22, 3, 3) global rotations, float64; fps, floor_m, treadmill, belt (3,) m/s.
    None when the take holds fewer than MIN_KNOTS knots. Raises ValueError when the npz disagrees with
    its index row (gender, frames, fps at the manifest's printed precision when the manifest has one)."""
    from ..body import frames
    from ..body.rotations import rot6d_to_matrix
    from ..body.joints import fk
    from ..data.prep import amass_to_body_clock
    g = str(seq["gender"]).strip().lower()
    if g != str(row["gender"]).strip().lower():
        raise ValueError(f"{row['take_id']}: gender {g!r} in the npz, {row['gender']!r} in the index")
    fps, n_native = float(seq["fps"]), int(seq["poses"].shape[0])
    if n_native != int(row["n_frames"]):
        raise ValueError(f"{row['take_id']}: {n_native} frames in the npz, {row['n_frames']} in the manifest")
    want = float(row["fps"])
    if math.isfinite(want) and round(fps, _decimals(want)) != want:
        raise ValueError(f"{row['take_id']}: fps {fps!r} in the npz, {want!r} in the manifest")
    if (n_native - 1) / fps * RATE + 1 < MIN_KNOTS:
        return None
    r6, tr, betas = amass_to_body_clock(seq, rate=RATE, cutoff=CUTOFF_HZ, y_up=True, dtype=torch.float64, body=body)
    if r6.shape[0] < MIN_KNOTS:
        return None
    rest = body.rest_joints22(betas.to(body.J_template.dtype)).to(torch.float64)
    R_local = rot6d_to_matrix(r6)
    pos, R = fk(R_local, tr, rest)
    floor = float(frames.floor_height(pos))
    up = torch.tensor([0.0, floor, 0.0], dtype=torch.float64)
    pos, tr = pos - up, tr - up
    st = frames.treadmill_stats(pos, RATE, floor=0.0)
    belt = st["belt"].double() if st["treadmill"] else torch.zeros(3, dtype=torch.float64)
    if st["treadmill"]:
        pos, tr = frames.galilean_shift(pos, belt, RATE), frames.galilean_shift(tr, belt, RATE)
    return {"r6": r6, "tr": tr, "betas": betas, "rest": rest, "pos": pos, "R": R, "fps": fps, "floor_m": floor,
            "treadmill": bool(st["treadmill"]), "belt": belt}


def window_frame(pos, R, start, n):
    """Window [start, start + n) of a take in its window frame: joints (n, 22, 3) m, root yaw (n,) rad,
    R_yaw (3, 3), origin (3,) (frames.heading_frame at knot n // 2)."""
    from ..body import frames
    from ..body.rotations import yaw_angle
    p = pos[start:start + n]
    Rr = R[start:start + n, 0]
    R_yaw, origin = frames.heading_frame(p[None], Rr[None], n // 2)
    joints = frames.apply_frame(p[None], R_yaw, origin)[0]
    yaw = yaw_angle(R_yaw[0] @ Rr)
    return joints, yaw, R_yaw[0], origin[0]


def heading_points(x, pelvis, root_yaw):
    """Points x (..., 3) relative to the pelvis (..., 3) in the pelvis-heading frame: rot_y(-root_yaw)
    (x - pelvis), root_yaw (...) rad; numpy float64."""
    v = np.asarray(x, dtype=np.float64) - np.asarray(pelvis, dtype=np.float64)
    th = np.asarray(root_yaw, dtype=np.float64)
    c, s = np.cos(th), -np.sin(th)
    return np.stack([c * v[..., 0] + s * v[..., 2], v[..., 1], -s * v[..., 0] + c * v[..., 2]], -1)


# ---------------------------------------------------------------- inputs
RENDER_CHUNK = 20000                       # timestamps per reference-renderer call (exact: every call sees all knots)


class _Renderer:
    """Mount priors, device levels and the input configuration, loaded once per build. device: where the
    reference IMU renderer runs, in float64 (cpu or cuda; the float32 signals it returns are equal)."""

    def __init__(self, device="cpu"):
        from ..kernels.nuisance import MountPriors
        self.priors = MountPriors.load()
        self.R0 = self.priors.rotation(PARAMS["mount_family"], list(SITES)).double()
        self.levels = device_levels(self.priors)
        self.config = input_config()
        self.config_sha256 = common.sha256_json(self.config)
        self.device = torch.device(device)
        self._bodies = {}

    def _body(self, body):
        if self.device.type == "cpu":
            return body
        if body.gender not in self._bodies:
            from ..body.smplh import SMPLHBody
            self._bodies[body.gender] = SMPLHBody(body.gender, dtype=torch.float64).to(self.device)
        return self._bodies[body.gender]

    def segment_signals(self, body, take):
        """Clean acc (g) and gyro (deg/s) of SITES in the segment frames (identity mounts), (T, 6, 6)
        float64 on the CPU, from the reference renderer on the whole take; and the take's trajectory on the
        CPU (camera times)."""
        from ..body.trajectory import BodyTrajectory
        from ..factory import imu_render
        d = self.device
        traj = BodyTrajectory(self._body(body), take["r6"].to(d), take["tr"].to(d), take["betas"].to(d), rate=RATE,
                              rest=take["rest"].to(d))
        T = take["r6"].shape[0]
        idx = [imu_render.SITES.index(s) for s in SITES]
        parts = []
        for a in range(0, T, RENDER_CHUNK):
            # times computed on the CPU: a CUDA division by a scalar multiplies by its reciprocal, and k / 100
            # rounding to either side of knot k picks the left or the right Hermite segment, whose second
            # derivatives differ (Catmull-Rom is C1 only)
            t = (torch.arange(a, min(T, a + RENDER_CHUNK), dtype=torch.float64) / RATE).to(d)
            sig, _, _ = imu_render.render(traj, t, list(SITES))
            parts.append(sig[:, idx, :N_CH].double())
        cpu = BodyTrajectory(body, take["r6"], take["tr"], take["betas"], rate=RATE, rest=take["rest"])
        return torch.cat(parts), cpu

    def window_imu(self, seg, start, n, gen):
        """Mounts (6, 3, 3), biases (6, 6) and the band copy of the noisy window: values (n, 6, 6) float32,
        valid (n, 6) bool."""
        from ..body.rotations import axis_angle_to_matrix
        from ..data.window_store import band_copy
        from ..data.windows import Stream
        from ..factory import imu_render
        S = len(SITES)
        theta = torch.randn(S, 3, generator=gen, dtype=torch.float64) * math.radians(MOUNT_JITTER_DEG)
        M = self.R0 @ axis_angle_to_matrix(theta)
        b_acc = torch.randn(S, 3, generator=gen, dtype=torch.float64) * self.levels["bias_acc"]
        b_gyr = torch.randn(S, 3, generator=gen, dtype=torch.float64) * self.levels["bias_gyro"]
        n_acc = torch.randn(n, S, 3, generator=gen, dtype=torch.float64) * self.levels["noise_acc"]
        n_gyr = torch.randn(n, S, 3, generator=gen, dtype=torch.float64) * self.levels["noise_gyro"]
        w = seg[start:start + n]
        Mt = M.transpose(-1, -2)
        acc = torch.einsum("sij,tsj->tsi", Mt, w[..., :3]) + b_acc + n_acc
        gyr = torch.einsum("sij,tsj->tsi", Mt, w[..., 3:]) + b_gyr + n_gyr
        acc = acc.clamp(-PARAMS["acc_range_g"], PARAMS["acc_range_g"])
        gyr = gyr.clamp(-PARAMS["gyro_range_dps"], PARAMS["gyro_range_dps"])
        values = np.zeros((n, S, 9), dtype=np.float32)
        values[..., :3], values[..., 3:6] = acc.numpy(), gyr.numpy()
        cm = np.zeros((S, 9), dtype=bool)
        cm[:, :6] = True
        stream = Stream(modality="imu", role="input", fps=RATE, t=np.arange(n, dtype=np.float64) / RATE,
                        values=values, chan_mask=cm, valid=np.ones((n, S), dtype=bool),
                        meta={"site_ids": [imu_render.SITES.index(s) for s in SITES], "acc_content": "with_gravity"})
        out, info = band_copy(stream, return_info=True)
        if info.plan.k != 1 or len(info.idx) != n:
            raise ValueError(f"band copy of a {RATE:g} Hz window kept {len(info.idx)} of {n} samples")
        return M, torch.cat([b_acc, b_gyr], -1), out.values[..., :N_CH].astype(np.float32), out.valid

    def window_camera(self, traj, start, n, R_yaw, origin, pelvis_mean, gen):
        """sample_camera and render_pose2d of one window: (record, camera params (21,) K, R, t)."""
        from ..body import frames
        from ..factory.camera_render import render_pose2d, sample_camera
        cam = sample_camera(gen, pelvis_mean, family=PARAMS["camera_family"], fps=CAMERA_FPS)
        tc = torch.arange(0.0, (n - 1) / RATE + 1e-9, 1.0 / CAMERA_FPS, dtype=torch.float64)
        pos, _ = traj.pose_at(start / RATE + tc)
        pos = frames.apply_frame(pos[None], R_yaw[None], origin[None])[0]
        record, _ = render_pose2d(pos[:, CANON15], cam, gen)
        params = torch.cat([cam.K.reshape(-1), cam.R.reshape(-1), cam.t.reshape(-1)]).numpy()
        return record, params


# ---------------------------------------------------------------- build
class _Shard:
    """Raw little-endian arrays of one (split, archive) folder, written row block by row block."""

    def __init__(self, folder, render):
        self.folder = folder
        self.names = [k for k in FILES if render or k in TRUTH_FILES]
        self.files = {k: open(os.path.join(folder, self.fname(k)), "wb") for k in self.names}
        self.count = {"knot": 0, "cam": 0, "window": 0}

    @staticmethod
    def fname(k):
        """File name of array k: the name and its dtype code (joints.f4, imu_valid.u1, cam_params.f8)."""
        dt = FILES[k][0]
        return k + "." + dt.lstrip("<")

    def write(self, k, arr):
        dt, shape, per = FILES[k]
        a = np.ascontiguousarray(np.asarray(arr), dtype=dt)
        if a.shape[1:] != shape:
            raise ValueError(f"{k}: shape {a.shape}, expected (n, *{shape})")
        self.files[k].write(a.tobytes())

    def close(self):
        for f in self.files.values():
            f.close()


def _write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def build_shard(split, archive, rows, root, render=True, force=False, log=None, device="cpu", code=None):
    """Truth (and inputs when render) of one archive under <root>/<split>/<archive>; returns shard.json.
    A shard built with the same configuration, code and take list is kept unless force. device: where
    the reference IMU renderer runs (float64). code: the input code hash of the build (default
    input_code_sha256())."""
    import shutil
    from ..body.smplh import SMPLHBody
    from ..data import amass
    log = log or (lambda *a: None)
    lock = split == "lockbox"
    rule_sha = common.rule(RULE_PATH, GATE)[1] if lock else ""
    final = os.path.join(_check_root(split, root), split, archive)
    rend = _Renderer(device)
    code = input_code_sha256() if code is None else code
    want = common.sha256_bytes(("\n".join(rows["take_id"]) + "\n").encode("utf-8"))
    meta_path = os.path.join(final, "shard.json")
    if os.path.exists(meta_path) and not force:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("config_sha256") == rend.config_sha256 and meta.get("code_sha256") == code
                and meta.get("requested_takes_sha256") == want and meta.get("render") == bool(render)):
            log(f"{split}/{archive}: kept ({meta['n_windows']} windows)")
            return meta
    t0 = time.time()
    tmp = final + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    from ..data.windows import task_id
    seqs = amass.load_many(archive, list(rows["take_id"]), allow_lockbox=lock)
    bodies = {}
    shard = _Shard(tmp, render)
    windows, takes = [], []
    valid = {"imu": 0, "cam": 0}
    try:
        for _, row in rows.iterrows():
            seq = seqs.pop(row["take_id"])
            g = str(seq["gender"]).strip().lower()
            if g not in bodies:
                bodies[g] = SMPLHBody(g, dtype=torch.float64)
            tk = take_body(seq, row, bodies[g])
            trow = {"take_id": row["take_id"], "archive": archive, "subject_ns": row["subject_ns"],
                    "person_id": row["person_id"], "gender": row["gender"], "fps_native": repr(float(seq["fps"])),
                    "fps_manifest": "" if not math.isfinite(float(row["fps"])) else repr(float(row["fps"])),
                    "n_frames": int(row["n_frames"]), "n_knots": 0, "n_windows": 0, "dropped": "short",
                    "treadmill": 0, "belt_x": "0.0", "belt_z": "0.0", "floor_m": "",
                    "betas": " ".join(repr(float(b)) for b in np.asarray(seq["betas"], dtype=np.float64))}
            if tk is None:
                takes.append(trow)
                continue
            ws = window_starts(tk["pos"].shape[0])
            trow.update({"n_knots": int(tk["pos"].shape[0]), "n_windows": len(ws), "dropped": "",
                         "treadmill": int(tk["treadmill"]), "belt_x": repr(float(tk["belt"][0])),
                         "belt_z": repr(float(tk["belt"][2])), "floor_m": repr(tk["floor_m"])})
            takes.append(trow)
            seg, traj = rend.segment_signals(bodies[g], tk) if render else (None, None)
            task = task_id("amass", row["take_id"])
            for k, (s, n) in enumerate(ws):
                joints, yaw, R_yaw, origin = window_frame(tk["pos"], tk["R"], s, n)
                seed = window_seed(row["take_id"], k, rend.config_sha256)
                wrow = {"window": shard.count["window"], "take_id": row["take_id"], "archive": archive,
                        "subject_ns": row["subject_ns"], "person_id": row["person_id"], "gender": row["gender"],
                        "task": task, "k": k, "start": s, "n": n, "offset": shard.count["knot"],
                        "cam_offset": shard.count["cam"], "cam_n": 0, "cam_fps": "", "cam_scale": "",
                        "seed": str(seed)}
                shard.write("joints", joints.float().numpy())
                shard.write("root_yaw", yaw.float().numpy())
                if render:
                    gen = torch.Generator().manual_seed(seed)
                    M, bias, imu, ok = rend.window_imu(seg, s, n, gen)
                    record, cam = rend.window_camera(traj, s, n, R_yaw, origin, joints[:, PELVIS].mean(0), gen)
                    shard.write("imu", imu)
                    shard.write("imu_valid", ok.astype(np.uint8))
                    valid["imu"] += int(ok.sum())
                    valid["cam"] += int(record["valid_mask"].sum())
                    shard.write("mounts", M.float().numpy()[None])
                    shard.write("biases", bias.float().numpy()[None])
                    shard.write("cam", record["coords"][..., :2])
                    shard.write("cam_valid", record["valid_mask"].astype(np.uint8))
                    shard.write("cam_params", cam[None])
                    wrow.update({"cam_n": int(record["n_frames"]), "cam_fps": repr(float(record["fps"])),
                                 "cam_scale": repr(float(record["scale"]))})
                    shard.count["cam"] += int(record["n_frames"])
                shard.count["knot"] += n
                shard.count["window"] += 1
                windows.append(wrow)
    finally:
        shard.close()
    _write_csv(os.path.join(tmp, "windows.csv"), WINDOW_COLUMNS, windows)
    _write_csv(os.path.join(tmp, "takes.csv"), TAKE_COLUMNS, takes)
    files = {}
    for k in shard.names:
        fn = _Shard.fname(k)
        dt, shape, per = FILES[k]
        files[k] = {"file": fn, "dtype": dt, "shape": [shard.count[per], *shape],
                    "bytes": os.path.getsize(os.path.join(tmp, fn)), "sha256": common.sha256_file(os.path.join(tmp, fn))}
    for name in ("windows.csv", "takes.csv"):
        files[name] = {"file": name, "sha256": common.sha256_file(os.path.join(tmp, name))}
    persons = sorted({w["person_id"] for w in windows})
    meta = {"version": VERSION, "gate": GATE, "split": split, "archive": archive, "render": bool(render),
            "n_takes": len(takes), "n_dropped_short": sum(t["dropped"] == "short" for t in takes),
            "n_windows": shard.count["window"], "n_knots": shard.count["knot"], "n_cam": shard.count["cam"],
            "n_treadmill_takes": sum(int(t["treadmill"]) for t in takes),
            "imu_valid": valid["imu"], "cam_valid": valid["cam"], "n_persons": len(persons), "persons_sha256": common.sha256_bytes(("\n".join(persons) + "\n").encode("utf-8")),
            "config": rend.config, "config_sha256": rend.config_sha256, "params_sha256": common.sha256_json(PARAMS),
            "code_sha256": code, "rule_sha256": rule_sha, "requested_takes_sha256": want, "files": files,
            "render_device": str(rend.device), "seconds": round(time.time() - t0, 1),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(tmp, "shard.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    log(f"{split}/{archive}: {meta['n_windows']} windows, {meta['n_knots']} knots, {meta['n_dropped_short']} short "
        f"takes, {meta['seconds']:.0f} s")
    return meta


def build(split, root=None, take_ids=None, archives=None, limit=None, render=True, force=False, log=None,
          device="cpu"):
    """Truth and inputs of a split under root (default default_root(split)), one shard per archive, then
    <root>/<split>/manifest.json. The lockbox needs the registered G1 rule (Refused before any row is
    listed otherwise). device: where the reference IMU renderer runs (float64). The input code hash is
    taken once, before the first shard, and checked again at the end: a file of INPUT_CODE edited during
    the build raises ValueError before the manifest is written (rerun the build). Returns the manifest."""
    _check_split(split)
    if split == "lockbox":
        common.rule(RULE_PATH, GATE)
    root = _check_root(split, default_root(split) if root is None else root)
    rows = _rows(split, take_ids, archives, limit)
    code = input_code_sha256()
    shards = {a: build_shard(split, a, g.reset_index(drop=True), root, render, force, log, device, code)
              for a, g in rows.groupby("archive", sort=True)}
    now = code_sha256(INPUT_CODE)
    if now != code:
        raise ValueError(f"the input code changed during the build ({code[:12]} at the start, {now[:12]} now): "
                         "rerun the build")
    return write_manifest(split, root, sorted(shards))


def write_manifest(split, root, archives):
    """<root>/<split>/manifest.json over the shard.json files of `archives`: their SHA-256, the
    configuration and input-code hashes and the manifest SHA-256 (over the shard hashes, the configuration
    and the code). Shards built under another configuration or another input code (a file of INPUT_CODE
    edited during a build) raise ValueError: rerun the build, which rebuilds the stale shards."""
    base = os.path.join(root, split)
    shards, config, code, persons = {}, None, None, set()
    for a in archives:
        p = os.path.join(base, a, "shard.json")
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if config is not None and m["config_sha256"] != config:
            raise ValueError(f"{split}/{a}: configuration {m['config_sha256'][:12]} differs from {config[:12]}")
        if code is not None and m["code_sha256"] != code:
            raise ValueError(f"{split}/{a}: input code {m['code_sha256'][:12]} differs from {code[:12]} (a file of "
                             "g1.INPUT_CODE changed during the build): rerun the build")
        config, code = m["config_sha256"], m["code_sha256"]
        with open(os.path.join(base, a, "windows.csv"), encoding="utf-8", newline="") as f:
            persons |= {r["person_id"] for r in csv.DictReader(f)}
        shards[a] = {"shard_sha256": common.sha256_file(p), "n_windows": m["n_windows"], "n_knots": m["n_knots"],
                     "render": m["render"], "persons_sha256": m["persons_sha256"], "code_sha256": m["code_sha256"]}
    ids = sorted(persons)
    body = {"version": VERSION, "gate": GATE, "split": split, "config_sha256": config, "code_sha256": code,
            "n_persons": len(ids), "persons_sha256": common.sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")),
            "shards": shards}
    body["manifest_sha256"] = common.sha256_json(body)
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(body, f, indent=1)
        f.write("\n")
    return body


# ---------------------------------------------------------------- reading
class G1Set:
    """The G1 truth (and inputs) of one split, memory-mapped. windows: DataFrame of every window (a shard
    column added); arrays: per shard {name: memmap}. The lockbox is opened only by mdfm.gates code with
    allow_lockbox=True, and only after a read of the registered G1 rule is logged. verify (default: True for
    the lockbox only) also checks the manifest's own SHA-256 and the SHA-256 of every array file against
    its shard.json (ValueError on a mismatch), so that the logged input hash describes the bytes scored."""

    def __init__(self, split, root=None, allow_lockbox=False, need_inputs=False, verify=None):
        import pandas as pd
        from ..data.lockbox_flag import gates_override     # the shipped flag check (no AMASS module, plan phase E)
        _check_split(split)
        granted = gates_override(allow_lockbox, sys._getframe(1))
        if split == "lockbox":
            self._lockbox_guard(granted)
        self.split = split
        self.root = _check_root(split, default_root(split) if root is None else root)
        base = os.path.join(self.root, split)
        with open(os.path.join(base, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        verify = (split == "lockbox") if verify is None else bool(verify)
        if verify and common.sha256_json({k: v for k, v in self.manifest.items() if k != "manifest_sha256"}) \
                != self.manifest.get("manifest_sha256"):
            raise ValueError(f"{split}: manifest.json does not match its own SHA-256")
        tables, self.arrays, self.shards = [], [], []
        for s, (a, info) in enumerate(sorted(self.manifest["shards"].items())):
            folder = os.path.join(base, a)
            p = os.path.join(folder, "shard.json")
            if common.sha256_file(p) != info["shard_sha256"]:
                raise ValueError(f"{split}/{a}: shard.json changed since the manifest was written")
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
            if meta["split"] != split or meta["archive"] != a:
                raise Refused(f"{folder}: shard.json names {meta['split']}/{meta['archive']}")
            if need_inputs and not meta["render"]:
                raise ValueError(f"{split}/{a}: truth only, no inputs")
            wpath = os.path.join(folder, "windows.csv")
            df = pd.read_csv(wpath, dtype=str, keep_default_na=False, encoding="utf-8")
            if is_cody(a, *df[["take_id", "subject_ns", "person_id"]].to_numpy().ravel().tolist()):
                raise Refused(f"{folder}: a CODY name in a G1 shard")
            if common.sha256_file(wpath) != meta["files"]["windows.csv"]["sha256"]:
                raise ValueError(f"{split}/{a}: windows.csv changed since the shard was built")
            for c in ("window", "task", "k", "start", "n", "offset", "cam_offset", "cam_n"):
                df[c] = df[c].astype(np.int64)
            arrs = {}
            for k, d in meta["files"].items():
                if k in FILES:
                    fp = os.path.join(folder, d["file"])
                    if verify and common.sha256_file(fp) != d["sha256"]:
                        raise ValueError(f"{split}/{a}: {d['file']} changed since the shard was built")
                    arrs[k] = np.memmap(fp, dtype=d["dtype"], mode="r", shape=tuple(d["shape"]))
            self.arrays.append(arrs)
            self.shards.append(meta)
            tables.append(df.assign(shard=s))
        if not tables:
            raise FileNotFoundError(f"no shard under {base}: run scripts/build_g1_inputs.py")
        self.windows = pd.concat(tables, ignore_index=True)
        self._check_persons()

    def _lockbox_guard(self, granted):
        """Before a lockbox set opens: the flag granted to an mdfm/gates caller, and a logged read of the
        registered G1 rule (the open sets' subclass, g1_open.OpenLockboxSet, checks its own read row instead)."""
        if not granted:
            raise Refused("the G1 lockbox inputs are opened by mdfm/gates only")
        _, rule_sha = common.rule(RULE_PATH, GATE)
        if not any(r["gate"] == GATE and r["rule_sha256"] == rule_sha for r in common.read_log()):
            raise Refused("no lockbox read of the registered G1 rule is logged")

    def _check_persons(self):
        from ..data import amass_index
        idx = amass_index.index()
        split_of = dict(zip(idx["take_id"], idx["split"]))
        bad = [t for t in self.windows["take_id"].unique() if split_of.get(t) != self.split]
        if bad:
            raise Refused(f"windows of takes outside the {self.split} split: {bad[:3]}")
        if self.split != "lockbox" and set(self.windows["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
            raise Refused(f"a lockbox person in the {self.split} inputs")

    def __len__(self):
        return len(self.windows)

    @property
    def persons(self):
        return sorted(self.windows["person_id"].unique())

    def truth(self, i):
        """joints (n, 22, 3) and root_yaw (n,) of window i, float64."""
        r = self.windows.iloc[int(i)]
        a, o, n = self.arrays[int(r["shard"])], int(r["offset"]), int(r["n"])
        return np.asarray(a["joints"][o:o + n], dtype=np.float64), np.asarray(a["root_yaw"][o:o + n], dtype=np.float64)

    def inputs(self, i):
        """The model inputs of window i: imu (n, 6, 6) float32 (acc g, gyro deg/s) and imu_valid (n, 6) of
        SITES, wrist column WRIST_COL; cam (n_c, 15, 2) torso units, cam_valid (n_c, 15), cam_fps, cam_scale
        and the camera params (21,) (K, R, t)."""
        r = self.windows.iloc[int(i)]
        s = int(r["shard"])
        a = self.arrays[s]
        if "imu" not in a:
            raise ValueError(f"{self.split}: truth only, no inputs")
        o, n, co, cn = int(r["offset"]), int(r["n"]), int(r["cam_offset"]), int(r["cam_n"])
        w = int(r["window"])
        return {"imu": np.asarray(a["imu"][o:o + n]), "imu_valid": np.asarray(a["imu_valid"][o:o + n]).astype(bool),
                "cam": np.asarray(a["cam"][co:co + cn]), "cam_valid": np.asarray(a["cam_valid"][co:co + cn]).astype(bool),
                "cam_fps": float(r["cam_fps"]), "cam_scale": float(r["cam_scale"]),
                "cam_params": np.asarray(a["cam_params"][w])}


# ---------------------------------------------------------------- scores
def yaw_fit(truth, pred):
    """Closed-form yaw (rad) about +y minimizing sum |rot_y(theta) pred - truth|^2 over all points of
    (n, J, 3) arrays: theta = atan2(B, A), A = sum (t_x p_x + t_z p_z), B = sum (t_x p_z - t_z p_x); 0
    when A = B = 0."""
    t, p = np.asarray(truth, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    A = float(np.sum(t[..., 0] * p[..., 0] + t[..., 2] * p[..., 2]))
    B = float(np.sum(t[..., 0] * p[..., 2] - t[..., 2] * p[..., 0]))
    return math.atan2(B, A) if (A != 0.0 or B != 0.0) else 0.0


def turn_y(x, theta):
    """rot_y(theta) applied to points x (..., 3) (rot_y of mdfm.body.rotations: +z turns toward +x)."""
    c, s = math.cos(theta), math.sin(theta)
    x = np.asarray(x, dtype=np.float64)
    return np.stack([c * x[..., 0] + s * x[..., 2], x[..., 1], -s * x[..., 0] + c * x[..., 2]], -1)


def mpjpe_yaw(pred, truth, joints=None):
    """Per-frame MPJPE (n,) m of pred against truth (n, 22, 3) (pred may be a constant (22, 3)), both
    pelvis-relative, after one yaw fitted over the scored joints of every frame; and the yaw (rad)."""
    t = np.asarray(truth, dtype=np.float64)
    p = np.broadcast_to(np.asarray(pred, dtype=np.float64), t.shape)
    t = t - t[:, PELVIS:PELVIS + 1]
    p = p - p[:, PELVIS:PELVIS + 1]
    if joints is not None:
        t, p = t[:, joints], p[:, joints]
    th = yaw_fit(t, p)
    return np.linalg.norm(turn_y(p, th) - t, axis=-1).mean(-1), th


def procrustes_errors(pred, truth):
    """Per-frame PA-MPJPE (n,) m: similarity Procrustes (rotation, translation, scale; no reflection) of
    pred onto truth per frame, (n, J, 3) each (pred may be a constant (J, 3)); a frame whose prediction
    has no spread gets scale 0 (every joint at the truth centroid)."""
    Y = np.asarray(truth, dtype=np.float64)
    X = np.broadcast_to(np.asarray(pred, dtype=np.float64), Y.shape)
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    Xc, Yc = X - mx, Y - my
    H = np.einsum("nji,njk->nik", Xc, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones_like(S)
    D[:, 2] = d
    R = np.einsum("nji,nj,nkj->nik", Vt, D, U)          # R = V diag(1, 1, d) U^T
    var = np.sum(Xc * Xc, axis=(1, 2))
    s = np.where(var > 0, np.sum(S * D, axis=1) / np.where(var > 0, var, 1.0), 0.0)
    aligned = s[:, None, None] * np.einsum("nij,nkj->nki", R, Xc) + my
    return np.linalg.norm(aligned - Y, axis=-1).mean(-1)


def wrist_truth(joints, root_yaw):
    """The G1b truth (n, 3) m: the left wrist relative to the pelvis in the pelvis-heading frame."""
    j = np.asarray(joints, dtype=np.float64)
    return heading_points(j[:, WRIST_JOINT], j[:, PELVIS], root_yaw)


def predictive_scores(draws, y, c, level=PARAMS["conformal_level"], dof=PARAMS["ellipsoid_dof"], energy=True):
    """Per-frame scores of a predictive given as draws ((n, m, 3), or one shared (m, 3) set) at truths y
    (n, 3): covered (n,) bool, degenerate (n,) bool, band width (n,) m (scaled by c), energy score (n,) m
    (None unless energy), radius (n,)."""
    D = np.asarray(draws, dtype=np.float64)
    mean, cov = conformal.moment_match(D, ddof=PARAMS["draws_ddof"])
    r, bad = conformal.mahalanobis(y, mean, cov)
    n = np.asarray(y).shape[0]
    cov_ok = conformal.covered(r, c, dof, level) & ~bad
    width = np.broadcast_to(conformal.band_width(cov, c, level), (n,))
    es = conformal.energy_score(D, y) if energy else None
    return {"covered": cov_ok, "degenerate": np.broadcast_to(bad, (n,)), "width": width, "energy": es, "radius": r}


def radii(gset, predictor, key="wrist"):
    """Mahalanobis radii (N,) of every knot of every window of gset under the predictor's `key`
    predictive, and the degenerate count: the calibration scores of the conformal factor."""
    out, bad = [], 0
    for i in range(len(gset)):
        J, yaw = gset.truth(i)
        pred = predictor.predict(gset, i)
        D = pred[key]
        if D is None:
            raise ValueError(f"the predictor has no {key} predictive")
        mean, cov = conformal.moment_match(np.asarray(D, dtype=np.float64), ddof=PARAMS["draws_ddof"])
        r, b = conformal.mahalanobis(wrist_truth(J, yaw), mean, cov)
        out.append(r)
        bad += int(np.broadcast_to(b, r.shape).sum())
    return np.concatenate(out), bad


def fit_conformal(gset, predictor, key="wrist"):
    """The conformal factor c of the predictor's `key` predictive on a calibration set (every knot a
    score): {'c', 'n_frames', 'n_degenerate', 'coverage_at_c'}."""
    if gset.split != "calib":
        raise Refused(f"conformal factors are fitted on the calib archives only, not {gset.split}")
    r, bad = radii(gset, predictor, key)
    c = conformal.ellipsoid_factor(r, PARAMS["ellipsoid_dof"], PARAMS["conformal_level"])
    cov = float(np.mean(conformal.covered(r, c, PARAMS["ellipsoid_dof"], PARAMS["conformal_level"])))
    return {"c": c, "n_frames": int(r.size), "n_degenerate": bad, "coverage_at_c": cov}


METRICS = ("g1a_m", "g1a15_m", "g1c_m", "g1c22_m", "coverage", "degenerate", "width_m", "energy_m", "width6_m")


def score_window(truth, root_yaw, pred, conformal_c):
    """Window metrics (dict of METRICS, means over the window's knots; degenerate is a count) of one
    prediction dict (module docstring) against the truth joints (n, 22, 3) and root yaw (n,)."""
    J = np.asarray(truth, dtype=np.float64)
    g1a, _ = mpjpe_yaw(pred["joints_six"], J)
    g1a15, _ = mpjpe_yaw(pred["joints_six"], J, CANON15)
    Pc = np.broadcast_to(np.asarray(pred["joints_camera"], dtype=np.float64), J.shape)
    g1c = procrustes_errors(Pc[:, CANON15], J[:, CANON15])
    g1c22 = procrustes_errors(Pc, J)
    y = wrist_truth(J, root_yaw)
    ps = predictive_scores(pred["wrist"], y, conformal_c["wrist"])
    w6 = math.nan
    if pred.get("wrist_six") is not None and conformal_c.get("wrist_six") is not None:
        w6 = float(predictive_scores(pred["wrist_six"], y, conformal_c["wrist_six"], energy=False)["width"].mean())
    return {"g1a_m": float(g1a.mean()), "g1a15_m": float(g1a15.mean()), "g1c_m": float(g1c.mean()),
            "g1c22_m": float(g1c22.mean()), "coverage": float(ps["covered"].mean()),
            "degenerate": int(ps["degenerate"].sum()), "width_m": float(ps["width"].mean()),
            "energy_m": float(ps["energy"].mean()), "width6_m": w6}


def score(gset, predictor, windows=None, log=None):
    """One row per window: its keys (window, take_id, archive, person_id, task, n) and METRICS. The
    predictor's conformal factors must come from the calib archives."""
    import pandas as pd
    log = log or (lambda *a: None)
    rows = []
    idx = range(len(gset)) if windows is None else windows
    t0 = time.time()
    for j, i in enumerate(idx):
        J, yaw = gset.truth(i)
        m = score_window(J, yaw, predictor.predict(gset, i), predictor.conformal)
        r = gset.windows.iloc[int(i)]
        rows.append({"window": int(i), "take_id": r["take_id"], "archive": r["archive"], "person_id": r["person_id"],
                     "task": int(r["task"]), "n": int(r["n"]), **m})
        if (j + 1) % 500 == 0:
            log(f"scored {j + 1} windows, {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the population prior
def mean_pose_sums(chunks):
    """Sum (22, 3) and count of pelvis-relative joints in each knot's pelvis-heading frame over chunks of
    (rot6d (k, 22, 6), rest (22, 3) m) float64 tensors."""
    from ..body.rotations import rot6d_to_matrix, yaw_angle
    from ..body.joints import fk
    total, n = np.zeros((22, 3)), 0
    for r6, rest in chunks:
        pos, R = fk(rot6d_to_matrix(r6), torch.zeros(3, dtype=r6.dtype), rest)
        yaw = yaw_angle(R[:, 0]).numpy()
        p = pos.numpy()
        total += heading_points(p, p[:, PELVIS:PELVIS + 1], yaw[:, None]).sum(0)
        n += p.shape[0]
    return total, n


def wrist_draws(r6, rest):
    """Wrist positions (m, 3) m relative to the pelvis in the pelvis-heading frame of single knots:
    rot6d (m, 22, 6), rest (m, 22, 3) float64 tensors."""
    from ..body.rotations import rot6d_to_matrix, yaw_angle
    from ..body.joints import fk
    pos, R = fk(rot6d_to_matrix(r6), torch.zeros(r6.shape[0], 3, dtype=r6.dtype), rest)
    return heading_points(pos[:, WRIST_JOINT].numpy(), pos[:, PELVIS].numpy(), yaw_angle(R[:, 0]).numpy())


@dataclass
class PopulationPrior:
    """The trivial baseline of G1 (spec section 6), from train persons only: mean_pose (22, 3) m, the
    mean pelvis-relative pose in the pelvis-heading frame (a constant prediction for G1a and G1c);
    draws (m, 3) m, wrist positions of single train knots in the same frame (one shared predictive for
    G1b); c_wrist, its conformal factor on the calib archives (None until fitted); meta: provenance, the
    train person list included."""
    mean_pose: np.ndarray
    draws: np.ndarray
    c_wrist: float = None
    meta: dict = field(default_factory=dict)
    name: str = "population_prior"

    def predict(self, gset, i):
        return {"joints_six": self.mean_pose, "joints_camera": self.mean_pose, "wrist": self.draws, "wrist_six": None}

    @property
    def conformal(self):
        if self.c_wrist is None:
            raise ValueError("the population prior has no conformal factor: fit it on the calib archives")
        return {"wrist": float(self.c_wrist), "wrist_six": None}

    @property
    def train_persons(self):
        return frozenset(self.meta.get("train_persons", ()))

    @property
    def train_persons_sha256(self):
        return self.meta.get("train_persons_sha256", "")

    def to_dict(self):
        return {"name": self.name, "mean_pose": np.asarray(self.mean_pose).tolist(),
                "draws": np.asarray(self.draws).tolist(), "c_wrist": self.c_wrist, "meta": self.meta}

    @property
    def sha256(self):
        """SHA-256 of the canonical JSON of the prior (the 'checkpoint' hash of a trivial-baseline run)."""
        return common.sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["mean_pose"], dtype=np.float64), np.asarray(d["draws"], dtype=np.float64),
                   d.get("c_wrist"), dict(d.get("meta", {})), d.get("name", "population_prior"))

    def save(self, path):
        common.write_text(path, json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def population_prior(store, seed=PARAMS["prior_draw_seed"], stride=PARAMS["prior_mean_stride"],
                     n_draws=PRIOR_DRAWS):
    """The PopulationPrior of a train BodyStore (mdfm.data.body_store): mean pose over every stride-th knot
    of every take; n_draws wrist positions, one uniform knot of each of n_draws subject-balanced train
    windows (BodyStore.sample with numpy Generator(seed)). Any other split is refused."""
    from ..data import amass_index
    from ..body.smplh import SkeletonBank
    if store.split != "train":
        raise Refused(f"the population prior comes from the train persons only, not {store.split}")
    others = set(amass_index.LOCKBOX_PERSONS) | set(amass_index.index().query("split != 'train'")["person_id"])
    if set(store.persons) & others:
        raise Refused("a person of another split in the train store")
    bank = SkeletonBank(torch.float64)
    with torch.no_grad():
        rest = bank.rest(torch.from_numpy(store.betas), torch.from_numpy(store.gender))
    tk = store.takes

    def chunks():
        for i in range(len(tk)):
            a = store.arrays[int(tk["shard"].iat[i])]["rot6d"]
            o, n = int(tk["offset"].iat[i]), int(tk["n_knots"].iat[i])
            yield torch.from_numpy(np.asarray(a[o:o + n:stride], dtype=np.float64)), rest[i]

    with torch.no_grad():
        total, count = mean_pose_sums(chunks())
        gen = np.random.default_rng(seed)
        bw = store.sample(gen, n_draws, win_s=WIN_S)
        f = gen.integers(0, bw.lengths.numpy())
        rows = torch.arange(n_draws)
        r6 = bw.rot6d[rows, torch.from_numpy(f)].double()
        d = wrist_draws(r6, bank.rest(bw.betas.double(), bw.gender))
    persons = sorted(store.persons.tolist())
    meta = {"source": "train body store", "store_root": common.rel(store.root), "stride": int(stride),
            "n_knots_mean": int(count), "n_takes": int(len(tk)), "seed": int(seed), "n_draws": int(n_draws),
            "draw_persons": sorted(set(bw.person.tolist())), "train_persons": persons,
            "train_persons_sha256": amass_index.person_list_sha256(persons)}
    return PopulationPrior(total / count, d, None, meta)


# ---------------------------------------------------------------- results
def _pwm(t, col):
    return common.person_weighted_mean(t[col].to_numpy(), t["person_id"].to_numpy(), t["n"].to_numpy())


def _boot(t, col, stat=None, cols=None):
    v = t[cols].to_numpy() if cols else t[col].to_numpy()
    return common.cluster_bootstrap(v, t["person_id"].to_numpy(), PARAMS["bootstrap_n"], PARAMS["bootstrap_seed"],
                                    t["n"].to_numpy(), PARAMS["bootstrap_level"], stat)


def lines(table, baseline, predictor, rule_sha256, scale=1000.0):
    """The three GateResults (G1a, G1b, G1c) of a scored table against the baseline table (the population
    prior on the same windows; the table itself for the prior), lengths in mm."""
    persons = table["person_id"].nunique()
    ck = predictor.sha256
    tp = predictor.train_persons_sha256
    out = []
    for gate_line, col, pas, kil, higher, unit in (
            ("G1a six IMUs MPJPE", "g1a_m", PARAMS["g1a_pass_mm"], PARAMS["g1a_kill_mm"], False, "mm"),
            ("G1b single wrist coverage", "coverage", PARAMS["g1b_coverage_pass"], PARAMS["g1b_coverage_kill"], True, ""),
            ("G1c single camera PA-MPJPE", "g1c_m", PARAMS["g1c_pass_mm"], PARAMS["g1c_kill_mm"], False, "mm")):
        f = 1.0 if col == "coverage" else scale
        point = f * _pwm(table, col)
        lo, hi = _boot(table, col)
        base = f * _pwm(baseline, col)
        extra, conds = {}, ()
        if col == "coverage":
            width = scale * _pwm(table, "width_m")
            w6 = table["width6_m"].to_numpy()
            ratio_w = width / (scale * _pwm(table, "width6_m")) if np.isfinite(w6).all() else None
            es, es0 = scale * _pwm(table, "energy_m"), scale * _pwm(baseline, "energy_m")
            ratio_e = es / es0 if es0 > 0 else None
            c_w = None if ratio_w is None else bool(ratio_w <= PARAMS["g1b_width_ratio_max"])
            c_e = None if ratio_e is None else bool(ratio_e <= PARAMS["g1b_energy_ratio_max"])
            conds = (c_w, c_e)
            extra = {"band_width_mm": width, "width_ratio_to_six_imu": ratio_w, "width_bound_met": c_w,
                     "energy_score_mm": es, "energy_score_prior_mm": es0, "energy_ratio_to_prior": ratio_e,
                     "energy_bound_met": c_e, "degenerate_frames": int(table["degenerate"].sum()),
                     "n_frames": int(table["n"].sum())}
        v = common.verdict(point, pas, kil, higher, conds)
        out.append(common.GateResult(GATE, gate_line, point, (f * lo, f * hi), pas, kil, v, base, rule_sha256, ck, tp,
                                     persons, unit, higher, extra))
    return out


def secondary(table, scale=1000.0):
    """Secondary rows without kill (spec section 6): the other joint set of G1a and G1c, the G1b width and
    energy score, walking windows only; point and interval in mm (coverage unitless)."""
    from ..data.windows import task_vocabulary
    rows = []
    walk = task_vocabulary()["ids"].get("walk")
    for name, sub in (("all windows", table), ("walking windows (task walk)", table[table["task"] == walk])):
        if sub.empty:
            continue
        for label, col, f in (("G1a MPJPE, 15 canonical joints", "g1a15_m", scale), ("G1a MPJPE, 22 joints", "g1a_m", scale),
                              ("G1c PA-MPJPE, 15 canonical joints", "g1c_m", scale),
                              ("G1c PA-MPJPE, 22 joints", "g1c22_m", scale), ("G1b coverage", "coverage", 1.0),
                              ("G1b band width", "width_m", scale), ("G1b energy score", "energy_m", scale)):
            lo, hi = _boot(sub, col)
            rows.append({"set": name, "row": label, "point": f * _pwm(sub, col), "lo": f * lo, "hi": f * hi,
                         "n_persons": int(sub["person_id"].nunique()), "n_windows": int(len(sub)),
                         "n_frames": int(sub["n"].sum())})
    return rows


def per_group(table, key, scale=1000.0):
    """Person-weighted means per value of `key` (person_id or archive): mm, coverage unitless."""
    import pandas as pd
    out = []
    for g, sub in table.groupby(key, sort=True):
        r = {key: g, "n_persons": int(sub["person_id"].nunique()), "n_windows": int(len(sub)), "n_frames": int(sub["n"].sum())}
        if key == "person_id":
            r["archive"] = ",".join(sorted(sub["archive"].unique()))
        for col, name, f in (("g1a_m", "g1a_mpjpe_mm", scale), ("g1a15_m", "g1a_mpjpe15_mm", scale),
                             ("g1c_m", "g1c_pa_mpjpe_mm", scale), ("g1c22_m", "g1c_pa_mpjpe22_mm", scale),
                             ("coverage", "g1b_coverage", 1.0), ("width_m", "g1b_width_mm", scale),
                             ("energy_m", "g1b_energy_mm", scale)):
            r[name] = f * _pwm(sub, col)
        r["g1b_degenerate_frames"] = int(sub["degenerate"].sum())
        out.append(r)
    return pd.DataFrame(out)


def _check_train_persons(predictor):
    """Before a lockbox read: the predictor's training person list is non-empty, matches its SHA-256 and
    holds no lockbox person (Refused otherwise), so that a read is never logged for a predictor the
    harness would refuse after opening the set."""
    from ..data import amass_index
    persons = set(predictor.train_persons)
    if not persons:
        raise Refused("a lockbox read needs the predictor's training person list (empty)")
    if amass_index.person_list_sha256(persons) != predictor.train_persons_sha256:
        raise Refused("the predictor's training person list does not match its SHA-256")
    if persons & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("lockbox persons among the predictor's training persons")


def run(split, predictor, root=None, checkpoint_sha256=None, reread=False, reason="", log=None):
    """Score a predictor on a split. The lockbox needs the registered G1 rule; the root (gates-only store),
    the input manifest and the predictor's training persons (_check_train_persons) are checked first, then
    the read is logged (common.log_lockbox_read) before G1Set opens any array; a second read of the same
    checkpoint and rule is refused unless reread with a reason. Returns (G1Set, per-window table, rule
    SHA-256, read row or None)."""
    _check_split(split)
    read = None
    if split == "lockbox":
        _, rule_sha = common.rule(RULE_PATH, GATE)
        root = _check_root(split, default_root(split) if root is None else root)
        _check_train_persons(predictor)
        with open(os.path.join(root, split, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        if man.get("split") != split or man.get("gate") != GATE:
            raise Refused(f"{root}: the manifest names {man.get('gate')}/{man.get('split')}, not {GATE}/{split}")
        ck = checkpoint_sha256 or predictor.sha256
        read = common.log_lockbox_read(GATE, ck, rule_sha, reread, reason, inputs_sha256=man["manifest_sha256"],
                                       code_sha256=code_sha256(SCORE_CODE))
        gset = G1Set(split, root, allow_lockbox=True)
    else:
        with open(RULE_PATH, "rb") as f:
            rule_sha = common.sha256_bytes(f.read())
        gset = G1Set(split, root)
    overlap = set(gset.persons) & set(predictor.train_persons)
    if overlap:
        raise Refused(f"persons of the {split} set among the predictor's training persons: {sorted(overlap)[:3]}")
    return gset, score(gset, predictor, log=log), rule_sha, read
