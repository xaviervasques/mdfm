"""EMG kernel (design section 3): per-channel log-envelopes from the body through a monotone decoder.

No forward physics exists for raw EMG waveforms, so the kernel predicts the log-envelope of each
channel. Observation side: log_envelope() follows the EMG band copy of the window store (spec
2.2, mdfm/data/window_store.py): band-pass 20 to 450 Hz (4th-order Butterworth, zero phase,
sosfiltfilt with padlen max(3 (2 sections + 1), ceil(fps / 20)); a raw signal at or below 900 Hz
is refused), 20 ms moving RMS (a centred window of 2 round(0.01 fps) + 1 samples that shrinks at
the ends), natural log with a per-channel floor at the 5th percentile of the envelope, at least
1e-6 in the signal unit (mV). It runs on one contiguous valid run; decimation belongs to the store.

Model side: EMGDecoder maps the leg moments of mdfm.kernels.dynamics (moment_channels of
leg_torques: hip flexion, knee flexion, ankle dorsiflexion, hip abduction, N m/kg, anatomical
signs) and the joint angles and rates of the same channels to

  env(m, leg) = sum_k softplus(a_mk) relu(s_mk tau_k) + softplus(c_m) sum_k |s_mk| |tau_k|
                + softplus(b_m) + softplus(MLP(angles, rates))_m
  log_env_hat = log env

with s_mk in {-1, 0, +1} the action of muscle m on channel k from muscle_actions.yaml (+1 when
the muscle produces the positive anatomical moment, -1 the negative one, 0 when it does not cross
the joint). The first term is the agonist drive, the second the co-contraction that grows with
the moment magnitude at the joints the muscle crosses, the last two a tonic level and a small
non-negative kinematic term. Every term is non-decreasing in |tau_k|, so d env / d |tau| >= 0.

Channels, not muscle names: a record's EMG channel set varies (stroke_gait_vancriekinge: 36
channel sets, 73 records with a muscle and side twice, reports/stage1_sources), so the decoder
reads per-channel indices from channel_table(): each column carries its muscle id and side, two
electrodes on one muscle get the same prediction, and emg_nll removes a log gain per record and
per column, so each electrode keeps its own amplitude. Muscles without a modelled action (trunk,
upper limb, face, forearm_array, unknown), names outside the vocabulary and sides other than L or R
are masked.

The body enters detached in the pilot (spec 1.5, item 2): EMGDecoder.forward detaches its
inputs unless detach_body=False.
"""
import functools
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import butter, sosfiltfilt

from ..body.kinematics import check_precision
from .dynamics import MOMENT_CHANNELS

ACTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muscle_actions.yaml")
# "joint action" of muscle_actions.yaml -> (moment channel of dynamics.MOMENT_CHANNELS, sign)
ACTION_CHANNELS = {"hip flexion": (0, 1.0), "hip extension": (0, -1.0),
                   "knee flexion": (1, 1.0), "knee extension": (1, -1.0),
                   "ankle dorsiflexion": (2, 1.0), "ankle plantarflexion": (2, -1.0),
                   "hip abduction": (3, 1.0), "hip adduction": (3, -1.0)}
# actions of the modelled joints that the pilot records without a moment channel; any other hip, knee or
# ankle action is refused, so that a misspelled action cannot drop out of the decoder silently
UNMODELLED_LEG_ACTIONS = frozenset({"hip internal_rotation", "hip external_rotation", "knee internal_rotation",
                                    "knee external_rotation"})
SIDES = {"L": 0, "R": 1}
SIGNAL_KINDS = ("raw", "bandpassed", "rectified", "envelope")
BAND_HZ = (20.0, 450.0)
RMS_S = 0.02
FLOOR_PERCENTILE = 5.0
FLOOR_MIN = 1e-6                   # in the signal unit (mV)
FLOOR_LOG_SIGMA = 0.05             # floor of the log-envelope noise scale [cfg]
N_FEATURES = 2 * len(MOMENT_CHANNELS)


# ---------------------------------------------------------------- observation side
def _moving_mean(x, n):
    """Centred moving mean over n samples (n odd) along axis 0; the window shrinks at both ends."""
    h = n // 2
    L = x.shape[0]
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    i = np.arange(L)
    lo, hi = np.maximum(i - h, 0), np.minimum(i + h + 1, L)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def envelope(signal, fps, signal_kind="raw"):
    """20 ms moving RMS (T, C) of an EMG signal (T, C) in its own unit (module docstring). raw: band-pass
    first; bandpassed and rectified: RMS only; envelope: returned as is (clipped at 0). ValueError on
    non-finite samples (the caller runs it per contiguous valid run), on a raw signal at or below 900 Hz
    or not longer than the filter padding."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signal must be (T, C), got {x.shape}")
    if signal_kind not in SIGNAL_KINDS:
        raise ValueError(f"signal_kind must be one of {SIGNAL_KINDS}, got {signal_kind!r}")
    if not np.isfinite(x).all():
        raise ValueError("non-finite EMG samples: run per contiguous valid run")
    if signal_kind == "envelope":
        return np.maximum(x, 0.0)
    if signal_kind == "raw":
        if fps <= 2 * BAND_HZ[1]:
            raise ValueError(f"EMG at {fps:g} Hz cannot carry the {BAND_HZ[0]:g} to {BAND_HZ[1]:g} Hz band")
        sos = butter(4, list(BAND_HZ), btype="bandpass", fs=fps, output="sos")
        pad = max(3 * (2 * len(sos) + 1), int(math.ceil(fps / BAND_HZ[0])))
        if x.shape[0] <= pad:
            raise ValueError(f"{x.shape[0]} samples, the filter needs more than {pad}")
        x = sosfiltfilt(sos, x, axis=0, padlen=pad)
    n = 2 * int(round(RMS_S * fps / 2)) + 1
    return np.sqrt(np.maximum(_moving_mean(x * x, n), 0.0))


def log_envelope(signal, fps, signal_kind="raw", floor=None):
    """Log-envelope (T, C) of an EMG signal (T, C) and the per-channel floor (C,) it used: log max(env,
    floor), floor the 5th percentile of the envelope over the signal, at least FLOOR_MIN (pass the
    record's floor when the signal is one run of it)."""
    env = envelope(signal, fps, signal_kind)
    if floor is None:
        floor = np.maximum(np.percentile(env, FLOOR_PERCENTILE, axis=0), FLOOR_MIN)
    floor = np.asarray(floor, dtype=np.float64)
    return np.log(np.maximum(env, floor[None])), floor


# ---------------------------------------------------------------- muscle table
class MuscleTable:
    """names (M,) of the vocabulary in muscle_actions.yaml order; action (M, 4) float64 in {-1, 0, 1} on
    the moment channels; modelled (M,) bool (at least one modelled action); index {name: row}. A hip, knee
    or ankle action outside ACTION_CHANNELS and UNMODELLED_LEG_ACTIONS raises ValueError."""

    def __init__(self, rows):
        self.names = tuple(r["muscle"] for r in rows)
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate muscle rows in the action table")
        self.index = {n: i for i, n in enumerate(self.names)}
        self.action = np.zeros((len(rows), len(MOMENT_CHANNELS)))
        for i, r in enumerate(rows):
            for act in r.get("actions") or []:
                if act.split(" ", 1)[0] in ("hip", "knee", "ankle") and act not in ACTION_CHANNELS \
                        and act not in UNMODELLED_LEG_ACTIONS:
                    raise ValueError(f"{r['muscle']}: unknown leg action {act!r}")
                if act in ACTION_CHANNELS:
                    k, s = ACTION_CHANNELS[act]
                    if self.action[i, k] not in (0.0, s):
                        raise ValueError(f"{r['muscle']}: opposite actions on {MOMENT_CHANNELS[k]}")
                    self.action[i, k] = s
        self.modelled = np.abs(self.action).sum(1) > 0


def read_actions(path=ACTIONS_PATH):
    """The parsed muscle_actions.yaml (dict)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=4)
def load_table(path=ACTIONS_PATH):
    """MuscleTable of muscle_actions.yaml."""
    return MuscleTable(read_actions(path)["muscles"])


def channel_table(channels, table=None):
    """Per-channel indices of a record's EMG channels (a list of dicts with muscle and side, as
    windows.Stream.meta['channels'] or the record's channels): muscle (C,) int64 row of the table (-1 for a
    name outside it), side (C,) int64 (0 L, 1 R, -1 otherwise), modelled (C,) bool. Duplicated muscles keep
    one column each. A channel whose contract kind is given and is not 'emg' (ecg, eog, aux,
    unresolved_emg_ecg) or whose resolved flag is false is not modelled (windows.Stream.meta keeps neither
    field, so this holds on record channel tables)."""
    table = load_table() if table is None else table
    mus = np.array([table.index.get(str(c.get("muscle")), -1) for c in channels], dtype=np.int64)
    side = np.array([SIDES.get(str(c.get("side")), -1) for c in channels], dtype=np.int64)
    emg = np.array([(c.get("kind") is None or str(c.get("kind")) == "emg")
                    and (c.get("resolved") is None or bool(c.get("resolved"))) for c in channels], dtype=bool)
    modelled = (mus >= 0) & (side >= 0) & table.modelled[np.maximum(mus, 0)] & emg
    return mus, side, modelled


# ---------------------------------------------------------------- decoder
class EMGDecoder(torch.nn.Module):
    """Monotone decoder from leg moments and joint kinematics to per-channel log-envelopes (module
    docstring). Buffers action (M, 4) and modelled (M,) come from the muscle table (non-persistent: the
    table file is their source)."""

    def __init__(self, hidden=64, table=None):
        super().__init__()
        table = load_table() if table is None else table
        M = len(table.names)
        self.names = table.names
        self.register_buffer("action", torch.tensor(table.action, dtype=torch.float32), persistent=False)
        self.register_buffer("modelled", torch.tensor(table.modelled), persistent=False)
        self.a_raw = torch.nn.Parameter(torch.zeros(M, len(MOMENT_CHANNELS)))
        self.c_raw = torch.nn.Parameter(torch.full((M,), -2.0))
        self.b_raw = torch.nn.Parameter(torch.zeros(M))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(N_FEATURES, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, M))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.constant_(self.mlp[-1].bias, -4.0)

    def activation(self, moments, angles, rates):
        """Envelope (N, 2, M) > 0 of every muscle on both legs from moments, angles, rates (N, 2, 4)."""
        s = self.action.to(moments.dtype)
        drive = torch.relu(moments[..., None, :] * s)                               # (N, 2, M, 4)
        act = (drive * (F.softplus(self.a_raw.to(moments.dtype)) * s.abs())).sum(-1)
        act = act + F.softplus(self.c_raw.to(moments.dtype)) * (moments.abs()[..., None, :] * s.abs()).sum(-1)
        act = act + F.softplus(self.b_raw.to(moments.dtype))
        feat = torch.cat([angles, rates], -1).to(self.mlp[0].weight.dtype)
        return act + F.softplus(self.mlp(feat)).to(moments.dtype)

    def forward(self, moments, angles, rates, muscle, side, detach_body=True):
        """log_env_hat (N, C) and mask (N, C) bool. moments (N, 2, 4) N m/kg (dynamics.moment_channels),
        angles (N, 2, 4) rad and rates (N, 2, 4) rad/s (dynamics.joint_angles); muscle, side (N, C) or (C,)
        int64 from channel_table. Masked entries hold 0. The mask sees muscle and side only: the contract kind
        and resolved flag are in channel_table's modelled, which the caller ANDs into emg_nll's valid."""
        check_precision(moments, angles, rates)
        if detach_body:
            moments, angles, rates = moments.detach(), angles.detach(), rates.detach()
        act = self.activation(moments, angles, rates)
        N = act.shape[0]
        muscle = torch.as_tensor(muscle, device=act.device)
        side = torch.as_tensor(side, device=act.device)
        C = muscle.shape[-1]
        muscle, side = muscle.expand(N, C), side.expand(N, C)
        mask = (muscle >= 0) & (side >= 0) & self.modelled[muscle.clamp_min(0)]
        rows = torch.arange(N, device=act.device)[:, None]
        val = act[rows, side.clamp_min(0), muscle.clamp_min(0)]
        return torch.where(mask, torch.log(val), torch.zeros((), dtype=val.dtype, device=val.device)), mask


# ---------------------------------------------------------------- likelihood
def emg_nll(pred, obs, valid, w, log_sigma, weight=None, record=None, n_windows=None, floor=FLOOR_LOG_SIGMA):
    """Negative log-likelihood (W,) of log-envelopes. pred, obs (N, C) log-envelopes (C: the padded
    channel columns of the micro-step, each window's own channel table); valid (N, C) bool (observation
    valid, the decoder mask and channel_table's modelled); w (N,) int64 window of each point; log_sigma broadcastable to (W,) or
    (W, C); weight (N,) point weights (default 1); record (W,) int64 groups of windows from one record
    (default one per window); n_windows W (default max(w) + 1).

    The per-record, per-column log gain is removed in closed form: the weighted mean of obs - pred over the
    group's valid points of that column (the difference of the means). Residuals then follow a Gaussian
    with scale sqrt(floor^2 + exp(2 log_sigma)). Invariant to adding a constant to obs per record and
    column."""
    check_precision(pred)
    dev, dt = pred.device, pred.dtype
    N, C = pred.shape
    w = w.to(dev)
    W = int(n_windows) if n_windows is not None else (int(record.shape[0]) if record is not None
                                                      else int(w.max().item()) + 1)
    m = valid.to(dev, torch.bool)
    wt = torch.ones(N, dtype=dt, device=dev) if weight is None else weight.to(dev, dt)
    rec = torch.arange(W, device=dev) if record is None else record.to(dev)
    _, gid_w = torch.unique(rec, return_inverse=True)
    n_g = int(gid_w.max().item()) + 1 if W else 0
    key = (gid_w[w][:, None] * C + torch.arange(C, device=dev)[None]).reshape(-1)
    q = (wt[:, None] * m).to(dt)
    d = torch.where(m, obs.to(dev, dt) - pred, torch.zeros((), dtype=dt, device=dev))
    S_d = torch.zeros(n_g * C, dtype=dt, device=dev).index_add(0, key, (q * d).reshape(-1))
    S_1 = torch.zeros(n_g * C, dtype=dt, device=dev).index_add(0, key, q.reshape(-1))
    gain = (S_d / S_1.clamp_min(torch.finfo(dt).tiny))[key].reshape(N, C)
    ls = torch.as_tensor(log_sigma, dtype=dt, device=dev)
    ls = (ls[:, None] if ls.dim() == 1 else ls).expand(W, C)
    sigma = (floor ** 2 + torch.exp(2 * ls)).sqrt()[w]
    r = d - gain
    nll = (0.5 * (r / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2 * math.pi)) * q
    return torch.zeros(W, dtype=dt, device=dev).index_add(0, w, nll.sum(-1))
