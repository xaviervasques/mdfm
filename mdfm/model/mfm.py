"""The assembled stage-1 model (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5 and section 3 step 5,
task B4.15): tokenizer, encoder, posterior heads, motion and shape priors, time-query decoder, nuisance
heads, ForceHead, EMG decoder, per-source joint offsets and a non-persistent SkeletonBank.

Flow of a micro-step (Batch, built from windows.Window objects by batch_from_windows until the collate of
task B4.17 exists):
- encode(batch) -> Posterior: tokenizer and encoder under the caller's autocast (bf16 in training), then
  float32 heads: mu_z, logvar_z (W, P, 32) at the latent positions, mu_b, logvar_b (W, 16) at the subject
  position, and the nuisance (Nuisance) from the nuisance-token outputs only (nuisance_from_hidden): one
  IMUNuisance (W, 14) indexed by the 14 canonical sites, one CameraNuisance (W,), one pose3d log sigma (W,).
- decode(z, beta, knots, n_patches, lengths) -> decoder.BodyOut at the requested (window, knot) pairs with the
  root grid of the windows (the root path is integrated from the centre knot lengths // 2, so the lengths are
  required; reports/bench/root_motion_diagnosis.md, fix 1), the whole decoder in float32 whatever the caller's
  autocast (decoder.py; reports/bench/phaseA_diagnosis.md, C1); dense() places them in (W, T) knot buffers
  (KnotBody), the knots outside the decoded set holding the mean pose (the rotations of decoder.mean_R) and
  the decoder's transl0.
- render(body, batch, nuis) -> predictions at every kernel point from one KnotBatch.kinematics call, under
  a disabled autocast in float32: IMU acc (N, 14, 3) g and gyro (N, 14, 3) deg/s (IMUNuisance.predict with
  the site defaults of factory.imu_render.site_offsets and the family's prior rotations R0); pose (N, 15,
  3) m through kernels.pose.to_canonical15 with the source's joint offsets, pose2d also with the camera of
  the nuisance aimed at the window-mean predicted pelvis of the stream's points; per-foot vertical force
  (N, 2) body weights from ForceHead (the Newton floor split by the decoder's contact logits); EMG
  log-envelopes (N, C) from the detached body (departure 2: leg torques, joint angles, EMGDecoder).
- log_lik(pred, batch) -> (W,): minus the sum of the kernel NLLs (nll_terms: imu_nll, pose3d_nll with the
  root track when present, pose2d_nll, force_nll, emg_nll), each a per-window sum with the point weights.
- log_joint(z, beta, nuis, batch, body=None) = log_lik + log p(z | sex, task, beta) + log p(beta | sex) +
  log p(nuis | family) (W,); with a body given, z and beta are not read and only log_lik and the nuisance
  prior remain (refinement with the body held at truth, model/refine.py). The condition dropout of the
  motion prior is off in log_joint (a density, not a training augmentation).
- posterior_samples(batch, n=16, refine_steps=20) -> Samples at every knot of every window.

Sizes: the pilot configuration (PILOT, 12 x 512 encoder, 4-layer decoder of width 512, 4-layer prior of
width 256) is asserted to hold 57M to 62M parameters (PILOT_RANGE); TINY runs on the CPU in the tests.

JEPA (optional, docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2): attach_jepa adds model/jepa.JEPA as the
submodule jepa (the EMA target copy of the tokenizer and encoder, and the predictor), which the training loop
reads in phase B when the config's jepa section is enabled. Without it, self.jepa is None, the model and its
state dict are those of a model built before JEPA existed, and the jepa.* entries of a JEPA run's state dict
are not read (attach_jepa, State dicts).

Name tables (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.1, task S2.3): MFMConfig.names, a
train/names.NameTables, sets the tables a model holds one row per name of: the sources of SourceOffsets (equal to
MFMConfig.sources), the tasks of the motion prior's task embedding (its vocabulary file), the layout, device and
camera vocabularies of the tokenizer's nuisance tables (tokenize.nuisance_fields), and the mount families, which
must equal the families of the mount priors the heads load. names None (the default, and every stage-1 config,
train/config.model_config) builds the stage-1 model from the module constants: SOURCES, the version-1 vocabulary
(prior.TASKS), tokenize.NUIS_FIELDS; a model built from names.STAGE1 has the same state-dict keys, shapes and
initial values. A Batch carries the nuisance vocabularies its tokens were coded with (nuis_fields, None for
tokenize.NUIS_FIELDS), and the model refuses a batch whose sources, mount families or nuisance vocabularies differ
from its own.

Open body (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4; MFMConfig.body 'open', OPEN_PILOT, OPEN_TINY,
configs/stage1_open.yaml). The shape vector beta of every interface above becomes the phenotype latent u (W, 6) of
mdfm/body/openbody.py (phenotype = lo + (hi - lo) sigmoid(u)): PhenotypeHead in place of ShapeHead, the pooled
phenotype prior of reports/open_body_priors/priors.json in all three sex rows (open_shape_prior) in place of the
per-gender beta prior, the motion prior and the decoder's FiLM conditioned on the 6 values, the decoder's mean pose
and window priors from reports/open_body_priors, the mount priors of reports/open_mount_priors, and the skeleton
openbody.LatentSkeletonBank (rest(u, gender): the P3 basis, the gender code not read). Its buffers are
non-persistent, so the open model's state dict holds no basis value either. With body 'smplh' (the default field
value, v1 as born) the SMPL+H SkeletonBank is built only behind the reference flag of mdfm/body/backend.py
(MDFM_SMPLH_REFERENCE=1); a model built with skeleton=False reads no body file.

Licence guard: the SkeletonBank arrays are non-persistent buffers, so the state_dict holds no SMPL+H array.
audit_state_dict screens a state dict, or any nested dict, list or tuple of tensors (an optimizer state_dict,
a checkpoint), for SMPL+H content: any dimension equal to 6,890 (vertices) or 52 (joints with hands), SMPL+H
array names in keys, and tensors whose bytes (float32 or float64) hash to an SMPL+H array as SMPLHBody
stores it (shapedirs cut to 16 betas) or to the derived J_template and J_dirs of each gender and of the
three stacked as SkeletonBank holds them, or that equal one of them by allclose (tensors up to 100,000
values). A hash cannot catch an arbitrary sub-slice; the shape screen and the non-persistent buffers are the
guard there. The priors (motion prior weights, the shape prior of reports/body_store/priors.json, the mean
pose, the mount priors) are statistics and learned weights, not SMPL+H arrays.
"""
import hashlib
import math
from dataclasses import dataclass, field, fields, replace
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from .. import paths
from ..body.hermite import hermite_gather, support
from ..body.kinematics import BodyKin, KnotBatch
from ..body.rotations import matrix_to_rot6d
from ..body.sex import GENDERS
from ..data.canonical import Refused, is_cody
from ..data.sources import PILOT as PILOT_SOURCES
from ..factory import layouts as LY
from ..factory.imu_render import SITES, site_offsets
from ..kernels.camera import camera_from_params
from ..kernels.dynamics import joint_angles, leg_torques, moment_channels
from ..kernels.emg import EMGDecoder, channel_table, emg_nll
from ..kernels.force import ForceHead, com_acc, contact_prob, force_nll
from ..kernels.imu import imu_nll
from ..kernels.nuisance import CameraNuisance, IMUNuisance, SourceOffsets
from ..kernels.pose import pose2d_nll, pose3d_nll, root_nll, to_canonical15, window_mean
from . import tokenize as TK
from .decoder import BodyDecoder, all_knots, scatter_dense
from .encoder import Encoder, n_parameters
from .posterior import LatentHead, PhenotypeHead, ShapeHead, sample
from .prior import TASK_UNKNOWN, TASKS, Cond, MotionPrior, PhenotypePrior, ShapePrior
from .heads import NuisanceHeads

RATE = 100.0
# the size of the v1 shape vector (16 betas, mdfm.body.smplh.N_BETAS; tests/test_public_imports.py checks the
# equality): a constant here, so that this module imports no SMPL+H module (plan phase E step 2)
N_BETAS = 16
SOURCES = ("synthetic",) + tuple(PILOT_SOURCES)
MODALITIES = ("imu", "pose3d", "pose2d", "force", "emg")
N_SITES = len(SITES)
PILOT_RANGE = (57_000_000, 62_000_000)
FORCE_LOG_SIGMA0 = math.log(0.05)      # [cfg] BW above the 0.01 BW floor of force_nll
EMG_LOG_SIGMA0 = math.log(0.5)         # [cfg] log-envelope units above the 0.05 floor of emg_nll
SEGMENT = 16                           # [cfg] consecutive samples per kernel segment (spec 2.2, collate)
KIND_COL = TK.FIELD_NAMES.index("kind")
FEET = ((7, 10), (8, 11))              # (ankle, foot) joints of the left and right foot


@dataclass(frozen=True)
class MFMConfig:
    """Sizes of the model (spec 2.5): encoder d x layers with heads, stems of stem_hidden, latent d_z;
    prior width, depth, heads and span; decoder width, depth, heads and Fourier features; nuisance head
    hidden width; ForceHead and EMG decoder hidden widths; the source names of SourceOffsets (row 0
    synthetic); skeleton: load the skeleton bank (for the SMPL+H body it needs the MoveAtlas SMPL+H files and the
    reference flag); names: the run's name tables (train/names.NameTables; None: the stage-1 tables of the module
    constants, module docstring); body: 'smplh' (v1 as born, behind the reference flag of mdfm/body/backend.py) or
    'open' (Anny, module docstring, Open body)."""
    d: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio: int = 4
    stem_hidden: int = 1024
    d_z: int = 32
    prior_d: int = 256
    prior_layers: int = 4
    prior_heads: int = 8
    prior_span: int = 8
    dec_d: int = 512
    dec_layers: int = 4
    dec_heads: int = 8
    n_fourier: int = 32
    head_hidden: int = 1024
    force_hidden: int = 64
    emg_hidden: int = 64
    sources: tuple = SOURCES
    rate: float = RATE
    skeleton: bool = True
    names: object = None
    body: str = "smplh"


PILOT = MFMConfig()
TINY = MFMConfig(d=64, layers=2, heads=4, stem_hidden=128, prior_d=32, prior_layers=1, prior_heads=4, dec_d=64,
                 dec_layers=1, dec_heads=4, n_fourier=8, head_hidden=64, force_hidden=16, emg_hidden=16)
OPEN_PILOT = replace(PILOT, body="open")
OPEN_TINY = replace(TINY, body="open")


def open_inputs():
    """{name: path} of the open body's run-time inputs (mdfm/train/bodies.OPEN_INPUTS)."""
    from ..train.bodies import OPEN as _OPEN, inputs
    return inputs(_OPEN)


def open_shape_prior(path=None):
    """The open model's phenotype prior: the pooled row of reports/open_body_priors/priors.json in all three sex rows
    (decided on 2026-11-13: the gender value does not carry sex, so the prior is not sex-conditioned)."""
    import json
    path = open_inputs()["priors"] if path is None else path
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)["phenotype_prior"]["neutral"]
    mean = torch.as_tensor(sp["mean"], dtype=torch.float64)
    cov = torch.as_tensor(sp["cov"], dtype=torch.float64)
    return PhenotypePrior(mean.expand(3, -1).clone(), cov.expand(3, -1, -1).clone())


# ---------------------------------------------------------------- containers
@dataclass
class Nuisance:
    """Nuisance of W windows: imu IMUNuisance (W, 14) by canonical site, camera CameraNuisance (W,),
    pose3d_log_sigma (W,) log m."""
    imu: IMUNuisance = None
    camera: CameraNuisance = None
    pose3d_log_sigma: torch.Tensor = None

    def tensors(self):
        """{name: tensor} of every tensor field, names 'imu.<field>', 'camera.<field>', 'pose3d_log_sigma'."""
        out = {}
        for part in ("imu", "camera"):
            obj = getattr(self, part)
            if obj is not None:
                for f in fields(obj):
                    v = getattr(obj, f.name)
                    if v is not None:
                        out[f"{part}.{f.name}"] = v
        if self.pose3d_log_sigma is not None:
            out["pose3d_log_sigma"] = self.pose3d_log_sigma
        return out

    def with_tensors(self, values):
        """A Nuisance with the fields named in values ({name: tensor}, as tensors()) replaced."""
        parts = {}
        for part in ("imu", "camera"):
            obj = getattr(self, part)
            if obj is not None:
                upd = {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith(part + ".")}
                parts[part] = replace(obj, **upd)
            else:
                parts[part] = None
        return Nuisance(parts["imu"], parts["camera"], values.get("pose3d_log_sigma", self.pose3d_log_sigma))

    def detach(self):
        return self.with_tensors({k: v.detach() for k, v in self.tensors().items()})


@dataclass
class Posterior:
    """Amortized (or refined) posterior of W windows: mu_z, logvar_z (W, P, 32); mu_b, logvar_b (W, 16);
    nuis (Nuisance); mask (W, P) bool, the valid patches."""
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    mu_b: torch.Tensor
    logvar_b: torch.Tensor
    nuis: Nuisance
    mask: torch.Tensor

    def detach(self):
        return Posterior(self.mu_z.detach(), self.logvar_z.detach(), self.mu_b.detach(), self.logvar_b.detach(),
                         self.nuis.detach(), self.mask)


class KnotBody(NamedTuple):
    """A body on the knot clock of W windows: rot6d (W, T, 22, 6), transl (W, T, 3) m in the window frame,
    contact_logit (W, T, 2) or None (contacts then from the kinematics, kernels.force.contact_prob), beta
    (W, 16), gender (W,) int64 SkeletonBank codes, lengths (W,) int64 valid knots."""
    rot6d: torch.Tensor
    transl: torch.Tensor
    contact_logit: torch.Tensor
    beta: torch.Tensor
    gender: torch.Tensor
    lengths: torch.Tensor

    def to(self, device=None, dtype=None):
        def mv(x):
            if x is None:
                return None
            return x.to(device=device, dtype=dtype if x.is_floating_point() else None)
        return KnotBody(*(mv(x) for x in self))


class Samples(NamedTuple):
    """n posterior draws at every knot: rot6d (n, W, T, 22, 6), transl (n, W, T, 3), contact_logit (n, W, T,
    2), pos_logvar (n, W, T, 22), joints (n, W, T, 22, 3) m in the window frame, beta (n, W, 16); lengths
    (W,) valid knots (entries past a window's length are the fill values)."""
    rot6d: torch.Tensor
    transl: torch.Tensor
    contact_logit: torch.Tensor
    pos_logvar: torch.Tensor
    joints: torch.Tensor
    beta: torch.Tensor
    lengths: torch.Tensor


@dataclass
class Points:
    """Kernel points of one modality in a micro-step: w (N,) int64 window, t (N,) float32 s from the window
    start, obs and valid (N, ...) in contract units, weight (N,) float32 (the stream's samples over the
    sampled ones), group (N,) int64 stream-window id, extra {name: (N, ...)} (pose3d root, root_valid; emg
    muscle, side)."""
    w: torch.Tensor
    t: torch.Tensor
    obs: torch.Tensor
    valid: torch.Tensor
    weight: torch.Tensor
    group: torch.Tensor
    extra: dict = field(default_factory=dict)

    def select(self, idx, weight=None):
        return Points(self.w[idx], self.t[idx], self.obs[idx], self.valid[idx],
                      self.weight[idx] if weight is None else weight, self.group[idx],
                      {k: v[idx] for k, v in self.extra.items()})

    def to(self, device):
        return Points(self.w.to(device), self.t.to(device), self.obs.to(device), self.valid.to(device),
                      self.weight.to(device), self.group.to(device), {k: v.to(device) for k, v in self.extra.items()})


@dataclass
class Batch:
    """A micro-step (module docstring): tokens (tokenize.TokenBatch); per window (W,) int64 sex, task,
    gender (SkeletonBank codes), source_id (into sources), person (group ids of person_id), lengths (knots
    of the window, the last at (lengths - 1) / rate s); points {modality: Points}; win {name: (W, ...)}
    per-window kernel fields (imu_family, imu_grav, imu_present, p3_scale, p3_unit, p3_zup, p2_scale,
    f_mask, f_offset); knots (w, k) (M,) int64, the knots to decode (the Hermite supports of every point,
    plus extra ones); truth KnotBody or None; sources and families, the name tuples the ids refer to;
    nuis_fields, the vocabularies of the tokens' nuisance codes (None: tokenize.NUIS_FIELDS)."""
    tokens: TK.TokenBatch
    sex: torch.Tensor
    task: torch.Tensor
    gender: torch.Tensor
    source_id: torch.Tensor
    person: torch.Tensor
    lengths: torch.Tensor
    points: dict
    win: dict
    knots: tuple
    truth: KnotBody = None
    sources: tuple = SOURCES
    families: tuple = ("generic", "xsens13")
    rate: float = RATE
    nuis_fields: dict = None

    @property
    def n_windows(self):
        return int(self.sex.shape[0])

    @property
    def T(self):
        return int(self.lengths.max())

    def cond(self, beta):
        """Cond of the motion prior with the shape coefficients beta (W, 16)."""
        return Cond(self.sex.to(beta.device), self.task.to(beta.device), beta)

    def to(self, device):
        mv = lambda x: x.to(device)  # noqa: E731
        return replace(self, tokens=self.tokens.to(device), sex=mv(self.sex), task=mv(self.task),
                       gender=mv(self.gender), source_id=mv(self.source_id), person=mv(self.person),
                       lengths=mv(self.lengths), points={k: p.to(device) for k, p in self.points.items()},
                       win={k: mv(v) for k, v in self.win.items()}, knots=tuple(mv(x) for x in self.knots),
                       truth=None if self.truth is None else self.truth.to(device))

    def subsample(self, per_stream, generator=None):
        """The batch with at most per_stream points per stream-window drawn uniformly without replacement
        (a torch.Generator on any device may be given; the draws are made on its device), each kept point's
        weight times n_group / n_kept, and the knots reduced to the Hermite supports of the kept points."""
        pts = {}
        rdev = generator.device if generator is not None else torch.device("cpu")
        for m, p in self.points.items():
            n = p.w.numel()
            if not n:
                pts[m] = p
                continue
            g = p.group
            r = torch.rand(n, generator=generator, device=rdev).to(g.device)
            order = torch.argsort(r)
            order = order[torch.argsort(g[order], stable=True)]
            gs = g[order]
            counts = torch.bincount(g)
            start = torch.cumsum(counts, 0) - counts
            rank = torch.arange(n, device=g.device) - start[gs]
            idx = torch.sort(order[rank < per_stream])[0]
            kept = torch.clamp(counts, max=per_stream).to(p.weight.dtype)
            wt = p.weight[idx] * counts[g[idx]].to(p.weight.dtype) / kept[g[idx]]
            pts[m] = p.select(idx, wt)
        return replace(self, points=pts, knots=support_knots(pts, self.lengths, self.rate))


# ---------------------------------------------------------------- batch builder
def mount_families():
    """The mount family order of reports/mount_priors/priors.json."""
    from ..kernels.nuisance import PRIORS_PATH
    import json
    with open(PRIORS_PATH, encoding="utf-8") as f:
        return tuple(json.load(f)["family_order"])


def mount_family(layout, families):
    """Id in families of the mount prior of a layout family (factory/layouts.FAMILY_RULES; generic when
    the layout is unknown or has no measured prior)."""
    name = LY.FAMILY_RULES.get(layout, {}).get("mount", "generic")
    return families.index(name) if name in families else families.index("generic")


def sample_indices(n, per_stream, segment, rng):
    """Sorted sample indices of a stream of n samples: all when per_stream is None or at least n, else the
    union of max(1, per_stream // segment) segments of `segment` consecutive samples at uniform starts."""
    if per_stream is None or per_stream >= n or n <= segment:
        return np.arange(n)
    m = max(1, int(per_stream) // int(segment))
    starts = rng.integers(0, n - segment + 1, size=m)
    return np.unique((starts[:, None] + np.arange(segment)[None]).reshape(-1))


def support_knots(points, lengths, rate=RATE):
    """(w, k) (M,) int64: the unique knots that the Hermite values at every point of `points` read, in
    float32 and in float64 alike: t x rate can floor to different segments in the two precisions on a knot
    (float32 0.29 s x 100 is 29.0, float64 28.99999917), and the kinematics run in the body's dtype."""
    ws = [p.w for p in points.values() if p.w.numel()]
    if not ws:
        e = torch.zeros(0, dtype=torch.int64, device=lengths.device)
        return e, e
    w = torch.cat(ws)
    t = torch.cat([p.t for p in points.values() if p.w.numel()])
    T = int(lengths.max())
    L = lengths.to(w.device)[w]
    k = torch.cat([support(t.float(), L, rate), support(t.double(), L, rate)], 1)
    key = torch.unique((w[:, None] * T + k).reshape(-1))
    return key // T, key % T


def _single(streams, modality, i):
    s = [x for x in streams if x.role == ("target" if modality in ("force", "emg") else "input")
         and x.modality == modality]
    if len(s) > 1:
        raise ValueError(f"window {i} holds {len(s)} {modality} streams; the pilot model takes one per window")
    return s[0] if s else None


def batch_from_windows(windows, sources=SOURCES, points_per_stream=None, segment=SEGMENT, extra_knots=0, gen=None,
                       rate=RATE, row_len=None, families=None, allow_eval=False, nuis_fields=None):
    """Batch of windows.Window objects (class docstring): tokens of the input streams; kernel points of
    every stream at its native (band-copy) samples, all of them or segments of `segment` samples totalling
    about points_per_stream per stream (sample_indices, numpy Generator or seed gen); knots: the Hermite
    supports of the points plus extra_knots random knots per window; truth from Window.body when every
    window carries one (synthetic). Knots per window: the truth's lengths (a sample beyond its last knot
    raises), else ceil(duration x rate) + 1, so that the last knot lies at or after every sample (a sample
    past the last knot would read the end knot's value with zero derivatives, body/hermite.py). At most one
    stream per modality per window; a pose stream needs a positive scale, a pose3d stream a known unit. A
    window whose source or person id is a CODY name (data.canonical.is_cody) is refused (canonical.Refused)
    unless allow_eval=True (an evaluation stage: CODY never trains); the namespace test on the index belongs
    to the record reader (canonical.check), which every real window has passed. A source outside `sources`
    raises ValueError. A missing task covariate reads as the vocabulary's unknown. nuis_fields: the nuisance
    descriptor vocabularies of the tokens (tokenize.nuisance_fields; None: tokenize.NUIS_FIELDS), those of the
    model that reads the batch (MFM.nuis_fields), carried by the Batch."""
    rng = gen if isinstance(gen, np.random.Generator) else np.random.default_rng(gen)
    families = mount_families() if families is None else tuple(families)
    wins = list(windows)
    W = len(wins)
    tokens = TK.tokenize(wins, row_len=row_len, nuis_fields=nuis_fields)
    lengths = np.zeros(W, np.int64)
    sex, task, gender, source_id = (np.zeros(W, np.int64) for _ in range(4))
    win = {"imu_family": np.full(W, families.index("generic"), np.int64), "imu_grav": np.zeros(W, bool),
           "imu_present": np.zeros((W, N_SITES), bool), "p3_scale": np.ones(W, np.float32),
           "p3_unit": np.ones(W, np.float32), "p3_zup": np.zeros(W, bool), "p2_scale": np.ones(W, np.float32),
           "f_mask": np.zeros((W, 2), bool), "f_offset": np.zeros((W, 2), np.float32)}
    parts = {m: [] for m in MODALITIES}
    n_group = 0
    for i, wd in enumerate(wins):
        if not allow_eval and is_cody(wd.source, wd.person_id):
            raise Refused(f"window {i}: CODY is evaluation-only ({wd.source} {wd.person_id})")
        if wd.source not in sources:
            raise ValueError(f"window {i}: source {wd.source!r} is not one of {sources}")
        source_id[i] = sources.index(wd.source)
        sex[i], task[i] = int(wd.cov.get("sex", 2)), int(wd.cov.get("task", TASK_UNKNOWN))
        gender[i] = int(wd.body["gender"]) if wd.body is not None else sex[i]
        t_last = max([float(np.max(s.t)) for s in wd.streams if len(s.t)] + [0.0])
        if wd.body is not None:
            lengths[i] = int(wd.body["lengths"])
            if t_last > (lengths[i] - 1 + 1e-3) / rate:
                raise ValueError(f"window {i}: a sample at {t_last:.4f} s lies beyond the last truth knot at "
                                 f"{(lengths[i] - 1) / rate:.4f} s")
        else:
            lengths[i] = int(math.ceil(max(wd.duration, t_last) * rate - 1e-6)) + 1
        for m in MODALITIES:
            s = _single(wd.streams, m, i)
            if s is None or not len(s.t):
                continue
            idx = sample_indices(len(s.t), points_per_stream, segment, rng)
            n = len(idx)
            base = {"w": np.full(n, i, np.int64), "t": np.asarray(s.t, np.float32)[idx],
                    "weight": np.full(n, len(s.t) / n, np.float32), "group": np.full(n, n_group, np.int64)}
            n_group += 1
            vals, valid, cm = np.asarray(s.values), np.asarray(s.valid, bool), np.asarray(s.chan_mask, bool)
            meta = s.meta or {}
            if m == "imu":
                sites = np.asarray(meta["site_ids"], np.int64)
                obs = np.zeros((n, N_SITES, 6), np.float32)
                ok = np.zeros((n, N_SITES, 6), bool)
                obs[:, sites] = vals[idx][..., :6]
                ok[:, sites] = valid[idx][..., None] & cm[None, :, :6]
                win["imu_family"][i] = mount_family(meta.get("layout"), families)
                win["imu_grav"][i] = meta.get("acc_content") == "linear_no_gravity"
                win["imu_present"][i, sites] = True
                base.update(obs=obs, valid=ok)
            elif m in ("pose3d", "pose2d"):
                cols = 3 if m == "pose3d" else 2
                base.update(obs=vals[idx].astype(np.float32), valid=valid[idx] & cm[None, :, :cols].all(-1))
                if not float(meta.get("scale") or 0.0) > 0:
                    raise ValueError(f"window {i}: {m} stream without a positive scale")
                if m == "pose3d":
                    if meta.get("unit_to_si") is None:
                        raise ValueError(f"window {i}: pose3d stream in an unknown unit (unit_to_si None)")
                    win["p3_scale"][i] = float(meta["scale"])
                    win["p3_unit"][i] = float(meta["unit_to_si"])
                    if meta.get("root") is not None:
                        base["root"] = np.asarray(meta["root"], np.float32)[idx]
                        base["root_valid"] = np.asarray(meta["root_valid"], bool)[idx]
                else:
                    win["p2_scale"][i] = float(meta["scale"])
            elif m == "force":
                base.update(obs=vals[idx][..., 8].astype(np.float32), valid=valid[idx] & cm[None, :, 8])
                win["f_mask"][i] = cm[:, 8]
            else:
                mus, side, modelled = channel_table(meta.get("channels") or [])
                if len(mus) != vals.shape[1]:
                    raise ValueError(f"window {i}: EMG stream of {vals.shape[1]} channels with a channel table "
                                     f"of {len(mus)}")
                base.update(obs=vals[idx][..., 0].astype(np.float32),
                            valid=valid[idx] & cm[None, :, 0] & modelled[None],
                            muscle=np.broadcast_to(mus, (n, len(mus))).copy(),
                            side=np.broadcast_to(side, (n, len(side))).copy())
            parts[m].append(base)
    T = int(lengths.max())
    points = {}
    for m, items in parts.items():
        if not items:
            continue
        if m == "pose3d" and any("root" in b for b in items):
            for b in items:
                n = len(b["w"])
                b.setdefault("root", np.zeros((n, 3), np.float32))
                b.setdefault("root_valid", np.zeros(n, bool))
        if m == "emg":
            C = max(b["obs"].shape[1] for b in items)
            for b in items:
                pad = C - b["obs"].shape[1]
                if pad:
                    n = len(b["w"])
                    b["obs"] = np.concatenate([b["obs"], np.zeros((n, pad), np.float32)], 1)
                    b["valid"] = np.concatenate([b["valid"], np.zeros((n, pad), bool)], 1)
                    b["muscle"] = np.concatenate([b["muscle"], np.full((n, pad), -1, np.int64)], 1)
                    b["side"] = np.concatenate([b["side"], np.full((n, pad), -1, np.int64)], 1)
        cat = lambda k: torch.from_numpy(np.ascontiguousarray(np.concatenate([b[k] for b in items])))  # noqa: E731
        extra = {k: cat(k) for k in ("root", "root_valid", "muscle", "side") if k in items[0]}
        points[m] = Points(cat("w"), cat("t"), cat("obs"), cat("valid"), cat("weight"), cat("group"), extra)
    lengths_t = torch.from_numpy(lengths)
    knots = support_knots(points, lengths_t, rate)
    if extra_knots:
        ek = [(i, k) for i in range(W) for k in rng.integers(0, lengths[i], size=int(extra_knots))]
        key = torch.unique(torch.cat([knots[0] * T + knots[1], torch.tensor([i * T + k for i, k in ek])]))
        knots = (key // T, key % T)
    truth = None
    if W and all(wd.body is not None for wd in wins):
        def pad(x, L):
            x = torch.as_tensor(np.asarray(x[:L]), dtype=torch.float32)
            return torch.cat([x, x[-1:].expand((T - L,) + tuple(x.shape[1:]))]) if L < T else x
        truth = KnotBody(torch.stack([pad(wd.body["rot6d"], int(lengths[i])) for i, wd in enumerate(wins)]),
                         torch.stack([pad(wd.body["transl"], int(lengths[i])) for i, wd in enumerate(wins)]), None,
                         torch.stack([torch.as_tensor(np.asarray(wd.body["betas"]), dtype=torch.float32)
                                      for wd in wins]),
                         torch.from_numpy(gender.copy()), lengths_t.clone())
    _, person = np.unique(np.asarray([str(wd.person_id) for wd in wins]), return_inverse=True)
    return Batch(tokens=tokens, sex=torch.from_numpy(sex), task=torch.from_numpy(task), gender=torch.from_numpy(gender),
                 source_id=torch.from_numpy(source_id), person=torch.as_tensor(person.reshape(-1), dtype=torch.int64),
                 lengths=lengths_t, points=points,
                 win={k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in win.items()},
                 knots=knots, truth=truth, sources=tuple(sources), families=families, rate=float(rate),
                 nuis_fields=None if nuis_fields is None else TK.nuisance_fields(nuis_fields))


# ---------------------------------------------------------------- the model
def _no_autocast(device):
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


class MFM(nn.Module):
    """The stage-1 model (module docstring). MFM(cfg=PILOT, priors=None): priors a kernels.nuisance
    MountPriors (default reports/mount_priors/priors.json). With cfg.names (a train/names.NameTables), the
    task embedding of the motion prior holds one row per task of its vocabulary and the tokenizer's nuisance
    tables follow its layout, device and camera names; ValueError when its sources differ from cfg.sources, its
    'unknown' task is not the motion prior's condition-dropout row (prior.TASK_UNKNOWN) or its mount families
    differ from the priors' (module docstring, Name tables)."""

    def __init__(self, cfg=PILOT, priors=None):
        super().__init__()
        self.cfg = cfg
        self.rate = float(cfg.rate)
        names = cfg.names
        n_tasks, nuis_fields = len(TASKS), None
        if names is not None:
            if tuple(names.sources) != tuple(cfg.sources):
                raise ValueError(f"name-table sources {names.sources} differ from the model's {cfg.sources}")
            tasks = tuple(names.tasks())
            if "unknown" not in tasks or tasks.index("unknown") != TASK_UNKNOWN:
                raise ValueError(f"task 'unknown' of {names.vocabulary_path} is not at id {TASK_UNKNOWN}, the motion "
                                 f"prior's condition-dropout row")
            n_tasks, nuis_fields = len(tasks), names.nuis_fields()
        self.names = names
        body = str(cfg.body)
        if body not in ("smplh", "open"):
            raise ValueError(f"body {body!r}, expected 'smplh' or 'open'")
        self.body = body
        self.tokenizer = TK.Tokenizer(cfg.d, cfg.stem_hidden, nuis_fields=nuis_fields)
        self.encoder = Encoder(cfg.d, cfg.layers, cfg.heads, cfg.mlp_ratio)
        if body == "open":
            from ..kernels.nuisance import MountPriors
            from .decoder import anchor_from_report, mean_pose_from_report, transl0_from_report
            inp = open_inputs()
            self.shape_prior = open_shape_prior(inp["priors"])
            self.latent_head = LatentHead(cfg.d, cfg.d_z)     # registered after the shape prior, as in v1
            self.shape_head = PhenotypeHead(cfg.d, mu0=self.shape_prior.mean[2], logvar0=self.shape_prior.logvar0(2))
            n_shape = self.shape_prior.k
            dec_kw = {"mean_pose": mean_pose_from_report(inp["priors"]), "transl0": transl0_from_report(
                inp["window_priors"]), "anchor": anchor_from_report(inp["window_priors"])}
            priors = MountPriors.load(inp["mount_priors"]) if priors is None else priors
        else:
            self.shape_prior = ShapePrior()
            self.latent_head = LatentHead(cfg.d, cfg.d_z)
            self.shape_head = ShapeHead(cfg.d, N_BETAS, mu0=self.shape_prior.mean[2],
                                        logvar0=self.shape_prior.logvar0(2))
            n_shape, dec_kw = N_BETAS, {}
        self.n_shape = int(n_shape)
        self.prior = MotionPrior(cfg.d_z, cfg.prior_d, cfg.prior_layers, cfg.prior_heads, cfg.prior_span,
                                 n_tasks=n_tasks, n_betas=n_shape)
        self.decoder = BodyDecoder(cfg.dec_d, cfg.dec_layers, cfg.dec_heads, cfg.d_z, n_betas=n_shape,
                                   n_fourier=cfg.n_fourier, rate=cfg.rate, **dec_kw)
        self.heads = NuisanceHeads(cfg.d, cfg.head_hidden, priors)
        if names is not None and tuple(self.heads.families) != tuple(names.mount_families):
            raise ValueError(f"name-table mount families {names.mount_families} differ from the priors' "
                             f"{self.heads.families}")
        self.force_head = ForceHead(cfg.force_hidden)
        self.emg_decoder = EMGDecoder(cfg.emg_hidden)
        self.source_offsets = SourceOffsets(len(cfg.sources), tuple(cfg.sources).index("synthetic"))
        self.force_log_sigma = nn.Parameter(torch.tensor(FORCE_LOG_SIGMA0))
        self.emg_log_sigma = nn.Parameter(torch.tensor(EMG_LOG_SIGMA0))
        if not cfg.skeleton:
            self.skeleton = None
        elif body == "open":
            from ..body.openbody import LatentSkeletonBank
            self.skeleton = LatentSkeletonBank(torch.float32)
        else:
            from ..body.backend import require_reference
            require_reference("MFM with the SMPL+H skeleton (the v1 model)")
            from ..body.smplh import SkeletonBank      # the v1 branch only: no SMPL+H module at the top
            self.skeleton = SkeletonBank(torch.float32)
        # the optional JEPA module (attach_jepa), registered last: None holds no parameter, buffer or state
        self.add_module("jepa", None)

    def attach_jepa(self, width=256, depth=4, heads=4):
        """Attach the JEPA module of the auxiliary loss (model/jepa.JEPA: an EMA target copy of the tokenizer and
        the encoder, and the predictor of `depth` blocks of `width` with `heads` heads) as self.jepa, registered
        after every other module so that its parameters come last; returns it. The training loop calls it when
        the config's jepa section is enabled (docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2).

        State dicts. With the module attached, a state dict without any jepa.* entry (a checkpoint written
        without JEPA) loads: the target then copies the loaded tokenizer and encoder (JEPA.sync) and the
        predictor keeps its initialization; a state dict with only some of the jepa.* entries is refused as
        usual. Without the module (self.jepa None), the jepa.* entries of a state dict are not read, so the
        weights of a JEPA run load into a model without it (evaluation)."""
        from .jepa import JEPA
        if self.jepa is not None:
            raise ValueError("a JEPA module is already attached")
        self.jepa = JEPA(self.tokenizer, self.encoder, width, depth, heads).to(
            device=self.device, dtype=self.latent_head.proj.weight.dtype)
        if not getattr(self, "_jepa_hook", False):
            self.register_load_state_dict_post_hook(_jepa_post_load)
            self._jepa_hook = True
        return self.jepa

    @property
    def device(self):
        return self.latent_head.proj.weight.device

    @property
    def nuis_fields(self):
        """The nuisance descriptor vocabularies of the tokenizer (tokenize.NUIS_FIELDS for a stage-1 model), the
        nuis_fields of the batches it reads (batch_from_windows)."""
        return self.tokenizer.nuisance.fields

    def _check(self, batch):
        if tuple(batch.sources) != tuple(self.cfg.sources):
            raise ValueError(f"batch sources {batch.sources} differ from the model's {self.cfg.sources}")
        if tuple(batch.families) != tuple(self.heads.families):
            raise ValueError(f"batch mount families {batch.families} differ from the model's {self.heads.families}")
        mine = self.tokenizer.nuisance.fields
        theirs = TK.NUIS_FIELDS if getattr(batch, "nuis_fields", None) is None else batch.nuis_fields
        if theirs is not mine and dict(theirs) != dict(mine):
            raise ValueError("batch nuisance vocabularies differ from the model's (tokenize the windows with "
                             "nuis_fields=model.nuis_fields)")

    # ------------------------------------------------------------ encoder side
    def encode(self, batch):
        """Posterior of the batch's windows (module docstring)."""
        self._check(batch)
        tb = batch.tokens
        x = self.tokenizer(tb)
        h = self.encoder(x, tb.t.to(x.device), tb.mask().to(x.device))
        return self.posterior_from_hidden(h, batch)

    def posterior_from_hidden(self, h, batch):
        """Posterior from encoder outputs h (R, L, d): latent and subject positions for the Gaussian heads,
        nuisance positions for the nuisance heads."""
        tb = batch.tokens
        R, L, d = h.shape
        flat = h.reshape(R * L, d)
        lat = tb.lat_idx.to(h.device)
        mask = lat >= 0
        mu_z, logvar_z = self.latent_head(flat[lat.clamp_min(0)])
        mu_b, logvar_b = self.shape_head(flat[tb.subj_idx.to(h.device)])
        return Posterior(mu_z, logvar_z, mu_b, logvar_b, self.nuisance_from_hidden(h, batch), mask)

    def nuisance_positions(self, batch):
        """Flat positions (Nn,) row x L + position of the nuisance tokens, the only positions the heads read."""
        tb = batch.tokens
        return tb.nuis_row * tb.t.shape[1] + tb.nuis_pos

    def nuisance_from_hidden(self, h, batch):
        """Nuisance from the encoder outputs at the nuisance-token positions: IMU site tokens into (W, 14),
        pose2d tokens into the camera head, pose3d tokens into the pose3d head; windows without a token
        get the head's output at a zero input."""
        tb = batch.tokens
        W, d = batch.n_windows, h.shape[-1]
        dev = h.device
        hn = h.reshape(-1, d)[self.nuisance_positions(batch).to(dev)]
        kind = tb.nuis_desc[:, KIND_COL].to(dev)
        win, site = tb.nuis_win.to(dev), tb.nuis_site.to(dev)
        imu = kind == TK.NUIS_FIELDS["kind"].index("imu_site")
        dense = torch.zeros(W, N_SITES, d, dtype=h.dtype, device=dev).index_put((win[imu], site[imu]), hn[imu])
        out = {}
        for name in ("pose2d", "pose3d"):
            sel = kind == TK.NUIS_FIELDS["kind"].index(name)
            out[name] = torch.zeros(W, d, dtype=h.dtype, device=dev).index_put((win[sel],), hn[sel])
        return Nuisance(self.heads.imu(dense, batch.win["imu_family"].to(dev)), self.heads.camera(out["pose2d"]),
                        self.heads.pose3d(out["pose3d"]))

    # ------------------------------------------------------------ decoder side
    def decode(self, z, beta, knots, n_patches=None, lengths=None):
        """decoder.BodyOut at knots (w, k) from latents z (W, P, 32) and betas (W, 16), with the root grid of the
        windows of lengths (W,) knots (required, ValueError without: the root is integrated from the centre knot
        lengths // 2)."""
        return self.decoder(z, beta, knots, n_patches, lengths)

    def dense(self, out, beta, batch):
        """KnotBody (W, T) of decoded knots out (BodyOut) with the betas `beta` of the kinematics; undecoded
        knots hold the mean pose (matrix_to_rot6d of decoder.mean_R, the rotations the decoder starts from), the
        decoder's transl0 and zero contact logits."""
        W, T = batch.n_windows, batch.T
        dt = out.rot6d.dtype
        fill = matrix_to_rot6d(self.decoder.mean_R).to(dt)
        return KnotBody(scatter_dense(out.rot6d, out.w, out.k, W, T, fill),
                        scatter_dense(out.transl, out.w, out.k, W, T, self.decoder.transl0.to(dt)),
                        scatter_dense(out.contact_logit, out.w, out.k, W, T),
                        beta, batch.gender.to(out.rot6d.device), batch.lengths.to(out.rot6d.device))

    # ------------------------------------------------------------ kernels
    def kinematics(self, body, batch):
        """(BodyKin at every kernel point of the batch from one call, {modality: slice}, rest joints (W, 22,
        3)), in the body's dtype (float32 or float64), autocast off."""
        dev = body.rot6d.device
        with _no_autocast(dev):
            dt = torch.float64 if body.rot6d.dtype == torch.float64 else torch.float32
            rest = self.skeleton.rest(body.beta.to(dt), body.gender.to(dev))
            kb = KnotBatch(body.rot6d.to(dt), body.transl.to(dt), rest, self.rate, body.lengths.to(dev))
            mods = [m for m in MODALITIES if m in batch.points and batch.points[m].w.numel()]
            if not mods:
                return None, {}, rest
            sl, a = {}, 0
            for m in mods:
                n = batch.points[m].w.numel()
                sl[m] = slice(a, a + n)
                a += n
            w = torch.cat([batch.points[m].w for m in mods]).to(dev)
            t = torch.cat([batch.points[m].t for m in mods]).to(dev, dt)
            return kb.kinematics(w, t), sl, rest

    def _force(self, kin, logits, sex):
        feet = torch.tensor(FEET, device=kin.p.device)
        c = kin.p[:, feet].mean(2)                                       # (N, 2, 3)
        v = kin.v[:, feet].mean(2)
        pred, fl, _ = self.force_head(com_acc(kin.a, sex), logits, c[..., 1], v)
        return pred, fl

    def _contact_logits(self, body, kin, p):
        if body.contact_logit is not None:
            return hermite_gather(body.contact_logit.to(kin.p.dtype), p.w, p.t.to(kin.p.dtype), self.rate,
                                  body.lengths)
        cp = contact_prob(kin.p, kin.v).clamp(1e-6, 1 - 1e-6)
        return torch.log(cp) - torch.log1p(-cp)

    def render(self, body, batch, nuis, kin=None):
        """Predictions {modality: dict} at every kernel point (module docstring); kin: the output of
        kinematics(body, batch) when already computed (refinement with the body held fixed)."""
        self._check(batch)
        kinall, sl, rest = self.kinematics(body, batch) if kin is None else kin
        pred = {}
        dev = body.rot6d.device
        with _no_autocast(dev):
            W = batch.n_windows
            for m, s in sl.items():
                k = BodyKin(*(x[s] for x in kinall))
                p = batch.points[m]
                w = p.w.to(dev)
                if m == "imu":
                    joints, off0 = site_offsets(rest)
                    fam = batch.win["imu_family"].to(dev)
                    R0 = self.heads.priors().rotation(fam).to(k.p.dtype)
                    grav = batch.win["imu_grav"].to(dev)[w][:, None]
                    acc, gyro = nuis.imu.predict(k, w, joints.to(dev), off0, R0, gravity_removed=grav)
                    pred[m] = {"acc": acc, "gyro": gyro, "log_scale": nuis.imu.log_scale[w]}
                elif m in ("pose3d", "pose2d"):
                    offs = self.source_offsets(batch.source_id.to(dev)).to(k.p.dtype)
                    p15 = to_canonical15(k.p, k.R, offs[w])
                    if m == "pose3d":
                        pred[m] = {"joints": p15, "pelvis": k.p[:, 0], "log_sigma": nuis.pose3d_log_sigma}
                    else:
                        ones = torch.ones(len(w), dtype=k.p.dtype, device=dev)
                        target = window_mean(p15[:, 0].detach(), ones, w, W)
                        c = nuis.camera
                        cam = camera_from_params(c.rot6d.to(k.p.dtype), c.log_distance.to(k.p.dtype),
                                                 c.log_focal.to(k.p.dtype), target)
                        pred[m] = {"joints": p15, "camera": cam, "sigma_px": torch.exp(c.log_sigma_px)}
                elif m == "force":
                    sex = batch.sex.to(dev)[w]
                    bw, fl = self._force(k, self._contact_logits(body, k, p), sex)
                    pred[m] = {"bw": bw, "flight": fl}
                else:
                    kd = BodyKin(*(x.detach() for x in k))
                    sex = batch.sex.to(dev)[w]
                    bw, _ = self._force(kd, self._contact_logits(body, kd, p).detach(), sex)
                    tau = leg_torques(kd, bw.detach(), sex=sex)
                    ang, rates = joint_angles(kd)
                    log_env, ok = self.emg_decoder(moment_channels(tau), ang, rates, p.extra["muscle"].to(dev),
                                                   p.extra["side"].to(dev))
                    pred[m] = {"log_env": log_env, "mask": ok}
        return pred

    def nll_terms(self, pred, batch):
        """{modality: (W,)} kernel negative log-likelihoods (module docstring), autocast off."""
        W = batch.n_windows
        terms = {}
        for m, pr in pred.items():
            p = batch.points[m]
            dev = next(iter(pr.values())).device
            with _no_autocast(dev):
                terms[m] = self._nll(m, pr, p, {k: v.to(dev) for k, v in batch.win.items()}, W, dev)
        return terms

    def _nll(self, m, pr, p, win, W, dev):
        w, wt = p.w.to(dev), p.weight.to(dev)
        if m == "imu":
            return imu_nll((pr["acc"], pr["gyro"]), p.obs.to(dev), p.valid.to(dev), pr["log_scale"], wt, w, W)
        if m == "pose3d":
            j = pr["joints"]
            nll, yaw = pose3d_nll(j, p.obs.to(dev, j.dtype), win["p3_scale"], win["p3_unit"], p.valid.to(dev),
                                  pr["log_sigma"], w, W, z_up=win["p3_zup"], weight=wt, return_yaw=True)
            if "root" in p.extra:
                nll = nll + root_nll(pr["pelvis"], p.extra["root"].to(dev, j.dtype), p.extra["root_valid"].to(dev),
                                     w, W, yaw=yaw.detach(), weight=wt)
            return nll
        if m == "pose2d":
            j = pr["joints"]
            return pose2d_nll(j, pr["camera"], p.obs.to(dev, j.dtype), win["p2_scale"], p.valid.to(dev), w, W,
                              sigma0=pr["sigma_px"], weight=wt)
        if m == "force":
            return force_nll(pr["bw"], p.obs.to(dev), w, win["f_mask"], p.valid.to(dev), self.force_log_sigma,
                             weight=wt, offset=win["f_offset"], flight=pr["flight"])
        return emg_nll(pr["log_env"], p.obs.to(dev), p.valid.to(dev) & pr["mask"], w, self.emg_log_sigma, weight=wt,
                       n_windows=W)

    def log_lik(self, pred, batch):
        """log p(observations | body, nuisance) (W,): minus the sum of nll_terms."""
        terms = list(self.nll_terms(pred, batch).values())
        if not terms:
            return torch.zeros(batch.n_windows, device=self.device)
        return -sum(terms)

    def nuis_log_prior(self, nuis, batch):
        """log p(nuisance | layout family) (W,): mount tilt and yaw and IMU biases at the present sites
        (kernels.nuisance.MountPriors.log_prior); the camera and the pose3d scale carry flat priors."""
        dev = nuis.imu.mount_res6d.device
        pr = self.heads.priors()
        with _no_autocast(dev):
            return pr.log_prior(nuis.imu, batch.win["imu_family"].to(dev), None, mask=batch.win["imu_present"].to(dev))

    def nuisance_scales(self, nuis, batch):
        """{name: scale} of the nuisance tensors (Nuisance.tensors names) whose natural unit is not 1: the IMU
        biases in units of the device prior's standard deviation of each window's family and site (W, 14,
        3); refinement steps these in those units."""
        pr = self.heads.priors()
        fam = batch.win["imu_family"].to(nuis.imu.bias_acc.device)
        return {"imu.bias_acc": pr.per_site(pr.bias_acc_sd, fam).to(nuis.imu.bias_acc.dtype),
                "imu.bias_gyro": pr.per_site(pr.bias_gyro_sd, fam).to(nuis.imu.bias_gyro.dtype)}

    def log_joint(self, z, beta, nuis, batch, body=None, kin=None):
        """log p(x, z, beta, nuis) (W,) (module docstring); body a KnotBody held fixed (z and beta unread),
        kin its kinematics(body, batch) when already computed."""
        if body is None:
            out = self.decode(z, beta, batch.knots, batch.tokens.n_patches.to(z.device), batch.lengths.to(z.device))
            body = self.dense(out, beta, batch)
        lj = self.log_lik(self.render(body, batch, nuis, kin), batch)
        lj = lj + self.nuis_log_prior(nuis, batch).to(lj.dtype)
        if z is not None:
            mask = batch.tokens.lat_idx.to(z.device) >= 0
            lj = lj + self.prior.log_prob(z, batch.cond(beta), mask, dropout=False).to(lj.dtype)
            lj = lj + self.shape_prior.log_prob(beta, batch.gender.to(beta.device)).to(lj.dtype)
        return lj

    @torch.no_grad()
    def posterior_samples(self, batch, n=16, refine_steps=20, points_per_stream=256, generator=None):
        """Samples of n draws of z and beta from the (refined) posterior, decoded at every knot; generator: a
        torch.Generator on the model's device, for the latent draws and the refinement's point draws."""
        from .refine import refine
        post = self.encode(batch)
        if refine_steps:
            post = refine(self, post, batch, refine_steps, points_per_stream=points_per_stream, generator=generator)
        W, T = batch.n_windows, batch.T
        dev = post.mu_z.device
        w, k = all_knots(batch.lengths.to(dev))
        npw = batch.tokens.n_patches.to(dev)
        keys = ("rot6d", "transl", "contact_logit", "pos_logvar", "joints", "beta")
        acc = {key: [] for key in keys}
        for _ in range(int(n)):
            z = sample(post.mu_z, post.logvar_z, generator=generator) * post.mask[..., None]
            beta = sample(post.mu_b, post.logvar_b, generator=generator)
            out = self.decode(z, beta, (w, k), npw, batch.lengths.to(dev))
            body = self.dense(out, beta, batch)
            with _no_autocast(dev):
                rest = self.skeleton.rest(beta.float(), body.gender)
                kb = KnotBatch(body.rot6d.float(), body.transl.float(), rest, self.rate, body.lengths)
                joints = kb.pose(w, k.float() / self.rate)[0]
            acc["rot6d"].append(body.rot6d)
            acc["transl"].append(body.transl)
            acc["contact_logit"].append(body.contact_logit)
            acc["pos_logvar"].append(scatter_dense(out.pos_logvar, w, k, W, T))
            acc["joints"].append(scatter_dense(joints, w, k, W, T))
            acc["beta"].append(beta)
        return Samples(*(torch.stack(acc[key]) for key in keys), batch.lengths)


def _jepa_post_load(model, incompatible):
    """Load hook of a model with a JEPA module (MFM.attach_jepa): when every jepa.* entry is missing from the
    loaded state dict, the target copies the loaded tokenizer and encoder and the missing entries are dropped
    from the report; otherwise nothing changes."""
    jm = getattr(model, "jepa", None)
    if jm is None:
        return
    miss = [k for k in incompatible.missing_keys if k.startswith("jepa.")]
    if not miss or set(miss) != {"jepa." + k for k in jm.state_dict()}:
        return
    jm.sync(model.tokenizer, model.encoder)
    for k in miss:
        incompatible.missing_keys.remove(k)


def n_parameters_of(cfg=PILOT):
    """Parameter count of MFM(cfg) (built without the skeleton bank, which holds no parameter)."""
    return n_parameters(MFM(replace(cfg, skeleton=False)))


# ---------------------------------------------------------------- state_dict audit
SCREEN_DIMS = (6890, 52)
SCREEN_NAMES = ("shapedirs", "posedirs", "v_template", "j_regressor", "j_template", "j_dirs", "kintree")
SMPLH_KEYS = ("v_template", "shapedirs", "J_regressor", "posedirs", "weights")


def smplh_arrays(allow_reference=False):
    """{(gender, name): float64 array} of every SMPL+H array as SMPLHBody stores it (shapedirs cut to 16
    betas) and the derived J_template (22, 3) and J_dirs (22, 3, 16), for the three genders, read from the
    MoveAtlas model files (read-only); plus ('bank', 'J_template') (3, 22, 3) and ('bank', 'J_dirs') (3, 22,
    3, 16), the three genders stacked as SkeletonBank holds them. Behind the reference flag (mdfm.body.backend):
    the open model's audit reads no SMPL+H file."""
    from ..body.backend import require_reference
    require_reference("mfm.smplh_arrays", allow_reference)
    from ..body.smplh import _joint_basis               # behind the flag only: no SMPL+H module at the top
    out = {}
    for g in GENDERS:
        with np.load(paths.smplh_model(g)) as m:
            for k in SMPLH_KEYS:
                a = np.asarray(m[k], dtype=np.float64)
                out[(g, k)] = a[:, :, :N_BETAS] if k == "shapedirs" else a
            out[(g, "J_template")], out[(g, "J_dirs")] = _joint_basis(m)
    for k in ("J_template", "J_dirs"):
        out[("bank", k)] = np.stack([out[(g, k)] for g in GENDERS])
    return out


def _sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _tensors(obj, prefix=""):
    """(name, tensor) of every tensor inside nested dicts, lists and tuples (a state dict, an optimizer
    state_dict, a checkpoint dict), names joined with '.'."""
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{prefix}.{i}" if prefix else str(i))


def audit_state_dict(sd, arrays=None, small=100_000):
    """Problems (list of str, empty when clean) of a state dict (module docstring), or of any nested dict,
    list or tuple of tensors (an optimizer state_dict, a checkpoint), every tensor at any depth screened;
    arrays: smplh_arrays() (read when None, behind the reference flag); {} screens shapes and names only (the
    open body's audit, mdfm/train/provenance.py)."""
    arrays = smplh_arrays() if arrays is None else arrays
    hashes = {}
    for key, a in arrays.items():
        for dt in (np.float32, np.float64):
            hashes[_sha(a.astype(dt))] = key
    problems = []
    for name, v in _tensors(sd):
        shape = tuple(v.shape)
        if any(s in SCREEN_DIMS for s in shape):
            problems.append(f"{name}: shape {shape} has a dimension of 6890 or 52")
        if any(s in name.lower() for s in SCREEN_NAMES):
            problems.append(f"{name}: an SMPL+H array name")
        if not v.is_floating_point() or not v.numel():
            continue
        x = v.detach().cpu()
        for dt in (torch.float32, torch.float64):
            hit = hashes.get(_sha(x.to(dt).numpy()))
            if hit is not None:
                problems.append(f"{name}: bytes equal SMPL+H {hit}")
        if x.numel() <= small:
            xd = x.double().numpy()
            for key, a in arrays.items():
                if a.size == xd.size and np.allclose(xd.reshape(a.shape), a, rtol=1e-5, atol=1e-7):
                    problems.append(f"{name}: values equal SMPL+H {key}")
    return problems
