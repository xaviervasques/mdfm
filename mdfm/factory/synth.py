"""The GPU synthetic renderer of the factory (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.4 and 3,
task B4.10): AMASS bodies of the body store rendered into windows.Window objects with the streams of a
layout (factory/layouts.py), without gradients, in macro-batches.

Per window (RenderSpec): the layout, drawn with layouts.sample_layout, and its nuisance, drawn here from
a per-window seed so that a window renders the same in any batch. Clock: every stream of a window starts
at t_first uniform within one sample period of the layout's first stream, t_k = t_first + k / fps, and
stops before the window's duration (L - 1) / 100 s, the span of its L valid knots (no query leaves the
Hermite span). One KnotBatch holds the macro-batch; KnotBatch.kinematics runs once per point, in chunks
of CHUNK points, and every kernel reads that one evaluation:
- IMU: kernels.imu.imu_predict at the 14 sites of imu_render.site_offsets (the reference placement of
  imu_render.site_table, batched) plus a placement jitter (PLACEMENT_SD_M per axis [cfg], norm clipped at
  the nuisance bound OFFSET_MAX_M), mounts drawn from the prior of the layout's mount family
  (prior_mounts: the draw of MountPriors.sample_mounts, reports/mount_priors/priors.json, batched over
  the windows), constant biases drawn from the device family's spreads; then, on the host, white noise
  per sample at the device family's levels (for xsens13 the per-site levels of
  reports/g2_renderer/device_noise.json; the generic levels scaled by a log-uniform factor) and
  saturation at the device ranges. Magnetometer channels are zero and masked; channels outside the
  layout's mask (gyro of an acc-only layout) too. No residual spectrum: OD9 (decided 2026-09-28) admits
  synthetic data by utility, not by the discriminator of scripts/g2_renderer_realism.py.
- pose2d: the canonical 15 joints at the camera frames through factory/camera_render.py: sample_camera
  aimed at the window-mean pelvis, then keep_in_view (below), the tracker process and the canonical 2D
  normalization.
- pose3d: camera_render.render_pose3d at 100 Hz, per-source joint offsets zero (the synthetic-truth row
  of kernels.nuisance.SourceOffsets), Student-t joint noise POSE3D_NOISE_M, a uniform lab heading.
- force totals (targets): kernels.force.newton_bw (centre-of-mass acceleration of the de Leva table,
  contact_prob split) times a gain m g with m uniform in MASS_KG [cfg], in the record unit (N, or
  'unknown' for the weargait insoles); on the channels of the layout mask (the per-foot totals). Samples
  in flight (kernels.force.flight: both contact probabilities below 0.5, no foot near the y = 0 floor),
  where newton_bw leaves the split undefined and asks the caller to mask, are invalid and read 0 (truth:
  flight). A plate layout carries one plate of PLATE_M [cfg] centred under one foot (drawn) at its stance
  sample nearest the window centre, axes along the window frame: that foot reads its load while its
  contact point (the mean of its ankle and foot joints on the floor) lies on the plate, 0 elsewhere; the
  other foot has no plate and is absent from the channel mask, as in a stroke trial where one foot meets
  no plate.
Every stream then passes through data.window_store.band_copy (40 Hz zero-phase low-pass and decimation
to about 100 Hz on native timestamps for IMU and force, pose as recorded), as real streams do. IMU and
force streams of one modality, rate, length and channel count are band-copied together (their groups
side by side in one Stream): band_copy filters each group on its own valid runs, so the result is that
of one call per stream (tests/test_synth.py checks it bit for bit).

Camera choice (B4.8 left open that a static camera loses a walking subject in a long window):
keep_in_view moves the sampled camera back along its optical axis, keeping its direction, focal length
and principal point, until every canonical joint of the window projects inside the central 1 -
IN_VIEW_MARGIN of the image at a depth of at least MIN_DEPTH_M; the added distance is kept in the
truth (dolly_m). A long walk is thus filmed from farther away, as a lab camera framing a walkway is.

Truth (Stream.truth, Window.body): knots, contacts, lengths, betas, gender, take; per stream the drawn
nuisance (mounts sensor to segment, placement, biases, noise levels, camera, heading, gain, plate) and the
joint velocities v and accelerations a (T, 22, 3) of the body at the stream's kept sample times (the
samples band_copy keeps). render_pair renders the same bodies a second time under resampled nuisance
and device noise, with the same concrete layout.

Amplitude randomization (AmplitudeAug, off unless SynthSampler.amp_aug is set; F2 lever 2, task F2L2,
docs/proposals/F2_LEVER2_RULE.md; config section render_aug, which train/loop.py hands to the Trainer's renderer).
In each IMU stream of a window rendered without spec.clean, at every site of AmplitudeAug.sites the stream holds,
the deviation of the clean angular rate and of the clean specific force (biases included, before white noise and
saturation) from their means over the stream's samples is multiplied by its own gain: x' = m + g (x - m) per
channel, m the window mean, so that the H2 features G_pc (half the central range of the angular rate) and A_dyn
(the RMS norm of the specific force minus its mean) scale by g exactly, while the biases, the mean gravity
direction and the mean turning rate are kept. Gains are log-uniform within exp(+-gyro_log_max) and
exp(+-acc_log_max), drawn per window, stream, site and feature from numpy.random.default_rng((window seed,
AMP_AUG_SALT, stream index)), a generator of its own, so that every other draw of the window (mounts, placement,
biases, noise, camera) stays that of the render without it; the second render of a pair draws its own gains. The
gains of the stream's sites (1 where none is drawn) go to Stream.truth['amp_gain'] (S, 2), acc then gyro, the order
of kernels.nuisance.IMUNuisance.gain. The model's own gain nuisance stays bounded by GAIN_LOG_MAX (5 percent).

Precision: the renderer runs in the dtype given (float32 by default, float64 for reference checks).
Against the reference renderer imu_render.render, float64 meets the spec's 1e-4 g and 1e-2 deg/s and
float32 on the GPU does not (the Hermite second derivative sums knot values weighted by about 6 rate^2);
the measured errors of each precision are in reports/layouts/summary.md and tests/test_synth.py.
"""
import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..body.kinematics import KnotBatch
from ..body.rotations import axis_angle_to_matrix
from ..data import window_store as WS
from ..data import windows as W
from ..kernels import pose as P
from ..kernels.force import com_acc, contact_prob, flight, newton_bw
from ..kernels.imu import G, imu_predict
from ..kernels.nuisance import OFFSET_MAX_M, MountPriors
from . import camera_render as CR
from . import layouts as L
from .imu_render import SITES, site_offsets

CHUNK = 131072              # kinematics points per call (reports/bench/bench_kinematics.md)
RATE = 100.0                # body clock, Hz
SOURCE = "synthetic"        # Window.source and stream meta source of every rendered window
PLACEMENT_SD_M = 0.01       # [cfg] placement jitter per axis, norm clipped at OFFSET_MAX_M
MASS_KG = (50.0, 100.0)     # [cfg] force gain m g, m uniform in this range
PLATE_M = (0.4, 0.6)        # [cfg] plate width (window-frame x) and length (z), m
IN_VIEW_MARGIN = 0.1        # [cfg] share of the half image kept free around the body
MIN_DEPTH_M = 0.5           # [cfg] nearest joint depth after keep_in_view
POSE3D_NOISE_M = 0.003      # camera_render.render_pose3d default
POSE3D_NU = 4
PAIR_SHARE = 0.25           # [cfg] share of windows rendered twice (spec 3)
FEET = ((7, 10), (8, 11))   # (ankle, foot) joints of the left and right foot
NO_ROTATIONS = torch.empty(0, dtype=torch.float64)   # render_pose3d reads R_glob only with offsets


@dataclass
class RenderSpec:
    """One window's render: a concrete Layout; mount (S, 3, 3) sensor to segment for the layout's IMU
    sites, or None to draw it from the layout's mount prior; clean: no placement jitter, bias, white noise
    or saturation on the IMU, no pose noise, the clean tracker (camera_render.TrackerProcess.clean)."""
    layout: L.Layout
    mount: object = None
    clean: bool = False


AMP_AUG_SALT = 0xF212       # second entropy word of the amplitude-gain generator (module docstring)


@dataclass(frozen=True)
class AmplitudeAug:
    """Amplitude randomization of the IMU renders (module docstring): sites, names of imu_render.SITES; gyro_log_max
    and acc_log_max, the half-widths of the log-uniform gains of the angular rate and of the specific force."""
    sites: tuple
    gyro_log_max: float
    acc_log_max: float

    def __post_init__(self):
        bad = sorted(set(self.sites) - set(SITES))
        if not self.sites or bad or len(set(self.sites)) != len(self.sites):
            raise ValueError(f"amplitude randomization sites {list(self.sites)}: distinct names of {SITES}")
        for v in (self.gyro_log_max, self.acc_log_max):
            if not (math.isfinite(float(v)) and float(v) >= 0.0):
                raise ValueError(f"amplitude randomization half-width {v}: finite and non-negative")

    @classmethod
    def of(cls, section):
        """The AmplitudeAug of a config section (train.config.RenderAugSection), None while it is disabled."""
        if section is None or not bool(section.enabled):
            return None
        return cls(tuple(str(s) for s in section.sites), float(section.gyro_log_max), float(section.acc_log_max))

    def gains(self, site_ids, seed, stream):
        """(S, 2) float64 gains (acc, gyro) of the sites site_ids of one stream: drawn for every site of the stream
        from default_rng((seed, AMP_AUG_SALT, stream)), uniform u in [-1, 1) per site and feature, gain exp(u x
        half-width) at the sites of self.sites and 1 elsewhere."""
        rng = np.random.default_rng((int(seed), AMP_AUG_SALT, int(stream)))
        u = rng.uniform(-1.0, 1.0, (len(site_ids), 2))
        on = np.array([SITES[i] in self.sites for i in site_ids], bool)
        g = np.exp(u * np.array([self.acc_log_max, self.gyro_log_max]))
        return np.where(on[:, None], g, 1.0)

    @staticmethod
    def apply(x, gains):
        """x (T, S, 6) acc then gyro, float64: each channel's deviation from its mean over T scaled by the gain of
        its site and feature (gains (S, 2), acc then gyro); a new array."""
        g = np.repeat(gains, 3, axis=1)[None]                      # (1, S, 6)
        m = x.mean(axis=0, keepdims=True)
        return np.where(g != 1.0, m + g * (x - m), x)


def prior_mounts(priors, family, site, z):
    """Sensor to segment rotations (M, 3, 3) float64 of MountPriors.sample_mounts for M (family id, site id)
    rows given their standard normal draws z (M, 3): R0 Exp(theta), theta = sigma_tilt (z - (z . u) u) +
    sigma_yaw (z . u) u with u the site's up_sensor."""
    f = torch.as_tensor(family, dtype=torch.int64)
    s = torch.as_tensor(site, dtype=torch.int64)
    z = torch.as_tensor(z, dtype=torch.float64)
    pr = priors.to(dtype=torch.float64, device="cpu")
    u = pr.up_sensor[f, s]
    st, sy = pr.sigma_tilt[f, s][:, None], pr.sigma_yaw[f, s][:, None]
    zu = (z * u).sum(-1, keepdim=True)
    theta = st * (z - zu * u) + sy * zu * u
    return pr.R0[f, s] @ axis_angle_to_matrix(theta)


def keep_in_view(cam, pos15, margin=IN_VIEW_MARGIN, min_depth=MIN_DEPTH_M):
    """(camera, dolly m): cam (a single undistorted camera.Camera) moved back along its optical axis by
    the smallest distance that puts every joint of pos15 (T, 15, 3) m inside the central 1 - margin of
    the image (both axes, about the principal point) at a depth of at least min_depth. Moving the centre
    by d along the axis adds d to every depth and leaves x and y of the camera frame unchanged: t' = t +
    d e_z."""
    R, t0 = np.asarray(cam.R, dtype=np.float64), np.asarray(cam.t, dtype=np.float64)
    X = np.asarray(pos15, dtype=np.float64).reshape(-1, 3) @ R.T + t0
    K = np.asarray(cam.K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hx = (1 - margin) * min(cx, cam.width - cx)
    hy = (1 - margin) * min(cy, cam.height - cy)
    need = np.maximum(np.maximum(fx * np.abs(X[:, 0]) / hx, fy * np.abs(X[:, 1]) / hy), min_depth)
    d = max(0.0, float((need - X[:, 2]).max()))
    t = cam.t.clone()
    t[2] += d
    return cam._replace(t=t), d


def band_copies(streams):
    """[(Stream, BandInfo)]: window_store.band_copy of each stream; IMU and force streams of one modality,
    rate, length and channel count are copied in one call with their groups side by side (band_copy
    filters each group on its own valid runs, so each part equals the copy of its stream alone)."""
    out = [None] * len(streams)
    groups = {}
    for k, s in enumerate(streams):
        if s.modality in ("imu", "force"):
            key = (s.modality, float(s.fps), s.values.shape[0], s.values.shape[2])
            groups.setdefault(key, []).append(k)
        else:
            out[k] = WS.band_copy(s, return_info=True)
    for ks in groups.values():
        if len(ks) == 1:
            out[ks[0]] = WS.band_copy(streams[ks[0]], return_info=True)
            continue
        first = streams[ks[0]]
        stack = W.Stream(modality=first.modality, role=first.role, fps=first.fps, t=first.t,
                         values=np.concatenate([streams[k].values for k in ks], 1),
                         chan_mask=np.concatenate([streams[k].chan_mask for k in ks], 0),
                         valid=np.concatenate([streams[k].valid for k in ks], 1), meta=dict(first.meta))
        b, info = WS.band_copy(stack, return_info=True)
        g0 = 0
        for k in ks:
            s = streams[k]
            g1 = g0 + s.values.shape[1]
            out[k] = (W.Stream(modality=s.modality, role=s.role, fps=b.fps, t=s.t[info.idx].copy(),
                               values=b.values[:, g0:g1].copy(), chan_mask=s.chan_mask.copy(),
                               valid=b.valid[:, g0:g1].copy(), meta=dict(s.meta), truth=s.truth), info)
            g0 = g1
    return out


def concat_bodies(a, b):
    """Two BodyWindows (data.body_store) stacked along the window axis."""
    return type(a)(*[np.concatenate([x, y]) if isinstance(x, np.ndarray) else torch.cat([x, y]) for x, y in zip(a, b)])


def select_bodies(a, idx):
    """The BodyWindows rows idx."""
    idx = np.asarray(idx, dtype=np.int64)
    return type(a)(*[x[idx] if isinstance(x, np.ndarray) else x[torch.from_numpy(idx)] for x in a])


def _rng(gen):
    return gen if isinstance(gen, np.random.Generator) else np.random.default_rng(gen)


class SynthSampler:
    """The synthetic renderer (module docstring). store: data.body_store.BodyStore (for draw() and take
    ids); skel: SkeletonBank (built in dtype on device when None); layouts: {name: Layout} (default
    layouts.load_layouts()); priors: MountPriors (default reports/mount_priors/priors.json); devices:
    {name: DeviceFamily} (default layouts.device_families()); device: torch device (cuda when available);
    dtype: float32 or float64; chunk: kinematics points per call; mixture: {family: weight} of draw()
    (default layouts.DEFAULT_MIXTURE). timing: wall seconds of render() summed per phase since the last
    reset: plan (clocks and nuisance draws), kinematics (the GPU passes with their host copies), streams
    (host-side noise, camera and tracker, pose normalization, force, Window assembly), band_copy. amp_aug: an
    AmplitudeAug applied to every IMU stream rendered without spec.clean (module docstring), None (default) for
    none; train/loop.Trainer sets it from the config's render_aug section."""

    PHASES = ("plan", "kinematics", "streams", "band_copy")

    def __init__(self, store=None, skel=None, layouts=None, priors=None, device=None, dtype=torch.float32,
                 chunk=CHUNK, devices=None, mixture=None, amp_aug=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.store = store
        self.take_ids = None if store is None else store.takes["take_id"].to_numpy()
        if skel is None:                    # the v1 SMPL+H bank by default: imported here only (plan phase E step 2)
            from ..body.smplh import SkeletonBank
            skel = SkeletonBank(dtype)
        if skel.J_template.dtype != dtype or skel.J_template.device != self.device:
            skel = copy.deepcopy(skel).to(device=self.device, dtype=dtype)    # the caller's bank stays as it is
        self.skel = skel
        self.layouts = L.load_layouts() if layouts is None else layouts
        self.priors = (MountPriors.load() if priors is None else priors).to(dtype=torch.float64, device="cpu")
        self.devices = L.device_families() if devices is None else devices
        self.mixture = L.DEFAULT_MIXTURE if mixture is None else mixture
        self.chunk = int(chunk)
        self.amp_aug = amp_aug
        self.reset_timing()

    def reset_timing(self):
        self.timing = dict.fromkeys(self.PHASES, 0.0)

    def _tick(self, phase, t0):
        t1 = time.perf_counter()
        self.timing[phase] += t1 - t0
        return t1

    # ------------------------------------------------------------ draws
    def draw(self, gen, n, mixture=None, win_s=10.0):
        """(BodyWindows, [RenderSpec]) of n windows: bodies from store.sample (subject-balanced), layouts
        from layouts.sample_layout."""
        if self.store is None:
            raise ValueError("draw() needs a body store")
        gen = _rng(gen)
        bodies = self.store.sample(gen, n, win_s)
        specs = [RenderSpec(L.sample_layout(gen, mixture or self.mixture, self.layouts)) for _ in range(n)]
        return bodies, specs

    def sample(self, gen, n, mixture=None, pair_share=PAIR_SHARE, band=True):
        """(windows, pairs): n windows drawn and rendered, of which round(pair_share n), chosen uniformly, are
        rendered a second time under resampled nuisance; the second renders follow the n first ones and
        pairs lists their (first, second) positions."""
        gen = _rng(gen)
        bodies, specs = self.draw(gen, n, mixture)
        m = int(round(pair_share * n))
        idx = np.sort(gen.choice(n, size=m, replace=False)) if m else np.zeros(0, np.int64)
        if m:
            bodies = concat_bodies(bodies, select_bodies(bodies, idx))
            specs = specs + [RenderSpec(specs[i].layout, None, specs[i].clean) for i in idx]
        out = self.render(bodies, specs, gen, band=band)
        return out, [(int(i), n + k) for k, i in enumerate(idx)]

    def render_pair(self, bodies, specs, gen, band=True):
        """[(Window, Window)]: each body rendered twice with its concrete layout, the second time under a
        resampled nuisance (mounts, placement, biases, noise, camera, heading, gain, plate)."""
        specs = [s if isinstance(s, RenderSpec) else RenderSpec(s) for s in specs]
        again = [RenderSpec(s.layout, None, s.clean) for s in specs]
        out = self.render(concat_bodies(bodies, bodies), specs + again, gen, band=band)
        n = len(specs)
        return list(zip(out[:n], out[n:]))

    # ------------------------------------------------------------ render
    def render(self, bodies, specs, gen, band=True):
        """[windows.Window] of bodies (data.body_store.BodyWindows, window frame) under specs (RenderSpec
        or concrete Layout per window); gen: numpy Generator or int seed (one seed per window is drawn from
        it first). band=False keeps the streams at the layout rate before band_copy (reference checks)."""
        gen = _rng(gen)
        specs = [s if isinstance(s, RenderSpec) else RenderSpec(s) for s in specs]
        n = len(specs)
        if n != len(bodies.lengths):
            raise ValueError(f"{n} specs for {len(bodies.lengths)} bodies")
        for s in specs:
            if not s.layout.concrete:
                raise ValueError(f"{s.layout.name}: draw a concrete layout first (layouts.concrete)")
        t0 = time.perf_counter()
        seeds = gen.integers(0, 2 ** 62, size=n)
        lengths = bodies.lengths.numpy().astype(np.int64)
        dur = (lengths - 1) / RATE
        plans = [self._plan(i, specs[i], dur[i], int(seeds[i]), band) for i in range(n)]
        self._mounts(plans)
        t0 = self._tick("plan", t0)
        dev, dt = self.device, self.dtype
        with torch.no_grad():
            betas = bodies.betas.to(device=dev, dtype=dt)
            gender = bodies.gender.to(device=dev, dtype=torch.int64)
            rest = self.skel.rest(betas, gender)
            kb = KnotBatch(bodies.rot6d.to(device=dev, dtype=dt), bodies.transl.to(device=dev, dtype=dt), rest,
                           rate=RATE, lengths=bodies.lengths.to(dev))
            self._imu_pass(kb, rest, plans)
            self._pose_pass(kb, plans)
            self._force_pass(kb, plans, gender)
            del kb, rest
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = self._tick("kinematics", t0)
        raw = [(pl, st, self._assemble(pl, st)) for pl in plans for st in pl["streams"]]
        t0 = self._tick("streams", t0)
        if band:
            copies = band_copies([r[2] for r in raw])
            for (pl, st, _), (b, info) in zip(raw, copies):
                if not np.array_equal(info.idx, st["keep"]):
                    raise RuntimeError(f"band_copy kept {len(info.idx)} samples, the render planned {len(st['keep'])}")
                pl.setdefault("out", []).append(b)
        else:
            for pl, st, s in raw:
                pl.setdefault("out", []).append(s)
        t0 = self._tick("band_copy", t0)
        out = []
        for i, pl in enumerate(plans):
            take = int(bodies.take[i])
            take_id = str(self.take_ids[take]) if self.take_ids is not None else f"take{take}"
            body = {"rot6d": bodies.rot6d[i].numpy(), "transl": bodies.transl[i].numpy(),
                    "contact": bodies.contact[i].numpy(), "valid": bodies.valid[i].numpy(),
                    "lengths": int(lengths[i]), "betas": bodies.betas[i].numpy(), "gender": int(bodies.gender[i]),
                    "take": take, "take_id": take_id, "start": int(bodies.start[i]), "rate": RATE,
                    "seed": int(seeds[i])}
            out.append(W.Window(source=SOURCE, person_id=str(bodies.person[i]), key=f"{take_id}@{int(bodies.start[i])}",
                                synthetic=True, t0=0.0, duration=float(dur[i]), streams=pl["out"],
                                cov={"sex": int(bodies.gender[i]), "age_band": W.AGE_UNKNOWN,
                                     "task": int(bodies.task[i])}, body=body))
        self._tick("streams", t0)
        return out

    # ------------------------------------------------------------ per-window draws
    def _plan(self, i, spec, dur, seed, band):
        """Sample times and nuisance draws of window i, in a fixed order from its seed."""
        rng = np.random.default_rng(seed)
        lay = spec.layout
        t_first = float(rng.uniform(0.0, 1.0 / lay.streams[0].fps[0]))
        pl = {"i": i, "rng": rng, "seed": seed, "spec": spec, "t_first": t_first, "dur": dur, "streams": []}
        for s in lay.streams:
            fps = float(s.fps[0])
            t = t_first + np.arange(int(math.floor((dur - t_first) * fps)) + 1, dtype=np.float64) / fps
            t = t[t < dur]
            if len(t) < 2:
                raise ValueError(f"window {i}: {len(t)} samples of {s.modality} at {fps} Hz in {dur} s")
            k = WS.band_plan(s.modality, fps).k if band else 1
            st = {"layout": s, "fps": fps, "t": t, "keep": np.arange(0, len(t), k), "truth": {}}
            if s.modality == "imu":
                st.update(self._imu_draws(lay, s, spec, rng))
            elif s.modality == "force":
                mass = float(rng.uniform(*MASS_KG))
                st["gain"] = mass * G
                st["truth"].update(mass_kg=mass, gain=mass * G)
                if s.plate:
                    st["plate_foot"] = int(rng.integers(2))
            pl["streams"].append(st)
        return pl

    def _imu_draws(self, lay, s, spec, rng):
        S = len(s.sites)
        dev = self.devices[lay.device]
        ids = np.array(s.site_ids)
        z = None if spec.mount is not None else rng.standard_normal((S, 3))
        if spec.clean:
            jitter, scale = np.zeros((S, 3)), 1.0
            ba, bg = np.zeros((S, 3)), np.zeros((S, 3))
        else:
            jitter = rng.normal(0.0, PLACEMENT_SD_M, (S, 3))
            norm = np.linalg.norm(jitter, axis=1, keepdims=True)
            jitter = np.where(norm > OFFSET_MAX_M, jitter * OFFSET_MAX_M / np.maximum(norm, 1e-12), jitter)
            lo, hi = dev.noise_scale
            scale = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if hi > lo else float(lo)
            ba = rng.standard_normal((S, 3)) * dev.bias_acc_sd[ids]
            bg = rng.standard_normal((S, 3)) * dev.bias_gyro_sd[ids]
        cm = s.chan_mask
        ba, bg = ba * cm[:, 0:3], bg * cm[:, 3:6]
        on = 0.0 if spec.clean else 1.0
        truth = {"offset_res": jitter.astype(np.float32), "bias_acc": ba.astype(np.float32),
                 "bias_gyro": bg.astype(np.float32),
                 "noise_acc": (dev.noise_acc[ids] * scale * on).astype(np.float32),
                 "noise_gyro": (dev.noise_gyro[ids] * scale * on).astype(np.float32),
                 "noise_scale": scale, "mount_family": lay.mount, "device": lay.device}
        mount = None if spec.mount is None else \
            torch.as_tensor(spec.mount, dtype=torch.float64).reshape(S, 3, 3).numpy()
        return {"z": z, "mount": mount, "jitter": jitter, "bias_acc": ba, "bias_gyro": bg, "truth": truth}

    def _mounts(self, plans):
        """The prior mount draws of every IMU stream of the batch, in one prior_mounts call."""
        rows, fam, site, z = [], [], [], []
        for pl in plans:
            for st in pl["streams"]:
                if st.get("z") is not None:
                    f = int(self.priors.family_index(pl["spec"].layout.mount))
                    rows.append((st, len(site)))
                    fam += [f] * len(st["layout"].sites)
                    site += st["layout"].site_ids
                    z.append(st["z"])
        if rows:
            M = prior_mounts(self.priors, fam, site, np.concatenate(z)).numpy()
            for st, a in rows:
                st["mount"] = M[a:a + len(st["layout"].sites)]
        for pl in plans:
            for st in pl["streams"]:
                if st["layout"].modality == "imu":
                    st["truth"]["mount"] = st["mount"].astype(np.float32)

    # ------------------------------------------------------------ kinematics passes
    def _points(self, plans, kinds):
        """Flat queries of the streams of the given modalities: w, t (device), the kept flags and the
        (plan, stream, start, stop) of each stream."""
        ws, ts, keeps, where, n = [], [], [], [], 0
        for pl in plans:
            for st in pl["streams"]:
                if st["layout"].modality not in kinds:
                    continue
                m = len(st["t"])
                ws.append(np.full(m, pl["i"], dtype=np.int64))
                ts.append(st["t"])
                kp = np.zeros(m, bool)
                kp[st["keep"]] = True
                keeps.append(kp)
                where.append((pl, st, n, n + m))
                n += m
        if not n:
            return None
        w = torch.from_numpy(np.concatenate(ws)).to(self.device)
        t = torch.from_numpy(np.concatenate(ts)).to(device=self.device, dtype=self.dtype)
        return w, t, np.concatenate(keeps), where

    def _chunks(self, kb, w, t, keep, where, fn):
        """fn(kin, sl) over chunks of the points; the truth v, a (float32) at the kept points of each
        stream go into its truth."""
        outs, vs, as_ = [], [], []
        for a in range(0, len(w), self.chunk):
            sl = slice(a, a + self.chunk)
            kin = kb.kinematics(w[sl], t[sl])
            outs.append(fn(kin, sl))
            k = torch.from_numpy(keep[sl]).to(self.device)
            vs.append(kin.v[k].float().cpu().numpy())
            as_.append(kin.a[k].float().cpu().numpy())
            del kin
        v, a = np.concatenate(vs), np.concatenate(as_)
        ck = np.concatenate([[0], np.cumsum(keep)])
        for pl, st, lo, hi in where:
            st["truth"]["v"], st["truth"]["a"] = v[ck[lo]:ck[hi]], a[ck[lo]:ck[hi]]
        return outs

    def _imu_pass(self, kb, rest, plans):
        q = self._points(plans, ("imu",))
        if q is None:
            return
        w, t, keep, where = q
        dev, dt = self.device, self.dtype
        n_w, n_s = kb.W, len(SITES)
        joints, off0 = site_offsets(rest)                               # (14,), (W, 14, 3)
        off0_h = off0.double().cpu().numpy()
        jit = np.zeros((n_w, n_s, 3))
        mnt = np.tile(np.eye(3), (n_w, n_s, 1, 1))
        ba, bg = np.zeros((n_w, n_s, 3)), np.zeros((n_w, n_s, 3))
        grav = np.zeros(n_w, bool)
        smax = max(len(st["layout"].sites) for _, st, _, _ in where)
        sel = np.zeros((n_w, smax), np.int64)
        for pl, st, _, _ in where:
            i, ids = pl["i"], st["layout"].site_ids
            jit[i, ids], mnt[i, ids] = st["jitter"], st["mount"]
            ba[i, ids], bg[i, ids] = st["bias_acc"], st["bias_gyro"]
            grav[i] = st["layout"].acc_content == "linear_no_gravity"
            sel[i, :len(ids)] = ids
            st["truth"]["offset"] = (off0_h[i, ids] + jit[i, ids]).astype(np.float32)
        on = lambda x: torch.from_numpy(x).to(device=dev, dtype=dt)  # noqa: E731
        off = off0 + on(jit) if jit.any() else off0
        mount, ba, bg = on(mnt), on(ba), on(bg)
        grav = torch.from_numpy(grav).to(dev)
        sel = torch.from_numpy(sel).to(dev)
        jt = joints.to(dev)

        def fn(kin, sl):
            ww = w[sl]
            acc, gyro = imu_predict(kin, jt, off[ww], mount[ww], ba[ww], bg[ww], gravity_removed=grav[ww][:, None])
            idx = sel[ww][..., None].expand(-1, -1, 3)
            return torch.cat([acc.gather(1, idx), gyro.gather(1, idx)], -1).cpu().numpy()

        sig = np.concatenate(self._chunks(kb, w, t, keep, where, fn))
        for pl, st, lo, hi in where:
            st["clean"] = sig[lo:hi, :len(st["layout"].sites)]

    def _pose_pass(self, kb, plans):
        q = self._points(plans, ("pose2d", "pose3d"))
        if q is None:
            return
        w, t, keep, where = q
        p = torch.cat(self._chunks(kb, w, t, keep, where, lambda kin, sl: kin.p.double().cpu()))
        for pl, st, lo, hi in where:
            st["p"] = p[lo:hi]

    def _force_pass(self, kb, plans, gender):
        q = self._points(plans, ("force",))
        if q is None:
            return
        w, t, keep, where = q
        feet = torch.tensor([j for f in FEET for j in f], device=self.device)

        def fn(kin, sl):
            cp = contact_prob(kin.p, kin.v)
            bw = newton_bw(com_acc(kin.a, gender[w[sl]]), cp)
            return bw.double().cpu().numpy(), cp.double().cpu().numpy(), kin.p[:, feet].double().cpu().numpy()

        outs = self._chunks(kb, w, t, keep, where, fn)
        bw, cp, fp = (np.concatenate([o[k] for o in outs]) for k in range(3))
        for pl, st, lo, hi in where:
            st["bw"], st["cp"], st["feet"] = bw[lo:hi], cp[lo:hi], fp[lo:hi].reshape(-1, 2, 2, 3)

    # ------------------------------------------------------------ streams
    def _assemble(self, pl, st):
        s = st["layout"]
        meta = {"source": SOURCE, "layout": pl["spec"].layout.name}
        make = {"imu": self._imu_stream, "pose2d": self._pose2d_stream, "pose3d": self._pose3d_stream,
                "force": self._force_stream}[s.modality]
        stream = make(pl, st, meta)
        stream.truth["t_first_s"] = pl["t_first"]
        return stream

    def _imu_stream(self, pl, st, meta):
        s, lay, rng = st["layout"], pl["spec"].layout, pl["rng"]
        dev = self.devices[lay.device]
        x = st["clean"].astype(np.float64)                         # (T, S, 6) acc g, gyro deg/s
        T, S = x.shape[:2]
        cm = s.chan_mask
        tr = st["truth"]
        if self.amp_aug is not None and not pl["spec"].clean:
            k = next(j for j, other in enumerate(pl["streams"]) if other is st)
            gains = self.amp_aug.gains(s.site_ids, pl["seed"], k)
            x = AmplitudeAug.apply(x, gains)
            tr["amp_gain"] = gains.astype(np.float32)
        if not pl["spec"].clean:
            x[..., 0:3] += rng.standard_normal((T, S, 3)) * tr["noise_acc"][None].astype(np.float64)
            x[..., 3:6] += rng.standard_normal((T, S, 3)) * tr["noise_gyro"][None].astype(np.float64)
            x[..., 0:3] = np.clip(x[..., 0:3], -dev.acc_range_g, dev.acc_range_g)
            x[..., 3:6] = np.clip(x[..., 3:6], -dev.gyro_range_dps, dev.gyro_range_dps)
        values = np.zeros((T, S, 9), np.float32)
        values[..., 0:6] = np.where(cm[None, :, 0:6], x, 0.0)
        meta.update(site_ids=s.site_ids, acc_content=s.acc_content, device=lay.device)
        return W.Stream(modality="imu", role=s.role, fps=st["fps"], t=st["t"].astype(np.float32), values=values,
                        chan_mask=cm.copy(), valid=np.ones((T, S), bool), meta=meta, truth=tr)

    def _pose2d_stream(self, pl, st, meta):
        s = st["layout"]
        tg = torch.Generator().manual_seed(pl["seed"])
        pos15 = st["p"][:, P.SEL]
        cam = CR.sample_camera(tg, pos15[:, P.ROOT].mean(0), family=s.camera or "generic", fps=st["fps"])
        cam, dolly = keep_in_view(cam, pos15)
        tracker = CR.TrackerProcess.clean() if pl["spec"].clean else CR.TrackerProcess()
        rec, tt = CR.render_pose2d(pos15, cam, tg, tracker)
        values, cm, valid, m = W.arrays("pose2d", rec)
        meta.update({k: v for k, v in m.items() if k in W.META_KEYS})
        meta["camera"] = s.camera or "generic"
        tr = st["truth"]
        tr.update(camera={"K": cam.K.numpy(), "R": cam.R.numpy(), "t": cam.t.numpy(), "width": cam.width,
                          "height": cam.height, "fps": float(cam.fps)}, dolly_m=dolly, uv=tt["uv"],
                  jitter_s=tt["jitter_s"], lost=tt["lost"], in_image=tt["in_image"], confidence=tt["confidence"])
        return W.Stream(modality="pose2d", role=s.role, fps=float(cam.fps), t=st["t"].astype(np.float32),
                        values=values, chan_mask=cm, valid=valid, meta=meta, truth=tr)

    def _pose3d_stream(self, pl, st, meta):
        s = st["layout"]
        tg = torch.Generator().manual_seed(pl["seed"])
        clean = pl["spec"].clean
        rec, tt = CR.render_pose3d(st["p"], NO_ROTATIONS, tg, offsets=None, fps=st["fps"],
                                   noise_m=0.0 if clean else POSE3D_NOISE_M, nu=POSE3D_NU)
        values, cm, valid, m = W.arrays("pose3d", rec)
        meta.update({k: v for k, v in m.items() if k in W.META_KEYS})
        tr = st["truth"]
        tr.update(heading=tt["heading"], joints=tt["joints"], offsets=np.zeros((15, 3), np.float32),
                  noise_m=0.0 if clean else POSE3D_NOISE_M)
        return W.Stream(modality="pose3d", role=s.role, fps=st["fps"], t=st["t"].astype(np.float32), values=values,
                        chan_mask=cm, valid=valid, meta=meta, truth=tr)

    def _force_stream(self, pl, st, meta):
        s = st["layout"]
        bw = st["bw"]                                              # (T, 2) body weights
        T = bw.shape[0]
        load = bw * st["gain"]
        tr = st["truth"]
        cm = s.chan_mask
        if s.plate:
            f = st["plate_foot"]
            row = next(r for r in cm if r.any())
            cm = np.zeros_like(cm)
            cm[f] = row
            c = st["feet"][:, f].mean(1)[:, [0, 2]]                # contact point of the plate foot, (T, 2) x, z
            stance = np.flatnonzero(st["cp"][:, f] > 0.5)
            mid = np.argmin(np.abs(st["t"] - pl["dur"] / 2)) if not len(stance) else \
                stance[np.argmin(np.abs(st["t"][stance] - pl["dur"] / 2))]
            centre = c[mid]
            on = (np.abs(c[:, 0] - centre[0]) <= PLATE_M[0] / 2) & (np.abs(c[:, 1] - centre[1]) <= PLATE_M[1] / 2)
            occ = np.zeros((T, 2))
            occ[:, f] = on
            load = load * occ
            tr.update(plate={"foot": f, "centre_xz_m": centre.astype(np.float32), "size_xz_m": np.array(PLATE_M)},
                      on_plate=on[st["keep"]])
        fl = flight(torch.from_numpy(st["cp"])).numpy()          # (T,) the split of newton_bw is undefined
        valid = np.repeat(~fl[:, None], 2, 1)
        values = np.zeros((T, 2, 9), np.float32)
        values[..., 8] = load
        values = np.where(cm[None] & valid[..., None], values, 0.0).astype(np.float32)
        meta["unit"] = s.unit or "N"
        tr.update(bw=bw[st["keep"]].astype(np.float32), contact_prob=st["cp"][st["keep"]].astype(np.float32),
                  flight=fl[st["keep"]])
        return W.Stream(modality="force", role=s.role, fps=st["fps"], t=st["t"].astype(np.float32), values=values,
                        chan_mask=cm.copy(), valid=valid, meta=meta, truth=tr)
