"""Anny fitted to motion-capture landmarks: the body truth of the open body store (plan docs/MODEL_OPEN_PORT_PLAN.md,
step M3; scripts/build_open_body_store.py).

Landmarks. Per frame the 15 canonical joints of MoveAtlas (their maps from markers, model outputs or BVH joints, in
metres, y up, the root kept: mdfm/data/open_mocap_raw.py) and one toe landmark per foot. Each landmark l rides one
segment of the program's skeleton: pred_l = p_j(l) + G_s(l) (c_l + o_l), with p_j the position of a program joint,
G_s the global rotation of the segment's joint, c_l a rest offset (zero, except the toe: the rest vector from the
ankle to the program's foot joint, Anny's third-toe MTP head) and o_l a per-person offset in the segment's rest frame
(LANDMARKS). The offsets absorb what a landmark is on each source (a skin marker over the greater trochanter, an
acromion marker, the centroid of head markers) against the joint it stands for; their prior keeps them within a few
centimetres (OFFSET_SD_M).

Unknowns. Per person: the phenotype latent u (6,) of mdfm/body/openbody.py and the offsets o (17, 3). Per frame: the
22 local rotations (axis-angle, row 0 the root orientation) and the translation (the pelvis is transl plus the rest
pelvis, as everywhere in the program). The rest skeleton is the P3 basis (openbody.RestBasis, differentiable).

Objective (fit_frames), each term a sum over frames:
- data: pseudo-Huber of the landmark residual, scale DATA_SD_M, knee at HUBER_M, on valid landmarks only;
- prior: the phenotype prior (mdfm.model.prior.PhenotypePrior, the unknown-sex row) and the offsets' Gaussian prior;
- pose: the anatomical joint-limit table of the synthetic patient (reports/openbody/joint_limits.json, squared
  excess in radians, LIMIT_W), the joints no landmark observes held near their rest (the toes 10 and 11 and the
  wrists 20 and 21, UNOBSERVED_W), the spine's three joints and the neck sharing the trunk's bend (SPINE_W), the
  collars near their rest (COLLAR_W) and a small pull of every rotation to zero (POSE_W);
- time (with smooth=True): the second difference of the joint positions (ACC_SD_M per frame at 100 Hz) and of the
  rotations (ROT_ACC_SD).
Solved by L-BFGS (strong Wolfe line search). Stage 1 (fit_person): u, o and the poses of up to SHAPE_FRAMES frames
spread over the person's takes. Stage 2 (fit_take): u and o fixed, the poses of every frame of a take on the 100 Hz
clock, first on a 10 Hz subsample (from init_pose, the time terms scaled to its step), then refined at 100 Hz, in
windows of WINDOW_S seconds joined at the midpoint of their overlap.

Every constant marked [design] was set on 2026-11-12 for the first measure (reports/open_body_store/fit_measure.json)
and is recorded with each store build.
"""
import json
import math
import os

import numpy as np
import torch

from . import openbody as BO
from .joints import N_BODY, PARENTS, fk
from .rotations import axis_angle_to_matrix

F32 = torch.float32
RATE = 100.0
CUTOFF_HZ = 12.0            # as the v1 body store (data/prep.lowpass, 4th-order zero-phase Butterworth)
GAP_S = 0.1                 # [design] a landmark gap up to this long is bridged linearly; longer stays missing
# landmark -> (program joint, segment joint): the 15 canonical joints in MoveAtlas order, then the two toes; the hips
# ride the thigh (a greater trochanter marker, a joint centre) or, for a pelvis-marker source, the pelvis (segments)
CANON15 = ("pelvis", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
           "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
LANDMARKS = (("pelvis", 0, 0), ("neck", 12, 9), ("head", 15, 15), ("left_shoulder", 16, 13),
             ("right_shoulder", 17, 14), ("left_elbow", 18, 16), ("right_elbow", 19, 17), ("left_wrist", 20, 18),
             ("right_wrist", 21, 19), ("left_hip", 1, 1), ("right_hip", 2, 2), ("left_knee", 4, 1),
             ("right_knee", 5, 2), ("left_ankle", 7, 4), ("right_ankle", 8, 5), ("left_toe", 7, 7),
             ("right_toe", 8, 8))
N_LM = len(LANDMARKS)
HIPS = (9, 10)                                  # the hip landmarks


def segments(hip_on_pelvis=False):
    """(17,) the segment joint of each landmark: LANDMARKS', the hips on the pelvis (joint 0) when the source's hip
    landmark is a pelvis marker (an anterior superior iliac spine or a front waist marker: it moves with the pelvis,
    not with the thigh; on the thigh, its offset turns with the hip's flexion, 8 to 10 cm of error seated)."""
    out = [s for _, _, s in LANDMARKS]
    if hip_on_pelvis:
        for i in HIPS:
            out[i] = 0
    return out
TOE_REST = {15: (7, 10), 16: (8, 11)}          # landmark index -> (ankle, foot joint): c = rest[foot] - rest[ankle]
OFFSET_SD_M = 0.04          # [design] per-person landmark offset prior
HEAD_OFFSET_SD_M = 0.10     # [design] the head landmark (head markers' centroid) against Anny's skull-base joint
DATA_SD_M = 0.01            # [design]
HUBER_M = 0.03              # [design]
LIMIT_W = 100.0             # [design] per rad^2 of excess
UNOBSERVED_W = 10.0         # [design] per rad^2
SPINE_W = 10.0              # [design] per rad^2 of difference between consecutive trunk joints
COLLAR_W = 1.0              # [design] per rad^2
POSE_W = 0.01               # [design] per rad^2
ACC_SD_M = 0.002            # [design] joint position second difference per 0.01 s step (20 m/s^2)
ROT_ACC_SD = 0.02           # [design] rad per step^2
SHAPE_FRAMES = 400          # [design]
WINDOW_S = 30.0             # [design]
OVERLAP_S = 2.0             # [design]
COARSE_STEP = 10            # 100 Hz -> 10 Hz for the first pass
GUARD_M = 0.01             # [design]
COARSE_GUARD_M = 0.03      # [design]
COARSE_RETRIES = 2         # [design]
GUARD_PAD = 10             # [design] knots
COARSE_SMOOTH = True        # [design] the time terms in the 10 Hz pass too (scaled to its step): no isolated flip
UNOBSERVED = (10, 11, 20, 21)
TRUNK = (3, 6, 9, 12)
COLLARS = (13, 14)
LIMITS_JSON = os.path.join(BO.TREE, "reports", "openbody", "joint_limits.json")


def design_constants():
    """{name: value} of every constant of the objective (recorded with each build)."""
    keys = ("RATE", "CUTOFF_HZ", "GAP_S", "OFFSET_SD_M", "HEAD_OFFSET_SD_M", "DATA_SD_M", "HUBER_M", "LIMIT_W",
            "UNOBSERVED_W", "SPINE_W", "COLLAR_W", "POSE_W", "ACC_SD_M", "ROT_ACC_SD", "SHAPE_FRAMES", "WINDOW_S",
            "OVERLAP_S", "COARSE_STEP", "COARSE_SMOOTH", "GUARD_M", "GUARD_PAD",
            "COARSE_GUARD_M", "COARSE_RETRIES", "SEGMENT_SD_M")
    g = globals()
    return {k: g[k] for k in keys}


def joint_limits_rad():
    """(lo, hi) (22, 3) float32 radians of the anatomical table (reports/openbody/joint_limits.json 'body')."""
    with open(LIMITS_JSON, encoding="utf-8") as f:
        d = json.load(f)["body"]
    lo = torch.tensor(np.deg2rad(np.asarray(d["lo_deg"], np.float64)), dtype=F32)
    hi = torch.tensor(np.deg2rad(np.asarray(d["hi_deg"], np.float64)), dtype=F32)
    return lo, hi


# ---------------------------------------------------------------- landmarks on the 100 Hz clock
def _bridge(x, max_gap):
    """x (T,) with NaN: gaps up to max_gap samples filled linearly (interior only)."""
    ok = np.isfinite(x)
    if ok.all() or not ok.any():
        return x
    idx = np.flatnonzero(ok)
    out = x.copy()
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
    return out


def to_body_clock(x, fps, rate=RATE, cutoff=CUTOFF_HZ):
    """x (T, L, 3) m at fps with NaN for missing -> (x (N, L, 3) on the rate clock, valid (N, L) bool): short gaps
    bridged (GAP_S), a 4th-order zero-phase Butterworth low-pass at cutoff on each landmark's valid span (missing
    samples filled linearly for the filter only, then masked again), linear resampling at rate."""
    from scipy.signal import butter, filtfilt
    T, L, _ = x.shape
    t = np.arange(T) / float(fps)
    n = int(math.floor((T - 1) / float(fps) * rate)) + 1
    tn = np.arange(n) / rate
    out = np.full((n, L, 3), np.nan)
    valid = np.zeros((n, L), dtype=bool)
    b, a = butter(4, cutoff / (fps / 2)) if cutoff < fps / 2 else (None, None)
    for j in range(L):
        y = np.stack([_bridge(x[:, j, c], int(round(GAP_S * fps))) for c in range(3)], -1)
        ok = np.isfinite(y).all(-1)
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        filled = np.stack([np.interp(np.arange(T), idx, y[ok, c]) for c in range(3)], -1)
        if b is not None and T > 3 * 5 * 3:
            filled = filtfilt(b, a, filled, axis=0)
        out[:, j] = np.stack([np.interp(tn, t, filled[:, c]) for c in range(3)], -1)
        # a resampled sample is valid when both native neighbours were valid
        k = np.clip(np.floor(tn * fps).astype(int), 0, T - 1)
        k1 = np.clip(k + 1, 0, T - 1)
        valid[:, j] = ok[k] & ok[k1]
    out[~valid] = np.nan
    return out, valid


def clean(landmarks):
    """Masks (NaN) the samples MoveAtlas's ingest masks: beyond 1 km, or farther than 8 torso lengths from the
    pelvis (ingest_open_mocap.Build.add); returns a copy and the counts."""
    x = np.array(landmarks, dtype=np.float64, copy=True)
    absurd = np.isfinite(x).all(-1) & (np.abs(x) > 1e3).any(-1)
    x[absurd] = np.nan
    torso = np.linalg.norm(x[:, 1] - x[:, 0], axis=-1)
    torso = torso[np.isfinite(torso) & (torso > 1e-6)]
    far = 0
    if torso.size:
        d = np.linalg.norm(x - x[:, :1], axis=-1)
        bad = np.isfinite(d) & (d > 8 * np.median(torso))
        far = int(bad.sum())
        x[bad] = np.nan
    return x, {"absurd": int(absurd.sum()), "far": far}


def rotvec(R):
    """(..., 3, 3) -> (..., 3) rotation vectors, angle in [0, pi], through the quaternion (the largest of its four
    components first, Shepperd's rule): exact near a half turn, where the arccos of the trace and the antisymmetric
    part lose the axis (a body facing -z turns by pi about y; mdfm.body.rotations.matrix_to_axis_angle, kept for
    tests and export, clamps its cosine there and returns a wrong angle). Computed in float64."""
    R = R.double()
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cand = torch.stack([tr, m00, m11, m22], -1)
    k = cand.argmax(-1)
    q = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)
    t0 = torch.sqrt((1.0 + tr).clamp_min(1e-30)) * 2
    q0 = torch.stack([0.25 * t0, (R[..., 2, 1] - R[..., 1, 2]) / t0, (R[..., 0, 2] - R[..., 2, 0]) / t0,
                      (R[..., 1, 0] - R[..., 0, 1]) / t0], -1)
    t1 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-30)) * 2
    q1 = torch.stack([(R[..., 2, 1] - R[..., 1, 2]) / t1, 0.25 * t1, (R[..., 0, 1] + R[..., 1, 0]) / t1,
                      (R[..., 0, 2] + R[..., 2, 0]) / t1], -1)
    t2 = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-30)) * 2
    q2 = torch.stack([(R[..., 0, 2] - R[..., 2, 0]) / t2, (R[..., 0, 1] + R[..., 1, 0]) / t2, 0.25 * t2,
                      (R[..., 1, 2] + R[..., 2, 1]) / t2], -1)
    t3 = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-30)) * 2
    q3 = torch.stack([(R[..., 1, 0] - R[..., 0, 1]) / t3, (R[..., 0, 2] + R[..., 2, 0]) / t3,
                      (R[..., 1, 2] + R[..., 2, 1]) / t3, 0.25 * t3], -1)
    q = torch.where((k == 0)[..., None], q0, torch.where((k == 1)[..., None], q1,
                                                        torch.where((k == 2)[..., None], q2, q3)))
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)          # w >= 0: angle in [0, pi]
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(n, q[..., :1])
    return torch.where(n > 1e-12, v / n.clamp_min(1e-30) * ang, 2 * v)


# ---------------------------------------------------------------- the model
class Fit(torch.nn.Module):
    """Parameters of one person (u, o) and of F frames (aa (F, 22, 3), transl (F, 3)). The root orientation of a
    frame is Rb0 exp(aa[:, 0]), Rb0 (F, 3, 3) a fixed base (the identity unless set): a body that turns about itself
    keeps a small root vector, never one near 2 pi, where the rotation vector is singular."""

    def __init__(self, F, basis, prior, u0=None, o0=None, hip_on_pelvis=False):
        super().__init__()
        self.basis, self.prior = basis, prior
        self.u = torch.nn.Parameter(torch.zeros(BO.N_PHENO, dtype=F32) if u0 is None else u0.clone().float())
        self.o = torch.nn.Parameter(torch.zeros(N_LM, 3, dtype=F32) if o0 is None else o0.clone().float())
        self.aa = torch.nn.Parameter(torch.zeros(F, N_BODY, 3, dtype=F32))
        self.tr = torch.nn.Parameter(torch.zeros(F, 3, dtype=F32))
        self.Rb0 = torch.eye(3, dtype=F32).repeat(F, 1, 1)
        self.lo, self.hi = joint_limits_rad()
        sd = torch.full((N_LM, 1), OFFSET_SD_M, dtype=F32)
        sd[2] = HEAD_OFFSET_SD_M
        self.osd = sd
        self.jl = torch.tensor([j for _, j, _ in LANDMARKS])
        self.sl = torch.tensor(segments(hip_on_pelvis))

    def rest(self):
        return self.basis.rest(BO.to_phenotype(self.u)[None])[0]

    def offsets_rest(self, rest):
        c = torch.zeros(N_LM, 3, dtype=F32)
        for li, (a, f) in TOE_REST.items():
            c[li] = rest[f] - rest[a]
        return c + self.o

    def local(self, aa=None):
        """(F, 22, 3, 3) local rotations, the root on its base (class docstring)."""
        aa = self.aa if aa is None else aa
        R = axis_angle_to_matrix(aa)
        if aa is self.aa:
            R = torch.cat([(self.Rb0 @ R[:, 0])[:, None], R[:, 1:]], 1)
        return R

    def full_aa(self):
        """(F, 22, 3) the rotation vectors of the local rotations (the root's with its base, angle at most pi)."""
        with torch.no_grad():
            return rotvec(self.local().double()).float()

    def set_root_base(self):
        """Fold the root vector into the base: Rb0 <- Rb0 exp(aa[:, 0]), aa[:, 0] <- 0."""
        with torch.no_grad():
            self.Rb0 = self.local()[:, 0].detach().clone()
            self.aa[:, 0] = 0.0

    def predict(self, aa=None, tr=None, rest=None):
        """(landmarks (F, 17, 3), joints (F, 22, 3), R global (F, 22, 3, 3)); aa given: plain rotation vectors."""
        tr = self.tr if tr is None else tr
        rest = self.rest() if rest is None else rest
        R = self.local(aa)
        P, G = fk(R, tr, rest)
        off = self.offsets_rest(rest)
        lm = P[:, self.jl] + (G[:, self.sl] @ off[None, :, :, None])[..., 0]
        return lm, P, G

    def loss(self, target, w, smooth=False, shape=True, rest=None, step=1):
        lm, P, G = self.predict(rest=rest)
        r = torch.where(w[..., None], lm - target, torch.zeros_like(lm))
        d2 = (r ** 2).sum(-1)
        data = (HUBER_M ** 2 * (torch.sqrt(1.0 + d2 / HUBER_M ** 2) - 1.0)).sum() / DATA_SD_M ** 2
        aa = self.aa
        body = aa[:, 1:]
        lo, hi = self.lo[1:], self.hi[1:]
        lim = (torch.relu(body - hi) ** 2 + torch.relu(lo - body) ** 2).sum() * LIMIT_W
        pose = (aa[:, list(UNOBSERVED)] ** 2).sum() * UNOBSERVED_W + (aa[:, list(COLLARS)] ** 2).sum() * COLLAR_W
        tr_ = aa[:, list(TRUNK)]
        pose = pose + ((tr_[:, 1:] - tr_[:, :-1]) ** 2).sum() * SPINE_W + (body ** 2).sum() * POSE_W
        total = data + lim + pose
        if smooth and aa.shape[0] > 2:
            acc = P[2:] - 2 * P[1:-1] + P[:-2]
            racc = aa[2:, 1:] - 2 * aa[1:-1, 1:] + aa[:-2, 1:]
            R0 = G[:, 0]                             # the root's rotation matrix: free of the vector's wrap
            racc0 = R0[2:] - 2 * R0[1:-1] + R0[:-2]
            k = float(step) ** 2                    # a second difference over `step` knots scales with step^2
            total = total + (acc ** 2).sum() / (k * ACC_SD_M) ** 2 +                 ((racc ** 2).sum() + 0.5 * (racc0 ** 2).sum()) / (k * ROT_ACC_SD) ** 2
        if shape:
            unk = torch.tensor([2])
            total = total - self.prior.log_prob(self.u[None], unk).sum() + 0.5 * ((self.o / self.osd) ** 2).sum()
        return total, data


def init_root(target, w, rest):
    """(aa (F, 22, 3), transl (F, 3)) float32: the root orientation of each frame by Kabsch on the torso landmarks
    (pelvis, neck, shoulders, hips) against the rest joints, the other joints at rest, transl puts the rest pelvis on
    the pelvis landmark (or the torso centroid when the pelvis is missing)."""
    ids = (0, 1, 3, 4, 9, 10)
    joints = [LANDMARKS[i][1] for i in ids]
    Q = rest[joints].double()
    F = target.shape[0]
    aa = torch.zeros(F, N_BODY, 3, dtype=torch.float64)
    tr = torch.zeros(F, 3, dtype=torch.float64)
    for f in range(F):
        m = w[f, list(ids)]
        if int(m.sum()) < 3:
            continue
        X = target[f, list(ids)].double()[m]
        Y = Q[m]
        xc, yc = X.mean(0), Y.mean(0)
        H = (Y - yc).T @ (X - xc)
        U, _, Vt = torch.linalg.svd(H)
        D = torch.eye(3, dtype=torch.float64)
        D[2, 2] = torch.sign(torch.det(Vt.T @ U.T))
        R = Vt.T @ D @ U.T
        aa[f, 0] = rotvec(R)
        tr[f] = xc - R @ yc + R @ Q[0] - Q[0]             # the pelvis is rest[0] + tr, rotated about itself
    # frames without a torso take their neighbour's
    good = (aa[:, 0].abs().sum(-1) > 0) | (tr.abs().sum(-1) > 0)
    if good.any() and not good.all():
        gi = torch.nonzero(good)[:, 0]
        for f in torch.nonzero(~good)[:, 0]:
            k = gi[torch.argmin((gi - f).abs())]
            aa[f, 0], tr[f] = aa[k, 0], tr[k]
    return aa.float(), tr.float()


# limb chains of the initial pose: (joint, its child joint in the rest skeleton, landmark at the joint, landmark
# toward which the bone points); processed in this order, parents first
ALIGN = ((1, 4, 9, 11), (2, 5, 10, 12), (4, 7, 11, 13), (5, 8, 12, 14), (7, 10, 13, 15), (8, 11, 14, 16),
         (9, 12, 0, 1), (16, 18, 3, 5), (17, 19, 4, 6), (18, 20, 5, 7), (19, 21, 6, 8), (12, 15, 1, 2))


def _min_rot(a, b):
    """(F, 3, 3) the minimal rotations taking the directions a (F, 3) onto b (F, 3) (identity where either is
    degenerate or they are opposite)."""
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v = torch.linalg.cross(a, b, dim=-1)
    c = (a * b).sum(-1)
    K = torch.zeros(a.shape[0], 3, 3, dtype=a.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K = K - K.transpose(1, 2)
    R = torch.eye(3, dtype=a.dtype) + K + (K @ K) / (1.0 + c).clamp_min(1e-6)[:, None, None]
    bad = (c < -0.999) | ~torch.isfinite(R).all(-1).all(-1)
    R[bad] = torch.eye(3, dtype=a.dtype)
    return R


def init_pose(target, w, rest):
    """(aa (F, 22, 3), transl (F, 3)) float32: init_root's root, then each joint of ALIGN turned so that its rest
    bone points along the observed direction between its two landmarks (the minimal rotation, the parent's global
    rotation taken as fitted so far; a joint whose landmarks are missing keeps its rest). The spine's turn is put on
    spine3 alone; the fit spreads it."""
    aa, tr = init_root(target, w, rest)
    R = axis_angle_to_matrix(aa.double())
    X = target.double()
    rest = rest.double()
    for j, c, la, lb in ALIGN:
        _, G = fk(R, tr.double(), rest)
        Gp = G[:, PARENTS[j]]
        b = (Gp @ (rest[c] - rest[j])[:, None])[..., 0]
        d = X[:, lb] - X[:, la]
        ok = w[:, la] & w[:, lb]
        Rw = _min_rot(b, d)                       # world turn of the bone
        Rl = Gp.transpose(1, 2) @ Rw @ Gp          # as a local rotation after the parent
        Rl[~ok] = torch.eye(3, dtype=Rl.dtype)
        R[:, j] = Rl
    return rotvec(R).float(), tr


def _lbfgs(params, closure, iters):
    """One L-BFGS run. A line search that diverges (a step past float32's range raises inside torch's update, as CMU
    take 090 of part 1 did on 2026-11-13) or leaves a non-finite parameter puts every parameter back where the run
    started: the pass is skipped, the take keeps its start and its guards, and its residual is judged as any other.
    A run that ends finite is unchanged."""
    start = [p.detach().clone() for p in params]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)
    try:
        opt.step(closure)
        ok = all(bool(torch.isfinite(p).all()) for p in params)
    except RuntimeError as e:
        if "overflow" not in str(e):
            raise
        ok = False
    if not ok:
        with torch.no_grad():
            for p, s in zip(params, start):
                p.copy_(s)


# segment lengths read for the phenotype's start: (landmark a, landmark b, joint a, joint b), the limbs only (thighs,
# shanks, upper arms, forearms): the trunk's and the girdles' landmarks (the pelvis markers' centroid, the shoulder
# midpoint, the acromion markers) stand farther from Anny's joints than a limb's two ends do, and read as a body
# shape they drive the phenotype to its bounds (a probe of 2026-11-12: gender 0.9 and age 0.96 for every person)
SEGMENTS = ((9, 11, 1, 4), (10, 12, 2, 5), (11, 13, 4, 7), (12, 14, 5, 8), (3, 5, 16, 18), (4, 6, 17, 19),
            (5, 7, 18, 20), (6, 8, 19, 21))
SEGMENT_SD_M = 0.01         # [design]


def init_phenotype(target, w, basis, prior, iters=60):
    """(6,) the phenotype latent whose rest segment lengths (SEGMENTS: thighs, shanks, upper arms, forearms) match the median landmark distances of the frames, under the phenotype prior: a
    start for stage 1 that does not leave the body's size to the landmark offsets."""
    obs, ok = [], []
    for la, lb, _, _ in SEGMENTS:
        sel = w[:, la] & w[:, lb]
        d = (target[sel, la] - target[sel, lb]).norm(dim=-1)
        obs.append(d.median() if d.numel() else torch.tensor(0.0))
        ok.append(bool(d.numel()))
    obs, ok = torch.stack(obs).float(), torch.tensor(ok)
    u = torch.nn.Parameter(prior.mean[2].float().clone())

    def closure():
        u.grad = None
        r = basis.rest(BO.to_phenotype(u)[None])[0]
        L = torch.stack([(r[a] - r[b]).norm() for _, _, a, b in SEGMENTS])
        loss = (((L - obs) / SEGMENT_SD_M) ** 2 * ok).sum() * 0.5 - prior.log_prob(u[None], torch.tensor([2])).sum()
        loss.backward()
        return loss
    _lbfgs([u], closure, iters)
    return u.detach()


def fit_person(frames, basis, prior, iters=(100, 300), hip_on_pelvis=False):
    """Stage 1: frames = (target (F, 17, 3) m, valid (F, 17) bool) sampled over a person's takes -> (u, o, the
    fitted Fit). The phenotype starts from init_phenotype; first the poses alone, then u, o and the poses
    together."""
    target, w = (torch.as_tensor(frames[0], dtype=F32), torch.as_tensor(frames[1], dtype=torch.bool))
    target = torch.nan_to_num(target)
    u0 = init_phenotype(target, w, basis, prior)
    m = Fit(target.shape[0], basis, prior, u0=u0, hip_on_pelvis=hip_on_pelvis)
    with torch.no_grad():
        aa, tr = init_pose(target, w, m.rest())
        m.aa.copy_(aa)
        m.tr.copy_(tr)
    m.set_root_base()

    def run(params, n, shape):
        def closure():
            for p in params:
                p.grad = None
            loss, _ = m.loss(target, w, smooth=False, shape=shape)
            loss.backward()
            return loss
        _lbfgs(params, closure, n)
    run([m.aa, m.tr], iters[0], False)
    run([m.aa, m.tr, m.u, m.o], iters[1], True)
    return m.u.detach().clone(), m.o.detach().clone(), m


def unwrap(aa):
    """(T, J, 3) axis-angle made continuous in time: at each frame, among the vectors of the same rotation v + 2 pi k
    v / |v| (k = -1, 0, 1), the one nearest the previous frame's (the smoothness term and any interpolation read the
    vectors, not the rotations)."""
    out = aa.clone()
    for f in range(1, out.shape[0]):
        v = out[f]
        u = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cand = torch.stack([v - 2 * math.pi * u, v, v + 2 * math.pi * u])        # (3, J, 3)
        k = (cand - out[f - 1][None]).norm(dim=-1).argmin(0)                     # (J,)
        out[f] = cand[k, torch.arange(v.shape[0])]
    return out


def interp_rotations(aa_c, k, a):
    """(T, J, 3) axis-angle at fine frames between coarse frames k and k + 1 with fraction a (T, 1): the geodesic
    R_k exp(a log(R_k^T R_k+1)), unwrapped."""
    R = axis_angle_to_matrix(aa_c.double())
    Rk, Rk1 = R[k], R[k + 1]
    rel = rotvec(Rk.transpose(-1, -2) @ Rk1)
    Ri = Rk @ axis_angle_to_matrix(rel * a.double()[..., None])
    return unwrap(rotvec(Ri)).to(aa_c.dtype)


def _fit_window(target, w, u, o, basis, prior, iters=(100, 60), hip_on_pelvis=False):
    """Stage 2 on one window (T frames at 100 Hz): coarse 10 Hz pass from the root initialization, then
    the 100 Hz refinement with the time terms. Returns aa (T, 22, 3), tr (T, 3)."""
    T = target.shape[0]
    m = Fit(T, basis, prior, u0=u, o0=o, hip_on_pelvis=hip_on_pelvis)
    m.u.requires_grad_(False)
    m.o.requires_grad_(False)
    with torch.no_grad():
        rest = m.rest()
    n = max(2, int(math.ceil((T - 1) / COARSE_STEP)) + 1)
    idx = torch.unique(torch.round(torch.linspace(0, T - 1, n)).long())     # near-uniform, both ends kept
    step = (T - 1) / max(1, len(idx) - 1)
    mc = Fit(len(idx), basis, prior, u0=u, o0=o, hip_on_pelvis=hip_on_pelvis)
    mc.u.requires_grad_(False)
    mc.o.requires_grad_(False)
    tc, wc = target[idx], w[idx]
    with torch.no_grad():
        aa0, tr0 = init_pose(tc, wc, rest)
        mc.aa.copy_(unwrap(aa0))
        mc.tr.copy_(tr0)
    mc.set_root_base()

    def closure_c():
        mc.aa.grad = mc.tr.grad = None
        loss, _ = mc.loss(tc, wc, smooth=COARSE_SMOOTH, shape=False, rest=rest, step=step)
        loss.backward()
        return loss
    _lbfgs([mc.aa, mc.tr], closure_c, iters[0])
    # guard: a coarse knot left farther than COARSE_GUARD_M from its landmarks (a local minimum, the root turned
    # the wrong way) restarts from its own init_pose, and the pass runs again, at most COARSE_RETRIES times
    for _ in range(COARSE_RETRIES):
        with torch.no_grad():
            lm, _, _ = mc.predict(rest=rest)
            d2 = torch.where(wc, ((lm - tc) ** 2).sum(-1), torch.zeros(()))
            bad = torch.sqrt(d2.sum(1) / wc.sum(1).clamp_min(1)) > COARSE_GUARD_M
            if not bool(bad.any()):
                break
            Rfull = mc.local()
            Rfull[bad] = axis_angle_to_matrix(aa0[bad])
            mc.Rb0 = Rfull[:, 0].clone()
            mc.aa[:, 0] = 0.0
            mc.aa[bad, 1:] = aa0[bad, 1:]
            mc.tr[bad] = tr0[bad]
        _lbfgs([mc.aa, mc.tr], closure_c, iters[0])
    with torch.no_grad():
        t = torch.arange(T, dtype=F32)
        tcf = idx.to(F32)
        k = torch.clamp(torch.searchsorted(tcf, t, right=True) - 1, 0, len(idx) - 2)
        a = ((t - tcf[k]) / (tcf[k + 1] - tcf[k]))[:, None]
        full = mc.full_aa()
        root = interp_rotations(full[:, :1], k, a)            # the root by the geodesic between coarse knots
        if len(idx) >= 4:
            # the other joints and the translation by a cubic spline through the coarse knots: no kink at the knots
            # for the time terms of the 100 Hz pass to iron out
            from scipy.interpolate import CubicSpline
            tn, fn = tcf.numpy().astype(np.float64), t.numpy().astype(np.float64)
            rest_j = torch.as_tensor(CubicSpline(tn, unwrap(full[:, 1:].double()).numpy(), axis=0)(fn))
            m.tr.copy_(torch.as_tensor(CubicSpline(tn, mc.tr.detach().double().numpy(), axis=0)(fn)))
        else:
            rest_j = interp_rotations(full[:, 1:], k, a)
            m.tr.copy_(mc.tr[k] * (1 - a) + mc.tr[k + 1] * a)
        m.aa.copy_(torch.cat([root, rest_j.to(F32)], 1))
    m.set_root_base()

    def closure():
        m.aa.grad = m.tr.grad = None
        loss, _ = m.loss(target, w, smooth=True, shape=False, rest=rest)
        loss.backward()
        return loss

    def frame_rms():
        with torch.no_grad():
            lm, _, _ = m.predict(rest=rest)
            d2 = torch.where(w, ((lm - target) ** 2).sum(-1), torch.zeros(()))
            return torch.sqrt(d2.sum(1) / w.sum(1).clamp_min(1))
    init_aa, init_tr, init_R0 = m.aa.detach().clone(), m.tr.detach().clone(), m.Rb0.clone()
    before = frame_rms()
    _lbfgs([m.aa, m.tr], closure, iters[1])
    # guard: a frame the 100 Hz pass left more than GUARD_M worse than its spline start (a line-search step into
    # another basin) goes back to the start, with GUARD_PAD knots around it, and the pass runs once more
    worse = frame_rms() > before + GUARD_M
    if bool(worse.any()):
        sel = torch.nn.functional.max_pool1d(worse.float()[None, None], 2 * GUARD_PAD + 1, 1, GUARD_PAD)[0, 0] > 0
        with torch.no_grad():
            m.aa[sel], m.tr[sel] = init_aa[sel], init_tr[sel]
            m.Rb0[sel] = init_R0[sel]
        _lbfgs([m.aa, m.tr], closure, iters[1])
    return m.full_aa(), m.tr.detach()


def fit_take(target, valid, u, o, basis, prior, rate=RATE, iters=(100, 60), hip_on_pelvis=False):
    """Stage 2 over a whole take on the rate clock: target (T, 17, 3) m (NaN missing), valid (T, 17) -> aa (T, 22, 3),
    tr (T, 3) float32, in windows of WINDOW_S with OVERLAP_S joined at the overlap's midpoint."""
    target = torch.nan_to_num(torch.as_tensor(target, dtype=F32))
    w = torch.as_tensor(valid, dtype=torch.bool)
    T = target.shape[0]
    W, O = int(WINDOW_S * rate), int(OVERLAP_S * rate)
    if T <= W:
        return _fit_window(target, w, u, o, basis, prior, iters=iters, hip_on_pelvis=hip_on_pelvis)
    aa = torch.zeros(T, N_BODY, 3, dtype=F32)
    tr = torch.zeros(T, 3, dtype=F32)
    start, keep_from = 0, 0
    while True:
        end = min(T, start + W)
        a, t = _fit_window(target[start:end], w[start:end], u, o, basis, prior, iters=iters,
                           hip_on_pelvis=hip_on_pelvis)
        cut = end if end == T else end - O // 2
        aa[keep_from:cut] = a[keep_from - start:cut - start]
        tr[keep_from:cut] = t[keep_from - start:cut - start]
        if end == T:
            break
        keep_from = cut
        start = end - O
    return aa, tr


def residuals(target, valid, aa, tr, u, o, basis, prior, hip_on_pelvis=False):
    """Per-landmark residual norms (T, 17) m (NaN where the landmark is missing) and the joints (T, 22, 3)."""
    m = Fit(1, basis, prior, u0=u, o0=o, hip_on_pelvis=hip_on_pelvis)
    with torch.no_grad():
        rest = m.rest()
        lm, P, _ = m.predict(aa=aa, tr=tr, rest=rest)
        d = (lm - torch.nan_to_num(torch.as_tensor(target, dtype=F32))).norm(dim=-1)
        d[~torch.as_tensor(valid, dtype=torch.bool)] = float("nan")
    return d, P, rest


__all__ = ["CANON15", "LANDMARKS", "Fit", "fit_person", "fit_take", "residuals", "to_body_clock", "clean",
           "design_constants", "PARENTS"]
