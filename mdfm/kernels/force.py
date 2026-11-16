"""Force kernel (design section 3): vertical ground reaction per foot from whole-body momentum.

The Newtonian floor of the design: the total vertical ground reaction is m (g + a_com,y),
where a_com is the acceleration of the body's centre of mass, and it is split between the
feet by soft contact weights from foot height and speed. In body weights (BW) the floor is
newton_bw = (1 + a_com,y / g) x share, which keeps the mass out of the kernel. ForceHead adds a
zero-initialized learned residual to the split (it equals the floor at initialization); a
learned term must beat the floor (design gate G4). The 8 insole positions are not modelled in
the pilot: force_nll reads the per-foot totals only.

The centre of mass uses segment masses and centre-of-mass positions from de Leva (1996,
J Biomech 29:1223, adjustments to Zatsiorsky-Seluyanov), sex-specific, placed on the SMPL
joints: each segment runs from a proximal to a distal joint, and its centre of mass sits at
the tabled fraction of that length from the proximal end. The trunk is split into the de
Leva upper, middle and lower trunk on the SMPL spine chain; hands and feet use the wrist to
a fixed hand length and the ankle to the SMPL foot joint. com15 places the same table on the
15 canonical joints (whole trunk from neck to pelvis, feet at the ankles) for the model-free
Newton baseline of the G4 dry run. Sex codes: 0 female, 1 male, 2 unknown (the SkeletonBank
codes); unknown uses the mean of the female and male fractions.

force_nll compares predicted per-foot totals in BW with observed totals in native units
through a per-record gain (native units per BW) profiled in closed form. The residuals are in BW
(observation / gain - prediction), so the likelihood carries the log Jacobian log(gain / y_ref) of
each scored sample (y_ref the RMS of the group's scored observations, which holds no parameter):
without it a smaller prediction buys a larger gain and smaller residuals, and the loss would pay
the head to move load onto the foot that no plate sees. Observed totals are
first corrected at record level by prepare_totals (once per record, before windows are cut):
- the per-record, per-foot unloaded offset is removed where the foot unloads during the record
  (reports/stage1_sources: stroke plate totals carry unloaded levels up to 966 N);
- a plate foot that never unloads is masked, record-wide: on the non-lockbox stroke records these
  are the static trials where the whole body stands on one plate that MoveAtlas assigns to one foot
  (the records of stroke_plates.csv with a plate never below 20 N), which no per-foot model can
  attribute;
- plate sources see a foot only while it stands on its plates: samples farther than
  PLATE_MARGIN_S from a plate contact are masked (a stance on the floor between plates reads 0). A
  contact is a run of at least MIN_CONTACT_S above the unloaded band: most runs above the band on
  the stroke plate totals are one-sample spikes of the unloaded plate, and a spike must not open a
  scored window on an off-plate stance.
Insole totals keep a foot that never unloads (quiet standing) with its offset left at 0.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..body.kinematics import check_precision
from ..body.joints import BODY_JOINTS
from ..body.trajectory import second_order
from .camera import CANONICAL_15
from .imu import G

J = {n: i for i, n in enumerate(BODY_JOINTS)}
J15 = {n: i for i, n in enumerate(CANONICAL_15)}

# (segment, proximal joint, distal joint, mass fraction female, male, COM fraction female, male)
# de Leva 1996, table 4; the three trunk parts are mapped on the SMPL spine chain
# (neck to spine3, spine3 to spine1, spine1 to pelvis).
DE_LEVA = [
    ("head", "neck", "head", 0.0668, 0.0694, 0.4841, 0.5002),
    ("upper_trunk", "neck", "spine3", 0.1545, 0.1596, 0.5050, 0.5066),
    ("middle_trunk", "spine3", "spine1", 0.1465, 0.1633, 0.4512, 0.4502),
    ("lower_trunk", "spine1", "pelvis", 0.1247, 0.1117, 0.4920, 0.6115),
    ("left_upper_arm", "left_shoulder", "left_elbow", 0.0255, 0.0271, 0.5754, 0.5772),
    ("right_upper_arm", "right_shoulder", "right_elbow", 0.0255, 0.0271, 0.5754, 0.5772),
    ("left_forearm", "left_elbow", "left_wrist", 0.0138, 0.0162, 0.4559, 0.4574),
    ("right_forearm", "right_elbow", "right_wrist", 0.0138, 0.0162, 0.4559, 0.4574),
    ("left_thigh", "left_hip", "left_knee", 0.1478, 0.1416, 0.3612, 0.4095),
    ("right_thigh", "right_hip", "right_knee", 0.1478, 0.1416, 0.3612, 0.4095),
    ("left_shank", "left_knee", "left_ankle", 0.0481, 0.0433, 0.4416, 0.4459),
    ("right_shank", "right_knee", "right_ankle", 0.0481, 0.0433, 0.4416, 0.4459),
    ("left_foot", "left_ankle", "left_foot", 0.0129, 0.0137, 0.4014, 0.4415),
    ("right_foot", "right_ankle", "right_foot", 0.0129, 0.0137, 0.4014, 0.4415),
]
# hands: de Leva mass fractions, COM placed at the wrist (hands are at rest in version 1)
HANDS = [("left_hand", "left_wrist", 0.0056, 0.0061), ("right_hand", "right_wrist", 0.0056, 0.0061)]
# de Leva 1996, table 4, whole trunk (CERV to MIDH): the three trunk parts together, on the canonical
# neck to pelvis; mass fractions equal the sum of the three parts (0.4257 and 0.4346)
TRUNK15 = ("trunk", "neck", "pelvis", 0.4257, 0.4346, 0.4964, 0.5138)
# canonical 15 has no foot joint: the de Leva foot masses sit at the ankles
FEET15 = [("left_foot", "left_ankle", 0.0129, 0.0137), ("right_foot", "right_ankle", 0.0129, 0.0137)]

SEX_UNKNOWN = 2
FLOOR_SIGMA_BW = 0.01          # noise floor of force_nll, BW [cfg]
PRIOR_LOG_SD = 0.1             # sd of the log-normal gain prior [cfg]
BAND_N = 20.0                  # N: unloaded band of scripts/probe_stage1_sources.offset
BAND_REL = 0.03                # unit not N: the band as a fraction of p99 - p5 of the foot total [cfg]
UNLOADS_REL = 0.2              # a foot unloads when its unloaded level is at most this fraction of its p99 [cfg]
PLATE_MARGIN_S = 0.1           # s scored on each side of a plate contact [cfg]
MIN_CONTACT_S = 0.02           # s: shorter runs above the unloaded band are spikes, not plate contacts [cfg]


def _fractions(sex_code, m_f, m_m, c_f, c_m):
    if sex_code == 0:
        return m_f, c_f
    if sex_code == 1:
        return m_m, c_m
    return 0.5 * (m_f + m_m), 0.5 * (c_f + c_m)


def _table(segments, hands, index, n_joints):
    """COM weights (3, n_joints) float64: row s gives com = sum_j w[s, j] joint_j for sex code s, the
    mass fractions renormalized to sum to one over the modelled segments."""
    W = np.zeros((3, n_joints))
    for s in range(3):
        for _, a, b, mf, mm, cf, cm in segments:
            m, c = _fractions(s, mf, mm, cf, cm)
            W[s, index[a]] += m * (1.0 - c)
            W[s, index[b]] += m * c
        for _, a, mf, mm in hands:
            m, _ = _fractions(s, mf, mm, 0.0, 0.0)
            W[s, index[a]] += m
        W[s] /= W[s].sum()
    return W


def _segments15():
    """de Leva segments on the canonical 15 joints: head, whole trunk, arms, thighs, shanks; feet at the
    ankles (no foot joint), hands at the wrists."""
    keep = [s for s in DE_LEVA if s[0] == "head" or s[0].endswith(("upper_arm", "forearm", "thigh", "shank"))]
    return [TRUNK15] + keep


W22 = _table(DE_LEVA, HANDS, J, len(BODY_JOINTS))
W15 = _table(_segments15(), HANDS + FEET15, J15, len(CANONICAL_15))


def sex_codes(sex):
    """Sex codes from a label ('female' or 'F' 0, 'male' or 'M' 1, 'neutral', 'unknown' or None 2; other
    labels raise ValueError), an int, or an int64 tensor of codes (returned as is)."""
    if torch.is_tensor(sex):
        if sex.dtype != torch.int64:
            raise TypeError(f"sex codes must be int64, got {sex.dtype}")
        return sex
    if isinstance(sex, (int, np.integer)):
        if not 0 <= int(sex) <= 2:
            raise ValueError(f"sex code must be 0, 1 or 2, got {sex}")
        return int(sex)
    from ..body.sex import gender_code       # a label's code (mdfm/body/sex.py: no SMPL+H module, plan phase E)
    return gender_code(sex)


def _weights(table, sex, x):
    """COM weights for x (..., J, 3): (J,) for one sex, (..., J) for a tensor of codes over the batch dims."""
    code = sex_codes(sex)
    if isinstance(x, np.ndarray):
        return table[np.asarray(code)]
    W = torch.as_tensor(table, dtype=x.dtype, device=x.device)
    return W[code.to(x.device)] if torch.is_tensor(code) else W[code]


def com(joints, sex="female"):
    """joints (..., 22, 3) -> centre of mass (..., 3), with the de Leva fractions renormalized to sum to
    one over the modelled segments. sex: a label, a code, or int64 codes of shape joints.shape[:-2]."""
    w = _weights(W22, sex, joints)
    return (joints * w[..., None]).sum(-2)


def com_acc(acc, sex):
    """Centre-of-mass acceleration (N, 3) m/s^2 from the joint accelerations acc (N, 22, 3) (BodyKin.a)
    and sex (N,) int64 codes (or one label): com() is linear in joint positions, so it maps accelerations
    to the centre-of-mass acceleration."""
    check_precision(acc)
    return com(acc, sex)


def com15(joints15, sex):
    """Canonical 15 joints (..., 15, 3) m -> centre of mass (..., 3) m (torch or numpy), de Leva segments
    on the canonical skeleton (module docstring). sex is required (a label, a code, or int64 codes; 2 or
    'unknown' for the mean table): no default sex. The weights sum to one, so com15 of root-centred
    coordinates plus the root track equals com15 of world coordinates (the G4 dry run: canonical pose
    x scale plus the raw root track)."""
    w = _weights(W15, sex, joints15)
    return (joints15 * w[..., None]).sum(-2)


def com_kinematics(traj, t, sex="female"):
    """Centre-of-mass position, velocity and acceleration (N, 3) at times t (N,), from one pose
    evaluation under nested jvp (the trajectory's rest joints are computed once per subject)."""
    return second_order(lambda tt: com(traj.pose_at(tt)[0], sex), t)


def contact_weights(foot_height, foot_speed, h0=0.05, s_h=0.01, v0=0.3, s_v=0.1):
    """Soft contact probability per foot from height above the floor (m) and speed (m/s)."""
    return torch.sigmoid((h0 - foot_height) / s_h) * torch.sigmoid((v0 - foot_speed) / s_v)


def _floor_per_point(floor, like, trailing):
    """floor, a number or a tensor of shape () or (N,), shaped to broadcast against like (N, *trailing)."""
    if torch.is_tensor(floor) and floor.dim() == 1:
        return floor.to(like.device, like.dtype).reshape((-1,) + (1,) * trailing)
    return floor


def contact_prob(p, v, floor=0.0, up=1):
    """Soft contact probability per foot (N, 2) [left, right] from joint positions and velocities (N, 22,
    3): the larger of contact_weights of the ankle and of the foot joint (the ankle or foot joint rule of
    the spec's contact labels), heights above the floor (up coordinate = floor, a number or (N,))."""
    ids = torch.tensor([[J["left_ankle"], J["left_foot"]], [J["right_ankle"], J["right_foot"]]], device=p.device)
    h = p[:, ids, up] - _floor_per_point(floor, p, 2)
    w = contact_weights(h, v[:, ids].norm(dim=-1))                                # (N, 2 feet, 2 joints)
    return w.amax(-1)


def vertical_grf(traj, t, mass, sex="female", floor=0.0, up=1):
    """Per-foot vertical ground reaction (N, 2) [left, right] in newtons, and the contact
    weights (N, 2). The total is m (g + a_com,up); when neither foot is in contact the total
    is still returned on the weights' normalization floor, so the caller must mask flight.
    The centre of mass and the ankles come from one pose evaluation under nested jvp."""
    feet = [J["left_ankle"], J["right_ankle"]]

    def f(tt):
        pos = traj.pose_at(tt)[0]
        return com(pos, sex), pos[:, feet]

    (_, pos), (_, vel), (a, _) = second_order(f, t)
    total = mass * (G + a[:, up])
    h = pos[..., up] - floor
    w = contact_weights(h, vel.norm(dim=-1))
    share = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
    return total[:, None] * share, w


# ---------------------------------------------------------------- the floor in body weights
def newton_bw(com_acc, contact_prob, up=1):
    """Per-foot vertical ground reaction (N, 2) [left, right] in body weights: (1 + a_com,up / g) x share,
    share = contact_prob / (sum over feet), from com_acc (N, 3) m/s^2 and contact_prob (N, 2) in [0, 1].
    In flight (flight()) the split is undefined: the caller masks those points."""
    check_precision(com_acc, contact_prob)
    share = contact_prob / contact_prob.sum(-1, keepdim=True).clamp_min(1e-6)
    return (1.0 + com_acc[:, up] / G)[:, None] * share


def flight(contact_prob):
    """(N,) bool: both contact probabilities (N, 2) below 0.5."""
    return (contact_prob < 0.5).all(-1)


class ForceHead(torch.nn.Module):
    """The learned split of the Newton floor: share = softmax over feet of log p + r, with p = sigmoid of
    the decoder contact logits and r a zero-initialized MLP of foot heights and velocities. softmax(log p)
    = p / (p_L + p_R), so at initialization the head equals newton_bw(com_acc, sigmoid(logits)) exactly;
    the total stays the floor's 1 + a_com,up / g. Flight: both contact probabilities below 0.5."""

    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(6, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.SiLU(), torch.nn.Linear(hidden, 2))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, com_acc, contact_logits, foot_height, foot_vel, up=1):
        """com_acc (N, 3) m/s^2; contact_logits (N, 2); foot_height (N, 2) m above the floor; foot_vel
        (N, 2, 3) m/s. Returns pred_bw (N, 2) body weights, flight (N,) bool, share (N, 2)."""
        check_precision(com_acc, contact_logits, foot_height, foot_vel)
        vy = foot_vel[..., up]
        horiz = (foot_vel.pow(2).sum(-1) - vy.pow(2)).clamp_min(0.0).add(1e-12).sqrt()
        r = self.mlp(torch.cat([foot_height, vy, horiz], -1).to(self.mlp[0].weight.dtype)).to(com_acc.dtype)
        share = torch.softmax(F.logsigmoid(contact_logits) + r, -1)
        pred = (1.0 + com_acc[:, up] / G)[:, None] * share
        return pred, flight(torch.sigmoid(contact_logits)), share


# ---------------------------------------------------------------- record-level preparation
def unloaded_level(x, band):
    """Unloaded level of a foot total x (T,): the median of the samples within `band` of its 5th
    percentile (scripts/probe_stage1_sources.offset)."""
    x = np.asarray(x, dtype=np.float64)
    base = np.percentile(x, 5)
    return float(np.median(x[x <= base + band]))


def _dilate(mask, n):
    """Binary dilation of a (T,) bool mask by n samples on each side."""
    if n <= 0 or not mask.any():
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    T = len(mask)
    i = np.arange(T)
    return (c[np.minimum(i + n + 1, T)] - c[np.maximum(i - n, 0)]) > 0


def _long_runs(mask, n):
    """(T,) bool mask keeping only its True runs of at least n samples."""
    out = np.zeros_like(mask)
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= n:
            out[a:b] = True
    return out


def prepare_totals(total, present, fps, unit="N", *, coverage, valid=None, margin_s=PLATE_MARGIN_S,
                   min_contact_s=MIN_CONTACT_S):
    """Record-level correction of per-foot totals (module docstring), run once per record on the whole
    record before windows are cut. total (T, 2) native units [left, right]; present (2,) bool (sensor_mask
    column 8); fps Hz; unit 'N' or other; coverage, keyword and required, 'plate' (force plates: a foot is
    seen only on its plates) or 'insole' (always under the foot): the insole rule would keep a plate foot
    that never unloads, so there is no default; valid (T, 2) bool or None.

    Returns offset (2,) native units to subtract (the unloaded level where the foot unloads, else 0),
    usable (2,) bool, and valid (T, 2) bool, the samples force_nll may score. The unloaded band is BAND_N
    for newtons, else BAND_REL x (p99 - p5); a foot unloads when its unloaded level is at most
    UNLOADS_REL x its p99. Plate feet: usable only when they unload; scored within margin_s of a plate
    contact, a contact being a run of at least min_contact_s (at least one sample) of valid samples above
    offset + band."""
    if coverage not in ("plate", "insole"):
        raise ValueError(f"coverage must be 'plate' or 'insole', got {coverage!r}")
    x = np.asarray(total, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"total must be (T, 2), got {x.shape}")
    ok = np.isfinite(x) & (np.ones_like(x, bool) if valid is None else np.asarray(valid, dtype=bool))
    ok &= np.asarray(present, dtype=bool)[None]
    offset, usable = np.zeros(2), np.zeros(2, bool)
    out = np.zeros_like(ok)
    for j in range(2):
        xj = x[ok[:, j], j]
        if not len(xj):
            continue
        p5, p99 = np.percentile(xj, [5, 99])
        band = BAND_N if unit == "N" else BAND_REL * (p99 - p5)
        level = unloaded_level(xj, band)
        unloads = level <= UNLOADS_REL * p99
        offset[j] = level if unloads else 0.0
        if coverage == "insole":
            usable[j], out[:, j] = True, ok[:, j]
        elif unloads:
            contact = ok[:, j] & (np.nan_to_num(x[:, j]) - offset[j] > band)
            contact = _long_runs(contact, max(1, int(round(min_contact_s * fps))))
            usable[j] = True
            out[:, j] = ok[:, j] & _dilate(contact, int(round(margin_s * fps)))
    return offset, usable, out


def gain_prior(mass_kg, unit):
    """Prior centre of the gain (native units per BW): m g for a record in newtons with a known mass, NaN
    (flat prior) otherwise."""
    if unit == "N" and mass_kg is not None and math.isfinite(float(mass_kg)) and float(mass_kg) > 0:
        return float(mass_kg) * G
    return float("nan")


# ---------------------------------------------------------------- likelihood
def _t_nll(r, sigma, nu):
    """-log Student-t density of residuals r with scale sigma and nu degrees of freedom."""
    const = math.lgamma(nu / 2) - math.lgamma((nu + 1) / 2) + 0.5 * math.log(nu * math.pi)
    return torch.log(sigma) + const + 0.5 * (nu + 1) * torch.log1p((r / sigma) ** 2 / nu)


def force_nll(pred_bw, obs, w, sensor_mask, valid, log_sigma, weight=None, offset=None, record=None, gain0=None,
              flight=None, nu=4.0, floor=FLOOR_SIGMA_BW, prior_log_sd=PRIOR_LOG_SD, return_gain=False):
    """Negative log-likelihood (W,) of per-foot totals, and the gain (W,) of each window's record group when
    return_gain.

    pred_bw (N, 2) predicted totals in BW; obs (N, 2) observed totals in native units, or (N, 2, 9) (column
    8, the total, is read); w (N,) int64 window of each point; sensor_mask (W, 2, 9) bool (column 8: foot
    present) or (W, 2); valid (N, 2) bool (the record-level mask of prepare_totals at these samples);
    log_sigma, BW, broadcastable to (W,); weight (N,) point weights (default 1); offset (W, 2) native units
    from prepare_totals (default 0); record (W,) int64 groups of windows sharing one gain (default one per
    window); gain0 (W,) prior centre of the gain from gain_prior (NaN: flat); flight (N,) bool, masked.

    Per record group, the gain g (native units per BW) is the weighted least-squares ratio sum y p /
    sum p^2 over scored samples, y = obs - offset and p = pred_bw (swing samples, where p is near 0, carry
    no weight: the gain is profiled on stance). With a finite gain0, log g is the precision-weighted mean of
    log g_ls (precision sum p^2 / sigma^2) and log gain0 (precision 1 / prior_log_sd^2), and the prior term
    (log g - log gain0)^2 / (2 prior_log_sd^2) is added, split equally over the group's windows. Residuals
    y / g - p are in BW under a Student-t with scale sqrt(floor^2 + exp(2 log_sigma)), and each scored sample
    adds the log Jacobian log(g / y_ref) of y -> y / g, y_ref the weighted RMS of the group's scored y (data
    only). A uniform scale of the prediction is then absorbed by the gain and log_sigma, never rewarded
    (without the Jacobian, shrinking the prediction shrinks the residuals: plate groups, where each foot is
    scored at other times, would pay the head to load the unseen foot). The NLL is invariant to a change of
    the observation unit (with gain0 changed alike)."""
    check_precision(pred_bw)
    if obs.dim() == 3:
        obs = obs[..., 8]
    present = sensor_mask[..., 8] if sensor_mask.dim() == 3 else sensor_mask
    W = sensor_mask.shape[0]
    dev, dt = pred_bw.device, pred_bw.dtype
    w = w.to(dev)
    obs = obs.to(dev, dt)
    m = valid.to(dev, torch.bool) & present.to(dev, torch.bool)[w]
    if flight is not None:
        m = m & ~flight.to(dev, torch.bool)[:, None]
    wt = torch.ones(len(w), dtype=dt, device=dev) if weight is None else weight.to(dev, dt)
    y = obs if offset is None else obs - offset.to(dev, dt)[w]
    y = torch.where(m, y, torch.zeros((), dtype=dt, device=dev))
    ls = torch.as_tensor(log_sigma, dtype=dt, device=dev).expand(W)
    sigma = (floor ** 2 + torch.exp(2 * ls)).sqrt()[w][:, None]              # (N, 1)
    rec = torch.arange(W, device=dev) if record is None else record.to(dev)
    groups, gid_w = torch.unique(rec, return_inverse=True)
    G_n = len(groups)
    gid = gid_w[w]
    q = (wt[:, None] * m) / sigma ** 2                                          # (N, 2)
    S_yp = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, (q * y * pred_bw).sum(-1))
    S_pp = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, (q * pred_bw ** 2).sum(-1))
    S_yy = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, (q * y ** 2).sum(-1))
    S_11 = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, q.sum(-1))
    tiny = torch.finfo(dt).tiny
    info = S_pp > 0
    safe_pp = torch.where(info, S_pp, torch.ones_like(S_pp))
    g_ls = S_yp / safe_pp
    # a scale-covariant floor keeps the gain positive when predictions and observations disagree in sign
    g_ls = torch.maximum(g_ls, 1e-3 * (S_yy / safe_pp).clamp_min(tiny).sqrt())
    # no predicted load in the group: the RMS observation reads as 1 BW (flat prior) or the prior centre
    rms = (S_yy / S_11.clamp_min(tiny)).clamp_min(tiny).sqrt()
    g_ls = torch.where(info, g_ls, rms)
    u_ls = torch.log(g_ls)
    prior_g = torch.zeros(G_n, dtype=dt, device=dev)
    if gain0 is not None:
        g0 = torch.full((G_n,), float("nan"), dtype=dt, device=dev).index_put((gid_w,), gain0.to(dev, dt))
        has = torch.isfinite(g0) & (g0 > 0)
        u0 = torch.log(torch.where(has, g0, torch.ones_like(g0)))
        prec0 = 1.0 / prior_log_sd ** 2
        prec = torch.where(info, S_pp, torch.zeros_like(S_pp))
        u_map = (u_ls * prec + u0 * prec0) / (prec + prec0)
        u = torch.where(has, u_map, u_ls)
        prior_g = torch.where(has, 0.5 * (u - u0) ** 2 * prec0, prior_g)
    else:
        u = u_ls
    gain = torch.exp(u)[gid][:, None]
    r = y / gain - pred_bw
    # log Jacobian of y -> y / g per scored sample, relative to the RMS of the group's scored y (no parameter)
    q0 = (wt[:, None] * m).detach()
    S0_yy = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, (q0 * y.detach() ** 2).sum(-1))
    S0_11 = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid, q0.sum(-1))
    log_ref = 0.5 * torch.log((S0_yy / S0_11.clamp_min(tiny)).clamp_min(tiny))
    jac = (u - log_ref)[gid][:, None]                                           # (N, 1)
    nll = ((_t_nll(r, sigma, nu) + jac) * m).sum(-1) * wt
    out = torch.zeros(W, dtype=dt, device=dev).index_add(0, w, nll)
    count = torch.zeros(G_n, dtype=dt, device=dev).index_add(0, gid_w, torch.ones(W, dtype=dt, device=dev))
    out = out + (prior_g / count)[gid_w]
    return (out, torch.exp(u)[gid_w]) if return_gain else out
