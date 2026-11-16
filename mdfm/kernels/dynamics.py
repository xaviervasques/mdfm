"""Leg inverse dynamics (design section 3, the torque input of the EMG kernel).

leg_torques() returns the net joint moments of hip, knee and ankle of both legs from the body
kinematics (mdfm.body.kinematics.BodyKin) and the vertical ground reaction per foot in body
weights, by bottom-up Newton-Euler over foot, shank and thigh (the link-segment method of
Winter, Biomechanics and Motor Control of Human Movement, 4th ed., Wiley 2009, chapter
"Kinetics: forces and moments of force"). For each segment with mass m, centre of mass c,
proximal joint P and distal load (F_d applied at D, free moment M_d):
  F_P = m (a_c - g) - F_d
  M_P = I alpha + omega x (I omega) - (D - c) x F_d - (P - c) x F_P - M_d
with the foot loaded by the ground reaction at the centre of pressure (no free moment) and each
upper segment loaded by minus the joint force and moment of the segment below. M_P is the net
moment the proximal segment exerts on the distal one (the internal joint moment), returned in
the parent segment frame (hip in the pelvis frame, knee in the thigh frame, ankle in the shank
frame), in N m per kg of body mass: segment masses are de Leva fractions and the ground reaction
enters as grf_bw x g per kg, so the mass cancels.

Segments are those of mdfm.kernels.force.DE_LEVA: thigh from hip to knee, shank from knee to
ankle, foot from ankle to the SMPL foot joint; their orientation, angular velocity and
acceleration are those of their proximal joint (the SMPL segment frames, x to the subject's left,
y up and z forward at rest). Inertia about the centre of mass from the de Leva radii of gyration
(1996, table 4, percent of segment length): the transverse radius about the medio-lateral axis
(the segment x axis made orthogonal to the long axis), the sagittal radius about the third axis
(antero-posterior at rest), the longitudinal radius about the long axis (proximal to distal
joint). The force kernel is vertical: grf_bw (N, 2) loads each foot vertically, and grf_vector adds
the horizontal whole-body reaction split by vertical load. The free moment is left out in the
pilot, and the default centre of pressure stays at the ankle-foot midpoint (it does not travel to
the forefoot at push-off).

Anatomical sign conventions on the SMPL axes (checked in tests/test_dynamics.py): a positive
rotation about x extends the hip, flexes the knee and plantarflexes the ankle, so hip flexion,
knee flexion and ankle dorsiflexion moments are FLEXION_SIGN x the x component; hip abduction
is +z for the left leg and -z for the right one. moment_channels() and joint_angles() return
the four channels of MOMENT_CHANNELS in these anatomical signs.
"""
import numpy as np
import torch

from ..body.kinematics import check_precision
from .force import DE_LEVA, J, _floor_per_point, sex_codes
from .imu import G

# hip, knee, ankle and foot joints of the left and right leg (SMPL)
LEGS = ((J["left_hip"], J["left_knee"], J["left_ankle"], J["left_foot"]),
        (J["right_hip"], J["right_knee"], J["right_ankle"], J["right_foot"]))
PELVIS = J["pelvis"]
JOINTS = ("hip", "knee", "ankle")
SEGMENTS = ("thigh", "shank", "foot")
# de Leva 1996, table 4: radii of gyration as fractions of segment length, (sagittal, transverse,
# longitudinal), female then male
RADII = {"thigh": ((0.369, 0.364, 0.162), (0.329, 0.329, 0.149)),
         "shank": ((0.271, 0.267, 0.093), (0.255, 0.249, 0.103)),
         "foot": ((0.299, 0.279, 0.139), (0.257, 0.245, 0.124))}
MOMENT_CHANNELS = ("hip_flexion", "knee_flexion", "ankle_dorsiflexion", "hip_abduction")
FLEXION_SIGN = (-1.0, 1.0, -1.0)       # x component to hip flexion, knee flexion, ankle dorsiflexion
ABDUCTION_SIGN = (1.0, -1.0)           # z component of the hip moment to abduction, left and right leg


def _segment_table():
    """mass (3, 3), COM fraction (3, 3), radii (3, 3, 3) as (sex code, segment[, sagittal, transverse,
    longitudinal]); code 2 is the mean of the female and male values."""
    rows = {name.split("_", 1)[1]: row for name, *row in DE_LEVA if name.startswith("left_")}
    mass, comf, radii = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3, 3))
    for k, seg in enumerate(SEGMENTS):
        _, _, mf, mm, cf, cm = rows[seg]
        rf, rm = RADII[seg]
        mass[:, k] = (mf, mm, 0.5 * (mf + mm))
        comf[:, k] = (cf, cm, 0.5 * (cf + cm))
        radii[:, k] = (rf, rm, tuple(0.5 * (a + b) for a, b in zip(rf, rm)))
    return mass, comf, radii


MASS, COMF, RADII_T = _segment_table()


def _per_point(table, sex, n, like):
    t = torch.as_tensor(table, dtype=like.dtype, device=like.device)
    code = sex_codes(2 if sex is None else sex)
    if torch.is_tensor(code):
        return t[code.to(like.device)]
    return t[code].expand(n, *t.shape[1:])


def _inertia_times(e, r2m, v):
    """sum_k r2m_k e_k (e_k . v): the world inertia tensor sum_k r2m_k e_k e_k^T applied to v (..., 3)."""
    return sum(r2m[..., k, None] * e[k] * (e[k] * v).sum(-1, keepdim=True) for k in range(3))


def grf_vector(grf_bw, com_acc, up=1):
    """Ground reaction vector per foot (N, 2, 3) in body weights: the vertical totals grf_bw (N, 2) plus the
    horizontal whole-body reaction a_com,h / g (from com_acc (N, 3) m/s^2) split in proportion to the
    vertical loads, which is exact in single support."""
    check_precision(grf_bw, com_acc)
    e_up = torch.zeros(3, dtype=com_acc.dtype, device=com_acc.device)
    e_up[up] = 1.0
    horiz = (com_acc - com_acc[:, up, None] * e_up) / G                 # (N, 3)
    share = grf_bw / grf_bw.sum(-1, keepdim=True).clamp_min(1e-6)
    return grf_bw[..., None] * e_up + share[..., None] * horiz[:, None]


def leg_torques(kin, grf_bw, cop=None, sex=None, floor=0.0, up=1):
    """Net joint moments tau (N, 2 legs [left, right], 3 joints [hip, knee, ankle], 3) in N m per kg, in
    the parent segment frame (module docstring). kin: BodyKin of the 22 joints at N points; grf_bw (N, 2)
    vertical ground reaction per foot in body weights, or (N, 2, 3) the reaction vector (grf_vector); cop
    (N, 2, 3) m the centre of pressure, default the midpoint of the ankle and foot joints placed on the
    floor (up coordinate = floor, a number or (N,)); sex (N,) int64 codes or one label (None: unknown, the
    mean of the de Leva tables)."""
    check_precision(kin.p, grf_bw)
    N = kin.p.shape[0]
    legs = torch.tensor(LEGS, device=kin.p.device)                       # (2, 4)
    P, A = kin.p[:, legs], kin.a[:, legs]                                 # (N, 2, 4, 3)
    R, Wv, Al = kin.R[:, legs], kin.omega[:, legs], kin.alpha[:, legs]
    mass = _per_point(MASS, sex, N, kin.p)[:, None]                       # (N, 1, 3)
    comf = _per_point(COMF, sex, N, kin.p)[:, None]
    radii = _per_point(RADII_T, sex, N, kin.p)[:, None]                   # (N, 1, 3, 3)
    e_up = torch.zeros(3, dtype=kin.p.dtype, device=kin.p.device)
    e_up[up] = 1.0
    g_vec = -G * e_up
    if cop is None:
        cop = 0.5 * (P[:, :, 2] + P[:, :, 3])
        cop = cop - (cop[..., up] - _floor_per_point(floor, kin.p, 1))[..., None] * e_up
    F_d = (grf_bw if grf_bw.dim() == 3 else grf_bw[..., None] * e_up) * G   # (N, 2, 3) N/kg
    D = cop
    M_d = torch.zeros_like(F_d)
    moments = [None, None, None]
    for s in (2, 1, 0):                                                   # foot, shank, thigh
        p0, p1 = P[:, :, s], P[:, :, s + 1]
        f = comf[..., s, None]
        c = p0 + f * (p1 - p0)
        a_c = A[:, :, s] + f * (A[:, :, s + 1] - A[:, :, s])
        m = mass[..., s, None]
        bone = p1 - p0
        L2 = (bone * bone).sum(-1, keepdim=True)
        u = bone / L2.sqrt()
        x = R[:, :, s, :, 0]                                              # segment x axis in the world
        e1 = x - (x * u).sum(-1, keepdim=True) * u
        e1 = e1 / e1.norm(dim=-1, keepdim=True)
        e2 = torch.cross(u, e1, dim=-1)
        rad = radii[:, :, s]                                              # (N, 1, 3): sagittal, transverse, long
        r2m = m * L2 * torch.stack([rad[..., 1], rad[..., 0], rad[..., 2]], -1) ** 2
        e = (e1, e2, u)                                                   # medio-lateral, antero-posterior, long
        om, al = Wv[:, :, s], Al[:, :, s]
        H = _inertia_times(e, r2m, al) + torch.cross(om, _inertia_times(e, r2m, om), dim=-1)
        F_p = m * (a_c - g_vec) - F_d
        M_p = H - torch.cross(D - c, F_d, dim=-1) - torch.cross(p0 - c, F_p, dim=-1) - M_d
        moments[s] = M_p
        F_d, D, M_d = -F_p, p0, -M_p
    M = torch.stack(moments, 2)                                           # (N, 2, 3, 3) world
    parent = torch.stack([kin.R[:, PELVIS, None].expand(N, 2, 3, 3), R[:, :, 0], R[:, :, 1]], 2)
    return (parent.transpose(-1, -2) @ M[..., None])[..., 0]


def moment_channels(tau):
    """(N, 2, 4) N m/kg of MOMENT_CHANNELS in anatomical signs from leg_torques output (N, 2, 3, 3)."""
    s = torch.tensor(FLEXION_SIGN, dtype=tau.dtype, device=tau.device)
    ab = torch.tensor(ABDUCTION_SIGN, dtype=tau.dtype, device=tau.device)
    return torch.cat([tau[..., 0] * s, (tau[:, :, 0, 2] * ab)[..., None]], -1)


def joint_angles(kin):
    """Angles (N, 2, 4) rad and rates (N, 2, 4) rad/s of MOMENT_CHANNELS in anatomical signs: from the
    rotation of each segment relative to its parent, R_rel = R_parent^T R_child, the angle about x
    (atan2(R_rel[2, 1], R_rel[1, 1])) and, for the hip, about z (atan2(R_rel[1, 0], R_rel[0, 0])); rates are
    the x and z components of the relative angular velocity in the parent frame."""
    check_precision(kin.p)
    N = kin.p.shape[0]
    legs = torch.tensor(LEGS, device=kin.p.device)
    Rl, Wl = kin.R[:, legs[:, :3]], kin.omega[:, legs[:, :3]]            # (N, 2, 3, ...): thigh, shank, foot
    Rp = torch.cat([kin.R[:, PELVIS, None, None].expand(N, 2, 1, 3, 3), Rl[:, :, :2]], 2)
    Wp = torch.cat([kin.omega[:, PELVIS, None, None].expand(N, 2, 1, 3), Wl[:, :, :2]], 2)
    Rrel = Rp.transpose(-1, -2) @ Rl
    wrel = (Rp.transpose(-1, -2) @ (Wl - Wp)[..., None])[..., 0]
    s = torch.tensor(FLEXION_SIGN, dtype=kin.p.dtype, device=kin.p.device)
    ab = torch.tensor(ABDUCTION_SIGN, dtype=kin.p.dtype, device=kin.p.device)
    flex = torch.atan2(Rrel[..., 2, 1], Rrel[..., 1, 1]) * s
    abd = torch.atan2(Rrel[:, :, 0, 1, 0], Rrel[:, :, 0, 0, 0]) * ab
    angles = torch.cat([flex, abd[..., None]], -1)
    rates = torch.cat([wrel[..., 0] * s, (wrel[:, :, 0, 2] * ab)[..., None]], -1)
    return angles, rates
