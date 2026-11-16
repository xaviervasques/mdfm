"""Window frame, floor, foot contacts and the treadmill belt frame of a body on the 100 Hz clock (spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 2.0 and 2.1, task B4.4).

World: y up, metres. Window frame: floor at y = 0; at the window's centre knot the pelvis heading (the
root's SMPL rest +z axis projected on the floor) is +z and the pelvis x and z are 0. A frame is a yaw
R_yaw (3, 3) about +y and an origin (3,) on the floor: x_frame = R_yaw (x - origin), R_frame = R_yaw R.

Floor: the 2nd percentile over a take of the lowest foot joint (SMPL joints 10 and 11, which sit inside
the foot, so the floor is a joint floor, the one the plausibility term and the contact rule use).
Contact of a foot: its ankle (7, 8) or foot (10, 11) joint within h0 of the floor and slower than v0
(3D speed), the thresholds of mdfm.kernels.force.contact_weights.

Treadmill: the stance foot is the foot whose foot joint is lower, a stance knot one where that joint is
within h0 of the floor. A take is a treadmill take when it is steady upright gait on a moving floor
[cfg]: the median horizontal speed of the stance foot over its stance knots exceeds TREADMILL_FOOT, the
median horizontal pelvis speed stays below TREADMILL_PELVIS, the trunk is upright (median cosine between
the pelvis-to-neck vector and +y above TREADMILL_UPRIGHT) and the stance foot alternates (each foot is
the stance foot on at least TREADMILL_BALANCE of the stance knots). The spec's pelvis bound of 0.2 m/s
is raised to 0.3: BMLrub treadmill jogs sway above 0.2 (reports/body_store/summary.md counts them), and
the two gait conditions reject the swimming, lying and one-footed takes that the speeds alone let
through. The belt frame is a Galilean shift of constant velocity, transl - v_belt k / rate, with v_belt
the mean horizontal velocity of the foot joints in contact with the belt (within h0 of the floor and
slower than v0 relative to the belt, a fixed point reached from the median stance-foot velocity):
rotations and accelerations are unchanged (Hermite reproduces a linear term exactly), so rendered IMU
signals are too, and the stance foot comes to rest at mid-stance. The stance knots (lower foot within h0)
also hold touch-down and push-off, a larger share of a jog's short contacts than of a walk's, so the median
stance-foot speed of a jog stays above that of a walk (reports/body_store/summary.md measures both, and the
mid-stance velocity, on the served takes).

Joint velocities, when not given, are central differences of the knots (one-sided at the ends); the
body store passes the exact Hermite velocities of mdfm.body.kinematics.
"""
import torch

from .rotations import rot_y, yaw_angle

UP = 1
PELVIS = 0
NECK = 12
ANKLES = (7, 8)             # left, right
FEET = (10, 11)             # left, right: SMPL foot joints, inside the foot
H0 = 0.05                   # m above the floor, kernels/force.contact_weights
V0 = 0.3                    # m/s, kernels/force.contact_weights
FLOOR_Q = 0.02              # quantile of the lowest foot joint
TREADMILL_FOOT = 0.3        # m/s, median stance-foot horizontal speed above it [cfg]
TREADMILL_PELVIS = 0.3      # m/s, median pelvis horizontal speed below it [cfg; the spec's 0.2 missed jogs]
TREADMILL_UPRIGHT = 0.8     # median cosine of the pelvis-to-neck vector with +y above it [cfg]
TREADMILL_BALANCE = 0.1     # share of the stance knots on each foot, at least [cfg]
BELT_ITERS = 3              # contact-consistent refinements of the belt velocity
# canonical 15 joints (MoveAtlas configs/canonical_skeleton.yaml): left_hip 9, right_hip 10
C15_LEFT_HIP, C15_RIGHT_HIP = 9, 10


# ---------------------------------------------------------------- window frame
def heading_frame(joints, R_root, centre):
    """joints (B, T, J, 3) m with joint 0 the pelvis (J >= 1), R_root (B, T, 3, 3) root orientation,
    centre an int or (B,) int64 knot -> R_yaw (B, 3, 3), origin (B, 3) m: at the centre knot the
    frame puts the pelvis heading on +z and the pelvis x and z at 0; origin y is 0, so the floor
    stays at y = 0. The heading is undefined when the root's +z axis is vertical (yaw 0 then)."""
    B = joints.shape[0]
    c = torch.as_tensor(centre, dtype=torch.int64, device=joints.device).expand(B)
    rows = torch.arange(B, device=joints.device)
    yaw = yaw_angle(R_root[rows, c])
    pelvis = joints[rows, c, PELVIS]
    origin = torch.stack([pelvis[:, 0], torch.zeros_like(pelvis[:, 0]), pelvis[:, 2]], -1)
    return rot_y(-yaw), origin


def apply_frame(x, R_yaw, origin):
    """Points x (B, ..., 3) m -> R_yaw (x - origin), per row b."""
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (3,)
    return torch.einsum("bij,b...j->b...i", R_yaw, x - origin.reshape(shape))


def rotate_frame(R, R_yaw):
    """Rotations R (B, ..., 3, 3) -> R_yaw R, per row b."""
    return torch.einsum("bij,b...jk->b...ik", R_yaw, R)


def frame_knots(rot6d, transl, rest_root, R_yaw, origin):
    """Body knots in a frame: rot6d (B, T, 22, 6) (row 0 the root orientation), transl (B, T, 3) m,
    rest_root (B, 3) m the rest pelvis of each row's subject -> rot6d, transl such that forward
    kinematics gives apply_frame of the original joints: root columns rotated by R_yaw (Gram-Schmidt
    commutes with a rotation), transl' = R_yaw (transl + rest_root - origin) - rest_root."""
    B = rot6d.shape[0]
    root = rot6d[:, :, 0].reshape(B, -1, 2, 3)
    root = torch.einsum("bij,btkj->btki", R_yaw, root).reshape(B, -1, 6)
    r6 = torch.cat([root[:, :, None], rot6d[:, :, 1:]], 2)
    tr = apply_frame(transl + rest_root[:, None], R_yaw, origin) - rest_root[:, None]
    return r6, tr


def hips_heading(coords15, valid, centre=None):
    """Heading (rad, about +y from +z) of a canonical 15-joint pose (T, 15, 3), y up, at the frame
    nearest `centre` (default T // 2) where both hips are valid (valid (T, 15) bool): forward is the
    hip line (left minus right) crossed with +y, projected on the floor. ValueError when no frame has
    both hips valid."""
    coords15 = torch.as_tensor(coords15)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    T = coords15.shape[0]
    ok = valid[:, C15_LEFT_HIP] & valid[:, C15_RIGHT_HIP]
    if not bool(ok.any()):
        raise ValueError("no frame with both hips valid")
    c = T // 2 if centre is None else int(centre)
    frames = torch.nonzero(ok)[:, 0]
    k = frames[torch.argmin((frames - c).abs())]
    hip = coords15[k, C15_LEFT_HIP] - coords15[k, C15_RIGHT_HIP]
    # forward = hip x (0, 1, 0) = (-hip_z, 0, hip_x)
    return torch.atan2(-hip[2], hip[0])


# ---------------------------------------------------------------- floor and contacts
def _velocity(joints, rate, vel):
    """Joint velocities (..., T, 22, 3) m/s: `vel` when given, else central differences along T."""
    if vel is not None:
        return vel
    return torch.gradient(joints, spacing=1.0 / rate, dim=-3)[0]


def floor_height(joints):
    """joints (..., T, 22, 3) m -> (...) m: the FLOOR_Q quantile over T of the lowest foot joint
    (the lower of SMPL joints 10 and 11 at each knot)."""
    low = torch.minimum(joints[..., FEET[0], UP], joints[..., FEET[1], UP])
    return torch.quantile(low, FLOOR_Q, dim=-1)


def contact_labels(joints, rate=100.0, h0=H0, v0=V0, vel=None, floor=None):
    """joints (B, T, 22, 3) m -> (B, T, 2) bool [left, right]: the ankle or the foot joint of a foot
    within h0 of the floor and slower than v0 (3D speed). vel (B, T, 22, 3) m/s, default central
    differences at `rate`; floor (B,) m or a scalar, default floor_height of each row. A single take
    (T, 22, 3) with a scalar floor gives (T, 2)."""
    vel = _velocity(joints, rate, vel)
    if floor is None:
        floor = floor_height(joints)
    floor = torch.as_tensor(floor, dtype=joints.dtype, device=joints.device)
    floor = floor.reshape(floor.shape + (1, 1))           # per row over (T, 22); a scalar broadcasts
    h = joints[..., UP] - floor
    s = vel.norm(dim=-1)
    near = (h < h0) & (s < v0)
    a, f = list(ANKLES), list(FEET)
    return near[..., a] | near[..., f]


# ---------------------------------------------------------------- treadmill
def _horizontal(v):
    """(..., 3) -> (..., 3) with the up component zeroed."""
    return v * torch.tensor([1.0, 0.0, 1.0], dtype=v.dtype, device=v.device)


def stance(joints, floor=None, h0=H0):
    """joints (T, 22, 3) -> stance foot (T,) int64 (0 left, 1 right: the lower foot joint) and the
    stance knots (T,) bool (that joint within h0 of the floor, floor default floor_height)."""
    if floor is None:
        floor = floor_height(joints)
    y = joints[:, list(FEET), UP]
    side = (y[:, 1] < y[:, 0]).long()
    low = y.gather(1, side[:, None])[:, 0]
    return side, (low - floor) < h0


def trunk_upright(joints):
    """joints (T, 22, 3) -> median over T of the cosine between the pelvis-to-neck vector and +y (1 for
    an upright trunk, about 0 lying)."""
    trunk = joints[:, NECK] - joints[:, PELVIS]
    return float((trunk[:, UP] / trunk.norm(dim=-1).clamp_min(1e-9)).median())


def treadmill_rule(stats, foot=TREADMILL_FOOT, pelvis=TREADMILL_PELVIS, upright=TREADMILL_UPRIGHT,
                   balance=TREADMILL_BALANCE):
    """True when a treadmill_stats dict is steady upright gait on a moving floor: stance-foot speed above
    `foot` m/s, pelvis speed below `pelvis` m/s, trunk cosine above `upright`, each foot the stance foot
    on at least `balance` of the stance knots."""
    return bool(stats["n_stance"] > 0 and stats["foot_speed"] > foot and stats["pelvis_speed"] < pelvis
                and stats["upright"] > upright and stats["balance"] >= balance)


def treadmill_stats(joints, rate=100.0, vel=None, floor=None, h0=H0, v0=V0, iters=BELT_ITERS):
    """joints (T, 22, 3) m -> dict: foot_speed (m/s) the median horizontal speed of the stance foot
    over its stance knots (nan without one), pelvis_speed (m/s) the median horizontal pelvis speed,
    n_stance the number of stance knots, balance the smaller share of the stance knots on one foot (0.5
    for even alternation, 0 when one foot never stands), upright (trunk_upright), treadmill
    (treadmill_rule with the default thresholds), and belt (3,) m/s, the horizontal velocity that brings
    the stance foot to rest (zero without a stance knot): first the per-axis median of the stance-foot
    velocity over the stance knots, then `iters` times the mean horizontal velocity of the foot joints
    (10, 11) in contact in the frame of the current estimate (within h0 of the floor and slower than
    v0 relative to it), which leaves out the flight and swing knots of a jog."""
    vel = _velocity(joints, rate, vel)
    if floor is None:
        floor = floor_height(joints)
    side, on = stance(joints, floor, h0)
    feet = _horizontal(vel[:, list(FEET)])
    v_st = feet.gather(1, side[:, None, None].expand(-1, 1, 3))[:, 0]
    pelvis = _horizontal(vel[:, PELVIS]).norm(dim=-1)
    n = int(on.sum())
    belt = torch.zeros(3, dtype=joints.dtype, device=joints.device)
    foot_speed, balance = float("nan"), 0.0
    if n:
        foot_speed = float(v_st[on].norm(dim=-1).median())
        right = int(side[on].sum())
        balance = min(right, n - right) / n
        belt = v_st[on].median(0).values
        low = (joints[:, list(FEET), UP] - floor) < h0
        full = vel[:, list(FEET)]
        for _ in range(iters):
            inside = low & ((full - belt).norm(dim=-1) < v0)
            if not bool(inside.any()):
                break
            belt = feet[inside].mean(0)
    out = {"foot_speed": foot_speed, "pelvis_speed": float(pelvis.median()), "belt": belt, "n_stance": n,
           "balance": balance, "upright": trunk_upright(joints)}
    out["treadmill"] = treadmill_rule(out)
    return out


def is_treadmill(joints, rate=100.0, vel=None, floor=None, **thresholds):
    """joints (T, 22, 3) m -> True for steady upright gait on a moving floor (treadmill_rule; thresholds
    foot, pelvis, upright, balance override the defaults)."""
    return treadmill_rule(treadmill_stats(joints, rate, vel, floor), **thresholds)


def belt_velocity(joints, rate=100.0, vel=None, floor=None):
    """(3,) m/s: the horizontal belt velocity, the mean velocity of the foot joints in contact on the
    belt (treadmill_stats)."""
    return treadmill_stats(joints, rate, vel, floor)["belt"]


def galilean_shift(x, v, rate=100.0):
    """x (T, ..., 3) sampled at k / rate -> x - v k / rate, v (3,) m/s."""
    k = torch.arange(x.shape[0], dtype=x.dtype, device=x.device) / rate
    return x - k.reshape((-1,) + (1,) * (x.dim() - 1)) * v.to(x.dtype)


def belt_frame(transl, joints, rate=100.0, vel=None, floor=None):
    """transl (T, 3) m of a treadmill take -> transl in the belt frame: the constant-velocity shift that
    removes the mean stance-foot velocity (belt_velocity). IMU signals are invariant to it."""
    return galilean_shift(transl, belt_velocity(joints, rate, vel, floor), rate)
