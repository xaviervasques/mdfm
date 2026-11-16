"""Unit tests of the IMU kernel (design section 3) on a real AMASS motion and on a static pose.

The AMASS take is the standing KIT take of tests/amass_fixture.py (a train archive)."""
import os

import pytest
import torch

from amass_fixture import TAKE, assert_train_take
from mdfm import paths
from mdfm.body.rotations import Z_UP_TO_Y_UP, axis_angle_to_matrix, matrix_to_rot6d, skew

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


def _trajectory(cutoff=12.0):
    """A calm AMASS take on the 100 Hz body clock, low-passed as the synthetic factory does."""
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    seq = load_sequence(*TAKE)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=cutoff)
    g = seq["gender"] if seq["gender"] in ("male", "female", "neutral") else "neutral"
    return BodyTrajectory(SMPLHBody(g, dtype=torch.float64), r6, tr, betas, rate=100.0), 100.0


def test_static_pose_reads_gravity_and_no_rotation():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.kernels.imu import G, imu
    body = SMPLHBody("neutral", dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(50, 22, 3, 3).clone()
    traj = BodyTrajectory(body, matrix_to_rot6d(R), torch.zeros(50, 3, dtype=torch.float64),
                          torch.zeros(16, dtype=torch.float64))
    t = torch.linspace(0.05, 0.4, 20, dtype=torch.float64)
    acc, gyro = imu(traj, t, joint=4, offset=torch.tensor([0.0, -0.2, 0.05], dtype=torch.float64))
    assert torch.allclose(acc, torch.tensor([0.0, G, 0.0], dtype=torch.float64).expand_as(acc), atol=1e-9)
    assert gyro.abs().max() < 1e-9
    acc0, _ = imu(traj, t, joint=4, offset=torch.zeros(3, dtype=torch.float64), gravity_removed=True)
    assert acc0.abs().max() < 1e-9


def test_rigid_body_decomposition_and_gyro_on_amass():
    from mdfm.body.hermite import hermite  # noqa: F401
    traj, fps = _trajectory()
    # mid-interval times: a finite difference across a knot of the C1 interpolant is not exact
    t = (torch.arange(20, 160, 3, dtype=torch.float64) + 0.5) / fps
    r = torch.tensor([0.03, -0.12, 0.04], dtype=torch.float64)
    k = 4  # left knee joint: the sensor sits on the left shank
    p0, v0, a0, R, w, wdot = traj.point_kinematics(t, k, torch.zeros(3, dtype=torch.float64))
    ps, vs, as_, Rs, ws, _ = traj.point_kinematics(t, k, r)
    rw = (R @ r[:, None])[..., 0]
    pred = a0 + torch.cross(wdot, rw, dim=-1) + torch.cross(w, torch.cross(w, rw, dim=-1), dim=-1)
    assert (as_ - pred).abs().max() < 1e-6 * max(1.0, as_.abs().max().item())
    # angular velocity against a central finite difference of the rotation
    dt = 1e-6
    _, Rp = traj.point_at(t + dt, k, r)
    _, Rm = traj.point_at(t - dt, k, r)
    Rdot = (Rp - Rm) / (2 * dt)
    w_fd = torch.stack([(Rdot @ R.transpose(-1, -2))[:, 2, 1], (Rdot @ R.transpose(-1, -2))[:, 0, 2],
                        (Rdot @ R.transpose(-1, -2))[:, 1, 0]], -1)
    assert (w - w_fd).abs().max() < 1e-4 * max(1.0, w.abs().max().item())
    # velocity against a finite difference of the position
    pp, _ = traj.point_at(t + dt, k, r)
    pm, _ = traj.point_at(t - dt, k, r)
    assert ((pp - pm) / (2 * dt) - vs).abs().max() < 1e-5 * max(1.0, vs.abs().max().item())


def test_imu_readings_are_plausible_on_amass():
    from mdfm.kernels.imu import imu
    traj, fps = _trajectory()
    t = torch.linspace(0.1, 3.0, 291, dtype=torch.float64)
    acc, gyro = imu(traj, t, joint=4, offset=torch.tensor([0.0, -0.15, 0.05], dtype=torch.float64))
    norm = acc.norm(dim=-1)
    # a standing take (arm movements only): specific force stays near 1 g on the shank
    assert (norm.median() - 9.81).abs() < 1.0
    assert gyro.abs().max() < 20.0
    assert torch.isfinite(acc).all() and torch.isfinite(gyro).all()


def test_skew_cross_consistency():
    a, b = torch.randn(10, 3), torch.randn(10, 3)
    assert torch.allclose((skew(a) @ b[..., None])[..., 0], torch.cross(a, b, dim=-1), atol=1e-6)
