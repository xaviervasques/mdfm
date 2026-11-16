"""Unit tests of the camera and force kernels (design section 3).

The AMASS take is the standing KIT take of tests/amass_fixture.py (a train archive)."""
import os

import pytest
import torch

from amass_fixture import TAKE, assert_train_take
from mdfm import paths
from mdfm.kernels.camera import SMPL_TO_CANONICAL_15, intrinsics, keypoint_nll, look_at, project

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


def test_projection_geometry():
    K = intrinsics(1000.0, 1000.0, 640.0, 360.0)
    R, t = look_at((0.0, 1.0, 3.0), (0.0, 1.0, 0.0))
    pts = torch.tensor([[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.5, 0.0]], dtype=torch.float64)
    uv, z = project(pts, K, R, t)
    assert torch.allclose(z, torch.full((3,), 3.0, dtype=torch.float64))
    assert torch.allclose(uv[0], torch.tensor([640.0, 360.0], dtype=torch.float64))
    # x right and y down in the image: a point to the camera's right and a higher point
    assert uv[1, 0] > 640 and uv[2, 1] < 360
    assert torch.allclose((uv[1] - uv[0]).norm(), torch.tensor(1000.0 * 0.5 / 3.0, dtype=torch.float64))


def test_keypoint_nll_prefers_truth_and_downweights_low_confidence():
    torch.manual_seed(0)
    obs = torch.rand(15, 2, dtype=torch.float64) * 500
    conf = torch.full((15,), 0.9, dtype=torch.float64)
    valid = torch.ones(15, dtype=torch.float64)
    assert keypoint_nll(obs, obs, conf, valid) < keypoint_nll(obs + 20, obs, conf, valid)
    off = obs.clone()
    off[3] += 300
    lo = conf.clone()
    lo[3] = 0.1
    # an outlier on a low-confidence joint costs less than on a confident one
    assert keypoint_nll(off, obs, lo, valid) - keypoint_nll(obs, obs, lo, valid) < \
        keypoint_nll(off, obs, conf, valid) - keypoint_nll(obs, obs, conf, valid)
    masked = valid.clone()
    masked[3] = 0
    assert torch.allclose(keypoint_nll(off, obs, conf, masked), keypoint_nll(obs, obs, conf, masked))


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_body_projects_into_the_image_and_gradients_flow():
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    body = SMPLHBody("neutral", dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(10, 22, 3, 3).clone()
    r6 = matrix_to_rot6d(R).requires_grad_(True)
    traj = BodyTrajectory(body, r6, torch.zeros(10, 3, dtype=torch.float64), torch.zeros(16, dtype=torch.float64))
    pos, _ = traj.pose_at(torch.tensor([0.02], dtype=torch.float64))
    K = intrinsics(1000.0, 1000.0, 640.0, 360.0)
    Rc, tc = look_at((0.0, 0.3, 4.0), (0.0, 0.3, 0.0))
    uv, z = project(pos[:, SMPL_TO_CANONICAL_15], K, Rc, tc)
    assert (uv[..., 0] > 0).all() and (uv[..., 0] < 1280).all() and (uv[..., 1] > 0).all() and (uv[..., 1] < 720).all()
    loss = keypoint_nll(uv, uv.detach() + 5, torch.ones(1, 15, dtype=torch.float64), torch.ones(1, 15, dtype=torch.float64))
    loss.backward()
    assert r6.grad is not None and torch.isfinite(r6.grad).all() and r6.grad.abs().sum() > 0


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_static_standing_gives_body_weight_on_both_feet():
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.kernels.force import vertical_grf
    from mdfm.kernels.imu import G
    body = SMPLHBody("neutral", dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(30, 22, 3, 3).clone()
    betas = torch.zeros(16, dtype=torch.float64)
    # put the feet on the floor: lift the body so the lower ankle sits 3 cm above y = 0
    ank = body.rest_joints(betas[None])[0, [7, 8], 1].min()
    tr = torch.zeros(30, 3, dtype=torch.float64)
    tr[:, 1] = -ank + 0.03
    traj = BodyTrajectory(body, matrix_to_rot6d(R), tr, betas)
    t = torch.linspace(0.02, 0.25, 12, dtype=torch.float64)
    f, w = vertical_grf(traj, t, mass=70.0)
    assert torch.allclose(f.sum(-1), torch.full((12,), 70.0 * G, dtype=torch.float64), rtol=1e-9)
    # both feet on the floor carry a share (the rest-pose ankles differ by millimetres)
    share = f[:, 0] / f.sum(-1)
    assert ((share > 0.3) & (share < 0.7)).all()


def test_contact_weights_follow_height_and_speed():
    from mdfm.kernels.force import contact_weights
    h = torch.tensor([0.0, 0.3, 0.0], dtype=torch.float64)
    v = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    w = contact_weights(h, v)
    assert w[0] > 0.9 and w[1] < 1e-6 and w[2] < 1e-6


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_force_follows_centre_of_mass_acceleration_on_amass():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    from mdfm.kernels.force import com_kinematics, vertical_grf
    from mdfm.kernels.imu import G
    seq = load_sequence(*TAKE)
    body = SMPLHBody(seq["gender"], dtype=torch.float64)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=body)
    traj = BodyTrajectory(body, r6, tr, betas)
    t = torch.linspace(0.2, 2.8, 131, dtype=torch.float64)
    f, w = vertical_grf(traj, t, mass=60.0, sex="female")
    _, _, a = com_kinematics(traj, t, "female")
    assert torch.allclose(f.sum(-1), 60.0 * (G + a[:, 1]), rtol=1e-9)
    # a standing take: the total stays within 20 percent of body weight
    assert ((f.sum(-1) / (60.0 * G) - 1).abs() < 0.2).float().mean() > 0.9
