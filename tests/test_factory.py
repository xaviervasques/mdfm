"""Tests of the virtual IMU factory against the single-sensor kernel and the canonical contract.

The AMASS take is the standing KIT take of tests/amass_fixture.py (a train archive)."""
import os

import pytest
import torch

from amass_fixture import TAKE, assert_train_take
from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


def _traj():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    seq = load_sequence(*TAKE)
    body = SMPLHBody(seq["gender"], dtype=torch.float64)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=body)
    return BodyTrajectory(body, r6, tr, betas)


def test_factory_matches_single_sensor_kernel():
    from mdfm.factory.imu_render import G0, SITES, render, site_table
    from mdfm.kernels.imu import imu
    traj = _traj()
    t = torch.linspace(0.3, 2.5, 23, dtype=torch.float64)
    sig, loc, ch = render(traj, t)
    assert sig.shape == (23, 14, 9) and loc.all() and ch[:, :6].all() and not ch[:, 6:].any()
    table = site_table(traj.body.rest_joints(traj.betas[None])[0, :22])
    for s in ("shank_L", "wrist_R", "lower_back", "foot_R"):
        k, off = table[s]
        acc, gyro = imu(traj, t, joint=k, offset=off)
        j = SITES.index(s)
        assert torch.allclose(sig[:, j, 0:3].double(), acc / G0, atol=1e-5)
        assert torch.allclose(sig[:, j, 3:6].double(), torch.rad2deg(gyro), atol=1e-3)


def test_standing_reads_one_g_everywhere_and_device_model_behaves():
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.factory.imu_render import Device, render
    body = SMPLHBody("neutral", dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(40, 22, 3, 3).clone()
    traj = BodyTrajectory(body, matrix_to_rot6d(R), torch.zeros(40, 3, dtype=torch.float64),
                          torch.zeros(16, dtype=torch.float64))
    t = torch.linspace(0.05, 0.3, 26, dtype=torch.float64)
    sig, _, _ = render(traj, t)
    norm = sig[..., 0:3].norm(dim=-1)
    assert torch.allclose(norm, torch.full_like(norm, 9.81 / 9.80665), atol=1e-5)
    assert sig[..., 3:6].abs().max() < 1e-6
    gen = torch.Generator().manual_seed(0)
    noisy, _, _ = render(traj, t, device=Device(), generator=gen)
    d = (noisy - sig)[..., 0:3]
    assert 0.001 < d.std() < 0.1
    sat, _, _ = render(traj, t, device=Device(acc_range_g=0.5), generator=torch.Generator().manual_seed(1))
    assert sat[..., 0:3].abs().max() <= 0.5 + 1e-6


def test_partial_layout_masks_absent_sites():
    from mdfm.factory.imu_render import SITES, render
    traj = _traj()
    t = torch.linspace(0.3, 0.6, 4, dtype=torch.float64)
    sig, loc, ch = render(traj, t, sites=["lower_back", "wrist_L"])
    assert loc.sum() == 2 and loc[SITES.index("lower_back")] and loc[SITES.index("wrist_L")]
    assert sig[:, ~loc].abs().max() == 0
