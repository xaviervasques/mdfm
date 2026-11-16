"""The open body store's fit (plan docs/MODEL_OPEN_PORT_PLAN.md, step M3; mdfm/body/openfit.py and
mdfm/data/open_mocap_raw.py): the 100 Hz clock, the cleaning, the up axis, and a round trip on synthetic landmarks."""
import os

import numpy as np
import pytest
import torch

from mdfm.body import openbody as BO
from mdfm.body import openfit as OF
from mdfm.data import open_mocap_raw as OR

needs_basis = pytest.mark.skipif(not os.path.exists(BO.BASIS_JSON), reason="scripts/build_open_rest_basis.py not run")


def test_up_axes_turn_to_y_and_stay_rotations():
    for ax in ("+x", "-x", "+y", "-y", "+z", "-z"):
        R = OR.up_to_y(ax)
        v = np.zeros(3)
        v["xyz".index(ax[1])] = 1.0 if ax[0] == "+" else -1.0
        assert np.allclose(R @ v, [0, 1, 0]) and np.isclose(np.linalg.det(R), 1.0) and np.allclose(R @ R.T, np.eye(3))


def test_the_body_clock_bridges_short_gaps_and_keeps_long_ones():
    fps = 200.0
    t = np.arange(400) / fps
    x = np.zeros((400, 2, 3))
    x[:, 0, 0] = np.sin(2 * np.pi * 1.0 * t)
    x[:, 1, 1] = t
    x[100:110, 0] = np.nan            # 0.05 s: bridged
    x[200:300, 1] = np.nan            # 0.5 s: stays missing
    y, v = OF.to_body_clock(x, fps)
    assert y.shape[0] == int(np.floor(399 / fps * 100)) + 1
    assert v[:, 0].all() and np.allclose(y[:, 0, 0], np.sin(2 * np.pi * np.arange(y.shape[0]) / 100.0), atol=2e-2)
    k = np.arange(y.shape[0]) / 100.0
    gap = (k > 200 / fps + 0.01) & (k < 299 / fps - 0.01)
    assert not v[gap, 1].any() and v[~((k >= 199 / fps) & (k <= 300 / fps)), 1].all()
    assert np.isnan(y[~v]).all()


def test_clean_masks_far_and_absurd_samples():
    x = np.zeros((5, 15, 3))
    x[:, 1, 1] = 0.5                 # neck 0.5 m above the pelvis
    x[2, 4] = [10.0, 0, 0]           # 20 torso lengths away
    x[3, 5] = [2e3, 0, 0]            # beyond a kilometre
    y, n = OF.clean(x)
    assert np.isnan(y[2, 4]).all() and np.isnan(y[3, 5]).all() and n == {"absurd": 1, "far": 1}
    assert np.isfinite(y[0]).all()


def test_toe_rules():
    labels = ["L_FM1", "L_FM5", "R_FM2", "R_FM1"]
    pts = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    t = OR.toe_landmarks("umons_taichi", labels, pts)
    assert np.allclose(t[:, 0], 0.5 * (pts[:, 0] + pts[:, 1])) and np.allclose(t[:, 1], pts[:, 2])
    assert np.isnan(OR.toe_landmarks("cmu_mocap", labels, pts)).all()


@needs_basis
def test_the_fit_recovers_synthetic_landmarks():
    from mdfm.model.prior import PhenotypePrior
    torch.manual_seed(0)
    basis, prior = BO.load_basis(), PhenotypePrior()
    u_true = torch.tensor([0.3, 0.2, -0.4, 0.1, 0.6, 0.0])
    truth = OF.Fit(30, basis, prior, u0=u_true)
    with torch.no_grad():
        truth.aa.copy_(0.25 * torch.randn(30, 22, 3))
        truth.aa[:, list(OF.UNOBSERVED)] = 0.0
        truth.aa[:, 0] = torch.tensor([0.0, 0.5, 0.0]) + 0.1 * torch.randn(30, 3)
        truth.tr.copy_(torch.randn(30, 3) * 0.2)
        target, _, _ = truth.predict()
    valid = torch.ones(30, OF.N_LM, dtype=torch.bool)
    u, o, m = OF.fit_person((target.numpy(), valid.numpy()), basis, prior, iters=(30, 80))
    d, _, _ = OF.residuals(target.numpy(), valid.numpy(), m.full_aa(), m.tr.detach(), u, o, basis, prior)
    assert float(np.sqrt(np.nanmean(d.numpy() ** 2))) < 0.01


def test_rotvec_is_exact_near_a_half_turn():
    import math
    from mdfm.body.rotations import axis_angle_to_matrix
    torch.manual_seed(1)
    v = torch.randn(4000, 3, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    ang = torch.cat([math.pi - 1e-6 * torch.rand(2000, 1, dtype=torch.float64),
                     math.pi * torch.rand(2000, 1, dtype=torch.float64)])
    R = axis_angle_to_matrix(v * ang)
    assert torch.allclose(axis_angle_to_matrix(OF.rotvec(R.float())), R, atol=2e-6)
    assert float(OF.rotvec(R).norm(dim=-1).max()) <= math.pi + 1e-9


def test_pelvis_marker_hips_ride_the_pelvis():
    assert [OF.segments()[i] for i in OF.HIPS] == [1, 2]
    assert [OF.segments(True)[i] for i in OF.HIPS] == [0, 0]
    assert set(OR.HIP_ON_PELVIS) == {"gait120", "cmu_mocap", "push_recovery_perturbations", "deepdance",
                                     "lencioni_gait_stairs"}

