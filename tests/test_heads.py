"""Tests of the nuisance heads (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15): at
initialization the heads return the prior centres (mount at the family's R0, site-default placement, zero
biases, unit gains, the centre camera of the generic family, whose target lands on the principal point);
the bounds hold on extreme inputs; no head reads a body-typed position (the heads read the nuisance-token
positions only, and rewriting every body position of the encoder output leaves the nuisance unchanged).
"""
import math
from dataclasses import replace

import numpy as np
import torch

from mdfm.factory.camera_render import FAMILIES
from mdfm.kernels.camera import project
from mdfm.kernels.nuisance import GAIN_LOG_MAX, OFFSET_MAX_M, MountPriors
from mdfm.model import heads as H
from mdfm.model import masks
from mdfm.model import mfm as M

import model_fixture as FX


def test_heads_return_the_prior_centres_at_initialization():
    heads = H.NuisanceHeads(d=32, hidden=16).requires_grad_(False)
    pr = MountPriors.load()
    W = 3
    h = torch.randn(W, 14, 32)
    fam = torch.tensor([0, 1, 1])
    n = heads.imu(h, fam)
    assert float(n.mount_res6d.abs().max()) == 0.0 and float(n.offset_res.abs().max()) == 0.0
    assert float(n.bias_acc.abs().max()) == 0.0 and float(n.bias_gyro.abs().max()) == 0.0
    assert torch.equal(n.gain(), torch.ones(W, 14, 2))
    assert torch.allclose(n.log_scale, torch.tensor(H.LOG_SCALE0).expand(W, 14, 2))
    assert torch.allclose(n.mount(pr.rotation(fam).float()), pr.rotation(fam).float(), atol=1e-6)
    cam = heads.camera(torch.randn(W, 32))
    fam_c = FAMILIES["generic"]
    assert torch.allclose(torch.exp(cam.log_distance), torch.full((W,), sum(fam_c.distance) / 2))
    f = (fam_c.width / 2) / math.tan(math.radians(sum(fam_c.hfov_deg) / 2) / 2)
    assert torch.allclose(torch.exp(cam.log_focal), torch.full((W,), f))
    assert torch.allclose(cam.sigma_px(), torch.full((W,), 3.0))
    target = torch.tensor([[0.0, 0.9, 0.0], [1.0, 1.0, 2.0], [0.0, 0.8, -1.0]])
    R, t = cam.extrinsics(target)
    uv, depth = project(target[:, None], cam.intrinsics(), R, t)
    assert torch.allclose(uv[:, 0], torch.tensor([640.0, 360.0]).expand(W, 2), atol=1e-3)
    assert torch.allclose(depth[:, 0], torch.exp(cam.log_distance), atol=1e-5)
    centre = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
    assert bool((centre[:, 2] > target[:, 2]).all())          # the camera stands on +z of the target
    assert torch.allclose(heads.pose3d(torch.randn(W, 32)), torch.full((W,), H.POSE3D_LOG_SIGMA0))


def test_bounds_hold_on_extreme_inputs():
    heads = H.NuisanceHeads(d=32, hidden=16).requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for head in (heads.imu_head, heads.camera_head, heads.pose3d_head):
            head.net[-1].weight.copy_(1e3 * torch.randn(head.net[-1].weight.shape, generator=g))
    h = 1e4 * torch.randn(4, 14, 32, generator=g)
    fam = torch.tensor([0, 1, 1, 0])
    n = heads.imu(h, fam)
    assert float(n.mount_res6d[..., :3].norm(dim=-1).max()) < H.MOUNT_RES_MAX + 1e-6
    assert float(n.mount_res6d[..., 3:].norm(dim=-1).max()) < H.MOUNT_RES_MAX + 1e-6
    R = n.residual_rotation()
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand_as(R), atol=1e-5)
    assert float(n.offset().norm(dim=-1).max()) <= OFFSET_MAX_M + 1e-7
    assert float(n.gain().log().abs().max()) <= GAIN_LOG_MAX + 1e-7
    sd_a = heads.mp_bias_acc_sd[fam[:, None], torch.arange(14)]
    sd_g = heads.mp_bias_gyro_sd[fam[:, None], torch.arange(14)]
    assert bool((n.bias_acc.abs() <= H.BIAS_MAX_SD * sd_a + 1e-6).all())
    assert bool((n.bias_gyro.abs() <= H.BIAS_MAX_SD * sd_g + 1e-5).all())
    ls0 = torch.tensor(H.LOG_SCALE0)
    assert bool(((n.log_scale - ls0).abs() <= H.LOG_SCALE_HALF + 1e-6).all())
    cam = heads.camera(1e4 * torch.randn(4, 32, generator=g))
    assert bool(((cam.log_distance - H.CAM_LOG_DISTANCE0).abs() <= H.CAM_HALF[0] + 1e-6).all())
    assert bool(((cam.log_focal - H.CAM_LOG_FOCAL0).abs() <= H.CAM_HALF[1] + 1e-6).all())
    assert bool(((cam.log_sigma_px - H.CAM_LOG_SIGMA0).abs() <= H.CAM_HALF[2] + 1e-6).all())
    Rc = cam.rotation()
    assert torch.allclose(Rc @ Rc.transpose(-1, -2), torch.eye(3).expand_as(Rc), atol=1e-5)
    ls = heads.pose3d(1e4 * torch.randn(4, 32, generator=g))
    assert bool(((ls - H.POSE3D_LOG_SIGMA0).abs() <= H.POSE3D_HALF + 1e-6).all())
    for x in (n.mount_res6d, n.bias_acc, n.bias_gyro, n.log_scale, cam.rot6d, ls):
        assert torch.isfinite(x).all()


def test_no_head_reads_a_body_position():
    model = M.MFM(replace(M.TINY, skeleton=False)).eval().requires_grad_(False)
    with torch.no_grad():
        for head in (model.heads.imu_head, model.heads.camera_head, model.heads.pose3d_head):
            torch.nn.init.normal_(head.net[-1].weight, std=0.5)
    wins = [FX.make_window(2.0, seed=0), FX.make_window(1.5, seed=3, person="p1", sex=0), FX.make_pose_window(1.0)]
    batch = M.batch_from_windows(wins, gen=0)
    tb = batch.tokens
    pos = model.nuisance_positions(batch)
    assert bool((tb.ttype.reshape(-1)[pos] == masks.NUISANCE).all())
    assert int(pos.numel()) == int((tb.ttype == masks.NUISANCE).sum())
    R, L = tb.t.shape
    g = torch.Generator().manual_seed(1)
    h = torch.randn(R, L, 64, generator=g)
    body = (tb.ttype == masks.BODY)[..., None]
    h2 = torch.where(body, 10 * torch.randn(R, L, 64, generator=g), h)
    a, b = model.nuisance_from_hidden(h, batch), model.nuisance_from_hidden(h2, batch)
    ta, tb_ = a.tensors(), b.tensors()
    assert set(ta) == set(tb_)
    for k in ta:
        assert torch.equal(ta[k], tb_[k]), k
    # and the nuisance does depend on its own tokens
    nz = ~body
    h3 = torch.where(nz, h + 1.0, h)
    c = model.nuisance_from_hidden(h3, batch).tensors()
    assert float((c["imu.mount_res6d"] - ta["imu.mount_res6d"]).abs().max()) > 0


def test_the_family_order_of_the_batch_is_the_heads_order():
    heads = H.NuisanceHeads(d=16, hidden=8)
    assert heads.families == M.mount_families() == tuple(MountPriors.load().families)
    assert M.mount_family("xsens13", heads.families) == heads.families.index("xsens13")
    assert M.mount_family("nonan3", heads.families) == heads.families.index("generic")
    assert M.mount_family(None, heads.families) == heads.families.index("generic")
    pr = heads.priors()
    assert np.allclose(pr.R0.numpy(), MountPriors.load().R0.float().numpy())
