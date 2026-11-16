"""Tests of the pose kernels (spec 2.3, task B4.8): the MoveAtlas canonical normalization, the
canonical joint map with per-source offsets, and the pose3d, pose2d and root likelihoods.

normalize_3d is checked against the MoveAtlas canonical record of the KIT take of
tests/amass_fixture.py (a train archive, read through mdfm.data.canonical.load without any flag).
"""
import math
import os

import numpy as np
import pytest
import torch

from amass_fixture import TAKE, assert_train_take, canonical_row
from mdfm import paths
from mdfm.body.rotations import Z_UP_TO_Y_UP, axis_angle_to_matrix, rot_y
from mdfm.kernels import pose as P
from mdfm.kernels.camera import (SMPL_TO_CANONICAL_15, Camera, camera_from_params, intrinsics, keypoint_nll,
                                 look_at, project, stack_cameras)
from pose_fixture import stick

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
D = torch.float64


def _windows(W, n):
    return torch.repeat_interleave(torch.arange(W), n)


def _t_const(nu, dim):
    return math.lgamma((nu + dim) / 2) - math.lgamma(nu / 2) - dim / 2 * math.log(nu * math.pi)


# ---------------------------------------------------------------- normalization
def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


def test_the_reader_refuses_cody_rows_without_the_evaluation_flag():
    from mdfm.data import canonical
    for row in ({"source": "cody_3", "modality": "pose", "subject_ns": "cody_3:p", "key": "k"},
                {"source": "cody_1", "modality": "pose", "subject_ns": "adult:P01", "key": "k"}):
        with pytest.raises(canonical.Refused):
            canonical.load(row)


def test_the_reader_refuses_a_cody_namespace_under_another_source_name():
    from mdfm.data import canonical
    assert "adult" in canonical.cody_namespaces()
    with pytest.raises(canonical.Refused):
        canonical.check({"source": "li_taati", "modality": "pose", "subject_ns": "adult:P01", "key": "k"})


def _ma_normalize(coords, dim):
    """Transcribed from MoveAtlas build_canonical.normalize (read, never imported)."""
    c = coords.astype(float).copy()
    valid = ~np.isnan(c).any(axis=-1)
    root_ok = np.isfinite(c[:, 0, :]).all(axis=-1)
    torso = np.linalg.norm(c[:, 0, :] - c[:, 1, :], axis=-1)
    good = torso[np.isfinite(torso) & (torso > 1e-6)]
    scale = float(np.median(good)) if good.size else 1.0
    c = c - c[:, 0:1, :]
    if dim == 2:
        c[:, :, 1] = -c[:, :, 1]
    c = c / scale
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0), scale, valid & root_ok[:, None]


def test_normalize_2d_equals_the_moveatlas_rule_bit_for_bit_on_edge_cases():
    rng = np.random.default_rng(1)
    for trial in range(100):
        T = int(rng.integers(1, 30))
        uv = rng.normal(500, 150, size=(T, 15, 2))
        uv[rng.random((T, 15)) < rng.random() * 0.5] = np.nan
        if trial % 3 == 0:
            uv[rng.integers(T), 0, 1] = np.nan             # half a pelvis
        if trial % 10 == 1:
            uv[:, 1] = uv[:, 0]                            # zero torso everywhere: scale 1
        if trial % 7 == 2:
            uv[rng.integers(T), 3, 0] = np.inf             # an infinite coordinate: detected, written 0
        c, s, v = P.normalize_2d(uv)
        ref, s_ref, v_ref = _ma_normalize(uv, 2)
        assert s == s_ref and np.array_equal(v, v_ref), trial
        assert np.array_equal(c[..., :2], ref.astype(np.float32)) and (c[..., 2] == 0).all(), trial
        x = np.concatenate([uv, rng.normal(size=(T, 15, 1))], -1)
        c3, s3, v3 = P.normalize_3d(x)
        ref3, s3_ref, v3_ref = _ma_normalize(x, 3)
        assert s3 == s3_ref and np.array_equal(v3, v3_ref) and np.array_equal(c3, ref3.astype(np.float32)), trial


def test_normalize_2d_reproduces_a_real_2d_record():
    """A li_taati record (2d_pixel, RELEASE, not CODY) read without any flag, taken back to image pixels
    (y down, pelvis anywhere, invalid joints NaN) and normalized again."""
    from mdfm.data import canonical
    idx = canonical.index()
    rows = idx[(idx["source"] == "li_taati") & (idx["modality"] == "pose")].sort_values(["subject_ns", "key"])
    if rows.empty or not os.path.exists(canonical.container_path(rows.iloc[0]["container"])):
        pytest.skip("li_taati canonical container not found")
    rec = canonical.load(rows.iloc[0][["source", "modality", "subject_ns", "key"]].to_dict())
    assert rec["modality"] == "2d_pixel" and not rec["lockbox"] and rec["tier"] == "RELEASE"
    ref = np.asarray(rec["coords"])
    vm = np.asarray(rec["valid_mask"], dtype=bool)
    uv = ref[..., :2].astype(np.float64) * rec["scale"]
    uv[..., 1] = -uv[..., 1]
    uv = uv + np.array([640.0, 360.0])
    uv[~vm] = np.nan
    c, s, v = P.normalize_2d(uv)
    assert np.abs(c - ref).max() < 1e-5 and abs(s - rec["scale"]) < 1e-6 * rec["scale"]
    assert np.array_equal(v, vm)
    assert np.median(ref[:, 2, 1]) > 0                     # the head sits above the pelvis (y up)


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_normalize_3d_reproduces_the_moveatlas_amass_canonical_record():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.data import canonical
    from mdfm.data.amass import load_sequence
    seq = load_sequence(*TAKE)
    body = SMPLHBody(seq["gender"], dtype=D)
    p = torch.as_tensor(seq["poses"], dtype=D)
    R = axis_angle_to_matrix(p[:, :66].reshape(-1, 22, 3))
    tr = torch.as_tensor(seq["trans"], dtype=D)
    b = torch.as_tensor(seq["betas"], dtype=D).expand(p.shape[0], -1)
    with torch.no_grad():
        J, Rg = body(R[:, 0], R[:, 1:], tr, b)
    M = Z_UP_TO_Y_UP.to(D)
    J, Rg = J @ M.T, M @ Rg                        # the world turned from z-up to y-up
    j15 = P.to_canonical15(J, Rg, torch.zeros(15, 3, dtype=D))
    assert torch.equal(j15, J[:, SMPL_TO_CANONICAL_15])
    coords, scale, valid = P.normalize_3d(j15)
    rec = canonical.load(canonical_row())
    assert not rec["lockbox"] and rec["tier"] == "RELEASE" and rec["modality"] == "3d_meter"
    ref = np.asarray(rec["coords"])
    assert coords.dtype == np.float32 and coords.shape == ref.shape
    err = float(np.abs(coords - ref).max())
    assert err < 1e-4, f"max abs difference {err}"
    assert abs(scale - rec["scale"]) < 1e-6
    assert np.array_equal(valid, np.asarray(rec["valid_mask"]))
    # the record un-normalized and moved anywhere normalizes back to itself
    x = ref.astype(np.float64) * rec["scale"] + np.array([0.3, 1.0, -2.0])
    c2, s2, _ = P.normalize_3d(x)
    assert np.abs(c2 - ref).max() < 1e-6 and abs(s2 - rec["scale"]) < 1e-9


def test_normalization_rules_nan_mask_and_image_flip():
    torch.manual_seed(0)
    x = np.random.default_rng(0).normal(size=(5, 15, 3))
    x[:, 1] = x[:, 0] + np.array([0.0, 0.5, 0.0])          # torso 0.5
    x[3, 1] = x[3, 0] + np.array([0.0, 0.9, 0.0])          # one long torso frame
    nan = x.copy()
    nan[1, 0] = np.nan                                     # pelvis lost: the frame is invalid
    nan[2, 7] = np.nan                                     # one wrist lost
    c, s, v = P.normalize_3d(nan)
    mask = ~np.isnan(nan).any(-1)
    c_m, s_m, v_m = P.normalize_3d(np.nan_to_num(nan, nan=5.0), valid=mask)
    assert np.array_equal(c, c_m) and s == s_m and np.array_equal(v, v_m)
    assert s == pytest.approx(0.5) and not v[1].any() and (c[1] == 0).all()
    assert not v[2, 7] and v[2].sum() == 14 and (c[2, 7] == 0).all()
    assert np.allclose(c[0], (x[0] - x[0, 0]) / 0.5, atol=1e-6)
    # 2D: image y down becomes y up, z is 0, the torso median is in pixels
    uv = x[..., :2] * 100 + 300
    c2, s2, v2 = P.normalize_2d(uv)
    assert c2.shape == (5, 15, 3) and (c2[..., 2] == 0).all() and s2 == pytest.approx(50.0) and v2.all()
    assert np.allclose(c2[..., 0], (uv[..., 0] - uv[:, :1, 0]) / s2, atol=1e-5)
    assert np.allclose(c2[..., 1], -(uv[..., 1] - uv[:, :1, 1]) / s2, atol=1e-5)
    # no usable torso: scale 1
    assert P.normalize_2d(np.full((3, 15, 2), np.nan))[1] == 1.0


def test_to_canonical15_offsets_turn_with_their_segment():
    torch.manual_seed(1)
    pos = torch.randn(4, 22, 3, dtype=D)
    R = torch.eye(3, dtype=D).expand(4, 22, 3, 3).clone()
    R[:, 0] = rot_y(torch.tensor(math.pi / 2, dtype=D))    # pelvis turned by 90 degrees
    R[:, 4] = rot_y(torch.tensor(-math.pi / 2, dtype=D))   # left knee segment (shank) turned
    off = torch.zeros(15, 3, dtype=D)
    off[9] = torch.tensor([0.0, 0.0, 0.1])                 # left hip 10 cm forward in the pelvis frame
    off[13] = torch.tensor([0.05, 0.0, 0.0])               # left ankle 5 cm lateral in the shank frame
    off[11] = torch.tensor([0.0, 0.02, 0.0])               # left knee 2 cm up in the thigh frame
    # the head (a skull landmark such as the nose) turns with SMPL joint 15, not with the neck segment
    R[:, 15] = rot_y(torch.tensor(math.pi / 2, dtype=D))    # head turned by 90 degrees on the neck
    off[2] = torch.tensor([0.0, 0.06, 0.11])               # nose: up and forward in the head frame
    j = P.to_canonical15(pos, R, off)
    base = pos[:, SMPL_TO_CANONICAL_15]
    assert torch.allclose(j[:, 9] - base[:, 9], torch.tensor([0.1, 0.0, 0.0], dtype=D).expand(4, 3), atol=1e-12)
    assert torch.allclose(j[:, 13] - base[:, 13], torch.tensor([0.0, 0.0, 0.05], dtype=D).expand(4, 3),
                          atol=1e-12)
    assert torch.allclose(j[:, 11] - base[:, 11], torch.tensor([0.0, 0.02, 0.0], dtype=D).expand(4, 3), atol=1e-12)
    assert torch.allclose(j[:, 2] - base[:, 2], torch.tensor([0.11, 0.06, 0.0], dtype=D).expand(4, 3), atol=1e-12)
    others = [i for i in range(15) if i not in (2, 9, 11, 13)]
    assert torch.equal(j[:, others], base[:, others])
    assert torch.equal(P.to_canonical15(pos), base)
    assert P.CANONICAL_PARENT == [0, 9, 15, 13, 14, 16, 17, 18, 19, 0, 0, 1, 2, 4, 5]


# ---------------------------------------------------------------- pose 3D
def _pose3d_case(W=3, n=12, seed=0, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(W * n, 15, 3, generator=g, dtype=D) * 0.3
    pred[:, :, 1] += 1.0
    w = _windows(W, n)
    yaw = torch.rand(W, generator=g, dtype=D) * 2 * math.pi - math.pi
    scale = 0.4 + 0.2 * torch.rand(W, generator=g, dtype=D)
    rel = pred - pred[:, :1]
    obs_m = rel @ rot_y(-yaw)[w].transpose(-1, -2) + noise * torch.randn(rel.shape, generator=g, dtype=D)
    obs_m[:, 0] = 0
    return pred, obs_m / scale[w][:, None, None], scale, yaw, w


def test_pose3d_zero_residual_at_truth_and_invariances():
    pred, obs, scale, yaw, w = _pose3d_case()
    valid = torch.ones(pred.shape[:2], dtype=torch.bool)
    r, th = P.pose3d_residual(pred, obs, scale, w, 3, valid=valid)
    assert r.abs().max() < 1e-12
    assert torch.allclose(th, yaw, atol=1e-12)
    log_sigma = torch.log(torch.tensor([0.02, 0.05, 0.03], dtype=D))
    nll = P.pose3d_nll(pred, obs, scale, 1.0, valid, log_sigma, w, 3)
    sig = torch.sqrt(torch.exp(2 * log_sigma) + 0.01 ** 2)
    # 14 terms per frame: the pelvis, zero on both sides, carries none
    assert torch.allclose(nll, 12 * 14 * (-_t_const(4.0, 3) + 3 * torch.log(sig)), atol=1e-9)
    no_root = valid.clone()
    no_root[:, 0] = False
    assert torch.equal(P.pose3d_nll(pred, obs, scale, 1.0, no_root, log_sigma, w, 3), nll)
    # translation of the prediction, a common yaw of prediction and observation, a yaw of either
    g = torch.Generator().manual_seed(5)
    moved = pred + torch.randn(pred.shape[0], 1, 3, generator=g, dtype=D)
    turn = rot_y(torch.tensor([0.7, -1.2, 2.5], dtype=D))[w]
    for p, o in ((moved, obs), (pred @ turn.transpose(-1, -2), obs @ turn.transpose(-1, -2)),
                 (pred @ turn.transpose(-1, -2), obs), (pred, obs @ turn.transpose(-1, -2))):
        assert torch.allclose(P.pose3d_nll(p, o, scale, 1.0, valid, log_sigma, w, 3), nll, atol=1e-9)
    # unit_to_si: millimetre records with a scale in mm
    assert torch.allclose(P.pose3d_nll(pred, obs, scale * 1000, 1e-3, valid, log_sigma, w, 3), nll, atol=1e-9)
    # a z-up source: obs recorded z-up is turned to y-up first
    obs_z = obs @ Z_UP_TO_Y_UP.to(D)
    assert torch.allclose(P.pose3d_nll(pred, obs_z, scale, 1.0, valid, log_sigma, w, 3, z_up=True), nll, atol=1e-9)
    per_window = torch.tensor([False, True, False])
    obs_mix = torch.where((w == 1)[:, None, None], obs_z, obs)
    assert torch.allclose(P.pose3d_nll(pred, obs_mix, scale, 1.0, valid, log_sigma, w, 3, z_up=per_window), nll,
                          atol=1e-9)
    assert P.pose3d_nll(pred, obs_z, scale, 1.0, valid, log_sigma, w, 3).min() > nll.max() + 1.0


def test_closed_form_yaw_equals_brute_force():
    pred, obs, scale, yaw, w = _pose3d_case(W=4, n=20, seed=3, noise=0.03)
    valid = torch.rand(pred.shape[:2], generator=torch.Generator().manual_seed(4)) > 0.2
    _, th = P.pose3d_residual(pred, obs, scale, w, 4, valid=valid)

    def cost(grid, p, o, m):
        # least-squares cost of each candidate yaw, by turning the observed points explicitly
        return (((torch.einsum("gij,nkj->gnki", rot_y(grid), o) - p) ** 2).sum(-1) * m).sum((1, 2))

    for k in range(4):
        sel = w == k
        p = (pred[sel] - pred[sel][:, :1])
        o = obs[sel] * scale[k]
        m = valid[sel].to(D)
        coarse = torch.arange(-180.0, 180.0, 1.0, dtype=D) * math.pi / 180
        b0 = float(coarse[cost(coarse, p, o, m).argmin()])
        fine = b0 + torch.arange(-1.5, 1.5, 0.01, dtype=D) * math.pi / 180
        best = float(fine[cost(fine, p, o, m).argmin()])
        diff = math.degrees(abs(math.remainder(float(th[k]) - best, 2 * math.pi)))
        assert diff < 0.1, f"window {k}: closed form and brute force differ by {diff} degrees"
        assert math.degrees(abs(math.remainder(float(th[k] - yaw[k]), 2 * math.pi))) < 5.0


def test_missing_joints_contribute_zero_and_gradients_stay_finite():
    pred, obs, scale, _, w = _pose3d_case(W=3, n=8, seed=6, noise=0.01)
    valid = torch.ones(pred.shape[:2], dtype=torch.bool)
    valid[::3, 4] = False
    valid[w == 2] = False                                  # a window without any valid joint
    base, th = P.pose3d_nll(pred, obs, scale, 1.0, valid, None, w, 3, return_yaw=True)
    junk = obs.clone()
    junk[~valid] = 50.0
    again, th2 = P.pose3d_nll(pred, junk, scale, 1.0, valid, None, w, 3, return_yaw=True)
    assert torch.equal(base, again) and torch.equal(th, th2)
    assert base[2] == 0 and th[2] == 0
    p = pred.clone().requires_grad_(True)
    ls = torch.zeros(3, dtype=D, requires_grad=True)
    P.pose3d_nll(p, junk, scale, 1.0, valid, ls, w, 3).sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(ls.grad).all() and p.grad.abs().sum() > 0
    assert p.grad[w == 2].abs().max() == 0
    # point weights scale each point's term
    wt = torch.full((pred.shape[0],), 2.0, dtype=D)
    assert torch.allclose(P.pose3d_nll(pred, obs, scale, 1.0, valid, None, w, 3, weight=wt), 2 * base)


def test_pose3d_with_source_offsets_and_precision_rules():
    g = torch.Generator().manual_seed(7)
    pos = torch.randn(10, 22, 3, generator=g, dtype=D) * 0.3
    R = axis_angle_to_matrix(torch.randn(10, 22, 3, generator=g, dtype=D) * 0.5)
    off = torch.randn(15, 3, generator=g, dtype=D) * 0.03
    off[0] = 0
    true = P.to_canonical15(pos, R, off)
    obs = (true - true[:, :1]) / 0.5
    w = torch.zeros(10, dtype=torch.long)
    valid = torch.ones(10, 15, dtype=torch.bool)
    r, _ = P.pose3d_residual(P.to_canonical15(pos, R, off), obs, 0.5, w, 1, valid=valid)
    assert r.abs().max() < 1e-12
    r0, _ = P.pose3d_residual(P.to_canonical15(pos), obs, 0.5, w, 1, valid=valid)
    assert r0.norm(dim=-1).max() > 0.01
    with pytest.raises(TypeError):
        P.pose3d_nll(true.half(), obs.half(), 0.5, 1.0, valid, None, w, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(RuntimeError):
        P.pose3d_nll(true, obs, 0.5, 1.0, valid, None, w, 1)
    with pytest.raises(ValueError):
        P.to_canonical15(pos, None, off)


# ---------------------------------------------------------------- pose 2D
def test_pose2d_zero_residual_at_truth_masks_and_per_window_cameras():
    pos = stick(20, 30.0, 1.0, speed=1.0)
    cams = []
    for eye in ((0.5, 1.5, 4.0), (4.0, 1.0, 0.5)):
        Rc, tc = look_at(eye, (0.0, 0.95, 0.3))
        cams.append(Camera(intrinsics(1000.0, 1000.0, 640.0, 360.0), Rc, tc, 0.0, 0.0, 1280, 720, 30.0))
    obs, scales, valids = [], [], []
    for c in cams:
        uv, _ = project(pos, c.K, c.R, c.t)
        co, s, v = P.normalize_2d(uv)
        obs.append(torch.as_tensor(co, dtype=D))
        scales.append(s)
        valids.append(torch.as_tensor(v))
    obs, valid, w = torch.cat(obs), torch.cat(valids), _windows(2, 20)
    scale = torch.tensor(scales, dtype=D)
    pred = torch.cat([pos, pos])
    cam = stack_cameras(cams)
    r = P.pose2d_residual(pred, cam, obs, scale, w)
    assert r.abs().max() < 1e-3                           # float32 coords of a few hundred pixels
    nll = P.pose2d_nll(pred, cam, obs, scale, valid, w)
    zero = 14 * 20 * -(_t_const(4.0, 2) - 2 * math.log(3.0))    # 14 terms per frame, none for the pelvis
    assert torch.allclose(nll, torch.full((2,), zero, dtype=D), rtol=1e-9)
    # per-window cameras equal one call per window with a single camera
    for k in range(2):
        sel = w == k
        one = P.pose2d_nll(pred[sel], cams[k], obs[sel], scales[k], valid[sel], torch.zeros(20, dtype=torch.long))
        assert torch.allclose(one[0], nll[k], rtol=1e-10)
    # masked joints contribute zero; a per-window sigma0
    off = pred.clone()
    off[:, 7] += 0.3
    m = valid.clone()
    m[:, 7] = False
    assert torch.allclose(P.pose2d_nll(off, cam, obs, scale, m, w), P.pose2d_nll(pred, cam, obs, scale, m, w))
    assert (P.pose2d_nll(off, cam, obs, scale, valid, w) > nll + 100).all()
    s0 = torch.tensor([3.0, 6.0], dtype=D)
    alt = P.pose2d_nll(pred, cam, obs, scale, valid, w, sigma0=s0)
    assert torch.allclose(alt[1] - nll[1], torch.tensor(14 * 20 * 2 * math.log(2.0), dtype=D))


def test_pose2d_gradients_reach_the_body_and_the_camera_nuisance():
    pos = stick(10, 30.0, 1.0)
    target = pos[:, 0].mean(0)
    rot6d = torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], dtype=D, requires_grad=True)
    ld = torch.tensor([math.log(4.0)], dtype=D, requires_grad=True)
    lf = torch.tensor([math.log(1000.0)], dtype=D, requires_grad=True)
    cam = camera_from_params(rot6d, ld, lf, target[None])
    uv, depth = project(target[None, None], cam.K[0], cam.R[0], cam.t[0])
    assert torch.allclose(uv[0, 0], torch.tensor([640.0, 360.0], dtype=D)) and torch.allclose(depth, ld.exp())
    w = torch.zeros(10, dtype=torch.long)
    uv, _ = project(pos, cam.K[0].detach(), cam.R[0].detach(), cam.t[0].detach())
    co, s, v = P.normalize_2d(uv.detach())
    p = (pos + 0.01).requires_grad_(True)
    P.pose2d_nll(p, cam, torch.as_tensor(co, dtype=D), torch.tensor([s], dtype=D), torch.as_tensor(v), w).sum() \
        .backward()
    for x in (p, rot6d, ld, lf):
        assert x.grad is not None and torch.isfinite(x.grad).all()
    assert p.grad.abs().sum() > 0 and lf.grad.abs().sum() > 0


def test_camera_from_params_equals_the_camera_nuisance_and_scores_truth_at_zero_residual():
    from mdfm.kernels.nuisance import CameraNuisance
    g = torch.Generator().manual_seed(12)
    W, T = 3, 15
    pos = stick(T, 30.0, 1.0, speed=0.5)
    target = pos[:, 0].mean(0).expand(W, 3)
    nuis = CameraNuisance(torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], dtype=D).expand(W, 6)
                          + 0.2 * torch.randn(W, 6, generator=g, dtype=D),
                          torch.log(3.0 + 3.0 * torch.rand(W, generator=g, dtype=D)),
                          torch.log(800.0 + 300.0 * torch.rand(W, generator=g, dtype=D)),
                          torch.log(torch.tensor([2.0, 3.0, 5.0], dtype=D)))
    cam = camera_from_params(nuis.rot6d, nuis.log_distance, nuis.log_focal, target)
    R, t = nuis.extrinsics(target)
    assert torch.allclose(cam.R, R, atol=1e-12) and torch.allclose(cam.t, t, atol=1e-12)
    assert torch.allclose(cam.K, nuis.intrinsics(), atol=1e-12)
    # obs rendered through the nuisance camera, scored with its sigma_px: the zero-residual constant
    w = _windows(W, T)
    obs, scales, valids = [], [], []
    for k in range(W):
        uv, _ = project(pos, cam.K[k], cam.R[k], cam.t[k])
        co, s, v = P.normalize_2d(uv)
        obs.append(torch.as_tensor(co, dtype=D))
        scales.append(s)
        valids.append(torch.as_tensor(v))
    valid = torch.cat(valids)
    assert valid.all()
    nll = P.pose2d_nll(pos.repeat(W, 1, 1), cam, torch.cat(obs), torch.tensor(scales, dtype=D), valid, w,
                       sigma0=nuis.sigma_px())
    zero = -T * 14 * (_t_const(4.0, 2) - 2 * torch.log(nuis.sigma_px()))
    assert torch.allclose(nll, zero, rtol=1e-9)


def test_gradcheck_of_the_pose_likelihoods():
    """float64 gradcheck: pose3d through the closed-form yaw and the learned sigma, pose2d through the
    camera nuisance parameters and sigma_px, root through its fitted yaw."""
    pred, obs, scale, _, w = _pose3d_case(W=2, n=3, seed=13, noise=0.02)
    valid = torch.rand(pred.shape[:2], generator=torch.Generator().manual_seed(14)) > 0.2
    f3 = lambda p, ls: P.pose3d_nll(p, obs, scale, 1.0, valid, ls, w, 2)  # noqa: E731
    assert torch.autograd.gradcheck(f3, (pred.clone().requires_grad_(True),
                                         torch.tensor([-3.0, -3.5], dtype=D, requires_grad=True)), fast_mode=True)
    pos = stick(4, 30.0, 1.0)
    target = pos[:, 0].mean(0)[None]
    uv, _ = project(pos, *(x.detach()[0] for x in camera_from_params(
        torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], dtype=D), torch.tensor([math.log(4.0)], dtype=D),
        torch.tensor([math.log(1000.0)], dtype=D), target)[:3]))
    co, s, v = P.normalize_2d(uv + 2.0 * torch.randn(uv.shape, generator=torch.Generator().manual_seed(15),
                                                     dtype=D))
    w2 = torch.zeros(4, dtype=torch.long)

    def f2(p, r6, ld, lf, lsp):
        cam = camera_from_params(r6, ld, lf, target)
        return P.pose2d_nll(p, cam, torch.as_tensor(co, dtype=D), torch.tensor([s], dtype=D), torch.as_tensor(v),
                            w2, 1, sigma0=torch.exp(lsp))
    args = (pos.clone().requires_grad_(True),
            torch.tensor([[-1.0, 0.05, 0.0, 0.02, -1.0, 0.0]], dtype=D, requires_grad=True),
            torch.tensor([math.log(4.0)], dtype=D, requires_grad=True),
            torch.tensor([math.log(1000.0)], dtype=D, requires_grad=True),
            torch.tensor([math.log(3.0)], dtype=D, requires_grad=True))
    assert torch.autograd.gradcheck(f2, args, fast_mode=True)
    g = torch.Generator().manual_seed(16)
    track = torch.randn(8, 3, generator=g, dtype=D) * 0.3
    obs_r = track @ rot_y(torch.tensor(0.4, dtype=D)).T + 0.01 * torch.randn(8, 3, generator=g, dtype=D)
    fr = lambda p: P.root_nll(p, obs_r, torch.ones(8, dtype=torch.bool), torch.zeros(8, dtype=torch.long), 1)  # noqa
    assert torch.autograd.gradcheck(fr, (track.clone().requires_grad_(True),), fast_mode=True)


def test_keypoint_nll_reduce_and_batched_projection():
    g = torch.Generator().manual_seed(2)
    a, b = torch.rand(4, 15, 2, generator=g, dtype=D) * 500, torch.rand(4, 15, 2, generator=g, dtype=D) * 500
    valid = torch.rand(4, 15, generator=g) > 0.3
    per = keypoint_nll(a, b, None, valid, sigma0=3.0, reduce="none")
    assert per.shape == (4, 15) and (per[~valid] == 0).all()
    assert torch.allclose(per.sum(), keypoint_nll(a, b, torch.ones(4, 15, dtype=D), valid.to(D), sigma0=3.0))
    with pytest.raises(ValueError):
        keypoint_nll(a, b, None, valid, reduce="mean")
    cams = [Camera(intrinsics(900.0 + 100 * k, 950.0, 600.0, 350.0), *look_at((k, 1.2, 4.0), (0.0, 1.0, 0.0)),
                   0.05 * k, -0.01 * k) for k in range(3)]
    pts = torch.randn(3, 15, 3, generator=g, dtype=D) * 0.3 + torch.tensor([0.0, 1.0, 0.0], dtype=D)
    batch = stack_cameras(cams)
    uv, z = project(pts, batch.K, batch.R, batch.t, batch.k1, batch.k2)
    for k, c in enumerate(cams):
        uk, zk = project(pts[k], c.K, c.R, c.t, c.k1, c.k2)
        assert torch.allclose(uv[k], uk, atol=1e-10) and torch.allclose(z[k], zk, atol=1e-12)


# ---------------------------------------------------------------- root
def test_root_nll_window_mean_yaw_and_masks():
    g = torch.Generator().manual_seed(9)
    W, n = 2, 30
    w = _windows(W, n)
    t = torch.arange(n, dtype=D) / 30
    track = torch.stack([0.05 * torch.sin(6 * t), 0.95 + 0.02 * torch.cos(12 * t), 1.2 * t], -1)
    pred = torch.cat([track, track]) + torch.tensor([[0.3, 0.0, -1.0]], dtype=D)
    yaw = torch.tensor([0.4, -2.2], dtype=D)
    obs = torch.einsum("nij,nj->ni", rot_y(-yaw)[w], torch.cat([track, track]))
    obs = obs + torch.randn(W, 3, generator=g, dtype=D)[w] * 5    # any lab origin per window
    valid = torch.ones(W * n, dtype=torch.bool)
    nll, th = P.root_nll(pred, obs, valid, w, W, return_yaw=True)
    zero = n * (-_t_const(4.0, 3) + 3 * math.log(0.01))
    assert torch.allclose(nll, torch.full((W,), zero, dtype=D), atol=1e-9)
    assert torch.allclose(th, yaw, atol=1e-9)
    assert torch.allclose(P.root_nll(pred, obs, valid, w, W, yaw=yaw), nll, atol=1e-9)
    assert (P.root_nll(pred, obs, valid, w, W, yaw=yaw + 0.3) > nll + 1).all()
    # a masked sample contributes nothing, even far away
    m = valid.clone()
    m[5] = False
    junk = obs.clone()
    junk[5] += 100.0
    assert torch.allclose(P.root_nll(pred, junk, m, w, W), P.root_nll(pred, obs, m, w, W), atol=1e-9)


# ---------------------------------------------------------------- independent references, devices
def test_likelihoods_equal_independent_student_t_references():
    """pose3d, pose2d and root NLL at a given yaw against scipy.stats.multivariate_t and a numpy OpenCV
    projection with radial distortion: per-window cameras, confidences and sigma0, a z-up window, a
    millimetre record (unit_to_si 1e-3), point weights and masks."""
    from scipy.stats import multivariate_t
    rng = np.random.default_rng(42)
    W, n, nu = 3, 6, 4.0
    N = W * n
    w = np.repeat(np.arange(W), n)
    wt = torch.as_tensor(w)
    valid = rng.random((N, 15)) > 0.2
    weight = rng.uniform(0.5, 2.0, N)
    log_sigma = np.log(rng.uniform(0.005, 0.05, W))
    sig = np.sqrt(np.exp(2 * log_sigma) + 0.01 ** 2)
    yaw = rng.uniform(-math.pi, math.pi, W)
    roty = rot_y(torch.as_tensor(yaw)).numpy()

    def t_logpdf(x, s, dim):
        return multivariate_t(loc=np.zeros(dim), shape=s * s * np.eye(dim), df=nu).logpdf(x)

    def terms(s):
        m = valid[s] * weight[s][:, None]
        m[:, 0] = 0                                        # no pelvis term
        return m

    # pose3d: obs in mm, window 1 stored z-up
    pred = rng.normal(0, 0.3, (N, 15, 3)) + rng.normal(0, 2, (N, 1, 3))
    obs = rng.normal(0, 1, (N, 15, 3))
    obs[:, 0] = 0
    scale = rng.uniform(400, 600, W)
    z_up = np.array([False, True, False])
    ref = np.zeros(W)
    for k in range(W):
        s = w == k
        o = obs[s] * scale[k] * 1e-3
        if z_up[k]:
            o = o @ Z_UP_TO_Y_UP.double().numpy().T
        r = (pred[s] - pred[s][:, :1]) - o @ roty[k].T
        ref[k] = -(t_logpdf(r.reshape(-1, 3), sig[k], 3).reshape(-1, 15) * terms(s)).sum()
    got = P.pose3d_nll(torch.as_tensor(pred), torch.as_tensor(obs), torch.as_tensor(scale), 1e-3,
                       torch.as_tensor(valid), torch.as_tensor(log_sigma), wt, W, z_up=torch.as_tensor(z_up),
                       weight=torch.as_tensor(weight), yaw=torch.as_tensor(yaw))
    assert np.allclose(got.numpy(), ref, rtol=1e-10, atol=0)
    # pose2d: per-window cameras with distortion, confidence, per-window sigma0
    cams = [Camera(intrinsics(rng.uniform(700, 1100), rng.uniform(700, 1100), 640.0, 360.0),
                   *look_at((rng.uniform(-3, 3), rng.uniform(0.8, 1.8), rng.uniform(3, 5)), (0.0, 0.95, 0.0)),
                   float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.02, 0.02))) for _ in range(W)]
    body = rng.normal(0, 0.3, (N, 15, 3)) + np.array([0.0, 1.0, 0.0])
    obs2 = np.zeros((N, 15, 3))
    obs2[..., :2] = rng.normal(0, 1, (N, 15, 2))
    obs2[:, 0] = 0
    scale_px, sigma0 = rng.uniform(80, 200, W), rng.uniform(2, 6, W)
    conf = rng.uniform(0.2, 1.0, (N, 15))
    ref2 = np.zeros(W)
    for k, c in enumerate(cams):
        s = w == k
        pc = body[s] @ c.R.numpy().T + c.t.numpy()
        xy = pc[..., :2] / pc[..., 2:]
        r2 = (xy ** 2).sum(-1, keepdims=True)
        xy = xy * (1 + c.k1 * r2 + c.k2 * r2 ** 2)
        uv = xy * np.diag(c.K.numpy())[:2] + c.K.numpy()[:2, 2]
        d = (uv - uv[:, :1]) * np.array([1.0, -1.0]) - obs2[s, :, :2] * scale_px[k]
        sd = sigma0[k] / conf[s]                           # scale sigma0 / confidence px
        lp = t_logpdf((d / sd[..., None]).reshape(-1, 2), 1.0, 2).reshape(-1, 15) - 2 * np.log(sd)
        ref2[k] = -(lp * terms(s)).sum()
    got2 = P.pose2d_nll(torch.as_tensor(body), stack_cameras(cams), torch.as_tensor(obs2), torch.as_tensor(scale_px),
                        torch.as_tensor(valid), wt, W, conf=torch.as_tensor(conf), sigma0=torch.as_tensor(sigma0),
                        weight=torch.as_tensor(weight))
    assert np.allclose(got2.numpy(), ref2, rtol=1e-10, atol=0)
    # root: both tracks relative to their weighted window mean, the observation turned by the yaw
    pel = rng.normal(0, 0.5, (N, 3))
    robs = rng.normal(0, 0.5, (N, 3)) + rng.normal(0, 5, (W, 3))[w]
    rv = rng.random(N) > 0.2
    ref3 = np.zeros(W)
    for k in range(W):
        s = w == k
        m = rv[s] * weight[s]
        p = pel[s] - (pel[s] * m[:, None]).sum(0) / m.sum()
        o = (robs[s] - (robs[s] * m[:, None]).sum(0) / m.sum()) @ roty[k].T
        ref3[k] = -(t_logpdf(p - o, sig[k], 3) * m).sum()
    got3 = P.root_nll(torch.as_tensor(pel), torch.as_tensor(robs), torch.as_tensor(rv), wt, W,
                      log_sigma=torch.as_tensor(log_sigma), yaw=torch.as_tensor(yaw), weight=torch.as_tensor(weight))
    assert np.allclose(got3.numpy(), ref3, rtol=1e-10, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_pose_likelihoods_in_float32_on_the_gpu_equal_float64_on_the_cpu():
    """Values, yaws and gradients; per-window inputs (scale, log_sigma, z_up) and the source offsets stay
    on the CPU in float64, as a caller may hold them."""
    g = torch.Generator().manual_seed(0)
    W, n = 3, 40
    w = _windows(W, n)
    pos = torch.randn(W * n, 22, 3, generator=g, dtype=D) * 0.3 + torch.tensor([0.0, 1.0, 0.0], dtype=D)
    Rg = axis_angle_to_matrix(torch.randn(W * n, 22, 3, generator=g, dtype=D) * 0.4)
    off = torch.randn(15, 3, generator=g, dtype=D) * 0.02
    obs3 = torch.randn(W * n, 15, 3, generator=g, dtype=D)
    obs2 = torch.cat([torch.randn(W * n, 15, 2, generator=g, dtype=D), torch.zeros(W * n, 15, 1, dtype=D)], -1)
    obs3[:, 0] = obs2[:, 0] = 0
    valid = torch.rand(W * n, 15, generator=g) > 0.2
    root_obs = torch.randn(W * n, 3, generator=g, dtype=D)
    rot6d = torch.tensor([[-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]], dtype=D) + 0.1 * torch.randn(W, 6, generator=g, dtype=D)
    scale = torch.tensor([0.5, 0.52, 0.48], dtype=D)
    ls = torch.log(torch.tensor([0.02, 0.03, 0.05], dtype=D))
    z_up = torch.tensor([False, True, False])

    def run(device, dtype):
        x = pos.to(device, dtype).requires_grad_(True)
        r6 = rot6d.to(device, dtype).requires_grad_(True)
        cast = lambda t: t.to(device, dtype)  # noqa: E731
        j15 = P.to_canonical15(x, cast(Rg), off)
        n3, y3 = P.pose3d_nll(j15, cast(obs3), scale, 1.0, valid.to(device), ls, w.to(device), W, z_up=z_up,
                              return_yaw=True)
        cam = camera_from_params(r6, cast(torch.log(torch.tensor([3.0, 4.0, 5.0], dtype=D))),
                                 cast(torch.log(torch.tensor([900.0, 1000.0, 1100.0], dtype=D))),
                                 cast(torch.tensor([[0.0, 1.0, 0.0]], dtype=D).expand(W, 3)))
        n2 = P.pose2d_nll(j15, cam, cast(obs2), torch.tensor([100.0, 150.0, 120.0], dtype=D), valid.to(device),
                          w.to(device), W)
        nr, yr = P.root_nll(j15[:, 0], cast(root_obs), valid[:, 0].to(device), w.to(device), W, return_yaw=True)
        (n3.sum() + n2.sum() + nr.sum()).backward()
        return [t.detach().cpu().double() for t in (n3, y3, n2, nr, yr, x.grad, r6.grad)]

    for a, b in zip(run("cuda", torch.float32), run("cpu", D)):
        assert torch.isfinite(a).all()
        assert float((a - b).abs().max()) <= 1e-4 * float(b.abs().max())
