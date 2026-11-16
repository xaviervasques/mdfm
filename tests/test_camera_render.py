"""Tests of the camera sampler, the tracker-failure process and the synthetic pose renders (spec 2.4,
task B4.8).

The real body is the KIT take of tests/amass_fixture.py (a train archive); the other tests use the
stick figure of tests/pose_fixture.py.
"""
import math
import os

import numpy as np
import pytest
import torch

from amass_fixture import TAKE, assert_train_take
from mdfm import paths
from mdfm.body.rotations import axis_angle_to_matrix
from mdfm.factory import camera_render as CR
from mdfm.kernels import pose as P
from mdfm.kernels.camera import Camera, intrinsics, look_at, project, stack_cameras
from pose_fixture import MOVING, stick

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
D = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _zero_nll(n_joints, sigma0=3.0, nu=4.0):
    const = math.lgamma((nu + 2) / 2) - math.lgamma(nu / 2) - math.log(nu * math.pi)
    return n_joints * -(const - 2 * math.log(sigma0))


def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


def test_sample_camera_ranges_target_and_determinism():
    target = torch.tensor([0.4, 0.95, 1.5], dtype=D)
    g = _gen(0)
    rates = set()
    for _ in range(200):
        cam = CR.sample_camera(g, target)
        eye = -cam.R.T @ cam.t
        horiz = float(torch.hypot(eye[0] - target[0], eye[2] - target[2]))
        assert 3.0 <= horiz <= 6.0 and 0.8 <= float(eye[1]) <= 1.8
        uv, z = project(target, cam.K, cam.R, cam.t)
        assert torch.allclose(uv, torch.tensor([640.0, 360.0], dtype=D), atol=1e-9) and z > 0
        hfov = math.degrees(2 * math.atan(640.0 / float(cam.K[0, 0])))
        assert 60.0 - 1e-9 <= hfov <= 80.0 + 1e-9 and cam.K[0, 0] == cam.K[1, 1]
        assert (cam.width, cam.height) == (1280, 720) and torch.allclose(cam.R @ cam.R.T, torch.eye(3, dtype=D))
        rates.add(cam.fps)
    assert rates == {25.0, 30.0, 50.0, 60.0}
    a, b = CR.sample_camera(_gen(3), target), CR.sample_camera(_gen(3), target)
    assert all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])) and a.fps == b.fps
    assert CR.sample_camera(_gen(3), target, fps=30.0).fps == 30.0
    # a target from the model (float32, with gradient, on the GPU when there is one) gives the same camera
    t32 = target.float().requires_grad_(True)
    targets = [t32] + ([t32.detach().cuda()] if torch.cuda.is_available() else [])
    for tg in targets:
        c = CR.sample_camera(_gen(3), tg)
        assert c.R.device.type == "cpu" and torch.allclose(c.t, a.t, atol=1e-6) and torch.equal(c.K, a.K)
    with pytest.raises(ValueError):
        CR.sample_camera(_gen(0), target, family="tripod_cody")


def _score_clean(pos15, seeds):
    """Clean renders of pos15 under sampled cameras, scored by pose2d_nll as one batch of windows."""
    target = pos15[:, 0].mean(0)
    cams, obs, scales, valids = [], [], [], []
    for s in seeds:
        cam = CR.sample_camera(_gen(s), target, fps=30.0)
        rec, truth = CR.render_pose2d(pos15, cam, _gen(s), CR.TrackerProcess.clean())
        assert rec["dim"] == 2 and rec["modality"] == "2d_pixel" and rec["coords"].dtype == np.float32
        assert rec["coords"].shape == (pos15.shape[0], 15, 3) and (rec["coords"][..., 2] == 0).all()
        assert list(rec["coord_mask"]) == [True, True, False] and rec["fps"] == 30.0
        assert np.array_equal(rec["valid_mask"], truth["in_image"] & truth["in_image"][:, :1])
        cams.append(cam)
        obs.append(torch.as_tensor(rec["coords"], dtype=D))
        scales.append(rec["scale"])
        valids.append(torch.as_tensor(rec["valid_mask"]))
    T, W = pos15.shape[0], len(seeds)
    w = torch.repeat_interleave(torch.arange(W), T)
    pred, obs, valid = pos15.repeat(W, 1, 1), torch.cat(obs), torch.cat(valids)
    cam, scale = stack_cameras(cams), torch.tensor(scales, dtype=D)
    r = P.pose2d_residual(pred, cam, obs, scale, w)
    nll = P.pose2d_nll(pred, cam, obs, scale, valid, w)
    n_valid = valid.reshape(W, T, 15)[..., 1:].sum((1, 2)).to(D)     # likelihood terms: no pelvis term
    return r, valid, nll, n_valid


def test_noise_free_render_scored_by_pose2d_nll_gives_zero_residual():
    pos = stick(60, 30.0, 1.0, speed=1.2)                  # 2 s walking, limbs swinging
    r, valid, nll, n_valid = _score_clean(pos, range(6))
    assert valid.float().mean() > 0.9
    assert (r[valid].abs().max()) < 1e-3                   # px, float32 coords
    assert torch.allclose(nll, _zero_nll(1) * n_valid, rtol=1e-9)


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_noise_free_render_of_a_real_body_gives_zero_residual():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    seq = load_sequence(*TAKE)
    body = SMPLHBody(seq["gender"], dtype=D)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=body)
    traj = BodyTrajectory(body, r6, tr, betas)
    t = 1.0 + torch.arange(90, dtype=D) / 30.0             # 3 s at 30 fps
    with torch.no_grad():
        pos, _ = traj.pose_at(t)
    pos15 = P.to_canonical15(pos)
    pos15 = pos15 - torch.tensor([pos15[:, 0, 0].mean(), 0.0, pos15[:, 0, 2].mean()], dtype=D)
    pos15[..., 1] -= pos15[..., 1].min()                  # feet on the floor, window frame
    r, valid, nll, n_valid = _score_clean(pos15, range(4))
    assert valid.float().mean() > 0.9 and r[valid].abs().max() < 1e-3
    assert torch.allclose(nll, _zero_nll(1) * n_valid, rtol=1e-9)


def test_dropouts_rise_with_joint_speed():
    tr = CR.TrackerProcess()
    speed = torch.linspace(0.0, 20.0, 41, dtype=D)[:, None]
    for occ in (0.0, 0.5):
        p = tr.entry_prob(speed, torch.full_like(speed, occ))
        assert (p.diff(dim=0) > 0).all()
    assert tr.entry_prob(torch.zeros(1, 1, dtype=D), torch.zeros(1, 1, dtype=D)).item() == pytest.approx(0.01)
    # the same frontal-plane swing at rising frequency: same geometry, joint speed x f
    R, t = look_at((0.0, 1.2, 4.0), (0.0, 0.95, 0.0))
    cam = Camera(intrinsics(1000.0, 1000.0, 640.0, 360.0), R, t, 0.0, 0.0, 1280, 720, 30.0)
    rates, speeds = [], []
    for f in (0.0, 1.0, 3.0):
        lost, spd = [], []
        for seed in range(3):
            rec, truth = CR.render_pose2d(stick(900, 30.0, f, amp=0.4, plane="frontal"), cam, _gen(seed))
            lost.append(truth["lost"][:, MOVING].mean())
            uv = torch.as_tensor(truth["uv"], dtype=D)
            spd.append(float(CR.image_speed(uv, 30.0)[:, MOVING].mean()))
        rates.append(float(np.mean(lost)))
        speeds.append(float(np.mean(spd)))
    assert speeds[0] < speeds[1] < speeds[2]
    assert rates[0] < rates[1] < rates[2]
    assert rates[2] > 2 * rates[0], f"lost fractions {rates}"


def test_render_is_deterministic_under_a_seed():
    pos = stick(120, 30.0, 1.5, speed=1.0)
    target = pos[:, 0].mean(0)

    def run(seed):
        g = _gen(seed)
        cam = CR.sample_camera(g, target)
        return CR.render_pose2d(pos, cam, g)

    (a, ta), (b, tb), (c, _) = run(11), run(11), run(12)
    for k in ("coords", "valid_mask"):
        assert np.array_equal(a[k], b[k])
    assert a["scale"] == b["scale"] and a["fps"] == b["fps"]
    for k in ("uv", "jitter_s", "entry_prob", "lost", "in_image"):
        assert np.array_equal(ta[k], tb[k])
    assert not np.array_equal(a["coords"], c["coords"])
    g22 = _gen(1)
    body = torch.randn(30, 22, 3, generator=g22, dtype=D) * 0.3
    Rg = axis_angle_to_matrix(torch.randn(30, 22, 3, generator=g22, dtype=D) * 0.4)
    r1, t1 = CR.render_pose3d(body, Rg, _gen(5))
    r2, t2 = CR.render_pose3d(body, Rg, _gen(5))
    r3, _ = CR.render_pose3d(body, Rg, _gen(6))
    assert np.array_equal(r1["coords"], r2["coords"]) and t1["heading"] == t2["heading"]
    assert not np.array_equal(r1["coords"], r3["coords"])


def test_tracker_noise_jitter_and_out_of_image():
    pos = stick(300, 30.0, 0.5, amp=0.3, plane="frontal")
    R, t = look_at((0.0, 1.2, 4.0), (0.0, 0.95, 0.0))
    cam = Camera(intrinsics(1000.0, 1000.0, 640.0, 360.0), R, t, 0.0, 0.0, 1280, 720, 30.0)
    noisy = CR.TrackerProcess(base_rate=0.0, jitter_s=0.0)
    rec, truth = CR.render_pose2d(pos, cam, _gen(0), noisy)
    assert rec["valid_mask"].all() and not truth["lost"].any()
    w = torch.zeros(300, dtype=torch.long)
    r = P.pose2d_residual(pos, cam, torch.as_tensor(rec["coords"], dtype=D), rec["scale"], w)
    # residual of pelvis-centred pixels: noise of the joint minus noise of the pelvis
    med = float(r[:, 1:].norm(dim=-1).median())
    assert 3.0 < med < 7.0, f"median residual {med} px"
    # jitter moves the sampled frames by d_k fps of a frame
    rec_j, truth_j = CR.render_pose2d(pos, cam, _gen(0), CR.TrackerProcess(noise_px=0.0, base_rate=0.0,
                                                                            jitter_s=0.004))
    assert 0.002 < float(np.std(truth_j["jitter_s"])) < 0.006
    # a camera looking away sees nothing
    R2, t2 = look_at((0.0, 1.2, 4.0), (0.0, 1.2, 8.0))
    away = Camera(intrinsics(1000.0, 1000.0, 640.0, 360.0), R2, t2, 0.0, 0.0, 1280, 720, 30.0)
    rec_a, truth_a = CR.render_pose2d(pos, away, _gen(0))
    assert not truth_a["in_image"].any() and not rec_a["valid_mask"].any() and rec_a["scale"] > 0


def test_jitter_truth_is_the_applied_shift_and_the_dropout_chain_persists():
    # a figure translating at constant velocity: linear interpolation is exact, so each jittered frame
    # is the figure at k / fps + d_k with d_k the returned shift, clamped end frames included
    T, fps = 200, 30.0
    v = torch.tensor([0.3, -0.1, 1.2], dtype=D)
    base = torch.randn(1, 15, 3, generator=_gen(0), dtype=D)
    t = torch.arange(T, dtype=D) / fps
    pos = base + t[:, None, None] * v
    for seed in range(20):
        jit, d = CR.jitter_frames(pos, fps, 0.01, _gen(seed))
        assert torch.allclose(jit, base + (t + d)[:, None, None] * v, atol=1e-12)
        assert float((t + d).min()) >= 0.0 and float((t + d).max()) <= float(t[-1]) + 1e-12
    # the two-state chain at a constant entry probability p: stationary lost fraction
    # p / (p + 1 - persist), mean lost run 1 / (1 - persist)
    tr = CR.TrackerProcess(base_rate=0.01, speed_gain=0.0, occlusion_gain=0.0, persist=0.6)
    lost = tr.dropouts(torch.full((5000, 32), 0.01, dtype=D), _gen(1)).numpy()
    starts = lost[0].sum() + (lost[1:] & ~lost[:-1]).sum()
    assert abs(lost.mean() - 0.01 / 0.41) < 0.004, f"lost fraction {lost.mean()}"
    assert abs(lost.sum() / starts - 2.5) < 0.25, f"mean lost run {lost.sum() / starts} frames"


def test_occlusion_proxy_follows_depth_order():
    uv = torch.tensor([[[100.0, 100.0]] * 15], dtype=D)
    uv[0, :, 0] += torch.arange(15, dtype=D) * 1000        # all joints far apart in the image
    uv[0, 13] = uv[0, 7]                                   # left ankle on the left wrist in the image
    depth = torch.full((1, 15), 4.0, dtype=D)
    depth[0, 13] = 4.3                                     # the ankle behind the wrist
    occ = CR.occlusion(uv, depth, torso_px=100.0)
    assert occ[0, 13] > 0.99 and occ[0, 7] == 0 and occ[0, [0, 1, 2]].abs().max() == 0
    # bone neighbours never hide each other (the left knee sits on the left ankle's bone)
    uv[0, 11] = uv[0, 13]
    depth[0, 11] = 3.0
    depth[0, 7] = 5.0
    occ = CR.occlusion(uv, depth, torso_px=100.0)
    assert occ[0, 13] == 0 and CR.occluders()[13, 11] == 0 and CR.occluders()[13, 7] == 1


def test_render_pose3d_contract_zero_residual_and_heading():
    g = _gen(2)
    pos = torch.randn(40, 22, 3, generator=g, dtype=D) * 0.3
    Rg = axis_angle_to_matrix(torch.randn(40, 22, 3, generator=g, dtype=D) * 0.4)
    off = torch.randn(15, 3, generator=g, dtype=D) * 0.02
    rec, truth = CR.render_pose3d(pos, Rg, _gen(8), offsets=off, fps=100.0, noise_m=0.0)
    assert rec["dim"] == 3 and rec["modality"] == "3d_meter" and rec["coords"].dtype == np.float32
    assert rec["coords"].shape == (40, 15, 3) and rec["valid_mask"].all() and rec["fps"] == 100.0
    assert list(rec["coord_mask"]) == [True, True, True] and rec["scale"] > 0
    w = torch.zeros(40, dtype=torch.long)
    obs = torch.as_tensor(rec["coords"], dtype=D)
    pred = P.to_canonical15(pos, Rg, off)
    r, yaw = P.pose3d_residual(pred, obs, rec["scale"], w, 1)
    assert r.abs().max() < 1e-5
    assert abs(math.remainder(float(yaw[0]) + truth["heading"], 2 * math.pi)) < 1e-6
    r_no, _ = P.pose3d_residual(P.to_canonical15(pos), obs, rec["scale"], w, 1)
    assert r_no.norm(dim=-1).max() > 0.01
    noisy, _ = CR.render_pose3d(pos, Rg, _gen(8), offsets=off, noise_m=0.003)
    rn, _ = P.pose3d_residual(pred, torch.as_tensor(noisy["coords"], dtype=D), noisy["scale"], w, 1)
    med = float(rn[:, 1:].norm(dim=-1).median())
    assert 0.002 < med < 0.012, f"median 3D residual {med} m"


def test_sampled_cameras_are_cameras_of_the_nuisance_family():
    """Every sample_camera camera equals camera_from_params(matrix_to_rot6d(R), log |eye - target|, log f,
    target) with no roll: the truth camera of a synthetic window is reachable by the camera nuisance."""
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.kernels.camera import camera_from_params
    g = _gen(21)
    spread = torch.tensor([2.0, 0.1, 2.0], dtype=D)
    for _ in range(100):
        target = torch.tensor([0.0, 0.95, 0.0], dtype=D) + spread * torch.randn(3, generator=g, dtype=D)
        cam = CR.sample_camera(g, target)
        eye = -cam.R.T @ cam.t
        rebuilt = camera_from_params(matrix_to_rot6d(cam.R)[None], torch.log((eye - target).norm())[None],
                                     torch.log(cam.K[0, 0])[None], target[None])
        for a, b in zip(rebuilt[:3], cam[:3]):
            assert torch.allclose(a[0], b, atol=1e-9)
        assert float(cam.R[0, 1]) == 0.0                   # camera x axis horizontal: no roll


def test_the_dropout_chain_follows_the_entry_probability_and_persistence():
    """P(lost at k | detected at k - 1) = p_k and P(lost at k | lost at k - 1) = max(persist, p_k), for
    entry probabilities below and above persist."""
    tr = CR.TrackerProcess(persist=0.6)
    probs = (0.02, 0.2, 0.8)
    lost = tr.dropouts(torch.tensor(probs, dtype=D).expand(10000, 3), _gen(4)).numpy()
    prev, cur = lost[:-1], lost[1:]
    for j, p in enumerate(probs):
        for given, want in ((~prev[:, j], p), (prev[:, j], max(0.6, p))):
            n = int(given.sum())
            rate = float(cur[given, j].mean())
            assert abs(rate - want) < 5 * math.sqrt(want * (1 - want) / n), (p, want, rate, n)


def test_pose2d_nll_of_a_tracked_render_is_lowest_at_the_true_camera():
    """Render with the default tracker (noise, dropouts, jitter), then pose2d_nll with the true body over
    the log focal (distance at truth) and the log distance (focal at truth) of the nuisance camera: the
    minimum sits at the truth within 0.015 (grid 0.005)."""
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.kernels.camera import camera_from_params
    T = 90
    pos = stick(T, 30.0, 1.0, speed=0.5)
    target = pos[:, 0].mean(0)
    grid = torch.arange(-0.1, 0.1001, 0.005, dtype=D)
    G = len(grid)
    w = torch.repeat_interleave(torch.arange(G), T)
    for seed in range(4):
        g = _gen(seed)
        cam = CR.sample_camera(g, target, fps=30.0)
        rec, _ = CR.render_pose2d(pos, cam, g)
        eye = -cam.R.T @ cam.t
        r6 = matrix_to_rot6d(cam.R)[None].expand(G, 6)
        ld, lf = torch.log((eye - target).norm()), torch.log(cam.K[0, 0])
        obs = torch.as_tensor(rec["coords"], dtype=D).repeat(G, 1, 1)
        valid = torch.as_tensor(rec["valid_mask"]).repeat(G, 1)
        for c in (camera_from_params(r6, ld.expand(G), lf + grid, target[None].expand(G, 3)),
                  camera_from_params(r6, ld + grid, lf.expand(G), target[None].expand(G, 3))):
            nll = P.pose2d_nll(pos.repeat(G, 1, 1), c, obs, rec["scale"], valid, w, G)
            assert abs(float(grid[nll.argmin()])) <= 0.015 + 1e-12, f"seed {seed}"
