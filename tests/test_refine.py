"""Tests of the refinement (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15): steps = 0 is the
identity; the objective decreases on a toy batch; no model parameter receives a gradient or changes;
straight_through carries the refined values forward and the amortized gradient backward; with the body
held at its truth knots, MFM.render reproduces the synthetic renderer's clean IMU (mounts, placement and
kinematics plumbed alike), and a synthetic xsens13 window with every mount 20 degrees off its prior centre
is refined to under 5 degrees in 20 steps, on 8 random draws of the 13 mount errors at once.

Body: the KIT walk of tests/test_synth.py (a train archive, 5 s), turned into BodyWindows here (floor at y =
0, pelvis x and z at 0 at the centre knot), so that no test needs the body store. The tests with a body
need the MoveAtlas SMPL+H model files and the AMASS take (the cache or the raw archive, read-only).
"""
import math
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.body.rotations import axis_angle_to_matrix, geodesic, matrix_to_rot6d
from mdfm.kernels.nuisance import IMUNuisance, identity6d
from mdfm.model import mfm as M
from mdfm.model.refine import BETAS, LR, STEPS_EVAL, refine, straight_through

import model_fixture as FX

F64 = torch.float64
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
WALK = ("KIT", "KIT/11/WalkingStraightForwards01_poses.npz")
N_DRAWS = 8
ERROR_DEG = 20.0


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return M.MFM(M.TINY)


@pytest.fixture(scope="module")
def toy(tiny):
    return M.batch_from_windows([FX.make_window(2.0, seed=0), FX.make_window(1.5, seed=3, person="p1", sex=0),
                                 FX.make_pose_window(1.0, seed=5, person="p2")], gen=0)


def _walk_bodies(n_windows, n=500, K=1000):
    """BodyWindows of the first n knots of the KIT walk, repeated n_windows times (tests/test_synth.py)."""
    from mdfm.body import frames
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.smplh import SMPLHBody, SkeletonBank, gender_code
    from mdfm.data import amass_index
    from mdfm.data.amass import load_sequence
    from mdfm.data.body_store import BodyWindows
    from mdfm.data.prep import amass_to_body_clock
    if os.path.exists(amass_index.INDEX_PATH):
        assert amass_index.check_member(*WALK) == "train"
    seq = load_sequence(*WALK)
    r6, tr, betas = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=SMPLHBody(seq["gender"], dtype=F64))
    g = gender_code(seq["gender"])
    r6, tr = r6[:n], tr[:n].clone()
    rest = SkeletonBank(F64).rest(betas[None].to(F64), torch.tensor([g]))[0]
    joints = KnotBatch(r6[None], tr[None], rest[None]).pose(torch.zeros(n, dtype=torch.int64),
                                                              torch.arange(n, dtype=F64) / 100.0)[0]
    tr[:, 1] -= frames.floor_height(joints)
    tr[:, [0, 2]] -= joints[n // 2, 0, [0, 2]]
    R6 = torch.cat([r6, r6[-1:].expand(K - n, 22, 6)]).float()
    TR = torch.cat([tr, tr[-1:].expand(K - n, 3)]).float()
    rep = lambda x: x[None].expand((n_windows,) + tuple(x.shape)).clone()  # noqa: E731
    return BodyWindows(rep(R6), rep(TR), torch.zeros(n_windows, K, 2, dtype=torch.bool),
                       (torch.arange(K) < n)[None].expand(n_windows, K).clone(), torch.full((n_windows,), n),
                       rep(betas.float()), torch.full((n_windows,), g),
                       np.array([f"test:kit:{i}" for i in range(n_windows)]),
                       torch.zeros(n_windows, dtype=torch.int64), torch.zeros(n_windows, dtype=torch.int64),
                       torch.zeros(n_windows, dtype=torch.int64))


@pytest.fixture(scope="module")
def mounted():
    """N_DRAWS clean xsens13 windows of the KIT walk, every mount ERROR_DEG off the prior centre about a random
    axis (seeded), rendered with band copies (band=True) and without (band=False); the prior rotations R0
    (13, 3, 3) and true mounts (N_DRAWS, 13, 3, 3) float64."""
    from mdfm.body.smplh import SkeletonBank
    from mdfm.factory import layouts as L
    from mdfm.factory.synth import RenderSpec, SynthSampler
    from mdfm.kernels.nuisance import MountPriors
    lay = L.load_layouts()["xsens13"]
    pr = MountPriors.load()
    sites = list(lay.streams[0].site_ids)
    R0 = pr.R0[pr.families.index("xsens13"), sites]
    rng = np.random.default_rng(20260929)
    ax = rng.standard_normal((N_DRAWS, len(sites), 3))
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    M_true = R0 @ axis_angle_to_matrix(torch.tensor(ax * math.radians(ERROR_DEG)))
    sampler = SynthSampler(layouts={"xsens13": lay}, skel=SkeletonBank(torch.float32), dtype=torch.float32,
                           device="cpu")
    bodies = _walk_bodies(N_DRAWS)
    specs = [RenderSpec(lay, mount=M_true[i].numpy(), clean=True) for i in range(N_DRAWS)]
    return {"band": sampler.render(bodies, specs, 0), "raw": sampler.render(bodies, specs, 0, band=False),
            "R0": R0, "M_true": M_true, "sites": sites}


def _prior_centre(model, batch):
    """Posterior of a batch at the nuisance heads' prior centres (zero head inputs); latents unused."""
    W, P = batch.n_windows, batch.tokens.lat_idx.shape[1]
    d = model.cfg.d
    fam = batch.win["imu_family"]
    nuis = M.Nuisance(model.heads.imu(torch.zeros(W, 14, d), fam), model.heads.camera(torch.zeros(W, d)),
                      model.heads.pose3d(torch.zeros(W, d)))
    z = torch.zeros(W, P, model.cfg.d_z)
    return M.Posterior(z, z.clone(), torch.zeros(W, 16), torch.zeros(W, 16), nuis.detach(), batch.tokens.lat_idx >= 0)


# ---------------------------------------------------------------- tests without a body
@needs_ma
def test_steps_zero_is_the_identity(tiny, toy):
    post = tiny.encode(toy)
    out = refine(tiny, post, toy, 0)
    assert torch.equal(out.mu_z, post.mu_z) and torch.equal(out.mu_b, post.mu_b)
    for k, v in post.nuis.tensors().items():
        assert torch.equal(out.nuis.tensors()[k], v)
    assert not out.mu_z.requires_grad


@needs_ma
def test_objective_decreases_and_no_parameter_moves(tiny, toy):
    before = {k: v.clone() for k, v in tiny.state_dict().items()}
    tiny.zero_grad(set_to_none=True)
    post = tiny.encode(toy)
    hist = []
    out = refine(tiny, post, toy, 5, history=hist)
    with torch.no_grad():
        j0 = -tiny.log_joint(post.mu_z, post.mu_b, post.nuis, toy).sum()
        j5 = -tiny.log_joint(out.mu_z, out.mu_b, out.nuis, toy).sum()
    assert len(hist) == 5 and abs(hist[0] - float(j0)) <= 1e-4 * abs(float(j0)) + 1e-3
    assert float(j5) < float(j0), (float(j0), float(j5))
    assert all(p.grad is None for p in tiny.parameters())
    after = tiny.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert torch.equal(out.logvar_z, post.logvar_z.detach()) and torch.equal(out.logvar_b, post.logvar_b.detach())
    assert float((out.mu_z - post.mu_z.detach()).abs().max()) > 0 and not out.mu_z.requires_grad
    # the subsampled form runs and stays finite
    sub = refine(tiny, post, toy, 3, points_per_stream=16, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(sub.mu_z).all() and all(torch.isfinite(v).all() for v in sub.nuis.tensors().values())


@needs_ma
def test_straight_through_forwards_the_refined_values(tiny, toy):
    post = tiny.encode(toy)
    ref = refine(tiny, post, toy, 2)
    st = straight_through(post, ref)
    assert torch.allclose(st.mu_z, ref.mu_z) and torch.allclose(st.mu_b, ref.mu_b)
    for k, v in ref.nuis.tensors().items():
        assert torch.allclose(st.nuis.tensors()[k], v, atol=1e-6)
    tiny.zero_grad(set_to_none=True)
    (st.mu_z.sum() + st.nuis.imu.bias_acc.sum()).backward()
    assert tiny.latent_head.proj.weight.grad is not None and float(tiny.latent_head.proj.weight.grad.abs().sum()) > 0
    tiny.zero_grad(set_to_none=True)


# ---------------------------------------------------------------- with the body held at truth
@needs_ma
def test_render_reproduces_the_synthetic_imu_at_truth(tiny, mounted):
    batch = M.batch_from_windows(mounted["raw"][:2])
    assert batch.truth is not None and batch.families[int(batch.win["imu_family"][0])] == "xsens13"
    post = _prior_centre(tiny, batch)
    sites = mounted["sites"]
    dR = mounted["R0"].transpose(-1, -2) @ mounted["M_true"][:2]              # (2, 13, 3, 3)
    res = post.nuis.imu.mount_res6d.clone()
    res[:, sites] = (matrix_to_rot6d(dR) - identity6d(F64)).float()
    nuis = M.Nuisance(IMUNuisance(res, post.nuis.imu.offset_res, post.nuis.imu.bias_acc, post.nuis.imu.bias_gyro,
                                  post.nuis.imu.log_scale, post.nuis.imu.gain_res), post.nuis.camera,
                      post.nuis.pose3d_log_sigma)
    with torch.no_grad():
        pred = tiny.render(batch.truth, batch, nuis)["imu"]
    p = batch.points["imu"]
    acc = (pred["acc"] - p.obs[..., :3]).abs()[p.valid[..., :3]]
    gyro = (pred["gyro"] - p.obs[..., 3:]).abs()[p.valid[..., 3:]]
    assert int(p.valid.sum()) == 2 * 499 * 13 * 6
    assert float(acc.max()) < 1e-4 and float(gyro.max()) < 1e-2, (float(acc.max()), float(gyro.max()))


@needs_ma
def test_twenty_degree_mounts_are_refined_under_five_degrees(tiny, mounted):
    batch = M.batch_from_windows(mounted["band"])
    post = _prior_centre(tiny, batch)
    sites = mounted["sites"]
    start = torch.rad2deg(geodesic(mounted["R0"], mounted["M_true"]))
    assert torch.allclose(start, torch.full_like(start, ERROR_DEG), atol=1e-6)
    out = refine(tiny, post, batch, STEPS_EVAL, lr=LR, body=batch.truth)
    assert STEPS_EVAL == 20 and BETAS == (0.5, 0.9)
    mount = mounted["R0"] @ out.nuis.imu.residual_rotation()[:, sites].double()
    err = torch.rad2deg(geodesic(mount, mounted["M_true"]))                    # (N_DRAWS, 13) deg
    assert float(err.max()) < 5.0, err.max(1).values.tolist()
    assert all(p.grad is None for p in tiny.parameters())
