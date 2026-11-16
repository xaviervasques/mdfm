"""Tests of the time-query decoder (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.5, task B4.15; amended by
reports/bench/phaseA_diagnosis.md, fixes 1 to 5, and reports/bench/root_motion_diagnosis.md, fix 1): knots decoded
one at a time equal the knots decoded together to 1e-6 (float64; float32 to 1e-5), with random weights so that the
check is not vacuous; at initialization every knot is the training mean pose of reports/body_store/priors.json (as
rotations, the root without its own yaw) standing still at the anchor (reports/body_store/window_priors.json) with
zero contact logits, and the initial body stands on the floor; a loaded mean pose rebuilds the unsaved mean
rotations; the whole decoder runs in float32 inside a bf16 autocast (equal to float32 decoding to 1e-6); the
rotation outputs are orthonormal whatever the weights; the content logits of the cross-attention stay within
+-QK_SCALE_MAX whatever the weights; the most local head attends near the query at initialization; padding patches
are never keys; FiLM never passes a gradient to the betas; scatter_dense places knots and keeps the fill,
differentiably; the pilot decoder holds about 12.7M parameters.

Integrated root (root-motion diagnosis, fix 1): constant rates decode a straight line and a circle of radius v /
omega through the anchor, the yaw exact, in windows of 1,000, 617 and 100 knots; a constant yaw rate turns past pi;
the centre knot L // 2 sits on the anchor with yaw 0 whatever the weights and the latents; the per-knot residual
stays within 2 sqrt(2) RES_MAX of the integrated path; the decoder refuses a call without the window lengths; the
true grid rates of a KIT walk of the train split (the body store) put through the operator give the window-frame
pelvis within 5 cm and the yaw within 2 deg at every knot (the discretization check of the grid).
"""
import math
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.body.rotations import rot6d_to_matrix, rot_y, yaw_angle
from mdfm.model.decoder import (C_RT, GRID_KNOTS, N_OUT, N_ROT, POS_LOGVAR0, POS_LOGVAR_MAX, POS_LOGVAR_MIN,
                                QK_SCALE_MAX, RES_MAX, BodyDecoder, all_knots, anchor_from_report, default_tau,
                                grid_of, group_slots, hermite_on_grid, integrate_root, knot_pairs,
                                mean_pose_from_report, mean_rotations, residual_rotations, scatter_dense,
                                transl0_from_report, yaw_rotate)
from mdfm.model.encoder import n_parameters
from mdfm.model.posterior import soft_clamp

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
TAU4 = (0.125, 0.5, 2.0, None)
F64 = torch.float64


def _randomized(dec, seed=0, std=0.2):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in dec.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * (std if p.dim() > 1 else 0.05))
        for mod in dec.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.weight.add_(1.0)
    return dec.eval().requires_grad_(False)


def _inputs(dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    W, P = 3, 8
    return (torch.randn(W, P, 32, generator=g, dtype=dtype), torch.randn(W, 16, generator=g, dtype=dtype),
            torch.tensor([8, 5, 3]), torch.tensor([200, 120, 70]))


def _long_inputs(seed=4, W=2, P=40):
    """10 s windows: 40 patches, 1,000 knots each."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(W, P, 32, generator=g), torch.randn(W, 16, generator=g), torch.full((W,), P),
            torch.full((W,), 1000))


def _rates_decoder(omega, u, dtype=F64):
    """A tiny decoder whose every query reads the yaw rate omega (rad/s) and the heading-frame velocity u
    (lateral, forward) m/s: output weights 0, the rate columns of the output bias set."""
    dec = BodyDecoder(d=32, layers=1, heads=4, n_fourier=4).to(dtype).eval().requires_grad_(False)
    with torch.no_grad():
        dec.out.weight.zero_()
        dec.out.bias.zero_()
        dec.out.bias[C_RT] = float(omega)
        dec.out.bias[C_RT + 1] = float(u[0])
        dec.out.bias[C_RT + 2] = float(u[1])
    return dec


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-6), (torch.float32, 1e-5)])
def test_knots_decoded_one_at_a_time_equal_the_knots_decoded_together(dtype, tol):
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8).to(dtype))
    z, beta, npw, lengths = _inputs(dtype)
    w, k = all_knots(lengths)
    dense = dec(z, beta, (w, k), npw, lengths)
    assert float(dense.rot6d.std()) > 0.05                  # random weights: the knots differ
    worst = 0.0
    for i in range(0, len(w), 29):
        one = dec(z, beta, torch.stack([w[i:i + 1], k[i:i + 1]], 1), npw, lengths)
        for a, b in zip(one[:4], dense[:4]):
            worst = max(worst, float((a - b[i:i + 1]).abs().max()))
    sub = torch.randperm(len(w), generator=torch.Generator().manual_seed(2))[:57]
    part = dec(z, beta, (w[sub], k[sub]), npw, lengths)
    for a, b in zip(part[:4], dense[:4]):
        worst = max(worst, float((a - b[sub]).abs().max()))
    for a, b in zip(part.root, dense.root):                 # the grid depends on z, beta and the lengths only
        worst = max(worst, float((a.double() - b.double()).abs().max()))
    assert worst <= tol, worst


def test_checkpointed_blocks_give_the_same_values_and_gradients():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8).double()).train().requires_grad_(True)
    z, beta, npw, lengths = _inputs(F64)
    w, k = all_knots(lengths)
    runs = []
    for flag in (False, True):
        dec.checkpoint_blocks = flag
        dec.zero_grad(set_to_none=True)
        zz = z.clone().requires_grad_(True)
        out = dec(zz, beta, (w, k), npw, lengths)
        loss = (out.rot6d ** 2).sum() + (out.transl ** 2).sum() + out.pos_logvar.sum() + (out.root.X ** 2).sum()
        loss.backward()
        runs.append((out.rot6d.detach(), zz.grad, [p.grad.clone() for p in dec.parameters()]))
    (r0, gz0, gp0), (r1, gz1, gp1) = runs
    assert torch.equal(r0, r1)
    assert float((gz0 - gz1).abs().max()) <= 1e-12
    assert all(float((a - b).abs().max()) <= 1e-12 for a, b in zip(gp0, gp1))
    with torch.no_grad():                                    # no autograd: the plain path
        assert torch.equal(dec(z, beta, (w, k), npw, lengths).rot6d, r1)


def test_initialization_is_the_mean_pose():
    dec = BodyDecoder(d=64, layers=1, heads=4, n_fourier=8)
    z, beta, npw, lengths = _inputs(torch.float32)
    w, k = all_knots(lengths)
    with torch.no_grad():
        out = dec(z, beta, (w, k), npw, lengths)
    mp = mean_pose_from_report().float()
    R_mean = mean_rotations(mp.double()).float()
    assert float((rot6d_to_matrix(out.rot6d) - R_mean).abs().max()) <= 1e-6
    assert abs(float(yaw_angle(R_mean[0]))) < 1e-6          # the mean root carries no yaw of its own
    assert float((R_mean[1:] - rot6d_to_matrix(mp)[1:]).abs().max()) <= 1e-6
    tr0, an = transl0_from_report().float(), anchor_from_report().float()
    assert torch.equal(dec.transl0, tr0) and torch.equal(dec.anchor_xz, an)
    start = torch.stack([an[0], tr0[1], an[1]])
    assert torch.equal(out.transl, start.expand_as(out.transl))
    assert 0.9 < float(tr0[1]) < 1.3                        # the pelvis of a standing adult above the floor
    assert float(an.abs().max()) < 0.1                      # minus the rest pelvis offset (a few cm)
    assert torch.equal(out.contact_logit, torch.zeros_like(out.contact_logit))
    pl0 = soft_clamp(torch.tensor(POS_LOGVAR0), POS_LOGVAR_MIN, POS_LOGVAR_MAX)
    assert torch.allclose(out.pos_logvar, pl0.expand_as(out.pos_logvar))
    for x in (out.root.psi, out.root.omega, out.root.u, out.root.X):
        assert float(x.abs().max()) == 0.0                  # standing still, yaw 0
    assert torch.equal(dec.mean_pose, mp)
    sd = dec.state_dict()
    assert "mean_R" not in sd and "inv_tau" not in sd and "anchor_xz" in sd


def test_a_loaded_mean_pose_rebuilds_the_mean_rotations():
    """mean_R is not saved: loading a state_dict whose mean_pose differs from the decoder's (a checkpoint written
    under another reports/body_store/priors.json) rebuilds it, through a parent module too, so the decoder turns
    around the checkpoint's own mean pose (the root without its own yaw); an equal mean_pose leaves mean_R bit for
    bit."""
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    src = BodyDecoder(d=64, layers=1, heads=4, n_fourier=8)
    sd = src.state_dict()
    R = rot6d_to_matrix(sd["mean_pose"].double())
    R = R @ axis_angle_to_matrix(torch.tensor([0.2, 0.0, 0.1], dtype=F64))
    other = matrix_to_rot6d(rot_y(torch.tensor(0.7, dtype=F64)) @ R)
    sd["mean_pose"] = other.float()
    dec = BodyDecoder(d=64, layers=1, heads=4, n_fourier=8)
    parent = torch.nn.ModuleDict({"decoder": dec})
    parent.load_state_dict({"decoder." + k: v for k, v in sd.items()})
    want = mean_rotations(other.float().double()).float()
    assert float((dec.mean_R - want).abs().max()) <= 1e-6
    assert float((dec.mean_R[1:] - rot6d_to_matrix(other[1:]).float()).abs().max()) <= 1e-6
    z, beta, npw, lengths = _inputs(torch.float32)
    with torch.no_grad():
        out = dec(z, beta, all_knots(lengths), npw, lengths)
    assert float((rot6d_to_matrix(out.rot6d) - want).abs().max()) <= 1e-6
    same = BodyDecoder(d=64, layers=1, heads=4, n_fourier=8)
    R0 = same.mean_R.clone()
    same.load_state_dict(src.state_dict())
    assert torch.equal(same.mean_R, R0)


def test_outputs_are_float32_under_bf16_autocast():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8))
    z, beta, npw, lengths = _inputs(torch.float32)
    w, k = all_knots(lengths)
    ref = dec(z, beta, (w, k), npw, lengths)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = dec(z, beta, (w, k), npw, lengths)
    for a in out[:4]:
        assert a.dtype == torch.float32
    assert torch.isfinite(out.rot6d).all()
    for a, b in zip(out[:4], ref[:4]):
        assert float((a - b).abs().max()) <= 1e-6        # the trunk ignores the caller's bf16 autocast
    assert out.root.u.dtype == torch.float32 and float((out.root.X - ref.root.X).abs().max()) <= 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_decoder_ignores_a_cuda_bf16_autocast():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8)).cuda()
    z, beta, npw, lengths = (x.cuda() for x in _inputs(torch.float32))
    w, k = all_knots(lengths)
    ref = dec(z, beta, (w, k), npw, lengths)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dec(z, beta, (w, k), npw, lengths)
    for a, b in zip(out[:4], ref[:4]):
        assert a.dtype == torch.float32 and float((a - b).abs().max()) <= 1e-6


def test_the_decoder_needs_the_window_lengths():
    dec = BodyDecoder(d=32, layers=1, heads=4, n_fourier=4)
    z, beta, npw, lengths = _inputs(torch.float32)
    with pytest.raises(ValueError, match="lengths"):
        dec(z, beta, all_knots(lengths), npw)
    with pytest.raises(ValueError, match="lengths"):
        dec(z, beta, all_knots(lengths), npw, lengths[:2])


def test_rotation_outputs_are_orthonormal_whatever_the_weights():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8, tau=TAU4))
    with torch.no_grad():
        dec.out.weight.mul_(1e3)
    z, beta, npw, lengths = _inputs(torch.float32)
    out = dec(z, beta, all_knots(lengths), npw, lengths)
    a1, a2 = out.rot6d[..., :3], out.rot6d[..., 3:]
    assert float(out.rot6d.abs().max()) <= 1.0 + 1e-5       # no free scale: the raw outputs reach about 1e3
    assert float((a1.norm(dim=-1) - 1).abs().max()) < 1e-5
    assert float((a2.norm(dim=-1) - 1).abs().max()) < 1e-5
    assert float((a1 * a2).sum(-1).abs().max()) < 1e-5
    assert dec.out.out_features == N_OUT and N_ROT == 2 + 21 * 3


# ---------------------------------------------------------------- the integrated root
def test_the_grid_is_anchored_at_the_centre_knot():
    g = grid_of(torch.tensor([1000, 617, 100, 1]))
    assert g.n.tolist() == [101, 63, 11, 2] and g.c.tolist() == [500, 308, 50, 0]
    for w, (L, c) in enumerate(((1000, 500), (617, 308), (100, 50), (1, 0))):
        k = g.k[g.w == w]
        assert bool((torch.diff(k) == GRID_KNOTS).all()) and int(k[0]) <= 0 and int(k[-1]) >= L - 1
        assert int(k[0]) > -GRID_KNOTS and int(k[-1]) < L - 1 + GRID_KNOTS or L == 1
        assert c in k.tolist() and bool((g.i[g.w == w] == (k - c) // GRID_KNOTS).all())


@pytest.mark.parametrize("omega", [0.0, 0.5])
def test_root_is_the_integral_of_its_rates(omega):
    """Constant rates: omega = 0 decodes the +z line at 1.2 m/s from the anchor to 1e-5 m at every knot; omega =
    0.5 rad/s decodes psi = 0.5 (t - t_c) to 1e-5 rad (the trapezoid and the Hermite are exact for a constant rate)
    and the circle of radius 2.4 m through the anchor, tangent to +z there, within 1 cm, in windows of 1,000, 617 and
    100 knots at once. The path is a function of the local rates alone: what the root-motion diagnosis (C1) found
    missing."""
    v = 1.2
    dec = _rates_decoder(omega, (0.0, v))
    lengths = torch.tensor([1000, 617, 100])
    W = len(lengths)
    z, beta = torch.zeros(W, 40, 32, dtype=F64), torch.zeros(W, 16, dtype=F64)
    w, k = all_knots(lengths)
    out = dec(z, beta, (w, k), torch.tensor([40, 25, 4]), lengths)
    tc = (lengths // 2)[w].double()
    tau = (k.double() - tc) / 100.0
    psi = yaw_angle(rot6d_to_matrix(out.rot6d[:, 0]))
    want_psi = omega * tau
    assert float((torch.remainder(psi - want_psi + math.pi, 2 * math.pi) - math.pi).abs().max()) <= 1e-5
    X = out.transl[:, [0, 2]] - dec.anchor_xz.double()
    if omega == 0.0:
        want = torch.stack([torch.zeros_like(tau), v * tau], -1)
        assert float((X - want).norm(dim=-1).max()) <= 1e-5
    else:
        R = v / omega                                      # 2.4 m
        want = torch.stack([R * (1 - torch.cos(omega * tau)), R * torch.sin(omega * tau)], -1)
        assert float((X - want).norm(dim=-1).max()) <= 0.01
        centre = torch.tensor([R, 0.0], dtype=F64)
        assert float(((X - centre).norm(dim=-1) - R).abs().max()) <= 0.01
    assert float(out.transl[:, 1].std()) == 0.0           # the vertical root is not integrated


def test_a_constant_yaw_rate_turns_past_pi():
    dec = _rates_decoder(1.3, (0.0, 0.0))
    lengths = torch.tensor([1001])
    z, beta = torch.zeros(1, 40, 32, dtype=F64), torch.zeros(1, 16, dtype=F64)
    out = dec(z, beta, (torch.tensor([0, 0]), torch.tensor([500, 1000])), None, lengths)
    g = out.root
    assert float(g.psi[g.k == 1000]) == pytest.approx(6.5, abs=1e-9)      # 1.3 rad/s over 5 s, past pi
    R = rot6d_to_matrix(out.rot6d[:, 0])
    assert float(yaw_angle(R[1])) == pytest.approx(6.5 - 2 * math.pi, abs=1e-9)
    # the rotation head: psi and psi + 2 pi give one body; the tilt turns the root about horizontal axes only
    mean_R = mean_rotations(mean_pose_from_report())
    y = torch.zeros(3, N_ROT, dtype=F64)
    Rr = rot6d_to_matrix(residual_rotations(y, mean_R, torch.tensor([0.0, 2.0, 2.0 + 4 * math.pi], dtype=F64)))
    assert float((Rr[0] - mean_R).abs().max()) < 1e-12 and float((Rr[2] - Rr[1]).abs().max()) < 1e-9
    assert float((Rr[1, 0] - rot_y(torch.tensor(2.0, dtype=F64)) @ mean_R[0]).abs().max()) < 1e-12
    y[0, 0] = 0.05
    Rt = rot6d_to_matrix(residual_rotations(y[:1], mean_R, torch.tensor([0.3], dtype=F64)))
    assert abs(float(yaw_angle(Rt[0, 0])) - 0.3) < 0.01


def test_the_centre_knot_is_anchored_whatever_the_weights():
    """With the output weights times 1e3, the centre knot L // 2 decodes the anchor (x, z) and yaw 0 to 1e-5 for any
    latent: the window frame's origin cannot drift (the root-motion diagnosis measured the centre 0.146 m off)."""
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8))
    with torch.no_grad():
        dec.out.weight.mul_(1e3)
    lengths = torch.tensor([1000, 617, 100])
    c = lengths // 2
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        z, beta = 3 * torch.randn(3, 40, 32, generator=g), torch.randn(3, 16, generator=g)
        out = dec(z, beta, (torch.arange(3), c), torch.tensor([40, 25, 4]), lengths)
        assert float((out.transl[:, [0, 2]] - dec.anchor_xz).abs().max()) <= 1e-5
        centre = out.root.k == c[out.root.w]
        assert int(centre.sum()) == 3 and float(out.root.psi[centre].abs().max()) == 0.0
        assert float(out.root.X[centre].abs().max()) == 0.0
        assert float(out.root.u.abs().max()) > 10.0            # the weights are far from 0: the check is not vacuous


def test_the_heading_frame_residual_is_bounded():
    """With the output weights times 1e3, the horizontal root stays within 2 sqrt(2) RES_MAX of the anchor plus the
    integrated path at every knot (the knot's residual and the centre's, each at most RES_MAX per component)."""
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8).double())
    with torch.no_grad():
        dec.out.weight.mul_(1e3)
    z, beta, npw, lengths = _inputs(F64)
    w, k = all_knots(lengths)
    out = dec(z, beta, (w, k), npw, lengths)
    g = grid_of(lengths)
    h = GRID_KNOTS / 100.0
    rg = out.root
    psi_g, X_g, V = integrate_root(rg.omega, rg.u, g, h)
    assert float((psi_g - rg.psi).abs().max()) == 0.0 and float((X_g - rg.X).abs().max()) == 0.0
    X = hermite_on_grid(X_g, V, g, w, k, h)
    r = out.transl[:, [0, 2]] - dec.anchor_xz.double() - X
    assert float(r.norm(dim=-1).max()) <= 2 * math.sqrt(2) * RES_MAX + 1e-9
    assert float(r.norm(dim=-1).max()) > RES_MAX                  # saturated residuals: the bound holds, not 0
    assert float(yaw_rotate(torch.tensor([[1.0, 0.0]]), torch.tensor([0.5]))[0, 1]) == pytest.approx(-math.sin(0.5))


def test_root_operator_reproduces_a_true_walk():
    """The discretization check of the grid (root-motion diagnosis, fix 1): the true grid rates (losses.
    truth_root_grid, the decoder's targets) of the first 10 s of KIT/11/WalkingStraightForwards01, a train take of
    the body store, through the decoder's operator give the window-frame pelvis within 5 cm and the root yaw within
    2 deg at every knot."""
    from mdfm.data.body_store import BodyStore, store_root
    from mdfm.model.mfm import KnotBody
    from mdfm.train import losses as LS
    if not os.path.isdir(os.path.join(store_root(), "train")):
        pytest.skip("body store not built")
    store = BodyStore(split="train")
    t = store.takes
    i = int(t.index[t["take_id"] == "KIT/11/WalkingStraightForwards01_poses.npz"][0])
    bw = store.windows(np.array([i]), np.array([0]))
    truth = KnotBody(bw.rot6d.double(), bw.transl.double(), None, bw.betas, bw.gender, bw.lengths.long())
    lengths = truth.lengths
    assert int(lengths[0]) > 300
    tg = LS.truth_root_grid(truth, 100.0, lengths)
    g = grid_of(lengths)
    h = GRID_KNOTS / 100.0
    psi_g, X_g, V = integrate_root(tg.omega, tg.u, g, h)
    w, k = all_knots(lengths)
    psi = hermite_on_grid(psi_g, tg.omega, g, w, k, h)
    X = hermite_on_grid(X_g, V, g, w, k, h)
    c = lengths[0] // 2
    yaw = yaw_angle(rot6d_to_matrix(truth.rot6d[0, :, 0]))
    ey = torch.remainder(psi - (yaw[k] - yaw[c]) + math.pi, 2 * math.pi) - math.pi
    ex = (X - (truth.transl[0, k] - truth.transl[0, c])[:, [0, 2]]).norm(dim=-1)
    assert float(truth.transl[0, :, 2].max() - truth.transl[0, :, 2].min()) > 1.0      # it walks
    assert float(ex.max()) <= 0.05, float(ex.max())
    assert math.degrees(float(ey.abs().max())) <= 2.0, math.degrees(float(ey.abs().max()))


# ---------------------------------------------------------------- attention
def test_content_logits_are_bounded_whatever_the_weights():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8, tau=TAU4))
    with torch.no_grad():
        for blk in dec.blocks:
            blk.q.weight.mul_(1e3)
            blk.k.weight.mul_(1e3)
            blk.nq.weight.mul_(10.0)
            blk.scale_raw.fill_(50.0)
        dec.nm.weight.mul_(10.0)
    z, beta, npw, lengths = _inputs(torch.float32)
    s = dec._setup(z * 100, beta, all_knots(lengths), npw)
    x, worst = s.x, 0.0
    with torch.no_grad():
        for blk in dec.blocks:
            q, k = blk.qk(x, s.mem, s.qcs, s.kcs)
            worst = max(worst, float((q @ k.transpose(-1, -2)).abs().max()))
            x = blk(x, s.mem, s.qcs, s.kcs, s.bias)
    assert worst <= QK_SCALE_MAX + 1e-4, worst
    assert worst > 0.5 * QK_SCALE_MAX                        # near the bound: the check is not vacuous


def test_local_heads_attend_near_the_query():
    dec = BodyDecoder(d=64, layers=2, heads=4, n_fourier=8, tau=TAU4).eval()
    z, beta, npw, lengths = _long_inputs()
    s = dec._setup(z, beta, all_knots(lengths), npw)
    near = ((s.tq[:, :, None] - s.tp[None, None, :]).abs() <= 0.5).float()                   # (W, Q, P)
    x = s.x
    with torch.no_grad():
        for blk in dec.blocks:
            q, k = blk.qk(x, s.mem, s.qcs, s.kcs)
            p = torch.softmax(q @ k.transpose(-1, -2) + s.bias, -1)                           # (W, H, Q, P)
            mass = (p * near[:, None]).sum(-1)                                                 # (W, H, Q)
            head = {t: float(mass[:, h][s.valid].mean()) for h, t in enumerate(TAU4)}
            assert head[0.125] >= 0.9, head
            assert head[None] < 0.5, head                    # the head without bias reads the whole window
            x = blk(x, s.mem, s.qcs, s.kcs, s.bias)
    stats = dec.attention_stats(z, beta, all_knots(lengths), npw)
    assert len(stats) == 2 and all(1.0 < b["entropy"] < np.log(40) for b in stats), stats
    assert all(abs(b["scale"] - 2.0) < 1e-6 for b in stats)
    assert default_tau(8) == (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, None) and default_tau(4)[-1] is None
    with pytest.raises(ValueError):
        BodyDecoder(d=64, layers=1, heads=4, n_fourier=8, tau=(0.125, None))


def test_film_passes_no_gradient_to_the_betas():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8)).requires_grad_(True)
    z, beta, npw, lengths = _inputs(torch.float32)
    beta = beta.clone().requires_grad_(True)
    out = dec(z, beta, all_knots(lengths), npw, lengths)
    with torch.no_grad():                                   # FiLM reads beta
        assert float((out.rot6d - dec(z, beta * 0, all_knots(lengths), npw, lengths).rot6d).abs().max()) > 1e-3
    loss = out.rot6d.sum() + out.transl.sum() + out.root.u.sum()
    g = torch.autograd.grad(loss, beta, allow_unused=True)[0]
    assert g is None or float(g.abs().max()) == 0.0


def test_padding_patches_are_never_keys():
    dec = _randomized(BodyDecoder(d=64, layers=2, heads=4, n_fourier=8).double())
    z, beta, npw, lengths = _inputs(torch.float64)
    w, k = all_knots(lengths)
    out = dec(z, beta, (w, k), npw, lengths)
    z2 = z.clone()
    z2[1, 5:] = 50.0
    z2[2, 3:] = -50.0
    out2 = dec(z2, beta, (w, k), npw, lengths)
    assert torch.equal(out.rot6d, out2.rot6d) and torch.equal(out.transl, out2.transl)
    assert torch.equal(out.root.u, out2.root.u)


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_the_initial_body_stands_on_the_floor():
    """At initialization (tiny MFM) every knot, posed on the neutral prior-mean skeleton, keeps every ankle and
    foot joint within -0.05 to 0.20 m of the y = 0 floor (with transl0 = 0 the feet stood at -1.11 m)."""
    from mdfm.body.smplh import fk
    from mdfm.model import mfm as M
    torch.manual_seed(0)
    model = M.MFM(M.TINY).eval()
    z, beta, npw, lengths = _long_inputs(W=2, P=40)
    w, k = all_knots(lengths)
    with torch.no_grad():
        out = model.decode(z, beta, (w, k), npw, lengths)
        g = torch.full((len(w),), 2, dtype=torch.int64)
        rest = model.skeleton.rest(model.shape_prior.mean[g], g)
        pos, _ = fk(rot6d_to_matrix(out.rot6d), out.transl, rest)
    feet = pos[:, [7, 8, 10, 11], 1]
    assert float(feet.min()) >= -0.05 and float(feet.max()) <= 0.20, (float(feet.min()), float(feet.max()))
    # the anchored root puts the pelvis near the window frame's origin at the centre knot
    c = (w == 0) & (k == 500)
    assert float(pos[c, 0][:, [0, 2]].abs().max()) < 0.03


def test_scatter_dense_places_knots_and_keeps_the_fill():
    vals = torch.randn(5, 22, 6, requires_grad=True)
    fill = torch.randn(22, 6)
    w, k = torch.tensor([0, 0, 1, 2, 2]), torch.tensor([3, 7, 0, 1, 9])
    d = scatter_dense(vals, w, k, 3, 10, fill)
    assert tuple(d.shape) == (3, 10, 22, 6)
    assert torch.equal(d[w, k], vals.detach())
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[w, k] = False
    assert torch.equal(d[mask], fill.expand(int(mask.sum()), 22, 6))
    d.sum().backward()
    assert torch.equal(vals.grad, torch.ones_like(vals))


def test_knot_helpers():
    w, k = all_knots(torch.tensor([3, 1, 2]))
    assert w.tolist() == [0, 0, 0, 1, 2, 2] and k.tolist() == [0, 1, 2, 0, 0, 1]
    slot, Q = group_slots(torch.tensor([2, 0, 2, 2, 0]), 3)
    assert slot.tolist() == [0, 0, 1, 2, 1] and Q == 3
    a, b = knot_pairs(torch.tensor([[1, 5], [0, 2]]))
    assert a.tolist() == [1, 0] and b.tolist() == [5, 2]


def test_pilot_decoder_holds_about_12_7m_parameters():
    dec = BodyDecoder()
    n = n_parameters(dec)
    assert 12_500_000 <= n <= 13_000_000, n
    assert dec.out.out_features == N_OUT == 2 + 21 * 3 + 3 + 2 + 22 + 3
    assert sum(b.scale_raw.numel() for b in dec.blocks) == 32 and dec.tau == default_tau(8)
    assert np.isclose(float(dec.freqs[0]), 0.05) and np.isclose(float(dec.freqs[-1]), 20.0)
