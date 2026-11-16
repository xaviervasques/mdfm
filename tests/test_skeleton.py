"""Tests of the trainable body core (spec B4.2, docs/MFM_MODEL_CODE_SPEC_2026-09-28.md sections
2.1 and 7): the rsqrt Gram-Schmidt under nested jvp, yaw and geodesic helpers, the joint-space
shape basis, the gender-routed SkeletonBank, rest joints computed once per subject, empty
state_dicts, forward kinematics against the MoveAtlas canonical record of the KIT take of
tests/amass_fixture.py (a train archive, read through mdfm.data.canonical.load), and the GPU
memory of forward kinematics with backward (skipped without a GPU).
"""
import math
import os

import numpy as np
import pytest
import torch
from torch.func import jvp

from amass_fixture import TAKE, assert_train_take, canonical_row
from mdfm import paths
from mdfm.body.hermite import hermite
from mdfm.body.rotations import (Z_UP_TO_Y_UP, axis_angle_to_matrix, geodesic, matrix_to_rot6d,
                                 rot6d_to_matrix, rot_y, yaw_angle)

F64 = torch.float64
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


def _rot_x(phi):
    c, s = math.cos(phi), math.sin(phi)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=F64)


def _nested(f, t):
    """f(t), df/dt and d2f/dt2 by two nested torch.func.jvp calls."""
    ones = torch.ones_like(t)

    def df(tt):
        return jvp(f, (tt,), (ones,))

    (y, dy), (_, d2y) = jvp(df, (t,), (ones,))
    return y, dy, d2y


# rotations

def test_rot6d_backward_through_two_nested_jvp_passes_gradcheck():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(4, 2, 6, generator=g, dtype=F64)  # 4 knots of 2 joints on the 100 Hz clock
    t = torch.tensor([0.004, 0.013, 0.026], dtype=F64)  # inside the three segments

    def derivs(yy):
        return _nested(lambda tt: rot6d_to_matrix(hermite(yy, tt, 100.0)), t)

    R, dR, d2R = derivs(y)
    assert d2R.abs().max() > 1.0  # a genuine second derivative, not a constant
    # the second derivative is the time derivative of the first (central difference inside a segment)
    h = 1e-6
    fd = (_nested(lambda tt: rot6d_to_matrix(hermite(y, tt, 100.0)), t + h)[1]
          - _nested(lambda tt: rot6d_to_matrix(hermite(y, tt, 100.0)), t - h)[1]) / (2 * h)
    assert (fd - d2R).abs().max() < 1e-5 * d2R.abs().max()
    assert torch.autograd.gradcheck(derivs, (y.clone().requires_grad_(True),))


def test_rot6d_gives_orthonormal_matrices():
    x = torch.randn(1000, 6, generator=torch.Generator().manual_seed(1), dtype=F64) * 3
    R = rot6d_to_matrix(x)
    eye = torch.eye(3, dtype=F64).expand_as(R)
    assert (R.transpose(-1, -2) @ R - eye).abs().max() < 1e-10
    assert (torch.linalg.det(R) - 1).abs().max() < 1e-10


def test_yaw_angle_and_geodesic():
    th = torch.linspace(-3.1, 3.1, 101, dtype=F64)
    assert (yaw_angle(rot_y(th)) - th).abs().max() < 1e-12
    # rot_y is the right-handed rotation about +y (the axis-angle (0, th, 0)): +z turns toward +x
    aa = torch.stack([torch.zeros_like(th), th, torch.zeros_like(th)], -1)
    assert (rot_y(th) - axis_angle_to_matrix(aa)).abs().max() < 1e-12
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    assert (rot_y(torch.tensor(math.pi / 2, dtype=F64)) @ ez - torch.tensor([1.0, 0.0, 0.0], dtype=F64)).abs().max() < 1e-12
    # a pitch about x after the yaw leaves the heading of +z unchanged
    assert (yaw_angle(rot_y(th) @ _rot_x(0.4)) - th).abs().max() < 1e-12
    # geodesic: rotation angle of R1^T R2, symmetric, zero on the diagonal
    d = geodesic(rot_y(th), rot_y(th.flip(0)))
    wrapped = torch.remainder(th - th.flip(0) + math.pi, 2 * math.pi) - math.pi
    assert (d - wrapped.abs()).abs().max() < 1e-9
    aa = torch.randn(200, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    aa = aa / aa.norm(dim=-1, keepdim=True) * torch.linspace(0.0, 3.1, 200, dtype=F64)[:, None]
    R = axis_angle_to_matrix(aa)
    Q = axis_angle_to_matrix(torch.randn(200, 3, generator=torch.Generator().manual_seed(3), dtype=F64))
    assert (geodesic(Q, Q @ R) - aa.norm(dim=-1)).abs().max() < 1e-9
    assert (geodesic(Q @ R, Q) - geodesic(Q, Q @ R)).abs().max() < 1e-12
    v = torch.zeros(3, dtype=F64, requires_grad=True)
    (geodesic(torch.eye(3, dtype=F64), axis_angle_to_matrix(v)) ** 2).backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().max() == 0


# skeleton

@pytest.fixture(scope="module")
def bodies():
    from mdfm.body.smplh import GENDERS, SMPLHBody
    return {g: SMPLHBody(g, dtype=F64) for g in GENDERS}


@pytest.fixture(scope="module")
def bank():
    from mdfm.body.smplh import SkeletonBank
    return SkeletonBank(F64)


def _random_pose(n, seed, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    R = axis_angle_to_matrix(torch.randn(n, 22, 3, generator=g, dtype=F64) * scale)
    return R, torch.randn(n, 3, generator=g, dtype=F64), torch.randn(n, 16, generator=g, dtype=F64) * 2


@needs_ma
def test_rest_from_basis_equals_vertex_path_for_all_genders(bodies, bank):
    from mdfm.body.smplh import GENDERS
    betas = torch.randn(16, 16, generator=torch.Generator().manual_seed(4), dtype=F64) * 2
    for k, g in enumerate(GENDERS):
        vertex = bodies[g].rest_joints(betas)[:, :22]
        assert (bodies[g].rest_joints22(betas) - vertex).abs().max() < 1e-9
        codes = torch.full((16,), k, dtype=torch.int64)
        assert (bank.rest(betas, codes) - vertex).abs().max() < 1e-9
        assert tuple(bodies[g].J_template.shape) == (22, 3) and tuple(bodies[g].J_dirs.shape) == (22, 3, 16)
    assert tuple(bank.J_template.shape) == (3, 22, 3) and tuple(bank.J_dirs.shape) == (3, 22, 3, 16)
    # the three skeletons differ: routing matters
    r = bank.rest(betas[:1].expand(3, -1), torch.arange(3))
    assert min((r[i] - r[j]).abs().max() for i, j in ((0, 1), (0, 2), (1, 2))) > 0.005


@needs_ma
def test_mixed_gender_batch_equals_per_gender_calls(bodies, bank):
    from mdfm.body.smplh import GENDERS, gender_code
    gender = torch.tensor([0, 1, 2, 2, 1, 0, 1, 2, 0], dtype=torch.int64)
    R, tr, betas = _random_pose(len(gender), seed=5)
    rest = bank.rest(betas, gender)
    pos, Rg = bank.fk(R, tr, rest)
    for k, g in enumerate(GENDERS):
        sel = gender == k
        assert (rest[sel] - bank.rest(betas[sel], gender[sel])).abs().max() < 1e-12
        p_k, R_k = bodies[g](R[sel, 0], R[sel, 1:], tr[sel], betas[sel])
        assert (pos[sel] - p_k).abs().max() < 1e-9 and (Rg[sel] - R_k).abs().max() < 1e-12
        assert gender_code(g) == k
    assert gender_code(None) == 2 and gender_code("unknown") == 2 and gender_code("") == 2
    # the source-table spellings (mdfm.data.sources: 'F', 'M') route to the gendered skeletons
    assert [gender_code(s) for s in ("F", "M", "Female", " male ", b"female")] == [0, 1, 0, 1, 0]
    for bad in ("fem", "x", "2"):
        with pytest.raises(ValueError):
            gender_code(bad)
    with pytest.raises(TypeError):
        bank.rest(betas[:2], torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError):
        bank.rest(betas[:2], torch.tensor([0, 3]))


@needs_ma
def test_forward_with_rest_and_broadcast_fk(bodies):
    from mdfm.body.smplh import fk
    body = bodies["male"]
    R, tr, betas = _random_pose(6, seed=6)
    b = betas[:1].expand(6, -1)
    p0, R0 = body(R[:, 0], R[:, 1:], tr, b)
    rest = body.rest_joints22(betas[0])
    assert tuple(rest.shape) == (22, 3)
    # one subject's rest (22, 3) broadcasts over the timestamps; betas is not read when rest is given
    p1, R1 = body(R[:, 0], R[:, 1:], tr, None, rest=rest)
    p2, R2 = fk(R, tr, rest.expand(6, 22, 3))
    assert (p0 - p1).abs().max() < 1e-12 and (p1 - p2).abs().max() < 1e-12
    assert (R0 - R1).abs().max() == 0 and (R1 - R2).abs().max() == 0
    # extra leading dimensions broadcast
    p3, _ = fk(R.reshape(2, 3, 22, 3, 3), tr.reshape(2, 3, 3), rest)
    assert (p3.reshape(6, 22, 3) - p1).abs().max() < 1e-12


@needs_ma
def test_state_dicts_hold_no_smplh_array(bodies, bank):
    for body in bodies.values():
        assert len(body.state_dict()) == 0
        names = dict(body.named_buffers())
        assert {"J_template", "J_dirs", "v_template", "shapedirs", "J_regressor"} <= set(names)
    assert len(bank.state_dict()) == 0
    assert {"J_template", "J_dirs", "parents"} <= set(dict(bank.named_buffers()))

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.skel = bank
            self.body = bodies["neutral"]
            self.head = torch.nn.Linear(4, 2)

    assert sorted(Holder().state_dict()) == ["head.bias", "head.weight"]


@needs_ma
def test_fk_from_the_bank_matches_moveatlas_canonical_amass():
    from mdfm.body.smplh import SkeletonBank, fk, gender_code
    from mdfm.data import canonical
    from mdfm.data.amass import load_sequence
    assert_train_take()
    seq = load_sequence(*TAKE)
    bank = SkeletonBank(F64)
    p = torch.as_tensor(seq["poses"], dtype=F64)
    R = axis_angle_to_matrix(p[:, :66].reshape(-1, 22, 3))
    tr = torch.as_tensor(seq["trans"], dtype=F64)
    betas = torch.as_tensor(seq["betas"], dtype=F64)[None]
    rest = bank.rest(betas, torch.tensor([gender_code(seq["gender"])]))[0]  # once for the subject
    with torch.no_grad():
        J, _ = fk(R, tr, rest)
    J = J @ Z_UP_TO_Y_UP.to(F64).T
    SEL = [0, 12, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    c = J[:, SEL] - J[:, :1]
    c = c / (c[:, 1] - c[:, 0]).norm(dim=-1).median()
    rec = canonical.load(canonical_row())
    assert not rec["lockbox"] and rec["tier"] == "RELEASE"
    ref = torch.as_tensor(np.asarray(rec["coords"]), dtype=F64)
    assert ref.shape == c.shape
    err = (c - ref).abs().max().item()
    assert err < 1e-4, f"max abs difference {err}"


# rest joints once per subject through trajectory, imu_render and force

def _standing_motion(n=40, seed=7):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 22, 3, generator=g, dtype=F64) * 0.2
    wig = 0.05 * torch.sin(torch.linspace(0, 6, n, dtype=F64))[:, None, None]
    R = axis_angle_to_matrix(base + wig * torch.randn(1, 22, 3, generator=g, dtype=F64))
    tr = torch.zeros(n, 3, dtype=F64)
    tr[:, 1] = 0.95
    return matrix_to_rot6d(R), tr


@needs_ma
def test_rest_is_computed_once_per_subject(bodies, bank, monkeypatch):
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.factory.imu_render import render
    from mdfm.kernels.force import com_kinematics, vertical_grf
    from mdfm.kernels.imu import imu
    body = bodies["female"]
    calls = {"basis": 0}
    basis = body.rest_joints22

    def counted(b):
        calls["basis"] += 1
        return basis(b)

    def vertex_path(b):
        raise AssertionError("the vertex path must not run per timestamp")

    monkeypatch.setattr(body, "rest_joints22", counted)
    monkeypatch.setattr(body, "rest_joints", vertex_path)
    r6, tr = _standing_motion()
    betas = torch.randn(16, generator=torch.Generator().manual_seed(8), dtype=F64)
    traj = BodyTrajectory(body, r6, tr, betas)
    t = torch.linspace(0.02, 0.35, 34, dtype=F64)
    sig, _, _ = render(traj, t)
    f, _ = vertical_grf(traj, t, mass=60.0, sex="female")
    com_kinematics(traj, t, "female")
    imu(traj, t, joint=4, offset=torch.tensor([0.0, -0.2, 0.05], dtype=F64))
    traj.pose_at(t)
    assert calls["basis"] == 1
    # rest given from the bank: the same renders
    rest = bank.rest(betas[None], torch.tensor([0]))[0]
    traj_b = BodyTrajectory(body, r6, tr, betas, rest=rest)
    sig_b, _, _ = render(traj_b, t)
    f_b, _ = vertical_grf(traj_b, t, mass=60.0, sex="female")
    assert calls["basis"] == 1
    assert (sig - sig_b).abs().max() < 1e-5 and (f - f_b).abs().max() < 1e-6
    # one subject only: a batched bank row or batched betas is refused, not broadcast
    with pytest.raises(ValueError):
        BodyTrajectory(body, r6, tr, betas, rest=rest[None])
    with pytest.raises(ValueError):
        BodyTrajectory(body, r6, tr, betas[None])
    # a float32 bank row is cast to the trajectory's dtype
    traj_32 = BodyTrajectory(body, r6, tr, None, rest=rest.float())
    assert traj_32.rest.dtype == F64 and (traj_32.rest - rest).abs().max() < 1e-6
    assert calls["basis"] == 1


@needs_ma
def test_backward_through_nested_jvp_of_the_body(bodies):
    """Gradients of IMU-type quantities (point acceleration and angular velocity, two nested jvp
    through Hermite, 6D, the shape basis and forward kinematics) against a central difference."""
    from mdfm.body.trajectory import BodyTrajectory
    body = bodies["neutral"]
    r6, tr = _standing_motion(n=6, seed=9)
    betas = torch.randn(16, generator=torch.Generator().manual_seed(10), dtype=F64)
    t = torch.tensor([0.013, 0.024, 0.037], dtype=F64)
    off = torch.tensor([0.02, -0.15, 0.05], dtype=F64)
    g = torch.Generator().manual_seed(11)
    wa, ww = torch.randn(3, 3, generator=g, dtype=F64), torch.randn(3, 3, generator=g, dtype=F64)

    def loss(k6, kt, b):
        _, _, a, _, w, _ = BodyTrajectory(body, k6, kt, b).point_kinematics(t, 5, off)
        return (a * wa).sum() + (w * ww).sum()

    k6, kt, b = (x.clone().requires_grad_(True) for x in (r6, tr, betas))
    loss(k6, kt, b).backward()
    # a second backward pass on a new evaluation of the same trainable betas: its own graph
    loss(k6, kt, b).backward()
    grads = [x.grad / 2 for x in (k6, kt, b)]
    assert all(torch.isfinite(gr).all() for gr in grads) and grads[2].abs().max() > 0
    d = [torch.randn(x.shape, generator=g, dtype=F64) for x in (r6, tr, betas)]
    h = 1e-6
    with torch.no_grad():
        plus = loss(*(x + h * dx for x, dx in zip((r6, tr, betas), d)))
        minus = loss(*(x - h * dx for x, dx in zip((r6, tr, betas), d)))
    fd = (plus - minus) / (2 * h)
    an = sum((gr * dx).sum() for gr, dx in zip(grads, d))
    assert abs(fd - an) < 1e-6 * max(1.0, abs(an.item()))


# GPU memory

@needs_ma
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_fk_of_65536_timestamps_with_backward_stays_under_half_a_gigabyte():
    """Forward kinematics with backward, no jvp, float32: 6D rotations to matrices, rest joints
    once per subject from the bank (betas trainable), then fk; peak allocation above what was
    allocated before the call."""
    from mdfm.body.smplh import SkeletonBank
    dev = torch.device("cuda")
    n = 65536
    bank = SkeletonBank(torch.float32).to(dev)
    g = torch.Generator().manual_seed(12)
    R = axis_angle_to_matrix(torch.randn(n, 22, 3, generator=g) * 0.4)
    r6 = matrix_to_rot6d(R).to(dev).requires_grad_(True)
    tr = torch.randn(n, 3, generator=g).to(dev).requires_grad_(True)
    betas = torch.randn(1, 16, generator=g).to(dev).requires_grad_(True)
    gender = torch.tensor([1], device=dev)
    torch.cuda.synchronize(dev)
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    rest = bank.rest(betas, gender)[0]
    pos, Rg = bank.fk(rot6d_to_matrix(r6), tr, rest)
    ((pos ** 2).sum() + Rg.sum()).backward()
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) - base
    assert torch.isfinite(r6.grad).all() and torch.isfinite(betas.grad).all() and betas.grad.abs().max() > 0
    assert peak < 0.5e9, f"peak {peak / 1e9:.3f} GB, {peak / n / 1e3:.2f} KB per timestamp"
