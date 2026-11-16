"""Unit tests of the body core: rotations, Hermite interpolation, SMPL+H forward kinematics.

The forward kinematics is checked against the joints MoveAtlas computed for AMASS with
human_body_prior (build_canonical_amass.py: z-up to y-up, 22 body joints mapped to the canonical
15, root centred, divided by the median torso length), read through mdfm.data.canonical.load for
the KIT take of tests/amass_fixture.py (a train archive). The test needs the MoveAtlas tree
(read-only).
"""
import math
import os

import numpy as np
import pytest
import torch

from amass_fixture import TAKE, assert_train_take, canonical_row
from mdfm import paths
from mdfm.body.hermite import hermite
from mdfm.body.rotations import (Z_UP_TO_Y_UP, axis_angle_to_matrix, matrix_to_axis_angle,
                                 matrix_to_rot6d, rot6d_to_matrix)

torch.manual_seed(0)
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def test_rotation_round_trips():
    aa = torch.randn(500, 3) * 1.5
    aa = aa * (torch.rand(500, 1) * math.pi / aa.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
    R = axis_angle_to_matrix(aa)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(500), atol=1e-5)
    assert torch.allclose(axis_angle_to_matrix(matrix_to_axis_angle(R)), R, atol=1e-4)
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(R)), R, atol=1e-5)
    assert torch.allclose(axis_angle_to_matrix(torch.zeros(1, 3)), torch.eye(3)[None], atol=1e-7)


def test_hermite_interpolates_and_differentiates():
    rate = 100.0
    ts = torch.arange(0, 2, 1 / rate, dtype=torch.float64)
    y = torch.stack([torch.sin(2 * math.pi * 1.3 * ts), torch.cos(2 * math.pi * 0.7 * ts)], -1)
    assert torch.allclose(hermite(y, ts, rate), y, atol=1e-12)
    tq = torch.linspace(0.3, 1.7, 57, dtype=torch.float64)
    dy = torch.func.jvp(lambda t: hermite(y, t, rate), (tq,), (torch.ones_like(tq),))[1]
    true = torch.stack([2 * math.pi * 1.3 * torch.cos(2 * math.pi * 1.3 * tq),
                        -2 * math.pi * 0.7 * torch.sin(2 * math.pi * 0.7 * tq)], -1)
    assert (dy - true).abs().max() < 0.02 * true.abs().max()


def test_fixture_take_is_in_a_train_archive():
    assert_train_take()


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_fk_matches_moveatlas_canonical_amass():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.data import canonical
    from mdfm.data.amass import load_sequence
    # a rigid record: some MoveAtlas AMASS records are corrupted, none of KIT
    # (scripts/probe_amass_canonical_rigidity.py)
    seq = load_sequence(*TAKE)
    g = seq["gender"] if seq["gender"] in ("male", "female", "neutral") else "neutral"
    body = SMPLHBody(g, dtype=torch.float64)
    p = torch.as_tensor(seq["poses"], dtype=torch.float64)
    R = axis_angle_to_matrix(p[:, :66].reshape(-1, 22, 3))
    tr = torch.as_tensor(seq["trans"], dtype=torch.float64)
    b = torch.as_tensor(seq["betas"], dtype=torch.float64).expand(p.shape[0], -1)
    with torch.no_grad():
        J, _ = body(R[:, 0], R[:, 1:], tr, b)
    J = J @ Z_UP_TO_Y_UP.to(J.dtype).T
    SEL = [0, 12, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    c = J[:, SEL] - J[:, :1]
    torso = (c[:, 1] - c[:, 0]).norm(dim=-1)
    c = c / torso.median()
    rec = canonical.load(canonical_row())
    assert not rec["lockbox"] and rec["tier"] == "RELEASE"
    ref = torch.as_tensor(np.asarray(rec["coords"]), dtype=torch.float64)
    assert ref.shape == c.shape
    err = (c - ref).abs().max().item()
    assert err < 1e-4, f"max abs difference {err}"


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_vertices_agree_with_joint_fk():
    from mdfm.body.smplh import SMPLHBody
    body = SMPLHBody("neutral", dtype=torch.float64)
    aa = torch.randn(3, 22, 3, dtype=torch.float64) * 0.3
    R = axis_angle_to_matrix(aa)
    tr = torch.randn(3, 3, dtype=torch.float64)
    b = torch.randn(3, 16, dtype=torch.float64)
    J, _ = body(R[:, 0], R[:, 1:], tr, b)
    V = body.vertices(R[:, 0], R[:, 1:], tr, b)
    # rest pose: skinned vertices equal the shaped template, joints equal the regressed ones
    I = torch.eye(3, dtype=torch.float64).expand(3, 22, 3, 3)
    V0 = body.vertices(I[:, 0], I[:, 1:], torch.zeros(3, 3, dtype=torch.float64), b)
    v_shaped = body.v_template + torch.einsum("vcs,bs->bvc", body.shapedirs, b)
    assert torch.allclose(V0, v_shaped, atol=1e-9)
    # posed: the pelvis joint regressed from posed vertices stays near the FK pelvis
    assert V.shape == (3, 6890, 3)
    assert ((V.mean(1) - J[:, 0]).norm(dim=-1) < 0.5).all()


def test_checkpoint_never_holds_smplh_arrays():
    if not HAVE_MA:
        pytest.skip("MoveAtlas SMPL+H model not found")
    from mdfm.body.smplh import SMPLHBody
    assert len(SMPLHBody("neutral").state_dict()) == 0


@pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
def test_y_up_conversion_is_an_exact_world_rotation():
    from mdfm.body.smplh import SMPLHBody
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    seq = load_sequence(*TAKE)
    body = SMPLHBody(seq["gender"], dtype=torch.float64)
    fps = seq["fps"]
    r6, tr, betas = amass_to_body_clock(seq, rate=fps, cutoff=None, body=body)
    R = rot6d_to_matrix(r6)
    J_y, _ = body(R[:, 0], R[:, 1:], tr, betas.expand(R.shape[0], -1))
    p = torch.as_tensor(seq["poses"][:, :66], dtype=torch.float64).reshape(-1, 22, 3)
    Rz = axis_angle_to_matrix(p)
    J_z, _ = body(Rz[:, 0], Rz[:, 1:], torch.as_tensor(seq["trans"], dtype=torch.float64),
                  betas.expand(p.shape[0], -1))
    n = min(J_y.shape[0], J_z.shape[0])
    assert (J_y[:n] - J_z[:n] @ Z_UP_TO_Y_UP.to(torch.float64).T).abs().max() < 1e-6  # 6D round trip and knot evaluation
