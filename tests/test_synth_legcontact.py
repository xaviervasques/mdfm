"""Tests of the leg-to-leg contact check of the synthetic patient, mdfm/synth/legcontact.py (shared change SH8):

- the leg skinning is the body model's one skinning call (viewer.knot_vertices) on the leg vertices, on the open
  body (the default since the port's wave 1) as on the reference;
- the calibrations read on the SMPL+H surface (the inner thigh band's width, the healthy walk's soft contacts, the
  distal band, the former walk's knees) run on the reference behind its flag; on the open body they were read again
  by their rules (open-body port, wave 3 follow-up, D36 to D38; scripts/openbody_legcontact_calibration.py,
  reports/openbody/legcontact_calibration.json): the band is 0 (the rest thighs stand apart up to the crotch), the
  healthy walk holds both tolerances with its inner thighs apart, the former healthy walk fails the check (by its
  feet on the open body);
- the reference's rest pose, its thighs touching below the crotch on the neutral mesh, reads a positive gap once the
  inner thigh band is left out, and a band half as wide reads them pressed together;
- the mesh gap agrees with a dense reference (barycentric samples against every nearby face) within ERR_TOL_M on
  knots of a healthy walk and of a walk whose knees pass into each other, near contact and in penetration; the
  fitted capsule model is recorded, not used;
- the sign: legs adducted step by step close their gap monotonically, cross zero and pass into each other; the
  validity entry fails beyond the tolerance and holds at rest;
- the classes: the soft zone is the thigh more than KNEE_ZONE_M above the knee (the foot never); the inner thighs of
  the healthy walk of gait.py press together (soft contacts) within SOFT_TOL_M; the distal band's reading (the thigh
  up to KNEE_ZONE_M + DISTAL_BAND_M above the knee counted hard) lies at or below the hard gap, in the record only;
- the calibration on normal gait (2026-11-03): the healthy walk of gait.py (its base STEP_WIDTH_M, the swing guard)
  holds both tolerances, while the former healthy walk (a 0.09 m base without the guard) passed its knees into each
  other beyond TOL_M (a hard contact at the knee), as the crouched narrow walk without the guard does.
Procedural bodies in the neutral shape only; no store, lockbox, EVAL, CODY or care_pd record is read.
"""
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


@pytest.fixture(scope="module")
def LC():
    torch.set_num_threads(2)
    from mdfm.synth import legcontact
    return legcontact


def _reference(monkeypatch):
    """The core on the SMPL+H reference, behind its flag (body.model_name); the caches keyed by the model name."""
    from mdfm.synth import openbody as OB
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")


def _healthy_walk():
    from mdfm.synth import gait
    return gait.generate(gait.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)


@pytest.fixture(scope="module")
def walk():
    from mdfm.synth import gait
    b, info = gait.generate(gait.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)
    return b, info


@pytest.fixture(scope="module")
def narrow():
    """A crouched walk on a narrow base without the guard: the knees pass into each other."""
    from mdfm.synth import gait
    p = replace(gait.spastic_diplegic(2), swing_abduction_deg=(0.0, 0.0), swing_rotation_deg=(0.0, 0.0),
                toe_out_deg=(2.0, 2.0), step_width_m=0.05, leg_gap_m=None)
    b, info = gait.generate(p, duration_s=6.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    return b, info


def test_the_leg_skinning_is_the_mesh(LC, walk):
    from mdfm.synth.viewer import knot_vertices
    b, _ = walk
    ks = [0, 150, 333, 500]
    V = knot_vertices(b, ks)
    m = LC.leg_mesh()
    sk = LC.skin_legs(b.local_R().numpy()[ks], b.transl.numpy()[ks], m)
    for side in LC.SIDES:
        assert np.abs(sk[side] - V[:, m[side]["ids"]]).max() < 1e-9


def test_the_open_rest_legs_stand_apart(LC):
    """On the open body the rest legs stand apart with no band (the band's rule gives 0: the narrowest band whose rest
    gap is positive, D36) and the dense reference agrees."""
    assert LC.crotch_band_m() == LC.CROTCH_BAND_M == 0.0
    m = LC.leg_mesh()
    assert m["crotch_band_m"] == 0.0
    r = LC.rest_gap()
    assert 0.0 < r["gap_m"] < 0.2
    R = np.repeat(np.eye(3)[None, None], 22, 1)
    assert abs(LC.reference_gap(R, np.zeros((1, 3)), radius_m=0.08)[0] - r["gap_m"]) < LC.ERR_TOL_M
    for side in LC.SIDES:
        assert len(m[side]["ids"]) > 500 and m[side]["bnd"].any()
        assert m[side]["soft_point"].any() and not m[side]["soft_point"].all()


def test_the_open_healthy_walk_holds_both_tolerances(LC, walk):
    """The calibration read again on the open body (D37): the healthy walk of gait.py keeps its inner thighs apart
    (no soft contact: the SOFT_TOL_M rule holds with room) and no knee, shank or foot passes into the other leg
    beyond TOL_M."""
    b, _ = walk
    g = LC.leg_gap(b)
    v = LC.validity(b, g)
    assert v["ok"] and v["hard_min_m"] > -LC.TOL_M and v["soft_min_m"] > 0.0, v
    # the basis of the calibration still holds on the open body: the former healthy walk (the 0.09 m base without the
    # guard nor the turn's widening) fails the check, by its feet here (at the knee on SMPL+H)
    from mdfm.synth import gait
    p = replace(gait.NORMAL, step_width_m=0.09, leg_gap_m=None, turn_base_m=0.0)
    f, _ = gait.generate(p, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)
    vf = LC.validity(f)
    assert not vf["ok"] and vf["hard_min_m"] < -LC.TOL_M and vf["hard_knots_beyond_tol"] > 0, vf


def test_the_rest_pose_reads_its_gap(LC, monkeypatch):
    _reference(monkeypatch)
    assert LC.crotch_band_m() == LC.CROTCH_BAND_REFERENCE_M
    m = LC.leg_mesh()
    r = LC.rest_gap()
    # the band leaves the touching inner thighs out; past it the rest legs stand apart (the band about the narrowest
    # that does: a band half as wide reads the rest thighs pressed together)
    assert 0.0 < r["gap_m"] < 0.2
    b0 = LC.CROTCH_BAND_REFERENCE_M
    try:
        LC.CROTCH_BAND_REFERENCE_M = 0.5 * b0
        LC._mesh.cache_clear()
        assert LC.rest_gap()["gap_m"] < 0.0
    finally:
        LC.CROTCH_BAND_REFERENCE_M = b0
        LC._mesh.cache_clear()
    R = np.repeat(np.eye(3)[None, None], 22, 1)
    ref = LC.reference_gap(R, np.zeros((1, 3)), radius_m=0.08)
    assert abs(ref[0] - r["gap_m"]) < LC.ERR_TOL_M
    for side in LC.SIDES:
        assert len(m[side]["ids"]) > 500 and m[side]["bnd"].any()
        assert m[side]["soft_point"].any() and not m[side]["soft_point"].all()


def test_the_mesh_gap_agrees_with_the_dense_reference(LC, walk, narrow):
    for b, _ in (walk, narrow):
        g = LC.leg_gap(b)
        h = g["gap_m"]
        # the deepest knot, a knot near contact and a knot a centimetre apart (when the walk has them)
        ks = sorted({int(np.argmin(h)), int(np.argmin(np.abs(h))), int(np.argmin(np.abs(h - 0.01)))})
        e = LC.mesh_error(b.local_R().numpy()[ks], b.transl.numpy()[ks])
        assert e["n_in_range"] >= 2
        assert e["max_error_m"] < LC.ERR_TOL_M, e
        # the fitted capsules are far coarser than the mesh (the reason the mesh is measured)
        assert e["capsule_max_error_m"] > 5 * LC.ERR_TOL_M


def test_adducted_legs_close_their_gap_and_pass_into_each_other(LC):
    from mdfm.synth.gait import _rot_axis
    deg = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    R = np.repeat(np.eye(3)[None, None], 22, 1).repeat(len(deg), 0)
    R[:, 1] = _rot_axis("z", -deg)                       # the left hip adducted (toward -x)
    R[:, 2] = _rot_axis("z", deg)                        # the right hip adducted (toward +x)
    g = LC.gap_from_pose(R, np.zeros((len(deg), 3)), cull_m=0.2)
    gap = g["gap_m"]
    assert np.all(np.diff(gap) < 0), gap
    assert gap[0] > 0.0 and gap[-1] < -LC.TOL_M
    assert g["hard_m"][-1] < -LC.TOL_M                   # the knees pass into each other


def test_the_validity_entry(LC, narrow):
    from mdfm.synth.body import Body
    b, _ = narrow
    v = LC.validity(b)
    assert not v["ok"] and v["hard_min_m"] < -LC.TOL_M and v["hard_knots_beyond_tol"] > 0
    assert v["hard_where"] is not None
    R = torch.eye(3, dtype=torch.float64).expand(3, 22, 3, 3)
    from mdfm.body.rotations import matrix_to_rot6d
    rest = Body(matrix_to_rot6d(R), torch.zeros(3, 3), torch.zeros(3, 2, dtype=torch.bool))
    vr = LC.validity(rest)
    assert vr["ok"] and vr["hard_min_m"] >= LC.CULL_M - 1e-12          # beyond the measured range


def test_the_healthy_walk_presses_the_inner_thighs_within_the_soft_tolerance(LC, monkeypatch):
    """The neutral body's thighs touch at rest below the crotch; in the healthy walk of gait.py they press together
    (soft contacts) within SOFT_TOL_M, and no knee, shank or foot passes into the other leg beyond TOL_M (the SMPL+H
    surface, on the reference)."""
    _reference(monkeypatch)
    b, _ = _healthy_walk()
    g = LC.leg_gap(b)
    assert g["soft_m"].min() < 0.0 and g["soft_m"].min() > -LC.SOFT_TOL_M
    assert LC.report(g)["soft_knots_beyond_tol"] == 0
    v = LC.validity(b, g)
    assert v["ok"] and v["hard_min_m"] > -LC.TOL_M, v


def test_the_soft_zone_is_the_thigh_above_the_knee(LC):
    m = LC.leg_mesh()
    from mdfm.synth.viewer import GENDERS, dominant_joint
    dom = dominant_joint(GENDERS[2])
    for side in LC.SIDES:
        s = m[side]
        a, f = LC.LEG[side][2:]
        foot = np.isin(dom[s["ids"]], (a, f))
        assert np.all(s["along_v"][foot] == LC.NO_ALONG)
        assert np.array_equal(s["soft_point"], s["along_point"] > LC.KNEE_ZONE_M)
        soft_v = s["along_v"] > LC.KNEE_ZONE_M
        # every soft vertex lies on the thigh (the hip's or the knee's, above the knee joint), most are the hip's
        assert np.all(np.isin(dom[s["ids"]][soft_v], LC.LEG[side][:2]))
        assert (dom[s["ids"]][soft_v] == LC.LEG[side][0]).mean() > 0.9
        assert s["soft_point"].any() and not s["soft_point"].all()


def test_the_distal_band_reading_counts_the_thigh_above_the_knee_as_hard(LC, monkeypatch):
    """distal_gap: the soft zone's edge moved to KNEE_ZONE_M + DISTAL_BAND_M; its soft class lies inside the
    default one, its hard gap at or below the default hard gap on every knot (the distal band's contacts added to the
    knee, shank and foot), and the healthy walk's inner thighs give some of their contacts to it; the reading sits in
    the validity entry beside 'ok', which it does not change (the SMPL+H surface, on the reference)."""
    _reference(monkeypatch)
    b, _ = _healthy_walk()
    m0, m1 = LC.leg_mesh(), LC.leg_mesh(knee_zone_m=LC.KNEE_ZONE_M + LC.DISTAL_BAND_M)
    for side in LC.SIDES:
        s0, s1 = m0[side]["soft_point"], m1[side]["soft_point"]
        assert not np.any(s1 & ~s0) and np.any(s0 & ~s1)
        assert m0[side]["knee_zone_m"] == LC.KNEE_ZONE_M
    g = LC.leg_gap(b)
    d = LC.distal_gap(b, g)
    assert np.all(d["hard_m"] <= g["hard_m"] + 1e-12)
    assert d["hard_m"].min() < g["hard_m"].min()                       # the walk presses that band together
    v = LC.validity(b, g)
    r = LC.report(g, b.rate)
    assert {k: v[k] for k in r} == r and v["ok"] == (r["hard_knots_beyond_tol"] == 0 == r["soft_knots_beyond_tol"])
    assert v["distal_min_m"] == float(d["hard_m"].min()) and v["distal_band_m"][1] > v["distal_band_m"][0]


def test_the_former_healthy_walk_passed_its_knees_into_each_other(LC, monkeypatch):
    """The basis of the calibration: the healthy walk of gait.py before 2026-11-03 (the 0.09 m base of pirker2017
    without the swing guard) passed a knee into the other leg beyond TOL_M in straight walking (no turn here; the
    SMPL+H surface, on the reference)."""
    _reference(monkeypatch)
    from mdfm.synth import gait
    p = replace(gait.NORMAL, step_width_m=0.09, leg_gap_m=None, turn_base_m=0.0)
    b, info = gait.generate(p, duration_s=7.0, start_at_s=1.0, stop_at_s=5.0, seed=7)
    g = LC.leg_gap(b)
    v = LC.validity(b, g)
    assert not v["ok"] and v["hard_min_m"] < -LC.TOL_M and v["hard_knots_beyond_tol"] > 0, v
    assert "thigh_lower" in v["hard_where"] or "shank_upper" in v["hard_where"]          # at the knee
