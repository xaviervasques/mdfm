"""Tests of the SH2 extension of the joint-limit table, mdfm/synth/limits_ext.py, and of the dystonia operator that
reads it, mdfm/synth/dystonia.py (docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md SH2, owner's go of 2026-11-02):

- the extension widens only its listed bounds (the ankle's inversion to the 30 deg and the toes' flexion to the
  35 deg of jarm2022), each with a source and the number read, never narrows a bound, and leaves the healthy table
  (body.joint_limits) untouched; since the open-body port's wave 1 the healthy table is the anatomical one, which
  already holds those two bounds (the widening opens nothing beyond it);
- the shoulder's adduction bound opens only under a collar raised over the base, by the collar's rise at the same
  knot and never more (the coupling): an adducted shoulder under a resting collar is a violation in both tables, the
  same shoulder under a collar raised by as much is a violation of the healthy table only; a collar the base itself
  holds raised opens nothing (the arising's folded arms: a proximal arm posture on that base is the same under both
  tables); the healthy collar bound raises the shoulder joint by about 5 cm (the docstring's number);
- the dystonia operator's shoulder elevation keeps the arm hanging (the humerus keeps the level 0's direction, the
  shoulder joint rises), inside the dystonia table, its level the TWSTRS third of the collar's room, on both sides;
  the arm 'ride' of the operator before SH2 turns the humerus with the collar;
- every other region of the operator gives the same body under both tables on a base whose collars rest; the truth
  records its table and the extension's sources; the healthy table's validity is body.validity unchanged;
- the table 'shrug' (task sh1c, the session decision of 2026-11-03 under the owner's delegation): the healthy table
  plus the coupling and nothing else (the ankle and toe bounds healthy, the shoulder's adduction opened by the
  collar's rise), its record names the decision and no widened bound, its check reports the coupling; a healthy
  shrug with the arm hanging is inside it and outside the healthy table.
Every body is procedural in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read.
"""
import os

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)


@pytest.fixture(scope="module")
def LX():
    from mdfm.synth import limits_ext
    return limits_ext


@pytest.fixture(scope="module")
def D():
    from mdfm.synth import dystonia
    return dystonia


@pytest.fixture(scope="module")
def stand():
    from mdfm.synth.base import static_body
    return static_body("stand", 3.0)


@pytest.fixture(scope="module")
def sit():
    from mdfm.synth.base import static_body
    return static_body("sit", 3.0)


def _humerus_turn(base, out, s, e):
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    ub, uo = Jb[:, e] - Jb[:, s], Jo[:, e] - Jo[:, s]
    # the angle by atan2 (open-body port, wave 4): arccos of a cosine at 1 reads float64 noise as 1.2e-6 deg on the
    # open body's arm, past the 1e-6 the hanging-arm assertions ask
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(ub, uo), axis=-1), (ub * uo).sum(-1)))


# ---------------------------------------------------------------- the table
def test_the_extension_widens_only_its_listed_bounds_each_from_a_read_source(LX):
    from mdfm.synth.body import joint_limits
    h = joint_limits()
    lo0, hi0 = h.lo.copy(), h.hi.copy()
    x = LX.dystonia_limits()
    listed = {(j, c, side) for j, c, side, *_ in LX.WIDEN}
    for j in range(22):
        for c in range(3):
            assert x.lo[j, c] <= h.lo[j, c] and x.hi[j, c] >= h.hi[j, c]            # never narrows
            if (j, c, "lo") not in listed:
                assert x.lo[j, c] == h.lo[j, c]
            if (j, c, "hi") not in listed:
                assert x.hi[j, c] == h.hi[j, c]
    inv, toe = LX.INVERSION_DEG, LX.TOE_FLEXION_DEG
    assert (inv, toe) == (30.0, 35.0)                                                # jarm2022, as read
    assert x.lo[7, 2] == -inv and x.hi[8, 2] == inv and x.hi[10, 0] == toe and x.hi[11, 0] == toe
    # the anatomical table (wave 1) already holds jarm2022's bounds: the widening opens nothing beyond it
    assert h.lo[7, 2] == -inv and h.hi[8, 2] == inv and h.hi[10, 0] == toe and h.hi[11, 0] == toe
    for *_, src, read in LX.WIDEN:
        assert src in LX.SOURCES and read
    assert set(LX.CORROBORATE) | set(LX.COLLAR_SOURCES) <= set(LX.SOURCES)
    rec = LX.record("dystonia")
    assert {w["source"] for w in rec["widened"]} <= set(rec["sources"]) <= set(LX.SOURCES)
    np.testing.assert_array_equal(joint_limits().lo, lo0)                            # the healthy table untouched
    np.testing.assert_array_equal(joint_limits().hi, hi0)
    assert LX.record("healthy")["table"] == "healthy"


def test_the_shoulder_adduction_opens_only_by_the_collar_elevation(LX):
    from mdfm.synth.body import joint_limits
    h = joint_limits()
    for e in (0.0, 10.0, LX.collar_cap("left")):
        x = LX.dystonia_limits((e, e))
        assert x.lo[16, 2] == pytest.approx(h.lo[16, 2] - e) and x.hi[17, 2] == pytest.approx(h.hi[17, 2] + e)
        assert x.hi[16, 2] == h.hi[16, 2] and x.lo[17, 2] == h.lo[17, 2]             # abduction untouched
        assert x.lo[13, 2] == h.lo[13, 2] and x.hi[13, 2] == h.hi[13, 2]             # the collar's own bound kept
    big = LX.dystonia_limits((1000.0, 1000.0))
    assert big.lo[16, 2] == pytest.approx(h.lo[16, 2] - LX.collar_cap("left"))      # capped by the collar's bound
    assert big.hi[17, 2] == pytest.approx(h.hi[17, 2] + LX.collar_cap("right"))


def test_the_coupling_reads_the_collar_rise_over_the_base(LX, stand):
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    from mdfm.synth.body import components_deg, joint_limits
    h = joint_limits()
    T = 3
    R0 = stand.local_R()[:T].clone()
    R0[:, 13] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(30.0)], dtype=torch.float64))  # base raised
    R = R0.clone()
    for k, (collar, shoulder) in enumerate(((30.0, 0.0), (30.0, 5.0), (40.0, 5.0))):
        R[k, 13] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(collar)], dtype=torch.float64))
        R[k, 16] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(h.lo[16, 2] - shoulder)],
                                                     dtype=torch.float64))
    c, cb = components_deg(matrix_to_rot6d(R)), components_deg(matrix_to_rot6d(R0))
    np.testing.assert_allclose(LX.collar_rise(c, cb)[:, 0], [0.0, 0.0, 10.0], atol=1e-6)
    ex = LX.excess("dystonia", c, cb, rows=[16])[:, 0, 2]
    assert ex[1] > 4.9 and ex[2] <= 0.0                       # the base's 30 deg opens nothing, its rise of 10 does
    assert LX.excess("dystonia", c, rows=[16])[1, 0, 2] <= 0.0   # without a base: the rise over rest
    rep = LX.coupling_report(_body_with(stand, R0), _body_with(stand, R))
    assert not rep["ok"] and rep["left"]["collar_rise_deg_max"] == pytest.approx(10.0, abs=1e-6)


def _body_with(body, R):
    from dataclasses import replace
    from mdfm.body.rotations import matrix_to_rot6d
    T = R.shape[0]
    return replace(body, rot6d=matrix_to_rot6d(R), transl=body.transl[:T], contact=body.contact[:T])


def test_a_raised_collar_of_the_base_opens_nothing(D, stand):
    """A level 0 that holds its collars raised (at the collar's elevation bound on both sides, set on the stand; the
    folded arms of motions.arise held them above 20 deg until SH9 refolded them lower): the proximal arm's adduction is
    the same body under both tables there (review M3 of the SH2 wave: the absolute elevation opened up to 40 deg).
    Open-body port, wave 4 (D26): the base raised its collars 30 deg past the AMASS table's 40 deg-wide room; on the
    anatomical table the collar's elevation bound is jarm2022's 20 deg, collar_rise clips at it, so the base holds the
    bound itself."""
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.synth.limits_ext import collar_cap, collar_rise
    from mdfm.synth.body import components_deg
    R = stand.local_R().clone()
    for j, s, side in ((13, 1.0, "left"), (14, -1.0, "right")):                    # +z left, -z right (COUPLED)
        R[:, j] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(collar_cap(side) * s)],
                                                    dtype=torch.float64))
    b = _body_with(stand, R)
    rise = collar_rise(components_deg(b.rot6d))
    assert rise.min() > 15.0 and rise.max() <= collar_cap("left") + 1e-6           # the base's collars are raised
    a, ta = D.dystonic_posture(b, "proximal_arm", "adduction", "left", amplitude_grade=3, seed=1)
    h, th = D.dystonic_posture(b, "proximal_arm", "adduction", "left", amplitude_grade=3, seed=1, table="healthy")
    assert torch.equal(a.rot6d, h.rot6d) and ta["params"]["possible_deg"] == th["params"]["possible_deg"]
    assert ta["validity"]["limits_new"] == 0 and ta["validity"]["coupling"]["ok"]


def test_the_healthy_collar_bound_raises_the_shoulder_about_5_cm(LX, D, stand):
    for side in ("left", "right"):
        out, tr = D.dystonic_posture(stand, "shoulder_elevation", "elevation", side, level=LX.collar_cap(side) - 1e-6,
                                     seed=1)
        assert 5.0 < tr["params"]["shoulder_rise_cm_max"] < 6.0 and tr["validity"]["limits_new"] == 0


def test_the_coupling_is_read_knot_by_knot(LX, stand):
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    from mdfm.synth.body import joint_limits
    T = 4
    lo = float(joint_limits().lo[16, 2])
    R = stand.local_R()[:T].clone()
    # the shoulder read from the healthy adduction bound (open-body port, wave 4, D26: the knots stood at -80 deg less
    # the turn, the SMPL+H stand's hanging arm, with the AMASS bound at -80.99; the anatomical bound is -90): inside
    # by 10, past it by 10, past it by 10 under a collar risen 20, past it by 25 under the same collar
    for k, (collar, past) in enumerate(((0.0, -10.0), (0.0, 10.0), (20.0, 10.0), (20.0, 25.0))):
        R[k, 13] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(collar)], dtype=torch.float64))
        R[k, 16] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(lo - past)], dtype=torch.float64))
    r6 = matrix_to_rot6d(R)
    from mdfm.synth.body import components_deg
    ex_d = LX.excess("dystonia", components_deg(r6), rows=[16])[:, 0, 2]
    ex_h = LX.excess("healthy", components_deg(r6), rows=[16])[:, 0, 2]
    assert ex_d[0] <= 0 and ex_d[1] > 0 and ex_d[2] <= 0 and ex_d[3] > 0          # opened by the collar, no more
    assert (ex_h[1:] > 0).all()


# ---------------------------------------------------------------- the dystonia operator
@pytest.mark.parametrize("side", ["left", "right"])
def test_shoulder_elevation_raises_the_shoulder_with_the_arm_hanging(D, stand, side):
    from mdfm.synth import grades as G
    s, e = (16, 18) if side == "left" else (17, 19)
    rise = []
    for g in (1, 2, 3):
        out, tr = D.dystonic_posture(stand, "shoulder_elevation", "elevation", side, amplitude_grade=g, seed=g)
        p, v = tr["params"], tr["validity"]
        assert p["arm"] == "hang" and p["limits_table"]["table"] == "dystonia" and "limits_ext" in tr["sources"]
        want = G.twstrs_level_deg("shoulder_elevation", g, p["possible_deg"])
        assert p["reachable"] and p["level_deg"] == pytest.approx(want)
        assert np.abs(tr["arrays"]["deviation_deg"] - p["applied_deg"]).max() < 1e-6   # the collar holds the level
        assert _humerus_turn(stand, out, s, e).max() < 1e-6                           # the arm hangs as at level 0
        assert v["limits_new"] == 0 and v["coupling"]["ok"] and p["limit_scaled_share"] == 0.0
        rise.append(p["shoulder_rise_cm_max"])
    assert 0.5 < rise[0] < rise[1] < rise[2]                                          # the shoulder rises (cm)


def test_the_arm_rides_the_collar_in_the_operator_before_sh2(D, stand):
    out, tr = D.dystonic_posture(stand, "shoulder_elevation", "elevation", "left", amplitude_grade=2, seed=2,
                                 arm="ride", table="healthy")
    p = tr["params"]
    assert p["limits_table"]["table"] == "healthy" and "limits_ext" not in tr["sources"]
    assert abs(_humerus_turn(stand, out, 16, 18).max() - p["applied_deg"]) < 1.0     # the humerus turns with it
    assert tr["validity"]["limits_new"] == 0 and "coupling" not in tr["validity"]
    with pytest.raises(ValueError):
        D.dystonic_posture(stand, "neck", "rotation_left", amplitude_grade=1, arm="hang")


@pytest.mark.parametrize("region,direction,base,kw", [("neck", "rotation_left", "stand", {}),
                                                      ("trunk", "lateral_left", "stand", {}),
                                                      ("proximal_arm", "abduction", "stand", {}),
                                                      ("distal_arm", "elbow_flexion", "stand", {}),
                                                      ("proximal_leg", "abduction", "stand", {}),
                                                      ("distal_leg", "plantarflexion_ankle", "sit", {})])
def test_other_regions_are_the_same_under_both_tables(D, stand, sit, region, direction, base, kw):
    body = stand if base == "stand" else sit
    a, ta = D.dystonic_posture(body, region, direction, "left", amplitude_grade=2, seed=1, **kw)
    b, tb = D.dystonic_posture(body, region, direction, "left", amplitude_grade=2, seed=1, table="healthy", **kw)
    assert torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)
    assert ta["params"]["possible_deg"] == tb["params"]["possible_deg"]
    assert ta["validity"]["limits_new"] == tb["validity"]["limits_new"] == 0


def test_the_healthy_check_is_body_validity(LX, D, stand):
    from mdfm.synth.body import validity
    out, _ = D.dystonic_posture(stand, "neck", "tilt_left", amplitude_grade=1, seed=1)
    assert LX.check("healthy", stand, out) == validity(stand, out)
    v = LX.check("dystonia", stand, out)
    assert v["limits_table"] == "dystonia" and v["limits_new"] == 0 and v["coupling"]["ok"]
    with pytest.raises(ValueError):
        LX.check("other", stand, out)


# ---------------------------------------------------------------- the healthy shrug (task sh1c)
def test_the_shrug_table_is_the_healthy_table_with_the_coupling_only(LX):
    from mdfm.synth.body import joint_limits
    h = joint_limits()
    assert LX.SHRUG in LX.TABLES and LX.SHRUG in LX.COUPLED_TABLES and LX.HEALTHY not in LX.COUPLED_TABLES
    x0 = LX.coupled_limits(LX.SHRUG)
    np.testing.assert_array_equal(x0.lo, h.lo)                                        # no rise: the healthy table
    np.testing.assert_array_equal(x0.hi, h.hi)
    for e in (5.0, 20.0):
        x = LX.coupled_limits(LX.SHRUG, (e, e))
        assert x.lo[16, 2] == pytest.approx(h.lo[16, 2] - e) and x.hi[17, 2] == pytest.approx(h.hi[17, 2] + e)
        m = np.ones((22, 3), bool)
        m[16, 2] = m[17, 2] = False
        np.testing.assert_array_equal(x.lo[m], h.lo[m])                            # nothing else: no WIDEN
        np.testing.assert_array_equal(x.hi[m], h.hi[m])
    assert x.lo[7, 2] == h.lo[7, 2] and x.hi[10, 0] == h.hi[10, 0]                  # the ankle and the toes healthy
    rec = LX.record(LX.SHRUG)
    assert rec["table"] == "shrug" and rec["widened"] == [] and rec["sources"] == []
    assert "2026-11-03" in rec["decision"] and "decisions.md" in rec["decision"] and len(rec["coupled"]) == 2
    assert LX.record("dystonia")["table"] == "dystonia" and len(LX.record("dystonia")["widened"]) == len(LX.WIDEN)
    with pytest.raises(ValueError):
        LX.coupled_limits(LX.HEALTHY)


def test_a_healthy_shrug_with_the_arm_hanging_is_inside_the_shrug_table_only(LX, stand):
    """The collar raised by 15 deg over the stand, the shoulder turned back by as much (the arm hangs as before):
    outside the healthy table (its adduction bound lies 10 deg past the stand's hanging arm, at 80 deg), inside the
    table 'shrug', the coupling holding; the same shoulder under a resting collar is outside both. Open-body port,
    wave 4 (D26): the rise was 10 deg, past the AMASS bound's 0.99 deg of room; on the anatomical table 10 deg reaches
    the bound exactly, so the rise is 15."""
    from mdfm.body.rotations import axis_angle_to_matrix
    R0 = stand.local_R()
    R = R0.clone()
    E = axis_angle_to_matrix(torch.tensor([0.0, 0.0, np.radians(15.0)], dtype=torch.float64))
    R[:, 13] = E @ R0[:, 13]
    R[:, 16] = R[:, 13].transpose(-1, -2) @ R0[:, 13] @ R0[:, 16]
    out = _body_with(stand, R)
    assert _humerus_turn(stand, out, 16, 18).max() < 1e-6                           # the arm hangs as at level 0
    assert LX.check("healthy", stand, out)["limits_new"] > 0
    v = LX.check(LX.SHRUG, stand, out)
    assert v["limits_table"] == "shrug" and v["limits_new"] == 0 and v["coupling"]["ok"]
    assert v["coupling"]["left"]["collar_rise_deg_max"] == pytest.approx(15.0, abs=1e-6)
    R2 = R0.clone()
    R2[:, 16] = R[:, 16]
    assert LX.check(LX.SHRUG, stand, _body_with(stand, R2))["limits_new"] > 0      # no rise: nothing opens
