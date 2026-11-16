"""Tests of the physiological background of the synthetic patient, mdfm/synth/physio.py (shared change SH1, layer
physio-v5), as catalogue.catalogue_set lays it on the symptom catalogue:

- the layer is a function of time on a fixed horizon: a short clip takes the prefix of a long one bit for bit, the
  draws do not depend on the posture kind, the kinds carry their components only, the seated lean only leans forward;
  the finger lift is one-sided, smooth and without a constant offset;
- the local fade holds the clip around a failing knot and ramps back slowly; the anchoring weights of the hands sum to
  at most 1, ramp without a jump and cross-fade between two supports; the inside ramp never leaves its mask;
- the amplitudes follow their sources: the breathing rate and inspiratory share (Motamedi-Fakhr et al. 2017), the
  standing sway RMS (Foulger et al. 2025), the seated trunk range at the neck joint over 90 s (Melo et al. 2009);
- the transfer on procedural bodies (and, for the fingers, on the seated rest-tremor level 0): a member's rows held
  equal to its level 0's are the level 0's output rows bit for bit (body rows, pelvis translation, hand rows), the
  joints whose rows are all held keep the level 0's positions, a member's own rows (body and fingers) take the level
  0's delta, a member of a member (a level-0 chain) keeps its direct level 0's rows, a member longer than its level 0
  takes its own solve beyond it without a jump (its fingers keep their layer), and the own corrections never touch a
  held row; the anchor reading of a member (a moving hand reported apart);
- physio-v5 on procedural bodies: the rest weight of a member's own anchoring (none while the wrist moves faster than
  ARM_REST_M_S, a ramp without a jump), the vertical goal of a leg whose hip or knee row is held (the planted foot
  keeps its horizontal place, its height takes the residual), the sway scaled on a member's own pelvis and leg rows
  where its own hip at the table's bound would leave it, the limit guard reading the dystonia table at each knot (the
  shoulder's adduction widened by the collar's rise), a member's own finger rows taking the level 0's finger delta
  (physio-v4, kept) and, held still in a posture far from the level 0's or with FINGER_RULE 'all', no finger drift, a
  series root's anchoring gated where its hand is flung off its support, an arm turned off a scene box on own rows;
- one series, one layer on the catalogue: a clip and its direct level 0 (chains included) carry the series' seed,
  kind and layer digest, and every row, hand row and translation the two hold equal is equal bit for bit after the
  layer (joint positions held equal within 1e-9 m), over every family and over the pairs the physio-v2 layer broke
  (athetosis, chorea of the arms, hemiballism, kinetic, essential and intention tremors at the target, the multifocal
  dystonia, the tardive limbs, the arm myoclonus posture);
- the layer changes a symptom minus its level 0 by at most DIFF_SHARE of the sign (or DIFF_FLOOR_M), the sign and
  the change read on the joints and on the mesh vertices (physio.pair_change: a finger sign moves vertices only);
- the clips the physio-v2 layer refused on the added acceleration and the physio-v4 layer refused (the hemiballism
  g1, the knee flexed 30 deg on a held hip, the functional axial jerk) pass, the reach pausing on its way to the
  support passes with the default anchoring (the rest gate a fallback), a passing default anchoring that turned
  with a moving hand gives way to the rest gate where that passes too (the Holmes tremor on the armrest) and stays
  where the gate would leave a pausing hand off its target (the reach pausing above the target block), and a sample
  of wrapped clips over every
  family stays valid (no new joint-limit violation, no skate, no foot moved in stance, no contact lost, no depth added
  inside a scene box or the body or between the legs, no jump, the hands on their supports); the truth's validity
  carries the verdict for the exports; the joint-limit table a clip is judged on;
- supine, the spine does not breathe (the head stays on the couch); the seated hands rest on their supports;
- the series of physio.SKIP and the walks are skipped and keep their bodies; the parkinsonism series left wrapped are
  those whose builders carry no background of their own;
- determinism; exam.exam_set() is not wrapped, catalogue_set(background=False) gives the bodies before SH1, and a
  wrapped build names the clip's own module for the code records of the exports.
Left to scripts/synth_background_sweep.py (reports/synth/background/summary.md) for their build times (minutes each;
the sweep records them): invol_hemiballism_E1_left_g2 and g3, invol_athetosis_E1_a3_d4, post_W_mirror, post_W_overflow,
post_W_writers_cramp_g2, post_E1_pnkd_excerpt, post_E7_pkd_dystonic
(the dystonia table in the guard), post_E7_torsion_dystonia_sev3 (the sway scaled on its own rows),
post_E7_cd_geste_C0 (the anchor error where the own limit guard holds the elbow, reported apart). Since the open-body
port's wave 7 the exam set holds no AMASS take (its stand, walk and freezing clips are procedural). Procedural bodies
in the neutral shape only; CPU; no store, lockbox, EVAL, CODY or care_pd record is read.
"""
import inspect
import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")

# open-body port, wave 7 (exam_v10): the exam set holds no AMASS take any more (its references and freezing clips are
# procedural); the set stays for the guard of built() and is empty
AMASS = set()
# clip and direct level 0 pairs: the physio-v2 pairs above a tenth of the sign or with a joint held equal moved
# (athetosis, chorea of the arms, the tardive limbs, the hemiballism, the E3 tremors at the target, the multifocal
# dystonia, the arm myoclonus posture, the knee flexion), the clips v2 refused on the added acceleration, and one pair
# per other family and posture (seated rest tremor, standing chorea, standing neck dystonia, seated jerks), with the
# two level-0 chains whose level 0 is not their series root
PAIRS = ["invol_athetosis_E1_a2_d4", "invol_athetosis_E1_a2_d3", "invol_chorea_E1_arms_g1", "invol_chorea_E1_arms_g2",
         "invol_td_E1_limbs_s2", "invol_hemiballism_E1_left_g1", "invol_lid_peak_E1_s4",
         "invol_td_stereotypy_E1_rubbing_s4", "tremor_E3_kinetic_g1b", "tremor_E3_kinetic_g3", "tremor_E3_kinetic_g4",
         "tremor_E3_essential_tetras2p5", "tremor_E3_essential_tetras3", "tremor_E3_intention_sara2",
         "post_E7_multifocal", "dystonia_E7_distal_leg_knee_flexion_g2", "dystonia_E7_neck_rotation_left_g3",
         "jerk_mdyst_E1_right_arm_posture_only", "jerk_mdyst_E1_right_arm_f3_a2", "tremor_E1_rest_g2",
         "invol_chorea_E7_g2", "jerk_myoclonus_E1_multifocal_f2_a2", "tremor_E1_dystonic_head_g2",
         # physio-v4: hand signs (physio-v3 faded the member's own fingers near a support while its level 0's kept
         # the drift: 2.77 and 1.04 of the hand sign), and a hand on the thigh whose forearm brushes the belly
         "tremor_E1_pillrolling_g1", "invol_athetosis_E1_a1_d4", "invol_td_E1_rocking_s4"]
CHAINS = ("jerk_mdyst_E1_right_arm_f3_a2", "tremor_E1_dystonic_head_g2")
# the clips physio-v2 refused on the added acceleration and built here (hemiballism g2 and g3: the sweep)
REFUSED_V2 = ["tremor_E3_kinetic_g3", "tremor_E3_kinetic_g4", "tremor_E3_essential_tetras2p5",
              "tremor_E3_essential_tetras3", "invol_lid_peak_E1_s4", "invol_td_stereotypy_E1_rubbing_s4"]
# the clips physio-v4 refused, which physio-v5 passes (their measures: reports/synth/background/summary.md): the
# hemiballism flings its arm at metres per second off and onto the thigh (the own anchoring ramped in before the
# landing turned with the flung arm), the knee flexed 30 deg on the level 0's hip row (the foot slid on physio-v4's
# flexion arc), the functional axial jerk (read where the jerked hand still moved on its support)
REFUSED_V4 = ["invol_hemiballism_E1_left_g1", "dystonia_E7_start_distal_leg_knee_flexion_30deg",
              "jerk_functional_E1_axial_sev3_dur2"]
# a hand pausing on its way to its support: a rest gate by default switched the own anchoring on and off while the
# wrist stood still; the gate is a fallback of the transfer, the default anchors everywhere as physio-v4
REST_SLOW = ["reach_E3_decomposition_icars2"]
# pairs above DIFF_SHARE, with the share they keep and why: none since physio-v6 (the rubbing stereotypy, the
# athetotic and tardive fingers and the rest tremor had bounds of 0.08 to 0.13; the series' finger drift, SERIES_QUIET,
# was the larger part of their change: reports/synth/background/summary.md and finger_rule.json)
SHARE_ABOVE = {}
# a sample of wrapped clips over the families and postures not in PAIRS (supine, arising, hands touching the face,
# finger tapping, outstretched arms, stepping, asterixis, ataxia, parkinsonian sequence effect)
SAMPLE = ["tremor_E1_rest_g0", "tremor_E2_postural_g2", "brady_E4a_ft_decrement_g1", "dystonia_E7_level0",
          "dystonia_E7_shoulder_elevation_elevation_g2", "ref_arise_E7", "post_E7_cd_geste_C1",
          "jerk_asterixis_E2_conn2_sev2", "invol_lid_diphasic_E7_stepping_s2", "park_E4L_sequence_ps_level2",
          "atax_SARA7_ddk_g2", "atax_SARA8_heelshin_g2"]
DIFF_FLOOR_M = 0.001
DIFF_SHARE = 0.05            # the owner's rule: the difference between a symptom and its level 0 stays the symptom
HELD_POS_M = 1e-9            # a joint whose rows and ancestors' rows are all held keeps the level 0's position
WALK_PAIR = "gait_E7p_retropulsion_g1"   # a walking clip and its level 0 (physio-v7: the layer on their still spans)


@pytest.fixture(scope="module")
def P():
    torch.set_num_threads(2)
    from mdfm.synth import physio
    return physio


@pytest.fixture(scope="module")
def cat(P):
    from mdfm.synth import catalogue as C
    specs = C.catalogue_set()
    return {s.clip_id: s for s in specs}


_BUILT = {}


def built(cat, cid):
    """(layered body, truth, raw body) of a catalogue clip, built once per session (one raw build)."""
    if cid not in _BUILT:
        assert cid not in AMASS
        raw, truth = cat[cid].build.raw()
        b, tr = cat[cid].build.with_background(raw, truth)
        _BUILT[cid] = (b, tr, raw)
    return _BUILT[cid]


def _hand6d(b):
    from mdfm.synth.hands import _eye6, has_hands
    return b.hand6d if has_hands(b) else _eye6(b.T)


def _assert_held(P, raw, raw0, out, out0, tag=""):
    """Every row, hand row and translation raw holds equal to raw0 is equal in out and out0 bit for bit, and the
    joints whose rows and ancestors' rows are all held keep their positions within HELD_POS_M; returns the held
    shares (body rows, hand rows)."""
    from mdfm.body.smplh import PARENTS
    T = min(raw.T, raw0.T)
    held = P.rows_equal(raw.rot6d, raw0.rot6d, T)
    assert torch.equal(out.rot6d[:T][torch.from_numpy(held)], out0.rot6d[:T][torch.from_numpy(held)]), tag
    th = P.rows_equal(raw.transl[:, None], raw0.transl[:, None], T)[:, 0]
    assert torch.equal(out.transl[:T][torch.from_numpy(th)], out0.transl[:T][torch.from_numpy(th)]), tag
    hh = P.rows_equal(_hand6d(raw), _hand6d(raw0), T)
    assert torch.equal(_hand6d(out)[:T][torch.from_numpy(hh)], _hand6d(out0)[:T][torch.from_numpy(hh)]), tag
    chain = np.zeros_like(held)                                       # the joint and all its ancestors held
    for j in range(held.shape[1]):
        chain[:, j] = held[:, j] & (th if j == 0 else chain[:, PARENTS[j]])
    if chain.any():
        d = (out.joints()[:T] - out0.joints()[:T]).norm(dim=-1).numpy()
        assert float(d[chain].max()) <= HELD_POS_M, (tag, float(d[chain].max()))
    return float(held.mean()), float(hh.mean())


# ---------------------------------------------------------------- the layer
def test_the_layer_is_a_function_of_time(P):
    rng = np.random.default_rng(3)
    band = P.Band(rng, P.SWAY_BAND_HZ)
    t_long = np.arange(12001) / 100.0
    t_short = np.arange(3001) / 100.0
    assert np.array_equal(band(t_long)[:3001], band(t_short))
    assert np.array_equal(band.integral(t_long)[:3001], band.integral(t_short))
    L = P.make_layer(P.series_seed("tremor_E1_rest_g0"), "seated")
    assert len(L["t"]) == int(round(P.HORIZON_S * 100)) + 1
    assert not L["breath"].flags.writeable
    pre = P.layer_prefix(L, 4501)
    assert all(np.array_equal(pre[k], L[k][:4501]) for k in L)
    with pytest.raises(ValueError):
        P.layer_prefix(L, len(L["t"]) + 1)
    seed = P.series_seed("x")
    st, sit, sup, tra, wlk = (P.make_layer(seed, k) for k in ("standing", "seated", "supine", "transfer", "walking"))
    assert np.array_equal(st["breath"], sit["breath"]) and np.array_equal(st["head_yaw_deg"], sit["head_yaw_deg"])
    for k in ("head_yaw_deg", "head_pitch_deg", "lean_ap_deg", "lean_ml_deg", "sway_ap_m", "sway_ml_m",
              "fingers_deg"):
        assert not np.any(sup[k]), k                                      # supine: breathing only
    for k in ("lean_ap_deg", "fingers_deg"):
        assert not np.any(tra[k]) and not np.any(wlk[k]), k               # arising and walking: no trunk lean, no
    for k in ("breath", "head_yaw_deg", "sway_ap_m", "sway_ml_m"):         # finger drift; the standing signals
        assert np.array_equal(tra[k], st[k]) and np.array_equal(wlk[k], st[k]), k   # (physio-v7: their spans)
    assert not np.any(st["lean_ap_deg"]) and not np.any(sit["sway_ap_m"])
    assert sit["lean_ap_deg"].min() == 0.0                                # the seated lean leans forward only
    assert P.series_seed("a") != P.series_seed("b") and P.series_seed("a") == P.series_seed("a")
    with pytest.raises(ValueError):
        P.make_layer(seed, "running")
    assert P.VERSION == "physio-v10"                                     # plan phase E (D145 and after)
    # the finger lift: never flexes, about the drift where it extends by more than e, about 0 where it would flex by
    # more than e, smooth, -e/2 at 0, and a function of the drift at each knot (the prefix property holds)
    e = P.FINGER_LIFT_SOFT_DEG
    x = np.linspace(-20.0, 20.0, 4001)
    y = P.finger_lift(x)
    assert np.all(y <= 0.0) and np.all(np.diff(y) > 0.0) and abs(float(P.finger_lift(0.0)) + 0.5 * e) < 1e-12
    assert np.all(np.abs(y[x < -10 * e] - x[x < -10 * e]) < 0.03 * e) and np.all(y[x > 10 * e] > -0.03 * e)
    assert np.abs(np.diff(y, 2)).max() < 1e-3                             # no kink
    f = L["fingers_deg"]
    assert np.array_equal(P.finger_lift(f[:4501]), P.finger_lift(f)[:4501])


def test_the_local_fade_the_support_weights_and_the_inside_ramp(P):
    f = P.local_fade([500], 1200, 100.0)
    hold, ramp = int(P.FADE_HOLD_S * 100), int(P.ANCHOR_RAMP_S * 100)
    assert np.all(f[500 - hold:500 + hold + 1] == 0.0) and f[0] == 1.0 and f[-1] == 1.0
    assert np.all(f[500 + hold + ramp + 1:] == 1.0) and np.abs(np.diff(f)).max() <= 2.0 / ramp + 1e-12   # slow
    assert np.all(P.local_fade([900], 1200, 100.0, f) <= f)
    lab = {"left": np.r_[np.full(100, -1), np.full(100, -2), np.full(200, -1)],
           "right": np.r_[np.full(60, 1), np.full(90, -1), np.full(250, P.HEAD)]}
    for side in ("left", "right"):
        w, on = P.support_weights(lab, side, 100.0)
        tot = sum(w.values())
        assert np.all(tot <= 1.0 + 1e-12) and np.all(on[tot > 0])
        assert np.abs(np.diff(tot)).max() < 0.03                                      # ramps, no jump
    w, _ = P.support_weights(lab, "right", 100.0)
    assert w[P.HEAD][160:].min() == 1.0 and w[1][160:].max() == 0.0                 # the face holds its run
    assert w[1][0] == 1.0 and abs(w[1][105] + w[P.HEAD][105] - 1.0) < 1e-12        # the lap before it, a cross-fade
    m = np.r_[np.zeros(50, bool), np.ones(200, bool), np.zeros(20, bool), np.ones(30, bool), np.zeros(50, bool)]
    r = P.inside_ramp(m, 20)
    assert np.all(r[~m] == 0.0) and r[150] == 1.0 and np.all(r[270:300] == 0.0)    # never off the mask; short runs 0
    assert np.abs(np.diff(r)).max() < 0.1                                           # no jump


def test_the_amplitudes_follow_their_sources(P):
    from mdfm.synth.base import static_body
    from mdfm.body.smplh import fk
    rates, shares, sway = [], [], []
    for s in range(12):
        L = P.make_layer(P.series_seed(f"cal{s}"), "standing")
        b = L["breath"]
        up = np.flatnonzero((b[1:] > 0.5) & (b[:-1] <= 0.5))              # inhalations crossing half
        rates.append(60.0 * (len(up) - 1) / ((up[-1] - up[0]) / 100.0))
        rising = np.diff(b) > 0
        shares.append(float(rising.mean()))
        sway.append([float(np.sqrt((L["sway_ap_m"] ** 2).mean())), float(np.sqrt((L["sway_ml_m"] ** 2).mean()))])
    # SOURCES['motamedifakhr2017']: 14.13 breaths per minute, tI/tE 0.75 (inspiration 0.4286 of the cycle)
    assert np.mean(rates) == pytest.approx(P.BREATH_PER_MIN, rel=0.03)
    assert np.mean(shares) == pytest.approx(P.BREATH_TI_TE / (1 + P.BREATH_TI_TE), abs=0.01)
    # SOURCES['foulger2025']: the pelvis sway RMS (their mean within a tenth; a single 120 s layer, whose slowest
    # sinusoids last 10 s, within a third)
    sway = np.array(sway)
    assert np.allclose(sway.mean(0), P.SWAY_RMS_M, rtol=0.1)
    assert np.all(np.abs(sway / np.array(P.SWAY_RMS_M) - 1.0) < 1.0 / 3.0)
    # SOURCES['melo2009']: the 90 s range of the neck joint (the T2 proxy) on the procedural sit, over 16 seeds
    sit = static_body("sit", 90.0)
    R0, tr, rest = sit.local_R(), sit.transl, sit.rest()
    J0 = fk(R0, tr, rest)[0][:, 12].numpy()
    ranges = []
    for s in range(16):
        L = P.layer_prefix(P.make_layer(P.series_seed(f"sit{s}"), "seated"), sit.T)
        Lz = {k: (v if k in ("lean_ap_deg", "lean_ml_deg") else np.zeros_like(v)) for k, v in L.items()}
        R = R0.clone()
        P._compose_trunk(R, Lz, "seated")
        J = fk(R, tr, rest)[0][:, 12].numpy()
        ranges.append([np.ptp(J[:, 2] - J0[:, 2]), np.ptp(J[:, 0] - J0[:, 0])])
    assert np.allclose(np.mean(ranges, 0), P.SIT_T2_RANGE_M, rtol=0.15)
    g = P.sit_gains()
    assert 0.3 < g["lever_ap_m"] < 0.5 and 3.0 < g["range_per_rms"] < 7.0


# ---------------------------------------------------------------- the transfer on procedural bodies
def _bump(T, a, b):
    """(T,) 0 to 1: a raised-cosine bump over knots [a, b)."""
    x = np.zeros(T)
    k = np.arange(a, b)
    x[a:b] = 0.5 - 0.5 * np.cos(2.0 * math.pi * (k - a) / (b - a))
    return x


def _with_rotation(body, joint, axis, deg):
    """A copy of body whose row joint takes an extra rotation about axis by deg (T,) (parent frame); other rows bit
    for bit."""
    from mdfm.synth.body import axis_rotation
    R = body.local_R()
    R[:, joint] = axis_rotation(axis, torch.from_numpy(np.radians(deg))) @ R[:, joint]
    return body.with_rotations(R, [joint])


def test_the_transfer_holds_the_rows_a_member_shares(P):
    from mdfm.synth.base import static_body
    from mdfm.synth.hands import hands_of
    l0 = hands_of(static_body("stand", 4.0))
    T = l0.T
    layer = P.make_layer(P.series_seed("unit_stand"), "standing")
    kw = dict(scene=None, table="healthy", resting={"left": True, "right": True}, kind="standing")
    out0, rec0 = P.solve(l0, layer, **kw)
    assert rec0["validity"]["ok"], rec0["validity"]
    # a member: the left elbow bends over 1 to 3 s, the right hip and knee over 1.5 to 2.5 s (its own rows), the
    # rest held
    m1 = _with_rotation(l0, 18, "y", -25.0 * _bump(T, 100, 300))
    m1 = _with_rotation(m1, 2, "x", -8.0 * _bump(T, 150, 250))
    m1 = _with_rotation(m1, 5, "x", 12.0 * _bump(T, 150, 250))
    out1, rec1 = P.transfer(m1, l0, out0, layer, reference="unit_l0", **kw)
    _assert_held(P, m1, l0, out1, out0, "member")
    own = ~P.rows_equal(m1.rot6d, l0.rot6d, T)
    assert own[:, 18].sum() > 100 and own[:, [3, 9, 12, 15, 16, 20]].sum() == 0
    dl = P.deltas(l0, out0)
    R1 = out1.local_R().numpy()
    want = P._mul(dl["R"], m1.local_R().numpy())
    k = np.flatnonzero(own[:, 18])
    assert np.abs(R1[k, 18] - want[k, 18]).max() < 1e-9               # the own rows take the level 0's delta (the
    #                                                                   rot6d storage rounds by about 1e-12)
    assert rec1["transfer"]["rows_corrected_held"] == 0
    assert rec1["guard"]["legs_fixed_knots"]["right"] > 0             # the bent leg re-solved on its own rows
    assert rec1["ankle_error_m"] <= 1e-6
    assert rec1["validity"]["ok"], rec1["validity"]
    # a member of the member (a level-0 chain): the head turns too; the rows it shares with its level 0 stay its
    m2 = _with_rotation(m1, 15, "y", 15.0 * _bump(T, 50, 350))
    out2, rec2 = P.transfer(m2, m1, out1, layer, reference="unit_m1", **kw)
    _assert_held(P, m2, m1, out2, out1, "chain")
    assert torch.equal(out2.rot6d[:, 18], out1.rot6d[:, 18])          # the elbow the two share, as the level 0's
    # a member longer than its level 0: its own solve beyond it, blended in without a jump
    ext = replace(m1, rot6d=torch.cat([m1.rot6d, m1.rot6d[-1:].expand(150, -1, -1)], 0),
                  transl=torch.cat([m1.transl, m1.transl[-1:].expand(150, -1)], 0),
                  contact=torch.cat([m1.contact, m1.contact[-1:].expand(150, -1)], 0),
                  hand6d=torch.cat([m1.hand6d, m1.hand6d[-1:].expand(150, -1, -1)], 0))
    out3, rec3 = P.transfer(ext, l0, out0, layer, own=lambda: P.solve(ext, layer, **kw), reference="unit_l0", **kw)
    assert rec3["transfer"]["beyond_knots"] == 150 and "beyond_reference" in rec3["deviations"]
    _assert_held(P, ext, l0, out3, out0, "longer")
    assert rec3["validity"]["accel_excess_m_s2"] <= P.ACCEL_TOL_M_S2 and rec3["validity"]["ok"], rec3["validity"]
    # determinism of the transfer
    out1b, rec1b = P.transfer(m1, l0, out0, layer, reference="unit_l0", **kw)
    assert torch.equal(out1.rot6d, out1b.rot6d) and torch.equal(out1.hand6d, out1b.hand6d)
    assert json.dumps(rec1, sort_keys=True, default=float) == json.dumps(rec1b, sort_keys=True, default=float)


def test_a_members_own_finger_rows_take_the_level0_lift(P, cat, monkeypatch):
    # physio-v4, kept in physio-v5 and v6 (FINGER_RULE 'held'): on a seated level 0 whose hands rest on the armrests
    # with the fingers lifting (the head-sign level 0: the rest-tremor one takes no finger drift since physio-v6,
    # SERIES_QUIET), a member whose left index MCP flexes over 2 to 6 s (its own rows) takes the level 0's
    # finger delta there, no fade; held hand rows are the level 0's. With FINGER_RULE 'all' (the session's decision of
    # 2026-11-03, measured to raise the finger pairs' change, reports/synth/background/finger_rule.json) the run takes
    # no finger drift inside it, the drift fading out from the held rows over FINGER_STILL_RAMP_S without a jump. A
    # posture held still far from the level 0's takes none under 'held' either (post_E1_fixed_fist). A longer member
    # keeps a finger layer beyond its level 0's end (its own solve's, blended in)
    from mdfm.body.rotations import matrix_to_rot6d
    from mdfm.synth import hands as HD
    cid = "dystonia_E1_level0"
    s = cat[cid]
    b0, tr0, raw0 = built(cat, cid)
    raw0 = HD.hands_of(raw0) if not HD.has_hands(raw0) else raw0
    g = tr0["background"]
    assert "fingers" in g["components"] and g["fingers_lift_only"] == ["left", "right"]
    layer = P.make_layer(g["seed"], g["kind"], quiet=tuple(g["quiet"]))
    T = raw0.T
    j = HD.hand_joint("left", "index", 0)
    row = HD.hand_row(j)
    ax = torch.as_tensor(HD.hand_frame()["left"]["joints"][j]["flex"], dtype=torch.float64)
    Rh = raw0.hand_R()
    Rh[:, row] = P._exp(ax[None] * torch.from_numpy(np.radians(30.0 * _bump(T, 200, 600)))[:, None]) @ Rh[:, row]
    m1 = replace(raw0, hand6d=matrix_to_rot6d(Rh))
    kw = dict(scene=s.scene, table="healthy", resting=g["resting_hands"], kind=g["kind"])
    assert P.FINGER_RULE == "held"
    out1, rec1 = P.transfer(m1, raw0, b0, layer, reference=cid, **kw)
    _assert_held(P, m1, raw0, out1, b0, "finger member")
    hown = ~P.rows_equal(m1.hand6d, raw0.hand6d, T)
    k = np.flatnonzero(hown[:, row])
    assert len(k) > 300 and hown.sum() == len(k)
    want = P._mul(P.deltas(raw0, b0)["H"], m1.hand_R().numpy())
    assert np.abs(out1.hand_R().numpy()[k, row] - want[k, row]).max() < 1e-9    # the level 0's lift, not faded
    assert "own_finger_fade" not in rec1["deviations"] and "own_fingers_still" not in rec1["deviations"]
    assert rec1["validity"]["ok"], rec1["validity"]
    monkeypatch.setattr(P, "FINGER_RULE", "all")
    out1s, rec1s = P.transfer(m1, raw0, b0, layer, reference=cid, **kw)
    monkeypatch.setattr(P, "FINGER_RULE", "held")
    _assert_held(P, m1, raw0, out1s, b0, "finger member, still")
    assert "own_fingers_still" in rec1s["deviations"]
    assert len(k) - 2 <= rec1s["guard"]["fingers_still_knots"] <= len(k)          # the run's end knots ramp from 1
    assert rec1s["validity"]["ok"], rec1s["validity"]
    Ro, Rm = out1s.hand_R().numpy()[:, row], m1.hand_R().numpy()[:, row]
    inner = np.arange(k[0] + 2 * int(P.FINGER_STILL_RAMP_S * 100), k[-1] - 2 * int(P.FINGER_STILL_RAMP_S * 100))
    assert len(inner) > 100 and np.abs(Ro[inner] - Rm[inner]).max() < 1e-9           # the clip's own row, no drift
    assert np.abs(np.diff(Ro - Rm, axis=0)).max() < 5e-3                               # no jump at the run's ends
    n = 150
    ext = replace(m1, rot6d=torch.cat([m1.rot6d, m1.rot6d[-1:].expand(n, -1, -1)], 0),
                  transl=torch.cat([m1.transl, m1.transl[-1:].expand(n, -1)], 0),
                  contact=torch.cat([m1.contact, m1.contact[-1:].expand(n, -1)], 0),
                  hand6d=torch.cat([m1.hand6d, m1.hand6d[-1:].expand(n, -1, -1)], 0))
    out2, rec2 = P.transfer(ext, raw0, b0, layer, own=lambda: P.solve(ext, layer, **kw), reference=cid, **kw)
    mid = HD.hand_row(HD.hand_joint("right", "middle", 1))
    assert rec2["transfer"]["beyond_knots"] == n
    lay = (out2.hand_R()[T:, mid] - ext.hand_R()[T:, mid]).abs().amax((-1, -2))
    assert float(lay.max()) > 1e-4                                       # the finger layer goes on beyond


def test_a_fist_held_still_takes_no_finger_drift(P, cat):
    # FINGER_RULE 'held': the clenched fist is fixed from the first knot, its rows far from the level 0's, so they take
    # no drift inside the run (a fixed contracture does not drift), the ramp at the clip's ends aside; the difference
    # with its level 0 falls within the owner's share
    cid = "post_E1_fixed_fist"
    b, tr, raw = built(cat, cid)
    g = tr["background"]
    assert "own_fingers_still" in g["deviations"] and g["validity"]["ok"]
    l, _, l0 = built(cat, cat[cid].level0)
    from mdfm.synth import hands as HD
    row = HD.hand_row(HD.hand_joint("left", "index", 0))
    n = 2 * int(P.FINGER_STILL_RAMP_S * 100)
    assert np.abs(b.hand_R().numpy()[n:-n, row] - raw.hand_R().numpy()[n:-n, row]).max() < 1e-9
    pc = P.pair_change(raw, b, l0, l)
    assert pc["share"] <= DIFF_SHARE, pc


def test_a_root_takes_the_rest_gate_only_where_it_passes_more(P, monkeypatch):
    # solve(): a root failing the acceleration check tries the anchoring only where the hand rests, keeps it where that
    # fails fewer checks and names it; where the check still fails, the damped elbow step (physio-v6, 'anchor_damped'),
    # kept likewise; where neither is kept, both together (physio-v10), kept likewise; a root passing never runs
    # either, so its body and its series keep their bits
    def stub(failed):
        def apply(body, layer, rest_gate=False, arm_damp=0.0, **kw):
            key = (rest_gate, arm_damp > 0.0)
            calls.append(key)
            if arm_damp > 0.0:
                assert arm_damp == P.ARM_FIX_DAMP_M
            f = failed[key]
            return key, {"validity": {"failed": list(f), "ok": not f, "_bad_knots": np.zeros(0, int)},
                         "guard": {}, "deviations": {}}
        return apply
    accel = ["accel_excess_m_s2"]
    F, T = False, True
    for failed, gate, damped, n in (({(F, F): accel, (T, F): []}, T, F, [(F, F), (T, F)]),
                                    ({(F, F): accel, (T, F): accel, (F, T): []}, F, T, [(F, F), (T, F), (F, T)]),
                                    ({(F, F): accel, (T, F): accel, (F, T): accel, (T, T): []}, T, T,
                                     [(F, F), (T, F), (F, T), (T, T)]),
                                    ({(F, F): accel, (T, F): accel, (F, T): accel, (T, T): accel}, F, F,
                                     [(F, F), (T, F), (F, T), (T, T)]),
                                    ({(F, F): accel, (T, F): ["accel_excess_m_s2", "anchor_error_m"],
                                      (F, T): accel, (T, T): []}, T, T, [(F, F), (T, F), (F, T), (T, T)]),
                                    ({(F, F): []}, F, F, [(F, F)])):
        calls = []
        monkeypatch.setattr(P, "apply", stub(failed))
        out, rec = P.solve(None, None)
        assert out == (gate, damped) and calls == n
        assert bool(rec["guard"].get("anchor_rest_gate")) == gate and ("anchor_rest_gate" in rec["deviations"]) == gate
        assert bool(rec["guard"].get("anchor_damped")) == damped and ("anchor_damped" in rec["deviations"]) == damped


def test_the_raise_after_a_rest_passes_with_the_damped_anchoring(P, cat):
    # the settling of 2026-11-10: the right hand lifting off the armrest in the raise after REST_BEFORE_RAISE_S of
    # rest, its elbow near straight, took the other elbow solution for one knot under the undamped anchoring (the
    # acceleration check refused the root at 3, 3.5 and 4 s); the damped step passes it, and its two clips with it
    from mdfm.synth import exam as EX
    assert EX.REST_BEFORE_RAISE_S == 4.0
    root = "tremor_E2_raise_long_rest_g0"
    # plan phase E (physio-v10): on the open body both arms straighten off the armrests at the raise's onset under
    # the anchoring at full weight; the gate alone and the damped step alone each still failed the acceleration check
    # and the root was refused; the two together pass (scripts/openbody_phaseE_checks.py, section raise)
    b, tr, raw = built(cat, root)
    g = tr["background"]
    assert g["validity"]["ok"] and "anchor_damped" in g["deviations"] and g["guard"]["anchor_damped"]
    assert "anchor_rest_gate" in g["deviations"] and g["guard"]["anchor_rest_gate"]
    kw = dict(scene=cat[root].scene, table="healthy", resting=g["resting_hands"], kind=g["kind"])
    layer = P.make_layer(g["seed"], g["kind"], quiet=tuple(g["quiet"]))
    _, r0 = P.apply(raw, layer, **kw)
    assert "accel_excess_m_s2" in r0["validity"]["failed"]                  # the undamped anchoring flips the arm
    for cid in ("tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2"):
        _, tr2, _ = built(cat, cid)
        g2 = tr2["background"]
        assert g2["validity"]["ok"] and "anchor_damped" in g2["series_deviations"], (cid, g2["validity"]["failed"])


def test_an_arm_pressed_into_a_scene_box_turns_off_it_on_its_own_rows(P):
    # scene_turn on the procedural sit, a box beside the left elbow and the trunk turned into it (an arm on an armrest's
    # edge): an own shoulder row takes the turn about the line to the wrist, the wrist stays; a held one leaves it to
    # the first own spine row
    from mdfm.synth.base import static_body
    from mdfm.synth.hands import hands_of
    from mdfm.synth.viewer import knot_vertices, dominant_joint
    base = hands_of(static_body("sit", 0.6))
    T = base.T
    out = _with_rotation(base, 3, "y", 4.0 * np.sin(np.pi * np.arange(T) / (T - 1)) ** 2)
    V0 = knot_vertices(base, [0])[0]
    arm = V0[dominant_joint("neutral") == 18]
    near = arm[arm[:, 2] < float(np.median(arm[:, 2]))]                  # the upper end of the elbow's vertices
    lo = np.array([near[:, 0].max() + 1e-4, near[:, 1].min() - 0.05, near[:, 2].min() - 0.05])
    boxes = [(lo, np.eye(3), np.array([0.1, np.ptp(near[:, 1]) + 0.1, np.ptp(near[:, 2]) + 0.1]))]

    def added(o):
        ks = np.arange(0, T, 5)
        db, do = P.box_depth(knot_vertices(base, ks), boxes)[0], P.box_depth(knot_vertices(o, ks), boxes)[0]
        return float(np.maximum(0.0, do - np.maximum(db, 0.0)).max())
    assert added(out) > 2.0 * P.SCENE_TOL_M
    own = np.ones((T, 22), bool)
    om = P.scene_turn(base, out, own, 100.0, boxes)
    assert set(om) == {16}
    o2 = P.with_row_turn(out, om, own)
    assert added(o2) < 0.5 * added(out)
    moved = np.linalg.norm(o2.joints().numpy() - out.joints().numpy(), axis=-1)
    assert moved[:, 20].max() < 0.1 * moved[:, 18].max()                 # the hand keeps its place, the elbow swings
    assert np.array_equal(o2.rot6d[:, 17:].numpy(), out.rot6d[:, 17:].numpy())   # the rows beyond it bit for bit
    own[:, 16] = False
    om = P.scene_turn(base, out, own, 100.0, boxes)
    assert set(om) == {9}
    o3 = P.with_row_turn(out, om, own)
    assert added(o3) < added(out) and torch.equal(o3.rot6d[:, 16], out.rot6d[:, 16])


def test_the_clips_beside_the_armrests_stay_valid_on_their_chair(P, cat):
    # whichever armrests the default chair draws: the functional tremor's ballistic movement off the armrest and the
    # tonic tic's trunk turning beside it pass, the held rows stay the level 0's (the rest gate and the box turn they
    # take on the longer armrests: the two tests above)
    for cid in ("tremor_E2f_functional_g0", "tremor_E2f_functional_g2", "jerk_tonic_tics_E1_f1_i4"):
        b, tr, raw = built(cat, cid)
        assert tr["background"]["validity"]["ok"], cid
        if cat[cid].level0 is not None:
            b0, _, raw0 = built(cat, cat[cid].level0)
            _assert_held(P, raw, raw0, b, b0, cid)


def test_a_finger_pressed_into_the_body_fades_on_the_own_rows(P, cat):
    # physio-v4: the chorea's hand, turned on the thigh, took the series' finger lift into the body (1.36 mm when this
    # was set) and the lift faded on the clip's own finger rows; since wave SH1d the palms rest on the thighs (no palm
    # vertex inside the skin) and no clip of the catalogue needs the fade (the sweep of that wave): the clip stands
    # within the bounds without it, the held rows stay the level 0's
    cid = "invol_chorea_E1_g4"
    b, tr, raw = built(cat, cid)
    b0, _, raw0 = built(cat, cat[cid].level0)
    g = tr["background"]
    assert "own_finger_fade" not in g["deviations"]
    v = g["validity"]
    assert v["body_added_m"] <= P.BODY_TOL_M and v["scene_added_m"] <= P.SCENE_TOL_M
    _assert_held(P, raw, raw0, b, b0, cid)


def test_the_rest_weight_and_the_vertical_goal(P):
    # physio-v5: a member's own anchoring rests where the wrist moves slower than ARM_REST_M_S: none within the edge of
    # a faster knot, whole in a long still run, a ramp without a jump; a slow tremor of the resting hand stays at rest
    T, n_edge = 600, 10
    speed = np.zeros(T)
    speed[200:230] = 2.0
    w = P.rest_weight(speed, 100.0, n_edge)
    assert np.all((w >= 0.0) & (w <= 1.0)) and np.all(w[200 - n_edge:230 + n_edge] == 0.0)
    assert np.all(w[:150] == 1.0) and np.all(w[290:] == 1.0)
    assert np.abs(np.diff(w)).max() <= 1.0 / (P.ARM_REST_S * 100.0) + 1e-9         # a step of the moving mean at most
    tremor = 0.8 * P.ARM_REST_M_S * np.abs(np.sin(2.0 * math.pi * 5.0 * np.arange(T) / 100.0))
    assert np.all(P.rest_weight(tremor, 100.0, n_edge) == 1.0)
    # the vertical goal: the point of the vertical through A at the distance L from P0, nearest A; the sphere's point
    # nearest A where no point of the vertical lies at L
    P0 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    A = torch.tensor([[0.10, 0.10, 0.05], [0.80, 0.90, 0.0]], dtype=torch.float64)
    L = torch.tensor([float((A[0] - P0[0]).norm()) + 0.003, 0.5], dtype=torch.float64)
    G = P.vertical_goal(P0, A, L)
    assert torch.allclose(G[0, [0, 2]], A[0, [0, 2]], atol=1e-15) and abs(float((G[0] - P0[0]).norm() - L[0])) < 1e-12
    assert 0.003 < float(A[0, 1] - G[0, 1]) < 0.004                      # lower by the residual over the leg's cosine
    assert abs(float((G[1] - P0[1]).norm()) - 0.5) < 1e-12 and float(torch.cross(G[1] - P0[1], A[1] - P0[1], dim=0)
                                                                     .norm()) < 1e-12


def test_the_guard_reads_the_table_the_clip_is_judged_on(P):
    # physio-v5: the own limit guard of a clip judged on limits_ext's dystonia table reads it at each knot: a left
    # shoulder at the healthy adduction bound whose collar the layer raises by 2 deg may adduct 1 deg further (the
    # coupling widens the bound by the collar's rise), where the healthy table blends it back
    from mdfm.synth.body import axis_rotation, joint_limits
    lim = joint_limits()
    T = 40
    eye = torch.eye(3, dtype=torch.float64).expand(T, 22, 3, 3).clone()
    R_in = eye.clone()
    R_in[:, 16] = axis_rotation("z", torch.full((T,), math.radians(lim.lo[16, 2]), dtype=torch.float64))
    R_out = R_in.clone()
    R_out[:, 13] = axis_rotation("z", torch.full((T,), math.radians(2.0), dtype=torch.float64)) @ R_in[:, 13]
    R_out[:, 16] = axis_rotation("z", torch.full((T,), math.radians(-1.0), dtype=torch.float64)) @ R_in[:, 16]
    Rh, gh = P.guard_joints(R_out, R_in, (13, 16), lim, 100.0)
    Rd, gd = P.guard_joints(R_out, R_in, (13, 16), lim, 100.0, table="dystonia")
    assert gh.get(16, 0) == T and 16 not in gd and torch.equal(Rd, R_out)


def test_a_member_leg_on_a_held_row_and_at_its_bound(P):
    # physio-v5 on the procedural stand: (1) a member whose left knee flexes 20 deg on its own row under the level 0's
    # hip row, its ankle turned back by 20 deg so that the foot stays flat: the ankle keeps its horizontal place
    # exactly, its height takes the residual (reported); (1b) the same knee without the ankle's turn, the foot on its
    # toes (toe_stance), its ankle row its own: the ball keeps its place exactly, the foot turns about it; (2) a member
    # whose left hip stands at the table's abduction bound, its pelvis, translation and leg rows its own: the sway is
    # scaled down on those rows instead of the guard undoing the leg's solve, the foot stays planted
    from mdfm.synth.base import static_body
    from mdfm.synth.body import components_deg, joint_limits
    from mdfm.synth.hands import hands_of
    l0 = hands_of(static_body("stand", 4.0))
    T = l0.T
    layer = P.make_layer(P.series_seed("unit_stand"), "standing")
    kw = dict(scene=None, table="healthy", resting={"left": True, "right": True}, kind="standing")
    out0, _ = P.solve(l0, layer, **kw)
    m = _with_rotation(_with_rotation(l0, 4, "x", 20.0 * np.ones(T)), 7, "x", -20.0 * np.ones(T))
    o, r = P.transfer(m, l0, out0, layer, reference="unit_l0", **kw)
    v = r["validity"]
    _assert_held(P, m, l0, o, out0, "knee alone")
    Jm, Jo = m.joints().numpy(), o.joints().numpy()
    assert v["toe_knots"] == 0
    assert np.abs((Jo - Jm)[:, 7, [0, 2]]).max() < 1e-9 and np.abs(Jo[:, 7, 1] - Jm[:, 7, 1]).max() > 5e-4
    assert v["ok"] and v["foot_moved_m"] <= P.FOOT_TOL_M and v["foot_height_moved_m"] > 5e-4
    mt = _with_rotation(_with_rotation(l0, 4, "x", 20.0 * np.ones(T)), 7, "x", 1e-4 * np.ones(T))   # ankle row own
    ot, rt = P.transfer(mt, l0, out0, layer, reference="unit_l0", **kw)
    vt = rt["validity"]
    _assert_held(P, mt, l0, ot, out0, "knee alone, on the toes")
    Jm, Jo = mt.joints().numpy(), ot.joints().numpy()
    assert vt["toe_knots"] == T and vt["ok"], vt
    assert np.abs(Jo[:, 10] - Jm[:, 10]).max() < 1e-9 and np.abs(Jo[:, 7] - Jm[:, 7]).max() > 5e-4
    assert vt["foot_moved_m"] < 1e-9 and vt["foot_height_moved_m"] < 1e-9
    lim = joint_limits()
    m2 = _with_rotation(l0, 1, "z", (lim.hi[1, 2] - components_deg(l0.rot6d)[0, 1, 2]) * np.ones(T))
    m2 = _with_rotation(m2, 0, "y", 1e-4 * np.ones(T))
    m2 = replace(m2, transl=m2.transl + 1e-6)
    for j in (2, 4, 5, 7, 8):
        m2 = _with_rotation(m2, j, "x", 1e-4 * np.ones(T))
    o2, r2 = P.transfer(m2, l0, out0, layer, reference="unit_l0", **kw)
    v2 = r2["validity"]
    _assert_held(P, m2, l0, o2, out0, "hip at its bound")
    assert "own_sway_scaled" in r2["deviations"] and r2["guard"]["sway_scaled_knots"] > 0
    assert "own_limit_guard" not in r2["deviations"]
    assert v2["ok"] and v2["foot_moved_m"] <= 1e-6 and v2["limits_new"] == 0, v2


def test_the_anchor_reading_of_a_member(P):
    # a member's wrist error: read as it is where its own correction is whole, less what a held arm hands over in
    # the ramp from it, the held arm itself reported apart
    T, n = 100, 5
    err = np.zeros(T)
    err[30:40] = 0.002                                                    # the held arm 2 mm off the goal
    err[40:45] = np.linspace(0.0022, 0.0004, 5)                           # the ramp of the own rows takes it back
    err[70] = 0.0015                                                      # a miss where the own rows are whole
    touch = np.zeros(T, bool)
    touch[20:90] = True
    own_arm = np.ones(T, bool)
    own_arm[:40] = False
    full = np.zeros(T, bool)
    full[45:] = True
    r = P._anchor_reading(err, touch, own_arm, full, n)
    assert r["anchor_error_held_rows_m"] == pytest.approx(0.002)
    assert r["anchor_error_ramp_m"] == pytest.approx(0.0022)
    assert r["anchor_error_m"] == pytest.approx(0.0015)                  # the whole rows' miss, not the inherited 2 mm
    err[42] = 0.004                                                       # the ramp adding 2 mm of its own
    assert P._anchor_reading(err, touch, own_arm, full, n)["anchor_error_m"] == pytest.approx(0.002)
    # a hand standing still for a pause shorter than the rest gate's ramp (rest below 1, no own anchoring there) holds
    # the check; moving, the same miss is reported apart
    err[42] = 0.0
    err[60:64] = 0.009
    rest = np.ones(T)
    rest[55:70] = 0.5
    r = P._anchor_reading(err, touch, own_arm, full, n, rest)
    assert r["anchor_error_moving_m"] == pytest.approx(0.009) and r["anchor_error_m"] == pytest.approx(0.0015)
    assert r["anchor_error_dwell_m"] == 0.0
    dwell = np.zeros(T, bool)
    dwell[58:66] = True
    r = P._anchor_reading(err, touch, own_arm, full, n, rest, dwell=dwell)
    assert r["anchor_error_m"] == pytest.approx(0.009) == r["anchor_error_dwell_m"]
    assert r["anchor_error_moving_m"] == 0.0
    # the gate a refused default falls back on: the pause reported, not judged
    r = P._anchor_reading(err, touch, own_arm, full, n, rest, dwell=dwell, binds=False)
    assert r["anchor_error_m"] == pytest.approx(0.0015) and r["anchor_error_dwell_m"] == pytest.approx(0.009)
    assert r["anchor_error_moving_m"] == pytest.approx(0.009)


def test_the_dwell_window_leaves_out_a_reversal(P):
    # the knots _transfer reads as a hand standing still: a pause, not the zero of a tremor's speed
    T, rate = 200, 100.0
    n = int(round(P.ARM_DWELL_S * rate))
    tremor = 0.3 * np.abs(np.sin(2.0 * math.pi * 5.0 * np.arange(T) / rate))
    assert (tremor <= P.ARM_DWELL_M_S).any() and not (P._moving_max(tremor, n) <= P.ARM_DWELL_M_S).any()
    pause = np.full(T, 0.3)
    pause[80:105] = 0.0
    assert (P._moving_max(pause, n) <= P.ARM_DWELL_M_S).sum() == 25 - 2 * n


# ---------------------------------------------------------------- series on the catalogue
@pytest.mark.parametrize("clip_id", PAIRS)
def test_a_clip_and_its_level0_share_one_layer(P, cat, clip_id):
    s = cat[clip_id]
    root = P.series_root(clip_id, cat)
    b, tr, raw = built(cat, clip_id)
    b0, tr0, raw0 = built(cat, s.level0)
    g, g0 = tr["background"], tr0["background"]
    assert g["series"] == g0["series"] == root and g["seed"] == g0["seed"] == P.series_seed(root)
    assert g["kind"] == g0["kind"] and not g["skipped"] and g["version"] == P.VERSION
    assert g["quiet"] == g0["quiet"] == list(P.SERIES_QUIET.get(root, ()))
    L = P.make_layer(g["seed"], g["kind"], quiet=tuple(g["quiet"]))
    assert g["layer_sha256"] == P.layer_digest(L, b.T) and g0["layer_sha256"] == P.layer_digest(L, b0.T)
    assert g["transfer"]["reference"] == s.level0 and g["transfer"]["rows_corrected_held"] == 0
    assert (s.level0 != root) == (clip_id in CHAINS)
    held, _ = _assert_held(P, raw, raw0, b, b0, clip_id)
    assert held > 0.0, clip_id                                        # every pair here shares rows


@pytest.mark.parametrize("clip_id", PAIRS)
def test_the_symptom_minus_its_level0_is_kept(P, cat, clip_id):
    s = cat[clip_id]
    c, tc, c0 = built(cat, clip_id)
    l, tl, l0 = built(cat, s.level0)
    pc = P.pair_change(c0, c, l0, l)                                     # joints and mesh vertices
    change, sign = pc["change_m"], pc["sign_m"]
    assert sign > 0.0
    assert change <= max(DIFF_FLOOR_M, SHARE_ABOVE.get(clip_id, DIFF_SHARE) * sign), (pc, change / sign)
    T = min(c.T, l.T)
    lay = np.linalg.norm(l.joints().numpy()[:T] - l0.joints().numpy()[:T], axis=-1).max()
    assert lay > 0.0005                                                   # the level 0 moves


# ---------------------------------------------------------------- validity of the wrapped clips
def test_the_clips_physio_v4_refused_pass_by_their_rules(P, cat):
    # the hemiballism: its own anchoring only while the hand rests, the error where it moves reported apart
    _, tr, _ = built(cat, "invol_hemiballism_E1_left_g1")
    g = tr["background"]
    v = g["validity"]
    assert v["ok"] and "own_arms" in g["deviations"]
    assert v["anchor_error_m"] <= P.ANCHOR_TOL_M < v["anchor_error_moving_m"]
    assert v["accel_excess_m_s2"] <= P.ACCEL_TOL_M_S2
    # the knee flexed 30 deg on the level 0's hip row, the foot on its toes: the ball stays where it stands, the ankle
    # takes the residual about it
    _, tr, _ = built(cat, "dystonia_E7_start_distal_leg_knee_flexion_30deg")
    g = tr["background"]
    v = g["validity"]
    assert v["ok"] and "own_legs" in g["deviations"] and g["guard"]["legs_fixed_knots"]["left"] > 0
    assert v["toe_knots"] > 0 and v["foot_moved_m"] <= 1e-6 and v["foot_height_moved_m"] <= 1e-6
    assert g["ankle_error_m"] > 1e-3                                       # the residual is there, about the ball
    assert v["skate_added_m_s"] <= 1e-6
    # the functional axial jerk: the hand jerked onto its support passes; since physio-v7 with the default anchoring
    # (physio-v6 needed the rest gate, its error where the hand stood still 1 mm and more): its wrists, carried where its
    # level 0 holds them, take the level 0's layered place (transfer, HELD_WRIST_TOL)
    _, tr, _ = built(cat, "jerk_functional_E1_axial_sev3_dur2")
    v = tr["background"]["validity"]
    assert v["ok"] and v["anchor_error_m"] <= P.ANCHOR_TOL_M
    assert tr["background"]["guard"]["arm_rest_s"] == 0.0 and any(tr["background"]["guard"]["wrists_held_knots"].values())
    # the hemiballism takes the rest gate (a fallback), the pausing reach the default anchoring
    for cid, rest_s in (("invol_hemiballism_E1_left_g1", P.ARM_REST_S), (REST_SLOW[0], 0.0)):
        g = built(cat, cid)[1]["background"]
        assert g["validity"]["ok"] and g["guard"]["arm_rest_s"] == rest_s, (cid, g["guard"])
    # the pausing reach: its default anchoring turned where the hand moves, but the gate fails, so the default stays
    g = built(cat, REST_SLOW[0])[1]["background"]
    assert any(g["guard"]["arms_fixed_moving_knots"].values())


def test_the_writing_table_takes_no_finger_drift_and_is_read_again_on_the_layer(P, cat):
    # the hand resting on the writing table bears on a curled fingertip, so a finger lift took its lowest point off
    # the table; the table check, in the truth of every writing clip, is read on the layered body
    from mdfm.synth import postures as PO
    cid = "post_W_writing_level0"
    assert cat[cid].scene in P.FINGER_TABLE_SCENES
    b, tr, raw = built(cat, cid)
    g = tr["background"]
    assert g["resting_hands"]["left"] and g["fingers_lift_only"] == []
    assert np.array_equal(b.hand6d.numpy(), raw.hand6d.numpy())
    t = tr["validity"]["table"]
    assert t == PO.table_report_of(b, tr["params"]["table_check"]) and t["ok"]
    assert t["edge_gap_mm"] != PO.table_report_of(raw, tr["params"]["table_check"])["edge_gap_mm"]


def test_a_hand_pausing_on_its_target_keeps_the_default_anchoring(P, cat):
    # the reach pauses with the fingertip on the target block, shorter than the rest gate's ramp: the gate would take
    # the anchoring off that pause and add a miss of the target the raw clip does not have; the default anchoring
    # passes and the gate does not, its anchor check reading the knots where the hand stands still
    cid = "reach_E3_decomposition_icars3"
    b, tr, raw = built(cat, cid)
    g = tr["background"]
    v = g["validity"]
    assert v["ok"] and g["guard"]["arm_rest_s"] == 0.0, g["guard"]
    assert any(g["guard"]["arms_fixed_moving_knots"].values())           # the gate was tried
    assert 0.0 < v["anchor_error_dwell_m"] <= P.ANCHOR_TOL_M
    b0, _, raw0 = built(cat, cat[cid].level0)
    _assert_held(P, raw, raw0, b, b0, cid)


def test_a_passing_anchoring_on_a_moving_hand_takes_the_rest_gate(P, cat):
    # the Holmes tremor shakes the forearm on and off its support (labels flickering every 0.1 to 0.3 s): the default
    # anchoring passes but turned with the moving hand (472 knots of the left arm, share 0.048 of the sign); the rest
    # gate passes too and is kept (none, 0.027), the change within DIFF_SHARE of the sign (the error on the moving
    # hand reported apart). The reach clip since wave SH1d part C: tremor_E1_holmes_g3, the test's clip until then,
    # is levelled at the hand since part B (157a93f), its forearm swings less and its gated anchoring still turns on
    # 815 knots where the hand moves within the gate's ramp (scratchpad measure of part C; its pair share 0.125 is
    # listed by the sweep)
    cid = "tremor_E3_holmes_g3"
    b, tr, raw = built(cat, cid)
    g = tr["background"]
    v = g["validity"]
    assert v["ok"] and g["guard"]["arm_rest_s"] == P.ARM_REST_S, g["guard"]
    # plan phase E (D26, the open body): the gate's anchoring turns on 4 knots of the left arm that the record counts
    # as moving (the gate's weight below 1: arms_fixed_moving_knots), none on SMPL+H; they lie inside the gate's own
    # ramp before a touch (its weight 0.01 to 0.03, the anchoring's at most 0.0013), the wrist moving at 0.236 to
    # 0.242 m/s against its support, under ARM_REST_M_S (scripts/openbody_phaseE_checks.py section holmes): the gate
    # anchors no knot where the hand moves faster than its threshold
    assert sum(g["guard"]["arms_fixed_moving_knots"].values()) <= 4, g["guard"]
    from mdfm.body.joints import fk
    from mdfm.synth.viewer import scene_floor
    l, _, l0 = built(cat, cat[cid].level0)
    kind = g["kind"]
    layer = P.make_layer(g["seed"], kind, quiet=tuple(g["quiet"]))
    trace = {}
    P._transfer(raw, l0, l, layer, scene=cat[cid].scene, table=v["limits_table"], kind=kind,
                resting=g["resting_hands"], rest_s=P.ARM_REST_S, _trace=trace)
    boxes = P.scene_boxes(cat[cid].scene, raw, scene_floor(raw, 5.0)[0])
    labels = P.support_labels(raw, boxes, P.support_joints(kind) + P.TRUNK_TOUCH, approach=False)
    J_in, G_in = fk(raw.local_R(), raw.transl, raw.rest())
    for side, t in trace.items():
        on = t["w"] > 0.0
        assert int(on.sum()) == g["guard"]["arms_fixed_knots"][side], side
        speed = P.wrist_speed(J_in, G_in, labels, side, float(raw.rate))
        assert not np.any(on & (speed > P.ARM_REST_M_S)), side
    assert v["scene_added_m"] <= P.SCENE_TOL_M
    assert v["anchor_error_m"] <= P.ANCHOR_TOL_M and v["anchor_error_moving_m"] > 0.0
    l, _, l0 = built(cat, cat[cid].level0)
    pc = P.pair_change(raw, b, l0, l)
    assert pc["change_m"] <= DIFF_SHARE * pc["sign_m"], pc


def test_a_gate_that_leaves_the_pair_above_the_aim_takes_a_longer_ramp(P, cat):
    """physio-v8 (Phase D of docs/PLAN_2026-11-12.md): tremor_E1_holmes_g3's gated anchoring (ramp ARM_REST_S) turned
    on 815 knots where its hand moves on and off the armrest, the pair's change 0.125 of its sign (the sweep of
    d262de7); the longer ramps of ARM_REST_LONGER_S are tried in turn and the first that passes and keeps the change
    under PAIR_SHARE_AIM is taken (0.4 s here: 0.033 on the scratchpad probe)."""
    # open-body port, wave 8 second part (D167): on Anny the clip's rest gate (ARM_REST_S) already keeps the pair's
    # change under PAIR_SHARE_AIM (its gated anchoring turns on no knot of either arm), so no longer ramp is tried;
    # the clip no longer exercises the longer ramps, whose rule is unchanged (design section 20). The reading on the
    # open body:
    cid = "tremor_E1_holmes_g3"
    b, tr, raw = built(cat, cid)
    g = tr["background"]
    assert g["validity"]["ok"] and g["guard"]["arm_rest_s"] == P.ARM_REST_S, g["guard"]
    assert not any(g["guard"]["arms_fixed_knots"].values())
    assert g["guard"]["pair_share_gate"] <= P.PAIR_SHARE_AIM and "pair_share_gate_short" not in g["guard"]
    l, _, l0 = built(cat, cat[cid].level0)
    assert P.pair_change(raw, b, l0, l)["share"] == pytest.approx(g["guard"]["pair_share_gate"], abs=1e-12)


@pytest.mark.parametrize("clip_id", REFUSED_V2 + REFUSED_V4 + REST_SLOW + SAMPLE)
def test_wrapped_clips_stay_valid(P, cat, clip_id):
    from mdfm.synth.hands import has_hands
    b, tr, raw = built(cat, clip_id)
    g = tr["background"]
    assert not g["skipped"] and g["version"] == P.VERSION
    v = g["validity"]
    assert v["limits_new"] == 0, v
    member = g["transfer"]["reference"] is not None
    assert v["skate_tol_m_s"] == (P.SKATE_TOL_MEMBER_M_S if member else P.SKATE_TOL_M_S)
    assert v["skate_added_m_s"] <= v["skate_tol_m_s"] and (v["contact_lost_share"] or 0.0) == 0.0
    assert v["scene_added_m"] <= P.SCENE_TOL_M and v["body_added_m"] <= P.BODY_TOL_M
    assert v["legs_added_m"] <= P.BODY_TOL_M and v["accel_excess_m_s2"] <= P.ACCEL_TOL_M_S2
    assert v["foot_moved_m"] <= P.FOOT_TOL_M
    assert v["anchor_error_m"] <= P.ANCHOR_TOL_M
    assert v["ok"] and v["failed"] == []
    for side, a in g["anchors"].items():
        assert a["anchor_error_m"] <= P.ANCHOR_TOL_M, (side, a)
    tb = tr["validity"]["background"]                                     # read by the exports (module_failures)
    assert tb == {"ok": True, "failed": [], "version": P.VERSION, "limits_table": v["limits_table"]}
    if g["kind"] == "standing":                                          # the root's legs re-solved in full; a
        if g["transfer"]["reference"] is None:                           # member's own rows (the residual of a
            assert g["ankle_error_m"] <= 1e-6                            # held hip or knee row in height only)
        else:
            # a foot on its toes keeps its ball, the foot checks read there; its ankle turns about it
            limit = 0.01 if v["toe_knots"] else max(1e-3, v["foot_height_moved_m"] + 1e-9)
            assert g["ankle_error_m"] <= limit
    assert torch.equal(b.contact, raw.contact) and b.T == raw.T
    if g["fingers_lift_only"]:                                            # resting hands on a support
        assert has_hands(b) and "fingers" in g["components"]
    assert set(g["fingers_lift_only"]) <= {s for s, r in g["resting_hands"].items() if r}
    assert set(g["components"]) <= set(P.KINDS[g["kind"]])
    for k in ("series", "seed", "kind", "skipped", "kind_of_clip", "amplitudes", "sources", "layer_sha256",
              "knots", "guard", "anchors", "deviations", "touch_share", "resting_hands", "transfer",
              "series_deviations", "fingers_lift_only"):
        assert k in g, k
    json.dumps(g)                                                         # JSON-ready
    lay = np.linalg.norm((b.joints() - raw.joints()).numpy(), axis=-1).max()
    assert 0.0005 < lay < 0.05                                            # alive, by millimetres


def test_the_joint_limit_table_of_a_clip(P, cat):
    # physio-v5: any table of limits_ext a truth names (the healthy shrug's 'shrug' where that module has it) is the
    # one the guard and the checks read
    from mdfm.synth import limits_ext as LX
    for name in LX.TABLES:
        assert P.table_of({"params": {"limits_table": {"table": name}}}) == name
        assert P.table_of({"validity": {"limits_table": name}}) == name
    assert P.table_of({"params": {"limits_table": "no such table"}}) == "healthy" and P.table_of(None) == "healthy"
    for cid, name in (("dystonia_E7_shoulder_elevation_elevation_g2", "dystonia"), ("dystonia_E7_level0", "healthy")):
        _, tr, _ = built(cat, cid)
        assert P.table_of(tr) == name, cid
        assert tr["background"]["validity"]["limits_table"] == name, cid
        assert tr["validity"]["background"]["limits_table"] == name, cid


def test_a_failing_layer_reaches_the_exports(P, cat):
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_for_physio",
                                                  os.path.join(paths.TREE, "scripts", "render_skeleton_catalogue.py"))
    rsc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rsc)
    _, tr, _ = built(cat, "tremor_E1_rest_g0")
    assert "background" not in rsc.module_failures(tr)
    bad = dict(tr, validity=dict(tr["validity"], background=dict(tr["validity"]["background"], ok=False)))
    assert "background" in rsc.module_failures(bad) and "background" in rsc.refusal(bad, module_checks=True)
    v = dict(tr["background"]["validity"], anchor_error_m=2 * P.ANCHOR_TOL_M)
    assert P.failed_checks(v) == ["anchor_error_m"]


def test_the_sweep_reads_a_clips_own_arm_contact(P, cat, tmp_path, monkeypatch):
    # the settling of 2026-11-10: the sweep's record reads a clip's own arms' contact (involuntary.arm_contact, on its
    # build), so that export_ok is false where it refuses the clip, and the summary lists such a clip
    # (invol_choreoathetosis_E1_a3_d3 passed the layer and was refused by it, unseen by the sweep)
    import importlib.util
    spec = importlib.util.spec_from_file_location("synth_background_sweep_for_physio",
                                                  os.path.join(paths.TREE, "scripts", "synth_background_sweep.py"))
    sw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    r = sw.record("invol_td_E1_rocking_s4", cat, P, {})
    assert r["arm_contact"]["ok"] and r["export_ok"] and r["ok"] and len(r["raw_sha256"]) == 64
    assert set(r["arm_contact"]["after_cm_max"]) == {"left", "right"}
    bad = dict(r, arm_contact=dict(r["arm_contact"], ok=False, after_cm_max={"left": 0.95, "right": 0.5}),
               export_ok=False)
    monkeypatch.setattr(sw, "OUT", str(tmp_path))
    with open(tmp_path / "sweep_0of1.json", "w", encoding="utf-8") as f:
        json.dump({"synth_sha256": "x", "physio_version": P.VERSION, "clips": {"a": r, "b": bad}}, f)
    import argparse
    sw.summary(argparse.Namespace(one_digest=False))       # the summary's arguments since wave 7 (--one-digest)
    with open(tmp_path / "summary.json", encoding="utf-8") as f:
        sm = json.load(f)
    assert sm["arm_contact_refused"] == {"b": {"left": 0.95, "right": 0.5}} and sm["export_refused"] == ["b"]
    assert sm["ok"] == 2                                                 # the layer passes both
    with open(tmp_path / "summary.md", encoding="utf-8") as f:
        md = f.read()
    assert "- b: left 0.950, right 0.500" in md and "refused for the export (the layer or the arms' contact): 1" in md


def test_supine_the_spine_does_not_breathe(P, cat):
    b, tr, raw = built(cat, "atax_SARA8_heelshin_g2")
    assert tr["background"]["kind"] == "supine"
    # the layer moves no spine row of its level 0 (supine: breathing only, at the shoulders); a knot where the clip
    # holds a row within EQUAL_ROW_TOL of its level 0's raw row takes the level 0's row, copied (the transfer's
    # identity), which differs from the clip's own by at most that tolerance: spine3 at one knot, 9.1e-10, on the
    # open body (the heel-shin's deviation rising from zero there). Elsewhere the rows are the clip's own, bit for bit
    # (wave 8 second part, D160)
    b0, _, raw0 = built(cat, cat["atax_SARA8_heelshin_g2"].level0)
    T = min(raw.T, raw0.T)
    for j in (0, 3, 6, 9, 12, 15):
        held = torch.from_numpy(P.rows_equal(raw.rot6d[:, j:j + 1], raw0.rot6d[:, j:j + 1], T)[:, 0])
        free = torch.ones(b.T, dtype=torch.bool)
        free[:T] = ~held
        assert torch.equal(b.rot6d[free, j], raw.rot6d[free, j]), j
        if held.any():
            assert float((b.rot6d[:T][held, j] - raw.rot6d[:T][held, j]).abs().max()) <= P.EQUAL_ROW_TOL, j
            assert torch.equal(b.rot6d[:T][held, j], b0.rot6d[:T][held, j]), j
    # the head stays on the couch: 1e-12 m on SMPL+H; on the open body the held spine3 row's copy (above) carries it
    # by 2.7e-10 m at that knot, under the held rows' tolerance times the chain's length (D160)
    assert float((b.joints() - raw.joints())[:, 15].norm(dim=-1).max()) < P.EQUAL_ROW_TOL
    assert float((b.joints() - raw.joints())[:, 16].norm(dim=-1).max()) > 5e-4    # the shoulders breathe


def test_the_seated_hands_rest_on_their_supports(P, cat):
    b, tr, raw = built(cat, "tremor_E1_rest_g0")
    g = tr["background"]
    assert g["kind"] == "seated" and g["resting_hands"] == {"left": True, "right": True}
    assert g["transfer"]["reference"] is None                             # the series root: its own solve
    # physio-v6 (SERIES_QUIET): the rest tremors turn the forearm and the wrist, so the series' hands take no finger
    # drift; the head-sign level 0 keeps it on its resting hands (and takes no head drift)
    assert g["quiet"] == ["fingers_left", "fingers_right"] and g["fingers_lift_only"] == []
    assert "fingers" not in g["components"] and "head" in g["components"]
    from mdfm.synth import hands as HD
    assert torch.equal(_hand6d(b), _hand6d(raw))
    bd, trd, rawd = built(cat, "dystonia_E1_level0")
    gd = trd["background"]
    assert gd["quiet"] == ["head"] and gd["fingers_lift_only"] == ["left", "right"]    # resting fingers only lift
    assert "head" not in gd["components"] and "fingers" in gd["components"]
    fl = np.array([HD.hand_components(x.hand_R().numpy())[HD.hand_joint("left", "index", 0)][0]
                   for x in (bd, HD.hands_of(rawd))])
    d = fl[0] - fl[1]
    assert float(d.max()) <= 1e-6 < float(-d.min())                      # never flexed beyond the posture
    # physio-v4: no constant offset (v3 held the resting fingers open by the drift's horizon maximum): about half the
    # time the fingers rest within the lift's softness of their posture, read on the series' lift over the layer's
    # horizon (this level 0 lasts 15 s, a few periods of the drift's slowest band)
    lift = P.finger_lift(P.make_layer(gd["seed"], gd["kind"], quiet=tuple(gd["quiet"]))["fingers_deg"][:, 0, 0])
    assert float((lift >= -P.FINGER_LIFT_SOFT_DEG).mean()) > 0.3 and float(np.median(lift)) > -P.FINGER_LIFT_SOFT_DEG
    for a in g["anchors"].values():
        assert a["anchored_share"] == 1.0 and a["supports"] == [-2]       # the forearms on the armrests
    w = (b.joints() - raw.joints()).norm(dim=-1)
    assert float(w[:, [20, 21]].max()) < 1e-9 < float(w[:, 15].max())    # the wrists stay, the head moves
    _, tr2, _ = built(cat, "tremor_E2_postural_g2")
    assert tr2["background"]["resting_hands"] == {"left": False, "right": False}   # held outstretched
    _, tr3, _ = built(cat, "brady_E4a_ft_decrement_g1")
    assert tr3["background"]["resting_hands"]["left"] is False                     # the tapping hand
    _, tr4, _ = built(cat, "post_E7_cd_geste_C1")
    assert tr4["background"]["touch_share"] > 0.0 and 15 in tr4["background"]["anchors"]["right"]["supports"]


def test_a_series_quiet_part_is_still_in_its_whole_series(P, cat):
    # physio-v6 (SERIES_QUIET, the session's decision of 2026-11-10): the quiet parts are zero in the series' layer,
    # every other signal the full layer's bit for bit; a hand on which a member turns the forearm, the wrist or the
    # fingers takes no finger drift and a head carrying a member's sign no head drift, so the pair stays under the
    # owner's share (the rest tremor at 0.125 of the sign and the pitch head tremor at 0.052 before)
    seed = P.series_seed("tremor_E1_rest_g0")
    full = P.make_layer(seed, "seated")
    for quiet in (("fingers_left",), ("fingers_left", "fingers_right"), ("head",)):
        q = P.make_layer(seed, "seated", quiet=quiet)
        for k in full:
            if k == "fingers_deg":
                for si, side in enumerate(P.ARM):
                    want = np.zeros_like(full[k][:, si]) if f"fingers_{side}" in quiet else full[k][:, si]
                    assert np.array_equal(q[k][:, si], want), (quiet, side)
            elif k in ("head_yaw_deg", "head_pitch_deg") and "head" in quiet:
                assert not np.any(q[k]), k
            else:
                assert np.array_equal(q[k], full[k]), (quiet, k)
    with pytest.raises(ValueError):
        P.make_layer(seed, "seated", quiet=("trunk",))
    roots = {P.series_root(c, cat) for c in cat}
    for r, parts in P.SERIES_QUIET.items():
        assert r in roots and cat[r].level0 is None and set(parts) <= set(P.QUIET_PARTS), r
        assert P.skip_reason(cat[r]) is None, r
    for cid, quiet in (("tremor_E1_rest_g2", ["fingers_left", "fingers_right"]), ("tremor_E1_head_pitch_g2", ["head"]),
                       ("tremor_E1_pillrolling_supination_g2", ["fingers_left"])):
        c, tc, c0 = built(cat, cid)
        l, tl, l0 = built(cat, cat[cid].level0)
        assert tc["background"]["quiet"] == tl["background"]["quiet"] == quiet
        pc = P.pair_change(c0, c, l0, l)
        assert pc["share"] <= DIFF_SHARE, (cid, pc)


# ---------------------------------------------------------------- the spans of the layer (physio-v7)
@pytest.fixture(scope="module")
def sweep():
    import importlib.util
    spec = importlib.util.spec_from_file_location("synth_background_sweep_for_spans",
                                                  os.path.join(paths.TREE, "scripts", "synth_background_sweep.py"))
    sw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    return sw


def test_the_span_weights(P):
    # wave SH1d part C: 1 on a still run, a raised cosine from 0 over STILL_FADE_S inside it at an edge the body
    # moves across, none at the clip's ends; a pause shorter than its fades takes none; a freezing hold fades at both
    # ends, its ramps unlayered
    r = P.run_ramp(100, True, True, 30)
    assert r[0] == 0.0 and r[-1] == 0.0 and r[30:70].min() == 1.0 and np.all(np.diff(r[:31]) > 0.0)
    assert np.abs(np.diff(r)).max() < math.pi / 30
    m = int(round(P.STILL_FADE_S * 100))
    w = P.weight_of_runs([(0, 80), (200, 260), (400, 600)], 600, 100.0)
    assert np.all(w[:80 - m] == 1.0) and w[79] == 0.0                     # the clip's start, a fade before the walk
    assert not np.any(w[80:400])                                          # 0.6 s between two walks: none
    assert w[400] == 0.0 and np.all(w[400 + m:] == 1.0)                   # a fade after the walk, the clip's end
    tr = {"params": {"episodes": [{"ramp_in_s": [1.0, 1.5], "hold_s": [1.5, 4.0], "ramp_out_s": [4.0, 4.5],
                                   "manifestation": "akinesia", "context": "straight"}]}}
    h = P.hold_weight(tr, 600, 100.0)
    assert not np.any(h[:151]) and not np.any(h[399:]) and np.all(h[150 + m:400 - m] == 1.0)
    rec = P.span_record(h, 100.0)
    assert rec["spans_s"] == [[1.51, 3.99]] and 0.0 < rec["full_share"] < rec["share"]


def test_a_walk_takes_the_layer_on_its_still_spans_only(P, sweep):
    # M3a of the review (wave SH1d part C): a walk stands before its first step and after its last; the standing layer
    # there (breathing, the head drift, the sway), crossfaded, the walk itself unlayered bit for bit; the stance after
    # a 90 deg turn sways too (its pelvis read in its heading's frame by the sway's guard)
    from mdfm.synth.base import walk
    b, _ = walk(12.0, stop_at_s=8.0, turns=[(3.0, 90.0, 2.0)])
    runs = P.still_runs(b)
    assert len(runs) == 2 and runs[0][0] == 0 and runs[-1][1] == b.T
    w = P.still_weight(b)
    layer = P.make_layer(P.series_seed("a walk"), "walking")
    out, rec = P.solve(b, layer, kind="walking", resting={"left": False, "right": False}, span=w)
    v = rec["validity"]
    assert v["ok"] and v["failed"] == [] and rec["components"] == ["breath", "head", "sway"]
    assert rec["guard"]["sway_scaled_knots"] == 0 and not rec["deviations"]
    z = torch.from_numpy(w <= 0.0)
    assert torch.equal(out.rot6d[z], b.rot6d[z]) and torch.equal(out.transl[z], b.transl[z])
    d = (out.joints() - b.joints()).norm(dim=-1).numpy()
    end = slice(runs[-1][0] + int(round(P.STILL_FADE_S * 100)), b.T)
    assert d[end, 0].max() > 1e-3 and d.max() < 0.03                     # the pelvis sways, by millimetres
    assert not sweep.statue_of(b)["ok"] and sweep.statue_of(out)["ok"]
    assert rec["span"]["spans_s"][0][0] == 0.0 and rec["span"]["keys"] == list(P.FADED)


def test_the_walking_series_carry_one_layer(P, cat, sweep):
    # M3a on the catalogue: a walking clip and its level 0, each its own solve on its own still spans, the knots the
    # two hold equal in full where their spans weigh alike the level 0's output bit for bit, no statue left, the
    # pair's change within the owner's share
    cid = WALK_PAIR
    b, tr, raw = built(cat, cid)
    b0, tr0, raw0 = built(cat, cat[cid].level0)
    for x, t, r in ((b, tr, raw), (b0, tr0, raw0)):
        g = t["background"]
        assert g["kind"] == "walking" and g["validity"]["ok"] and not g["skipped"], g["validity"]["failed"]
        w = P.still_weight(r)
        z = torch.from_numpy(w <= 0.0)
        assert torch.equal(x.rot6d[z], r.rot6d[z]) and torch.equal(x.transl[z], r.transl[z])
        assert w.max() == 1.0 and not sweep.statue_of(r)["ok"] and sweep.statue_of(x)["ok"]
    assert tr["background"]["transfer"]["level0"] == cat[cid].level0
    T = min(raw.T, raw0.T)
    w, w0 = P.still_weight(raw)[:T], P.still_weight(raw0)[:T]
    same = (w == w0) & (w > 0.0)                                          # outside the spans each clip is its own
    whole = P.rows_equal(raw.rot6d, raw0.rot6d, T).all(-1) & same & (
        (raw.transl[:T] - raw0.transl[:T]).abs().amax(-1) <= P.EQUAL_ROW_TOL).numpy()
    k = torch.from_numpy(whole)
    assert k.any() and torch.equal(b.rot6d[:T][k], b0.rot6d[:T][k]) and torch.equal(b.transl[:T][k], b0.transl[:T][k])
    assert tr["background"]["transfer"]["knots_held_share"] == pytest.approx(float(whole.mean()))
    pc = P.pair_change(raw, b, raw0, b0)
    assert pc["share"] <= DIFF_SHARE, pc


def test_the_tetras9_series_takes_the_layer(P, cat, sweep):
    # M3c of the review: TETRAS 9 left SKIP once its grade 3 swung beyond the layered level 0 (part B: an ankle sway
    # of 9 cm at the head); under the layer the level 0 is no statue, every grade keeps its sign (the pair's change
    # within the owner's share, the rows held equal the level 0's) and grade 3's head travels forward and backward at
    # least twice the layered level 0's
    b0, t0, r0 = built(cat, "standing_tremor_E10_posture_g0")
    assert t0["background"]["kind"] == "standing" and t0["background"]["validity"]["ok"]
    assert not sweep.statue_of(r0)["ok"] and sweep.statue_of(b0)["ok"]
    head0 = float(np.ptp(b0.joints().numpy()[:, 15, 2]))
    for g in (1, 2, 3):
        b, t, r = built(cat, f"standing_tremor_E10_tetras9_g{g}")
        assert t["background"]["validity"]["ok"], t["background"]["validity"]["failed"]
        assert P.pair_change(r, b, r0, b0)["share"] <= DIFF_SHARE
        _assert_held(P, r, r0, b, b0, g)
    assert float(np.ptp(b.joints().numpy()[:, 15, 2])) >= 2.0 * head0


def test_the_transfer_sways_in_its_standing_phase(P, cat):
    # wave SH1d part C: the arising's layer sways once the body stands (stand_weight: its still runs at the pelvis's
    # top), the seated phase and the rise keep the pelvis, the legs and the translation bit for bit
    b, tr, raw = built(cat, "ref_arise_E7")
    g = tr["background"]
    assert g["kind"] == "transfer" and "sway" in g["components"] and g["validity"]["ok"]
    assert g["span"]["keys"] == ["sway_ap_m", "sway_ml_m"] and len(g["span"]["spans_s"]) == 1
    w = P.stand_weight(raw)
    z = torch.from_numpy(w <= 0.0)
    legs = [0, 1, 2, 4, 5, 7, 8]
    assert torch.equal(b.rot6d[z][:, legs], raw.rot6d[z][:, legs]) and torch.equal(b.transl[z], raw.transl[z])
    J, J0 = b.joints().numpy(), raw.joints().numpy()
    on = w >= 1.0
    assert on.any() and np.linalg.norm(J[on, 0] - J0[on, 0], axis=-1).max() > 2e-3
    assert J0[on, 0, 1].min() >= J0[:, 0, 1].max() - P.STAND_HIGH_M


# ---------------------------------------------------------------- skipped series
def test_the_skipped_series_keep_their_bodies(P, cat):
    roots = {P.series_root(c, cat) for c in cat}
    assert set(P.SKIP) <= roots
    cache = next(iter(cat.values())).build.cache
    for r in P.SKIP:
        c = cache.context(r)                                              # never built: no root body needed
        assert c["skip"] == P.SKIP[r] and "resting" not in c
    for s in cat.values():                                                # physio-v7: the walks take the layer
        r = P.series_root(s.clip_id, cat)                                 # on their still spans; the freezing
        if P.posture_kind(cat[r]) == "walking":                           # clips' walk lays it in its own build
            assert (P.skip_reason(cat[r]) is not None) == (r == P.FREEZING_ROOT), s.clip_id   # (wave 7)
    assert "standing_tremor_E10_posture_g0" not in P.SKIP                # TETRAS 9: its grade 3 exceeds the layer
    for cid in ("atax_ROMB_cerebellar", "park_E1_motorstate_offon", "atax_SARA2_natural_g3", "atax_TITUB_g2"):
        raw, truth = cat[cid].build.raw()
        b, tr = cat[cid].build.with_background(raw, truth)
        assert tr["background"]["skipped"] and tr["background"]["reason"]
        assert torch.equal(b.rot6d, raw.rot6d) and torch.equal(b.transl, raw.transl)


def test_the_parkinsonism_series_left_wrapped_carry_no_background(P, cat):
    from mdfm.synth import parkinsonism as PK
    roots = {P.series_root(c, cat) for c, s in cat.items() if s.family == "parkinsonism"}
    wrapped = {r for r in roots if r not in P.SKIP}
    assert wrapped == {"park_E4aL_sequence_ft_level0", "park_E4L_sequence_ps_level0", "park_E4a_ulfreeze_level0",
                       "park_E4aL_ulfreeze_level0"}
    for fn in (PK.sequence_effect, PK.upper_limb_freezing):
        assert "background_signals" not in inspect.getsource(fn)
    for r in roots - wrapped:
        assert "own physiological background" in P.SKIP[r]
    atax = {P.series_root(c, cat) for c, s in cat.items() if s.family == "ataxia"} - set(P.SKIP)
    assert atax == {"atax_SARA7_ddk_g0", "atax_SARA8_heelshin_g0", "atax_SARA8_heelshin_right_g0"}


# ---------------------------------------------------------------- determinism and wrapping
def test_determinism(P):
    from mdfm.synth import catalogue as C
    from mdfm.synth.common import truth_json
    out = []
    for _ in range(2):
        P.make_layer.cache_clear()
        by = {s.clip_id: s for s in C.catalogue_set()}
        b, tr = by["jerk_myoclonus_E7_propriospinal_f2_a3"].build()
        out.append((b, truth_json(tr["background"])))
    (b1, t1), (b2, t2) = out
    assert torch.equal(b1.rot6d, b2.rot6d) and torch.equal(b1.transl, b2.transl)
    assert torch.equal(_hand6d(b1), _hand6d(b2)) and t1 == t2


def test_the_exam_set_is_not_wrapped(P, cat):
    from mdfm.synth import catalogue as C
    from mdfm.synth import exam as EX
    assert not any(isinstance(s.build, P.LayeredBuild) for s in EX.exam_set())
    assert all(isinstance(s.build, P.LayeredBuild) for s in cat.values())
    plain = {s.clip_id: s for s in C.catalogue_set(background=False)}
    assert not any(isinstance(s.build, P.LayeredBuild) for s in plain.values())
    b, _ = plain["dystonia_E7_level0"].build()
    raw, _ = cat["dystonia_E7_level0"].build.raw()
    assert torch.equal(b.rot6d, raw.rot6d)
    for cid, mod in (("tremor_E3_kinetic_g2", "mdfm.synth.tremors"), ("invol_chorea_E1_g2", "mdfm.synth.involuntary"),
                     ("dystonia_E7_level0", "mdfm.synth.exam")):
        fn = getattr(cat[cid].build, "func", cat[cid].build)
        fn = getattr(fn, "func", fn)
        assert fn.__module__ == mod, cid                                  # the code records name the clip's module


def test_the_posture_kinds(P, cat):
    ids = ("tremor_E1_rest_g0", "atax_SARA8_heelshin_g0", "ref_arise_E7", "gait_E8_level0", "dystonia_E7_level0",
           "post_W_writing_level0", "tremor_E3_reach_left_g0", "gait_E7p_retropulsion_g0")
    kinds = {cid: P.posture_kind(cat[cid]) for cid in ids}
    assert kinds == {"tremor_E1_rest_g0": "seated", "atax_SARA8_heelshin_g0": "supine", "ref_arise_E7": "transfer",
                     "gait_E8_level0": "walking", "dystonia_E7_level0": "standing", "post_W_writing_level0": "seated",
                     "tremor_E3_reach_left_g0": "seated", "gait_E7p_retropulsion_g0": "walking"}
    for cid, s in cat.items():                                            # a clip's kind is its series' kind
        if cid in AMASS:
            continue
        assert P.posture_kind(s) == P.posture_kind(cat[P.series_root(cid, cat)]), cid


def test_the_layered_pairs_are_no_statues(P, cat, sweep):
    # the statue check of lane 1 (scripts/render_skeleton_catalogue.check_statue, M3) on every clip and level 0 pair
    # built here (the sweep reads every layered clip: reports/synth/background/summary.md)
    ids = PAIRS + SAMPLE + [WALK_PAIR] + [f"standing_tremor_E10_tetras9_g{g}" for g in (1, 2, 3)]
    for cid in ids:
        for c in (cid, cat[cid].level0):
            if c is None:
                continue
            b, tr, _ = built(cat, c)
            if tr["background"]["kind"] != "supine":
                assert sweep.statue_of(b)["ok"], c


def test_the_scene_and_fold_checks_are_read_again_on_the_layered_body(P, cat):
    """Phase D of docs/PLAN_2026-11-12.md: the layered truth copied 'fold', 'hand_thigh', 'hand_trunk' and 'backrest'
    from the raw body (only the writing clips' 'table' was read again); layer_checks_of reads them on the layered
    body: the arising level 0 through its phases, the paroxysmal and geste clips through params['layer_checks']
    (postures.scene_checks_record), and nothing for a clip without them."""
    from mdfm.synth import motions as MO
    from mdfm.synth import postures as PO
    b, tr, raw = built(cat, "ref_arise_E7")
    assert tr["background"]["checks_read_on_layer"] == ["armrest", "fold", "hand_thigh"]
    again = MO.arise_checks_of(b, MO.arise_checks_record(tr["params"]))
    assert again["fold"] == tr["validity"]["fold"] and tr["validity"]["fold"]["ok"]       # read on the layered body
    assert MO.arise_checks_of(raw, MO.arise_checks_record(tr["params"]))["fold"]["ok"]
    assert again["fold"]["n_knots"] > 0 and tr["validity"]["armrest"]["ok"] and tr["validity"]["hand_thigh"]["ok"]
    s15 = PO.H.hands_of(PO.static_body("stand", 0.1))
    rec = PO.scene_checks_record("geste", side="right", knots=[0, 5])
    got = P.layer_checks_of(s15, {"params": {"layer_checks": rec}})
    assert set(got) == {"hand_head", "hand_trunk"} and got["hand_head"]["n_knots"] == 2
    assert P.layer_checks_of(s15, {"operator": "tremor", "params": {}}) == {}
    with pytest.raises(ValueError):
        PO.scene_checks_record("table")



def test_the_mesh_caches_are_keyed_on_the_body_model(P, monkeypatch):
    """Open-body port, wave 7 (design section 16.7): physio cached its skinned vertex sets on nothing, so that in one
    process an open-body read followed by a read on the SMPL+H reference took the open body's vertex ids (an
    IndexError of the exam's freezing clip on the reference). Each mesh cache is keyed on the selected model's name:
    the reference's ids index its own mesh, and the open body's come back from their own entry unchanged."""
    from mdfm.synth import openbody as OB
    from mdfm.synth import viewer as V
    monkeypatch.delenv(OB.BACKEND_ENV, raising=False)
    open_ids = P._probe_ids()
    n_open = len(V.rest_vertices())
    assert P._model() == OB.OPEN and open_ids["body"].max() < n_open and len(P._faces()) == len(V.faces())
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")
    ref_ids = P._probe_ids()
    n_ref = len(V.rest_vertices())
    assert P._model() == OB.SMPLH_REFERENCE and n_ref != n_open
    assert ref_ids["body"].max() < n_ref and ref_ids is not open_ids
    assert int(P._body_surface()[2].shape[0]) == n_ref
    monkeypatch.delenv(OB.BACKEND_ENV)
    assert P._probe_ids() is open_ids and int(P._body_surface()[2].shape[0]) == n_open
