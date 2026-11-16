"""Tests of the parametric gait generator of the symptom catalogue, mdfm/synth/gait.py (task CAT-GAIT):

- the healthy walk of NORMAL lies inside the normative ranges (gait.healthy_ranges: speed, cadence, step length
  and width, stance share, knee flexion in loading and in swing, minimum foot clearance, arm swing, elbow flexion,
  trunk lean, step variability), read back by measure_gait from the body alone;
- contacts are exact: the contact point of every stance foot does not slip (stance_slip), no sole goes below the
  floor in swing beyond the tolerance, every joint lies inside the SP1 joint-limit table, no NaN;
- determinism: the same seed gives the same body bit for bit, another seed another one;
- each pathology's realized parameters follow its grade: MDS-UPDRS 3.10 grades 0 to 2 (step length, speed, arm
  swing and its asymmetry, trunk lean, foot clearance; the realized step lengths at the asked ones), SARA gait 1 to
  3 (step variability and base width rising), tandem footprints on the line and off it, steppage (the foot hangs,
  the hip and knee lift, the knee near the cited CMT1A value), hemiparesis (the stiff paretic knee), antalgic stance
  asymmetry and the guarded painful swing, retropulsion step counts;
- the swing lift: low flat-footed gaits swing both knees alike (the clearance asked where the lift acts, the longer
  right leg's pelvic side hiked in its swing), no healthy swing knee far above the others, the dip lift localized;
- sudden knee buckles fall in a stance of their leg, drop the pelvis in about 0.13 s and fling the arms; the arms
  take back the trunk's lateral lean; the waddling gait's lumbar hyperlordosis; a turn widens the inside footprint;
- the overlay hook: an arm overlay changes only that joint; a swing-only leg overlay keeps the stance feet planted;
- every catalogue spec is unique and its level-0 partner exists; a sample of specs (one per family) passes the
  generator's validity checks, the choreic clip's chorea coming from the involuntary family's engine;
- scissoring (shared change SH8): the spastic diplegic walk lands its feet on the midline, some past it (the signed
  step widths, read back without bias: the healthy walk's footprints lie half its step width from the midline), and
  brings its knees together, touching at most: the legs check holds (legcontact), the swing hips adduct in mid
  swing, no swing hip is thrown outward beyond its toe-off and heel-strike values more than the healthy walk's, the
  guard adds little; a stronger adduction passes the legs into each other without the guard and not with it; the
  base widens inside the turn only; the milder level keeps a wider base and the knees further apart; the guard is
  on by default (since 2026-11-03) and the gait validity carries the legs entry;
- the leg-to-leg calibration (2026-11-03): the healthy clips (the level-0 course, the tandem of SARA 0, the pull
  test's two steps) keep their legs apart within the legcontact tolerances, their distal band's reading (in the
  record only) no deeper than pinned; the former healthy course did not, and the leg failure alone refuses it
  (gait.ok); the double-support guard sets a landing foot wider; the post-overlay guard bumps a swing that a leg
  overlay passed into the other leg.
- wave SH1d: the continuity check (contact jumps, the floor skim, the knee flexion speed, the joint acceleration)
  and its positive controls; a turning stance foot pivots on one sole point; the reviewed snaps are gone (the
  shuffling releases, the bilateral steppage's turn among them); the start hesitation's steps in place stay low.
Procedural bodies in the neutral shape only; no store, lockbox, EVAL, CODY or care_pd record is read.
"""
import os
from dataclasses import replace as replace_

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


@pytest.fixture(scope="module")
def GA():
    torch.set_num_threads(2)
    from mdfm.synth import gait
    return gait


@pytest.fixture(scope="module")
def healthy(GA):
    b, info = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)
    return b, info, GA.measure_gait(b)


@pytest.fixture(scope="module")
def pd(GA):
    out = {}
    for g in (0, 1, 2):
        p, _ = GA.parkinsonian(g)
        b, info = GA.generate(p, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=11 + g)
        out[g] = (b, info, GA.summary(GA.measure_gait(b)))
    return out


def test_the_healthy_walk_lies_inside_the_normative_ranges(GA, healthy):
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, the stance share near 0.60 (perry2010):
    # kept; the read-back's stance share and double support read the generator's touch at STANCE_READ_TOL_M,
    # re-derived on Anny's sole (the toe's rest clearance 0.15 mm, its rise 0.68 mm on the first knot after toe-off),
    # and end at the first lift-off (D103, D104): 0.61 and 0.611; passes
    b, info, m = healthy
    s = GA.summary(m)
    assert m["n_steady"] >= 6
    for key, (lo, hi, src) in GA.healthy_ranges().items():
        assert s[key] is not None and lo <= s[key] <= hi, (key, s[key], lo, hi, src)


def test_the_healthy_walk_reads_back_its_own_parameters(GA, healthy):
    b, info, m = healthy
    s = GA.summary(m)
    p = GA.NORMAL
    assert abs(s["step_length_m"] / p.step_m - 1) < 0.03
    assert abs(s["cadence_spm"] / p.cadence_spm - 1) < 0.03
    assert abs(s["speed_m_s"] / (p.step_m * p.cadence_spm / 60.0) - 1) < 0.04
    assert abs(s["step_width_m"] - p.step_width_m) < 0.015
    for side in GA.SIDES:
        assert GA.pair(p.clearance_m)[0] - 0.004 < s[f"clearance_{side}_m"] < 0.03
        assert s[f"contact_pitch_{side}_deg"] > 8.0                      # heel strike with the toes up


def test_contacts_are_exact_and_joints_stay_inside_the_table(GA, healthy):
    from mdfm.synth.body import limit_violations
    b, info, _ = healthy
    sl = GA.stance_slip(b, info)
    assert sl["n"] > 500 and sl["p95_m_s"] < 1e-3
    assert limit_violations(b.rot6d)["outside"] == 0
    v = GA.validity(b, info)
    assert v["gait"]["ok"] and v["swing_lowest_m"] > -GA.PENETRATION_TOL_M
    assert torch.isfinite(b.rot6d).all() and torch.isfinite(b.transl).all()
    # the body starts and ends standing with both feet down, flat
    assert bool(info["stance"][0].all()) and bool(info["stance"][-1].all())
    assert np.abs(info["pitch_deg"][[0, -1]]).max() < 1e-9


def test_the_slip_check_sees_a_sliding_stance_foot(GA, healthy):
    """Positive control: the whole body slid sideways at 0.1 m/s reads that speed at the contact point."""
    from dataclasses import replace
    b, info, _ = healthy
    tr = b.transl.clone()
    tr[:, 0] += 0.1 * b.times()
    sl = GA.stance_slip(replace(b, transl=tr), info)
    assert abs(sl["p95_m_s"] - 0.1) < 0.01


def test_the_generator_is_deterministic(GA):
    a, _ = GA.generate(GA.NORMAL, duration_s=5.0, seed=3)
    b, _ = GA.generate(GA.NORMAL, duration_s=5.0, seed=3)
    c, _ = GA.generate(GA.NORMAL, duration_s=5.0, seed=4)
    assert torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)
    assert not torch.equal(a.transl, c.transl)


def test_parkinsonian_grades_follow_the_grade_map(GA, pd):
    s = {g: pd[g][2] for g in pd}
    for g in (1, 2):
        p, _ = GA.parkinsonian(g)
        assert abs(s[g]["step_length_m"] / p.step_m - 1) < 0.06, g
    assert s[0]["step_length_m"] > s[1]["step_length_m"] > s[2]["step_length_m"]
    assert s[0]["speed_m_s"] > s[1]["speed_m_s"] > s[2]["speed_m_s"]
    # zanardi2021 at grade 1: stride about 0.16 m shorter, speed about 0.15 m/s slower, cadence not lower
    assert 0.10 < 2 * (s[0]["step_length_m"] - s[1]["step_length_m"]) < 0.22
    assert 0.08 < s[0]["speed_m_s"] - s[1]["speed_m_s"] < 0.25
    assert s[1]["cadence_spm"] >= s[0]["cadence_spm"] - 1.0
    # the more affected (left) arm swings less, the asymmetry rises, the trunk stoops, the feet lift less
    assert s[0]["shoulder_range_left_deg"] > s[1]["shoulder_range_left_deg"] > s[2]["shoulder_range_left_deg"]
    assert s[2]["shoulder_range_left_deg"] < s[2]["shoulder_range_right_deg"]
    assert s[0]["arm_swing_asymmetry_pct"] < s[1]["arm_swing_asymmetry_pct"] < s[2]["arm_swing_asymmetry_pct"]
    assert s[0]["trunk_lean_deg"] < s[1]["trunk_lean_deg"] < s[2]["trunk_lean_deg"]
    assert s[0]["clearance_left_m"] > s[1]["clearance_left_m"] and s[0]["clearance_left_m"] > s[2]["clearance_left_m"]
    assert s[0]["contact_pitch_left_deg"] > s[2]["contact_pitch_left_deg"]     # less heel strike
    for g in pd:
        b, info, _ = pd[g]
        assert GA.validity(b, info)["gait"]["ok"], g
    with pytest.raises(ValueError):
        GA.parkinsonian(3)


def test_ataxic_variability_and_base_rise_with_the_sara_grade(GA):
    """SARA gait 2 and 3 widen the base and the step variability; SARA 1 is a near-normal free walk (its difficulty
    shows in tandem walking, sara): its base within 2 cm of the healthy one and its width SD under 2 cm (the former
    SARA 1 walk, 3.5 cm wider with a 2.7 cm SD, read as SARA 2 in the 2026-11-03 review)."""
    rows = {}
    for g in (0, 1, 2, 3):
        p, _ = GA.ataxic(g)
        b, info = GA.generate(p, duration_s=16.0, start_at_s=1.0, seed=21)
        rows[g] = GA.summary(GA.measure_gait(b))
    assert rows[0]["step_width_m"] < rows[2]["step_width_m"] < rows[3]["step_width_m"]
    assert rows[0]["step_length_cv"] < rows[2]["step_length_cv"] < rows[3]["step_length_cv"]
    assert rows[0]["step_width_sd_m"] < rows[3]["step_width_sd_m"]
    assert abs(rows[1]["step_width_m"] - rows[0]["step_width_m"]) < 0.02 and rows[1]["step_width_sd_m"] < 0.02
    assert rows[1]["step_width_m"] < rows[2]["step_width_m"]


def test_tandem_footprints_lie_on_the_line_or_off_it(GA):
    for g in (0, 1):
        p = GA.tandem(g)
        b, info = GA.generate(p, duration_s=GA.TANDEM_DURATION_S, start_at_s=GA.TANDEM_START_S, seed=5)
        lat = [abs(f["ankle"][0]) for s in GA.SIDES for f in info["footprints"][s]
               if not f["first"] and not f["final"]]
        if g == 0:
            assert max(lat) < 0.02
        else:
            assert sum(x > 0.06 for x in lat) == len(GA.TANDEM_MISSTEPS[1])


def test_steppage_hemiparetic_and_antalgic_show_their_sides(GA):
    # open-body kind (wave 5 second follow-up, D26): (a) the 78 deg upper bound was set on the SMPL+H body's realized
    # knee about park2023's 70.5; on Anny the dropped foot's clearance lift sets the knee (78.14 deg at the former
    # asks). Wave 8 second part (D161): the bound kept, the motion fixed: the dropped side asks STEPPAGE_KNEE_DEG and
    # STEPPAGE_HIP_DEG, its knee near 74.7 deg (design section 20)
    from mdfm.synth.body import components_deg
    b, info = GA.generate(GA.steppage(("left",)), duration_s=8.0, start_at_s=1.0, seed=2)
    sw = info["swing_weight"]
    comp = components_deg(b.rot6d)
    mid = sw[:, 0] > 0.9
    # the left foot hangs plantarflexed in mid swing, more than the right one
    assert comp[mid, 7, 0].mean() > 8.0 and comp[mid, 7, 0].mean() > comp[sw[:, 1] > 0.9, 8, 0].mean() + 10.0
    s = GA.summary(GA.measure_gait(b))
    # the knee and the hip lift higher on the dropped side, the knee near the 70.5 deg of moderate CMT1A (park2023),
    # not the 89 deg of the former values whose heel flicked back
    assert s["knee_swing_left_deg"] > s["knee_swing_right_deg"] + 3.0 and 64.0 < s["knee_swing_left_deg"] < 78.0
    assert (-comp[sw[:, 0] > 0, 1, 0]).max() > (-comp[sw[:, 1] > 0, 2, 0]).max() + 5.0     # hip flexion
    b, info = GA.generate(GA.hemiparetic("left"), duration_s=10.0, start_at_s=1.0, seed=2)
    s = GA.summary(GA.measure_gait(b))
    assert s["knee_swing_left_deg"] < s["knee_swing_right_deg"] - 10.0
    assert s["shoulder_range_left_deg"] < 5.0                      # the held arm does not swing (trunk motion only)
    b, info = GA.generate(GA.antalgic("left"), duration_s=10.0, start_at_s=1.0, seed=2)
    s = GA.summary(GA.measure_gait(b))
    assert s["stance_share_left"] < s["stance_share_right"] - 0.08
    # the painful leg swings guarded: its knee near the 40 deg asked, well below the sound side's (the toe dragging
    # after toe-off used to raise it to about 68 deg, a high step). Open-body kind (wave 8 second part, D172): (b), a
    # clinical property, kept; on Anny the painful knee read 66.7 deg with the ankle at 12 deg, the motion fixed
    # (ANTALGIC_ANKLE_DEG)
    assert s["knee_swing_left_deg"] < 48.0 and s["knee_swing_left_deg"] < s["knee_swing_right_deg"] - 12.0


def test_low_flat_footed_swings_are_symmetric(GA, pd):
    """The clearance lift of a low, flat-footed gait is asked where it acts (CLEAR_CORE) and the longer right leg's
    pelvic side hikes while it swings (leg_length_comp): both swing knees alike (the walking-on-ice gait read 38 deg
    on the left and 56 on the right)."""
    # open-body kind (wave 5 second follow-up, D26): (a) the 4.0 deg symmetry band of the grade 2 swing knees was read
    # on SMPL+H; the leg-length part is re-derived (D78); the swing knees read 4.23 deg apart. Wave 8 second part
    # (D162): the band kept, the motion fixed: the low flat-footed swings dorsiflex the ankle early to
    # LOW_FLAT_ANKLE_DEG (and the grade 2 walk's dip lift by the ankle first), so that the clearance lift no longer
    # doubles at the knee (design section 20)
    g = GA.geometry()
    # wave 5 follow-up (D26): the SMPL+H rest's right leg reached about 3 mm lower, which leg_length_comp hiked away;
    # the open body's canonical rest is mirror-symmetric (section 3.2), its legs reach alike, so nothing is hiked
    assert abs(g.leg_length_diff_m) < 1e-9 and abs(g.obliquity_comp_deg) < 1e-6
    s = pd[2][2]
    assert abs(s["knee_swing_left_deg"] - s["knee_swing_right_deg"]) < 4.0
    # 14 s: the slow walk's stop lands at 12.5 s (gait.validity refuses a clip that cuts the stop's last step)
    b, info = GA.generate(GA.functional("walking_on_ice"), duration_s=14.0, start_at_s=1.0, stop_at_s=10.0, seed=4)
    s = GA.summary(GA.measure_gait(b))
    assert abs(s["knee_swing_left_deg"] - s["knee_swing_right_deg"]) < 4.0 and s["knee_swing_right_deg"] < 45.0
    assert GA.validity(b, info)["gait"]["ok"]
    # no swing knee of the healthy walk lies far above the others (a dip lift that stacked on the clearance lift
    # raised single swings to about 80 deg)
    # open-body kind (wave 8 second part): (b), kept; this clause, hidden until the grade 2 band passed, reads the
    # healthy walk's first swing (gait initiation) at 78.28 deg on Anny: left open, a change of gait.NORMAL changes
    # every walk (design 20.7, D170)
    b, info = GA.generate(GA.NORMAL, duration_s=11.0, start_at_s=1.0, stop_at_s=8.5, seed=7)
    peaks = [w["knee_peak_deg"] for side in GA.SIDES for w in info["swings"][side] if w["clearance_m"] is not None]
    assert max(peaks) < 72.0


def test_the_dip_lift_acts_where_the_sole_dips(GA):
    s = np.linspace(0.0, 1.0, 101)
    for sc in (0.05, 0.3, 0.6, 0.95):
        ce = GA._dip_bump(s, sc)
        assert ce[0] == 0.0 and ce[-1] == 0.0 and ce.max() <= 1.0
        assert ce[int(round(sc * 100))] > 0.5                       # effective at the dip itself
    assert GA._dip_bump(s, 0.05)[40:].max() < 0.15                  # and not far from it


def test_sudden_knee_buckles_fall_in_a_stance_and_fling_the_arms(GA):
    from mdfm.synth.body import components_deg
    p = replace_(GA.NORMAL, knee_buckles=(("left", 3.0, 24.0, 0.13, 0.4), ("right", 6.0, 24.0, 0.13, 0.4)),
                 buckle_arm_deg=18.0)
    b, info = GA.generate(p, duration_s=10.0, start_at_s=1.0, stop_at_s=8.5, seed=3)
    b0, _ = GA.generate(GA.NORMAL, duration_s=10.0, start_at_s=1.0, stop_at_s=8.5, seed=3)
    y, y0 = b.joints().numpy()[:, 0, 1], b0.joints().numpy()[:, 0, 1]
    c, c0 = components_deg(b.rot6d), components_deg(b0.rot6d)
    for bk in info["buckles"]:
        i = GA.SIDES.index(bk["side"])
        k = int(round(bk["t_depth_s"] * b.rate))
        assert bk["t_depth_s"] > (3.0 if i == 0 else 6.0) and info["stance"][k, i]
        assert (y0 - y)[k - 5:k + 6].max() > 0.025                  # the pelvis drops
        assert (y0 - y)[k + 60:k + 80].max() < 0.01                  # and is back 0.6 s later
        assert (y0 - y)[k - 25] < 0.005                               # the drop takes about 0.13 s
        assert abs(abs(c[k, 16, 2] - c0[k, 16, 2]) - 18.0) < 3.0     # the arms fling out at the dip
    assert GA.validity(b, info)["gait"]["ok"]


def test_the_arms_take_back_the_trunk_lean(GA):
    """arm_lean_follow: the upper arms' coronal angle in the world follows the trunk's lateral lean much less than an
    arm locked to the thorax."""
    def slope(follow):
        p = replace_(GA.NORMAL, trunk_sway_deg=10.0, trunk_sway_hz=0.3, arm_lean_follow=follow,
                     arm_swing_deg=(0.0, 0.0))
        b, info = GA.generate(p, duration_s=10.0, start_at_s=1.0, seed=5)
        J = b.joints().numpy()
        psi = info["heading_rad"]
        lft = np.stack([np.cos(psi), -np.sin(psi)], -1)
        tr = J[:, 12] - J[:, 0]
        lean = np.degrees(np.arctan2((tr[:, [0, 2]] * lft).sum(-1), tr[:, 1]))
        ua = J[:, 18] - J[:, 16]
        arm = np.degrees(np.arctan2((ua[:, [0, 2]] * lft).sum(-1), -ua[:, 1]))
        k = slice(200, 900)
        return abs(np.polyfit(lean[k], arm[k], 1)[0])
    assert slope(0.7) < 0.5 * slope(0.0)


def test_the_waddling_gait_carries_a_lumbar_hyperlordosis(GA):
    from mdfm.synth.body import components_deg
    # 10 s: the stop's last step lands at 9.5 s (gait.validity refuses a clip that cuts it)
    b, info = GA.generate(GA.waddling(), duration_s=10.0, start_at_s=1.0, stop_at_s=7.0, seed=6)
    assert abs(np.mean(info["pelvis_tilt_deg"]) - 15.0) < 1.0      # the pelvis tilted forward
    c = components_deg(b.rot6d)
    assert c[:, 3, 0].mean() < -6.0                                 # the lumbar spine extended
    J = b.joints().numpy()
    th = J[:, 12] - J[:, 9]                                        # the thorax's line kept near its -2 deg
    psi = info["heading_rad"]
    pitch = np.degrees(np.arctan2(th[:, 0] * np.sin(psi) + th[:, 2] * np.cos(psi), th[:, 1]))
    r = b.rest().numpy()
    pitch0 = np.degrees(np.arctan2(r[12, 2] - r[9, 2], r[12, 1] - r[9, 1]))
    assert abs((pitch - pitch0)[300:600].mean() + 2.0) < 3.0
    assert GA.validity(b, info)["gait"]["ok"]


def test_a_turn_widens_the_inside_footprint(GA):
    # the guards off in both walks (the double-support guard may set a landing foot wider in one and not the other)
    turns = ((4.0, 180.0, 2.4),)
    p = replace_(GA.NORMAL, leg_gap_m=None)
    a, ia = GA.generate(replace_(p, turn_widen_m_rad=0.0), duration_s=10.0, start_at_s=1.0, turns=turns, seed=8)
    b, ib = GA.generate(p, duration_s=10.0, start_at_s=1.0, turns=turns, seed=8)
    psi = ib["heading_rad"]
    widest = 0.0
    for side in GA.SIDES:
        sg = GA.SIGN[side]
        for fa, fb in zip(ia["footprints"][side], ib["footprints"][side]):
            if fb["first"] or fb["final"]:
                continue
            d = np.subtract(fb["ankle"], fa["ankle"])[[0, 2]]
            k = fb["kp"]
            lft = np.array([np.cos(psi[k]), -np.sin(psi[k])])
            turn = psi[fb["k_to"]] - psi[fb["k_hs"]]
            want = GA.NORMAL.turn_widen_m_rad * abs(turn) if turn * sg > 0 else 0.0
            assert abs(sg * float(d @ lft) - want) < 1e-6 and np.linalg.norm(d) < want + 1e-6
            widest = max(widest, want)
    assert widest > 0.05                       # the inside stance that carries most of the turn steps 5 cm wider
    assert GA.validity(b, ib)["limits_outside"] == 0


def test_retropulsion_takes_the_steps_of_its_grade(GA):
    for g in (0, 1, 2):
        b, info = GA.retropulsion(g)
        n = GA.count_steps(b, after_s=1.0)
        lo, hi = {0: (1, 2), 1: (3, 5), 2: (6, 99)}[g]
        assert lo <= n <= hi, (g, n)
        J = b.joints().numpy()
        assert J[-1, 0, 2] < J[0, 0, 2] - 0.1                      # the body ends behind where it stood


def test_the_overlay_hook(GA):
    from mdfm.synth.body import components_deg
    base, _ = GA.generate(GA.NORMAL, duration_s=5.0, seed=9)

    def arm(t, info):
        return {18: (GA._rot_axis("y", np.full(len(t), -10.0)), "child")}
    out, _ = GA.generate(GA.NORMAL, duration_s=5.0, seed=9, overlays=(arm,))
    d = np.abs(components_deg(out.rot6d) - components_deg(base.rot6d)).max((0, 2))
    assert d[18] > 5.0 and np.delete(d, 18).max() < 1e-9

    def legs(t, info):
        jerk = 15.0 * info["swing_weight"][:, 0]
        return {4: (GA._rot_axis("x", jerk), "child")}
    out, info = GA.generate(GA.NORMAL, duration_s=6.0, seed=9, overlays=(legs,))
    assert GA.stance_slip(out, info)["p95_m_s"] < 1e-3


def test_the_catalogue_specs(GA):
    # open-body kind (wave 5 second follow-up, D26): (b) a check property, every sample clip valid: the festination's
    # knee rate fixed in the motion (the swing's bumps widened, D101, D106: 707.9 to 620.8 deg/s); passes
    sp = GA.specs()
    ids = [s.clip_id for s in sp]
    assert len(set(ids)) == len(ids) and GA.LEVEL0 in ids
    for s in sp:
        assert s.since == "catalogue_v1" and (s.level0 is None or s.level0 in ids)
        assert set(s.views) <= {"track_left", "examiner", "left", "back", "front"}
        for txt in (s.operator, s.grade_text, s.params_text, s.note):
            assert chr(0x2014) not in txt and chr(0x2013) not in txt, s.clip_id
        # the tandem's turned-out feet are stated in the spec's note, the same on SARA 0 to 2
        assert (s.note == GA.TANDEM_NOTE) == s.clip_id.startswith("gait_E8t_tandem_sara"), s.clip_id
    assert f"{GA.TANDEM_TOE_OUT_DEG:g} deg" in GA.TANDEM_NOTE
    sample = ["gait_E8_parkinsonian_festination", "gait_E8t_tandem_sara1", "gait_E8_choreic",
              "gait_E8_nph_magnetic", "gait_E7p_retropulsion_g1"]
    for s in sp:
        if s.clip_id in sample:
            b, truth = s.build()
            assert truth["validity"]["gait"]["ok"], s.clip_id
            if s.clip_id == "gait_E8_choreic":
                # the upper-body chorea comes from the involuntary family's engine, every region at its level
                ch = truth["params"]["chorea"]
                assert ch["engine"] == "involuntary.compose_clip" and ch["level_share_of_rom"] == 0.2
                assert set(ch["regions"]) == {"trunk", "neck", "arm_left", "arm_right"}
                assert all(r["reachable"] for r in ch["regions"].values())
                # the contact guard of involuntary.py keeps the choreic hands out of the thighs and the trunk (the
                # right index tip came within 10 mm of the femur's axis without it, 2026-11-03 review)
                cg = ch["contact_guard"]
                assert cg is not None and all(100 * cg["tol_m"] >= x["after_cm_max"] - 0.05
                                              for x in cg["sides"].values()), cg


@pytest.fixture(scope="module")
def scissor(GA):
    out = {}
    for lv in (2, 1):
        # 10 s: the stop's last step lands at 9.4 s (gait.validity refuses a clip that cuts it)
        b, info = GA.generate(GA.spastic_diplegic(lv), duration_s=10.0, start_at_s=1.0, stop_at_s=7.5, seed=3)
        out[lv] = (b, info, GA.scissor_record(b, info))
    return out


def test_scissoring_brings_the_knees_together_without_passing_them_through(GA, scissor, healthy):
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, the scissoring knees coming together:
    # kept; on Anny the mid-swing knee distance reads 0.151 m against the healthy 0.172 less 0.03, the motion not
    # fixed: carried to wave 8's gait pass (design 16.9, D109), left open there with its readings (design 20.7, D170)
    from mdfm.synth import legcontact as LC
    from mdfm.synth.body import components_deg
    b, info, rec = scissor[2]
    v = GA.validity(b, info)
    assert v["gait"]["ok"] and v["limits_outside"] == 0
    assert v["legs"]["ok"] and v["legs"]["hard_min_m"] > -LC.TOL_M
    assert rec["hard_gap_min_m"] < LC.TOUCH_M                                 # the knees touch
    # the feet land on the midline: the median signed step width (each footprint from the other foot's line of
    # progression) within 1.5 cm of zero, each side's median footprint within 1 cm of the midline, some past it
    assert abs(rec["step_width_signed_median_m"]) < 0.015, rec
    for side in GA.SIDES:
        assert abs(rec["footprint_offset_median_m"][side]) < 0.01, rec
    assert rec["share_past_midline"] > 0.1 and rec["n_footprints"] >= 4
    # the read-back is unbiased: the healthy walk's footprints lie half its step width from the midline on each side
    hb, hinfo, hm = healthy
    hrec = GA.scissor_record(hb, hinfo)
    assert abs(hrec["step_width_signed_median_m"] - hm["step_width_m"]["median"]) < 0.005
    for side in GA.SIDES:
        assert abs(hrec["footprint_offset_median_m"][side] - 0.5 * hm["step_width_m"]["median"]) < 0.005
    assert hrec["share_past_midline"] == 0.0
    # the knee joints come closer in mid swing than in the healthy walk
    assert rec["knee_distance_mid_swing_median_m"] < hrec["knee_distance_mid_swing_median_m"] - 0.03
    # the guard takes back part of the asked adduction, not all of it: the swing hips still adduct in mid swing
    c = components_deg(b.rot6d)
    sw = info["swing_weight"]
    assert c[sw[:, 0] > 0.9, 1, 2].mean() < -1.0 and c[sw[:, 1] > 0.9, 2, 2].mean() > 1.0
    assert rec["guard_added_abduction_deg"]["n_swings"] > 8


def test_the_scissoring_swing_adducts_without_an_outward_kick(GA, scissor, healthy):
    """The asked adduction comes in once the swing knee is level with the stance knee (swing_late_share), the
    internal rotation carries the swinging foot around the stance foot, the guard's abduction is a broad bump: no
    swing hip abducts beyond the larger of its toe-off and heel-strike values by more than 2 deg past the healthy
    walk's largest such excursion (the hip's frontal angle in the clinical sequence, flexion first), and the guard
    adds little (a guard bump of about 70 ms right after toe-off threw the leg out to hip width before sweeping it
    across, a circumduction that adductor spasticity prevents). The frontal speed itself is not bounded by the
    healthy walk's: the asked adduction sweeps in and back out within the late swing."""
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, no circumduction kick in a spastic
    # swing: kept; the guard's abduction on Anny's wider legs kicks 6.77 deg against the healthy 1.96 plus 2, the
    # motion not fixed: carried to wave 8's gait pass (design 16.9, D109), left open there (design 20.7, D170)

    def excursion(b, info):
        R = b.local_R().numpy()
        out = []
        for side, h, sg in (("left", 1, 1.0), ("right", 2, -1.0)):
            _, d, _ = GA._xzy_angles(R[:, h])                    # abduction positive on the left, negative on the right
            for s_ in info["swings"][side]:
                k0, k1 = s_["k_to"], s_["k_hs"]
                if k0 < 150 or k1 > b.T - 150:                     # gait initiation and the stop left out
                    continue
                out.append(float((sg * d[k0:k1 + 1]).max() - max(sg * d[k0], sg * d[k1])))
        return max(out)
    hb, hinfo, _ = healthy
    ref = excursion(hb, hinfo)
    for lv in (2, 1):
        b, info, rec = scissor[lv]
        assert excursion(b, info) < ref + 2.0, (lv, excursion(b, info), ref)
        assert rec["guard_added_abduction_deg"]["median"] < 2.0, rec


def test_the_scissoring_base_widens_inside_the_turn_only(GA):
    """turn_base_m widens the base over the turn's own window: the steps of the approach and of the exit keep the
    scissoring base, the turn's middle steps are wider (generator footprints, the step width in the heading frame
    of the later footprint)."""
    p = replace_(GA.spastic_diplegic(2), leg_gap_m=None)
    b, info = GA.course(p, distance_m=3.0, turn_s=GA.SCISSOR_TURN_S, turn_share=GA.SCISSOR_TURN_SHARE, seed=5)
    ts, _, dur = info["turns"][0][:3]
    fp = sorted((f["kp"], s, np.asarray(f["ankle"])) for s in GA.SIDES for f in info["footprints"][s]
                if not f["first"] and not f["final"])
    psi = np.asarray(info["heading_rad"])
    inside, outside = [], []
    for (k0, s0, a0), (k1, s1, a1) in zip(fp, fp[1:]):
        if s0 == s1:
            continue
        lft = np.array([np.cos(psi[k1]), -np.sin(psi[k1])])
        w = (1.0 if s1 == "left" else -1.0) * float((a1[[0, 2]] - a0[[0, 2]]) @ lft)
        t0, t1 = k0 / b.rate, k1 / b.rate
        (outside if (t1 < ts or t0 > ts + dur) else inside).append(w)
    assert len(outside) >= 6 and len(inside) >= 4
    assert max(abs(w) for w in outside) < 0.04, outside
    assert max(inside) > 0.5 * GA.SCISSOR_TURN_BASE_M, inside


def test_the_guard_keeps_a_stronger_adduction_from_passing_the_legs_into_each_other(GA):
    """The scissoring walk asked a stronger swing adduction (16 deg): without the guard its knees pass into each
    other, with it they touch at most, the guard abducting a few degrees where needed."""
    from mdfm.synth import legcontact as LC
    p = replace_(GA.spastic_diplegic(2), swing_abduction_deg=(-16.0, -16.0))
    b, info = GA.generate(replace_(p, leg_gap_m=None), duration_s=6.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    v = GA.validity(b, info)
    assert not v["legs"]["ok"] and v["legs"]["hard_knots_beyond_tol"] > 0
    assert info["leg_guard"] == []
    b, info = GA.generate(p, duration_s=6.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    v = GA.validity(b, info)
    assert v["legs"]["ok"] and v["legs"]["hard_min_m"] > -LC.TOL_M
    add = [g["added_deg"] for g in info["leg_guard"]]
    assert 0.0 < max(add) < GA.LEG_GUARD_MAX_DEG and all(g["holds"] for g in info["leg_guard"])


def test_the_mild_scissoring_keeps_a_wider_base(GA, scissor):
    from mdfm.synth import legcontact as LC
    b, info, rec = scissor[1]
    _, _, rec2 = scissor[2]
    assert GA.validity(b, info)["legs"]["ok"]
    for side in GA.SIDES:
        assert rec["footprint_offset_median_m"][side] > rec2["footprint_offset_median_m"][side] + 0.01
    assert rec["knee_distance_mid_swing_median_m"] > rec2["knee_distance_mid_swing_median_m"]
    assert rec["hard_gap_min_m"] > -LC.TOL_M


def test_the_guard_is_on_by_default_and_the_validity_carries_the_legs(GA, healthy):
    """Since the leg-to-leg calibration of 2026-11-03 every gait has the swing guard (LEG_GAP_M): on the healthy
    walk's base it adds at most about a degree of hip abduction to a few swings."""
    b, info, _ = healthy
    assert GA.NORMAL.leg_gap_m == GA.LEG_GAP_M == 0.0 and len(info["leg_guard"]) >= 10
    assert all(g["holds"] for g in info["leg_guard"])
    assert max(g["added_deg"] for g in info["leg_guard"]) < 2.0
    v = GA.validity(b, info)
    assert set(v["legs"]) >= {"ok", "hard_min_m", "soft_min_m", "tol_m", "soft_tol_m"} and v["legs"]["ok"]
    sp = {s.clip_id: s for s in GA.specs()}
    assert "gait_E8_spastic_diplegic" in sp and "gait_E8_spastic_diplegic_mild" in sp
    assert sp["gait_E8_spastic_diplegic_mild"].since == "catalogue_v1"
    assert "scissoring" in sp["gait_E8_spastic_diplegic"].params_text
    assert not any(k.startswith("scissoring (the feet") for k in GA.NOT_RENDERED)


HEALTHY_IDS = ("gait_E8_level0", "gait_E8t_tandem_sara0", "gait_E7p_retropulsion_g0")
# the distal band's reading of the healthy clips (legcontact.distal_gap: the thigh up to KNEE_ZONE_M + DISTAL_BAND_M
# above the knee joint counted hard; outside the check) as measured on 2026-11-03, the level 0's in its 180 deg turn
# (9.74 s, the medial thighs just above the knees pressed together); the pull test's legs never come within the
# measured range (legcontact.CULL_M)
DISTAL_READ_M = {"gait_E8_level0": -0.0092, "gait_E8t_tandem_sara0": -0.0033, "gait_E7p_retropulsion_g0": 0.02}
DISTAL_SLACK_M = 0.001          # a pinned reading may come out at most this much deeper


def test_the_healthy_walks_keep_their_legs_apart(GA):
    """The calibration of 2026-11-03 (STEP_WIDTH_M, TURN_BASE_M, LEG_GAP_M, the tandem's TANDEM_HALF_M and
    TANDEM_TOE_OUT_DEG): the healthy clips of the catalogue (the level-0 course with its 180 deg turn, the tandem
    of SARA 0, the pull test's two recovery steps) hold the leg-to-leg check (no knee, shank or foot deeper than
    legcontact.TOL_M into the other leg, the inner thighs within SOFT_TOL_M), the guard adding little on the
    level 0's swings; the distal band's reading (the thigh just above the knee counted hard, in the record only)
    no deeper than its pinned value (DISTAL_READ_M); the former healthy course (the 0.09 m base, no guard, no
    widening in the turn) did not hold the check, and that alone refuses it (gait.ok, LEGS_REFUSE)."""
    from mdfm.synth import legcontact as LC
    sp = {s.clip_id: s for s in GA.specs()}
    for cid in HEALTHY_IDS:
        b, truth = sp[cid].build()
        legs = truth["validity"]["legs"]
        assert legs["ok"] and legs["hard_min_m"] > -LC.TOL_M and legs["soft_min_m"] > -LC.SOFT_TOL_M, (cid, legs)
        assert truth["validity"]["gait"]["ok"], cid
        assert legs["distal_min_m"] <= legs["hard_min_m"] + 1e-12
        assert legs["distal_min_m"] > DISTAL_READ_M[cid] - DISTAL_SLACK_M, (cid, legs)
        if cid == GA.LEVEL0:
            gr = truth["params"]["leg_guard"]
            assert gr["n_not_held"] == 0 and gr["added_deg_max"] < 2.0, gr
    p = replace_(GA.NORMAL, step_width_m=0.09, leg_gap_m=None, turn_base_m=0.0)
    b, info = GA.course(p, distance_m=4.0, seed=GA.SEED)
    vv = GA.validity(b, info)
    v = vv["legs"]
    assert not v["ok"] and v["hard_min_m"] < -LC.TOL_M, v
    # the leg failure alone refuses the walk (LEGS_REFUSE): every other check of the gait holds
    assert vv["gait"]["legs_in_ok"] and not vv["gait"]["ok"]
    assert vv["limits_outside"] == 0 and vv["stance_slip_p95_m_s"] < GA.SLIP_TOL_M_S and vv["finite"]
    assert vv["swing_lowest_m"] > -GA.PENETRATION_TOL_M


def test_the_double_support_guard_sets_the_landing_foot_wider(GA):
    """Short steps on a narrow base: without the guards the two planted legs pass into each other in double support
    (the trailing knee against the landing thigh); with them each such landing foot is set down wider (a footprint
    offset recorded in info['ds_push']) and the planted legs stay within the legcontact check."""
    from mdfm.synth import legcontact as LC
    # wave 5 follow-up (D26): on the open body the unguarded short steps on the 0.04 m base keep the planted legs
    # 5.53 mm apart, and on a 0.02 m base pass them 7.72 mm into each other, past legcontact.TOL_M
    # (scripts/openbody_wave5_checks.py followup, ds_control_unguarded): the control takes the 0.02 m base
    p = replace_(GA.NORMAL, step_m=0.30, step_width_m=0.02, step_width_sd_m=0.0)
    # 7 s: the stop's last step lands at 6.4 s (gait.validity refuses a clip that cuts it)
    b0, i0 = GA.generate(replace_(p, leg_gap_m=None), duration_s=7.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    g0 = LC.leg_gap(b0)
    ds0 = np.asarray(i0["stance"], bool).all(1)
    assert g0["hard_m"][ds0].min() < -LC.TOL_M and i0["ds_push"] == []
    b, info = GA.generate(p, duration_s=7.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    pushes = info["ds_push"]
    assert len(pushes) >= 2 and all(x["outward_m"] > GA.DS_MARGIN_M for x in pushes)
    offs = {(s_, j): d for s_, j, d, _ in info["params"]["footprint_offsets"]}
    for x in pushes:
        assert offs[(x["side"], x["stance"])] >= x["outward_m"] - 1e-12
        f = info["footprints"][x["side"]][x["stance"]]
        f0 = i0["footprints"][x["side"]][x["stance"]]
        assert GA.SIGN[x["side"]] * (f["ankle"][0] - f0["ankle"][0]) > 0.5 * x["outward_m"]   # set down wider
    g = LC.leg_gap(b)
    ds = np.asarray(info["stance"], bool).all(1)
    assert g["hard_m"][ds].min() > -LC.TOL_M and GA.validity(b, info, g)["legs"]["ok"]


def test_the_post_overlay_guard_bumps_a_swing_the_overlay_passed_into_the_other_leg(GA):
    """A swing-only leg overlay (6 deg of left hip adduction weighted by the swing, composed after the swing guard
    measured the swing) passes the left swing knee into the right leg beyond the check; the post-overlay guard
    (_overlay_swing_contacts) records each such swing with its depth and adds an abduction bump to it
    (swing_extra_abd), the walk generated again: the final legs hold the check and the gait validity."""
    from mdfm.synth import legcontact as LC

    def adduct(t, info):
        return {1: (GA._rot_axis("z", -6.0 * info["swing_weight"][:, 0]), "parent")}
    # 7 s: the stop's last step lands at 6.4 s (gait.validity refuses a clip that cuts it)
    b, info = GA.generate(GA.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=4.5, seed=3, overlays=(adduct,))
    og = info["overlay_guard"]
    assert len(og) >= 2 and all(x["side"] == "left" and x["deg"] > 0.0 for x in og), og
    first = [x for x in og if x["round"] == 0]
    assert first and all(x["hard_m"] < -LC.TOL_M or x["soft_m"] < -LC.SOFT_TOL_M for x in first), first
    extra = info["params"]["swing_extra_abd"]
    assert len(extra) == len(og) and all(e[0] == "left" for e in extra)
    v = GA.validity(b, info)
    assert v["legs"]["ok"] and v["legs"]["hard_min_m"] > -LC.TOL_M and v["gait"]["ok"], v["legs"]
    _, i0 = GA.generate(GA.NORMAL, duration_s=7.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
    assert i0["overlay_guard"] == [] and i0["params"]["swing_extra_abd"] == ()


# ---------------------------------------------------------------- wave SH1d (the clinical review of 2026-11-03)
def test_the_continuity_check_sees_a_jumping_stance_foot(GA, healthy):
    """Positive control of gait.continuity: the healthy walk holds it (no touching sole point moves more than
    CONTACT_JUMP_MAX_M between two frames at 30 fps, no joint above PEAK_ACC_MAX_M_S2 in acceleration); the same
    body moved 5 cm sideways for a few knots jumps its stance foot and is refused."""
    b, info, _ = healthy
    c = GA.continuity(b, info)
    assert c["ok"] and c["contact_jump_worst_m"] <= GA.CONTACT_JUMP_MAX_M, c
    assert c["peak_acc_m_s2"] <= GA.PEAK_ACC_MAX_M_S2, c
    tr = b.transl.clone()
    k = int(5.0 * b.rate)
    tr[k:k + 4, 0] += 0.05
    bad = GA.continuity(replace_(b, transl=tr), info)
    assert not bad["ok"] and bad["contact_jump_worst_m"] > GA.CONTACT_JUMP_MAX_M, bad
    assert GA.validity(b, info)["continuity"]["ok"]
    # the floor skim reads the sole whatever the stance flags (a swing foot launched or gliding at floor height):
    # with no knot held in stance, the same jump escapes the stance reading and not the skim
    none = dict(info, stance=np.zeros_like(np.asarray(info["stance"], bool)))
    assert c["floor_skim_worst_m"] <= GA.CONTACT_JUMP_MAX_M, c
    sk = GA.continuity(replace_(b, transl=tr), none)
    assert sk["contact_jump_worst_m"] == 0.0 and sk["floor_skim_worst_m"] > GA.CONTACT_JUMP_MAX_M and not sk["ok"], sk


def test_the_continuity_check_sees_a_knee_flick(GA, healthy):
    """The knee flexion speed bound of gait.continuity (KNEE_RATE_MAX_DEG_S, read at 30 fps): the healthy walk holds
    it; the same body with one knee flexed 40 deg more over three knots (30 ms) flicks it and is refused."""
    from mdfm.body.rotations import matrix_to_rot6d, rot6d_to_matrix
    b, info, _ = healthy
    c = GA.continuity(b, info)
    assert c["knee_rate_worst_deg_s"] <= GA.KNEE_RATE_MAX_DEG_S, c
    r6 = b.rot6d.clone()
    k = int(4.0 * b.rate)
    kn = GA.LEG["left"][1]
    R = rot6d_to_matrix(r6[k:k + 3, kn])
    r6[k:k + 3, kn] = matrix_to_rot6d(R @ torch.from_numpy(GA._rot_axis("x", 40.0)).to(R.dtype))
    bad = GA.continuity(replace_(b, rot6d=r6), info)
    assert bad["knee_rate_worst_deg_s"] > GA.KNEE_RATE_MAX_DEG_S and not bad["ok"], bad


def test_a_turning_stance_foot_pivots_on_one_sole_point(GA):
    """A planted foot turning with the heading (pivot_share) pivots on the roll's contact point while it rolls and on
    its touching sole point nearest the ball of the foot while it stands flat (wave SH1d review of lane gait: the
    forward-most touching point the pivot was on the flat foot before left the ball and the lateral sole moving 25
    to 32 mm in each turning stance): in every stance of a healthy 180 deg turn whose foot turns by 20 deg or more,
    some sole point touching the floor (within PIVOT_NEAR_M, the generator's touching tolerance) for at least a third
    of the stance travels 6 mm or less over those knots (1.0 to 5.5 mm read in wave SH1d; the review asked 3 mm:
    the rigid foot turning on its toe tip in the push-off moves the ball, the stance with the most turn by most);
    the stance contact stays exact (stance_slip) and the walk valid."""
    # open-body kind (wave 5 second follow-up, D26): (b) a physical property, one still sole point through a turning
    # contact: the motion fixed, the push-off pivot turns a held foot only where it touches (PIVOT_TOE_NEAR_M 1 mm,
    # D102: the course's slip 5.564 to 0.172 mm/s); passes
    b, info = GA.course(GA.NORMAL, distance_m=3.0, seed=GA.SEED)
    J, G = b.fk()
    st = np.asarray(info["stance"], bool)
    n_turning = 0
    for i, side in enumerate(GA.SIDES):
        X = GA.sole_cloud_points(b, side, (J, G))
        floor = float(np.median(X[..., 1].min(1)[st[:, i]]))
        fw = G.numpy()[:, GA.LEG[side][2], :, 2]
        for a, c in GA._runs(st[:, i]):
            yaw = np.degrees(np.ptp(np.unwrap(np.arctan2(fw[a:c + 1, 0], fw[a:c + 1, 2]))))
            if c - a < 10 or yaw < 20.0:
                continue
            n_turning += 1
            on = X[a:c + 1, :, 1] <= floor + GA.PIVOT_NEAR_M
            need = int(np.ceil((c - a + 1) / 3))
            trav = [float(np.linalg.norm(np.ptp(X[a:c + 1][on[:, m], m][:, [0, 2]], axis=0)))
                    for m in range(X.shape[1]) if on[:, m].sum() >= need]
            assert trav and min(trav) <= 0.006, (side, a / b.rate, yaw, min(trav, default=None))
    assert n_turning >= 3
    v = GA.validity(b, info)
    assert v["stance_slip_p95_m_s"] < GA.SLIP_TOL_M_S and v["gait"]["ok"], v["gait"]


SNAP_IDS = ("gait_E8_trendelenburg_right", "gait_E8_dystonic_left_l2", "gait_E7p_retropulsion_g2",
            "gait_E8_fog_shuffling_g3", "gait_E8_steppage_bilateral")


def test_the_reviewed_snaps_are_gone(GA):
    """The one-frame snaps of the 2026-11-03 review (the Trendelenburg turn's landing, a 5.8 cm jump; the dystonic
    foot, 84 m/s2; the pull test's backward steps, 182 m/s2) and of the wave SH1d review of lane gait (the shuffling
    start episode's release, the right foot sliding flat 35 cm at floor height, 50.9 mm in one frame; the bilateral
    steppage's turn, a knee flick of 844 deg/s at 78.8 m/s2): each clip holds the continuity check (its contact
    feet and its sole on the floor move at most CONTACT_JUMP_MAX_M a frame at 30 fps, no knee faster than
    KNEE_RATE_MAX_DEG_S, no joint above PEAK_ACC_MAX_M_S2) and the gait validity; the pull test's recovery steps are
    low, monotone, minimum-jerk foot paths (retro_steps_read), the pull marked."""
    sp = {s.clip_id: s for s in GA.specs()}
    for cid in SNAP_IDS:
        b, truth = sp[cid].build()
        c = truth["validity"]["continuity"]
        assert c["ok"] and c["contact_jump_worst_m"] <= GA.CONTACT_JUMP_MAX_M, (cid, c)
        assert c["floor_skim_worst_m"] <= GA.CONTACT_JUMP_MAX_M, (cid, c)
        assert c["knee_rate_worst_deg_s"] <= GA.KNEE_RATE_MAX_DEG_S, (cid, c)
        assert c["peak_acc_m_s2"] <= GA.PEAK_ACC_MAX_M_S2, (cid, c)
        assert truth["validity"]["gait"]["ok"], cid
        if cid.startswith("gait_E7p"):
            r = truth["params"]["backward_steps"]
            assert r["all_monotone"] and r["max"]["sole_high_m"] < 0.06 and r["max"]["knee_peak_deg"] < 45.0, r
            assert truth["params"]["stimulus_marks"] == [{"t_s": GA.RETRO_PULL_S, "dur_s": GA.PULL_MARK_S,
                                                          "text": "PULL"}]


@pytest.fixture(scope="module")
def tandem_built(GA):
    sp = {s.clip_id: s for s in GA.specs()}
    return {g: sp[f"gait_E8t_tandem_sara{g}"].build() for g in (0, 1, 2)}


def test_the_tandem_clips_take_enough_steps_for_their_anchor(GA, tandem_built):
    """SARA gait (sara): 0 allows one tandem misstep, 1 shows its difficulty only over 10 consecutive tandem steps.
    Every tandem clip takes at least TANDEM_MIN_STEPS tandem steps (read back from the body: the stance runs before
    the stop); SARA 0 keeps the line, SARA 1 sets TANDEM_MISSTEPS (two, more than the one SARA 0 allows) beside it,
    SARA 2 leaves the line after its first steps; all three hold the gait validity."""
    rd = {g: tandem_built[g][1]["params"]["tandem_read"] for g in (0, 1, 2)}
    for g in (0, 1, 2):
        assert rd[g]["steps"] >= GA.TANDEM_MIN_STEPS, (g, rd[g])
        assert tandem_built[g][1]["validity"]["gait"]["ok"], g
    assert rd[0]["missteps"] == 0
    assert rd[1]["missteps"] == len(GA.TANDEM_MISSTEPS[1]) >= 2
    assert rd[2]["missteps"] >= rd[2]["steps"] - 6
    assert "two steps off the line" in {s.clip_id: s for s in GA.specs()}["gait_E8t_tandem_sara1"].params_text


def test_the_lateral_sway_is_soft_limited_and_the_head_rights_itself(GA, monkeypatch):
    """The astasia-abasia walk's large random sway (trunk_sway_deg 14): the spine's lean stays within the soft limit
    (TRUNK_SWAY_SOFT_DEG, a share carried by the pelvis), and the head rights itself on roll: the slope of its roll on
    the trunk's lean (trunk_head_roll) under 0.5 in magnitude; with the righting off the head tilts with the trunk."""
    p = GA.functional("astasia_abasia")
    b, info = GA.generate(p, duration_s=14.0, start_at_s=1.0, stop_at_s=11.0, seed=GA.SEED + 61)
    r = GA.trunk_head_roll(b)
    assert abs(r["slope"]) < 0.5, r
    assert r["lean_p95_deg"] < GA.TRUNK_SWAY_SOFT_DEG and abs(r["lean_max_deg"]) < 1.2 * GA.TRUNK_SWAY_SOFT_DEG, r
    monkeypatch.setattr(GA, "HEAD_RIGHTING", 0.0)
    b0, _ = GA.generate(p, duration_s=14.0, start_at_s=1.0, stop_at_s=11.0, seed=GA.SEED + 61)
    assert GA.trunk_head_roll(b0)["slope"] < -0.5


def test_the_dystonic_foot_inverts_in_swing_without_a_steppage(GA):
    """The dystonic foot (2026-11-03 review: it inverted 2.8 deg in swing against the 16 asked, its knee at 91 deg
    read as a steppage): read in the world (foot_roll_read), the left foot's lateral side lies lower at mid swing
    than the healthy walk's by at least half the asked swing inversion, and lower in mid stance by at least half the
    stance inversion; its swing knee peaks at most 12 deg above the healthy one; no guard bump throws the swing out."""
    p = GA.dystonic("left", 2)
    b, info = GA.generate(p, duration_s=9.0, start_at_s=1.0, stop_at_s=6.5, seed=GA.SEED + 41)
    b0, i0 = GA.generate(GA.NORMAL, duration_s=9.0, start_at_s=1.0, stop_at_s=6.5, seed=GA.SEED + 41)
    r, r0 = GA.foot_roll_read(b, info, "left"), GA.foot_roll_read(b0, i0, "left")
    asked = GA.pair(p.swing_inversion_deg)[0]
    assert r["mid_swing_deg"] - r0["mid_swing_deg"] < -0.5 * asked, (r, r0)
    assert r["mid_stance_deg"] - r0["mid_stance_deg"] < -0.5 * GA.pair(p.stance_inversion_deg)[0], (r, r0)
    assert r["swing_knee_peak_deg"] - r0["swing_knee_peak_deg"] < 12.0, (r, r0)
    assert max((g["added_deg"] for g in info["leg_guard"]), default=0.0) < 3.0
    assert GA.validity(b, info)["gait"]["ok"]


def test_the_hesitation_and_freezing_clips_share_the_walk_of_their_level_0(GA):
    """The start hesitation and the freezing by shuffling are built on the 3.10 grade 1 walk and paired with it
    (PD_WALK_LEVEL0): outside the hesitation and the episodes every parameter is the grade 1 walk's, the start ramp
    of the hesitation and the swing ankle's timing of the shuffle (swing_ankle_share) aside; the shuffle's dip
    synergy acts on the swings of its episodes only (dip_synergy, None outside them)."""
    sp = {s.clip_id: s for s in GA.specs()}
    for cid in ("gait_E8_parkinsonian_start_hesitation", "gait_E8_fog_shuffling_g3"):
        assert sp[cid].level0 == GA.PD_WALK_LEVEL0 and GA.PD_WALK_LEVEL0 in sp
    g1, _ = GA.parkinsonian(1)
    sh, _ = GA.start_hesitation(1)
    fog, eps, tm = GA.fog_shuffling()
    t_sh = np.array([6.0, 9.0])
    gap = tm["turn_end_s"] + 1.0
    t_fog = np.array([gap, gap + 0.5])
    assert all(not (e["onset_s"] - GA.FOG_RAMP_S <= x <= e["offset_s"] + GA.FOG_RAMP_S) for e in eps for x in t_fog)
    assert g1.dip_synergy is None and all(fog.dip_synergy(float(x)) is None for x in t_fog)
    assert any(fog.dip_synergy(e["onset_s"]) == GA.FOG_DIP_SYNERGY for e in eps)
    for q, tt, skip in ((sh, t_sh, {"start_s", "arm_swing_step_ref_m"}),
                        (fog, t_fog, {"swing_ankle_share", "dip_synergy"})):
        for f in g1.__dataclass_fields__:
            if f in skip:
                continue
            a, b_ = getattr(g1, f), getattr(q, f)
            if callable(b_) or (isinstance(b_, tuple) and any(callable(x) for x in b_)):
                for i in (0, 1):
                    assert np.allclose(GA._side_at(b_, i, tt), GA._side_at(a, i, tt)), f
            else:
                assert a == b_, f


def test_the_shuffle_pushes_off_less_at_its_onsets(GA):
    """The freezing by shuffling's push-off (liftoff_pitch_deg) leaves the walk's sooner at an episode's onset
    (onset_fn on FOG_PUSH_OFF_ONSET_POWER, the step length's mirrored power) and comes back at its offset as on the
    envelope itself (episodes_fn, power 1): the straight episode's onset rolled the foot to 36 deg before a 0.23 s
    swing, 102.6 m/s2 at the heel on the page."""
    fog, eps, tm = GA.fog_shuffling()
    g1, _ = GA.parkinsonian(1)
    out, inside = GA.pair(g1.liftoff_pitch_deg)[0], GA.FOG_SHUFFLE["liftoff_pitch_deg"]
    win = [(0.0, tm["lead_s"] + GA.FOG_START_S)] + [(e["onset_s"], e["offset_s"]) for e in eps[1:]]
    plain = GA.episodes_fn(out, inside, win, GA.FOG_RAMP_S)
    h = 0.5 * GA.FOG_RAMP_S
    for e in eps[1:]:                                    # the turn's and the straight walk's onsets
        t = np.linspace(e["onset_s"] - h + 0.01, e["onset_s"] + h - 0.01, 9)
        assert np.all(GA._side_at(fog.liftoff_pitch_deg, 0, t) < plain(t))
        t = np.linspace(e["offset_s"] - h, e["offset_s"] + h, 9)
        assert np.allclose(GA._side_at(fog.liftoff_pitch_deg, 0, t), plain(t), atol=1e-12)
    t = np.linspace(0.0, eps[0]["offset_s"] + h, 50)    # the start episode: as before at both edges
    assert np.allclose(GA._side_at(fog.liftoff_pitch_deg, 1, t), plain(t), atol=1e-12)
    assert np.allclose(GA.onset_fn(out, inside, win, GA.FOG_RAMP_S, 1.0)(np.linspace(0, 30, 301)),
                       plain(np.linspace(0, 30, 301)), atol=1e-12)


def test_the_magnetic_gait_does_not_pivot_its_planted_feet(GA):
    """The magnetic NPH gait's flat stance foot does not pivot with its slow turn (NPH_PIVOT_SHARE 0: pivoting 0.9
    of the heading change, its turning stance read 14.22 mm at its least-moving sole point on the page, against
    5.17); the turn itself is the course's."""
    p, kw = GA.nph_magnetic()
    assert p.pivot_share == GA.NPH_PIVOT_SHARE == 0.0 and kw["turn_s"] == 6.5


def test_the_start_hesitation_swings_the_arms_with_its_steps(GA):
    """Stepping in place (the start hesitation) the arms swing little, the ankles stay low (the 2026-11-03 review read
    the heels 8 to 9 cm up) and the knees flex little (34 to 36 deg in the review; the wave SH1d review of lane gait
    read every right step in place at 43 deg against 14.5 standing): on every step in place, both sides, the knee
    (the angle between the thigh and the shank) peaks less than 20 deg above its standing value and the ankle rises
    less than 2.5 cm; once walking, the arms swing as in the 3.10 grade 1 walk (arm_swing_step_ref_m). The walk stops
    at 7 s (a walk cut by the clip's end is refused by the continuity check)."""
    # open-body kind (wave 5 second follow-up, D26): (b) a clinical property, the steps in place low: kept; on Anny
    # the heels' range in place reads 8.4 cm against 5 cm (a 4 mm clearance in place takes a lift that flexes the knee
    # 25 to 30 deg over standing), the motion not fixed: carried to wave 8's gait pass (design 16.9, D109), left open
    # there with its readings and tries (design 20.7, D170)
    p, _ = GA.start_hesitation(1)
    b, info = GA.generate(p, duration_s=10.0, start_at_s=1.0, stop_at_s=7.0, seed=GA.SEED + 6)
    J = b.joints().numpy()
    psi = info["heading_rad"]
    fwd = np.stack([np.sin(psi), np.cos(psi)], -1)
    rng = {}
    for name, (a, c) in (("in_place", (1.8, 3.2)), ("walking", (6.5, 8.5))):
        k = slice(int(a * b.rate), int(c * b.rate))
        w = [((J[k, wr][:, [0, 2]] - J[k, 0][:, [0, 2]]) * fwd[k]).sum(-1) for wr in (20, 21)]
        rng[name] = max(float(np.ptp(x)) for x in w)
    assert rng["in_place"] < 0.3 * rng["walking"], rng
    heel = J[int(1.8 * b.rate):int(3.2 * b.rate), [7, 8], 1]
    assert float(heel.max() - heel.min()) < 0.05
    for s in GA.SIDES:
        h, kn, an = GA.LEG[s][:3]
        u, v = J[:, h] - J[:, kn], J[:, an] - J[:, kn]
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        knee = 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        k_st = int(0.5 * b.rate)
        stand_knee, stand_ankle = float(knee[:k_st].mean()), float(J[:k_st, an, 1].mean())
        sw = [w for w in info["swings"][s] if w["k_to"] / b.rate < 3.2]
        assert len(sw) >= 2, s
        for w in sw:
            k0, k1 = w["k_to"], w["k_hs"]
            assert float(knee[k0:k1 + 1].max()) - stand_knee < 20.0, (s, k0 / b.rate, float(knee[k0:k1 + 1].max()))
            assert float(J[k0:k1 + 1, an, 1].max()) - stand_ankle < 0.025, (s, k0 / b.rate)
    assert GA.validity(b, info)["gait"]["ok"]


def test_the_stoop_is_read_back_and_orders_the_parkinsonian_grades(GA):
    """The stoop read back from the body (trunk_pitch, in every gait truth): the pelvis-to-neck line leans forward
    more at 3.10 grade 1 than in the healthy walk and more again at grade 2 (the 2026-11-03 review read 8 deg more
    than the healthy walk at grade 2)."""
    rows = {}
    for g, p in ((0, GA.NORMAL), (1, GA.parkinsonian(1)[0]), (2, GA.parkinsonian(2)[0])):
        b, _ = GA.generate(p, duration_s=6.0, start_at_s=1.0, stop_at_s=4.5, seed=3)
        rows[g] = GA.trunk_pitch(b)
    lean = {g: r["pelvis_neck_walk_deg"] for g, r in rows.items()}
    assert lean[0] < lean[1] < lean[2] and lean[2] - lean[0] > 5.0, rows
