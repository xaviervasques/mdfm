"""Tests of the folded arms of the arising (shared change SH9 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md, lane
fold of 2026-11-03), in mdfm/synth/motions.py: ARM_POSTURES['fold'], ARM_TARGETS['fold'], GRIP, skin_gap,
fold_report and the fold checks of arise_validity.

- the posture: inside the healthy joint-limit table (body.joint_limits) on the folded stand and sit, every arm
  component of ARM_POSTURES['fold'] at least 1 deg inside its bounds (the module docstring's claim), the hands inside
  the hand limits, each side's segments where its own ARM_TARGETS say (a regression record of the search's result);
- on the folded stand (STAND_REF): the forearms one above the other in front of the lower chest (where their axes
  cross the midsagittal plane the left lies above and in front of the right, both above the spine2 joint and below
  the midpoint of the spine2 and neck joints, a test definition of the lower chest), outside the trunk's skin, their
  skins at least FOLD_FOREARM_CLEAR_M apart, each hand
  outside the opposite upper arm and within FOLD_HAND_NEAR_M of it, the forearms and hands outside the trunk; the
  folded arms before SH9 (OLD_FOLD, the thumb at rest) fail on the stand: the forearms through each other (on SMPL+H
  the right hand inside the left upper arm too), entries that fail fold_report's 'entry_ok' (positive control; their
  trunk entry comes at the lean);
- skin_gap's sign: a point outside a part beyond its open end (the parts are open where they meet the next one) is
  outside for skin_gap (winding number) and inside for the sign of its closest triangle (reach._signed_to_mesh); the
  thigh entry that the SH2 review read for the version before SH9 at the forward lean (reach.clearance) was such a
  vertex;
- the arising level 0 (motions.arise): at every knot of the rise (the knots before and after it are still, as
  tests/test_synth_pathology.py checks) the fold holds: the forearms apart, the hands near the opposite upper arms,
  the forearms and hands outside the trunk and the thighs and out of every box of arise_chair; the realized ankle
  plantarflexion within DROP_TOL_DEG of the cited one by the standing hip drop (inside DROP_RANGE_M), the
  dorsiflexion within 0.5 deg as before; arise_validity carries the checks ('fold', 'armrest', 'hand_thigh') and they
  pass, 'fold' refusing on an entry alone (its 'ok' fold_report's 'entry_ok', the margins in 'margins_ok');
- the users of the fold and of motions.arise (the exam's ref_arise_E7, the parkinsonian arising with the hands moved
  to the chair arms, sitting down en bloc, the two paroxysmal kinesigenic clips) build and pass
  render_mesh_video.failed_checks; the paroxysmal kinesigenic clips carry the validity 'fold' of unfold_check, ok,
  with a lag;
- the solve in the tree (task sh1c): ARM_POSTURES['fold'] and ARM_TARGETS['fold'] are the record of
  scripts/solve_fold_posture.py (reports/synth/fold/fold_posture.json), whose digest is the solver's
  (fold_solver_digest), whose checks on the mesh pass, and which solve_fold reproduces (about seven minutes); the
  digest covers every function of motions that solve_fold reaches and every module value it reads (an AST walk),
  and the base postures;
- opening the fold (task sh1c): fold_open_shares gives each arm its clock; unfold_to_stand at the paroxysmal
  kinesigenic clip's unfold (postures' timing) keeps the forearms, hands, trunk and thighs apart, where the one-clock
  unfold (its lag 0, the version before) put the right hand into the left forearm; unfold_check reads every knot of
  the opening and UNFOLD_CHECK_PAD_S after it and refuses the one-clock unfold.
Bodies are procedural, in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read; nothing is written.
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.synth import hands as H
from mdfm.synth import motions as MO
from mdfm.synth.body import joint_limits, limit_violations

RATE = 100.0
OLD_FOLD = {"left": ((12.78, -60.53, 40.47), (-26.83, -37.56, -80.98), (-9.50, -96.15, -2.76), (7.83, 0.28, -7.21)),
            "right": ((17.55, 61.37, -27.29), (-24.74, 30.47, 74.80), (1.37, 89.17, 4.89), (7.83, 1.17, 5.76))}
#   ARM_POSTURES['fold'] before SH9 (fc523fb), with GRIP's fingers and the thumb at rest


@pytest.fixture(scope="module")
def stand():
    return MO.posture_body(MO.STAND_REF, 0.02)


@pytest.fixture(scope="module")
def risen():
    return MO.arise(standing_s=0.5)


def _crossing(J, e, w):
    """(3,) the point of the segment from joint e to joint w where it crosses the midsagittal plane x = 0."""
    s = J[e, 0] / (J[e, 0] - J[w, 0])
    assert 0.0 < s < 1.0
    return J[e] + s * (J[w] - J[e])


def test_the_folded_arms_lie_inside_the_tables_and_where_their_targets_say(stand):
    for name in ("stand_arms_folded", "sit_arms_folded"):
        b = MO.posture_body(name, 0.02)
        assert limit_violations(b.rot6d)["outside"] == 0, name
        assert H.hand_limit_report(b.hand_R())["ok"], name
    for side in ("left", "right"):
        err = MO.arm_errors("fold", side)
        assert set(err) == {"wrist_mm", "upper", "fore", "hand", "palm"}
        assert err["wrist_mm"] < 1.0 and max(v for k, v in err.items() if k != "wrist_mm") < 1.0, (side, err)
    assert MO.ARM_POSTURES["fold"]["left"] != MO.ARM_POSTURES["fold"]["right"]      # not mirrored
    lim = joint_limits()
    for side, joints in MO.ARM_JOINTS.items():
        a = np.asarray(MO.ARM_POSTURES["fold"][side], dtype=np.float64)
        assert np.minimum(a - lim.lo[list(joints)], lim.hi[list(joints)] - a).min() >= 1.0, side


def test_the_folded_forearms_lie_one_above_the_other_apart_with_the_hands_on_the_upper_arms(stand):
    J = stand.joints().numpy()[0]
    cl, cr = _crossing(J, 18, 20), _crossing(J, 19, 21)
    assert cl[1] > cr[1] and cl[2] > cr[2]                         # the left above and in front of the right
    for c in (cl, cr):                  # the lower chest: from the spine2 joint up to midway to the neck joint
        assert J[6, 1] < c[1] < 0.5 * (J[6, 1] + J[12, 1])
    fr = MO.fold_report(stand, [0])
    assert fr["ok"], fr
    assert fr["forearms"]["min_mm"] >= 1000.0 * MO.FOLD_FOREARM_CLEAR_M
    for c in ("hand_left_upper_right", "hand_right_upper_left"):
        assert 0.0 <= fr[c]["min_mm"] <= fr[c]["max_mm"] <= 1000.0 * MO.FOLD_HAND_NEAR_M, (c, fr[c])
    for c in ("hand_left_forearm_right", "hand_right_forearm_left", "hands", "trunk_left", "trunk_right"):
        assert fr[c]["min_mm"] > 0.0, (c, fr[c])


def test_the_folded_arms_before_sh9_fail_the_checks(monkeypatch):
    monkeypatch.setitem(MO.ARM_POSTURES, "fold", OLD_FOLD)
    monkeypatch.setattr(MO, "GRIP", {k: v for k, v in MO.GRIP.items() if k != "thumb"})
    old = MO.posture_body(MO.STAND_REF, 0.02)
    fr = MO.fold_report(old, [0], checks=("forearms", "hand_right_upper_left"))
    # D26, open-body port wave 6: on the open body the pre-SH9 numbers still pass the forearms through each other
    # (21.5 mm deep) but hold the right hand 15.3 mm off the left upper arm (inside it on SMPL+H)
    assert fr["forearms"]["min_mm"] < -20.0 and fr["hand_right_upper_left"]["min_mm"] > 0.0 and not fr["ok"]
    assert not fr["entry_ok"]                    # an entry: the refusing rule of arise_validity's 'fold'


def test_skin_gap_reads_an_open_part_by_the_winding_number():
    """On the forearm's axis 2 cm beyond the wrist joint, a point lies outside the forearm part (open at the wrist):
    the winding number is under one half and skin_gap gives its distance, while the sign of the closest triangle
    (reach._signed_to_mesh) calls it inside; a point a third of the way from the skin to the middle of the axis is
    inside for both."""
    from mdfm.synth.reach import _signed_to_mesh
    b = MO.posture_body("stand", 0.02)
    ids, F = MO._fold_sets()["fore_l"]
    V = H.body_points(b, ids, knots=[0])[0]
    J = b.joints().numpy()[0]
    none = np.zeros((0, 3), dtype=np.int64)
    mid = 0.5 * (J[18] + J[20])
    v = V[np.argmin(np.linalg.norm(V - mid, axis=1))]
    p = (v + (mid - v) / 3.0)[None]
    assert MO._winding(p, V, F)[0] > 0.5 and MO.skin_gap(p, none, V, F) < -0.005
    assert _signed_to_mesh(p, V, F)[0] == pytest.approx(MO.skin_gap(p, none, V, F), abs=1e-12)
    out = (J[20] + 0.02 * (J[20] - J[18]) / np.linalg.norm(J[20] - J[18]))[None]
    assert MO._winding(out, V, F)[0] < 0.5
    g = MO.skin_gap(out, none, V, F)
    assert g > 0.01 and _signed_to_mesh(out, V, F)[0] == pytest.approx(-g, abs=1e-12)


def test_the_arising_level0_keeps_the_fold_clear_at_every_knot_of_the_rise(risen):
    b, info = risen
    k_on, k_end = int(round(info["onset_s"] * RATE)), int(round(info["end_s"] * RATE))
    fr = MO.fold_report(b, np.arange(k_on, k_end + 1), MO.arise_chair(b, 0.0))
    assert fr["n_knots"] == k_end - k_on + 1
    assert fr["ok"], fr
    for c in ("thigh_left", "thigh_right", "trunk_left", "trunk_right", "hand_left_forearm_right",
              "hand_right_forearm_left"):
        assert fr[c]["min_mm"] > 0.0, (c, fr[c])
    assert fr["forearms"]["min_mm"] >= 1000.0 * MO.FOLD_FOREARM_CLEAR_M
    for c in ("hand_left_upper_right", "hand_right_upper_left"):
        assert 0.0 <= fr[c]["min_mm"] and fr[c]["max_mm"] <= 1000.0 * MO.FOLD_HAND_NEAR_M, (c, fr[c])
    for c in ("chair_left", "chair_right", "armrest_left", "armrest_right"):
        assert fr[c]["depth_mm_max"] == 0.0, (c, fr[c])
    assert fr["entry_ok"]
    assert limit_violations(b.rot6d)["outside"] == 0 and info["hand_limits_ok"] and info["balance"]["ok"]
    assert abs(info["realized_ankle_plantarflexion_deg"] - MO.ANKLE_PLANTAR_DEG) <= MO.DROP_TOL_DEG
    assert info["realized_ankle_dorsiflexion_deg"] == pytest.approx(MO.ANKLE_DORSI_DEG, abs=0.5)
    lo, hi = MO.DROP_RANGE_M
    assert lo < info["standing_hip_drop_m"] < hi and info["sources"]["standing_hip_drop"] == "module"


def test_arise_validity_carries_the_fold_checks(risen):
    b, info = risen
    v = MO.arise_validity(b, info)
    assert v["fold"]["ok"] and v["fold"]["margins_ok"] and v["armrest"]["ok"] and v["hand_thigh"]["ok"]
    assert v["fold"]["ok"] == v["fold"]["entry_ok"]
    assert v["fold"]["n_knots"] == len(np.unique(np.r_[np.arange(int(round(info["onset_s"] * RATE)),
                                                                 int(round(info["end_s"] * RATE)) + 1,
                                                                 MO.FOLD_CHECK_STEP),
                                                       int(round(info["end_s"] * RATE))]))
    assert "fold" not in MO.arise_validity(b)                      # without the info: the other users' own checks


def _script():
    spec = importlib.util.spec_from_file_location("render_mesh_video_script",
                                                  os.path.join(paths.TREE, "scripts", "render_mesh_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("clip_id", ["ref_arise_E7", "park_E7_arise_g2", "park_E22_sitdown_enbloc",
                                     "post_E7_pkd_dystonic", "post_E7_pkd_choreic"])
def test_the_users_of_the_fold_build_and_pass_their_checks(clip_id):
    from mdfm.synth import exam as EX
    from mdfm.synth import parkinsonism as PK
    from mdfm.synth import postures as PO
    specs = {s.clip_id: s for s in list(EX.exam_set()) + list(PK.specs()) + list(PO.specs())}
    body, truth = specs[clip_id].build()
    assert _script().failed_checks(truth) == [], clip_id
    assert torch.isfinite(body.rot6d).all() and torch.isfinite(body.transl).all()
    if clip_id == "ref_arise_E7":                                  # the exam clip carries the fold checks
        assert truth["validity"]["fold"]["ok"] and truth["validity"]["armrest"]["ok"]
    if clip_id.startswith("post_E7_pkd"):                          # the opening read on the final body, every knot
        fold = truth["validity"]["fold"]
        assert fold["ok"] and fold["hand_right_forearm_left"] >= 0.0
        lag = truth["params"]["arms_unfold_lag"]                 # the smallest lag that keeps the skins apart (D26)
        assert (lag["lag"] == 0.0) == (lag["tried"][0]["min_mm"] >= 1000.0 * MO.FOLD_OPEN_CLEAR_M)
    if clip_id.startswith("post_E7_pkd"):     # the opening read on the final body, the attack included (unfold_check)
        f = truth["validity"]["fold"]
        assert f["ok"] and min(f[c] for c in MO.FOLD_OPEN_CHECKS) >= 0.0, (clip_id, f)


# ---------------------------------------------------------------- the solve in the tree (task sh1c)
def _record():
    with open(MO.fold_record_path(), encoding="utf-8") as f:
        return json.load(f)


def test_the_fold_is_read_from_its_current_record():
    rec = _record()
    assert MO._FOLD_STATE["state"] == "read" and rec["solver_digest"] == MO.fold_solver_digest()
    fold = {s: tuple(tuple(a) for a in rec["fold"][s]) for s in ("left", "right")}
    assert MO.ARM_POSTURES["fold"] == fold
    d = MO.fold_directions(fold)
    for s in ("left", "right"):
        assert {k: tuple(v) for k, v in rec["arm_targets"][s].items()} == d[s]
        assert {k: v for k, v in MO.ARM_TARGETS["fold"][s].items()} == d[s]
    v = rec["verify"]
    assert v["stand"]["ok"] and v["rise"]["ok"] and v["rise"]["entry_ok"] and v["arise"]["balance_ok"]
    assert all(n == 0 for n in v["limits_outside"].values()) and v["hand_limits_ok"]
    assert rec["solve_settings"] == json.loads(json.dumps(MO.FOLD_SOLVE))


def test_the_solver_digest_covers_what_the_solve_reads(monkeypatch):
    """Every function of motions that solve_fold reaches (an AST walk of the module) is in FOLD_SOLVER_CODE and every
    module value it reads is in FOLD_SOLVER_VALUES (but the fold's own entry, ARM_POSTURES, and the record's path and
    state); the base postures it reads move the digest (review of task sh1c: _arise_info, _mirror and POSTURES3 were
    left out)."""
    import ast
    tree = ast.parse(open(MO.__file__, encoding="utf-8").read())
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    values = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    seen, todo, used = set(), ["solve_fold"], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for x in ast.walk(defs[name]):
            if isinstance(x, ast.Name) and x.id in defs:
                todo.append(x.id)
            elif isinstance(x, ast.Name) and x.id in values:
                used.add(x.id)
    assert seen <= set(MO.FOLD_SOLVER_CODE), sorted(seen - set(MO.FOLD_SOLVER_CODE))
    assert used - {"ARM_POSTURES", "FOLD_RECORD", "_FOLD_STATE"} <= set(MO.FOLD_SOLVER_VALUES), sorted(used)
    d = MO.fold_solver_digest()
    orig = MO.posture_rotations
    monkeypatch.setattr(MO, "posture_rotations", lambda name: orig("stand"))      # another seated posture
    assert MO.fold_solver_digest() != d


def test_the_solver_digest_reads_code_not_docstrings():
    """A docstring, a comment or a line break leaves the digest's text alone; a change of the code does not."""
    def one():
        def f(x):
            """A docstring."""
            return x + 1
        return f

    def reworded():
        def f(x):
            """Another docstring, longer."""
            # a comment
            return (x
                    + 1)
        return f

    def other():
        def f(x):
            """A docstring."""
            return x + 2
        return f
    assert MO._code_digest_text(one()) == MO._code_digest_text(reworded())
    assert MO._code_digest_text(one()) != MO._code_digest_text(other())
    # source text, not ast.dump: the dump's fields change with the Python version, which would make the record stale
    assert MO._code_digest_text(one()) == "def f(x):\n    return x + 1"


def test_a_stale_record_is_refused(monkeypatch):
    monkeypatch.setitem(MO._FOLD_STATE, "state", "stale")
    monkeypatch.delitem(MO.ARM_POSTURES, "fold")
    with pytest.raises(RuntimeError, match="solve_fold_posture"):
        MO.arm_rotations("fold", "left")
    assert len(MO.arm_rotations("fold", "left", _record()["fold"])) == 4           # an explicit fold is read


def test_the_solve_reproduces_its_record():
    res = MO.solve_fold()
    rec = _record()
    for s in ("left", "right"):
        np.testing.assert_allclose(np.asarray(res["fold"][s]), np.asarray(rec["fold"][s]), atol=0.011)
    np.testing.assert_allclose(MO.fold_q(res["ik"]), MO.fold_q(rec["solve"]["ik"]), atol=0.011)


# ---------------------------------------------------------------- opening the fold (task sh1c)
def test_each_arm_takes_its_own_clock():
    s = np.linspace(-0.1, 1.1, 13)
    both = MO.fold_open_shares(s)
    assert np.array_equal(both["left"], both["right"]) and both["left"].min() == 0.0 and both["left"].max() == 1.0
    sh = MO.fold_open_shares(s, 0.2, "left")
    assert sh["right"][s >= 0.8].min() == 1.0 and sh["left"][s <= 0.2].max() == 0.0     # one ends, one starts late
    np.testing.assert_allclose(sh["right"][(s > 0) & (s < 0.8)], s[(s > 0) & (s < 0.8)] / 0.8)


def test_unfold_to_stand_keeps_the_skins_apart_at_the_pkd_unfold(monkeypatch):
    from mdfm.synth import postures as PO
    b0, ainfo = MO.arise(standing_s=PO.DUR_S)
    t = np.arange(b0.T) / b0.rate
    t_lift = ainfo["onset_s"] + MO.LIFTOFF_SHARE * ainfo["rise_s"]
    up = np.flatnonzero((t >= t_lift) & (PO.trunk_flexion_deg(b0) <= PO.PKD["unfold_trunk_deg"]))
    k0, n = int(up[0]), int(round(PO.PKD["unfold_dur_s"] * b0.rate))
    R, Rh, w, rec = MO.unfold_to_stand(b0, k0, n)
    # the smallest lag whose rehearsal keeps the skins apart (D26, open-body port wave 6: on SMPL+H's fold the
    # one-clock try put the right hand into the left forearm; the open body's fold opens on one clock, its lag 0)
    one_ok = rec["tried"][0]["min_mm"] >= 1000.0 * MO.FOLD_OPEN_CLEAR_M
    assert rec["tried"][0]["lag"] == 0.0 and (rec["lag"] == 0.0) == one_ok
    assert one_ok or rec["lag_side"] in ("left", "right")
    b = b0.with_rotations(R, list(range(13, 22))).with_hands(Rh)
    fr = MO.fold_report(b, np.arange(k0, min(b.T, k0 + n + 30)), MO.arise_chair(b, 0.0), checks=MO.FOLD_OPEN_CHECKS)
    assert fr["entry_ok"], fr
    assert float((b.local_R()[:k0] - b0.local_R()[:k0]).abs().max()) < 1e-12           # the fold kept before
    assert torch.equal(R[:k0], b0.local_R()[:k0]) and torch.equal(Rh[:k0], b0.hand_R()[:k0])
    st = MO.posture_rotations("stand")
    assert float((b.local_R()[k0 + n + 1:, 16] - st[16]).abs().max()) < 1e-9          # the arms hang after
    assert min(w["left"][k0 + n:].min(), w["right"][k0 + n:].min()) == 1.0
    c = MO.unfold_check(b, k0, n)                  # the validity entry: every knot of the opening and after it
    assert c["ok"] and c["knots"] == [k0, k0 + n + int(round(MO.UNFOLD_CHECK_PAD_S * b.rate)) - 1], c
    assert set(c["at_s"]) == set(MO.FOLD_OPEN_CHECKS) and min(c[x] for x in MO.FOLD_OPEN_CHECKS) >= 0.0
    monkeypatch.setattr(MO, "fold_open_lag", lambda realize, n_: (0.0, None, {"tried": []}))     # one clock
    R1, Rh1, _, _ = MO.unfold_to_stand(b0, k0, n)
    one = MO.unfold_check(b0.with_rotations(R1, list(range(13, 22))).with_hands(Rh1), k0, n)
    assert one["ok"] == one_ok, one            # the one-clock opening refused where its rehearsal entered (SMPL+H)
