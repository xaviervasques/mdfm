"""Tests of scripts/g3_supervised_baseline.py on synthetic records and targets (no weargait file is read): the rule
hash is asserted before any target is read (a changed rule, hash file or script constant stops the run before the
loader is called; the rule file of the tree matches the constant); walkway passes as units (interval, mean stride,
pass speed, a pass with one of the two targets only); window centres and starts (inside the pass, at least one, kept
inside the record); the input as the model reads it in G3 (magnetometer dropped, invalid samples, masked channels
and absent sites as NaN); the window features are those of scripts/g2_renderer_realism.py; the lockbox guards (a
lockbox person in a set, a record with lockbox True or any flag other than a plain False, a record of a person outside
the set); predictions equal to the targets score zero through the G3 lines, and each walkway stride and pass is scored
against the prediction of its own pass; the whole pipeline on synthetic records learns speed and stride better than
the population baseline, on the full dev set and on the set without the two excluded persons; the person-grouped
cross-validation keeps each person out of its own fit; the report carries the label and no em-dash; main() on 54 + 14
synthetic controls asserts the rule before it reads a target, writes results.json, summary.md and _provenance.json
with the label, newer than the rule files, and stops without writing on a lockbox person or a lockbox record.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
import g3_supervised_baseline as B  # noqa: E402
from mdfm.gates import g3  # noqa: E402

G2 = B.G2
EM_DASH = chr(0x2014)
LOC_IDS = np.array([G2.SITES.index(s) for s in G2.SITES13])
RATE = 100.0


# ---------------------------------------------------------------- synthetic data
def _gait(i, n_persons):
    """Speed (m/s) and cadence (steps/s) of synthetic person i."""
    speed = 0.8 + 0.8 * i / max(n_persons - 1, 1)
    return speed, 1.6 + 0.25 * speed


def _record(person, T, speed, cadence, seed, lockbox=False):
    """A synthetic weargait-like IMU record (T, 13, 9) whose amplitudes and frequencies follow speed and cadence."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / RATE
    sig = np.zeros((T, 13, 9), np.float32)
    for j in range(13):
        ph = 0.3 * j
        sig[:, j, 0] = 1.0 + 0.4 * speed * np.sin(2 * np.pi * cadence * t + ph)
        sig[:, j, 1] = 0.3 * speed * np.cos(2 * np.pi * cadence * t + ph)
        sig[:, j, 2] = 0.1 * speed * np.sin(4 * np.pi * cadence * t)
        sig[:, j, 3:6] = (120.0 * speed * np.sin(np.pi * cadence * t + ph))[:, None] * np.array([1.0, 0.5, 0.2])
        sig[:, j, 6:9] = 40.0
    sig += rng.normal(0.0, 0.02, sig.shape).astype(np.float32)
    return {"signal": sig, "location_ids": LOC_IDS.copy(), "sample_valid": np.ones((T, 13), bool),
            "channel_mask": np.ones((13, 9), bool), "lockbox": lockbox, "person_id": person, "tier": "TRAIN",
            "fps": RATE}


def _targets(persons, gaits, tasks=("SelfPace", "HurriedPace"), passes=3, T=3000):
    """Targets of synthetic persons: per trial `passes` straight passes of 4 s (ICs alternating feet one step apart,
    strides of the same foot two footfalls apart, stride = speed x 2 / cadence), pass speed = speed."""
    trials, ics, strides, ps = [], [], [], []
    for p, (v0, c0) in zip(persons, gaits):
        for task in tasks:
            v, c = (v0, c0) if task == "SelfPace" else (1.2 * v0, 1.1 * c0)
            tid = f"{p.split(':')[1]}_{task}"
            trials.append((tid, p, task, T))
            for k in range(passes):
                t = np.round(2.0 + 8.0 * k + np.arange(0.0, 4.0, 1.0 / c), 2)
                feet = ["L" if i % 2 == 0 else "R" for i in range(len(t))]
                ics += [(tid, p, task, f, float(x)) for f, x in zip(feet, t)]
                for i in range(2, len(t)):
                    strides.append((tid, p, task, str(k), feet[i], float(t[i - 2]), float(t[i]), v * 2.0 / c))
                ps.append((tid, p, task, str(k), float(t[0]), float(t[-1]), len(t), v))
    tg = g3.Targets(tuple(sorted(persons)), False,
                    pd.DataFrame(trials, columns=["trial_id", "person_id", "task", "samples"]),
                    pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"]),
                    pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"]),
                    pd.DataFrame(strides, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                   "length_m"]),
                    pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                              "speed_m_s"]),
                    pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})
    return tg


def _records(tg, gaits_by_person, seed=0):
    out = []
    for r in tg.trials.itertuples():
        v, c = gaits_by_person[r.person_id]
        if r.task != "SelfPace":
            v, c = 1.2 * v, 1.1 * c
        out.append((r.trial_id, _record(r.person_id, r.samples, v, c, seed + len(out))))
    return out


def _cohort(n_train=12, n_dev=6):
    names = [f"weargait:P{i:02d}" for i in range(n_train + n_dev)]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(names))
    gaits = {names[i]: _gait(int(order[i]), len(names)) for i in range(len(names))}
    train, dev = names[:n_train], names[n_train:]
    tg_tr = _targets(train, [gaits[p] for p in train])
    tg_dev = _targets(dev, [gaits[p] for p in dev])
    return train, dev, gaits, tg_tr, tg_dev


# ---------------------------------------------------------------- rule hash
def _rule_files(tmp_path, text="rule v1\n"):
    rule = tmp_path / "rule.txt"
    rule.write_bytes(text.encode("utf-8"))
    sha = B.sha256_file(str(rule))
    (tmp_path / "rule.sha256").write_text(f"{sha}  rule.txt\n", encoding="utf-8")
    return str(rule), str(tmp_path / "rule.sha256"), sha


def test_the_rule_hash_is_asserted_before_any_target_is_read(tmp_path):
    rule, shaf, sha = _rule_files(tmp_path)
    calls = []

    def loader(persons):
        calls.append(tuple(persons))
        return _targets(["weargait:A"], [(1.0, 1.8)])

    got, tgs = B.checked_targets({"dev": ["weargait:A"]}, loader, rule, shaf, sha)
    assert got == sha and calls == [("weargait:A",)] and set(tgs) == {"dev"}
    calls.clear()
    with open(rule, "ab") as f:                                 # the rule edited after its hash was written
        f.write(b"one more line\n")
    with pytest.raises(AssertionError, match="rule hash mismatch"):
        B.checked_targets({"dev": ["weargait:A"]}, loader, rule, shaf, sha)
    assert calls == []
    rule, shaf, sha = _rule_files(tmp_path, "rule v2\n")         # file and hash file agree, the script does not
    with pytest.raises(AssertionError):
        B.checked_targets({"dev": ["weargait:A"]}, loader, rule, shaf, "0" * 64)
    assert calls == []


def test_a_kept_lockbox_target_row_stops_the_run(tmp_path):
    rule, shaf, sha = _rule_files(tmp_path)

    def loader(persons):
        tg = _targets(["weargait:A"], [(1.0, 1.8)])
        tg.counts["lockbox_rows"] = 1
        return tg

    with pytest.raises(AssertionError, match="lockbox"):
        B.checked_targets({"dev": ["weargait:A"]}, loader, rule, shaf, sha)


def test_the_rule_file_of_the_tree_matches_the_script():
    if os.path.isfile(os.path.join(TREE, "PUBLIC_TREE.md")) and not os.path.exists(B.RULE_PATH):
        pytest.skip("the public copy does not carry reports/bench/g3_supervised_baseline/rule.txt")
    assert os.path.exists(B.RULE_PATH) and os.path.exists(B.RULE_SHA_PATH), "the rule files are part of the tree"
    assert B.check_rule() == B.RULE_SHA256
    with open(B.RULE_PATH, encoding="utf-8") as f:
        text = f.read()
    assert EM_DASH not in text and "54 training-fold" in text and "WHC003" in text and "WHC026" in text


# ---------------------------------------------------------------- units and windows
def test_pass_units_span_the_pass_and_hold_both_targets():
    tg = _targets(["weargait:A"], [(1.0, 2.0)], tasks=("SelfPace",), passes=2)
    tg.passes = tg.passes[tg.passes["pass_id"] != "1"].reset_index(drop=True)     # pass 1: strides only
    extra = tg.passes.iloc[[0]].assign(pass_id="9", t0=20.0, t1=23.0, speed_m_s=1.3)  # pass 9: speed only
    tg.passes = pd.concat([tg.passes, extra], ignore_index=True)
    u = B.pass_units(tg).set_index("pass_id")
    s0 = tg.strides[tg.strides["pass_id"] == "0"]
    assert u.at["0", "t0"] == pytest.approx(min(s0["t0"].min(), tg.passes.at[0, "t0"]))
    assert u.at["0", "t1"] == pytest.approx(max(s0["t1"].max(), tg.passes.at[0, "t1"]))
    assert u.at["0", "stride_m"] == pytest.approx(s0["length_m"].mean()) and u.at["0", "speed_m_s"] == 1.0
    assert u.at["0", "n_strides"] == len(s0)
    assert np.isnan(u.at["1", "speed_m_s"]) and np.isfinite(u.at["1", "stride_m"])
    assert np.isnan(u.at["9", "stride_m"]) and u.at["9", "n_strides"] == 0 and u.at["9", "speed_m_s"] == 1.3
    assert (u.at["9", "t0"], u.at["9", "t1"]) == (20.0, 23.0)


def test_window_centres_lie_in_the_pass_and_starts_inside_the_record():
    c = B.window_centres(2.0, 5.5)
    assert len(c) == 4 and np.allclose(np.diff(c), 1.0) and min(c) >= 2.0 and max(c) <= 5.5
    assert np.mean(c) == pytest.approx(3.75)
    assert B.window_centres(4.0, 4.3) == [pytest.approx(4.15)]
    assert B.window_centres(1.0, 4.0) == [pytest.approx(x) for x in (1.0, 2.0, 3.0, 4.0)]
    assert B.window_starts(3.0, 3.0, 1000) == [200]
    assert B.window_starts(0.2, 0.2, 1000) == [0]                      # shifted inside the record
    assert B.window_starts(9.9, 9.9, 1000) == [800]
    assert B.window_starts(0.5, 0.5, 150) == []                        # record shorter than a window
    for s in B.window_starts(0.0, 9.99, 1000):
        assert 0 <= s <= 1000 - B.WIN


def test_the_input_is_acc_and_gyro_with_invalid_masked_and_absent_sites_nan():
    rec = _record("weargait:A", 400, 1.0, 1.8, 0)
    rec["sample_valid"][10:20, 3] = False
    rec["channel_mask"][5, 3:6] = False
    keep = [j for j in range(13) if j != 7]                            # drop the record's 8th sensor
    rec["signal"], rec["location_ids"] = rec["signal"][:, keep], rec["location_ids"][keep]
    rec["sample_valid"], rec["channel_mask"] = rec["sample_valid"][:, keep], rec["channel_mask"][keep]
    before = rec["signal"].copy()
    x = B.site_array(rec)
    assert x.shape == (400, 13, 6) and x.dtype == np.float32
    assert np.array_equal(rec["signal"], before)                       # the record is left unchanged
    absent = G2.SITES13.index(G2.SITES[int(LOC_IDS[7])])
    assert np.isnan(x[:, absent]).all()
    s3 = G2.SITES13.index(G2.SITES[int(LOC_IDS[3])])
    assert np.isnan(x[10:20, s3]).all() and np.isfinite(x[20:, s3]).all()
    s5 = G2.SITES13.index(G2.SITES[int(LOC_IDS[5])])
    assert np.isnan(x[:, s5, 3:6]).all() and np.isfinite(x[:, s5, 0:3]).all()
    others = [s for s in range(13) if s not in (absent, s3, s5)]
    assert np.isfinite(x[:, others]).all()
    assert np.allclose(x[:, others[0]], before[:, list(LOC_IDS[keep]).index(G2.SITES.index(G2.SITES13[others[0]])), :6])


def test_window_features_are_the_g2_features():
    x = B.site_array(_record("weargait:A", 600, 1.2, 1.9, 1))
    f = B.window_features(x, [0, 150, 400])
    assert f.shape == (3, B.N_FEATURES) and B.N_FEATURES == 624
    assert np.allclose(f[1], np.asarray(G2.features(x[150:150 + B.WIN]), np.float32))
    x[160, 4, 2] = np.nan                                               # one channel of one site, one sample
    g = B.window_features(x, [150])[0]
    assert np.isnan(g).sum() == 5 + len(G2.BANDS)                       # that channel's features only
    assert np.allclose(g[np.isfinite(g)], f[1][np.isfinite(g)])


# ---------------------------------------------------------------- lockbox guards
def test_lockbox_persons_and_records_are_refused():
    with pytest.raises(AssertionError, match="lockbox"):
        B.assert_no_lockbox({"train": ["weargait:A"], "dev": ["weargait:B"]}, ["weargait:B"])
    B.assert_no_lockbox({"train": ["weargait:A"]}, ["weargait:C"])
    tg = _targets(["weargait:A"], [(1.0, 1.8)], tasks=("SelfPace",))
    u = B.pass_units(tg)
    rec = _record("weargait:A", 3000, 1.0, 1.8, 0, lockbox=True)
    with pytest.raises(AssertionError, match="lockbox"):
        B.build_rows([("A_SelfPace", rec)], u, ["weargait:A"])
    rec = _record("weargait:A", 3000, 1.0, 1.8, 0)
    with pytest.raises(AssertionError, match="outside"):
        B.build_rows([("A_SelfPace", rec)], u, ["weargait:B"])
    rec["tier"] = "EVAL"
    with pytest.raises(AssertionError, match="tier"):
        B.build_rows([("A_SelfPace", rec)], u, ["weargait:A"])
    X, idx, c = B.build_rows([("A_SelfPace", _record("weargait:A", 3000, 1.0, 1.8, 0))], u, ["weargait:A"])
    assert c["passes_with_windows"] == len(u) and X.shape == (len(idx), B.N_FEATURES)


def test_a_record_whose_lockbox_flag_is_not_plainly_false_is_refused():
    for flag in (None, True, np.True_, "True", "true", 1, "1", "yes"):
        with pytest.raises(AssertionError, match="lockbox"):
            B.check_record("A_SelfPace", _record("weargait:A", 10, 1.0, 1.8, 0, lockbox=flag), ["weargait:A"])
    for flag in (False, np.False_, "False", "false", 0, "0"):
        B.check_record("A_SelfPace", _record("weargait:A", 10, 1.0, 1.8, 0, lockbox=flag), ["weargait:A"])
    rec = _record("weargait:A", 10, 1.0, 1.8, 0)
    rec["tier"] = "EXCLUDED_V1"
    with pytest.raises(AssertionError, match="tier"):
        B.check_record("A_SelfPace", rec, ["weargait:A"])
    rec = _record("weargait:A", 10, 1.0, 1.8, 0)
    del rec["person_id"]
    with pytest.raises(AssertionError, match="outside"):
        B.check_record("A_SelfPace", rec, ["weargait:A"])


# ---------------------------------------------------------------- scoring
def test_predictions_equal_to_the_targets_score_zero_through_the_g3_lines():
    tg = _targets(["weargait:A", "weargait:B"], [(1.0, 1.8), (1.4, 2.0)])
    u = B.pass_units(tg)
    p = B.predictions_of(u, {"stride_m": u["stride_m"].to_numpy(), "speed_m_s": u["speed_m_s"].to_numpy()}, tg)
    assert len(p.strides) == len(tg.strides) and len(p.speeds) == len(tg.passes)
    s = B.score(tg, [p])["supervised"]
    assert s["stride_mae_cm"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert s["speed_mae_m_s"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert s["stride_mae_cm"]["matched_share"] == 1.0 and s["speed_mae_m_s"]["matched_share"] == 1.0
    off = {"stride_m": u["stride_m"].to_numpy() + 0.05, "speed_m_s": u["speed_m_s"].to_numpy() - 0.1}
    s = B.score(tg, [B.predictions_of(u, off, tg)])["supervised"]
    assert s["stride_mae_cm"]["point"] == pytest.approx(5.0) and s["speed_mae_m_s"]["point"] == pytest.approx(0.1)
    assert s["stride_mae_cm"]["interval"] is not None and s["stride_mae_cm"]["n_persons"] == 2
    nan = {"stride_m": np.where(u["person_id"] == "weargait:B", np.nan, u["stride_m"]), "speed_m_s": u["speed_m_s"]}
    s = B.score(tg, [B.predictions_of(u, nan, tg)])["supervised"]   # a pass without prediction is unmatched
    assert s["stride_mae_cm"]["persons_unmatched"] == ["weargait:B"]


def test_each_walkway_stride_and_pass_gets_the_prediction_of_its_own_pass():
    tg = _targets(["weargait:A", "weargait:B"], [(1.0, 1.8), (1.4, 2.0)], passes=3)
    u = B.pass_units(tg)
    own = 1.0 + 0.01 * np.arange(len(u))                               # a distinct prediction per pass
    p = B.predictions_of(u, {"stride_m": own, "speed_m_s": own + 0.5}, tg)
    want = dict(zip(zip(u["trial_id"], u["pass_id"]), own))
    m = g3.match_strides(p, tg)
    assert len(m) == len(tg.strides) and m["matched"].all() and (m["end_diff_s"] == 0).all()
    assert np.allclose(m["pred_m"], [want[(t, str(k))] for t, k in zip(m["trial_id"], m["pass_id"])])
    sp = g3.match_speeds(p, tg)
    assert sp["matched"].all()
    assert np.allclose(sp["pred_m_s"], [want[(t, str(k))] + 0.5 for t, k in zip(sp["trial_id"], sp["pass_id"])])


def test_the_pipeline_learns_gait_from_synthetic_records():
    train, dev, gaits, tg_tr, tg_dev = _cohort()
    excluded = tuple(dev[:2])
    res = B.run(tg_tr, tg_dev, _records(tg_tr, gaits), _records(tg_dev, gaits, seed=100), train, dev,
                excluded=excluded, cv=False, log=lambda *a, **k: None)
    assert res["label"] == B.LABEL and res["cv_train"] is None
    assert res["sets"]["dev14"] == sorted(dev) and res["sets"]["dev12"] == sorted(set(dev) - set(excluded))
    c = res["counts"]
    assert c["train"]["passes_with_windows"] == c["train"]["passes"] == len(tg_tr.passes)
    assert c["dev"]["records"] == len(tg_dev.trials) and c["dev"]["passes_without_window"] == 0
    for sname, n in (("dev14", len(dev)), ("dev12", len(dev) - 2)):
        sup, pop, ww = (res["lines"][sname][k] for k in ("supervised", "population", "walkway"))
        assert sup["stride_mae_cm"]["n_persons"] == n and sup["speed_mae_m_s"]["n_persons"] == n
        assert sup["stride_mae_cm"]["matched_share"] == 1.0 and sup["speed_mae_m_s"]["matched_share"] == 1.0
        assert ww["stride_mae_cm"]["point"] == pytest.approx(0.0, abs=1e-9)
        assert sup["speed_mae_m_s"]["point"] < 0.6 * pop["speed_mae_m_s"]["point"]
        assert sup["stride_mae_cm"]["point"] < 0.6 * pop["stride_mae_cm"]["point"]
        lo, hi = sup["speed_mae_m_s"]["interval"]
        assert lo <= sup["speed_mae_m_s"]["point"] <= hi
    assert not set(excluded) & set(res["lines"]["dev12"]["supervised"]["stride_mae_cm"]["per_person"])
    pp = res["per_pass"]
    assert len(pp) == len(tg_dev.passes) and np.isfinite(pp["pred_speed_m_s"]).all()


def test_the_grouped_cross_validation_keeps_each_person_out_of_its_fit():
    train, _, gaits, tg_tr, _ = _cohort(n_train=10, n_dev=0)
    u = B.pass_units(tg_tr)
    X, idx, _ = B.build_rows(_records(tg_tr, gaits), u, train)
    seen = []
    real = B.fit_and_predict_units

    def spy(X_tr, idx_tr, units_tr, X_te, idx_te, n_te, params=None):
        tr_p, te_p = set(units_tr["person_id"].to_numpy()[idx_tr]), set(units_tr["person_id"].to_numpy()[idx_te])
        seen.append((tr_p, te_p))
        return real(X_tr, idx_tr, units_tr, X_te, idx_te, n_te, params)

    small = dict(B.HGB_PARAMS, max_iter=20, min_samples_leaf=5)
    B.fit_and_predict_units = spy
    try:
        oof = B.grouped_cv(X, idx, u, folds=5, params=small)
    finally:
        B.fit_and_predict_units = real
    assert len(seen) == 5 and all(not (a & b) for a, b in seen)
    assert set().union(*(b for _, b in seen)) == set(train)
    assert np.isfinite(oof["stride_m"]).all() and np.isfinite(oof["speed_m_s"]).all()


def test_the_report_carries_the_label_and_no_em_dash(tmp_path):
    train, dev, gaits, tg_tr, tg_dev = _cohort(n_train=8, n_dev=4)
    small = dict(B.HGB_PARAMS, max_iter=20, min_samples_leaf=5)
    res = B.run(tg_tr, tg_dev, _records(tg_tr, gaits), _records(tg_dev, gaits, seed=50), train, dev,
                excluded=tuple(dev[:2]), params=small, cv=True, log=lambda *a, **k: None)
    rule, shaf, sha = _rule_files(tmp_path)
    out = B.finalize(res, sha, train, dev, ["weargait:Z"], {"lockbox_rows": 0}, {"lockbox_rows": 0}, 0,
                     {"head": "abc123"}, "2026-10-01T12:00:00", rule, shaf)
    text = B.summary_md(out)
    assert B.LABEL in text and EM_DASH not in text and "abc123" in text
    assert "| dev14 (4) | supervised |" in text and "| dev12 (2) | supervised |" in text
    assert out["lockbox"]["persons_in_lockbox"] == 0 and out["cv_train"]["supervised_cv"]["stride_mae_cm"]["n_persons"] == 8
    with pytest.raises(AssertionError, match="lockbox"):
        B.finalize(res, sha, train, dev, [dev[0]], {"lockbox_rows": 0}, {"lockbox_rows": 0}, 0, {}, "t", rule, shaf)


# ---------------------------------------------------------------- main, on synthetic persons
def _main_setup(monkeypatch, tmp_path, locked=("weargait:LB001",), lockbox_trial=None):
    """main() on 54 synthetic training and 14 synthetic dev controls (WHC003 and WHC026 among them): the g3
    readers, the priority, the SLOW pause and the MoveAtlas stamp are replaced, the outputs go to tmp_path,
    the rule is the tree's own. Returns (calls log, train, dev)."""
    train = sorted(f"weargait:TR{i:03d}" for i in range(54))
    dev = sorted([f"weargait:DV{i:03d}" for i in range(12)] + list(B.EXCLUDED_FROM_12))
    names = sorted(train + dev)
    order = np.random.default_rng(11).permutation(len(names))
    gaits = {p: _gait(int(order[i]), len(names)) for i, p in enumerate(names)}
    tgs = {"train": _targets(train, [gaits[p] for p in train], passes=2, T=2000),
           "dev": _targets(dev, [gaits[p] for p in dev], passes=2, T=2000)}
    calls = []
    real_check = B.check_rule

    def check_rule(*a, **k):
        calls.append("rule")
        return real_check(*a, **k)

    def load_targets(persons, *a, **k):
        calls.append("targets")
        return tgs["train"] if set(persons) == set(train) else tgs["dev"]

    def imu_records(persons, trials, *a, **k):
        calls.append("records")
        for r in trials.itertuples():
            v, c = gaits[r.person_id]
            if r.task != "SelfPace":
                v, c = 1.2 * v, 1.1 * c
            yield r.trial_id, _record(r.person_id, r.samples, v, c, int(r.Index), lockbox=r.trial_id == lockbox_trial)

    monkeypatch.setattr(g3, "controls", lambda fold: list(train if fold == "train" else dev))
    monkeypatch.setattr(g3, "lockbox_persons", lambda: sorted(locked))
    monkeypatch.setattr(g3, "load_targets", load_targets)
    monkeypatch.setattr(g3, "imu_records", imu_records)
    monkeypatch.setattr(B, "check_rule", check_rule)
    monkeypatch.setattr(B, "OUT", str(tmp_path))
    monkeypatch.setattr(B, "below_normal", lambda: "unchanged (test)")
    monkeypatch.setattr(B, "wait_while_slow", lambda *a, **k: 0.0)
    monkeypatch.setattr(B, "_stamp", lambda: {"head": "abc123"})
    monkeypatch.setattr(B, "HGB_PARAMS", dict(B.HGB_PARAMS, max_iter=20, min_samples_leaf=5))
    return calls, train, dev


def test_main_asserts_the_rule_before_the_targets_and_writes_the_labelled_outputs(tmp_path, monkeypatch):
    import json
    calls, train, dev = _main_setup(monkeypatch, tmp_path)
    B.main()
    assert calls[0] == "rule" and calls.index("targets") > calls.index("rule")
    assert calls.index("records") > max(i for i, c in enumerate(calls) if c == "targets")
    files = {f: (tmp_path / f).read_bytes().decode("utf-8") for f in ("results.json", "summary.md", "_provenance.json")}
    for f, text in files.items():
        assert B.LABEL in text and EM_DASH not in text and "\r" not in text, f
    res, prov = json.loads(files["results.json"]), json.loads(files["_provenance.json"])
    assert res["rule"]["sha256"] == prov["rule_sha256"] == B.RULE_SHA256
    assert res["lockbox"] == {"persons_in_lockbox": 0, "target_lockbox_rows": 0, "records_lockbox_true": 0,
                              "asserted": True, "reader_flags": {"allow_eval": False, "allow_lockbox": False}}
    assert res["sets"]["train"] == train and res["sets"]["dev14"] == dev
    assert res["sets"]["dev12"] == sorted(set(dev) - set(B.EXCLUDED_FROM_12)) and len(res["sets"]["dev12"]) == 12
    for sname, n in (("dev14", 14), ("dev12", 12)):
        for line in ("stride_mae_cm", "speed_mae_m_s"):
            st = res["lines"][sname]["supervised"][line]
            assert st["n_persons"] == n and st["matched_share"] == 1.0 and len(st["interval"]) == 2
    assert res["cv_train"]["supervised_cv"]["stride_mae_cm"]["n_persons"] == 54
    assert "| dev14 (14) | supervised |" in files["summary.md"] and "| dev12 (12) | supervised |" in files["summary.md"]
    assert prov["moveatlas_stamp"] == {"head": "abc123"} and res["moveatlas"]["head"] == "abc123"
    rule_t = max(os.path.getmtime(B.RULE_PATH), os.path.getmtime(B.RULE_SHA_PATH))
    assert all(os.path.getmtime(tmp_path / f) > rule_t for f in files)


def test_main_stops_on_a_lockbox_person_before_the_rule_and_the_targets(tmp_path, monkeypatch):
    calls, _, dev = _main_setup(monkeypatch, tmp_path, locked=("weargait:DV003",))
    with pytest.raises(AssertionError, match="lockbox"):
        B.main()
    assert calls == [] and not list(tmp_path.iterdir())


def test_main_stops_on_a_lockbox_record_and_writes_nothing(tmp_path, monkeypatch):
    calls, _, _ = _main_setup(monkeypatch, tmp_path, lockbox_trial="DV004_HurriedPace")
    with pytest.raises(AssertionError, match="lockbox"):
        B.main()
    assert "records" in calls and not list(tmp_path.iterdir())
