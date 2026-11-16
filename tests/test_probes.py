"""Tests of mdfm/eval/probes.py and scripts/probe_frozen_tokens.py on synthetic features, targets and windows (no
weargait file, record or checkpoint is read):

- person cross-fitting: every row of a person lands in one fold, no fold puts a person on both sides, check_disjoint
  refuses an overlap, cross_fit and the penalty choice inside it never see a scored person, and a feature that names
  the person but not the target gains nothing out of fold (it would, were a person on both sides);
- ridge: the closed form equals the normal equations on standardized features, a constant column is harmless, a
  non-finite row predicts NaN;
- known strides are recovered on synthetic features (fit on some persons, scored on others, far better than the
  population mean), through the G3 lines on synthetic walkway targets (predictions equal to the targets score 0, a
  pass without a prediction is unmatched);
- pooling of spans, the most central window of a time, the IC labels per sub-patch and their decoding (labels as
  scores give back every IC within half a sub-patch; a ridge on features that carry the labels learns them);
- the G10-style leakage hook: a feature carrying the label leaks, noise does not, and the covariate probe without
  covariates is at chance;
- the script: encode_windows on a tiny model returns one row per latent patch of every window and the measurement
  mean of the tokenizer; trial_rows keeps the most central reading of each patch; the probe stage runs on synthetic
  features and writes a report with the label and no em-dash; the summary puts checkpoints side by side;
- the script's plumbing: the feature file round-trips and refuses another checkpoint's name; the decoded-body
  (diag.json), J1 (g3_dev.json, per_person.csv) and supervised files are read with their schemas into the paired
  comparison (an infinite J1 value left out); check_record refuses a lockbox or EVAL record and a person outside
  the set; collect pools each pass from its own trial's most central rows, leaves a trial without its record
  empty, and keeps the token rows near the walkway.
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
import probe_frozen_tokens as PF  # noqa: E402
from mdfm.eval import probes as P  # noqa: E402
from mdfm.gates import g3  # noqa: E402

EM_DASH = chr(0x2014)


# ---------------------------------------------------------------- synthetic walkway targets
def _targets(persons, gaits, tasks=("SelfPace", "HurriedPace"), passes=3, T=3000):
    """g3.Targets of synthetic persons: per trial `passes` passes of 4 s, ICs alternating feet one step apart,
    strides two footfalls apart (stride = speed x 2 / cadence), pass speed = speed."""
    trials, ics, strides, ps = [], [], [], []
    for p, (v0, c0) in zip(persons, gaits):
        for task in tasks:
            v, c = (v0, c0) if task == "SelfPace" else (1.2 * v0, 1.1 * c0)
            tid = f"{p.split(':')[1]}_{task}"
            trials.append((tid, p, task, T))
            for k in range(passes):
                t = np.round(2.0 + 8.0 * k + np.arange(0.0, 4.0, 1.0 / c), 3)
                feet = ["L" if i % 2 == 0 else "R" for i in range(len(t))]
                ics += [(tid, p, task, f, float(x)) for f, x in zip(feet, t)]
                for i in range(2, len(t)):
                    strides.append((tid, p, task, str(k), feet[i], float(t[i - 2]), float(t[i]), v * 2.0 / c))
                ps.append((tid, p, task, str(k), float(t[0]), float(t[-1]), len(t), v))
    ic = pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"])
    return g3.Targets(tuple(sorted(persons)), False,
                      pd.DataFrame(trials, columns=["trial_id", "person_id", "task", "samples"]), ic, ic.copy(),
                      pd.DataFrame(strides, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                     "length_m"]),
                      pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                                "speed_m_s"]),
                      pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})


def _cohort(n_train=20, n_dev=8, seed=7):
    names = [f"weargait:{'HC' if i % 2 else 'WHC'}{i:03d}" for i in range(n_train + n_dev)]
    rng = np.random.default_rng(seed)
    gaits = {n: (float(rng.uniform(0.8, 1.6)), float(rng.uniform(1.6, 2.1))) for n in names}
    train, dev = names[:n_train], names[n_train:]
    return train, dev, gaits, _targets(train, [gaits[p] for p in train]), _targets(dev, [gaits[p] for p in dev])


def _features(units, D=48, noise=0.05, seed=0):
    """Pass features carrying stride and speed linearly, plus a person-specific nuisance and noise."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal(D), rng.standard_normal(D)
    pers = {p: rng.standard_normal(D) for p in sorted(set(units["person_id"]))}
    X = (np.outer(units["stride_m"].to_numpy(), w1) + np.outer(units["speed_m_s"].to_numpy(), w2)
         + 0.1 * np.stack([pers[p] for p in units["person_id"]]) + noise * rng.standard_normal((len(units), D)))
    return X


# ---------------------------------------------------------------- cross-fitting
def test_cross_fitting_never_puts_one_person_on_both_sides(monkeypatch):
    persons = np.repeat([f"p{i:02d}" for i in range(23)], 7)
    folds = P.person_folds(persons, k=5, seed=3)
    for p in np.unique(persons):
        assert len(set(folds[persons == p])) == 1
    seen = np.zeros(len(persons), int)
    for tr, te in P.cross_fit_splits(persons, k=5, seed=3):
        assert not set(persons[tr]) & set(persons[te])
        seen[te] += 1
    assert (seen == 1).all()
    with pytest.raises(ValueError):
        P.check_disjoint(["a", "b"], ["b", "c"])
    with pytest.raises(ValueError):
        P.person_folds(["a", "b", "c"], k=5)
    # cross_fit: every fit (and the penalty choice inside it) holds no scored person
    calls = []
    real = P.fit_probe

    def spy(X, Y, pers, *a, **k):
        calls.append(set(np.asarray(pers).astype(str)))
        return real(X, Y, pers, *a, **k)

    monkeypatch.setattr(P, "fit_probe", spy)
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((len(persons), 4)), rng.standard_normal(len(persons))
    oof, alphas = P.cross_fit(X, y, persons, k=5, seed=3)
    assert len(calls) == 5 and len(alphas) == 5 and np.isfinite(oof).all()
    for (tr, te), fitted in zip(P.cross_fit_splits(persons, 5, 3), calls):
        assert fitted == set(persons[tr]) and not fitted & set(persons[te])


def test_a_person_identity_feature_gains_nothing_out_of_fold():
    rng = np.random.default_rng(1)
    ids = [f"p{i:02d}" for i in range(30)]
    persons = np.repeat(ids, 10)
    y_person = {p: rng.standard_normal() for p in ids}
    y = np.array([y_person[p] for p in persons]) + 0.05 * rng.standard_normal(len(persons))
    emb = {p: rng.standard_normal(40) for p in ids}
    X = np.stack([emb[p] for p in persons]) + 0.01 * rng.standard_normal((len(persons), 40))
    oof, _ = P.cross_fit(X, y, persons)
    base = np.mean(np.abs(y - y.mean()))
    assert np.mean(np.abs(oof - y)) > 0.8 * base           # identity alone predicts nothing for a new person
    m = P.fit_ridge(X, y, 1e-2)                            # in sample it would look excellent
    assert np.mean(np.abs(m.predict(X) - y)) < 0.3 * base


# ---------------------------------------------------------------- ridge
def test_ridge_equals_the_normal_equations_and_handles_constant_columns_and_nan():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 5))
    X[:, 3] = 7.0                                           # no variance
    y = X[:, 0] - 2 * X[:, 1] + 0.1 * rng.standard_normal(60)
    m = P.fit_ridge(X, y, 3.0)
    keep = [0, 1, 2, 4]
    mu, sd = X[:, keep].mean(0), X[:, keep].std(0)
    Z = (X[:, keep] - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + 3.0 * np.eye(4), Z.T @ (y - y.mean()))
    assert np.allclose(m.predict(X), Z @ beta + y.mean(), atol=1e-9)
    Xn = X.copy()
    Xn[4, 2] = np.nan
    p = m.predict(Xn)
    assert np.isnan(p[4]) and np.isfinite(np.delete(p, 4)).all()
    a, cv = P.select_alpha(X, y, np.repeat(np.arange(12), 5))
    assert a in P.ALPHAS and len(cv) == len(P.ALPHAS)


def test_known_strides_are_recovered_on_synthetic_features():
    train, dev, gaits, tg_tr, tg_dev = _cohort()
    u_tr, u_dev = P.pass_units(tg_tr), P.pass_units(tg_dev)
    assert np.allclose(u_tr["stride_m"], [gaits[p][0] * (1.0 if t == "SelfPace" else 1.2) * 2.0
                                          / (gaits[p][1] * (1.0 if t == "SelfPace" else 1.1))
                                          for p, t in zip(u_tr["person_id"], u_tr["task"])])
    P.check_disjoint(u_tr["person_id"], u_dev["person_id"])
    both = pd.concat([u_tr, u_dev], ignore_index=True)       # one feature map for both sets
    Xb = _features(both, seed=0)
    X_tr, X_dev = Xb[:len(u_tr)], Xb[len(u_tr):]
    pred = {}
    for t in ("stride_m", "speed_m_s"):
        m, _ = P.fit_probe(X_tr, u_tr[t].to_numpy(), u_tr["person_id"])
        pred[t] = m.predict(X_dev)
        spread = np.mean(np.abs(u_dev[t] - u_tr[t].mean()))
        assert np.mean(np.abs(pred[t] - u_dev[t])) < 0.15 * spread, t
    lines = P.score(tg_dev, [P.pass_predictions(u_dev, pred, tg_dev, "probe")])["probe"]
    assert lines["stride_mae_cm"]["point"] < 2.0 and lines["stride_mae_cm"]["matched_share"] == 1.0
    assert lines["speed_mae_m_s"]["point"] < 0.03 and lines["speed_mae_m_s"]["n_persons"] == len(dev)


def test_predictions_equal_to_the_targets_score_zero_and_a_missing_pass_is_unmatched():
    _, dev, _, _, tg = _cohort(n_train=5, n_dev=6)
    u = P.pass_units(tg)
    pred = {"stride_m": u["stride_m"].to_numpy().copy(), "speed_m_s": u["speed_m_s"].to_numpy().copy()}
    lines = P.score(tg, [P.pass_predictions(u, pred, tg, "x")])["x"]
    assert lines["stride_mae_cm"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert lines["speed_mae_m_s"]["point"] == pytest.approx(0.0, abs=1e-12)
    pred["stride_m"][0] = np.nan
    pred["speed_m_s"][0] = np.nan
    lines = P.score(tg, [P.pass_predictions(u, pred, tg, "x")])["x"]
    assert lines["stride_mae_cm"]["matched_share"] < 1.0 and lines["speed_mae_m_s"]["matched_share"] < 1.0
    sub = P.subset_targets(tg, dev[:3])
    assert set(sub.strides["person_id"]) == set(dev[:3]) and sub.persons == tuple(sorted(dev[:3]))


# ---------------------------------------------------------------- pooling, central windows, ICs
def test_pool_spans_gives_mean_then_std_and_nan_for_an_empty_span():
    t = np.array([0.1, 0.2, 0.3, 5.0])
    F = np.array([[1.0, 10.0], [3.0, 10.0], [np.nan, 1.0], [9.0, 9.0]])
    out, n = P.pool_spans(t, F, [(0.0, 0.35), (6.0, 7.0)])
    assert n.tolist() == [2, 0]                             # the row with a NaN is left out
    assert np.allclose(out[0], [2.0, 10.0, 1.0, 0.0]) and np.isnan(out[1]).all()


def test_most_central_picks_the_window_whose_middle_is_nearest_ties_to_the_earlier():
    starts, durs = [0.0, 5.0, 8.0], [10.0, 10.0, 10.0]
    t = np.array([1.0, 7.4, 7.5, 9.0, 12.0, 17.5])
    win = np.array([0, 0, 0, 1, 1, 2])
    assert P.most_central(starts, durs, win, t).tolist() == [True, True, True, True, False, True]
    # 7.5 s: windows 0 (middle 5) and 1 (middle 10) at equal distance: the earlier one
    assert P.most_central(starts, durs, [1], [7.5]).tolist() == [False]
    assert P.most_central(starts, durs, [2], [9.0]).tolist() == [False]       # window 1 is nearer (10 vs 13)


def _ic_tokens(tg):
    """Token starts every 0.25 s over each trial's first 30 s."""
    rows = [(t, s) for t in tg.trials["trial_id"] for s in np.arange(0.0, 30.0, 0.25)]
    return np.array([r[1] for r in rows]), np.array([r[0] for r in rows])


def test_ic_labels_per_sub_patch_decode_back_to_the_reference():
    _, _, _, _, tg = _cohort(n_train=5, n_dev=4)
    u = P.pass_units(tg)
    starts, tids = _ic_tokens(tg)
    Y, lab = P.ic_targets(starts, tids, tg.walkway_ic, u[["trial_id", "t0", "t1"]])
    assert Y.shape == (len(starts), 2 * P.SUB_BINS) and lab.any() and not lab.all()
    assert Y[~lab].sum() == 0                               # every IC's patch is labelled (first and last included)
    assert Y.sum() == len(tg.walkway_ic) and Y.sum(axis=1).max() <= 2
    ics = P.decode_ics(Y, starts, tids)
    w = P.PATCH_S / P.SUB_BINS
    ref = tg.walkway_ic
    for r in ref.itertuples():
        got = ics[(ics["trial_id"] == r.trial_id) & (ics["foot"] == r.foot)]["t_s"].to_numpy()
        assert np.min(np.abs(got - r.t_s)) <= w / 2 + 1e-9
    assert len(ics) == len(ref)
    # two hits of one foot within the NMS distance keep the higher score
    S = np.zeros((2, 2 * P.SUB_BINS))
    S[0, 1], S[1, 0] = 0.9, 0.6
    two = P.decode_ics(S, [1.0, 1.25], ["a", "a"])
    assert len(two) == 1 and two["t_s"].iloc[0] == pytest.approx(1.0 + 1.5 * w)


def test_a_ridge_learns_ic_sub_patches_from_features_that_carry_them():
    _, _, _, _, tg = _cohort(n_train=5, n_dev=14, seed=11)
    u = P.pass_units(tg)
    starts, tids = _ic_tokens(tg)
    persons = np.array([t.split("_")[0] for t in tids])
    Y, lab = P.ic_targets(starts, tids, tg.walkway_ic, u[["trial_id", "t0", "t1"]])
    rng = np.random.default_rng(5)
    M = rng.standard_normal((Y.shape[1], 64))
    X = Y @ M + 0.05 * rng.standard_normal((len(Y), 64))
    fit = np.isin(persons, persons[:len(persons) // 2])
    m, _ = P.fit_probe(X[fit & lab], Y[fit & lab], persons[fit & lab])
    sc = ~fit
    ics = P.decode_ics(m.predict(X[sc]), starts[sc], tids[sc])
    ref = tg.walkway_ic[tg.walkway_ic["trial_id"].isin(set(tids[sc]))]
    d = g3.ic_differences(ics, ref)
    assert d["matched"].mean() > 0.95 and np.median(np.abs(d["diff_s"])) <= 0.03


# ---------------------------------------------------------------- leakage hook
def test_the_leakage_hook_finds_a_label_carried_by_the_features_and_not_noise():
    rng = np.random.default_rng(4)
    persons = np.repeat([f"p{i:02d}" for i in range(24)], 6)
    lab = np.where(np.arange(len(persons)) // 6 % 2 == 0, "siteA", "siteB")
    noise = rng.standard_normal((len(persons), 10))
    carry = noise.copy()
    carry[:, 0] += 3.0 * (lab == "siteA")
    hit = P.leakage_probe(carry, lab, persons)
    null = P.leakage_probe(noise, lab, persons)
    assert hit["covariate_balanced_accuracy"] == pytest.approx(0.5)
    assert hit["balanced_accuracy"] > 0.9 and hit["leakage_points"] > 35
    assert abs(null["leakage_points"]) < 25 and null["chance"] == 0.5 and null["n_persons"] == 24
    cov = (lab == "siteA").astype(float)[:, None]           # a covariate that names the label: no leakage left
    assert P.leakage_probe(carry, lab, persons, covariates=cov)["leakage_points"] < 10


# ---------------------------------------------------------------- the script
def _tiny_model():
    import torch

    from mdfm.model import mfm as M
    torch.manual_seed(0)
    m = M.MFM(replace(M.TINY, skeleton=False))
    m.eval()
    return m


def test_encode_windows_returns_one_row_per_latent_patch_and_the_measurement_means():
    import torch

    import model_fixture as MF
    from mdfm.model import tokenize as TK
    model = _tiny_model()
    wins = [MF.make_window(2.0, seed=0), MF.make_window(1.3, seed=1)]
    wins = [replace(w, streams=[s for s in w.streams if s.modality == "imu"]) for w in wins]
    out = PF.encode_windows(model, wins, batch_windows=1)
    d = model.cfg.d
    for w, f in zip(wins, out):
        P_ = int(np.ceil(w.duration / 0.25 - 1e-9))
        assert f["latent"].shape == (P_, d) and f["mu_z"].shape == (P_, model.cfg.d_z)
        assert f["measurement"].shape == (P_, d) and f["subject"].shape == (d,)
        assert np.isfinite(f["latent"]).all() and (f["n_measurement"] > 0).all()
    # the stem mean equals the tokenizer output averaged over the patch's measurement tokens
    tb = TK.tokenize(wins[:1])
    with torch.no_grad():
        x = model.tokenizer(tb).reshape(-1, d).numpy()
    kind, t = tb.kind.reshape(-1).numpy(), tb.t.reshape(-1).numpy()
    sel = (kind == TK.MEASUREMENT) & (np.floor(t / 0.25) == 3)
    assert np.allclose(out[0]["stem"][3], x[sel].mean(axis=0), atol=1e-5)


def test_trial_rows_keep_the_most_central_reading_of_each_patch():
    class Win:
        def __init__(self, t0, dur):
            self.t0, self.duration = t0, dur
    wins = [Win(0.0, 10.0), Win(5.0, 10.0), Win(8.0, 10.0)]
    feats = [{"latent": np.full((40, 2), k, np.float32), "measurement": np.zeros((40, 2), np.float32),
              "stem": np.zeros((40, 2), np.float32), "mu_z": np.zeros((40, 1), np.float32),
              "subject": np.full(2, 10 + k, np.float32)} for k in range(3)]
    rows = PF.trial_rows(feats, wins)
    c = rows["centre"]
    assert len(np.unique(np.round(c, 6))) == len(c)          # one reading per patch time on windows 0 and 1
    assert np.all(np.diff(c[rows["window"] == 0]) > 0)
    assert rows["window"][np.argmin(np.abs(c - 9.125))] == 1  # 9.125 s: window 1 (middle 10) over window 0 (middle 5)
    assert (rows["subject"][:, 0] == 10 + rows["window"]).all()


def _synthetic_data(seed=0):
    train, dev, gaits, tg_tr, tg_dev = _cohort(n_train=20, n_dev=8, seed=seed)
    data = {"units": {"train": P.pass_units(tg_tr), "dev": P.pass_units(tg_dev)}}
    both = pd.concat([data["units"]["train"], data["units"]["dev"]], ignore_index=True)
    Xb = _features(both, D=24, seed=seed)
    n_tr = len(data["units"]["train"])
    rng = np.random.default_rng(seed)
    for s, (a, b), tg in (("train", (0, n_tr), tg_tr), ("dev", (n_tr, len(both)), tg_dev)):
        X = {v: Xb[a:b] for v in PF.VARIANTS}
        starts, tids = _ic_tokens(tg)
        Y, _ = P.ic_targets(starts, tids, tg.walkway_ic, data["units"][s][["trial_id", "t0", "t1"]])
        M = np.random.default_rng(9).standard_normal((Y.shape[1], 16))
        lat = Y @ M + 0.05 * rng.standard_normal((len(Y), 16))
        data[s] = {"X": X, "n_rows": np.full(b - a, 12), "sex": np.zeros(b - a, int),
                   "tokens": {"trial_id": tids, "person_id": np.array([f"weargait:{t.split('_')[0]}" for t in tids]),
                              "start": starts, "latent": lat.astype(np.float16), "mu_z": lat.astype(np.float32)},
                   "counts": {"records": 2, "windows": 2, "passes": b - a, "passes_with_rows": b - a,
                              "passes_without_rows": 0, "trials_without_record": [], "token_rows_kept": len(starts),
                              "rows_per_pass_median": 12.0}}
    return train, dev, tg_tr, tg_dev, data


def test_the_probe_stage_runs_on_synthetic_features_and_the_report_is_labelled(tmp_path, monkeypatch):
    train, dev, tg_tr, tg_dev, data = _synthetic_data()
    monkeypatch.setattr(g3, "controls", lambda fold: sorted(dev) if fold == "dev" else sorted(train))
    res0 = PF.run_probes(data, tg_tr, tg_dev, excluded=(dev[0], dev[1]), log=lambda *a, **k: None)
    assert res0["sets"]["dev12"] == sorted(set(dev) - {dev[0], dev[1]})
    lat = res0["passes"]["latent"]["lines"]
    pop = res0["population"]
    for s in ("dev14", "dev12"):
        for ln in ("stride_mae_cm", "speed_mae_m_s"):
            assert lat[s][ln]["point"] < 0.3 * pop[s][ln]["point"], (s, ln)
    assert lat["train_crossfit"]["stride_mae_cm"]["n_persons"] == len(train)
    icl = res0["ic"]["variants"]["latent"]["lines"]["dev14"]
    assert icl["matched_share"] > 0.95 and icl["point"] < 30.0
    assert res0["leakage"] is not None and set(res0["leakage"]["classes"]) == {"HC_NLS", "WHC_WPD"}
    # a full results dict through the report writer and the summary
    ck = {"path": "cache/runs/x/ckpt_B_000001.pt", "name": "x_ckpt_B_000001", "sha256": "0" * 64, "weights": "ema",
          "phase": "B", "step": 1}
    res = {"what": PF.LABEL, "checkpoint": ck, "inference": {"window_s": 10.0, "hop_s": 5.0, "device": "cpu"},
           "sets": {"train": train, "dev14": res0["sets"]["dev14"], "dev12": res0["sets"]["dev12"],
                    "train_among_checkpoint_persons": 0},
           "excluded_from_12": res0["excluded_from_12"], "passes": res0["passes"], "ic": res0["ic"],
           "population": res0["population"], "leakage": res0["leakage"],
           "counts": {"train": data["train"]["counts"], "dev": data["dev"]["counts"]},
           "constants": {"boot_n": 200, "seed": 1}, "lockbox": {"persons_in_lockbox": 0, "target_lockbox_rows": 0},
           "timing_s": {"tokens": 1.0, "probes": 1.0, "all": 2.0}}
    monkeypatch.setattr(PF, "TREE", str(tmp_path))
    res["comparison"] = PF.comparison(res, "x_ckpt_B_000001")
    assert res["comparison"]["decoded"]["diag"] is None
    res = PF.clean(res)
    text = PF.report_md(res)
    assert PF.LABEL in text and EM_DASH not in text and "| stride MAE (cm) | dev12" in text
    out = tmp_path / "reports" / "bench" / "frozen_tokens_x_ckpt_B_000001"
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(res), encoding="utf-8")
    md, js = PF.write_summary(["x_ckpt_B_000001"], out_stem=str(tmp_path / "summary"), tree=str(tmp_path),
                              log=lambda *a, **k: None)
    s = open(md, encoding="utf-8").read()
    assert "x_ckpt_B_000001 probe | x_ckpt_B_000001 decoded" in s and EM_DASH not in s
    assert json.load(open(js, encoding="utf-8"))["names"] == ["x_ckpt_B_000001"]


def test_paired_difference_and_reading():
    pr = PF.paired({"a": 1.0, "b": 2.0, "c": 3.0, "d": float("inf")}, {"a": 3.0, "b": 4.5, "c": 5.0, "e": 1.0},
                   n_boot=200, seed=0)
    assert pr["n_persons"] == 3 and pr["mean"] == pytest.approx(-2.1666666, abs=1e-6)
    assert PF.reading(pr) == "probe lower"
    assert PF.reading({"interval": [-1.0, 1.0]}) == "no difference at 95 %"
    assert PF.family("weargait:NLS901 (control)") == "HC_NLS" and PF.family("weargait:WHC026") == "WHC_WPD"


# ---------------------------------------------------------------- the script's plumbing
def test_the_feature_file_round_trips_and_refuses_another_checkpoint(tmp_path):
    _, _, _, _, data = _synthetic_data()
    meta = {"checkpoint": {"name": "x_ckpt_B_000001"}, "limit_persons": 0}
    path = PF.save_features(str(tmp_path / "f" / "features.npz"), data, meta)
    got, m = PF.load_features(path)
    assert m == meta
    PF.check_features_meta(m, "x_ckpt_B_000001")
    with pytest.raises(SystemExit):
        PF.check_features_meta(m, "x_ckpt_A_000001", path)
    for s in ("train", "dev"):
        a, b = data["units"][s], got["units"][s]
        assert list(a["trial_id"]) == list(b["trial_id"]) and list(a["pass_id"]) == list(b["pass_id"])
        assert np.allclose(a["stride_m"], b["stride_m"], atol=1e-9, equal_nan=True)
        for v in PF.VARIANTS:
            assert got[s]["X"][v].shape == data[s]["X"][v].shape
            assert np.allclose(got[s]["X"][v], data[s]["X"][v], rtol=1e-6, atol=1e-6, equal_nan=True)
        assert (got[s]["n_rows"] == data[s]["n_rows"]).all() and (got[s]["sex"] == data[s]["sex"]).all()
        assert got[s]["counts"] == data[s]["counts"]
        for k in ("trial_id", "person_id", "start", "latent", "mu_z"):
            assert np.array_equal(got[s]["tokens"][k], data[s]["tokens"][k])


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_the_decoded_body_j1_and_supervised_files_are_read_with_their_schemas(tmp_path, monkeypatch):
    name = "x_ckpt_B_000001"
    d14 = ["weargait:HC1", "weargait:HC2", "weargait:WHC003", "weargait:WHC026"]
    d12 = d14[:2]
    pp14 = {p: {"stride_cm": 20.0 + i, "speed_ms": 0.2 + 0.01 * i} for i, p in enumerate(d14)}

    def sub(pp):   # the keys of a diag.json subset (scripts/eval_g3_diag.py); speed in m/s under the ms names
        return {"stride_mae_cm": 21.0, "speed_mae_ms": 0.21, "stride_mae_cm_interval": [18.0, 24.0],
                "speed_mae_ms_interval": [0.18, 0.24], "stride_mape_pct": 15.0, "n_persons": len(pp),
                "per_person": pp}
    diag = {"checkpoint": {"sha256": "c" * 64, "weights": "ema"},
            "inference": {"refine_steps": 20, "points_per_stream": 256, "window_s": 10.0, "hop_s": 5.0, "seed": 1},
            "rows": {"base": {"all14": sub(pp14), "no_whc003_whc026": sub({p: pp14[p] for p in d12}),
                              "ic_walkway": {"median_ms": 14.0, "interval": [11.0, 17.0], "matched_share": 1.0,
                                             "offset_ms": 90.0, "subsets": {"no_whc003_whc026": 13.0}}}},
            "shank_gyro_detector": {"walkway": {"median_ms": None, "matched_share": 0.9}}}
    _write_json(tmp_path / "reports" / "gates" / "g3" / f"diag_{name}" / "diag.json", diag)
    j1 = tmp_path / "reports" / "gates" / "g3" / f"dev_{name}"
    _write_json(j1 / "g3_dev.json", {"checkpoint": {"sha256": "c" * 64}, "lines": {"dev": {"model": {
        "stride_mae_cm": {"point": 26.5}, "speed_mae_m_s": {"point": 0.27}, "ic_median_ms_walkway": {"point": 14.3}}}}})
    rows = [("dev", "model", "ic_median_ms_walkway", p, v) for p, v in zip(d14, (10.0, 12.0, "inf", 20.0))]
    rows += [("dev", "walkway", "ic_median_ms_walkway", p, 0.0) for p in d14]
    rows += [("all68", "model", "ic_median_ms_walkway", p, 99.0) for p in d14]
    pd.DataFrame(rows, columns=["row", "predictor", "line", "person_id", "value"]).to_csv(
        j1 / "per_person.csv", index=False, encoding="utf-8")
    sup = {"run_started": "t", "what": "w", "lines": {s: {"supervised": {
        "stride_mae_cm": {"point": 12.0 - k, "interval": [9.0, 15.0]},
        "speed_mae_m_s": {"point": 0.09, "interval": [0.06, 0.13]}}} for k, s in enumerate(PF.SETS)}}
    _write_json(tmp_path / PF.SUPERVISED, sup)
    monkeypatch.setattr(PF, "TREE", str(tmp_path))
    dec = PF.decoded_body(name)
    assert dec["dev14"]["speed_mae_m_s"] == {"point": 0.21, "interval": [0.18, 0.24]}
    assert dec["dev12"]["per_person"]["stride_mae_cm"] == {p: pp14[p]["stride_cm"] for p in d12}
    assert dec["dev12"]["ic_median_ms"]["point"] == 13.0 and dec["dev14"]["ic_median_ms"]["offset_ms"] == 90.0
    assert dec["diag"]["checkpoint_sha256"] == "c" * 64 and dec["diag"]["inference"]["refine_steps"] == 20
    assert dec["j1"]["dev14"] == {"stride_mae_cm": 26.5, "speed_mae_m_s": 0.27, "ic_median_ms": 14.3}
    ic_pp = dec["j1"]["ic_per_person"]
    assert len(ic_pp) == 4 and ic_pp["weargait:WHC003"] == float("inf") and ic_pp["weargait:HC2"] == 12.0
    assert PF.supervised()["dev12"]["stride_mae_cm"]["point"] == 11.0
    probe_pp = {p: pp14[p]["stride_cm"] - 10.0 - i for i, p in enumerate(d14)}
    res = {"sets": {"dev14": d14, "dev12": d12},
           "passes": {PF.PRIMARY: {"per_person": {s: {"stride_mae_cm": probe_pp,
                                                       "speed_mae_m_s": {p: 0.1 for p in d14}} for s in PF.SETS}}},
           "ic": {"variants": {PF.PRIMARY: {"per_person": {s: {p: 25.0 for p in d14} for s in PF.SETS}}}}}
    cmp_ = PF.comparison(res, name)
    pr = cmp_["paired_primary"]
    assert cmp_["supervised"]["dev14"]["stride_mae_cm"]["point"] == 12.0
    assert pr["dev12"]["stride_mae_cm"]["n_persons"] == 2
    assert pr["dev12"]["stride_mae_cm"]["mean"] == pytest.approx(-10.5)
    assert pr["dev14"]["stride_mae_cm"]["reading"] == "probe lower"
    assert pr["dev14"]["speed_mae_m_s"]["n_persons"] == 4
    assert pr["dev14"]["ic_median_ms"]["n_persons"] == 3            # the infinite J1 value is left out
    assert pr["dev12"]["ic_median_ms"]["mean"] == pytest.approx(25.0 - 11.0)
    assert pr["dev12"]["ic_median_ms"]["decoded_source"] == "J1 report per_person.csv"


def test_check_record_refuses_a_lockbox_or_eval_record_and_a_person_outside_the_set():
    PF.check_record("t", {"lockbox": False, "person_id": "a"}, ["a"])
    PF.check_record("t", {"lockbox": np.False_, "person_id": "a", "tier": "TRAIN"}, ["a"])
    for rec in ({"lockbox": True, "person_id": "a"}, {"lockbox": "True", "person_id": "a"}, {"person_id": "a"},
                {"lockbox": False, "person_id": "a", "tier": "EVAL"}, {"lockbox": False, "person_id": "b"}):
        with pytest.raises(AssertionError):
            PF.check_record("t", rec, ["a"])


def test_collect_pools_each_pass_from_its_trial_rows_and_keeps_rows_near_the_walkway(monkeypatch):
    from mdfm.eval import infer as I
    persons = ["weargait:HC001", "weargait:WHC002"]
    tg = _targets(persons, [(1.0, 1.8), (1.3, 2.0)])
    units = P.pass_units(tg)

    class Win:
        def __init__(self, t0, cov):
            self.t0, self.duration, self.cov = t0, 10.0, cov

    def windows(rec):              # 30 s records: 10 s windows every 5 s, as infer.record_windows cuts them
        return [Win(t0, {"sex": rec["sex"]}) for t0 in (0.0, 5.0, 10.0, 15.0, 20.0)], None, 100.0

    def encode(wins):              # each row carries its own patch centre (s) and a constant
        out = []
        for w in wins:
            c = w.t0 + (np.arange(40) + 0.5) * 0.25
            f = np.stack([c, np.ones_like(c)], axis=1).astype(np.float32)
            out.append({"latent": f, "measurement": f, "stem": f, "mu_z": f[:, :1],
                        "subject": np.full(2, w.t0, np.float32)})
        return out

    monkeypatch.setattr(I, "record_windows", windows)
    first = tg.trials["trial_id"].iloc[0]                         # left without its record
    recs = [(t, {"lockbox": False, "person_id": p, "sex": 1}) for t, p in
            zip(tg.trials["trial_id"], tg.trials["person_id"]) if t != first]
    out = PF.collect(encode, recs, units, tg, persons, log=lambda *a, **k: None)
    assert out["counts"]["trials_without_record"] == [first] and out["counts"]["records"] == len(recs)
    grid = (np.arange(120) + 0.5) * 0.25                          # every patch centre of 30 s, read once
    for i, u in units.iterrows():
        x = out["X"]["latent"][i]
        if u["trial_id"] == first:
            assert np.isnan(x).all() and out["n_rows"][i] == 0 and out["sex"][i] == -1
            continue
        inside = grid[(grid >= u["t0"] - 1e-9) & (grid <= u["t1"] + 1e-9)]
        assert out["n_rows"][i] == len(inside) and out["sex"][i] == 1
        assert np.allclose(x, [inside.mean(), 1.0, inside.std(), 0.0], atol=1e-4)
        assert np.allclose(out["X"]["mu_z"][i], [inside.mean(), inside.std()], atol=1e-4)
    tok = out["tokens"]
    assert first not in set(tok["trial_id"]) and tok["latent"].dtype == np.float16
    spans = {t: g[["t0", "t1"]].to_numpy() for t, g in units.groupby("trial_id")}
    for t, a in zip(tok["trial_id"], tok["start"]):
        c = a + 0.125
        near = ((c >= spans[t][:, 0] - PF.KEEP_PAD_S) & (c <= spans[t][:, 1] + PF.KEEP_PAD_S)).any()
        ics = tg.walkway_ic.loc[tg.walkway_ic["trial_id"] == t, "t_s"].to_numpy()
        assert near or (np.abs(ics - c) <= PF.KEEP_PAD_S).any()
    per_trial = pd.Series(tok["trial_id"]).value_counts()
    assert (per_trial < 120).all() and len(per_trial) == len(tg.trials) - 1
