"""Tests of scripts/g4_grf_bars.py (task GRF): the masked multi-output ridge, the frame-to-patch mapping, the token
design and its inverse, the mirrored kinematic features, the context offsets, the premium rule and the prediction
check. Synthetic data only: no record, target or checkpoint is read."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)


def _load():
    key = "_test_g4_grf_bars"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", "g4_grf_bars.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load()


def test_masked_ridge_equals_probes_ridge_without_missing():
    from mdfm.eval import probes
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    X[:, 3] = 2.0                                           # a column without variance
    Y = X @ rng.normal(size=(6, 3)) + rng.normal(size=(80, 3))
    for a in (0.1, 10.0):
        m = G.masked_ridge_path(X, Y, (a,))[0]
        r = probes.fit_ridge(X, Y, a)
        Xq = rng.normal(size=(5, 6))
        assert np.allclose(m.predict(Xq), r.predict(Xq), atol=1e-9)


def test_masked_ridge_column_ignores_its_missing_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    Y = X @ rng.normal(size=(4, 2))
    Ym = Y.copy()
    Ym[:20, 1] = np.nan                                     # column 1 loses rows; column 0 keeps all
    full = G.masked_ridge_path(X, Y, (1.0,))[0]
    part = G.masked_ridge_path(X, Ym, (1.0,))[0]
    Xq = rng.normal(size=(7, 4))
    assert np.allclose(full.predict(Xq)[:, 0], part.predict(Xq)[:, 0])
    # column 1 fitted on rows 20.. only: near-exact recovery of a noiseless linear map with a small penalty
    small = G.masked_ridge_path(X, Ym, (1e-8,))[0]
    assert np.allclose(small.predict(Xq)[:, 1], (Xq @ np.linalg.lstsq(X, Y, rcond=None)[0])[:, 1], atol=1e-5)
    Xn = Xq.copy()
    Xn[0, 0] = np.nan
    assert np.isnan(part.predict(Xn)[0]).all()


def test_masked_select_alpha_and_cross_fit_keep_persons_apart():
    rng = np.random.default_rng(2)
    persons = np.repeat([f"p{i}" for i in range(10)], 12)
    X = rng.normal(size=(120, 5))
    Y = X @ rng.normal(size=(5, 4)) + 0.1 * rng.normal(size=(120, 4))
    Y[rng.random(Y.shape) < 0.3] = np.nan
    Y[np.isnan(Y).all(axis=1), 0] = 0.0
    a, cv = G.masked_select_alpha(X, Y, persons)
    assert a in cv and cv[a] == min(cv.values())
    oof, alphas = G.masked_cross_fit(X, Y, persons)
    assert oof.shape == Y.shape and len(alphas) == 5 and np.isfinite(oof).all()


def test_frame_patches_most_central_window_and_offsets():
    fps = 100.0
    t = 10.0 + np.arange(1500) / fps                        # 15 s on the recording clock
    starts, durs = [10.0, 15.0], [10.0, 10.0]
    npat = [40, 40]
    w, p, j = G.frame_patches(t, starts, durs, npat, fps)
    assert (w >= 0).all()
    assert w[0] == 0 and p[0] == 0 and j[0] == 0
    # at t = 17.5 both windows cover the frame; window 1 (middle 20.0) is nearer than window 0 (middle 15.0)? no:
    # |17.5 - 15| = 2.5 = |17.5 - 20|: the tie goes to the earlier window
    i = 750
    assert abs(t[i] - 17.5) < 1e-9 and w[i] == 0
    i = 760                                                 # 17.6: nearer window 1's middle
    assert w[i] == 1 and p[i] == int((t[i] - 15.0) // 0.25) and j[i] == 10
    rec = (np.asarray(starts)[w] + 0.25 * p + j / fps)
    assert np.allclose(rec, t)
    assert (j >= 0).all() and (j < 25).all()


def test_frame_patches_uncovered_frames():
    t = np.arange(50) / 100.0
    w, p, j = G.frame_patches(t, [], [], [], 100.0)
    assert (w == -1).all() and (p == -1).all() and (j == -1).all()


class _Trial:
    def __init__(self, person, key, t, obs, obs_valid):
        self.person, self.key, self.t, self.obs, self.obs_valid = person, key, t, obs, obs_valid


def _token_arrays(trials, d=3, n_off=25, rng=None):
    """A synthetic token file: one window per trial from its first frame, patches of 25 frames."""
    rng = rng or np.random.default_rng(3)
    arr = {"trial_person": [], "trial_key": [], "trial_start": [], "trial_len": [], "frame_t": [], "frame_row": [],
           "frame_off": [], "latent": []}
    n_rows = n_frames = 0
    for tr in trials:
        T = len(tr.t)
        P = int(np.ceil(T / n_off))
        arr["trial_person"].append(tr.person)
        arr["trial_key"].append(tr.key)
        arr["trial_start"].append(n_frames)
        arr["trial_len"].append(T)
        arr["frame_t"].append(tr.t)
        arr["frame_row"].append(n_rows + np.arange(T) // n_off)
        arr["frame_off"].append(np.arange(T) % n_off)
        arr["latent"].append(rng.normal(size=(P, d)))
        n_rows += P
        n_frames += T
    out = {k: np.concatenate(v) if k in ("frame_t", "frame_row", "frame_off", "latent") else np.asarray(v)
           for k, v in arr.items()}
    index = {(p, k): i for i, (p, k) in enumerate(zip(out["trial_person"], out["trial_key"]))}
    return out, index


def test_token_design_and_frames_round_trip():
    rng = np.random.default_rng(4)
    trials = []
    for i in range(3):
        T = 60 + 10 * i
        obs = rng.normal(size=(T, 2))
        valid = rng.random((T, 2)) < 0.6
        trials.append(_Trial(f"p{i}", "k", np.arange(T) / 100.0, obs, valid))
    arr, index = _token_arrays(trials)
    scored = {(tr.person, tr.key): tr.obs_valid for tr in trials}
    X, Y, ps, rows = G.token_design(arr, index, trials, scored, "latent", 25)
    assert X.shape[0] == Y.shape[0] == len(ps) == len(rows)
    assert np.isfinite(Y).sum() == sum(int(tr.obs_valid.sum()) for tr in trials)
    lookup = np.full(len(arr["latent"]), -1)
    lookup[rows] = np.arange(len(rows))

    def rp(r):
        o = np.full((len(r), Y.shape[1]), np.nan)
        o[lookup[r] >= 0] = Y[lookup[r][lookup[r] >= 0]]
        return o

    for tr in trials:
        back = G.token_frames(arr, index, tr, rp, 25)
        assert np.allclose(back[tr.obs_valid], tr.obs[tr.obs_valid])
        assert np.isnan(back[~tr.obs_valid]).all()


def _walk(T=300, fps=100.0, seed=5):
    """A synthetic walk: 15 joints in metres, y up, walking along a slanted horizontal direction."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    base = np.array([[0, 1.0, 0], [0, 1.5, 0], [0, 1.7, 0], [0, 1.45, -0.18], [0, 1.45, 0.18], [0, 1.2, -0.22],
                     [0, 1.2, 0.22], [0, 0.95, -0.24], [0, 0.95, 0.24], [0, 0.95, -0.09], [0, 0.95, 0.09],
                     [0, 0.5, -0.1], [0, 0.5, 0.1], [0, 0.08, -0.1], [0, 0.08, 0.1]], dtype=np.float64)
    phase = 2 * np.pi * 0.9 * t
    X = np.repeat(base[None], T, axis=0)
    X[:, :, 0] += 1.2 * t[:, None]
    for j, s in ((11, 1), (13, 1), (12, -1), (14, -1), (7, -1), (8, 1)):
        X[:, j, 0] += 0.25 * s * np.sin(phase)
        X[:, j, 1] += 0.04 * np.maximum(0, s * np.cos(phase))
    X[:, 0, 1] += 0.02 * np.cos(2 * phase)
    X += 0.001 * rng.normal(size=X.shape)
    th = 0.6
    Rz = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
    return t, X @ Rz.T


def _grf_trial(world, t):
    from mdfm.gates import g4
    T = len(t)
    return g4.GRFTrial(person="p", key="k", fps=100.0, t=t, world=world, obs=np.zeros((T, 2)),
                       obs_valid=np.ones((T, 2), bool), mass_kg=70.0, mass_from="c3d", sex="M")


def test_base_features_mirror_foot_one_is_the_mirrored_walk():
    t, W = _walk()
    F, ok, newton = G.base_features(_grf_trial(W, t))
    assert F.shape == (len(t), 2, G.N_BASE) and G.N_BASE == 100
    assert np.isfinite(F[ok]).all() and np.isnan(F[~ok]).all()
    # the mirror image of the walk about the vertical plane z = 0, left and right joints swapped
    Wm = W[:, list(G.MIRROR)] * np.array([1.0, 1.0, -1.0])
    Fm, okm, nm = G.base_features(_grf_trial(Wm, t))
    assert np.array_equal(ok, okm)
    assert np.allclose(Fm[ok, 0], F[ok, 1], atol=1e-8)
    assert np.allclose(Fm[ok, 1], F[ok, 0], atol=1e-8)
    assert np.allclose(nm[ok], newton[ok][:, ::-1], atol=1e-10)


def test_base_features_heading_invariant():
    t, W = _walk()
    th = 1.1
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    F, ok, _ = G.base_features(_grf_trial(W, t))
    Fr, okr, _ = G.base_features(_grf_trial(W @ Ry.T + np.array([3.0, 0.0, -2.0]), t))
    assert np.array_equal(ok, okr)
    assert np.allclose(F[ok], Fr[ok], atol=1e-8)


def test_with_context_clamps_inside_runs():
    T = 12
    F = np.arange(T, dtype=np.float64)[:, None, None] * np.ones((1, 2, 1))
    ok = np.ones(T, bool)
    ok[5] = False
    out = G.with_context(F, ok, 100.0, context_s=(-0.02, 0.0, 0.03))
    assert np.isnan(out[5]).all()
    assert list(out[0, 0]) == [0, 0, 3]                      # clamped at the run start
    assert list(out[4, 0]) == [2, 4, 4]                      # clamped at the run end (frame 4)
    assert list(out[6, 0]) == [6, 6, 9]                      # the second run starts at frame 6


def _table(body, comp, n=20, seed=6):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"person_id": [f"p{i}" for i in range(n)], "newton_r": np.full(n, 0.6),
                      "newton_rmse_pct_bw": np.full(n, 40.0)})
    for name, (rmse, r) in {**body, **comp}.items():
        t[f"{name}_rmse_pct_bw"] = rmse + rng.normal(scale=0.5, size=n)
        t[f"{name}_r"] = r + rng.normal(scale=0.005, size=n)
    return t


def test_premium_test_readings():
    t = _table({"b_1": (10.0, 0.9), "b_2": (10.5, 0.9)}, {"c_1": (20.0, 0.8), "c_2": (21.0, 0.8)})
    x = G.premium_test(t, ("b_1", "b_2"), ("c_1", "c_2"), "grf_rmse")
    assert x["reading"] == "PREMIUM" and x["difference"] < 0 and x["interval"][1] < 0
    assert abs(x["margin"] - max(0.1 * x["comparator_mean"], 2 * x["seed_spread"])) < 1e-12
    y = G.premium_test(t, ("b_1", "b_2"), ("c_1", "c_2"), "grf_r_gain")
    assert y["reading"] == "PREMIUM" and y["margin"] == pytest.approx(0.03)
    z = G.premium_test(t, ("c_1", "c_2"), ("b_1", "b_2"), "grf_rmse")
    assert z["reading"] == "LOSS"
    # separated and the interval below 0, but within the margin: NO PREMIUM
    t2 = _table({"b_1": (19.0, 0.81), "b_2": (19.1, 0.81)}, {"c_1": (20.0, 0.8)})
    w = G.premium_test(t2, ("b_1", "b_2"), ("c_1",), "grf_rmse")
    assert w["separation"] and w["interval_better"] and not w["margin_met"] and w["reading"] == "NO PREMIUM"


def test_best_bar_and_set_reading():
    lines = {"a": {"grf_rmse": {"point": 12.0}, "grf_r_gain": {"point": 0.1}},
             "b1": {"grf_rmse": {"point": 10.0}, "grf_r_gain": {"point": 0.3}},
             "b2": {"grf_rmse": {"point": 14.0}, "grf_r_gain": {"point": 0.0}}}
    bars = {"A": ("a",), "B": ("b1", "b2")}
    assert G.best_bar(lines, bars, "grf_rmse") == "A"          # A 12 against B's seed mean 12: tie to the first
    assert G.best_bar(lines, bars, "grf_r_gain") == "B"
    assert G.set_reading({"grf_rmse": {"reading": "LOSS"}, "grf_r_gain": {"reading": "LOSS"}}) == "LOSS"
    assert G.set_reading({"grf_rmse": {"reading": "LOSS"}, "grf_r_gain": {"reading": "PREMIUM"}}) == "NO PREMIUM"


def test_fill_scored_counts_and_fills():
    tr = _Trial("p", "k", np.arange(4) / 100.0, np.zeros((4, 2)), np.array([[1, 0], [1, 1], [0, 0], [1, 1]], bool))
    pred = {("p", "k"): np.array([[np.nan, np.nan], [1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])}
    out, n = G.fill_scored(pred, [tr], {("p", "k"): tr.obs_valid}, 0.5)
    assert n == 2
    assert out[("p", "k")][0, 0] == 0.5 and out[("p", "k")][1, 1] == 0.5 and np.isnan(out[("p", "k")][0, 1])


def test_check_prediction_refuses_a_changed_file(tmp_path):
    p = tmp_path / "prediction.txt"
    p.write_text("x\n", encoding="utf-8")
    sha = G.sha256_file(str(p))
    s = tmp_path / "prediction.sha256"
    s.write_text(sha + "  prediction.txt\n", encoding="utf-8")
    assert G.check_prediction(str(p), str(s), sha) == sha
    p.write_text("y\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        G.check_prediction(str(p), str(s), sha)
