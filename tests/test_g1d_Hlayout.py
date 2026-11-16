"""Tests of scripts/g1d_Hlayout.py: the pure functions (exposure probabilities, kNN distance, motion features,
gap attribution, partial rank correlation, the dash guard). No store, checkpoint or GPU is read."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))

import g1d_Hlayout as H  # noqa: E402


def test_random_contains_brute_force():
    import itertools
    n, want = 6, {0, 1}
    tot = 0.0
    for k in range(1, n + 1):
        subs = list(itertools.combinations(range(n), k))
        tot += sum(want <= set(s) for s in subs) / len(subs)
    assert H.p_random_contains(want, n, 1, n) == pytest.approx(tot / n)


def test_random_exact():
    assert H.p_random_exact(14, 1, 14, 6) == pytest.approx(1 / 14 / math.comb(14, 6))
    assert H.p_random_exact(14, 7, 14, 6) == 0.0


def test_knn_mean_and_exclusion():
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    d = H.knn_mean(X, X, 1)
    assert np.allclose(d, 0.0)
    groups = np.array([0, 0, 1, 1])
    d = H.knn_mean(X, X, 1, exclude=lambda a, b: groups[a:b, None] == groups[None])
    assert np.allclose(d, [3.0, 2.0, 2.0, 9.0])


def test_feature_block_still_body():
    n, K = 2, 300
    r6 = torch.zeros(n, K, 22, 6)
    r6[..., 0] = 1.0
    r6[..., 4] = 1.0                    # identity rotations (columns e1, e2)
    tr = np.zeros((n, K, 3))
    tr[1, :, 2] = np.arange(K) / 100.0  # 1 m/s forward
    f = H.feature_block(r6.numpy(), tr, np.array([K, 250]), np.array([[0.0, 0.9, 0.0], [0.0, 0.9, 0.0]]))
    names = H.feature_names()
    assert f.shape == (2, len(names))
    assert np.allclose(f[:, :126], 0.0, atol=1e-5)
    assert f[0, names.index("pelvis_h_mean")] == pytest.approx(0.9)
    assert f[0, names.index("hspeed_mean")] == pytest.approx(0.0)
    assert f[1, names.index("hspeed_mean")] == pytest.approx(1.0)
    assert f[0, names.index("tilt_mean")] == pytest.approx(0.0, abs=1e-4)


def test_gap_explained_recovers_coefficient():
    rng = np.random.default_rng(0)
    rows = []
    for arc, d0 in (("EyesJapanDataset", 1.0), ("HDM05", 1.5)):
        for i in range(200):
            d = d0 + rng.normal(0, 0.1)
            m = rng.uniform(0.02, 0.1)
            rows.append({"archive": arc, "person_id": f"{arc}{i % 4}", "n": 1000, "d": d, "motion_m": m,
                         "g1a_m": (40.0 + 20.0 * d + 100.0 * m) / 1000.0})
    out = H.gap_explained(pd.DataFrame(rows))
    assert out["coef_d_mm"] == pytest.approx(20.0, rel=1e-6)
    assert out["HDM05"]["share"] == pytest.approx(
        out["HDM05"]["predicted_by_d_mm"] / out["HDM05"]["observed_gap_mm"])


def test_partial_spearman_removes_confounder():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    df = pd.DataFrame({"archive": "a", "z": z, "x": z + 0.01 * rng.normal(size=500),
                       "y": z + 0.01 * rng.normal(size=500)})
    assert H.spearman(df["x"], df["y"]) > 0.9
    assert abs(H.partial_spearman(df, "x", "y", "z")) < 0.3


def test_write_text_refuses_dashes(tmp_path):
    with pytest.raises(ValueError):
        H.write_text(str(tmp_path / "a.md"), "a " + chr(0x2014) + " b")
    H.write_text(str(tmp_path / "b.md"), "a, b\n")
    assert (tmp_path / "b.md").read_text(encoding="utf-8") == "a, b\n"
