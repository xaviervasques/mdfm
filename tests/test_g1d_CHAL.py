"""Tests of the pure functions of scripts/g1d_CHAL.py on synthetic inputs (no model, no data)."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import g1d_CHAL as C  # noqa: E402


def test_prediction_section_and_hash():
    text = "# T\n\n## Prediction\n\nline a\nline b\n\n\n## Files\n\n- x\n"
    sec = C.prediction_section(text, "## Files")
    assert sec == "## Prediction\n\nline a\nline b\n"
    assert hashlib.sha256(sec.encode()).hexdigest() == hashlib.sha256(b"## Prediction\n\nline a\nline b\n").hexdigest()
    assert C.prediction_section("no section", "## Files") is None
    assert C.prediction_section("## Prediction\nx\n", "## Files") is None


def test_parse_hash_line():
    h, t = C.parse_hash_line("abc  some file section hashed 2026-10-06T05:01:12\n")
    assert h == "abc" and t == "2026-10-06T05:01:12"


def test_pooled_means_precision_weighted():
    mu = np.array([[0.0], [2.0], [10.0]])
    lv = np.log(np.array([[1.0], [1.0], [4.0]]))
    out = C.pooled_means(mu, lv, ["a", "a", "b"])
    assert np.allclose(out[:, 0], [1.0, 1.0, 10.0])
    out2 = C.pooled_means(mu, lv, ["a", "a", "a"])
    w = np.array([1.0, 1.0, 0.25])
    assert np.allclose(out2[:, 0], np.sum(w * mu[:, 0]) / w.sum())


def test_skeleton_error_is_pelvis_relative():
    rest = np.zeros((2, 3, 3))
    rest[:, 1, 0] = 0.1
    shifted = rest + 5.0                       # a global shift changes nothing
    assert np.allclose(C.skeleton_error_mm(shifted, rest), 0.0)
    longer = rest.copy()
    longer[:, 1, 0] = 0.13                     # one joint 30 mm off, mean over 3 joints
    assert np.allclose(C.skeleton_error_mm(longer, rest), 10.0)


def test_take_bias_parts_bias_and_noise():
    rng = np.random.default_rng(0)
    bias = np.zeros((22, 3))
    bias[5, 1] = 0.02
    d_bias = np.repeat(bias[None], 4, 0)                # identical offsets: pure bias
    b2, tot = C.take_bias_parts(d_bias)
    assert np.isclose(b2, tot) and tot > 0
    noise = rng.normal(0, 0.01, size=(4000, 22, 3))     # zero-mean scatter: bias near 0
    b2n, totn = C.take_bias_parts(noise)
    assert b2n / totn < 0.01
    b2s, _ = C.take_bias_parts(d_bias[:1])
    assert np.isnan(b2s)


def test_corrected_share_and_reading():
    parts = {"a": (1.0, 1.0), "b": (0.0, 1.0), "c": (float("nan"), 5.0)}
    assert np.isclose(C.corrected_bias_share(parts, ["a", "b", "c"]), 0.5)
    assert C.reading_bias(0.71) == "NO_ROOM"
    assert C.reading_bias(0.50) == "ROOM"
    assert C.reading_bias(0.6) == "BETWEEN"
    assert C.reading_bias(float("nan")) == "NOT_EVALUATED"


def test_bias_share_window_level():
    D = np.zeros((3, 2, 3))
    D[:, 0, 0] = 1.0
    assert np.isclose(C.bias_share(D), 1.0)
    D2 = np.zeros((2, 2, 3))
    D2[0, 0, 0], D2[1, 0, 0] = 1.0, -1.0
    assert np.isclose(C.bias_share(D2), 0.0)


def test_select_subset_whole_takes():
    w = pd.DataFrame({"person_id": ["p"] * 7 + ["q"] * 2,
                      "take_id": ["t1", "t1", "t2", "t3", "t3", "t4", "t5", "u1", "u2"]})
    sub = C.select_subset(w, per_person=2, seed=1)
    chosen = w.iloc[sub]
    for t in chosen["take_id"].unique():                 # every chosen take is whole
        assert (w["take_id"] == t).sum() == (chosen["take_id"] == t).sum()
    assert chosen.groupby("person_id")["take_id"].nunique().to_dict() == {"p": 2, "q": 2}
    assert sub == C.select_subset(w, per_person=2, seed=1)


def test_dup_tail():
    assert C._dup_tail("a\nb\nc\nd\ne\nf\nb\nc\nd\n")
    assert not C._dup_tail("a\nb\nc\nd\ne\nf\n")
