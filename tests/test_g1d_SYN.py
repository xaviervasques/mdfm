"""Tests of scripts/g1d_SYN.py on synthetic take rows (no file read)."""
import importlib.util
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("g1d_SYN", os.path.join(ROOT, "scripts", "g1d_SYN.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _t(p, w, s, k=1000):
    return {"person_id": p, "windows": w, "knots": k, "s": s, "g1a": 80.0 + s, "tb": 80.0}


def test_eligible_persons_needs_both_classes():
    takes = [_t("a", 1, 0), _t("a", 6, 0), _t("b", 2, 0), _t("c", 7, 0), _t("c", 3, 0)]
    assert S.eligible_persons(takes) == ["a"]


def test_few_minus_many_is_knot_weighted_and_person_averaged():
    takes = [_t("a", 1, 10.0, 100), _t("a", 2, 4.0, 300), _t("a", 5, 1.0), _t("b", 1, 3.0), _t("b", 8, 3.0)]
    # a: few 0.25*10 + 0.75*4 = 5.5, many 1.0 -> 4.5; b: 0 -> mean 2.25
    assert math.isclose(S.few_minus_many(takes, ["a", "b"]), 2.25)


def test_within_slope_recovers_a_planted_slope_despite_person_offsets():
    takes = []
    for p, off in (("a", 0.0), ("b", 50.0)):
        for w in (1, 2, 4, 8):
            takes.append(_t(p, w, off - 3.0 * math.log2(w)))
    assert math.isclose(S.within_slope(takes, ["a", "b"]), -3.0, abs_tol=1e-9)


def test_bootstrap_interval_contains_a_constant_effect():
    takes = [_t(p, w, 5.0 if w <= 2 else 1.0) for p in "abcd" for w in (1, 6)]
    lo, hi = S.person_bootstrap(takes, list("abcd"), lambda tk, ps: S.few_minus_many(tk, ps), b=50, seed=1)
    assert math.isclose(lo, 4.0) and math.isclose(hi, 4.0)


def test_reading_rule():
    assert S.reading(3.0, [0.5, 6.0], [-2.0, -0.1]) == "HELPS"
    assert S.reading(3.0, [0.5, 6.0], [-2.0, 0.3]) == "UNCLEAR"
    assert S.reading(-0.1, [-2.0, 1.0], [-1.0, 1.0]) == "NO_POOLING_GAIN"
    assert S.reading(1.0, [-0.5, 2.0], [-1.0, 1.0]) == "NO_POOLING_GAIN"
    assert S.reading(1.5, [-0.5, 4.0], [-1.0, 1.0]) == "UNCLEAR"


def test_best_case_distances():
    r = S.best_case(81.8, 7.2)
    assert math.isclose(r["best_case_mm"], 74.6) and math.isclose(r["above_kill_mm"], 4.6, abs_tol=1e-9)
    assert math.isclose(r["above_pass_mm"], 29.6, abs_tol=1e-9)


def test_ni_margin_takes_the_larger_term():
    assert math.isclose(S.ni_margin(100.0, 99.0), 5.0)
    assert math.isclose(S.ni_margin(10.0, 7.0), 6.0)
