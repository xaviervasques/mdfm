"""Tests of scripts/build_admissible_nulls.py (task T12): the Q10 admissible-null count against brute-force
enumeration, the Holm step bounds of NR-7.3 and NR-7.5, the draw count for Q1 and the summary rendering.
No MoveAtlas file is read.
"""
import importlib.util
import itertools
import os
from fractions import Fraction

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "build_admissible_nulls.py")


def _script():
    spec = importlib.util.spec_from_file_location("build_admissible_nulls_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()
GENES = {"TOR1A": 9, "SGCE": 5, "VPS16": 5, "THAP1": 4, "PANK2": 3, "KMT2B": 2, "GNAL": 2, "PLA2G6": 1, "AOPEP": 1}


def test_k_matches_enumeration():
    for n_a, n_b in ((9, 5), (5, 5), (4, 15), (3, 2)):
        assignments = sum(1 for _ in itertools.combinations(range(n_a + n_b), n_b))
        assert S.admissible_k(n_a, n_b) == assignments - 1
    assert S.admissible_k(9, 5) == 2001


def test_holm_steps():
    steps = S.holm_alpha_min(Fraction(1, 2002))
    assert len(steps) == S.M == 6
    assert steps[0] == Fraction(6, 2002) and steps[-1] == Fraction(1, 2002)
    assert all(a > b for a, b in zip(steps, steps[1:]))


def test_min_draws():
    assert S.min_draws(Fraction(5, 100)) == 119
    assert S.min_draws(Fraction(1, 100)) == 599
    b = S.min_draws(Fraction(5, 100))
    assert Fraction(1, b + 1) <= Fraction(5, 100) / 6 < Fraction(1, b)


def test_contrast_rows():
    rows = S.contrast_rows(GENES)
    q10 = rows[0]
    assert q10["role"].startswith("primary") and q10["k"] == 2001 and q10["n_a"] == 9 and q10["n_b"] == 5
    assert q10["pass_decision_possible_k_ge_20"] is True and q10["enumerated_k_le_1e6"] is True
    assert q10["holm_alpha_min_step_1"] == S.fmt(Fraction(6, 2002))
    assert q10["nr_7_5_ok_at_alpha_0p05"] is True
    by = {r["contrast"]: r for r in rows}
    assert by["SGCE against VPS16"]["k"] == 251
    iron = [r for r in rows if r["contrast"].startswith("brain iron")][0]
    assert iron["n_a"] == 4 and iron["n_b"] == 15 and iron["k"] == 3875
    assert iron["holm_alpha_min_step_1"] == "outside the Holm family"


def test_nr_7_5_fails_between_alpha_over_m_and_alpha():
    # k = C(7, 2) - 1 = 20, p_min = 1/21: above 0.05 / 6 and below 0.05, so the check must fail at 0.05.
    q10 = S.contrast_rows({"TOR1A": 5, "SGCE": 2})[0]
    assert q10["k"] == 20 and q10["pass_decision_possible_k_ge_20"] is True
    assert q10["nr_7_5_ok_at_alpha_0p05"] is False and q10["nr_7_5_ok_at_alpha_0p01"] is False
    assert q10["holm_alpha_min_step_1"] == S.fmt(Fraction(6, 21))
    assert q10["holm_alpha_min_step_6"] == S.fmt(Fraction(1, 21))
    # k = C(5, 2) - 1 = 9: p_min = 0.1 fails at every reference alpha, and k < 20 (NR-1.8).
    q10 = S.contrast_rows({"TOR1A": 3, "SGCE": 2})[0]
    assert q10["k"] == 9 and q10["pass_decision_possible_k_ge_20"] is False
    assert q10["nr_7_5_ok_at_alpha_0p05"] is False
    assert q10["holm_alpha_min_step_1"] == S.fmt(Fraction(6, 10))
    # k = C(6, 3) - 1 = 19: one below the NR-1.8 bound.
    assert S.contrast_rows({"TOR1A": 3, "SGCE": 3})[0]["pass_decision_possible_k_ge_20"] is False


def test_missing_gene_gives_no_k():
    rows = S.contrast_rows({"TOR1A": 9})
    assert rows[0]["k"] == 0 and rows[0]["genes_missing"] == "SGCE"


STAMP = {"root": "x", "head": "h" * 40, "diff_sha256": "d" * 64, "diff_bytes": 3, "time": "2099-01-01T00:00:00"}
Q1_STAMP = {"head": "q" * 40, "diff_sha256": "e" * 64}


def _q1_row(scope, enumerable, k="72220413630873599999999", hem="8"):
    return {"scope": scope, "files": "all", "open06": "with_open06", "n_patients": "4", "n_hemispheres": hem,
            "n_files": "48", "k": k, "log10_k": "22.859", "smallest_p": "1.38465e-23",
            "pass_decision_possible_k_ge_20": "True", "enumeration_feasible_k_le_1e6": enumerable}


def test_select_q1_rows_never_merges_other_scopes():
    rows = [_q1_row("q1", "False"), _q1_row("all_ramp", "False", hem="11")]
    q1, other = S.select_q1_rows(rows)
    assert [r["scope"] for r in q1] == ["q1"] and [r["scope"] for r in other] == ["all_ramp"]
    q1, other = S.select_q1_rows([_q1_row("all_ramp", "False", hem="11")])
    assert q1 == [] and len(other) == 1


def test_summary_with_and_without_q1():
    rows = S.contrast_rows(GENES)
    q1_all = [_q1_row("q1", "False"), _q1_row("all_ramp", "False", k="4242", hem="11")]
    q1, other = S.select_q1_rows(q1_all)
    with_q1 = S.summary_md(rows, GENES, q1, True, STAMP, other, Q1_STAMP)
    without = S.summary_md(rows, GENES, [], False, STAMP)
    no_q1_scope = S.summary_md(rows, GENES, [], True, STAMP, other, Q1_STAMP)
    assert "merged from oxford_q1.csv" in with_q1 and "B >= 119" in with_q1
    assert "Every merged row has k above the enumeration limit" in with_q1
    assert "Rows with drawn nulls: 1 of 1." in with_q1
    assert "stay in oxford_q1.csv: all_ramp (1 rows, 11 hemispheres)" in with_q1
    assert "4242" not in with_q1                        # the all_ramp row is not merged
    assert f"MoveAtlas HEAD {'q' * 40}, uncommitted diff SHA-256 {'e' * 64}" in with_q1
    assert "oxford_q1.csv is absent" in without
    assert "has no row of scope q1 (scopes found: all_ramp); Q1 is not merged." in no_q1_scope
    assert "merged from oxford_q1.csv" not in no_q1_scope and "B >= 119" not in no_q1_scope
    for text in (with_q1, without, no_q1_scope):
        assert "k = C(9 + 5, 5) - 1 = 2001 non-identity assignments, enumerated" in text
        assert "| gene labels permuted across patients; enumerated | 2001 |" in text
        assert f"MoveAtlas measured against: HEAD {'h' * 40}, uncommitted diff SHA-256 {'d' * 64}" in text
        assert "2099-01-01T00:00:00" not in text
        assert "gene or its surgical correlates" in text
        assert "\u2014" not in text and "\u2013" not in text


def test_summary_q1_enumerated_or_mixed_is_not_called_drawn():
    rows = S.contrast_rows(GENES)
    enum = S.summary_md(rows, GENES, [_q1_row("q1", "True", k="719")], True, STAMP)
    assert "Every merged row has k within the enumeration limit" in enum
    assert "Every merged row has k above" not in enum and "B >= 119" not in enum
    assert "within hemisphere; enumerated | 719 (by scope) | 1 / (k + 1) by scope |" in enum
    mixed = S.summary_md(rows, GENES, [_q1_row("q1", "True", k="719"), _q1_row("q1", "False")], True, STAMP)
    assert "1 of 2 merged rows have k above the enumeration limit" in mixed and "B >= 119" in mixed
    assert "drawn or enumerated by scope" in mixed
