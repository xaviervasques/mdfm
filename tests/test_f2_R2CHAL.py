"""Tests of scripts/f2_R2CHAL.py (CPU, no data beyond the task's own report and hash file)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod():
    spec = importlib.util.spec_from_file_location("_t_f2_R2CHAL", os.path.join(TREE, "scripts", "f2_R2CHAL.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_registered_prediction_hash_holds():
    assert M.check_prediction() == M.PREDICTION_SHA256


def test_own_section_includes_closing_blank_line():
    t = "# x\n\n## Prediction\n\nabc\n\n## Run\n\nz\n"
    assert M.own_section(t) == "## Prediction\n\nabc\n\n"
    assert M.prediction_section(t) == "## Prediction\n\nabc\n"


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.stack([np.ones(500), rng.normal(size=500), rng.normal(size=500)], 1)
    y = X @ np.array([0.3, 1.2, -0.4])
    assert np.allclose(M.ols(X, y), [0.3, 1.2, -0.4])


def test_relation_weights_persons_equally():
    # person a: 1 stride on the line 0 + 1 x; person b: 9 strides on 1 + 1 x; equal weights give alpha 0.5
    rows = [{"person_id": "a", "c": 110.0 * math.exp(0.1), "l": 1.0}]
    rows[0]["S"] = math.exp(0.0 + 0.1)
    for k in range(9):
        x = 0.05 * (k - 4)
        rows.append({"person_id": "b", "c": 110.0 * math.exp(x), "l": 1.0, "S": math.exp(1.0 + x)})
    rows.append({"person_id": "a", "c": 110.0 * math.exp(-0.1), "l": 1.0, "S": math.exp(-0.1)})
    a, b = M.relation(pd.DataFrame(rows))
    assert abs(b - 1.0) < 1e-9
    assert abs(a - 0.5) < 1e-9


def test_c2_points_reads_a_stride_reading_decoder():
    rng = np.random.default_rng(1)
    n = 300
    S = np.exp(rng.normal(0.1, 0.15, n))
    c = np.exp(rng.normal(math.log(110), 0.1, n))
    l = np.exp(rng.normal(math.log(0.9), 0.05, n))
    D = 0.85 * S
    p = pd.DataFrame({"S": S, "c": c, "l": l, "D": D})
    r = M.c2_points(p)
    assert abs(r["b_S"] - 1) < 1e-9 and abs(r["b_c"]) < 1e-9 and abs(r["b_l"]) < 1e-9


def test_c1_points_cadence_decoder_reads_h_one():
    # real strides 1.1 x the AMASS relation; decoded = g_sim x relation: h = 1
    a, b, g = 0.3, 0.4, 0.96
    rows = []
    for p in range(5):
        for k in range(10):
            c = 100.0 + 3 * k
            l = 0.85 + 0.02 * p
            A = l * math.exp(a + b * (math.log(c) - M.C_CENTRE))
            rows.append({"person_id": f"p{p}", "c": c, "l": l, "S": 1.1 * A, "D": g * A})
    r = M.c1_points(pd.DataFrame(rows), (a, b), g)
    assert abs(r["h"] - 1) < 1e-9


def test_tempo_fit_recovers_b_c_and_classes():
    rng = np.random.default_rng(2)
    rows = []
    for p in range(12):
        for k in range(60):
            S = math.exp(rng.normal(0.1, 0.15))
            c = math.exp(rng.normal(math.log(105), 0.12))
            l = 0.9
            rows.append({"person_id": f"p{p}", "S": S, "c": c, "l": l * math.exp(rng.normal(0, 0.03)),
                         "y": 0.9 * S * (c / 105) ** 0.3 * math.exp(rng.normal(0, 0.01))})
    d = pd.DataFrame(rows)
    f = M.tempo_fit(d, "y", "S", "l", n_boot=200)
    assert abs(f["b_c"] - 0.3) < 0.02 and abs(f["b_S"] - 1) < 0.02
    assert M.q3_class(f) == "SIM TEMPO"
    assert M.q3_class({"b_c": 0.0, "b_c_interval": [-0.1, 0.1]}) == "NO SIM TEMPO"
    assert M.q3_class({"b_c": 0.1, "b_c_interval": [-0.1, 0.25]}) == "UNRESOLVED"
    assert M.q3_class({"b_c": 0.2, "b_c_interval": [-0.2, 0.5]}) == "UNRESOLVED"


def test_guarded_csv_refuses_lockbox_and_drops_other_persons(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("person_id,v\nweargait:A,1\nweargait:B,2\n", encoding="utf-8")
    d = M.guarded_csv(str(p), ["weargait:A"], set())
    assert list(d["person_id"]) == ["weargait:A"]
    with pytest.raises(M.Refused):
        M.guarded_csv(str(p), ["weargait:A"], {"weargait:B"})
    q = tmp_path / "f.csv"
    q.write_text("person_id,v\ncody:X,1\n", encoding="utf-8")
    with pytest.raises(M.Refused):
        M.guarded_csv(str(q), [], set())


def test_compare_flags_differences():
    ok, rows = M.compare({"a": 1.0, "b": 2.0}, {"a": 1.0 + 1e-8, "b": 2.1})
    assert not ok and rows["a"]["equal"] and not rows["b"]["equal"]
