"""Tests of scripts/cpl_posthoc2.py on hand-made tables (no store, no record, no person)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "cpl_posthoc2.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t_cpl_posthoc2", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def _pop(rng, n_persons, prefix, b, kappa=1.0, n=25):
    rows = []
    for k in range(n_persons):
        stature = rng.uniform(1.5, 1.9)
        l = M.C.LEG_RATIO * stature
        for j in range(n):
            T = rng.uniform(0.9, 1.3)
            S = l * math.exp(rng.normal(0.35, 0.12))
            rows.append({"person_id": f"{prefix}{k}", "t0": 0.0, "t1": T, "S": S, "stature_m": stature,
                         "rom_wrist": math.exp(3.0 + b * math.log(S / l) + math.log(kappa) + rng.normal(0, 0.03))})
    return M.C.finish(pd.DataFrame(rows))


def test_site_row_recovers_kappa_and_gamma():
    rng = np.random.default_rng(5)
    am = _pop(rng, 30, "a", 0.6)
    real = _pop(rng, 25, "r", 0.6, kappa=0.93)
    row = M.site_row(real, am, "rom_wrist", 200, 1)
    assert row["kappa"] == pytest.approx(0.93, abs=0.015)
    assert row["gamma"] == pytest.approx(math.exp(math.log(0.93) / 0.6), abs=0.03)
    assert row["gamma_interval"][0] <= row["gamma"] <= row["gamma_interval"][1]
    assert M.reading(row) == "CARRIES SHORT"


def test_reading_not_scaling():
    row = {"beta": [3.0, 0.05, 0.0], "gamma": 0.5, "gamma_interval": [0.4, 0.6]}
    assert M.reading(row) == "NOT SCALING"
    row = {"beta": [3.0, 0.5, 0.0], "gamma": 1.0, "gamma_interval": [0.97, 1.03]}
    assert M.reading(row) == "MATCHED"


def test_sexp_saturates():
    assert M.sexp(0.0) == 1.0
    assert M.sexp(1e4) == float("inf")
