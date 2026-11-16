"""Tests of scripts/f2_H5data.py on hand-made data (no store, no decode, no person record)."""
import hashlib
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "f2_H5data.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t_f2_H5data", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def _population(rng, n_persons, prefix, a, b, n=40, noise=0.02):
    rows = []
    for k in range(n_persons):
        leg = rng.uniform(0.8, 0.95)
        for j in range(n):
            c = rng.uniform(95, 130)
            s = leg * math.exp(a + b * (math.log(c) - M.C_CENTRE) + rng.normal(0, noise))
            rows.append({"person_id": f"{prefix}{k}", "trial_id": f"{prefix}{k}_t{j // 10}", "pass_id": str(j // 5),
                         "c": c, "S": s, "l": leg, "l_prior": 0.89})
    return pd.DataFrame(rows)


def test_prediction_hash_is_checked(tmp_path):
    rep = tmp_path / "r.md"
    rep.write_text("# t\n\n## Prediction\n\ntext\n\n## Run\n\nlater\n", encoding="utf-8")
    sec = M.prediction_section(rep.read_text(encoding="utf-8"))
    assert sec == "## Prediction\n\ntext\n"
    sha = hashlib.sha256(sec.encode("utf-8")).hexdigest()
    (tmp_path / "p.sha256").write_text(sha + "\nhashed now\n", encoding="utf-8")
    assert M.check_prediction(str(rep), str(tmp_path / "p.sha256"), sha) == sha
    with pytest.raises(AssertionError):
        M.check_prediction(str(rep), str(tmp_path / "p.sha256"), "0" * 64)


def test_registered_prediction_hash_holds():
    assert M.check_prediction() == M.PREDICTION_SHA256


def test_fit_relation_recovers_the_line():
    rng = np.random.default_rng(0)
    d = _population(rng, 30, "a", 0.3, 1.1, noise=0.001)
    coef, _ = M.fit_relation(d)
    assert abs(coef[0] - 0.3) < 0.01 and abs(coef[1] - 1.1) < 0.05


def test_cadence_decoder_reads_supported_and_reading_decoder_not():
    rng = np.random.default_rng(1)
    amass = _population(rng, 25, "amass:p", 0.30, 1.1)
    real = _population(rng, 20, "weargait:HC", 0.42, 0.9)        # longer strides at a given cadence
    g = 0.96
    a_pred = M.predict(M.fit_relation(amass)[0], np.log(real["c"].to_numpy()), real["l"].to_numpy())
    cad = real.assign(D=g * a_pred * np.exp(rng.normal(0, 0.01, len(real))))
    r = M.analyse(cad, amass, g, n_boot=60, seed=1)
    assert abs(r["C1"]["h"] - 1) < 0.1
    assert abs(r["C2"]["b_S"]) < 0.3 and r["C2"]["class"] in ("FROM CADENCE", "UNRESOLVED")
    assert r["C3"]["R"] < 1
    read = real.assign(D=0.85 * real["S"] * np.exp(rng.normal(0, 0.01, len(real))))
    r2 = M.analyse(read, amass, g, n_boot=60, seed=1)
    assert r2["C2"]["b_S"] > 0.8 and r2["C2"]["class"] == "READ"
    assert M.reading(r2["C2"]["class"], r2["C1"]["h"], r2["C1"]["interval"])[0] == "NOT SUPPORTED"
    assert r2["C3"]["R"] > 1


def test_reading_clauses():
    assert M.reading("FROM CADENCE", 0.8, [0.4, 1.2])[0] == "SUPPORTED"
    assert M.reading("FROM CADENCE", 0.8, [0.2, 1.2])[0] == "PARTLY"
    assert M.reading("READ", 0.9, [0.6, 1.2])[0] == "NOT SUPPORTED"
    assert M.reading("BETWEEN", 0.1, [-0.1, 0.2])[0] == "NOT SUPPORTED"
    assert M.reading("FROM CADENCE", 0.1, [-0.1, 0.2])[0] == "PARTLY"
    assert M.reading("BETWEEN", 0.4, [0.1, 0.7])[0] == "PARTLY"


def test_c2_class_bounds():
    assert M.c2_class(0.75, [0.6, 0.9]) == "READ"
    assert M.c2_class(0.1, [-0.1, 0.4]) == "FROM CADENCE"
    assert M.c2_class(0.4, [0.1, 0.65]) == "BETWEEN"
    assert M.c2_class(0.4, [-0.3, 0.9]) == "UNRESOLVED"


def test_read_strides_refuses_a_lockbox_person_and_drops_others(tmp_path):
    cols = "row,kind,trial_id,person_id,task,foot,pass_id,t0_s,t1_s,reference,prediction,signed_error,unit,matched\n"
    ok = "base,stride,T1,weargait:A,SelfPace,L,0,1.0,2.1,120.0,100.0,-20.0,cm,True\n"
    other = "base,stride,T2,weargait:B,SelfPace,L,0,1.0,2.1,130.0,110.0,-20.0,cm,True\n"
    p = tmp_path / "e.csv"
    p.write_text(cols + ok + other, encoding="utf-8")
    d = M.read_strides(str(p), {"weargait:A"}, set())
    assert list(d["person_id"]) == ["weargait:A"] and abs(d["S"].iloc[0] - 1.2) < 1e-12
    from mdfm.data.canonical import Refused
    with pytest.raises(Refused):
        M.read_strides(str(p), {"weargait:A"}, {"weargait:B"})


def test_clean_strides_bounds():
    d = pd.DataFrame({"t0": [0.0, 0.0, 0.0], "t1": [1.0, 0.5, 1.1], "S": [1.2, 1.2, 0.1], "D": [1.0, 1.0, 1.0]})
    k, cnt = M.clean_strides(d)
    assert cnt == {"read": 3, "kept": 1, "dropped": 2} and abs(k["c"].iloc[0] - 120.0) < 1e-12
