"""Tests of scripts/f2_CHAL.py on synthetic inputs only (no real stride, speed or person value)."""
import hashlib
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)


def _load():
    spec = importlib.util.spec_from_file_location("_f2chal_test", os.path.join(TREE, "scripts", "f2_CHAL.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load()


def _units(fn, persons=30, per=40, lo=0.6, hi=1.9, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(persons):
        x = rng.uniform(lo, hi, per)
        for v in x:
            rows.append({"person_id": f"weargait:HC{i:03d}", "x": v, "y": fn(v) * (1 + rng.normal(0, 0.01))})
    return pd.DataFrame(rows)


def test_share_h_values():
    assert C.share_h(0.97, 0.85, 0.97) == pytest.approx(1.0)
    assert C.share_h(0.85, 0.85, 0.97) == pytest.approx(0.0)
    assert math.isnan(C.share_h(0.9, 0.85, 0.85))


def test_band_test_constant_gain_reads_minor():
    u = _units(lambda v: 0.85 * v)                       # a level shift: no range effect
    t = C.band_test(u, (0.8, 1.2), 0.97, n_boot=300)
    assert t["g_in"] == pytest.approx(0.85, abs=0.01)
    assert C.band_reading(t) == "MINOR"


def test_band_test_compressive_decoder_reads_main():
    # decoded = 0.97 at 1.0 and slope 0.5: the gain falls at fast speeds only
    u = _units(lambda v: 0.97 + 0.5 * (v - 1.0), lo=0.8, hi=2.4)
    t = C.band_test(u, (0.8, 1.2), 0.97, n_boot=300)
    assert t["g_in"] > t["g_all"]
    assert C.band_reading(t) == "MAIN"


def test_band_reading_not_evaluated_and_mixed():
    t = {"persons_in_band": 5, "h": 0.9, "h_interval": [0.5, 1.0]}
    assert C.band_reading(t) == "NOT EVALUATED"
    t = {"persons_in_band": 30, "h": 0.4, "h_interval": [0.1, 0.8]}
    assert C.band_reading(t) == "MIXED"
    t = {"persons_in_band": 30, "h": 0.6, "h_interval": [-0.1, 0.9]}
    assert C.band_reading(t) == "MIXED"


def test_range_reading():
    assert C.range_reading({1: "MAIN", 2: "MAIN"}) == "RANGE MAIN"
    assert C.range_reading({1: "MINOR", 2: "MINOR"}) == "RANGE MINOR"
    assert C.range_reading({1: "MINOR", 2: "MAIN"}) == "MIXED"
    assert C.range_reading({1: "NOT EVALUATED", 2: "MAIN"}) == "NOT EVALUATED"


def test_lopo_close_to_in_sample_for_homogeneous_persons():
    u = _units(lambda v: 0.85 * v)
    g = C.pooled_gain(u["x"], u["y"])
    s_in = np.mean(list(C.person_mae(u, g).values()))
    lo = C.lopo(u)
    assert lo["line"] >= s_in - 1e-12
    assert lo["line"] - s_in < 0.05 * s_in


def test_lopo_flags_a_heterogeneous_fold():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):
        g = 0.7 + 0.03 * i
        for v in rng.uniform(1, 1.5, 20):
            rows.append({"person_id": f"p{i}", "x": v, "y": g * v})
    u = pd.DataFrame(rows)
    g = C.pooled_gain(u["x"], u["y"])
    s_in = np.mean(list(C.person_mae(u, g).values()))
    assert C.lopo(u)["line"] > s_in


def test_x_reading():
    ok = {"a": {"S_in": 6.0, "S_lopo_minus_in": 0.2, "V_in": 0.055, "V_lopo_minus_in": 0.001}}
    assert C.x_reading(ok) == "TRANSFERS"
    bad = {"a": {"S_in": 6.0, "S_lopo_minus_in": 0.5, "V_in": 0.055, "V_lopo_minus_in": 0.001}}
    assert C.x_reading(bad) == "DOES NOT TRANSFER"


def test_f_share():
    assert C.f_share(0.942, 0.884) == pytest.approx(1 - math.log(0.942) / math.log(0.884))
    assert C.f_share(0.884, 0.884) == pytest.approx(0.0)


def test_person_weighted_gain():
    u = pd.DataFrame({"person_id": ["a"] * 3 + ["b"], "x": [1.0, 1.0, 1.0, 1.0], "y": [0.8, 0.8, 0.8, 0.9]})
    assert C.pooled_gain(u["x"], u["y"]) == pytest.approx(0.8)
    assert C.person_weighted_gain(u) == pytest.approx(0.85)


def test_scan_numbers_and_sections():
    found, missing = C.scan_numbers("gain 0.847 and 19.51 cm, f -0.279, line 1 and 12.3", [0.84713, 19.514, -0.2791])
    assert found == ["0.847", "19.51", "-0.279"] and missing == []
    found, missing = C.scan_numbers("share 33.4 percent and 0.555", [0.334])
    assert missing == ["0.555"]
    s = C.sections("# T\nx\n## A\n1.23\n### H1, shape\n2.34\n## Challenge 2026\n9.99\n")
    assert "H1, shape" in s and "2.34" in s["H1, shape"] and s["Challenge 2026"].strip() == "9.99"


def test_check_prediction(tmp_path):
    text = "## Prediction\n\nsomething fixed\n"
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (tmp_path / "prediction.txt").write_text(text, encoding="utf-8", newline="\n")
    (tmp_path / "prediction.sha256").write_text(h + "  x\n", encoding="utf-8")
    rep = tmp_path / "r.md"
    rep.write_text("# R\n\n" + text + "\n## Results\n", encoding="utf-8", newline="\n")
    assert C.check_prediction(str(rep), str(tmp_path), h) == h
    rep.write_text("# R\n\n## Prediction\n\nsomething changed\n", encoding="utf-8", newline="\n")
    with pytest.raises(AssertionError):
        C.check_prediction(str(rep), str(tmp_path), h)
    with pytest.raises(AssertionError):
        C.check_prediction(str(rep), str(tmp_path), "0" * 64)


def _events(tmp_path, persons):
    rows = []
    for p in persons:
        for k, (u, ref) in {"stride": ("cm", 120.0), "speed": ("m/s", 1.1)}.items():
            rows.append({"row": "base", "kind": k, "trial_id": f"{p}_t", "person_id": p, "task": "SelfPace",
                         "foot": "L", "pass_id": "1", "t0_s": 0, "t1_s": 1, "reference": ref,
                         "prediction": ref * 0.85, "signed_error": 0, "unit": u, "matched": "True"})
    path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_read_all_rows_guards_lockbox_and_outsiders(tmp_path):
    from mdfm.data.canonical import Refused
    path = _events(tmp_path, ["weargait:HC001", "weargait:HC002"])
    ev = C.read_all_rows(path, ["weargait:HC001", "weargait:HC002"], ["weargait:HC999"], ["weargait:HC001"])
    assert set(ev["person_id"]) == {"weargait:HC001"}
    st, sp = C.matched_units(ev)
    assert len(st) == 1 and len(sp) == 1 and st["y"].iloc[0] == pytest.approx(102.0)
    with pytest.raises(Refused):
        C.read_all_rows(path, None, ["weargait:HC002"], ["weargait:HC001"])
    with pytest.raises(AssertionError):
        C.read_all_rows(path, ["weargait:HC001"], [], ["weargait:HC001"])


def test_discipline_scan(tmp_path):
    d = tmp_path / "H9"
    d.mkdir()
    pd.DataFrame({"person_id": ["weargait:HC001", "HC002", "other_s1"]}).to_csv(d / "a.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(d / "b.csv", index=False)
    sets = {"dev14": ["weargait:HC001"], "train54": ["weargait:HC003"]}
    rows = {r["file"].split("/")[-1]: r for r in C.discipline_scan(sets, ["weargait:HC002"], str(tmp_path))}
    assert rows["a.csv"]["persons"] == 2 and rows["a.csv"]["other_ids"] == 1
    assert rows["a.csv"]["outside"] == 1 and rows["a.csv"]["lockbox"] == 1
    assert rows["b.csv"]["person_column"] is None


def test_summary_has_no_dash_or_forbidden_words():
    with open(os.path.join(TREE, "scripts", "f2_CHAL.py"), encoding="utf-8") as f:
        src = f.read()
    assert "\u2014" not in src and "\u2013" not in src          # no em-dash, no en-dash
