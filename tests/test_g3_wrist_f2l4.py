"""Tests of scripts/g3_wrist_f2l4.py (task WRB) on synthetic events, frames and gains: no target, record, event file or
checkpoint is read except the two prediction files' hashes and the fold lists."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "g3_wrist_f2l4.py")


@pytest.fixture(scope="module")
def B():
    for p in (os.path.join(TREE, "scripts"), TREE):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("g3_wrist_f2l4_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PERSONS = [f"weargait:P{i:02d}" for i in range(6)]


def frame(values, line="stride_mae_cm", name="x", persons=PERSONS, per=3, matched=True):
    """Row frame (comparison.ROW_COLUMNS): `per` rows per person, every value of person i equal to values[i]."""
    from mdfm.baselines import comparison as CMP
    rows = []
    for i, p in enumerate(persons):
        for k in range(per):
            v = float(values[i] if np.ndim(values) else values)
            rows.append({"line": line, "variant": name, "chosen": False, "person_id": p, "trial_id": f"{p}_T",
                         "task": "SelfPace", "pass_id": str(k), "foot": "L", "t0": float(k), "t1": k + 1.0,
                         "t_s": np.nan, "target": 1.2, "prediction": 1.2 + v / 100.0, "matched": matched,
                         "enters": matched, "value": v})
    return pd.DataFrame(rows)[CMP.ROW_COLUMNS]


def events_csv(path, persons=PERSONS, row="family_wrists", unit_stride="cm"):
    """A synthetic events.csv in eval_g3_diag's columns: per person two strides (one unmatched for person 0) and one
    pass, plus rows of another row name that must be ignored."""
    rows = []
    for i, p in enumerate(persons):
        for k in range(2):
            m = not (i == 0 and k == 1)
            ref, pred = 130.0 + i, 130.0 + i - (10.0 + i)
            rows.append([row, "stride", f"{p}_T", p, "SelfPace", "L", "0", 1.0 + k, 2.1 + k, ref, pred,
                         (pred - ref) if m else np.nan, unit_stride, m])
        rows.append([row, "speed", f"{p}_T", p, "SelfPace", "", "0", 1.0, 4.0, 1.3, 1.1 - 0.01 * i, -0.2 - 0.01 * i,
                     "m/s", True])
        rows.append(["base", "stride", f"{p}_T", p, "SelfPace", "L", "0", 1.0, 2.1, 130.0, 0.0, -130.0, "cm", True])
    cols = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference",
            "prediction", "signed_error", "unit", "matched"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8")
    return path


def test_prediction_hashes_hold(B):
    assert B.check_prediction() == B.PREDICTION_SHA256


def test_prediction_hash_refuses_an_edit(B, tmp_path):
    p = tmp_path / "prediction.txt"
    p.write_text("x\n", encoding="utf-8")
    s = tmp_path / "prediction.sha256"
    s.write_text(B.sha256_file(str(p)) + "  prediction.txt\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        B.check_prediction(str(p), str(s), B.PREDICTION_SHA256)
    p.write_text("y\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        B.check_prediction(str(p), str(s), B.sha256_file(str(s)))


def test_check_pinned_refuses_a_changed_input(B, tmp_path):
    (tmp_path / "a.csv").write_text("1\n", encoding="utf-8")
    good = {"a.csv": B.sha256_file(str(tmp_path / "a.csv"))}
    assert B.check_pinned(str(tmp_path), good) == good
    with pytest.raises(AssertionError):
        B.check_pinned(str(tmp_path), {"a.csv": "0" * 64})


def test_part_persons_is_a_partition(B):
    sets = {"dev14": [f"d{i}" for i in range(14)], "train54": [f"t{i}" for i in range(54)]}
    parts = [B.part_persons(sets, i, 4) for i in range(4)]
    every = [x for p in parts for x in p]
    assert len(every) == 68 and len(set(every)) == 68
    assert {x for x in every if x[0] == "dev14"} == {("dev14", p) for p in sets["dev14"]}
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
    with pytest.raises(ValueError):
        B.part_persons(sets, 4, 4)


def test_read_events_builds_g3_tables_and_the_line(B, tmp_path):
    W = B.WRS()
    path = events_csv(str(tmp_path / "events.csv"))
    st, sp = B.read_events(path, PERSONS)
    assert len(st) == 12 and len(sp) == 6 and int(st["matched"].sum()) == 11
    assert np.isnan(st.loc[~st["matched"], "err_cm"]).all()
    fr = B.table_frame(st, sp, "d_raw")
    want = np.mean([10.0 + i for i in range(6)])          # every matched stride of person i is off by 10 + i cm
    assert abs(W.set_stats(fr, PERSONS, "stride_mae_cm")["point"] - want) < 1e-12
    assert abs(W.set_stats(fr, PERSONS, "speed_mae_m_s")["point"] - np.mean([0.2 + 0.01 * i for i in range(6)])) < 1e-12
    assert abs(W.set_stats(fr, PERSONS[:3], "stride_mae_cm")["point"] - 11.0) < 1e-12


def test_read_events_refuses_other_persons_and_units(B, tmp_path):
    from mdfm.data.canonical import Refused
    path = events_csv(str(tmp_path / "events.csv"))
    with pytest.raises(Refused):
        B.read_events(path, PERSONS[:-1])
    bad = events_csv(str(tmp_path / "bad.csv"), unit_stride="mm")
    with pytest.raises(ValueError):
        B.read_events(bad, PERSONS)
    with pytest.raises(ValueError):
        B.read_events(path, PERSONS, row="no_such_row")


def test_scale_tables_divide_and_keep_the_matching(B, tmp_path):
    st, sp = B.read_events(events_csv(str(tmp_path / "events.csv")), PERSONS)
    s1, v1 = B.scale_tables(st, sp, 1.0, 1.0)
    assert np.array_equal(s1["err_cm"].to_numpy(), st["err_cm"].to_numpy(), equal_nan=True)
    g_s, g_v = 0.9, 0.85
    s2, v2 = B.scale_tables(st, sp, g_s, g_v)
    m = st["matched"].to_numpy()
    assert np.array_equal(s2["matched"].to_numpy(), m)
    assert np.allclose(s2["pred_m"], st["pred_m"] / g_s)
    assert np.allclose(s2.loc[m, "err_cm"], 100.0 * (st.loc[m, "pred_m"] / g_s - st.loc[m, "length_m"]))
    assert np.isnan(s2.loc[~m, "err_cm"]).all()
    assert np.allclose(v2["err_m_s"], sp["pred_m_s"] / g_v - sp["speed_m_s"])
    assert np.allclose(st["pred_m"], s1["pred_m"])             # the input is not modified


def test_stride_gain_is_the_median_ratio_of_matched_strides(B, tmp_path):
    st, sp = B.read_events(events_csv(str(tmp_path / "events.csv")), PERSONS)
    fr = B.table_frame(st, sp, "d_raw")
    m = st[st["matched"]]
    want = float(np.median(m["pred_m"] / m["length_m"]))
    g = B.stride_gain(fr, PERSONS)
    assert abs(g["gain"] - want) < 1e-12 and g["n_strides"] == 11


def test_scale_reading_cases(B):
    g_all = {"f2l4_B": 0.92, "v1_B": 0.85}                     # rise_a 0.07
    base = {"v1_B": 0.85, "s2_B": 0.88}                        # d_w 0.03, 2 d_w 0.06
    s = B.scale_reading({**base, "f2l4_B": 0.92}, g_all, "stride")
    assert s["reading"] == "TRANSFERS" and abs(s["tau"] - 1.0) < 1e-12
    assert abs(s["needs_g_wrist_above"] - 0.91) < 1e-12 and abs(s["drop_all_to_wrist"]) < 1e-12
    s = B.scale_reading({**base, "f2l4_B": 0.90}, g_all, "stride")
    assert s["reading"] == "NOT SHOWN" and not s["beyond_2d"] and s["at_least_half"]
    s = B.scale_reading({"v1_B": 0.85, "s2_B": 0.851, "f2l4_B": 0.87}, g_all, "stride")
    assert s["reading"] == "NOT SHOWN" and s["beyond_2d"] and not s["at_least_half"]
    s = B.scale_reading({**base, "f2l4_B": 0.80}, g_all, "stride")
    assert s["reading"] == "NOT SHOWN" and s["rise_w"] < 0


def test_premium_rule_with_one_body_seed(B):
    W = B.WRS()
    comp = [frame(10.0), frame(11.0)]
    t = W.premium_test([frame(5.0)], comp, PERSONS, "stride_mae_cm")
    assert t["separation"] == "body" and t["premium"] and not t["loss"]
    assert abs(t["seed_spread"] - 1.0) < 1e-12 and abs(t["margin"] - 2.0) < 1e-12
    t = W.premium_test([frame(9.0)], comp, PERSONS, "stride_mae_cm")
    assert t["separation"] == "body" and not t["premium"]       # -1.5 is not at most -m = -2
    t = W.premium_test([frame(20.0)], comp, PERSONS, "stride_mae_cm")
    assert t["loss"] and not t["premium"]
    t = W.premium_test([frame(10.5)], comp, PERSONS, "stride_mae_cm")
    assert t["separation"] is None and not t["premium"] and not t["loss"]


def test_comparisons_read_both_sets(B):
    W = B.WRS()
    bars = {n: [frame(30.0)] for n in ("a1", "a2", "b1", "b2")}
    bars.update({a: [frame(10.0), frame(11.0)] for a in W.TOKEN_ARMS})
    sp = {n: [f.assign(line="speed_mae_m_s", value=f["value"] / 100.0) for f in v] for n, v in bars.items()}
    allbars = {n: [pd.concat([a, b], ignore_index=True) for a, b in zip(bars[n], sp[n])] for n in bars}
    body = pd.concat([frame(5.0), frame(0.05, line="speed_mae_m_s")], ignore_index=True)
    sets = {"dev12": PERSONS[:4], "dev14": PERSONS}
    _, rd = W.comparisons({"d_gain": [body]}, allbars, sets)
    assert rd["d_gain"]["stride_mae_cm"] == "PREMIUM" and rd["d_gain"]["speed_mae_m_s"] == "PREMIUM"
    body2 = pd.concat([frame(20.0), frame(0.20, line="speed_mae_m_s")], ignore_index=True)
    _, rd = W.comparisons({"d_raw": [body2]}, allbars, sets)
    assert rd["d_raw"]["stride_mae_cm"] == "LOSS" and rd["d_raw"]["speed_mae_m_s"] == "LOSS"


def test_answer_leaves_out_device_dependent_lines(B):
    lines = ("stride_mae_cm", "speed_mae_m_s")
    res = {"scale": {"stride": {"reading": "NOT SHOWN"}},
           "readings": {"d_raw": {ln: "LOSS" for ln in lines}, "d_gain": {"stride_mae_cm": "PREMIUM", "speed_mae_m_s": "NO PREMIUM"}},
           "device_dependent": {"d_raw": {ln: False for ln in lines}, "d_gain": {"stride_mae_cm": True, "speed_mae_m_s": False}}}
    a = B.answer(res)
    assert not a["scale_transfers_to_wrists"] and not a["wrist_premium"]
    assert a["device_dependent_lines"] == ["d_gain stride_mae_cm"]
    res["device_dependent"]["d_gain"]["stride_mae_cm"] = False
    res["scale"]["stride"]["reading"] = "TRANSFERS"
    a = B.answer(res)
    assert a["scale_transfers_to_wrists"] and a["wrist_premium"] and a["premium_lines"] == ["d_gain stride_mae_cm"]


def test_wrs_bars_and_point_check(B):
    W = B.WRS()
    names = ["a1", "a2", "b1", "b2"] + [f"{a}_{k}" for a, ks in W.TOKEN_ARMS.items() for k in ks]
    frames = {n: pd.concat([frame(10.0 + i), frame(0.1, line="speed_mae_m_s")], ignore_index=True)
              for i, n in enumerate(names)}
    bars = B.wrs_bars(frames)
    assert set(bars) == {"a1", "a2", "b1", "b2"} | set(W.TOKEN_ARMS)
    assert all(len(bars[a]) == 2 for a in W.TOKEN_ARMS) and all(len(bars[n]) == 1 for n in ("a1", "a2", "b1", "b2"))
    sets = {"dev12": PERSONS[:4], "dev14": PERSONS}
    res = {"rows": {n: {s: {"stride_mae_cm": {"point": 10.0 + i}, "speed_mae_m_s": {"point": 0.1}} for s in sets}
                    for i, n in enumerate(names)}}
    assert B.check_wrs_points(frames, res, sets) < 1e-12
    res["rows"]["a1"]["dev12"]["stride_mae_cm"]["point"] = 10.5
    with pytest.raises(AssertionError):
        B.check_wrs_points(frames, res, sets)


def test_short_id_and_decode_hash(B):
    assert B.short_id("weargait:NLS901 (control)") == "NLS901" and B.short_id("weargait:HC103") == "HC103"
    h = B.decode_code_sha256()
    assert len(h) == 64 and h == B.decode_code_sha256()
    assert all(os.path.exists(f) for f in B.decode_code_files())
    hh = B.head_decode_code_sha256()                            # the committed versions (git show HEAD:path)
    assert len(hh) == 64 and hh == B.head_decode_code_sha256()


def test_smoke_needs_its_own_folders(B):
    with pytest.raises(SystemExit):
        B.main(["--decode", "--sets", "train54", "--limit-persons", "1"])
    with pytest.raises(SystemExit):
        B.main(["--assemble", "--sets", "nope", "--out", "x", "--cache", "y"])
