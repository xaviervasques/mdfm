"""Tests of scripts/g3_zupt_pd_coverage.py (task ZPD, post hoc coverage) on synthetic run files (no weargait record,
no file of the tree except the last test, which checks the tree's own post hoc outputs against the run's): the
classes of an unmatched stride (no_foot, gap, merged, split, shifted) and the lag of the event times; the recomputed
end differences equal those of g3.match_strides, and a changed value or flag fails; the person table lists a person
with no matched stride first, with an undefined error, the merged length ratio and the body's errors (raw and with
the run's F1 gain); the row summary's pooled shares,
floor counts and class shares; the row sets (dev12 is dev14 within the valid controls, pd_t19dev the PD trials of the
T19 dev persons); the run checks pass on equal rows and fail on a changed share, person count or trial count; a
lockbox person in any input and a lag other than the registered one stop compute; the report carries the label, no
em or en dash, every person of dev12 and pd_all and the lowest ten of the other rows.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
import g3_zupt_pd as P  # noqa: E402
import g3_zupt_pd_coverage as C  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.gates import g3  # noqa: E402

LAG = P.FIT_LAG_S
LOCKED = ["weargait:L01"]


# ---------------------------------------------------------------- synthetic run files
def _walk(trial, person, foot, starts, dur=1.0, length=1.2, pass_id="0", group="pd"):
    return pd.DataFrame({"group": group, "trial_id": trial, "person_id": person, "task": "SelfPace",
                         "pass_id": pass_id, "foot": foot, "t0": [float(s) for s in starts],
                         "t1": [float(s) + dur for s in starts], "length_m": length})


def _zupt(trial, person, foot, ivals, speed=1.2, group="pd"):
    """ZUPT strides with zupt_seg event times ivals (so t0_ff = a0 + LAG), length speed times duration."""
    a0 = np.array([a for a, _ in ivals], np.float64)
    a1 = np.array([b for _, b in ivals], np.float64)
    return pd.DataFrame({"person_id": person, "trial_id": trial, "foot": foot, "t0_ff": a0 + LAG, "t1_ff": a1 + LAG,
                         "length_m": speed * (a1 - a0), "duration_s": a1 - a0, "group": group})


def _fill(walk, zupt, body=True):
    """walk with the zupt_ff and zupt_seg columns of g3.match_strides (the harness's own matching)."""
    walk = walk.reset_index(drop=True)
    tg = SimpleNamespace(strides=walk)
    out = walk.copy()
    for name, lag in (("zupt_ff", 0.0), ("zupt_seg", LAG)):
        pred = SimpleNamespace(strides=pd.DataFrame({"trial_id": zupt["trial_id"], "foot": zupt["foot"],
                                                     "t0": zupt["t0_ff"] - lag, "t1": zupt["t1_ff"] - lag,
                                                     "length_m": zupt["length_m"]}))
        m = g3.match_strides(pred, tg)
        out[f"{name}_matched"] = m["matched"].to_numpy(bool)
        out[f"{name}_m"] = np.where(m["matched"], m["pred_m"], np.nan)
        out[f"{name}_end_diff_s"] = m["end_diff_s"].to_numpy(np.float64)
    if body:
        out["body_cm"] = 100.0 * out["length_m"]
        out["body_matched"] = True
    return out


def _classes_case():
    """One trial, foot L: matched, merged, gap, split, shifted walkway strides; foot R without any ZUPT stride."""
    z = _zupt("A", "weargait:P00", "L", [(0, 1), (1, 2), (2, 4.1), (8, 8.4), (8.4, 9), (10.3, 11.3)])
    w = pd.concat([_walk("A", "weargait:P00", "L", [0, 2, 5, 8, 10]), _walk("A", "weargait:P00", "R", [0.5])],
                  ignore_index=True)
    return z, _fill(w, z)


def test_classes_of_unmatched_strides_and_the_lag():
    z, w = _classes_case()
    assert list(w["zupt_seg_matched"]) == [True, False, False, False, False, False]
    cl = C.classify(w, C.seg_strides(z, LAG))
    assert list(cl["cls"]) == ["matched", "merged", "gap", "split", "shifted", "no_foot"]
    assert cl.loc[1, "span_s"] == pytest.approx(2.1) and cl.loc[4, "start_offset_s"] == pytest.approx(0.3)
    assert np.isinf(cl.loc[5, "end_diff_s"]) and cl.loc[5, "zupt_index"] == -1
    # without the lag the same strides sit 0.14 s late: the recomputed end differences change
    cl0 = C.classify(w, C.seg_strides(z, 0.0))
    assert cl0.loc[0, "end_diff_s"] == pytest.approx(LAG)


def test_end_differences_equal_the_harness_and_a_change_fails():
    z, w = _classes_case()
    cl = C.classify(w, C.seg_strides(z, LAG))
    assert C.check_end_diffs(w, cl)
    bad = w.copy()
    bad.loc[1, "zupt_seg_end_diff_s"] += 0.01
    with pytest.raises(AssertionError):
        C.check_end_diffs(bad, cl)
    flip = w.copy()
    flip.loc[0, "zupt_seg_matched"] = False
    with pytest.raises(AssertionError):
        C.check_end_diffs(flip, C.classify(flip, C.seg_strides(z, LAG)))


def _pd_case():
    """Three PD persons: P00 all matched, P01 none (one merged stride and gaps), P02 half matched (shifted)."""
    zs, ws = [], []
    zs.append(_zupt("P00_A", "weargait:P00", "L", [(i, i + 1) for i in range(6)]))
    ws.append(_walk("P00_A", "weargait:P00", "L", range(1, 5)))
    zs.append(_zupt("P01_A", "weargait:P01", "L", [(0, 1), (1, 3.05), (6.5, 7.5)], speed=1.25))
    ws.append(_walk("P01_A", "weargait:P01", "L", [1, 2, 4]))
    zs.append(_zupt("P02_A", "weargait:P02", "L", [(0, 1), (1, 2), (2.3, 3.3), (3.3, 4.3)]))
    ws.append(_walk("P02_A", "weargait:P02", "L", [0, 1, 2, 3]))
    z = pd.concat(zs, ignore_index=True)
    w = _fill(pd.concat(ws, ignore_index=True), z)
    ps = pd.DataFrame({"group": "pd", "trial_id": ["P00_A", "P01_A", "P02_A"],
                       "person_id": ["weargait:P00", "weargait:P01", "weargait:P02"], "task": "SelfPace",
                       "pass_id": "0", "t0": 0.0, "t1": 5.0, "speed_m_s": [1.2, 1.25, 1.2],
                       "zupt_seg_matched": [True, False, True], "zupt_seg_m_s": [1.21, np.nan, 1.17]})
    return z, w, ps


def test_person_table_lists_no_match_first_with_an_undefined_error():
    z, w, ps = _pd_case()
    cl = C.classify(w, C.seg_strides(z, LAG))
    speed = {(r.trial_id, str(r.pass_id)): r.speed_m_s for r in ps.itertuples()}
    t = C.person_table(w, cl, ps, speed, "pd_all", g_s=0.8)
    assert [r["person_id"] for r in t] == ["weargait:P01", "weargait:P02", "weargait:P00"]
    p01, p02, p00 = t
    assert p01["body_mae_cm"] == pytest.approx(0.0) and p01["body_gain_mae_cm"] == pytest.approx(30.0)  # 120 / 0.8
    assert np.isnan(C.person_table(w, cl, ps, speed, "pd_all")[0]["body_gain_mae_cm"])            # no gain given
    assert p01["zupt_seg_share"] == 0 and np.isnan(p01["zupt_seg_mae_cm"]) and p01["unmatched"] == 3
    assert p01["n_merged"] == 2 and p01["n_gap"] == 1 and p01["body_share"] == 1.0
    assert p01["merged_len_ratio_median"] == pytest.approx(1.0)          # length = speed times duration
    assert p01["zupt_seg_pass_share"] == 0.0 and np.isnan(p01["zupt_seg_speed_mae_m_s"])
    assert p02["zupt_seg_share"] == 0.5 and p02["n_shifted"] == 2 and p02["zupt_seg_mae_cm"] == pytest.approx(0.0)
    assert p00["zupt_seg_share"] == 1.0 and p00["unmatched"] == 0
    assert p02["zupt_seg_speed_mae_m_s"] == pytest.approx(0.03)


def test_row_summary_counts():
    z, w, ps = _pd_case()
    cl = C.classify(w, C.seg_strides(z, LAG))
    speed = {(r.trial_id, str(r.pass_id)): r.speed_m_s for r in ps.itertuples()}
    t = C.person_table(w, cl, ps, speed, "pd_all")
    s = C.row_summary(t, w, cl, speed)
    assert s["persons"] == 3 and s["strides"] == 11 and s["pooled_share"] == pytest.approx(6 / 11)
    assert s["pooled_share_without_no_match"] == pytest.approx(6 / 8)
    assert s["persons_below_floor"] == 2 and s["persons_without_match"] == ["weargait:P01"]
    assert s["persons_with_a_match"] == 2 and s["unmatched"] == 5 and s["unmatched_held_below_floor"] == 1.0
    assert s["unmatched_by_class"] == {"no_foot": 0, "gap": 1, "merged": 2, "split": 0, "shifted": 2}
    assert s["gap_or_merged_share"] == pytest.approx(3 / 5) and s["merged_zupt_strides"] == 1
    assert s["share_min"] == 0.0 and s["share_p50"] == 0.5


def test_row_sets():
    st = pd.concat([_walk("D0_A", "weargait:D0", "L", [0], group="dev14"),
                    _walk("D1_A", "weargait:D1", "L", [0], group="dev14"),
                    _walk("P0_A", "weargait:P0", "L", [0]), _walk("P1_A", "weargait:P1", "L", [0])],
                   ignore_index=True)
    tr = _walk("T0_A", "weargait:T0", "L", [0], group="train54")
    s = C.row_sets(st, tr, {"D0_A", "X_A"}, ["weargait:P1"])
    assert s == {"train54": {"T0_A"}, "dev12": {"D0_A"}, "dev14": {"D0_A", "D1_A"}, "pd_all": {"P0_A", "P1_A"},
                 "pd_t19dev": {"P1_A"}}


def _run():
    """A loaded run (load_run's dict) of synthetic files, with results.json and train_fit.json made from the walks."""
    tz = _zupt("T0_A", "weargait:T0", "L", [(i, i + 1) for i in range(4)], group="train54")
    tw = _fill(_walk("T0_A", "weargait:T0", "L", [0, 1, 2], group="train54"), tz, body=False)
    dz = pd.concat([_zupt("D0_A", "weargait:D0", "L", [(i, i + 1) for i in range(4)], group="dev14"),
                    _zupt("D1_A", "weargait:D1", "L", [(0, 1), (1, 2)], group="dev14")], ignore_index=True)
    dw = pd.concat([_walk("D0_A", "weargait:D0", "L", [0, 1, 2], group="dev14"),
                    _walk("D1_A", "weargait:D1", "L", [0, 3], group="dev14"),
                    _walk("D1_A", "weargait:D1", "R", [0.5], group="dev14")], ignore_index=True)
    pz, pw, pps = _pd_case()
    zupt = pd.concat([tz, dz, pz], ignore_index=True)
    st = pd.concat([_fill(dw, dz), pw], ignore_index=True)
    dps = pd.DataFrame({"group": "dev14", "trial_id": ["D0_A", "D1_A"], "person_id": ["weargait:D0", "weargait:D1"],
                        "task": "SelfPace", "pass_id": "0", "t0": 0.0, "t1": 4.0, "speed_m_s": 1.2,
                        "zupt_seg_matched": True, "zupt_seg_m_s": 1.2})
    ps = pd.concat([dps, pps], ignore_index=True)
    sets = C.row_sets(st, tw, {"D0_A"}, ["weargait:P02"])

    def line(df):
        m = df["zupt_seg_matched"].astype(bool)
        return {"matched_share": float(m.mean()), "n_persons": int(df.loc[m, "person_id"].nunique()),
                "n_events": int(len(df))}

    lines = {r: {P.SEG: {"stride_mae_cm": line(st[st["trial_id"].isin(sets[r])])}} for r in C.ROWS if r != "train54"}
    results = {"fit": {"lag": {"lag_s": LAG}}, "lines": lines, "f1_gains": {"g_S": 0.8, "g_V": 0.8},
               "sets": {r: {"n_trials": len(sets[r])} for r in C.ROWS if r != "train54"}}
    train_fit = {"train54_lines": {P.SEG: {"stride_mae_cm": line(tw)}}}
    return {"strides": st, "passes": ps, "zupt": zupt, "train": tw, "results": results, "train_fit": train_fit}


def test_compute_on_a_synthetic_run_and_the_run_checks():
    rd = _run()
    out = C.compute(rd, {"D0_A"}, ["weargait:P02"], LOCKED)
    s = out["rows"]
    assert s["train54"]["pooled_share"] == 1.0 and s["dev12"]["pooled_share"] == 1.0
    assert s["dev14"]["unmatched_by_class"]["no_foot"] == 1 and s["dev14"]["unmatched_by_class"]["gap"] == 1
    assert s["pd_all"]["persons_without_match"] == ["weargait:P01"] and s["pd_t19dev"]["persons"] == 1
    assert out["persons"]["pd_all"][0]["person_id"] == "weargait:P01"
    assert out["persons"]["pd_all"][0]["body_gain_mae_cm"] == pytest.approx(30.0)      # the run's g_S is used
    assert np.isnan(out["persons"]["train54"][0]["body_share"])                          # no body on train54
    assert set(out["unmatched"]["row"]) == {"dev14", "pd_all", "pd_t19dev"}
    sets = C.row_sets(rd["strides"], rd["train"], {"D0_A"}, ["weargait:P02"])
    for key, val in (("matched_share", 0.5), ("n_persons", 3), ("n_events", 99)):
        bad = json.loads(json.dumps(rd["results"]))
        bad["lines"]["pd_all"][P.SEG]["stride_mae_cm"][key] = val
        with pytest.raises(AssertionError):
            C.check_against_run(s, bad, rd["train_fit"], sets)
    bad = json.loads(json.dumps(rd["results"]))
    bad["sets"]["dev12"]["n_trials"] = 2
    with pytest.raises(AssertionError):
        C.check_against_run(s, bad, rd["train_fit"], sets)


def test_compute_refuses_a_lockbox_person_and_another_lag():
    rd = _run()
    for key in ("strides", "zupt", "train", "passes"):
        r2 = dict(rd)
        df = rd[key].copy()
        df.loc[df.index[0], "person_id"] = LOCKED[0]
        r2[key] = df
        with pytest.raises(Refused):
            C.compute(r2, {"D0_A"}, ["weargait:P02"], LOCKED)
    r3 = dict(rd)
    r3["results"] = json.loads(json.dumps(rd["results"]))
    r3["results"]["fit"]["lag"]["lag_s"] = LAG + 0.01
    with pytest.raises(AssertionError):
        C.compute(r3, {"D0_A"}, ["weargait:P02"], LOCKED)


def test_the_report_carries_the_label_every_person_of_the_full_rows_and_no_dash():
    rd = _run()
    out = C.compute(rd, {"D0_A"}, ["weargait:P02"], LOCKED)
    many = [dict(out["persons"]["dev14"][0], person_id=f"weargait:X{i:02d}") for i in range(15)]
    res = {"floor": g3.MATCHED_SHARE_FLOOR, "rows": out["rows"],
           "persons": dict(out["persons"], dev14=many, pd_all=many),
           "run": {"prediction_sha256": P.PRED_SHA256, "run_started": "t", "moveatlas_head": "h", "lag_s": LAG,
                   "f1_g_S": 0.8},
           "timing_s": 0.0}
    text = C.summary_md(C.Z.clean(res))
    assert C.LABEL in text and "\u2014" not in text and "\u2013" not in text
    sec = {h: text.split(f"## {h}:")[1].split("\n## ")[0] for h in ("dev14", "pd_all")}
    assert sum(f"weargait:X{i:02d} |" in sec["pd_all"] for i in range(15)) == 15        # every person
    assert sum(f"weargait:X{i:02d} |" in sec["dev14"] for i in range(15)) == C.LOWEST_N  # the lowest ten


def test_the_tree_outputs_describe_the_run():
    path = os.path.join(C.OUT, "results.json")
    if not os.path.exists(path):
        pytest.skip("no post hoc output in the tree")
    with open(path, encoding="utf-8") as f:
        ph = json.load(f)
    with open(os.path.join(C.RUN, "results.json"), encoding="utf-8") as f:
        run = json.load(f)
    assert ph["run"]["prediction_sha256"] == P.PRED_SHA256 == run["prediction"]["sha256"]
    for row in ("dev12", "dev14", "pd_all", "pd_t19dev"):
        line = run["lines"][row][P.SEG]["stride_mae_cm"]
        assert abs(ph["rows"][row]["pooled_share"] - line["matched_share"]) < 1e-9
        assert ph["rows"][row]["persons_with_a_match"] == line["n_persons"]
        assert [r["zupt_seg_share"] for r in ph["persons"][row]] == sorted(r["zupt_seg_share"]
                                                                           for r in ph["persons"][row])
