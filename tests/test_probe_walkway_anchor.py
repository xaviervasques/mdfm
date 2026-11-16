"""Tests of scripts/probe_walkway_anchor.py (task L4): the footfall tables, the anchor segment of a window, the strides
inside a window, the analytic anchored share against a brute-force count and at its edges, the span weighting of the
expected share, the scope summary, the cell-size refit, the replay check against a log, the margins and the gain scan,
the person guards, the column whitelist and the label-free names. Synthetic passes and fixtures only: no store, no
record, no target file is read.

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe -m pytest -q \
        -p no:cacheprovider tests/test_probe_walkway_anchor.py
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)


def _mod():
    key = "_t_probe_walkway_anchor"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", "probe_walkway_anchor.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


PA = _mod()
CELL_CM = 1.25


def _row(t, foot, heel, pid="0", partial="False", event="IC", trial="X_SelfPace", **extra):
    r = {"trial_id": trial, "person_id": "weargait:X", "task": "SelfPace", "lockbox": "False", "foot": foot,
         "event": event, "walkway_t_s": f"{t:.2f}", "pass_id": pid, "heel_x_cells": "" if heel is None else str(heel),
         "partial": partial}
    r.update(extra)
    return r


def _walk(t0=1.0, n=8, step_t=0.5, step_cells=60, first_foot="R", pid="0", partial_ends=True):
    """One straight pass: n footfalls alternating feet, a step every step_t s and step_cells cells; the first and the
    last footfall partial when partial_ends."""
    rows = []
    feet = ("R", "L") if first_foot == "R" else ("L", "R")
    for i in range(n):
        part = "True" if partial_ends and i in (0, n - 1) else "False"
        rows.append(_row(t0 + i * step_t, feet[i % 2], 10 + i * step_cells, pid=pid, partial=part))
    return rows


def _pass(t, foot, full=None, heel=None, pid="p"):
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    return PA.Pass(pid, t, np.asarray(foot), np.arange(n, dtype=np.float64) * 0.6 if heel is None else np.asarray(heel),
                   np.ones(n, bool) if full is None else np.asarray(full, bool))


# ---------------------------------------------------------------- footfalls and segments
def test_project_keeps_the_whitelist_only():
    r = _row(1.0, "L", 5, group="PD", phase="Walk", stride_cm="100")
    p = PA.project(r)
    assert tuple(p) == PA.ANCHOR_COLUMNS
    assert "group" not in p and "phase" not in p and "stride_cm" not in p


def test_pass_footfalls_builds_sorted_passes():
    rows = _walk() + [_row(2.2, "L", 200, event="FO"), _row(9.0, "R", 30, pid="")]
    rows = [PA.project(r) for r in rows[::-1]]
    P = PA.pass_footfalls(rows, CELL_CM)
    assert list(P) == ["X_SelfPace"] and len(P["X_SelfPace"]) == 1
    p = P["X_SelfPace"][0]
    assert np.all(np.diff(p.t) > 0) and len(p.t) == 8
    assert p.full.tolist() == [False] + [True] * 6 + [False]
    assert p.heel_m[1] == pytest.approx(70 * CELL_CM / 100.0)


def test_pass_footfalls_missing_heel_is_not_full():
    rows = [PA.project(r) for r in (_row(1.0, "L", 10), _row(1.5, "R", None), _row(2.0, "L", 130))]
    p = PA.pass_footfalls(rows, CELL_CM)["X_SelfPace"][0]
    assert p.full.tolist() == [True, False, True]
    assert math.isnan(p.heel_m[1])


def test_segment_same_foot_and_any_foot():
    p = PA.pass_footfalls([PA.project(r) for r in _walk()], CELL_CM)["X_SelfPace"][0]
    # full footfalls at 1.5 (L), 2.0 (R), 2.5 (L), 3.0 (R), 3.5 (L), 4.0 (R)
    s = PA.segment_of(p, 0.0, 10.0)
    assert s.foot == "L" and s.t_a == pytest.approx(1.5) and s.t_b == pytest.approx(3.5)
    assert s.n_strides == 2 and s.n_full == 6
    assert s.dist_m == pytest.approx(4 * 60 * CELL_CM / 100.0)
    assert s.speed_m_s == pytest.approx(4 * 60 * CELL_CM / 100.0 / 2.0)
    assert s.any_t_a == pytest.approx(1.5) and s.any_t_b == pytest.approx(4.0)
    assert s.any_dist_m == pytest.approx(5 * 60 * CELL_CM / 100.0)


def test_segment_window_edges_and_minimum():
    p = PA.pass_footfalls([PA.project(r) for r in _walk()], CELL_CM)["X_SelfPace"][0]
    # [2.0, 3.0] holds R 2.0, L 2.5, R 3.0: one R stride, edges inclusive; any-foot form has 3 footfalls
    s = PA.segment_of(p, 2.0, 3.0)
    assert s is not None and s.foot == "R" and s.n_strides == 1 and s.n_full == 3
    # [2.0, 2.9] holds R 2.0, L 2.5 only: no stride, no segment
    assert PA.segment_of(p, 2.0, 2.9) is None
    # ties on duration and strides go to L
    q = PA.Pass("1", np.array([0.0, 0.5, 1.0, 1.5]), np.array(["L", "R", "L", "R"]), np.array([0.0, 0.6, 1.2, 1.8]),
                np.ones(4, bool))
    assert PA.segment_of(q, 0.0, 2.0).foot == "L"


def test_segment_tie_on_duration_goes_to_more_strides():
    # L full at 0.0 and 2.0 (the L at 1.0 partial): 2.0 s, 1 stride; R full at 0.5, 1.5, 2.5: 2.0 s, 2 strides
    q = _pass([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], ["L", "R", "L", "R", "L", "R"], full=[1, 1, 0, 1, 1, 1])
    s = PA.segment_of(q, 0.0, 3.0)
    assert s.foot == "R" and s.n_strides == 2 and s.t_a == pytest.approx(0.5) and s.t_b == pytest.approx(2.5)


def test_any_foot_form_needs_three_full_footfalls():
    # L 1.0 and L 2.0 full, the R between them partial: a same-foot segment, but only 2 full footfalls
    q = _pass([1.0, 1.5, 2.0], ["L", "R", "L"], full=[1, 0, 1])
    s = PA.segment_of(q, 0.0, 5.0)
    assert s is not None and s.n_full == 2 and s.n_strides == 1
    assert math.isnan(s.any_dist_m) and math.isnan(s.any_t_a)


def test_partial_and_missing_heel_never_anchor():
    p = PA.Pass("0", np.array([1.0, 1.5, 2.0]), np.array(["L", "R", "L"]), np.array([0.0, np.nan, 1.2]),
                np.array([False, False, True]))
    assert PA.segment_of(p, 0.0, 5.0) is None


def test_strides_inside_counts_g3_strides():
    p = PA.pass_footfalls([PA.project(r) for r in _walk()], CELL_CM)["X_SelfPace"][0]
    # G3 strides: i >= 2 with both ends full: (1.5, 2.5), (2.0, 3.0), (2.5, 3.5), (3.0, 4.0)
    assert PA.strides_inside([p], 0.0, 10.0) == 4
    assert PA.strides_inside([p], 2.0, 1.0) == 1
    assert PA.strides_inside([p], 2.1, 1.0) == 0


def test_strides_inside_needs_the_other_foot_between():
    q = _pass([1.0, 1.5, 2.0], ["L", "L", "L"])
    assert PA.strides_inside([q], 0.0, 5.0) == 0


# ---------------------------------------------------------------- analytic share
def test_union_measure():
    assert PA.union_measure([(0, 1), (0.5, 2), (3, 4)], 0, 10) == pytest.approx(3.0)
    assert PA.union_measure([(-5, 1), (9, 20)], 0, 10) == pytest.approx(2.0)
    assert PA.union_measure([], 0, 10) == 0.0


@pytest.mark.parametrize("span", [30.0, 12.0, 10.0, 6.0])
def test_anchored_share_matches_brute_force(span):
    rows = _walk(t0=2.0, n=10, step_t=0.55) + _walk(t0=11.0, n=8, step_t=0.6, first_foot="L", pid="1")
    passes = PA.pass_footfalls([PA.project(r) for r in rows], CELL_CM)["X_SelfPace"]
    dur = min(10.0, span)
    share = PA.anchored_share(passes, span)
    if span - dur <= 1e-9:
        brute = 1.0 if PA.window_segments(passes, 0.0, dur) else 0.0
    else:
        grid = np.linspace(0.0, span - dur, 4001)
        brute = np.mean([bool(PA.window_segments(passes, a, dur)) for a in grid])
    assert share == pytest.approx(brute, abs=2e-3)


def test_anchor_interval_lower_edge_far_from_zero():
    # L full at 21.5 and 22.5 (R at 22.0): windows of 10 s starting in [12.5, 21.5] hold the L stride
    q = _pass([21.5, 22.0, 22.5], ["L", "R", "L"])
    assert PA.anchor_intervals([q], 10.0) == [(12.5, 21.5)]
    assert PA.anchored_share([q], 60.0) == pytest.approx(9.0 / 50.0, rel=1e-12)


def test_anchored_share_brute_force_far_from_zero():
    rows = _walk(t0=25.0, n=10, step_t=0.55) + _walk(t0=40.0, n=8, step_t=0.6, first_foot="L", pid="1")
    passes = PA.pass_footfalls([PA.project(r) for r in rows], CELL_CM)["X_SelfPace"]
    span, dur = 60.0, 10.0
    grid = np.linspace(0.0, span - dur, 20001)
    brute = np.mean([bool(PA.window_segments(passes, a, dur)) for a in grid])
    assert PA.anchored_share(passes, span) == pytest.approx(brute, abs=5e-4)


def test_anchored_share_short_span_clamps_the_window():
    # span 6 s: the window is the whole recording (dur 6); an L stride ending at 7.0 lies outside it
    q = _pass([1.0, 4.0, 7.0], ["L", "R", "L"])
    assert PA.anchored_share([q], 6.0) == 0.0
    # an L stride ending exactly at the span's end: interval [0.0, 0.5] holds t0 = 0
    r = _pass([0.5, 3.0, 6.0], ["L", "R", "L"])
    assert PA.anchored_share([r], 6.0) == 1.0


def test_expected_share_is_span_weighted():
    q = _pass([21.5, 22.0, 22.5], ["L", "R", "L"])           # anchored_share 0.18 on a span of 60 s
    recs = [("SelfPace", 60.0, [q]), ("Balance", 180.0, []), ("TUG", 60.0, [q])]
    share = PA.anchored_share([q], 60.0)
    assert PA.expected_share(recs, ("SelfPace",)) == pytest.approx(60.0 / 300.0 * share)
    assert PA.expected_share(recs, ("SelfPace", "TUG")) == pytest.approx(120.0 / 300.0 * share)
    assert PA.expected_share([], ("SelfPace",)) == 0.0


# ---------------------------------------------------------------- scope summary
def test_scope_summary_shares_and_seconds():
    seg = PA.Segment("0", "L", 1.0, 3.0, 2.4, 2, 5)
    seg2 = PA.Segment("1", "R", 5.0, 6.0, 1.0, 1, 3)
    table = [
        {"task": "SelfPace", "segments": [seg, seg2], "tokens": 100, "dur": 10.0, "strides": 4},
        {"task": "SelfPace", "segments": [], "tokens": 100, "dur": 10.0, "strides": 0},
        {"task": "TUG", "segments": [seg], "tokens": 50, "dur": 10.0, "strides": 2},
        {"task": "Balance", "segments": [], "tokens": 150, "dur": 10.0, "strides": 0},
    ]
    tot = {"tokens": 1000, "wg_tokens": 400, "syn_seconds": 300.0}
    out = PA.scope_summary(table, ("SelfPace",), tot, 2)
    assert out["anchored_windows"] == 1 and out["segments"] == 2
    assert out["share_of_weargait_windows"] == pytest.approx(1 / 4)
    assert out["anchored_tokens"] == 100
    assert out["share_of_weargait_tokens"] == pytest.approx(100 / 400)
    assert out["share_of_all_phase_b_tokens"] == pytest.approx(100 / 1000)
    assert out["anchored_windows_per_micro_step"] == pytest.approx(0.5)
    assert out["anchored_seconds"] == pytest.approx(3.0)
    assert out["anchored_seconds_per_micro_step"] == pytest.approx(1.5)
    assert out["anchored_seconds_over_synthetic_seconds"] == pytest.approx(3.0 / 300.0)
    assert out["anchored_share_of_window_seconds"]["p50"] == pytest.approx(0.3)


# ---------------------------------------------------------------- cell size, guards
def _trial(person, task, pkmas, cells, lockbox="False"):
    return {"trial_id": f"{person}_{task}", "person_id": f"weargait:{person}", "task": task, "lockbox": lockbox,
            "pkmas_stride_length_cm": pkmas, "ww_stride_cells_mean": cells}


def test_cell_fit_median_of_selfpace_ratios():
    rows = [_trial("A", "SelfPace", "127.0", "100.000"), _trial("B", "SelfPace", "131.0", "100.000"),
            _trial("C", "SelfPace", "126.0", "100.000"), _trial("D", "SelfPace", "", "100.000"),
            _trial("E", "SelfPace", "120.0", "0.000"), _trial("F", "HurriedPace", "200.0", "100.000")]
    fit = PA.cell_fit(rows)
    assert fit["n"] == 3 and fit["cell_cm"] == pytest.approx(1.27)
    assert fit["persons"] == ["weargait:A", "weargait:B", "weargait:C"]
    with pytest.raises(AssertionError):
        PA.cell_fit([_trial("A", "SelfPace", "127.0", "100.000", lockbox="True")])
    assert PA.cell_fit([_trial("A", "HurriedPace", "127.0", "100.000")])["n"] == 0


def _sets():
    from mdfm.data import sources
    train, dev14, dev12 = ["weargait:A", "weargait:B"], ["weargait:C", "weargait:D"], ["weargait:C"]
    f1 = {"train54_sha256": sources.person_hash(train), "dev14_sha256": sources.person_hash(dev14)}
    src = {"person_sha256": {"train": f1["train54_sha256"], "dev": f1["dev14_sha256"]}}
    return train, dev14, dev12, f1, src


def test_check_sets_accepts_and_refuses():
    from mdfm.data import sources
    train, dev14, dev12, f1, src = _sets()
    assert PA.check_sets(train, dev14, dev12, set(), f1, src)["n_train54"] == 2
    with pytest.raises(AssertionError):                       # a lockbox person in train54
        PA.check_sets(train, dev14, dev12, {"weargait:A"}, f1, src)
    with pytest.raises(AssertionError):                       # dev12 outside dev14
        PA.check_sets(train, dev14, ["weargait:E"], set(), f1, src)
    with pytest.raises(AssertionError):                       # train54 not F1's
        PA.check_sets(train + ["weargait:Z"], dev14, dev12, set(), f1, src)
    with pytest.raises(AssertionError):                       # the window store's train fold not train54
        PA.check_sets(train, dev14, dev12, set(), f1, {"person_sha256": {"train": "x", "dev": f1["dev14_sha256"]}})
    over = ["weargait:A", "weargait:C"]                         # train and dev overlap with matching hashes
    f2 = {"train54_sha256": sources.person_hash(over), "dev14_sha256": sources.person_hash(dev14)}
    s2 = {"person_sha256": {"train": f2["train54_sha256"], "dev": f2["dev14_sha256"]}}
    with pytest.raises(AssertionError):
        PA.check_sets(over, dev14, dev12, set(), f2, s2)


def test_guard_replayed():
    rows = [(0, "weargait:A", 1, 0.0, 10.0, 100), (1, "weargait:B", 2, 0.0, 10.0, 100)]
    assert PA.guard_replayed(rows, ["weargait:A", "weargait:B"], {"weargait:C"}) == 2
    with pytest.raises(AssertionError):
        PA.guard_replayed(rows, ["weargait:A"], set())
    with pytest.raises(AssertionError):
        PA.guard_replayed(rows, ["weargait:A", "weargait:B"], {"weargait:B"})


def test_reviewed_pd_folds(tmp_path):
    path = tmp_path / "folds.csv"
    lines = ["source,group,person_id,fold", "weargait_pd,PD,weargait:P1,dev", "weargait_pd,PD,weargait:P2,train",
             "other,PD,weargait:A,dev"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out, pd_all = PA.reviewed_pd_folds(["weargait:A"], ["weargait:C"], set(), str(path))
    assert out["persons"] == {"dev": 1, "train": 1} and pd_all == {"weargait:P1", "weargait:P2"}
    with pytest.raises(AssertionError):                       # a PD-fold person in train54
        PA.reviewed_pd_folds(["weargait:P1"], ["weargait:C"], set(), str(path))
    with pytest.raises(AssertionError):                       # a PD-fold person in dev14
        PA.reviewed_pd_folds(["weargait:A"], ["weargait:P2"], set(), str(path))
    with pytest.raises(AssertionError):                       # a lockbox person in the PD folds
        PA.reviewed_pd_folds(["weargait:A"], ["weargait:C"], {"weargait:P2"}, str(path))


def _patch_targets(monkeypatch, tmp_path, phases_lockbox="False"):
    from mdfm.gates import g3
    events = [_row(1.0, "L", 10, trial="A_SelfPace", person_id="weargait:A"),
              _row(1.5, "R", 70, trial="A_SelfPace", person_id="weargait:A"),
              _row(2.0, "L", 130, trial="A_SelfPace", person_id="weargait:A")]
    trials = [dict(_trial("A", "SelfPace", "127.0", "100.000"), dt_irregular="0", time_nonfinite="0", samples="500")]

    def fake_read_rows(path, persons, **kw):
        rows = events if path == g3.EVENTS_CSV else trials if path == g3.TRIALS_CSV else None
        assert rows is not None, path
        return [r for r in rows if r["person_id"] in set(persons)]

    (tmp_path / "phases.csv").write_text(
        "trial_id,person_id,group,task,lockbox,run_index,label,start_s,end_s,duration_s,samples\n"
        f"A_SelfPace,weargait:A,control,SelfPace,{phases_lockbox},0,Walk,0.0,5.0,5.0,500\n", encoding="utf-8")
    monkeypatch.setattr(g3, "target_meta", lambda: (1.25, g3.TARGET_RULE))
    monkeypatch.setattr(g3, "read_rows", fake_read_rows)
    monkeypatch.setattr(g3, "TARGETS_DIR", str(tmp_path))


def test_read_targets_uses_the_train_cell(monkeypatch, tmp_path):
    _patch_targets(monkeypatch, tmp_path)
    passes, passes_g3, trials, first, counts = PA.read_targets(["weargait:A"])
    assert counts["cell_cm"] == pytest.approx(1.27) and counts["cell_cm_g3_summary"] == pytest.approx(1.25)
    assert passes["A_SelfPace"][0].heel_m[2] == pytest.approx(1.30 * 1.27)
    assert passes_g3["A_SelfPace"][0].heel_m[2] == pytest.approx(1.30 * 1.25)
    assert first == {"A_SelfPace": 0.0} and counts["cell_fit_train54"]["n_persons"] == 1


def test_read_targets_refuses_a_lockbox_phase_row(monkeypatch, tmp_path):
    _patch_targets(monkeypatch, tmp_path, phases_lockbox="True")
    with pytest.raises(AssertionError):
        PA.read_targets(["weargait:A"])


# ---------------------------------------------------------------- replay check
def test_replay_check_every_interval_and_weargait_tokens():
    n = 250
    micro = {"windows": np.full(n, 10, np.int64), "tokens": np.full(n, 100, np.int64),
             "wg_tokens": np.full(n, 10, np.int64), "wg_windows": np.ones(n, np.int64)}
    # the first interval's log counts 98 useful tokens per micro-step (the plan 100): the same 1,000 weargait tokens
    log = [(50, 1000, 1000 / 9800, 100, 98.0), (100, 998, 0.1, 100, 100.0), (125, 500, 0.1, 50, 100.0)]
    out = PA.replay_check(micro, log)
    assert out["intervals_compared"] == 3 and out["intervals_partial"] == 1 and out["micro_steps_compared"] == 250
    assert out["windows_plan_minus_log"]["p100"] == 2 and out["intervals_windows_differ"] == 1
    assert out["intervals_plan_fewer_windows"] == 0 and out["intervals_weargait_tokens_equal"] == 3
    assert out["tokens_plan_minus_log_useful"]["p100"] == pytest.approx(200.0)
    log2 = log[:2] + [(125, 500, 0.11, 50, 100.0)]
    out2 = PA.replay_check(micro, log2)
    assert out2["intervals_weargait_tokens_equal"] == 2
    assert out2["weargait_tokens_plan_minus_log"]["p0"] == pytest.approx(-50.0)
    assert PA.replay_check(micro, log + [(200, 10, 0.1, 100, 100.0)])["intervals_compared"] == 3


# ---------------------------------------------------------------- margins and the gain scan
def test_margins_take_the_larger_term():
    assert PA.success_margin(19.5149, 0.1274) == pytest.approx(1.95149)
    assert PA.success_margin(1.0, 0.2) == pytest.approx(0.4)
    assert PA.ni_margin(26.386, 0.8822) == pytest.approx(1.7644)
    assert PA.ni_margin(100.0, 1.0) == pytest.approx(5.0)


def test_gain_scan_head_and_smallest_share():
    import pandas as pd
    g = 0.85
    x = np.array([100.0 + 10.0 * i for i in range(6)])
    units = pd.DataFrame({"person_id": [f"weargait:P{i}" for i in range(6)], "x": x, "y": g * x})
    raw = (1.0 - g) * x.mean()
    out = PA.gain_scan(units, g, 0.495 * raw)
    assert out["raw_recomputed"] == pytest.approx(raw)
    assert out["head_line"] == pytest.approx(0.0, abs=1e-9)        # f = 1 divides the decoded values by g
    assert out["head_improvement"] == pytest.approx(raw)
    assert out["smallest_share_meeting"]["f"] == pytest.approx(0.5)
    assert out["at_half"]["line"] == pytest.approx(0.5 * raw)
    assert out["at_quarter"]["gain"] == pytest.approx(g + 0.25 * (1 - g))


# ---------------------------------------------------------------- label-free names and the term size
def test_segments_carry_no_label_key():
    from mdfm.train.losses import assert_label_free, label_key
    p = PA.pass_footfalls([PA.project(r) for r in _walk()], CELL_CM)["X_SelfPace"][0]
    s = PA.segment_of(p, 0.0, 10.0)
    assert_label_free(s)
    assert not any(label_key(n) for n in PA.NAMES_TO_CHECK)
    out = PA.label_free_names()
    assert out["ok"] and all(out["controls"].values())


def test_label_free_names_not_ok_when_a_control_passes(monkeypatch):
    from mdfm.train import losses
    monkeypatch.setattr(losses, "label_key", lambda name: False)
    out = PA.label_free_names()
    assert not out["ok"] and not any(out["controls"].values())


def test_weight_arithmetic(monkeypatch):
    ref = {"step": 1, "loss": 1000.0, "sup_root_rate_u": 200.0, "sup_root_rate_yaw": 1800.0, "sup_root_rate": 2000.0,
           "hsic_active": 0.5, "weights_b": {}, "weighted": {"imu_nll": 4000.0, "sup_deriv": 400.0}}
    monkeypatch.setattr(PA, "v1_last_record", lambda weights_b=None: ref)
    scopes = {"g3": {"segment_speed_m_s": {"p50": 1.0}, "anchored_seconds_per_micro_step": 2.0,
                     "anchored_share_of_window_seconds": {"p50": 0.5}}}
    out = PA.weight_arithmetic(scopes, {"windows": 1000}, 10, 0.8)
    x = out["g3"]
    assert x["speed_error_m_s"] == pytest.approx(0.2)
    assert x["per_knot"] == pytest.approx(0.04 / 0.02)
    assert x["per_micro_step"] == pytest.approx(200 * 2.0)
    assert x["per_window_log_mean"] == pytest.approx(4.0)
    assert x["per_anchored_window"] == pytest.approx(500 * 2.0)
    assert x["per_window_over_v1_sup_root_rate_u"] == pytest.approx(0.02)
    o = x["per_window_over_v1"]
    assert o["sup_root_rate"] == pytest.approx(0.002) and o["sup_root_rate_yaw"] == pytest.approx(4.0 / 1800.0)
    assert o["loss"] == pytest.approx(0.004) and o["imu_nll_weighted"] == pytest.approx(0.001)
    assert o["sup_deriv_weighted"] == pytest.approx(0.01)


def test_scopes_and_whitelist_constants():
    assert set(PA.SCOPES["g3"]) == {"SelfPace", "HurriedPace"}
    assert set(PA.SCOPES["walkway"]) - set(PA.SCOPES["g3"]) == set(PA.MAT_TASKS)
    assert not set(PA.OTHER_TASKS) & set(PA.SCOPES["walkway"])
    assert "group" not in PA.ANCHOR_COLUMNS and "phase" not in PA.ANCHOR_COLUMNS
    assert PA.FOLDS_REVIEWED.replace("\\", "/").endswith(
        "reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/folds_proposed.csv")
