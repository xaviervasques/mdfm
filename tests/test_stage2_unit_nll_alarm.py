"""Tests of scripts/stage2_unit_nll_alarm.py (task S2CA): its readings of the unit_nll_rise alarm read as the loop
reads it (loop.unit_nll_rising for one read, loop.alarm_streaks for reads in a row), on toy series; no run folder,
no store, no GPU."""
import importlib.util
import json
import os

import numpy as np
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("_s2ca_script", os.path.join(TREE, "scripts", "stage2_unit_nll_alarm.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

LP = pytest.importorskip("mdfm.train.loop")


def loop_flags(ho, tr, reads=3):
    """The loop's unit_nll_rising at every read of one unit's series (the history up to that read)."""
    out = []
    for t in range(len(ho)):
        hist = {"u": list(zip(ho[:t + 1], tr[:t + 1]))}
        out.append(bool(LP.unit_nll_rising(hist, reads=reads)))
    return out


def test_rising_k_is_the_loops_test_on_random_series():
    rng = np.random.default_rng(5)
    for _ in range(200):
        ho, tr = rng.normal(size=9), rng.normal(size=9)
        for k in (3, 4):
            ours = A.rising_k(ho[None, :], tr[None, :], k)[0].tolist()
            assert ours == loop_flags(ho, tr, reads=k)


def test_rising_k_needs_both_parts_and_strict_steps():
    ho = np.array([[1.0, 2.0, 3.0, 3.0, 4.0, 5.0]])
    tr = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    assert A.rising_k(ho, tr, 3)[0].tolist() == [False, False, True, False, False, True]
    assert not A.rising_k(ho, -tr, 3).any()


def test_fire_step_counts_reads_in_a_row_as_alarm_streaks(tmp_path):
    counts = [0, 1, 0, 1, 1, 0, 1, 1, 1]
    assert A.fire_step(np.array([counts]), 1)[0] == 1
    assert A.fire_step(np.array([counts]), 2)[0] == 4
    assert A.fire_step(np.array([counts]), 3)[0] == 8
    assert A.fire_step(np.array([counts]), 4)[0] == -1
    # the loop's count of a dev alarm in a row, over the same reads (unit_nll_rise above 0.0 from step 0)
    log = tmp_path / "log.jsonl"
    with open(log, "w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(counts):
            f.write(json.dumps({"event": "dev", "phase": "C", "step": 1000 * i,
                                "dev": {"model": {"unit_nll_rise": c}}}) + "\n")
    streak = [LP.alarm_streaks(str(log), "C", 1000 * (i + 1), {"unit_nll_rise": 2})["unit_nll_rise"]
              for i in range(len(counts))]
    first2 = next(i for i, s in enumerate(streak) if s >= 2)
    first3 = next(i for i, s in enumerate(streak) if s >= 3)
    assert (first2, first3) == (4, 8)


def test_reading_counts_add_units_and_floors_cut():
    up = np.array([[0.0, 1.0, 2.0, 4.0]])
    down = -up
    flat = np.zeros((1, 4))
    series = {"a": (up, down), "b": (flat, flat)}
    sig = {"a": (1.0, 1.0), "b": (1.0, 1.0)}
    assert A.reading_counts(series, "pattern", 3, None, sig)[0].tolist() == [0, 0, 1, 1]
    # the held-out rises over three reads by 2, then 3: both above 1 x sqrt(2), only 3 above 2 x sqrt(2)
    assert A.reading_counts(series, "pattern", 3, 1.0, sig)[0].tolist() == [0, 0, 1, 1]
    assert A.reading_counts(series, "pattern", 3, 2.0, sig)[0].tolist() == [0, 0, 0, 1]
    # the gap rises by 4, then 6: both above 2 x sqrt(2), only 6 above 3 x sqrt(2)
    assert A.reading_counts(series, "gap", 3, 2.0, sig)[0].tolist() == [0, 0, 1, 1]
    assert A.reading_counts(series, "gap", 3, 3.0, sig)[0].tolist() == [0, 0, 0, 1]
    # two units raised at one read count twice
    two = {"a": (up, down), "c": (up, down)}
    assert A.reading_counts(two, "pattern", 3, None, {"a": (1.0, 1.0), "c": (1.0, 1.0)})[0].tolist() == [0, 0, 2, 2]


def test_baseline_reads_after_its_first_k_reads():
    ho = np.array([[0.0, 0.0, 0.0, 3.0, 4.0, 1.0]])
    hits = A.unit_hits("baseline", ho, np.zeros_like(ho), 3, 3.0, 1.0, 1.0)[0].tolist()
    # threshold 3 x 1 x sqrt(4/3) = 3.46: read 4 (4.0) only
    assert hits == [False, False, False, False, True, False]


def test_simulate_null_matches_the_exact_share_without_correlation():
    params = {"u": {"sd_ho": 1.0, "sd_tr": 1.0, "rho": 0.0}}
    r = A.simulate(params, n_sim=40000, seed=3)["readings"]["loop_3_reads"]
    assert r["unit_per_read"]["u"] == pytest.approx(1.0 / 36.0, rel=0.05)


def test_censoring_names_the_streak_a_stop_cuts():
    def run(ho, tr):
        return {"u": {"held_out": {"series": {str(1000 * (i + 1)): v for i, v in enumerate(ho)}},
                      "train": {"series": {str(1000 * (i + 1)): v for i, v in enumerate(tr)}}}}
    L = {"val_units": ["u"],
         "replicate": {"u": {"held_out": {"rms_over_sqrt2_from_2000": 1.0}, "gap": {"rms_over_sqrt2_from_2000": 1.0}}},
         # r1: the loop's pattern at the last read only (a one-read stop there); r2: the pattern at the last two reads
         "per_run": {"r1": run([5.0, 1.0, 2.0, 3.0], [0.0, 3.0, 2.0, 1.0]),
                     "r2": run([0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]),
                     "r3": run([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])}}
    c = A.censoring_on_logs(L, runs=("r1", "r2", "r3"))
    assert c["r1"]["last_read"] == 4000
    r1 = c["r1"]["readings"]
    assert r1["loop_3_reads"]["fires"] == 4000 and not r1["loop_3_reads"]["open"]
    assert r1["consecutive_2"]["fires"] is None and r1["consecutive_2"]["streak_at_last_read"] == 1
    assert r1["consecutive_2"]["open"] and r1["consecutive_2"]["units_at_last_read"] == ["u"]
    r2 = c["r2"]["readings"]
    assert r2["consecutive_2"]["fires"] == 4000 and r2["consecutive_3"]["streak_at_last_read"] == 2
    assert r2["consecutive_3"]["open"]
    r3 = c["r3"]["readings"]
    assert all(v["fires"] is None and v["streak_at_last_read"] == 0 and not v["open"] for v in r3.values())


def test_phase_reads_end_at_the_phase_end():
    xs = A.phase_reads(35754, 1000, 1000)
    assert xs[0] == 1000 and xs[-2] == 35000 and xs[-1] == 35754 and len(xs) == 36
    assert A.phase_reads(3000, 1000, 1000) == [1000, 2000, 3000]
