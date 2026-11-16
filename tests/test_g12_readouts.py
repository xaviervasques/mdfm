"""Tests of the G12 readouts and the section 8 comparison (task G12R; mdfm/baselines/readouts.py,
mdfm/baselines/comparison.py, scripts/g12_readout.py, scripts/g12_compare.py, the per-row output of
mdfm/eval/probes.leakage_probe), on synthetic data only: fake frozen tokens, synthetic walkway targets and tiny models.
No weargait file, record or checkpoint is read.

- the line from per-row predictions is the G3 harness's point on the same rows (stride, speed, IC with an unmatched
  reference IC and a person without a matched stride);
- the probe fit never sees a dev person (every fit and every penalty choice spied), and dev targets or dev features
  change no fit; the variant choice reads the fitting set only (dev features turned upside down leave it unchanged);
  run_probes checks its rows against g3's points;
- the separation rule's exact behaviour (strict, ties, mirror, NaN, higher is better), the comparator pick (better
  seed mean, ties to mfr, an unusable arm left out), the reading (both conditions, either side);
- the bootstrap: the person is the unit, a person drawn twice counts twice, absent persons drop out of a resample, the
  fast path equals the general path on the drawn rows, one seed against one seed equals common.cluster_bootstrap of
  the per-person differences, infinite and undefined differences, nothing refitted inside;
- NOT EVALUATED: an unregistered rule, too few seeds per model, an arm unusable, G8, G10 and G7 always with a reason;
  sources scored against different MoveAtlas commits or none, a source's dev set that is not the comparison's, a
  baseline seed the body has not, three baseline seeds that are not the body's;
- leakage_probe: its default return is unchanged, rows=True adds per-row predictions that give back the leakage number
  and feed the general bootstrap;
- tokens: encode_windows on a tiny body model and on a tiny G12 model of the same seed give the same rows, J0's for the
  shared variants; collect pools each pass and keeps three token variants; a checkpoint is loaded once, by the kind
  its run folder's config.json tells; the training side of mdfm/baselines imports no reading side;
- the scripts: g12_readout's main runs end to end on the CPU for tiny synthetic inputs (g3's data side and the model
  loader replaced: a tiny G12 model's tokens on synthetic windows), loads the checkpoint once, finds its FINISHED run
  and flags the verified final checkpoint, writes its five outputs and its feature file, refits from that file with
  --run-dir and --from-features to the same lines, and stops (before any record is read) on a dev control among the
  checkpoint's training persons, a lockbox person among the controls, a kept lockbox target row, folds of other sizes
  or overlapping, --limit-persons without --out, a feature file of another checkpoint or of other passes; it reads only
  a FINISHED run; g12_compare reads the body's per-person rows and the readouts, checks them against their reports,
  refuses partial, unverified or duplicated readouts, states the dev12 source of the body's points and writes NOT
  EVALUATED with what is missing.
"""
import importlib.util
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm.baselines import comparison as CMP  # noqa: E402
from mdfm.baselines import readouts as RO  # noqa: E402
from mdfm.eval import probes as P  # noqa: E402
from mdfm.gates import common, g3  # noqa: E402

EM_DASH, EN_DASH = chr(0x2014), chr(0x2013)


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_g12r_test", os.path.join(TREE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- synthetic walkway targets and features
def _targets(persons, gaits, tasks=("SelfPace", "HurriedPace"), passes=3, T=3000):
    """g3.Targets of synthetic persons (as tests/test_probes.py): per trial `passes` passes of 4 s, ICs alternating feet
    one step apart, strides two footfalls apart, pass speed = speed."""
    trials, ics, strides, ps = [], [], [], []
    for p, (v0, c0) in zip(persons, gaits):
        for task in tasks:
            v, c = (v0, c0) if task == "SelfPace" else (1.2 * v0, 1.1 * c0)
            tid = f"{p.split(':')[1]}_{task}"
            trials.append((tid, p, task, T))
            for k in range(passes):
                t = np.round(2.0 + 8.0 * k + np.arange(0.0, 4.0, 1.0 / c), 3)
                feet = ["L" if i % 2 == 0 else "R" for i in range(len(t))]
                ics += [(tid, p, task, f, float(x)) for f, x in zip(feet, t)]
                for i in range(2, len(t)):
                    strides.append((tid, p, task, str(k), feet[i], float(t[i - 2]), float(t[i]), v * 2.0 / c))
                ps.append((tid, p, task, str(k), float(t[0]), float(t[-1]), len(t), v))
    ic = pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"])
    return g3.Targets(tuple(sorted(persons)), False,
                      pd.DataFrame(trials, columns=["trial_id", "person_id", "task", "samples"]), ic, ic.copy(),
                      pd.DataFrame(strides, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                     "length_m"]),
                      pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                                "speed_m_s"]),
                      pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})


def _cohort(n_train=12, n_dev=6, seed=7):
    names = [f"weargait:{'HC' if i % 2 else 'WHC'}{i:03d}" for i in range(n_train + n_dev)]
    rng = np.random.default_rng(seed)
    gaits = {n: (float(rng.uniform(0.8, 1.6)), float(rng.uniform(1.6, 2.1))) for n in names}
    train, dev = names[:n_train], names[n_train:]
    return train, dev, _targets(train, [gaits[p] for p in train]), _targets(dev, [gaits[p] for p in dev])


def _ic_tokens(tg):
    rows = [(t, p, s) for t, p in zip(tg.trials["trial_id"], tg.trials["person_id"]) for s in np.arange(0.0, 30.0, 0.25)]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def _data(train, dev, tg_tr, tg_dev, signal=None, seed=0, D=12):
    """Synthetic readout features: per variant, pass rows that carry stride and speed when the variant is in
    `signal[set]` (noise otherwise), token rows that carry the IC sub-patch labels in the same way."""
    signal = signal or {"train": RO.VARIANTS, "dev": RO.VARIANTS}
    rng = np.random.default_rng(seed)
    W = {v: (rng.standard_normal(D), rng.standard_normal(D)) for v in RO.VARIANTS}
    M = {v: rng.standard_normal((2 * P.SUB_BINS, D)) for v in RO.IC_VARIANTS}
    data = {"units": {"train": P.pass_units(tg_tr), "dev": P.pass_units(tg_dev)}}
    for s, tg in (("train", tg_tr), ("dev", tg_dev)):
        u = data["units"][s]
        X = {}
        for v in RO.VARIANTS:
            noise = rng.standard_normal((len(u), D))
            if v in signal[s]:
                X[v] = (np.outer(u["stride_m"], W[v][0]) + np.outer(u["speed_m_s"], W[v][1]) + 0.05 * noise)
            else:
                X[v] = noise
        tids, pids, starts = _ic_tokens(tg)
        Y, _ = P.ic_targets(starts, tids, tg.walkway_ic, u[["trial_id", "t0", "t1"]])
        tok = {"trial_id": tids, "person_id": pids, "start": starts}
        for v in RO.IC_VARIANTS:
            noise = rng.standard_normal((len(Y), D))
            tok[v] = ((Y @ M[v] + 0.05 * noise) if v in signal[s] else noise).astype(np.float16)
        data[s] = {"X": X, "n_rows": np.full(len(u), 12), "tokens": tok,
                   "counts": {"records": len(tg.trials), "windows": 5 * len(tg.trials), "passes": len(u),
                              "passes_with_rows": len(u), "passes_without_rows": 0, "trials_without_record": [],
                              "token_rows_kept": len(starts), "rows_per_pass_median": 12.0}}
    return data


@pytest.fixture
def cohort(monkeypatch):
    train, dev, tg_tr, tg_dev = _cohort()
    monkeypatch.setattr(g3, "controls", lambda fold: sorted(dev) if fold == "dev" else sorted(train))
    monkeypatch.setattr(g3, "DEV_SETS", {"dev14": (), "dev12": (dev[0], dev[1])})
    return train, dev, tg_tr, tg_dev


# ---------------------------------------------------------------- rows and the line statistic
def test_the_line_from_rows_is_the_g3_point_on_the_same_rows(cohort):
    train, dev, _, tg = cohort
    u = P.pass_units(tg)
    rng = np.random.default_rng(3)
    pred = {"stride_m": u["stride_m"].to_numpy() + 0.05 * rng.standard_normal(len(u)),
            "speed_m_s": u["speed_m_s"].to_numpy() + 0.05 * rng.standard_normal(len(u))}
    lone = u.index[u["person_id"] == dev[2]]
    pred["stride_m"][lone] = np.nan                        # a person without any matched stride drops out
    pred["speed_m_s"][lone[:1]] = np.nan                   # an unmatched pass
    pp = P.pass_predictions(u, pred, tg, "x")
    st, st_tab = g3.stride_lines(pp, tg)
    sp, sp_tab = g3.speed_line(pp, tg)
    ref = g3.reference_ics(tg, "walkway")
    ics = ref[["trial_id", "foot", "t_s"]].copy()
    ics["t_s"] += 0.01 * rng.standard_normal(len(ics))
    drop = ref["person_id"].to_numpy() == dev[3]
    ics = ics[~(drop & (np.arange(len(ics)) % 3 != 0))]     # most ICs of one person unmatched: an infinite value
    icl, ic_tab = g3.ic_line(g3.Predictions("x", ics), ref, 0.0)
    rows = pd.concat([RO._pass_rows(st_tab, "stride_mae_cm", "v", "err_cm", "pred_m"),
                      RO._pass_rows(sp_tab, "speed_mae_m_s", "v", "err_m_s", "pred_m_s"), RO._ic_rows(ic_tab, "v")],
                     ignore_index=True)
    assert list(rows.columns) == CMP.ROW_COLUMNS
    assert st["stride_mae_cm"]["persons_unmatched"] == [dev[2]] and icl["per_person"][dev[3]] == float("inf")
    for ln, stats in (("stride_mae_cm", st["stride_mae_cm"]), ("speed_mae_m_s", sp), ("ic_median_ms", icl)):
        g = rows[rows["line"] == ln]
        how = CMP.LINES[ln]["how"]
        got = CMP.line_point(g, how)
        assert got == stats["point"] or got == pytest.approx(stats["point"], rel=1e-12), ln
        ids, v = CMP.person_values(g, how)
        assert dict(zip(ids, v)) == pytest.approx(stats["per_person"]), ln
        sub = dev[2:]
        tg_sub = P.subset_targets(tg, sub)
        if ln == "stride_mae_cm":
            want = g3.stride_lines(pp, tg_sub)[0][ln]["point"]
        elif ln == "speed_mae_m_s":
            want = g3.speed_line(pp, tg_sub)[0]["point"]
        else:
            want = g3.ic_line(g3.Predictions("x", ics), g3.reference_ics(tg_sub, "walkway"), 0.0)[0]["point"]
        assert CMP.line_point(g, how, sub) == want or CMP.line_point(g, how, sub) == pytest.approx(want), ln
    assert np.isnan(CMP.line_point(rows.iloc[0:0], "mean"))


def test_the_person_is_the_unit_of_the_line():
    rows = pd.DataFrame({"line": "stride_mae_cm", "person_id": ["a"] * 100 + ["b"], "enters": True,
                         "value": [1.0] * 100 + [9.0]})
    assert CMP.line_point(rows, "mean") == pytest.approx(5.0)            # not the row mean, 1.08
    rows.loc[0, "enters"] = False
    rows.loc[1, "value"] = np.nan                                        # NaN skipped inside a person, as in g3
    assert CMP.line_point(rows, "mean") == pytest.approx(5.0)
    v, present = CMP.values_on(["b", "a", "c"], rows, "mean")
    assert present.tolist() == [True, True, False] and v[:2].tolist() == [9.0, 1.0] and np.isnan(v[2])


# ---------------------------------------------------------------- separation, comparator, reading
def test_the_separation_rule_exactly():
    s = CMP.separation
    assert s([1, 2, 3], [4, 5, 6]) == CMP.BODY
    assert s([1, 2, 4], [4, 5, 6]) is None                                # a tie at 4 is no separation
    assert s([1, 2, 5], [4, 5, 6]) is None                                # overlap
    assert s([7, 8, 9], [4, 5, 6]) == "comparator"                        # the mirror
    assert s([1, 2, 3], [4, 5, float("nan")]) is None and s([], [1]) is None
    assert s([1, 2, 3], [4, 5, float("inf")]) == CMP.BODY                 # an infinite comparator seed is worse
    assert s([float("inf")] * 3, [float("inf")] * 3) is None
    assert s([7, 8, 9], [4, 5, 6], higher_is_better=True) == CMP.BODY
    assert s([1, 2, 3], [4, 5, 6], higher_is_better=True) == "comparator"
    assert s([3], [3]) is None


def test_the_comparator_is_the_arm_with_the_better_seed_mean():
    pick = CMP.pick_comparator
    assert pick({"mfr": [5, 6, 7], "jepa": [4, 6, 7]}) == ("jepa", {"mfr": 6.0, "jepa": pytest.approx(17 / 3)})
    assert pick({"mfr": [5, 6, 7], "jepa": [7, 6, 5]})[0] == "mfr"                  # a tie goes to mfr
    assert pick({"mfr": [5, 6, 7], "jepa": [1, 1, 1]}, unusable={"jepa": "collapsed"})[0] == "mfr"
    assert pick({"mfr": [5, 6, 7], "jepa": [1, 1, 1]}, higher_is_better=True)[0] == "mfr"
    assert pick({"mfr": [5, 6, float("inf")], "jepa": [7, 8, 9]})[0] == "jepa"
    assert pick({"mfr": [5, 6, float("nan")], "jepa": [7, 8, 9]})[0] is None
    assert pick({"jepa": [1, 2, 3]}, unusable={"jepa": "out"}) == (None, {})


def test_the_reading_needs_the_separation_and_the_interval():
    r = CMP.reading
    assert r(CMP.BODY, [-3.0, -0.1]) == ("body better", True, False)
    assert r(CMP.BODY, [-3.0, 0.1]) == ("no difference", False, False)          # separated, interval crosses 0
    assert r(None, [-3.0, -1.0]) == ("no difference", False, False)             # interval alone
    assert r("comparator", [0.2, 3.0]) == ("body worse", False, True)
    assert r("comparator", [-0.2, 3.0])[0] == "no difference"
    assert r(CMP.BODY, None)[0] == "no difference"
    assert r(CMP.BODY, [0.1, 2.0], higher_is_better=True)[1] is True            # higher: the body's side is above 0
    assert r(CMP.BODY, [-0.1, 2.0], higher_is_better=True)[1] is False


# ---------------------------------------------------------------- the person bootstrap
def _pp_rows(values, line="stride_mae_cm"):
    """One entering row per person (the per-person source format)."""
    return pd.DataFrame({"line": line, "person_id": list(values), "enters": True,
                         "value": [float(x) for x in values.values()]})


def test_a_person_drawn_twice_counts_twice_and_an_absent_person_drops_out():
    v = np.array([1.0, 2.0, 10.0])
    allp = np.ones(3, bool)
    assert CMP.resample_line(v, allp, [[0, 0, 2]])[0] == pytest.approx(4.0)      # (1 + 1 + 10) / 3
    assert CMP.resample_line(v, allp, [[2, 2, 2]])[0] == pytest.approx(10.0)
    present = np.array([True, False, True])
    assert CMP.resample_line(v, present, [[0, 1, 2]])[0] == pytest.approx(5.5)   # the absent slot is left out
    assert CMP.resample_line(v, present, [[0, 1, 1]])[0] == pytest.approx(1.0)
    assert np.isnan(CMP.resample_line(v, present, [[1, 1, 1]])[0])
    assert CMP.resample_line([1.0, np.inf, 2.0], allp, [[0, 1, 2], [0, 2, 2]]).tolist() == [np.inf, pytest.approx(5 / 3)]
    # the general path relabels each drawn slot as its own person, so the duplicates count twice there too
    rows = pd.concat([_pp_rows({"a": 1.0, "b": 2.0, "c": 10.0}), pd.DataFrame(
        {"line": "stride_mae_cm", "person_id": ["a"] * 4, "enters": True, "value": [1.0] * 4})], ignore_index=True)
    drawn = CMP.resample_rows(rows, ["a", "b", "c"], [0, 0, 2])
    assert sorted(set(drawn["person_id"])) == ["u0", "u1", "u2"] and len(drawn) == 5 + 5 + 1
    assert CMP.line_point(drawn, "mean") == pytest.approx(4.0)


def test_the_fast_bootstrap_equals_the_general_path_on_the_drawn_rows():
    rng = np.random.default_rng(1)
    persons = [f"p{i}" for i in range(9)]

    def frame(shift):
        n = rng.integers(1, 6, size=len(persons))
        pid = np.repeat(persons, n)
        val = np.abs(rng.normal(5 + shift, 2, size=len(pid)))
        return pd.DataFrame({"line": "ic_median_ms", "person_id": pid, "enters": rng.random(len(pid)) > 0.1,
                             "value": val})
    body = [frame(0.0) for _ in range(3)]
    comp = [frame(1.0) for _ in range(2)]
    fast = CMP.bootstrap_line(body, comp, persons, "median", n_boot=60, seed=5)
    slow = CMP.bootstrap_rows(body, comp, persons, lambda r: CMP.line_point(r, "median"), n_boot=60, seed=5)
    assert np.allclose(fast["diffs"], slow["diffs"], equal_nan=True)
    assert fast["interval"] == pytest.approx(slow["interval"]) and fast["n_boot"] == 60 and fast["n_persons"] == 9


def test_one_seed_against_one_seed_is_the_cluster_bootstrap_of_the_per_person_differences():
    rng = np.random.default_rng(2)
    persons = [f"weargait:HC{i:03d}" for i in range(14)]
    b = {p: float(x) for p, x in zip(persons, rng.uniform(5, 15, 14))}
    c = {p: float(x) for p, x in zip(persons, rng.uniform(8, 20, 14))}
    got = CMP.bootstrap_line([_pp_rows(b)], [_pp_rows(c)], persons)
    want = common.cluster_bootstrap([b[p] - c[p] for p in persons], persons, n=g3.BOOT_N, seed=g3.SEED)
    assert got["interval"] == pytest.approx(list(want), rel=1e-12) and got["seed"] == g3.SEED == CMP.SEED
    assert got["n_boot"] == g3.BOOT_N == CMP.BOOT_N and got["undefined"] == 0
    assert np.array_equal(CMP.draws(14), np.random.default_rng(g3.SEED).integers(0, 14, size=(g3.BOOT_N, 14)))


def test_infinite_and_undefined_differences():
    assert CMP.percentile_interval([1.0, 2.0, 3.0, 4.0]) == pytest.approx(list(np.quantile([1, 2, 3, 4], [0.025, 0.975])))
    lo, hi = CMP.percentile_interval([-np.inf] * 5 + list(np.linspace(-3, -1, 95)))
    assert lo == -np.inf and hi == pytest.approx(np.quantile(np.linspace(-3, -1, 95).tolist() + [-9] * 5, 0.975))
    assert CMP.percentile_interval([1.0, np.nan]) is None
    persons = ["a", "b", "c"]
    body = [_pp_rows({"a": 1.0, "b": 2.0, "c": 3.0}, "ic_median_ms")]
    comp = [_pp_rows({"a": 9.0, "b": np.inf, "c": 9.0}, "ic_median_ms")]
    out = CMP.bootstrap_line(body, comp, persons, "median", n_boot=200)
    assert out["interval"][0] == -np.inf and out["undefined"] == 0
    both = CMP.bootstrap_line(comp, comp, persons, "median", n_boot=200)
    assert both["interval"] is None and both["undefined"] > 0                     # inf minus inf


def test_nothing_is_refitted_inside_the_bootstrap(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a fit inside the comparison")
    for name in ("fit_probe", "ridge_path", "cross_fit", "select_alpha"):
        monkeypatch.setattr(P, name, boom)
    persons = [f"p{i}" for i in range(6)]
    body = {s: _pp_rows({p: 1.0 + 0.1 * s + i for i, p in enumerate(persons)}) for s in range(3)}
    arms = {"mfr": {s: _pp_rows({p: 9.0 + 0.1 * s + i for i, p in enumerate(persons)}) for s in range(3)}}
    out = CMP.compare_line("stride_mae_cm", persons, body, arms, n_boot=100)
    assert out["separation"] == CMP.BODY and out["reading"] == "body better" and out["interval"][1] < 0
    assert out["difference"] == pytest.approx(-8.0)


# ---------------------------------------------------------------- NOT EVALUATED
LINE_SHIFT = {"stride_mae_cm": 0.0, "speed_mae_m_s": 50.0, "ic_median_ms": 100.0}


def _seeds(persons, base, n):
    """{seed: rows of the three lines}, each line shifted by LINE_SHIFT so that a line read from another's rows shows."""
    return {s: pd.concat([_pp_rows({p: base + LINE_SHIFT[ln] + 0.1 * s + 0.2 * i for i, p in enumerate(persons)}, ln)
                          for ln in CMP.LINES], ignore_index=True) for s in range(n)}


def test_not_evaluated_until_the_rule_is_registered_and_every_model_has_three_seeds():
    persons = [f"p{i}" for i in range(8)]
    sets = {"dev14": persons, "dev12": persons[2:]}
    body, mfr = _seeds(persons, 2.0, 2), _seeds(persons, 9.0, 1)
    out = CMP.evaluate(body, {"mfr": mfr}, sets, registered=False, notes={"body": "third seed OPEN"})
    assert out["status"] == CMP.NOT_EVALUATED and "lines" not in out
    miss = " | ".join(out["missing"])
    assert "not registered by the owner" in miss and "body: 2 of 3 seeds" in miss and "third seed OPEN" in miss
    assert "mfr: 1 of 3" in miss and "jepa: 0 of 3" in miss
    assert out["verdict"].startswith(CMP.NOT_EVALUATED)
    # the rows are reported without a verdict
    assert out["points"]["dev12"]["stride_mae_cm"]["mfr"][0] == pytest.approx(9.0 + 0.2 * np.mean(range(2, 8)))
    assert out["points"]["dev14"]["ic_median_ms"]["body"][1] == pytest.approx(102.1 + 0.2 * 3.5)
    assert set(out["points"]["dev14"]["ic_median_ms"]) == {"body", "mfr"}
    for k in ("G8", "G10", "G7"):
        assert out["other_lines"][k]["status"] == CMP.NOT_EVALUATED and out["other_lines"][k]["reason"]
    # registered, still too few seeds: no rule item, the seed items stay
    out = CMP.evaluate(_seeds(persons, 2.0, 3), {"mfr": _seeds(persons, 9.0, 3)}, sets, registered=True)
    assert out["status"] == CMP.NOT_EVALUATED and out["missing"] == ["jepa: 0 of 3 seeds with readouts (none)"]
    # the JEPA arm unusable (collapsed): the MFR arm alone is the comparator
    out = CMP.evaluate(_seeds(persons, 2.0, 3), {"mfr": _seeds(persons, 9.0, 3)}, sets, registered=True,
                       unusable={"jepa": "collapsed"})
    assert out["status"] == CMP.EVALUATED and out["missing"] == []
    for s in sets:
        for ln in CMP.LINES:
            c = out["lines"][s][ln]
            assert c["comparator"] == "mfr" and c["separation"] == CMP.BODY and c["beats"], (s, ln)
            assert c["difference"] == pytest.approx(-7.0) and c["interval"][1] < 0, (s, ln)
    assert out["lines"]["dev14"]["ic_median_ms"]["body_mean"] == pytest.approx(102.8)
    assert out["g3"]["dev14"]["beats_every_line"] and "NOT SHOWN" in out["verdict"]
    # both arms usable, three seeds each: the better arm is the comparator
    out = CMP.evaluate(_seeds(persons, 2.0, 3), {"mfr": _seeds(persons, 9.0, 3), "jepa": _seeds(persons, 1.0, 3)},
                       sets, registered=True)
    c = out["lines"]["dev14"]["stride_mae_cm"]
    assert c["comparator"] == "jepa" and c["separation"] == "comparator" and c["reading"] == "body worse"
    out = CMP.evaluate(_seeds(persons, 2.0, 3), {"mfr": {}, "jepa": {}}, sets, registered=True,
                       unusable={"mfr": "x", "jepa": "y"})
    assert out["status"] == CMP.NOT_EVALUATED and "no usable baseline arm" in out["missing"][-1]


def test_one_comparison_reads_one_moveatlas_commit_its_own_sets_and_paired_seeds():
    persons = [f"p{i}" for i in range(8)]
    sets = {"dev14": persons, "dev12": persons[2:]}
    body, mfr = _seeds(persons, 2.0, 3), _seeds(persons, 9.0, 3)

    def prov():
        return {"body": {s: {"moveatlas_head": "e" * 40, "sets": {"dev14": list(persons)}} for s in body},
                "mfr": {s: {"moveatlas_head": "e" * 40, "sets": {k: list(v) for k, v in sets.items()}} for s in mfr},
                "jepa": {0: {"moveatlas_head": "b" * 40, "sets": {"dev14": persons[:3]}}}}   # unusable: not checked

    def run(p, b=body, m=mfr):
        return CMP.evaluate(b, {"mfr": m}, sets, registered=True, unusable={"jepa": "out"}, provenance=p)
    ok = run(prov())
    assert ok["status"] == CMP.EVALUATED and ok["missing"] == []
    p = prov()
    p["mfr"][1]["moveatlas_head"] = "a" * 40
    out = run(p)
    assert out["status"] == CMP.NOT_EVALUATED and "lines" not in out and len(out["missing"]) == 1
    assert "different commits" in out["missing"][0] and "mfr seed 1 aaaaaaa" in out["missing"][0]
    p["mfr"][1]["moveatlas_head"] = None
    assert run(p)["missing"] == ["MoveAtlas: no commit stated by mfr seed 1"]
    p = prov()
    p["mfr"][2]["sets"]["dev12"] = persons[3:]
    assert run(p)["missing"] == ["mfr seed 2: its dev12 persons (5) are not the comparison's dev12 (6)"]
    p = prov()
    p["body"][0]["sets"]["dev14"] = persons[::-1]                     # the order does not matter
    assert run(p)["missing"] == []
    assert run(None)["status"] == CMP.EVALUATED                         # no provenance: not checked
    # seeds: a baseline seed the body has not, or three seeds that are not the body's
    odd = dict(zip((0, 1, 7), mfr.values()))
    miss = run(None, m=odd)["missing"]
    assert len(miss) == 1 and miss[0].startswith("mfr: seeds 7 have no body seed of the same value (rule section 4")
    miss = run(None, b=_seeds(persons, 2.0, 4))["missing"]
    assert miss == [f"mfr: seeds 0, 1, 2 are not the body's (0, 1, 2, 3; {CMP.SEED_PAIRING})"]
    one = CMP.evaluate(body, {"mfr": dict(list(mfr.items())[:1])}, sets, registered=True, unusable={"jepa": "out"})
    assert one["missing"] == ["mfr: 1 of 3 seeds with readouts (0)"]  # a subset of the body's seeds: the count only


# ---------------------------------------------------------------- leakage_probe rows
def test_leakage_probe_rows_are_optional_and_give_back_the_number():
    rng = np.random.default_rng(4)
    persons = np.repeat([f"p{i:02d}" for i in range(20)], 5)
    lab = np.where(np.arange(len(persons)) // 5 % 2 == 0, "siteA", "siteB")
    X = rng.standard_normal((len(persons), 6))
    X[:, 0] += 2.0 * (lab == "siteA")
    cov = rng.standard_normal((len(persons), 1))
    old = P.leakage_probe(X, lab, persons, covariates=cov)
    assert set(old) == {"balanced_accuracy", "covariate_balanced_accuracy", "leakage_points", "chance", "n_rows",
                        "n_persons", "classes"}
    new = P.leakage_probe(X, lab, persons, covariates=cov, rows=True)
    assert {k: v for k, v in new.items() if k != "rows"} == old
    r = new["rows"]
    assert r["person_id"] == persons.tolist() and r["label"] == lab.tolist() and len(r["predicted"]) == len(lab)

    def leak(df):
        return 100.0 * (P.balanced_accuracy(df["label"], df["predicted"])
                        - P.balanced_accuracy(df["label"], df["covariate_predicted"]))
    df = pd.DataFrame(r)
    assert leak(df) == pytest.approx(old["leakage_points"])
    ids = sorted(set(persons))
    same = CMP.bootstrap_rows([df], [df.assign(predicted=df["covariate_predicted"])], ids, leak, idx=[list(range(20))])
    assert same["diffs"][0] == pytest.approx(old["leakage_points"])   # every person once: the point (minus 0)
    none = P.leakage_probe(X, lab, persons, rows=True)["rows"]
    assert none["covariate_predicted"] is None


# ---------------------------------------------------------------- the probe stage
def test_the_probe_fit_never_sees_a_dev_person(cohort, monkeypatch):
    train, dev, tg_tr, tg_dev = cohort
    data = _data(train, dev, tg_tr, tg_dev)
    fits = []
    real_fit, real_select = P.fit_probe, P.select_alpha

    def fit_spy(X, Y, persons, *a, **k):
        fits.append(set(np.asarray(persons).astype(str)))
        return real_fit(X, Y, persons, *a, **k)

    def select_spy(X, Y, persons, *a, **k):
        fits.append(set(np.asarray(persons).astype(str)))
        return real_select(X, Y, persons, *a, **k)
    monkeypatch.setattr(P, "fit_probe", fit_spy)
    monkeypatch.setattr(P, "select_alpha", select_spy)
    res = RO.run_probes(data, tg_tr, tg_dev, log=lambda *a, **k: None)
    assert len(fits) > 20 and all(not (s & set(dev)) for s in fits) and all(s <= set(train) for s in fits)
    # dev targets and dev features change no fit: the training-side numbers stay, the dev predictions follow the
    # dev features only
    data2 = _data(train, dev, tg_tr, tg_dev)
    for v in RO.VARIANTS:
        data2["dev"]["X"][v] = data2["dev"]["X"][v] * 0.0
    tg_dev2 = replace(tg_dev, strides=tg_dev.strides.assign(length_m=tg_dev.strides["length_m"] * 3.0))
    res2 = RO.run_probes(data2, tg_tr, tg_dev2, log=lambda *a, **k: None)
    assert res2["crossfit"] == res["crossfit"] and res2["chosen"] == res["chosen"]
    for v in RO.VARIANTS:
        assert res2["variants"]["pass"][v]["selection"] == res["variants"]["pass"][v]["selection"]
    assert res["rows_check"]["checked"] == 2 * (2 * len(RO.VARIANTS) + len(RO.IC_VARIANTS))
    with pytest.raises(ValueError):                          # a dev person among the fitting rows is refused
        leak = _data(train, dev, tg_tr, tg_dev)
        leak["units"]["train"] = pd.concat([leak["units"]["train"], leak["units"]["dev"].iloc[:1]], ignore_index=True)
        leak["train"]["X"] = {v: np.vstack([x, leak["dev"]["X"][v][:1]]) for v, x in leak["train"]["X"].items()}
        RO.pass_stage(leak, tg_tr, tg_dev, log=lambda *a, **k: None)


def test_the_variant_choice_reads_the_fitting_set_only(cohort):
    train, dev, tg_tr, tg_dev = cohort
    # on the fitting set only the measurement variant carries the gait; on dev only latent does
    data = _data(train, dev, tg_tr, tg_dev, signal={"train": ("measurement",), "dev": ("latent",)})
    res = RO.run_probes(data, tg_tr, tg_dev, log=lambda *a, **k: None)
    assert res["chosen"] == {"stride_mae_cm": "measurement", "speed_mae_m_s": "measurement",
                             "ic_median_ms": "measurement"}
    for ln in RO.LINES:
        assert res["crossfit"][ln]["measurement"] < res["crossfit"][ln]["latent"], ln
        assert res["lines"]["dev14"][ln]["variant"] == "measurement"
    # the reverse on the fitting set flips the choice, whatever dev holds
    flip = _data(train, dev, tg_tr, tg_dev, signal={"train": ("latent",), "dev": ("measurement",)})
    assert set(RO.run_probes(flip, tg_tr, tg_dev, log=lambda *a, **k: None)["chosen"].values()) == {"latent"}
    rows = res["rows"]
    assert set(rows[rows["chosen"]]["variant"]) == {"measurement"}
    assert set(rows.loc[rows["in_dev12"], "person_id"]) == set(dev[2:])
    assert RO.choose({"latent": 2.0, "measurement": 2.0}) == "latent"            # ties to the first candidate
    assert RO.choose({"latent": float("nan"), "measurement": float("inf")}) == "measurement"
    assert RO.choose({"latent": 1.0, "measurement": 0.5, "stem": 0.1}) == "measurement"  # stem is not a candidate


def test_the_fitting_set_offset_follows_g3s_rule_without_its_dev_guard(cohort):
    _, _, tg_tr, _ = cohort
    ref = g3.reference_ics(tg_tr, "walkway")
    pred = ref[["trial_id", "foot", "t_s"]].copy()
    pred["t_s"] += 0.04
    off, n = RO.fit_offset_on(pred, ref)
    assert off == pytest.approx(0.04) and n == len(ref)
    with pytest.raises(ValueError):
        g3.fit_offset(pred, ref)                                 # g3's own fit refuses non-dev persons
    assert np.isnan(RO.fit_offset_on(pred.assign(t_s=pred["t_s"] + 1000.0), ref)[0])   # nothing within 0.3 s


# ---------------------------------------------------------------- tokens
def _imu_windows(n=2):
    import model_fixture as MF
    out = []
    for k in range(n):
        w = MF.make_window(2.0 + 0.7 * k, seed=k)
        out.append(replace(w, streams=[s for s in w.streams if s.modality == "imu"]))
    return out


def test_encode_windows_reads_the_same_tokens_from_a_body_and_a_g12_model_of_one_seed():
    import torch

    import probe_frozen_tokens as PF
    from mdfm.baselines import config as GC
    from mdfm.baselines import model as BM
    from mdfm.model import mfm as M
    tiny = replace(M.TINY, skeleton=False)
    torch.manual_seed(3)
    body = M.MFM(tiny).eval()
    torch.manual_seed(3)
    base = BM.G12Model.of("mfr", tiny, GC.HeadSection(width=32, depth=1, heads=2)).eval()
    wins = _imu_windows()
    a = RO.encode_windows(body, wins, batch_windows=1)
    b = RO.encode_windows(base, wins, batch_windows=2)
    j0 = PF.encode_windows(body, wins, batch_windows=1)
    for w, fa, fb, fj in zip(wins, a, b, j0):
        P_ = int(np.ceil(w.duration / 0.25 - 1e-9))
        assert set(fa) == {"latent", "measurement", "stem", "n_measurement", "subject"}
        assert fa["latent"].shape == (P_, tiny.d) and fa["subject"].shape == (tiny.d,)
        for v in ("latent", "measurement", "stem", "subject", "n_measurement"):
            assert np.allclose(fa[v], fb[v], atol=1e-5, equal_nan=True), v
            assert np.allclose(fa[v], fj[v], atol=1e-6, equal_nan=True), v


def test_collect_pools_each_pass_and_keeps_three_token_variants(monkeypatch):
    from mdfm.eval import infer as I
    persons = ["weargait:HC001", "weargait:WHC002"]
    tg = _targets(persons, [(1.0, 1.8), (1.3, 2.0)])
    units = P.pass_units(tg)

    class Win:
        def __init__(self, t0):
            self.t0, self.duration, self.cov = t0, 10.0, {}

    monkeypatch.setattr(I, "record_windows", lambda rec: ([Win(t) for t in (0.0, 5.0, 10.0, 15.0, 20.0)], None, 100.0))

    def encode(wins):
        out = []
        for w in wins:
            c = w.t0 + (np.arange(40) + 0.5) * 0.25
            f = np.stack([c, np.ones_like(c)], axis=1).astype(np.float32)
            out.append({"latent": f, "measurement": 2 * f, "stem": 3 * f, "subject": np.full(2, w.t0, np.float32)})
        return out
    recs = [(t, {"lockbox": False, "person_id": p}) for t, p in zip(tg.trials["trial_id"], tg.trials["person_id"])]
    out = RO.collect(encode, recs, units, tg, persons, log=lambda *a, **k: None)
    grid = (np.arange(120) + 0.5) * 0.25
    for i, u in units.iterrows():
        inside = grid[(grid >= u["t0"] - 1e-9) & (grid <= u["t1"] + 1e-9)]
        assert out["n_rows"][i] == len(inside)
        assert np.allclose(out["X"]["latent"][i], [inside.mean(), 1.0, inside.std(), 0.0], atol=1e-4)
        assert np.allclose(out["X"]["stem"][i], 3 * np.array([inside.mean(), 1.0, inside.std(), 0.0]), atol=1e-3)
        assert out["X"]["subject"][i].shape == (2,)
    tok = out["tokens"]
    assert set(tok) == {"trial_id", "person_id", "start", "latent", "measurement", "stem"}
    assert all(tok[v].dtype == np.float16 for v in RO.IC_VARIANTS) and len(tok["start"]) == out["counts"]["token_rows_kept"]
    with pytest.raises(AssertionError):
        RO.collect(encode, [("t", {"lockbox": True, "person_id": persons[0]})], units, tg, persons)


def test_a_checkpoint_is_loaded_once_by_its_kind(tmp_path, monkeypatch):
    from mdfm.baselines import load as BL
    from mdfm.eval import infer as I
    base, body, bare = tmp_path / "g12_mfr_s1", tmp_path / "pilot", tmp_path / "bare"
    for d in (base, body, bare):
        d.mkdir()
    (base / "config.json").write_text(json.dumps({"g12": {}, "run": {}}), encoding="utf-8")
    (body / "config.json").write_text(json.dumps({"seed": 1, "model": {}}), encoding="utf-8")
    assert RO.kind_of(base / "ckpt_B_000001.pt") == "baseline" and RO.kind_of(body / "ckpt_B_000001.pt") == "body"
    assert RO.kind_of(bare / "ckpt_B_000001.pt") is None
    calls = []

    def frozen(path, weights="ema", device="cpu"):
        calls.append("frozen")
        raise ValueError(f"{path} is not a G12 baseline checkpoint (its configuration has no g12 section)")

    class Loaded:
        config_json = json.dumps({"seed": 7})
        model = None

        def describe(self):
            return {"path": "p", "sha256": "s"}

        def trained_persons(self):
            return frozenset({"a"})
    monkeypatch.setattr(BL, "load_frozen", frozen)
    monkeypatch.setattr(I, "load_checkpoint", lambda *a, **k: calls.append("body") or Loaded())
    from mdfm.train import config as CF
    monkeypatch.setattr(CF, "from_json", lambda s: type("C", (), {"seed": json.loads(s)["seed"]})())
    kind, _, info = RO.load_any(body / "ckpt_B_000001.pt")
    assert kind == "body" and calls == ["body"] and info["seed"] == 7 and info["trained_persons"] == {"a"}
    calls.clear()
    assert RO.load_any(bare / "ckpt_B_000001.pt")[0] == "body" and calls == ["frozen", "body"]
    calls.clear()
    with pytest.raises(ValueError):
        RO.load_any(base / "ckpt_B_000001.pt")               # a baseline folder never falls back to the body loader
    assert calls == ["frozen"]


def test_the_training_side_imports_no_reading_side():
    import ast
    for m in ("config", "data", "model", "compute", "trainer", "load"):
        with open(os.path.join(TREE, "mdfm", "baselines", f"{m}.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.add(f"{'.' * node.level}{node.module or ''}")
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
        bad = {n for n in names if "gates" in n or n in ("readouts", "comparison") or n.endswith((".readouts",
                                                                                                    ".comparison"))}
        assert not bad, (m, bad)


# ---------------------------------------------------------------- the readout script
def _finished_log(run, sha, verify_path):
    """A FINISHED G12 run's log.jsonl (scripts/read_g12.read_train): its compute record and its verified checkpoint."""
    recs = [{"event": "start", "g12": {"arm": "mfr", "steps": [4, 10]}},
            {"event": "log", "phase": "B", "step": 10, "loss": 1.0, "compute": {"wall_s": 5.0, "ratio": 0.1}},
            {"event": "compute", "finished": True, "wall_s": 50.0, "budget_wall_s": 49.0, "ratio": 50.0 / 49.0,
             "steps": 14, "tokens": 9},
            {"event": "g12_verify", "path": verify_path, "sha256": sha, "step": 10, "drift": []}]
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


@pytest.fixture
def readout_main(cohort, monkeypatch, tmp_path):
    """scripts/g12_readout.py with its data side on the synthetic cohort: g3's controls (cohort), fold sizes, lockbox
    list, targets, IMU records and model input; infer.record_windows on synthetic IMU windows; RO.load_any on a tiny
    G12 model; the MoveAtlas head fixed. Everything else (the guards, the run reading, the token pass, the probes, the
    writers) is the script's own. Returns (script, state, checkpoint): state holds what the replaced parts serve or
    saw ('lockbox', 'trained', 'lockbox_rows', 'drop_dev_pass', 'loads', 'records'); the checkpoint is a file of a run
    folder whose log reads FINISHED with this file as its verified checkpoint."""
    import torch

    from mdfm.baselines import config as GC
    from mdfm.baselines import model as BM
    from mdfm.eval import infer as I
    from mdfm.model import mfm as M
    from mdfm.train import provenance as PV
    import model_fixture as MF
    sc = _script("g12_readout")
    train, dev, tg_tr, tg_dev = cohort
    run = tmp_path / "runs" / "g12_mfr_s1"
    run.mkdir(parents=True)
    ckpt = run / "ckpt_B_000010.pt"
    ckpt.write_bytes(b"a stand-in file: load_any is replaced")
    _finished_log(run, common.sha256_file(str(ckpt)), str(ckpt).replace("\\", "/"))
    state = {"lockbox": ["weargait:HC999"], "trained": set(train[:3]), "lockbox_rows": 0, "drop_dev_pass": False,
             "loads": [], "records": []}
    monkeypatch.setattr(sc, "FOLD_SIZES", {"train": len(train), "dev": len(dev)})
    monkeypatch.setattr(g3, "lockbox_persons", lambda: sorted(state["lockbox"]))

    def load_targets(persons, **k):
        assert not k                                            # no lockbox flag, no read row
        tg = P.subset_targets(tg_tr if set(persons) <= set(train) else tg_dev, persons)
        if state["drop_dev_pass"] and set(persons) <= set(dev):
            last = tg.passes.iloc[-1]
            gone = (tg.strides["trial_id"] == last["trial_id"]) & (tg.strides["pass_id"] == last["pass_id"])
            tg = replace(tg, passes=tg.passes.iloc[:-1].reset_index(drop=True),
                         strides=tg.strides[~gone].reset_index(drop=True))
        return replace(tg, counts={**tg.counts, "lockbox_rows": state["lockbox_rows"]})

    def imu_records(persons, trials, **k):
        assert not k
        for i, (t, p) in enumerate(zip(trials["trial_id"], trials["person_id"])):
            assert p in set(persons)
            state["records"].append(p)
            yield t, {"lockbox": False, "person_id": p, "seed": i}

    def windows(rec):
        out = []
        for k, t0 in enumerate((0.0, 5.0, 10.0, 15.0, 20.0)):
            w = MF.make_window(10.0, seed=k + 7 * int(rec["seed"]), person=rec["person_id"])
            out.append(replace(w, streams=[s for s in w.streams if s.modality == "imu"], t0=t0))
        return out, None, 100.0
    torch.manual_seed(0)
    model = BM.G12Model.of("mfr", replace(M.TINY, skeleton=False), GC.HeadSection(width=32, depth=1, heads=2)).eval()

    def load_any(path, weights="ema", device="cpu"):
        state["loads"].append((str(path), weights, str(device)))
        desc = {"path": common.rel(str(path)), "name": I.checkpoint_name(path), "sha256": common.sha256_file(str(path)),
                "weights": weights, "step": 10, "phase": "B", "kind": "baseline", "arm": "mfr", "seed": 1}
        return "baseline", model, {**desc, "trained_persons": frozenset(state["trained"]), "describe": desc}
    monkeypatch.setattr(g3, "load_targets", load_targets)
    monkeypatch.setattr(g3, "imu_records", imu_records)
    monkeypatch.setattr(g3, "model_input", lambda rec: dict(rec))
    monkeypatch.setattr(I, "record_windows", windows)
    monkeypatch.setattr(PV, "moveatlas_head", lambda root=None: "e" * 40)
    monkeypatch.setattr(RO, "load_any", load_any)
    return sc, state, ckpt


def _same_value(a, b):
    return a == b or (isinstance(a, float) and a == pytest.approx(b, rel=1e-4))


def test_the_readout_script_runs_on_the_cpu_for_tiny_synthetic_inputs(readout_main, cohort, tmp_path):
    sc, state, ckpt = readout_main
    train, dev, _, _ = cohort
    out, feats = tmp_path / "out", tmp_path / "f" / "features.npz"
    assert sc.main(["--checkpoint", str(ckpt), "--device", "cpu", "--out", str(out), "--features", str(feats)]) == 0
    assert state["loads"] == [(str(ckpt), "ema", "cpu")]                  # loaded once, EMA weights
    assert sorted(set(state["records"])) == sorted(train + dev)
    for f in ("results.json", "summary.md", "per_row.csv", "per_pass.csv", "_provenance.json"):
        assert (out / f).exists(), f
    js = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert (js["kind"], js["arm"], js["seed"], js["partial"], js["representation_row"]) == ("baseline", "mfr", 1, False,
                                                                                             False)
    assert js["training"]["verdict"] == "FINISHED" and js["final_verified"] is True      # found by training_of
    assert js["sets"]["train"] == sorted(train) and js["sets"]["dev14"] == sorted(dev)
    assert js["sets"]["dev12"] == sorted(dev[2:]) and js["sets"]["train_among_checkpoint_persons"] == 3
    assert js["lockbox"] == {"persons_in_lockbox": 0, "target_lockbox_rows": 0,
                             "reader_flags": {"allow_eval": False, "allow_lockbox": False}}
    assert js["moveatlas_head"] == "e" * 40 and js["inference"]["device"] == "cpu" and not js["inference"]["amp_bf16"]
    assert js["features"]["sha256"] == common.sha256_file(str(feats)) and set(js["chosen"]) == set(RO.LINES)
    assert js["rows"]["sha256"] == common.sha256_file(str(out / "per_row.csv")) and js["rows"]["check"]["checked"] > 0
    rows = pd.read_csv(out / "per_row.csv", encoding="utf-8")
    assert list(rows.columns) == CMP.ROW_COLUMNS + ["in_dev12"] and set(rows["person_id"]) <= set(dev)
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert sc.LABEL in text and EM_DASH not in text and EN_DASH not in text
    prov = json.loads((out / "_provenance.json").read_text(encoding="utf-8"))
    assert prov["checkpoint_sha256"] == common.sha256_file(str(ckpt)) and prov["moveatlas_head"] == "e" * 40
    _, meta = RO.load_features(str(feats))
    assert meta["checkpoint"]["name"] == "g12_mfr_s1_ckpt_B_000010" and meta["kind"] == "baseline"
    # --run-dir (the FINISHED run's verified checkpoint) with --from-features: no model, the same lines
    out2 = tmp_path / "out2"
    assert sc.main(["--run-dir", str(ckpt.parent), "--from-features", "--device", "cpu", "--out", str(out2),
                    "--features", str(feats)]) == 0
    assert len(state["loads"]) == 1
    js2 = json.loads((out2 / "results.json").read_text(encoding="utf-8"))
    assert js2["chosen"] == js["chosen"] and js2["final_verified"] is True and js2["training"]["verdict"] == "FINISHED"
    for s in ("dev14", "dev12"):
        for ln in RO.LINES:
            assert _same_value(js2["lines"][s][ln]["point"], js["lines"][s][ln]["point"]), (s, ln)
    # a feature file of another checkpoint, or of other passes than the targets read now: refused
    other = tmp_path / "runs" / "g12_mfr_s2" / "ckpt_B_000010.pt"
    with pytest.raises(SystemExit, match="holds the features of"):
        sc.main(["--checkpoint", str(other), "--from-features", "--device", "cpu", "--out", str(tmp_path / "o3"),
                 "--features", str(feats)])
    state["drop_dev_pass"] = True
    with pytest.raises(SystemExit, match="passes differ"):
        sc.main(["--checkpoint", str(ckpt), "--from-features", "--device", "cpu", "--out", str(tmp_path / "o4"),
                 "--features", str(feats)])
    assert not (tmp_path / "o3").exists() and not (tmp_path / "o4").exists()
    assert sc.final_verified({"verdict": "FINISHED", "verify": {"sha256": "c" * 64}}, "a" * 64) is False
    assert sc.final_verified(None, "a" * 64) is None


def test_the_readout_script_stops_before_what_it_must_not_read(readout_main, cohort, tmp_path, monkeypatch):
    from mdfm.data.canonical import Refused
    sc, state, ckpt = readout_main
    train, dev, _, _ = cohort
    out = tmp_path / "out"
    argv = ["--checkpoint", str(ckpt), "--device", "cpu", "--out", str(out), "--features", str(tmp_path / "f.npz")]
    # a dev control among the checkpoint's training persons: refused after the load, before any record
    state["trained"] = set(train[:3]) | {dev[4]}
    with pytest.raises(Refused, match="dev controls among the checkpoint's training persons"):
        sc.main(argv)
    assert len(state["loads"]) == 1 and state["records"] == []
    state["trained"] = set(train[:3])
    # a lockbox person among the controls, a kept lockbox target row, folds of other sizes: stopped before the load
    state["lockbox"] = [dev[0]]
    with pytest.raises(SystemExit, match="a lockbox person among the non-lockbox controls"):
        sc.main(argv)
    state["lockbox"] = ["weargait:HC999"]
    state["lockbox_rows"] = 1
    with pytest.raises(SystemExit, match="a lockbox target row was kept"):
        sc.main(argv)
    state["lockbox_rows"] = 0
    monkeypatch.setitem(sc.FOLD_SIZES, "train", 54)
    with pytest.raises(SystemExit, match="folds: train 12, dev 6"):
        sc.main(argv)
    monkeypatch.setitem(sc.FOLD_SIZES, "train", len(train))
    monkeypatch.setattr(g3, "controls", lambda fold: sorted(dev) if fold == "dev" else sorted(train[:-1] + dev[:1]))
    with pytest.raises(SystemExit, match=re.escape(f"overlap ['{dev[0]}']")):
        sc.main(argv)
    # a smoke (--limit-persons) never takes the default output folder
    with pytest.raises(SystemExit, match="needs an explicit --out"):
        sc.main(["--checkpoint", str(ckpt), "--device", "cpu", "--limit-persons", "2"])
    assert len(state["loads"]) == 1 and state["records"] == [] and not out.exists()
    assert not (tmp_path / "f.npz").exists()


def test_the_readout_reads_only_a_finished_run(tmp_path):
    sc = _script("g12_readout")
    run = tmp_path / "cache" / "runs" / "g12_mfr_s1"
    run.mkdir(parents=True)
    recs = [{"event": "start", "g12": {"arm": "mfr", "steps": [10, 30]}},
            {"event": "log", "phase": "B", "step": 30, "loss": 1.0, "compute": {"wall_s": 5.0, "ratio": 0.1}}]
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    with pytest.raises(SystemExit):
        sc.resolve_run_dir(str(run), tree=str(tmp_path))
    recs += [{"event": "compute", "finished": True, "wall_s": 50.0, "budget_wall_s": 49.0, "ratio": 50.0 / 49.0,
              "steps": 40, "tokens": 9},
             {"event": "g12_verify", "path": "cache/runs/g12_mfr_s1\\ckpt_B_000030.pt", "sha256": "0" * 64,
              "step": 30, "drift": []}]
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    path, block = sc.resolve_run_dir(str(run), tree=str(tmp_path))
    assert path.replace("\\", "/").endswith("cache/runs/g12_mfr_s1/ckpt_B_000030.pt") and block["verdict"] == "FINISHED"
    assert block["compute"]["tokens"] == 9 and sc.final_verified(block, "0" * 64) is True
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs[:2] + [
        {"event": "compute_overrun", "phase": "B", "step": 20, "ratio": 1.3}]), encoding="utf-8")
    with pytest.raises(SystemExit):                              # a stopped seed is not read
        sc.resolve_run_dir(str(run), tree=str(tmp_path))


# ---------------------------------------------------------------- the compare script
def _body_report(tree, name, persons, values, ic_inf=None):
    d = tree / "reports" / "gates" / "g3" / name
    d.mkdir(parents=True)
    rows = []
    for ln, src in (("stride_mae_cm", "stride_mae_cm"), ("speed_mae_m_s", "speed_mae_m_s"),
                    ("ic_median_ms", "ic_median_ms_walkway")):
        for i, p in enumerate(persons):
            v = "inf" if (ln == "ic_median_ms" and p == ic_inf) else values[ln] + 0.1 * i
            rows.append(("dev", "model", src, p, v))
        rows += [("dev", "walkway", src, p, 0.0) for p in persons]
    pd.DataFrame(rows, columns=["row", "predictor", "line", "person_id", "value"]).to_csv(
        d / "per_person.csv", index=False, encoding="utf-8")
    pts = {src: float(np.inf) if (ln == "ic_median_ms" and ic_inf) else values[ln] + 0.1 * np.mean(range(len(persons)))
           for ln, src in (("stride_mae_cm", "stride_mae_cm"), ("speed_mae_m_s", "speed_mae_m_s"),
                           ("ic_median_ms", "ic_median_ms_walkway"))}
    g = {"checkpoint": {"path": f"cache/runs/{name}.pt", "sha256": "c" * 64, "persons_sha256": {"all": "d" * 64}},
         "moveatlas_head": "e" * 40, "sets": {"dev": persons},
         "lines": {"dev": {"model": {k: {"point": v} for k, v in pts.items()}}}}
    (d / "g3_dev.json").write_text(json.dumps(g), encoding="utf-8")
    return f"reports/gates/g3/{name}"


def _readout(tree, name, persons, kind="baseline", arm="mfr", seed=20260928, final=True, partial=False, base=9.0,
             head="e" * 40):
    d = tree / "reports" / "bench" / "g12" / "readouts" / name
    d.mkdir(parents=True)
    rows = pd.concat([_pp_rows({p: base + 0.1 * i for i, p in enumerate(persons)}, ln) for ln in CMP.LINES],
                     ignore_index=True)
    for c in CMP.ROW_COLUMNS:
        if c not in rows:
            rows[c] = np.nan
    rows["chosen"] = True
    rows["variant"] = "latent"
    rows = rows[CMP.ROW_COLUMNS]
    rows["in_dev12"] = rows["person_id"].isin(persons[2:])
    rows.to_csv(d / "per_row.csv", index=False, encoding="utf-8")
    sets = {"dev14": persons, "dev12": persons[2:]}
    lines = {s: {ln: {"point": CMP.line_point(rows[rows["line"] == ln], "mean", p)} for ln in CMP.LINES}
             for s, p in sets.items()}
    res = {"task": "G12R", "kind": kind, "arm": arm if kind == "baseline" else "body", "seed": seed, "partial": partial,
           "final_verified": final if kind == "baseline" else None, "sets": sets, "lines": lines,
           "chosen": {ln: "latent" for ln in CMP.LINES}, "checkpoint": {"path": f"cache/runs/{name}.pt",
                                                                          "sha256": "f" * 64},
           "training": {"verdict": "FINISHED", "compute": {"wall_s": 1.0, "ratio": 0.99, "steps": 3, "tokens": 4}},
           "moveatlas_head": head, "rows": {"path": f"reports/bench/g12/readouts/{name}/per_row.csv",
                    "sha256": common.sha256_file(str(d / "per_row.csv"))}}
    (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return d


def test_the_compare_script_reads_its_sources_and_says_what_is_missing(tmp_path):
    sc = _script("g12_compare")
    persons = [f"weargait:HC{i:03d}" for i in range(6)]
    sets = {"dev14": persons, "dev12": persons[2:]}
    (tmp_path / "docs" / "rules").mkdir(parents=True)
    (tmp_path / "docs" / "rules" / "G12_RULE.md").write_text("Status: DRAFT\n", encoding="utf-8")
    body = {20260928: _body_report(tmp_path, "dev_a", persons, {"stride_mae_cm": 20.0, "speed_mae_m_s": 0.2,
                                                                "ic_median_ms": 14.0}),
            20261001: _body_report(tmp_path, "dev_b", persons, {"stride_mae_cm": 21.0, "speed_mae_m_s": 0.21,
                                                                "ic_median_ms": 15.0}, ic_inf=persons[1])}
    _readout(tmp_path, "g12_mfr_s20260928_ckpt_B_000010", persons)
    _readout(tmp_path, "g12_mfr_s20261001_ckpt_B_000010", persons, seed=20261001, final=False)
    _readout(tmp_path, "g12_jepa_s20260928_smoke", persons, arm="jepa", partial=True)
    _readout(tmp_path, "phaseA_root_full_ckpt_B_032044", persons, kind="body", seed=20260928, base=7.0)
    res, table = sc.compare(tree=str(tmp_path), readouts=str(tmp_path / "reports" / "bench" / "g12" / "readouts"),
                            body_reports=body, sets=sets)
    assert res["status"] == CMP.NOT_EVALUATED and res["rule"]["registered"] is False
    miss = " | ".join(res["missing"])
    assert "not registered by the owner" in miss and "body: 2 of 3" in miss and "item 5" in miss
    assert "mfr: 1 of 3" in miss and "jepa: 0 of 3" in miss
    notes = " | ".join(res["notes"])
    assert "partial" in notes and "not the verified final checkpoint" in notes
    assert res["points"]["dev14"]["stride_mae_cm"]["mfr"][20260928] == pytest.approx(9.25)
    assert res["points"]["dev14"]["ic_median_ms"]["body"][20261001] == float("inf")
    assert res["sources"]["body"][20261001]["points_check"]["ic_median_ms"]["same"]
    assert res["representation_rows"][0]["points"]["dev12"]["stride_mae_cm"] == pytest.approx(7.0 + 0.1 * 3.5)
    assert {r[0] for r in table} == {"body", "mfr", "representation"}
    from mdfm.data import sources as SRC
    assert res["sources"]["arms"]["mfr"][20260928]["sets_sha256"]["dev14"] == SRC.person_hash(persons)
    assert res["sources"]["body"][20260928]["sets_sha256"] == {"dev14": SRC.person_hash(persons)}
    assert not any(m.startswith("MoveAtlas") or "body seed" in m for m in res["missing"])
    assert not any(n.startswith("representation") for n in res["notes"])
    out = sc.write(res, table, str(tmp_path / "out"))
    md = open(os.path.join(out, "g12.md"), encoding="utf-8").read()
    assert "## Status: NOT EVALUATED" in md and EM_DASH not in md and EN_DASH not in md and "G8: NOT EVALUATED" in md
    assert sc.BODY_DEV12_NOTE in md and "eval_g3_diag" in md
    js = json.load(open(os.path.join(out, "g12.json"), encoding="utf-8"))
    assert js["points"]["dev14"]["ic_median_ms"]["body"]["20261001"] == "inf"
    pp = pd.read_csv(os.path.join(out, "per_person.csv"), encoding="utf-8")
    assert len(pp) == len(table) and set(pp["set"]) == {"dev14", "dev12"}
    # a readout scored against another MoveAtlas commit, or a seed the body has not: listed as missing
    _readout(tmp_path, "g12_jepa_s20260928_ckpt_B_000010", persons, arm="jepa", head="a" * 40)
    _readout(tmp_path, "g12_jepa_s20261005_ckpt_B_000010", persons, arm="jepa", seed=20261005)
    _readout(tmp_path, "pilot_v1_s2_ckpt_B_032044", persons, kind="body", seed=20261001, head="b" * 40)
    res, _ = sc.compare(tree=str(tmp_path), readouts=str(tmp_path / "reports" / "bench" / "g12" / "readouts"),
                        body_reports=body, sets=sets)
    miss = " | ".join(res["missing"])
    assert "different commits" in miss and "jepa seed 20260928 aaaaaaa" in miss and "body seed 20261001 eeeeeee" in miss
    assert "jepa: seeds 20261005 have no body seed of the same value" in miss
    assert any(n.startswith("representation seed 20261001") and "bbbbbbb" in n for n in res["notes"])
    for d in ("g12_jepa_s20260928_ckpt_B_000010", "g12_jepa_s20261005_ckpt_B_000010", "pilot_v1_s2_ckpt_B_032044"):
        for f in ("results.json", "per_row.csv"):
            os.remove(tmp_path / "reports" / "bench" / "g12" / "readouts" / d / f)
        os.rmdir(tmp_path / "reports" / "bench" / "g12" / "readouts" / d)
    # two readouts of one arm and seed: neither is read
    _readout(tmp_path, "g12_mfr_s20260928_again", persons)
    res, _ = sc.compare(tree=str(tmp_path), readouts=str(tmp_path / "reports" / "bench" / "g12" / "readouts"),
                        body_reports=body, sets=sets)
    assert "mfr seed 20260928: 2 readouts" in " | ".join(res["notes"]) and "mfr: 0 of 3" in " | ".join(res["missing"])
    # a body report whose rows do not give its own point is refused
    bad = tmp_path / "reports" / "gates" / "g3" / "dev_a" / "g3_dev.json"
    g = json.loads(bad.read_text(encoding="utf-8"))
    g["lines"]["dev"]["model"]["stride_mae_cm"]["point"] = 1.0
    bad.write_text(json.dumps(g), encoding="utf-8")
    with pytest.raises(SystemExit):
        sc.body_rows(body[20260928], str(tmp_path), persons)
    # a per_row.csv changed after its results.json: refused
    p = tmp_path / "reports" / "bench" / "g12" / "readouts" / "g12_mfr_s20260928_ckpt_B_000010" / "per_row.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        sc.readout_rows(str(p.parent / "results.json"), str(tmp_path))
