"""Tests of the G8 dev harness, weargait_pd line (task G8R; mdfm/gates/g8.py, scripts/eval_g8_dev.py), on synthetic data
only: no clinical table, record, target or checkpoint of the tree is read, except the lockbox split file
(reports/lockbox/lockbox_persons.csv, through mdfm.gates.g3, which checks its hash) for the refusal tests.

- qwk is exact: equal to Cohen's weighted kappa computed from the full 5 x 5 confusion matrix with quadratic weights
  (random grades, every category count), 1 for perfect agreement, exactly 0 for a constant prediction, NaN when every
  grade and prediction is one value, a hand-computed case; grades outside 0 to 4 or not integers raise;
- round_clip rounds half up and clips to the scale, NaN kept; the majority grade ties to the lower grade;
- the bootstrap: the closed form equals the general path (mdfm.baselines.comparison.bootstrap_rows with
  g8.row_statistic on the drawn rows, each slot its own person) resample by resample; a person drawn twice counts twice;
  bootstrap_difference equals the general path's difference; undefined items are counted;
- the folds never mix persons: every ridge fit of run_readout and crossfit_readout (spied) holds no scored person, a
  fitting set overlapping the scored set raises, the pool rows give each person one prediction per item from a fit
  without it, and scored features turned upside down change no fit; the variant choice reads the fitting set only;
- the qmap cut points give the fitting set's grade shares; a person without features gets the majority grade, flagged,
  and counted once as a filled person and once per item as a filled row;
- refusals: a lockbox person, a CODY id and another source's id are refused by guard, read_items, record_persons,
  pd_folds and person_features before any file or index opens; a lockbox id with the ' (control)' suffix is refused
  the same way (guard compares bare ids on both sides); read_items skips other rows, strips ' (control)', reads '-' as
  no grade, raises on a grade outside the scale and on a person found twice;
- the script's pool rows of a token run hold both candidate variants, the one chosen on pd_train flagged;
- gate_result: pass at 0.50, kill below 0.40, undetermined between, higher is better; the script's lines_of and
  attainability read rows and folders built here.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/test_g8.py
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from mdfm import paths
from mdfm.baselines import comparison as CMP
from mdfm.data.canonical import Refused
from mdfm.eval import probes
from mdfm.gates import g8

TREE = paths.TREE
SCRIPT = os.path.join(TREE, "scripts", "eval_g8_dev.py")
HAVE_LOCKBOX = os.path.exists(g8.LOCKBOX_CSV)
needs_lockbox = pytest.mark.skipif(not HAVE_LOCKBOX, reason="reports/lockbox not built")


def _script():
    spec = importlib.util.spec_from_file_location("eval_g8_dev_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def qwk_matrix(y, p, k=5):
    """Cohen 1968 weighted kappa from the k x k confusion matrix, quadratic weights (i - j)^2 / (k - 1)^2."""
    O = np.zeros((k, k))
    for a, b in zip(y.astype(int), p.astype(int)):
        O[a, b] += 1
    n = O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / n
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    return 1.0 - (W * O).sum() / (W * E).sum()


# ---------------------------------------------------------------- qwk
def test_qwk_equals_the_confusion_matrix_formula():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(3, 40))
        y = rng.integers(0, 5, n).astype(float)
        p = np.clip(y + rng.integers(-2, 3, n), 0, 4).astype(float)
        if len(np.unique(y)) == 1 and len(np.unique(p)) == 1:
            continue
        assert g8.qwk(y, p) == pytest.approx(qwk_matrix(y, p), abs=1e-12)


def test_qwk_special_cases():
    y = np.array([0, 1, 2, 3, 4, 1, 1], float)
    assert g8.qwk(y, y) == pytest.approx(1.0)
    for c in range(5):
        assert g8.qwk(y, np.full(len(y), c)) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(g8.qwk(np.ones(5), np.ones(5)))
    assert g8.qwk(np.ones(5), np.array([1, 1, 2, 1, 1.0])) == pytest.approx(0.0, abs=1e-12)
    # hand case: y = (0, 1, 2), p = (0, 2, 2): O = 1, E = (1/3) sum_a sum_c (y_a - p_c)^2 = (8 + 3 + 4) / 3 = 5
    assert g8.qwk(np.array([0, 1, 2.]), np.array([0, 2, 2.])) == pytest.approx(0.8)
    assert np.isnan(g8.qwk(np.zeros(0), np.zeros(0)))
    for bad in (np.array([0, 5.]), np.array([0, 1.5]), np.array([-1, 0.]), np.array([0, np.nan])):
        with pytest.raises(ValueError):
            g8.qwk(np.array([0, 1.]), bad)
    with pytest.raises(ValueError):
        g8.qwk(np.array([0, 1, 2.]), np.array([0, 1.]))


def test_round_clip_and_majority():
    x = np.array([0.49, 0.5, 1.5, 2.5, -0.7, -0.4, 4.6, 3.49, np.nan, np.inf])
    out = g8.round_clip(x)
    assert out[:8].tolist() == [0, 1, 2, 3, 0, 0, 4, 3]
    assert np.isnan(out[8]) and np.isnan(out[9])
    assert g8.majority([1, 1, 2, 2, 0, np.nan]) == 1.0
    assert g8.majority([3, 3, 0]) == 3.0
    assert np.isnan(g8.majority([np.nan]))


def _rows(persons, rng, items=g8.ITEMS, model="m", missing=None):
    parts = []
    for it in items:
        ps = [p for p in persons if not (missing and (p, it) in missing)]
        y = rng.integers(0, 4, len(ps)).astype(float)
        p = np.clip(y + rng.integers(-1, 2, len(ps)), 0, 4).astype(float)
        parts.append(g8.make_rows(model, "s", "round", "latent", it, ps, ["PD"] * len(ps), y, p + 0.1, p,
                                  np.zeros(len(ps), bool)))
    return pd.concat(parts, ignore_index=True)


def test_mean_qwk_is_the_mean_of_defined_items():
    rng = np.random.default_rng(2)
    persons = [f"weargait:P{i:03d}" for i in range(20)]
    r = _rows(persons, rng)
    m, by = g8.mean_qwk(r)
    assert m == pytest.approx(np.mean([by[i] for i in g8.ITEMS]))
    r2 = r.copy()
    r2.loc[r2["item"] == "3.11", ["target", "prediction"]] = 0.0          # undefined item leaves the mean
    m2, by2 = g8.mean_qwk(r2)
    assert np.isnan(by2["3.11"]) and m2 == pytest.approx(np.mean([by2[i] for i in ("3.10", "3.12", "3.13")]))


# ---------------------------------------------------------------- bootstrap
def test_closed_form_bootstrap_equals_the_general_path():
    rng = np.random.default_rng(3)
    persons = sorted(f"weargait:P{i:03d}" for i in range(12))
    r = _rows(persons, rng, missing={(persons[0], "3.12"), (persons[3], "3.10")})
    idx = g8.draws(len(persons), 60, 7)
    fast, undefined, _ = g8.resample_qwk(r, persons, idx)
    for b in range(idx.shape[0]):
        slow = g8.row_statistic(CMP.resample_rows(r, persons, idx[b]))
        if np.isnan(slow):
            assert np.isnan(fast[b])
        else:
            assert fast[b] == pytest.approx(slow, abs=1e-12)
    # a person drawn twice counts twice: the same as two persons holding that person's rows
    sub = persons[1:6]
    one = r[r["person_id"] == sub[0]]
    copies = pd.concat([r[r["person_id"].isin(sub)], one.assign(person_id="weargait:COPY")], ignore_index=True)
    m, _, _ = g8.resample_qwk(r[r["person_id"].isin(sub)], sub, np.array([[0, 0, 1, 2, 3, 4]]))
    assert m[0] == pytest.approx(g8.row_statistic(copies), abs=1e-12)
    assert m[0] != pytest.approx(g8.row_statistic(r[r["person_id"].isin(sub)]), abs=1e-9)


def test_bootstrap_difference_equals_the_general_path():
    rng = np.random.default_rng(4)
    persons = sorted(f"weargait:P{i:03d}" for i in range(10))
    body = [_rows(persons, rng, model="body") for _ in range(2)]
    comp = [_rows(persons, rng, model="mfr") for _ in range(3)]
    fast = g8.bootstrap_difference(body, comp, persons, n_boot=40, seed=11)
    slow = CMP.bootstrap_rows(body, comp, persons, g8.row_statistic, n_boot=40, seed=11)
    ok = np.isfinite(slow["diffs"])
    assert np.allclose(fast["diffs"][ok], slow["diffs"][ok], atol=1e-12)
    assert (np.isnan(fast["diffs"]) == ~ok).all()


def test_bootstrap_counts_undefined_items():
    persons = [f"weargait:P{i}" for i in range(6)]
    y = np.array([0, 0, 0, 0, 0, 1.])
    r = pd.concat([g8.make_rows("m", "s", "round", "", it, persons, ["PD"] * 6, y, y, y, np.zeros(6, bool))
                   for it in g8.ITEMS], ignore_index=True)
    b = g8.bootstrap_qwk(r, n_boot=500, seed=1)
    assert b["resamples_with_undefined_item"] > 0           # resamples without the graded person
    line = g8.line_of(r)
    assert line["point"] == pytest.approx(1.0) and line["n_persons"] == 6


# ---------------------------------------------------------------- probes: folds never mix persons
def _features(n=40, d=6, seed=5):
    rng = np.random.default_rng(seed)
    persons = np.array([f"weargait:N{i:03d}" for i in range(n)])
    X = rng.normal(size=(n, d))
    items = {it: np.clip(np.round(1.2 + X[:, k] + 0.3 * rng.normal(size=n)), 0, 4) for k, it in enumerate(g8.ITEMS)}
    items["3.11"][3] = np.nan
    return persons, X, items, np.array(["PD"] * n)


def _spy(monkeypatch):
    seen = []
    real = probes.ridge_path

    def spy(X, Y, alphas=probes.ALPHAS):
        seen.append(np.asarray(X).copy())
        return real(X, Y, alphas)
    monkeypatch.setattr(probes, "ridge_path", spy)
    return seen


def _rows_of(seen_X, X, persons):
    out = []
    for F in seen_X:
        hit = set()
        for row in F:
            j = np.flatnonzero(np.all(np.isclose(X, row[None]), axis=1))
            hit |= set(persons[j])
        out.append(hit)
    return out


def test_run_readout_fits_hold_no_scored_person(monkeypatch):
    persons, X, items, groups = _features()
    fit, score = set(persons[:30]), set(persons[30:])
    seen = _spy(monkeypatch)
    rows, diag = g8.run_readout(X, persons, groups, items, fit, score, "m", "pd_dev", "latent")
    assert seen and all(not (s & score) for s in _rows_of(seen, X, persons))
    assert set(rows["person_id"]) <= score and set(rows["readout"]) == {"round", "qmap"}
    assert rows["prediction"].between(0, 4).all() and (rows["prediction"] == rows["prediction"].round()).all()
    with pytest.raises(ValueError):
        g8.run_readout(X, persons, groups, items, fit | {persons[31]}, score, "m", "pd_dev", "latent")


def test_scored_features_change_no_fit():
    persons, X, items, groups = _features()
    fit, score = set(persons[:30]), set(persons[30:])
    _, d1 = g8.run_readout(X, persons, groups, items, fit, score, "m", "pd_dev", "latent")
    X2 = X.copy()
    X2[30:] = -X2[30:] * 7.0
    _, d2 = g8.run_readout(X2, persons, groups, items, fit, score, "m", "pd_dev", "latent")
    for it in g8.ITEMS:
        assert d1[it]["alpha"] == d2[it]["alpha"] and d1[it]["cuts"] == d2[it]["cuts"]
        pd.testing.assert_frame_equal(d1[it]["cf_rows"], d2[it]["cf_rows"])
    best1, s1 = g8.choose_variant({"latent": d1, "measurement": d1})
    best2, s2 = g8.choose_variant({"latent": d2, "measurement": d2})
    assert best1 == best2 == "latent" and s1 == s2


def test_pool_rows_give_each_person_one_prediction_from_a_fit_without_it(monkeypatch):
    persons, X, items, groups = _features(n=45)
    seen = _spy(monkeypatch)
    rows = g8.crossfit_readout(X, persons, groups, items, set(persons), "m", "pd_pool", "latent")
    r = rows[rows["readout"] == "round"]
    assert not r.duplicated(["person_id", "item"]).any()
    assert len(r) == sum(np.isfinite(items[i]).sum() for i in g8.ITEMS)
    folds = probes.person_folds(np.array(sorted(persons)), g8.CROSS_FOLDS, probes.SEED)
    held = [{p for p, k in zip(sorted(persons), folds) if k == f} for f in range(g8.CROSS_FOLDS)]
    hits = _rows_of(seen, X, persons)
    assert hits and all(any(not (h & held[f]) for f in range(g8.CROSS_FOLDS)) for h in hits)
    # and each outer fold's scored persons were read by fits that held none of them
    for f in range(g8.CROSS_FOLDS):
        assert any(h and not (h & held[f]) for h in hits)


def test_choose_variant_and_qmap():
    persons, X, items, groups = _features()
    fit, score = set(persons[:30]), set(persons[30:])
    _, dl = g8.run_readout(X, persons, groups, items, fit, score, "m", "s", "latent")
    rng = np.random.default_rng(9)
    _, dn = g8.run_readout(rng.normal(size=X.shape), persons, groups, items, fit, score, "m", "s", "measurement")
    best, scores = g8.choose_variant({"latent": dl, "measurement": dn})
    assert best == ("latent" if scores["latent"] >= scores["measurement"] else "measurement")
    y = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4, 4.])
    raw = np.linspace(-1, 5, 10)
    cuts = g8.qmap_cuts(raw, y)
    got = g8.qmap_apply(raw, cuts)
    assert np.bincount(got.astype(int), minlength=5).tolist() == np.bincount(y.astype(int), minlength=5).tolist()


def test_a_person_without_features_gets_the_majority_grade():
    persons, X, items, groups = _features()
    X = X.copy()
    X[35] = np.nan
    rows, _ = g8.run_readout(X, persons, groups, items, set(persons[:30]), set(persons[30:]), "m", "s", "latent")
    r = rows[(rows["person_id"] == persons[35]) & (rows["readout"] == "round")]
    assert r["filled"].all() and len(r)
    for it in g8.ITEMS:
        assert r.loc[r["item"] == it, "prediction"].iloc[0] == g8.majority(items[it][:30])
    # filled counts person-item rows, filled_persons the persons: one person, four rows
    line = g8.line_of(rows[rows["readout"] == "round"], n_boot=20)
    assert line["filled"] == 4 and line["filled_persons"] == 1
    assert g8.gate_result(line, "m", "s", "round", "latent").extra["filled_persons"] == 1


# ---------------------------------------------------------------- refusals
@needs_lockbox
def test_lockbox_and_other_ids_are_refused_before_any_file(monkeypatch, tmp_path):
    locked = g8.lockbox_persons()
    assert g8.lockbox_counts() == {"PD": 20, "control": 17}
    missing = str(tmp_path / "absent.csv")

    def no_index():
        raise AssertionError("the index was read")
    monkeypatch.setattr(g8.canonical, "index", no_index)
    for ids in ([locked[0]], ["weargait:NLS002", locked[-1]], ["cody_1:S01"], ["gait120:S999"]):
        with pytest.raises(Refused):
            g8.guard(ids)
        with pytest.raises(Refused):
            g8.read_items(ids, files={"PD": missing, "control": missing})
        with pytest.raises(Refused):
            g8.record_persons(ids)
        with pytest.raises(Refused):
            g8.person_features(lambda w: [], iter(()), ids)
    locked_pd = [p for p, g in g8.lockbox_rows() if g == "PD"][0]
    folds = tmp_path / "folds.csv"
    folds.write_text("source,group,person_id,fold\nweargait_pd,PD,weargait:NLS002,train\n"
                     f"weargait_pd,PD,{locked_pd},dev\n", encoding="utf-8")
    from mdfm.data import sources
    ids = {"train": ["weargait:NLS002"], "dev": [locked_pd]}
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps({"folds": {g8.FOLD_KEY: {f"sha256_{k}": sources.person_hash(v) for k, v in ids.items()}}}),
                    encoding="utf-8")
    with pytest.raises(Refused):
        g8.pd_folds(str(folds), str(summ))


@needs_lockbox
def test_a_lockbox_id_with_the_control_suffix_is_refused_before_any_file(monkeypatch, tmp_path):
    """read_items matches a table row on the bare id, so 'weargait:<lockbox id> (control)' must be refused too."""
    locked_c = [p for p, g in g8.lockbox_rows() if g == "control"][0]
    missing = str(tmp_path / "absent.csv")

    def no_index():
        raise AssertionError("the index was read")
    monkeypatch.setattr(g8.canonical, "index", no_index)
    for ids in ([locked_c + " (control)"], ["weargait:NLS002", locked_c + g8.CONTROL_SUFFIX]):
        with pytest.raises(Refused):
            g8.guard(ids)
        with pytest.raises(Refused):
            g8.read_items(ids, groups=(g8.GROUP_PD, g8.GROUP_CONTROL), files={"PD": missing, "control": missing})
        with pytest.raises(Refused):
            g8.record_persons(ids)
        with pytest.raises(Refused):
            g8.person_features(lambda w: [], iter(()), ids)


def test_guard_compares_bare_ids_on_both_sides(monkeypatch):
    """A split file id with the suffix refuses the bare id and the suffixed one; other ids pass."""
    monkeypatch.setattr(g8, "lockbox_persons", lambda: ["weargait:HC900 (control)", "weargait:NLS950"])
    for ids in (["weargait:HC900"], ["weargait:HC900 (control)"], ["weargait:NLS950 (control)"]):
        with pytest.raises(Refused):
            g8.guard(ids)
    assert g8.guard(["weargait:HC901", "weargait:HC901 (control)"]) == {"weargait:HC901", "weargait:HC901 (control)"}
    assert g8.bare_id("weargait:HC900 (control)") == "weargait:HC900" == g8.bare_id(" weargait:HC900 ")


@needs_lockbox
def test_read_items_reads_only_the_requested_rows(tmp_path):
    head = ["Subject ID", "x"] + list(g8.ITEM_COLUMNS.values()) + list(g8.PART2.values()) + [g8.DAYS_COLUMN, g8.MED_COLUMN]
    lines = [",".join(["-"] * len(head)), ",".join(head),
             ",".join(["NLS900", "a", "1", "0", "2", "-", "1", "0", "0", "ON"]),
             ",".join(["NLS901 (control)", "a", "0", "0", "0", "1", "", "0", "14", "Off"]),
             ",".join(["NLS902", "a", "9", "9", "9", "9", "9", "9", "x", "?"])]          # not requested: never parsed
    f = tmp_path / "pd.csv"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = g8.read_items(["weargait:NLS900", "weargait:NLS901"], files={"PD": str(f)})
    assert df["person_id"].tolist() == ["weargait:NLS900", "weargait:NLS901"]
    a = df.set_index("person_id")
    assert a.loc["weargait:NLS900", "3.12"] == 2 and np.isnan(a.loc["weargait:NLS900", "3.13"])
    assert a.loc["weargait:NLS901", "3.13"] == 1 and np.isnan(a.loc["weargait:NLS901", "2.12"])
    assert a.loc["weargait:NLS901", "days_since_part3"] == 14 and g8.med_label(a.loc["weargait:NLS901", "med_state"]) == "OFF"
    # an id that keeps the ' (control)' suffix in the records answers to the table's bare id
    s = g8.read_items(["weargait:NLS901 (control)"], files={"PD": str(f)})
    assert s["person_id"].tolist() == ["weargait:NLS901 (control)"] and s["3.13"].iloc[0] == 1
    with pytest.raises(ValueError):
        g8.read_items(["weargait:NLS900", "weargait:NLS900 (control)"], files={"PD": str(f)})
    with pytest.raises(ValueError):
        g8.read_items(["weargait:NLS902"], files={"PD": str(f)})
    f2 = tmp_path / "dup.csv"
    f2.write_text("\n".join(lines[:3] + lines[2:3]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        g8.read_items(["weargait:NLS900"], files={"PD": str(f2)})


# ---------------------------------------------------------------- verdicts and the script's tables
def test_gate_result_lines():
    persons = [f"weargait:P{i}" for i in range(8)]
    for point, want in ((0.50, "pass"), (0.62, "pass"), (0.45, "undetermined"), (0.40, "undetermined"),
                        (0.39, "kill")):
        line = {"point": point, "per_item": {}, "n_persons_by_item": {}, "filled": 0, "interval": [0.1, 0.9],
                "bootstrap": {}, "n_persons": len(persons)}
        r = g8.gate_result(line, "m", "pd_dev", "round", "latent")
        assert r.verdict == want and r.higher_is_better and r.pass_line == 0.50 and r.kill_line == 0.40


def _synthetic_sets(n_pd=40, n_c=20, seed=21):
    """A load_sets-like dict over synthetic persons (PD train and dev, controls train and dev) and their features."""
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    pd_ids = [f"weargait:NLS9{i:02d}" for i in range(n_pd)]
    c_ids = [f"weargait:WHC9{i:02d}" for i in range(n_c)]
    folds = {**{p: ("pd_dev" if i % 4 == 0 else "pd_train") for i, p in enumerate(pd_ids)},
             **{p: ("control_dev" if i % 4 == 0 else "control_train") for i, p in enumerate(c_ids)}}
    everyone = sorted(folds)
    sev = rng.normal(size=len(everyone))
    rows = []
    for p, s in zip(everyone, sev):
        g = {i: float(np.clip(np.round(1 + s + 0.4 * rng.normal()), 0, 4)) for i in g8.ITEMS}
        rows.append({"person_id": p, "group": "PD" if p in pd_ids else "control", "fold": folds[p], "scored": True,
                     "out_reason": "", **{f"item_{i}": g[i] for i in g8.ITEMS},
                     **{f"part2_{i}": float(np.clip(np.round(1 + s), 0, 4)) for i in g8.PART2},
                     "days_since_part3": 0.0, "med_state": "ON", "speed_selfpace": 1.2 - 0.1 * s,
                     "speed_hurriedpace": 1.6 - 0.1 * s})
    P = pd.DataFrame(rows)
    S = {"persons": P, "fold_info": {}, "scored": everyone, "everyone": everyone,
         "targets": SimpleNamespace(counts={"lockbox_rows": 0, "passes": 0}),
         "pd_train": sorted(p for p in pd_ids if folds[p] == "pd_train"),
         "pd_dev": sorted(p for p in pd_ids if folds[p] == "pd_dev"),
         "c_train": sorted(p for p in c_ids if folds[p] == "control_train"),
         "c_dev": sorted(p for p in c_ids if folds[p] == "control_dev")}
    X = {v: np.column_stack([sev + 0.3 * rng.normal(size=len(sev)), rng.normal(size=(len(sev), 5))])
         for v in g8.VARIANTS}
    feats = {"persons": everyone, "X": X, "n_rows": np.full(len(everyone), 10), "counts": {"persons_without_rows": []}}
    return S, feats


def test_script_baselines_and_token_rows():
    S = _script()
    sets, feats = _synthetic_sets()
    rows = S.run_baselines(sets)
    assert set(rows["model"]) == {"majority", "speed", "part2", "others"}
    assert set(rows["set"]) == {"pd_dev", "all_dev", "pd_pool", "all_pool"}
    maj = rows[(rows["model"] == "majority") & (rows["set"] == "pd_dev")]
    assert g8.mean_qwk(maj)[0] == pytest.approx(0.0, abs=1e-12)
    dev = set(sets["pd_dev"])
    assert set(rows.loc[rows["set"] == "pd_dev", "person_id"]) == dev
    # the token probes: every variant on pd_dev, the chosen one flagged, the pool row with the chosen variant
    trows, choice, pools, diag = S.run_tokens(sets, feats, "m", set(), False)
    assert set(trows.loc[trows["set"] == "pd_dev", "variant"]) == set(g8.VARIANTS)
    ch = choice["pd_dev"]["chosen"]
    assert ch in g8.CANDIDATES
    assert trows.loc[(trows["set"] == "pd_dev") & trows["chosen"], "variant"].unique().tolist() == [ch]
    # the pool row for both candidates (the choice was made on persons the pool scores), the chosen one flagged
    pool = trows[trows["set"] == "pd_pool"]
    assert set(pool["variant"]) == set(g8.CANDIDATES) and pools["pd_pool"].startswith("computed")
    assert pool.loc[pool["chosen"], "variant"].unique().tolist() == [ch]
    assert not pool.duplicated(["variant", "readout", "person_id", "item"]).any()
    # a checkpoint trained on a pd_train person loses the pool row, keeps pd_dev
    trows2, _, pools2, _ = S.run_tokens(sets, feats, "m", {sets["pd_train"][0]}, False)
    assert "pd_pool" not in set(trows2["set"]) and pools2["pd_pool"].startswith("not computed")
    lines, results = S.lines_of(trows)
    assert all(r["gate"] == "G8" and r["higher_is_better"] for r in results)


@needs_lockbox
def test_script_main_without_model_writes_its_files(monkeypatch, tmp_path):
    S = _script()
    sets, _ = _synthetic_sets()
    monkeypatch.setattr(S, "load_sets", lambda with_controls=True: sets)
    monkeypatch.setattr(g8, "lockbox_index_counts", lambda index=None, tasks=g8.TASKS: {"PD": {"persons": 20,
                                                                                           "with_both_tasks": 19}})
    out = tmp_path / "dev_baselines"
    assert S.main(["--no-model", "--out", str(out)]) == 0
    assert sorted(os.listdir(out)) == ["_provenance.json", "g8_dev.json", "g8_dev.md", "per_row.csv", "persons.csv"]
    res = json.loads((out / "g8_dev.json").read_text(encoding="utf-8"))
    assert res["kind"] == "baselines" and res["lockbox"]["persons_requested_in_lockbox"] == 0
    assert res["census"]["lockbox_split_file"] == {"PD": 20, "control": 17}
    rows = pd.read_csv(out / "per_row.csv")
    one = rows[(rows["model"] == "speed") & (rows["set"] == "pd_dev") & (rows["readout"] == "round")]
    key = "speed|pd_dev|round|"
    assert res["lines"][key]["point"] == pytest.approx(g8.mean_qwk(one)[0])
    assert chr(0x2014) not in (out / "g8_dev.md").read_text(encoding="utf-8")


def test_script_lines_and_attainability(tmp_path):
    S = _script()
    rng = np.random.default_rng(12)
    persons = [f"weargait:P{i:02d}" for i in range(14)]
    rows = _rows(persons, rng, model="speed")
    rows["set"], rows["variant"] = "pd_dev", ""            # a baseline has no variant
    lines, results = S.lines_of(rows)
    key = "speed|pd_dev|round|"
    assert lines[key]["point"] == pytest.approx(g8.mean_qwk(rows)[0])
    assert results[0]["extra"]["model"] == "speed" and results[0]["gate"] == "G8"
    d = tmp_path / "dev_baselines"
    d.mkdir()
    (d / "g8_dev.json").write_text(json.dumps(S.clean({"kind": "baselines", "lines": lines, "gate_results": results,
                                                       "moveatlas_head": "abc"})), encoding="utf-8")
    att = S.attainability(str(tmp_path))
    assert len(att) == 1 and att[0]["kind"] == S.KINDS["speed"]
    assert att[0]["reaches_pass"] == (att[0]["point"] >= 0.5)
    assert att[0]["filled_rows"] == 0 and att[0]["filled_persons"] == 0
    md = S.attainability_md(att)
    assert "speed" in md and "filled persons" in md and "## Pool rows" not in md
    # a token run's pool row of the candidate not chosen goes to its own table
    other = dict(att[0], model="tok", kind="predictor (baseline tokens)", set="pd_pool", variant="latent", chosen=False)
    md2 = S.attainability_md(att + [other])
    assert "## Pool rows of the candidate variant not chosen" in md2 and "| pd_pool | tok |" in md2.split("## Pool rows")[1]
