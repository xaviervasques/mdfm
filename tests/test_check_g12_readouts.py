"""Tests of scripts/check_g12_readouts.py (task G12IC) on synthetic data only (the synthetic walkway cohort and fake
frozen tokens of tests/test_g12_readouts.py, the readout computed by mdfm/baselines/readouts.run_probes): no weargait
file, record or checkpoint is read.

- a clean readout passes every check (sets, feature file, folds, targets, units, one prediction per pass, the refit
  from the fitting set alone, the lines from the rows, the comparison's points, the checkpoint's persons);
- each defect the checks look for is caught: a dev or lockbox person among the fitting rows, a dev person among the
  checkpoint's training persons, a target that is not the harness's, a stride value in metres, two predictions in one
  pass, a prediction no fit on the fitting set gives, a wrong dev12 flag, a comparison point that is not the readout's;
- the IC diagnostic: the AUC (ties, perfect, reversed), the neighbouring rows, a variant with contact signal against
  one without, and the line at the rule's threshold equal to the readout's own IC line;
- the span reference and the written report (no em-dash, every check listed).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm.baselines import readouts as RO  # noqa: E402
from mdfm.eval import probes as P  # noqa: E402
from mdfm.gates import g3  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CK = _load(os.path.join(TREE, "scripts", "check_g12_readouts.py"), "check_g12_readouts_test")
TR = _load(os.path.join(TREE, "tests", "test_g12_readouts.py"), "test_g12_readouts_helpers_for_checks")
QUIET = dict(log=lambda *a, **k: None)


@pytest.fixture
def cohort(monkeypatch):
    train, dev, tg_tr, tg_dev = TR._cohort()
    train, dev = sorted(train), sorted(dev)
    monkeypatch.setattr(g3, "controls", lambda fold: list(dev) if fold == "dev" else list(train))
    monkeypatch.setattr(g3, "DEV_SETS", {"dev14": (), "dev12": (dev[0], dev[1])})
    return train, dev, tg_tr, tg_dev


def _readout(cohort, signal=None):
    """(data, res, rows, per_pass, comp, body_n, dev12) of a synthetic readout, rows and per_pass as per_row.csv and
    per_pass.csv read back."""
    train, dev, tg_tr, tg_dev = cohort
    data = TR._data(train, dev, tg_tr, tg_dev, signal=signal)
    r0 = RO.run_probes(data, tg_tr, tg_dev, **QUIET)
    rows = r0["rows"].copy()
    rows["pass_id"] = rows["pass_id"].astype(str)
    per_pass = r0["per_pass"].copy()
    per_pass["pass_id"] = per_pass["pass_id"].astype(str)
    res = {"kind": "baseline", "arm": "mfr", "seed": 20260928, "variants": r0["variants"], "lines": r0["lines"],
           "sets": {"train": list(train), "dev14": list(dev), "dev12": r0["sets"]["dev12"]}}
    comp = {"points": {s: {ln: {"mfr": {"20260928": r0["lines"][s][ln]["point"]}} for ln in RO.LINES}
                       for s in ("dev14", "dev12")}}
    body_n = {"synthetic": {"stride": len(tg_dev.strides), "speed": len(tg_dev.passes), "ic": len(tg_dev.walkway_ic)}}
    return data, res, rows, per_pass, comp, body_n, r0["sets"]["dev12"]


def _all_checks(cohort, data, res, rows, per_pass, comp, body_n, dev12, trained=None, locked=()):
    train, dev, tg_tr, tg_dev = cohort
    out = []
    u_tr = data["units"]["train"]
    CK.check_results_sets(out, "x", res, train, dev, dev12)
    CK.check_features(out, "x", data, train, dev, locked)
    CK.check_checkpoint(out, "x", set(train) if trained is None else trained, train, dev, locked)
    CK.check_folds(out, "x", u_tr, dev)
    CK.check_rows(out, "x", rows, per_pass, tg_dev, body_n, RO.VARIANTS)
    CK.check_refit(out, "x", data, u_tr, per_pass, RO.VARIANTS)
    CK.check_lines(out, "x", rows, res, dev, dev12, RO.VARIANTS)
    CK.check_compare(out, "x", res, comp)
    return out


def _failed(out):
    return sorted({c["id"] for c in out if not c["ok"]})


# ---------------------------------------------------------------- the checks
def test_a_clean_readout_passes_every_check(cohort):
    r = _readout(cohort)
    out = _all_checks(cohort, *r)
    assert out and _failed(out) == []
    ids = {c["id"] for c in out}
    assert ids == {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    g = []
    train, dev, _, _ = cohort
    CK.check_global_sets(g, train, dev, r[-1], set(), {"report": list(dev)}, (dev[0], dev[1]), (12, 6))
    assert _failed(g) == [], [c for c in g if not c["ok"]]
    CK.check_global_sets(g, train, dev, r[-1], {train[0]}, {"report": list(dev[1:])}, (dev[0], dev[1]),
                         (12, 6))
    assert _failed(g) == ["1"]


def test_each_defect_is_caught(cohort):
    train, dev, tg_tr, tg_dev = cohort
    data, res, rows, per_pass, comp, body_n, dev12 = _readout(cohort)
    base = (data, res, rows, per_pass, comp, body_n, dev12)

    def failed(**kw):
        args = dict(zip(("data", "res", "rows", "per_pass", "comp", "body_n", "dev12"), base))
        extra = {k: kw.pop(k) for k in ("trained", "locked") if k in kw}
        args.update(kw)
        return _failed(_all_checks(cohort, *args.values(), **extra))

    # a dev person among the fitting units or token rows
    leak = {**data, "units": {**data["units"], "train": data["units"]["train"].assign(
        person_id=[dev[0]] + list(data["units"]["train"]["person_id"][1:]))}}
    assert "2" in failed(data=leak)
    tok = dict(data["train"]["tokens"])
    tok["person_id"] = np.array([dev[1]] + list(tok["person_id"][1:]))
    assert failed(data={**data, "train": {**data["train"], "tokens": tok}}) == ["2"]
    # a lockbox person among the fitting persons; a dev person among the checkpoint's training persons
    assert failed(locked={train[3]}) == ["2", "3"]          # a lockbox person trained on, too
    assert failed(trained=set(train) | {dev[2]}) == ["3"]
    # a stride target that is not the harness's
    bad = rows.copy()
    i = bad.index[(bad["line"] == "stride_mae_cm") & (bad["variant"] == "latent")][0]
    bad.loc[i, "target"] += 0.01
    assert "5" in failed(rows=bad)
    # a stride value in metres instead of cm
    bad = rows.copy()
    m = bad["line"] == "stride_mae_cm"
    bad.loc[m, "value"] = bad.loc[m, "value"] / 100.0
    assert "6" in failed(rows=bad)
    # two stride predictions in one pass
    bad = rows.copy()
    i = bad.index[(bad["line"] == "stride_mae_cm") & (bad["variant"] == "stem")][0]
    bad.loc[i, "prediction"] += 0.05
    bad.loc[i, "value"] = 100.0 * abs(bad.loc[i, "prediction"] - bad.loc[i, "target"])
    assert "7" in failed(rows=bad)
    # a pass prediction that no fit on the fitting set gives (as a dev target entering a fit would)
    pp = per_pass.copy()
    pp.loc[0, "pred_speed_m_s_subject"] += 0.01
    assert "8" in failed(per_pass=pp)
    # a wrong dev12 flag
    bad = rows.copy()
    bad.loc[bad["person_id"] == dev12[0], "in_dev12"] = False
    assert "9" in failed(rows=bad)
    # a comparison point that is not the readout's (IC included, inf against a number)
    c2 = {"points": {s: {ln: dict(v) for ln, v in d.items()} for s, d in comp["points"].items()}}
    c2["points"]["dev12"]["ic_median_ms"] = {"mfr": {"20260928": "inf"}}
    if np.isfinite(float(res["lines"]["dev12"]["ic_median_ms"]["point"])):
        assert failed(comp=c2) == ["9"]
    c2["points"]["dev12"]["ic_median_ms"] = {"mfr": {"20260928": res["lines"]["dev12"]["ic_median_ms"]["point"]}}
    c2["points"]["dev14"]["stride_mae_cm"] = {"mfr": {"20260928": float(res["lines"]["dev14"]["stride_mae_cm"]["point"])
                                                      + 0.01}}
    assert failed(comp=c2) == ["9"]


# ---------------------------------------------------------------- diagnostics
def test_auc_and_neighbouring_rows():
    assert CK.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert CK.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert CK.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert CK.auc([0.1, np.nan, 0.9], [0, 1, 1]) == 1.0
    assert np.isnan(CK.auc([0.1, 0.2], [1, 1]))
    X = np.arange(8, dtype=float).reshape(4, 2)
    C = CK.context_rows(X, [0.0, 0.25, 0.5, 0.0], ["a", "a", "a", "b"])
    assert C.shape == (4, 6)
    assert np.isnan(C[0, :2]).all() and (C[0, 2:] == [0, 1, 2, 3]).all()
    assert (C[1] == [0, 1, 2, 3, 4, 5]).all()
    assert np.isnan(C[2, 4:]).all() and np.isnan(C[3, [0, 1, 4, 5]]).all() and (C[3, 2:4] == [6, 7]).all()


def test_the_ic_diagnostic_tells_contact_signal_from_none_and_matches_the_readout(cohort):
    train, dev, tg_tr, tg_dev = cohort
    sig = {"train": ("latent", "stem", "subject"), "dev": ("latent", "stem", "subject")}
    data, res, *_ = _readout(cohort, signal=sig)
    u_tr, u_dev = data["units"]["train"], data["units"]["dev"]
    d = CK.ic_diagnostic(data, tg_tr, tg_dev, u_tr, u_dev, thresholds=(0.3, 0.5), context=True)
    lat, mea = d["variants"]["latent"], d["variants"]["measurement"]
    for foot in P.FEET:
        assert lat["patches"][foot]["auc"] > 0.95 and lat["patches"][foot]["hit"] > 0.8
        assert 0.3 < mea["patches"][foot]["auc"] < 0.7
    assert [t["threshold"] for t in lat["thresholds"]] == [0.3, 0.5]
    for v in RO.IC_VARIANTS:                      # the line at the rule's threshold is the readout's own IC line
        for s in ("dev14", "dev12"):
            got = d["variants"][v]["thresholds"][-1][s]["point"]
            ref = res["variants"]["ic"]["variants"][v]["dev"][s]["point"]
            assert CK._same(got, ref), (v, s, got, ref)
    assert lat["context"]["patches"]["L"]["auc"] > 0.95 and lat["context"]["line"]["threshold"] == P.IC_THRESHOLD
    assert d["labelled"]["train"] > 0 and d["labelled"]["dev"] > 0


def test_the_span_reference_and_the_written_report(cohort, tmp_path):
    train, dev, tg_tr, tg_dev = cohort
    data, res, rows, per_pass, comp, body_n, dev12 = _readout(cohort)
    span = CK.span_reference(data["units"]["train"], data["units"]["dev"], tg_dev)
    assert set(span) == {"dev14", "dev12"} and all(np.isfinite(list(v.values())).all() for v in span.values())
    checks = _all_checks(cohort, data, res, rows, per_pass, comp, body_n, dev12)
    ic = CK.ic_diagnostic(data, tg_tr, tg_dev, data["units"]["train"], data["units"]["dev"], context=False)
    out = {"what": CK.LABEL, "task": CK.TASK, "time": "2026-10-13T00:00:00+02:00", "script_sha256": "0" * 64,
           "moveatlas_head": "", "thresholds": list(CK.THRESHOLDS), "checks": checks,
           "n_pass": sum(c["ok"] for c in checks), "n_fail": sum(not c["ok"] for c in checks), "span_reference": span,
           "ic": {"x": ic}, "readouts": {"x": {"kind": "baseline", "arm": "mfr", "seed": 1, "moveatlas_head": "",
                                              "results_sha256": "a", "per_row_sha256": "b", "per_pass_sha256": "c",
                                              "features": "f", "features_sha256": "d"}},
           "body_reports": {"r": {"seed": 1, "g3_dev_sha256": "e", "per_person_sha256": "g"}},
           "compare": {"path": "p", "sha256": "h"}}
    CK.write(out, str(tmp_path))
    md = (tmp_path / "checks.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and chr(0x2013) not in md
    assert md.count("| PASS |") == len(checks) and "| FAIL |" not in md
    assert "inf" in md or "n/a" in md or "ms" in md
    js = (tmp_path / "checks.json").read_text(encoding="utf-8")
    assert '"n_fail": 0' in js
