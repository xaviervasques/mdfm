"""Tests of scripts/neural_mdd.py (weekend task NRV) on synthetic inputs; no report of the tree is read.

The lockbox guard refuses the persons file; the lockbox table parser reads sizes and groups; the
ds004998 unit count follows the two readings of OPEN-43 and leaves out interleaved subjects; the
normal and t multipliers match their closed forms; the bootstrap enumeration has C(2n - 1, n)
resamples with weights summing to 1, its weighted type-1 quantiles equal those of the explicit
n^n expansion, and its k_n approaches the normal form at large n; k_n, the absent factor and the
false pass equal direct computations on the same simulated bounds and mean what they claim; the
paired AUROC bootstrap and its lower bounds match a direct computation, its power is monotone in
delta, and the bisection brackets the target power; the AUROC-1 ceiling check and its verdict;
the NR-6.5.1 sensefog count; main() on a synthetic tree writes every output, keeps every MDD equal
to margin + k_n x sigma_d, and never opens a lockbox persons file.
"""
import builtins
import csv
import importlib.util
import itertools
import json
import math
import os

import numpy as np
import pytest
from scipy import stats

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("neural_mdd_test", os.path.join(TREE, "scripts", "neural_mdd.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()

SUMMARY = """# Manifest tiers and lockbox

## Lockbox

| source | lane | pool persons | lockbox persons | groups |
|---|---|---|---|---|
| amass | body | 470 | 53 | subsets X1, X2 |
| cap_sleep | neural | 34 | 7 | all 7 |
| pd_rest_eeg_ds004584 | neural | 149 | 30 | Control 10, PD 20 |
| pd_walking_eeg_ds007526 | neural | 144 | 29 | all 29 |

Lockbox persons: 119.
Lockbox file SHA-256: {sha}.

## Changes from v1.1

| source | lane | pool persons | lockbox persons | groups |
|---|---|---|---|---|
| ignored | neural | 1 | 1 | all 1 |
"""
SHA = "ab" * 32


def test_guard_refuses_lockbox_persons(tmp_path):
    p = tmp_path / "lockbox_persons.csv"
    p.write_text("source,lane,person_id,group,rule\n", encoding="utf-8")
    with pytest.raises(M.LockboxValueRefused):
        M.read_csv(str(p))
    with pytest.raises(M.LockboxValueRefused):
        M.sha256_file(str(p))
    with pytest.raises(M.LockboxValueRefused):
        M.read_text(str(tmp_path / "lockbox_persons_v1_1.csv"))


def test_parse_lockbox_summary():
    rows, sha = M.parse_lockbox_summary(SUMMARY.format(sha=SHA))
    assert sha == SHA
    assert set(rows) == {"amass", "cap_sleep", "pd_rest_eeg_ds004584", "pd_walking_eeg_ds007526"}
    assert rows["pd_rest_eeg_ds004584"]["groups"] == {"Control": 10, "PD": 20}
    assert rows["cap_sleep"]["lockbox_persons"] == 7 and rows["cap_sleep"]["pool_persons"] == 34
    assert rows["amass"]["groups"] is None
    with pytest.raises(ValueError):
        M.parse_groups("Control 10, PD 19", 30)


def test_ds004998_units_readings_and_interleaved():
    adm = [
        {"sub": "a", "task": "HoldL", "acq": "MedOff", "g15_header_verdict": "pass"},
        {"sub": "a", "task": "HoldR", "acq": "MedOn", "g15_header_verdict": "pass"},
        {"sub": "b", "task": "MoveL", "acq": "MedOff", "g15_header_verdict": "pass"},
        {"sub": "b", "task": "MoveL", "acq": "MedOn", "g15_header_verdict": "pass"},
        {"sub": "c", "task": "Rest", "acq": "MedOff", "g15_header_verdict": "pass"},
        {"sub": "c", "task": "Rest", "acq": "MedOn", "g15_header_verdict": "pass"},
        {"sub": "d", "task": "HoldL", "acq": "MedOff", "g15_header_verdict": "pass"},
        {"sub": "d", "task": "HoldL", "acq": "MedOn", "g15_header_verdict": "fail"},
    ]
    census = [{"sub": "c", "interleaved": "yes"}, {"sub": "b", "interleaved": "no"}]
    cls = M.ds004998_units(adm, census, "class")
    side = M.ds004998_units(adm, census, "side")
    assert cls["subjects_both_states"] == 3          # a (hold class), b, c; d's MedOn failed G15
    assert side["subjects_both_states"] == 2         # a has HoldL off and HoldR on only
    assert cls["pooled"] == 2 and side["pooled"] == 1
    assert cls["interleaved_in_both"] == ["c"]


def test_order_null_feasible_columns():
    rows = [
        {"sub": "a", "unit": "acquisitions", "reading": "class", "feasible": "no", "feasible_counting_ties": "yes"},
        {"sub": "b", "unit": "acquisitions", "reading": "class", "feasible": "yes", "feasible_counting_ties": "yes"},
        {"sub": "a", "unit": "bids_runs", "reading": "class", "feasible": "no", "feasible_counting_ties": "yes"},
    ]
    assert M.order_null_feasible(rows, "acquisitions", "feasible") == {"class": (1, 2)}
    assert M.order_null_feasible(rows, "acquisitions", "feasible_counting_ties") == {"class": (2, 2)}
    assert M.order_null_feasible(rows, "bids_runs", "feasible_counting_ties") == {"class": (1, 1)}


def test_normal_and_t_factors():
    # Bloom 1995 Table 1: 2.80 for a two-sided 0.05 test at 80 percent power
    assert M.z_factor(1, 0.05, 0.80) == pytest.approx(2.8016, abs=1e-4)
    for n in (5, 12, 30):
        kt = M.t_factor(n, 0.05, 0.80)
        df = n - 1
        power = stats.nct.sf(stats.t.ppf(0.975, df), df, kt * math.sqrt(n))
        assert power == pytest.approx(0.80, abs=1e-8)
        assert kt > M.z_factor(n, 0.05, 0.80)
    assert M.t_factor(400, 0.05, 0.80) == pytest.approx(M.z_factor(400, 0.05, 0.80), rel=0.01)


def test_multisets_count_and_weights():
    for n in (2, 3, 5):
        c, w = M.multisets(n)
        assert c.shape == (math.comb(2 * n - 1, n), n)
        assert (c.sum(axis=1) == n).all()
        assert w.sum() == pytest.approx(1.0)
    c, w = M.multisets(2)
    # (2,0) and (0,2) have probability 1/4 each, (1,1) 1/2
    probs = {tuple(r): p for r, p in zip(c.tolist(), w.tolist())}
    assert probs[(1, 1)] == pytest.approx(0.5)


def test_order_stat_index_type1():
    assert M.order_stat_index(0.025, 2000) == 49
    assert M.order_stat_index(0.975, 2000) == 1949
    assert M.order_stat_index(0.5, 1) == 0


def test_powers_constant():
    assert M.POWERS == (0.80, math.sqrt(0.80), 0.90)
    assert M.G16_MARGIN == 0.0 and M.G21_MARGIN == 0.05


def _explicit_means(x):
    """All n^n equally likely ordered resamples of the n units, as means (n^n, sims)."""
    n = x.shape[0]
    idx = np.array(list(itertools.product(range(n), repeat=n)))
    return np.sort(x[idx].mean(axis=1), axis=0)


def _type1_index(p, b):
    return max(0, math.ceil(p * b - 1e-9) - 1)


def test_weighted_type1_matches_explicit_expansion():
    n = 3
    x = np.random.default_rng(11).standard_normal((n, 40))
    expl = _explicit_means(x)                                   # (27, 40)
    c, w = M.multisets(n)
    means = (c.astype(float) / n) @ x
    order = np.argsort(means, axis=0, kind="stable")
    sv = np.take_along_axis(means, order, axis=0)
    wc = np.cumsum(w[order], axis=0)
    for p in (0.01, 1 / 27, 2 / 27, 0.1, 5 / 27, 0.5, 13 / 27, 0.9, 22 / 27, 26 / 27, 0.99):
        got = M.weighted_type1(sv, wc, p)
        assert np.allclose(got, expl[_type1_index(p, 27)]), p


def test_boot_bounds_enumerated_equal_explicit_expansion():
    n, sims = 3, 60
    alpha = 10 / 27                                             # bounds at 5/27 and 22/27 of the 27 draws
    lo, hi, enumerated, resamples = M.boot_bounds_mean(n, alpha, sims, 2000, np.random.default_rng(5))
    assert enumerated and resamples == 10
    expl = _explicit_means(np.random.default_rng(5).standard_normal((n, sims)))
    assert np.allclose(lo, expl[4]) and np.allclose(hi, expl[21])
    lo2, hi2, _, _ = M.boot_bounds_mean(n, 0.05, sims, 2000, np.random.default_rng(5))
    assert np.allclose(lo2, expl[0]) and np.allclose(hi2, expl[26])


@pytest.mark.parametrize("n,boot,expect", [(5, 200, "enumerated"), (12, 100, "drawn")])
def test_cluster_factors_equal_direct_computation(n, boot, expect):
    sims, seed, alpha = 800, 13, 0.05
    rows = M.cluster_factors(n, alpha, M.POWERS, sims, boot, seed)
    lo, hi, enumerated, _ = M.boot_bounds_mean(n, alpha, sims, boot, np.random.default_rng([seed, n]))
    assert enumerated == (expect == "enumerated")
    for p in M.POWERS:
        r = {x["method"]: x for x in rows if x["power"] == p}
        assert r["z"]["k"] == pytest.approx(M.z_factor(n, alpha, p)) and r["z"]["absent_factor"] == r["z"]["k"]
        assert r["t"]["k"] == pytest.approx(M.t_factor(n, alpha, p)) and r["t"]["absent_factor"] == r["t"]["k"]
        b = r["boot"]
        assert b["resampling"] == expect
        assert b["k"] == pytest.approx(-np.quantile(lo, 1 - p))
        assert b["absent_factor"] == pytest.approx(np.quantile(hi, p))
        assert b["false_pass_at_0"] == pytest.approx(np.mean(lo > 0))
        # meaning: bounds shift one for one with delta (common random numbers), so at delta = k
        # the lower bound clears 0 with probability p, and at sigma_d = margin / a and a zero gain
        # the upper bound falls below the margin with probability p
        assert np.mean(lo + b["k"] > 0) == pytest.approx(p, abs=2 / sims)
        assert np.mean(hi < b["absent_factor"]) == pytest.approx(p, abs=2 / sims)


def test_boot_bounds_scale_and_large_n():
    rng = np.random.default_rng(1)
    lo, hi, enumerated, resamples = M.boot_bounds_mean(3, 0.05, 500, 2000, rng)
    assert enumerated and resamples == math.comb(5, 3)
    assert (lo <= hi).all()
    rows = M.cluster_factors(40, 0.05, (0.80,), 3000, 1000, seed=7)
    kb = next(r for r in rows if r["method"] == "boot")
    kz = next(r for r in rows if r["method"] == "z")
    assert kb["resampling"] == "drawn"
    assert kb["k"] == pytest.approx(kz["k"], rel=0.12)
    assert kb["false_pass_at_0"] < 0.06


def test_auroc_bootstrap_matches_direct_and_is_monotone():
    sim = M.AurocSim(6, 4, 0.5, sims=30, boot=50, alpha=0.05, seed=3, chunk=7)
    h0, h1 = sim.auc_matrices(0.7, 0.1)
    d = h1 - h0
    rng = np.random.default_rng(0)
    cp = rng.multinomial(6, np.full(6, 1 / 6))
    cn = rng.multinomial(4, np.full(4, 1 / 4))
    via_matrix = float(cp @ d[0] @ cn) / 24
    pos = np.repeat(np.arange(6), cp)
    neg = np.repeat(np.arange(4), cn)
    direct = np.mean([d[0][i, j] for i in pos for j in neg])
    assert via_matrix == pytest.approx(direct)
    assert float(d[0].mean()) == pytest.approx(float(h1[0].mean() - h0[0].mean()))
    powers = [sim.run(0.7, x)[0] for x in (0.0, 0.1, 0.2, 0.29)]
    assert powers == sorted(powers)


def test_auroc_lower_bounds_equal_direct_type1_quantile():
    n_pos, n_neg, sims, boot, chunk = 6, 4, 30, 50, 7
    sim = M.AurocSim(n_pos, n_neg, 0.5, sims=sims, boot=boot, alpha=0.05, seed=3, chunk=chunk)
    power, diff, lo = sim.run(0.7, 0.1)
    h0, h1 = sim.auc_matrices(0.7, 0.1)
    d = h1 - h0
    rng = np.random.default_rng(sim.boot_seed)
    expect = np.empty(sims)
    for s0 in range(0, sims, chunk):
        cp = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=boot)
        cn = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=boot)
        for s in range(s0, min(sims, s0 + chunk)):
            bd = np.sort([float(cp[b] @ d[s] @ cn[b]) / (n_pos * n_neg) for b in range(boot)])
            expect[s] = bd[_type1_index(0.025, boot)]
    assert np.allclose(lo, expect)
    assert power == pytest.approx(np.mean(expect > 0))
    assert np.allclose(diff, d.mean(axis=(1, 2)))


def test_auroc_mdd_bisection_brackets_target():
    a0, target, steps = 0.7, 0.5, 6
    sim = M.AurocSim(12, 8, 0.5, sims=60, boot=100, alpha=0.05, seed=4, chunk=20)
    m, p_max = sim.mdd(a0, target, 0.0, steps=steps)
    assert m is not None and m > 0
    width = ((1.0 - a0) - 1e-6) / 2 ** steps
    assert sim.run(a0, m)[0] >= target
    assert sim.run(a0, m - width)[0] < target
    assert p_max == sim.run(a0, (1.0 - a0) - 1e-6)[0]
    none, p_max2 = sim.mdd(a0, 1.01, 0.0, steps=steps)
    assert none is None and p_max2 == p_max


def test_auroc_ceiling_and_verdict():
    rows = M.auroc_ceiling(6, 4, 0.05, sims=50, boot=40, seed=1, a0s=(0.7,), rhos=(0.0, 0.5))
    cells = [r for r in rows if r["scope"] == "cell"]
    pooled = [r for r in rows if r["scope"] != "cell"]
    assert len(cells) == 2 and len(pooled) == 1
    for r in cells:
        direct = M.AurocSim(6, 4, r["rho"], 50, 40, 0.05, 2).run(0.7, 0.3 - 1e-6)[0]
        assert r["power_at_auc_1"] == pytest.approx(direct)
        assert r["mc_se"] == pytest.approx(math.sqrt(max(direct * (1 - direct), 1e-12) / 50))
    assert pooled[0]["sims"] == 100
    assert pooled[0]["power_at_auc_1"] == pytest.approx(np.mean([r["power_at_auc_1"] for r in cells]))
    assert M.ceiling_verdict(0.90, 0.002, 0.80) == "yes"
    assert M.ceiling_verdict(0.886, 0.002, 0.8944) == "no"
    assert M.ceiling_verdict(0.892, 0.002, 0.8944) == "borderline"
    assert M.ceiling_verdict(0.8990, 0.002, 0.8944) == "yes"


def test_sensefog_freezers_from_draft_text():
    text = ("### 6.5 Q5 (exploratory)\n\n- NR-6.5.1. Data [A3 Q5]: multimodal_fog, 12 PD; sensefog, 198 s of "
            "labelled freezing in 4\n  subjects and 0.047 h of voluntary stops.\n")
    assert M.sensefog_freezers(text) == (198, 4)
    assert M.sensefog_freezers("- NR-6.5.1. Data: multimodal_fog only.\n") is None


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _rows_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.mark.parametrize("with_draft", [True, False])
def test_main_on_synthetic_tree(tmp_path, monkeypatch, with_draft):
    tree = str(tmp_path / "tree")
    if with_draft:
        _write(os.path.join(tree, "docs/rules/NEURAL_RULES_DRAFT.md"),
               "- NR-6.5.1. Data [A3 Q5]: multimodal_fog, 12 PD; sensefog, 198 s of labelled freezing in 4\n"
               "  subjects and 0.047 h of voluntary stops.\n")
    _write(os.path.join(tree, "reports/lockbox/summary.md"), SUMMARY.format(sha=SHA))
    _write(os.path.join(tree, "reports/lockbox/lockbox.sha256"), f"{SHA}  lockbox_persons.csv\n")
    _write(os.path.join(tree, "reports/lockbox/lockbox_persons.csv"), "source,lane,person_id,group,rule\nx,neural,SECRET,PD,r\n")
    _write(os.path.join(tree, "reports/manifest_tiers/hashes.json"),
           json.dumps({"lockbox_sha256": SHA, "moveatlas_head": "abc1234 test"}))
    tiers = [
        ("cap_sleep", 34, 7), ("pd_rest_eeg_ds004584", 149, 30), ("pd_walking_eeg_ds007526", 144, 29),
        ("multimodal_fog", 12, 0), ("sensefog", 11, 0),
    ]
    _rows_csv(os.path.join(tree, "reports/manifest_tiers/tiers_by_source.csv"),
              ["source", "modality", "tier", "lane", "records", "persons", "lockbox_records", "lockbox_persons"],
              [{"source": s, "modality": "eeg", "tier": "RELEASE", "lane": "neural", "records": p,
                "persons": p, "lockbox_records": lb, "lockbox_persons": lb} for s, p, lb in tiers])
    adm = []
    for sub in ("a", "b", "c", "d"):
        for acq in ("MedOff", "MedOn"):
            adm.append({"run_id": f"{sub}_{acq}", "sub": sub, "task": "HoldL", "acq": acq, "parts": "",
                        "n_samples": 1, "g15_header_verdict": "pass", "g15_reasons": "", "duplicates_left_out": ""})
    _rows_csv(os.path.join(tree, "reports/ds004998/admitted_runs.csv"), list(adm[0]), adm)
    _rows_csv(os.path.join(tree, "reports/ds004998/order_census.csv"), ["sub", "interleaved"],
              [{"sub": "a", "interleaved": "yes"}, {"sub": "b", "interleaved": "no"},
               {"sub": "c", "interleaved": "no"}, {"sub": "d", "interleaved": "no"}])
    feas = [{"sub": s, "unit": u, "reading": rd, "feasible": "no", "feasible_counting_ties": "yes"}
            for s in "abcd" for u in ("acquisitions", "bids_runs") for rd in ("class", "side")]
    _rows_csv(os.path.join(tree, "reports/ds004998/order_null_feasibility.csv"), list(feas[0]), feas)
    _write(os.path.join(tree, "reports/ds004998/_provenance.json"),
           json.dumps({"moveatlas_stamp": {"head": "f" * 40, "diff_sha256": "0" * 64}}))

    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    # the script waits while the queue's real SLOW flag exists (a long training sets it); the test must not
    monkeypatch.setattr(M, "wait_while_slow", lambda *a, **k: None)
    monkeypatch.setattr(builtins, "open", spy)
    out = str(tmp_path / "out")
    rc = M.main(["--tree", tree, "--out", out, "--a0", "0.7", "--rho", "0.5",
                 "--sims", "200", "--boot", "100", "--auc-sims", "40", "--auc-boot", "60", "--steps", "3",
                 "--auc-ceiling-sims", "50"])
    monkeypatch.setattr(builtins, "open", real_open)
    assert rc == 0
    assert not any(os.path.basename(p).startswith("lockbox_persons") for p in opened)
    for f in ("sizes.csv", "cluster_factors.csv", "auroc_ds004584.csv", "auroc_ceiling.csv", "mdd.csv", "summary.md",
              "_provenance.json"):
        assert os.path.exists(os.path.join(out, f)), f
    with open(os.path.join(out, "sizes.csv"), encoding="utf-8") as fh:
        sizes = {r["target"]: r for r in csv.DictReader(fh)}
    med = next(r for t, r in sizes.items() if t.startswith("ds004998"))
    assert med["n_units"] == "3"                      # 4 subjects with both states, minus 1 interleaved
    cap = next(r for t, r in sizes.items() if t.startswith("cap_sleep"))
    assert cap["n_units"] == "7"
    q5 = [r for t, r in sizes.items() if "labelled freezing" in t]
    assert [r["n_units"] for r in q5] == (["4"] if with_draft else [])
    with open(os.path.join(out, "summary.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "SECRET" not in text and "\u2014" not in text and "\u2013" not in text
    prov = json.load(open(os.path.join(out, "_provenance.json"), encoding="utf-8"))
    assert prov["lockbox_persons_file_opened"] is False
    assert prov["facts"]["lockbox_sha256_consistent"] is True
    with open(os.path.join(out, "mdd.csv"), encoding="utf-8") as fh:
        mdd = list(csv.DictReader(fh))
    g21 = [r for r in mdd if r["gate"] == "G21" and r["parameter"] != "" and abs(float(r["parameter"]) - 0.1) < 1e-9
           and r["method"] == "z" and abs(float(r["power"]) - 0.8) < 1e-9]
    assert g21 and all(float(r["mdd"]) > 0.05 for r in g21)
    # every grid MDD is margin + k_n x sigma_d with the k_n of cluster_factors.csv, and every
    # absent bound is margin / a_n
    with open(os.path.join(out, "cluster_factors.csv"), encoding="utf-8") as fh:
        cf = {(int(r["n"]), r["method"], round(float(r["power"]), 6)): r for r in csv.DictReader(fh)}
    n_grid = n_absent = 0
    for r in mdd:
        if r["mdd_over_sigma"] == "":
            continue
        f = cf[(int(r["n_units"]), r["method"], round(float(r["power"]), 6))]
        assert float(r["mdd_over_sigma"]) == pytest.approx(float(f["k"]), abs=2e-6)
        if r["parameter"] != "":
            n_grid += 1
            assert float(r["mdd"]) == pytest.approx(float(r["margin"]) + float(f["k"]) * float(r["parameter"]), abs=3e-6)
        elif float(r["margin"]) > 0:
            n_absent += 1
            assert float(r["absent_sigma_max"]) == pytest.approx(float(r["margin"]) / float(f["absent_factor"]), abs=3e-6)
    assert n_grid > 0 and n_absent > 0
    assert any(r["gate"] == "G21" and r["parameter"] != "" for r in mdd)
    assert "cannot pass" in text and "can read only undetermined" not in text
    with open(os.path.join(out, "auroc_ceiling.csv"), encoding="utf-8") as fh:
        ceil = list(csv.DictReader(fh))
    assert [r["scope"] for r in ceil] == ["cell", "pooled over rho"]
