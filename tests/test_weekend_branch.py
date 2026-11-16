"""Tests of scripts/weekend_branch.py on the fixture tree of tests/fixtures/weekend_branch/make_tree.py: every
row of the branch stage both firing and not firing, a missing input printing 'not evaluated' and blocking R3,
both G3 json names (g3_dev.json, g3_sim.json) read, and the seeds, v2check, reference and extension stages.
"""
import datetime
import importlib.util
import json
import os

import pytest

from mdfm import paths

HERE = os.path.dirname(os.path.abspath(__file__))
NOW = "2026-10-04T17:45:00"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load("weekend_branch_script", os.path.join(paths.TREE, "scripts", "weekend_branch.py"))
F = _load("weekend_branch_fixture", os.path.join(HERE, "fixtures", "weekend_branch", "make_tree.py"))


def run(root, stage, *extra, write=False):
    """(result, printed text) of one stage on a fixture root."""
    argv = ["--root", str(root), "--stage", stage, "--now", NOW, *extra] + (["--write"] if write else [])
    a = W.parse(argv)
    ctx = W.Ctx(a)
    res = W.STAGES[stage](ctx)
    text = W.render(ctx, stage, res, W.frozen_check(ctx.root))
    if write:
        W.write_outputs(ctx, stage, res, W.frozen_check(ctx.root), text)
    return res, "\n".join(text)


def rows(res):
    return {r["id"]: r for r in res["rows"]}


def branch(tmp_path, **over):
    F.build(str(tmp_path), **over)
    res, text = run(tmp_path, "branch")
    return rows(res), res, text


# ---------------------------------------------------------------- hashes and helpers
def test_constants_hash_is_canonical_and_stable():
    c = W.canonical_constants()
    assert c == json.dumps(json.loads(c), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert W.constants_sha256() == W.constants_sha256() and len(W.constants_sha256()) == 64
    assert W.CONSTANTS["lines"]["g1a_mm"] == {"pass": 45.0, "kill": 70.0}
    assert W.CONSTANTS["bootstrap"]["resamples"] == 10000 and W.CONSTANTS["bootstrap"]["seed"] == 20261001


def test_frozen_check_reads_the_handoff(tmp_path):
    assert W.frozen_check(str(tmp_path))["match"] is None
    h = tmp_path / "docs" / "handoffs"
    h.mkdir(parents=True)
    good = W.frozen_check(str(tmp_path))
    (h / "MDFM_HANDOFF_5.md").write_text(f"weekend_branch.py SHA-256: {good['script']}\n"
                                         f"constants SHA-256: {good['constants']}\n", encoding="utf-8")
    assert W.frozen_check(str(tmp_path))["match"] is True
    (h / "MDFM_HANDOFF_5.md").write_text(f"weekend_branch.py SHA-256: {'0' * 64}\n"
                                         f"constants SHA-256: {good['constants']}\n", encoding="utf-8")
    assert W.frozen_check(str(tmp_path))["match"] is False


def test_bootstrap_is_deterministic_and_paired():
    a = {f"p{i}": 10.0 + i for i in range(12)}
    b = {f"p{i}": 8.0 + i + (0.1 if i % 2 else -0.1) for i in range(12)}
    p1, p2 = W.paired(a, b), W.paired(a, b)
    assert p1 == p2 and p1["n"] == 12 and abs(p1["point"] - 2.0) < 1e-9
    assert p1["excludes_0"] and p1["interval"][0] > 1.8
    q = W.paired({"p0": 1.0}, {"p0": 0.0})
    assert q["point"] is None and "paired units" in q["reason"]
    assert W.short_id("weargait:NLS901 (control)") == "NLS901"
    assert not W.in_subset("weargait:WHC026", "no_whc003_whc026") and W.in_subset("weargait:WHC026", "no_whc003")


# ---------------------------------------------------------------- the base scenario
def test_branch_base_scenario(tmp_path):
    r, res, text = branch(tmp_path)
    assert r["R0"]["verdict"] == "R0 holds"
    assert r["R1"]["verdict"] == "clear of the kill lines"
    assert "pass" not in r["R1"]["verdict"].replace("pass lines", "")
    assert "x - pass" in text and "x - kill" in text
    assert r["R-IC"]["verdict"] == "reported only: margin not met"
    assert r["R-insim"]["verdict"] == "G3 pass unreachable by sim-to-real fixes alone"
    assert r["R-AB"]["verdict"] == "phase B improved"
    assert r["M1"]["verdict"] == "does not fire" and r["M1"]["status"] == "evaluated"
    assert r["M2"]["verdict"] == "fires"
    assert r["M4"]["verdict"] == "fires"
    assert r["M6"]["verdict"] == "does not fire"
    assert res["v2_item"] == "M2" and res["v2_line"] == "stride_mae_cm"
    assert set(r["R2a"]["data"]["proposals"]) == {"refine60 (G1a)", "lowpass12 (S)"}
    assert r["R3"]["verdict"].startswith("R3 does not apply")
    assert abs(r["M2"]["data"]["fraction"] - 2.0 / 3.0) < 0.02
    assert abs(r["M4"]["data"]["fraction"] - 6.0 / 17.0) < 0.03


# ---------------------------------------------------------------- R0
@pytest.mark.parametrize("over,failed", [({"phase_b_end": False}, "verify_present"),
                                         ({"alarm": True}, "no_alarm"),
                                         ({"mismatches": ["weargait_pd"]}, "report_mismatches_empty")])
def test_r0_fails(tmp_path, over, failed):
    r, res, _ = branch(tmp_path, **over)
    assert r["R0"]["verdict"].startswith("R0 fails") and failed in r["R0"]["verdict"]
    assert "R0 fails" in res["decision"]


def test_r0_exit_code_from_job_log(tmp_path):
    F.build(str(tmp_path))
    F.write(os.path.join(str(tmp_path), "reports/gpu_queue/job_log.csv"),
            "name,cls,argv,exit_code\nb,2,python scripts/train_stage1.py --run-dir cache/runs/phaseA_root_full,1\n")
    res, _ = run(tmp_path, "branch")
    assert rows(res)["R0"]["verdict"] == "R0 fails: exit_0"
    res, _ = run(tmp_path, "branch", "--exit-code", "0")
    assert rows(res)["R0"]["verdict"] == "R0 holds"


# ---------------------------------------------------------------- R1, R-IC, R-insim
def test_r1_not_clear_and_never_pass(tmp_path):
    r, _, _ = branch(tmp_path, g1a=75.0)
    assert r["R1"]["verdict"] == "not clear of the kill lines: above g1a_mm"
    assert r["R1"]["data"]["above_kill"] == ["g1a_mm"]


def test_r_ic_margin_met(tmp_path):
    r, _, _ = branch(tmp_path, margin=(10.0, 13.33))
    assert r["R-IC"]["verdict"] == "reported only: margin met"


def test_r_insim_below_line(tmp_path):
    r, _, text = branch(tmp_path, insim=4.0)
    assert r["R-insim"]["verdict"] == "in-sim forward stride MAE at most 5 cm"
    assert "G3 pass unreachable" not in text


# ---------------------------------------------------------------- R-AB
def test_r_ab_did_not_improve_bars_m1(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    VB = F.person_map(F.V_B12, 1.0, 0.4)
    r, res, _ = branch(tmp_path, S_A=F.shift(SB, -1.0, F.JIT), V_A=F.shift(VB, -0.01), hurried_bias=-0.15,
                       S_lowpass12=F.shift(SB, 0.0, F.ALT), m4_effect=1.0)
    assert r["R-AB"]["verdict"] == "phase B did not improve"
    assert r["M1"]["data"]["fires"] and "barred by R-AB" in r["M1"]["verdict"]
    assert res["v2_item"] is None
    assert "firing items barred by R-AB: M1" in res["r3"]


def test_r_ab_inconclusive(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    VB = F.person_map(F.V_B12, 1.0, 0.4)
    r, _, _ = branch(tmp_path, S_A=F.shift(SB, 0.05, F.ALT), V_A=F.shift(VB, 0.0, [x / 100 for x in F.ALT]))
    assert r["R-AB"]["verdict"] == "inconclusive"


# ---------------------------------------------------------------- menu items
def test_m1_fires_with_all_conditions(tmp_path):
    r, res, _ = branch(tmp_path, hurried_bias=-0.15)
    assert r["M1"]["verdict"] == "fires"
    assert abs(r["M1"]["data"]["fraction"] - 0.4) < 0.03
    assert res["v2_item"] == "M2"          # 0.67 of the gap beats 0.40


def test_m1_warp_not_above(tmp_path):
    warp = {"rows": [{"factor": 1.0, "stride_mae_cm": 9.0}, {"factor": 1.5, "stride_mae_cm": 8.0}]}
    r, _, _ = branch(tmp_path, hurried_bias=-0.15, warp=warp)
    assert r["M1"]["verdict"] == "does not fire" and r["M1"]["status"] == "evaluated"


def test_m1_warp_without_stride_error_is_not_evaluated(tmp_path):
    warp = {"warp": {"1.25": {"decoded_ratio": 1.2}, "1.5": {"decoded_ratio": 1.4}, "1.0": {"decoded_ratio": 1.0}}}
    r, _, _ = branch(tmp_path, hurried_bias=-0.15, warp=warp)
    assert r["M1"]["status"] == "not evaluated" and "no stride error" in r["M1"]["verdict"]


def test_m2_does_not_fire_and_m6_fires(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    r, res, _ = branch(tmp_path, S_lowpass12=F.shift(SB, 0.0, F.ALT), S_model_legs=F.shift(SB, -2.5, F.JIT))
    assert r["M2"]["verdict"] == "does not fire"
    assert r["M6"]["verdict"] == "fires"
    assert res["v2_item"] == "M6"


def test_m4_does_not_fire(tmp_path):
    r, _, _ = branch(tmp_path, m4_effect=1.0)
    assert r["M4"]["verdict"] == "does not fire" and r["M4"]["status"] == "evaluated"


def test_m4_only_firing_item_is_v2(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    r, res, _ = branch(tmp_path, S_lowpass12=F.shift(SB, 0.0, F.ALT))
    assert r["M4"]["verdict"] == "fires" and res["v2_item"] == "M4" and res["v2_line"] == "g1a_mm"


def test_tie_goes_to_g3():
    def item(i, frac):
        return {"id": i, "status": "evaluated", "verdict": "fires", "data": {"fires": True, "fraction": frac}}
    v2, r3, missing, barred = W.select_v2([item("M4", 0.5), item("M2", 0.5)], "phase B improved")
    assert v2 == "M2" and not missing and not barred
    v2, _, _, _ = W.select_v2([item("M4", 0.6), item("M2", 0.5)], "phase B improved")
    assert v2 == "M4"


def test_no_gap_never_fires(tmp_path):
    SB = F.person_map([3.0] * 12, 40.0, 20.0)
    eff = W.paired({k: 3.0 + i for i, k in enumerate(SB)}, {k: 2.0 + i for i, k in enumerate(SB)})
    out = W.menu_item("M2", eff, 4.0)
    assert out["verdict"] == "does not fire: line at or below its pass line" and not out["data"]["fires"]


# ---------------------------------------------------------------- R3 and missing inputs
def _none_fires():
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    return dict(S_lowpass12=F.shift(SB, 0.0, F.ALT), m4_effect=1.0)


def test_r3_fires_when_every_row_is_evaluated_and_none_fires(tmp_path):
    r, res, _ = branch(tmp_path, **_none_fires())
    assert all(r[m]["status"] == "evaluated" and not r[m]["data"]["fires"] for m in ("M1", "M2", "M4", "M6"))
    assert r["R3"]["verdict"] == "R3: every menu row evaluated and none fires, so there is no v2"
    assert res["v2_item"] is None


@pytest.mark.parametrize("over,item,why", [
    ({"drop_rows": ("lowpass12",)}, "M2", "no diag row lowpass12"),
    ({"drop_rows": ("model_legs",)}, "M6", "no diag row model_legs"),
    ({"drop_base_keys": ("spearman_speed_error",)}, "M1", "no spearman_speed_error"),
    ({"drop_base_keys": ("selfpace_stride_mae_cm",)}, "M1", "no per-person SelfPace"),
    ({"warp": None}, "M1", "missing: reports/bench/root_motion_"),
    ({"decompose": False}, "M4", "no decompose.json"),
    ({"drop_dec_keys": ("per_take",)}, "M4", "per_take lacks"),
])
def test_missing_input_is_not_evaluated_and_blocks_r3(tmp_path, over, item, why):
    o = _none_fires()
    o.update(over)
    r, res, text = branch(tmp_path, **o)
    assert r[item]["status"] == "not evaluated" and r[item]["verdict"].startswith("not evaluated")
    assert why in r[item]["verdict"]
    assert r["R3"]["verdict"].startswith("R3 blocked") and item in r["R3"]["verdict"]
    assert res["v2_item"] is None
    assert f"{item}: not evaluated" in text


def test_unreadable_inputs_are_not_evaluated(tmp_path):
    F.build(str(tmp_path), **_none_fires())
    F.write(os.path.join(str(tmp_path), "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.json"),
            "{not json")
    F.write(os.path.join(str(tmp_path), "reports/gates/g3/diag_phaseA_root_full_ckpt_B_032044/diag.json"), "[1, 2]")
    res, _ = run(tmp_path, "branch")
    r = rows(res)
    assert r["M4"]["status"] == "not evaluated" and "unreadable" in r["M4"]["verdict"]
    assert r["M2"]["status"] == "not evaluated" and "not a json object" in r["M2"]["verdict"]
    assert res["v2_item"] is None and r["R3"]["verdict"].startswith("R3 blocked")


def test_missing_row_with_another_item_firing_names_v2(tmp_path):
    r, res, _ = branch(tmp_path, drop_rows=("model_legs",))
    assert r["M6"]["status"] == "not evaluated" and res["v2_item"] == "M2"
    assert "not evaluated: M6" in r["R3"]["verdict"]


def test_r2a_no_proposal(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    r, _, _ = branch(tmp_path, r60_gain=2.0, S_lowpass12=F.shift(SB, -1.0, F.JIT))
    assert r["R2a"]["verdict"] == "no proposal" and r["R2a"]["status"] == "evaluated"


# ---------------------------------------------------------------- both json names, fallbacks
@pytest.mark.parametrize("b_name,a_name", [("g3_dev.json", "g3_sim.json"), ("g3_sim.json", "g3_dev.json")])
def test_both_g3_json_names_are_read(tmp_path, b_name, a_name):
    r, _, text = branch(tmp_path, diag_a=False, diag_b=False, g3_b_name=b_name, g3_a_name=a_name)
    assert r["R-AB"]["verdict"] == "phase B improved"
    assert r["R1"]["verdict"] == "clear of the kill lines"
    assert r["R-IC"]["status"] == "evaluated"
    assert b_name in text and a_name in text
    assert r["M2"]["status"] == "not evaluated"          # the menu needs diag.json


def test_diag_rows_at_top_level_and_extras_at_top_level(tmp_path):
    F.build(str(tmp_path))
    p = os.path.join(str(tmp_path), "reports/gates/g3/diag_phaseA_root_full_ckpt_B_032044/diag.json")
    with open(p, encoding="utf-8") as f:
        doc = json.load(f)
    rows_ = doc["rows"]
    bias = rows_["base"].pop("hurried_speed_bias_ms")
    flat = dict(rows_)
    flat["hurried_speed_bias_ms"] = {"no_whc003_whc026": -0.2, "all14": bias}
    F.write_json(p, flat)
    res, _ = run(tmp_path, "branch")
    r = rows(res)
    assert r["M1"]["verdict"] == "fires"          # the top-level per-subset bias -0.2 is read on the 12
    assert r["M2"]["verdict"] == "fires"


# ---------------------------------------------------------------- outputs
def test_write_outputs(tmp_path):
    F.build(str(tmp_path))
    res, _ = run(tmp_path, "branch", write=True)
    stamp = datetime.datetime.fromisoformat(NOW).strftime("%Y%m%d_%H%M%S")
    js = os.path.join(str(tmp_path), "reports", "gpu_queue", f"branch_branch_{stamp}.json")
    md = js[:-5] + ".md"
    assert os.path.exists(js) and os.path.exists(md)
    with open(js, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["v2_item"] == "M2" and doc["constants_sha256"] == W.constants_sha256()
    assert doc["inputs"]["warp B"]["exists"] and doc["inputs"]["warp B"]["sha256"]
    with open(md, encoding="utf-8") as f:
        text = f.read()
    assert chr(0x2014) not in text and "R3 does not apply" in text


def test_cli_main_prints_rows(tmp_path, capsys):
    F.build(str(tmp_path))
    assert W.main(["--root", str(tmp_path), "--stage", "branch", "--now", NOW]) == 0
    out = capsys.readouterr().out
    for rid in ("R0:", "R1:", "R-IC:", "R-insim:", "R-AB:", "M1:", "M2:", "M4:", "M6:", "R2a:", "R3:", "decision:"):
        assert rid in out
    assert W.main(["--hashes"]) == 0


# ---------------------------------------------------------------- other stages
def test_seeds(tmp_path):
    F.build(str(tmp_path))
    res, _ = run(tmp_path, "seeds")
    d4, de = res["seed_diff"]["ckpt_B_004000"], res["seed_diff"]["B_end"]
    assert abs(d4["G1a"]["abs_diff"] - 1.0) < 1e-9 and d4["G1a"]["variant"] == "lt2"
    assert 0.4 < d4["S12"]["abs_diff"] < 0.6
    assert de["G1a"]["variant"] == "full" and abs(de["G1a"]["abs_diff"] - 1.0) < 1e-9
    assert all(v["abs_diff"] is not None for v in d4.values())


def test_seeds_without_seed2_end(tmp_path):
    F.build(str(tmp_path), s2_end=False)
    res, _ = run(tmp_path, "seeds")
    r = rows(res)
    assert r["seeds-end"]["status"] == "not evaluated" and "B_end" not in res["seed_diff"]


def _chain(tmp_path, **over):
    F.build(str(tmp_path), **over)
    run(tmp_path, "branch", write=True)
    run(tmp_path, "seeds", write=True)


def test_v2check_go(tmp_path):
    _chain(tmp_path)
    res, text = run(tmp_path, "v2check")
    assert res["go"] and res["decision"] == "go" and res["target"] == "S12"
    assert "10 and 5 percent only" not in text


@pytest.mark.parametrize("over,why", [({"v2_reserved": 14.5}, "peak reserved"),
                                      ({"v2_alarm": True}, "no alarm"),
                                      ({"v2_nonfinite": (100, 101, 102)}, "no non-finite stop"),
                                      ({"g1a_v2": 75.0}, "other G1a")])
def test_v2check_no_go(tmp_path, over, why):
    _chain(tmp_path, **over)
    res, _ = run(tmp_path, "v2check")
    assert not res["go"] and res["decision"].startswith("no-go") and why in res["decision"]


def test_v2check_target_not_improved(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    _chain(tmp_path, S_v2=F.shift(SB, 3.9))
    res, _ = run(tmp_path, "v2check")
    assert not res["go"] and "target S12" in res["decision"]


def test_v2check_nonfinite_not_consecutive_is_no_stop(tmp_path):
    _chain(tmp_path, v2_nonfinite=(100, 102, 104))
    res, _ = run(tmp_path, "v2check")
    assert res["go"]


def test_v2check_without_seed_numbers(tmp_path):
    F.build(str(tmp_path))
    run(tmp_path, "branch", write=True)
    res, text = run(tmp_path, "v2check")
    assert "10 and 5 percent only" in text and res["go"]


def test_v2check_without_v2_item(tmp_path):
    F.build(str(tmp_path), **_none_fires())
    run(tmp_path, "branch", write=True)
    res, _ = run(tmp_path, "v2check")
    assert not res["go"] and "names no v2 item" in res["decision"]


def test_reference_stays_v1(tmp_path):
    _chain(tmp_path)
    res, _ = run(tmp_path, "reference")
    assert res["reference_checkpoint"].replace("\\", "/").endswith("phaseA_root_full/ckpt_B_032044.pt")
    r = rows(res)
    assert r["candidate pilot_v1_s2"]["verdict"] == "does not beat v1"
    assert r["candidate pilot_v2"]["status"] == "not evaluated"      # its run has no phase B end


def test_reference_switches_to_a_candidate_that_beats_v1(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    F.build(str(tmp_path), S_s2_end=F.shift(SB, -2.0, F.JIT))
    seeds = os.path.join(str(tmp_path), "reports", "gpu_queue", "branch_seeds_20261004_000000.json")
    F.write_json(seeds, {"seed_diff": {"B_end": {"S12": {"abs_diff": 0.3}, "G1a": {"abs_diff": 1.0}}}})
    res, text = run(tmp_path, "reference")
    assert res["reference_checkpoint"].replace("\\", "/").endswith("pilot_v1_s2/ckpt_B_032044.pt")
    assert "post-selection, optimistic" in res["decision"] and res["label"] == "post-selection, optimistic"


def test_reference_without_seed_difference_stays_v1(tmp_path):
    SB = F.person_map(F.S_B12, 40.0, 20.0)
    F.build(str(tmp_path), S_s2_end=F.shift(SB, -2.0, F.JIT))
    res, _ = run(tmp_path, "reference")
    assert "phaseA_root_full" in res["reference_checkpoint"].replace("\\", "/")
    assert "no seed difference" in rows(res)["candidate pilot_v1_s2"]["verdict"]


def test_extension_extends_and_writes_extend(tmp_path):
    F.build(str(tmp_path))
    res, text = run(tmp_path, "extension", write=True)
    assert res["extend"] and abs(res["s_per_step"] - 2.4) < 1e-6
    assert res["ext_steps"] == int((13.25 - 1.5) * 3600 // 2.4)
    ext = os.path.join(str(tmp_path), "cache", "runs", "queue", "EXTEND")
    with open(ext, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["ext_steps"] == res["ext_steps"] and doc["ext_ckpt"].replace("\\", "/").endswith("ckpt_B_032044.pt")
    # a later 'no' removes the EXTEND this script wrote
    a = W.parse(["--root", str(tmp_path), "--stage", "extension", "--now", "2026-10-05T02:00:00", "--write"])
    ctx = W.Ctx(a)
    res2 = W.stage_extension(ctx)
    W.write_outputs(ctx, "extension", res2, W.frozen_check(ctx.root), W.render(ctx, "extension", res2,
                                                                                W.frozen_check(ctx.root)))
    assert not res2["extend"] and "under 4 h remain" in res2["decision"] and not os.path.exists(ext)


@pytest.mark.parametrize("over,why", [({"g1a_lt2_end": 65.0}, "G1 curve does not fall"),
                                      ({"S_75": None}, "S does not fall")])
def test_extension_no(tmp_path, over, why):
    if "S_75" in over:
        SB = F.person_map(F.S_B12, 40.0, 20.0)
        over = {"S_75": F.shift(SB, -0.5, F.JIT)}
    F.build(str(tmp_path), **over)
    res, _ = run(tmp_path, "extension")
    assert not res["extend"] and why in res["decision"]


NB = "phaseA_root_full_ckpt_B_032044"


# ---------------------------------------------------------------- input names and readings of the other tasks
def test_warp_json_named_by_the_probe_tag(tmp_path):
    """T07 queues probe_root_motion.py --tag _warp15, which writes root_motion_<name>__warp15.json."""
    F.build(str(tmp_path), hurried_bias=-0.15, **_none_fires())
    bench = os.path.join(str(tmp_path), "reports", "bench")
    tagged = os.path.join(bench, f"root_motion_{NB}__warp15.json")
    os.replace(os.path.join(bench, f"root_motion_{NB}_warp15.json"), tagged)
    res, _ = run(tmp_path, "branch")
    r = rows(res)
    assert r["M1"]["status"] == "evaluated" and r["M1"]["verdict"] == "fires"
    assert res["v2_item"] == "M1"
    os.remove(tagged)
    m1 = rows(run(tmp_path, "branch")[0])["M1"]
    assert m1["status"] == "not evaluated" and f"root_motion_{NB}_warp15.json" in m1["verdict"]


def test_diag_scalars_written_with_their_interval(tmp_path):
    r, _, _ = branch(tmp_path, hurried_bias={"point": -0.15, "interval": [-0.2, -0.1]},
                     spearman={"rho": -0.6, "p": 0.04})
    assert r["M1"]["verdict"] == "fires"
    assert W.scalar({"value": 2.0}) == 2.0 and W.scalar({"interval": [1, 2]}) is None and W.scalar(True) is None


def test_per_take_values_with_or_without_the_unit():
    assert W.take_value({"g1a_mm": 50.0}, "g1a_mm") == 50.0
    assert W.take_value({"g1a": 51.0}, "g1a_mm") == 51.0
    assert abs(W.take_value({"g1a_m": 0.052}, "g1a_mm") - 52.0) < 1e-9
    assert W.take_value({"g1a_mm": "x"}, "g1a_mm") is None


def _job_log(root, rows_):
    import csv
    path = os.path.join(str(root), "reports", "gpu_queue", "job_log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "cls", "state", "argv", "exit_code"])
        for name, state, argv, code in rows_:
            w.writerow([name, 2, state, json.dumps(argv), code])


def test_r0_job_log_rows_of_another_run_folder_are_left_out(tmp_path):
    F.build(str(tmp_path))
    py = "<conda>/envs/mdfm/python.exe"
    v2 = [py, "scripts/train_stage1.py", "--phase", "B", "--run-dir", "cache/runs/pilot_v2", "--resume",
          "cache/runs/phaseA_root_full/ckpt_A_010681.pt"]
    resume = [py, "scripts/train_stage1.py", "--phase", "B", "--resume",
              "<repo>\\cache\\runs\\phaseA_root_full\\ckpt_B_024000.pt"]
    _job_log(tmp_path, [("phaseB_root_full_resume", "done", resume, 0), ("pilot_v2_train", "failed", v2, 1)])
    r0 = rows(run(tmp_path, "branch")[0])["R0"]
    assert r0["verdict"] == "R0 holds" and r0["data"]["exit_code"] == 0 and "job log" in r0["lines"][3]
    _job_log(tmp_path, [("phaseB_root_full_resume", "done", resume, 0),
                        ("phaseB_root_full_resume2", "timeout", resume, "")])
    r0 = rows(run(tmp_path, "branch")[0])["R0"]
    assert r0["verdict"] == "R0 fails: exit_0" and "timeout" in str(r0["data"]["exit_code"])
    assert W.job_run_folder(["x", "--resume", "cache/runs/pilot_v1_s2/ckpt_A_010681.pt"]) == "pilot_v1_s2"
    assert W.job_run_folder(["x", "--run-dir", "cache/runs/pilot_v2", "--resume", "a/b.pt"]) == "pilot_v2"
    assert W.job_run_folder(["x", "--phase", "B"]) is None


def test_crash_without_verify_event_names_the_evaluated_checkpoint(tmp_path):
    F.build(str(tmp_path), phase_b_end=False)
    for base in ("reports/gates/g3", "reports/gates/g3_sim", "reports/gates/g1", "reports/bench"):
        p = os.path.join(str(tmp_path), base)
        for f in os.listdir(p):
            if NB in f:
                os.replace(os.path.join(p, f), os.path.join(p, f.replace(NB, "phaseA_root_full_ckpt_B_006000")))
    # no verify event and nothing fired by the queue runner: the default name, so no row reads a mid-run checkpoint
    res, _ = run(tmp_path, "branch")
    assert res["names"]["B"] == NB and "default name" in rows(res)["R0"]["lines"][-1]
    assert rows(res)["R1"]["status"] == "not evaluated"
    # the queue runner's fallback checkpoint (state.json phase_b.fired) names the B end
    state = os.path.join(str(tmp_path), "cache", "runs", "queue", "state.json")
    F.write_json(state, {"phase_b": {"fired": {"ckpt": "cache/runs/phaseA_root_full/ckpt_B_006000.pt",
                                               "name": "phaseA_root_full_ckpt_B_006000", "how": "R0 fallback"}}})
    res, _ = run(tmp_path, "branch")
    r = rows(res)
    assert res["names"]["B"] == "phaseA_root_full_ckpt_B_006000" and "state.json" in r["R0"]["lines"][-1]
    assert r["R0"]["verdict"].startswith("R0 fails") and "R0 fails" in res["decision"]
    assert r["R1"]["verdict"] == "clear of the kill lines" and res["v2_item"] == "M2"
    # a fired checkpoint of another run folder is not used
    F.write_json(state, {"phase_b": {"fired": {"ckpt": "cache/runs/pilot_v1_s2/ckpt_B_006000.pt"}}})
    assert run(tmp_path, "branch")[0]["names"]["B"] == NB
    res, _ = run(tmp_path, "branch", "--name", "phaseA_root_full_ckpt_B_006000")
    assert res["names"]["B"] == "phaseA_root_full_ckpt_B_006000" and "(--name)" in rows(res)["R0"]["lines"][-1]


def test_absolute_windows_checkpoint_path_gives_the_queue_name():
    recs = [{"event": "start", "steps": [10681, 32044]},
            {"event": "verify", "phase": "B", "step": 32044,
             "path": "<repo>\\cache\\runs\\phaseA_root_full\\ckpt_B_032044.pt"}]
    name, v = W.end_name(recs, "cache/runs/phaseA_root_full", "B")
    assert name == NB and v["step"] == 32044


def test_r_ab_not_evaluated_holds_back_a_firing_m1(tmp_path):
    F.build(str(tmp_path), hurried_bias=-0.15, diag_a=False, **_none_fires())
    import shutil
    shutil.rmtree(os.path.join(str(tmp_path), "reports/gates/g3_sim/dev_phaseA_root_full_ckpt_A_010681"))
    res, text = run(tmp_path, "branch")
    r = rows(res)
    assert r["R-AB"]["status"] == "not evaluated"
    assert r["M1"]["status"] == "not evaluated" and "R-AB is not evaluated" in r["M1"]["verdict"]
    assert res["v2_item"] is None and r["R3"]["verdict"].startswith("R3 blocked") and "M1" in r["R3"]["verdict"]
    # an allowed item still fires and is named
    F.build(str(tmp_path), hurried_bias=-0.15, diag_a=False)
    shutil.rmtree(os.path.join(str(tmp_path), "reports/gates/g3_sim/dev_phaseA_root_full_ckpt_A_010681"))
    res, _ = run(tmp_path, "branch")
    assert res["v2_item"] == "M2" and "not evaluated: M1" in rows(res)["R3"]["verdict"]


def test_r2a_prints_distances_and_reads_not_evaluated_without_inputs(tmp_path):
    r, _, _ = branch(tmp_path)
    assert all("x - pass" in x and "x - kill" in x for x in r["R2a"]["lines"])
    r, _, _ = branch(tmp_path / "none", decompose=False, drop_rows=("refine60", "lowpass12"))
    assert r["R2a"]["status"] == "not evaluated" and r["R2a"]["verdict"].startswith("not evaluated: ")


def test_r_insim_and_r1_not_evaluated_without_inputs(tmp_path):
    F.build(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), f"reports/gates/g3_sim/insim_{NB}/insim.json"))
    os.remove(os.path.join(str(tmp_path), f"reports/gates/g1/dev_{NB}/g1.json"))
    res, text = run(tmp_path, "branch")
    r = rows(res)
    assert r["R-insim"]["status"] == "not evaluated" and "missing" in r["R-insim"]["verdict"]
    assert r["R1"]["status"] == "not evaluated" and "G1a: no g1.json" in r["R1"]["verdict"]
    assert "G3 pass unreachable" not in text


def test_extension_rate_skips_dev_and_checkpoint_spans():
    recs = [{"event": "log", "phase": "B", "step": 50, "time": "2026-10-01T10:00:00"},
            {"event": "log", "phase": "B", "step": 100, "time": "2026-10-01T10:02:00"},
            {"event": "checkpoint", "phase": "B", "step": 100},
            {"event": "log", "phase": "B", "step": 150, "time": "2026-10-01T10:10:00"},
            {"event": "log", "phase": "B", "step": 200, "time": "2026-10-01T10:12:00"}]
    assert abs(W.seconds_per_step(recs) - 2.4) < 1e-9
