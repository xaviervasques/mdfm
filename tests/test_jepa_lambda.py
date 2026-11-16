"""Tests of scripts/jepa_lambda.py (task J2b): lambda of experiment E1 from the smoke run's log
(docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2.7), the smoke's checks and speed, the config write and the E1
job file, on hand-made logs (no run, no record, CPU only).

- lambda = mean(g_R) / mean(g_J) over the 'jepa_grad' events of phase B steps 0 to 49 of the last call: a step
  with a null norm is left out and counted, a step with g_J = 0 is kept and counted, the events of an earlier call
  are not read; a missing or repeated step fails the lambda check.
- The checks: a clean smoke passes; an alarm event, a nonfinite event, a peak reserved memory at 14.0 GB, a
  collapse ratio below 0.01, a null loss, an end before step 200, no verify event, a smoke at weight 2.0 and a
  missing init_from event each fail their check; the collapse line is loop.ALARMS['jepa_collapse'].
- Speed: the smoke's rate over all its intervals and after step 50, a reference's rate on the same steps and over
  its whole phase B (the last record of a step logged twice), their ratio and E1's projected hours.
- --write-config on a copy of configs/stage1_pilot_jepa.yaml: only the weight line changes and the copy parses to
  the old configuration with jepa.weight = lambda rounded to 6 significant digits; a second write leaves it as it is;
  a config the smoke did not run is refused; with a failing check the config is not touched, lambda.json and
  lambda.md say why and the command exits. --e1-job writes a job that queue_job.normalize accepts: training, cls 2,
  train_stage1.py --phase B --init-from (the smoke's source) --config (the copy), no --max-steps and no
  --jepa-grad-steps, its rate the largest projection, requiring lambda.json; with --e1-max-steps 4000 the same
  command cut at 4,000 steps (the JP2 screen). A later call on the same smoke log keeps the earlier calls' records in
  lambda.json and lambda.md (the config write, with its SHA-256 before and after, and every job file written), a
  second --write-config included; another smoke log starts them afresh.
- The outputs hold no em-dash or en-dash.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import replace

import pytest

from mdfm import paths
from mdfm.train import config as CF
from mdfm.train import loop as LP

SCRIPT = os.path.join(paths.TREE, "scripts", "jepa_lambda.py")
CONFIG = os.path.join(paths.TREE, "configs", "stage1_pilot_jepa.yaml")
SOURCE = "<repo>/cache/runs/phaseA_root_full/ckpt_A_010681.pt"
N_B = 32044


def load_script():
    spec = importlib.util.spec_from_file_location("jepa_lambda_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


JL = load_script()
sys.path.insert(0, os.path.join(paths.TREE, "scripts"))
import queue_job as QJ  # noqa: E402


def g_j(s):
    return None if s == 7 else (0.0 if s == 9 else 1.0 + 0.01 * s)


def g_r(s):
    return 2.0e5 + 1000.0 * s


def write_jsonl(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def smoke(folder, cfg_sha, *, weight=1.0, steps=200, peak=13.2, collapse=0.8, loss=1.0e6, grad=range(50),
          extra=(), verify=True, init=True):
    """A smoke run's log.jsonl in folder: an earlier crashed call, then the smoke's call (start, init_from,
    jepa_grad events at the steps of `grad`, a log event every 50 steps (200 s for the first interval, 150 s
    after), end, verify) with the `extra` events."""
    recs = [{"event": "start", "name": "stage1_pilot_jepa", "config_sha256": cfg_sha},
            {"event": "jepa_grad", "phase": "B", "step": 0, "g_J": 1e-9, "g_R": 1.0}]
    recs.append({"event": "start", "name": "stage1_pilot_jepa", "config_sha256": cfg_sha, "steps": [10681, N_B],
                 "jepa": {"section": dict(CF.to_dict(CF.JepaSection(enabled=True)), weight=weight), "grad_steps": 50}})
    if init:
        recs.append({"event": "init_from", "phase": "B", "step": 0,
                     "source": {"path": SOURCE, "sha256": "ab" * 32, "step": 10681, "name": "stage1_pilot"}})
    recs += [{"event": "jepa_grad", "phase": "B", "step": s, "g_J": g_j(s), "g_R": g_r(s)} for s in grad]
    for k in range(1, steps // 50 + 1):
        recs.append({"event": "log", "phase": "B", "step": 50 * k, "optimizer_steps": 50,
                     "wall_s": 200.0 if k == 1 else 150.0, "loss": loss, "memory_gb": {"max_reserved": peak - 0.1},
                     "jepa": {"windows": 30, "loss": 0.4, "collapse_ratio": collapse}})
    recs += list(extra)
    recs.append({"event": "end", "phase": "B", "step": steps, "memory_gb": {"max_reserved": peak}})
    if verify:
        recs.append({"event": "verify", "phase": "B", "path": f"{folder}/ckpt_B_{steps:06d}.pt", "drift": []})
    write_jsonl(os.path.join(folder, "log.jsonl"), recs)
    return folder


def reference(folder):
    """A reference run: phase A logs, phase B logs at 2.4 s per step to step 400 except 4.8 s for its last
    interval, step 100 logged twice (a crashed call first, 999 s)."""
    recs = [{"event": "log", "phase": "A", "step": 50, "optimizer_steps": 50, "wall_s": 50.0}]
    recs.append({"event": "log", "phase": "B", "step": 100, "optimizer_steps": 50, "wall_s": 999.0})
    for k in range(1, 9):
        recs.append({"event": "log", "phase": "B", "step": 50 * k, "optimizer_steps": 50,
                     "wall_s": 240.0 if k == 8 else 120.0})
    write_jsonl(os.path.join(folder, "log.jsonl"), recs)
    return folder


@pytest.fixture
def config_copy(tmp_path):
    p = str(tmp_path / "stage1_pilot_jepa.yaml")
    shutil.copyfile(CONFIG, p)
    return p


def args(run, out, *more, ref=None):
    return ["--run", str(run), "--out", str(out), "--reference", *([str(ref)] if ref else [])] + list(more)


def expected_lambda():
    kept = [s for s in range(50) if g_j(s) is not None]
    return (sum(g_r(s) for s in kept) / len(kept)) / (sum(g_j(s) for s in kept) / len(kept))


def test_lambda_is_the_ratio_of_the_mean_norms(tmp_path, config_copy):
    sha = CF.sha256(CF.load(config_copy))
    run = smoke(str(tmp_path / "smoke"), sha)
    ref = reference(str(tmp_path / "ref"))
    out = tmp_path / "out"
    assert JL.main(args(run, out, ref=ref)) == 0
    res = json.load(open(out / "lambda.json", encoding="utf-8"))
    lam = res["lambda"]
    assert lam["lambda"] == pytest.approx(expected_lambda(), rel=1e-12)
    assert lam["kept"] == 49 and lam["left_out"] == [7] and lam["zero_g_J"] == [9]
    assert lam["missing"] == [] and lam["repeated"] == [] and len(lam["steps"]) == 50
    assert res["lambda_config"] == float(f"{expected_lambda():.6g}")
    assert res["passed"] and all(c["ok"] for c in res["checks"])
    assert res["smoke"]["source"] == SOURCE and res["smoke"]["config_sha256"] == sha
    assert JL.COLLAPSE_LINE == LP.ALARMS["jepa_collapse"][2]
    # speed: smoke 3.25 s per step over 200 steps, 3.0 after step 50; reference 2.4 on steps 51 to 200, 2.7 whole
    sp = res["speed"]
    assert sp["smoke"]["all"] == {"s_per_step": pytest.approx(3.25), "steps": 200}
    assert sp["smoke"]["after_grad_logging"]["s_per_step"] == pytest.approx(3.0)
    assert sp["smoke"]["after_grad_logging"]["steps"] == 150
    r = sp["references"][JL.tree_rel(ref)]
    assert r["same_steps"] == {"s_per_step": pytest.approx(2.4), "steps": 150}
    assert r["whole_phase_b"] == {"s_per_step": pytest.approx(2.7), "steps": 400}
    assert r["ratio"] == pytest.approx(1.25) and r["projected_s_per_step"] == pytest.approx(3.375)
    assert sp["projection_h"][JL.tree_rel(ref)] == pytest.approx(3.375 * N_B / 3600)
    assert sp["projection_h"]["smoke_rate"] == pytest.approx(3.0 * N_B / 3600)
    for name in ("lambda.json", "lambda.md"):
        text = open(out / name, encoding="utf-8").read()
        assert "\u2014" not in text and "\u2013" not in text
    assert "All checks hold: yes" in open(out / "lambda.md", encoding="utf-8").read()


@pytest.mark.parametrize("case", ["missing", "repeated"])
def test_a_missing_or_repeated_step_fails_the_lambda_check(tmp_path, case):
    grad = [s for s in range(50) if s != 12] if case == "missing" else list(range(50)) + [12]
    run = smoke(str(tmp_path / "smoke"), "x", grad=grad)
    res = JL.run(argparse.Namespace(
        run=run, reference=[], steps=50, expect_steps=200, max_reserved_gb=14.0, out=str(tmp_path / "out"),
        write_config=None, config=CONFIG, e1_job=None, e1_run_dir=JL.E1_RUN))
    assert not res["passed"] and [c["name"] for c in res["checks"] if not c["ok"]] == ["lambda"]
    assert res["lambda"][case] == [12]


@pytest.mark.parametrize("change,failing", [
    ({"extra": [{"event": "alarm", "phase": "B", "step": 100}]}, "no_bad_events"),
    ({"extra": [{"event": "nonfinite", "phase": "B", "step": 3}]}, "no_bad_events"),
    ({"peak": 14.0}, "peak_reserved"),
    ({"collapse": 0.005}, "logs"),
    ({"loss": None}, "logs"),
    ({"steps": 150}, "end"),
    ({"verify": False}, "verify"),
    ({"weight": 2.0}, "start"),
    ({"init": False}, "init_from"),
])
def test_each_check_fails_on_its_defect(tmp_path, change, failing):
    run = smoke(str(tmp_path / "smoke"), "x", **change)
    lam = JL.lambda_of(JL.last_call(JL.read_jsonl(os.path.join(run, "log.jsonl"))[0]))
    chk = JL.checks(JL.last_call(JL.read_jsonl(os.path.join(run, "log.jsonl"))[0]), lam)
    assert [c["name"] for c in chk if not c["ok"]] == [failing]


def test_write_config_changes_the_weight_line_only(tmp_path, config_copy):
    before_text = open(config_copy, encoding="utf-8").read()
    old = CF.load(config_copy)
    run = smoke(str(tmp_path / "smoke"), CF.sha256(old))
    ref = reference(str(tmp_path / "ref"))
    out = tmp_path / "out"
    job = tmp_path / "out" / "e1.job.json"
    JL.main(args(run, out, "--write-config", config_copy, "--e1-job", str(job), ref=ref))
    value = float(f"{expected_lambda():.6g}")
    new = CF.load(config_copy)
    assert new.jepa.weight == value
    assert CF.to_dict(new) == CF.to_dict(replace(old, jepa=replace(old.jepa, weight=value)))
    a, b = before_text.split("\n"), open(config_copy, encoding="utf-8").read().split("\n")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(diff) == 1 and b[diff[0]].strip().startswith(f"weight: {value!r}")
    res = json.load(open(out / "lambda.json", encoding="utf-8"))
    c = res["config"]
    assert c["written"] and c["sha256_before"] == CF.sha256(old) and c["sha256_after"] == CF.sha256(new)
    assert c["weight_before"] == old.jepa.weight and c["weight_after"] == value
    # the E1 job file
    j = json.load(open(job, encoding="utf-8"))
    QJ.normalize(j)
    assert j["name"] == "e1_jepa_train" and j["cls"] == 2 and j["training"] and j["train_phase"] == "B"
    argv = j["argv"]
    assert argv[1] == "scripts/train_stage1.py" and argv[argv.index("--init-from") + 1] == JL.tree_rel(SOURCE)
    assert argv[argv.index("--config") + 1] == JL.tree_rel(config_copy)
    assert argv[argv.index("--run-dir") + 1] == JL.E1_RUN == j["run_dir"]
    assert "--max-steps" not in argv and "--jepa-grad-steps" not in argv and "--resume" not in argv
    assert j["s_per_step"] == pytest.approx(3.375) and j["est_hours"] == pytest.approx(round(3.375 * N_B / 3600, 2))
    assert j["train_steps"] == N_B and f"{JL.tree_rel(out)}/lambda.json" in j["requires"]
    assert res["e1_job"]["path"] == JL.tree_rel(job) and res["e1_job"]["name"] == "e1_jepa_train"
    # the JP2 screen of the proposal's section 3a: the same job cut at 4,000 steps
    screen = tmp_path / "out" / "e1_screen.job.json"
    JL.main(args(run, out, "--config", config_copy, "--e1-job", str(screen), "--e1-max-steps", "4000", ref=ref))
    sj = json.load(open(screen, encoding="utf-8"))
    QJ.normalize(sj)
    assert sj["name"] == "e1_jepa_screen" and sj["train_steps"] == 4000 and sj["argv"][:-2] == argv
    assert sj["argv"][-2:] == ["--max-steps", "4000"] and sj["outputs"] == [f"{JL.E1_RUN}/ckpt_B_004000.pt"]
    assert sj["est_hours"] == pytest.approx(round(3.375 * 4000 / 3600, 2))
    # the second call keeps the first's records: the config write and the first job file beside the second
    res2 = json.load(open(out / "lambda.json", encoding="utf-8"))
    assert res2["config"] == c and res2["e1_job"]["name"] == "e1_jepa_screen"
    assert [(j["path"], j["name"]) for j in res2["e1_jobs"]] == sorted(
        [(JL.tree_rel(job), "e1_jepa_train"), (JL.tree_rel(screen), "e1_jepa_screen")])
    md = open(out / "lambda.md", encoding="utf-8").read()
    assert f"jepa.weight {old.jepa.weight} -> {value} (written at {c['at']})" in md
    assert "job e1_jepa_train" in md and "job e1_jepa_screen" in md
    # a second write leaves the config as it is and the record of the first write in place
    file_sha = JL.sha256_file(config_copy)
    JL.main(args(run, out, "--write-config", config_copy, ref=ref))
    res3 = json.load(open(out / "lambda.json", encoding="utf-8"))
    assert JL.sha256_file(config_copy) == file_sha and res3["config"] == c and "e1_job" not in res3
    assert c["sha256_before"] == CF.sha256(old) and c["sha256_after"] == CF.sha256(new)
    assert [j["name"] for j in res3["e1_jobs"]] == [j["name"] for j in res2["e1_jobs"]]
    assert f"(written at {c['at']})" in open(out / "lambda.md", encoding="utf-8").read()
    # write_config itself reports the value already held
    held = JL.write_config(config_copy, value, CF.sha256(old))
    assert held["written"] is False and held["sha256_before"] == held["sha256_after"] == CF.sha256(new)
    # another smoke log starts the records afresh
    run2 = smoke(str(tmp_path / "smoke2"), CF.sha256(old), collapse=0.7)
    JL.main(args(run2, out, ref=ref))
    res4 = json.load(open(out / "lambda.json", encoding="utf-8"))
    assert res4["passed"] and not {"config", "e1_job", "e1_jobs"} & set(res4)
    # a config the smoke did not run is refused
    other = str(tmp_path / "other.yaml")
    shutil.copyfile(CONFIG, other)
    with pytest.raises(ValueError, match="not the configuration the smoke ran"):
        JL.write_config(other, value, "0" * 64)


def test_a_failing_smoke_writes_no_config(tmp_path, config_copy):
    sha = CF.sha256(CF.load(config_copy))
    before = JL.sha256_file(config_copy)
    run = smoke(str(tmp_path / "smoke"), sha, extra=[{"event": "alarm", "phase": "B", "step": 100}])
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="no_bad_events"):
        JL.main(args(run, out, "--write-config", config_copy, "--e1-job", str(out / "e1.json")))
    assert JL.sha256_file(config_copy) == before and not os.path.exists(out / "e1.json")
    res = json.load(open(out / "lambda.json", encoding="utf-8"))
    assert res["passed"] is False and "config" not in res
    assert "All checks hold: NO" in open(out / "lambda.md", encoding="utf-8").read()


def test_the_weight_line_is_that_of_the_jepa_section():
    text = open(CONFIG, encoding="utf-8").read()
    i, indent = JL.weight_line(text)
    assert text.split("\n")[i].strip().startswith("weight:") and indent == "  "
    assert text.split("\n")[i - 1].strip() == "enabled: true"
    with pytest.raises(ValueError, match="no top-level jepa"):
        JL.weight_line("name: x\nlosses:\n  weights: {}\n")
    with pytest.raises(ValueError, match="2 weight lines"):
        JL.weight_line("jepa:\n  weight: 1.0\n  weight: 2.0\n")
