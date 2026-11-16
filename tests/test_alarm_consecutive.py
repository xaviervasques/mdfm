"""schedule.alarm_consecutive (task S2BA; mdfm/train/config.py ALARM_CONSECUTIVE_DOC, mdfm/train/loop.py check_alarms
and alarm_streaks): an alarm read on k of its reads in a row, off by default, every config keeping its SHA-256.

The loop tests call Trainer.check_alarms on a bare Trainer (no model, no store), as tests/test_hsic_strata.py does; the
log records are written as Trainer.log writes them."""
import glob
import json
import os
import types
from dataclasses import replace

import pytest

from mdfm.train import config as CF
from mdfm.train import loop as LP

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2A_SHA = "db4029e1a1710bc5af663cebd1ba9cadd1c789c9a093367075813fb29d713f22"      # rule.txt, Config line
S2B_SHA = "e31ab7098cebcee984730926287b41eed9ad76e1a239215ef02af6cf1ff01def"      # amendment 2, Config line
PILOT_SHA = "7f761b1ac425a2dfefc4e546908fa766e2545d915f9c2a88dcb8d199f140b2b5"    # v1's start event
# the configs written to set the field, each pinned by its own test: configs/stage2_s2c.yaml (task S2C, amendment 3 of
# the stage-2 read rule; tests/test_stage2_s2c.py) and configs/stage2_s2d.yaml (task S2D, amendment 4, a DRAFT;
# tests/test_stage2_s2d.py)
SETS_THE_FIELD = ("stage2_s2c.yaml", "stage2_s2d.yaml")


def bare(tmp_path, consecutive=None, stop=False, phase="C", step=1000):
    """A Trainer without model or store: what check_alarms reads (cfg, sched, run folder, log, echo)."""
    cfg = CF.RunConfig()
    extra = {} if consecutive is None else {"alarm_consecutive": dict(consecutive)}
    sc = replace(cfg.schedule, stop_on_alarm=stop, **extra)
    tr = LP.Trainer.__new__(LP.Trainer)
    tr.cfg = replace(cfg, schedule=sc)
    tr.sched = types.SimpleNamespace(in_phase=phase, step=step)
    tr.run_dir = str(tmp_path)
    tr.log_path = str(tmp_path / "log.jsonl")
    tr.echo = lambda *a, **k: None
    tr.checkpoints = []
    tr.alarm_streak, tr.alarm_streak_phase = None, None
    return tr


def beta(v):
    return {"event": "log", "health": {"beta_error": v, "decoder": [{"entropy": 2.0}], "logvar_z": -4.0}}


def at(tr, step, rec, source="log"):
    """The loop's order: the record logged at `step`, then check_alarms on it."""
    tr.sched.step = step
    if source == "log":
        tr.log(dict(rec, phase=tr.sched.in_phase, step=step))
    else:
        tr.log({"event": "dev", "phase": tr.sched.in_phase, "step": step, "dev": rec})
    return LP.Trainer.check_alarms(tr, rec, source)


def events(tmp_path, name):
    with open(tmp_path / "log.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f if json.loads(x).get("event") == name]


# ---------------------------------------------------------------- the config field
def test_the_alarm_names_copy_loop_alarms():
    assert set(CF.ALARM_NAMES) == set(LP.ALARMS) and len(CF.ALARM_NAMES) == len(LP.ALARMS)
    assert LP.ALARMS["beta_error"] == ("log", "above", 2.0, 1000)          # the default alarms are unchanged


def test_the_field_is_off_by_default_and_every_config_keeps_its_sha256():
    assert CF.ScheduleSection().alarm_consecutive == {}
    assert "alarm_consecutive" not in CF.canonical_json(CF.RunConfig())
    seen = 0
    for p in sorted(glob.glob(os.path.join(TREE, "configs", "*.yaml"))):
        if os.path.basename(p).startswith("._") or os.path.basename(p) in SETS_THE_FIELD:
            continue
        try:
            cfg = CF.load(p)
        except ValueError:                       # configs/g12_baseline.yaml is not a RunConfig
            continue
        assert cfg.schedule.alarm_consecutive == {} and "alarm_consecutive" not in CF.canonical_json(cfg), p
        seen += 1
    # the public copy carries the three open configurations only (the builder's CONFIGS_LEFT_OUT)
    assert seen >= (3 if os.path.isfile(os.path.join(TREE, "PUBLIC_TREE.md")) else 9)
    assert CF.sha256(CF.load(os.path.join(TREE, "configs", "stage2_s2a.yaml"))) == S2A_SHA
    assert CF.sha256(CF.load(os.path.join(TREE, "configs", "stage2_s2b.yaml"))) == S2B_SHA
    assert CF.sha256(CF.load(os.path.join(TREE, "configs", "stage1_pilot.yaml"))) == PILOT_SHA


def test_a_set_field_enters_the_hash_and_round_trips(tmp_path):
    base = CF.load(os.path.join(TREE, "configs", "stage2_s2b.yaml"))
    cfg = replace(base, schedule=replace(base.schedule, alarm_consecutive={"beta_error": 3}))
    assert CF.problems(cfg) == []
    assert CF.sha256(cfg) != S2B_SHA and '"alarm_consecutive":{"beta_error":3}' in CF.canonical_json(cfg)
    back = CF.from_json(CF.canonical_json(cfg))
    assert back == cfg and CF.sha256(back) == CF.sha256(cfg)
    # YAML: the field read from a file
    text = open(os.path.join(TREE, "configs", "stage2_s2b.yaml"), encoding="utf-8").read()
    text = text.replace("  stop_on_alarm: true", "  stop_on_alarm: true\n  alarm_consecutive: {beta_error: 3}", 1)
    p = tmp_path / "s2c.yaml"
    p.write_text(text, encoding="utf-8")
    assert CF.load(str(p)) == cfg


@pytest.mark.parametrize("bad", [{"beta_errors": 3}, {"beta_error": 0}, {"beta_error": True}, {"beta_error": 2.5},
                                 {"beta_error": "3"}])
def test_the_field_refuses_unknown_names_and_bad_counts(bad):
    cfg = replace(CF.RunConfig(), schedule=replace(CF.ScheduleSection(), alarm_consecutive=bad))
    assert any("alarm_consecutive" in p for p in CF.problems(cfg))
    with pytest.raises(ValueError):
        CF.check(cfg)


# ---------------------------------------------------------------- the loop
def test_the_default_reads_one_record_as_before(tmp_path):
    tr = bare(tmp_path, stop=True)
    assert at(tr, 8500, beta(1.18)) == []
    tr.checkpoint = lambda: tr.checkpoints.append("ckpt")
    with pytest.raises(LP.AlarmError, match="beta_error"):
        at(tr, 8550, beta(2.0431))
    alarm = events(tmp_path, "alarm")
    want = [{"name": "beta_error", "value": 2.0431, "above": 2.0, "from_step": 1000}]
    assert len(alarm) == 1 and alarm[0]["alarms"] == want
    assert events(tmp_path, "alarm_watch") == [] and tr.checkpoints == ["ckpt"]


def test_k_consecutive_records_raise_the_alarm_and_a_record_below_resets(tmp_path):
    tr = bare(tmp_path, {"beta_error": 3})
    seq = [(950, 2.5), (1000, 2.1), (1050, 2.1), (1100, 1.0), (1150, 2.1), (1200, None), (1250, 2.2), (1300, 2.3)]
    got = {s: at(tr, s, beta(v)) for s, v in seq}
    assert all(got[s] == [] for s, _ in seq[:-1])
    assert [(h["name"], h["consecutive"], h["of"]) for h in got[1300]] == [("beta_error", 3, 3)]
    watch = events(tmp_path, "alarm_watch")
    got_watch = [(w["step"], w["alarms"][0]["consecutive"]) for w in watch]
    assert got_watch == [(1000, 1), (1050, 2), (1150, 1), (1250, 2)]
    assert [a["step"] for a in events(tmp_path, "alarm")] == [1300]
    # the other log alarms are still read on one record
    hit = at(tr, 1350, {"event": "log", "health": {"beta_error": 1.0, "decoder": [{"entropy": 0.1}]}})
    assert [h["name"] for h in hit] == ["attention_entropy"] and "consecutive" not in hit[0]


def test_k_consecutive_stops_the_run_at_the_kth(tmp_path):
    tr = bare(tmp_path, {"beta_error": 2}, stop=True)
    tr.checkpoint = lambda: tr.checkpoints.append("ckpt")
    assert at(tr, 8550, beta(2.04)) == [] and tr.checkpoints == []
    with pytest.raises(LP.AlarmError, match="beta_error"):
        at(tr, 8600, beta(2.01))
    assert tr.checkpoints == ["ckpt"]


def test_the_run_of_reads_survives_a_resume(tmp_path):
    tr = bare(tmp_path, {"beta_error": 3})
    for s, v in ((1000, 2.1), (1050, 2.2)):
        assert at(tr, s, beta(v)) == []
    # a later call's records (a crashed call beyond the checkpoint) and another phase's are not read
    with open(tmp_path / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(beta(1.0), phase="C", step=1200)) + "\n")
        f.write(json.dumps(dict(beta(1.0), phase="B", step=1060)) + "\n")
    again = bare(tmp_path, {"beta_error": 3})       # a new Trainer on the run folder (resume)
    assert LP.alarm_streaks(again.log_path, "C", 1100, {"beta_error": 3}) == {"beta_error": 2}
    hits = at(again, 1100, beta(2.3))
    assert [(h["name"], h["consecutive"]) for h in hits] == [("beta_error", 3)]
    assert LP.alarm_streaks(str(tmp_path / "none.jsonl"), "C", 1100, {"beta_error": 3}) == {"beta_error": 0}


def test_a_dev_alarm_reads_its_dev_events(tmp_path):
    tr = bare(tmp_path, {"spread_ratio": 2}, step=2000)
    low = {"model": {"spread_ratio": 0.2, "joint_error_mm": 100.0, "joint_error_z0_mm": 150.0}}
    assert at(tr, 2000, low, "dev") == []
    assert [h["name"] for h in at(tr, 3000, low, "dev")] == ["spread_ratio"]
    assert LP.alarm_streaks(tr.log_path, "C", 4000, {"spread_ratio": 2}) == {"spread_ratio": 2}


def test_alarm_values_read_what_check_alarms_read():
    rec = {"health": {"beta_error": 1.5, "logvar_z": -3.0, "decoder": [{"entropy": 1.2}, {"entropy": 0.9}]},
           "jepa": {"collapse_ratio": 0.4}}
    assert LP.alarm_values(rec, "log") == {"attention_entropy": 0.9, "beta_error": 1.5, "logvar_z": -3.0,
                                           "jepa_collapse": 0.4}
    d = LP.alarm_values({"model": {"joint_error_mm": 10.0, "joint_error_z0_mm": 12.5}}, "dev")
    assert d["z0_margin_mm"] == pytest.approx(2.5) and d["spread_ratio"] is None
    assert LP.alarm_values({}, "other") == {}
