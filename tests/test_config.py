"""Tests of the run configuration, mdfm/train/config.py (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.6 and
section 5), for the optional jepa section of task J2a (docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 2).

- The pilot configs keep the config SHA-256 recorded before the section existed (Trainer.resume refuses a
  checkpoint whose config hash moved): configs/stage1_pilot.yaml 7f761b1a... and configs/stage1_pilot_s2.yaml
  a81f22e9..., both recorded on 2026-10-05 before the first J2a edit.
- canonical_json leaves the jepa section out while it equals its defaults (absent, empty, or written with its
  default values hash alike); any other value puts the whole section in, and only it moves; a YAML file with
  the section loads; round trips through JSON keep it.
- The section's values are checked: unknown keys refused, shares, block lengths, EMA decays, predictor sizes
  and the weight validated by problems().
"""
import os
from dataclasses import replace

import pytest

from mdfm import paths
from mdfm.train import config as CF

PILOT_SHA256 = {"stage1_pilot.yaml": "7f761b1ac425a2dfefc4e546908fa766e2545d915f9c2a88dcb8d199f140b2b5",
                "stage1_pilot_s2.yaml": "a81f22e914d39dfd9cacee422dcfd3cd2f05cf36d9df7741abf47e8500c7470c"}


def _pilot(name="stage1_pilot.yaml"):
    return CF.load(os.path.join(paths.TREE, "configs", name))


@pytest.mark.parametrize("name", sorted(PILOT_SHA256))
def test_the_pilot_configs_keep_their_sha256(name):
    cfg = _pilot(name)
    assert CF.sha256(cfg) == PILOT_SHA256[name]
    assert cfg.jepa == CF.JepaSection() and not cfg.jepa.enabled
    assert "jepa" not in CF.as_json_dict(cfg) and '"jepa"' not in CF.canonical_json(cfg)
    assert CF.from_json(CF.canonical_json(cfg)) == cfg


def test_the_jepa_section_is_left_out_at_its_defaults():
    base = _pilot()
    d = CF.to_dict(base)
    assert "jepa" not in d
    for written in ({}, {"enabled": False}, CF.to_dict(CF.JepaSection())):
        same = CF.from_dict(dict(d, jepa=written))
        assert same == base and CF.sha256(same) == CF.sha256(base)
    off = CF.from_dict(dict(d, jepa={"enabled": False, "weight": 0.5}))
    assert CF.to_dict(off)["jepa"]["weight"] == 0.5 and CF.sha256(off) != CF.sha256(base)
    on = replace(base, jepa=CF.JepaSection(enabled=True))
    d_on = CF.to_dict(on)
    assert set(d_on["jepa"]) == {f for f in CF.to_dict(CF.JepaSection())} and d_on["jepa"]["enabled"] is True
    assert {k: v for k, v in d_on.items() if k != "jepa"} == d
    assert CF.from_json(CF.canonical_json(on)) == on and CF.sha256(on) != CF.sha256(base)
    assert CF.check(on) is on
    assert "enabled" in CF.to_dict(CF.JepaSection())          # a section alone is never shortened


def test_a_yaml_file_with_the_jepa_section_loads(tmp_path):
    with open(os.path.join(paths.TREE, "configs", "stage1_pilot.yaml"), encoding="utf-8") as f:
        text = f.read()
    p = tmp_path / "with_jepa.yaml"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\njepa:\n  enabled: true\n  weight: 0.25\n  pred_depth: 2\n")
    cfg = CF.load(p)
    assert cfg.jepa == CF.JepaSection(enabled=True, weight=0.25, pred_depth=2)
    assert replace(cfg, jepa=CF.JepaSection()) == _pilot()


@pytest.mark.parametrize("bad", [{"weight": -1.0}, {"mask_min": 0.4}, {"mask_min": 0.0}, {"mask_max": 1.0},
                                 {"block_min": 0}, {"block_min": 9}, {"max_blocks": 0}, {"min_patches": 0},
                                 {"ema_start": 1.1, "ema_end": 1.0}, {"ema_start": 0.999, "ema_end": 0.99},
                                 {"pred_heads": 3}, {"pred_width": 12, "pred_heads": 4}, {"pred_depth": 0}])
def test_jepa_values_are_checked(bad):
    cfg = replace(_pilot(), jepa=CF.JepaSection(enabled=True, **bad))
    assert any(p.startswith("jepa") for p in CF.problems(cfg))
    with pytest.raises(ValueError):
        CF.check(cfg)


def test_unknown_jepa_keys_are_refused():
    d = CF.to_dict(_pilot())
    with pytest.raises(ValueError, match="unknown keys"):
        CF.from_dict(dict(d, jepa={"enabled": True, "lambda": 1.0}))
    with pytest.raises(ValueError):
        CF.from_dict(dict(d, jepa={"enabled": "yes"}))
