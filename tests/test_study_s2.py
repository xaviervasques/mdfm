"""Tests of scripts/study_s2.py: the Holm adjustment of NR-7.3 and the slot 6 record while the posture is open."""
import importlib.util
import os

import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("study_s2_t", os.path.join(TREE, "scripts", "study_s2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S2 = _script()


def test_holm_adjusted_matches_the_step_down_form():
    p = [0.01, 0.04, 0.03, 0.005, 1.0, 0.2]
    adj = S2.holm_adjusted(p)
    # ordered 0.005, 0.01, 0.03, 0.04, 0.2, 1.0 -> 6x, 5x, 4x, 3x, 2x, 1x with the running maximum
    assert adj == pytest.approx([0.05, 0.12, 0.12, 0.03, 1.0, 0.4])


def test_a_p_of_one_reads_one_whatever_the_others():
    for others in ([0.0] * 5, [1e-4, 0.5, 0.9, 0.01, 0.02]):
        assert S2.holm_adjusted(others + [1.0])[-1] == 1.0


def test_slot6_record_when_the_posture_is_open():
    rec = S2.slot6_not_run({"settled": False}, {"x": "y"}, {"head": "abc"})
    assert rec["outcome"] == "not run" and rec["p_raw"] == 1.0 and rec["holm_adjusted_p"] == 1.0
    assert rec["interval"] is None and rec["k"] is None and rec["m"] == 6 and rec["alpha"] == 0.05


def test_s2_refuses_to_run_its_primary_when_the_posture_is_settled(monkeypatch, tmp_path):
    monkeypatch.setattr(S2.S1, "load_posture", lambda: {"settled": True, "probe_v2_json_sha256": "h"})
    monkeypatch.setattr(S2.S1, "sha256_file", lambda p: S2.NR_SHA256 if p.endswith("NEURAL_RULES_DRAFT.md") else
                        (S2.S1.REGISTERED.get(os.path.relpath(p, TREE).replace(os.sep, "/"), "h")))
    with pytest.raises(SystemExit, match="settled"):
        S2.main(["--moveatlas", str(tmp_path)])
