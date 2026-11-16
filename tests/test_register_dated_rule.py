"""Tests of scripts/register_dated_rule.py on a temporary rule file, registry and read log (never the tree's own)."""
import csv
import hashlib
import os
import sys

import pytest

from mdfm import paths
from mdfm.data.canonical import Refused
from mdfm.gates import common

SCRIPTS = os.path.join(paths.TREE, "scripts")
if SCRIPTS not in sys.path:
    sys.path.append(SCRIPTS)
import register_dated_rule as R  # noqa: E402

TEXT = "# G9 rule, dated\n\nStatus: DRAFT for the owner, not approved, not registered.\n\nBody line.\nStatus: later\n"


def _setup(tmp_path, text=TEXT):
    p = tmp_path / "G9_RULE_2026-11-12.md"
    p.write_bytes(text.encode("utf-8"))
    reg = tmp_path / "gate_rule_hashes.csv"
    log = tmp_path / "lockbox_reads.csv"
    return str(p), str(reg), str(log), hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_approves_only_the_status_line_and_registers(tmp_path):
    p, reg, log, sha = _setup(tmp_path)
    res = R.approve("G9", p, sha[:12], "2026-11-13", registry_path=reg, log_path=log)
    new = open(p, "rb").read().decode("utf-8")
    assert new == TEXT.replace("Status: DRAFT for the owner, not approved, not registered.",
                               "Status: APPROVED by the owner on 2026-11-13.", 1)
    assert "Status: later" in new                       # a later status-like line is left alone
    assert res["row"]["sha256"] == hashlib.sha256(new.encode("utf-8")).hexdigest() == res["sha_after"]
    assert res["sha_before"] == sha and res["row"]["approved_by"] == "owner" and res["row"]["version"] == "1"
    again = R.approve("G9", p, sha[:12], "2026-11-13", registry_path=reg, log_path=log)
    assert again["already"] and again["row"]["sha256"] == res["sha_after"]
    with open(reg, encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_refuses_another_text_and_leaves_it(tmp_path):
    p, reg, log, sha = _setup(tmp_path)
    with pytest.raises(Refused):
        R.approve("G9", p, "0" * 12, "2026-11-13", registry_path=reg, log_path=log)
    assert open(p, "rb").read() == TEXT.encode("utf-8") and not os.path.exists(reg)
    with pytest.raises(ValueError):
        R.approve("G9", p, sha[:8], "2026-11-13", registry_path=reg, log_path=log)


def test_restores_the_file_when_registration_is_refused(tmp_path):
    p, reg, log, sha = _setup(tmp_path)
    common.ensure_read_log(log)
    common._append_row(log, common.READ_LOG_COLUMNS, {"read_at": "x", "gate": "G9"})
    with pytest.raises(Refused):
        R.approve("G9", p, sha[:12], "2026-11-13", registry_path=reg, log_path=log)
    assert open(p, "rb").read() == TEXT.encode("utf-8")


def test_refuses_a_file_without_a_draft_status(tmp_path):
    p, reg, log, sha = _setup(tmp_path, "# x\n\nStatus: adopted by the session.\n")
    with pytest.raises(Refused):
        R.approve("G9", p, sha[:12], "2026-11-13", registry_path=reg, log_path=log)
