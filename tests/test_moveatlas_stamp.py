"""Tests of scripts/moveatlas_stamp.py with subprocess mocked: the three git commands carry the safe.directory
and -C options of the MoveAtlas root and GIT_OPTIONAL_LOCKS=0, and nothing else runs; the head, the diff
SHA-256 and the untracked list come from their outputs (renames and modified files are not untracked); a
failing git command raises; the manifest_tiers note sets its build date against batch 0002B2; --out refuses a
path inside MoveAtlas.
"""
import hashlib
import importlib.util
import json
import os
import subprocess

import pytest

from mdfm import paths


def _script():
    spec = importlib.util.spec_from_file_location("moveatlas_stamp_script",
                                                  os.path.join(paths.TREE, "scripts", "moveatlas_stamp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()
ROOT = "<moveatlas>"
HEAD = "f2e6f99271b1bc10a578a67d368fa850ae03f3eb"
DIFF = b"diff --git a/INDEX.md b/INDEX.md\n--- a/INDEX.md\n+++ b/INDEX.md\n@@ -1 +1 @@\n-old\n+new\n"
STATUS = (b" M INDEX.md\0?? set_stated_fps.py\0R  docs/new name.md\0docs/old.md\0"
          b"?? tests/test_amass_builder.py\0?? paper/package final.zip\0 D gone.txt\0")


class FakeGit:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, argv, capture_output=False, env=None, timeout=None, **kw):
        self.calls.append((list(argv), dict(env or {})))
        command = tuple(argv[5:])
        if self.fail and command[0] == self.fail:
            return subprocess.CompletedProcess(argv, 128, b"", b"fatal: not a git repository")
        out = {("rev-parse", "HEAD"): HEAD.encode() + b"\n", ("diff",): DIFF}.get(command)
        if command[0] == "status":
            out = STATUS
        assert out is not None, f"unexpected git command {command}"
        return subprocess.CompletedProcess(argv, 0, out, b"")


def _provenance(tree, run_at):
    d = os.path.join(tree, "reports", "manifest_tiers")
    os.makedirs(d)
    with open(os.path.join(d, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"script": "build_tiers_and_lockbox.py", "run_at": run_at,
                   "moveatlas_head": "f2e6f99 2026-09-24 Registry corrections"}, f)


def test_stamp_from_git_outputs(monkeypatch, tmp_path):
    fake = FakeGit()
    monkeypatch.setattr(S.subprocess, "run", fake)
    _provenance(str(tmp_path), "2026-09-28T11:18:21")
    s = S.stamp(ROOT + "/", tree=str(tmp_path))
    assert s["root"] == ROOT
    assert s["head"] == HEAD
    assert s["diff_sha256"] == hashlib.sha256(DIFF).hexdigest()
    assert s["diff_bytes"] == len(DIFF)
    assert s["untracked"] == ["paper/package final.zip", "set_stated_fps.py", "tests/test_amass_builder.py"]
    commands = [tuple(argv[5:]) for argv, _ in fake.calls]
    assert commands == [("rev-parse", "HEAD"), ("diff",),
                        ("status", "--porcelain=v1", "-z", "--untracked-files=all")]
    for argv, env in fake.calls:
        assert argv[:5] == ["git", "-c", f"safe.directory={ROOT}", "-C", ROOT]
        assert env["GIT_OPTIONAL_LOCKS"] == "0"
    t = s["manifest_tiers"]
    assert t["built_on"] == "2026-09-28" and t["before_batch"] is True and t["moveatlas_head"] == "f2e6f99"
    assert "built on 2026-09-28" in s["note"] and "before batch 0002B2" in s["note"]
    assert "2026-09-29" in s["note"]


def test_root_from_environment(monkeypatch, tmp_path):
    fake = FakeGit()
    monkeypatch.setattr(S.subprocess, "run", fake)
    monkeypatch.setenv("MOVEATLAS_ROOT", "E:\\copy\\moveatlas\\")
    s = S.stamp(tree=str(tmp_path))
    assert s["root"] == "E:/copy/moveatlas"
    assert all(argv[2] == "safe.directory=E:/copy/moveatlas" and argv[4] == "E:/copy/moveatlas"
               for argv, _ in fake.calls)


@pytest.mark.parametrize("command", ["rev-parse", "diff", "status"])
def test_failing_git_raises(monkeypatch, tmp_path, command):
    monkeypatch.setattr(S.subprocess, "run", FakeGit(fail=command))
    with pytest.raises(RuntimeError, match="exited 128"):
        S.stamp(ROOT, tree=str(tmp_path))


def test_untracked_parsing():
    assert S.untracked(b"") == []
    assert S.untracked(b"?? a b.py\0") == ["a b.py"]
    # the source of a rename is skipped even when it looks like an untracked entry
    assert S.untracked(b"R  new.py\0?? fake.py\0?? real.py\0") == ["real.py"]
    assert S.untracked(b"C  copy.py\0orig.py\0 M x.py\0") == []


def test_manifest_tiers_after_batch_and_missing(tmp_path):
    _provenance(str(tmp_path), "2026-10-02T09:00:00")
    t = S.manifest_tiers_build(str(tmp_path))
    assert t["before_batch"] is False and "on or after" in t["note"]
    m = S.manifest_tiers_build(str(tmp_path / "nowhere"))
    assert m["built_on"] is None and "not readable" in m["note"]


def test_main_writes_out_and_refuses_moveatlas(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(S.subprocess, "run", FakeGit())
    out = tmp_path / "stamp" / "moveatlas.json"
    assert S.main(["--moveatlas", ROOT, "--out", str(out)]) == 0
    with open(out, encoding="utf-8") as f:
        written = json.load(f)
    assert written["head"] == HEAD and written["untracked"][0] == "paper/package final.zip"
    assert json.loads(capsys.readouterr().out)["diff_sha256"] == written["diff_sha256"]
    with pytest.raises(SystemExit, match="refusing"):
        S.main(["--moveatlas", ROOT, "--out", ROOT + "/reports/stamp.json"])
