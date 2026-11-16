"""Tests of scripts/run_suite.py on a fake tree (a tests folder of three files, scripts, configs and a queue):
a fresh lock of another task leaves its test file out, a lock older than 3 h and the caller's own lock leave
nothing out, the pass writes a green marker with the code hash, the SHA-256 of every scripts/*.py and
configs/*.yaml (sidecars and subfolders left out), the pass count and the files left out; a failure exits
non-zero with no marker; the suite lock waits for a live holder and takes over a dead or empty one, and its
removal survives another handle holding the file open (Windows); without --task every lock counts as another
task's; the test environment never carries an empty CUDA_VISIBLE_DEVICES.
"""
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from mdfm import paths


def _script():
    spec = importlib.util.spec_from_file_location("run_suite_script",
                                                  os.path.join(paths.TREE, "scripts", "run_suite.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()
CODE = "c0de" * 16


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def tree(tmp_path):
    t = str(tmp_path / "tree")
    _write(os.path.join(t, "tests", "test_a.py"), "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n")
    _write(os.path.join(t, "tests", "test_b.py"), "def test_broken():\n    assert False\n")
    _write(os.path.join(t, "tests", "test_c.py"), "def test_three():\n    pass\n")
    _write(os.path.join(t, "scripts", "x.py"), "X = 1\n")
    _write(os.path.join(t, "scripts", "._x.py"), "sidecar\n")
    _write(os.path.join(t, "scripts", "sub", "z.py"), "Z = 1\n")
    _write(os.path.join(t, "configs", "y.yaml"), "a: 1\n")
    _write(os.path.join(t, "configs", "notes.txt"), "not a config\n")
    return t


def _lock(tree, name, text, age_h=0.0):
    p = os.path.join(tree, "cache", "runs", "queue", "locks", name)
    _write(p, text)
    if age_h:
        t = time.time() - age_h * 3600.0
        os.utime(p, (t, t))
    return p


def _run(tree, task=None):
    out = io.StringIO()
    rc = S.run(tree, task=task, python=sys.executable, wait_s=5.0, poll_s=0.05,
               code_sha=lambda t: CODE, cuda_check=lambda env, python: None, out=out)
    return rc, out.getvalue()


def _markers(tree):
    green = os.path.join(tree, "cache", "runs", "queue", "green")
    return sorted(os.path.join(green, n) for n in os.listdir(green)) if os.path.isdir(green) else []


def test_lock_exclusion_and_marker(tree):
    _lock(tree, "T02.lock", "paths: scripts/x.py, tests/test_b.py, tests/test_missing.py\n")
    _lock(tree, "T03.lock", "tests\\test_c.py\n", age_h=4.0)          # stale: test_c still runs
    _lock(tree, "T00.lock", "tests/test_a.py\n")                        # the caller's own lock
    _lock(tree, "._T09.lock", "tests/test_a.py\n")                      # an AppleDouble sidecar
    rc, log = _run(tree, task="T00")
    assert rc == 0, log
    [path] = _markers(tree)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["format"] == "mdfm-green/1" and m["task"] == "T00"
    assert m["code_sha256"] == CODE
    assert m["files"] == {"configs/y.yaml": _sha(os.path.join(tree, "configs", "y.yaml")),
                          "scripts/x.py": _sha(os.path.join(tree, "scripts", "x.py"))}
    assert m["passed"] == 3 and m["counts"]["passed"] == 3 and "failed" not in m["counts"]
    assert m["left_out"] == ["tests/test_b.py"]
    assert m["locks_used"] == {"T02.lock": ["test_b.py", "test_missing.py"]}
    assert list(m["locks_stale"]) == ["T03.lock"] and m["locks_stale"]["T03.lock"] >= 3.9
    assert m["locks_own"] == ["T00.lock"]
    assert m["changed_during_run"] == []
    assert m["env"] == {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2"}
    assert m["argv"][1:] == ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "--ignore=tests/test_b.py"]
    assert len(m["tests_sha256"]) == 64
    assert os.path.basename(path)[:-5].replace("_", "").isdigit()
    assert not os.path.exists(os.path.join(tree, "cache", "runs", "queue", "suite.lock"))


def test_failure_exits_nonzero_without_marker(tree):
    _lock(tree, "T02.lock", "tests/test_b.py\n")
    rc, log = _run(tree, task="T02")              # its own lock: test_b runs and fails
    assert rc != 0 and "FAIL" in log
    assert _markers(tree) == []
    assert not os.path.exists(os.path.join(tree, "cache", "runs", "queue", "suite.lock"))


def test_script_changed_during_run_is_not_vouched(tree, monkeypatch):
    real = S.run_pytest

    def edit_then_run(argv, cwd, env, out=None):
        _write(os.path.join(cwd, "scripts", "x.py"), "X = 2\n")
        return real(argv, cwd, env, out)

    monkeypatch.setattr(S, "run_pytest", edit_then_run)
    _lock(tree, "T02.lock", "tests/test_b.py\n")
    rc, log = _run(tree)
    assert rc == 0, log
    [path] = _markers(tree)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["changed_during_run"] == ["scripts/x.py"] and list(m["files"]) == ["configs/y.yaml"]


def test_code_changed_during_run_writes_no_marker(tree):
    shas = iter(["a" * 64, "b" * 64])
    _lock(tree, "T02.lock", "tests/test_b.py\n")
    rc = S.run(tree, python=sys.executable, wait_s=5.0, poll_s=0.05, code_sha=lambda t: next(shas),
               cuda_check=lambda env, python: None, out=io.StringIO())
    assert rc == S.EXIT_CODE_CHANGED and _markers(tree) == []


def test_cuda_visible_stops_before_the_run(tree):
    def visible(env, python):
        raise S.CudaVisible("CUDA is visible")

    rc = S.run(tree, python=sys.executable, code_sha=lambda t: CODE, cuda_check=visible, out=io.StringIO())
    assert rc == S.EXIT_CUDA and _markers(tree) == []


def test_suite_env_never_empty():
    env = S.suite_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "16", "PATH": "p"})
    assert env["CUDA_VISIBLE_DEVICES"] == "-1" and env["OMP_NUM_THREADS"] == "2" and env["PATH"] == "p"


def test_parse_counts():
    lines = ["....F", "FAILED tests/test_b.py::test_broken", "1 failed, 12 passed, 2 skipped, 3 warnings in 1.50s"]
    counts, summary = S.parse_counts(lines)
    assert counts == {"failed": 1, "passed": 12, "skipped": 2, "warnings": 3}
    assert summary.startswith("1 failed")
    assert S.parse_counts(["no tests ran in 0.01s"]) == ({}, "")


def _suite_lock(tmp_path, info):
    p = str(tmp_path / "queue" / "suite.lock")
    _write(p, json.dumps(info))
    return p


def test_suite_lock_waits_for_a_live_holder(tmp_path):
    p = _suite_lock(tmp_path, {"pid": os.getpid(), "host": S.socket.gethostname(), "start_epoch": time.time()})
    with pytest.raises(S.LockTimeout):
        S.SuiteLock(p, timeout_s=0.3, poll_s=0.05, out=io.StringIO()).acquire()
    assert os.path.exists(p)                                        # never removed from a live holder


@pytest.mark.parametrize("case", ["dead", "reused", "foreign_old"])
def test_suite_lock_takes_over_a_stale_holder(tmp_path, case):
    host = S.socket.gethostname()
    if case == "dead":
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        info = {"pid": child.pid, "host": host, "start_epoch": time.time()}
    elif case == "reused":                                          # the PID belongs to a later process
        info = {"pid": os.getpid(), "host": host, "start_epoch": time.time() - 10 * 365 * 86400.0}
    else:                                                           # another host, older than 3 h
        info = {"pid": 1, "host": "another-host", "start_epoch": time.time() - 4 * 3600.0}
    p = _suite_lock(tmp_path, info)
    with S.SuiteLock(p, task="T00", timeout_s=2.0, poll_s=0.05, out=io.StringIO()) as lock:
        with open(p, encoding="utf-8") as f:
            held = json.load(f)
        assert held["pid"] == os.getpid() and held["task"] == "T00" and lock.info == held
    assert not os.path.exists(p)


def test_suite_lock_of_a_fresh_foreign_holder_is_kept(tmp_path):
    p = _suite_lock(tmp_path, {"pid": 1, "host": "another-host", "start_epoch": time.time()})
    with pytest.raises(S.LockTimeout):
        S.SuiteLock(p, timeout_s=0.2, poll_s=0.05, out=io.StringIO()).acquire()


def test_suite_lock_takes_over_an_empty_lock(tmp_path):
    p = _suite_lock(tmp_path, {})
    _write(p, "")                                                   # a crash between creating and writing it
    with S.SuiteLock(p, timeout_s=2.0, poll_s=0.05, out=io.StringIO()) as lock:
        assert lock.info["pid"] == os.getpid()
    assert not os.path.exists(p)


def _close_later(f, delay_s=0.3):
    t = threading.Timer(delay_s, f.close)
    t.start()
    return t


def test_lock_removal_waits_for_a_reader(tmp_path):
    """On Windows a file held open by another handle cannot be removed (a waiter reads the lock at each poll):
    release and takeover retry instead of raising PermissionError."""
    p = str(tmp_path / "queue" / "suite.lock")
    lock = S.SuiteLock(p, timeout_s=2.0, poll_s=0.05, out=io.StringIO()).acquire()
    reader = open(p, encoding="utf-8")
    t = _close_later(reader)
    lock.release()
    t.join()
    assert not os.path.exists(p) and lock.info is None

    _write(p, json.dumps({"pid": os.getpid(), "host": S.socket.gethostname(),
                          "start_epoch": time.time() - 10 * 365 * 86400.0}))        # a dead holder (reused PID)
    reader = open(p, encoding="utf-8")
    t = _close_later(reader)
    with S.SuiteLock(p, task="T00", timeout_s=3.0, poll_s=0.05, out=io.StringIO()) as lock:
        assert lock.info["task"] == "T00"
    t.join()
    assert not os.path.exists(p)


def test_without_task_every_lock_counts(tree):
    _lock(tree, "T00.lock", "tests/test_b.py\n")
    rc, log = _run(tree, task=None)                 # T00's own lock, but the run does not know it
    assert rc == 0, log
    [path] = _markers(tree)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["task"] is None and m["left_out"] == ["tests/test_b.py"] and m["locks_own"] == []
    assert "no --task" in log
