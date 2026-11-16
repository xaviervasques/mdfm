"""Run the test suite of this tree under the queue lock and leave a green marker on a pass.

Usage: python scripts/run_suite.py [--task T02] [--wait-hours 3]

1. Test environment: CUDA_VISIBLE_DEVICES=-1 (never an empty value: PowerShell drops an empty variable and
   the GPU shows again) and OMP_NUM_THREADS=2, set in this process and given to every child; a child python
   then asserts that torch sees no CUDA device (exit 11 otherwise).
2. Lock: waits for, then holds, cache/runs/queue/suite.lock, created exclusively and holding the PID, the
   host, the start time and the task. A lock whose holder is gone (no process with that PID, or a process
   started after the lock was written, which is a reused PID) is taken over; a lock of another host, or one
   whose holder this process cannot inspect (no psutil, access denied), is taken over once it is 3 h old; a
   lock whose content is still unreadable 0.5 s later (a crash between creating and writing it) is taken over
   at once. The wait ends after --wait-hours (exit 10). On Windows a file cannot be removed while another
   process holds it open, which a waiter does for a moment at each poll: the removal of the lock (release or
   takeover) is retried for 5 s, and so is a create refused while the previous lock is still being removed.
   Two waiters taking over the same dead lock at the same instant could both remove it; the window is a few
   milliseconds and only follows a crash.
3. Task locks: every cache/runs/queue/locks/*.lock younger than 3 h (file modification time) that belongs to
   another task than --task (default: the MDFM_TASK variable) is read as text; each test file name it
   contains (test_<name>.py, with or without a folder) that exists under tests/ is left out of the run with
   --ignore. Older locks are listed in the marker and not used; the lock named <task>.lock is the caller's
   own and is not used either. Only test_*.py names count: a lock naming a script or a fixture leaves out
   nothing. Without --task (nor MDFM_TASK) every lock counts as another task's, the caller's own included,
   and the run says so.
4. Hashes, read before and after the run: mdfm.train.provenance.code_sha256(), the SHA-256 of every
   scripts/*.py and configs/*.yaml (AppleDouble sidecars left out), and a SHA-256 over tests/**/*.py made the
   way code_sha256 is made.
5. Run, in two passes (task QROB): python -m pytest -q -p no:cacheprovider tests [--ignore ...] from the tree
   root, at BelowNormal priority (nice 10 off Windows), the output streamed. The gating pass selects -m "not live"
   and alone decides the marker; the live pass then selects -m live (the tests tests/live_tests.txt lists, marked
   by tests/conftest.py: their outcome follows files that jobs, readings or the MoveAtlas tree write or move) and
   never blocks. The selection travels in PYTEST_ADDOPTS (any inherited value is replaced), so that the argv of
   both passes is the one above. The live pass runs at most LIVE_TIMEOUT_S (its process tree is killed after) and
   follows the gating pass whatever its outcome.
6. Pass (gating pass: pytest exit 0 and at least one test passed): cache/runs/queue/green/<YYYYmmdd_HHMMSS>.json,
   written atomically right after the gating pass, with format, time, host, task, code_sha256, files ({relative
   path: SHA-256} of scripts/*.py and configs/*.yaml), tests_sha256, passed, counts, summary, left_out, locks_used,
   locks_stale, locks_own, changed_during_run, argv, python, env, pytest_addopts, duration_s and live ({"status":
   "running"} until the live pass ends, then its record: select, exit, status, counts, summary, failed, timed_out,
   duration_s; the marker is replaced atomically, and kept as first written when the replacement is refused). A
   script or config file that changed during the gating pass is dropped from files and named in
   changed_during_run, so that no marker vouches for a version that was not tested; a change of the mdfm code
   during the gating pass writes no marker (exit 12).
   Failure: pytest's own non-zero exit code of the gating pass (exit 14 when pytest exits 0 with no test passed),
   no marker. The live pass changes no exit code.
7. The last line (the one the runner logs): the gating outcome, then the live pass's counts and its failed ids.
   A live pass that fails (failed or erroring tests, an exit other than 0 and 5, a timeout) writes
   reports/gpu_queue/live_failures.md; once that file exists, every live pass rewrites it (a clean pass says so).

Exit codes: 0 pass and marker written; 1 to 5 pytest's; 10 lock wait timed out; 11 CUDA visible or torch
check failed; 12 mdfm code changed during the run; 13 the hashes could not be read; 14 no test passed.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMAT = "mdfm-green/1"
SUITE_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2"}
LOCK_MAX_AGE_S = 3 * 3600           # task locks older than this are ignored
SUITE_LOCK_STALE_S = 3 * 3600       # a suite.lock whose holder cannot be checked is taken over after this
PYTEST_ARGS = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
TEST_NAME = re.compile(r"(?<![A-Za-z0-9_])test_[A-Za-z0-9_]+\.py")
COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b")
GATE_ADDOPTS = '-m "not live"'      # the gating pass: decides the marker
LIVE_ADDOPTS = "-m live"            # the live pass: tests/live_tests.txt, never blocks
LIVE_TIMEOUT_S = 30 * 60
LIVE_FAILURES = os.path.join("reports", "gpu_queue", "live_failures.md")
LIVE_LINE = "run_suite: live pass"  # the line that opens the live pass in the log (scripts/gpu_queue.py reads it)
FAILED_ID = re.compile(r"^(FAILED|ERROR) (.+?)(?: - .*)?$")
NAMED_IN_LINE = 5                   # failed ids named in the last line

EXIT_LOCK_TIMEOUT = 10
EXIT_CUDA = 11
EXIT_CODE_CHANGED = 12
EXIT_HASHES = 13
EXIT_NO_PASS = 14


class LockTimeout(RuntimeError):
    """The suite lock stayed held by a live process for the whole wait."""


class CudaVisible(RuntimeError):
    """torch sees a CUDA device in the test environment, or the check could not run."""


def queue_dir(tree=TREE):
    return os.path.join(tree, "cache", "runs", "queue")


def suite_env(base=None):
    """The environment of the test run: `base` (default os.environ) with CUDA_VISIBLE_DEVICES=-1,
    OMP_NUM_THREADS=2 and PYTHONIOENCODING=utf-8."""
    env = dict(os.environ if base is None else base)
    env.update(SUITE_ENV)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _low_priority():
    """Popen keyword arguments that start a child at BelowNormal priority (nice 10 off Windows)."""
    if os.name == "nt":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


def assert_no_cuda(env, python=sys.executable):
    """Raise CudaVisible unless a child python with `env` imports torch and finds no CUDA device."""
    code = "import torch; assert not torch.cuda.is_available(), 'CUDA is visible'"
    try:
        r = subprocess.run([python, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900, **_low_priority())
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CudaVisible(f"torch check could not run: {e!r}") from e
    if r.returncode != 0:
        raise CudaVisible(f"torch check failed (exit {r.returncode}): {r.stderr.strip()[-500:]}")


# ---------------------------------------------------------------- suite lock
def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _remove(path, wait_s=5.0, poll_s=0.05):
    """Remove `path`; True when it is gone. A missing file counts as removed. On Windows a PermissionError
    (another process holds the file open for a moment, a waiter reading the lock) is retried for wait_s;
    False when it persists."""
    t0 = time.time()
    while True:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if time.time() - t0 >= wait_s:
                return False
            time.sleep(poll_s)


def holder_alive(info, now=None):
    """Whether the holder of a suite.lock (its parsed content) still runs. Content that is not a JSON object
    (unreadable, empty) counts as dead. A lock of another host, or one whose holder cannot be checked here (no
    psutil, access denied, no integer PID), counts as alive until SUITE_LOCK_STALE_S old."""
    now = time.time() if now is None else now
    if not isinstance(info, dict):
        return False
    start = info.get("start_epoch")
    start = float(start) if isinstance(start, (int, float)) else None
    old = start is None or now - start > SUITE_LOCK_STALE_S
    pid = info.get("pid")
    if info.get("host") != socket.gethostname() or not isinstance(pid, int):
        return not old
    try:
        import psutil
    except ImportError:
        return not old
    try:
        created = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError):
        return not old
    if start is not None and created > start + 2.0:      # the PID now belongs to a later process
        return False
    return True


class SuiteLock:
    """Exclusive lock file: waits for it, holds it inside the with block, removes it on exit."""

    def __init__(self, path, task=None, timeout_s=3 * 3600.0, poll_s=10.0, out=None):
        self.path, self.task, self.timeout_s, self.poll_s = path, task, timeout_s, poll_s
        self.out = sys.stdout if out is None else out
        self.info = None

    def _try_create(self):
        info = {"pid": os.getpid(), "host": socket.gethostname(), "task": self.task,
                "start": datetime.datetime.now().isoformat(timespec="seconds"), "start_epoch": time.time()}
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                return False
            except PermissionError:          # Windows: the previous lock is still being removed (delete pending)
                if time.time() - t0 >= 5.0:  # a lasting refusal is a real permission problem
                    raise
                time.sleep(0.05)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(info, f)
            f.write("\n")
        self.info = info
        return True

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        t0 = time.time()
        said = None
        while True:
            if self._try_create():
                return self
            info = _read_json(self.path)
            if info is None and os.path.exists(self.path):
                time.sleep(min(self.poll_s, 0.5))        # being written: read it again
                info = _read_json(self.path)
            if not os.path.exists(self.path):            # released meanwhile
                continue
            if not holder_alive(info):
                print(f"run_suite: taking over the stale {os.path.basename(self.path)} {info}", file=self.out)
                if _remove(self.path):
                    continue
                print(f"run_suite: cannot remove {self.path} yet (held open by another process)", file=self.out)
            if time.time() - t0 >= self.timeout_s:
                raise LockTimeout(f"{self.path} still held after {time.time() - t0:.0f} s: {info}")
            key = json.dumps(info, sort_keys=True)
            if key != said:
                print(f"run_suite: waiting for {self.path} held by {info}", file=self.out, flush=True)
                said = key
            time.sleep(self.poll_s)

    def release(self):
        if self.info is None:
            return
        info = _read_json(self.path)
        if isinstance(info, dict) and info.get("pid") == self.info["pid"] \
                and info.get("start_epoch") == self.info["start_epoch"] and not _remove(self.path):
            print(f"run_suite: could not remove {self.path}; the next waiter takes it over once this process "
                  "has ended", file=self.out, flush=True)
        self.info = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


# ---------------------------------------------------------------- task locks
def _test_files(tree, tests="tests"):
    """{file name: [tree-relative paths]} of the test_*.py files under tests/ (sidecars and caches left out)."""
    out = {}
    root = os.path.join(tree, tests)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not d.startswith(".")]
        for name in filenames:
            if name.startswith("test_") and name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), tree).replace("\\", "/")
                out.setdefault(name, []).append(rel)
    return out


def task_locks(tree=TREE, task=None, now=None, max_age_s=LOCK_MAX_AGE_S):
    """Read cache/runs/queue/locks/*.lock. Returns a dict: left_out (sorted tree-relative paths of the
    existing test files named in fresh locks of other tasks), used ({lock: [test file names]}), stale (locks
    older than max_age_s, with their age in hours) and own (the lock of `task`)."""
    now = time.time() if now is None else now
    locks = os.path.join(queue_dir(tree), "locks")
    used, stale, own = {}, {}, []
    names = sorted(os.listdir(locks)) if os.path.isdir(locks) else []
    for name in names:
        p = os.path.join(locks, name)
        if name.startswith("._") or not name.endswith(".lock") or not os.path.isfile(p):
            continue
        if task and name[:-len(".lock")].lower() == str(task).lower():
            own.append(name)
            continue
        try:
            age = now - os.path.getmtime(p)
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:                                    # removed while being read
            continue
        if age > max_age_s:
            stale[name] = round(age / 3600.0, 2)
            continue
        used[name] = sorted(set(TEST_NAME.findall(text)))
    present = _test_files(tree)
    left = sorted({rel for found in used.values() for n in found for rel in present.get(n, [])})
    return {"left_out": left, "used": used, "stale": stale, "own": own}


# ---------------------------------------------------------------- hashes
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes(tree=TREE):
    """{tree-relative path: SHA-256} of every scripts/*.py and configs/*.yaml (sidecars left out)."""
    out = {}
    for folder, pattern in (("scripts", "*.py"), ("configs", "*.yaml")):
        for p in glob.glob(os.path.join(tree, folder, pattern)):
            if os.path.basename(p).startswith("._") or not os.path.isfile(p):
                continue
            out[os.path.relpath(p, tree).replace("\\", "/")] = sha256_file(p)
    return dict(sorted(out.items()))


def tests_sha256(tree=TREE):
    """SHA-256 over tests/**/*.py: one line '<relative path>\\t<file SHA-256>' per file, sorted (the rule of
    mdfm.train.provenance.code_sha256)."""
    root = os.path.join(tree, "tests")
    lines = []
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if os.path.basename(p).startswith("._") or "__pycache__" in p:
            continue
        lines.append(f"{os.path.relpath(p, root).replace(os.sep, '/')}\t{sha256_file(p)}\n")
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def mdfm_code_sha256(tree=TREE):
    """mdfm.train.provenance.code_sha256() of the package of `tree` (imports torch: call it with the suite
    environment set)."""
    if tree not in sys.path:
        sys.path.insert(0, tree)
    from mdfm.train import provenance
    return provenance.code_sha256()


def tree_state(tree, code_sha):
    return {"code_sha256": code_sha(tree), "files": file_hashes(tree), "tests_sha256": tests_sha256(tree)}


# ---------------------------------------------------------------- pytest
def parse_counts(lines):
    """Counts of the last pytest summary line among `lines` ({'passed': n, ...}) and that line."""
    for line in reversed(lines):
        found = COUNT.findall(line)
        if found and (" in " in line or "passed" in line or "failed" in line):
            counts = {}
            for n, word in found:
                key = {"error": "errors", "warning": "warnings"}.get(word, word)
                counts[key] = counts.get(key, 0) + int(n)
            return counts, line.strip().strip("=").strip()
    return {}, ""


def pass_env(env, addopts):
    """The environment of one pass: `env` with PYTEST_ADDOPTS set to the pass's marker selection (an inherited
    PYTEST_ADDOPTS is replaced, so that no other option reaches either pass)."""
    e = dict(env)
    e["PYTEST_ADDOPTS"] = addopts
    return e


def _kill_tree(proc):
    """Kill proc and its descendants (psutil; taskkill /T on Windows without it)."""
    try:
        import psutil
        try:
            for ch in psutil.Process(proc.pid).children(recursive=True):
                try:
                    ch.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    except ImportError:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    try:
        proc.kill()
    except OSError:
        pass


def run_pytest_timed(argv, cwd, env, out=None, timeout_s=None):
    """Run argv at low priority with the output streamed to `out`; returns (exit code, last 200 lines, timed out).
    With timeout_s, the process tree is killed once that many seconds have passed."""
    out = sys.stdout if out is None else out
    tail = []
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            **_low_priority())
    fired = threading.Event()
    timer = None
    if timeout_s is not None:
        def _expire():
            fired.set()
            _kill_tree(proc)
        timer = threading.Timer(timeout_s, _expire)
        timer.daemon = True
        timer.start()
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            print(line, file=out, flush=True)
            tail.append(line)
            del tail[:-200]
        proc.wait()
    except BaseException:
        _kill_tree(proc)
        proc.wait()
        raise
    finally:
        if timer is not None:
            timer.cancel()
    return proc.returncode, tail, fired.is_set()


def run_pytest(argv, cwd, env, out=None):
    """Run argv at low priority with the output streamed to `out`; returns (exit code, last 200 lines)."""
    rc, tail, _ = run_pytest_timed(argv, cwd, env, out)
    return rc, tail


def failed_ids(lines):
    """The node ids of pytest's short summary among `lines` ('FAILED <id>' and 'ERROR <id>', message cut), in order,
    each once."""
    out = []
    for line in lines:
        m = FAILED_ID.match(line.strip())
        if m and m.group(2).strip() not in out:
            out.append(m.group(2).strip())
    return out


def name_ids(ids, k=NAMED_IN_LINE):
    """The first k ids joined, with the count of the others."""
    return ", ".join(ids[:k]) + (f" and {len(ids) - k} more" if len(ids) > k else "")


# ---------------------------------------------------------------- the live pass
def live_status(rc, failed, timed_out):
    """passed, none selected (pytest exit 5), failed, timed out, or error (any other exit)."""
    if timed_out:
        return "timed out"
    if rc == 0 and not failed:
        return "passed"
    if rc == 5 and not failed:
        return "none selected"
    if rc == 1 or failed:
        return "failed"
    return "error"


def live_pass(argv, tree, env, out, timeout_s=LIVE_TIMEOUT_S):
    """Run the live tests (-m live) with the gating pass's argv; never raises. Returns its record."""
    print(f"{LIVE_LINE}: {' '.join(argv[1:])} with PYTEST_ADDOPTS={LIVE_ADDOPTS} (never blocks; at most "
          f"{timeout_s:g} s)", file=out, flush=True)
    t0 = time.time()
    try:
        rc, tail, timed_out = run_pytest_timed(argv, tree, pass_env(env, LIVE_ADDOPTS), out, timeout_s)
    except Exception as e:  # noqa: BLE001
        return {"select": LIVE_ADDOPTS, "exit": None, "status": "error", "counts": {}, "summary": repr(e)[:300],
                "failed": [], "timed_out": False, "duration_s": round(time.time() - t0, 1)}
    counts, summary = parse_counts(tail)
    failed = failed_ids(tail)
    return {"select": LIVE_ADDOPTS, "exit": rc, "status": live_status(rc, failed, timed_out), "counts": counts,
            "summary": summary, "failed": failed, "timed_out": timed_out, "duration_s": round(time.time() - t0, 1)}


def live_text(live):
    """The live pass in the last line."""
    s = live["status"]
    if s == "passed":
        return f"live pass: {live['summary']}"
    if s == "none selected":
        return "live pass: no live test selected (pytest exit 5)"
    if s == "timed out":
        return (f"live pass: TIMED OUT after {live['duration_s']:g} s, process tree killed"
                + (f"; failed so far: {name_ids(live['failed'])}" if live["failed"] else ""))
    head = f"live pass: {s.upper()} (pytest exit {live['exit']}; {live['summary'] or 'no summary line'})"
    return head + (f"; failed: {name_ids(live['failed'])}" if live["failed"] else "")


def write_live_failures(tree, live, gate_line, marker_rel, now, task=None, out=None):
    """reports/gpu_queue/live_failures.md: written when the live pass did not pass (failed or erroring tests, an exit
    other than 0 and 5, a timeout), rewritten after every later pass while the file exists (a clean pass says so).
    Returns the path written, else None; a refused write is reported, never raised."""
    out = sys.stdout if out is None else out
    path = os.path.join(tree, LIVE_FAILURES)
    bad = live["status"] not in ("passed", "none selected")
    if not bad and not os.path.exists(path):
        return None
    L = ["# Live test failures", "",
         f"Written {now.isoformat(timespec='seconds')} by scripts/run_suite.py (task {task or 'none'}). The live pass "
         "(pytest -m live: the tests of tests/live_tests.txt, whose outcome follows files that jobs, readings or the "
         "MoveAtlas tree write or move) never decides the green marker: a failure here holds no job. It shows a test "
         "whose expectation the tree's outputs have moved past; update the test or its inputs, then the next suite "
         "rewrites this file.", "",
         f"- gating pass: {gate_line}",
         f"- green marker: {marker_rel or 'none written'}",
         f"- live pass: {live['status']}, pytest exit {live['exit']}, {live['summary'] or 'no summary line'}, "
         f"{live['duration_s']:g} s" + (", process tree killed at the time limit" if live["timed_out"] else ""), ""]
    if bad:
        L.append(f"## Failed ({len(live['failed'])})")
        L.append("")
        L += [f"- {i}" for i in live["failed"]] or ["- no FAILED or ERROR line in the output (see the suite log)"]
    else:
        L.append("No live failure in this pass.")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(L) + "\n")
        os.replace(tmp, path)
    except OSError as e:
        print(f"run_suite: {LIVE_FAILURES} not written: {e!r}", file=out, flush=True)
        return None
    return path


def write_marker(green, marker, now):
    """Write the marker atomically to green/<YYYYmmdd_HHMMSS>.json (a suffix when that name exists)."""
    os.makedirs(green, exist_ok=True)
    stem = now.strftime("%Y%m%d_%H%M%S")
    path, k = os.path.join(green, stem + ".json"), 1
    while os.path.exists(path):
        path, k = os.path.join(green, f"{stem}_{k}.json"), k + 1
    tmp = os.path.join(green, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(marker, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def update_marker(path, marker, wait_s=5.0, poll_s=0.05):
    """Replace the marker at `path` atomically with `marker`. A refused replacement (Windows: the runner reading the
    file at that instant) is retried for wait_s; False when it persists (the marker stays as first written)."""
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.live.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(marker, f, indent=1, sort_keys=False)
        f.write("\n")
    t0 = time.time()
    while True:
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            if time.time() - t0 >= wait_s:
                _remove(tmp)
                return False
            time.sleep(poll_s)


def gate_outcome(rc, tail, tree, code_sha, before, locks, task, python, argv, env, duration, queue, out):
    """The decision of the gating pass: (exit code, the gating part of the last line, marker or None, its path or
    None). The marker is written here, before the live pass, with live {"status": "running"}."""
    counts, summary = parse_counts(tail)
    if rc != 0:
        failed = failed_ids(tail)
        return rc, (f"FAIL pytest exit {rc} ({summary}); no marker"
                    + (f"; failed: {name_ids(failed)}" if failed else "")), None, None
    if counts.get("passed", 0) < 1:
        return EXIT_NO_PASS, f"FAIL no test passed ({summary!r}); no marker", None, None
    try:
        after = tree_state(tree, code_sha)
    except Exception as e:  # noqa: BLE001
        return EXIT_HASHES, f"FAIL cannot hash the tree after the run: {e!r}", None, None
    if after["code_sha256"] != before["code_sha256"]:
        return EXIT_CODE_CHANGED, "FAIL the mdfm code changed during the run; no marker", None, None
    changed = sorted(k for k in set(before["files"]) | set(after["files"])
                     if before["files"].get(k) != after["files"].get(k))
    if before["tests_sha256"] != after["tests_sha256"]:
        changed.append("tests/**/*.py")
    now = datetime.datetime.now()
    marker = {
        "format": FORMAT,
        "time": now.isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "task": task,
        "code_sha256": before["code_sha256"],
        "files": {k: v for k, v in before["files"].items() if k not in changed},
        "tests_sha256": before["tests_sha256"],
        "passed": counts["passed"],
        "counts": counts,
        "summary": summary,
        "left_out": locks["left_out"],
        "locks_used": locks["used"],
        "locks_stale": locks["stale"],
        "locks_own": locks["own"],
        "changed_during_run": changed,
        "argv": [os.path.basename(python)] + argv[1:],
        "python": python,
        "env": {k: env[k] for k in SUITE_ENV},
        "pytest_addopts": GATE_ADDOPTS,
        "duration_s": duration,
        "live": {"status": "running", "select": LIVE_ADDOPTS},
    }
    path = write_marker(os.path.join(queue, "green"), marker, now)
    print(f"run_suite: gating PASS {summary}; marker {os.path.relpath(path, tree)}", file=out, flush=True)
    if changed:
        print(f"run_suite: changed during the run, not vouched for: {changed}", file=out)
    return 0, f"PASS {summary}; marker {os.path.relpath(path, tree)}", marker, path


def run(tree=TREE, task=None, python=None, wait_s=3 * 3600.0, poll_s=10.0, code_sha=None, cuda_check=None,
        out=None, live_timeout_s=LIVE_TIMEOUT_S):
    """The whole run; returns the exit code (0 only when a marker was written)."""
    out = sys.stdout if out is None else out
    python = python or sys.executable
    code_sha = mdfm_code_sha256 if code_sha is None else code_sha
    cuda_check = assert_no_cuda if cuda_check is None else cuda_check
    env = suite_env()
    try:
        cuda_check(env, python)
    except CudaVisible as e:
        print(f"run_suite: FAIL {e}", file=out)
        return EXIT_CUDA
    queue = queue_dir(tree)
    try:
        with SuiteLock(os.path.join(queue, "suite.lock"), task=task, timeout_s=wait_s, poll_s=poll_s, out=out):
            locks = task_locks(tree, task)
            try:
                before = tree_state(tree, code_sha)
            except Exception as e:  # noqa: BLE001
                print(f"run_suite: FAIL cannot hash the tree: {e!r}", file=out)
                return EXIT_HASHES
            argv = [python] + PYTEST_ARGS + [f"--ignore={p}" for p in locks["left_out"]]
            print(f"run_suite: {' '.join(argv[1:])}", file=out)
            if locks["left_out"]:
                print(f"run_suite: left out (task locks {sorted(locks['used'])}): {locks['left_out']}", file=out)
                if not task:
                    print("run_suite: no --task (nor MDFM_TASK): every lock counts as another task's; pass --task "
                          "<id> so that your own lock leaves nothing out", file=out)
            if locks["stale"]:
                print(f"run_suite: task locks older than 3 h, not used: {locks['stale']}", file=out)
            print(f"run_suite: gating pass with PYTEST_ADDOPTS={GATE_ADDOPTS} (decides the marker); the live pass "
                  f"({LIVE_ADDOPTS}, tests/live_tests.txt) follows and never blocks", file=out, flush=True)
            t0 = time.time()
            rc, tail = run_pytest(argv, tree, pass_env(env, GATE_ADDOPTS), out)
            duration = round(time.time() - t0, 1)
            code, line, marker, path = gate_outcome(rc, tail, tree, code_sha, before, locks, task, python, argv,
                                                    env, duration, queue, out)
            live = live_pass(argv, tree, env, out, timeout_s=live_timeout_s)
            rel = os.path.relpath(path, tree) if path else None
            if path:
                marker["live"] = live
                try:
                    kept = not update_marker(path, marker)
                except OSError as e:
                    kept = True
                    print(f"run_suite: {e!r}", file=out)
                if kept:
                    print(f"run_suite: {rel} kept as first written (its replacement with the live pass was refused)",
                          file=out)
            write_live_failures(tree, live, line, rel, datetime.datetime.now(), task, out)
            print(f"run_suite: {line}; {live_text(live)}", file=out, flush=True)
            return code
    except LockTimeout as e:
        print(f"run_suite: FAIL {e}", file=out)
        return EXIT_LOCK_TIMEOUT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default=os.environ.get("MDFM_TASK") or None,
                   help="the caller's task id: its own lock <task>.lock leaves nothing out (default MDFM_TASK)")
    p.add_argument("--wait-hours", type=float, default=3.0, help="longest wait for suite.lock (default 3)")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.environ.update(SUITE_ENV)          # before any import of torch in this process (code_sha256)
    return run(TREE, task=a.task, wait_s=a.wait_hours * 3600.0)


if __name__ == "__main__":
    sys.exit(main())
