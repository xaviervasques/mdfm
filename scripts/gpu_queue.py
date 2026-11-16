"""GPU queue runner of the weekend 2026-10-01 to 2026-10-05: one pass every 60 s over the job files that
scripts/queue_job.py writes in cache/runs/queue/jobs/ (fields, placeholders and post-train templates: its
docstring); at most one GPU job and one CPU job at a time.

Status files: cache/runs/queue/heartbeat.txt (time, PID and number of the last pass), runner.pid, current.pid
(the PID of the running GPU job; absent when none), state.json (what the runner keeps across restarts),
runner.log; reports/gpu_queue/STATUS.md (the trigger phase_b_done, the GPU verdict, the flags present, the
running and pending jobs with what each one waits on, seconds per step of every training run in view, the last
ten exits, the suite runs, the locks).

GPU free, all of: no python process in the list of nvidia-smi --query-compute-apps=pid,process_name (a process
that holds a CUDA context; under WDDM the list also holds every graphics process, which are not python and do not
count; a PID whose name nvidia-smi cannot read counts when the process list, psutil or Get-CimInstance
Win32_Process, names it python), decision A13 of docs/OWNER_DECISIONS_2026-11-12.md; nvidia-smi
--query-gpu=memory.used below 2,000 MiB; no cache/runs/queue/gpu.lock held by a live PID. When the process query
fails, the check falls back to the command lines: no python process whose command line contains train_stage1.py,
train_g12, eval_, probe_, bench_, gate_ or video_generate (under that rule alone a CPU run of an eval_ script, which
holds no CUDA context, held the queue on 2026-10-15). The runner holds gpu.lock around each GPU job.

Eligible, all of: not_before passed; every 'after' job exited 0 (phase_b_done: the trigger has fired); every
soft_after job finished, in any state, or soft_deadline passed; requires exist; no not_if match pending or
running and no not_if_any match in any state; no placeholder left unresolved; no HOLD file and, for training, no
R0 file; no lock of cache/runs/queue/locks/ younger than 3 h that covers mdfm/ or a uses path (older locks are
listed and ignored); a marker of cache/runs/queue/green/ with the current code_sha256 and the current SHA-256 of
every uses path. When the marker is all that is missing, the runner runs scripts/run_suite.py --task gpu_queue
itself, in the background, at most once per 30 minutes, and only while the system's free commit (GlobalMemoryStatusEx
ullAvailPageFile) is at least SUITE_START_FREE_GB (9 GB); a running suite is stopped (its process tree killed, the run
not counted as a failure) when the free commit falls under SUITE_STOP_FREE_GB (4 GB), decision A12 of
docs/OWNER_DECISIONS_2026-11-12.md (the floor of the session's own CPU runs, reports/weekend_2026-10-01/tasks/S29.md;
E1 crashed on the commit limit beside a suite on 2026-10-07, S28.md). A free commit that cannot be read does not hold
the suite. A failure goes to STATUS.md. After SUITE_STREAK (3) runs in
a row that fail for the same jobs, runner.log gets a line naming the failing tests of the last run (the FAILED and
ERROR lines of its gating pass), and STATUS.md shows that line while the streak lasts. A uses path that is missing,
or that run_suite.py does not hash (anything but scripts/*.py and configs/*.yaml), is reported as such and calls
for no suite run.

Choice: the lowest cls, then the lowest priority, then the name; a training job starts only when no class-1 job is
eligible. Deadline guard: a job whose estimated end passes the deadline (the --until time; default
2026-10-05T07:30, the weekend's end) is logged skipped_deadline
(evaluation: est_hours; training: the remaining steps times the measured seconds per step, plus
TRAIN_OVERHEAD_H, plus the est_hours of its post-train template jobs when it runs phase B).

Launch: training at AboveNormal priority, GPU evaluations at Normal, CPU steps at BelowNormal; stdout and stderr
to cache/runs/queue/logs/<name>.log; CUDA_VISIBLE_DEVICES=0 for a GPU job unless its env says otherwise, -1 for a
CPU job. Before a training job, mdfm/, configs/, scripts/ and docs/rules/ are zipped to
<run_dir>/code_snapshot_<time>.zip.

Job log: reports/gpu_queue/job_log.csv, one row per job once it is over (LOG_COLUMNS); the runner's CPU steps
get rows named step:<name>. While the file cannot be written (held open by another program), its rows wait in
state.json and count as written. A job whose program cannot start is logged failed.

Exits. An evaluation's non-zero exit is logged and the queue goes on. Training: exit 3 (alarm), an 'alarm' event
in its log.jsonl or ProvenanceError in its console log writes cache/runs/queue/R0, which blocks training jobs
only (evaluations go on with checkpoints that checkpoint.load verifies); FloatingPointError or another crash gets
one resume, the same argv with --resume <latest checkpoint of its run_dir>; a second failure writes R0.
Timeouts (timeout_hours) kill the job's process tree and are logged. HOLD stops new launches only.

Restart safety: a job with a row in job_log.csv is skipped. A job of state.json whose process still runs is
adopted (waited for, its exit code read). A training job whose process ended while the runner was down, or that
has a console log but no state, is judged from its run folder: an 'end' event with a 'verify' event of its phase
counts as exit 0; otherwise (no 'end' event, no live process) it is resumed once.

Speed: for every training run in view (the queue's training job; the running phase B of PHASE_B), seconds per
step over the last 500 steps of its current call, from the wall_s of its 'log' events, leaving out each interval
that holds a dev or checkpoint step and the first interval of a call. cache/runs/queue/SLOW is written above
1.05 x the baseline and deleted when the rate falls back below or no training runs. Baseline: the s_per_step of
reports/gpu_queue/baseline_s_per_step.json for phase B, the first 500 steps of the call otherwise.

Post-train templates: when a training run of the queue logs its phase B 'verify' event, the templates whose runs
filter matches its folder name become job files.

CPU steps of scripts/weekend_branch.py, run when the script exists (synchronously, BelowNormal, at most 20
minutes, three tries 30 minutes apart): --stage branch --write once J3 is over (J3_NAMES and the jobs of group
J3 all finished; names never queued are waited for until J3_DEADLINE); --stage seeds --write once the post-train
jobs of pilot_v1_s2 are over; --stage reference --write before the first class-4 job (the reference_checkpoint
of the newest reports/gpu_queue/branch_reference_*.json gives {ref_ckpt}); --stage extension --write before the
first class-5 job (it writes cache/runs/queue/EXTEND, JSON or key=value lines, with ext_ckpt and ext_steps).
After each GPU block (GPU jobs ran and none follows at once, or a training job is next):
scripts/summarize_weekend.py and scripts/gate_table.py when they exist.

Trigger phase_b_done (PHASE_B): the phase B launched on 2026-09-30 (PID 4224, checked to be python running
train_stage1.py --phase B, its start time kept against PID reuse). It fires on a 'verify' event with phase B and
step 32,044 in cache/runs/phaseA_root_full/log.jsonl once that process is gone; {phase_b_ckpt} is the event's
path. Never on 'phase_end', which is written before the final save. When the process is gone and no such event
appears within 10 minutes, the tail of cache/runs/phaseB_root_full.log decides: an alarm or ProvenanceError writes
R0, and {phase_b_ckpt} becomes the latest ckpt_B_*.pt that checkpoint.load verifies (a CPU child,
--verify-checkpoints); FloatingPointError or another crash gets one resume, the job phaseB_root_full_resume
(scripts/train_stage1.py --phase B --resume <latest ckpt_B_*.pt> --prefetch 2 --memory-cap-gb 14.5, the launch
command of handoff 4 section 2j), whose failure writes R0 and falls back the same way. An alarm event of phase B
beside the verify event also writes R0. A python process at that PID whose command line cannot be read counts as
the phase B still running; a verify event that appears while the resume job is queued but not launched fires
the trigger, and the resume job is logged skipped.

Usage (mdfm environment, from the tree root):
  python scripts/gpu_queue.py [--until 2026-10-05T07:30]   the runner, in the foreground; from --until on it
                                                          launches nothing and it ends when its jobs have ended
  python scripts/gpu_queue.py --dry-run [--write-status]  one pass that launches and writes nothing: prints the
                                                          trigger, the GPU verdict, the eligible job and what
                                                          every pending job waits on (--write-status: also
                                                          STATUS.md, marked dry run)
  python scripts/gpu_queue.py --once                      one real pass
  python scripts/gpu_queue.py --verify-checkpoints A.pt B.pt   (CPU) the first that checkpoint.load verifies
"""
import argparse
import copy
import csv
import datetime as dt
import fnmatch
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import queue_job as QJ  # noqa: E402

STATE_FORMAT = "mdfm-queue-state/1"
DEADLINE = "2026-10-05T07:30"
PASS_S = 60.0
GPU_CMD = ("train_stage1.py", "train_g12", "eval_", "probe_", "bench_", "gate_", "video_generate")
GPU_FREE_MIB = 2000
GB = 10 ** 9
SUITE_START_FREE_GB = 9.0                         # the suite starts only at or above this free commit (decision A12)
SUITE_STOP_FREE_GB = 4.0                          # a running suite is stopped under this free commit (decision A12)
LOCK_MAX_AGE_S = 3 * 3600
SUITE_EVERY_S = 30 * 60
SUITE_TASK = "gpu_queue"
SUITE_STREAK = 3                                  # failed suite runs in a row, for the same jobs, that name their tests
SUITE_LIVE_LINE = "run_suite: live pass"          # run_suite.py's live pass starts at this line of its log
SUITE_FAILED = re.compile(r"^(FAILED|ERROR) (.+?)(?: - .*)?$")
SLOW_FACTOR = 1.05
SPAN_STEPS = 500
STEP_TIMEOUT_S = 20 * 60
STEP_RETRY_S = 30 * 60
STEP_TRIES = 3
VERIFY_TIMEOUT_S = 45 * 60
VERIFY_WAIT_S = 600
TRAIN_OVERHEAD_H = 0.25
PHASE_B = {"pid": 4224, "run_dir": "cache/runs/phaseA_root_full", "text_log": "cache/runs/phaseB_root_full.log",
           "phase": "B", "steps": 32044, "resume_job": "phaseB_root_full_resume",
           "resume_args": ["--prefetch", "2", "--memory-cap-gb", "14.5"]}
J3_NAMES = tuple(QJ.J3_SOFT)
J3_DEADLINE = QJ.SEED2_SOFT_DEADLINE
SEED2_RUN = "pilot_v1_s2"
LOG_COLUMNS = ("name", "cls", "priority", "state", "argv", "cuda_visible_devices", "omp_num_threads", "start", "end",
               "exit_code", "outputs", "gpu_name", "code_sha256", "entry_sha256", "config_sha256", "snapshot_sha256",
               "note")
SNAPSHOT_DIRS = ("mdfm", "configs", "scripts", "docs/rules")
PH_REASON = {"phase_b_ckpt": "waiting phase_b_done", "phase_b_name": "waiting phase_b_done",
             "ref_ckpt": "stage: reference", "ref_name": "stage: reference",
             "ext_ckpt": "stage: extension", "ext_steps": "stage: extension"}
MARKED_PATH = re.compile(r"(scripts/[^/]+\.py|configs/[^/]+\.yaml)")   # the files of a run_suite.py marker
ALARM_STOP = re.compile(r"AlarmError|\"alarm_stop\": \"training alarm")
ALARM_LIST = re.compile(r"^alarm at phase", re.M)       # train_stage1.py lists every alarm of the run folder

if os.name == "nt":
    NO_WINDOW = 0x08000000
    PRIORITY = {"above": 0x00008000, "normal": 0x00000020, "below": 0x00004000}
else:
    NO_WINDOW, PRIORITY = 0, {}


def _flags(kind):
    return (PRIORITY[kind] | NO_WINDOW) if os.name == "nt" else 0


def iso(t):
    return t.isoformat(timespec="seconds")


# ---------------------------------------------------------------- files
class Layout:
    """The paths of the queue under a tree."""

    def __init__(self, tree=TREE):
        self.tree = tree
        self.queue = QJ.queue_dir(tree)
        self.jobs, self.templates = QJ.jobs_dir(tree), QJ.templates_dir(tree)
        q = self.queue
        self.logs, self.green, self.locks = (os.path.join(q, x) for x in ("logs", "green", "locks"))
        self.heartbeat, self.runner_pid, self.current_pid = (os.path.join(q, x) for x in
                                                             ("heartbeat.txt", "runner.pid", "current.pid"))
        self.gpu_lock, self.state, self.runner_log = (os.path.join(q, x) for x in ("gpu.lock", "state.json",
                                                                                   "runner.log"))
        self.hold, self.r0, self.slow, self.extend = (os.path.join(q, x) for x in ("HOLD", "R0", "SLOW", "EXTEND"))
        self.reports = os.path.join(tree, "reports", "gpu_queue")
        self.job_log = os.path.join(self.reports, "job_log.csv")
        self.status = os.path.join(self.reports, "STATUS.md")
        self.baseline = os.path.join(self.reports, "baseline_s_per_step.json")

    def abs(self, path):
        p = str(path).replace("/", os.sep)
        return p if os.path.isabs(p) else os.path.join(self.tree, p)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_text(path, text):
    """Write text atomically (temporary file in the same folder, then os.replace), UTF-8 with LF."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for i in range(50):                       # Windows: a reader may hold the target open for a moment
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            if i == 49:
                raise
            time.sleep(0.1)
    return path


def append_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def remove(path):
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def read_jsonl(path):
    """The JSON objects of a .jsonl file, one per line; unparsable lines (a partial last line) are skipped."""
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append(r)
    except OSError:
        return []
    return out


def tail_text(path, start=0, max_bytes=1 << 16):
    """The text of a file from byte `start`, at most its last max_bytes."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(int(start), size - max_bytes, 0))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def suite_failed_tests(text):
    """The node ids of the FAILED and ERROR lines of a run_suite.py log before its live pass (the gating pass; the
    whole log when it has no live pass), in order, each once."""
    out = []
    for line in text.splitlines():
        if line.startswith(SUITE_LIVE_LINE):
            break
        m = SUITE_FAILED.match(line.strip())
        if m and m.group(2).strip() not in out:
            out.append(m.group(2).strip())
    return out


def streak_line(streak):
    """The line of a run of failed suite runs for the same jobs (runner.log and STATUS.md)."""
    tests = streak.get("tests") or []
    named = ", ".join(tests[:10]) + (f" and {len(tests) - 10} more" if len(tests) > 10 else "")
    return (f"run_suite failed {streak['count']} times in a row for the same jobs "
            f"({', '.join(streak.get('jobs') or []) or 'none'}); failing tests of the last run (exit "
            f"{streak.get('exit')}): " + (named or "none named in its log"))


def last_call(records):
    """The records from the last 'start' event on (the current call of a run folder)."""
    idx = [i for i, r in enumerate(records) if r.get("event") == "start"]
    return records[idx[-1]:] if idx else list(records)


# ---------------------------------------------------------------- processes and the GPU
def list_processes():
    """[{pid, ppid, name, cmdline, create_time}] of every process: psutil when installed, else
    Get-CimInstance Win32_Process through PowerShell (no create_time then)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    out = []
    if psutil is not None:
        for p in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
            i = p.info
            cmd = i.get("cmdline") or []
            out.append({"pid": i["pid"], "ppid": i.get("ppid"), "name": i.get("name") or "",
                        "cmdline": " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd),
                        "create_time": i.get("create_time")})
        return out
    ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
          "ConvertTo-Json -Compress")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=180, creationflags=NO_WINDOW)
    rows = json.loads(r.stdout or "[]")
    for row in rows if isinstance(rows, list) else [rows]:
        out.append({"pid": int(row["ProcessId"]), "ppid": row.get("ParentProcessId"), "name": row.get("Name") or "",
                    "cmdline": row.get("CommandLine") or "", "create_time": None})
    return out


def is_python(name):
    return os.path.basename(str(name or "")).lower().startswith("python")


def gpu_users(procs, patterns=GPU_CMD):
    """The python processes whose command line contains one of `patterns`."""
    return [p for p in procs if is_python(p.get("name")) and any(s in (p.get("cmdline") or "") for s in patterns)]


def proc_by_pid(procs, pid):
    return next((p for p in procs if p.get("pid") == pid), None)


def same_process(p, create_time):
    """p (a process row) is the process started at create_time (None: not checked)."""
    if p is None:
        return False
    if create_time is None or p.get("create_time") is None:
        return True
    return abs(float(p["create_time"]) - float(create_time)) < 2.0


def nvidia_query():
    """The text of nvidia-smi --query-gpu=memory.used,name (csv, no header, no units), None on failure."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                           creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def nvidia_apps_query():
    """The text of nvidia-smi --query-compute-apps=pid,process_name (csv, no header), None on failure."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                           creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def parse_nvidia_apps(text):
    """[(pid, process name)] of nvidia-smi's compute-app list; None when text is None or a line is unreadable."""
    if text is None:
        return None
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid, _, name = line.partition(",")
        try:
            out.append((int(pid.strip()), name.strip()))
        except ValueError:
            return None
    return out


def gpu_users_nvidia(procs, apps):
    """The processes of nvidia-smi's list (parse_nvidia_apps) that are python: by the name nvidia-smi gives, else, for
    a name it cannot read ('[Insufficient Permissions]'), by the name of that PID in the process list."""
    out = []
    for pid, name in apps:
        p = proc_by_pid(procs, pid)
        if is_python(name) or (name.startswith("[") and p is not None and is_python(p.get("name"))):
            out.append(p if p is not None else {"pid": pid, "name": name, "cmdline": name})
    return out


def free_commit():
    """The system's free commit in bytes (GlobalMemoryStatusEx ullAvailPageFile), None off Windows or on failure."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MSX(ctypes.Structure):
            _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MSX()
        m.dwLength = ctypes.sizeof(MSX)
        if not ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return int(m.ullAvailPageFile)
    except Exception:  # noqa: BLE001  (a measure never stops the runner)
        return None


def parse_nvidia(text):
    """(memory.used in MiB of the first GPU, its name), (None, None) when unreadable."""
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        try:
            return int(float(parts[0])), (parts[1] if len(parts) > 1 else None)
        except ValueError:
            return None, None
    return None, None


def gpu_lock_holder(path, procs):
    """The content of gpu.lock when its PID is a live process (same start time when recorded), else None."""
    info = QJ.read_json(path)
    if not isinstance(info, dict) or not isinstance(info.get("pid"), int):
        return None
    p = proc_by_pid(procs, info["pid"])
    return info if same_process(p, info.get("create_time")) else None


def gpu_verdict(procs, nvidia_text, lock_path, apps_text=None):
    """(free, reasons): free when no python process holds the GPU, memory.used is below GPU_FREE_MIB and no live PID
    holds gpu.lock. The python processes that hold it are those of nvidia-smi's compute-app list (apps_text,
    gpu_users_nvidia); when apps_text is None or unreadable, the GPU_CMD python processes of the command lines."""
    why = []
    apps = parse_nvidia_apps(apps_text)
    if apps is None:
        for p in gpu_users(procs):
            why.append(f"python PID {p['pid']} runs {_short(p.get('cmdline'))}")
    else:
        for p in gpu_users_nvidia(procs, apps):
            why.append(f"python PID {p['pid']} holds the GPU (nvidia-smi): {_short(p.get('cmdline'))}")
    mem, _ = parse_nvidia(nvidia_text)
    if mem is None:
        why.append("nvidia-smi memory.used unreadable")
    elif mem >= GPU_FREE_MIB:
        why.append(f"nvidia-smi memory.used {mem:,} MiB (free below {GPU_FREE_MIB:,})")
    held = gpu_lock_holder(lock_path, procs)
    if held:
        why.append(f"gpu.lock held by PID {held['pid']} ({held.get('job')})")
    return not why, why


def _short(cmd, n=140):
    s = " ".join(str(cmd or "").split())
    return s if len(s) <= n else s[:n - 3] + "..."


def kill_tree(pid):
    """Kill a process and its descendants (psutil, else taskkill /T /F)."""
    try:
        import psutil
    except ImportError:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)
        return
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=60)


class Child:
    """A process the runner waits for: a Popen it launched, or a PID adopted after a restart (waited for in a
    thread through psutil, which reads the exit code of any process on Windows)."""

    def __init__(self, popen=None, pid=None):
        self.popen = popen
        self.pid = popen.pid if popen is not None else int(pid)
        self._done, self._rc = False, None
        if popen is None:
            threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        rc = None
        try:
            import psutil
            rc = psutil.Process(self.pid).wait()
        except Exception:  # noqa: BLE001
            rc = None
        self._rc, self._done = rc, True

    def poll(self):
        """(ended, exit code or None)."""
        if self.popen is not None:
            rc = self.popen.poll()
            return rc is not None, rc
        return self._done, self._rc


# ---------------------------------------------------------------- locks and green markers
_TREE_PREFIX = re.compile(r"(?i)\S*?md_fm/mdfm/")
_TOKEN = re.compile(r"[A-Za-z0-9_.\-/*?\[\]]+")


def lock_tokens(text):
    """Path-like tokens of a lock file, made tree-relative (any prefix ending in MD_FM/mdfm/ removed)."""
    t = _TREE_PREFIX.sub("", str(text).replace("\\", "/"))
    out = []
    for tok in _TOKEN.findall(t):
        while tok.startswith("./"):
            tok = tok[2:]
        tok = tok.rstrip(".,;:")
        if tok:
            out.append(tok)
    return out


def lock_covers(text, rel):
    """True when a lock's text names the tree-relative path rel: the path itself, its file name, a folder that
    holds it (a token ending in '/') or a glob that matches it."""
    rel = str(rel).replace("\\", "/").strip("/")
    base = rel.rsplit("/", 1)[-1]
    for tok in lock_tokens(text):
        if tok in (rel, base):
            return True
        if tok.endswith("/") and rel.startswith(tok):
            return True
        if any(ch in tok for ch in "*?[") and (fnmatch.fnmatchcase(rel, tok) or fnmatch.fnmatchcase(base, tok)):
            return True
    return False


def lock_covers_mdfm(text):
    """True when a lock names a path of the mdfm/ package."""
    return any(tok == "mdfm" + "/" or tok.startswith("mdfm/") for tok in lock_tokens(text))


def read_locks(folder, now_epoch=None, max_age_s=LOCK_MAX_AGE_S):
    """(fresh {lock name: text}, stale {lock name: age in hours}) of folder/*.lock (sidecars left out)."""
    now_epoch = time.time() if now_epoch is None else now_epoch
    fresh, stale = {}, {}
    for p in sorted(glob.glob(os.path.join(folder, "*.lock"))):
        name = os.path.basename(p)
        if name.startswith("._"):
            continue
        try:
            age = now_epoch - os.path.getmtime(p)
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if age > max_age_s:
            stale[name] = round(age / 3600.0, 2)
        else:
            fresh[name] = text
    return fresh, stale


def code_sha256(tree=TREE):
    """mdfm.train.provenance.code_sha256 of tree/mdfm, recomputed without importing the package (one line
    '<relative path>\\t<file SHA-256>' per .py file, sorted; sidecars and caches left out)."""
    root = os.path.join(tree, "mdfm")
    lines = []
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if os.path.basename(p).startswith("._") or "__pycache__" in p:
            continue
        rel = os.path.relpath(p, root).replace("\\", "/")
        lines.append(f"{rel}\t{sha256_file(p)}\n")
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def green_markers(folder):
    """[(path, marker)] of folder/*.json, newest file name first."""
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json")), reverse=True):
        if os.path.basename(p).startswith(("._", ".")):
            continue
        m = QJ.read_json(p)
        if isinstance(m, dict):
            out.append((p, m))
    return out


def green_match(markers, code_sha, uses_sha):
    """The path of the newest marker whose code_sha256 is code_sha and whose files hold every {path: sha} of
    uses_sha, else None."""
    for p, m in markers:
        if m.get("code_sha256") != code_sha:
            continue
        files = m.get("files") or {}
        if all(files.get(k) == v for k, v in uses_sha.items()):
            return p
    return None


# ---------------------------------------------------------------- run folders
def clean_intervals(records, phase=None):
    """[(step0, step1, seconds)] between consecutive 'log' events of one call (and of `phase`), leaving out an
    interval that holds a dev or checkpoint step (their work runs after the log event of their step, so it
    lands in the interval that starts there) and the first interval of a call. Seconds: wall_s of the closing
    event, else the difference of the event times."""
    marks = {(r.get("phase"), r.get("step")) for r in records if r.get("event") in ("dev", "checkpoint")}
    out, prev = [], None
    for r in records:
        ev = r.get("event")
        if ev in ("start", "resume"):
            prev = None
            continue
        if ev != "log" or (phase is not None and r.get("phase") != phase):
            continue
        if prev is not None and prev.get("phase") == r.get("phase") and r.get("step", 0) > prev.get("step", 0):
            s0, s1 = int(prev["step"]), int(r["step"])
            if not any((r.get("phase"), s) in marks for s in range(s0, s1)):
                secs = r.get("wall_s")
                if not isinstance(secs, (int, float)):
                    try:
                        secs = (dt.datetime.fromisoformat(r["time"]) - dt.datetime.fromisoformat(prev["time"])
                                ).total_seconds()
                    except (KeyError, ValueError, TypeError):
                        secs = None
                if secs is not None and secs > 0:
                    out.append((s0, s1, float(secs)))
        prev = r
    return out


def span_rate(intervals, which="last", span=SPAN_STEPS):
    """Seconds per step over the clean intervals of the last (or first) `span` steps; None when they cover less
    than half of the span."""
    if not intervals:
        return None
    if which == "last":
        edge = intervals[-1][1]
        sel = [iv for iv in intervals if iv[1] > edge - span]
    else:
        edge = intervals[0][0]
        sel = [iv for iv in intervals if iv[0] < edge + span]
    steps = sum(b - a for a, b, _ in sel)
    if steps < span / 2:
        return None
    return sum(s for _, _, s in sel) / steps


def last_step(records, phase=None):
    """The step of the last 'log' event (of `phase`), None when there is none."""
    for r in reversed(records):
        if r.get("event") == "log" and (phase is None or r.get("phase") == phase):
            return int(r.get("step", 0))
    return None


def checkpoints(run_dir, phase=None):
    """The ckpt_<phase>_<step>.pt files of a run folder, newest first (phase C after phase B after phase A, then by
    step); phase C is a stage-2 run's (train_stage1.py --phase C), whose crash resume needs its latest ckpt_C."""
    out = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_*_*.pt")):
        m = re.match(r"ckpt_([ABC])_(\d+)\.pt$", os.path.basename(p))
        if m and (phase is None or m.group(1) == phase):
            out.append(((m.group(1), int(m.group(2))), p))
    return [p for _, p in sorted(out, reverse=True)]


def with_resume(argv, ckpt):
    """argv with --resume ckpt (replacing a --resume value already there); argv unchanged when ckpt is None."""
    if ckpt is None:
        return list(argv)
    a = list(argv)
    if "--resume" in a:
        i = a.index("--resume")
        if i + 1 < len(a):
            a[i + 1] = ckpt
            return a
    return a + ["--resume", ckpt]


def classify(rc, tail, seg):
    """How a training call ended: 'alarm' (exit 3, an 'alarm' event of the call's records seg, the alarm stop
    of train_stage1.py in its console; the 'alarm at phase' lines it prints list the alarms of every call of
    the run folder, so they count only when seg is empty), 'provenance' (ProvenanceError), 'ok' (exit 0; or
    exit unknown with 'end' and 'verify' events), 'fpe' (FloatingPointError) or 'crash'."""
    tail = tail or ""
    if rc == 3 or any(r.get("event") == "alarm" for r in seg) or ALARM_STOP.search(tail) or \
            (not seg and ALARM_LIST.search(tail)):
        return "alarm"
    if "ProvenanceError" in (tail or ""):
        return "provenance"
    if rc == 0:
        return "ok"
    if rc is None and any(r.get("event") == "end" for r in seg) and any(r.get("event") == "verify" for r in seg):
        return "ok"
    if "FloatingPointError" in (tail or ""):
        return "fpe"
    return "crash"


def config_of(argv):
    for i, a in enumerate(argv[:-1]):
        if a == "--config":
            return argv[i + 1]
    return None


def read_extend(path):
    """{ext_ckpt, ext_steps} of the EXTEND file (JSON, or key=value / key: value lines), {} when absent."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return {}
    out = {}
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            out = {k: d[k] for k in ("ext_ckpt", "ext_steps") if d.get(k) is not None}
    except ValueError:
        for line in text.splitlines():
            m = re.match(r"\s*(ext_ckpt|ext_steps)\s*[:=]\s*(\S+)", line)
            if m:
                out[m.group(1)] = m.group(2)
    if "ext_ckpt" in out:
        out["ext_ckpt"] = QJ.tree_rel(out["ext_ckpt"])
    if "ext_steps" in out:
        out["ext_steps"] = str(int(float(out["ext_steps"])))
    return out


def read_job_log(path):
    """{name: row} of job_log.csv (the last row of a name)."""
    out = {}
    try:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("name"):
                    out[row["name"]] = row
    except OSError:
        pass
    return out


def append_job_log(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: "" if row.get(k) is None else row.get(k) for k in LOG_COLUMNS})


def row_ok(row):
    """A job that ended well: state 'done' with exit 0, or with no exit code when its end was judged from its
    run folder (a training job that ended while the runner was down: 'end' and 'verify' events)."""
    return row.get("state") == "done" and str(row.get("exit_code") if row.get("exit_code") is not None else "") \
        in ("0", "")


def code_snapshot(tree, run_dir, when):
    """Zip mdfm/, configs/, scripts/ and docs/rules/ (sidecars and caches left out) to
    <run_dir>/code_snapshot_<time>.zip; returns (path, SHA-256)."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"code_snapshot_{when.strftime('%Y%m%d_%H%M%S')}.zip")
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for d in SNAPSHOT_DIRS:
            root = os.path.join(tree, *d.split("/"))
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(x for x in dirnames if x != "__pycache__" and not x.startswith("."))
                for fn in sorted(filenames):
                    if fn.startswith(("._", ".")) or fn.endswith((".pyc", ".pyo", ".tmp")):
                        continue
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, tree).replace("\\", "/"))
    os.replace(tmp, path)
    return path, sha256_file(path)


def baseline_rate(path):
    d = QJ.read_json(path)
    v = d.get("s_per_step") if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def seed_rates(tree=TREE):
    """Seconds per step for the seed-2 estimates: phase A from the clean intervals of seed 1's phase A
    (cache/runs/phaseA_root_full/log.jsonl), phase B from reports/gpu_queue/baseline_s_per_step.json."""
    L = Layout(tree)
    recs = read_jsonl(os.path.join(L.abs(PHASE_B["run_dir"]), "log.jsonl"))
    iv = clean_intervals(recs, "A")
    a = sum(s for _, _, s in iv) / max(sum(b - x for x, b, _ in iv), 1) if iv else None
    b = baseline_rate(L.baseline)
    if a is None or b is None:
        raise RuntimeError("seed_rates: seed 1's phase A log or the phase B baseline is missing")
    return {"A": round(a, 4), "B": round(b, 4)}


# ---------------------------------------------------------------- the runner
class Runner:
    """The queue (module docstring). The functions it calls out to can be replaced (tests): procs_fn,
    nvidia_fn, now_fn, verify_fn (paths -> the first that checkpoint.load verifies), popen."""

    def __init__(self, tree=TREE, python=QJ.PYTHON, procs_fn=None, nvidia_fn=None, now_fn=None, verify_fn=None,
                 popen=None, out=None, deadline=DEADLINE, until=None, phase_b=None, j3_deadline=J3_DEADLINE,
                 apps_fn=None, commit_fn=None):
        self.tree, self.L, self.python = tree, Layout(tree), python
        self.procs_fn = procs_fn or list_processes
        self.nvidia_fn = nvidia_fn or nvidia_query
        # the GPU process list and the free commit: the real queries only for a runner on the real machine (no
        # nvidia_fn, no procs_fn given); a runner built with fakes reads neither unless given its own
        self.apps_fn = apps_fn or (nvidia_apps_query if nvidia_fn is None else (lambda: None))
        self.commit_fn = commit_fn or (free_commit if procs_fn is None else (lambda: None))
        self.now = now_fn or dt.datetime.now
        self.verify_fn = verify_fn or self._verify_subprocess
        self.popen = popen or subprocess.Popen
        self.out = sys.stdout if out is None else out
        self.deadline = QJ.parse_time(deadline)
        self.until = QJ.parse_time(until) if until else self.deadline
        self.pb = dict(PHASE_B if phase_b is None else phase_b)
        self.j3_deadline = QJ.parse_time(j3_deadline)
        self.children, self.suite_child = {}, None
        self.state = self.load_state()
        self.pass_no = 0
        self.pid = os.getpid()
        self.dry = False

    # ------------------------------------------------------------ state and log
    def load_state(self):
        s = QJ.read_json(self.L.state)
        s = s if isinstance(s, dict) else {}
        s["format"] = STATE_FORMAT
        for k, v in (("phase_b", {}), ("running", {"gpu": None, "cpu": None}), ("attempts", {}), ("resume", {}),
                     ("expanded", {}), ("steps", {}), ("suite", {}), ("block_dirty", False), ("reruns", {})):
            s.setdefault(k, copy.deepcopy(v))
        s["running"].setdefault("gpu", None)
        s["running"].setdefault("cpu", None)
        return s

    def save_state(self):
        if not self.dry:
            write_text(self.L.state, json.dumps(self.state, indent=1, sort_keys=True) + "\n")

    def append_row(self, row):
        """Append a row to job_log.csv. A refused write (the file held open by another program) keeps the row in
        state.json; it is written at the next passes and counts as written meanwhile (read_rows)."""
        if self.dry:
            return
        self.state.setdefault("pending_rows", []).append(
            {k: "" if row.get(k) is None else str(row.get(k)) for k in LOG_COLUMNS})
        self.flush_rows()

    def flush_rows(self):
        pend = self.state.get("pending_rows") or []
        while pend:
            try:
                append_job_log(self.L.job_log, pend[0])
            except OSError as e:
                self.log(f"job_log.csv not writable ({e!r}); {len(pend)} rows kept in state.json")
                self.save_state()
                return False
            pend.pop(0)
        return True

    def read_rows(self):
        """{name: row} of job_log.csv and of the rows still waiting to be written."""
        out = read_job_log(self.L.job_log)
        for r in self.state.get("pending_rows") or []:
            out[r["name"]] = dict(r)
        return out

    def log(self, msg):
        line = f"{iso(self.now())} {msg}"
        try:
            print(line, file=self.out, flush=True)
        except Exception:  # noqa: BLE001
            pass
        if not self.dry:
            append_text(self.L.runner_log, line + "\n")

    # ------------------------------------------------------------ one pass
    def gather(self):
        c = {"now": self.now(), "notes": [], "sha": {}}
        c["jobs"], c["bad_jobs"] = self.load_jobs()
        c["rows"] = self.read_rows()
        c["procs"] = self.procs_fn()
        c["nvidia_text"] = self.nvidia_fn()
        c["gpu_mem"], c["gpu_name"] = parse_nvidia(c["nvidia_text"])
        c["nvidia_apps"] = self.apps_fn()
        c["free_commit"] = self.commit_fn()
        c["flags"] = {k: os.path.exists(getattr(self.L, k.lower())) for k in ("HOLD", "R0", "SLOW", "EXTEND")}
        c["fresh_locks"], c["stale_locks"] = read_locks(self.L.locks, time.time())
        c["markers"] = green_markers(self.L.green)
        c["code_sha"] = code_sha256(self.tree)
        return c

    def load_jobs(self):
        jobs, bad = {}, {}
        for p in sorted(glob.glob(os.path.join(self.L.jobs, "*.json"))):
            if os.path.basename(p).startswith(("._", ".")):
                continue
            d = QJ.read_json(p)
            try:
                j = QJ.normalize(d)
            except QJ.JobError as e:
                bad[os.path.basename(p)] = str(e)
                continue
            j["_file"] = p
            if j["name"] in jobs:
                bad[os.path.basename(p)] = f"duplicate name {j['name']}"
                continue
            jobs[j["name"]] = j
        return jobs, bad

    def running_names(self):
        return {r["name"] for r in self.state["running"].values() if r}

    def values(self):
        st, v = self.state, {}
        f = st["phase_b"].get("fired")
        if f:
            v["phase_b_ckpt"], v["phase_b_name"] = f["ckpt"], f["name"]
        ref = (st["steps"].get("reference") or {}).get("ref_ckpt")
        if ref:
            v["ref_ckpt"], v["ref_name"] = ref, QJ.checkpoint_name(self.L.abs(ref))
        v.update(read_extend(self.L.extend))
        return v

    def file_sha(self, c, rel):
        if rel not in c["sha"]:
            p = self.L.abs(rel)
            c["sha"][rel] = sha256_file(p) if os.path.isfile(p) else None
        return c["sha"][rel]

    def green_for(self, job, c):
        shas = {u: self.file_sha(c, u) for u in job["uses"]}
        if any(v is None for v in shas.values()):
            return None
        return green_match(c["markers"], c["code_sha"], shas)

    def reasons(self, job, c, v):
        """What keeps a pending job from launching (empty: eligible); module docstring."""
        st, now, name, r = self.state, c["now"], job["name"], []
        if job.get("not_before") and now < QJ.parse_time(job["not_before"]):
            r.append(f"not before {job['not_before']}")
        for a in job["after"]:
            if a in QJ.TRIGGERS:
                if not st["phase_b"].get("fired"):
                    r.append("waiting phase_b_done")
                continue
            row = c["rows"].get(a)
            if row is None:
                r.append(f"waiting {a}" + ("" if a in c["jobs"] else " (not queued)"))
            elif not row_ok(row):
                r.append(f"blocked: {a} ended {row.get('state')} (exit {row.get('exit_code')})")
        if job["soft_after"] and now < QJ.parse_time(job["soft_deadline"]):
            left = [s for s in job["soft_after"] if s not in c["rows"]]
            if left:
                r.append(f"soft-waiting {', '.join(left)} until {job['soft_deadline']}")
        missing = QJ.placeholders([job["argv"], job["requires"], job["outputs"], job["env"]]) - set(v)
        for ph in sorted(missing):
            why = PH_REASON.get(ph, f"waiting {{{ph}}}")
            if why.startswith("stage: ") and (st["steps"].get(why[7:]) or {}).get("ok"):
                why = f"waiting {{{ph}}} (the {why[7:]} stage gave none)"
            r.append(why)
        res = QJ.substitute(job, v)
        for p in res["requires"]:
            if not QJ.placeholders(p) and not os.path.exists(self.L.abs(p)):
                r.append(f"requires {p}")
        active = {n for n in c["jobs"] if n not in c["rows"]} | self.running_names()
        for g in job["not_if"]:
            m = sorted(n for n in active if n != name and fnmatch.fnmatchcase(n, g))
            if m:
                r.append(f"not_if {g}: {m[0]}")
        every = set(c["jobs"]) | set(c["rows"]) | self.running_names()
        for g in job["not_if_any"]:
            m = sorted(n for n in every if n != name and fnmatch.fnmatchcase(n, g))
            if m:
                r.append(f"not_if_any {g}: {m[0]}")
        if c["flags"]["HOLD"]:
            r.append("HOLD")
        if job["training"] and c["flags"]["R0"]:
            r.append("R0")
        r += self.lock_reasons(job, c)
        r += self.green_reasons(job, c)
        return list(dict.fromkeys(r))

    def green_reasons(self, job, c):
        """[] when a green marker vouches for every uses path. A uses path that is missing, or that run_suite.py
        does not hash (it hashes scripts/*.py and configs/*.yaml only), gets its own reason: no suite run can
        clear it, so it does not call for one."""
        bad = [u for u in job["uses"] if not MARKED_PATH.fullmatch(u) or self.file_sha(c, u) is None]
        if bad:
            return [f"uses {u}: " + ("missing" if self.file_sha(c, u) is None else "not hashed by run_suite.py")
                    for u in bad]
        return [] if self.green_for(job, c) else ["no green marker for " + ", ".join(job["uses"])]

    def lock_reasons(self, job, c):
        r = []
        for lk, text in c["fresh_locks"].items():
            if lock_covers_mdfm(text):
                r.append(f"lock {lk} covers mdfm/")
            for u in job["uses"]:
                if lock_covers(text, u):
                    r.append(f"lock {lk} covers {u}")
        return r

    def pass_once(self, dry=False):
        """One pass (module docstring); dry: decide and report, launch and write nothing. Returns the
        context dict (verdicts, reasons, the chosen jobs)."""
        self.dry = bool(dry)
        saved = copy.deepcopy(self.state) if dry else None
        try:
            return self._pass(dry)
        finally:
            if dry:
                self.state = saved
                self.dry = False

    def _pass(self, dry):
        self.pass_no += 1
        if not dry:
            self.flush_rows()
        c = self.gather()
        if not dry:
            self.poll_running(c)
            self.poll_suite(c)
        self.phase_b_trigger(c, dry)
        self.speed(c, dry)
        c["rows"] = self.read_rows()
        c["jobs"], c["bad_jobs"] = self.load_jobs()
        v = self.values()
        self.cpu_steps(c, dry)
        c["rows"] = self.read_rows()
        v = self.values()
        busy = self.running_names() | set(self.state["resume"])
        pending = [j for n, j in c["jobs"].items() if n not in c["rows"] and n not in busy]
        c["reasons"] = {j["name"]: self.reasons(j, c, v) for j in pending}
        eligible = [j for j in pending if not c["reasons"][j["name"]]]
        key = lambda j: (j["cls"], j["priority"], j["name"])  # noqa: E731
        cls1 = any(j["cls"] == 1 for j in eligible)
        c["eligible_gpu"] = sorted([j for j in eligible if j["gpu"] and not (j["training"] and cls1)], key=key)
        c["eligible_cpu"] = sorted([j for j in eligible if not j["gpu"]], key=key)
        c["gpu_free"], c["gpu_why"] = gpu_verdict(c["procs"], c["nvidia_text"], self.L.gpu_lock, c.get("nvidia_apps"))
        c["launched"] = []
        self.stage_for_waiting(c, pending, dry)
        self.launch_gpu(c, v, dry)
        self.launch_cpu(c, v, dry)
        if not dry and self.state["block_dirty"] and self.state["running"]["gpu"] is None:
            self.block_steps(c)
        need = [n for n, why in c["reasons"].items() if why and all(w.startswith("no green marker") for w in why)]
        c["suite_needed"] = need
        if not dry:
            self.start_suite(c, need)
        self.save_state()
        status = self.status_text(c, dry)
        c["status"] = status
        if not dry:
            write_text(self.L.heartbeat, f"{iso(c['now'])} pid {self.pid} pass {self.pass_no}\n")
            try:                                   # STATUS.md held open by a reader: the pass still counts
                write_text(self.L.status, status)
            except OSError as e:
                self.log(f"STATUS.md not written: {e!r}")
        return c

    # ------------------------------------------------------------ launching
    def child_env(self, job):
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        if job["gpu"]:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        env.update(job["env"])
        if not job["gpu"]:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        return env

    def estimate_hours(self, job, resume=False):
        """Hours to the end of the job (module docstring, deadline guard)."""
        if not job["training"] or not job.get("train_steps"):
            return float(job["est_hours"])
        phase = job.get("train_phase") or "B"
        rate = job.get("s_per_step") or (baseline_rate(self.L.baseline) if phase == "B" else None)
        if not rate:
            return float(job["est_hours"])
        done = 0
        if resume:
            done = last_step(read_jsonl(os.path.join(self.L.abs(job["run_dir"]), "log.jsonl")), phase) or 0
        h = max(int(job["train_steps"]) - done, 0) * float(rate) / 3600.0 + TRAIN_OVERHEAD_H
        if phase == "B":
            h += self.post_train_hours(os.path.basename(job["run_dir"].rstrip("/")))
        return h

    def post_train_hours(self, run):
        h = 0.0
        for t in self.load_templates():
            if QJ.runs_match(t["runs"], run):
                h += sum(float(j.get("est_hours", 0)) for j in t["jobs"])
        return h

    def load_templates(self):
        out = []
        for p in sorted(glob.glob(os.path.join(self.L.templates, "*.json"))):
            if os.path.basename(p).startswith(("._", ".")):
                continue
            t = QJ.read_json(p)
            try:
                out.append(QJ.normalize_template(t))
            except QJ.JobError as e:
                self.log(f"template {os.path.basename(p)} refused: {e}")
        return out

    def deadline_ok(self, job, c, resume=False):
        end = c["now"] + dt.timedelta(hours=self.estimate_hours(job, resume))
        return end <= self.deadline, end

    def skip_deadline(self, job, end, c):
        self.log(f"{job['name']}: skipped_deadline (estimated end {iso(end)} after {iso(self.deadline)})")
        self.append_row({"name": job["name"], "cls": job["cls"], "priority": job["priority"],
                         "state": "skipped_deadline", "argv": json.dumps(job["argv"]),
                         "note": f"estimated end {iso(end)} after the deadline {iso(self.deadline)}"})
        c["rows"] = self.read_rows()

    def launch_gpu(self, c, v, dry):
        st = self.state
        c["gpu_next"] = None
        if st["running"]["gpu"] is not None:
            return
        resumes = sorted(st["resume"].items())
        if any(j["cls"] == 1 for j in c["eligible_gpu"] + c["eligible_cpu"]):
            resumes = []                         # a training resume, like a training job, waits for class 1
        if c["flags"]["HOLD"] or c["now"] >= self.until:
            c["gpu_next"] = ("HOLD" if c["flags"]["HOLD"] else "past --until") + ": no launch"
            return
        for name, rs in resumes:
            job = rs["job"]
            why = [w for w in self.reasons(job, c, v) if w in ("HOLD", "R0") or w.startswith(("lock ", "no green"))]
            if why:
                c["gpu_next"] = f"resume of {name} waits: {'; '.join(why)}"
                return
            c["gpu_next"] = f"resume of {name}"
            if not c["gpu_free"]:
                return
            ok, end = self.deadline_ok(job, c, resume=True)
            if not ok:
                st["resume"].pop(name)
                self.skip_deadline(job, end, c)
                continue
            if not dry:
                st["resume"].pop(name)
                self.launch(job, "gpu", c, v, argv=rs["argv"], note=f"resume after {rs['reason']}")
            c["launched"].append(name)
            return
        for job in list(c["eligible_gpu"]):
            if job["name"] in c["rows"]:
                continue
            c["gpu_next"] = job["name"]
            if not c["gpu_free"]:
                return
            ok, end = self.deadline_ok(job, c)
            if not ok:
                if not dry:
                    self.skip_deadline(job, end, c)
                continue
            if dry:
                c["launched"].append(job["name"])
                return
            if job["training"] and st["block_dirty"]:
                self.block_steps(c)
            self.launch(job, "gpu", c, v)
            c["launched"].append(job["name"])
            return

    def launch_cpu(self, c, v, dry):
        c["cpu_next"] = None
        if self.state["running"]["cpu"] is not None or c["flags"]["HOLD"] or c["now"] >= self.until:
            return
        for job in c["eligible_cpu"]:
            c["cpu_next"] = job["name"]
            ok, end = self.deadline_ok(job, c)
            if not ok:
                if not dry:
                    self.skip_deadline(job, end, c)
                continue
            if not dry:
                self.launch(job, "cpu", c, v)
            c["launched"].append(job["name"])
            return

    def launch(self, job, slot, c, v, argv=None, note=""):
        """Start a job in a slot ('gpu' or 'cpu'), record it in state.json and the status files."""
        st, now = self.state, c["now"]
        j = QJ.substitute({k: x for k, x in job.items() if k != "_file"}, v)
        argv = list(argv or j["argv"])
        env = self.child_env(j)
        os.makedirs(self.L.logs, exist_ok=True)
        log_path = os.path.join(self.L.logs, f"{j['name']}.log")
        pos = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        attempt = len(st["attempts"].get(j["name"], [])) + 1
        rec = {"name": j["name"], "job": j, "slot": slot, "start": iso(now), "attempt": attempt, "argv": argv,
               "env": {k: env.get(k, "") for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}, "log": log_path,
               "log_pos": pos, "note": note, "code_sha256": c["code_sha"],
               "entry_sha256": self.file_sha(c, j["uses"][0]) if j["uses"] else None,
               "gpu_name": c["gpu_name"] if slot == "gpu" else "", "snapshot": None, "snapshot_sha256": None,
               "jsonl_count": None}
        if j["training"]:
            run_dir = self.L.abs(j["run_dir"])
            snap, ssha = code_snapshot(self.tree, run_dir, now)
            rec["snapshot"], rec["snapshot_sha256"] = QJ.tree_rel(snap, self.tree), ssha
            rec["jsonl_count"] = len(read_jsonl(os.path.join(run_dir, "log.jsonl")))
        kind = "above" if j["training"] else ("normal" if slot == "gpu" else "below")
        if slot == "gpu":
            write_text(self.L.gpu_lock, json.dumps({"pid": self.pid, "job": j["name"], "start": iso(now)}) + "\n")
        try:
            with open(log_path, "ab") as f:
                f.write(f"=== {iso(now)} gpu_queue attempt {attempt}: {json.dumps(argv)}\n".encode("utf-8"))
                f.flush()
                p = self.popen(argv, cwd=self.tree, env=env, stdout=f, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, creationflags=_flags(kind))
        except OSError as e:                     # the program could not start: logged, the slot stays free
            if slot == "gpu":
                remove(self.L.gpu_lock)
            self.log(f"{j['name']}: launch failed: {e!r}")
            st["attempts"].setdefault(j["name"], []).append({"start": iso(now), "end": iso(now), "exit": None,
                                                             "timed_out": False, "argv": argv})
            self.row(rec, "failed", "", c, note=f"launch failed: {e!r}")
            return
        rec["pid"] = p.pid
        rec["create_time"] = _create_time(p.pid)
        self.children[slot] = Child(popen=p)
        st["running"][slot] = rec
        if slot == "gpu":
            write_text(self.L.gpu_lock, json.dumps({"pid": self.pid, "create_time": _create_time(self.pid),
                                                    "job": j["name"], "child": p.pid, "start": iso(now)}) + "\n")
            write_text(self.L.current_pid, f"{p.pid}\n")
        self.log(f"launched {j['name']} ({slot}, PID {p.pid}, attempt {attempt}): {' '.join(argv)}")
        self.save_state()

    # ------------------------------------------------------------ exits
    def poll_running(self, c):
        for slot in ("gpu", "cpu"):
            rec = self.state["running"].get(slot)
            if not rec:
                continue
            child = self.children.get(slot)
            if child is None:
                child = self.adopt(slot, rec, c)
                if child is None:
                    continue
            ended, rc = child.poll()
            if ended:
                self.finish(slot, rec, rc, c)
                continue
            limit = rec["job"].get("timeout_hours")
            if limit and c["now"] - QJ.parse_time(rec["start"]) > dt.timedelta(hours=float(limit)):
                self.log(f"{rec['name']}: timeout after {limit} h, killing the process tree of PID {child.pid}")
                kill_tree(child.pid)
                for _ in range(60):
                    ended, rc = child.poll()
                    if ended:
                        break
                    time.sleep(1)
                self.finish(slot, rec, rc, c, timed_out=True)

    def adopt(self, slot, rec, c):
        """Adopt the process of a running job of state.json (restart), or settle a job whose process ended
        while the runner was down; returns the Child or None."""
        p = proc_by_pid(c["procs"], rec.get("pid"))
        if rec.get("pid") and same_process(p, rec.get("create_time")):
            ch = Child(pid=rec["pid"])
            self.children[slot] = ch
            if slot == "gpu":
                write_text(self.L.gpu_lock, json.dumps({"pid": self.pid, "create_time": _create_time(self.pid),
                                                        "job": rec["name"], "child": rec["pid"],
                                                        "start": rec["start"]}) + "\n")
                write_text(self.L.current_pid, f"{rec['pid']}\n")
            self.log(f"adopted {rec['name']} (PID {rec['pid']}), started {rec['start']}")
            return ch
        name = rec["name"]
        if rec["job"]["training"]:
            self.log(f"{name}: its process ended while the runner was down; judged from its run folder")
            self.finish(slot, rec, None, c, how="ended while the runner was down")
            return None
        n = self.state["reruns"].get(name, 0) + 1
        self.state["reruns"][name] = n
        self.state["running"][slot] = None
        self._release(slot)
        if n > 1:
            self.row(rec, "failed", "", c, note="its process ended twice while the runner was down")
        else:
            self.log(f"{name}: its process ended while the runner was down; it runs again")
        return None

    def _release(self, slot):
        """After a GPU job: remove gpu.lock (when this runner holds it) and current.pid; the block is dirty."""
        if slot == "gpu":
            info = QJ.read_json(self.L.gpu_lock)
            if not isinstance(info, dict) or info.get("pid") in (self.pid, None):
                remove(self.L.gpu_lock)
            remove(self.L.current_pid)
            self.state["block_dirty"] = True

    def row(self, rec, state, rc, c, note="", config_sha=""):
        job = rec["job"]
        notes = [x for x in (rec.get("note"), note) if x]
        self.append_row({
            "name": job["name"], "cls": job["cls"], "priority": job["priority"], "state": state,
            "argv": json.dumps(rec.get("argv") or job["argv"]),
            "cuda_visible_devices": rec.get("env", {}).get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": rec.get("env", {}).get("OMP_NUM_THREADS", ""), "start": rec.get("start", ""),
            "end": iso(c["now"]), "exit_code": "" if rc is None else rc, "outputs": ";".join(job.get("outputs") or []),
            "gpu_name": rec.get("gpu_name") or "", "code_sha256": rec.get("code_sha256") or "",
            "entry_sha256": rec.get("entry_sha256") or "", "config_sha256": config_sha or "",
            "snapshot_sha256": rec.get("snapshot_sha256") or "", "note": "; ".join(notes)})
        c["rows"] = self.read_rows()

    def finish(self, slot, rec, rc, c, timed_out=False, how=""):
        """The end of a job (module docstring, exits)."""
        st, job, name = self.state, rec["job"], rec["name"]
        self.children.pop(slot, None)
        st["running"][slot] = None
        self._release(slot)
        att = st["attempts"].setdefault(name, [])
        att.append({"start": rec["start"], "end": iso(c["now"]), "exit": rc, "timed_out": timed_out,
                    "argv": rec["argv"]})
        note = "; ".join(x for x in (how, f"attempt {len(att)}" if len(att) > 1 else "") if x)
        self.log(f"{name} ended: exit {rc}{' (timeout)' if timed_out else ''}")
        if not job["training"]:
            state = "timeout" if timed_out else ("done" if rc == 0 else "failed")
            if timed_out:
                note = "; ".join(x for x in (note, f"timeout after {job.get('timeout_hours')} h, tree killed") if x)
            self.row(rec, state, rc, c, note=note)
            return
        run_dir = self.L.abs(job["run_dir"])
        recs = read_jsonl(os.path.join(run_dir, "log.jsonl"))
        seg = recs[rec["jsonl_count"]:] if rec.get("jsonl_count") is not None else last_call(recs)
        tail = tail_text(rec["log"], rec.get("log_pos", 0))
        cfg_sha = next((r.get("config_sha256") for r in seg if r.get("event") == "start"), "")
        verify_b = [r for r in seg if r.get("event") == "verify" and r.get("phase") == "B"]
        if timed_out:
            state, note = "timeout", "; ".join(x for x in (note, f"timeout after {job.get('timeout_hours')} h, "
                                                                 "tree killed") if x)
        else:
            kind = classify(rc, tail, seg)
            if kind == "ok":
                state = "done"
            elif kind in ("alarm", "provenance"):
                self.write_r0(f"{name}: {kind} (exit {rc})")
                state, note = "r0", "; ".join(x for x in (note, f"{kind}: R0 written") if x)
            elif len(att) <= int(job.get("max_resumes", 1)):
                cks = checkpoints(run_dir)
                ck = QJ.tree_rel(cks[0], self.tree) if cks else None
                st["resume"][name] = {"argv": with_resume(rec["argv"], ck), "reason": f"{kind} (exit {rc})",
                                      "time": iso(c["now"]), "job": job}
                self.log(f"{name}: {kind} (exit {rc}); resume queued from {ck}")
                return
            else:
                self.write_r0(f"{name}: second failure, {kind} (exit {rc})")
                state, note = "r0", "; ".join(x for x in (note, f"{kind} after its resume: R0 written") if x)
        self.row(rec, state, rc, c, note=note, config_sha=cfg_sha)
        if verify_b:
            self.expand_for(job, verify_b[-1], c)

    def write_r0(self, reason):
        if self.dry:
            return
        append_text(self.L.r0, f"{iso(self.now())} {reason}\n")
        self.log(f"R0 written: {reason}")

    # ------------------------------------------------------------ post-train templates
    def expand_for(self, job, verify, c):
        """Write the job files of the templates whose runs filter matches the run (once per run)."""
        st = self.state
        run_rel = job["run_dir"].rstrip("/")
        run = os.path.basename(run_rel)
        if run in st["expanded"] or self.dry:
            return
        ck = QJ.tree_rel(self.L.abs(QJ.tree_rel(verify["path"], self.tree)), self.tree)
        vals = {"ckpt": ck, "name": QJ.checkpoint_name(self.L.abs(ck)), "run_dir": run_rel, "run": run,
                "config": config_of(job["argv"]) or QJ.CONFIG1, "ckpt_a4000": f"{run_rel}/ckpt_A_004000.pt",
                "ckpt_b4000": f"{run_rel}/ckpt_B_004000.pt"}
        names = []
        for t in self.load_templates():
            if not QJ.runs_match(t["runs"], run):
                continue
            for tj in t["jobs"]:
                jd = QJ.substitute(dict(tj, cls=t["cls"]), vals)
                jd.update(template=t["name"], run=run)
                try:
                    QJ.write_job(jd, tree=self.tree)
                    names.append(jd["name"])
                except QJ.JobError as e:
                    self.log(f"template {t['name']}: {e}")
        st["expanded"][run] = {"time": iso(c["now"]), "ckpt": ck, "jobs": names}
        self.log(f"post-train templates expanded for {run} ({ck}): {len(names)} jobs")

    # ------------------------------------------------------------ the trigger phase_b_done
    def phase_b_trigger(self, c, dry):
        st, pb = self.state, self.state["phase_b"]
        conf = self.pb
        if pb.get("fired"):
            f = pb["fired"]
            c["trigger"] = f"phase_b_done fired {f['time']}: {f['ckpt']} ({f['how']})"
            return
        if not conf.get("pid"):
            c["trigger"] = "no phase_b_done trigger configured"
            return
        run_dir = self.L.abs(conf["run_dir"])
        recs = read_jsonl(os.path.join(run_dir, "log.jsonl"))
        p = proc_by_pid(c["procs"], conf["pid"])
        cmd = (p or {}).get("cmdline") or ""
        # a python process at that PID whose command line cannot be read (access denied) counts as the phase B
        # still running: reading it as gone would queue a resume over a live run
        match = bool(p) and is_python(p.get("name")) and (not cmd or (
            "train_stage1.py" in cmd and re.search(r"--phase\s+B(\s|$)", cmd) is not None))
        if "pid_check" not in pb:
            pb["pid_check"] = {"time": iso(c["now"]), "pid": conf["pid"], "matches": match,
                               "cmdline": _short(cmd, 300) if cmd or not p else "(command line unreadable)",
                               "create_time": (p or {}).get("create_time")}
        alive = match and same_process(p, pb["pid_check"].get("create_time"))
        c["pid_check"] = pb["pid_check"]
        step = last_step(last_call(recs), conf["phase"])
        if alive:
            c["trigger"] = (f"waiting phase_b_done: PID {conf['pid']} (train_stage1.py --phase B) runs, last logged "
                            f"step {step if step is not None else 'none'} of {conf['steps']:,}")
            return
        rj = conf["resume_job"]
        ver = [r for r in recs if r.get("event") == "verify" and r.get("phase") == conf["phase"]
               and r.get("step") == conf["steps"]]
        launched = rj in self.running_names() or rj in st["resume"]
        queued = rj in c["jobs"] and rj not in c["rows"] and not launched
        if launched or (queued and not ver):
            c["trigger"] = f"waiting phase_b_done: the resume job {rj} is queued or running (last step {step})"
            return
        if ver and queued:                       # the phase end came after all: the resume is not needed
            self.log(f"{rj}: not launched, the phase B verify event at step {conf['steps']:,} is in the log")
            self.append_row({"name": rj, "cls": c["jobs"][rj]["cls"], "priority": c["jobs"][rj]["priority"],
                             "state": "skipped", "argv": json.dumps(c["jobs"][rj]["argv"]),
                             "note": f"phase B logged its verify event at step {conf['steps']:,}; resume not needed"})
            c["rows"] = self.read_rows()
        if ver:
            v = ver[-1]
            idx = max(i for i, r in enumerate(recs) if r is v)
            starts = [i for i, r in enumerate(recs[:idx]) if r.get("event") == "start"]
            call = recs[starts[-1] if starts else 0:idx + 1]
            alarms = [r for r in call if r.get("event") == "alarm"]
            if alarms:
                self.write_r0(f"phase B of {conf['run_dir']}: alarm at step {alarms[-1].get('step')} "
                              f"({alarms[-1].get('source')}); evaluations go on")
            self.fire(c, v.get("path"), f"verify event of {v.get('time')}", dry)
            return
        if pb.get("gone_since") is None:
            pb["gone_since"] = iso(c["now"])
        waited = (c["now"] - QJ.parse_time(pb["gone_since"])).total_seconds()
        if waited < VERIFY_WAIT_S:
            c["trigger"] = (f"waiting phase_b_done: PID {conf['pid']} gone since {pb['gone_since']}, waiting "
                            f"{(VERIFY_WAIT_S - waited) / 60:.0f} more min for the verify event (last step {step})")
            return
        tail = tail_text(self.L.abs(conf["text_log"]))
        resumed = rj in c["rows"] or pb.get("resume_queued")
        if resumed and rj in c["jobs"]:
            tail += "\n" + tail_text(os.path.join(self.L.logs, f"{rj}.log"))
        kind = classify(None, tail, [r for r in last_call(recs) if r.get("phase") == conf["phase"]])
        if kind == "ok":                               # 'end' and 'verify' of a shortened call: not the phase end
            kind = "crash"
        if kind in ("alarm", "provenance") or resumed:
            why = kind if not resumed else f"{kind} after the resume"
            if not pb.get("r0_written") and not (resumed and rj in c["rows"] and c["rows"][rj].get("state") == "r0"):
                self.write_r0(f"phase B of {conf['run_dir']}: {why}, no verify event at step {conf['steps']:,}")
                pb["r0_written"] = iso(c["now"])
            if dry:
                c["trigger"] = f"phase_b_done would fall back ({why}): the latest ckpt_B that checkpoint.load verifies"
                return
            last = pb.get("no_fallback")
            if last and (c["now"] - QJ.parse_time(last)).total_seconds() < STEP_RETRY_S:
                c["trigger"] = f"phase_b_done cannot fire: {why} and no ckpt_B verified at {last} (retried every 30 min)"
                return
            cks = [QJ.tree_rel(x, self.tree) for x in checkpoints(run_dir, conf["phase"])]
            got = self.verify_fn(cks) if cks else None
            if got:
                self.fire(c, got, f"R0 fallback ({why}): latest verified ckpt_B", dry)
            else:
                c["trigger"] = f"phase_b_done cannot fire: {why} and no ckpt_B verifies"
                pb["no_fallback"] = iso(c["now"])
            return
        cks = checkpoints(run_dir, conf["phase"])
        if not cks:
            if not pb.get("r0_written"):
                self.write_r0(f"phase B of {conf['run_dir']}: {kind} and no ckpt_B to resume from")
                pb["r0_written"] = iso(c["now"])
            c["trigger"] = f"phase_b_done cannot fire: {kind}, no ckpt_B"
            return
        ck = QJ.tree_rel(cks[0], self.tree)
        if dry:
            c["trigger"] = f"waiting phase_b_done: phase B ended by {kind}; a resume from {ck} would be queued"
            return
        job = {"name": rj, "cls": 2, "priority": 0, "gpu": True, "training": True, "run_dir": conf["run_dir"],
               "argv": [self.python, "scripts/train_stage1.py", "--phase", "B", "--resume", ck] + conf["resume_args"],
               "train_phase": "B", "train_steps": conf["steps"], "max_resumes": 0, "group": "phase_b",
               "s_per_step": baseline_rate(self.L.baseline),
               "est_hours": max(0.5, (conf["steps"] - (step or 0)) * (baseline_rate(self.L.baseline) or 2.5) / 3600.0
                                + TRAIN_OVERHEAD_H),
               "outputs": [f"{conf['run_dir']}/ckpt_B_{conf['steps']:06d}.pt"],
               "note": f"one resume of the phase B of {conf['run_dir']} after {kind}"}
        try:
            QJ.write_job(job, tree=self.tree)
        except QJ.JobError as e:
            self.log(f"phase B resume job refused: {e}")
            return
        pb["resume_queued"] = iso(c["now"])
        self.log(f"phase B ended by {kind} without its verify event; resume job {rj} queued from {ck}")
        c["trigger"] = f"waiting phase_b_done: phase B ended by {kind}; resume job {rj} queued from {ck}"

    def fire(self, c, path, how, dry):
        ck = QJ.tree_rel(self.L.abs(QJ.tree_rel(path, self.tree)), self.tree)
        f = {"time": iso(c["now"]), "ckpt": ck, "name": QJ.checkpoint_name(self.L.abs(ck)), "how": how}
        c["trigger"] = f"phase_b_done fires: {ck} ({how})"
        if dry:
            return
        self.state["phase_b"]["fired"] = f
        self.log(f"phase_b_done fired: {ck} ({how})")

    def _verify_subprocess(self, paths):
        """The first of `paths` that checkpoint.load verifies, read by a CPU child (--verify-checkpoints)."""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
        argv = [self.python, os.path.join("scripts", "gpu_queue.py"), "--verify-checkpoints", *paths]
        try:
            r = subprocess.run(argv, cwd=self.tree, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=VERIFY_TIMEOUT_S, creationflags=_flags("below"))
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"checkpoint verification failed to run: {e!r}")
            return None
        for line in reversed(r.stdout.splitlines()):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "verified" in d:
                self.log(f"checkpoint verification: {d.get('verified')} (refused: {d.get('errors')})")
                return d.get("verified")
        self.log(f"checkpoint verification gave no answer (exit {r.returncode}): {r.stderr[-300:]}")
        return None

    # ------------------------------------------------------------ speed
    def speed(self, c, dry):
        views = []
        conf, pb = self.pb, self.state["phase_b"]
        p = proc_by_pid(c["procs"], conf.get("pid"))
        if conf.get("pid") and pb.get("pid_check", {}).get("matches") and not pb.get("fired") and \
                same_process(p, pb["pid_check"].get("create_time")) and "train_stage1.py" in (p or {}).get("cmdline", ""):
            views.append({"run": os.path.basename(conf["run_dir"]), "run_dir": conf["run_dir"], "phase": "B",
                          "steps": conf["steps"], "offset": None})
        rec = self.state["running"].get("gpu")
        if rec and rec["job"]["training"]:
            j = rec["job"]
            views.append({"run": os.path.basename(j["run_dir"]), "run_dir": j["run_dir"],
                          "phase": j.get("train_phase"), "steps": j.get("train_steps"), "offset": rec.get("jsonl_count")})
        rows = []
        for vw in views:
            recs = read_jsonl(os.path.join(self.L.abs(vw["run_dir"]), "log.jsonl"))
            seg = recs[vw["offset"]:] if vw["offset"] is not None else last_call(recs)
            iv = clean_intervals(seg, vw["phase"])
            rate = span_rate(iv, "last")
            base = baseline_rate(self.L.baseline) if vw["phase"] == "B" else span_rate(iv, "first")
            step = last_step(seg, vw["phase"])
            eta = None
            if rate and step is not None and vw["steps"]:
                eta = c["now"] + dt.timedelta(seconds=max(int(vw["steps"]) - step, 0) * rate)
            rows.append(dict(vw, rate=rate, base=base, step=step, eta=eta,
                             ratio=(rate / base) if rate and base else None))
        c["speed"] = rows
        slow = [r for r in rows if r["ratio"] is not None and r["ratio"] > SLOW_FACTOR]
        c["slow"] = slow
        if dry:
            return
        if slow:
            text = "".join(f"{iso(c['now'])} {r['run']} phase {r['phase']} step {r['step']}: {r['rate']:.4f} s/step, "
                           f"baseline {r['base']:.4f}, ratio {r['ratio']:.3f} > {SLOW_FACTOR}\n" for r in slow)
            if not os.path.exists(self.L.slow):
                self.log("SLOW written: " + text.strip())
            write_text(self.L.slow, text)
        elif os.path.exists(self.L.slow):
            remove(self.L.slow)
            self.log("SLOW deleted: " + ("; ".join(f"{r['run']} {r['rate']:.4f} s/step" for r in rows if r["rate"])
                                         or "no training runs"))

    # ------------------------------------------------------------ CPU steps
    def run_cmd(self, key, script, args, c, timeout=STEP_TIMEOUT_S):
        """Run scripts/<script> synchronously (BelowNormal, CUDA hidden); a row step:<key>; returns the exit
        code (None when it did not run)."""
        stamp = c["now"].strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.L.logs, exist_ok=True)
        log = os.path.join(self.L.logs, f"step_{key}_{stamp}.log")
        argv = [self.python, f"scripts/{script}", *args]
        env = self.child_env({"gpu": False, "env": {}})
        start = self.now()
        rc, note = None, ""
        try:
            with open(log, "ab") as f:
                r = subprocess.run(argv, cwd=self.tree, env=env, stdout=f, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, timeout=timeout, creationflags=_flags("below"))
            rc = r.returncode
        except subprocess.TimeoutExpired:
            note = f"timeout after {timeout / 60:.0f} min"
        except OSError as e:
            note = repr(e)
        self.log(f"step {key}: exit {rc} {note}".rstrip())
        self.append_row({"name": f"step:{key}", "cls": 0, "priority": 0, "state": "done" if rc == 0 else "failed",
                         "argv": json.dumps(argv), "cuda_visible_devices": "-1",
                         "omp_num_threads": env.get("OMP_NUM_THREADS", ""), "start": iso(start),
                         "end": iso(self.now()), "exit_code": "" if rc is None else rc,
                         "outputs": QJ.tree_rel(log, self.tree), "code_sha256": c["code_sha"], "note": note})
        return rc

    def once_step(self, key, script, args, c, dry):
        """A weekend_branch.py stage run once (three tries, 30 min apart); True once it passed."""
        s = self.state["steps"].get(key) or {}
        if s.get("ok"):
            return True
        if not os.path.isfile(os.path.join(self.tree, "scripts", script)):
            c["notes"].append(f"step {key}: scripts/{script} absent, not run")
            return False
        if s.get("tries", 0) >= STEP_TRIES or (s.get("time") and (c["now"] - QJ.parse_time(s["time"]))
                                               .total_seconds() < STEP_RETRY_S):
            return False
        if dry:
            c["notes"].append(f"step {key} would run now: scripts/{script} {' '.join(args)}")
            return False
        rc = self.run_cmd(key, script, args, c)
        s = {"time": iso(c["now"]), "exit": rc, "ok": rc == 0, "tries": s.get("tries", 0) + 1}
        if key == "reference" and rc == 0:
            s["ref_ckpt"] = self.reference_checkpoint()
        self.state["steps"][key] = s
        self.save_state()
        return rc == 0

    def reference_checkpoint(self):
        files = sorted(glob.glob(os.path.join(self.L.reports, "branch_reference_*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("._")]
        d = QJ.read_json(files[-1]) if files else None
        v = d.get("reference_checkpoint") if isinstance(d, dict) else None
        if isinstance(v, dict):
            v = v.get("path") or v.get("checkpoint")
        return QJ.tree_rel(v, self.tree) if isinstance(v, str) and v else None

    def j3_done(self, c):
        names = set(J3_NAMES) | {n for n, j in c["jobs"].items() if j.get("group") == "J3"}
        present = [n for n in names if n in c["jobs"] or n in c["rows"]]
        if not present or any(n not in c["rows"] for n in present):
            return False
        missing = [n for n in J3_NAMES if n not in c["jobs"] and n not in c["rows"]]
        return not missing or c["now"] >= self.j3_deadline

    def cpu_steps(self, c, dry):
        if self.j3_done(c):
            self.once_step("branch", "weekend_branch.py", ["--stage", "branch", "--write"], c, dry)
        exp = self.state["expanded"].get(SEED2_RUN)
        if exp and exp.get("jobs") and all(n in c["rows"] for n in exp["jobs"]):
            self.once_step("seeds", "weekend_branch.py", ["--stage", "seeds", "--write"], c, dry)

    def stage_for_waiting(self, c, pending, dry):
        """Run the reference or extension stage when the best job waiting on one comes before every eligible
        GPU job (classes 4 and 5)."""
        key = lambda j: (j["cls"], j["priority"], j["name"])  # noqa: E731
        waiting = sorted([j for j in pending if c["reasons"][j["name"]] and
                          all(w.startswith("stage: ") for w in c["reasons"][j["name"]])], key=key)
        if not waiting or c["flags"]["HOLD"]:
            return
        first = waiting[0]
        if c["eligible_gpu"] and key(c["eligible_gpu"][0]) < key(first):
            return
        stage = c["reasons"][first["name"]][0][7:]
        self.once_step(stage, "weekend_branch.py", ["--stage", stage, "--write"], c, dry)

    def block_steps(self, c):
        for script in ("summarize_weekend.py", "gate_table.py"):
            if os.path.isfile(os.path.join(self.tree, "scripts", script)):
                self.run_cmd(script[:-3], script, [], c)
        self.state["block_dirty"] = False

    # ------------------------------------------------------------ the suite
    def poll_suite(self, c):
        s = self.state["suite"]
        if self.suite_child is None and s.get("running") and s.get("pid"):
            p = proc_by_pid(c["procs"], s["pid"])
            if same_process(p, s.get("create_time")):
                self.suite_child = Child(pid=s["pid"])
            else:
                s["running"] = False
        if self.suite_child is None:
            return
        ended, rc = self.suite_child.poll()
        free = c.get("free_commit")
        if not ended and free is not None and free < SUITE_STOP_FREE_GB * GB:
            kill_tree(self.suite_child.pid)
            self.suite_child = None
            s.update(running=False, last_exit=None, last_end=iso(c["now"]),
                     last_line=f"stopped: free commit {free / GB:.1f} GB under {SUITE_STOP_FREE_GB:g} GB")
            self.log(f"run_suite stopped (PID {s.get('pid')}): {s['last_line']}")
            return
        if not ended:
            return
        self.suite_child = None
        lines = [x for x in tail_text(s.get("log", "")).splitlines() if x.startswith("run_suite:")]
        s.update(running=False, last_exit=rc, last_end=iso(c["now"]), last_line=lines[-1] if lines else "")
        self.log(f"run_suite ended: exit {rc}; {s['last_line']}")
        self.suite_streak(s, rc)

    def suite_streak(self, s, rc):
        """Count the suite runs in a row that fail for the same jobs (state.json suite.fail_streak); from the
        SUITE_STREAK-th on, runner.log names the failing tests of the last run. A pass ends the streak; an unknown exit
        code (an adopted run) leaves it as it is."""
        if rc is None:
            return
        if rc == 0:
            s.pop("fail_streak", None)
            return
        jobs = sorted(s.get("for_jobs") or [])
        prev = s.get("fail_streak") or {}
        n = int(prev.get("count", 0)) + 1 if prev.get("jobs") == jobs else 1
        tests = suite_failed_tests(tail_text(s.get("log", ""), 0, 1 << 22))
        s["fail_streak"] = {"jobs": jobs, "count": n, "exit": rc, "tests": tests, "log": s.get("log", "")}
        if n >= SUITE_STREAK:
            self.log(streak_line(s["fail_streak"]))

    def start_suite(self, c, need):
        s = self.state["suite"]
        if not need or s.get("running") or self.suite_child is not None:
            return
        if s.get("last_start") and (c["now"] - QJ.parse_time(s["last_start"])).total_seconds() < SUITE_EVERY_S:
            return
        free = c.get("free_commit")
        if free is not None and free < SUITE_START_FREE_GB * GB:
            line = f"run_suite held: free commit {free / GB:.1f} GB under {SUITE_START_FREE_GB:g} GB"
            if not s.get("held"):                  # logged once on entering the hold, not at every pass
                self.log(line)
            s["held"] = line
            return
        s.pop("held", None)
        os.makedirs(self.L.logs, exist_ok=True)
        log = os.path.join(self.L.logs, f"run_suite_{c['now'].strftime('%Y%m%d_%H%M%S')}.log")
        argv = [self.python, "scripts/run_suite.py", "--task", SUITE_TASK, "--wait-hours", "1"]
        env = self.child_env({"gpu": False, "env": {"OMP_NUM_THREADS": "2"}})
        with open(log, "ab") as f:
            p = self.popen(argv, cwd=self.tree, env=env, stdout=f, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, creationflags=_flags("below"))
        self.suite_child = Child(popen=p)
        s.update(running=True, pid=p.pid, create_time=_create_time(p.pid), last_start=iso(c["now"]), log=log,
                 for_jobs=sorted(need)[:10])
        self.log(f"run_suite started (PID {p.pid}) for {len(need)} jobs without a green marker")

    # ------------------------------------------------------------ startup
    def other_runner(self, procs):
        try:
            with open(self.L.runner_pid, encoding="utf-8") as f:
                pid = int(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            return None
        p = proc_by_pid(procs, pid)
        cmd = (p or {}).get("cmdline") or ""
        if pid != self.pid and p and "gpu_queue.py" in cmd and "--dry-run" not in cmd and "--verify" not in cmd:
            return pid
        return None

    def startup(self):
        """Restart safety (module docstring): adopt or settle the running jobs of state.json, judge training
        jobs launched without a state, expand templates a finished run is owed. Returns False when another
        runner holds runner.pid."""
        c = self.gather()
        other = self.other_runner(c["procs"])
        if other:
            self.log(f"another runner (PID {other}) holds {self.L.runner_pid}; not starting")
            return False
        write_text(self.L.runner_pid, f"{self.pid}\n")
        self.log(f"runner started (PID {self.pid}), until {iso(self.until)}, deadline {iso(self.deadline)}")
        for slot in ("gpu", "cpu"):
            rec = self.state["running"].get(slot)
            if rec:
                self.adopt(slot, rec, c)
        if self.state["running"].get("gpu") is None:
            if not gpu_lock_holder(self.L.gpu_lock, c["procs"]):
                remove(self.L.gpu_lock)
            remove(self.L.current_pid)
        c["rows"] = self.read_rows()
        known = self.running_names() | set(self.state["resume"])
        for name, job in c["jobs"].items():
            if not job["training"] or name in c["rows"] or name in known:
                continue
            launched = name in self.state["attempts"] or os.path.exists(os.path.join(self.L.logs, f"{name}.log"))
            if not launched:
                continue
            self.settle_training(job, c)
        for name, job in c["jobs"].items():
            row = c["rows"].get(name)
            run = os.path.basename((job.get("run_dir") or "").rstrip("/"))
            if job["training"] and row and run and run not in self.state["expanded"] and \
                    row.get("state") in ("done", "r0"):
                recs = read_jsonl(os.path.join(self.L.abs(job["run_dir"]), "log.jsonl"))
                vb = [r for r in recs if r.get("event") == "verify" and r.get("phase") == "B"]
                if vb and job.get("train_phase", "B") == "B":
                    self.expand_for(job, vb[-1], c)
        self.save_state()
        return True

    def settle_training(self, job, c):
        """A training job launched before (a console log or attempts) with no row and no running record: an
        'end' event with a 'verify' of its phase in the last call counts as done; with a live train_stage1.py
        process on its run folder it is held; otherwise it is resumed once."""
        name, run_dir = job["name"], self.L.abs(job["run_dir"])
        live = [p for p in gpu_users(c["procs"], ("train_stage1.py",)) if job["run_dir"] in
                (p.get("cmdline") or "").replace("\\", "/")]
        if live:
            self.log(f"{name}: a train_stage1.py process on its run folder runs (PID {live[0]['pid']}); held")
            self.state["resume"].pop(name, None)
            self.state["running"]["gpu"] = self.state["running"]["gpu"] or {
                "name": name, "job": job, "slot": "gpu", "start": iso(c["now"]), "attempt": 1, "argv": job["argv"],
                "env": {}, "log": os.path.join(self.L.logs, f"{name}.log"), "log_pos": 0, "note": "adopted by command line",
                "pid": live[0]["pid"], "create_time": live[0].get("create_time"), "jsonl_count": None}
            self.adopt("gpu", self.state["running"]["gpu"], c)
            return
        seg = last_call(read_jsonl(os.path.join(run_dir, "log.jsonl")))
        phase = job.get("train_phase")
        ended = any(r.get("event") == "end" for r in seg) and \
            any(r.get("event") == "verify" and (phase is None or r.get("phase") == phase) for r in seg)
        rec = {"name": name, "job": job, "start": "", "argv": job["argv"], "env": {}, "note": ""}
        if ended:
            self.row(rec, "done", "", c, note="finished while the runner kept no state (end and verify events)")
            return
        att = self.state["attempts"].setdefault(name, [])
        if len(att) > int(job.get("max_resumes", 1)):
            self.write_r0(f"{name}: no end event after its resume")
            self.row(rec, "r0", "", c, note="no end event after its resume: R0 written")
            return
        att.append({"start": "", "end": iso(c["now"]), "exit": None, "timed_out": False, "argv": job["argv"]})
        cks = checkpoints(run_dir)
        ck = QJ.tree_rel(cks[0], self.tree) if cks else None
        self.state["resume"][name] = {"argv": with_resume(job["argv"], ck), "reason": "restart: no end event",
                                      "time": iso(c["now"]), "job": {k: x for k, x in job.items() if k != "_file"}}
        self.log(f"{name}: run folder without an end event and no live process; resume queued from {ck}")

    def run_forever(self, interval=PASS_S):
        if not self.startup():
            return 2
        while True:
            t0 = time.time()
            try:
                self.pass_once()
            except Exception:  # noqa: BLE001
                self.log("pass failed:\n" + traceback.format_exc())
                try:
                    append_text(self.L.status, f"\nPass {self.pass_no} failed at {iso(self.now())}: see "
                                               f"cache/runs/queue/runner.log\n")
                except OSError:
                    pass
            if self.now() >= self.until and not any(self.state["running"].values()):
                break
            while time.time() - t0 < interval:
                time.sleep(min(1.0, max(0.0, interval - (time.time() - t0))))
        self.log("runner ended (past --until, no job running)")
        try:
            with open(self.L.runner_pid, encoding="utf-8") as f:
                if f.read().split()[0] == str(self.pid):
                    remove(self.L.runner_pid)
        except (OSError, IndexError):
            pass
        return 0

    # ------------------------------------------------------------ STATUS.md
    def status_text(self, c, dry):
        st, now = self.state, c["now"]
        L = []
        L.append("# GPU queue status\n")
        mode = "dry run (no runner running from this command; nothing launched)" if dry else \
            f"runner PID {self.pid}, pass {self.pass_no}"
        L.append(f"Written {iso(now)} by scripts/gpu_queue.py ({mode}). Deadline {iso(self.deadline)}; launches "
                 f"until {iso(self.until)}.\n")
        L.append("## Trigger phase_b_done\n")
        L.append(f"- {c.get('trigger', 'not evaluated')}")
        pc = c.get("pid_check") or st["phase_b"].get("pid_check")
        if pc:
            L.append(f"- PID check ({pc['time']}): PID {pc['pid']} {'is' if pc['matches'] else 'is NOT'} python "
                     f"running train_stage1.py --phase B: `{pc.get('cmdline') or 'no such process'}`")
        L.append("")
        L.append("## GPU\n")
        L.append(f"- {'free' if c['gpu_free'] else 'busy'}" + ("" if c["gpu_free"] else ": " + "; ".join(c["gpu_why"])))
        L.append(f"- nvidia-smi: {c['gpu_name'] or 'unknown'}, memory.used "
                 f"{'unreadable' if c['gpu_mem'] is None else format(c['gpu_mem'], ',') + ' MiB'}")
        L.append("")
        L.append("## Flags\n")
        L.append("- " + ", ".join(f"{k} {'present' if v else 'absent'}" for k, v in c["flags"].items()))
        for k in ("R0", "SLOW", "HOLD"):
            if c["flags"][k]:
                text = tail_text(getattr(self.L, k.lower()), 0, 2000).strip().splitlines()
                L.append(f"- {k}: " + (text[-1] if text else "(empty)"))
        L.append("")
        L.append("## Running\n")
        run = [r for r in st["running"].values() if r]
        if not run:
            L.append("- none")
        for r in run:
            L.append(f"- {r['name']} ({r['slot']}, PID {r.get('pid')}, since {r['start']}, attempt {r.get('attempt')}): "
                     f"`{_short(' '.join(r['argv']), 200)}`")
        for n, rs in sorted(st["resume"].items()):
            L.append(f"- resume pending: {n} after {rs['reason']}: `{_short(' '.join(rs['argv']), 200)}`")
        L.append("")
        L.append("## Next\n")
        L.append(f"- GPU slot: {c.get('gpu_next') or 'no eligible GPU job'}"
                 + ("" if c["gpu_free"] or not c.get("gpu_next") else " (waits for the GPU)"))
        L.append(f"- CPU slot: {c.get('cpu_next') or 'no eligible CPU job'}")
        if dry:
            L.append(f"- would launch now: {', '.join(c['launched']) or 'nothing'}")
        else:
            L.append(f"- launched this pass: {', '.join(c['launched']) or 'nothing'}")
        L.append("")
        pend = sorted(((c["jobs"][n], why) for n, why in c["reasons"].items()),
                      key=lambda x: (x[0]["cls"], x[0]["priority"], x[0]["name"]))
        L.append(f"## Pending ({len(pend)})\n")
        if pend:
            L.append("| job | cls | priority | gpu | est h | waits on |")
            L.append("|---|---|---|---|---|---|")
            for j, why in pend:
                L.append(f"| {j['name']} | {j['cls']} | {j['priority']} | {'yes' if j['gpu'] else 'no'} | "
                         f"{j['est_hours']:g} | {'; '.join(why) or 'eligible'} |")
        else:
            L.append("- none")
        if c["bad_jobs"]:
            L.append("")
            for f, e in sorted(c["bad_jobs"].items()):
                L.append(f"- refused job file {f}: {e}")
        L.append("")
        L.append("## Training speed (seconds per step, last 500 steps, dev and checkpoint intervals left out)\n")
        if not c.get("speed"):
            L.append("- no training run in view")
        else:
            L.append("| run | phase | step | s/step | baseline | ratio | ETA |")
            L.append("|---|---|---|---|---|---|---|")
            for r in c["speed"]:
                f = lambda x, d=4: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
                L.append(f"| {r['run']} | {r['phase']} | {r['step'] if r['step'] is not None else 'n/a'} | "
                         f"{f(r['rate'])} | {f(r['base'])} | {f(r['ratio'], 3)} | "
                         f"{iso(r['eta']) if r['eta'] else 'n/a'} |")
        L.append("")
        L.append("## Last ten exits\n")
        rows = list(c["rows"].values())
        rows = sorted(rows, key=lambda r: r.get("end") or "")[-10:]
        if not rows:
            L.append("- none")
        else:
            L.append("| job | state | exit | start | end | note |")
            L.append("|---|---|---|---|---|---|")
            for r in reversed(rows):
                L.append(f"| {r['name']} | {r.get('state')} | {r.get('exit_code')} | {r.get('start')} | {r.get('end')} | "
                         f"{(r.get('note') or '').replace('|', '/')} |")
        L.append("")
        L.append("## Suite (scripts/run_suite.py, run by the runner)\n")
        s = st["suite"]
        if s.get("last_start"):
            L.append(f"- last start {s['last_start']}" + (" (running)" if s.get("running") else
                                                          f", exit {s.get('last_exit')}: {s.get('last_line', '')}"))
            if not s.get("running") and s.get("last_exit") not in (0, None):
                L.append(f"- FAILURE: run_suite exit {s.get('last_exit')}; log {QJ.tree_rel(s.get('log', ''), self.tree)}")
        else:
            L.append("- not run by the runner")
        if s.get("held"):
            L.append(f"- {s['held']}")
        if int((s.get("fail_streak") or {}).get("count", 0)) >= SUITE_STREAK:
            L.append(f"- REPEATED FAILURE: {streak_line(s['fail_streak'])}")
        if c.get("suite_needed"):
            L.append(f"- jobs held only by a missing green marker: {', '.join(sorted(c['suite_needed'])[:8])}")
        L.append("")
        L.append("## Locks and steps\n")
        L.append(f"- fresh locks (under 3 h): {', '.join(sorted(c['fresh_locks'])) or 'none'}; stale, ignored: "
                 + (", ".join(f"{k} ({v} h)" for k, v in sorted(c["stale_locks"].items())) or "none"))
        for k, v in sorted(st["steps"].items()):
            L.append(f"- step {k}: {v}")
        for n in c["notes"]:
            L.append(f"- {n}")
        L.append("")
        return "\n".join(L)


def _create_time(pid):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001
        return None


def verify_checkpoints(paths, out=None):
    """(CPU child) Print {'verified': the first path that checkpoint.load reads without error, 'errors'}."""
    out = sys.stdout if out is None else out
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if TREE not in sys.path:
        sys.path.insert(0, TREE)
    from mdfm.train import checkpoint as CK
    errors = {}
    for p in paths:
        try:
            CK.load(os.path.join(TREE, p) if not os.path.isabs(p) else p, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            errors[p] = repr(e)[:400]
            continue
        print(json.dumps({"verified": p, "errors": errors}), file=out, flush=True)
        return 0
    print(json.dumps({"verified": None, "errors": errors}), file=out, flush=True)
    return 1


def dry_report(c):
    """The lines a dry run prints before STATUS.md."""
    out = [f"gpu_queue dry run at {iso(c['now'])}",
           f"phase_b_done: {c.get('trigger')}",
           f"GPU free: {'yes' if c['gpu_free'] else 'no'}" + ("" if c["gpu_free"] else " (" + "; ".join(c["gpu_why"]) + ")"),
           f"eligible GPU job: {c['eligible_gpu'][0]['name'] if c['eligible_gpu'] else 'none'}",
           f"eligible CPU job: {c['eligible_cpu'][0]['name'] if c['eligible_cpu'] else 'none'}",
           f"would launch: {', '.join(c['launched']) or 'nothing'}",
           f"pending jobs: {len(c['reasons'])}; held only by a missing green marker: {len(c.get('suite_needed') or [])}"]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--until", default=DEADLINE, help="no launch from this local time on (default the deadline)")
    p.add_argument("--interval", type=float, default=PASS_S)
    p.add_argument("--once", action="store_true", help="one real pass")
    p.add_argument("--dry-run", action="store_true", help="one pass that launches and writes nothing")
    p.add_argument("--write-status", action="store_true", help="with --dry-run: also write STATUS.md")
    p.add_argument("--verify-checkpoints", nargs="+", default=None)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.verify_checkpoints:
        return verify_checkpoints(a.verify_checkpoints)
    r = Runner(until=a.until, deadline=a.until)
    if a.dry_run:
        c = r.pass_once(dry=True)
        print(dry_report(c))
        print()
        print(c["status"])
        if a.write_status:
            write_text(r.L.status, c["status"])
        return 0
    if a.once:
        if not r.startup():
            return 2
        r.pass_once()
        return 0
    return r.run_forever(a.interval)


if __name__ == "__main__":
    sys.exit(main())
