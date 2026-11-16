"""B4.6 and B4.13b: the window store of the real stage-1 sources, its read-back and its report.

Builds the band copies of every admitted recording of the six pilot sources (mdfm/data/window_store.py:
IMU and force low-passed at 40 Hz zero-phase and decimated on native timestamps, EMG as 20 ms
log-envelopes, pose as recorded, float16 memmaps; the raw root track of mdfm.data.mocap_raw on the pose3d
streams of gait120 and stroke_gait_vancriekinge, float32) into the store root (MDFM_WINDOW_STORE, else
cache/window_store of this tree, gitignored; <store>/store/windows_v1 once the owner creates <store>,
spec question 12). Then reads every shard back (WindowStore.check: no lockbox, EVAL, EXCLUDED_V1 or CODY
person; blocks tile their shards; masked entries 0; root files tile the pose3d shards), measures the band on
analytic signals with the same band_copy, and checks that the gates-only store paths are refused. With
--verify-rebuild DIR, builds a second time into DIR and compares the person-list and file SHA-256 of every
source built. --sources rebuilds some sources only; the SHA-256 of every file of the store before and after
the build are compared (store_diff.json), so a partial rebuild shows which files it changed. The root check
compares, on --root-check recordings per root source, the stored pose x scale plus the stored root with the
raw joints of mocap_raw.raw_joints15.

Writes reports/window_store/: summary.md, sources.csv, hashes.json, check.json, band_checks.json,
rebuild.json (with --verify-rebuild), store_diff.json, root_check.json, _provenance.json. Records are read
through mdfm.data.sources and mdfm.data.mocap_raw with the reader flags at False (read-only on MoveAtlas).
Run: <conda>/envs/mdfm/python.exe scripts/build_window_store.py --verify-rebuild <dir>
B4.13b: ... --sources gait120,stroke_gait_vancriekinge --verify-rebuild <dir>
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
               help="MoveAtlas working tree (read-only)")
p.add_argument("--store", default=None, help="store root (default: MDFM_WINDOW_STORE, else cache/window_store)")
p.add_argument("--report", default=None, help="report folder (default reports/window_store, with --stage2 "
                                               "reports/window_store_s2)")
p.add_argument("--sources", default=None, help="comma-separated pilot sources (default: all six)")
p.add_argument("--limit", type=int, default=None, help="recordings per source (a quick run; not the full store)")
p.add_argument("--verify-rebuild", default=None, help="build a second time into this root and compare the hashes")
p.add_argument("--skip-build", action="store_true", help="report on the store as it is")
p.add_argument("--root-check", type=int, default=20,
               help="recordings per root-track source for the world-joint check (0: none)")
p.add_argument("--stage2", choices=["s2a"], default=None,
               help="build the stage-2 store of run S2a into the S2 root (MDFM_WINDOW_STORE_S2, else "
                    "cache/window_store_s2) and report under reports/window_store_s2 (section Stage 2 below)")
p.add_argument("--permutations", type=int, default=None, help="stage 2: group-permuted repeats of the mounting probe")
ARGS = p.parse_args()
ARGS.moveatlas = ARGS.moveatlas.replace("\\", "/").rstrip("/")
os.environ["MOVEATLAS_ROOT"] = ARGS.moveatlas
sys.stdout.reconfigure(encoding="utf-8")
if ARGS.report is None:
    ARGS.report = os.path.join(TREE, "reports", "window_store_s2" if ARGS.stage2 else "window_store")

import numpy as np  # noqa: E402

import mdfm  # noqa: E402,F401
from mdfm import paths  # noqa: E402
from mdfm.data import canonical as C  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.data import windows as W  # noqa: E402

paths.assert_not_moveatlas(ARGS.report)
os.makedirs(ARGS.report, exist_ok=True)
if ARGS.stage2:
    SPECS = S.STAGE2_S2A
    ROOT = WS.assert_stage2_root(ARGS.store or WS.store_root(kind="s2"), "RELEASE")
    NAMES = [s.strip() for s in ARGS.sources.split(",")] if ARGS.sources else list(SPECS)
else:
    ROOT = WS.assert_store_root(ARGS.store or WS.store_root())
    NAMES = [s.strip() for s in ARGS.sources.split(",")] if ARGS.sources else list(S.PILOT)


def rel(path):
    a = os.path.abspath(path).replace("\\", "/")
    t = TREE.replace("\\", "/")
    return a[len(t) + 1:] if a.lower().startswith(t.lower() + "/") else a


def moveatlas_head():
    r = subprocess.run(["git", "-c", f"safe.directory={ARGS.moveatlas}", "-C", ARGS.moveatlas, "log", "-1",
                        "--format=%h %ad %s", "--date=short"], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or f"(git failed: {r.stderr.strip()[:200]})"


def dump(name, obj):
    with open(os.path.join(ARGS.report, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=WS._json_default)
        f.write("\n")


def build_all(root, tag, attempts=3):
    """build_source for each source into root; {source: (summary, seconds of the successful attempt)}. A source
    whose build raises OSError (a transient 'Invalid argument' when a MoveAtlas container is reopened on the
    exFAT disk) is built again from scratch, at most `attempts` times; build_source replaces a source whole. A
    refusal (canonical.Refused, a PermissionError) is never retried."""
    WS.assert_store_root(root)
    f = S.folds()
    out = {}
    for n in NAMES:
        folds = dict(zip(f.loc[f["source"] == n, "person_id"], f.loc[f["source"] == n, "fold"]))
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                s = WS.build_source(n, out=root, limit=ARGS.limit, folds=folds, stats=collections.Counter())
                break
            except OSError as e:
                if isinstance(e, PermissionError) or attempt == attempts:      # canonical.Refused never retries
                    raise
                print(f"[{tag}] {n}: attempt {attempt} failed after {time.time() - t0:.0f} s: {e}", flush=True)
        out[n] = (s, round(time.time() - t0, 1))
        print(f"[{tag}] {n}: {s['recordings']} recordings, {s['records']} records in {out[n][1]} s", flush=True)
    WS.write_store_json(root)
    return out


# ---------------------------------------------------------------- band on analytic signals
def band_checks():
    from scipy.signal import sosfreqz
    rng = np.random.default_rng(20260928)
    res = {"plans": {}}
    for modality, fps in (("imu", 100.0), ("imu", 200.0), ("imu", 500.0), ("force", 100.0), ("force", 1000.0),
                          ("emg", 1000.0), ("emg", 2000.0), ("pose3d", 100.0)):
        pl = WS.band_plan(modality, fps)
        row = {"kind": pl.kind, "k": pl.k, "fps_out": pl.fps_out, "padlen": pl.padlen, "rms_n": pl.rms_n}
        if pl.sos is not None:
            freqs = [10.0, 20.0, 40.0, 60.0, 450.0] if modality == "emg" else [10.0, 40.0, 45.0, 60.0]
            freqs = [f for f in freqs if f < fps / 2]                      # below Nyquist only
            _, h = sosfreqz(pl.sos, worN=np.array(freqs), fs=fps)
            row["zero_phase_gain"] = {f"{f:g} Hz": round(float(abs(x) ** 2), 6) for f, x in zip(freqs, h)}
        res["plans"][f"{modality}@{fps:g}"] = row
    # 10 Hz and 60 Hz sinusoids at 500 Hz through band_copy
    fps, T = 500.0, 5000
    t = np.arange(T) / fps
    x = np.zeros((T, 2, 9), np.float32)
    x[:, 0, 0] = np.sin(2 * np.pi * 10 * t)
    x[:, 1, 0] = np.sin(2 * np.pi * 60 * t + 0.3)
    st = W.Stream("imu", "input", fps, t.astype(np.float32), x, np.ones((2, 9), bool), np.ones((T, 2), bool), {})
    b = WS.band_copy(st)
    inner = (b.t >= 0.5) & (b.t <= 9.5)
    tt = b.t[inner].astype(np.float64)
    A = np.stack([np.sin(2 * np.pi * 10 * tt), np.cos(2 * np.pi * 10 * tt)], axis=1)
    coef, *_ = np.linalg.lstsq(A, b.values[inner, 0, 0], rcond=None)
    res["sine_500hz"] = {"k": int(round(fps / b.fps)), "fps_out": b.fps, "amplitude_10hz": round(float(np.hypot(*coef)), 6),
                         "rms_ratio_60hz": round(float(np.sqrt(np.mean(b.values[inner, 1, 0].astype(np.float64) ** 2))
                                                       / np.sqrt(0.5)), 6)}
    # EMG burst of 50 ms at 1.0 s on 5 uV noise
    res["emg_burst"] = {}
    for fps in (1000.0, 2000.0):
        T = int(3 * fps)
        t = np.arange(T) / fps
        e = 0.005 * rng.normal(size=(T, 1, 1))
        burst = (t >= 1.0) & (t < 1.05)
        e[burst, 0, 0] += 0.5 * rng.normal(size=burst.sum())
        st = W.Stream("emg", "target", fps, t.astype(np.float32), e.astype(np.float32), np.ones((1, 1), bool),
                      np.ones((T, 1), bool), {"signal_kind": "raw"})
        b = WS.band_copy(st)
        res["emg_burst"][f"{fps:g}"] = {"burst_s": [1.0, 1.05], "peak_s": round(float(b.t[np.argmax(b.values[:, 0, 0])]), 4),
                                        "peak_minus_median_log_mv": round(float(b.values[:, 0, 0].max()
                                                                                - np.median(b.values[:, 0, 0])), 3)}
    # EMG zero-fill: 0.3 s of exact zeros before a 2.7 s raw trial at 1000 Hz (the stroke C3D padding)
    fps, T = 1000.0, 3000
    e = 0.05 * rng.normal(size=(T, 1, 1))
    e[:300] = 0.0
    st = W.Stream("emg", "target", fps, (np.arange(T) / fps).astype(np.float32), e.astype(np.float32),
                  np.ones((1, 1), bool), np.ones((T, 1), bool), {"signal_kind": "raw"})
    b, info = WS.band_copy(st, return_info=True)
    pad = info.idx < 300
    res["emg_zero_fill"] = {"native_zero_samples": 300, "masked_as_zero_fill": info.zero_fill,
                            "kept_valid_in_fill": int(b.valid[pad].sum()), "kept_valid_after": int(b.valid[~pad].sum()),
                            "kept_after": int((~pad).sum()),
                            "lowest_minus_median_log_mv": round(float(b.values[~pad, 0, 0].min()
                                                                      - np.median(b.values[~pad, 0, 0])), 3)}
    # a NaN run: the outputs outside the run do not depend on it
    fps, T = 1000.0, 3000
    x = (200 + 50 * rng.normal(size=(T, 1, 9))).astype(np.float32)
    valid = np.ones((T, 1), bool)
    valid[1200:1500] = False
    y = x.copy()
    x[1200:1500] = 0
    y[1200:1500] = 1e4
    outs = [WS.band_copy(W.Stream("force", "target", fps, (np.arange(T) / fps).astype(np.float32), v,
                                  np.ones((1, 9), bool), valid, {}), return_info=True) for v in (x, y)]
    idx = outs[0][1].idx
    gap = (idx >= 1200) & (idx < 1500)
    res["nan_run"] = {"max_abs_diff_outside": float(np.abs(outs[0][0].values - outs[1][0].values).max()),
                      "kept_valid_outside": int(outs[0][0].valid[~gap].sum()), "kept_outside": int((~gap).sum()),
                      "kept_valid_inside": int(outs[0][0].valid[gap].sum())}
    # store roots in the gates-only store
    res["refused_roots"] = {}
    for root in ("<store>/store/lockbox/windows_v1", paths.GATES_STORE + "/windows_v1",
                 os.path.join(paths.TREE, "reports", "window_store")):
        try:
            WS.assert_store_root(root)
            res["refused_roots"][rel(root)] = "accepted"
        except C.Refused:
            res["refused_roots"][rel(root)] = "refused"
    return res


def target_grid(records):
    """Per source and target stream (records of WindowStore): records whose kept times sit on the input grid of
    their recording (same output rate, offsets a whole number of input periods apart), records with the input's
    n, and records whose last kept sample lies at or past the span end t0 + duration."""
    inp = records[records["role"] == "input"].set_index(["source", "subject_ns", "key"])
    out = {}
    for (source, stream), g in records[records["role"] == "target"].groupby(["source", "stream"], sort=False):
        j = g.join(inp[["offset", "fps", "n"]], on=["source", "subject_ns", "key"], rsuffix="_in")
        ph = (j["offset"] - j["offset_in"]) * j["fps_in"]
        on = (np.abs(ph - np.round(ph)) < 1e-6) & (j["fps"] == j["fps_in"])
        last = j["offset"] + (j["n"] - 1) * j["k"] / j["fps_native"]
        out[f"{source}/{stream}"] = {"records": int(len(j)), "on_input_grid": int(on.sum()),
                                     "same_n_as_input": int((j["n"] == j["n_in"]).sum()),
                                     "last_at_or_past_span_end": int((last >= j["span_t0"] + j["span_s"] - 1e-9).sum())}
    return out


def sha_text(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def disk_state(root):
    """{source: {'files': {path relative to the source folder: SHA-256 of the bytes on disk}, 'person_sha256':
    of its source.json}} for every source folder of a store root (._ sidecars left out), and the SHA-256 of
    store.json under the key ''."""
    out = {}
    if not os.path.isdir(root):
        return out
    for name in WS._source_dirs(root):
        d = os.path.join(root, name)
        files = {}
        for dp, _, fn in os.walk(d):
            for n in fn:
                if not n.startswith("._"):
                    p_ = os.path.join(dp, n)
                    files[os.path.relpath(p_, d).replace("\\", "/")] = WS.sha256_file(p_)
        with open(os.path.join(d, "source.json"), encoding="utf-8") as f:
            out[name] = {"files": files, "person_sha256": json.load(f)["person_sha256"]}
    sj = os.path.join(root, "store.json")
    out[""] = WS.sha256_file(sj) if os.path.isfile(sj) else None
    return out


def store_diff(before, after):
    """Per source: rebuilt in this run or not, the files unchanged, changed, added and removed between the two
    disk states, and whether the person-list hashes are equal."""
    doc = {"sources": {}, "store_json_changed": before.get("") != after.get("")}
    for name in sorted((set(before) | set(after)) - {""}):
        a = (before.get(name) or {}).get("files", {})
        b = (after.get(name) or {}).get("files", {})
        doc["sources"][name] = {
            "rebuilt": name in NAMES, "in_before": name in before, "in_after": name in after,
            "unchanged": sorted(k for k in a if k in b and a[k] == b[k]),
            "changed": sorted(k for k in a if k in b and a[k] != b[k]),
            "added": sorted(k for k in b if k not in a), "removed": sorted(k for k in a if k not in b),
            "person_sha256_equal": (before.get(name) or {}).get("person_sha256") == (after.get(name) or {}).get("person_sha256")}
    return doc


def root_world_check(root, n_per_source):
    """On n recordings per root source (evenly spaced in records order): the stored root equals
    mocap_raw.root_for_paired and the stored pose (float16) x scale plus the stored root gives the raw joints of
    mocap_raw.raw_joints15 (m); error over the joint samples valid in the pose, the root and the raw joints."""
    from mdfm.data import mocap_raw as M
    store = WS.WindowStore(root)
    out = {}
    r = store.records
    for source in M.SOURCES:
        rows = r[(r["source"] == source) & (r["stream"] == "pose3d")]
        if rows.empty or n_per_source <= 0:
            continue
        pick = rows.iloc[np.unique(np.linspace(0, len(rows) - 1, min(n_per_source, len(rows))).round().astype(int))]
        errs, used, total, equal, root_samples = [], 0, 0, 0, 0
        for i, row in pick.iterrows():
            pr = next(S.paired(source, recordings={(row["subject_ns"], row["key"])}))
            s = store[i]
            root_, valid, rt = M.root_for_paired(pr)
            ok = valid & np.isfinite(root_).all(axis=1)
            got_ok = s.meta.get("root_valid", np.zeros(len(s.t), bool))
            got = s.meta.get("root", np.zeros((len(s.t), 3), np.float32))
            equal += bool(len(s.t) == len(ok) and np.array_equal(got_ok, ok)
                          and np.array_equal(got[ok], root_[ok].astype(np.float32)))
            if source == M.GAIT120:
                pos = {int(f): j for j, f in enumerate(rt.frames)}
                sel = np.array([pos[int(f)] for f in pr.pairing["pose_frames"]])
            else:
                sel = np.arange(len(pr.t["pose3d"]))
            raw = M.raw_joints15(source, pr.subject_ns, pr.key)[sel][:len(s.t)]
            world = s.values.astype(np.float64) * float(s.meta["scale"]) + got[:, None, :]
            use = s.valid & got_ok[:, None] & np.isfinite(raw).all(axis=-1)
            errs.append(np.linalg.norm(world - raw, axis=-1)[use])
            used += int(use.sum())
            total += int(use.size)
            root_samples += len(ok)
        e = np.concatenate(errs) if errs else np.zeros(0)
        out[source] = {"recordings": int(len(pick)), "root_equal_to_root_for_paired": int(equal),
                       "joint_samples": total, "joint_samples_compared": used, "root_samples": root_samples,
                       "err_max_m": float(e.max()) if e.size else None,
                       "err_p99_m": float(np.percentile(e, 99)) if e.size else None,
                       "err_median_m": float(np.median(e)) if e.size else None,
                       "keys": [f"{a} {b}" for a, b in zip(pick["subject_ns"], pick["key"])]}
        print(f"root check {source}: {out[source]['root_equal_to_root_for_paired']} of {len(pick)} equal, "
              f"max world-joint error {out[source]['err_max_m']}", flush=True)
    store.close()
    return out


# ---------------------------------------------------------------- Stage 2 (task S2.6)
# --stage2 s2a: the store of run S2a (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, sections 4.4 and 4.5, the S2.6 row of
# section 7). The five units of mdfm.data.sources.STAGE2_S2A are built at tier RELEASE under the name tables
# sources.s2a_names() (version-2 vocabulary) into the S2 root only; the stage-1 root is hashed file by file before and
# after. The process runs at BelowNormal priority (psutil) so that the GPU training keeps its CPU, and the training
# speed rows of reports/gpu_queue/STATUS.md (read only) are logged at the start, after the build and at the end.
# Then: the read-back (WindowStore.check: no lockbox, EVAL, EXCLUDED_V1, CODY or INTERNAL person; every whole
# window validated under the version-2 task count), the persons of the store against those lists in the index, the
# fold draws against reports/stage2_sources/folds_proposed.csv, the tilt check of the turned sites (on the native
# records through the build's stats, on the stored band copies here), the mounting probe on clinical_gait_multipatho
# (mdfm.data.mounting: before and after canonicalization, and group-permuted persons as chance; no threshold; with
# the wider readings of mounting.WIDER_ARMS beside it), the second build with --verify-rebuild, and the pending
# sources (sources.STAGE2_PENDING). Writes
# reports/window_store_s2/: summary.md, sources.csv, hashes.json, check.json, folds.json, tilt.json,
# mounting_probe.json, names.json, stage1_root.json, speed.json, rebuild.json (with --verify-rebuild), run.log,
# _provenance.json.
SLOW_FLAG = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
STATUS_MD = os.path.join(TREE, "reports", "gpu_queue", "STATUS.md")
FOLDS_PROPOSED = os.path.join(TREE, "reports", "stage2_sources", "folds_proposed.csv")
FOLDS_SUMMARY = os.path.join(TREE, "reports", "stage2_sources", "summary.json")
PROBE_SOURCE = "clinical_gait_multipatho"
LOG_LINES = []


def log2(*a):
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} " + " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG_LINES.append(line)


def below_normal():
    """BelowNormal priority for this process (Windows); what was done."""
    try:
        import psutil
        proc = psutil.Process()
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return "BelowNormal" if proc.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS else f"asked, reads {proc.nice()}"
        proc.nice(10)
        return "nice 10"
    except Exception as e:  # noqa: BLE001
        return f"unchanged ({type(e).__name__}: {e})"


def training_speed():
    """The 'Written' line and the rows of the 'Training speed' table of reports/gpu_queue/STATUS.md (read only),
    and whether cache/runs/queue/SLOW exists."""
    out = {"at": dt.datetime.now().isoformat(timespec="seconds"), "slow_flag": os.path.exists(SLOW_FLAG),
           "written": None, "rows": []}
    try:
        with open(STATUS_MD, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as e:
        out["error"] = str(e)
        return out
    out["written"] = next((x for x in lines if x.startswith("Written ")), None)
    if "## Training speed" in "\n".join(lines):
        i = lines.index(next(x for x in lines if x.startswith("## Training speed")))
        for x in lines[i + 1:]:
            if x.startswith("## "):
                break
            if x.startswith("| ") and not x.startswith("| run ") and not x.startswith("|---"):
                out["rows"].append(x)
    if os.path.exists(SLOW_FLAG):
        try:
            with open(SLOW_FLAG, encoding="utf-8") as f:
                out["slow_text"] = f.read().strip()[:300]
        except OSError:
            pass
    return out


def listing(root):
    """{relative path: [size, SHA-256]} of every file under a root (AppleDouble sidecars left out) and the SHA-256 of
    the sorted listing."""
    files = {}
    if os.path.isdir(root):
        for d, _, fs in os.walk(root):
            for n in fs:
                if n.startswith("._"):
                    continue
                q = os.path.join(d, n)
                files[os.path.relpath(q, root).replace("\\", "/")] = [os.path.getsize(q), WS.sha256_file(q)]
    text = "\n".join(f"{k}\t{v[0]}\t{v[1]}" for k, v in sorted(files.items()))
    return {"root": rel(root), "files": len(files), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "entries": files}


def build_s2(root, tag, names, index, folds_frame, attempts=3):
    """build_source of each unit of SPECS into a stage-2 root at tier RELEASE under `names`; {source: (summary,
    seconds, stats)}; retried on OSError as build_all."""
    out = {}
    for n in NAMES:
        folds = dict(zip(folds_frame.loc[folds_frame["source"] == n, "person_id"],
                         folds_frame.loc[folds_frame["source"] == n, "fold"]))
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            st = collections.Counter()
            try:
                s = WS.build_source(n, out=root, limit=ARGS.limit, folds=folds, stats=st, specs=SPECS, tier="RELEASE",
                                    names=names, index=index)
                break
            except OSError as e:
                if isinstance(e, PermissionError) or attempt == attempts:
                    raise
                log2(f"[{tag}] {n}: attempt {attempt} failed after {time.time() - t0:.0f} s: {e}")
        out[n] = (s, round(time.time() - t0, 1), st)
        log2(f"[{tag}] {n}: {s['recordings']} recordings, {s['records']} records in {out[n][1]} s")
    WS.write_store_json(root)
    return out


def forbidden_persons(store, index):
    """Persons of the store with an index row that is lockbox, EVAL, EXCLUDED_V1, CODY or INTERNAL, per list."""
    persons = set(store.records["person_id"])
    lists = {"lockbox": set(index.loc[index["lockbox"].astype(bool), "person_id"]) | set(WS.lockbox_persons()),
             "EVAL": set(index.loc[index["tier"] == "EVAL", "person_id"]),
             "EXCLUDED_V1": set(index.loc[index["tier"] == "EXCLUDED_V1", "person_id"]),
             "CODY": set(index.loc[index["cody"].astype(bool), "person_id"]),
             "INTERNAL": set(index.loc[index["tier"] == "INTERNAL", "person_id"])}
    return {"persons": len(persons), **{k: sorted(persons & v) for k, v in lists.items()}}


def fold_doc(folds_frame, store_hashes):
    import pandas as pd
    with open(FOLDS_SUMMARY, encoding="utf-8") as f:
        proposed = json.load(f)["folds"]
    prop = pd.read_csv(FOLDS_PROPOSED, dtype=str, keep_default_na=False, encoding="utf-8")
    doc = {"folds_proposed_sha256": sha_text(FOLDS_PROPOSED), "units": {}}
    for n, spec in SPECS.items():
        f = folds_frame[folds_frame["source"] == n]
        draw = {fold: S.person_hash(g["person_id"]) for fold, g in f.groupby("fold")}
        u = {"unit": spec.unit, "fold": spec.fold, "share": spec.fold_share, "seed_key": spec.fold_key or n,
             "persons": int(len(f)), "per_fold": {k: int(v) for k, v in f["fold"].value_counts().sort_index().items()},
             "draw_sha256": draw, "store_sha256": store_hashes.get(n, {})}
        key = spec.fold_key
        if key in proposed:
            want = proposed[key]
            mine = dict(zip(f["person_id"], f["fold"]))
            theirs = dict(zip(prop.loc[prop["source"] == n, "person_id"], prop.loc[prop["source"] == n, "fold"]))
            u["proposed"] = {"sha256_dev": want["sha256_dev"], "sha256_train": want["sha256_train"],
                             "draw_dev_equal": draw.get("dev") == want["sha256_dev"],
                             "draw_train_equal": draw.get("train") == want["sha256_train"],
                             "per_person_equal": mine == theirs,
                             "store_dev_equal": store_hashes.get(n, {}).get("dev") == want["sha256_dev"]}
            stored_dev = set(f.loc[f["fold"] == "dev", "person_id"])
            u["proposed"]["dev_persons_without_stored_recording"] = sorted(stored_dev - STORED_PERSONS.get(n, set()))
        doc["units"][n] = u
    return doc


STORED_PERSONS = {}


def tilt_doc(recs_by_source, built):
    """Per tilted source: the build's native check (stats tilt_*) and, per site, the stored mean against the up axis
    and the turn angle (quantiles over the records)."""
    doc = {}
    for src, recs in recs_by_source.items():
        st = built[src][2] if src in built else {}
        d = {"native": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in sorted(st.items())
                        if k.startswith("tilt_")}, "sites": {}}
        by = collections.defaultdict(list)
        for r in recs:
            for s in r["per_site"]:
                by[s["site"]].append(s)
        for site, items in by.items():
            turned = [s for s in items if s["turned"] and "angle_stored_deg" in s]
            a = np.array([s["angle_stored_deg"] for s in turned])
            t = np.array([s["turn_deg"] for s in turned])
            q = (lambda v: [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] if len(v) else None)
            d["sites"][site] = {"records": len(items), "turned": len(turned),
                                "stored_angle_deg_max": round(float(a.max()), 4) if len(a) else None,
                                "stored_angle_deg_p10_p50_p90": q(a), "turn_deg_p10_p50_p90": q(t),
                                "turn_deg_max": round(float(t.max()), 4) if len(t) else None}
        doc[src] = d
    return doc


def probe_doc(recs, index):
    from mdfm.data import mounting as MT
    g = index[(index["source"] == PROBE_SOURCE) & ~index["lockbox"].astype(bool) & (index["tier"] == "RELEASE")]
    groups = {}
    for pid, grp in zip(g["person_id"], g["group"]):
        groups.setdefault(pid, set()).add(grp)
    bad = {p: sorted(v) for p, v in groups.items() if len(v) != 1}
    if bad:
        raise ValueError(f"persons of several groups: {list(bad.items())[:5]}")
    site_names = list(L_.STAGE2_FAMILY_RULES["multipatho4"]["sites"])
    persons = np.array([r["person_id"] for r in recs])
    y = np.array([next(iter(groups[p])) for p in persons])
    perms = MT.PERMUTATIONS if ARGS.permutations is None else ARGS.permutations
    doc = {"source": PROBE_SOURCE, "sites": site_names, "folds": MT.FOLDS, "permutations": perms, "seed": MT.SEED,
           "C": MT.C_REG, "class_weight": "balanced",
           "records_per_group": {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
           "persons_per_group": {k: len({p for p, gg in zip(persons, y) if gg == k}) for k in sorted(set(y))},
           "arms": {}}
    if len(set(persons.tolist())) < MT.FOLDS or len(set(y.tolist())) < 2:
        doc["not_run"] = f"{len(set(persons.tolist()))} persons and {len(set(y.tolist()))} groups for {MT.FOLDS} folds"
        log2("probe not run:", doc["not_run"])
        return doc
    for arm, before in (("before", True), ("after", False)):
        X, cols, undefined = MT.feature_matrix(recs, site_names, before)
        res = MT.probe(X, y, persons, permutations=perms)
        res["columns"] = cols
        res["undefined_angles_set_to_0"] = undefined
        doc["arms"][arm] = res
        log2(f"probe {arm}: balanced accuracy {res['balanced_accuracy']:.4f}, chance mean {res['chance']['mean']}")
    doc["wider_arms"] = {}
    for arm, (kind, before) in MT.WIDER_ARMS.items():
        X, cols, undefined = MT.feature_matrix(recs, site_names, before, kind)
        res = MT.probe(X, y, persons, permutations=perms)
        res["kind"], res["before"], res["columns"], res["undefined_set_to_0"] = kind, before, cols, undefined
        doc["wider_arms"][arm] = res
        log2(f"wider reading {arm}: balanced accuracy {res['balanced_accuracy']:.4f}, chance mean "
             f"{res['chance']['mean']}")
    return doc


def unit_rows(store, summaries, built, names):
    r = store.records
    tasks = names.tasks()
    rows = []
    for n, spec in SPECS.items():
        if n not in summaries:
            continue
        s = summaries[n]
        d = r[(r["source"] == n) & (r["role"] == "input")]
        main = spec.streams[0].kind
        st = s["stats"]
        rows.append({"unit": spec.unit, "layout": spec.layout, "recordings": s["recordings"], "records": s["records"],
                     "persons": ", ".join(f"{k} {v}" for k, v in s["persons"].items()),
                     "input_hours": round(s["streams"].get(main, {}).get("seconds", 0.0) / 3600, 3),
                     "sex_unknown_records": int((d["sex"] == 2).sum()),
                     "age_unknown_records": int((d["age_band"] == 8).sum()),
                     "tasks": ", ".join(f"{tasks[int(k)]} {int(v)}"
                                        for k, v in d["task"].value_counts().sort_index().items()),
                     "dropped": ", ".join(f"{k[8:]} {v}" for k, v in st.items() if k.startswith("dropped_")) or "0",
                     "records_short": st.get("records_short", 0),
                     "irregular_grid": st.get("recordings_irregular_grid", 0),
                     "build_s": built.get(n, (None, ""))[1]})
    return rows


def stage2_main():
    t_start = time.time()
    prio = below_normal()
    speed = {"start": training_speed()}
    head = moveatlas_head()
    names = S.s2a_names()
    log2(f"stage 2 s2a; priority {prio}; MoveAtlas {head}; root {rel(ROOT)}; name tables {names.sha256()}")
    log2("training speed at start:", speed["start"]["rows"], "SLOW flag", speed["start"]["slow_flag"])
    index = C.index()
    stage1_root = WS.store_root()
    s1_before = listing(stage1_root)
    log2(f"stage-1 root {s1_before['root']}: {s1_before['files']} files, listing {s1_before['sha256']}")
    folds_frame = S.folds(specs=SPECS)
    built = build_s2(ROOT, "build", names, index, folds_frame)
    speed["after_build"] = training_speed()
    log2("training speed after the build:", speed["after_build"]["rows"])
    store = WS.WindowStore(ROOT, tier="RELEASE", names=names)
    t0 = time.time()
    check = store.check(index=index)
    check["seconds"] = round(time.time() - t0, 1)
    n_tasks = len(names.tasks())
    for i in range(len(store.recordings)):
        W.validate(store.window(i), n_tasks=n_tasks)
    check["windows_validated"] = int(len(store.recordings))
    check["forbidden"] = forbidden_persons(store, index)
    check["target_grid"] = target_grid(store.records)
    log2("read-back:", {k: v for k, v in check.items() if k not in ("persons", "max_abs", "forbidden", "target_grid")})
    hashes = store.person_hashes()
    summaries = dict(store.sources)
    for n in summaries:
        STORED_PERSONS[n] = set(store.records.loc[store.records["source"] == n, "person_id"])
    folds = fold_doc(folds_frame, hashes)
    recs = {src: MT.store_records(store, src) for src, spec in SPECS.items() if spec.tilt and src in summaries}
    tilt = tilt_doc(recs, built)
    probe = probe_doc(recs[PROBE_SOURCE], index) if PROBE_SOURCE in recs else None
    units = unit_rows(store, summaries, built, names)
    with open(os.path.join(ROOT, "store.json"), encoding="utf-8") as f:
        store_json = json.load(f)
    marker = WS.read_marker(ROOT)
    store.close()
    rebuild = None
    if ARGS.verify_rebuild:
        other = WS.assert_stage2_root(ARGS.verify_rebuild, "RELEASE")
        second = build_s2(other, "rebuild", names, index, folds_frame)
        s2 = WS.WindowStore(other, tier="RELEASE", names=names)
        h2 = s2.person_hashes()
        s2.close()
        where = rel(other)                       # a scratch root outside the tree is not named in the report
        rebuild = {"root": where if not os.path.isabs(where) else "a scratch root outside this tree",
                   "sources_built": NAMES,
                   "person_hashes_equal": {n: h2.get(n) for n in NAMES} == {n: hashes.get(n) for n in NAMES},
                   "sources": {n: {"person_sha256_equal":
                                   summaries[n]["person_sha256"] == second[n][0]["person_sha256"],
                                   "files_sha256_equal": summaries[n]["files_sha256"] == second[n][0]["files_sha256"],
                                   "files": len(summaries[n]["files_sha256"]), "seconds": second[n][1]} for n in NAMES}}
        dump("rebuild.json", rebuild)
        log2("rebuild:", rebuild["person_hashes_equal"],
             {n: v["files_sha256_equal"] for n, v in rebuild["sources"].items()})
    s1_after = listing(stage1_root)
    s1 = {"before": {k: v for k, v in s1_before.items() if k != "entries"},
          "after": {k: v for k, v in s1_after.items() if k != "entries"},
          "equal": s1_before["entries"] == s1_after["entries"],
          "data_sha256_note": "the file listing covers every byte the data SHA-256 of a sampler reads"}
    speed["end"] = training_speed()
    log2("stage-1 root unchanged:", s1["equal"], "; training speed at the end:", speed["end"]["rows"])
    dump("stage1_root.json", s1)
    dump("speed.json", speed)
    dump("check.json", check)
    dump("folds.json", folds)
    dump("tilt.json", tilt)
    if probe is not None:
        dump("mounting_probe.json", probe)
    dump("names.json", {"sha256": names.sha256(), "tables": names.to_dict(), "tasks": list(names.tasks()),
                        "vocabulary_sha256": names.vocabulary_sha256()})
    dump("hashes.json", {"person_sha256": hashes, "files_sha256": {n: s["files_sha256"] for n, s in summaries.items()},
                         "store_json": store_json, "marker": marker})
    import csv as csv_
    with open(os.path.join(ARGS.report, "sources.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv_.DictWriter(f, fieldnames=list(units[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(units)
    code = {m: sha_text(os.path.join(TREE, "mdfm", "data", m))
            for m in ("window_store.py", "sources.py", "windows.py", "mounting.py", "canonical.py")}
    code["scripts/build_window_store.py"] = sha_text(os.path.abspath(__file__))
    from mdfm.data import mounting as MT_
    prov = {"script": "build_window_store.py --stage2 s2a", "moveatlas_root": ARGS.moveatlas, "moveatlas_head": head,
            "run_at": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "store_root": rel(ROOT), "tier": "RELEASE", "names_sha256": names.sha256(),
            "vocabulary": names.vocabulary_path, "vocabulary_sha256": names.vocabulary_sha256(), "limit": ARGS.limit,
            "sources": NAMES, "priority": prio, "mount_priors_sha256": MT_.priors_sha256(),
            "index_sha256": {rel(C.manifest_path()): sha_text(C.manifest_path()),
                             rel(C.index_path()): sha_text(C.index_path())},
            "code_sha256": code, "build_seconds": {n: v[1] for n, v in built.items()},
            "runtime_s": round(time.time() - t_start, 1)}
    dump("_provenance.json", prov)
    with open(os.path.join(ARGS.report, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(stage2_summary(prov, units, check, folds, tilt, probe, rebuild, s1, speed, summaries, store_json,
                               names))
    with open(os.path.join(ARGS.report, "run.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    print(f"done in {time.time() - t_start:.0f} s; report {rel(ARGS.report)}", flush=True)


def _q(v):
    return "" if v is None else ", ".join(f"{x:g}" for x in v)


def stage2_summary(prov, units, check, folds, tilt, probe, rebuild, s1, speed, summaries, store_json, names):
    L = ["# Stage-2 window store, run S2a (task S2.6)", "",
         "Generated by scripts/build_window_store.py --stage2 s2a; every number below is read from the store it built "
         "(source.json, records.csv, the shards) and from check.json, folds.json, tilt.json, mounting_probe.json, "
         "stage1_root.json, speed.json and rebuild.json of this directory. Spec: docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, "
         f"sections 4.4 and 4.5 and the S2.6 row of section 7. MoveAtlas commit: {prov['moveatlas_head']}. Store root: "
         f"{prov['store_root']} (tier {store_json.get('tier')}, marker {WS.MARKER}). Name tables "
         f"{prov['names_sha256']} (names.json; vocabulary {prov['vocabulary']}, SHA-256 {prov['vocabulary_sha256']}). "
         f"Process priority: {prov['priority']}."
         + (f" QUICK RUN: {ARGS.limit} recordings per source." if ARGS.limit else ""), "",
         "## Done-when checks of S2.6", "", "| check | measured |", "|---|---|"]
    fx = check["forbidden"]
    L.append(f"| a read-back finds no lockbox, EVAL, CODY or INTERNAL person | WindowStore.check at tier RELEASE: "
             f"{check['records']} records, {check['recordings']} recordings, {check['shards']} shards, "
             f"{check['values_read']} values read, persons refused {check['persons_refused']}; the {fx['persons']} "
             f"persons of the store against the index: lockbox {len(fx['lockbox'])}, EVAL {len(fx['EVAL'])}, "
             f"EXCLUDED_V1 {len(fx['EXCLUDED_V1'])}, CODY {len(fx['CODY'])}, INTERNAL {len(fx['INTERNAL'])}; "
             f"{check['windows_validated']} whole-recording windows validated under the {len(names.tasks())} tasks "
             f"of the version-2 vocabulary |")
    for n, u in folds["units"].items():
        if "proposed" in u:
            pr = u["proposed"]
            L.append(f"| the dev fold of {u['unit']} hashes as in folds_proposed.csv | draw seeded by "
                     f"'{u['seed_key']}': "
                     f"dev SHA-256 {u['draw_sha256'].get('dev')} (proposed {pr['sha256_dev']}): equal "
                     f"{pr['draw_dev_equal']}; train equal {pr['draw_train_equal']}; per person equal "
                     f"{pr['per_person_equal']}; stored dev persons hash equal {pr['store_dev_equal']} (dev persons "
                     "without a stored recording: "
                     f"{', '.join(pr['dev_persons_without_stored_recording']) or 'none'}) |")
    for src, d in tilt.items():
        nat = d["native"]
        mx = max((v["stored_angle_deg_max"] or 0.0) for v in d["sites"].values()) if d["sites"] else None
        L.append(f"| each turned site's mean acceleration lies on its canonical up axis ({src}; a check of the code) | "
                 f"native records (the build): {nat.get('tilt_sites_turned', 0)} sites turned, largest angle after the "
                 f"turn {nat.get('tilt_angle_max_deg', 0.0):.3g} deg, kept unturned: no acceleration "
                 f"{nat.get('tilt_sites_identity_no_acc', 0)}, mean under {MT.MIN_MEAN_G} g "
                 f"{nat.get('tilt_sites_identity_short_mean', 0)}; stored band copies: largest angle {mx:.4g} deg |")
    if probe is not None and probe.get("not_run"):
        L.append(f"| mounting probe on {probe['source']} | not run: {probe['not_run']} |")
    elif probe is not None:
        a, b = probe["arms"]["before"], probe["arms"]["after"]
        L.append(f"| mounting probe on {probe['source']} (no threshold; for the owner before S2.11) | balanced "
                 f"accuracy of the group, before canonicalization {a['balanced_accuracy']:.4f} (chance "
                 f"{a['chance']['mean']:.4f}), "
                 f"after {b['balanced_accuracy']:.4f} (chance {b['chance']['mean']:.4f})"
                 + "".join(f"; wider reading {k} {v['balanced_accuracy']:.4f} (chance {v['chance']['mean']:.4f})"
                           for k, v in (probe.get("wider_arms") or {}).items())
                 + "; section Mounting probe |")
    if rebuild:
        L.append(f"| person-list hashes equal over two builds | second build into {rebuild['root']}: person hashes "
                 f"equal {rebuild['person_hashes_equal']}; files equal: "
                 + ", ".join(f"{n} {v['files_sha256_equal']}" for n, v in rebuild["sources"].items()) + " |")
    else:
        L.append("| person-list hashes equal over two builds | not run (--verify-rebuild) |")
    L.append(f"| the stage-1 root unchanged | {s1['before']['root']}: {s1['before']['files']} files, listing SHA-256 "
             f"{s1['before']['sha256']} before, {s1['after']['sha256']} after: equal {s1['equal']} |")
    L.append("| force-only and 2D sources listed as pending | "
             + "; ".join(f"{k}: pending item {v}" for k, v in S.STAGE2_PENDING.items())
             + "; koa_pd_nm: not in stage 2 (item 6) |")
    L += ["", "## Units", "", "persons per fold of the store (source.json); input hours from the stored input stream; "
          "sex and age unknown: records whose source has no demographic table in mdfm.data.sources (sex 2, age "
          "band 8); "
          "tasks under the version-2 vocabulary.", ""]
    L += ["| " + " | ".join(units[0]) + " |", "|" + "---|" * len(units[0])]
    L += ["| " + " | ".join(str(r[c]) for c in units[0]) + " |" for r in units]
    L += ["", "## Folds", "", f"folds.json; folds_proposed.csv SHA-256 {folds['folds_proposed_sha256']}.", "",
          "| unit | fold | share | seed key | persons | per fold | draw SHA-256 per fold |",
          "|---|---|---|---|---|---|---|"]
    for n, u in folds["units"].items():
        L.append(f"| {u['unit']} | {u['fold']} | {u['share']} | {u['seed_key']} | {u['persons']} | "
                 f"{', '.join(f'{k} {v}' for k, v in u['per_fold'].items())} | "
                 f"{'; '.join(f'{k} {v}' for k, v in u['draw_sha256'].items())} |")
    L += ["", "## Tilt canonicalization", "",
          "mdfm.data.mounting.turn_record at store build; canonical up axes: the generic family of "
          f"reports/mount_priors/priors.json (SHA-256 {prov['mount_priors_sha256']}). Stored angle: the stored mean "
          "acceleration (band copy, float16) against the up axis; turn: the raw mean (R^T times the stored mean) "
          "against the up axis, the size of the turn. Quantiles p10, p50, p90 over the turned records.", "",
          "| source | site | records | turned | stored angle max (deg) | stored angle p10, p50, p90 | turn p10, p50, "
          "p90 "
          "(deg) | turn max |", "|---|---|---|---|---|---|---|---|"]
    for src, d in tilt.items():
        for site, v in d["sites"].items():
            L.append(f"| {src} | {site} | {v['records']} | {v['turned']} | {v['stored_angle_deg_max']} | "
                     f"{_q(v['stored_angle_deg_p10_p50_p90'])} | {_q(v['turn_deg_p10_p50_p90'])} | "
                     f"{v['turn_deg_max']} |")
    if probe is not None and not probe.get("not_run"):
        L += ["", "## Mounting probe (no threshold, for the owner before S2.11)", "",
              f"On {probe['source']}: per record, for each of the sites {', '.join(probe['sites'])}, a presence flag "
              "and the heading features that the turn leaves (cosine and sine of twice the angle, in the horizontal "
              "plane of the turned frame, of the first principal axis of the horizontal acceleration and of the "
              "angular rate, from the stored band copies); before canonicalization the mean-acceleration direction of "
              "each site "
              "in the sensor frame is added (3 numbers). The heading angles are the same numbers in both arms by "
              "construction (mdfm.data.mounting, module docstring), so the arms differ by what the turn removes. "
              f"Classifier: standardized features, logistic regression (C {probe['C']}, class weights "
              f"{probe['class_weight']}), {probe['folds']} person-grouped folds (StratifiedGroupKFold, seed "
              f"{probe['seed']}); balanced accuracy of the pooled out-of-fold predictions per record, and per person "
              f"(majority of the person's records). Chance: the same classifier on {probe['permutations']} "
              "group-permuted person sets. Gait differs between the groups too, so the heading features carry gait as "
              "well as mounting: the numbers bound what a linear reader takes from these features, not the mounting "
              "alone.", "",
              "Records per group: " + ", ".join(f"{k} {v}" for k, v in probe["records_per_group"].items())
              + "; persons per group: " + ", ".join(f"{k} {v}" for k, v in probe["persons_per_group"].items()) + ".",
              "", "| arm | features | records | persons | balanced accuracy | per person | chance mean | chance sd | "
              "chance "
              "min, max | permutations at or above |", "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ("before", "after"):
            r = probe["arms"][arm]
            c = r["chance"]
            L.append(f"| {arm} | {r['features']} | {r['records']} | {r['persons']} | {r['balanced_accuracy']:.4f} | "
                     f"{r['person_balanced_accuracy']:.4f} | {c['mean']:.4f} | {c['sd']:.4f} | {c['min']:.4f}, "
                     f"{c['max']:.4f} | {c['at_or_above_observed']} of {len(c['values'])} |")
        L.append("")
        L.append("Undefined angles set to 0: " + ", ".join(f"{k} {v['undefined_angles_set_to_0']}"
                                                          for k, v in probe["arms"].items()) + ".")
        if probe.get("wider_arms"):
            L += ["", "### Wider linear readings (added after the review of S2.6; not the spec's probe)", "",
                  "The heading angles are a narrow view of what the turn leaves. The same classifier, folds and "
                  "permutations read, per site, the presence flag and the 3 x 3 covariance of the acceleration and of "
                  "the angular rate divided by its trace (6 numbers each; mdfm.data.mounting.covariance): in the "
                  "sensor frame with the mean-acceleration direction (covariance_before), and in the turned frame "
                  "(covariance_after); and the presence flags alone (presence: which sites a record holds). These "
                  "covariances carry the gait's shape as well as the mounting, so the difference between the two "
                  "covariance arms, not their level, is what the turn removes from this reader.", "",
                  "| arm | features | balanced accuracy | per person | chance mean | chance sd | chance min, max | "
                  "permutations at or above | undefined set to 0 |", "|---|---|---|---|---|---|---|---|---|"]
            for arm, r in probe["wider_arms"].items():
                c = r["chance"]
                L.append(f"| {arm} | {r['features']} | {r['balanced_accuracy']:.4f} | "
                         f"{r['person_balanced_accuracy']:.4f} | {c['mean']:.4f} | {c['sd']:.4f} | {c['min']:.4f}, "
                         f"{c['max']:.4f} | {c['at_or_above_observed']} of {len(c['values'])} | "
                         f"{r['undefined_set_to_0']} |")
    L += ["", "## Pending stage-2 sources", "",
          "Outside S2a by the spec (section 2 and section 9): "
          + "; ".join(f"{k} (item {v}: {'force input stem' if v == 4 else 'modys reader, postures, tracker'})"
                      for k, v in S.STAGE2_PENDING.items()) + "; koa_pd_nm deferred to a B6 INTERNAL run (item 6).", ""]
    L += ["## Training speed during the build (reports/gpu_queue/STATUS.md, read only)", ""]
    for k in ("start", "after_build", "end"):
        sp = speed.get(k) or {}
        L.append(f"- {k} ({sp.get('at')}): SLOW flag {sp.get('slow_flag')}; " + "; ".join(sp.get("rows") or ["no row"]))
    L += ["", "## Person-list hashes", "", "| source | fold | persons | SHA-256 |", "|---|---|---|---|"]
    for n, s in summaries.items():
        for fold, h in s["person_sha256"].items():
            L.append(f"| {n} | {fold} | {s['persons'][fold]} | {h} |")
    L.append("")
    return "\n".join(L)


if ARGS.stage2:
    from mdfm.data import mounting as MT  # noqa: E402
    from mdfm.factory import layouts as L_  # noqa: E402
    stage2_main()
    sys.exit(0)

t_start = time.time()
try:
    with open(os.path.join(ARGS.report, "_provenance.json"), encoding="utf-8") as f_:
        PREVIOUS = json.load(f_)
except (OSError, ValueError):
    PREVIOUS = None
head = moveatlas_head()
print("MoveAtlas:", head, "| store:", rel(ROOT), flush=True)
before = None if ARGS.skip_build else disk_state(ROOT)
built = {} if ARGS.skip_build else build_all(ROOT, "build")
diff = None
if before is not None:
    diff = store_diff(before, disk_state(ROOT))
    dump("store_diff.json", diff)
store = WS.WindowStore(ROOT)
t0 = time.time()
index = C.index()
check = store.check(index=index)
check["seconds"] = round(time.time() - t0, 1)
print("read-back:", {k: v for k, v in check.items() if k not in ("persons", "max_abs")}, flush=True)
windows_ok = 0
for i in range(len(store.recordings)):
    W.validate(store.window(i))
    windows_ok += 1
check["windows_validated"] = windows_ok
check["target_grid"] = target_grid(store.records)
hashes = store.person_hashes()
summaries = dict(store.sources)
store.close()

rebuild = None
if ARGS.verify_rebuild:
    other = WS.assert_store_root(ARGS.verify_rebuild)
    second = build_all(other, "rebuild")
    s2 = WS.WindowStore(other)
    h2 = s2.person_hashes()
    s2.close()
    rebuild = {"root": rel(other), "sources_built": NAMES,
               "person_hashes_equal": {n: h2.get(n) for n in NAMES} == {n: hashes.get(n) for n in NAMES}, "sources": {}}
    for n in NAMES:
        a, b = summaries[n], second[n][0]
        rebuild["sources"][n] = {"person_sha256_equal": a["person_sha256"] == b["person_sha256"],
                                 "files_sha256_equal": a["files_sha256"] == b["files_sha256"],
                                 "files": len(a["files_sha256"]), "seconds": second[n][1]}
    dump("rebuild.json", rebuild)

bands = band_checks()
dump("band_checks.json", bands)
dump("check.json", check)
root_check = root_world_check(ROOT, ARGS.root_check)
dump("root_check.json", root_check)
dump("hashes.json", {"person_sha256": hashes, "files_sha256": {n: s["files_sha256"] for n, s in summaries.items()},
                     "note": "person_sha256: SHA-256 of the sorted person ids joined by newlines, per source and fold"})

# ---------------------------------------------------------------- tables
rows = []
for n, s in summaries.items():
    for kind, a in s["streams"].items():
        gs = a["group_samples"]
        f16 = {k: v for k, v in s["f16"].items() if k.startswith(kind + ".")}
        rows.append({"source": n, "stream": kind, "records": a["records"], "hours": round(a["seconds"] / 3600, 3),
                     "kept_samples": a["samples"], "group_samples": gs,
                     "group_samples_invalid": gs - a["group_samples_valid"],
                     "group_samples_invalid_in_input": a["group_samples_invalid_in_input"],
                     "group_samples_made_valid": a["group_samples_made_valid"],
                     "native_group_samples_masked_short": a["native_group_samples_masked_short"],
                     "native_group_samples_zero_fill": a.get("native_group_samples_zero_fill", 0),
                     "records_zero_fill": a.get("records_zero_fill", 0),
                     "group_samples_f16_overflow": a["group_samples_f16_overflow"],
                     "k": "|".join(sorted(k[2:] for k in a if k.startswith("k="))),
                     "fps_native": "|".join(sorted(k[11:] for k in a if k.startswith("fps_native="))),
                     "grid_max_dev_s": a["grid_max_dev_s"],
                     "f16_max_abs_err": "|".join(f"{k.split('.')[1]} {v['max_abs_err']:.4g}" for k, v in f16.items()),
                     "f16_share_over_bound": "|".join(f"{k.split('.')[1]} {v['over_bound'] / v['entries']:.3g}"
                                                      for k, v in f16.items() if v["over_bound"] is not None),
                     "bytes": os.path.getsize(os.path.join(ROOT, n, kind, "values.f16"))
                     + os.path.getsize(os.path.join(ROOT, n, kind, "valid.u1"))})
cols = list(rows[0]) if rows else []
with open(os.path.join(ARGS.report, "sources.csv"), "w", encoding="utf-8", newline="") as f:
    import csv
    wr = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)


def md_table(items, columns):
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in items:
        out.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return out


code = {m: sha_text(os.path.join(TREE, "mdfm", "data", m))
        for m in ("window_store.py", "sources.py", "windows.py", "mocap_raw.py", "c3d.py")}
prov = {"script": "build_window_store.py", "moveatlas_root": ARGS.moveatlas, "moveatlas_head": head,
        "run_at": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
        "store_root": rel(ROOT), "format": WS.FORMAT, "limit": ARGS.limit, "sources": NAMES,
        "index_sha256": {rel(C.manifest_path()): sha_text(C.manifest_path()), rel(C.index_path()): sha_text(C.index_path())},
        "code_sha256": code, "build_seconds": {n: v[1] for n, v in built.items()},
        "runtime_s": round(time.time() - t_start, 1)}
if set(NAMES) != set(summaries) and PREVIOUS:
    prov["previous_provenance"] = PREVIOUS          # the run that built the sources this run left as they were
dump("_provenance.json", prov)

L = ["# Window store (plan tasks B4.6 and B4.13b)", "",
     f"Generated by scripts/build_window_store.py; every number below is read from the store it built (source.json, "
     f"records.csv, the shards) and from band_checks.json, check.json, rebuild.json, store_diff.json and "
     f"root_check.json of this directory. Sources built in this run: {', '.join(built) if built else 'none'}. MoveAtlas "
     f"commit: {head}. Store root: {rel(ROOT)} (format {WS.FORMAT}; <store> does not exist yet, spec question 12). "
     f"Rules in mdfm/data/window_store.py." + (f" QUICK RUN: {ARGS.limit} recordings per source." if ARGS.limit else ""),
     "", "## Done-when checks of B4.6", "", "| check | measured |", "|---|---|"]
sine = bands["sine_500hz"]
L.append(f"| filter: at 500 Hz a 10 Hz sinusoid passes and a 60 Hz one is removed | k {sine['k']}, output "
         f"{sine['fps_out']:g} Hz; 10 Hz amplitude {sine['amplitude_10hz']}; 60 Hz RMS ratio {sine['rms_ratio_60hz']} "
         f"(zero-phase gain at 60 Hz {bands['plans']['imu@500']['zero_phase_gain']['60 Hz']}) |")
nr = bands["nan_run"]
L.append(f"| a NaN run does not spread past its edges | 1000 Hz force, 300-sample run: outputs outside the run differ by "
         f"{nr['max_abs_diff_outside']} when the run's content changes; kept samples valid outside "
         f"{nr['kept_valid_outside']} of {nr['kept_outside']}, inside {nr['kept_valid_inside']} |")
grid = {f"{r['source']}/{r['stream']}": r["grid_max_dev_s"] for r in rows}
L.append(f"| kept timestamps are native | offset + j k / fps_native against the native times of every stored record: "
         f"max deviation {max(grid.values()) if grid else float('nan'):.3g} s over {len(grid)} source streams |")
tg = check["target_grid"]
if tg:
    L.append("| target streams share the input's sample grid (targets filtered over their whole record, kept inside the "
             "input span) | " + "; ".join(f"{k}: on the input grid {v['on_input_grid']} of {v['records']}, same n "
                                          f"{v['same_n_as_input']}, last sample at or past the span end "
                                          f"{v['last_at_or_past_span_end']}" for k, v in tg.items()) + " |")
f16rows = [(n, k, v) for n, s in summaries.items() for k, v in s["f16"].items()]
acc = [v for _, k, v in f16rows if k == "imu.acc"]
gyro = [v for _, k, v in f16rows if k == "imu.gyro"]
if acc and gyro:
    L.append(f"| float16 round trip under 1 mg and 0.1 deg/s | max error acc {max(v['max_abs_err'] for v in acc):.4g} g, "
             f"gyro {max(v['max_abs_err'] for v in gyro):.4g} deg/s; entries over the bound: acc "
             f"{sum(v['over_bound'] for v in acc)} of {sum(v['entries'] for v in acc)}, gyro "
             f"{sum(v['over_bound'] for v in gyro)} of {sum(v['entries'] for v in gyro)} (float16 spacing: the bounds hold "
             f"below 4 g and 256 deg/s; above, the error is at most 2^-11 of the value; the largest errors sit on the "
             f"largest stored values, section Largest stored magnitudes) |")
eb = bands["emg_burst"]
L.append("| a synthetic 50 ms EMG burst peaks inside the burst | " + "; ".join(
    f"{k} Hz: peak at {v['peak_s']} s for the burst {v['burst_s'][0]} to {v['burst_s'][1]} s, "
    f"{v['peak_minus_median_log_mv']} log mV above the median" for k, v in eb.items()) + " |")
L.append(f"| a read-back of every shard finds no lockbox, EVAL or CODY person | {check['records']} records, "
         f"{check['recordings']} recordings, {check['shards']} shards, {check['bytes']} bytes, {check['values_read']} "
         f"values read in {check['seconds']} s; persons refused {check['persons_refused']}; "
         f"{check['windows_validated']} whole-recording windows validated |")
if rebuild:
    L.append(f"| person-list hashes stable across two builds | second build of {', '.join(rebuild['sources'])} into "
             f"{rebuild['root']}: person hashes equal {rebuild['person_hashes_equal']}; files equal: "
             + ", ".join(f"{n} {v['files_sha256_equal']}" for n, v in rebuild["sources"].items()) + " |")
else:
    L.append("| person-list hashes stable across two builds | not run (--verify-rebuild) |")
L.append("| a lockbox-store path is refused | " + "; ".join(f"{k}: {v}" for k, v in bands["refused_roots"].items()) + " |")
wf = summaries.get("weargait_pd", {}).get("streams", {}).get("force")
if wf:
    L.append(f"| the weargait insole samples masked by mdfm.data.sources stay masked | {wf['records']} insole records: "
             f"{wf['group_samples_invalid_in_input']} kept foot samples invalid in the input (sources.pair zero gaps "
             f"and NaN), {wf['group_samples'] - wf['group_samples_valid']} invalid in the store, "
             f"{wf['group_samples_made_valid']} made valid |")
emg = {n: s["streams"]["emg"] for n, s in summaries.items() if "emg" in s.get("streams", {})}
zf = bands["emg_zero_fill"]
if emg:
    L.append("| raw EMG zero-fill (a channel at exactly 0 for 20 ms or more) is masked, not stored as signal | "
             f"analytic, 1000 Hz with 300 leading zeros: {zf['masked_as_zero_fill']} samples masked, kept valid in the "
             f"fill {zf['kept_valid_in_fill']}, after it {zf['kept_valid_after']} of {zf['kept_after']}, lowest value "
             f"{zf['lowest_minus_median_log_mv']} log mV from the median; store: "
             + "; ".join(f"{n}: {v.get('native_group_samples_zero_fill', 0)} native channel samples in "
                         f"{v.get('records_zero_fill', 0)} of {v['records']} records" for n, v in emg.items()) + " |")
root_src = {n: s["root"] for n, s in summaries.items() if s.get("root")}
if root_src:
    parts = []
    for n, v in root_src.items():
        p3 = v["streams"].get("pose3d", {})
        parts.append(f"{n}: {p3.get('records_with_root', 0)} of {p3.get('records', 0)} pose3d records carry a track "
                     f"({len(v['missing'])} unreadable), valid root samples {p3.get('samples_valid', 0)} of "
                     f"{p3.get('samples_with_root', 0)}")
    L.append("| B4.13b: the pose3d shards of gait120 and stroke_gait_vancriekinge carry the raw root track | "
             + "; ".join(parts) + " |")
if check.get("root"):
    L.append("| B4.13b: the root files tile the pose3d shards (read-back) | " + "; ".join(
        f"{k}: {v['records']} blocks, {v['samples']} samples, {v['records_with_root']} with a valid sample, "
        f"{v['samples_valid']} valid samples, {v['bytes']} bytes, largest abs {v['max_abs_m']:.4g} m"
        for k, v in check["root"].items()) + " |")
if root_check:
    L.append("| B4.13b: stored pose x scale plus stored root gives the raw joints | " + "; ".join(
        f"{k}: {v['recordings']} recordings evenly spaced, stored root equal to mocap_raw.root_for_paired on "
        f"{v['root_equal_to_root_for_paired']}; world-joint error over {v['joint_samples_compared']} of "
        f"{v['joint_samples']} joint samples: median {v['err_median_m'] * 1e3:.3f} mm, p99 {v['err_p99_m'] * 1e3:.3f} mm, "
        f"max {v['err_max_m'] * 1e3:.3f} mm" for k, v in root_check.items() if v["err_max_m"] is not None) + " |")
if diff:
    L.append("| B4.13b: the build changed no file but the ones it had to (SHA-256 of every store file before and after) "
             "| " + "; ".join(f"{n} ({'rebuilt' if v['rebuilt'] else 'not rebuilt'}): {len(v['unchanged'])} files "
                             f"unchanged, changed {v['changed'] or 'none'}, added {v['added'] or 'none'}, removed "
                             f"{v['removed'] or 'none'}, person hashes equal {v['person_sha256_equal']}"
                             for n, v in diff["sources"].items()) + " |")

L += ["", "## Sources", "",
      "Recordings under 1 s are dropped by windows.cut (records_short); recordings without their input stream or "
      "whose streams do not pair are dropped by sources.paired (dropped_*).", ""]
src_rows = []
for n, s in summaries.items():
    st = s["stats"]
    src_rows.append({"source": n, "recordings": s["recordings"], "records": s["records"],
                     "persons": ", ".join(f"{k} {v}" for k, v in s["persons"].items()),
                     "records_short": st.get("records_short", 0),
                     "dropped": ", ".join(f"{k[8:]} {v}" for k, v in st.items() if k.startswith("dropped_")) or "0",
                     "irregular_grid": st.get("recordings_irregular_grid", 0),
                     "build_s": built.get(n, (None, ""))[1]})
L += md_table(src_rows, list(src_rows[0])) if src_rows else []
L += ["", "## Streams", "", "group samples: kept samples x groups (IMU sites, feet, EMG channels, joints); invalid "
      "includes the source's own gaps (weargait insole zeros, sample_valid), the runs too short to filter and the "
      "float16 overflows.", ""]
L += md_table(rows, [c for c in cols if c != "grid_max_dev_s"]) if rows else []
L += ["", "## Largest stored magnitudes", "",
      "From the read-back of every shard (check.json, max_abs): per source and channel family, the largest stored "
      "|value|, the median over records of each record's largest |value|, the record holding the largest, and the "
      "signed range of the valid entries (lowest and highest, with the medians over records of each record's lowest "
      "and highest). Units: imu acc g, gyro deg/s; force N (weargait: unit unknown); emg the natural log of the "
      "record unit (mV after the record's unit_scale; a log-envelope below 0 is under 1 mV, so in a family whose "
      "values are mostly negative the largest |value| is the quietest floor, not a peak: the signed columns tell "
      "which); pose3d canonical torso units.", "",
      "| source/family | records | largest abs | median of record max abs | record of the largest abs | lowest | "
      "median record low | median record high | highest |", "|---|---|---|---|---|---|---|---|---|"]


def g6(x):
    return "" if x is None else f"{x:.6g}"


for k, v in check["max_abs"].items():
    L.append(f"| {k} | {v['records']} | {v['max']:.6g} | {v['median_record_max']:.6g} | {v['max_person_id']} "
             f"{v['max_key']} | {g6(v.get('lo'))} | {g6(v.get('median_record_lo'))} | {g6(v.get('median_record_hi'))} | "
             f"{g6(v.get('hi'))} |")
if root_src:
    L += ["", "## Root tracks", "",
          "The raw pelvis track of mdfm.data.mocap_raw on the pose3d streams (gait120 PELV_C of the TRC, stroke PELO of "
          "the C3D rotated to y up), float32 m in root.f32 and root_valid.u1 beside the pose3d shard (source.json, root). "
          "pose valid, root invalid: stored pose samples with a valid joint whose root sample is invalid.", "",
          "| source | pose3d records | with a track | unreadable | samples with a track | root valid | pose valid, root "
          "invalid | records all valid | records none valid | largest abs (m) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in root_src.items():
        p3 = v["streams"].get("pose3d", {})
        L.append(f"| {n} | {p3.get('records', 0)} | {p3.get('records_with_root', 0)} | {len(v['missing'])} | "
                 f"{p3.get('samples_with_root', 0)} | {p3.get('samples_valid', 0)} | "
                 f"{p3.get('samples_pose_valid_root_invalid', 0)} | {p3.get('records_root_all_valid', 0)} | "
                 f"{p3.get('records_root_none_valid', 0)} | {p3.get('max_abs_m', 0.0):.4g} |")
    for n, v in root_src.items():
        for subject_ns, key, why in v["missing"][:10]:
            L.append(f"- {n} {subject_ns} {key}: {why}")
if diff:
    L += ["", "## Store files before and after this build", "",
          f"store_diff.json: SHA-256 of every file under each source folder before the build and after it. store.json "
          f"changed: {diff['store_json_changed']}.", "",
          "| source | rebuilt | unchanged | changed | added | removed | person hashes equal |", "|---|---|---|---|---|---|---|"]
    for n, v in diff["sources"].items():
        L.append(f"| {n} | {v['rebuilt']} | {len(v['unchanged'])} | {', '.join(v['changed']) or '-'} | "
                 f"{', '.join(v['added']) or '-'} | {', '.join(v['removed']) or '-'} | {v['person_sha256_equal']} |")
L += ["", "## Band plans", "", "| stream | kind | k | fps out | padlen | rms_n | zero-phase gain |", "|---|---|---|---|---|---|---|"]
for k, v in bands["plans"].items():
    L.append(f"| {k} | {v['kind']} | {v['k']} | {v['fps_out']:g} | {v['padlen']} | {v['rms_n']} | "
             + ", ".join(f"{f} {g}" for f, g in v.get("zero_phase_gain", {}).items()) + " |")
L += ["", "## Person-list hashes", "", "SHA-256 of the sorted person ids joined by newlines (hashes.json).", "",
      "| source | fold | persons | SHA-256 |", "|---|---|---|---|"]
for n, s in summaries.items():
    for fold, h in s["person_sha256"].items():
        L.append(f"| {n} | {fold} | {s['persons'][fold]} | {h} |")
L.append("")
with open(os.path.join(ARGS.report, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(L))
print(f"done in {time.time() - t_start:.0f} s; report {rel(ARGS.report)}", flush=True)
