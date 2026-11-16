"""Tests of scripts/probe_seizeit2_events.py (task T28) on synthetic inputs; no MoveAtlas file is read.

The script parses its arguments when loaded, so it is loaded once with a synthetic MoveAtlas root (a folder
holding INDEX.md) and an output folder under tmp_path; every path the tests use is set on the module.
The tests check the draft label mapping, the lockbox guard and the lockbox hash check, the rest-norm scale
and the per-axis fit on a synthetic mov EDF whose gains and offsets are known (the same result whatever
the read chunk), and one run of main() on a BIDS tree of 12 kept and 25 lockbox subjects: the lockbox
files are never opened nor globbed, their seizures never counted, the class counts and the feasibility per
scope are the expected ones, the contained run is removed (a run contained in some modalities only is kept
and named), and the feasibility is printed. The guard still refuses a lockbox path under python -O, which
strips assert statements.
"""
import csv
import hashlib
import ast
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "probe_seizeit2_events.py")

KEPT = [f"sub-{i:03d}" for i in range(1, 13)]
LOCK = [f"sub-{i:03d}" for i in range(101, 126)]
ACC_LABELS = ["EEG SD ACC X", "EEG SD ACC Y", "EEG SD ACC Z", "ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z"]
EV_COLS = ["onset", "duration", "eventType", "lateralization", "localization", "vigilance", "confidence", "channels",
           "dateTime", "recordingDuration"]
LEVELS = {"bckg": "background", "impd": "impedance check", "sz_foc_ia_m_hyperkinetic": "hyperkinetic",
          "sz_foc_a_nm": "non-motor", "sz_foc_ia_m_tonic": "tonic", "sz_foc_ua_um": "unknown motor",
          "sz_gen_nm_typical": "typical absence"}
# the synthetic mov EDF: header units per g, per-axis gain and offset (g) of the EEG module
UNITS_PER_G = 0.5
GAIN = np.array([1.03, 0.97, 1.0])
OFFSET_G = np.array([0.02, -0.01, 0.0])
FS = 25


@pytest.fixture(scope="module")
def P(tmp_path_factory):
    root = tmp_path_factory.mktemp("moveatlas_stub")
    (root / "INDEX.md").write_text("stub\n", encoding="utf-8")
    out = tmp_path_factory.mktemp("load_out")
    old = sys.argv[:]
    sys.argv = ["probe_seizeit2_events.py", "--moveatlas", str(root), "--out", str(out)]
    try:
        spec = importlib.util.spec_from_file_location("probe_seizeit2_events_t28_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


# ------------------------------------------------------------------------------- synthetic files
def write_edf(path, phys, labels, fs=FS, pmin=-1.0, pmax=1.0, dmin=-32768, dmax=32767):
    """A plain EDF of 1 s records; phys is (n_samples, n_signals) in physical units."""
    ns = len(labels)
    nrec = phys.shape[0] // fs
    g = (pmax - pmin) / (dmax - dmin)
    dig = np.clip(np.round((phys[:nrec * fs] - pmin) / g + dmin), dmin, dmax).astype("<i2")

    def f(s, w):
        return str(s).ljust(w)[:w].encode("ascii")
    head = (f("0", 8) + f("X", 80) + f("Y", 80) + f("01.01.20", 8) + f("00.00.00", 8) + f(256 + 256 * ns, 8)
            + f("", 44) + f(nrec, 8) + f(1, 8) + f(ns, 4))
    sh = (b"".join(f(lab, 16) for lab in labels) + f("", 80) * ns + f("g", 8) * ns + f(f"{pmin:g}", 8) * ns
          + f(f"{pmax:g}", 8) * ns + f(dmin, 8) * ns + f(dmax, 8) * ns + f("", 80) * ns + f(fs, 8) * ns + f("", 32) * ns)
    data = dig.reshape(nrec, fs, ns).transpose(0, 2, 1).tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(head + sh + data)


def orientations():
    base = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (-1, 1, 1), (1, -1, -1), (-1, -1, 1)]
    return [np.array(v, float) / np.linalg.norm(v) for v in base]


def synthetic_mov(seed=0):
    """12 orientations at rest (10 blocks of 10 s each), 10 active blocks, one zero block and one block with
    a saturated EEG-module X sample; returns the (n, 6) physical array."""
    rng = np.random.default_rng(seed)
    n_blk = 10 * FS
    parts = []
    for a in orientations():
        eeg = UNITS_PER_G * (GAIN * a + OFFSET_G)
        ecg = UNITS_PER_G * a
        x = np.tile(np.concatenate([eeg, ecg]), (10 * n_blk, 1))
        parts.append(x + rng.normal(0, 0.0005, x.shape))
    t = np.arange(10 * n_blk) / FS
    act = np.tile(UNITS_PER_G * np.array([0, 0, 1.0, 0, 0, 1.0]), (len(t), 1))
    act[:, [0, 3]] += 0.2 * np.sin(2 * np.pi * 2 * t)[:, None]
    parts.append(act)
    zero = np.zeros((n_blk, 6))
    parts.append(zero)
    sat = np.tile(UNITS_PER_G * np.array([0, 0, 1.0, 0, 0, 1.0]), (n_blk, 1)) + rng.normal(0, 0.0005, (n_blk, 6))
    sat[5, 0] = 1.0
    parts.append(sat)
    return np.concatenate(parts)


# ------------------------------------------------------------------------------- label mapping
def test_class_of_draft_mapping(P):
    expect = {
        "bckg": "not_seizure", "impd": "not_seizure",
        "sz_foc_a_m_hyperkinetic": "hyperkinetic", "sz_foc_ia_m_hyperkinetic": "hyperkinetic",
        "sz_uo_m_hyperkinetic": "hyperkinetic",
        "sz_foc_a_nm": "explicitly_non_motor", "sz_foc_ia_nm_behavior": "explicitly_non_motor",
        "sz_gen_nm_typical": "explicitly_non_motor", "sz_uo_nm_behavior": "explicitly_non_motor",
        "sz_foc_ia_m_tonic": "tonic", "sz_gen_m_tonic": "tonic",
        "sz_uo_m_tonicMyio": "other_motor", "sz_gen_m_tonicClonic": "other_motor", "sz_foc_a_m_clonic": "other_motor",
        "sz_foc_ia_m_automatisms": "automatisms", "sz_foc_f2b": "focal_to_bilateral",
        "sz_foc_a_um": "unknown_or_unspecified", "sz_foc_ia": "unknown_or_unspecified", "sz": "unknown_or_unspecified",
        "sz_gen": "unknown_or_unspecified", "sz_uo": "unknown_or_unspecified",
        "sz_foc_ua_m": "motor_unspecified", "sz_gen_m": "motor_unspecified",
        "artefact": "unlisted",
    }
    for lab, cls in expect.items():
        assert P.class_of(lab)[0] == cls, lab
    # unknown or unspecified and motor unspecified are never negatives: only the nm labels are
    assert P.class_of("sz_foc_ua_um")[0] != "explicitly_non_motor"
    assert P.class_of("sz_foc_ua_m")[0] != "explicitly_non_motor"


# ------------------------------------------------------------------------------- lockbox guard
def test_guard_refuses_lockbox_paths(P, tmp_path):
    g = P.Guard(frozenset({"sub-101"}))
    ok = tmp_path / "raw" / "sub-001" / "ses-01" / "eeg" / "sub-001_ses-01_task-x_run-01_events.tsv"
    ok.parent.mkdir(parents=True)
    ok.write_text("onset\n", encoding="utf-8")
    with g.open(str(ok), encoding="utf-8", kind="events_tsv") as f:
        assert f.read() == "onset\n"
    assert g.opened == {"sub-001": 1} and g.by_kind == {"events_tsv": 1}
    bad = str(tmp_path / "raw" / "sub-101" / "ses-01" / "eeg" / "sub-101_ses-01_task-x_run-01_events.tsv")
    with pytest.raises(AssertionError, match="lockbox"):
        g.open(bad, encoding="utf-8")
    with pytest.raises(AssertionError, match="lockbox"):
        g.open(bad.replace("/", "\\"), encoding="utf-8")
    with pytest.raises(AssertionError):
        g.open_meta(str(ok), encoding="utf-8")
    assert "sub-101" not in g.opened


def test_lockbox_guard_holds_under_python_O(tmp_path):
    """python -O strips assert statements: every check of the script is an explicit raise, and the guard
    still refuses a lockbox path in a child python started with -O."""
    with open(SCRIPT, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    root = tmp_path / "moveatlas_stub"
    root.mkdir()
    (root / "INDEX.md").write_text("stub\n", encoding="utf-8")
    bad = tmp_path / "raw" / "sub-101" / "ses-01" / "eeg" / "sub-101_ses-01_task-x_run-01_events.tsv"
    bad.parent.mkdir(parents=True)
    bad.write_text("onset\n", encoding="utf-8")
    code = "\n".join([
        "import importlib.util, sys",
        "script, root, out, bad = sys.argv[1:5]",
        "sys.argv = ['probe_seizeit2_events.py', '--moveatlas', root, '--out', out]",
        "spec = importlib.util.spec_from_file_location('p', script)",
        "P = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(P)",
        "assert False, 'assert statements are active: -O did not apply'",
        "g = P.Guard(frozenset({'sub-101'}))",
        "try:",
        "    g.open(bad, encoding='utf-8').close()",
        "    print('opened', dict(g.opened))",
        "except AssertionError as e:",
        "    print('refused' if 'lockbox' in str(e) else 'other: ' + str(e))",
    ])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    r = subprocess.run([sys.executable, "-O", "-c", code, SCRIPT, str(root), str(tmp_path / "out"), str(bad)],
                       capture_output=True, text=True, env=env, timeout=300, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "refused", r.stdout


def test_load_lockbox_checks_hash(P, tmp_path, monkeypatch):
    lb_csv, lb_sha = tmp_path / "lockbox_persons.csv", tmp_path / "lockbox.sha256"
    write_lockbox(lb_csv, lb_sha)
    monkeypatch.setattr(P, "LOCKBOX_CSV", str(lb_csv))
    monkeypatch.setattr(P, "LOCKBOX_SHA", str(lb_sha))
    subs, _ = P.load_lockbox()
    assert subs == frozenset(LOCK)
    lb_sha.write_text("0" * 64 + "  lockbox_persons.csv\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256"):
        P.load_lockbox()


# ------------------------------------------------------------------------------- scale pass
def _scale(P, path):
    g = P.Guard(frozenset({"sub-101"}))
    hd = P.edf_header(path, g)
    assert hd["size_ok"]
    mods = {m: [P.chan_info(hd, lab) for lab in labels] for m, labels in P.MODULES}
    return P.scale_file(path, hd, mods, g), hd


def test_scale_and_axis_fit_recover_known_gains(P, tmp_path):
    path = str(tmp_path / "sub-001" / "sub-001_ses-01_task-szMonitoring_run-01_mov.edf")
    write_edf(path, synthetic_mov(), ACC_LABELS)
    res, hd = _scale(P, path)
    e, c = res["eeg_module"], res["ecgemg_module"]
    assert e["status"] == "ok" and c["status"] == "ok"
    assert e["n_blocks"] == 12 * 10 + 10 + 2 == hd["nrec"] // 10
    assert e["blocks_zero"] == 1 and e["blocks_saturated"] == 1 and e["blocks_implausible"] == 0
    assert c["blocks_zero"] == 1 and c["blocks_saturated"] == 0
    assert e["blocks_rest"] == 120 and c["blocks_rest"] == 121          # the active blocks are not at rest
    assert c["rest_norm_p50"] == pytest.approx(UNITS_PER_G, abs=2e-4)
    assert c["scale_to_g"] == pytest.approx(1 / UNITS_PER_G, abs=2e-3)
    assert e["scale_to_g"] == pytest.approx(1 / UNITS_PER_G, rel=0.05)
    assert e["covered"] and e["fit"] and e["fit_converged"]
    # raw = units * (GAIN * a + OFFSET_G) and u = raw / p50, so (u + offset) * gain = a with
    # gain_k = p50 / (units * GAIN_k) and offset_k = -units * OFFSET_G_k / p50
    p50 = e["rest_norm_p50"]
    for k, ax in enumerate("xyz"):
        assert e[f"gain_{ax}"] == pytest.approx(p50 / (UNITS_PER_G * GAIN[k]), abs=3e-3)
        assert e[f"offset_{ax}"] == pytest.approx(-UNITS_PER_G * OFFSET_G[k] / p50, abs=3e-3)
    assert e["err_fit_mean_abs"] < 0.3 * e["err_scalar_mean_abs"]
    assert e["err_fit_mean_abs"] < 0.002
    assert abs(e["half_change_rel"]) < 0.1


def test_scale_same_result_whatever_the_chunk(P, tmp_path, monkeypatch):
    path = str(tmp_path / "sub-001" / "sub-001_ses-01_task-szMonitoring_run-02_mov.edf")
    write_edf(path, synthetic_mov(seed=3), ACC_LABELS)
    whole, _ = _scale(P, path)
    monkeypatch.setattr(P, "CHUNK_REC", 70)                            # 7 blocks per read, not a divisor of 1,320
    chunked, _ = _scale(P, path)
    assert whole.keys() == chunked.keys()
    for m in ("eeg_module", "ecgemg_module"):
        assert whole[m].keys() == chunked[m].keys()
        for k, v in whole[m].items():
            if isinstance(v, float):
                assert chunked[m][k] == pytest.approx(v, rel=1e-9, abs=1e-12), (m, k)
            else:
                assert chunked[m][k] == v, (m, k)


# ------------------------------------------------------------------------------- main on a synthetic tree
def write_lockbox(lb_csv, lb_sha):
    with open(lb_csv, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "lane", "person_id", "group", "rule"])
        w.writerow(["other_source", "body", "other_source:001", "all", "x"])
        for s in LOCK:
            w.writerow(["seizeit2", "neural", "seizeit2:" + s[4:], "all", "addendum 2.4 (25)"])
    with open(lb_csv, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(lb_sha, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha}  lockbox_persons.csv\n")


def key(s, run):
    return f"{s}_ses-01_task-szMonitoring_run-{run:02d}"


def ev_row(onset, dur, label, vig="awake"):
    return [onset, dur, label, "n/a", "n/a", vig, "n/a", "n/a", "2020-01-01 00:00:00", 3600]


def build_tree(tmp_path):
    here = tmp_path / "mdfm"
    raw = tmp_path / "moveatlas" / "datasets" / "eeg" / "seizeit2" / "raw"
    for d in (here / "reports" / "lockbox", here / "reports" / "sync_seizeit2", here / "reports" / "person_ids", raw):
        d.mkdir(parents=True)
    (tmp_path / "moveatlas" / "INDEX.md").write_text("stub\n", encoding="utf-8")
    write_lockbox(here / "reports" / "lockbox" / "lockbox_persons.csv", here / "reports" / "lockbox" / "lockbox.sha256")
    (raw / "events.json").write_text(json.dumps({"eventType": {"Levels": LEVELS}}), encoding="utf-8")
    events = {}                                   # key -> rows
    runs = []                                     # (key, subject, verdict, emg, chest)
    for i, s in enumerate(KEPT, start=1):
        r1 = [ev_row(0, 3600, "bckg")]
        if i == 1:
            r1 += [ev_row(100, 30, "sz_foc_ia_m_hyperkinetic", "asleep"), ev_row(50, -2, "impd")]
        r2 = [ev_row(0, 3600, "bckg")]
        if i <= 10:
            r2.append(ev_row(200, 40, "sz_foc_ia_m_hyperkinetic"))
        if i <= 9:
            r2.append(ev_row(500, 20, "sz_foc_a_nm"))
        if i == 12:
            r2.append(ev_row(300, 10, "sz_foc_ua_um"))
        r3 = [ev_row(0, 3600, "bckg")]
        if i == 11:
            r3.append(ev_row(700, 25, "sz_foc_ia_m_tonic"))
        events[key(s, 1)], events[key(s, 2)], events[key(s, 3)] = r1, r2, r3
        runs += [(key(s, 1), s, "pass", True, True), (key(s, 2), s, "void", True, True),
                 (key(s, 3), s, "not_checkable", True, False)]
        if i == 12:                               # run-04 is contained in run-02 and repeats its seizure
            events[key(s, 4)] = [ev_row(100, 10, "sz_foc_ua_um")]
            runs.append((key(s, 4), s, "void", True, True))
    for s in LOCK:                                # every lockbox person carries a non-motor seizure in a pass run
        events[key(s, 1)] = [ev_row(0, 3600, "bckg"), ev_row(400, 30, "sz_foc_a_nm"),
                             ev_row(900, 30, "sz_foc_ia_m_tonic")]
        runs.append((key(s, 1), s, "pass", True, True))
    for k, rows in events.items():
        s = k.split("_")[0]
        p = raw / s / "ses-01" / "eeg" / f"{k}_events.tsv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(EV_COLS)
            w.writerows(rows)
    write_edf(str(raw / "sub-001" / "ses-01" / "mov" / f"{key('sub-001', 1)}_mov.edf"), synthetic_mov(), ACC_LABELS)
    lb_mov = raw / LOCK[0] / "ses-01" / "mov" / f"{key(LOCK[0], 1)}_mov.edf"
    lb_mov.parent.mkdir(parents=True)
    lb_mov.write_bytes(b"not an EDF")
    sync = here / "reports" / "sync_seizeit2"
    with open(sync / "runs.csv", "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "subject", "verdict", "has_emg_edf", "has_mov_edf", "acc_ecgemg_module", "eeg_duration_s",
                    "mov_duration_s"])
        for k, s, verdict, emg, chest in runs:
            w.writerow([k, s, verdict, str(emg).lower(), "true", str(chest).lower(), "3600", "3600"])
    with open(sync / "contained_runs.csv", "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run_a", "run_b", "offset_records", "contained_all_modalities"])
        w.writerow([key("sub-012", 4), key("sub-012", 2), "200", "true"])
        w.writerow([key("sub-011", 3), key("sub-011", 1), "10", "false"])      # some modalities only: kept
        w.writerow([key(LOCK[1], 1), key(LOCK[0], 1), "10", "false"])          # lockbox row: dropped
    with open(sync / "windows.csv", "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["key", "subject", "status", "raw_norm_eeg", "raw_norm_ecgemg"])
        w.writerow([key("sub-001", 1), "sub-001", "ok", "0.5", "0.5"])
        w.writerow([key("sub-001", 1), "sub-001", "gap", "9", "9"])
        w.writerow([key(LOCK[0], 1), LOCK[0], "ok", "0.5", "0.5"])
    with open(here / "reports" / "person_ids" / "person_map.csv", "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "subject_ns", "person_id"])
        for s in KEPT + LOCK:
            w.writerow(["seizeit2", "seizeit2:" + s[4:], "seizeit2:" + s[4:]])
    return here, raw


def rows_of(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_main(P, tmp_path, monkeypatch, no_scale=False):
    """main() on the synthetic tree with every open and glob of the module recorded."""
    here, raw = build_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    opened, globbed = [], []
    real_glob = P.glob.glob

    def spy_open(path, *a, **kw):
        opened.append(str(path))
        return open(path, *a, **kw)

    def spy_glob(pattern, *a, **kw):
        globbed.append(str(pattern))
        return real_glob(pattern, *a, **kw)
    stub = types.ModuleType("moveatlas_stamp")
    stub.stamp = lambda root: {"root": root, "head": "0" * 40, "diff_sha256": "1" * 64, "untracked": []}
    monkeypatch.setitem(sys.modules, "moveatlas_stamp", stub)
    monkeypatch.setattr(P, "open", spy_open, raising=False)
    monkeypatch.setattr(P.glob, "glob", spy_glob)
    monkeypatch.setattr(P, "below_normal", lambda: True)
    for name, val in (("RAW", str(raw).replace("\\", "/")), ("OUT", str(out)), ("CACHE", str(out / "_cache")),
                      ("LOCKBOX_CSV", str(here / "reports" / "lockbox" / "lockbox_persons.csv")),
                      ("LOCKBOX_SHA", str(here / "reports" / "lockbox" / "lockbox.sha256")),
                      ("SYNC", str(here / "reports" / "sync_seizeit2")),
                      ("PERSON_MAP", str(here / "reports" / "person_ids" / "person_map.csv")),
                      ("SLOW", str(tmp_path / "no_such_SLOW"))):
        monkeypatch.setattr(P, name, val)
    for name, val in (("fresh", True), ("no_scale", no_scale), ("limit", 0), ("keep_cache", False)):
        monkeypatch.setattr(P.OPT, name, val)
    P.main()
    return out, opened, globbed


def test_main_skips_lockbox_and_counts_classes(P, tmp_path, monkeypatch, capsys):
    out, opened, globbed = run_main(P, tmp_path, monkeypatch)
    printed = capsys.readouterr().out

    # lockbox: never opened, never globbed, never counted
    for p in opened + globbed:
        assert not any(s in p.replace("\\", "/") for s in LOCK), p
    raw_opened = [p.replace("\\", "/") for p in opened if p.replace("\\", "/").startswith(P.RAW + "/")]
    assert len(raw_opened) == 1 + 37 + 2                                  # events.json, events files, mov header and data
    assert all(p.split("/")[len(P.RAW.split("/"))] in KEPT for p in raw_opened if not p.endswith("/events.json"))
    lb = json.loads((out / "lockbox_check.json").read_text(encoding="utf-8"))
    assert lb["subjects_listed"] == 37 and lb["lockbox_dropped_before_reading"] == 25 and lb["subjects_kept"] == 12
    assert lb["lockbox_subjects_opened"] == 0 and lb["subjects_opened"] == 12
    assert lb["runs_csv_lockbox_rows_discarded"] == 25 and lb["windows_csv_lockbox_rows_discarded"] == 1
    assert lb["files_opened_by_kind"] == {"events_tsv": 12 * 3 + 1, "mov_edf": 2}

    # class counts per scope
    cc = {(r["scope"], r["class"]): (int(r["seizures"]), int(r["patients"])) for r in rows_of(out / "class_counts.csv")}
    assert cc[("all_runs", "hyperkinetic")] == (11, 10)
    assert cc[("all_runs", "explicitly_non_motor")] == (9, 9)              # 34 if a lockbox person had been read
    assert cc[("all_runs", "tonic")] == (1, 1)
    assert cc[("all_minus_contained", "tonic")] == (1, 1)                  # its run is contained in part only
    assert cc[("all_runs", "unknown_or_unspecified")] == (2, 1)
    assert cc[("all_minus_contained", "unknown_or_unspecified")] == (1, 1)
    assert cc[("pass_minus_contained", "hyperkinetic")] == (1, 1)
    assert cc[("pass_minus_contained", "explicitly_non_motor")] == (0, 0)
    assert cc[("pass_minus_contained_emg", "hyperkinetic")] == (1, 1)
    assert cc[("chest_module_emg_minus_contained", "hyperkinetic")] == (11, 10)
    assert cc[("chest_module_emg_minus_contained", "tonic")] == (0, 0)
    assert cc[("chest_module_emg_minus_contained", "unknown_or_unspecified")] == (1, 1)
    feas = {(r["scope"], r["class"]): r for r in rows_of(out / "feasibility.csv")}
    assert feas[("all_runs", "hyperkinetic")]["feasible"] == "True"
    assert feas[("all_runs", "explicitly_non_motor")]["feasible"] == "False"
    assert feas[("all_runs", "explicitly_non_motor")]["patients_short"] == "1"
    assert feas[("pass_minus_contained", "hyperkinetic")]["patients_short"] == "9"
    cont = rows_of(out / "contained_events.csv")
    assert len(cont) == 1 and cont[0]["matches_in_run_b"] == "1" and cont[0]["onset_shift_s"] == "200.0"
    S = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert S["events_negative_duration_by_label"] == {"impd": 1}
    assert S["scopes"]["all_runs"]["patients_hyperkinetic_and_non_motor"] == 9
    assert S["scopes"]["pass_minus_contained"]["primary_feasible"] is False
    assert S["contained_runs_nonlockbox"] == [key("sub-012", 4)]
    assert S["contained_runs_partial_nonlockbox_kept"] == [key("sub-011", 3)]

    # the feasibility is printed
    assert "Q8 feasibility on the non-lockbox persons" in printed
    assert "pass_minus_contained" in printed and "hyperkinetic" in printed

    # the scale pass read the kept mov file only, and the cache was removed after a complete run
    sc = rows_of(out / "acc_scale_per_file.csv")
    assert {r["subject"] for r in sc} == {"sub-001"}
    ecg = [r for r in sc if r["module"] == "ecgemg_module"][0]
    assert float(ecg["ratio_to_sync"]) == pytest.approx(1.0, abs=1e-3)
    assert not (out / "_cache").exists()
    prov = json.loads((out / "_provenance.json").read_text(encoding="utf-8"))
    assert set(prov["inputs"]) == {"lockbox_persons.csv", "sync_seizeit2/runs.csv", "sync_seizeit2/contained_runs.csv",
                                   "person_ids/person_map.csv", "sync_seizeit2/windows.csv"}
    assert prov["moveatlas_unchanged_during_run"] is True
    with open(P.__file__, "rb") as f:
        assert prov["script_sha256"] == hashlib.sha256(f.read()).hexdigest()

    # the written reports carry the full-cohort statement and no em-dash
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "883" in md and "owner's permission" in md
    assert "discarded as the sync files were parsed: runs.csv 25, windows.csv 1." in md
    assert "contained in some modalities only, kept: " + key("sub-011", 3) in md
    for name in os.listdir(out):
        if os.path.isfile(out / name):
            assert chr(0x2014) not in (out / name).read_text(encoding="utf-8"), name


def test_main_without_scale_pass(P, tmp_path, monkeypatch):
    out, opened, globbed = run_main(P, tmp_path, monkeypatch, no_scale=True)
    for p in opened + globbed:
        assert not any(s in p.replace("\\", "/") for s in LOCK), p
    assert not any(p.endswith("_mov.edf") for p in opened)
    assert not (out / "acc_scale_per_file.csv").exists()
    assert not (out / "_cache").exists()                                # the empty cache folder is not left behind
    assert "Scale pass not run." in (out / "summary.md").read_text(encoding="utf-8")
    prov = json.loads((out / "_provenance.json").read_text(encoding="utf-8"))
    assert "sync_seizeit2/windows.csv" not in prov["inputs"]
    feas = {(r["scope"], r["class"]): r["patients"] for r in rows_of(out / "feasibility.csv")}
    assert feas[("all_runs", "explicitly_non_motor")] == "9"
