"""Tests of scripts/probe_oxford_structure.py (task T11) on synthetic inputs; no MoveAtlas file is read.

The tests check the guards (the rule hash constant matches rule.txt and rule.sha256, a changed rule stops the
script, the README copies must be identical), the side-effect rule of part A (single block, 4.5 mA cap,
contiguous nondecreasing steps, complete last block), the admissible-null count of part C against the
NR-6.1.7 formula (scope, ramps only, OPEN-06 blocks), the isolated-pulse mask at the file edges, and one
read of a synthetic SmrData .mat in a zip: only the LFP and command channels are read, and the pulse count,
isolated pulses, pair, train rate, command, flat channel, blank channel, contact-level reading and anchor
verdict are the expected ones.
"""
import csv
import hashlib
import importlib.util
import io
import math
import os
import zipfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "probe_oxford_structure.py")


def _script():
    spec = importlib.util.spec_from_file_location("probe_oxford_structure_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


# ----------------------------------------------------------------------------- guards

def test_rule_hash_constant_matches_rule_file():
    if not os.path.isfile(S.RULE):
        pytest.skip("reports/oxford_nonramp/rule.txt absent")
    with open(S.RULE, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    with open(S.RULE_HASH_FILE, encoding="utf-8") as f:
        side = f.read().split()[0]
    assert got == S.RULE_SHA256 == side
    assert S.assert_rule_hash() == S.RULE_SHA256


def test_changed_rule_stops(tmp_path, monkeypatch):
    rule = tmp_path / "rule.txt"
    rule.write_bytes(b"another rule\n")
    side = tmp_path / "rule.sha256"
    side.write_text(hashlib.sha256(b"another rule\n").hexdigest() + "  rule.txt\n", encoding="utf-8")
    monkeypatch.setattr(S, "RULE", str(rule))
    monkeypatch.setattr(S, "RULE_HASH_FILE", str(side))
    with pytest.raises(SystemExit):
        S.assert_rule_hash()


def test_channel_selection_reads_lfp_and_command_only():
    for lab in ("L1", "L8", "R1", "R8"):
        assert S.LFP_RE.fullmatch(lab)
    for lab in ("Am1", "Am2", "Cur1", "Cur2"):
        assert S.CMD_RE.fullmatch(lab)
    for lab in ("L9", "R13", "L12", "LSCM", "RTr", "AcHx", "AccX", "EMG1", "AUX1", "Rs1", "C3", "F-L1"):
        assert not S.LFP_RE.fullmatch(lab) and not S.CMD_RE.fullmatch(lab)


README_LINES = ["Oxford data", "", "In this dataset, we progressively increased stimulation intensity in steps of 0.5 mA "
                "until either the side effect threshold or 4.5 mA were reached. More text.",
                "In Participants 5 to 7, we applied single pulse stimulation (25 pulses) to all 8 contact levels."]


def _zip_with_readme(path, folder, lines):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{folder}/README.txt", "\n".join(lines) + "\n")
        zf.writestr(f"__MACOSX/{folder}/._README.txt", b"\x00\x05")


def test_readme_citation_and_identical_copies(tmp_path):
    zips = []
    for name in ("A.zip", "B.zip"):
        _zip_with_readme(tmp_path / name, name[:-4], README_LINES)
        zips.append(str(tmp_path / name))
    r = S.read_readmes(zips)
    assert (r["stop_line"], r["single_line"], r["step_mA"], r["cap_mA"], r["pulses_per_level"]) == (3, 4, 0.5, 4.5, 25)
    assert r["cite"] == "A.zip:A/README.txt line 3; B.zip:B/README.txt line 3"
    assert r["fragment"] == "until either the side effect threshold or 4.5 mA were reached"
    _zip_with_readme(tmp_path / "C.zip", "C", README_LINES[:2] + [README_LINES[2] + " Changed."] + README_LINES[3:])
    with pytest.raises(SystemExit):
        S.read_readmes(zips + [str(tmp_path / "C.zip")])


# ----------------------------------------------------------------------------- part A

def _summ(file, amps, pattern="ascending_ramp", patient="P9", hemi="L", level="2", repeats=0):
    return {"file": file, "patient": patient, "hemisphere": hemi, "contact_level": level, "step_pattern": pattern,
            "amplitudes_mA": " ".join(f"{a:g}" for a in amps), "n_repeat_amplitude": str(repeats)}


def _g15(file, n, last="complete", first=None):
    out = []
    for b in range(n):
        d = last if b == n - 1 else (first if (first and b == 0) else "complete")
        out.append({"file": file, "on_block": str(b), "command_on_s": "30.0", "delivery": d})
    return out


def test_side_effect_rule():
    readme = {"step_mA": 0.5, "cap_mA": 4.5, "cite": "z:README.txt line 24", "sha256": "x"}
    cases = {
        "single.mat": ([2.0], "complete", None, "not reached or ambiguous", "single_block", None),
        "cap.mat": ([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], "complete", None, "not reached or ambiguous",
                    "reached_4p5_cap", None),
        "ok.mat": ([0.5, 1, 1.5, 2, 2.5, 3], "complete", None, "inferred", "ramp stopped below the cap", 3.0),
        "repeat.mat": ([0.5, 1, 1, 1.5, 2], "complete", "no_artefact", "inferred", "ramp stopped below the cap", 2.0),
        "gap.mat": ([0.5, 1, 2, 2.5], "complete", None, "not reached or ambiguous", "steps_not_contiguous_0p5mA", None),
        "down.mat": ([0.5, 1, 1.5, 1], "complete", None, "not reached or ambiguous", "amplitudes_not_nondecreasing",
                     None),
        "cut.mat": ([0.5, 1, 1.5], "interrupted", None, "not reached or ambiguous", "last_block_interrupted", None),
        "late.mat": ([1, 1.5, 2], "complete", None, "not reached or ambiguous", "steps_not_contiguous_0p5mA", None),
    }
    summ, g15 = [], []
    for f, (amps, last, first, *_rest) in cases.items():
        summ.append(_summ(f, amps, pattern="single_block" if len(amps) == 1 else "ascending_ramp"))
        g15 += _g15(f, len(amps), last=last, first=first)
    rows = {r["file"]: r for r in S.side_effect_rows(summ, g15, readme)}
    for f, (amps, _l, _f, status, reason, step) in cases.items():
        r = rows[f]
        assert (r["status"], r["reason"], r["inferred_side_effect_step_mA"]) == (status, reason, step), f
        assert r["use"] == S.EVIDENCE and r["readme_citation"] == "z:README.txt line 24"
        assert r["step_below_mA"] == (None if step is None else step - 0.5)
    assert rows["repeat.mat"]["n_open06_blocks"] == 1


# ----------------------------------------------------------------------------- part C

def test_q1_k_matches_formula():
    mh = [{"patient": "P2", "hemisphere": "L", "n_ramp": "4", "n_ramp_with_head_accelerometer": "4"},
          {"patient": "P2", "hemisphere": "R", "n_ramp": "3", "n_ramp_with_head_accelerometer": "3"},
          {"patient": "P4", "hemisphere": "L", "n_ramp": "2", "n_ramp_with_head_accelerometer": "0"}]
    p2 = "P2"                           # files a* and b*: P2, whose head accelerometer covers every ramp (mh)
    summ = [_summ("a2", [0.5, 1], patient=p2, level="2"), _summ("a3", [0.5, 1], patient=p2, level="3"),
            _summ("a3b", [1], patient=p2, level="3", pattern="single_block"),
            _summ("a8", [0.5, 1], patient=p2, level="8"),
            _summ("b2", [0.5, 1], patient=p2, hemi="R", level="2"),
            _summ("b4", [0.5, 1], patient=p2, hemi="R", level="4"),
            _summ("b5", [2], patient=p2, hemi="R", level="5", pattern="single_block"),
            _summ("c2", [0.5, 1], patient="P4", level="2"), _summ("c3", [0.5, 1], patient="P4", level="3")]
    g15 = []
    for s in summ:
        n = len(s["amplitudes_mA"].split())
        g15 += _g15(s["file"], n)
    for g in g15:                       # b4 loses every ON block without OPEN-06, c2 one of two
        if g["file"] == "b4" or (g["file"] == "c2" and g["on_block"] == "0"):
            g["delivery"] = "no_artefact"
    rows = {(r["scope"], r["files"], r["open06"]): r for r in S.q1_rows(summ, g15, mh)}

    def k(*hemis):
        tot = 1
        for levels in hemis:
            f = math.factorial(len(levels))
            for lv in set(levels):
                f //= math.factorial(levels.count(lv))
            tot *= f
        return tot - 1

    # q1 scope: P2L and P2R only (P4 has no head accelerometer), level 8 left out
    assert rows[("q1", "all", "with_open06")]["k"] == str(k([2, 3, 3], [2, 4, 5]))
    assert rows[("q1", "ramps_only", "with_open06")]["k"] == str(k([2, 3], [2, 4]))
    assert rows[("q1", "all", "without_open06")]["k"] == str(k([2, 3, 3], [2, 5]))
    assert rows[("q1", "all", "without_open06")]["files_left_out"] == "b4"
    assert rows[("q1", "ramps_only", "without_open06")]["k"] == str(k([2, 3], [2]))
    assert rows[("all_ramp", "all", "with_open06")]["k"] == str(k([2, 3, 3, 8], [2, 4, 5], [2, 3]))
    assert rows[("all_ramp", "all", "without_open06")]["n_on_blocks_left_out"] == 3
    r = rows[("q1", "ramps_only", "without_open06")]
    assert r["smallest_p"] == f"{1 / (int(r['k']) + 1):.6g}"
    assert r["pass_decision_possible_k_ge_20"] is False and r["draws_needed_open33"] is False
    assert rows[("q1", "all", "with_open06")]["n_hemispheres"] == 2


# ----------------------------------------------------------------------------- part B

def test_isolated_mask_edges():
    t = np.array([0, 100, 1000, 1010, 3000])
    assert S.isolated_mask(t, 50).tolist() == [True, True, False, False, True]
    assert S.isolated_mask(np.array([7]), 50).tolist() == [True]
    assert S.isolated_mask(np.zeros(0, np.int64), 50).tolist() == []
    # an interval equal to the gap is not above it
    assert S.isolated_mask(np.array([0, 50, 200]), 50).tolist() == [False, False, True]


FS = 4096
LFP = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)]


def _write_mat(path, labels, data):
    """A minimal MATLAB v7.3 SmrData file: Fs, WvTits (references to uint16 strings) and WvData (samples x channels)."""
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        ds = [refs.create_dataset(f"t{i}", data=np.array([ord(c) for c in lab], np.uint16).reshape(-1, 1))
              for i, lab in enumerate(labels)]
        g = f.create_group("SmrData")
        g.create_dataset("Fs", data=np.array([[float(FS)]]))
        g.create_dataset("WvTits", data=np.array([[d.ref for d in ds]], dtype=h5py.ref_dtype))
        g.create_dataset("WvData", data=data)


@pytest.fixture(scope="module")
def synthetic_file(tmp_path_factory):
    rng = np.random.default_rng(11)
    n = 40 * FS
    labels = LFP + ["Am1", "Am2", "LSCM"]
    X = rng.normal(0.0, 1.0, (n, len(labels)))
    X[:, labels.index("L1")] = 0.0                      # flat contact
    X[:, labels.index("Am1")] = 0.0
    X[:, labels.index("Am2")] = 0.0
    X[:, labels.index("LSCM")] = np.nan                 # EMG samples: never read
    iso = [FS * (1 + i) for i in range(25)]              # 25 pulses 1 s apart
    pair = [30 * FS, 30 * FS + 20]                       # 20 samples apart
    train = [34 * FS + 32 * i for i in range(64)]        # 128 Hz for 0.5 s
    amp = {"L2": 1000.0, "L3": 900.0, "L4": 400.0, "L5": 300.0, "L6": 250.0, "L7": 200.0, "L8": 150.0}
    for p in iso + pair + train:
        for lab, a in amp.items():
            j = labels.index(lab)
            X[p, j] += a
            X[p + 1, j] -= a
            X[p + 2, j] += a / 2
    X[train[0] - 10:train[-1] + 10, labels.index("Am1")] = 1000.0
    d = tmp_path_factory.mktemp("oxford_synth")
    mat = d / "P9_Left_Single.mat"
    _write_mat(str(mat), labels, X)
    zpath = d / "P9.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(mat, "P9/Patient9_Left/P9_Left_Single.mat")
    with zipfile.ZipFile(zpath) as zf:
        zi = zf.getinfo("P9/Patient9_Left/P9_Left_Single.mat")
    return str(zpath), zi, labels, len(iso), len(train)


def test_read_keeps_lfp_and_command_only(synthetic_file):
    zpath, zi, labels, _, _ = synthetic_file
    with zipfile.ZipFile(zpath) as zf:
        raw = zf.read(zi)
    fs, got_labels, X = S.read_lfp_and_command(raw)
    assert fs == FS and got_labels == labels
    assert sorted(got_labels[j] for j in X) == sorted(LFP + ["Am1", "Am2"])
    assert labels.index("LSCM") not in X


def test_analyze_synthetic_single_file(synthetic_file):
    zpath, zi, labels, n_iso, n_train = synthetic_file
    inv = {"hemisphere": "L", "neck_emg": "LSCM", "patient": "P9", "kind": "single-pulse", "contact_level": ""}
    row, trains, series, chans, t = S.analyze_file(zpath, zi, inv)
    assert row["n_pulses"] == n_iso + 2 + n_train == len(t)
    assert row["n_isolated"] == n_iso and row["n_isolated_gap_1p0s"] == 0 and row["n_isolated_gap_0p1s"] == n_iso
    assert row["n_isolated_strict"] == n_iso
    assert (row["n_pairs"], row["n_trains"], row["n_pulses_in_trains"]) == (1, 1, n_train)
    assert row["train_rate_hz_min"] == row["train_rate_hz_max"] == 128.0
    assert row["n_short_pairs"] == 1 and row["short_pair_interval_samples"] == "20:1"
    assert row["pair_interval_ms_min"] == round(1000 * 20 / FS, 3)
    assert row["lfp_flat"] == "L1" and row["n_detection_channels"] == 15
    assert row["command_present"] is True and row["command_active"] == "Am1"
    assert row["command_values_uA"] == "1000" and row["command_segments"] == 3
    assert row["pulses_at_command_nonzero"] == n_train and row["pulses_at_command_zero"] == n_iso + 2
    assert row["n_series"] == 1 and row["series_sizes"] == str(n_iso)
    assert row["isolated_by_blank_channel"] == f"L1:{n_iso}"
    assert row["top2_channels_recorded_hemisphere"] == "L2 L3"
    assert row["contact_reading"] == "consistent with stimulation at level 1" and row["reading_level"] == 1
    assert row["spearman_distance_vs_peak"] == -1.0
    assert row["anchor_verdict"] == "anchor possible"
    ser = [s for s in series if s["row_kind"] == "series"]
    assert len(ser) == 1 and ser[0]["blank_channels"] == "L1" and ser[0]["modal_peak_channel"] == "L2"
    assert ser[0]["ipi_median_s"] == 1.0
    tr = [r for r in trains if r["kind"] == "train"][0]
    assert tr["command_values_uA"] == "1000" and tr["n_pulses"] == n_train
    assert {c["channel"] for c in chans} == set(LFP)
    # no EMG sample entered: an EMG channel full of NaN would have turned some output into NaN
    for r in [row] + trains + series + chans:
        for v in r.values():
            assert not (isinstance(v, float) and math.isnan(v))
    # verdicts on the same pulses under another inventory row
    row2, *_ = S.analyze_file(zpath, zi, dict(inv, neck_emg=""))
    assert row2["anchor_verdict"] == "no: no EMG channel"


def test_outputs_carry_no_time_stamp():
    """D2: only provenance files carry a time; the written CSV and summary have none."""
    for rel in ("reports/oxford_nonramp/files.csv", "reports/oxford_nonramp/summary.md",
                "reports/oxford_lfp/side_effect_steps.csv", "reports/admissible_nulls/oxford_q1.csv"):
        p = os.path.join(TREE, rel)
        if not os.path.isfile(p):
            pytest.skip(f"{rel} absent")
        with open(p, encoding="utf-8") as f:
            text = f.read()
        assert "2026-" not in text and "\u2014" not in text
    with open(os.path.join(TREE, "reports/oxford_lfp/side_effect_steps.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 62 and {r["use"] for r in rows} == {S.EVIDENCE}


# ----------------------------------------------------------------------------- source and summary

def test_source_has_no_outcome_computation():
    """Structure only: no spectral, coherence or filtering call, and only LFP and command channels are read."""
    with open(SCRIPT, encoding="utf-8") as f:
        src = f.read().lower()
    for word in ("welch", "psd", "fft", "coheren", "spectr", "scipy.signal", "hilbert", "butter", "bandpass",
                 "d_b =", "d_b=", "contrast("):
        assert word not in src, word
    assert S.LFP_RE.pattern == "[LR][1-8]" and S.CMD_RE.pattern == "(Am|Cur)[12]"
    assert "\u2014" not in src and chr(0x2014) not in src


def test_summary_md_renders_every_branch(synthetic_file):
    """summary_md on synthetic rows: the P5, Paired, anchor and OPEN-06 sentences, no time stamp, no em-dash."""
    zpath, zi, labels, n_iso, n_train = synthetic_file
    inv = {"hemisphere": "L", "neck_emg": "LSCM", "patient": "P9", "kind": "single-pulse", "contact_level": ""}
    row, trains, series, chans, _t = S.analyze_file(zpath, zi, inv)
    p5 = dict(row, file="P5_Left_Single.mat", patient="P5", n_isolated=0, anchor_verdict="no: no isolated pulse")
    paired = dict(row, file="L7_L2_Paired.mat", kind="paired-pulse", contact_level_from_name="2",
                  anchor_verdict="no: no EMG channel", neck_emg="")
    series_all = list(series) + [dict(s, file="L7_L2_Paired.mat") for s in series]
    readme = {"copies": [{"zip": "A.zip"}, {"zip": "B.zip"}], "sha256": "ab" * 32, "stop_line": 24,
              "fragment": "until either the side effect threshold or 4.5 mA were reached", "step_mA": 0.5,
              "cap_mA": 4.5, "single_line": 26, "pulses_per_level": 25, "cite": "z:README.txt line 24"}
    summ = [_summ("a2", [0.5, 1, 1.5], patient="P2", level="2"), _summ("a3", [0.5, 1], patient="P2", level="3"),
            _summ("c2", [0.5, 1], patient="P4", level="2"), _summ("c3", [2], patient="P4", level="3",
                                                                 pattern="single_block")]
    g15 = []
    for s in summ:
        g15 += _g15(s["file"], len(s["amplitudes_mA"].split()))
    for g in g15:                                        # c2 block 0: an OPEN-06 block outside the Q1 scope
        if g["file"] == "c2" and g["on_block"] == "0":
            g["delivery"] = "no_artefact"
    mh = [{"patient": "P2", "hemisphere": "L", "n_ramp": "2", "n_ramp_with_head_accelerometer": "2"},
          {"patient": "P4", "hemisphere": "L", "n_ramp": "2", "n_ramp_with_head_accelerometer": "0"}]
    se_rows = S.side_effect_rows(summ, g15, readme)
    q1 = S.q1_rows(summ, g15, mh)
    p5_seg = [{"segment": 1, "uA": 500.0, "s": 37.8, "prev": 3367, "prev_hz": "122.92", "mine": 4880},
              {"segment": 2, "uA": 600.0, "s": 30.0, "prev": 3900, "prev_hz": "130.0", "mine": 3900}]
    text = S.summary_md(readme, se_rows, q1, [row, p5, paired], trains, series_all, (62, 62), p5_seg,
                        ["L2", "L3"], S.RULE_SHA256)
    assert "2026-" not in text and chr(0x2014) not in text
    assert "Files that can anchor NR-6.2.1 on structure: P9_Left_Single.mat." in text
    assert "P5_Left_Single.mat holds no isolated pulse" in text
    assert "the largest shortfall per second is in segment 1 (500 uA, 37.8 s): 3367 against 4880 here" in text
    assert "L7_L2_Paired.mat: the pulses of its 1 short pairs" in text
    assert "They are the two contacts nearest to level 1, the only flat LFP level of the left lead" in text
    assert "the file name gives level 2" in text
    assert "P2L: inferred side-effect step per contact level: L2 1.5 mA, L3 1 mA." in text
    assert "Leaving the OPEN-06 blocks out changes no k: they lie in c2 (patients P4; Q1 scope patients P2)" in text
    assert "flat or absent in 62 of 62 files" in text
