"""Tests of the three census probes of weekend task T23 on synthetic inputs (no MoveAtlas file is read).

scripts/probe_neural_manifests.py: rates are distinct at the printed precision, and the per-source table
counts records, ids, hours, sites and head-sensor records.
scripts/probe_merk.py: the .vhdr parser, the chunked force moments against numpy (and no moment without a
finite sample), and a run of main() on a two-run BIDS iEEG folder written here: STN and ECoG MNI rows
matched to the per-subject files after the metre to millimetre conversion, the sample checks of OPEN-12
(a wrong sidecar duration fails the run), force centring and unit variance, and the person map columns.
scripts/probe_ataxie_diagnoses.py: labels are normalized, classed by the legend, an unknown label is
never counted as genetic, and missing labels are counted apart.
"""
import csv
import importlib.util
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t23_test", os.path.join(TREE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NM = _load("probe_neural_manifests")
MK = _load("probe_merk")
AX = _load("probe_ataxie_diagnoses")


@pytest.fixture
def stub_stamp(monkeypatch):
    stub = types.ModuleType("moveatlas_stamp")
    stub.stamp = lambda root: {"root": root, "head": "0" * 40}
    monkeypatch.setitem(sys.modules, "moveatlas_stamp", stub)
    return stub


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------------- neural manifests
def test_fset_distinct_at_printed_precision():
    vals = ["1776.1989342761485", "1776.1989342803518", "1785.7142857", "1754.3859649240237", "1785.7142857"]
    assert NM.fset(vals) == "1754.39|1776.2|1785.71"
    assert NM.fset(["1000.0", "1000"]) == "1000"
    assert NM.fset(["512", "100", "128.0"]) == "100|128|512"


def test_per_source_tables_counts():
    eeg = pd.DataFrame({"source": ["a", "a", "b"], "subject_ns": ["a:1", "a:2", "b:1"], "fps": ["100", "100", "250"],
                        "n_frames": ["360000", "180000", "250"], "n_channels": ["2", "2", "3"],
                        "sites": ["C3|C4", "C3|C4", "FZ|CZ|PZ"]})
    emg = pd.DataFrame({"source": ["a"], "subject_ns": ["a:1"], "fps": ["1776.1989342761485"], "n_frames": ["10"],
                        "n_channels": ["2"], "kinds": ["emg"]})
    imu = pd.DataFrame({"source": ["c", "c", "c"], "subject_ns": ["c:1", "c:1", "c:2"], "fps": ["60", "60", "60"],
                        "n_frames": ["60", "60", "60"], "locations": ["head|trunk", "trunk", "head"]})
    force = pd.DataFrame({"source": ["d"], "subject_ns": ["d:1"], "fps": ["1000"], "n_frames": ["2000"]})
    ps = NM.per_source_tables({"eeg": eeg, "emg": emg, "imu": imu, "force": force})
    a = ps[(ps.family == "eeg") & (ps.source == "a")].iloc[0]
    assert (a.records, a.subject_ids, a.hours, a.fps_values, a.n_channels_values) == (2, 2, 1.5, "100", "2")
    assert (a.n_sites, a.site_combinations) == (2, 1)
    c = ps[(ps.family == "imu") & (ps.source == "c")].iloc[0]
    assert (c.records, c.subject_ids, c.head_records, c.n_sites, c.site_combinations) == (3, 2, 2, 2, 3)
    d = ps[ps.family == "force"].iloc[0]
    assert (d.records, d.record_s_max, d.n_channels_values) == (1, 2.0, "")


# ------------------------------------------------------------------------------- merk: units
def test_vhdr_channels_parses_names_resolution_unit_and_datapoints(tmp_path):
    p = tmp_path / "x.vhdr"
    p.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nCodepage=UTF-8\n"
                 "DataPoints=1234\nNumberOfChannels=3\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
                 "[Channel Infos]\n; comment\nCh1=A\\1B,,0.1,µV\nCh2=MOV_LEFT,,2\nCh3=C,,,mV\n[Comment]\nCh9=Z,,1,V\n",
                 encoding="utf-8", newline="\n")
    chans, fmt, points = MK.vhdr_channels(str(p))
    assert fmt == "IEEE_FLOAT_32" and points == 1234
    assert chans == [("A,B", 0.1, "µV"), ("MOV_LEFT", 2.0, "µV"), ("C", 1.0, "mV")]
    q = tmp_path / "y.vhdr"
    q.write_text("[Common Infos]\n[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\nCh1=A,,1,uV\n",
                 encoding="utf-8", newline="\n")
    assert MK.vhdr_channels(str(q))[2] is None


def test_force_stats_matches_numpy_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(MK, "CHUNK", 7)
    rng = np.random.default_rng(0)
    n_times, n_ch = 50, 3
    data = rng.normal(size=(n_times, n_ch)).astype("<f4")
    data[:10, 2] = 0.0
    data[10:13, 2] = 1.0
    data[20, 1] = np.nan
    path = tmp_path / "x.eeg"
    data.tofile(path)
    out = MK.force_stats(str(path), n_ch, n_times, [("a", 0, 1.0), ("b", 1, 2.0), ("c", 2, 1.0), ("d", 0, float("nan"))])
    x = data[:, 0].astype(np.float64)
    assert out["a"]["n"] == 50 and out["a"]["mean"] == pytest.approx(x.mean()) and out["a"]["std"] == pytest.approx(x.std())
    y = data[:, 1].astype(np.float64) * 2.0
    y = y[np.isfinite(y)]
    assert out["b"]["nonfinite"] == 1 and out["b"]["n"] == 49 and out["b"]["std"] == pytest.approx(y.std())
    assert out["c"]["share_zero"] == pytest.approx(10 / 50) and out["c"]["share_one"] == pytest.approx(3 / 50)
    # an unknown unit gives a NaN scale: no finite sample, so no moment (never counted as centred)
    assert out["d"]["n"] == 0 and np.isnan(out["d"]["mean"]) and np.isnan(out["d"]["std"])


# ------------------------------------------------------------------------------- merk: synthetic tree
def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _tsv(path, header, rows):
    _write(path, "\t".join(header) + "\n" + "".join("\t".join(str(v) for v in r) + "\n" for r in rows))


def _run(raw, sub, ses, run, chans, data, json_duration, mni_rows):
    """chans: [(name, type)]; data (n_times, n_ch) in volts; mni_rows: [(name, x, y, z)] in metres."""
    base = f"{sub}_{ses}_task-force_run-{run}"
    res = 0.1                                     # 0.1 uV per stored unit, as pybv writes them
    (data / (res * 1e-6)).astype("<f4").tofile(os.path.join(raw, base + "_ieeg.eeg"))
    _write(os.path.join(raw, base + "_ieeg.vhdr"),
           "Brain Vision Data Exchange Header File Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
           f"DataFile={base}_ieeg.eeg\nMarkerFile={base}_ieeg.vmrk\nDataFormat=BINARY\n"
           f"DataOrientation=MULTIPLEXED\nNumberOfChannels={len(chans)}\nSamplingInterval=1000\n\n"
           "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n[Channel Infos]\n"
           + "".join(f"Ch{i + 1}={n},,{res},µV\n" for i, (n, _) in enumerate(chans)))
    _write(os.path.join(raw, base + "_ieeg.vmrk"),
           "Brain Vision Data Exchange Marker File, Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
           f"DataFile={base}_ieeg.eeg\n\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")
    _write(os.path.join(raw, base + "_ieeg.json"),
           json.dumps({"SamplingFrequency": 1000, "RecordingDuration": json_duration, "iEEGReference": "n/a"}))
    _tsv(os.path.join(raw, base + "_channels.tsv"), ["name", "type", "units", "status"],
         [(n, t, "µV", "good") for n, t in chans])
    _tsv(os.path.join(raw, base + "_space-mni_electrodes.tsv"), ["name", "x", "y", "z", "size"],
         [(n, x, y, z, "n/a") for n, x, y, z in mni_rows])
    _write(os.path.join(raw, f"{sub}_{ses}_space-mni_coordsystem.json"),
           json.dumps({"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "m"}))


def _merk_tree(tmp_path, bad_duration=False):
    root = tmp_path / "ma"
    raw = root / "datasets" / "inversion_testbed" / "merk_neumann_ecog_lfp" / "raw"
    raw.mkdir(parents=True)
    _tsv(raw / "participants.tsv", ["participant_id", "disease", "center", "center_ID",
                                    "Length_of_diagnosis_prior_to_surgery"],
         [("sub-000", "PD", "X", "11", "5"), ("sub-001", "PD", "X", "12", "7")])
    rng = np.random.default_rng(1)
    n = 1000
    z = rng.normal(size=n)
    z = (z - z.mean()) / z.std()
    w = rng.normal(size=n)
    w = 2.0 * (w - w.mean()) / w.std()            # centred, standard deviation 2
    clean = np.clip(z, 0, None) / np.clip(z, 0, None).max()
    stn = {"STN_RIGHT_0": (0.012, -0.013, -0.005), "STN_RIGHT_1": (0.0125, -0.0125, -0.003),
           "STN_RIGHT_2": (0.013, -0.012, -0.001), "STN_RIGHT_3": (0.0135, -0.0115, 0.001)}
    ecog0 = {"ECOG_RIGHT_0": (0.040, -0.020, 0.060), "ECOG_RIGHT_1": (0.041, -0.010, 0.061)}
    chans0 = [("ECOG_RIGHT_0", "ECOG"), ("ECOG_RIGHT_1", "ECOG"), ("STN_RIGHT_0", "SEEG"), ("STN_RIGHT_1", "SEEG"),
              ("STN_RIGHT_2", "SEEG"), ("MOV_RIGHT", "MISC"), ("MOV_LEFT", "MISC"), ("MOV_RIGHT_CLEAN", "MISC"),
              ("MOV_LEFT_CLEAN", "MISC")]
    data0 = np.column_stack([rng.normal(size=n) * 1e-5 for _ in range(5)] + [w, z, clean, clean])
    mni0 = [(k, *v) for k, v in ecog0.items()] + [(k, *stn[k]) for k in ("STN_RIGHT_0", "STN_RIGHT_1", "STN_RIGHT_2")] \
        + [(m, "n/a", "n/a", "n/a") for m in ("MOV_RIGHT", "MOV_LEFT", "MOV_RIGHT_CLEAN", "MOV_LEFT_CLEAN")]
    _run(str(raw), "sub-000", "ses-right", 0, chans0, data0, (n - 1) / 1000, mni0)
    _tsv(raw / "sub-000_electrodes.tsv", ["name", "x", "y", "z"],
         [(k, v[0] * 1000, v[1] * 1000, v[2] * 1000) for k, v in {**ecog0, **stn}.items()])
    _write(raw / "sub-000_coordsystem.json", json.dumps({"iEEGCoordinateSystem": "MNI", "iEEGCoordinateUnits": "mm"}))
    ecog1 = {"ECOG_LEFT_0": (-0.040, -0.020, 0.060)}
    chans1 = [("ECOG_LEFT_0", "ECOG"), ("MOV_LEFT", "MISC"), ("MOV_RIGHT", "MISC")]
    data1 = np.column_stack([rng.normal(size=n) * 1e-5, z, z[::-1]])
    _run(str(raw), "sub-001", "ses-left", 2, chans1, data1, 5.0 if bad_duration else (n - 1) / 1000,
         [(k, *v) for k, v in ecog1.items()] + [("MOV_LEFT", "n/a", "n/a", "n/a"), ("MOV_RIGHT", "n/a", "n/a", "n/a")])
    _tsv(raw / "sub-001_electrodes.tsv", ["name", "x", "y", "z"], [(k, v[0] * 1000, v[1] * 1000, v[2] * 1000)
                                                                   for k, v in ecog1.items()])
    _write(raw / "sub-001_coordsystem.json", json.dumps({"iEEGCoordinateSystem": "MNI", "iEEGCoordinateUnits": "mm"}))
    return root


def _run_merk(tmp_path, monkeypatch, root):
    out, pmap = tmp_path / "out", tmp_path / "map.csv"
    monkeypatch.setattr(MK, "below_normal", lambda: None)
    monkeypatch.setattr(MK, "SLOW", str(tmp_path / "no_slow"))
    monkeypatch.setattr(sys, "argv", ["probe_merk.py", "--moveatlas", str(root), "--out", str(out),
                                      "--person-map", str(pmap)])
    MK.main()
    with open(out / "facts.json", encoding="utf-8") as f:
        facts = json.load(f)
    checks = {r["check_id"]: r for r in read_csv_rows(out / "addendum_checks.csv")}
    return out, pmap, facts, checks


def test_merk_main_on_synthetic_tree(tmp_path, monkeypatch, stub_stamp):
    pytest.importorskip("mne")
    out, pmap, facts, checks = _run_merk(tmp_path, monkeypatch, _merk_tree(tmp_path))
    assert (facts["runs"], facts["stn_runs"], facts["stn_subjects"], facts["ecog_runs"]) == (2, 1, 1, 2)
    assert facts["channel_totals"] == {"ECOG": 3, "SEEG": 3, "MISC": 6}
    mn = facts["mni"]
    assert (mn["stn_runs_with_mni"], mn["seeg_rows_finite"], mn["ecog_rows_finite"]) == (1, 3, 3)
    assert mn["max_abs_diff_run_vs_subject_mm"] == 0.0 and mn["finite_run_rows_unmatched"] == 0
    assert mn["runs_session_units_unknown"] == 0 and mn["runs_with_rows_without_diff"] == 0
    assert mn["stn_rows_equal_subject_contacts_pattern"] == {"0; 1; 2": 1} and mn["subject_lead_contacts"] == [4]
    stn = read_csv_rows(out / "stn_mni.csv")[0]
    assert float(stn["centroid_absx_mm"]) == pytest.approx(12.5, abs=1e-3)
    assert float(stn["centroid_y_mm"]) == pytest.approx(-12.5, abs=1e-3)
    sm = facts["samples"]
    assert (sm["sample_counts_match"], sm["vhdr_datapoints_declared"], sm["stn_runs_sample_counts_match"]) == (2, 0, 1)
    assert checks["MRK-11"]["verdict"] == "confirmed" and "DataPoints" in checks["MRK-11"]["measured"]
    assert checks["MRK-08"]["verdict"] == "corrected"
    fm = facts["force_main"]
    # run 0: MOV_LEFT z-scored, MOV_RIGHT centred with sd 2; run 1: both z-scored
    assert (fm["runs_channels"], fm["centred"], fm["unit_variance"], fm["zscore_consistent"]) == (4, 4, 3, 3)
    assert fm["std_max"] == pytest.approx(2.0, rel=1e-4)
    fc = facts["force_clean"]
    assert (fc["runs_channels"], fc["max_equal_one"], fc["min_equal_zero"]) == (2, 2, 2)
    rows = read_csv_rows(pmap)
    assert list(rows[0].keys()) == ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"]
    assert [r["subject_ns"] for r in rows] == ["merk_neumann_ecog_lfp:sub-000", "merk_neumann_ecog_lfp:sub-001"]
    assert all(r["ambiguity"] == "" for r in rows)
    with open(out / "summary.md", encoding="utf-8") as f:
        text = f.read()
    assert chr(0x2014) not in text and "The MNI fact" in text


def test_merk_wrong_sidecar_duration_fails_the_run(tmp_path, monkeypatch, stub_stamp):
    pytest.importorskip("mne")
    out, _, facts, checks = _run_merk(tmp_path, monkeypatch, _merk_tree(tmp_path, bad_duration=True))
    assert facts["samples"]["json_duration_match"] == 1 and facts["samples"]["sample_counts_match"] == 1
    assert checks["MRK-11"]["verdict"] == "corrected"
    bad = [r for r in read_csv_rows(out / "samples_per_run.csv") if r["subject"] == "sub-001"][0]
    assert bad["json_duration_match"] == "False" and bad["sample_counts_match"] == "False"


# ------------------------------------------------------------------------------- ataxie
def test_ataxie_classes_and_unknown_labels(tmp_path, monkeypatch, stub_stamp):
    pytest.importorskip("openpyxl")
    root = tmp_path / "ma"
    raw = root / "datasets" / "pose" / "ataxie_kinect_dib" / "raw"
    raw.mkdir(parents=True)
    rows = [("P1", "PA", "FRDA"), ("P2", "PA", "frda "), ("P3", "PA", "sca2"), ("P4", "CA", "ADCK3"),
            ("P5", "CA", " Joubert S."), ("P6", "CA", "uca"), ("P7", "CA", "xyz"), ("P8", "CA", None),
            ("H1", "H", None), ("H2", "H", None)]
    d = pd.DataFrame(rows, columns=["Participant", "Group", "Diagnosis"])
    d["Sara"] = 1.0
    d["Speed [m/s]"] = 1.0
    with pd.ExcelWriter(raw / AX.XLSX, engine="openpyxl") as w:
        d.to_excel(w, sheet_name="Sheet1", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Sheet2", index=False)
    out = tmp_path / "out"
    monkeypatch.setattr(AX, "below_normal", lambda: None)
    monkeypatch.setattr(AX, "SLOW", str(tmp_path / "no_slow"))
    monkeypatch.setattr(sys, "argv", ["probe_ataxie_diagnoses.py", "--moveatlas", str(root), "--out", str(out)])
    AX.main()
    with open(out / "facts.json", encoding="utf-8") as f:
        F = json.load(f)
    assert F["participants"] == 10 and F["patients"] == 8 and F["groups"] == {"CA": 5, "H": 2, "PA": 3}
    assert F["genetic_labels"] == {"adck3": 1, "frda": 2, "joubert s.": 1, "sca2": 1}
    assert F["genetic_patients"] == 5 and F["gene_named_or_single_gene"] == 4
    assert F["labels_not_in_legend"] == ["xyz"] and F["patients_missing_label"] == 1
    assert F["patients_by_class"]["not_in_legend"] == 1 and F["patients_by_class"]["cause_undetermined"] == 1
    checks = {r["check_id"]: r["verdict"] for r in read_csv_rows(out / "addendum_checks.csv")}
    assert checks == {"ATX-01": "qualified", "ATX-02": "confirmed", "ATX-03": "corrected", "ATX-04": "confirmed"}
    counts = read_csv_rows(out / "diagnosis_counts.csv")
    assert {(r["group"], r["label"], r["class"], r["n"]) for r in counts if r["label"] == "xyz"} == \
        {("CA", "xyz", "not_in_legend", "1")}
