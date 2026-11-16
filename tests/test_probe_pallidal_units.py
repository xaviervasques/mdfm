"""Tests of scripts/probe_pallidal_units.py (task T12) on synthetic inputs; no MoveAtlas file is read.

They check that only identifier columns are read (usecols and the frame assertion), that the feature
values of a synthetic file never reach the frame, the patient key when a label occurs under two genes, the
per-gene and per-patient counts, the parse of the addendum 1.5 line, and one end-to-end run with the
MoveAtlas stamp replaced.
"""
import csv
import importlib.util
import os

import pytest

pd = pytest.importorskip("pandas")

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(TREE, "scripts", "probe_pallidal_units.py")


def _script():
    spec = importlib.util.spec_from_file_location("probe_pallidal_units_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()

HEADER = ["patient", "gene", "hemisphere", "trajectory", "depth", "spike_pattern", "firing_rate", "is_bursting"]
ROWS = [
    ["p1", "TOR1A", "left", "central", "-4", "tonic", "SECRET1", "1"],
    ["p1", "TOR1A", "right", "anterior", "-2.5", "burst", "SECRET2", "0"],
    ["p2", "TOR1A", "left", "central", "-6", "tonic", "SECRET3", "1"],
    ["p3", "SGCE", "left", "medial", "1", "pause", "SECRET4", "0"],
    ["p3", "SGCE", "left", "medial", "0.5", "pause", "SECRET5", "0"],
    ["p3", "SGCE", "left", "medial", "0", "pause", "SECRET6", "0"],
]


def _write(path, header=HEADER, rows=ROWS):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_usecols_identifier_only():
    use = S.identifier_usecols(HEADER)
    assert use == ["patient", "gene", "hemisphere", "trajectory", "depth"]
    assert set(use) <= set(S.IDENTIFIER_COLUMNS)


def test_read_identifiers_never_holds_feature_values(tmp_path):
    p = _write(tmp_path / "units.csv")
    header, use, df = S.read_identifiers(p)
    assert header == HEADER
    assert list(df.columns) == ["patient", "gene", "hemisphere", "trajectory", "depth"]
    flat = df.astype(str).values.ravel().tolist()
    assert not any(v.startswith("SECRET") for v in flat)
    assert "tonic" not in flat


def test_frame_assertion_rejects_a_feature_column():
    df = pd.DataFrame({"patient": ["p1"], "firing_rate": ["3"]})
    with pytest.raises(AssertionError):
        S.assert_identifier_frame(df, HEADER)


def test_source_reads_with_usecols_and_asserts():
    with open(SCRIPT, encoding="utf-8") as f:
        src = f.read()
    assert "usecols=use" in src
    assert "assert_identifier_frame(df, header)" in src
    assert "\u2014" not in src and "\u2013" not in src


def test_counts_and_where_columns(tmp_path):
    _, _, df = S.read_identifiers(_write(tmp_path / "units.csv"))
    key, colliding = S.patient_keys(df)
    assert colliding == []
    genes = {r["gene"]: r for r in S.per_gene(df, key)}
    assert genes["TOR1A"]["n_patients"] == 2 and genes["TOR1A"]["n_units"] == 3
    assert genes["SGCE"]["n_patients"] == 1 and genes["SGCE"]["n_units"] == 3
    assert genes["TOR1A"]["units_per_patient_min"] == 1 and genes["TOR1A"]["units_per_patient_max"] == 2
    pats = {r["patient"]: r for r in S.per_patient(df, key)}
    assert pats["p1"]["n_hemispheres"] == 2 and pats["p1"]["trajectories"] == "anterior central"
    assert pats["p3"]["depth_min"] == "0" and pats["p3"]["depth_max"] == "1"
    idv = S.identifier_values(df)
    assert {"column": "hemisphere", "value": "left", "n_units": 5} in idv


def test_patient_label_collision_uses_gene_key(tmp_path):
    rows = ROWS + [["p1", "SGCE", "left", "central", "-3", "tonic", "SECRET7", "1"]]
    _, _, df = S.read_identifiers(_write(tmp_path / "units.csv", rows=rows))
    key, colliding = S.patient_keys(df)
    assert colliding == ["p1"]
    assert set(key) == {"TOR1A/p1", "TOR1A/p2", "SGCE/p3", "SGCE/p1"}


def test_parse_addendum(tmp_path):
    text = ("| pallidal_units_dystonia | published | CC BY 4.0 | table | x | table only | 1,714 GPi units from 32 "
            "patients as 38 unit-feature columns plus patient, gene, hemisphere, trajectory, depth; no age (S) | AUX |\n"
            "\n"
            "- pallidal_units_dystonia: 1,714 units, 32 patients; patients per gene TOR1A 9, SGCE 5; units per gene "
            "TOR1A 434, SGCE 384 (S).\n")
    p = tmp_path / "addendum.md"
    p.write_text(text, encoding="utf-8")
    ref = S.parse_addendum(str(p))
    assert ref["found_1_5"] and ref["found_table"]
    assert ref["units"] == 1714 and ref["patients"] == 32
    assert ref["patients_per_gene"] == {"TOR1A": 9, "SGCE": 5}
    assert ref["units_per_gene"] == {"TOR1A": 434, "SGCE": 384}
    assert ref["n_features"] == 38
    assert ref["identifier_names"] == ["patient", "gene", "hemisphere", "trajectory", "depth"]


def test_absent_covariates_uses_tokens():
    assert S.absent_covariates(["burst_average_spikes", "inter_burst_duration", "isi_mean"]) == []
    assert S.absent_covariates(["age_at_surgery", "centre"]) == ["age_at_surgery", "centre"]


def test_run_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "moveatlas"
    raw = root / os.path.dirname(S.CSV_REL)
    raw.mkdir(parents=True)
    p = _write(raw / os.path.basename(S.CSV_REL))
    sha = S.sha256_file(p)
    (root / S.SHA_REL).write_text(f"{sha}  {os.path.basename(S.CSV_REL)}  1\n", encoding="utf-8")
    monkeypatch.setattr(S, "stamp", lambda r: {"root": r, "head": "h" * 40, "diff_sha256": "d" * 64,
                                               "diff_bytes": 7, "time": "2099-01-01T00:00:00"})
    out = tmp_path / "out"
    r = S.run(str(root), str(out))
    assert r["n_units"] == len(ROWS) and r["n_patients"] == 3
    assert r["usecols"] == ["patient", "gene", "hemisphere", "trajectory", "depth"]
    for name in ("columns.csv", "patients_per_gene.csv", "units_per_patient.csv", "identifier_values.csv",
                 "addendum_diff.csv", "summary.md", "_provenance.json"):
        assert (out / name).is_file()
    cols = list(csv.DictReader(open(out / "columns.csv", encoding="utf-8")))
    assert [c["column"] for c in cols if c["values_read"] == "yes"] == r["usecols"]
    for name in ("columns.csv", "patients_per_gene.csv", "units_per_patient.csv", "identifier_values.csv",
                 "summary.md"):
        body = (out / name).read_text(encoding="utf-8")
        assert "SECRET" not in body and "tonic" not in body
        assert "\u2014" not in body and "\u2013" not in body
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "equal to" in summary and "gene or its surgical correlates" in summary
    assert f"MoveAtlas measured against: HEAD {'h' * 40}, uncommitted diff SHA-256 {'d' * 64} (7 bytes)" in summary
    assert "2099-01-01T00:00:00" not in summary          # the stamp time stays out: two runs give equal bytes
