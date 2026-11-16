"""Tests of scripts/probe_ds004998.py on synthetic listings (no file of the ds004998 tree is read).

The split-part merge groups the FIF parts of one BIDS run, orders them by split number, and reports a
contiguous run only when the parts abut on the acquisition sample clock; gaps, overlaps, missing or repeated
parts, mixed names and header mismatches are named. A run-2 whose sample range lies inside its run-1 (the tail
copy found in ds004998) is not a separate acquisition, and units that share an acq_time cannot carry an order
null. The run events come from the first part's events.tsv when it covers the run.

The last test runs main() end to end on a small synthetic BIDS tree written with MNE in a temporary folder
(split parts written by MNE itself, a run-2 tail copy, an empty-room run, an absent participant): headers
only, no state column beside a segment quantity, no spectral column, and the person map schema.
"""
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import sys

import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("probe_ds004998_script",
                                                  os.path.join(TREE, "scripts", "probe_ds004998.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()
SUB = "sub-AAA111_ses-PeriOp"
MD = "1925-01-01T09:00:00+00:00"


def part(name, first, last, meas_date=MD, sfreq=2000.0, **kw):
    d = {"name": name, "first_samp": first, "last_samp": last, "meas_date": meas_date, "sfreq": sfreq}
    d.update(kw)
    return d


def by_id(runs):
    return {r["run_id"]: r for r in runs}


# ---------------------------------------------------------------------------------------------- names

def test_parse_bids_name_entities():
    e = S.parse_bids_name(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif")
    assert e == {"sub": "AAA111", "ses": "PeriOp", "task": "HoldL", "acq": "MedOff", "run": "1", "split": "02",
                 "suffix": "meg", "ext": "fif"}
    noise = S.parse_bids_name(f"/x/y/{SUB}_task-noise_meg.fif")
    assert noise["task"] == "noise" and noise["acq"] is None and noise["run"] is None and noise["split"] is None
    assert S.parse_bids_name(f"._{SUB}_task-HoldL_acq-MedOff_run-1_meg.fif") is None
    assert S.parse_bids_name("participants.tsv") is None
    assert S.run_id(S.parse_bids_name(f"{SUB}_task-MoveR_acq-MedOn_run-2_split-01_meg.fif")) == \
        f"{SUB}_task-MoveR_acq-MedOn_run-2"
    assert S.task_class("HoldR") == "hold" and S.task_class("MoveL") == "move" and S.task_class("Rest") == "rest"
    assert S.task_class("noise") is None


# ---------------------------------------------------------------------------------------------- split merge

def test_merge_two_contiguous_parts_listed_out_of_order():
    listing = [part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", 3000, 3999),
               part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 1000, 2999),
               {"name": f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_events.tsv"},
               {"name": f"._{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif"}]
    runs = S.merge_split_parts(listing)
    assert len(runs) == 1
    r = runs[0]
    assert r["run_id"] == f"{SUB}_task-HoldL_acq-MedOff_run-1"
    assert r["parts"] == [f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif",
                          f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif"]
    assert r["continuity"] == "contiguous" and r["joins"] == [0] and r["problems"] == []
    assert (r["first_samp"], r["last_samp"]) == (1000, 3999)
    assert r["n_samples"] == 3000 and r["span_samples"] == 3000 and r["n_parts"] == 2


def test_merge_single_file_and_noise_run():
    runs = by_id(S.merge_split_parts([part(f"{SUB}_task-MoveL_acq-MedOn_run-1_meg.fif", 0, 999),
                                      part(f"{SUB}_task-noise_meg.fif", 50, 149)]))
    assert runs[f"{SUB}_task-MoveL_acq-MedOn_run-1"]["continuity"] == "single"
    assert runs[f"{SUB}_task-MoveL_acq-MedOn_run-1"]["n_samples"] == 1000
    noise = runs[f"{SUB}_task-noise"]
    assert noise["continuity"] == "single" and noise["acq"] == "" and noise["run"] == ""


@pytest.mark.parametrize("second_first, status, join", [(3005, "gap", 5), (2990, "overlap", -10)])
def test_merge_gap_and_overlap_are_named(second_first, status, join):
    runs = S.merge_split_parts([part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 1000, 2999),
                                part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", second_first, 3999)])
    r = runs[0]
    assert r["continuity"] == status and r["joins"] == [join]
    assert r["problems"] and str(abs(join)) in r["problems"][0]
    assert r["n_samples"] == 2000 + (3999 - second_first + 1)
    assert r["span_samples"] == 3000


def test_merge_missing_repeated_and_mixed_parts():
    missing = S.merge_split_parts([part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99),
                                   part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-03_meg.fif", 200, 299)])[0]
    assert missing["continuity"] == "missing_part" and "missing split 02" in missing["problems"][0]
    only_second = S.merge_split_parts([part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", 100, 199)])[0]
    assert only_second["continuity"] == "missing_part" and "01" in only_second["problems"][0]
    mixed = S.merge_split_parts([part(f"{SUB}_task-HoldL_acq-MedOff_run-1_meg.fif", 0, 99),
                                 part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99)])[0]
    assert mixed["continuity"] == "mixed"
    repeated = S.merge_split_parts([part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99),
                                    part(f"/other/{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99)])[0]
    assert repeated["continuity"] == "duplicate_part"


def test_merge_header_mismatch_and_no_header():
    mism = S.merge_split_parts([
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99),
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", 100, 199, meas_date="1925-01-01T10:00:00+00:00")])[0]
    assert mism["continuity"] == "header_mismatch"
    rate = S.merge_split_parts([
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99),
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", 100, 199, sfreq=1000.0)])[0]
    assert rate["continuity"] == "header_mismatch"
    bare = S.merge_split_parts([{"name": f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif"},
                                {"name": f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif"}])[0]
    assert bare["continuity"] == "no_header" and bare["n_parts"] == 2 and "first_samp" not in bare


def test_merge_sums_skips_of_the_parts():
    r = S.merge_split_parts([
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-01_meg.fif", 0, 99, skip_entries=1, skip_samples=10),
        part(f"{SUB}_task-HoldL_acq-MedOff_run-1_split-02_meg.fif", 100, 199, skip_entries=2, skip_samples=30)])[0]
    assert r["continuity"] == "contiguous"
    assert r["skip_entries"] == 3 and r["skip_samples"] == 40


# ---------------------------------------------------------------------------------------------- acquisitions

def test_tail_copy_is_not_a_separate_acquisition():
    listing = [part(f"{SUB}_task-HoldR_acq-MedOff_run-1_split-01_meg.fif", 1000, 2999),
               part(f"{SUB}_task-HoldR_acq-MedOff_run-1_split-02_meg.fif", 3000, 3499),
               part(f"{SUB}_task-HoldR_acq-MedOff_run-2_meg.fif", 2800, 3499),
               part(f"{SUB}_task-MoveR_acq-MedOff_run-1_meg.fif", 0, 999, meas_date="1925-01-01T09:30:00+00:00"),
               part(f"{SUB}_task-MoveR_acq-MedOff_run-2_meg.fif", 0, 499, meas_date="1925-01-01T10:30:00+00:00")]
    runs = S.merge_split_parts(listing)
    groups = S.group_acquisitions(runs)
    r = by_id(runs)
    run1, run2 = r[f"{SUB}_task-HoldR_acq-MedOff_run-1"], r[f"{SUB}_task-HoldR_acq-MedOff_run-2"]
    assert run1["relation_to_primary"] == "primary" and run1["separate_acquisition"]
    assert run2["relation_to_primary"] == "contained" and not run2["separate_acquisition"]
    assert run2["primary_run_id"] == run1["run_id"] and run2["acquisition_id"] == run1["acquisition_id"]
    assert groups[run1["acquisition_id"]][0] is run1
    # a run-2 with its own meas_date is its own acquisition
    assert r[f"{SUB}_task-MoveR_acq-MedOff_run-2"]["relation_to_primary"] == "primary"
    assert len(groups) == 3


def test_range_relation():
    assert S.range_relation(0, 9, 0, 9) == "identical"
    assert S.range_relation(5, 9, 0, 9) == "contained"
    assert S.range_relation(0, 9, 5, 7) == "contains"
    assert S.range_relation(10, 19, 0, 9) == "disjoint"
    assert S.range_relation(5, 19, 0, 9) == "overlapping"


# ---------------------------------------------------------------------------------------------- order

def U(state, task, start, end=None):
    return {"state": state, "task": task, "start": start, "end": start + 600 if end is None else end,
            "label": f"{state}:{task}:{start}"}


def test_order_census_blocks_and_interleave():
    ordered = S.order_census([U("MedOff", "HoldL", 0), U("MedOff", "MoveL", 1500),
                              U("MedOn", "HoldL", 5000), U("MedOn", "MoveL", 6000)])
    assert ordered["all_off_before_on"] and not ordered["interleaved"]
    assert ordered["gap_start_to_start_s"] == 3500 and ordered["gap_end_to_start_s"] == 5000 - 2100
    inter = S.order_census([U("MedOff", "HoldR", 0), U("MedOn", "HoldR", 2000), U("MedOff", "MoveR", 4000)])
    assert inter["interleaved"] and inter["off_after_first_on"] == ["MedOff:MoveR:4000"]
    one_state = S.order_census([U("MedOff", "Rest", 0)])
    assert one_state["all_off_before_on"] is None and one_state["gap_start_to_start_s"] is None


def test_order_null_needs_distinct_acq_time():
    # run-1 and its tail copy share an acq_time: counted as two runs, but not orderable
    units = [U("MedOff", "HoldL", 0), U("MedOff", "HoldL", 0), U("MedOff", "MoveL", 1500), U("MedOn", "HoldL", 5000)]
    for reading in ("class", "side"):
        f = S.order_null_feasibility(units, reading)
        assert f["feasible_counting_ties"] and not f["feasible"]
        assert f["any_task_two_or_more"] and f["any_task_two_or_more_counting_ties"]
    side = S.order_null_feasibility([U("MedOff", "HoldL", 0), U("MedOff", "HoldR", 900)], "side")
    cls = S.order_null_feasibility([U("MedOff", "HoldL", 0), U("MedOff", "HoldR", 900)], "class")
    assert not side["feasible"] and cls["feasible"]
    assert cls["per_key"] == {"hold": {"n": 2, "n_distinct_start": 2}}
    medon_only = S.order_null_feasibility([U("MedOn", "MoveR", 0), U("MedOn", "MoveR", 900)], "side")
    assert not medon_only["feasible"] and medon_only["per_key"] == {}


# ---------------------------------------------------------------------------------------------- events

def test_run_events_first_part_covers_the_run():
    # part 1: 0..100 s, part 2: 100..150 s on the acquisition clock (offset 1000 s)
    bounds = [(1000.0, 1100.0), (1100.0, 1150.0)]
    first = [(0.0, 60.0, "rest"), (70.0, 70.0, "HoldL")]          # the hold reaches 40 s into part 2
    second_same = [(0.0, 40.0, "HoldL")]
    evs, src, comps = S.run_events([first, second_same], bounds)
    assert src == "first_part_covers_run"
    assert evs == [(1000.0, 1060.0, "rest"), (1070.0, 1140.0, "HoldL")]
    assert comps == [(1, 0, 0.0, 0.0, True)]
    second_other = [(0.0, 10.0, "HoldL"), (20.0, 15.0, "HoldL")]
    evs2, src2, comps2 = S.run_events([first, second_other], bounds)
    assert evs2 == evs and src2 == src
    k, unpaired, d_on, d_du, equal = comps2[0]
    assert k == 1 and unpaired == 1 and not equal and d_du == pytest.approx(30.0)


def test_run_events_joins_parts_when_the_first_does_not_reach():
    bounds = [(0.0, 100.0), (100.0, 200.0)]
    first = [(0.0, 50.0, "rest"), (80.0, 20.0, "MoveL")]          # ends exactly at the boundary
    second = [(0.0, 15.0, "MoveL"), (40.0, 10.0, "MoveL")]        # continuation, then a new segment
    evs, src, comps = S.run_events([first, second], bounds)
    assert src == "parts_joined" and comps == []
    assert evs == [(0.0, 50.0, "rest"), (80.0, 115.0, "MoveL"), (140.0, 150.0, "MoveL")]
    merged, n = S.merge_part_events([first, None], [0.0, 100.0])
    assert n == 0 and len(merged) == 2


def test_clip_and_compare_events():
    evs = [(0.0, 50.0, "rest"), (60.0, 130.0, "HoldR")]
    assert S.clip_events(evs, 100.0, 150.0) == [(0.0, 30.0, "HoldR")]
    assert S.events_equal([(0.0, 30.0, "HoldR")], [(0.004, 29.996, "HoldR")])
    assert not S.events_equal([(0.0, 30.0, "HoldR")], [(0.0, 30.0, "HoldL")])
    assert S.events_diff([(1.0, 5.0, "MoveL")], [(1.5, 4.0, "MoveL"), (9.0, 1.0, "rest")]) == (1, 0.5, 1.0)
    assert S.union_intervals([(5, 8), (0, 2), (1, 3)]) == [(0, 3), (5, 8)]
    assert S.overlap_s([(0, 10), (5, 12)], [(8, 20)]) == 4


def test_blind_key_hides_the_state():
    ev_a = [(0.0, 300.0, "rest"), (310.0, 370.0, "HoldL")]
    ev_b = [(0.0, 299.5, "rest"), (305.0, 360.0, "HoldL")]
    a = S.blind_key("AAA111", "HoldL", ev_a)
    b = S.blind_key("AAA111", "HoldL", ev_b)
    assert a != b and len(a) == 10 and "Med" not in a
    assert a == S.blind_key("AAA111", "HoldL", list(reversed(ev_a)))      # order of the events does not matter
    # the key is not a function of the run name: no run name of the subject and task gives it back
    for state in ("MedOff", "MedOn"):
        for run in ("1", "2"):
            name = f"{SUB}_task-HoldL_acq-{state}_run-{run}"
            for salt in (S.BLIND_SALT, "ds004998-segments-v1|"):
                assert hashlib.sha256((salt + name).encode("utf-8")).hexdigest()[:10] not in (a, b)


# ---------------------------------------------------------------------------------------------- main, end to end

SF = 2000.0
CH = [("MEG0111", "mag"), ("EEG001", "eeg"), ("EEG002", "eeg"), ("EMG061", "emg"), ("STI101", "stim")]
T9 = datetime.datetime(2020, 1, 1, 9, 0, 0, tzinfo=datetime.timezone.utc)


def _at(minutes):
    return T9 + datetime.timedelta(minutes=minutes)


def _tsv(path, cols, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _fif(meg_dir, stem, first_samp, n, meas_date, run_events, split_size=None):
    """Write <stem>_meg.fif (split by MNE when split_size is given), its meg.json and channels.tsv, with
    run_events (onset from the first sample, duration, label) stored as FIF annotations. Returns the file
    names written."""
    import mne
    import numpy as np
    info = mne.create_info([c for c, _ in CH], SF, [t for _, t in CH])
    data = np.random.default_rng(first_samp).standard_normal((len(CH), n)) * 1e-6
    data[-1] = 0.0
    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose="ERROR")
    raw.set_meas_date(meas_date)
    t0 = first_samp / SF
    raw.set_annotations(mne.Annotations([t0 + o for o, d, l in run_events], [d for o, d, l in run_events],
                                        [l for o, d, l in run_events], orig_time=meas_date))
    kw = {"split_size": split_size, "split_naming": "bids"} if split_size else {}
    raw.save(os.path.join(meg_dir, stem + "_meg.fif"), overwrite=True, verbose="ERROR", **kw)
    with open(os.path.join(meg_dir, stem + "_meg.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"SamplingFrequency": SF, "RecordingDuration": (n - 1) / SF, "MEGChannelCount": 1,
                   "EEGChannelCount": 2, "EMGChannelCount": 1, "TriggerChannelCount": 1}, f)
    _tsv(os.path.join(meg_dir, stem + "_channels.tsv"), ["name", "type", "units", "description", "status"],
         [(c, t.upper(), "V", "ElectroMyoGram" if t == "emg" else "x", "good") for c, t in CH])
    return sorted(x for x in os.listdir(meg_dir) if x.startswith(stem + "_") and x.endswith("_meg.fif"))


def _events(meg_dir, name, evs):
    _tsv(os.path.join(meg_dir, name.replace("_meg.fif", "_events.tsv")), ["onset", "duration", "trial_type"],
         [(o, d, l) for o, d, l in evs])


def _synthetic_tree(root):
    """sub-AAA: HoldL MedOff run-1 in two MNE split parts with its run-2 tail copy, MoveL MedOff, HoldL and
    MoveL MedOn, an empty-room run; sub-BBB: Rest MedOff and MedOn; sub-CCC listed in participants.tsv only."""
    raw_root = os.path.join(root, *S.RAW_REL.split("/"))
    os.makedirs(raw_root)
    _tsv(os.path.join(raw_root, "participants.tsv"), ["participant_id", "age", "sex", "diseaseDuration"],
         [("sub-AAA", 60, "M", 10), ("sub-BBB", 55, "F", 7), ("sub-CCC", 55, "F", 7)])
    scans = {"AAA": [], "BBB": []}
    for sub in scans:
        mdir = os.path.join(raw_root, f"sub-{sub}", "ses-PeriOp", "montage")
        os.makedirs(mdir)
        _tsv(os.path.join(mdir, f"sub-{sub}_ses-PeriOp_montage.tsv"),
             ["right_contacts_old", "right_contacts_new", "left_contacts_old", "left_contacts_new"],
             [("EEG001", "LFP-right-0", "EEG002", "LFP-left-0")])
        os.makedirs(os.path.join(raw_root, f"sub-{sub}", "ses-PeriOp", "meg"))

    def rec(sub, ent, first, n, minutes, evs, split=None, part_events=None):
        meg = os.path.join(raw_root, f"sub-{sub}", "ses-PeriOp", "meg")
        names = _fif(meg, f"sub-{sub}_ses-PeriOp_{ent}", first, n, _at(minutes), evs, split)
        for k, name in enumerate(names):
            _events(meg, name, part_events[k] if part_events else evs)
            scans[sub].append((f"meg/{name}", _at(minutes).strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
        return names

    run_evs = [(0.0, 20.0, "rest"), (22.0, 13.0, "HoldL"), (30.0, 1.0, "bad_lfp")]
    split = rec("AAA", "task-HoldL_acq-MedOff_run-1", 10000, 80000, 0, run_evs, "2MB",
                part_events=[run_evs, [(0.0, 4.0, "HoldL")]])
    rec("AAA", "task-HoldL_acq-MedOff_run-2", 70000, 20000, 0, [(0.0, 5.0, "HoldL"), (0.0, 1.0, "bad_lfp")])
    rec("AAA", "task-MoveL_acq-MedOff_run-1", 5000, 20000, 20, [(0.0, 5.0, "rest"), (6.0, 3.0, "MoveL")])
    rec("AAA", "task-HoldL_acq-MedOn_run-1", 7000, 20000, 60, [(0.0, 5.0, "rest"), (6.0, 3.5, "HoldL")])
    rec("AAA", "task-MoveL_acq-MedOn_run-1", 3000, 20000, 80, [(0.0, 4.0, "rest"), (5.0, 3.0, "MoveL")])
    meg = os.path.join(raw_root, "sub-AAA", "ses-PeriOp", "meg")
    for name in _fif(meg, "sub-AAA_ses-PeriOp_task-noise", 0, 4000, _at(-10), []):
        scans["AAA"].append((f"meg/{name}", _at(-10).strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
    rec("BBB", "task-Rest_acq-MedOff_run-1", 1000, 20000, 0, [(0.0, 10.0, "rest")])
    rec("BBB", "task-Rest_acq-MedOn_run-1", 1000, 20000, 40, [(0.0, 9.0, "rest")])
    for sub, rows in scans.items():
        _tsv(os.path.join(raw_root, f"sub-{sub}", "ses-PeriOp", f"sub-{sub}_ses-PeriOp_scans.tsv"),
             ["filename", "acq_time"], rows)
    return split


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_main_on_a_synthetic_tree(tmp_path, monkeypatch):
    pytest.importorskip("mne")
    split = _synthetic_tree(str(tmp_path / "ma"))
    assert split == ["sub-AAA_ses-PeriOp_task-HoldL_acq-MedOff_run-1_split-01_meg.fif",
                     "sub-AAA_ses-PeriOp_task-HoldL_acq-MedOff_run-1_split-02_meg.fif"]
    out, pmap = tmp_path / "out", tmp_path / "pm" / "map_test.csv"
    monkeypatch.setattr(S, "SLOW_FLAG", str(tmp_path / "no_slow_flag"))
    monkeypatch.setattr(S, "set_below_normal", lambda: "unchanged (test)")
    monkeypatch.setattr(S, "stamp_provenance", lambda root: {"moveatlas_stamp": {"head": "synthetic"}})
    monkeypatch.setattr(sys, "argv", ["probe_ds004998.py", "--moveatlas", str(tmp_path / "ma"), "--out", str(out),
                                      "--person-map", str(pmap)])
    S.main()

    runs = {r["run_id"]: r for r in _rows(out / "runs.csv")}
    assert len(runs) == 8
    r1 = runs["sub-AAA_ses-PeriOp_task-HoldL_acq-MedOff_run-1"]
    assert (r1["n_parts"], r1["continuity"], r1["joins"], r1["skip_entries"]) == ("2", "contiguous", "0", "0")
    assert (r1["n_samples"], r1["samples_minus_json"], r1["chain_ok"], r1["g15_header_verdict"]) == \
        ("80000", "0", "yes", "pass")
    assert r1["events_source"] == "first_part_covers_run" and r1["parts_fif_equal_run_events"] == "2"
    assert (r1["later_part_tsv_compared"], r1["later_part_tsv_equal"]) == ("1", "0")
    copy = runs["sub-AAA_ses-PeriOp_task-HoldL_acq-MedOff_run-2"]
    assert (copy["relation_to_primary"], copy["separate_acquisition"], copy["ends_with_primary"]) == \
        ("contained", "no", "yes")
    assert copy["offset_in_primary_samples"] == "60000" and copy["primary_run_id"] == r1["run_id"]
    assert all(r["g15_header_verdict"] == "pass" for r in runs.values())

    st = {(r["acq"], r["task"]): r for r in _rows(out / "state_task_counts.csv")}
    assert (st[("MedOff", "HoldL")]["bids_runs"], st[("MedOff", "HoldL")]["acquisitions"]) == ("2", "1")
    oc = {r["sub"]: r for r in _rows(out / "order_census.csv")}
    assert oc["AAA"]["all_medoff_before_medon"] == "yes" and float(oc["AAA"]["gap_acq_time_last_off_to_first_on_min"]) == 40.0
    assert oc["AAA"]["acq_time_ties"] == "1" and oc["BBB"]["n_acquisitions"] == "2"
    fz = {(r["sub"], r["unit"], r["reading"]): r for r in _rows(out / "order_null_feasibility.csv")}
    assert fz[("AAA", "bids_runs", "side")]["feasible_counting_ties"] == "yes"
    assert fz[("AAA", "bids_runs", "side")]["feasible"] == "no"
    assert fz[("AAA", "acquisitions", "class")]["feasible_counting_ties"] == "no"

    # segments: one row per recording, no column that names the state, the run or the parts, and a key that no
    # run name gives back
    with open(out / "segments.csv", encoding="utf-8") as f:
        seg_cols = f.readline().strip().split(",")
    seg = _rows(out / "segments.csv")
    assert len(seg) == 6 and seg_cols[:3] == ["sub", "task", "blind_key"]
    assert not {"acq", "state", "run", "run_id", "parts", "split"} & set(seg_cols)
    assert not any("Med" in v for r in seg for v in r.values())
    keys = {r["blind_key"] for r in seg}
    assert len(keys) == 6
    for rid in runs:
        for salt in (S.BLIND_SALT, "ds004998-segments-v1|"):
            assert hashlib.sha256((salt + rid).encode("utf-8")).hexdigest()[:10] not in keys
    assert {r["task"]: r for r in _rows(out / "segments_by_task.csv")}["HoldL"]["acquisitions"] == "2"

    # no spectral column in any output table; no state column beside anything but run listings and counts
    spectral = ("psd", "spectr", "power", "beta", "gamma", "theta", "alpha", "coheren", "band", "hz")
    for name in os.listdir(out):
        if name.endswith(".csv"):
            with open(out / name, encoding="utf-8") as f:
                head = f.readline().lower()
            assert not any(w in head for w in spectral), name
            if name in ("segments.csv", "segments_by_task.csv", "event_labels.csv"):
                assert "acq" not in head.split(","), name
            else:   # a table with the run name carries no event count, which would join the blind rows to it
                assert not {"n_events", "n_annotations_in_fif", "n_fif_annotations", "n_rest",
                            "n_task_segments"} & set(head.strip().split(",")), name

    pm = _rows(pmap)
    with open(pmap, encoding="utf-8") as f:
        assert f.readline().strip() == "source,subject_ns,person_id,rule,evidence,ambiguity"
    assert [r["person_id"] for r in pm] == ["pd_meg_stn_lfp_ds004998:sub-AAA", "pd_meg_stn_lfp_ds004998:sub-BBB"]
    assert all(r["subject_ns"] == r["person_id"] and r["source"] == S.SOURCE for r in pm)
    assert "sub-CCC" in pm[1]["ambiguity"] and "absent on disk" in pm[1]["ambiguity"]
    subs = {r["participant_id"]: r for r in _rows(out / "subjects.csv")}
    assert subs["sub-CCC"]["on_disk"] == "no" and subs["sub-CCC"]["in_participants_tsv"] == "yes"
    er = _rows(out / "empty_room.csv")
    assert len(er) == 1 and er[0]["sub"] == "AAA" and er[0]["position"].startswith("before the first task run")

    checks = _rows(out / "addendum_checks.csv")
    with open(out / "addendum_checks.csv", encoding="utf-8") as f:
        assert f.readline().strip() == "item,addendum_statement,measured,verdict,output"
    assert len(checks) >= 15 and all(c["verdict"] and c["measured"] for c in checks)
    with open(out / "_provenance.json", encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["headers_only"] is True and prov["preload"] is False and prov["fif_files"] == 9
    assert prov["moveatlas_stamp"] == {"head": "synthetic"} and prov["person_map"].endswith("pm/map_test.csv")
    with open(out / "summary.md", encoding="utf-8") as f:
        summ = f.read()
    assert chr(0x2014) not in summ and "## 9. Addendum checks" in summ
