"""scripts/probe_orientation_sources.py (weekend task ORI): the orientation-column classifier, the Xsens Euler
convention, the relative-heading discrepancy of two sensors on one rigid segment, the quiet-stance pairs of
sensors on different segments and their per-site decomposition, the phase timing of a training log, the
token growth of an orientation stream, the person gate that keeps lockbox files closed, and the admission
of canonical records through mdfm.data.canonical.check.

Synthetic only: no file of MoveAtlas, of the record index or of the reports of this tree is read; the raw-file
tests write their own small files in a pytest temporary folder.
"""
import importlib.util
import json
import os
import zipfile

import numpy as np
import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location("probe_orientation_sources_ori_test",
                                                  os.path.join(TREE, "scripts", "probe_orientation_sources.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load()


def test_classify_columns_weargait_and_others():
    head = ["Time", "Forehead_Acc_X", "Forehead_Mag_X", "Forehead_Mag_Y", "Forehead_Mag_Z", "Forehead_Roll",
            "Forehead_Pitch", "Forehead_Yaw", "Forehead_OriInc_q0", "Forehead_FreeAcc_E", "Pelvis course (deg)",
            "Pelvis pitch (deg)", "Pelvis roll (deg)", "Quat_W", "PacketCounter"]
    c = P.classify_columns(head)
    assert c["euler"] == ["Forehead_Pitch", "Forehead_Roll", "Forehead_Yaw"]
    assert c["magnetometer"] == ["Forehead_Mag_X", "Forehead_Mag_Y", "Forehead_Mag_Z"]
    assert c["orientation_increment"] == ["Forehead_OriInc_q0"]
    assert c["quaternion"] == ["Quat_W"]            # the OriInc column is not counted as a quaternion
    assert c["course_pitch_roll"] == ["Pelvis course (deg)", "Pelvis pitch (deg)", "Pelvis roll (deg)"]
    assert c["free_acc_earth"] == ["Forehead_FreeAcc_E"]
    assert P.classify_columns(["PacketCounter", "Acc_X", "Gyr_X"]) == {}


def test_weargait_sites_and_nonan_segments():
    head = [f"L_LatShank_{s}" for s in ("Roll", "Pitch", "Yaw")] + ["R_LatShank_Roll", "R_LatShank_Pitch"]
    assert P.weargait_sites(head, ("Roll", "Pitch", "Yaw")) == ["shank_L"]
    nh = ["time", "Pelvis course (deg)", "Pelvis pitch (deg)", "Pelvis roll (deg)", "Head course (deg)",
          "Head pitch (deg)", "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-x (deg/s)"]
    assert P.nonan_orientation_segments(nh) == ["Pelvis"]


def test_euler_convention_matches_the_up_vector_of_the_frames_probe():
    rng = np.random.default_rng(0)
    roll, pitch, yaw = rng.uniform(-60, 60, 50), rng.uniform(-60, 60, 50), rng.uniform(-180, 180, 50)
    R = P.euler_to_R(roll, pitch, yaw)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0)
    # earth up seen in the sensor frame = R^T e_z = (-sin p, sin r cos p, cos r cos p)
    # (scripts/probe_weargait_sensor_frames.py, up_from_euler)
    r, p = np.deg2rad(roll), np.deg2rad(pitch)
    up = np.stack([-np.sin(p), np.sin(r) * np.cos(p), np.cos(r) * np.cos(p)], -1)
    assert np.allclose(np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0])), up, atol=1e-12)
    # yaw alone is a rotation about earth z
    assert np.allclose(P.euler_to_R(0.0, 0.0, 30.0), P.Rz(np.deg2rad(30.0)), atol=1e-12)


def test_twist_about_z_is_exact_under_swing():
    for theta in (-2.5, -0.3, 0.0, 0.7, 3.0):
        D = P.Rz(theta)
        tw, sw = P.twist_about_z(D[None])
        assert np.isclose(tw[0], theta, atol=1e-12) and np.isclose(sw[0], 0.0, atol=1e-7)
    # a swing about a horizontal axis applied after the twist leaves the twist (swing-twist D = S T)
    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    D = rx(0.2) @ P.Rz(0.5)
    tw, sw = P.twist_about_z(D[None])
    assert np.isclose(tw[0], 0.5, atol=1e-12) and np.isclose(sw[0], 0.2, atol=1e-12)


def _rigid_pair(n=6000, fs=100.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    # a moving segment: smooth random Euler angles
    roll = 20 * np.sin(2 * np.pi * 0.9 * t) + rng.normal(0, 0.1, n)
    pitch = 35 * np.sin(2 * np.pi * 1.0 * t + 0.4)
    yaw = 40 * np.sin(2 * np.pi * 0.05 * t) + 10 * t
    Rs = P.euler_to_R(roll, pitch, yaw)
    C = P.euler_to_R(5.0, -12.0, 80.0)
    Ra = np.einsum("nij,jk->nik", Rs, C)
    return t, Rs, Ra


def test_pair_discrepancy_zero_for_a_rigid_pair_and_common_heading_error():
    t, Rs, Ra = _rigid_pair()
    psi, tilt = P.pair_discrepancy(Rs, Ra)
    assert np.abs(psi).max() < 1e-9 and np.abs(tilt).max() < 1e-6
    # the same heading error on both sensors cancels
    E = P.Rz(np.deg2rad(3.0 * t / 60.0 + 2.0 * np.sin(t)))
    psi, tilt = P.pair_discrepancy(np.einsum("nij,njk->nik", E, Rs), np.einsum("nij,njk->nik", E, Ra))
    assert np.abs(psi).max() < 1e-9


def test_pair_discrepancy_recovers_a_heading_drift_of_one_sensor():
    t, Rs, Ra = _rigid_pair()
    drift = np.deg2rad(6.0 * t / 60.0)            # 6 deg/min on the second sensor only
    psi, tilt = P.pair_discrepancy(Rs, np.einsum("nij,njk->nik", P.Rz(drift), Ra))
    st = P.heading_stats(t, psi, tilt)
    assert np.isclose(st["psi_slope_deg_min"], 6.0, rtol=0.02)
    # C fitted on a drifting pair leaks part of the +-3 deg drift into tilt while the segment moves
    assert np.abs(np.rad2deg(tilt)).max() < 2.0
    assert st["n_chunks"] == 6
    assert np.isclose(st["chunk_lf_net_median_deg"], 1.0, atol=0.05)   # 6 deg/min over 10 s
    assert st["psi_hf_rms_deg"] < 0.3                     # the same leak, modulated by the segment motion


def test_heading_stats_separates_fast_noise_from_slow_wander():
    rng = np.random.default_rng(3)
    t = np.arange(3000) / 100.0
    psi = np.deg2rad(rng.normal(0.0, 2.0, len(t)))
    st = P.heading_stats(t, psi, np.zeros_like(t))
    assert 1.8 < st["psi_hf_rms_deg"] < 2.1
    assert st["chunk_lf_range_median_deg"] < 1.5          # white noise has little slow wander
    assert st["n_chunks"] == 3


def test_phase_hours_from_log_events():
    ev = [{"event": "start", "time": "2026-09-30T16:00:00"},
          {"event": "phase_end", "phase": "A", "time": "2026-09-30T20:30:00"},
          {"event": "start", "time": "2026-09-30T21:00:00"},
          {"event": "resume", "phase": "A", "time": "2026-09-30T21:00:05"},
          {"event": "start", "time": "2026-09-30T22:00:00"},
          {"event": "resume", "phase": "A", "time": "2026-09-30T22:00:00"},
          {"event": "phase_end", "phase": "B", "time": "2026-10-01T19:30:00"}]
    h = P.phase_hours(ev)
    assert np.isclose(h["phase_A_h"], 4.5) and np.isclose(h["phase_B_h"], 21.5)
    assert np.isclose(h["total_h"], 26.0) and np.isclose(h["phase_A_share"], 4.5 / 26.0)


def test_person_gate():
    g = P.PersonGate({"weargait:HC100", "weargait:NLS902 (control)", "weargait:HC902"}, {"weargait:HC902",
                                                                                      "weargait:NLS902"})
    assert g.ok("weargait:HC100")
    assert not g.ok("weargait:HC902")                        # lockbox
    assert not g.ok("weargait:NLS902 (control)")             # lockbox under its bare id
    assert not g.ok("weargait:HC999")                        # not in the index: never opened
    assert g.refused == {"weargait:HC902", "weargait:NLS902 (control)", "weargait:HC999"}


def test_raw_weargait_never_opens_a_refused_file(tmp_path, monkeypatch):
    d = tmp_path / "datasets" / "imu" / "weargait_pd" / "raw" / "Version 1" / "CONTROL PARTICIPANTS" / "CSV files"
    d.mkdir(parents=True)
    cols = ["Time"] + [f"{pre}_{s}" for pre in P.WEARGAIT_LOCMAP for s in
                       ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z", "Mag_X", "Mag_Y", "Mag_Z",
                        "Roll", "Pitch", "Yaw")]
    (d / "HC100_SelfPace.csv").write_text(",".join(cols) + "\n0 sec," + ",".join(["0"] * (len(cols) - 1)) + "\n",
                                          encoding="utf-8")
    (d / "HC902_SelfPace.csv").write_text("lockbox", encoding="utf-8")
    opened = []
    real = P.read_header_line

    def spy(path):
        opened.append(os.path.basename(path))
        return real(path)
    monkeypatch.setattr(P, "read_header_line", spy)
    gate = P.PersonGate({"weargait:HC100", "weargait:HC902"}, {"weargait:HC902"})
    trials, s = P.raw_weargait(str(tmp_path), gate, {("weargait:HC100", "SelfPace")})
    assert opened == ["HC100_SelfPace.csv"]
    assert len(trials) == 1 and trials[0]["euler_sites"] == 13 and trials[0]["euler_g1_six"] == 1
    assert trials[0]["euler_every_imu_site"] == 1 and trials[0]["mag_sites"] == 13
    assert s == {"files_listed": 2, "records_nonlockbox": 1, "records_with_euler_csv": 1, "records_without_csv": 0}


def test_raw_clinical_gait_counts_non_lockbox_trials(tmp_path):
    zp = tmp_path / "datasets" / "imu" / "clinical_gait_multipatho" / "raw"
    zp.mkdir(parents=True)
    with zipfile.ZipFile(zp / "dataset.zip", "w") as z:
        for subj, sess in (("HS_1", "1"), ("HS_2", "1")):
            pre = f"dataset/data/healthy/HS/{subj}/{subj}_{sess}/{subj}_{sess}_"
            for site in ("HE", "LB", "LF", "RF"):
                z.writestr(pre + f"raw_data_{site}.txt", "PacketCounter\tAcc_X\tGyr_X\tMag_X\tMag_Y\tMag_Z\n1\t0\t0\t0\t0\t0\n")
            z.writestr(pre + "processed_data.txt", "PacketCounter\tHE_FreeAcc_X\n1\t0\n")
            z.writestr(pre + "meta.json", json.dumps({"sensor": "MTw Awinda XSens"}))
    gate = P.PersonGate({"clinical_gait:HS_1", "clinical_gait:HS_2"}, {"clinical_gait:HS_2"})
    r = P.raw_clinical_gait(str(tmp_path), gate)
    assert r["trials"] == 1 and r["persons"] == 1
    assert r["raw_file_kinds"] == {"magnetometer": 4}
    assert r["processed_file_kinds"] == {"free_acc_earth": 1}
    assert r["sensor_field"] == {"MTw Awinda XSens": 1}
    assert r["trials_with_mag_every_raw_file"] == 1


def test_spearman_and_quant():
    rho, n = P.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert np.isclose(rho, 1.0) and n == 4
    assert P.quant([np.nan, 1.0, 2.0, 3.0], (50,)) == [2.0]
    assert P.quant([], (50,)) == [None]


def _labels():
    return (["unlabeled"] * 300 + ["EO_FeetTogether"] * 1200 + ["unlabeled"] * 200 + ["EC_FeetTogether"] * 1200
            + ["Standing"] * 140 + ["Walk"] * 500 + ["Standing"] * 600 + ["Turn"] * 100 + ["Standing"] * 90)


def test_quiet_bouts_trim_and_bout_pairs():
    b = P.quiet_bouts(_labels(), fs=100.0, trim_s=0.5)
    # the last Standing run (90 samples) is emptied by the 2 x 50 sample trim
    assert b == [("EO_FeetTogether", 350, 1450), ("EC_FeetTogether", 1750, 2850), ("Standing", 2950, 2990),
                 ("Standing", 3590, 4090)]
    # EO with the EC of the same stance; the first Standing bout lasts 0.4 s < 1 s: only one Standing bout left
    assert P.bout_pairs(b, fs=100.0, min_s=1.0) == [(0, 1)]
    assert P.bout_pairs(b, fs=100.0, min_s=0.3) == [(0, 1), (2, 3)]
    assert P.quiet_bouts(["Walk"] * 50) == [] and P.quiet_bouts([]) == []


def test_quiet_pair_stats_chunks_slow_and_fast_parts():
    rng = np.random.default_rng(5)
    a = np.linspace(0.0, 3.0, 1100)                      # a 3 deg ramp over an 11 s bout
    b = rng.normal(0.0, 1.0, 600)                        # 6 s of white noise, SD 1 deg
    st = P.quiet_pair_stats(np.r_[a, b], [(0, 1100), (1100, 1700)], fs=100.0)
    # chunks: 10 s of the ramp (its 1 s tail is not scored) and the 6 s noise bout
    assert st["n_chunks"] == 2
    assert np.isclose(st["bout_medians_deg"][0], 1.5, atol=0.01)
    # the ramp's slow range over its first 10 s is about 2.7 deg, the noise's slow part is small
    assert 1.4 < st["slow_range_deg"] < 1.7               # median of about 2.7 and about 0.4
    assert 0.5 < st["fast_rms_deg"] < 0.65               # about sqrt(600 / 1700) of the noise's fast RMS (about 0.99)


def _still_body(n, rng):
    """Euler triples of the eight sensors of a person standing still, a constant orientation each."""
    pref = sorted(set(P.CROSS_SITES.values()) | {p for ab in P.COLOC_PAIRS.values() for p in ab})
    return {p: np.tile([rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-180, 180)], (n, 1)) for p in pref}


def _with_heading(E, h_deg):
    R = np.einsum("nij,njk->nik", P.Rz(np.deg2rad(h_deg)), P.euler_to_R(E[:, 0], E[:, 1], E[:, 2]))
    roll = np.rad2deg(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    pitch = np.rad2deg(-np.arcsin(np.clip(R[:, 2, 0], -1, 1)))
    yaw = np.rad2deg(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return np.stack([roll, pitch, yaw], 1)


def test_cross_trial_common_heading_cancels_and_one_sensor_wander_shows():
    rng = np.random.default_rng(7)
    lab = np.array(_labels())
    n = len(lab)
    eul = _still_body(n, rng)
    common = 20.0 * np.sin(np.arange(n) / 300.0)          # one heading error on every sensor: cancels
    eul = {p: _with_heading(E, common) for p, E in eul.items()}
    rows = {r["pair"]: r for r in P.cross_trial(lab, eul)}
    assert len(rows) == 17
    for r in rows.values():
        assert r["slow_sd_deg"] < 1e-6 and r["fast_rms_deg"] < 1e-6 and r["between_abs_change_deg"] < 1e-6
    # a wander of the head sensor alone: every pair with the head sees it, no other pair does
    eul["Forehead"] = _with_heading(eul["Forehead"], 3.0 * np.sin(np.arange(n) / 150.0))
    rows = {r["pair"]: r for r in P.cross_trial(lab, eul)}
    for name, r in rows.items():
        if "head" in name.split("~"):
            assert r["slow_sd_deg"] > 0.5
        else:
            assert r["slow_sd_deg"] < 1e-6
    assert rows["head~lower_back"]["between_n"] == 1 and rows["head~lower_back"]["n_bouts"] == 4
    lev, res, k = P.site_decomposition({tuple(nm.split("~")): rows[nm]["slow_sd_deg"]
                                        for nm in rows if rows[nm]["kind"] == "cross"}, tuple(P.CROSS_SITES))
    assert k == 15 and res < 1e-6
    assert lev["head"] > 0.5 and max(v for s, v in lev.items() if s != "head") < 1e-3


def test_site_decomposition_recovers_independent_levels():
    sd = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 3.0}
    s = list(sd)
    levels = {(x, y): np.sqrt(sd[x] ** 2 + sd[y] ** 2) for i, x in enumerate(s) for y in s[i + 1:]}
    lev, res, k = P.site_decomposition(levels, s)
    assert k == 6 and res < 1e-9 and all(np.isclose(lev[x], sd[x]) for x in s)
    # a common error shared by a and b breaks additivity: the misfit shows in the residual
    levels[("a", "b")] = 0.1
    assert P.site_decomposition(levels, s)[1] > 0.05
    # too few pairs: no fit
    assert np.isnan(P.site_decomposition({("a", "b"): 1.0}, s)[1])


def test_token_growth_and_mixture_reader(tmp_path):
    fam, lo, hi = P.token_growth({"six": (527.0, 6.0), "cam": (82.0, 0.0)}, {"six": 0.5, "cam": 0.5}, ("six",), 40)
    assert fam["six"]["tokens_ori"] == 767.0 and fam["six"]["tokens_ori_upper"] == 773.0
    assert fam["cam"]["tokens_ori"] == 82.0
    assert np.isclose(lo, (767 + 82) / (527 + 82)) and np.isclose(hi, (773 + 82) / (527 + 82))
    cfg = tmp_path / "c.yaml"
    cfg.write_text("mixture:\n  rows: [128, 640, 1280]           # [B4.18] bucket rows\n  layouts:   # [cfg] x\n"
                   "    xsens13: 0.15\n    g1_six: 0.10      # comment\n  sources: [a, b]\n", encoding="utf-8")
    mix, rows = P.stage1_mixture(str(cfg))
    assert rows == [128, 640, 1280] and mix == {"xsens13": 0.15, "g1_six": 0.10}


def test_stage_records_admits_rows_through_canonical_check(tmp_path, monkeypatch):
    import pandas as pd
    from mdfm.data import canonical as C
    zp = tmp_path / "c.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("dummy", "x")
    idx = pd.DataFrame([{"source": "weargait_pd", "modality": "imu", "group": "control", "container": "c.zip",
                         "record": r, "person_id": f"p{r}", "subject_ns": f"weargait:p{r}", "lockbox": False,
                         "cody": False, "tier": "TRAIN"} for r in (0, 1)])
    read, checked = [], []

    def fake_check(row, allow_eval=False, allow_lockbox=False):
        checked.append(row["record"])
        if row["record"] == 1:
            raise C.Refused("refused by the test")
        return row

    def fake_npy(zf, name):
        read.append(name)
        return np.ones((1, 9), dtype=bool) if "channel_mask" in name else np.array([0])
    monkeypatch.setattr(C, "index", lambda: idx)
    monkeypatch.setattr(C, "check", fake_check)
    monkeypatch.setattr(C, "container_path", lambda rel: str(zp))
    monkeypatch.setattr(C, "read_header", lambda cont, zf=None: {"kind": "list", "records": [{}, {}]})
    monkeypatch.setattr(C, "record_meta", lambda header, r: {})
    monkeypatch.setattr(C, "member_prefix", lambda r: f"r{r}/")
    monkeypatch.setattr(C, "read_npy", fake_npy)
    monkeypatch.setattr(P, "stage1_sources", lambda *a: ["weargait_pd"])
    monkeypatch.setattr(P, "stage2_rows", lambda *a: [])
    res, _ = P.stage_records(str(tmp_path))
    assert sorted(checked) == [0, 1] and res["imu_rows_refused_by_check"] == 1
    assert read == ["r0/channel_mask.npy", "r0/location_ids.npy"]          # nothing of record 1 was read
    assert res["rows"][0]["mag_records_every_site"] == 1 and res["rows"][0]["gyro_records_every_site"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
