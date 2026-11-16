"""Stage-1 real sources (plan task B4.5; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 1.3 and 2.2):
what the pilot admits, and the facts its pairing rules rest on, measured from the files.

Every record is read through mdfm.data.canonical with its flags at False (lockbox, EVAL, CODY and
EXCLUDED_V1 refused), except the stroke plate arrays (processed/aux_force/, not in the index), which are
opened only for a force row the reader admitted. Writes reports/stage1_sources/:
- sources.csv: per source and stream, admitted rows, persons, recordings, hours (n_frames / fps of the
  index) and fps values, beside the same counts over every non-lockbox RELEASE row of the source
  (controls and PD, stroke patients included);
- content.csv: per source and stream, the channel content met in the full pass (sites and kept
  channels, acc_content as recorded, mag_unit, force sensor masks and unit, EMG muscles and units);
- pairs.csv: per source, recordings, paired, dropped by reason, 10 s windows (stride 10 s, 1 s minimum),
  tails dropped and their seconds; every window passes windows.validate;
- pose_checks.csv: 50 random admitted pose records per 3D pose source (seed 20260928): torso length in
  metres (median |neck - pelvis| of coords x scale), the same in torso units (the MoveAtlas normalization
  gives 1), the largest |pelvis| coordinate (records are root-centred), and the up axis (median neck
  minus mid-ankle direction) as recorded and after sources.pair;
- stroke_plates.csv: every non-lockbox stroke force record: the plates of the aux_force array whose |Fz|
  sum equals each foot total (to 1e-3 N plus 1e-5 relative), the unloaded offset of each foot (median of
  the samples within 20 N of its 5th percentile), minimum and peak, plates never below 20 N; the
  kinematic check (median height of the other ankle minus this ankle over the frames where this foot
  carries more than offset + max(20 N, 10 percent of its peak), from the canonical z-up pose: positive
  when the loaded foot is the lower one; verdict consistent above 1 cm, swapped below -1 cm); for the
  able-bodied subjects, the check against the dataset's own stride events (Data.Sub(i).events of
  MAT_normalizedData_AbleBodiedAdults: each stride with GoodForcePlateLanding 1 should show a loading
  onset, an upward crossing of offset + 20 N, of the same foot within 30 ms of ICstart, trial Trial
  mapped to C3D trial Trial - 1, and the hypothesis Trial is scored too);
- demographics.csv: per source, the demographic table, persons matched, sex, ages and masses; the stroke
  subject mapping (MAT Data.Sub(i) = S(i)) is also scored by the correlation between the Height of
  sub_char and the subject's median torso length (coords x scale) over the full pass, for the identity
  and for mappings shifted by 1 and 2;
- folds.csv, and the SHA-256 of each sorted person list in summary.json;
- tasks.csv: recordings per task of each pilot source, amass takes per task (index rows only);
- raw-file checks in summary.json: weargait_raw, 40 random admitted weargait recordings with IMU and
  insoles (seed 20260928) against their raw CSV (insole totals against LTotalForce and RTotalForce,
  the zero-gap mask of sources.pair against the samples where both raw totals are NaN, lower-back acc
  against LowerBack_Acc / 9.80665 at the same rows); gait120_trc, the step frames of the pairing rule
  (EMG gait_events) against the TRC headers (OrigDataStartFrame, NumFrames) of every admitted trial;
- summary.json, summary.md, _provenance.json.
Run: <conda>/envs/mdfm/python.exe scripts/probe_stage1_sources.py [--quick]
(--quick: 5 recordings per source, 5 pose records, 10 force records; for a dry run only).
"""
import argparse
import collections
import datetime as dt
import glob
import itertools
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm import paths  # noqa: E402
from mdfm.data import canonical as C  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from mdfm.data import windows as W  # noqa: E402
from _common import moveatlas_head  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SEED = 20260928
N_POSE = 50
TORSO_BAND = (0.35, 0.65)
KIN_CM = 0.01
ONSET_MS = 30.0
POSE_SOURCES = ("gait120", "stroke_gait_vancriekinge")
STROKE = "stroke_gait_vancriekinge"
AB_MAT = "pose/stroke_gait_vancriekinge/raw/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat"
N_RAW = 40
G0 = 9.80665
GAIT120_TRC = {"level_walking": "LevelWalking", "stair_ascent": "StairAscent", "stair_descent": "StairDescent",
               "slope_ascent": "SlopeAscent", "slope_descent": "SlopeDescent", "sit_to_stand": "SitToStand",
               "stand_to_sit": "StandToSit"}


def locations():
    import yaml
    with open(f"{paths.MOVEATLAS}/configs/canonical_imu.yaml", encoding="utf-8") as f:
        block = re.search(r"^locations:\s*\[[^\]]*\]", f.read(), re.M).group(0)
    return yaml.safe_load(block)["locations"]


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.percentile(x, p)) if len(x) else float("nan")


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


# ---------------------------------------------------------------- 1. rows and hours (index only)
def source_table(adm, idx):
    out = []
    base = idx[(idx["tier"] == "RELEASE") & ~idx["lockbox"] & (idx["container"] != "")]
    for source in S.PILOT:
        spec = S.SOURCES[source]
        for s in spec.streams:
            a = adm[(adm["source"] == source) & (adm["stream"] == s.kind)]
            b = base[(base["source"] == source) & (base["modality"] == s.modality)]
            out.append({"source": source, "stream": s.kind, "role": s.role,
                        "healthy_filter": "group in " + "|".join(spec.groups) if spec.groups else "all rows",
                        "rows": len(a), "persons": a["person_id"].nunique(),
                        "recordings": a[["subject_ns", "recording"]].drop_duplicates().shape[0],
                        "hours": round(float((a["n_frames"] / a["fps"]).sum() / 3600), 2),
                        "fps": "|".join(f"{v:g}" for v in sorted(a["fps"].unique())),
                        "all_nonlockbox_rows": len(b), "all_nonlockbox_persons": b["person_id"].nunique(),
                        "all_nonlockbox_hours": round(float((b["n_frames"] / b["fps"]).sum() / 3600), 2)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 2. full pass: pairing, content, windows
def signature(kind, rec, sites):
    if kind == "imu":
        cm = np.asarray(rec["channel_mask"], bool)
        ids = [int(i) for i in np.asarray(rec["location_ids"]).ravel()]
        chans = {tuple(r[:6]) for r in cm.tolist()}
        kept = "|".join(("acc" if c[0] else "") + ("+gyro" if c[3] else "") for c in sorted(chans))
        return {"sites": "|".join(sites[i] for i in ids), "channels_kept": kept,
                "magnetometer_in_record": bool(cm[:, 6:].any()),
                "acc_content": rec.get("acc_content") or "absent (contract default: with gravity)",
                "mag_unit": rec.get("mag_unit") or "", "sample_valid": "sample_valid" in rec}
    if kind == "pose3d":
        return {"dim": rec.get("dim"), "modality": rec.get("modality"),
                "joints_present": int(np.asarray(rec["joint_mask"], bool).sum())}
    if kind == "force":
        m = np.asarray(rec["sensor_mask"], bool)
        return {"sensor_mask": "|".join("".join("1" if b else "0" for b in row) for row in m.tolist()),
                "unit": rec.get("unit")}
    if kind == "emg":
        return {"muscles": "|".join(f"{c.get('muscle')}_{c.get('side')}" for c in rec.get("channels") or []),
                "unit": rec.get("unit"), "unit_scale": rec.get("unit_scale"), "signal_kind": rec.get("signal_kind")}
    return {}


def full_pass(quick, sites):
    pairs, content, extra = [], collections.Counter(), {}
    scales = collections.defaultdict(list)          # stroke subject -> torso length (m) of each paired trial
    for source in S.PILOT:
        t0 = time.time()
        st = collections.Counter()
        leaks, emg_p99, emg_flag, emg_n, ages, gaps, bdiff, win_s = 0, [], 0, 0, {}, 0, [], 0.0
        zg = collections.Counter()
        lead_s = []
        for pr in S.paired(source, stats=st, limit=5 if quick else None):
            if "force_zero_gap_samples" in pr.pairing:
                sv = np.asarray(pr.streams["force"]["sample_valid"], bool)[:, 0]
                zg["records"] += 1
                zg["samples"] += len(sv)
                zg["gap_samples"] += pr.pairing["force_zero_gap_samples"]
                zg["records_with_a_gap"] += int(pr.pairing["force_zero_gap_samples"] > 0)
                zg["records_starting_with_a_gap"] += int(not sv[0])
                lead_s.append((int(np.argmax(sv)) if sv.any() else len(sv)) / float(pr.streams["force"]["fps"]))
            if source == STROKE:
                scales[pr.subject_ns].append(float(pr.streams["pose3d"]["scale"]))
            for kind, rec in pr.streams.items():
                content[(source, kind, json.dumps(signature(kind, rec, sites), sort_keys=True))] += 1
                if kind == "emg":
                    emg_n += 1
                    emg_flag += "AMPLITUDE FLAG" in str(rec.get("notes") or "")
                    emg_p99.append(float(np.nanpercentile(np.abs(rec["signal"]), 99)))
            if source == "sixmwt_lifespan":
                ages[pr.subject_ns] = pr.streams["imu"].get("age")
            if source == "gait120":
                fr = np.asarray(pr.pairing["pose_frames"])
                gaps += int((np.diff(fr) > 1).any())
                bdiff += pr.pairing["boundary_diff_m"]
            for w in W.cut(pr, stats=st):
                W.validate(w)
                leaks += bool(W.forbidden_keys(w))
                win_s += w.duration
        row = {"source": source, "paired": st["paired"], "windows": st["windows"], "window_hours": round(win_s / 3600, 2),
               "tails_dropped": st["tails_dropped"], "tail_s_dropped": round(float(st["tail_s_dropped"]), 2),
               "records_short": st["records_short"], "stream_windows_empty": st["stream_windows_empty"],
               "windows_with_forbidden_keys": leaks}
        row.update({k: v for k, v in st.items() if k.startswith(("dropped_", "rows_dropped_"))})
        pairs.append(row)
        extra[source] = {"emg_records": emg_n, "emg_amplitude_flag": emg_flag,
                         "emg_p99_mV": [r4(q(emg_p99, 0)), r4(q(emg_p99, 50)), r4(q(emg_p99, 100))] if emg_p99 else None,
                         "sixmwt_ages_on_records": sum(a is not None and np.isfinite(a) for a in ages.values()) if ages else None,
                         "gait120_trials_with_a_step_gap": gaps if source == "gait120" else None,
                         "gait120_boundary_diff_m_max": r4(max(bdiff)) if bdiff else None,
                         "gait120_boundaries": len(bdiff) if source == "gait120" else None,
                         "seconds": round(time.time() - t0, 1)}
        if zg:
            extra[source]["force_zero_gaps"] = dict(zg, leading_gap_s_median_p90_max=[
                r4(q(lead_s, 50)), r4(q(lead_s, 90)), r4(q(lead_s, 100))])
        print(f"{source}: {dict(st)} in {time.time() - t0:.0f} s", flush=True)
    rows = [dict(zip(("source", "stream"), k[:2]), records=n, **json.loads(k[2])) for k, n in sorted(content.items())]
    return pd.DataFrame(pairs).fillna(0), pd.DataFrame(rows), extra, scales


# ---------------------------------------------------------------- 3. pose checks
def up_axis(u):
    i = int(np.argmax(np.abs(u)))
    return ("+" if u[i] > 0 else "-") + "xyz"[i]


def neck_minus_ankle(X, valid):
    ok = valid[:, 1] & valid[:, 13] & valid[:, 14]
    v = X[ok, 1] - 0.5 * (X[ok, 13] + X[ok, 14])
    return np.median(v, axis=0)


def pose_checks(adm, quick):
    out = []
    rng = np.random.default_rng(SEED)
    for source in POSE_SOURCES:
        rows = adm[(adm["source"] == source) & (adm["stream"] == "pose3d")].reset_index(drop=True)
        pick = rng.choice(len(rows), 5 if quick else N_POSE, replace=False)
        for row in rows.iloc[np.sort(pick)].to_dict("records"):
            rec = C.load(row)
            c = np.asarray(rec["coords"], dtype=np.float64)
            X = c * float(rec["scale"])
            valid = np.asarray(rec["valid_mask"], bool)
            both = valid[:, 0] & valid[:, 1]
            u = neck_minus_ankle(X, valid)
            after = u @ S.UP_ROTATION.T if S.SOURCES[source].pairing == "c3d_trial" else u
            out.append({"source": source, "subject_ns": row["subject_ns"], "key": row["key"], "n_frames": len(c),
                        "scale": r4(rec["scale"]),
                        "torso_m": r4(np.median(np.linalg.norm(X[both, 1] - X[both, 0], axis=1))),
                        "torso_units": r4(np.median(np.linalg.norm(c[both, 1] - c[both, 0], axis=1))),
                        "pelvis_abs_max": float(np.abs(c[:, 0]).max()),
                        "up_recorded": up_axis(u), "up_after_pair": up_axis(after),
                        "neck_minus_ankle_m": "|".join(f"{v:.3f}" for v in u)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. stroke plates
def offset(x):
    """Unloaded level of a foot total: median of the samples within 20 N of its 5th percentile."""
    base = np.percentile(x, 5)
    return float(np.median(x[x <= base + 20.0]))


def plate_subset(total, fz):
    """Plates (1-based) whose |Fz| sum equals `total` (the smallest such set), or None."""
    k = fz.shape[1]
    tol = 1e-3 + 1e-5 * float(np.abs(total).max())
    for n in range(k + 1):
        for sub in itertools.combinations(range(k), n):
            s = fz[:, list(sub)].sum(axis=1) if sub else np.zeros(len(total))
            if np.abs(s - total).max() <= tol:
                return [i + 1 for i in sub]
    return None


def ab_events():
    """Stride events of the able-bodied MAT file: {subject number: {side: [(trial, ICstart, good)]}}."""
    import h5py
    p = f"{paths.MOVEATLAS}/datasets/{AB_MAT}"

    def arr(ds):
        if ds.attrs.get("MATLAB_empty", 0):                # MATLAB [] is stored as its dimensions
            return np.zeros(0)
        return np.asarray(ds, dtype=float).ravel()

    out = {}
    with h5py.File(p, "r") as f:
        refs = f["Sub"]["events"]
        for i in range(refs.shape[0]):
            g = f[refs[i, 0]]
            d = {}
            for side in "LR":
                tr, ic, good = (arr(g[f"{side}_{k}"]) for k in ("Trial", "ICstart", "GoodForcePlateLanding"))
                ok = np.isfinite(tr) & np.isfinite(ic) & np.isfinite(good)
                d[side] = list(zip(tr[ok].astype(int).tolist(), ic[ok].tolist(), good[ok].astype(int).tolist()))
            out[i + 1] = d
    return out


def onset_lag_ms(x, fps, frame, thr, search=10):
    """ms from frame (0-based, 100 Hz) to the nearest upward crossing of thr within +-search frames; None."""
    r = int(round(fps / 100.0))
    lo, hi = max(1, (frame - search) * r), min(len(x) - 1, (frame + search) * r)
    if hi <= lo:
        return None
    seg = x[lo - 1:hi + 1]
    up = np.where((seg[:-1] <= thr) & (seg[1:] > thr))[0] + lo
    if not len(up):
        return None
    s = up[np.argmin(np.abs(up - frame * r))]
    return (s - frame * r) * 1000.0 / fps


def stroke_plates(idx, quick):
    base = idx[(idx["source"] == STROKE) & (idx["tier"] == "RELEASE") & ~idx["lockbox"] & (idx["container"] != "")]
    force = base[base["modality"] == "force"].sort_values(["subject_ns", "key"])
    pose = {(r["subject_ns"], r["key"]): r for r in base[base["modality"] == "pose"].to_dict("records")}
    if quick:
        force = force.iloc[:10]
    events = ab_events()
    out = []
    for row in force.to_dict("records"):
        rec = C.load(row)
        F = np.asarray(rec["force"], dtype=np.float64)[:, :, 8]
        fps = float(rec["fps"])
        mask = np.asarray(rec["sensor_mask"], bool)[:, 8]
        rel = (f"datasets/pose/stroke_gait_vancriekinge/processed/aux_force/"
               f"{re.sub(r'[^A-Za-z0-9_-]', '_', row['subject_ns'] + '__' + row['key'])}.npz")
        r = {"subject_ns": row["subject_ns"], "group": row["group"], "key": row["key"], "n_force": len(F), "fps": fps}
        fz = None
        if os.path.exists(C.container_path(rel)):
            header = C.read_header(rel)
            with zipfile.ZipFile(C.container_path(rel)) as zf:
                aux = C.read_record(zf, header, -1)
            labels = list(aux["labels"])
            plates = sorted(int(l[2:]) for l in labels if l.startswith("Fz"))
            fz = np.abs(np.stack([np.asarray(aux["signal"], np.float64)[:, labels.index(f"Fz{k}")] for k in plates], 1))
            r["plates_in_aux"] = "|".join(map(str, plates))
        prow = pose.get((row["subject_ns"], row["key"]))
        z = None
        if prow is not None:
            p = C.load(prow)
            c = np.asarray(p["coords"], np.float64) * float(p["scale"])
            if len(c) * round(fps / 100.0) == len(F):
                z = (c[:, 13, 2], c[:, 14, 2])                # recorded z-up
        subj = row["subject_ns"].rsplit(":S", 1)
        num = int(subj[1]) if row["group"] == "able_bodied" and len(subj) == 2 else None
        trial = re.match(r"gait_SUBJ(\d+)_(\d+)$", row["key"])
        for j, side in enumerate("LR"):
            x = F[:, j]
            o = offset(x) if mask[j] else float("nan")
            peak = float(x.max())
            sub = plate_subset(x, fz) if fz is not None else None
            r[f"present_{side}"] = bool(mask[j])
            r[f"plates_{side}"] = "" if sub is None else "|".join(map(str, sub))
            r[f"plates_{side}_never_unloaded"] = "" if sub is None else "|".join(
                str(k) for k in sub if fz[:, k - 1].min() > 20.0)
            r[f"offset_{side}_N"], r[f"min_{side}_N"], r[f"peak_{side}_N"] = r4(o), r4(x.min()), r4(peak)
            if mask[j] and z is not None:
                thr = o + max(20.0, 0.1 * (peak - o))
                ld = x[:: int(round(fps / 100.0))][: len(z[0])] > thr
                if ld.sum() >= 10:
                    d = float(np.median(z[1 - j][ld] - z[j][ld]))
                    r[f"kin_{side}_m"] = r4(d)
                    r[f"kin_{side}"] = "consistent" if d > KIN_CM else "swapped" if d < -KIN_CM else "undetermined"
                else:
                    r[f"kin_{side}"] = "no_load"
            elif mask[j]:
                r[f"kin_{side}"] = "no_pose"
            if num is not None and trial and int(trial.group(1)) == num and mask[j]:
                ctrial = int(trial.group(2))
                for hyp, shift in (("", 1), ("_trial_as_index", 0)):
                    good = [ic for tr, ic, g in events.get(num, {}).get(side, []) if g == 1 and tr - shift == ctrial]
                    same = [onset_lag_ms(x, fps, int(round(ic)) - 1, o + 20.0) for ic in good]
                    other = [onset_lag_ms(F[:, 1 - j], fps, int(round(ic)) - 1, offset(F[:, 1 - j]) + 20.0)
                             if mask[1 - j] else None for ic in good]
                    r[f"events{hyp}_{side}"] = len(good)
                    r[f"events{hyp}_{side}_same_foot_onset"] = sum(v is not None and abs(v) <= ONSET_MS for v in same)
                    r[f"events{hyp}_{side}_other_foot_onset"] = sum(v is not None and abs(v) <= ONSET_MS for v in other)
                    if not hyp:
                        r[f"events_{side}_lag_ms"] = "|".join(f"{v:.0f}" for v in same if v is not None)
        out.append(r)
    return pd.DataFrame(out)


def plates_summary(pl):
    s = {"force_records": len(pl), "by_group": pl["group"].value_counts().to_dict()}
    for side in "LR":
        present = pl[pl[f"present_{side}"]]
        s[f"foot_{side}"] = {
            "present": len(present),
            "plates_matched": int((present[f"plates_{side}"] != "").sum()),
            "plate_sets": present[f"plates_{side}"].value_counts().to_dict(),
            "offset_N_median_p90_max": [r4(q(present[f"offset_{side}_N"], p)) for p in (50, 90, 100)],
            "records_min_above_1N": int((present[f"min_{side}_N"] > 1.0).sum()),
            "records_with_a_never_unloaded_plate": int((present[f"plates_{side}_never_unloaded"] != "").sum()),
            "kinematic": {g: present.loc[present["group"] == g, f"kin_{side}"].value_counts().to_dict()
                          for g in sorted(pl["group"].unique())},
        }
        for hyp in ("", "_trial_as_index"):
            col = f"events{hyp}_{side}"
            if col in pl:
                s[f"foot_{side}"][f"events{hyp}"] = {
                    "good_landings": int(pl[col].fillna(0).sum()),
                    "same_foot_onset_within_30ms": int(pl[f"{col}_same_foot_onset"].fillna(0).sum()),
                    "other_foot_onset_within_30ms": int(pl[f"{col}_other_foot_onset"].fillna(0).sum())}
        lags = [float(v) for cell in pl.get(f"events_{side}_lag_ms", pd.Series(dtype=str)).fillna("") for v in str(cell).split("|") if v]
        s[f"foot_{side}"]["event_lag_ms_median_p10_p90"] = [r4(q(lags, p)) for p in (50, 10, 90)]
    return s


# ---------------------------------------------------------------- 4b. raw-file checks
def weargait_raw_check(adm, quick, sites):
    """Paired weargait recordings against their raw CSV (module docstring); admitted recordings only."""
    w = adm[adm["source"] == "weargait_pd"]
    kinds = w.groupby(["subject_ns", "recording"])["stream"].agg(frozenset)
    both = sorted(k for k, v in kinds.items() if v == {"imu", "force"})
    rng = np.random.default_rng(SEED)
    pick = {both[i] for i in rng.choice(len(both), 5 if quick else N_RAW, replace=False)}
    lb = sites.index("lower_back")
    s = collections.Counter()
    acc_diff = []
    for pr in S.paired("weargait_pd", recordings=pick):
        pid = pr.subject_ns.split(":", 1)[1]
        hits = [h for h in glob.glob(f"{paths.MOVEATLAS}/datasets/imu/weargait_pd/raw/**/CONTROL PARTICIPANTS/CSV files/"
                                     f"{glob.escape(pid)}_{pr.key}.csv", recursive=True)
                if not os.path.basename(h).startswith("._")]
        s["recordings"] += 1
        if len(hits) != 1:
            s["raw_file_not_unique"] += 1
            continue
        cols = ["LTotalForce", "RTotalForce", "LowerBack_Acc_X", "LowerBack_Acc_Y", "LowerBack_Acc_Z"]
        raw = pd.read_csv(hits[0], usecols=cols, encoding="utf-8")
        tot = raw[cols[:2]].to_numpy(float)
        x = np.asarray(pr.streams["force"]["force"], np.float64)[:, :, 8]
        masked = ~np.asarray(pr.streams["force"]["sample_valid"], bool)[:, 0]
        n = len(pr.t["imu"])
        s["same_n_as_raw"] += int(len(tot) == len(x) == n)
        if len(tot) != len(x):
            continue
        fin = np.isfinite(tot)
        s["totals_equal_where_raw_finite"] += int(np.array_equal(tot[fin], x[fin]))
        s["raw_nan_foot_samples"] += int((~fin).sum())
        s["raw_nan_foot_samples_read_as_0"] += int((x[~fin] == 0).sum())
        both_nan = ~fin[:, 0] & ~fin[:, 1]
        s["samples"] += len(x)
        s["raw_both_nan"] += int(both_nan.sum())
        s["raw_both_nan_masked"] += int((both_nan & masked).sum())
        s["masked_with_both_raw_finite"] += int((masked & fin.all(axis=1)).sum())
        s["raw_one_foot_nan_left_valid"] += int(((fin[:, 0] ^ fin[:, 1]) & ~masked).sum())
        imu = pr.streams["imu"]
        ids = [int(i) for i in np.asarray(imu["location_ids"]).ravel()]
        if lb in ids:
            a = np.asarray(imu["signal"], np.float64)[:, ids.index(lb), 0:3] * G0
            r = raw[cols[2:]].to_numpy(float)
            ok = np.isfinite(a).all(axis=1) & np.isfinite(r).all(axis=1)
            d = float(np.abs(a[ok] - r[ok]).max()) if ok.any() else float("nan")
            acc_diff.append(d)
            s["lower_back_acc_equal_same_rows"] += int(d < 1e-3)
    s = dict(s)
    s["lower_back_acc_max_abs_diff_m_s2"] = r4(max(acc_diff)) if acc_diff else None
    return s


def gait120_trc_check(adm):
    """Step frames of the pairing rule (EMG gait_events) against the raw TRC headers, every admitted trial."""
    g = adm[adm["source"] == "gait120"]
    events = {}
    for row in g[g["stream"] == "emg"].to_dict("records"):
        ge = C.load(row).get("gait_events") or {}
        events[(row["subject_ns"], row["recording"])] = ge.get("step_target_frames_100hz")
    steps = collections.defaultdict(list)
    for sns, rk, key in zip(g.loc[g["stream"] == "pose3d", "subject_ns"], g.loc[g["stream"] == "pose3d", "recording"],
                            g.loc[g["stream"] == "pose3d", "key"]):
        steps[(sns, rk)].append(int(re.search(r"_step(\d+)$", key).group(1)))
    by_zip = collections.defaultdict(list)
    for sns, rk in steps:
        num = int(sns.split(":S", 1)[1])
        lo = (num - 1) // 10 * 10 + 1
        by_zip[f"Gait120_{lo:03d}_to_{lo + 9:03d}.zip"].append((sns, rk))
    s = collections.Counter()
    for zname, trials in sorted(by_zip.items()):
        with zipfile.ZipFile(f"{paths.MOVEATLAS}/datasets/emg/gait120/raw/{zname}") as z:
            members = set(z.namelist())
            for sns, rk in trials:
                s["trials"] += 1
                task, trial = re.fullmatch(r"(.+)_trial(\d+)", rk).groups()
                trc = []
                for k in sorted(steps[(sns, rk)]):
                    m = f"{sns.split(':', 1)[1]}/MotionCapture/{GAIT120_TRC[task]}/TRC/Trial{int(trial):02d}/Step{k:02d}.trc"
                    if m not in members:
                        trc = None
                        break
                    with z.open(m) as f:
                        head = [f.readline().decode("utf-8", "replace") for _ in range(3)]
                    v = dict(zip(head[1].split(), head[2].split()))
                    s["trc_at_100hz"] += int(float(v["DataRate"]) == 100.0)
                    trc.append([int(v["OrigDataStartFrame"]), int(v["OrigDataStartFrame"]) + int(v["NumFrames"]) - 1])
                if trc is None:
                    s["trc_missing"] += 1
                    continue
                rule = events.get((sns, rk))
                if not rule:
                    s["no_gait_events"] += 1
                    continue
                s["rule_equals_trc"] += int([[int(round(float(a))), int(round(float(b)))] for a, b in rule] == trc)
    return dict(s)


def stroke_mapping_check(scales):
    """Correlation between sub_char Height (MAT Data.Sub(i)) and the median torso length of subject S(i + shift)."""
    import h5py
    torso = {int(k.rsplit(":S", 1)[1]): float(np.median(v)) for k, v in scales.items()}
    height = {}
    with h5py.File(f"{paths.MOVEATLAS}/datasets/{AB_MAT}", "r") as f:
        refs = f["Sub"]["sub_char"]
        for i in range(refs.shape[0]):
            height[i + 1] = float(np.asarray(f[refs[i, 0]]["Height"]).ravel()[0])
    out = {"subjects": len(torso), "mat_subjects": len(height), "corr_height_torso_by_shift": {}}
    for shift in (-2, -1, 0, 1, 2):
        ids = [k for k in sorted(torso) if k + shift in height]
        if len(ids) > 2:
            out["corr_height_torso_by_shift"][str(shift)] = r4(np.corrcoef([torso[k] for k in ids],
                                                                           [height[k + shift] for k in ids])[0, 1])
    return out


# ---------------------------------------------------------------- 5. demographics
def demographics_table(adm, extra):
    out = []
    for source in S.PILOT:
        subjects = sorted(set(adm.loc[adm["source"] == source, "subject_ns"]))
        info = [S.demographic(source, s) for s in subjects]
        d = S.demographics(source)
        sex = collections.Counter((i.get("sex") or "unknown") for i in info)
        ages = [i.get("age") for i in info if i.get("age") is not None and np.isfinite(i.get("age"))]
        mass = [i.get("mass_kg") for i in info if i.get("mass_kg") is not None and np.isfinite(i.get("mass_kg"))]
        files = sorted({f.replace(paths.MOVEATLAS + "/", "") for f in d["file"]}) if len(d) else []
        out.append({"source": source, "files": " + ".join(files) or "none", "table_rows": len(d),
                    "admitted_subjects": len(subjects), "matched": sum(bool(i) for i in info),
                    "female": sex.get("F", 0), "male": sex.get("M", 0), "sex_unknown": sex.get("unknown", 0),
                    "age_known": len(ages), "age_min": r4(q(ages, 0)), "age_median": r4(q(ages, 50)),
                    "age_max": r4(q(ages, 100)), "mass_known": len(mass),
                    "ages_on_records": "" if extra.get(source, {}).get("sixmwt_ages_on_records") is None
                    else str(int(extra[source]["sixmwt_ages_on_records"]))})
    return pd.DataFrame(out)


def weargait_columns():
    """Column names of the weargait tables around sex, and rows whose Gender and Sex differ."""
    import csv
    import glob
    res = {}
    for pattern in S.DEMOGRAPHIC_FILES["weargait_pd"]:
        p = glob.glob(f"{paths.MOVEATLAS}/datasets/{pattern}")[0]
        with open(p, encoding="utf-8-sig", newline="") as f:
            t = list(csv.reader(f))
        h = next(i for i, r in enumerate(t) if r and r[0].strip() == "Subject ID")
        cols = [c.strip() for c in t[h]]
        body = [r for r in t[h + 1:] if r and r[0].strip()]
        g, s = cols.index("Gender"), cols.index("Sex")
        blank = [r[g].strip() in ("", "-") for r in body]
        res[os.path.basename(p)] = {"columns": [c for c in cols[:12] if c], "rows": len(body),
                                    "gender_blank": sum(blank),
                                    "gender_differs_from_sex": sum(not b and r[g].strip().lower() != r[s].strip().lower()
                                                                   for b, r in zip(blank, body)),
                                    "sex_values": dict(collections.Counter(r[s].strip() for r in body))}
    return res


# ---------------------------------------------------------------- 6. tasks
def task_table(adm, idx):
    out = []
    for source in S.PILOT:
        a = adm[adm["source"] == source][["subject_ns", "recording"]].drop_duplicates()
        names = collections.Counter()
        for rk in a["recording"]:
            names[(W.task_name(source, rk), re.sub(r"\d+", "#", rk))] += 1
        for (task, pattern), n in sorted(names.items()):
            out.append({"source": source, "task": task, "key_pattern": pattern, "recordings": n})
    am = idx[(idx["source"] == "amass") & (idx["tier"] == "RELEASE") & ~idx["lockbox"] & (idx["container"] != "")]
    for task, n in collections.Counter(W.task_name("amass", k) for k in am["key"]).most_common():
        out.append({"source": "amass (non-lockbox RELEASE takes, index only)", "task": task, "key_pattern": "", "recordings": n})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- main
def md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else str(int(v)) if v.is_integer() else str(v)
        return str(v)

    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "stage1_sources"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--md-only", action="store_true", help="rewrite summary.md from the outputs of the last run")
    a = ap.parse_args()
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.md_only:
        write_md(a.out)
        return
    t_start = time.time()
    idx = C.index()
    adm = S.rows()
    for row in adm.to_dict("records"):
        C.check(row)
    summary = {"moveatlas_head": moveatlas_head(paths.MOVEATLAS), "quick": a.quick,
               "admitted_rows": len(adm), "admitted_rows_pass_check_without_flags": len(adm)}
    src = source_table(adm, idx)
    src.to_csv(os.path.join(a.out, "sources.csv"), index=False, lineterminator="\n")
    summary["persons"] = adm.groupby("source")["person_id"].nunique().to_dict()

    folds = S.folds()
    folds.to_csv(os.path.join(a.out, "folds.csv"), index=False, lineterminator="\n")
    align, test = S.g2_halves()
    dev = set(folds[(folds["source"] == "weargait_pd") & (folds["fold"] == "dev")]["person_id"])
    summary["folds"] = {"counts": {f"{s}|{f}": int(n) for (s, f), n in folds.groupby(["source", "fold"]).size().items()},
                        "sha256": S.fold_hashes(folds), "seed": S.SEED,
                        "g2_align": len(align), "g2_test": len(test), "weargait_dev_in_g2_test": len(dev & set(test)),
                        "weargait_dev": len(dev), "g2_test_sha256": S.person_hash(test),
                        "g2_align_sha256": S.person_hash(align)}

    tasks = task_table(adm, idx)
    tasks.to_csv(os.path.join(a.out, "tasks.csv"), index=False, lineterminator="\n")
    vocab = W.task_vocabulary()
    summary["task_vocabulary"] = {"sha256": vocab["sha256"], "status": vocab["status"], "tasks": vocab["tasks"]}

    pc = pose_checks(adm, a.quick)
    pc.to_csv(os.path.join(a.out, "pose_checks.csv"), index=False, lineterminator="\n")
    print("pose checks done", flush=True)

    pl = stroke_plates(idx, a.quick)
    pl.to_csv(os.path.join(a.out, "stroke_plates.csv"), index=False, lineterminator="\n")
    print("stroke plates done", flush=True)

    sites = locations()
    pairs, content, extra, scales = full_pass(a.quick, sites)
    pairs.to_csv(os.path.join(a.out, "pairs.csv"), index=False, lineterminator="\n")
    content.to_csv(os.path.join(a.out, "content.csv"), index=False, lineterminator="\n")
    summary["pass_extra"] = extra
    summary["stroke_mapping"] = stroke_mapping_check(scales)
    summary["weargait_raw"] = weargait_raw_check(adm, a.quick, sites)
    summary["gait120_trc"] = gait120_trc_check(adm)
    print("raw-file checks done", flush=True)

    demo = demographics_table(adm, extra)
    demo.to_csv(os.path.join(a.out, "demographics.csv"), index=False, lineterminator="\n")
    summary["weargait_tables"] = weargait_columns()
    summary["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, default=str)
    prov = {"script": os.path.basename(__file__), "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_head": summary["moveatlas_head"], "run_at": dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "quick": a.quick, "seed": SEED,
            "inputs": ["cache/manifest/manifest_records.csv.gz", "cache/manifest/record_index.csv.gz",
                       "MoveAtlas canonical records of the pilot sources (mdfm.data.canonical)",
                       "datasets/pose/stroke_gait_vancriekinge/processed/aux_force/*.npz",
                       f"datasets/{AB_MAT}"] + [f"datasets/{p}" for v in S.DEMOGRAPHIC_FILES.values() for p in v]
            + ["datasets/imu/weargait_pd/raw/Version 1/**/CONTROL PARTICIPANTS/CSV files/*.csv (40 recordings)",
               "datasets/emg/gait120/raw/Gait120_*.zip (TRC headers of admitted subjects)"]}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    write_md(a.out)
    print(f"done in {summary['runtime_s']} s", flush=True)


def checks(out, s, src, pairs):
    """The done-when lines of B4.5 (spec section 7) and the facts that change a later task, as measured."""
    al = {}
    p = os.path.join(out, "gait120_alignment.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            al = json.load(fh)
    pr = {r["source"]: r for r in pairs.to_dict("records")}
    sr = {(r["source"], r["stream"]): r for r in src.to_dict("records")}
    pc, sp, f, ex = s["pose_checks"], s["stroke_plates"], s["folds"], s["pass_extra"]
    wg = pr.get("weargait_pd", {})
    wr, trc, sm = s.get("weargait_raw", {}), s.get("gait120_trc", {}), s.get("stroke_mapping", {})
    zg = ex.get("weargait_pd", {}).get("force_zero_gaps", {})
    s002 = al.get("s002_level_walking_trial01", {})
    L = ["## Done-when checks of B4.5", "",
         "| check | measured |", "|---|---|",
         f"| every admitted row passes canonical.check without flags | {s['admitted_rows_pass_check_without_flags']} of "
         f"{s['admitted_rows']} rows |",
         f"| 68 weargait controls and 138 stroke able-bodied persons admitted | {s['persons']['weargait_pd']} and "
         f"{s['persons']['stroke_gait_vancriekinge']} |",
         "| torso length coords x scale in 0.35 to 0.65 m on 50 records per pose source | " + "; ".join(
             f"{k}: {v['torso_m_in_band']} of {v['records']} (min {v['torso_m_min_median_max'][0]}, max "
             f"{v['torso_m_min_median_max'][2]} m)" for k, v in pc.items()) + " |",
         "| up axis of each 3D pose source measured, stroke rotated to y-up | " + "; ".join(
             f"{k}: recorded {v['up_recorded']}, after sources.pair {v['up_after_pair']}" for k, v in pc.items()) + " |",
         f"| weargait IMU and insole pairs share n at 100 Hz | {sr[('weargait_pd', 'force')]['recordings']} insole "
         f"recordings, all at fps {sr[('weargait_pd', 'force')]['fps']} beside IMU at {sr[('weargait_pd', 'imu')]['fps']}; "
         f"paired {wg.get('paired')}, refused for a clock mismatch {int(wg.get('dropped_clock', 0))}; against the raw "
         f"CSV of {wr.get('recordings')} random recordings: same n on {wr.get('same_n_as_raw')}, lower-back acc equal at "
         f"the same rows on {wr.get('lower_back_acc_equal_same_rows')} |",
         "| S002 gait120 frames 456 to 750 (EMG) and 468 to 679 (pose), 575 kept once, own scale per step | " + (
             f"EMG {s002.get('emg_frames')}, pose {s002.get('pose_frames')} ({s002.get('pose_frame_count')} frames, "
             f"shared {s002.get('shared_frames')}), step scales {s002.get('step_scales')}" if s002 else "not measured")
         + f"; the step frames of the rule equal the raw TRC headers on {trc.get('rule_equals_trc')} of "
         f"{trc.get('trials')} admitted trials |",
         "| the alignment probe covers every non-lockbox gait120 trial | " + (
             f"{al.get('trials')} of {al.get('trials_in_index')} trials, status {al.get('status')}" if al else "not run")
         + " |",
         f"| the stroke plate-to-foot mapping is tabled | stroke_plates.csv: {sp['force_records']} records, foot totals "
         f"matched to plate sets L {sp['foot_L']['plates_matched']} of {sp['foot_L']['present']}, R "
         f"{sp['foot_R']['plates_matched']} of {sp['foot_R']['present']} |",
         f"| no field outside the whitelist survives cut | {int(sum(r['windows'] for r in pr.values()))} windows validated, "
         f"{int(sum(r['windows_with_forbidden_keys'] for r in pr.values()))} with a forbidden key (and "
         "tests/test_windows.py injects label fields) |",
         f"| the 14 weargait dev controls lie in the G2 TEST half | {f['weargait_dev_in_g2_test']} of {f['weargait_dev']} |",
         f"| folds hashed per source | {sum(len(v) for v in f['sha256'].values())} person-list hashes over "
         f"{len(f['sha256'])} sources |", ""]
    content = pd.read_csv(os.path.join(out, "content.csv"), encoding="utf-8", keep_default_na=False)
    emg = content[(content["stream"] == "emg") & (content["source"] == "stroke_gait_vancriekinge")]
    dup = sum(int(r["records"]) for r in emg.to_dict("records")
              if len(r["muscles"].split("|")) != len(set(r["muscles"].split("|"))))
    tasks = pd.read_csv(os.path.join(out, "tasks.csv"), encoding="utf-8", keep_default_na=False)
    am = tasks[tasks["source"].str.startswith("amass")]
    am_unknown = int(am.loc[am["task"] == "unknown", "recordings"].sum())
    kin = {side: sp[f"foot_{side}"]["kinematic"] for side in "LR"}
    ev = {side: sp[f"foot_{side}"].get("events", {}) for side in "LR"}
    ev2 = {side: sp[f"foot_{side}"].get("events_trial_as_index", {}) for side in "LR"}
    demo = s.get("weargait_tables", {})
    L += ["## Findings", "",
          f"- weargait_pd: {sr[('weargait_pd', 'imu')]['persons']} controls hold IMU recordings, "
          f"{sr[('weargait_pd', 'force')]['persons']} of them insole totals; the insole unit is not stated (unit unknown "
          "in the windows).",
          f"- weargait_pd insole gaps: on {wr.get('recordings')} random recordings the canonical totals equal the raw "
          f"LTotalForce and RTotalForce where those are finite on {wr.get('totals_equal_where_raw_finite')}, and "
          f"{wr.get('raw_nan_foot_samples_read_as_0')} of the {wr.get('raw_nan_foot_samples')} foot samples where the raw "
          "total is NaN read 0 N in the canonical record (the insole was not streaming; MoveAtlas writes 0, a correction "
          "for the owner). sources.pair marks a sample invalid where both foot totals read exactly 0: of the "
          f"{wr.get('raw_both_nan')} samples with both raw totals NaN it masks {wr.get('raw_both_nan_masked')}, it masks "
          f"{wr.get('masked_with_both_raw_finite')} samples whose raw totals are finite (both 0), and "
          f"{wr.get('raw_one_foot_nan_left_valid')} one-foot NaN samples stay valid at 0 N. Over the full pass: "
          f"{zg.get('gap_samples')} of {zg.get('samples')} insole samples masked, in {zg.get('records_with_a_gap')} of "
          f"{zg.get('records')} records, {zg.get('records_starting_with_a_gap')} of which start with a gap (leading gap s "
          f"median, p90, max: {zg.get('leading_gap_s_median_p90_max')}).",
          f"- stroke able-bodied: EMG for {sr[('stroke_gait_vancriekinge', 'emg')]['persons']} of "
          f"{sr[('stroke_gait_vancriekinge', 'pose3d')]['persons']} persons, force for "
          f"{sr[('stroke_gait_vancriekinge', 'force')]['persons']}; EMG amplitude flag on "
          f"{ex.get('stroke_gait_vancriekinge', {}).get('emg_amplitude_flag')} of "
          f"{ex.get('stroke_gait_vancriekinge', {}).get('emg_records')} paired EMG records.",
          "- Stroke plates: each foot total is the |Fz| sum of the plates MoveAtlas assigned to that foot over the whole "
          "trial, so a plate whose zero is offset, or that carries a standing load all along, enters the total: records with a "
          f"plate never below 20 N, L {sp['foot_L']['records_with_a_never_unloaded_plate']}, R "
          f"{sp['foot_R']['records_with_a_never_unloaded_plate']}; records whose foot total never drops to 1 N, L "
          f"{sp['foot_L']['records_min_above_1N']}, R {sp['foot_R']['records_min_above_1N']}; unloaded offset (N) "
          f"median, p90, max: L {sp['foot_L']['offset_N_median_p90_max']}, R {sp['foot_R']['offset_N_median_p90_max']}. "
          "force_nll should remove the per-record offset or mask these records (B4.9).",
          f"- Plate to foot, kinematic check (loaded foot the lower ankle): L {kin['L']}, R {kin['R']}.",
          f"- Plate to foot, the dataset's own stride events (able-bodied, GoodForcePlateLanding 1, trial Trial - 1): "
          f"L {ev['L'].get('same_foot_onset_within_30ms')} of {ev['L'].get('good_landings')} landings show a same-foot "
          f"onset within 30 ms of ICstart, {ev['L'].get('other_foot_onset_within_30ms')} an other-foot onset; R "
          f"{ev['R'].get('same_foot_onset_within_30ms')} of {ev['R'].get('good_landings')}, other foot "
          f"{ev['R'].get('other_foot_onset_within_30ms')}; lag ms median, p10, p90 L "
          f"{sp['foot_L']['event_lag_ms_median_p10_p90']}, R {sp['foot_R']['event_lag_ms_median_p10_p90']}. With Trial "
          f"read as the C3D index instead: L {ev2['L'].get('same_foot_onset_within_30ms')} of "
          f"{ev2['L'].get('good_landings')}, R {ev2['R'].get('same_foot_onset_within_30ms')} of "
          f"{ev2['R'].get('good_landings')}. The same agreement confirms that MAT Sub(i) is subject S(i), which "
          "sources.demographics uses for sex, age and mass; so does the correlation between sub_char Height and the "
          f"median torso length of the subject's records over {sm.get('subjects')} subjects, by mapping shift "
          f"{sm.get('corr_height_torso_by_shift')}.",
          "- Demographics: weargait_pd sex, age and weight (kg) live in the raw CONTROLS and PD tables ("
          + "; ".join(f"{k}: {v['rows']} rows, Sex {v['sex_values']}, Gender blank on {v['gender_blank']} and "
                      f"different from Sex on {v['gender_differs_from_sex']}" for k, v in demo.items())
          + "); gait120 in Subject_Demographics.xlsx (males only); stroke able-bodied in the MAT v7.3 file "
          "(Data.Sub(i).sub_char: Male, Age, Weight kg, Height mm); nonan in the GaitPrint subject tables; "
          "sixmwt_lifespan in each DATA/Subject k/Data.mat of 6MWT_IMU_Dataset.zip (Age, BodyWeight kg, and Sex, a "
          "MATLAB string object decoded from the MAT file's MCOS subsystem with scipy; demographics.csv).",
          f"- gait120: trials with a gap between steps {ex.get('gait120', {}).get('gait120_trials_with_a_step_gap')}; "
          f"shared boundary frames {ex.get('gait120', {}).get('gait120_boundaries')}, whose two copies differ by at most "
          f"{al.get('ok', {}).get('boundary_diff_mm_max')} mm; paired trials shorter than 1 s (no window): "
          f"{int(pr.get('gait120', {}).get('records_short', 0))}. Level walking, TA and GAS medialis envelope peaks "
          "against the steps placed by the rule: " + (
              "TA within 15 percent of a step end on {ta_within_15_of_a_step_end}, GAS in 25 to 60 percent on "
              "{gas_in_25_60}, of {scored} trials".format(**al["per_task"]["level_walking"]["in_band"])
              if al.get("per_task", {}).get("level_walking", {}).get("in_band") else "not measured") + " "
          "(gait120_alignment.md).",
          f"- stroke able-bodied: paired trials shorter than 1 s (no window): "
          f"{int(pr.get('stroke_gait_vancriekinge', {}).get('records_short', 0))} of "
          f"{int(pr.get('stroke_gait_vancriekinge', {}).get('paired', 0))}; the EMG channel set varies between "
          f"trials ({len(emg)} distinct sets over {int(emg['records'].astype(int).sum())} paired records) and {dup} "
          "records hold a muscle and side twice (electrode variants of one muscle, ingest_stroke_gait_analogs.py), "
          "which the EMG decoder must index by channel, not by muscle name.",
          f"- Task vocabulary: every admitted recording of the six pilot sources maps to a task; the DRAFT amass "
          f"keyword rules leave {am_unknown} of {int(am['recordings'].sum())} non-lockbox RELEASE takes unknown "
          "(tasks.csv).", ""]
    return L


def pose_summary(pc):
    return {s: {"records": int(len(g)), "torso_m_min_median_max": [r4(g["torso_m"].min()), r4(g["torso_m"].median()),
                                                                     r4(g["torso_m"].max())],
                "torso_m_in_band": int(g["torso_m"].between(*TORSO_BAND).sum()),
                "torso_units_max_abs_dev_from_1": r4((g["torso_units"] - 1).abs().max()),
                "pelvis_abs_max": float(g["pelvis_abs_max"].max()),
                "up_recorded": g["up_recorded"].value_counts().to_dict(),
                "up_after_pair": g["up_after_pair"].value_counts().to_dict()}
            for s, g in pc.groupby("source")}


def read_plates(path):
    pl = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    numeric = re.compile(r"(offset|min|peak)_[LR]_N|n_force|fps|kin_[LR]_m"
                         r"|events(_trial_as_index)?_[LR](_same_foot_onset|_other_foot_onset)?")
    for c in pl.columns:
        if numeric.fullmatch(c):
            pl[c] = pd.to_numeric(pl[c], errors="coerce")
    for side in "LR":
        pl[f"present_{side}"] = pl[f"present_{side}"] == "True"
    return pl


def write_md(out):
    """summary.md from summary.json and the CSVs of `out`; the pose and plate sections of summary.json are
    recomputed from pose_checks.csv and stroke_plates.csv, so that the JSON and the page always agree."""
    rd = lambda name: pd.read_csv(os.path.join(out, name), encoding="utf-8", keep_default_na=False)
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    src, pairs, content, demo, tasks = (rd(n) for n in ("sources.csv", "pairs.csv", "content.csv", "demographics.csv",
                                                         "tasks.csv"))
    pc = pd.read_csv(os.path.join(out, "pose_checks.csv"), encoding="utf-8")
    s["pose_checks"] = pose_summary(pc)
    s["stroke_plates"] = plates_summary(read_plates(os.path.join(out, "stroke_plates.csv")))
    s["weargait_tables"] = weargait_columns()
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(s, fh, indent=1, default=str)
    sp, f = s["stroke_plates"], s["folds"]
    L = ["# Stage-1 real sources (plan task B4.5)", "",
         "Generated by scripts/probe_stage1_sources.py; every number below is read from summary.json, the CSVs of "
         "this directory and gait120_alignment.json (scripts/probe_gait120_emg_alignment.py). MoveAtlas commit: "
         + s["moveatlas_head"] + ". Rules and definitions in the script docstrings and in mdfm/data/sources.py."
         + (" QUICK RUN (subsets only)." if s["quick"] else ""), ""]
    L += checks(out, s, src, pairs)
    L += ["## Admitted rows", "",
         f"Admitted rows: {s['admitted_rows']}, all passing canonical.check without flags. Persons: "
         + ", ".join(f"{k} {v}" for k, v in sorted(s["persons"].items())) + ".", ""]
    L += md_table(src)
    L += ["", "## Pairing and windows", "",
          "10 s windows, stride 10 s, 1 s minimum, native samples; every window passed windows.validate.", ""]
    L += md_table(pairs)
    ex = s["pass_extra"]
    g = ex.get("gait120", {})
    L += ["", f"gait120: trials with a gap between their steps: {g.get('gait120_trials_with_a_step_gap')}; shared "
          f"boundary frames: {g.get('gait120_boundaries')} (distance between their two copies in gait120_alignment.md).",
          "EMG records in the pass and the amplitude flag of their notes: " + "; ".join(
              f"{k} {v['emg_records']} records, flag on {v['emg_amplitude_flag']}, record p99 |signal| min, median, max "
              f"{v['emg_p99_mV']} mV" for k, v in ex.items() if v.get("emg_records")) + ".", "",
          "## Channel content", ""]
    L += md_table(content.fillna(""))
    L += ["", "## 3D pose: torso length and up axis", "",
          f"{N_POSE} random admitted pose records per source (seed {SEED}); torso band {TORSO_BAND[0]} to {TORSO_BAND[1]} m.", "",
          "| source | records | torso m min / median / max | in band | max abs torso units minus 1 | max abs pelvis coordinate | up as recorded | up after sources.pair |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in s["pose_checks"].items():
        L.append(f"| {k} | {v['records']} | {v['torso_m_min_median_max']} | {v['torso_m_in_band']} | "
                 f"{v['torso_units_max_abs_dev_from_1']} | {v['pelvis_abs_max']} | {v['up_recorded']} | {v['up_after_pair']} |")
    L += ["", "## Stroke force plates", "",
          f"Non-lockbox force records: {sp['force_records']} ({sp['by_group']}). Per record in stroke_plates.csv.", "",
          "| foot | present | plates matched | plate sets | unloaded offset N median / p90 / max | records whose minimum exceeds 1 N | records with a plate never below 20 N | kinematic check by group | dataset events: good landings, same-foot onset within 30 ms, other-foot onset (Trial - 1) | same with Trial as the index | event lag ms median / p10 / p90 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for side in "LR":
        v = sp[f"foot_{side}"]
        e, e2 = v.get("events", {}), v.get("events_trial_as_index", {})
        L.append(f"| {side} | {v['present']} | {v['plates_matched']} | {v['plate_sets']} | {v['offset_N_median_p90_max']} | "
                 f"{v['records_min_above_1N']} | {v['records_with_a_never_unloaded_plate']} | {v['kinematic']} | "
                 f"{e.get('good_landings')}, {e.get('same_foot_onset_within_30ms')}, {e.get('other_foot_onset_within_30ms')} | "
                 f"{e2.get('good_landings')}, {e2.get('same_foot_onset_within_30ms')}, {e2.get('other_foot_onset_within_30ms')} | "
                 f"{v['event_lag_ms_median_p10_p90']} |")
    L += ["", "## Demographics", ""]
    L += md_table(demo.fillna(""))
    L += ["", "weargait_pd tables: " + "; ".join(f"{k}: {v['rows']} rows, columns {v['columns']}, Gender blank on "
                                               f"{v['gender_blank']} rows and different from Sex on "
                                               f"{v['gender_differs_from_sex']}, Sex values {v['sex_values']}"
                                               for k, v in s["weargait_tables"].items()) + ".", "",
          "## Folds", "",
          f"Seed {f['seed']}. G2 halves (scripts/g2_renderer_realism.py rule): ALIGN {f['g2_align']}, TEST {f['g2_test']} "
          f"(SHA-256 {f['g2_test_sha256']}); weargait dev controls {f['weargait_dev']}, of which in the TEST half "
          f"{f['weargait_dev_in_g2_test']}.", "",
          "| source | fold | persons | SHA-256 of the sorted list |", "|---|---|---|---|"]
    for key, n in sorted(f["counts"].items()):
        so, fo = key.split("|")
        L.append(f"| {so} | {fo} | {n} | {f['sha256'][so][fo]} |")
    L += ["", "## Tasks", "", f"docs/rules/task_vocabulary.yaml ({s['task_vocabulary']['status']}), SHA-256 "
          f"{s['task_vocabulary']['sha256']}.", ""]
    L += md_table(tasks)
    L += ["", f"Runtime: {s['runtime_s']} s.", ""]
    text = "\n".join(L)
    assert "\u2014" not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
