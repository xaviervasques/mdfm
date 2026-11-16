"""Stage-2 sources (weekend task T19; design docs/MFM_DESIGN_2026-09-24.md section 5, stage 2): the training
portions after lockbox v1.2, measured from the canonical records, with proposed dev folds for the weargait_pd
PD persons and the stroke patients, and the site completeness of every non-lockbox weargait_pd person.

Sources and groups (column group of cache/manifest/manifest_records.csv.gz, scripts/build_tiers_and_lockbox.py):
pads, weargait_pd, physionet_gaitpdb, gaitndd, clinical_gait_multipatho, imu_inthewild_iprognosis,
modys_video, stroke_gait_vancriekinge and gaitrec, every group of each. The column stage2_scope marks the
groups that design section 5 trains in stage 2 (weargait_pd PD, the stroke patients, the gaitrec orthopaedic
patients, every group of the other sources); the weargait_pd and stroke able-bodied controls and the gaitrec
healthy controls are read beside them (stage-1 sources, needed for the weargait site table). koa_pd_nm
(INTERNAL) is counted from the index only: none of its records is opened.

Reader: mdfm.data.canonical with every flag at False. Records come from canonical.iterate(sources=<source>,
lane="body") with allow_eval and allow_lockbox left at False (no lockbox, EVAL, CODY or EXCLUDED_V1 row is
selected, and canonical.check refuses each row again before its record is read); if iterate stops on a record
whose metadata disagree with the index, the remaining rows are read one by one with canonical.load and the
failures are counted. A refusal (canonical.Refused, a PermissionError) is never counted as a failure: it stops
the run. Checks of the reader, per source: canonical.load on one lockbox row must raise Refused
(before any container opens), and canonical.load on 3 rows already read must return the same main array.
--quick reads 12 rows per source, group and modality with canonical.load (a dry run only).

Per record of the training portion (the non-lockbox rows):
- hours: n / fps from the record's main array and fps, beside the index n_frames / fps; records without fps are
  listed in fps_missing.csv with the rate scripts/probe_frame_rates.py recovered (reports/nuisance/
  frame_rates.csv) where one exists;
- IMU: sites (location_ids in the 14-location taxonomy of MoveAtlas configs/canonical_imu.yaml), channels kept
  per site (channel_mask), acc_content and mag_unit as recorded, the sample_valid share, the non-finite share of
  the real channels, the median |acc| in g over valid samples, and the gravity test on the norm of the mean acc
  vector over valid samples (its DC part: gravity gives about 1 g, linear acceleration averages out), median over
  the record's sites: with_gravity in 0.7 to 1.3 g, gravity_removed below 0.3 g, ambiguous otherwise; per site
  the up axis (the signed axis of the largest component of the mean acc, an accelerometer at rest reading +1 g
  along up; kept only where that site's mean-acc norm is in 0.7 to 1.3 g) and the p99 |gyro| in the record's
  unit (contract deg/s);
- force: sensor_mask, unit and calibration as recorded, the sample_valid share, dropout_fraction, the peak of
  each foot total (channel 8) present in sensor_mask, and the non-finite share where sample_valid holds;
- pose: dim, modality, coord_mask, joints present, the valid_mask share over present joints, scale, and the up
  axis as recorded (signed axis of the median neck minus mid-ankle vector, joints 1, 13 and 14, over frames
  where the three are valid; in image coordinates for 2d_pixel records);
- EMG: channels, unit, unit_scale, signal_kind, the amplitude flag of notes, the p99 |signal|;
- the diagnosis field a record carries (pads condition, gaitrec label, stroke pose group), counted only.
Demographics: the source's own tables (pads raw/patients/patient_<id>.json; weargait_pd CONTROLS and PD tables
through mdfm.data.sources; physionet_gaitpdb demographics.txt, its Gender code decoded by the words of
demographics.html; gaitndd subject-description.txt; clinical_gait_multipatho the meta.json of the first
training trial of each subject; gaitrec GRF_metadata.csv, per field the first session holding a value, SEX
kept as its code; the stroke patients' MAT_normalizedData_PostStrokeAdults sub_char, whose Sub(i) is matched to
the subject numbers (TVC folder numbers, with gaps) by the correlation of Height with the median torso length
(pose scale) of the subject's records, as scripts/probe_stage1_sources.py did for the able-bodied file
(demo_stroke_patients); none on disk for imu_inthewild_iprognosis and modys_video). Only the persons of the
training portion are looked up, and no value of a lockbox person enters an output: the physionet_gaitpdb
Gender code is decoded from the training persons' rows alone, the pads source-table condition counts are over
the training persons alone, and each stroke Sub(i) mapping hypothesis correlates only the entries it assigns
to training subjects (a mapping is one to one, so no entry it assigns to a lockbox subject enters).
Diagnosis of the groups labelled 'all' (pads, imu_inthewild_iprognosis, modys_video): where it lives, and the
training-portion counts when it lives on the records; consistency of the group column with the record field
(gaitrec label, stroke pose group) and with clinical_gait_multipatho pathologyKey.
Orientation: per person and IMU site, the modal up axis of its records (orientation.csv; for
clinical_gait_multipatho with the cohort of its meta.json), so that a mounting convention shared by a cohort shows.
Weargait site completeness: per non-lockbox weargait_pd person and IMU recording, the sites listed and the share
of valid samples per site; a site is held in a recording when listed with at least 50 percent valid samples.
Proposed dev folds, drawn like mdfm.data.sources.folds: 20 percent of the persons rounded half up, numpy
default_rng seeded by the SHA-256 of "<seed>:<key>" (keys weargait_pd:PD and stroke_gait_vancriekinge:stroke),
over the sorted non-lockbox RELEASE persons of the group with an input record in the index (IMU, pose), the
first n of the permutation; each sorted person list hashed (sources.person_hash).
CPU discipline: the process sets its own priority to BelowNormal; while cache/runs/queue/SLOW exists it sleeps
10 minutes between two sources and logs each pause (pauses.csv and run.log).
Writes reports/stage2_sources/: sources.csv, fps_missing.csv, content.csv, shapes.csv, units.csv,
imu_sites.csv, orientation.csv, validity.csv, demographics.csv, diagnosis.csv, weargait_sites.csv,
weargait_site_records.csv, folds_proposed.csv, pauses.csv, run.log, summary.json, summary.md, _provenance.json
(scripts/moveatlas_stamp.py).
--compare OLD_DIR (with --out NEW_DIR, both holding a full run) reads no record: it writes compare.json and
compare.md in NEW_DIR with every output file compared byte for byte (the differing cells, JSON paths, run.log
lines and summary.md lines), the manifest rows of the sources in the manifest the 2026-10-01 run read
(OLD_MANIFEST) against the current one (identifier columns on every row, every column on the non-lockbox rows;
booleans and counts only), the modification times of the containers of the non-lockbox rows against the old
run's start, and the MoveAtlas commits between the two heads with the tracked files they change that name a
source, a pooled manifest or a file the probe reads (git log and git diff --name-only, read-only).
Run: CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe scripts/probe_stage2_sources.py
[--quick] [--out DIR] [--md-only: rewrite summary.md from the outputs of the last run] [--compare OLD_DIR]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm import paths  # noqa: E402
from mdfm.data import canonical as C  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from moveatlas_stamp import stamp  # noqa: E402

SOURCES = ("pads", "weargait_pd", "physionet_gaitpdb", "gaitndd", "clinical_gait_multipatho",
           "imu_inthewild_iprognosis", "modys_video", "stroke_gait_vancriekinge", "gaitrec")
COUNT_ONLY = ("koa_pd_nm",)
# design section 5, stage 2: training portion of persons after lockbox, of the source's persons
DESIGN = {"pads": (375, 469), "weargait_pd": (148, 185), "physionet_gaitpdb": (132, 165), "gaitndd": (51, 64),
          "clinical_gait_multipatho": (208, 260), "imu_inthewild_iprognosis": (36, 45), "modys_video": (27, 34),
          "stroke_gait_vancriekinge": (40, 50), "gaitrec": (1667, 2084), "koa_pd_nm": (77, 96)}
DESIGN_GROUPS = {"stroke_gait_vancriekinge": "stroke", "gaitrec": "not HC"}
CONTROLS = {("weargait_pd", "control"), ("stroke_gait_vancriekinge", "able_bodied"), ("gaitrec", "HC"),
            ("physionet_gaitpdb", "Co"), ("gaitndd", "control"), ("clinical_gait_multipatho", "HS")}
FOLDS = (("weargait_pd", "PD", "imu"), ("stroke_gait_vancriekinge", "stroke", "pose"))
FOLD_SHARE = 0.2
SEED = S.SEED
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
PAUSE_S = 600
HELD_SHARE = 0.5
GRAV = (0.7, 1.3)
NO_GRAV = 0.3
N_QUICK = 12
N_LOADCHECK = 3
ACC, GYRO, MAG = slice(0, 3), slice(3, 6), slice(6, 9)
NAN = float("nan")
# fields canonical.load adds from the index (canonical module docstring), left out of the record-field list
READER_FIELDS = ("manifest", "manifest_source", "person_id", "key", "tier", "lane", "scope", "lockbox")
OUTPUTS = sorted(["sources.csv", "fps_missing.csv", "content.csv", "shapes.csv", "units.csv", "imu_sites.csv",
                  "orientation.csv", "validity.csv", "demographics.csv", "diagnosis.csv", "weargait_sites.csv",
                  "weargait_site_records.csv", "folds_proposed.csv", "pauses.csv", "run.log", "summary.json",
                  "summary.md", "_provenance.json"])
# --compare: the manifest the 2026-10-01 run read (records_sha256 ca5a2d9eaada..., built against MoveAtlas f2e6f99;
# reports/provenance_history/summary.md), and the tracked MoveAtlas files the probe reads
OLD_MANIFEST = os.path.join(TREE, "cache", "manifest", "manifest_records_ca5a2d9eaada.csv.gz")
READ_TRACKED = ("configs/canonical_imu.yaml", "datasets/pose/modys_video/datacard.md")
MANIFEST_ID_COLS = ("source", "manifest_source", "modality", "subject_ns", "person_id", "key", "tier", "lane",
                    "scope", "lockbox", "group")
COMPARE_OUT = ("compare.json", "compare.md")
LOG = []


def set_below_normal():
    """BelowNormal priority for this process (Windows), so that training keeps the CPU."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetPriorityClass.restype = wintypes.BOOL
        k.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k.GetPriorityClass.restype = wintypes.DWORD
        h = k.GetCurrentProcess()
        return bool(k.SetPriorityClass(h, 0x00004000)) and k.GetPriorityClass(h) == 0x00004000
    os.nice(5)
    return True


def log(msg):
    """Print a line and append it to run.log of the output folder (LOG[0], opened by main)."""
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    if LOG:
        LOG[0].write(line + "\n")
        LOG[0].flush()


def locations():
    import yaml
    with open(f"{paths.MOVEATLAS}/configs/canonical_imu.yaml", encoding="utf-8") as f:
        block = re.search(r"^locations:\s*\[[^\]]*\]", f.read(), re.M).group(0)
    return yaml.safe_load(block)["locations"]


def num(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def r4(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.percentile(x, p)) if len(x) else NAN


def qs(x, ps=(10, 50, 90)):
    v = [r4(q(x, p)) for p in ps]
    return "" if all(a is None for a in v) else "|".join("" if a is None else f"{a:g}" for a in v)


def up_axis(u):
    u = np.asarray(u, dtype=np.float64)
    if not len(u) or not np.isfinite(u).all() or not np.abs(u).max() > 0:
        return ""
    i = int(np.argmax(np.abs(u)))
    return ("+" if u[i] > 0 else "-") + "xyz"[i]


def shape_of(a):
    return " x ".join(["T"] + [str(int(x)) for x in np.asarray(a).shape[1:]])


def channels_name(ch):
    out = []
    for name, sl in (("acc", ACC), ("gyro", GYRO), ("mag", MAG)):
        k = int(np.asarray(ch[sl], bool).sum())
        if k == 3:
            out.append(name)
        elif k:
            out.append(f"{name}({k}/3)")
    return "+".join(out) or "none"


def scope(source, group):
    if source == "weargait_pd":
        return group == "PD"
    if source == "stroke_gait_vancriekinge":
        return group == "stroke"
    if source == "gaitrec":
        return group != "HC"
    return True


def role(source, group):
    if (source, group) in CONTROLS:
        return "control"
    if group == "all":
        return "not in the index (diagnosis.csv)"
    return "patient"


# ---------------------------------------------------------------- reading
def selection(idx, source):
    """The rows canonical.iterate(sources=source, lane='body') selects with its flags at False."""
    named = (idx["source"] == source) | (idx["manifest_source"] == source)
    sel = (named & (idx["container"] != "") & (idx["lane"] == "body") & ~idx["lockbox"]
           & ~idx["tier"].isin(["EVAL", "EXCLUDED_V1"]) & ~idx["cody"])
    rows = idx[sel].sort_values(["container", "record"], kind="stable")
    return rows[~rows.duplicated(["container", "record"])]


def read_source(idx, source, quick, stats):
    """(row, record) pairs of the training portion of a source (module docstring)."""
    if quick:
        rows = selection(idx, source).groupby(["modality", "group"], sort=True).head(N_QUICK)
        for row in rows.to_dict("records"):
            try:
                yield row, C.load(row)
            except C.Refused:                    # a refusal is never counted as a read error
                raise
            except (ValueError, LookupError, OSError) as e:
                stats["load_errors"] += 1
                stats["load_error_last"] = f"{type(e).__name__}: {e}"[:300]
        return
    done = set()
    try:
        for row, rec in C.iterate(sources=source, lane="body"):
            done.add((row["container"], int(row["record"])))
            yield row, rec
    except C.Refused:                            # PermissionError, hence OSError: re-raised, never read around
        raise
    except (ValueError, LookupError, OSError) as e:
        stats["iterate_stopped"] = f"{type(e).__name__}: {e}"[:300]
        for row in selection(idx, source).to_dict("records"):
            if (row["container"], int(row["record"])) in done:
                continue
            try:
                yield row, C.load(row)
            except C.Refused:
                raise
            except (ValueError, LookupError, OSError) as e2:
                stats["load_errors"] += 1
                stats["load_error_last"] = f"{type(e2).__name__}: {e2}"[:300]


def reader_checks(idx, source, read_rows):
    """canonical.load refuses a lockbox row of the source; canonical.load returns the arrays iterate returned."""
    out = {"lockbox_rows_tried": 0, "lockbox_rows_refused": 0, "load_rows_compared": 0, "load_rows_equal": 0}
    lb = idx[(idx["source"] == source) & idx["lockbox"]]
    if len(lb):
        out["lockbox_rows_tried"] = 1
        try:
            C.load(lb.iloc[0].to_dict())
        except C.Refused:
            out["lockbox_rows_refused"] = 1
    for row, main in read_rows[:N_LOADCHECK]:
        rec = C.load(row)
        out["load_rows_compared"] += 1
        a = np.asarray(rec[C.MAIN_ARRAY[row["modality"]]])
        out["load_rows_equal"] += int(a.shape == main.shape and np.array_equal(a, main, equal_nan=True))
    return out


# ---------------------------------------------------------------- per-record facts
def base_facts(row, rec, n):
    fps = num(rec.get("fps"))
    fps_i = num(row.get("fps"))
    return {"source": row["source"], "group": row["group"], "modality": row["modality"],
            "subject_ns": row["subject_ns"], "person_id": row["person_id"], "key": row["key"],
            "recording": S.recording_key(row["source"], row["modality"], row["key"]),
            "n": n, "n_index": int(row["n_frames"]), "fps": fps, "fps_index": fps_i,
            "hours": n / fps / 3600 if fps > 0 else NAN,
            "hours_index": int(row["n_frames"]) / fps_i / 3600 if fps_i > 0 else NAN}


def diag_facts(rec):
    for k in ("condition", "group", "cohort", "diagnosis", "label"):
        v = rec.get(k)
        if v is None or isinstance(v, np.ndarray) or isinstance(v, (list, dict)):
            continue
        return {"diag_field": k, "diag_value": str(v)}
    return {"diag_field": "", "diag_value": ""}


def imu_facts(rec, sites_names):
    x = np.asarray(rec["signal"])
    cm = np.asarray(rec["channel_mask"], bool)
    ids = [int(i) for i in np.asarray(rec["location_ids"]).ravel()]
    T, L = x.shape[0], x.shape[1]
    has_sv = rec.get("sample_valid") is not None
    sv = np.asarray(rec["sample_valid"], bool).reshape(T, L) if has_sv else np.ones((T, L), bool)
    real = np.broadcast_to(cm[None], x.shape)
    fin = np.isfinite(x)
    per_site, norms, means = [], [], []
    for j, lid in enumerate(ids):
        site = sites_names[lid] if 0 <= lid < len(sites_names) else f"id{lid}"
        d = {"site": site, "channels": channels_name(cm[j]), "valid_share": float(sv[:, j].mean()) if T else NAN,
             "acc_norm_med": NAN, "up": "", "acc_mean_norm": NAN, "gyro_p99": NAN, "mag_norm_med": NAN}
        if cm[j, ACC].all():
            a = x[:, j, ACC].astype(np.float64)
            ok = sv[:, j] & np.isfinite(a).all(axis=1)
            if ok.any():
                nrm = np.linalg.norm(a[ok], axis=1)
                d["acc_norm_med"] = float(np.median(nrm))
                m = a[ok].mean(axis=0)
                d["acc_mean_norm"] = float(np.linalg.norm(m))
                if GRAV[0] <= d["acc_mean_norm"] <= GRAV[1]:
                    d["up"] = up_axis(m)
                norms.append(nrm)
                means.append(d["acc_mean_norm"])
        if cm[j, GYRO].all():
            g = x[:, j, GYRO]
            ok = sv[:, j] & np.isfinite(g).all(axis=1)
            if ok.any():
                d["gyro_p99"] = float(np.percentile(np.abs(g[ok]), 99))
        if cm[j, MAG].all():
            mg = x[:, j, MAG].astype(np.float64)
            ok = sv[:, j] & np.isfinite(mg).all(axis=1)
            if ok.any():
                d["mag_norm_med"] = float(np.median(np.linalg.norm(mg[ok], axis=1)))
        per_site.append(d)
    acc_med = float(np.median(np.concatenate(norms))) if norms else NAN
    acc_dc = float(np.median(means)) if means else NAN
    cls = ("" if not np.isfinite(acc_dc) else "with_gravity" if GRAV[0] <= acc_dc <= GRAV[1]
           else "gravity_removed" if acc_dc < NO_GRAV else "ambiguous")
    sig = {"sites": "|".join(d["site"] for d in per_site),
           "channels_kept": "|".join(sorted({d["channels"] for d in per_site})),
           "acc_content": rec.get("acc_content") or "absent (contract default: with gravity)",
           "mag_unit": rec.get("mag_unit") or "", "sample_valid": has_sv}
    f = {"shape": shape_of(x), "sig": json.dumps(sig, sort_keys=True), "has_mask": has_sv,
         "valid_n": int(sv.sum()), "valid_total": int(sv.size), "any_invalid": bool(has_sv and not sv.all()),
         "nonfinite": int((~fin & real).sum()), "nreal": int(real.sum()),
         "nonfinite_where_valid": int((~fin & real & sv[:, :, None]).sum()),
         "acc_med": acc_med, "acc_dc": acc_dc, "acc_class": cls, "acc_content": sig["acc_content"],
         "mag_unit": sig["mag_unit"],
         "sites": sig["sites"], "n_sites": len(per_site),
         "gyro_p99": q([d["gyro_p99"] for d in per_site], 50),
         "dropout_fraction": num(rec.get("dropout_fraction"))}
    return f, per_site


def force_facts(rec):
    x = np.asarray(rec["force"])
    m = np.asarray(rec["sensor_mask"], bool)
    T = x.shape[0]
    has_sv = rec.get("sample_valid") is not None
    sv = np.asarray(rec["sample_valid"], bool).reshape(T, -1) if has_sv else np.ones((T, x.shape[1]), bool)
    real = np.broadcast_to(m[None], x.shape)
    fin = np.isfinite(x)
    peaks = {}
    for j, side in enumerate("LR"):
        tot = x[:, j, 8].astype(np.float64)
        ok = np.isfinite(tot) & sv[:, j]
        peaks[side] = float(tot[ok].max()) if m[j, 8] and ok.any() else NAN
    sig = {"sensor_mask": "|".join("".join("1" if b else "0" for b in row) for row in m.tolist()),
           "unit": rec.get("unit") or "absent (contract: N)", "calibration": rec.get("calibration") or "",
           "sample_valid": has_sv}
    return {"shape": shape_of(x), "sig": json.dumps(sig, sort_keys=True), "has_mask": has_sv,
            "valid_n": int(sv.sum()), "valid_total": int(sv.size), "any_invalid": bool(has_sv and not sv.all()),
            "nonfinite": int((~fin & real).sum()), "nreal": int(real.sum()),
            "nonfinite_where_valid": int((~fin & real & sv[:, :, None]).sum()),
            "unit": sig["unit"], "peak_L": peaks["L"], "peak_R": peaks["R"],
            "dropout_fraction": num(rec.get("dropout_fraction"))}


def pose_facts(rec):
    c = np.asarray(rec["coords"], dtype=np.float64)
    jm = np.asarray(rec["joint_mask"], bool)
    cm = np.asarray(rec["coord_mask"], bool)
    vm = np.asarray(rec["valid_mask"], bool)
    T = c.shape[0]
    ok = vm[:, 1] & vm[:, 13] & vm[:, 14]
    u = np.median(c[ok, 1] - 0.5 * (c[ok, 13] + c[ok, 14]), axis=0) if ok.any() else np.full(3, NAN)
    u = np.where(cm, u, 0.0)
    real = vm[:, jm]
    fin = np.isfinite(c[:, jm][:, :, cm])
    sig = {"dim": rec.get("dim"), "pose_space": rec.get("modality"),
           "coord_mask": "".join("1" if b else "0" for b in cm.tolist()), "joints_present": int(jm.sum())}
    return {"shape": shape_of(c), "sig": json.dumps(sig, sort_keys=True), "has_mask": True,
            "valid_n": int(real.sum()), "valid_total": int(real.size), "any_invalid": bool(not real.all()),
            "nonfinite": int((~fin).sum()), "nreal": int(fin.size),
            "nonfinite_where_valid": int((~fin & real[:, :, None]).sum()),
            "pose_modality": rec.get("modality"), "dim": rec.get("dim"), "scale": num(rec.get("scale")),
            "up": up_axis(u), "up_frames": int(ok.sum()), "T": T}


def emg_facts(rec):
    x = np.asarray(rec["signal"])
    ch = rec.get("channels") or []
    fin = np.isfinite(x)
    sig = {"n_channels": int(x.shape[1]) if x.ndim == 2 else None, "unit": rec.get("unit"),
           "unit_scale": rec.get("unit_scale"), "signal_kind": rec.get("signal_kind"),
           "kinds": "|".join(sorted({str(c.get("kind")) for c in ch if isinstance(c, dict)}))}
    return {"shape": shape_of(x), "sig": json.dumps(sig, sort_keys=True), "has_mask": False,
            "valid_n": 0, "valid_total": 0, "any_invalid": False,
            "nonfinite": int((~fin).sum()), "nreal": int(fin.size), "nonfinite_where_valid": int((~fin).sum()),
            "unit": str(rec.get("unit")), "emg_p99": float(np.nanpercentile(np.abs(x), 99)) if fin.any() else NAN,
            "amp_flag": "AMPLITUDE FLAG" in str(rec.get("notes") or "")}


def measure(row, rec, sites_names):
    """(facts, per-site rows, main array) of one record."""
    mod = row["modality"]
    main = np.asarray(rec[C.MAIN_ARRAY[mod]])
    f = base_facts(row, rec, int(main.shape[0]))
    f.update(diag_facts(rec))
    per_site = []
    if mod == "imu":
        g, per_site = imu_facts(rec, sites_names)
    elif mod == "force":
        g = force_facts(rec)
    elif mod == "pose":
        g = pose_facts(rec)
    elif mod == "emg":
        g = emg_facts(rec)
    else:
        g = {"shape": shape_of(main), "sig": "{}"}
    f.update(g)
    return f, per_site, main


# ---------------------------------------------------------------- pauses
def pause_if_slow(after, before, pauses, out):
    if not os.path.exists(SLOW):
        return
    t0 = dt.datetime.now()
    log(f"PAUSE {PAUSE_S} s between {after} and {before}: {SLOW} exists")
    time.sleep(PAUSE_S)
    t1 = dt.datetime.now()
    pauses.append({"started": t0.isoformat(timespec="seconds"), "ended": t1.isoformat(timespec="seconds"),
                   "seconds": round((t1 - t0).total_seconds(), 1), "after_source": after, "before_source": before,
                   "slow_file": "cache/runs/queue/SLOW", "slow_still_present": os.path.exists(SLOW)})
    pd.DataFrame(pauses).to_csv(os.path.join(out, "pauses.csv"), index=False, lineterminator="\n")
    log(f"PAUSE end after {pauses[-1]['seconds']} s")


# ---------------------------------------------------------------- demographics
def _sex(v):
    s = str(v).strip().lower()
    return {"male": "M", "m": "M", "female": "F", "f": "F"}.get(s, "")


def demo_pads(subjects):
    out = {}
    for s in subjects:
        p = f"{paths.MOVEATLAS}/datasets/imu/pads/raw/patients/patient_{s.split(':', 1)[1]}.json"
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                j = json.load(fh)
            out[s] = {"sex": _sex(j.get("gender")), "age": num(j.get("age")), "mass_kg": num(j.get("weight")),
                      "height_m": num(j.get("height")) / 100.0}
    return out, "datasets/imu/pads/raw/patients/patient_<id>.json (gender, age, weight kg, height cm)"


def demo_sources(source, subjects):
    out = {}
    for s in subjects:
        d = S.demographic(source, s)
        if d:
            out[s] = {"sex": d.get("sex") or "", "age": num(d.get("age")), "mass_kg": num(d.get("mass_kg")),
                      "height_m": NAN}
    files = " + ".join(S.DEMOGRAPHIC_FILES[source])
    return out, f"datasets/{files} (mdfm.data.sources.demographic: sex, age, mass; height not read)"


def gaitpdb_sex_codes(z, member_html, table):
    """{code: 'M' or 'F'} from the words of demographics.html against the codes of demographics.txt."""
    text = re.sub(r"<[^>]+>", " ", z.read(member_html).decode("latin-1"))
    tok = text.split()
    words = {}
    for i, t in enumerate(tok[:-4]):
        if re.fullmatch(r"(Ga|Ju|Si)(Co|Pt)\d+", t) and tok[i + 4].lower() in ("male", "female"):
            words[t] = _sex(tok[i + 4])
    pairs = collections.Counter((r["Gender"].strip(), words[r["ID"]]) for r in table if r["ID"] in words)
    codes = {}
    for (code, sex), n in pairs.items():
        codes.setdefault(code, collections.Counter())[sex] += n
    return {c: v.most_common(1)[0][0] for c, v in codes.items() if len(v) == 1}, dict(
        (f"{c}->{s}", n) for (c, s), n in pairs.items())


def demo_gaitpdb(subjects):
    p = f"{paths.MOVEATLAS}/datasets/imu/physionet_gaitpdb/raw/gaitpdb.zip"
    base = "gait-in-parkinsons-disease-1.0.0/"
    want = {s.split(":", 1)[1] for s in subjects}
    with zipfile.ZipFile(p) as z:
        lines = z.read(base + "demographics.txt").decode("latin-1").splitlines()
        rdr = csv.reader(lines, delimiter="\t")
        head = [h.strip() for h in next(rdr)]
        # training persons only: the Gender code is decoded, and its evidence counted, without a lockbox person
        table = [d for d in (dict(zip(head, [c.strip() for c in r])) for r in rdr if r and r[0].strip())
                 if d.get("ID") in want]
        codes, evidence = gaitpdb_sex_codes(z, base + "demographics.html", table)
    by = {r["ID"]: r for r in table}
    out = {}
    for s in subjects:
        r = by.get(s.split(":", 1)[1])
        if r:
            out[s] = {"sex": codes.get(r.get("Gender", "").strip(), ""), "age": num(r.get("Age")),
                      "mass_kg": num(r.get("Weight")), "height_m": num(r.get("Height"))}
    return out, (f"datasets/imu/physionet_gaitpdb/raw/gaitpdb.zip {base}demographics.txt (Gender code by the words "
                 f"of demographics.html: {sorted(codes.items())}, pairs {evidence})")


def demo_gaitndd(subjects):
    p = f"{paths.MOVEATLAS}/datasets/imu/gaitndd/raw/gaitndd.zip"
    m = "gait-in-neurodegenerative-disease-database-1.0.0/subject-description.txt"
    with zipfile.ZipFile(p) as z:
        lines = z.read(m).decode("latin-1").splitlines()
    rdr = csv.reader(lines, delimiter="\t")
    head = [h.strip() for h in next(rdr)]
    want = {s.split(":", 1)[1] for s in subjects}
    by = {}
    for r in rdr:
        if r and r[0].strip() in want:                   # training persons only
            by[r[0].strip()] = dict(zip(head, [c.strip() for c in r]))
    out = {}
    for s in subjects:
        r = by.get(s.split(":", 1)[1])
        if r:
            out[s] = {"sex": _sex(r.get("gender")), "age": num(r.get("AGE(YRS)")), "mass_kg": num(r.get("Weight(kg)")),
                      "height_m": num(r.get("HEIGHT(meters)"))}
    return out, f"datasets/imu/gaitndd/raw/gaitndd.zip {m}"


def demo_clinical(subjects, keys_by_subject):
    p = f"{paths.MOVEATLAS}/datasets/imu/clinical_gait_multipatho/raw/dataset.zip"
    out, path_keys = {}, {}
    with zipfile.ZipFile(p) as z:
        metas = {os.path.basename(n)[: -len("_meta.json")]: n for n in z.namelist() if n.endswith("_meta.json")}
        for s in subjects:
            keys = sorted(keys_by_subject.get(s, []))
            hit = next((k for k in keys if k in metas), None)
            if hit is None:
                continue
            j = json.loads(z.read(metas[hit]).decode("utf-8"))
            out[s] = {"sex": _sex(j.get("gender")), "age": num(j.get("age")), "mass_kg": num(j.get("weight")),
                      "height_m": num(j.get("height"))}
            path_keys[s] = (str(j.get("pathologyKey")), str(j.get("group")))
    return out, "datasets/imu/clinical_gait_multipatho/raw/dataset.zip <subject>_<trial>_meta.json", path_keys


def demo_gaitrec(subjects):
    p = f"{paths.MOVEATLAS}/datasets/imu/gaitrec/raw/GRF_metadata.csv"
    want = {s.split(":", 1)[1]: s for s in subjects}
    out = {}
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = want.get(r["SUBJECT_ID"].strip())
            if s is None:
                continue
            code = r.get("SEX", "").strip()
            new = {"sex": f"code {code}" if code not in ("", "NaN") else "", "age": num(r.get("AGE")),
                   "mass_kg": num(r.get("BODY_MASS")), "height_m": num(r.get("HEIGHT")) / 100.0}
            old = out.setdefault(s, new)
            missing = lambda v: v == "" or (isinstance(v, float) and not np.isfinite(v))  # noqa: E731
            for k, v in new.items():                 # per field, the first session holding a value
                if missing(old[k]) and not missing(v):
                    old[k] = v
    return out, "datasets/imu/gaitrec/raw/GRF_metadata.csv (per field the first session with a value; SEX as its code)"


STROKE_MAT = "pose/stroke_gait_vancriekinge/raw/MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"


def stroke_sub_char():
    import h5py
    out = {}
    with h5py.File(f"{paths.MOVEATLAS}/datasets/{STROKE_MAT}", "r") as f:
        refs = f["Sub"]["sub_char"]
        for i in range(refs.shape[0]):
            g = f[refs[i, 0]]
            out[i + 1] = {k: float(np.asarray(g[k]).ravel()[0]) for k in ("Male", "Age", "Weight", "Height") if k in g}
    return out


def demo_stroke_patients(subjects, torso, all_numbers):
    """Which MAT Sub(i) is stroke subject S<n> (n: the TVC folder number of the raw zip, with gaps)? Two
    hypotheses, each with shifts -2 to 2: i = n (number) and i = rank of n among the sorted subject numbers of
    the index (rank, the 50 folders in order). Each is scored by corr(sub_char Height, median torso length of
    the subject's pose records) over the training subjects; the coverage uses the best mapping when its
    correlation is at least 0.5 and exceeds every other by 0.2, and no mapping otherwise."""
    sub = stroke_sub_char()
    rank = {n: i + 1 for i, n in enumerate(sorted(all_numbers))}
    t = {int(s.rsplit(":S", 1)[1]): v for s, v in torso.items() if np.isfinite(v)}
    maps = {}
    for name, f in (("number", lambda n: n), ("rank", lambda n: rank.get(n))):
        for shift in (-2, -1, 0, 1, 2):
            maps[f"{name}{shift:+d}"] = {n: (f(n) + shift if f(n) is not None else None) for n in t}
    corr = {}
    for name, m in maps.items():
        ids = [n for n in sorted(t) if m[n] in sub and np.isfinite(sub[m[n]].get("Height", NAN))]
        if len(ids) > 2:
            corr[name] = r4(np.corrcoef([t[n] for n in ids], [sub[m[n]]["Height"] for n in ids])[0, 1])
    ranked = sorted(((v, k) for k, v in corr.items() if v is not None), reverse=True)
    best = ranked[0][1] if ranked and ranked[0][0] >= 0.5 and (
        len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.2) else None
    out = {}
    if best is not None:
        name, shift = re.fullmatch(r"(number|rank)([+-]\d)", best).groups()
        for s in subjects:
            n = int(s.rsplit(":S", 1)[1])
            i = (n if name == "number" else rank.get(n, -99)) + int(shift)
            v = sub.get(i)
            if v:
                out[s] = {"sex": {1.0: "M", 0.0: "F"}.get(v.get("Male"), ""), "age": v.get("Age", NAN),
                          "mass_kg": v.get("Weight", NAN), "height_m": v.get("Height", NAN) / 1000.0}
    return out, f"datasets/{STROKE_MAT} (Sub(i).sub_char: Male, Age, Weight kg, Height mm)", {
        "mat_subjects": len(sub), "index_subject_numbers": len(rank), "training_subjects_with_torso": len(t),
        "subject_numbers_min_max": [min(rank), max(rank)] if rank else None,
        "corr_height_torso_by_mapping": corr, "mapping_used": best}


def demographics(F, idx, extra):
    """demographics.csv rows: per source and group of the training portion (module docstring)."""
    rows = []
    subj = F.groupby(["source", "group"])["subject_ns"].agg(lambda x: sorted(set(x))).to_dict()
    person_of = dict(zip(F["subject_ns"], F["person_id"]))
    keys = F.groupby("subject_ns")["key"].agg(list).to_dict()
    torso = (F[(F["source"] == "stroke_gait_vancriekinge") & (F["modality"] == "pose")]
             .groupby("subject_ns")["scale"].median().to_dict())
    cache = {}
    for (source, group), subjects in sorted(subj.items()):
        table, note = "", ""
        if source == "pads":
            d, table = demo_pads(subjects)
        elif source == "weargait_pd" or (source == "stroke_gait_vancriekinge" and group == "able_bodied"):
            d, table = demo_sources(source, subjects)
        elif source == "physionet_gaitpdb":
            d, table = demo_gaitpdb(subjects)
        elif source == "gaitndd":
            d, table = demo_gaitndd(subjects)
        elif source == "clinical_gait_multipatho":
            d, table, pk = demo_clinical(subjects, keys)
            cache.setdefault("clinical_pathology_keys", {}).update(pk)
        elif source == "gaitrec":
            d, table = demo_gaitrec(subjects)
        elif source == "stroke_gait_vancriekinge" and group == "stroke":
            numbers = {int(x.rsplit(":S", 1)[1]) for x in idx.loc[(idx["source"] == source)
                                                                  & (idx["group"] == "stroke"), "subject_ns"]}
            d, table, chk = demo_stroke_patients(subjects, {s: torso.get(s, NAN) for s in subjects}, numbers)
            extra["stroke_patient_mapping"] = chk
            note = (f"Sub(i) mapping used: {chk['mapping_used'] or 'none supported, no person matched'} "
                    "(summary.md, Demographics)")
        elif source == "imu_inthewild_iprognosis":
            d, table = {}, "none on disk per subject (the raw pickle annotation holds UPDRS items and pd_status only)"
        elif source == "modys_video":
            d, table = {}, "none on disk per child (data_clinical_scoring.xlsx holds ID, video, time, scorer, scores)"
        else:
            d, table = {}, "not looked up"
        persons = sorted({person_of[s] for s in subjects})
        matched = {person_of[s]: v for s, v in d.items()}
        cache.setdefault("per_person", {}).update(matched)
        vals = list(matched.values())
        ages = [v["age"] for v in vals if np.isfinite(v["age"])]
        sex = collections.Counter(v["sex"] or "unknown" for v in vals)
        rows.append({"source": source, "group": group, "stage2_scope": scope(source, group), "table": table,
                     "persons": len(persons), "matched": len(matched),
                     "sex_known": sum(1 for v in vals if v["sex"]),
                     "sex_values": "|".join(f"{k} {n}" for k, n in sorted(sex.items())),
                     "age_known": len(ages), "age_min_median_max": qs(ages, (0, 50, 100)),
                     "mass_known": sum(1 for v in vals if np.isfinite(v["mass_kg"])),
                     "height_known": sum(1 for v in vals if np.isfinite(v["height_m"])), "note": note})
    return pd.DataFrame(rows), cache


# ---------------------------------------------------------------- diagnosis
def diagnosis(F, idx, cache):
    rows = []
    pads = F[F["source"] == "pads"]
    per = pads.groupby("person_id")["diag_value"].agg(lambda x: sorted(set(x)))
    multi = int((per.map(len) > 1).sum())
    table = pd.read_csv(f"{paths.MOVEATLAS}/datasets/imu/pads/processed/pads_subject_table.csv", dtype=str,
                        keep_default_na=False, encoding="utf-8")
    table = table[table["subject_ns"].isin(set(pads["subject_ns"]))]        # training persons only
    tab = table["condition"].value_counts().to_dict()
    for value, g in pads.groupby("diag_value"):
        rows.append({"source": "pads", "group": "all", "diagnosis_source":
                     "record field condition (ingest_pads.py, from raw/patients/patient_<id>.json)",
                     "value": value, "training_persons": g["person_id"].nunique(), "training_records": len(g),
                     "source_table_training_persons": tab.get(value, 0),
                     "note": "source table processed/pads_subject_table.csv, rows of the training persons only "
                             f"({len(table)}); persons with two values: {multi}"})
    ipr = F[F["source"] == "imu_inthewild_iprognosis"]
    with zipfile.ZipFile(f"{paths.MOVEATLAS}/datasets/imu/imu_inthewild_iprognosis/raw/"
                         "i-PROGNOSIS_DS2.10_Sub1_IMUdataPhonecalls.zip") as z:
        desc = z.read("dataset_description.txt").decode("utf-8", "replace")
    m = re.search(r"(\d+)\s+PD patients\s+and\s+(\d+)\s+Healthy controls", " ".join(desc.split()))
    stated = f"{m.group(1)} PD and {m.group(2)} healthy controls" if m else "not found"
    rows.append({"source": "imu_inthewild_iprognosis", "group": "all", "diagnosis_source":
                 "raw tremor_dataset.pickle annotation pd_status per subject (dataset_description.txt of the raw zip); "
                 "not carried by the canonical records or the index; not read here (a 1.9 GB pickle)",
                 "value": "unknown per person", "training_persons": ipr["person_id"].nunique(),
                 "training_records": len(ipr), "source_table_training_persons": "", "note": f"stated by the source: {stated}; "
                 f"record diagnosis fields: {sorted(set(ipr['diag_field']) - {''}) or 'none'}"})
    mod = F[F["source"] == "modys_video"]
    with open(f"{paths.MOVEATLAS}/datasets/pose/modys_video/datacard.md", encoding="utf-8") as fh:
        pop = next((ln.strip()[len("- population:"):].strip() for ln in fh if ln.startswith("- population:")), "")
    rows.append({"source": "modys_video", "group": "all", "diagnosis_source":
                 "the source population (every child has dyskinetic cerebral palsy; datacard population line); no "
                 "per-child field", "value": "dyskinetic cerebral palsy (all)",
                 "training_persons": mod["person_id"].nunique(), "training_records": len(mod),
                 "source_table_training_persons": "", "note": f"datacard: {pop[:160]}; record diagnosis fields: "
                 f"{sorted(set(mod['diag_field']) - {''}) or 'none'}"})
    # consistency of the group column with what the records or the raw metadata carry
    gr = F[F["source"] == "gaitrec"]
    rows.append({"source": "gaitrec", "group": "(each)", "diagnosis_source": "record field label (CLASS_LABEL) "
                 "against the index group (GRF_metadata.csv CLASS_LABEL)", "value": "label equals group",
                 "training_persons": gr["person_id"].nunique(), "training_records": len(gr),
                 "agreeing": f"{int((gr['diag_value'] == gr['group']).sum())} records",
                 "source_table_training_persons": "", "note": ""})
    st = F[(F["source"] == "stroke_gait_vancriekinge") & (F["modality"] == "pose")]
    rows.append({"source": "stroke_gait_vancriekinge", "group": "(each)", "diagnosis_source": "pose record field "
                 "group against the index group (namespace of subject_ns)", "value": "group equals group",
                 "training_persons": st["person_id"].nunique(), "training_records": len(st),
                 "agreeing": f"{int((st['diag_value'] == st['group']).sum())} records", "source_table_training_persons": "",
                 "note": "pose records only (the force and EMG records carry no group field)"})
    pk = cache.get("clinical_pathology_keys", {})
    cg = F[F["source"] == "clinical_gait_multipatho"]
    subj = cg.drop_duplicates("subject_ns")
    agree = sum(pk.get(s, ("",))[0] == g for s, g in zip(subj["subject_ns"], subj["group"]))
    rows.append({"source": "clinical_gait_multipatho", "group": "(each)", "diagnosis_source": "pathologyKey of the "
                 "first training trial's meta.json against the index group (id prefix)", "value": "pathologyKey equals group",
                 "training_persons": cg["person_id"].nunique(), "training_records": len(cg),
                 "agreeing": f"{agree} of {len(subj)} subjects", "source_table_training_persons": "",
                 "note": f"subjects without a meta.json: {len(subj) - len(pk)}"})
    return pd.DataFrame(rows, columns=["source", "group", "diagnosis_source", "value", "training_persons",
                                       "training_records", "agreeing", "source_table_training_persons", "note"])


# ---------------------------------------------------------------- weargait sites
def weargait_sites(site_rows, F, idx, sites_names, folds_prop):
    w = site_rows[site_rows["source"] == "weargait_pd"].copy()
    w["held"] = w["valid_share"] >= HELD_SHARE
    force_keys = set(zip(*[idx.loc[(idx["source"] == "weargait_pd") & (idx["modality"] == "force") & ~idx["lockbox"], c]
                           for c in ("subject_ns", "key")]))
    all13 = [s for s in sites_names if s != "pocket"]
    rec_rows = []
    for (pid, sns, key, group), g in w.groupby(["person_id", "subject_ns", "key", "group"], sort=True):
        listed = list(g["site"])
        held = list(g.loc[g["held"], "site"])
        rec_rows.append({"person_id": pid, "group": group, "key": key, "n_sites_listed": len(listed),
                         "n_sites_held": len(held), "sites_held": "|".join(held),
                         "sites_missing": "|".join(s for s in all13 if s not in listed),
                         "sites_listed_not_held": "|".join(s for s in listed if s not in held),
                         "min_site_valid_share": r4(g["valid_share"].min()),
                         "insole_record": (sns, key) in force_keys})
    R = pd.DataFrame(rec_rows)
    try:
        st1 = S.folds()
        st1 = dict(zip(st1.loc[st1["source"] == "weargait_pd", "person_id"], st1.loc[st1["source"] == "weargait_pd", "fold"]))
    except Exception as e:  # noqa: BLE001
        st1 = {}
        log(f"stage-1 folds unavailable: {e}")
    prop = dict(zip(folds_prop["person_id"], folds_prop["fold"])) if len(folds_prop) else {}
    P = []
    for (pid, group), g in R.groupby(["person_id", "group"], sort=True):
        held_sets = [set(x.split("|")) - {""} for x in g["sites_held"]]
        every = set.intersection(*held_sets) if held_sets else set()
        some = set.union(*held_sets) if held_sets else set()
        cnt = collections.Counter(s for hs in held_sets for s in hs)
        P.append({"person_id": pid, "group": group,
                  "stage1_fold": st1.get(pid, "") if group == "control" else "",
                  "proposed_fold": prop.get(pid, "") if group == "PD" else "",
                  "recordings": len(g), "recordings_all_13_held": int((g["n_sites_held"] == len(all13)).sum()),
                  "n_sites_held_in_every_recording": len(every),
                  "sites_never_held": "|".join(s for s in all13 if s not in some),
                  "sites_held_in_some_not_all": "|".join(f"{s} {cnt[s]}" for s in all13 if s in some and s not in every),
                  "layouts": len({x for x in g["sites_held"]}),
                  "insole_recordings": int(g["insole_record"].sum())})
    return R, pd.DataFrame(P)


# ---------------------------------------------------------------- folds
def proposed_folds(idx):
    out, info = [], {}
    for source, group, modality in FOLDS:
        sel = ((idx["source"] == source) & (idx["group"] == group) & (idx["modality"] == modality)
               & (idx["tier"] == "RELEASE") & ~idx["lockbox"] & (idx["container"] != ""))
        persons = sorted(set(idx.loc[sel, "person_id"]))
        n_hold = S._round_half_up(FOLD_SHARE * len(persons))
        key = f"{source}:{group}"
        rng = np.random.default_rng(S._seed(SEED, key))
        hold = {persons[i] for i in rng.permutation(len(persons))[:n_hold]}
        out += [{"source": source, "group": group, "person_id": p, "fold": "dev" if p in hold else "train"}
                for p in persons]
        lock = set(idx.loc[(idx["source"] == source) & idx["lockbox"], "person_id"])
        info[key] = {"pool": f"non-lockbox RELEASE persons of group {group} with a {modality} row in the index",
                     "persons": len(persons), "dev": len(hold), "train": len(persons) - len(hold),
                     "seed": SEED, "seed_key": key, "share": FOLD_SHARE,
                     "sha256_pool": S.person_hash(persons), "sha256_dev": S.person_hash(sorted(hold)),
                     "sha256_train": S.person_hash(sorted(set(persons) - hold)),
                     "dev_in_lockbox": len(hold & lock)}
    return pd.DataFrame(out, columns=["source", "group", "person_id", "fold"]), info


def fold_balance(folds, info, idx, per_person):
    """Per proposed fold: persons with a non-lockbox row of each modality of the source, and the sex and age of
    the persons the demographic tables match (demographics.csv), dev against train."""
    for key, v in info.items():
        source, group = key.split(":", 1)
        f = folds[(folds["source"] == source) & (folds["group"] == group)]
        rows = idx[(idx["source"] == source) & (idx["group"] == group) & ~idx["lockbox"]]
        for fold in ("dev", "train"):
            ps = set(f.loc[f["fold"] == fold, "person_id"])
            v[f"{fold}_persons_with"] = {m: len(set(rows.loc[rows["modality"] == m, "person_id"]) & ps)
                                         for m in sorted(set(rows["modality"]))}
            d = [per_person[p] for p in sorted(ps) if p in per_person]
            sex = collections.Counter(x["sex"] or "unknown" for x in d)
            v[f"{fold}_sex"] = dict(sorted(sex.items()))
            v[f"{fold}_age_median"] = r4(q([x["age"] for x in d], 50))
            v[f"{fold}_matched"] = len(d)


# ---------------------------------------------------------------- tables
def index_counts(idx, sources):
    d = idx[idx["source"].isin(sources)]
    g = d.groupby(["source", "group", "modality"])
    t = pd.DataFrame({"index_rows": g.size(), "index_persons": g["person_id"].nunique(),
                      "lockbox_rows": g["lockbox"].sum(),
                      "lockbox_persons": g.apply(lambda x: x.loc[x["lockbox"], "person_id"].nunique()),
                      "nonlockbox_rows": g.apply(lambda x: int((~x["lockbox"]).sum())),
                      "nonlockbox_persons": g.apply(lambda x: x.loc[~x["lockbox"], "person_id"].nunique()),
                      "tier": g["tier"].agg(lambda x: "|".join(sorted(set(x))))})
    return t.reset_index()


def fps_text(x):
    v = sorted({round(float(a), 4) for a in x if np.isfinite(a)})
    if not v:
        return ""
    if len(v) <= 4:
        return "|".join(f"{a:g}" for a in v)
    return f"{v[0]:g} to {v[-1]:g} ({len(v)} values, median {np.median([a for a in x if np.isfinite(a)]):g})"


def source_table(F, idx, stats, recovered):
    ic = index_counts(idx, SOURCES + COUNT_ONLY)
    rows = []
    for r in ic.to_dict("records"):
        s, g, m = r["source"], r["group"], r["modality"]
        f = F[(F["source"] == s) & (F["group"] == g) & (F["modality"] == m)] if len(F) else F
        row = {"source": s, "group": g, "modality": m, "tier": r["tier"], "stage2_scope": scope(s, g),
               "role": role(s, g), "read": "counts only (index)" if s in COUNT_ONLY else "records",
               "records": len(f), "persons": f["person_id"].nunique() if len(f) else 0,
               "recordings": f[["subject_ns", "recording"]].drop_duplicates().shape[0] if len(f) else 0,
               "hours": round(float(f["hours"].sum(min_count=1)), 3) if len(f) else NAN,
               "hours_index": round(float(f["hours_index"].sum(min_count=1)), 3) if len(f) else NAN,
               "records_without_fps": int((~np.isfinite(f["fps"].astype(float))).sum()) if len(f) else 0,
               "fps": fps_text(f["fps"]) if len(f) else ""}
        if s in COUNT_ONLY:
            d = idx[(idx["source"] == s) & (idx["group"] == g) & (idx["modality"] == m) & ~idx["lockbox"]]
            row.update({"persons": d["person_id"].nunique(), "recordings": len(d),
                        "hours_index": round(float((d["n_frames"] / d["fps"]).sum() / 3600), 3),
                        "fps": fps_text(d["fps"]) + " (index)"})
        if len(f) and row["records_without_fps"]:
            rf = [recovered.get((a, b), NAN) for a, b in zip(f["subject_ns"], f["key"])]
            row["hours_at_recovered_fps"] = round(float(np.nansum(f["n"].to_numpy() / np.asarray(rf)) / 3600), 3)
            row["recovered_fps"] = fps_text(rf)
        row.update({k: r[k] for k in ("index_rows", "index_persons", "nonlockbox_rows", "nonlockbox_persons",
                                      "lockbox_rows", "lockbox_persons")})
        row["read_errors"] = stats.get(s, {}).get("load_errors", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def source_totals(F, idx):
    out = []
    for s in SOURCES + COUNT_ONLY:
        d = idx[idx["source"] == s]
        pool = d if s not in DESIGN_GROUPS else (d[d["group"] == "stroke"] if s == "stroke_gait_vancriekinge"
                                                 else d[d["group"] != "HC"])
        f = F[F["source"] == s] if len(F) else F
        in_pool = {"stroke_gait_vancriekinge": lambda g: g == "stroke", "gaitrec": lambda g: g != "HC"}.get(
            s, lambda g: True)
        fs = f[[in_pool(g) for g in f["group"]]] if len(f) else f
        out.append({"source": s, "design_pool": DESIGN_GROUPS.get(s, "all groups"),
                    "design_training_persons": DESIGN[s][0], "design_persons": DESIGN[s][1],
                    "index_persons": pool["person_id"].nunique(),
                    "index_nonlockbox_persons": pool.loc[~pool["lockbox"], "person_id"].nunique(),
                    "index_lockbox_persons": pool.loc[pool["lockbox"], "person_id"].nunique(),
                    "persons_read": fs["person_id"].nunique() if len(fs) else (
                        "counts only" if s in COUNT_ONLY else 0)})
    return pd.DataFrame(out)


def content_table(F):
    rows = []
    for (s, g, m, sig), f in F.groupby(["source", "group", "modality", "sig"], sort=True):
        d = {"source": s, "group": g, "modality": m, "records": len(f), "persons": f["person_id"].nunique()}
        d.update(json.loads(sig))
        rows.append(d)
    return pd.DataFrame(rows)


def shapes_table(F):
    rows = []
    for (s, g, m, sh), f in F.groupby(["source", "group", "modality", "shape"], sort=True):
        rows.append({"source": s, "group": g, "modality": m, "array": C.MAIN_ARRAY[m], "shape": sh, "records": len(f),
                     "T_min_median_max": qs(f["n"].astype(float), (0, 50, 100)),
                     "n_equals_index_n_frames": int((f["n"] == f["n_index"]).sum())})
    return pd.DataFrame(rows)


def units_table(F):
    rows = []
    for (s, g, m), f in F.groupby(["source", "group", "modality"], sort=True):
        d = {"source": s, "group": g, "modality": m, "records": len(f)}
        if m == "imu":
            d.update({"unit": "acc g, gyro deg/s (contract)", "acc_content_field": "|".join(sorted(set(f["acc_content"]))),
                      "mag_unit": "|".join(sorted(set(f["mag_unit"]) - {""})),
                      "acc_norm_g_p10_p50_p90": qs(f["acc_med"]),
                      "acc_mean_norm_g_p10_p50_p90": qs(f["acc_dc"]),
                      "acc_class": "|".join(f"{k} {n}" for k, n in sorted(collections.Counter(f["acc_class"]).items())),
                      "gyro_p99_p10_p50_p90": qs(f["gyro_p99"]),
                      "up_axis": "see imu_sites.csv"})
        elif m == "force":
            d.update({"unit": "|".join(sorted(set(f["unit"]))),
                      "peak_L_p10_p50_p90": qs(f["peak_L"]), "peak_R_p10_p50_p90": qs(f["peak_R"]),
                      "up_axis": "none (foot totals)"})
        elif m == "pose":
            d.update({"unit": "|".join(sorted({str(x) for x in f["pose_modality"]})) + " (coords / scale)",
                      "scale_p10_p50_p90": qs(f["scale"]),
                      "up_axis": "|".join(f"{k or 'undetermined'} {n}" for k, n in sorted(collections.Counter(f["up"]).items())),
                      "up_note": "as recorded; stroke 3D records are rotated to +y by sources.pair (stage 1)"})
        elif m == "emg":
            d.update({"unit": "|".join(sorted(set(f["unit"]))), "emg_p99_p10_p50_p90": qs(f["emg_p99"]),
                      "amplitude_flag_records": int(f["amp_flag"].sum()), "up_axis": "none"})
        rows.append(d)
    return pd.DataFrame(rows)


def imu_sites_table(SR):
    rows = []
    for (s, g, site), f in SR.groupby(["source", "group", "site"], sort=True):
        ups = collections.Counter(f["up"])
        rows.append({"source": s, "group": g, "site": site, "records": len(f), "persons": f["person_id"].nunique(),
                     "channels": "|".join(f"{k} {n}" for k, n in sorted(collections.Counter(f["channels"]).items())),
                     "valid_share_p10_p50_p90": qs(f["valid_share"]),
                     "acc_norm_g_p10_p50_p90": qs(f["acc_norm_med"]),
                     "acc_mean_norm_g_p50": r4(q(f["acc_mean_norm"], 50)),
                     "up_axis": "|".join(f"{k or 'not with_gravity'} {n}" for k, n in sorted(ups.items())),
                     "up_axis_mode": max(((n, k) for k, n in ups.items() if k), default=(0, ""))[1],
                     "gyro_p99_p10_p50_p90": qs(f["gyro_p99"]),
                     "mag_norm_p50": r4(q(f["mag_norm_med"], 50))})
    return pd.DataFrame(rows)


def orientation_table(SR, cache):
    """Persons per source, group, site and modal up axis (the most frequent non-empty up axis of the person's
    records at that site; 'none' when no record of the person has one); clinical_gait_multipatho also carries
    the cohort its meta.json names (healthy, neuro, ortho)."""
    cohort = {s: v[1] for s, v in cache.get("clinical_pathology_keys", {}).items()}
    rows = []
    for (s, g, site), f in SR.groupby(["source", "group", "site"], sort=True):
        modal = {}
        for pid, h in f.groupby("person_id"):
            c = collections.Counter(u for u in h["up"] if u)
            modal[pid] = c.most_common(1)[0][0] if c else "none"
        coh = "|".join(sorted({cohort[x] for x in f["subject_ns"] if x in cohort}))
        for axis, n in sorted(collections.Counter(modal.values()).items()):
            rows.append({"source": s, "group": g, "cohort": coh, "site": site, "up_axis_mode": axis, "persons": n})
    return pd.DataFrame(rows, columns=["source", "group", "cohort", "site", "up_axis_mode", "persons"])


def validity_table(F):
    rows = []
    mask_name = {"imu": "sample_valid (T, L)", "force": "sample_valid (T, 2)", "pose": "valid_mask (T, 15) on present joints",
                 "emg": "none"}
    for (s, g, m), f in F.groupby(["source", "group", "modality"], sort=True):
        with_mask = f[f["has_mask"].astype(bool)]
        vn, vt = int(with_mask["valid_n"].sum()), int(with_mask["valid_total"].sum())
        rows.append({"source": s, "group": g, "modality": m, "mask": mask_name.get(m, ""), "records": len(f),
                     "records_with_mask": len(with_mask),
                     "valid_share": r4(vn / vt) if vt else None,
                     "record_valid_share_p10_p50_p90": qs((with_mask["valid_n"] / with_mask["valid_total"]).tolist())
                     if len(with_mask) else "",
                     "records_with_an_invalid_sample": int(f["any_invalid"].astype(bool).sum()),
                     "records_without_a_valid_sample": int(((with_mask["valid_n"] == 0)
                                                            & (with_mask["valid_total"] > 0)).sum()),
                     "nonfinite_share_real_channels": r4(f["nonfinite"].sum() / max(1, f["nreal"].sum())),
                     "nonfinite_where_valid": int(f["nonfinite_where_valid"].sum()),
                     "dropout_fraction_p50": r4(q(f["dropout_fraction"], 50)) if "dropout_fraction" in f else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- comparison of two runs
_ABSENT = "(absent)"


def _flat(d, prefix=""):
    """{dotted path: value} of a JSON object; a list is one value."""
    if isinstance(d, dict) and d:
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def _jkey(v):
    return json.dumps(v, sort_keys=True, default=str)


def _short(v, n=160):
    s = v if isinstance(v, str) else _jkey(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def _md_text(lines):
    """summary.md lines for compare.md: prose is echoed, table rows are only counted. The tables mirror the CSVs
    and summary.json, whose differences the comparison counts per column or lists per path; an older run's
    table rows may hold whole-source aggregates that include lockbox persons (review fix R1 of T19), and a
    comparison must not copy them into a new file."""
    prose = [x for x in lines if x.strip() and not x.lstrip().startswith("|")]
    n = sum(1 for x in lines if x.lstrip().startswith("|"))
    return " / ".join(prose + ([f"({n} table rows, not echoed)"] if n else []))


def _log_norm(line):
    """A run.log line without its time stamp, durations and output folder."""
    for pat in (r"^\[[^\]]*\] ", r"\bin [\d.]+ s\b", r", out \S+$"):
        line = re.sub(pat, "", line)
    return line


def _rows_key(df, cols):
    return sorted(map(tuple, df[list(cols)].to_numpy().tolist()))


def _rel(p):
    """p relative to the tree, with forward slashes; the absolute path when p sits on another drive (Windows
    relpath raises ValueError across drives, so --out or OLD_DIR may be a scratch folder on C:)."""
    try:
        return os.path.relpath(p, TREE).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def manifest_check(old_manifest):
    """Per source of the probe: the rows of the manifest the old run read against the current manifest, on the
    identifier columns for every row and on every column for the non-lockbox rows (booleans and counts only)."""
    import hashlib
    mo = pd.read_csv(old_manifest, dtype=str, keep_default_na=False, encoding="utf-8")
    mn = pd.read_csv(C.manifest_path(), dtype=str, keep_default_na=False, encoding="utf-8")
    cols = [c for c in mo.columns if c in mn.columns]
    idc = [c for c in MANIFEST_ID_COLS if c in cols]
    rows = []
    for s in SOURCES + COUNT_ONLY:
        a, b = mo[mo["source"] == s], mn[mn["source"] == s]
        na, nb = (x[~x["lockbox"].map(C._truthy).astype(bool)] for x in (a, b))
        rows.append({"source": s, "rows_old": len(a), "rows_new": len(b),
                     "identifier_columns_equal": _rows_key(a, idc) == _rows_key(b, idc),
                     "nonlockbox_rows_old": len(na), "nonlockbox_rows_new": len(nb),
                     "nonlockbox_rows_every_column_equal": _rows_key(na, cols) == _rows_key(nb, cols)})
    sha = {}
    for name, p in (("old", old_manifest), ("new", C.manifest_path())):
        with open(p, "rb") as fh:
            sha[name] = hashlib.sha256(fh.read()).hexdigest()
    return rows, {"old_file": _rel(old_manifest), "new_file": _rel(C.manifest_path()),
                  "old_file_sha256": sha["old"], "new_file_sha256": sha["new"], "columns_compared": cols,
                  "columns_only_old": [c for c in mo.columns if c not in mn.columns],
                  "columns_only_new": [c for c in mn.columns if c not in mo.columns]}


def container_times(since):
    """Per source of the probe: the containers of its non-lockbox rows and how many were modified after `since`
    (ISO time; file modification times only, no container is opened)."""
    idx = C.index()
    t0 = dt.datetime.fromisoformat(since).timestamp()
    rows = []
    for s in SOURCES:
        cont = sorted(set(selection(idx, s)["container"]))
        mt = [os.path.getmtime(C.container_path(c)) for c in cont]
        rows.append({"source": s, "containers": len(cont),
                     "newest_modified": dt.datetime.fromtimestamp(max(mt)).isoformat(timespec="seconds") if mt else "",
                     "modified_after_old_run_start": sum(1 for t in mt if t > t0)})
    return rows


def moveatlas_changes(old_head, new_head):
    """Commits between the two MoveAtlas heads and the tracked files they change that name a source of the probe,
    a pooled manifest, or a tracked file the probe reads."""
    import moveatlas_stamp as MS
    root = MS.default_root()
    log_ = MS._git(root, "log", "--format=%h %ad %s", "--date=short", f"{old_head}..{new_head}").decode(
        "utf-8", "replace").splitlines()
    names = MS._git(root, "diff", "--name-only", old_head, new_head).decode("utf-8", "replace").splitlines()
    out = []
    for n in names:
        hit = [s for s in SOURCES + COUNT_ONLY if s in n]
        if n in READ_TRACKED:
            how = "read by the probe"
        elif re.fullmatch(r"datasets/_canonical_\w*manifest\w*\.csv", n):
            how = "pooled manifest: reaches the probe through cache/manifest (scripts/build_tiers_and_lockbox.py)"
        elif hit:
            how = "not read by the probe"
        else:
            continue
        out.append({"path": n, "sources_named": "|".join(hit), "reaches_the_probe": how})
    return log_, out


def compare_runs(old, new, old_manifest=OLD_MANIFEST):
    """compare.json and compare.md in `new`: every output file of the two runs, byte for byte; for a differing CSV
    its differing cells, for a differing JSON its differing paths, for run.log the lines that differ once time
    stamps, durations and the output folder are removed, for summary.md the differing lines; then the evidence on
    the inputs: the manifest rows of the probe's sources (manifest_check), the container modification times
    against the old run's start (container_times) and the MoveAtlas commits between the two heads
    (moveatlas_changes)."""
    import difflib
    names = sorted({n for d in (old, new) for n in os.listdir(d) if not n.startswith("._")
                    and os.path.isfile(os.path.join(d, n)) and n not in COMPARE_OUT})
    files, jdiff, mdiff, ldiff = [], [], [], []
    for n in names:
        po, pn = os.path.join(old, n), os.path.join(new, n)
        row = {"file": n}
        if not (os.path.exists(po) and os.path.exists(pn)):
            row["status"] = "only new" if os.path.exists(pn) else "only old"
            files.append(row)
            continue
        with open(po, "rb") as fa, open(pn, "rb") as fb:
            row["status"] = "identical" if fa.read() == fb.read() else "differs"
        if row["status"] == "differs" and n.endswith(".csv"):
            a = pd.read_csv(po, dtype=str, keep_default_na=False, encoding="utf-8")
            b = pd.read_csv(pn, dtype=str, keep_default_na=False, encoding="utf-8")
            row["detail"] = f"rows {len(a)} and {len(b)}; same columns {list(a.columns) == list(b.columns)}"
            if a.shape == b.shape:
                ne = (a.to_numpy() != b.to_numpy()).sum(axis=0)       # per column position; names of the new file
                by = ", ".join(f"{c} {int(k)}" for c, k in zip(b.columns, ne) if k)
                row["detail"] += f"; differing cells {int(ne.sum())}" + (f"; by column {by}" if by else "")
        elif row["status"] == "differs" and n.endswith(".json"):
            with open(po, encoding="utf-8") as fa, open(pn, encoding="utf-8") as fb:
                fo, fn = _flat(json.load(fa)), _flat(json.load(fb))
            k0 = len(jdiff)
            for k in sorted(set(fo) | set(fn)):
                vo, vn = fo.get(k, _ABSENT), fn.get(k, _ABSENT)
                if _jkey(vo) != _jkey(vn):
                    jdiff.append({"file": n, "path": k, "old": _short(vo), "new": _short(vn)})
            row["detail"] = f"{len(jdiff) - k0} differing paths"
        elif row["status"] == "differs" and n == "run.log":
            with open(po, encoding="utf-8") as fa, open(pn, encoding="utf-8") as fb:
                lo, ln = fa.read().splitlines(), fb.read().splitlines()
            for i in range(max(len(lo), len(ln))):
                a = _log_norm(lo[i]) if i < len(lo) else _ABSENT
                b = _log_norm(ln[i]) if i < len(ln) else _ABSENT
                if a != b:
                    ldiff.append({"line": i + 1, "old": _short(a), "new": _short(b)})
            row["detail"] = (f"lines {len(lo)} and {len(ln)}; differing once time stamps, durations and the output "
                             f"folder are removed: {len(ldiff)}")
        elif row["status"] == "differs" and n.endswith(".md"):
            with open(po, encoding="utf-8") as fa, open(pn, encoding="utf-8") as fb:
                lo, ln = fa.read().splitlines(), fb.read().splitlines()
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, lo, ln, autojunk=False).get_opcodes():
                if tag != "equal":
                    mdiff.append({"kind": tag, "old_lines": f"{i1 + 1}-{i2}" if i2 > i1 else "",
                                  "new_lines": f"{j1 + 1}-{j2}" if j2 > j1 else "",
                                  "old": _short(_md_text(lo[i1:i2]), 300), "new": _short(_md_text(ln[j1:j2]), 300)})
            row["detail"] = f"lines {len(lo)} and {len(ln)}; differing blocks {len(mdiff)}"
        files.append(row)
    with open(os.path.join(old, "summary.json"), encoding="utf-8") as fa, \
            open(os.path.join(new, "summary.json"), encoding="utf-8") as fb:
        so, sn = json.load(fa), json.load(fb)
    with open(os.path.join(old, "_provenance.json"), encoding="utf-8") as fa:
        since = json.load(fa)["moveatlas_stamp"]["time"]
    man, man_info = manifest_check(old_manifest) if os.path.exists(old_manifest) else ([], {"missing": old_manifest})
    commits, changed = moveatlas_changes(so["moveatlas_head"], sn["moveatlas_head"])
    res = {"old": _rel(old), "new": _rel(new),
           "old_moveatlas_head": so["moveatlas_head"], "new_moveatlas_head": sn["moveatlas_head"],
           "files": files, "json_paths": jdiff, "summary_md_blocks": mdiff, "run_log_lines": ldiff,
           "manifest": man, "manifest_files": man_info, "old_run_start": since, "containers": container_times(since),
           "moveatlas_commits": commits, "moveatlas_files": changed,
           "run_at": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(new, "compare.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=str)
    st = collections.Counter(r["status"] for r in files)
    L = [f"# Stage-2 source probe: {res['new']} against {res['old']}", "",
         f"Generated by scripts/probe_stage2_sources.py --compare at {res['run_at']}; compare.json holds the same "
         f"facts. MoveAtlas HEAD {res['old_moveatlas_head']} (old run) and {res['new_moveatlas_head']} (new run).", "",
         "## Output files", "", "Status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())) + ".", ""]
    L += md_table(pd.DataFrame(files).reindex(columns=["file", "status", "detail"]).fillna(""))
    L += ["", f"## JSON paths that differ ({len(jdiff)})", ""]
    L += md_table(pd.DataFrame(jdiff, columns=["file", "path", "old", "new"]))
    L += ["", f"## summary.md blocks that differ ({len(mdiff)})", ""]
    L += md_table(pd.DataFrame(mdiff, columns=["kind", "old_lines", "new_lines", "old", "new"]))
    L += ["", f"## run.log lines that differ once time stamps, durations and the output folder are removed "
          f"({len(ldiff)})", ""]
    L += md_table(pd.DataFrame(ldiff, columns=["line", "old", "new"]))
    L += ["", "## Manifest rows of the probe's sources", "",
          f"Old: {man_info.get('old_file')} (SHA-256 {man_info.get('old_file_sha256')}); new: {man_info.get('new_file')} "
          f"(SHA-256 {man_info.get('new_file_sha256')}). Identifier columns {', '.join(MANIFEST_ID_COLS)} on every "
          f"row; every column ({', '.join(man_info.get('columns_compared', []))}) on the non-lockbox rows. Columns "
          f"only in the old file: {man_info.get('columns_only_old') or 'none'}; only in the new file: "
          f"{man_info.get('columns_only_new') or 'none'}.", ""]
    L += md_table(pd.DataFrame(man))
    L += ["", "## Containers read (non-lockbox rows), modification times", "",
          f"Against the start of the old run ({since}); no container is opened.", ""]
    L += md_table(pd.DataFrame(res["containers"]))
    L += ["", f"## MoveAtlas between {so['moveatlas_head'][:7]} and {sn['moveatlas_head'][:7]}", "",
          "Commits: " + ("; ".join(commits) or "none") + ".", "",
          "Tracked files changed that name a source of the probe, a pooled manifest or a file the probe reads:", ""]
    L += md_table(pd.DataFrame(changed, columns=["path", "sources_named", "reaches_the_probe"]))
    L.append("")
    text = "\n".join(L)
    assert "\u2014" not in text and "\u2013" not in text
    with open(os.path.join(new, "compare.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "stage2_sources"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--md-only", action="store_true", help="rewrite summary.md from the outputs of the last run")
    ap.add_argument("--compare", default=None, metavar="OLD_DIR",
                    help="compare the run in --out with the run in OLD_DIR (writes compare.json and compare.md "
                         "in --out; reads no record)")
    a = ap.parse_args()
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.md_only:
        write_md(a.out)
        return
    if a.compare:
        res = compare_runs(a.compare, a.out)
        print(json.dumps({r["file"]: r["status"] for r in res["files"]}, indent=1))
        return
    LOG.append(open(os.path.join(a.out, "run.log"), "w", encoding="utf-8", newline="\n"))
    prio = set_below_normal()
    t_start = time.time()
    log(f"start, BelowNormal priority set: {prio}, quick {a.quick}, out {a.out}")
    st0 = stamp(paths.MOVEATLAS)
    sites_names = locations()
    idx = C.index()
    recovered = {}
    fr = os.path.join(TREE, "reports", "nuisance", "frame_rates.csv")
    if os.path.exists(fr):
        d = pd.read_csv(fr, dtype=str, keep_default_na=False, encoding="utf-8")
        recovered = {(s, k): num(v) for s, k, v in zip(d["subject_ns"], d["take"], d["fps"])}
    facts, site_rows, pauses, stats, checks, fields = [], [], [], {}, {}, collections.defaultdict(set)
    for i, source in enumerate(SOURCES):
        if i:
            pause_if_slow(SOURCES[i - 1], source, pauses, a.out)
        t0 = time.time()
        st = collections.Counter()
        kept, kept_kinds = [], set()
        for row, rec in read_source(idx, source, a.quick, st):
            f, per_site, main = measure(row, rec, sites_names)
            facts.append(f)
            fields[(source, row["modality"])].update(k for k, v in rec.items()
                                                     if k not in READER_FIELDS and not isinstance(v, np.ndarray))
            for d in per_site:
                site_rows.append(dict(d, source=source, group=row["group"], person_id=row["person_id"],
                                      subject_ns=row["subject_ns"], key=row["key"]))
            kind = (row["modality"], row["group"])
            if len(kept) < N_LOADCHECK and (kind not in kept_kinds or st["records"] in (1, 2)):
                kept.append((row, main))
                kept_kinds.add(kind)
            st["records"] += 1
        checks[source] = reader_checks(idx, source, kept)
        stats[source] = dict(st, seconds=round(time.time() - t0, 1))
        log(f"{source}: {dict(st)} in {time.time() - t0:.0f} s; checks {checks[source]}")
    F = pd.DataFrame(facts)
    SR = pd.DataFrame(site_rows)
    log("records measured; tables")
    w = lambda df, name: df.to_csv(os.path.join(a.out, name), index=False, lineterminator="\n")  # noqa: E731
    src = source_table(F, idx, stats, recovered)
    w(src, "sources.csv")
    totals = source_totals(F, idx)
    miss = F[~np.isfinite(F["fps"].astype(float))][["source", "group", "modality", "subject_ns", "person_id", "key", "n"]].copy()
    miss["fps_index"] = F.loc[miss.index, "fps_index"]
    miss["recovered_fps"] = [recovered.get((s, k), NAN) for s, k in zip(miss["subject_ns"], miss["key"])]
    miss["recovered_by"] = ["reports/nuisance/frame_rates.csv" if np.isfinite(v) else "" for v in miss["recovered_fps"]]
    w(miss, "fps_missing.csv")
    w(content_table(F), "content.csv")
    w(shapes_table(F), "shapes.csv")
    w(units_table(F), "units.csv")
    w(imu_sites_table(SR), "imu_sites.csv")
    w(validity_table(F), "validity.csv")
    extra = {}
    log("demographics")
    demo, cache = demographics(F, idx, extra)
    w(demo, "demographics.csv")
    w(diagnosis(F, idx, cache), "diagnosis.csv")
    w(orientation_table(SR, cache), "orientation.csv")
    folds, fold_info = proposed_folds(idx)
    fold_balance(folds, fold_info, idx, cache.get("per_person", {}))
    w(folds, "folds_proposed.csv")
    R, P = weargait_sites(SR, F, idx, sites_names, folds[folds["source"] == "weargait_pd"])
    w(R, "weargait_site_records.csv")
    w(P, "weargait_sites.csv")
    if not pauses:
        pd.DataFrame(columns=["started", "ended", "seconds", "after_source", "before_source", "slow_file",
                              "slow_still_present"]).to_csv(os.path.join(a.out, "pauses.csv"), index=False,
                                                             lineterminator="\n")
    st1 = stamp(paths.MOVEATLAS)
    summary = {"quick": a.quick, "moveatlas_head": st0["head"], "moveatlas_diff_sha256": st0["diff_sha256"],
               "moveatlas_unchanged_during_run": st0["head"] == st1["head"] and st0["diff_sha256"] == st1["diff_sha256"],
               "manifest_note": st0["note"], "reader_stats": stats, "reader_checks": checks,
               "source_totals": totals.to_dict("records"),
               "record_fields": {f"{s}|{m}": sorted(v) for (s, m), v in sorted(fields.items())},
               "folds": fold_info, "pauses": pauses, "pauses_logged": len(pauses),
               "source_boundaries": len(SOURCES) - 1, "extra": extra,
               "weargait": weargait_summary(R, P, sites_names),
               "runtime_s": round(time.time() - t_start, 1)}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, default=str)
    prov = {"script": os.path.basename(__file__), "moveatlas_root": paths.MOVEATLAS, "moveatlas_stamp": st0,
            "moveatlas_stamp_end": st1, "run_at": dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "quick": a.quick, "seed": SEED,
            "reader": "mdfm.data.canonical.iterate(sources=<source>, lane='body') and canonical.load, every flag False",
            "inputs": ["cache/manifest/manifest_records.csv.gz", "cache/manifest/record_index.csv.gz",
                       "MoveAtlas canonical records of the sources (mdfm.data.canonical)",
                       "configs/canonical_imu.yaml (locations)", "reports/nuisance/frame_rates.csv",
                       "datasets/imu/pads/raw/patients/patient_<id>.json (training persons)",
                       "datasets/imu/pads/processed/pads_subject_table.csv (condition counts)",
                       "datasets/imu/weargait_pd/raw/*/CONTROLS and PD - Demographic+Clinical - datasetV1.csv",
                       "datasets/imu/physionet_gaitpdb/raw/gaitpdb.zip (demographics.txt, demographics.html)",
                       "datasets/imu/gaitndd/raw/gaitndd.zip (subject-description.txt)",
                       "datasets/imu/clinical_gait_multipatho/raw/dataset.zip (meta.json of training subjects)",
                       "datasets/imu/gaitrec/raw/GRF_metadata.csv (training subjects)",
                       f"datasets/{STROKE_MAT}",
                       "datasets/pose/stroke_gait_vancriekinge/raw/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat",
                       "datasets/imu/imu_inthewild_iprognosis/raw/i-PROGNOSIS_DS2.10_Sub1_IMUdataPhonecalls.zip "
                       "(dataset_description.txt only)", "datasets/pose/modys_video/datacard.md"],
            "outputs": OUTPUTS}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1, default=str)
    write_md(a.out)
    log(f"done in {summary['runtime_s']} s")


def weargait_summary(R, P, sites_names):
    all13 = [s for s in sites_names if s != "pocket"]
    out = {"persons": len(P), "recordings": len(R), "held_share_threshold": HELD_SHARE}
    for grp, g in P.groupby("group"):
        out[grp] = {"persons": len(g),
                    "persons_all_13_held_in_every_recording": int((g["n_sites_held_in_every_recording"] == len(all13)).sum()),
                    "persons_with_a_site_never_held": int((g["sites_never_held"] != "").sum()),
                    "recordings": int(g["recordings"].sum()),
                    "recordings_all_13_held": int(g["recordings_all_13_held"].sum())}
    lay = collections.Counter(R["sites_held"])
    out["layouts"] = [{"sites_held": k, "recordings": n,
                       "persons": int(R.loc[R["sites_held"] == k, "person_id"].nunique())} for k, n in lay.most_common()]
    out["site_never_held_persons"] = {s: int(P["sites_never_held"].str.split("|").map(lambda x, s=s: s in x).sum())
                                      for s in all13}
    return out


# ---------------------------------------------------------------- summary.md
def md_table(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]

    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else str(int(v)) if v.is_integer() else f"{v:g}"
        return str(v).replace("|", "\\|")

    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return lines


def findings(s, src, units, val, diag, P, tot, f, orient):
    """The facts of the run that change a later task, each read from the outputs."""
    def row(df, **kw):
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k].astype(str) == str(v)).to_numpy()
        return df[m].iloc[0].to_dict() if m.any() else {}

    L = ["## Findings", ""]
    off = tot[tot["design_training_persons"].astype(str) != tot["index_nonlockbox_persons"].astype(str)]
    L.append(f"- Training portions: the index holds the design's training persons for {len(tot) - len(off)} of {len(tot)} "
             "sources" + ("" if off.empty else ", not for " + ", ".join(off["source"])) + "; every record of them was "
             "read except koa_pd_nm (counts only), with " + str(int(pd.to_numeric(src["read_errors"]).sum()))
             + " read errors.")
    m, mv, mu = (row(src, source="modys_video"), row(val, source="modys_video"), row(units, source="modys_video"))
    L.append(f"- modys_video: {m.get('records_without_fps')} of {m.get('records')} training records carry no fps (record "
             f"and index); reports/nuisance/frame_rates.csv recovers {m.get('recovered_fps')} fps for them, "
             f"{m.get('hours_at_recovered_fps')} h at that rate. valid_mask share over present joints "
             f"{mv.get('valid_share')}; {mv.get('records_without_a_valid_sample')} records hold no valid joint sample at "
             f"all; up axis (neck minus mid-ankle, image coordinates): {mu.get('up_axis')}.")
    ip = row(src, source="imu_inthewild_iprognosis")
    L.append(f"- imu_inthewild_iprognosis: one pocket site, acc only, fps {ip.get('fps')} (phones sample irregularly; "
             "each record carries its session's median rate). Its diagnosis (pd_status) is on neither the records nor "
             "the index: " + str(row(diag, source="imu_inthewild_iprognosis").get("note", "")) + ".")
    pu = row(units, source="pads")
    L.append(f"- pads: acc_content {pu.get('acc_content_field')} on every record; the gravity test on the mean-acc "
             f"norm gives {pu.get('acc_class')} (median acc norm p10/p50/p90 {pu.get('acc_norm_g_p10_p50_p90')} g, "
             f"mean-acc norm {pu.get('acc_mean_norm_g_p10_p50_p90')} g): no gravity channel, so no up axis. Diagnosis "
             "from the record field condition: " + ", ".join(
                 f"{r['value']} {r['training_persons']}" for r in diag[diag["source"] == "pads"].to_dict("records"))
             + " training persons.")
    cg = orient[(orient["source"] == "clinical_gait_multipatho") & orient["site"].isin(["head", "lower_back"])]
    by = collections.defaultdict(collections.Counter)
    for r in cg.to_dict("records"):
        by[(r["cohort"], r["site"])][r["up_axis_mode"]] += int(r["persons"])
    L.append("- clinical_gait_multipatho sensor orientation follows the cohort: persons by modal up axis, "
             + "; ".join(f"{c} {site}: " + ", ".join(f"{a} {n}" for a, n in sorted(v.items()))
                         for (c, site), v in sorted(by.items()))
             + ". A model can read the cohort from the mounting; the orientation needs randomizing or "
               "canonicalizing before stage 2 (design section 8).")
    for name in ("gaitrec", "physionet_gaitpdb", "gaitndd", "weargait_pd"):
        rows = units[(units["source"] == name) & (units["modality"] == "force")]
        L.append(f"- {name} force: unit field " + "; ".join(
            f"{r['group']} '{r['unit']}', peak L p10/p50/p90 {r['peak_L_p10_p50_p90']}" for r in rows.to_dict("records"))
            + ".")
    se = units[(units["source"] == "stroke_gait_vancriekinge") & (units["modality"] == "emg")]
    L.append("- stroke_gait_vancriekinge EMG: amplitude flag in notes on " + "; ".join(
        f"{r['group']} {int(float(r['amplitude_flag_records']))} of {r['records']} records (record p99 of the absolute "
        f"signal, p10/p50/p90 {r['emg_p99_p10_p50_p90']} 'mV')" for r in se.to_dict("records")) + ".")
    sm = s.get("extra", {}).get("stroke_patient_mapping") or {}
    corr = sm.get("corr_height_torso_by_mapping") or {}
    used = sm.get("mapping_used")
    others = max([abs(v) for k, v in corr.items() if k != used and v is not None] or [0])
    L.append(f"- Stroke patients' demographics: subject numbers {sm.get('subject_numbers_min_max')} with gaps against "
             f"MAT Sub01 to Sub{sm.get('mat_subjects', 0):02d}; " + (
                 f"the Height to torso correlation supports mapping {used} ({corr.get(used)}; at most {others} in "
                 f"absolute value for the other mappings, number+0 {corr.get('number+0')}), used for the coverage."
                 if used else f"no mapping reaches 0.5 with a margin of 0.2 ({corr}), so no stroke patient is matched."))
    wg = s["weargait"]
    whc = P[P["person_id"] == "weargait:WHC003"]
    w3 = whc.iloc[0].to_dict() if len(whc) else {}
    pd_dev = P[(P["group"] == "PD") & (P["proposed_fold"] == "dev")]
    L.append(f"- weargait_pd sites: {wg['PD']['persons_all_13_held_in_every_recording']} of {wg['PD']['persons']} PD and "
             f"{wg['control']['persons_all_13_held_in_every_recording']} of {wg['control']['persons']} controls hold all "
             f"13 sites in every IMU recording; WHC003 (stage-1 fold {w3.get('stage1_fold')}) holds "
             f"{w3.get('n_sites_held_in_every_recording')} sites in each of its {w3.get('recordings')} recordings and never "
             f"{w3.get('sites_never_held')}. Of the {len(pd_dev)} proposed PD dev persons, "
             f"{int((pd_dev['n_sites_held_in_every_recording'] == 13).sum())} hold all 13 in every recording; the others: "
             + "; ".join(f"{r['person_id']} " + (f"never holds {r['sites_never_held']}" if r["sites_never_held"]
                                                 else "holds every site in at least one recording")
                         + (f" and holds {r['sites_held_in_some_not_all']} only in that many of its "
                            f"{r['recordings']} recordings" if r["sites_held_in_some_not_all"]
                            else " and holds each other site in every recording")
                         for r in pd_dev[pd_dev["n_sites_held_in_every_recording"] != 13].to_dict("records")) + ".")
    fw = val[(val["source"] == "weargait_pd") & (val["modality"] == "force")]
    L.append("- weargait_pd insole totals: sample_valid share " + ", ".join(
        f"{r['group']} {r['valid_share']}" for r in fw.to_dict("records")) + " (batch 0002B2 marks the dropouts).")
    L.append("- Proposed dev folds: " + "; ".join(f"{k} {v['dev']} of {v['persons']} persons, none in the lockbox"
                                                  if not v["dev_in_lockbox"] else f"{k}: {v['dev_in_lockbox']} IN THE LOCKBOX"
                                                  for k, v in f.items()) + " (the owner decides before any use).")
    L.append("")
    return [ln.replace("|", ", ") for ln in L]


def write_md(out):
    rd = lambda name: pd.read_csv(os.path.join(out, name), encoding="utf-8", keep_default_na=False)  # noqa: E731
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as fh:
        s = json.load(fh)
    src, demo, diag, val, units, sites, P, folds, content, shapes, miss, pauses = (rd(n) for n in (
        "sources.csv", "demographics.csv", "diagnosis.csv", "validity.csv", "units.csv", "imu_sites.csv",
        "weargait_sites.csv", "folds_proposed.csv", "content.csv", "shapes.csv", "fps_missing.csv", "pauses.csv"))
    tot = pd.DataFrame(s["source_totals"])
    covered = sorted(set(src["source"]))
    f = s["folds"]
    wg = s["weargait"]
    chk = s["reader_checks"]
    L = ["# Stage-2 sources: training portions after lockbox v1.2 (weekend task T19)", "",
         "Generated by scripts/probe_stage2_sources.py; every number below is read from summary.json and the CSVs of "
         f"this directory. MoveAtlas HEAD {s['moveatlas_head']}, working-tree diff SHA-256 {s['moveatlas_diff_sha256']} "
         f"(unchanged during the run: {s['moveatlas_unchanged_during_run']}); {s['manifest_note']} Definitions in the "
         "script docstring." + (" QUICK RUN (subsets only)." if s["quick"] else ""), "",
         "## Done-when checks", "", "| check | measured |", "|---|---|",
         f"| every source has a row | {len(covered)} sources in sources.csv: {', '.join(covered)} |",
         "| the proposed fold lists are hashed | " + "; ".join(
             f"{k}: {v['dev']} dev of {v['persons']}, dev SHA-256 {v['sha256_dev']}" for k, v in f.items()) + " |",
         f"| the provenance is present | _provenance.json with the scripts/moveatlas_stamp.py stamp at start and end |",
         f"| every pause is logged | cache/runs/queue/SLOW checked at {s['source_boundaries']} source boundaries, "
         f"present at {s['pauses_logged']}: {s['pauses_logged']} pauses of {PAUSE_S} s, each in pauses.csv and run.log |",
         "| reader flags all False | canonical.load refused " + ", ".join(
             f"{k} {v['lockbox_rows_refused']} of {v['lockbox_rows_tried']}" for k, v in chk.items())
         + " lockbox rows; canonical.load equals the iterated record on " + str(sum(v["load_rows_equal"] for v in chk.values()))
         + " of " + str(sum(v["load_rows_compared"] for v in chk.values())) + " rows |", ""]
    L += findings(s, src, units, val, diag, P, tot, f, rd("orientation.csv"))
    L += ["## Persons against design section 5", "",
          "Pool: the persons the design counts (stroke patients only, gaitrec orthopaedic patients only, every group "
          "elsewhere). persons_read counts the persons of that pool whose records were read.", ""]
    L += md_table(tot)
    L += ["", "## Training portion per source, group and modality", "",
          "hours: n / fps of the records; hours_index: n_frames / fps of the index; records without fps carry no "
          "hours (hours_at_recovered_fps uses reports/nuisance/frame_rates.csv). koa_pd_nm: index counts only.", ""]
    L += md_table(src, [c for c in src.columns if c not in ("index_rows", "index_persons")])
    L += ["", f"Records without fps: {len(miss)} (fps_missing.csv), by source: "
          + (", ".join(f"{k} {v}" for k, v in miss["source"].value_counts().items()) or "none")
          + "; recovered rates where known: "
          + (", ".join(f"{k:g} fps on {v}" for k, v in pd.to_numeric(miss["recovered_fps"], errors="coerce").value_counts().items())
             or "none") + ".", "",
          "## Channel content (sites, channels, masks)", ""]
    L += md_table(content)
    L += ["", "## Units and up axis", ""]
    L += md_table(units)
    L += ["", "### IMU sites: acc magnitude, up axis, gyro", "",
          "up_axis: the signed sensor axis of the mean acc (with_gravity sites only); acc_content as recorded is in "
          "units.csv and content.csv.", ""]
    L += md_table(sites)
    orient = rd("orientation.csv")
    cg = orient[(orient["source"] == "clinical_gait_multipatho") & orient["site"].isin(["head", "lower_back"])]
    piv = (cg.assign(cell=cg["up_axis_mode"] + " " + cg["persons"].astype(str))
           .groupby(["group", "cohort", "site"], sort=True)["cell"].agg(", ".join).reset_index()
           .rename(columns={"cell": "persons by modal up axis"}))
    L += ["", "### clinical_gait_multipatho: modal up axis per person, head and lower back", "",
          "Each person counted once per site at the up axis most of its records show (orientation.csv holds every "
          "source and site); cohort from the meta.json group field.", ""]
    L += md_table(piv)
    L += ["", "## Validity masks", ""]
    L += md_table(val)
    L += ["", "## Stream shapes", ""]
    L += md_table(shapes)
    L += ["", "## Demographics coverage (training portion)", ""]
    L += md_table(demo)
    sm = s.get("extra", {}).get("stroke_patient_mapping")
    if sm:
        L += ["", f"Stroke patients: the subject numbers of the index run from {sm['subject_numbers_min_max']} "
              f"({sm['index_subject_numbers']} subjects, the TVC folder numbers of the raw zip) while the MAT file holds "
              f"Sub01 to Sub{sm['mat_subjects']:02d}. Correlation of sub_char Height with the median torso length of the "
              f"subject's pose records over {sm['training_subjects_with_torso']} training subjects, per mapping (Sub "
              f"index = the subject number, or its rank among the sorted numbers, plus a shift): "
              f"{sm['corr_height_torso_by_mapping']}. Mapping used for the coverage: "
              f"{sm['mapping_used'] or 'none (no mapping reaches 0.5 with a margin of 0.2), so no stroke patient is matched'}."]
    L += ["", "## Diagnosis source of the groups labelled 'all', and group consistency", ""]
    L += md_table(diag)
    L += ["", "## Weargait site completeness (non-lockbox persons; input to the M3 and WHC003 decision)", "",
          f"A site is held in a recording when listed with at least {wg['held_share_threshold']:g} of its samples valid. "
          f"{wg['persons']} persons, {wg['recordings']} IMU recordings. "
          + "; ".join(f"{g}: {v['persons']} persons, {v['persons_all_13_held_in_every_recording']} hold all 13 sites in "
                      f"every recording, {v['persons_with_a_site_never_held']} never hold some site; "
                      f"{v['recordings_all_13_held']} of {v['recordings']} recordings hold all 13"
                      for g, v in wg.items() if isinstance(v, dict) and "persons" in v) + ".", "",
          "Persons never holding a site: " + ", ".join(f"{k} {v}" for k, v in wg["site_never_held_persons"].items() if v)
          + ".", "", "Layouts of held sites (recordings, persons):", ""]
    L += md_table(pd.DataFrame(wg["layouts"]))
    inc = P[(P["sites_never_held"] != "") | (P["sites_held_in_some_not_all"] != "")]
    L += ["", f"Persons with a site missing from some recording ({len(inc)}; every person in weargait_sites.csv):", ""]
    L += md_table(inc)
    L += ["", "## Proposed dev folds", "",
          "Drawn like mdfm.data.sources.folds (module docstring of the script); folds_proposed.csv lists the persons.", "",
          "| key | pool | persons | dev | train | SHA-256 pool | SHA-256 dev | SHA-256 train | dev in lockbox |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in f.items():
        L.append(f"| {k} | {v['pool']} | {v['persons']} | {v['dev']} | {v['train']} | {v['sha256_pool']} | "
                 f"{v['sha256_dev']} | {v['sha256_train']} | {v['dev_in_lockbox']} |")
    L += ["", "| key | fold | persons with a record, per modality | demographics matched | sex | age median |",
          "|---|---|---|---|---|---|"]
    for k, v in f.items():
        for fold in ("dev", "train"):
            L.append(f"| {k} | {fold} | {v.get(f'{fold}_persons_with')} | {v.get(f'{fold}_matched')} | "
                     f"{v.get(f'{fold}_sex')} | {v.get(f'{fold}_age_median')} |")
    devp = P[P["proposed_fold"] == "dev"]
    L += ["", f"weargait_pd PD dev persons holding all 13 sites in every recording: "
          f"{int((devp['n_sites_held_in_every_recording'] == 13).sum())} of {len(devp)}.", ""]
    L += ["## Pauses", "", f"{len(pauses)} pauses of {PAUSE_S} s while cache/runs/queue/SLOW existed.", ""]
    if len(pauses):
        L += md_table(pauses)
        L.append("")
    L += ["## Reader", "", "Records per source and seconds: " + "; ".join(
        f"{k} {v.get('records', 0)} records in {v.get('seconds')} s"
        + (f", load errors {v['load_errors']}" if v.get("load_errors") else "")
        + (f", iterate stopped: {v['iterate_stopped']}" if v.get("iterate_stopped") else "")
        for k, v in s["reader_stats"].items()) + ".", "",
          "Record fields (non-array) per source and modality: " + "; ".join(
              f"{k}: {', '.join(v)}" for k, v in s["record_fields"].items()) + ".", "",
          f"Runtime: {s['runtime_s']} s.", ""]
    text = "\n".join(L)
    assert "\u2014" not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
