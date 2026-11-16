"""merk_neumann_ecog_lfp: reader facts, MNI electrode rows, force channels and samples per run.

Weekend task T23 (2); stage N0 port of the merk checks of addendum 1 (sections 1.1, 1.5, 1.6
item 12, 2.2 and 9), NR-3.11 and OPEN-12 of docs/rules/NEURAL_RULES_DRAFT.md, and item 1 of
the appendix of docs/feedback/CLINICAL_QUESTIONS_CROSSWALK.md.

Source (read-only): MoveAtlas datasets/inversion_testbed/merk_neumann_ecog_lfp/raw/, a flat BIDS
iEEG copy of Harvard Dataverse doi:10.7910/DVN/IO2FLM (CC0). Per run: BrainVision .vhdr, .vmrk
and .eeg (IEEE float32, multiplexed), _ieeg.json, _channels.tsv and
_space-mni_electrodes.tsv; per subject and session: _space-mni_coordsystem.json and
_scans.tsv; per subject: _electrodes[-1].tsv and _coordsystem[-1].json; participants.tsv.
The acq_time column of the scans files is never read.

Reader: mne.io.read_raw_brainvision(preload=False) for the header (rate, samples, channel
names); the .eeg binary is read only for the force channels (MOV_*), by a memory map in
chunks, one run at a time. Values are reported as MNE reads them: stored float x the
channel resolution of the .vhdr x 1e-6 (the header unit is microvolt, MNE returns volt).

Outputs (reports/merk/ by default):
  runs.csv             one row per run: subject, session, run, rate, samples, duration,
                       channel counts by type, ECoG and STN sides, force channels, bad
                       channels, MNI rows, markers
  force.csv            one row per run and MOV channel: mean, standard deviation, range,
                       share of samples at exactly 0 and at 1 (within 1e-6), non-finite count
  electrodes_mni.csv   one row per run: MNI rows by type with finite coordinates, the declared
                       coordinate system and unit of the session file, and the match of every
                       named row to the per-subject electrodes files (max difference in mm)
  stn_mni.csv          one row per run with STN channels: STN rows, their coordinate ranges in
                       mm, and which contacts of the per-subject lead they equal
  subject_electrodes.csv  one row per per-subject electrodes file: declared system and unit,
                       rows by prefix
  samples_per_run.csv  OPEN-12: samples per run, the byte check of the multiplexed binary,
                       the sidecar duration check, and whether neural and force channels share
                       the one file (the G15 line "matching sample counts per run")
  addendum_checks.csv  claim, source, measured, verdict
  facts.json, summary.md, _provenance.json (MoveAtlas stamp from scripts/moveatlas_stamp.py)
Person map: reports/person_ids/map_merk_neumann_ecog_lfp.csv (columns of the other maps:
source, subject_ns, person_id, rule, evidence, ambiguity). The subject namespace
merk_neumann_ecog_lfp:<participant_id> is defined here: the set has no canonical manifest row.

Fixed definitions. A MOV channel is "centred" when its mean over the run lies within
ZS_MEAN_TOL of 0, and has "unit variance" when its standard deviation (ddof 0) lies within
ZS_STD_TOL of 1; "consistent with z-scoring of its run" means both. A run's samples "match"
when the .eeg size equals channels x samples x 4 bytes, the .vhdr DataPoints (when declared)
equals the sample count, and the sidecar RecordingDuration equals (samples - 1) / rate within
one sample. Without DataPoints MNE takes the sample count from the .eeg size, so the byte check
then only tests that the size divides evenly and the sidecar duration is the independent
check. The STN centroid of a run is the mean of its
STN rows with x replaced by |x|; a run is "far from the other STN runs" when its centroid lies
more than STN_FAR_MM from the median centroid of all STN runs (descriptive, no atlas).

Descriptive only: no statistic links a neural channel to force, movement or an outcome.
The process lowers its own priority to below normal and waits while cache/runs/queue/SLOW
exists.

Run: python scripts/probe_merk.py [--moveatlas <moveatlas>] [--out DIR]
"""
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")
SRC = "merk_neumann_ecog_lfp"
RAW_REL = "datasets/inversion_testbed/merk_neumann_ecog_lfp/raw"
RUN_RE = re.compile(r"^(sub-\d+)_(ses-[A-Za-z]+)_task-([A-Za-z]+)_run-(\d+)_ieeg\.vhdr$")
ELEC_RE = re.compile(r"^(sub-\d+)_electrodes(-\d+)?\.tsv$")
ZS_MEAN_TOL = 0.01
ZS_STD_TOL = 0.05
STN_FAR_MM = 10.0
CHUNK = 200_000
UNIT_TO_SI = {"µV": 1e-6, "uV": 1e-6, "mV": 1e-3, "V": 1.0, "nV": 1e-9}


# ---------------------------------------------------------------- process discipline
def below_normal():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


def wait_if_slow():
    waited = 0
    while os.path.exists(SLOW):
        if waited == 0:
            print("SLOW present: waiting", flush=True)
        time.sleep(30)
        waited += 30
    return waited


# ---------------------------------------------------------------- readers
def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def vhdr_channels(path):
    """[(name, resolution, unit)] from the [Channel Infos] block of a .vhdr, the binary format, and the
    DataPoints value of [Common Infos] (None when the header does not declare it)."""
    chans, fmt, points, in_block = [], None, None, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("BinaryFormat="):
                fmt = s.split("=", 1)[1]
            if s.startswith("DataPoints="):
                try:
                    points = int(s.split("=", 1)[1])
                except ValueError:
                    points = None
            if s.startswith("["):
                in_block = s == "[Channel Infos]"
                continue
            if in_block and re.match(r"^Ch\d+=", s):
                parts = s.split("=", 1)[1].split(",")
                name = parts[0].replace("\\1", ",")
                res = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
                unit = parts[3] if len(parts) > 3 and parts[3] else "µV"
                chans.append((name, res, unit))
    return chans, fmt, points


def count_markers(path):
    if not os.path.isfile(path):
        return None
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if re.match(r"^Mk\d+=", line.strip()):
                n += 1
    return n


def force_stats(eeg_path, n_ch, n_times, picks):
    """Per picked channel: running moments and counts over the multiplexed float32 binary."""
    mm = np.memmap(eeg_path, dtype="<f4", mode="r", shape=(n_times, n_ch))
    acc = {name: {"n": 0, "s": 0.0, "ss": 0.0, "min": np.inf, "max": -np.inf, "zero": 0, "one": 0,
                  "nonfinite": 0} for name, _, _ in picks}
    for start in range(0, n_times, CHUNK):
        block = np.asarray(mm[start:start + CHUNK])
        for name, idx, scale in picks:
            x = block[:, idx].astype(np.float64) * scale
            fin = np.isfinite(x)
            a = acc[name]
            a["nonfinite"] += int((~fin).sum())
            x = x[fin]
            a["n"] += x.size
            a["s"] += float(x.sum())
            a["ss"] += float((x * x).sum())
            if x.size:
                a["min"] = min(a["min"], float(x.min()))
                a["max"] = max(a["max"], float(x.max()))
            a["zero"] += int((x == 0).sum())
            a["one"] += int((np.abs(x - 1.0) <= 1e-6).sum())
    del mm
    out = {}
    for name, a in acc.items():
        if a["n"] == 0:
            # no finite sample (or an unknown unit, scale NaN): no moment, so never "centred"
            out[name] = {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"),
                         "share_zero": float("nan"), "share_one": float("nan"), "nonfinite": a["nonfinite"]}
            continue
        n = a["n"]
        mean = a["s"] / n
        var = max(a["ss"] / n - mean * mean, 0.0)
        out[name] = {"n": a["n"], "mean": mean, "std": var ** 0.5, "min": a["min"], "max": a["max"],
                     "share_zero": a["zero"] / n, "share_one": a["one"] / n, "nonfinite": a["nonfinite"]}
    return out


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def side_of(name):
    m = re.match(r"^(ECOG|STN|MOV)_(LEFT|RIGHT)", name)
    return m.group(2).lower() if m else ""


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "merk"))
    ap.add_argument("--person-map", default=os.path.join(TREE, "reports", "person_ids", "map_merk_neumann_ecog_lfp.csv"))
    a = ap.parse_args()
    root = a.moveatlas.replace("\\", "/").rstrip("/")
    raw = os.path.join(root, RAW_REL)
    if not os.path.isdir(raw):
        sys.exit(f"merk raw folder not found: {raw}")
    os.makedirs(a.out, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    below_normal()
    import mne
    mne.set_log_level("ERROR")
    t0 = time.time()
    slow_waited = 0

    files = sorted(f for f in os.listdir(raw) if not f.startswith("._") and os.path.isfile(os.path.join(raw, f)))
    raw_bytes = sum(os.path.getsize(os.path.join(raw, f)) for f in files)
    vhdrs = [f for f in files if RUN_RE.match(f)]
    participants = read_tsv(os.path.join(raw, "participants.tsv"))

    # per-subject electrodes files
    subj_elec, subj_rows = {}, []
    for f in files:
        m = ELEC_RE.match(f)
        if not m:
            continue
        sub, suf = m.group(1), m.group(2) or ""
        cs_path = os.path.join(raw, f"{sub}_coordsystem{suf}.json")
        cs = read_json(cs_path) if os.path.isfile(cs_path) else {}
        rows = read_tsv(os.path.join(raw, f))
        unit = cs.get("iEEGCoordinateUnits", "")
        to_mm = {"m": 1000.0, "mm": 1.0, "cm": 10.0}.get(unit, float("nan"))
        named = {}
        for r in rows:
            xyz = np.array([fnum(r.get(k)) for k in ("x", "y", "z")]) * to_mm
            named[r["name"]] = xyz
        subj_elec.setdefault(sub, []).append({"file": f, "named": named})
        prefixes = {}
        for r in rows:
            p = re.sub(r"_\d+$", "", r["name"])
            fin = all(np.isfinite(fnum(r.get(k))) for k in ("x", "y", "z"))
            c = prefixes.setdefault(p, [0, 0])
            c[0] += 1
            c[1] += int(fin)
        subj_rows.append({"subject": sub, "file": f, "coordsystem": cs.get("iEEGCoordinateSystem", ""),
                          "units": unit, "rows": len(rows),
                          "rows_by_prefix": "; ".join(f"{p} {c[0]} ({c[1]} finite)" for p, c in sorted(prefixes.items())),
                          "stn_rows": sum(c[0] for p, c in prefixes.items() if p.startswith("STN")),
                          "ecog_rows": sum(c[0] for p, c in prefixes.items() if p.startswith("ECOG"))})

    runs, force_rows, elec_rows, stn_rows, sample_rows = [], [], [], [], []
    for i, fn in enumerate(vhdrs):
        slow_waited += wait_if_slow()
        sub, ses, task, run = RUN_RE.match(fn).groups()
        base = fn[:-len("_ieeg.vhdr")]
        vpath = os.path.join(raw, fn)
        r = mne.io.read_raw_brainvision(vpath, preload=False, verbose="error")
        sf, n_times, names = float(r.info["sfreq"]), int(r.n_times), list(r.ch_names)
        del r
        vch, fmt, vpoints = vhdr_channels(vpath)
        eeg_path = os.path.join(raw, base + "_ieeg.eeg")
        size = os.path.getsize(eeg_path)
        bps = {"IEEE_FLOAT_32": 4, "INT_16": 2, "INT_32": 4}.get(fmt, 0)
        # Without DataPoints, MNE takes the sample count from the .eeg size (floor division), so this
        # check then only tests that the size divides evenly; the sidecar duration is the independent one.
        bytes_match = size == len(vch) * n_times * bps
        points_match = None if vpoints is None else vpoints == n_times
        js = read_json(os.path.join(raw, base + "_ieeg.json"))
        chs = read_tsv(os.path.join(raw, base + "_channels.tsv"))
        types = {c["name"]: c["type"] for c in chs}
        status = {c["name"]: c.get("status", "") for c in chs}
        units_tsv = sorted({c.get("units", "") for c in chs})
        n_by = {t: sum(1 for c in chs if c["type"] == t) for t in sorted({c["type"] for c in chs})}
        ecog = [c["name"] for c in chs if c["type"] == "ECOG"]
        seeg = [c["name"] for c in chs if c["type"] == "SEEG"]
        mov = [c["name"] for c in chs if c["name"].startswith("MOV_")]
        bad = [c["name"] for c in chs if status.get(c["name"]) == "bad"]
        json_dur = fnum(js.get("RecordingDuration"))
        dur_json_match = abs(json_dur - (n_times - 1) / sf) <= 1.0 / sf if np.isfinite(json_dur) else False
        n_mk = count_markers(os.path.join(raw, base + "_ieeg.vmrk"))
        runs.append({
            "subject": sub, "session": ses, "run": int(run), "task": task, "sfreq_hz": sf, "n_samples": n_times,
            "duration_s": round(n_times / sf, 3), "json_sfreq_hz": js.get("SamplingFrequency"),
            "json_duration_s": json_dur, "n_channels": len(names),
            "names_match_channels_tsv": names == [c["name"] for c in chs],
            "n_ecog": n_by.get("ECOG", 0), "n_seeg": n_by.get("SEEG", 0), "n_misc": n_by.get("MISC", 0),
            "other_types": "; ".join(f"{t} {n}" for t, n in n_by.items() if t not in ("ECOG", "SEEG", "MISC")),
            "ecog_side": "|".join(sorted({side_of(x) for x in ecog})),
            "stn_present": bool(seeg), "stn_side": "|".join(sorted({side_of(x) for x in seeg})),
            "stn_channels": "|".join(seeg),
            "force_channels": "|".join(mov), "bad_channels": len(bad),
            "bad_by_type": "; ".join(f"{t} {sum(1 for b in bad if types[b] == t)}" for t in sorted({types[b] for b in bad})),
            "channels_tsv_units": "|".join(units_tsv),
            "json_iEEGReference": js.get("iEEGReference", ""),
            "markers": n_mk, "eeg_bytes": size})
        sample_rows.append({
            "subject": sub, "session": ses, "run": int(run), "sfreq_hz": sf, "n_samples": n_times,
            "n_channels": len(vch), "n_neural": len(ecog) + len(seeg), "n_force": len(mov),
            "binary_format": fmt, "eeg_bytes": size, "bytes_expected": len(vch) * n_times * bps,
            "bytes_match": bytes_match, "vhdr_datapoints": "" if vpoints is None else vpoints,
            "vhdr_datapoints_match": "" if points_match is None else points_match, "json_duration_s": json_dur,
            "duration_from_samples_s": round((n_times - 1) / sf, 3), "json_duration_match": dur_json_match,
            "neural_and_force_in_one_file": bool((ecog or seeg) and mov),
            "sample_counts_match": bool(bytes_match and dur_json_match and points_match is not False
                                        and (ecog or seeg) and mov)})

        # MNI rows of the run
        epath = os.path.join(raw, base + "_space-mni_electrodes.tsv")
        cspath = os.path.join(raw, f"{sub}_{ses}_space-mni_coordsystem.json")
        cs = read_json(cspath) if os.path.isfile(cspath) else {}
        unit = cs.get("iEEGCoordinateUnits", "")
        to_mm = {"m": 1000.0, "mm": 1.0, "cm": 10.0}.get(unit, float("nan"))
        erows = read_tsv(epath) if os.path.isfile(epath) else []
        fin = {}
        coords = {}
        for e in erows:
            xyz = np.array([fnum(e.get(k)) for k in ("x", "y", "z")])
            t = types.get(e["name"], "absent_from_channels_tsv")
            c = fin.setdefault(t, [0, 0])
            c[0] += 1
            if np.all(np.isfinite(xyz)):
                c[1] += 1
                coords[e["name"]] = xyz * to_mm
        # match to the per-subject electrodes files
        best = None
        for sfile in subj_elec.get(sub, []):
            matched = [n for n in coords if n in sfile["named"] and np.all(np.isfinite(sfile["named"][n]))]
            if not matched:
                continue
            d = max(float(np.max(np.abs(coords[n] - sfile["named"][n]))) for n in matched)
            cand = (len(matched), -d, sfile["file"], d)
            if best is None or cand[:2] > best[:2]:
                best = cand
        elec_rows.append({
            "subject": sub, "session": ses, "run": int(run), "file_present": os.path.isfile(epath),
            "rows": len(erows),
            "ecog_rows": fin.get("ECOG", [0, 0])[0], "ecog_finite": fin.get("ECOG", [0, 0])[1],
            "seeg_rows": fin.get("SEEG", [0, 0])[0], "seeg_finite": fin.get("SEEG", [0, 0])[1],
            "misc_rows": fin.get("MISC", [0, 0])[0], "misc_finite": fin.get("MISC", [0, 0])[1],
            "other_rows": sum(v[0] for k, v in fin.items() if k not in ("ECOG", "SEEG", "MISC")),
            "session_coordsystem": cs.get("iEEGCoordinateSystem", ""), "session_units": unit,
            "session_units_known": bool(np.isfinite(to_mm)),
            "subject_file_matched": best[2] if best else "", "rows_matched": best[0] if best else 0,
            "max_abs_diff_mm": r3(best[3]) if best else None})
        if seeg:
            st = {n: coords[n] for n in seeg if n in coords}
            arr = np.array(list(st.values())) if st else np.zeros((0, 3))
            # which contacts of the per-subject lead the run rows equal (within 0.001 mm)
            eq = []
            for n, v in st.items():
                hit = ""
                for sfile in subj_elec.get(sub, []):
                    for sn, sv in sfile["named"].items():
                        if sn.startswith("STN") and np.all(np.isfinite(sv)) and np.max(np.abs(sv - v)) < 1e-3:
                            hit = sn
                            break
                    if hit:
                        break
                eq.append(f"{n}={hit or 'none'}")
            lead = sorted({sn for sfile in subj_elec.get(sub, []) for sn in sfile["named"]
                           if sn.startswith("STN_" + (side_of(seeg[0]) or "").upper())})
            cen = (np.array([np.abs(arr[:, 0]).mean(), arr[:, 1].mean(), arr[:, 2].mean()])
                   if len(arr) else np.full(3, np.nan))
            stn_rows.append({
                "subject": sub, "session": ses, "run": int(run), "stn_channels": len(seeg), "stn_rows_finite": len(st),
                "centroid_absx_mm": r3(cen[0]), "centroid_y_mm": r3(cen[1]), "centroid_z_mm": r3(cen[2]),
                "x_mm_min": r3(arr[:, 0].min()) if len(arr) else None, "x_mm_max": r3(arr[:, 0].max()) if len(arr) else None,
                "y_mm_min": r3(arr[:, 1].min()) if len(arr) else None, "y_mm_max": r3(arr[:, 1].max()) if len(arr) else None,
                "z_mm_min": r3(arr[:, 2].min()) if len(arr) else None, "z_mm_max": r3(arr[:, 2].max()) if len(arr) else None,
                "subject_lead_contacts": len(lead), "run_row_equals_subject_contact": "; ".join(eq)})

        # force channels: data
        res = {n: (res_, u) for n, res_, u in vch}
        picks = [(n, names.index(n), res[n][0] * UNIT_TO_SI.get(res[n][1], float("nan"))) for n in mov]
        if picks and bytes_match and fmt == "IEEE_FLOAT_32":
            stats = force_stats(eeg_path, len(vch), n_times, picks)
            for n in mov:
                s = stats[n]
                force_rows.append({
                    "subject": sub, "session": ses, "run": int(run), "channel": n,
                    "header_unit": res[n][1], "tsv_units": next((c.get("units", "") for c in chs if c["name"] == n), ""),
                    "n": s["n"], "mean": s["mean"], "std": s["std"], "min": s["min"], "max": s["max"],
                    "share_zero": round(s["share_zero"], 6), "share_one": round(s["share_one"], 6),
                    "nonfinite": s["nonfinite"],
                    "centred": abs(s["mean"]) <= ZS_MEAN_TOL,
                    "unit_variance": abs(s["std"] - 1.0) <= ZS_STD_TOL,
                    "zscore_consistent": abs(s["mean"]) <= ZS_MEAN_TOL and abs(s["std"] - 1.0) <= ZS_STD_TOL})
        print(f"[{i + 1}/{len(vhdrs)}] {fn}: {n_times} samples, {len(ecog)} ECOG, {len(seeg)} SEEG, {len(mov)} MOV",
              flush=True)

    runs.sort(key=lambda x: (x["subject"], x["session"], x["run"]))
    for lst in (force_rows, elec_rows, stn_rows, sample_rows):
        lst.sort(key=lambda x: (x["subject"], x["session"], x["run"], x.get("channel", "")))
    cents = np.array([[s["centroid_absx_mm"], s["centroid_y_mm"], s["centroid_z_mm"]] for s in stn_rows
                      if s["centroid_absx_mm"] is not None], dtype=float)
    med_cen = np.median(cents, axis=0) if len(cents) else np.full(3, np.nan)
    for s in stn_rows:
        if s["centroid_absx_mm"] is None:
            s["centroid_dist_from_median_mm"], s["far_from_other_stn_runs"] = None, None
            continue
        dd = float(np.linalg.norm(np.array([s["centroid_absx_mm"], s["centroid_y_mm"], s["centroid_z_mm"]]) - med_cen))
        s["centroid_dist_from_median_mm"] = r3(dd)
        s["far_from_other_stn_runs"] = dd > STN_FAR_MM
    near = [s for s in stn_rows if s["far_from_other_stn_runs"] is False]
    write_csv(os.path.join(a.out, "runs.csv"), runs)
    write_csv(os.path.join(a.out, "force.csv"), force_rows)
    write_csv(os.path.join(a.out, "electrodes_mni.csv"), elec_rows)
    write_csv(os.path.join(a.out, "stn_mni.csv"), stn_rows)
    write_csv(os.path.join(a.out, "subject_electrodes.csv"), subj_rows)
    write_csv(os.path.join(a.out, "samples_per_run.csv"), sample_rows)

    # ---------------------------------------------------------------- facts
    dur = np.array([x["duration_s"] for x in runs])
    subs_disk = sorted({x["subject"] for x in runs})
    stn_runs = [x for x in runs if x["stn_present"]]
    stn_subs = sorted({x["subject"] for x in stn_runs})
    ecog_runs = [x for x in runs if x["n_ecog"] > 0]
    mov_lr = [x for x in runs if {"MOV_LEFT", "MOV_RIGHT"} <= set(x["force_channels"].split("|"))]
    fr_main = [f for f in force_rows if f["channel"] in ("MOV_LEFT", "MOV_RIGHT")]
    fr_clean = [f for f in force_rows if f["channel"].endswith("_CLEAN")]
    stn_with_mni = [e for e in elec_rows if e["seeg_rows"] > 0 and e["seeg_finite"] == e["seeg_rows"]]
    ecog_with_mni = [e for e in elec_rows if e["ecog_rows"] > 0 and e["ecog_finite"] == e["ecog_rows"]]
    seeg_runs_keys = {(x["subject"], x["session"], x["run"]) for x in stn_runs}
    stn_mni_keys = {(e["subject"], e["session"], e["run"]) for e in stn_with_mni}
    subj_stn = sorted({s["subject"] for s in subj_rows if s["stn_rows"] > 0})
    subj_files_cs = sorted((cs_, u_, sum(1 for s in subj_rows if (s["coordsystem"], s["units"]) == (cs_, u_)))
                           for cs_, u_ in {(s["coordsystem"], s["units"]) for s in subj_rows})
    sess_cs = sorted({(e["session_coordsystem"], e["session_units"]) for e in elec_rows})
    max_diff = max((e["max_abs_diff_mm"] for e in elec_rows if e["max_abs_diff_mm"] is not None), default=None)
    unmatched_rows = sum((e["ecog_finite"] + e["seeg_finite"]) - e["rows_matched"] for e in elec_rows)
    sides = {}
    for x in runs:
        sides.setdefault(x["subject"], set()).add(x["session"])
    n_types = {"ECOG": sum(x["n_ecog"] for x in runs), "SEEG": sum(x["n_seeg"] for x in runs),
               "MISC": sum(x["n_misc"] for x in runs)}
    stn_by_side = {}
    for x in runs:
        for n in filter(None, x["stn_channels"].split("|")):
            k = "STN_" + side_of(n).upper()
            stn_by_side[k] = stn_by_side.get(k, 0) + 1
    ses_mismatch = [x for x in runs if x["ecog_side"] and ("ses-" + x["ecog_side"]) != x["session"]]
    eq_pattern = {}
    for s in stn_rows:
        pat = "; ".join(p.split("=")[1].rsplit("_", 1)[-1] if "=" in p else p for p in s["run_row_equals_subject_contact"].split("; "))
        eq_pattern[pat] = eq_pattern.get(pat, 0) + 1
    facts = {
        "raw_files": len(files), "raw_bytes": raw_bytes, "runs": len(runs),
        "participants_tsv_rows": len(participants), "subjects_on_disk": len(subs_disk),
        "participants_disease": sorted({p.get("disease", "") for p in participants}),
        "runs_per_subject": {s: sum(1 for x in runs if x["subject"] == s) for s in subs_disk},
        "subjects_two_sessions": sum(1 for v in sides.values() if len(v) > 1),
        "sfreq_values": sorted({x["sfreq_hz"] for x in runs}),
        "duration_s": {"min": float(dur.min()), "median": float(np.median(dur)), "max": float(dur.max()),
                       "total_h": float(dur.sum() / 3600.0)},
        "ecog_runs": len(ecog_runs), "stn_runs": len(stn_runs), "stn_subjects": len(stn_subs),
        "channel_totals": n_types, "stn_channels_by_side": stn_by_side,
        "runs_with_mov_left_and_right": len(mov_lr),
        "force_main": {"runs_channels": len(fr_main),
                       "centred": sum(1 for f in fr_main if f["centred"]),
                       "unit_variance": sum(1 for f in fr_main if f["unit_variance"]),
                       "zscore_consistent": sum(1 for f in fr_main if f["zscore_consistent"]),
                       "abs_mean_max": max((abs(f["mean"]) for f in fr_main), default=None),
                       "std_min": min((f["std"] for f in fr_main), default=None),
                       "std_median": float(np.median([f["std"] for f in fr_main])) if fr_main else None,
                       "std_max": max((f["std"] for f in fr_main), default=None),
                       "header_units": sorted({f["header_unit"] for f in fr_main}),
                       "tsv_units": sorted({f["tsv_units"] for f in fr_main})},
        "force_clean": {"runs_channels": len(fr_clean),
                        "share_zero_min": min((f["share_zero"] for f in fr_clean), default=None),
                        "share_zero_median": float(np.median([f["share_zero"] for f in fr_clean])) if fr_clean else None,
                        "share_zero_max": max((f["share_zero"] for f in fr_clean), default=None),
                        "max_equal_one": sum(1 for f in fr_clean if abs(f["max"] - 1.0) <= 1e-6),
                        "min_equal_zero": sum(1 for f in fr_clean if f["min"] == 0),
                        "share_one_max": max((f["share_one"] for f in fr_clean), default=None),
                        "min": min((f["min"] for f in fr_clean), default=None),
                        "max": max((f["max"] for f in fr_clean), default=None)},
        "mni": {"run_files": sum(1 for e in elec_rows if e["file_present"]),
                "runs_ecog_all_finite": len(ecog_with_mni), "runs_stn_all_finite": len(stn_with_mni),
                "stn_runs_with_mni": len(seeg_runs_keys & stn_mni_keys),
                "ecog_rows_total": sum(e["ecog_rows"] for e in elec_rows),
                "ecog_rows_finite": sum(e["ecog_finite"] for e in elec_rows),
                "seeg_rows_total": sum(e["seeg_rows"] for e in elec_rows),
                "seeg_rows_finite": sum(e["seeg_finite"] for e in elec_rows),
                "session_coordsystems": [list(x) for x in sess_cs],
                "subject_files": len(subj_rows), "subject_coordsystems": [list(x) for x in subj_files_cs],
                "subjects_with_stn_contacts_in_subject_files": len(subj_stn),
                "subjects_with_stn_runs": len(stn_subs),
                "stn_run_subjects_without_subject_stn_rows": sorted(set(stn_subs) - set(subj_stn)),
                "max_abs_diff_run_vs_subject_mm": max_diff, "finite_run_rows_unmatched": int(unmatched_rows),
                "runs_session_units_unknown": sum(1 for e in elec_rows if e["rows"] and not e["session_units_known"]),
                "runs_with_rows_without_diff": sum(1 for e in elec_rows if (e["ecog_finite"] + e["seeg_finite"]) > 0
                                                   and e["max_abs_diff_mm"] is None),
                "stn_rows_equal_subject_contacts_pattern": eq_pattern,
                "subject_lead_contacts": sorted({s["subject_lead_contacts"] for s in stn_rows}),
                "stn_x_abs_mm": [min(min(abs(s["x_mm_min"]), abs(s["x_mm_max"])) for s in stn_rows),
                                 max(max(abs(s["x_mm_min"]), abs(s["x_mm_max"])) for s in stn_rows)] if stn_rows else None,
                "stn_y_mm": [min(s["y_mm_min"] for s in stn_rows), max(s["y_mm_max"] for s in stn_rows)] if stn_rows else None,
                "stn_z_mm": [min(s["z_mm_min"] for s in stn_rows), max(s["z_mm_max"] for s in stn_rows)] if stn_rows else None,
                "stn_median_centroid_mm": [r3(v) for v in med_cen],
                "stn_far_runs": [f"{s['subject']} {s['session']} run-{s['run']} ({s['centroid_dist_from_median_mm']:.1f} mm)"
                                 for s in stn_rows if s["far_from_other_stn_runs"]],
                "stn_far_subjects": sorted({s["subject"] for s in stn_rows if s["far_from_other_stn_runs"]}),
                "stn_near_runs": sum(1 for s in stn_rows if s["far_from_other_stn_runs"] is False),
                "stn_near_x_abs_mm": [min(min(abs(s["x_mm_min"]), abs(s["x_mm_max"])) for s in near),
                                      max(max(abs(s["x_mm_min"]), abs(s["x_mm_max"])) for s in near)] if near else None,
                "stn_near_y_mm": [min(s["y_mm_min"] for s in near), max(s["y_mm_max"] for s in near)] if near else None,
                "stn_near_z_mm": [min(s["z_mm_min"] for s in near), max(s["z_mm_max"] for s in near)] if near else None},
        "samples": {"bytes_match": sum(1 for s in sample_rows if s["bytes_match"]),
                    "json_duration_match": sum(1 for s in sample_rows if s["json_duration_match"]),
                    "neural_and_force_in_one_file": sum(1 for s in sample_rows if s["neural_and_force_in_one_file"]),
                    "vhdr_datapoints_declared": sum(1 for s in sample_rows if s["vhdr_datapoints"] != ""),
                    "vhdr_datapoints_match": sum(1 for s in sample_rows if s["vhdr_datapoints_match"] is True),
                    "sample_counts_match": sum(1 for s in sample_rows if s["sample_counts_match"]),
                    "stn_runs_sample_counts_match": sum(1 for s in sample_rows if s["sample_counts_match"]
                                                        and (s["subject"], s["session"], s["run"]) in seeg_runs_keys),
                    "n_samples_min": min(s["n_samples"] for s in sample_rows),
                    "n_samples_max": max(s["n_samples"] for s in sample_rows)},
        "ieeg_reference_values": sorted({str(x["json_iEEGReference"]) for x in runs}),
        "ecog_side_differs_from_session": len(ses_mismatch),
        "bad_channels_total": sum(x["bad_channels"] for x in runs),
        "markers_total": sum(x["markers"] or 0 for x in runs),
        "ea_ui_mat_files": sorted(f for f in files if f.startswith("ea_ui")),
    }
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1, default=float)

    # ---------------------------------------------------------------- person map
    pm_rows = []
    cids = [p.get("center_ID", "") for p in participants]
    dx = [p.get("Length_of_diagnosis_prior_to_surgery", "") for p in participants]
    for p in participants:
        pid = p["participant_id"]
        rr = [x for x in runs if x["subject"] == pid]
        ses_counts = {}
        for x in rr:
            ses_counts[x["session"]] = ses_counts.get(x["session"], 0) + 1
        n_stn = sum(1 for x in rr if x["stn_present"])
        efiles = [s for s in subj_rows if s["subject"] == pid]
        cid_unique = cids.count(p.get("center_ID", "")) == 1 and p.get("center_ID", "") not in ("", "n/a")
        dx_unique = dx.count(p.get("Length_of_diagnosis_prior_to_surgery", "")) == 1
        ev = (f"participants.tsv row {pid}: disease {p.get('disease', '')}, center {p.get('center', '')}; its center_ID "
              f"{'is shared by no other row' if cid_unique else 'is shared with another row or missing'}; its "
              f"Length_of_diagnosis_prior_to_surgery value {'is shared by no other row' if dx_unique else 'is shared with another row'}; "
              f"raw files: {len(rr)} runs ("
              + ", ".join(f"{k} {v}" for k, v in sorted(ses_counts.items()))
              + f"), ECoG in {sum(1 for x in rr if x['n_ecog'] > 0)}, STN LFP in {n_stn}; per-subject electrodes files "
              f"{len(efiles)} ({', '.join(s['coordsystem'] + ' ' + s['units'] for s in efiles)})")
        amb = "" if (cid_unique and rr) else ("no run on disk" if not rr else "center_ID shared or missing")
        pm_rows.append({"source": SRC, "subject_ns": f"{SRC}:{pid}", "person_id": f"{SRC}:{pid}",
                        "rule": "bids_participant_distinct_center_id", "evidence": ev, "ambiguity": amb})
    os.makedirs(os.path.dirname(a.person_map), exist_ok=True)
    write_csv(a.person_map, pm_rows, ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"])
    facts["person_map"] = {"rows": len(pm_rows), "persons": len({r["person_id"] for r in pm_rows}),
                           "ambiguous": sum(1 for r in pm_rows if r["ambiguity"])}

    # ---------------------------------------------------------------- checks
    checks = []

    def add(cid, claim, source, measured, verdict):
        checks.append({"check_id": cid, "claim": claim, "source": source, "measured": measured, "verdict": verdict})

    def ok(*c):
        return "confirmed" if all(c) else "corrected"

    add("MRK-01", "16 participant ids; the paper analyses 11 PD", "addendum 1.1, 1.5; OPEN-63",
        f"participants.tsv {len(participants)} rows, disease {'|'.join(facts['participants_disease'])}; "
        f"{len(subs_disk)} subject prefixes with runs on disk; {facts['subjects_two_sessions']} subjects with runs in both "
        f"sessions (left and right); which 11 the paper analysed is not stated in the files",
        "qualified" if (len(participants) == 16 and len(subs_disk) == 16) else "corrected")
    add("MRK-02", "Sensorimotor ECoG strip in 73 runs; STN LFP in 20 of 73; 1000 Hz", "addendum 1.1",
        f"{len(runs)} runs; ECoG channels in {len(ecog_runs)}; STN (SEEG) channels in {len(stn_runs)} runs of "
        f"{len(stn_subs)} subjects; rates {'|'.join(f'{v:g}' for v in facts['sfreq_values'])} Hz (the strip's "
        f"sensorimotor position is not checked)",
        ok(len(runs) == 73, len(ecog_runs) == 73, len(stn_runs) == 20, facts["sfreq_values"] == [1000.0]))
    fm = facts["force_main"]
    fc = facts["force_clean"]
    add("MRK-03", "Grip force left and right in all 73 runs, z-scored, so unitless", "addendum 1.1, 1.6 item 12",
        f"MOV_LEFT and MOV_RIGHT in {len(mov_lr)} of {len(runs)} runs. Of {fm['runs_channels']} run-channels, "
        f"{fm['centred']} are centred (|mean| <= {ZS_MEAN_TOL}; max |mean| {fm['abs_mean_max']:.2g}) and "
        f"{fm['unit_variance']} have unit variance (|std - 1| <= {ZS_STD_TOL}); std {fm['std_min']:.4f} to "
        f"{fm['std_max']:.4f}, median {fm['std_median']:.4f}: standardized values, but not to unit variance within "
        f"each run as stored: centred on the run, scaled by a factor other than the run's own standard deviation; "
        f"the files label them "
        f"{'|'.join(fm['header_units'])} (.vhdr) and {'|'.join(fm['tsv_units'])} (channels.tsv), a unit the values do "
        f"not carry. MOV_*_CLEAN ({fc['runs_channels']} run-channels): minimum 0 in {fc['min_equal_zero']}, maximum "
        f"exactly 1 in {fc['max_equal_one']}, zero in {fc['share_zero_min']:.3f} to {fc['share_zero_max']:.3f} of the "
        f"samples (median {fc['share_zero_median']:.3f}): a rescaled trace, not the force",
        ("confirmed" if fm["zscore_consistent"] == fm["runs_channels"]
         else "qualified" if fm["centred"] == fm["runs_channels"] else "corrected")
        if (len(mov_lr) == 73 and fm["runs_channels"] == 2 * len(mov_lr)) else "corrected")
    d = facts["duration_s"]
    add("MRK-04", "Run durations 39 to 569 s, median 288 s", "addendum 1.1, 1.6 item 12, 9; NEURAL_RULES_DRAFT F4 table",
        f"{len(runs)} runs; {d['min']:.1f} to {d['max']:.1f} s, median {d['median']:.1f} s; {d['total_h']:.2f} h in all",
        ok(round(d["min"]) == 39, round(d["max"]) == 569, round(d["median"]) == 288))
    add("MRK-05", "Datacard: 10-25 min/run", "datacard (addendum 1.6 item 12)",
        f"{sum(1 for x in runs if 600 <= x['duration_s'] <= 1500)} of {len(runs)} runs last 10 to 25 min; the longest "
        f"is {d['max'] / 60:.1f} min", ok(all(600 <= x["duration_s"] <= 1500 for x in runs)))
    add("MRK-06", "1.4 GB on disk; datacard: 537 files", "addendum 1.1; datacard",
        f"{len(files)} files (AppleDouble sidecars excluded), {raw_bytes / 1e9:.3f} GB",
        ok(round(raw_bytes / 1e9, 1) == 1.4, len(files) == 537))
    add("MRK-07", "Datacard: 871 ECOG, 60 SEEG (STN_LEFT 36 + STN_RIGHT 24), 292 MISC channels over the 73 runs",
        "datacard", f"ECOG {n_types['ECOG']}, SEEG {n_types['SEEG']} ("
        + " + ".join(f"{k} {v}" for k, v in sorted(stn_by_side.items())) + f"), MISC {n_types['MISC']}",
        ok(n_types["ECOG"] == 871, n_types["SEEG"] == 60, n_types["MISC"] == 292,
           stn_by_side.get("STN_LEFT") == 36, stn_by_side.get("STN_RIGHT") == 24))
    mn = facts["mni"]
    add("MRK-08", "No imaging localizes contacts for any in-scope cohort; every in-scope set is without contact "
        "localization (N-LFP token identity is lead level and montage, never a nucleus)",
        "addendum 2.2; NR-3.11",
        f"merk: {mn['run_files']} run-level space-mni electrodes files; ECoG rows with finite coordinates in "
        f"{mn['runs_ecog_all_finite']} runs ({mn['ecog_rows_finite']} of {mn['ecog_rows_total']} rows); STN rows with "
        f"finite coordinates in {mn['stn_runs_with_mni']} of the {len(stn_runs)} STN runs ({mn['seeg_rows_finite']} of "
        f"{mn['seeg_rows_total']} rows); {mn['subject_files']} per-subject electrodes files declared "
        + "; ".join(f"{s} in {u} ({n})" for s, u, n in mn["subject_coordsystems"])
        + f", {mn['subjects_with_stn_contacts_in_subject_files']} subjects with STN contacts in them; Lead-DBS files: "
        + ", ".join(facts["ea_ui_mat_files"]),
        "corrected" if mn["stn_runs_with_mni"] > 0 else "confirmed")
    add("MRK-09", "merk has MNI coordinates for STN contacts in the runs that carry STN LFP", "crosswalk appendix item 1",
        f"{mn['stn_runs_with_mni']} of {len(stn_runs)} STN runs ({len(stn_subs)} subjects) carry finite STN coordinates; "
        f"run rows equal the per-subject rows within {mn['max_abs_diff_run_vs_subject_mm']} mm after unit conversion "
        f"({mn['finite_run_rows_unmatched']} finite run rows without a per-subject row; "
        f"{mn['runs_session_units_unknown']} runs with an unknown session unit, {mn['runs_with_rows_without_diff']} "
        f"runs with finite rows and no comparison); STN coordinates of the "
        f"{mn['stn_near_runs']} runs within {STN_FAR_MM:g} mm of the median centroid: |x| "
        f"{mn['stn_near_x_abs_mm'][0]:.1f} to {mn['stn_near_x_abs_mm'][1]:.1f} mm, y {mn['stn_near_y_mm'][0]:.1f} to "
        f"{mn['stn_near_y_mm'][1]:.1f} mm, z {mn['stn_near_z_mm'][0]:.1f} to {mn['stn_near_z_mm'][1]:.1f} mm; "
        f"farther: " + ("; ".join(mn["stn_far_runs"]) or "none"),
        ("confirmed" if not mn["stn_far_runs"] else "qualified")
        if (mn["stn_runs_with_mni"] == len(stn_runs) and len(stn_runs) > 0 and mn["runs_session_units_unknown"] == 0
            and mn["runs_with_rows_without_diff"] == 0) else "corrected")
    add("MRK-10", "Datacard: 3 bipolar STN channels", "datacard; NR-3.11 (montage)",
        f"STN channels per STN run: {'|'.join(str(v) for v in sorted({s['stn_channels'] for s in stn_rows}))}; the "
        f"run rows equal per-subject lead contacts (pattern over runs: "
        + "; ".join(f"[{k}] in {v}" for k, v in sorted(eq_pattern.items()))
        + f") of a {'|'.join(map(str, mn['subject_lead_contacts']))}-contact lead; iEEGReference "
        f"{'|'.join(facts['ieeg_reference_values'])} in every sidecar, so the files do not declare the montage",
        "qualified")
    sm = facts["samples"]
    n_dp = sm["vhdr_datapoints_declared"]
    dp_note = (f"no .vhdr declares DataPoints, so the sample count is the byte size over channels x 4 (as MNE "
               f"reads it) and the byte check tests that the size divides evenly: it does"
               if n_dp == 0 else
               f".vhdr DataPoints declared in {n_dp} runs and equal to the sample count in "
               f"{sm['vhdr_datapoints_match']}; byte size equals channels x samples x 4")
    add("MRK-11", "G15 line for merk: matching sample counts per run (OPEN-12)", "NR-2.12; OPEN-12",
        f"neural and force channels share one multiplexed binary in {sm['neural_and_force_in_one_file']} of {len(runs)} "
        f"runs, so their sample counts are equal by construction; {dp_note} in {sm['bytes_match']}; the independent "
        f"check, sidecar RecordingDuration equal to (samples - 1) / rate within one sample, holds in "
        f"{sm['json_duration_match']}; all checks in {sm['sample_counts_match']} runs, "
        f"{sm['stn_runs_sample_counts_match']} of them with STN; samples per run {sm['n_samples_min']:,} to "
        f"{sm['n_samples_max']:,} (samples_per_run.csv)",
        ok(sm["sample_counts_match"] == len(runs)))
    add("MRK-12", "Movement-disorder patients with brain and body on one clock: merk 11 to 16 PD", "addendum 1.5",
        f"{len(participants)} participants, all {'|'.join(facts['participants_disease'])}, {len(subs_disk)} with runs; "
        f"person map: {facts['person_map']['persons']} persons, {facts['person_map']['ambiguous']} ambiguous rows",
        ok(len(subs_disk) == 16, facts["participants_disease"] == ["PD"]))
    write_csv(os.path.join(a.out, "addendum_checks.csv"), checks,
              ["check_id", "claim", "source", "measured", "verdict"])

    # ---------------------------------------------------------------- provenance and summary
    import moveatlas_stamp
    prov = {"script": os.path.basename(__file__), "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "mne": mne.__version__, "moveatlas_root": root,
            "moveatlas_stamp": moveatlas_stamp.stamp(root), "source": RAW_REL,
            "raw_files": len(files), "raw_bytes": raw_bytes,
            "participants_tsv_sha256": hashlib.sha256(open(os.path.join(raw, "participants.tsv"), "rb").read()).hexdigest(),
            "parameters": {"ZS_MEAN_TOL": ZS_MEAN_TOL, "ZS_STD_TOL": ZS_STD_TOL, "STN_FAR_MM": STN_FAR_MM,
                           "CHUNK": CHUNK},
            "slow_waited_s": slow_waited, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1, default=str)
    write_summary(a.out, checks, facts, prov, a.person_map)
    vc = {}
    for c in checks:
        vc[c["verdict"]] = vc.get(c["verdict"], 0) + 1
    print(f"{len(checks)} checks: {vc}; {time.time() - t0:.0f} s")


def write_summary(out, checks, facts, prov, person_map):
    mn, sm, fm, d = facts["mni"], facts["samples"], facts["force_main"], facts["duration_s"]
    try:
        pm_rel = os.path.relpath(person_map, TREE).replace("\\", "/")
    except ValueError:  # another drive than the tree (Windows): keep the absolute path
        pm_rel = os.path.abspath(person_map).replace("\\", "/")
    L = ["# merk_neumann_ecog_lfp: reader facts, MNI electrode rows, force and samples per run", "",
         f"Generated by scripts/probe_merk.py on {prov['run_at'][:10]} against MoveAtlas "
         f"{prov['moveatlas_stamp'].get('head', '?')} (stamp in _provenance.json), from {prov['source']} "
         f"({facts['raw_files']} files, {facts['raw_bytes'] / 1e9:.3f} GB). Every number below is in facts.json or a "
         f"CSV of this folder.", "",
         "## The MNI fact", "",
         f"The set localizes its contacts. Of the {facts['stn_runs']} runs with STN LFP ({facts['stn_subjects']} subjects), "
         f"{mn['stn_runs_with_mni']} carry finite MNI coordinates for every STN channel ({mn['seeg_rows_finite']} rows); "
         f"{mn['runs_ecog_all_finite']} of {facts['runs']} runs carry them for every ECoG channel "
         f"({mn['ecog_rows_finite']} of {mn['ecog_rows_total']} rows). The run files are named space-mni while their "
         f"session coordsystem files declare "
         + "; ".join(f"{s} in {u}" for s, u in mn["session_coordsystems"])
         + f"; the {mn['subject_files']} per-subject electrodes files declare "
         + "; ".join(f"{s} in {u} ({n} file{'' if n == 1 else 's'})" for s, u, n in mn["subject_coordsystems"])
         + f", and the run rows, converted to millimetres, equal the per-subject rows within "
         f"{mn['max_abs_diff_run_vs_subject_mm']} mm. The STN coordinates of {mn['stn_near_runs']} runs lie within "
         f"{STN_FAR_MM:g} mm of the median centroid (|x| {mn['stn_median_centroid_mm'][0]}, y "
         f"{mn['stn_median_centroid_mm'][1]}, z {mn['stn_median_centroid_mm'][2]} mm) and span |x| "
         f"{mn['stn_near_x_abs_mm'][0]:.1f} to {mn['stn_near_x_abs_mm'][1]:.1f} mm, y {mn['stn_near_y_mm'][0]:.1f} to "
         f"{mn['stn_near_y_mm'][1]:.1f} mm, z {mn['stn_near_z_mm'][0]:.1f} to {mn['stn_near_z_mm'][1]:.1f} mm; "
         + (f"the runs of {', '.join(mn['stn_far_subjects'])} lie farther ({'; '.join(mn['stn_far_runs'])}), so their "
            "STN rows need a check before any location-dependent use. " if mn["stn_far_runs"] else "no run lies farther. ")
         + "This contradicts addendum 2.2 and NR-3.11, which state that no in-scope set has contact localization "
         "(check MRK-08); the crosswalk appendix, item 1, keeps the statement for the dystonia sets. The montage of "
         "the STN channels is not declared in the run files (check MRK-10).", "",
         "## Runs, signals and samples", "",
         f"- {facts['runs']} runs of {facts['subjects_on_disk']} subjects at "
         f"{'|'.join(f'{v:g}' for v in facts['sfreq_values'])} Hz; {d['min']:.1f} to {d['max']:.1f} s, median "
         f"{d['median']:.1f} s, {d['total_h']:.2f} h in all; {facts['subjects_two_sessions']} subjects recorded in both "
         "sessions (left and right hemisphere).",
         f"- ECoG in {facts['ecog_runs']} runs; STN (SEEG) in {facts['stn_runs']}; channel totals "
         + ", ".join(f"{k} {v}" for k, v in facts["channel_totals"].items()) + ".",
         f"- Force: MOV_LEFT and MOV_RIGHT in {facts['runs_with_mov_left_and_right']} runs. {fm['centred']} of "
         f"{fm['runs_channels']} run-channels are centred, {fm['unit_variance']} have unit variance (std "
         f"{fm['std_min']:.4f} to {fm['std_max']:.4f}): centred on each run, scaled by a factor other than the run's own "
         f"standard deviation. The "
         f"files label them {'|'.join(fm['header_units'])}; they carry no physical unit, so a force descriptor on merk "
         "is relative within a run. MOV_*_CLEAN channels are zero in most samples and peak at exactly 1 (force.csv): "
         "a rescaled trace, not the force.",
         f"- OPEN-12: neural and force channels share one multiplexed binary in every run that has both, so their sample "
         f"counts match by construction; {sm['sample_counts_match']} of {facts['runs']} runs pass the byte and sidecar "
         f"duration checks ({sm['vhdr_datapoints_declared']} .vhdr files declare DataPoints; without it the byte check "
         f"only tests an even division and the sidecar duration is the independent check). samples_per_run.csv lists "
         f"every run, the candidate admitted list of the G15 line.", "",
         "## Verdicts", "", "| id | claim | source | measured | verdict |", "|---|---|---|---|---|"]
    for c in checks:
        L.append(f"| {c['check_id']} | {c['claim']} | {c['source']} | {c['measured']} | {c['verdict']} |")
    L += ["", "## Person map", "",
          f"{pm_rel}: {facts['person_map']['rows']} rows, one per participants.tsv row, {facts['person_map']['persons']} "
          f"persons, {facts['person_map']['ambiguous']} flagged. Rule: the BIDS participant id is the person; no two rows "
          "share a center_ID. Both hemisphere sessions of one id are one person. The datacard cites 11 PD analysed by "
          "the source paper; the files do not name them (OPEN-63).", "",
          "## Limits", "",
          "- The sensorimotor position of the ECoG strips and the nucleus of the STN contacts are not checked against an "
          "atlas; the coordinate ranges above are descriptive.",
          "- Medication state is not in the files read (addendum 1.7)."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
