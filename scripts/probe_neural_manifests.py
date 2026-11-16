"""Neural census from the MoveAtlas canonical manifests: one verdict per addendum claim.

Weekend task T23 (1); stage N0 port of the manifest checks of addendum 1 section 9
(docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md, sections 1.1 to 1.5, 2.2 and 9).

Inputs (read-only, MoveAtlas tree):
  datasets/_canonical_*_manifest.csv   every file of the glob is hashed and counted; the
                                       eeg, emg, imu and force manifests carry the checks
  meta.json member of the canonical EMG records of sensefog, multimodal_fog and
  physionet_multimodal_gait, and of the canonical EEG records of physionet_multimodal_gait
                                       (only that member of each .npz is read, never a signal)
  datasets/eeg/physionet_multimodal_gait/raw/Dataset/EEG_Data/EEG.csv/*_raw.csv
                                       the header lines up to the channel-name row only; the
                                       'Reference location' line is the one value kept
  datasets/eeg/pd_rest_eeg_ds004584/raw/sub-*/eeg/*_eeg.json
                                       the EEGReference field only

Outputs (reports/neural_manifests/ by default):
  manifests.csv        one row per manifest file of the glob: rows, sources, bytes, SHA-256
  per_source.csv       one row per family and source: records, subject ids, hours, rates,
                       channel counts, sites or locations, head-sensor records, record seconds
  emg_sites.csv        EMG channels by kind, muscle and side for the three sets declared
                       synchronized, with the number of records that carry each
  addendum_checks.csv  claim, source, measured, verdict (confirmed, corrected, qualified)
  facts.json           every number used by the checks
  summary.md           the same in prose
  _provenance.json     MoveAtlas stamp (scripts/moveatlas_stamp.py) and input hashes

Definitions. Hours are n_frames / fps summed per source. Subject ids are distinct subject_ns
values (namespaced per source, so sums over sources do not collide). "PD and healthy EEG" is
every EEG source other than seizeit2 (epilepsy) and cap_sleep (sleep): the manifests carry
no diagnosis, so the grouping is by source, as in the addendum. The three sets declared
synchronized are sensefog, multimodal_fog and physionet_multimodal_gait (addendum 1.1).
Percentiles use numpy's linear interpolation.

Descriptive only: no statistic links a neural signal to a movement or an outcome.
The process lowers its own priority to below normal and waits while
cache/runs/queue/SLOW exists.

Run: python scripts/probe_neural_manifests.py [--moveatlas <moveatlas>] [--out DIR]
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
import datetime as _dt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SLOW = os.path.join(TREE, "cache", "runs", "queue", "SLOW")

SYNC_SETS = ["sensefog", "multimodal_fog", "physionet_multimodal_gait"]
NON_PD_HEALTHY_EEG = ["seizeit2", "cap_sleep"]
EMG_META_SOURCES = ["sensefog", "multimodal_fog", "physionet_multimodal_gait"]
NEURAL_WALKING_IMU = ["sensefog", "multimodal_fog", "physionet_multimodal_gait"]
PHYSIONET_RAW_EEG = "datasets/eeg/physionet_multimodal_gait/raw/Dataset/EEG_Data/EEG.csv"
DS004584_RAW = "datasets/eeg/pd_rest_eeg_ds004584/raw"


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


# ---------------------------------------------------------------- helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def fnum(x, nd):
    return f"{x:,.{nd}f}"


def fset(values):
    """Distinct values at the printed precision (6 significant digits), ascending.

    Rates computed from timestamps differ in the last float digits (1776.1989342761485 and
    1776.1989342803518): they are one rate, so the set is taken after formatting.
    """
    out = []
    for v in sorted({float(v) for v in values}):
        s = f"{v:g}"
        if s not in out:
            out.append(s)
    return "|".join(out)


def npz_meta(path):
    with np.load(path, allow_pickle=False) as z:
        raw = z["meta.json"]
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    return json.loads(raw)["meta"]


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def provenance(out, root, extra):
    import moveatlas_stamp
    prov = {"script": os.path.basename(__file__),
            "run_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "moveatlas_root": root,
            "moveatlas_stamp": moveatlas_stamp.stamp(root)}
    prov.update(extra)
    with open(os.path.join(out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1, default=str)
    return prov


# ---------------------------------------------------------------- measurement
def per_source_tables(man):
    rows = []
    for fam in ["eeg", "emg", "imu", "force"]:
        d = man[fam].copy()
        d["fpsf"] = d.fps.astype(float)
        d["sec"] = d.n_frames.astype(float) / d.fpsf
        for src, g in d.groupby("source", sort=True):
            r = {"family": fam, "source": src, "records": len(g),
                 "subject_ids": g.subject_ns.nunique(),
                 "hours": round(g.sec.sum() / 3600.0, 4),
                 "fps_values": fset(g.fps),
                 "n_channels_values": "", "sites_or_locations": "", "n_sites": "",
                 "site_combinations": "", "head_records": "",
                 "record_s_min": round(g.sec.min(), 3),
                 "record_s_p10": round(float(np.percentile(g.sec, 10)), 3),
                 "record_s_median": round(float(np.median(g.sec)), 3),
                 "record_s_max": round(g.sec.max(), 3)}
            if "n_channels" in g:
                r["n_channels_values"] = "|".join(str(v) for v in sorted({int(x) for x in g.n_channels}))
            col = {"eeg": "sites", "emg": "kinds", "imu": "locations"}.get(fam)
            if col:
                sets = [tuple(x for x in s.split("|") if x) for s in g[col]]
                union = sorted({x for s in sets for x in s})
                r["sites_or_locations"] = "|".join(union)
                r["n_sites"] = len(union)
                r["site_combinations"] = len(set(sets))
                if fam == "imu":
                    r["head_records"] = sum("head" in s for s in sets)
            rows.append(r)
    return pd.DataFrame(rows)


def emg_meta(root, emg):
    """Per-record EMG channel identity from the record meta, for the synchronized sets."""
    rows, per_record = [], {}
    for src in EMG_META_SOURCES:
        g = emg[emg.source == src]
        counts = {}
        rec = []
        for p in g.record_path:
            wait_if_slow()
            meta = npz_meta(os.path.join(root, p))
            ch = meta.get("channels") or []
            if isinstance(ch, str):
                ch = json.loads(ch)
            kinds = [c.get("kind") for c in ch]
            rec.append({"n": len(ch), "emg": kinds.count("emg"),
                        "other": {k: kinds.count(k) for k in set(kinds) if k != "emg"},
                        "fps": meta.get("fps"), "signal_kind": meta.get("signal_kind")})
            keys = [(c.get("kind"), c.get("muscle") or "", c.get("side") or "") for c in ch]
            for key in set(keys):
                nrec, nch = counts.get(key, (0, 0))
                counts[key] = (nrec + 1, nch + keys.count(key))
        for (kind, muscle, side), (nrec, nch) in sorted(counts.items()):
            rows.append({"source": src, "kind": kind, "muscle": muscle, "side": side,
                         "records_with_channel": nrec, "channels": nch, "records": len(g)})
        per_record[src] = rec
    return rows, per_record


def physionet_reference(root, eeg):
    g = eeg[eeg.source == "physionet_multimodal_gait"]
    all_pz = 0
    refs = {}
    for p in g.record_path:
        wait_if_slow()
        meta = npz_meta(os.path.join(root, p))
        ch = meta.get("channels") or []
        if isinstance(ch, str):
            ch = json.loads(ch)
        rr = {str(c.get("ref")).upper() for c in ch}
        for r in rr:
            refs[r] = refs.get(r, 0) + 1
        if rr == {"PZ"}:
            all_pz += 1
    raw_dir = os.path.join(root, PHYSIONET_RAW_EEG)
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith("_raw.csv") and not f.startswith("._"))
    header_ref = {}
    for fn in files:
        val = None
        with open(os.path.join(raw_dir, fn), encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if line.startswith("Time,") or i > 40:
                    break
                if line.startswith("# Reference location"):
                    parts = line.rstrip("\n").split(",")
                    val = parts[1].strip() if len(parts) > 1 else ""
        header_ref[val or "(absent)"] = header_ref.get(val or "(absent)", 0) + 1
    return {"records": len(g), "records_all_channels_ref_pz": all_pz, "ref_values_by_records": refs,
            "raw_csv_files": len(files), "raw_header_reference_counts": header_ref}


def ds004584_reference(root):
    files = sorted(glob.glob(os.path.join(root, DS004584_RAW, "sub-*", "eeg", "*_eeg.json")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    vals = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            v = json.load(fh).get("EEGReference", "(absent)")
        vals[str(v)] = vals.get(str(v), 0) + 1
    return {"sidecars": len(files), "EEGReference_counts": vals}


# ---------------------------------------------------------------- checks
class Checks:
    def __init__(self):
        self.rows = []

    def add(self, cid, claim, source, measured, verdict):
        self.rows.append({"check_id": cid, "claim": claim, "source": source,
                          "measured": measured, "verdict": verdict})


def all_true(*conds):
    return "confirmed" if all(conds) else "corrected"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "neural_manifests"))
    a = ap.parse_args()
    root = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(root, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {root}")
    os.makedirs(a.out, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    below_normal()
    wait_if_slow()
    t0 = time.time()

    # Manifests of the glob.
    paths = sorted(p for p in glob.glob(os.path.join(root, "datasets", "_canonical_*_manifest.csv"))
                   if not os.path.basename(p).startswith("._"))
    man_rows, man, inputs = [], {}, {}
    for p in paths:
        d = read_manifest(p)
        name = os.path.basename(p)
        fam = name[len("_canonical_"):-len("_manifest.csv")]
        h = sha256(p)
        man_rows.append({"manifest": "datasets/" + name, "family": fam, "rows": len(d),
                         "sources": d.source.nunique() if "source" in d else "",
                         "bytes": os.path.getsize(p), "sha256": h})
        inputs["datasets/" + name] = {"bytes": os.path.getsize(p), "sha256": h}
        man[fam] = d
    write_csv(os.path.join(a.out, "manifests.csv"), man_rows)
    for fam in ["eeg", "emg", "imu", "force"]:
        if fam not in man:
            sys.exit(f"manifest of family {fam} missing")

    ps = per_source_tables(man)
    ps.to_csv(os.path.join(a.out, "per_source.csv"), index=False, encoding="utf-8", lineterminator="\n")

    def row(fam, src):
        r = ps[(ps.family == fam) & (ps.source == src)]
        if len(r) != 1:
            return None
        return r.iloc[0]

    eeg, emg, imu, force = man["eeg"], man["emg"], man["imu"], man["force"]
    for d in (eeg, emg, imu, force):
        d["sec"] = d.n_frames.astype(float) / d.fps.astype(float)

    emg_rows, emg_rec = emg_meta(root, emg)
    write_csv(os.path.join(a.out, "emg_sites.csv"), emg_rows)
    phys_ref = physionet_reference(root, eeg)
    ds4584_ref = ds004584_reference(root)

    facts = {"per_source": json.loads(ps.to_json(orient="records")), "physionet_reference": phys_ref,
             "ds004584_reference": ds4584_ref}
    C = Checks()

    # ---- EEG per source
    def eeg_check(cid, src, section, claim, ch, fps, recs, hours, ids, extra_measured="", ch_ok=None):
        r = row("eeg", src)
        chs = r.n_channels_values
        ok_ch = (chs == str(ch)) if ch_ok is None else ch_ok
        ok = [ok_ch, r.fps_values == f"{fps:g}", int(r.records) == recs,
              round(float(r.hours), 2) == hours, int(r.subject_ids) == ids]
        measured = (f"{chs} channels; {r.fps_values} Hz; {int(r.records)} records; "
                    f"{float(r.hours):.2f} h; {int(r.subject_ids)} subject ids" + extra_measured)
        C.add(cid, claim, section, measured, all_true(*ok))
        return r

    eeg_check("EEG-01", "sensefog", "addendum 1.1",
              "sensefog: scalp EEG 48 sites, 1000 Hz, 45 records, 2.73 h; 11 PD with raw EEG",
              48, 1000, 45, 2.73, 11)
    eeg_check("EEG-02", "multimodal_fog", "addendum 1.1",
              "multimodal_fog: scalp EEG 25 channels, 500 Hz, 56 records, 3.63 h; 12 PD",
              25, 500, 56, 3.63, 12)
    # physionet: the channel count is checked against the site list below (EEG-10).
    r = row("eeg", "physionet_multimodal_gait")
    ok = [r.fps_values == "300", int(r.records) == 176, round(float(r.hours), 2) == 2.93, int(r.subject_ids) == 59]
    C.add("EEG-03", "physionet_multimodal_gait: scalp EEG 300 Hz, 176 records, 2.93 h; 59 healthy adults with EEG",
          "addendum 1.1",
          f"{r.fps_values} Hz; {int(r.records)} records; {float(r.hours):.2f} h; {int(r.subject_ids)} subject ids",
          all_true(*ok))
    r = row("eeg", "seizeit2")
    sz_sites = eeg[eeg.source == "seizeit2"].sites.value_counts().to_dict()
    ok = [r.n_channels_values == "2", r.fps_values == "256", int(r.records) == 2850,
          round(float(r.hours), 2) == 11626.25, int(r.subject_ids) == 125]
    C.add("EEG-04", "seizeit2: behind-the-ear EEG, 2 bipolar channels, 256 Hz, 2,850 records, 11,626.25 h; 125 subjects",
          "addendum 1.1, 1.5",
          f"{r.n_channels_values} channels (site pairs: " + "; ".join(f"{k} {v}" for k, v in sz_sites.items())
          + f"); {r.fps_values} Hz; {int(r.records):,} records; {float(r.hours):,.2f} h; {int(r.subject_ids)} subject ids",
          all_true(*ok))
    r = row("eeg", "cap_sleep")
    chs = [int(x) for x in r.n_channels_values.split("|")]
    fpss = [float(x) for x in r.fps_values.split("|")]
    ok = [min(chs) == 2, max(chs) == 13, min(fpss) == 100, max(fpss) == 512,
          round(float(r.hours), 2) == 331.52, int(r.subject_ids) == 35]
    C.add("EEG-05", "cap_sleep: 35 harmonized ids; PSG EEG, 2 to 13 channels, 100 to 512 Hz, 331.52 h",
          "addendum 1.1, 1.5",
          f"{int(r.subject_ids)} subject ids; {r.n_channels_values} channels; {r.fps_values} Hz; "
          f"{int(r.records)} records; {float(r.hours):.2f} h", all_true(*ok))
    eeg_check("EEG-06", "pd_rest_eeg_onoff_ds002778", "addendum 1.2",
              "pd_rest_eeg_onoff_ds002778: 31 subjects, EEG 32 channels at 512 Hz, 46 records, 2.52 h",
              32, 512, 46, 2.52, 31)
    eeg_check("EEG-07", "pd_rest_eeg_ds004584", "addendum 1.2",
              "pd_rest_eeg_ds004584: 149 subjects, 63 channels at 500 Hz, 149 records, 6.64 h",
              63, 500, 149, 6.64, 149)
    eeg_check("EEG-08", "pd_walking_eeg_ds007526", "addendum 1.2",
              "pd_walking_eeg_ds007526: 144 subjects, 60 channels at 250 Hz, 277 records, 19.49 h",
              60, 250, 277, 19.49, 144)

    # ---- EEG totals
    tot_src = eeg.source.nunique()
    tot_rec = len(eeg)
    tot_ids = eeg.subject_ns.nunique()
    tot_h = eeg.sec.sum() / 3600.0
    C.add("EEG-09", "Harmonized EEG: 8 sources, 3,634 records, 566 subject ids, 11,995.7 h", "addendum 1.5, 9",
          f"{tot_src} sources, {tot_rec:,} records, {tot_ids} subject ids, {fnum(tot_h, 1)} h",
          all_true(tot_src == 8, tot_rec == 3634, tot_ids == 566, round(tot_h, 1) == 11995.7))
    facts["eeg_totals"] = {"sources": tot_src, "records": tot_rec, "subject_ids": tot_ids, "hours": tot_h}

    # ---- physionet site list against '19 channels, reference Pz'
    pg = eeg[eeg.source == "physionet_multimodal_gait"]
    site_lists = pg.sites.value_counts()
    sites = sorted({s for x in pg.sites for s in x.split("|") if s})
    scalp = [s for s in sites if s not in ("A1", "A2")]
    has_pz = "PZ" in sites
    n_ch = sorted({int(x) for x in pg.n_channels})
    meas = (f"{'|'.join(str(x) for x in n_ch)} channels in every record ({len(site_lists)} distinct site list over "
            f"{len(pg)} records): {len(scalp)} scalp sites ({' '.join(scalp)}) plus "
            f"{' and '.join(s for s in sites if s in ('A1', 'A2'))}; PZ {'is' if has_pz else 'is not'} a data column. "
            f"Record meta: every channel referenced to PZ in {phys_ref['records_all_channels_ref_pz']} of "
            f"{phys_ref['records']} records. Raw DSI-Streamer headers ('Reference location'): "
            + "; ".join(f"{k} in {v}" for k, v in phys_ref["raw_header_reference_counts"].items())
            + f" of {phys_ref['raw_csv_files']} files")
    ch_ok = n_ch == [19]
    ref_ok = (phys_ref["records_all_channels_ref_pz"] == phys_ref["records"]
              and set(phys_ref["raw_header_reference_counts"]) == {"Pz"})
    C.add("EEG-10", "physionet_multimodal_gait: scalp EEG 19 channels, reference Pz", "addendum 1.1, 2.2", meas,
          all_true(ch_ok, ref_ok))
    facts["physionet_eeg_sites"] = {"n_channels": n_ch, "sites": sites, "scalp_sites": scalp, "pz_is_column": has_pz,
                                    "distinct_site_lists": int(len(site_lists)), "records": int(len(pg)),
                                    "reference_ok": bool(ref_ok)}
    r = ds4584_ref
    C.add("EEG-11", "Declared reference Pz in pd_rest_eeg_ds004584", "addendum 2.2",
          f"EEGReference in {r['sidecars']} BIDS sidecars: " + "; ".join(f"{k} {v}" for k, v in r["EEGReference_counts"].items())
          + f"; PZ {'is' if 'PZ' in row('eeg', 'pd_rest_eeg_ds004584').sites_or_locations.split('|') else 'is not'} a manifest site",
          all_true(set(r["EEGReference_counts"]) == {"Pz"}))

    # ---- PD and healthy totals, synchronized sets
    pdh = eeg[~eeg.source.isin(NON_PD_HEALTHY_EEG)]
    pdh_h, pdh_ids = pdh.sec.sum() / 3600.0, pdh.subject_ns.nunique()
    C.add("EEG-12", "The PD and healthy EEG is 37.9 h over 406 subject ids", "addendum 1.5",
          f"{fnum(pdh_h, 2)} h over {pdh_ids} subject ids in the {pdh.source.nunique()} EEG sources other than "
          f"{' and '.join(NON_PD_HEALTHY_EEG)} ({', '.join(sorted(pdh.source.unique()))}); the manifest carries no "
          f"diagnosis, so the grouping is by source",
          all_true(round(pdh_h, 1) == 37.9, pdh_ids == 406))
    sy = eeg[eeg.source.isin(SYNC_SETS)]
    sy_h = sy.sec.sum() / 3600.0
    per = {s: eeg[eeg.source == s].sec.sum() / 3600.0 for s in SYNC_SETS}
    C.add("EEG-13", "The three sets declared synchronized hold 9.3 h of EEG; physionet_multimodal_gait 2.93 h",
          "addendum 1.5, 9",
          f"{fnum(sy_h, 2)} h: " + ", ".join(f"{s} {per[s]:.2f} h" for s in SYNC_SETS)
          + f"; without physionet_multimodal_gait (fails G15): {fnum(sy_h - per['physionet_multimodal_gait'], 2)} h",
          all_true(round(sy_h, 1) == 9.3, round(per["physionet_multimodal_gait"], 2) == 2.93))
    facts["pd_healthy_eeg"] = {"hours": pdh_h, "subject_ids": pdh_ids, "sources": sorted(pdh.source.unique())}
    facts["synchronized_eeg"] = {"hours": sy_h, "per_source_hours": per}

    # ---- record lengths
    ph = eeg[eeg.source == "physionet_multimodal_gait"].sec
    C.add("EEG-14", "physionet_multimodal_gait: every EEG record 59.87 s", "addendum 1.1, 9",
          f"{len(ph)} records; seconds min {ph.min():.3f}, max {ph.max():.3f}; n_frames values "
          + "|".join(sorted(set(eeg[eeg.source == 'physionet_multimodal_gait'].n_frames))),
          all_true(round(ph.min(), 2) == 59.87, round(ph.max(), 2) == 59.87))
    mf = eeg[eeg.source == "multimodal_fog"].sec
    p10, med = float(np.percentile(mf, 10)), float(np.median(mf))
    C.add("EEG-15", "multimodal_fog record seconds p10 25.5 and median 79.5", "addendum 9",
          f"{len(mf)} records; p10 {p10:.3f} s, median {med:.3f} s, min {mf.min():.3f} s, max {mf.max():.3f} s",
          all_true(round(p10, 1) == 25.5, round(med, 1) == 79.5))
    facts["record_seconds"] = {"physionet_min": ph.min(), "physionet_max": ph.max(),
                               "multimodal_fog_p10": p10, "multimodal_fog_median": med}

    # ---- EMG
    def emg_sites_of(src):
        return [x for x in emg_rows if x["source"] == src]

    r = row("emg", "sensefog")
    s = emg_sites_of("sensefog")
    muscles = sorted({(x["muscle"], x["side"]) for x in s if x["kind"] == "emg"})
    full = all(x["records_with_channel"] == x["records"] for x in s if x["kind"] == "emg")
    per_site = sorted({x["channels"] // max(x["records_with_channel"], 1) for x in s if x["kind"] == "emg"})
    C.add("EMG-01", "sensefog: leg EMG (TA, GA, bilateral) 1000 Hz, 45 records", "addendum 1.1, 2.2",
          f"{r.n_channels_values} EMG channels per record = {len(muscles)} muscle sites "
          f"({', '.join(m + ' ' + sd for m, sd in muscles)}) with {'|'.join(map(str, per_site))} electrodes each "
          f"per the record meta; "
          f"{r.fps_values} Hz; {int(r.records)} records; {int(r.subject_ids)} subject ids; "
          f"every site in every record: {'yes' if full else 'no'}",
          all_true(r.fps_values == "1000", int(r.records) == 45,
                   {m for m, _ in muscles} == {"tibialis_anterior", "gastrocnemius"},
                   {sd for _, sd in muscles} == {"L", "R"}))
    r = row("emg", "multimodal_fog")
    rec = emg_rec["multimodal_fog"]
    n_emg = sorted({x["emg"] for x in rec})
    n_tot = sorted({x["n"] for x in rec})
    other = {}
    for x in rec:
        for k, v in x["other"].items():
            other.setdefault(k, set()).add(v)
    s = emg_sites_of("multimodal_fog")
    musc = sorted({(x["muscle"], x["side"]) for x in s if x["kind"] == "emg"})
    C.add("EMG-02", "multimodal_fog: 5 EMG, ECG and EOG channels, 500 Hz", "addendum 1.1, 2.2",
          f"{'|'.join(map(str, n_tot))} channels per record, of which EMG {'|'.join(map(str, n_emg))} and "
          + ", ".join(f"{k} {'|'.join(map(str, sorted(v)))}" for k, v in sorted(other.items()))
          + f" (record meta); EMG muscle sites over the {len(rec)} records: "
          + ", ".join(f"{m} {sd} in {next(x['records_with_channel'] for x in s if x['muscle'] == m and x['side'] == sd and x['kind'] == 'emg')}"
                      for m, sd in musc)
          + f"; {r.fps_values} Hz",
          "qualified" if (n_tot == [5] and r.fps_values == "500") else "corrected")
    r = row("emg", "physionet_multimodal_gait")
    rec = emg_rec["physionet_multimodal_gait"]
    s = emg_sites_of("physionet_multimodal_gait")
    musc = sorted({(x["muscle"], x["side"]) for x in s if x["kind"] == "emg"})
    kinds = sorted({x["signal_kind"] for x in rec})
    C.add("EMG-03", "physionet_multimodal_gait: EMG as a 16.05 Hz envelope, 169 records, 58 subjects",
          "addendum 1.1, 2.2",
          f"{float(r.fps_values):.4f} Hz; signal_kind {'|'.join(kinds)} (record meta); {int(r.records)} records; "
          f"{int(r.subject_ids)} subject ids; {r.n_channels_values} channels = {len(musc)} muscle sites "
          f"({len({m for m, _ in musc})} muscles x {len({sd for _, sd in musc})} sides)",
          all_true(round(float(r.fps_values), 2) == 16.05, kinds == ["envelope"], int(r.records) == 169,
                   int(r.subject_ids) == 58))
    r = row("emg", "seizeit2_emg")
    C.add("EMG-04", "seizeit2: deltoid EMG 256 Hz, 2,804 records, 124 subjects", "addendum 1.1, 2.2",
          f"source seizeit2_emg: {r.n_channels_values} channel; {r.fps_values} Hz; {int(r.records):,} records; "
          f"{int(r.subject_ids)} subject ids (the muscle is not in the manifest)",
          all_true(r.fps_values == "256", int(r.records) == 2804, int(r.subject_ids) == 124))
    r = row("emg", "cap_sleep_chin")
    C.add("EMG-05", "cap_sleep chin EMG is canonical for 3 records only", "addendum 1.1, 1.6 item 8",
          f"source cap_sleep_chin: {int(r.records)} records, {int(r.subject_ids)} subject ids, {r.fps_values} Hz",
          all_true(int(r.records) == 3))
    rates = {s: row("emg", s).fps_values for s in ["sensefog", "multimodal_fog", "seizeit2_emg"]}
    C.add("EMG-06", "EMG drive band rates: sensefog 1000 Hz, multimodal_fog 500 Hz, seizeit2 256 Hz", "addendum 2.2",
          "; ".join(f"{k} {v} Hz" for k, v in rates.items()),
          all_true(rates["sensefog"] == "1000", rates["multimodal_fog"] == "500", rates["seizeit2_emg"] == "256"))
    r = row("emg", "gait120")
    C.add("EMG-07", "gait120: 120 healthy subjects, 2000 Hz EMG, 4,125 EMG records", "addendum 1.3",
          f"{int(r.subject_ids)} subject ids; {r.fps_values} Hz; {int(r.records):,} records; {r.n_channels_values} channels",
          all_true(int(r.subject_ids) == 120, r.fps_values == "2000", int(r.records) == 4125))
    r = row("emg", "ninapro")
    C.add("EMG-08", "ninapro: hand EMG, 201 records, 67 subjects", "addendum 1.4",
          f"{int(r.records)} records; {int(r.subject_ids)} subject ids; {r.fps_values} Hz; {r.n_channels_values} channels",
          all_true(int(r.records) == 201, int(r.subject_ids) == 67))
    r = row("emg", "cervical_motion")
    cm_rates = emg[emg.source == "cervical_motion"].fps.astype(float).round(1)
    rate_counts = cm_rates.value_counts()
    modal_rate, modal_n = float(rate_counts.index[0]), int(rate_counts.iloc[0])
    pmap = os.path.join(TREE, "reports", "person_ids", "map_cervical_motion_multiplanar.csv")
    pnote, amb = "", 0
    if os.path.isfile(pmap):
        pm = read_manifest(pmap)
        amb = int((pm.ambiguity != "").sum())
        pnote = (f"; reports/person_ids/map_cervical_motion_multiplanar.csv: {pm.person_id.nunique()} person ids, "
                 f"{amb} rows flagged as a possible same person")
    # 14 adults: 14 ids, or 15 ids of which a flagged pair may be one person (the map flags rows, two per pair).
    ids_ok = int(r.subject_ids) == 14 or (int(r.subject_ids) == 15 and amb >= 2)
    C.add("EMG-09", "cervical_motion_multiplanar: 14 asymptomatic adults, 127 records, 4.39 h; 2 EMG channels at 1785.7 Hz",
          "addendum 1.3",
          f"source cervical_motion: {int(r.subject_ids)} subject ids; {int(r.records)} records; {float(r.hours):.2f} h; "
          f"{r.n_channels_values} channels; {len(rate_counts)} distinct rates at 0.1 Hz, from {cm_rates.min():.1f} to "
          f"{cm_rates.max():.1f} Hz, {modal_n} of {len(cm_rates)} records at {modal_rate:.1f} Hz" + pnote,
          "qualified" if (int(r.records) == 127 and round(float(r.hours), 2) == 4.39 and r.n_channels_values == "2"
                          and abs(modal_rate - 1785.7) < 0.05 and ids_ok) else "corrected")
    facts["cervical_motion_rates_0p1hz"] = {f"{k:.1f}": int(v) for k, v in rate_counts.items()}
    r = row("emg", "drummer_tremor_model")
    C.add("EMG-10", "drummer_tremor_model: 18 healthy subjects, 2000 Hz EMG", "addendum 1.3",
          f"{int(r.subject_ids)} subject ids; {r.fps_values} Hz; {int(r.records)} records; {r.n_channels_values} channels",
          all_true(int(r.subject_ids) == 18, r.fps_values == "2000"))

    # ---- IMU
    def locs(src):
        return row("imu", src).sites_or_locations.split("|")

    r = row("imu", "sensefog")
    e_ids = set(eeg[eeg.source == "sensefog"].subject_ns)
    i_ids = set(imu[imu.source == "sensefog"].subject_ns)
    C.add("IMU-01", "sensefog: APDM IMU 128 Hz, 43 records, sites lower_back, ankle_L, ankle_R, no head sensor",
          "addendum 1.1, 1.5",
          f"{r.fps_values} Hz; {int(r.records)} records; {int(r.n_sites)} sites ({' '.join(locs('sensefog'))}) in "
          f"{int(r.site_combinations)} combination; head in {int(r.head_records)} records; {int(r.subject_ids)} IMU subject "
          f"ids against {len(e_ids)} EEG subject ids ({len(i_ids - e_ids)} with IMU and no EEG)",
          all_true(r.fps_values == "128", int(r.records) == 43,
                   set(locs("sensefog")) == {"lower_back", "ankle_L", "ankle_R"}, int(r.head_records) == 0))
    r = row("imu", "multimodal_fog")
    C.add("IMU-02", "multimodal_fog: IMU resampled to 500 Hz, sites lower_back, wrist_L, shank_L, shank_R in varying "
          "combinations, no head sensor", "addendum 1.1, 1.5",
          f"{r.fps_values} Hz; {int(r.records)} records; {int(r.n_sites)} sites ({' '.join(locs('multimodal_fog'))}) in "
          f"{int(r.site_combinations)} combinations; head in {int(r.head_records)} records; the 100 Hz native rate is "
          f"not in the manifest",
          all_true(r.fps_values == "500", set(locs("multimodal_fog")) == {"lower_back", "wrist_L", "shank_L", "shank_R"},
                   int(r.site_combinations) > 1, int(r.head_records) == 0))
    r = row("imu", "physionet_multimodal_gait")
    pe = set(eeg[eeg.source == "physionet_multimodal_gait"].subject_ns)
    pi = set(imu[imu.source == "physionet_multimodal_gait"].subject_ns)
    C.add("IMU-03", "physionet_multimodal_gait: Xsens 60 Hz at 9 sites including head in all 167 records; "
          "57 subjects with EEG and IMU", "addendum 1.1, 1.5, 9",
          f"{r.fps_values} Hz; {int(r.records)} records; {int(r.n_sites)} sites ({' '.join(locs('physionet_multimodal_gait'))}); "
          f"head in {int(r.head_records)} of {int(r.records)} records; {len(pe & pi)} subject ids in both the EEG and the "
          f"IMU manifest ({len(pe)} EEG, {len(pi)} IMU)",
          all_true(r.fps_values == "60", int(r.records) == 167, int(r.n_sites) == 9,
                   int(r.head_records) == 167, len(pe & pi) == 57))
    r = row("imu", "fog_turning_filho")
    C.add("IMU-04", "fog_turning_filho: 35 PD (IMU manifest), 128 Hz, 106 records; the IMU manifest says lower back",
          "addendum 1.3, 1.6 item 13",
          f"{int(r.subject_ids)} subject ids; {r.fps_values} Hz; {int(r.records)} records; locations "
          f"{' '.join(locs('fog_turning_filho'))}; head in {int(r.head_records)} records",
          all_true(int(r.subject_ids) == 35, r.fps_values == "128", int(r.records) == 106,
                   locs("fog_turning_filho") == ["lower_back"]))
    r = row("imu", "hd_accel_jnet")
    C.add("IMU-05", "hd_accel_jnet: wrist accelerometer 100 Hz, 35 records, 35 subject ids", "addendum 1.3",
          f"{r.fps_values} Hz; {int(r.records)} records; {int(r.subject_ids)} subject ids; locations "
          f"{' '.join(locs('hd_accel_jnet'))}",
          all_true(r.fps_values == "100", int(r.records) == 35, int(r.subject_ids) == 35,
                   all(x.startswith("wrist") for x in locs("hd_accel_jnet"))))
    imu_ps = ps[ps.family == "imu"]
    heads = {rr.source: (int(rr.head_records), int(rr.records)) for rr in imu_ps.itertuples() if int(rr.head_records) > 0}
    neural_heads = {s: heads.get(s, (0, int(row("imu", s).records))) for s in NEURAL_WALKING_IMU + ["fog_turning_filho"]}
    others = {s: v for s, v in heads.items() if s not in NEURAL_WALKING_IMU}
    ok_neural = (neural_heads["physionet_multimodal_gait"][0] == neural_heads["physionet_multimodal_gait"][1] == 167
                 and all(neural_heads[s][0] == 0 for s in ["sensefog", "multimodal_fog", "fog_turning_filho"]))
    C.add("IMU-06", "Head sensors: of the walking sets only physionet_multimodal_gait carries a head IMU (all 167 records); "
          "none in sensefog, multimodal_fog or fog_turning_filho", "addendum 1.1, 1.5, 9",
          "head records: " + "; ".join(f"{s} {v[0]} of {v[1]}" for s, v in neural_heads.items())
          + ". Other IMU sources with a head sensor, none with a neural signal in the manifests: "
          + ("; ".join(f"{s} {v[0]} of {v[1]}" for s, v in sorted(others.items())) or "none"),
          ("qualified" if others else "confirmed") if ok_neural else "corrected")
    facts["head_sensor_records"] = heads

    # ---- Force
    r = row("force", "physionet_multimodal_gait")
    C.add("FRC-01", "physionet_multimodal_gait: force plates 173 records", "addendum 1.1",
          f"{int(r.records)} records; {int(r.subject_ids)} subject ids; {r.fps_values} Hz",
          all_true(int(r.records) == 173))
    r = row("force", "gaitndd")
    C.add("FRC-02", "gaitndd: 64 subjects, foot-switch force 300 Hz", "addendum 1.3",
          f"{int(r.subject_ids)} subject ids; {int(r.records)} records; {r.fps_values} Hz",
          all_true(int(r.subject_ids) == 64, r.fps_values == "300"))

    # ---- Patients on one clock, manifest part
    s1 = int(row("eeg", "sensefog").subject_ids)
    s2 = int(row("eeg", "multimodal_fog").subject_ids)
    C.add("PAT-01", "Movement-disorder patients with brain and body on one clock: sensefog 11, multimodal_fog 12",
          "addendum 1.5",
          f"EEG subject ids: sensefog {s1}, multimodal_fog {s2}; ds004998, merk and oxford are raw and outside the "
          f"manifests (merk: reports/merk/; oxford: reports/oxford_lfp/)",
          all_true(s1 == 11, s2 == 12))

    write_csv(os.path.join(a.out, "addendum_checks.csv"), C.rows)
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1, default=float)

    prov = provenance(a.out, root, {
        "inputs": inputs,
        "emg_meta_records_read": {s: len(emg_rec[s]) for s in emg_rec},
        "physionet_eeg_meta_records_read": phys_ref["records"],
        "physionet_raw_csv_headers_read": phys_ref["raw_csv_files"],
        "ds004584_sidecars_read": ds4584_ref["sidecars"],
        "definitions": {"sync_sets": SYNC_SETS, "not_pd_or_healthy_eeg": NON_PD_HEALTHY_EEG,
                        "percentile": "numpy linear interpolation"},
        "seconds": round(time.time() - t0, 1)})
    write_summary(a.out, C.rows, ps, facts, prov)
    print(f"{len(C.rows)} checks:", pd.Series([r['verdict'] for r in C.rows]).value_counts().to_dict())


def write_summary(out, checks, ps, facts, prov):
    st = prov.get("moveatlas_stamp", {})
    head = st.get("head") if isinstance(st, dict) else None
    L = ["# Neural census from the canonical manifests", "",
         f"Generated by scripts/probe_neural_manifests.py on {prov['run_at'][:10]} against MoveAtlas "
         f"{head or '(see _provenance.json)'}; the stamp (HEAD, working-tree diff hash, untracked files) and the "
         f"SHA-256 of every manifest read are in _provenance.json. Every number below is in facts.json or a CSV of "
         f"this folder.", "",
         "## Verdicts", "",
         "| id | claim | source | measured | verdict |", "|---|---|---|---|---|"]
    for r in checks:
        L.append(f"| {r['check_id']} | {r['claim']} | {r['source']} | {r['measured']} | {r['verdict']} |")
    vc = pd.Series([r["verdict"] for r in checks]).value_counts().to_dict()
    L += ["", "Verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(vc.items())) + ". Confirmed: every number of the "
          "claim matches at the claim's precision. Corrected: at least one does not. Qualified: the numbers match and "
          "the manifests add a fact the claim leaves out.", ""]
    fx = facts["physionet_eeg_sites"]
    ears = [s for s in fx["sites"] if s not in fx["scalp_sites"]]
    L += ["## The physionet_multimodal_gait montage", "",
          f"The EEG manifest lists {'|'.join(map(str, fx['n_channels']))} channels for each of the {fx['records']} "
          f"records: {len(fx['scalp_sites'])} scalp sites and {' and '.join(ears)}. Pz is "
          f"{'a' if fx['pz_is_column'] else 'not a'} data column"
          + (": it is the hardware reference of every channel (record meta and raw headers, check EEG-10)"
             if fx["reference_ok"] else "; the reference check EEG-10 did not find Pz everywhere")
          + f". The {len(fx['scalp_sites'])} scalp data channels plus the Pz reference make "
          f"{len(fx['scalp_sites']) + (0 if fx['pz_is_column'] else 1)} scalp positions, the count of the claim "
          f"'19 channels'; the harmonized record holds {len(fx['scalp_sites'])} scalp channels and {len(ears)} ear "
          f"channels ({' and '.join(ears)}), {fx['n_channels'][0]} data channels in all"
          + (", every one against Pz." if fx["reference_ok"] else "."), "",
          "## Per-source tables", "",
          "per_source.csv: records, subject ids, hours, rates, channel counts, sites, head-sensor records and record "
          "seconds for every source of the EEG, EMG, IMU and force manifests. emg_sites.csv: EMG channel identity "
          "from the record meta for the three sets declared synchronized. manifests.csv: every file of the glob "
          "datasets/_canonical_*_manifest.csv with its SHA-256.", "",
          "## Limits", "",
          "- The manifests carry no diagnosis: the PD and healthy total groups sources, not persons.",
          "- Subject ids are manifest ids, not persons; person maps are in reports/person_ids/.",
          "- Raw-only sets (ds004998, oxford_dystonia_lfp, merk_neumann_ecog_lfp, the sensefog Percept streams) are "
          "outside the manifests; merk is in reports/merk/."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
