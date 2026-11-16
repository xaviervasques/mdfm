"""Person-identity audit for amass, infant_pose_chambers and sensefog (task T0.2b).

Patient-level splits need one id per PERSON. This script reads the MoveAtlas tree
READ-ONLY (pooled canonical manifests, per-source processed files, raw folders and the
source documentation) and writes one person map per source:
reports/person_ids/map_amass.csv, map_infant_pose_chambers.csv, map_sensefog.csv
(columns source, subject_ns, person_id, rule, evidence, ambiguity; one row per canonical
subject_ns found in any pooled manifest), plus audit tables, a summary and a provenance
file named _provenance_amass_infant_sensefog.json (the directory is shared with the
sibling T0.2 scripts, so the generic _provenance.json name is left to the combiner).

amass: the 24 raw archives (.tar.bz2) are streamed once, read-only, to collect the body
shape (betas), gender and body model (surface_model_type) stored in every motion file. The
per-archive results are cached under cache/person_ids_amass_raw/ of this tree (gitignored); a re-run
reuses a cache whose archive size, mtime and CACHE_VERSION still match. No betas are written to the versioned outputs, only
distances. The rule set is documented in the summary.

Run: <conda>/python.exe scripts/probe_person_ids_amass_infant_sensefog.py [--workers 6]
"""
import argparse
import collections
import datetime as dt
import glob
import hashlib
import io
import itertools
import json
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _common import moveatlas_head, parse_args, read_csv  # noqa: E402

TAG = "amass_infant_sensefog"
MAP_COLS = ["source", "subject_ns", "person_id", "rule", "evidence", "ambiguity"]


# ----------------------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------------------
def get_args():
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--workers", type=int, default=6, help="parallel readers of the amass archives")
    own.add_argument("--no-amass-raw", action="store_true",
                     help="do not stream the raw amass archives (use the cache only; missing archives are reported)")
    mine, rest = own.parse_known_args()
    if "--out" not in rest:
        rest += ["--out", os.path.join(TREE, "reports", "person_ids")]
    sys.argv = [sys.argv[0]] + rest
    a = parse_args(__doc__.splitlines()[0])
    a.workers = mine.workers
    a.no_amass_raw = mine.no_amass_raw
    return a


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def sha8(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:8]


def pooled_ids(ds_root, source):
    """subject_ns of one source across every pooled canonical manifest (and per-source extras)."""
    out = collections.defaultdict(set)
    for f in sorted(glob.glob(os.path.join(ds_root, "_canonical_*manifest*.csv"))):
        b = os.path.basename(f)
        if b.startswith("._") or not b.endswith(".csv"):
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8", usecols=lambda c: c in ("source", "subject_ns"))
        if "source" not in d.columns:
            continue
        for s in d.loc[d["source"] == source, "subject_ns"].unique():
            out[s].add(b)
    return out


# ----------------------------------------------------------------------------------------
# amass: streaming reader of one raw archive (runs in a worker process)
# ----------------------------------------------------------------------------------------
TEXT_EXT = (".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".html")
# 2: each motion record also keeps the value of surface_model_type (smplx or smplh) when the file has the key. Betas
# of SMPL-X and SMPL+H live in different shape spaces, so shape distances are only taken inside one body model.
CACHE_VERSION = 2


def _scalar_str(v):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return ""
    x = a[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def scan_archive(path, cache_path):
    st = os.stat(path)
    key = {"archive": os.path.basename(path), "size": st.st_size, "mtime": int(st.st_mtime), "version": CACHE_VERSION}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                c = json.load(f)
            if c.get("key") == key and c.get("complete"):
                return os.path.basename(path), "cached", c.get("seconds", 0.0)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    members, texts = [], {}
    with tarfile.open(path, "r|bz2") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            base = os.path.basename(name)
            if base.startswith("._"):
                members.append({"name": name, "size": m.size, "kind": "sidecar"})
                continue
            if name.endswith(".npz"):
                rec = {"name": name, "size": m.size, "kind": "npz"}
                try:
                    b = tf.extractfile(m).read()
                    with np.load(io.BytesIO(b), allow_pickle=False) as z:
                        keys = list(z.files)
                        rec["keys"] = keys
                        if "betas" in keys:
                            rec["betas"] = [float(x) for x in np.asarray(z["betas"], dtype=np.float64).ravel()]
                        if "gender" in keys:
                            rec["gender"] = _scalar_str(z["gender"])
                        if "surface_model_type" in keys:
                            rec["surface_model_type"] = _scalar_str(z["surface_model_type"])
                        for fk in ("mocap_framerate", "mocap_frame_rate"):
                            if fk in keys:
                                rec["fps"] = float(np.asarray(z[fk]).ravel()[0])
                        if "trans" in keys:
                            rec["n_frames"] = int(np.asarray(z["trans"]).shape[0])
                except Exception as e:  # noqa: BLE001
                    rec["error"] = repr(e)[:300]
                members.append(rec)
            else:
                members.append({"name": name, "size": m.size, "kind": "other"})
                low = base.lower()
                if m.size <= 262144 and (low.endswith(TEXT_EXT) or "license" in low or "readme" in low):
                    try:
                        texts[name] = tf.extractfile(m).read().decode("utf-8", "replace")
                    except Exception as e:  # noqa: BLE001
                        texts[name] = f"(unreadable: {e!r})"
    sec = round(time.time() - t0, 1)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"key": key, "complete": True, "seconds": sec, "members": members, "texts": texts}, f)
    os.replace(tmp, cache_path)
    return os.path.basename(path), "scanned", sec


def build_amass_cache(a, raw_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    tars = sorted((p for p in glob.glob(os.path.join(raw_dir, "*.tar.bz2")) if not os.path.basename(p).startswith("._")),
                  key=lambda p: -os.path.getsize(p))
    jobs = [(p, os.path.join(cache_dir, os.path.basename(p).replace(".tar.bz2", ".json"))) for p in tars]
    status = {}
    if a.no_amass_raw:
        for p, c in jobs:
            ver = None
            if os.path.isfile(c):
                with open(c, encoding="utf-8") as f:
                    ver = json.load(f).get("key", {}).get("version")
            status[os.path.basename(p)] = ("cached" if ver == CACHE_VERSION else
                                           f"stale cache version {ver} (not rescanned: --no-amass-raw)" if os.path.isfile(c)
                                           else "missing (not scanned: --no-amass-raw)")
        return status, len(tars)
    print(f"amass: {len(jobs)} archives, {sum(os.path.getsize(p) for p, _ in jobs) / 1e9:.1f} GB, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(scan_archive, p, c): p for p, c in jobs}
        for fu in as_completed(futs):
            p = futs[fu]
            try:
                arc, st, sec = fu.result()
                status[arc] = st
                print(f"  {arc}: {st} ({sec} s)", flush=True)
            except Exception as e:  # noqa: BLE001
                status[os.path.basename(p)] = f"error: {e!r}"[:300]
                print(f"  {os.path.basename(p)}: ERROR {e!r}", flush=True)
    return status, len(tars)


# ----------------------------------------------------------------------------------------
# amass: identity rules and audit
# ----------------------------------------------------------------------------------------
def amass_performer(archive, folder):
    """Performer key from the folder naming convention of the sub-dataset, or None.

    ACCAD   Female1General_c3d, Female1Walking_c3d, ... : the performer is the leading FemaleN or MaleN.
    CMU     18_19_Justin, 20_21_Justin1, 22_23_justin : two-actor sessions split by AMASS per actor; the
            actor is the trailing name without its digits, case-folded.
    DanceDB 20120805_VasoAristeidou : capture date then dancer name; the dancer is the name.
    HUMAN4D Subject1_2_Aude, Subject2_Aude : the actor is the trailing name.
    DFaust, MoSh, PosePrior (5-digit folder) and SSM (date_5-digit): MPI-IS subject numbers.
    """
    m = None
    if archive == "ACCAD":
        m = re.match(r"^(Female\d+|Male\d+)", folder)
        return ("ACCAD:" + m.group(1), "performer_name_in_folder") if m else (None, None)
    if archive == "CMU":
        m = re.fullmatch(r"\d+_\d+_([A-Za-z]+?)\d*", folder)
        return ("CMU:" + m.group(1).lower(), "performer_name_in_folder") if m else (None, None)
    if archive == "DanceDB":
        m = re.fullmatch(r"\d{8}_(.+)", folder)
        return ("DanceDB:" + m.group(1), "performer_name_in_folder") if m else (None, None)
    if archive == "HUMAN4D":
        m = re.fullmatch(r"Subject\d+(?:_\d+)?_(.+)", folder)
        return ("HUMAN4D:" + m.group(1), "performer_name_in_folder") if m else (None, None)
    if archive in ("DFaust", "MoSh", "PosePrior"):
        m = re.fullmatch(r"(\d{5})", folder)
    elif archive == "SSM":
        m = re.fullmatch(r"\d{8}_(\d{5})", folder)
    if m:
        return "MPI:" + m.group(1), "mpi_subject_number"
    return None, None


# sub-datasets whose every folder carries a distinct performer label inside one study (names, initials or
# participant numbers of one study): two different labels there are documented different persons. Named
# performers of ACCAD, CMU, DanceDB and HUMAN4D and the MPI numbers enter through their performer key; the
# unnamed ACCAD folders and the numbered CMU folders do not enter (their identity is the open question).
AMASS_DISTINCT_LABEL_ARCHIVES = ("EyesJapanDataset", "HDM05", "BMLmovi", "BMLhandball", "TotalCapture", "HumanEva", "GRAB")
# sub-datasets whose folders are bare numeric subject ids of a motion database (SFU included: its 4-digit folders are
# SFU Motion Capture Database subject numbers, with no performer list on disk, the same case as the KIT family).
AMASS_NUMERIC_DATABASE_ARCHIVES = ("KIT", "EKUT", "CNRS", "WEIZMANN", "SFU")


def amass_doc_gap(archive, folder, perf_key):
    """Documentation gap of the sub-dataset naming convention (id-level ambiguity code), or ''."""
    if archive == "ACCAD" and perf_key is None:
        return ("accad_unnamed_performer: the folder does not name its performer while named ACCAD performers "
                "(Female1, Male1, Male2) each span several folders; settled by the ACCAD motion capture lab session list")
    if archive == "CMU" and perf_key is None:
        return ("cmu_session_number: CMU subject numbers index capture sessions (the 18_19, 20_21 and 22_23 folders show "
                "two actors across three session pairs) and no actor list is on disk; settled by the CMU Graphics Lab "
                "capture log (actor per subject number)")
    if archive in AMASS_NUMERIC_DATABASE_ARCHIVES:
        return ("numeric_database_id: numeric subject id with no performer list on disk; settled by the subject records "
                "of the motion database the ids come from")
    if archive in ("GRAB", "SOMA"):
        return ("mpi_unnumbered: an MPI-IS capture (GRAB is hosted at grab.is.tue.mpg.de per the datacard; SOMA per its paper) "
                "without the MPI subject number that links DFaust and MoSh, so the performer may recur in another MPI archive; "
                "settled by the MPI-IS subject numbers of these captures")
    return ""


def audit_amass(a, facts, cache_dir, scan_status, n_tars):
    ds = os.path.join(a.moveatlas, "datasets")
    root = os.path.join(ds, "socle", "amass")
    man = read_csv(os.path.join(root, "processed", "canonical_manifest.csv"))
    pooled = pooled_ids(ds, "amass")
    man["archive"] = man["subject_ns"].str.split(":").str[1]
    parts = man["take"].str.split("/")
    man["top"] = parts.str[0]
    man["folder"] = parts.str[1]
    man["expected_ns"] = "amass:" + man["archive"] + ":" + man["top"] + "_" + man["folder"]
    rule_ok = int((man["subject_ns"] == man["expected_ns"]).sum())

    # raw archive caches
    members = {}
    tar_motions = collections.Counter()
    tar_motion_folders = set()
    shape_files = {}
    missing_cache = []
    for arc in sorted(man["archive"].unique()):
        p = os.path.join(cache_dir, f"{arc}.json")
        if not os.path.isfile(p):
            missing_cache.append(arc)
            continue
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
        for m in c["members"]:
            if m["kind"] != "npz":
                continue
            members[m["name"]] = m
            base = os.path.basename(m["name"])
            if "poses" in m.get("keys", []) and "trans" in m.get("keys", []):
                tar_motions[arc] += 1
                tar_motion_folders.add(os.path.dirname(m["name"]))
            elif base == "shape.npz" or ("poses" not in m.get("keys", [])):
                shape_files[os.path.dirname(m["name"])] = m
    man["read"] = man["take"].map(lambda t: t in members and "betas" in members[t])

    # one row per canonical id
    rows = []
    vec = {}
    for sid, g in man.groupby("subject_ns", sort=True):
        arc, top, folder = g["archive"].iloc[0], g["top"].iloc[0], g["folder"].iloc[0]
        recs = [members[t] for t in g["take"] if t in members and "betas" in members[t]]
        shapes = collections.Counter(tuple(np.round(r["betas"], 6)) for r in recs)
        genders = sorted({r.get("gender", "") for r in recs})
        # files without the key are AMASS SMPL+H G fits (dmpls); an old cache without the value reads 'unknown'
        models = sorted({r.get("surface_model_type", "unknown" if "surface_model_type" in r.get("keys", []) else "smplh")
                         for r in recs})
        if shapes:
            modal = np.array(shapes.most_common(1)[0][0])
            vec[sid] = modal
        sf = shape_files.get(f"{top}/{folder}")
        sf_same = (sf is not None and "betas" in sf and shapes and tuple(np.round(sf["betas"], 6)) == shapes.most_common(1)[0][0])
        key, rule = amass_performer(arc, folder)
        rows.append({"subject_ns": sid, "archive": arc, "top": top, "folder": folder, "n_motions_manifest": len(g),
                     "n_motions_read": len(recs), "n_shapes": len(shapes), "gender": "/".join(genders),
                     "surface_model": "/".join(models),
                     "shape_file": sf is not None, "shape_file_equals_motion_shape": bool(sf_same),
                     "performer_key": key or "", "performer_rule": rule or ""})
    ft = pd.DataFrame(rows)
    gender = dict(zip(ft["subject_ns"], ft["gender"]))
    smodel = dict(zip(ft["subject_ns"], ft["surface_model"]))

    # merges: performer keys (MPI numbers only when the gender agrees across the group) and identical shapes
    parent = {s: s for s in ft["subject_ns"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    key_groups = ft[ft["performer_key"] != ""].groupby("performer_key")["subject_ns"].apply(list).to_dict()
    mpi_checks = []
    for k, ids in key_groups.items():
        if len(ids) < 2:
            continue
        gs = {gender[i] for i in ids}
        if k.startswith("MPI:"):
            mpi_checks.append({"key": k, "ids": ids, "gender_agrees": len(gs) == 1, "gender": "/".join(sorted(gs))})
            if len(gs) != 1:
                continue
        for i in ids[1:]:
            union(ids[0], i)
    shape_groups = collections.defaultdict(list)
    for s, v in vec.items():
        shape_groups[tuple(v)].append(s)
    identical = [sorted(v) for v in shape_groups.values() if len(v) > 1]
    for ids in identical:
        for i in ids[1:]:
            union(ids[0], i)
    root_of = {s: find(s) for s in ft["subject_ns"]}
    members_of = collections.defaultdict(list)
    for s, r in root_of.items():
        members_of[r].append(s)

    # permutation probability that every MPI number shared by two archives agrees in gender by chance
    mpi_shared = [c for c in mpi_checks]
    n_sh = len(mpi_shared)
    n_fem = sum(1 for c in mpi_shared if c["gender"] == "female")
    from math import comb
    p_perm = (1 / comb(n_sh, n_fem)) if n_sh else None

    # shape distances: documented same performer (name rule) versus documented different performers
    sids = sorted(vec)
    X = np.stack([vec[s] for s in sids])
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    ix = {s: i for i, s in enumerate(sids)}
    pk = dict(zip(ft["subject_ns"], ft["performer_key"]))
    arcof = dict(zip(ft["subject_ns"], ft["archive"]))
    folderof = dict(zip(ft["subject_ns"], ft["folder"]))

    def label(s):
        if pk[s]:
            return pk[s]
        if arcof[s] in AMASS_DISTINCT_LABEL_ARCHIVES:
            return arcof[s] + ":" + folderof[s]
        return None

    same, null = [], []
    for x, y in itertools.combinations(sids, 2):
        if gender[x] != gender[y] or smodel[x] != smodel[y]:
            continue
        if pk[x] and pk[x] == pk[y] and not pk[x].startswith("MPI:"):
            same.append(D[ix[x], ix[y]])
        elif arcof[x] == arcof[y] and label(x) and label(y) and label(x) != label(y):
            null.append(D[ix[x], ix[y]])
    same, null = np.array(same), np.array(null)
    auc = float((same[:, None] < null[None, :]).mean() + 0.5 * (same[:, None] == null[None, :]).mean()) if len(same) and len(null) else None
    thr = float(null.min()) if len(null) else 0.0
    mpi_d = [float(D[ix[c["ids"][0]], ix[c["ids"][1]]]) for c in mpi_checks if len(c["ids"]) == 2]

    # nearest other person (same gender, same body model: SMPL-X and SMPL+H betas are not comparable) and mutual
    # nearest neighbours
    nn = {}
    for s in sids:
        best = None
        for t in sids:
            if t == s or gender[t] != gender[s] or smodel[t] != smodel[s] or root_of[t] == root_of[s]:
                continue
            d = D[ix[s], ix[t]]
            if best is None or d < best[0]:
                best = (d, t)
        nn[s] = best
    cand = []
    for s in sids:
        if nn[s] is None:
            continue
        d, t = nn[s]
        if nn.get(t) and nn[t][1] == s and s < t and d < thr:
            ns = [int(x) for x in (re.sub(r"\D", "", folderof[s]), re.sub(r"\D", "", folderof[t])) if x]
            consecutive = len(ns) == 2 and arcof[s] == arcof[t] and abs(ns[0] - ns[1]) == 1
            cand.append({"subject_ns_a": s, "subject_ns_b": t, "gender": gender[s], "shape_distance": round(float(d), 3),
                         "same_archive": arcof[s] == arcof[t], "adjacent_numbers": consecutive,
                         "reason": "mutual nearest body shapes closer than any two documented different performers"})
    # near-identical performer names inside one sub-dataset (spelling variants)
    import difflib
    names = sorted({k for k in pk.values() if k and not k.startswith("MPI:")})
    name_pairs = []
    for x, y in itertools.combinations(names, 2):
        if x.split(":")[0] == y.split(":")[0]:
            r = difflib.SequenceMatcher(None, x.split(":")[1].lower(), y.split(":")[1].lower()).ratio()
            if r >= 0.85:
                name_pairs.append((x, y, round(r, 3)))
    for x, y, r in name_pairs:
        for s in key_groups.get(x, []):
            for t in key_groups.get(y, []):
                cand.append({"subject_ns_a": s, "subject_ns_b": t, "gender": gender[s] + "|" + gender[t],
                             "shape_distance": round(float(D[ix[s], ix[t]]), 3) if s in ix and t in ix else "",
                             "same_archive": True, "adjacent_numbers": False, "reason": f"performer names similar (ratio {r})"})
    for c in mpi_checks:
        if not c["gender_agrees"]:
            for s, t in itertools.combinations(c["ids"], 2):
                cand.append({"subject_ns_a": s, "subject_ns_b": t, "gender": c["gender"], "shape_distance": "",
                             "same_archive": False, "adjacent_numbers": False, "reason": "same MPI number, gender differs"})
    cands = pd.DataFrame(cand, columns=["subject_ns_a", "subject_ns_b", "gender", "shape_distance", "same_archive",
                                        "adjacent_numbers", "reason"])
    partner = collections.defaultdict(list)
    for r in cands.itertuples():
        partner[r.subject_ns_a].append((r.subject_ns_b, r.shape_distance, r.reason))
        partner[r.subject_ns_b].append((r.subject_ns_a, r.shape_distance, r.reason))

    # person ids
    def person_key(rootid):
        ids = members_of[rootid]
        keys = sorted({pk[i] for i in ids if pk[i]})
        # a performer key names the person only when every id carrying it sits in this group (an MPI number whose
        # genders disagree is not merged, and its ids must not share one person_id through the key)
        if keys and set(key_groups.get(keys[0], [])) <= set(ids):
            return keys[0]
        s = sorted(ids)[0]
        return arcof[s] + ":" + folderof[s]

    out = []
    for r in ft.itertuples():
        s = r.subject_ns
        grp = sorted(members_of[root_of[s]])
        others = [o for o in grp if o != s]
        if others:
            rule = "mpi_subject_number" if pk[s].startswith("MPI:") else ("performer_name_in_folder" if pk[s] else "identical_body_shape")
        else:
            rule = "sub_dataset_folder"
        ev = [f"folder {r.top}/{r.folder}: {r.n_motions_manifest} manifest motions, {r.n_motions_read} read from raw/{r.archive}.tar.bz2, "
              f"{r.n_shapes} body shape (betas) and gender {r.gender} over all of them"]
        if others:
            ds_ = [float(D[ix[s], ix[o]]) for o in others if s in ix and o in ix]
            ev.append(f"same person as {', '.join(others)} (key {person_key(root_of[s])}); shape distances to them "
                      f"{min(ds_):.2f} to {max(ds_):.2f}, separately fitted")
        if nn.get(s):
            ev.append(f"nearest other person by shape (same gender, {smodel[s]} body model): {nn[s][1]} at {nn[s][0]:.2f}")
        amb = []
        gap = amass_doc_gap(r.archive, r.folder, pk[s] or None)
        if gap:
            amb.append(gap)
        for o, d, why in partner.get(s, []):
            amb.append(f"candidate same person as {o} ({why}{'' if d == '' else f', distance {d}'}); body shape cannot settle it")
        out.append({"source": "amass", "subject_ns": s, "person_id": "amass:person:" + person_key(root_of[s]),
                    "rule": rule, "evidence": "; ".join(ev), "ambiguity": " | ".join(amb)})
    mp = pd.DataFrame(out, columns=MAP_COLS)
    write_csv(mp, os.path.join(a.out, "map_amass.csv"))
    ft["person_id"] = ft["subject_ns"].map(dict(zip(mp["subject_ns"], mp["person_id"])))
    ft["nearest_other_person"] = ft["subject_ns"].map(lambda s: nn[s][1] if nn.get(s) else "")
    ft["nearest_shape_distance"] = ft["subject_ns"].map(lambda s: round(float(nn[s][0]), 3) if nn.get(s) else "")
    write_csv(ft, os.path.join(a.out, "amass_folders.csv"))
    write_csv(cands, os.path.join(a.out, "amass_candidate_pairs.csv"))

    per_arc = ft.groupby("archive").agg(ids=("subject_ns", "size"), persons=("person_id", "nunique"),
                                        motions=("n_motions_manifest", "sum"), read=("n_motions_read", "sum")).reset_index()
    per_arc["tar_motion_files"] = per_arc["archive"].map(lambda x: tar_motions.get(x, 0))
    write_csv(per_arc, os.path.join(a.out, "amass_archives.csv"))
    merged = {k: sorted(v) for k, v in ((person_key(r), members_of[r]) for r in members_of) if len(v) > 1}
    facts["amass"] = {
        "canonical_ids": int(ft["subject_ns"].nunique()), "pooled_ids": len(pooled),
        "pooled_equals_processed_manifest": set(pooled) == set(ft["subject_ns"]),
        "manifest_rows": int(len(man)), "rows_matching_rule_archive_top_folder": rule_ok,
        "archives": int(man["archive"].nunique()), "raw_archives": n_tars, "scan_status": scan_status,
        "archives_without_cache": missing_cache,
        "manifest_motions_read_with_betas": int(man["read"].sum()),
        "tar_motion_files": int(sum(tar_motions.values())),
        "tar_motion_files_not_in_manifest_by_archive": {k: int(tar_motions[k] - v) for k, v in man.groupby("archive").size().items()
                                                        if tar_motions[k] != v},
        "tar_performer_folders": len(tar_motion_folders),
        "tar_performer_folders_without_canonical_id": sorted(tar_motion_folders - set(man["top"] + "/" + man["folder"])),
        "ids_with_more_than_one_shape": int((ft["n_shapes"] > 1).sum()),
        "ids_with_more_than_one_gender": int(ft["gender"].str.contains("/").sum()),
        "ids_by_surface_model": {k: int(v) for k, v in ft["surface_model"].value_counts().sort_index().items()},
        "archives_by_surface_model": {k: sorted(g["archive"].unique()) for k, g in ft.groupby("surface_model")},
        "ids_with_more_than_one_surface_model": int(ft["surface_model"].str.contains("/").sum()),
        "ids_with_shape_file": int(ft["shape_file"].sum()),
        "shape_files_equal_to_motion_shape": int(ft["shape_file_equals_motion_shape"].sum()),
        "identical_shapes_across_ids": identical,
        "persons": int(mp["person_id"].nunique()),
        "merged_groups": merged, "ids_in_merged_groups": sum(len(v) for v in merged.values()),
        "mpi_numbers_shared_by_two_archives": n_sh, "mpi_shared_gender_agree": sum(1 for c in mpi_checks if c["gender_agrees"]),
        "mpi_shared_female": n_fem, "mpi_gender_agreement_chance_probability": p_perm,
        "mpi_pair_shape_distance_min": round(min(mpi_d), 3) if mpi_d else None,
        "mpi_pair_shape_distance_max": round(max(mpi_d), 3) if mpi_d else None,
        "shape_same_performer_pairs": int(len(same)),
        "shape_same_performer_distance_min": round(float(same.min()), 3) if len(same) else None,
        "shape_same_performer_distance_median": round(float(np.median(same)), 3) if len(same) else None,
        "shape_same_performer_distance_max": round(float(same.max()), 3) if len(same) else None,
        "shape_different_performer_pairs": int(len(null)),
        "shape_different_performer_distance_min": round(thr, 3),
        "shape_different_performer_distance_p1": round(float(np.percentile(null, 1)), 3) if len(null) else None,
        "shape_different_performer_distance_median": round(float(np.median(null)), 3) if len(null) else None,
        "shape_auc_same_closer_than_different": round(auc, 3) if auc is not None else None,
        "shape_median_norm_by_archive": {k: round(float(np.median([np.linalg.norm(vec[s]) for s in g["subject_ns"] if s in vec])), 2)
                                          for k, g in ft.groupby("archive")},
        "candidate_pairs": int(len(cands)),
        "candidate_pairs_by_reason": cands["reason"].str.replace(r" \(ratio.*", "", regex=True).value_counts().to_dict(),
        "candidate_pairs_same_archive": int(cands["same_archive"].astype(bool).sum()),
        "candidate_pairs_adjacent_numbers": int(cands["adjacent_numbers"].astype(bool).sum()),
        "similar_name_pairs": name_pairs,
        "ids_with_doc_gap": int(sum(1 for r in ft.itertuples() if amass_doc_gap(r.archive, r.folder, pk[r.subject_ns] or None))),
        "ids_with_doc_gap_by_code": dict(collections.Counter(amass_doc_gap(r.archive, r.folder, pk[r.subject_ns] or None).split(":")[0]
                                                             for r in ft.itertuples() if amass_doc_gap(r.archive, r.folder, pk[r.subject_ns] or None))),
        "ids_with_any_ambiguity": int((mp["ambiguity"] != "").sum()),
    }
    print(f"amass: {facts['amass']['canonical_ids']} ids -> {facts['amass']['persons']} persons", flush=True)
    return mp


# ----------------------------------------------------------------------------------------
# sensefog
# ----------------------------------------------------------------------------------------
def audit_sensefog(a, facts):
    ds = os.path.join(a.moveatlas, "datasets")
    root = os.path.join(ds, "eeg", "sensefog")
    raw = os.path.join(root, "raw", "SenseFOG-main")
    doc = os.path.join(root, "raw", "_doc_repo")
    ids = pooled_ids(ds, "sensefog")
    tasks = collections.defaultdict(lambda: collections.defaultdict(set))
    for fn, mod in (("_canonical_eeg_manifest.csv", "eeg"), ("_canonical_emg_manifest.csv", "emg"), ("_canonical_imu_manifest.csv", "imu")):
        d = read_csv(os.path.join(ds, fn))
        d = d[d["source"] == "sensefog"]
        for s, t in zip(d["subject_ns"], d["task"]):
            tasks[s][mod].add(t)
    subs = sorted(x for x in os.listdir(raw) if x.startswith("sub-") and os.path.isdir(os.path.join(raw, x)))
    sessions = {s: sorted(x for x in os.listdir(os.path.join(raw, s)) if x.startswith("ses-")) for s in subs}
    with open(os.path.join(doc, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    m = re.search(r"(\d+)\s+patients", readme)
    n_readme = int(m.group(1)) if m else None
    # the programmer clock of a Percept session can be wrong (dates decades before the device existed): a session date
    # counts for the device test only inside the README's years of subject and data recruitment
    my = re.search(r"recruitment:\s*(\d{4})\s*-\s*(\d{4})", readme)
    rec_years = (int(my.group(1)), int(my.group(2))) if my else None
    subj_files = {}
    for p in sorted(glob.glob(os.path.join(doc, "Matlab_scripts", "*Subject Files", "sub-*-datafile.m"))):
        if os.path.basename(p).startswith("._"):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        mm = re.search(r"subjectdata\.subjectnr\s*=\s*'([^']+)'", txt)
        subj_files[os.path.basename(p)] = mm.group(1) if mm else None

    # Percept JSON: device counter per session, identity fields summarized as counts only
    js = sorted(p.replace("\\", "/") for p in glob.glob(os.path.join(raw, "sub-*", "ses-*", "*", "*.json"))
                if not os.path.basename(p).startswith("._"))
    prow = []
    field_vals = collections.defaultdict(lambda: collections.defaultdict(set))  # field -> value hash -> set(sub)
    field_files = collections.defaultdict(lambda: collections.defaultdict(list))  # field -> value hash -> files
    field_shape = {}  # name value hash -> shape of the string (never the string itself)
    for p in js:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        sub = p.split("/")[-4]
        ses = p.split("/")[-3]
        fin = d.get("DeviceInformation", {}).get("Final", {})
        pin = d.get("PatientInformation", {}).get("Final", {})
        t = None
        try:
            t = dt.datetime.fromisoformat(str(d.get("SessionDate")).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            pass
        prow.append({"sub": sub, "session": ses, "time": t, "counter": fin.get("AccumulatedTherapyOnTimeSinceImplant")})
        # first and last name are pooled: a placeholder string can sit in either field
        for fld, v in (("PatientName", pin.get("PatientFirstName")), ("PatientName", pin.get("PatientLastName")),
                       ("PatientDateOfBirth", pin.get("PatientDateOfBirth")), ("PatientId", pin.get("PatientId")),
                       ("NeurostimulatorSerialNumber", fin.get("NeurostimulatorSerialNumber")), ("ImplantDate", fin.get("ImplantDate"))):
            if v not in (None, ""):
                h = sha8(v)
                field_vals[fld][h].add(sub)
                if f"{sub}/{ses}" not in field_files[fld][h]:
                    field_files[fld][h].append(f"{sub}/{ses}")
                if fld == "PatientName":
                    field_shape[h] = {"length": len(str(v)), "has_digit": bool(re.search(r"\d", str(v))),
                                      "looks_like_code": bool(re.fullmatch(r"(?i)[a-z]{0,8}[-_ ]?\d{1,4}", str(v).strip())),
                                      # generic placeholder words found inside the value (the word from this fixed list
                                      # is reported, never the value)
                                      "generic_words": sorted({w for w in ("name", "anonym", "patient", "test", "unknown",
                                                                           "pseudo", "dummy", "subject", "xxx")
                                                               if w in str(v).lower()})}
    def plausible(t):
        return t is not None and rec_years is not None and rec_years[0] <= t.year <= rec_years[1]

    by_sub = collections.defaultdict(list)
    for r in prow:
        by_sub[r["sub"]].append(r)
    per = {}
    for sub in sorted(by_sub):
        g = by_sub[sub]
        times = sorted(r["time"] for r in g if plausible(r["time"]))
        med_t = times[len(times) // 2] if times else None
        n_bad = sum(1 for r in g if not plausible(r["time"]))
        n_off = sum(1 for x in times if abs((x - med_t).total_seconds()) > 86400) if med_t is not None else 0
        c = [int(r["counter"]) for r in g if r["counter"] is not None]
        per[sub] = {"n_json": len(g), "counter_min": min(c), "counter_max": max(c), "counter_spread": max(c) - min(c),
                    "counter_median": int(np.median(c)), "median_time": med_t,
                    "sessions_dated_outside_recruitment_years": n_bad,
                    "plausible_sessions_dated_off_median_by_gt_1_day": n_off}
    pair_rows = []
    for x, y in itertools.combinations(sorted(per), 2):
        px, py = per[x], per[y]
        overlap = not (px["counter_max"] < py["counter_min"] or py["counter_max"] < px["counter_min"])
        row = {"sub_a": x, "sub_b": y, "date_testable": px["median_time"] is not None and py["median_time"] is not None,
               "earlier": "", "later": "", "days_apart": "", "counter_lower_at_later_date": "", "counter_ranges_overlap": overlap}
        if row["date_testable"]:
            early, late = (px, py) if px["median_time"] <= py["median_time"] else (py, px)
            row.update({"earlier": x if early is px else y, "later": y if early is px else x,
                        "days_apart": round((late["median_time"] - early["median_time"]).total_seconds() / 86400, 2),
                        "counter_lower_at_later_date": late["counter_median"] < early["counter_median"]})
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    n_pairs = len(pairs)
    n_date = sum(1 for r in pair_rows if r["date_testable"])

    # MoveAtlas processed LFP aux files: which identity-like metadata keys they carry, and whether any raw
    # identity value that is unique to one subject reappears in them (string search, values never printed)
    unique_hashes = {h for fld in field_vals for h, s in field_vals[fld].items() if len(s) == 1}
    unique_vals = []
    for p in js:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pin = d.get("PatientInformation", {}).get("Final", {})
        fin = d.get("DeviceInformation", {}).get("Final", {})
        for v in (pin.get("PatientFirstName"), pin.get("PatientLastName"), fin.get("NeurostimulatorSerialNumber"), fin.get("ImplantDate")):
            if v not in (None, "") and sha8(v) in unique_hashes:
                unique_vals.append(str(v))
    aux_keys, aux_dev, aux_hits, aux_files = set(), set(), 0, 0

    def _walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                _walk(v, pre + "/" + k)
        elif isinstance(o, list):
            for v in o:
                _walk(v, pre + "[]")
        else:
            aux_keys.add(pre)

    import zipfile
    for p in sorted(glob.glob(os.path.join(root, "processed", "aux", "*_lfp.npz"))):
        if os.path.basename(p).startswith("._"):
            continue
        with zipfile.ZipFile(p) as zf:
            if "meta.json" not in zf.namelist():
                continue
            txt = zf.read("meta.json").decode("utf-8", "replace")
        aux_files += 1
        m_ = json.loads(txt)
        _walk(m_)
        aux_dev.add(sha8(json.dumps(m_.get("meta", {}).get("device"), sort_keys=True)))
        aux_hits += sum(1 for v in set(unique_vals) if len(v) >= 4 and v in txt)
    ident_like = sorted(k for k in aux_keys if re.search(r"(?i)patient|serial|implant|name|birth|date|device", k))
    n_proven = sum(1 for r in pair_rows if r["counter_lower_at_later_date"] is True)
    n_overlap = sum(1 for r in pair_rows if r["counter_ranges_overlap"])
    write_csv(pd.DataFrame([{"sub": k, **{kk: vv for kk, vv in v.items() if kk != "median_time"}} for k, v in sorted(per.items())]),
              os.path.join(a.out, "sensefog_percept_sessions.csv"))
    write_csv(pairs, os.path.join(a.out, "sensefog_percept_pairs.csv"))

    field_summary = {}
    for fld in ("PatientName", "PatientDateOfBirth", "PatientId", "NeurostimulatorSerialNumber", "ImplantDate"):
        vals = field_vals.get(fld, {})
        shared = [h for h, s in vals.items() if len(s) > 1]
        single = {h: sorted(s) for h, s in vals.items() if len(s) == 1}
        field_summary[fld] = {
            "files_non_empty": len({f for h in vals for f in field_files[fld][h]}),
            "distinct_values": len(vals),
            "values_shared_by_several_subjects": len(shared),
            "values_unique_to_one_subject": len(single),
            "files_with_a_subject_unique_value": sorted({f for h in single for f in field_files[fld][h]}),
        }
        if fld == "PatientName":
            field_summary[fld]["subject_unique_values_shape"] = [
                {"subjects": single[h], **field_shape[h]} for h in sorted(single, key=lambda k: single[k])]

    rows = []
    for sid in sorted(ids):
        sub = sid.split(":", 1)[1]
        mods = {k: sorted(v) for k, v in tasks[sid].items()}
        sf = f"{sub}-datafile.m"
        ev = [f"raw folder {sub} holds sessions {','.join(sessions.get(sub, []))} (tasks of one visit, Med OFF and Stim OFF per the README)",
              f"manifests: " + "; ".join(f"{k} {len(v)} tasks" for k, v in sorted(mods.items())),
              f"subject file {sf} subjectnr={subj_files.get(sf)}"]
        if sub in per:
            ev.append(f"Percept JSON {per[sub]['n_json']} sessions, therapy-time counter spread {per[sub]['counter_spread']} (one device)")
            if per[sub]["sessions_dated_outside_recruitment_years"]:
                ev.append(f"{per[sub]['sessions_dated_outside_recruitment_years']} of its Percept sessions carry a date outside the "
                          f"README recruitment years {rec_years[0]}-{rec_years[1]} (programmer clock)")
        else:
            ev.append("no Percept JSON in the release (derived _lfpalg.mat only)")
        ev.append(f"README states {n_readme} patients for {len(subs)} sub folders")
        rows.append({"source": "sensefog", "subject_ns": sid, "person_id": f"sensefog:person:{sub}",
                     "rule": "sub_folder_is_patient", "evidence": "; ".join(ev), "ambiguity": ""})
    mp = pd.DataFrame(rows, columns=MAP_COLS)
    write_csv(mp, os.path.join(a.out, "map_sensefog.csv"))
    facts["sensefog"] = {
        "canonical_ids": len(ids), "canonical_ids_by_manifest": {k: sorted(v) for k, v in sorted(ids.items())},
        "persons": int(mp["person_id"].nunique()), "raw_sub_folders": len(subs),
        "sessions_per_sub": {k: len(v) for k, v in sessions.items()}, "readme_patients": n_readme,
        "subject_files": len(subj_files), "subject_file_numbers": sorted(v for v in subj_files.values() if v),
        "ids_without_raw_folder": sorted(s for s in ids if s.split(":", 1)[1] not in subs),
        "raw_folders_without_id": sorted(s for s in subs if f"sensefog:{s}" not in ids),
        "percept_json_files": len(js), "percept_subjects": len(per),
        "max_within_subject_counter_spread": max(v["counter_spread"] for v in per.values()),
        "min_between_subject_counter_gap": int(min(abs(per[x]["counter_median"] - per[y]["counter_median"]) for x, y in itertools.combinations(per, 2))),
        "subject_pairs_tested": n_pairs, "pairs_counter_ranges_overlap": n_overlap,
        "readme_recruitment_years": list(rec_years) if rec_years else None,
        "sessions_dated_outside_readme_recruitment_years": sum(v["sessions_dated_outside_recruitment_years"] for v in per.values()),
        "sessions_dated_outside_readme_recruitment_years_by_subject": {k: v["sessions_dated_outside_recruitment_years"]
                                                                      for k, v in per.items() if v["sessions_dated_outside_recruitment_years"]},
        "subjects_without_a_session_date_inside_recruitment_years": sorted(k for k, v in per.items() if v["median_time"] is None),
        "subject_pairs_date_testable": n_date, "pairs_counter_lower_at_later_date": n_proven,
        "plausible_sessions_dated_off_median_by_gt_1_day": sum(v["plausible_sessions_dated_off_median_by_gt_1_day"] for v in per.values()),
        "session_folder_names": sorted({x for v in sessions.values() for x in v}),
        "identity_fields": field_summary,
        "processed_aux_lfp_files": aux_files, "processed_aux_identity_like_keys": ident_like,
        "processed_aux_distinct_device_values": len(aux_dev),
        "processed_aux_raw_identity_value_occurrences": aux_hits,
    }
    print(f"sensefog: {len(ids)} ids -> {mp['person_id'].nunique()} persons", flush=True)
    return mp


# ----------------------------------------------------------------------------------------
# infant_pose_chambers
# ----------------------------------------------------------------------------------------
YT_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def audit_infant(a, facts):
    ds = os.path.join(a.moveatlas, "datasets")
    root = os.path.join(ds, "pose", "infant_pose_chambers")
    raw = os.path.join(root, "raw")
    ids = pooled_ids(ds, "infant_pose_chambers")
    audit = read_csv(os.path.join(root, "processed", "chambers_ingest_audit.csv"))
    yt = read_csv(os.path.join(raw, "meta_data_youtube_dataset.csv"))
    url = read_csv(os.path.join(raw, "URL_pose_dataset.csv"))
    lab = read_csv(os.path.join(raw, "URL_labelled_dataset.csv"))
    clin = read_csv(os.path.join(raw, "meta_data_clinical_dataset.csv"))
    with open(os.path.join(raw, "search_terms.txt"), encoding="utf-8") as f:
        terms = re.findall(r"'([^']*)'", f.read())
    twin_terms = [t for t in terms if "twin" in t.lower()]
    try:
        import subprocess
        sup = subprocess.run(["pdftotext", "-layout", os.path.join(raw, "Supplemental Table 1.pdf"), "-"],
                             capture_output=True, text=True, encoding="utf-8", timeout=120).stdout
        sup_ids = sorted({int(m.group(1)) for m in re.finditer(r"^\s*(\d{1,3})\s+[MF]\b", sup, flags=re.M)})
    except Exception:  # noqa: BLE001
        sup_ids = []

    url["yid"] = url["url"].map(lambda u: (YT_RE.search(u).group(1) if YT_RE.search(u) else u))
    lab["yid"] = lab["url"].map(lambda u: (YT_RE.search(u).group(1) if YT_RE.search(u) else u))
    url["video_number"] = url["video_number"].astype(int)
    yt["video_number"] = yt["video_number"].astype(int)
    yt["infant_id"] = yt["infant_id"].astype(int)
    lab_yids = set(lab["yid"])
    vn2yid = dict(zip(url["video_number"], url["yid"]))
    vn2note = dict(zip(url["video_number"], url["infant_notes"]))
    yid_clips = url.groupby("yid")["video_number"].apply(lambda s: sorted(int(x) for x in s)).to_dict()
    meta_vns = set(yt["video_number"])
    yt["yid"] = yt["video_number"].map(vn2yid)
    yid_meta_infants = yt.groupby("yid")["infant_id"].apply(lambda s: sorted(int(x) for x in s)).to_dict()

    aud = audit.copy()
    aud["kind"] = aud["subject_ns"].str.extract(r"infant_pose_chambers:([a-z]+)\d+$", expand=False)
    recs = aud.groupby("subject_ns").size().to_dict()
    take_by_id = aud.groupby("subject_ns")["take"].apply(list).to_dict()

    twin_ids = {}
    for yid, infs in yid_meta_infants.items():
        if len(infs) > 1:
            for i in infs:
                twin_ids[i] = (yid, infs)
    twin_pairs = sorted({tuple(v[1]) for v in twin_ids.values()})
    inf2vn = dict(zip(yt["infant_id"], yt["video_number"]))
    twin_lr = sum(1 for p in twin_pairs if len(p) == 2 and
                  sorted(vn2note.get(inf2vn[i], "") for i in p) == ["infant on left", "infant on right"])

    # where the registry figure 94 can come from: the datacard lines that carry it, and the video counts of the raw tables
    with open(os.path.join(root, "datacard.md"), encoding="utf-8") as f:
        card_94 = [ln.strip() for ln in f if re.search(r"\b94\b", ln)]
    pey = pd.read_csv(os.path.join(raw, "pose_estimates_youtube_dataset.csv"), usecols=["video"], dtype=str, encoding="utf-8")
    pec = pd.read_csv(os.path.join(raw, "pose_estimates_clinical_dataset.csv"), usecols=["video", "infant", "session"], dtype=str,
                      encoding="utf-8")

    rows = []
    checks = collections.Counter()
    for sid in sorted(ids, key=lambda s: (re.sub(r"\d+$", "", s), int(re.search(r"(\d+)$", s).group(1)))):
        kind = re.search(r":([a-z]+)\d+$", sid).group(1)
        num = int(re.search(r"(\d+)$", sid).group(1))
        if kind == "yt":
            r = yt[yt["infant_id"] == num]
            checks["yt_ids"] += 1
            if len(r) != 1:
                checks["yt_ids_without_single_meta_row"] += 1
            vn = int(r["video_number"].iloc[0]) if len(r) else None
            video = r["video"].iloc[0] if len(r) else ""
            takes = take_by_id.get(sid, [])
            if takes == [video]:
                checks["yt_take_matches_meta_video"] += 1
            yid = vn2yid.get(vn)
            clips = yid_clips.get(yid, [])
            infs = yid_meta_infants.get(yid, [])
            note = vn2note.get(vn, "")
            ev = [f"meta_data_youtube_dataset.csv: {video} (clip {vn}) infant_id {num}, age {r['age_in_weeks'].iloc[0] if len(r) else '?'} weeks",
                  f"URL group u{sha8(yid)} holds pose-dataset clips {clips}, {len(infs)} of them in the normative metadata"]
            if note:
                ev.append(f"clip note '{note}'")
            if yid in lab_yids:
                ev.append("its URL is also in URL_labelled_dataset.csv (pose-model training videos)")
            amb = ("channel_unverified: one infant_id per YouTube video and no uploader field in the release, so one infant filmed "
                   "at several ages under different URLs cannot be excluded; settled by the uploader of each URL in "
                   "URL_pose_dataset.csv or by the authors' infant list")
            if num in twin_ids:
                others = [f"infant_pose_chambers:yt{i}" for i in twin_ids[num][1] if i != num]
                pair_ids = [f"yt{p[0]}+yt{p[1]}" for p in twin_pairs if num not in p]
                ev.append(f"twin: {', '.join(others)} is the other infant of the same video (left and right), a different person")
                amb = (f"twin_series: this twin pair may be the same twins as another pair filmed at another age "
                       f"({', '.join(pair_ids)}); {len(twin_terms)} of the search terms ask for twins at given ages; "
                       f"settled by the uploader of each URL in URL_pose_dataset.csv")
            rows.append({"source": "infant_pose_chambers", "subject_ns": sid, "person_id": f"infant_pose_chambers:person:yt{num}",
                         "rule": "youtube_infant_id", "evidence": "; ".join(ev), "ambiguity": amb})
        elif kind == "cl":
            r = clin[clin["Infant"].astype(int) == num]
            checks["cl_ids"] += 1
            n_sess = r["Session"].nunique()
            ages = sorted(set(r["Chronological age (Months)"]))
            ev = [f"meta_data_clinical_dataset.csv: infant {num}, {len(r)} videos over {n_sess} sessions "
                  f"({len(ages)} distinct chronological ages); the ingest merged every session and trial under one id "
                  f"({recs.get(sid, 0)} canonical records)",
                  f"Supplemental Table 1 lists infant {num}: {'yes' if num in sup_ids else 'no'}"]
            if num in sup_ids:
                checks["cl_in_supplemental_table"] += 1
            rows.append({"source": "infant_pose_chambers", "subject_ns": sid, "person_id": f"infant_pose_chambers:person:cl{num}",
                         "rule": "clinical_infant_number", "evidence": "; ".join(ev), "ambiguity": ""})
        else:
            rows.append({"source": "infant_pose_chambers", "subject_ns": sid, "person_id": f"infant_pose_chambers:person:{sid.split(':', 1)[1]}",
                         "rule": "unknown_prefix", "evidence": "", "ambiguity": "prefix not documented"})
    mp = pd.DataFrame(rows, columns=MAP_COLS)
    write_csv(mp, os.path.join(a.out, "map_infant_pose_chambers.csv"))

    url_rows = int(len(url))
    shared_url_groups = {y: c for y, c in yid_clips.items() if len(c) > 1}
    facts["infant_pose_chambers"] = {
        "canonical_ids": len(ids), "canonical_ids_yt": checks["yt_ids"], "canonical_ids_cl": checks["cl_ids"],
        "persons": int(mp["person_id"].nunique()),
        "canonical_records": int(len(audit)),
        "yt_ids_with_exactly_one_meta_row": checks["yt_ids"] - checks["yt_ids_without_single_meta_row"],
        "yt_ids_whose_single_take_is_the_meta_video": checks["yt_take_matches_meta_video"],
        "meta_youtube_rows": int(len(yt)), "meta_youtube_distinct_infant_id": int(yt["infant_id"].nunique()),
        "meta_youtube_distinct_video": int(yt["video"].nunique()),
        "url_pose_rows": url_rows, "url_pose_max_row_index": int(url.iloc[:, 0].astype(int).max()),
        "url_pose_distinct_urls": int(url["yid"].nunique()),
        "url_pose_clips_not_in_meta": int(len(set(url["video_number"]) - meta_vns)),
        "url_pose_clips_not_in_meta_sharing_url_with_a_kept_infant": int(sum(
            1 for vn in set(url["video_number"]) - meta_vns if yid_meta_infants.get(vn2yid[vn]))),
        "url_groups_with_several_clips": len(shared_url_groups),
        "url_groups_with_several_meta_infants": sum(1 for v in yid_meta_infants.values() if len(v) > 1),
        "twin_pairs": [list(p) for p in twin_pairs],
        "twin_pairs_with_clip_notes_left_and_right": twin_lr,
        "datacard_lines_with_94": card_94,
        "pose_estimates_youtube_videos": int(pey["video"].nunique()),
        "pose_estimates_clinical_videos": int(pec["video"].nunique()),
        "pose_estimates_clinical_infants": int(pec["infant"].nunique()),
        "pose_estimates_clinical_infant_sessions": int(len(pec[["infant", "session"]].drop_duplicates())),
        "url_labelled_rows": int(len(lab)), "url_labelled_distinct_urls": int(lab["yid"].nunique()),
        "meta_infants_whose_url_is_in_labelled_set": int(yt["yid"].isin(lab_yids).sum()),
        "search_terms": len(terms), "search_terms_with_twin": len(twin_terms),
        "clinical_meta_rows": int(len(clin)), "clinical_infants": int(clin["Infant"].nunique()),
        "clinical_sessions_per_infant": {int(k): int(v) for k, v in clin.groupby(clin["Infant"].astype(int))["Session"].nunique().items()},
        "clinical_infants_with_several_sessions": int((clin.groupby("Infant")["Session"].nunique() > 1).sum()),
        "supplemental_table_infants": len(sup_ids),
        "cl_ids_in_supplemental_table": checks["cl_in_supplemental_table"],
    }
    print(f"infant_pose_chambers: {len(ids)} ids -> {mp['person_id'].nunique()} persons", flush=True)
    return mp


# ----------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------
def write_summary(a, facts, maps, head):
    fa, fi, fs = facts["amass"], facts["infant_pose_chambers"], facts["sensefog"]
    cands = read_csv(os.path.join(a.out, "amass_candidate_pairs.csv"))
    L = ["# Person ids of amass, infant_pose_chambers and sensefog (task T0.2b)", "",
         f"Measured by scripts/probe_person_ids_amass_infant_sensefog.py against MoveAtlas commit {' '.join(head.split(' ')[:2])}. "
         "Every number below is copied from the script's outputs in this directory "
         f"(_provenance_{TAG}.json holds the full fact list).", "",
         "| source | canonical subject ids | persons | main rule | ids with an ambiguity flag |", "|---|---|---|---|---|"]
    for src, mp in maps.items():
        rule = mp["rule"].value_counts().idxmax()
        L.append(f"| {src} | {mp['subject_ns'].nunique()} | {mp['person_id'].nunique()} | {rule} | {int((mp['ambiguity'] != '').sum())} |")
    L += ["", "Maps: map_amass.csv, map_infant_pose_chambers.csv, map_sensefog.csv (columns source, subject_ns, person_id, rule, "
          "evidence, ambiguity; one row per canonical subject_ns found in any pooled manifest). A flagged pair stays two "
          "persons in person_id; the flag says what document would merge them.", ""]

    # amass
    mg = fa["merged_groups"]
    L += ["## amass", "",
          f"How the ids were formed: subject_ns = amass:<archive>:<top folder>_<performer folder>, from the take path "
          f"<top folder>/<performer folder>/<motion>.npz; {fa['rows_matching_rule_archive_top_folder']} of {fa['manifest_rows']} "
          f"manifest rows follow it, over {fa['archives']} archives. The pooled manifest holds the same ids as "
          f"processed/canonical_manifest.csv: {fa['pooled_equals_processed_manifest']}.", "",
          f"Raw check: the {fa['raw_archives']} archives were streamed read-only; {fa['manifest_motions_read_with_betas']} of "
          f"{fa['manifest_rows']} manifest motions were found and read with their body shape (betas) and gender. Ids whose "
          f"motions carry more than one shape: {fa['ids_with_more_than_one_shape']}; more than one gender: "
          f"{fa['ids_with_more_than_one_gender']}. So each id is exactly one AMASS shape fit (one performer as AMASS processed it). "
          f"Identical shapes shared by two ids: {len(fa['identical_shapes_across_ids'])}: every folder was fitted separately, "
          "including folders of the same named performer.", "",
          f"The registry's ~344 is the datacard's aggregate figure, marked there as to be confirmed per sub-dataset; the "
          f"{fa['canonical_ids']} ids are the performer folders of the {fa['archives']} sub-datasets on disk "
          f"({fa['tar_performer_folders']} folders with motions in the archives, "
          f"{len(fa['tar_performer_folders_without_canonical_id'])} of them without a canonical id; motion files left out of "
          "the manifest: " + ", ".join(f"{k} {v}" for k, v in sorted(fa["tar_motion_files_not_in_manifest_by_archive"].items()))
          + "). After the merges below they are "
          f"{fa['persons']} persons (amass_archives.csv counts persons per archive; the MPI persons count in both DFaust and MoSh).", "",
          f"Merges ({len(mg)} groups, {fa['ids_in_merged_groups']} ids):", ""]
    for k, v in sorted(mg.items()):
        L.append(f"- amass:person:{k}: {', '.join(v)}")
    L += ["",
          "Rules: performer_name_in_folder (ACCAD FemaleN or MaleN prefix; CMU two-actor folders 18_19, 20_21, 22_23 split "
          "per actor name; DanceDB date_dancer; HUMAN4D SubjectN_actor), mpi_subject_number (the 5-digit MPI-IS subject "
          "number shared by DFaust_67 and MPI_mosh folders), sub_dataset_folder (every other folder is its own person; "
          f"the numeric folders of {', '.join(AMASS_NUMERIC_DATABASE_ARCHIVES)} carry the numeric_database_id flag).", "",
          f"MPI numbers: {fa['mpi_numbers_shared_by_two_archives']} numbers occur in two archives, gender agrees for "
          f"{fa['mpi_shared_gender_agree']} of them ({fa['mpi_shared_female']} female); a chance pairing agreeing on all has "
          f"probability {fa['mpi_gender_agreement_chance_probability']:.4f}. Their shape distances run "
          f"{fa['mpi_pair_shape_distance_min']} to {fa['mpi_pair_shape_distance_max']}.", "",
          f"Body shape cannot settle identity: {fa['shape_same_performer_pairs']} pairs of separately fitted folders of one "
          f"named performer lie at distances {fa['shape_same_performer_distance_min']} to "
          f"{fa['shape_same_performer_distance_max']} (median {fa['shape_same_performer_distance_median']}), while "
          f"{fa['shape_different_performer_pairs']} pairs of documented different performers of the same sub-dataset "
          f"and gender start at {fa['shape_different_performer_distance_min']} (1st percentile "
          f"{fa['shape_different_performer_distance_p1']}, median {fa['shape_different_performer_distance_median']}); "
          f"the probability that a same-performer pair is closer than a different-performer pair is "
          f"{fa['shape_auc_same_closer_than_different']}. CMU shapes sit near the mean body (median shape norm "
          f"{fa['shape_median_norm_by_archive'].get('CMU')} against {fa['shape_median_norm_by_archive'].get('BMLmovi')} for BMLmovi), "
          "so CMU folders are close to each other whoever performed them.", "",
          "Body models: SMPL-X and SMPL+H betas live in different shape spaces, so every shape distance and nearest "
          "neighbour is taken inside one body model (surface_model_type of the motion files; files without the key are "
          "AMASS SMPL+H fits): ids per model " + ", ".join(f"{k} {v}" for k, v in fa["ids_by_surface_model"].items())
          + " (" + "; ".join(f"{k}: {', '.join(v)}" for k, v in fa["archives_by_surface_model"].items() if k != "smplh")
          + f"); ids whose motion files mix models: {fa['ids_with_more_than_one_surface_model']}.", "",
          f"Shape candidates: {fa['candidate_pairs']} pairs are mutual nearest shapes closer than any two documented different "
          f"performers ({fa['candidate_pairs_same_archive']} inside one archive, {fa['candidate_pairs_adjacent_numbers']} "
          "with adjacent subject numbers). They are flagged, not merged:", "",
          "| id a | id b | gender | distance | adjacent numbers |", "|---|---|---|---|---|"]
    for r in cands.itertuples():
        L.append(f"| {r.subject_ns_a} | {r.subject_ns_b} | {r.gender} | {r.shape_distance} | {r.adjacent_numbers} |")
    L += ["", f"Similar performer names inside one sub-dataset (possible spelling variants): {len(fa['similar_name_pairs'])}.", "",
          f"Documentation gaps (id-level flags, {fa['ids_with_doc_gap']} ids): " +
          ", ".join(f"{k} {v}" for k, v in sorted(fa["ids_with_doc_gap_by_code"].items())) + ". The text of each flag in "
          "map_amass.csv names the document that would settle it.", ""]

    # infant
    L += ["## infant_pose_chambers", "",
          f"How the ids were formed (processed/chambers_ingest_audit.csv): yt<k> is infant_id k of "
          f"raw/meta_data_youtube_dataset.csv ({fi['canonical_ids_yt']} ids, one video each: "
          f"{fi['yt_ids_whose_single_take_is_the_meta_video']} ids hold exactly the metadata video); cl<k> is the CHOP infant "
          f"number of raw/meta_data_clinical_dataset.csv ({fi['canonical_ids_cl']} ids; the ingest already merged every "
          f"session and trial of an infant: {fi['clinical_infants_with_several_sessions']} infants have more than one session). "
          f"All {fi['cl_ids_in_supplemental_table']} clinical infants appear in Supplemental Table 1 "
          f"({fi['supplemental_table_infants']} infants listed).", "",
          f"Registry '94 + 85 meta': the 85 'meta' are the {fi['meta_youtube_rows']} rows of the YouTube metadata, i.e. the yt ids "
          f"themselves, not further persons. The 94 is written in the datacard ("
          + " / ".join(f"'{x.lstrip('- ')}'" for x in fi["datacard_lines_with_94"]) + "), and no raw table counts 94: "
          f"pose_estimates_youtube_dataset.csv holds {fi['pose_estimates_youtube_videos']} videos, "
          f"pose_estimates_clinical_dataset.csv {fi['pose_estimates_clinical_videos']} videos of "
          f"{fi['pose_estimates_clinical_infants']} infants ({fi['pose_estimates_clinical_infant_sessions']} infant sessions), "
          f"URL_labelled_dataset.csv {fi['url_labelled_rows']} rows, and URL_pose_dataset.csv {fi['url_pose_rows']} clips "
          f"(row index 0 to {fi['url_pose_max_row_index']}) from {fi['url_pose_distinct_urls']} distinct URLs; "
          f"{fi['url_pose_clips_not_in_meta']} of these clips are not in the metadata, "
          f"{fi['url_pose_clips_not_in_meta_sharing_url_with_a_kept_infant']} of them further clips of a URL whose infant is kept "
          "(the authors kept one clip per infant) and the rest from URLs left out. The figshare record text is the document "
          "that would explain the 94.", "",
          f"Person rule: one person per id ({fi['persons']} persons). Twins: {len(fi['twin_pairs'])} URLs show two infants "
          f"({fi['twin_pairs_with_clip_notes_left_and_right']} of these pairs carry the clip notes 'infant on left' and "
          "'infant on right'), kept as two persons each: "
          + ", ".join(f"yt{p[0]} and yt{p[1]}" for p in fi["twin_pairs"]) + ".", "",
          f"Unresolved: a family channel can post the same infant at several ages, and the release has no uploader field. "
          f"All {fi['canonical_ids_yt']} yt ids carry the flag channel_unverified; the twin ids also carry twin_series "
          f"({fi['search_terms_with_twin']} of {fi['search_terms']} search terms ask for twins at given ages). Settled by the "
          "uploader of each URL in URL_pose_dataset.csv (YouTube metadata) or the authors' infant list.", "",
          f"Note for evaluation, not for splits: {fi['meta_infants_whose_url_is_in_labelled_set']} of the "
          f"{fi['meta_youtube_rows']} normative infants have their URL in URL_labelled_dataset.csv, the videos used to "
          "train the domain-adapted pose model that produced the pose estimates.", ""]

    # sensefog
    idf = fs["identity_fields"]
    L += ["## sensefog", "",
          f"How the ids were formed: subject_ns = sensefog:<raw sub folder>; {fs['canonical_ids']} ids across the eeg, emg "
          f"and imu manifests (sensefog:sub-10 only in imu: its EEG exists only as derived _eegalg.mat). Raw folders: "
          f"{fs['raw_sub_folders']}; subject files of the documentation repo: {fs['subject_files']}; README: "
          f"{fs['readme_patients']} patients. Sessions ({', '.join(fs['session_folder_names'])}) are tasks "
          "of one visit inside a sub folder and never became separate ids.", "",
          f"Device check (Percept JSON, {fs['percept_json_files']} files, {fs['percept_subjects']} subjects): the implant's "
          f"therapy-time counter (DeviceInformation AccumulatedTherapyOnTimeSinceImplant) spreads at most {fs['max_within_subject_counter_spread']} units inside a subject and the "
          f"closest two subjects differ by {fs['min_between_subject_counter_gap']}; no two subjects' counter ranges overlap "
          f"({fs['pairs_counter_ranges_overlap']} of {fs['subject_pairs_tested']} pairs overlap). Session dates count only "
          f"inside the README's recruitment years {fs['readme_recruitment_years'][0]}-{fs['readme_recruitment_years'][1]}: "
          f"{fs['sessions_dated_outside_readme_recruitment_years']} sessions carry a date outside them ("
          + ", ".join(f"{k} {v}" for k, v in sorted(fs["sessions_dated_outside_readme_recruitment_years_by_subject"].items()))
          + ") while their counter stays continuous with the subject's other sessions, a programmer clock fault and not a "
          f"second visit; subjects left with no usable date: {', '.join(fs['subjects_without_a_session_date_inside_recruitment_years']) or 'none'}. "
          f"Of the {fs['subject_pairs_date_testable']} pairs where both subjects have a usable date, "
          f"{fs['pairs_counter_lower_at_later_date']} show the lower counter at the later date, which a single device cannot "
          "do, so those pairs are two devices; the other pairs are consistent with two devices without proving it. Usable "
          f"session dates more than a day away from their subject's median date: {fs['plausible_sessions_dated_off_median_by_gt_1_day']}.",
          "", f"Persons: {fs['persons']} (rule sub_folder_is_patient). No ambiguity left.", "",
          "Privacy note for the owner (values never printed): in the raw Percept JSON of the public Mendeley deposit the "
          "identity fields are blank or share one placeholder value across subjects, except: name fields with "
          f"{idf['PatientName']['values_unique_to_one_subject']} value(s) unique to one subject ("
          + "; ".join(f"{', '.join(x['subjects'])}: {x['length']} characters, digits {x['has_digit']}, code-like {x['looks_like_code']}, "
                      f"generic placeholder words inside: {', '.join(x['generic_words']) or 'none'}"
                      for x in idf["PatientName"].get("subject_unique_values_shape", []))
          + f"; files {', '.join(idf['PatientName']['files_with_a_subject_unique_value'])}), NeurostimulatorSerialNumber with "
          f"{idf['NeurostimulatorSerialNumber']['values_unique_to_one_subject']} (files "
          f"{', '.join(idf['NeurostimulatorSerialNumber']['files_with_a_subject_unique_value'])}) and ImplantDate with "
          f"{idf['ImplantDate']['values_unique_to_one_subject']} (files {', '.join(idf['ImplantDate']['files_with_a_subject_unique_value'])}). "
          "Whether the name values are pseudonyms or real names is for the owner to look at. The MoveAtlas processed LFP files "
          f"({fs['processed_aux_lfp_files']} meta.json) carry the identity-like keys {', '.join(fs['processed_aux_identity_like_keys'])}; "
          f"the device key holds {fs['processed_aux_distinct_device_values']} distinct value across all of them, and the "
          f"subject-unique raw values occur {fs['processed_aux_raw_identity_value_occurrences']} times in them. Session dates "
          "are kept there, which matters for a derived release.", ""]

    L += ["## Unresolved ambiguities and the document that settles each", "",
          "- amass, ACCAD folders without a performer name: the ACCAD motion capture lab session list.",
          "- amass, CMU numbered folders: the CMU Graphics Lab capture log (actor per subject number).",
          f"- amass, {', '.join(AMASS_NUMERIC_DATABASE_ARCHIVES)} numeric ids: the subject records of the motion database "
          "the ids come from.",
          "- amass, GRAB and SOMA: the MPI-IS subject numbers of these captures (to link them with DFaust, MoSh, PosePrior, SSM).",
          "- amass, shape candidate pairs (table above): the same documents; body shape alone cannot decide.",
          "- infant_pose_chambers, every YouTube id and the twin pairs in particular: the uploader of each URL.",
          "- infant_pose_chambers, the registry figure 94: the figshare record text.",
          "- sensefog: none.", ""]
    with open(os.path.join(a.out, f"summary_{TAG}.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    A = get_args()
    T0 = time.time()
    FACTS = {}
    AMASS_ROOT = os.path.join(A.moveatlas, "datasets", "socle", "amass")
    # derived AMASS data (betas) never enters reports/: the cache lives in the gitignored cache/
    CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "person_ids_amass_raw")
    STATUS, NTARS = build_amass_cache(A, os.path.join(AMASS_ROOT, "raw"), CACHE)
    MAPS = {"amass": audit_amass(A, FACTS, CACHE, STATUS, NTARS),
            "infant_pose_chambers": audit_infant(A, FACTS),
            "sensefog": audit_sensefog(A, FACTS)}
    HEAD = moveatlas_head(A.moveatlas)
    write_summary(A, FACTS, MAPS, HEAD)
    prov = {"script": os.path.basename(sys.argv[0]), "task": "T0.2b", "moveatlas_root": A.moveatlas, "moveatlas_head": HEAD,
            "run_at": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "seconds": round(time.time() - T0, 1),
            "outputs": sorted(f for f in os.listdir(A.out) if f.endswith((".csv", ".md")) and
                              (f.startswith(("map_amass", "map_infant", "map_sensefog", "amass_", "sensefog_")) or TAG in f)),
            "facts": FACTS}
    with open(os.path.join(A.out, f"_provenance_{TAG}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1, default=str)
    print("provenance:", HEAD)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))} for k, v in FACTS.items()}, indent=1, default=str))
