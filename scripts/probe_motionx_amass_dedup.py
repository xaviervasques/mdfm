"""Deduplicate motion_x against amass before any hold-out (plan task T0.7c).

Motion-X is known to include motion derived from other mocap collections: in its source
index the humanml subset is HumanML3D rebuilt from AMASS, and its GRAB subset is the GRAB
capture that AMASS also ships. This probe looks for motion_x records of the MoveAtlas
canonical pose corpus (datasets/_canonical_manifest.csv and the processed shards) that
duplicate an amass record, in three tiers.

1. metadata: the subset of every canonical motion_x record against the Motion-X source index
   (raw/Motion-X-main/datasets/all.txt, mocap-dataset-process/humanml_index.csv) and against
   the subsets of the Hugging Face release on disk; clip names against the Motion-X mocap
   clip names and against the amass take names.
2. content hash: every frame, root-centred as in the canonical files, is turned about the
   vertical so that its hip axis lies on +x (per-frame heading removed), divided by the
   record's limb scale (median of the mean length of the 8 rigid limb bones; the canonical
   torso scale is a median over the take, so a crop of a take would not keep it), rounded to
   HASH_Q limb units and hashed. A motion_x clip is an exact duplicate when at
   least EXACT_FRAC of its frames hit one amass record at one start frame and at the stride
   set by the two frame rates (amass native rate over 30 Hz). Mirrored clips are tested too.
3. near-duplicates: a 10 Hz signature of the same heading-free pose, windows of 1 s (0.4 s
   for short clips) reduced by PCA, an exact nearest-neighbour index of every amass window
   (GPU), plus an exhaustive scan of every alignment of the whole clip on the 10 Hz amass
   signatures (GPU; the window index alone misses the nearest alignment of many clips), then
   for each clip the RMS joint distance over the aligned span at 30 Hz, refined over
   sub-frame start offsets, against the best candidates. A clip is flagged when this
   distance is at most the NULL_PCT percentile of a null (amass crops searched against amass
   with their own subject removed) and the aligned span covers MIN_COVER of the clip.
   Synthetic duplicates (amass crops resampled to 30 Hz, bone lengths perturbed, jittered,
   mirrored, turned) measure the recall at that threshold.

AMASS side: the canonical records flagged as non-rigid by
reports/probe_amass_canonical_rigidity/records.csv are NOT excluded: their joints are
recomputed from the raw archives with the MD_FM package (mdfm.data.amass.load_many, one
sequential pass per archive, sequences cached under cache/amass/; mdfm.body.smplh forward
kinematics), with the MoveAtlas build recipe (first 66 pose values, first 16 betas, z-up to
y-up, SMPL 22 to canonical 15, root-centred, divided by the median torso length). Records
whose pelvis-to-neck variation exceeds TORSO_CHECK are recomputed too, to test that
criterion (version 1 of the rigidity probe flagged on it). The same pass reads the frame rate
of the SMPL-X stageii subsets, which the amass manifest leaves empty, and recomputes a few
rigid records to check that the recipe reproduces the canonical joints. The feature and hash
caches carry a fingerprint of the recomputed record set and are rebuilt when it changes.

Reads MoveAtlas read-only. Cache (resumable) under cache/motionx_amass_dedup/ of this tree
(gitignored). Writes reports/nuisance/motionx_amass_duplicates.csv, motionx_amass_nearest.csv,
motionx_amass_controls.csv, motionx_amass_recomputed.csv, motionx_amass_summary.md and
_provenance_motionx_amass_dedup.json.

Run: <conda>/envs/mdfm/python.exe scripts/probe_motionx_amass_dedup.py
"""
import collections
import csv
import glob
import json
import os
import re
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
for _p in (HERE, TREE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 20260928
# archive subset name -> first folder of the take (original AMASS names; the Motion-X
# mocap README uses the same table)
AMASS_TAKE_ROOT = {"BMLrub": "BioMotionLab_NTroje", "DFaust": "DFaust_67",
                   "EyesJapanDataset": "Eyes_Japan_Dataset", "HDM05": "MPI_HDM05", "MoSh": "MPI_mosh",
                   "PosePrior": "MPI_Limits", "SSM": "SSM_synced", "TCDHands": "TCD_handMocap",
                   "Transitions": "Transitions_mocap"}
# Motion-X source-index subsets built from mocap collections that AMASS also holds
MX_AMASS_DERIVED = ("humanml", "GRAB")
MX_MOCAP = ("humanml", "GRAB", "EgoBody")
SEL22 = [0, 12, 15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]      # SMPL 22 -> canonical 15
MIRROR = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]         # canonical left <-> right
MIRROR14 = [j - 1 for j in MIRROR[1:]]
PARENT = [-1, 0, 1, 1, 1, 3, 4, 5, 6, 0, 0, 9, 10, 11, 12]
BONE_GROUP = [-1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]         # torso, head, shoulder, upper arm, forearm, hip, thigh, shank
LIMB = [(3, 5), (4, 6), (5, 7), (6, 8), (9, 11), (10, 12), (11, 13), (12, 14)]   # rigid limb bones (canonical indices)
HASH_Q = 0.01              # rounding step of the content hash, torso units
EXACT_FRAC = 0.9           # fraction of frames that must hit one record at one alignment
MAXHIT = 64                # amass frames kept per query frame key
SIG_RATE = 10.0            # Hz of the near-duplicate signature
MX_FPS = 30.0
WIN = {"long": 10, "short": 4}          # window length in signature frames
QSTRIDE = {"long": 5, "short": 2}       # query window stride in signature frames
NPCA = {"long": 48, "short": 32}
PCA_SAMPLE = (200000, 100000)           # amass and motion_x windows used to fit the PCA
K_NN = 16
M_CAND = 32
N_REFINE = 3
REFINE_HALF = 0.15         # s, sub-frame start search half-width
REFINE_STEP = 1.0 / 120.0  # s
MIN_COVER = 0.8
NULL_PCT = 1.0
N_NULL = 2000
N_POS = 1000               # per perturbation level
N_HASH_POS = 500
PERTURB = {"light": (0.03, 0.005), "heavy": (0.10, 0.02)}   # bone-length half-range, jitter sd (torso units)
KNN_CHUNK = 512
EX_COLS = 32               # query frames per block of the exhaustive scan
TORSO_CHECK = 0.05         # pelvis-to-neck variation above which a record is recomputed as a check
POOL = 6


# ----------------------------------------------------------------------------- geometry

def heading_free(c):
    """(T, 15, 3) root-centred y-up -> (T, 42): each frame turned about y so that the hip axis
    (left_hip - right_hip) lies on +x; the pelvis (origin) is dropped. Falls back to the
    shoulder axis when the hip axis is vertical."""
    c = np.asarray(c, np.float64)
    h = c[:, 9] - c[:, 10]
    s = c[:, 3] - c[:, 4]
    hx, hz = h[:, 0], h[:, 2]
    bad = hx * hx + hz * hz < 1e-8
    hx = np.where(bad, s[:, 0], hx)
    hz = np.where(bad, s[:, 2], hz)
    phi = np.arctan2(hz, hx)
    cp, sp = np.cos(phi)[:, None], np.sin(phi)[:, None]
    x, y, z = c[:, 1:, 0], c[:, 1:, 1], c[:, 1:, 2]
    out = np.stack([x * cp + z * sp, y, -x * sp + z * cp], -1)
    return out.reshape(c.shape[0], 42).astype(np.float32)


def mirror_feat(f):
    """Mirror heading-free features (N, 42): swap left and right, negate x."""
    g = f.reshape(-1, 14, 3)[:, MIRROR14].copy()
    g[..., 0] *= -1
    return g.reshape(-1, 42)


def to15(f):
    """(T, 42) heading-free -> (T, 15, 3) with the pelvis at the origin."""
    p = np.zeros((f.shape[0], 15, 3), np.float64)
    p[:, 1:] = f.reshape(-1, 14, 3)
    return p


def normalise15(p):
    p = p - p[:, :1]
    torso = np.linalg.norm(p[:, 1] - p[:, 0], axis=-1)
    good = torso[np.isfinite(torso) & (torso > 1e-6)]
    return p / (float(np.median(good)) if good.size else 1.0)


def interp_rows(X, u):
    """Linear interpolation of rows of X (T, D) at fractional indices u (any shape)."""
    T = X.shape[0]
    uc = np.clip(u, 0, T - 1)
    i0 = np.minimum(np.floor(uc).astype(np.int64), max(T - 2, 0))
    i1 = np.minimum(i0 + 1, T - 1)
    w = (uc - i0)[..., None].astype(np.float32)
    return X[i0] * (1 - w) + X[i1] * w


def signature10(nat, fps):
    """Native heading-free features (T, 42) at fps -> 10 Hz signature by linear interpolation."""
    T = nat.shape[0]
    if T < 2:
        return nat[:1].copy()
    n = int(np.floor((T - 1) / fps * SIG_RATE + 1e-6)) + 1
    return interp_rows(nat, np.arange(n) / SIG_RATE * fps).astype(np.float32)


# ----------------------------------------------------------------------------- hashing

def row_hash(f):
    """Rows of heading-free features -> uint64 hash of the values rounded to HASH_Q."""
    q = np.rint(f / HASH_Q).astype(np.int64)
    h = np.full(q.shape[0], np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for j in range(q.shape[1]):
        h ^= q[:, j].astype(np.uint64)
        h *= prime
    h ^= h >> np.uint64(29)
    return h


def limb_scales(X, start, length):
    """Per record: median over frames of the mean length of the 8 limb bones (upper arm,
    forearm, thigh, shank, both sides). These bones are rigid in SMPL forward kinematics,
    unlike the pelvis-to-neck torso of the canonical normalisation (the spine bends), so the
    scale of a crop equals the scale of the whole take."""
    out = np.ones(len(start))
    for i, (s, n) in enumerate(zip(start, length)):
        if n < 1:
            continue
        p = X[s:s + n].reshape(-1, 14, 3)
        b = np.mean([np.linalg.norm(p[:, a - 1] - p[:, c - 1], axis=-1) for a, c in LIMB], axis=0)
        b = b[np.isfinite(b) & (b > 1e-6)]
        if b.size:
            out[i] = float(np.median(b))
    return out


def hash_frames(X, start, length, scale, chunk=1 << 20):
    """Hash keys of every frame, each record divided by its limb scale first."""
    if not len(X):
        return np.zeros(0, np.uint64)
    sc = np.repeat(scale, length).astype(np.float32)
    return np.concatenate([row_hash(X[s:s + chunk] / sc[s:s + chunk, None]) for s in range(0, X.shape[0], chunk)])


def hash_query(Xq, q_start, q_len, H, rec_fps):
    """Exact-content search. Xq (N, 42) query frames at 30 Hz, concatenated per query.
    Returns arrays per query: best fraction of frames matched at one alignment, record,
    mirror flag, amass start frame, and the number of frame hits looked at."""
    import pandas as pd
    nq = len(q_start)
    qscale = limb_scales(Xq, q_start, q_len)
    parts = []
    for mir in (0, 1):
        keys = hash_frames(Xq if mir == 0 else mirror_feat(Xq), q_start, q_len, qscale)
        lo = np.searchsorted(H["keys"], keys, "left")
        hi = np.searchsorted(H["keys"], keys, "right")
        cnt = np.minimum(hi - lo, MAXHIT)
        qi = np.nonzero(cnt)[0]
        if not len(qi):
            continue
        c = cnt[qi]
        rep = np.repeat(qi, c)
        within = np.arange(len(rep)) - np.repeat(np.cumsum(c) - c, c)
        pos = lo[rep] + within
        parts.append(pd.DataFrame({"g": rep, "mir": mir, "rec": H["rec"][pos], "fr": H["frame"][pos]}))
    best = np.zeros(nq)
    best_rec = np.full(nq, -1, np.int64)
    best_mir = np.zeros(nq, np.int64)
    best_off = np.zeros(nq, np.int64)
    if not parts:
        return best, best_rec, best_mir, best_off, 0
    df = pd.concat(parts, ignore_index=True)
    n_hits = len(df)
    df["q"] = np.searchsorted(q_start, df["g"].to_numpy(), "right") - 1
    df["t"] = df["g"].to_numpy() - q_start[df["q"].to_numpy()]
    s = rec_fps[df["rec"].to_numpy()] / MX_FPS
    df["off"] = np.rint(df["fr"].to_numpy() - s * df["t"].to_numpy()).astype(np.int64)
    df = df[["q", "mir", "rec", "off", "t"]]
    # a start frame c collects the query frames that hit at c - 1, c or c + 1 (rounding of a
    # non-integer stride); each query frame counts once per start frame
    ex = pd.concat([df.assign(off=df["off"] + d) for d in (-1, 0, 1)], ignore_index=True)
    tot = ex.groupby(["q", "mir", "rec", "off"])["t"].nunique().rename("score").reset_index()
    tot = tot.sort_values(["q", "score"], ascending=[True, False], kind="stable").drop_duplicates("q")
    qv = tot["q"].to_numpy()
    best[qv] = np.minimum(tot["score"].to_numpy() / q_len[qv], 1.0)
    best_rec[qv] = tot["rec"].to_numpy()
    best_mir[qv] = tot["mir"].to_numpy()
    best_off[qv] = tot["off"].to_numpy()
    return best, best_rec, best_mir, best_off, n_hits


# ----------------------------------------------------------------------------- raw archives

def load_worker(job):
    """One sequential pass over one AMASS archive with mdfm.data.amass.load_many (it caches
    every requested sequence under cache/amass/); returns the frame rate of each member."""
    subset, members = job
    from mdfm.data import amass as mdfm_amass
    t0 = time.time()
    seqs = mdfm_amass.load_many(subset, members)
    fps = {m: float(s["fps"]) for m, s in seqs.items()}
    print(f"  raw {subset}: {len(fps)} sequences, {time.time() - t0:.0f} s", flush=True)
    return subset, fps


# ----------------------------------------------------------------------------- tables

def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tables(ma):
    pooled = read_rows(f"{ma}/datasets/_canonical_manifest.csv")
    cols = ["source", "subject_ns", "take", "dim", "fps", "n_frames", "modality"]
    checks = {}
    tabs = {}
    for src, proc in (("amass", "socle/amass"), ("motion_x", "socle/motion_x")):
        pr = [r for r in pooled if r["source"] == src]
        pp = read_rows(f"{ma}/datasets/{proc}/processed/canonical_manifest.csv")
        checks[src] = {"pooled_rows": len(pr), "processed_rows": len(pp),
                       "identical": len(pr) == len(pp) and all([a[c] for c in cols] == [b[c] for c in cols] for a, b in zip(pr, pp))}
        tabs[src] = pr
    root2sub = {v: k for k, v in AMASS_TAKE_ROOT.items()}
    amass = []
    per = collections.defaultdict(int)
    for r in tabs["amass"]:
        sub = root2sub.get(r["take"].split("/")[0], r["take"].split("/")[0])
        i = per[sub]
        per[sub] += 1
        amass.append({"subset": sub, "record": f"r{i:06d}", "take": r["take"], "subject_ns": r["subject_ns"],
                      "fps": float(r["fps"]) if r["fps"] else float("nan"), "n_frames": int(r["n_frames"])})
    mx = []
    per = collections.defaultdict(int)
    for r in tabs["motion_x"]:
        sub = r["subject_ns"].split(":")[1]
        i = per[sub]
        per[sub] += 1
        mx.append({"subset": sub, "record": f"r{i:06d}", "clip": r["take"], "subject_ns": r["subject_ns"],
                   "fps": float(r["fps"]), "n_frames": int(r["n_frames"])})
    return amass, mx, checks


def attach_rigidity(amass, path):
    rig = {(r["subset"], r["record"]): r for r in read_rows(path)}
    mism = 0
    for a in amass:
        r = rig.get((a["subset"], a["record"]))
        a["corrupted"] = bool(r) and r["corrupted"] == "True"
        a["rig_thigh_cv"] = float(r["thigh_cv"]) if r else float("nan")
        a["rig_torso_cv"] = float(r["torso_cv"]) if r else float("nan")
        # recomputed: the flagged records, plus those whose pelvis-to-neck variation exceeds
        # TORSO_CHECK (the criterion of version 1 of the rigidity probe), to test that criterion
        a["recompute"] = a["corrupted"] or a["rig_torso_cv"] > TORSO_CHECK
        if r is None or int(r["frames"]) != a["n_frames"]:
            mism += 1
    return mism


# ----------------------------------------------------------------------------- metadata tier

def norm_name(s):
    s = os.path.basename(s)
    for suf in ("_poses.npz", "_stageii.npz", ".npz", ".npy"):
        if s.endswith(suf):
            s = s[:-len(suf)]
            break
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def take_key(t):
    """Take file name without its folders and without a leading subject number (CMU '105_36' -> '36')."""
    return re.sub(r"^\d+_", "", os.path.basename(t))


def loose_name(s):
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"_?clip_?\d+$", "", s.lower()))


def metadata_tier(ma, amass, mx):
    raw = f"{ma}/datasets/socle/motion_x/raw"
    out = {}
    out["canonical_subsets"] = dict(collections.Counter(r["subset"] for r in mx))
    tree = glob.glob(f"{raw}/motion_x_plusplus/.cache/huggingface/trees/*.json")
    tree = [t for t in tree if not os.path.basename(t).startswith("._")]
    hf = set()
    if tree:
        with open(tree[0], encoding="utf-8") as f:
            files = json.load(f).get("files", {})
        hf = {k.rsplit("/", 1)[-1][:-4] for k in files if k.endswith(".zip")}
    out["hf_release_subsets"] = sorted(hf)
    with open(f"{raw}/Motion-X-main/datasets/all.txt", encoding="utf-8") as f:
        idx = [ln.strip() for ln in f if ln.strip()]
    out["index_subsets"] = dict(collections.Counter(x.split("/")[0] for x in idx))
    base_by_sub = collections.defaultdict(set)
    for x in idx:
        base_by_sub[x.split("/")[0]].add(x.rsplit("/", 1)[-1])
    # humanml_index: HumanML3D source paths, the AMASS takes behind Motion-X's humanml subset
    hrows = read_rows(f"{raw}/Motion-X-main/mocap-dataset-process/humanml_index.csv")
    takes = {a["take"] for a in amass}
    hsrc = collections.Counter()
    hin = set()
    for r in hrows:
        sp = r["source_path"].replace("./pose_data/", "")
        root = sp.split("/")[0]
        hsrc[root] += 1
        t = sp[:-len("_poses.npy")] + "_poses.npz" if sp.endswith("_poses.npy") else sp
        if t in takes:
            hin.add(t)
    out["humanml_index_rows"] = len(hrows)
    out["humanml_index_by_source"] = dict(hsrc)
    out["humanml_index_amass_takes_in_moveatlas"] = len(hin)
    amass_grab = [a for a in amass if a["subset"] == "GRAB"]
    grab_keys = {a["take"][len("GRAB/"):-len("_stageii.npz")] for a in amass_grab if a["take"].endswith("_stageii.npz")}
    grab_idx = {x.split("/", 1)[1] for x in idx if x.startswith("GRAB/")}
    out["index_grab_clips_with_amass_take"] = len(grab_idx & grab_keys)
    rows = []
    # subset tier: canonical motion_x records whose subset is an AMASS-derived Motion-X subset
    derived = {s.lower() for s in MX_AMASS_DERIVED}
    sub_hits = [r for r in mx if r["subset"].lower() in derived]
    for r in sub_hits:
        rows.append({"mx": r, "amass": None, "method": "metadata_subset", "distance": ""})
    out["metadata_subset_records"] = len(sub_hits)
    # name tier 1: canonical clip ids that appear as clip names of the Motion-X mocap subsets
    mocap_names = {}
    for s in MX_MOCAP:
        for b in base_by_sub.get(s, set()):
            mocap_names.setdefault(b, s)
    name_hits_idx = [r for r in mx if r["clip"] in mocap_names]
    out["clip_ids_in_index"] = {s: sum(1 for r in mx if r["clip"] in base_by_sub.get(s, set()))
                                for s in sorted(base_by_sub)}
    out["metadata_name_index_records"] = len(name_hits_idx)
    # loose names: lower-case alphanumerics, clip suffix dropped; which index subsets carry them
    loose = collections.defaultdict(set)
    for x in idx:
        loose[loose_name(x.rsplit("/", 1)[-1])].add(x.split("/")[0])
    lc = collections.Counter()
    for r in mx:
        for s in loose.get(loose_name(r["clip"]), ()):
            lc[(r["subset"], s)] += 1
    out["loose_name_matches"] = [{"canonical_subset": a, "index_subset": b, "records": n} for (a, b), n in sorted(lc.items())]
    out["loose_name_matches_mocap_subsets"] = sum(n for (a, b), n in lc.items() if b in MX_MOCAP)
    out["loose_name_records_matched"] = sum(1 for r in mx if loose_name(r["clip"]) in loose)
    # name tier 2: loose clip id (Motion-X "_clipN" suffix dropped) equal to a loose amass take
    # name (file suffix dropped); a strict comparison can never match, every clip id ends in _clipN
    by_norm = collections.defaultdict(list)
    for i, a in enumerate(amass):
        k = loose_name(norm_name(a["take"]))
        if k:
            by_norm[k].append(i)
    name_pairs = []
    for j, r in enumerate(mx):
        for i in by_norm.get(loose_name(r["clip"]), []):
            name_pairs.append((j, i))
    out["metadata_name_take_pairs"] = len(name_pairs)
    out["metadata_name_take_records"] = len({j for j, _ in name_pairs})
    return out, rows, name_hits_idx, name_pairs


# ----------------------------------------------------------------------------- features

def recomputed_fingerprint(recomputed):
    """Identity of the recomputed record set, stamped on the caches built from it."""
    import hashlib
    return hashlib.sha1(",".join(sorted(recomputed)).encode("utf-8")).hexdigest()


def load_amass_features(ma, amass, recomputed, cache):
    fpath, mpath = os.path.join(cache, "amass_feat.npy"), os.path.join(cache, "amass_feat.json")
    fp = recomputed_fingerprint(recomputed)
    if os.path.exists(fpath) and os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("complete") and meta.get("recomputed_fp") == fp:
            print("  amass features from cache", flush=True)
            return np.load(fpath), np.array(meta["start"], np.int64), np.array(meta["length"], np.int64)
        print("  amass feature cache built from another recomputed set (or unstamped): rebuilding", flush=True)
    lengths = []
    for a in amass:
        key = f"{a['subset']}/{a['record']}"
        lengths.append(recomputed[key].shape[0] if key in recomputed else a["n_frames"])
    lengths = np.array(lengths, np.int64)
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    feat = np.lib.format.open_memmap(fpath + ".part", mode="w+", dtype=np.float32, shape=(int(lengths.sum()), 42))
    by_sub = collections.defaultdict(list)
    for i, a in enumerate(amass):
        by_sub[a["subset"]].append(i)
    bad = 0
    for sub in sorted(by_sub):
        t0 = time.time()
        z = np.load(f"{ma}/datasets/socle/amass/processed/canonical/amass_{sub}.npz")
        for i in by_sub[sub]:
            key = f"{sub}/{amass[i]['record']}"
            if key in recomputed:
                c = recomputed[key]
            else:
                c = z[f"{amass[i]['record']}/coords"]
                if c.shape[0] != amass[i]["n_frames"]:
                    bad += 1
            feat[start[i]:start[i] + lengths[i]] = heading_free(c)
        print(f"  amass features {sub}: {len(by_sub[sub])} records, {time.time() - t0:.0f} s", flush=True)
    feat.flush()
    del feat
    os.replace(fpath + ".part", fpath)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"start": start.tolist(), "length": lengths.tolist(), "frame_mismatch": bad, "complete": True,
                   "recomputed_fp": fp}, f)
    return np.load(fpath), start, lengths


def load_mx_features(ma, mx, cache):
    fpath, mpath = os.path.join(cache, "mx_feat.npy"), os.path.join(cache, "mx_feat.json")
    if os.path.exists(fpath) and os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("complete"):
            print("  motion_x features from cache", flush=True)
            return np.load(fpath), np.array(meta["start"], np.int64), np.array(meta["length"], np.int64), meta["frame_mismatch"]
    lengths = np.array([r["n_frames"] for r in mx], np.int64)
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    feat = np.zeros((int(lengths.sum()), 42), np.float32)
    by_sub = collections.defaultdict(list)
    for i, r in enumerate(mx):
        by_sub[r["subset"]].append(i)
    bad = 0
    for sub in sorted(by_sub):
        z = np.load(f"{ma}/datasets/socle/motion_x/processed/canonical/motionx_{sub}.npz")
        keys = sorted(k[:-len("/coords")] for k in z.files if k.endswith("/coords"))
        if len(keys) != len(by_sub[sub]):
            sys.exit(f"motion_x shard {sub}: {len(keys)} records, manifest {len(by_sub[sub])}")
        for i in by_sub[sub]:
            c = z[f"{mx[i]['record']}/coords"]
            if c.shape[0] != lengths[i]:
                bad += 1
                c = c[:lengths[i]]
            feat[start[i]:start[i] + c.shape[0]] = heading_free(c)
        print(f"  motion_x features {sub}: {len(by_sub[sub])} records", flush=True)
    np.save(fpath, feat)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"start": start.tolist(), "length": lengths.tolist(), "frame_mismatch": bad, "complete": True}, f)
    return feat, start, lengths, bad


# ----------------------------------------------------------------------------- recompute

N_VALID = 2   # rigid records per archive recomputed to check the recipe


def recompute_corrupted(ma, amass, cache):
    """Read the needed sequences with mdfm.data.amass.load_many (one pass per archive, pool of
    processes): the records to recompute (a["recompute"]: flagged, or pelvis-to-neck check),
    N_VALID rigid records per archive (recipe check) and, for the subsets whose manifest has no
    frame rate, every record (for its rate). Then forward kinematics of the recomputed and
    check records with mdfm.body.smplh. Role "corrupted" in the rows means "recomputed"."""
    rpath = os.path.join(cache, "recomputed.npz")
    ipath = os.path.join(cache, "recomputed.json")
    want = {f"{a['subset']}/{a['record']}" for a in amass if a["recompute"]}
    if os.path.exists(rpath) and os.path.exists(ipath):
        with open(ipath, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("complete"):
            z = np.load(rpath)
            if set(z.files) == want:
                print("  recomputed records from cache", flush=True)
                return {k: z[k] for k in z.files}, info
            print(f"  recomputation cache holds {len(z.files)} records, {len(want)} wanted now: recomputing", flush=True)
    from mdfm.data import amass as mdfm_amass
    by_sub = collections.defaultdict(list)
    for a in amass:
        by_sub[a["subset"]].append(a)
    jobs, role = [], {}
    for sub, rows in by_sub.items():
        needed = [a["take"] for a in rows if a["recompute"]]
        want_fps = any(not np.isfinite(a["fps"]) for a in rows)
        if not needed and not want_fps:
            continue
        valid = [a["take"] for a in rows if not a["recompute"]][:N_VALID]
        for t in needed:
            role[t] = "corrupted"
        for t in valid:
            role[t] = "validation"
        members = [a["take"] for a in rows] if want_fps else needed + valid
        jobs.append((os.path.getsize(mdfm_amass.archive(sub)), (sub, members)))
    jobs = [j for _, j in sorted(jobs, key=lambda x: -x[0])]
    fps_all = {}
    from multiprocessing import Pool
    print(f"  reading {len(jobs)} archives with mdfm.data.amass.load_many, {POOL} processes", flush=True)
    with Pool(min(POOL, len(jobs))) as pool:
        for sub, fps in pool.imap_unordered(load_worker, jobs):
            fps_all.update(fps)
    import torch
    import mdfm  # noqa: F401  (sets the torch cache folder)
    from mdfm.body.rotations import Z_UP_TO_Y_UP, axis_angle_to_matrix
    from mdfm.body.smplh import SMPLHBody
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bodies = {}
    take2rec = {a["take"]: a for a in amass}
    out, rows = {}, []
    for sub in sorted(by_sub):
        todo = [t for t in (a["take"] for a in by_sub[sub]) if t in role]
        if not todo:
            continue
        canon = np.load(f"{ma}/datasets/socle/amass/processed/canonical/amass_{sub}.npz")
        for t in todo:
            seq = mdfm_amass.load_sequence(sub, t)       # cache hit after load_many
            g = seq["gender"] if seq["gender"] in ("male", "female", "neutral") else "neutral"
            if g not in bodies:
                bodies[g] = SMPLHBody(g, dtype=torch.float64).to(dev)
            p = seq["poses"][:, :66]
            tr = seq["trans"]
            be = torch.as_tensor(seq["betas"], dtype=torch.float64, device=dev)
            J = []
            with torch.no_grad():
                for s in range(0, p.shape[0], 8192):
                    pp = torch.as_tensor(p[s:s + 8192], dtype=torch.float64, device=dev)
                    R = axis_angle_to_matrix(pp.reshape(-1, 22, 3))
                    jj, _ = bodies[g](R[:, 0], R[:, 1:], torch.as_tensor(tr[s:s + 8192], dtype=torch.float64, device=dev),
                                      be.expand(pp.shape[0], -1))
                    jj = jj @ Z_UP_TO_Y_UP.to(jj).T
                    J.append(jj[:, SEL22].cpu().numpy())
            c = normalise15(np.concatenate(J)).astype(np.float32)
            a = take2rec[t]
            ref = canon[f"{a['record']}/coords"]
            thigh = np.linalg.norm(c[:, 9] - c[:, 11], axis=-1)
            torso = np.linalg.norm(c[:, 1] - c[:, 0], axis=-1)
            rows.append({"subset": sub, "record": a["record"], "take": t, "role": role[t],
                         "frames_canonical": ref.shape[0], "frames_recomputed": c.shape[0],
                         "thigh_cv": round(float(thigh.std() / max(thigh.mean(), 1e-12)), 6),
                         "torso_cv": round(float(torso.std() / max(torso.mean(), 1e-12)), 6),
                         "max_abs_diff_to_canonical": round(float(np.abs(c - ref).max()), 6) if ref.shape == c.shape else ""})
            if role[t] == "corrupted":
                out[f"{sub}/{a['record']}"] = c
        print(f"  recomputed {sub}: {len(todo)} records", flush=True)
    np.savez(rpath, **out)
    info = {"rows": rows, "fps_all": fps_all, "not_found": [], "device": dev, "complete": True}
    with open(ipath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(info, f)
    return out, info


# ----------------------------------------------------------------------------- near-duplicate search

def window_starts(n10, W, stride):
    if n10 < W:
        return []
    s = list(range(0, n10 - W + 1, stride))
    if s[-1] != n10 - W:
        s.append(n10 - W)
    return s


def gather_windows(X10, starts, W):
    return np.concatenate([X10[starts + j] for j in range(W)], axis=1)


def fit_pca(A10, a_starts, Q10, q_starts, W, k, rng):
    sa = a_starts[rng.choice(len(a_starts), min(PCA_SAMPLE[0], len(a_starts)), replace=False)]
    sq = q_starts[rng.choice(len(q_starts), min(PCA_SAMPLE[1], len(q_starts)), replace=False)] if len(q_starts) else q_starts
    M = np.concatenate([gather_windows(A10, np.sort(sa), W), gather_windows(Q10, np.sort(sq), W)]).astype(np.float64)
    mu = M.mean(0)
    M -= mu
    cov = M.T @ M / M.shape[0]
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:k]
    return mu.astype(np.float32), v[:, order].astype(np.float32), float(w[order].sum() / w.sum())


def project(X10, starts, W, comp, mu):
    out = np.zeros((len(starts), comp.shape[1]), np.float32)
    for j in range(W):
        out += X10[starts + j] @ comp[j * 42:(j + 1) * 42]
    out -= mu @ comp
    return out


class KNN:
    def __init__(self, proj, subj):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.X = torch.as_tensor(proj, device=self.dev)
        self.xn = (self.X * self.X).sum(1)
        self.S = torch.as_tensor(subj, device=self.dev)

    def query(self, Q, qsubj=None, k=K_NN):
        torch = self.torch
        n = Q.shape[0]
        oi = np.empty((n, k), np.int64)
        od = np.empty((n, k), np.float32)
        chunk = KNN_CHUNK
        s = 0
        while s < n:
            try:
                q = torch.as_tensor(Q[s:s + chunk], device=self.dev)
                d = (q @ self.X.T).mul_(-2.0).add_(self.xn[None, :]).add_((q * q).sum(1, keepdim=True))
                if qsubj is not None:
                    qs = torch.as_tensor(qsubj[s:s + chunk], device=self.dev)
                    d.masked_fill_(self.S[None, :] == qs[:, None], float("inf"))
                v, i = torch.topk(d, k, dim=1, largest=False)
                oi[s:s + chunk] = i.cpu().numpy()
                od[s:s + chunk] = v.clamp_min(0).sqrt().cpu().numpy()
                s += chunk
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk = max(16, chunk // 2)
        return oi, od


def min_rows10(L30):
    """Aligned 10 Hz frames (every third frame of a 30 Hz clip of L30 frames) that guarantee the
    coverage rule at 30 Hz: C aligned 10 Hz frames give at least 3 (C - 1) + 1 aligned 30 Hz
    frames, so C >= (MIN_COVER L30 - 1) / 3 + 1. Without it a boundary alignment kept at 10 Hz
    can fall to a coverage of 0.77 at 30 Hz and push out a covered one of the same record."""
    return int(np.ceil((MIN_COVER * L30 - 1) / 3.0 + 1 - 1e-9))


def cover10_ok(cv, L10, L30):
    return np.asarray(cv) * L10 >= min_rows10(L30) - 1e-6


class ExactScan:
    """Exhaustive search of every alignment of a 10 Hz query on the amass 10 Hz signatures (GPU).

    The signatures are laid out with `gap` empty rows before every record, gap > (1 - MIN_COVER)
    x the longest query, so a window that keeps at least MIN_COVER of its frames on real rows
    lies in one record, and its middle row belongs to that record. A window's distance is the
    RMS joint distance over its real rows (the coverage rule of coarse_dist)."""

    def __init__(self, X10, n10, subj, gap):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        R = len(n10)
        self.G = int(gap)
        self.n10 = np.asarray(n10, np.int64)
        rec_of = np.repeat(np.arange(R), self.n10)
        row = np.arange(len(rec_of)) + self.G * (rec_of + 1)
        self.pstart = np.concatenate([[0], np.cumsum(self.n10)[:-1]]) + self.G * (np.arange(R) + 1)
        self.tot = int(self.n10.sum()) + self.G * (R + 1)
        P = np.zeros((self.tot, 42), np.float32)
        P[row] = X10
        self.rowrec = np.full(self.tot, -1, np.int64)
        self.rowrec[row] = rec_of
        V = np.zeros(self.tot, np.float32)
        V[row] = 1.0
        rowsubj = np.full(self.tot, -2, np.int64)
        rowsubj[row] = np.asarray(subj, np.int64)[rec_of]
        self.P = torch.as_tensor(P, device=self.dev)
        self.Pn = (self.P * self.P).sum(1)
        self.V = torch.as_tensor(V, device=self.dev)
        self.cV = torch.as_tensor(np.concatenate([[0], np.cumsum(V.astype(np.int64))]), device=self.dev)
        self.rowsubj = torch.as_tensor(rowsubj, device=self.dev)

    def query(self, x, L30, excl=-1, k=N_REFINE):
        """x (L, 42) at 10 Hz, taken from a 30 Hz clip of L30 frames -> up to k (record, start in
        record 10 Hz frames, 10 Hz distance), one per record, best first, over the alignments
        that keep min_rows10(L30) frames of x on the record."""
        torch = self.torch
        L = x.shape[0]
        c_min = min_rows10(L30)
        if L - self.G >= c_min:   # a window across two records would pass the coverage rule
            raise ValueError(f"query of {L} frames is longer than the gap allows ({self.G})")
        M = self.tot - L + 1
        X = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=self.dev)
        D = torch.zeros(M, device=self.dev)
        for c0 in range(0, L, EX_COLS):   # fixed-width column blocks keep the GPU memory small and constant
            Xc = X[c0:c0 + EX_COLS]
            w = Xc.shape[0]
            E = self.P @ Xc.T
            E.mul_(-2.0).add_(self.Pn[:, None]).add_((Xc * Xc).sum(1)[None, :]).clamp_min_(0).mul_(self.V[:, None])
            # D[o] += sum over t in the block of E[o + t, t - c0]
            D += torch.as_strided(E, (M, w), (w, w + 1), c0 * w).sum(1)
            del E
        C = (self.cV[L:L + M] - self.cV[:M]).to(D.dtype)
        bad = C < c_min - 1e-6
        if excl >= 0:
            bad |= self.rowsubj[L // 2:L // 2 + M] == excl
        D = (D / (14.0 * C.clamp_min(1.0))).masked_fill_(bad, float("inf"))
        out = []
        for _ in range(k):
            v, o = torch.min(D, 0)
            v, o = float(v), int(o)
            if not np.isfinite(v):
                break
            r = int(self.rowrec[o + L // 2])
            D[max(int(self.pstart[r]) - L // 2, 0):int(self.pstart[r] + self.n10[r]) - L // 2] = float("inf")
            out.append((r, o - int(self.pstart[r]), float(np.sqrt(v))))
        return out


def coarse_dist(xq, rec, off, A10, a10_start, a10_len):
    L = xq.shape[0]
    idx = off[:, None] + np.arange(L)[None, :]
    ln = a10_len[rec][:, None]
    valid = (idx >= 0) & (idx < ln)
    g = a10_start[rec][:, None] + np.clip(idx, 0, ln - 1)
    e = ((A10[g] - xq[None]) ** 2).reshape(len(rec), L, 14, 3).sum(-1).mean(-1)
    n = valid.sum(1)
    d = np.sqrt((e * valid).sum(1) / np.maximum(n, 1))
    d[n == 0] = np.inf
    return d, n / L


def refine_dist(xq, nat, fps, t0):
    L = xq.shape[0]
    T = nat.shape[0]
    deltas = np.arange(-REFINE_HALF, REFINE_HALF + 1e-9, REFINE_STEP)
    u = (t0 + deltas[:, None] + np.arange(L)[None, :] / MX_FPS) * fps
    valid = (u >= -1e-6) & (u <= T - 1 + 1e-6)
    a = interp_rows(nat, u)
    e = ((a - xq[None]) ** 2).reshape(len(deltas), L, 14, 3).sum(-1).mean(-1)
    n = valid.sum(1)
    d = np.sqrt((e * valid).sum(1) / np.maximum(n, 1))
    d[n == 0] = np.inf
    cover = n / L
    score = np.where(cover >= MIN_COVER, d, d + 100.0)
    k = int(np.argmin(score))
    return float(d[k]), float(cover[k]), float(t0 + deltas[k])


def search(Q30, q_start, q_len, q_subj, A, label, want_cands=False):
    """Near-duplicate search of queries (30 Hz heading-free features, concatenated) against
    the amass index A. q_subj: amass subject id to exclude per query (-1 for none)."""
    t0 = time.time()
    nq = len(q_start)
    sig = [Q30[q_start[i]:q_start[i] + q_len[i]][::3] for i in range(nq)]
    n10 = np.array([s.shape[0] for s in sig], np.int64)
    s10 = np.concatenate([[0], np.cumsum(n10)[:-1]]).astype(np.int64)
    Q10 = np.concatenate(sig) if nq else np.zeros((0, 42), np.float32)
    Q10m = mirror_feat(Q10)
    res = {"rec": np.full(nq, -1, np.int64), "d": np.full(nq, np.inf), "cover": np.zeros(nq),
           "off_s": np.full(nq, np.nan), "mir": np.zeros(nq, np.int64), "d10": np.full(nq, np.inf),
           "config": np.array([""] * nq, dtype=object), "cands": [None] * nq, "from_exact": np.zeros(nq, np.int64)}
    cand_rows = []
    for cfg in ("long", "short"):
        W = WIN[cfg]
        sel = np.nonzero((n10 >= W) & ((n10 < WIN["long"]) if cfg == "short" else True))[0]
        if not len(sel):
            continue
        qq, ql = [], []
        for i in sel:
            st = window_starts(int(n10[i]), W, QSTRIDE[cfg])
            qq += [i] * len(st)
            ql += st
        qq, ql = np.array(qq, np.int64), np.array(ql, np.int64)
        g = s10[qq] + ql
        ix = A["idx"][cfg]
        P = np.concatenate([project(Q10, g, W, ix["comp"], ix["mu"]), project(Q10m, g, W, ix["comp"], ix["mu"])])
        qsub = np.concatenate([q_subj[qq], q_subj[qq]]) if (q_subj >= 0).any() else None
        nn_i, nn_d = ix["knn"].query(P, qsub)
        qq2, ql2 = np.concatenate([qq, qq]), np.concatenate([ql, ql])
        mir = np.concatenate([np.zeros(len(qq), np.int64), np.ones(len(qq), np.int64)])
        rec = ix["win_rec"][nn_i]
        off = ix["win_loc"][nn_i] - ql2[:, None]
        cand_rows.append((np.repeat(qq2, K_NN), rec.ravel(), off.ravel(), np.repeat(mir, K_NN), nn_d.ravel()))
        res["config"][sel] = cfg
        print(f"  {label}: {cfg} windows {len(qq)} x2 searched, {time.time() - t0:.0f} s", flush=True)
    if not cand_rows:
        return res
    # exhaustive 10 Hz scan: the best N_REFINE records per mirror image join the candidates of
    # the approximate index (which alone misses the nearest alignment of many clips)
    import torch
    torch.cuda.empty_cache()
    ex = [None] * nq
    searched = np.nonzero(res["config"] != "")[0]
    for c, i in enumerate(searched):
        x0 = Q10[s10[i]:s10[i] + n10[i]]
        x1 = Q10m[s10[i]:s10[i] + n10[i]]
        ex[i] = [(r, o, mm) for mm, xx in ((0, x0), (1, x1))
                 for r, o, _ in A["exact"].query(xx, int(q_len[i]), int(q_subj[i]))]
        if (c + 1) % 5000 == 0:
            print(f"  {label}: exact scan {c + 1}/{len(searched)}, {time.time() - t0:.0f} s", flush=True)
    print(f"  {label}: exact scan done, {time.time() - t0:.0f} s", flush=True)
    q = np.concatenate([c[0] for c in cand_rows])
    rec = np.concatenate([c[1] for c in cand_rows])
    off = np.concatenate([c[2] for c in cand_rows])
    mir = np.concatenate([c[3] for c in cand_rows])
    d = np.concatenate([c[4] for c in cand_rows])
    o = np.lexsort((d, mir, off, rec, q))
    q, rec, off, mir, d = q[o], rec[o], off[o], mir[o], d[o]
    first = np.ones(len(q), bool)
    first[1:] = (q[1:] != q[:-1]) | (rec[1:] != rec[:-1]) | (off[1:] != off[:-1]) | (mir[1:] != mir[:-1])
    q, rec, off, mir, d = q[first], rec[first], off[first], mir[first], d[first]
    o = np.lexsort((d, q))
    q, rec, off, mir, d = q[o], rec[o], off[o], mir[o], d[o]
    bounds = np.searchsorted(q, np.arange(nq + 1))
    for i in range(nq):
        a, b = bounds[i], min(bounds[i + 1], bounds[i] + M_CAND)
        r_, o_, m_ = rec[a:b], off[a:b], mir[a:b]
        n_knn = len(r_)
        if ex[i]:
            e = np.array(ex[i], np.int64)
            r_, o_, m_ = np.concatenate([r_, e[:, 0]]), np.concatenate([o_, e[:, 1]]), np.concatenate([m_, e[:, 2]])
        if not len(r_):
            continue
        if want_cands:
            res["cands"][i] = np.unique(np.concatenate([rec[bounds[i]:bounds[i + 1]], r_]))
        x0 = Q10[s10[i]:s10[i] + n10[i]]
        x1 = Q10m[s10[i]:s10[i] + n10[i]]
        dc = np.full(len(r_), np.inf)
        cv = np.zeros(len(r_))
        for mm, xx in ((0, x0), (1, x1)):
            k = m_ == mm
            if k.any():
                dc[k], cv[k] = coarse_dist(xx, r_[k], o_[k], A["X10"], A["s10"], A["n10"])
        score = np.where(cover10_ok(cv, n10[i], q_len[i]), dc, dc + 100.0)
        order = np.argsort(score, kind="stable")
        seen, picks = set(), []
        for j in order:
            key = (int(r_[j]), int(m_[j]))
            if key in seen:
                continue
            seen.add(key)
            picks.append(j)
            if len(picks) == N_REFINE:
                break
        xq0 = Q30[q_start[i]:q_start[i] + q_len[i]]
        best = (np.inf, 0.0, np.nan, -1, 0, np.inf)
        best_j = -1
        for j in picks:
            r = int(r_[j])
            xq = xq0 if m_[j] == 0 else mirror_feat(xq0)
            nat = A["nat"][A["s_nat"][r]:A["s_nat"][r] + A["n_nat"][r]]
            dd, cc, ts = refine_dist(xq, nat, A["fps"][r], o_[j] / SIG_RATE)
            sc = dd if cc >= MIN_COVER else dd + 100.0
            bs = best[0] if best[1] >= MIN_COVER else best[0] + 100.0
            if sc < bs:
                best = (dd, cc, ts, r, int(m_[j]), float(dc[j]))
                best_j = j
        res["d"][i], res["cover"][i], res["off_s"][i], res["rec"][i], res["mir"][i], res["d10"][i] = best
        res["from_exact"][i] = int(best_j >= n_knn)
        if (i + 1) % 5000 == 0:
            print(f"  {label}: {i + 1}/{nq} queries refined, {time.time() - t0:.0f} s", flush=True)
    print(f"  {label}: done in {time.time() - t0:.0f} s", flush=True)
    return res


def pair_distance(xq, r, A):
    """Best aligned distance of one query against one amass record (all offsets, both mirrors)."""
    L10 = xq[::3]
    n = int(A["n10"][r])
    offs = np.arange(-L10.shape[0] + 1, n, dtype=np.int64)
    best = (np.inf, 0.0, np.nan, 0)
    if not len(offs) or not len(L10):
        return best
    for mm in (0, 1):
        xx = L10 if mm == 0 else mirror_feat(L10)
        dc, cv = coarse_dist(xx, np.full(len(offs), r, np.int64), offs, A["X10"], A["s10"], A["n10"])
        sc = np.where(cover10_ok(cv, L10.shape[0], xq.shape[0]), dc, dc + 100.0)
        j = int(np.argmin(sc))
        nat = A["nat"][A["s_nat"][r]:A["s_nat"][r] + A["n_nat"][r]]
        dd, cc, ts = refine_dist(xq if mm == 0 else mirror_feat(xq), nat, A["fps"][r], offs[j] / SIG_RATE)
        if (dd if cc >= MIN_COVER else dd + 100) < (best[0] if best[1] >= MIN_COVER else best[0] + 100):
            best = (dd, cc, ts, mm)
    return best


# ----------------------------------------------------------------------------- controls

def synth(nat, fps, dur, t_start, rng, level, mirror, frame_pick=False):
    T = nat.shape[0]
    L = max(1, int(round(dur * MX_FPS)))
    if frame_pick:
        i0 = int(round(t_start * fps))
        idx = np.rint(i0 + np.arange(L) * fps / MX_FPS).astype(np.int64)
        X = nat[idx[idx < T]]
    else:
        u = (t_start + np.arange(L) / MX_FPS) * fps
        X = interp_rows(nat, u[u <= T - 1 + 1e-6])
    P = to15(X)
    if level is not None:
        half, sd = PERTURB[level]
        f = rng.uniform(1 - half, 1 + half, 8)
        V = P[:, 1:] - P[:, PARENT[1:]]
        Q = np.zeros_like(P)
        for j in range(1, 15):
            Q[:, j] = Q[:, PARENT[j]] + V[:, j - 1] * f[BONE_GROUP[j]]
        Q[:, 1:] += rng.normal(0.0, sd, Q[:, 1:].shape)
        P = Q
    if mirror:
        P = P[:, MIRROR].copy()
        P[..., 0] *= -1
    a = rng.uniform(-np.pi, np.pi)
    ca, sa = np.cos(a), np.sin(a)
    x, z = P[..., 0].copy(), P[..., 2].copy()
    P[..., 0], P[..., 2] = ca * x + sa * z, -sa * x + ca * z
    return heading_free(normalise15(P))


def make_controls(amass, A, mx_durations, rng):
    """Synthetic queries: null (other subject), positives (light, heavy), exact-hash positives."""
    subj_ids = A["subj"]
    elig = np.nonzero(np.isfinite(A["fps"]) & (A["n_nat"] >= 2) &
                      ((A["n_nat"] - 1) / np.where(np.isfinite(A["fps"]), A["fps"], 1.0) >= 0.5))[0]
    sets = []
    for kind, n, level, fp in (("null", N_NULL, "light", False), ("pos_light", N_POS, "light", False),
                               ("pos_heavy", N_POS, "heavy", False), ("hash_pos", N_HASH_POS, None, True)):
        recs = rng.choice(elig, n, replace=False)
        for r in recs:
            r = int(r)
            fps = float(A["fps"][r])
            D = (A["n_nat"][r] - 1) / fps
            dur = float(min(rng.choice(mx_durations), D))
            if fp:
                i_start = int(rng.integers(0, max(1, int((D - dur) * fps) + 1)))
                t_start = i_start / fps
            else:
                t_start = float(rng.uniform(0, max(D - dur, 0.0)))
            mirror = bool(rng.random() < 0.5)
            nat = A["nat"][A["s_nat"][r]:A["s_nat"][r] + A["n_nat"][r]]
            x = synth(nat, fps, dur, t_start, rng, level, mirror, frame_pick=fp)
            sets.append({"kind": kind, "src": r, "t_start": t_start, "dur": x.shape[0] / MX_FPS, "mirror": int(mirror),
                         "x": x, "excl": int(subj_ids[r]) if kind == "null" else -1})
    return sets


# ----------------------------------------------------------------------------- main

def main():
    if "--out" not in sys.argv:
        sys.argv += ["--out", os.path.join(TREE, "reports", "nuisance")]
    from _common import parse_args, write_provenance
    args = parse_args(__doc__.splitlines()[0])
    ma = args.moveatlas
    os.environ["MOVEATLAS_ROOT"] = ma
    from mdfm import paths
    cache = paths.cache_dir("motionx_amass_dedup")
    paths.assert_not_moveatlas(cache)
    paths.assert_not_moveatlas(args.out)
    T0 = time.time()
    rng = np.random.default_rng(SEED)
    prov = write_provenance(types.SimpleNamespace(moveatlas=ma, out=cache),
                            {"task": "T0.7c", "seed": SEED, "hash_q": HASH_Q, "exact_frac": EXACT_FRAC,
                             "sig_rate": SIG_RATE, "win": WIN, "npca": NPCA, "k_nn": K_NN, "m_cand": M_CAND,
                             "n_refine": N_REFINE, "min_cover": MIN_COVER, "null_pct": NULL_PCT,
                             "perturb": PERTURB, "cache": cache.replace("\\", "/")})

    print("[1/7] tables", flush=True)
    amass, mx, checks = load_tables(ma)
    rig_csv = os.path.join(TREE, "reports", "probe_amass_canonical_rigidity", "records.csv")
    if not os.path.exists(rig_csv):
        sys.exit(f"missing {rig_csv}: run scripts/probe_amass_canonical_rigidity.py first")
    rig_mismatch = attach_rigidity(amass, rig_csv)
    n_corrupt = sum(a["corrupted"] for a in amass)
    print(f"  amass {len(amass)} records ({n_corrupt} corrupted, {sum(a['recompute'] for a in amass)} to recompute), "
          f"motion_x {len(mx)} records", flush=True)

    print("[2/7] metadata tier", flush=True)
    meta, dup_rows, name_idx_hits, name_pairs = metadata_tier(ma, amass, mx)

    print("[3/7] raw archives and recomputation of the corrupted amass records", flush=True)
    recomputed, rinfo = recompute_corrupted(ma, amass, cache)
    fps_raw = rinfo["fps_all"]
    n_fps_filled = 0
    for a in amass:
        if not np.isfinite(a["fps"]) and a["take"] in fps_raw and np.isfinite(fps_raw[a["take"]]):
            a["fps"] = float(fps_raw[a["take"]])
            n_fps_filled += 1
    n_fps_missing = sum(not np.isfinite(a["fps"]) for a in amass)

    print("[4/7] features", flush=True)
    nat, s_nat, n_nat = load_amass_features(ma, amass, recomputed, cache)
    mxf, mx_start, mx_len, mx_mismatch = load_mx_features(ma, mx, cache)
    fps = np.array([a["fps"] for a in amass])
    subj_names = sorted({a["subject_ns"] for a in amass})
    subj_of = {s: i for i, s in enumerate(subj_names)}
    A = {"nat": nat, "s_nat": s_nat, "n_nat": n_nat, "fps": fps,
         "subj": np.array([subj_of[a["subject_ns"]] for a in amass], np.int64)}

    print("[5/7] content hash tier", flush=True)
    hpath = os.path.join(cache, "hash_index_limbscale.npz")
    ok = np.isfinite(fps)
    stamp = f"{recomputed_fingerprint(recomputed)}|{HASH_Q}|{int(ok.sum())}"
    H = None
    if os.path.exists(hpath):
        with np.load(hpath) as z:
            if "stamp" in z.files and str(z["stamp"]) == stamp:
                H = {k: z[k] for k in ("keys", "rec", "frame")}
        print("  hash index from cache" if H is not None else "  hash index cache stale or unstamped: rebuilding", flush=True)
    if H is None:
        t0 = time.time()
        a_scale = limb_scales(nat, s_nat, n_nat)
        rec_of_frame = np.repeat(np.arange(len(amass)), n_nat)
        frame_of = np.arange(len(nat)) - np.repeat(s_nat, n_nat)
        keep = ok[rec_of_frame]
        keys = hash_frames(nat, s_nat, n_nat, a_scale)[keep]
        rec_of_frame, frame_of = rec_of_frame[keep], frame_of[keep]
        o = np.argsort(keys, kind="stable")
        H = {"keys": keys[o], "rec": rec_of_frame[o].astype(np.int32), "frame": frame_of[o].astype(np.int32)}
        np.savez(hpath + ".part.npz", stamp=np.array(stamp), **H)
        os.replace(hpath + ".part.npz", hpath)
        print(f"  hash index: {len(keys)} frames, {time.time() - t0:.0f} s", flush=True)
    fps_safe = np.where(ok, fps, MX_FPS)
    hb, hr, hm, ho, h_hits = hash_query(mxf, mx_start, mx_len, H, fps_safe)
    exact = np.nonzero(hb >= EXACT_FRAC)[0]
    print(f"  content hash: {h_hits} frame hits, {len(exact)} exact duplicates, max fraction {hb.max():.3f}", flush=True)

    print("[6/7] near-duplicate index", flush=True)
    sig = [signature10(nat[s_nat[r]:s_nat[r] + n_nat[r]], fps[r]) if ok[r] else np.zeros((0, 42), np.float32)
           for r in range(len(amass))]
    n10 = np.array([s.shape[0] for s in sig], np.int64)
    s10 = np.concatenate([[0], np.cumsum(n10)[:-1]]).astype(np.int64)
    X10 = np.concatenate(sig)
    del sig
    A.update({"X10": X10, "s10": s10, "n10": n10, "idx": {}})
    mx10 = [mxf[mx_start[i]:mx_start[i] + mx_len[i]][::3] for i in range(len(mx))]
    mn10 = np.array([s.shape[0] for s in mx10], np.int64)
    ms10 = np.concatenate([[0], np.cumsum(mn10)[:-1]]).astype(np.int64)
    MX10 = np.concatenate(mx10)
    energy = np.array([float(s.std(0).mean()) if s.shape[0] > 1 else 0.0 for s in mx10])
    del mx10
    ex_gap = int(np.floor((1.0 - MIN_COVER) * int(mn10.max()) + 1e-6)) + 1
    A["exact"] = ExactScan(X10, n10, A["subj"], ex_gap)
    print(f"  exact scan layout: {A['exact'].tot} rows (gap {ex_gap} rows before each record)", flush=True)
    pca_info = {}
    for cfg in ("long", "short"):
        W = WIN[cfg]
        wr, wl = [], []
        for r in range(len(amass)):
            if n10[r] >= W:
                k = n10[r] - W + 1
                wr.append(np.full(k, r, np.int64))
                wl.append(np.arange(k, dtype=np.int64))
        wr, wl = np.concatenate(wr), np.concatenate(wl)
        ga = s10[wr] + wl
        qs = np.concatenate([ms10[i] + np.array(window_starts(int(mn10[i]), W, QSTRIDE[cfg]), np.int64)
                             for i in range(len(mx)) if mn10[i] >= W])
        mu, comp, var = fit_pca(X10, ga, MX10, qs, W, NPCA[cfg], rng)
        proj = project(X10, ga, W, comp, mu)
        A["idx"][cfg] = {"mu": mu, "comp": comp, "win_rec": wr, "win_loc": wl,
                         "knn": KNN(proj, A["subj"][wr])}
        pca_info[cfg] = {"windows": int(len(wr)), "variance_kept": round(var, 4)}
        print(f"  index {cfg}: {len(wr)} amass windows, PCA {NPCA[cfg]} components keep {var:.3f} of the variance", flush=True)
    del MX10

    print("[7/7] controls and motion_x search", flush=True)
    mx_dur = (mx_len - 1) / MX_FPS
    ctrl = make_controls(amass, A, mx_dur, rng)
    cq = [c["x"] for c in ctrl]
    c_len = np.array([x.shape[0] for x in cq], np.int64)
    c_start = np.concatenate([[0], np.cumsum(c_len)[:-1]]).astype(np.int64)
    CQ = np.concatenate(cq)
    c_excl = np.array([c["excl"] for c in ctrl], np.int64)
    cres = search(CQ, c_start, c_len, c_excl, A, "controls", want_cands=True)
    chb, chr_, chm, cho, _ = hash_query(CQ, c_start, c_len, H, fps_safe)
    # distance of each positive to its true source at the true alignment
    d_true = np.full(len(ctrl), np.nan)
    for i, c in enumerate(ctrl):
        if c["kind"].startswith("pos"):
            r = c["src"]
            xq = CQ[c_start[i]:c_start[i] + c_len[i]]
            nat_r = nat[s_nat[r]:s_nat[r] + n_nat[r]]
            d_true[i] = refine_dist(xq if not c["mirror"] else mirror_feat(xq), nat_r, fps[r], c["t_start"])[0]
    kinds = np.array([c["kind"] for c in ctrl])
    eff = np.where(cres["cover"] >= MIN_COVER, cres["d"], np.inf)
    # best match carrying the same take name as the source (subject prefix of CMU names dropped)
    same_name = np.array([cres["rec"][i] >= 0 and take_key(amass[int(cres["rec"][i])]["take"]) == take_key(amass[c["src"]]["take"])
                          for i, c in enumerate(ctrl)])
    null_eff = eff[kinds == "null"]
    tau = float(np.percentile(null_eff, NULL_PCT))
    tau_strict = float(np.min(null_eff))
    print(f"  threshold (null {NULL_PCT} percentile): {tau:.4f} torso units; null minimum {tau_strict:.4f}", flush=True)

    mres = search(mxf, mx_start, mx_len, np.full(len(mx), -1, np.int64), A, "motion_x")
    m_eff = np.where(mres["cover"] >= MIN_COVER, mres["d"], np.inf)
    near = np.nonzero(m_eff <= tau)[0]
    near_strict = np.nonzero(m_eff < tau_strict)[0]

    # name-matched pairs: distance of each pair
    pair_d = []
    for j, i in name_pairs:
        xq = mxf[mx_start[j]:mx_start[j] + mx_len[j]]
        pair_d.append(pair_distance(xq, i, A) if ok[i] else (np.inf, 0.0, np.nan, 0))

    # ------------------------------------------------------------------ outputs
    out = args.out

    def amass_id(r):
        return f"amass_{amass[r]['subset']}/{amass[r]['record']}"

    def fmt(x, nd=4):
        return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)

    dup = []
    for d in dup_rows:
        r = d["mx"]
        dup.append({"motion_x_record": r["subject_ns"], "amass_record": "", "method": d["method"], "distance": "",
                    "motion_x_shard_record": f"motionx_{r['subset']}/{r['record']}", "amass_shard_record": "",
                    "coverage": "", "amass_start_s": "", "mirrored": ""})
    for r in name_idx_hits:
        dup.append({"motion_x_record": r["subject_ns"], "amass_record": "", "method": "metadata_name_mocap_index",
                    "distance": "", "motion_x_shard_record": f"motionx_{r['subset']}/{r['record']}",
                    "amass_shard_record": "", "coverage": "", "amass_start_s": "", "mirrored": ""})
    for (j, i), (dd, cc, ts, mm) in zip(name_pairs, pair_d):
        if not (cc >= MIN_COVER and dd <= tau):
            continue   # a name match is a duplicate only when the content agrees; all pairs are in the summary
        dup.append({"motion_x_record": mx[j]["subject_ns"], "amass_record": amass[i]["take"],
                    "method": "metadata_name_take", "distance": fmt(dd),
                    "motion_x_shard_record": f"motionx_{mx[j]['subset']}/{mx[j]['record']}",
                    "amass_shard_record": amass_id(i), "coverage": fmt(cc, 3), "amass_start_s": fmt(ts, 3), "mirrored": mm})
    for j in exact:
        r = int(hr[j])
        dd = pair_distance(mxf[mx_start[j]:mx_start[j] + mx_len[j]], r, A)[0]
        dup.append({"motion_x_record": mx[j]["subject_ns"], "amass_record": amass[r]["take"], "method": "content_hash",
                    "distance": fmt(dd), "motion_x_shard_record": f"motionx_{mx[j]['subset']}/{mx[j]['record']}",
                    "amass_shard_record": amass_id(r), "coverage": fmt(hb[j], 3),
                    "amass_start_s": fmt(ho[j] / fps_safe[r], 3), "mirrored": int(hm[j])})
    for j in near:
        r = int(mres["rec"][j])
        dup.append({"motion_x_record": mx[j]["subject_ns"], "amass_record": amass[r]["take"], "method": "near_duplicate",
                    "distance": fmt(mres["d"][j]), "motion_x_shard_record": f"motionx_{mx[j]['subset']}/{mx[j]['record']}",
                    "amass_shard_record": amass_id(r), "coverage": fmt(mres["cover"][j], 3),
                    "amass_start_s": fmt(mres["off_s"][j], 3), "mirrored": int(mres["mir"][j])})
    cols = ["motion_x_record", "amass_record", "method", "distance", "motion_x_shard_record", "amass_shard_record",
            "coverage", "amass_start_s", "mirrored"]
    with open(os.path.join(out, "motionx_amass_duplicates.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(dup)

    with open(os.path.join(out, "motionx_amass_nearest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["motion_x_record", "subset", "n_frames", "duration_s", "motion_energy", "hash_best_fraction",
                    "near_config", "nn_amass_record", "nn_distance", "nn_coverage", "nn_amass_start_s", "nn_mirrored",
                    "nn_distance_10hz", "near_duplicate", "near_duplicate_strict", "nn_from_exact_scan"])
        for j, r in enumerate(mx):
            k = int(mres["rec"][j])
            w.writerow([r["subject_ns"], r["subset"], r["n_frames"], round(float(mx_dur[j]), 3), round(float(energy[j]), 4),
                        round(float(hb[j]), 3), mres["config"][j] or "too_short",
                        amass[k]["take"] if k >= 0 else "", fmt(mres["d"][j]), fmt(mres["cover"][j], 3),
                        fmt(mres["off_s"][j], 3), int(mres["mir"][j]) if k >= 0 else "", fmt(mres["d10"][j]),
                        int(m_eff[j] <= tau), int(m_eff[j] < tau_strict), int(mres["from_exact"][j])])

    with open(os.path.join(out, "motionx_amass_controls.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "source_amass_record", "source_subject_ns", "crop_start_s", "duration_s", "mirrored",
                    "best_amass_record", "best_distance", "best_coverage", "best_is_source", "source_in_candidates",
                    "distance_to_source", "flagged", "hash_best_fraction", "hash_best_is_source", "best_same_take_name"])
        for i, c in enumerate(ctrl):
            r = c["src"]
            b = int(cres["rec"][i])
            cands = cres["cands"][i]
            kind = "null_other_subject" if c["kind"] == "null" else c["kind"]   # a bare "null" reads as missing in pandas
            w.writerow([kind, amass[r]["take"], amass[r]["subject_ns"], round(c["t_start"], 3), round(c["dur"], 3),
                        c["mirror"], amass[b]["take"] if b >= 0 else "", fmt(cres["d"][i]), fmt(cres["cover"][i], 3),
                        int(b == r), int(cands is not None and r in set(cands.tolist())), fmt(d_true[i]),
                        int(eff[i] <= tau), round(float(chb[i]), 3), int(int(chr_[i]) == r), int(same_name[i])])

    with open(os.path.join(out, "motionx_amass_recomputed.csv"), "w", encoding="utf-8", newline="") as f:
        cols_r = ["subset", "record", "take", "role", "frames_canonical", "frames_recomputed", "thigh_cv", "torso_cv",
                  "max_abs_diff_to_canonical"]
        w = csv.DictWriter(f, fieldnames=cols_r, lineterminator="\n")
        w.writeheader()
        flag_of = {(a["subset"], a["record"]): a["corrupted"] for a in amass}
        for r in rinfo["rows"]:
            # role: flagged (corrupted in records.csv), torso_check (recomputed for its pelvis-to-neck
            # variation only), validation (rigid record, recipe check)
            role = "validation" if r["role"] == "validation" else (
                "flagged" if flag_of[(r["subset"], r["record"])] else "torso_check")
            w.writerow({**r, "role": role})

    # ------------------------------------------------------------------ summary
    rrows = rinfo["rows"]
    rc = [r for r in rrows if r["role"] == "corrupted"]
    rv = [r for r in rrows if r["role"] == "validation"]
    rc_thigh_max = max([r["thigh_cv"] for r in rc] or [float("nan")])
    rc_torso5 = sum(1 for r in rc if r["torso_cv"] <= 0.05)
    rc_frames_eq = sum(1 for r in rc if r["frames_canonical"] == r["frames_recomputed"])
    rig_of = {(a["subset"], a["record"]): a for a in amass}
    rc_same = [r for r in rc if r["max_abs_diff_to_canonical"] != "" and r["max_abs_diff_to_canonical"] <= 1e-4]
    rc_same_torso_only = sum(1 for r in rc_same if rig_of[(r["subset"], r["record"])]["rig_thigh_cv"] <= 0.05)
    rc_same_torso = [r["torso_cv"] for r in rc_same]
    same_keys = {(r["subset"], r["record"]) for r in rc_same}
    rc_diff = [r for r in rc if (r["subset"], r["record"]) not in same_keys]
    rc_diff_thigh = sum(1 for r in rc_diff if rig_of[(r["subset"], r["record"])]["rig_thigh_cv"] > 0.05)
    rc_flagged = sum(1 for r in rc if rig_of[(r["subset"], r["record"])]["corrupted"])
    rc_same_flagged = sum(1 for r in rc_same if rig_of[(r["subset"], r["record"])]["corrupted"])
    rc_diff_flagged = sum(1 for r in rc_diff if rig_of[(r["subset"], r["record"])]["corrupted"])
    v_max = max([r["max_abs_diff_to_canonical"] for r in rv if r["max_abs_diff_to_canonical"] != ""] or [float("nan")])
    counts = collections.Counter(d["method"] for d in dup)
    flagged_any = {d["motion_x_record"] for d in dup if d["method"] in ("metadata_subset", "metadata_name_take", "content_hash",
                                                                       "near_duplicate")}

    def q(v, p):
        v = v[np.isfinite(v)]
        return round(float(np.percentile(v, p)), 4) if len(v) else ""

    L = []
    L += ["# motion_x against amass: duplicates before any hold-out", "",
          "Measured by scripts/probe_motionx_amass_dedup.py (plan task T0.7c); provenance in "
          "_provenance_motionx_amass_dedup.json. Distances are in torso units (pelvis to neck length; the canonical "
          "files are divided by it).", "",
          "## Counts per method", "",
          "| method | rows | motion_x records |", "|---|---|---|"]
    for mth in ("metadata_subset", "metadata_name_mocap_index", "metadata_name_take", "content_hash", "near_duplicate"):
        L.append(f"| {mth} | {counts.get(mth, 0)} | {len({d['motion_x_record'] for d in dup if d['method'] == mth})} |")
    L.append(f"| near_duplicate below the null minimum (subset of near_duplicate) | {len(near_strict)} | {len(near_strict)} |")
    L += ["",f"motion_x records flagged as duplicates by the subset, name-and-content, hash or near-duplicate tier: "
          f"{len(flagged_any)} of {len(mx)}. A (motion_x, amass) name match enters the file only when its aligned "
          f"distance is at most the near-duplicate threshold; every name-matched pair is listed in the metadata "
          f"section.", ""]
    L += ["## Inputs", "",
          f"- amass: {len(amass)} canonical records; motion_x: {len(mx)} canonical records "
          f"({len({r['clip'] for r in mx})} distinct clip names). Pooled manifest identical to the per-source manifests: "
          f"amass {checks['amass']['identical']}, motion_x {checks['motion_x']['identical']}.",
          f"- amass records whose rigidity row disagrees with the manifest (frames or missing): {rig_mismatch}.",
          f"- Corrupted amass records (reports/probe_amass_canonical_rigidity/records.csv): {n_corrupt}. They were "
          f"recomputed from the raw archives, not excluded, together with every record whose canonical pelvis-to-neck "
          f"variation exceeds {TORSO_CHECK} (the criterion of version 1 of the rigidity probe, tested here): "
          f"{len(rc)} records recomputed with mdfm.body.smplh on {rinfo['device']} ({rc_flagged} of them flagged); "
          f"largest thigh-length variation (coefficient of variation) after recomputation: {fmt(rc_thigh_max, 6)}; "
          f"pelvis-to-neck variation at most 5 percent after recomputation: {rc_torso5} (the pelvis-to-neck segment "
          f"spans the spine chain, which bends, so it is not a rigid bone); same frame count as the canonical record: "
          f"{rc_frames_eq}.",
          f"- Of these {len(rc)} recomputed records, {len(rc_same)} are reproduced by the recomputation within 1e-4 torso "
          f"units: their canonical joints are correct ({rc_same_torso_only} of them with a canonical thigh variation "
          f"at most 5 percent, {rc_same_flagged} of them flagged in records.csv; pelvis-to-neck variation "
          f"{fmt(min(rc_same_torso or [float('nan')]), 4)} to {fmt(max(rc_same_torso or [float('nan')]), 4)} after "
          f"recomputation). The other {len(rc_diff)} differ from their recomputation ({rc_diff_thigh} of them with a "
          f"canonical thigh variation above 5 percent, {rc_diff_flagged} of them flagged in records.csv): these are the "
          f"corrupted ones. The recomputed joints replace all {len(rc)} here.",
          f"- Recipe check on {len(rv)} rigid records recomputed the same way: largest absolute difference to the "
          f"canonical joints {fmt(v_max, 6)} torso units.",
          f"- amass frame rates missing from the manifest and read from the raw archives: {n_fps_filled}; still missing "
          f"(records left out of the hash and near-duplicate index): {n_fps_missing}.",
          f"- motion_x shard records whose length differs from the manifest: {mx_mismatch}.", ""]
    L += ["## Metadata tier", "",
          "| canonical motion_x subset | records |", "|---|---|"]
    for s, n in sorted(meta["canonical_subsets"].items()):
        L.append(f"| {s} | {n} |")
    L += ["", f"Subsets of the Motion-X++ Hugging Face release on disk: {', '.join(meta['hf_release_subsets'])}.", "",
          "Motion-X source index (Motion-X-main/datasets/all.txt), clips per subset, and how many canonical clip ids "
          "carry a clip name of that subset:", "",
          "| index subset | index clips | canonical clip ids with that name |", "|---|---|---|"]
    for s, n in sorted(meta["index_subsets"].items(), key=lambda x: -x[1]):
        L.append(f"| {s} | {n} | {meta['clip_ids_in_index'].get(s, 0)} |")
    n_in_index = sum(meta["clip_ids_in_index"].values())
    L += ["", f"Canonical clip ids equal to a source-index clip name, all subsets: {n_in_index}. With loose names "
          f"(lower-case alphanumerics, clip suffix dropped), {meta['loose_name_records_matched']} canonical records match "
          f"an index name; the index subsets they match:", "",
          "| canonical subset | index subset | records |", "|---|---|---|"]
    for m in meta["loose_name_matches"]:
        L.append(f"| {m['canonical_subset']} | {m['index_subset']} | {m['records']} |")
    L += ["", f"Loose-name matches into the mocap subsets ({', '.join(MX_MOCAP)}): "
          f"{meta['loose_name_matches_mocap_subsets']}. {len(mx) - meta['loose_name_records_matched']} canonical records "
          f"carry no index name even loosely, so names alone cannot clear a record; the subset comparison and the "
          f"content tiers carry the evidence."]
    rel = {s.lower() for s in meta["hf_release_subsets"]}
    in_rel = [s for s in MX_AMASS_DERIVED if s.lower() in rel]
    L += ["", f"Canonical motion_x records in the AMASS-derived index subsets ({', '.join(MX_AMASS_DERIVED)}): "
          f"{meta['metadata_subset_records']}. AMASS-derived subsets present in the release on disk: "
          f"{', '.join(in_rel) if in_rel else 'none'}. Motion-X's humanml_index.csv has "
          f"{meta['humanml_index_rows']} rows; they point to {meta['humanml_index_amass_takes_in_moveatlas']} distinct "
          f"amass takes present in MoveAtlas. Motion-X GRAB index clips whose name is an amass GRAB take: "
          f"{meta['index_grab_clips_with_amass_take']}. These would be duplicates if Motion-X's mocap subsets were "
          f"ever rebuilt into the corpus.", "",
          f"Name tier: canonical motion_x clip ids equal to a clip name of the Motion-X mocap subsets "
          f"({', '.join(MX_MOCAP)}): {meta['metadata_name_index_records']}; (motion_x, amass) pairs whose loose "
          f"names are equal (lower-case alphanumerics, the motion_x clip suffix and the amass file suffix dropped): "
          f"{meta['metadata_name_take_pairs']} over {meta['metadata_name_take_records']} motion_x records"
          + (f", aligned distance median {q(np.array([p[0] for p in pair_d]), 50)}, minimum "
             f"{q(np.array([p[0] for p in pair_d]), 0)}; at or below the threshold: "
             f"{sum(1 for p in pair_d if p[1] >= MIN_COVER and p[0] <= tau)}." if pair_d else "."), ""]
    if name_pairs:
        L += ["| motion_x record | amass take | aligned distance | coverage |", "|---|---|---|---|"]
        for (j, i), (dd, cc, ts, mm) in zip(name_pairs, pair_d):
            L.append(f"| {mx[j]['subject_ns']} | {amass[i]['take']} | {fmt(dd)} | {fmt(cc, 3)} |")
        L.append("")
    hp = kinds == "hash_pos"
    L += ["## Content hash tier", "",
          f"Every amass frame ({len(H['keys'])} frames of {int(ok.sum())} records) is indexed, heading removed and divided "
          f"by the record's limb scale (median of the mean length of the 8 rigid limb bones), rounded to {HASH_Q} limb "
          f"units; motion_x frames and their mirror images are looked up, a clip counts as an exact duplicate at "
          f"{EXACT_FRAC} of its frames on one record at one start frame and the stride of the two frame rates.", "",
          f"- motion_x exact duplicates: {len(exact)}; largest fraction of a clip's frames matched at one alignment: "
          f"{round(float(hb.max()), 3)}; clips with any frame matched: {int((hb > 0).sum())}; frame hits looked at: "
          f"{h_hits}.",
          f"- Control: {int(hp.sum())} amass crops taken by frame picking at 30 Hz (random start, half mirrored, random "
          f"heading, re-normalised): found as exact duplicates of their source: "
          f"{int(((chb >= EXACT_FRAC) & (chr_ == np.array([c['src'] for c in ctrl]))) [hp].sum())}.", ""]
    pl, ph, nu = kinds == "pos_light", kinds == "pos_heavy", kinds == "null"
    src = np.array([c["src"] for c in ctrl])
    durs = np.array([c["dur"] for c in ctrl])
    L += ["## Near-duplicate tier", "",
          f"Signature: heading-free pose at {SIG_RATE:g} Hz; windows of {WIN['long']} frames (clips of at least "
          f"{WIN['long'] / SIG_RATE:g} s) or {WIN['short']} frames (shorter clips), PCA to {NPCA['long']} and "
          f"{NPCA['short']} components (variance kept {pca_info['long']['variance_kept']} and "
          f"{pca_info['short']['variance_kept']}), exact nearest neighbours among {pca_info['long']['windows']} and "
          f"{pca_info['short']['windows']} amass windows, {K_NN} per query window, both mirror images, gives the best "
          f"{M_CAND} alignments per clip; an exhaustive scan of every alignment of the whole clip on the {SIG_RATE:g} Hz "
          f"amass signatures (both mirror images) adds the best alignment of each of its {N_REFINE} best records. These "
          f"candidates are scored at {SIG_RATE:g} Hz (with enough aligned {SIG_RATE:g} Hz frames to guarantee a coverage "
          f"of {MIN_COVER} at 30 Hz), the best {N_REFINE} records refined at 30 Hz over start offsets of plus or "
          f"minus {REFINE_HALF} s in steps of 1/120 s. Distance: root mean square joint position error over the aligned "
          f"frames and 14 joints; coverage: aligned share of the clip, at least {MIN_COVER}. Best match taken from an "
          f"exhaustive-scan candidate rather than from the approximate index (a closer record or a closer start of the "
          f"same record): motion_x {int(mres['from_exact'].sum())} of {int((mres['config'] != '').sum())} searched "
          f"clips, controls {int(cres['from_exact'].sum())} of {int((cres['config'] != '').sum())}.", "",
          f"Threshold: {NULL_PCT} percentile of the null, {round(tau, 4)} torso units. Null: {int(nu.sum())} amass crops "
          f"(duration drawn from the motion_x durations, light perturbation) searched with every record of their own "
          f"subject removed.", "",
          "| control | queries | distance p1 | p5 | p50 | p95 | flagged | best is source |", "|---|---|---|---|---|---|---|---|"]
    for name, m in (("null", nu), ("positive, light", pl), ("positive, heavy", ph)):
        L.append(f"| {name} | {int(m.sum())} | {q(eff[m], 1)} | {q(eff[m], 5)} | {q(eff[m], 50)} | {q(eff[m], 95)} | "
                 f"{int((eff[m] <= tau).sum())} | {int((cres['rec'][m] == src[m]).sum()) if name != 'null' else ''} |")
    L += ["", f"Perturbations: light = bone lengths scaled by up to {PERTURB['light'][0]} per bone group and jitter of "
          f"{PERTURB['light'][1]} torso units; heavy = {PERTURB['heavy'][0]} and {PERTURB['heavy'][1]}; both resampled "
          f"from the native rate to 30 Hz at a random sub-frame start, half mirrored, random heading.", "",
          "Recall of the positives at the threshold by duration:", "",
          "| duration | light: queries | light: flagged | heavy: queries | heavy: flagged |", "|---|---|---|---|---|"]
    for lo, hi in ((0, 1), (1, 3), (3, 1e9)):
        b = (durs >= lo) & (durs < hi)
        L.append(f"| {lo} to {hi if hi < 1e9 else 'max'} s | {int((pl & b).sum())} | {int((pl & b & (eff <= tau)).sum())} | "
                 f"{int((ph & b).sum())} | {int((ph & b & (eff <= tau)).sum())} |")
    L += ["", f"Distance of the positives to their true source at the true start: median light "
          f"{q(d_true[pl], 50)}, heavy {q(d_true[ph], 50)}; 95 percentile light {q(d_true[pl], 95)}, heavy "
          f"{q(d_true[ph], 95)}.", ""]
    nb = nu & (eff <= tau)
    pairs_c = collections.Counter(tuple(sorted((amass[int(src[i])]["subject_ns"], amass[int(cres["rec"][i])]["subject_ns"])))
                                  for i in np.nonzero(nb & same_name)[0])
    L += [f"Null crops at or below the threshold: {int(nb.sum())}; of them, matched to a take of the same name filed "
          f"under another amass subject id: {int((nb & same_name).sum())}. Subject-id pairs behind these matches: "
          + (", ".join(f"{a} and {b} ({n})" for (a, b), n in sorted(pairs_c.items())) if pairs_c else "none") + ". "
          f"Over all {int(nu.sum())} null crops, same-name matches under another subject id: {int((nu & same_name).sum())}, "
          f"distance median {q(eff[nu & same_name], 50)}."
          + (f" The low tail of the null is mostly made of takes with the same name under another subject id whose joint "
             f"trajectories are nearly identical (distance median {q(eff[nb & same_name], 50)} at or below the threshold; "
             f"light-perturbed copies of a take: median {q(eff[pl], 50)}), so the threshold sits at the level of "
             f"duplicates inside amass rather than of unrelated motions." if (nb & same_name).sum() * 2 > nb.sum() else "")
          + f" Without the same-name matches the null 1 percentile is {q(eff[nu & ~same_name], 1)}. These subject-id "
          f"pairs matter for the person map of the amass splits.", ""]
    mv = np.isfinite(m_eff)
    L += ["### motion_x", "",
          f"Clips searched: {int((mres['config'] != '').sum())} ({int((mres['config'] == 'long').sum())} long, "
          f"{int((mres['config'] == 'short').sum())} short); too short for a signature (under {WIN['short']} frames at "
          f"{SIG_RATE:g} Hz): {int((mres['config'] == '').sum())}; with an aligned candidate covering at least "
          f"{MIN_COVER}: {int(mv.sum())}.", "",
          f"Nearest amass distance of motion_x clips: p1 {q(m_eff, 1)}, p5 {q(m_eff, 5)}, p50 {q(m_eff, 50)}, "
          f"p95 {q(m_eff, 95)}. Near-duplicates at the threshold: {len(near)}. Below the null minimum ({round(tau_strict, 4)}, "
          f"strict threshold, recall of the positives at it: light "
          f"{int((pl & (eff < tau_strict)).sum())} of {int(pl.sum())}, heavy {int((ph & (eff < tau_strict)).sum())} of "
          f"{int(ph.sum())}): {len(near_strict)}.", ""]
    jmin = int(np.argmin(m_eff)) if mv.any() else -1
    if jmin >= 0:
        L += [f"Closest motion_x clip: {mx[jmin]['subject_ns']} ({round(float(mx_dur[jmin]), 3)} s, motion energy "
              f"{round(float(energy[jmin]), 4)}) at {q(m_eff, 0)} from {amass[int(mres['rec'][jmin])]['take']}.", ""]
    L += ["motion_x clips at or below other cut-offs taken from the controls (the positive cut-offs ask how many clips "
          "come as close to amass as a perturbed copy does):", "",
          "| cut-off | value | motion_x clips at or below |", "|---|---|---|"]
    for name, v in (("null, 1 percentile (the threshold)", tau), ("null, minimum", tau_strict),
                    ("null without same-name matches, 1 percentile", q(eff[nu & ~same_name], 1)),
                    ("light positives, 95 percentile", q(eff[pl], 95)), ("heavy positives, 5 percentile", q(eff[ph], 5)),
                    ("heavy positives, median", q(eff[ph], 50)), ("heavy positives, 95 percentile", q(eff[ph], 95)),
                    ("null, median", q(eff[nu], 50))):
        if v == "":
            continue
        L.append(f"| {name} | {round(float(v), 4)} | {int((m_eff <= float(v)).sum())} |")
    L += ["", "Nearest amass distance by motion_x subset (clips with an aligned candidate):", "",
          "| subset | clips | minimum | p5 | median |", "|---|---|---|---|---|"]
    subs_arr = np.array([r["subset"] for r in mx])
    for s in sorted(meta["canonical_subsets"]):
        m = (subs_arr == s) & mv
        L.append(f"| {s} | {int(m.sum())} | {q(m_eff[m], 0)} | {q(m_eff[m], 5)} | {q(m_eff[m], 50)} |")
    L.append("")
    if len(near):
        L += ["| motion_x subset | near-duplicates | of records |", "|---|---|---|"]
        cs = collections.Counter(mx[j]["subset"] for j in near)
        for s in sorted(meta["canonical_subsets"]):
            L.append(f"| {s} | {cs.get(s, 0)} | {meta['canonical_subsets'][s]} |")
        ca = collections.Counter(amass[int(mres['rec'][j])]["subset"] for j in near)
        L += ["", "Matched amass subsets: " + ", ".join(f"{k} {v}" for k, v in ca.most_common()) + ".",
              "", f"Motion energy (mean temporal standard deviation of the heading-free joints, torso units) of the "
              f"near-duplicates: median {q(energy[near], 50)}; of all motion_x clips: median {q(energy, 50)}. "
              f"Near-duplicates with energy below the 10th percentile of all clips ({q(energy, 10)}): "
              f"{int((energy[near] < np.percentile(energy, 10)).sum())}.", "",
              f"Near-duplicates lasting under 1 s: {int((mx_dur[near] < 1).sum())}.", ""]
    L += [f"Run time: {round((time.time() - T0) / 60, 1)} min. MoveAtlas: {prov['moveatlas_head']}.", ""]
    text = "\n".join(L)
    assert chr(0x2014) not in text
    with open(os.path.join(out, "motionx_amass_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    prov.update({"outputs": ["motionx_amass_duplicates.csv", "motionx_amass_nearest.csv", "motionx_amass_controls.csv",
                             "motionx_amass_recomputed.csv", "motionx_amass_summary.md"],
                 "threshold": tau, "threshold_strict": tau_strict, "pca": pca_info, "minutes": round((time.time() - T0) / 60, 1)})
    with open(os.path.join(out, "_provenance_motionx_amass_dedup.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
