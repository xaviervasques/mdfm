"""B4.4: the AMASS body store (float32 bodies of the train, dev and calib takes) and its report.

Builds one shard per (split, archive) with mdfm.data.body_store.build_shard, in worker processes (one
archive each; the raw archives are read once, sequentially, through mdfm.data.amass, which refuses
lockbox and SMPL-X archives before any cache read or tarfile.open), then reads every shard back
(body_store.verify: no lockbox, dev, calib or excluded person in train, no CODY name, sizes and offsets
consistent) and writes reports/body_store/: summary.md, archives.csv, treadmill.csv, priors.json (per
gender Gaussian shape priors, the mean pose in the heading frame and the joint-limit percentiles, all
from the train persons in the store), hashes.json and _provenance.json.

Store root: --root, else MDFM_BODY_STORE, else cache/body_store/ of this tree (gitignored); the target
<store>/store/bodies_v1 waits on owner question 12. A shard already built against the same index,
configuration and take list is kept unless --force.

Full build: <conda>/envs/mdfm/python.exe scripts/build_body_store.py --jobs 4
Report only: <conda>/envs/mdfm/python.exe scripts/build_body_store.py --report-only
"""
import argparse
import csv
import datetime
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                   help="MoveAtlas working tree (read-only)")
    p.add_argument("--out", default=os.path.join(TREE, "reports", "body_store"))
    p.add_argument("--root", default=None, help="store root (default MDFM_BODY_STORE, else cache/body_store)")
    p.add_argument("--splits", default="train,dev,calib")
    p.add_argument("--archives", default=None, help="comma-separated archives (default: every archive of the splits)")
    p.add_argument("--jobs", type=int, default=4, help="worker processes, one archive each")
    p.add_argument("--device", default=None, help="kinematics device (default cuda when available)")
    p.add_argument("--force", action="store_true", help="rebuild shards that are up to date")
    p.add_argument("--report-only", action="store_true", help="read the existing shards and write the report")
    p.add_argument("--stride", type=int, default=10, help="knot stride of the joint-limit percentiles")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    return a


def ram_bytes():
    """Physical RAM in bytes (Windows GlobalMemoryStatusEx), None elsewhere."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpage", "apage", "tvirt", "avirt", "aext")]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.total)
    except Exception:  # noqa: BLE001
        return None


def names(df, k=12):
    """Up to k take ids of df, sorted, comma-separated ('none' when empty)."""
    ids = sorted(df["take_id"])
    return ", ".join(ids[:k]) + (", ..." if len(ids) > k else "") if ids else "none"


def served_stance(store, rows, device):
    """Belt frame as served, per take row of `store` (a BodyStore): forward kinematics of BodyStore.take in
    float64, then (1) the median horizontal stance-foot speed over the stance knots (frames.treadmill_stats
    with central differences: the quantity of the spec's belt-frame test) and (2) the median velocity along
    the belt direction of the foot joints at mid-stance (local minima of their height within H0 of the
    floor), the bias of the belt estimate. m/s, two lists."""
    import numpy as np
    import torch
    from mdfm.body import frames as F
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.body.smplh import SkeletonBank, fk
    bank = SkeletonBank(torch.float64).to(device)
    speed, mid = [], []
    for i in rows:
        a = store.take(i)
        rest = bank.rest(torch.from_numpy(store.betas[i:i + 1]).to(device),
                         torch.tensor([int(store.gender[i])], device=device))[0]
        J, _ = fk(rot6d_to_matrix(torch.from_numpy(np.array(a["rot6d"], dtype=np.float64)).to(device)),
                  torch.from_numpy(a["transl"]).to(device), rest)
        speed.append(F.treadmill_stats(J, store.rate)["foot_speed"])
        V = torch.gradient(J, spacing=1.0 / store.rate, dim=0)[0]
        u = torch.from_numpy(a["belt"] / np.linalg.norm(a["belt"])).to(device)
        floor = F.floor_height(J)
        at = []
        for f in F.FEET:
            y = J[:, f, F.UP] - floor
            m = torch.nonzero((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]) & (y[1:-1] < F.H0))[:, 0] + 1
            at.append(V[m, f] @ u)
        at = torch.cat(at)
        mid.append(float(at.median()) if len(at) else float("nan"))
    return speed, mid


def main():
    args = parse()
    os.environ["MOVEATLAS_ROOT"] = args.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as np
    import pandas as pd
    import torch
    import mdfm  # noqa: F401
    from mdfm import paths
    from mdfm.data import amass_index as I
    from mdfm.data import body_store as S
    from mdfm.body.smplh import BODY_JOINTS

    paths.assert_not_moveatlas(args.out)
    os.makedirs(args.out, exist_ok=True)
    root = S.check_root((args.root or S.store_root()).replace("\\", "/").rstrip("/"))
    os.makedirs(root, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in S.SPLITS:
            sys.exit(f"unknown or refused split {s!r}: the body store holds {S.SPLITS}")
    want = set(a.strip() for a in args.archives.split(",")) if args.archives else None
    idx = I.index()
    idx["hours"] = idx["n_frames"] / idx["fps"] / 3600.0
    jobs = []
    for s in splits:
        rows = idx[idx["split"] == s]
        for a, g in rows.groupby("archive"):
            if want is None or a in want:
                jobs.append((float(g["hours"].sum()), s, a))
    if want:
        unknown = want - {a for _, _, a in jobs}
        if unknown:
            sys.exit(f"archives not in the splits {splits}: {sorted(unknown)}")
    jobs.sort(reverse=True)                       # longest archives first
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    t_build = None
    if not args.report_only:
        t0 = time.time()
        threads = max(1, (os.cpu_count() or 4) // max(1, args.jobs))
        print(f"building {len(jobs)} shards under {root} with {args.jobs} workers on {device}", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            fut = {ex.submit(S.shard_job, s, a, root, S.CUTOFF_HZ, S.RATE, device, args.force, threads): (s, a)
                   for _, s, a in jobs}
            for f in as_completed(fut):
                s, a = fut[f]
                m = f.result()
                print(f"done {s}/{a}: {m['n_takes']} takes, {m['n_knots']} knots, {m['seconds']} s "
                      f"({time.time() - t0:.0f} s elapsed)", flush=True)
        t_build = time.time() - t0

    # ------------------------------------------------------------ read back
    check = S.verify(root)
    print("verified:", {s: len(check[s]) for s in S.SPLITS})
    tables = {}
    metas = {}
    for s in S.SPLITS:
        for a in S._shards(root, s):
            folder = os.path.join(root, s, a)
            tables[(s, a)] = S._read_takes(os.path.join(folder, "takes.csv")).assign(split=s)
            with open(os.path.join(folder, "shard.json"), encoding="utf-8") as f:
                metas[(s, a)] = json.load(f)
    if not tables:
        sys.exit("no shard in the store")
    allt = pd.concat(tables.values(), ignore_index=True)
    for c in ("foot_speed", "pelvis_speed", "upright", "stance_balance", "belt_x", "belt_z", "floor_m", "contact_l",
              "contact_r"):
        allt[c] = allt[c].astype(float)
    allt["hours"] = allt["n_knots"] / S.RATE / 3600.0
    allt["stem"] = allt["take_id"].str.rsplit("/", n=1).str[-1].str.replace(r"\d*_(poses|stageii)\.npz$", "",
                                                                             regex=True)
    # the belt frame as served, for the detected takes
    allt["foot_speed_belt"] = np.nan
    allt["midstance_belt"] = np.nan
    t0 = time.time()
    for s in S.SPLITS:
        sel = allt.index[(allt["split"] == s) & (allt["treadmill"] == 1)]
        if len(sel):
            served = S.BodyStore(root, s)
            row_of = pd.Series(np.arange(len(served)), index=served.takes["take_id"].to_numpy())
            sp, mid = served_stance(served, row_of.loc[allt.loc[sel, "take_id"]].to_numpy(), device)
            allt.loc[sel, "foot_speed_belt"] = sp
            allt.loc[sel, "midstance_belt"] = mid
    print(f"served belt frame of {int(allt['treadmill'].sum())} detected takes in {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------ per archive
    ORDER = {s: i for i, s in enumerate(S.SPLITS)}
    arch_rows = []
    for s in S.SPLITS:
        for a in sorted(idx.loc[idx["split"] == s, "archive"].unique()):
            gi = idx[(idx["split"] == s) & (idx["archive"] == a)]
            m = metas.get((s, a))
            t = tables.get((s, a))
            row = {"split": s, "archive": a, "built": int(m is not None), "takes_index": len(gi),
                   "persons_index": gi["person_id"].nunique(), "hours_index": round(float(gi["hours"].sum()), 4)}
            if m is not None:
                row.update({"takes_stored": m["n_takes"], "takes_short": m["n_skipped_short"],
                            "persons_stored": t["person_id"].nunique(), "knots": m["n_knots"],
                            "hours_store": round(m["n_knots"] / S.RATE / 3600.0, 4),
                            "treadmill_detected": int(t["treadmill"].sum()),
                            "treadmill_named": int(t["treadmill_named"].sum()),
                            "detected_and_named": int((t["treadmill"] & t["treadmill_named"]).sum()),
                            "contact_l": round(float(t["contact_l"].astype(float).mul(t["n_knots"]).sum()
                                                     / t["n_knots"].sum()), 4),
                            "contact_r": round(float(t["contact_r"].astype(float).mul(t["n_knots"]).sum()
                                                     / t["n_knots"].sum()), 4),
                            "bytes": sum(v["bytes"] for v in m["arrays"].values()),
                            "seconds_load": m["seconds_load"], "seconds": m["seconds"], "device": m["device"]})
            arch_rows.append(row)
    fields = ["split", "archive", "built", "takes_index", "persons_index", "hours_index", "takes_stored",
              "takes_short", "persons_stored", "knots", "hours_store", "treadmill_detected", "treadmill_named",
              "detected_and_named", "contact_l", "contact_r", "bytes", "seconds_load", "seconds", "device"]
    with open(os.path.join(args.out, "archives.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(arch_rows)

    tm = allt[(allt["treadmill"] == 1) | (allt["treadmill_named"] == 1)]
    tm[["split", "archive", "take_id", "treadmill_named", "treadmill", "n_knots", "foot_speed", "pelvis_speed",
        "upright", "stance_balance", "belt_x", "belt_z", "contact_l", "contact_r", "foot_speed_belt",
        "midstance_belt"]].to_csv(os.path.join(args.out, "treadmill.csv"), index=False, lineterminator="\n",
                                  float_format="%.4f")

    # ------------------------------------------------------------ train statistics
    train_built = sorted(a for (s, a) in tables if s == "train")
    train_all = sorted(idx.loc[idx["split"] == "train", "archive"].unique())
    priors = None
    if train_built:
        tt = allt[allt["split"] == "train"]
        store = S.BodyStore(root, "train")
        assert set(store.takes["person_id"]) == set(tt["person_id"])
        t0 = time.time()
        pose = S.pose_statistics(store, stride=args.stride)
        print(f"pose statistics over {pose['n_knots']} train knots in {time.time() - t0:.0f} s", flush=True)
        priors = {"version": S.VERSION, "status": "from the train persons of the store",
                  "train_archives": train_built, "train_archives_missing": sorted(set(train_all) - set(train_built)),
                  "train_persons": int(tt["person_id"].nunique()),
                  "train_persons_sha256": I.person_list_sha256(tt["person_id"]),
                  "index_sha256": S.index_sha256(),
                  "shape_prior": {"note": "Gaussian over the 16 AMASS betas, one vector per person (the mean over "
                                          "its takes); neutral pools both sexes (sex unknown)",
                                  **S.shape_priors(tt)},
                  "mean_pose": {"note": "mean 6D value per joint over every train knot, row 0 the root orientation "
                                        "with each knot's own yaw removed (heading frame); rot6d_to_matrix gives "
                                        "the rotation", "joints": BODY_JOINTS, "rot6d": pose["mean_rot6d"],
                                "n_knots": pose["n_knots"]},
                  "joint_limits": {"note": f"percentiles {pose['percentiles']} of each axis-angle component (rad) "
                                           f"per joint, root in the heading frame, over every {pose['stride']}th "
                                           "knot of each train shard", "joints": BODY_JOINTS,
                                   "percentiles": pose["percentiles"], "lo": pose["aa_lo"], "hi": pose["aa_hi"],
                                   "n_sampled": pose["n_sampled"], "stride": pose["stride"]}}
        with open(os.path.join(args.out, "priors.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(priors, f, indent=1)
            f.write("\n")

    # ------------------------------------------------------------ hashes
    with open(os.path.join(TREE, "reports", "amass_split", "hashes.json"), encoding="utf-8") as f:
        split_hashes = json.load(f)
    idx_sha = S.index_sha256()
    person_sha = check["persons_sha256"]
    complete = {s: all((s, a) in tables for a in idx.loc[idx["split"] == s, "archive"].unique()) for s in S.SPLITS}
    hashes = {"index_sha256": idx_sha, "index_sha256_equals_amass_split": idx_sha == split_hashes["index_csv_sha256"],
              "task_vocabulary_sha256": S.vocabulary_sha256(), "store_version": S.VERSION,
              "person_lists": {s: {"store_sha256": person_sha[s],
                                   "index_sha256": split_hashes["person_lists"][s]["sha256"],
                                   "split_complete": complete[s],
                                   "equal": person_sha[s] == split_hashes["person_lists"][s]["sha256"]}
                               for s in S.SPLITS},
              "shards": {f"{s}/{a}": {"takes_csv_sha256": m["takes_csv_sha256"], "persons_sha256": m["persons_sha256"]}
                         for (s, a), m in sorted(metas.items())}}
    with open(os.path.join(args.out, "hashes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(hashes, f, indent=1)
        f.write("\n")

    # ------------------------------------------------------------ summary
    store_bytes = sum(r.get("bytes", 0) for r in arch_rows)
    ram = ram_bytes()
    L = ["# AMASS body store (B4.4)", "",
         "Built by scripts/build_body_store.py with mdfm/data/body_store.py and mdfm/body/frames.py; provenance in "
         "_provenance.json. Takes come from the AMASS take index (cache/amass_index/takes.csv.gz) through "
         "mdfm.data.amass, which refuses lockbox and SMPL-X archives before any cache read or tarfile.open; "
         "every shard was read back (body_store.verify): no lockbox, dev, calib or excluded person in a train "
         "shard, no CODY name, offsets and array sizes consistent.", "",
         f"Store root: {root} ({'default cache/body_store/ of this tree, gitignored' if root == S.store_root({}) else 'set by --root or ' + S.STORE_ENV}); "
         f"<store> exists: {os.path.isdir('<store>')}. The store moves to {S.STORE_TARGET} once the owner creates "
         f"<store> (question 12): move the folder and set {S.STORE_ENV}.", "",
         f"Index SHA-256: {idx_sha} (equal to reports/amass_split/hashes.json: "
         f"{hashes['index_sha256_equals_amass_split']}). Task vocabulary SHA-256: {hashes['task_vocabulary_sha256']} "
         "(docs/rules/task_vocabulary.yaml, DRAFT).", "",
         f"Pipeline per take: prep.amass_to_body_clock ({S.RATE:.0f} Hz, {S.CUTOFF_HZ:.0f} Hz zero-phase low-pass, "
         "y up, gendered SMPL+H body), joint positions and Hermite velocities at every knot from "
         "mdfm.body.kinematics (float32), floor to y = 0 (2nd percentile of the lowest foot joint), belt velocity "
         "of treadmill takes, contacts in the belt frame (ankle or foot joint within 0.05 m of the floor and slower "
         f"than 0.3 m/s). float32 rot6d (N, 22, 6), transl (N, 3) m, uint8 contact (N, 2). Takes shorter than "
         f"{S.MIN_S:.0f} s are skipped. transl.f32 holds treadmill takes in the lab frame; BodyStore applies the "
         "belt shift (belt_x, belt_z of takes.csv) in float64 on read (see the treadmill section).", ""]
    missing = [(r["split"], r["archive"]) for r in arch_rows if not r["built"]]
    if missing:
        L += [f"**Partial build.** Built: {', '.join(f'{s}/{a}' for s, a in sorted(tables))}. Not built: "
              f"{', '.join(f'{s}/{a}' for s, a in missing)}. Full build: <conda>/envs/mdfm/python.exe "
              "scripts/build_body_store.py --jobs 4 (built shards are kept).", ""]
    L += ["## Hours per split", "",
          "| split | archives built | takes stored | takes under 1 s | persons | knots | hours (store) | hours (index) | "
          "treadmill takes |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S.SPLITS:
        rs = [r for r in arch_rows if r["split"] == s]
        b = [r for r in rs if r["built"]]
        pers = allt.loc[allt["split"] == s, "person_id"].nunique()
        L.append(f"| {s} | {len(b)} of {len(rs)} | {sum(r['takes_stored'] for r in b):,} | "
                 f"{sum(r['takes_short'] for r in b):,} | {pers} | {sum(r['knots'] for r in b):,} | "
                 f"{sum(r['hours_store'] for r in b):.2f} | {sum(r['hours_index'] for r in b):.2f} | "
                 f"{sum(r['treadmill_detected'] for r in b):,} |")
    L += ["", "Hours (store) count knots / 100 Hz; hours (index) are manifest frames / fps of the same archives.", "",
          "## Hours per archive", "",
          "| split | archive | takes stored | under 1 s | persons | hours (store) | treadmill detected | named treadmill | "
          "both | contact left, right | build s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(arch_rows, key=lambda r: (ORDER[r["split"]], r["archive"])):
        if not r["built"]:
            L.append(f"| {r['split']} | {r['archive']} | not built | | | ({r['hours_index']:.2f} in the index) | | | | | |")
            continue
        L.append(f"| {r['split']} | {r['archive']} | {r['takes_stored']:,} | {r['takes_short']} | {r['persons_stored']} | "
                 f"{r['hours_store']:.2f} | {r['treadmill_detected']:,} | {r['treadmill_named']:,} | "
                 f"{r['detected_and_named']} | {r['contact_l']:.3f}, {r['contact_r']:.3f} | {r['seconds']:.0f} |")
    det = allt[allt["treadmill"] == 1]
    fr = S.frames
    # belt-frame drift of the detected takes against the stored lab-frame translation
    drift = float((np.hypot(det["belt_x"], det["belt_z"]) * (det["n_knots"] - 1) / S.RATE).max()) if len(det) else 0.0
    lab = 0.0
    for (s, a), t in tables.items():
        d = t[t["treadmill"] == 1]
        if len(d):
            arr = np.memmap(os.path.join(root, s, a, "transl.f32"), dtype="<f4", mode="r").reshape(-1, 3)
            for o, n in zip(d["offset"], d["n_knots"]):
                lab = max(lab, float(np.abs(arr[o:o + n][:, [0, 2]]).max()))
            del arr
    speeds = allt[(allt["foot_speed"] > fr.TREADMILL_FOOT) & (allt["pelvis_speed"] < fr.TREADMILL_PELVIS)]
    not_up = speeds[speeds["upright"] <= fr.TREADMILL_UPRIGHT]
    one_foot = speeds[(speeds["upright"] > fr.TREADMILL_UPRIGHT) & (speeds["stance_balance"] < fr.TREADMILL_BALANCE)]
    sway = det[det["pelvis_speed"] >= 0.2]
    gait = allt[(allt["archive"] == "BMLrub") & allt["stem"].str.contains("treadmill|normal_walk|normal_jog")]
    gait_miss = gait[gait["treadmill"] == 0]
    L += ["", "## Treadmill takes", "",
          f"Rule (mdfm/body/frames.py, [cfg]): median horizontal stance-foot speed over the stance knots above "
          f"{fr.TREADMILL_FOOT} m/s, median horizontal pelvis speed below {fr.TREADMILL_PELVIS} m/s, trunk upright "
          f"(median cosine of the pelvis-to-neck vector with +y above {fr.TREADMILL_UPRIGHT}) and alternating stance "
          f"(each foot the stance foot on at least {fr.TREADMILL_BALANCE} of the stance knots). Detected: {len(det):,} "
          f"takes ({det['hours'].sum():.2f} h), all in {', '.join(sorted(det['archive'].unique())) or 'no archive'}; "
          f"take name containing 'treadmill': {int(allt['treadmill_named'].sum()):,}; "
          f"both: {int((allt['treadmill'] & allt['treadmill_named']).sum()):,}. Detected takes are served in the belt "
          "frame. Per take: treadmill.csv.", "",
          f"Belt-frame drift over a detected take (belt speed x duration): up to {drift:.1f} m, where the float32 "
          f"spacing is {float(np.spacing(np.float32(drift))):.2e} m; the stored lab-frame translation of the detected "
          f"takes stays within {lab:.2f} m of the origin horizontally (spacing "
          f"{float(np.spacing(np.float32(lab))):.2e} m), and BodyStore applies the shift in float64.", "",
          f"The two speed conditions alone pass {len(speeds):,} takes; the trunk condition rejects {len(not_up)} "
          f"({names(not_up)}), the alternation condition {len(one_foot)} ({names(one_foot)}). The spec's pelvis "
          f"bound of 0.2 m/s would miss {len(sway)} detected takes (pelvis speed {sway['pelvis_speed'].min():.3f} to "
          f"{sway['pelvis_speed'].max():.3f} m/s; {int(sway['treadmill_named'].sum())} named 'treadmill')."
          if len(sway) else "", "",
          f"BMLrub takes named treadmill, normal_walk or normal_jog: {len(gait):,}; not detected: {len(gait_miss)}"
          + (" (" + ", ".join(f"{r.take_id} with foot {r.foot_speed:.2f} and pelvis {r.pelvis_speed:.2f} m/s"
                             for r in gait_miss.head(6).itertuples()) + ")." if len(gait_miss) else "."), "",
          "| take name stem | archive | detected | stored | median belt speed m/s |", "|---|---|---|---|---|"]
    g = allt.groupby(["stem", "archive"])
    stems = g["treadmill"].agg(["sum", "size"]).reset_index()
    stems = stems[stems["sum"] > 0].sort_values(["sum", "stem"], ascending=[False, True])
    for _, r in stems.head(25).iterrows():
        d = det[(det["stem"] == r["stem"]) & (det["archive"] == r["archive"])]
        sp = float(np.hypot(d["belt_x"], d["belt_z"]).median())
        L.append(f"| {r['stem']} | {r['archive']} | {int(r['sum'])} | {int(r['size'])} | {sp:.2f} |")
    named_miss = allt[(allt["treadmill_named"] == 1) & (allt["treadmill"] == 0)]
    L += ["", f"Named treadmill takes not detected: {len(named_miss)}"
          + (f" ({', '.join(named_miss['take_id'].head(5))}{', ...' if len(named_miss) > 5 else ''})." if len(named_miss) else "."),
          ""]
    if len(det):
        kind = np.where(det["stem"].str.contains("jog"), "jog (name)", "walk (name)")
        L += ["Contacts of the detected takes in the belt frame (share of knots per foot, weighted by knots), by the gait "
              "in the take name:", "",
              "| detected takes | takes | hours | median belt speed m/s | contact left | contact right |",
              "|---|---|---|---|---|---|"]
        for k, d in det.groupby(kind):
            w = d["n_knots"] / d["n_knots"].sum()
            L.append(f"| {k} | {len(d):,} | {d['hours'].sum():.2f} | {float(np.hypot(d['belt_x'], d['belt_z']).median()):.2f} | "
                     f"{float((d['contact_l'] * w).sum()):.3f} | {float((d['contact_r'] * w).sum()):.3f} |")
        L.append("")
        L += [f"Stance foot of the detected takes in the belt frame as served (BodyStore.take, forward kinematics in "
              f"float64, central differences; treadmill.csv columns foot_speed_belt and midstance_belt): per take, the "
              f"median horizontal stance-foot speed over the stance knots (the lower foot joint within {fr.H0} m of the "
              "floor), the quantity the spec's belt-frame test bounds by 0.1 m/s, and the median velocity along the belt "
              f"of the foot joints at mid-stance (local minima of their height within {fr.H0} m of the floor), the bias "
              "of the belt estimate. Quantiles over takes:", "",
              "| detected takes | takes | stance-foot speed: median | p90 | max | takes below 0.1 m/s | mid-stance "
              "velocity along the belt: median | p10 | p90 |", "|---|---|---|---|---|---|---|---|---|"]
        for k, d in det.groupby(kind):
            sp, md = d["foot_speed_belt"], d["midstance_belt"]
            L.append(f"| {k} | {len(d):,} | {sp.median():.3f} | {sp.quantile(0.9):.3f} | {sp.max():.3f} | "
                     f"{int((sp < 0.1).sum()):,} | {md.median():.3f} | {md.quantile(0.1):.3f} | {md.quantile(0.9):.3f} |")
        L.append("")
    L += ["## Tasks (docs/rules/task_vocabulary.yaml, DRAFT)", "", "| task | " + " | ".join(S.SPLITS) + " |",
          "|---|" + "---|" * len(S.SPLITS)]
    th = allt.groupby(["task_name", "split"])["hours"].sum().unstack(fill_value=0.0)
    for tname in th.index:
        L.append(f"| {tname} | " + " | ".join(f"{th.loc[tname].get(s, 0.0):.2f}" for s in S.SPLITS) + " |")
    L += ["", "Hours (store) per task and split.", ""]
    if priors is not None:
        sp = priors["shape_prior"]
        L += ["## Train statistics (priors.json)", "",
              f"From the {priors['train_persons']} train persons of the store (archives {', '.join(train_built)}"
              + (f"; not built: {', '.join(priors['train_archives_missing'])}" if priors["train_archives_missing"] else "")
              + f"), person-list SHA-256 {priors['train_persons_sha256']}. Statistics of AMASS fits, not SMPL+H "
              "arrays.", "",
              "| shape prior | persons | mean beta_0 | sd beta_0 | mean beta_1 | sd beta_1 | largest within-person spread |",
              "|---|---|---|---|---|---|---|"]
        for gname in ("female", "male", "neutral"):
            p = sp[gname]
            cov = np.array(p["cov"])
            L.append(f"| {gname} | {p['n_persons']} | {p['mean'][0]:.3f} | {np.sqrt(cov[0, 0]):.3f} | {p['mean'][1]:.3f} | "
                     f"{np.sqrt(cov[1, 1]):.3f} | {p['max_within_person']:.2e} |")
        tt = allt[allt["split"] == "train"]
        B = tt[S.BETAS].astype(float)
        spread = (B - B.groupby(tt["person_id"]).transform("mean")).abs().max(axis=1).groupby(tt["person_id"]).max()
        vary = spread[spread > 1e-6].sort_values(ascending=False)
        subj = tt.groupby("person_id")["subject_ns"].nunique()
        L += ["", f"Each person enters the prior once, with the mean betas of its takes. Persons whose betas differ "
              f"across takes: {len(vary)}"
              + (" (" + "; ".join(f"{p}, largest difference {v:.2f}, {subj[p]} subject folders" for p, v in vary.items())
                 + ")." if len(vary) else ".")]
        jl = priors["joint_limits"]
        lo, hi = np.array(jl["lo"]), np.array(jl["hi"])
        L += ["", f"Mean pose: mean 6D value per joint over {priors['mean_pose']['n_knots']:,} train knots, root in the "
              f"heading frame. Joint limits: percentiles {jl['percentiles']} of each axis-angle component over "
              f"{jl['n_sampled']:,} knots (every {jl['stride']}th knot); widest joint range "
              f"{BODY_JOINTS[int(np.argmax((hi - lo).max(1)))]} ({float((hi - lo).max()):.2f} rad), narrowest "
              f"{BODY_JOINTS[int(np.argmin((hi - lo).max(1)))]} ({float((hi - lo).max(1).min()):.2f} rad on its widest "
              "axis).", ""]
    L += ["## Person lists", "", "| split | store SHA-256 | equals the index list | split complete |", "|---|---|---|---|"]
    for s in S.SPLITS:
        h = hashes["person_lists"][s]
        L.append(f"| {s} | {h['store_sha256']} | {h['equal']} | {h['split_complete']} |")
    built = [r for r in arch_rows if r["built"]]
    ends = [datetime.datetime.fromisoformat(m["built_at"]) for m in metas.values()]
    starts = [e - datetime.timedelta(seconds=m["seconds"]) for e, m in zip(ends, metas.values())]
    span_min = (max(ends) - min(starts)).total_seconds() / 60.0
    L += ["", "## Size and time", "",
          f"Store arrays: {store_bytes / 2**30:.2f} GiB for {int(allt['n_knots'].sum()):,} knots; physical RAM "
          f"{(ram or 0) / 2**30:.2f} GiB. Shard build times (reading the raw archive through mdfm.data.amass, prep, "
          f"kinematics on {', '.join(sorted({r['device'] for r in built}))}): sum {sum(r['seconds'] for r in built) / 60:.1f} "
          f"min, longest {max(r['seconds'] for r in built) / 60:.1f} min ({max(built, key=lambda r: r['seconds'])['archive']}); "
          f"shards run in parallel worker processes. The shards in the store were built from {min(starts):%Y-%m-%d %H:%M} "
          f"to {max(ends):%Y-%m-%d %H:%M} (shard.json built_at less seconds), {span_min:.1f} min from the first start to "
          "the last end."
          + (f" This run: {t_build / 60:.1f} min wall time with {args.jobs} workers (shards already built are kept)."
             if t_build is not None else " This run read the shards back and wrote the report only (--report-only).")]
    text = "\n".join(L) + "\n"
    assert "\u2014" not in text
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    git = subprocess.run(["git", "-c", f"safe.directory={args.moveatlas}", "-C", args.moveatlas, "log", "-1"],
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    head = [" ".join(ln.split()) for ln in git.stdout.splitlines() if ln.strip()]
    prov = {"script": os.path.basename(sys.argv[0]), "argv": sys.argv[1:], "moveatlas_root": args.moveatlas,
            "moveatlas_head": " ".join(head[:1] + head[2:4]) if head else f"(git failed: {git.stderr.strip()[:200]})",
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "torch": torch.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "store_root": root, "jobs": args.jobs, "build_seconds": None if t_build is None else round(t_build, 1),
            "shards_built": [min(starts).isoformat(timespec="seconds"), max(ends).isoformat(timespec="seconds")],
            "index_sha256": idx_sha, "ram_bytes": ram, "store_bytes": store_bytes}
    with open(os.path.join(args.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    print("provenance:", prov["moveatlas_head"])
    print(text)


if __name__ == "__main__":
    main()
