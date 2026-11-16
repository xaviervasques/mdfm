"""The open body store (plan docs/MODEL_OPEN_PORT_PLAN.md, step M3): Anny fitted to the open motion capture of the
body store's candidate sources, train folds only, weighted by person.

Sources (plan section 2.1; reports/open_mocap/body_store_candidates.md): style100, ceti_age_kinematics, cmu_mocap
(without the second folder of its 13 byte-identical folder pairs), deepdance (a repeated take kept once),
umons_taichi, moore_perturbed_gait, push_recovery_perturbations, notredame_noncyclic, lencioni_gait_stairs, gait120
and stroke_gait_vancriekinge able-bodied. The open sources' records pass the rules of
scripts/open_mocap_body_candidates.py in its order (eval, lockbox, duplicate, possible duplicate, a canonical joint
without a landmark, dev fold), its dev folds checked against reports/open_mocap/body_store_folds.csv; gait120 and
stroke able-bodied take their train folds from mdfm.data.sources.folds, a gait120 trial being its steps joined
(mdfm.data.mocap_raw). Never a stroke patient (only the able_bodied group), never shida_pd_gait, never a dev,
lockbox, EVAL or CODY person: every candidate person goes through mdfm.train.provenance.problems (the lockbox file,
hash-checked, the tiers records' EVAL, lockbox and CODY rows, the canary) with the dev persons held out, and every
take through mdfm.data.canonical.check, before any record or raw file opens; the refusals are counted by reason.

Fit (mdfm/body/openfit.py): landmarks from the raw files (mdfm/data/open_mocap_raw.py: the 15 canonical joints of the
MoveAtlas joint maps and one toe per foot, metres, y up, root kept), the 100 Hz clock after a 12 Hz zero-phase
low-pass; per person the phenotype and the landmark offsets on up to openfit.SHAPE_FRAMES frames spread over the
person's takes (stage 1); per take the 22 rotations and the translation of every knot (stage 2). Then, as the v1
store: the floor moved to y = 0 (mdfm.body.frames.floor_height on the fitted joints), a treadmill take's belt
velocity, the contact labels in the belt frame (frames.contact_labels).

Modes:
- --measure: the fit's cost and quality on MEASURE_TAKES takes spread over the sources (two per open source, one of
  gait120 and of stroke able-bodied, distinct persons, numpy default_rng seeded by SEED), each fitted with its own
  stage 1: per take the RMS distance between fitted and observed landmarks, per landmark the same, the CPU seconds
  of stage 1 and of stage 2 per minute of motion, and the projection over the candidate hours and persons.
  Writes reports/open_body_store/fit_measure.json and fit_measure.md. The residual bound of the store is set by rule
  from these takes: the upper Tukey fence (Q3 + 1.5 IQR) of their RMS.
- --run [--part i --parts n]: the full fit, resumable: part i of n takes every n-th take of the sorted list (source,
  person, key); a person's stage 1 is computed once (a lock file) and read by every part; each person's stage 1 and
  each take's arrays land as files under <root>/parts/ (written to a temporary name, then renamed), and a file
  already there is kept. Rerun the same command to resume.
- --pack: the store from the parts: per source <root>/train/<source>/rot6d.f32 (N, 22, 6), transl.f32 (N, 3),
  contact.u1 (N, 2), takes.csv (offset and n_knots per take, the phenotype, person_weight 1 / n_persons and
  take_share = knots of the take / knots of its person: windows are drawn by person, not by hour), shard.json;
  the takes over the residual bound left out and counted; <root>/hashes.json (SHA-256 of every array and table, the
  person lists' SHA-256) and <root>/_provenance.json; for the complete store at the default root, the report
  reports/open_body_store/store.json.

Dev split (--split dev, step M4): the same rules and fit for the dev folds of the same sources (the dev persons of
reports/open_mocap/body_store_folds.csv for the open sources, the dev fold of mdfm.data.sources.folds for gait120 and
stroke able-bodied, the same healthy filter), into a separate store, default <store>/bodies_open_v1_dev
(DEV_STORE_TARGET), laid out as <root>/dev/<source>/...; the person guard then holds the train persons out, so that no
train person enters it. Its report is reports/open_body_store/store_dev.json. The training's dev hook reads it
(mdfm/data/open_body_store.OpenTrainStore(split='dev')). Lockbox, EVAL, CODY and shida_pd_gait rows are refused as
for train; a stroke patient never enters.

G1 split (--split g1, step M4; docs/rules/G1_RULE_OPEN_2026-11-13.md section 2): the truth bodies of G1's open sets
that no other store holds, the stroke_gait_vancriekinge patients outside the lockbox (index group stroke, lockbox
False; 40 persons), fitted by the same rule into cache/g1/g1_open_v1/bodies (G1_STORE_TARGET), laid out as
<root>/g1/<source>/...; their takes never train (no sampler reads this root), the open sources are not read, and a
lockbox row is refused before anything opens. The gait120 half of G1 reads the dev store (the dev fold).

Root: --root, else MDFM_OPEN_BODY_STORE, else <store>/bodies_open_v1 (STORE_TARGET); a root inside MoveAtlas or
inside this tree outside cache/ is refused. No SMPL+H array is read or written; Anny through the adapter's guard
(topology and rig 'anny', its cache pinned to <store>/anny_cache).

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_body_store.py --measure
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_body_store.py --run --part 0 --parts 3
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_body_store.py --pack
"""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))

STORE_ENV = "MDFM_OPEN_BODY_STORE"
STORE_TARGET = "<store>/bodies_open_v1"
VERSION = "bodies_open_v1"
OUT = os.path.join(TREE, "reports", "open_body_store")
SEED = 20261112
MEASURE_TAKES = 20
THREADS = 4
SHAPE_TAKES = 24          # [design] takes read for a person's stage 1
LOCK_STALE_S = 3600       # a stage-1 lock older than this is taken as left by a stopped part
OPEN_TRAIN = ("style100", "ceti_age_kinematics", "cmu_mocap", "deepdance", "umons_taichi", "moore_perturbed_gait",
              "push_recovery_perturbations", "notredame_noncyclic", "lencioni_gait_stairs")
STAGE1 = ("gait120", "stroke_gait_vancriekinge")
NEVER = ("shida_pd_gait",)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def person_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def git_head(root=TREE, extra=()):
    r = subprocess.run(["git", *extra, "-C", root, "log", "-1", "--format=%h"], capture_output=True, text=True)
    return r.stdout.strip()


DEV_STORE_TARGET = "<store>/bodies_open_v1_dev"
G1_STORE_TARGET = os.path.join(TREE, "cache", "g1", "g1_open_v1", "bodies").replace(chr(92), "/")
SPLITS = ("train", "dev", "g1")


def store_root(arg=None, split="train"):
    default = {"train": STORE_TARGET, "dev": DEV_STORE_TARGET, "g1": G1_STORE_TARGET}[split]
    env = os.environ.get({"train": STORE_ENV, "dev": STORE_ENV + "_DEV", "g1": STORE_ENV + "_G1"}[split])
    root = (arg or env or default).replace("\\", "/").rstrip("/")
    a = os.path.normcase(os.path.abspath(root))
    ma = os.path.normcase(os.path.abspath(os.environ.get("MOVEATLAS_ROOT", "<moveatlas>")))
    tree = os.path.normcase(os.path.abspath(TREE))
    if a.startswith(ma) or (a.startswith(tree) and not a.startswith(os.path.join(tree, "cache"))):
        sys.exit(f"store root {root} refused: inside MoveAtlas or inside this tree outside cache/")
    return root


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


# ---------------------------------------------------------------- candidates and refusals
def candidates(split="train"):
    """(takes, refusals, info): takes [{source, subject_ns, person_id, key, fps, n_frames, tier, lockbox, group,
    hours}] of the candidates of `split` ('train', or 'dev': the dev folds, module docstring) that pass every rule and
    guard; refusals Counter of (source, reason) in takes and
    persons; info per source. No record is opened but the joint_mask members of the open sources' containers
    (scripts/open_mocap_body_candidates.joint_masks)."""
    import open_mocap_body_candidates as OC
    from mdfm.data import canonical, sources as S
    from mdfm.train import provenance as PV
    rec_path = os.path.join(TREE, "cache", "manifest", "manifest_records.csv.gz")
    with open(os.path.join(TREE, "reports", "manifest_tiers", "hashes.json"), encoding="utf-8") as f:
        hashes = json.load(f)
    if sha256_file(rec_path) != hashes["records_file_sha256"]:
        sys.exit(f"{rec_path} does not hash to records_file_sha256 of hashes.json")
    with gzip.open(rec_path, "rt", encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    with open(os.path.join(TREE, "reports", "open_mocap", "cmu_duplicate_folders.csv"), encoding="utf-8",
              newline="") as f:
        dup_ns = {r["subject_ns_duplicate"] for r in csv.DictReader(f)}
    with open(os.path.join(TREE, "reports", "open_mocap", "body_store_folds.csv"), encoding="utf-8", newline="") as f:
        fold_csv = {(r["source"], r["person_id"]): r["fold"] for r in csv.DictReader(f)}
    ma = os.environ.get("MOVEATLAS_ROOT", "<moveatlas>")
    refused = collections.Counter()
    if split not in SPLITS:
        sys.exit(f"split {split!r}, expected one of {SPLITS}")
    takes, info, dev_all = [], {}, collections.defaultdict(set)
    train_all = collections.defaultdict(set)
    for s in NEVER:
        n = sum(1 for r in records if r["source"] == s)
        refused[(s, "never (EVAL source), not opened")] += n
    for s in (OPEN_TRAIN if split != "g1" else ()):
        rows = [r for r in records if r["source"] == s]
        persons = sorted({r["person_id"] for r in rows})
        dev = OC.draw_dev(s, persons) if len(persons) > 1 else set()
        for p in persons:
            if fold_csv.get((s, p)) != ("dev" if p in dev else "train"):
                sys.exit(f"{s} {p}: fold differs from reports/open_mocap/body_store_folds.csv")
        dev_all[s] |= dev
        masks = OC.joint_masks(ma, s)
        for r in rows:
            take, _, occ = r["key"].partition("#")
            if r["tier"] != "RELEASE":
                refused[(s, f"tier {r['tier']}")] += 1
                continue
            if r["lockbox"] == "True":
                refused[(s, "lockbox row")] += 1
                continue
            if r["subject_ns"] in dup_ns:
                refused[(s, "duplicate folder")] += 1
                continue
            if occ:
                refused[(s, "possible duplicate")] += 1
                continue
            found = masks.get((r["subject_ns"], take), [])
            if not found or not found[0].all():
                refused[(s, "canonical joint without a landmark")] += 1
                continue
            if (r["person_id"] in dev) != (split == "dev"):
                refused[(s, "dev fold" if split == "train" else "train fold")] += 1
                if split == "dev":
                    train_all[s].add(r["person_id"])
                continue
            takes.append(dict(r))
    specs = {n: S.SOURCES[n] for n in (STAGE1 if split != "g1" else ("stroke_gait_vancriekinge",))}
    fr = S.folds(specs=specs)
    idx = canonical.index()
    for n, spec in specs.items():
        fold = dict(zip(fr.loc[fr["source"] == n, "person_id"], fr.loc[fr["source"] == n, "fold"]))
        rows = idx[(idx["source"] == n) & (idx["modality"] == "pose")].to_dict("records")
        seen = set()
        for r in rows:
            r = {k: (str(v) if v is not None else "") for k, v in r.items()}
            if split == "g1":
                if r.get("group") != "stroke":
                    refused[(n, f"group {r.get('group')} (G1's stroke half reads the patients)")] += 1
                    continue
            elif spec.groups is not None and r.get("group") not in spec.groups:
                refused[(n, f"group {r.get('group')} (not able-bodied)")] += 1
                continue
            if r.get("tier") != "RELEASE":
                refused[(n, f"tier {r.get('tier')}")] += 1
                continue
            if str(r.get("lockbox")) == "True":
                refused[(n, "lockbox row")] += 1
                continue
            f = fold.get(r["person_id"])
            if split == "g1":
                if f is not None:
                    refused[(n, f"fold {f} (a person of the stage-1 folds)")] += 1
                    continue
            elif f != split:
                refused[(n, f"fold {f}")] += 1
                if f == "dev":
                    dev_all[n].add(r["person_id"])
                if f == "train":
                    train_all[n].add(r["person_id"])
                continue
            key = r["key"]
            if n == "gait120":
                key = re.sub(r"_step\d+$", "", key)
            if (r["subject_ns"], key) in seen:
                continue
            seen.add((r["subject_ns"], key))
            takes.append({"source": n, "subject_ns": r["subject_ns"], "person_id": r["person_id"], "key": key,
                          "fps": r.get("fps", ""), "n_frames": r.get("n_frames", ""), "tier": r["tier"],
                          "lockbox": r.get("lockbox", "False"), "group": r.get("group", "")})
    # the person guard, before any record opens
    PV.canary_check()
    persons = collections.defaultdict(set)
    for t in takes:
        persons[t["source"]].add(t["person_id"])
    held = dev_all if split == "train" else train_all      # the other split's persons never enter this one
    bad = PV.problems({s: sorted(v) for s, v in persons.items()}, held_out={s: sorted(v) for s, v in held.items()})
    bad_ids = {}
    for b in bad:
        src, _, rest = b.partition(" ")
        pid, _, why = rest.partition(": ")
        bad_ids[(src, pid)] = why
    kept = []
    for t in takes:
        why = bad_ids.get((t["source"], t["person_id"]))
        if why is not None:
            refused[(t["source"], f"person guard ({why})")] += 1
            continue
        try:
            canonical.check(t)
        except canonical.Refused as e:
            refused[(t["source"], f"canonical.check: {str(e).split(':')[0]}")] += 1
            continue
        kept.append(t)
    for s in OPEN_TRAIN + STAGE1:
        sel = [t for t in kept if t["source"] == s]
        info[s] = {"takes": len(sel), "persons": len({t["person_id"] for t in sel}),
                   "manifest_hours": sum(int(t["n_frames"]) / float(t["fps"]) for t in sel if t["fps"] and
                                         t["n_frames"] and s not in STAGE1) / 3600.0}
    return kept, refused, info, {"records_sha256": hashes["records_sha256"], "records_file_sha256":
                                 hashes["records_file_sha256"], "guard_problems": len(bad)}


# ---------------------------------------------------------------- one take
def landmarks(take):
    from mdfm.body import openfit as OF
    from mdfm.data import open_mocap_raw as OR
    d = OR.load_take(take)
    x = np.concatenate([d["landmarks"], d["toes"]], 1)
    x, masked = OF.clean(x)
    xc, valid = OF.to_body_clock(x, d["fps"])
    with np.errstate(all="ignore"):
        pn = float(np.nanmedian(np.linalg.norm(x[:, 1] - x[:, 0], axis=-1)))
    return xc, valid, {"fps_native": d["fps"], "n_native": int(x.shape[0]), "up_axis": d["up_axis"],
                       "pelvis_neck_m": pn, "frame_cut": d.get("frame_cut"), **masked}


def shape_frames(items, n):
    """Up to n frames (target, valid) spread over the takes' knots where every canonical landmark is valid."""
    from mdfm.body import openfit as OF
    pool = []
    for k, (x, v) in enumerate(items):
        ok = np.flatnonzero(v[:, :15].all(1))
        pool += [(k, int(i)) for i in ok]
    if not pool:
        return None
    sel = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)]
    X = np.stack([items[k][0][i] for k, i in sel])
    V = np.stack([items[k][1][i] for k, i in sel])
    assert X.shape[1] == OF.N_LM
    return X, V


def finish(aa, tr, P):
    """rot6d (T, 22, 6) f32, transl (T, 3) f32 with the floor at y = 0, contact (T, 2) u1 and the take's fields
    (floor, treadmill, belt), by the v1 store's rules (mdfm.data.body_store.finish_take)."""
    import torch
    from mdfm.body import frames
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    J = P.double()
    floor = frames.floor_height(J)
    st = frames.treadmill_stats(J, 100.0, floor=floor)
    tread = bool(st["treadmill"])
    belt = st["belt"].double() if tread else torch.zeros(3, dtype=torch.float64)
    up = torch.tensor([0.0, float(floor), 0.0], dtype=torch.float64)
    Jb = frames.galilean_shift(J, belt, 100.0) - up
    vel = (torch.gradient(J, spacing=0.01, dim=0)[0] - belt)
    contact = frames.contact_labels(Jb[None], 100.0, vel=vel[None], floor=0.0)[0]
    r6 = matrix_to_rot6d(axis_angle_to_matrix(aa.double())).float().numpy()
    trn = (tr.double() - up).float().numpy()
    fields = {"floor_m": float(floor), "treadmill": int(tread), "belt_x": float(belt[0]), "belt_z": float(belt[2]),
              "foot_speed": float(st["foot_speed"]), "pelvis_speed": float(st["pelvis_speed"]),
              "contact_l": float(contact[:, 0].double().mean()), "contact_r": float(contact[:, 1].double().mean())}
    return r6, trn, contact.numpy().astype(np.uint8), fields


def hop(take):
    """True when the take's source has its hip landmark on a pelvis marker (open_mocap_raw.HIP_ON_PELVIS)."""
    from mdfm.data import open_mocap_raw as OR
    return take["source"] in OR.HIP_ON_PELVIS


def rms(d):
    d = np.asarray(d, dtype=np.float64)
    return float(np.sqrt(np.nanmean(d ** 2)))


# ---------------------------------------------------------------- measure
def measure(a):
    import torch
    from mdfm.body import openbody as BO
    from mdfm.body import openfit as OF
    from mdfm.model.prior import PhenotypePrior
    torch.set_num_threads(THREADS)
    t0 = time.time()
    takes, refused, info, hashes = candidates()
    rng = np.random.default_rng(SEED)
    pick = []
    for s in OPEN_TRAIN + STAGE1:
        sel = [t for t in takes if t["source"] == s]
        persons = sorted({t["person_id"] for t in sel})
        k = 2 if s in OPEN_TRAIN else 1
        chosen = [persons[i] for i in rng.permutation(len(persons))[:k]]
        if len(chosen) < k:                                # one-person source: two takes of that person
            chosen = chosen * k
        used = set()
        for p in chosen:
            mine = sorted((t for t in sel if t["person_id"] == p and t["key"] not in used), key=lambda t: t["key"])
            t = mine[int(rng.integers(len(mine)))]
            used.add(t["key"])
            pick.append(t)
    basis, prior = BO.load_basis(), PhenotypePrior()
    rows = []
    for t in pick:
        r = {k: t[k] for k in ("source", "subject_ns", "person_id", "key")}
        try:
            a0 = time.process_time()
            w0 = time.time()
            x, v, meta = landmarks(t)
            r["load_s"] = time.time() - w0
            sf = shape_frames([(x, v)], OF.SHAPE_FRAMES)
            if sf is None:
                r["error"] = "no frame with every canonical landmark"
                rows.append(r)
                continue
            c1, w1 = time.process_time(), time.time()
            u, o, _ = OF.fit_person(sf, basis, prior, hip_on_pelvis=hop(t))
            c2, w2 = time.process_time(), time.time()
            aa, tr = OF.fit_take(x, v, u, o, basis, prior, hip_on_pelvis=hop(t))
            c3, w3 = time.process_time(), time.time()
            d, P, rest = OF.residuals(x, v, aa, tr, u, o, basis, prior, hip_on_pelvis=hop(t))
            d = d.numpy()
            minutes = x.shape[0] / 100.0 / 60.0
            r.update({"minutes": minutes, "knots": int(x.shape[0]), "valid_share": float(v.mean()),
                      "stage1_cpu_s": c2 - c1, "stage1_wall_s": w2 - w1, "stage2_cpu_s": c3 - c2,
                      "stage2_wall_s": w3 - w2, "stage2_cpu_s_per_min": (c3 - c2) / minutes,
                      "stage2_wall_s_per_min": (w3 - w2) / minutes, "rms_m": rms(d), "rms_canonical15_m": rms(d[:, :15]),
                      "rms_toes_m": rms(d[:, 15:]), "p95_m": float(np.nanpercentile(d, 95)),
                      "rms_by_landmark_m": {n: rms(d[:, i]) for i, (n, _, _) in enumerate(OF.LANDMARKS)},
                      "phenotype": dict(zip(BO.PHENOTYPE_KEYS, map(float, BO.to_phenotype(u)))),
                      "offsets_norm_m": {n: float(np.linalg.norm(o[i].numpy())) for i, (n, _, _) in enumerate(OF.LANDMARKS)},
                      "rest_stature_proxy_m": float(rest[15, 1] - min(rest[10, 1], rest[11, 1])), **meta})
            fin = finish(aa, tr, P)
            r.update({k: fin[3][k] for k in ("treadmill", "contact_l", "contact_r", "floor_m")})
            r["total_cpu_s"] = time.process_time() - a0
        except Exception as e:  # noqa: BLE001 (counted in the report)
            r["error"] = f"{type(e).__name__}: {e}"
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("source", "key", "minutes", "rms_m", "stage2_cpu_s_per_min",
                                                "error")}), flush=True)
    ok = [r for r in rows if "error" not in r]
    rr = np.array([r["rms_m"] for r in ok])
    q1, q3 = np.percentile(rr, [25, 75])
    bound = residual_bound(rr)
    mins = sum(r["minutes"] for r in ok)
    s2 = sum(r["stage2_cpu_s"] for r in ok) / mins
    s2w = sum(r["stage2_wall_s"] for r in ok) / mins
    s1 = float(np.median([r["stage1_cpu_s"] for r in ok]))
    s1w = float(np.median([r["stage1_wall_s"] for r in ok]))
    load_per_min = sum(r["load_s"] for r in ok) / mins
    proj = project(ok, info)
    out = {"script": "scripts/build_open_body_store.py --measure", "run_at": datetime.datetime.now().isoformat(
           timespec="seconds"), "mdfm_head": git_head(), "threads": THREADS, "seed": SEED, "takes": rows,
           "n_ok": len(ok), "n_error": len(rows) - len(ok), "minutes_fitted": mins,
           "rms_m": {"median": float(np.median(rr)), "q1": float(q1), "q3": float(q3), "max": float(rr.max()),
                     "pooled": rms(np.concatenate([np.full(1, r["rms_m"]) for r in ok]))},
           "residual_bound_m": bound, "residual_bound_rule": BOUND_RULE,
           "tukey_fence_m": float(q3 + 1.5 * (q3 - q1)),
           "stage2_cpu_s_per_min": s2, "stage2_wall_s_per_min": s2w, "stage1_cpu_s_median": s1,
           "stage1_wall_s_median": s1w, "load_wall_s_per_min": load_per_min,
           "projection": proj,
           "candidates": {"by_source": info, "refused": {f"{s} | {why}": n for (s, why), n in sorted(refused.items())},
                          **hashes},
           "fit_constants": OF.design_constants(), "licence": BO.licence_record(),
           "basis_sha256": BO.file_sha256(BO.BASIS_VALUES), "seconds": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fit_measure.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    write_measure_md(out)
    write_report_provenance(out, "--measure")
    print(json.dumps({k: out[k] for k in ("rms_m", "residual_bound_m", "stage2_cpu_s_per_min", "stage1_cpu_s_median",
                                          "projection")}, indent=1))


def write_report_provenance(m, mode):
    """reports/open_body_store/_provenance.json: the script, the mode, the MD_FM and MoveAtlas heads, the inputs'
    hashes (records, basis, licence record)."""
    prov = {"script": f"scripts/build_open_body_store.py {mode}", "run_at": datetime.datetime.now().isoformat(
            timespec="seconds"), "measure_run_at": m.get("run_at"), "mdfm_head_of_measure": m.get("mdfm_head"),
            "mdfm_head": git_head(), "moveatlas_head": git_head(os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                                                                   ("-c", "safe.directory=<moveatlas>")),
            "records_sha256": m["candidates"].get("records_sha256"), "basis_sha256": m.get("basis_sha256"),
            "outputs": ["fit_measure.json", "fit_measure.md"]}
    with open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)


def write_measure_md(m):
    L = ["# Open body store: the fit measured on 20 takes (step M3)", "",
         f"scripts/build_open_body_store.py --measure, MD_FM {m['mdfm_head']}, {m['run_at']}, {m['threads']} CPU threads "
         f"in one process, seed {m['seed']}. Fit: mdfm/body/openfit.py (constants in fit_measure.json). Each take "
         "fitted with its own stage 1 (phenotype and landmark offsets on up to 400 of its frames), then stage 2 "
         "on every knot of the 100 Hz clock.", "",
         "| source | key | minutes | RMS mm | canonical 15 mm | toes mm | p95 mm | stage 2 CPU s per min | treadmill |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in m["takes"]:
        if "error" in r:
            L.append(f"| {r['source']} | {r['key']} | | error: {r['error'][:80]} | | | | | |")
            continue
        L.append(f"| {r['source']} | {r['key']} | {r['minutes']:.2f} | {1000 * r['rms_m']:.1f} | "
                 f"{1000 * r['rms_canonical15_m']:.1f} | {1000 * r['rms_toes_m']:.1f} | {1000 * r['p95_m']:.1f} | "
                 f"{r['stage2_cpu_s_per_min']:.0f} | {r['treadmill']} |")
    ok = [r for r in m["takes"] if "error" not in r]
    names = list(ok[0]["rms_by_landmark_m"]) if ok else []
    L += ["", "RMS by landmark over the takes (mm, the median of the takes' values): " + ", ".join(
        f"{n} {1000 * float(np.nanmedian([r['rms_by_landmark_m'][n] for r in ok])):.1f}" for n in names) +
        " (the toes: the takes with toe markers)", "",
        f"Takes fitted: {m['n_ok']} ({m['n_error']} errors), {m['minutes_fitted']:.1f} minutes of motion. RMS per take: "
        f"median {1000 * m['rms_m']['median']:.1f} mm, Q1 {1000 * m['rms_m']['q1']:.1f}, Q3 {1000 * m['rms_m']['q3']:.1f}, "
        f"max {1000 * m['rms_m']['max']:.1f} mm. Residual bound of the store ({m['residual_bound_rule']}): "
        f"{1000 * m['residual_bound_m']:.1f} mm (the upper Tukey fence would be {1000 * m.get('tukey_fence_m', 0):.1f} mm; "
        "the reason for the rule: build_open_body_store.residual_bound).", "",
        f"Cost: stage 2 {m['stage2_cpu_s_per_min']:.0f} CPU s ({m['stage2_wall_s_per_min']:.0f} wall s) per minute of "
        f"motion; stage 1 {m['stage1_cpu_s_median']:.0f} CPU s per person (median); reading the raw files "
        f"{m['load_wall_s_per_min']:.1f} wall s per minute.", "",
        f"Projection over the candidates ({m['projection']['candidate_hours']:.2f} h, "
        f"{m['projection']['candidate_persons_after_guard']} persons after the guard), by the cost model "
        f"stage 2 CPU s = {m['projection']['a_cpu_s_per_take']:.1f} per take + "
        f"{m['projection']['b_cpu_s_per_minute']:.1f} per minute of motion (non-negative least squares over the measured "
        f"takes), "
        f"stage 1 {m['projection']['s1_cpu_s_per_person']:.0f} s per person: "
        f"{m['projection']['cpu_hours_one_process']:.1f} CPU hours of one {m['threads']}-thread process, plus "
        f"{m['projection']['read_wall_hours']:.1f} wall hours of reading. The CPU was shared with the four parts of the "
        "full run and other sessions while the takes were measured, so the seconds are those of a loaded machine. "
        "By source (CPU hours): " + ", ".join(
            f"{s} {v['cpu_hours']:.1f}" for s, v in m["projection"]["by_source"].items()) + ".", "",
        "Refusals before any record opened (source | reason: records):", ""]
    for k, n in m["candidates"]["refused"].items():
        L.append(f"- {k}: {n}")
    text = "\n".join(L) + "\n"
    if "\u2014" in text:
        sys.exit("em-dash in fit_measure.md")
    with open(os.path.join(OUT, "fit_measure.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


BOUND_RULE = "3 x the median RMS of the measured takes"


def residual_bound(rr):
    """The store's residual bound from the measured takes' RMS (BOUND_RULE). Decided on 2026-11-13: the upper Tukey
    fence of the 20 takes (Q3 + 1.5 IQR, 8.8 mm) mixes marker and suit captures and refused 57 percent of the first
    293 ceti_age_kinematics takes of the run, whose residual (median 9.9 mm) is the inertial suit's skeleton estimate
    and not a failed fit; three times the median keeps every capture's typical take and refuses the failed fits
    (CMU 088_05, an acrobatic take, 27 mm)."""
    return float(3.0 * np.median(np.asarray(rr, dtype=np.float64)))


def project(ok, info):
    """The full fit's cost from the measured takes: stage 2 CPU seconds = a + b x minutes per take (non-negative
    least squares over the takes: a fixed cost per take, the iterations of both passes, and a cost per minute of motion), stage 1
    the median CPU seconds per person, reading the raw files the measured wall seconds per minute; applied per source
    to its takes, hours and persons after the guard (open sources: n_frames / fps of the tiers file; gait120 and
    stroke able-bodied: their train hours of reports/open_mocap/body_store_candidates.json)."""
    with open(os.path.join(TREE, "reports", "open_mocap", "body_store_candidates.json"), encoding="utf-8") as f:
        cand = json.load(f)
    mins = np.array([r["minutes"] for r in ok])
    cpu = np.array([r["stage2_cpu_s"] for r in ok])
    from scipy.optimize import nnls
    A = np.stack([np.ones_like(mins), mins], 1)
    (a, b), _ = nnls(A, cpu)                    # non-negative: a fixed cost per take, a cost per minute
    s1 = float(np.median([r["stage1_cpu_s"] for r in ok]))
    load = sum(r["load_s"] for r in ok) / mins.sum()
    by, tot_cpu, tot_h, tot_p = {}, 0.0, 0.0, 0
    for s, i in info.items():
        h = cand["pilot"][s]["train_hours"] if s in cand.get("pilot", {}) else i["manifest_hours"]
        c = i["takes"] * a + h * 60 * b + i["persons"] * s1
        by[s] = {"takes": i["takes"], "hours": h, "persons": i["persons"], "cpu_hours": c / 3600.0,
                 "read_wall_hours": h * 60 * load / 3600.0}
        tot_cpu += c
        tot_h += h
        tot_p += i["persons"]
    return {"cost_model": "stage 2 CPU s = a + b x minutes per take; stage 1 s1 per person",
            "a_cpu_s_per_take": float(a), "b_cpu_s_per_minute": float(b), "s1_cpu_s_per_person": s1,
            "load_wall_s_per_minute": float(load), "by_source": by, "candidate_hours": tot_h,
            "candidate_persons_after_guard": tot_p, "cpu_hours_one_process": tot_cpu / 3600.0,
            "read_wall_hours": sum(v["read_wall_hours"] for v in by.values())}


def rewrite_report():
    """--report: the projection and fit_measure.md again from fit_measure.json's takes (no fit is rerun)."""
    with open(os.path.join(OUT, "fit_measure.json"), encoding="utf-8") as f:
        m = json.load(f)
    ok = [r for r in m["takes"] if "error" not in r]
    m["projection"] = project(ok, m["candidates"]["by_source"])
    rr = np.array([r["rms_m"] for r in ok])
    m["residual_bound_m"], m["residual_bound_rule"] = residual_bound(rr), BOUND_RULE
    q1, q3 = np.percentile(rr, [25, 75])
    m["tukey_fence_m"] = float(q3 + 1.5 * (q3 - q1))
    with open(os.path.join(OUT, "fit_measure.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=1)
    write_measure_md(m)
    write_report_provenance(m, "--report")
    print(json.dumps(m["projection"], indent=1))


# ---------------------------------------------------------------- run (resumable)
def _atomic_save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run(a):
    import torch
    from mdfm.body import openbody as BO
    from mdfm.body import openfit as OF
    from mdfm.model.prior import PhenotypePrior
    torch.set_num_threads(a.threads)
    root = store_root(a.root, a.split)
    parts = os.path.join(root, "parts")
    os.makedirs(parts, exist_ok=True)
    takes, refused, info, hashes = candidates(a.split)
    by_person = collections.defaultdict(list)
    for t in takes:
        by_person[(t["source"], t["person_id"])].append(t)
    # part i of n takes every n-th take of the sorted list (source, person, key): one performer's 22 h spread over
    # every part; a person's stage 1 is computed once, by the first part that needs it (a lock file), and read by
    # the others
    ordered = [t for k in sorted(by_person) for t in sorted(by_person[k], key=lambda t: t["key"])]
    mine_set = {(t["source"], t["person_id"], t["key"]) for t in ordered[a.part::a.parts]}
    keys = [k for k in sorted(by_person) if any((k[0], k[1], t["key"]) in mine_set for t in by_person[k])]
    basis, prior = BO.load_basis(), PhenotypePrior()
    log = open(os.path.join(root, f"run_part{a.part}of{a.parts}.log"), "a", encoding="utf-8", newline="\n")
    print(f"part {a.part}/{a.parts}: {len(mine_set)} of {len(ordered)} takes, {len(keys)} persons", file=log, flush=True)
    for src, pid in keys:
        pdir = os.path.join(parts, src, safe(pid))
        os.makedirs(pdir, exist_ok=True)
        todo = [t for t in sorted(by_person[(src, pid)], key=lambda t: t["key"])
                if (src, pid, t["key"]) in mine_set
                and not os.path.exists(os.path.join(pdir, safe(t["key"]) + ".npz"))
                and not os.path.exists(os.path.join(pdir, safe(t["key"]) + ".error.json"))]
        if not todo:
            continue
        def load(t):
            try:
                return landmarks(t)
            except Exception as e:  # noqa: BLE001 (recorded per take)
                with open(os.path.join(pdir, safe(t["key"]) + ".error.json"), "w", encoding="utf-8",
                          newline="\n") as f:
                    json.dump({"take": t, "error": f"{type(e).__name__}: {e}"}, f)
                return None
        mine_takes = sorted(by_person[(src, pid)], key=lambda t: t["key"])
        pfile = os.path.join(pdir, "person.json")
        lock = pfile + ".lock"
        mine_lock = False
        while not os.path.exists(pfile):
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                mine_lock = True
                break
            except FileExistsError:
                if time.time() - os.path.getmtime(lock) > LOCK_STALE_S:     # a part stopped while it held it
                    os.remove(lock)
                    continue
                time.sleep(5)
        if os.path.exists(pfile):
            if mine_lock:
                os.remove(lock)
            with open(pfile, encoding="utf-8") as f:
                pj = json.load(f)
            u, o = torch.tensor(pj["u"]), torch.tensor(pj["o"])
        else:
            # stage 1 reads at most SHAPE_TAKES takes, evenly spread over the person's sorted takes (memory: one
            # performer holds 810 takes)
            pick = [mine_takes[int(i)] for i in np.unique(np.linspace(0, len(mine_takes) - 1,
                                                                       min(SHAPE_TAKES, len(mine_takes))).astype(int))]
            items = [r for r in (load(t) for t in pick) if r is not None]
            sf = shape_frames([(x, v) for x, v, _ in items], OF.SHAPE_FRAMES)
            del items
            if sf is None:
                print(f"{src} {pid}: no frame for stage 1", file=log, flush=True)
                os.remove(lock)
                continue
            c0 = time.process_time()
            u, o, _ = OF.fit_person(sf, basis, prior, hip_on_pelvis=hop({"source": src}))
            pj = {"source": src, "person_id": pid, "u": u.tolist(), "o": o.tolist(),
                  "phenotype": dict(zip(BO.PHENOTYPE_KEYS, map(float, BO.to_phenotype(u)))),
                  "frames": int(sf[0].shape[0]), "cpu_s": time.process_time() - c0,
                  "basis_sha256": BO.file_sha256(BO.BASIS_VALUES), "fit_constants": OF.design_constants()}
            tmp = pfile + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(pj, f)
            os.replace(tmp, pfile)
            os.remove(lock)
        for t in todo:
            got = load(t)
            if got is None:
                continue
            x, v, meta = got
            c0 = time.process_time()
            try:   # the fit, its residuals and the frames' reads (a take of one knot has no velocity)
                aa, tr = OF.fit_take(x, v, u, o, basis, prior, hip_on_pelvis=hop(t))
                d, P, _ = OF.residuals(x, v, aa, tr, u, o, basis, prior, hip_on_pelvis=hop(t))
                r6, trn, contact, fields = finish(aa, tr, P)
            except Exception as e:  # noqa: BLE001 (recorded per take: a part never stops on one take)
                with open(os.path.join(pdir, safe(t["key"]) + ".error.json"), "w", encoding="utf-8",
                          newline="\n") as f:
                    json.dump({"take": t, "n_frames_read": int(x.shape[0]),
                               "error": f"fit: {type(e).__name__}: {e}"}, f)
                print(json.dumps({"source": src, "person_id": pid, "key": t["key"], "error": type(e).__name__}),
                      file=log, flush=True)
                continue
            d = d.numpy()
            info_t = {**{k: t[k] for k in ("source", "subject_ns", "person_id", "key", "fps", "n_frames")}, **meta,
                      **fields, "n_knots": int(r6.shape[0]), "rms_m": rms(d), "rms_canonical15_m": rms(d[:, :15]),
                      "rms_toes_m": rms(d[:, 15:]), "valid_share": float(v.mean()), "cpu_s": time.process_time() - c0}
            _atomic_save(os.path.join(pdir, safe(t["key"]) + ".npz"), rot6d=r6, transl=trn, contact=contact,
                         info=np.frombuffer(json.dumps(info_t).encode("utf-8"), dtype=np.uint8))
            print(json.dumps({k: info_t[k] for k in ("source", "person_id", "key", "n_knots", "rms_m", "cpu_s")}),
                  file=log, flush=True)
    print(f"part {a.part}/{a.parts} done", file=log, flush=True)


# ---------------------------------------------------------------- pack
def pack(a):
    from mdfm.body import openbody as BO
    root = store_root(a.root, a.split)
    with open(os.path.join(OUT, "fit_measure.json"), encoding="utf-8") as f:
        bound = json.load(f)["residual_bound_m"]
    takes, refused, info, hashes = candidates(a.split)
    want = {(t["source"], t["person_id"], t["key"]) for t in takes}
    rows, missing, over, errors = collections.defaultdict(list), [], collections.Counter(), collections.Counter()
    persons_ph = {}
    for src, pid, key in sorted(want):
        pdir = os.path.join(root, "parts", src, safe(pid))
        p = os.path.join(pdir, safe(key) + ".npz")
        if os.path.exists(os.path.join(pdir, safe(key) + ".error.json")):
            errors[src] += 1
            continue
        if not os.path.exists(p):
            missing.append((src, pid, key))
            continue
        with np.load(p) as z:
            inf = json.loads(bytes(z["info"]).decode("utf-8"))
            if inf["rms_m"] > bound:
                over[src] += 1
                continue
            rows[src].append((inf, p, None, None))     # the arrays are read again at writing: one take in memory
        if (src, pid) not in persons_ph:
            with open(os.path.join(pdir, "person.json"), encoding="utf-8") as f:
                persons_ph[(src, pid)] = json.load(f)["phenotype"]
    if missing and not a.allow_missing:
        sys.exit(f"{len(missing)} takes not fitted yet (first {missing[:3]}): resume --run, or pass --allow-missing")
    n_persons = len({(s, i["person_id"]) for s, v in rows.items() for i, _, _, _ in v})
    person_knots = collections.Counter()
    for s, v in rows.items():
        for i, _, _, _ in v:
            person_knots[(s, i["person_id"])] += i["n_knots"]
    out_hashes, lists = {}, {}
    for s, v in sorted(rows.items()):
        d = os.path.join(root, a.split, s)
        os.makedirs(d, exist_ok=True)
        # streamed take by take, in the takes' order: the same bytes as one concatenation, one take in memory
        names = (("rot6d.f32", "rot6d", "<f4"), ("transl.f32", "transl", "<f4"), ("contact.u1", "contact", "u1"))
        outs = {name: open(os.path.join(d, name), "wb") for name, _, _ in names}
        try:
            for _, path, _, _ in v:
                with np.load(path) as z:
                    for name, field, dt in names:
                        np.ascontiguousarray(z[field].astype(dt)).tofile(outs[name])
        finally:
            for f in outs.values():
                f.close()
        for name, _, _ in names:
            out_hashes[f"{a.split}/{s}/{name}"] = sha256_file(os.path.join(d, name))
        cols = ["take_id", "source", "subject_ns", "person_id", "key", *BO.PHENOTYPE_KEYS, "fps_native", "n_native",
                "offset", "n_knots", "person_weight", "take_share", "treadmill", "belt_x", "belt_z", "floor_m",
                "contact_l", "contact_r", "rms_m", "up_axis"]
        off = 0
        with open(os.path.join(d, "takes.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(cols)
            for i, _, _, _ in v:
                ph = persons_ph[(s, i["person_id"])]
                w.writerow([f"{s}/{i['subject_ns']}/{i['key']}", s, i["subject_ns"], i["person_id"], i["key"],
                            *[repr(ph[k]) for k in BO.PHENOTYPE_KEYS], repr(i["fps_native"]), i["n_native"], off,
                            i["n_knots"], repr(1.0 / n_persons), repr(i["n_knots"] / person_knots[(s, i["person_id"])]),
                            i["treadmill"], repr(i["belt_x"]), repr(i["belt_z"]), repr(i["floor_m"]),
                            repr(i["contact_l"]), repr(i["contact_r"]), repr(i["rms_m"]), i["up_axis"]])
                off += i["n_knots"]
        out_hashes[f"{a.split}/{s}/takes.csv"] = sha256_file(os.path.join(d, "takes.csv"))
        lists[s] = sorted({i["person_id"] for i, _, _, _ in v})
        with open(os.path.join(d, "shard.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"version": VERSION, "source": s, "knots": off, "rate": 100.0, "world": "y up, metres, floor "
                       "at y = 0 per take; treadmill takes in the lab frame, belt shift on read"}, f, indent=1)
    doc = {"version": VERSION, "split": a.split, "files": out_hashes, "person_lists_sha256": {s: person_hash(v) for s, v in lists.items()},
           "persons": {s: len(v) for s, v in lists.items()}, "residual_bound_m": bound,
           "weighting": "by person: person_weight 1 / n_persons, take_share the take's knots over its person's",
           "basis_sha256": BO.file_sha256(BO.BASIS_VALUES), "licence": BO.licence_record(), **hashes}
    with open(os.path.join(root, "hashes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    prov = {"script": "scripts/build_open_body_store.py --pack" + (f" --split {a.split}" if a.split != "train" else ""),
            "mdfm_head": git_head(),
            "moveatlas_head": git_head(os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                                       ("-c", "safe.directory=<moveatlas>")),
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "missing": len(missing),
            "over_bound": dict(over), "errors": dict(errors)}
    with open(os.path.join(root, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    if root == store_root(None, a.split) and not missing:     # the report describes the complete store only
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, {"train": "store.json", "dev": "store_dev.json", "g1": "store_g1.json"}[a.split]), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"root": root, **doc, **prov}, f, indent=1)
    print(json.dumps({"persons": doc["persons"], "missing": len(missing), "over": dict(over), "errors": dict(errors)}))


def recheck_bound(a):
    """--recheck-bound: the residual bound read again on the store's current fit (after the pelvis-marker hips of
    2026-11-13): the RMS of the 20 measured takes as their parts now hold them, three times their median against
    the bound of fit_measure.json, and the takes over each bound by source over every part. Writes
    reports/open_body_store/bound_recheck.json and .md."""
    root = store_root(a.root)
    with open(os.path.join(OUT, "fit_measure.json"), encoding="utf-8") as f:
        m = json.load(f)
    cur = []
    for r in m["takes"]:
        p = os.path.join(root, "parts", r["source"], safe(r["person_id"]), safe(r["key"]) + ".npz")
        with np.load(p) as z:
            cur.append(json.loads(bytes(z["info"]).decode("utf-8"))["rms_m"])
    cur = np.array(cur)
    new_bound = residual_bound(cur)
    old_bound = float(m["residual_bound_m"])
    over = collections.defaultdict(lambda: [0, 0, 0])
    import glob
    for p in glob.glob(os.path.join(root, "parts", "*", "*", "*.npz")):
        if os.path.basename(p).startswith("._"):
            continue
        with np.load(p) as z:
            inf = json.loads(bytes(z["info"]).decode("utf-8"))
        o = over[inf["source"]]
        o[0] += 1
        o[1] += int(inf["rms_m"] > old_bound)
        o[2] += int(inf["rms_m"] > new_bound)
    out = {"script": "scripts/build_open_body_store.py --recheck-bound", "mdfm_head": git_head(),
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "measured_takes": len(cur),
           "rms_measured_m": [float(x) for x in m["takes_rms"]] if "takes_rms" in m else
           [float(r["rms_m"]) for r in m["takes"]], "rms_now_m": cur.tolist(),
           "median_now_m": float(np.median(cur)), "bound_now_m": new_bound, "bound_kept_m": old_bound,
           "by_source": {s: {"takes": v[0], "over_kept_bound": v[1], "over_bound_now": v[2]} for s, v in sorted(over.items())}}
    with open(os.path.join(OUT, "bound_recheck.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    L = ["# The residual bound read again on the current fit", "",
         f"scripts/build_open_body_store.py --recheck-bound, MD_FM {out['mdfm_head']}: the 20 takes of fit_measure.md "
         f"as the store's parts now hold them: median RMS {1000 * out['median_now_m']:.1f} mm (at the measure "
         f"{1000 * float(np.median(out['rms_measured_m'])):.1f} mm), three times it {1000 * new_bound:.1f} mm against the "
         f"bound kept, {1000 * old_bound:.1f} mm.", "",
         "| source | takes | over the kept bound | over the bound read now |", "|---|---|---|---|"]
    for s_, v in out["by_source"].items():
        L.append(f"| {s_} | {v['takes']} | {v['over_kept_bound']} | {v['over_bound_now']} |")
    with open(os.path.join(OUT, "bound_recheck.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--measure", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--pack", action="store_true")
    p.add_argument("--report", action="store_true", help="rewrite fit_measure's projection and .md from its takes")
    p.add_argument("--recheck-bound", action="store_true", help="the bound read again on the store's current fit")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--root", default=None)
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--split", default="train", choices=SPLITS, help="the folds fitted and packed (module docstring)")
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.measure:
        measure(a)
    elif a.run:
        run(a)
    elif a.pack:
        pack(a)
    elif a.report:
        rewrite_report()
    elif a.recheck_bound:
        recheck_bound(a)
    else:
        p.error("one of --measure, --run, --pack, --report")


if __name__ == "__main__":
    main()
