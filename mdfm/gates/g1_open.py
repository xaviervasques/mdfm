"""G1 on the open sets (docs/rules/G1_RULE_OPEN_2026-11-13.md, registered as G1 version 2 on 2026-11-14; decision P8
of docs/MODEL_OPEN_PORT_PLAN.md): the kernel round trip of mdfm/gates/g1.py on Anny bodies fitted to the markers of
held-out gait120 persons and stroke patients, in place of AMASS.

Sets. Development and calibration are the seeded halves (halves, the rule's section 3 exactly) of two pools: the
gait120 dev fold of mdfm.data.sources.folds (19 persons) and the stroke_gait_vancriekinge patients outside the lockbox
(index group stroke, lockbox False: 40 persons). The halves are drawn from the person lists only and written with their
SHA-256 to reports/gates/g1/open_halves/ by scripts/build_g1_open_inputs.py --halves before any input is built. The
lockbox (24 gait120 and 10 stroke persons of reports/lockbox/lockbox_persons.csv, lockbox_sets) is opened only by
read_lockbox, inside a logged read (section "The lockbox read" below).

Truth. The fitted takes of the open body stores (mdfm/data/open_body_store.py, the residual bound applied by the
reader): the gait120 persons from the dev store (the dev fold, scripts/build_open_body_store.py --split dev), the
stroke patients from the G1 store (--split g1, cache/g1/g1_open_v1/bodies). Per take: the stored rotations and
translation low-passed at CUTOFF_HZ (mdfm.data.prep.lowpass, zero-phase fourth-order Butterworth, as v1's
amass_to_body_clock), forward kinematics on the take's P3 rest skeleton, the floor at y = 0, the treadmill rule, the
windows of g1 with the open minimum (MIN_WINDOW_S = 1.0 s, the rule's section 4) and their frames.

Inputs. g1's renderer with the open mount priors (reports/open_mount_priors/priors.json: the mount rotations and the
device levels) and the open body's trajectory (no SMPL+H body: BodyTrajectory on the rest joints); every other input
setting is g1's (PARAMS). The input configuration names the open version, the open minimum and the open priors'
SHA-256, so its hash, and every window seed, differ from v1's.

Scoring: g1's functions unchanged (score, lines, fit_conformal), on OpenG1Set, which checks its persons against the
halves file and the lockbox. The trivial baseline is the population prior of the open train store
(open_population_prior). No AMASS row and no SMPL+H array is read.

The lockbox read (read_lockbox; the rule's "How it becomes binding" paragraph and sections 1 to 4):
1. The read declaration (load_declaration: a JSON file of DECLARATION_FORMAT with the fields DECLARATION_KEYS of the
   2026-11-12 version's section 1) chooses the rule (declared_rule): a training corpus without AMASS (no source amass,
   no AMASS archive) reads this module's RULE_PATH, registered for G1 (common.rule: refused when unregistered or
   edited), and checks the SHA-256 of the 2026-11-12 version (registered for G1 as well) and of the draft that it
   adopts (ADOPTS); a corpus with AMASS reads the 2026-11-12 version and the draft's SHA-256, and runs on g1.run,
   never here (its harness, the 2026-11-12 version's sections 1 and 2, is not part of this module).
2. Checks that need no lockbox data, all before the read is logged: the halves file equals the rule's draw
   (check_halves); the lockbox sets from the member file at its SHA-256 (lockbox_sets: 24 gait120 persons, 10 stroke
   patients); the declaration (check_declaration: its checkpoint file, SHA-256 and training lists against the
   predictor's, its corpus free of AMASS, EVAL and CODY sources, its training persons disjoint from the lockbox and
   both halves, the mode ema_s20_nopool, the halves' and the member file's SHA-256); the trivial baseline from the
   open train store's persons only (check_prior); the root inside the gates-only store. Then, on the halves (no
   lockbox data): the conformal factors of the model and of the prior fitted on the calibration half, and the dev
   rows of both on the development half (the rule's section 6). The plan of the read (read_plan: rule, declaration,
   sets, halves, conformal factors, input configuration, fit code and constants) is hashed.
3. The read is logged (common.log_lockbox_read: the plan's SHA-256 as the input hash, the declaration's SHA-256 in
   the note, the code hash of READ_CODE); a second read of the same checkpoint and rule needs reread and a reason.
4. Inside the read only (build_lockbox, refused without the logged row, check_read_row): the lockbox persons' index
   rows (lockbox_takes, the readers' lockbox flag granted to this module), the fits by the store builder's rule
   (fit_person: scripts/build_open_body_store.py's stage 1 and stage 2, the landmarks read with the lockbox flag,
   cached under the gates-only store <root>/lockbox/_fits/ with the fit provenance's SHA-256), the residual bound of
   the store (a take over it refused and counted, a person without a kept take unscored and listed), the truth and
   inputs of build_shard under <root>/lockbox/<source>, and the manifest. OpenLockboxSet opens them only with the
   read's row. The lines (g1.lines) of the model against the prior on the same windows, per source rows
   (g1.per_group by source), the secondary rows; the result (OpenLockboxRead) is returned and nothing is written to
   reports/ (scripts/read_g1_open_lockbox.py writes it).
"""
import collections
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import paths
from ..data.canonical import Refused, is_cody
from . import common
from . import g1 as G

VERSION = "g1_open_v1"
SEED_TAG = "G1open"
SOURCES = ("gait120", "stroke_gait_vancriekinge")
SPLITS = ("dev", "calib")
MIN_WINDOW_S = 1.0
MIN_KNOTS = int(round(MIN_WINDOW_S * G.RATE))
HALVES_SEED = "20260928:g1_open_halves:"
HALVES_DIR = os.path.join(paths.TREE, "reports", "gates", "g1", "open_halves")
HALVES_JSON = os.path.join(HALVES_DIR, "halves.json")
OPEN_PRIORS = os.path.join(paths.TREE, "reports", "open_mount_priors", "priors.json")
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G1_RULE_OPEN_2026-11-13.md")
INPUT_CODE = G.INPUT_CODE + ("mdfm/gates/g1_open.py", "mdfm/data/open_body_store.py", "mdfm/body/openbody.py",
                             "mdfm/body/openfit.py", "scripts/build_open_body_store.py")
STORE_OF = {"gait120": "dev", "stroke_gait_vancriekinge": "g1"}
LOCKBOX = "lockbox"
# the lockbox read (module docstring, "The lockbox read")
RULE_V1_PATH = os.path.join(paths.TREE, "docs", "rules", "G1_RULE_2026-11-12.md")
DRAFT_PATH = os.path.join(paths.TREE, "docs", "rules", "G1_RULE.md")
# what each rule adopts, by its first paragraph: (path attribute of this module, SHA-256 named there)
ADOPTS = {"open": (("RULE_V1_PATH", "c3aceff307fff29528c4b0cf90eed2a499af6a92822ba9e824a2df48ccf76da1"),
                   ("DRAFT_PATH", "f9e099219d06e0a53f1b9a8d7578800e9f4fb3c86a869786ed786c2044f0d504")),
          "v1": (("DRAFT_PATH", "f9e099219d06e0a53f1b9a8d7578800e9f4fb3c86a869786ed786c2044f0d504"),)}
AMASS_SOURCE = "amass"
NEVER_SOURCES = ("shida_pd_gait",)          # the rule's section 1, "Refused always"
LOCKBOX_CSV = os.path.join(paths.TREE, "reports", "lockbox", "lockbox_persons.csv")
LOCKBOX_SHA256 = "619dfdacef464387ab5447299534e98b152507dea9bca91c95f68e3324762162"   # lockbox v1.2 (rule section 1)
LOCKBOX_COUNTS = {"gait120": 24, "stroke_gait_vancriekinge": 10}                       # rule section 1
LOCKBOX_GROUP = {"stroke_gait_vancriekinge": "stroke"}                                  # every stroke member a patient
INFERENCE_MODE = "ema_s20_nopool"           # the 2026-11-12 version's section 2
MODE_SETTINGS = {"weights": "ema", "refine_steps": G.PARAMS["refine_steps"], "points_per_stream": G.PARAMS["refine_points"],
                 "latent_samples": G.PARAMS["latent_samples"], "decoder_draws": G.PARAMS["decoder_draws"],
                 "shape_pooling": "none"}
MODE_BATCH_WINDOWS = 16
DECLARATION_FORMAT = "mdfm-g1-read-declaration/1"
DECLARATION_KEYS = ("format", "gate", "checkpoint", "config_sha256", "body", "corpus", "train_persons",
                    "train_persons_sha256", "inference", "sets")
FIT_CODE = ("mdfm/gates/g1_open.py", "mdfm/body/openfit.py", "mdfm/body/openbody.py", "mdfm/body/frames.py",
            "mdfm/body/rotations.py", "mdfm/model/prior.py", "mdfm/data/open_mocap_raw.py", "mdfm/data/mocap_raw.py",
            "mdfm/data/open_body_store.py", "scripts/build_open_body_store.py")
READ_CODE = tuple(dict.fromkeys(G.SCORE_CODE + INPUT_CODE + FIT_CODE))


def default_root():
    return os.path.join(paths.CACHE, "g1", VERSION).replace("\\", "/")


# ---------------------------------------------------------------- the halves (rule section 3)
def pools():
    """{source: sorted person ids} of the development pools (module docstring), from the fold table and the index's
    metadata only (no record opens): gait120's dev fold, the stroke patients outside the lockbox."""
    from ..data import canonical, sources as S
    f = S.folds()
    out = {"gait120": sorted(set(f.loc[(f["source"] == "gait120") & (f["fold"] == "dev"), "person_id"]))}
    idx = canonical.index()
    r = idx[(idx["source"] == "stroke_gait_vancriekinge") & (idx["group"] == "stroke")]
    out["stroke_gait_vancriekinge"] = sorted(set(r.loc[r["lockbox"].astype(str) != "True", "person_id"]))
    for s, ids in out.items():
        if is_cody(s, *ids):
            raise Refused("a CODY name in a G1 pool")
    return out


def halves(source, persons):
    """(development, calibration) sorted lists of one source's pool by the rule's seeded draw."""
    persons = sorted(str(p) for p in persons)
    seed = int(hashlib.sha256((HALVES_SEED + source).encode("utf-8")).hexdigest()[:16], 16)
    perm = np.random.default_rng(seed).permutation(len(persons))
    n_dev = int(math.floor(len(persons) / 2 + 0.5))
    dev = sorted(persons[i] for i in perm[:n_dev])
    return dev, sorted(set(persons) - set(dev))


def draw_halves():
    """{'pools', 'halves': {split: {source: ids}}, 'sha256': {split: {source: person_hash}}} of the rule's draw."""
    from ..data.sources import person_hash
    pl = pools()
    hv = {"dev": {}, "calib": {}}
    for s in SOURCES:
        hv["dev"][s], hv["calib"][s] = halves(s, pl[s])
    return {"rule": common.rel(RULE_PATH), "seed_form": HALVES_SEED + "<source>", "pools": pl,
            "pool_sha256": {s: person_hash(v) for s, v in pl.items()}, "halves": hv,
            "sha256": {k: {s: person_hash(v) for s, v in d.items()} for k, d in hv.items()}}


def read_halves(path=HALVES_JSON):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- inputs
def input_config(priors_path=OPEN_PRIORS):
    """g1.input_config with the open version, seed tag, minimum window and mount priors."""
    cfg = G.input_config()
    cfg.update({"version": VERSION, "seed_tag": SEED_TAG, "min_window_s": MIN_WINDOW_S,
                "mount_priors": common.rel(priors_path), "mount_priors_sha256": common.sha256_file(priors_path),
                "body": "anny (mdfm/body/openbody.py, P3 basis)"})
    return cfg


def window_seed(take_id, k, config_sha256):
    h = common.sha256_bytes(f"{take_id}\n{int(k)}\n{config_sha256}\n{SEED_TAG}".encode("utf-8"))
    return int(h[:16], 16) & ((1 << 63) - 1)


def window_starts(n_knots):
    """g1.window_starts with the open minimum (MIN_KNOTS)."""
    out = []
    for s in range(0, int(n_knots), G.KNOTS):
        n = min(G.KNOTS, int(n_knots) - s)
        if n >= MIN_KNOTS:
            out.append((s, n))
    return out


class OpenRenderer(G._Renderer):
    """g1's renderer on the open body: the open mount priors and device levels, the trajectory on the take's rest
    joints (no SMPL+H body), the CPU."""

    def __init__(self, priors_path=OPEN_PRIORS):
        from ..kernels.nuisance import MountPriors
        self.priors = MountPriors.load(priors_path)
        self.R0 = self.priors.rotation(G.PARAMS["mount_family"], list(G.SITES)).double()
        self.levels = G.device_levels(self.priors)
        self.config = input_config(priors_path)
        self.config_sha256 = common.sha256_json(self.config)
        self.device = torch.device("cpu")
        self._bodies = {}

    def segment_signals(self, body, take):
        from ..body.trajectory import BodyTrajectory
        from ..factory import imu_render
        traj = BodyTrajectory(None, take["r6"], take["tr"], None, rate=G.RATE, rest=take["rest"])
        T = take["r6"].shape[0]
        idx = [imu_render.SITES.index(s) for s in G.SITES]
        parts = []
        for a in range(0, T, G.RENDER_CHUNK):
            t = torch.arange(a, min(T, a + G.RENDER_CHUNK), dtype=torch.float64) / G.RATE
            sig, _, _ = imu_render.render(traj, t, list(G.SITES))
            parts.append(sig[:, idx, :G.N_CH].double())
        return torch.cat(parts), traj


def open_take(store, i):
    """The take i of an open store on the body clock in the lab frame (g1.take_body's dict): low-passed rotations
    and translation, the P3 rest joints, forward kinematics, the floor at y = 0, the treadmill rule. None under
    MIN_KNOTS."""
    from ..body import frames
    from ..body.joints import fk
    from ..body.rotations import rot6d_to_matrix
    from ..data.prep import lowpass
    a = store.take(i)
    if len(a["transl"]) < MIN_KNOTS:
        return None
    r6 = torch.as_tensor(np.ascontiguousarray(lowpass(np.asarray(a["rot6d"], np.float64), G.RATE, G.CUTOFF_HZ)))
    tr = torch.as_tensor(np.ascontiguousarray(lowpass(np.asarray(a["transl"], np.float64), G.RATE, G.CUTOFF_HZ)))
    rest = torch.as_tensor(store.rest_joints[i], dtype=torch.float64)
    R_local = rot6d_to_matrix(r6)
    pos, R = fk(R_local, tr, rest)
    floor = float(frames.floor_height(pos))
    up = torch.tensor([0.0, floor, 0.0], dtype=torch.float64)
    pos, tr = pos - up, tr - up
    st = frames.treadmill_stats(pos, G.RATE, floor=0.0)
    belt = st["belt"].double() if st["treadmill"] else torch.zeros(3, dtype=torch.float64)
    if st["treadmill"]:
        pos, tr = frames.galilean_shift(pos, belt, G.RATE), frames.galilean_shift(tr, belt, G.RATE)
    return {"r6": r6, "tr": tr, "betas": None, "rest": rest, "pos": pos, "R": R, "fps": G.RATE, "floor_m": floor,
            "treadmill": bool(st["treadmill"]), "belt": belt}


def _store(source):
    from ..data.open_body_store import OpenTrainStore
    return OpenTrainStore(split=STORE_OF[source])


def build_shard(split, source, persons, root, render=True, log=None, store=None, read_row=None,
                registry_path=common.RULE_HASHES, log_path=common.READ_LOG):
    """The truth and inputs of one source's half under <root>/<split>/<source> (g1's shard format), every take of
    the half's persons in the source's open store (or in `store`, a FittedStore of the same contract); returns the
    shard.json dict. The split 'lockbox' is built only inside a logged read (read_row, check_read_row), from the
    read's own fits (a FittedStore of split 'lockbox'), under the gates-only store; no other caller builds it."""
    import shutil
    from ..data.windows import task_id
    log = log or (lambda *a: None)
    rule_sha = ""
    if split == LOCKBOX:
        check_read_row(read_row, registry_path, log_path)
        if store is None or store.split != LOCKBOX:
            raise Refused("the open lockbox shards are built from the read's own fits only")
        if not paths.in_gates_store(root):
            raise Refused(f"the open lockbox inputs live in the gates-only store only, not {root}")
        rule_sha = read_row["rule_sha256"]
    elif split not in SPLITS:
        raise Refused(f"the open G1 harness builds dev and calib only, not {split!r}")
    elif paths.in_gates_store(root):
        raise Refused(f"open {split} inputs never go to the gates-only store: {root}")
    elif store is not None and store.split == LOCKBOX:
        raise Refused("lockbox fits never enter a dev or calib shard")
    store = _store(source) if store is None else store
    tk = store.takes
    persons = set(persons)
    sel = [i for i in tk.sort_values("take_id", kind="stable").index if tk.at[i, "person_id"] in persons]
    missing = sorted(persons - set(tk.loc[sel, "person_id"]))
    rend = OpenRenderer()
    final = os.path.join(root, split, source)
    tmp = final + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    t0 = time.time()
    shard = G._Shard(tmp, render)
    windows, takes = [], []
    valid = {"imu": 0, "cam": 0}
    try:
        for i in sel:
            row = tk.loc[i]
            take = open_take(store, i)
            trow = {"take_id": row["take_id"], "archive": source, "subject_ns": row["subject_ns"],
                    "person_id": row["person_id"], "gender": "", "fps_native": repr(float(row["fps_native"])),
                    "fps_manifest": "", "n_frames": int(row["n_native"]), "n_knots": 0, "n_windows": 0,
                    "dropped": "short", "treadmill": 0, "belt_x": "0.0", "belt_z": "0.0", "floor_m": "",
                    "betas": " ".join(repr(float(x)) for x in store.phenotype[i])}
            if take is None:
                takes.append(trow)
                continue
            ws = window_starts(take["pos"].shape[0])
            trow.update({"n_knots": int(take["pos"].shape[0]), "n_windows": len(ws), "dropped": "" if ws else "short",
                         "treadmill": int(take["treadmill"]), "belt_x": repr(float(take["belt"][0])),
                         "belt_z": repr(float(take["belt"][2])), "floor_m": repr(take["floor_m"])})
            takes.append(trow)
            if not ws:
                continue
            seg, traj = rend.segment_signals(None, take) if render else (None, None)
            task = task_id(source, row["key"])
            for k, (s, n) in enumerate(ws):
                joints, yaw, R_yaw, origin = G.window_frame(take["pos"], take["R"], s, n)
                seed = window_seed(row["take_id"], k, rend.config_sha256)
                wrow = {"window": shard.count["window"], "take_id": row["take_id"], "archive": source,
                        "subject_ns": row["subject_ns"], "person_id": row["person_id"], "gender": "",
                        "task": task, "k": k, "start": s, "n": n, "offset": shard.count["knot"],
                        "cam_offset": shard.count["cam"], "cam_n": 0, "cam_fps": "", "cam_scale": "", "seed": str(seed)}
                shard.write("joints", joints.float().numpy())
                shard.write("root_yaw", yaw.float().numpy())
                if render:
                    gen = torch.Generator().manual_seed(seed)
                    M, bias, imu, ok = rend.window_imu(seg, s, n, gen)
                    record, cam = rend.window_camera(traj, s, n, R_yaw, origin, joints[:, G.PELVIS].mean(0), gen)
                    shard.write("imu", imu)
                    shard.write("imu_valid", ok.astype(np.uint8))
                    valid["imu"] += int(ok.sum())
                    valid["cam"] += int(record["valid_mask"].sum())
                    shard.write("mounts", M.float().numpy()[None])
                    shard.write("biases", bias.float().numpy()[None])
                    shard.write("cam", record["coords"][..., :2])
                    shard.write("cam_valid", record["valid_mask"].astype(np.uint8))
                    shard.write("cam_params", cam[None])
                    wrow.update({"cam_n": int(record["n_frames"]), "cam_fps": repr(float(record["fps"])),
                                 "cam_scale": repr(float(record["scale"]))})
                    shard.count["cam"] += int(record["n_frames"])
                shard.count["knot"] += n
                shard.count["window"] += 1
                windows.append(wrow)
    finally:
        shard.close()
    G._write_csv(os.path.join(tmp, "windows.csv"), G.WINDOW_COLUMNS, windows)
    G._write_csv(os.path.join(tmp, "takes.csv"), G.TAKE_COLUMNS, takes)
    files = {}
    for k in shard.names:
        fn = G._Shard.fname(k)
        dt, shape, per = G.FILES[k]
        files[k] = {"file": fn, "dtype": dt, "shape": [shard.count[per], *shape],
                    "bytes": os.path.getsize(os.path.join(tmp, fn)), "sha256": common.sha256_file(os.path.join(tmp, fn))}
    for name in ("windows.csv", "takes.csv"):
        files[name] = {"file": name, "sha256": common.sha256_file(os.path.join(tmp, name))}
    got = sorted({w["person_id"] for w in windows})
    meta = {"version": VERSION, "gate": G.GATE, "split": split, "archive": source, "render": bool(render),
            "n_takes": len(takes), "n_dropped_short": sum(t["dropped"] == "short" for t in takes),
            "n_windows": shard.count["window"], "n_knots": shard.count["knot"], "n_cam": shard.count["cam"],
            "n_treadmill_takes": sum(int(t["treadmill"]) for t in takes), "imu_valid": valid["imu"],
            "cam_valid": valid["cam"], "n_persons": len(got),
            "persons_sha256": common.sha256_bytes(("\n".join(got) + "\n").encode("utf-8")),
            "persons_without_take": missing, "body_store": {"split": store.split, "root": store.root,
                                                            **store.hashes()},
            "config": rend.config, "config_sha256": rend.config_sha256, "params_sha256": common.sha256_json(G.PARAMS),
            "code_sha256": G.code_sha256(INPUT_CODE), "rule_sha256": rule_sha, "requested_takes_sha256": common.sha256_bytes(
                ("\n".join(tk.loc[sel, "take_id"]) + "\n").encode("utf-8")), "files": files,
            "render_device": "cpu", "seconds": round(time.time() - t0, 1), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(tmp, "shard.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    log(f"{split}/{source}: {meta['n_windows']} windows, {meta['n_knots']} knots of {len(got)} persons, "
        f"{meta['n_dropped_short']} short takes, {meta['seconds']:.0f} s")
    return meta


def build(split, root=None, render=True, log=None, halves_path=HALVES_JSON):
    """Both sources' shards of a split from the halves file, then <root>/<split>/manifest.json (g1.write_manifest)."""
    root = default_root() if root is None else root
    paths.assert_not_moveatlas(root)
    if paths.in_gates_store(root):
        raise Refused(f"open {split} inputs never go to the gates-only store: {root}")
    hv = read_halves(halves_path)
    check_halves(hv)
    for s in SOURCES:
        build_shard(split, s, hv["halves"][split][s], root, render, log)
    return G.write_manifest(split, root, list(SOURCES))


def check_halves(hv):
    """Raise Refused unless the halves file equals a fresh draw and holds no lockbox, CODY or EVAL person."""
    from ..train import provenance as PV
    fresh = draw_halves()
    if fresh["halves"] != hv["halves"]:
        raise Refused("the halves file differs from the rule's draw")
    persons = {s: hv["halves"]["dev"][s] + hv["halves"]["calib"][s] for s in SOURCES}
    bad = PV.problems(persons)
    if bad:
        raise Refused(f"the G1 open halves hold refused persons: {bad[:3]}")
    return True


class OpenG1Set(G.G1Set):
    """g1.G1Set on the open inputs: dev or calib only; its persons are those of the halves file's split, and none a
    lockbox person."""

    def __init__(self, split, root=None, verify=False, halves_path=HALVES_JSON):
        if split not in SPLITS:
            raise Refused(f"the open G1 sets are dev and calib, not {split!r}")
        self._halves = read_halves(halves_path)
        super().__init__(split, default_root() if root is None else root, allow_lockbox=False, verify=verify)

    def _check_persons(self):
        from ..train import provenance as PV
        want = set(self._halves["halves"][self.split]["gait120"]) | \
            set(self._halves["halves"][self.split]["stroke_gait_vancriekinge"])
        got = set(self.windows["person_id"])
        if got - want:
            raise Refused(f"persons outside the {self.split} half: {sorted(got - want)[:3]}")
        lock = PV.deny_lists().lockbox
        if got & set(lock):
            raise Refused(f"a lockbox person in the open {self.split} inputs")


def open_population_prior(store, seed=G.PARAMS["prior_draw_seed"], stride=G.PARAMS["prior_mean_stride"],
                          n_draws=G.PRIOR_DRAWS):
    """g1.population_prior on the open train store (mdfm/data/open_body_store.OpenTrainStore, split train): the mean
    pose over every stride-th knot of every take on its P3 rest joints, n_draws wrist positions of person-balanced
    train windows."""
    from ..data.sources import person_hash
    if store.split != "train":
        raise Refused(f"the population prior comes from the train persons only, not {store.split}")
    tk = store.takes
    rest = torch.as_tensor(store.rest_joints, dtype=torch.float64)

    def chunks():
        for i in range(len(tk)):
            a = store._raw(i)["rot6d"]
            yield torch.from_numpy(np.asarray(a[::stride], dtype=np.float64)), rest[i]

    with torch.no_grad():
        total, count = G.mean_pose_sums(chunks())
        gen = np.random.default_rng(seed)
        bw = store.sample(gen, n_draws, win_s=G.WIN_S)
        f = gen.integers(0, bw.lengths.numpy())
        rows = torch.arange(n_draws)
        r6 = bw.rot6d[rows, torch.from_numpy(f)].double()
        d = G.wrist_draws(r6, rest[bw.take])
    persons = store.person_list()
    meta = {"source": "open train body store", "store_root": store.root, "stride": int(stride),
            "n_knots_mean": int(count), "n_takes": int(len(tk)), "seed": int(seed), "n_draws": int(n_draws),
            "draw_persons": sorted(set(bw.person.tolist())), "train_persons": persons,
            "train_persons_sha256": person_hash(persons), "store_hashes": store.hashes()}
    return G.PopulationPrior(total / count, d, None, meta)


# ---------------------------------------------------------------- the lockbox read: declaration and rule
GATE = G.GATE
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _hex(x):
    return isinstance(x, str) and bool(_HEX64.match(x))


def load_declaration(path):
    """(declaration, SHA-256 of the file's bytes) of a read declaration (module docstring, step 1). Refused when the
    file is not JSON, holds an em-dash, or its format, gate or a field of DECLARATION_KEYS is wrong or missing."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    if common.EM_DASH in text:
        raise Refused(f"{common.rel(path)} holds an em-dash")
    try:
        d = json.loads(text)
    except ValueError as e:
        raise Refused(f"{common.rel(path)} is not a JSON read declaration ({e})") from e
    if not isinstance(d, dict) or d.get("format") != DECLARATION_FORMAT:
        raise Refused(f"{common.rel(path)}: format {d.get('format') if isinstance(d, dict) else None!r}, expected "
                      f"{DECLARATION_FORMAT!r}")
    missing = [k for k in DECLARATION_KEYS if k not in d]
    if missing:
        raise Refused(f"{common.rel(path)}: the declaration lacks {missing}")
    if d["gate"] != GATE:
        raise Refused(f"{common.rel(path)} declares a read of {d['gate']!r}, not {GATE}")
    return d, common.sha256_bytes(raw)


def declaration_kind(decl):
    """'v1' when the declared training corpus holds AMASS (the source amass, or any archive: only AMASS is declared
    by archive), else 'open' (the open rule's first paragraph)."""
    corpus = decl.get("corpus") or {}
    sources = {str(s).strip().lower() for s in (corpus.get("sources") or ())}
    return "v1" if AMASS_SOURCE in sources or list(corpus.get("archives") or ()) else "open"


def declared_rule(decl, registry_path=common.RULE_HASHES):
    """The rule a declaration reads (module docstring, step 1): {kind, path, text, sha256, adopts [{path, sha256}]}.
    The rule file through common.rule (registered for G1, unchanged); each file it adopts named in its text by the
    SHA-256 of ADOPTS and found at that SHA-256 (the 2026-11-12 version also registered for G1). Refused otherwise."""
    kind = declaration_kind(decl)
    path = RULE_PATH if kind == "open" else RULE_V1_PATH
    text, sha = common.rule(path, gate=GATE, registry_path=registry_path)
    adopts = []
    for attr, want in ADOPTS[kind]:
        p = globals()[attr]
        if want not in text:
            raise Refused(f"{common.rel(path)} does not name the SHA-256 {want[:12]} of the file it adopts")
        try:
            got = common.rule(p, gate=GATE, registry_path=registry_path)[1] if attr == "RULE_V1_PATH" \
                else common.sha256_file(p)
        except Refused as e:
            raise Refused(f"{common.rel(path)} adopts {common.rel(p)}: {e}") from e
        if got != want:
            raise Refused(f"{common.rel(path)} adopts {common.rel(p)} at SHA-256 {want[:12]}, the file reads {got[:12]}")
        adopts.append({"path": common.rel(p), "sha256": got})
    return {"kind": kind, "path": path, "text": text, "sha256": sha, "adopts": adopts}


# ---------------------------------------------------------------- the lockbox read: sets and checks
def lockbox_sets(path=None):
    """{source: sorted person ids} of the open lockbox (rule section 1) from the member file at the SHA-256 the rule
    names (LOCKBOX_SHA256): its gait120 rows and its stroke_gait_vancriekinge rows, every one of these of group
    stroke, in the counts of LOCKBOX_COUNTS. Reads the person list only. Refused otherwise."""
    path = LOCKBOX_CSV if path is None else path
    got = common.sha256_file(path)
    if got != LOCKBOX_SHA256:
        raise Refused(f"{common.rel(path)} has SHA-256 {got[:12]}; the rule names lockbox v1.2, {LOCKBOX_SHA256[:12]}")
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] in SOURCES]
    for r in rows:
        want = LOCKBOX_GROUP.get(r["source"])
        if want is not None and r["group"] != want:
            raise Refused(f"{r['source']} lockbox member of group {r['group']!r}: the rule's set is the patients only")
    out = {s: sorted({r["person_id"] for r in rows if r["source"] == s}) for s in SOURCES}
    if {s: len(v) for s, v in out.items()} != dict(LOCKBOX_COUNTS):
        raise Refused(f"open lockbox sets of {({s: len(v) for s, v in out.items()})} persons, the rule names "
                      f"{dict(LOCKBOX_COUNTS)}")
    if is_cody(*[p for v in out.values() for p in v]):
        raise Refused("a CODY name in the open lockbox sets")
    return out


def _halves_persons(hv):
    return {p for split in SPLITS for ids in hv["halves"][split].values() for p in ids}


def check_declaration(decl, predictor, hv, lock, tiers_path=common.TIERS):
    """The declaration of an open read against the predictor, the halves and the lockbox sets (module docstring, step
    2): the checkpoint file exists and hashes to the declared SHA-256, which is the predictor's; the configuration
    SHA-256 and the body are named; no corpus source is CODY, EVAL or in NEVER_SOURCES; the training lists hash to the
    declared SHA-256 (provenance.list_hashes 'all', the checkpoint's own form), equal the checkpoint's lists when the
    predictor carries them (predictor.loaded.persons) and hold no CODY name, no lockbox person and no person of either
    half; the mode is INFERENCE_MODE (and the predictor's settings MODE_SETTINGS with MODE_BATCH_WINDOWS when it
    reports them); the halves' SHA-256 are those of the halves file and the member file's that of the rule. Returns
    {checkpoint_sha256, checkpoint_file, train_persons, train_persons_sha256, mode, settings}."""
    from ..train import provenance as PV
    ck = decl["checkpoint"]
    if not isinstance(ck, dict) or not _hex(ck.get("sha256")) or not ck.get("file"):
        raise Refused("the declaration names the checkpoint file and its lowercase hex SHA-256")
    path = ck["file"] if os.path.isabs(ck["file"]) else os.path.join(paths.TREE, ck["file"])
    if not os.path.isfile(path):
        raise Refused(f"the declared checkpoint file {ck['file']} does not exist")
    if common.sha256_file(path) != ck["sha256"]:
        raise Refused(f"the declared checkpoint file does not hash to {ck['sha256'][:12]}")
    if getattr(predictor, "sha256", None) != ck["sha256"]:
        raise Refused("the predictor is not the declared checkpoint (SHA-256)")
    if not _hex(decl["config_sha256"]):
        raise Refused("the declaration names the checkpoint's configuration SHA-256")
    body = decl["body"]
    if not isinstance(body, dict) or not all(isinstance(body.get(k), str) and body[k].strip()
                                             for k in ("name", "version", "joint_map")):
        raise Refused("the declaration names the body model, its version and its joint map")
    corpus = decl["corpus"]
    sources = [str(s) for s in (corpus.get("sources") or ())] if isinstance(corpus, dict) else []
    if not sources:
        raise Refused("the declaration names the training corpus's sources")
    for s in sources:
        if is_cody(s):
            raise Refused(f"a CODY source in the declared corpus: {s}")
        if s in NEVER_SOURCES or (os.path.exists(tiers_path) and "EVAL" in common.source_tiers(s, tiers_path)):
            raise Refused(f"an EVAL source in the declared corpus: {s}")
    lists = decl["train_persons"]
    if not isinstance(lists, dict) or not lists or not all(isinstance(v, list) and v for v in lists.values()):
        raise Refused("the declaration names the training persons per source")
    lists = {str(s): sorted(str(p) for p in v) for s, v in lists.items()}
    sha = PV.list_hashes(lists)["all"]
    if sha != decl["train_persons_sha256"]:
        raise Refused(f"the declared training persons hash to {sha[:12]}, not the declared "
                      f"{str(decl['train_persons_sha256'])[:12]}")
    ids = {p for v in lists.values() for p in v}
    if is_cody(*ids):
        raise Refused("a CODY name among the declared training persons")
    own = getattr(getattr(predictor, "loaded", None), "persons", None)
    if isinstance(own, dict):
        if {s: set(v) for s, v in own.items()} != {s: set(v) for s, v in lists.items()}:
            raise Refused("the checkpoint's training lists differ from the declared ones")
    elif set(getattr(predictor, "train_persons", ()) or ()) - ids:
        raise Refused("the predictor names training persons the declaration does not")
    scored = {p for v in lock.values() for p in v} | _halves_persons(hv)
    if ids & scored:
        raise Refused(f"training persons among the scored persons: {sorted(ids & scored)[:3]}")
    inf = decl["inference"]
    if not isinstance(inf, dict) or inf.get("mode") != INFERENCE_MODE:
        raise Refused(f"the read's mode is {INFERENCE_MODE} (the 2026-11-12 version's section 2), declared "
                      f"{inf.get('mode') if isinstance(inf, dict) else None!r}")
    settings = predictor.settings() if callable(getattr(predictor, "settings", None)) else None
    if settings is not None:
        diff = {k: (settings.get(k), v) for k, v in MODE_SETTINGS.items() if settings.get(k) != v}
        if getattr(predictor, "batch_windows", None) != MODE_BATCH_WINDOWS:
            diff["batch_windows"] = (getattr(predictor, "batch_windows", None), MODE_BATCH_WINDOWS)
        if diff:
            raise Refused(f"the predictor's settings differ from the mode {INFERENCE_MODE}: {diff}")
        settings = dict(settings, batch_windows=MODE_BATCH_WINDOWS)
    sets = decl["sets"] if isinstance(decl["sets"], dict) else {}
    if sets.get("lockbox_csv_sha256") != LOCKBOX_SHA256:
        raise Refused("the declaration names another lockbox member file than the rule's")
    if sets.get("halves_sha256") != hv["sha256"]:
        raise Refused("the declaration names other halves than those of the halves file (rule section 3)")
    return {"checkpoint_sha256": ck["sha256"], "checkpoint_file": common.rel(path), "train_persons": lists,
            "train_persons_sha256": sha, "mode": INFERENCE_MODE, "settings": settings}


def check_prior(prior, hv, lock):
    """The trivial baseline (rule section 4): the population prior of the open train store, its training person list
    hashing to its SHA-256 and disjoint from the lockbox and both halves. Refused otherwise."""
    from ..data.sources import person_hash
    meta = getattr(prior, "meta", None) or {}
    if meta.get("source") != "open train body store":
        raise Refused("the trivial baseline is the population prior of the open train store (open_population_prior)")
    tp = sorted(prior.train_persons)
    if not tp or person_hash(tp) != prior.train_persons_sha256:
        raise Refused("the prior's training person list is empty or does not match its SHA-256")
    scored = {p for v in lock.values() for p in v} | _halves_persons(hv)
    if set(tp) & scored:
        raise Refused(f"the prior was drawn from scored persons: {sorted(set(tp) & scored)[:3]}")
    return True


def lockbox_root(root=None):
    """The root of the open lockbox inputs and fits: <gates-only store>/g1_open_v1 by default; refused outside the
    gates-only store."""
    root = (paths.gates_store_dir(VERSION) if root is None else str(root)).replace("\\", "/")
    paths.assert_not_moveatlas(root)
    if not paths.in_gates_store(root):
        raise Refused(f"the open lockbox inputs live in the gates-only store only, not {root}")
    return root


def residual_bound_m():
    """The open body store's residual bound (mdfm.data.open_body_store.residual_bound_m, rule section 2)."""
    from ..data.open_body_store import residual_bound_m as bound
    return bound()


_BUILDER = {}


def store_builder():
    """scripts/build_open_body_store.py as a module (its fit path: SHAPE_TAKES, shape_frames, finish, hop, rms, safe,
    _atomic_save), loaded once."""
    if "m" not in _BUILDER:
        import importlib.util
        p = os.path.join(paths.TREE, "scripts", "build_open_body_store.py")
        spec = importlib.util.spec_from_file_location("mdfm_g1_open_store_builder", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _BUILDER["m"] = m
    return _BUILDER["m"]


def fit_provenance():
    """The fit of the truth as the input manifest names it (rule section 2): the code SHA-256 of FIT_CODE, the SHA-256
    of the fit's data files (residual bound, rest basis, joint-limit table), openfit's constants, the residual bound
    and the stage-1 settings of the store builder."""
    from ..body import openbody as BO
    from ..body import openfit as OF
    from ..data.open_body_store import FIT_MEASURE
    B = store_builder()
    files = {common.rel(p): common.sha256_file(p) for p in (FIT_MEASURE, BO.BASIS_VALUES, OF.LIMITS_JSON)}
    return {"code": list(FIT_CODE), "code_sha256": G.code_sha256(FIT_CODE), "files": files,
            "constants": OF.design_constants(), "residual_bound_m": residual_bound_m(),
            "shape_takes": int(B.SHAPE_TAKES), "shape_frames": int(OF.SHAPE_FRAMES),
            "phenotype_prior": "mdfm.model.prior.PhenotypePrior(), as scripts/build_open_body_store.py --run"}


def read_plan(rl, declaration_sha256, info, lock, hv, halves_path, dev, calib, conformal, prior, root, render):
    """The input manifest of an open read, written before the read is logged (its SHA-256 is the logged input hash):
    the rule and what it adopts, the declaration, the checkpoint and its training persons, the mode, the lockbox
    sets, the halves, the dev and calib manifests, the conformal factors fitted on the calibration half, the prior,
    the input configuration and the fit (fit_provenance)."""
    from ..data.sources import person_hash
    cfg = input_config()
    return {"version": VERSION, "gate": GATE, "split": LOCKBOX,
            "rule": {"path": common.rel(rl["path"]), "sha256": rl["sha256"], "adopts": rl["adopts"]},
            "declaration_sha256": declaration_sha256, "checkpoint_sha256": info["checkpoint_sha256"],
            "checkpoint_file": info["checkpoint_file"], "train_persons_sha256": info["train_persons_sha256"],
            "mode": info["mode"], "settings": info["settings"],
            "lockbox": {"member_file": common.rel(LOCKBOX_CSV), "sha256": LOCKBOX_SHA256,
                        "n": {s: len(v) for s, v in lock.items()},
                        "persons_sha256": {s: person_hash(v) for s, v in lock.items()}},
            "halves_file_sha256": common.sha256_file(halves_path), "halves_sha256": hv["sha256"],
            "dev_manifest_sha256": dev.manifest["manifest_sha256"],
            "calib_manifest_sha256": calib.manifest["manifest_sha256"], "conformal": conformal,
            "prior": {"sha256": prior.sha256, "train_persons_sha256": prior.train_persons_sha256},
            "inputs": {"config": cfg, "config_sha256": common.sha256_json(cfg), "params_sha256":
                       common.sha256_json(G.PARAMS), "min_window_s": MIN_WINDOW_S,
                       "code_sha256": G.code_sha256(INPUT_CODE)},
            "fit": fit_provenance(), "root": root, "render": bool(render)}


def check_read_row(read_row, registry_path=common.RULE_HASHES, log_path=common.READ_LOG):
    """Refused unless read_row is a row of the lockbox read log for G1 under the open rule as registered now: a flag,
    a made-up row, or a row of a read under another rule (the 2026-11-12 version) opens nothing here."""
    if not isinstance(read_row, dict) or read_row.get("gate") != GATE:
        raise Refused("the open lockbox opens only with the row of a logged G1 read (common.log_lockbox_read)")
    _, sha = common.rule(RULE_PATH, gate=GATE, registry_path=registry_path)
    if read_row.get("rule_sha256") != sha:
        raise Refused("the read row names another rule than the registered open G1 rule")
    keys = ("read_at", "gate", "checkpoint_sha256", "rule_sha256")
    if not any(all(r[k] == str(read_row.get(k, "")) for k in keys) for r in common.read_log(log_path)):
        raise Refused(f"the read row is not in {common.rel(log_path)}")
    return read_row


# ---------------------------------------------------------------- the lockbox read: truth inside the read
def lockbox_takes(lock):
    """(takes, refused) of the lockbox persons: their pose rows of the index (rows only, no record), gait120 trials
    joined over their steps, as scripts/build_open_body_store.py lists a source's takes, each through
    canonical.check with the lockbox flag (this module is mdfm.gates). Called inside the logged read only
    (build_lockbox). A member without a lockbox row is refused; a row of another tier is counted in refused."""
    from ..data import canonical
    idx = canonical.index()
    takes, refused, seen = [], collections.Counter(), set()
    for s in SOURCES:
        sel = idx[(idx["source"] == s) & (idx["modality"] == "pose") & idx["person_id"].isin(sorted(lock[s]))]
        for r in sel.to_dict("records"):
            r = {k: ("" if v is None else str(v)) for k, v in r.items()}
            if r.get("lockbox", "").strip().lower() not in ("true", "1"):
                raise Refused(f"{s} {r['person_id']}: a member of the lockbox file with a row outside the lockbox")
            if s in LOCKBOX_GROUP and r.get("group") != LOCKBOX_GROUP[s]:
                raise Refused(f"{s} {r['person_id']}: group {r.get('group')!r}, the rule's set is the patients")
            if r.get("tier") != "RELEASE":
                refused[f"{s}: tier {r.get('tier')}"] += 1
                continue
            key = re.sub(r"_step\d+$", "", r["key"]) if s == "gait120" else r["key"]
            if (s, r["subject_ns"], key) in seen:
                continue
            seen.add((s, r["subject_ns"], key))
            t = {"source": s, "subject_ns": r["subject_ns"], "person_id": r["person_id"], "key": key,
                 "fps": r.get("fps", ""), "n_frames": r.get("n_frames", ""), "tier": r["tier"], "lockbox": "True",
                 "group": r.get("group", "")}
            if is_cody(s, t["subject_ns"], t["person_id"], key):
                raise Refused("a CODY name among the lockbox rows")
            canonical.check(t, allow_lockbox=True)
            takes.append(t)
    return takes, dict(refused)


def lockbox_landmarks(take):
    """scripts/build_open_body_store.py's landmarks(take), the take read with the lockbox flag (granted to this
    module only): the cleaned landmarks on the 100 Hz clock, their validity and the take's fields."""
    from ..body import openfit as OF
    from ..data import open_mocap_raw as OR
    d = OR.load_take(take, allow_lockbox=True)
    x = np.concatenate([d["landmarks"], d["toes"]], 1)
    x, masked = OF.clean(x)
    xc, valid = OF.to_body_clock(x, d["fps"])
    with np.errstate(all="ignore"):
        pn = float(np.nanmedian(np.linalg.norm(x[:, 1] - x[:, 0], axis=-1)))
    return xc, valid, {"fps_native": d["fps"], "n_native": int(x.shape[0]), "up_axis": d["up_axis"],
                       "pelvis_neck_m": pn, "frame_cut": d.get("frame_cut"), **masked}


def _rest_of(phenotype):
    from ..body import openbody as BO
    with torch.no_grad():
        return BO.load_basis(torch.float64).rest(torch.as_tensor(np.asarray(phenotype, np.float64))[None])[0].numpy()


def _fitted(t, info, r6, trn, contact, phenotype, rest):
    return {"take_id": f"{t['source']}/{t['subject_ns']}/{t['key']}", "source": t["source"],
            "subject_ns": t["subject_ns"], "person_id": t["person_id"], "key": t["key"],
            "fps_native": float(info["fps_native"]), "n_native": int(info["n_native"]), "rot6d": r6, "transl": trn,
            "contact": contact, "treadmill": int(info["treadmill"]), "belt_x": float(info["belt_x"]),
            "belt_z": float(info["belt_z"]), "floor_m": float(info["floor_m"]), "rms_m": float(info["rms_m"]),
            "phenotype": phenotype, "rest": rest}


def fit_person(source, person_id, takes, cache_dir, fit_sha256, log=None):
    """The fits of one lockbox person by the store builder's rule (scripts/build_open_body_store.py run: stage 1 on
    up to SHAPE_TAKES takes spread over the person's sorted takes, stage 2 on every take, finish), the landmarks
    read by lockbox_landmarks; cached under cache_dir (inside the gates-only store) as person.json and one npz per
    take, each carrying fit_sha256 (a cached file of another fit is fitted again). A take that fails is counted, as
    in the store. Returns {'takes': [fitted dicts], 'errors': [...], 'stage1': bool}."""
    from ..body import openbody as BO
    from ..body import openfit as OF
    from ..model.prior import PhenotypePrior
    if not paths.in_gates_store(cache_dir):
        raise Refused(f"lockbox fits are cached in the gates-only store only, not {cache_dir}")
    log = log or (lambda *a: None)
    B = store_builder()
    os.makedirs(cache_dir, exist_ok=True)
    ordered = sorted(takes, key=lambda t: t["key"])
    errors = []

    def load(t):
        try:
            return lockbox_landmarks(t)
        except Exception as e:  # noqa: BLE001 (counted per take, as the store's run records it)
            errors.append({"key": t["key"], "error": f"{type(e).__name__}: {e}"})
            return None
    basis, prior = BO.load_basis(), PhenotypePrior()
    pfile = os.path.join(cache_dir, "person.json")
    pj = None
    if os.path.exists(pfile):
        with open(pfile, encoding="utf-8") as f:
            pj = json.load(f)
        if pj.get("fit_sha256") != fit_sha256:
            pj = None
        else:
            u, o = torch.tensor(pj["u"]), torch.tensor(pj["o"])
    if pj is None:
        pick = [ordered[int(i)] for i in np.unique(np.linspace(0, len(ordered) - 1,
                                                               min(B.SHAPE_TAKES, len(ordered))).astype(int))]
        items = [r for r in (load(t) for t in pick) if r is not None]
        sf = B.shape_frames([(x, v) for x, v, _ in items], OF.SHAPE_FRAMES)
        del items
        if sf is None:
            log(f"{source} {person_id}: no frame for stage 1")
            return {"takes": [], "errors": errors, "stage1": False}
        u, o, _ = OF.fit_person(sf, basis, prior, hip_on_pelvis=B.hop({"source": source}))
        pj = {"source": source, "person_id": person_id, "u": u.tolist(), "o": o.tolist(),
              "phenotype": dict(zip(BO.PHENOTYPE_KEYS, map(float, BO.to_phenotype(u)))), "frames": int(sf[0].shape[0]),
              "fit_sha256": fit_sha256}
        tmp = pfile + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(pj, f)
        os.replace(tmp, pfile)
    phen = np.array([pj["phenotype"][k] for k in BO.PHENOTYPE_KEYS], dtype=np.float64)
    rest = _rest_of(phen)
    out = []
    for t in ordered:
        p = os.path.join(cache_dir, B.safe(t["key"]) + ".npz")
        if os.path.exists(p):
            with np.load(p) as z:
                info = json.loads(bytes(z["info"]).decode("utf-8"))
                if info.get("fit_sha256") == fit_sha256:
                    out.append(_fitted(t, info, z["rot6d"], z["transl"], z["contact"], phen, rest))
                    continue
        got = load(t)
        if got is None:
            continue
        x, v, meta = got
        try:   # the fit, its residuals and the take's fields (finish), as the store's run
            aa, tr = OF.fit_take(x, v, u, o, basis, prior, hip_on_pelvis=B.hop(t))
            d, P, _ = OF.residuals(x, v, aa, tr, u, o, basis, prior, hip_on_pelvis=B.hop(t))
            r6, trn, contact, fields = B.finish(aa, tr, P)
        except Exception as e:  # noqa: BLE001 (counted per take)
            errors.append({"key": t["key"], "error": f"fit: {type(e).__name__}: {e}"})
            continue
        d = d.numpy()
        info = {**{k: t[k] for k in ("source", "subject_ns", "person_id", "key", "fps", "n_frames")}, **meta, **fields,
                "n_knots": int(r6.shape[0]), "rms_m": B.rms(d), "rms_canonical15_m": B.rms(d[:, :15]),
                "rms_toes_m": B.rms(d[:, 15:]), "valid_share": float(v.mean()), "fit_sha256": fit_sha256}
        B._atomic_save(p, rot6d=r6, transl=trn, contact=contact,
                       info=np.frombuffer(json.dumps(info).encode("utf-8"), dtype=np.uint8))
        out.append(_fitted(t, info, r6, trn, contact, phen, rest))
        log(f"{source} {person_id} {t['key']}: {info['n_knots']} knots, RMS {1000 * info['rms_m']:.1f} mm")
    return {"takes": out, "errors": errors, "stage1": True}


class FittedStore:
    """Fitted takes of one source held in memory with the reader contract that build_shard and open_take use (that
    of OpenTrainStore: takes, take(i), rest_joints, phenotype, split, root, hashes()); the takes whose RMS residual
    exceeds bound_m are left out and counted (n_refused_bound), as the store's reader does. split 'lockbox' for the
    read's own fits (build_lockbox); a dev or calib FittedStore serves fixtures."""

    COLUMNS = ("take_id", "source", "subject_ns", "person_id", "key", "fps_native", "n_native", "treadmill", "belt_x",
               "belt_z", "floor_m", "rms_m")

    def __init__(self, split, root, fitted, bound_m):
        import pandas as pd
        self.split, self.root, self.bound_m = split, str(root), float(bound_m)
        keep = sorted((f for f in fitted if float(f["rms_m"]) <= self.bound_m), key=lambda f: f["take_id"])
        self.n_refused_bound = len(fitted) - len(keep)
        self._fits = keep
        self.takes = pd.DataFrame([{k: f[k] for k in self.COLUMNS} for f in keep], columns=list(self.COLUMNS))
        self.phenotype = np.array([np.asarray(f["phenotype"], np.float64) for f in keep]).reshape(len(keep), -1)
        self.rest_joints = np.array([np.asarray(f["rest"], np.float64) for f in keep]).reshape(len(keep), 22, 3)

    def __len__(self):
        return len(self._fits)

    def take(self, i):
        """rot6d (T, 22, 6), transl (T, 3) float64 m (the belt frame for a treadmill take), contact (T, 2)."""
        f = self._fits[int(i)]
        tr = np.asarray(f["transl"], np.float64)
        if int(f["treadmill"]):
            k = np.arange(len(tr), dtype=np.float64)[:, None] / G.RATE
            tr = tr - k * np.array([f["belt_x"], 0.0, f["belt_z"]])
        return {"rot6d": np.asarray(f["rot6d"]), "transl": tr, "contact": np.asarray(f["contact"])}

    def hashes(self):
        from ..data.sources import person_hash
        h = hashlib.sha256()
        for f in self._fits:
            h.update(f["take_id"].encode("utf-8"))
            for k in ("rot6d", "transl"):
                h.update(np.ascontiguousarray(f[k]).tobytes())
        return {"fits_sha256": h.hexdigest(), "bound_m": self.bound_m, "n_takes": len(self._fits),
                "n_refused_bound": int(self.n_refused_bound),
                "persons_sha256": person_hash(sorted({f["person_id"] for f in self._fits}))}


def build_lockbox(read_row, root, lock, render=True, log=None, registry_path=common.RULE_HASHES,
                  log_path=common.READ_LOG):
    """The truth and inputs of the open lockbox inside a logged read (module docstring, step 4): refused without the
    read's row (check_read_row); the lockbox rows (lockbox_takes), the fits (fit_person, cached under
    <root>/lockbox/_fits), the residual bound (FittedStore), the shards (build_shard) and the manifest under
    <root>/lockbox. Returns (manifest, per-source fit report)."""
    check_read_row(read_row, registry_path, log_path)
    root = lockbox_root(root)
    log = log or (lambda *a: None)
    B = store_builder()
    takes, refused = lockbox_takes(lock)
    fit_sha = common.sha256_json(fit_provenance())
    bound = residual_bound_m()
    fits_dir = os.path.join(root, LOCKBOX, "_fits")
    report = {}
    for s in SOURCES:
        fitted, errors, without, no_stage1 = [], [], [], []
        for pid in lock[s]:
            mine = [t for t in takes if t["source"] == s and t["person_id"] == pid]
            if not mine:
                without.append(pid)
                continue
            got = fit_person(s, pid, mine, os.path.join(fits_dir, s, B.safe(pid)), fit_sha, log)
            fitted += got["takes"]
            errors += [dict(e, person_id=pid) for e in got["errors"]]
            if not got["stage1"]:
                no_stage1.append(pid)
        store = FittedStore(LOCKBOX, fits_dir, fitted, bound)
        meta = build_shard(LOCKBOX, s, lock[s], root, render, log, store=store, read_row=read_row,
                           registry_path=registry_path, log_path=log_path)
        with open(os.path.join(root, LOCKBOX, s, "windows.csv"), encoding="utf-8", newline="") as f:
            scored = {r["person_id"] for r in csv.DictReader(f)}
        report[s] = {"persons": len(lock[s]), "takes_listed": sum(t["source"] == s for t in takes),
                     "takes_fitted": len(fitted), "fit_errors": len(errors), "errors": errors,
                     "over_bound": int(store.n_refused_bound), "kept": len(store), "bound_m": bound,
                     "dropped_short": int(meta["n_dropped_short"]), "windows": int(meta["n_windows"]),
                     "persons_without_take": without, "persons_without_stage1": no_stage1,
                     "unscored": sorted(set(lock[s]) - scored),
                     "refused_rows": {k: v for k, v in refused.items() if k.startswith(s + ":")}}
    return G.write_manifest(LOCKBOX, root, list(SOURCES)), report


class OpenLockboxSet(G.G1Set):
    """g1.G1Set on the open lockbox inputs: opened only with the row of the logged read under the registered open
    rule (check_read_row), from the gates-only store, every array file verified; its persons are lockbox persons of
    their source and of no half."""

    def __init__(self, root, read_row, lock, registry_path=common.RULE_HASHES, log_path=common.READ_LOG,
                 halves_path=HALVES_JSON):
        check_read_row(read_row, registry_path, log_path)
        self._row, self._registry_path, self._log_path = read_row, registry_path, log_path
        self._lock = {s: set(v) for s, v in lock.items()}
        self._halves = read_halves(halves_path)
        super().__init__(LOCKBOX, lockbox_root(root), allow_lockbox=True, verify=True)

    def _lockbox_guard(self, granted):
        if not granted:
            raise Refused("the open G1 lockbox inputs are opened by mdfm/gates only")
        check_read_row(self._row, self._registry_path, self._log_path)

    def _check_persons(self):
        bad = sorted({p for a, p in zip(self.windows["archive"], self.windows["person_id"])
                      if p not in self._lock.get(a, ())})
        if bad:
            raise Refused(f"windows of persons outside the open lockbox sets: {bad[:3]}")
        if set(self.windows["person_id"]) & _halves_persons(self._halves):
            raise Refused("a person of a half in the open lockbox inputs")


# ---------------------------------------------------------------- the lockbox read
@dataclass
class _Declared:
    """The declared checkpoint as g1.lines names it: its SHA-256 and its training persons' SHA-256."""
    sha256: str
    train_persons_sha256: str


@dataclass
class OpenLockboxRead:
    """What read_lockbox returns (nothing is written to reports/)."""
    results: list
    prior_results: list
    dev_results: list
    dev_prior_results: list
    table: object
    baseline: object
    per_source: object
    prior_per_source: object
    per_person: object
    secondary: list
    fits: dict
    manifest: dict
    plan: dict
    plan_sha256: str
    declaration_sha256: str
    rule: dict
    read_row: dict
    root: str
    extra: dict = field(default_factory=dict)


def _fit_conformal(calib, p):
    """The conformal factors of a predictor on the calibration half (rule section 4, inside the read): both wrist
    predictives for a model (mdfm.eval.infer.fit_conformal, c_wrist and c_six), the single one for the prior."""
    if hasattr(p, "c_six"):
        from ..eval import infer
        fits = infer.fit_conformal(calib, p)
        p.c_wrist, p.c_six = fits["wrist"]["c"], fits["wrist_six"]["c"]
        return fits
    fit = G.fit_conformal(calib, p)
    p.c_wrist = fit["c"]
    return {"wrist": fit}


def read_lockbox(declaration_path, predictor, prior, reason="", reread=False, root=None, inputs_root=None,
                 halves_path=HALVES_JSON, render=True, registry_path=common.RULE_HASHES, log_path=common.READ_LOG,
                 tiers_path=common.TIERS, log=None):
    """The open G1 lockbox read of one declared checkpoint (module docstring, "The lockbox read"). predictor: the
    declared checkpoint as a G1 predictor (mdfm.eval.infer.G1Predictor under the mode ema_s20_nopool); prior: the
    population prior of the open train store (open_population_prior). reason: logged with the read (a reread needs
    one). root: the gates-only store folder of the lockbox inputs (default <gates store>/g1_open_v1); inputs_root:
    the dev and calib inputs (default cache/g1/g1_open_v1). Every refusal that needs no lockbox data comes before
    the read is logged; the lockbox rows, fits and inputs exist only after it. Returns an OpenLockboxRead."""
    log = log or (lambda *a: None)
    decl, decl_sha = load_declaration(declaration_path)
    if declaration_kind(decl) != "open":
        raise Refused("a declaration whose corpus holds AMASS reads the 2026-11-12 version through g1.run, not the "
                      "open path")
    rl = declared_rule(decl, registry_path)
    rule_sha = rl["sha256"]
    ck = decl["checkpoint"].get("sha256") if isinstance(decl["checkpoint"], dict) else None
    prior_reads = [r for r in common.read_log(log_path) if (r["gate"], r["checkpoint_sha256"], r["rule_sha256"])
                   == (GATE, ck, rule_sha)]
    if prior_reads and not reread:
        raise Refused(f"G1 was read on the open lockbox with checkpoint {str(ck)[:12]} on {prior_reads[0]['read_at']}: "
                      "a second read needs reread and a reason")
    if reread and (not prior_reads or not str(reason).strip()):
        raise Refused("a reread needs a logged read of this checkpoint and rule, and a reason")
    hv = read_halves(halves_path)
    check_halves(hv)
    lock = lockbox_sets()
    info = check_declaration(decl, predictor, hv, lock, tiers_path)
    check_prior(prior, hv, lock)
    root = lockbox_root(root)
    calib = OpenG1Set("calib", inputs_root, verify=True, halves_path=halves_path)
    dev = OpenG1Set("dev", inputs_root, verify=True, halves_path=halves_path)
    conf = {"model": _fit_conformal(calib, predictor), "prior": _fit_conformal(calib, prior)}
    who = _Declared(info["checkpoint_sha256"], info["train_persons_sha256"])
    dev_table, dev_base = G.score(dev, predictor, log=log), G.score(dev, prior)
    dev_results = G.lines(dev_table, dev_base, who, rule_sha)
    dev_prior = G.lines(dev_base, dev_base, prior, rule_sha)
    if declared_rule(decl, registry_path)["sha256"] != rule_sha:
        raise Refused("the open G1 rule changed during the dev run: no lockbox read")
    plan = read_plan(rl, decl_sha, info, lock, hv, halves_path, dev, calib, conf, prior, root, render)
    plan_sha = common.sha256_json(plan)
    row = common.log_lockbox_read(GATE, info["checkpoint_sha256"], rule_sha, reread=reread, reason=reason,
                                  inputs_sha256=plan_sha, code_sha256=G.code_sha256(READ_CODE),
                                  note=f"open declaration {decl_sha}", log_path=log_path, registry_path=registry_path)
    log(f"lockbox read logged at {row['read_at']}")
    manifest, fits = build_lockbox(row, root, lock, render, log, registry_path, log_path)
    gset = OpenLockboxSet(root, row, lock, registry_path, log_path, halves_path)
    overlap = set(gset.persons) & {p for v in info["train_persons"].values() for p in v}
    if overlap:
        raise Refused(f"lockbox persons among the declared training persons: {sorted(overlap)[:3]}")
    table, base = G.score(gset, predictor, log=log), G.score(gset, prior)
    rule_doc = {"kind": rl["kind"], "path": common.rel(rl["path"]), "sha256": rule_sha, "adopts": rl["adopts"]}
    counts = {name: {m["archive"]: {"takes": m["n_takes"], "dropped_short": m["n_dropped_short"],
                                    "windows": m["n_windows"], "persons": m["n_persons"],
                                    "over_bound": (m.get("body_store") or {}).get("n_refused_bound")}
                     for m in gs.shards} for name, gs in (("dev", dev), ("calib", calib), (LOCKBOX, gset))}
    return OpenLockboxRead(
        results=G.lines(table, base, who, rule_sha), prior_results=G.lines(base, base, prior, rule_sha),
        dev_results=dev_results, dev_prior_results=dev_prior, table=table, baseline=base,
        per_source=G.per_group(table, "archive"), prior_per_source=G.per_group(base, "archive"),
        per_person=G.per_group(table, "person_id"), secondary=G.secondary(table), fits=fits, manifest=manifest,
        plan=plan, plan_sha256=plan_sha, declaration_sha256=decl_sha, rule=rule_doc, read_row=row, root=root,
        extra={"dev_per_source": G.per_group(dev_table, "archive"), "train_persons": info["train_persons"],
               "window_counts": counts})
