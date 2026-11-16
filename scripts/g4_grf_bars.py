"""GRF (wave 18, 2026-10-16; docs/notes/STRATEGY_2026-10-15.md section 4, rank 5): the learned bars for ground
reaction force (G4) on the G4 development persons. Does the force head of the decoded body keep its premium over the
Newton floor when the bars are learned: a frozen-token ridge probe of per-foot vertical GRF (the body seeds v1_B and
s2_B, and the G12 baselines MFR and JEPA, two seeds each) and a supervised pose-to-GRF regressor on the measured
kinematics (no model), each fitted on the training fold only? Every output carries LABEL: dev diagnostics, not a G4
read, not a G12 comparison.

Prediction and method: reports/bench/g4_grf/prediction.txt, hashed before any run (prediction.sha256 beside it and
PREDICTION_SHA256 here); check_prediction asserts the three agree before any target, record or checkpoint is read, in
every mode, so a prediction edited after the fact stops the script.

Sets (prediction.txt section 1): fit, the 110 able-bodied persons of the stroke_gait_vancriekinge train fold
(mdfm.data.sources.folds), the persons whose plate totals trained the force head of every checkpoint here (asserted
equal to each checkpoint's training list of that source); scored, AB28 (the able-bodied dev fold, eval_g4_dev's
secondary row), ST38 (the 40 non-lockbox stroke patients, 38 with a pose and plate pair, the rule's development row
(a)) and ST7 (the stage-2 stroke dev fold of 8, 7 scored, the memo's [G4] row; beside, a subset of ST38). The S arm,
beside and deciding nothing: the bars refitted on the fit set plus the 32 stroke patients of the stage-2 stroke train
fold, scored on ST7. Lockbox and CODY names are refused before any record opens (eval_g4_dev.check_sets).

Truth, samples, statistics: mdfm/gates/g4.py, imported, never edited: g4.grf_trials (body mass from the C3D, else the
MAT file; plate truth at the pose frames), g4.grf_person_table (per person RMSE in percent BW and Pearson r over its
scored (frame, foot) samples; a sample is scored where the plate is valid and every predictor is finite, the Newton
floor among them), g4.summary (mean over persons, grf_r_gain against the Newton floor) and g4.person_bootstrap (2,000
resamples, seed 20260929). Every bar predicts every scored sample (a non-finite prediction there takes the bar's
training mean of its target column, counted), so the samples equal those of the body's rows; asserted per person
against the reports (samples, Newton RMSE and r).

Rows (prediction.txt section 2):
- b1, supervised ridge (mdfm/eval/probes.fit_probe) and b2, gradient boosting (the HGB_PARAMS of
  scripts/g3_supervised_baseline.py) on per-frame kinematic features (kinematic_features: the world joints of the
  harness low-passed as the Newton floor does, in a heading frame, positions, velocities, centre-of-mass velocity and
  acceleration, the floor's contact probabilities and BW, at five context offsets), one model for both feet (the
  right foot mirrored);
- token probes: per 0.25 s patch token (mdfm/baselines/readouts.encode_windows on the G4 dev script's pose input), a
  ridge to the per-frame GRF of each foot at each of the patch's 25 frames (MaskedRidge, one penalty), variant latent
  or measurement chosen by the person cross-fitted grf_rmse on the fit set;
- the body's force-head rows, read from reports/gates/g4/dev_<checkpoint>_g4h/ (BODY_REPORTS).
Premium rule (section 3): per line (grf_rmse, grf_r_gain) and set, the comparator is the bar with the best seed-mean
point; PREMIUM when both body seeds are better than every comparator seed, the paired person bootstrap interval of the
seed-mean difference lies wholly on the better side of 0, and the difference is at least m better (grf_rmse: m =
max(0.10 x the comparator's point, 2 d); grf_r_gain: m = max(0.03, 2 d); d the larger seed spread); LOSS for the
mirror of the first two; NO PREMIUM otherwise. Deciding sets AB28 and ST38.

Modes (all CPU: CUDA_VISIBLE_DEVICES=-1 is set before torch loads):
  --tokens KEY [KEY ...] | all   the records of every set read once (g4.paired_records, flags False), the pose input
                 and windows of eval_g4_dev, then per checkpoint the frozen patch tokens and each pose frame's (patch,
                 offset) into cache/g4_grf/tokens_<name>.npz (resumable: a file of the same checkpoint and settings is
                 kept), with tokens_<name>.json beside it.
  (default)      the read: every bar fitted and scored, the body rows, the lines, the premium tests and the
                 predictions; reports/bench/g4_grf/ results.json, summary.md, per_person.csv, _provenance.json.
  --print-jobs   the commands of a full run.
--limit-persons N (smoke) keeps the first N persons of each set and needs --out and --cache.

Run (mdfm environment; OMP_NUM_THREADS=2):
  python scripts/g4_grf_bars.py --tokens all
  python scripts/g4_grf_bars.py
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # every mode runs on the CPU
    os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
for _p in (HERE, TREE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
from mdfm.data.canonical import Refused  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g4_grf")
CACHE = os.path.join(TREE, "cache", "g4_grf")
PREDICTION_PATH = os.path.join(OUT, "prediction.txt")
PREDICTION_SHA_PATH = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "e71e09df5ca3667a5e306aab3fbbde5f129707ed6900ba21f9659f45892b9341"
LABEL = "dev diagnostic, not a G4 read, not a G12 comparison"
TASK = "GRF"
PATCH_S = 0.25                      # s, the tokenizer's latent patch (model/tokenize.PATCH_S)
CONTEXT_S = (-0.2, -0.1, 0.0, 0.1, 0.2)
MARGIN_FRACTION = 0.10              # grf_rmse margin: share of the comparator's point
DECIDING = ("AB28", "ST38")
SETS = ("AB28", "ST38", "ST7")
LINES = ("grf_rmse", "grf_r_gain")
VARIANTS = ("latent", "measurement", "stem")
CANDIDATES = ("latent", "measurement")
STAGE2_FOLDS = os.path.join(TREE, "reports", "stage2_sources", "reviewed_ecf7bbf_2026-10-06", "folds_proposed.csv")
BODY_REPORTS = {"v1_B": "reports/gates/g4/dev_phaseA_root_full_ckpt_B_032044_g4h",
                "s2_B": "reports/gates/g4/dev_pilot_v1_s2_ckpt_B_032044_g4h"}
STROKEFOLD_REPORT = "reports/gates/g4/dev_phaseA_root_full_ckpt_B_032044_strokefold"
CHECK_TOL = 1e-6                    # absolute tolerance of the per-person checks against the reports


@dataclass(frozen=True)
class End:
    """A checkpoint: key, report name, tree-relative path, kind (body or baseline), arm, seed, SHA-256."""
    key: str
    name: str
    path: str
    kind: str
    arm: str
    seed: int
    sha256: str


ENDS = (
    End("v1_B", "phaseA_root_full_ckpt_B_032044", "cache/runs/phaseA_root_full/ckpt_B_032044.pt", "body", "body",
        20260928, "cffccdac4109dfbfb921ed4ebcb690ce27dd2d35045a3afdaaca7d38f321a677"),
    End("s2_B", "pilot_v1_s2_ckpt_B_032044", "cache/runs/pilot_v1_s2/ckpt_B_032044.pt", "body", "body",
        20261001, "5e97c4fe3b208f8f11850a5318ed827d26316f1e2fc867355d74f50a3d96e023"),
    End("mfr_s1", "g12_mfr_s20260928_ckpt_B_056774", "cache/runs/g12_mfr_s20260928/ckpt_B_056774.pt", "baseline",
        "mfr", 20260928, "f1b23a15147537b9ddbead491774a59ba9b41e001438173669e984e55d3c38a0"),
    End("mfr_s2", "g12_mfr_s20261001_ckpt_B_056774", "cache/runs/g12_mfr_s20261001/ckpt_B_056774.pt", "baseline",
        "mfr", 20261001, "ef9fc2b7b2c30806e48b7639b4b75bbef05bab90dca8fa4a84387e6f67dbbb4d"),
    End("jepa_s1", "g12_jepa_s20260928_ckpt_B_047617", "cache/runs/g12_jepa_s20260928/ckpt_B_047617.pt", "baseline",
        "jepa", 20260928, "18a4e382d6c3ee498a17c4a255dac5457afb8e648a53dc95e0d075a1b1b9d1a9"),
    End("jepa_s2", "g12_jepa_s20261001_ckpt_B_047617", "cache/runs/g12_jepa_s20261001/ckpt_B_047617.pt", "baseline",
        "jepa", 20261001, "c974dc55e3d7df9c0f2b6ce836b3f02fa4a3d24dd33dfc46adcb0cce6839e925"),
)
END = {e.key: e for e in ENDS}
BODY_KEYS = ("v1_B", "s2_B")
TOKEN_ARMS = {"tok_body": ("v1_B", "s2_B"), "tok_mfr": ("mfr_s1", "mfr_s2"), "tok_jepa": ("jepa_s1", "jepa_s2")}
BARS = {"b1": ("b1",), "b2": ("b2",), **{arm: tuple(f"tok_{k}" for k in keys) for arm, keys in TOKEN_ARMS.items()}}
GROUPS = {"supervised": ("b1", "b2"), "token": tuple(TOKEN_ARMS)}
BODY = tuple(f"body_{k}" for k in BODY_KEYS)
TOKEN_SETTINGS = {"window_s": None, "hop_s": None, "min_s": None, "patch_s": PATCH_S, "weights": "ema",
                  "device": "cpu", "amp_bf16": False, "batch_windows": None}


# ---------------------------------------------------------------- siblings, prediction, folders
def _sibling(name):
    key = f"_grf_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def G4DEV():
    """scripts/eval_g4_dev.py, imported (never edited, never run as a process here)."""
    return _sibling("eval_g4_dev")


def SB():
    """scripts/g3_supervised_baseline.py (HGB_PARAMS, below_normal, wait_while_slow)."""
    return _sibling("g3_supervised_baseline")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(path=PREDICTION_PATH, sha_path=PREDICTION_SHA_PATH, expected=PREDICTION_SHA256):
    """The SHA-256 of the prediction file, asserted equal to its sidecar and to PREDICTION_SHA256."""
    got = sha256_file(path)
    with open(sha_path, encoding="utf-8") as f:
        side = f.read().split()[0].strip()
    if not (got == side == expected):
        raise AssertionError(f"prediction hash mismatch: file {got}, sidecar {side}, script {expected}")
    return got


def rel(path):
    from mdfm.gates import common
    return common.rel(path)


def token_path(end, cache=None):
    return os.path.join(cache or CACHE, f"tokens_{end.name}.npz")


def token_settings():
    from mdfm.baselines import readouts as RO
    from mdfm.eval import infer as I
    return {**TOKEN_SETTINGS, "window_s": I.WIN_S, "hop_s": I.HOP_S, "min_s": I.MIN_S,
            "batch_windows": RO.BATCH_WINDOWS}


# ---------------------------------------------------------------- sets
def stage2_stroke(fold, path=STAGE2_FOLDS):
    """Sorted person ids of the stage-2 proposed folds file with source stroke_gait_vancriekinge, group stroke and the
    fold given ('train' or 'dev')."""
    from mdfm.gates import g4
    with open(path, encoding="utf-8", newline="") as f:
        return sorted(str(r["person_id"]) for r in csv.DictReader(f)
                      if r["source"] == g4.STROKE and r["group"] == "stroke" and r["fold"] == fold)


def sets_of(limit=0):
    """{train, AB28, ST38, ST7, st32}: person lists (module docstring, Sets), refused before any record opens when a
    set holds a CODY name or a lockbox person, or when the fit set meets a scored set. limit keeps the first N persons
    of each (ST7 then the persons of ST38's first N that are in the fold, st32 likewise)."""
    from mdfm.data import sources
    from mdfm.gates import g4
    E = G4DEV()
    f = sources.folds()
    train = sorted(f.loc[(f["source"] == g4.STROKE) & (f["fold"] == "train"), "person_id"])
    able = E.able_bodied_dev()
    stroke, _ = E.stroke_set("all")
    st7, _ = E.stroke_set("fold")
    st32 = stage2_stroke("train")
    if not (set(st7) <= set(stroke) and set(st32) <= set(stroke) and not set(st7) & set(st32)):
        raise AssertionError("the stage-2 stroke folds are not a partition inside the 40 non-lockbox stroke patients")
    if limit:
        train, able, stroke = train[:limit], able[:limit], stroke[:limit]
        st7 = [p for p in st7 if p in set(stroke)] or st7[:1]
        st32 = [p for p in st32 if p in set(stroke) and p not in set(st7)][:limit]
        stroke = sorted(set(stroke) | set(st7))
    E.check_sets({"train": train, "able_bodied_dev": able, "stroke_dev": stroke, "stroke_train32": st32},
                 template_key="train")
    for name, ps in (("AB28", able), ("ST38", stroke)):
        if set(ps) & set(train):
            raise Refused(f"fit persons in the scored set {name}")
    return {"train": list(train), "AB28": list(able), "ST38": list(stroke), "ST7": list(st7), "st32": list(st32)}


def check_training_lists(train, scored, ends=ENDS, exact=True):
    """Per checkpoint: its training list of stroke_gait_vancriekinge (persons/*.txt beside it) equals the fit set
    (holds it, exact False: a smoke run's subset), and no scored person is in any of its training lists
    (eval_g4_dev.person_lists, assert_disjoint). Returns {key: {list sizes}}."""
    from mdfm.gates import g4
    E = G4DEV()
    out = {}
    for e in ends:
        lists = E.person_lists(os.path.join(TREE, e.path))
        have = set(lists.get(g4.STROKE, ()))
        if (have != set(train)) if exact else not (set(train) <= have):
            raise AssertionError(f"{e.path}: its {g4.STROKE} training list is not the fit set")
        E.assert_disjoint(scored, lists)
        out[e.key] = {k: len(v) for k, v in lists.items()}
    return out


# ---------------------------------------------------------------- tokens
def frame_patches(t, starts, durations, n_patches, fps, patch_s=PATCH_S):
    """(window (T,) int64, patch (T,) int64, offset (T,) int64), -1 where no window covers a frame: each pose frame at
    time t goes to the window whose span [t0, t0 + duration) holds it nearest its middle (ties to the earlier window,
    probes.most_central's rule), its patch p = floor((t - t0) / patch_s) and its frame offset j in the patch, j =
    round((t - t0 - p patch_s) fps), 0 <= j < round(patch_s fps). ValueError when a patch index reaches the window's
    patch count."""
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(starts, dtype=np.float64)
    d = np.asarray(durations, dtype=np.float64)
    T = len(t)
    win = np.full(T, -1, np.int64)
    patch = np.full(T, -1, np.int64)
    off = np.full(T, -1, np.int64)
    if not len(s) or not T:
        return win, patch, off
    cover = (t[:, None] >= s[None] - 1e-9) & (t[:, None] < (s + d)[None] - 1e-9)
    dist = np.where(cover, np.abs(t[:, None] - (s + d / 2.0)[None]), np.inf)
    best = np.argmin(dist, axis=1)
    ok = np.isfinite(dist[np.arange(T), best])
    n_off = int(round(patch_s * fps))
    r = t - s[best]
    p = np.floor(r / patch_s + 1e-6).astype(np.int64)
    j = np.rint((r - p * patch_s) * fps).astype(np.int64)
    up = j >= n_off
    p[up] += 1
    j[up] -= n_off
    j = np.maximum(j, 0)
    npat = np.asarray(n_patches, dtype=np.int64)[best]
    if (ok & (p >= npat)).any():
        raise ValueError(f"a frame falls in patch {int(p[ok & (p >= npat)][0])} of a window of "
                         f"{int(npat[ok & (p >= npat)][0])} patches")
    win[ok], patch[ok], off[ok] = best[ok], p[ok], j[ok]
    return win, patch, off


def token_inputs(persons, stats=None, log=print):
    """[{person, key, t (T,), fps, windows}] of every paired stroke recording of the persons (g4.paired_records, reader
    flags False): the G4 dev script's pose input (window_store.attach_raw_root, eval_g4_dev.pose_stream, the
    covariates of windows.covariates, eval_g4_dev.cut_windows). Each record is refused for a CODY name or a lockbox
    person (eval_g4_dev.G4Predictor.refuse) and checked to belong to the persons."""
    import collections

    from mdfm.data import window_store as WS
    from mdfm.data import windows as W
    from mdfm.gates import g4
    E = G4DEV()
    stats = collections.Counter() if stats is None else stats
    want = set(persons)
    out = []
    t0 = time.time()
    for pr in g4.paired_records(g4.STROKE, list(persons), stats=stats):
        E.G4Predictor.refuse(pr)
        if pr.person_id not in want:
            raise AssertionError(f"{pr.person_id} outside the requested persons")
        pr = WS.attach_raw_root(pr)
        stream, t64 = E.pose_stream(pr)
        cov = W.covariates(dict(pr.streams["pose3d"], key=pr.key), pr.source)
        wins, _ = E.cut_windows(stream, t64, pr.source, pr.person_id, pr.key, cov)
        out.append({"person": pr.person_id, "key": pr.key, "t": np.asarray(t64, dtype=np.float64),
                    "fps": float(stream.fps), "windows": wins})
        stats["trials_without_window"] += int(not wins)
    log(f"token inputs: {len(out)} recordings of {len(want)} persons ({time.time() - t0:.0f} s)", flush=True)
    return out


def encode_inputs(model, inputs, batch_windows=None):
    """Arrays of the token file of one checkpoint (module docstring, --tokens): per trial its person, key, frame span;
    per frame its time, token row (-1 uncovered) and offset; per token row its trial, window, patch, measurement count
    and the variants (float32)."""
    from mdfm.baselines import readouts as RO
    B = RO.BATCH_WINDOWS if batch_windows is None else batch_windows
    tr_person, tr_key, tr_start, tr_len = [], [], [], []
    f_t, f_row, f_off = [], [], []
    r_trial, r_win, r_patch, r_nm = [], [], [], []
    feats = {v: [] for v in VARIANTS}
    n_frames = n_rows = 0
    for i, rec in enumerate(inputs):
        t = rec["t"]
        tr_person.append(rec["person"])
        tr_key.append(rec["key"])
        tr_start.append(n_frames)
        tr_len.append(len(t))
        row = np.full(len(t), -1, np.int64)
        off = np.full(len(t), -1, np.int64)
        wins = rec["windows"]
        if wins:
            enc = RO.encode_windows(model, wins, B)
            npat = [int(e["latent"].shape[0]) for e in enc]
            w, p, j = frame_patches(t, [x.t0 for x in wins], [x.duration for x in wins], npat, rec["fps"])
            base = np.concatenate([[0], np.cumsum(npat)[:-1]]).astype(np.int64)
            cov = w >= 0
            row[cov] = n_rows + base[w[cov]] + p[cov]
            off[cov] = j[cov]
            for k, e in enumerate(enc):
                r_trial.append(np.full(npat[k], i, np.int64))
                r_win.append(np.full(npat[k], k, np.int64))
                r_patch.append(np.arange(npat[k], dtype=np.int64))
                r_nm.append(np.asarray(e["n_measurement"], dtype=np.float32))
                for v in VARIANTS:
                    feats[v].append(np.asarray(e[v], dtype=np.float32))
            n_rows += int(sum(npat))
        f_t.append(t)
        f_row.append(row)
        f_off.append(off)
        n_frames += len(t)
    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    out = {"trial_person": np.asarray(tr_person, dtype=str), "trial_key": np.asarray(tr_key, dtype=str),
           "trial_start": np.asarray(tr_start, np.int64), "trial_len": np.asarray(tr_len, np.int64),
           "frame_t": cat(f_t, np.float64), "frame_row": cat(f_row, np.int64), "frame_off": cat(f_off, np.int64),
           "row_trial": cat(r_trial, np.int64), "row_window": cat(r_win, np.int64), "row_patch": cat(r_patch, np.int64),
           "row_n_measurement": cat(r_nm, np.float32)}
    for v in VARIANTS:
        out[v] = np.concatenate(feats[v]).astype(np.float32) if feats[v] else np.zeros((0, 0), np.float32)
    return out


def save_tokens(path, arrays, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.asarray(json.dumps(clean(meta))), **arrays)
    os.replace(tmp, path)
    with open(path[:-4] + ".json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(clean(meta), indent=1) + "\n")
    return path


def load_tokens(path):
    """(arrays, meta, index {(person, key): trial number}) of a token file."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arr = {k: z[k] for k in z.files if k != "meta"}
    index = {(str(p), str(k)): i for i, (p, k) in enumerate(zip(arr["trial_person"], arr["trial_key"]))}
    return arr, meta, index


def run_tokens(keys, limit=0, cache=None, log=print):
    """--tokens (module docstring): the inputs read once, then per checkpoint the token file (resumable)."""
    import collections

    from mdfm.baselines import readouts as RO
    from mdfm.train import provenance as PV
    check_prediction()
    sets = sets_of(limit)
    scored = {"AB28": sets["AB28"], "ST38": sets["ST38"]}
    lists = check_training_lists(sets["train"], scored, [END[k] for k in keys], exact=not limit)
    persons = sorted(set(sets["train"]) | set(sets["AB28"]) | set(sets["ST38"]))
    settings = token_settings()
    todo = []
    for k in keys:
        p = token_path(END[k], cache)
        if os.path.exists(p):
            _, meta, _ = load_tokens(p)
            if (meta.get("checkpoint") or {}).get("sha256") == END[k].sha256 and meta.get("settings") == settings \
                    and meta.get("persons_sha256") == _person_hash(persons):
                log(f"tokens {END[k].name}: kept ({rel(p)})", flush=True)
                continue
        todo.append(k)
    if not todo:
        return
    stats = collections.Counter()
    inputs = token_inputs(persons, stats, log)
    for k in todo:
        end = END[k]
        t0 = time.time()
        kind, model, info = RO.load_any(os.path.join(TREE, end.path), weights="ema", device="cpu")
        if info["sha256"] != end.sha256:
            raise AssertionError(f"{end.path}: SHA-256 {info['sha256']}, expected {end.sha256}")
        if kind != end.kind:
            raise AssertionError(f"{end.path}: loaded as {kind}, expected {end.kind}")
        trained = info["trained_persons"]
        for name, ps in scored.items():
            if set(ps) & trained:
                raise Refused(f"{name} persons among the checkpoint's training persons: {sorted(set(ps) & trained)[:3]}")
        if not set(sets["train"]) <= trained:
            raise AssertionError(f"{end.path}: fit persons outside its training persons")
        arrays = encode_inputs(model, inputs, settings["batch_windows"])
        meta = {"what": LABEL, "task": TASK, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": info["describe"], "kind": kind, "settings": settings, "persons": len(persons),
                "persons_sha256": _person_hash(persons), "limit_persons": limit, "trials": int(len(inputs)),
                "token_rows": int(len(arrays["row_trial"])), "frames": int(len(arrays["frame_t"])),
                "frames_uncovered": int((arrays["frame_row"] < 0).sum()), "read_stats": dict(stats),
                "training_lists": lists.get(k), "seconds": round(time.time() - t0, 1),
                "prediction_sha256": PREDICTION_SHA256, "moveatlas_head": PV.moveatlas_head()}
        save_tokens(token_path(end, cache), arrays, meta)
        log(f"tokens {end.name}: {meta['trials']} trials, {meta['token_rows']} rows, {meta['frames_uncovered']} frames "
            f"uncovered ({meta['seconds']} s)", flush=True)
        del model, arrays
        gc.collect()


# ---------------------------------------------------------------- masked multi-output ridge (token probes)
@dataclass
class MaskedRidge:
    """A fitted ridge of (N, D) features on (N, C) targets: predict(X) = ((X - mean) / scale) @ coef + intercept, NaN
    on a row with a non-finite feature; scale inf for a feature without variance (probes.Ridge's convention)."""
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        bad = ~np.isfinite(X).all(axis=1)
        out = ((np.where(bad[:, None], 0.0, X) - self.mean) / self.scale) @ self.coef + self.intercept
        out[bad] = np.nan
        return out


def masked_ridge_path(X, Y, alphas):
    """[MaskedRidge] at each alpha of features X (N, D) on targets Y (N, C) with missing entries (NaN). Rows with a
    non-finite feature or without any target are left out; features standardized over the rows kept (probes.Ridge's
    convention); each output column fitted on its own finite rows, centred there, by one eigendecomposition of its
    Gram matrix (the closed form of probes.ridge_path when no entry is missing). A column with fewer than 2 rows
    predicts its mean (0 without a row)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).any(axis=1)
    if keep.sum() < 2:
        raise ValueError(f"{int(keep.sum())} rows to fit")
    X, Y = X[keep], Y[keep]
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    scale = np.where(sd > 1e-12 * np.maximum(1.0, np.abs(mean)), sd, np.inf)
    Z = (X - mean) / scale
    D, C = Z.shape[1], Y.shape[1]
    alphas = [float(a) for a in alphas]
    coef = np.zeros((len(alphas), D, C))
    inter = np.zeros((len(alphas), C))
    for c in range(C):
        m = np.isfinite(Y[:, c])
        n = int(m.sum())
        if n < 2:
            inter[:, c] = float(Y[m, c].mean()) if n else 0.0
            continue
        Zc = Z[m]
        mu = Zc.mean(axis=0)
        A = Zc - mu
        y = Y[m, c]
        yb = float(y.mean())
        e, V = np.linalg.eigh(A.T @ A)
        e = np.maximum(e, 0.0)
        Vb = V.T @ (A.T @ (y - yb))
        for i, a in enumerate(alphas):
            w = V @ (Vb / (e + a))
            coef[i, :, c] = w
            inter[i, c] = yb - mu @ w
    return [MaskedRidge(a, mean, scale, coef[i], inter[i]) for i, a in enumerate(alphas)]


def masked_select_alpha(X, Y, persons, alphas=None, k=None, seed=None):
    """(alpha, {alpha: out-of-fold MSE}): the person folds of probes.cross_fit_splits inside the fitting rows, the
    squared error summed over every finite target entry of the held-out rows, divided by their count; the first
    minimum (the smallest alpha of equal values)."""
    from mdfm.eval import probes
    alphas = probes.ALPHAS if alphas is None else alphas
    k = probes.FOLDS if k is None else k
    seed = probes.SEED if seed is None else seed
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    p = np.asarray(persons).astype(str)
    sse = np.zeros(len(alphas))
    n = 0
    for tr, te in probes.cross_fit_splits(p, k, seed):
        okx = np.isfinite(X[te]).all(axis=1)
        te = te[okx]
        fin = np.isfinite(Y[te])
        if not fin.any():
            continue
        path = masked_ridge_path(X[tr], Y[tr], alphas)
        for i, m in enumerate(path):
            r = m.predict(X[te]) - Y[te]
            sse[i] += float((r[fin] ** 2).sum())
        n += int(fin.sum())
    if not n:
        raise ValueError("no out-of-fold target to select alpha")
    mse = sse / n
    i = int(np.argmin(mse))
    return float(alphas[i]), {float(a): float(v) for a, v in zip(alphas, mse)}


def masked_fit_probe(X, Y, persons, alphas=None, k=None, seed=None):
    """(MaskedRidge at the selected alpha fitted on every row, {'alpha', 'cv_mse'})."""
    from mdfm.eval import probes
    alphas = probes.ALPHAS if alphas is None else alphas
    a, cv = masked_select_alpha(X, Y, persons, alphas, k, seed)
    return masked_ridge_path(X, Y, (a,))[0], {"alpha": a, "cv_mse": cv}


def masked_cross_fit(X, Y, persons, k=None, seed=None):
    """(out-of-fold predictions (N, C), [alpha of each outer fold]): outer person folds (seed), the alpha chosen inside
    each fitting part by its own person folds (seed + 1), as probes.cross_fit."""
    from mdfm.eval import probes
    k = probes.FOLDS if k is None else k
    seed = probes.SEED if seed is None else seed
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    p = np.asarray(persons).astype(str)
    out = np.full(Y.shape, np.nan)
    chosen = []
    for tr, te in probes.cross_fit_splits(p, k, seed):
        m, sel = masked_fit_probe(X[tr], Y[tr], p[tr], None, k, seed + 1)
        out[te] = m.predict(X[te])
        chosen.append(sel["alpha"])
    return out, chosen


def token_design(arr, index, trials, scored, variant, n_off):
    """(X (R, D), Y (R, 2 n_off), persons (R,), rows (R,) token row numbers) of the fit trials: the token rows holding
    at least one scored sample, Y[r, foot n_off + j] the plate GRF (BW) of the frame at offset j of row r, NaN where
    that (frame, foot) is not scored. scored: {(person, key): (T, 2) bool}."""
    feats = arr[variant]
    rows, Ys, ps = {}, [], []
    for tr in trials:
        i = index.get((tr.person, tr.key))
        if i is None:
            continue
        a, n = int(arr["trial_start"][i]), int(arr["trial_len"][i])
        if n != len(tr.t) or not np.allclose(arr["frame_t"][a:a + n], tr.t, atol=1e-6):
            raise ValueError(f"{tr.person} {tr.key}: token frames differ from the trial's pose frames")
        row, off = arr["frame_row"][a:a + n], arr["frame_off"][a:a + n]
        S = scored[(tr.person, tr.key)]
        for f in np.flatnonzero((row >= 0) & S.any(axis=1)):
            r = int(row[f])
            if r not in rows:
                rows[r] = len(Ys)
                Ys.append(np.full(2 * n_off, np.nan))
                ps.append(tr.person)
            y = Ys[rows[r]]
            for foot in (0, 1):
                if S[f, foot]:
                    y[foot * n_off + int(off[f])] = float(tr.obs[f, foot])
    order = np.asarray(list(rows), dtype=np.int64)
    X = feats[order].astype(np.float64) if len(order) else np.zeros((0, feats.shape[1]))
    return X, (np.stack(Ys) if Ys else np.zeros((0, 2 * n_off))), np.asarray(ps, dtype=str), order


def token_frames(arr, index, trial, row_pred, n_off):
    """(T, 2) per-frame prediction of a trial from predictions per token row ({row: (2 n_off,)} as an array lookup:
    row_pred(rows) -> (len(rows), 2 n_off)); NaN where no row covers a frame."""
    out = np.full((len(trial.t), 2), np.nan)
    i = index.get((trial.person, trial.key))
    if i is None:
        return out
    a, n = int(arr["trial_start"][i]), int(arr["trial_len"][i])
    if n != len(trial.t) or not np.allclose(arr["frame_t"][a:a + n], trial.t, atol=1e-6):
        raise ValueError(f"{trial.person} {trial.key}: token frames differ from the trial's pose frames")
    row, off = arr["frame_row"][a:a + n], arr["frame_off"][a:a + n]
    cov = row >= 0
    if not cov.any():
        return out
    P = row_pred(row[cov])
    j = off[cov]
    out[cov, 0] = P[np.arange(len(j)), j]
    out[cov, 1] = P[np.arange(len(j)), n_off + j]
    return out


# ---------------------------------------------------------------- supervised features (b1, b2)
L_HIP, R_HIP = 9, 10
ANKLES = (13, 14)
UP = 1
MIRROR = (0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13)   # CANONICAL_15 left and right swapped
BASE_NAMES = ([f"rel_{j}_{a}" for j in range(1, 15) for a in "ful"] + ["pelvis_h", "ankle_h_own", "ankle_h_other"]
              + [f"vel_{j}_{a}" for j in range(15) for a in "ful"] + [f"com_v_{a}" for a in "ful"]
              + [f"com_a_{a}" for a in "ful"] + ["contact_own", "contact_other", "newton_bw_own", "newton_total"])
N_BASE = len(BASE_NAMES)                                       # 100


def heading_frames(y):
    """(T, 3, 3) rows forward, up, lateral of each frame: lateral the horizontal part of right hip minus left hip,
    normalized; forward = up x lateral; NaN where the hips are missing or coincide horizontally."""
    lat = np.array(y[:, R_HIP] - y[:, L_HIP], dtype=np.float64)
    lat[:, UP] = 0.0
    n = np.linalg.norm(lat, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.where(n > 1e-6, lat / n, np.nan)
    up = np.zeros_like(lat)
    up[:, UP] = 1.0
    fwd = np.cross(up, lat)
    return np.stack([fwd, up, lat], axis=1)


def to_heading(R, v):
    """v (T, ..., 3) in the lab frame expressed in the heading frames R (T, 3, 3): (forward, up, lateral)."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 2:
        return np.einsum("tij,tj->ti", R, v)
    return np.einsum("tij,tkj->tki", R, v)


def base_features(trial):
    """(F (T, 2, N_BASE), ok (T,), newton (T, 2)): per frame and foot the base features (BASE_NAMES; prediction.txt
    section 2 (b)) from the trial's world joints, NaN where the Newton floor is undefined (ok False). The world joints
    are low-passed as g4.newton_floor does (12 Hz on runs of at least 45 valid frames, g4.lowpass_runs), velocities
    and the centre of mass's velocity and acceleration by central differences (force.com15, the recorded sex), heights
    above the trial's stance height (the 5th percentile of both ankles), the floor's contact probabilities and BW
    (g4.newton_floor). Foot 1 (right) is the mirror image: left and right joints swapped, lateral coordinates
    negated, own and other foot swapped."""
    from mdfm.gates import g4
    from mdfm.kernels import force
    world = np.asarray(trial.world, dtype=np.float64)
    T = world.shape[0]
    fps = float(trial.fps)
    frame_ok = np.isfinite(world).all(axis=(1, 2))
    yf, kept = g4.lowpass_runs(world.reshape(T, -1), frame_ok, fps)
    y = yf.reshape(T, 15, 3)
    vel, _, ok = g4._central(y, kept, fps)
    com = force.com15(y, trial.sex)
    cv, ca, _ = g4._central(com, kept, fps)
    newton, ok_n, p = g4.newton_floor(world, fps, trial.sex)
    if not np.array_equal(ok, ok_n):
        raise AssertionError(f"{trial.person} {trial.key}: feature frames differ from the Newton floor's")
    height = y[:, ANKLES, UP]
    stance = np.percentile(height[kept], g4.STANCE_PCT) if kept.any() else np.nan
    R = heading_frames(y)
    rel15 = to_heading(R, y - y[:, :1])
    velh = to_heading(R, vel)
    cvh, cah = to_heading(R, cv), to_heading(R, ca)
    total = 1.0 + ca[:, UP] / force.G
    out = np.full((T, 2, N_BASE), np.nan)
    for k in (0, 1):
        perm = np.arange(15) if k == 0 else np.asarray(MIRROR)
        sgn = np.array([1.0, 1.0, 1.0 if k == 0 else -1.0])
        own, other = (0, 1) if k == 0 else (1, 0)
        parts = [(rel15[:, perm][:, 1:] * sgn).reshape(T, -1),
                 (y[:, 0, UP] - stance)[:, None], (height[:, own] - stance)[:, None], (height[:, other] - stance)[:, None],
                 (velh[:, perm] * sgn).reshape(T, -1), cvh * sgn, cah * sgn,
                 p[:, own, None], p[:, other, None], newton[:, own, None], total[:, None]]
        out[:, k] = np.concatenate(parts, axis=1)
    out[~ok] = np.nan
    return out, ok, newton


def with_context(F, ok, fps, context_s=CONTEXT_S):
    """(T, 2, len(context_s) N_BASE): the base features at each context offset round(c fps), the offset index clamped
    into the frame's run of Newton-defined frames; NaN outside those runs."""
    from mdfm.gates import g4
    T = F.shape[0]
    out = np.full((T, 2, len(context_s) * F.shape[2]), np.nan)
    for s, e in g4._runs(np.asarray(ok, dtype=bool)):
        idx = np.arange(s, e)
        blocks = [F[np.clip(idx + int(round(c * fps)), s, e - 1)] for c in context_s]
        out[s:e] = np.concatenate(blocks, axis=-1)
    return out


def kinematic_features(trial):
    """(features (T, 2, 500), ok (T,), newton (T, 2)) of a GRFTrial: base_features with_context."""
    F, ok, newton = base_features(trial)
    return with_context(F, ok, trial.fps), ok, newton


def scored_mask(trial, newton):
    """(T, 2) bool: the harness's scored samples of the Newton floor, plate valid where the floor is finite."""
    return np.asarray(trial.obs_valid, dtype=bool) & np.isfinite(newton)


def supervised_design(trials):
    """(X (n, 500) float64, y (n,), persons (n,), where [(trial number, frame, foot)]) of the scored samples of the fit
    trials, both feet stacked."""
    Xs, ys, ps, where = [], [], [], []
    for i, tr in enumerate(trials):
        F, ok, newton = kinematic_features(tr)
        S = scored_mask(tr, newton)
        for foot in (0, 1):
            f = np.flatnonzero(S[:, foot])
            if not len(f):
                continue
            Xs.append(F[f, foot])
            ys.append(tr.obs[f, foot])
            ps += [tr.person] * len(f)
            where += [(i, int(x), foot) for x in f]
    X = np.concatenate(Xs).astype(np.float64) if Xs else np.zeros((0, len(CONTEXT_S) * N_BASE))
    return X, (np.concatenate(ys) if ys else np.zeros(0)), np.asarray(ps, dtype=str), where


def hgb(params=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**(SB().HGB_PARAMS if params is None else params))


def predict_supervised(models, trials):
    """{name: {(person, key): (T, 2)}} of fitted supervised models {name: model with predict} at the frames of trials
    holding a plate-valid sample where the features are finite (NaN elsewhere; those frames are never scored)."""
    out = {name: {} for name in models}
    for tr in trials:
        F, ok, _ = kinematic_features(tr)
        use = ok & np.asarray(tr.obs_valid, dtype=bool).any(axis=1)
        for name, m in models.items():
            y = np.full((len(tr.t), 2), np.nan)
            f = np.flatnonzero(use)
            if len(f):
                for foot in (0, 1):
                    y[f, foot] = np.asarray(m.predict(F[f, foot]), dtype=np.float64).reshape(-1)
            out[name][(tr.person, tr.key)] = y
    return out


# ---------------------------------------------------------------- fallback, tables, lines
def fill_scored(pred, trials, scored, fill_value):
    """pred {(person, key): (T, 2)} with every scored sample finite: a non-finite prediction at a scored sample takes
    fill_value (a number, or (T, 2) -> value per foot via a callable of (trial, foot, frames)); returns (pred, count)."""
    n = 0
    for tr in trials:
        k = (tr.person, tr.key)
        y = np.array(pred.get(k, np.full((len(tr.t), 2), np.nan)), dtype=np.float64)
        S = scored[k]
        bad = S & ~np.isfinite(y)
        if bad.any():
            for foot in (0, 1):
                f = np.flatnonzero(bad[:, foot])
                if len(f):
                    y[f, foot] = fill_value(tr, foot, f) if callable(fill_value) else float(fill_value)
            n += int(bad.sum())
        pred[k] = y
    return pred, n


def predictor_of(pred):
    """callable(GRFTrial) -> (T, 2) from {(person, key): (T, 2)}."""
    def f(trial):
        return pred[(trial.person, trial.key)]
    return f


def person_table(trials, preds):
    """g4.grf_person_table with the Newton floor and the predictions {name: {(person, key): (T, 2)}}."""
    from mdfm.gates import g4
    return g4.grf_person_table(trials, {"newton": g4.newton_predict, **{k: predictor_of(v) for k, v in preds.items()}})


def lines_of(table, names):
    """{name: {grf_rmse, grf_r, grf_r_gain: {point, interval, n_persons}}} by g4.summary (the Newton floor's r gain
    absent)."""
    import pandas as pd

    from mdfm.gates import g4
    s = g4.summary(table, pd.DataFrame(), ["newton", *names], [])
    out = {}
    for k in ["newton", *names]:
        out[k] = {"grf_rmse": s["grf_rmse_pct_bw"][k], "grf_r": s["grf_r"][k]}
        if k != "newton":
            out[k]["grf_r_gain"] = s["grf_r_gain"][k]
    return out


def per_person_values(table, name, line):
    """(n_persons,) the per-person value of a line: grf_rmse, the RMSE in percent BW; grf_r_gain, r minus the Newton
    floor's r."""
    if line == "grf_rmse":
        return table[f"{name}_rmse_pct_bw"].to_numpy(np.float64)
    return table[f"{name}_r"].to_numpy(np.float64) - table["newton_r"].to_numpy(np.float64)


def spread(points):
    pts = [float(x) for x in points]
    return float(max(pts) - min(pts)) if len(pts) > 1 else 0.0


def premium_test(table, body_names, comp_names, line):
    """The premium rule of prediction.txt section 3 for one line on one set's person table: separation, the paired
    person bootstrap interval (g4.person_bootstrap of the per-person differences, mean over persons) and the margin."""
    from mdfm.gates import g4
    lower = line == "grf_rmse"
    B = np.stack([per_person_values(table, n, line) for n in body_names])
    Cm = np.stack([per_person_values(table, n, line) for n in comp_names])
    ok = np.isfinite(B).all(axis=0) & np.isfinite(Cm).all(axis=0)
    bpts = [float(np.mean(b[ok])) for b in B]
    cpts = [float(np.mean(c[ok])) for c in Cm]
    diff = B[:, ok].mean(axis=0) - Cm[:, ok].mean(axis=0)
    point = float(diff.mean()) if len(diff) else float("nan")
    lo, hi = g4.person_bootstrap(diff, np.mean)
    d = max(spread(bpts), spread(cpts))
    cmean = float(np.mean(cpts))
    m = max(MARGIN_FRACTION * cmean, 2.0 * d) if lower else max(g4.R_GAIN, 2.0 * d)

    def better(a, b):
        return a < b if lower else a > b

    sep = all(better(b, c) for b in bpts for c in cpts)
    sep_worse = all(better(c, b) for b in bpts for c in cpts)
    iv_better = (hi < 0) if lower else (lo > 0)
    iv_worse = (lo > 0) if lower else (hi < 0)
    margin = (point <= -m) if lower else (point >= m)
    premium = bool(sep and iv_better and margin)
    loss = bool(sep_worse and iv_worse)
    return {"line": line, "body": list(body_names), "comparator": list(comp_names), "body_points": bpts,
            "comparator_points": cpts, "body_mean": float(np.mean(bpts)), "comparator_mean": cmean,
            "difference": point, "interval": [lo, hi], "persons": int(ok.sum()), "seed_spread": d, "margin": m,
            "needs_body_mean": cmean - m if lower else cmean + m, "separation": bool(sep),
            "separation_worse": bool(sep_worse), "interval_better": bool(iv_better), "interval_worse": bool(iv_worse),
            "margin_met": bool(margin), "premium": premium, "loss": loss,
            "reading": "PREMIUM" if premium else ("LOSS" if loss else "NO PREMIUM")}


def best_bar(lines, bars, line):
    """The bar (a key of bars) with the best seed-mean point on a line ({predictor: lines}); ties to the first."""
    lower = line == "grf_rmse"
    best, key = None, None
    for b, names in bars.items():
        v = float(np.mean([lines[n][line]["point"] for n in names]))
        k = v if lower else -v
        if best is None or k < key:
            best, key = b, k
    return best


def set_reading(tests):
    """PREMIUM when both lines read PREMIUM, LOSS when both read LOSS, NO PREMIUM otherwise."""
    r = [tests[ln]["reading"] for ln in LINES]
    return "PREMIUM" if all(x == "PREMIUM" for x in r) else ("LOSS" if all(x == "LOSS" for x in r) else "NO PREMIUM")


# ---------------------------------------------------------------- the body's rows
def body_rows(set_name, persons, tree=TREE):
    """{key: DataFrame person_id, samples, newton_rmse_pct_bw, newton_r, model_rmse_pct_bw, model_r} of the body
    seeds' force-head rows on a set (BODY_REPORTS: per_person_grf_able_bodied.csv for AB28, per_person_grf.csv for
    ST38 and its subset ST7)."""
    import pandas as pd
    out = {}
    for k, d in BODY_REPORTS.items():
        fn = "per_person_grf_able_bodied.csv" if set_name == "AB28" else "per_person_grf.csv"
        t = pd.read_csv(os.path.join(tree, d, fn), encoding="utf-8")
        t = t[t["person_id"].isin(set(persons))].reset_index(drop=True)
        out[k] = t
    return out


def attach_body(table, rows):
    """The person table with the body seeds' columns body_<key>_rmse_pct_bw and body_<key>_r, after the checks: the
    same persons, per person the same samples and the Newton floor's RMSE and r within CHECK_TOL. Returns (table,
    checks)."""
    checks = {}
    t = table.copy()
    for k, r in rows.items():
        m = t[["person_id", "samples", "newton_rmse_pct_bw", "newton_r"]].merge(r, on="person_id", how="outer",
                                                                              suffixes=("", "_report"),
                                                                              indicator=True)
        both = m["_merge"] == "both"
        same_persons = bool(both.all())
        samples = bool((m.loc[both, "samples"] == m.loc[both, "samples_report"]).all())
        dr = float(np.nanmax(np.abs(m.loc[both, "newton_rmse_pct_bw"] - m.loc[both, "newton_rmse_pct_bw_report"])))
        dc = float(np.nanmax(np.abs(m.loc[both, "newton_r"] - m.loc[both, "newton_r_report"])))
        checks[k] = {"persons": int(both.sum()), "same_persons": same_persons, "same_samples": samples,
                     "newton_rmse_max_abs_diff": dr, "newton_r_max_abs_diff": dc,
                     "passed": bool(same_persons and samples and dr <= CHECK_TOL and dc <= CHECK_TOL)}
        if not checks[k]["passed"]:
            raise AssertionError(f"body rows {k}: the samples differ from the report's ({checks[k]})")
        mm = r.set_index("person_id")
        t[f"body_{k}_rmse_pct_bw"] = t["person_id"].map(mm["model_rmse_pct_bw"]).astype(float)
        t[f"body_{k}_r"] = t["person_id"].map(mm["model_r"]).astype(float)
    return t, checks


# ---------------------------------------------------------------- the read
def fit_tokens(end, train_trials, sets_trials, scored, n_off, cache=None, s_trials=None, log=print):
    """The token probe of one checkpoint (prediction.txt section 2 (c)): per variant the person cross-fitted lines on
    the fit set (out-of-fold row predictions, every variant on the same samples) and the final fit on the fit set;
    the chosen variant (CANDIDATES, lower cross-fitted grf_rmse); predictions on every scored trial; the S arm
    (s_trials: the fit set plus the stage-2 stroke train fold) with the chosen variant. Returns (result dict, {variant:
    {(person, key): (T, 2)}} on the scored trials, S-arm predictions or None)."""
    arr, meta, index = load_tokens(token_path(end, cache))
    if (meta.get("checkpoint") or {}).get("sha256") != end.sha256:
        raise SystemExit(f"{rel(token_path(end, cache))} holds another checkpoint's tokens")
    oof, final, info = {}, {}, {}
    for v in VARIANTS:
        t0 = time.time()
        X, Y, ps, rows = token_design(arr, index, train_trials, scored, v, n_off)
        P_oof, alphas = masked_cross_fit(X, Y, ps)
        look = np.full(len(arr["row_trial"]), -1, np.int64)
        look[rows] = np.arange(len(rows))
        oof[v] = {}
        for tr in train_trials:
            def rp(r, P=P_oof, L=look):
                i = L[r]
                o = np.full((len(r), P.shape[1]), np.nan)
                o[i >= 0] = P[i[i >= 0]]
                return o
            oof[v][(tr.person, tr.key)] = token_frames(arr, index, tr, rp, n_off)
        m, sel = masked_fit_probe(X, Y, ps)
        col_mean = np.array([np.nanmean(Y[:, c]) if np.isfinite(Y[:, c]).any() else 0.0 for c in range(Y.shape[1])])
        final[v] = (m, col_mean)
        info[v] = {"alpha": sel["alpha"], "crossfit_alphas": alphas, "fit_rows": int(len(X)),
                   "fit_targets": int(np.isfinite(Y).sum()), "seconds": round(time.time() - t0, 1)}
        log(f"  tokens {end.key} {v}: {len(X)} rows, alpha {sel['alpha']:g} ({info[v]['seconds']} s)", flush=True)
    cf_table = person_table(train_trials, {v: oof[v] for v in VARIANTS})
    cf_lines = lines_of(cf_table, list(VARIANTS))
    crossfit = {v: {"grf_rmse": cf_lines[v]["grf_rmse"]["point"], "grf_r_gain": cf_lines[v]["grf_r_gain"]["point"],
                    "persons": cf_lines[v]["grf_rmse"]["n_persons"]} for v in VARIANTS}
    chosen = min(CANDIDATES, key=lambda v: (not np.isfinite(crossfit[v]["grf_rmse"]),
                                             crossfit[v]["grf_rmse"] if np.isfinite(crossfit[v]["grf_rmse"]) else 0.0,
                                             CANDIDATES.index(v)))

    def predict_all(model, col_mean, variant, trials):
        feats = arr[variant]
        pred = {}
        for tr in trials:
            def rp(r, M=model, Fe=feats):
                return M.predict(Fe[r].astype(np.float64))
            pred[(tr.person, tr.key)] = token_frames(arr, index, tr, rp, n_off)
        filled, n_fill = fill_scored(pred, trials, scored,
                                     lambda trial, foot, f: col_mean[foot * n_off + np.clip(_offsets(arr, index, trial,
                                                                                                     f), 0, n_off - 1)])
        return filled, n_fill

    preds, fills = {}, {}
    all_scored = [tr for trs in sets_trials.values() for tr in trs]
    for v in VARIANTS:
        preds[v], fills[v] = predict_all(final[v][0], final[v][1], v, all_scored)
    s_pred, s_info = None, None
    if s_trials is not None:
        X, Y, ps, _ = token_design(arr, index, s_trials, scored, chosen, n_off)
        m, sel = masked_fit_probe(X, Y, ps)
        col_mean = np.array([np.nanmean(Y[:, c]) if np.isfinite(Y[:, c]).any() else 0.0 for c in range(Y.shape[1])])
        s_pred, s_fill = predict_all(m, col_mean, chosen, sets_trials["ST7"])
        s_info = {"alpha": sel["alpha"], "fit_rows": int(len(X)), "fit_targets": int(np.isfinite(Y).sum()),
                  "fallback_samples": s_fill}
    res = {"key": end.key, "arm": end.arm, "seed": end.seed, "checkpoint": meta.get("checkpoint"),
           "token_file": {"path": rel(token_path(end, cache)), "sha256": sha256_file(token_path(end, cache)),
                          "written_at": meta.get("written_at"), "moveatlas_head": meta.get("moveatlas_head"),
                          "frames_uncovered": meta.get("frames_uncovered"), "trials": meta.get("trials")},
           "variants": info, "crossfit": crossfit, "chosen": chosen, "fallback_samples": fills, "s_arm": s_info}
    return res, preds, s_pred


def _offsets(arr, index, trial, frames):
    """The patch offsets of some frames of a trial (0 where no row covers a frame)."""
    i = index.get((trial.person, trial.key))
    if i is None:
        return np.zeros(len(frames), np.int64)
    a = int(arr["trial_start"][i])
    off = arr["frame_off"][a + np.asarray(frames, dtype=np.int64)]
    return np.where(off >= 0, off, 0)


def run_read(limit=0, out=None, cache=None, log=print):
    """The read (module docstring)."""
    import collections

    import pandas as pd

    from mdfm.eval import probes
    from mdfm.gates import common, g4
    from mdfm.train import provenance as PV
    pred_sha = check_prediction()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    out = out or OUT
    reads_before = sha256_file(common.READ_LOG) if os.path.exists(common.READ_LOG) else None
    sets = sets_of(limit)
    lists = check_training_lists(sets["train"], {"AB28": sets["AB28"], "ST38": sets["ST38"]},
                                 exact=not limit)
    stats = {k: collections.Counter() for k in ("train", "AB28", "ST38", "st32")}
    left = []
    trials = {k: g4.grf_trials(sets[k], stats=stats[k], left_out=left) for k in ("train", "AB28", "ST38", "st32")}
    trials["ST7"] = [tr for tr in trials["ST38"] if tr.person in set(sets["ST7"])]
    fps = {float(tr.fps) for trs in trials.values() for tr in trs}
    if len(fps) != 1:
        raise AssertionError(f"pose rates {sorted(fps)}: one rate expected")
    n_off = int(round(PATCH_S * fps.pop()))
    scored = {}
    for trs in trials.values():
        for tr in trs:
            scored[(tr.person, tr.key)] = scored_mask(tr, g4.newton_predict(tr))
    counts = {k: {"persons": len(sets[k]), "persons_with_trials": len({t.person for t in trials[k]}),
                  "trials": len(trials[k]), "scored_samples": int(sum(scored[(t.person, t.key)].sum() for t in trials[k]))}
              for k in trials}
    log("trials: " + ", ".join(f"{k} {v['trials']} ({v['scored_samples']} samples)" for k, v in counts.items()),
        flush=True)
    dev_trials = {s: trials[s] for s in SETS}
    all_scored = [tr for s in ("AB28", "ST38") for tr in trials[s]]

    # supervised bars
    t0 = time.time()
    X, y, ps, where = supervised_design(trials["train"])
    b1, sel1 = probes.fit_probe(X, y, ps)
    b2 = hgb().fit(X, y)
    oof1, alphas1 = probes.cross_fit(X, y, ps)
    oof2 = np.full(len(y), np.nan)
    for tr_i, te_i in probes.cross_fit_splits(ps):
        oof2[te_i] = hgb().fit(X[tr_i], y[tr_i]).predict(X[te_i])
    cf = {"b1": {}, "b2": {}}
    for name, o in (("b1", oof1), ("b2", oof2)):
        for tr in trials["train"]:
            cf[name][(tr.person, tr.key)] = np.full((len(tr.t), 2), np.nan)
        for (i, f, foot), v in zip(where, o):
            tr = trials["train"][i]
            cf[name][(tr.person, tr.key)][f, foot] = v
    sup_cf = lines_of(person_table(trials["train"], cf), ["b1", "b2"])
    y_mean = float(np.mean(y))
    n_fit = {"rows": int(len(y)), "persons": int(len(set(ps))), "features": int(X.shape[1])}
    del X
    gc.collect()
    Xs, ys, pss, _ = supervised_design(trials["train"] + trials["st32"])
    b1s, sel1s = probes.fit_probe(Xs, ys, pss)
    b2s = hgb().fit(Xs, ys)
    ys_mean = float(np.mean(ys))
    n_fit_s = {"rows": int(len(ys)), "persons": int(len(set(pss)))}
    del Xs
    gc.collect()
    sup = predict_supervised({"b1": b1, "b2": b2}, all_scored)
    sup_s = predict_supervised({"b1": b1s, "b2": b2s}, trials["ST7"])
    fills = {}
    for name in ("b1", "b2"):
        sup[name], fills[name] = fill_scored(sup[name], all_scored, scored, y_mean)
        sup_s[name], fills[f"{name}_s"] = fill_scored(sup_s[name], trials["ST7"], scored, ys_mean)
    sup_info = {"b1": {"alpha": sel1["alpha"], "crossfit_alphas": alphas1}, "b2": {"params": SB().HGB_PARAMS},
                "fit": n_fit, "fit_s_arm": n_fit_s, "b1_s_alpha": sel1s["alpha"],
                "crossfit": {k: {ln: sup_cf[k][ln]["point"] for ln in ("grf_rmse", "grf_r_gain")} for k in ("b1", "b2")},
                "seconds": round(time.time() - t0, 1)}
    log(f"supervised bars: fitted on {n_fit['rows']} samples of {n_fit['persons']} persons ({sup_info['seconds']} s)",
        flush=True)

    # token bars
    tok, tok_pred, tok_s = {}, {}, {}
    for e in ENDS:
        res, preds, s_pred = fit_tokens(e, trials["train"], dev_trials, scored, n_off, cache,
                                        s_trials=trials["train"] + trials["st32"], log=log)
        tok[e.key] = res
        tok_pred[f"tok_{e.key}"] = preds[res["chosen"]]
        for v in VARIANTS:
            tok_pred[f"tokvar_{e.key}_{v}"] = preds[v]
        tok_s[f"tok_{e.key}"] = s_pred
        log(f"tokens {e.key}: chosen {res['chosen']} (cross-fitted {res['crossfit'][res['chosen']]['grf_rmse']:.3f})",
            flush=True)

    # tables, lines, tests
    names = ["b1", "b2", *[f"tok_{e.key}" for e in ENDS]]
    var_names = [f"tokvar_{e.key}_{v}" for e in ENDS for v in VARIANTS]
    tables, lines, checks, tests, group_tests, readings = {}, {}, {}, {}, {}, {}
    for s in SETS:
        preds = {**{k: sup[k] for k in ("b1", "b2")}, **{k: tok_pred[k] for k in [*names[2:], *var_names]}}
        t = person_table(trials[s], preds)
        t, checks[s] = attach_body(t, body_rows(s, sets[s]))
        tables[s] = t
        lines[s] = lines_of(t, [*names, *BODY, *var_names])
        tests[s] = {}
        group_tests[s] = {}
        for ln in LINES:
            comp = best_bar(lines[s], BARS, ln)
            tests[s][ln] = {"comparator_bar": comp, **premium_test(t, BODY, BARS[comp], ln)}
            group_tests[s][ln] = {}
            for g, members in GROUPS.items():
                gb = best_bar(lines[s], {b: BARS[b] for b in members}, ln)
                group_tests[s][ln][g] = {"comparator_bar": gb, **premium_test(t, BODY, BARS[gb], ln)}
        readings[s] = set_reading(tests[s])
    # S arm on ST7
    s_preds = {"b1_s": sup_s["b1"], "b2_s": sup_s["b2"], **{f"{k}_s": v for k, v in tok_s.items()}}
    t_s = person_table(trials["ST7"], s_preds)
    t_s, checks["ST7_s_arm"] = attach_body(t_s, body_rows("ST7", sets["ST7"]))
    s_names = list(s_preds)
    lines_s = lines_of(t_s, [*s_names, *BODY])
    bars_s = {f"{b}_s": tuple(f"{n}_s" for n in members) for b, members in BARS.items()}
    tests_s = {}
    for ln in LINES:
        comp = best_bar(lines_s, bars_s, ln)
        tests_s[ln] = {"comparator_bar": comp, **premium_test(t_s, BODY, bars_s[comp], ln)}
    # strokefold beside
    sf = pd.read_csv(os.path.join(TREE, STROKEFOLD_REPORT, "per_person_grf.csv"), encoding="utf-8")
    strokefold = {"grf_rmse": float(sf["model_rmse_pct_bw"].mean()),
                  "grf_r_gain": float((sf["model_r"] - sf["newton_r"]).mean()), "persons": int(len(sf))}
    reads_after = sha256_file(common.READ_LOG) if os.path.exists(common.READ_LOG) else None
    res = {"what": LABEL, "task": TASK, "mode": "read", "run_started": started, "partial": bool(limit),
           "prediction": {"path": rel(PREDICTION_PATH), "sha256": pred_sha},
           "sets": {k: {"persons": len(v), "sha256": _person_hash(v)} for k, v in sets.items()},
           "training_lists": lists, "counts": counts, "pairing": {k: dict(v) for k, v in stats.items()},
           "left_out": left, "n_offsets": n_off, "supervised": sup_info, "fallback_samples": fills,
           "tokens": tok, "lines": lines, "checks": checks, "tests": tests, "group_tests": group_tests,
           "readings": readings, "s_arm": {"lines": lines_s, "tests": tests_s},
           "strokefold_v1_B": strokefold,
           "lockbox": {"read_lockbox_called": False, "lockbox_reads_csv_sha256_before": reads_before,
                       "lockbox_reads_csv_sha256_after": reads_after, "unchanged": reads_before == reads_after},
           "moveatlas_head": PV.moveatlas_head()}
    res["predictions"] = predictions_check(res)
    res["timing_s"] = round(time.time() - t_start, 1)
    os.makedirs(out, exist_ok=True)
    pp = []
    for s in SETS:
        pp.append(tables[s].assign(set=s))
    pp.append(t_s.assign(set="ST7_s_arm"))
    per_person = pd.concat(pp, ignore_index=True)
    per_person = per_person[["set"] + [c for c in per_person.columns if c != "set"]]
    pp_path = common.write_csv(per_person, os.path.join(out, "per_person.csv"))
    res["per_person"] = {"path": rel(pp_path), "sha256": sha256_file(pp_path), "rows": int(len(per_person))}
    common.write_text(os.path.join(out, "results.json"), json.dumps(clean(res), indent=1) + "\n")
    common.write_text(os.path.join(out, "summary.md"), summary_md(res))
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(provenance(started)), indent=1) + "\n")
    log(f"written {rel(out)} ({res['timing_s']} s)", flush=True)
    return res


# ---------------------------------------------------------------- predictions (prediction.txt section 4)
def within(lo, hi, x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and lo <= v <= hi


def predictions_check(res):
    """P1 to P9 of prediction.txt, each met or missed, with the values read."""
    L = res["lines"]

    def pt(s, n, ln):
        return L[s][n][ln]["point"]

    tok_seeds = {arm: [pt("AB28", f"tok_{k}", "grf_rmse") for k in keys] for arm, keys in TOKEN_ARMS.items()}
    out = {}
    v = (pt("AB28", "b2", "grf_rmse"), pt("AB28", "b2", "grf_r_gain"))
    out["P1"] = {"what": "b2 on AB28: grf_rmse in [10, 17], grf_r_gain in [0.30, 0.39]",
                 "met": within(10, 17, v[0]) and within(0.30, 0.39, v[1]), "value": v}
    v = (pt("AB28", "b1", "grf_rmse"),)
    out["P2"] = {"what": "b1 on AB28: grf_rmse in [15, 24]", "met": within(15, 24, v[0]), "value": v}
    v = tuple(tok_seeds["tok_body"] + tok_seeds["tok_mfr"])
    out["P3"] = {"what": "body and MFR token probes on AB28: every seed's grf_rmse in [14, 22]",
                 "met": all(within(14, 22, x) for x in v), "value": v}
    v = tuple(tok_seeds["tok_jepa"])
    out["P4"] = {"what": "JEPA token probes on AB28: both seeds' grf_rmse in [16, 26]",
                 "met": all(within(16, 26, x) for x in v), "value": v}
    v = (pt("ST38", "b2", "grf_rmse"), pt("ST38", "b2", "grf_r_gain"))
    out["P5"] = {"what": "b2 on ST38: grf_rmse in [18, 26], grf_r_gain in [0.05, 0.15]",
                 "met": within(18, 26, v[0]) and within(0.05, 0.15, v[1]), "value": v}
    v = tuple(pt("ST38", f"tok_{e.key}", "grf_rmse") for e in ENDS)
    out["P6"] = {"what": "token probes on ST38: every seed's grf_rmse in [20, 28]",
                 "met": all(within(20, 28, x) for x in v), "value": v}
    t = res["tests"]
    v = (t["AB28"]["grf_rmse"]["reading"], t["AB28"]["grf_r_gain"]["reading"])
    out["P7"] = {"what": "readings on AB28: LOSS on grf_rmse and on grf_r_gain", "met": v == ("LOSS", "LOSS"),
                 "value": v}
    v = (t["ST38"]["grf_rmse"]["reading"], t["ST38"]["grf_r_gain"]["reading"])
    out["P8"] = {"what": "readings on ST38: not PREMIUM on either line", "met": "PREMIUM" not in v, "value": v}
    v = (res["s_arm"]["lines"]["b2_s"]["grf_rmse"]["point"], pt("ST7", "b2", "grf_rmse"))
    out["P9"] = {"what": "S arm: b2 refitted with the stroke train patients reads a lower grf_rmse on ST7 than b2 of "
                         "the main arm",
                 "met": bool(np.isfinite(v[0]) and np.isfinite(v[1]) and v[0] < v[1]), "value": v}
    return out


# ---------------------------------------------------------------- output helpers
def _person_hash(persons):
    from mdfm.data import sources
    return sources.person_hash(sorted(persons))


def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    if not iv or fin(iv[0]) is None:
        return "n/a"
    return f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [clean(v) for v in (sorted(o) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def code_files():
    from mdfm.baselines import readouts
    from mdfm.eval import probes
    from mdfm.gates import g4
    return [os.path.abspath(__file__), g4.__file__, readouts.__file__, probes.__file__,
            os.path.join(HERE, "eval_g4_dev.py"), os.path.join(HERE, "g3_supervised_baseline.py")]


def provenance(started):
    import torch

    from mdfm import paths
    from mdfm.gates import common
    from mdfm.train import provenance as PV
    files = code_files()
    return {"script": "scripts/g4_grf_bars.py", "what": LABEL, "task": TASK, "run_started": started,
            "run_finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_head": PV.moveatlas_head(), "python": sys.version.split()[0], "torch": torch.__version__,
            "prediction_sha256": PREDICTION_SHA256, "code_sha256": common.code_sha256(files),
            "code_files": [common.rel(f) for f in files], "device": "cpu",
            "inputs": ["MoveAtlas canonical pose and force records of stroke_gait_vancriekinge (train fold, able-bodied "
                       "dev fold, the 40 non-lockbox stroke patients), reader flags False",
                       "the raw root tracks of mdfm.data.mocap_raw", "the token files of cache/g4_grf/",
                       "the body's per-person rows of " + ", ".join(BODY_REPORTS.values())]}


def summary_md(r):
    L = r["lines"]
    T = r["tests"]
    rows_names = [("newton", "Newton floor (G4 baseline, no labels)"),
                  ("b1", "b1 supervised ridge, kinematic features"),
                  ("b2", "b2 supervised gradient boosting, kinematic features"),
                  ("tok_v1_B", "token probe, body v1_B"), ("tok_s2_B", "token probe, body s2_B"),
                  ("tok_mfr_s1", "token probe, MFR seed 20260928"), ("tok_mfr_s2", "token probe, MFR seed 20261001"),
                  ("tok_jepa_s1", "token probe, JEPA seed 20260928"),
                  ("tok_jepa_s2", "token probe, JEPA seed 20261001"),
                  ("body_v1_B", "body force head v1_B (report)"), ("body_s2_B", "body force head s2_B (report)")]
    out = [f"# GRF bars: learned bars for ground reaction force (G4)", "",
           f"{LABEL[0].upper() + LABEL[1:]}. Written by scripts/g4_grf_bars.py; every number below is read from results.json and "
           f"per_person.csv of this folder. Prediction and method: {r['prediction']['path']} (SHA-256 "
           f"{r['prediction']['sha256']}, asserted before any target, record or checkpoint was read). No lockbox, EVAL, "
           f"CODY or care_pd record (read log unchanged: {r['lockbox']['unchanged']}). MoveAtlas HEAD "
           f"{r['moveatlas_head']}." + (" PARTIAL RUN (limit-persons)." if r["partial"] else ""), ""]
    for s in SETS:
        c = r["counts"][s]
        out += [f"## {s}: {c['persons_with_trials']} persons, {c['trials']} trials, {c['scored_samples']} scored samples"
                + (" (beside, deciding nothing)" if s not in DECIDING else ""), "",
                "| row | grf_rmse (% BW) | 95 % | grf_r | grf_r_gain | 95 % |", "|---|---|---|---|---|---|"]
        for n, label in rows_names:
            x = L[s][n]
            g = x.get("grf_r_gain")
            out.append(f"| {label} | {fmt(x['grf_rmse']['point'])} | {fmt_iv(x['grf_rmse']['interval'])} | "
                       f"{fmt(x['grf_r']['point'], 4)} | {fmt(g['point'], 4) if g else ''} | "
                       f"{fmt_iv(g['interval'], 4) if g else ''} |")
        out.append("")
    out += ["## Premium tests (prediction.txt section 3)", "",
            "| set | line | comparator | body mean / bar / difference [95 %] | margin | premium needs body mean | "
            "reading |", "|---|---|---|---|---|---|---|"]
    for s in SETS:
        for ln in LINES:
            t = T[s][ln]
            nd = 2 if ln == "grf_rmse" else 4
            out.append(f"| {s} | {ln} | {t['comparator_bar']} | {fmt(t['body_mean'], nd)} / "
                       f"{fmt(t['comparator_mean'], nd)} / {fmt(t['difference'], nd)} {fmt_iv(t['interval'], nd)} | "
                       f"{fmt(t['margin'], nd)} | {fmt(t['needs_body_mean'], nd)} | {t['reading']} |")
    out += ["", "Set readings (PREMIUM when both lines read PREMIUM, LOSS when both read LOSS): "
            + ", ".join(f"{s} {r['readings'][s]}" for s in SETS) + ".", "",
            "## Against each group's best bar", "", "| set | line | group | comparator | difference [95 %] | reading |",
            "|---|---|---|---|---|---|"]
    for s in SETS:
        for ln in LINES:
            for g, t in r["group_tests"][s][ln].items():
                nd = 2 if ln == "grf_rmse" else 4
                out.append(f"| {s} | {ln} | {g} | {t['comparator_bar']} | {fmt(t['difference'], nd)} "
                           f"{fmt_iv(t['interval'], nd)} | {t['reading']} |")
    sa = r["s_arm"]
    out += ["", "## S arm (beside): bars refitted with the stage-2 stroke train fold, scored on ST7", "",
            "| row | grf_rmse (% BW) | 95 % | grf_r_gain | 95 % |", "|---|---|---|---|---|"]
    for n in ["b1_s", "b2_s", *[f"tok_{e.key}_s" for e in ENDS], *BODY]:
        x = sa["lines"][n]
        out.append(f"| {n} | {fmt(x['grf_rmse']['point'])} | {fmt_iv(x['grf_rmse']['interval'])} | "
                   f"{fmt(x['grf_r_gain']['point'], 4)} | {fmt_iv(x['grf_r_gain']['interval'], 4)} |")
    out += ["", "S-arm tests: " + "; ".join(f"{ln} against {sa['tests'][ln]['comparator_bar']}: "
                                            f"{sa['tests'][ln]['reading']}" for ln in LINES) + ".", "",
            "## Token probes: variant choice and cross-fitted lines on the fit set", "",
            "| checkpoint | chosen | cross-fitted grf_rmse latent / measurement / stem | alpha (chosen) |",
            "|---|---|---|---|"]
    for k, t in r["tokens"].items():
        cf = t["crossfit"]
        out.append(f"| {k} | {t['chosen']} | {fmt(cf['latent']['grf_rmse'])} / {fmt(cf['measurement']['grf_rmse'])} / "
                   f"{fmt(cf['stem']['grf_rmse'])} | {t['variants'][t['chosen']]['alpha']:g} |")
    sp = r["supervised"]
    out += ["", f"Supervised bars: fitted on {sp['fit']['rows']} scored samples of {sp['fit']['persons']} persons, "
            f"{sp['fit']['features']} features; cross-fitted on the fit set: b1 grf_rmse "
            f"{fmt(sp['crossfit']['b1']['grf_rmse'])}, b2 {fmt(sp['crossfit']['b2']['grf_rmse'])}. Fallback samples "
            f"(a scored sample without a finite prediction): {r['fallback_samples']}; tokens: "
            + ", ".join(f"{k} {t['fallback_samples'][t['chosen']]}" for k, t in r["tokens"].items()) + ".", "",
            "## Predictions", "", "| id | prediction | met | value |", "|---|---|---|---|"]
    for k, p in r["predictions"].items():
        val = ", ".join(str(x) if isinstance(x, str) else fmt(x, 4) for x in p["value"])
        out.append(f"| {k} | {p['what']} | {'met' if p['met'] else 'missed'} | {val} |")
    sf = r["strokefold_v1_B"]
    out += ["", "## Checks", "",
            "- Body rows: per person the same samples and the Newton floor's RMSE and r as the reports (within "
            f"{CHECK_TOL:g}): " + "; ".join(f"{s} " + ", ".join(f"{k} {'passed' if c['passed'] else 'FAILED'}"
                                                                for k, c in r["checks"][s].items()) for s in SETS) + ".",
            f"- ST7 from the strokefold run of v1_B ({STROKEFOLD_REPORT}): grf_rmse {fmt(sf['grf_rmse'], 3)}, "
            f"grf_r_gain {fmt(sf['grf_r_gain'], 4)} over {sf['persons']} persons, against "
            f"{fmt(L['ST7']['body_v1_B']['grf_rmse']['point'], 3)} and "
            f"{fmt(L['ST7']['body_v1_B']['grf_r_gain']['point'], 4)} from the g4h run used here.",
            f"- Time of this read: {r.get('timing_s')} s.", ""]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main
def job_commands():
    py = "<conda>/envs/mdfm/python.exe"
    return [f"{py} scripts/g4_grf_bars.py --tokens all", f"{py} scripts/g4_grf_bars.py"]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tokens", nargs="+", choices=(*END, "all"))
    g.add_argument("--print-jobs", action="store_true")
    p.add_argument("--limit-persons", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--cache", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.print_jobs:
        print("\n".join(job_commands()))
        return 0
    if a.limit_persons and not (a.out and a.cache):
        raise SystemExit("--limit-persons needs --out and --cache (a smoke run never takes the default folders)")
    print(f"process priority: {SB().below_normal()} (the GPU training's data loader shares the CPU)", flush=True)
    SB().wait_while_slow()
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if a.tokens:
        keys = [e.key for e in ENDS] if "all" in a.tokens else list(dict.fromkeys(a.tokens))
        run_tokens(keys, a.limit_persons, a.cache)
    else:
        run_read(a.limit_persons, a.out, a.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
