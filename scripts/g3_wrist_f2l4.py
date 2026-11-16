"""WRB (test B, owner-approved 2026-10-20): the wrist-only G3 bars of task WRS read on the walkway-anchored body of F2
lever 4 (f2l4_B, cache/runs/pilot_f2l4/ckpt_B_032044.pt). Does a body anchored to a real measurement in training keep
the stride scale when it sees only the wrists, and does it then show a premium over the wrist bars? Every output
carries LABEL: dev diagnostics, not a G3 read, not a G12 comparison; every f2l4 row is a 'walkway-anchored body' row
(reports/bench/f2_lever4/rule.txt section 8).

Prediction and method: reports/bench/g3_wrist_f2l4/prediction.txt, hashed before this script existed
(prediction.sha256 beside it and PREDICTION_SHA256 here); check_prediction asserts the three agree, and WRS's
prediction file (its premium rule) is asserted too, before any target, record, event or checkpoint is read, in every
mode.

Rows (prediction.txt section 2):
- d_raw (primary): the family_wrists rows of the f2l4 ablation events (GPU, bf16), as row frames in WRS's layout;
- d_gain (primary): d_raw divided by one training-fold gain per line, fitted (scripts/g3_gain_check.py fit_gain
  through scripts/g3_wrist_bars.py body_gain) on a CPU float32 wrist-only decode of the 54 with f2l4_B (--decode);
- c_raw, c_gain (check, one device): the same CPU decode run on the 14, raw and with the same gains;
- bars: WRS's rows read from reports/bench/g3_wrist/per_row.csv (no bar is refit).
Readings (section 3): WRS's premium rule (scripts/g3_wrist_bars.py comparisons, imported) on the primary arms, the
same on the check arms (a differing line is 'device-dependent'), and the scale reading S on the 54 (rise of the
wrist-only gain over v1_B's beyond twice the v1-recipe seed spread and at least half the all-site rise).

Modes (all CPU: CUDA_VISIBLE_DEVICES=-1 is set before torch loads):
  --decode --part I --parts N   the wrist-only f2l4 decode of the persons of part I (round robin over the 14 then the
                 54): per-trial predictions in cache/g3_wrist_f2l4/decode_<name>/trials/ (resumable: a trial file of
                 the same checkpoint and settings is kept), a part log beside them; no target value is written.
  --assemble     every trial file of the 14 and the 54 into reports/bench/g3_wrist_f2l4/decode_<name>/ (ics.csv,
                 strides.csv, speeds.csv, decode.json, _provenance.json) in WRS's layout.
  (default)      the read: rows, gains, readings, predictions; reports/bench/g3_wrist_f2l4/ results.json, summary.md,
                 per_row.csv, _provenance.json.
  --print-jobs   the commands of a full run.
--limit-persons N and --sets (smoke) need --out and --cache.

Run (mdfm environment; OMP_NUM_THREADS=2; four parts in parallel):
  python scripts/g3_wrist_f2l4.py --decode --part 0 --parts 4    (and parts 1 to 3)
  python scripts/g3_wrist_f2l4.py --assemble
  python scripts/g3_wrist_f2l4.py
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time

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

OUT = os.path.join(TREE, "reports", "bench", "g3_wrist_f2l4")
CACHE = os.path.join(TREE, "cache", "g3_wrist_f2l4")
PREDICTION_PATH = os.path.join(OUT, "prediction.txt")
PREDICTION_SHA_PATH = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "487bc91c0da06415136fbe23a54f54c73f9fda1cec979bf4e567006f6b9a1db3"
WRS_PREDICTION_SHA256 = "01fec4d7b2c3f8f252f02b7f5160864ed488fd3343b54563102c1f2bd19ab1a5"
ROW_LABEL = "walkway-anchored body"
LABEL = f"dev diagnostic, not a G3 read, not a G12 comparison; f2l4 rows: {ROW_LABEL}"
TASK = "WRB"
SETS_DECODED = ("dev14", "train54")

# inputs of the read, each pinned by its SHA-256 (prediction.txt sections 0 and 2)
F2L4_SHA256 = "734c9ae4a9cbddb544c3a96263ddfa6787c599a02b7b4ae1daa7c46f357fe9a0"
ABL_F2L4 = "reports/gates/g3/diag_pilot_f2l4_ckpt_B_032044_ablations_bw32"
ABL_V1 = "reports/gates/g3/diag_phaseA_root_full_ckpt_B_032044_ablations_bw32"
PINNED = {
    f"{ABL_F2L4}/events.csv": "35dc7a6a74d4574666ee96b66992655077a4b1d244dc4f278cb83028918687ee",
    f"{ABL_F2L4}/diag.json": "1ec91ed059d9e7970a5f0ccced5e50f6130d8f6fb0b829b9dc9d693d568955e7",
    f"{ABL_V1}/events.csv": "004b3eb34f3093244f2ebbd613c1b92935b9448c802e4a54d39bb60138f19362",
    "reports/bench/g3_wrist/results.json": "e137cf99ca7554501e97b2aaf8ca60b8becb46d726647fbae261bddd22fbcdd4",
    "reports/bench/g3_wrist/per_row.csv": "f6c28e992d6218fc19234fd91d44f3ee1e19febbe1f09ad0f46fa7b4c6006983",
}
ABL_ROW = "family_wrists"
ABL_SUBSETS = {"dev14": "all14", "dev12": "no_whc003_whc026"}       # diag.json subsets of the two sets
ALL_SITE_DECODES = {"v1_B": "phaseA_root_full_ckpt_B_032044", "s2_B": "pilot_v1_s2_ckpt_B_032044",
                    "f2l4_B": "pilot_f2l4_ckpt_B_032044"}            # reports/bench/g3_gain_check/decode_<name>/
SCALE_FRACTION = 0.5                 # rise_w at least this share of rise_a (F2 lever 4 T3's 'follows')
SCALE_SPREADS = 2.0                  # rise_w beyond this many v1-recipe seed spreads


# ---------------------------------------------------------------- siblings, prediction, the end
def _load(name, key):
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def WRS():
    """scripts/g3_wrist_bars.py (task WRS), imported unchanged."""
    return _load("g3_wrist_bars", "_wrb_g3_wrist_bars")


def END():
    """The f2l4 end in WRS's End form (seed: v1's, whose phase A end f2l4 continues)."""
    W = WRS()
    return W.End("f2l4_B", "pilot_f2l4_ckpt_B_032044", "cache/runs/pilot_f2l4/ckpt_B_032044.pt", "body", "body",
                 20260928, F2L4_SHA256)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(path=PREDICTION_PATH, sha_path=PREDICTION_SHA_PATH, expected=PREDICTION_SHA256):
    """The SHA-256 of this task's prediction file, asserted equal to its sidecar and to PREDICTION_SHA256; then WRS's
    prediction file (the premium rule) asserted by WRS's own check."""
    got = sha256_file(path)
    with open(sha_path, encoding="utf-8") as f:
        side = f.read().split()[0].strip()
    if not (got == side == expected):
        raise AssertionError(f"prediction hash mismatch: file {got}, sidecar {side}, script {expected}")
    W = WRS()
    if W.PREDICTION_SHA256 != WRS_PREDICTION_SHA256:
        raise AssertionError(f"WRS prediction constant {W.PREDICTION_SHA256}, expected {WRS_PREDICTION_SHA256}")
    W.check_prediction()
    return got


def check_pinned(tree=TREE, pinned=None):
    """{path: SHA-256} of the pinned inputs, each asserted equal to its pin."""
    out = {}
    for p, want in (PINNED if pinned is None else pinned).items():
        got = sha256_file(os.path.join(tree, p))
        if got != want:
            raise AssertionError(f"{p}: SHA-256 {got}, pinned {want}")
        out[p] = got
    return out


def decode_dirs(out=None, cache=None):
    """(report folder, per-trial cache folder, part-log folder) of the f2l4 wrist-only decode."""
    name = END().name
    base = os.path.join(cache or CACHE, f"decode_{name}")
    return os.path.join(out or OUT, f"decode_{name}"), os.path.join(base, "trials"), os.path.join(base, "parts")


def rel(path):
    from mdfm.gates import common
    return common.rel(path)


# ---------------------------------------------------------------- the decode (--decode, --assemble)
def decode_code_files():
    """The files of the decode path (WRS's code files and the model package)."""
    files = [os.path.join(HERE, "g3_wrist_bars.py"), os.path.join(HERE, "eval_g3_diag.py"),
             os.path.join(HERE, "g3_gain_check.py"), os.path.join(TREE, "mdfm", "gates", "g3.py"),
             os.path.join(TREE, "mdfm", "eval", "infer.py"), os.path.join(TREE, "mdfm", "baselines", "readouts.py"),
             os.path.join(TREE, "mdfm", "readouts", "gait.py"), os.path.join(TREE, "mdfm", "data", "canonical.py")]
    mdir = os.path.join(TREE, "mdfm", "model")
    files += [os.path.join(mdir, f) for f in sorted(os.listdir(mdir)) if f.endswith(".py") and not f.startswith("._")]
    return files


def decode_code_sha256():
    """SHA-256 of the decode path: the code hash of decode_code_files and the source of this script's decode
    functions (so that a later edit of the read part leaves it unchanged)."""
    from mdfm.gates import common
    src = "".join(inspect.getsource(f) for f in (part_persons, run_decode_part, decode_code_files))
    return hashlib.sha256((common.code_sha256(decode_code_files()) + "\n" + src).encode("utf-8")).hexdigest()


def part_persons(sets, part, parts):
    """[(set name, person)] of part `part` of `parts`: round robin over the persons of SETS_DECODED in order."""
    if not (0 <= int(part) < int(parts)):
        raise ValueError(f"part {part} of {parts}")
    every = [(s, p) for s in SETS_DECODED for p in sets[s]]
    return [x for k, x in enumerate(every) if k % int(parts) == int(part)]


def run_decode_part(part, parts, limit=0, only=None, out=None, cache=None, log=print):
    """--decode (module docstring): the wrist-only f2l4 decode of one part's persons, WRS's settings and code path."""
    from mdfm.eval import infer as I
    from mdfm.gates import g3
    from mdfm.gates import common
    from mdfm.train import provenance as PV
    check_prediction()
    W = WRS()
    end = END()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    code_at_start = decode_code_sha256()
    train, dev, _ = W.sets_of(0)
    sets = {"dev14": list(dev), "train54": list(train)}
    if only:
        sets = {s: (v if s in only else []) for s, v in sets.items()}
    if limit:
        sets = {s: v[:int(limit)] for s, v in sets.items()}
    mine = part_persons(sets, part, parts)
    _, tdir, pdir = decode_dirs(out, cache)
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    ld = I.load_checkpoint(os.path.join(TREE, end.path), weights=W.DECODE_SETTINGS["weights"], device="cpu")
    if ld.sha256 != end.sha256:
        raise AssertionError(f"{end.path}: SHA-256 {ld.sha256}, expected {end.sha256}")
    trained = ld.trained_persons()
    if set(dev) & trained:
        raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev) & trained)}")
    pred = I.G3Predictor(ld, refine_steps=W.DECODE_SETTINGS["refine_steps"], points=W.DECODE_SETTINGS["points"],
                         batch_windows=W.DECODE_SETTINGS["batch_windows"], seed=W.DECODE_SETTINGS["seed"],
                         pool=W.DECODE_SETTINGS["pool"])
    settings = dict(W.DECODE_SETTINGS, window_s=I.WIN_S, hop_s=I.HOP_S, sites=list(W.WRIST_SITES))
    counts = {}
    for set_name in SETS_DECODED:
        persons = [p for s, p in mine if s == set_name]
        if not persons:
            continue
        tg = g3.load_targets(persons)
        if tg.counts["lockbox_rows"]:
            raise Refused("a lockbox target row was kept")
        info = {"run": 0, "kept": 0}
        n = len(tg.trials)
        for k, (tid, rec) in enumerate(W.wrist_records(persons, tg.trials, info)):
            path = os.path.join(tdir, f"{tid}.json")
            prev = W.trial_record(path)
            if prev and prev.get("checkpoint_sha256") == end.sha256 and prev.get("settings") == settings \
                    and prev.get("set") == set_name:
                info["kept"] += 1
                continue
            t0 = time.time()
            d = {"trial_id": tid, "person_id": str(rec.get("person_id")), "set": set_name,
                 "checkpoint_sha256": end.sha256, "settings": settings, **W.decode_trial(pred, tid, rec, tg)}
            d = W.clean(d)
            common.write_text(path, json.dumps(d) + "\n")
            info["run"] += 1
            log(f"  part {part}/{parts} {set_name} {k + 1}/{n} {tid}: {len(d['strides'])} strides "
                f"({time.time() - t0:.0f} s)", flush=True)
        counts[set_name] = {"persons": len(persons), "trials": n, **info}
    doc = {"what": LABEL, "task": TASK, "mode": "decode", "part": int(part), "parts": int(parts), "run_started": started,
           "run_finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "partial": bool(limit or only),
           "persons": [[s, p] for s, p in mine], "counts": counts, "settings": settings,
           "checkpoint": {**ld.describe(), "key": end.key},
           "trained_persons_among_train54": len(set(train) & trained),
           "decode_code_sha256_at_start": code_at_start, "decode_code_sha256_at_end": decode_code_sha256(),
           "prediction_sha256": PREDICTION_SHA256, "moveatlas_head": PV.moveatlas_head(),
           "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
           "timing_s": round(time.time() - t_start, 1)}
    common.write_text(os.path.join(pdir, f"part_{int(part)}_of_{int(parts)}.json"), json.dumps(W.clean(doc), indent=1) + "\n")
    log(f"decode part {part}/{parts}: {counts} ({doc['timing_s']} s)", flush=True)
    return doc


def head_decode_code_sha256(tree=TREE):
    """decode_code_sha256 computed on the committed (git HEAD) versions of decode_code_files instead of the files on
    disk: equal to a part's start hash when the part ran the committed decode path (another task may edit those
    files in the working tree while the parts run)."""
    import subprocess
    rows = []
    for p in decode_code_files():
        r = rel(p)
        safe = os.path.abspath(tree).replace("\\", "/")
        got = subprocess.run(["git", "-c", f"safe.directory={safe}", "-C", tree, "show", f"HEAD:{r}"], capture_output=True)
        if got.returncode != 0:
            raise RuntimeError(f"git show HEAD:{r}: {got.stderr.decode('utf-8', 'replace').strip()}")
        rows.append((r, hashlib.sha256(got.stdout).hexdigest()))
    code = hashlib.sha256("".join(f"{r}\t{h}\n" for r, h in sorted(rows)).encode("utf-8")).hexdigest()
    src = "".join(inspect.getsource(f) for f in (part_persons, run_decode_part, decode_code_files))
    return hashlib.sha256((code + "\n" + src).encode("utf-8")).hexdigest()


def part_logs(pdir):
    """The part logs of a decode folder (sorted by name)."""
    if not os.path.isdir(pdir):
        return []
    out = []
    for f in sorted(os.listdir(pdir)):
        if f.startswith("part_") and f.endswith(".json"):
            with open(os.path.join(pdir, f), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def run_assemble(limit=0, only=None, out=None, cache=None, log=print):
    """--assemble (module docstring): the trial files of every trial of the decoded sets into WRS's decode layout.
    Every trial of the sets' targets must hold a trial file of this checkpoint and these settings."""
    import pandas as pd

    from mdfm.eval import infer as I
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    check_prediction()
    W = WRS()
    end = END()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    train, dev, _ = W.sets_of(0)
    sets = {"dev14": list(dev), "train54": list(train)}
    if only:
        sets = {s: (v if s in only else []) for s, v in sets.items()}
    if limit:
        sets = {s: v[:int(limit)] for s, v in sets.items()}
    rep, tdir, pdir = decode_dirs(out, cache)
    logs = part_logs(pdir)
    if not logs:
        raise SystemExit(f"{rel(pdir)}: no part log")
    settings = dict(W.DECODE_SETTINGS, window_s=I.WIN_S, hop_s=I.HOP_S, sites=list(W.WRIST_SITES))
    rows = {"ics": [], "strides": [], "speeds": []}
    counts, missing = {}, []
    for set_name in SETS_DECODED:
        persons = sets[set_name]
        if not persons:
            continue
        tg = g3.load_targets(persons)
        if tg.counts["lockbox_rows"]:
            raise Refused("a lockbox target row was kept")
        n_found = 0
        for tid in tg.trials["trial_id"].astype(str):
            d = W.trial_record(os.path.join(tdir, f"{tid}.json"))
            if not d or d.get("checkpoint_sha256") != end.sha256 or d.get("settings") != settings \
                    or d.get("set") != set_name:
                missing.append(tid)
                continue
            if d["person_id"] not in set(persons):
                raise Refused(f"{tid}: person outside the {set_name} set")
            n_found += 1
            for kind in rows:
                rows[kind] += [[set_name, tid, d["person_id"]] + list(r) for r in d[kind]]
        counts[set_name] = {"persons": len(persons), "trials": int(len(tg.trials)), "trial_files": n_found,
                            "target_counts": tg.counts}
    if missing:
        raise SystemExit(f"{len(missing)} trials without a trial file of this decode: {missing[:5]}")
    cols = {"ics": ["foot", "t_s"], "strides": ["foot", "t0", "t1", "length_m"], "speeds": ["pass_id", "speed_m_s"]}
    files = {}
    for kind, c in cols.items():
        df = pd.DataFrame(rows[kind], columns=["set", "trial_id", "person_id"] + c)
        files[kind] = common.write_csv(df, os.path.join(rep, f"{kind}.csv"))
    codes = sorted({lg["decode_code_sha256_at_start"] for lg in logs} | {lg["decode_code_sha256_at_end"] for lg in logs})
    res = {"what": LABEL, "task": TASK, "mode": "assemble", "run_started": started, "partial": bool(limit or only),
           "checkpoint": logs[0]["checkpoint"], "settings": settings, "counts": counts,
           "parts": [{k: lg[k] for k in ("part", "parts", "run_started", "run_finished", "counts", "timing_s",
                                         "decode_code_sha256_at_start", "decode_code_sha256_at_end", "moveatlas_head",
                                         "environment")} for lg in logs],
           "decode_code_sha256": codes, "decode_code_sha256_now": decode_code_sha256(),
           "trained_persons_among_train54": logs[0]["trained_persons_among_train54"],
           "files": {k: {"path": rel(v), "sha256": common.sha256_file(v)} for k, v in files.items()},
           "prediction_sha256": PREDICTION_SHA256, "moveatlas_head": PV.moveatlas_head(),
           "timing_s": round(sum(float(lg["timing_s"]) for lg in logs), 1),
           "timing_note": "sum of the part processes' times (parallel parts)"}
    if res["checkpoint"]["sha256"] != end.sha256:
        raise AssertionError("part log of another checkpoint")
    common.write_text(os.path.join(rep, "decode.json"), json.dumps(W.clean(res), indent=1) + "\n")
    common.write_text(os.path.join(rep, "_provenance.json"), json.dumps(W.clean(provenance(started)), indent=1) + "\n")
    log(f"assembled {rel(rep)}: {counts and {k: v['trial_files'] for k, v in counts.items()}}", flush=True)
    return res


# ---------------------------------------------------------------- frames of the ablation events
TEXT_COLUMNS = ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")


def read_events(path, allowed, row=ABL_ROW):
    """(stride table, speed table) of one row of an events.csv of scripts/eval_g3_diag.py, in the columns of
    g3.match_strides and g3.match_speeds (length_m, pred_m, err_cm; speed_m_s, pred_m_s, err_m_s; matched). The
    persons of the file are read first and must lie in `allowed` (the 14)."""
    import pandas as pd
    who = pd.read_csv(path, encoding="utf-8", usecols=["person_id"], dtype=str)
    bad = set(who["person_id"].dropna()) - {str(p) for p in allowed}
    if bad:
        raise Refused(f"{os.path.basename(path)}: {len(bad)} persons outside the dev controls")
    ev = pd.read_csv(path, encoding="utf-8", dtype={c: str for c in TEXT_COLUMNS})
    ev = ev[ev["row"] == row]
    if not len(ev):
        raise ValueError(f"{os.path.basename(path)}: no row {row}")
    m = ev["matched"].astype(str).str.strip().str.lower() == "true"
    num = {c: pd.to_numeric(ev[c], errors="coerce").to_numpy(np.float64)
           for c in ("t0_s", "t1_s", "reference", "prediction", "signed_error")}
    tables = {}
    for kind, unit in (("stride", "cm"), ("speed", "m/s")):
        k = (ev["kind"] == kind).to_numpy()
        if (ev.loc[k, "unit"] != unit).any():
            raise ValueError(f"{os.path.basename(path)}: {kind} rows in another unit than {unit}")
        base = pd.DataFrame({"trial_id": ev.loc[k, "trial_id"].to_numpy(), "person_id": ev.loc[k, "person_id"].to_numpy(),
                             "task": ev.loc[k, "task"].to_numpy(), "pass_id": ev.loc[k, "pass_id"].fillna("").to_numpy(),
                             "foot": ev.loc[k, "foot"].fillna("").to_numpy(), "t0": num["t0_s"][k], "t1": num["t1_s"][k],
                             "matched": m.to_numpy()[k]})
        if kind == "stride":
            base["length_m"] = num["reference"][k] / 100.0
            base["pred_m"] = num["prediction"][k] / 100.0
            base["err_cm"] = np.where(base["matched"], num["signed_error"][k], np.nan)
        else:
            base["speed_m_s"] = num["reference"][k]
            base["pred_m_s"] = num["prediction"][k]
            base["err_m_s"] = np.where(base["matched"], num["signed_error"][k], np.nan)
        if base["matched"].any():
            v = base.loc[base["matched"], ["err_cm" if kind == "stride" else "err_m_s"]].to_numpy()
            if not np.isfinite(v).all():
                raise ValueError(f"{os.path.basename(path)}: a matched {kind} row without a finite error")
        tables[kind] = base.reset_index(drop=True)
    return tables["stride"], tables["speed"]


def scale_tables(st, sp, g_s, g_v):
    """The stride and speed tables with every prediction divided by its gain and the errors recomputed (matching
    unchanged); g_s = g_v = 1 returns copies with the recorded errors."""
    st, sp = st.copy(), sp.copy()
    if g_s != 1.0:
        st["pred_m"] = st["pred_m"] / g_s
        st["err_cm"] = np.where(st["matched"], 100.0 * (st["pred_m"] - st["length_m"]), np.nan)
    if g_v != 1.0:
        sp["pred_m_s"] = sp["pred_m_s"] / g_v
        sp["err_m_s"] = np.where(sp["matched"], sp["pred_m_s"] - sp["speed_m_s"], np.nan)
    return st, sp


def table_frame(st, sp, name):
    """Row frame (comparison.ROW_COLUMNS) of a stride and a speed table, as WRS's row_frames builds it."""
    import pandas as pd

    from mdfm.baselines import readouts as RO
    return pd.concat([RO._pass_rows(st, "stride_mae_cm", name, "err_cm", "pred_m"),
                      RO._pass_rows(sp, "speed_mae_m_s", name, "err_m_s", "pred_m_s")], ignore_index=True)


def stride_gain(frame, persons):
    """F1's gain on a row frame: median of decoded over walkway over the matched strides of the persons."""
    from mdfm.baselines import comparison as CMP
    f = CMP.of_line(frame, "stride_mae_cm")
    f = f[f["matched"].astype(bool) & f["person_id"].astype(str).isin({str(p) for p in persons})]
    x, y = f["target"].to_numpy(np.float64), f["prediction"].to_numpy(np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return {"gain": float(np.median(y[ok] / x[ok])) if ok.any() else float("nan"), "n_strides": int(ok.sum())}


# ---------------------------------------------------------------- WRS's bars
def bool_col(s):
    return s.astype(str).str.strip().str.lower() == "true"


def wrs_frames(tree=TREE):
    """{row: frame} of WRS's per_row.csv (SHA-256 pinned), matched and enters as booleans."""
    import pandas as pd
    p = os.path.join(tree, "reports", "bench", "g3_wrist", "per_row.csv")
    rows = pd.read_csv(p, encoding="utf-8", dtype={"pass_id": str, "trial_id": str, "person_id": str, "foot": str,
                                                    "row": str, "line": str, "variant": str, "task": str})
    rows["matched"] = bool_col(rows["matched"])
    rows["enters"] = bool_col(rows["enters"])
    return {str(k): g.drop(columns=[c for c in ("row", "in_dev12") if c in g.columns]).reset_index(drop=True)
            for k, g in rows.groupby("row", sort=False)}


def wrs_bars(frames):
    """WRS's bars in the form of its comparisons: {a1, a2, b1, b2: [frame], tok_<arm>: [seed frames]}."""
    W = WRS()
    bars = {n: [frames[n]] for n in ("a1", "a2", "b1", "b2")}
    for arm, keys in W.TOKEN_ARMS.items():
        bars[arm] = [frames[f"{arm}_{k}"] for k in keys]
    return bars


def check_wrs_points(frames, wrs_res, sets, names=None, tol=1e-9):
    """Every recomputed point of WRS's rows (W.set_stats on per_row.csv) against WRS results.json: the largest
    relative difference; AssertionError beyond tol."""
    W = WRS()
    worst = 0.0
    for name in (names or sorted(wrs_res["rows"])):
        for s, persons in sets.items():
            for ln in W.LINES:
                a = float(W.set_stats(frames[name], persons, ln)["point"])
                b = float(wrs_res["rows"][name][s][ln]["point"])
                d = abs(a - b) / max(1.0, abs(b))
                if not (d <= tol):
                    raise AssertionError(f"WRS row {name} {s} {ln}: per_row.csv {a}, results.json {b}")
                worst = max(worst, d)
    return worst


# ---------------------------------------------------------------- the scale reading
def scale_reading(g_wrist, g_all, line):
    """S of prediction.txt section 3 (b) on one line: g_wrist, g_all {end: gain} with ends f2l4_B, v1_B (and s2_B
    for the wrist spread)."""
    rise_w = g_wrist["f2l4_B"] - g_wrist["v1_B"]
    d_w = abs(g_wrist["v1_B"] - g_wrist["s2_B"])
    rise_a = g_all["f2l4_B"] - g_all["v1_B"]
    beyond = bool(rise_w > SCALE_SPREADS * d_w)
    half = bool(rise_w >= SCALE_FRACTION * rise_a)
    transfers = beyond and half
    return {"line": line, "g_wrist": dict(g_wrist), "g_all": dict(g_all), "rise_w": rise_w, "d_w": d_w,
            "rise_a": rise_a, "beyond_2d": beyond, "at_least_half": half,
            "needs_g_wrist_above": g_wrist["v1_B"] + max(SCALE_SPREADS * d_w, SCALE_FRACTION * rise_a),
            "tau": rise_w / rise_a if rise_a != 0 else float("nan"),
            "drop_all_to_wrist": g_all["f2l4_B"] - g_wrist["f2l4_B"],
            "reading": "TRANSFERS" if transfers else "NOT SHOWN"}


def all_site_gains(train, dev, tree=TREE):
    """{end: {'stride', 'speed'}} of the all-site train54 decodes (reports/bench/g3_gain_check/decode_<name>/
    train_events.csv, row base) by fit_gain, with each file's SHA-256."""
    from mdfm.gates import g3
    G = WRS().GAIN()
    locked = set(g3.lockbox_persons())
    out = {}
    for end, name in ALL_SITE_DECODES.items():
        p = os.path.join(tree, "reports", "bench", "g3_gain_check", f"decode_{name}", "train_events.csv")
        st, sp, counts = G.read_events(p, train, locked, row="base")
        out[end] = {"stride": G.fit_gain(st, train, dev, locked)["gain"], "speed": G.fit_gain(sp, train, dev, locked)["gain"],
                    "counts": counts, "path": rel(p), "sha256": sha256_file(p)}
    return out


def check_all_site(gains, tree=TREE, tol=1e-9):
    """The all-site gains against the reports that printed them: f2_lever4 T1 (f2l4_B, v1_B stride),
    g3_gain_check results (v1_B, s2_B both lines), the SCALE chain (f2l4_B speed, printed at 6 digits)."""
    with open(os.path.join(tree, "reports", "bench", "f2_lever4", "results.json"), encoding="utf-8") as f:
        t1 = json.load(f)["T"]["T1"]["ends"]
    with open(os.path.join(tree, "reports", "bench", "g3_gain_check", "results.json"), encoding="utf-8") as f:
        gc = json.load(f)["ends"]
    with open(os.path.join(tree, "reports", "bench", "scale", "chain", "results.json"), encoding="utf-8") as f:
        ch = json.load(f)["train54"]
    pairs = [("f2l4_B", "stride", t1["f2l4_B"]["train54_stride_gain"], tol),
             ("v1_B", "stride", t1["v1_B"]["train54_stride_gain"], tol),
             ("v1_B", "stride", gc["v1_B"]["gain"]["stride"]["gain"], tol),
             ("v1_B", "speed", gc["v1_B"]["gain"]["speed"]["gain"], tol),
             ("s2_B", "stride", gc["s2_B"]["gain"]["stride"]["gain"], tol),
             ("s2_B", "speed", gc["s2_B"]["gain"]["speed"]["gain"], tol),
             ("f2l4_B", "speed", ch["f2l4_B"]["speed"]["gain"], 5.1e-7)]
    out = []
    for end, line, want, t in pairs:
        got = gains[end][line]
        if not abs(got - float(want)) <= t:
            raise AssertionError(f"all-site gain {end} {line}: {got}, printed {want}")
        out.append({"end": end, "line": line, "computed": got, "printed": float(want)})
    return out


# ---------------------------------------------------------------- paired comparisons (beside)
def paired(body_frames, comp_frames, persons, line):
    """Body minus comparator seed means on a line and set with the paired person-clustered interval
    (comparison.bootstrap_line, 2,000 resamples, seed 20260929)."""
    from mdfm.baselines import comparison as CMP
    W = WRS()
    bf = [CMP.of_line(f, line) for f in body_frames]
    cf = [CMP.of_line(f, line) for f in comp_frames]
    bp = [CMP.line_point(f, "mean", persons) for f in bf]
    cp = [CMP.line_point(f, "mean", persons) for f in cf]
    boot = CMP.bootstrap_line(bf, cf, persons, "mean", W.BOOT_N, W.SEED)
    return {"body_points": bp, "comparator_points": cp, "difference": CMP.seed_mean(bp) - CMP.seed_mean(cp),
            "interval": boot["interval"], "undefined_resamples": boot["undefined"]}


def short_id(p):
    return str(p).split(":", 1)[-1].split(" ")[0]


# ---------------------------------------------------------------- predictions
def within(lo, hi, x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return bool(lo <= v <= hi) if math.isfinite(v) else None


def predictions_check(res):
    """P1 to P9 of prediction.txt section 4 from the results: {id: {'what', 'values', 'met'}}."""
    R = res["rows"]

    def pt(row, s, line):
        return ((R.get(row) or {}).get(s) or {}).get(line, {}).get("point")

    out = {}

    def add(pid, what, checks):
        out[pid] = {"what": what, "values": {k: v for k, (v, _) in checks.items()},
                    "met": all(ok is True for _, ok in checks.values()) if checks else None}

    p1 = res["checks"]["p1_reconstruction"]
    add("P1", "d_raw rows from events.csv equal diag.json family_wrists (relative 1e-9)",
        {f"{k}": (v["relative_difference"], bool(v["relative_difference"] <= 1e-9)) for k, v in p1.items()})
    add("P2", "CPU check against the GPU row", {
        "c_raw dev12 stride in [18.9, 22.9]": (pt("c_raw", "dev12", "stride_mae_cm"), within(18.9, 22.9, pt("c_raw", "dev12", "stride_mae_cm"))),
        "c_raw dev14 stride in [20.8, 24.9]": (pt("c_raw", "dev14", "stride_mae_cm"), within(20.8, 24.9, pt("c_raw", "dev14", "stride_mae_cm"))),
        "c_raw dev12 speed in [0.19, 0.24]": (pt("c_raw", "dev12", "speed_mae_m_s"), within(0.19, 0.24, pt("c_raw", "dev12", "speed_mae_m_s")))})
    g = res["gains"]
    gs, gv = g["stride"]["gain"], g["speed"]["gain"]
    add("P3", "wrist-only gains of f2l4_B on the 54", {
        "g_S in [0.86, 0.93]": (gs, within(0.86, 0.93, gs)), "g_V in [0.82, 0.91]": (gv, within(0.82, 0.91, gv)),
        "|g_S - 0.8898| at most 0.03": (abs(gs - 0.8898), bool(abs(gs - 0.8898) <= 0.03))})
    add("P4", "d_gain", {
        "dev12 stride in [12.0, 18.0]": (pt("d_gain", "dev12", "stride_mae_cm"), within(12.0, 18.0, pt("d_gain", "dev12", "stride_mae_cm"))),
        "dev12 speed in [0.10, 0.17]": (pt("d_gain", "dev12", "speed_mae_m_s"), within(0.10, 0.17, pt("d_gain", "dev12", "speed_mae_m_s"))),
        "dev14 stride in [12.5, 19.5]": (pt("d_gain", "dev14", "stride_mae_cm"), within(12.5, 19.5, pt("d_gain", "dev14", "stride_mae_cm")))})
    rd = res["readings"]
    add("P5", "readings (primary)", {
        "d_raw stride LOSS": (rd["d_raw"]["stride_mae_cm"], rd["d_raw"]["stride_mae_cm"] == "LOSS"),
        "d_raw speed LOSS": (rd["d_raw"]["speed_mae_m_s"], rd["d_raw"]["speed_mae_m_s"] == "LOSS"),
        "d_gain stride not PREMIUM": (rd["d_gain"]["stride_mae_cm"], rd["d_gain"]["stride_mae_cm"] != "PREMIUM"),
        "d_gain speed not PREMIUM": (rd["d_gain"]["speed_mae_m_s"], rd["d_gain"]["speed_mae_m_s"] != "PREMIUM")})
    sc = res["scale"]
    add("P6", "scale reading S", {f"{ln} NOT SHOWN": (sc[ln]["reading"], sc[ln]["reading"] == "NOT SHOWN")
                                  for ln in ("stride", "speed")})
    sl = (((res["describe"].get("d_gain") or {}).get("dev12") or {}).get("stride_mae_cm") or {}).get("within_slope", {})
    v = sl.get("point")
    add("P7", "within-person slope of d_gain, dev12, stride per pass", {
        "in [0.25, 0.80]": (v, within(0.25, 0.80, v)), "below 0.904": (v, None if v is None else bool(float(v) < 0.904))})
    pe = res["anchor_effect"]["c_gain_vs_v1_B"]["stride_mae_cm"]["dev12"]["difference"]
    add("P8", "paired anchor effect on the gain arm, CPU, dev12 stride", {"in [-4.0, 3.0]": (pe, within(-4.0, 3.0, pe))})
    rc = res["readings_check"]
    add("P9", "check readings equal the primary readings", {
        f"{arm} {ln}": (rc[arm][ln], rc[arm][ln] == rd[arm][ln]) for arm in ("d_raw", "d_gain")
        for ln in ("stride_mae_cm", "speed_mae_m_s")})
    return out


# ---------------------------------------------------------------- the read
def run_read(out=None, log=print):
    """The default mode (module docstring)."""
    from mdfm.gates import common, g3
    from mdfm.train import provenance as PV
    check_prediction()
    W = WRS()
    G = W.GAIN()
    end = END()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    rep = os.path.abspath(out or OUT)
    pinned = check_pinned()
    train, dev, dev12 = W.sets_of(0)
    sets = {"dev12": dev12, "dev14": dev}
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise Refused("a lockbox target row was kept")
    with open(os.path.join(TREE, "reports", "bench", "g3_wrist", "results.json"), encoding="utf-8") as f:
        wrs = json.load(f)
    if wrs["sets"]["dev14_sha256"] != W._person_hash(dev) or wrs["sets"]["dev12_sha256"] != W._person_hash(dev12) \
            or wrs["sets"]["train_sha256"] != W._person_hash(train):
        raise AssertionError("WRS's sets differ from the sets read now")
    # WRS's bars (and its v1 rows), recomputed from per_row.csv and checked against its results.json
    wf = wrs_frames()
    wrs_check = check_wrs_points(wf, wrs, sets)
    bars = wrs_bars(wf)
    # primary rows: the f2l4 ablation events (GPU)
    with open(os.path.join(TREE, ABL_F2L4, "diag.json"), encoding="utf-8") as f:
        diag = json.load(f)
    if diag["checkpoint"]["sha256"] != F2L4_SHA256:
        raise AssertionError("diag.json of another checkpoint")
    for s, sub in ABL_SUBSETS.items():
        if sorted(diag["sets"]["subsets"][sub]) != sorted(short_id(p) for p in sets[s]):
            raise AssertionError(f"diag subset {sub} is not the set {s}")
    st, sp = read_events(os.path.join(TREE, ABL_F2L4, "events.csv"), dev)
    frames = {"d_raw": table_frame(st, sp, "d_raw_f2l4_B")}
    p1 = {}
    for s, sub in ABL_SUBSETS.items():
        for ln, key in (("stride_mae_cm", "stride_mae_cm"), ("speed_mae_m_s", "speed_mae_ms")):
            a = float(W.set_stats(frames["d_raw"], sets[s], ln)["point"])
            b = float(diag["rows"][ABL_ROW][sub][key])
            p1[f"{s} {ln}"] = {"events": a, "diag": b, "relative_difference": abs(a - b) / max(1.0, abs(b))}
    # the CPU decode: gains on the 54, the check rows on the 14
    P, meta = W.read_decode(end, train, dev, rep)
    if meta.get("task") != TASK:
        raise AssertionError("decode folder of another task")
    gains = W.body_gain(P["train54"], tg_train, train, dev)
    gs, gv = gains["stride"]["gain"], gains["speed"]["gain"]
    frames["d_gain"] = table_frame(*scale_tables(st, sp, gs, gv), "d_gain_f2l4_B")
    _, frames["c_raw"] = W.row_frames(P["dev14"], tg_dev, "c_raw_f2l4_B")
    _, frames["c_gain"] = W.row_frames(W.scaled(P["dev14"], gs, gv, "c_gain_f2l4_B"), tg_dev, "c_gain_f2l4_B")
    # v1_B's GPU family_wrists row and the same mixed-device gain arm (beside)
    st1, sp1 = read_events(os.path.join(TREE, ABL_V1, "events.csv"), dev)
    g1s, g1v = wrs["gains"]["v1_B"]["stride"]["gain"], wrs["gains"]["v1_B"]["speed"]["gain"]
    frames["v1_B_gpu_raw"] = table_frame(st1, sp1, "v1_B_gpu_raw")
    frames["v1_B_gpu_gain"] = table_frame(*scale_tables(st1, sp1, g1s, g1v), "v1_B_gpu_gain")
    v1_gpu_check = {s: {ln: float(W.set_stats(frames["v1_B_gpu_raw"], sets[s], ln)["point"]) for ln in W.LINES}
                    for s in W.SETS}
    for s in W.SETS:
        for ln in W.LINES:
            b = wrs["context"]["rows"]["gpu_family_wrists_v1_B"][s][ln]
            if abs(v1_gpu_check[s][ln] - b) > 1e-9 * max(1.0, abs(b)):
                raise AssertionError(f"v1_B GPU family_wrists {s} {ln}: {v1_gpu_check[s][ln]}, WRS context {b}")
    # lines of every row
    rows = {name: {s: {ln: W.set_stats(fr, sets[s], ln) for ln in W.LINES} for s in W.SETS} for name, fr in frames.items()}
    # readings
    comp, readings = W.comparisons({"d_raw": [frames["d_raw"]], "d_gain": [frames["d_gain"]]}, bars, sets)
    comp_c, readings_c = W.comparisons({"d_raw": [frames["c_raw"]], "d_gain": [frames["c_gain"]]}, bars, sets)
    device_dependent = {arm: {ln: readings[arm][ln] != readings_c[arm][ln] for ln in W.LINES} for arm in readings}
    # the scale reading on the 54
    ag = all_site_gains(train, dev)
    ag_check = check_all_site(ag)
    g_wrist = {"stride": {"f2l4_B": gs, "v1_B": wrs["gains"]["v1_B"]["stride"]["gain"],
                          "s2_B": wrs["gains"]["s2_B"]["stride"]["gain"]},
               "speed": {"f2l4_B": gv, "v1_B": wrs["gains"]["v1_B"]["speed"]["gain"],
                         "s2_B": wrs["gains"]["s2_B"]["speed"]["gain"]}}
    scale = {ln: scale_reading(g_wrist[ln], {e: ag[e][ln] for e in ag}, ln) for ln in ("stride", "speed")}
    dev_gain = {s: stride_gain(frames["d_raw"], sets[s]) for s in W.SETS}
    with open(os.path.join(TREE, "reports", "bench", "f2_lever4", "results.json"), encoding="utf-8") as f:
        l4 = json.load(f)
    t3 = l4["T"]["T3"]["rows"][ABL_ROW]["f2l4_B"]["gain"]
    if abs(dev_gain["dev12"]["gain"] - t3) > 1e-9:
        raise AssertionError(f"dev12 wrist gain {dev_gain['dev12']['gain']}, F2 lever 4 T3 {t3}")
    # the paired anchor effect (beside)
    pairs = {"d_raw_vs_v1_B_gpu": ([frames["d_raw"]], [frames["v1_B_gpu_raw"]]),
             "d_gain_vs_v1_B_gpu_gain": ([frames["d_gain"]], [frames["v1_B_gpu_gain"]]),
             "c_raw_vs_v1_B": ([frames["c_raw"]], [wf["d_raw_v1_B"]]),
             "c_gain_vs_v1_B": ([frames["c_gain"]], [wf["d_gain_v1_B"]]),
             "c_raw_vs_v1_recipe": ([frames["c_raw"]], [wf["d_raw_v1_B"], wf["d_raw_s2_B"]]),
             "c_gain_vs_v1_recipe": ([frames["c_gain"]], [wf["d_gain_v1_B"], wf["d_gain_s2_B"]])}
    anchor = {k: {ln: {s: paired(b, c, sets[s], ln) for s in W.SETS} for ln in W.LINES} for k, (b, c) in pairs.items()}
    # within-person slope, r and gain per pass (beside)
    describe = {name: {s: {ln: G.describe(W.pass_units_of(fr, ln, sets[s]), n_boot=W.BOOT_N, seed=W.SEED)
                           for ln in W.LINES} for s in W.SETS} for name, fr in frames.items()}
    import pandas as pd
    per_row = pd.concat([fr.assign(row=name) for name, fr in frames.items()], ignore_index=True)
    per_row["in_dev12"] = per_row["person_id"].isin(set(dev12))
    rows_path = common.write_csv(per_row, os.path.join(rep, "per_row.csv"))
    context = {"all_sites_f2l4_B": {"dev12": {"stride_mae_cm": l4["R1"]["S12"]["f2l4"], "speed_mae_m_s": l4["R1"]["V12"]["f2l4"]},
                                    "dev14": {"stride_mae_cm": l4["R2"]["S14"]["f2l4"], "speed_mae_m_s": l4["R2"]["V14"]["f2l4"]},
                                    "source": "reports/bench/f2_lever4/results.json R1, R2 (GPU)"},
               "wrs_rows": {k: {s: {ln: wrs["rows"][k][s][ln]["point"] for ln in W.LINES} for s in W.SETS}
                            for k in ("d_raw_v1_B", "d_raw_s2_B", "d_gain_v1_B", "d_gain_s2_B", "tok_body_v1_B",
                                      "tok_body_s2_B")},
               "wrs_readings": wrs["readings"]}
    res = {"what": LABEL, "row_label": ROW_LABEL, "task": TASK, "run_started": started,
           "prediction": {"path": rel(PREDICTION_PATH), "sha256": PREDICTION_SHA256},
           "wrs_prediction_sha256": WRS_PREDICTION_SHA256, "pinned_inputs": pinned,
           "sets": {"train": train, "train_sha256": W._person_hash(train), "dev14": dev, "dev14_sha256": W._person_hash(dev),
                    "dev12": dev12, "dev12_sha256": W._person_hash(dev12)},
           "rows": {k: {s: {ln: {kk: vv for kk, vv in v.items() if kk != "per_person"} for ln, v in x.items()}
                        for s, x in r.items()} for k, r in rows.items()},
           "per_person": {k: {s: {ln: v["per_person"] for ln, v in x.items()} for s, x in r.items()} for k, r in rows.items()},
           "gains": gains, "dev_wrist_stride_gain": dev_gain,
           "comparisons": comp, "readings": readings, "comparisons_check": comp_c, "readings_check": readings_c,
           "device_dependent": device_dependent, "scale": scale, "all_site_gains": ag,
           "anchor_effect": anchor, "describe": describe, "context": context,
           "decode": {"settings": meta["settings"], "counts": meta["counts"], "parts": meta["parts"],
                      "decode_code_sha256": meta["decode_code_sha256"], "decode_code_sha256_now": decode_code_sha256(),
                      "decode_code_sha256_of_head": head_decode_code_sha256(),
                      "parts_started_on_head": all(p["decode_code_sha256_at_start"] == head_decode_code_sha256()
                                                   for p in meta["parts"]),
                      "moveatlas_head": meta["moveatlas_head"], "files": meta["files"], "timing_s": meta["timing_s"]},
           "checks": {"p1_reconstruction": p1, "wrs_points_max_relative_difference": wrs_check,
                      "v1_B_gpu_family_wrists": v1_gpu_check, "all_site_gains_against_reports": ag_check,
                      "dev12_wrist_gain_against_T3": {"computed": dev_gain["dev12"]["gain"], "T3": t3}},
           "per_row": {"path": rel(rows_path), "sha256": common.sha256_file(rows_path), "n_rows": int(len(per_row))},
           "lockbox": {"persons_in_lockbox": len((set(train) | set(dev)) & set(g3.lockbox_persons())),
                       "target_lockbox_rows": int(tg_train.counts["lockbox_rows"]) + int(tg_dev.counts["lockbox_rows"])},
           "moveatlas_head": PV.moveatlas_head()}
    res["answer"] = answer(res)
    res["predictions"] = predictions_check(res)
    res["timing_s"] = round(time.time() - t_start, 1)
    res = W.clean(res)
    text = summary_md(res)
    common.write_text(os.path.join(rep, "results.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(rep, "summary.md"), text)
    common.write_text(os.path.join(rep, "_provenance.json"), json.dumps(W.clean(provenance(started)), indent=1) + "\n")
    log(text)
    return res


def answer(res):
    """prediction.txt section 3 (c): the two statements and what holds."""
    transfers = res["scale"]["stride"]["reading"] == "TRANSFERS"
    prem = [f"{arm} {ln}" for arm in ("d_raw", "d_gain") for ln in ("stride_mae_cm", "speed_mae_m_s")
            if res["readings"][arm][ln] == "PREMIUM" and not res["device_dependent"][arm][ln]]
    return {"scale_transfers_to_wrists": transfers, "wrist_premium": bool(prem), "premium_lines": prem,
            "device_dependent_lines": [f"{arm} {ln}" for arm, x in res["device_dependent"].items() for ln, v in x.items() if v]}


# ---------------------------------------------------------------- summary
def fmt(x, nd=2):
    return WRS().fmt(x, nd)


def fmt_iv(iv, nd=2):
    return WRS().fmt_iv(iv, nd)


ROW_TITLES = (("d_raw", "walkway-anchored body f2l4_B, raw (GPU events, primary)"),
              ("d_gain", "walkway-anchored body f2l4_B, one training-fold gain per line (primary)"),
              ("c_raw", "walkway-anchored body f2l4_B, raw (CPU decode, check)"),
              ("c_gain", "walkway-anchored body f2l4_B, one training-fold gain (CPU decode, check)"),
              ("v1_B_gpu_raw", "v1_B raw (GPU events, beside)"),
              ("v1_B_gpu_gain", "v1_B GPU events with its CPU wrist gain (beside)"))


def summary_md(r):
    """summary.md: every number read from the results dict as written to results.json."""
    W = WRS()
    L = [f"# Wrist-only G3 bars read on the walkway-anchored body f2l4_B ({r['what']})", "",
         "Written by scripts/g3_wrist_f2l4.py (task WRB, test B) under "
         f"{r['prediction']['path']} (SHA-256 {r['prediction']['sha256']}) and WRS's premium rule "
         f"(reports/bench/g3_wrist/prediction.txt, SHA-256 {r['wrs_prediction_sha256']}), both asserted before any "
         "target, record, event or checkpoint was read. Every number below is read from results.json of this folder. "
         "Input of every row: the G3 model input cut to wrist_L and wrist_R. Gains fitted on the 54 training-fold "
         "controls only; scored on dev12 and dev14 with the G3 harness's lines and person bootstrap (2,000 resamples, "
         f"seed 20260929). No lockbox, EVAL, CODY or care_pd record. MoveAtlas {r.get('moveatlas_head')}.", "",
         "## Rows", "",
         "| row | dev12 stride (cm) | 95 % | dev12 speed (m/s) | 95 % | dev14 stride (cm) | 95 % | dev14 speed (m/s) | 95 % | "
         "strides matched (dev14) | within-person slope, stride per pass (dev12) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, title in ROW_TITLES:
        x = r["rows"].get(key)
        if not x:
            continue
        a, b = x["dev12"], x["dev14"]
        ws = ((r["describe"].get(key) or {}).get("dev12") or {}).get("stride_mae_cm", {}).get("within_slope", {})
        L.append(f"| {title} | {fmt(a['stride_mae_cm']['point'])} | {fmt_iv(a['stride_mae_cm']['interval'])} | "
                 f"{fmt(a['speed_mae_m_s']['point'], 3)} | {fmt_iv(a['speed_mae_m_s']['interval'], 3)} | "
                 f"{fmt(b['stride_mae_cm']['point'])} | {fmt_iv(b['stride_mae_cm']['interval'])} | "
                 f"{fmt(b['speed_mae_m_s']['point'], 3)} | {fmt_iv(b['speed_mae_m_s']['interval'], 3)} | "
                 f"{fmt(b['stride_mae_cm']['matched_share'], 3)} | {fmt(ws.get('point'), 3)} {fmt_iv(ws.get('interval'), 3)} |")
    c = r["context"]["wrs_rows"]
    L += ["", "WRS rows on the same sets (CPU): " + "; ".join(
        f"{k} {fmt(v['dev12']['stride_mae_cm'])} / {fmt(v['dev14']['stride_mae_cm'])} cm, "
        f"{fmt(v['dev12']['speed_mae_m_s'], 3)} / {fmt(v['dev14']['speed_mae_m_s'], 3)} m/s" for k, v in c.items())
          + " (dev12 / dev14). All-site f2l4_B (GPU, F2 lever 4): " +
          f"{fmt(r['context']['all_sites_f2l4_B']['dev12']['stride_mae_cm'])} / "
          f"{fmt(r['context']['all_sites_f2l4_B']['dev14']['stride_mae_cm'])} cm, "
          f"{fmt(r['context']['all_sites_f2l4_B']['dev12']['speed_mae_m_s'], 3)} / "
          f"{fmt(r['context']['all_sites_f2l4_B']['dev14']['speed_mae_m_s'], 3)} m/s.", ""]
    g = r["gains"]
    dg = r["dev_wrist_stride_gain"]
    L += ["## Gains and the scale reading S (on the 54)", "",
          f"Wrist-only gains of f2l4_B on the 54 (CPU decode): g_S {fmt(g['stride']['gain'], 4)} over "
          f"{g['matched_train']['strides']} of {g['matched_train']['of_strides']} walkway strides, g_V "
          f"{fmt(g['speed']['gain'], 4)} over {g['matched_train']['passes']} of {g['matched_train']['of_passes']} passes. "
          f"Dev wrist stride gain of the GPU events (seen before the prediction): dev12 {fmt(dg['dev12']['gain'], 4)}, "
          f"dev14 {fmt(dg['dev14']['gain'], 4)}.", "",
          "| line | g_wrist f2l4_B | g_wrist v1_B | g_wrist s2_B | g_all f2l4_B | g_all v1_B | rise_w | 2 d_w | rise_a | "
          "share kept tau | drop all to wrist (f2l4_B) | TRANSFERS needs g_wrist above | reading |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ln in ("stride", "speed"):
        s = r["scale"][ln]
        L.append(f"| {ln} | {fmt(s['g_wrist']['f2l4_B'], 4)} | {fmt(s['g_wrist']['v1_B'], 4)} | {fmt(s['g_wrist']['s2_B'], 4)} | "
                 f"{fmt(s['g_all']['f2l4_B'], 4)} | {fmt(s['g_all']['v1_B'], 4)} | {fmt(s['rise_w'], 4)} | "
                 f"{fmt(2 * s['d_w'], 4)} | {fmt(s['rise_a'], 4)} | {fmt(s['tau'], 3)} | {fmt(s['drop_all_to_wrist'], 4)} | "
                 f"{fmt(s['needs_g_wrist_above'], 4)} | **{s['reading']}** |")
    L += ["", "## Premium rule (WRS's, one f2l4 seed)", "",
          "| arm | line | set | comparator | f2l4 point | comparator mean | difference | 95 % | margin m | premium needs (at "
          "most) | separation | premium | loss |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, cmp_key in (("primary", "comparisons"), ("check", "comparisons_check")):
        for arm in ("d_raw", "d_gain"):
            for line in W.LINES:
                nd = 2 if line == "stride_mae_cm" else 3
                for s in W.SETS:
                    t = r[cmp_key][arm][line][s]["best"]
                    L.append(f"| {arm} ({label}) | {line} | {s} | {t['bar']} | {fmt(t['body_mean'], nd)} | "
                             f"{fmt(t['comparator_mean'], nd)} | {fmt(t['difference'], nd)} | {fmt_iv(t['interval'], nd)} | "
                             f"{fmt(t['margin'], nd)} | {fmt(t['required_body_mean'], nd)} | {t['separation']} | "
                             f"{t['premium']} | {t['loss']} |")
    L += ["", "Readings (primary): " + "; ".join(f"{arm} {ln}: **{r['readings'][arm][ln]}**" for arm in ("d_raw", "d_gain")
                                                 for ln in W.LINES) + ".",
          "Readings (check, CPU): " + "; ".join(f"{arm} {ln}: {r['readings_check'][arm][ln]}" for arm in ("d_raw", "d_gain")
                                                for ln in W.LINES) + ".",
          "Device-dependent lines: " + (", ".join(r["answer"]["device_dependent_lines"]) or "none") + ".", "",
          "Against each group's best bar (primary; difference f2l4 minus bar, 95 % interval, premium / loss):", "",
          "| arm | line | set | classical | supervised | token |", "|---|---|---|---|---|---|"]
    for arm in ("d_raw", "d_gain"):
        for line in W.LINES:
            nd = 2 if line == "stride_mae_cm" else 3
            for s in W.SETS:
                b = r["comparisons"][arm][line][s]
                cells = [f"{b[gr]['bar']}: {fmt(b[gr]['difference'], nd)} {fmt_iv(b[gr]['interval'], nd)} "
                         f"({'premium' if b[gr]['premium'] else 'loss' if b[gr]['loss'] else 'neither'})" for gr in W.GROUPS]
                L.append(f"| {arm} | {line} | {s} | " + " | ".join(cells) + " |")
    L += ["", "## Beside: the paired anchor effect at the wrists (f2l4_B minus v1_B; negative is better)", "",
          "| pair | line | dev12 difference | 95 % | dev14 difference | 95 % |", "|---|---|---|---|---|---|"]
    for k, x in r["anchor_effect"].items():
        for line in W.LINES:
            nd = 2 if line == "stride_mae_cm" else 3
            a, b = x[line]["dev12"], x[line]["dev14"]
            L.append(f"| {k} | {line} | {fmt(a['difference'], nd)} | {fmt_iv(a['interval'], nd)} | "
                     f"{fmt(b['difference'], nd)} | {fmt_iv(b['interval'], nd)} |")
    ans = r["answer"]
    L += ["", "## Answer (prediction.txt section 3 (c))", "",
          f"- Scale transfers to the wrists (S stride TRANSFERS): {ans['scale_transfers_to_wrists']}.",
          f"- Wrist premium of the anchored body (a PREMIUM line, not device-dependent): {ans['wrist_premium']}"
          + (f" ({', '.join(ans['premium_lines'])})" if ans["premium_lines"] else "") + ".", "",
          "## Predictions (prediction.txt section 4)", "", "| id | what | met | values |", "|---|---|---|---|"]
    for pid, p in r["predictions"].items():
        vals = "; ".join(f"{k}: {fmt(v, 4) if WRS().fin(v) is not None else v}" for k, v in p["values"].items())
        L.append(f"| {pid} | {p['what']} | {p['met']} | {vals} |")
    ck = r["checks"]
    d = r["decode"]
    L += ["", "## Checks", "",
          f"- Lockbox: persons of the sets in the lockbox list {r['lockbox']['persons_in_lockbox']}; target lockbox rows "
          f"{r['lockbox']['target_lockbox_rows']} (asserted 0).",
          "- Pinned inputs (SHA-256 asserted): " + ", ".join(r["pinned_inputs"]) + ".",
          f"- WRS rows recomputed from its per_row.csv against its results.json: largest relative difference "
          f"{ck['wrs_points_max_relative_difference']}.",
          "- All-site train54 gains recomputed against the reports that printed them: " + "; ".join(
              f"{x['end']} {x['line']} {fmt(x['computed'], 6)}" for x in ck["all_site_gains_against_reports"]) + ".",
          f"- Dev12 wrist stride gain of the events against F2 lever 4 T3: {fmt(ck['dev12_wrist_gain_against_T3']['computed'], 6)} "
          f"and {fmt(ck['dev12_wrist_gain_against_T3']['T3'], 6)}.",
          f"- CPU decode: {d['settings']['device']}, {len(d['parts'])} parallel parts, {d['timing_s']} s of process time; "
          f"decode code SHA-256 at the parts' start and end {', '.join(d['decode_code_sha256'])}; of the committed "
          f"files (git HEAD) {d['decode_code_sha256_of_head']}; every part started on HEAD's decode path: "
          f"{d['parts_started_on_head']}; on disk at this read {d['decode_code_sha256_now']}; MoveAtlas "
          f"{d['moveatlas_head']}.",
          f"- Per-row file {r['per_row']['path']} (SHA-256 {r['per_row']['sha256']}, {r['per_row']['n_rows']} rows).",
          f"- Time of this read: {r.get('timing_s')} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- output helpers
def provenance(started):
    from mdfm.gates import common
    import torch
    W = WRS()
    files = [os.path.abspath(__file__)] + decode_code_files() + [
        os.path.join(HERE, "g3_supervised_baseline.py"), os.path.join(HERE, "g2_renderer_realism.py"),
        os.path.join(TREE, "mdfm", "eval", "probes.py"), os.path.join(TREE, "mdfm", "baselines", "comparison.py")]
    return {"script": "scripts/g3_wrist_f2l4.py", "label": LABEL, "task": TASK, "run_at": started,
            "checkpoint": {"key": "f2l4_B", "path": END().path, "sha256": F2L4_SHA256},
            "prediction_sha256": PREDICTION_SHA256, "wrs_prediction_sha256": WRS_PREDICTION_SHA256,
            "code_sha256": common.code_sha256(files), "code_files": [rel(f) for f in files],
            "decode_code_sha256": decode_code_sha256(), "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "moveatlas_stamp": W.DIAG().stamp_moveatlas(),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}


# ---------------------------------------------------------------- main
def job_commands():
    py = "<conda>/envs/mdfm/python.exe"
    cmds = [f"{py} scripts/g3_wrist_f2l4.py --decode --part {i} --parts 4" for i in range(4)]
    cmds += [f"{py} scripts/g3_wrist_f2l4.py --assemble", f"{py} scripts/g3_wrist_f2l4.py"]
    return cmds


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--decode", action="store_true")
    g.add_argument("--assemble", action="store_true")
    g.add_argument("--print-jobs", action="store_true")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--sets", default=None, help="smoke: comma-separated subset of dev14,train54")
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
    only = tuple(s.strip() for s in a.sets.split(",")) if a.sets else None
    if only and set(only) - set(SETS_DECODED):
        raise SystemExit(f"--sets: {sorted(set(only) - set(SETS_DECODED))} not in {SETS_DECODED}")
    if (a.limit_persons or only) and not (a.out and a.cache):
        raise SystemExit("--limit-persons and --sets need --out and --cache (a smoke run never takes the default folders)")
    print(f"process priority: {WRS().SB().below_normal()}", flush=True)
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if a.decode:
        run_decode_part(a.part, a.parts, a.limit_persons, only, a.out, a.cache)
    elif a.assemble:
        run_assemble(a.limit_persons, only, a.out, a.cache)
    else:
        run_read(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
