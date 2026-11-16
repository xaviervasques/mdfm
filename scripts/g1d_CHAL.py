"""G1D-CHAL: adversarial challenge of the G1 diagnosis (docs/notes/G1_DIAGNOSIS_2026-10-06.md) and of the lever 1
rule draft (docs/proposals/G1_LEVER1_RULE_DRAFT.md). Development diagnostic on the AMASS dev archives, not a gate
read. The prediction of the one new measurement (stage shape) was written and hashed before any error value was
computed by this script: reports/weekend_2026-10-01/tasks/G1D-CHAL.md, section Prediction (hash in
reports/bench/g1_diag/CHAL/prediction.sha256).

Stages (each writes under --out, default reports/bench/g1_diag/CHAL):
- hashes: no error value. Recomputes the SHA-256 of the Prediction section of each G1D report (G1D-Hstand,
  G1D-Hprior, G1D-Hlayout, G1D-SYN) under the extraction rule its prediction.sha256 line states, and sets the
  hash time beside the earliest time stamp of the outputs that hold error values. Writes hashes.json.
- attain: no error value. The window subset of the shape test (up to TAKES_PER_PERSON whole takes per dev person,
  seed SEED), its counts, the truth betas' spread within person (truth only), and the CPU time of the shape pass on
  a few windows (no error is computed). Writes attain.json.
- reproduce: recomputes from the source files the numbers the note and the rule draft cite (g1.json lines,
  decompose.json rows, the per-take pooling check, the archive split of the true-betas gain on the G1 statistic,
  the per-window difference between the decomposition pass and the harness pass). Writes reproduce.json.
- shape (CPU): the per-window shape posterior of the six-IMU input (probe_g1_decompose.posterior_pass, no pooling,
  20 refinement steps, the amortized posterior kept), on the attain subset, and each window's truth betas. Writes
  shape.npz and shape_run.json.
- read: the rest-pose skeleton error (22 joints, pelvis-relative, mm, the model's skeleton for both the estimate
  and the truth) of the window, take-pooled and person-pooled shape means (posterior.subject_pool, precision
  weighted), the bias share of the person's window shape error, and the readings of the prediction. Writes
  shape.json and shape.md.

Data: the dev G1Set (G1Set refuses the lockbox), the dev raw takes for the truth betas (mdfm.data.amass refuses
lockbox members), and existing report files. No EVAL, CODY or care_pd record, no GPU (CUDA_VISIBLE_DEVICES=-1).
Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
    <conda>/envs/mdfm/python.exe scripts/g1d_CHAL.py --stage hashes|attain|reproduce|shape|read
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (TREE, os.path.join(TREE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g1_diag", "CHAL")
TASKS = "reports/weekend_2026-10-01/tasks"
DIAG = "reports/bench/g1_diag"
DECOMPOSE = "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.json"
DECOMPOSE_WIN = "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/windows.csv"
G1_V1 = "reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/g1.json"
G1_V1_WIN = "reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/windows.csv"
G1_S2 = "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_032044/g1.json"
CHECKPOINT = os.path.join(TREE, "cache", "runs", "phaseA_root_full", "ckpt_B_032044.pt")
SEED = 20261006
TAKES_PER_PERSON = 3
STEPS = 20
POINTS = 256
BATCH = 16
B_BOOT = 2000
MM = 1000.0
DASHES = (chr(0x2013), chr(0x2014))

# criteria of the prediction (section Prediction of reports/weekend_2026-10-01/tasks/G1D-CHAL.md), never changed
BIAS_NOROOM = 0.71   # corrected person-bias share of the take-pooled squared skeleton offset at least this: NO_ROOM
BIAS_ROOM = 0.50     # at most this: ROOM; between: BETWEEN
NEED = 0.294         # share of the true-betas ceiling lever 1 must recover (margins.json, margin_over_ceiling)

REPORTS = {
    "Hstand": ("G1D-Hstand.md", "## Files"),
    "Hprior": ("G1D-Hprior.md", "## Results"),
    "Hlayout": ("G1D-Hlayout.md", "Hash:"),
    "SYN": ("G1D-SYN.md", "## Files"),
    "CHAL": ("G1D-CHAL.md", "## Files"),
}
# outputs whose time stamps are the first error reads of each task (stage run_at or file time)
ERROR_OUTPUTS = {
    "Hstand": ["Hstand/_provenance_analyze.json"],
    "Hprior": ["Hprior/run.json"],
    "Hlayout": ["Hlayout/arms_provenance.json", "Hlayout/results.json"],
    "SYN": ["SYN/pooling.json"],
}


# ---------------------------------------------------------------- io
def _p(rel):
    return os.path.join(TREE, rel)


def _load(rel):
    with open(_p(rel), encoding="utf-8") as f:
        return json.load(f)


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write(path, text):
    for d in DASHES:
        if d in text:
            raise ValueError(f"dash character in {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1) + "\n")


def _moveatlas_head():
    root = os.environ.get("MOVEATLAS_ROOT", "<moveatlas>")
    try:
        r = subprocess.run(["git", "-c", "safe.directory=<moveatlas>", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def _provenance(stage, inputs=()):
    return {"script": "scripts/g1d_CHAL.py", "stage": stage, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script_sha256": _sha_file(os.path.abspath(__file__)),
            "inputs": {p: _sha_file(_p(p)) for p in inputs}, "moveatlas_head": _moveatlas_head()}


# ---------------------------------------------------------------- pure functions (tested)
def prediction_section(text, stop):
    """The Prediction section of a report: from the line '## Prediction' to the line before the first later line
    that starts with `stop`, trailing blank lines removed, LF, one trailing newline. None when absent."""
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        a = lines.index("## Prediction")
    except ValueError:
        return None
    b = None
    for j in range(a + 1, len(lines)):
        if lines[j].startswith(stop):
            b = j
            break
    if b is None:
        return None
    sec = lines[a:b]
    while sec and not sec[-1].strip():
        sec.pop()
    return "\n".join(sec) + "\n"


def parse_hash_line(line):
    """(sha256, time stamp) of a prediction.sha256 line ('<hex>  <what> hashed <time>')."""
    parts = line.strip().split()
    t = parts[-1] if "hashed" in parts else None
    return parts[0], t


def pooled_means(mu, logvar, groups):
    """Precision-weighted mean per group (posterior.subject_pool in numpy): mu (W, k), logvar (W, k), groups (W,)
    keys. Returns (W, k), each row its group's pooled mean."""
    mu, lv = np.asarray(mu, float), np.asarray(logvar, float)
    keys = np.asarray([str(g) for g in groups])
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    prec = np.exp(-lv)
    G = int(inv.max()) + 1
    P = np.zeros((G, mu.shape[1]))
    M_ = np.zeros((G, mu.shape[1]))
    np.add.at(P, inv, prec)
    np.add.at(M_, inv, prec * mu)
    return (M_ / P)[inv]


def skeleton_error_mm(rest_hat, rest_true):
    """Mean over the joints of the distance of pelvis-relative rest joints (..., J, 3) m, in mm."""
    a = rest_hat - rest_hat[..., :1, :]
    b = rest_true - rest_true[..., :1, :]
    return np.linalg.norm(a - b, axis=-1).mean(-1) * MM


def bias_share(D):
    """D (W, J, 3): one person's window skeleton offsets (estimate minus truth, pelvis-relative). Share of the
    mean squared offset that is the squared offset of the person mean: |mean D|^2 / mean |D|^2 (summed over
    joints). 1: a pure person-level bias; near 1/W: pure noise around the truth."""
    D = np.asarray(D, float)
    m = D.mean(0)
    tot = float(np.mean(np.sum(D ** 2, axis=(1, 2))))
    return float(np.sum(m ** 2) / tot) if tot > 0 else float("nan")


def person_mean(values, persons):
    """Mean over persons of each person's mean of values."""
    values, persons = np.asarray(values, float), np.asarray([str(p) for p in persons])
    return float(np.mean([values[persons == p].mean() for p in np.unique(persons)]))


def person_boot(fn, persons, b=B_BOOT, seed=SEED):
    """Percentile 95 percent interval of fn(list of person keys) over b person resamples (duplicates kept)."""
    rng = np.random.default_rng(seed)
    ps = sorted(set(str(p) for p in persons))
    vals = []
    for _ in range(int(b)):
        pick = [ps[i] for i in rng.integers(0, len(ps), len(ps))]
        v = fn(pick)
        if math.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def take_bias_parts(d_take):
    """d_take (T, J, 3): one person's take-pooled skeleton offsets (estimate minus truth, pelvis-relative, m).
    Returns (bias2, total): bias2 = max(|mean_t d_t|^2 - s^2 / T, 0) with s^2 = sum_t |d_t - mean|^2 / (T - 1), the
    unbiased squared person-level bias that survives any amount of person pooling when the take errors are
    independent around it; total = mean_t |d_t|^2, the squared offset of the harness (take pooling). Sums over
    joints and axes. T < 2: (nan, total)."""
    d = np.asarray(d_take, float)
    T = d.shape[0]
    total = float(np.mean(np.sum(d ** 2, axis=(1, 2))))
    if T < 2:
        return float("nan"), total
    m = d.mean(0)
    s2 = float(np.sum((d - m) ** 2) / (T - 1))
    return max(float(np.sum(m ** 2)) - s2 / T, 0.0), total


def corrected_bias_share(parts, persons):
    """Mean over persons of bias2 over the mean over persons of total, persons with a finite bias2 only."""
    ok = [p for p in persons if math.isfinite(parts[p][0])]
    if not ok:
        return float("nan")
    return float(np.mean([parts[p][0] for p in ok]) / np.mean([parts[p][1] for p in ok]))


def reading_bias(b):
    """The pre-registered reading of the corrected bias share b."""
    if not math.isfinite(b):
        return "NOT_EVALUATED"
    if b >= BIAS_NOROOM:
        return "NO_ROOM"
    if b <= BIAS_ROOM:
        return "ROOM"
    return "BETWEEN"


def select_subset(windows, per_person=TAKES_PER_PERSON, seed=SEED):
    """Window indices of up to per_person whole takes per person, drawn with seed (sorted take ids per person,
    one rng). windows: DataFrame with person_id and take_id, row order = window index."""
    rng = np.random.default_rng(seed)
    out = []
    for p in sorted(windows["person_id"].unique()):
        takes = sorted(windows.loc[windows["person_id"] == p, "take_id"].unique())
        k = min(per_person, len(takes))
        pick = set(rng.choice(len(takes), size=k, replace=False).tolist())
        chosen = {takes[i] for i in pick}
        out += [int(i) for i in np.flatnonzero(windows["take_id"].isin(chosen) & (windows["person_id"] == p))]
    return sorted(out)


# ---------------------------------------------------------------- stage hashes
def stage_hashes(a):
    res = {}
    for name, (fname, stop) in REPORTS.items():
        if name == "CHAL":
            continue
        rel = f"{TASKS}/{fname}"
        with open(_p(rel), encoding="utf-8") as f:
            text = f.read()
        with open(_p(f"{DIAG}/{name}/prediction.sha256"), encoding="utf-8") as f:
            stored, t = parse_hash_line(f.read())
        sec = prediction_section(text, stop)
        got = hashlib.sha256(sec.encode("utf-8")).hexdigest() if sec else None
        stamps = {}
        for o in ERROR_OUTPUTS[name]:
            path = _p(f"{DIAG}/{o}")
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            stamps[o] = {"run_at": doc.get("run_at") or doc.get("provenance", {}).get("run_at"), "finished": doc.get("finished"),
                         "file_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))}
        res[name] = {"report": rel, "stored": stored, "recomputed": got, "verifies": got == stored, "hashed_at": t,
                     "error_outputs": stamps, "report_has_duplicated_tail": _dup_tail(text)}
    res["provenance"] = _provenance("hashes")
    _write_json(os.path.join(a.out, "hashes.json"), res)
    print(json.dumps(res, indent=1))


def _dup_tail(text, k=3):
    """True when the report's last k non-blank lines also occur, in order, earlier in the file (a restored file
    whose old tail stayed appended)."""
    lines = [x for x in text.replace("\r\n", "\n").split("\n") if x.strip()]
    if len(lines) < 2 * k:
        return False
    tail = lines[-k:]
    for j in range(len(lines) - 2 * k + 1):
        if lines[j:j + k] == tail:
            return True
    return False


# ---------------------------------------------------------------- stage attain
def _truth_betas(archive, take):
    from mdfm.data import amass
    seq = amass.load_many(archive, [take])[take]
    return np.asarray(seq["betas"], dtype=np.float64)[:16], str(seq["gender"]).strip().lower()


def stage_attain(a):
    from mdfm.gates import g1
    ds = g1.G1Set("dev")
    W = ds.windows
    sub = select_subset(W)
    sw = W.iloc[sub]
    per = sw.groupby("person_id").agg(windows=("take_id", "size"), takes=("take_id", "nunique"))
    # truth betas: every dev take (truth only)
    tb = {}
    for (arch, take), _ in W.groupby(["archive", "take_id"]):
        tb[take] = _truth_betas(arch, take)[0]
    spread = {}
    for p, g in W.groupby("person_id"):
        bs = np.stack([tb[t] for t in g["take_id"].unique()])
        spread[p] = float(np.abs(bs - bs[0]).max())
    res = {"dev_windows": int(len(W)), "dev_persons": int(W["person_id"].nunique()),
           "subset_windows": len(sub), "subset_takes": int(sw["take_id"].nunique()),
           "subset_persons": int(sw["person_id"].nunique()), "subset_list": sub,
           "subset_per_person": {p: {"windows": int(r.windows), "takes": int(r.takes)} for p, r in per.iterrows()},
           "subset_windows_per_take": {str(k): int(v) for k, v in
                                       sw.groupby("take_id").size().value_counts().sort_index().items()},
           "truth_betas_max_abs_diff_within_person": spread,
           "persons_with_constant_truth_betas": int(sum(v == 0.0 for v in spread.values())),
           "dev_manifest_sha256": ds.manifest["manifest_sha256"]}
    if a.time_windows:
        import torch
        import probe_g1_decompose as P
        from mdfm.eval import infer as I
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        L = I.load_checkpoint(a.checkpoint, device="cpu")
        idx = sub[:a.time_windows]
        wins = [I.g1_windows(ds, j, ("six",))["six"] for j in idx]
        t0 = time.time()
        P.posterior_pass(L.model, wins, STEPS, POINTS, None, SEED, BATCH, keep_amortized=True)
        res["cpu_seconds_per_window"] = (time.time() - t0) / len(idx)
        res["cpu_timed_windows"] = len(idx)
    res["provenance"] = _provenance("attain")
    _write_json(os.path.join(a.out, "attain.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("subset_list",
                                                                "truth_betas_max_abs_diff_within_person")}, indent=1))


# ---------------------------------------------------------------- stage reproduce
def _kw(df, col):
    return float((df[col] * df["n"]).sum() / df["n"].sum())


def g1_stat(df, col):
    """Mean over persons of the knot-weighted person means."""
    return float(np.mean([_kw(g, col) for _, g in df.groupby("person_id")]))


def stage_reproduce(a):
    import pandas as pd
    import g1d_SYN as S
    v1, s2, dj = _load(G1_V1), _load(G1_S2), _load(DECOMPOSE)
    out = {"g1_v1_lines": v1["lines"].get("G1a") if isinstance(v1["lines"], dict) else v1["lines"],
           "g1_s2_lines": s2["lines"].get("G1a") if isinstance(s2["lines"], dict) else s2["lines"]}
    dw = pd.read_csv(_p(DECOMPOSE_WIN), encoding="utf-8")
    gw = pd.read_csv(_p(G1_V1_WIN), encoding="utf-8")
    rows = {}
    for col in ("g1a_m", "pa_m", "true_betas_m", "true_betas_skel_m", "true_root_m", "true_root_local_m",
                "sub_arms_m", "sub_legs_m", "sub_spine_head_m", "sub_arms_local_m", "refine0_m", "best_m", "floor_m",
                "heading_m"):
        rows[col] = g1_stat(dw, col) * MM
    out["decompose_g1_stat_mm"] = rows
    out["decompose_change_mm"] = {c: rows["g1a_m"] - v for c, v in rows.items() if c != "g1a_m"}
    # per-window difference between the decomposition pass (blocks 32) and the harness pass (blocks 64)
    m = dw[["window", "g1a_m"]].merge(gw[["window", "g1a_m"]], on="window", suffixes=("_dec", "_gate"))
    diff = (m["g1a_m_dec"] - m["g1a_m_gate"]) * MM
    out["pass_difference_mm"] = {"windows": int(len(m)), "rms": float(np.sqrt(np.mean(diff ** 2))),
                                 "median_abs": float(np.median(np.abs(diff))), "max_abs": float(np.abs(diff).max()),
                                 "share_above_10mm": float(np.mean(np.abs(diff) > 10.0)),
                                 "gate_window_sd": float(gw["g1a_m"].std() * MM)}
    # shape gain on the G1 statistic, split by archive (each person weighs 1/23)
    dw["s_m"] = dw["g1a_m"] - dw["true_betas_m"]
    per = {p: (g["archive"].iat[0], _kw(g, "s_m") * MM) for p, g in dw.groupby("person_id")}
    n_p = len(per)
    by_arch = {}
    for p, (arch, s) in per.items():
        e = by_arch.setdefault(arch, {"persons": 0, "contribution_mm": 0.0, "person_gains_mm": []})
        e["persons"] += 1
        e["contribution_mm"] += s / n_p
        e["person_gains_mm"].append(round(s, 2))
    out["shape_gain_g1_stat_mm"] = sum(s for _, s in per.values()) / n_p
    out["shape_gain_by_archive_on_g1_stat"] = by_arch
    top = max(per.items(), key=lambda kv: kv[1][1])
    out["shape_gain_top_person"] = {"person": top[0], "gain_mm": top[1][1], "contribution_mm": top[1][1] / n_p}
    # pooling check recomputed with the SYN functions and independently here
    takes = [{"person_id": t["person_id"], "archive": t["archive"], "windows": int(t["windows"]),
              "knots": int(t["knots"]), "s": float(t["g1a_mm"]) - float(t["true_betas_mm"]),
              "g1a": float(t["g1a_mm"])} for t in dj["per_take"]]
    el = S.eligible_persons(takes)
    varied = sorted(p for p in {t["person_id"] for t in takes}
                    if len({t["windows"] for t in takes if t["person_id"] == p}) >= 2)
    out["pooling_recomputed"] = {"few_minus_many_mm": S.few_minus_many(takes, el),
                                 "within_slope_mm_per_doubling": S.within_slope(takes, varied),
                                 "eligible": el, "varied_persons": len(varied)}
    # what a pure independent-noise shape error would predict for that slope: s proportional to the shape error,
    # which scales as windows^-1/2, so s(2n) = s(n) / sqrt(2): slope = -(1 - 1/sqrt 2) x mean s of varied persons
    s_mean = float(np.mean([np.average([t["s"] for t in takes if t["person_id"] == p],
                                       weights=[t["knots"] for t in takes if t["person_id"] == p])
                            for p in varied]))
    out["pooling_noise_model"] = {"mean_s_varied_persons_mm": s_mean,
                                  "expected_slope_if_linear_in_error": -(1 - 2 ** -0.5) * s_mean,
                                  "expected_slope_if_quadratic_in_error": -0.5 * s_mean}
    # Hlayout and Hstand outputs cited by the note
    hl, hs, hp = _load(f"{DIAG}/Hlayout/results.json"), _load(f"{DIAG}/Hstand/analysis.json"), \
        _load(f"{DIAG}/Hprior/results.json")
    out["hlayout_arms"] = hl.get("arms")
    out["hstand_runs_keys"] = list(hs.get("runs", {}).keys())
    out["hprior_c2"] = hp.get("c2") if "c2" in hp else {k: hp[k] for k in list(hp)[:6]}
    out["provenance"] = _provenance("reproduce", [G1_V1, G1_S2, DECOMPOSE, DECOMPOSE_WIN, G1_V1_WIN])
    _write_json(os.path.join(a.out, "reproduce.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("hlayout_arms", "hprior_c2")}, indent=1,
                     default=str))


# ---------------------------------------------------------------- stage shape
def stage_shape(a):
    import torch
    import probe_g1_decompose as P
    from mdfm.eval import infer as I
    from mdfm.gates import g1
    from mdfm.train import provenance as PV
    from mdfm.gates import common
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    with open(os.path.join(OUT, "attain.json"), encoding="utf-8") as f:
        att = json.load(f)
    ds = g1.G1Set("dev")
    sub = select_subset(ds.windows)
    if sub != att["subset_list"]:
        raise ValueError("subset differs from attain.json")
    if a.limit:
        sub = sub[:a.limit]
    L = I.load_checkpoint(a.checkpoint, device="cpu")
    if set(ds.persons) & set(L.persons.get("amass", ())):
        raise ValueError("a dev person among the checkpoint's training persons")
    model = L.model
    t0 = time.time()
    mu20, lv20, mu0, lv0 = [], [], [], []
    for b0 in range(0, len(sub), BATCH):
        idx = sub[b0:b0 + BATCH]
        wins = [I.g1_windows(ds, j, ("six",))["six"] for j in idx]
        take0 = str(ds.windows["take_id"].iat[idx[0]])
        st = P.posterior_pass(model, wins, STEPS, POINTS, None, I.seed_of(SEED, "dev", take0, "six"), BATCH,
                              keep_amortized=True)
        for c in st["chunks"]:
            mu20.append(c["post"].mu_b.float().cpu().numpy())
            lv20.append(c["post"].logvar_b.float().cpu().numpy())
            mu0.append(c["post0"].mu_b.float().cpu().numpy())
            lv0.append(c["post0"].logvar_b.float().cpu().numpy())
        print(f"{b0 + len(idx)} / {len(sub)} windows, {time.time() - t0:.0f} s", flush=True)
    W = ds.windows.iloc[sub]
    tb, gtrue = {}, {}
    for (arch, take), _ in W.groupby(["archive", "take_id"]):
        tb[take], gtrue[take] = _truth_betas(arch, take)
    from mdfm.body.smplh import gender_code
    np.savez_compressed(os.path.join(a.out, "shape.npz"), window=np.asarray(sub), mu20=np.concatenate(mu20),
                        lv20=np.concatenate(lv20), mu0=np.concatenate(mu0), lv0=np.concatenate(lv0),
                        beta_true=np.stack([tb[t] for t in W["take_id"]]),
                        gender=np.asarray([gender_code(g) for g in W["gender"]], dtype=np.int64),
                        take_id=W["take_id"].to_numpy().astype(str), person_id=W["person_id"].to_numpy().astype(str),
                        archive=W["archive"].to_numpy().astype(str))
    _write_json(os.path.join(a.out, "shape_run.json"), {
        "checkpoint": L.describe(), "steps": STEPS, "points": POINTS, "batch_windows": BATCH, "seed": SEED,
        "windows": len(sub), "limit": a.limit, "dev_manifest_sha256": ds.manifest["manifest_sha256"],
        "code_sha256": common.code_sha256([os.path.abspath(__file__), P.__file__, I.__file__, g1.__file__]),
        "moveatlas_head": PV.moveatlas_head(), "seconds": round(time.time() - t0, 1),
        "gender_mismatch_windows": int(sum(gender_code(gtrue[t]) != gender_code(g) for t, g in
                                           zip(W["take_id"], W["gender"]))),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ---------------------------------------------------------------- stage read
def shape_readings(rest_fn, z):
    """All statistics of stage read from the arrays of shape.npz (z: dict-like) and rest_fn(beta (W,16), gender
    (W,)) -> rest joints (W, 22, 3) m."""
    persons, takes = z["person_id"], z["take_id"]
    truth = rest_fn(z["beta_true"], z["gender"])
    res = {}
    for tag in ("20", "0"):
        mu, lv = z["mu" + tag], z["lv" + tag]
        est = {"window": mu, "take": pooled_means(mu, lv, takes), "person": pooled_means(mu, lv, persons),
               "person_plain_mean": pooled_means(mu, np.zeros_like(lv), persons)}
        err = {k: skeleton_error_mm(rest_fn(v, z["gender"]), truth) for k, v in est.items()}
        r = {f"skeleton_error_{k}_mm": person_mean(v, persons) for k, v in err.items()}
        ratio = lambda ps: (np.mean([err["person"][persons == p].mean() for p in ps])  # noqa: E731
                            / np.mean([err["take"][persons == p].mean() for p in ps]))
        r["ratio_person_over_take"] = float(ratio(sorted(set(persons))))
        r["ratio_interval"] = person_boot(ratio, persons)
        # window-level bias share (descriptive) and the take-level corrected share (the criterion)
        Rw = rest_fn(mu, z["gender"])
        D = (Rw - Rw[:, :1]) - (truth - truth[:, :1])
        Rt = rest_fn(est["take"], z["gender"])
        Dt = (Rt - Rt[:, :1]) - (truth - truth[:, :1])
        shares = {p: bias_share(D[persons == p]) for p in sorted(set(persons))}
        r["window_bias_share_mean"] = float(np.mean(list(shares.values())))
        parts = {}
        for p in sorted(set(persons)):
            sel = persons == p
            tk = takes[sel]
            first = [np.flatnonzero(tk == t)[0] for t in sorted(set(tk))]
            parts[p] = take_bias_parts(Dt[sel][first])
        r["take_bias_parts_m2"] = {p: list(v) for p, v in parts.items()}
        r["corrected_bias_share"] = corrected_bias_share(parts, sorted(parts))
        r["corrected_bias_share_interval"] = person_boot(lambda ps: corrected_bias_share(parts, ps), persons)
        r["removable_share_of_squared_offset"] = 1.0 - r["corrected_bias_share"]
        r["removable_share_of_offset_magnitude"] = 1.0 - math.sqrt(r["corrected_bias_share"])
        by_arch = {}
        for arch in sorted(set(z["archive"])):
            ps = sorted(set(persons[z["archive"] == arch]))
            by_arch[arch] = {"persons": len(ps),
                             "take_mm": float(np.mean([err["take"][persons == p].mean() for p in ps])),
                             "person_mm": float(np.mean([err["person"][persons == p].mean() for p in ps])),
                             "corrected_bias_share": corrected_bias_share(parts, ps)}
        r["by_archive"] = by_arch
        res[tag] = r
    res["corr_mu0_mu20"] = float(np.corrcoef(z["mu0"].ravel(), z["mu20"].ravel())[0, 1])
    res["reading"] = reading_bias(res["20"]["corrected_bias_share"])
    res["reading_0_steps"] = reading_bias(res["0"]["corrected_bias_share"])
    return res


def stage_read(a):
    import torch
    from mdfm.eval import infer as I
    z = dict(np.load(os.path.join(a.out, "shape.npz")))
    L = I.load_checkpoint(a.checkpoint, device="cpu")
    sk = L.model.skeleton

    def rest_fn(beta, gender):
        with torch.no_grad():
            return sk.rest(torch.as_tensor(np.asarray(beta), dtype=torch.float32),
                           torch.as_tensor(np.asarray(gender), dtype=torch.int64)).double().numpy()

    res = shape_readings(rest_fn, z)
    res["windows"] = int(len(z["window"]))
    res["persons"] = int(len(set(z["person_id"])))
    res["provenance"] = _provenance("read")
    _write_json(os.path.join(a.out, "shape.json"), res)
    r20, r0 = res["20"], res["0"]
    md = ["# Shape test of lever 1 (scripts/g1d_CHAL.py --stage read)", "",
          f"{res['windows']} dev windows, {res['persons']} persons, whole takes ({TAKES_PER_PERSON} per person at "
          "most). Rest-pose skeleton error, 22 joints, pelvis-relative, mean over persons of the person means, mm.", "",
          "| steps | window shape | take-pooled | person-pooled (subset) | person plain mean | person over take |"
          " interval | corrected bias share | interval | window bias share |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in (("20", r20), ("0", r0)):
        md.append(f"| {tag} | {r['skeleton_error_window_mm']:.2f} | {r['skeleton_error_take_mm']:.2f} | "
                  f"{r['skeleton_error_person_mm']:.2f} | {r['skeleton_error_person_plain_mean_mm']:.2f} | "
                  f"{r['ratio_person_over_take']:.3f} | {r['ratio_interval'][0]:.3f} to {r['ratio_interval'][1]:.3f} | "
                  f"{r['corrected_bias_share']:.3f} | {r['corrected_bias_share_interval'][0]:.3f} to "
                  f"{r['corrected_bias_share_interval'][1]:.3f} | {r['window_bias_share_mean']:.3f} |")
    md += ["", f"Removable by unlimited person pooling, 20 steps: {r20['removable_share_of_squared_offset']:.3f} of the "
           f"squared take-pooled offset, {r20['removable_share_of_offset_magnitude']:.3f} of its magnitude (lever 1 "
           f"needs {NEED} of the true-betas ceiling).",
           "By archive (20 steps): " + "; ".join(
               f"{k} {v['persons']} persons, take {v['take_mm']:.2f}, person {v['person_mm']:.2f}, corrected bias "
               f"share {v['corrected_bias_share']:.3f}" for k, v in r20["by_archive"].items()) + ".",
           f"Correlation of the 0-step and 20-step window shape means: {res['corr_mu0_mu20']:.4f}.",
           f"Reading (20 steps, the criterion): {res['reading']}; at 0 steps: {res['reading_0_steps']}.", ""]
    _write(os.path.join(a.out, "shape.md"), "\n".join(md))
    print("\n".join(md))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("hashes", "attain", "reproduce", "shape", "read"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--time-windows", type=int, default=0, help="attain: time the shape pass on this many windows")
    ap.add_argument("--limit", type=int, default=0, help="shape: first windows of the subset only (smoke test)")
    return ap.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "-1" and a.stage in ("attain", "shape", "read"):
        raise SystemExit("set CUDA_VISIBLE_DEVICES=-1: this script never uses the GPU")
    os.makedirs(a.out, exist_ok=True)
    {"hashes": stage_hashes, "attain": stage_attain, "reproduce": stage_reproduce, "shape": stage_shape,
     "read": stage_read}[a.stage](a)


if __name__ == "__main__":
    main()
