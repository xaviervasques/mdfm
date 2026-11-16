"""G12 readouts (docs/rules/G12_RULE.md sections 6 and 7; revision docs/rules/G12_RULE_REVISION_2026-10-08.md item 8;
task G12R): the frozen tokens of a G12 baseline checkpoint (or of a body checkpoint, the representation row), the
section 6 probes fitted on the 54 training-fold weargait_pd controls, and the three G3 dev lines of the chosen
variants with their per-row predictions. Dev only: no lockbox, EVAL or CODY record, no sealed read, no comparison
(mdfm/baselines/comparison.py, scripts/g12_compare.py).

This module is the reading side of the baselines: it reads labels (the G3 walkway targets) and is imported by no module
of the training side (config, data, model, compute, trainer).

Weights (load_any): a baseline checkpoint through mdfm/baselines/load.load_frozen (the checkpoint checks of
train/checkpoint.load, tokenizer and encoder only, EMA weights by default); a body checkpoint (no g12 section in its
configuration) through mdfm/eval/infer.load_checkpoint, read the same way (tokenizer and encoder only).

Tokens (encode_windows, trial_rows): the tokenizer and the encoder of the model, one window per row, under the
standard typed mask (TokenBatch.mask(), no hidden token, the input every arm meets at probe time), bf16 autocast on
CUDA, no refinement, no decoder, no latent head; the G3 inference windows (mdfm/eval/infer.record_windows: band copy,
10 s windows every 5 s, the last one on the record's end), each 0.25 s patch read from the window where it is most
central (mdfm/eval/probes.most_central). Variants (VARIANTS), as J0 (scripts/probe_frozen_tokens.py) reads them less
mu_z, which a no-body model has not: latent (the encoder output at the latent position), measurement (the encoder
outputs at the patch's measurement tokens, averaged), stem (the tokenizer outputs there, before attention), subject
(the encoder output at the window's subject token). The rule's candidates (CANDIDATES) are latent and measurement;
stem and subject are reported.

Probes (mdfm/eval/probes.py, as J0 runs them): ridge regression, the penalty chosen by person folds inside the
fitting set (probes.fit_probe). Pass probes (stride_m, speed_m_s): per walkway pass the mean and the standard
deviation of the token rows whose patch centre lies in the pass span (subject: the mean); every walkway stride of a
pass gets its pass's prediction (probes.pass_predictions). IC probe per 50 ms sub-patch (probes.ic_targets,
probes.decode_ics) on the latent, measurement and stem token rows (IC_VARIANTS), fitted on the labelled rows of the
fitting set, applied to the dev rows near a walkway pass or IC (KEEP_PAD_S).

Variant choice (choose, revision item 8): per line, the candidate with the lower person cross-fitted line on the
fitting set (the 54), ties to the first candidate (latent); never a dev or lockbox person. Pass lines: probes.cross_fit
(outer person folds, the penalty chosen inside each fitting part) scored by g3.stride_lines and g3.speed_line on the
54's targets. IC line: cross_fit_scores (the same outer folds over the token rows' persons, each fit on the labelled
rows of the other folds) decoded, one constant offset fitted on these out-of-fold ICs (fit_offset_on: the median of
predicted minus reference over the nearest predictions within g3.IC_MATCH_S, g3's rule on the scored set), then
g3.ic_line on the 54's walkway references.

Dev lines (the G3 harness's own statistics, comparable with the body's dev rows): every probe fitted once on all 54
and applied to the dev controls; stride_mae_cm and speed_mae_m_s by g3.stride_lines and g3.speed_line, ic_median_ms
(walkway reference) by g3.ic_line with one offset per variant fitted on the 14 dev controls (g3.fit_offsets, the
harness's offset rule); on the 14 and on the 12 (g3.DEV_SETS: the 14 without WHC003 and WHC026), with the offset of
the 14 on both, as the harness's sets do. Intervals: g3's person bootstrap (2,000 resamples, seed 20260929).

Per-row predictions (rows_of, comparison.ROW_COLUMNS): every walkway stride, pass and walkway reference IC of the 14
with its person, for every variant, the chosen ones flagged; run_probes checks that the line recomputed from the rows
(comparison.line_point) equals g3's point on both sets for every variant and line.
"""
import json
import os
import time

import numpy as np

from . import comparison as CMP

VARIANTS = ("latent", "measurement", "stem", "subject")
CANDIDATES = ("latent", "measurement")
IC_VARIANTS = ("latent", "measurement", "stem")
TOKEN_VARIANTS = IC_VARIANTS
TARGETS = ("stride_m", "speed_m_s")
LINES = tuple(CMP.LINES)                       # stride_mae_cm, speed_mae_m_s, ic_median_ms
PASS_LINE = {"stride_mae_cm": "stride_m", "speed_mae_m_s": "speed_m_s"}
KEEP_PAD_S = 1.0
BATCH_WINDOWS = 16
LABEL = "G12 readout, dev only, not a G12 comparison"


# ---------------------------------------------------------------- loading
def kind_of(path):
    """'baseline' or 'body' from the config.json of the checkpoint's run folder (a G12 run stores {'g12', 'run'}),
    None without a readable one."""
    p = os.path.join(os.path.dirname(os.path.abspath(str(path))), "config.json")
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    return "baseline" if isinstance(cfg, dict) and "g12" in cfg else "body"


def load_any(path, weights="ema", device="cpu"):
    """(kind, model, info) of a checkpoint: kind 'baseline' (load.load_frozen) or 'body' (infer.load_checkpoint, a
    checkpoint whose configuration has no g12 section), told by kind_of so that the checkpoint is loaded once (each
    load may log a tolerated records drift beside it, as every checkpoint read does), else by load_frozen's refusal;
    info {path, name, sha256, weights, step, phase, arm, seed, config_sha256, trained_persons (set), describe}."""
    from ..eval import infer as I
    from ..gates import common
    from ..train import config as CF
    from ..train import provenance as PV
    from . import load as BL
    fz = None
    kind = kind_of(path)
    if kind != "body":
        try:
            fz = BL.load_frozen(path, weights=weights, device=device)
        except ValueError as e:
            if kind == "baseline" or "is not a G12 baseline checkpoint" not in str(e):
                raise
    if fz is not None:
        g12 = fz.config["g12"]
        persons = frozenset(p for ids in fz.persons.values() for p in ids)
        desc = {"path": common.rel(fz.path), "name": I.checkpoint_name(fz.path), "sha256": fz.sha256,
                "weights": fz.weights, "step": fz.step, "phase": fz.phase, "kind": "baseline", "arm": fz.arm,
                "seed": int(g12["seed"]), "config_sha256": fz.config_sha256,
                "parameters": {"tokenizer": int(sum(p.numel() for p in fz.model.tokenizer.parameters())),
                               "encoder": int(sum(p.numel() for p in fz.model.encoder.parameters()))},
                "persons": {s: len(v) for s, v in sorted(fz.persons.items())},
                "persons_sha256": PV.list_hashes(fz.persons), "drift": list(fz.drift)}
        return "baseline", fz.model, {**desc, "trained_persons": persons, "describe": desc}
    ld = I.load_checkpoint(path, weights=weights, device=device)
    desc = ld.describe()
    desc.update(kind="body", arm="body", seed=int(CF.from_json(ld.config_json).seed))
    return "body", ld.model, {**desc, "trained_persons": ld.trained_persons(), "describe": desc}


# ---------------------------------------------------------------- tokens
def _amp(device, amp=None):
    import torch
    dev = torch.device(device)
    on = dev.type == "cuda" if amp is None else bool(amp)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=on)


def encode_windows(model, windows, batch_windows=BATCH_WINDOWS, amp=None):
    """[{variant: array}] per window: latent, measurement, stem (P, d) float32 per latent patch, n_measurement (P,)
    (NaN measurement and stem rows where 0) and subject (d,). model: anything with .tokenizer, .encoder and .device
    (a G12Model or an MFM); the tokenizer and the encoder run under no_grad with the standard typed mask."""
    import torch

    from ..model import tokenize as TK
    dev = model.device
    out = []
    B = max(1, int(batch_windows))
    for a in range(0, len(windows), B):
        ch = list(windows[a:a + B])
        tb = TK.tokenize(ch).to(dev)
        with torch.no_grad():
            with _amp(dev, amp):
                x = model.tokenizer(tb)
                h = model.encoder(x, tb.t.to(x.device), tb.mask().to(x.device))
            R, L, d = h.shape
            hf = h.float().reshape(R * L, d)
            xf = x.float().reshape(R * L, d)
            lat = tb.lat_idx
            Pmax = int(lat.shape[1])
            hl = hf[lat.clamp_min(0)]
            kind, seg, t = tb.kind.reshape(-1), tb.seg.reshape(-1), tb.t.reshape(-1)
            m = kind == TK.MEASUREMENT
            patch = torch.clamp(torch.floor(t[m] / TK.PATCH_S).long(), 0, Pmax - 1)
            key = seg[m] * Pmax + patch
            n = len(ch) * Pmax
            hs = torch.zeros(n, d, device=hf.device).index_add_(0, key, hf[m])
            xs = torch.zeros(n, d, device=hf.device).index_add_(0, key, xf[m])
            cnt = torch.bincount(key, minlength=n).to(hf.dtype)
            safe = cnt.clamp_min(1.0)[:, None]
            nan = torch.full_like(hs, float("nan"))
            hm = torch.where(cnt[:, None] > 0, hs / safe, nan).reshape(len(ch), Pmax, d)
            xm = torch.where(cnt[:, None] > 0, xs / safe, nan).reshape(len(ch), Pmax, d)
            res = {"latent": hl.cpu().numpy(), "measurement": hm.cpu().numpy(), "stem": xm.cpu().numpy(),
                   "n_measurement": cnt.reshape(len(ch), Pmax).cpu().numpy(), "subject": hf[tb.subj_idx].cpu().numpy()}
        npat = tb.n_patches.cpu().numpy()
        for j in range(len(ch)):
            P = int(npat[j])
            out.append({k: (v[j] if k == "subject" else v[j, :P]).astype(np.float32) for k, v in res.items()})
    return out


def trial_rows(feats, windows, patch_s=0.25):
    """Token rows of one trial, the most central reading of each patch only: start and centre (s, trial clock),
    window index and every variant (subject repeated over its window's rows)."""
    from ..eval import probes
    starts = np.array([float(w.t0) for w in windows])
    durs = np.array([float(w.duration) for w in windows])
    k = np.concatenate([np.full(f["latent"].shape[0], i) for i, f in enumerate(feats)]).astype(np.int64)
    p = np.concatenate([np.arange(f["latent"].shape[0]) for f in feats]).astype(np.int64)
    a = starts[k] + p * patch_s
    c = a + patch_s / 2.0
    central = probes.most_central(starts, durs, k, c)
    rows = {"start": a, "centre": c, "window": k}
    for v in ("latent", "measurement", "stem"):
        rows[v] = np.concatenate([f[v] for f in feats])
    rows["subject"] = np.stack([f["subject"] for f in feats])[k]
    return {key: val[central] for key, val in rows.items()}


def keep_mask(centre, spans, ics, pad=KEEP_PAD_S):
    """(N,) bool: a row centre within pad of a span (t0, t1) or of a walkway IC time."""
    c = np.asarray(centre, dtype=np.float64)
    keep = np.zeros(len(c), bool)
    for a, b in np.asarray(spans, dtype=np.float64).reshape(-1, 2):
        keep |= (c >= a - pad) & (c <= b + pad)
    for t in np.asarray(ics, dtype=np.float64).reshape(-1):
        keep |= np.abs(c - t) <= pad
    return keep


def check_record(tid, rec, persons):
    """AssertionError unless the record carries lockbox False, no EVAL tier, and a person of `persons`."""
    lb = rec.get("lockbox")
    if lb is None or (lb is not False and str(lb).strip().lower() not in ("false", "0")):
        raise AssertionError(f"{tid}: record with lockbox {lb!r}")
    if str(rec.get("tier", "")) in ("EVAL", "EXCLUDED_V1"):
        raise AssertionError(f"{tid}: record of tier {rec.get('tier')}")
    if rec.get("person_id") not in set(persons):
        raise AssertionError(f"{tid}: person {rec.get('person_id')!r} outside the requested set")


def collect(encode, records, units, targets, persons, log=print):
    """Features of one person set. encode: windows -> [per-window dict] (encode_windows bound to a model); records:
    iterable of (trial_id, model input record); units: probes.pass_units of the set's targets. Returns {'X':
    {variant: (n_units, D)}, 'n_rows' (n_units,), 'tokens' {trial_id, person_id, start, latent, measurement, stem
    (float16)}, 'counts'}."""
    from ..eval import infer as I
    from ..eval import probes
    by_trial = {t: g.index.to_numpy() for t, g in units.groupby("trial_id")}
    ics = {t: g["t_s"].to_numpy() for t, g in targets.walkway_ic.groupby("trial_id")}
    X = {v: [None] * len(units) for v in VARIANTS}
    n_rows = np.zeros(len(units), np.int64)
    tok = {k: [] for k in ("trial_id", "person_id", "start") + TOKEN_VARIANTS}
    n_rec = n_win = 0
    seen = set()
    t0 = time.time()
    for tid, rec in records:
        check_record(tid, rec, persons)
        seen.add(tid)
        n_rec += 1
        wins, _, _ = I.record_windows(rec)
        if not wins:
            continue
        n_win += len(wins)
        rows = trial_rows(encode(wins), wins)
        idx = by_trial.get(tid, np.zeros(0, np.int64))
        spans = units.loc[idx, ["t0", "t1"]].to_numpy(np.float64) if len(idx) else np.zeros((0, 2))
        if len(idx):
            for v in VARIANTS:
                how = ("mean",) if v == "subject" else probes.POOL
                f, n = probes.pool_spans(rows["centre"], rows[v], spans, how)
                for j, i in enumerate(idx):
                    X[v][i] = f[j]
                if v == "latent":
                    n_rows[idx] = n
        keep = keep_mask(rows["centre"], spans, ics.get(tid, ()))
        m = int(keep.sum())
        tok["trial_id"] += [tid] * m
        tok["person_id"] += [str(rec.get("person_id"))] * m
        tok["start"].append(rows["start"][keep])
        for v in TOKEN_VARIANTS:
            tok[v].append(rows[v][keep].astype(np.float16))
        log(f"  {tid}: {len(wins)} windows, {len(idx)} passes, {m} token rows kept ({time.time() - t0:.0f} s)",
            flush=True)
    dims = {v: next((x.shape[0] for x in X[v] if x is not None), 0) for v in VARIANTS}
    Xa = {v: np.stack([x if x is not None else np.full(dims[v], np.nan) for x in X[v]]) if len(units)
          else np.zeros((0, dims[v])) for v in VARIANTS}
    tokens = {"trial_id": np.asarray(tok["trial_id"], dtype=str), "person_id": np.asarray(tok["person_id"], dtype=str),
              "start": np.concatenate(tok["start"]) if tok["start"] else np.zeros(0)}
    for v in TOKEN_VARIANTS:
        tokens[v] = np.concatenate(tok[v]) if tok[v] else np.zeros((0, 0), np.float16)
    counts = {"records": n_rec, "windows": n_win, "passes": int(len(units)),
              "passes_with_rows": int((n_rows > 0).sum()), "passes_without_rows": int((n_rows == 0).sum()),
              "trials_without_record": sorted(set(units["trial_id"]) - seen), "token_rows_kept": int(len(tokens["start"])),
              "rows_per_pass_median": float(np.median(n_rows[n_rows > 0])) if (n_rows > 0).any() else None}
    return {"X": Xa, "n_rows": n_rows, "tokens": tokens, "counts": counts}


# ---------------------------------------------------------------- feature file
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_clean(v) for v in (sorted(o) if isinstance(o, (set, frozenset)) else o)]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return _clean(o.item())
    return o


def save_features(path, data, meta):
    """data {'units': {set: DataFrame}, set: collect() output} and meta (dict) into one npz (no pickle)."""
    arrs = {"meta": np.asarray(json.dumps(_clean(meta)))}
    for s in ("train", "dev"):
        arrs[f"{s}__units"] = np.asarray(data["units"][s].to_json(orient="split"))
        d = data[s]
        for v in VARIANTS:
            arrs[f"{s}__X__{v}"] = d["X"][v].astype(np.float32)
        arrs[f"{s}__n_rows"] = d["n_rows"]
        arrs[f"{s}__counts"] = np.asarray(json.dumps(_clean(d["counts"])))
        for k, v in d["tokens"].items():
            arrs[f"{s}__tok__{k}"] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)
    return path


def load_features(path):
    """(data, meta) of save_features."""
    import io

    import pandas as pd
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    data = {"units": {}}
    for s in ("train", "dev"):
        u = pd.read_json(io.StringIO(str(z[f"{s}__units"])), orient="split")
        u["pass_id"] = u["pass_id"].astype(str)
        data["units"][s] = u
        data[s] = {"X": {v: z[f"{s}__X__{v}"].astype(np.float64) for v in VARIANTS}, "n_rows": z[f"{s}__n_rows"],
                   "counts": json.loads(str(z[f"{s}__counts"])),
                   "tokens": {k: z[f"{s}__tok__{k}"] for k in ("trial_id", "person_id", "start") + TOKEN_VARIANTS}}
    return data, meta


# ---------------------------------------------------------------- probes
def strip(stats):
    """Line stats without the per-person table (kept separately)."""
    return {k: v for k, v in stats.items() if k != "per_person"}


def choose(crossfit, candidates=CANDIDATES):
    """The candidate with the lowest cross-fitted point (a NaN never chosen over a number; ties to the earlier
    candidate)."""
    best, key = None, None
    for v in candidates:
        x = crossfit.get(v)
        x = float("nan") if x is None else float(x)
        k = (np.isnan(x), x if not np.isnan(x) else 0.0)
        if best is None or k < key:
            best, key = v, k
    return best


def fit_offset_on(pred_ics, ref_ics, tol=None):
    """(offset_s, n matched) by g3's offset rule (the median of predicted minus reference over the nearest
    predictions within tol) on any set of persons; (NaN, 0) without a match. g3.fit_offset itself fits on the dev
    controls only; this is its rule for the fitting set's cross-fitted ICs."""
    from ..gates import g3
    tol = g3.IC_MATCH_S if tol is None else tol
    d = g3.ic_differences(pred_ics, ref_ics, 0.0, tol)
    d = d[d["matched"]]
    return (float(np.median(d["diff_s"])), int(len(d))) if len(d) else g3.UNFITTED


def cross_fit_scores(X, Y, persons, fit_mask, k=None, seed=None):
    """(out-of-fold scores (N, T), [alpha of each fold]): the outer person folds of probes.cross_fit_splits over every
    row; in each, a probe (probes.fit_probe, the penalty chosen inside by person folds, seed + 1) fitted on the rows of
    the other folds where fit_mask holds, applied to every row of the held-out fold."""
    from ..eval import probes
    k = probes.FOLDS if k is None else k
    seed = probes.SEED if seed is None else seed
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    p = np.asarray(persons).astype(str)
    fm = np.asarray(fit_mask, dtype=bool)
    out = np.full((len(X), Y.shape[1]), np.nan)
    alphas = []
    for tr, te in probes.cross_fit_splits(p, k, seed):
        f = tr[fm[tr]]
        probes.check_disjoint(p[f], p[te])
        m, sel = probes.fit_probe(X[f], Y[f], p[f], probes.ALPHAS, k, seed + 1)
        out[te] = m.predict(X[te])
        alphas.append(sel["alpha"])
    return out, alphas


def dev_sets(tg_dev):
    """({set: persons}, {set: Targets}) of the harness's dev sets (g3.DEV_SETS: dev14, dev12)."""
    from ..eval import probes
    from ..gates import g3
    d14 = sorted(tg_dev.persons)
    sets = {name: sorted(set(d14) - set(out)) for name, out in g3.DEV_SETS.items()}
    return sets, {name: (tg_dev if set(p) == set(d14) else probes.subset_targets(tg_dev, p)) for name, p in sets.items()}


def _pass_rows(table, line, variant, value_col, pred_col):
    import pandas as pd
    t = table
    n = len(t)
    nan = np.full(n, np.nan)
    if line == "stride_mae_cm":
        target, pred = t["length_m"].to_numpy(np.float64), t["pred_m"].to_numpy(np.float64)
        foot, t0, t1 = t["foot"].astype(str).to_numpy(), t["t0"].to_numpy(np.float64), t["t1"].to_numpy(np.float64)
    else:
        target, pred = t["speed_m_s"].to_numpy(np.float64), t["pred_m_s"].to_numpy(np.float64)
        foot, t0, t1 = np.full(n, ""), t["t0"].to_numpy(np.float64), t["t1"].to_numpy(np.float64)
    matched = t["matched"].to_numpy(bool)
    return pd.DataFrame({"line": line, "variant": variant, "chosen": False, "person_id": t["person_id"].astype(str),
                         "trial_id": t["trial_id"].astype(str), "task": t["task"].astype(str),
                         "pass_id": t["pass_id"].astype(str), "foot": foot, "t0": t0, "t1": t1, "t_s": nan,
                         "target": target, "prediction": pred, "matched": matched, "enters": matched,
                         "value": np.abs(t[value_col].to_numpy(np.float64))})[CMP.ROW_COLUMNS]


def _ic_rows(table, variant):
    import pandas as pd
    t = table
    n = len(t)
    return pd.DataFrame({"line": "ic_median_ms", "variant": variant, "chosen": False,
                         "person_id": t["person_id"].astype(str), "trial_id": t["trial_id"].astype(str),
                         "task": t["task"].astype(str), "pass_id": "", "foot": t["foot"].astype(str),
                         "t0": np.full(n, np.nan), "t1": np.full(n, np.nan), "t_s": t["t_s"].to_numpy(np.float64),
                         "target": t["t_s"].to_numpy(np.float64), "prediction": t["pred_t_s"].to_numpy(np.float64),
                         "matched": t["matched"].to_numpy(bool), "enters": t["covered"].to_numpy(bool),
                         "value": t["abs_err_ms"].to_numpy(np.float64)})[CMP.ROW_COLUMNS]


def pass_stage(data, tg_train, tg_dev, variants=VARIANTS, log=print):
    """{variant: {'selection', 'crossfit' {line: stats}, 'dev' {set: {line: stats}}, 'per_person' {set: {line:
    {person: value}}}, 'n_features'}}, {variant: rows of the 14}, {variant: dev pass predictions}: the pass probes
    (module docstring). Every fit holds training-fold rows only (probes.check_disjoint on the persons)."""
    from ..eval import probes
    from ..gates import g3
    u_tr, u_dev = data["units"]["train"], data["units"]["dev"]
    p_tr = u_tr["person_id"].to_numpy().astype(str)
    probes.check_disjoint(p_tr, u_dev["person_id"])
    _, tgs = dev_sets(tg_dev)
    out, rows, preds = {}, {}, {}
    for v in variants:
        t0 = time.time()
        Xtr, Xdev = data["train"]["X"][v], data["dev"]["X"][v]
        pred, oof, sel = {}, {}, {}
        for tgt in TARGETS:
            y = u_tr[tgt].to_numpy(np.float64)
            m, s = probes.fit_probe(Xtr, y, p_tr)
            pred[tgt] = m.predict(Xdev)
            oof[tgt], alphas = probes.cross_fit(Xtr, y, p_tr)
            sel[tgt] = {"alpha": s["alpha"], "crossfit_alphas": alphas,
                        "fit_rows": int((np.isfinite(Xtr).all(axis=1) & np.isfinite(y)).sum())}
        p_oof = probes.pass_predictions(u_tr, oof, tg_train, v)
        cf = {"stride_mae_cm": g3.stride_lines(p_oof, tg_train)[0]["stride_mae_cm"],
              "speed_mae_m_s": g3.speed_line(p_oof, tg_train)[0]}
        p_dev = probes.pass_predictions(u_dev, pred, tg_dev, v)
        dev, per_person = {}, {}
        for sname, tg in tgs.items():
            st, st_tab = g3.stride_lines(p_dev, tg)
            sp, sp_tab = g3.speed_line(p_dev, tg)
            dev[sname] = {"stride_mae_cm": strip(st["stride_mae_cm"]), "speed_mae_m_s": strip(sp)}
            per_person[sname] = {"stride_mae_cm": st["stride_mae_cm"].get("per_person", {}),
                                 "speed_mae_m_s": sp.get("per_person", {})}
            if tg is tg_dev:
                rows[v] = [_pass_rows(st_tab, "stride_mae_cm", v, "err_cm", "pred_m"),
                           _pass_rows(sp_tab, "speed_mae_m_s", v, "err_m_s", "pred_m_s")]
        out[v] = {"selection": sel, "crossfit": {ln: strip(s) for ln, s in cf.items()}, "dev": dev,
                  "per_person": per_person, "n_features": int(Xtr.shape[1])}
        preds[v] = pred
        log(f"pass probe {v}: cross-fitted stride {cf['stride_mae_cm']['point']:.3f} cm, speed "
            f"{cf['speed_mae_m_s']['point']:.4f} m/s; dev14 stride {dev['dev14']['stride_mae_cm']['point']:.3f} cm "
            f"({time.time() - t0:.1f} s)", flush=True)
    return out, rows, preds


def ic_stage(data, tg_train, tg_dev, variants=IC_VARIANTS, log=print):
    """({'labelled_rows', 'rows_train', 'rows_dev', 'variants' {variant: {'alpha', 'offset_ms', 'offset_n',
    'crossfit' {'line', 'offset_ms', 'offset_n', 'alphas', 'n_ics'}, 'dev' {set: stats}, 'per_person', 'n_ics'}}},
    {variant: rows of the 14}): the IC probe (module docstring)."""
    from ..eval import probes
    from ..gates import g3
    u_tr = data["units"]["train"]
    tr, dv = data["train"]["tokens"], data["dev"]["tokens"]
    p_tr = np.asarray(tr["person_id"]).astype(str)
    probes.check_disjoint(p_tr, dv["person_id"])
    _, tgs = dev_sets(tg_dev)
    Y, lab = probes.ic_targets(tr["start"], tr["trial_id"], tg_train.walkway_ic, u_tr[["trial_id", "t0", "t1"]])
    ref_tr = g3.reference_ics(tg_train, "walkway")
    out = {"labelled_rows": int(lab.sum()), "rows_train": int(len(lab)), "rows_dev": int(len(dv["start"])),
           "variants": {}}
    rows = {}
    for v in variants:
        t0 = time.time()
        Xtr = np.asarray(tr[v], dtype=np.float64)
        name = f"ic_probe_{v}"
        m, s = probes.fit_probe(Xtr[lab], Y[lab], p_tr[lab])
        ics = probes.decode_ics(m.predict(np.asarray(dv[v], dtype=np.float64)), dv["start"], dv["trial_id"])
        pred = g3.Predictions(name, ics[["trial_id", "foot", "t_s"]].copy())
        off, n_off = g3.fit_offsets([pred], tg_dev)[(name, "walkway")]
        dev, per_person = {}, {}
        for sname, tg in tgs.items():
            line, tab = g3.ic_line(pred, g3.reference_ics(tg, "walkway"), off)
            per_person[sname] = line.get("per_person", {})
            dev[sname] = strip(line)
            if tg is tg_dev:
                rows[v] = [_ic_rows(tab, v)]
        S, alphas = cross_fit_scores(Xtr, Y, p_tr, lab)
        ics_cf = probes.decode_ics(S, tr["start"], tr["trial_id"])
        p_cf = g3.Predictions(name + "_crossfit", ics_cf[["trial_id", "foot", "t_s"]].copy())
        off_cf, n_cf = fit_offset_on(p_cf.ics, ref_tr)
        cf_line, _ = g3.ic_line(p_cf, ref_tr, off_cf)
        out["variants"][v] = {"alpha": s["alpha"], "offset_ms": 1000.0 * off if g3.offset_fitted(off) else None,
                              "offset_n": int(n_off), "n_ics": int(len(ics)), "dev": dev, "per_person": per_person,
                              "crossfit": {"line": strip(cf_line), "alphas": alphas, "n_ics": int(len(ics_cf)),
                                           "offset_ms": 1000.0 * off_cf if g3.offset_fitted(off_cf) else None,
                                           "offset_n": int(n_cf)}}
        log(f"IC probe {v}: cross-fitted {cf_line['point']:.1f} ms; dev14 {dev['dev14']['point']:.1f} ms, matched "
            f"share {dev['dev14']['matched_share']:.3f} ({time.time() - t0:.1f} s)", flush=True)
    return out, rows


def _same(a, b, tol=1e-9):
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run_probes(data, tg_train, tg_dev, variants=VARIANTS, ic_variants=IC_VARIANTS, log=print):
    """Every probe on features `data` (collect or load_features), the variant choice and the dev lines of the chosen
    variants; no file read or written. Returns {'sets', 'variants' {'pass', 'ic'}, 'crossfit' {line: {variant:
    point}}, 'chosen' {line: variant}, 'lines' {set: {line: stats + variant}}, 'per_person' {set: {line: {person:
    value}}}, 'rows' (DataFrame, comparison.ROW_COLUMNS plus in_dev12), 'per_pass' (DataFrame), 'rows_check',
    'seconds'}."""
    import pandas as pd
    t_start = time.time()
    sets, _ = dev_sets(tg_dev)
    passes, prow, preds = pass_stage(data, tg_train, tg_dev, variants, log)
    ic, irow = ic_stage(data, tg_train, tg_dev, ic_variants, log)
    crossfit = {ln: {v: passes[v]["crossfit"][ln]["point"] for v in variants} for ln in PASS_LINE}
    crossfit["ic_median_ms"] = {v: ic["variants"][v]["crossfit"]["line"]["point"] for v in ic_variants}
    chosen = {ln: choose(crossfit[ln]) for ln in LINES}
    lines, per_person = {}, {}
    for s in sets:
        lines[s], per_person[s] = {}, {}
        for ln in LINES:
            v = chosen[ln]
            src = ic["variants"][v] if ln == "ic_median_ms" else passes[v]
            st = src["dev"][s] if ln == "ic_median_ms" else src["dev"][s][ln]
            lines[s][ln] = {**st, "variant": v}
            per_person[s][ln] = src["per_person"][s] if ln == "ic_median_ms" else src["per_person"][s][ln]
    parts = [f for v in variants for f in prow.get(v, [])] + [f for v in ic_variants for f in irow.get(v, [])]
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CMP.ROW_COLUMNS)
    rows["chosen"] = [chosen[ln] == v for ln, v in zip(rows["line"], rows["variant"])]
    rows["in_dev12"] = rows["person_id"].isin(set(sets["dev12"]))
    worst, checked = 0.0, 0
    for s, persons in sets.items():
        for ln in LINES:
            for v in (ic_variants if ln == "ic_median_ms" else variants):
                g = rows[(rows["line"] == ln) & (rows["variant"] == v)]
                got = CMP.line_point(g, CMP.LINES[ln]["how"], persons)
                ref = (ic["variants"][v]["dev"][s] if ln == "ic_median_ms" else passes[v]["dev"][s][ln])["point"]
                if not _same(got, ref):
                    raise AssertionError(f"{s} {ln} {v}: the rows give {got}, g3 {ref}")
                if np.isfinite(got) and np.isfinite(ref):
                    worst = max(worst, abs(got - ref))
                checked += 1
    u = data["units"]["dev"]
    per_pass = u.assign(n_rows=data["dev"]["n_rows"],
                        **{f"pred_{t}_{v}": preds[v][t] for v in variants for t in TARGETS})
    return {"sets": sets, "variants": {"pass": passes, "ic": ic}, "crossfit": crossfit, "chosen": chosen,
            "lines": lines, "per_person": per_person, "rows": rows, "per_pass": per_pass,
            "rows_check": {"checked": checked, "max_abs_difference": worst},
            "seconds": round(time.time() - t_start, 1)}
