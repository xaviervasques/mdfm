"""Frozen-token linear probes on the pilot encoder (weekend task T36, run J0): does the real-sensor gait failure sit
in the representation, or in decoding and root integration? Every output carries the label LABEL.

Inputs. A checkpoint (mdfm/eval/infer.load_checkpoint, EMA weights by default) and the weargait_pd walkway passes of
the G3 targets (mdfm/gates/g3.load_targets, tasks SelfPace and HurriedPace). The 54 training-fold controls fit every
probe (a phase B encoder trained on their windows without labels, so the fit is in-sample for that encoder); the 14
dev controls score it, and the 12 of them without WHC003 and WHC026 are reported too. Records come from
g3.imu_records (canonical.load with every reader flag False), each asserted lockbox False and inside its set; the model
reads them as in G3 (g3.model_input: 13 IMUs, acc and gyro, magnetometer masked; mdfm/eval/infer.record_windows: band
copy, 10 s windows every 5 s, the last one on the record's end). A dev control among the checkpoint's training persons
stops the run.

Tokens. Per window, the tokenizer and the encoder of the checkpoint (the path of MFM.encode, bf16 autocast on CUDA, no
refinement, no decoder). At each 0.25 s latent patch p of window k (trial time t0_k + (p + 0.5) x 0.25 s):
- latent (PRIMARY, fixed before any result): the encoder output at the latent position, what the latent head reads;
- measurement: the encoder outputs at that patch's measurement tokens (every site, acc and gyro), averaged;
- stem: the tokenizer outputs at those measurement tokens (before any attention), averaged;
- mu_z: the amortized posterior mean of the latent head (32 values, what the decoder reads before refinement);
- subject: the encoder output at the window's subject token (what the shape head reads), one per window.
Each trial-clock patch is read from the window where it is most central (probes.most_central).

Pass probes (mdfm/eval/probes.py). Per walkway pass (probes.pass_units), the mean and the standard deviation of the
token rows whose patch centre lies in the pass span (subject: the mean only). Ridge probes of the pass's mean walkway
stride and of its speed, the penalty chosen by person folds inside the 54 (probes.fit_probe), fitted on the 54 and
applied to the dev passes; scored by the G3 lines (every walkway stride gets its pass's predicted stride:
g3.stride_lines, g3.speed_line, person-clustered intervals of g3). Diagnostic: person cross-fitted predictions inside
the 54 (probes.cross_fit, the penalty chosen inside each fitting part).

IC per sub-patch (probes.ic_targets, probes.decode_ics) on the latent and mu_z rows: a ridge from a patch's row to the
50 ms sub-patch of each foot's walkway IC, fitted on the labelled rows of the 54 (patches inside their pass spans) and
applied to the dev rows near a walkway pass or IC (KEEP_PAD_S); the IC line of g3 against the walkway reference
(g3.ic_line) with one offset fitted on the 14 (g3.fit_offsets), on the 14 and on the 12.

Comparison, read and never recomputed (paths and SHA-256 in results.json): the decoded body of the same checkpoint,
the base row of reports/gates/g3/diag_<name>/diag.json (subsets all14 and no_whc003_whc026: stride, speed, walkway IC
line), with the dev model line of its J1 report (reports/gates/g3/dev_<name>/g3_dev.json, or
reports/gates/g3_sim/dev_<name>/g3_sim.json for a phase A checkpoint) as a cross-check; the supervised baseline of
reports/bench/g3_supervised_baseline/results.json. The population baseline of g3 is computed here from the 54. Paired
difference probe minus decoded body: per person, mean over persons, person bootstrap interval (common.cluster_bootstrap,
g3.BOOT_N resamples, seed g3.SEED).

Leakage hook, exploratory (probes.leakage_probe), not a G10 read: the weargait subject-id family (HC and NLS against WHC
and WPD, which design section G10 (c) reads as plausibly its two sites, unconfirmed) from the pooled latent rows of the
68 passes' persons, against the same probe fed task and sex only; persons cross-fitted.

Never: a lockbox, EVAL or CODY record, a decoder pass, a write to a checkpoint. The GPU only through the queue
(scripts/queue_job.py); a CPU run sets CUDA_VISIBLE_DEVICES=-1 and --device cpu.

Writes <out> (default reports/bench/frozen_tokens_<name>): results.json, summary.md, per_pass.csv, _provenance.json; the
token features to <features> (default cache/runs/frozen_tokens/<name>/features.npz), so that --from-features refits on
the CPU without the model. --summary NAME [NAME ...] writes reports/bench/frozen_tokens_summary.md and .json from the
results.json of the named checkpoints, side by side.

Run (mdfm environment):
  python scripts/probe_frozen_tokens.py --checkpoint cache/runs/phaseA_root_full/ckpt_A_010681.pt --memory-gb 12
      --out reports/bench/frozen_tokens_phaseA_root_full_ckpt_A_010681
  python scripts/probe_frozen_tokens.py --summary phaseA_root_full_ckpt_A_010681 phaseA_root_full_ckpt_B_032044
"""
import argparse
import json
import math
import os
import re
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import numpy as np  # noqa: E402

LABEL = "frozen-token probe, dev only, not a G3 read"
TASK = "T36 (weekend run J0)"
VARIANTS = ("latent", "measurement", "stem", "mu_z", "subject")
PRIMARY = "latent"
IC_VARIANTS = ("latent", "mu_z")
TARGETS = ("stride_m", "speed_m_s")
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")
SETS = ("dev14", "dev12")
DIAG_SUBSET = {"dev14": "all14", "dev12": "no_whc003_whc026"}
KEEP_PAD_S = 1.0
BATCH_WINDOWS = 16
SUPERVISED = "reports/bench/g3_supervised_baseline/results.json"      # under TREE
SUMMARY_STEM = os.path.join(TREE, "reports", "bench", "frozen_tokens_summary")
FAMILIES = {"HC": "HC_NLS", "NLS": "HC_NLS", "WHC": "WHC_WPD", "WPD": "WHC_WPD"}
LINES = (("stride_mae_cm", "stride MAE (cm)", 2), ("speed_mae_m_s", "speed MAE (m/s)", 3),
         ("ic_median_ms", "IC median, walkway (ms)", 1))


# ---------------------------------------------------------------- small helpers
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    """JSON-ready: non-finite floats as null, numpy scalars and arrays as Python values."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def rel(path):
    from mdfm.gates import common
    return common.rel(path)


def sha256_file(path):
    from mdfm.gates import common
    return common.sha256_file(path)


def family(person_id):
    """The subject-id family of a weargait person id ('HC_NLS' or 'WHC_WPD'), None when unknown."""
    m = re.match(r"([A-Za-z]+)", str(person_id).split(":", 1)[-1])
    return FAMILIES.get(m.group(1).upper()) if m else None


def check_record(tid, rec, persons):
    """AssertionError unless the record carries lockbox False, no EVAL tier, and a person of `persons`."""
    lb = rec.get("lockbox")
    if lb is None or (lb is not False and str(lb).strip().lower() not in ("false", "0")):
        raise AssertionError(f"{tid}: record with lockbox {lb!r}")
    if str(rec.get("tier", "")) in ("EVAL", "EXCLUDED_V1"):
        raise AssertionError(f"{tid}: record of tier {rec.get('tier')}")
    if rec.get("person_id") not in set(persons):
        raise AssertionError(f"{tid}: person {rec.get('person_id')!r} outside the requested set")


# ---------------------------------------------------------------- tokens
def _amp(device, amp=None):
    import torch
    dev = torch.device(device)
    on = dev.type == "cuda" if amp is None else bool(amp)
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=on)


def encode_windows(model, windows, batch_windows=BATCH_WINDOWS, amp=None):
    """[{variant: array}] per window (module docstring, Tokens): latent, measurement, stem (P, d) and mu_z (P, d_z)
    float32 per latent patch, n_measurement (P,) the measurement tokens averaged at each patch (NaN rows where 0), and
    subject (d,). The tokenizer and the encoder run as MFM.encode runs them (one window per row), under no_grad."""
    import torch

    from mdfm.model import tokenize as TK
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
            mu, _ = model.latent_head(hl)
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
            subj = hf[tb.subj_idx]
            res = {"latent": hl.cpu().numpy(), "measurement": hm.cpu().numpy(), "stem": xm.cpu().numpy(),
                   "mu_z": mu.float().cpu().numpy(), "n_measurement": cnt.reshape(len(ch), Pmax).cpu().numpy(),
                   "subject": subj.cpu().numpy()}
        npat = tb.n_patches.cpu().numpy()
        for j in range(len(ch)):
            P = int(npat[j])
            out.append({k: (v[j] if k == "subject" else v[j, :P]).astype(np.float32) for k, v in res.items()})
    return out


def trial_rows(feats, windows, patch_s=0.25):
    """Token rows of one trial, the most central reading of each patch only: start and centre (s, trial clock),
    window index, and every variant's features (subject repeated over its window's rows)."""
    from mdfm.eval import probes
    starts = np.array([float(w.t0) for w in windows])
    durs = np.array([float(w.duration) for w in windows])
    k = np.concatenate([np.full(f["latent"].shape[0], i) for i, f in enumerate(feats)]).astype(np.int64)
    p = np.concatenate([np.arange(f["latent"].shape[0]) for f in feats]).astype(np.int64)
    a = starts[k] + p * patch_s
    c = a + patch_s / 2.0
    central = probes.most_central(starts, durs, k, c)
    cat = lambda v: np.concatenate([f[v] for f in feats])  # noqa: E731
    rows = {"start": a, "centre": c, "window": k}
    for v in ("latent", "measurement", "stem", "mu_z"):
        rows[v] = cat(v)
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


def unit_features(rows, spans):
    """({variant: (n_spans, D)} pooled features, (n_spans,) rows per span) of one trial's rows (probes.pool_spans:
    mean and std, subject the mean only)."""
    from mdfm.eval import probes
    out, n = {}, None
    for v in VARIANTS:
        how = ("mean",) if v == "subject" else probes.POOL
        out[v], nv = probes.pool_spans(rows["centre"], rows[v], spans, how)
        if v == PRIMARY:
            n = nv
    return out, n


def collect(encode, records, units, targets, persons, log=print, wait=None):
    """Features of one person set. encode: windows -> [per-window dict] (encode_windows bound to a model); records:
    iterable of (trial_id, model input record); units: probes.pass_units of the set's targets. Returns {'X':
    {variant: (n_units, D)}, 'n_rows': (n_units,), 'sex': (n_units,), 'tokens': {trial_id, person_id, start, latent,
    mu_z}, 'counts'}."""
    from mdfm.eval import infer as I
    by_trial = {t: g.index.to_numpy() for t, g in units.groupby("trial_id")}
    ics = {t: g["t_s"].to_numpy() for t, g in targets.walkway_ic.groupby("trial_id")}
    X = {v: [None] * len(units) for v in VARIANTS}
    n_rows = np.zeros(len(units), np.int64)
    sex = np.full(len(units), -1, np.int64)
    tok = {"trial_id": [], "person_id": [], "start": [], "latent": [], "mu_z": []}
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
            uf, n = unit_features(rows, spans)
            for v in VARIANTS:
                for j, i in enumerate(idx):
                    X[v][i] = uf[v][j]
            n_rows[idx] = n
            sex[idx] = int(wins[0].cov.get("sex", -1))
        keep = keep_mask(rows["centre"], spans, ics.get(tid, ()))
        m = int(keep.sum())
        tok["trial_id"] += [tid] * m
        tok["person_id"] += [str(rec.get("person_id"))] * m
        tok["start"].append(rows["start"][keep])
        tok["latent"].append(rows["latent"][keep].astype(np.float16))
        tok["mu_z"].append(rows["mu_z"][keep].astype(np.float32))
        log(f"  {tid}: {len(wins)} windows, {len(idx)} passes, {m} token rows kept ({time.time() - t0:.0f} s)",
            flush=True)
        if wait is not None:
            wait()
    dims = {v: next((x.shape[0] for x in X[v] if x is not None), 0) for v in VARIANTS}
    Xa = {v: np.stack([x if x is not None else np.full(dims[v], np.nan) for x in X[v]]) if len(units)
          else np.zeros((0, dims[v])) for v in VARIANTS}
    tokens = {"trial_id": np.asarray(tok["trial_id"], dtype=str), "person_id": np.asarray(tok["person_id"], dtype=str),
              "start": np.concatenate(tok["start"]) if tok["start"] else np.zeros(0),
              "latent": np.concatenate(tok["latent"]) if tok["latent"] else np.zeros((0, 0), np.float16),
              "mu_z": np.concatenate(tok["mu_z"]) if tok["mu_z"] else np.zeros((0, 0), np.float32)}
    counts = {"records": n_rec, "windows": n_win, "passes": int(len(units)),
              "passes_with_rows": int((n_rows > 0).sum()), "passes_without_rows": int((n_rows == 0).sum()),
              "trials_without_record": sorted(set(units["trial_id"]) - seen), "token_rows_kept": int(len(tokens["start"])),
              "rows_per_pass_median": float(np.median(n_rows[n_rows > 0])) if (n_rows > 0).any() else None}
    return {"X": Xa, "n_rows": n_rows, "sex": sex, "tokens": tokens, "counts": counts}


# ---------------------------------------------------------------- feature file
def save_features(path, data, meta):
    """data {'units': {set: DataFrame}, set: collect() output} and meta (dict) into one npz (no pickle)."""
    arrs = {"meta": np.asarray(json.dumps(clean(meta)))}
    for s in ("train", "dev"):
        arrs[f"{s}__units"] = np.asarray(data["units"][s].to_json(orient="split"))
        d = data[s]
        for v in VARIANTS:
            arrs[f"{s}__X__{v}"] = d["X"][v].astype(np.float32)
        arrs[f"{s}__n_rows"] = d["n_rows"]
        arrs[f"{s}__sex"] = d["sex"]
        arrs[f"{s}__counts"] = np.asarray(json.dumps(clean(d["counts"])))
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
                   "sex": z[f"{s}__sex"], "counts": json.loads(str(z[f"{s}__counts"])),
                   "tokens": {k: z[f"{s}__tok__{k}"] for k in ("trial_id", "person_id", "start", "latent", "mu_z")}}
    return data, meta


def check_features_meta(meta, name, path=""):
    """SystemExit unless the feature file's meta names checkpoint `name` (a feature file given by --features for
    another checkpoint would otherwise land in this checkpoint's folder, beside its decoded body)."""
    got = (meta.get("checkpoint") or {}).get("name")
    if got != name:
        raise SystemExit(f"{path or 'the feature file'} holds the features of {got!r}, not of {name!r}")


# ---------------------------------------------------------------- probes
def strip_lines(lines):
    """Line stats without the per-person tables of g3 (kept separately)."""
    return {ln: {k: v for k, v in st.items() if k != "per_person"} for ln, st in lines.items()}


def pass_probes(data, tg_train, tgs, variants=VARIANTS, log=print):
    """{variant: {'selection', 'lines' {set: lines}, 'per_person' {set: {line: {person: value}}}, 'n_features'}},
    {variant: dev pass predictions}: the ridge pass probes of the module docstring."""
    from mdfm.eval import probes
    u_tr, u_dev = data["units"]["train"], data["units"]["dev"]
    p_tr = u_tr["person_id"].to_numpy().astype(str)
    probes.check_disjoint(p_tr, u_dev["person_id"])
    out, preds = {}, {}
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
        lines, per_person = {}, {}
        for sname, tg in list(tgs.items()) + [("train_crossfit", tg_train)]:
            units = u_tr if sname == "train_crossfit" else u_dev
            pp = oof if sname == "train_crossfit" else pred
            sc = probes.score(tg, [probes.pass_predictions(units, pp, tg, v)])[v]
            lines[sname] = strip_lines(sc)
            per_person[sname] = {ln: st.get("per_person", {}) for ln, st in sc.items()}
        out[v] = {"selection": sel, "lines": lines, "per_person": per_person, "n_features": int(Xtr.shape[1])}
        preds[v] = pred
        st = lines["dev14"]
        log(f"probe {v}: dev14 stride {fmt(st['stride_mae_cm']['point'])} cm, speed "
            f"{fmt(st['speed_mae_m_s']['point'], 3)} m/s ({time.time() - t0:.1f} s)", flush=True)
    return out, preds


def ic_probes(data, tg_train, tg_dev, tgs, variants=IC_VARIANTS, log=print):
    """{variant: {'alpha', 'offset_ms', 'offset_n', 'n_ics', 'lines' {set: ic line}, 'per_person'}} of the IC probe
    per sub-patch (module docstring), plus the labelled-row counts."""
    from mdfm.eval import probes
    from mdfm.gates import g3
    u_tr = data["units"]["train"]
    tr, dv = data["train"]["tokens"], data["dev"]["tokens"]
    probes.check_disjoint(tr["person_id"], dv["person_id"])
    Y, lab = probes.ic_targets(tr["start"], tr["trial_id"], tg_train.walkway_ic, u_tr[["trial_id", "t0", "t1"]])
    out = {"labelled_rows": int(lab.sum()), "rows_train": int(len(lab)), "rows_dev": int(len(dv["start"])),
           "ic_share_of_labelled": {f: float(Y[lab][:, i * probes.SUB_BINS:(i + 1) * probes.SUB_BINS].sum(1).mean())
                                    if lab.any() else None for i, f in enumerate(probes.FEET)},
           "variants": {}}
    for v in variants:
        t0 = time.time()
        Xtr = np.asarray(tr[v], dtype=np.float64)
        m, s = probes.fit_probe(Xtr[lab], Y[lab], tr["person_id"][lab])
        S = m.predict(np.asarray(dv[v], dtype=np.float64))
        ics = probes.decode_ics(S, dv["start"], dv["trial_id"])
        name = f"ic_probe_{v}"
        pred = g3.Predictions(name, ics[["trial_id", "foot", "t_s"]].copy())
        off, n_off = g3.fit_offsets([pred], tg_dev)[(name, "walkway")]
        lines, per_person = {}, {}
        for sname, tg in tgs.items():
            line, _ = g3.ic_line(pred, g3.reference_ics(tg, "walkway"), off)
            per_person[sname] = line.pop("per_person", {})
            lines[sname] = line
        out["variants"][v] = {"alpha": s["alpha"], "offset_ms": 1000.0 * off if g3.offset_fitted(off) else None,
                              "offset_n": int(n_off), "n_ics": int(len(ics)), "lines": lines, "per_person": per_person}
        log(f"IC probe {v}: dev14 {fmt(lines['dev14']['point'], 1)} ms, matched share "
            f"{fmt(lines['dev14']['matched_share'], 3)} ({time.time() - t0:.1f} s)", flush=True)
    return out


def leakage(data, variant=PRIMARY):
    """The exploratory leakage hook run (module docstring) on the pooled pass rows of both sets, or None when the
    rows do not hold two families."""
    from mdfm.eval import probes
    X, lab, per, cov = [], [], [], []
    for s in ("train", "dev"):
        u = data["units"][s]
        fam = np.array([family(p) for p in u["person_id"]], dtype=object)
        x = data[s]["X"][variant]
        ok = np.isfinite(x).all(axis=1) & np.array([f is not None for f in fam])
        X.append(x[ok])
        lab += list(fam[ok])
        per += list(u["person_id"].to_numpy()[ok])
        sx = np.asarray(data[s]["sex"])[ok]
        hurried = (u["task"].to_numpy()[ok] == "HurriedPace").astype(float)
        cov.append(np.stack([hurried, (sx == 0).astype(float), (sx == 1).astype(float)], axis=1))
    if len(set(lab)) < 2 or len(set(per)) < probes.FOLDS:
        return None
    r = probes.leakage_probe(np.concatenate(X), np.asarray(lab), np.asarray(per), np.concatenate(cov))
    r.update(variant=variant, label="subject-id family (HC and NLS against WHC and WPD)",
             covariates=["task HurriedPace", "sex code 0", "sex code 1"],
             persons_per_class={c: int(len(set(np.asarray(per)[np.asarray(lab) == c]))) for c in r["classes"]})
    return r


def population_lines(tg_train, tgs):
    """{set: lines} of the population baseline of g3 (training fold) on the dev sets."""
    from mdfm.eval import probes
    from mdfm.gates import g3
    base = g3.population_baseline(tg_train)
    return {s: strip_lines(probes.score(tg, [g3.population_predictions(base, tg)])["population"])
            for s, tg in tgs.items()}


def dev_sets(tg_dev, excluded=EXCLUDED_FROM_12):
    """({'dev14': persons, 'dev12': persons}, {set: Targets})."""
    from mdfm.eval import probes
    d14 = sorted(tg_dev.persons)
    d12 = sorted(set(d14) - set(excluded))
    return {"dev14": d14, "dev12": d12}, {"dev14": tg_dev, "dev12": probes.subset_targets(tg_dev, d12)}


def run_probes(data, tg_train, tg_dev, excluded=EXCLUDED_FROM_12, variants=VARIANTS, ic_variants=IC_VARIANTS,
               log=print):
    """Every probe on features `data` (collect or load_features); no file read or written."""
    sets, tgs = dev_sets(tg_dev, excluded)
    t0 = time.time()
    passes, preds = pass_probes(data, tg_train, tgs, variants, log)
    ic = ic_probes(data, tg_train, tg_dev, tgs, ic_variants, log)
    leak = leakage(data)
    u = data["units"]["dev"]
    per_pass = u.assign(n_rows=data["dev"]["n_rows"], **{f"pred_{t}": preds[PRIMARY][t] for t in TARGETS}) \
        if PRIMARY in preds else u
    return {"sets": sets, "excluded_from_12": list(excluded), "primary": PRIMARY, "passes": passes, "ic": ic,
            "population": population_lines(tg_train, tgs), "leakage": leak, "per_pass": per_pass,
            "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- comparison (read, not recomputed)
def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def decoded_body(name, tree=None):
    """The decoded-body lines of checkpoint `name`: the base row of its diag.json (stride and speed with intervals
    and per person, the walkway IC line) on the 14 and the 12, and the dev model line of its J1 report; None parts
    when a file is missing. Paths and SHA-256 of the files read included. tree: default TREE."""
    tree = tree or TREE
    out = {"diag": None, "j1": None, "dev14": {}, "dev12": {}}
    p = os.path.join(tree, "reports", "gates", "g3", f"diag_{name}", "diag.json")
    if os.path.exists(p):
        d = _read_json(p)
        base = d["rows"]["base"]
        out["diag"] = {"path": rel(p), "sha256": sha256_file(p), "checkpoint_sha256": d["checkpoint"].get("sha256"),
                       "weights": d["checkpoint"].get("weights"), "inference": {k: d["inference"].get(k) for k in
                                                                              ("refine_steps", "points_per_stream",
                                                                               "window_s", "hop_s", "seed")}}
        for s, sub in DIAG_SUBSET.items():
            b = base[sub]
            out[s] = {"stride_mae_cm": {"point": b.get("stride_mae_cm"), "interval": b.get("stride_mae_cm_interval")},
                      "speed_mae_m_s": {"point": b.get("speed_mae_ms"), "interval": b.get("speed_mae_ms_interval")},
                      "stride_mape_pct": {"point": b.get("stride_mape_pct")}, "n_persons": b.get("n_persons"),
                      "per_person": {"stride_mae_cm": {k: v.get("stride_cm") for k, v in b.get("per_person", {}).items()},
                                     "speed_mae_m_s": {k: v.get("speed_ms") for k, v in b.get("per_person", {}).items()}}}
        ic = base.get("ic_walkway") or {}
        out["dev14"]["ic_median_ms"] = {"point": ic.get("median_ms"), "interval": ic.get("interval"),
                                        "matched_share": ic.get("matched_share"), "offset_ms": ic.get("offset_ms")}
        out["dev12"]["ic_median_ms"] = {"point": (ic.get("subsets") or {}).get("no_whc003_whc026"), "interval": None}
        sg = (d.get("shank_gyro_detector") or {}).get("walkway") or {}
        out["shank_gyro_walkway"] = {"median_ms": sg.get("median_ms"), "matched_share": sg.get("matched_share")}
    for p, stem in ((os.path.join(tree, "reports", "gates", "g3", f"dev_{name}", "g3_dev.json"), "g3_dev"),
                    (os.path.join(tree, "reports", "gates", "g3_sim", f"dev_{name}", "g3_sim.json"), "g3_sim")):
        if os.path.exists(p):
            g = _read_json(p)
            ml = g["lines"]["dev"]["model"]
            out["j1"] = {"path": rel(p), "sha256": sha256_file(p), "checkpoint_sha256": g["checkpoint"].get("sha256"),
                         "dev14": {ln: (ml.get(key) or {}).get("point") for ln, key in
                                   (("stride_mae_cm", "stride_mae_cm"), ("speed_mae_m_s", "speed_mae_m_s"),
                                    ("ic_median_ms", "ic_median_ms_walkway"))}}
            pp = os.path.join(os.path.dirname(p), "per_person.csv")
            if os.path.exists(pp):
                import pandas as pd
                t = pd.read_csv(pp, encoding="utf-8")
                t = t[(t["row"] == "dev") & (t["predictor"] == "model") & (t["line"] == "ic_median_ms_walkway")]
                out["j1"]["per_person_csv"] = {"path": rel(pp), "sha256": sha256_file(pp)}
                out["j1"]["ic_per_person"] = {str(r.person_id): (float(r.value) if fin(r.value) is not None
                                                                 else float("inf")) for r in t.itertuples()}
            break
    return out


def supervised(path=None):
    """The supervised baseline's dev lines (stride, speed) on the 14 and the 12, with the path and SHA-256 read
    (default SUPERVISED under TREE); None when the file is missing."""
    path = path or os.path.join(TREE, SUPERVISED)
    if not os.path.exists(path):
        return None
    r = _read_json(path)
    out = {"path": rel(path), "sha256": sha256_file(path), "run_started": r.get("run_started"), "what": r.get("what")}
    for s in SETS:
        sup = r["lines"][s]["supervised"]
        out[s] = {ln: {"point": sup[ln]["point"], "interval": sup[ln].get("interval")}
                  for ln in ("stride_mae_cm", "speed_mae_m_s") if ln in sup}
    return out


def paired(probe_pp, decoded_pp, n_boot=None, seed=None):
    """Probe minus decoded body per person over the persons both hold (finite values): {'mean', 'interval',
    'n_persons'} with a person bootstrap interval (common.cluster_bootstrap)."""
    from mdfm.gates import common, g3
    ids = sorted(k for k in set(probe_pp) & set(decoded_pp)
                 if fin(probe_pp[k]) is not None and fin(decoded_pp[k]) is not None)
    if len(ids) < 2:
        return {"mean": None, "interval": None, "n_persons": len(ids)}
    diff = np.array([float(probe_pp[k]) - float(decoded_pp[k]) for k in ids])
    iv = common.cluster_bootstrap(diff, ids, n=n_boot or g3.BOOT_N, seed=g3.SEED if seed is None else seed)
    return {"mean": float(diff.mean()), "interval": list(iv), "n_persons": len(ids)}


def reading(pair):
    """'probe lower', 'decoded lower' or 'no difference at 95 %' from a paired interval."""
    iv = pair.get("interval") if pair else None
    if not iv or fin(iv[0]) is None:
        return "n/a"
    return "probe lower" if iv[1] < 0 else "decoded lower" if iv[0] > 0 else "no difference at 95 %"


def comparison(res, name):
    """The comparison block of results.json: decoded body, supervised baseline, paired differences of the primary
    probe: stride and speed against the diag's base row; the IC line against the per-person values of the J1 report
    (the diag keeps no per-person IC table for its base row), persons with a finite value on both sides."""
    dec = decoded_body(name)
    sup = supervised()
    pairs = {}
    for s in SETS:
        pairs[s] = {}
        for ln in ("stride_mae_cm", "speed_mae_m_s"):
            pp = res["passes"][PRIMARY]["per_person"][s].get(ln, {})
            dp = (dec.get(s) or {}).get("per_person", {}).get(ln, {})
            pr = paired(pp, dp)
            pr.update(reading=reading(pr), decoded_source="diag base row")
            pairs[s][ln] = pr
        ic_v = (res.get("ic") or {}).get("variants", {}).get(PRIMARY)
        j1pp = (dec.get("j1") or {}).get("ic_per_person")
        if ic_v and j1pp is not None:
            keep = set(res["sets"][s])
            pr = paired({k: v for k, v in ic_v["per_person"][s].items() if k in keep},
                        {k: v for k, v in j1pp.items() if k in keep})
            pr.update(reading=reading(pr), decoded_source="J1 report per_person.csv")
            pairs[s]["ic_median_ms"] = pr
    return {"decoded": dec, "supervised": sup, "paired_primary": pairs}


# ---------------------------------------------------------------- report
def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def main_rows(res):
    """[(line key, line title, digits, set, probe stats, decoded stats, supervised stats, population stats, pair)] of
    the primary table."""
    out = []
    cmp_ = res["comparison"]
    for ln, title, nd in LINES:
        for s in SETS:
            if ln == "ic_median_ms":
                pr = _get(res, "ic", "variants", PRIMARY, "lines", s) or {}
                pop = None
            else:
                pr = _get(res, "passes", PRIMARY, "lines", s, ln) or {}
                pop = _get(res, "population", s, ln) or {}
            out.append((ln, title, nd, s, pr, _get(cmp_, "decoded", s, ln) or {},
                        _get(cmp_, "supervised", s, ln) or {}, pop, _get(cmp_, "paired_primary", s, ln) or {}))
    return out


def report_md(res):
    """summary.md of one checkpoint (every number read from the results dict as written to results.json)."""
    ck = res["checkpoint"]
    cmp_ = res["comparison"]
    dec, sup = cmp_["decoded"], cmp_["supervised"]
    L = [f"# Frozen-token probes, {ck['name']} ({LABEL})", "",
         "Written by scripts/probe_frozen_tokens.py (task T36, run J0; probes mdfm/eval/probes.py); every number below is "
         "read from results.json of this folder, the decoded-body and supervised columns from the files it cites. Dev "
         "only, not a G3 read: no lockbox, EVAL or CODY row or record was read.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,}. Tokens: tokenizer and encoder only (no refinement, no decoder), "
         f"{res['inference']['window_s']:g} s windows every {res['inference']['hop_s']:g} s, each 0.25 s patch read from "
         f"its most central window; device {res['inference']['device']}.",
         f"Fit: the {len(res['sets']['train'])} training-fold controls ({res['sets']['train_among_checkpoint_persons']} of "
         "them among the checkpoint's training persons, whose windows it saw without labels); score: the "
         f"{len(res['sets']['dev14'])} dev controls and the {len(res['sets']['dev12'])} without "
         f"{', '.join(res['excluded_from_12'])}. Primary probe: {PRIMARY} (fixed before any result); ridge penalty by "
         "person folds inside the fitting set.", "",
         "## Probe beside the decoded body (primary probe)", "",
         "| line | set | probe | 95 % interval | decoded body | 95 % interval | supervised baseline | population | "
         "probe minus decoded, paired | 95 % interval | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ln, title, nd, s, pr, dc, sp, pop, pair in main_rows(res):
        L.append(f"| {title} | {s} ({len(res['sets'][s])}) | {fmt(pr.get('point'), nd)} | {fmt_iv(pr.get('interval'), nd)} | "
                 f"{fmt(dc.get('point'), nd)} | {fmt_iv(dc.get('interval'), nd)} | {fmt(sp.get('point'), nd)} | "
                 f"{fmt((pop or {}).get('point'), nd)} | {fmt(pair.get('mean'), nd)} | {fmt_iv(pair.get('interval'), nd)} | "
                 f"{pair.get('reading', 'n/a')} |")
    d = dec.get("diag") or {}
    j1 = dec.get("j1") or {}
    same = "n/a" if not d else ("the same as this run's" if d.get("checkpoint_sha256") == ck["sha256"]
                                else f"DIFFERENT from this run's ({d.get('checkpoint_sha256')})")
    L += ["", f"Decoded body: base row of {d.get('path', 'n/a (diag.json not found)')} (SHA-256 {d.get('sha256', 'n/a')}; "
          f"subsets all14 and no_whc003_whc026; its checkpoint SHA-256 {same}), stride and speed of the body decoded "
          f"after {_get(d, 'inference', 'refine_steps') if d else 'n/a'} refinement steps as scripts/eval_g3_dev.py runs "
          "it, IC its walkway line.",
          f"J1 report: {j1.get('path', 'n/a')} (SHA-256 {j1.get('sha256', 'n/a')}): dev model stride "
          f"{fmt(_get(j1, 'dev14', 'stride_mae_cm'))} cm, speed {fmt(_get(j1, 'dev14', 'speed_mae_m_s'), 3)} m/s, walkway IC "
          f"{fmt(_get(j1, 'dev14', 'ic_median_ms'), 1)} ms on the 14 (a separate run of the same harness).",
          f"Supervised baseline: {(sup or {}).get('path', 'n/a')} (SHA-256 {(sup or {}).get('sha256', 'n/a')}, run "
          f"{(sup or {}).get('run_started', 'n/a')}). Population: the population baseline of g3 from the training fold.",
          "Paired: per person, probe minus decoded body (stride and speed: the diag's per-person values; IC: the J1 "
          f"report's per_person.csv), mean over persons, person bootstrap ({res['constants']['boot_n']} resamples, seed "
          f"{res['constants']['seed']}); 'probe lower' when the interval lies below 0.", "",
          "## Every probe variant", "",
          "| variant | features | stride dev14 (cm) | stride dev12 | speed dev14 (m/s) | speed dev12 | stride, 54 cross-fitted | "
          "speed, 54 cross-fitted | alpha stride | alpha speed |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in res["passes"].items():
        g = lambda s, ln: _get(r, "lines", s, ln, "point")  # noqa: E731
        L.append(f"| {v}{' (primary)' if v == PRIMARY else ''} | {r['n_features']} | {fmt(g('dev14', 'stride_mae_cm'))} | "
                 f"{fmt(g('dev12', 'stride_mae_cm'))} | {fmt(g('dev14', 'speed_mae_m_s'), 3)} | "
                 f"{fmt(g('dev12', 'speed_mae_m_s'), 3)} | {fmt(g('train_crossfit', 'stride_mae_cm'))} | "
                 f"{fmt(g('train_crossfit', 'speed_mae_m_s'), 3)} | {fmt(r['selection']['stride_m']['alpha'], 2)} | "
                 f"{fmt(r['selection']['speed_m_s']['alpha'], 2)} |")
    L += ["", "Variants: latent, the encoder output at the latent positions; measurement, the encoder outputs at the "
          "measurement tokens of each patch, averaged; stem, the tokenizer outputs there (no attention); mu_z, the "
          "amortized posterior mean; subject, the encoder output at the window's subject token. Per pass: mean and "
          "standard deviation over the pass's patches (subject: mean).", "",
          "## IC per sub-patch (50 ms), walkway reference", "",
          "| variant | dev14 (ms) | 95 % interval | matched share | dev12 (ms) | matched share | offset (ms) | ICs read | alpha |",
          "|---|---|---|---|---|---|---|---|---|"]
    ic = res.get("ic") or {}
    for v, r in (ic.get("variants") or {}).items():
        a, b = r["lines"].get("dev14", {}), r["lines"].get("dev12", {})
        L.append(f"| {v} | {fmt(a.get('point'), 1)} | {fmt_iv(a.get('interval'), 1)} | {fmt(a.get('matched_share'), 3)} | "
                 f"{fmt(b.get('point'), 1)} | {fmt(b.get('matched_share'), 3)} | {fmt(r.get('offset_ms'), 1)} | "
                 f"{r.get('n_ics')} | {fmt(r.get('alpha'), 2)} |")
    sg = dec.get("shank_gyro_walkway") or {}
    L += ["", f"Labelled training rows {ic.get('labelled_rows')} of {ic.get('rows_train')}; dev rows {ic.get('rows_dev')}. "
          f"Decoded body (diag base row): walkway IC {fmt(_get(dec, 'dev14', 'ic_median_ms', 'point'), 1)} ms on the 14, "
          f"{fmt(_get(dec, 'dev12', 'ic_median_ms', 'point'), 1)} on the 12; shank-gyroscope detector "
          f"{fmt(sg.get('median_ms'), 1)} ms (matched share {fmt(sg.get('matched_share'), 3)}). A value reads n/a when a "
          "person has more than half of its reference ICs unmatched (infinite, as in g3).", ""]
    lk = res.get("leakage")
    L += ["## Leakage hook (exploratory, not a G10 read)", ""]
    if lk:
        L += [f"{lk['label']}, from the pooled {lk['variant']} rows of {lk['n_rows']} passes of {lk['n_persons']} persons "
              f"({', '.join(f'{c} {n}' for c, n in lk['persons_per_class'].items())} persons), persons cross-fitted: "
              f"balanced accuracy {fmt(lk['balanced_accuracy'], 3)} against {fmt(lk['covariate_balanced_accuracy'], 3)} for "
              f"the probe fed {', '.join(lk['covariates'])} only; leakage {fmt(lk['leakage_points'], 1)} points (chance "
              f"{fmt(lk['chance'], 3)}).", ""]
    else:
        L += ["Not run: the rows hold fewer than two families or too few persons.", ""]
    L += ["## Counts", "", "| set | records | windows | passes | passes with rows | rows per pass (median) | token rows kept |",
          "|---|---|---|---|---|---|---|"]
    for s in ("train", "dev"):
        c = res["counts"][s]
        L.append(f"| {s} | {c['records']} | {c['windows']} | {c['passes']} | {c['passes_with_rows']} | "
                 f"{fmt(c.get('rows_per_pass_median'), 1)} | {c['token_rows_kept']} |")
    L += ["", f"Lockbox: persons of either set in the weargait_pd lockbox list {res['lockbox']['persons_in_lockbox']}; "
          f"target rows kept {res['lockbox']['target_lockbox_rows']} (asserted 0; records read with every reader flag False "
          "and checked one by one).",
          f"MoveAtlas: HEAD {res.get('moveatlas_head') or 'n/a'}. Time: tokens {fmt(res['timing_s'].get('tokens'), 1)} s, "
          f"probes {fmt(res['timing_s'].get('probes'), 1)} s, all {fmt(res['timing_s'].get('all'), 1)} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- summary over checkpoints
def results_path(name, tree=TREE):
    return os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}", "results.json")


def summary_md(results, sources):
    """frozen_tokens_summary.md: the checkpoints of `results` {name: results dict} side by side (every number read
    from their results.json files, listed in `sources` {name: (path, sha256)})."""
    names = list(results)
    L = [f"# Frozen-token probes, checkpoints side by side ({LABEL})", "",
         "Written by scripts/probe_frozen_tokens.py --summary; every number below is read from the results.json of each "
         "checkpoint's folder (listed at the end), which cites its decoded-body and supervised sources. Dev only, not a "
         f"G3 read. Primary probe: {PRIMARY} (encoder output at the latent positions, mean and standard deviation over a "
         "pass's patches; ridge fitted on the 54 training-fold controls).", "",
         "## Probe beside the decoded body", "",
         "| line | set | " + " | ".join(f"{n} probe | {n} decoded" for n in names) + " | supervised baseline |",
         "|---|---|" + "---|---|" * len(names) + "---|"]
    first = results[names[0]]
    for i, (ln, title, nd, s, *_rest) in enumerate(main_rows(first)):
        cells = []
        for n in names:
            r = main_rows(results[n])[i]
            cells += [f"{fmt(r[4].get('point'), nd)} {fmt_iv(r[4].get('interval'), nd)}".strip(),
                      f"{fmt(r[5].get('point'), nd)} {fmt_iv(r[5].get('interval'), nd)}".strip()]
        sp = main_rows(first)[i][6]
        L.append(f"| {title} | {s} | " + " | ".join(cells) + f" | {fmt(sp.get('point'), nd)} {fmt_iv(sp.get('interval'), nd)}".rstrip() + " |")
    L += ["", "Cells: point and 95 % person-bootstrap interval. Decoded: the base row of reports/gates/g3/diag_<name>/"
          "diag.json (IC on the 12 without an interval there). Supervised: reports/bench/g3_supervised_baseline/"
          "results.json (no IC line).", "",
          "## Paired difference, probe minus decoded body (mean over persons, 95 % interval, reading)", "",
          "| line | set | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for i, (ln, title, nd, s, *_rest) in enumerate(main_rows(first)):
        cells = []
        for n in names:
            pr = main_rows(results[n])[i][8]
            cells.append(f"{fmt(pr.get('mean'), nd)} {fmt_iv(pr.get('interval'), nd)} {pr.get('reading', 'n/a')} "
                         f"(n {pr.get('n_persons', 'n/a')})")
        L.append(f"| {title} | {s} | " + " | ".join(cells) + " |")
    for ln, title, nd in LINES[:2]:
        L += ["", f"## Every variant: {title}, dev14 / dev12 / 54 cross-fitted", "",
              "| variant | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for v in VARIANTS:
            cells = []
            for n in names:
                r = _get(results[n], "passes", v, "lines") or {}
                cells.append(" / ".join(fmt(_get(r, s, ln, "point"), nd) for s in ("dev14", "dev12", "train_crossfit")))
            L.append(f"| {v}{' (primary)' if v == PRIMARY else ''} | " + " | ".join(cells) + " |")
    L += ["", "## IC per sub-patch, walkway (ms), dev14 / dev12, matched share on the 14", "",
          "| variant | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for v in IC_VARIANTS:
        cells = []
        for n in names:
            r = _get(results[n], "ic", "variants", v) or {}
            cells.append(f"{fmt(_get(r, 'lines', 'dev14', 'point'), 1)} / {fmt(_get(r, 'lines', 'dev12', 'point'), 1)}, "
                         f"{fmt(_get(r, 'lines', 'dev14', 'matched_share'), 3)}")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L += ["", "## Leakage hook (exploratory, not a G10 read): subject-id family, balanced accuracy of the token probe / of "
          "the task and sex probe, leakage points", "", "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    cells = []
    for n in names:
        lk = results[n].get("leakage") or {}
        cells.append(f"{fmt(lk.get('balanced_accuracy'), 3)} / {fmt(lk.get('covariate_balanced_accuracy'), 3)}, "
                     f"{fmt(lk.get('leakage_points'), 1)}")
    L += ["| " + " | ".join(cells) + " |", "", "## Sources", ""]
    for n in names:
        ck = results[n]["checkpoint"]
        L.append(f"- {n}: {sources[n][0]} (SHA-256 {sources[n][1]}); checkpoint {ck['path']} (SHA-256 {ck['sha256']}), "
                 f"{ck['weights']} weights, phase {ck['phase']} step {ck['step']:,}; training-fold controls among its "
                 f"training persons {results[n]['sets']['train_among_checkpoint_persons']}.")
    return "\n".join(L) + "\n"


def write_summary(names, out_stem=None, tree=TREE, log=print):
    """Read the results.json of each name and write <out_stem>.md and .json (default reports/bench/
    frozen_tokens_summary); returns the two paths."""
    from mdfm.gates import common
    out_stem = out_stem or SUMMARY_STEM
    results, sources = {}, {}
    for n in names:
        p = results_path(n, tree)
        if not os.path.exists(p):
            raise SystemExit(f"no results for {n}: {p}")
        results[n] = _read_json(p)
        sources[n] = (rel(p), sha256_file(p))
    text = summary_md(results, sources)
    js = {"what": LABEL, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "names": list(names),
          "sources": {n: {"path": s[0], "sha256": s[1]} for n, s in sources.items()},
          "main": {n: [{"line": r[0], "set": r[3], "probe": r[4].get("point"), "probe_interval": r[4].get("interval"),
                        "decoded": r[5].get("point"), "decoded_interval": r[5].get("interval"),
                        "supervised": r[6].get("point"), "paired": r[8]} for r in main_rows(results[n])]
                   for n in names}}
    md = common.write_text(out_stem + ".md", text)
    jp = common.write_text(out_stem + ".json", json.dumps(clean(js), indent=1, ensure_ascii=False) + "\n")
    log(text)
    return md, jp


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=12.0, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--batch-windows", type=int, default=BATCH_WINDOWS)
    p.add_argument("--out", default=None, help="default reports/bench/frozen_tokens_<name>")
    p.add_argument("--features", default=None, help="default cache/runs/frozen_tokens/<name>/features.npz")
    p.add_argument("--from-features", action="store_true", help="refit from the feature file, no model")
    p.add_argument("--limit-persons", type=int, default=0, help="smoke: the first N persons of each set (needs --out)")
    p.add_argument("--summary", nargs="+", metavar="NAME", help="write the side-by-side summary of these checkpoints")
    p.add_argument("--summary-out", default=None, help="file stem of the summary (default reports/bench/"
                                                       "frozen_tokens_summary)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if a.summary:
        write_summary(a.summary, a.summary_out)
        return 0
    if not a.checkpoint:
        raise SystemExit("--checkpoint is required (or --summary)")
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    import torch

    from mdfm import paths
    from mdfm.data import sources as SRC
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.eval import probes
    from mdfm.gates import common, g3
    from mdfm.model import encoder as ENC
    from mdfm.model import tokenize as TK
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    name = I.checkpoint_name(a.checkpoint)
    if a.limit_persons and not a.out:
        raise SystemExit("--limit-persons needs an explicit --out (a partial run never takes the default folder)")
    out = os.path.abspath(a.out or os.path.join(TREE, "reports", "bench", f"frozen_tokens_{name}"))
    paths.assert_not_moveatlas(out)
    features = os.path.abspath(a.features or os.path.join(
        TREE, "cache", "runs", "frozen_tokens", name + (f"_limit{a.limit_persons}" if a.limit_persons else ""),
        "features.npz"))
    code_files = [probes.__file__, os.path.abspath(__file__), I.__file__, g3.__file__, TK.__file__, ENC.__file__]
    code_sha = common.code_sha256(code_files)

    train, dev = g3.controls("train"), g3.controls("dev")
    if len(train) != 54 or len(dev) != 14 or set(train) & set(dev):
        raise SystemExit(f"folds: train {len(train)}, dev {len(dev)}, overlap {sorted(set(train) & set(dev))}")
    if sorted(set(EXCLUDED_FROM_12) - set(dev)):
        raise SystemExit(f"excluded persons not among the dev controls: {sorted(set(EXCLUDED_FROM_12) - set(dev))}")
    locked = set(g3.lockbox_persons())
    if (set(train) | set(dev)) & locked:
        raise SystemExit("a lockbox person among the non-lockbox controls")
    if a.limit_persons:
        train = train[:max(int(a.limit_persons), 2 * probes.FOLDS)]        # nested folds need enough persons
        dev = dev[:max(int(a.limit_persons), 2)]
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise SystemExit("a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    print(f"checkpoint {name}; targets: train {tg_train.counts['passes']} passes, dev {tg_dev.counts['passes']}",
          flush=True)

    mem = {}
    if a.from_features:
        data, meta = load_features(features)
        check_features_meta(meta, name, rel(features))
        for s, u in (("train", u_tr), ("dev", u_dev)):
            f = data["units"][s]
            if len(f) != len(u) or list(f["trial_id"]) != list(u["trial_id"]) or \
                    list(f["pass_id"].astype(str)) != list(u["pass_id"].astype(str)):
                raise SystemExit(f"{rel(features)}: {s} passes differ from the targets read now")
            data["units"][s] = u
        ck, inference = meta["checkpoint"], meta["inference"]
        train_in_ckpt = meta["train_among_checkpoint_persons"]
        t_tok = meta.get("tokens_s")
        print(f"features read from {rel(features)} (written {meta.get('written_at')})", flush=True)
    else:
        dev_t = torch.device(a.device)
        if dev_t.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda without a visible GPU (the GPU runs through scripts/queue_job.py)")
        cap = LP.cap_memory(dev_t, a.memory_gb)
        if dev_t.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev_t)
        loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev_t)
        trained = loaded.trained_persons()
        if set(dev) & trained:
            raise Refused(f"dev controls among the checkpoint's training persons: {sorted(set(dev) & trained)}")
        train_in_ckpt = len(set(train) & trained)
        print(f"loaded {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step {loaded.step}; "
              f"training-fold controls among its training persons: {train_in_ckpt}", flush=True)

        def encode(wins):
            return encode_windows(loaded.model, wins, a.batch_windows)

        def records(persons, tg):
            for tid, rec in g3.imu_records(persons, tg.trials):          # canonical.load, reader flags False
                yield tid, g3.model_input(rec)

        t0 = time.time()
        data = {"units": {"train": u_tr, "dev": u_dev}}
        data["train"] = collect(encode, records(train, tg_train), u_tr, tg_train, train)
        data["dev"] = collect(encode, records(dev, tg_dev), u_dev, tg_dev, dev)
        t_tok = round(time.time() - t0, 1)
        if dev_t.type == "cuda":
            mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev_t) / 1e9,
                   "max_reserved_gb": torch.cuda.max_memory_reserved(dev_t) / 1e9, "cap_gb": cap}
        ck = loaded.describe()
        inference = {"window_s": I.WIN_S, "hop_s": I.HOP_S, "batch_windows": a.batch_windows, "weights": a.weights,
                     "device": str(dev_t), "amp_bf16": dev_t.type == "cuda", "refine_steps": 0, "decoder": False,
                     "memory": mem}
        meta = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": ck, "inference": inference,
                "train_among_checkpoint_persons": train_in_ckpt, "tokens_s": t_tok, "code_sha256": code_sha,
                "limit_persons": a.limit_persons}
        save_features(features, data, meta)
        print(f"features written to {rel(features)} ({t_tok} s)", flush=True)
        del loaded

    t0 = time.time()
    res0 = run_probes(data, tg_train, tg_dev)
    t_probe = round(time.time() - t0, 1)
    res = {"what": LABEL, "task": TASK, "run_started": run_started, "partial": bool(a.limit_persons),
           "checkpoint": ck, "inference": inference, "features": {"path": rel(features), "sha256": sha256_file(features)},
           "sets": {"train": sorted(train), "train_sha256": SRC.person_hash(sorted(train)),
                    "dev14": res0["sets"]["dev14"], "dev14_sha256": SRC.person_hash(res0["sets"]["dev14"]),
                    "dev12": res0["sets"]["dev12"], "dev12_sha256": SRC.person_hash(res0["sets"]["dev12"]),
                    "train_among_checkpoint_persons": train_in_ckpt},
           "excluded_from_12": res0["excluded_from_12"], "primary": PRIMARY,
           "counts": {"train": data["train"]["counts"], "dev": data["dev"]["counts"],
                      "targets": {"train": tg_train.counts, "dev": tg_dev.counts}},
           "constants": {"alphas": list(probes.ALPHAS), "folds": probes.FOLDS, "seed": g3.SEED, "probe_seed": probes.SEED,
                         "boot_n": g3.BOOT_N, "sub_bins": probes.SUB_BINS, "ic_threshold": probes.IC_THRESHOLD,
                         "ic_nms_s": probes.IC_NMS_S, "span_margin_s": probes.SPAN_MARGIN_S, "keep_pad_s": KEEP_PAD_S,
                         "pool": list(probes.POOL)},
           "passes": res0["passes"], "ic": res0["ic"], "population": res0["population"], "leakage": res0["leakage"],
           "lockbox": {"persons_in_lockbox": len((set(train) | set(dev)) & locked),
                       "target_lockbox_rows": int(tg_train.counts["lockbox_rows"]) + int(tg_dev.counts["lockbox_rows"]),
                       "reader_flags": {"allow_eval": False, "allow_lockbox": False}},
           "moveatlas_head": PV.moveatlas_head()}
    res["comparison"] = comparison(res, name)
    res["timing_s"] = {"tokens": t_tok, "probes": t_probe, "all": round(time.time() - t_start, 1)}
    res = clean(res)
    text = report_md(res)
    common.write_text(os.path.join(out, "results.json"), json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    common.write_text(os.path.join(out, "summary.md"), text)
    common.write_csv(res0["per_pass"], os.path.join(out, "per_pass.csv"))
    prov = {"script": "scripts/probe_frozen_tokens.py", "label": LABEL, "task": TASK, "run_at": run_started,
            "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": res["moveatlas_head"],
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "checkpoint": ck.get("path"), "checkpoint_sha256": ck.get("sha256"), "features": res["features"],
            "code_sha256": code_sha, "code_files": [rel(f) for f in code_files],
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(clean(prov), indent=1) + "\n")
    print(text)
    print(f"written {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
