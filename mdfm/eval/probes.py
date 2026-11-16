"""Linear probes on frozen encoder tokens (weekend task T36, J0; reused by the G12 baseline, T25).

Question the probes answer: does a gait quantity sit in a representation (a linear read of frozen tokens recovers
it), or is it lost later (decoding, root integration)? Everything here is numpy; nothing loads a model, a record or
a target file, so the callers decide what is read and the tests run on synthetic features.

Person cross-fitting. Every fit that is scored keeps each person on one side: person_folds deals the sorted
person ids, permuted by a seed, round-robin into k folds (every row of a person in one fold); cross_fit_splits
yields the (fit, score) row indices of each fold and asserts that no person sits on both sides (check_disjoint,
ValueError otherwise). select_alpha picks the ridge penalty by these folds inside the fitting set only; cross_fit
nests that choice inside each outer fold, so a scored person never informs the penalty that scores it.

Ridge (Ridge, ridge_path): features standardized by the fitting set (a column without variance contributes
nothing), targets centred, closed form through one SVD per fit, the penalty on the standardized coefficients from
the grid ALPHAS; the chosen alpha minimizes the out-of-fold mean squared error summed over the target columns (the
first of equal values, the smallest alpha). A row with a non-finite feature or target is left out of a fit, and a
row with a non-finite feature predicts NaN.

Pass probes (stride length and speed per walkway pass). pass_units: one row per walkway pass of a g3.Targets
(trial, person, task, pass), its span (the union of the pass speed interval and of its walkway strides), the mean
walkway stride (m) and the pass speed (m/s), as scripts/g3_supervised_baseline.py defines its units. pool_spans: the
mean and the standard deviation over the token rows whose time lies in a span (both pooled, so that an
oscillation amplitude survives the pooling). pass_predictions turns pass predictions into g3.Predictions (every
walkway stride of a pass gets its pass's predicted stride, every pass its predicted speed), scored by the G3 lines
(score: g3.stride_lines, g3.speed_line; subset_targets restricts a Targets to some persons).

IC per sub-patch. A token stands for its 0.25 s patch [start, start + 0.25) on the trial clock, cut into
SUB_BINS sub-patches of 50 ms. ic_targets labels a token when its patch lies inside a span of its trial within
SPAN_MARGIN_S, one patch, so that the patches holding a pass's first and last footfalls count (a walking step lasts
longer than a patch, so no footfall off the walkway falls in a labelled patch): per foot (FEET order) a one-hot over
the sub-patches holding a reference IC of that foot, zeros when the patch holds none; unlabelled tokens (outside
every span: off the walkway, where an IC has no reference) never enter a fit. A ridge on the labelled tokens gives
per foot SUB_BINS scores; decode_ics emits an IC at the centre of the best sub-patch when the foot's scores sum above
IC_THRESHOLD (the probe's estimate of the probability that the patch holds an IC), then keeps, per trial and foot,
the higher-scored of two ICs closer than IC_NMS_S (readouts.gait.REFRACTORY_S). most_central marks, for tokens of
overlapping windows, the window where the token's time is most central (readouts.gait.stitch's rule, ties to the
earlier window), so that a trial-clock patch is decoded once.

Leakage hook (G10 style, design section G10 (a)): leakage_probe fits a person-cross-fitted linear classifier (ridge
on one-hot labels, argmax) of a nuisance label (source, site, device family) from features, and the same probe on the
legitimate covariates only (chance, 1 / classes, when none is given: what any constant prediction scores); the leakage
number is the difference of the two balanced accuracies in points. Exploratory until a G10 rule fixes the strata and
the features. With rows=True it also returns the per-row out-of-fold predictions with each row's person (task G12R),
from which the number is recomputed on any resample of persons.
"""
from dataclasses import dataclass

import numpy as np

PATCH_S = 0.25                     # s, the latent patch of the tokenizer (model/tokenize.PATCH_S)
SUB_BINS = 5                       # sub-patches of 50 ms per patch (IC probe)
FEET = ("L", "R")                  # readouts.gait.FEET
ALPHAS = tuple(float(10.0 ** e) for e in np.arange(-2.0, 6.01, 0.5))
FOLDS = 5
SEED = 20261004
IC_THRESHOLD = 0.5
IC_NMS_S = 0.3                     # readouts.gait.REFRACTORY_S
SPAN_MARGIN_S = 0.25               # one patch
UNIT_KEYS = ["trial_id", "person_id", "task", "pass_id"]
POOL = ("mean", "std")


# ---------------------------------------------------------------- person cross-fitting
def person_folds(persons, k=FOLDS, seed=SEED):
    """(N,) int fold of each row: the sorted unique persons permuted by seed and dealt round-robin into k folds,
    every row of a person in its person's fold. ValueError with fewer than k persons."""
    p = np.asarray(persons).astype(str)
    ids = np.unique(p)
    if len(ids) < k:
        raise ValueError(f"{len(ids)} persons for {k} folds")
    order = np.random.default_rng(int(seed)).permutation(len(ids))
    fold_of_id = np.empty(len(ids), np.int64)
    fold_of_id[order] = np.arange(len(ids)) % k
    return fold_of_id[np.searchsorted(ids, p)]


def check_disjoint(fit_persons, score_persons):
    """ValueError when a person is on both sides of a fit."""
    both = sorted(set(np.asarray(fit_persons).astype(str)) & set(np.asarray(score_persons).astype(str)))
    if both:
        raise ValueError(f"{len(both)} persons on both sides of a fit, e.g. {both[:3]}")


def cross_fit_splits(persons, k=FOLDS, seed=SEED):
    """[(fit rows, score rows)] of the k person folds, each checked by check_disjoint."""
    p = np.asarray(persons).astype(str)
    folds = person_folds(p, k, seed)
    out = []
    for f in range(k):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        check_disjoint(p[tr], p[te])
        out.append((tr, te))
    return out


# ---------------------------------------------------------------- ridge
def _2d(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return Y[:, None] if Y.ndim == 1 else Y


def _finite_rows(X, Y):
    return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)


@dataclass
class Ridge:
    """A fitted ridge: predict(X) = ((X - mean) / scale) @ coef + y_mean, scale inf for a column without variance;
    (N,) for a target fitted as (N,), else (N, T); NaN on a row with a non-finite feature."""
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray
    squeeze: bool = False

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        bad = ~np.isfinite(X).all(axis=1)
        out = ((np.where(bad[:, None], 0.0, X) - self.mean) / self.scale) @ self.coef + self.y_mean
        out[bad] = np.nan
        return out[:, 0] if self.squeeze else out


def ridge_path(X, Y, alphas=ALPHAS):
    """[Ridge] of the standardized ridge of (X, Y) at each alpha, from one SVD (rows with a non-finite value left
    out). Y (N,) or (N, T)."""
    X = np.asarray(X, dtype=np.float64)
    squeeze = np.ndim(Y) == 1
    Y = _2d(Y)
    keep = _finite_rows(X, Y)
    if keep.sum() < 2:
        raise ValueError(f"{int(keep.sum())} finite rows to fit")
    X, Y = X[keep], Y[keep]
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    scale = np.where(sd > 1e-12 * np.maximum(1.0, np.abs(mean)), sd, np.inf)
    Z = (X - mean) / scale
    y_mean = Y.mean(axis=0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    UtY = U.T @ (Y - y_mean)
    out = []
    for a in alphas:
        coef = Vt.T @ ((s / (s * s + float(a)))[:, None] * UtY)
        out.append(Ridge(float(a), mean, scale, y_mean, coef, squeeze))
    return out


def fit_ridge(X, Y, alpha):
    """Ridge of (X, Y) at one alpha."""
    return ridge_path(X, Y, (alpha,))[0]


def select_alpha(X, Y, persons, alphas=ALPHAS, k=FOLDS, seed=SEED):
    """(alpha, {alpha: out-of-fold MSE}) over the person folds of the fitting set (module docstring)."""
    X = np.asarray(X, dtype=np.float64)
    Y = _2d(Y)
    p = np.asarray(persons).astype(str)
    sse = np.zeros(len(alphas))
    n = 0
    for tr, te in cross_fit_splits(p, k, seed):
        ok = _finite_rows(X[te], Y[te])
        if not ok.any():
            continue
        path = ridge_path(X[tr], Y[tr], alphas)
        te = te[ok]
        for i, m in enumerate(path):
            sse[i] += float(((_2d(m.predict(X[te])) - Y[te]) ** 2).sum())
        n += len(te)
    if not n:
        raise ValueError("no finite out-of-fold row to select alpha")
    mse = sse / n
    i = int(np.argmin(mse))
    return float(alphas[i]), {float(a): float(v) for a, v in zip(alphas, mse)}


def fit_probe(X, Y, persons, alphas=ALPHAS, k=FOLDS, seed=SEED):
    """(Ridge at the selected alpha fitted on every row, {'alpha', 'cv_mse'})."""
    a, cv = select_alpha(X, Y, persons, alphas, k, seed)
    return fit_ridge(X, Y, a), {"alpha": a, "cv_mse": cv}


def cross_fit(X, Y, persons, alphas=ALPHAS, k=FOLDS, seed=SEED):
    """(out-of-fold predictions, [alpha of each outer fold]): outer person folds (seed), the alpha selected inside
    each fitting part by its own person folds (seed + 1); a scored person is never in the fit nor in the selection
    that scores it."""
    X = np.asarray(X, dtype=np.float64)
    squeeze = np.ndim(Y) == 1
    Y = _2d(Y)
    p = np.asarray(persons).astype(str)
    out = np.full(Y.shape, np.nan)
    chosen = []
    for tr, te in cross_fit_splits(p, k, seed):
        m, sel = fit_probe(X[tr], Y[tr], p[tr], alphas, k, seed + 1)
        out[te] = _2d(m.predict(X[te]))
        chosen.append(sel["alpha"])
    return (out[:, 0] if squeeze else out), chosen


# ---------------------------------------------------------------- pass units and pooling
def pass_units(targets):
    """One row per walkway pass of a g3.Targets (module docstring): UNIT_KEYS, t0, t1, n_strides, stride_m (NaN
    without a stride), speed_m_s (NaN without a pass speed)."""
    import pandas as pd
    s = targets.strides.copy()
    p = targets.passes.copy()
    s["pass_id"] = s["pass_id"].astype(str)
    p["pass_id"] = p["pass_id"].astype(str)
    if p.duplicated(UNIT_KEYS).any():
        raise ValueError("a pass listed twice in the targets")
    sg = (s.groupby(UNIT_KEYS, sort=True)
          .agg(s_t0=("t0", "min"), s_t1=("t1", "max"), stride_m=("length_m", "mean"), n_strides=("length_m", "size"))
          .reset_index()) if len(s) else pd.DataFrame(columns=UNIT_KEYS + ["s_t0", "s_t1", "stride_m", "n_strides"])
    pg = p[UNIT_KEYS + ["t0", "t1", "speed_m_s"]].rename(columns={"t0": "p_t0", "t1": "p_t1"})
    u = sg.merge(pg, on=UNIT_KEYS, how="outer")
    for c in ("s_t0", "s_t1", "p_t0", "p_t1", "stride_m", "speed_m_s"):
        u[c] = pd.to_numeric(u[c], errors="coerce").astype(float)
    u["t0"] = u[["s_t0", "p_t0"]].min(axis=1)
    u["t1"] = u[["s_t1", "p_t1"]].max(axis=1)
    u["n_strides"] = pd.to_numeric(u["n_strides"], errors="coerce").fillna(0).astype(int)
    u = u.sort_values(UNIT_KEYS, kind="stable").reset_index(drop=True)
    return u[UNIT_KEYS + ["t0", "t1", "n_strides", "stride_m", "speed_m_s"]]


def pool_spans(times, feats, spans, how=POOL):
    """((n_spans, D x len(how)) pooled features, (n_spans,) row counts): for each span (t0, t1) the statistics
    `how` ('mean', 'std', in that order of blocks) over the rows of feats (N, D) whose time lies in [t0, t1] (a row
    with a non-finite value left out); NaN for a span without a row."""
    t = np.asarray(times, dtype=np.float64)
    F = np.asarray(feats, dtype=np.float64).reshape(len(t), -1)
    spans = np.asarray(spans, dtype=np.float64).reshape(-1, 2)
    D = F.shape[1]
    out = np.full((len(spans), D * len(how)), np.nan)
    n = np.zeros(len(spans), np.int64)
    finite = np.isfinite(F).all(axis=1) & np.isfinite(t)
    for i, (a, b) in enumerate(spans):
        sel = finite & (t >= a - 1e-9) & (t <= b + 1e-9)
        n[i] = int(sel.sum())
        if not n[i]:
            continue
        x = F[sel]
        parts = []
        for h in how:
            if h == "mean":
                parts.append(x.mean(axis=0))
            elif h == "std":
                parts.append(x.std(axis=0))
            else:
                raise ValueError(f"unknown pooling {h!r}")
        out[i] = np.concatenate(parts)
    return out, n


def most_central(win_start, win_dur, token_win, token_time):
    """(N,) bool: True when a token's window is the window most central at the token's time among the windows that
    cover it (start <= t < start + duration; distance of t from the window's middle, ties to the earlier window)."""
    s = np.asarray(win_start, dtype=np.float64)
    d = np.asarray(win_dur, dtype=np.float64)
    w = np.asarray(token_win, dtype=np.int64)
    t = np.asarray(token_time, dtype=np.float64)
    if not len(t):
        return np.zeros(0, bool)
    cover = (t[:, None] >= s[None] - 1e-9) & (t[:, None] < (s + d)[None] - 1e-9)
    dist = np.where(cover, np.abs(t[:, None] - (s + d / 2.0)[None]), np.inf)
    best = np.argmin(dist, axis=1)              # the first of equal distances: the earlier window
    return (best == w) & np.isfinite(dist[np.arange(len(t)), w])


# ---------------------------------------------------------------- predictions and lines
def pass_predictions(units, pred, targets, name="probe"):
    """g3.Predictions of pass predictions {'stride_m': (n_units,), 'speed_m_s': (n_units,)}: every walkway stride of
    the targets gets its pass's predicted stride with its own foot and end times, every pass its predicted speed; a
    stride or pass whose pass has no finite prediction is left out (unmatched in the lines)."""
    import pandas as pd

    from mdfm.gates import g3
    key = list(zip(units["trial_id"], units["pass_id"].astype(str)))
    ps = {k: float(v) for k, v in zip(key, pred["stride_m"]) if np.isfinite(v)}
    pv = {k: float(v) for k, v in zip(key, pred["speed_m_s"]) if np.isfinite(v)}
    s = targets.strides
    sk = list(zip(s["trial_id"], s["pass_id"].astype(str)))
    st = s.loc[[k in ps for k in sk], ["trial_id", "foot", "t0", "t1"]].copy()
    st["length_m"] = [ps[k] for k in sk if k in ps]
    sp = pd.DataFrame([(t, p, v) for (t, p), v in pv.items()], columns=["trial_id", "pass_id", "speed_m_s"])
    return g3.Predictions(name, None, st.reset_index(drop=True), sp)


def subset_targets(tg, persons):
    """The g3.Targets restricted to `persons` (every frame filtered on person_id)."""
    from mdfm.gates import g3
    keep = set(persons)
    f = lambda d: d[d["person_id"].isin(keep)].reset_index(drop=True)  # noqa: E731
    trials = f(tg.trials)
    offsets = tg.offsets[tg.offsets["trial_id"].isin(set(trials["trial_id"]))].reset_index(drop=True)
    return g3.Targets(tuple(sorted(keep & set(tg.persons))), tg.lockbox, trials, f(tg.walkway_ic), f(tg.insole_ic),
                      f(tg.strides), f(tg.passes), offsets, {"strides": int(len(f(tg.strides))),
                                                             "passes": int(len(f(tg.passes)))})


def score(targets, predictions):
    """{name: {line: stats}} of the G3 stride and speed lines (g3.stride_lines, g3.speed_line) of each predictor."""
    from mdfm.gates import g3
    out = {}
    for p in predictions:
        s = {}
        if p.strides is not None:
            lines, _ = g3.stride_lines(p, targets)
            s.update(lines)
        if p.speeds is not None:
            s["speed_mae_m_s"], _ = g3.speed_line(p, targets)
        out[p.name] = s
    return out


# ---------------------------------------------------------------- IC per sub-patch
def ic_targets(starts, trial_ids, ref_ics, spans, bins=SUB_BINS, patch_s=PATCH_S, margin=SPAN_MARGIN_S):
    """(Y (N, len(FEET) x bins), labelled (N,) bool) of tokens whose patches start at `starts` (s, trial clock) in
    trials `trial_ids` (module docstring). ref_ics: DataFrame (trial_id, foot, t_s); spans: DataFrame (trial_id, t0,
    t1), the spans inside which every IC has a reference."""
    st = np.asarray(starts, dtype=np.float64)
    tid = np.asarray(trial_ids).astype(str)
    Y = np.zeros((len(st), len(FEET) * bins))
    lab = np.zeros(len(st), bool)
    w = patch_s / bins
    sp = {str(t): g[["t0", "t1"]].to_numpy(np.float64) for t, g in spans.groupby("trial_id")}
    ics = {(str(t), str(f)): np.sort(g["t_s"].to_numpy(np.float64))
           for (t, f), g in ref_ics.groupby(["trial_id", "foot"])}
    for t in np.unique(tid):
        rows = np.flatnonzero(tid == t)
        a = st[rows]
        if t in sp:
            s = sp[t]
            lab[rows] = ((a[:, None] >= s[None, :, 0] - margin - 1e-9)
                         & (a[:, None] + patch_s <= s[None, :, 1] + margin + 1e-9)).any(axis=1)
        for fi, foot in enumerate(FEET):
            x = ics.get((t, foot))
            if x is None or not len(x):
                continue
            j = np.searchsorted(x, a - 1e-9, side="left")
            ok = j < len(x)
            j = np.minimum(j, len(x) - 1)
            ok &= x[j] < a + patch_s - 1e-9
            b = np.clip(np.floor((x[j] - a) / w + 1e-9).astype(np.int64), 0, bins - 1)
            Y[rows[ok], fi * bins + b[ok]] = 1.0
    return Y, lab


def decode_ics(scores, starts, trial_ids, bins=SUB_BINS, patch_s=PATCH_S, threshold=IC_THRESHOLD, nms_s=IC_NMS_S):
    """DataFrame (trial_id, foot, t_s, score) of the ICs read from per-token scores (N, len(FEET) x bins) (module
    docstring): one IC per token and foot whose scores sum above threshold, at the centre of the best sub-patch, then
    per trial and foot the higher-scored of two ICs within nms_s kept."""
    import pandas as pd
    st = np.asarray(starts, dtype=np.float64)
    S = np.asarray(scores, dtype=np.float64).reshape(len(st), -1)
    tid = np.asarray(trial_ids).astype(str)
    w = patch_s / bins
    rows = []
    for fi, foot in enumerate(FEET):
        s = S[:, fi * bins:(fi + 1) * bins]
        p = s.sum(axis=1)
        b = np.argmax(np.where(np.isfinite(s), s, -np.inf), axis=1)
        keep = np.isfinite(p) & (p > threshold)
        for i in np.flatnonzero(keep):
            rows.append((tid[i], foot, float(st[i] + (b[i] + 0.5) * w), float(p[i])))
    df = pd.DataFrame(rows, columns=["trial_id", "foot", "t_s", "score"])
    out = []
    for (t, foot), g in df.groupby(["trial_id", "foot"], sort=True):
        g = g.sort_values(["score", "t_s"], ascending=[False, True], kind="stable")
        kept = []
        for r in g.itertuples():
            if all(abs(r.t_s - k) >= nms_s - 1e-9 for k in kept):
                kept.append(r.t_s)
                out.append((t, foot, r.t_s, r.score))
    return pd.DataFrame(out, columns=["trial_id", "foot", "t_s", "score"]).sort_values(
        ["trial_id", "foot", "t_s"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- leakage hook (G10 style)
def balanced_accuracy(y_true, y_pred):
    """Mean over the classes present in y_true of the share of their rows predicted right."""
    y = np.asarray(y_true).astype(str)
    p = np.asarray(y_pred).astype(str)
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def _classify(X_fit, lab_fit, persons_fit, X_score, classes, alphas, k, seed):
    Y = (np.asarray(lab_fit).astype(str)[:, None] == np.asarray(classes)[None]).astype(np.float64)
    m, _ = fit_probe(X_fit, Y, persons_fit, alphas, k, seed)
    return np.asarray(classes)[np.argmax(_2d(m.predict(X_score)), axis=1)]


def leakage_probe(features, labels, persons, covariates=None, alphas=ALPHAS, k=FOLDS, seed=SEED, rows=False):
    """G10-style leakage number (module docstring): {'balanced_accuracy' of the person-cross-fitted probe on the
    features, 'covariate_balanced_accuracy' of the same probe on the covariates (chance when None),
    'leakage_points' 100 x their difference, 'chance' 1 / classes, 'n_rows', 'n_persons', 'classes'}. Every row is
    predicted by a fit that holds none of its person's rows (cross_fit_splits), and so is the penalty choice.
    rows=True adds 'rows': the per-row out-of-fold predictions with each row's person, {'person_id', 'label',
    'predicted', 'covariate_predicted' (None without covariates)} as lists in the input order, which a person bootstrap
    of the leakage number resamples (docs/rules/G12_RULE.md section 8 and item 3 of its section 11); the default return
    is unchanged."""
    X = np.asarray(features, dtype=np.float64)
    lab = np.asarray(labels).astype(str)
    p = np.asarray(persons).astype(str)
    classes = np.unique(lab)
    if len(classes) < 2:
        raise ValueError("a leakage probe needs at least two classes")
    C = None if covariates is None else np.asarray(covariates, dtype=np.float64).reshape(len(lab), -1)
    pred_x = np.empty(len(lab), dtype=object)
    pred_c = np.empty(len(lab), dtype=object)
    for tr, te in cross_fit_splits(p, k, seed):
        pred_x[te] = _classify(X[tr], lab[tr], p[tr], X[te], classes, alphas, k, seed + 1)
        if C is not None:
            pred_c[te] = _classify(C[tr], lab[tr], p[tr], C[te], classes, alphas, k, seed + 1)
    ba = balanced_accuracy(lab, pred_x)
    bc = 1.0 / len(classes) if C is None else balanced_accuracy(lab, pred_c)
    out = {"balanced_accuracy": ba, "covariate_balanced_accuracy": bc, "leakage_points": 100.0 * (ba - bc),
           "chance": 1.0 / len(classes), "n_rows": int(len(lab)), "n_persons": int(len(np.unique(p))),
           "classes": classes.tolist()}
    if rows:
        out["rows"] = {"person_id": p.tolist(), "label": lab.tolist(), "predicted": [str(x) for x in pred_x],
                       "covariate_predicted": None if C is None else [str(x) for x in pred_c]}
    return out
