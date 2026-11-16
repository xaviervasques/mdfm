"""Section 8 of docs/rules/G12_RULE.md (DRAFT; revision docs/rules/G12_RULE_REVISION_2026-10-08.md, hashed): the
comparison of the body with its comparator on a line, from per-row predictions (task G12R). Numpy and pandas only:
nothing here loads a model, a record or a target file, and nothing is fitted (the rows are the predictions each seed
made once on the full set; the bootstrap never refits anything).

Rows (ROW_COLUMNS, one frame per arm and seed): one row per unit of a line with its person: line, variant, chosen,
person_id, identification columns (trial_id, task, pass_id, foot, t0, t1, t_s), target, prediction, matched, enters
and value. 'enters' marks the rows the G3 harness passes to its line statistic (mdfm/gates/g3._line: the matched
strides of stride_mae_cm, the matched passes of speed_mae_m_s, the covered reference ICs of ic_median_ms) and 'value'
is the row's value there (absolute stride error in cm, absolute speed error in m/s, absolute IC error in ms, infinite
for an unmatched reference IC). A source that keeps per-person values only (the decoded body's per_person.csv of the
G3 harness) gives one entering row per person whose value is the person's value: the statistic below reads both the
same way.

Line statistic (line_point, the G3 lines): per person the mean (stride, speed) or the median (IC) of the entering
values (NaN skipped, as pandas does in g3._person_summary), then the mean over persons (np.mean: an infinite or NaN
person value propagates, as in g3._line); a person without an entering row drops out of the line; NaN without any.
On the same rows this is g3's point exactly (tests/test_g12_readouts.py).

Separation (separation, section 8 (i)): the body separates when every body seed's point is strictly better than every
comparator seed's point; the comparator separates in the mirror case; a tie, an overlap or a NaN point separates
nothing. Better is lower for every G3 line (LINES); higher_is_better flips it.

Comparator (pick_comparator, section 8): the baseline arm with the better mean of its seed points; ties to the
earlier arm of ARMS (mfr, the design's baseline); an arm marked unusable (collapsed, or taken out by a revision) is
left out, so the MFR arm alone is the comparator when the JEPA arm is out or collapsed.

Interval (bootstrap_line, section 8 (ii)): one index matrix of draws (n_boot x P, rng = numpy default_rng(seed),
integers(0, P), as mdfm/gates/common.cluster_bootstrap draws) over the line's persons sorted; each resample takes the
drawn persons with replacement (a person drawn twice is two units and counts twice), recomputes the line of every seed
of both models from the drawn persons' rows (resample_line: the mean over the drawn slots of the persons' values, a
slot whose person has no entering row left out), then the difference of the body's seed mean and the comparator's seed
mean; the interval is the 2.5 and 97.5 percentiles of the differences (percentile_interval: np.quantile, linear;
with infinite differences the same positions, an infinite neighbour returned; None when a difference is undefined,
infinite minus infinite or a resample without any entering person). For the G3 lines the per-person decomposition is
exact; bootstrap_rows is the general path (a statistic recomputed on the drawn rows themselves, each drawn slot
relabelled as its own person), for population statistics (G8's QWK, G10's leakage number) and as the reference of
the fast path in the tests. n_boot and seed: the gate rule's (G3: 2,000 resamples, seed 20260929, the constants
BOOT_N and SEED of mdfm/gates/g3.py).

Reading (reading): the body beats the comparator on a line when it separates and the interval lies wholly on its side
of 0 (below 0 for a lower-is-better line); it loses in the mirror case; no difference otherwise.

Preconditions (evaluate): no comparison is computed (status NOT EVALUATED, with the list of what is missing) while
the owner has not registered docs/rules/G12_RULE.md (draft section 1, revision header), while the body or a usable
baseline arm has fewer seeds with rows than the rule asks (three, section 4; both arms in, revision item 1), while a
usable arm holds a seed the body has not, or holds its three seeds and they are not the body's (section 4 pairs a
baseline seed with the body seed of the same value; section 11 item 5), while no usable arm is left, or (source_checks,
when the caller passes the sources' provenance) while the sources do not all state one MoveAtlas commit or a source's
dev set is not the comparison's. The points of every arm, seed, set and line with rows are reported in every case
("MFR seeds alone may be read on dev probes and reported as rows, without a verdict", revision item 1). G8, G10 and
G7 lines are NOT EVALUATED here (OTHER_LINES): G8 and G10 have no rule files and bind only through a later dated
revision (revision item 3); G7 reads under the CODY consensus rule draft (item 4), whose readout is not built.
"""
import math

import numpy as np

NOT_EVALUATED = "NOT EVALUATED"
EVALUATED = "EVALUATED"
BODY = "body"
ARMS = ("mfr", "jepa")             # mdfm/baselines/config.ARMS order: ties of the comparator pick go to mfr
SEEDS_REQUIRED = 3                 # rule section 4
BOOT_N = 2000                      # mdfm/gates/g3.BOOT_N (docs/rules/G3_RULE.md: 2,000 resamples)
SEED = 20260929                    # mdfm/gates/g3.SEED (docs/rules/G3_RULE.md: seed 20260929)
LEVEL = 0.95
LINES = {"stride_mae_cm": {"how": "mean", "higher_is_better": False},
         "speed_mae_m_s": {"how": "mean", "higher_is_better": False},
         "ic_median_ms": {"how": "median", "higher_is_better": False}}
ROW_COLUMNS = ["line", "variant", "chosen", "person_id", "trial_id", "task", "pass_id", "foot", "t0", "t1", "t_s",
               "target", "prediction", "matched", "enters", "value"]
RULE_PATH = "docs/rules/G12_RULE.md"
SEED_PAIRING = ("rule section 4: a baseline and the body of the same seed start from the same weights; section 11 "
                "item 5: the same seed values for the body and the baselines")
OTHER_LINES = {
    "G8": "no rule file (docs/rules has no G8 rule); revision 2026-10-08 item 3: the G8 lines bind only through a "
          "later dated revision that carries their attainability, hashed before any G12 comparison reads them",
    "G10": "no rule file (docs/rules has no G10 rule); revision 2026-10-08 item 3: the G10 lines bind only through a "
           "later dated revision that carries their attainability; mdfm/eval/probes.leakage_probe(rows=True) keeps "
           "the per-row predictions with each row's person that interval (ii) needs",
    "G7": "read under docs/rules/CODY_CONSENSUS_RULE_DRAFT.md (a DRAFT; revision 2026-10-08 item 4 adopts section 8 "
          "for G7 on the G12 side); no G7 readout of the baselines exists",
}


# ---------------------------------------------------------------- the line statistic
def _frame(rows):
    import pandas as pd
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def _enters(rows):
    e = rows["enters"]
    if e.dtype == object:
        e = e.map(lambda x: str(x).strip().lower() in ("true", "1"))
    return e.astype(bool).to_numpy()


def person_values(rows, how="mean", persons=None):
    """(ids sorted (P,), values (P,)) of the persons with an entering row (module docstring), restricted to
    `persons` when given; the per-person mean or median of the entering values, NaN skipped (pandas), a person whose
    entering values are all NaN kept with NaN."""
    import pandas as pd
    r = _frame(rows)
    r = r[_enters(r)]
    if persons is not None:
        r = r[r["person_id"].astype(str).isin(set(map(str, persons)))]
    if not len(r):
        return np.zeros(0, dtype=str), np.zeros(0)
    s = pd.Series(pd.to_numeric(r["value"], errors="coerce").to_numpy(np.float64)).groupby(
        r["person_id"].astype(str).to_numpy())
    v = s.mean() if how == "mean" else s.median()
    return v.index.to_numpy().astype(str), v.to_numpy(np.float64)


def line_point(rows, how="mean", persons=None):
    """The line on `rows` (module docstring): the mean over persons of person_values; NaN without any person."""
    ids, v = person_values(rows, how, persons)
    return float(np.mean(v)) if len(ids) else float("nan")


def values_on(persons, rows, how="mean"):
    """(v (P,), present (P,) bool) over `persons` in their given order: each person's value, present False (and v
    NaN) for a person without an entering row."""
    ids, v = person_values(rows, how, persons)
    got = dict(zip(ids.tolist(), v.tolist()))
    p = [str(x) for x in persons]
    return (np.array([got.get(x, np.nan) for x in p], dtype=np.float64),
            np.array([x in got for x in p], dtype=bool))


# ---------------------------------------------------------------- separation, comparator, reading
def _isnan(x):
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def separation(body_points, comp_points, higher_is_better=False):
    """'body' when every body point is strictly better than every comparator point, 'comparator' in the mirror
    case, None otherwise (a tie, an overlap, an empty side or a NaN point)."""
    b = np.asarray(list(body_points), dtype=np.float64)
    c = np.asarray(list(comp_points), dtype=np.float64)
    if not len(b) or not len(c) or np.isnan(b).any() or np.isnan(c).any():
        return None
    if higher_is_better:
        b, c = -b, -c
    if b.max() < c.min():
        return BODY
    if b.min() > c.max():
        return "comparator"
    return None


def seed_mean(points):
    """Mean of seed points (inf propagates); NaN for none or a NaN point."""
    p = np.asarray(list(points), dtype=np.float64)
    if not len(p) or np.isnan(p).any():
        return float("nan")
    return float(np.mean(p))


def pick_comparator(points, higher_is_better=False, unusable=None):
    """(arm or None, {arm: seed mean}) over the arms of ARMS order that hold points and are not in `unusable`
    (module docstring); None when no arm remains or a remaining arm's mean is NaN."""
    unusable = unusable or {}
    means = {a: seed_mean(points[a]) for a in ARMS if a in points and a not in unusable and len(points[a])}
    if not means or any(_isnan(m) for m in means.values()):
        return None, means
    best = None
    for a in ARMS:
        if a not in means:
            continue
        if best is None or ((means[a] > means[best]) if higher_is_better else (means[a] < means[best])):
            best = a
    return best, means


def side(interval, higher_is_better=False):
    """'body' when the interval of body minus comparator lies wholly on the body's side of 0, 'comparator' in the
    mirror case, None otherwise or without an interval."""
    if not interval or any(_isnan(x) for x in interval):
        return None
    lo, hi = float(interval[0]), float(interval[1])
    if higher_is_better:
        return BODY if lo > 0 else ("comparator" if hi < 0 else None)
    return BODY if hi < 0 else ("comparator" if lo > 0 else None)


def reading(sep, interval, higher_is_better=False):
    """('body better' | 'body worse' | 'no difference', beats, loses) of section 8."""
    s = side(interval, higher_is_better)
    beats = sep == BODY and s == BODY
    loses = sep == "comparator" and s == "comparator"
    return ("body better" if beats else "body worse" if loses else "no difference"), beats, loses


# ---------------------------------------------------------------- the person bootstrap
def draws(n_persons, n_boot=BOOT_N, seed=SEED):
    """(n_boot, n_persons) indices of the persons drawn with replacement in each resample (the draws of
    common.cluster_bootstrap for the same seed and person count)."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_persons), size=(int(n_boot), int(n_persons)))


def resample_line(v, present, idx):
    """(n,) the line in each resample (module docstring): the mean over the drawn slots whose person enters the line
    (a person drawn twice counts twice), NaN when no drawn person enters; an infinite or NaN person value
    propagates."""
    v = np.asarray(v, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    idx = np.asarray(idx)
    w = present[idx]
    x = np.where(w, v[idx], 0.0)
    cnt = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = x.sum(axis=1)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def percentile_interval(diffs, level=LEVEL):
    """[lo, hi] at the central `level` of the differences (module docstring); None when a difference is NaN."""
    d = np.asarray(diffs, dtype=np.float64)
    if not len(d) or np.isnan(d).any():
        return None
    a = (1.0 - level) / 2.0
    if np.isfinite(d).all():
        lo, hi = np.quantile(d, [a, 1.0 - a])
        return [float(lo), float(hi)]
    s = np.sort(d)
    out = []
    for q in (a, 1.0 - a):
        h = (len(s) - 1) * q
        i = int(np.floor(h))
        j = min(i + 1, len(s) - 1)
        t = h - i
        x, y = s[i], s[j]
        if x == y or t == 0.0:
            out.append(float(x))
        elif not np.isfinite(x):
            out.append(float(x))
        elif not np.isfinite(y):
            out.append(float(y))
        else:
            out.append(float(x + (y - x) * t))
    return out


def _seed_lines(frames, persons, how, idx):
    return np.stack([resample_line(*values_on(persons, f, how), idx) for f in frames])


def bootstrap_line(body, comp, persons, how="mean", n_boot=BOOT_N, seed=SEED, level=LEVEL, idx=None):
    """Interval (ii) of a G3 line (module docstring). body, comp: lists of row frames, one per seed; persons: the
    line's persons. Returns {'interval' [lo, hi] or None, 'n_boot', 'seed', 'n_persons', 'undefined' (resamples whose
    difference is NaN), 'diffs' (n,)}."""
    p = sorted(str(x) for x in persons)
    idx = draws(len(p), n_boot, seed) if idx is None else np.asarray(idx)
    b = _seed_lines(body, p, how, idx)
    c = _seed_lines(comp, p, how, idx)
    with np.errstate(invalid="ignore"):
        diff = b.mean(axis=0) - c.mean(axis=0)
    return {"interval": percentile_interval(diff, level), "n_boot": int(idx.shape[0]), "seed": int(seed),
            "n_persons": len(p), "undefined": int(np.isnan(diff).sum()), "diffs": diff}


def resample_rows(rows, persons, slots):
    """The rows of one resample: for each slot k, the rows of person persons[slots[k]] with person_id set to the
    slot's own id 'u<k>' (so that a person drawn twice is two persons of the statistic)."""
    import pandas as pd
    r = _frame(rows)
    by = {str(p): g for p, g in r.groupby(r["person_id"].astype(str))}
    parts = []
    for k, j in enumerate(np.asarray(slots).tolist()):
        g = by.get(str(persons[j]))
        if g is not None and len(g):
            parts.append(g.assign(person_id=f"u{k}"))
    return pd.concat(parts, ignore_index=True) if parts else r.iloc[0:0].copy()


def bootstrap_rows(body, comp, persons, statistic, n_boot=BOOT_N, seed=SEED, level=LEVEL, idx=None):
    """Interval (ii) by the general path (module docstring): statistic(frame) -> float on each seed's drawn rows.
    Same return as bootstrap_line."""
    p = sorted(str(x) for x in persons)
    idx = draws(len(p), n_boot, seed) if idx is None else np.asarray(idx)
    diff = np.empty(idx.shape[0])
    for i, slots in enumerate(idx):
        b = [statistic(resample_rows(f, p, slots)) for f in body]
        c = [statistic(resample_rows(f, p, slots)) for f in comp]
        with np.errstate(invalid="ignore"):
            diff[i] = np.mean(b) - np.mean(c)
    return {"interval": percentile_interval(diff, level), "n_boot": int(idx.shape[0]), "seed": int(seed),
            "n_persons": len(p), "undefined": int(np.isnan(diff).sum()), "diffs": diff}


# ---------------------------------------------------------------- one line, then the whole comparison
def of_line(rows, line):
    """The rows of one line (all rows when the frame has no line column)."""
    r = _frame(rows)
    return r[r["line"].astype(str) == line] if "line" in r.columns else r


def compare_line(line, persons, body, arms, unusable=None, n_boot=BOOT_N, seed=SEED, level=LEVEL, spec=None):
    """Section 8 on one line and set. body: {seed: rows}; arms: {arm: {seed: rows}} (rows of every line; the line's
    are taken); persons: the set's persons. Returns the points of every model and seed, the comparator and the seed
    means, the separation, the difference of the seed means, the interval (ii) and the reading."""
    spec = spec or LINES[line]
    how, hib = spec["how"], bool(spec["higher_is_better"])
    body = {s: of_line(r, line) for s, r in body.items()}
    arms = {a: {s: of_line(r, line) for s, r in by.items()} for a, by in arms.items()}
    pts = {BODY: {s: line_point(r, how, persons) for s, r in sorted(body.items())}}
    for a, by_seed in arms.items():
        pts[a] = {s: line_point(r, how, persons) for s, r in sorted(by_seed.items())}
    comp, means = pick_comparator({a: list(pts[a].values()) for a in arms}, hib, unusable)
    out = {"line": line, "how": how, "higher_is_better": hib, "n_persons": len(persons), "points": pts,
           "arm_means": means, "body_mean": seed_mean(pts[BODY].values()), "comparator": comp}
    if comp is None:
        out.update(separation=None, difference=None, interval=None, reading=NOT_EVALUATED, beats=False, loses=False,
                   note="no usable comparator arm")
        return out
    sep = separation(pts[BODY].values(), pts[comp].values(), hib)
    boot = bootstrap_line([body[s] for s in sorted(body)], [arms[comp][s] for s in sorted(arms[comp])], persons,
                          how, n_boot, seed, level)
    rd, beats, loses = reading(sep, boot["interval"], hib)
    with np.errstate(invalid="ignore"):
        diff = out["body_mean"] - means[comp]
    out.update(separation=sep, difference=diff, interval=boot["interval"], n_boot=boot["n_boot"], seed=boot["seed"],
               undefined_resamples=boot["undefined"], reading=rd, beats=beats, loses=loses)
    return out


def _seeds_text(seeds):
    return ", ".join(str(s) for s in sorted(seeds)) or "none"


def preconditions(body, arms, registered, unusable=None, seeds_required=SEEDS_REQUIRED, notes=None):
    """[what is missing] before a comparison may be computed (module docstring); notes {who: text} are appended to
    the matching lines (for example the third seed being OPEN)."""
    unusable = unusable or {}
    notes = notes or {}
    miss = []
    if not registered:
        miss.append(f"rule: {RULE_PATH} is not registered by the owner (draft section 1: no G12 comparison is "
                    "computed before the owner registers it; revision 2026-10-08 header)")
    if len(body) < seeds_required:
        miss.append(f"body: {len(body)} of {seeds_required} seeds with dev rows ({_seeds_text(body)})"
                    + (f"; {notes[BODY]}" if notes.get(BODY) else ""))
    usable = [a for a in ARMS if a not in unusable]
    for a in ARMS:
        if a in unusable:
            continue
        n = len(arms.get(a) or {})
        if n < seeds_required:
            miss.append(f"{a}: {n} of {seeds_required} seeds with readouts ({_seeds_text(arms.get(a) or {})})"
                        + (f"; {notes[a]}" if notes.get(a) else ""))
        extra = sorted(set(arms.get(a) or {}) - set(body))
        if extra:
            miss.append(f"{a}: seeds {_seeds_text(extra)} have no body seed of the same value ({SEED_PAIRING})")
        elif n >= seeds_required and len(body) >= seeds_required and set(arms[a]) != set(body):
            miss.append(f"{a}: seeds {_seeds_text(arms[a])} are not the body's ({_seeds_text(body)}; {SEED_PAIRING})")
    if not usable:
        miss.append("no usable baseline arm (every arm marked unusable)")
    return miss


def source_checks(provenance, sets, unusable=None):
    """[what is missing] for sources that cannot enter one comparison together. provenance: {model: {seed:
    {'moveatlas_head': commit or None, 'sets': {set name: persons}}}} of the body and the arms (a set a source does
    not state, such as the body's dev12 taken from its 14, is not checked). Every source must state one and the same
    MoveAtlas commit (the targets the rows were scored against; a changed record means the dependent reports are
    rerun), and every set a source states must hold the comparison's persons of that set. Unusable arms are left
    out, as in preconditions."""
    unusable = unusable or {}
    miss, heads = [], {}
    for model, by_seed in (provenance or {}).items():
        if model in unusable:
            continue
        for s, p in sorted((by_seed or {}).items()):
            who = f"{model} seed {s}"
            heads[who] = (p or {}).get("moveatlas_head") or None
            for name, persons in sorted(((p or {}).get("sets") or {}).items()):
                if name in sets and persons is not None and \
                        sorted(map(str, persons)) != sorted(map(str, sets[name])):
                    miss.append(f"{who}: its {name} persons ({len(persons)}) are not the comparison's {name} "
                                f"({len(sets[name])})")
    unknown = sorted(w for w, h in heads.items() if h is None)
    if unknown:
        miss.append(f"MoveAtlas: no commit stated by {', '.join(unknown)}")
    known = {w: h for w, h in heads.items() if h is not None}
    if len(set(known.values())) > 1:
        miss.append("MoveAtlas: the sources were scored against different commits ("
                    + ", ".join(f"{w} {h[:7]}" for w, h in sorted(known.items()))
                    + "); one comparison reads one commit (rerun the sources scored against another)")
    return miss


def evaluate(body, arms, sets, registered=False, unusable=None, lines=None, seeds_required=SEEDS_REQUIRED,
             n_boot=BOOT_N, seed=SEED, level=LEVEL, notes=None, provenance=None):
    """The section 8 comparison of the G3 lines (module docstring). body: {seed: rows}; arms: {arm: {seed: rows}};
    sets: {set name: persons}; registered: whether the owner registered the rule (the caller reads the registry);
    unusable: {arm: reason}; provenance: source_checks' input (None: not checked). Returns {'status', 'missing',
    'points' {set: {line: {model: {seed: point}}}}, 'lines' {set: {line: compare_line}} (status EVALUATED only), 'g3'
    {set: every line beaten} (EVALUATED only), 'verdict', 'other_lines'}."""
    unusable = dict(unusable or {})
    lines = lines or LINES
    arms = {a: dict(arms.get(a) or {}) for a in ARMS}
    pts = {}
    for sname, persons in sets.items():
        pts[sname] = {}
        for ln, spec in lines.items():
            pts[sname][ln] = {m: {s: line_point(of_line(r, ln), spec["how"], persons) for s, r in sorted(by.items())}
                              for m, by in ((BODY, body),) + tuple(arms.items()) if by}
    missing = preconditions(body, arms, registered, unusable, seeds_required, notes) + \
        source_checks(provenance, sets, unusable)
    out = {"status": NOT_EVALUATED, "missing": missing, "unusable": unusable, "seeds_required": seeds_required,
           "points": pts,
           "other_lines": {k: {"status": NOT_EVALUATED, "reason": v} for k, v in OTHER_LINES.items()},
           "constants": {"n_boot": n_boot, "seed": seed, "level": level, "arms": list(ARMS)}}
    if out["missing"]:
        out["verdict"] = NOT_EVALUATED + ": " + "; ".join(out["missing"])
        return out
    usable = {a: arms[a] for a in ARMS if a not in unusable}
    out["lines"] = {s: {ln: compare_line(ln, persons, body, usable, unusable, n_boot, seed, level, spec)
                        for ln, spec in lines.items()} for s, persons in sets.items()}
    out["g3"] = {s: {"beats_every_line": all(r["beats"] for r in d.values()),
                     "loses_a_line": any(r["loses"] for r in d.values())} for s, d in out["lines"].items()}
    out["status"] = EVALUATED
    out["verdict"] = ("premium NOT SHOWN under revision 2026-10-08 (the premium needs every primary line of G3, G8 and "
                      "G10; the G8 and G10 lines do not bind under it); demotion NOT EVALUATED (no G8 or G7 line); the G3 "
                      "rows of both sets are reported (docs/rules/G3_RULE.md item 5 decides between the 14 and the 12)")
    return out
