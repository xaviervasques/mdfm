"""G8 harness, weargait_pd line: MDS-UPDRS items 3.10 to 3.13 read from 13 IMUs (design section 9, G8: "weargait_pd
items 3.10 to 3.13 mean QWK at least 0.50, kill below 0.40"; rule docs/rules/G8_RULE.md, DRAFT until the owner approves
and registers it; docs/rules/G12_RULE.md section 7 and item 3 of its section 11). Dev only: this module has no lockbox
read path. Every function that takes persons refuses a weargait_pd lockbox person and any id outside weargait_pd (CODY
named) before a file opens (guard), as mdfm/gates/g3.py does.

Sets (rule section 2). PD persons only (the primary line; the PD-and-controls set is reported beside it, rule section
10 item 1): T19's proposed weargait_pd PD fold (reports/stage2_sources/folds_proposed.csv, its person hashes checked
against reports/stage2_sources/summary.json), the fold of the G3 PD half (scripts/eval_g3_pd_dev.py): pd_train (64
persons, the fitting set) and pd_dev (16, the scored set). pd_pool (both) carries a reported row cross-fitted over 5
person folds. Controls (the reported PD-and-controls rows): the 54 training-fold and 14 dev controls of
mdfm.data.sources.folds (g3.controls). A person enters an item's line when the item holds a grade for it in the
source's clinical table and both G3 tasks (SelfPace, HurriedPace) hold an IMU record of it. The 20 PD lockbox persons
and the 17 lockbox controls are counted from reports/lockbox/lockbox_persons.csv only; guard refuses them (with or
without the ' (control)' suffix) before any file opens, and of their clinical rows only the subject id is looked at: the
CSV reader tokenizes every row of a table, and read_items skips a row whose id is not requested on that id, before any
other cell is read or interpreted.

Items (ITEMS): MDS-UPDRS part III 3.10 gait, 3.11 freezing of gait, 3.12 postural stability, 3.13 posture, columns
MDSUPDRS_3-10 to MDSUPDRS_3-13 of "PD - Demographic+Clinical - datasetV1.csv" and "CONTROLS - Demographic+Clinical -
datasetV1.csv" (MoveAtlas datasets/imu/weargait_pd/raw/Version 1); '-' or an empty cell is no grade; a grade outside the
scale 0 to 4 raises. Covariates kept beside them: part II 2.12 (walking and balance) and 2.13 (freezing), the days
since the part III evaluation, the medication state of part III (3b).

Statistic (rule section 6). qwk: Cohen's weighted kappa with quadratic weights over integer grades, 1 - O / E with O =
sum over persons (y - p)^2 and E = (1 / N) sum over pairs (a, c) (y_a - p_c)^2 (the expected disagreement under
independent margins); undefined (NaN) when E is 0 (every grade and every prediction one value). A constant prediction
reads exactly 0 whenever the grades vary (O = E). The line: the mean over the items with a defined QWK (mean_qwk).
Interval: person-clustered percentile bootstrap (BOOT_N resamples, SEED, 95 percent): each resample draws the line's
persons with replacement, a person drawn twice counting twice, and recomputes every item's QWK on the drawn persons'
rows, then their mean (bootstrap_qwk; the closed form with draw counts as weights equals comparison.bootstrap_rows on
the drawn rows, tested). An item whose QWK is undefined in a resample leaves that resample's mean; the resamples with
such an item are counted.

Readout (rule section 4). A person's features: the frozen tokens of the model (tokenizer and encoder only,
mdfm/baselines/readouts.encode_windows: the G3 inference windows, each 0.25 s patch read from its most central window)
over every patch of its SelfPace and HurriedPace trials, pooled: mean and standard deviation per feature (person_features;
variant subject: the mean). Per item, a ridge (mdfm/eval/probes.fit_probe: features standardized by the fitting set,
the penalty chosen by 5 person folds inside the fitting set) fitted on the fitting set's graded persons and applied to the
scored persons; the continuous prediction is rounded half up and clipped to 0 to 4 (round_clip). A person whose prediction
is not finite (no token row) gets the fitting set's majority grade and is counted (filled: person-item rows; and the
persons holding one). Variant (latent or measurement) chosen per model by the person cross-fitted mean QWK on the
fitting set (probes.cross_fit; ties to latent), never on a person of the scored set it decides (pd_dev). The reported
pool row (crossfit_readout) does not choose: the script computes it for both candidates and flags the one chosen on
pd_train, persons the pool also scores.

Baselines (rule section 7), same rounding and filling: majority (the fitting set's most frequent grade, ties to the lower
grade: QWK 0 by construction), speed (a ridge on the person's mean walkway pass speed in SelfPace and in HurriedPace,
reports/weargait_targets through g3.load_targets: an instrument, not the IMU). Oracles, reported and deciding nothing:
part2 (a ridge on part II 2.12 and 2.13, the patient's own report) and others (per item, a ridge on the other three items
of the four). A quantile discretization of every probe (qmap: cut points on the raw prediction that give the fitting set's
cross-fitted predictions the fitting set's grade shares) is a reported diagnostic of the rounding (rule section 10).

Rows (ROW_COLUMNS): one row per scored person and item with its person, target grade, raw and rounded prediction, filled
flag and model, set, variant and readout names; the line and its interval are recomputed from the rows alone (line_of,
bootstrap_qwk), as docs/rules/G12_RULE.md section 8 (ii) asks of a population statistic; bootstrap_difference is the
interval of a difference of seed means over the same draws.
"""
import csv
import functools
import glob
import json
import math
import os

import numpy as np

from mdfm import paths
from mdfm.data import canonical
from mdfm.data.canonical import Refused

GATE = "G8"
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G8_RULE.md")
SOURCE = "weargait_pd"
PREFIX = "weargait:"
TASKS = ("SelfPace", "HurriedPace")             # g3.TASKS: the tasks of the G3 lines
ITEMS = ("3.10", "3.11", "3.12", "3.13")
ITEM_NAMES = {"3.10": "gait", "3.11": "freezing of gait", "3.12": "postural stability", "3.13": "posture"}
ITEM_COLUMNS = {i: "MDSUPDRS_" + i.replace(".", "-") for i in ITEMS}
PART2 = {"2.12": "MDSUPDRS_2-12", "2.13": "MDSUPDRS_2-13"}     # walking and balance; freezing (patient-reported)
DAYS_COLUMN = "Days since Part III Clinical Evaluation"
MED_COLUMN = "3b"
SCALE = (0, 4)
LINE = "weargait_items_mean_qwk"
PASS_LINE = 0.50
KILL_LINE = 0.40
HIGHER_IS_BETTER = True
SEED = 20261015                                 # bootstrap seed of this rule (G3's is 20260929)
BOOT_N = 2000
LEVEL = 0.95
CROSS_FOLDS = 5
GROUP_PD, GROUP_CONTROL = "PD", "control"
CLINICAL_FILES = {GROUP_PD: "imu/weargait_pd/raw/*/PD - Demographic+Clinical - datasetV1.csv",
                  GROUP_CONTROL: "imu/weargait_pd/raw/*/CONTROLS - Demographic+Clinical - datasetV1.csv"}
FOLDS_CSV = os.path.join(paths.TREE, "reports", "stage2_sources", "folds_proposed.csv")
STAGE2_SUMMARY = os.path.join(paths.TREE, "reports", "stage2_sources", "summary.json")
FOLD_KEY = "weargait_pd:PD"
LOCKBOX_CSV = os.path.join(paths.TREE, "reports", "lockbox", "lockbox_persons.csv")
SET_DEV, SET_TRAIN, SET_POOL = "pd_dev", "pd_train", "pd_pool"
SET_ALL_DEV, SET_ALL_TRAIN, SET_ALL_POOL = "all_dev", "all_train", "all_pool"   # PD and controls (reported)
CANDIDATES = ("latent", "measurement")
VARIANTS = ("latent", "measurement", "stem", "subject")
POOL = ("mean", "std")
ROW_COLUMNS = ["model", "set", "readout", "variant", "chosen", "item", "person_id", "group", "target", "raw",
               "prediction", "filled"]


# ---------------------------------------------------------------- persons
@functools.lru_cache(maxsize=1)
def lockbox_rows():
    """The weargait_pd rows (person_id, group) of reports/lockbox/lockbox_persons.csv (the split file; through
    mdfm.gates.g3, whose reader checks the file against reports/lockbox/lockbox.sha256)."""
    from mdfm.gates import g3
    return tuple((r["person_id"], r["group"]) for r in g3._lockbox_rows())


def lockbox_persons():
    return sorted(p for p, _ in lockbox_rows())


def lockbox_counts():
    """{group: number of weargait_pd lockbox persons} from the split file only (no clinical row is read)."""
    out = {}
    for _, g in lockbox_rows():
        out[g] = out.get(g, 0) + 1
    return out


def lockbox_index_counts(index=None, tasks=TASKS):
    """{group: {'persons', 'with_both_tasks'}} of the weargait_pd lockbox persons in the canonical index's metadata
    (persons with an IMU index row, and with one of every G3 task); no record is opened and no clinical row is read."""
    df = canonical.index() if index is None else index
    sel = df[(df["source"] == SOURCE) & (df["modality"] == "imu") & df["lockbox"].astype(bool)]
    out = {}
    for g, x in sel.groupby("group"):
        keys = x.groupby("person_id")["key"].apply(lambda s: set(map(str, s)))
        out[str(g)] = {"persons": int(len(keys)), "with_both_tasks": int(sum(set(tasks) <= k for k in keys))}
    return out


CONTROL_SUFFIX = " (control)"


def bare_id(p):
    """The id without the source file names' ' (control)' suffix that three control ids keep in the records (MoveAtlas
    datacard of weargait_pd, known issues): read_items matches a table row to a requested id on this form, so the
    lockbox refusal compares this form too."""
    return str(p).replace(CONTROL_SUFFIX, "").strip()


def guard(persons):
    """The set of requested ids, refused (Refused) before any file opens when one is outside weargait_pd (CODY named) or
    is a weargait_pd lockbox person, with or without the ' (control)' suffix on either side (bare_id): G8 has no lockbox
    read path in this harness (rule section 9)."""
    ps = {str(p) for p in persons}
    other = sorted(p for p in ps if not p.startswith(PREFIX))
    if other:
        cody = [p for p in other if canonical.is_cody(p) or p.split(":", 1)[0] in canonical.CODY_NAMESPACES]
        raise Refused(f"G8 reads weargait_pd persons only; {len(other)} other ids"
                      + (f", {len(cody)} of them CODY (evaluation-only, never a G8 input)" if cody else ""))
    locked_bare = {bare_id(p) for p in lockbox_persons()}
    locked = sorted(p for p in ps if bare_id(p) in locked_bare)
    if locked:
        raise Refused(f"G8: {len(locked)} weargait_pd lockbox persons requested; the dev harness never reads the lockbox")
    return ps


def pd_folds(path=None, summary=None):
    """(train, dev, info) of T19's proposed weargait_pd PD fold (the G3 PD half's: scripts/eval_g3_pd_dev.t19_folds):
    the person ids of folds_proposed.csv (group PD) by fold, their SHA-256 (sources.person_hash) equal to those of
    summary.json, the two folds disjoint, no lockbox person (Refused otherwise)."""
    from mdfm.data import sources
    path, summary = path or FOLDS_CSV, summary or STAGE2_SUMMARY
    folds = {"dev": [], "train": []}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["source"] == SOURCE and r["group"] == GROUP_PD:
                if r["fold"] not in folds:
                    raise ValueError(f"fold {r['fold']!r} in {path}")
                folds[r["fold"]].append(r["person_id"])
    with open(summary, encoding="utf-8") as f:
        rec = json.load(f)["folds"][FOLD_KEY]
    train, dev = sorted(folds["train"]), sorted(folds["dev"])
    for name, ids in (("dev", dev), ("train", train)):
        if sources.person_hash(ids) != rec[f"sha256_{name}"]:
            raise ValueError(f"T19 {name} fold of {path} does not match its SHA-256 in {summary}")
    if set(dev) & set(train):
        raise ValueError("T19 dev and train folds overlap")
    guard(set(dev) | set(train))
    info = {"path": paths_rel(path), "summary": paths_rel(summary), "train": len(train), "dev": len(dev),
            "sha256_train": rec["sha256_train"], "sha256_dev": rec["sha256_dev"], "seed": rec.get("seed")}
    return train, dev, info


def control_folds():
    """(train, dev) of the non-lockbox weargait_pd controls (g3.controls: 54 and 14)."""
    from mdfm.gates import g3
    train, dev = g3.controls("train"), g3.controls("dev")
    guard(set(train) | set(dev))
    return train, dev


def paths_rel(p):
    try:
        r = os.path.relpath(os.path.abspath(p), paths.TREE)
    except ValueError:
        return str(p).replace("\\", "/")
    return r.replace("\\", "/")


def record_persons(persons, index=None, tasks=TASKS):
    """{person: sorted G3 tasks with an IMU record} from the canonical index (no record opened), refused like guard;
    a requested person with a lockbox index row raises."""
    want = guard(persons)
    df = canonical.index() if index is None else index
    sel = df[(df["source"] == SOURCE) & (df["modality"] == "imu") & df["person_id"].astype(str).isin(want)
             & df["key"].isin(tasks)]
    if sel["lockbox"].astype(bool).any():
        raise Refused("a lockbox IMU index row among the requested persons")
    out = {p: [] for p in sorted(want)}
    for p, k in zip(sel["person_id"].astype(str), sel["key"].astype(str)):
        if k not in out[p]:
            out[p].append(k)
    return {p: sorted(v) for p, v in out.items()}


# ---------------------------------------------------------------- the clinical table
def _grade(v, where):
    s = str(v).strip()
    if s in ("", "-"):
        return float("nan")
    x = float(s)
    if x != int(x) or not SCALE[0] <= x <= SCALE[1]:
        raise ValueError(f"{where}: grade {s!r} outside the item scale {SCALE}")
    return float(x)


def _number(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def clinical_file(group, root=None):
    root = (root or paths.MOVEATLAS).replace("\\", "/").rstrip("/")
    files = sorted(glob.glob(f"{root}/datasets/{CLINICAL_FILES[group]}"))
    if len(files) != 1:
        raise FileNotFoundError(f"{CLINICAL_FILES[group]}: {len(files)} files under {root}")
    return files[0]


def read_items(persons, groups=(GROUP_PD,), root=None, files=None):
    """DataFrame (person_id, group, the four items as floats with NaN for no grade, part II 2.12 and 2.13, days since part
    III, medication state of part III) of the requested persons. guard first (a lockbox or non-weargait id is refused
    before any file opens, suffix or not); each table is tokenized row by row by the CSV reader and a row whose subject
    id (bare_id, on both sides) is not requested is skipped before any other cell is read or interpreted; a requested
    person found twice raises. files: {group: path} to read instead of the MoveAtlas tables (tests)."""
    import pandas as pd
    want = guard(persons)
    # three control ids keep the source file names' ' (control)' suffix in the records, the table has the bare id
    # (MoveAtlas datacard of weargait_pd, known issues): a table row answers the requested id with or without it
    by_bare = {}
    for p in want:
        by_bare.setdefault(bare_id(p), []).append(p)
    if any(len(v) > 1 for v in by_bare.values()):
        raise ValueError(f"requested ids equal up to the ' (control)' suffix: {[v for v in by_bare.values() if len(v) > 1]}")
    rows, seen = [], set()
    for group in groups:
        path = (files or {}).get(group) or clinical_file(group, root)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            cols = None
            for r in reader:
                if cols is None:
                    if r and r[0].strip() == "Subject ID":
                        cols = [c.strip() for c in r]
                        at = {c: i for i, c in enumerate(cols) if c}
                        missing = [c for c in ITEM_COLUMNS.values() if c not in at]
                        if missing:
                            raise ValueError(f"{path}: no column {missing}")
                    continue
                if not r or not r[0].strip():
                    continue
                bare = PREFIX + bare_id(r[0])
                if bare not in by_bare:
                    continue
                pid = by_bare[bare][0]
                if pid in seen:
                    raise ValueError(f"{pid} found twice in the clinical tables")
                seen.add(pid)
                cell = lambda c: r[at[c]] if c in at and at[c] < len(r) else ""  # noqa: E731
                row = {"person_id": pid, "group": group}
                for i, c in ITEM_COLUMNS.items():
                    row[i] = _grade(cell(c), f"{pid} {c}")
                for i, c in PART2.items():
                    row[i] = _grade(cell(c), f"{pid} {c}")
                row["days_since_part3"] = _number(cell(DAYS_COLUMN))
                row["med_state"] = str(cell(MED_COLUMN)).strip()
                rows.append(row)
        if cols is None:
            raise ValueError(f"{path}: no header row 'Subject ID'")
    df = pd.DataFrame(rows, columns=["person_id", "group", *ITEMS, *PART2, "days_since_part3", "med_state"])
    return df.sort_values("person_id", kind="stable").reset_index(drop=True)


def med_label(v):
    """The medication state of part III, normalized: ON, OFF or the source's text ('' for none)."""
    s = str(v).strip()
    return {"on": "ON", "off": "OFF", "-": ""}.get(s.lower(), s)


# ---------------------------------------------------------------- statistic
def round_clip(x, lo=SCALE[0], hi=SCALE[1]):
    """Round half up to an integer grade and clip to [lo, hi]; NaN stays NaN."""
    a = np.asarray(x, dtype=np.float64)
    out = np.clip(np.floor(a + 0.5), lo, hi)
    return np.where(np.isfinite(a), out, np.nan)


def _check_grades(a, what):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all() or (a != np.round(a)).any() or (a < SCALE[0]).any() or (a > SCALE[1]).any():
        raise ValueError(f"{what}: integer grades within {SCALE} expected")
    return a


def qwk(y, p):
    """Quadratic weighted kappa of integer grades (module docstring); NaN when the expected disagreement is 0 or there
    is no pair."""
    y = _check_grades(y, "targets")
    p = _check_grades(p, "predictions")
    if y.shape != p.shape:
        raise ValueError(f"{y.shape} targets for {p.shape} predictions")
    if not len(y):
        return float("nan")
    n = float(len(y))
    o = float(((y - p) ** 2).sum())
    e = float((y ** 2).sum() - 2.0 * y.sum() * p.sum() / n + (p ** 2).sum())
    return float("nan") if e <= 1e-12 else 1.0 - o / e


def item_keys(col):
    """The item column as ITEMS strings: a CSV reader turns '3.10' into the number 3.1, so every value is read back as a
    number with two decimals."""
    return col.map(lambda v: f"{float(v):.2f}")


def qwk_by_item(rows):
    """{item: QWK} over the rows of each item (ROW_COLUMNS frame: target, prediction)."""
    out = {}
    keys = item_keys(rows["item"])
    for item in ITEMS:
        g = rows[keys == item]
        out[item] = qwk(g["target"].to_numpy(np.float64), g["prediction"].to_numpy(np.float64)) if len(g) else float("nan")
    return out


def mean_qwk(rows):
    """(mean over the items with a defined QWK, {item: QWK}); NaN mean when none is defined."""
    by = qwk_by_item(rows)
    v = [x for x in by.values() if np.isfinite(x)]
    return (float(np.mean(v)) if v else float("nan")), by


def row_statistic(rows):
    """The line on any rows frame (for mdfm.baselines.comparison.bootstrap_rows): mean_qwk's mean."""
    return mean_qwk(rows)[0]


def _item_arrays(rows, persons):
    """{item: (y (P,), p (P,), present (P,) bool)} over `persons` (sorted ids), one row per person and item."""
    idx = {p: i for i, p in enumerate(persons)}
    out = {}
    keys = item_keys(rows["item"])
    for item in ITEMS:
        g = rows[keys == item]
        y = np.zeros(len(persons))
        p = np.zeros(len(persons))
        present = np.zeros(len(persons), bool)
        for pid, t, q in zip(g["person_id"].astype(str), g["target"].to_numpy(np.float64),
                             g["prediction"].to_numpy(np.float64)):
            j = idx.get(pid)
            if j is None:
                continue
            if present[j]:
                raise ValueError(f"{pid} holds two rows of item {item}")
            y[j], p[j], present[j] = t, q, True
        _check_grades(y[present], f"targets of {item}")
        _check_grades(p[present], f"predictions of {item}")
        out[item] = (y, p, present)
    return out


def draws(n_persons, n_boot=BOOT_N, seed=SEED):
    """(n_boot, n_persons) indices drawn with replacement (the draws of mdfm.baselines.comparison.draws)."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_persons), size=(int(n_boot), int(n_persons)))


def resample_qwk(rows, persons, idx):
    """((n,) mean QWK per resample, (n,) count of items undefined in it, {item: (n,) QWK}) on the drawn persons, by
    the closed form with the draw counts as weights (module docstring)."""
    persons = sorted(str(p) for p in persons)
    idx = np.asarray(idx)
    W = np.zeros((idx.shape[0], len(persons)))
    np.add.at(W, (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), idx.reshape(-1)), 1.0)
    per = {}
    for item, (y, p, present) in _item_arrays(rows, persons).items():
        w = W * present[None]
        n = w.sum(axis=1)
        o = w @ ((y - p) ** 2)
        sy, sp = w @ y, w @ p
        e = w @ (y ** 2) - 2.0 * sy * sp / np.maximum(n, 1e-300) + w @ (p ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where((n > 0) & (e > 1e-12), 1.0 - o / np.where(e > 1e-12, e, 1.0), np.nan)
        per[item] = k
    K = np.stack([per[i] for i in ITEMS])
    undefined = np.isnan(K).sum(axis=0)
    defined = len(ITEMS) - undefined
    total = np.where(np.isnan(K), 0.0, K).sum(axis=0)
    mean = np.where(defined > 0, total / np.maximum(defined, 1), np.nan)
    return mean, undefined, per


def percentile(v, level=LEVEL):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return [float(lo), float(hi)]


def bootstrap_qwk(rows, persons=None, n_boot=BOOT_N, seed=SEED, level=LEVEL):
    """{'interval', 'n_boot', 'seed', 'n_persons', 'resamples_with_undefined_item', 'resamples_undefined'} of the line
    on rows (persons: the line's persons, default the rows' persons)."""
    ps = sorted({str(p) for p in rows["person_id"]}) if persons is None else sorted(str(p) for p in persons)
    if not ps:
        return {"interval": None, "n_boot": int(n_boot), "seed": int(seed), "n_persons": 0,
                "resamples_with_undefined_item": 0, "resamples_undefined": 0}
    mean, undefined, _ = resample_qwk(rows, ps, draws(len(ps), n_boot, seed))
    return {"interval": percentile(mean, level), "n_boot": int(n_boot), "seed": int(seed), "n_persons": len(ps),
            "resamples_with_undefined_item": int((undefined > 0).sum()), "resamples_undefined": int(np.isnan(mean).sum())}


def bootstrap_difference(body, comp, persons, n_boot=BOOT_N, seed=SEED, level=LEVEL):
    """Interval (ii) of docs/rules/G12_RULE.md section 8 for this line: body, comp lists of rows frames (one per seed);
    each resample recomputes every seed's line on the drawn persons, then the difference of the two seed means.
    {'interval', 'diffs', 'undefined'} (a resample whose difference is NaN is left out and counted)."""
    ps = sorted(str(p) for p in persons)
    idx = draws(len(ps), n_boot, seed)
    b = np.stack([resample_qwk(r, ps, idx)[0] for r in body]).mean(axis=0)
    c = np.stack([resample_qwk(r, ps, idx)[0] for r in comp]).mean(axis=0)
    d = b - c
    return {"interval": percentile(d, level), "diffs": d, "undefined": int(np.isnan(d).sum()), "n_boot": int(n_boot),
            "seed": int(seed), "n_persons": len(ps)}


def line_of(rows, persons=None, n_boot=BOOT_N, seed=SEED):
    """{'point', 'per_item', 'n_persons', 'n_persons_by_item', 'filled', 'filled_persons', 'interval' and the bootstrap's
    counts} of the rows (one model, set, readout and variant). 'filled' counts person-item rows given the majority
    grade, 'filled_persons' the persons holding at least one such row."""
    r = rows if persons is None else rows[rows["person_id"].astype(str).isin(set(map(str, persons)))]
    point, by = mean_qwk(r)
    boot = bootstrap_qwk(r, persons, n_boot, seed)
    keys = item_keys(r["item"])
    n_by = {i: int((keys == i).sum()) for i in ITEMS}
    filled = r["filled"].astype(bool)
    return {"point": point, "per_item": by, "n_persons": int(r["person_id"].nunique()), "n_persons_by_item": n_by,
            "filled": int(filled.sum()), "filled_persons": int(r.loc[filled, "person_id"].nunique()),
            "interval": boot["interval"], "bootstrap": {k: v for k, v in boot.items() if k != "interval"}}


def spearman(a, b):
    """Spearman correlation (average ranks); NaN when a side is constant or has fewer than 3 values."""
    import pandas as pd
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- features
def person_features(encode, records, persons, log=print):
    """Pooled frozen-token features of each person (module docstring). encode: windows -> [per-window dict]
    (readouts.encode_windows bound to a model); records: iterable of (trial_id, model input record). Returns {'persons'
    (sorted), 'X' {variant: (P, D)} (NaN rows for a person without a token row), 'n_rows' (P,), 'counts'}."""
    import time

    from mdfm.baselines import readouts as RO
    from mdfm.eval import infer as I
    persons = sorted(guard(persons))
    rows = {p: {v: [] for v in VARIANTS} for p in persons}
    n_rec = n_win = 0
    seen = set()
    t0 = time.time()
    for tid, rec in records:
        RO.check_record(tid, rec, persons)
        pid = str(rec["person_id"])
        seen.add(tid)
        n_rec += 1
        wins, _, _ = I.record_windows(rec)
        if not wins:
            continue
        n_win += len(wins)
        tr = RO.trial_rows(encode(wins), wins)
        for v in VARIANTS:
            rows[pid][v].append(np.asarray(tr[v], dtype=np.float64))
        log(f"  {tid}: {len(wins)} windows, {len(tr['start'])} token rows ({time.time() - t0:.0f} s)", flush=True)
    X, n_rows = {}, np.zeros(len(persons), np.int64)
    dims = {}
    for v in VARIANTS:
        out = []
        for j, p in enumerate(persons):
            parts = rows[p][v]
            if not parts:
                out.append(None)
                continue
            m = np.concatenate(parts)
            ok = np.isfinite(m).all(axis=1)
            if v == "latent":
                n_rows[j] = int(ok.sum())
            if not ok.any():
                out.append(None)
                continue
            m = m[ok]
            out.append(m.mean(axis=0) if v == "subject" else np.concatenate([m.mean(axis=0), m.std(axis=0)]))
            dims[v] = out[-1].shape[0]
        X[v] = np.stack([x if x is not None else np.full(dims.get(v, 0), np.nan) for x in out]) if persons else \
            np.zeros((0, 0))
    counts = {"records": n_rec, "windows": n_win, "persons": len(persons),
              "persons_without_rows": [p for p, n in zip(persons, n_rows) if n == 0], "trials": sorted(seen)}
    return {"persons": persons, "X": X, "n_rows": n_rows, "counts": counts}


def save_features(path, feats, meta):
    """person_features output and meta into one npz (no pickle)."""
    arrs = {"meta": np.asarray(json.dumps(meta, default=_json_default)), "persons": np.asarray(feats["persons"]),
            "n_rows": feats["n_rows"], "counts": np.asarray(json.dumps(feats["counts"], default=_json_default))}
    for v, x in feats["X"].items():
        arrs[f"X__{v}"] = np.asarray(x, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(paths.assert_not_moveatlas(tmp), **arrs)
    os.replace(tmp, path)
    return path


def load_features(path):
    z = np.load(path, allow_pickle=False)
    feats = {"persons": [str(p) for p in z["persons"]], "n_rows": z["n_rows"], "counts": json.loads(str(z["counts"])),
             "X": {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("X__")}}
    return feats, json.loads(str(z["meta"]))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ---------------------------------------------------------------- probes and baselines
def majority(grades):
    """The most frequent grade of the finite grades (ties to the lower grade); NaN without any."""
    g = np.asarray(grades, dtype=np.float64)
    g = g[np.isfinite(g)]
    if not len(g):
        return float("nan")
    vals, cnt = np.unique(g, return_counts=True)
    return float(vals[np.argmax(cnt)])


def _fit_predict(X_fit, y_fit, p_fit, X_score):
    """Continuous ridge predictions (probes.fit_probe on the graded fitting rows) for X_score, and the chosen alpha."""
    from mdfm.eval import probes
    ok = np.isfinite(y_fit) & np.isfinite(X_fit).all(axis=1)
    m, sel = probes.fit_probe(X_fit[ok], y_fit[ok], p_fit[ok])
    return m.predict(X_score), sel["alpha"]


def qmap_cuts(raw_cf, y_fit):
    """Cut points c_0 <= ... <= c_3 on raw predictions that give the fitting set's cross-fitted raw predictions the
    fitting set's grade shares (the quantile of raw_cf at each cumulative share of the grades 0 to 3)."""
    ok = np.isfinite(raw_cf) & np.isfinite(y_fit)
    r, y = np.asarray(raw_cf)[ok], np.asarray(y_fit)[ok]
    shares = np.array([(y <= g).mean() for g in range(SCALE[0], SCALE[1])])
    return np.quantile(r, np.clip(shares, 0.0, 1.0))


def qmap_apply(raw, cuts):
    raw = np.asarray(raw, dtype=np.float64)
    g = (raw[:, None] > np.asarray(cuts)[None]).sum(axis=1).astype(np.float64) + SCALE[0]
    return np.where(np.isfinite(raw), g, np.nan)


def item_probe(X_fit, y_fit, p_fit, X_score, k=CROSS_FOLDS, seed=None):
    """One item: {'raw' (score rows), 'alpha', 'majority', 'raw_cf' (fitting rows, person cross-fitted), 'cuts'}. The
    fitting rows without a grade enter neither the fit nor the cross-fit."""
    from mdfm.eval import probes
    seed = probes.SEED if seed is None else seed
    y_fit = np.asarray(y_fit, dtype=np.float64)
    p_fit = np.asarray(p_fit).astype(str)
    X_fit = np.asarray(X_fit, dtype=np.float64)
    raw, alpha = _fit_predict(X_fit, y_fit, p_fit, np.asarray(X_score, dtype=np.float64))
    g = np.isfinite(y_fit)
    raw_cf = np.full(len(y_fit), np.nan)
    oof, _ = probes.cross_fit(X_fit[g], y_fit[g], p_fit[g], k=k, seed=seed)
    raw_cf[g] = oof
    return {"raw": raw, "alpha": alpha, "majority": majority(y_fit), "raw_cf": raw_cf,
            "cuts": qmap_cuts(raw_cf, y_fit)}


def grades_of(raw, fill, readout="round"):
    """(grades, filled) of raw predictions: rounded and clipped (readout 'round') or quantile-mapped ('qmap' with
    fill = (majority, cuts)); a non-finite raw value gets the majority grade."""
    maj, cuts = fill if isinstance(fill, tuple) else (fill, None)
    g = round_clip(raw) if readout == "round" else qmap_apply(raw, cuts)
    filled = ~np.isfinite(g)
    return np.where(filled, maj, g), filled


def make_rows(model, set_name, readout, variant, item, persons, groups, targets, raw, grades, filled, chosen=False):
    import pandas as pd
    ok = np.isfinite(np.asarray(targets, dtype=np.float64))
    return pd.DataFrame({"model": model, "set": set_name, "readout": readout, "variant": variant, "chosen": bool(chosen),
                         "item": item, "person_id": np.asarray(persons)[ok].astype(str),
                         "group": np.asarray(groups)[ok].astype(str), "target": np.asarray(targets)[ok],
                         "raw": np.asarray(raw, dtype=np.float64)[ok], "prediction": np.asarray(grades)[ok],
                         "filled": np.asarray(filled, dtype=bool)[ok]})[ROW_COLUMNS]


def run_readout(X, persons, groups, items, fit_persons, score_persons, model, set_name, variant, seed=None):
    """Rows of one feature matrix X (P, D) over persons (P,), or of {item: (P, D)} (an item-specific predictor such as
    the oracle 'others'), on one fitting set and one scored set, every item, both readouts (round, qmap), with the fit's
    diagnostics: (rows frame, {item: {'alpha', 'majority', 'cuts', 'spearman_raw', 'cf_rows'}}). fit_persons and
    score_persons are disjoint (probes.check_disjoint). cf_rows: the rounded cross-fitted rows of the fitting set (the
    variant choice reads them)."""
    import pandas as pd

    from mdfm.eval import probes
    persons = np.asarray(persons).astype(str)
    groups = np.asarray(groups).astype(str)
    fi = np.flatnonzero(np.isin(persons, list(fit_persons)))
    si = np.flatnonzero(np.isin(persons, list(score_persons)))
    probes.check_disjoint(persons[fi], persons[si])
    parts, diag = [], {}
    for item in ITEMS:
        y = np.asarray(items[item], dtype=np.float64)
        Xi = np.asarray(X[item] if isinstance(X, dict) else X, dtype=np.float64)
        res = item_probe(Xi[fi], y[fi], persons[fi], Xi[si], seed=seed)
        for readout, fill in (("round", res["majority"]), ("qmap", (res["majority"], res["cuts"]))):
            g, filled = grades_of(res["raw"], fill, readout)
            parts.append(make_rows(model, set_name, readout, variant, item, persons[si], groups[si], y[si], res["raw"], g,
                                   filled))
        gcf, fcf = grades_of(res["raw_cf"], res["majority"], "round")
        cf_rows = make_rows(model, "crossfit_fit", "round", variant, item, persons[fi], groups[fi], y[fi], res["raw_cf"],
                            gcf, fcf)
        diag[item] = {"alpha": res["alpha"], "majority": res["majority"], "cuts": res["cuts"].tolist(),
                      "spearman_raw": spearman(res["raw"], y[si]), "cf_rows": cf_rows}
    return pd.concat(parts, ignore_index=True), diag


def crossfit_readout(X, persons, groups, items, pool_persons, model, set_name, variant, k=CROSS_FOLDS, seed=None):
    """Rows of the pool row (rule section 2): outer person folds over the pool (probes.person_folds, seed), each fold's
    persons read by a probe fitted on the other folds (run_readout), so that no person informs its own prediction. The
    variant is given, not chosen here: a variant chosen on pd_train was chosen on persons this row scores."""
    import pandas as pd

    from mdfm.eval import probes
    seed = probes.SEED if seed is None else seed
    pool = np.asarray(sorted(set(map(str, pool_persons))))
    folds = probes.person_folds(pool, k, seed)
    parts = []
    for f in range(k):
        score, fit = set(pool[folds == f]), set(pool[folds != f])
        r, _ = run_readout(X, persons, groups, items, fit, score, model, set_name, variant, seed + 1)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def choose_variant(diags, candidates=CANDIDATES):
    """(variant, {variant: cross-fitted mean QWK}) by the fitting set's rounded cross-fitted rows; the higher wins, a
    NaN never over a number, ties to the earlier candidate."""
    import pandas as pd
    scores = {}
    for v in candidates:
        if v not in diags:
            continue
        rows = pd.concat([d["cf_rows"] for d in diags[v].values()], ignore_index=True)
        scores[v] = mean_qwk(rows)[0]
    best = None
    for v in candidates:
        if v not in scores:
            continue
        x = scores[v]
        if best is None or (np.isfinite(x) and (not np.isfinite(scores[best]) or x > scores[best])):
            best = v
    return best, scores


def gate_result(line, model, set_name, readout, variant, trivial=0.0, rule_sha256="", checkpoint_sha256="",
                train_persons_sha256="", extra=None):
    """A common.GateResult of the line (rule section 6): pass at least PASS_LINE, kill below KILL_LINE, on the point."""
    from mdfm.gates import common
    v = common.verdict(line["point"], PASS_LINE, KILL_LINE, higher_is_better=True)
    ex = {"model": model, "set": set_name, "readout": readout, "variant": variant, "per_item": line["per_item"],
          "n_persons_by_item": line["n_persons_by_item"], "filled": line["filled"],
          "filled_persons": line.get("filled_persons"), "bootstrap": line["bootstrap"]}
    ex.update(extra or {})
    iv = tuple(line["interval"]) if line["interval"] else None
    return common.GateResult(GATE, LINE, line["point"], iv, PASS_LINE, KILL_LINE, v, float(trivial), rule_sha256,
                             checkpoint_sha256, train_persons_sha256, line["n_persons"], "QWK", True, ex)
