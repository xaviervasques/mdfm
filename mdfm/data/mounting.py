"""Tilt canonicalization of IMU records and the mounting probe of stage 2 (spec
docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.5, departure 6 and risk 4; task S2.6).

Turn (turn_record): for a source whose sensor mounting follows the cohort (clinical_gait_multipatho,
imu_inthewild_iprognosis; SourceSpec.tilt of mdfm.data.sources), each site of each record is turned once, at store
build, so that its mean acceleration points along the site's canonical up axis. The mean is taken over the record's
valid samples of that site (sample_valid, finite acceleration). The canonical up axis of a site is the up_sensor row
of the generic mount family of reports/mount_priors/priors.json (identity mount: the segment's up in the renderer's
standing posture, scripts/build_mount_priors.py), so a synthetic body rendered with the generic family and a turned
real record share their up axis. The turn is the smallest rotation taking the unit mean acceleration m onto u
(Rodrigues about m x u; for m opposite u, half a turn about the horizontal axis of u nearest to the sensor's x
axis); it acts on the acceleration, the angular rate and the magnetometer triplets of every sample alike, so the
record stays one rigid sensor frame. A site with no valid acceleration sample, or whose mean acceleration is shorter
than MIN_MEAN_G [cfg], keeps the identity (counted). A gravity-free accelerometer (acc_content linear_no_gravity:
pads) cannot show the tilt and is refused. The rotations (sensor to turned frame, one 3 x 3 per site in record
order) sit in the stream meta under 'tilt_rotation' (mdfm.data.windows.META_KEYS); the encoder never reads them.

What the turn leaves (risk 4): it fixes the up axis only. The sensor's heading about that axis, and with it the
axes of the horizontal acceleration and of the angular rate, keep the mounting. heading_features() measures them:
per site, the cosine and sine of twice the angle, in the horizontal plane of the turned frame (the plane normal to
u, with the basis horizontal_basis(u)), of the first principal axis of the horizontal acceleration and of the
angular rate (an axis has no sign, hence twice the angle). Measured in the sensor frame before the turn, in the
plane normal to the mean acceleration with the basis the turn carries back (R^T e1, R^T e2), the same angles come
out, by construction: the turn is a rotation. So the probe's arm before canonicalization holds the same heading
features plus the mean-acceleration direction of each site in the sensor frame (what the turn removes), and the arm
after holds the heading features alone.

Probe (probe): a person-grouped 5-fold linear classifier (standardized features, multinomial logistic regression,
class weights balanced [cfg]) of a record's group from its features; balanced accuracy of the pooled out-of-fold
predictions, and the same classifier on group-permuted persons (each person keeps its records, the persons' groups
are shuffled among the persons) as chance. Folds: StratifiedGroupKFold on the persons. No threshold: the numbers
are written for the owner (spec section 7, S2.6 and S2.11).

Wider readings (WIDER_ARMS, added after the review of S2.6; not the spec's probe): the heading angles are a narrow
view of what the turn leaves, so the same classifier also reads, per site, the 3 x 3 covariance of the acceleration
and of the angular rate divided by its trace (covariance(): the shape of the spread without its size, 6 numbers each),
in the sensor frame with the mean-acceleration direction (covariance_before) and in the turned frame
(covariance_after), and the presence flags alone (presence: which sites a record holds). These covariances carry the
gait's shape as well as the mounting.

This module imports no torch and reads no record.
"""
import functools
import hashlib
import json
import math
import os

import numpy as np

from mdfm import paths

PRIORS_PATH = os.path.join(paths.TREE, "reports", "mount_priors", "priors.json")
UP_FAMILY = "generic"          # the mount family whose up_sensor is the canonical up axis
MIN_MEAN_G = 0.5               # [cfg] a site whose mean acceleration is shorter keeps the identity
FOLDS = 5                      # [cfg] person-grouped folds of the probe
PERMUTATIONS = 20              # [cfg] group-permuted repeats of the probe (chance)
C_REG = 1.0                    # [cfg] inverse regularization of the logistic regression
SEED = 20261007                # [cfg] folds and permutations of the probe
ACC, GYRO, MAG = slice(0, 3), slice(3, 6), slice(6, 9)
COV_ENTRIES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))      # the 6 distinct entries of a 3 x 3 covariance
KINDS = ("heading", "covariance", "presence")                        # feature_matrix kinds
# the wider readings of the probe (module docstring): arm name -> (feature_matrix kind, before)
WIDER_ARMS = {"covariance_before": ("covariance", True), "covariance_after": ("covariance", False),
              "presence": ("presence", False)}


# ---------------------------------------------------------------- canonical up axes
@functools.lru_cache(maxsize=None)
def _priors(path=PRIORS_PATH):
    with open(path, "rb") as f:
        raw = f.read()
    doc = json.loads(raw.decode("utf-8"))
    return doc, hashlib.sha256(raw).hexdigest()


def priors_sha256(path=PRIORS_PATH):
    """SHA-256 of the priors file the canonical up axes come from."""
    return _priors(path)[1]


def sites(path=PRIORS_PATH):
    """Site names of the 14-site taxonomy in id order (the priors file's, equal to factory.imu_render.SITES)."""
    return tuple(_priors(path)[0]["sites"])


def canonical_up(site_id, path=PRIORS_PATH):
    """(3,) float64 unit canonical up axis of a site id (module docstring)."""
    up = np.asarray(_priors(path)[0]["families"][UP_FAMILY]["up_sensor"], dtype=np.float64)
    if not 0 <= int(site_id) < len(up):
        raise ValueError(f"site id {site_id} outside the {len(up)} sites")
    u = up[int(site_id)]
    return u / np.linalg.norm(u)


# ---------------------------------------------------------------- rotations
def _horizontal_from(u, axis):
    """Unit component of `axis` normal to u."""
    h = axis - np.dot(axis, u) * u
    return h / np.linalg.norm(h)


def horizontal_basis(u):
    """(e1, e2): an orthonormal basis of the plane normal to the unit vector u, e1 the component of the coordinate
    axis least aligned with u, e2 = u x e1 (so e1, e2, u is right-handed)."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    e1 = _horizontal_from(u, np.eye(3)[int(np.argmin(np.abs(u)))])
    return e1, np.cross(u, e1)


def turn_rotation(m, u):
    """3 x 3 rotation R (float64, det +1) of least angle with R m/|m| = u/|u| (module docstring)."""
    m = np.asarray(m, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a, b = m / np.linalg.norm(m), u / np.linalg.norm(u)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        k = _horizontal_from(b, np.eye(3)[0] if abs(b[0]) < 0.9 else np.eye(3)[1])     # half a turn about k
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def angle_deg(a, b):
    """Angle in degrees between two vectors."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))     # exact near 0 and 180


def mean_acc(signal, channel_mask, sample_valid=None):
    """(G, 3) mean acceleration per site over its valid samples (sample_valid and every acceleration channel finite)
    and (G,) the count of those samples; a site without an acceleration channel counts 0."""
    x = np.asarray(signal, dtype=np.float64)
    cm = np.asarray(channel_mask, dtype=bool)
    T, G = x.shape[:2]
    ok = np.ones((T, G), bool) if sample_valid is None else np.asarray(sample_valid, dtype=bool).copy()
    ok &= np.isfinite(x[..., ACC]).all(axis=2)
    ok &= cm[:, ACC].all(axis=1)[None]
    n = ok.sum(axis=0)
    s = np.where(ok[..., None], x[..., ACC], 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n[:, None] > 0, s / np.maximum(n, 1)[:, None], np.nan), n


def turn_record(record):
    """(record, rotations, info): a copy of a canonical IMU record (signal (T, G, 9), location_ids, channel_mask,
    sample_valid) with each site turned (module docstring), its (G, 3, 3) rotations and the counts {sites, turned,
    identity_no_acc, identity_short_mean, angle_max_deg (the largest angle between a turned site's mean
    acceleration and its up axis, after the turn)}. The input record is not changed."""
    if (record.get("acc_content") or "with_gravity") == "linear_no_gravity":
        raise ValueError("a gravity-free accelerometer cannot show its tilt: no tilt canonicalization")
    x = np.asarray(record["signal"], dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 9:
        raise ValueError(f"imu signal of shape {x.shape}, expected (T, G, 9)")
    ids = [int(i) for i in np.asarray(record["location_ids"]).ravel()]
    if len(ids) != x.shape[1]:
        raise ValueError(f"{len(ids)} location ids for {x.shape[1]} sites")
    m, n = mean_acc(x, record["channel_mask"], record.get("sample_valid"))
    rots = np.repeat(np.eye(3)[None], x.shape[1], axis=0)
    turned = np.zeros(x.shape[1], bool)
    info = {"sites": len(ids), "turned": 0, "identity_no_acc": 0, "identity_short_mean": 0, "angle_max_deg": 0.0}
    out = x.copy()
    for g, sid in enumerate(ids):
        if n[g] == 0:
            info["identity_no_acc"] += 1
            continue
        if not np.linalg.norm(m[g]) >= MIN_MEAN_G:
            info["identity_short_mean"] += 1
            continue
        u = canonical_up(sid)
        R = turn_rotation(m[g], u)
        rots[g] = R
        turned[g] = True
        for sl in (ACC, GYRO, MAG):
            out[:, g, sl] = x[:, g, sl] @ R.T
        info["turned"] += 1
    if info["turned"]:                    # the check of the code: the turned signal's own mean, against u
        m2, _ = mean_acc(out, record["channel_mask"], record.get("sample_valid"))
        for g, sid in enumerate(ids):
            if turned[g]:
                info["angle_max_deg"] = max(info["angle_max_deg"], angle_deg(m2[g], canonical_up(sid)))
    rec = dict(record)
    rec["signal"] = out
    return rec, rots, info


def rotation_meta(rots):
    """The 'tilt_rotation' meta entry: one 3 x 3 nested list per site (float, full precision)."""
    return [[[float(v) for v in row] for row in R] for R in np.asarray(rots, dtype=np.float64)]


# ---------------------------------------------------------------- heading features
def principal_angle(xy):
    """Angle (rad, in [0, pi)) of the first principal axis of (n, 2) points (centred); nan for fewer than 3 points or
    no spread."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return math.nan
    c = np.cov((xy - xy.mean(axis=0)).T)
    if not np.isfinite(c).all() or np.trace(c) <= 0:
        return math.nan
    w, v = np.linalg.eigh(c)
    a = v[:, int(np.argmax(w))]
    return float(math.atan2(a[1], a[0]) % math.pi)


def heading_features(acc, gyro, valid, u):
    """[cos 2a, sin 2a] of the first principal axis of the horizontal acceleration, then of the horizontal angular
    rate (gyro None: acceleration only), in the plane normal to u with horizontal_basis(u), over the valid samples
    (module docstring); nan where the angle is undefined."""
    e1, e2 = horizontal_basis(u)
    B = np.stack([e1, e2], axis=1)                    # (3, 2)
    ok = np.asarray(valid, dtype=bool)
    out = []
    for v in (acc, gyro):
        if v is None:
            continue
        a = principal_angle(np.asarray(v, dtype=np.float64)[ok] @ B)
        out += [math.cos(2 * a), math.sin(2 * a)] if math.isfinite(a) else [math.nan, math.nan]
    return out


def covariance(v):
    """(3, 3) covariance of (n, 3) points divided by its trace (the shape of the spread, not its size); None for
    fewer than 3 points, a non-finite value or no spread."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 3:
        return None
    c = np.cov((v - v.mean(axis=0)).T)
    t = float(np.trace(c)) if np.isfinite(c).all() else math.nan
    if not t > 0:
        return None
    return c / t


def cov_entries(c):
    """The COV_ENTRIES of a covariance (6 floats; nan when c is None)."""
    return [math.nan] * len(COV_ENTRIES) if c is None else [float(c[i, j]) for i, j in COV_ENTRIES]


# ---------------------------------------------------------------- probe
def balanced_accuracy(y, pred):
    """Mean over the classes present in y of the share of their records predicted right."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def cross_validate(X, y, persons, seed=SEED, folds=FOLDS):
    """Out-of-fold predictions (array like y) of the person-grouped classifier (module docstring)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=y.dtype)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups=persons):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C_REG, class_weight="balanced", max_iter=5000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def permuted_groups(y, persons, rng):
    """Record groups after shuffling the persons' groups among the persons (each person keeps one group)."""
    persons = np.asarray(persons)
    first = {}
    for p, g in zip(persons, np.asarray(y)):
        if first.setdefault(p, g) != g:
            raise ValueError(f"person {p} has records of two groups")
    order = sorted(first)
    shuffled = dict(zip(order, rng.permutation([first[p] for p in order])))
    return np.array([shuffled[p] for p in persons])


def probe(X, y, persons, seed=SEED, folds=FOLDS, permutations=PERMUTATIONS):
    """{balanced_accuracy, person_balanced_accuracy (majority of a person's out-of-fold predictions, ties to the
    first class in sorted order), chance: {values, mean, sd, min, max, at_or_above_observed}} of the classifier on
    (X, y) grouped by persons (module docstring). Rows with a non-finite feature are refused (ValueError)."""
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("non-finite features: impute before the probe")
    y = np.asarray(y)
    persons = np.asarray(persons)
    pred = cross_validate(X, y, persons, seed, folds)
    out = {"records": int(len(y)), "persons": int(len(set(persons.tolist()))), "features": int(X.shape[1]),
           "classes": sorted(set(y.tolist())), "balanced_accuracy": balanced_accuracy(y, pred),
           "person_balanced_accuracy": _person_ba(y, pred, persons)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(permutations):
        yp = permuted_groups(y, persons, rng)
        vals.append(balanced_accuracy(yp, cross_validate(X, yp, persons, seed, folds)))
    v = np.array(vals)
    out["chance"] = {"values": [float(a) for a in v], "mean": float(v.mean()) if len(v) else None,
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()) if len(v) else None,
                     "max": float(v.max()) if len(v) else None,
                     "at_or_above_observed": int((v >= out["balanced_accuracy"]).sum())}
    return out


def _person_ba(y, pred, persons):
    truth, guess = [], []
    for p in sorted(set(persons.tolist())):
        sel = persons == p
        vals, counts = np.unique(pred[sel], return_counts=True)
        truth.append(y[sel][0])
        guess.append(vals[int(np.argmax(counts))])
    return balanced_accuracy(np.array(truth), np.array(guess))


# ---------------------------------------------------------------- reading a built store
def store_records(store, source):
    """Per stored input IMU record of `source` in a mdfm.data.window_store.WindowStore, one dict: person_id, key,
    sites (names), and per site: n (valid acceleration samples), mean (the stored mean acceleration, g), up (the
    canonical up axis), rotation (the tilt_rotation of the meta, identity without one), angle_stored_deg (stored
    mean against up: the turn seen through band copy and float16), turn_deg (the raw mean R^T mean against up: how
    far the turn went), heading (heading_features of the stored samples, acceleration and, when present, angular
    rate), mean_raw_dir (R^T mean / |mean|, the mean-acceleration direction in the sensor frame), cov_turned (the
    cov_entries of covariance() of the stored acceleration, then of the stored angular rate, each over its valid
    finite samples; nan for an absent angular rate) and cov_sensor (the same covariances in the sensor frame,
    R^T C R, since a stored sample is R times the sensor's)."""
    names = sites()
    r = store.records
    rows = r[(r["source"] == source) & (r["stream"] == "imu") & (r["role"] == "input")]
    out = []
    for i, row in rows.iterrows():
        st = store.stream(i)
        meta = st.meta
        ids = [int(x) for x in meta["site_ids"]]
        rots = np.asarray(meta.get("tilt_rotation") or [np.eye(3).tolist()] * len(ids), dtype=np.float64)
        m, n = mean_acc(st.values, st.chan_mask, st.valid)
        has_gyro = st.chan_mask[:, GYRO].all(axis=1)
        rec = {"person_id": row["person_id"], "key": row["key"], "sites": [names[s] for s in ids], "per_site": []}
        for g, sid in enumerate(ids):
            u = canonical_up(sid)
            R = rots[g]
            d = {"site": names[sid], "n": int(n[g]), "up": u, "rotation": R, "turned": not np.allclose(R, np.eye(3))}
            if n[g]:
                d["mean"] = m[g]
                d["angle_stored_deg"] = angle_deg(m[g], u)
                raw = R.T @ m[g]
                d["turn_deg"] = angle_deg(raw, u)
                d["mean_raw_dir"] = raw / np.linalg.norm(raw)
                ok = st.valid[:, g]
                d["heading"] = heading_features(st.values[:, g, ACC], st.values[:, g, GYRO] if has_gyro[g] else None,
                                                ok, u)
                covs = []
                for sl, present in ((ACC, True), (GYRO, bool(has_gyro[g]))):
                    v = np.asarray(st.values[:, g, sl], dtype=np.float64)
                    covs.append(covariance(v[ok & np.isfinite(v).all(axis=1)]) if present else None)
                d["cov_turned"] = [x for c in covs for x in cov_entries(c)]
                d["cov_sensor"] = [x for c in covs for x in cov_entries(None if c is None else R.T @ c @ R)]
            rec["per_site"].append(d)
        out.append(rec)
    return out


def feature_matrix(records, site_names, before, kind="heading"):
    """(X, names, undefined): per record, for each site of site_names, present (0 or 1) and then, by kind:
    heading, the heading features of store_records (4 numbers: acceleration then angular rate); covariance, its
    cov_sensor with before=True and its cov_turned with before=False (12 numbers: acceleration then angular rate);
    presence, nothing more (before is not used). Zeros where the site is absent or a number is undefined (the
    undefined ones counted in `undefined`). With before=True (heading and covariance) the mean-acceleration direction
    in the sensor frame follows (3 numbers, zeros when absent). A site is present when it has a valid acceleration
    sample."""
    if kind not in KINDS:
        raise ValueError(f"feature kind {kind!r}, expected one of {KINDS}")
    mean = before and kind != "presence"
    cols = []
    for s in site_names:
        cols.append(f"{s}.present")
        if kind == "heading":
            cols += [f"{s}.acc_cos2", f"{s}.acc_sin2", f"{s}.gyro_cos2", f"{s}.gyro_sin2"]
        elif kind == "covariance":
            cols += [f"{s}.{t}_c{'xyz'[a]}{'xyz'[b]}" for t in ("acc", "gyro") for a, b in COV_ENTRIES]
        if mean:
            cols += [f"{s}.mean_x", f"{s}.mean_y", f"{s}.mean_z"]
    width = len(cols) // max(len(site_names), 1)
    X = np.zeros((len(records), len(cols)))
    undefined = 0
    for i, rec in enumerate(records):
        by = {d["site"]: d for d in rec["per_site"]}
        j = 0
        for s in site_names:
            d = by.get(s)
            if d is not None and d.get("heading") is not None:
                X[i, j] = 1.0
                if kind == "heading":
                    h = np.asarray(d["heading"] + [math.nan] * (4 - len(d["heading"])), dtype=np.float64)
                elif kind == "covariance":
                    h = np.asarray(d["cov_sensor"] if before else d["cov_turned"], dtype=np.float64)
                else:
                    h = np.zeros(0)
                undefined += int((~np.isfinite(h)).sum())
                X[i, j + 1:j + 1 + len(h)] = np.where(np.isfinite(h), h, 0.0)
                if mean:
                    X[i, j + 1 + len(h):j + 4 + len(h)] = d["mean_raw_dir"]
            j += width
    return X, cols, undefined
