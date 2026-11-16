"""Gate G2, first half (design section 9): can a discriminator tell real IMU walking from rendered?

Real: weargait_pd healthy controls (the source's CONTROLS table: HC, WHC and some NLS ids),
walking trials SelfPace and HurriedPace, the 13
Xsens sites of the MoveAtlas canonical IMU records (datasets/imu/weargait_pd/processed/
canonical_imu.npz, records in the order of the pooled IMU manifest; acc in g, gyro in deg/s,
100 Hz). Version 2 (2026-09-28): only the samples labelled Walk in the GeneralEvent column of
the raw trial CSV count (the trials also hold Standing and Turn phases, about a third of the
samples, which straight AMASS walking does not contain; version 1 used every sample and its
outputs are in reports/g2_renderer/v1_all_phases/), and samples a sensor did not deliver
(sample_valid false) are masked for that sensor. A window must lie inside one Walk phase.
The rendered side gets the matching rule: only the steady-walking frames of each AMASS take
count (the KIT takes start and stop standing), a frame being steady when the horizontal
pelvis speed relative to the stance ankle (mdfm.factory.gait), smoothed by a 1 s moving
median, exceeds half its 95th percentile over the take. Rendered takes are cached under
cache/g2_rendered/ per mounting.
Version 4 (2026-09-28): lockbox persons (reports/lockbox/lockbox_persons.csv, through the person
map) never enter G2, since the renderer is tuned against G2 and the weargait_pd lockbox serves
G3; controls whose MoveAtlas id carries the suffix ' (control)' (three NLS ids) are matched to
the CONTROLS table after stripping it (version 3 and earlier left them out); the device model
takes its white-noise levels and gyroscope bias spread from reports/g2_renderer/device_noise.json
(scripts/g2_device_noise.py, measured on quiet standing of ALIGN-half controls) when that file
exists; the body low-pass cutoff is G2_BODY_CUTOFF_HZ (environment, default 12). Residual spectrum
(version 4, the first step of the renderer loop of design section 13, a stand-in for the
soft-tissue filter of design section 3): per site and channel, the power the real ALIGN-half
walking windows carry above the rendered windows in 5 to 50 Hz (mean periodograms of 2 s
windows, negative gaps set to zero) is added to every rendered take as stationary Gaussian noise
with that spectrum; the discriminator is scored with and without it (the verdict and the per-site table use the
renderer without it, since it raised the balanced accuracy), and prepare() gives the other
G2 scripts the same data. Rendered: the straight-walking AMASS takes of reports/amass_walks/takes.csv (KIT
overground, BMLrub treadmill; no G1 lockbox subset), rendered by mdfm.factory.imu_render at
100 Hz with the default device model.

Mounting: per site, one rotation from the default segment frame to the Xsens frame, fitted at
population level (Wahba's problem on two vectors: the mean specific force over walking, which
points against gravity, and the first principal axis of the angular rate) using the real
controls of the ALIGN half only; the discriminator is scored on the other half (TEST) and on
all rendered persons, never on a person whose data fitted the mounting. The principal axis has
no sign, and both signs fit the two vectors equally well; version 3 (2026-09-28) picks the sign
whose rotated rendered signals match the pooled real ALIGN signals best in distribution (the
5th, 25th, 50th, 75th and 95th percentiles of each acc and gyro channel, each scaled by the real
channel's interquartile range). Version 2 chose the sign by the vector residual alone, which
cannot tell them apart, and turned the shank and ankle sensors by 180 degrees about the segment
axis (reports/g2_renderer/feature_gaps.md of that run: the median sagittal angular rate had
opposite signs).

Matching: rendered takes are resampled so that their cadence histogram (5 steps/min bins)
follows the real TEST cadence histogram (cadence of a trial, real or rendered, from the
dominant 1.2 to 3.0 Hz frequency of the lower-back acceleration magnitude, one formula for both).

Discriminator: 2 s windows, 1 s hop; per site and channel (acc xyz, gyro xyz) the mean,
standard deviation, 10th, 50th and 90th percentiles and the power in 0.5 to 3, 3 to 8 and 8 to
20 Hz; HistGradientBoostingClassifier; 5-fold GroupKFold on persons; balanced accuracy with a
person-cluster bootstrap interval (1,000 draws). G2 lines: pass at most 0.70, kill above 0.85.
Per-site discriminators show where realism fails.

Needs the mdfm environment. Writes reports/g2_renderer/ (summary.md, results.json,
per_site.csv, mounting.json). Run: <conda>/envs/mdfm/python.exe scripts/g2_renderer_realism.py
"""
import csv
import json
import os
import sys
import zlib

import numpy as np
import torch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401
from mdfm import paths  # noqa: E402
from mdfm.body.trajectory import BodyTrajectory  # noqa: E402
# the SMPL+H body and the AMASS reader are imported where the AMASS walks are rendered (rendered_takes), so that a
# script reusing this module's functions on the open body (scripts/g2_open_render.py) imports neither
from mdfm.factory.imu_render import SITES, Device, render  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(TREE, "reports", "g2_renderer")
os.makedirs(OUT, exist_ok=True)
RATE, WIN, HOP = 100.0, 200, 100
REAL_TASKS = ("SelfPace", "HurriedPace")
REAL_PHASE = "Walk"
BODY_CUTOFF_HZ = float(os.environ.get("G2_BODY_CUTOFF_HZ", "12"))


def lockbox_persons():
    rows = csv.DictReader(open(os.path.join(TREE, "reports", "lockbox", "lockbox_persons.csv"), encoding="utf-8"))
    return {r["person_id"] for r in rows}


def person_of():
    rows = csv.DictReader(open(os.path.join(TREE, "reports", "person_ids", "person_map.csv"), encoding="utf-8"))
    return {r["subject_ns"]: r["person_id"] for r in rows if r["source"] == "weargait_pd"}


def device_model():
    """Device of the rendered takes: measured noise levels when reports/g2_renderer/device_noise.json
    exists (version 4), else the factory defaults."""
    p = os.path.join(OUT, "device_noise.json")
    if not os.path.exists(p):
        return Device(), "factory defaults"
    d = json.load(open(p, encoding="utf-8"))
    by = {r["site"]: r for r in d["per_site"]}
    acc = torch.tensor([by[s]["acc_sigma_g"] for s in SITES13], dtype=torch.float64)
    gyr = torch.tensor([by[s]["gyro_sigma_dps"] for s in SITES13], dtype=torch.float64)
    gb = torch.tensor([by[s]["gyro_bias_sd_dps"] for s in SITES13], dtype=torch.float64)
    return Device(acc_noise_g=acc, gyro_noise_dps=gyr, gyro_bias_dps=gb), "measured per site and axis (device_noise.json)"
RAW_CONTROLS = f"{paths.MOVEATLAS}/datasets/imu/weargait_pd/raw/Version 1/CONTROL PARTICIPANTS/CSV files"
SITES13 = [s for s in SITES if s != "pocket"]
rng = np.random.default_rng(20260928)


# ---------------------------------------------------------------- real data
def weargait_controls():
    """Control subject IDs from the source's CONTROLS table (HC, WHC and some NLS ids)."""
    import glob
    p = glob.glob(f"{paths.MOVEATLAS}/datasets/imu/weargait_pd/raw/*/CONTROLS - Demographic+Clinical - datasetV1.csv")[0]
    rows = list(csv.reader(open(p, encoding="utf-8-sig", newline="")))
    head = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Subject ID")
    return {r[0].strip() for r in rows[head + 1:] if r and r[0].strip()}


def real_records(phase=REAL_PHASE, include_lockbox=False):
    controls = weargait_controls()
    locked = set() if include_lockbox else lockbox_persons()
    pmap = person_of()
    rows = [r for r in csv.DictReader(open(f"{paths.MOVEATLAS}/datasets/_canonical_imu_manifest.csv",
                                           encoding="utf-8", newline="")) if r["source"] == "weargait_pd"]
    z = np.load(f"{paths.MOVEATLAS}/datasets/imu/weargait_pd/processed/canonical_imu.npz")
    out = []
    for i, r in enumerate(rows):
        sig = z[f"r{i:06d}/signal"]
        assert sig.shape[0] == int(r["n_frames"]), (i, r)
        pid = r["subject_ns"].split(":")[-1]
        if pid.replace(" (control)", "") not in controls or r["task"] not in REAL_TASKS:
            continue
        if pmap.get(r["subject_ns"], r["subject_ns"]) in locked:
            continue
        loc = z[f"r{i:06d}/location_ids"]
        valid = z[f"r{i:06d}/sample_valid"]
        ev = phase_labels(pid, r["task"])
        assert len(ev) == sig.shape[0], (pid, r["task"], len(ev), sig.shape)
        sig = sig.astype(np.float32).copy()
        sig[~valid.astype(bool)] = np.nan
        sig[ev != phase] = np.nan
        out.append({"person": r["subject_ns"], "task": r["task"], "signal": sig, "loc": loc, "valid": valid,
                    "phase_fraction": float((ev == phase).mean())})
    return out


def phase_labels(pid, task):
    """GeneralEvent labels of the raw trial CSV (one per sample, same length as the canonical record)."""
    import pandas as pd
    df = pd.read_csv(f"{RAW_CONTROLS}/{pid}_{task}.csv", usecols=["GeneralEvent"], encoding="utf-8")
    return df["GeneralEvent"].astype(str).str.strip().to_numpy()


def to_sites13(sig, loc_ids):
    """(T, L, 9) with canonical location ids -> (T, 13, 6) acc and gyro in SITES13 order."""
    x = np.full((sig.shape[0], 13, 6), np.nan, np.float32)
    for j, lid in enumerate(loc_ids):
        name = SITES[int(lid)]
        if name in SITES13:
            x[:, SITES13.index(name)] = sig[:, j, :6]
    return x


def cadence_from_lower_back(x):
    """steps/min from the dominant 1.2 to 3.0 Hz frequency of the lower-back acceleration
    magnitude (one peak per step)."""
    a = np.linalg.norm(x[:, SITES13.index("lower_back"), 0:3], axis=1)
    a = a - np.nanmean(a)
    a = np.nan_to_num(a)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    f = np.fft.rfftfreq(len(a), 1 / RATE)
    band = (f >= 1.2) & (f <= 3.0)
    return 60.0 * float(f[band][np.argmax(spec[band])])


# ---------------------------------------------------------------- rendered data
def steady_mask(traj, t, up=1):
    """Steady-walking frames at times t: horizontal pelvis speed relative to the stance ankle
    (the formula of mdfm.factory.gait.speed_and_cadence), 1 s moving median, above half its
    95th percentile."""
    from scipy.ndimage import median_filter
    ones = torch.ones_like(t)
    (pos, _), (vel, _) = torch.func.jvp(lambda tt: traj.pose_at(tt), (t,), (ones,))
    pos, vel = pos.detach().numpy(), vel.detach().numpy()
    stance = np.where(pos[:, 7, up] <= pos[:, 8, up], 7, 8)
    rel = vel[:, 0] - vel[np.arange(len(stance)), stance]
    horiz = np.linalg.norm(np.delete(rel, up, axis=1), axis=1)
    sm = median_filter(horiz, size=int(RATE), mode="nearest")
    return sm > 0.5 * np.percentile(sm, 95)


def _mount_tag(mount):
    dev, _ = device_model()
    extra = f"_c{BODY_CUTOFF_HZ:g}_n{zlib.crc32(repr(dev).encode()):08x}"
    if mount is None:
        return "nomount" + extra
    h = zlib.crc32(np.concatenate([np.asarray(mount[s]).ravel() for s in SITES13]).round(6).tobytes())
    return f"mount_{h:08x}" + extra


def rendered_takes(mount):
    tag = _mount_tag(mount)
    cpath = os.path.join(paths.cache_dir("g2_rendered"), f"{tag}.npz")
    rows = list(csv.DictReader(open(os.path.join(TREE, "reports", "amass_walks", "takes.csv"), encoding="utf-8")))
    if os.path.exists(cpath):
        z = np.load(cpath)
        out = []
        for i, r in enumerate(rows):
            if f"x{i}" not in z.files:
                continue
            parts = r["take"].split("/")
            x = z[f"x{i}"]
            out.append({"person": f"amass:{r['subset']}:{parts[1] if len(parts) > 2 else parts[0]}", "take": r["take"],
                        "x": x, "cadence": cadence_from_lower_back(x)})
        return out
    from mdfm.body.smplh import SMPLHBody
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    bodies = {}
    out, store = [], {}
    for i, r in enumerate(rows):
        seq = load_sequence(r["subset"], r["take"])
        g = r["gender"]
        body = bodies.setdefault(g, SMPLHBody(g, dtype=torch.float64))
        r6, tr, b = amass_to_body_clock(seq, rate=RATE, cutoff=BODY_CUTOFF_HZ, body=body)
        traj = BodyTrajectory(body, r6, tr, b, rate=RATE)
        t = torch.arange(0.5, (r6.shape[0] - 1) / RATE - 0.5, 1 / RATE, dtype=torch.float64)
        if t.numel() < WIN:
            continue
        gen = torch.Generator().manual_seed(zlib.crc32(r["take"].encode()))
        with torch.no_grad():
            sig, _, _ = render(traj, t, sites=SITES13, mount=mount, device=device_model()[0], generator=gen)
        x = sig[:, [SITES.index(s) for s in SITES13], :6].numpy().copy()
        x[~steady_mask(traj, t)] = np.nan
        parts = r["take"].split("/")
        person = f"amass:{r['subset']}:{parts[1] if len(parts) > 2 else parts[0]}"
        out.append({"person": person, "take": r["take"], "x": x, "cadence": cadence_from_lower_back(x)})
        store[f"x{i}"] = x
    np.savez(cpath, **store)
    return out


# ---------------------------------------------------------------- mounting fit
def site_vectors(xs):
    """Per site: mean specific-force direction and first principal axis of the gyro, pooled."""
    acc = [np.nanmean(x[:, :, 0:3], 0) for x in xs]
    g = np.nanmean(np.stack(acc), 0)
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    ax = []
    for s in range(13):
        w = np.concatenate([x[:, s, 3:6] for x in xs])
        w = w[np.isfinite(w).all(1)]
        _, _, vt = np.linalg.svd(w - w.mean(0), full_matrices=False)
        ax.append(vt[0])
    return g, np.stack(ax)


def wahba(a_vecs, b_vecs):
    """Rotation Q with b ~ Q a (rows are vector pairs)."""
    B = sum(np.outer(b, a) for a, b in zip(a_vecs, b_vecs))
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt


QS = (5, 25, 50, 75, 95)


def _pooled(xs, s):
    v = np.concatenate([x[:, s, :] for x in xs])
    return v[np.isfinite(v).all(1)]


def _distribution_distance(real_v, rend_v, Q):
    rot = np.concatenate([rend_v[:, 0:3] @ Q.T, rend_v[:, 3:6] @ Q.T], axis=1)
    pr = np.percentile(real_v, QS, axis=0)
    pm = np.percentile(rot, QS, axis=0)
    iqr = pr[3] - pr[1] + 1e-9
    return float((((pr - pm) / iqr) ** 2).sum())


def fit_mounting(real_x, rend_x, report=None):
    g_r, ax_r = site_vectors(real_x)
    g_m, ax_m = site_vectors(rend_x)
    mount, resid = {}, {}
    for s, name in enumerate(SITES13):
        rv, mv = _pooled(real_x, s), _pooled(rend_x, s)
        mv = mv[np.random.default_rng(s).choice(len(mv), min(len(mv), 200000), replace=False)]
        best = None
        for sign in (1.0, -1.0):
            Q = wahba([g_m[s], ax_m[s]], [g_r[s], sign * ax_r[s]])
            err = np.degrees(np.arccos(np.clip(Q @ g_m[s] @ g_r[s], -1, 1))) + \
                np.degrees(np.arccos(np.clip(abs(Q @ ax_m[s] @ ax_r[s]), -1, 1)))
            dist = _distribution_distance(rv, mv, Q)
            if report is not None:
                report.append({"site": name, "sign": sign, "vector_residual_deg": round(float(err), 2),
                               "distribution_distance": round(dist, 3)})
            if best is None or dist < best[2]:
                best = (Q, err, dist)
        mount[name] = torch.as_tensor(best[0].T)  # render() uses R_seg @ mount; readings = Q (segment readings)
        resid[name] = round(float(best[1]), 2)
    return mount, resid


# ---------------------------------------------------------------- residual spectrum
RESID_LO_HZ = 5.0
FW = np.fft.rfftfreq(WIN, 1 / RATE)


def mean_periodogram(xs):
    """(13, 6, F) mean one-sided periodogram (units^2/Hz) over the complete 2 s windows of xs."""
    acc, n = np.zeros((13, 6, len(FW))), 0
    for x in xs:
        for w in windows(x):
            c = w - w.mean(0)
            acc += (2.0 * np.abs(np.fft.rfft(c, axis=0)) ** 2 / (RATE * WIN)).transpose(1, 2, 0)
            n += 1
    return acc / max(n, 1)


def fit_residual(real_xs, rend_xs):
    d = np.clip(mean_periodogram(real_xs) - mean_periodogram(rend_xs), 0.0, None)
    d[:, :, FW < RESID_LO_HZ] = 0.0
    return d


def apply_residual(x, d, seed):
    """x (T, 13, 6) plus stationary Gaussian noise with one-sided density d (13, 6, F) on FW."""
    g = np.random.default_rng(seed)
    T = x.shape[0]
    f = np.fft.rfftfreq(T, 1 / RATE)
    out = x.copy()
    for s in range(13):
        for c in range(6):
            p = np.interp(f, FW, d[s, c])
            if not p.any():
                continue
            X = np.sqrt(p * T * RATE / 4.0) * (g.standard_normal(len(f)) + 1j * g.standard_normal(len(f)))
            X[0] = 0.0
            out[:, s, c] = x[:, s, c] + np.fft.irfft(X, n=T)
    return out


# ---------------------------------------------------------------- features and discriminator
BANDS = ((0.5, 3.0), (3.0, 8.0), (8.0, 20.0))


def windows(x):
    for s in range(0, x.shape[0] - WIN + 1, HOP):
        w = x[s:s + WIN]
        if np.isfinite(w).all():
            yield w


def features(w, sites=None):
    idx = range(13) if sites is None else sites
    f = np.fft.rfftfreq(WIN, 1 / RATE)
    out = []
    for s in idx:
        c = w[:, s, :]
        out += list(c.mean(0)) + list(c.std(0)) + list(np.percentile(c, [10, 50, 90], axis=0).ravel())
        p = np.abs(np.fft.rfft(c - c.mean(0), axis=0)) ** 2
        for lo, hi in BANDS:
            out += list(np.log(p[(f >= lo) & (f < hi)].sum(0) + 1e-9))
    return out


def discriminate(real, rend, sites=None, folds=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X, y, g = [], [], []
    for item, label in [(r, 1) for r in real] + [(r, 0) for r in rend]:
        for w in windows(item["x"]):
            X.append(features(w, sites))
            y.append(label)
            g.append(item["person"])
    X, y, g = np.asarray(X, np.float32), np.asarray(y), np.asarray(g)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])

    def bacc(idx):
        yy, pp = y[idx], pred[idx]
        return 0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean())

    allidx = np.arange(len(y))
    point = bacc(allidx)
    persons = np.unique(g)
    by = {p: np.where(g == p)[0] for p in persons}
    boot = []
    for _ in range(1000):
        pick = rng.choice(persons, len(persons), replace=True)
        idx = np.concatenate([by[p] for p in pick])
        if len(np.unique(y[idx])) == 2:
            boot.append(bacc(idx))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"balanced_accuracy": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "windows_real": int((y == 1).sum()), "windows_rendered": int((y == 0).sum()),
            "persons_real": int(len(np.unique(g[y == 1]))), "persons_rendered": int(len(np.unique(g[y == 0])))}


# ---------------------------------------------------------------- shared preparation
def prepare(seed=20260928):
    """The G2 data of version 4 for every G2 script: real ALIGN and TEST halves, rendered takes
    without and with the mounting fit and with the residual spectrum, and the cadence-matched
    indices into the rendered takes."""
    rg = np.random.default_rng(seed)
    real = real_records()
    for r in real:
        r["x"] = to_sites13(r["signal"], r["loc"])
        r["cadence"] = cadence_from_lower_back(r["x"])
    persons = sorted({r["person"] for r in real})
    perm = rg.permutation(len(persons))
    align_p = {persons[i] for i in perm[: len(persons) // 2]}
    real_align = [r for r in real if r["person"] in align_p]
    real_test = [r for r in real if r["person"] not in align_p]
    rend0 = rendered_takes(mount=None)
    mount, resid = fit_mounting([r["x"] for r in real_align], [r["x"] for r in rend0])
    rend = rendered_takes(mount=mount)
    d = fit_residual([r["x"] for r in real_align], [r["x"] for r in rend])
    rend_res = [dict(r, x=apply_residual(r["x"], d, zlib.crc32(r["take"].encode())).astype(np.float32)) for r in rend]
    bins = np.arange(60, 160, 5)
    hr, _ = np.histogram([r["cadence"] for r in real_test], bins)
    which = np.digitize([r["cadence"] for r in rend], bins) - 1
    chosen = []
    for b, n_real in enumerate(hr):
        pool = [i for i in range(len(rend)) if which[i] == b]
        k = int(round(n_real / max(hr.sum(), 1) * len(rend)))
        if pool and k:
            chosen += list(rg.choice(pool, k, replace=len(pool) < k))
    return {"real_align": real_align, "real_test": real_test, "rend0": rend0, "rend": rend, "rend_res": rend_res,
            "chosen": chosen, "mount": mount, "resid": resid, "residual_density": d}


# ---------------------------------------------------------------- main
def main():
    global peak, GYRO_IN_RAD
    real = real_records()
    for r in real:
        r["x"] = to_sites13(r["signal"], r["loc"])
        r["cadence"] = cadence_from_lower_back(r["x"])
    # unit check of the real gyroscope (the MoveAtlas datacard notes an export in rad/s before
    # 2026-09-17): walking shank rates peak at hundreds of deg/s, a few rad/s
    shank = np.concatenate([r["x"][:, SITES13.index("shank_L"), 3:6] for r in real])
    peak = float(np.nanpercentile(np.abs(shank), 99))
    GYRO_IN_RAD = peak < 20.0
    if GYRO_IN_RAD:
        for r in real:
            r["x"][:, :, 3:6] = np.degrees(r["x"][:, :, 3:6])
    print(f"real shank gyro 99th percentile {peak:.2f}; treated as {'rad/s, converted' if GYRO_IN_RAD else 'deg/s'}", flush=True)
    persons = sorted({r["person"] for r in real})
    perm = rng.permutation(len(persons))
    align_p = {persons[i] for i in perm[: len(persons) // 2]}
    real_align = [r for r in real if r["person"] in align_p]
    real_test = [r for r in real if r["person"] not in align_p]
    print(f"real trials {len(real)} ({len(persons)} controls): align {len(real_align)}, test {len(real_test)}", flush=True)

    rend0 = rendered_takes(mount=None)
    print(f"rendered takes {len(rend0)}", flush=True)
    sign_report = []
    mount, resid = fit_mounting([r["x"] for r in real_align], [r["x"] for r in rend0], report=sign_report)
    with open(os.path.join(OUT, "mounting_sign.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sign_report[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sign_report)
    json.dump({k: v.numpy().round(6).tolist() for k, v in mount.items()} | {"residual_deg": resid},
              open(os.path.join(OUT, "mounting.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    rend = rendered_takes(mount=mount)

    # cadence matching: resample rendered takes to the real TEST cadence histogram
    bins = np.arange(60, 160, 5)
    hr, _ = np.histogram([r["cadence"] for r in real_test], bins)
    cad_m = np.array([r["cadence"] for r in rend])
    which = np.digitize(cad_m, bins) - 1
    target = len(rend)
    chosen = []
    for b, n_real in enumerate(hr):
        pool = [i for i in range(len(rend)) if which[i] == b]
        k = int(round(n_real / max(hr.sum(), 1) * target))
        if pool and k:
            chosen += list(rng.choice(pool, k, replace=len(pool) < k))
    rend_matched = [rend[i] for i in chosen]
    print(f"matched rendered takes {len(rend_matched)} (unique {len(set(chosen))})", flush=True)

    d = fit_residual([r["x"] for r in real_align], [r["x"] for r in rend])
    np.save(os.path.join(paths.cache_dir("g2_rendered"), "residual_density.npy"), d)
    rend_res = [dict(r, x=apply_residual(r["x"], d, zlib.crc32(r["take"].encode())).astype(np.float32)) for r in rend]
    res = {"all_sites_with_residual": discriminate(real_test, [rend_res[i] for i in chosen]),
           "all_sites": discriminate(real_test, rend_matched)}
    print("with residual", res["all_sites_with_residual"], flush=True)
    # diagnostic: the same test with the real signals low-passed at the 12 Hz of the rendered
    # bodies (mdfm.data.prep), which separates the bandwidth gap from the other realism gaps
    from mdfm.data.prep import lowpass  # noqa: E402
    real_test_lp = [dict(r, x=np.where(np.isfinite(r["x"]), lowpass(np.nan_to_num(r["x"]), RATE, 12.0), np.nan).astype(np.float32))
                    for r in real_test]
    res["all_sites_real_lowpassed_12hz"] = discriminate(real_test_lp, rend_matched)
    res["without_mounting_fit"] = discriminate(real_test, [rend0[i] for i in chosen])
    per_site = []
    for s, name in enumerate(SITES13):
        r = discriminate(real_test, rend_matched, sites=[s])
        per_site.append({"site": name, **r, "mounting_residual_deg": resid[name]})
        print(name, r["balanced_accuracy"], flush=True)
    res["cadence_real_test_quartiles"] = np.percentile([r["cadence"] for r in real_test], [25, 50, 75]).round(1).tolist()
    res["cadence_rendered_matched_quartiles"] = np.percentile([r["cadence"] for r in rend_matched], [25, 50, 75]).round(1).tolist()
    ba = res["all_sites"]["balanced_accuracy"]  # the renderer without the residual spectrum, which version 4 did not adopt
    res["verdict"] = "pass" if ba <= 0.70 else ("kill" if ba > 0.85 else "undetermined")
    res["real_gyro_p99_shank"] = round(peak, 2)
    res["real_gyro_converted_from_rad"] = GYRO_IN_RAD
    json.dump(res, open(os.path.join(OUT, "results.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    with open(os.path.join(OUT, "per_site.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_site[0]) , lineterminator="\n")
        w.writeheader()
        for r in per_site:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    md = ["# G2 renderer realism, first half", "", "Measured by scripts/g2_renderer_realism.py (method in its docstring).", "",
          f"All 13 sites: balanced accuracy {ba} (95 percent person-bootstrap interval {res['all_sites']['ci95']}); "
          f"G2 lines: pass at most 0.70, kill above 0.85; verdict: {res['verdict']}.",
          f"Without the population mounting fit: {res['without_mounting_fit']['balanced_accuracy']}.",
          f"Diagnostic, with the residual spectrum (not adopted): {res['all_sites_with_residual']['balanced_accuracy']} {res['all_sites_with_residual']['ci95']}.",
          f"Diagnostic, real signals low-passed at 12 Hz like the rendered bodies: "
          f"{res['all_sites_real_lowpassed_12hz']['balanced_accuracy']} {res['all_sites_real_lowpassed_12hz']['ci95']}.",
          f"Windows: real {res['all_sites']['windows_real']} from {res['all_sites']['persons_real']} controls, rendered "
          f"{res['all_sites']['windows_rendered']} from {res['all_sites']['persons_rendered']} AMASS performers.",
          f"Cadence quartiles, real test {res['cadence_real_test_quartiles']}, rendered matched {res['cadence_rendered_matched_quartiles']}.",
          f"Device model: {device_model()[1]} {device_model()[0]}; body low-pass cutoff {BODY_CUTOFF_HZ:g} Hz; lockbox persons excluded.", "",
          "| site | balanced accuracy | 95 percent interval | mounting residual (deg) |", "|---|---|---|---|"]
    md += [f"| {r['site']} | {r['balanced_accuracy']} | {r['ci95']} | {r['mounting_residual_deg']} |" for r in per_site]
    text = "\n".join(md) + "\n"
    assert chr(0x2014) not in text
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
