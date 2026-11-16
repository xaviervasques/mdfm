"""Mounting and device priors of the IMU nuisance (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.3,
task B4.7), read by mdfm.kernels.nuisance.MountPriors.

Families. xsens13: the 13 Xsens sites of weargait_pd, measured here. generic: every site of the
14-site taxonomy, centred on the anatomical default (identity mount) with 10 degrees of tilt and yaw
spread [cfg], and the Device defaults of mdfm.factory.imu_render; the pocket site of xsens13 (no
Xsens sensor) takes the generic row.

Prior persons: the weargait_pd controls of the "fit" set of reports/weargait_frames (outside lockbox
v1.2; that probe was rerun with lockbox v1.2 on 2026-09-28) that have Walk samples in their SelfPace
or HurriedPace IMU records, minus the 14 G3 dev controls of mdfm.data.sources.folds(). Asserted: no
lockbox person (reports/lockbox/lockbox_persons.csv, its SHA-256 checked against
reports/lockbox/lockbox.sha256 and recorded) and no dev person enters, every dev control is present
in the walking records (so the removal acts), the persons lie in the fit set, and the
weargait_frames lockbox flags agree with that lockbox.

xsens13 centre R0 (sensor to segment): the population mounting fit of scripts/g2_renderer_realism.py
(fit_mounting) on the Walk samples of the prior persons' canonical IMU records against the rendered
AMASS walks without mounting (reports/amass_walks/takes.csv, KIT and BMLrub, train archives only,
asserted; the cached renders of G2 version 4). Wahba's problem on two vectors per site, the mean
specific force and the first principal axis of the angular rate. That axis has no sign; where it
lies near the horizontal plane both signs fit the two vectors about equally (the vector residuals
of mounting_sign.csv), so the sign is the one whose rotated rendered signals match the pooled real
signals best in distribution (version 3 of the G2 script). The summary counts the sites where the
two residuals differ by less than SIGN_TIE_DEG and the sites where the residual alone would have
picked the other sign.

Spread per site, around R0, split as in MountPriors: tilt (off the vertical) and yaw (about it).
Tilt: the static up vectors of the prior persons in quiet standing (reports/weargait_frames/
site_gravity.csv, static_ok), angles to their mean; sigma_tilt = median / sqrt(2 ln 2), the median
of a 2D Gaussian's radius. Yaw: on the sites whose walking angular rate has one dominant axis
(median energy share of the first principal axis over the prior persons' rows of
reports/weargait_frames/site_gyro_axis.csv at least the weak_axis_energy threshold of
weargait_frames), one Wahba fit per person on that person's own two vectors (axis sign toward the
population's), the rotation vector of R0^T M_person projected on up_sensor; sigma_yaw =
median |yaw| / 0.6745, the median of a half-normal. On the other sites (head, trunk, lower_back) the
walking axis does not fix the yaw, and sigma_yaw = sigma_tilt. Every statistic that enters the
priors comes from the prior persons (or the ALIGN half, for the device); the fit-set rows of
site_summary.csv, which hold the dev controls, serve only as the comparison target.

Vertical axes: up_sensor = R0^T u_seg, with u_seg the segment's up in the renderer's standing
posture: the mean of R_seg^T e_y over the frames of the KIT walks of reports/amass_walks/takes.csv
where every one of the 22 joints moves below 0.1 m/s for at least 0.5 s [cfg] (the SMPL rest pose
is a T-pose, so its wrist frames do not stand for a standing body). Compared per site with the
fit-set mean up of reports/weargait_frames/site_summary.csv: the dominant axis and the angle
against that site's reported spread (median and p90).

Device: xsens13 white noise per sample and gyroscope bias spread per site and axis from
reports/g2_renderer/device_noise.json (quiet standing of the ALIGN half of G2; asserted: the half
that scripts/g2_device_noise.py draws from the walkers equals mdfm.data.sources.g2_halves(), and it
is disjoint from the dev controls and the lockbox); the accelerometer bias spread is not measured
and keeps the Device default.

Writes reports/mount_priors/: priors.json, sites.csv, mounting_sign.csv, persons.csv, summary.md,
_provenance.json. Needs the mdfm environment; about a minute (the renders come from cache/g2_rendered,
AMASS takes from cache/amass). Run:
<conda>/envs/mdfm/python.exe scripts/build_mount_priors.py
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)

GENERIC_SPREAD_DEG = 10.0      # [cfg] spec 2.3
SIGN_TIE_DEG = 5.0             # [cfg] reporting only: two signs whose vector residuals differ less are a tie
STILL_SPEED_M_S = 0.1          # [cfg] standing frames of the renderer's posture
STILL_MIN_FRAMES = 50          # [cfg] 0.5 s at 100 Hz
RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))
HALF_NORMAL_MEDIAN = 0.6744897501960817
FRAMES = os.path.join(TREE, "reports", "weargait_frames")
PREFIX = "weargait:"


def rows_of(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def json_of(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def align_half(walkers, seed):
    """The ALIGN half as scripts/g2_device_noise.py draws it: the sorted walkers of
    g2_renderer_realism.real_records(), permuted with the G2 seed, first half."""
    persons = sorted(walkers)
    perm = np.random.default_rng(seed).permutation(len(persons))
    return {persons[i] for i in perm[: len(persons) // 2]}


def energy_median(gyro_rows, persons, site):
    """Median energy share of the first principal axis of the walking angular rate over the rows of
    reports/weargait_frames/site_gyro_axis.csv of persons at site (rows without a share skipped)."""
    persons = set(persons)
    x = [float(r["pc1_energy_frac"]) for r in gyro_rows
         if r["site"] == site and PREFIX + r["participant"] in persons and r["pc1_energy_frac"]]
    if not x:
        raise ValueError(f"no walking-axis energy share of a prior person at {site}")
    return float(np.median(x)), len(x)


def lockbox_persons():
    """(person ids, SHA-256) of reports/lockbox/lockbox_persons.csv, the hash checked against
    reports/lockbox/lockbox.sha256."""
    import hashlib
    path = os.path.join(TREE, "reports", "lockbox", "lockbox_persons.csv")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(TREE, "reports", "lockbox", "lockbox.sha256"), encoding="utf-8") as f:
        expected = {ln.split()[1]: ln.split()[0] for ln in f if ln.strip()}["lockbox_persons.csv"]
    if digest != expected:
        raise ValueError(f"lockbox_persons.csv SHA-256 {digest} differs from lockbox.sha256 {expected}")
    return {r["person_id"] for r in rows_of(path)}, digest


def frames_sets():
    """(fit, locked): person ids of the controls of reports/weargait_frames outside and inside the
    lockbox, over its gravity and gyroscope tables."""
    fit, locked = set(), set()
    for name in ("site_gravity.csv", "site_gyro_axis.csv"):
        for r in rows_of(os.path.join(FRAMES, name)):
            (locked if r["in_lockbox"] == "1" else fit).add(PREFIX + r["participant"])
    return fit, locked


def check_persons(persons, fit, lockbox, dev):
    """Refuse a prior person set that holds a lockbox or dev person or leaves the fit set."""
    persons = set(persons)
    bad = sorted(persons & set(lockbox))
    if bad:
        raise ValueError(f"lockbox persons in the mount prior: {bad}")
    bad = sorted(persons & set(dev))
    if bad:
        raise ValueError(f"G3 dev persons in the mount prior: {bad}")
    bad = sorted(persons - set(fit))
    if bad:
        raise ValueError(f"persons outside the weargait_frames fit set: {bad}")
    return sorted(persons)


def angle_deg(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def axis_name(u):
    k = int(np.argmax(np.abs(u)))
    return ("+" if u[k] > 0 else "-") + "XYZ"[k]


def to_so3(R):
    """The nearest rotation in float64 (the fit runs on float32 signals: orthonormal to about 1e-7)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, np.float64))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def standing_up(sites):
    """Segment up (14, 3) of each site in the renderer's standing posture and the counts."""
    import torch
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data import amass_index as I
    from mdfm.data.amass import load_sequence
    from mdfm.data.prep import amass_to_body_clock
    from mdfm.factory.imu_render import site_table
    rows = [r for r in rows_of(os.path.join(TREE, "reports", "amass_walks", "takes.csv")) if r["subset"] == "KIT"]
    acc = {s: np.zeros(3) for s in sites}
    frames, takes, bodies = 0, 0, {}
    for r in rows:
        assert I.check_member(r["subset"], r["take"]) == "train", r["take"]
        seq = load_sequence(r["subset"], r["take"])
        body = bodies.setdefault(r["gender"], SMPLHBody(r["gender"], dtype=torch.float64))
        r6, tr, b = amass_to_body_clock(seq, rate=100.0, cutoff=12.0, body=body)
        traj = BodyTrajectory(body, r6, tr, b)
        with torch.no_grad():
            pos, rot = traj.pose_at(torch.arange(r6.shape[0], dtype=torch.float64) / 100.0)
        pos, rot = pos.numpy(), rot.numpy()
        still = np.linalg.norm(np.gradient(pos, 0.01, axis=0), axis=-1).max(1) < STILL_SPEED_M_S
        keep = np.zeros_like(still)
        start = None
        for i, v in enumerate(np.append(still, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= STILL_MIN_FRAMES:
                    keep[start:i] = True
                start = None
        if not keep.any():
            continue
        takes += 1
        frames += int(keep.sum())
        table = site_table(traj.rest)
        for s in sites:
            acc[s] += rot[keep][:, table[s][0], 1, :].sum(0)     # R^T e_y = row 1 of R
    up = np.stack([acc[s] / np.linalg.norm(acc[s]) for s in sites])
    return up, {"kit_takes": len(rows), "takes_with_standing": takes, "standing_frames": frames}


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Mounting and device priors of the IMU nuisance")
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                   help="MoveAtlas working tree (read-only)")
    p.add_argument("--out", default=os.path.join(TREE, "reports", "mount_priors"))
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(a.moveatlas, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {a.moveatlas} (no INDEX.md)")
    os.environ["MOVEATLAS_ROOT"] = a.moveatlas      # before mdfm.paths is imported
    from mdfm import paths
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    return a


def main():
    a = parse_args()
    from _common import write_provenance
    t0 = time.time()
    import g2_renderer_realism as g2
    import torch
    from mdfm.data import amass_index as I
    from mdfm.data import sources
    from mdfm.factory.imu_render import SITES, Device
    sites13 = g2.SITES13

    # ------------------------------------------------ persons
    lockbox, lockbox_sha = lockbox_persons()
    fit, frames_locked = frames_sets()
    assert frames_locked <= lockbox, sorted(frames_locked - lockbox)
    assert not (fit & lockbox), sorted(fit & lockbox)
    folds = sources.folds()
    wg = folds[folds["source"] == "weargait_pd"]
    dev = set(wg.loc[wg["fold"] == "dev", "person_id"])
    real = g2.real_records()
    walkers = {r["person"] for r in real}
    assert dev <= walkers, sorted(dev - walkers)
    persons = check_persons(walkers - dev, fit, lockbox, dev)
    real = [r for r in real if r["person"] in set(persons)]
    for r in real:
        r["x"] = g2.to_sites13(r["signal"], r["loc"])
    shank = np.concatenate([r["x"][:, sites13.index("shank_L"), 3:6] for r in real])
    assert np.nanpercentile(np.abs(shank), 99) > 20.0, "gyroscope not in deg/s"
    align, _ = sources.g2_halves()
    noise = json_of(os.path.join(TREE, "reports", "g2_renderer", "device_noise.json"))
    assert set(align) == align_half(walkers, sources.G2_SEED), "device-noise ALIGN half differs from g2_halves()"
    assert not (set(align) & dev) and not (set(align) & lockbox) and noise["align_persons"] == len(align)
    print(f"prior persons {len(persons)} (walkers {len(walkers)}, dev {len(dev)}, fit set {len(fit)}, "
          f"lockbox controls {len(frames_locked)})", flush=True)

    # ------------------------------------------------ population fit (G2 fit_mounting, sign by distribution)
    amass_rows = rows_of(os.path.join(TREE, "reports", "amass_walks", "takes.csv"))
    assert all(I.check_member(r["subset"], r["take"]) == "train" for r in amass_rows)
    rend = g2.rendered_takes(mount=None)
    real_x, rend_x = [r["x"] for r in real], [r["x"] for r in rend]
    sign_report = []
    mount, resid = g2.fit_mounting(real_x, rend_x, report=sign_report)
    mount = {name: torch.as_tensor(to_so3(m.numpy())) for name, m in mount.items()}
    g_m, ax_m = g2.site_vectors(rend_x)
    g_r, _ = g2.site_vectors(real_x)
    g2_mount = json_of(os.path.join(TREE, "reports", "g2_renderer", "mounting.json"))
    print(f"mounting fitted on {len(real)} trials against {len(rend)} rendered takes ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------ standing posture of the renderer
    u_seg, stand = standing_up(SITES)
    print(f"standing frames {stand['standing_frames']} in {stand['takes_with_standing']} KIT takes "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------ spreads
    thr = json_of(os.path.join(FRAMES, "_provenance.json"))["thresholds"]
    summ = {r["site"]: r for r in rows_of(os.path.join(FRAMES, "site_summary.csv")) if r["set"] == "fit"}
    gyro_rows = rows_of(os.path.join(FRAMES, "site_gyro_axis.csv"))
    grav = [r for r in rows_of(os.path.join(FRAMES, "site_gravity.csv"))
            if PREFIX + r["participant"] in set(persons) and r["static_ok"] == "1"]
    by_person = {}
    for r in real:
        by_person.setdefault(r["person"], []).append(r["x"])
    table = []
    for s, name in enumerate(sites13):
        M0 = mount[name].numpy()
        k = SITES.index(name)
        up = M0.T @ u_seg[k]
        ups = np.array([[float(r[c]) for c in ("up_x", "up_y", "up_z")] for r in grav if r["site"] == name])
        mean_up = ups.mean(0) / np.linalg.norm(ups.mean(0))
        tilt_static = np.array([angle_deg(x, mean_up) for x in ups])
        sigma_tilt = float(np.median(tilt_static)) / RAYLEIGH_MEDIAN
        a_pop = M0.T @ ax_m[s]
        yaws, totals = [], []
        for p in persons:
            xs = by_person[p]
            f = np.concatenate([x[:, s, 0:3] for x in xs])
            w = np.concatenate([x[:, s, 3:6] for x in xs])
            f, w = f[np.isfinite(f).all(1)], w[np.isfinite(w).all(1)]
            if len(w) < thr["min_walk_samples"] or not len(f):
                continue
            gp = f.mean(0) / np.linalg.norm(f.mean(0))
            _, _, vt = np.linalg.svd(w - w.mean(0), full_matrices=False)
            ap = vt[0] if vt[0] @ a_pop >= 0 else -vt[0]
            th = rotvec(M0.T @ g2.wahba([g_m[s], ax_m[s]], [gp, ap]).T)
            yaws.append(float(np.degrees(th @ up)))
            totals.append(float(np.degrees(np.linalg.norm(th))))
        energy, n_energy = energy_median(gyro_rows, persons, name)
        strong = energy >= thr["weak_axis_energy"]
        sigma_yaw = float(np.median(np.abs(yaws))) / HALF_NORMAL_MEDIAN if strong else sigma_tilt
        ref = np.array([float(summ[name][c]) for c in ("up_mean_x", "up_mean_y", "up_mean_z")])
        ang = angle_deg(up, ref)
        signs = {row["sign"]: row for row in sign_report if row["site"] == name}
        chosen = min(signs, key=lambda sg: signs[sg]["distribution_distance"])
        table.append({
            "site": name, "sign": int(chosen), "distribution_distance_plus": signs[1.0]["distribution_distance"],
            "distribution_distance_minus": signs[-1.0]["distribution_distance"],
            "vector_residual_plus_deg": signs[1.0]["vector_residual_deg"],
            "vector_residual_minus_deg": signs[-1.0]["vector_residual_deg"], "residual_deg": resid[name],
            "angle_to_g2_v4_mount_deg": round(angle_deg_rot(M0, np.array(g2_mount[name])), 2),
            "up_x": round(float(up[0]), 4), "up_y": round(float(up[1]), 4), "up_z": round(float(up[2]), 4),
            "vertical_axis": axis_name(up), "frames_axis": summ[name]["vertical_axis"],
            "angle_to_frames_up_deg": round(ang, 2),
            "frames_spread_median_deg": round(float(summ[name]["up_spread_median_deg"]), 2),
            "frames_spread_p90_deg": round(float(summ[name]["up_spread_p90_deg"]), 2),
            "within_p90": ang <= float(summ[name]["up_spread_p90_deg"]),
            "real_walk_vs_static_deg": round(angle_deg(g_r[s], ref), 2),
            "rendered_walk_vs_standing_deg": round(angle_deg(g_m[s], u_seg[k]), 2),
            "n_static": len(ups), "static_tilt_median_deg": round(float(np.median(tilt_static)), 2),
            "static_tilt_p90_deg": round(float(np.percentile(tilt_static, 90)), 2),
            "sigma_tilt_deg": round(sigma_tilt, 3), "walk_axis_energy": round(energy, 3), "n_energy": n_energy,
            "yaw_source": "walking_axis" if strong else "isotropic", "n_walk": len(yaws),
            "yaw_abs_median_deg": round(float(np.median(np.abs(yaws))), 2),
            "sigma_yaw_deg": round(sigma_yaw, 3),
            "person_angle_median_deg": round(float(np.median(totals)), 2),
            "person_angle_p90_deg": round(float(np.percentile(totals, 90)), 2)})
    rows = {r["site"]: r for r in table}

    # ------------------------------------------------ families
    dev_default = Device()
    noise_site = {r["site"]: r for r in noise["per_site"]}

    def family(fitted_sites):
        out = {k: [] for k in ("R0", "up_sensor", "sigma_tilt_deg", "sigma_yaw_deg", "bias_acc_sd_g",
                               "bias_gyro_sd_dps", "noise_acc_g", "noise_gyro_dps", "fitted", "sign")}
        for k, name in enumerate(SITES):
            fitted = name in fitted_sites
            R0 = mount[name].numpy() if fitted else np.eye(3)
            up = R0.T @ u_seg[k]
            out["R0"].append(R0.tolist())
            out["up_sensor"].append((up / np.linalg.norm(up)).tolist())
            out["sigma_tilt_deg"].append(rows[name]["sigma_tilt_deg"] if fitted else GENERIC_SPREAD_DEG)
            out["sigma_yaw_deg"].append(rows[name]["sigma_yaw_deg"] if fitted else GENERIC_SPREAD_DEG)
            out["bias_acc_sd_g"].append([dev_default.acc_bias_g] * 3)
            out["bias_gyro_sd_dps"].append(noise_site[name]["gyro_bias_sd_dps"] if fitted else [dev_default.gyro_bias_dps] * 3)
            out["noise_acc_g"].append(noise_site[name]["acc_sigma_g"] if fitted else [dev_default.acc_noise_g] * 3)
            out["noise_gyro_dps"].append(noise_site[name]["gyro_sigma_dps"] if fitted else [dev_default.gyro_noise_dps] * 3)
            out["fitted"].append(fitted)
            out["sign"].append(rows[name]["sign"] if fitted else None)
        return out

    priors = {
        "version": 1, "sites": list(SITES), "family_order": ["generic", "xsens13"],
        "families": {"generic": family(()), "xsens13": family(set(sites13))},
        "segment_up_standing": u_seg.tolist(), "standing": stand,
        "persons": {"n": len(persons), "sha256": sources.person_hash(persons), "fit_set": len(fit),
                    "dev_excluded": len(dev), "dev_sha256": sources.person_hash(dev),
                    "lockbox_controls_excluded": len(frames_locked), "lockbox_sha256": lockbox_sha,
                    "trials": len(real),
                    "device_noise_align_persons": len(align)},
        "rendered_takes": len(rend),
        "conventions": "R0 maps sensor to segment coordinates (M = R0 @ dR); up_sensor = R0^T segment_up_standing; "
                       "sigmas in degrees; biases and noise in g and deg/s",
    }
    out = a.out
    with open(os.path.join(out, "priors.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(priors, f, indent=1)
    for fname, data in (("sites.csv", table), ("mounting_sign.csv", sign_report),
                        ("persons.csv", [{"person_id": p} for p in persons])):
        with open(os.path.join(out, fname), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    prov = write_provenance(a, {"runtime_s": round(time.time() - t0, 1), "outputs": sorted(
        ["priors.json", "sites.csv", "mounting_sign.csv", "persons.csv", "summary.md", "_provenance.json"])})
    write_summary(out, priors, table, prov)
    print(f"done in {time.time() - t0:.0f} s")


def angle_deg_rot(A, B):
    c = (np.trace(A.T @ B) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def write_summary(out, priors, table, prov):
    p = priors["persons"]
    outside = [r["site"] for r in table if not r["within_p90"]]
    axes = sum(r["vertical_axis"] == r["frames_axis"] for r in table)
    ties = [r["site"] for r in table
            if abs(r["vector_residual_plus_deg"] - r["vector_residual_minus_deg"]) < SIGN_TIE_DEG]
    by_residual = {r["site"]: 1 if r["vector_residual_plus_deg"] <= r["vector_residual_minus_deg"] else -1
                   for r in table}
    differ = [r["site"] for r in table if by_residual[r["site"]] != r["sign"]]
    md = ["# Mounting priors of the IMU nuisance (task B4.7)", "",
          "Built by scripts/build_mount_priors.py (method in its docstring); read by "
          "mdfm.kernels.nuisance.MountPriors. MoveAtlas " + prov["moveatlas_head"].split(" ")[0] + ".", "",
          f"Prior persons: {p['n']} weargait_pd controls (SHA-256 {p['sha256']}), the {p['fit_set']} controls of the "
          f"reports/weargait_frames fit set minus the {p['dev_excluded']} G3 dev controls (SHA-256 {p['dev_sha256']}); "
          f"the {p['lockbox_controls_excluded']} lockbox controls never enter (lockbox_persons.csv SHA-256 "
          f"{p['lockbox_sha256']}). Walking trials: {p['trials']}; rendered "
          f"AMASS walks: {priors['rendered_takes']}. Device noise from the {p['device_noise_align_persons']} ALIGN-half "
          "persons of G2 (disjoint from the dev controls and the lockbox).",
          f"Standing posture of the renderer: {priors['standing']['standing_frames']} frames in "
          f"{priors['standing']['takes_with_standing']} of {priors['standing']['kit_takes']} KIT walks.", "",
          f"Vertical axes: the prior's standing up axis equals the reports/weargait_frames axis on {axes} of "
          f"{len(table)} sites; the angle to the fit-set mean up lies within that site's p90 spread on "
          f"{len(table) - len(outside)} of {len(table)} sites" + (f" (outside: {', '.join(outside)})." if outside else "."),
          "The head: the weargait walkers' mean specific force in the forehead sensor lies "
          f"{next(r for r in table if r['site'] == 'head')['real_walk_vs_static_deg']} degrees from their quiet-standing up, "
          "the renderer's walkers' "
          f"{next(r for r in table if r['site'] == 'head')['rendered_walk_vs_standing_deg']} degrees from their standing up; "
          "the walking fit carries that difference of head pitch into the head mount.", "",
          f"Gyroscope axis sign: the two signs' vector residuals differ by less than {SIGN_TIE_DEG:g} degrees on "
          f"{len(ties)} of {len(table)} sites ({', '.join(ties) if ties else 'none'}), where only the distribution "
          f"of the signals decides; the distribution criterion picks the sign of the smaller vector residual on "
          f"{len(table) - len(differ)} of {len(table)} sites" + (f" (not on: {', '.join(differ)})." if differ else "."),
          "",
          "| site | sign | distribution distance + / - | vector residual + / - (deg) | vs G2 v4 mount (deg) | up axis "
          "(frames) | angle to frames up (deg) | frames spread median / p90 (deg) | sigma tilt (deg) | sigma yaw (deg), "
          "source | person angle median / p90 (deg) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['site']} | {r['sign']:+d} | {r['distribution_distance_plus']} / {r['distribution_distance_minus']} | "
                  f"{r['vector_residual_plus_deg']} / {r['vector_residual_minus_deg']} | {r['angle_to_g2_v4_mount_deg']} | "
                  f"{r['vertical_axis']} ({r['frames_axis']}) | {r['angle_to_frames_up_deg']} | "
                  f"{r['frames_spread_median_deg']} / {r['frames_spread_p90_deg']} | {r['sigma_tilt_deg']:.2f} | "
                  f"{r['sigma_yaw_deg']:.2f}, {r['yaw_source']} | {r['person_angle_median_deg']} / {r['person_angle_p90_deg']} |")
    iso = [r for r in table if r["yaw_source"] == "isotropic"]
    md += ["", "Generic family and the pocket site of xsens13: identity mount, "
           f"{GENERIC_SPREAD_DEG:g} degrees of tilt and yaw spread [cfg], Device defaults. xsens13 accelerometer bias "
           "spread: the Device default (not measured).",
           "The per-person angle is that of one Wahba fit per person on the person's own walking vectors against R0"
           + ("; on " + ", ".join(f"{r['site']} ({r['walk_axis_energy']})" for r in iso) + " the prior persons' median "
              "energy share of the first principal axis of the walking angular rate lies below the weak-axis threshold "
              "of reports/weargait_frames, so that angle is not a mounting spread and the yaw spread there is the tilt "
              "spread." if iso else ".")]
    text = "\n".join(md) + "\n"
    assert chr(0x2014) not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
