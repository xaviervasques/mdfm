"""Mounting and device priors of the IMU nuisance on the open body (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4):
reports/open_mount_priors/priors.json, in the schema of reports/mount_priors/priors.json (scripts/
build_mount_priors.py; read by mdfm.kernels.nuisance.MountPriors.load(path) and mdfm.data.mounting(path=...)).

The method is that of scripts/build_mount_priors.py (its docstring), whose helpers this script imports. What
depends on the body is recomputed on the open body; what does not is computed by the same code from the same
inputs, and checked equal to v1:

Body-dependent, recomputed:
- segment_up_standing (14, 3), each site's segment up in the renderer's standing posture, and so up_sensor = R0^T
  segment_up_standing in both families. v1 took the still frames of its 343 AMASS KIT walks. Here (STANDING rule):
  every take of the open body store under the residual bound (rms_m at most RMS_BOUND_M), at most
  TAKES_PER_PERSON takes per person drawn without replacement by numpy default_rng(SEED) over the persons in sorted
  order; the take's knots low-passed at 12 Hz as the G2 renders are, the joints and segment rotations by forward
  kinematics on the take's rest joints (P3 basis); a frame stands when every one of the 22 joints moves below
  STILL_SPEED_M_S for at least STILL_MIN_FRAMES (v1's rule) and, because the store holds sitting, squatting,
  one-leg and bent-over tasks that v1's walks did not, the body is upright: both feet in contact (the store's
  contact labels), each knee within UPRIGHT_KNEE_DEG of straight (hip-knee and knee-ankle lines), the
  pelvis-to-neck line within UPRIGHT_TRUNK_DEG of the vertical. Person-weighted: the mean of R_seg^T e_y over a
  person's standing frames, then the mean over the persons with standing frames, normalized. The same rule on the
  G2 open walks alone (v1's choice of takes) is written for comparison.
- R0 of xsens13 and its gyroscope-axis sign: the population fit of g2_renderer_realism.fit_mounting on the prior
  persons' walking trials against the open walks rendered without mounting (scripts/g2_open_render.py,
  cache/g2_rendered_open), in place of the rendered AMASS walks. One rule added: where the fit's two-vector residual
  exceeds GRAVITY_ONLY_RESIDUAL_DEG, no rotation matches the rendered walking axis to the real one (on the open
  store the head's rotation about the neck is not observed by the joint-landmark fit, so its rendered angular rate
  is not a head's), and R0 is the smallest rotation taking the rendered mean specific force onto the real one;
  sites.csv says which sites (column fit). The sign column keeps v1's meaning, relative to the principal axis as
  the SVD orients it, whose orientation is arbitrary: compare R0, not the signs, across bodies.
- sigma_yaw on the walking-axis sites: the per-person Wahba fits against the open renders' site vectors and R0.

Not body-dependent, the same computation as v1 (asserted equal to reports/mount_priors/priors.json):
- the prior persons (54 weargait_pd controls) and their hashes, sigma_tilt (quiet standing of the real persons),
  the walking-axis energy that picks the yaw source;
- the device statistics: noise_acc_g, noise_gyro_dps and bias_gyro_sd_dps of xsens13 from
  reports/g2_renderer/device_noise.json (quiet standing of the G2 ALIGN half), bias_acc_sd_g and the generic family
  from the Device defaults of mdfm.factory.imu_render; the generic family's R0 (identity) and spreads (10 degrees).

The schema is v1's: the same top-level keys, family_order, family keys and nested keys. Two keys keep v1's names
with the open meaning, stated here and in summary.md: standing.kit_takes counts the store takes scanned by the
STANDING rule, rendered_takes the open walks rendered by scripts/g2_open_render.py.

No SMPL+H body is built and no AMASS take is read: the rest joints come from the P3 basis
(mdfm.data.open_body_store.OpenBodyStore.rest), the renders from scripts/g2_open_render.py. (mdfm.body.trajectory
imports the constant N_BETAS of mdfm/body/smplh.py; no SMPL+H array is read through it.)

Writes reports/open_mount_priors/: priors.json, sites.csv, mounting_sign.csv, persons.csv, standing.csv,
summary.md, _provenance.json. Run after scripts/g2_open_render.py --render-only (the unmounted renders):
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_mount_priors.py
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)

import build_mount_priors as V1  # noqa: E402  (helpers only; its main() and standing_up() are v1's)

OUT = os.path.join(TREE, "reports", "open_mount_priors")
V1_DIR = os.path.join(TREE, "reports", "mount_priors")
RMS_BOUND_M = 0.0167
TAKES_PER_PERSON = 8          # [cfg] takes drawn per person for the standing posture
SEED = 20261113               # [cfg] the draw of those takes
STILL_SPEED_M_S = V1.STILL_SPEED_M_S
STILL_MIN_FRAMES = V1.STILL_MIN_FRAMES
UPRIGHT_KNEE_DEG = 20.0       # [cfg] knee within this angle of straight
UPRIGHT_TRUNK_DEG = 20.0      # [cfg] pelvis-to-neck line within this angle of the vertical
BODY_CUTOFF_HZ = 12.0         # as the G2 renders (g2_renderer_realism.BODY_CUTOFF_HZ default)
RATE = 100.0
GRAVITY_ONLY_RESIDUAL_DEG = 45.0   # [cfg] above this two-vector residual, R0 from the specific force alone


def gravity_rotation(a, b):
    """The smallest rotation Q with Q a = b for unit vectors a, b (Rodrigues; a and b not opposite)."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    import subprocess
    r = subprocess.run(["git", "-C", TREE, "log", "-1", "--format=%h"], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------- standing posture on the open body
def standing_takes(store, seed=SEED, per_person=TAKES_PER_PERSON):
    """Take indices of the STANDING rule's draw (module docstring): per person in sorted order, at most per_person
    takes under the residual bound, drawn without replacement."""
    gen = np.random.default_rng(seed)
    out = []
    for key, idx in sorted(store.persons().items()):
        ok = [i for i in idx if store.takes[i]["rms_m"] <= RMS_BOUND_M]
        if not ok:
            continue
        pick = gen.permutation(len(ok))[:per_person]
        out += [ok[k] for k in sorted(pick)]
    return out


def runs_at_least(mask, n):
    keep = np.zeros_like(mask)
    start = None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= n:
                keep[start:i] = True
            start = None
    return keep


def _angle(a, b):
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def take_standing(store, i, rest_cache, sites):
    """(sum over standing frames of R_seg^T e_y per site (S, 3), standing frames, still frames) of take i."""
    import torch
    from mdfm.body.joints import fk
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.data.prep import lowpass
    from mdfm.factory.imu_render import SITE_RULES
    tk = store.take(i)
    key = tuple(store.phenotype(i))
    if key not in rest_cache:
        rest_cache[key] = torch.as_tensor(store.rest(i), dtype=torch.float64)
    r6 = np.ascontiguousarray(lowpass(np.asarray(tk["rot6d"], dtype=np.float64), RATE, BODY_CUTOFF_HZ))
    tr = np.ascontiguousarray(lowpass(np.asarray(tk["transl"], dtype=np.float64), RATE, BODY_CUTOFF_HZ))
    with torch.no_grad():
        pos, rot = fk(rot6d_to_matrix(torch.as_tensor(r6)), torch.as_tensor(tr), rest_cache[key])
    pos, rot = pos.numpy(), rot.numpy()
    if pos.shape[0] < STILL_MIN_FRAMES:
        return np.zeros((len(sites), 3)), 0, 0
    still = np.linalg.norm(np.gradient(pos, 1.0 / RATE, axis=0), axis=-1).max(1) < STILL_SPEED_M_S
    still = runs_at_least(still, STILL_MIN_FRAMES)
    contact = np.asarray(tk["contact"]).astype(bool).all(1)
    knee_l = _angle(pos[:, 4] - pos[:, 1], pos[:, 7] - pos[:, 4]) < UPRIGHT_KNEE_DEG
    knee_r = _angle(pos[:, 5] - pos[:, 2], pos[:, 8] - pos[:, 5]) < UPRIGHT_KNEE_DEG
    trunk = _angle(pos[:, 12] - pos[:, 0], np.array([0.0, 1.0, 0.0])) < UPRIGHT_TRUNK_DEG
    stand = still & contact & knee_l & knee_r & trunk
    acc = np.stack([rot[stand][:, SITE_RULES[s][0], 1, :].sum(0) for s in sites])   # R^T e_y = row 1 of R
    return acc, int(stand.sum()), int(still.sum())


def standing_up_open(store, sites, takes, log=print):
    """Person-weighted segment up (S, 3) over the standing frames of the given takes, the counts, and the rows of
    standing.csv (one per take)."""
    per_person, rows, rest_cache, t0 = {}, [], {}, time.time()
    for k, i in enumerate(takes):
        t = store.takes[i]
        acc, n_stand, n_still = take_standing(store, i, rest_cache, sites)
        rows.append({"take_id": t["take_id"], "source": t["source"], "person_id": t["person_id"],
                     "n_knots": t["n_knots"], "still_frames": n_still, "standing_frames": n_stand})
        if n_stand:
            p = per_person.setdefault((t["source"], t["person_id"]), [np.zeros((len(sites), 3)), 0])
            p[0] += acc
            p[1] += n_stand
        if (k + 1) % 500 == 0:
            log(f"  standing: {k + 1} of {len(takes)} takes ({time.time() - t0:.0f} s)")
    if not per_person:
        raise ValueError("no standing frame in the drawn takes")
    m = np.mean([a / n for a, n in per_person.values()], axis=0)
    up = m / np.linalg.norm(m, axis=1, keepdims=True)
    stand = {"kit_takes": len(takes), "takes_with_standing": int(sum(r["standing_frames"] > 0 for r in rows)),
             "standing_frames": int(sum(r["standing_frames"] for r in rows))}
    extra = {"persons_with_standing": len(per_person), "still_frames": int(sum(r["still_frames"] for r in rows)),
             "persons_drawn": len({(r["source"], r["person_id"]) for r in rows})}
    return up, stand, extra, rows


# ---------------------------------------------------------------- the priors
def build(log=print):
    """Compute the open priors (module docstring): returns (priors dict, sites table, sign report, persons,
    standing rows, comparison extras). Imports no SMPL+H body and reads no AMASS take."""
    import torch
    import g2_open_render as GO
    from mdfm.data import sources
    from mdfm.factory.imu_render import SITES, Device
    G = GO.g2()
    t0 = time.time()
    sites13 = G.SITES13

    # ------------------------------------------------ persons (as v1)
    lockbox, lockbox_sha = V1.lockbox_persons()
    fit, frames_locked = V1.frames_sets()
    assert frames_locked <= lockbox, sorted(frames_locked - lockbox)
    assert not (fit & lockbox), sorted(fit & lockbox)
    folds = sources.folds()
    wg = folds[folds["source"] == "weargait_pd"]
    dev = set(wg.loc[wg["fold"] == "dev", "person_id"])
    real = G.real_records()
    walkers = {r["person"] for r in real}
    assert dev <= walkers, sorted(dev - walkers)
    persons = V1.check_persons(walkers - dev, fit, lockbox, dev)
    real = [r for r in real if r["person"] in set(persons)]
    for r in real:
        r["x"] = G.to_sites13(r["signal"], r["loc"])
    shank = np.concatenate([r["x"][:, sites13.index("shank_L"), 3:6] for r in real])
    assert np.nanpercentile(np.abs(shank), 99) > 20.0, "gyroscope not in deg/s"
    align, _ = sources.g2_halves()
    noise = V1.json_of(os.path.join(TREE, "reports", "g2_renderer", "device_noise.json"))
    assert set(align) == V1.align_half(walkers, sources.G2_SEED), "device-noise ALIGN half differs from g2_halves()"
    assert not (set(align) & dev) and not (set(align) & lockbox) and noise["align_persons"] == len(align)
    log(f"prior persons {len(persons)} ({time.time() - t0:.0f} s)")

    # ------------------------------------------------ population fit against the open renders
    store = GO.open_store()
    walks = GO.walk_takes(store)
    rend = GO.rendered_takes_open(None, store, walks, log=log)
    real_x, rend_x = [r["x"] for r in real], [r["x"] for r in rend]
    sign_report = []
    mount, resid = G.fit_mounting(real_x, rend_x, report=sign_report)
    mount = {name: torch.as_tensor(V1.to_so3(m.numpy())) for name, m in mount.items()}
    g_m0, ax_m0 = G.site_vectors(rend_x)
    g_r0, ax_r0 = G.site_vectors(real_x)
    fit_kind, resid_two = {}, dict(resid)
    for s, name in enumerate(sites13):
        fit_kind[name] = "two_vectors"
        if resid[name] > GRAVITY_ONLY_RESIDUAL_DEG:
            # the rendered and real walking axes do not match under any rotation (the open fit does not observe
            # this segment's rotation about its own axis): R0 from the mean specific force alone, the smallest
            # rotation, and the residual of both vectors under it
            Q = gravity_rotation(g_m0[s], g_r0[s])
            mount[name] = torch.as_tensor(V1.to_so3(Q.T))
            resid[name] = round(float(V1.angle_deg(Q @ g_m0[s], g_r0[s]) + np.degrees(np.arccos(np.clip(
                abs(Q @ ax_m0[s] @ ax_r0[s]), -1.0, 1.0)))), 2)
            fit_kind[name] = "gravity_only"
    g_m, ax_m = G.site_vectors(rend_x)
    g_r, _ = G.site_vectors(real_x)
    log(f"mounting fitted on {len(real)} trials against {len(rend)} open renders ({time.time() - t0:.0f} s)")

    # ------------------------------------------------ standing posture of the open renderer
    draw = standing_takes(store)
    u_seg, stand, stand_extra, stand_rows = standing_up_open(store, SITES, draw, log=log)
    walk_idx = [r["index"] for r in walks if not r["over_bound"]]
    u_walk, stand_walk, walk_extra, _ = standing_up_open(store, SITES, walk_idx, log=log)
    log(f"standing frames {stand['standing_frames']} in {stand['takes_with_standing']} of {len(draw)} takes "
        f"({time.time() - t0:.0f} s)")

    # ------------------------------------------------ spreads (v1's loop, open renders and posture)
    thr = V1.json_of(os.path.join(V1.FRAMES, "_provenance.json"))["thresholds"]
    summ = {r["site"]: r for r in V1.rows_of(os.path.join(V1.FRAMES, "site_summary.csv")) if r["set"] == "fit"}
    gyro_rows = V1.rows_of(os.path.join(V1.FRAMES, "site_gyro_axis.csv"))
    grav = [r for r in V1.rows_of(os.path.join(V1.FRAMES, "site_gravity.csv"))
            if V1.PREFIX + r["participant"] in set(persons) and r["static_ok"] == "1"]
    by_person = {}
    for r in real:
        by_person.setdefault(r["person"], []).append(r["x"])
    v1_pri = V1.json_of(os.path.join(V1_DIR, "priors.json"))
    v1_R0 = {s: np.array(v1_pri["families"]["xsens13"]["R0"][k]) for k, s in enumerate(v1_pri["sites"])}
    g2o_path = os.path.join(TREE, "reports", "g2_open", "mounting.json")
    g2o = V1.json_of(g2o_path) if os.path.exists(g2o_path) else None
    table = []
    for s, name in enumerate(sites13):
        M0 = mount[name].numpy()
        k = SITES.index(name)
        up = M0.T @ u_seg[k]
        ups = np.array([[float(r[c]) for c in ("up_x", "up_y", "up_z")] for r in grav if r["site"] == name])
        mean_up = ups.mean(0) / np.linalg.norm(ups.mean(0))
        tilt_static = np.array([V1.angle_deg(x, mean_up) for x in ups])
        sigma_tilt = float(np.median(tilt_static)) / V1.RAYLEIGH_MEDIAN
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
            th = V1.rotvec(M0.T @ G.wahba([g_m[s], ax_m[s]], [gp, ap]).T)
            yaws.append(float(np.degrees(th @ up)))
            totals.append(float(np.degrees(np.linalg.norm(th))))
        energy, n_energy = V1.energy_median(gyro_rows, persons, name)
        strong = energy >= thr["weak_axis_energy"]
        sigma_yaw = float(np.median(np.abs(yaws))) / V1.HALF_NORMAL_MEDIAN if strong else sigma_tilt
        ref = np.array([float(summ[name][c]) for c in ("up_mean_x", "up_mean_y", "up_mean_z")])
        ang = V1.angle_deg(up, ref)
        signs = {row["sign"]: row for row in sign_report if row["site"] == name}
        chosen = min(signs, key=lambda sg: signs[sg]["distribution_distance"])
        table.append({
            "site": name, "sign": int(chosen), "distribution_distance_plus": signs[1.0]["distribution_distance"],
            "distribution_distance_minus": signs[-1.0]["distribution_distance"],
            "vector_residual_plus_deg": signs[1.0]["vector_residual_deg"],
            "vector_residual_minus_deg": signs[-1.0]["vector_residual_deg"], "residual_deg": resid[name],
            "fit": fit_kind[name], "two_vector_residual_deg": resid_two[name],
            "angle_to_v1_R0_deg": round(V1.angle_deg_rot(M0, v1_R0[name]), 2),
            "angle_to_g2_open_mount_deg": (round(V1.angle_deg_rot(M0, np.array(g2o[name])), 2) if g2o else ""),
            "up_x": round(float(up[0]), 4), "up_y": round(float(up[1]), 4), "up_z": round(float(up[2]), 4),
            "vertical_axis": V1.axis_name(up), "frames_axis": summ[name]["vertical_axis"],
            "angle_to_frames_up_deg": round(ang, 2),
            "frames_spread_median_deg": round(float(summ[name]["up_spread_median_deg"]), 2),
            "frames_spread_p90_deg": round(float(summ[name]["up_spread_p90_deg"]), 2),
            "within_p90": ang <= float(summ[name]["up_spread_p90_deg"]),
            "real_walk_vs_static_deg": round(V1.angle_deg(g_r[s], ref), 2),
            "rendered_walk_vs_standing_deg": round(V1.angle_deg(g_m[s], u_seg[k]), 2),
            "rendered_axis_to_vertical_deg": round(float(np.degrees(np.arccos(np.clip(
                abs(ax_m[s] @ u_seg[k]) / np.linalg.norm(ax_m[s]) / np.linalg.norm(u_seg[k]), 0.0, 1.0)))), 2),
            "n_static": len(ups), "static_tilt_median_deg": round(float(np.median(tilt_static)), 2),
            "static_tilt_p90_deg": round(float(np.percentile(tilt_static, 90)), 2),
            "sigma_tilt_deg": round(sigma_tilt, 3), "walk_axis_energy": round(energy, 3), "n_energy": n_energy,
            "yaw_source": "walking_axis" if strong else "isotropic", "n_walk": len(yaws),
            "yaw_abs_median_deg": round(float(np.median(np.abs(yaws))), 2),
            "sigma_yaw_deg": round(sigma_yaw, 3),
            "person_angle_median_deg": round(float(np.median(totals)), 2),
            "person_angle_p90_deg": round(float(np.percentile(totals, 90)), 2)})
    rows = {r["site"]: r for r in table}

    # ------------------------------------------------ families (v1's code)
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
            out["sigma_tilt_deg"].append(rows[name]["sigma_tilt_deg"] if fitted else V1.GENERIC_SPREAD_DEG)
            out["sigma_yaw_deg"].append(rows[name]["sigma_yaw_deg"] if fitted else V1.GENERIC_SPREAD_DEG)
            out["bias_acc_sd_g"].append([dev_default.acc_bias_g] * 3)
            out["bias_gyro_sd_dps"].append(noise_site[name]["gyro_bias_sd_dps"] if fitted
                                           else [dev_default.gyro_bias_dps] * 3)
            out["noise_acc_g"].append(noise_site[name]["acc_sigma_g"] if fitted else [dev_default.acc_noise_g] * 3)
            out["noise_gyro_dps"].append(noise_site[name]["gyro_sigma_dps"] if fitted
                                         else [dev_default.gyro_noise_dps] * 3)
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
                    "trials": len(real), "device_noise_align_persons": len(align)},
        "rendered_takes": len(rend),
        "conventions": "R0 maps sensor to segment coordinates (M = R0 @ dR); up_sensor = R0^T segment_up_standing; "
                       "sigmas in degrees; biases and noise in g and deg/s; open body (Anny through the P3 basis): "
                       "standing.kit_takes counts the open body store takes scanned for the standing posture, "
                       "rendered_takes the open walks of scripts/g2_open_render.py",
    }
    extras = {"store": store, "walks": walks, "standing_extra": stand_extra, "u_walk": u_walk,
              "standing_walks": stand_walk, "standing_walks_extra": walk_extra, "g2_open_mounting": g2o is not None}
    return priors, table, sign_report, persons, stand_rows, extras


# ---------------------------------------------------------------- comparison with v1
def same_value(a, b, tol=0.0):
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=0.0, atol=tol, equal_nan=True))


def compare_v1(priors, table):
    """Per site and per quantity, open against reports/mount_priors/priors.json and sites.csv."""
    v1 = V1.json_of(os.path.join(V1_DIR, "priors.json"))
    v1_sites = {r["site"]: r for r in V1.rows_of(os.path.join(V1_DIR, "sites.csv"))}
    sites = priors["sites"]
    out = {"per_site": [], "device_equal": {}, "persons_equal": priors["persons"] == v1["persons"]}
    for key in ("bias_acc_sd_g", "bias_gyro_sd_dps", "noise_acc_g", "noise_gyro_dps", "fitted"):
        out["device_equal"][key] = all(same_value(priors["families"][f][key], v1["families"][f][key])
                                       for f in priors["family_order"])
    out["generic_spreads_equal"] = all(same_value(priors["families"]["generic"][k], v1["families"]["generic"][k])
                                       for k in ("R0", "sigma_tilt_deg", "sigma_yaw_deg"))
    rows = {r["site"]: r for r in table}
    for k, s in enumerate(sites):
        uo, uv = np.array(priors["segment_up_standing"][k]), np.array(v1["segment_up_standing"][k])
        r = {"site": s, "segment_up_angle_deg": round(V1.angle_deg(uo, uv), 2)}
        for f in ("generic", "xsens13"):
            r[f"up_sensor_{f}_angle_deg"] = round(V1.angle_deg(priors["families"][f]["up_sensor"][k],
                                                               v1["families"][f]["up_sensor"][k]), 2)
        if s in rows:
            o, w = rows[s], v1_sites[s]
            r.update({"R0_angle_deg": o["angle_to_v1_R0_deg"], "sign_open": o["sign"], "sign_v1": int(w["sign"]),
                      "fit": o["fit"], "rendered_axis_to_vertical_deg": o["rendered_axis_to_vertical_deg"],
                      "residual_open_deg": o["residual_deg"], "residual_v1_deg": float(w["residual_deg"]),
                      "sigma_tilt_open_deg": o["sigma_tilt_deg"], "sigma_tilt_v1_deg": float(w["sigma_tilt_deg"]),
                      "sigma_yaw_open_deg": o["sigma_yaw_deg"], "sigma_yaw_v1_deg": float(w["sigma_yaw_deg"]),
                      "yaw_source_open": o["yaw_source"], "yaw_source_v1": w["yaw_source"],
                      "angle_to_frames_up_open_deg": o["angle_to_frames_up_deg"],
                      "angle_to_frames_up_v1_deg": float(w["angle_to_frames_up_deg"]),
                      "within_p90_open": o["within_p90"], "within_p90_v1": w["within_p90"] == "True",
                      "rendered_walk_vs_standing_open_deg": o["rendered_walk_vs_standing_deg"],
                      "rendered_walk_vs_standing_v1_deg": float(w["rendered_walk_vs_standing_deg"])})
        out["per_site"].append(r)
    out["sigma_tilt_equal"] = all(abs(r["sigma_tilt_open_deg"] - r["sigma_tilt_v1_deg"]) < 1e-9
                                  for r in out["per_site"] if "sigma_tilt_open_deg" in r)
    out["v1_standing"] = v1["standing"]
    out["v1_rendered_takes"] = v1["rendered_takes"]
    return out


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_summary(out, priors, table, cmp, extras, prov):
    p = priors["persons"]
    st, sx = priors["standing"], extras["standing_extra"]
    sw, swx = extras["standing_walks"], extras["standing_walks_extra"]
    v1s = cmp["v1_standing"]
    walk_ang = [round(V1.angle_deg(a, b), 2) for a, b in zip(extras["u_walk"], priors["segment_up_standing"])]
    outside = [r["site"] for r in table if not r["within_p90"]]
    axes = sum(r["vertical_axis"] == r["frames_axis"] for r in table)
    flips = [r["site"] for r in cmp["per_site"] if "sign_open" in r and r["sign_open"] != r["sign_v1"]]
    ties = [r["site"] for r in table
            if abs(r["vector_residual_plus_deg"] - r["vector_residual_minus_deg"]) < V1.SIGN_TIE_DEG]
    L = ["# Mounting priors of the IMU nuisance on the open body (step M4)", "",
         f"Built by scripts/build_open_mount_priors.py (method in its docstring), MD_FM {prov['mdfm_head']}, MoveAtlas "
         f"{prov['moveatlas_head'].split(' ')[0]}, the open body store {prov['store_root']}; v1 is "
         "reports/mount_priors/ (scripts/build_mount_priors.py, SMPL+H, AMASS KIT standing and KIT and BMLrub walks). "
         "Same schema as v1 (priors.json keys, family_order, family keys); read with "
         "MountPriors.load(path) and mdfm.data.mounting(path=...).", "",
         "## What is recomputed and what is not", "",
         f"Recomputed on the open body: segment_up_standing and so up_sensor in both families, R0 and the gyroscope "
         f"axis sign of xsens13, sigma_yaw of the walking-axis sites. The renders the fit reads: {priors['rendered_takes']} "
         f"open walks of scripts/g2_open_render.py (v1: {cmp['v1_rendered_takes']} AMASS walks).",
         f"The same computation from the same inputs, equal to v1: the prior persons ({p['n']} weargait_pd controls, "
         f"same hashes: {cmp['persons_equal']}), sigma_tilt (equal on the 13 sites: {cmp['sigma_tilt_equal']}), the "
         "device statistics (" + ", ".join(f"{k} {v}" for k, v in cmp["device_equal"].items()) + "; from "
         "reports/g2_renderer/device_noise.json and the Device defaults of mdfm.factory.imu_render), the generic "
         f"family's identity mount and 10 degree spreads (equal: {cmp['generic_spreads_equal']}).",
         "Not ported from a surface: nothing in these priors reads a body surface. The IMU site offsets of the renders "
         "are imu_render.site_table on the open rest joints (fractions of the rest bones plus the constant surface "
         "offsets of SITE_RULES); reports/open_layouts/summary.md compares them with v1 and with Anny's skin.", "",
         "## Standing posture", "",
         f"STANDING rule (docstring): {st['kit_takes']} store takes drawn ({sx['persons_drawn']} persons, at most "
         f"{TAKES_PER_PERSON} per person, seed {SEED}, residual bound {RMS_BOUND_M} m); still frames {sx['still_frames']}, "
         f"of which standing upright {st['standing_frames']} in {st['takes_with_standing']} takes of "
         f"{sx['persons_with_standing']} persons (person-weighted). v1: {v1s['standing_frames']} frames in "
         f"{v1s['takes_with_standing']} of {v1s['kit_takes']} KIT walks.",
         f"The same rule on the open G2 walks alone (v1's choice of takes): {sw['standing_frames']} frames in "
         f"{sw['takes_with_standing']} of {sw['kit_takes']} takes, {swx['persons_with_standing']} persons; angle to "
         f"the rule's segment up per site (degrees): " + ", ".join(f"{s} {a}" for s, a in zip(priors["sites"], walk_ang)) + ".",
         "", "| site | segment up open against v1 (deg) | up_sensor generic (deg) | up_sensor xsens13 (deg) |",
         "|---|---|---|---|"]
    for r in cmp["per_site"]:
        L.append(f"| {r['site']} | {r['segment_up_angle_deg']} | {r['up_sensor_generic_angle_deg']} | "
                 f"{r['up_sensor_xsens13_angle_deg']} |")
    L += ["", "## xsens13, open against v1", "",
          f"Vertical axes: the open standing up axis equals the reports/weargait_frames axis on {axes} of {len(table)} "
          f"sites; the angle to the fit-set mean up lies within that site's p90 spread on {len(table) - len(outside)} "
          f"of {len(table)} sites" + (f" (outside: {', '.join(outside)})." if outside else ".") +
          f" The sign column differs from v1 on {len(flips)} sites" + (f" ({', '.join(flips)})" if flips else "") +
          " (it is relative to the principal axis as the SVD orients it, arbitrary in each run: R0 is what compares)"
          f"; the two signs' vector residuals differ by less than {V1.SIGN_TIE_DEG:g} degrees on {len(ties)} sites"
          + (f" ({', '.join(ties)})." if ties else ".") +
          f" R0 from the specific force alone (two-vector residual above {GRAVITY_ONLY_RESIDUAL_DEG:g} degrees): "
          + (", ".join(f"{r['site']} (two-vector residual {r['two_vector_residual_deg']} degrees)" for r in table
                       if r["fit"] == "gravity_only") or "none") + ".", "",
          "R0 open against v1 splits into the tilt of the sensor's up axis (the up_sensor column) and a rotation "
          "about that axis. Where the rendered walking axis lies near the vertical (last column, open renders), the "
          "two vectors of the fit are nearly parallel and leave that rotation to the small remainder: there a large R0 "
          "angle with a small up_sensor angle is a heading the data do not fix, in v1 as here.", "",
          "| site | fit | R0 open vs v1 (deg) | up_sensor open vs v1 (deg) | sign open / v1 | residual open / v1 (deg) | "
          "sigma yaw open / v1 (deg) | yaw source | angle to frames up open / v1 (deg) | rendered walk vs standing "
          "open / v1 (deg) | open rendered walking axis to the vertical (deg) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in cmp["per_site"]:
        if "R0_angle_deg" not in r:
            continue
        L.append(f"| {r['site']} | {r['fit']} | {r['R0_angle_deg']} | {r['up_sensor_xsens13_angle_deg']} | "
                 f"{r['sign_open']:+d} / {r['sign_v1']:+d} | "
                 f"{r['residual_open_deg']} / {r['residual_v1_deg']} | {r['sigma_yaw_open_deg']:.2f} / "
                 f"{r['sigma_yaw_v1_deg']:.2f} | {r['yaw_source_open']} | {r['angle_to_frames_up_open_deg']} / "
                 f"{r['angle_to_frames_up_v1_deg']} | {r['rendered_walk_vs_standing_open_deg']} / "
                 f"{r['rendered_walk_vs_standing_v1_deg']} | {r['rendered_axis_to_vertical_deg']} |")
    if extras["g2_open_mounting"]:
        L += ["", "Angle of R0 to the G2 open ALIGN-half fit (reports/g2_open/mounting.json), degrees: " +
              ", ".join(f"{r['site']} {r['angle_to_g2_open_mount_deg']}" for r in table) + "."]
    L += ["", f"priors.json SHA-256 {prov['priors_sha256']}.", ""]
    text = "\n".join(L)
    assert chr(0x2014) not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(a.moveatlas, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {a.moveatlas} (no INDEX.md)")
    os.environ["MOVEATLAS_ROOT"] = a.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    from _common import moveatlas_head
    from mdfm import paths
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    priors, table, sign_report, persons, stand_rows, extras = build(log=lambda *x: print(*x, flush=True))
    with open(os.path.join(a.out, "priors.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(priors, f, indent=1)
    write_csv(os.path.join(a.out, "sites.csv"), table)
    write_csv(os.path.join(a.out, "mounting_sign.csv"), sign_report)
    write_csv(os.path.join(a.out, "persons.csv"), [{"person_id": x} for x in persons])
    write_csv(os.path.join(a.out, "standing.csv"), stand_rows)
    cmp = compare_v1(priors, table)
    store = extras["store"]
    prov = {"script": "scripts/build_open_mount_priors.py", "moveatlas_root": a.moveatlas,
            "moveatlas_head": moveatlas_head(a.moveatlas), "mdfm_head": git_head(), "body": "open (Anny, P3 basis)",
            "store_root": store.root,
            "store_takes_sha256": {k: v for k, v in store.hashes["files"].items() if k.endswith("takes.csv")},
            "renders": "cache/g2_rendered_open (scripts/g2_open_render.py)",
            "rules": {"rms_bound_m": RMS_BOUND_M, "takes_per_person": TAKES_PER_PERSON, "seed": SEED,
                      "still_speed_m_s": STILL_SPEED_M_S, "still_min_frames": STILL_MIN_FRAMES,
                      "upright_knee_deg": UPRIGHT_KNEE_DEG, "upright_trunk_deg": UPRIGHT_TRUNK_DEG,
                      "body_cutoff_hz": BODY_CUTOFF_HZ},
            "v1_priors_sha256": file_sha256(os.path.join(V1_DIR, "priors.json")),
            "priors_sha256": file_sha256(os.path.join(a.out, "priors.json")),
            "comparison": {k: v for k, v in cmp.items() if k != "per_site"},
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "runtime_s": round(time.time() - t0, 1),
            "outputs": ["_provenance.json", "mounting_sign.csv", "persons.csv", "priors.json", "sites.csv",
                        "standing.csv", "summary.md"]}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    write_summary(a.out, priors, table, cmp, extras, prov)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
