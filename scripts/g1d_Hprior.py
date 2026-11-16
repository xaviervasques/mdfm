"""G1D-Hprior: do the motion prior and the refinement pull the decoded arms of the six-IMU body toward the training
mean pose, with an under-dispersed posterior? Development diagnostic on the AMASS dev archives, not a gate read.

The prediction and its criteria were written and hashed before any error value was read:
reports/weekend_2026-10-01/tasks/G1D-Hprior.md, section Prediction (hash in
reports/bench/g1_diag/Hprior/prediction.sha256).

Arm angles (arm_angles), from joint positions alone, so they need no frame alignment: trunk down axis d = pelvis
minus neck; per side the upper arm u = elbow minus shoulder and the forearm f = wrist minus elbow; shoulder
elevation = angle(u, d) (0 deg hanging, 90 deg horizontal); elbow flexion = angle(u, f) (0 deg straight). Order
ANGLES: shoulder_L, shoulder_R, elbow_L, elbow_R. The training mean arm pose: the same angles of the mean pose of
reports/gates/g1/population_prior/prior.json (train persons only, pelvis-relative heading-frame joint means).

Stages (default out reports/bench/g1_diag/Hprior):
- attain: truth only, no model and no error value: the window sets, the truth angle spreads, the spread of the
  truth around the training mean arm pose, the calib truth spreads, the train wrist-draw spread of prior.json,
  and the columns of the decompose windows.csv. Writes attain.json.
- run (CPU): the main set (up to PER_PERSON windows per dev person, seed SEED) and the partial set (the first 2
  takes of each dev archive, the refine60 set of reports/bench/g1_decompose_*): encode (probe_g1_decompose.
  posterior_pass, shape pooled per take, blocks of whole takes up to BLOCK windows), the posterior mean decoded at
  0 and 20 refinement steps (60 on the partial set), DRAWS latent draws of the 20-step posterior (z and the pooled
  beta, as the decompose best-of-16), and on the main set two ancestral samples of the motion prior (the
  condition of the window, beta the shape prior mean of its gender, z patch by patch with noise) and its noiseless
  mean path. Per window: arm angles every STRIDE knots, the left wrist in the pelvis-heading frame (prior, truth),
  G1a and the arm-joint error (probe_g1_decompose.g1a_parts). Writes run_windows.csv, angles.npz, run.json.
- read: the statistics and the criteria of the prediction (C1 to C4), the decompose-file dispersion ratios, and
  the verdict. Writes results.json and summary.md.

Data: dev windows (scored), calib truth (attain, spreads only), the train statistics of prior.json; G1Set refuses
the lockbox; no EVAL, CODY or care_pd record. Run (mdfm environment, CPU):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \
    <conda>/envs/mdfm/python.exe scripts/g1d_Hprior.py --stage attain|run|read
"""
import argparse
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (TREE, os.path.join(TREE, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g1_diag", "Hprior")
DECOMPOSE = os.path.join(TREE, "reports", "bench", "g1_decompose_phaseA_root_full_ckpt_B_032044")
PRIOR_JSON = os.path.join(TREE, "reports", "gates", "g1", "population_prior", "prior.json")
CHECKPOINT = os.path.join(TREE, "cache", "runs", "phaseA_root_full", "ckpt_B_032044.pt")
SEED = 20261006
PER_PERSON = 5
PARTIAL_TAKES = 2
STRIDE = 20             # knots between two angle samples (0.2 s)
DRAWS = 16
BLOCK = 8
STEPS = (0, 20, 60)
PRIOR_SAMPLES = 2
BOOT_N = 1000
MM = 1000.0
ANGLES = ("shoulder_L", "shoulder_R", "elbow_L", "elbow_R")
PELVIS, NECK = 0, 12
ARM = {"L": (16, 18, 20), "R": (17, 19, 21)}     # shoulder, elbow, wrist joints (SMPL body order)
ARM_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21)    # collars, shoulders, elbows, wrists (decompose group 'arms')
COVERAGE_CALIBRATED = 15.0 / 17.0                # truth inside the range of 16 exchangeable draws

# criteria of the prediction (section Prediction of the task report), never changed after a read
C1_UNDER = 0.50          # range coverage at most this: under-dispersed
C1_NOT = 0.75            # at least this: not under-dispersed
C2_B_MAIN, C2_SHARE_MAIN = 0.80, 0.50
C2_B_NOT, C2_SHARE_NOT = 0.90, 0.20
C3_DROP, C3_FLAT = 0.05, 0.02
C4_MAIN, C4_NOT = 0.80, 0.90


# ---------------------------------------------------------------- pure functions
def _angle(a, b):
    """Angle in degrees between vectors a and b (..., 3)."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def arm_angles(J):
    """(..., 4) degrees [shoulder_L, shoulder_R, elbow_L, elbow_R] of joints J (..., 22, 3) (module docstring)."""
    J = np.asarray(J, dtype=np.float64)
    d = J[..., PELVIS, :] - J[..., NECK, :]
    out = []
    for side in ("L", "R"):
        s, e, w = ARM[side]
        out.append(_angle(J[..., e, :] - J[..., s, :], d))
    for side in ("L", "R"):
        s, e, w = ARM[side]
        out.append(_angle(J[..., e, :] - J[..., s, :], J[..., w, :] - J[..., e, :]))
    return np.stack(out, -1)


def pull_stats(d, t, m):
    """Shrinkage of decoded values d toward a reference m against the truth t (1-D arrays, one angle):
    b the least squares slope of d on t; sd_ratio sd(d) / sd(t); lam the no-intercept fit of (d - t) on (m - t)
    (the fraction of the way toward m); share the part of sum (d - t)^2 that this pull explains (0 when lam <= 0);
    rmse; mean_d, mean_t."""
    d, t = np.asarray(d, dtype=np.float64), np.asarray(t, dtype=np.float64)
    vt = float(np.var(t))
    b = float(np.mean((d - d.mean()) * (t - t.mean())) / vt) if vt > 0 else float("nan")
    e, g = d - t, m - t
    sg, se = float(np.sum(g * g)), float(np.sum(e * e))
    lam = float(np.sum(e * g) / sg) if sg > 0 else float("nan")
    share = float(np.sum(e * g) ** 2 / (sg * se)) if (sg > 0 and se > 0 and lam > 0) else 0.0
    vd = float(np.var(d))
    # b_rev, the slope of the truth on the decoded value (added after the prediction hash, not a criterion): about
    # 1 for a calibrated conditional mean, whose shrinkage only reflects missing information; above 1 when the
    # decoded value is shrunk more than its information warrants
    b_rev = float(np.mean((d - d.mean()) * (t - t.mean())) / vd) if vd > 0 else float("nan")
    return {"b": b, "b_rev": b_rev, "sd_ratio": float(np.std(d) / math.sqrt(vt)) if vt > 0 else float("nan"), "lam": lam,
            "share": share, "rmse": float(math.sqrt(se / max(len(d), 1))), "mean_d": float(d.mean()),
            "mean_t": float(t.mean()), "n": int(len(d))}


def within(x, wid):
    """x (n,) minus the mean of its window (wid (n,) window ids)."""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    for w in np.unique(wid):
        k = wid == w
        out[k] = x[k] - x[k].mean()
    return out


def draw_calibration(draws, t, post_mean):
    """Calibration of draws (m, n) against truths t (n,): coverage of the draws' range [min, max] (exchangeable
    draws and truth: (m - 1) / (m + 1)), median |z| with z = (t - draw mean) / draw sd (ddof 1), the ratio of the RMS
    draw sd to the RMS error of post_mean (n,) (about 1 for a calibrated Gaussian posterior), the mean normalized
    rank of the truth among the draws."""
    D = np.asarray(draws, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    inside = (t >= D.min(0)) & (t <= D.max(0))
    sd = D.std(0, ddof=1)
    z = (t - D.mean(0)) / np.maximum(sd, 1e-9)
    err = np.asarray(post_mean, dtype=np.float64) - t
    rank = (D < t[None]).sum(0) / D.shape[0]
    return {"range_coverage": float(inside.mean()), "median_abs_z": float(np.median(np.abs(z))),
            "sd_over_rmse": float(math.sqrt(np.mean(sd ** 2)) / max(math.sqrt(np.mean(err ** 2)), 1e-12)),
            "rms_sd": float(math.sqrt(np.mean(sd ** 2))), "rms_err": float(math.sqrt(np.mean(err ** 2))),
            "mean_rank": float(rank.mean()), "n": int(t.size)}


def knot_weighted(values, n):
    v, n = np.asarray(values, dtype=np.float64), np.asarray(n, dtype=np.float64)
    return float(np.sum(v * n) / np.sum(n))


def boot_persons(persons, fn, n=BOOT_N, seed=SEED):
    """95 percent person-clustered bootstrap interval of fn(list of persons) -> float."""
    rng = np.random.default_rng(seed)
    ps = np.asarray(sorted(set(persons)))
    vals = []
    for _ in range(int(n)):
        v = fn(list(rng.choice(ps, size=len(ps), replace=True)))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reading_c1(cov):
    if cov <= C1_UNDER:
        return "UNDER_DISPERSED"
    if cov >= C1_NOT:
        return "NOT_UNDER_DISPERSED"
    return "BETWEEN"


def reading_c2(b, share):
    if b <= C2_B_MAIN and share >= C2_SHARE_MAIN:
        return "MAIN"
    if b >= C2_B_NOT or share < C2_SHARE_NOT:
        return "NOT_MAIN"
    return "BETWEEN"


def reading_c3(b0, b20, b60):
    if b60 <= b20 - C3_DROP and b20 <= b0:
        return "PULLS"
    if abs(b60 - b20) < C3_FLAT:
        return "FLAT"
    return "BETWEEN"


def reading_c4(angle_ratio, wrist_ratio):
    if angle_ratio <= C4_MAIN and wrist_ratio <= C4_MAIN:
        return "PRIOR_NARROW"
    if angle_ratio >= C4_NOT and wrist_ratio >= C4_NOT:
        return "PRIOR_NOT_NARROW"
    return "BETWEEN"


def verdict(c1, c2, c3, c4):
    if c2 == "NOT_MAIN":
        return "not_supported"
    if c1 == "UNDER_DISPERSED" and c2 == "MAIN" and (c3 == "PULLS" or c4 == "PRIOR_NARROW"):
        return "supported"
    return "partly"


# ---------------------------------------------------------------- sets
def select(gset, per_person=PER_PERSON, partial_takes=PARTIAL_TAKES, seed=SEED):
    """(main, partial) window indices of a dev G1Set: up to per_person random windows of each person (seed), and
    the windows of the first partial_takes takes of each archive (probe_g1_decompose.subset)."""
    w = gset.windows
    rng = np.random.default_rng(seed)
    main = []
    for p in sorted(w["person_id"].unique()):
        idx = np.flatnonzero(w["person_id"].to_numpy() == p)
        k = min(per_person, len(idx))
        main += [int(i) for i in rng.choice(idx, size=k, replace=False)]
    keep = set()
    for _, g in w.groupby("archive", sort=True):
        keep |= set(list(dict.fromkeys(g["take_id"].tolist()))[:int(partial_takes)])
    partial = [int(i) for i in range(len(w)) if w["take_id"].iat[i] in keep]
    return sorted(main), partial


def blocks(gset, idx, size=BLOCK):
    """Blocks of whole takes (in index order) of at most size windows (a larger take alone)."""
    takes = gset.windows["take_id"].to_numpy().astype(str)
    by = {}
    for i in sorted(idx):
        by.setdefault(takes[i], []).append(int(i))
    out, cur = [], []
    for t, members in by.items():
        if cur and len(cur) + len(members) > size:
            out.append(cur)
            cur = []
        cur = cur + members
    if cur:
        out.append(cur)
    return out


def mean_pose_angles(path=PRIOR_JSON):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    mp = np.asarray(d["mean_pose"], dtype=np.float64)
    dr = np.asarray(d["draws"], dtype=np.float64)
    return arm_angles(mp), dr, d["meta"].get("train_persons_sha256")


# ---------------------------------------------------------------- attain
def stage_attain(a, log):
    from mdfm.gates import g1
    import pandas as pd
    ds = g1.G1Set("dev")
    main, partial = select(ds)
    m_ang, dr, tsha = mean_pose_angles()

    def truth_angles(gs, idx):
        rows, wid = [], []
        for i in idx:
            J, _ = gs.truth(i)
            rows.append(arm_angles(J[::STRIDE]))
            wid += [i] * rows[-1].shape[0]
        return np.concatenate(rows), np.asarray(wid)

    out = {"what": "attainability numbers (truth only, no model, no error value)", "seed": SEED,
           "train_mean_pose_angles_deg": dict(zip(ANGLES, m_ang.tolist())), "train_persons_sha256": tsha,
           "train_wrist_draws_sd_mm": (MM * dr.std(0, ddof=1)).tolist(), "train_wrist_draws_n": int(len(dr))}
    for name, idx in (("main", main), ("partial", partial)):
        A, wid = truth_angles(ds, idx)
        w = ds.windows.iloc[idx]
        out[name] = {"windows": len(idx), "persons": int(w["person_id"].nunique()),
                     "takes": int(w["take_id"].nunique()), "knots": int(w["n"].astype(int).sum()),
                     "angle_samples": int(A.shape[0]),
                     "truth_sd_deg": dict(zip(ANGLES, A.std(0).tolist())),
                     "truth_within_window_sd_deg": dict(zip(ANGLES, [float(within(A[:, k], wid).std())
                                                                     for k in range(4)])),
                     "truth_mean_deg": dict(zip(ANGLES, A.mean(0).tolist())),
                     "rms_truth_minus_train_mean_deg": dict(zip(ANGLES, np.sqrt(((A - m_ang) ** 2).mean(0)).tolist())),
                     "windows_list": idx}
    cs = g1.G1Set("calib")
    rng = np.random.default_rng(SEED)
    cidx = sorted(int(i) for i in rng.choice(len(cs), size=min(200, len(cs)), replace=False))
    A, _ = truth_angles(cs, cidx)
    wr = np.concatenate([g1.wrist_truth(*cs.truth(i))[::STRIDE] for i in cidx])
    out["calib"] = {"windows": len(cidx), "truth_sd_deg": dict(zip(ANGLES, A.std(0).tolist())),
                    "wrist_sd_mm": (MM * wr.std(0)).tolist()}
    wr = np.concatenate([g1.wrist_truth(*ds.truth(i))[::STRIDE] for i in main])
    out["main"]["truth_wrist_sd_mm"] = (MM * wr.std(0)).tolist()
    cols = list(pd.read_csv(os.path.join(DECOMPOSE, "windows.csv"), nrows=0).columns)
    out["decompose_columns_present"] = {c: c in cols for c in ("g1a_m", "draw_mean_m", "best_m", "n", "person_id")}
    write_json(os.path.join(a.out, "attain.json"), out)
    log(json.dumps({k: v for k, v in out.items() if k not in ("main", "partial")}, indent=1))
    for k in ("main", "partial"):
        log(k, json.dumps({kk: vv for kk, vv in out[k].items() if kk != "windows_list"}, indent=1))


# ---------------------------------------------------------------- run
def _decode_z(model, z, mask, beta, lengths, npw, gender):
    """Joints (M, 22, 3) and global rotations of latents z decoded at every knot (infer._pose)."""
    from mdfm.eval import infer as I
    from mdfm.model.decoder import all_knots
    w, k = all_knots(lengths)
    bo = model.decode(z * mask[..., None].to(z.dtype), beta, (w, k), npw, lengths)
    return I._pose(model, bo.rot6d, bo.transl, beta, gender, w)


def prior_sample(prior, cond, mask, gen, noise=True):
    """z (W, P, d_z) drawn patch by patch from the autoregressive motion prior (noise=False: its mean path)."""
    import torch
    W, P = mask.shape
    z = torch.zeros(W, P, prior.d_z, device=mask.device, dtype=prior.out.weight.dtype)
    for p in range(P):
        mu, ls = prior.params(z, cond, mask, dropout=False)
        v = mu[:, p]
        if noise:
            v = v + torch.exp(ls[:, p]) * torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype)
        z[:, p] = v * mask[:, p, None].to(v.dtype)
    return z


def stage_run(a, log):
    import torch
    import pandas as pd
    import probe_g1_decompose as P
    from mdfm.body.rotations import yaw_angle
    from mdfm.eval import infer as I
    from mdfm.gates import g1
    from mdfm.model import mfm as M
    from mdfm.model.posterior import sample
    from mdfm.train import provenance as PV

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    t_all = time.time()
    att = read_json(os.path.join(a.out, "attain.json"))
    ds = g1.G1Set("dev")
    main, partial = select(ds)
    if main != att["main"]["windows_list"] or partial != att["partial"]["windows_list"]:
        raise ValueError("window sets differ from attain.json")
    L = I.load_checkpoint(a.checkpoint, device=a.device)
    model = L.model
    if set(ds.persons) & set(L.persons.get("amass", ())):
        raise ValueError("a dev person among the checkpoint's training persons")
    lim = a.limit
    main_s, part_s = set(main), set(partial)
    todo = sorted(main_s | part_s)
    if lim:
        todo = todo[:lim]
    rows, arr = [], {}
    gen_p = torch.Generator(device=model.device).manual_seed(SEED)
    for bi, idx in enumerate(blocks(ds, todo)):
        tb = time.time()
        takes = [str(ds.windows["take_id"].iat[j]) for j in idx]
        wins = [I.g1_windows(ds, j, ("six",))["six"] for j in idx]
        seed = I.seed_of(SEED, "dev", takes[0], "six")
        bw = max(BLOCK, len(idx))
        st = P.posterior_pass(model, wins, 20, a.points, takes, seed, bw, keep_amortized=True)
        need60 = any(j in part_s for j in idx)
        st60 = P.posterior_pass(model, wins, 60, a.points, takes, seed, bw) if need60 else None
        c = st["chunks"][0]
        if len(st["chunks"]) != 1:
            raise ValueError("a block of more than one chunk")
        dec = {0: P._decode(model, c["post0"], st["mu_p0"], c, None)[1].cpu().numpy(),
               20: P._decode(model, c["post"], st["mu_p"], c, None)[1].cpu().numpy()}
        if st60 is not None:
            c6 = st60["chunks"][0]
            dec[60] = P._decode(model, c6["post"], st60["mu_p"], c6, None)[1].cpu().numpy()
        mask = c["post"].mask[..., None].to(c["post"].mu_z.dtype)
        draws = []
        for _ in range(DRAWS):
            z = sample(c["post"].mu_z, c["post"].logvar_z, generator=st["gen"]) * mask
            beta = sample(st["mu_p"], st["lv_p"], generator=st["gen"])
            draws.append(P._decode(model, c["post"], beta, c, None, z=z)[1].cpu().numpy())
        # motion prior samples: condition of the window, beta the shape prior mean of its gender
        batch = M.batch_from_windows(wins, sources=model.cfg.sources, families=model.heads.families,
                                     rate=model.rate, gen=0).to(model.device)
        pmask = batch.tokens.lat_idx.to(model.device) >= 0
        gender = batch.gender.to(model.device)
        mb = model.shape_prior.mean[gender].float()
        cond = batch.cond(mb)
        pri = []
        for s in range(PRIOR_SAMPLES + 1):
            z = prior_sample(model.prior, cond, pmask, gen_p, noise=s < PRIOR_SAMPLES)
            pos, R = _decode_z(model, z, pmask, mb, batch.lengths, batch.tokens.n_patches, gender)
            pri.append((pos.cpu().numpy(), yaw_angle(R[:, 0].float()).cpu().numpy()))
        cuts = P._cuts(c)
        for (s0, n), j in zip(cuts, idx):
            J, yaw = ds.truth(j)
            J, yaw = J[:n], yaw[:n]
            sl = slice(s0, s0 + n)
            r = ds.windows.iloc[int(j)]
            row = {"window": int(j), "take_id": r["take_id"], "archive": r["archive"], "person_id": r["person_id"],
                   "task": int(r["task"]), "n": int(n), "in_main": j in main_s, "in_partial": j in part_s}
            for s, p in dec.items():
                ek, ej = P.g1a_parts(p[sl], J)
                row[f"g1a{s}_m"] = float(ek.mean())
                row[f"arm{s}_m"] = float(ej[list(ARM_JOINTS)].mean())
            ed = [P.g1a_parts(d[sl], J) for d in draws]
            arm_d = np.array([e[1][list(ARM_JOINTS)].mean() for e in ed])
            g_d = np.array([e[0].mean() for e in ed])
            row.update(draw_g1a_mean_m=float(g_d.mean()), draw_g1a_best_m=float(g_d.min()),
                       draw_arm_mean_m=float(arm_d.mean()), draw_arm_best_m=float(arm_d.min()))
            rows.append(row)
            ks = slice(0, n, STRIDE)
            arr[f"t_{j}"] = arm_angles(J[ks]).astype(np.float32)
            for s, p in dec.items():
                arr[f"d{s}_{j}"] = arm_angles(p[sl][ks]).astype(np.float32)
            arr[f"dr_{j}"] = np.stack([arm_angles(d[sl][ks]) for d in draws]).astype(np.float32)
            arr[f"wt_{j}"] = g1.wrist_truth(J[ks], yaw[ks]).astype(np.float32)
            for s, (pp, py) in enumerate(pri):
                q, qy = pp[sl][ks], py[sl][ks]
                tag = f"p{s}" if s < PRIOR_SAMPLES else "pm"
                arr[f"{tag}_{j}"] = arm_angles(q).astype(np.float32)
                arr[f"w{tag}_{j}"] = g1.heading_points(q[:, g1.WRIST_JOINT], q[:, g1.PELVIS], qy).astype(np.float32)
        log(f"block {bi + 1}: {len(idx)} windows, {len(rows)} done, {time.time() - tb:.0f} s")
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "run_windows.csv"), index=False, encoding="utf-8", lineterminator="\n")
    np.savez_compressed(os.path.join(a.out, "angles.npz"), **arr)
    from mdfm.gates import common
    write_json(os.path.join(a.out, "run.json"), {
        "checkpoint": L.describe(), "device": str(model.device), "points": a.points, "draws": DRAWS,
        "prior_samples": PRIOR_SAMPLES, "stride_knots": STRIDE, "seed": SEED, "block_windows": BLOCK,
        "limit": lim, "windows": len(rows), "dev_manifest_sha256": ds.manifest["manifest_sha256"],
        "code_sha256": common.code_sha256([os.path.abspath(__file__), P.__file__, I.__file__, g1.__file__]),
        "moveatlas_head": PV.moveatlas_head(), "seconds": round(time.time() - t_all, 1),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"run: {len(rows)} windows in {time.time() - t_all:.0f} s")


# ---------------------------------------------------------------- read
def stage_read(a, log):
    import pandas as pd
    att = read_json(os.path.join(a.out, "attain.json"))
    run = read_json(os.path.join(a.out, "run.json"))
    df = pd.read_csv(os.path.join(a.out, "run_windows.csv"), encoding="utf-8")
    Z = np.load(os.path.join(a.out, "angles.npz"))
    m_ang, dr, _ = mean_pose_angles()
    person = dict(zip(df["window"], df["person_id"]))

    def gather(ws, key):
        return np.concatenate([Z[f"{key}_{w}"] for w in ws]), np.concatenate(
            [[w] * Z[f"t_{w}"].shape[0] for w in ws])

    res = {"checkpoint_sha256": run["checkpoint"]["sha256"], "moveatlas_head": run["moveatlas_head"],
           "train_mean_pose_angles_deg": dict(zip(ANGLES, m_ang.tolist()))}
    main = [int(w) for w in df.loc[df["in_main"], "window"]]
    part = [int(w) for w in df.loc[df["in_partial"], "window"]]

    def pull_block(ws, key):
        T, wid = gather(ws, "t")
        D, _ = gather(ws, key)
        per = {an: pull_stats(D[:, k], T[:, k], m_ang[k]) for k, an in enumerate(ANGLES)}
        wi = {an: pull_stats(within(D[:, k], wid), within(T[:, k], wid), 0.0) for k, an in enumerate(ANGLES)}
        mb = float(np.mean([per[x]["b"] for x in ANGLES]))
        ms = float(np.mean([per[x]["share"] for x in ANGLES]))
        return {"per_angle": per, "within_window": {x: {"b": wi[x]["b"], "sd_ratio": wi[x]["sd_ratio"]} for x in wi},
                "mean_b": mb, "mean_share": ms, "mean_sd_ratio": float(np.mean([per[x]["sd_ratio"] for x in ANGLES])),
                "mean_rmse_deg": float(np.mean([per[x]["rmse"] for x in ANGLES])),
                "mean_b_rev": float(np.mean([per[x]["b_rev"] for x in ANGLES])),
                "mean_lam": float(np.mean([per[x]["lam"] for x in ANGLES])),
                "mean_within_b": float(np.mean([wi[x]["b"] for x in ANGLES]))}

    def boot_pull(ws, key, field):
        pw = {}
        for w in ws:
            pw.setdefault(person[w], []).append(w)

        def fn(ps):
            sel = [w for p in ps for w in pw[p]]
            return pull_block(sel, key)[field]
        return boot_persons(list(pw), fn)

    # C2 on the main set, 0 and 20 steps
    res["main"] = {"windows": len(main), "persons": int(df.loc[df["in_main"], "person_id"].nunique())}
    for s in (0, 20):
        blk = pull_block(main, f"d{s}")
        if s == 20:
            blk["mean_b_interval"] = boot_pull(main, "d20", "mean_b")
            blk["mean_share_interval"] = boot_pull(main, "d20", "mean_share")
        res["main"][f"steps{s}"] = blk
    # C1: draws of the 20-step posterior
    T, _ = gather(main, "t")
    DR = np.concatenate([Z[f"dr_{w}"] for w in main], axis=1)        # (16, n, 4)
    D20, _ = gather(main, "d20")
    cal = {an: draw_calibration(DR[:, :, k], T[:, k], D20[:, k]) for k, an in enumerate(ANGLES)}
    pooled_cov = float(np.mean([cal[x]["range_coverage"] for x in ANGLES]))
    res["main"]["calibration"] = {"per_angle": cal, "range_coverage_mean": pooled_cov,
                                  "range_coverage_calibrated": COVERAGE_CALIBRATED,
                                  "sd_over_rmse_mean": float(np.mean([cal[x]["sd_over_rmse"] for x in ANGLES]))}
    mm = df[df["in_main"]]
    res["main"]["positions_mm"] = {k: MM * knot_weighted(mm[c], mm["n"]) for k, c in (
        ("g1a_steps0", "g1a0_m"), ("g1a_steps20", "g1a20_m"), ("arm_steps0", "arm0_m"), ("arm_steps20", "arm20_m"),
        ("g1a_single_draw", "draw_g1a_mean_m"), ("g1a_best_of_16", "draw_g1a_best_m"),
        ("arm_single_draw", "draw_arm_mean_m"), ("arm_best_of_16", "draw_arm_best_m"))}
    # existing decompose file: dispersion ratios
    dw = pd.read_csv(os.path.join(DECOMPOSE, "windows.csv"), encoding="utf-8")
    g, sd_, bt = (knot_weighted(dw[c], dw["n"]) for c in ("g1a_m", "draw_mean_m", "best_m"))
    res["decompose_file"] = {"windows": int(len(dw)), "g1a_kw_mm": MM * g, "single_draw_kw_mm": MM * sd_,
                             "best_of_16_kw_mm": MM * bt, "single_over_mean": sd_ / g, "best_over_mean": bt / g,
                             "single_over_mean_calibrated_gaussian": math.sqrt(2.0),
                             "implied_draw_spread_mm": MM * math.sqrt(max(sd_ ** 2 - g ** 2, 0.0))}
    # C3: partial set, 0, 20, 60 steps
    pr = {}
    for s in STEPS:
        pr[f"steps{s}"] = pull_block(part, f"d{s}")
    pp = df[df["in_partial"]]
    pr["positions_mm"] = {f"{k}_steps{s}": MM * knot_weighted(pp[f"{k}{s}_m"], pp["n"])
                          for k in ("g1a", "arm") for s in STEPS}
    pr["windows"], pr["persons"] = len(part), int(pp["person_id"].nunique())
    res["partial"] = pr
    # C4: motion prior samples against the truth, on the main set
    P0 = np.concatenate([gather(main, f"p{s}")[0] for s in range(PRIOR_SAMPLES)])
    PM, _ = gather(main, "pm")
    WP = np.concatenate([np.concatenate([Z[f"wp{s}_{w}"] for w in main]) for s in range(PRIOR_SAMPLES)])
    WT = np.concatenate([Z[f"wt_{w}"] for w in main])
    tr_sd = MM * dr.std(0, ddof=1)
    ang_ratio = {an: float(P0[:, k].std() / T[:, k].std()) for k, an in enumerate(ANGLES)}
    res["prior"] = {"samples_per_window": PRIOR_SAMPLES,
                    "sample_sd_deg": dict(zip(ANGLES, P0.std(0).tolist())),
                    "mean_path_sd_deg": dict(zip(ANGLES, PM.std(0).tolist())),
                    "truth_sd_deg": dict(zip(ANGLES, T.std(0).tolist())),
                    "decoded20_sd_deg": dict(zip(ANGLES, D20.std(0).tolist())),
                    "sample_mean_deg": dict(zip(ANGLES, P0.mean(0).tolist())),
                    "truth_mean_deg": dict(zip(ANGLES, T.mean(0).tolist())),
                    "angle_sd_ratio": ang_ratio, "angle_sd_ratio_mean": float(np.mean(list(ang_ratio.values()))),
                    "wrist_sample_sd_mm": (MM * WP.std(0)).tolist(), "wrist_truth_dev_sd_mm": (MM * WT.std(0)).tolist(),
                    "wrist_train_draws_sd_mm": tr_sd.tolist(),
                    "wrist_sd_ratio_to_train": (MM * WP.std(0) / tr_sd).tolist(),
                    "wrist_sd_ratio_to_train_mean": float(np.mean(MM * WP.std(0) / tr_sd)),
                    "wrist_sd_ratio_to_dev_truth_mean": float(np.mean(WP.std(0) / WT.std(0)))}
    c1 = reading_c1(pooled_cov)
    c2 = reading_c2(res["main"]["steps20"]["mean_b"], res["main"]["steps20"]["mean_share"])
    c3 = reading_c3(pr["steps0"]["mean_b"], pr["steps20"]["mean_b"], pr["steps60"]["mean_b"])
    c4 = reading_c4(res["prior"]["angle_sd_ratio_mean"], res["prior"]["wrist_sd_ratio_to_train_mean"])
    res["readings"] = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "verdict": verdict(c1, c2, c3, c4)}
    write_json(os.path.join(a.out, "results.json"), res)
    md = summary_md(res, att)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    log(md)


def summary_md(r, att):
    f1 = lambda x: f"{x:.1f}"  # noqa: E731
    f2 = lambda x: f"{x:.2f}"  # noqa: E731
    L = ["# G1D-Hprior results", "",
         "Written by scripts/g1d_Hprior.py --stage read from run.json, run_windows.csv, angles.npz and attain.json "
         "of this folder. Dev diagnostic on the AMASS dev archives, not a gate read. Checkpoint SHA-256 "
         f"{r['checkpoint_sha256']}; MoveAtlas HEAD {r['moveatlas_head']}.", "",
         f"Readings: C1 {r['readings']['C1']}, C2 {r['readings']['C2']}, C3 {r['readings']['C3']}, "
         f"C4 {r['readings']['C4']}; verdict {r['readings']['verdict']}.", "",
         "## C2: pull toward the training mean arm pose (main set)", "",
         f"Main set: {r['main']['windows']} windows of {r['main']['persons']} persons.", "",
         "| steps | angle | b | b_rev | sd ratio | lambda | share | rmse deg | mean decoded | mean truth | "
         "train mean pose |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in (0, 20):
        for an in ANGLES:
            p = r["main"][f"steps{s}"]["per_angle"][an]
            L.append(f"| {s} | {an} | {f2(p['b'])} | {f2(p['b_rev'])} | {f2(p['sd_ratio'])} | {f2(p['lam'])} | "
                     f"{f2(p['share'])} | "
                     f"{f1(p['rmse'])} | {f1(p['mean_d'])} | {f1(p['mean_t'])} | "
                     f"{f1(r['train_mean_pose_angles_deg'][an])} |")
    m20 = r["main"]["steps20"]
    L += ["", f"20 steps: mean b {f2(m20['mean_b'])} (95% person bootstrap {f2(m20['mean_b_interval'][0])} to "
              f"{f2(m20['mean_b_interval'][1])}), mean share {f2(m20['mean_share'])} "
              f"({f2(m20['mean_share_interval'][0])} to {f2(m20['mean_share_interval'][1])}), mean within-window b "
              f"{f2(m20['mean_within_b'])}.", "",
          "## C1: calibration of the 20-step posterior draws (main set)", "",
          "| angle | range coverage | median abs z | rms draw sd deg | rms error deg | sd over rmse | mean rank |",
          "|---|---|---|---|---|---|---|"]
    for an in ANGLES:
        c = r["main"]["calibration"]["per_angle"][an]
        L.append(f"| {an} | {f2(c['range_coverage'])} | {f2(c['median_abs_z'])} | {f1(c['rms_sd'])} | "
                 f"{f1(c['rms_err'])} | {f2(c['sd_over_rmse'])} | {f2(c['mean_rank'])} |")
    cal = r["main"]["calibration"]
    L += ["", f"Mean range coverage {f2(cal['range_coverage_mean'])} (calibrated {f2(cal['range_coverage_calibrated'])}).",
          "", "Positions (knot-weighted mm, main set): " + ", ".join(
              f"{k} {f1(v)}" for k, v in r["main"]["positions_mm"].items()) + ".", ""]
    d = r["decompose_file"]
    L += [f"Decompose windows.csv ({d['windows']} windows, knot-weighted): G1a {f1(d['g1a_kw_mm'])} mm, single draw "
          f"{f1(d['single_draw_kw_mm'])} mm, best of 16 {f1(d['best_of_16_kw_mm'])} mm; single over mean "
          f"{d['single_over_mean']:.3f} (a calibrated Gaussian posterior: {d['single_over_mean_calibrated_gaussian']:.3f}); "
          f"implied draw spread {f1(d['implied_draw_spread_mm'])} mm.", "",
          "## C3: refinement steps (partial set)", "",
          f"Partial set: {r['partial']['windows']} windows of {r['partial']['persons']} persons.", "",
          "| steps | mean b | mean b_rev | mean sd ratio | mean share | mean within b | mean rmse deg | G1a mm | arm mm |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in STEPS:
        p = r["partial"][f"steps{s}"]
        pm = r["partial"]["positions_mm"]
        L.append(f"| {s} | {f2(p['mean_b'])} | {f2(p['mean_b_rev'])} | {f2(p['mean_sd_ratio'])} | {f2(p['mean_share'])} | "
                 f"{f2(p['mean_within_b'])} | {f1(p['mean_rmse_deg'])} | {f1(pm[f'g1a_steps{s}'])} | "
                 f"{f1(pm[f'arm_steps{s}'])} |")
    pr = r["prior"]
    L += ["", "## C4: spread of the motion prior (main set conditions)", "",
          "| angle | prior sample sd | prior mean path sd | decoded 20 sd | truth sd | sample over truth | "
          "prior sample mean | truth mean |", "|---|---|---|---|---|---|---|---|"]
    for an in ANGLES:
        L.append(f"| {an} | {f1(pr['sample_sd_deg'][an])} | {f1(pr['mean_path_sd_deg'][an])} | "
                 f"{f1(pr['decoded20_sd_deg'][an])} | {f1(pr['truth_sd_deg'][an])} | {f2(pr['angle_sd_ratio'][an])} | "
                 f"{f1(pr['sample_mean_deg'][an])} | {f1(pr['truth_mean_deg'][an])} |")
    L += ["", "Left wrist in the pelvis-heading frame, sd per axis x, y, z (mm): prior samples "
          + ", ".join(f1(x) for x in pr["wrist_sample_sd_mm"]) + "; train draws of prior.json "
          + ", ".join(f1(x) for x in pr["wrist_train_draws_sd_mm"]) + "; dev truth "
          + ", ".join(f1(x) for x in pr["wrist_truth_dev_sd_mm"])
          + f". Mean ratio to train {f2(pr['wrist_sd_ratio_to_train_mean'])}, to dev truth "
            f"{f2(pr['wrist_sd_ratio_to_dev_truth_mean'])}; mean angle sd ratio {f2(pr['angle_sd_ratio_mean'])}.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- io
def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=("attain", "run", "read"))
    p.add_argument("--out", default=OUT)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="first windows only (a smoke run, labelled)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    {"attain": stage_attain, "run": stage_run, "read": stage_read}[a.stage](a, log)


if __name__ == "__main__":
    main()
