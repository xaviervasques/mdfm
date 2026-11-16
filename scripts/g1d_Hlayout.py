"""G1 diagnosis, hypothesis Hlayout (task G1D-Hlayout, reports/weekend_2026-10-01/tasks/G1D-Hlayout.md).

Hypothesis: the six-IMU layout g1_six is too rare in training (0.10 of the synthetic layout draws) or its
sensor placements differ from the G1 protocol's, so the model reads it worse than layouts it sees more; and the
dev archives differ in motion from training. Every output goes under --out (default
reports/bench/g1_diag/Hlayout). Data: the AMASS dev G1 set (cache/g1/g1_v1/dev), the dev and train body
stores; no lockbox, calib, EVAL, CODY or care_pd record is opened.

Subcommands:
- facts (reads no error value): (1) exposure of the G1 sites in the run's layout mixture (config.json of the run
  folder) with the token counts of layouts.token_count, and the real-source token shares of the run log (mean over
  the phase B log events); (2) the per-site comparison of the training draw of g1_six (factory/synth.py: the mount
  prior of MountPriors.sample_mounts, the placement jitter PLACEMENT_SD_M, the xsens13 DeviceFamily of
  device_noise.json) with the G1 protocol (gates/g1.py: R0 Exp(theta), theta ~ N(0, 5 deg^2 I), no placement
  jitter, the per-site levels of priors.json); (3) motion statistics (feature_block) of every dev window of the
  G1 set, read from the dev body store at the window's take and first knot, and of every non-overlapping 10 s
  window of the train body store; a PCA fitted on the standardized train features; per window the mean distance
  to its K nearest train windows (dev) or to the K nearest train windows of other persons (train,
  leave-person-out) or of other archives (train, leave-archive-out); the dissimilarity d of a dev window is its
  distance over the median leave-person-out distance of the train windows. Writes facts.json and
  similarity.csv.
- arms (reads error values): G1a (posterior mean after --refine-steps steps; refine0, the amortized posterior)
  of the same dev windows under three inputs, with the harness blocks (infer.G1Predictor.block), seeds
  (infer.seed_of(seed, 'dev', first take, 'six')) and per-take shape pooling: six_protocol (the stored G1
  inputs, as the harness), six_factory (factory.synth.SynthSampler render of the body-store window under the
  g1_six layout: the training draw of mounts, placement, biases and noise) and xs13_factory (the same under
  the xsens13 layout, 13 sites). Also body_check_m: the G1a of the body-store window's own forward kinematics
  against the stored G1 truth (it shows the factory renders the scored motion). Rows appended per block to
  arms_windows.csv (a rerun skips the windows already written).
- analyze (reads error values): the gate's per-window G1a (reports/gates/g1/dev_<name>/windows.csv, v1 and seed
  2) joined with similarity.csv and the decompose motion_m; the arms. Writes results.json and results.md.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
SCRIPTS = os.path.join(TREE, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g1_diag", "Hlayout")
RUN_DIR = os.path.join(TREE, "cache", "runs", "phaseA_root_full")
CKPT = os.path.join(RUN_DIR, "ckpt_B_032044.pt")
GATE_RUNS = {"v1": "dev_phaseA_root_full_ckpt_B_032044", "s2": "dev_pilot_v1_s2_ckpt_B_032044"}
DECOMPOSE = os.path.join(TREE, "reports", "bench", "g1_decompose_phaseA_root_full_ckpt_B_032044")
SEED = 20260929                 # the --seed default of scripts/eval_g1_dev.py
ARMS = ("six_protocol", "six_factory", "xs13_factory")
ARM_LAYOUT = {"six_factory": "g1_six", "xs13_factory": "xsens13"}
FACTORY_SALT = "G1D-Hlayout"
K_NN = 10
STRIDE = 2                      # knots between feature samples (50 Hz)
LAG = 5                         # knots of the angular and linear speeds (0.05 s)
PCA_VAR = 0.90
PCA_MAX = 30
MM = 1000.0
PASS_MM = 45.0
GAP_SHARE = 0.30                # the branch-rule fraction of the gap (handoff 5 section 1, M rows)
ARM_COLUMNS = ("window", "take_id", "archive", "person_id", "task", "n", "arm", "g1a_m", "refine0_m",
               "body_check_m")


def log(*m):
    print(*m, flush=True)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_default)
        f.write("\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_text(path, text):
    for ch in (chr(0x2014), chr(0x2013)):
        if ch in text:
            raise ValueError(f"{path}: an em or en dash in the text")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- facts (1): exposure
def p_random_contains(sites_wanted, n_sites, k_lo, k_hi):
    """P(a uniform k-subset of n_sites contains a fixed set of m sites), k uniform in [k_lo, k_hi]."""
    m = len(sites_wanted)
    tot = 0.0
    for k in range(k_lo, k_hi + 1):
        if k >= m:
            tot += math.comb(n_sites - m, k - m) / math.comb(n_sites, k)
    return tot / (k_hi - k_lo + 1)


def p_random_exact(n_sites, k_lo, k_hi, m):
    return (1.0 / (k_hi - k_lo + 1)) / math.comb(n_sites, m) if k_lo <= m <= k_hi else 0.0


def exposure(run_dir=RUN_DIR):
    from mdfm.factory import layouts as L
    from mdfm.factory.imu_render import SITES
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    mix = cfg["mixture"]["layouts"]
    tot = sum(mix.values())
    lays = L.load_layouts()
    six = set(L.G1_SIX)
    fam = {}
    for name, w in mix.items():
        lay = lays[name]
        imu = lay.imu if any(s.modality == "imu" for s in lay.streams) else None
        p = w / tot
        if lay.k_sites is not None:
            k_lo, k_hi = lay.k_sites
            n_all = len(imu.sites)
            p_all6 = p_random_contains(six, n_all, k_lo, k_hi)
            p_exact = p_random_exact(n_all, k_lo, k_hi, len(six))
            per_site = {s: (sum(k / n_all for k in range(k_lo, k_hi + 1)) / (k_hi - k_lo + 1)) for s in L.G1_SIX}
            # expected tokens over k (acc and gyro groups: 2 per site, one nuisance token per site)
            P = L.n_patches(10.0)
            toks = float(np.mean([P + 1 + k * (2 * P + 1) for k in range(k_lo, k_hi + 1)]))
            sites = "random subset"
        else:
            sites_l = list(imu.sites) if imu is not None else []
            p_all6 = 1.0 if six <= set(sites_l) else 0.0
            p_exact = 1.0 if set(sites_l) == six else 0.0
            per_site = {s: 1.0 if s in sites_l else 0.0 for s in L.G1_SIX}
            toks = float(L.token_count(L.concrete(lay, np.random.default_rng(0)), 10.0))
            sites = "|".join(sites_l)
        mount = lay.mount
        fam[name] = {"share": p, "sites": sites, "mount": mount, "device": lay.device, "p_all_six": p_all6,
                     "p_exactly_six": p_exact, "p_site": per_site, "tokens_10s": toks}
    share_tok = {n: f["share"] * f["tokens_10s"] for n, f in fam.items()}
    st = sum(share_tok.values())
    out = {"mixture_source": common_rel(os.path.join(run_dir, "config.json")), "families": fam}
    out["window_share_all_six"] = sum(f["share"] * f["p_all_six"] for f in fam.values())
    out["window_share_exactly_six"] = sum(f["share"] * f["p_exactly_six"] for f in fam.values())
    out["window_share_all_six_xsens13_mount"] = sum(f["share"] * f["p_all_six"] for f in fam.values()
                                                     if f["mount"] == "xsens13")
    out["window_share_site"] = {s: sum(f["share"] * f["p_site"][s] for f in fam.values()) for s in L.G1_SIX}
    out["token_share_all_six"] = sum(share_tok[n] * fam[n]["p_all_six"] for n in fam) / st
    out["token_share_g1_six_family"] = share_tok["g1_six"] / st
    # synthetic token budget of the run: tokens_total x (share_a x synthetic A + (1 - share_a) x synthetic B)
    sch = cfg["schedule"]
    sa = float(sch["share_a"])
    syn = cfg["mixture"]["synthetic_share"]
    syn_tokens = float(sch["tokens_total"]) * (sa * syn[0] + (1 - sa) * syn[1])
    out["synthetic_tokens_planned"] = syn_tokens
    out["g1_six_windows_planned"] = syn_tokens * share_tok["g1_six"] / st / fam["g1_six"]["tokens_10s"]
    out["all_six_windows_planned"] = sum(syn_tokens * share_tok[n] * fam[n]["p_all_six"] / st / fam[n]["tokens_10s"]
                                         for n in fam)
    out["xsens13_windows_planned"] = sum(syn_tokens * share_tok[n] / st / fam[n]["tokens_10s"]
                                         for n in ("xsens13", "xsens13_insoles"))
    out["planned_note"] = ("tokens_total x synthetic share of the phases, split by the token share of each family "
                           "for 10 s windows; pair renders and shorter windows not counted (an estimate)")
    # real token shares of the run's phase B log events
    shares, nlog = {}, 0
    with open(os.path.join(run_dir, "log.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"event": "log"' not in line:
                continue
            d = json.loads(line)
            if d.get("phase") != "B" or "token_share" not in d:
                continue
            nlog += 1
            for k, v in d["token_share"].items():
                shares[k] = shares.get(k, 0.0) + float(v)
    out["phase_b_token_share_mean"] = {k: v / nlog for k, v in sorted(shares.items())} if nlog else {}
    out["phase_b_log_events"] = nlog
    out["sites_all"] = list(SITES)
    return out


def common_rel(path):
    from mdfm.gates import common
    return common.rel(path)


# ---------------------------------------------------------------- facts (2): placement
def placement():
    from mdfm.factory import layouts as L
    from mdfm.factory import synth as SY
    from mdfm.factory.imu_render import SITE_RULES, SITES
    from mdfm.gates import g1
    from mdfm.kernels.nuisance import OFFSET_MAX_M, MountPriors
    pri = MountPriors.load()
    dev_train = L.device_families()["xsens13"]
    lv = g1.device_levels(pri)
    lays = L.load_layouts()
    six_train = lays["g1_six"].imu
    R0_train = pri.rotation(lays["g1_six"].mount, list(six_train.sites)).double()
    R0_g1 = pri.rotation(g1.PARAMS["mount_family"], list(g1.SITES)).double()
    order = [list(six_train.sites).index(s) for s in g1.SITES]
    deg = 180.0 / math.pi
    rows = []
    for j, s in enumerate(g1.SITES):
        k = SITES.index(s)
        st = float(pri.per_site(pri.sigma_tilt, "xsens13", [s])[0]) * deg
        sy = float(pri.per_site(pri.sigma_yaw, "xsens13", [s])[0]) * deg
        rms_train = math.sqrt(2 * st ** 2 + sy ** 2)
        rms_g1 = math.sqrt(3) * g1.MOUNT_JITTER_DEG
        rows.append({
            "site": s, "rule_segment_joint": SITE_RULES[s][0], "rule_bone_fraction": SITE_RULES[s][2],
            "rule_surface_offset_m": list(SITE_RULES[s][3]),
            "R0_max_abs_diff": float((R0_train[order[j]] - R0_g1[j]).abs().max()),
            "mount_train_sigma_tilt_deg": st, "mount_train_sigma_yaw_deg": sy,
            "mount_train_rms_deg": rms_train, "mount_g1_rms_deg": rms_g1,
            "placement_train_rms_m": math.sqrt(3) * SY.PLACEMENT_SD_M, "placement_g1_rms_m": 0.0,
            "noise_acc_train_g": dev_train.noise_acc[k].tolist(), "noise_acc_g1_g": lv["noise_acc"][j].tolist(),
            "noise_gyro_train_dps": dev_train.noise_gyro[k].tolist(), "noise_gyro_g1_dps": lv["noise_gyro"][j].tolist(),
            "bias_acc_train_g": dev_train.bias_acc_sd[k].tolist(), "bias_acc_g1_g": lv["bias_acc"][j].tolist(),
            "bias_gyro_train_dps": dev_train.bias_gyro_sd[k].tolist(), "bias_gyro_g1_dps": lv["bias_gyro"][j].tolist(),
        })
    ratio = lambda a, b: float(np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
    for r in rows:
        r["noise_acc_ratio_train_over_g1"] = ratio(r["noise_acc_train_g"], r["noise_acc_g1_g"])
        r["noise_gyro_ratio_train_over_g1"] = ratio(r["noise_gyro_train_dps"], r["noise_gyro_g1_dps"])
        r["bias_acc_ratio_train_over_g1"] = ratio(r["bias_acc_train_g"], r["bias_acc_g1_g"])
        r["bias_gyro_ratio_train_over_g1"] = ratio(r["bias_gyro_train_dps"], r["bias_gyro_g1_dps"])
    same = {"rate_hz": (list(six_train.fps), [g1.RATE]), "acc_content": (six_train.acc_content, "with_gravity"),
            "channels": ("acc+gyro" if six_train.chan_mask[:, :6].all() else "other", "acc+gyro"),
            "mount_family": (lays["g1_six"].mount, g1.PARAMS["mount_family"]),
            "device_family": (lays["g1_six"].device, g1.PARAMS["device"]),
            "acc_range_g": (dev_train.acc_range_g, g1.PARAMS["acc_range_g"]),
            "gyro_range_dps": (dev_train.gyro_range_dps, g1.PARAMS["gyro_range_dps"]),
            "site_rule": ("imu_render.site_table (kernels.imu.imu_predict in the factory)",
                          "imu_render.site_table (imu_render.render, the reference renderer)"),
            "placement_bound_m": (OFFSET_MAX_M, 0.0)}
    return {"sites": rows, "protocol_vs_training": {k: {"training": v[0], "g1": v[1]} for k, v in same.items()}}


# ---------------------------------------------------------------- facts (3): motion statistics
FEATURE_NAMES = None


def feature_block(rot6d, transl, lengths, rest_root):
    """Motion statistics of windows (window frame of BodyStore.windows): rot6d (n, K, 22, 6), transl (n, K, 3),
    lengths (n,), rest_root (n, 3) -> (n, F) float64. Per window over its valid knots: mean and standard
    deviation of the axis-angle of the 21 local joint rotations (126), mean local angular speed per joint over
    LAG knots (21) and its 90th percentile over joints and knots (1), root tilt from vertical mean and sd (2),
    pelvis height mean and sd (2), horizontal pelvis speed mean, sd and 90th percentile (3), vertical pelvis
    speed sd (1), absolute heading rate mean (1)."""
    from mdfm.body.rotations import geodesic, matrix_to_axis_angle, rot6d_to_matrix, yaw_angle
    n = rot6d.shape[0]
    out = []
    for b in range(n):
        L = int(lengths[b])
        r6 = torch.as_tensor(np.asarray(rot6d[b, :L]), dtype=torch.float64)
        R = rot6d_to_matrix(r6)                                         # (L, 22, 3, 3)
        idx = torch.arange(0, L, STRIDE)
        Rs = R[idx]
        aa = matrix_to_axis_angle(Rs[:, 1:]).numpy()                    # (m, 21, 3)
        f_aa = np.concatenate([aa.mean(0).ravel(), aa.std(0).ravel()])
        i0 = idx[idx + LAG < L]
        if len(i0) == 0:
            i0 = torch.tensor([0])
            i1 = torch.tensor([L - 1])
        else:
            i1 = i0 + LAG
        dt = float((i1[0] - i0[0]).item()) / 100.0 if (i1[0] - i0[0]).item() > 0 else 1.0
        ang = geodesic(R[i0, 1:], R[i1, 1:]).numpy() / dt               # (m, 21) rad/s
        f_w = np.concatenate([ang.mean(0), [np.quantile(ang, 0.9)]])
        up = R[idx, 0][:, 1, 1].clamp(-1.0, 1.0)
        tilt = torch.arccos(up).numpy()
        tr_w = np.asarray(transl[b, :L], dtype=np.float64)
        h = tr_w[np.asarray(idx), 1] + float(rest_root[b][1])
        v = (tr_w[np.asarray(i1)] - tr_w[np.asarray(i0)]) / dt
        vh = np.linalg.norm(v[:, [0, 2]], axis=1)
        yaw = yaw_angle(R[:, 0]).numpy()
        dyaw = np.angle(np.exp(1j * (yaw[np.asarray(i1)] - yaw[np.asarray(i0)]))) / dt
        f_r = [tilt.mean(), tilt.std(), h.mean(), h.std(), vh.mean(), vh.std(), np.quantile(vh, 0.9),
               v[:, 1].std(), np.abs(dyaw).mean()]
        out.append(np.concatenate([f_aa, f_w, f_r]))
    return np.stack(out)


def feature_names():
    from mdfm.body.smplh import BODY_JOINTS
    j = list(BODY_JOINTS)[1:22]
    names = [f"aa_mean_{x}_{c}" for x in j for c in "xyz"] + [f"aa_sd_{x}_{c}" for x in j for c in "xyz"]
    names += [f"angspeed_{x}" for x in j] + ["angspeed_p90"]
    names += ["tilt_mean", "tilt_sd", "pelvis_h_mean", "pelvis_h_sd", "hspeed_mean", "hspeed_sd", "hspeed_p90",
              "vspeed_sd", "yawrate_abs_mean"]
    return names


def store_features(store, takes, starts, chunk=64):
    feats = []
    for a in range(0, len(takes), chunk):
        bw = store.windows(takes[a:a + chunk], starts[a:a + chunk])
        rr = store.rest_root[np.asarray(takes[a:a + chunk])]
        feats.append(feature_block(bw.rot6d.numpy(), bw.transl.numpy(), bw.lengths.numpy(), rr))
    return np.concatenate(feats)


def train_windows(store, knots=1000, min_knots=200):
    """(takes, starts) of every non-overlapping window of the store (the G1 window rule: from knot 0 every
    `knots`, the last kept when it holds at least min_knots)."""
    tk, st = [], []
    for i, n in enumerate(store.takes["n_knots"].to_numpy(np.int64)):
        for s in range(0, int(n), knots):
            if min(knots, int(n) - s) >= min_knots:
                tk.append(i)
                st.append(s)
    return np.array(tk, np.int64), np.array(st, np.int64)


def knn_mean(Q, X, k, exclude=None, chunk=512):
    """Mean distance of each row of Q to its k nearest rows of X; exclude(i0, i1) -> bool mask (rows of the
    chunk, X rows) of pairs left out."""
    out = np.empty(len(Q))
    X2 = (X ** 2).sum(1)
    for a in range(0, len(Q), chunk):
        q = Q[a:a + chunk]
        d2 = (q ** 2).sum(1)[:, None] + X2[None] - 2 * q @ X.T
        d2 = np.maximum(d2, 0.0)
        if exclude is not None:
            d2[exclude(a, a + len(q))] = np.inf
        part = np.partition(d2, k, axis=1)[:, :k]
        out[a:a + len(q)] = np.sqrt(part).mean(1)
    return out


def similarity(out):
    from mdfm.data.body_store import BodyStore
    from mdfm.gates import g1
    t0 = time.time()
    gset = g1.G1Set("dev")
    tr = BodyStore(split="train")
    dv = BodyStore(split="dev")
    # dev windows of the G1 set -> rows of the dev body store
    row_of = {t: i for i, t in enumerate(dv.takes["take_id"])}
    w = gset.windows
    missing = sorted(set(w["take_id"]) - set(row_of))
    if missing:
        raise ValueError(f"{len(missing)} G1 dev takes absent from the dev body store, e.g. {missing[:2]}")
    d_take = np.array([row_of[t] for t in w["take_id"]], np.int64)
    d_start = w["start"].to_numpy(np.int64)
    nk = dv.takes["n_knots"].to_numpy(np.int64)[d_take]
    L = np.minimum(1000, nk - d_start)
    len_mismatch = int((L != w["n"].to_numpy(np.int64)).sum())
    log(f"dev windows {len(w)}, length mismatches with the G1 set {len_mismatch}")
    Fd = store_features(dv, d_take, d_start)
    log(f"dev features {Fd.shape} ({time.time() - t0:.0f} s)")
    t_take, t_start = train_windows(tr)
    Ft = store_features(tr, t_take, t_start)
    log(f"train features {Ft.shape} ({time.time() - t0:.0f} s)")
    mu, sd = Ft.mean(0), Ft.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Zt, Zd = (Ft - mu) / sd, (Fd - mu) / sd
    U, S, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    q = int(min(PCA_MAX, np.searchsorted(np.cumsum(var), PCA_VAR) + 1))
    P = Vt[:q].T
    c = Zt.mean(0)
    Yt, Yd = (Zt - c) @ P, (Zd - c) @ P
    t_person = pd.factorize(tr.takes["person_id"].to_numpy().astype(str)[t_take])[0]
    t_arch_s = tr.takes["archive"].to_numpy().astype(str)[t_take]
    t_arch = pd.factorize(t_arch_s)[0]
    lpo = knn_mean(Yt, Yt, K_NN, exclude=lambda a, b: t_person[a:b, None] == t_person[None])
    lao = knn_mean(Yt, Yt, K_NN, exclude=lambda a, b: t_arch[a:b, None] == t_arch[None])
    dd = knn_mean(Yd, Yt, K_NN)
    scale = float(np.median(lpo))
    sim = w[["take_id", "archive", "person_id", "task", "n"]].copy()
    sim.insert(0, "window", np.arange(len(w)))          # the G1Set row (the gate and decompose window id)
    sim["knn_dist"] = dd
    sim["d"] = dd / scale
    sim["d_pct_train_lpo"] = [float((lpo < x).mean()) for x in dd]
    for j, nm in enumerate(feature_names()):
        if nm.startswith(("angspeed_p90", "tilt_", "pelvis_h", "hspeed", "vspeed", "yawrate")):
            sim[nm] = Fd[:, j]
    sim.to_csv(os.path.join(out, "similarity.csv"), index=False, lineterminator="\n")
    names = feature_names()
    sel = [j for j, nm in enumerate(names) if nm.startswith(("angspeed_p90", "tilt_", "pelvis_h", "hspeed",
                                                              "vspeed", "yawrate"))]
    tr_tab = pd.DataFrame({"archive": t_arch_s, "lpo": lpo / scale, "lao": lao / scale})
    for j in sel:
        tr_tab[names[j]] = Ft[:, j]
    by_train = {a: {"windows": int(len(g)), "lpo_median": float(g["lpo"].median()),
                    "lao_median": float(g["lao"].median()),
                    **{names[j]: float(g[names[j]].median()) for j in sel}}
                for a, g in tr_tab.groupby("archive")}
    by_dev = {a: {"windows": int(len(g)), "persons": int(g["person_id"].nunique()), "d_median": float(g["d"].median()),
                  "d_mean": float(g["d"].mean()), "d_p90": float(g["d"].quantile(0.9)),
                  "share_above_train_lpo_p95": float((g["knn_dist"] > np.quantile(lpo, 0.95)).mean()),
                  **{names[j]: float(np.median(Fd[g.index.to_numpy(), j])) for j in sel}}
              for a, g in sim.groupby("archive")}
    return {"train_windows": int(len(Ft)), "dev_windows": int(len(Fd)), "features": len(names),
            "pca_components": q, "pca_variance": float(np.cumsum(var)[q - 1]), "k": K_NN,
            "lpo_median_distance": scale, "lpo_p95_over_median": float(np.quantile(lpo, 0.95) / scale),
            "dev_length_mismatches": len_mismatch, "by_train_archive": by_train, "by_dev_archive": by_dev,
            "train_all": {"lpo_median": 1.0, "lao_median": float(np.median(lao) / scale)},
            "seconds": round(time.time() - t0, 1)}


def attainability():
    """From existing files: the decompose rows that bound the effects (already reported numbers)."""
    with open(os.path.join(DECOMPOSE, "decompose.json"), encoding="utf-8") as f:
        dj = json.load(f)
    keys = ("g1a_mm", "true_root_orient_mm", "true_betas_mm", "pa_mm", "floor_mm")
    out = {k: dj.get(k) for k in keys}
    by = {}
    for grp in ("by_archive",):
        if grp in dj:
            by = dj[grp]
    out["by_archive"] = by
    rows = {r.get("row"): r for r in dj.get("rows", [])} if isinstance(dj.get("rows"), list) else {}
    out["rows_present"] = sorted(rows)[:40]
    return out


def cmd_attain(a):
    """Attainability of the pre-registered criteria from files that exist: facts.json, similarity.csv (no error
    value) and the archive and gap numbers of the decomposition already quoted in the task."""
    sim = pd.read_csv(os.path.join(a.out, "similarity.csv"), encoding="utf-8")
    with open(os.path.join(DECOMPOSE, "decompose.json"), encoding="utf-8") as f:
        dj = json.load(f)
    out = {"g1a_mm": dj["g1a_mm"], "gap_mm": dj["g1a_mm"] - PASS_MM,
           "threshold_mm": GAP_SHARE * (dj["g1a_mm"] - PASS_MM)}
    rows = dj["rows"]
    for k in ("sub_legs", "sub_spine_head", "sub_arms", "true_root_orient", "true_betas", "pa", "floor"):
        r = rows.get(k)
        if r is not None:
            out[f"{k}_change_mm"] = r.get("delta_mm")
            out[f"{k}_value_mm"] = r.get("point_mm")
    dk = {}
    for arc, g in sim.groupby("archive"):
        dk[arc] = {"d_knot_mean": float(np.average(g["d"], weights=g["n"])), "windows": int(len(g)),
                   "persons": int(g["person_id"].nunique())}
    out["d_by_archive"] = dk
    out["d_sd_window"] = float(sim["d"].std())
    out["d_p10_p90"] = [float(sim["d"].quantile(0.1)), float(sim["d"].quantile(0.9))]
    ba = dj["by_archive"]
    out["archive_quoted"] = {k: {"g1a_knot_mm": v["g1a_mm"], "g1a_person_mm": v["g1a_person_mm"],
                                 "true_betas_knot_mm": v.get("true_betas_mm")} for k, v in ba.items()}
    ref = "EyesJapanDataset"
    need = {}
    for arc in dk:
        if arc == ref:
            continue
        dd = dk[arc]["d_knot_mean"] - dk[ref]["d_knot_mean"]
        gap = ba[arc]["g1a_mm"] - ba[ref]["g1a_mm"]
        need[arc] = {"d_diff": dd, "gap_knot_mm": gap,
                     "coef_needed_for_half_gap_mm_per_d": (0.5 * gap / dd) if dd > 0 else None,
                     "half_gap_reachable_with_positive_coef": bool(dd > 0)}
    out["d_diff_to_ref"] = need
    out["windows"], out["persons"] = int(len(sim)), int(sim["person_id"].nunique())
    write_json(os.path.join(a.out, "attain.json"), out)
    log(json.dumps(out, indent=1, default=_default))


def cmd_facts(a):
    os.makedirs(a.out, exist_ok=True)
    res = {"script": "scripts/g1d_Hlayout.py facts", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    res["exposure"] = exposure()
    res["placement"] = placement()
    res["similarity"] = similarity(a.out)
    from mdfm.train import provenance as PV
    res["moveatlas_head"] = PV.moveatlas_head()
    write_json(os.path.join(a.out, "facts.json"), res)
    log(json.dumps({k: res["exposure"][k] for k in res["exposure"] if not isinstance(res["exposure"][k], dict)},
                   indent=1, default=_default))
    log(json.dumps(res["similarity"], indent=1, default=_default))
    for r in res["placement"]["sites"]:
        log(r["site"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                        if "ratio" in k or "rms" in k or "R0" in k})


# ---------------------------------------------------------------- arms
def factory_windows(sampler, store, row_of, gset, idx, layout, seed):
    """The G1 windows idx rendered by the factory under one concrete layout, as G1 inputs (source synthetic,
    synthetic False, no body, duration n / 100 s, cov of the G1 window)."""
    from dataclasses import replace
    from mdfm.data import windows as W
    from mdfm.factory.synth import RenderSpec
    from mdfm.eval import infer as I
    w = gset.windows
    takes = np.array([row_of[str(w["take_id"].iat[j])] for j in idx], np.int64)
    starts = np.array([int(w["start"].iat[j]) for j in idx], np.int64)
    bodies = store.windows(takes, starts)
    ns = [int(w["n"].iat[j]) for j in idx]
    if [int(x) for x in bodies.lengths] != ns:
        raise ValueError("body-store window lengths differ from the G1 windows")
    rend = sampler.render(bodies, [RenderSpec(layout)] * len(idx), seed)
    out = []
    for j, rw in zip(idx, rend):
        g = I.g1_windows(gset, j, ("six",))["six"]
        streams = [replace(s, truth=None) for s in rw.streams]
        out.append(W.validate(W.Window(g.source, g.person_id, g.key, False, 0.0, g.duration, streams,
                                       dict(g.cov))))
    return out, bodies


def body_check(bodies, store, gset, idx):
    """G1a (m) of the forward kinematics of the body-store windows against the stored G1 truth."""
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.body.smplh import fk
    from probe_g1_decompose import g1a_of
    out = []
    with torch.no_grad():
        for b, j in enumerate(idx):
            n = int(bodies.lengths[b])
            rest = store.skel_rest[int(bodies.take[b])]
            R = rot6d_to_matrix(bodies.rot6d[b, :n].double())
            pos, _ = fk(R, bodies.transl[b, :n].double(), torch.as_tensor(rest))
            J, _ = gset.truth(j)
            out.append(g1a_of(pos.float().numpy(), J[:n]))
    return out


def cmd_arms(a):
    from mdfm.data.body_store import BodyStore
    from mdfm.body.smplh import SkeletonBank
    from mdfm.eval import infer as I
    from mdfm.factory import layouts as L
    from mdfm.factory.synth import SynthSampler
    from mdfm.gates import g1
    from mdfm.train import loop as LP
    import probe_g1_decompose as PD
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        LP.cap_memory(dev, a.memory_gb)
    t_start = time.time()
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    model = loaded.model
    gset = g1.G1Set("dev")
    train = set(loaded.persons.get("amass", ()))
    if set(gset.persons) & train:
        raise RuntimeError("a dev person among the checkpoint's training persons")
    store = BodyStore(split="dev")
    with torch.no_grad():
        store.skel_rest = SkeletonBank(torch.float64).rest(torch.from_numpy(store.betas),
                                                           torch.from_numpy(store.gender)).numpy()
    row_of = {t: i for i, t in enumerate(store.takes["take_id"])}
    sampler = SynthSampler(store=store, device=dev if a.render_device is None else a.render_device,
                           dtype=torch.float32)
    lays = L.load_layouts()
    concrete = {arm: L.concrete(lays[name], np.random.default_rng(0)) for arm, name in ARM_LAYOUT.items()}
    idx_all = PD.subset(gset, a.takes_per_archive)
    if a.windows_per_archive:
        keep = []
        for _, g in gset.windows.iloc[idx_all].groupby("archive", sort=True):
            takes = list(dict.fromkeys(g["take_id"].tolist()))
            rng = np.random.default_rng(SEED)
            rng.shuffle(takes)
            got = []
            for t in takes:
                if len(got) >= a.windows_per_archive:
                    break
                got += g.index[g["take_id"] == t].tolist()
            keep += got
        idx_all = sorted(int(i) for i in keep)
    path = os.path.join(a.out, "arms_windows.csv")
    done = set()
    if os.path.exists(path) and not a.time_only:
        prev = pd.read_csv(path, encoding="utf-8")
        done = set(prev["window"].astype(int)) if len(prev) else set()
    elif not a.time_only:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(ARM_COLUMNS)
    blk = I.G1Predictor(loaded, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                        block_windows=a.block_windows, seed=a.seed, pool=True, configs=("six",))
    blk.subset = idx_all
    log(f"arms: {len(idx_all)} windows, {len(done)} already written; device {dev}; checkpoint {loaded.name}")
    seen = set(done)
    nblk = 0
    for i in idx_all:
        if i in seen:
            continue
        idx = blk.block(gset, i)
        seen |= set(idx)
        if set(idx) <= done:
            continue
        tb = time.time()
        takes = [str(gset.windows["take_id"].iat[j]) for j in idx]
        seed = I.seed_of(a.seed, gset.split, takes[0], "six")
        wins = {"six_protocol": [I.g1_windows(gset, j, ("six",))["six"] for j in idx]}
        bodies = None
        for arm, lay in concrete.items():
            fseed = I.seed_of(a.seed, FACTORY_SALT, takes[0], arm)
            wins[arm], bodies = factory_windows(sampler, store, row_of, gset, idx, lay, fseed)
        bc = [math.nan] * len(idx) if a.time_only else body_check(bodies, store, gset, idx)
        res = {}
        for arm in ARMS:
            st = PD.posterior_pass(model, wins[arm], a.refine_steps, a.points, takes, seed, a.batch_windows,
                                   keep_amortized=True)
            res[arm] = st
        if a.time_only:
            log(f"time-only block of {len(idx)} windows: {time.time() - tb:.1f} s")
            nblk += 1
            if nblk >= a.time_only:
                return
            continue
        rows = []
        for arm in ARMS:
            st = res[arm]
            a0 = 0
            for c in st["chunks"]:
                Wc = c["post"].mu_z.shape[0]
                jj = idx[a0:a0 + Wc]
                mb, mb0 = st["mu_p"][a0:a0 + Wc], st["mu_p0"][a0:a0 + Wc]
                bcc = bc[a0:a0 + Wc]
                a0 += Wc
                pos = PD._decode(model, c["post"], mb, c, None)[1].cpu().numpy()
                pos0 = PD._decode(model, c["post0"], mb0, c, None)[1].cpu().numpy()
                for (s0, n), j, b in zip(PD._cuts(c), jj, bcc):
                    J, _ = gset.truth(j)
                    r = gset.windows.iloc[int(j)]
                    rows.append([int(j), r["take_id"], r["archive"], r["person_id"], int(r["task"]), int(n), arm,
                                 PD.g1a_of(pos[s0:s0 + n], J[:n]), PD.g1a_of(pos0[s0:s0 + n], J[:n]), b])
        with open(path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
        done |= set(idx)
        nblk += 1
        log(f"block {nblk}: {len(idx)} windows ({len(done)} of {len(idx_all)}), {time.time() - tb:.1f} s, "
            f"{time.time() - t_start:.0f} s in all")
    prov = {"script": "scripts/g1d_Hlayout.py arms", "argv": sys.argv[1:], "checkpoint": common_rel(a.checkpoint),
            "checkpoint_sha256": loaded.sha256, "windows": len(idx_all), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "torch": torch.__version__, "device": str(dev), "seconds": round(time.time() - t_start, 1)}
    try:
        from mdfm.train import provenance as PV
        prov["moveatlas_head"] = PV.moveatlas_head()
    except Exception as e:  # noqa: BLE001
        prov["moveatlas_head"] = f"unavailable: {e}"
    write_json(os.path.join(a.out, "arms_provenance.json"), prov)


# ---------------------------------------------------------------- analyze
def pw(df, col, weight="n"):
    """The G1 statistic: mean over persons of the knot-weighted person means."""
    from mdfm.gates import common
    return common.person_weighted_mean(df[col].to_numpy(), df["person_id"].to_numpy(), df[weight].to_numpy())


def boot_diff(df, a_col, b_col, seed=SEED, n=2000):
    from mdfm.gates import common
    v = np.stack([df[a_col].to_numpy(), df[b_col].to_numpy()], 1)
    lo, hi = common.cluster_bootstrap(v, df["person_id"].to_numpy(), n=n, seed=seed, weights=df["n"].to_numpy(),
                                      stat=lambda m: m[:, 0] - m[:, 1])
    return lo, hi


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.correlation if hasattr(r, "correlation") else r[0])


def within_rank(df, col):
    return df.groupby("archive")[col].rank(pct=True)


def partial_spearman(df, x, y, z):
    """Spearman of x and y after removing a linear fit on the ranks of z (all ranks within archive)."""
    rx, ry, rz = within_rank(df, x), within_rank(df, y), within_rank(df, z)
    def resid(u):
        A = np.stack([np.ones(len(rz)), rz.to_numpy()], 1)
        beta, *_ = np.linalg.lstsq(A, u.to_numpy(), rcond=None)
        return u.to_numpy() - A @ beta
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def person_boot_stat(df, fn, n=2000, seed=SEED):
    persons = df["person_id"].unique()
    groups = {p: g for p, g in df.groupby("person_id")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(persons, size=len(persons), replace=True)
        d = pd.concat([groups[p] for p in pick], ignore_index=True)
        try:
            vals.append(fn(d))
        except Exception:  # noqa: BLE001
            continue
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (math.nan, math.nan)


def gap_explained(df, ref="EyesJapanDataset"):
    """OLS of window G1a (mm) on d and motion_m (knot-weighted), the archive gaps to ref predicted by d alone (the d
    coefficient times the archives' knot-weighted mean d difference) over the observed gaps."""
    w = df["n"].to_numpy(float)
    X = np.stack([np.ones(len(df)), df["d"].to_numpy(), df["motion_m"].to_numpy()], 1)
    y = MM * df["g1a_m"].to_numpy()
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    out = {"coef_d_mm": float(beta[1]), "coef_motion_mm_per_m": float(beta[2])}
    m = df.assign(w=w, y=y).groupby("archive").apply(
        lambda g: pd.Series({"d": np.average(g["d"], weights=g["w"]), "y": np.average(g["y"], weights=g["w"])}),
        include_groups=False)
    for arc in m.index:
        if arc == ref:
            continue
        obs = float(m.loc[arc, "y"] - m.loc[ref, "y"])
        pred = float(beta[1] * (m.loc[arc, "d"] - m.loc[ref, "d"]))
        out[arc] = {"observed_gap_mm": obs, "predicted_by_d_mm": pred, "share": pred / obs if obs else math.nan}
    return out


def cmd_analyze(a):
    res = {"script": "scripts/g1d_Hlayout.py analyze", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    sim = pd.read_csv(os.path.join(a.out, "similarity.csv"), encoding="utf-8")
    dec = pd.read_csv(os.path.join(DECOMPOSE, "windows.csv"), encoding="utf-8")[
        ["window", "motion_m", "true_betas_m"]]
    res["archive"] = {}
    for tag, name in GATE_RUNS.items():
        p = os.path.join(TREE, "reports", "gates", "g1", name, "windows.csv")
        if not os.path.exists(p):
            continue
        gw = pd.read_csv(p, encoding="utf-8")[["window", "take_id", "g1a_m"]]
        df = sim.merge(gw, on=["window", "take_id"]).merge(dec, on="window")
        if len(df) != len(sim):
            raise ValueError(f"{tag}: {len(df)} joined windows of {len(sim)}")
        r = {"source": common_rel(p), "windows": int(len(df)), "g1a_mm": MM * pw(df, "g1a_m")}
        r["by_archive"] = {arc: {"windows": int(len(g)), "persons": int(g["person_id"].nunique()),
                                 "g1a_mm": MM * pw(g, "g1a_m"),
                                 "g1a_knot_mm": MM * float(np.average(g["g1a_m"], weights=g["n"])),
                                 "true_betas_knot_mm": MM * float(np.average(g["true_betas_m"], weights=g["n"])),
                                 "d_median": float(g["d"].median()), "d_mean": float(g["d"].mean()),
                                 "motion_mm_median": MM * float(g["motion_m"].median())}
                           for arc, g in df.groupby("archive")}
        order_d = sorted(r["by_archive"], key=lambda k: r["by_archive"][k]["d_median"])
        order_e = sorted(r["by_archive"], key=lambda k: r["by_archive"][k]["g1a_mm"])
        r["order_by_d"], r["order_by_g1a"] = order_d, order_e
        r["spearman_pooled"] = spearman(df["d"], df["g1a_m"])
        r["spearman_within_archive"] = spearman(within_rank(df, "d"), within_rank(df, "g1a_m"))
        r["spearman_within_by_archive"] = {arc: spearman(g["d"], g["g1a_m"]) for arc, g in df.groupby("archive")}
        r["partial_within_motion"] = partial_spearman(df, "d", "g1a_m", "motion_m")
        r["partial_within_motion_ci"] = person_boot_stat(df, lambda d: partial_spearman(d, "d", "g1a_m", "motion_m"),
                                                         n=a.boot)
        r["spearman_d_motion_within"] = spearman(within_rank(df, "d"), within_rank(df, "motion_m"))
        pm = df.assign(e=df["g1a_m"] * df["n"], dn=df["d"] * df["n"]).groupby("person_id")[["e", "dn", "n"]].sum()
        r["spearman_person"] = spearman(pm["dn"] / pm["n"], pm["e"] / pm["n"])
        r["persons"] = int(len(pm))
        r["gap_explained"] = gap_explained(df)
        # not a criterion: the fitted d coefficient times the distance of the dev mean d from the train
        # leave-person-out median (d = 1), an extrapolation outside the dev range of d
        dmean = float(np.average(df["d"], weights=df["n"]))
        r["extrapolation_to_d1"] = {"d_knot_mean": dmean,
                                    "drop_mm": r["gap_explained"]["coef_d_mm"] * (dmean - 1.0),
                                    "share_windows_d_below_1_1": float((df["d"] < 1.1).mean())}
        shares = {}
        for arc in ("HDM05", "SFU"):
            shares[arc] = person_boot_stat(df, lambda d, arc=arc: gap_explained(d)[arc]["share"]
                                           if arc in set(d["archive"]) and "EyesJapanDataset" in set(d["archive"])
                                           else math.nan, n=a.boot)
        r["gap_share_ci"] = shares
        res["archive"][tag] = r
    p = os.path.join(a.out, "arms_windows.csv")
    aw = pd.read_csv(p, encoding="utf-8") if os.path.exists(p) else None
    if aw is not None and aw.groupby("window")["arm"].nunique().eq(len(ARMS)).sum() >= 2:
        wide = aw.pivot_table(index=["window", "take_id", "archive", "person_id", "n"], columns="arm",
                              values=["g1a_m", "refine0_m"]).reset_index()
        wide.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in wide.columns]
        wide = wide.dropna()
        bc = aw.groupby("window")["body_check_m"].max()
        arms = {"windows": int(len(wide)), "persons": int(wide["person_id"].nunique()),
                "takes": int(wide["take_id"].nunique()),
                "archives": {k: int(v) for k, v in wide.groupby("archive").size().items()},
                "body_check_max_mm": MM * float(bc.max()), "body_check_mean_mm": MM * float(bc.mean())}
        for arm in ARMS:
            for m in ("g1a_m", "refine0_m"):
                arms[f"{arm}_{m[:-2]}_mm"] = MM * pw(wide, f"{m}_{arm}")
        six = arms["six_protocol_g1a_mm"]
        gap = six - PASS_MM
        arms["gap_mm"] = gap
        for name, x, y in (("placement", "six_protocol", "six_factory"), ("layout", "six_factory", "xs13_factory"),
                           ("total", "six_protocol", "xs13_factory")):
            for m in ("g1a_m", "refine0_m"):
                ca, cb = f"{m}_{x}", f"{m}_{y}"
                d = MM * (pw(wide, ca) - pw(wide, cb))
                lo, hi = boot_diff(wide, ca, cb, n=a.boot)
                arms[f"{name}_{m[:-2]}"] = {"diff_mm": d, "ci_mm": [MM * lo, MM * hi], "share_of_gap": d / gap,
                                            "what": f"{x} minus {y}"}
        # harness check: six_protocol against the gate run on the same windows
        gw = pd.read_csv(os.path.join(TREE, "reports", "gates", "g1", GATE_RUNS["v1"], "windows.csv"),
                         encoding="utf-8")[["window", "g1a_m"]].rename(columns={"g1a_m": "gate_g1a_m"})
        hw = wide.merge(gw, on="window")
        arms["harness_gate_same_windows_mm"] = MM * pw(hw, "gate_g1a_m")
        arms["harness_max_abs_window_diff_mm"] = MM * float((hw["gate_g1a_m"] - hw["g1a_m_six_protocol"]).abs().max())
        arms["by_archive"] = {arc: {arm: MM * pw(g, f"g1a_m_{arm}") for arm in ARMS}
                              for arc, g in wide.groupby("archive")}
        res["arms"] = arms
        prov = os.path.join(a.out, "arms_provenance.json")
        if os.path.exists(prov):
            with open(prov, encoding="utf-8") as f:
                res["arms_provenance"] = json.load(f)
    write_json(os.path.join(a.out, "results.json"), res)
    log(json.dumps(res, indent=1, default=_default))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--out", default=OUT)
    t = sub.add_parser("attain")
    t.add_argument("--out", default=OUT)
    r = sub.add_parser("arms")
    r.add_argument("--checkpoint", default=CKPT)
    r.add_argument("--weights", default="ema", choices=("ema", "model"))
    r.add_argument("--device", default="cpu")
    r.add_argument("--render-device", default=None, help="device of the factory render (default --device)")
    r.add_argument("--memory-gb", type=float, default=12.0)
    r.add_argument("--out", default=OUT)
    r.add_argument("--refine-steps", type=int, default=20)
    r.add_argument("--points", type=int, default=256)
    r.add_argument("--batch-windows", type=int, default=32)
    r.add_argument("--block-windows", type=int, default=32)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--takes-per-archive", type=int, default=None, help="the first K takes of each archive")
    r.add_argument("--windows-per-archive", type=int, default=None,
                   help="whole takes drawn (seeded) per archive until this many windows")
    r.add_argument("--time-only", type=int, default=0, help="time this many blocks, score nothing, write nothing")
    z = sub.add_parser("analyze")
    z.add_argument("--out", default=OUT)
    z.add_argument("--boot", type=int, default=2000)
    return p.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    a = parse(argv)
    {"facts": cmd_facts, "attain": cmd_attain, "arms": cmd_arms, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
