"""G1D-Hstand: is the G1a error of the six-IMU body dominated by static and quasi-static windows, where six
accelerometers and gyroscopes leave the orientation of a still limb about the vertical axis undetermined?

Stages (each writes under --out, default reports/bench/g1_diag/Hstand):
- energy: per dev window the IMU signal energy read from the G1 dev inputs (cache/g1/g1_v1/dev through
  mdfm.gates.g1.G1Set; no error value, no model): per site the RMS over knots of |gyro - its window mean|
  (deg/s) and of |acc - its window mean| (g), the window energy (mean over the six sites of the gyro RMS, the
  primary measure), the arm and leg energies (wrists, shanks), the still share (knots whose largest site
  deviation from the site's window median gyro is under STILL_DPS); quintiles of the window energy; the knot
  share of each quintile and of each task (the attainability numbers of the prediction). Writes energy.csv,
  energy.json.
- analyze: joins energy.csv with the per-window G1a of the existing dev runs (reports/gates/g1/dev_<name>/
  windows.csv, default the v1 phase B end and seed 2, and the decompose windows.csv): error by energy quintile,
  task and archive (knot-weighted mean, G1 statistic, share of the error mass sum n * g1a), the counterfactual
  G1 statistic with static windows removed or down-weighted (an analysis weighting, not a rule change) with
  person-clustered bootstrap intervals of the change, and the pre-registered criteria C1 and C2. Writes
  analysis.json, analysis.md.
- dump: one model pass (the harness inference of scripts/probe_g1_decompose.py: encode, 20 refinement steps,
  posterior mean, shape pooled per take, blocks of 32) on the dev windows, scored per window by the axis split of
  the error: (a) per segment the world-frame error rotation E = rot_y(theta) R_pred R_true^T (theta the window's
  G1a yaw) split into its twist about the vertical axis and the remaining swing (tilt), (b) per bone the
  direction error split into elevation and azimuth about the vertical. Needs the truth rotations
  (probe_g1_decompose.TruthBodies, dev takes only). Writes axis_windows.csv (one row per window, sums of squared
  angles per group) and axis_dump.json. A GPU run goes through scripts/queue_job.py.
- axis: joins axis_windows.csv with energy.csv and reports the vertical-axis share per energy quintile and group
  (criterion C3). Writes axis.json, axis.md.

Data: the dev split only (G1Set refuses the lockbox; TruthBodies reads dev and calib takes only); no EVAL, CODY or
care_pd record. Run (mdfm environment, CPU): PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
  <conda>/envs/mdfm/python.exe scripts/g1d_Hstand.py --stage energy
"""
import argparse
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

import numpy as np  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench", "g1_diag", "Hstand")
MM = 1000.0
STILL_DPS = 20.0          # a knot is still when every site's gyro stays within this of its window median (deg/s)
N_Q = 5                   # energy quintiles
ARM_SITES = ("wrist_L", "wrist_R")
LEG_SITES = ("shank_L", "shank_R")
RUNS = {"v1_B_end": "reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/windows.csv",
        "seed2_B_end": "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_032044/windows.csv",
        "decompose_v1_B_end": "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/windows.csv"}


# ---------------------------------------------------------------- energy (inputs only)
def window_energy(imu, sites):
    """Energy features of one window from imu (n, 6, 6) float (acc g, gyro deg/s) of the sites `sites`:
    {gyro_rms_<site>, acc_rms_<site>, energy_dps, arm_dps, leg_dps, acc_g, still_share}."""
    x = np.asarray(imu, dtype=np.float64)
    acc, gyr = x[:, :, :3], x[:, :, 3:]
    g_dev = gyr - gyr.mean(0, keepdims=True)
    a_dev = acc - acc.mean(0, keepdims=True)
    g_rms = np.sqrt((g_dev ** 2).sum(-1).mean(0))          # (6,)
    a_rms = np.sqrt((a_dev ** 2).sum(-1).mean(0))
    g_med = np.median(gyr, axis=0, keepdims=True)
    still = (np.linalg.norm(gyr - g_med, axis=-1).max(-1) < STILL_DPS).mean()
    out = {}
    for k, s in enumerate(sites):
        out[f"gyro_rms_{s}"] = float(g_rms[k])
        out[f"acc_rms_{s}"] = float(a_rms[k])
    idx = {s: k for k, s in enumerate(sites)}
    out["energy_dps"] = float(g_rms.mean())
    out["arm_dps"] = float(np.mean([g_rms[idx[s]] for s in ARM_SITES]))
    out["leg_dps"] = float(np.mean([g_rms[idx[s]] for s in LEG_SITES]))
    out["acc_g"] = float(a_rms.mean())
    out["still_share"] = float(still)
    return out


def quintile(values, n_q=N_Q):
    """Quintile labels 1..n_q (1 the lowest) of values by equal window counts (rank based, ties by order)."""
    v = np.asarray(values, dtype=np.float64)
    r = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return (r * n_q // len(v) + 1).astype(int)


def stage_energy(a, log):
    import pandas as pd
    from mdfm.gates import g1
    from mdfm.data.windows import task_vocabulary
    names = list(task_vocabulary()["tasks"])
    gs = g1.G1Set("dev", a.root, need_inputs=True)
    rows = []
    t0 = time.time()
    for i in range(len(gs)):
        r = gs.windows.iloc[i]
        inp = gs.inputs(i)
        row = {"window": int(i), "shard": int(r["shard"]), "shard_window": int(r["window"]),
               "take_id": r["take_id"], "archive": r["archive"],
               "person_id": r["person_id"], "task": int(r["task"]), "task_name": names[int(r["task"])],
               "n": int(r["n"])}
        row.update(window_energy(inp["imu"], g1.SITES))
        row["imu_valid_share"] = float(inp["imu_valid"].mean())
        rows.append(row)
        if (i + 1) % 500 == 0:
            log(f"energy: {i + 1} windows, {time.time() - t0:.0f} s")
    t = pd.DataFrame(rows)
    for col, q in (("energy_dps", "q_energy"), ("arm_dps", "q_arm"), ("leg_dps", "q_leg")):
        t[q] = quintile(t[col].to_numpy())
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "energy.csv"), index=False, encoding="utf-8", lineterminator="\n")
    K = float(t["n"].sum())
    res = {"what": "IMU signal energy of the G1 dev windows (inputs only, no error value)",
           "manifest_sha256": gs.manifest["manifest_sha256"], "n_windows": int(len(t)), "n_knots": int(K),
           "n_persons": int(t["person_id"].nunique()), "still_dps": STILL_DPS, "sites": list(g1.SITES),
           "quintiles": [], "tasks": {}, "archives": {}}
    for q, s in t.groupby("q_energy"):
        res["quintiles"].append({"q": int(q), "n_windows": int(len(s)), "knot_share": float(s["n"].sum() / K),
                                 "energy_lo_dps": float(s["energy_dps"].min()),
                                 "energy_hi_dps": float(s["energy_dps"].max()),
                                 "still_share_mean": float(np.average(s["still_share"], weights=s["n"])),
                                 "n_persons": int(s["person_id"].nunique()),
                                 "stand_or_sit_stand_windows": int(s["task_name"].isin(["stand", "sit_stand"]).sum())})
    for key, col in (("tasks", "task_name"), ("archives", "archive")):
        for g, s in t.groupby(col):
            res[key][str(g)] = {"n_windows": int(len(s)), "knot_share": float(s["n"].sum() / K),
                                "energy_median_dps": float(s["energy_dps"].median()),
                                "q1_windows": int((s["q_energy"] == 1).sum())}
    res["still_windows_over_half"] = int((t["still_share"] > 0.5).sum())
    res["still_knot_share"] = float(np.average(t["still_share"], weights=t["n"]))
    res["time_s"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "energy.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    log(json.dumps({k: v for k, v in res.items() if k not in ("tasks", "archives")}, indent=1))
    log(json.dumps(res["tasks"], indent=1))
    return res


# ---------------------------------------------------------------- statistics
def person_table(t, col, weight="n"):
    """(persons (P,), person means (P,)) of column col weighted by column weight."""
    from mdfm.gates import common
    ids, m, _ = common.person_means(t[col].to_numpy(dtype=np.float64), t["person_id"].to_numpy(),
                                    t[weight].to_numpy(dtype=np.float64))
    return ids, m


def g1_stat(t, col="g1a_m", weight="n"):
    """The G1 statistic: mean over persons of the weighted person means."""
    return float(person_table(t, col, weight)[1].mean())


def kw_mean(t, col="g1a_m"):
    w = t["n"].to_numpy(dtype=np.float64)
    return float(np.sum(t[col].to_numpy(dtype=np.float64) * w) / np.sum(w))


def counterfactual(t, w_new, col="g1a_m", n_boot=2000, seed=20260929, level=0.95):
    """The G1 statistic under knot weights n * w_new (w_new (N,) >= 0; a person whose weights all vanish drops out)
    against the full one: {full_mm, reweighted_mm, change_mm (reweighted minus full), interval_mm (person-clustered
    percentile bootstrap of the change, persons resampled with replacement), persons_full, persons_kept}."""
    v = t[col].to_numpy(dtype=np.float64)
    p = t["person_id"].to_numpy().astype(str)
    n = t["n"].to_numpy(dtype=np.float64)
    w2 = n * np.asarray(w_new, dtype=np.float64)
    ids, inv = np.unique(p, return_inverse=True)
    s1, s2 = np.bincount(inv, n * v, len(ids)), np.bincount(inv, w2 * v, len(ids))
    k1, k2 = np.bincount(inv, n, len(ids)), np.bincount(inv, w2, len(ids))
    full = s1 / k1
    keep = k2 > 0
    red = np.where(keep, s2 / np.where(keep, k2, 1.0), np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(int(n_boot), len(ids)))
    f_b = full[idx].mean(1)
    r_b = np.nanmean(red[idx], 1)
    d_b = r_b - f_b
    a = (1.0 - level) / 2.0
    lo, hi = np.nanquantile(d_b, [a, 1 - a])
    F, R = float(full.mean()), float(np.nanmean(red))
    return {"full_mm": MM * F, "reweighted_mm": MM * R, "change_mm": MM * (R - F),
            "interval_mm": [MM * float(lo), MM * float(hi)], "persons_full": int(len(ids)),
            "persons_kept": int(keep.sum())}


def group_table(t, key, col="g1a_m"):
    """Per value of key: windows, persons, knot share, knot-weighted mean (mm), G1 statistic (mm), share of the
    error mass sum n * col, ratio of the error share to the knot share."""
    K = float(t["n"].sum())
    mass = float((t["n"] * t[col]).sum())
    out = []
    for g, s in t.groupby(key, sort=True):
        ks = float(s["n"].sum()) / K
        es = float((s["n"] * s[col]).sum()) / mass
        out.append({key: g if isinstance(g, str) else int(g), "n_windows": int(len(s)),
                    "n_persons": int(s["person_id"].nunique()), "knot_share": ks,
                    "kw_mm": MM * kw_mean(s, col), "g1_mm": MM * g1_stat(s, col), "error_share": es,
                    "error_over_knot_share": es / ks})
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def read_run(path_rel, energy):
    """The per-window G1a of one run joined with energy.csv on the window index (take ids checked equal)."""
    import pandas as pd
    w = pd.read_csv(os.path.join(TREE, path_rel), encoding="utf-8")
    keep = [c for c in w.columns if c in ("window", "take_id", "g1a_m") or (c.startswith("j") and c.endswith("_m"))]
    j = energy.merge(w[keep], on="window", suffixes=("", "_run"), validate="one_to_one")
    bad = int((j["take_id"] != j["take_id_run"]).sum())
    if bad or len(j) != len(w):
        raise ValueError(f"{path_rel}: {bad} windows whose take differs from energy.csv, {len(j)} of {len(w)} joined")
    return j.drop(columns=["take_id_run"])


def criteria(qt, overall_kw_mm, cf_q1, cf_q1_other=None):
    """C1 and C2 of the prediction from the quintile table, the overall knot-weighted G1a (mm) and the Q1-removal
    counterfactual(s)."""
    q1 = next(r for r in qt if r["q_energy"] == 1)
    c1 = {"q1_kw_mm": q1["kw_mm"], "overall_kw_mm": overall_kw_mm, "q1_error_share": q1["error_share"],
          "q1_knot_share": q1["knot_share"]}
    c1["holds"] = bool(q1["kw_mm"] > overall_kw_mm and q1["error_share"] >= 0.30)
    drop = -cf_q1["change_mm"]
    c2 = {"drop_mm": drop, "holds": bool(drop >= 8.0)}
    if cf_q1_other is not None:
        c2["other_seed_drop_mm"] = -cf_q1_other["change_mm"]
        c2["same_sign"] = bool(np.sign(drop) == np.sign(-cf_q1_other["change_mm"]))
        c2["holds"] = c2["holds"] and c2["same_sign"]
    return c1, c2


def stage_analyze(a, log):
    import pandas as pd
    from mdfm.body.smplh import BODY_JOINTS, N_BODY
    e = pd.read_csv(os.path.join(a.out, "energy.csv"), encoding="utf-8")
    groups = {"arms": [j for j, nm in enumerate(BODY_JOINTS[:N_BODY]) if any(w in nm for w in ("collar", "shoulder",
                                                                                                 "elbow", "wrist"))],
              "legs": [j for j, nm in enumerate(BODY_JOINTS[:N_BODY]) if any(w in nm for w in ("hip", "knee", "ankle",
                                                                                                 "foot"))]}
    res = {"what": "G1a by IMU energy, task and archive; counterfactual removal and down-weighting of static windows",
           "energy_csv_sha256": sha256_file(os.path.join(a.out, "energy.csv")), "runs": {}}
    tabs = {}
    for name, rel in RUNS.items():
        t = read_run(rel, e)
        tabs[name] = t
        r = {"source": rel, "source_sha256": sha256_file(os.path.join(TREE, rel)), "n_windows": int(len(t)),
             "g1_mm": MM * g1_stat(t), "kw_mm": MM * kw_mean(t),
             "spearman_energy_g1a": spearman(t["energy_dps"], t["g1a_m"]),
             "spearman_still_share_g1a": spearman(t["still_share"], t["g1a_m"]),
             "by_quintile": group_table(t, "q_energy"), "by_task": group_table(t, "task_name"),
             "by_archive": group_table(t, "archive")}
        r["by_archive_quintile"] = {arc: group_table(s, "q_energy") for arc, s in t.groupby("archive")}
        q = t["q_energy"].to_numpy()
        task = t["task_name"].to_numpy()
        cfs = {"remove_q1": (q != 1).astype(float), "remove_q1_q2": (q > 2).astype(float),
               "half_weight_q1": np.where(q == 1, 0.5, 1.0), "remove_q5": (q != 5).astype(float),
               "remove_stand_sit_stand": (~np.isin(task, ["stand", "sit_stand"])).astype(float),
               "remove_still_share_over_half": (t["still_share"].to_numpy() <= 0.5).astype(float),
               "weight_by_energy_rank": (q / 3.0)}
        r["counterfactual"] = {k: counterfactual(t, w) for k, w in cfs.items()}
        jcols = [f"j{j:02d}_m" for j in range(N_BODY)]
        if all(c in t.columns for c in jcols):
            for g, js in groups.items():
                t[f"{g}_m"] = t[[jcols[j] for j in js]].mean(axis=1)
            r["limb"] = {"arms_by_arm_quintile": group_table(t, "q_arm", "arms_m"),
                         "legs_by_leg_quintile": group_table(t, "q_leg", "legs_m"),
                         "spearman_arm_energy_arm_error": spearman(t["arm_dps"], t["arms_m"]),
                         "spearman_leg_energy_leg_error": spearman(t["leg_dps"], t["legs_m"])}
        res["runs"][name] = r
        log(f"{name}: G1 {r['g1_mm']:.2f} mm, kw {r['kw_mm']:.2f} mm, rho(energy, g1a) {r['spearman_energy_g1a']:.3f}")
    v1, s2 = res["runs"]["v1_B_end"], res["runs"]["seed2_B_end"]
    c1, c2 = criteria(v1["by_quintile"], v1["kw_mm"], v1["counterfactual"]["remove_q1"],
                      s2["counterfactual"]["remove_q1"])
    res["criteria"] = {"C1": c1, "C2": c2}
    if c1["q1_kw_mm"] <= c1["overall_kw_mm"] or c2["drop_mm"] < 3.0:
        v = "not_supported"
    elif c1["holds"] and c2["holds"]:
        v = "undetermined_needs_gpu (C3 pending)"
    else:
        v = "partly"
    res["verdict_c1_c2"] = v
    with open(os.path.join(a.out, "analysis.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    md = analysis_markdown(res)
    with open(os.path.join(a.out, "analysis.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    log(md)
    return res


def _f(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _gt(rows, key, title):
    L = [f"| {title} | windows | persons | knot share | knot-weighted mm | G1 statistic mm | error share | "
         "error share / knot share |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r[key]} | {r['n_windows']:,} | {r['n_persons']} | {_f(r['knot_share'], 3)} | {_f(r['kw_mm'])} | "
                 f"{_f(r['g1_mm'])} | {_f(r['error_share'], 3)} | {_f(r['error_over_knot_share'], 2)} |")
    return L


def analysis_markdown(res):
    L = ["# G1D-Hstand analysis", "", "Written by scripts/g1d_Hstand.py --stage analyze from energy.csv (SHA-256 "
         f"{res['energy_csv_sha256']}) and the per-window G1a of the runs below. Dev windows only. Q1 is the stillest "
         "fifth of the windows by gyro energy (mean over the six sites of the RMS of |gyro - window mean|).", ""]
    for name, r in res["runs"].items():
        L += [f"## {name}", "", f"Source {r['source']} (SHA-256 {r['source_sha256']}), {r['n_windows']:,} windows. "
              f"G1 statistic {_f(r['g1_mm'], 2)} mm, knot-weighted {_f(r['kw_mm'], 2)} mm. Spearman rank correlation "
              f"over windows: energy and G1a {_f(r['spearman_energy_g1a'], 3)}, still share and G1a "
              f"{_f(r['spearman_still_share_g1a'], 3)}.", ""]
        L += _gt(r["by_quintile"], "q_energy", "energy quintile") + [""]
        L += _gt(r["by_task"], "task_name", "task") + [""]
        L += _gt(r["by_archive"], "archive", "archive") + [""]
        L += ["| counterfactual weighting | G1 full mm | G1 reweighted mm | change mm | 95% interval | persons kept |",
              "|---|---|---|---|---|---|"]
        for k, c in r["counterfactual"].items():
            L.append(f"| {k} | {_f(c['full_mm'], 2)} | {_f(c['reweighted_mm'], 2)} | {_f(c['change_mm'], 2)} | "
                     f"{_f(c['interval_mm'][0], 2)} to {_f(c['interval_mm'][1], 2)} | {c['persons_kept']} of "
                     f"{c['persons_full']} |")
        L.append("")
        if "limb" in r:
            lb = r["limb"]
            L += [f"Limb errors (mean of the group's per-joint G1a errors) by the limb's own gyro energy quintile; "
                  f"Spearman arm energy and arm error {_f(lb['spearman_arm_energy_arm_error'], 3)}, leg energy and "
                  f"leg error {_f(lb['spearman_leg_energy_leg_error'], 3)}.", ""]
            L += _gt(lb["arms_by_arm_quintile"], "q_arm", "arm energy quintile (arm joints)") + [""]
            L += _gt(lb["legs_by_leg_quintile"], "q_leg", "leg energy quintile (leg joints)") + [""]
        for arc, rows in r["by_archive_quintile"].items():
            L += _gt(rows, "q_energy", f"{arc} energy quintile") + [""]
    c = res["criteria"]
    L += ["## Criteria (pre-registered)", "",
          f"- C1: Q1 knot-weighted {_f(c['C1']['q1_kw_mm'])} mm against overall {_f(c['C1']['overall_kw_mm'])} mm; "
          f"Q1 error share {_f(c['C1']['q1_error_share'], 3)} (knot share {_f(c['C1']['q1_knot_share'], 3)}, needed "
          f"0.30): {'holds' if c['C1']['holds'] else 'fails'}.",
          f"- C2: removing Q1 lowers the G1 statistic by {_f(c['C2']['drop_mm'], 2)} mm (seed 2: "
          f"{_f(c['C2'].get('other_seed_drop_mm'), 2)} mm; needed 8 mm, same sign): "
          f"{'holds' if c['C2']['holds'] else 'fails'}.",
          f"- Verdict on C1 and C2: {res['verdict_c1_c2']}.", ""]
    return "\n".join(L)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- axis split (model dump)
def twist_swing(E):
    """E (..., 3, 3) rotations -> (twist, swing) (...) rad: E = S T with T about the vertical +y (the twist,
    signed, 2 atan2(E02 - E20, 1 + trace)) and S about a horizontal axis (the swing, acos E11, the tilt of the
    vertical)."""
    E = np.asarray(E, dtype=np.float64)
    tr = E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2]
    tw = 2.0 * np.arctan2(E[..., 0, 2] - E[..., 2, 0], 1.0 + tr)
    tw = (tw + np.pi) % (2 * np.pi) - np.pi
    sw = np.arccos(np.clip(E[..., 1, 1], -1.0, 1.0))
    return tw, sw


def rot_y_np(theta):
    c, s = np.cos(theta), np.sin(theta)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def bone_angles(P, parents):
    """Elevation (asin of the vertical component) and azimuth (atan2(x, z)) of every bone (child minus parent),
    P (n, 22, 3); bones of joints 1..21. Returns (el (n, 21), az (n, 21), length (n, 21))."""
    b = P[:, 1:] - P[:, list(parents[1:])]
    L = np.linalg.norm(b, axis=-1)
    u = b / np.maximum(L[..., None], 1e-9)
    return np.arcsin(np.clip(u[..., 1], -1, 1)), np.arctan2(u[..., 0], u[..., 2]), L


AXIS_GROUPS = ("arms", "legs", "spine_head", "root", "limbs")


def axis_window(pos_p, Rg_p, J, Rg_t, groups, parents):
    """Axis split of one window (numpy, float64): pos_p (n, 22, 3) predicted joints, Rg_p (n, 22, 3, 3) predicted
    global rotations, J (n, 22, 3) and Rg_t the truth. The window's G1a yaw theta turns the prediction onto the
    truth. Returns {g1a_m, theta, and per group g: tw2_g, sw2_g (sums over knots and segments of the squared twist
    and swing angles of E = rot_y(theta) Rg_p Rg_t^T), twr2_g (twist after removing the pelvis's own twist at the
    knot), seg_g (count), ey2_g, exz2_g (sums of the squared vertical and horizontal joint error components, m^2,
    pelvis-relative, after the yaw), el2_g, az2_g (sums of the squared elevation error and of the azimuth error times
    the cosine of the true elevation, of the bones ending at the group's joints)}."""
    from mdfm.gates import g1
    n = J.shape[0]
    t = J - J[:, :1]
    p = pos_p[:n] - pos_p[:n, :1]
    th = g1.yaw_fit(t, p)
    pt = g1.turn_y(p, th)
    Ry = rot_y_np(np.array(th))
    E = Ry @ Rg_p[:n] @ np.swapaxes(Rg_t[:n], -1, -2)
    tw, sw = twist_swing(E)
    twr = (tw - tw[:, :1] + np.pi) % (2 * np.pi) - np.pi
    e = pt - t
    el_p, az_p, _ = bone_angles(pt, parents)
    el_t, az_t, _ = bone_angles(t, parents)
    dl = el_p - el_t
    da = (az_p - az_t + np.pi) % (2 * np.pi) - np.pi
    da = da * np.cos(el_t)
    out = {"g1a_m": float(np.linalg.norm(e, axis=-1).mean()), "theta": float(th), "n": int(n)}
    for g, js in groups.items():
        js = list(js)
        out[f"tw2_{g}"] = float((tw[:, js] ** 2).sum())
        out[f"sw2_{g}"] = float((sw[:, js] ** 2).sum())
        out[f"twr2_{g}"] = float((twr[:, js] ** 2).sum())
        out[f"seg_{g}"] = int(n * len(js))
        out[f"ey2_{g}"] = float((e[:, js, 1] ** 2).sum())
        out[f"exz2_{g}"] = float((e[:, js, 0] ** 2 + e[:, js, 2] ** 2).sum())
        bj = [j - 1 for j in js if j >= 1]
        out[f"el2_{g}"] = float((dl[:, bj] ** 2).sum()) if bj else 0.0
        out[f"az2_{g}"] = float((da[:, bj] ** 2).sum()) if bj else 0.0
    return out


def axis_groups():
    from mdfm.body.smplh import BODY_JOINTS, N_BODY
    nm = list(BODY_JOINTS[:N_BODY])
    pick = lambda *w: tuple(j for j, s in enumerate(nm) if any(x in s for x in w))  # noqa: E731
    arms, legs = pick("collar", "shoulder", "elbow", "wrist"), pick("hip", "knee", "ankle", "foot")
    return {"arms": arms, "legs": legs, "spine_head": pick("spine", "neck", "head"), "root": (0,),
            "limbs": tuple(sorted(arms + legs))}


def stage_dump(a, log):
    """The model pass of the module docstring (dump), on the windows of the first --limit-takes takes of each dev
    archive (every window when not given)."""
    import pandas as pd
    import torch
    sys.path.insert(0, os.path.join(TREE, "scripts"))
    import probe_g1_decompose as T
    from mdfm import paths
    from mdfm.body.rotations import matrix_to_rot6d, rot6d_to_matrix  # noqa: F401
    from mdfm.body.smplh import PARENTS
    from mdfm.data import amass_index
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.gates import g1
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV
    t_start = time.time()
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    paths.assert_not_moveatlas(a.out)
    gs = g1.G1Set("dev", a.root)
    if set(gs.persons) & set(loaded.persons.get("amass", ())):
        raise Refused("dev persons among the checkpoint's training persons")
    if set(gs.persons) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person in the dev inputs")
    idx = T.subset(gs, a.limit_takes)
    if a.max_windows:
        idx = idx[:int(a.max_windows)]
    truth = T.TruthBodies(gs)
    groups = axis_groups()
    model = loaded.model
    blk = I.G1Predictor(loaded, block_windows=a.block_windows, batch_windows=a.batch_windows, configs=(T.CONFIG,))
    blk.subset = list(idx)
    rows, done = [], set()
    log(f"dump: {loaded.name}, {len(idx)} windows on {dev}")
    with torch.no_grad():
        for i in idx:
            if i in done:
                continue
            bidx = blk.block(gs, i)
            done |= set(bidx)
            takes = [str(gs.windows["take_id"].iat[j]) for j in bidx]
            wins = [I.g1_windows(gs, j, (T.CONFIG,))[T.CONFIG] for j in bidx]
            st = T.posterior_pass(model, wins, a.refine_steps, a.points, takes, I.seed_of(a.seed, gs.split, takes[0],
                                  T.CONFIG), a.batch_windows)
            off = 0
            for c in st["chunks"]:
                Wc = c["post"].mu_z.shape[0]
                jj = bidx[off:off + Wc]
                mb = st["mu_p"][off:off + Wc]
                off += Wc
                Lc = c["lengths"].tolist()
                _, pos, Rg, _, w = T._decode(model, c["post"], mb, c, None)
                tr = [truth(j) for j in jj]
                Rt = torch.as_tensor(T._pad_rows([x["R_local"] for x in tr], Lc), dtype=torch.float32, device=dev)
                bt = torch.as_tensor(np.stack([x["betas"] for x in tr]), dtype=torch.float32, device=dev)
                zt = torch.zeros(Rt.shape[0], 3, dtype=torch.float32, device=dev)
                _, Rg_t = I._pose(model, matrix_to_rot6d(Rt), zt, bt, c["gender"], w)
                pos, Rg, Rg_t = (x.double().cpu().numpy() for x in (pos, Rg, Rg_t))
                for (s0, n), j in zip(T._cuts(c), jj):
                    J, _ = gs.truth(j)
                    sl = slice(s0, s0 + n)
                    r = gs.windows.iloc[int(j)]
                    row = {"window": int(j), "take_id": r["take_id"], "archive": r["archive"],
                           "person_id": r["person_id"], "task": int(r["task"])}
                    row.update(axis_window(pos[sl], Rg[sl], J[:n], Rg_t[sl], groups, PARENTS))
                    rows.append(row)
            log(f"dump: {len(rows)} windows, {time.time() - t_start:.0f} s")
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "axis_windows.csv"), index=False, encoding="utf-8", lineterminator="\n")
    meta = {"script": "scripts/g1d_Hstand.py --stage dump", "checkpoint": loaded.describe(),
            "checkpoint_sha256": loaded.sha256, "weights": a.weights, "device": str(dev), "memory_cap_gb": cap,
            "refine_steps": a.refine_steps, "points": a.points, "batch_windows": a.batch_windows,
            "block_windows": a.block_windows, "seed": a.seed, "limit_takes": a.limit_takes,
            "max_windows": a.max_windows, "n_windows": int(len(t)),
            "g1_mm": MM * g1_stat(t) if len(t) else None,
            "truth_check": {"takes": truth.n_takes, "windows": truth.n_windows, "max_deviation_mm": MM * truth.max_dev_m},
            "groups": {k: list(v) for k, v in groups.items()}, "manifest_sha256": gs.manifest["manifest_sha256"],
            "code_sha256": sha256_file(os.path.abspath(__file__)), "moveatlas_head": PV.moveatlas_head(),
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "time_s": round(time.time() - t_start, 1),
            "torch": torch.__version__, "argv": sys.argv[1:]}
    with open(os.path.join(a.out, "axis_dump.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, default=str)
        f.write("\n")
    log(f"dump written: {len(t)} windows, G1 {meta['g1_mm']} mm, {meta['time_s']} s")
    return meta


def axis_summary(t, groups=AXIS_GROUPS):
    """Per group: twist share sum tw2 / sum (tw2 + sw2), the same after the pelvis twist (twr2), the RMS twist and
    swing (deg), the horizontal share of the squared joint error, the azimuth share of the squared bone error."""
    out = {}
    for g in groups:
        tw, sw, twr = t[f"tw2_{g}"].sum(), t[f"sw2_{g}"].sum(), t[f"twr2_{g}"].sum()
        k = t[f"seg_{g}"].sum()
        ey, exz = t[f"ey2_{g}"].sum(), t[f"exz2_{g}"].sum()
        el, az = t[f"el2_{g}"].sum(), t[f"az2_{g}"].sum()
        out[g] = {"twist_share": float(tw / (tw + sw)) if tw + sw > 0 else None,
                  "twist_share_rel_pelvis": float(twr / (twr + sw)) if twr + sw > 0 else None,
                  "twist_rms_deg": float(np.degrees(np.sqrt(tw / k))), "swing_rms_deg": float(np.degrees(np.sqrt(sw / k))),
                  "horizontal_share": float(exz / (exz + ey)) if exz + ey > 0 else None,
                  "azimuth_share": float(az / (az + el)) if az + el > 0 else None}
    return out


def stage_axis(a, log):
    import pandas as pd
    e = pd.read_csv(os.path.join(a.out, "energy.csv"), encoding="utf-8")
    d = pd.read_csv(os.path.join(a.dump, "axis_windows.csv"), encoding="utf-8")
    t = d.merge(e[["window", "take_id", "n", "q_energy", "q_arm", "q_leg", "energy_dps", "task_name"]], on="window",
                suffixes=("", "_e"), validate="one_to_one")
    if (t["take_id"] != t["take_id_e"]).any() or (t["n"] != t["n_e"]).any():
        raise ValueError("axis_windows.csv and energy.csv disagree on a window's take or length")
    res = {"dump": common_rel(a.dump), "axis_windows_sha256": sha256_file(os.path.join(a.dump, "axis_windows.csv")),
           "n_windows": int(len(t)), "g1_mm": MM * g1_stat(t), "all": axis_summary(t), "by_quintile": {}}
    for q, s in t.groupby("q_energy"):
        res["by_quintile"][int(q)] = dict(axis_summary(s), n_windows=int(len(s)), kw_mm=MM * kw_mean(s))
    for col in ("q_arm", "q_leg"):
        res[f"by_{col}"] = {int(q): axis_summary(s, (("arms",) if col == "q_arm" else ("legs",)))
                            for q, s in t.groupby(col)}
    q1, q5 = res["by_quintile"].get(1), res["by_quintile"].get(5)
    if q1 and q5:
        s1, s5 = q1["limbs"]["twist_share"], q5["limbs"]["twist_share"]
        res["C3"] = {"q1_limb_twist_share": s1, "q5_limb_twist_share": s5, "difference": s1 - s5,
                     "holds": bool(s1 >= 0.5 and s1 - s5 >= 0.15)}
    with open(os.path.join(a.dump, "axis.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    L = [f"# G1D-Hstand axis split ({res['dump']})", "", f"{res['n_windows']:,} windows, G1 {_f(res['g1_mm'], 2)} mm.",
         "", "| set | group | twist share | twist share, pelvis twist removed | twist RMS deg | swing RMS deg | "
         "horizontal share of joint error | azimuth share of bone error |", "|---|---|---|---|---|---|---|---|"]
    sets = [("all", res["all"])] + [(f"Q{q}", v) for q, v in res["by_quintile"].items()]
    for nm, v in sets:
        for g in AXIS_GROUPS:
            x = v[g]
            L.append(f"| {nm} | {g} | {_f(x['twist_share'], 3)} | {_f(x['twist_share_rel_pelvis'], 3)} | "
                     f"{_f(x['twist_rms_deg'], 1)} | {_f(x['swing_rms_deg'], 1)} | {_f(x['horizontal_share'], 3)} | "
                     f"{_f(x['azimuth_share'], 3)} |")
    if "C3" in res:
        c = res["C3"]
        L += ["", f"C3: Q1 limb twist share {_f(c['q1_limb_twist_share'], 3)}, Q5 {_f(c['q5_limb_twist_share'], 3)}, "
              f"difference {_f(c['difference'], 3)} (needed at least 0.50 and 0.15): "
              f"{'holds' if c['holds'] else 'fails'}."]
    with open(os.path.join(a.dump, "axis.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))
    return res


def common_rel(p):
    try:
        return os.path.relpath(os.path.abspath(p), TREE).replace("\\", "/")
    except ValueError:
        return p


# ---------------------------------------------------------------- CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=("energy", "analyze", "dump", "axis"))
    p.add_argument("--out", default=OUT, help="output folder (energy, analyze; dump: the dump folder)")
    p.add_argument("--dump", default=None, help="axis: the folder of axis_windows.csv")
    p.add_argument("--root", default=None, help="G1 input root (default mdfm.gates.g1.default_root)")
    p.add_argument("--checkpoint", default="cache/runs/phaseA_root_full/ckpt_B_032044.pt")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-gb", type=float, default=None)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--batch-windows", type=int, default=32)
    p.add_argument("--block-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--limit-takes", type=int, default=None, help="dump: the first K takes of each dev archive")
    p.add_argument("--max-windows", type=int, default=None, help="dump: at most this many windows (a CPU test)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import mdfm  # noqa: F401
    log = lambda *m: print(*m, flush=True)  # noqa: E731
    if a.stage == "axis" and not a.dump:
        raise SystemExit("--stage axis needs --dump <folder of axis_windows.csv>")
    res = {"energy": stage_energy, "analyze": stage_analyze, "dump": stage_dump, "axis": stage_axis}[a.stage](a, log)
    from mdfm.train import provenance as PV
    folder = a.dump if a.stage == "axis" else a.out
    prov = {"script": "scripts/g1d_Hstand.py", "stage": a.stage, "argv": sys.argv[1:],
            "code_sha256": sha256_file(os.path.abspath(__file__)), "moveatlas_head": PV.moveatlas_head(),
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0]}
    with open(os.path.join(folder, f"_provenance_{a.stage}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    return res


if __name__ == "__main__":
    main()
