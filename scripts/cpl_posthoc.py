"""CPL post hoc checks PH1 to PH5 (reports/bench/cpl/posthoc_plan.txt, amendment 1 of the CPL prediction; post hoc,
written after the registered reading of scripts/cpl_coupling.py was read, hashed before this script ran). Dev
diagnostic, not a gate read; CPU only.

The same populations, strides and filters as scripts/cpl_coupling.py (its loaders' rules, its functions imported):
train54 walkway strides with the stride side's thigh, shank, ankle and foot gyroscopes; the AMASS train straight
forward walks and the AMASS dev forward walks, scored strides, ideal segment gyroscopes. Per stride and segment, beside
the registered ROM (recomputed, and checked against reports/bench/cpl/*_strides.csv):
- rom_sax: the ROM on the stride's own principal axis (stride_rom with the axis of that stride's samples);
- exc3d: the axis-free 3D excursion, the largest geodesic angle between two orientations of the segment within the
  stride, from the quaternion integration of the low-passed gyro (midpoint rate, no detrending);
- AMASS only, creep_ankle and creep_foot (PH4): the mean horizontal velocity of the ankle joint (7/8) and of the SMPL
  foot joint (10/11) of the stride's foot over the middle third of its first contact run in the stride, projected on
  the stride's ankle chord, over the stride speed.
Readings PH1 to PH5 as the plan states them (results.json, key posthoc).

Writes reports/bench/cpl/posthoc/: results.json, summary.md, real_strides_posthoc.csv, amass_train_strides_posthoc.csv,
amass_dev_strides_posthoc.csv, _provenance.json.

Run (mdfm environment, CPU only):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/cpl_posthoc.py [--limit-amass N] [--out DIR] [--boot N]
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "bench", "cpl", "posthoc")
PLAN = os.path.join(TREE, "reports", "bench", "cpl", "posthoc_plan.txt")
PLAN_SHA_FILE = os.path.join(TREE, "reports", "bench", "cpl", "posthoc_plan.sha256")
PLAN_SHA256 = "73db354c1adf1192904bf114b5ecc3f039f9c91964e98b251b8a40eeddab5fcf"
REGISTERED = os.path.join(TREE, "reports", "bench", "cpl")
LABEL = "dev diagnostic, not a gate read; post hoc (amendment 1)"
ANKLE_J = {"L": 7, "R": 8}
FOOT_J = {"L": 10, "R": 11}
PH1_POINT, PH1_UPPER = 0.97, 1.0
PH2_THIGH = 0.85
PH4_SHIFT = -0.03
PH5_BAND = (0.965, 1.035)


def _cpl():
    key = "_cplph_cpl_coupling"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, "cpl_coupling.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


C = _cpl()
SCRIPT_SHA256_AT_START = C.sha256_file(os.path.abspath(__file__))


def check_plan(path=PLAN, sha_file=PLAN_SHA_FILE, expected=PLAN_SHA256):
    return C.check_prediction(path, sha_file, expected)


# ---------------------------------------------------------------- rotation measures
def quat_mul(a, b):
    """Hamilton product of quaternion arrays (..., 4) [w, x, y, z]."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def integrate(gyro_dps, fps):
    """gyro (K, 3) deg/s in the body frame -> orientations (K, 4) from the identity, body-frame increments with the
    midpoint rate."""
    w = np.deg2rad(np.asarray(gyro_dps, dtype=np.float64))
    mid = 0.5 * (w[1:] + w[:-1]) / fps
    ang = np.linalg.norm(mid, axis=1)
    half = 0.5 * ang
    s = np.where(ang > 1e-12, np.sin(half) / np.where(ang > 1e-12, ang, 1.0), 0.5)
    dq = np.concatenate([np.cos(half)[:, None], s[:, None] * mid], 1)
    q = np.zeros((len(w), 4))
    q[0] = (1.0, 0.0, 0.0, 0.0)
    for k in range(len(dq)):
        q[k + 1] = quat_mul(q[k], dq[k])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def excursion3d(gyro_dps, fps):
    """The largest geodesic angle (deg) between two orientations within the window, NaN with a non-finite sample."""
    g = np.asarray(gyro_dps, dtype=np.float64)
    if len(g) < 3 or not np.isfinite(g).all():
        return float("nan")
    q = integrate(g, fps)
    d = np.abs(q @ q.T).min()
    return float(np.rad2deg(2.0 * np.arccos(min(1.0, d))))


def window_measures(gyro, windows, fps):
    """gyro (N, 3) low-passed -> (rom_trial, rom_sax, exc3d) arrays over windows [(i0, i1)]."""
    rom_trial, _, _, _ = C.segment_table(gyro, windows, fps)
    sax, exc = [], []
    for a, b in windows:
        if not (0 <= a < b < len(gyro)):
            sax.append(np.nan)
            exc.append(np.nan)
            continue
        w = gyro[a:b + 1]
        ax, _ = C.sagittal_axis(w)
        sax.append(C.stride_rom(w @ ax, 0, len(w) - 1, fps)[0] if ax is not None else np.nan)
        exc.append(excursion3d(w, fps))
    return np.asarray(rom_trial), np.asarray(sax), np.asarray(exc)


def creep(J, V, contact, foot, i0, i1, v_stride):
    """(creep_ankle, creep_foot) of one stride (module docstring), NaN when no contact run of three samples starts in
    the stride."""
    fi = 0 if foot == "L" else 1
    c = np.asarray(contact[:, fi], bool)
    a = i0 - 2 if i0 >= 2 else 0
    while a <= i1 and not c[a]:
        a += 1
    b = a
    while b <= i1 and b < len(c) and c[b]:
        b += 1
    if b - a < 3 or a > i1:
        return float("nan"), float("nan")
    lo, hi = a + (b - a) // 3, a + 2 * (b - a) // 3
    if hi <= lo:
        hi = lo + 1
    chord = J[i1, ANKLE_J[foot], [0, 2]] - J[i0, ANKLE_J[foot], [0, 2]]
    n = np.linalg.norm(chord)
    if n <= 0 or not np.isfinite(v_stride) or v_stride <= 0:
        return float("nan"), float("nan")
    u = chord / n
    out = []
    for j in (ANKLE_J[foot], FOOT_J[foot]):
        vel = V[lo:hi, j][:, [0, 2]].mean(0)
        out.append(float(vel @ u / v_stride))
    return out[0], out[1]


# ---------------------------------------------------------------- loaders (the registered rules)
def real_rows(persons, locked, log=print):
    import pandas as pd
    from mdfm.gates import g3
    C.guard(persons, locked, persons, "real persons")
    demo = C._sibling("f2_H1shape").read_demographics(list(persons), locked)
    tg = g3.load_targets(tuple(persons))
    if tg.counts.get("lockbox_rows"):
        raise AssertionError("lockbox rows in the targets")
    rows = []
    for tid, rec in g3.imu_records(tuple(persons), tg.trials):
        pid = str(rec.get("person_id"))
        C.guard([pid], locked, persons, tid)
        fps = float(rec["fps"])
        sig = np.asarray(rec["signal"], dtype=np.float64)
        loc = [int(x) for x in np.asarray(rec["location_ids"])]
        cm = np.asarray(rec["channel_mask"], dtype=bool)
        valid = np.asarray(rec["sample_valid"], dtype=bool) if rec.get("sample_valid") is not None \
            else np.ones(sig.shape[:2], bool)
        st = tg.strides[tg.strides["trial_id"] == tid]
        for foot in ("L", "R"):
            sf = st[st["foot"] == foot].sort_values("t0")
            if not len(sf):
                continue
            T = (sf["t1"] - sf["t0"]).to_numpy(np.float64)
            S = sf["length_m"].to_numpy(np.float64)
            sf = sf[(T >= C.T_RANGE[0]) & (T <= C.T_RANGE[1]) & (S >= C.S_MIN) & np.isfinite(T) & np.isfinite(S)]
            win = [(int(round(a * fps)), int(round(b * fps))) for a, b in zip(sf["t0"], sf["t1"])]
            seg = {}
            for name in C.SEGMENTS:
                site = C.REAL_SITES[name][foot]
                if site not in loc or not cm[loc.index(site), 3:6].all():
                    seg[name] = tuple(np.full(len(win), np.nan) for _ in range(3))
                    continue
                j = loc.index(site)
                g = sig[:, j, 3:6].copy()
                g[~valid[:, j]] = np.nan
                seg[name] = window_measures(C.lowpass(g, fps), win, fps)
            h = demo.get(pid, {}).get("height_m", float("nan"))
            for k, r in enumerate(sf.itertuples(index=False)):
                row = {"person_id": pid, "unit": tid, "task": r.task, "foot": foot, "t0": float(r.t0),
                       "t1": float(r.t1), "S": float(r.length_m), "stature_m": h}
                for name in C.SEGMENTS:
                    row[f"rom_{name}"], row[f"sax_{name}"], row[f"exc_{name}"] = (seg[name][0][k], seg[name][1][k],
                                                                                  seg[name][2][k])
                rows.append(row)
    d = C.finish(pd.DataFrame(rows))
    log(f"real: {len(d)} strides")
    return d


def truth_full(bodies, bank, n):
    """C.truth_rates with the exact joint velocities kept: joints, velocities (n, 22, 3), segment-frame rates deg/s,
    contacts."""
    import torch
    from mdfm.body import frames
    from mdfm.body.kinematics import KnotBatch
    f64 = torch.float64
    with torch.no_grad():
        rest = bank.rest(bodies.betas[:1].to(f64), bodies.gender[:1].to(torch.int64))
        kb = KnotBatch(bodies.rot6d[:1].to(f64), bodies.transl[:1].to(f64), rest, rate=C.RATE,
                       lengths=bodies.lengths[:1])
        t = torch.arange(n, dtype=f64) / C.RATE
        w = torch.zeros(n, dtype=torch.int64)
        P, V, Wb = [], [], []
        for a in range(0, n, 65536):
            kin = kb.kinematics(w[a:a + 65536], t[a:a + 65536])
            P.append(kin.p)
            V.append(kin.v)
            Wb.append(torch.einsum("njba,njb->nja", kin.R, kin.omega))
        J, V, Wb = torch.cat(P), torch.cat(V), torch.cat(Wb)
        contact = frames.contact_labels(J[None], C.RATE, vel=V[None], floor=0.0)[0]
    return J.numpy(), V.numpy(), np.rad2deg(Wb.numpy()), contact.numpy()


def amass_rows(split, limit=None, log=print):
    import pandas as pd
    import torch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import body_store
    from mdfm.readouts import gait
    ins = C._sibling("eval_gait_insim")
    h1 = C._sibling("f2_H1shape")
    store = body_store.BodyStore(split=split)
    takes = C.amass_takes(store, split)
    if limit:
        takes = takes.iloc[:int(limit)]
    bank = SkeletonBank(torch.float64)
    rows, stat = [], {}
    t_start = time.time()
    for k, tr in enumerate(takes.itertuples()):
        i, n = int(tr.i), int(tr.n_knots)
        bodies = store.windows([i], [0], win_s=n / C.RATE)
        J, V, Wb, contact = truth_full(bodies, bank, n)
        ics = gait.initial_contacts(contact.astype(np.float64), C.RATE)
        st = gait.strides(J, ics, rate=C.RATE)
        if not len(st):
            continue
        pel = J[:, gait.PELVIS][:, [0, 2]]
        ps = ins.passes_from_ics(ics, pel)
        ins.steady_straight(st, ps, pel)
        g = int(store.gender[i])
        key = (tuple(np.round(store.betas[i], 6)), g)
        if key not in stat:
            s_, _ = h1.stature_of(store.betas[i], g)
            stat[key] = float(np.asarray(s_))
        for foot in ("L", "R"):
            sf = st[(st["foot"] == foot) & st["scored"].astype(bool)].sort_values("t0")
            T = (sf["t1"] - sf["t0"]).to_numpy(np.float64)
            S = sf["length_m"].to_numpy(np.float64)
            sf = sf[(T >= C.T_RANGE[0]) & (T <= C.T_RANGE[1]) & (S >= C.S_MIN)]
            if not len(sf):
                continue
            win = [(int(round(a * C.RATE)), int(round(b * C.RATE))) for a, b in zip(sf["t0"], sf["t1"])]
            seg = {name: window_measures(C.lowpass(Wb[:, C.BODY_JOINTS[name][foot]], C.RATE), win, C.RATE)
                   for name in C.SEGMENTS}
            for j, r in enumerate(sf.itertuples(index=False)):
                a, b = win[j]
                v = float(r.length_m) / (float(r.t1) - float(r.t0))
                ca, cf = creep(J, V, contact, foot, a, b, v)
                row = {"person_id": str(tr.person_id), "unit": str(tr.take_id), "archive": str(tr.archive),
                       "treadmill": int(tr.treadmill), "foot": foot, "t0": float(r.t0), "t1": float(r.t1),
                       "S": float(r.length_m), "stature_m": stat[key], "creep_ankle": ca, "creep_foot": cf}
                for name in C.SEGMENTS:
                    row[f"rom_{name}"], row[f"sax_{name}"], row[f"exc_{name}"] = (seg[name][0][j], seg[name][1][j],
                                                                                  seg[name][2][j])
                rows.append(row)
        if (k + 1) % 200 == 0:
            log(f"  AMASS {split}: {k + 1} of {len(takes)} takes, {time.time() - t_start:.0f} s", flush=True)
    d = C.finish(pd.DataFrame(rows))
    log(f"AMASS {split}: {len(d)} strides", flush=True)
    return d


# ---------------------------------------------------------------- readings
def boot_median_of_persons(pm, n_boot, seed):
    keys = sorted(pm)
    v = np.array([pm[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [float(np.median(v[rng.integers(0, len(v), len(v))])) for _ in range(int(n_boot))]
    return float(np.median(v)), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], len(keys)


def ph1(real, n_boot):
    r = (real["rom_shank"] / real["rom_ankle"]).to_numpy()
    pm = C.person_median(real["person_id"].to_numpy(), r)
    point, iv, n = boot_median_of_persons(pm, n_boot, C.SEED + 101)
    rd = "MID-SHANK READS LESS" if (point < PH1_POINT and iv[1] < PH1_UPPER) else "NOT SHOWN"
    out = {"ratio_shank_over_ankle": point, "interval": iv, "persons": n, "reading": rd}
    for col in ("sax", "exc"):
        pmc = C.person_median(real["person_id"].to_numpy(), (real[f"{col}_shank"] / real[f"{col}_ankle"]).to_numpy())
        out[f"ratio_{col}"] = boot_median_of_persons(pmc, n_boot, C.SEED + 102)[:2]
    return out


def swap(d, measure):
    """A copy whose rom_<segment> columns hold another measure (sax or exc)."""
    e = d.copy()
    for s in C.SEGMENTS:
        e[f"rom_{s}"] = d[f"{measure}_{s}"]
    return e


def ph2(real, am, n_boot):
    out = {}
    for measure in ("rom", "sax", "exc"):
        r, a = C.complete(swap(real, measure) if measure != "rom" else real), \
            C.complete(swap(am, measure) if measure != "rom" else am)
        row = C.interval_row(C.Relations(a), r, n_boot, seed=C.SEED + 200)
        row["coefficients"] = C.coefficients(C.Relations(a).fit())
        row["medians"] = {s: {"real": C.med_of_persons(C.person_median(r["person_id"].to_numpy(),
                                                                        r[f"rom_{s}"].to_numpy())),
                              "amass": C.med_of_persons(C.person_median(a["person_id"].to_numpy(),
                                                                         a[f"rom_{s}"].to_numpy()))}
                          for s in C.SEGMENTS}
        out[measure] = row
    k = out["exc"]["point"]["kappa_thigh"]
    out["reading"] = "THIGH ROTATES LESS IN 3D" if k <= PH2_THIGH else "NOT SHOWN"
    return out


def ph3(real, am, gains, n_boot):
    rc, ac = C.complete(real), C.complete(am)
    fit = C.Relations(ac).fit()
    r = C.residuals(rc, fit)
    sources = {s: r[s] / fit[s][0][1] for s in C.SEGMENTS}
    sources["shank_foot"] = C.deltas(rc, fit, ("shank", "foot"))
    sources["thigh_shank_foot"] = C.deltas(rc, fit)
    out = {}
    for name, x in sources.items():
        gp = C.person_median(rc["person_id"].to_numpy(), x)
        ps = sorted(set(gp) & set(gains))
        a = np.array([gp[p] for p in ps])
        b = np.log(np.array([gains[p] for p in ps]))
        rr = C.pearson(a, b)
        rng = np.random.default_rng(C.SEED + 300)
        bs = []
        for _ in range(int(n_boot)):
            k = rng.integers(0, len(ps), len(ps))
            v = C.pearson(a[k], b[k])
            if np.isfinite(v):
                bs.append(v)
        iv = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out[name] = {"r": rr, "interval": iv, "n": len(ps), "linked": bool(np.isfinite(rr) and iv[0] > 0)}
    return out


def ph4(am, dev, n_boot):
    groups = {"treadmill": am[am["treadmill"] != 0], "overground_KIT": am[(am["treadmill"] == 0)],
              "dev_forward": dev}
    out = {}
    pms = {}
    for g, d in groups.items():
        out[g] = {}
        for col in ("creep_ankle", "creep_foot"):
            pm = C.person_median(d["person_id"].to_numpy(), d[col].to_numpy())
            pms[(g, col)] = pm
            point, iv, n = boot_median_of_persons(pm, n_boot, C.SEED + 400)
            out[g][col] = {"median": point, "interval": iv, "persons": n,
                           "finite_share": float(np.isfinite(d[col].to_numpy()).mean()) if len(d) else None}
    rng = np.random.default_rng(C.SEED + 401)
    for col in ("creep_ankle", "creep_foot"):
        a = np.array(list(pms[("treadmill", col)].values()))
        b = np.array(list(pms[("overground_KIT", col)].values()))
        diff = float(np.median(a) - np.median(b))
        bs = [float(np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]))
              for _ in range(int(n_boot))]
        iv = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out[f"treadmill_minus_overground_{col}"] = {"diff": diff, "interval": iv,
                                                    "reading": "BELT LOW" if diff <= PH4_SHIFT else "NOT SHOWN"}
    return out


def ph5(real, am, n_boot):
    rc, ac = C.complete(real, ("ankle", "foot")), C.complete(am)
    rel = C.Relations(ac)
    fit = rel.fit()
    pid = rc["person_id"].to_numpy()
    g = math.exp(C.med_of_persons(C.person_median(pid, C.deltas(rc, fit, ("ankle", "foot")))))
    iv = C.Boot(rel, rc, segments=("ankle", "foot")).run(n_boot, C.SEED + 500).get("gamma_L")
    rd = "DISTAL SITES MATCH" if PH5_BAND[0] <= g <= PH5_BAND[1] else "NOT SHOWN"
    return {"gamma_L_ankle_foot": g, "interval": iv, "persons": int(rc["person_id"].nunique()),
            "strides": int(len(rc)), "reading": rd}


def reproduction(d, path, keys):
    """Largest absolute difference of rom_<segment> against the registered stride table on the joined rows."""
    import pandas as pd
    reg = pd.read_csv(path, encoding="utf-8")
    m = d.merge(reg, on=keys, suffixes=("", "_reg"))
    out = {"rows_joined": int(len(m)), "rows_here": int(len(d)), "rows_registered": int(len(reg))}
    for s in C.SEGMENTS:
        x = (m[f"rom_{s}"] - m[f"rom_{s}_reg"]).abs()
        out[f"max_abs_diff_{s}"] = float(x.max()) if len(x) else None
    return out


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def fiv(iv, nd=3):
    return "" if not iv else f"[{iv[0]:.{nd}f}, {iv[1]:.{nd}f}]"


def markdown(res):
    L = ["# CPL post hoc checks PH1 to PH5 (amendment 1, post hoc)", "",
         "Dev diagnostic, not a gate read; post hoc: reports/bench/cpl/posthoc_plan.txt (SHA-256 "
         f"{res['plan_sha256']}) was written after the registered reading and hashed before this run. Written by "
         "scripts/cpl_posthoc.py; every number below is read from results.json of this folder. MoveAtlas "
         f"{res['moveatlas'].get('head', 'n/a')}.", "",
         f"Reproduction of the registered ROM: {res['reproduction']}.", ""]
    p1 = res["PH1"]
    L += [f"## PH1, same segment (real): **{p1['reading']}**", "",
          f"Median over persons of the person median ROM_shank / ROM_ankle-site: {f(p1['ratio_shank_over_ankle'])} "
          f"{fiv(p1['interval'])} ({p1['persons']} persons); on the stride's own axis {f(p1['ratio_sax'][0])} "
          f"{fiv(p1['ratio_sax'][1])}; 3D excursion {f(p1['ratio_exc'][0])} {fiv(p1['ratio_exc'][1])}.", ""]
    p2 = res["PH2"]
    L += [f"## PH2, rotation measure: **{p2['reading']}**", "",
          "| measure | gamma_L | 95 % | kappa thigh | 95 % | shank | 95 % | ankle | foot | 95 % |", "|---" * 10 + "|"]
    for m in ("rom", "sax", "exc"):
        pt, iv = p2[m]["point"], p2[m]["interval"]
        L.append(f"| {m} | {f(pt['gamma_L'])} | {fiv(iv.get('gamma_L'))} | {f(pt['kappa_thigh'])} | "
                 f"{fiv(iv.get('kappa_thigh'))} | {f(pt['kappa_shank'])} | {fiv(iv.get('kappa_shank'))} | "
                 f"{f(pt['kappa_ankle'])} | {f(pt['kappa_foot'])} | {fiv(iv.get('kappa_foot'))} |")
    L += ["", "Medians of person medians (deg), real / AMASS: " + "; ".join(
        f"{m} " + ", ".join(f"{s} {f(p2[m]['medians'][s]['real'], 1)} / {f(p2[m]['medians'][s]['amass'], 1)}"
                            for s in ("thigh", "shank", "ankle", "foot")) for m in ("rom", "sax", "exc")) + ".", ""]
    L += ["## PH3, person link per segment", "", "| source of gamma_p | r with log g_p | 95 % | persons | linked |",
          "|---|---|---|---|---|"]
    for k, v in res["PH3"].items():
        L.append(f"| {k} | {f(v['r'])} | {fiv(v['interval'])} | {v['n']} | {v['linked']} |")
    p4 = res["PH4"]
    L += ["", "## PH4, belt: mid-stance creep over stride speed (AMASS)", "",
          "| group | ankle | 95 % | foot joint | 95 % | persons |", "|---|---|---|---|---|---|"]
    for g in ("treadmill", "overground_KIT", "dev_forward"):
        a, b = p4[g]["creep_ankle"], p4[g]["creep_foot"]
        L.append(f"| {g} | {f(a['median'])} | {fiv(a['interval'])} | {f(b['median'])} | {fiv(b['interval'])} | "
                 f"{a['persons']} |")
    for col in ("creep_ankle", "creep_foot"):
        v = p4[f"treadmill_minus_overground_{col}"]
        L.append(f"\nTreadmill minus overground, {col}: {f(v['diff'])} {fiv(v['interval'])}: **{v['reading']}**.")
    p5 = res["PH5"]
    L += ["", f"## PH5, distal sites: **{p5['reading']}**", "",
          f"gamma_L over the ankle site and the foot: {f(p5['gamma_L_ankle_foot'])} {fiv(p5['interval'])} "
          f"({p5['persons']} persons, {p5['strides']} strides).", "", f"Run {res['run_started']}, "
          f"{res['timing_s']:.0f} s."]
    text = "\n".join(L)
    assert not any(ch in text for ch in "\u2013\u2014")
    return text


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-amass", type=int, default=None)
    ap.add_argument("--boot", type=int, default=C.BOOT_N)
    a = ap.parse_args(argv)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    plan = check_plan()
    import mdfm  # noqa: F401
    stamp = C._sibling("moveatlas_stamp").stamp()
    os.makedirs(a.out, exist_ok=True)
    train54, locked, _ = C.load_sets()
    real = real_rows(train54, locked)
    am = amass_rows("train", a.limit_amass)
    dev = amass_rows("dev", a.limit_amass)
    gains, gsrc = C.decoded_gains(train54, locked)
    res = {"what": LABEL, "task": "CPL", "plan_sha256": plan, "run_started": started, "moveatlas": stamp,
           "limit_amass": a.limit_amass, "f1_events": gsrc}
    res["reproduction"] = {
        "real": reproduction(real, os.path.join(REGISTERED, "real_strides.csv"), ["unit", "foot", "t0"]),
        "amass_train": reproduction(am, os.path.join(REGISTERED, "amass_train_strides.csv"), ["unit", "foot", "t0"]),
        "amass_dev": reproduction(dev, os.path.join(REGISTERED, "amass_dev_strides.csv"), ["unit", "foot", "t0"])}
    res["PH1"] = ph1(C.complete(real, ("shank", "ankle")), a.boot)
    res["PH2"] = ph2(real, am, a.boot)
    res["PH3"] = ph3(real, am, gains, a.boot)
    res["PH4"] = ph4(am, dev, a.boot)
    res["PH5"] = ph5(real, am, a.boot)
    res["timing_s"] = time.time() - t0
    C.write_json(os.path.join(a.out, "results.json"), res)
    for name, d in (("real_strides_posthoc.csv", real), ("amass_train_strides_posthoc.csv", am),
                    ("amass_dev_strides_posthoc.csv", dev)):
        d.to_csv(os.path.join(a.out, name), index=False, encoding="utf-8", lineterminator="\n")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(res) + "\n")
    C.write_json(os.path.join(a.out, "_provenance.json"), {
        "script": C.rel(os.path.abspath(__file__)), "script_sha256_at_start": SCRIPT_SHA256_AT_START,
        "script_sha256_at_write": C.sha256_file(os.path.abspath(__file__)),
        "cpl_coupling_sha256": C.sha256_file(os.path.join(HERE, "cpl_coupling.py")), "plan_sha256": plan,
        "moveatlas_stamp": stamp, "run_started": started, "label": LABEL,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                       "PYTHONDONTWRITEBYTECODE")}, "argv": sys.argv[1:]})
    print(json.dumps({k: res[k].get("reading") for k in ("PH1", "PH2", "PH5")}), f"{res['timing_s']:.0f} s")


if __name__ == "__main__":
    main()
