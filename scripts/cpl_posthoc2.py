"""CPL post hoc check PH6 (reports/bench/cpl/posthoc_plan_2.txt, amendment 2 of the CPL prediction; post hoc, hashed
before this script ran). Dev diagnostic, not a gate read; CPU only.

The wrists, the lower back, the trunk and the head, with the strides, filters, low-pass, unit axis and relation form of
scripts/cpl_coupling.py (its functions and those of scripts/cpl_posthoc.py imported): real train54 sites wrist_L /
wrist_R (locations 3 / 4, paired with the stride's side), lower_back (2), trunk (1), head (0); AMASS the renderer's
segment joints 18 / 19 (forearms), 0, 9, 15. Per stride: ROM on the unit's principal axis (rom_<site>) and the axis-free
3D excursion (exc_<site>). Per site and measure: the AMASS train relation log m = a + b log(S / l) + d (log c - log 110)
(persons weighted equally), kappa and gamma_site = exp(median over persons of the person median residual / b), joint
person bootstrap. Reading per site as posthoc_plan_2.txt states it.

Writes reports/bench/cpl/posthoc2/: results.json, summary.md, real_sites.csv, amass_train_sites.csv, _provenance.json.

Run (mdfm environment, CPU only):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/cpl_posthoc2.py [--limit-amass N] [--out DIR] [--boot N]
"""
import argparse
import importlib.util
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

OUT = os.path.join(TREE, "reports", "bench", "cpl", "posthoc2")
PLAN = os.path.join(TREE, "reports", "bench", "cpl", "posthoc_plan_2.txt")
PLAN_SHA_FILE = os.path.join(TREE, "reports", "bench", "cpl", "posthoc_plan_2.sha256")
PLAN_SHA256 = "2950fe8888aa3f30b8cabbd5e3353de95b54386ba44211302668eb5855200a12"
LABEL = "dev diagnostic, not a gate read; post hoc (amendment 2)"
SITES = ("wrist", "lower_back", "trunk", "head")
REAL_SITES = {"wrist": {"L": 3, "R": 4}, "lower_back": {"L": 2, "R": 2}, "trunk": {"L": 1, "R": 1},
              "head": {"L": 0, "R": 0}}
BODY_JOINTS = {"wrist": {"L": 18, "R": 19}, "lower_back": {"L": 0, "R": 0}, "trunk": {"L": 9, "R": 9},
               "head": {"L": 15, "R": 15}}
MEASURES = ("rom", "exc")
B_MIN = 0.15


def _load(name, key):
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


PH = _load("cpl_posthoc", "_cplph2_cpl_posthoc")
C = PH.C
SCRIPT_SHA256_AT_START = C.sha256_file(os.path.abspath(__file__))


def measures(gyro, windows, fps):
    """(rom on the unit axis, 3D excursion) per window."""
    rom, _, exc = PH.window_measures(gyro, windows, fps)
    return rom, exc


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
            for name in SITES:
                site = REAL_SITES[name][foot]
                if site not in loc or not cm[loc.index(site), 3:6].all():
                    seg[name] = (np.full(len(win), np.nan), np.full(len(win), np.nan))
                    continue
                j = loc.index(site)
                g = sig[:, j, 3:6].copy()
                g[~valid[:, j]] = np.nan
                seg[name] = measures(C.lowpass(g, fps), win, fps)
            h = demo.get(pid, {}).get("height_m", float("nan"))
            for k, r in enumerate(sf.itertuples(index=False)):
                row = {"person_id": pid, "unit": tid, "task": r.task, "foot": foot, "t0": float(r.t0),
                       "t1": float(r.t1), "S": float(r.length_m), "stature_m": h}
                for name in SITES:
                    row[f"rom_{name}"], row[f"exc_{name}"] = seg[name][0][k], seg[name][1][k]
                rows.append(row)
    d = C.finish(pd.DataFrame(rows))
    log(f"real: {len(d)} strides", flush=True)
    return d


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
    for k, tr in enumerate(takes.itertuples()):
        i, n = int(tr.i), int(tr.n_knots)
        bodies = store.windows([i], [0], win_s=n / C.RATE)
        J, Wb, contact = C.truth_rates(bodies, bank, n)
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
            seg = {name: measures(C.lowpass(Wb[:, BODY_JOINTS[name][foot]], C.RATE), win, C.RATE) for name in SITES}
            for j, r in enumerate(sf.itertuples(index=False)):
                row = {"person_id": str(tr.person_id), "unit": str(tr.take_id), "archive": str(tr.archive),
                       "treadmill": int(tr.treadmill), "foot": foot, "t0": float(r.t0), "t1": float(r.t1),
                       "S": float(r.length_m), "stature_m": stat[key]}
                for name in SITES:
                    row[f"rom_{name}"], row[f"exc_{name}"] = seg[name][0][j], seg[name][1][j]
                rows.append(row)
        if (k + 1) % 200 == 0:
            log(f"  AMASS {split}: {k + 1} of {len(takes)} takes", flush=True)
    d = C.finish(pd.DataFrame(rows))
    log(f"AMASS {split}: {len(d)} strides", flush=True)
    return d


def site_blocks(d, col):
    X = C.design(d, "stride")
    y = np.log(d[col].to_numpy())
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    return C.blocks(d["person_id"].to_numpy()[ok], X[ok], y[ok])


def site_values(real, col, beta):
    X = C.design(real, "stride")
    r = np.log(real[col].to_numpy()) - X @ beta
    pid = real["person_id"].to_numpy()
    return C.person_median(pid, r), C.person_median(pid, r / beta[1])


def sexp(x):
    """exp that returns inf (or 0) beyond the float range instead of raising: a slope b near zero in a bootstrap
    resample sends residual / b far out (a site whose rotation does not scale with the stride)."""
    return math.exp(x) if x < 700 else float("inf")


def site_row(real, am, col, n_boot, seed):
    bl = site_blocks(am, col)
    beta, s2 = C.solve(bl, sorted(bl))
    pk, pg = site_values(real, col, beta)
    out = {"beta": [float(x) for x in beta], "sigma2": s2, "kappa": sexp(C.med_of_persons(pk)),
           "gamma": sexp(C.med_of_persons(pg)), "persons": len(pk),
           "strides": int(np.isfinite(real[col].to_numpy()).sum())}
    rng = np.random.default_rng(seed)
    am_p = sorted(bl)
    rp_all = sorted(pk)
    ks, gs = [], []
    for _ in range(int(n_boot)):
        ap = [am_p[i] for i in rng.integers(0, len(am_p), len(am_p))]
        rp = [rp_all[i] for i in rng.integers(0, len(rp_all), len(rp_all))]
        try:
            b, _ = C.solve(bl, ap)
        except np.linalg.LinAlgError:
            continue
        k2, g2 = site_values(real, col, b)
        ks.append(sexp(float(np.median([k2[p] for p in rp if p in k2]))))
        gs.append(sexp(float(np.median([g2[p] for p in rp if p in g2]))))
    out["kappa_interval"] = [float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))]
    out["gamma_interval"] = [float(np.quantile(gs, 0.025)), float(np.quantile(gs, 0.975))]
    pa = C.person_median(am["person_id"].to_numpy(), am[col].to_numpy())
    pr = C.person_median(real["person_id"].to_numpy(), real[col].to_numpy())
    out["median_real_deg"] = C.med_of_persons(pr)
    out["median_amass_deg"] = C.med_of_persons(pa)
    return out


def reading(row):
    if abs(row["beta"][1]) < B_MIN:
        return "NOT SCALING"
    g, (lo, hi) = row["gamma"], row["gamma_interval"]
    if g <= C.EXPLAINS_POINT and hi < C.EXPLAINS_UPPER:
        return "CARRIES SHORT"
    if C.MATCHED_BAND[0] <= g <= C.MATCHED_BAND[1] and lo >= C.MATCHED_IV[0] and hi <= C.MATCHED_IV[1]:
        return "MATCHED"
    if g >= C.LONG_POINT and lo > C.LONG_LOWER:
        return "LONG"
    return "PARTLY"


def markdown(res):
    L = ["# CPL post hoc check PH6: wrists and trunk sites (amendment 2, post hoc)", "",
         "Dev diagnostic, not a gate read; post hoc: reports/bench/cpl/posthoc_plan_2.txt (SHA-256 "
         f"{res['plan_sha256']}), hashed before this run. Written by scripts/cpl_posthoc2.py; every number below is "
         f"read from results.json of this folder. MoveAtlas {res['moveatlas'].get('head', 'n/a')}.", "",
         "| site | measure | b (elasticity) | kappa | 95 % | gamma_site | 95 % | median deg real / AMASS | persons | "
         "reading |", "|---" * 10 + "|"]
    for s in SITES:
        for m in MEASURES:
            r = res["sites"][s][m]
            L.append(f"| {s} | {m} | {PH.f(r['beta'][1])} | {PH.f(r['kappa'])} | {PH.fiv(r['kappa_interval'])} | "
                     f"{PH.f(r['gamma'])} | {PH.fiv(r['gamma_interval'])} | {PH.f(r['median_real_deg'], 1)} / "
                     f"{PH.f(r['median_amass_deg'], 1)} | {r['persons']} | {r.get('reading', '')} |")
    L += ["", f"Counts: real {res['counts']['real']} strides, AMASS train {res['counts']['amass']} strides. Run "
          f"{res['run_started']}, {res['timing_s']:.0f} s."]
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
    plan = C.check_prediction(PLAN, PLAN_SHA_FILE, PLAN_SHA256)
    import mdfm  # noqa: F401
    stamp = C._sibling("moveatlas_stamp").stamp()
    os.makedirs(a.out, exist_ok=True)
    train54, locked, _ = C.load_sets()
    real = real_rows(train54, locked)
    am = amass_rows("train", a.limit_amass)
    real.to_csv(os.path.join(a.out, "real_sites.csv"), index=False, encoding="utf-8", lineterminator="\n")
    am.to_csv(os.path.join(a.out, "amass_train_sites.csv"), index=False, encoding="utf-8", lineterminator="\n")
    res = {"what": LABEL, "task": "CPL", "plan_sha256": plan, "run_started": started, "moveatlas": stamp,
           "limit_amass": a.limit_amass, "counts": {"real": int(len(real)), "amass": int(len(am))}, "sites": {}}
    for k, s in enumerate(SITES):
        res["sites"][s] = {}
        for j, m in enumerate(MEASURES):
            col = f"{m}_{s}"
            rr = real[np.isfinite(real[col].to_numpy()) & (real[col].to_numpy() > 0)].reset_index(drop=True)
            aa = am[np.isfinite(am[col].to_numpy()) & (am[col].to_numpy() > 0)].reset_index(drop=True)
            row = site_row(rr, aa, col, a.boot, C.SEED + 600 + 10 * k + j)
            if m == "rom":
                row["reading"] = reading(row)
            res["sites"][s][m] = row
    res["timing_s"] = time.time() - t0
    C.write_json(os.path.join(a.out, "results.json"), res)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(res) + "\n")
    C.write_json(os.path.join(a.out, "_provenance.json"), {
        "script": C.rel(os.path.abspath(__file__)), "script_sha256_at_start": SCRIPT_SHA256_AT_START,
        "script_sha256_at_write": C.sha256_file(os.path.abspath(__file__)),
        "cpl_coupling_sha256": C.sha256_file(os.path.join(HERE, "cpl_coupling.py")),
        "cpl_posthoc_sha256": C.sha256_file(os.path.join(HERE, "cpl_posthoc.py")), "plan_sha256": plan,
        "moveatlas_stamp": stamp, "run_started": started, "label": LABEL,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                       "PYTHONDONTWRITEBYTECODE")}, "argv": sys.argv[1:]})
    print({s: res["sites"][s]["rom"]["reading"] for s in SITES}, f"{res['timing_s']:.0f} s")


if __name__ == "__main__":
    main()
