"""F2-H6warp: does the model infer stride length from cadence? The in-sim evaluation on time-warped AMASS dev walks.

Label of every output: dev diagnostic in the synthetic domain, not a G3 read. Report:
reports/weekend_2026-10-01/tasks/F2-H6warp.md; outputs: reports/bench/f2_scale/H6warp/.

Hypothesis H6: the model reads cadence correctly and infers stride length largely from cadence through the
stride-cadence relation of its synthetic training walks, so that real walkers who take longer strides at a given
cadence are decoded short. Test: the same walks played slower or faster (the same stride at another cadence), the
decoded stride over the true stride at each factor.

Pre-registration: reports/bench/f2_scale/H6warp/prediction.txt, its SHA-256 in prediction.sha256 and here in
PREDICTION_SHA256; check_prediction asserts the three agree before any stride, cadence or speed value is read or
computed (every mode).

Modes:
- --dump (GPU job; CPU with --device cpu for a smoke run): for each factor f of --factors (default 1.0, 0.85, 0.7,
  1.15; f < 1 plays the motion slower), each walking take of the AMASS dev body store selected by
  scripts/eval_gait_insim.select_takes is resampled whole (warp_arrays: the knots k = 0 .. floor((n - 1) / f) at
  source time k f / 100 s, rot6d and translation by the cubic Hermite interpolant of mdfm.body.hermite, rotations
  re-orthonormalized; factor 1 is the take unchanged) and run through scripts/eval_gait_insim.run_take unchanged
  (xsens13 render through factory.synth.SynthSampler, so the IMU is recomputed from the warped body; band copy; 10 s
  windows every 5 s; infer.infer with the in-sim evaluation's settings; the truth from the warped knots, contacts
  from its exact velocities; g3 lines). Writes <out>/dump_<checkpoint name>[_<tag>]/: strides.csv (the scored true
  strides of every take with the matched prediction of the predictors model and model_at_true_ics), ref_strides.csv
  and model_strides.csv (every true and decoded stride, for the cadence check), takes.csv, dump.json (the g3 lines
  of each factor and row as scripts/eval_gait_insim.py computes them, settings, checkpoint, failures). Then --read
  on that folder.
- --read DUMP: the registered quantities of the Prediction (G, r, beta, f*, P, share, the bootstrap, the guards,
  the label; secondary lines), written to <out>/results.json, summary.md, per_factor.csv, _provenance.json (a dump
  made with --limit or non-default inference settings is a smoke dump: its outputs carry smoke in their names and
  its label is SMOKE). Secondary, prediction_amendment_1.txt (hash asserted): --extra-factors (default 0.55) are
  decoded in the same dump and read beside the registered factors, without a criterion (results.json,
  secondary_extra).
- --early: the preliminary secondary read of the existing scripts/probe_root_motion.py warp rows of v1_B
  (root_motion_phaseA_root_full_ckpt_B_032044_xsens13_warps.json and _warp15_warps.json), written to
  <out>/early.json; no criterion of its own.

Never read: a train, calib or lockbox take (BodyStore(split='dev'), select_takes, the checkpoint's training persons
and amass_index.LOCKBOX_PERSONS refused), a real, EVAL, CODY or care_pd record.

Run (mdfm environment, PYTHONDONTWRITEBYTECODE=1):
  python scripts/f2_H6warp.py --dump --checkpoint cache/runs/phaseA_root_full/ckpt_B_032044.pt --memory-gb 12 \
      --out reports/bench/f2_scale/H6warp
  python scripts/f2_H6warp.py --early
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
for p in (TREE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import eval_gait_insim as EG  # noqa: E402

TASK = "F2-H6warp"
LABEL = "dev diagnostic in the synthetic domain, not a G3 read"
RATE = 100.0
OUT = os.path.join(TREE, "reports", "bench", "f2_scale", "H6warp")
PREDICTION_TXT = os.path.join(OUT, "prediction.txt")
PREDICTION_SHA = os.path.join(OUT, "prediction.sha256")
PREDICTION_SHA256 = "66ea95ba4f6849ebe84f02414be3356f58917c4472cf77b47c90fae2937fd249"
AMENDMENT_TXT = os.path.join(OUT, "prediction_amendment_1.txt")
AMENDMENT_SHA = os.path.join(OUT, "prediction_amendment_1.sha256")
AMENDMENT_SHA256 = "7df4fadf51c6c86053ba7984da781b5a87e7e3287dc504d41f9174a5ee951c11"
CHECKPOINT = "cache/runs/phaseA_root_full/ckpt_B_032044.pt"
CHECKPOINT_SHA256 = "cffccdac4109dfbfb921ed4ebcb690ce27dd2d35045a3afdaaca7d38f321a677"
FACTORS = (1.0, 0.85, 0.7, 1.15)
EXTRA_FACTORS = (0.55,)         # secondary (prediction_amendment_1.txt): f* of the beta read lies below 0.7
SEED = 20260929                 # the in-sim evaluation's seed
F1_RESULTS = os.path.join(TREE, "reports", "bench", "g3_gain_check", "results.json")
H3_DUMPS = os.path.join(TREE, "reports", "bench", "f2_scale", "H3angles", "dumps_2026-10-05", "results.json")
TAKES_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
RM_FILES = (os.path.join(TREE, "reports", "bench", "root_motion_phaseA_root_full_ckpt_B_032044_xsens13_warps.json"),
            os.path.join(TREE, "reports", "bench", "root_motion_phaseA_root_full_ckpt_B_032044_warp15_warps.json"))
BETA_V = (0.4, 2.0)             # m/s, AMASS takes of the beta fit
BOOT_N = 2000
BOOT_SEED = 20261006
# criterion of the Prediction
SHARE_SUP, P_UP_SUP = 0.5, 0.97
SHARE_NOT, P_LO_NOT = 0.2, 0.937
G1_TOL, T_TOL, MATCH_FRAC = 0.01, 0.05, 0.8
PREDICTORS = ("model", "model_at_true_ics")


# ---------------------------------------------------------------- guards
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_prediction(txt=PREDICTION_TXT, sha=PREDICTION_SHA, expected=PREDICTION_SHA256):
    """AssertionError unless the frozen prediction, its hash file and the constant agree; returns the SHA-256."""
    got = sha256_file(txt)
    with open(sha, encoding="utf-8") as f:
        listed = f.read().split()[0]
    if not (got == listed == expected):
        raise AssertionError(f"prediction hash {got}, file {listed}, constant {expected}")
    return got


def check_amendment(txt=AMENDMENT_TXT, sha=AMENDMENT_SHA, expected=None):
    """AssertionError unless amendment 1, its hash file and the constant agree; returns the SHA-256."""
    return check_prediction(txt, sha, AMENDMENT_SHA256 if expected is None else expected)


def parse_extra(text, factors):
    """The secondary factors of --extra-factors ('' for none): positive, distinct, none among the factors."""
    if not str(text).strip():
        return ()
    parts = [s.strip() for s in str(text).split(",") if s.strip()]
    out = []
    for s in parts:
        f = float(s)
        if not math.isfinite(f) or f <= 0 or f in factors or f in out:
            raise ValueError(f"extra factor {s!r}: not positive, or repeated, or among the factors")
        out.append(f)
    return tuple(out)


def parse_factors(text):
    """The factors of a --factors value 'F[,F]' as floats in the given order; 1.0 must be among them; a factor
    that is not finite and positive, a repeat or an empty list raises ValueError."""
    parts = [s.strip() for s in str(text).split(",") if s.strip()]
    if not parts:
        raise ValueError("--factors lists no factor")
    out = []
    for s in parts:
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"factor {s!r} is not a number") from None
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"factor {s!r} is not a positive number")
        out.append(f)
    if len(set(out)) != len(out):
        raise ValueError(f"a factor is listed twice in {parts}")
    if 1.0 not in out:
        raise ValueError("the factors must include 1.0 (the reference)")
    return tuple(out)


# ---------------------------------------------------------------- the warp
def warped_len(n, factor):
    """Knots of a take of n knots played `factor` times as fast: floor((n - 1) / factor) + 1."""
    return int(math.floor((int(n) - 1) / float(factor) + 1e-9)) + 1


def warp_arrays(rot6d, transl, contact, factor, rate=RATE):
    """(rot6d (m, 22, 6), transl (m, 3), contact (m, 2)) of one take's knots rot6d (T, 22, 6), transl (T, 3),
    contact (T, 2) played `factor` times as fast: knot k at source time k factor / rate, rot6d and transl by the
    cubic Hermite (Catmull-Rom) interpolant of mdfm.body.hermite, rotations re-orthonormalized (Gram-Schmidt of
    rot6d_to_matrix, back to rot6d), contacts from the nearest source knot. Factor 1 returns the inputs unchanged.
    Float64 out for a warped take."""
    from mdfm.body.hermite import hermite
    from mdfm.body.rotations import matrix_to_rot6d, rot6d_to_matrix
    if float(factor) == 1.0:
        return rot6d, transl, contact
    T = int(rot6d.shape[0])
    m = warped_len(T, factor)
    q = (torch.arange(m, dtype=torch.float64) * float(factor)).clamp_max(T - 1)
    t = q / rate
    with torch.no_grad():
        r6 = hermite(torch.as_tensor(rot6d).double(), t, rate)
        r6 = matrix_to_rot6d(rot6d_to_matrix(r6))
        tr = hermite(torch.as_tensor(transl).double(), t, rate)
    ct = torch.as_tensor(contact)[torch.round(q).long().clamp(0, T - 1)]
    return r6, tr, ct


class WarpedStore:
    """A dev BodyStore seen through a time warp, for scripts/eval_gait_insim.render_take: takes is the store's
    table with n_knots set to the warped lengths; windows(takes, [0], win_s) returns each whole take warped
    (warp_arrays) in the window frame of the source take (a rigid frame: floor and gravity unchanged). Only whole
    takes from knot 0 are served."""

    def __init__(self, store, factor):
        self.base, self.factor = store, float(factor)
        self.rate = float(store.rate)
        self.split = getattr(store, "split", None)
        self.takes = store.takes.copy()
        self.takes["n_knots"] = [warped_len(n, self.factor) for n in store.takes["n_knots"].to_numpy(np.int64)]

    def windows(self, takes, starts, win_s):
        from mdfm.data.body_store import BodyWindows
        takes, starts = np.asarray(takes, np.int64), np.asarray(starts, np.int64)
        if len(takes) != 1 or int(starts[0]) != 0:
            raise ValueError("WarpedStore serves one whole take from knot 0")
        i = int(takes[0])
        n_src = int(self.base.takes["n_knots"].iloc[i])
        m = warped_len(n_src, self.factor)
        if int(round(float(win_s) * self.rate)) != m:
            raise ValueError(f"take {i}: {win_s} s asked, the warped take holds {m} knots")
        bw = self.base.windows([i], [0], win_s=n_src / self.rate)
        if self.factor == 1.0:
            return bw
        r6, tr, ct = warp_arrays(bw.rot6d[0, :n_src], bw.transl[0, :n_src], bw.contact[0, :n_src], self.factor,
                                 self.rate)
        return BodyWindows(r6[None].float(), tr[None].float(), ct[None].bool(), torch.ones(1, m, dtype=torch.bool),
                           torch.tensor([m], dtype=torch.int64), bw.betas, bw.gender, bw.person, bw.task, bw.take,
                           bw.start)


# ---------------------------------------------------------------- quantities of the Prediction
def person_table(df):
    """Per person of a strides table (columns person_id, length_m, matched, err_cm): t (mean true stride m over
    every row), b (mean signed error cm over the matched rows, NaN without one)."""
    g = df.groupby("person_id", sort=True)
    t = g["length_m"].mean()
    k = df[df["matched"].astype(bool)]
    b = k.groupby("person_id", sort=True)["err_cm"].mean().reindex(t.index)
    return pd.DataFrame({"t": t, "b": b})


def gain(df):
    """G = 1 + b / (100 t): b the person-weighted mean signed error (cm) over the matched rows, t the
    person-weighted mean true stride (m) over every row (F1's in-sim stride gain, scripts/g3_gain_check.py)."""
    pt = person_table(df)
    b = pt["b"].dropna()
    if not len(b) or not len(pt):
        return float("nan")
    return 1.0 + float(b.mean()) / (100.0 * float(pt["t"].mean()))


def fit_slope(x, y):
    """OLS slope of y on x (with intercept); NaN under 3 points or no spread."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan")
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def beta_takes(path=TAKES_CSV, v_range=BETA_V):
    """{beta, n}: the OLS slope of ln stride on ln cadence over the takes of reports/amass_walks/takes.csv with
    speed in v_range (stride = 120 speed / cadence, m; cadence steps/min)."""
    t = pd.read_csv(path, encoding="utf-8")
    v = pd.to_numeric(t["speed_m_s"], errors="coerce").to_numpy(np.float64)
    c = pd.to_numeric(t["cadence_steps_min"], errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(v) & np.isfinite(c) & (c > 0) & (v >= v_range[0]) & (v <= v_range[1])
    L = 120.0 * v[ok] / c[ok]
    return {"beta": fit_slope(np.log(c[ok]), np.log(L)), "n": int(ok.sum()), "v_range": list(v_range)}


def f_star(rho, beta):
    """rho^-(1 + 1/beta): the cadence of a real walker over that of an AMASS walk of the same stride (Prediction)."""
    if not (beta > 0 and rho > 0):
        return float("nan")
    return float(rho ** -(1.0 + 1.0 / beta))


def interp_loglog(factors, r, f):
    """(r at f, extrapolated): linear in (ln f, ln r) between the measured factors, the nearest segment outside."""
    x = np.log(np.asarray(factors, np.float64))
    y = np.log(np.asarray(r, np.float64))
    o = np.argsort(x)
    x, y = x[o], y[o]
    if not (np.isfinite(y).all() and f > 0 and len(x) >= 2):
        return float("nan"), False
    lf = math.log(f)
    if lf < x[0]:
        j, ext = 0, True
    elif lf > x[-1]:
        j, ext = len(x) - 2, True
    else:
        j, ext = min(int(np.searchsorted(x, lf, side="right")) - 1, len(x) - 2), False
    s = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
    return float(math.exp(y[j] + s * (lf - x[j]))), bool(ext)


def share_of(P, Q):
    """ln P / ln Q: the fraction of the real-over-sim shortfall Q produced by a tempo response P."""
    if not (P > 0 and 0 < Q < 1):
        return float("nan")
    return float(math.log(P) / math.log(Q))


def elasticity(factors, r):
    """The through-origin OLS slope of ln r on ln f (r(1) = 1)."""
    x, y = np.log(np.asarray(factors, np.float64)), np.log(np.asarray(r, np.float64))
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    return float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum()) if ok.any() else float("nan")


def classify(share, p_lo, p_hi, r):
    """The label of the Prediction's criterion: (label, reason). r: {factor: r(f)}."""
    r85, r70 = r.get(0.85, float("nan")), r.get(0.7, float("nan"))
    rising = r85 > 1 and r70 > 1
    if not all(math.isfinite(x) for x in (share, p_lo, p_hi, r85, r70)):
        return "UNDETERMINED", "a quantity of the criterion is not finite"
    if share >= SHARE_SUP and p_hi < P_UP_SUP and r70 < r85 < 1:
        return "SUPPORTED", f"share {share:.3f} >= {SHARE_SUP}, P upper {p_hi:.4f} < {P_UP_SUP}, r(0.7) < r(0.85) < 1"
    if rising:
        return "NOT SUPPORTED", "r rises as the walk slows (r(0.85) > 1 and r(0.7) > 1)"
    if share <= SHARE_NOT and p_lo > P_LO_NOT:
        return "NOT SUPPORTED", f"share {share:.3f} <= {SHARE_NOT} and P lower {p_lo:.4f} > {P_LO_NOT}"
    why = []
    if share >= SHARE_SUP:
        why.append(f"share {share:.3f} >= {SHARE_SUP}")
        if not p_hi < P_UP_SUP:
            why.append(f"but P upper {p_hi:.4f} not below {P_UP_SUP}")
        if not r70 < r85 < 1:
            why.append("but r(0.7) < r(0.85) < 1 fails")
    elif share <= SHARE_NOT:
        why.append(f"share {share:.3f} <= {SHARE_NOT} but P lower {p_lo:.4f} not above {P_LO_NOT}")
    else:
        why.append(f"share {share:.3f} between {SHARE_NOT} and {SHARE_SUP}")
    return "PARTLY", "; ".join(why)


def guards(G, t_ratio, match_ratio, g_sim):
    """[(guard, value, ok)] of the Prediction's validity guards."""
    out = [("G(1) within 0.01 of F1's in-sim stride gain", G.get(1.0, float("nan")) - g_sim,
            abs(G.get(1.0, float("nan")) - g_sim) <= G1_TOL)]
    for f, v in t_ratio.items():
        out.append((f"t({f:g}) / t(1) within {T_TOL} of 1", v, bool(math.isfinite(v) and abs(v - 1) <= T_TOL)))
    for f, v in match_ratio.items():
        out.append((f"matched share at {f:g} over that at 1 at least {MATCH_FRAC}", v,
                    bool(math.isfinite(v) and v >= MATCH_FRAC)))
    return out


def bootstrap(per_factor, factors, fs, Q, n=BOOT_N, seed=BOOT_SEED):
    """Person bootstrap (persons resampled jointly over the factors): {P: (lo, hi), share: (lo, hi), r_<f>: (lo,
    hi), e: (lo, hi)}. per_factor {f: person_table}; every table indexed by the same persons."""
    persons = sorted(set().union(*[set(t.index) for t in per_factor.values()]))
    nP = len(persons)
    gen = np.random.default_rng(seed)
    idx = gen.integers(nP, size=(n, nP))
    counts = np.stack([np.bincount(row, minlength=nP) for row in idx]).astype(np.float64)
    G = {}
    for f in factors:
        pt = per_factor[f].reindex(persons)
        t = pt["t"].to_numpy(np.float64)
        b = pt["b"].to_numpy(np.float64)
        ht, hb = np.isfinite(t), np.isfinite(b)
        tm = (counts * np.where(ht, t, 0)).sum(1) / np.maximum((counts * ht).sum(1), 1e-12)
        bm = (counts * np.where(hb, b, 0)).sum(1) / np.maximum((counts * hb).sum(1), 1e-12)
        G[f] = 1.0 + bm / (100.0 * tm)
    R = {f: G[f] / G[1.0] for f in factors}
    P = np.array([interp_loglog(factors, [R[f][k] for f in factors], fs)[0] for k in range(n)])
    S = np.array([share_of(p, Q) for p in P])
    E = np.array([elasticity(factors, [R[f][k] for f in factors]) for k in range(n)])
    pc = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))  # noqa: E731
    out = {"P": pc(P), "share": pc(S), "e": pc(E), "persons": nP, "n": n, "seed": seed}
    for f in factors:
        if f != 1.0:
            out[f"r_{f:g}"] = pc(R[f])
    return out


# ---------------------------------------------------------------- the dump (GPU)
def dump(a, factors, extras=()):
    """Decode the warped takes at the factors and the secondary extras (module docstring) and write the dump
    folder; returns its path."""
    check_prediction()
    if extras:
        check_amendment()
    from mdfm import paths
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index, body_store
    from mdfm.data.canonical import Refused
    from mdfm.eval import infer as I
    from mdfm.factory import layouts as LY
    from mdfm.factory import synth as SY
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    smoke = bool(a.limit) or a.refine_steps != I.STEPS_EVAL or a.points != I.POINTS_EVAL or a.seed != SEED \
        or a.weights != "ema" or tuple(factors) != FACTORS or a.no_pool
    folder = os.path.join(a.out, f"dump_{loaded.name}" + (f"_{a.tag}" if a.tag else "") + ("_smoke" if smoke else ""))
    paths.assert_not_moveatlas(folder)
    print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}); factors {factors}, secondary {extras}; out {folder}",
          flush=True)
    store = body_store.BodyStore(split="dev")
    if store.split != "dev":
        raise Refused("the H6 probe reads the dev body store only")
    _, aw_dev = EG.amass_walks_rows()
    takes = EG.select_takes(store.takes, extra=aw_dev)
    if a.limit:
        takes = takes.iloc[:int(a.limit)].reset_index(drop=True)
    clash = sorted(set(takes["person_id"]) & loaded.trained_persons())
    if clash:
        raise Refused(f"dev persons among the checkpoint's training persons: {clash[:3]}")
    if set(takes["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("a lockbox person among the dev takes")
    layout = LY.load_layouts()[EG.LAYOUT]
    sampler = SY.SynthSampler(store=store, device=dev)
    bank = SkeletonBank(torch.float64)
    fw = set(takes.loc[takes["forward"], "take_id"])
    S, REF, MOD, TK, lines, counts, failures = [], [], [], [], {}, {}, []
    for f in tuple(factors) + tuple(extras):
        wstore = WarpedStore(store, f)
        parts, infos = {}, []
        t0 = time.time()
        for k, take in takes.iterrows():
            wt = take.copy()
            wt["n_knots"] = int(wstore.takes["n_knots"].iloc[int(take["i"])])
            try:
                r = EG.run_take(loaded.model, sampler, wstore, wt, layout, bank, seed=a.seed,
                                refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                                pool=not a.no_pool)
            except Exception as e:  # noqa: BLE001  (recorded; the read keeps the takes present at every factor)
                failures.append({"factor": f, "take_id": str(take["take_id"]), "error": repr(e)[:300]})
                print(f"  x{f:g} [{k + 1}/{len(takes)}] {take['take_id']}: FAILED {e!r}"[:300], flush=True)
                continue
            infos.append({**r["info"], "factor": f})
            for key, df in r["tables"].items():
                parts.setdefault(key, []).append(df)
            print(f"  x{f:g} [{k + 1}/{len(takes)}] {r['info']['take_id']}: {r['info']['windows']} windows, "
                  f"{r['info']['seconds_infer']:.1f} s", flush=True)
        if not infos:
            continue
        tables = {kk: pd.concat(v, ignore_index=True) for kk, v in parts.items()}
        ids_all = [i["take_id"] for i in infos]
        rows = {"forward": [i for i in ids_all if i in fw], "all": ids_all}
        lines[f], counts[f] = {}, {}
        for row, ids in rows.items():
            if not ids:
                continue
            ln, tabs, cn = EG.row_lines(tables, ids)
            lines[f][row] = {n: {k2: {k3: v3 for k3, v3 in st.items() if k3 != "per_person"} for k2, st in s.items()}
                             for n, s in ln.items() if n in PREDICTORS}
            counts[f][row] = cn
            if row == "all":
                for pred in PREDICTORS:
                    m = tabs[pred]["strides"].copy()
                    m.insert(0, "predictor", pred)
                    m.insert(0, "factor", f)
                    m["forward"] = m["trial_id"].isin(fw)
                    S.append(m)
        ref = tables["ref_strides"].copy()
        ref.insert(0, "factor", f)
        REF.append(ref)
        mod = tables["model_strides"].copy()
        mod.insert(0, "factor", f)
        MOD.append(mod)
        TK.append(pd.DataFrame(infos))
        print(f"x{f:g}: {len(infos)} takes in {time.time() - t0:.0f} s", flush=True)
    os.makedirs(folder, exist_ok=True)
    keep = ["factor", "predictor", "trial_id", "person_id", "foot", "t0", "t1", "length_m", "scored", "pred_m",
            "end_diff_s", "matched", "err_cm", "forward"]
    sdf = pd.concat(S, ignore_index=True)
    EG.common.write_csv(sdf[[c for c in keep if c in sdf.columns]], os.path.join(folder, "strides.csv"))
    EG.common.write_csv(pd.concat(REF, ignore_index=True), os.path.join(folder, "ref_strides.csv"))
    EG.common.write_csv(pd.concat(MOD, ignore_index=True), os.path.join(folder, "model_strides.csv"))
    EG.common.write_csv(pd.concat(TK, ignore_index=True), os.path.join(folder, "takes.csv"))
    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    doc = {"task": TASK, "what": LABEL, "prediction_sha256": check_prediction(), "smoke": smoke,
           "checkpoint": loaded.describe(), "factors": list(factors), "extra_factors": list(extras),
           "amendment_sha256": check_amendment() if extras else None,
           "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "batch_windows": a.batch_windows,
                         "seed": a.seed, "shape_pooling": "none" if a.no_pool else "per take", "weights": a.weights,
                         "device": str(dev), "layout": EG.LAYOUT, "memory": mem, "limit": a.limit},
           "sets": {"n_takes": int(len(takes)), "n_persons": int(takes["person_id"].nunique()),
                    "forward_takes": int(takes["forward"].sum()),
                    "persons_sha256": amass_index.person_list_sha256(sorted(set(takes["person_id"]))),
                    "split": "dev", "trained_persons_among_takes": 0, "lockbox_persons_among_takes": 0},
           "lines": {f"{f:g}": v for f, v in lines.items()}, "counts": {f"{f:g}": v for f, v in counts.items()},
           "failures": failures, "moveatlas_head": PV.moveatlas_head(),
           "code_sha256": EG.common.code_sha256([os.path.abspath(__file__), EG.__file__, I.__file__]),
           "seconds": round(time.time() - t_start, 1), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    EG.common.write_text(os.path.join(folder, "dump.json"), json.dumps(EG.clean(doc), indent=1) + "\n")
    print(f"written {EG.common.rel(folder)} in {doc['seconds']} s", flush=True)
    return folder


# ---------------------------------------------------------------- the read (CPU)
def f1_gains(path=F1_RESULTS):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    e = d["ends"]["v1_B"]
    return {"g_real": float(e["gain"]["stride"]["gain"]), "g_sim": float(e["insim"]["stride_gain"])}


def rho_h3(path=H3_DUMPS):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    x = d["part_b"]["descriptive_dumps_2026_10_05"]["real_stride_over_insim_stride_at_equal_speed"]
    return {"rho": float(x["median_of_person_medians"]), "interval": [float(v) for v in x["interval"]]}


def factor_rows(S, factors, takes_by_f, forward):
    """{predictor: {f: strides rows}} on the takes present at every factor (paired), forward row or all."""
    common_t = set.intersection(*[set(takes_by_f[f]) for f in factors])
    out = {}
    for pred in PREDICTORS:
        out[pred] = {}
        for f in factors:
            d = S[(S["predictor"] == pred) & (np.isclose(S["factor"], f)) & S["trial_id"].isin(common_t)]
            if forward:
                d = d[d["forward"].astype(str).str.lower() == "true"]
            out[pred][f] = d
    return out, sorted(common_t)


def read(folder, out):
    """The registered quantities of the Prediction from a dump folder (module docstring)."""
    pred_sha = check_prediction()
    from mdfm.train import provenance as PV
    with open(os.path.join(folder, "dump.json"), encoding="utf-8") as f:
        dd = json.load(f)
    if dd.get("prediction_sha256") != pred_sha:
        raise AssertionError("the dump was made under another prediction")
    smoke = bool(dd.get("smoke"))
    factors = tuple(float(x) for x in dd["factors"])
    S = pd.read_csv(os.path.join(folder, "strides.csv"), encoding="utf-8")
    S["matched"] = S["matched"].astype(str).str.lower() == "true"
    TK = pd.read_csv(os.path.join(folder, "takes.csv"), encoding="utf-8")
    REF = pd.read_csv(os.path.join(folder, "ref_strides.csv"), encoding="utf-8")
    MOD = pd.read_csv(os.path.join(folder, "model_strides.csv"), encoding="utf-8")
    takes_by_f = {f: set(TK.loc[np.isclose(TK["factor"], f), "take_id"].astype(str)) for f in factors}
    f1, rh, bt = f1_gains(), rho_h3(), beta_takes()
    Q = f1["g_real"] / f1["g_sim"]
    fs = f_star(rh["rho"], bt["beta"])
    res = {"task": TASK, "what": LABEL, "prediction_sha256": pred_sha, "dump": EG.common.rel(folder),
           "smoke": smoke, "checkpoint_sha256": dd["checkpoint"].get("sha256"), "factors": list(factors),
           "f1": f1, "Q": Q, "rho": rh, "beta_train_walks": bt, "f_star": fs, "rows": {},
           "failures": dd.get("failures", [])}
    if dd["checkpoint"].get("sha256") != CHECKPOINT_SHA256 and not smoke:
        raise ValueError("the dump holds another checkpoint than v1_B")
    for row, fwd in (("forward", True), ("all", False)):
        rows, paired = factor_rows(S, factors, takes_by_f, fwd)
        block = {"paired_takes": len(paired), "predictors": {}}
        for pred in PREDICTORS:
            G = {f: gain(rows[pred][f]) for f in factors}
            r = {f: G[f] / G[1.0] for f in factors}
            t1 = rows[pred][1.0]["length_m"].mean() if len(rows[pred][1.0]) else float("nan")
            P, ext = interp_loglog(factors, [r[f] for f in factors], fs)
            pb = {"G": {f"{f:g}": G[f] for f in factors}, "r": {f"{f:g}": r[f] for f in factors},
                  "e": elasticity(factors, [r[f] for f in factors]), "P": P, "P_extrapolated": ext,
                  "share": share_of(P, Q),
                  "strides": {f"{f:g}": int(len(rows[pred][f])) for f in factors},
                  "matched": {f"{f:g}": int(rows[pred][f]["matched"].sum()) for f in factors},
                  "persons": {f"{f:g}": int(rows[pred][f]["person_id"].nunique()) for f in factors},
                  "true_stride_ratio": {f"{f:g}": float(rows[pred][f]["length_m"].mean() / t1) for f in factors}}
            ms = {f: float(rows[pred][f]["matched"].mean()) if len(rows[pred][f]) else float("nan") for f in factors}
            pb["matched_share"] = {f"{f:g}": ms[f] for f in factors}
            per = {f: person_table(rows[pred][f]) for f in factors}
            if all(len(per[f]) for f in factors) and math.isfinite(P):
                pb["bootstrap"] = bootstrap(per, factors, fs, Q)
            # sensitivity of P to rho and beta (point only)
            sens = {}
            for name, rr, bb in (("rho_lo", rh["interval"][0], bt["beta"]), ("rho_hi", rh["interval"][1], bt["beta"]),
                                 ("beta_minus_0.25", rh["rho"], bt["beta"] - 0.25),
                                 ("beta_plus_0.25", rh["rho"], bt["beta"] + 0.25)):
                fx = f_star(rr, bb)
                px, ex = interp_loglog(factors, [r[f] for f in factors], fx)
                sens[name] = {"f_star": fx, "P": px, "share": share_of(px, Q), "extrapolated": ex}
            pb["sensitivity"] = sens
            block["predictors"][pred] = pb
        # cadence check: true and decoded stride frequency of the paired takes
        cad = {}
        for f in factors:
            rf = REF[np.isclose(REF["factor"], f) & REF["trial_id"].isin(paired)]
            mf = MOD[np.isclose(MOD["factor"], f) & MOD["trial_id"].isin(paired)]
            if fwd:
                fw_ids = set(S.loc[S["forward"].astype(str).str.lower() == "true", "trial_id"])
                rf, mf = rf[rf["trial_id"].isin(fw_ids)], mf[mf["trial_id"].isin(fw_ids)]
            dt_t, dt_m = (rf["t1"] - rf["t0"]).mean(), (mf["t1"] - mf["t0"]).mean()
            cad[f"{f:g}"] = {"true_stride_time_s": float(dt_t), "model_stride_time_s": float(dt_m),
                             "decoded_over_true_cadence": float(dt_t / dt_m) if dt_m > 0 else float("nan")}
        block["cadence"] = cad
        # dev-walk beta from the factor-1 scored true strides
        r1 = REF[np.isclose(REF["factor"], 1.0) & REF["trial_id"].isin(paired)]
        r1 = r1[r1["scored"].astype(str).str.lower() == "true"]
        if fwd:
            r1 = r1[r1["trial_id"].isin(set(S.loc[S["forward"].astype(str).str.lower() == "true", "trial_id"]))]
        block["beta_dev_walks"] = {"beta": fit_slope(np.log(1.0 / (r1["t1"] - r1["t0"])), np.log(r1["length_m"])),
                                   "n": int(len(r1))}
        res["rows"][row] = block
    # secondary (amendment 1): the extra factors with the registered ones, predictor model, no criterion
    extras = tuple(float(x) for x in dd.get("extra_factors") or [])
    if extras:
        allf = tuple(sorted(factors + extras))
        tb_all = {**takes_by_f, **{f: set(TK.loc[np.isclose(TK["factor"], f), "take_id"].astype(str)) for f in extras}}
        sx = {"amendment_sha256": dd.get("amendment_sha256"), "factors": list(allf), "rows": {}}
        for row, fwd in (("forward", True), ("all", False)):
            if any(not tb_all[f] for f in allf):
                sx["rows"][row] = {"missing": [f for f in allf if not tb_all[f]]}
                continue
            rx, paired_x = factor_rows(S, allf, tb_all, fwd)
            m = rx["model"]
            G = {f: gain(m[f]) for f in allf}
            r = {f: G[f] / G[1.0] for f in allf}
            t1 = m[1.0]["length_m"].mean() if len(m[1.0]) else float("nan")
            P, ext = interp_loglog(allf, [r[f] for f in allf], fs)
            b = {"paired_takes": len(paired_x), "G": {f"{f:g}": G[f] for f in allf}, "r": {f"{f:g}": r[f] for f in allf},
                 "e": elasticity(allf, [r[f] for f in allf]), "P": P, "P_extrapolated": ext, "share": share_of(P, Q),
                 "strides": {f"{f:g}": int(len(m[f])) for f in allf},
                 "matched_share": {f"{f:g}": float(m[f]["matched"].mean()) if len(m[f]) else float("nan")
                                   for f in allf},
                 "true_stride_ratio": {f"{f:g}": float(m[f]["length_m"].mean() / t1) for f in allf}}
            per = {f: person_table(m[f]) for f in allf}
            if all(len(per[f]) for f in allf) and math.isfinite(P):
                b["bootstrap"] = bootstrap(per, allf, fs, Q)
            sx["rows"][row] = b
        res["secondary_extra"] = sx
    # the registered reading
    p = res["rows"]["forward"]["predictors"]["model"]
    r = {float(k): v for k, v in p["r"].items()}
    bs = p.get("bootstrap") or {}
    p_lo, p_hi = (bs.get("P") or (float("nan"), float("nan")))
    label, why = classify(p["share"], p_lo, p_hi, r)
    G = {float(k): v for k, v in p["G"].items()}
    t_ratio = {float(k): v for k, v in p["true_stride_ratio"].items() if float(k) != 1.0}
    m1 = p["matched_share"]["1"]
    m_ratio = {float(k): v / m1 for k, v in p["matched_share"].items() if float(k) != 1.0}
    gd = guards(G, t_ratio, m_ratio, f1["g_sim"])
    failed = [g[0] for g in gd if not g[2]]
    if failed:
        label, why = "UNDETERMINED", "guard failed: " + "; ".join(failed) + f" (criterion alone: {label}, {why})"
    if smoke:
        label, why = "SMOKE", f"smoke dump, not the registered reading (criterion as applied: {label}; {why})"
    if p.get("P_extrapolated"):
        why += "; P extrapolated beyond the factor grid"
    res["guards"] = [{"guard": g[0], "value": g[1], "pass": bool(g[2])} for g in gd]
    res["reading"] = {"label": label, "reason": why, "row": "forward", "predictor": "model"}
    res["moveatlas_head"] = PV.moveatlas_head()
    res["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sfx = "_smoke" if smoke else ""
    os.makedirs(out, exist_ok=True)
    EG.common.write_text(os.path.join(out, f"results{sfx}.json"), json.dumps(EG.clean(res), indent=1) + "\n")
    pf = []
    for row, b in res["rows"].items():
        for pred, x in b["predictors"].items():
            for f in factors:
                k = f"{f:g}"
                pf.append({"row": row, "predictor": pred, "factor": f, "G": x["G"][k], "r": x["r"][k],
                           "strides": x["strides"][k], "matched": x["matched"][k], "persons": x["persons"][k],
                           "true_stride_ratio": x["true_stride_ratio"][k], "matched_share": x["matched_share"][k],
                           "decoded_over_true_cadence": b["cadence"][k]["decoded_over_true_cadence"]})
    for row, b in (res.get("secondary_extra") or {}).get("rows", {}).items():
        for k in b.get("G", {}):
            pf.append({"row": f"{row}_extra", "predictor": "model", "factor": float(k), "G": b["G"][k], "r": b["r"][k],
                       "strides": b["strides"][k], "true_stride_ratio": b["true_stride_ratio"][k],
                       "matched_share": b["matched_share"][k]})
    EG.common.write_csv(pd.DataFrame(pf), os.path.join(out, f"per_factor{sfx}.csv"))
    EG.common.write_text(os.path.join(out, f"summary{sfx}.md"), summary(res))
    prov = {"script": "scripts/f2_H6warp.py", "run_at": res["run_at"], "moveatlas_head": res["moveatlas_head"],
            "dump_moveatlas_head": dd.get("moveatlas_head"),
            "code_sha256": {EG.common.rel(q): sha256_file(q) for q in (os.path.abspath(__file__), EG.__file__)},
            "inputs_sha256": {EG.common.rel(q): sha256_file(q) for q in
                              (F1_RESULTS, H3_DUMPS, TAKES_CSV, os.path.join(folder, "dump.json"),
                               os.path.join(folder, "strides.csv"))}}
    EG.common.write_text(os.path.join(out, f"_provenance{sfx}.json"), json.dumps(prov, indent=1) + "\n")
    print(json.dumps(EG.clean(res["reading"]), indent=1))
    return res


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def summary(res):
    L = [f"# {TASK}: decoded stride under a time warp (v1_B, AMASS dev walks)", "",
         f"Label: {LABEL}. Written by scripts/f2_H6warp.py on {res['run_at']} from {res['dump']}; prediction SHA-256 "
         f"{res['prediction_sha256']} (asserted before any value was read); MoveAtlas {str(res['moveatlas_head'])[:7]}.",
         "", f"Reading: **{res['reading']['label']}** ({res['reading']['reason']}).", "",
         f"Q = G_real / G_sim = {fmt(res['f1']['g_real'])} / {fmt(res['f1']['g_sim'])} = {fmt(res['Q'])}; rho "
         f"{fmt(res['rho']['rho'])}; beta (AMASS train walks, {res['beta_train_walks']['n']} takes) "
         f"{fmt(res['beta_train_walks']['beta'], 3)}; f* {fmt(res['f_star'])}.", ""]
    for row, b in res["rows"].items():
        L += [f"## Row {row} ({b['paired_takes']} takes present at every factor)", "",
              "| predictor | factor | G | r | strides | matched share | true stride / at 1 | decoded / true cadence |",
              "|---|---|---|---|---|---|---|---|"]
        for pred, x in b["predictors"].items():
            for k in x["G"]:
                L.append(f"| {pred} | {k} | {fmt(x['G'][k])} | {fmt(x['r'][k])} | {x['strides'][k]} | "
                         f"{fmt(x['matched_share'][k], 3)} | {fmt(x['true_stride_ratio'][k])} | "
                         f"{fmt(b['cadence'][k]['decoded_over_true_cadence'])} |")
        L.append("")
        for pred, x in b["predictors"].items():
            bs = x.get("bootstrap") or {}
            iv = lambda k: f"[{fmt(bs[k][0])}, {fmt(bs[k][1])}]" if k in bs else ""  # noqa: E731
            L.append(f"- {pred}: e {fmt(x['e'], 3)} {iv('e')}; P {fmt(x['P'])} {iv('P')}"
                     f"{' (extrapolated)' if x['P_extrapolated'] else ''}; share {fmt(x['share'], 3)} {iv('share')}; "
                     + "; ".join(f"r({k[2:]}) {iv(k)}" for k in bs if k.startswith("r_")) + ".")
            L.append("  Sensitivity: " + "; ".join(f"{n} f* {fmt(s['f_star'])} P {fmt(s['P'])} share "
                                                    f"{fmt(s['share'], 3)}" for n, s in x["sensitivity"].items()) + ".")
        L += [f"- beta of the dev-walk truth (factor 1, ln stride on ln stride frequency, {b['beta_dev_walks']['n']} "
              f"strides): {fmt(b['beta_dev_walks']['beta'], 3)}.", ""]
    sx = res.get("secondary_extra")
    if sx:
        L += ["## Secondary, amendment 1: the extra factors with the registered ones (predictor model, no criterion)",
              ""]
        for row, b in sx["rows"].items():
            if "G" not in b:
                L.append(f"- {row}: factors without a take: {b.get('missing')}")
                continue
            bs = b.get("bootstrap") or {}
            iv = lambda k: f" [{fmt(bs[k][0])}, {fmt(bs[k][1])}]" if k in bs else ""  # noqa: E731
            L.append(f"- {row} ({b['paired_takes']} takes): " + "; ".join(
                f"r({k}) {fmt(b['r'][k])} (true stride ratio {fmt(b['true_stride_ratio'][k])}, matched share "
                f"{fmt(b['matched_share'][k], 3)})" for k in b["r"]) + f"; e {fmt(b['e'], 3)}{iv('e')}; P "
                f"{fmt(b['P'])}{iv('P')}{' (extrapolated)' if b['P_extrapolated'] else ''}; share "
                f"{fmt(b['share'], 3)}{iv('share')}.")
        L.append("")
    L += ["## Guards", ""] +[f"- {g['guard']}: {fmt(g['value'])} ({'pass' if g['pass'] else 'FAIL'})"
                              for g in res["guards"]]
    if res["failures"]:
        L += ["", f"Failed take renders: {len(res['failures'])} (results.json, failures)."]
    L += ["", "Every number above comes from results.json beside this file.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- the early read (existing probe outputs)
def early(out, files=RM_FILES):
    """The preliminary secondary read of the scripts/probe_root_motion.py warp rows (module docstring)."""
    pred_sha = check_prediction()
    f1, rh, bt = f1_gains(), rho_h3(), beta_takes()
    Q = f1["g_real"] / f1["g_sim"]
    fs = f_star(rh["rho"], bt["beta"])
    pts, src = {}, []
    for path in files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if (d.get("checkpoint") or {}).get("sha256") != CHECKPOINT_SHA256:
            raise ValueError(f"{path}: not the v1_B checkpoint")
        rows = d["rows"]
        g = {float(x["factor"]): 1.0 + float(x["stride_bias_cm"]) / (100.0 * float(x["true_stride_m"])) for x in rows}
        for f, v in g.items():
            if f != 1.0 and f not in pts:           # a factor in both files: the first file's (both in sources)
                pts[f] = v / g[1.0]
        src.append({"file": EG.common.rel(path), "refine_steps": d.get("refine_steps"), "layout": d.get("layout"),
                    "split": d.get("split"), "G": {f"{f:g}": v for f, v in g.items()},
                    "r": {f"{f:g}": v / g[1.0] for f, v in g.items()},
                    "strides": {f"{float(x['factor']):g}": x["strides"] for x in rows},
                    "true_stride_m": {f"{float(x['factor']):g}": x["true_stride_m"] for x in rows}})
    fac = sorted(set(pts) | {1.0})
    r = [1.0 if f == 1.0 else pts[f] for f in fac]
    P, ext = interp_loglog(fac, r, fs)
    sh = share_of(P, Q)
    point = ("SUPPORTED-like" if sh >= SHARE_SUP else "NOT SUPPORTED-like" if sh <= SHARE_NOT else "PARTLY-like")
    res = {"task": TASK, "what": LABEL + "; preliminary secondary read of existing probe outputs, no criterion",
           "prediction_sha256": pred_sha, "Q": Q, "rho": rh["rho"], "beta_train_walks": bt, "f_star": fs,
           "r": {f"{f:g}": v for f, v in zip(fac, r)},
           "e_per_factor": {f"{f:g}": math.log(v) / math.log(f) for f, v in zip(fac, r) if f != 1.0},
           "P": P, "P_extrapolated": ext, "share": sh, "point_label": point, "sources": src,
           "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(out, exist_ok=True)
    EG.common.write_text(os.path.join(out, "early.json"), json.dumps(EG.clean(res), indent=1) + "\n")
    print(json.dumps(EG.clean({k: res[k] for k in ("f_star", "r", "e_per_factor", "P", "P_extrapolated", "share",
                                                    "point_label", "beta_train_walks")}), indent=1))
    return res


# ---------------------------------------------------------------- command line
def parse(argv=None):
    from mdfm.eval import infer as I
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", action="store_true", help="decode the warped takes, then read")
    ap.add_argument("--read", default=None, metavar="DUMP", help="read a dump folder")
    ap.add_argument("--early", action="store_true", help="the preliminary read of the existing probe outputs")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-gb", type=float, default=12.0)
    ap.add_argument("--factors", default=",".join(f"{f:g}" for f in FACTORS))
    ap.add_argument("--extra-factors", default=",".join(f"{f:g}" for f in EXTRA_FACTORS),
                    help="secondary factors of amendment 1 ('' for none)")
    ap.add_argument("--refine-steps", type=int, default=I.STEPS_EVAL)
    ap.add_argument("--points", type=int, default=I.POINTS_EVAL)
    ap.add_argument("--batch-windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="the first N takes only (a smoke dump)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if sum([a.dump, a.read is not None, a.early]) != 1:
        ap.error("one of --dump, --read DUMP, --early")
    try:
        a.factor_list = parse_factors(a.factors)
        a.extra_list = parse_extra(a.extra_factors, a.factor_list)
    except ValueError as e:
        ap.error(str(e))
    return a


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    check_prediction()
    if a.early:
        early(a.out)
    elif a.dump:
        folder = dump(a, a.factor_list, a.extra_list)
        read(folder, a.out)
    else:
        read(a.read, a.out)


if __name__ == "__main__":
    main()
