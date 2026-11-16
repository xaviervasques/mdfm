"""CPL post hoc checks PH7 to PH11 (reports/bench/cpl/posthoc_plan_3.txt, amendment 3 of the CPL prediction; post hoc,
written after the review of the CPL report and hashed before this script ran). Dev diagnostic, not a gate read; CPU
only; no model.

Reads the stride tables the registered run saved (reports/bench/cpl/real_strides.csv, amass_train_strides.csv; their
persons guarded: the real ones inside train54 and outside every lockbox list, the AMASS ones outside the AMASS lockbox)
and the train body store's takes table (metadata only). The relations, delta, gamma, kappa and the joint person
bootstrap are those of scripts/cpl_coupling.py (imported, not rewritten); h = log gamma / log Q with the registered Q.
- PH7 reference: gamma_L (thigh, shank, foot) and h against five AMASS references: pooled (the registered relation),
  stream (the strides weighted as the synthetic training stream draws them, BodyStore.sample through
  mdfm/factory/synth.py: a stride of duration T of person p weighs T / N_p, N_p the seconds of all p's takes), treadmill,
  overground, KIT.
- PH8 distal sites (ankle site and foot) per reference; PH9 the thigh left out (shank and foot) per reference, single
  sites, and each primary site's contribution to log gamma_L under the pooled reference; PH10 stature sensitivity
  (real stature x 0.97, 0.98, 1.02, 1.03; real leg ratio 0.51, 0.55; without weargait:HC134); PH11 stream shares of
  the straight forward walks and of their treadmill takes.

Writes reports/bench/cpl/posthoc3/: results.json, summary.md, _provenance.json.

Run (mdfm environment, CPU only):
  CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/cpl_posthoc3.py [--out DIR] [--boot N]
"""
import argparse
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

REGISTERED = os.path.join(TREE, "reports", "bench", "cpl")
OUT = os.path.join(REGISTERED, "posthoc3")
PLAN = os.path.join(REGISTERED, "posthoc_plan_3.txt")
PLAN_SHA_FILE = os.path.join(REGISTERED, "posthoc_plan_3.sha256")
PLAN_SHA256 = "93acb06f3ad52a5e88138ca37fd797b142cd58528705c3617c8c43257d630ec3"
LABEL = "dev diagnostic, not a gate read; post hoc (amendment 3)"
REFERENCES = ("pooled", "stream", "treadmill", "overground", "KIT")
SITE_SETS = {"thigh_shank_foot": ("thigh", "shank", "foot"), "shank_foot": ("shank", "foot"),
             "ankle_foot": ("ankle", "foot")}
SIZE_H, SIZE_LO = 0.75, 0.5                 # the registered upstream condition (prediction.txt section 4)
PH5_BAND = (0.965, 1.035)
STATURE_FACTORS = (0.97, 0.98, 1.02, 1.03)
LEG_RATIOS = (0.51, 0.55)
FLAGGED = "weargait:HC134"                  # height 47 in: known issue of the weargait_pd datacard (MoveAtlas 0003)


def _load(name, key):
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[key] = m
    spec.loader.exec_module(m)
    return m


C = _load("cpl_coupling", "_cplph3_cpl_coupling")
SCRIPT_SHA256_AT_START = C.sha256_file(os.path.abspath(__file__))


# ---------------------------------------------------------------- weighted relations (the stream reference)
def blocks_w(pid, X, y, w):
    """{person: (X'WX, X'Wy, y'Wy)} with the given row weights (a person's block weighs the sum of its rows)."""
    out = {}
    pid = np.asarray(pid)
    for p in np.unique(pid):
        m = pid == p
        W = w[m]
        out[p] = (X[m].T @ (W[:, None] * X[m]), X[m].T @ (W * y[m]), float(W @ (y[m] * y[m])))
    return out


class WRelations(C.Relations):
    """C.Relations with row weights: the same blocks, design and inverse relation, persons weighted by the sum of
    their rows' weights instead of equally. Boot and fit work on it unchanged."""

    def __init__(self, amass, row_w, kind="stride", segments=C.PRIMARY):
        self.kind, self.segments = kind, tuple(segments)
        self.blocks, self.persons = {}, {}
        row_w = np.asarray(row_w, dtype=np.float64)
        pid = amass["person_id"].to_numpy()
        X = C.design(amass, kind)
        for s in C.SEGMENTS:
            y = np.log(amass[f"rom_{s}"].to_numpy())
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            self.blocks[s] = blocks_w(pid[ok], X[ok], y[ok], row_w[ok])
        Xi = np.column_stack([np.ones(len(amass))] + [np.log(amass[f"rom_{s}"].to_numpy()) for s in C.PRIMARY]
                             + [np.log(amass["c"].to_numpy()) - C.C_CENTRE])
        yi = np.log(amass["S"].to_numpy() / amass["l"].to_numpy())
        ok = np.isfinite(Xi).all(1) & np.isfinite(yi)
        self.blocks["inverse"] = blocks_w(pid[ok], Xi[ok], yi[ok], row_w[ok])


def stream_weights(am, person_seconds):
    """Row weights of an AMASS stride table under BodyStore.sample's rule: T / N_p (N_p the person's seconds over
    all takes), scaled so that the person weights (sums of their rows) have a mean of 1."""
    T = am["T"].to_numpy(np.float64)
    Np = np.array([person_seconds[str(p)] for p in am["person_id"]], dtype=np.float64)
    w = T / Np
    per_person = {}
    for p, x in zip(am["person_id"].to_numpy(), w):
        per_person[p] = per_person.get(p, 0.0) + x
    return w / float(np.mean(list(per_person.values())))


def stream_shares(takes, persons, straight_idx):
    """PH11 on a takes table: the expected share of BodyStore.sample's knots (a person uniformly among `persons`, a
    take in proportion to its knots) in the rows straight_idx, and the treadmill share within them."""
    nk = takes["n_knots"].to_numpy(np.float64)
    Np = takes.groupby("person_id")["n_knots"].transform("sum").to_numpy(np.float64)
    w = nk / Np / float(len(persons))
    sel = np.zeros(len(takes), bool)
    sel[np.asarray(straight_idx, np.int64)] = True
    tread = takes["treadmill"].to_numpy() != 0
    walk = (takes["task_name"] == "walk").to_numpy()
    s = float(w[sel].sum())
    sw = float(w[walk].sum())
    return {"share_straight_forward": s, "treadmill_share_of_straight_forward": float(w[sel & tread].sum()) / s
            if s > 0 else float("nan"), "share_task_walk": sw,
            "treadmill_share_of_task_walk": float(w[walk & tread].sum()) / sw if sw > 0 else float("nan"),
            "persons_in_store": int(len(persons))}


def table_treadmill_shares(am, row_w):
    """Treadmill share of an AMASS stride table under persons weighted equally and under the stream weights."""
    t = (am["treadmill"].to_numpy() != 0).astype(np.float64)
    pid = am["person_id"].to_numpy()
    shares = [float(t[pid == p].mean()) for p in np.unique(pid)]
    n_tread = sum(1 for x in shares if x > 0)
    return {"persons_equal": float(np.mean(shares)), "stream": float((row_w * t).sum() / row_w.sum()),
            "persons": int(len(shares)), "persons_with_treadmill_strides": int(n_tread),
            "persons_with_overground_strides": int(sum(1 for x in shares if x < 1))}


# ---------------------------------------------------------------- rows
def h_of(g, Q):
    return math.log(g) / math.log(Q) if (g is not None and math.isfinite(g) and g > 0) else float("nan")


def h_interval(iv, Q):
    return sorted([h_of(x, Q) for x in iv]) if iv else None


def ref_row(rel, rc, segments, Q, n_boot, seed):
    """gamma over `segments`, its interval, h and its interval, single-site gammas and kappas, on the stride table rc
    against the relations `rel` (C.Relations or WRelations)."""
    fit = rel.fit()
    pt = C.statistics(rc, fit, segments)
    d = C.deltas(rc, fit, segments)
    ok = np.isfinite(d)
    iv = C.Boot(rel, rc, segments=segments).run(n_boot, seed) if n_boot else {}
    g = pt["gamma_L"]
    out = {"segments": list(segments), "gamma": g, "interval": iv.get("gamma_L"), "h": h_of(g, Q),
           "h_interval": h_interval(iv.get("gamma_L"), Q), "persons": int(len(set(rc["person_id"].to_numpy()[ok]))),
           "strides": int(ok.sum()),
           "single": {s: {"gamma": pt[f"gamma_{s}"], "interval": iv.get(f"gamma_{s}"),
                          "kappa": pt[f"kappa_{s}"], "kappa_interval": iv.get(f"kappa_{s}")} for s in C.SEGMENTS}}
    return out


def contributions(rc, fit, Q, segments=C.PRIMARY):
    """Each site's term w_s b_s r_s / sum w b^2 of delta: median over persons of the person median, its share of
    log gamma_L, and the exp of the sum of the site medians beside gamma_L (medians do not add)."""
    pid = rc["person_id"].to_numpy()
    r = C.residuals(rc, fit)
    D = sum(fit[s][0][1] ** 2 / fit[s][1] for s in segments)
    lg = C.med_of_persons(C.person_median(pid, C.deltas(rc, fit, segments)))
    out = {"log_gamma_L": lg, "gamma_L": math.exp(lg), "sites": {}}
    tot = 0.0
    for s in segments:
        term = (fit[s][0][1] / fit[s][1]) * r[s] / D
        c = C.med_of_persons(C.person_median(pid, term))
        tot += c
        out["sites"][s] = {"term": c, "factor": math.exp(c), "share_of_log_gamma_L": c / lg if lg else float("nan"),
                           "h_part": c / math.log(Q), "weight": (fit[s][0][1] ** 2 / fit[s][1]) / D}
    out["sum_of_terms"] = tot
    out["exp_sum_of_terms"] = math.exp(tot)
    return out


def with_leg(rc, factor):
    """A copy of a real stride table whose leg is factor x LEG_RATIO x stature."""
    e = rc.copy()
    e["l"] = C.LEG_RATIO * e["stature_m"] * float(factor)
    return e


def size_reading(row, names=("SIZE", "PART", "NONE"), prefix=""):
    """Thresholds of PH7's stream row (and PH9): h at least 0.75 with its lower bound at least 0.5; lower bound of h
    above 0; otherwise."""
    h, hi = row["h"], row["h_interval"]
    if hi is None or not math.isfinite(h):
        return "n/a"
    if h >= SIZE_H and hi[0] >= SIZE_LO:
        return f"{prefix}{names[0]}"
    if hi[0] > 0:
        return f"{prefix}{names[1]}"
    return f"{prefix}{names[2]}"


def band_reading(g):
    if not math.isfinite(g):
        return "n/a"
    if g < PH5_BAND[0]:
        return "BELOW"
    if g > PH5_BAND[1]:
        return "ABOVE"
    return "MATCH"


def readings(res):
    refs = res["PH7"]["references"]
    meets = {k: (v["thigh_shank_foot"]["h"] >= SIZE_H and v["thigh_shank_foot"]["h_interval"][0] >= SIZE_LO)
             for k, v in refs.items()}
    ph7 = "SIZE HOLDS ACROSS REFERENCES" if all(meets.values()) else "SIZE REFERENCE-DEPENDENT"
    stream = size_reading(refs["stream"]["thigh_shank_foot"], ("SIZE", "PART", "NONE"), "STREAM ")
    distal = {k: band_reading(v["ankle_foot"]["gamma"]) for k, v in refs.items()}
    ph8 = "DISTAL MATCH ROBUST" if all(x == "MATCH" for x in distal.values()) else "DISTAL MATCH REFERENCE-DEPENDENT"
    ph9 = {k: size_reading(refs[k]["shank_foot"], ("SIZE WITHOUT THE THIGH", "PART WITHOUT THE THIGH",
                                                    "NONE WITHOUT THE THIGH")) for k in ("pooled", "stream")}
    return {"PH7": ph7, "PH7_meets": meets, "PH7_stream": stream, "PH8": ph8, "PH8_per_reference": distal,
            "PH9": ph9}


# ---------------------------------------------------------------- inputs
def load_tables(locked, train54, real_path=None, am_path=None):
    """(real, amass) complete stride tables of the registered run, persons guarded."""
    import pandas as pd
    from mdfm.data import amass_index
    from mdfm.data.canonical import Refused
    real_path = real_path or os.path.join(REGISTERED, "real_strides.csv")
    am_path = am_path or os.path.join(REGISTERED, "amass_train_strides.csv")
    who = pd.read_csv(real_path, encoding="utf-8", usecols=["person_id"], dtype=str)
    C.guard(set(who["person_id"]), locked, train54, "real strides")
    am_who = pd.read_csv(am_path, encoding="utf-8", usecols=["person_id"], dtype=str)
    if set(am_who["person_id"]) & set(amass_index.LOCKBOX_PERSONS):
        raise Refused("AMASS strides: a lockbox person")
    real = pd.read_csv(real_path, encoding="utf-8", dtype={"person_id": str, "unit": str})
    am = pd.read_csv(am_path, encoding="utf-8", dtype={"person_id": str, "unit": str, "archive": str})
    return C.complete(real), C.complete(am)


def store_metadata():
    """(person seconds {person: s}, PH11 shares) from the train body store's takes table."""
    from mdfm.data import body_store
    store = body_store.BodyStore(split="train")
    tk = store.takes
    secs = (tk.groupby("person_id")["n_knots"].sum() / C.RATE).to_dict()
    straight = C.amass_takes(store, "train")
    shares = stream_shares(tk, store.persons, straight["i"].to_numpy(np.int64))
    shares["straight_forward_takes"] = int(len(straight))
    return {str(k): float(v) for k, v in secs.items()}, shares


# ---------------------------------------------------------------- analysis
def analyse(real, am, person_seconds, Q, n_boot, registered=None):
    res = {"Q": Q}
    row_w = stream_weights(am, person_seconds)
    tread = (am["treadmill"] != 0).to_numpy()
    subsets = {"treadmill": tread, "overground": ~tread, "KIT": (am["archive"] == "KIT").to_numpy()}
    rels = {"pooled": C.Relations(am), "stream": WRelations(am, row_w)}
    for k, m in subsets.items():
        rels[k] = C.Relations(am[m].reset_index(drop=True))
    refs = {}
    for i, name in enumerate(REFERENCES):
        refs[name] = {}
        for j, (sname, segs) in enumerate(SITE_SETS.items()):
            refs[name][sname] = ref_row(rels[name], real, segs, Q, n_boot, C.SEED + 700 + 10 * i + j)
        refs[name]["persons_amass"] = int(len(rels[name].blocks["thigh"]))
        refs[name]["coefficients"] = C.coefficients(rels[name].fit())
    res["PH7"] = {"references": refs}
    res["PH9_contributions_pooled"] = contributions(real, rels["pooled"].fit(), Q)
    res["PH9_contributions_stream"] = contributions(real, rels["stream"].fit(), Q)
    sens = {}
    for k, fct in enumerate(STATURE_FACTORS):
        sens[f"stature_x{fct:.2f}"] = ref_row(rels["pooled"], with_leg(real, fct), C.PRIMARY, Q, n_boot,
                                              C.SEED + 800 + k)
    for k, lr in enumerate(LEG_RATIOS):
        sens[f"leg_ratio_{lr:.2f}"] = ref_row(rels["pooled"], with_leg(real, lr / C.LEG_RATIO), C.PRIMARY, Q, n_boot,
                                              C.SEED + 810 + k)
    sub = real[real["person_id"] != FLAGGED].reset_index(drop=True)
    sens["without_HC134"] = ref_row(rels["pooled"], sub, C.PRIMARY, Q, n_boot, C.SEED + 820)
    base_h = refs["pooled"]["thigh_shank_foot"]["h"]
    res["PH10"] = {"rows": sens, "h_shift_per_2_percent_stature": {
        "minus": sens["stature_x0.98"]["h"] - base_h, "plus": sens["stature_x1.02"]["h"] - base_h},
        "HC134_stature_m": float(real.loc[real["person_id"] == FLAGGED, "stature_m"].iloc[0])
        if (real["person_id"] == FLAGGED).any() else None}
    res["table_treadmill_shares"] = table_treadmill_shares(am, row_w)
    if registered is not None:
        reg = {"pooled": registered["primary"]["point"]["gamma_L"],
               "treadmill": registered["secondary"]["relation_treadmill"]["point"]["gamma_L"],
               "overground": registered["secondary"]["relation_overground"]["point"]["gamma_L"],
               "KIT": registered["secondary"]["relation_KIT"]["point"]["gamma_L"]}
        res["reproduction"] = {k: {"registered": v, "here": refs[k]["thigh_shank_foot"]["gamma"],
                                   "abs_diff": abs(v - refs[k]["thigh_shank_foot"]["gamma"])} for k, v in reg.items()}
    res["readings"] = readings(res)
    return res


# ---------------------------------------------------------------- output
f, fiv = C.f, C.fiv


def markdown(res):
    rd = res["readings"]
    refs = res["PH7"]["references"]
    L = ["# CPL post hoc checks PH7 to PH11: reference, distal sites, the thigh, stature, stream (amendment 3)", "",
         "Dev diagnostic, not a gate read; post hoc: reports/bench/cpl/posthoc_plan_3.txt (SHA-256 "
         f"{res['plan_sha256']}), written after the review and hashed before this run. Written by "
         "scripts/cpl_posthoc3.py from the stride tables of the registered run; every number below is read from "
         f"results.json of this folder. MoveAtlas {res['moveatlas'].get('head', 'n/a')}. Q = {f(res['Q'], 6)}.", "",
         "Reproduction of the registered gamma_L (absolute difference): "
         + json.dumps({k: round(v["abs_diff"], 9) for k, v in res.get("reproduction", {}).items()}) + ".",
         "", f"## PH7 reference: **{rd['PH7']}**; stream row **{rd['PH7_stream']}**", "",
         "| reference | AMASS persons | site set | gamma | 95 % | h | 95 % | real persons | strides |", "|---" * 9 + "|"]
    for name in REFERENCES:
        for sname in SITE_SETS:
            r = refs[name][sname]
            L.append(f"| {name} | {refs[name]['persons_amass']} | {sname} | {f(r['gamma'])} | {fiv(r['interval'])} | "
                     f"{f(r['h'], 2)} | {fiv(r['h_interval'], 2)} | {r['persons']} | {r['strides']} |")
    L += ["", f"## PH8 distal sites: **{rd['PH8']}** ({rd['PH8_per_reference']})", "",
          f"## PH9 the thigh left out: {rd['PH9']}", "",
          "Single sites (gamma_s and kappa_s, from the thigh, shank and foot rows):", "",
          "| reference | gamma thigh | 95 % | shank | 95 % | ankle site | 95 % | foot | 95 % | kappa thigh | shank | "
          "ankle site | foot |", "|---" * 13 + "|"]
    for name in REFERENCES:
        sg = refs[name]["thigh_shank_foot"]["single"]
        L.append(f"| {name} | " + " | ".join(f"{f(sg[s]['gamma'])} | {fiv(sg[s]['interval'])}" for s in C.SEGMENTS)
                 + " | " + " | ".join(f(sg[s]["kappa"]) for s in C.SEGMENTS) + " |")
    for key in ("PH9_contributions_pooled", "PH9_contributions_stream"):
        c = res[key]
        L += ["", f"Contributions to log gamma_L ({key.rsplit('_', 1)[-1]} reference; gamma_L {f(c['gamma_L'])}, "
              f"exp of the summed site medians {f(c['exp_sum_of_terms'])}):", "",
              "| site | weight of b^2 / sigma^2 | term | factor | share of log gamma_L | part of h |", "|---" * 6 + "|"]
        for s, v in c["sites"].items():
            L.append(f"| {s} | {f(v['weight'])} | {f(v['term'], 4)} | {f(v['factor'])} | "
                     f"{f(v['share_of_log_gamma_L'], 2)} | {f(v['h_part'], 2)} |")
    p10 = res["PH10"]
    L += ["", "## PH10 stature (pooled reference, thigh, shank and foot)", "",
          "| row | gamma_L | 95 % | h | 95 % | persons |", "|---" * 6 + "|"]
    for k, r in p10["rows"].items():
        L.append(f"| {k} | {f(r['gamma'])} | {fiv(r['interval'])} | {f(r['h'], 2)} | {fiv(r['h_interval'], 2)} | "
                 f"{r['persons']} |")
    L += ["", f"h shift per 2 percent of stature: {f(p10['h_shift_per_2_percent_stature']['minus'], 3)} (x 0.98), "
          f"{f(p10['h_shift_per_2_percent_stature']['plus'], 3)} (x 1.02). HC134 stature in the table: "
          f"{f(p10['HC134_stature_m'], 4)} m.", "", "## PH11 stream shares", "",
          f"Train store (BodyStore.sample's rule): {res['PH11']}.", "",
          f"Straight forward stride table, treadmill share: {res['table_treadmill_shares']}.", "",
          f"Run {res['run_started']}, {res['timing_s']:.0f} s."]
    text = "\n".join(L)
    assert not any(ch in text for ch in (chr(0x2013), chr(0x2014)))
    return text


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boot", type=int, default=C.BOOT_N)
    a = ap.parse_args(argv)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    plan = C.check_prediction(PLAN, PLAN_SHA_FILE, PLAN_SHA256)
    import mdfm  # noqa: F401
    stamp = C._sibling("moveatlas_stamp").stamp()
    os.makedirs(a.out, exist_ok=True)
    train54, locked, _ = C.load_sets()
    real, am = load_tables(locked, train54)
    with open(os.path.join(REGISTERED, "results.json"), encoding="utf-8") as fh:
        registered = json.load(fh)
    Q = float(registered["size"]["Q"])
    chain = C.chain_gains()
    if abs(chain["train54_gain"] / chain["insim_gain"] - Q) > 1e-5:
        raise AssertionError("Q differs from the chain results")
    person_seconds, shares = store_metadata()
    print(f"real {len(real)} strides, AMASS {len(am)} strides; store metadata read", flush=True)
    res = {"what": LABEL, "task": "CPL", "plan_sha256": plan, "run_started": started, "moveatlas": stamp,
           "inputs": {k: {"path": C.rel(os.path.join(REGISTERED, k)),
                          "sha256": C.sha256_file(os.path.join(REGISTERED, k))}
                      for k in ("real_strides.csv", "amass_train_strides.csv", "results.json")},
           "boot": a.boot}
    res.update(analyse(real, am, person_seconds, Q, a.boot, registered))
    res["PH11"] = shares
    res["timing_s"] = time.time() - t0
    C.write_json(os.path.join(a.out, "results.json"), res)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(res) + "\n")
    C.write_json(os.path.join(a.out, "_provenance.json"), {
        "script": C.rel(os.path.abspath(__file__)), "script_sha256_at_start": SCRIPT_SHA256_AT_START,
        "script_sha256_at_write": C.sha256_file(os.path.abspath(__file__)),
        "cpl_coupling_sha256": C.sha256_file(os.path.join(HERE, "cpl_coupling.py")), "plan_sha256": plan,
        "moveatlas_stamp": stamp, "run_started": started, "label": LABEL,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                       "PYTHONDONTWRITEBYTECODE")}, "argv": sys.argv[1:]})
    print(json.dumps(res["readings"]), f"{res['timing_s']:.0f} s")


if __name__ == "__main__":
    main()
