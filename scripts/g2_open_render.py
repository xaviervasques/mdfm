"""Gate G2 on the open body (plan docs/MODEL_OPEN_PORT_PLAN.md, section 2.3 and step M4): the rendered side of
scripts/g2_renderer_realism.py drawn from the open body store's walks instead of the AMASS KIT and BMLrub walks.

What changes against v1 is the body source only. The real side, the ALIGN and TEST halves, the population mounting
fit, the cadence matching, the residual spectrum, the features and the discriminator are the functions of
scripts/g2_renderer_realism.py, imported and called in the order of its main() with its seeded generator (version
4 of G2); the v1 script and its cache (cache/g2_rendered) are untouched.

Walks (WALK_RULES, a stated rule on the store's source and take names; the store holds train folds only, never a
dev, lockbox or EVAL person, and this script asserts it again against reports/lockbox/lockbox_persons.csv and the
dev rows of mdfm.data.sources.folds()): the takes whose key names plain level walking:
- ceti_age_kinematics: tasks w01 (normal pace) and w02 (hurried), a 3.55 m path walked out and back five times
  (w03 and w04 carry a backpack or a crate);
- stroke_gait_vancriekinge: the able-bodied gait_ trials (the store holds no stroke patient);
- gait120: level_walking_trial; lencioni_gait_stairs: walking_ (not toe or heel walking);
- style100: Neutral_FW (the neutral style's forward walk; the other 99 styles are stylized gaits).
Not taken: cmu_mocap (its take names are numbers; no description of the takes in the store or the MoveAtlas raw
folder), moore_perturbed_gait and push_recovery_perturbations (perturbed walking), slopes, stairs, transfers,
dance, tai chi and the non-cyclic tasks. Takes whose fit residual rms_m exceeds the store's bound (16.7 mm) are
excluded. A take renders when, as in v1, the times from 0.5 s to its end minus 0.5 s hold at least one 2 s window
(WIN samples at 100 Hz); the counts say how many short walkway passes (gait120, lencioni) fall below it.

Rendering, per take: the stored rot6d and root translation (belt frame for a treadmill take, none here), low-passed
at G2_BODY_CUTOFF_HZ like v1's amass_to_body_clock (4th-order zero-phase Butterworth, mdfm.data.prep.lowpass, on
the 100 Hz knots: the store's fit already ran on 12 Hz landmarks), the rest joints of the take's phenotype through
the P3 basis (OpenBodyStore.rest), a BodyTrajectory with those rest joints (no SMPL+H body), the reference renderer
mdfm.factory.imu_render.render at 100 Hz over the 13 Xsens sites (site offsets of imu_render.site_table on the open
rest joints; the xsens13 layout of reports/open_layouts/layouts.json), the device model of g2.device_model()
(reports/g2_renderer/device_noise.json), the generator seeded by the CRC-32 of the take id, and v1's steady-walking
mask (g2.steady_mask), and a straight-walking mask (straight_mask): v1's AMASS walks were straight, the real side
keeps only the Walk phases (its Turn phases are labelled apart), but the ceti walks turn round five times per take
and style100's walk follows curves, so a frame counts only where the walking direction (the pelvis's horizontal
displacement over 2 s) turns by less than STRAIGHT_DEG_S in 1 s and stays so within STRAIGHT_MARGIN_S on both sides. Cached in cache/g2_rendered_open/<tag>.npz
with the tag of g2._mount_tag, the take ids and MASK_RULE, re-rendered when either differs.

Statistics: as g2_renderer_realism.main() (balanced accuracy of the discriminator, without the mounting fit, with
the residual spectrum, with the real side low-passed at 12 Hz, per site), plus one diagnostic: the renders mounted
with the xsens13 R0 of reports/open_mount_priors/priors.json when that file exists (fitted on every prior person, the
TEST half included, so not the G2 protocol).

Writes reports/g2_open/: takes.csv, results.json, per_site.csv, mounting.json, mounting_sign.csv, summary.md,
_provenance.json. Run (about 20 minutes on the CPU, renders included):
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/g2_open_render.py
    [--render-only] [--limit N]
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import zlib

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "g2_open")
CACHE_NAME = "g2_rendered_open"
RMS_BOUND_M = 0.0167
WALK_RULES = {
    "ceti_age_kinematics": r"_w0[12]_",
    "stroke_gait_vancriekinge": r"^gait_",
    "gait120": r"^level_walking_trial\d+$",
    "lencioni_gait_stairs": r"^walking_\d+$",
    "style100": r"^Neutral_FW$",
}
STRAIGHT_DEG_S = 15.0        # [cfg] the walking direction turns by less than this in 1 s on a straight frame
STRAIGHT_BASE_S = 1.0        # [cfg] the walking direction: the pelvis's displacement over 2 x this span
STRAIGHT_MARGIN_S = 0.5      # [cfg] and stays straight this long on both sides
MASK_RULE = f"steady and straight, ends clipped ({STRAIGHT_DEG_S:g} deg in 1 s, base {STRAIGHT_BASE_S:g} s, margin {STRAIGHT_MARGIN_S:g} s)"
V1_RESULTS = os.path.join(TREE, "reports", "g2_renderer", "results.json")
OPEN_PRIORS = os.path.join(TREE, "reports", "open_mount_priors", "priors.json")


def g2():
    import g2_renderer_realism as m
    return m


def open_store():
    from mdfm.data.open_body_store import OpenBodyStore
    return OpenBodyStore()


def lockbox_ids():
    with open(os.path.join(TREE, "reports", "lockbox", "lockbox_persons.csv"), encoding="utf-8", newline="") as f:
        return {r["person_id"] for r in csv.DictReader(f)}


def dev_ids():
    from mdfm.data import sources
    f = sources.folds()
    return set(f.loc[f["fold"] == "dev", "person_id"])


def walk_takes(store, limit=None):
    """The walking takes of the store by WALK_RULES, under the residual bound: list of dicts (index, take_id,
    source, person, key, n_knots, rms_m, renderable), in store order. Asserts no lockbox or dev person."""
    G = g2()
    out = []
    for i, t in enumerate(store.takes):
        pat = WALK_RULES.get(t["source"])
        if pat is None or not re.search(pat, t["key"]):
            continue
        n = t["n_knots"]
        span = np.arange(0.5, (n - 1) / G.RATE - 0.5, 1 / G.RATE).size
        out.append({"index": i, "take_id": t["take_id"], "source": t["source"], "person_id": t["person_id"],
                    "subject_ns": t["subject_ns"], "person": f"open:{t['source']}:{t['person_id']}", "key": t["key"],
                    "n_knots": n, "rms_m": t["rms_m"], "over_bound": t["rms_m"] > RMS_BOUND_M,
                    "renderable": t["rms_m"] <= RMS_BOUND_M and span >= G.WIN})
    ids = {r["person_id"] for r in out} | {r["subject_ns"] for r in out}
    bad = ids & (lockbox_ids() | dev_ids())
    if bad:
        raise ValueError(f"lockbox or dev persons among the open walks: {sorted(bad)[:5]}")
    if limit:
        keep = [r for r in out if r["renderable"]][:limit]
        out = [r for r in out if not r["renderable"]] + keep
        out.sort(key=lambda r: r["index"])
    return out


def trajectory(store, i, rest_cache):
    """BodyTrajectory of take i on the open body (rest joints of the take's phenotype, P3 basis), low-passed as v1."""
    import torch
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.prep import lowpass
    G = g2()
    tk = store.take(i)
    key = tuple(store.phenotype(i))
    if key not in rest_cache:
        rest_cache[key] = torch.as_tensor(store.rest(i), dtype=torch.float64)
    r6 = lowpass(np.asarray(tk["rot6d"], dtype=np.float64), G.RATE, G.BODY_CUTOFF_HZ)
    tr = lowpass(np.asarray(tk["transl"], dtype=np.float64), G.RATE, G.BODY_CUTOFF_HZ)
    return BodyTrajectory(None, torch.as_tensor(np.ascontiguousarray(r6)), torch.as_tensor(np.ascontiguousarray(tr)),
                          None, rate=G.RATE, rest=rest_cache[key])


def straight_mask(traj, t):
    """Frames at times t on a straight path (module docstring): the walking direction d(t), the pelvis's horizontal
    displacement from t - STRAIGHT_BASE_S to t + STRAIGHT_BASE_S (a span of two strides, so the steps' sway cancels;
    clipped to the take near its ends), turns by less than STRAIGHT_DEG_S between t - 0.5 s and t + 0.5 s (clipped
    likewise), and the frames within STRAIGHT_MARGIN_S on both sides inside the take do too."""
    import torch
    from scipy.ndimage import binary_erosion
    G = g2()
    with torch.no_grad():
        pos, _ = traj.pose_at(t)
    p = pos[:, 0].numpy()[:, [0, 2]]
    n, b, h = len(p), int(round(STRAIGHT_BASE_S * G.RATE)), int(round(0.5 * G.RATE))
    k = np.arange(n)
    d = p[np.minimum(k + b, n - 1)] - p[np.maximum(k - b, 0)]
    ang = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    turn = np.degrees(np.abs(ang[np.minimum(k + h, n - 1)] - ang[np.maximum(k - h, 0)]))
    m = int(round(STRAIGHT_MARGIN_S * G.RATE))
    return binary_erosion(turn < STRAIGHT_DEG_S, structure=np.ones(2 * m + 1, bool), border_value=1)


def rendered_takes_open(mount, store=None, takes=None, log=print):
    """Rendered open walks (list of dicts person, take, x (T, 13, 6) with NaN off the steady frames, cadence), as
    g2.rendered_takes but from the open body store; cached per mount tag and take list."""
    import torch
    from mdfm import paths
    G = g2()
    store = store or open_store()
    takes = walk_takes(store) if takes is None else takes
    rows = [r for r in takes if r["renderable"]]
    ids = [r["take_id"] for r in rows]
    cpath = os.path.join(paths.cache_dir(CACHE_NAME), f"{G._mount_tag(mount)}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=False)
        if list(z["take_ids"]) == ids and "mask_rule" in z.files and str(z["mask_rule"]) == MASK_RULE:
            return [{"person": r["person"], "take": r["take_id"], "x": z[f"x{k}"],
                     "cadence": G.cadence_from_lower_back(z[f"x{k}"])} for k, r in enumerate(rows)]
        log(f"{cpath}: take list or mask rule differs, rendering again")
    dev, _ = G.device_model()
    sites = [G.SITES.index(s) for s in G.SITES13]
    out, saved, rest_cache, t0 = [], {"take_ids": np.array(ids), "mask_rule": np.array(MASK_RULE)}, {}, time.time()
    for k, r in enumerate(rows):
        traj = trajectory(store, r["index"], rest_cache)
        t = torch.arange(0.5, (r["n_knots"] - 1) / G.RATE - 0.5, 1 / G.RATE, dtype=torch.float64)
        gen = torch.Generator().manual_seed(zlib.crc32(r["take_id"].encode()))
        with torch.no_grad():
            sig, _, _ = G.render(traj, t, sites=G.SITES13, mount=mount, device=dev, generator=gen)
        x = sig[:, sites, :6].numpy().copy()
        x[~(G.steady_mask(traj, t) & straight_mask(traj, t))] = np.nan
        out.append({"person": r["person"], "take": r["take_id"], "x": x, "cadence": G.cadence_from_lower_back(x)})
        saved[f"x{k}"] = x
        if (k + 1) % 50 == 0:
            log(f"  rendered {k + 1} of {len(rows)} ({time.time() - t0:.0f} s)")
    np.savez(cpath, **saved)
    return out


def window_count(x):
    return sum(1 for _ in g2().windows(x))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    p.add_argument("--out", default=OUT)
    p.add_argument("--render-only", action="store_true", help="the unmounted renders only (cache), no statistics")
    p.add_argument("--limit", type=int, default=None, help="render only the first N renderable takes (smoke)")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(a.moveatlas, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {a.moveatlas} (no INDEX.md)")
    os.environ["MOVEATLAS_ROOT"] = a.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    return a


def main():
    a = parse_args()
    import torch
    from _common import moveatlas_head
    G = g2()
    t0 = time.time()
    store = open_store()
    takes = walk_takes(store, a.limit)
    rend0 = rendered_takes_open(None, store, takes)
    print(f"open walks {len(takes)}, rendered {len(rend0)} ({time.time() - t0:.0f} s)", flush=True)
    if a.render_only:
        return
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------ g2_renderer_realism.main(), on the open renders
    G.rng = np.random.default_rng(20260928)
    rng = G.rng
    real = G.real_records()
    for r in real:
        r["x"] = G.to_sites13(r["signal"], r["loc"])
        r["cadence"] = G.cadence_from_lower_back(r["x"])
    shank = np.concatenate([r["x"][:, G.SITES13.index("shank_L"), 3:6] for r in real])
    peak = float(np.nanpercentile(np.abs(shank), 99))
    gyro_in_rad = peak < 20.0
    if gyro_in_rad:
        for r in real:
            r["x"][:, :, 3:6] = np.degrees(r["x"][:, :, 3:6])
    persons = sorted({r["person"] for r in real})
    perm = rng.permutation(len(persons))
    align_p = {persons[i] for i in perm[: len(persons) // 2]}
    real_align = [r for r in real if r["person"] in align_p]
    real_test = [r for r in real if r["person"] not in align_p]
    sign_report = []
    mount, resid = G.fit_mounting([r["x"] for r in real_align], [r["x"] for r in rend0], report=sign_report)
    rend = rendered_takes_open(mount, store, takes)
    bins = np.arange(60, 160, 5)
    hr, _ = np.histogram([r["cadence"] for r in real_test], bins)
    which = np.digitize(np.array([r["cadence"] for r in rend]), bins) - 1
    chosen = []
    for b, n_real in enumerate(hr):
        pool = [i for i in range(len(rend)) if which[i] == b]
        k = int(round(n_real / max(hr.sum(), 1) * len(rend)))
        if pool and k:
            chosen += list(rng.choice(pool, k, replace=len(pool) < k))
    rend_matched = [rend[i] for i in chosen]
    print(f"matched rendered takes {len(rend_matched)} (unique {len(set(chosen))}) ({time.time() - t0:.0f} s)",
          flush=True)
    d = G.fit_residual([r["x"] for r in real_align], [r["x"] for r in rend])
    rend_res = [dict(r, x=G.apply_residual(r["x"], d, zlib.crc32(r["take"].encode())).astype(np.float32))
                for r in rend]
    res = {"all_sites_with_residual": G.discriminate(real_test, [rend_res[i] for i in chosen]),
           "all_sites": G.discriminate(real_test, rend_matched)}
    print("all sites", res["all_sites"], f"({time.time() - t0:.0f} s)", flush=True)
    from mdfm.data.prep import lowpass
    real_test_lp = [dict(r, x=np.where(np.isfinite(r["x"]), lowpass(np.nan_to_num(r["x"]), G.RATE, 12.0),
                                       np.nan).astype(np.float32)) for r in real_test]
    res["all_sites_real_lowpassed_12hz"] = G.discriminate(real_test_lp, rend_matched)
    res["without_mounting_fit"] = G.discriminate(real_test, [rend0[i] for i in chosen])
    per_site = []
    for s, name in enumerate(G.SITES13):
        r = G.discriminate(real_test, rend_matched, sites=[s])
        per_site.append({"site": name, **r, "mounting_residual_deg": resid[name]})
        print(name, r["balanced_accuracy"], f"({time.time() - t0:.0f} s)", flush=True)
    res["cadence_real_test_quartiles"] = np.percentile([r["cadence"] for r in real_test], [25, 50, 75]).round(1).tolist()
    res["cadence_rendered_matched_quartiles"] = np.percentile([r["cadence"] for r in rend_matched],
                                                             [25, 50, 75]).round(1).tolist()
    ba = res["all_sites"]["balanced_accuracy"]
    res["verdict"] = "pass" if ba <= 0.70 else ("kill" if ba > 0.85 else "undetermined")
    res["real_gyro_p99_shank"] = round(peak, 2)
    res["real_gyro_converted_from_rad"] = gyro_in_rad
    # diagnostic: the renders mounted with the open mount priors' xsens13 R0 (every prior person, TEST included)
    if os.path.exists(OPEN_PRIORS):
        with open(OPEN_PRIORS, encoding="utf-8") as f:
            pri = json.load(f)
        fam = pri["families"]["xsens13"]
        m_pri = {s: torch.tensor(fam["R0"][pri["sites"].index(s)], dtype=torch.float64) for s in G.SITES13}
        rend_pri = rendered_takes_open(m_pri, store, takes)
        res["diagnostic_open_mount_priors_R0"] = G.discriminate(real_test, [rend_pri[i] for i in chosen])
        res["diagnostic_open_mount_priors_sha256"] = file_sha256(OPEN_PRIORS)

    # ------------------------------------------------ outputs
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(a.out, "mounting.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: v.numpy().round(6).tolist() for k, v in mount.items()} | {"residual_deg": resid}, f, indent=1)
    write_csv(os.path.join(a.out, "mounting_sign.csv"), sign_report)
    write_csv(os.path.join(a.out, "per_site.csv"),
              [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in per_site])
    rend_by = {r["take"]: r for r in rend0}
    take_rows = []
    for r in takes:
        x = rend_by.get(r["take_id"], {}).get("x")
        take_rows.append({"take_id": r["take_id"], "source": r["source"], "person": r["person"], "key": r["key"],
                          "n_knots": r["n_knots"], "rms_m": round(r["rms_m"], 5), "over_bound": int(r["over_bound"]),
                          "rendered": int(x is not None),
                          "steady_s": round(float(np.isfinite(x).all((1, 2)).sum()) / G.RATE, 2) if x is not None else 0.0,
                          "windows": window_count(x) if x is not None else 0})
    write_csv(os.path.join(a.out, "takes.csv"), take_rows)
    counts = count_table(take_rows)
    head = moveatlas_head(a.moveatlas)
    prov = {"script": "scripts/g2_open_render.py", "moveatlas_root": a.moveatlas, "moveatlas_head": head,
            "mdfm_head": git_head(), "store_root": store.root,
            "store_takes_sha256": {k: v for k, v in store.hashes["files"].items() if k.endswith("takes.csv")},
            "cache": f"cache/{CACHE_NAME}", "walk_rules": WALK_RULES, "rms_bound_m": RMS_BOUND_M,
            "limit": a.limit, "body_cutoff_hz": G.BODY_CUTOFF_HZ, "device_model": G.device_model()[1],
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "runtime_s": round(time.time() - t0, 1),
            "outputs": ["_provenance.json", "mounting.json", "mounting_sign.csv", "per_site.csv", "results.json",
                        "summary.md", "takes.csv"]}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    write_summary(a.out, res, per_site, counts, prov)


def file_sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    import subprocess
    r = subprocess.run(["git", "-C", TREE, "log", "-1", "--format=%h"], capture_output=True, text=True)
    return r.stdout.strip()


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def count_table(take_rows):
    """Per source: takes named by the rule, over the bound, rendered, persons rendered, hours of the rendered takes
    (knots / 100 Hz), steady hours, windows."""
    out = {}
    for r in take_rows:
        c = out.setdefault(r["source"], {"takes": 0, "over_bound": 0, "rendered": 0, "persons": set(), "hours": 0.0,
                                         "steady_hours": 0.0, "windows": 0})
        c["takes"] += 1
        c["over_bound"] += r["over_bound"]
        if r["rendered"]:
            c["rendered"] += 1
            c["persons"].add(r["person"])
            c["hours"] += r["n_knots"] / 100.0 / 3600.0
            c["steady_hours"] += r["steady_s"] / 3600.0
            c["windows"] += r["windows"]
    return out


def write_summary(out, res, per_site, counts, prov):
    with open(V1_RESULTS, encoding="utf-8") as f:
        v1 = json.load(f)
    tot = {k: sum(c[k] for c in counts.values()) for k in ("takes", "over_bound", "rendered", "windows")}
    persons = set().union(*[c["persons"] for c in counts.values()])
    hours = sum(c["hours"] for c in counts.values())
    steady = sum(c["steady_hours"] for c in counts.values())
    L = ["# G2 renderer realism on the open body", "",
         f"Measured by scripts/g2_open_render.py (method in its docstring), MD_FM {prov['mdfm_head']}, MoveAtlas "
         f"{prov['moveatlas_head'].split(' ')[0]}, the open body store {prov['store_root']}; the v1 numbers are "
         "reports/g2_renderer/results.json of scripts/g2_renderer_realism.py (AMASS KIT and BMLrub walks on SMPL+H)."
         + (f" Reduced run: the first {prov['limit']} renderable takes only." if prov["limit"] else ""), "",
         "## The open walks", "",
         "| source | takes named by the rule | over the residual bound | rendered (2 s window after the margins) | "
         "persons rendered | hours rendered | steady straight hours | 2 s windows (1 s hop) |", "|---|---|---|---|---|---|---|---|"]
    for s, c in sorted(counts.items()):
        L.append(f"| {s} | {c['takes']} | {c['over_bound']} | {c['rendered']} | {len(c['persons'])} | "
                 f"{c['hours']:.3f} | {c['steady_hours']:.3f} | {c['windows']} |")
    L.append(f"| all | {tot['takes']} | {tot['over_bound']} | {tot['rendered']} | {len(persons)} | {hours:.3f} | "
             f"{steady:.3f} | {tot['windows']} |")
    L += ["", "## The discriminator, open against v1", "",
          "| test | open: balanced accuracy | open: 95 percent interval | open: windows real / rendered, persons "
          "real / rendered | v1: balanced accuracy | v1: interval |", "|---|---|---|---|---|---|"]
    for key in ("all_sites", "without_mounting_fit", "all_sites_with_residual", "all_sites_real_lowpassed_12hz",
                "diagnostic_open_mount_priors_R0"):
        if key not in res:
            continue
        o, w = res[key], v1.get(key)
        L.append(f"| {key} | {o['balanced_accuracy']} | {o['ci95']} | {o['windows_real']} / {o['windows_rendered']}, "
                 f"{o['persons_real']} / {o['persons_rendered']} | {w['balanced_accuracy'] if w else ''} | "
                 f"{w['ci95'] if w else ''} |")
    L += ["", f"Verdict on the G2 lines (pass at most 0.70, kill above 0.85), all sites without the residual "
          f"spectrum: open {res['verdict']}, v1 {v1['verdict']}. Cadence quartiles (steps/min): real TEST "
          f"{res['cadence_real_test_quartiles']}, open rendered matched {res['cadence_rendered_matched_quartiles']}, "
          f"v1 rendered matched {v1['cadence_rendered_matched_quartiles']}."]
    if "diagnostic_open_mount_priors_R0" in res:
        L.append("The diagnostic line mounts the renders with the xsens13 R0 of reports/open_mount_priors/priors.json "
                 f"(SHA-256 {res['diagnostic_open_mount_priors_sha256']}), fitted on every prior person, the TEST half "
                 "included: not the G2 protocol.")
    L += ["", "| site | open balanced accuracy | open mounting residual (deg) |", "|---|---|---|"]
    L += [f"| {r['site']} | {r['balanced_accuracy']} | {r['mounting_residual_deg']} |" for r in per_site]
    text = "\n".join(L) + "\n"
    assert chr(0x2014) not in text
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
