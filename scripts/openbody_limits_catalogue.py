"""The symptom catalogue's poses against the anatomical joint-limit table of the open-body port (plan phase C, wave 1;
docs/OPEN_BODY_PORT_DESIGN.md section 5): which clips' local joint rotations fall outside the new table, by how
much, by family, beside the AMASS-percentile table they were tuned against.

Two steps:
- --build: builds every clip of catalogue.catalogue_set (raw bodies, background=False: the poses as the family
  modules make them, without the physiological layer's millimetre sway) from the source tree --tree (default this
  tree; the wave-1 measure uses an export of the commit before the port, so that the catalogue is the SMPL+H page's)
  and stores, per clip, the axis-angle components (deg) of the 21 local body rotations at every knot and, for a
  body with hands, the flexion and abduction components of the 30 hand joints (hands.hand_components) in a cache
  folder outside the tree (--cache, default <store>/openbody_limits_cache): program rotations only, no mesh, no
  shape and no model array. --parts n --part i builds one of n parts (resumable: a clip already stored is kept).
  The exam set holds no AMASS take since wave 7 (its stand, walk and freezing clips are procedural); --skip-amass
  skips nothing.
- --report: reads the cache and writes reports/openbody/catalogue_limits.json and catalogue_limits.md: per clip and
  per table (the anatomical table of reports/openbody/joint_limits.json, and the AMASS table of
  reports/body_store/priors.json for comparison) the components outside, their largest excess and the share of
  knots outside; the clips outside the anatomical table listed by family, the clip's level 0 beside it (a clip
  whose level 0 is already outside owes the excess to its base, not to its symptom); the hand joints outside the
  hand table (reports/openbody/joint_limits.json 'hands').
- the table each clip is judged on (wave 3): a clip whose truth names a table of limits_ext other than the healthy
  one (params limits_table: the shrugs of jerks.py on the 'shrug' table, whose shoulder adduction opens with the
  collar's rise) is read against that table too, and the report counts the clips outside the table they are judged
  on beside those outside the healthy anatomical table;
- the gait clips' own checks (wave 5): a clip whose truth carries gait.validity's record (the gait clips) keeps its
  scalars in the build log (gait ok, the stance slip, the deepest swing sole, the continuity, the leg-to-leg check's
  smallest hard and soft gaps and tolerances), and the report sums them (gait_checks);
- --bodies (wave 7): the cache also keeps each clip's raw body (its program rotations, translation and contact, and
  its hand rotations, float32), which scripts/openbody_compare.py --from-cache poses on both bodies (the comparison
  on every clip, without building the clips twice);
- --waves 2 (wave 2 and after): builds and reports only the clips of the modules the given waves own (WAVE), and
  --name names the report files (catalogue_limits_<name>.json and .md), so that a family wave measures its own
  clips on the current tree beside wave 1's report of the whole catalogue.

CPU only; no lockbox, EVAL, CODY or care_pd record is read; nothing is written inside the tree but the two report
files.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "openbody")
CACHE = "<store>/openbody_limits_cache"
TOL_DEG = 1e-4          # body.LIMIT_TOL_DEG is 1e-6; the cache stores float32, which reads a component held at a
#                         bound up to about 5e-6 deg past it (wave 3: the flapping elbows' pronation at 45)
# the clips built on AMASS takes, which needed the train body store: none since wave 7 (the exam set's stand, walk and
# freezing clips are procedural, exam_v10); --skip-amass is kept and skips nothing
AMASS = ()
# a clip id's prefix -> the port's wave of its module (docs/OPEN_BODY_PORT_DESIGN.md section 7)
WAVE = {"tremor": 2, "reach": 2, "isometric": 2, "standing": 2, "orthostatic": 2, "brady": 2, "invol": 3, "jerk": 3,
        "post": 4, "dystonia": 4, "gait": 5, "freezing": 5, "park": 6, "atax": 6, "ref": 7}


def _wave_of(clip_id):
    return WAVE.get(clip_id.split("_")[0])


def _git_head(path, safe=None):
    cmd = ["git"] + (["-c", f"safe.directory={safe}"] if safe else []) + ["-C", path, "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def build(a):
    tree = os.path.abspath(a.tree)
    sys.path.insert(0, tree)
    import torch
    from mdfm.body.rotations import matrix_to_axis_angle, rot6d_to_matrix
    from mdfm.synth import catalogue as CAT
    from mdfm.synth import hands as H
    torch.set_num_threads(a.threads)
    specs = CAT.catalogue_set(background=False)
    os.makedirs(a.cache, exist_ok=True)
    order = sorted(range(len(specs)), key=lambda i: specs[i].clip_id)
    if a.waves:
        order = [i for i in order if _wave_of(specs[i].clip_id) in a.waves]
    mine = [specs[i] for k, i in enumerate(order) if k % a.parts == a.part]
    log = os.path.join(a.cache, f"log_{a.part}of{a.parts}.jsonl")
    done = 0
    for s in mine:
        p = os.path.join(a.cache, f"{s.clip_id}.npz")
        if os.path.exists(p):
            continue
        if a.skip_amass and s.clip_id in AMASS:
            continue
        t0 = time.time()
        rec = {"clip_id": s.clip_id, "family": s.family, "segment": s.segment, "level0": s.level0,
               "grade": str(s.grade)}
        try:
            body, truth = s.build()
            lt = (truth.get("params") or {}).get("limits_table") if isinstance(truth, dict) else None
            if isinstance(lt, dict) and lt.get("table"):
                rec["limits_table"] = lt["table"]          # the table the clip is judged on (limits_ext), wave 3
            v = truth.get("validity") if isinstance(truth, dict) else None
            if isinstance(v, dict) and isinstance(v.get("gait"), dict) and isinstance(v.get("legs"), dict):
                lg = v["legs"]                             # wave 5: a gait clip's own checks, scalars only
                rec["gait_checks"] = {
                    "ok": bool(v["gait"].get("ok")), "continuity_ok": bool(v["gait"].get("continuity_ok")),
                    "stance_slip_p95_m_s": v.get("stance_slip_p95_m_s"), "swing_lowest_m": v.get("swing_lowest_m"),
                    "legs_ok": bool(lg.get("ok")), "hard_min_m": lg.get("hard_min_m"),
                    "soft_min_m": lg.get("soft_min_m"), "soft_tol_m": lg.get("soft_tol_m"), "tol_m": lg.get("tol_m"),
                    "distal_min_m": lg.get("distal_min_m"), "hard_where": lg.get("hard_where")}
            with torch.no_grad():
                comp = torch.rad2deg(matrix_to_axis_angle(rot6d_to_matrix(body.rot6d))).numpy()[:, 1:22]
            arrays = {"body_deg": comp.astype(np.float32)}
            if a.bodies:                                   # wave 7: the raw body for openbody_compare.py --from-cache
                arrays.update(rot6d=body.rot6d.numpy().astype(np.float32),
                              transl=body.transl.numpy().astype(np.float32),
                              contact=body.contact.numpy().astype(np.uint8), rate=np.float64(body.rate))
                if H.has_hands(body):
                    arrays["hand6d"] = body.hand6d.numpy().astype(np.float32)
            if H.has_hands(body):
                hc = H.hand_components(body.hand_R())
                joints = sorted(hc)
                arrays["hand_flex_deg"] = np.stack([hc[j][0] for j in joints], 1).astype(np.float32)
                arrays["hand_abd_deg"] = np.stack([hc[j][1] for j in joints], 1).astype(np.float32)
                arrays["hand_joints"] = np.asarray(joints, np.int64)
            np.savez_compressed(p, **arrays)
            rec.update(ok=True, knots=int(comp.shape[0]), build_s=round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        done += 1
        print(f"[{a.part}/{a.parts}] {s.clip_id} {rec.get('build_s', 'ERR')}", flush=True)
    meta = {"tree": tree.replace("\\", "/"), "mdfm_commit": a.commit or _git_head(tree),
            "moveatlas_commit": _git_head("<moveatlas>", "<moveatlas>"),
            "n_specs": len(specs), "bodies": bool(a.bodies), "built_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(a.cache, f"meta_{a.part}of{a.parts}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"part {a.part}/{a.parts}: {done} built", flush=True)


# ---------------------------------------------------------------- report
def _tables():
    with open(os.path.join(OUT, "joint_limits.json"), encoding="utf-8") as f:
        anat = json.load(f)
    with open(os.path.join(ROOT, "reports", "body_store", "priors.json"), encoding="utf-8") as f:
        jl = json.load(f)["joint_limits"]
    amass = {"lo": np.degrees(np.asarray(jl["lo"], np.float64)), "hi": np.degrees(np.asarray(jl["hi"], np.float64))}
    return anat, {"anatomical": {"lo": np.asarray(anat["body"]["lo_deg"]), "hi": np.asarray(anat["body"]["hi_deg"])},
                  "amass": amass}


def _excess(comp, lo, hi):
    """(T, 21, 3) deg outside [lo, hi] of rows 1 to 21 (<= 0 inside)."""
    return np.maximum(lo[None, 1:22] - comp, comp - hi[None, 1:22])


def _hand_excess(arrs, hands_tab):
    """[(joint name, excess deg)] of the hand joints outside the hand table, worst first."""
    out = []
    names = hands_tab["joint_names"]
    for j, fl, ab in zip(arrs["hand_joints"], arrs["hand_flex_deg"].T, arrs["hand_abd_deg"].T):
        row = hands_tab["joints"][names[str(int(j))]]
        e = max(row["flex_lo_deg"] - float(fl.min()), float(fl.max()) - row["flex_hi_deg"],
                float(np.abs(ab).max()) - row["abd_deg"])
        if e > TOL_DEG:
            out.append((names[str(int(j))], round(e, 2)))
    return sorted(out, key=lambda x: -x[1])


def _own_table(table, comp, names):
    """{'table', 'n_components', 'worst_deg', 'components'}: a clip judged on a table of limits_ext other than the
    healthy one (the shrug and dystonia tables: the shoulder's adduction bound opened by the collar's rise at each
    knot) read against that table (limits_ext.excess, the rise over rest), as the module's validity reads it."""
    sys.path.insert(0, ROOT)
    from mdfm.synth import limits_ext as LX
    full = np.concatenate([np.zeros((comp.shape[0], 1, 3)), comp], 1)
    ex = LX.excess(table, full)                              # (T, 21, 3)
    worst = ex.max(0)
    bad = [(names[j + 1] + "." + "xyz"[c], round(float(worst[j, c]), 2), round(float((ex[:, j, c] > TOL_DEG).mean()), 3))
           for j in range(21) for c in range(3) if worst[j, c] > TOL_DEG]
    bad.sort(key=lambda x: -x[1])
    return {"table": table, "n_components": len(bad), "worst_deg": bad[0][1] if bad else 0.0, "components": bad}


def report(a):
    anat, tabs = _tables()
    names = anat["body"]["joints"]
    logs = {}
    for p in sorted(glob.glob(os.path.join(a.cache, "log_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("ok") or not logs.get(r["clip_id"], {}).get("ok"):
                    logs[r["clip_id"]] = r      # the last record of a clip wins, a built one over an error (a rebuild)
    metas = []
    for p in sorted(glob.glob(os.path.join(a.cache, "meta_*.json"))):
        with open(p, encoding="utf-8") as f:
            metas.append(json.load(f))
    for m in metas:                          # the export's own folder is a scratch path: named by its commit only
        same = os.path.normcase(os.path.abspath(m["tree"])) == os.path.normcase(ROOT)
        m["tree"] = ("this tree" if same else
                     f"a checkout of MD_FM {m['mdfm_commit']} outside the tree (a git archive export or a worktree)")
    clips, errors = {}, {}
    for cid, r in sorted(logs.items()):
        if a.waves and _wave_of(cid) not in a.waves:
            continue
        if not r.get("ok"):
            errors[cid] = r.get("error", "").split(":")[0]
            continue
        arrs = np.load(os.path.join(a.cache, f"{cid}.npz"))
        comp = arrs["body_deg"].astype(np.float64)
        rec = {"family": r["family"], "segment": r["segment"], "level0": r["level0"], "knots": int(comp.shape[0])}
        for t, b in tabs.items():
            ex = _excess(comp, b["lo"], b["hi"])
            worst = ex.max(0)                                    # (21, 3)
            bad = [(names[j + 1] + "." + "xyz"[c], round(float(worst[j, c]), 2),
                    round(float((ex[:, j, c] > TOL_DEG).mean()), 3))
                   for j in range(21) for c in range(3) if worst[j, c] > TOL_DEG]
            bad.sort(key=lambda x: -x[1])
            rec[t] = {"n_components": len(bad), "worst_deg": bad[0][1] if bad else 0.0, "components": bad}
        if r.get("limits_table") and r["limits_table"] != "healthy":
            rec["own"] = _own_table(r["limits_table"], comp, names)
        if "hand_joints" in arrs.files:
            rec["hands_outside"] = _hand_excess(arrs, anat["hands"])
        if r.get("gait_checks"):
            rec["gait_checks"] = r["gait_checks"]                # wave 5: the gait clip's own checks
        clips[cid] = rec
    # by family: clips outside the anatomical table, and whether their level 0 is outside too
    fam = {}
    for cid, r in clips.items():
        f = fam.setdefault(r["family"], {"clips": 0, "outside_anatomical": 0, "outside_amass": 0,
                                         "outside_own_excess": 0, "clips_outside": []})
        f["clips"] += 1
        f["outside_amass"] += int(r["amass"]["n_components"] > 0)
        if r["anatomical"]["n_components"]:
            f["outside_anatomical"] += 1
            l0 = clips.get(r["level0"]) if r["level0"] else None
            base = {c[0]: c[1] for c in l0["anatomical"]["components"]} if l0 else {}
            own = [c for c in r["anatomical"]["components"] if c[1] > base.get(c[0], 0.0) + 0.5]
            f["outside_own_excess"] += int(bool(own))
            f["clips_outside"].append({"clip_id": cid, "worst_deg": r["anatomical"]["worst_deg"],
                                       "components": r["anatomical"]["components"][:6],
                                       "level0": r["level0"], "level0_outside": bool(base),
                                       "beyond_level0": [c[0] for c in own][:6]})
    for f in fam.values():
        f["clips_outside"].sort(key=lambda x: -x["worst_deg"])
    # by module (the clip id's prefix) and the port's wave that owns it (docs/OPEN_BODY_PORT_DESIGN.md section 7)
    mods = {}
    for cid, r in clips.items():
        pre = cid.split("_")[0]
        key = f"{pre} (wave {WAVE.get(pre, '?')})"
        m = mods.setdefault(key, {"clips": 0, "outside_anatomical": 0, "beyond_level0": 0, "outside_amass": 0,
                                  "components": {}})
        m["clips"] += 1
        m["outside_amass"] += int(r["amass"]["n_components"] > 0)
        if r["anatomical"]["n_components"]:
            m["outside_anatomical"] += 1
            row = next(c for c in fam[r["family"]]["clips_outside"] if c["clip_id"] == cid)
            m["beyond_level0"] += int(bool(row["beyond_level0"]) or not row["level0"])
            for name, e, _ in r["anatomical"]["components"]:
                m["components"][name] = max(m["components"].get(name, 0.0), e)
    for m in mods.values():
        m["components"] = dict(sorted(m["components"].items(), key=lambda kv: -kv[1]))
    # per component: how many clips leave the anatomical table there, and the largest excess
    comp_count = {}
    for cid, r in clips.items():
        for name, e, share in r["anatomical"]["components"]:
            c = comp_count.setdefault(name, {"clips": 0, "worst_deg": 0.0, "worst_clip": None})
            c["clips"] += 1
            if e > c["worst_deg"]:
                c["worst_deg"], c["worst_clip"] = e, cid
    hands_bad = {cid: r["hands_outside"] for cid, r in clips.items() if r.get("hands_outside")}
    out = {"script": "scripts/openbody_limits_catalogue.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "built": metas, "table": "reports/openbody/joint_limits.json", "table_sha256": hashlib.sha256(
               open(os.path.join(OUT, "joint_limits.json"), "rb").read()).hexdigest(),
           "amass_table": "reports/body_store/priors.json (joint_limits)", "waves": a.waves or None,
           "n_clips": len(clips),
           "n_errors": len(errors), "errors": errors,
           "n_outside_anatomical": sum(r["anatomical"]["n_components"] > 0 for r in clips.values()),
           "n_outside_amass": sum(r["amass"]["n_components"] > 0 for r in clips.values()),
           "n_hands_outside": len(hands_bad),
           "n_judged_on_own_table": sum("own" in r for r in clips.values()),
           "n_outside_own_table": sum(r.get("own", r["anatomical"])["n_components"] > 0 for r in clips.values()),
           "by_module": dict(sorted(mods.items())), "by_family": dict(sorted(fam.items())),
           "by_component": dict(sorted(comp_count.items(), key=lambda kv: -kv[1]["clips"])),
           "hands_outside": hands_bad, "clips": clips}
    stem = "catalogue_limits" + (f"_{a.name}" if a.name else "")
    with open(os.path.join(OUT, stem + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    title = "wave 1" if not a.waves else "the modules of wave " + ", ".join(str(w) for w in a.waves)
    lines = [f"# The catalogue's poses against the anatomical joint-limit table (open-body port, {title})", "",
             f"scripts/openbody_limits_catalogue.py, {out['run_at']}; built from {metas[0]['tree'] if metas else '?'} "
             f"(MD_FM {metas[0]['mdfm_commit'] if metas else '?'}, MoveAtlas "
             f"{metas[0]['moveatlas_commit'] if metas else '?'}), raw bodies (no background layer); table "
             f"reports/openbody/joint_limits.json (sha256 {out['table_sha256'][:8]}).", "",
             f"Clips read: {len(clips)} (errors {len(errors)}). Outside the anatomical table: "
             f"{out['n_outside_anatomical']}; outside the AMASS table: {out['n_outside_amass']}; hands outside the "
             f"hand table: {len(hands_bad)}. Judged on a table of limits_ext other than the healthy one (their truth's "
             f"limits_table: the shrug or dystonia coupling): {out['n_judged_on_own_table']}; outside the table each clip "
             f"is judged on: {out['n_outside_own_table']}. Not built: any clip in errors (since wave 7 the exam set holds no "
             f"AMASS take, owner decision C8).", "", "## By module and the wave that owns it", "",
             "| module (wave) | clips | outside anatomical | own excess (beyond the level 0, or a level 0) | outside AMASS "
             "| components (worst deg) |", "|---|---|---|---|---|---|"]
    for k, m in sorted(mods.items()):
        comps = ", ".join(f"{n} {e}" for n, e in list(m["components"].items())[:5])
        lines.append(f"| {k} | {m['clips']} | {m['outside_anatomical']} | {m['beyond_level0']} | {m['outside_amass']} "
                     f"| {comps} |")
    lines += ["", "## By family", "",
             "| family | clips | outside anatomical | beyond their level 0 | outside AMASS |", "|---|---|---|---|---|"]
    for k, f in sorted(fam.items()):
        lines.append(f"| {k} | {f['clips']} | {f['outside_anatomical']} | {f['outside_own_excess']} | "
                     f"{f['outside_amass']} |")
    lines += ["", "## By component (anatomical table)", "", "| component | clips | worst deg | worst clip |",
              "|---|---|---|---|"]
    for k, c in out["by_component"].items():
        lines.append(f"| {k} | {c['clips']} | {c['worst_deg']} | {c['worst_clip']} |")
    lines += ["", "## Clips outside, by family (worst six components, deg)", ""]
    for k, f in sorted(fam.items()):
        if not f["clips_outside"]:
            continue
        lines += [f"### {k}", ""]
        for c in f["clips_outside"]:
            comps = ", ".join(f"{n} {e}" for n, e, _ in c["components"])
            l0 = "" if not c["level0"] else (f"; level 0 {c['level0']} outside too" if c["level0_outside"] else
                                              f"; level 0 {c['level0']} inside")
            own = clips[c["clip_id"]].get("own")
            if own:
                l0 += (f"; judged on the {own['table']} table: " +
                       (", ".join(f"{n} {e}" for n, e, _ in own["components"][:6]) if own["components"] else "inside"))
            lines.append(f"- {c['clip_id']}: {comps}{l0}")
        lines.append("")
    if hands_bad:
        lines += ["## Hands outside the hand table", ""]
        for cid, b in sorted(hands_bad.items()):
            lines.append(f"- {cid}: " + ", ".join(f"{n} {e}" for n, e in b[:6]))
    gc = {cid: r["gait_checks"] for cid, r in clips.items() if r.get("gait_checks")}
    if gc:                                                     # wave 5: the gait clips' own checks (gait.validity)
        bad = sorted(cid for cid, g in gc.items() if not g["ok"])
        hard = min(gc.items(), key=lambda kv: kv[1]["hard_min_m"])
        soft = min(gc.items(), key=lambda kv: kv[1]["soft_min_m"])
        slip = max(gc.items(), key=lambda kv: kv[1]["stance_slip_p95_m_s"])
        low = min(gc.items(), key=lambda kv: kv[1]["swing_lowest_m"])
        lines += ["", "## The gait clips' own checks (gait.validity)", "",
                  f"Clips with the checks in their truth: {len(gc)}; passing (gait ok: the table, the stance slip, the "
                  f"swing sole, the continuity, the leg-to-leg check): {len(gc) - len(bad)}"
                  + (f"; failing: {', '.join(bad)}" if bad else "") + ".",
                  f"- legs, the smallest hard gap {1000 * hard[1]['hard_min_m']:.2f} mm ({hard[0]}, "
                  f"{hard[1]['hard_where']}; tolerance {1000 * hard[1]['tol_m']:.1f} mm), the smallest soft gap "
                  f"(the inner thighs) {1000 * soft[1]['soft_min_m']:.2f} mm ({soft[0]}; soft tolerance "
                  f"{1000 * soft[1]['soft_tol_m']:.1f} mm)",
                  f"- the largest stance slip (95th percentile) {1000 * slip[1]['stance_slip_p95_m_s']:.3f} mm/s "
                  f"({slip[0]}), the deepest swing sole {1000 * low[1]['swing_lowest_m']:.2f} mm ({low[0]})"]
        out["gait_checks"] = {"n": len(gc), "failing": bad, "hard_min_m": [hard[0], hard[1]["hard_min_m"]],
                              "soft_min_m": [soft[0], soft[1]["soft_min_m"]],
                              "slip_p95_max_m_s": [slip[0], slip[1]["stance_slip_p95_m_s"]],
                              "swing_lowest_m": [low[0], low[1]["swing_lowest_m"]]}
        with open(os.path.join(OUT, stem + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
    with open(os.path.join(OUT, stem + ".md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(clips)} clips, {out['n_outside_anatomical']} outside the anatomical table, "
          f"{out['n_outside_amass']} outside the AMASS table, {len(errors)} errors")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--commit", default=None, help="the MD_FM commit the --tree export was made from")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-amass", action="store_true")
    ap.add_argument("--bodies", action="store_true", help="also store each clip's raw body (scripts/openbody_compare.py "
                                                         "--from-cache, wave 7)")
    ap.add_argument("--waves", type=int, nargs="*", default=None, help="only the clips of these waves' modules")
    ap.add_argument("--name", default=None, help="the report files' suffix (catalogue_limits_<name>.json and .md)")
    a = ap.parse_args()
    if a.build:
        build(a)
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
