"""The page checks of the skeleton catalogue measured on a build of the catalogue set (docs/REMAINING_WORK_2026-11-10.md
section 1 item 5): the skate check's contacts split by kind, and the clips each check of
render_skeleton_catalogue.SKELETON_CHECKS refuses.

--build --data DIR [--part k --parts n]: builds the clips of part k of the catalogue set as the export does
(render_skeleton_catalogue.export_clip with the catalogue set's module checks and code record) into DIR/data, the
export's own data files (gzip JSON of the skeleton positions and the metadata; no vertex, face, shape coefficient or
model array, as the export). A data file whose code record matches the current code (stale_reason) is kept, so a
rerun after a change of the code rebuilds the clips that changed only. The clips the module checks refuse go to
DIR/refused_<k>of<n>.json. The parts are balanced on the build times of the background sweep
(reports/synth/background/sweep_*.json) when they are there.

--summary (--data DIR, or --data-from an export folder of the viewer root, read only): reads the data files, leaves
out the clips the page leaves out although their data exist (hidden_clips), runs the skeleton checks as the page does
(skeleton_checks) and reads every foot contact of the skate check (skate_contacts) by the rule in force (the sole
point that travels least, SKATE_MM) and by the named points alone (the rule before the sole cloud), each contact
classed: 'slide' (inside a declared slide of a dragged foot, read here although the check leaves it out), 'turning'
(the foot turns by TURN_YAW_DEG at least), 'straight' (the others); and apart the clips of the pivot set (PIVOT_SET:
the walks with their turns, the freezing clips built on the walk, the en-bloc turns). Writes
reports/synth/viewer/page_checks.json (the distributions, the clips each check refuses, the clips their refusal
takes with them, the code it read) and page_checks.md (the same in prose and tables).

Run from the tree whose code is measured; the data's code record (each file's module digests and the synth code
digest) goes into the report.
"""
import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_for_checks",
                                                  os.path.join(ROOT, "scripts", "render_skeleton_catalogue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_renderer()

OUT_JSON = os.path.join(ROOT, "reports", "synth", "viewer", "page_checks.json")
OUT_MD = os.path.join(ROOT, "reports", "synth", "viewer", "page_checks.md")
SWEEPS = os.path.join(ROOT, "reports", "synth", "background", "sweep_*of*.json")
PIVOT_SET = r"^(ref_walk_|gait_E\d+[a-z]*_|tremor_E8_walk|freezing_E8_|park_E\d+_.*(enbloc|turn))"
#   the clips whose contacts include the turns of a walk or a turn of the whole body: the reference walk, the
#   generated walks (each turns about at its middle), the walking tremor clips, the freezing clips built on the walk
#   and the en-bloc turns
QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)
READINGS = ("travel_mm", "named_travel_mm", "standing_travel_mm")
#   the readings of a contact (render_skeleton_catalogue._skate_read detail): the rule in force (the skin points of
#   the foot, its sole cloud among them, on the frames each one touches the floor: SKATE_TOUCH_MM), the named points
#   on their standing frames (the rule before the sole cloud), the skin points on their standing frames (the sole
#   cloud without SKATE_TOUCH_MM)


# ---------------------------------------------------------------- build
def part_plan(ids, parts, part):
    """The ids of part `part` of `parts`: greedy on the build times of the background sweep (the median time for a
    clip without one), the longest first, each to the part with the least time so far; in the set's order inside a
    part."""
    w = {}
    for p in glob.glob(SWEEPS):
        with open(p, encoding="utf-8") as f:
            for c, r in json.load(f).get("clips", {}).items():
                w[c] = float(r.get("raw_build_s") or 0.0) + float(r.get("layer_s") or 0.0)
    med = float(np.median(list(w.values()))) if w else 1.0
    load, owner = [0.0] * parts, {}
    for c in sorted(ids, key=lambda c: (-w.get(c, med), c)):
        k = int(np.argmin(load))
        owner[c] = k
        load[k] += w.get(c, med)
    return [c for c in ids if owner[c] == part]


def build(a):
    import torch
    torch.set_num_threads(a.threads)
    specs = R.set_specs("catalogue", list(R.DEFAULT_MODULES))
    by = {s.clip_id: s for s in specs}
    ids = part_plan([s.clip_id for s in specs], a.parts, a.part)
    if a.clips:
        ids = [c for c in ids if c in set(a.clips)]
    state = R.CodeState(specs, [])
    out = a.data.rstrip("/").replace("\\", "/")
    os.makedirs(out, exist_ok=True)
    rp = os.path.join(out, f"refused_{a.part}of{a.parts}.json")
    refused = {}
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            refused = json.load(f)
    with open(os.path.join(out, f"build_{a.part}of{a.parts}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"source_commit": a.source_commit or git_head(), "synth_code_sha256": state.synth,
                   "tree": ROOT.replace("\\", "/"), "clips": len(ids),
                   "started_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f,
                  indent=1)
    t0 = time.time()
    print(f"part {a.part} of {a.parts}: {len(ids)} clips", flush=True)
    for c in ids:
        s = by[c]
        p = R.data_path(out, c)
        code = state.of(s)
        if os.path.exists(p):
            why = R.stale_reason(R.read_code(p), code)
            if why is None:
                continue
            os.remove(p)
        old = refused.get(c)
        if (old is not None and old.get("code") == {k: code[k] for k in R.STALE_KEYS}
                and not any(str(x).startswith(R.BUILD_ERROR) for x in old["failed"])):
            continue                                       # refused by its module checks on this code: kept
        t1 = time.time()
        try:
            _, bad = R.export_clip(s, out, R.SET_CATALOGUE, module_checks=True, code=code)
        except Exception as e:  # noqa: BLE001 (a clip that raises is recorded, the run goes on)
            traceback.print_exc()
            bad = [f"{R.BUILD_ERROR}: {type(e).__name__}: {e}"[:400]]
        refused.pop(c, None)
        if bad:
            refused[c] = {"failed": bad, "code": {k: code[k] for k in R.STALE_KEYS}}
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(refused, f, indent=1, sort_keys=True)
        print(f"{c}: {'refused ' + str(bad) if bad else 'built'} in {time.time() - t1:.1f} s "
              f"(total {time.time() - t0:.0f} s)", flush=True)
    print("done", flush=True)


# ---------------------------------------------------------------- summary
def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), **{f"q{int(round(100 * q))}": round(float(np.quantile(x, q)), 2) for q in QUANTILES}}


def over(x, tol):
    return int((np.asarray(x, dtype=np.float64) > tol).sum())


def summary(a):
    sys.stdout.reconfigure(encoding="utf-8")
    src = (a.data_from or a.data).rstrip("/").replace("\\", "/")
    specs = R.set_specs("catalogue", list(R.DEFAULT_MODULES))
    metas, blobs = R.load_all(src, specs)
    hidden = R.hidden_clips(metas, blobs, specs)
    metas = [m for m in metas if m["clip_id"] not in hidden]
    blobs = {c: b for c, b in blobs.items() if c not in hidden}
    now = R.CodeState(specs, [])
    by_spec = {s.clip_id: s for s in specs}
    stale = sorted(m["clip_id"] for m in metas if R.stale_reason(m.get("code"), now.of(by_spec[m["clip_id"]])))
    module_refused = {}
    for p in sorted(glob.glob(os.path.join(src, "refused_*of*.json"))):
        with open(p, encoding="utf-8") as f:
            module_refused.update({c: r["failed"] for c, r in json.load(f).items()})
    if os.path.exists(os.path.join(src, R.REFUSED)):
        module_refused.update(R.read_refused(src))
    builds = {}
    for p in sorted(glob.glob(os.path.join(src, "build_*of*.json"))):
        with open(p, encoding="utf-8") as f:
            builds[os.path.basename(p)] = json.load(f)
    t0 = time.time()
    checks = R.skeleton_checks(metas, blobs)
    print(f"skeleton checks on {len(checks)} clips in {time.time() - t0:.0f} s", flush=True)
    refusing = [k for k in R.SKELETON_CHECKS if k not in R.SKELETON_REPORTED_ONLY]
    failed = R.checks_failed(checks)                          # every check, refusing or not
    fails = {c: [f"{R.SKELETON_FAIL}{k}" for k in v] for c, v in failed.items()}
    followers = R.refusal_followers(metas, specs, {**hidden, **fails})
    contacts = skate_contacts(metas, blobs)
    out = {"label": "SYNTHETIC", "data": src, "clips_read": len(checks), "hidden_before_checks": hidden,
           "module_refused": module_refused, "stale": stale,
           "code": {"git_head": git_head(), "synth_code_sha256": now.synth, "builds": builds,
                    "data_synth_code_sha256": sorted({(m.get("code") or {}).get("synth_code_sha256") or ""
                                                      for m in metas}),
                    "script": "scripts/skeleton_page_checks.py", "argv": sys.argv[1:]},
           "checks_refusing": refusing, "checks_reported_only": list(R.SKELETON_REPORTED_ONLY),
           "refused_by_check": {k: sorted(c for c, v in failed.items() if k in v) for k in R.SKELETON_CHECKS},
           "refused_followers": dict(sorted(followers.items())),
           "readings": {c: {k: checks[c][k] for k in v} for c, v in sorted(failed.items())},
           "skate": skate_distributions(contacts, checks),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=_plain)
        f.write("\n")
    with open(a.out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(out))
    if a.contacts_out:
        with open(a.contacts_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(contacts, f, default=_plain)
    print(json.dumps({k: len(v) for k, v in out["refused_by_check"].items()}), flush=True)
    print(f"wrote {a.out_json} and {a.out_md}")


def _plain(x):
    """A numpy scalar or array as plain JSON (json.dump's default)."""
    return x.item() if isinstance(x, np.generic) else np.asarray(x).tolist()


def git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a tree extracted without its git folder)
        return None


def skate_contacts(metas, blobs, geo=None, fps=R.FPS):
    """[{clip, pivot_set, exempt, side, span_s, frames, kind, yaw_deg, point, READINGS}] of every
    foot contact of the skate check on the shown clips (R.skate_contact_list, the declared slides read as 'slide'
    contacts), on the floor the page draws."""
    geo = R.contact_geometry() if geo is None else geo
    by = {m["clip_id"]: m for m in metas}
    roots = R.page_roots(metas)
    rows = []
    for m in metas:
        c = m["clip_id"]
        if m.get("supine"):
            continue
        floor_mm = 1000.0 * R.page_floor(m, by[roots[c]])[0]
        P = R.blob_positions(blobs[c]).astype(np.float64)
        for r in R.skate_contact_list(P, m["_names"], fps, floor_mm, geo, m.get("slides_s")):
            rows.append(dict(r, clip=c, pivot_set=bool(re.search(PIVOT_SET, c)),
                             exempt=bool(R.skate_exempt(c))))
    return rows


def skate_distributions(rows, checks, geo=None):
    """The skate contacts' readings (skate_contacts) by kind and by set: each reading of READINGS with its quantiles
    and its count over the tolerance (the rule in force over each contact's own tolerance 'tol_mm', the readings
    before it over SKATE_MM), and for the turning contacts the travel beyond SKATE_MM per radian of turn against the
    sole cloud's covering radius; the generated clips (not of SKATE_EXEMPT) apart from the captured ones."""
    geo = R.contact_geometry() if geo is None else geo
    out = {"rule": R.SKATE_RULE, "skate_mm": R.SKATE_MM, "turn_yaw_deg": R.TURN_YAW_DEG,
           "touch_mm": R.SKATE_TOUCH_MM, "cover_mm": round(float(geo["cover_mm"]), 2),
           "sole_samples": {sd: int(len(geo[f"foot_{sd}"]["cloud_w"])) for sd in R.SIDES}, "groups": {}}

    def group(sel):
        g = {"contacts": len(sel), "clips": len({r["clip"] for r in sel}),
             "yaw_deg": quantiles([r["yaw_deg"] for r in sel if r["yaw_deg"] is not None])}
        for k in READINGS:
            x = [r[k] for r in sel if r[k] is not None]
            g[k] = quantiles(x)
            g[k]["over"] = (int(sum(r["travel_mm"] > r["tol_mm"] for r in sel)) if k == "travel_mm" else
                            over(x, R.SKATE_MM))
        per = [(r["travel_mm"] - R.SKATE_MM) / np.radians(r["yaw_deg"]) for r in sel
               if r["yaw_deg"] is not None and r["yaw_deg"] >= R.TURN_YAW_DEG and r["travel_mm"] > R.SKATE_MM]
        g["excess_mm_per_rad_max"] = round(float(max(per)), 2) if per else None
        return g
    gen = [r for r in rows if not r["exempt"]]
    cap = [r for r in rows if r["exempt"]]
    for name, sel in (("generated", gen), ("captured", cap)):
        for kind in ("straight", "turning", "slide"):
            out["groups"][f"{name}/{kind}"] = group([r for r in sel if r["kind"] == kind])
        for ps in (True, False):
            for kind in ("straight", "turning"):
                key = f"{name}/{'pivot_set' if ps else 'other_clips'}/{kind}"
                out["groups"][key] = group([r for r in sel if r["kind"] == kind and r["pivot_set"] == ps])
    keys = ("clip", "side", "span_s", "kind", "yaw_deg", "point", "tol_mm") + READINGS
    live = [r for r in gen if r["kind"] != "slide"]
    out["over_contacts"] = [{k: r[k] for k in keys} for r in sorted(
        (r for r in live if r["travel_mm"] > r["tol_mm"]), key=lambda r: -r["travel_mm"])]
    out["turn_allowed_contacts"] = [{k: r[k] for k in keys} for r in sorted(
        (r for r in live if R.SKATE_MM < r["travel_mm"] <= r["tol_mm"]), key=lambda r: -r["travel_mm"])]
    out["slide_contacts"] = [{k: r[k] for k in keys} for r in rows if r["kind"] == "slide"]
    out["clips_over"] = sorted(c for c, r in checks.items() if (r.get("skate") or {}).get("ok") is False)
    return out


def _row(name, g):
    def q(x, k):
        return "" if x.get("n", 0) == 0 else f"{x[k]:g}"
    cells = [name, str(g["contacts"]), str(g["clips"]), q(g["yaw_deg"], "q100")]
    for k in READINGS:
        x = g[k]
        cells += [q(x, "q50"), q(x, "q95"), q(x, "q100"), str(x.get("over", 0))]
    return "| " + " | ".join(cells) + " |"


def _contact_row(r):
    def v(x):
        return "" if x is None else f"{x:g}"
    return (f"| {r['clip']} | {r['side']} | {r['span_s'][0]:g} to {r['span_s'][1]:g} | {r.get('kind', 'slide')} | "
            f"{v(r['yaw_deg'])} | {r['point']} | {v(r['tol_mm'])} | " + " | ".join(v(r[k]) for k in READINGS) + " |")


def markdown(d):
    s = d["skate"]
    L = ["# SYNTHETIC: the page checks of the skeleton catalogue", "",
         "Written by scripts/skeleton_page_checks.py --summary; do not edit by hand.", "",
         f"Data: {d['data']} ({d['clips_read']} clips read by the checks), built from commit(s) "
         f"{', '.join(sorted({str(b.get('source_commit')) for b in d['code']['builds'].values()})) or 'unknown'} "
         f"(synth code {', '.join(x[:12] for x in d['code']['data_synth_code_sha256'])}); checks read by the tree at "
         f"HEAD {d['code']['git_head']} (synth code {d['code']['synth_code_sha256'][:12]}, data files stale against "
         f"it: {len(d['stale'])}); {d['created_utc']}.", "",
         f"Checks refusing: {', '.join(d['checks_refusing']) or 'none'}; reported only: "
         f"{', '.join(d['checks_reported_only']) or 'none'}.", "",
         "## The skate check's contacts", "",
         f"Rule: {s['rule']}. Sole cloud: {s['sole_samples']['left']} and {s['sole_samples']['right']} samples (left, "
         f"right), covering radius {s['cover_mm']:g} mm; touching: within {s['touch_mm']:g} mm of the foot's lowest "
         "standing skin point.", "",
         "Travel in mm of a contact's reading, by three rules: 'rule' the rule in force (the skin points of the foot, "
         "its sole cloud among them, on the frames each one touches the floor), 'named' the rule before the sole "
         "cloud (the toe-block joint, the heel and lateral sole points and the toe tip, on their standing frames), "
         "'standing' the skin points on their standing frames (the sole cloud without the touching condition). 'over' "
         f"counts the contacts over their tolerance: each contact's own for the rule ({s['skate_mm']:g} mm plus the "
         f"covering radius times the turn), {s['skate_mm']:g} mm for the other two. A contact is 'turning' when the "
         f"foot turns by {s['turn_yaw_deg']:g} deg at least within it, 'slide' inside a declared slide of a dragged "
         "foot (read here, left out by the check). The pivot set: the walks, the walking tremor, the freezing clips "
         "built on the walk and the en-bloc turns; the captured clips (SKATE_EXEMPT) are read and not judged.", "",
         "| group | contacts | clips | yaw max deg | rule median | q95 | max | over | named median | q95 | max | over "
         "| standing median | q95 | max | over |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [_row(k, g) for k, g in s["groups"].items() if g["contacts"]]
    per = {k: g["excess_mm_per_rad_max"] for k, g in s["groups"].items()
           if g.get("excess_mm_per_rad_max") is not None and k.startswith("generated/") and k.count("/") == 1}
    if per:
        L += ["", f"Largest travel beyond {s['skate_mm']:g} mm per radian of turn (rule in force, the contacts that "
              f"turn by {s['turn_yaw_deg']:g} deg at least), against the "
              f"covering radius {s['cover_mm']:g} mm: " + ", ".join(f"{k} {v:g} mm" for k, v in per.items()) + "."]
    head = ["| clip | side | span s | kind | yaw deg | point | tol mm | rule mm | named mm | standing mm |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    L += ["", f"Generated contacts over their tolerance (refused): {len(s['over_contacts'])}.", ""]
    if s["over_contacts"]:
        L += head + [_contact_row(r) for r in s["over_contacts"]]
    L += ["", f"Generated contacts over {s['skate_mm']:g} mm that the turn's tolerance passes: "
              f"{len(s['turn_allowed_contacts'])}.", ""]
    if s["turn_allowed_contacts"]:
        L += head + [_contact_row(r) for r in s["turn_allowed_contacts"]]
    if s["slide_contacts"]:
        L += ["", "Contacts inside the declared slides of a dragged foot (left out by the check; their reading by the "
                  "rule against their tolerance):", ""]
        L += head + [_contact_row(dict(r, kind="slide")) for r in s["slide_contacts"]]
    L += ["", "## The clips each check refuses", ""]
    for k, v in d["refused_by_check"].items():
        L.append(f"- {k} ({len(v)}): {', '.join(v) if v else 'none'}")
    if d["refused_followers"]:
        L += ["", "Left out with them (a symptom clip whose level 0 is refused, a level 0 left without a shown "
                  "symptom clip):", ""]
        L += [f"- {c}: {', '.join(v)}" for c, v in d["refused_followers"].items()]
    if d["module_refused"]:
        L += ["", "Refused at build by the module checks (no data file):", ""]
        L += [f"- {c}: {', '.join(map(str, v))}" for c, v in sorted(d["module_refused"].items())]
    if d["hidden_before_checks"]:
        L += ["", "Left out by the page before the checks (hidden_clips):", ""]
        L += [f"- {c}: {', '.join(v)}" for c, v in sorted(d["hidden_before_checks"].items())]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data", default=f"{R.V.VIEWER_ROOT}/page_checks",
                    help="the folder the data files are built into and read from")
    ap.add_argument("--data-from", default=None, help="--summary: read an export folder instead (never written)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--clips", nargs="*", default=None, help="--build: only these clips of the part")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--source-commit", default=None,
                    help="--build: the commit the tree was extracted from (a tree without its git folder)")
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--contacts-out", default=None, help="--summary: also write every contact's reading there")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.build:
        build(a)
    if a.summary:
        summary(a)
    if not (a.build or a.summary):
        ap.error("--build or --summary")


if __name__ == "__main__":
    main()
