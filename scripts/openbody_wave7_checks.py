"""The open-body port's wave 7 readings that the design's section 18 cites (docs/OPEN_BODY_PORT_DESIGN.md): the
background layer's mesh caches keyed on the body model, the layered clips' scene and table checks read on Anny, the
exam set's references and freezing clips without AMASS, the SP3 stimuli rebuilt on the open body, and the export's
landmarks re-measured on Anny.

Sections (all by default, or those named with --only; each section's commit and time in 'measured'):
- physio_cache: in one process, physio's probe vertex sets, faces and body surface read on the open body, then on the
  SMPL+H reference behind its flag, then on the open body again: each read's largest vertex index against its mesh's
  size, and whether the open body's entry came back unchanged (before wave 7 the cache held one entry for both);
- background: the layered clips whose checks are read again on the layered body (physio.layer_checks_of): the writing
  level 0 and the writer's cramp (postures.table_report_of), the geste antagoniste and the dystonic PKD clip
  (postures.scene_checks_of), the arising's level 0 (motions.arise_checks_of): each check's verdict and readings, and
  the layer's own verdict;
- exam_refs: the exam set's references (exam.reference_stand, exam.reference_walk: the joint-limit table, the walk's
  gait.validity and read-back, its contexts), the freezing clips' walk (exam.freezing_walk) and its turn rule (the
  walk of FOG_WALK with its turn over 4, 5 and 6 s: legcontact.leg_gap's smallest hard and soft gaps, the contexts),
  each exam freezing clip as built (its episodes' supports, heel lifts and sole gaps at the onset and over the holds,
  the hold background's verdict and span, the clip's validity entries that fail), and the same freezing plans on the
  healthy walk for comparison (the least miss of support 'mesh' there);
- stimuli: every SP3 probe stimulus (stimuli.probe_set) read back on its own body (stimuli.read_back) beside the value
  the operator injected (stimuli.truth_value);
- landmarks: the head landmarks (skeleton.head_landmarks) and the catalogue's drawing landmarks
  (scripts/render_skeleton_catalogue.landmark_offsets) on the open body, beside the SMPL+H reference's (a child
  process behind the reference's flag; offsets only);
- page (named with --only, after the page's build): the catalogue page's manifest (<store>/synth_viewer/skeleton_v2,
  render_skeleton_catalogue.py --set catalogue --no-record: the tree's run record of the SMPL+H page is never
  rewritten), its data files' export formats and their provenance.

Output: reports/openbody/wave7_checks.json (numbers only; no mesh, shape or model array). CPU; procedural bodies on
the catalogue's one body; no store, lockbox, EVAL, CODY or care_pd record is read.

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/openbody_wave7_checks.py [--only s]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "wave7_checks.json")

import numpy as np  # noqa: E402
import torch  # noqa: E402

BACKGROUND_CLIPS = ("post_W_writing_level0", "post_W_writers_cramp_g2", "post_E7_cd_geste_C1", "post_E7_pkd_dystonic",
                    "ref_arise_E7")
FOG_TURNS_S = (4.0, 5.0, 6.0)
PAGE_DIR = "<store>/synth_viewer/skeleton_v2"      # render_skeleton_catalogue.py --set catalogue's folder


def _head():
    r = subprocess.run(["git", "-c", f"safe.directory={ROOT.replace(os.sep, '/')}", "-C", ROOT, "rev-parse", "--short",
                        "HEAD"], capture_output=True, text=True)        # a worktree on another disk: its own entry
    return r.stdout.strip() or None


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k != "arrays"}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"array {x.shape}"
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().numpy())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _failed(validity):
    return sorted(k for k, v in (validity or {}).items() if isinstance(v, dict) and v.get("ok") is False)


# ---------------------------------------------------------------- physio's mesh caches
def physio_cache():
    from mdfm.synth import openbody as OB
    from mdfm.synth import physio as P
    from mdfm.synth import viewer as V
    out = []
    first = None
    for step, ref in (("open", False), ("reference", True), ("open again", False)):
        if ref:
            os.environ[OB.BACKEND_ENV], os.environ[OB.REFERENCE_ENV] = OB.SMPLH_REFERENCE, "1"
        else:
            os.environ.pop(OB.BACKEND_ENV, None)
        ids = P._probe_ids()
        n = len(V.rest_vertices())
        Fb, _, inc = P._body_surface()
        row = {"read": step, "model": P._model(), "mesh_vertices": int(n),
               "probe_ids_max": int(max(int(np.max(v)) for k in ("hand", "arm") for v in ids[k].values())),
               "body_ids_max": int(ids["body"].max()), "faces": int(len(P._faces())),
               "body_surface_faces": int(len(Fb)), "incidence_rows": int(inc.shape[0])}
        row["ids_inside_mesh"] = row["probe_ids_max"] < n and row["body_ids_max"] < n
        if step == "open":
            first = ids
        if step == "open again":
            row["same_entry_as_first_open_read"] = ids is first
        out.append(row)
    os.environ.pop(OB.BACKEND_ENV, None)
    os.environ.pop(OB.REFERENCE_ENV, None)
    return {"reads": out, "cache": {"probe_ids_of": str(P._probe_ids_of.cache_info()),
                                    "body_surface_of": str(P._body_surface_of.cache_info())}}


# ---------------------------------------------------------------- the layered clips' checks on Anny
def background():
    from mdfm.synth import catalogue as C
    by = {s.clip_id: s for s in C.catalogue_set()}
    out = {}
    for cid in BACKGROUND_CLIPS:
        t0 = time.time()
        b, tr = by[cid].build()
        bg = tr.get("background") or {}
        v = tr.get("validity") or {}
        keep = {k: v[k] for k in ("table", "hand_head", "hand_trunk", "fold", "armrest", "hand_thigh", "backrest")
                if k in v}
        out[cid] = {"background": {"kind": bg.get("kind"), "ok": bg.get("ok"), "failed": bg.get("failed"),
                                   "checks_read_on_layer": bg.get("checks_read_on_layer"),
                                   "skipped": bg.get("skipped")},
                    "checks": _jsonable(keep), "failed": _failed(v), "seconds": round(time.time() - t0, 1)}
        print("background", cid, out[cid]["failed"], out[cid]["seconds"], flush=True)
    return out


# ---------------------------------------------------------------- the exam's references and freezing clips
def _episodes(tr):
    out = []
    for e in tr["params"]["episodes"]:
        out.append({"context": e["context"], "onset_s": _r(e["onset_s"], 2), "duration_s": _r(e["duration_s"], 2),
                    "support": e["support"], "heel_lift_mm": [_r(1000 * x, 2) for x in e["heel_lift_mesh_m"]],
                    "sole_gap_mm": _r(1000 * e["foot_gap_mesh_m"], 2),
                    "tremble_m_used": e.get("tremble_m_used")})
    return out


def _holds(out_body, tr):
    from mdfm.synth import freezing as F
    rate = out_body.rate
    worst_h, worst_g = 0.0, 0.0
    for e in tr["params"]["episodes"]:
        hk = np.arange(int(round(e["hold_s"][0] * rate)), int(round(e["hold_s"][1] * rate)), 5)
        h, g = F.heel_lift_mesh(out_body, hk)
        worst_h, worst_g = max(worst_h, float(h.max())), max(worst_g, float(g.max()))
    return {"heel_lift_max_mm": _r(1000 * worst_h, 2), "sole_gap_max_mm": _r(1000 * worst_g, 2)}


def exam_refs():
    from mdfm.synth import base as B
    from mdfm.synth import exam as EX
    from mdfm.synth import freezing as F
    from mdfm.synth import legcontact as LC
    from mdfm.synth.body import limit_violations
    out = {}
    t0 = time.time()
    st, stt = EX.reference_stand()
    out["ref_stand_E7"] = {"source": stt["params"]["source"], "knots": st.T,
                           "limits": _jsonable(limit_violations(st.rot6d)), "seconds": round(time.time() - t0, 1)}
    t0 = time.time()
    wk, wt = EX.reference_walk()
    m = wt["params"].get("measured") or {}
    out["ref_walk_E8"] = {"source": wt["params"]["source"], "knots": wk.T, "duration_s": _r(wk.duration, 2),
                          "gait_ok": (wt["validity"].get("gait") or {}).get("ok"), "failed": _failed(wt["validity"]),
                          "stance_slip_p95_mm_s": _r(1000 * wt["validity"].get("stance_slip_p95_m_s", np.nan), 3),
                          "measured": _jsonable(m),
                          "contexts": _jsonable(B.gait_contexts(wk)), "seconds": round(time.time() - t0, 1)}
    turns = {}
    for ts in FOG_TURNS_S:
        t0 = time.time()
        w = dict(EX.FOG_WALK, turns=((EX.FOG_WALK["turns"][0][0], EX.FOG_WALK["turns"][0][1], ts),))
        b, _ = B.walk(w["duration_s"], turns=[tuple(t) for t in w["turns"]], stop_at_s=w["stop_at_s"], kind=w["kind"])
        g = LC.leg_gap(b)
        turns[f"{ts:g}"] = {"hard_min_mm": _r(1000 * float(np.min(g["hard_m"])), 3),
                            "soft_min_mm": _r(1000 * float(np.min(g["soft_m"])), 3),
                            "hard_tol_mm": _r(1000 * LC.TOL_M, 1),
                            "keeps_hard_tol": bool(np.min(g["hard_m"]) >= -LC.TOL_M),
                            "limits": _jsonable(limit_violations(b.rot6d)), "contexts": _jsonable(B.gait_contexts(b)),
                            "seconds": round(time.time() - t0, 1)}
        print("fog turn", ts, turns[f"{ts:g}"]["hard_min_mm"], flush=True)
    out["fog_walk"] = {"walk": _jsonable(EX.FOG_WALK), "rule": EX.FOG_TURN_RULE, "turns": turns,
                       "chosen_s": EX.FOG_WALK["turns"][0][2]}
    specs = {s.clip_id: s for s in EX.exam_set()}
    clips = {}
    for cid in [EX.FOG_LEVEL0] + [f"freezing_E8_g{g}_trembling" for g in (1, 2, 3, 4)] + ["freezing_E8_g2_akinesia"]:
        t0 = time.time()
        b, tr = specs[cid].build()
        hb = tr["params"].get("hold_background") or {}
        rec = {"knots": b.T, "duration_s": _r(b.duration, 2), "failed": _failed(tr.get("validity")),
               "hold_background": {"ok": hb.get("ok"), "still": hb.get("still"), "seed": hb.get("seed"),
                                   "spans_s": _jsonable((hb.get("span") or {}).get("spans_s")),
                                   "foot_moved_mm": _r(1000 * hb.get("foot_moved_m", 0.0), 4)},
               "seconds": round(time.time() - t0, 1)}
        if tr["params"].get("episodes"):
            rec["episodes"] = _episodes(tr)
            rec["holds"] = _holds(b, tr)
            rec["limits_new"] = tr["validity"].get("limits_new")
        clips[cid] = rec
        print("exam", cid, rec["failed"], rec["seconds"], flush=True)
    out["freezing_clips"] = clips
    healthy = {}
    for g, man in ((1, "trembling"), (2, "trembling"), (3, "trembling"), (4, "trembling"), (2, "akinesia")):
        kw = dict(manifestation=man, support="mesh", arms="settle")
        if man == "trembling":
            kw["pattern"] = "alternating"
        _, tr = F.freezing_grade(wk, g, seed=EX.SEED + g, **kw)
        eps = _episodes(tr)
        healthy[f"g{g}_{man}"] = {"episodes": eps, "heel_lift_max_mm": max(max(e["heel_lift_mm"]) for e in eps),
                                  "sole_gap_max_mm": max(e["sole_gap_mm"] for e in eps)}
    out["on_the_healthy_walk"] = {"rule": "the same plans (exam.SEED, support 'mesh') on exam.reference_walk: the least "
                                          "miss of the onset's tolerances (freezing.snap_onset_mesh)",
                                  "mesh_heel_tol_mm": 1000 * F.MESH_HEEL_TOL_M,
                                  "mesh_support_tol_mm": 1000 * F.MESH_SUPPORT_TOL_M, "clips": healthy}
    return out


# ---------------------------------------------------------------- the SP3 stimuli on the open body
def stimuli():
    from mdfm.synth import stimuli as S
    t0 = time.time()
    stims = S.probe_set()
    rows = {}
    for s in stims:
        rb = S.read_back(s, s.body)
        tv = S.truth_value(s)
        rows[s.name] = {"read_back": _jsonable({k: v for k, v in rb.items() if k != "amplitudes_deg"}),
                        "truth": _jsonable({k: v for k, v in tv.items() if k != "amplitudes_deg"}),
                        "limits_new": (s.truth.get("validity") or {}).get("limits_new"), "knots": s.body.T}
        if "amplitudes_deg" in rb:
            a, b = np.asarray(rb["amplitudes_deg"]), np.asarray(tv["amplitudes_deg"])
            rows[s.name]["amplitudes_max_abs_diff_deg"] = _r(np.abs(a - b).max(), 4)
    return {"walk_kind": S.WALK_KIND, "n": len(stims), "stimuli": rows, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- the export's landmarks
def _landmarks_here():
    import importlib.util
    from mdfm.synth import skeleton as SK
    from mdfm.synth.body import model_name
    spec = importlib.util.spec_from_file_location("rsc", os.path.join(ROOT, "scripts", "render_skeleton_catalogue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    off = mod.landmark_offsets()
    return {"model": model_name(), "head": {k: list(v) for k, v in SK.head_landmarks().items()},
            "catalogue": {k: {"joint": int(j), "offset_m": list(o)} for k, (j, o) in off.items()},
            "round_m": {"head": SK.LANDMARK_ROUND_M, "catalogue": mod.LANDMARK_ROUND_M}}


def landmarks():
    from mdfm.synth import openbody as OB
    here = _landmarks_here()
    env = dict(os.environ, **{OB.BACKEND_ENV: OB.SMPLH_REFERENCE, OB.REFERENCE_ENV: "1",
                              "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--landmarks-child"], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    ref = json.loads(r.stdout.strip().splitlines()[-1])
    diff = {k: _r(1000 * float(np.linalg.norm(np.subtract(here["catalogue"][k]["offset_m"],
                                                           ref["catalogue"][k]["offset_m"]))), 1)
            for k in here["catalogue"]}
    hdiff = {k: _r(1000 * float(np.linalg.norm(np.subtract(here["head"][k], ref["head"][k]))), 1) for k in here["head"]}
    return {"open": here, "smplh_reference": ref, "offset_difference_mm": {"head": hdiff, "catalogue": diff}}


# ---------------------------------------------------------------- the page (export format skeleton_v7_open_body)
def page():
    import glob
    import gzip
    out_dir = PAGE_DIR
    with open(os.path.join(out_dir, "SYNTHETIC_skeleton_manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    formats, captured, n = {}, 0, 0
    for p in sorted(glob.glob(os.path.join(out_dir, "data", "SYNTHETIC_*_skeleton.json.gz"))):
        if os.path.basename(p).startswith("._"):
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        n += 1
        fm = (d.get("code") or {}).get("export_format")
        formats[fm] = formats.get(fm, 0) + 1
        captured += int(bool((d.get("provenance") or {}).get("captured")))
    keep = ("set", "page_sha256", "page_bytes", "clips", "clips_planned", "clips_by_family", "clips_by_since",
            "clips_with_animated_hands", "frames", "data_bytes", "refused", "deferred", "clips_refused_by_family",
            "stale", "skeleton_checks_flagged", "skeleton_checks_read", "reported_checks", "created_utc", "page_code",
            "script")
    return {"out": out_dir, "manifest": {k: man.get(k) for k in keep}, "data_files": n, "export_formats": formats,
            "data_files_captured": captured}


SECTIONS = {"physio_cache": physio_cache, "background": background, "exam_refs": exam_refs, "stimuli": stimuli,
            "landmarks": landmarks, "page": page}
DEFAULT = ("physio_cache", "background", "exam_refs", "stimuli", "landmarks")   # 'page' after the page's build


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(SECTIONS))
    ap.add_argument("--landmarks-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    if a.landmarks_child:
        print(json.dumps(_landmarks_here()))
        return
    rec = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rec = json.load(f)
    rec.setdefault("measured", {})
    for name in (a.only or DEFAULT):
        t0 = time.time()
        rec[name] = _jsonable(SECTIONS[name]())
        rec["measured"][name] = {"mdfm_commit": _head(), "at": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "seconds": round(time.time() - t0, 1)}
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print("wrote", name, rec["measured"][name], flush=True)
    rec["script"] = "scripts/openbody_wave7_checks.py"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
