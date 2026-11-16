"""The physiological background of the synthetic patient (shared change SH1, mdfm/synth/physio.py) laid on every clip
of the symptom catalogue: builds each wrapped clip of catalogue.catalogue_set() and records what the layer did to it.

Per clip (reports/synth/background/sweep_<part>of<parts>.json): skipped (and why) or the posture kind, the verdict of
the layer's checks ('ok', 'failed', the largest scene, body and acceleration excesses, the anchor error), the raw
body's digest (raw_sha256), the clip's own
arms' contact where its build reads one (involuntary.arm_contact; 'export_ok' false when either refuses), the clip's own
corrections (physio record 'deviations': on its own rows only) and the series root's deviations it takes, the guard
counts, the transfer's shares (rows held, unchanged, transferred, corrected; knots beyond the level 0), the build times
(the clip's own build, then the layer), the layer's largest displacement of the joints (head, wrists, pelvis), the
anchor errors (the clip's own arm rows, and apart the knots where its shoulder or elbow row is the level 0's, the knots
where its hand moves on its support, those where it stands still and those where the own limit guard holds the arm),
the distance a foot point moves
at the clip's stance knots (horizontal, and apart in height; a foot on its toes read on its ball), the span of a
walking clip's or a transfer's layer (physio-v7: the still spans, the standing phase), the statue check of the
skeleton page (scripts/render_skeleton_catalogue.check_statue, on the joints at the page's frame rate) on the raw and
the layered body; with a level 0
(physio.pair_change): the sign (the largest |clip - level 0| without the layer) and the change the layer brings to it
(with minus without), on the 22 body joints at
every knot and on every mesh vertex at every physio.PAIR_EVERY-th knot (a finger, a toe or a head turn moves vertices,
not joints), the change as a share of the sign (both read on joints and vertices), above SHARE_AIM the same share with
the finger rows of both clips kept raw (what the change owes the finger layer), and the identity measures: the
largest difference after the layer of the rows (body, hand, translation) the two hold equal before it, the largest
change of a joint position at the knots where the two hold that joint's row, its ancestors' rows and the translation
equal, and on a joint they hold equal over the clip (EQUAL_M).

Clips are dealt to the parts series by series (the root of the level-0 chain: physio.series_root), the largest series
first to the part holding the fewest clips (or, with --balance-from, the least build time recorded in earlier sweep
files), so that a part builds a level 0 once; --parts n --part i runs the i-th of n, --clips a subset (--tag names its
file apart: sweep_<tag>_<part>of<parts>.json, read last by --summary in the order they were written, so that a
rerun of a few clips, an error for instance, takes over their records). A part's file is resumable: a clip
recorded with the same code digest (mdfm/synth/*.py) is kept. --summary merges the parts' files (with --one-digest,
open-body port wave 7, only those built on the tree's digest, the others named) into summary.json and
summary.md (counts by verdict and
kind, the identity table, the failing clips, the clips' own corrections, the series roots' deviations, the pairs
above SHARE_AIM of the sign, the largest changes, the anchor errors on rows held, on moving hands, on hands standing
still and on guarded arms, the feet moved (horizontally and in height), the build times).
The clips of a series that takes no layer (physio.skip_reason of its root) are recorded as skipped
without being built, but the freezing clips (family 'freezing', on their own walk since the open-body port's wave 7,
exam_v10; on the AMASS walk before): their own build lays the standing layer on their holds and on the walk's still
runs (physio.with_hold_background), whose verdict, span and statue check are recorded. The AMASS takes (AMASS) were
left out by default (they needed the AMASS body store; --with-amass): the exam set holds none since wave 7.
--finger-rule measures physio.FINGER_RULE's values on a few pairs (finger_rule.json).

CPU only; procedural bodies in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read; nothing is
written outside reports/synth/background/.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "synth", "background")
AMASS = ()              # the exam set's AMASS takes: none since the open-body port's wave 7 (exam_v10, procedural)
EQUAL_M = 1e-6          # a joint whose clip - level 0 stays within this over the clip is held equal by the two
FINGER_PAIRS = ("tremor_E1_pillrolling_g1", "tremor_E1_pillrolling_supination_g2", "invol_athetosis_E1_a2_d4",
                "invol_athetosis_E1_a2_d3", "invol_td_E1_limbs_s2", "post_E1_fixed_fist")   # --finger-rule
SHARE_AIM = 0.05        # the change of clip - level 0 aimed at, as a share of the sign (the summary lists those above)
NO_SIGN_MM = 1e-6       # a pair whose sign on joints and vertices stays under this has none (the clip equals its level
#                         0 over their common knots): no share


def _sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def synth_sha256():
    """SHA-256 over mdfm/synth/*.py: one line '<name>\\t<file SHA-256>' per file, sorted (sidecars left out)."""
    files = glob.glob(os.path.join(ROOT, "mdfm", "synth", "*.py"))
    lines = [f"{os.path.basename(p)}\t{_sha256_file(p)}\n" for p in files if not os.path.basename(p).startswith("._")]
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def raw_sha256(b):
    """SHA-256 over a raw body's rot6d, transl and hand6d values (the last only for a body with hand joints): two
    sweeps' records name the clips whose own body changed."""
    h = hashlib.sha256()
    for name in ("rot6d", "transl", "hand6d"):
        a = getattr(b, name, None)
        if a is not None:
            h.update(a.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def plan(by, P, parts, clips=None, skip=AMASS, weights=None):
    """[[clip ids] per part]: whole series dealt to the parts, the largest series first, each to the part holding the
    fewest clips so far (ties to the lowest part); each series root first, its other clips in catalogue order. weights
    ({clip: seconds}, --balance-from): the series' build seconds instead of their clip counts (a clip unrecorded
    counts the median)."""
    series = {}
    for cid in by:
        if cid in skip or (clips and cid not in clips):
            continue
        series.setdefault(P.series_root(cid, by), []).append(cid)
    order = list(by)
    if weights:
        med = sorted(weights.values())[len(weights) // 2]
        size = {r: sum(weights.get(c, med) for c in mem) for r, mem in series.items()}
    else:
        size = {r: float(len(mem)) for r, mem in series.items()}
    out = [[] for _ in range(parts)]
    load = [0.0] * parts
    for root, mem in sorted(series.items(), key=lambda kv: (-size[kv[0]], order.index(kv[0]))):
        i = min(range(parts), key=lambda p: (load[p], p))
        out[i] += sorted(mem, key=lambda c: (c != root, order.index(c)))
        load[i] += size[root]
    return out


def weights_from(folder):
    """{clip: seconds} of the build and layer times recorded in the sweep files of folder."""
    w = {}
    for p in sorted(glob.glob(os.path.join(folder, "sweep_*.json"))):
        with open(p, encoding="utf-8") as f:
            for c, r in json.load(f).get("clips", {}).items():
                if "raw_build_s" in r and r.get("built", True):
                    w[c] = float(r["raw_build_s"]) + float(r.get("layer_s", 0.0))
    return w


_RSC = []


def statue_of(body, supine=False):
    """The skeleton page's statue check (render_skeleton_catalogue.check_statue) on a body's 22 joints, linearly
    resampled at the page's frame rate, on the export's grid (whole mm): {'ok', 'statue_s', 'spans_s'}."""
    import importlib.util
    import numpy as np
    if not _RSC:
        spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_for_sweep",
                                                      os.path.join(ROOT, "scripts", "render_skeleton_catalogue.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RSC.append(m)
    rsc = _RSC[0]
    J = body.joints().numpy()
    t = np.arange(int(np.floor((body.T - 1) / float(body.rate) * rsc.FPS + 1e-9)) + 1) / rsc.FPS
    k = t * float(body.rate)
    k0 = np.clip(np.floor(k).astype(int), 0, body.T - 1)
    k1 = np.clip(k0 + 1, 0, body.T - 1)
    a = (k - k0)[:, None, None]
    r = rsc.check_statue(np.rint(1000.0 * (J[k0] * (1.0 - a) + J[k1] * a)), rsc.FPS, supine)
    return {k_: r[k_] for k_ in ("ok", "statue_s", "spans_s")}


def record(cid, by, P, cache):
    """The record of one clip (module docstring)."""
    import numpy as np
    from mdfm.body.smplh import PARENTS
    from mdfm.synth.hands import _eye6, has_hands

    def get(c):
        if c not in cache:
            t0 = time.perf_counter()
            b, tr = by[c].build()
            t_all = time.perf_counter() - t0
            t = by[c].build.last_times or {"clip_s": t_all, "layer_s": 0.0}   # a skipped series sets no layer time
            raw, _ = by[c].build.raw()
            cache[c] = (b, tr, raw, t["clip_s"], t["layer_s"])
        return cache[c]
    root = P.series_root(cid, by)
    why = P.skip_reason(by[root])
    if why is not None and by[cid].family == "freezing":       # the holds of a freezing clip: its own build's layer
        b, tr, raw, t_raw, t_layer = get(cid)
        hb = (tr.get("params") or {}).get("hold_background") or {}
        hv = (tr.get("validity") or {}).get("hold_background") or {}
        return {"series": root, "skipped": True, "reason": why, "raw_build_s": round(t_raw, 2), "layer_s": 0.0,
                "raw_sha256": raw_sha256(raw), "hold_background": {
                    "ok": hv.get("ok"), "failed": hv.get("failed"), "span": hb.get("span"),
                    "deviations": hb.get("deviations"), "foot_moved_mm": 1000 * hb.get("foot_moved_m", 0.0),
                    "accel_excess_m_s2": hb.get("accel_excess_m_s2")},
                "statue": statue_of(b), "export_ok": bool(hv.get("ok", True))}
    if why is not None:                                          # a series that takes no layer: not built
        return {"series": root, "skipped": True, "reason": why,
                "raw_build_s": 0.0, "layer_s": 0.0, "built": False}
    b, tr, raw, t_raw, t_layer = get(cid)
    g = tr["background"]
    r = {"series": g.get("series"), "skipped": bool(g.get("skipped")), "raw_build_s": round(t_raw, 2),
         "layer_s": round(t_layer, 2), "raw_sha256": raw_sha256(raw)}
    if g.get("skipped"):
        r["reason"] = g.get("reason")
        return r
    v = g["validity"]
    r.update(kind=g["kind"], ok=bool(v["ok"]), failed=list(v["failed"]), limits_table=v["limits_table"],
             limits_new=v["limits_new"], scene_added_mm=1000 * v["scene_added_m"],
             body_added_mm=1000 * v["body_added_m"], body_deepened_mm=1000 * v["body_deepened_m"],
             legs_added_mm=1000 * v["legs_added_m"], accel_excess_m_s2=v["accel_excess_m_s2"],
             anchor_error_mm=1000 * v["anchor_error_m"],
             anchor_error_held_rows_mm=1000 * v.get("anchor_error_held_rows_m", 0.0),
             anchor_error_moving_mm=1000 * v.get("anchor_error_moving_m", 0.0),
             anchor_error_guarded_mm=1000 * v.get("anchor_error_guarded_m", 0.0),
             anchor_error_dwell_mm=1000 * v.get("anchor_error_dwell_m", 0.0),
             foot_moved_mm=1000 * v.get("foot_moved_m", 0.0), skate_tol_m_s=v.get("skate_tol_m_s"),
             foot_height_moved_mm=1000 * v.get("foot_height_moved_m", 0.0), toe_knots=int(v.get("toe_knots", 0)),
             deviations=sorted(g["deviations"]), guard=g["guard"], touch_share=g["touch_share"])
    ac = (tr.get("validity") or {}).get("arm_contact")           # the clip's own arms' contact (involuntary.py),
    if isinstance(ac, dict):                                    # read on its raw build: an export refuses it too
        r["arm_contact"] = {"ok": bool(ac.get("ok")), "after_cm_max": dict(ac.get("after_cm_max") or {}),
                            "tol_cm": ac.get("tol_cm"), "read_after": ac.get("read_after")}
    r["export_ok"] = bool(tr["validity"]["background"]["ok"]) and r.get("arm_contact", {}).get("ok", True)
    d = np.linalg.norm((b.joints() - raw.joints()).numpy(), axis=-1)
    r["layer_mm"] = {"max": 1000 * float(d.max()), "head": 1000 * float(d[:, 15].max()),
                     "wrists": 1000 * float(d[:, 20:22].max()), "pelvis": 1000 * float(d[:, 0].max())}
    r["knots"] = int(b.T)
    supine = g["kind"] == "supine"
    r["statue"] = statue_of(b, supine)
    r["statue_raw"] = statue_of(raw, supine)
    if "span" in g:
        r["span"] = g["span"]
    r["accel_added_m_s2"] = v["accel_added_m_s2"]
    r["skate_added_m_s"] = v["skate_added_m_s"]
    if "ankle_error_m" in g:
        r["ankle_error_mm"] = 1000 * g["ankle_error_m"]
    r["series_deviations"] = list(g.get("series_deviations", []))
    t = g.get("transfer") or {}
    r["transfer"] = {k: t.get(k) for k in ("reference", "reference_knots", "beyond_knots", "rows_held_share",
                                           "rows_unchanged_share", "rows_transferred_share", "rows_corrected",
                                           "rows_corrected_held", "own_solve", "level0", "knots_held_share",
                                           "copied_max_diff", "part_held_rows_maxdiff") if k in t}
    l0 = by[cid].level0
    if l0 is not None and l0 not in AMASS:
        lb, _, lraw, _, _ = get(l0)
        T = min(b.T, lb.T)
        wb = b.joints().numpy()[:T] - lb.joints().numpy()[:T]
        wo = raw.joints().numpy()[:T] - lraw.joints().numpy()[:T]
        ch = np.linalg.norm(wb - wo, axis=-1)
        sign = np.linalg.norm(wo, axis=-1)
        eq = sign.max(0) <= EQUAL_M
        pc = P.pair_change(raw, b, lraw, lb)
        held = P.rows_equal(raw.rot6d, lraw.rot6d, T)                     # rows the clip holds equal to its level 0
        out_d = (b.rot6d[:T] - lb.rot6d[:T]).abs().amax(-1).numpy()
        th = ((raw.transl[:T] - lraw.transl[:T]).abs().amax(-1) <= P.EQUAL_ROW_TOL).numpy()
        pos_eq = np.zeros_like(held)                                       # a joint whose row and ancestors' rows
        for jj in range(held.shape[1]):                                    # are all held (and the translation)
            pos_eq[:, jj] = held[:, jj] & (th if jj == 0 else pos_eq[:, PARENTS[jj]])
        h = {}
        if has_hands(b) or has_hands(lb):
            hc = raw.hand6d if has_hands(raw) else _eye6(raw.T)
            hl = lraw.hand6d if has_hands(lraw) else _eye6(lraw.T)
            hheld = P.rows_equal(hc, hl, T)
            oc = b.hand6d if has_hands(b) else _eye6(b.T)
            ol = lb.hand6d if has_hands(lb) else _eye6(lb.T)
            hd = (oc[:T] - ol[:T]).abs().amax(-1).numpy()
            h = {"held_hand_rows_share": float(hheld.mean()),
                 "held_hand_rows_out_maxdiff": float(hd[hheld].max()) if hheld.any() else 0.0}
        tr_held = ((raw.transl[:T] - lraw.transl[:T]).abs().amax(-1) <= P.EQUAL_ROW_TOL).numpy()
        tr_d = (b.transl[:T] - lb.transl[:T]).abs().amax(-1).numpy()
        r["level0"] = {"id": l0, "knots": int(lb.T), "sign_mm": 1000 * pc["sign_m"], "change_mm": 1000 * pc["change_m"],
                       "change_share": pc["share"],
                       "sign_joints_mm": 1000 * pc["joints"]["sign_m"],
                       "change_joints_mm": 1000 * pc["joints"]["change_m"],
                       "change_joint": pc["joints"]["joint"], "change_at_s": pc["joints"]["at_s"],
                       "sign_vertices_mm": 1000 * pc["vertices"]["sign_m"],
                       "change_vertices_mm": 1000 * pc["vertices"]["change_m"],
                       "change_vertex_joint": pc["vertices"]["joint"], "change_vertex_at_s": pc["vertices"]["at_s"],
                       "equal_joints": [int(x) for x in np.flatnonzero(eq)],
                       "equal_change_mm": 1000 * float(ch[:, eq].max()) if eq.any() else 0.0,
                       "held_rows_share": float(held.mean()),
                       "held_rows_out_maxdiff": float(out_d[held].max()) if held.any() else 0.0,
                       "held_transl_out_maxdiff": float(tr_d[tr_held].max()) if tr_held.any() else 0.0,
                       "held_positions_change_mm": 1000 * float(ch[pos_eq].max()) if pos_eq.any() else 0.0,
                       **h}
        if pc["share"] > SHARE_AIM and (has_hands(b) or has_hands(lb)):
            def raw_fingers(o, src):
                if not has_hands(o):                                  # a body without hand joints has no finger layer
                    return o
                return replace(o, hand6d=(src.hand6d if has_hands(src) else _eye6(src.T)).clone())
            r["level0"]["change_share_raw_fingers"] = P.pair_change(raw, raw_fingers(b, raw), lraw,
                                                                    raw_fingers(lb, lraw))["share"]
    return r


def run_part(args):
    import torch
    torch.set_num_threads(args.threads)
    from mdfm.synth import catalogue as C, physio as P
    specs = C.catalogue_set()
    by = {s.clip_id: s for s in specs}
    skip = () if args.with_amass else AMASS
    weights = weights_from(args.balance_from) if args.balance_from else None
    ids = plan(by, P, args.parts, set(args.clips) if args.clips else None, skip, weights)[args.part]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"sweep_{args.tag + '_' if args.tag else ''}{args.part}of{args.parts}.json")
    code = synth_sha256()
    res = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("synth_sha256") == code:
            res = old.get("clips", {})
    cache = {}
    level0s = {s.level0 for s in specs if s.level0 is not None}
    head = {"physio_version": P.VERSION, "synth_sha256": code, "script_sha256": _sha256_file(os.path.abspath(__file__)),
            "part": args.part, "parts": args.parts, "skipped_ids": list(skip)}
    print(f"part {args.part} of {args.parts}: {len(ids)} clips, {len(res)} already done", flush=True)
    for cid in ids:
        if cid in res and "error" not in res[cid]:
            continue
        try:
            res[cid] = record(cid, by, P, cache)
        except Exception as e:  # noqa: BLE001 (a clip that raises is recorded, the run goes on)
            res[cid] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
        r = res[cid]
        print(cid, json.dumps({k: r.get(k) for k in ("kind", "ok", "failed", "raw_build_s", "layer_s", "error")}),
              flush=True)
        if cid not in level0s:                                   # a level 0 is kept for its series' other clips
            cache.pop(cid, None)
        out = dict(head, created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   clips=res)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    print("done", flush=True)


def finger_rule(args):
    """reports/synth/background/finger_rule.json: the pairs of args.clips (default FINGER_PAIRS) with physio.FINGER_RULE
    at each of its values, their change as a share of the sign (the measure of the rule on the own finger rows)."""
    import torch
    torch.set_num_threads(args.threads)
    from mdfm.synth import catalogue as C, physio as P
    by = {s.clip_id: s for s in C.catalogue_set(background=False)}
    cache = P.SeriesCache(by)
    keep = P.FINGER_RULE
    res = {}
    for cid in args.clips or FINGER_PAIRS:
        res[cid] = {"level0": by[cid].level0}
        raw = by[cid].build()
        l_raw, l_out, _ = cache.layered(by[cid].level0)
        for rule in ("none", "held", "all"):
            P.FINGER_RULE = rule
            cache.refs.pop(cid, None)
            _, out, rec = cache.layered(cid, raw[0], raw[1])
            pc = P.pair_change(raw[0], out, l_raw, l_out)
            res[cid][rule] = {"share": pc["share"], "change_mm": 1000 * pc["change_m"], "sign_mm": 1000 * pc["sign_m"],
                              "failed": rec["validity"]["failed"],
                              "still_knots": rec["guard"].get("fingers_still_knots")}
            print(cid, rule, res[cid][rule], flush=True)
        P.FINGER_RULE = keep
    with open(os.path.join(OUT, "finger_rule.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"physio_version": P.VERSION, "synth_sha256": synth_sha256(), "pairs": res,
                   "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                  f, indent=1, sort_keys=True)


def summary(args):
    from mdfm.synth import physio as P
    P_EVERY = P.PAIR_EVERY

    def created(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("created_utc", "")
    # the tagged reruns last, the latest last (a rerun of a rerun's clips takes over)
    parts = sorted(glob.glob(os.path.join(OUT, "sweep_*of*.json")),
                   key=lambda p: (os.path.basename(p)[6].isalpha(), created(p) if os.path.basename(p)[6].isalpha()
                                  else "", os.path.basename(p)))
    clips, codes, versions = {}, set(), set()
    here, other = synth_sha256(), []
    for p in parts:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if args.one_digest and d["synth_sha256"] != here:       # wave 7: the sweep on one digest
            other.append(os.path.basename(p))
            continue
        codes.add(d["synth_sha256"])
        versions.add(d.get("physio_version"))
        clips.update(d["clips"])
    built = {k: r for k, r in clips.items() if "error" not in r}
    layered = {k: r for k, r in built.items() if not r["skipped"]}
    fail = {k: r["failed"] for k, r in layered.items() if not r["ok"]}
    arm_refused = {k: r["arm_contact"]["after_cm_max"] for k, r in layered.items()
                   if not r.get("arm_contact", {}).get("ok", True)}
    export_refused = sorted(k for k, r in built.items() if not r.get("export_ok", r.get("ok", True)))
    kinds = {}
    for r in layered.values():
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    dev, sdev = {}, {}
    for k, r in layered.items():
        for x in r["deviations"]:
            dev.setdefault(x, []).append(k)
        t_ = r.get("transfer") or {}
        if t_.get("reference") is None and t_.get("level0") is None:   # a series root (a walking member's own
            for x in r.get("series_deviations", []):                  # solve names its level 0 apart)
                sdev.setdefault(x, []).append(k)
    pairs = {k: r["level0"] for k, r in layered.items() if "level0" in r}
    nosign = sorted(k for k, v in pairs.items() if v["sign_mm"] < NO_SIGN_MM)
    pairs_s = {k: v for k, v in pairs.items() if v["sign_mm"] >= NO_SIGN_MM}
    vert_only = sorted(k for k, v in pairs_s.items() if v.get("sign_joints_mm", 0.0) < 0.001)
    worst = sorted(pairs_s.items(), key=lambda kv: -kv[1]["change_share"])[:25]
    above = sorted(((k, v) for k, v in pairs_s.items() if v["change_share"] > SHARE_AIM),
                   key=lambda kv: -kv[1]["change_share"])
    eq_worst = sorted(pairs.items(), key=lambda kv: -kv[1]["equal_change_mm"])[:10]

    walk_pairs = {k: v for k, v in pairs.items() if layered[k]["kind"] == "walking"}
    pairs_t = {k: v for k, v in pairs.items() if k not in walk_pairs}   # the transferred pairs (the identity table)

    def top(key, of=None):
        return max((v.get(key, 0.0) for v in (pairs_t if of is None else of).values()), default=0.0)
    walking_ident = {"pairs": len(walk_pairs), "held_rows_out_maxdiff": top("held_rows_out_maxdiff", walk_pairs),
                     "held_transl_out_maxdiff": top("held_transl_out_maxdiff", walk_pairs),
                     "equal_change_mm": top("equal_change_mm", walk_pairs),
                     "change_share_max": max((v["change_share"] for k, v in pairs_s.items() if k in walk_pairs),
                                             default=0.0),
                     "knots_held_share_min": min(((layered[k].get("transfer") or {}).get("knots_held_share", 1.0)
                                                  for k in walk_pairs), default=1.0)}
    ident = {"pairs": len(pairs_t), "held_rows_out_maxdiff": top("held_rows_out_maxdiff"),
             "held_hand_rows_out_maxdiff": top("held_hand_rows_out_maxdiff"),
             "held_transl_out_maxdiff": top("held_transl_out_maxdiff"),
             "held_positions_change_mm": top("held_positions_change_mm"), "equal_change_mm": top("equal_change_mm"),
             "change_share_max": max((v["change_share"] for v in pairs_s.values()), default=0.0),
             "pairs_above_aim": len(above), "pairs_without_sign": nosign,
             "pairs_with_a_vertex_sign_only": vert_only,
             "rows_corrected_held": sum((r.get("transfer") or {}).get("rows_corrected_held", 0)
                                        for r in layered.values())}
    beyond = sorted(k for k, r in layered.items() if (r.get("transfer") or {}).get("beyond_knots", 0) > 0)
    skate = sorted(((k, r.get("skate_added_m_s", 0.0)) for k, r in layered.items()), key=lambda kv: -kv[1])[:10]
    feet = sorted(((k, r.get("foot_moved_mm", 0.0)) for k, r in layered.items() if r.get("foot_moved_mm", 0.0) > 1e-6),
                  key=lambda kv: -kv[1])[:10]
    held_anchor = sorted(((k, r.get("anchor_error_held_rows_mm", 0.0)) for k, r in layered.items()
                          if r.get("anchor_error_held_rows_mm", 0.0) > 0.0), key=lambda kv: -kv[1])[:10]
    moving_anchor = sorted(((k, r.get("anchor_error_moving_mm", 0.0)) for k, r in layered.items()
                            if r.get("anchor_error_moving_mm", 0.0) > 0.0), key=lambda kv: -kv[1])[:10]
    guarded_anchor = sorted(((k, r.get("anchor_error_guarded_mm", 0.0)) for k, r in layered.items()
                             if r.get("anchor_error_guarded_mm", 0.0) > 0.0), key=lambda kv: -kv[1])[:10]
    dwell_anchor = sorted(((k, r.get("anchor_error_dwell_mm", 0.0)) for k, r in layered.items()
                           if r.get("anchor_error_dwell_mm", 0.0) > 0.0), key=lambda kv: -kv[1])[:10]
    statues = {k: r["statue"] for k, r in built.items() if "statue" in r}
    statue_fail = sorted(k for k, x in statues.items() if not x["ok"])
    statue_raw = sorted(k for k, r in layered.items() if not r.get("statue_raw", {}).get("ok", True))
    pair_statue = sorted(k for k, r in layered.items() if "level0" in r
                         and not (statues.get(k, {}).get("ok", True)
                                  and statues.get(r["level0"]["id"], {}).get("ok", True)))
    holds = {k: r["hold_background"] for k, r in built.items() if "hold_background" in r}
    spans = {k: r["span"] for k, r in layered.items() if "span" in r}
    feet_h = sorted(((k, r.get("foot_height_moved_mm", 0.0)) for k, r in layered.items()
                     if r.get("foot_height_moved_mm", 0.0) > 1e-6), key=lambda kv: -kv[1])[:10]
    s = {"parts": [os.path.basename(p) for p in parts if os.path.basename(p) not in other],
         "parts_on_another_digest": other, "synth_sha256": sorted(codes),
         "physio_version": sorted(v for v in versions if v), "clips": len(clips),
         "errors": sorted(k for k, r in clips.items() if "error" in r),
         "skipped": sorted(k for k, r in built.items() if r["skipped"]), "layered": len(layered), "kinds": kinds,
         "ok": len(layered) - len(fail), "failed": fail, "arm_contact_refused": arm_refused,
         "export_refused": export_refused, "deviations": {k: sorted(v) for k, v in dev.items()},
         "series_deviations": {k: sorted(v) for k, v in sdev.items()}, "identity": ident,
         "identity_walking": walking_ident, "beyond": beyond,
         "raw_build_s": sum(r["raw_build_s"] for r in built.values()),
         "layer_s": sum(r.get("layer_s", 0.0) for r in built.values()),
         "largest_change_share": [{"clip": k, **v} for k, v in worst],
         "above_aim": [{"clip": k, **v} for k, v in above],
         "largest_equal_change": [{"clip": k, **v} for k, v in eq_worst],
         "largest_skate_added": [{"clip": k, "skate_added_m_s": x} for k, x in skate],
         "largest_foot_moved": [{"clip": k, "foot_moved_mm": x} for k, x in feet],
         "largest_anchor_error_held_rows": [{"clip": k, "anchor_error_held_rows_mm": x} for k, x in held_anchor],
         "largest_anchor_error_moving": [{"clip": k, "anchor_error_moving_mm": x} for k, x in moving_anchor],
         "largest_anchor_error_guarded": [{"clip": k, "anchor_error_guarded_mm": x} for k, x in guarded_anchor],
         "largest_anchor_error_dwell": [{"clip": k, "anchor_error_dwell_mm": x} for k, x in dwell_anchor],
         "largest_foot_height_moved": [{"clip": k, "foot_height_moved_mm": x} for k, x in feet_h],
         "statue_checked": len(statues), "statue_failed": statue_fail, "statue_failed_raw": statue_raw,
         "pairs_with_a_statue": pair_statue, "hold_background": holds, "spans": spans}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(s, f, indent=1, sort_keys=True)
    lines = ["# Physiological background (SH1) over the symptom catalogue", "",
             "Built by scripts/synth_background_sweep.py; layer version(s) "
             f"{', '.join(s['physio_version'])}; code digest(s) of mdfm/synth: {', '.join(sorted(codes))}"
             + (f" (--one-digest: the part files on another digest left out: {', '.join(other)})" if other else "")
             + ".", "",
             "The layer is solved once per series, on the series root (the root of the level-0 chain); every other "
             "clip takes it from its direct level 0 as per-knot deltas, a row the clip holds equal to its level 0's "
             "taking the level 0's output row, copied, and the clip's own corrections touching its own rows only "
             "(physio.transfer).", "",
             f"- clips recorded: {s['clips']}; errors: {len(s['errors'])}; skipped series' clips: {len(s['skipped'])}",
             f"- clips with the layer: {s['layered']} ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})",
             f"- passing the layer's checks: {s['ok']}; failing: {len(fail)}",
             f"- refused by their own arms' contact (involuntary.arm_contact, read on the clip's build): "
             f"{len(arm_refused)}; refused for the export (the layer or the arms' contact): {len(export_refused)}",
             f"- clips longer than their level 0 (their own solve beyond it): {len(beyond)}"
             + (f" ({', '.join(beyond)})" if beyond else ""),
             f"- build time: the clips' own builds {s['raw_build_s']:.0f} s, the layer {s['layer_s']:.0f} s", "",
             "## Identity of the layer between a clip and its level 0", "",
             "| measure | value |", "|---|---|",
             f"| clip and level 0 pairs (transferred; the walking pairs below) | {ident['pairs']} |",
             "| largest difference of the output rows a clip holds equal to its level 0 (rot6d values) | "
             f"{ident['held_rows_out_maxdiff']:.3g} |",
             f"| the same on the hand rows | {ident['held_hand_rows_out_maxdiff']:.3g} |",
             f"| the same on the pelvis translation (m) | {ident['held_transl_out_maxdiff']:.3g} |",
             "| largest change of a joint position where the two hold its row, its ancestors' rows and the "
             "translation equal (mm) | "
             f"{ident['held_positions_change_mm']:.3g} |",
             f"| largest change on a joint the two hold equal over the clip (mm) | {ident['equal_change_mm']:.3g} |",
             f"| own corrections written on rows held equal | {ident['rows_corrected_held']} |",
             f"| largest change of clip minus level 0, share of the sign | {ident['change_share_max']:.4f} |",
             f"| pairs above {SHARE_AIM:.2f} of the sign | {ident['pairs_above_aim']} |",
             f"| pairs whose sign shows on mesh vertices only (under 0.001 mm on the 22 joints: fingers, toes, head "
             f"turn), with their share | {len(vert_only)} |",
             f"| pairs with no sign (under {NO_SIGN_MM} mm on joints and vertices; no share) | "
             f"{len(nosign)}: {', '.join(nosign) or 'none'} |", "",
             "The sign and the change are read on the 22 body joints at every knot and on every mesh vertex at every "
             f"{P_EVERY}-th knot (physio.pair_change); the share is the largest change over the largest sign.", "",
             "## Failing clips", ""]
    lines += [f"- {k}: {', '.join(v)}" for k, v in sorted(fail.items())] or ["- none"]
    wi = walking_ident
    lines[lines.index("## Failing clips"):lines.index("## Failing clips")] = [
        "The walking pairs (physio-v7: each clip its own solve on its own still spans, a knot the two hold equal in "
        "full where both spans weigh alike copied from the level 0's output): "
        f"{wi['pairs']} pairs; largest difference of an output row held equal {wi['held_rows_out_maxdiff']:.3g} "
        f"(at the knots held in part only: the own solve's), of the translation {wi['held_transl_out_maxdiff']:.3g} m; "
        f"largest change on a joint held equal over the clip {wi['equal_change_mm']:.3g} mm; largest share of the sign "
        f"{wi['change_share_max']:.4f}.", ""]
    lines += ["", "## The statue check of the skeleton page (a 1 s window with every body joint still; "
              "render_skeleton_catalogue.check_statue on the joints at the page's frame rate)", "",
              f"- clips read: {len(statues)} (the layered clips and the freezing clips); a statue left: "
              f"{len(statue_fail)}" + (f" ({', '.join(statue_fail)})" if statue_fail else ""),
              f"- layered clips that were a statue without the layer: {len(statue_raw)}",
              f"- clip and level 0 pairs with a statue in either: {len(pair_statue)}"
              + (f" ({', '.join(pair_statue)})" if pair_statue else "")]
    if spans:
        sk = {}
        for k, x in spans.items():
            sk.setdefault(layered[k]["kind"], []).append(x["share"])
        lines += [f"- the layer's span (physio-v7), {kd}: {len(v)} clips, layered share of the knots "
                  f"{min(v):.3f} to {max(v):.3f}" for kd, v in sorted(sk.items())]
    lines += ["", "## The freezing holds (their own build's standing layer, physio.with_hold_background)", ""]
    lines += [f"- {k}: {'ok' if h['ok'] else 'failed ' + ', '.join(h['failed'])}, holds "
              + ", ".join(f"{a:.2f} to {b:.2f} s" for a, b in (h.get("span") or {}).get("spans_s", []))
              + f", foot moved {h['foot_moved_mm']:.3g} mm, statue {statues[k]['statue_s']:.2f} s"
              for k, h in sorted(holds.items())] or ["- none"]
    lines += ["", "## Clips refused by their own arms' contact (cm inside, per side)", ""]
    lines += [f"- {k}: " + ", ".join(f"{s_} {x:.3f}" for s_, x in sorted(v.items()))
              for k, v in sorted(arm_refused.items())] or ["- none"]
    lines += ["", "## The clips' own corrections (on their own rows only)", ""]
    lines += [f"- {k}: {len(v)} clips" for k, v in sorted(dev.items())] or ["- none"]
    lines += ["", "## The series roots' own deviations (taken by every clip of their series)", ""]
    lines += [f"- {k}: {', '.join(v)}" for k, v in sorted(sdev.items())] or ["- none"]
    lines += ["", f"## Pairs above {SHARE_AIM:.2f} of the sign", ""]
    lines += [f"- {k} (level 0 {v['id']}): sign {v['sign_mm']:.1f} mm (joints {v['sign_joints_mm']:.1f}), change "
              f"{v['change_mm']:.2f} mm (joints {v['change_joints_mm']:.2f}, joint {v['change_joint']} at "
              f"{v['change_at_s']:.2f} s; vertices {v['change_vertices_mm']:.2f}, of joint {v['change_vertex_joint']} "
              f"at {v['change_vertex_at_s']:.2f} s), share {v['change_share']:.3f}"
              + (f" ({v['change_share_raw_fingers']:.3f} with the finger rows raw)"
                 if "change_share_raw_fingers" in v else "")
              + f"; own corrections {', '.join(layered[k]['deviations']) or 'none'}" for k, v in above] or ["- none"]
    lines += ["", "## Largest changes of clip minus level 0 (share of the sign)", "",
              "| clip | level 0 | sign mm | sign on joints mm | change mm | change on joints mm | share | "
              "where (joint, s) |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['id']} | {v['sign_mm']:.1f} | {v['sign_joints_mm']:.1f} | {v['change_mm']:.2f} | "
              f"{v['change_joints_mm']:.2f} | {v['change_share']:.3f} | "
              + (f"vertex of {v['change_vertex_joint']}, {v['change_vertex_at_s']:.2f}"
                 if v['change_vertices_mm'] >= v['change_joints_mm'] else
                 f"joint {v['change_joint']}, {v['change_at_s']:.2f}") + " |" for k, v in worst]
    lines += ["", "## Largest change on joints the clip and its level 0 hold equal", ""]
    lines += [f"- {k}: {v['equal_change_mm']:.3g} mm" for k, v in eq_worst]
    lines += ["", "## Largest foot speed in stance added by the layer (p95, m/s)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in skate]
    lines += ["", "## Largest distance a foot point moves at the clip's stance knots (mm; refused above "
              f"{1000 * P.FOOT_TOL_M:g} mm; a foot on its toes read on its ball, physio.toe_stance)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in feet] or ["- none"]
    lines += ["", "## Largest anchor error where a member's shoulder or elbow row is its level 0's (mm; the level "
              "0's row copied, which no own correction can turn: reported, not refused)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in held_anchor] or ["- none"]
    lines += ["", "## Largest anchor error where a member's hand moves on its support or its own anchoring ramps in "
              "at rest (mm; no own anchoring while the wrist moves faster than "
              f"{P.ARM_REST_M_S:g} m/s relative to its support: reported, not refused)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in moving_anchor] or ["- none"]
    lines += ["", "## Largest anchor error where a member's hand stands still on its support (mm; the wrist under "
              f"{P.ARM_DWELL_M_S:g} m/s relative to it at every knot within {P.ARM_DWELL_S:g} s: above "
              f"{1000 * P.ANCHOR_TOL_M:g} mm it refuses a rest gate tried on a passing default; reported for the "
              "default and for the gate a refused default falls back on)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in dwell_anchor] or ["- none"]
    lines += ["", "## Largest anchor error where a member's own limit guard holds its shoulder or elbow (mm; the arm "
              "at the bound of the table the clip is judged on, those knots read apart, the goal unchanged: reported, "
              "not refused)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in guarded_anchor] or ["- none"]
    lines += ["", "## Largest height a foot point moves at the clip's stance knots (mm; a member whose hip or knee row "
              "is its level 0's keeps its planted foot's horizontal place and takes the residual in height, a foot on "
              "its toes keeps its ball and turns about it: reported, not refused)", ""]
    lines += [f"- {k}: {x:.3g}" for k, x in feet_h] or ["- none"]
    rule = os.path.join(OUT, "finger_rule.json")
    if os.path.exists(rule):
        with open(rule, encoding="utf-8") as f:
            fr = json.load(f)["pairs"]
        lines += ["", "## Which own finger rows take no finger drift: the share of the sign at each value of "
                  "physio.FINGER_RULE (finger_rule.json; 'none' every row takes it, 'held' the rows held still in a "
                  "posture far from the level 0's take none, the rule in use, 'all' no own row takes it)", "",
                  "| clip | sign mm | none | held | all |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {r['none']['sign_mm']:.1f} | {r['none']['share']:.3f} | {r['held']['share']:.3f} | "
                  f"{r['all']['share']:.3f} |" for k, r in sorted(fr.items())]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({k: s[k] for k in ("clips", "layered", "ok", "kinds")}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--with-amass", action="store_true")
    ap.add_argument("--balance-from", default="", help="a folder of earlier sweep files whose build times balance "
                                                       "the parts (plan weights)")
    ap.add_argument("--tag", default="", help="a name inserted in the part's file name (a rerun of a few clips, "
                                              "read after the other parts by --summary, its records taking over)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--one-digest", action="store_true", help="--summary reads only the part files built on this "
                                                              "tree's code digest, and names the others")
    ap.add_argument("--finger-rule", action="store_true", help="measure physio.FINGER_RULE's values on FINGER_PAIRS or "
                                                               "--clips (finger_rule.json)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.summary:
        summary(args)
    elif args.finger_rule:
        finger_rule(args)
    else:
        run_part(args)


if __name__ == "__main__":
    main()
