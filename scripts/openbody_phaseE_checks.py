"""The readings of plan phase E step 1 (docs/PLAN_2026-11-12.md: the movement gallery rebuilt on the open body) that the
open-body design's section 21 cites (docs/OPEN_BODY_PORT_DESIGN.md): the open items of section 20.7 that this rebuild
fixes or leaves, each read on the code as committed and, where a value changed, at the former value put back.

Sections (all by default, or those named with --only; each section's commit and time in 'measured'); a section is a
function named section_<name> of this module, read in the order of the file:
- layer: the clips the background layer refused in section 20.7 (and the two clips refused after it by their module
  checks), each built raw and with the layer (spec.build.raw, with_background): the layer's verdict, the checks it
  fails and their readings, the deviations it records;
- the sections of the seated hands, of the involuntary and gait clips and of the layer's own rules follow, each with
  its docstring.

Output: reports/openbody/phaseE_checks.json (numbers only; no mesh, shape or model array). CPU; procedural bodies on
the catalogue's one body; no store, lockbox, EVAL, CODY or care_pd record is read.

    CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe \
        scripts/openbody_phaseE_checks.py [--only s ...]
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
OUT = os.path.join(ROOT, "reports", "openbody", "phaseE_checks.json")

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYER_CLIPS = ("tremor_E2_raise_long_rest_g0", "tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2",
               "jerk_functional_E7_axial_sev2_dur2", "invol_chorea_E2_fingers_g3", "invol_chorea_E2_fingers_g4",
               "invol_td_E1_rocking_s4", "invol_chorea_E1_g3", "post_E1_pnkd_excerpt")


def _head():
    r = subprocess.run(["git", "-c", f"safe.directory={ROOT.replace(os.sep, '/')}", "-C", ROOT, "rev-parse", "--short",
                        "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _failed(validity):
    return sorted(k for k, v in (validity or {}).items() if isinstance(v, dict) and v.get("ok") is False)


_CATALOGUE = {}


def catalogue():
    """{clip id: spec} of the symptom catalogue (built once)."""
    if not _CATALOGUE:
        from mdfm.synth import catalogue as C
        _CATALOGUE.update({s.clip_id: s for s in C.catalogue_set()})
    return _CATALOGUE


def layer_reading(cid):
    """The layer's verdict on one clip: built raw and with the layer, the checks it fails, their readings, the
    deviations, and the module checks that fail on the layered body."""
    s = catalogue()[cid]
    raw, truth = s.build.raw()
    b, tr = s.build.with_background(raw, truth)
    g = tr.get("background") or {}
    v = g.get("validity") or {}
    return {"layer_ok": v.get("ok"), "failed": v.get("failed"),
            "readings": {k: _jsonable(v.get(k)) for k in ("accel_excess_m_s2", "anchor_error_m", "limits_new",
                                                            "body_added_m", "scene_added_m") if k in v},
            "deviations": sorted((g.get("deviations") or {}).keys()),
            "series_deviations": sorted(g.get("series_deviations") or []),     # a list (LayeredBuild)
            "module_failed": _failed(tr.get("validity"))}


# ---------------------------------------------------------------- the layer's refusals
def section_layer():
    """The layer's verdict on LAYER_CLIPS (layer_reading)."""
    return {cid: layer_reading(cid) for cid in LAYER_CLIPS}


# ---------------------------------------------------------------- the seated hands and the Gowers reader
LA_CLIPS = ["brady_E6_la_level0"] + [f"brady_E6_la_{c}_g{g}" for c in ("decrement", "interruptions") for g in (1, 2, 3)] \
    + ["brady_E6_la_arrest_g3"]
ANKLE_CLIPS = [f"dystonia_E1_distal_leg_plantarflexion_ankle_g{g}" for g in (1, 2, 3, 4)]
MESH_STEP = 5                     # knots between two mesh readings of the hand-to-thigh gap (0.05 s)
_RENDER = {}


def _render():
    """scripts/render_skeleton_catalogue.py, loaded once (as scripts/skeleton_page_checks.py loads it)."""
    if not _RENDER:
        import importlib.util
        spec = importlib.util.spec_from_file_location("render_skeleton_catalogue_for_phaseE",
                                                      os.path.join(ROOT, "scripts", "render_skeleton_catalogue.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RENDER["m"] = mod
    return _RENDER["m"]


def _positions(body):
    """(N, K, 3) mm and names: the catalogue skeleton of a body on the page's clock and 1 mm grid."""
    R = _render()
    skel = R.catalogue_skeleton(body, R.FPS)
    return R.SK.quantize_mm(skel.positions, 1).astype(np.float64), list(skel.names)


def _rigid_thighs(P, names):
    """{'<hand>_hand_<leg>_thigh': {'mm', 't_s', 'point'}}: the least signed distance of each hand to each thigh on the
    page's rigid stand-in (check_self_contact's reading: a joint centre read SKIN_TOL_MM deeper, so that every point's
    limit is -THIGH_TOL_MM)."""
    R = _render()
    geo = R.contact_geometry()
    ix = {n: i for i, n in enumerate(names)}
    out = {}
    for leg in R.SIDES:
        g = geo[f"thigh_{leg}"]
        hip, knee, ankle, foot = (P[:, ix[f"{leg}_{j}"]] for j in ("hip", "knee", "ankle", "foot"))
        E = R.femur_frame(hip, knee, ankle, foot)
        for s in R.SIDES:
            best = (np.inf, 0, "")
            groups = [([ix[p] for p in R.HAND_JOINTS[s]], False), ([ix[p] for p in R.HAND_SKIN[s]], True)]
            pal = R.palm_skin(P, ix, geo, s)
            for Q, nm, skin in [(P[:, i], [names[j] for j in i], sk) for i, sk in groups] + (
                    [(pal, [f"{s}_palm"] * pal.shape[1], True)] if pal is not None else []):
                L = R._to_local(E, hip, Q)
                span = (L[..., 0] >= R.THIGH_T_MIN * g["length"]) & (L[..., 0] <= g["length"])
                d = np.where(span, R.signed_distance(L, g), np.inf) - (0.0 if skin else R.SKIN_TOL_MM)
                k, m = np.unravel_index(int(d.argmin()), d.shape)
                if d[k, m] < best[0]:
                    best = (float(d[k, m]), int(k), nm[m])
            out[f"{s}_hand_{leg}_thigh"] = {"mm": _r(best[0], 1), "t_s": _r(best[1] / R.FPS), "point": best[2]}
    return out


def _page(body, cid, root=None, truth=None):
    """The page's self-contact check of a body (positions as the export writes them; the clip's mesh touches from its
    truth, touch_readings) and the rigid thigh readings."""
    R = _render()
    P, names = _positions(body)
    touches = R.touch_readings(truth) if truth is not None else {}
    r = R.check_self_contact(P, names, R.contact_geometry(), cid, root, R.FPS, touches or None)
    return {"ok": r["ok"], "contacts": r["contacts"], "gaps": r["gaps"],
            "touches_on_mesh": r.get("touches_on_mesh"), "rigid": _rigid_thighs(P, names)}


def _mesh(body, step=MESH_STEP, sides=("left", "right")):
    """{side: {'min_mm', 't_s'}}: the least signed distance of each hand or forearm to the thighs' skin on the posed
    mesh (postures.arm_part_gap, the modules' reading) every step knots."""
    from mdfm.synth.postures import arm_part_gap
    k = np.unique(np.r_[np.arange(0, body.T, step), body.T - 1])
    out = {}
    for sd in sides:
        g = arm_part_gap(body, sd, k, "thighs")
        i = int(np.argmin(g))
        out[sd] = {"min_mm": _r(1000 * g[i], 2), "t_s": _r(k[i] / body.rate)}
    return out


def _clip(spec, label):
    """A catalogue clip built as the page builds it (spec.build, the layer on): its page and mesh readings."""
    b, tr = spec.build()
    p = tr.get("params") or {}
    return {"build": label, "page": _page(b, spec.clip_id, spec.level0, tr), "mesh": _mesh(b),
            "arms_out_deg": p.get("arms_out_deg"), "arm_lift": _jsonable(p.get("arm_lift")),
            "module_failed": _failed(tr.get("validity"))}


def section_seated_hands():
    """The seated hands placed into the open body's thighs (section 20.7): the leg agility ladder (exam.py on
    bradykinesia.arms_out: its level 0 on the raw generator at the committed LEG_AGILITY_ARMS_OUT_DEG, the grid's
    step below and 0, the former body; then the eight clips as the page builds them), the plantarflexed ankle ladder of
    DIS (dystonia.arm_lift in dystonic_posture) and the fixed fist (postures.distribution) as the page builds them,
    each at the committed code and with arm_lift held off (the former bodies): the page's self-contact reading (its
    rigid thighs) and the hand-to-thigh gap on the posed mesh."""
    from mdfm.synth import bradykinesia as BR, dystonia as D, exam as EX
    from mdfm.synth import catalogue as C
    out = {"leg_agility_level0_raw": {}}
    for deg in (BR.LEG_AGILITY_ARMS_OUT_DEG, BR.LEG_AGILITY_ARMS_OUT_DEG - 5.0, 0.0):
        base = BR.arms_out(EX._static("sit", EX._brady_duration("leg_agility", "level0", 0)), deg)
        b, tr = BR.bradykinesia(base, "leg_agility", "left", decrement_onset=None, seed=EX.SEED)
        out["leg_agility_level0_raw"][f"{deg:g}"] = {"page": _page(b, "brady_E6_la_level0"), "mesh": _mesh(b, 2),
                                                     "amplitude0_deg": tr["params"]["amplitude0_deg"]}
    specs = catalogue()
    out["committed"] = {cid: _clip(specs[cid], "committed") for cid in LA_CLIPS + ANKLE_CLIPS + ["post_E1_fixed_fist"]}
    keep = D.arm_lift
    try:                                             # the former bodies: no lift (fresh specs, a fresh layer cache)
        D.arm_lift = lambda body, weight, sides=("left", "right"), **kw: (body, {"sides": {sd: {"deg": 0.0}
                                                                                          for sd in sides}})
        fresh = {s.clip_id: s for s in C.catalogue_set()}
        out["former"] = {cid: _clip(fresh[cid], "arm_lift off") for cid in ANKLE_CLIPS[2:] + ["post_E1_fixed_fist"]}
    finally:
        D.arm_lift = keep
    return out


def section_sitdown():
    """The controlled sit-down's folded arms (parkinsonism.sitdown_fold_turn): park_E22_sitdown_level0 built at the
    committed SITDOWN_FOLD_TURN_DEG, the grid's step below and 0 (the former body): the fold over the descent
    (motions.fold_report, the arising's rule, the truth's 'fold'), the joint-limit table, the page's self-contact
    reading; the en bloc sit-down (no added lean, unchanged) beside it."""
    from mdfm.synth import parkinsonism as PK
    from mdfm.synth.body import limit_violations
    keep = PK.sitdown_fold_turn.__defaults__
    out = {}
    try:
        for deg in (PK.SITDOWN_FOLD_TURN_DEG, PK.SITDOWN_FOLD_TURN_DEG - 1.0, 0.0):
            PK.sitdown_fold_turn.__defaults__ = (deg,)
            b, tr = PK.sitdown("controlled")
            f = tr["validity"]["fold"]
            out[f"controlled_{deg:g}"] = {
                "fold_entry_ok": f["entry_ok"],
                "fold_min_mm": {c: _r(v["min_mm"], 2) for c, v in f.items() if isinstance(v, dict) and "min_mm" in v},
                "limits_outside": limit_violations(b.rot6d)["outside"], "module_failed": _failed(tr["validity"]),
                "page": _page(b, "park_E22_sitdown_level0")}
    finally:
        PK.sitdown_fold_turn.__defaults__ = keep
    b, tr = PK.sitdown("en_bloc")
    f = tr["validity"]["fold"]
    out["en_bloc"] = {"fold_entry_ok": f["entry_ok"], "module_failed": _failed(tr["validity"]),
                      "fold_min_mm": {c: _r(v["min_mm"], 2) for c, v in f.items() if isinstance(v, dict) and
                                      "min_mm" in v}, "page": _page(b, "park_E22_sitdown_enbloc",
                                                                    "park_E22_sitdown_level0")}
    return out


def section_gowers_reader():
    """Gowers' manoeuvre's declared touch (render_skeleton_catalogue.MESH_TOUCHES): park_E7_gowers as the page builds
    it, its truth's mesh reading of the planted hands (validity 'hand_thigh' 'touch_mm', measure_gowers) and the page's
    self-contact check read with it (committed) and without it (the rigid thigh's gap, the former reading)."""
    s = catalogue()["park_E7_gowers"]
    b, tr = s.build()
    v = tr["validity"]["hand_thigh"]
    R = _render()
    P, names = _positions(b)
    rigid = R.check_self_contact(P, names, R.contact_geometry(), s.clip_id, s.level0, R.FPS)
    return {"touch_mm": v.get("touch_mm"), "gap_min_m": v["gap_min_m"], "module_failed": _failed(tr["validity"]),
            "committed": _page(b, s.clip_id, s.level0, tr),
            "former_rigid": {"ok": rigid["ok"], "gaps": rigid["gaps"], "contacts": rigid["contacts"]}}


def _pnkd_reading(spec):
    """The PNKD excerpt built raw and with the layer: the 'hand_thigh' and 'hand_trunk' checks of each (its truth),
    the thigh guard's rounds, the module checks that fail."""
    raw, truth = spec.build.raw()
    b, tr = spec.build.with_background(raw, truth)
    keep = ("hand_thigh", "hand_trunk", "armrest", "backrest")
    return {"raw": {k: _jsonable(truth["validity"].get(k)) for k in keep},
            "layered": {k: _jsonable(tr["validity"].get(k)) for k in keep},
            "thigh_guard": [{sd: _jsonable(r.get(sd + "_thigh")) for sd in ("left", "right")}
                            for r in truth["params"]["trunk_guard"]],
            "raw_failed": _failed(truth["validity"]), "layered_failed": _failed(tr["validity"])}


def section_pnkd():
    """The PNKD excerpt's hand on the thigh (postures.paroxysmal_excerpt's thigh guard, PNKD_THIGH_CLEAR_M): built raw
    and with the layer at the committed clearance and with the guard off (the former body, the clearance at -1 m: no
    knot needs it), the scene checks of each body."""
    from mdfm.synth import catalogue as C, postures as PO
    out = {"committed": _pnkd_reading(catalogue()["post_E1_pnkd_excerpt"])}
    keep = PO.PNKD_THIGH_CLEAR_M
    try:
        PO.PNKD_THIGH_CLEAR_M = -1.0
        fresh = {s.clip_id: s for s in C.catalogue_set()}
        out["former"] = _pnkd_reading(fresh["post_E1_pnkd_excerpt"])
    finally:
        PO.PNKD_THIGH_CLEAR_M = keep
    return out


# ---------------------------------------------------------------- spacer: the involuntary and gait clips follow


# ---------------------------------------------------------------- the involuntary and gait clips
def renderer():
    """scripts/render_skeleton_catalogue.py loaded as a module (once): the page's checks and its skeleton. The same
    module as _render's: a second cache of its own, rebinding _RENDER under another key, made the sections of the
    seated hands fail with a KeyError when they followed one of these in one process."""
    return _render()


_PAGE_CACHE = {}
_RAW_POINTS = {}                 # {clip: the raw body's page points} of the last page_points of each clip


def page_points(cid, layered=True):
    """(positions (N, K, 3) mm on the export's 1 mm grid, names, truth, body rate, floor m) of a catalogue clip as the
    page draws it: spec.build() (raw and the background layer, as export_clip) or the raw body, through the page's
    catalogue skeleton at its clock; the floor by clip_meta's 'standing' rule (the catalogue export's)."""
    from mdfm.synth import skeleton as SK
    R = renderer()
    s = catalogue()[cid]
    body, truth = s.build.raw()                      # spec.build() is the raw build and then with_background
    _RAW_POINTS[cid] = SK.quantize_mm(R.catalogue_skeleton(body, R.FPS).positions, 1).astype(np.float64)
    if layered:
        body, truth = s.build.with_background(body, truth)
    skel = R.catalogue_skeleton(body, R.FPS)
    floor = float(R.clip_meta(s, truth, body, skel, R.SET_CATALOGUE, "standing")["floor"]["floor_y_m"])
    return SK.quantize_mm(skel.positions, 1).astype(np.float64), list(skel.names), truth, float(body.rate), floor


def level0_points(cid):
    """page_points of a level 0, read once."""
    if cid not in _PAGE_CACHE:
        _PAGE_CACHE[cid] = page_points(cid)
    return _PAGE_CACHE[cid]


def page_reading(cid, root, checks=("continuity", "self_contact", "skate")):
    """The page's skeleton checks of one clip built now (page_points) on its level 0's floor and rate readings
    (skeleton_checks' rules: the root's floor, rates_over against the root's rate_peaks), the rule texts left out;
    with the skate check, its contacts over SKATE_MM (skate_contact_list); the module checks the build failed."""
    R = renderer()
    P, names, truth, rate, _ = page_points(cid)
    P0, n0, t0, _, floor0 = level0_points(root)
    floor_mm = 1000.0 * floor0
    geo = R.contact_geometry()
    Pf, nf = R.with_sole_points(P, names, geo)
    ev, sl = R.event_spans(truth), R.slide_spans(truth, rate)
    out = {}
    if "continuity" in checks:
        rates0 = R.rate_peaks(P0, n0, R.FPS, R.event_spans(t0))
        out["continuity"] = R.check_continuity(Pf, nf, R.FPS, ev, floor_mm, geo["foot_rest_mm"], rates0, cid, sl)
    if "self_contact" in checks:
        out["self_contact"] = R.check_self_contact(P, names, geo, cid, root, R.FPS)
    if "skate" in checks:
        out["skate"] = R.check_skate(Pf, nf, R.FPS, floor_mm, R.skate_rest_mm(geo), cid, R.hold_spans(truth), sl,
                                     geo)
        out["skate_contacts_over_skate_mm"] = [r for r in R.skate_contact_list(P, names, R.FPS, floor_mm, geo, sl)
                                               if r["travel_mm"] > R.SKATE_MM]
    for v in out.values():
        if isinstance(v, dict):
            v.pop("rule", None)
    out["module_failed"] = _failed(truth.get("validity"))
    return out, (P, names, truth)


def thumb_reading(P, names, fps=30.0):
    """The two thumb tip landmarks' nearest approach on the page's points (mm, the time s)."""
    ix = {n: i for i, n in enumerate(names)}
    D = np.linalg.norm(P[:, ix["left_thumb_tip"]] - P[:, ix["right_thumb_tip"]], axis=-1)
    k = int(D.argmin())
    return {"thumb_tips_min_mm": _r(D[k], 2), "t_s": _r(k / fps, 3)}


def _arm_contact(truth):
    a = (truth.get("validity") or {}).get("arm_contact")
    return None if a is None else {"ok": a["ok"], "after_cm_max": {k: _r(v, 4) for k, v in a["after_cm_max"].items()},
                                   "read_after": a["read_after"]}


SEATED_THUMB_CLIPS = ("invol_chorea_E1_legs_g2", "invol_chorea_E1_legs_g4", "invol_chorea_E1_g3", "invol_chorea_E1_g4")


def _thumb_thigh_mm(P, names, side="left"):
    """The page's deepest reading of a thumb tip landmark inside the same side's thigh (mm, negative inside; the
    self-contact check's rigid thigh over its span of the femur) and its time."""
    R = renderer()
    geo = R.contact_geometry()
    ix = {n: i for i, n in enumerate(names)}
    g = geo[f"thigh_{side}"]
    hip, knee, ankle, foot = (P[:, ix[f"{side}_{j}"]] for j in ("hip", "knee", "ankle", "foot"))
    L = R._to_local(R.femur_frame(hip, knee, ankle, foot), hip, P[:, [ix[f"{side}_thumb_tip"]]])
    span = (L[..., 0] >= R.THIGH_T_MIN * g["length"]) & (L[..., 0] <= g["length"])
    d = np.where(span, R.signed_distance(L, g), np.inf)[:, 0]
    k = int(d.argmin())
    return {"mm": _r(d[k], 2), "t_s": _r(k / R.FPS, 3)}


def section_seated_thumbs():
    """The seated hands' thumb rule (involuntary.hand_room's thumbs, THUMB_ROOM_M): the seated leg chorea grades 2
    and 4 (refused by the page's self-contact check, the thumb tips within SKIN_TOL_MM), the seated chorea grade 3
    (refused by its arm_contact) and grade 4 (the clip the 20 mm try refuses), each built with the committed rule, with
    the wrists' rule alone (the former hand_room: thumbs ignored) and at THUMB_ROOM_M 20 mm (the try not kept): the
    page's self-contact check, the thumb tips' nearest approach, the left thumb's deepest reading in the left thigh
    (layered as the page draws it, and raw), the arms' contact entry, the module checks the build fails."""
    from mdfm.synth import involuntary as I
    out = {"thumb_room_m": I.THUMB_ROOM_M}
    orig, keep = I.hand_room, I.THUMB_ROOM_M
    for label in ("committed", "former", "tried_20mm"):
        if label == "former":
            I.hand_room = lambda body, base, W_left, W_right, approach=I.HAND_APPROACH_M, soft=I.HAND_ROOM_SOFT_M,                 thumbs=None, thumb_room=I.THUMB_ROOM_M: orig(body, base, W_left, W_right, approach, soft)
        if label == "tried_20mm":
            I.THUMB_ROOM_M = 0.020
        try:
            for cid in SEATED_THUMB_CLIPS:
                r, (P, names, truth) = page_reading(cid, "invol_E1_level0", ("self_contact",))
                Pr, nr = _RAW_POINTS[cid], names
                out.setdefault(cid, {})[label] = {"self_contact": r["self_contact"], **thumb_reading(P, names),
                                                  "left_thumb_in_left_thigh": _thumb_thigh_mm(P, names),
                                                  "left_thumb_in_left_thigh_raw": _thumb_thigh_mm(Pr, nr),
                                                  "arm_contact": _arm_contact(truth), "module_failed": r["module_failed"]}
        finally:
            I.hand_room, I.THUMB_ROOM_M = orig, keep
    return out


HEMIBALLISM_CLIPS = ("invol_hemiballism_E1_left_g1", "invol_hemiballism_E1_left_g2", "invol_hemiballism_E1_left_g3")


def section_hemiballism_kick():
    """The hemiballism kicks' speed bound (involuntary.BALLISM_KICK_SPEED): the three hemiballism clips built at the
    committed bound, with no bound (the former kicks) and at the values tried (3.0, 3.7): the page's continuity check
    (the upper arm's peak turn outside the declared ballistic intervals, against RATE_CAPS_DEG_S and the level 0's)."""
    from mdfm.synth import involuntary as I
    keep = I.BALLISM_KICK_SPEED
    out = {"committed": keep}
    tries = (("committed", keep), ("former", float("inf")), ("tried_3.0", 3.0), ("tried_3.7", 3.7))
    try:
        for label, v in tries:
            I.BALLISM_KICK_SPEED = v
            for cid in (HEMIBALLISM_CLIPS if label in ("committed", "former") else HEMIBALLISM_CLIPS[:1]):
                r, _ = page_reading(cid, "invol_E1_level0", ("continuity",))
                c = r["continuity"]
                out.setdefault(cid, {})[label] = {k: c[k] for k in ("ok", "rates_over", "rates_deg_s", "accel_max_m_s2",
                                                                     "accel_point")}
                out[cid][label]["module_failed"] = r["module_failed"]
    finally:
        I.BALLISM_KICK_SPEED = keep
    return out


def _gait_validity(truth):
    v = truth.get("validity") or {}
    return {k: v[k] for k in ("limits_outside", "limits_new", "limits_worst_deg", "stance_slip_p95_m_s",
                              "stance_slip_max_m_s", "swing_lowest_m") if k in v}


def section_nph_turn():
    """The magnetic NPH gait's turn (gait.NPH_PIVOT_SHARE): the clip built with its planted feet not pivoting (the
    committed 0) and pivoting through 0.9 of the heading change (GaitParams' default, the former value): the page's
    skate check on the level 0's floor (its turning contacts and the contacts over SKATE_MM), the continuity check,
    the gait validity's limits and stance slip, the module checks it fails, the turn's read-back."""
    from mdfm.synth import gait as G
    keep = G.NPH_PIVOT_SHARE
    out = {"committed": keep, "former": 0.9}
    try:
        for label, v in (("committed", keep), ("former", 0.9)):
            G.NPH_PIVOT_SHARE = v
            r, (P, names, truth) = page_reading("gait_E8_nph_magnetic", "gait_E8_level0", ("continuity", "skate"))
            sk = r["skate"]
            out[label] = {"skate": {k: sk[k] for k in ("ok", "travel_max_mm", "point", "span_s", "tol_mm", "contacts",
                                                       "contacts_over", "turning", "straight")},
                          "skate_contacts_over_skate_mm": r["skate_contacts_over_skate_mm"],
                          "continuity": {k: r["continuity"][k] for k in ("ok", "accel_max_m_s2", "accel_point",
                                                                         "foot_jump_max_mm", "rates_over")},
                          "gait_validity": _gait_validity(truth), "module_failed": r["module_failed"],
                          "course": (truth.get("params") or {}).get("course")}
    finally:
        G.NPH_PIVOT_SHARE = keep
    return out


def _fog_generate_reading():
    """The fog shuffling clip's gait alone (gait.fog_shuffling through gait.generate on the clip's seed, no read-back,
    no layer) read by the page's acceleration rule (second differences of the page's points at 30 fps): its largest
    readings, gait.validity's verdicts, the episodes."""
    from mdfm.synth import gait as G
    from mdfm.synth import skeleton as SK
    R = renderer()
    p, eps, tm = G.fog_shuffling()
    b, info = G.generate(p, duration_s=tm["duration_s"], start_at_s=tm["lead_s"], stop_at_s=tm["stop_at_s"],
                         turns=tm["turns"], seed=G.GAP_SEED + 7)
    P = SK.quantize_mm(R.catalogue_skeleton(b, R.FPS).positions, 1).astype(np.float64)
    names = list(R.catalogue_skeleton(b, R.FPS).names)
    acc = np.zeros(P.shape[:2])
    acc[1:-1] = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1) * R.FPS * R.FPS / 1000.0
    top = np.argsort(acc.max(1))[::-1][:4]
    v = G.validity(b, info)
    return {"limits_outside": v["limits_outside"], "limits_worst_deg": _r(v["limits_worst_deg"], 4),
            "accel_top": [{"t_s": _r(k / R.FPS, 3), "m_s2": _r(acc[k].max(), 1), "point": names[int(acc[k].argmax())]}
                          for k in top],
            "gait_validity_failed": _failed(v),
            "episodes_s": [[_r(e["onset_s"], 3), _r(e["offset_s"], 3)] for e in eps]}


def section_fog_shuffling():
    """The freezing by shuffling's push-off at the episodes' onsets (gait.FOG_PUSH_OFF_ONSET_POWER, gait.onset_fn):
    the clip as committed and with the former easing (power 1: the push-off on the envelope itself at both edges),
    each built as the page builds it (raw and layer) and read by the page's continuity and skate checks on its level
    0's floor and rate readings, with the gait validity, the module checks and the episodes' read-backs."""
    from mdfm.synth import gait as G
    keep = G.FOG_PUSH_OFF_ONSET_POWER
    out = {"committed": keep, "former": 1.0}
    try:
        for label, v in (("committed", keep), ("former", 1.0)):
            G.FOG_PUSH_OFF_ONSET_POWER = v
            r, (P, names, truth) = page_reading("gait_E8_fog_shuffling_g3", "gait_E8_parkinsonian_g1",
                                                ("continuity", "skate"))
            c = r["continuity"]
            out[label + "_page"] = {"continuity": {k: c[k] for k in ("ok", "accel_max_m_s2", "accel_point",
                                                                     "accel_t_s", "accel_frames_over",
                                                                     "foot_jump_max_mm", "rates_over")},
                                    "skate": {k: r["skate"][k] for k in ("ok", "travel_max_mm", "tol_mm",
                                                                         "contacts_over")},
                                    "gait_validity": _gait_validity(truth), "module_failed": r["module_failed"],
                                    "episodes": [{k: e.get(k) for k in ("context", "onset_s", "offset_s", "read")}
                                                 for e in (truth.get("params") or {}).get("episodes") or []]}
    finally:
        G.FOG_PUSH_OFF_ONSET_POWER = keep
    return out


def section_fog_shuffling_tries():
    """The freezing by shuffling's tries not kept, read on the gait alone (_fog_generate_reading: the page's
    acceleration rule on the points without the read-back and the layer, gait.validity's limits and verdicts), each
    against the committed gait and the former one read the same way: the cadence eased on the cubed envelope and the
    lift on the squared one (FOG_POWER, the push-off on the former envelope), and the push-off eased as the step
    length at both edges (gait.onset_fn replaced by episodes_fn at the powers 0.5 and 0.7)."""
    from mdfm.synth import gait as G
    keep, keep_power, keep_fn = G.FOG_PUSH_OFF_ONSET_POWER, dict(G.FOG_POWER), G.onset_fn
    out = {}
    try:
        out["committed"] = _fog_generate_reading()
        G.FOG_PUSH_OFF_ONSET_POWER = 1.0
        out["former"] = _fog_generate_reading()
        for label, v in (("cadence_power_3", dict(keep_power, cadence_spm=3.0)),
                         ("lift_power_2", dict(keep_power, clearance_m=2.0, knee_swing_deg=2.0))):
            G.FOG_POWER.clear()
            G.FOG_POWER.update(v)
            out[label] = dict(_fog_generate_reading(), fog_power=v)
        G.FOG_POWER.clear()
        G.FOG_POWER.update(keep_power)
        for pw in (0.5, 0.7):
            G.onset_fn = lambda outside, inside, iv, ramp_s, power, pw=pw: G.episodes_fn(outside, inside, iv,
                                                                                         ramp_s, pw)
            out[f"push_off_both_edges_{pw:g}"] = dict(_fog_generate_reading(), push_off_power=pw)
    finally:
        G.FOG_PUSH_OFF_ONSET_POWER, G.onset_fn = keep, keep_fn
        G.FOG_POWER.clear()
        G.FOG_POWER.update(keep_power)
    return out


def section_moving_toes():
    """The moving toes clip the skate check refuses (left open): its contact read by the page's rule, by the named
    points and by the skin points on their standing frames, and which sole points carry it (within SKATE_TOUCH_MM
    of the foot's lowest standing skin point on SKATE_POINT_SHARE of the contact), against the level 0's."""
    R = renderer()
    out = {}
    for cid in ("invol_moving_toes_E1_left_a3", "invol_E1_level0"):
        P, names, truth, rate, _ = page_points(cid)
        geo = R.contact_geometry()
        floor_mm = 1000.0 * level0_points("invol_E1_level0")[4]
        rows = [r for r in R.skate_contact_list(P, names, R.FPS, floor_mm, geo) if r["side"] == "left"]
        Pf, nf = R.with_sole_points(P, names, geo)
        ix = {n: i for i, n in enumerate(nf)}
        fr = R.skate_rest_mm(geo)
        nm = [p for p in fr if p.startswith("left_") and p in ix]
        X = np.stack([Pf[:, ix[p]] for p in nm], 1)
        on = R._standing_on_floor(X, np.array([fr[p] for p in nm]), floor_mm, R.FPS)
        C = R.sole_cloud(Pf, ix, geo, "left")
        g = geo["foot_left"]
        skin = [k for k, p in enumerate(nm) if p not in R.SK.JOINT_NAMES]
        Y = np.concatenate([X[:, skin], C], 1)
        Ys = np.concatenate([on[:, skin], R._standing_on_floor(C, g["cloud_rest_mm"], floor_mm, R.FPS)], 1)
        low = np.where(Ys, Y[..., 1], np.inf).min(1, keepdims=True)
        Yc = Ys & (Y[..., 1] <= low + R.SKATE_TOUCH_MM)
        pts = [nm[k] for k in skin] + [f"sole_{r}" for r in g["cloud_region"]]
        share = Yc.mean(0)
        trav = R._masked_travel(Y, Yc)
        regions = {}
        for reg in sorted(set(pts)):
            sel = [k for k, p in enumerate(pts) if p == reg]
            car = [k for k in sel if share[k] >= R.SKATE_POINT_SHARE]
            regions[reg] = {"points": len(sel), "carrying": len(car),
                            "least_travel_carrying_mm": _r(min(trav[k] for k in car), 2) if car else None,
                            "height_above_lowest_median_mm": _r(np.median(Y[:, sel, 1] - low, 0).min(), 2)}
        ixs = {n: i for i, n in enumerate(names)}
        joints = {n: _r(float(np.ptp(P[:, ixs[n]], 0).max()), 1) for n in ("left_ankle", "left_foot", "left_heel")}
        out[cid] = {"contacts": rows, "sole_regions": regions, "joint_range_mm": joints}
    return out


# ---------------------------------------------------------------- spacer: the layer's own rules follow


# ---------------------------------------------------------------- the layer's own rules
def _physio():
    from mdfm.synth import physio as P
    return P


def _excess(base, out):
    """(largest acceleration the layer adds beyond ACCEL_REL of the body's own, m/s2; its knot; its joint)."""
    P = _physio()
    Jb, Jo = base.joints().numpy(), out.joints().numpy()
    r2 = float(base.rate) ** 2
    e = (np.linalg.norm(np.diff(Jo - Jb, 2, axis=0), axis=-1) * r2
         - P.ACCEL_REL * np.linalg.norm(np.diff(Jb, 2, axis=0), axis=-1) * r2)
    k, j = np.unravel_index(int(np.argmax(e)), e.shape)
    return {"accel_excess_m_s2": float(e.max()), "at_s": (int(k) + 1) / float(base.rate), "joint": int(j)}


def _member(cid):
    """(spec, raw body, truth, level 0 raw, level 0 layered, layer, transfer keywords) of a member clip."""
    P = _physio()
    s = catalogue()[cid]
    raw, truth = s.build.raw()
    ref_raw, ref_out, _ = s.build.cache.layered(s.level0)
    c = s.build.cache.context(s.build.root_id)
    layer = P.make_layer(c["seed"], c["kind"], float(raw.rate), quiet=c["quiet"])
    kw = dict(scene=s.scene, table=P.table_of(truth), kind=c["kind"], resting=c["resting"], reference=s.level0)
    return s, raw, truth, ref_raw, ref_out, layer, kw


def _verdict(v):
    keys = ("accel_excess_m_s2", "anchor_error_m", "anchor_error_moving_m", "anchor_error_dwell_m", "limits_new",
            "limits_worst_deg", "foot_moved_m", "body_added_m", "scene_added_m", "legs_added_m", "skate_added_m_s")
    return {"ok": bool(v.get("ok")), "failed": list(v.get("failed") or []), **{k: v[k] for k in keys if k in v}}


def _same(a, b):
    from mdfm.synth.hands import has_hands
    return bool(torch.equal(a.rot6d, b.rot6d) and torch.equal(a.transl, b.transl)
                and (has_hands(a) == has_hands(b)) and (not has_hands(a) or torch.equal(a.hand6d, b.hand6d)))


def section_raise():
    """The raise after the longer rest (section 20.7's first refusal): the series root tremor_E2_raise_long_rest_g0
    laid with apply() four ways (the default anchoring, the rest gate, the damped elbow step, both: physio.solve's
    fallbacks; physio-v9's solve kept a fallback only where it failed fewer checks, so that the gate, failing as many
    as the default, was not kept and the damped step was tried on the default alone), each with its failed checks and
    the largest added acceleration (time, joint); then solve()'s verdict as committed and the layer's verdict on the
    root and its two tremor clips (layer_reading)."""
    P = _physio()
    root = "tremor_E2_raise_long_rest_g0"
    s = catalogue()[root]
    raw, truth = s.build.raw()
    c = s.build.cache.context(root, raw, truth)
    layer = P.make_layer(c["seed"], c["kind"], float(raw.rate), quiet=c["quiet"])
    kw = dict(scene=s.scene, table=P.table_of(truth), resting=c["resting"], kind=c["kind"], labels=c["labels"])
    out = {"apply": {}}
    for name, extra in (("default", {}), ("rest_gate", dict(rest_gate=True)),
                        ("damped", dict(arm_damp=P.ARM_FIX_DAMP_M)),
                        ("rest_gate_and_damped", dict(rest_gate=True, arm_damp=P.ARM_FIX_DAMP_M))):
        o, r = P.apply(raw, layer, **kw, **extra)
        out["apply"][name] = {**_verdict(r["validity"]), **{"largest": _excess(raw, o)}}
    o, r = P.solve(raw, layer, **kw)
    out["solve"] = {**_verdict(r["validity"]), "anchor_rest_gate": bool(r["guard"].get("anchor_rest_gate")),
                    "anchor_damped": bool(r["guard"].get("anchor_damped"))}
    out["layer"] = {cid: layer_reading(cid) for cid in (root, "tremor_E2_raise_postural_g2",
                                                         "tremor_E2_raise_reemergent_g2")}
    return out


def section_legs():
    """The functional axial jerk standing (sev2, dur2; section 20.7) and the standing chorea g2b: each member's
    transfer (physio.transfer) with a member's own pelvis keeping its legs' lengths under the level 0's sway
    (LEG_LENGTHS, its ramp from a held row set by LEG_LENGTHS_ACCEL_M_S2, committed), with fixed ramps
    (LEG_LENGTHS_FIXED_RAMP_S 0.05 and 0.2 s, the tries) and without the rule (physio-v9, LEG_LENGTHS False): the
    verdict, the largest added acceleration, the guard's shift; without the rule, the left leg's hip-to-ankle distance
    in the clip and in the layered body at the knot of the largest excess."""
    P = _physio()
    keep = (P.LEG_LENGTHS, P.LEG_LENGTHS_FIXED_RAMP_S)
    out = {}
    for cid in ("jerk_functional_E7_axial_sev2_dur2", "invol_chorea_E7_g2b"):
        s, raw, truth, ref_raw, ref_out, layer, kw = _member(cid)
        own = lambda: P.solve(raw, layer, **{k: v for k, v in kw.items() if k != "reference"})  # noqa: E731
        out[cid] = {}
        try:
            for name, flag, ramp in (("committed", True, None), ("fixed_ramp_0.05", True, 0.05),
                                     ("fixed_ramp_0.2", True, 0.2), ("former", False, None)):
                P.LEG_LENGTHS, P.LEG_LENGTHS_FIXED_RAMP_S = flag, ramp
                o, r = P.transfer(raw, ref_raw, ref_out, layer, own=own, **kw)
                x = _excess(raw, o)
                rec = {**_verdict(r["validity"]), "largest": x,
                       "leg_lengths_knots": r["guard"].get("leg_lengths_knots", 0),
                       "leg_lengths_shift_m": r["guard"].get("leg_lengths_shift_m", 0.0),
                       "deviations": sorted(r["deviations"]), "fixed_ramp_s": ramp, "leg_lengths": flag}
                if name == "former":
                    k = int(round(x["at_s"] * float(raw.rate)))
                    Jb, Jo = raw.joints(), o.joints()
                    rec["left_hip_ankle_m"] = {"at_s": x["at_s"], "clip": float((Jb[k, 7] - Jb[k, 1]).norm()),
                                               "layered": float((Jo[k, 7] - Jo[k, 1]).norm())}
                out[cid][name] = rec
        finally:
            P.LEG_LENGTHS, P.LEG_LENGTHS_FIXED_RAMP_S = keep
    return out


def section_trunk_guard():
    """The tardive rocking s4 (section 20.7): the member's transfer with its own limit guard on the trunk and legs
    before the arms' anchoring (TRUNK_GUARD_FIRST, committed) and after it (physio-v9, False): the verdict, each
    hand's anchor error, the knot of the largest and the wrist's distance from its goal there before the arms'
    correction (transfer's trace), the guard's joints."""
    P = _physio()
    cid = "invol_td_E1_rocking_s4"
    s, raw, truth, ref_raw, ref_out, layer, kw = _member(cid)
    keep = P.TRUNK_GUARD_FIRST
    out = {}
    calls = []
    orig = P._anchor_reading

    def spy(err, *a, **k):
        res = orig(err, *a, **k)
        calls.append(np.asarray(err))
        return res
    try:
        P._anchor_reading = spy
        for name, flag in (("committed", True), ("former", False)):
            P.TRUNK_GUARD_FIRST = flag
            calls.clear()
            trace = {}
            o, r = P._transfer(raw, ref_raw, ref_out, layer, _trace=trace, **kw)
            sides = {}
            for i, side in enumerate(trace):
                err = calls[-len(trace) + i]
                k = int(np.argmax(np.where(np.asarray(trace[side]["touch"]), err, -1.0)))
                sides[side] = {"anchor_error_m": r["anchors"][side]["anchor_error_m"], "largest_at_s": k / 100.0,
                               "wrist_from_goal_m": float(err[k]),
                               "before_arm_correction_m": float(trace[side]["err"][k])}
            out[name] = {**_verdict(r["validity"]), "sides": sides, "guard_limits": r["guard"]["limits"],
                         "trunk_guard_first": flag}
    finally:
        P.TRUNK_GUARD_FIRST = keep
        P._anchor_reading = orig
    return out


def section_chorea_g4():
    """The choreic fingers of E2 grade 4 (section 20.7: limits_new 2, 2.22 m/s2 and the anchor 2.29 mm on physio-v9):
    the member's transfer variants (physio.transfer's default and its fallbacks: the ramps and the rest gate), each
    with its failed checks, the largest added acceleration and the anchor errors, on the committed code; the rest
    gate's variant with the own limit guard smoothed over each of GUARD_SMOOTH_LONGER_S (physio-v10's last fallback);
    transfer()'s choice and verdict as committed (over GUARD_SMOOTH_S alone, the variants above at that smoothing:
    physio-v9's choice, the default)."""
    P = _physio()
    cid = "invol_chorea_E2_fingers_g4"
    s, raw, truth, ref_raw, ref_out, layer, kw = _member(cid)
    out = {}
    for ramp, smooth, rest_s, binds in ((P.ARM_FIX_SMOOTH_S, P.ARM_FIX_SMOOTH_S, 0.0, True),
                                        (P.GUARD_SMOOTH_S, P.ARM_FIX_SMOOTH_S, 0.0, False),
                                        (P.ARM_FIX_FAST_S, P.ARM_FIX_FAST_S, 0.0, False),
                                        (P.ARM_FIX_SMOOTH_S, P.ARM_FIX_SMOOTH_S, P.ARM_REST_S, False),
                                        (P.GUARD_SMOOTH_S, P.ARM_FIX_SMOOTH_S, P.ARM_REST_S, False),
                                        (P.ARM_FIX_FAST_S, P.ARM_FIX_FAST_S, P.ARM_REST_S, False)):
        o, r = P._transfer(raw, ref_raw, ref_out, layer, arm_ramp_s=ramp, arm_smooth_s=smooth, rest_s=rest_s,
                           dwell_binds=binds, **kw)
        out[f"ramp{ramp:g}_smooth{smooth:g}_rest{rest_s:g}"] = {
            **_verdict(r["validity"]), "largest": _excess(raw, o), "arms_fixed_knots": r["guard"]["arms_fixed_knots"],
            "guard_limits": r["guard"]["limits"]}
    # the rest gate's variant with the own limit guard smoothed over GUARD_SMOOTH_LONGER_S (transfer's last fallback)
    for gs in P.GUARD_SMOOTH_LONGER_S:
        o, r = P._transfer(raw, ref_raw, ref_out, layer, rest_s=P.ARM_REST_S, dwell_binds=False, guard_smooth_s=gs,
                           **kw)
        out[f"rest{P.ARM_REST_S:g}_guard_smooth{gs:g}"] = {**_verdict(r["validity"]), "largest": _excess(raw, o),
                                                            "guard_limits": r["guard"]["limits"]}
    own = lambda: P.solve(raw, layer, **{k: v for k, v in kw.items() if k != "reference"})  # noqa: E731
    o, r = P.transfer(raw, ref_raw, ref_out, layer, own=own, **kw)
    out["transfer"] = {**_verdict(r["validity"]), "largest": _excess(raw, o),
                       **{k: r["guard"].get(k) for k in ("arm_rest_s", "arm_ramp_s", "arm_smooth_s",
                                                         "limits_smooth_s")}}
    return out


def section_holmes():
    """The rest gate on a moving hand (tremor_E3_holmes_g3, section 18.8): the clip's record (its rest gate, the
    knots its own anchoring turns on and those of them where the gate's weight lies below 1, transfer's
    arms_fixed_moving_knots) and, at each of the latter, the wrist's speed relative to its support against
    ARM_REST_M_S, the largest within 0.2 s, the gate's weight; the pair's change as a share of the sign."""
    P = _physio()
    from mdfm.body.joints import fk
    from mdfm.synth.viewer import scene_floor
    cid = "tremor_E3_holmes_g3"
    s, raw, truth, ref_raw, ref_out, layer, kw = _member(cid)
    b, tr = s.build.with_background(raw, truth)
    g = tr["background"]
    out = {"record": {**_verdict(g["validity"]), **{k: g["guard"][k] for k in ("arm_rest_s", "arms_fixed_knots",
                                                                                "arms_fixed_moving_knots")},
                      "pair_share": P.pair_change(raw, b, ref_raw, ref_out)["share"]}}
    trace = {}
    P._transfer(raw, ref_raw, ref_out, layer, rest_s=g["guard"]["arm_rest_s"], _trace=trace, **kw)
    boxes = P.scene_boxes(s.scene, raw, scene_floor(raw, 5.0)[0]) if s.scene else []
    labels = P.support_labels(raw, boxes, P.support_joints(kw["kind"]) + (P.TRUNK_TOUCH if kw["kind"] != "standing"
                                                                           else ()), approach=False)
    J_in, G_in = fk(raw.local_R(), raw.transl, raw.rest())
    n_s = max(1, int(round(P.ARM_FIX_SMOOTH_S * float(raw.rate))))
    out["knots"] = {}
    for side, t in trace.items():
        speed = P.wrist_speed(J_in, G_in, labels, side, float(raw.rate))
        probe = P.rest_weight(speed, float(raw.rate), n_s, P.ARM_REST_S)
        ks = np.flatnonzero((t["w"] > 0.0) & (probe < 1.0 - 1e-12))
        out["knots"][side] = [{"at_s": int(k) / float(raw.rate), "weight": float(t["w"][k]),
                               "gate": float(probe[k]), "speed_m_s": float(speed[k]),
                               "speed_max_0p2s_m_s": float(speed[max(0, k - 20):k + 21].max()),
                               "touch": bool(t["touch"][k])} for k in ks]
    out["arm_rest_m_s"] = P.ARM_REST_M_S
    return out


def section_guard_tol():
    """D145's guard at half the checks' tolerance (GUARD_TOL_DEG): which layered clips it changes. A clip changes
    where the guard blends a row of its own solve or transfer (a blended row is held at the new bound; a row the
    guard leaves is unchanged bit for bit) or of a level 0 it transfers from: read on the background sweep's records
    (reports/synth/background, physio-v9, every layered clip: guard trunk, arms and limits) with their level-0
    chains; on the clips of section layer, the layered body built at the committed tolerance and at the former one
    (the guard called with tol body.LIMIT_TOL_DEG, restored), compared bit for bit, with the layer's verdicts."""
    import glob
    P = _physio()
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "synth", "background", "sweep_*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("physio_version") != "physio-v9":
            continue
        recs.update(d["clips"])
    layered = {c: r for c, r in recs.items() if not r.get("skipped")}

    def blends(r):
        g = r.get("guard") or {}
        return sum(sum((g.get(k) or {}).values()) for k in ("trunk", "arms", "limits"))
    memo = {}

    def changed(c):
        if c not in memo:
            memo[c] = None
            l0 = (layered[c].get("level0") or {}).get("id")
            memo[c] = "own" if blends(layered[c]) else (l0 if l0 in layered and l0 != c and changed(l0) else None)
        return memo[c]
    marks = {c: changed(c) for c in sorted(layered)}
    out = {"from_sweep": {"layered": len(layered), "guard_blends": sorted(c for c, v in marks.items() if v == "own"),
                          "through_level0": {c: v for c, v in marks.items() if v and v != "own"}}}
    orig = P.guard_joints

    def former(*a, **k):
        k.setdefault("tol", P.LIMIT_TOL_DEG)
        return orig(*a, **k)
    built = {}
    for cid in ("invol_chorea_E2_fingers_g3", "invol_td_E1_rocking_s4", "tremor_E2_raise_long_rest_g0"):
        s = catalogue()[cid]
        raw, truth = s.build.raw()
        b, tr = s.build.with_background(raw, truth)
        built[cid] = {"committed": _verdict(tr["background"]["validity"])}
        try:
            P.guard_joints = former
            from mdfm.synth import catalogue as C
            s2 = {x.clip_id: x for x in C.catalogue_set()}[cid]          # a fresh series cache
            b2, tr2 = s2.build.with_background(raw, truth)
        finally:
            P.guard_joints = orig
        built[cid].update(former=_verdict(tr2["background"]["validity"]), identical=_same(b, b2))
    out["built"] = built
    return out


def section_affected():
    """The layered members the physio-v10 transfer rules may change (LEG_LENGTHS: every member of a standing or
    transfer series, whose level 0 sways; TRUNK_GUARD_FIRST: the members whose sweep record names an own limit guard
    on a joint other than the arms' and an own arm correction), each built (with_background, committed: the layer's
    verdict) and transferred again with both rules off (physio-v9's transfer), compared bit for bit: 'changed', the
    guard's leg shift, the verdict at the former rules (the level 0 as committed: a level 0 that is itself a member
    reads its own change in its own entry). --only affected; the clip list may be cut with MDFM_PHASEE_PART=i/n or
    named in the file MDFM_PHASEE_CLIPS (one id per line; 'listed' gives the whole list)."""
    import glob
    P = _physio()
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "synth", "background", "sweep_*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("physio_version") == "physio-v9":
            recs.update(d["clips"])
    cat = catalogue()
    ids = []
    for cid, r in sorted(recs.items()):
        if r.get("skipped") or cid not in cat or not cat[cid].level0 or cid == r.get("series"):
            continue
        if (r.get("transfer") or {}).get("own_solve") == "walking":
            continue
        lim = (r.get("guard") or {}).get("limits") or {}
        arms = (r.get("guard") or {}).get("arms_fixed_knots") or {}
        trunk_first = any(int(j) not in P.ARM_JOINTS for j in lim) and any(arms.values())
        if r.get("kind") in ("standing", "transfer") or trunk_first:
            ids.append(cid)
    listed = list(ids)
    named = os.environ.get("MDFM_PHASEE_CLIPS")
    if named:
        with open(named, encoding="utf-8") as fh:
            keep_ids = {x.strip() for x in fh if x.strip()}
        ids = [c for c in ids if c in keep_ids]
    part = os.environ.get("MDFM_PHASEE_PART")
    if part:
        i, n = (int(x) for x in part.split("/"))
        ids = ids[i::n]
    keep = (P.LEG_LENGTHS, P.TRUNK_GUARD_FIRST)
    out = {}
    for cid in ids:
        t0 = time.time()
        s, raw, truth, ref_raw, ref_out, layer, kw = _member(cid)
        b, tr = s.build.with_background(raw, truth)
        g = tr["background"]
        own = lambda: P.solve(raw, layer, **{k: v for k, v in kw.items() if k != "reference"})  # noqa: E731
        try:
            P.LEG_LENGTHS, P.TRUNK_GUARD_FIRST = False, False
            b2, r2 = P.transfer(raw, ref_raw, ref_out, layer, own=own, **kw)
        finally:
            P.LEG_LENGTHS, P.TRUNK_GUARD_FIRST = keep
        out[cid] = {"kind": g.get("kind"), "committed": _verdict(g["validity"]),
                    "former": _verdict(r2["validity"]), "changed": not _same(b, b2),
                    "leg_lengths_knots": g["guard"].get("leg_lengths_knots", 0),
                    "leg_lengths_shift_m": g["guard"].get("leg_lengths_shift_m", 0.0),
                    "seconds": round(time.time() - t0, 1)}
        print(cid, out[cid]["committed"]["failed"], out[cid]["former"]["failed"], out[cid]["changed"], flush=True)
    return {"listed": listed, "clips": out}


# ---------------------------------------------------------------- end of the sections
def sections():
    """{name: function} of the module's section_<name> functions, in the order of the file."""
    return {k[len("section_"):]: v for k, v in globals().items() if k.startswith("section_") and callable(v)}


def main():
    secs = sections()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(secs))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=OUT, help="the output file (default the report of the tree)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(a.threads)
    rec = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            rec = json.load(f)
    rec.setdefault("measured", {})
    for name in (a.only or list(secs)):
        t0 = time.time()
        rec[name] = _jsonable(secs[name]())
        rec["measured"][name] = {"mdfm_commit": _head(), "at": datetime.datetime.now().isoformat(timespec="seconds"),
                                 "seconds": round(time.time() - t0, 1)}
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print("wrote", name, rec["measured"][name], flush=True)
    rec["script"] = "scripts/openbody_phaseE_checks.py"
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
