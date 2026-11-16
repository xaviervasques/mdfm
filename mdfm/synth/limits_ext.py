"""SH2 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md (owner's go of 2026-11-02): a targeted extension of the SP1
joint-limit table, read by the dystonia operators (dystonia.py and the dystonia operators of postures.py); its
shoulder coupling alone is also read by the healthy shrugs (the table 'shrug', below). The healthy table
(body.joint_limits: the 0.1 and 99.9 percentiles of reports/body_store/priors.json, learned on healthy AMASS motion)
stays the table of every level 0 and of every other module. A dystonic posture lies outside healthy
motion by nature, and three joints of that table blocked signs of the symptom catalogue: the shoulder raised with the
arm hanging under it (TWSTRS D), the inverted foot and the curled toes. Only those bounds are widened, each from a
range of motion read in a source (SOURCES), never further.

How a range-of-motion norm maps onto the table (body.py): a component of the table is the axis-angle vector of a
local joint rotation, in degrees, in the parent's frame of the SMPL+H template (x toward the body's left, y up, z
forward at rest); the template's rest (every rotation 0) stands with the feet at a right angle to the shanks, the toes
in line with the feet, the clavicles at rest and the arms horizontal. A rotation by an angle about one axis has that
angle as its only component, so a norm read from the anatomical neutral about the axis the measurement reads bounds
that component directly; for combined rotations the components are the projections of one axis-angle vector, close to
the separate angles at these sizes, not equal to them.
- ankle inversion (WIDEN): the foot (joints 7, 8) turned about the forward axis z of the shank's frame, the sole toward
  the midline (-z on the left, +z on the right, postures.FOOT_DIRS); the template's ankle at 0 is the neutral foot. The
  norm is the inversion of jarm2022, 30 deg, measured in the frontal plane (the sole's plane against the perpendicular
  to the shank's axis, the ankle at 0 deg): a rotation about the forward axis, the z component alone. menadue2006
  corroborates it: 31.5 deg by its anterior (sitting) method, which reads whole-foot supination (plantarflexion,
  inversion and adduction together, seen from the front), so an upper bound on a combined motion and not used as the
  bound (its rearfoot method, prone, reads the calcaneus alone: 15.0 deg). The healthy table gives 21.97 deg on the
  left and 28.34 on the right (reports/body_store/priors.json); both become 30.
- toe flexion (WIDEN): the toe segment (joints 10, 11: every toe at once, about the metatarsal heads) turned about x,
  +x toward the sole; the template's toes at 0 lie in line with the foot. The norm is the metatarsophalangeal flexion
  of jarm2022, 35 deg for the first toe and 35 deg for toes 2 to 5 (one SMPL+H joint moves all toes: the same value
  for both). ocran2023 corroborates it (active flexion about 45 deg for the first toe, about 40 for the lateral four).
  The healthy table gives 2.35 deg on the left and 11.14 on the right; both become 35.
- shoulder adduction coupled to the collar's rise (COUPLED): the collar (13, 14) is raised about z (+z on the left, -z
  on the right, dystonia.REGIONS). Raising it by e about the sternoclavicular joint turns the shoulder joint's parent
  frame by e, so a humerus that keeps hanging as before needs e more adduction at the shoulder (16: -z, 17: +z) in
  that frame: a rotation identity, no new angle. The rule covers the raise an operator adds, so e is the collar's rise
  over the base at the same knot (collar_rise: its elevation component minus the base's, clipped to [0, the collar's
  own elevation bound]; without a base, the rise over rest), and the shoulder's adduction bound, the healthy one or
  the base's own excursion beyond it, is widened by e at that knot and never by more: a shoulder whose collar has
  not risen over the base keeps that bound (a level 0 with raised collars opens nothing). The healthy adduction
  bounds are -80.99 deg on the left and 80.40 on the right (the hanging arm; the level-0 stand holds -80 and 80).
- the collar's own bound is kept (not widened). The healthy table allows 40.48 deg of elevation on the left and 43.49
  on the right. In angle that is at or above both norms read: jarm2022 gives shoulder girdle elevation 20 deg (the
  line from the acromion to the sternum's upper border against the line between the acromions, seen from behind),
  kenhub_sc gives elevation and depression together 40 deg and an angular range of 35 to 60 deg. In centimetres it
  is below kenhub_sc's: the clavicle elevates by about 10 cm there, while the healthy bound
  raises the SMPL+H shoulder joint by about 5 cm (dystonic_posture's shoulder elevation at the bound on the static
  stand, tests/test_synth_limits_ext.py), the template's collar-to-shoulder segment being about 10 cm long. Whether
  the TWSTRS D thirds should span about 5 cm or about 10 cm of shoulder rise is the owner's decision (open).

The healthy shrug (SHRUG, task sh1c): a session decision of 2026-11-03 under the owner's delegation
(reports/gpu_queue/decisions.md, 2026-11-03 evening) extends the coupling, and the coupling only, to the shrugs that
are not dystonic: a healthy shrug keeps the arm hanging too, and the healthy table's adduction bound there is an
artefact of percentile limits learned on motion that rarely shrugs. The table 'shrug' is the healthy table plus
COUPLED (no WIDEN: the ankle and toe widenings stay for the dystonia operators): its users are the jerks' shrug
(jerks.py, sites shrug_left and shrug_right, the shoulder turned back under the raised collar) and the standing
shrug of the 3.14 ladder (parkinsonism.py). A level 0 has no collar rise over itself, so the table opens nothing
there; record('shrug') names the decision. The decision's "level 0s are unchanged" holds for the table, not for every
body: the standing 3.14 level 0 (park_E7_spontaneity_g0) is drawn again, because its draws now include the standing
shrug, a kind built since task sh1c; the jerks' level 0s carry no shrug and stay as they were.

Since the open-body port's wave 1 (docs/OPEN_BODY_PORT_DESIGN.md section 12; owner decision C8): the healthy table
is the anatomical table of reports/openbody/joint_limits.json (scripts/openbody_limits.py, jarm2022 for every joint),
no longer the AMASS percentiles; the numbers of that table quoted above (21.97, 28.34, 2.35, 11.14, -80.99, 80.40,
40.48 and 43.49 deg) are the history of SH2. The anatomical table already holds jarm2022's foot inversion (30 deg)
and MTP flexion (35 deg), so WIDEN opens nothing beyond it and the dystonia table differs from the healthy one by the
coupling alone, as the shrug table does; its collar cap is the girdle elevation of jarm2022 (20 deg) and the
shoulder's adduction bound the hanging arm (-90 and 90 deg).

Tables (TABLES): 'healthy' the anatomical table, 'dystonia' this extension, 'shrug' the healthy table with the coupling
only. knot_bounds gives the per-knot bounds (the coupling makes them depend on the collar's rise over the base at the
same knot), box_at a JointLimits at one knot (for body.room_deg), dystonia_limits and coupled_limits the box at given
collar rises, violations and check the joint-limit counts and the validity of body.validity read against a table,
record the table's entry of a truth.
"""
from functools import lru_cache

import numpy as np

from .body import LIMIT_TOL_DEG, JointLimits, components_deg, joint_limits, validity

HEALTHY, DYSTONIA, SHRUG = "healthy", "dystonia", "shrug"
TABLES = (HEALTHY, DYSTONIA, SHRUG)
COUPLED_TABLES = (DYSTONIA, SHRUG)          # the tables whose shoulder adduction opens with the collar's rise
NAME = {HEALTHY: "anatomical healthy table (reports/openbody/joint_limits.json, body.joint_limits)",
        DYSTONIA: "anatomical healthy table with the SH2 dystonia extension (mdfm/synth/limits_ext.py)",
        SHRUG: "anatomical healthy table with the SH2 shoulder coupling only (mdfm/synth/limits_ext.py, the healthy "
               "shrug)"}
SHRUG_DECISION = ("session decision of 2026-11-03 under the owner's delegation (reports/gpu_queue/decisions.md, "
                  "2026-11-03 evening): the SH2 coupling extended to the non-dystonic shrugs, because a healthy shrug "
                  "keeps the arm hanging too; the ankle and toe widenings stay for the dystonia operators only")
SOURCES = {
    "jarm2022": "Japanese Association of Rehabilitation Medicine, Japanese Orthopaedic Association and Japanese "
                "Society for Surgery of the Foot. Joint range of motion: display and measurement (Kansetsu kadoiki "
                "hyoji narabini sokuteiho), revision of April 2022, Jpn J Rehabil Med 2021; 58: 1188-1200, J Jpn Soc "
                "Surg Foot 2021; 42: S372-S385, J Jpn Orthop Assoc 2022; 96: 75-86 (the measurement table, jarm.or.jp "
                "document kadou/03.pdf): foot inversion 0-30 deg (basic axis the perpendicular to the shank's axis "
                "in the frontal plane, moving axis the sole, knee flexed, ankle at 0 deg); MTP flexion 0-35 deg for "
                "the first toe and for toes 2 to 5; shoulder girdle elevation 0-20 deg, depression 0-10 deg",
    "menadue2006": "Menadue C, Raymond J, Kilbreath SL, Refshauge KM, Adams R. Reliability of two goniometric methods "
                   "of measuring active inversion and eversion range of motion at the ankle. BMC Musculoskelet Disord "
                   "2006; 7: 60, doi 10.1186/1471-2474-7-60 (full text, Table 1): 60 ankles of 30 adults aged 21 to 59 "
                   "(some with a past ankle injury), active inversion with a universal goniometer, sitting (anterior "
                   "method, whole-foot supination: plantarflexion, inversion and adduction) 31.5 +- 8.8 deg on day 1 "
                   "over all observers; prone (rearfoot: talocrural and subtalar only) 15.0 +- 6.1 deg (corroborates "
                   "the inversion bound, an upper bound on a combined motion)",
    "ocran2023": "Ocran E (reviewer Tempany D). Metatarsophalangeal (MTP) joints. Kenhub, last reviewed 30 October "
                 "2023 (kenhub.com): active flexion of the first metatarsophalangeal joint about 45 deg, of the "
                 "lateral four about 40 deg (corroborates the toe bound)",
    "kenhub_sc": "McLaren N (reviewer Mytilinaios D). Sternoclavicular joint. Kenhub, last reviewed 30 October 2023 "
                 "(kenhub.com): mean active elevation and depression together 40 deg; the clavicle elevates by about "
                 "10 cm and depresses by 3 cm, an angular range of 35 to 60 deg (read for the collar's bound, which is "
                 "not widened)",
}
INVERSION_DEG = 30.0         # jarm2022: foot inversion 0-30 deg
TOE_FLEXION_DEG = 35.0       # jarm2022: MTP flexion 0-35 deg, the first toe and toes 2 to 5
# (joint, component 0 x / 1 y / 2 z, bound 'lo' or 'hi', the widened bound deg, source key, the number read)
WIDEN = ((7, 2, "lo", -INVERSION_DEG, "jarm2022", "foot inversion 0-30 deg, frontal plane"),
         (8, 2, "hi", INVERSION_DEG, "jarm2022", "foot inversion 0-30 deg, frontal plane"),
         (10, 0, "hi", TOE_FLEXION_DEG, "jarm2022", "MTP flexion 0-35 deg, first toe and toes 2 to 5"),
         (11, 0, "hi", TOE_FLEXION_DEG, "jarm2022", "MTP flexion 0-35 deg, first toe and toes 2 to 5"))
CORROBORATE = ("menadue2006", "ocran2023")   # sources read beside WIDEN's (module docstring)
COLLAR_SOURCES = ("jarm2022", "kenhub_sc")   # read for the collar's bound, which is kept
# per side: the shoulder, its adduction sign about z, the collar, its elevation sign about z (dystonia.REGIONS)
COUPLED = {"left": (16, -1.0, 13, 1.0), "right": (17, 1.0, 14, -1.0)}


def _check(table):
    if table not in TABLES:
        raise ValueError(f"joint-limit table {table!r}, expected one of {TABLES}")


@lru_cache(maxsize=1)
def _widened():
    """(lo, hi) (22, 3) deg: the healthy table with WIDEN applied (the coupled shoulder bounds not yet)."""
    h = joint_limits()
    lo, hi = h.lo.copy(), h.hi.copy()
    for j, c, side, deg, _, _ in WIDEN:
        if side == "lo":
            lo[j, c] = min(lo[j, c], deg)
        else:
            hi[j, c] = max(hi[j, c], deg)
    return lo, hi


def _static(table):
    """(lo, hi) (22, 3) deg: a table's bounds before the coupling (the dystonia table WIDEN applied, the healthy and
    the shrug tables the healthy ones)."""
    _check(table)
    if table == DYSTONIA:
        return _widened()
    h = joint_limits()
    return h.lo, h.hi


def collar_cap(side):
    """The largest collar rise (deg) the coupling reads on side: the collar's healthy elevation bound."""
    h = joint_limits()
    _, _, c, es = COUPLED[side]
    return float(h.hi[c, 2] if es > 0 else -h.lo[c, 2])


def collar_rise(comp, base_comp=None):
    """(..., 2) deg [left, right]: each collar's rise over the base at the same knot (its elevation component minus
    the base's, base_comp of comp's shape) or, without a base, over rest (its elevation component), clipped to
    [0, collar_cap] (module docstring: the raise the coupling turns into shoulder adduction)."""
    comp = np.asarray(comp, dtype=np.float64)
    b = None if base_comp is None else np.asarray(base_comp, dtype=np.float64)
    out = []
    for side in ("left", "right"):
        _, _, c, es = COUPLED[side]
        e = es * comp[..., c, 2] - (0.0 if b is None else es * b[..., c, 2])
        out.append(np.clip(e, 0.0, collar_cap(side)))
    return np.stack(out, -1)


def collar_elevation(comp):
    """(..., 2) deg [left, right]: the collars' elevation components of comp (..., 22, 3) deg, clipped to
    [0, collar_cap] (collar_rise over rest)."""
    return collar_rise(comp)


def _apply_coupling(lo, hi, rise):
    """lo, hi (..., 22, 3) with each shoulder's adduction bound widened by its collar's rise (..., 2)."""
    for i, side in enumerate(("left", "right")):
        s, sg, _, _ = COUPLED[side]
        if sg < 0:
            lo[..., s, 2] = lo[..., s, 2] - rise[..., i]
        else:
            hi[..., s, 2] = hi[..., s, 2] + rise[..., i]
    return lo, hi


@lru_cache(maxsize=512)
def _box(e_left, e_right, table=DYSTONIA):
    h = joint_limits()
    lo, hi = (x.copy() for x in _static(table))
    lo, hi = _apply_coupling(lo, hi, np.array([e_left, e_right]))
    lo.setflags(write=False)
    hi.setflags(write=False)
    tag = "SH2" if table == DYSTONIA else "SH2 coupling only, the healthy shrug"
    return JointLimits(lo, hi, h.percentiles, h.path + f" + mdfm/synth/limits_ext.py ({tag})", h.n_sampled)


def coupled_limits(table, rise=(0.0, 0.0)):
    """A coupled table (COUPLED_TABLES: 'dystonia' or 'shrug') as a box (body.JointLimits) at given collar rises
    (deg, [left, right], clipped to [0, collar_cap]): its static bounds (_static) with each shoulder's adduction bound
    widened by its collar's rise. At rise 0 the shoulders keep the healthy bounds."""
    if table not in COUPLED_TABLES:
        raise ValueError(f"table {table!r} has no coupling, expected one of {COUPLED_TABLES}")
    e = [float(np.clip(x, 0.0, collar_cap(sd))) for x, sd in zip(rise, ("left", "right"))]
    return _box(round(e[0], 9), round(e[1], 9), table)


def dystonia_limits(rise=(0.0, 0.0)):
    """The dystonia table as a box (body.JointLimits) at given collar rises (deg, [left, right], clipped to
    [0, collar_cap]): WIDEN applied and each shoulder's adduction bound widened by its collar's rise. At rise 0 the
    shoulders keep the healthy bounds."""
    return coupled_limits(DYSTONIA, rise)


def widest(table=DYSTONIA):
    """A coupled table's box at the largest collar rises (default the dystonia table): no knot of the table reaches
    past it."""
    return coupled_limits(table, (collar_cap("left"), collar_cap("right")))


def box_at(table, comp_k=None, base_k=None):
    """A JointLimits of table at one knot: the healthy table, or a coupled table's box at the collar rises of the
    components comp_k (22, 3) deg over base_k (22, 3) (over rest without base_k; default no rise: the shoulders keep
    the healthy bounds, as at a base's own knot, where no collar has risen over the base)."""
    _check(table)
    if table == HEALTHY:
        return joint_limits()
    if comp_k is None:
        return coupled_limits(table)
    return coupled_limits(table, tuple(collar_rise(np.asarray(comp_k, dtype=np.float64), base_k)))


def knot_bounds(table, comp, base_comp=None):
    """(lo, hi) (T, 22, 3) deg: the bounds of table at each knot of the components comp (T, 22, 3) deg; with
    base_comp (T, 22, 3), the box [min(lo, base), max(hi, base)] (body.limit_violations). A coupled table's
    shoulder adduction bounds (the dystonia and the shrug tables) are then widened by each knot's collar rise over the
    base (over rest without one)."""
    _check(table)
    comp = np.asarray(comp, dtype=np.float64)
    T = comp.shape[0]
    wl, wh = _static(table)
    lo, hi = np.broadcast_to(wl, (T, 22, 3)).copy(), np.broadcast_to(wh, (T, 22, 3)).copy()
    if base_comp is not None:
        b = np.asarray(base_comp, dtype=np.float64)
        lo, hi = np.minimum(lo, b), np.maximum(hi, b)
    if table == HEALTHY:
        return lo, hi
    return _apply_coupling(lo, hi, collar_rise(comp, base_comp))


def excess(table, comp, base_comp=None, rows=None):
    """(T, n, 3) deg: how far each component of the rows (default 1 to 21) lies outside the table at its knot, or
    outside [min(lo, base), max(hi, base)] with base_comp (T, 22, 3) (knot_bounds); <= 0 inside."""
    rows = list(range(1, 22)) if rows is None else list(rows)
    comp = np.asarray(comp, dtype=np.float64)
    lo, hi = knot_bounds(table, comp, base_comp)
    a = comp[:, rows]
    return np.maximum(lo[:, rows] - a, a - hi[:, rows])


def violations(table, rot6d, base=None, tol=LIMIT_TOL_DEG):
    """{'outside', 'new' (None without base), 'worst_deg'}: body.limit_violations read against table at each knot
    (rot6d and base (T, 22, 6); 'outside' with the collar rises over rest, 'new' over the base)."""
    comp = components_deg(rot6d)
    ex0 = excess(table, comp)
    outside = int((ex0 > tol).sum())
    if base is None:
        return {"outside": outside, "new": None, "worst_deg": float(max(ex0.max(initial=0.0), 0.0))}
    ex = excess(table, comp, components_deg(base))
    return {"outside": outside, "new": int((ex > tol).sum()), "worst_deg": float(max(ex.max(initial=0.0), 0.0))}


def coupling_report(base, out):
    """{side: {'collar_rise_deg_max', 'adduction_beyond_healthy_deg_max'}, 'ok'}: how far each shoulder adducts past
    the healthy bound (or the base's excursion) and the collar's rise over the base that allows it (the coupling
    holds when the first never exceeds the second at a knot)."""
    h = joint_limits()
    a = components_deg(out.rot6d)
    b = components_deg(base.rot6d) if base.T == out.T else None
    rise = collar_rise(a, b)
    rep, ok = {}, True
    for i, side in enumerate(("left", "right")):
        s, sg, _, _ = COUPLED[side]
        bound = h.lo[s, 2] if sg < 0 else h.hi[s, 2]
        if b is not None:
            bound = np.minimum(bound, b[:, s, 2]) if sg < 0 else np.maximum(bound, b[:, s, 2])
        beyond = np.maximum(0.0, (bound - a[:, s, 2]) if sg < 0 else (a[:, s, 2] - bound))
        ok = ok and bool((beyond <= rise[:, i] + LIMIT_TOL_DEG).all())
        rep[side] = {"collar_rise_deg_max": float(rise[:, i].max()),
                     "adduction_beyond_healthy_deg_max": float(beyond.max())}
    rep["ok"] = ok
    return rep


def check(table, base, out, **kw):
    """body.validity(base, out, **kw): unchanged for the healthy table; for a coupled table (the dystonia and the
    shrug tables) its joint-limit counts read against that table at each knot ('limits_table' names it) and the
    coupling report."""
    _check(table)
    if table == HEALTHY:
        return validity(base, out, **kw)
    v = validity(base, out, limits=widest(table), **kw)
    lim = violations(table, out.rot6d, base.rot6d if base.T == out.T else None)
    v.update(limits_outside=lim["outside"], limits_new=lim["new"], limits_worst_deg=lim["worst_deg"],
             limits_table=table)
    v["coupling"] = coupling_report(base, out)
    return v


def record(table=DYSTONIA):
    """The table's entry of a truth's params: its name; for the dystonia table the widened bounds with their sources
    and the coupling rule; for the shrug table the coupling rule and the decision that extends it (SHRUG_DECISION)."""
    _check(table)
    if table == HEALTHY:
        return {"name": NAME[HEALTHY], "table": HEALTHY}
    h = joint_limits()
    coupled = [{"side": sd, "shoulder": s, "collar": c, "healthy_adduction_bound_deg":
                float(h.lo[s, 2] if sg < 0 else h.hi[s, 2]), "collar_cap_deg": collar_cap(sd),
                "rule": "the shoulder's adduction bound widened at each knot by its collar's rise over the base there"}
               for sd, (s, sg, c, _) in COUPLED.items()]
    if table == SHRUG:
        return {"name": NAME[SHRUG], "table": SHRUG, "widened": [], "coupled": coupled, "decision": SHRUG_DECISION,
                "sources": []}
    widen = [{"joint": j, "component": "xyz"[c], "bound": side,
              "healthy_deg": float(h.lo[j, c] if side == "lo" else h.hi[j, c]), "widened_deg": deg,
              "source": src, "read": read} for j, c, side, deg, src, read in WIDEN]
    return {"name": NAME[DYSTONIA], "table": DYSTONIA, "widened": widen, "coupled": coupled,
            "sources": sorted({w[4] for w in WIDEN} | set(CORROBORATE) | set(COLLAR_SOURCES))}
