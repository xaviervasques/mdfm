"""Leg-to-leg contact of the synthetic patient (shared change SH8 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md):
at every knot, the smallest distance between the surface of the left leg and the surface of the right leg of the
skinned mesh of the selected body model (body.body_model: the open body by default), negative when one leg passes
into the other (its depth), and the validity entry built on it.

Surfaces (leg_mesh): per side, the mesh vertices whose largest skinning weight lies on that side's hip, knee, ankle
or foot joint (1, 4, 7, 10 on the left; 2, 5, 8, 11 on the right) and the faces whose three vertices all belong to
that set, less the inner thigh band: the vertices medial to their hip joint whose rest height lies above the crotch
(the lowest vertex whose largest weight lies on the pelvis) minus the band (crotch_band_m). The two thighs of the
SMPL+H neutral rest mesh touch below the crotch and the surface runs on from one leg to the other there; the band
(CROTCH_BAND_REFERENCE_M) leaves that strip out, so that the rest pose reads its true gap (rest_gap,
tests/test_synth_legcontact.py). The open body's rest thighs stand apart up to the crotch: its band is 0
(CROTCH_BAND_M, the same rule; scripts/openbody_legcontact_calibration.py). Each leg is cut into pieces
(PIECES): the thigh (the faces of the hip's vertices) and the shank (the knee's) each in a proximal and a distal half
along the bone, the foot (the ankle's and the foot joint's) whole; a face belongs to the piece of most of its
vertices. The soft zone is the thigh above the knee: a place along the leg (along_v, the rest mesh's distance above
the knee joint along the femur, negative below it along the tibia, NO_ALONG on the foot) more than KNEE_ZONE_M above
the knee joint. A contact is soft when its query point (the mean place of its vertex pair) and the closest point of
the other leg (its place interpolated over its face) both lie in the soft zone: two soft surfaces pressed together
are the inner thighs, whose soft tissue gives way in a living body where the rigid mesh passes through instead;
every other contact (knee, shank, foot, or one of them against a thigh) is hard. (Until 2026-11-03 a face counted
as soft only with its three vertices soft, so that a thigh pressed against a face across the zone's edge, up to an
edge length above the knee zone, read as hard.)

Skinning (skin_legs): the body model's one skinning call restricted to the leg vertices (body_model().skin, hands
at rest), checked against viewer.knot_vertices by the tests; the joints and their global rotations by the program's
forward kinematics on the model's rest joints.

Distance (gap_from_pose, leg_gap), at every knot:
- culling: each piece's box, in the global frame of its first joint, spans the skinned vertices of its faces (so
  that the piece lies inside it); a query point of one leg (the vertices and the edge midpoints of its faces) is
  measured only if it lies inside a box of the other leg inflated by CULL_M + SLACK_M. A knot with no such point on
  either leg reads CULL_M, a lower bound ('exact' False): no surface point of a leg lies within CULL_M of the other.
- mesh: a first pass takes, for every measured point, its nearest measured point of the other leg (a k-d tree per
  knot, up to DEEP_M away) and signs that distance by the latter's normal (the sum of its faces' normals); the points
  behind the other leg's surface or within BAND_M of the smallest signed first-pass distance of their class (soft or
  hard), and those within BAND_M of the smallest unsigned one, N_EXACT at most for each, are measured exactly
  against the faces of their K_NEAR nearest points of the other leg:
  the point-to-triangle distance (Ericson), negative when the point lies behind the closest face (inside the other
  leg), positive when the closest point lies on an open edge of the surface (the top of the thigh, the band). Both
  directions; per knot the smallest over all contacts ('gap_m'), over the hard ones ('hard_m') and over the soft ones
  ('soft_m'). A depth beyond CULL_M + SLACK_M is read on the points near the surface only (approximate, a failure
  either way).
The error against the mesh comes from the query points' sampling (two surfaces meeting between samples) and from a
closest face outside the candidates: mesh_error measures it against a dense reference (barycentric samples of level
REF_LEVEL on every face against every nearby face of the other leg) and the tests hold it below ERR_TOL_M on knots
of a walk and of a scissoring walk. The fitted capsule model of the pieces (capsule_gap: the piece's bone, its
radius the mean distance of its vertices) is measured the same way, in the record only.

Validity (validity): 'ok' when no hard contact lies deeper than TOL_M and no soft one deeper than SOFT_TOL_M.
Beside it, outside 'ok', the distal band's reading (distal_gap, report 'distal_*'): the hard gap again with the soft
zone's edge at KNEE_ZONE_M + DISTAL_BAND_M, so that a contact of the thigh's distal band (the medial thigh just above
the condyles) counts with the knee. The rule before 2026-11-03 read most of that band as hard (a face across the
zone's edge counted as hard, and so did every vertex of the knee's skinning above the knee joint); the present rule
reads it as soft, which is what takes several gait clips under TOL_M, the healthy level 0 among them, beside the
changes of gait.py (tests/test_synth_gait.py pins the distal reading of the healthy walks).
Calibration on normal gait (2026-11-03, tests/test_synth_legcontact.py and tests/test_synth_gait.py): the healthy
walks of gait.py (the level-0 course with its 180 deg turn, the tandem of SARA 0, the pull test's two steps) hold
both tolerances; the former healthy walk (a 0.09 m base without the swing guard nor the turn's widening) passed its
knees into each other beyond TOL_M, as does a crouched narrow walk without the guard (scissoring); the soft class
takes the inner thighs only, the knees and shanks stay hard. That calibration was read on the SMPL+H surface; on the
open body (scripts/openbody_legcontact_calibration.py, reports/openbody/legcontact_calibration.json, open-body port,
wave 3 follow-up, D36 to D38): the band by its rule is 0; the healthy walk's inner thighs never touch (smallest soft
gap +1.7 mm with no band, against -13.2 mm on SMPL+H) and its smallest hard gap is the feet's (16.6 mm); of the
catalogue's healthy clips the level-0 course presses its inner thighs 4.6 mm in its turn, the tandem and the pull
test not at all; the former healthy walk still fails the check, by its feet (-10.1 mm); the seated inner thighs meet
at mid-thigh at 10.8 deg of knee closing, before the knees reach their clearance (13.75 deg; the knee guard keeps the
knees, the hard class; the inner thighs press within SOFT_TOL_M). KNEE_ZONE_M and DISTAL_BAND_M are anatomical (the
condyles' height over the joint), not readings of a mesh, and stay (on SMPL+H the knee's own skinning reached 2.5 to
3.1 cm above the knee joint, on the open body none above it); SOFT_TOL_M stays (the deepest healthy inner-thigh
contact of the open body, 4.6 mm, lies inside it; wave 5 reads the gait clips against it).
"""
import functools

import numpy as np
import torch

LEG = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}            # hip, knee, ankle, foot
SIDES = ("left", "right")
SIGN = {"left": 1.0, "right": -1.0}                                 # the subject's left is +x
PIECES = ("thigh_upper", "thigh_lower", "shank_upper", "shank_lower", "foot")
CROTCH_BAND_M = 0.0      # [module] inner thigh band left out below the crotch on the open body, by the rule of the
#                          band (about the narrowest band whose rest gap is positive on the neutral mesh, so that the
#                          inner thighs below it are measured): the open body's rest thighs stand 14.5 mm apart up to
#                          the crotch, so no band is needed (open-body port, wave 3 follow-up, D36;
#                          scripts/openbody_legcontact_calibration.py, reports/openbody/legcontact_calibration.json)
CROTCH_BAND_REFERENCE_M = 0.08   # [module] the same rule on the SMPL+H reference, whose rest thighs touch below the
#                          crotch (the rest gap -4.4 mm with no band, 3.8 mm past 8 cm, the calibration's record)
KNEE_ZONE_M = 0.03       # [module] the distal end of the thigh counted with the knee (hard): the femoral condyles,
#                          about this far above the knee joint
DISTAL_BAND_M = 0.03     # [module] the distal band of the soft zone (KNEE_ZONE_M to KNEE_ZONE_M + DISTAL_BAND_M above
#                          the knee joint), read in the record as hard (report 'distal_*', distal_gap), outside 'ok':
#                          the rule before 2026-11-03 read most contacts of this band as hard, its faces across the
#                          zone's edge counting with the knee
NO_ALONG = -1.0          # the place along the leg given to the foot's vertices (never soft)
CULL_M = 0.02            # [module] the measured range: a knot whose legs lie farther apart reads this (a lower
#                          bound)
SLACK_M = 0.02           # [module] a surface point lies within about half an edge of a query point (edges of the
#                          leg mesh up to about 6 cm); the boxes are inflated by CULL_M + SLACK_M
K_NEAR = 2               # nearest query points of the other leg whose faces are the exact candidates
BAND_M = 0.008           # [module] the first pass errs by the surface's bend over half an edge; points within this
#                          of their class's smallest first-pass distance are measured exactly (the N_EXACT nearest)
N_EXACT = 48             # [module] at most this many points per class and first-pass measure are measured exactly
TOL_M = 0.005            # the contact tolerance of contact.py (TOL_M): a hard contact may lie at most this deep
SOFT_TOL_M = 0.03        # [module, unsourced] the inner thighs may press this far into each other (soft tissue the
#                          rigid mesh cannot compress; the healthy walks of gait.py, the level-0 course and the
#                          tandem, read their deepest inner-thigh contact inside it, tests/test_synth_legcontact.py)
TOUCH_M = 0.005          # a knot whose hard gap lies below this reads as the legs touching
ERR_TOL_M = 0.002        # [module] the error bound of the mesh gap against the dense reference held by the tests
REF_LEVEL = 4            # barycentric subdivision of the dense reference (15 points per face)
CHUNK = 512              # knots skinned at once
BATCH = 128              # knots measured at once (a memory bound)
DEEP_M = 0.08            # [module] about half the thickness of the thigh: a point read deeper inside the other leg
#                          lies outside it (depths beyond CULL_M + SLACK_M are approximate, read on the points near)


# ---------------------------------------------------------------- the leg surfaces
def crotch_band_m(name=None):
    """m: the inner thigh band of the body model name (None: the selected one): CROTCH_BAND_REFERENCE_M on the SMPL+H
    reference, CROTCH_BAND_M on the open body (module docstring)."""
    from . import openbody as OB
    if name is None:
        from .body import model_name
        name = model_name()
    return CROTCH_BAND_REFERENCE_M if name == OB.SMPLH_REFERENCE else CROTCH_BAND_M


@functools.lru_cache(maxsize=8)
def _mesh(name, knee_zone_m=None):
    from .body import body_model
    from .viewer import dominant_joint, faces as mesh_faces, rest_vertices
    zone = KNEE_ZONE_M if knee_zone_m is None else float(knee_zone_m)
    band_m = crotch_band_m(name)
    faces = mesh_faces()
    dom = dominant_joint()
    v_all = rest_vertices()
    J52 = body_model().rest_joints52().numpy()
    parents = body_model().parents52()
    crotch_y = float(v_all[np.flatnonzero(dom == 0), 1].min())
    out = {"J": J52[:22].copy(), "parents": parents[:22], "crotch_y": crotch_y, "crotch_band_m": band_m}
    for side in SIDES:
        h, k, a, f = LEG[side]
        sg = SIGN[side]
        ids = np.flatnonzero(np.isin(dom, LEG[side]))
        band = (v_all[ids, 1] > crotch_y - band_m) & (sg * v_all[ids, 0] < sg * J52[h, 0])
        ids = ids[~band]
        loc = -np.ones(len(v_all), np.int64)
        loc[ids] = np.arange(len(ids))
        F = loc[faces[np.isin(faces, ids).all(1)]]                   # (nF, 3) local vertex indices
        V = v_all[ids]
        nV = len(ids)
        # vertex pieces: by dominant joint, the thigh and shank halves by the axial share along the bone. The place
        # of a vertex along its leg (along_v, rest mesh): its distance above the knee joint along the femur (the
        # hip's and the knee's vertices above the knee joint), negative below it along the tibia (the knee's
        # vertices below it), NO_ALONG on the foot (the ankle's and the foot joint's). The soft zone: the thigh more
        # than KNEE_ZONE_M (knee_zone_m) above the knee joint, whichever of the hip and the knee carries the vertex
        d = dom[ids]
        piece_v = np.full(nV, 4, np.int64)
        along_v = np.full(nV, NO_ALONG)
        u_f = (J52[h] - J52[k]) / np.linalg.norm(J52[h] - J52[k])
        u_t = (J52[a] - J52[k]) / np.linalg.norm(J52[a] - J52[k])
        for base, (j0, j1) in ((0, (h, k)), (2, (k, a))):
            sel = d == j0
            ax = J52[j1] - J52[j0]
            L = float(np.linalg.norm(ax))
            s = (V[sel] - J52[j0]) @ ax / (L * L)
            piece_v[sel] = base + (s >= 0.5).astype(np.int64)
            up = (V[sel] - J52[k]) @ u_f
            along_v[sel] = np.where(up >= 0.0, up, -np.maximum((V[sel] - J52[k]) @ u_t, 0.0))
        soft_v = along_v > zone
        pf = piece_v[F]                                    # a face's piece: most of its vertices' (a tie: the first)
        maj = np.where(pf[:, 1] == pf[:, 2], pf[:, 1], pf[:, 0])
        maj = np.where(pf[:, 0] == pf[:, 1], pf[:, 0], maj)
        # query points: the vertices, then the edge midpoints, as (i, j) vertex pairs
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        edges, e_count = np.unique(e, axis=0, return_counts=True)
        eidx = {(int(x), int(y)): nV + n for n, (x, y) in enumerate(edges)}
        pi = np.concatenate([np.arange(nV), edges[:, 0]])
        pj = np.concatenate([np.arange(nV), edges[:, 1]])
        inc = [[] for _ in range(len(pi))]                  # the faces holding each query point
        for fi, (x, y, z) in enumerate(F):
            for vv in (x, y, z):
                inc[vv].append(fi)
            for u, w in ((x, y), (y, z), (z, x)):
                inc[eidx[(min(u, w), max(u, w))]].append(fi)
        cand = -np.ones((len(pi), max(len(r) for r in inc)), np.int64)
        for n, r in enumerate(inc):
            cand[n, :len(r)] = r
        # open boundary edges (one face only): the top of the thigh and the band
        open_e = {(int(x), int(y)) for (x, y), c in zip(edges, e_count) if c == 1}
        bnd = np.array([[(min(u, w), max(u, w)) in open_e for u, w in ((x, y), (y, z), (z, x))] for x, y, z in F])
        pieces = []
        for p in range(len(PIECES)):
            fsel = np.flatnonzero(maj == p)
            vset = np.unique(F[fsel])
            j0, j1 = {0: (h, k), 1: (h, k), 2: (k, a), 3: (k, a), 4: (a, f)}[p]
            ax = J52[j1] - J52[j0]
            s = (V[vset] - J52[j0]) @ ax / float(ax @ ax)
            pieces.append({"name": PIECES[p], "faces": fsel, "verts": vset, "joints": (j0, j1),
                           "s": (float(s.min()), float(s.max()))})
        along_p = 0.5 * (along_v[pi] + along_v[pj])
        out[side] = {"ids": ids, "V": V, "F": F,
                     "pi": pi, "pj": pj, "cand": cand, "bnd": bnd, "pieces": pieces,
                     "point_piece": piece_v[pi], "face_piece": maj, "along_v": along_v, "along_point": along_p,
                     "soft_point": along_p > zone, "soft_face": soft_v[F].all(1), "knee_zone_m": zone}
    return out


def leg_mesh(betas=None, gender=2, knee_zone_m=None):
    """The leg surfaces of the selected body model in the catalogue's one shape (module docstring): {'J',
    'crotch_y', side: {'ids' mesh vertex ids, 'V' rest positions, 'F' faces (local indices), 'pieces': [{name,
    faces, verts, joints, s}], 'soft_point', 'soft_face', 'knee_zone_m', ...}}; knee_zone_m moves the soft zone's
    lower edge (None: KNEE_ZONE_M)."""
    from .body import model_name
    b = np.zeros(16) if betas is None else np.asarray(betas, dtype=np.float64).reshape(-1)
    if np.abs(b).max() > 0 or int(gender) != 2:
        raise ValueError("the leg surfaces are those of the neutral shape only (betas 0, gender code 2)")
    return _mesh(model_name(), None if knee_zone_m is None else float(knee_zone_m))


def skin_legs(R_local, transl, mesh):
    """{side: (N, V, 3)} the skinned leg vertices for local rotations R_local (N, 22, 3, 3) and translations
    transl (N, 3) (module docstring), the joint positions (N, 22, 3) under 'joints' and their global rotations
    (N, 22, 3, 3) under 'G'."""
    R = np.asarray(R_local, dtype=np.float64)
    N = R.shape[0]
    J, par = mesh["J"], mesh["parents"]
    G = np.empty((N, 22, 3, 3))
    P = np.empty((N, 22, 3))
    G[:, 0], P[:, 0] = R[:, 0], J[0]
    for i in range(1, 22):
        p = par[i]
        G[:, i] = G[:, p] @ R[:, i]
        P[:, i] = P[:, p] + np.einsum("nij,j->ni", G[:, p], J[i] - J[p])
    from .body import body_model
    tr = np.asarray(transl, dtype=np.float64)
    out = {"joints": P + tr[:, None], "G": G}
    ids = np.concatenate([mesh[side]["ids"] for side in SIDES])
    X = body_model().skin(R, tr, None, ids=ids)
    n0 = len(mesh[SIDES[0]]["ids"])
    out[SIDES[0]], out[SIDES[1]] = X[:, :n0], X[:, n0:]
    return out


# ---------------------------------------------------------------- geometry
def _dot(a, b):
    return (a * b).sum(-1)


def closest_on_triangles(p, a, b, c):
    """(..., 3) the closest point to p on the triangles (a, b, c) (Ericson, Real-Time Collision Detection, 5.1.5),
    every argument (..., 3) and finite."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        v = np.where(den != 0, vb / den, 0.0)
        w = np.where(den != 0, vc / den, 0.0)
        res = a + ab * v[..., None] + ac * w[..., None]
        # the regions in the reverse of Ericson's early-return order, so that the first region of his order wins
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        res = np.where(m[..., None], b + np.nan_to_num(t)[..., None] * (c - b), res)
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / (d2 - d6)
        res = np.where(m[..., None], a + np.nan_to_num(t)[..., None] * ac, res)
        res = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, res)
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / (d1 - d3)
        res = np.where(m[..., None], a + np.nan_to_num(t)[..., None] * ab, res)
        res = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, res)
        res = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, res)
    return res


def segment_distance(p0, p1, q0, q1):
    """(...,) the distance between the segments p0 p1 and q0 q1 (Ericson 5.1.9), arguments (..., 3)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    c, b = _dot(d1, r), _dot(d1, d2)
    den = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-12, np.clip((b * f - c * e) / den, 0.0, 1.0), 0.0)
        t = np.where(e > 1e-12, (b * s + f) / e, 0.0)
        s = np.where(t < 0, np.clip(np.where(a > 1e-12, -c / a, 0.0), 0.0, 1.0),
                     np.where(t > 1, np.clip(np.where(a > 1e-12, (b - c) / a, 0.0), 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p0 + d1 * s[..., None] - (q0 + d2 * t[..., None]), axis=-1)


def _points(X, m):
    return 0.5 * (X[:, m["pi"]] + X[:, m["pj"]])


def boxes(X, G, mesh, side):
    """(centre (N, P, 3), axes (N, P, 3, 3) as columns, half extents (N, P, 3)) of a side's pieces: the global frame
    of the piece's first joint (G (N, 22, 3, 3) global rotations), the extent of the skinned vertices of its faces
    X (N, V, 3) along those axes, so that every face of the piece lies inside its box."""
    m = mesh[side]
    N, P = X.shape[0], len(m["pieces"])
    cen, ax, half = np.zeros((N, P, 3)), np.zeros((N, P, 3, 3)), np.full((N, P, 3), -1.0)
    for i, pc in enumerate(m["pieces"]):
        if not len(pc["verts"]):
            continue
        Ri = G[:, pc["joints"][0]]
        loc = np.matmul(X[:, pc["verts"]], Ri)
        lo, hi = loc.min(1), loc.max(1)
        cen[:, i] = np.einsum("nij,nj->ni", Ri, 0.5 * (lo + hi))
        ax[:, i] = Ri
        half[:, i] = 0.5 * (hi - lo)
    return cen, ax, half


def _in_boxes(Y, box, delta):
    """(N, M) bool: the points Y (N, M, 3) inside any of the boxes (centre, axes, half) inflated by delta."""
    cen, ax, half = (torch.from_numpy(x) for x in box)
    Yt = torch.from_numpy(Y)
    out = torch.zeros(Y.shape[:2], dtype=torch.bool)
    for i in range(cen.shape[1]):
        if bool((half[:, i] < 0).all()):
            continue
        loc = torch.bmm(Yt - cen[:, i, None], ax[:, i])
        out |= (loc.abs() <= half[:, i, None] + delta).all(-1)
    return out.numpy()


def _signed_closest(p, tri, bnd, return_point=False):
    """(M,) signed distance of the points p (M, 3) to the closest of their candidate triangles tri (M, C, 3, 3)
    (non-finite rows ignored), and (M,) the column of that triangle (and (M, 3) the closest point when
    return_point). The sign: the point is inside when it lies
    behind the sum of the unit normals of every candidate face that shares the closest point (one face inside it,
    the two faces of an edge, the faces around a vertex: the pseudo-normal, so that a point beside a concave edge
    is not read inside by one face's plane); a closest point on an open boundary edge of the leg surface (bnd
    (M, C, 3) flags of the edges 0-1, 1-2, 2-0) reads outside."""
    ok = np.isfinite(tri).all((-1, -2))
    T = np.where(ok[..., None, None], tri, 0.0)
    P = np.repeat(p[:, None], tri.shape[1], 1)
    q = closest_on_triangles(P, T[..., 0, :], T[..., 1, :], T[..., 2, :])
    d = np.where(ok, np.linalg.norm(P - q, axis=-1), np.inf)
    c = np.argmin(d, 1)
    r = np.arange(len(p))
    qq, dd = q[r, c], d[r, c]
    share = ok & (np.linalg.norm(q - qq[:, None], axis=-1) < 1e-7)            # the faces holding the closest point
    n = np.cross(T[..., 1, :] - T[..., 0, :], T[..., 2, :] - T[..., 0, :])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-18)
    inside = _dot((n * share[..., None]).sum(1), p - qq) < 0
    on_b = np.zeros(len(p), bool)
    for e, (u, w) in enumerate(((0, 1), (1, 2), (2, 0))):
        sel = bnd[..., e] & share
        if sel.any():
            rr, cc = np.nonzero(sel)
            hit = segment_distance(qq[rr], qq[rr], T[rr, cc, u], T[rr, cc, w]) < 1e-9
            on_b[rr[hit]] = True
    if return_point:
        return np.where(inside & ~on_b, -dd, dd), c, qq
    return np.where(inside & ~on_b, -dd, dd), c


def _barycentric(q, a, b, c):
    """(..., 3) the barycentric weights of the points q in the triangles (a, b, c) (q on or near the triangle),
    each argument (..., 3)."""
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = _dot(v0, v0), _dot(v0, v1), _dot(v1, v1)
    d20, d21 = _dot(v2, v0), _dot(v2, v1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1.0 - v - w, v, w], -1)


def _face_normals(X, m):
    """(n, nF + 1, 3) the face normals of a leg (not unit) from its skinned vertices X (n, V, 3), a zero row last
    (the padding -1 of the incidence table)."""
    t = X[:, m["F"]]
    fn = np.cross(t[:, :, 1] - t[:, :, 0], t[:, :, 2] - t[:, :, 0])
    return np.concatenate([fn, np.zeros_like(fn[:, :1])], 1)


def _direction_gap(PA, PB, XB, FN, mA, mB, inA, inB, cull_m):
    """{'soft', 'hard': (value (n,), A point (n,), B face (n,))} of one direction at n knots: the query points of leg A
    inside the boxes of B (inA (n, npts)) against the faces of B near its query points inside the boxes of A (inB)
    (module docstring; the nearest points by a k-d tree per knot); inf where nothing is near."""
    from scipy.spatial import cKDTree
    n = PA.shape[0]
    out = {c: (np.full(n, np.inf), -np.ones(n, np.int64), -np.ones(n, np.int64)) for c in ("soft", "hard")}
    rk, ra, rn = [], [], []
    for k in range(n):
        a, b = np.flatnonzero(inA[k]), np.flatnonzero(inB[k])
        if not len(a) or not len(b):
            continue
        tree = cKDTree(PB[k, b])
        dmin, j = tree.query(PA[k, a], k=1, distance_upper_bound=DEEP_M)
        ok = np.isfinite(dmin)
        if not ok.any():
            continue
        a, dmin, bj = a[ok], dmin[ok], b[j[ok]]
        nrm = FN[k, mB["cand"][bj]].sum(1)                            # the nearest B point's normal: the sum of
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)   # its faces' normals
        est = _dot(PA[k, a] - PB[k, bj], nrm)
        # near the other leg's surface, or behind it and no deeper than DEEP_M (a point deep inside lies far from
        # every point of the surface; a point far behind a distant face lies outside)
        near = (dmin < cull_m + SLACK_M) | (est < 0)
        soft1 = mA["soft_point"][a] & mB["soft_point"][bj]
        sel = np.zeros(len(a), bool)
        for cls in (soft1, ~soft1):
            c = np.flatnonzero(near & cls)
            if len(c):
                # the points behind the other leg's surface (est < 0, the N_EXACT deepest: a misjudged first-pass
                # normal cannot hide the deepest one) or within BAND_M of the smallest estimate, and those within
                # BAND_M of the smallest unsigned distance
                o = np.argsort(est[c])[:N_EXACT]
                sel[c[o[(est[c][o] < est[c][o[0]] + BAND_M) | (est[c][o] < 0)]]] = True
                o = np.argsort(dmin[c])[:N_EXACT]
                sel[c[o[dmin[c][o] < dmin[c][o[0]] + BAND_M]]] = True
        if not sel.any():
            continue
        a = a[sel]
        kq = min(K_NEAR, len(b))
        _, nn = tree.query(PA[k, a], k=kq)
        nn = np.asarray(nn).reshape(len(a), kq)
        if kq < K_NEAR:
            nn = np.concatenate([nn, np.repeat(nn[:, -1:], K_NEAR - kq, 1)], 1)
        rk.append(np.full(len(a), k))
        ra.append(a)
        rn.append(b[nn])
    if not rk:
        return out
    kn, a_pt, bn = np.concatenate(rk), np.concatenate(ra), np.concatenate(rn)
    cand = mB["cand"][bn].reshape(len(kn), -1)
    valid = cand >= 0
    cf = np.where(valid, cand, 0)
    tri = XB[kn[:, None, None], mB["F"][cf]]                           # (M, C, 3, 3)
    tri[~valid] = np.inf
    s, c, qq = _signed_closest(PA[kn, a_pt], tri, mB["bnd"][cf], return_point=True)
    f = cf[np.arange(len(kn)), c]
    # soft: the point and the closest point of the other leg both in the soft zone, the latter's place along its leg
    # interpolated over its face (a face across the zone's edge is soft where its closest point lies above it)
    tf = tri[np.arange(len(kn)), c]
    wq = np.clip(_barycentric(qq, tf[:, 0], tf[:, 1], tf[:, 2]), 0.0, 1.0)
    along_q = (wq * mB["along_v"][mB["F"][f]]).sum(-1) / np.maximum(wq.sum(-1), 1e-12)
    soft = mA["soft_point"][a_pt] & (along_q > mB["knee_zone_m"])
    for name, cls in (("soft", soft), ("hard", ~soft)):
        v, ap, fb = out[name]
        if not cls.any():
            continue
        kk, ss, aa, ff = kn[cls], s[cls], a_pt[cls], f[cls]
        order = np.lexsort((ss, kk))                                   # per knot, the smallest first
        kk, ss, aa, ff = kk[order], ss[order], aa[order], ff[order]
        first = np.r_[True, kk[1:] != kk[:-1]]
        v[kk[first]], ap[kk[first]], fb[kk[first]] = ss[first], aa[first], ff[first]
    return out


def gap_from_pose(R_local, transl, betas=None, gender=2, cull_m=CULL_M, knee_zone_m=None):
    """Per knot of local rotations R_local (N, 22, 3, 3) and translations transl (N, 3) (module docstring):
    {'gap_m' (N,) the smallest signed leg-to-leg distance, 'hard_m' (N,) and 'soft_m' (N,) the smallest over the
    hard and over the soft contacts, each cull_m where no surface lies within cull_m (a lower bound), 'exact' (N,)
    bool where gap_m was measured, 'where' and 'where_hard' (N,) the piece pair index (left piece times len(PIECES)
    plus right piece) or -1}; knee_zone_m moves the soft zone's lower edge (None: KNEE_ZONE_M; distal_gap)."""
    mesh = leg_mesh(betas, gender, knee_zone_m)
    R = np.asarray(R_local, dtype=np.float64)
    tr = np.asarray(transl, dtype=np.float64)
    N = R.shape[0]
    res = {k: np.full(N, float(cull_m)) for k in ("gap_m", "hard_m", "soft_m")}
    where = {k: -np.ones(N, np.int64) for k in ("gap_m", "hard_m")}
    nP = len(PIECES)
    delta = cull_m + SLACK_M
    for c0 in range(0, N, CHUNK):
        sl = slice(c0, min(N, c0 + CHUNK))
        sk = skin_legs(R[sl], tr[sl], mesh)
        pts = {s: _points(sk[s], mesh[s]) for s in SIDES}
        box = {s: boxes(sk[s], sk["G"], mesh, s) for s in SIDES}
        inn = {s: _in_boxes(pts[s], box[o], delta) for s, o in (("left", "right"), ("right", "left"))}
        live = np.flatnonzero(inn["left"].any(1) & inn["right"].any(1))
        if not len(live):
            continue
        for b0 in range(0, len(live), BATCH):
            kb = live[b0:b0 + BATCH]
            for qa, qb in (("left", "right"), ("right", "left")):
                FN = _face_normals(sk[qb][kb], mesh[qb])
                r = _direction_gap(pts[qa][kb], pts[qb][kb], sk[qb][kb], FN, mesh[qa], mesh[qb], inn[qa][kb],
                                   inn[qb][kb], cull_m)
                for cls, (v, ia, fb) in r.items():
                    ok = np.isfinite(v)
                    if not ok.any():
                        continue
                    pa = mesh[qa]["point_piece"][np.where(ok, ia, 0)]
                    pb = mesh[qb]["face_piece"][np.where(ok, fb, 0)]
                    pair = np.where(qa == "left", pa * nP + pb, pb * nP + pa)
                    k = c0 + kb
                    key = "soft_m" if cls == "soft" else "hard_m"
                    better = ok & (v < res[key][k])
                    res[key][k[better]] = v[better]
                    if key == "hard_m":
                        where["hard_m"][k[better]] = pair[better]
                    better = ok & (v < res["gap_m"][k])
                    res["gap_m"][k[better]] = v[better]
                    where["gap_m"][k[better]] = pair[better]
    return {**res, "exact": res["gap_m"] < cull_m, "where": where["gap_m"], "where_hard": where["hard_m"]}


def leg_gap(body, cull_m=CULL_M):
    """gap_from_pose of a synth.body.Body's knots (module docstring)."""
    return gap_from_pose(body.local_R().numpy(), body.transl.numpy(), body.betas.numpy(), int(body.gender), cull_m)


def pair_name(where):
    """'left <piece> / right <piece>' of a 'where' index (-1: None)."""
    if where < 0:
        return None
    i, j = divmod(int(where), len(PIECES))
    return f"left {PIECES[i]} / right {PIECES[j]}"


def distal_gap(body, g=None, cull_m=CULL_M):
    """The hard gap of a body with the distal band of the thigh counted hard (DISTAL_BAND_M: the soft zone's edge
    moved to KNEE_ZONE_M + DISTAL_BAND_M above the knee joint): {'hard_m' (N,), 'where_hard' (N,)}, the deepest
    knee, shank, foot or distal-thigh contact per knot. Measured again on the knots where g (leg_gap(body) when None)
    read a soft contact within cull_m, the smaller of the two readings kept; elsewhere g's hard gap (no soft contact
    to count as hard)."""
    g = leg_gap(body, cull_m) if g is None else g
    hard = np.asarray(g["hard_m"], dtype=np.float64).copy()
    where = np.asarray(g["where_hard"]).copy()
    ks = np.flatnonzero(np.asarray(g["soft_m"]) < cull_m)
    if len(ks):
        g2 = gap_from_pose(body.local_R().numpy()[ks], body.transl.numpy()[ks], body.betas.numpy(), int(body.gender),
                           cull_m, knee_zone_m=KNEE_ZONE_M + DISTAL_BAND_M)
        better = g2["hard_m"] < hard[ks]
        hard[ks[better]] = g2["hard_m"][better]
        where[ks[better]] = g2["where_hard"][better]
    return {"hard_m": hard, "where_hard": where}


def report(g, rate=100.0, distal=None):
    """JSON-ready summary of a gap_from_pose result: the smallest hard gap (negative: the deepest penetration of a
    knee, shank or foot), its knot, time and piece pair, the smallest soft gap (the inner thighs), the share of knots
    whose hard gap lies below TOUCH_M (touching), the knots beyond each tolerance; with distal (a distal_gap result)
    the same reading with the thigh's distal band counted hard ('distal_min_m', its time and piece pair, the knots
    deeper than TOL_M), in the record only."""
    hard, soft = np.asarray(g["hard_m"]), np.asarray(g["soft_m"])
    kh, ks = int(np.argmin(hard)), int(np.argmin(soft))
    out = {"hard_min_m": float(hard[kh]), "hard_knot": kh, "hard_t_s": kh / rate,
           "hard_where": pair_name(g["where_hard"][kh]), "soft_min_m": float(soft[ks]), "soft_knot": ks,
           "soft_t_s": ks / rate, "touch_share": float((hard < TOUCH_M).mean()),
           "hard_knots_beyond_tol": int((hard < -TOL_M).sum()),
           "soft_knots_beyond_tol": int((soft < -SOFT_TOL_M).sum()),
           "tol_m": TOL_M, "soft_tol_m": SOFT_TOL_M, "touch_m": TOUCH_M, "measured_range_m": CULL_M}
    if distal is not None:
        dh = np.asarray(distal["hard_m"])
        kd = int(np.argmin(dh))
        out.update({"distal_min_m": float(dh[kd]), "distal_t_s": kd / rate,
                    "distal_where": pair_name(distal["where_hard"][kd]),
                    "distal_knots_beyond_tol": int((dh < -TOL_M).sum()),
                    "distal_band_m": [KNEE_ZONE_M, KNEE_ZONE_M + DISTAL_BAND_M]})
    return out


def validity(body, g=None):
    """The validity entry of a body (module docstring): report(leg_gap(body)) with 'ok' when no hard contact lies
    deeper than TOL_M and no soft one deeper than SOFT_TOL_M; the distal band's reading (distal_gap) recorded beside
    it, outside 'ok'."""
    g = leg_gap(body) if g is None else g
    r = report(g, body.rate, distal_gap(body, g))
    r["ok"] = r["hard_knots_beyond_tol"] == 0 and r["soft_knots_beyond_tol"] == 0
    return r


# ---------------------------------------------------------------- the error against the mesh
def _bary(level):
    w = [(i / level, j / level, (level - i - j) / level) for i in range(level + 1) for j in range(level + 1 - i)]
    return np.asarray(w)


def reference_gap(R_local, transl, betas=None, gender=2, level=REF_LEVEL, radius_m=0.06):
    """(N,) the dense reference of the gap at the given poses: barycentric samples of level `level` on every face of
    each leg against every face of the other leg whose centroid lies within radius_m + 5 cm of the sample (exact
    point to triangle, signed as in gap_from_pose), both directions; samples farther than radius_m from every vertex
    of the other leg are skipped (inf where none is left: the gap exceeds the measured range)."""
    from scipy.spatial import cKDTree
    mesh = leg_mesh(betas, gender)
    sk = skin_legs(np.asarray(R_local, dtype=np.float64), np.asarray(transl, dtype=np.float64), mesh)
    Wb = _bary(level)
    out = np.full(sk["left"].shape[0], np.inf)
    for n in range(len(out)):
        for qa, qb in (("left", "right"), ("right", "left")):
            XA, XB = sk[qa][n], sk[qb][n]
            FA, FB = mesh[qa]["F"], mesh[qb]["F"]
            S = np.einsum("sk,fkc->fsc", Wb, XA[FA]).reshape(-1, 3)
            dn, _ = cKDTree(XB).query(S, k=1)
            S = S[dn < radius_m]
            if not len(S):
                continue
            lists = cKDTree(XB[FB].mean(1)).query_ball_point(S, r=radius_m + 0.05)
            cf = -np.ones((len(S), max(len(x) for x in lists)), np.int64)
            for si, fl in enumerate(lists):
                cf[si, :len(fl)] = fl
            for a0 in range(0, len(S), 256):
                c = cf[a0:a0 + 256]
                valid = c >= 0
                c0 = np.where(valid, c, 0)
                tri = XB[FB[c0]]
                tri[~valid] = np.inf
                s, _ = _signed_closest(S[a0:a0 + 256], tri, mesh[qb]["bnd"][c0])
                out[n] = min(out[n], float(s.min()))
    return out


def capsule_gap(R_local, transl, betas=None, gender=2):
    """(N,) the gap of the fitted capsule model at the given poses: each piece a capsule about its bone (over the
    extent of its vertices along it) whose radius is the mean distance of its skinned vertices from that axis; the
    smallest axis-to-axis distance less both radii over the pairs of pieces."""
    mesh = leg_mesh(betas, gender)
    sk = skin_legs(np.asarray(R_local, dtype=np.float64), np.asarray(transl, dtype=np.float64), mesh)
    cap = {}
    for side in SIDES:
        a0, a1, rad = [], [], []
        J = sk["joints"]
        for pc in mesh[side]["pieces"]:
            j0, j1 = pc["joints"]
            ax = J[:, j1] - J[:, j0]
            p0, p1 = J[:, j0] + pc["s"][0] * ax, J[:, j0] + pc["s"][1] * ax
            x = sk[side][:, pc["verts"]]
            d = p1 - p0
            t = np.clip(_dot(x - p0[:, None], d[:, None]) / _dot(d, d)[:, None], 0.0, 1.0)
            rad.append(np.linalg.norm(x - (p0[:, None] + t[..., None] * d[:, None]), axis=-1).mean(1))
            a0.append(p0)
            a1.append(p1)
        cap[side] = (a0, a1, rad)
    out = np.full(sk["left"].shape[0], np.inf)
    for i in range(len(PIECES)):
        for j in range(len(PIECES)):
            d = segment_distance(cap["left"][0][i], cap["left"][1][i], cap["right"][0][j], cap["right"][1][j])
            out = np.minimum(out, d - cap["left"][2][i] - cap["right"][2][j])
    return out


def mesh_error(R_local, transl, betas=None, gender=2):
    """JSON-ready comparison of the mesh gap (gap_from_pose) and of the fitted capsule gap with the dense reference
    at the given poses: per pose the three gaps; the largest absolute error of each over the poses whose reference
    lies within CULL_M (the measured range)."""
    g = gap_from_pose(R_local, transl, betas, gender)
    ref = reference_gap(R_local, transl, betas, gender)
    cg = capsule_gap(R_local, transl, betas, gender)
    inr = ref < CULL_M
    return {"gap_m": g["gap_m"].tolist(), "reference_m": ref.tolist(), "capsule_gap_m": cg.tolist(),
            "max_error_m": float(np.abs(g["gap_m"] - ref)[inr].max()) if inr.any() else None,
            "capsule_max_error_m": float(np.abs(cg - ref)[inr].max()) if inr.any() else None,
            "n_in_range": int(inr.sum())}


def rest_gap(betas=None, gender=2):
    """The gap of the rest pose (identity rotations, no translation): {'gap_m', 'where'}, measured up to 0.2 m."""
    R = np.repeat(np.eye(3)[None, None], 22, 1)
    g = gap_from_pose(R, np.zeros((1, 3)), betas, gender, cull_m=0.2)
    return {"gap_m": float(g["gap_m"][0]), "where": pair_name(g["where"][0])}
