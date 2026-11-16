"""Foot contact of the leg operators (the SP1 open issue: leg postures and tasks were not contact-consistent).

Sole: per foot, the cloud of mesh vertices of the selected body model (body.body_model: the open body by default)
whose largest skinning weight lies on the ankle or the foot joint of that side (on the open body: the dominant Anny
bone owned by that joint, design D10) (7 and 10 on the left, 8 and 11 on the right) and whose rest height lies within SOLE_BAND_M of
the lowest of them (sole_cloud), each carried rigidly by its dominant joint from its rest offset in the body's
shape. Since no operator changes the foot joint's own rotation (10, 11), the ankle and foot segments move as one
rigid foot, and the cloud follows the skinned mesh up to the small weights of other joints (and, on the SMPL+H
reference, its pose blend shapes; tests/test_synth_pathology.py checks the mesh too). A foot's floor is the height of the lowest point of
its cloud in the base body at that knot; the foot stands on the floor when its lowest cloud point is there.

Rolling (roll): the base foot turned by a pitch phi about its lateral axis (the horizontal axis normal to its
forward direction; positive phi lowers the toes) without slipping: it turns about its lowest point until the
next point of the cloud in the direction of the turn reaches the floor, then about that one; only the vertices
of the cloud's convex hull in the foot's sagittal plane can touch, so the hull is what rolls. The lowest point
stays on the floor and no point of the cloud goes below it.

Two ways to keep a moved foot planted, by what the operator defines:
- the ankle (plant_ankle: ankle postures of dystonia.py, toe tapping of bradykinesia.py): the ankle's local
  rotation is the operator's; the foot rolls on the floor and the leg above it follows, solved by two-bone
  inverse kinematics with the hip fixed and the knee turned toward its base position (each segment turned from
  the base by its minimal rotation); the rolling pitch is solved per knot (a grid, then bisection) so that the
  ankle's flexion (the x component of its local rotation, the flexion axis of the table) is the operator's. The
  other components of the ankle's rotation differ from the operator's by the off-axis part of the leg's motion,
  reported (offaxis_deg).
- the knee or the hip (plant_compensate: knee and hip postures of dystonia.py): the leg down to the ankle is the
  operator's; the ankle flexes about its x axis in the parent frame (the toe-tapping axis) by the smallest angle
  that puts the foot's lowest cloud point on the floor (a grid, then bisection), within the ankle's room in the
  joint-limit table.
Mesh correction: the rigid cloud drives both solves, and the skinned mesh corrects them: after a solve, the
cloud is skinned by the body model's one skinning call (skin_points: body_model().skin on the cloud, hands at
rest), the gap between its lowest skinned point and the base's is taken off the floor target, and the solve is
repeated (MESH_PASSES passes), so that the lowest point of the mesh, not only of the rigid cloud, stays on the
floor. A knot where neither way works is reported infeasible; the operator then scales the posture down
(dystonia.py). The largest knot-to-knot change of the rolling pitch or of the compensation (phi_max_step_deg,
alpha_max_step_deg, the first step taken from the base) shows a jump: on a straight standing leg with the hip
fixed, a planted plantarflexion needs a finite knee bend before any ankle angle (the leg must shorten as the heel
rises), so its first knot jumps; seated, or with the knee bent, it starts from zero. A raised foot (leg agility; hip
flexion, whose operator lifts the leg by design) is not planted; only its not going below the floor is checked
(sole_report, on the skinned mesh).

What planting costs (review of 2026-10-05), reported, never hidden:
- foot_pose_report: the heel and the toe of the moved foot (the sole cloud vertices within HEEL_BAND_M of its most
  posterior and most anterior rest vertex, skinned) against the base at the same knot, the pitch of the
  heel-to-toe line, and how far the hip, knee and ankle turned from the base. With the pelvis fixed, a planted knee
  or hip posture on a standing leg is held on the toes (the leg shortens, the compensating ankle flexion puts the
  toes on the floor and lifts the heel); an ankle posture bends the knee and turns the hip above it.
- step_report: the largest knot-to-knot rotation of the moved leg's hip, knee and ankle. A planted solve can jump
  (the straight standing leg above); a joint turning faster than MAX_LEG_SPEED_DEG_S (a module value) and faster
  than in the base fails the check, so that no jump reaches a stimulus unflagged.
"""
import functools

import numpy as np
import torch

from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle
from .body import rest_joints, rotation_angle_deg
from .ik import rmin

LEG = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}           # hip, knee, ankle, foot
SOLE_BAND_M = 0.05               # rest height band of the sole cloud above the foot's lowest vertex
TOL_M = 0.005                    # the contact tolerance of the fix: no lift and no penetration beyond 5 mm
HEEL_BAND_M = 0.03               # heel and toe sets: cloud vertices within this of its most posterior or anterior one
MAX_LEG_SPEED_DEG_S = 500.0      # [module] largest angular speed of a planted leg joint before a step is a jump
UP = np.array([0.0, 1.0, 0.0])
GRID_DEG = np.arange(-70.0, 80.0 + 1e-9, 2.5)
POLE_GAIN = 3.0                  # weight of the forward direction as the leg bends past its base (_two_bone)
MESH_PASSES = 4                  # solves corrected by the skinned mesh (module docstring)
MESH_TOL_M = 2e-4                # a pass stops early when every knot's skinned lowest point is within this


# ---------------------------------------------------------------- the sole cloud
def _name():
    from .body import model_name
    return model_name()


def _neutral(gender, betas=None):
    """The catalogue's one shape (gender code 2, betas zero), on either body model."""
    if int(gender) != 2 or (betas is not None and any(float(x) for x in np.asarray(betas).reshape(-1))):
        raise ValueError("the sole cloud is measured on the neutral shape only (betas 0, gender code 2)")


def sole_vertices(gender=2):
    """{side: vertex ids (M,)} of the sole cloud (module docstring) on the rest mesh of the selected body model."""
    _neutral(gender)
    return _sole_vertices(_name())


@functools.lru_cache(maxsize=2)
def _sole_vertices(name):
    from .viewer import dominant_joint, rest_vertices
    v = rest_vertices()
    dom = dominant_joint()
    out = {}
    for side, (_, _, ankle, foot) in LEG.items():
        ids = np.flatnonzero(np.isin(dom, (ankle, foot)))
        low = v[ids, 1].min()
        out[side] = ids[v[ids, 1] <= low + SOLE_BAND_M]
    return out


@functools.lru_cache(maxsize=2)
def _sole_rest(name):
    """{side: (joints (M,), offsets (M, 3))}: each cloud vertex's dominant joint and its rest offset from it."""
    from .viewer import dominant_joint, rest_vertices
    dom = dominant_joint()
    r = rest_joints().numpy()
    v_all = rest_vertices()
    out = {}
    for side, ids in _sole_vertices(name).items():
        joints = dom[ids]
        out[side] = (joints, v_all[ids] - r[joints])
    return out


def sole_rest(body, side):
    _neutral(body.gender, body.betas.numpy())
    return _sole_rest(_name())[side]


def sole_points(J, G, joints, offsets):
    """(T, M, 3) world cloud points from joint positions J (T, 22, 3) and global rotations G (T, 22, 3, 3)."""
    return J[:, joints] + np.einsum("tkij,kj->tki", G[:, joints], offsets)


def body_sole_points(body, side, fk=None):
    J, G = body.fk() if fk is None else fk
    joints, off = sole_rest(body, side)
    return sole_points(np.asarray(J), np.asarray(G), joints, off)


def sole_low(body, fk=None):
    """(T, 2) height (m) of the lowest cloud point of the left and right foot."""
    fk = body.fk() if fk is None else fk
    return np.stack([body_sole_points(body, s, fk)[..., 1].min(-1) for s in ("left", "right")], -1)


def skin_points(R_local, transl, betas, gender, side):
    """(N, M, 3) the sole cloud of side skinned for local rotations R_local (N, 22, 3, 3) and translations
    transl (N, 3): the body model's one skinning call on the cloud's vertices (hands at rest)."""
    from .body import body_model
    _neutral(gender, betas)
    return body_model().skin(np.asarray(R_local, dtype=np.float64), np.asarray(transl, dtype=np.float64), None,
                             ids=sole_vertices(int(gender))[side])


def mesh_sole_low(body, chunk=512, knots=None):
    """(T, 2) height (m) of the lowest skinned point of the left and right sole cloud (the mesh's); with knots
    (indices), (len(knots), 2) at those knots only."""
    R = body.local_R().numpy()
    tr = body.transl.numpy()
    if knots is not None:
        k = np.asarray(knots, dtype=np.int64).reshape(-1)
        R, tr = R[k], tr[k]
    out = np.empty((len(R), 2))
    for a in range(0, len(R), chunk):
        b = slice(a, a + chunk)
        for i, side in enumerate(("left", "right")):
            out[b, i] = skin_points(R[b], tr[b], body.betas.numpy(), body.gender, side)[..., 1].min(-1)
    return out


def heel_toe(gender=2):
    """{side: (heel (H,), toe (K,))}: positions in the side's sole cloud (sole_vertices) of the vertices within
    HEEL_BAND_M of its most posterior and of its most anterior rest vertex (the rest mesh faces +z)."""
    _neutral(gender)
    return _heel_toe(_name())


@functools.lru_cache(maxsize=2)
def _heel_toe(name):
    from .viewer import rest_vertices
    v = rest_vertices()
    out = {}
    for side, ids in _sole_vertices(name).items():
        z = v[ids, 2]
        out[side] = (np.flatnonzero(z <= z.min() + HEEL_BAND_M), np.flatnonzero(z >= z.max() - HEEL_BAND_M))
    return out


def _heel_toe_pitch(body, side, chunk=512):
    """(heel y (T,), toe y (T,), pitch deg (T,)) of a body's foot on the skinned mesh: the mean height of the heel
    and of the toe set, and the angle of the heel-to-toe line against the horizontal (positive heel up)."""
    h, t = heel_toe(int(body.gender))[side]
    R, tr = body.local_R().numpy(), body.transl.numpy()
    hy, ty, pitch = (np.empty(body.T) for _ in range(3))
    for a in range(0, body.T, chunk):
        b = slice(a, a + chunk)
        x = skin_points(R[b], tr[b], body.betas.numpy(), body.gender, side)
        hp, tp = x[:, h].mean(1), x[:, t].mean(1)
        d = hp - tp
        hy[b], ty[b] = hp[:, 1], tp[:, 1]
        pitch[b] = np.degrees(np.arctan2(d[:, 1], np.linalg.norm(d[:, [0, 2]], axis=1)))
    return hy, ty, pitch


def foot_pose_report(base, out, side):
    """What planting the foot of side costs (module docstring), JSON-ready: the largest heel and toe lift (m) on
    the skinned mesh against the base at the same knot, the pitch change of the heel-to-toe line (deg, positive
    heel up, at the knot of its largest magnitude) and the largest rotation (deg) of the hip, knee and ankle local
    rotations away from the base's."""
    hb, tb, pb = _heel_toe_pitch(base, side)
    ho, to, po = _heel_toe_pitch(out, side)
    dp = po - pb
    k = int(np.argmax(np.abs(dp)))
    rep = {"side": side, "heel_lift_max_m": float((ho - hb).max()), "toe_lift_max_m": float((to - tb).max()),
           "pitch_change_deg": float(dp[k]), "pitch_knot": k, "heel_band_m": HEEL_BAND_M}
    Rb, Ro = base.local_R(), out.local_R()
    for name, j in zip(("hip", "knee", "ankle"), LEG[side][:3]):
        rep[f"{name}_change_deg"] = float(rotation_angle_deg(Rb[:, j].transpose(-1, -2) @ Ro[:, j]).max())
    return rep


def step_report(base, out, side, bound_deg_s=MAX_LEG_SPEED_DEG_S):
    """The continuity check of a leg operator's output (module docstring), JSON-ready: per joint of the moved leg
    (hip, knee, ankle) the largest knot-to-knot rotation (deg) of its local rotation in the output and in the base,
    and the knot after which the output's occurs; ok when, for every joint, that step times the rate stays within
    bound_deg_s or within the base's own largest step."""
    rep = {"side": side, "bound_deg_s": float(bound_deg_s), "rate_hz": float(out.rate), "joints": {}}
    ok = True
    Rb, Ro = base.local_R(), out.local_R()
    for name, j in zip(("hip", "knee", "ankle"), LEG[side][:3]):
        so = rotation_angle_deg(Ro[:-1, j].transpose(-1, -2) @ Ro[1:, j]).numpy() if out.T > 1 else np.zeros(1)
        sb = rotation_angle_deg(Rb[:-1, j].transpose(-1, -2) @ Rb[1:, j]).numpy() if base.T > 1 else np.zeros(1)
        mo, mb = float(so.max()), float(sb.max())
        rep["joints"][name] = {"max_step_deg": mo, "base_max_step_deg": mb, "knot": int(so.argmax())}
        ok = ok and (mo * out.rate <= bound_deg_s + 1e-9 or mo <= mb + 1e-9)
    rep["max_step_deg"] = max(v["max_step_deg"] for v in rep["joints"].values())
    rep["ok"] = bool(ok)
    return rep


def stance_knots(body, side, tol=0.01):
    """(T,) bool: knots where the body's foot of side stands on the floor: its lowest cloud point within tol of
    the 2nd percentile of both feet's lowest cloud points over the body."""
    low = sole_low(body)
    floor = float(np.percentile(low, 2.0))
    return low[:, 0 if side == "left" else 1] <= floor + tol


# ---------------------------------------------------------------- small rotation helpers (numpy, float64)
def _rot(axis, ang):
    """(N, 3, 3) rotations about unit axes (N, 3) by angles (N,) rad."""
    return axis_angle_to_matrix(torch.from_numpy(np.asarray(axis) * np.asarray(ang)[:, None])).numpy()


def _aa(R):
    return matrix_to_axis_angle(torch.from_numpy(np.ascontiguousarray(R))).numpy()


def _rmin(a, b):
    return rmin(torch.from_numpy(a), torch.from_numpy(b)).numpy()


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------- rolling
def foot_axes(G_foot):
    """(forward (N, 3), lateral (N, 3)) of a foot: the segment's z axis on the floor, and up x forward."""
    f = G_foot[:, :, 2].copy()
    f[:, 1] = 0.0
    f = _unit(f)
    return f, np.cross(UP, f)


def _hull(S, Y):
    """Indices of the 2D convex hull of points (S, Y) (Andrew's monotone chain)."""
    order = np.lexsort((Y, S))
    pts = list(order)

    def cross(o, a, b):
        return (S[a] - S[o]) * (Y[b] - Y[o]) - (Y[a] - Y[o]) * (S[b] - S[o])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def hull_points(P, G):
    """(N, K, 3) the vertices of each row's cloud P (N, M, 3) on its convex hull in the foot's sagittal plane
    (padded by repeating the first vertex)."""
    f, _ = foot_axes(G)
    S = np.einsum("nmi,ni->nm", P, f)
    Y = P[..., 1]
    hulls = [_hull(S[i], Y[i]) for i in range(len(P))]
    K = max(len(h) for h in hulls)
    idx = np.stack([np.r_[h, np.repeat(h[:1], K - len(h))] for h in hulls])
    return np.take_along_axis(P, idx[..., None], 1)


def roll(P, X, G, phi, floor):
    """The base foot rolled by pitch phi (N,) rad on the floor (module docstring). P (N, K, 3) the points that can
    touch (hull_points), X (N, M, 3) other points carried by the foot (the ankle), G (N, 3, 3) the foot's global
    rotation, floor (N,) its floor height. Returns (P' (N, K, 3), X' (N, M, 3), G' (N, 3, 3), pivot (N,))."""
    N, K = P.shape[:2]
    f, u = foot_axes(G)
    rows = np.arange(N)
    p0 = np.argmin(P[..., 1], 1)
    O = P[rows, p0].copy()
    O[:, 1] = floor
    allp = np.concatenate([P, X], 1)
    rel = allp - O[:, None]
    S = np.einsum("nki,ni->nk", rel, f)
    Y = allp[..., 1] - floor[:, None]
    L = np.einsum("nki,ni->nk", rel, u)
    piv = p0.copy()
    done = np.zeros(N)
    for _ in range(K + 2):
        rem = phi - done
        sp = S[rows, piv]
        ds = S[:, :K] - sp[:, None]
        y = Y[:, :K]
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = np.arctan(y / ds)
        ok = (np.sign(ds) == np.sign(rem)[:, None]) & (np.abs(ds) > 1e-12) & (y >= -1e-12)
        ok[rows, piv] = False
        hit = np.where(ok, np.abs(hit), np.inf)
        j = np.argmin(hit, 1)
        step = hit[rows, j]
        last = np.abs(rem) <= step
        d = np.where(last, rem, np.sign(rem) * np.where(np.isfinite(step), step, 0.0))
        c, s = np.cos(d)[:, None], np.sin(d)[:, None]
        dS = S - sp[:, None]
        S, Y = sp[:, None] + dS * c + Y * s, Y * c - dS * s
        done = done + d
        piv = np.where(last, piv, j)
        if last.all():
            break
    Y[rows, piv] = 0.0                                                 # the pivot on the floor exactly
    out = O[:, None] + f[:, None] * S[..., None] + u[:, None] * L[..., None]
    out[..., 1] = floor[:, None] + Y
    return out[:, :K], out[:, K:], _rot(u, phi) @ G, piv


# ---------------------------------------------------------------- the leg
def _two_bone(H, A, K0, fwd, L1, L2):
    """Knee positions for hips H, ankles A (n, 3), base knees K0 (n, 3) and the forward direction fwd (n, 3); ok
    (n,) False where the ankle lies beyond the leg's reach. The knee goes on the circle of points at L1 from H and
    L2 from A toward p = w0 + POLE_GAIN max(h - |w0|, 0) fwd, with w0 the base knee's offset from the hip-ankle
    axis and h the circle's radius: at the base's own hip-ankle distance the knee is the base knee (h = |w0|), and
    as the leg bends further it bends forward (a nearly straight leg has an ill-defined w0)."""
    d = A - H
    dist = np.linalg.norm(d, axis=1, keepdims=True)
    ok = (dist[:, 0] < L1 + L2 - 1e-9) & (dist[:, 0] > abs(L1 - L2) + 1e-9)
    u = d / np.maximum(dist, 1e-12)
    along = (L1 ** 2 - L2 ** 2 + dist ** 2) / (2 * np.maximum(dist, 1e-12))
    h = np.sqrt(np.clip(L1 ** 2 - along ** 2, 0.0, None))
    w0 = K0 - H
    w0 = w0 - (w0 * u).sum(1, keepdims=True) * u
    r0 = np.linalg.norm(w0, axis=1, keepdims=True)
    p = w0 + POLE_GAIN * np.clip(h - r0, 0.0, None) * fwd
    p = p - (p * u).sum(1, keepdims=True) * u
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    return H + along * u + h * p, ok


def _leg_locals(base_fk, k, side, A, G_foot, rest):
    """Local rotations (hip, knee, ankle) (n, 3, 3) of the leg of side at knots k with the hip fixed, the ankle at
    A (n, 3) and the foot's global rotation G_foot (n, 3, 3): two-bone IK (_two_bone: the base knee at the base's
    leg length, bending forward along the pelvis heading as the leg shortens), each segment turned from the base
    by its minimal rotation. ok (n,) False beyond reach."""
    h, kn, a, _ = LEG[side]
    J, G = base_fk
    L1 = float(np.linalg.norm(rest[kn] - rest[h]))
    L2 = float(np.linalg.norm(rest[a] - rest[kn]))
    H = J[k, h]
    fwd = G[k, 0][:, :, 2].copy()
    fwd[:, 1] = 0.0
    K, ok = _two_bone(H, A, J[k, kn], _unit(fwd), L1, L2)
    G_th = _rmin(_unit(J[k, kn] - H), _unit(K - H)) @ G[k, h]
    G_sh = _rmin(_unit(J[k, a] - J[k, kn]), _unit(A - K)) @ G[k, kn]
    parent = G[k, 0]                                                    # the hips' parent is the pelvis
    return parent.transpose(0, 2, 1) @ G_th, G_th.transpose(0, 2, 1) @ G_sh, G_sh.transpose(0, 2, 1) @ G_foot, ok


def _bisect(f, lo, hi, iters=40):
    """Roots of a vectorized f on brackets [lo, hi] (n,) where f changes sign."""
    flo = f(lo)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        fm = f(m)
        same = np.sign(fm) == np.sign(flo)
        lo, flo = np.where(same, m, lo), np.where(same, fm, flo)
        hi = np.where(same, hi, m)
    return 0.5 * (lo + hi)


def _bracket(values, grid, prefer):
    """Per row of values (n, G) on grid (G,): the bracket [a, b] of the sign change whose midpoint lies nearest
    prefer (n,); found False where no sign change exists (an exact zero counts as a change)."""
    sv = np.sign(values)
    ch = (sv[:, :-1] * sv[:, 1:] <= 0) & np.isfinite(values[:, :-1]) & np.isfinite(values[:, 1:])
    mid = 0.5 * (grid[:-1] + grid[1:])
    dist = np.where(ch, np.abs(mid[None] - prefer[:, None]), np.inf)
    i = np.argmin(dist, 1)
    found = np.isfinite(dist[np.arange(len(i)), i])
    return grid[i], grid[i + 1], found


def _unique_rows(*arrays):
    """(index of the first occurrence of each distinct row, inverse) over the concatenated rows of arrays."""
    key = np.concatenate([np.round(np.asarray(a).reshape(len(a), -1), 12) for a in arrays], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first, inv.reshape(-1)


def plant_ankle(base, R_out, side, knots):
    """Mode 'ankle' (module docstring). base: the operator's input body (its foot on the floor at the knots);
    R_out (T, 22, 3, 3): the operator's local rotations (the ankle row carries the posture or the task); knots
    (T,) bool: where to plant. Returns (R (T, 22, 3, 3) with the hip, knee and ankle rows of side re-solved at
    the knots, feasible (T,) bool, info)."""
    h, kn, a, ft = LEG[side]
    J, G = (x.numpy() for x in base.fk())
    rest = base.rest().numpy()
    R_out = np.asarray(R_out, dtype=np.float64)
    R = R_out.copy()
    feas = np.ones(base.T, bool)
    kk = np.flatnonzero(knots)
    info = {"mode": "ankle", "side": side, "knots": int(len(kk))}
    if not len(kk):
        return R, feas, info
    target = R_out[kk, a]
    first, inv = _unique_rows(target, J[kk], G[kk, h], G[kk, kn], G[kk, a])
    k = kk[first]
    n = len(k)
    joints, off = sole_rest(base, side)
    cloud = sole_points(J[k], G[k], joints, off)
    floor = cloud[..., 1].min(1)
    P = hull_points(cloud, G[k, a])
    Ab = J[k, a][:, None]
    Gf = G[k, a]
    tgt = target[first]
    base_flex = _aa(base.local_R().numpy()[k, a])[:, 0]
    betas, gender, tr = base.betas.numpy(), base.gender, base.transl.numpy()[k]
    mesh_floor = skin_points(base.local_R().numpy()[k], tr, betas, gender, side)[..., 1].min(1)
    want = _aa(tgt)[:, 0] - base_flex

    def solve(delta):
        def leg(phi):
            _, X, Gr, piv = roll(P, Ab, Gf, phi, floor)
            Rh, Rk, Ra, ok = _leg_locals((J, G), k, side, X[:, 0] + delta[:, None] * UP, Gr, rest)
            return Rh, Rk, Ra, ok, piv

        def resid(phi):
            _, _, Ra, ok, _ = leg(phi)
            return np.where(ok, _aa(tgt.transpose(0, 2, 1) @ Ra)[:, 0], np.nan)
        vals = np.stack([resid(np.full(n, np.radians(g))) for g in GRID_DEG], 1)
        lo, hi, found = _bracket(vals, np.radians(GRID_DEG), want)
        phi = _bisect(resid, lo, hi)
        Rh, Rk, Ra, ok, piv = leg(phi)
        return phi, Rh, Rk, Ra, found & ok & np.isfinite(resid(phi)), piv

    delta = np.zeros(n)
    for _ in range(MESH_PASSES):
        phi, Rh, Rk, Ra, good, piv = solve(delta)
        Rk_ = R_out[k].copy()
        Rk_[:, h], Rk_[:, kn], Rk_[:, a] = Rh, Rk, Ra
        gap = skin_points(Rk_, tr, betas, gender, side)[..., 1].min(1) - mesh_floor
        if np.all(np.abs(gap[good]) <= MESH_TOL_M):
            break
        delta = delta - np.where(good, gap, 0.0)
    E = _aa(tgt.transpose(0, 2, 1) @ Ra)
    R[kk, h], R[kk, kn], R[kk, a] = Rh[inv], Rk[inv], Ra[inv]
    gi = good[inv]
    bad = kk[~gi]
    R[bad] = R_out[bad]
    feas[bad] = False
    if good.any():
        ph = np.degrees(phi)[inv][gi]
        info["phi_max_step_deg"] = float(np.abs(np.diff(np.r_[0.0, ph])).max())   # from the base, knot to knot
        info.update(phi_deg_range=[float(np.degrees(phi[good]).min()), float(np.degrees(phi[good]).max())],
                    offaxis_deg=float(np.degrees(np.abs(E[good][:, 1:]).max())),
                    flex_error_deg=float(np.degrees(np.abs(E[good][:, 0]).max())),
                    mesh_gap_mm=float(1000 * np.abs(gap[good]).max()),
                    mesh_correction_mm=float(1000 * np.abs(delta[good]).max()))
    info.update(infeasible=int((~gi).sum()), unique_knots=int(n))
    return R, feas, info


def plant_compensate(base, R_out, side, knots):
    """Mode 'compensate' (module docstring): the ankle of side flexed about its parent-frame x axis, at the
    knots, by the smallest angle that puts the foot's lowest point on the base's floor, within the ankle's room
    in the joint-limit table. Returns (R, feasible (T,), info, ankle offset deg (T,))."""
    from ..body.joints import fk
    from .body import room_deg
    h, kn, a, ft = LEG[side]
    R_out = np.asarray(R_out, dtype=np.float64)
    R = R_out.copy()
    feas = np.ones(base.T, bool)
    alpha_all = np.zeros(base.T)
    kk = np.flatnonzero(knots)
    info = {"mode": "compensate", "side": side, "knots": int(len(kk))}
    if not len(kk):
        return R, feas, info, alpha_all
    rest = base.rest()
    Rb = base.local_R().numpy()
    betas, gender = base.betas.numpy(), base.gender
    first, inv = _unique_rows(R_out[kk][:, [0, 3, 6, 9, h, kn, a]], Rb[kk][:, [h, kn, a]], base.transl[kk].numpy())
    sel = kk[first]
    n = len(sel)
    tr = base.transl.numpy()[sel]
    Jb, Gb = (x.numpy() for x in base.fk())
    joints, off = sole_rest(base, side)
    fl = sole_points(Jb[sel], Gb[sel], joints, off)[..., 1].min(1)
    mesh_floor = skin_points(Rb[sel], tr, betas, gender, side)[..., 1].min(1)
    with torch.no_grad():
        Jo, Go = (x.numpy() for x in fk(torch.from_numpy(R_out[sel]), base.transl[sel], rest))
    Gp = Go[:, kn]                                                     # the ankle's parent: the shank
    Ra0 = R_out[sel, a]
    ex = np.repeat(np.array([[1.0, 0.0, 0.0]]), n, 0)
    foot_bone = (rest[ft] - rest[a]).numpy()
    on_ankle = joints == a

    def rigid_low(alpha):
        G7 = Gp @ _rot(ex, alpha) @ Ra0
        G10 = G7 @ R_out[sel, ft]
        J10 = Jo[:, a] + np.einsum("nij,j->ni", G7, foot_bone)
        p7 = Jo[:, a][:, None] + np.einsum("nij,kj->nki", G7, off[on_ankle])
        p10 = J10[:, None] + np.einsum("nij,kj->nki", G10, off[~on_ankle])
        return np.minimum(p7[..., 1].min(1), p10[..., 1].min(1))

    r_plus = np.array([room_deg(torch.from_numpy(Ra0[i]), a, "x", 1.0, "parent") for i in range(n)])
    r_minus = np.array([room_deg(torch.from_numpy(Ra0[i]), a, "x", -1.0, "parent") for i in range(n)])
    grid = np.radians(GRID_DEG)
    inside = (grid[None] <= np.radians(r_plus)[:, None] + 1e-12) & (grid[None] >= -np.radians(r_minus)[:, None] - 1e-12)
    lows = np.stack([rigid_low(np.full(n, g)) for g in grid], 1)
    low0 = rigid_low(np.zeros(n))

    def solve(target):
        vals = np.where(inside, lows - target[:, None], np.nan)
        zero = np.abs(low0 - target) <= 1e-12
        lo, hi, found = _bracket(vals, grid, np.zeros(n))
        alpha = np.where(zero, 0.0, _bisect(lambda x: rigid_low(x) - target, lo, hi))
        return alpha, found | zero

    target = fl.copy()
    for _ in range(MESH_PASSES):
        alpha, found = solve(target)
        Ra = _rot(ex, np.where(found, alpha, 0.0)) @ Ra0
        Rs = R_out[sel].copy()
        Rs[:, a] = Ra
        gap = skin_points(Rs, tr, betas, gender, side)[..., 1].min(1) - mesh_floor
        if np.all(np.abs(gap[found]) <= MESH_TOL_M):
            break
        target = target - np.where(found, gap, 0.0)
    R[kk, a] = Ra[inv]
    gi = found[inv]
    bad = kk[~gi]
    R[bad] = R_out[bad]
    feas[bad] = False
    alpha_all[kk] = np.degrees(np.where(found, alpha, 0.0))[inv]
    if found.any():
        info["alpha_max_step_deg"] = float(np.abs(np.diff(np.r_[0.0, alpha_all[kk][gi]])).max())
        info.update(alpha_deg_range=[float(np.degrees(alpha[found]).min()), float(np.degrees(alpha[found]).max())],
                    mesh_gap_mm=float(1000 * np.abs(gap[found]).max()),
                    mesh_correction_mm=float(1000 * np.abs(target - fl)[found].max()))
    info.update(infeasible=int((~gi).sum()), unique_knots=int(n))
    return R, feas, info, alpha_all


def sole_report(base, out, side, planted, tol=TOL_M):
    """The contact check of a leg operator's output (JSON-ready), on the skinned mesh (mesh_sole_low): for the
    foot of side, its lowest point against the base's at every knot (planted: within tol above or below; raised:
    never more than tol below), and the other foot's against its base."""
    lb, lo_ = mesh_sole_low(base), mesh_sole_low(out)
    i = 0 if side == "left" else 1
    o = 1 - i
    d = lo_[:, i] - lb[:, i]
    do = lo_[:, o] - lb[:, o]
    rep = {"side": side, "planted": bool(planted), "tol_m": tol, "moved_low_min_m": float(d.min()),
           "moved_low_max_m": float(d.max()), "other_low_absmax_m": float(np.abs(do).max())}
    rep["ok"] = bool((np.abs(do) <= tol).all() and (d >= -tol).all() and (not planted or (d <= tol).all()))
    return rep
