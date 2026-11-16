"""Freezing of gait episodes (docs/proposals/SYNTH_PATIENT_DRAFT.md, section 4).

Inverse recipe (draft section 4, plan SP1): an episode of known onset, duration and manifestation inserted
into a walking or turning take. The progression is slowed to a stop by a time warp of the base take: the
output at time t shows the base body at base time tau(t), tau advancing at a rate that falls from 1 to 0 by a
raised cosine over ramp_in_s, stays at 0 during the hold, and rises back over ramp_out_s. A time warp keeps
every pose of the take and slows every velocity by the same factor, so a stance foot stays planted and no
skate is added; after the episode the take resumes where it stopped, every later knot of the output being a
knot of the base exactly, shifted by the episode's duration (the tail of the base past the output length is
dropped). The episode is the interval where the rate is below one half: onset and offset at the ramp
midpoints, duration D = offset - onset, which equals the base time lost (DR-10.5: onset, offset, duration).
The hold freezes the base pose at base time onset, snapped to a knot of double support (both base contact
labels set), so that the frozen body stands on both feet. A take whose contact labels hold no double support in
its straight walking (an AMASS walk at about 1 m/s: the body-store rule labels a foot in contact only below
0.3 m/s, and at heel strike and toe-off the joints move faster) may use support 'quasi' (sp4-v3): where no
labelled double support lies within the search window, the onset is the knot whose higher foot lies lowest (the
lower of each foot's ankle and foot joint heights, the larger of the two feet, smallest). Support 'mesh'
(review of 2026-10-05) reads the skinned mesh instead of the labels and the joints: the onset is the knot of the
search window nearest the requested time where the two feet's lowest skinned sole points (contact.mesh_sole_low)
lie within MESH_SUPPORT_TOL_M of each other in height (where none does, the knot where they lie closest), so
that the frozen body, as a viewer draws it, stands on both soles (on the AMASS walk of the viewer, labelled
double-support knots can leave one sole well above the other). Two soles level with each other are not yet a
stance: at toe-off the trailing foot's lowest point is its toe, level with the leading sole, while its heel is
far up (the review of the English page, wave SH1d: on the AMASS walk, holds of 10 to 15 s on the toes of one
foot, the heel 22 to 24 cm up). Since exam_v7 support 'mesh' also asks both heels down: the lowest skinned heel
point of each foot (contact.heel_toe, the heel set of the sole cloud) lies within MESH_HEEL_TOL_M of the lower of
the two soles (heel_lift_mesh). The onset is the knot of the search window nearest the requested time that meets
both tolerances; where none does, the knot of the window that misses them least (the larger of the sole gap over
its tolerance and the higher heel over its own). The search window never leaves the episode's context
(plan_episodes, since the review of lane exam, wave SH1d: on the AMASS walk a fallback to the first qualifying
knot after the window carried the straight-walking freeze of grades 3 and 4 past its straight segment, into a
turn): for a straight-walking episode the straight segment, shortened at each end by the base time its ramps
take (half the ramp: the rate falls from 1 to 0 by a raised cosine), so that the whole episode, ramps included,
freezes straight walking; for a trigger episode the event's interval widened by search_s at each end (a freeze
on the approach to a turn or a doorway, or at its start, is the trigger's). On a natural walk both heels are
rarely down together (the trailing heel rises before the leading foot lands), so the least miss is common: its
heel lift is in the truth. Every episode's truth records its support, whether its onset knot is labelled double
support, the higher-foot height of the joints there (higher_foot_joint_m), the gap between the two soles on the
skinned mesh (foot_gap_mesh_m), each heel's lift over the lower sole there (heel_lift_mesh_m, left and right; both
None on a body of another shape than the catalogue's one, which contact.py's sole cloud does not read: an AMASS take),
the window its onset was drawn in (context_window_s, planned 'mesh' episodes, None otherwise) and its drift from the
requested time (drift_s, onset minus requested, None when no time was requested).

Manifestation during the hold:
- 'akinesia': the frozen pose, no motion.
- 'trembling': in-place oscillation of the legs in the freeze band (3 to 8 Hz, Moore et al. 2008; frequency an
  operator parameter, default 5.5 Hz) with both feet locked: the pelvis sinks and rises by tremble_m peak to
  peak below its frozen height ('in_phase', both knees flex together), or rolls about the forward axis so
  that each hip moves vertically by tremble_m peak to peak in opposite phase, the pelvis lowered by half that
  so that no leg is stretched ('alternating', the trunk above the pelvis kept still by a counter-rotation of
  the lower spine); each leg is solved by two-bone inverse kinematics with the ankle and the foot held at
  their frozen positions and orientation (the knee kept in the plane of its frozen position), so that the
  knees (the origin of the shank segment, DR-10.1 (b)) move vertically at the trembling frequency. The
  amplitude tremble_m (default 1 cm) is a module value: no amplitude of leg trembling was sourced. Where it would
  take a moved row (the legs; the pelvis's spine row in 'alternating') past the anatomical joint-limit table, or
  past the frozen knot's own components where that knot lies outside it, the episode trembles at the largest share
  of tremble_m that keeps every moved row inside (bisection, TREMBLE_SCALE_STEPS), recorded as tremble_m_used
  (open-body port, wave 5: the plantigrade double support of base.walk holds the trailing ankle near jarm2022's 20
  deg of dorsiflexion, and a hold sinking 1 cm took it past).
- 'shuffling' (small forward steps) is not in version 1: it needs a stride-scaling operator with contacts
  re-integrated, which the hypokinetic gait family will bring.
The exam set uses 'alternating' since exam_v7 (the review of the English page, M3b: in phase, both knees flexing
together at 5.5 Hz with the pelvis bouncing, read as a vibrating body; the trembling of the legs in freezing
alternates between them); 'in_phase' stays the default of the operator, bit for bit (the SP3 probe stimuli of
stimuli.py build both).

Arms during the hold (arms, wave SH1d): 'frozen' (the default, the operator before, bit for bit) keeps the frozen
base knot's arms for the whole hold, mid-swing (the review: one wrist 20 cm forward, the other 10 cm back, for up
to 15 s). 'settle' turns the collars, shoulders, elbows and wrists (ARM_JOINTS) from the frozen knot's rotations
toward the walk's arm swing centre (arm_centre: the chordal mean of each arm joint's local rotation over the base
body, the arm hanging as it swings about) by a raised cosine over ARM_SETTLE_S from the hold's first knot (about
0.5 s, the review's figure), and back over the ramp out, the weight falling from its value at the hold's end to 0
by a raised cosine while the walk resumes, so that every knot after the episode is the base's again. A hold
shorter than ARM_SETTLE_S settles part of the way. The truth records the weight reached per episode.

Standing background (a hook for physio.py, wave SH1d): the hold is a stance with both feet down; the standing layer
of physio (breathing, sway, a slow trunk drift) is to be applied over each episode's span from standing_spans(truth)
(the hold, with the ramp in and the ramp out over which it crossfades), the akinetic holds included. This module
adds no background of its own.

Known limit (the review of lane exam, wave SH1d): the hold freezes the base walk's stance at the onset knot, a full
stride with the feet far apart fore and aft, and the knees tremble in that long stance. A freeze usually comes
after the steps shorten, the feet close together; the steps before an onset are not shortened here. A later wave
may add the step length as an onset criterion of snap_onset_mesh (the shortest step in the window) or shorten the
steps before the onset.

Grade map (grades.FOG_GRADES, draft section 4): grade 1 one episode at one trigger (start, turn or doorway,
grades.GRADE_TRIGGERS), none in straight walking; grade 2 two trigger episodes; grade 3 one in straight walking
plus the trigger episodes of grade 2; grade 4 two in straight walking plus those (a module reading: the draft
keeps the trigger episodes explicitly for grade 3 only). Triggers are drawn with the weights of Schaafsma et al.
2003 among the start, turn and doorway events of the take (gait_contexts finds starts and turns; doorway
passages are scene geometry and come from the caller); reaching the destination, a context in that study, is
not a trigger of the grade map and is never drawn (a caller may still place an Episode with that context);
durations from the truncated lognormal of grades.fog_durations. Manifestation is a separate factor, crossed
with grade, never confounded with it.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..body.hermite import hermite
from ..body.rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d, rot6d_to_matrix
from ..body.joints import fk
from . import grades as G
from .base import gait_contexts
from .body import F64, Body, contact_labels, validity
from .common import as_rng, truth_record
from .ik import rmin

RAMP_IN_S = 0.5            # [module] progression ramp into the hold
RAMP_OUT_S = 0.5           # [module] ramp out of the hold
TREMBLE_RAMP_S = 0.25      # [module] onset and offset of the trembling inside the hold
TREMBLE_M = 0.01           # [module] trembling amplitude (pelvis or hip, peak to peak), unsourced
TREMBLE_SCALE_STEPS = 20   # bisection steps of the trembling's share where tremble_m would leave the table
SEARCH_S = 1.0             # [module] window searched for a double-support knot around a requested onset
LEGS = ({"hip": 1, "knee": 4, "ankle": 7}, {"hip": 2, "knee": 5, "ankle": 8})
SUPPORTS = ("double", "quasi", "mesh")
ARMS = ("frozen", "settle")
ARM_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21)   # the collars, shoulders, elbows and wrists (module docstring)
ARM_SETTLE_S = 0.5         # [module] the arms settle over this from the hold's first knot (the review: about 0.5 s)
MESH_SUPPORT_TOL_M = 0.005 # [module] two soles within this of each other stand together (contact.TOL_M)
MESH_HEEL_TOL_M = 0.02     # [module] a heel down: its lowest skinned point within this of the lower sole (the
#                            test test_freezing_holds_of_support_mesh_stand_with_both_heels_down holds it)


@dataclass
class Episode:
    """One episode: onset_base_s, the base time the hold freezes (a knot of double support); duration_s (a whole
    number of knots); manifestation ('trembling' or 'akinesia'); context (start, turn, doorway, destination,
    straight, or given); trembling frequency, amplitude (m) and pattern ('in_phase' or 'alternating'); support
    (SUPPORTS: 'double' needs a labelled double-support knot, 'quasi' and 'mesh' do not; module docstring)."""
    onset_base_s: float
    duration_s: float
    manifestation: str = "trembling"
    context: str = "given"
    tremble_hz: float = G.TREMBLE_DEFAULT_HZ
    tremble_m: float = TREMBLE_M
    pattern: str = "in_phase"
    requested_s: float = None
    support: str = "double"
    window_s: tuple = None


def double_support(body):
    """(T,) bool: both base contact labels set."""
    return (body.contact[:, 0] & body.contact[:, 1]).numpy()


def higher_foot_height(body):
    """(T,) m: for each knot, the larger over the two feet of the lower of the foot's ankle and foot joint
    heights (y; the floor at 0): small when both feet are down."""
    J = body.joints()
    low = torch.minimum(J[:, [7, 8], 1], J[:, [10, 11], 1])
    return low.max(1).values.numpy()


def foot_gap_mesh(body, knots=None):
    """(T,) m, or at the given knots: the height of the higher foot's lowest skinned sole point above the lower
    foot's (contact.mesh_sole_low)."""
    from .contact import mesh_sole_low
    low = mesh_sole_low(body, knots=knots)
    return np.abs(low[:, 0] - low[:, 1])


def heel_lift_mesh(body, knots):
    """(heel lift (N, 2) m, sole gap (N,) m) at the knots: for the left and right foot, the lowest skinned point of
    the heel set (contact.heel_toe) above the lower of the two feet's lowest skinned sole points; and the gap
    between the two soles (as foot_gap_mesh)."""
    from .contact import heel_toe, skin_points
    k = np.asarray(knots, dtype=np.int64).reshape(-1)
    R, tr = body.local_R().numpy()[k], body.transl.numpy()[k]
    ht = heel_toe(int(body.gender))
    heel, low = np.empty((len(k), 2)), np.empty((len(k), 2))
    for i, side in enumerate(("left", "right")):
        x = skin_points(R, tr, body.betas.numpy(), body.gender, side)
        heel[:, i] = x[:, ht[side][0], 1].min(1)
        low[:, i] = x[..., 1].min(-1)
    return heel - low.min(1, keepdims=True), np.abs(low[:, 0] - low[:, 1])


def snap_onset_mesh(body, t_s, search_s=SEARCH_S, tol_m=MESH_SUPPORT_TOL_M, margins_s=(0.0, 0.0),
                    heel_tol_m=MESH_HEEL_TOL_M, window_s=None):
    """(time, 'mesh', gap m): within the window (window_s, base times (lo, hi); by default search_s on each side of
    t_s) and margins_s from the body's start and end, the knot nearest t_s whose two soles lie within tol_m of each
    other in height on the skinned mesh and whose two heels lie within heel_tol_m of the lower sole
    (heel_lift_mesh); when none, the knot of the window that misses the two tolerances least (the larger of the gap
    over tol_m and the higher heel over heel_tol_m, smallest; module docstring). The onset never leaves the
    window. Raises ValueError when the window holds no knot inside the margins."""
    k = int(round(t_s * body.rate))
    first = int(math.ceil(margins_s[0] * body.rate - 1e-9))
    last = body.T - 1 - int(math.ceil(margins_s[1] * body.rate - 1e-9))
    if window_s is None:
        w = int(round(search_s * body.rate))
        lo, hi = max(first, k - w), min(last, k + w)
    else:
        lo = max(first, int(math.ceil(window_s[0] * body.rate - 1e-9)))
        hi = min(last, int(math.floor(window_s[1] * body.rate + 1e-9)))
    if hi < lo:
        where = f"within {search_s} s of {t_s:.2f} s" if window_s is None else \
            f"in {window_s[0]:.2f} to {window_s[1]:.2f} s"
        raise ValueError(f"no knot {where} inside the margins")
    idx = np.arange(lo, hi + 1)
    heel, gap = heel_lift_mesh(body, idx)
    miss = np.maximum(gap - tol_m, heel.max(1) - heel_tol_m)
    ok = np.flatnonzero(miss <= 0)
    i = ok[np.argmin(np.abs(idx[ok] - k))] if len(ok) else int(np.argmin(miss))
    return float(idx[i] / body.rate), "mesh", float(gap[i])


def snap_onset_quasi(body, t_s, search_s=SEARCH_S):
    """(time, 'double' or 'quasi', higher-foot height m): the double-support knot nearest t_s within search_s,
    or, when none, the knot of that window whose higher foot lies lowest (module docstring)."""
    try:
        t = snap_onset(body, t_s, search_s)
        return t, "double", float(higher_foot_height(body)[int(round(t * body.rate))])
    except ValueError:
        pass
    h = higher_foot_height(body)
    k = int(round(t_s * body.rate))
    w = int(round(search_s * body.rate))
    lo, hi = max(0, k - w), min(body.T - 1, k + w)
    j = lo + int(np.argmin(h[lo:hi + 1]))
    return float(j / body.rate), "quasi", float(h[j])


def snap_onset(body, t_s, search_s=SEARCH_S, ds=None):
    """The double-support knot time nearest t_s within search_s (ValueError when none)."""
    ds = double_support(body) if ds is None else ds
    k = int(round(t_s * body.rate))
    w = int(round(search_s * body.rate))
    lo, hi = max(0, k - w), min(body.T - 1, k + w)
    idx = np.flatnonzero(ds[lo:hi + 1]) + lo
    if not len(idx):
        raise ValueError(f"no double-support knot within {search_s} s of {t_s:.2f} s")
    return float(idx[np.argmin(np.abs(idx - k))] / body.rate)


def _tremble_pose(body, kf, delta, roll):
    """Local rotations (N, 22, 3, 3) and translations (N, 3) of the frozen knot kf with the pelvis moved up by
    delta (N,) m and rolled by roll (N,) rad about the horizontal heading axis, both legs solved so that the
    ankles and feet keep their frozen positions and orientations (module docstring)."""
    R = body.local_R()[kf]
    tr = body.transl[kf]
    rest = body.rest()
    J, Gl = fk(R, tr, rest)
    N = len(delta)
    fwd = Gl[0] @ torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    fwd = torch.stack([fwd[0], torch.zeros((), dtype=F64), fwd[2]])
    fwd = fwd / fwd.norm()
    Rr = axis_angle_to_matrix(torch.as_tensor(roll, dtype=F64)[:, None] * fwd)              # (N, 3, 3)
    R0 = Rr @ Gl[0]
    Rl = R.expand(N, 22, 3, 3).clone()
    Rl[:, 0] = R0
    Rl[:, 3] = R0.transpose(-1, -2) @ (Gl[0] @ R[3])                                     # spine1 global kept
    trn = tr.expand(N, 3).clone()
    trn[:, 1] = trn[:, 1] + torch.as_tensor(delta, dtype=F64)
    P = rest[0] + trn
    for leg in LEGS:
        h, k, a = leg["hip"], leg["knee"], leg["ankle"]
        L1 = float((rest[k] - rest[h]).norm())
        L2 = float((rest[a] - rest[k]).norm())
        H = P + (R0 @ (rest[h] - rest[0]))
        A = J[a].expand(N, 3)
        d = A - H
        dist = d.norm(dim=-1, keepdim=True)
        if bool((dist >= L1 + L2 - 1e-9).any()):
            raise ValueError("trembling stretches a leg beyond its length: lower tremble_m")
        u = d / dist
        along = (L1 ** 2 - L2 ** 2 + dist ** 2) / (2 * dist)
        hgt = torch.sqrt((L1 ** 2 - along ** 2).clamp_min(0.0))
        pole = J[k] - H
        w = pole - (pole * u).sum(-1, keepdim=True) * u
        w = w / w.norm(dim=-1, keepdim=True)
        K = H + along * u + hgt * w
        th_new = (K - H) / L1
        th_old = ((J[k] - J[h]) / L1).expand(N, 3)
        G_th = rmin(th_old, th_new) @ Gl[h]
        sh_new = (A - K) / L2
        sh_old = ((J[a] - J[k]) / L2).expand(N, 3)
        G_sh = rmin(sh_old, sh_new) @ Gl[k]
        Rl[:, h] = R0.transpose(-1, -2) @ G_th
        Rl[:, k] = G_th.transpose(-1, -2) @ G_sh
        Rl[:, a] = G_sh.transpose(-1, -2) @ Gl[a]
    return Rl, trn


def _catalogue_shape(body):
    """True for the catalogue's one shape (gender code 2, betas zero), the only one contact.py's sole cloud reads."""
    return int(body.gender) == 2 and not bool(torch.as_tensor(body.betas).abs().max() > 0)


def _num(x):
    """float(x), None for NaN (a reading not taken)."""
    x = float(x)
    return None if math.isnan(x) else x


def _tremble_inside(body, kf, Rl, moved):
    """True when the trembled rows `moved` (1 to 21) of Rl (N, 22, 3, 3) lie inside the joint-limit table, or inside
    the frozen knot kf's own components where that knot lies outside it (body.LIMIT_TOL_DEG)."""
    from .body import LIMIT_TOL_DEG, joint_limits
    lim = joint_limits()
    rows = [j for j in moved if j > 0]
    c = torch.rad2deg(matrix_to_axis_angle(Rl[:, rows])).numpy()
    c0 = torch.rad2deg(matrix_to_axis_angle(body.local_R()[kf, rows])).numpy()
    lo = np.minimum(lim.lo[rows], c0) - LIMIT_TOL_DEG
    hi = np.maximum(lim.hi[rows], c0) + LIMIT_TOL_DEG
    return bool(((c >= lo[None]) & (c <= hi[None])).all())


def arm_centre(body, joints=ARM_JOINTS):
    """(len(joints), 3, 3): the chordal mean of each joint's local rotation over the body's knots (the sum of the
    rotation matrices projected on the rotations by SVD), the arm swing centre of a walk (module docstring)."""
    R = body.local_R()[:, list(joints)].sum(0)
    U, _, Vt = torch.linalg.svd(R)
    d = torch.det(U @ Vt)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ D @ Vt


def _toward(R, C, w):
    """(N, J, 3, 3): rotations R (N, J, 3, 3) turned toward C (J, 3, 3) by the share w (N,) of the geodesic."""
    rel = matrix_to_axis_angle(R.transpose(-1, -2) @ C[None])
    return R @ axis_angle_to_matrix(rel * torch.as_tensor(w, dtype=F64)[:, None, None])


def standing_spans(truth):
    """[{'ramp_in_s', 'hold_s', 'ramp_out_s', 'manifestation', 'context'}] of a freezing truth: the spans where the
    body stops and stands, for physio's standing layer (module docstring, Standing background)."""
    return [{k: e[k] for k in ("ramp_in_s", "hold_s", "ramp_out_s", "manifestation", "context")}
            for e in truth["params"].get("episodes", [])]


def _knot(x, rate):
    k = x * rate
    if abs(k - round(k)) > 1e-6:
        raise ValueError(f"{x} s is not a whole number of knots at {rate} Hz")
    return int(round(k))


def freezing(body, episodes, seed=None, ramp_in_s=RAMP_IN_S, ramp_out_s=RAMP_OUT_S, tremble_ramp_s=TREMBLE_RAMP_S,
             check=True, grade=None, extend=True, arms="frozen", arm_settle_s=ARM_SETTLE_S):
    """(Body, truth) of a walking body with the given episodes (module docstring); an empty list returns a
    copy of the input. extend: the output lasts the base plus the episodes' durations, every base knot kept
    (the patient takes longer to walk the same path); False keeps the base length and drops the tail. arms:
    'frozen' or 'settle' (module docstring, Arms during the hold). The kinematic checks compare the output with
    the same time warp without trembling (every episode akinetic, the arms as asked), so that they read what the
    trembling adds; with arms 'settle' the arm rows are checked against the table apart (validity 'arm_settle').
    Raises ValueError when an onset is not a double-support knot, an episode leaves no hold after its ramps,
    episodes overlap, or (extend False) the last one does not end inside the output."""
    if arms not in ARMS:
        raise ValueError(f"arms {arms!r}, expected one of {ARMS}")
    eps = sorted(episodes, key=lambda e: e.onset_base_s)
    seed_val = seed if isinstance(seed, (int, np.integer)) else None
    if not eps:
        return body.copy(), truth_record("freezing", seed_val, grade if grade is not None else 0, "MDS-UPDRS 3.11",
                                         {"episodes": []}, ("mds_updrs",), None)
    rate, T0 = body.rate, body.T
    T = T0 + sum(_knot(e.duration_s, rate) for e in eps) if extend else T0
    ds = double_support(body)
    ri, ro = _knot(ramp_in_s, rate), _knot(ramp_out_s, rate)
    if ri % 2 or ro % 2:
        raise ValueError("ramps must be an even number of knots (onset and offset on knots)")
    # output-knot plan
    tau = np.empty(T)
    kind = np.zeros(T, np.int8)                 # 0 copy, 1 ramp (interpolated), 2 hold
    ep_of = np.full(T, -1, np.int64)
    rows, shift, k_prev_end = [], 0, 0
    for i, e in enumerate(eps):
        kb = _knot(e.onset_base_s, rate)
        D = _knot(e.duration_s, rate)
        if e.support not in SUPPORTS:
            raise ValueError(f"episode {i}: support {e.support!r}, expected one of {SUPPORTS}")
        if not ds[kb] and e.support == "double":
            raise ValueError(f"episode {i}: base onset {e.onset_base_s} s is not a double-support knot")
        if e.manifestation not in G.MANIFESTATIONS:
            raise ValueError(f"manifestation {e.manifestation!r}, expected one of {G.MANIFESTATIONS}")
        if D < (ri + ro) // 2 + 1:
            raise ValueError(f"episode {i}: {e.duration_s} s leaves no hold after its ramps")
        on = kb + shift                          # output knot of the onset
        a, b = on - ri // 2, on + D - ro // 2   # ramp-in start, ramp-out start
        end = b + ro                             # first knot after the ramp out
        if a < k_prev_end or a < 0:
            raise ValueError(f"episode {i} overlaps the previous one or starts before the body")
        if end > T - 1:
            raise ValueError(f"episode {i} ends at {end / rate:.2f} s, past the output's last knot")
        rows.append({"index": i, "k_on": on, "k_a": a, "k_hold0": on + ri // 2, "k_hold1": b, "k_b": b, "k_end": end,
                     "kb": kb, "D": D, "shift": shift})
        shift += D
        k_prev_end = end
    # tau per output knot: copied knots read the base at k minus the durations of the episodes before them
    k = np.arange(T)
    shift_at = np.zeros(T, np.int64)
    for r in rows:
        shift_at[r["k_end"]:] += r["D"]
    tau[:] = (k - shift_at) / rate
    t = k / rate
    for r in rows:
        ta, tb = r["k_a"] / rate, r["k_b"] / rate
        sel = (k > r["k_a"]) & (k < r["k_a"] + ri)
        tau[sel] = (r["k_a"] - r["shift"]) / rate + (t[sel] - ta) / 2 + (ramp_in_s / (2 * np.pi)) * np.sin(
            np.pi * (t[sel] - ta) / ramp_in_s)
        kind[sel] = 1
        sel = (k >= r["k_a"] + ri) & (k < r["k_b"])
        tau[sel] = r["kb"] / rate
        kind[sel] = 2
        ep_of[sel] = r["index"]
        sel = (k > r["k_b"]) & (k < r["k_end"])
        tau[sel] = r["kb"] / rate + (t[sel] - tb) / 2 - (ramp_out_s / (2 * np.pi)) * np.sin(
            np.pi * (t[sel] - tb) / ramp_out_s)
        kind[sel] = 1
        tau[r["k_b"]] = r["kb"] / rate             # the ramp-out start reads the frozen knot exactly
    # build the output
    r6 = torch.empty(T, 22, 6, dtype=F64)
    tr = torch.empty(T, 3, dtype=F64)
    copy = kind == 0
    src = np.round(tau[copy] * rate).astype(np.int64)
    if (src < 0).any() or (src > T0 - 1).any():
        raise ValueError("time map leaves the base body")
    r6[torch.from_numpy(copy)] = body.rot6d[torch.from_numpy(src)]
    tr[torch.from_numpy(copy)] = body.transl[torch.from_numpy(src)]
    ramp = kind == 1
    if ramp.any():
        tq = torch.from_numpy(tau[ramp])
        x6 = hermite(body.rot6d, tq, rate)
        r6[torch.from_numpy(ramp)] = matrix_to_rot6d(rot6d_to_matrix(x6))
        tr[torch.from_numpy(ramp)] = hermite(body.transl, tq, rate)
    episodes_truth = []
    h_joint = higher_foot_height(body)
    if _catalogue_shape(body):
        heel_mesh, gap_mesh = heel_lift_mesh(body, [r["kb"] for r in rows])
    else:                                       # the sole cloud reads the catalogue's one shape only (contact.py)
        heel_mesh, gap_mesh = np.full((len(rows), 2), np.nan), np.full(len(rows), np.nan)
    for r in rows:
        e = eps[r["index"]]
        hk = np.arange(r["k_a"] + ri, r["k_b"])
        n = len(hk)
        tm_used = None
        if e.manifestation == "akinesia":
            r6[torch.from_numpy(hk)] = body.rot6d[r["kb"]].expand(n, 22, 6)
            tr[torch.from_numpy(hk)] = body.transl[r["kb"]].expand(n, 3)
        else:
            if not G.FREEZE_BAND_HZ[0] <= e.tremble_hz <= G.FREEZE_BAND_HZ[1]:
                raise ValueError(f"trembling at {e.tremble_hz} Hz outside the freeze band {G.FREEZE_BAND_HZ}")
            th = (hk - hk[0]) / rate
            H = (n - 1) / rate                       # the envelope is 0 on the first and last hold knots
            trm = min(tremble_ramp_s, H / 2)
            env = np.ones(n)
            if trm > 0:
                env = np.minimum(np.clip(th / trm, 0, 1), np.clip((H - th) / trm, 0, 1))
                env = 0.5 - 0.5 * np.cos(np.pi * env)
            cosw = np.cos(2 * np.pi * e.tremble_hz * th)
            if e.pattern not in ("in_phase", "alternating"):
                raise ValueError(f"pattern {e.pattern!r}")
            moved = [1, 2, 4, 5, 7, 8] if e.pattern == "in_phase" else [0, 1, 2, 3, 4, 5, 7, 8]

            def trembled(tm):
                if e.pattern == "in_phase":          # the pelvis drops and rises by tm peak to peak
                    delta, roll = -0.5 * tm * env * (1.0 - cosw), np.zeros(n)
                else:                                # each hip by tm peak to peak, in opposite phase
                    rest = body.rest()
                    half = float((rest[1] - rest[2]).norm()) / 2
                    rho = math.asin(min(1.0, 0.5 * tm / half))
                    roll = rho * env * np.sin(2 * np.pi * e.tremble_hz * th)
                    delta = -half * math.sin(rho) * env
                return _tremble_pose(body, r["kb"], torch.from_numpy(delta), torch.from_numpy(roll))
            tm_used = e.tremble_m
            Rl, trn = trembled(tm_used)
            if not _tremble_inside(body, r["kb"], Rl, moved):
                # the trembling scaled into the table (module docstring): the largest share of tremble_m, by
                # bisection, at which no moved row leaves the table or the frozen knot's own excursion
                lo_s, hi_s = 0.0, 1.0
                for _ in range(TREMBLE_SCALE_STEPS):
                    m = 0.5 * (lo_s + hi_s)
                    if _tremble_inside(body, r["kb"], trembled(m * e.tremble_m)[0], moved):
                        lo_s = m
                    else:
                        hi_s = m
                tm_used = lo_s * e.tremble_m
                Rl, trn = trembled(tm_used)
            base6 = body.rot6d[r["kb"]].expand(n, 22, 6).clone()
            base6[:, moved] = matrix_to_rot6d(Rl[:, moved])
            r6[torch.from_numpy(hk)] = base6
            tr[torch.from_numpy(hk)] = trn
        on_s, off_s = r["k_on"] / rate, (r["k_on"] + r["D"]) / rate
        episodes_truth.append({"context": e.context, "manifestation": e.manifestation, "onset_s": on_s,
                               "offset_s": off_s, "duration_s": r["D"] / rate,
                               "hold_s": [hk[0] / rate, (hk[-1] + 1) / rate],
                               "ramp_in_s": [r["k_a"] / rate, (r["k_a"] + ri) / rate],
                               "ramp_out_s": [r["k_b"] / rate, r["k_end"] / rate],
                               "onset_base_s": r["kb"] / rate, "requested_s": e.requested_s,
                               "drift_s": None if e.requested_s is None else r["kb"] / rate - e.requested_s,
                               "context_window_s": None if e.window_s is None else [float(x) for x in e.window_s],
                               "tremble_hz": e.tremble_hz if e.manifestation == "trembling" else None,
                               "tremble_m": e.tremble_m if e.manifestation == "trembling" else None,
                               "tremble_m_used": tm_used if e.manifestation == "trembling" else None,
                               "pattern": e.pattern if e.manifestation == "trembling" else None,
                               "support": e.support, "labelled_double_support": bool(ds[r["kb"]]),
                               "higher_foot_joint_m": float(h_joint[r["kb"]]),
                               "foot_gap_mesh_m": _num(gap_mesh[len(episodes_truth)]),
                               "heel_lift_mesh_m": [_num(x) for x in heel_mesh[len(episodes_truth)]]})
    arm_info = None
    if arms == "settle":                        # the arms toward the swing centre (module docstring)
        C = arm_centre(body)
        wa = np.zeros(T)
        n_set = arm_settle_s * rate
        reached = []
        for r in rows:
            h0, kb_, ke = r["k_a"] + ri, r["k_b"], r["k_end"]
            u = np.clip((np.arange(h0, kb_ + 1) - h0) / max(n_set, 1e-9), 0.0, 1.0)
            rise = 0.5 - 0.5 * np.cos(np.pi * u)
            wa[h0:kb_ + 1] = rise
            v = (np.arange(kb_, ke) - kb_) / ro
            wa[kb_:ke] = rise[-1] * (0.5 + 0.5 * np.cos(np.pi * v))
            reached.append(float(rise[-1]))
        sel = np.flatnonzero(wa > 0)
        if len(sel):
            R = rot6d_to_matrix(r6[torch.from_numpy(sel)][:, list(ARM_JOINTS)])
            r6[torch.from_numpy(sel)[:, None], torch.as_tensor(ARM_JOINTS)[None]] = matrix_to_rot6d(
                _toward(R, C, wa[sel]))
        for et, w in zip(episodes_truth, reached):
            et["arms_settled_share"] = w
        arm_info = {"mode": "settle", "joints": list(ARM_JOINTS), "settle_s": float(arm_settle_s),
                    "centre": "chordal mean of each arm joint's local rotation over the base body (arm_centre)",
                    "rule": "raised cosine from the hold's first knot, back over the ramp out"}
    out = Body(r6, tr, torch.zeros(T, 2, dtype=torch.bool), body.betas.clone(), body.gender, body.rate)
    # contacts: copied knots keep the base labels, the others take the body-store rule on the output
    lab = contact_labels(out)
    ct = lab.clone()
    ct[torch.from_numpy(copy)] = body.contact[torch.from_numpy(src)]
    out = replace(out, contact=ct)
    params = {"episodes": episodes_truth, "ramp_in_s": ramp_in_s, "ramp_out_s": ramp_out_s,
              "tremble_ramp_s": tremble_ramp_s, "n_trigger": sum(e["context"] != "straight" for e in episodes_truth),
              "n_straight": sum(e["context"] == "straight" for e in episodes_truth),
              "arms": arm_info or {"mode": "frozen"}}
    val = None
    if check:
        still = [Episode(e.onset_base_s, e.duration_s, "akinesia", e.context, support=e.support) for e in eps]
        ref, _ = freezing(body, still, ramp_in_s=ramp_in_s, ramp_out_s=ramp_out_s, check=False, extend=extend,
                          arms=arms, arm_settle_s=arm_settle_s)
        val = validity(ref, out)
        if arms == "settle":                   # the settled arms against the table, apart from the trembling
            from .body import limit_violations
            frz, _ = freezing(body, still, ramp_in_s=ramp_in_s, ramp_out_s=ramp_out_s, check=False, extend=extend)
            lv = limit_violations(ref.rot6d, frz.rot6d)
            val["arm_settle"] = {"limits_new": int(lv["new"]), "worst_deg": float(lv["worst_deg"])}
        hold = kind == 2
        Jo = out.joints()
        from .body import foot_points
        fp = foot_points(Jo).numpy()
        moves = []
        for r in rows:
            hk = slice(r["k_a"] + ri, r["k_b"])
            moves.append(float(np.abs(fp[hk] - fp[hk][0]).max()))
        val["hold_foot_motion_m"] = max(moves) if moves else 0.0
        val["hold_knots"] = int(hold.sum())
    return out, truth_record("freezing", seed_val, grade, "MDS-UPDRS 3.11", params,
                             ("mds_updrs", "schaafsma2003", "wang2020", "moore2008"), val,
                             arrays={"tau_s": tau, "kind": kind, "episode": ep_of})


def plan_episodes(body, grade, seed=0, contexts=None, doorway_s=(), durations=None, manifestation="trembling",
                  tremble_hz=G.TREMBLE_DEFAULT_HZ, tremble_m=TREMBLE_M, pattern="in_phase", search_s=SEARCH_S,
                  tries=50, ramp_in_s=RAMP_IN_S, ramp_out_s=RAMP_OUT_S, support="double"):
    """[Episode] of a 3.11 grade (module docstring) on a walking body: trigger events (grades.GRADE_TRIGGERS:
    start, turn, doorway passages given in doorway_s) drawn with the Schaafsma et al. weights, straight-walking
    episodes in the middle of distinct straight segments (or at a third and two thirds of the longest one),
    durations from grades.fog_durations (or given, one per episode, triggers first), onsets snapped to double
    support (support 'quasi': the fallback of snap_onset_quasi; 'mesh': snap_onset_mesh, the skinned soles, inside
    the episode's context window: module docstring; each such episode carries its window, window_s). Raises
    ValueError when the take lacks the events, (support 'mesh') a straight segment is too short for the ramps, or the
    episodes cannot fit after `tries` draws."""
    if support not in SUPPORTS:
        raise ValueError(f"support {support!r}, expected one of {SUPPORTS}")
    g = int(grade)
    want = G.FOG_GRADES[g]
    if g == 0:
        return []
    rng = as_rng(seed)
    c = gait_contexts(body) if contexts is None else contexts
    events = [("start", a, b) for a, b in c.get("start", [])] + [("turn", a, b) for a, b in c.get("turn", [])] + \
        [("doorway", float(x), float(x)) for x in doorway_s]
    events = [e for e in events if e[0] in G.GRADE_TRIGGERS]
    if len(events) < want["trigger"]:
        raise ValueError(f"grade {g} needs {want['trigger']} trigger events, the take has {len(events)}")
    straight = sorted(c.get("straight", []), key=lambda s: s[0])
    spots = []                                   # (requested time, (segment start, segment end))
    if want["straight"]:
        if len(straight) >= want["straight"]:
            pick = sorted(rng.choice(len(straight), size=want["straight"], replace=False))
            spots = [((straight[i][0] + straight[i][1]) / 2, tuple(straight[i])) for i in pick]
        elif straight:
            a, b = max(straight, key=lambda s: s[1] - s[0])
            m = want["straight"] + 1
            spots = [(a + (b - a) * (j + 1) / m, (a + (b - a) * (j + 0.5) / m, a + (b - a) * (j + 1.5) / m))
                     for j in range(want["straight"])]
        else:
            raise ValueError(f"grade {g} needs straight walking, the take has none")
    w = np.array([G.TRIGGER_SHARES[e[0]] for e in events], dtype=np.float64)
    idx = rng.choice(len(events), size=want["trigger"], replace=False, p=w / w.sum())
    # context windows (module docstring): a trigger's interval widened by search_s, a straight segment shortened by
    # the base time of the ramps
    chosen = [(events[i][0], events[i][1], (events[i][1] - search_s, events[i][2] + search_s)) for i in idx] + \
        [("straight", t, (a + ramp_in_s / 2, b - ramp_out_s / 2)) for t, (a, b) in spots]
    for ctx, t, (lo, hi) in chosen if support == "mesh" else ():
        if hi < lo:
            raise ValueError(f"grade {g}: the {ctx} segment at {t:.2f} s is too short for the ramps")
    n = len(chosen)
    for _ in range(tries):
        durs = list(durations) if durations is not None else G.fog_durations(rng, n)
        if len(durs) != n:
            raise ValueError(f"{n} episodes, {len(durs)} durations")
        eps = []
        for (ctx, t, win), d in zip(chosen, durs):
            if support == "quasi":
                on, kind, _ = snap_onset_quasi(body, t, search_s)
            elif support == "mesh":
                on, kind, _ = snap_onset_mesh(body, t, search_s, window_s=win,
                                              margins_s=(ramp_in_s / 2, ramp_out_s / 2 + 1.0 / body.rate))
            else:
                on, kind = snap_onset(body, t, search_s), "double"
            eps.append(Episode(on, d, manifestation, ctx, tremble_hz, tremble_m, pattern, requested_s=float(t),
                               support=kind, window_s=(float(win[0]), float(win[1])) if support == "mesh" else None))
        try:
            freezing(body, eps, check=False, ramp_in_s=ramp_in_s, ramp_out_s=ramp_out_s)
            return sorted(eps, key=lambda e: e.onset_base_s)
        except ValueError:
            if durations is not None:
                raise
    raise ValueError(f"grade {g}: no fitting set of episode durations in {tries} draws")


def freezing_grade(body, grade, seed=0, arms="frozen", **kw):
    """(Body, truth) of a 3.11 grade: plan_episodes then freezing (module docstring); arms goes to freezing."""
    eps = plan_episodes(body, grade, seed, **kw)
    out, truth = freezing(body, eps, seed=seed if isinstance(seed, (int, np.integer)) else None, grade=int(grade),
                          arms=arms)
    truth["params"]["grade_map"] = dict(G.FOG_GRADES[int(grade)])
    return out, truth
