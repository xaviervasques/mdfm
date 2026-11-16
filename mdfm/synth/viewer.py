"""The local examination viewer of the synthetic patient, version (a) of SP4 (docs/proposals/SYNTH_PATIENT_DRAFT.md,
section 9; docs/proposals/EXPLORATION_PLAN_SYNTHETIC_PATIENT_AND_JEPA_2026-10-05.md, SP4): a body trajectory drawn
as the grey mesh of the selected body model (body.body_model: the open body by default) by the numpy rasterizer
of raster.py, from fixed clinical views, at 30 frames per second,
into H.264 MP4 files and PNG stills, every frame marked SYNTHETIC and captioned with the operator and the grade.

Mesh: frame k sits at t_k = k / fps on the body clock; the rotations (rot6d) and the translation are read at t_k
by the body's Hermite interpolant (mdfm.body.hermite, the interpolant of mdfm.body.kinematics), the rotations
re-orthonormalized (rot6d_to_matrix), and the vertices come from the backend's one skinning call (body_model().skin:
the open body's 13,718 vertices by linear blend skinning with the forearm's twist split, openbody.OpenBody.skin; the
SMPL+H reference's 6,890 with its pose blend shapes, smplh_reference.skin_smplh), hands at rest; a hands.HandBody
(task SP5) carries its 30 hand joints through the same interpolant and is skinned with them. At a frame on a knot the
mesh is the knot's pose exactly (tests/test_mesh_video.py). The model arrays stay in memory; frames and clips are
written only under VIEWER_ROOT, never in this tree. mesh_model is the SMPL+H mesh itself, reached on the reference
only: a caller of a later wave refuses on the open body until its wave ports it.

Views (VIEWS): fixed cameras placed from the body, never from the frames: an azimuth about the vertical from the
body's heading (the window frame puts it on +z; azimuth 0 is the camera in front of the body, 90 on the body's
left, the side +x), an elevation, a vertical field of view, and a region of the mesh to frame (every vertex, both
forearms and hands, the left hand, the left shank and foot, the left foot, both feet, the left toes, the head and
neck), fitted so that the region over every frame lies inside the image with a margin. The close-ups asked by the
clinical review of the symptom catalogue (2026-11-02, SH11 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md):
the left hand from its side, a three-quarter front-left view, the head and neck from the left, the left foot from
the front and the midline (azimuth negative: toward the body's right, as hand_left_medial), both feet from behind,
the left toes. Two views follow a walk: 'track_left' translates along the walk's principal axis with the pelvis
(a centred moving mean of 1 s), looking across the track from the side the body has on its left on the first
passage; 'examiner' stands behind the start of the walk, looking down the track (the patient walks
away, turns and comes back, as MDS-UPDRS 3.10 asks).

Scene: a checkerboard floor at the body's own mesh floor (scene_floor: the median over every frame of the lowest
vertex of the mesh, review of 2026-10-05; version 1 placed it from the neutral rest mesh, which left the AMASS
references floating or sunk), with the residual of each frame's lowest vertex against it recorded per clip (the
floor report: its range, percentiles and the share of frames within FLOOR_TOL_M), a drop shadow of the body (light
from above), a background gradient, and for seated segments a chair with arms (boxes placed from the neutral seated
postures, standing on the clip's floor; scene geometry only, the body has no contact channel, draft section 2; since
SH5 the backrest stands behind the trunk and the armrests outside the hips, chair_mesh, whose default fit 'sh5' (the
SH5 chair of 2026-11-02) ends each armrest 10 cm in front of the sit's elbow; two other fits are built and off:
'forearm' (2026-11-03, off by session decision: a top under the whole width of the forearm) and 'wrist' (wave SH1c:
each armrest run on to just past the sit's wrist, its top CHAIR_REST_GAP_M under the lowest arm vertex above it, the
forearm over the arm of the chair from the elbow to the wrist, the hand past its front end)), or without them
(chair_mesh armrests=False, scene kind 'chair_armless' of exam.scene_mesh: the AIMS position of the involuntary
family, the hands resting on the thighs, never uses the armrests; SH5).

Labels (mdfm.video.labels rules): SYNTHETIC in every file name, the MP4 title and comment, the PNG text chunks
and every JSON; a mark burned into every frame (top left) and a caption band (bottom: operator, grade and
parameters; segment and view; the time). check_out refuses any folder outside VIEWER_ROOT, this tree, MoveAtlas
and the gates-only store.

Per-frame texts (SH12): a clip's truth may carry params['frame_text'] ([{from_s, to_s, text}], a state such as the
Romberg eye state: the body has no eyes) and params['stimulus_marks'] ([{t_s, dur_s, text}], the jerks' stimuli);
frame_texts gathers them and the Annotator burns the texts active at each frame's time into the top band, between
the SYNTHETIC note and the view title (below the mark when the band is too narrow), a state in a blue box and a
stimulus in an orange one. A clip without them renders exactly as before.
"""
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch

from ..video import labels as LB
from . import raster as RS

# the clips' store, outside the tree; MDFM_SYNTH_VIEWER moves it (a public copy has no default store disk)
VIEWER_ROOT = os.environ.get("MDFM_SYNTH_VIEWER", "<store>/synth_viewer").replace("\\", "/").rstrip("/")
FPS = 30.0
SIZE = (960, 540)
NOTICE = ("SYNTHETIC: a body model rendered by the MD_FM synthetic patient viewer (SP4); not a recording of any "
          "person; internal research use only")
BODY_ALBEDO = np.array([0.56, 0.57, 0.60])
CHAIR_ALBEDO = np.array([0.55, 0.42, 0.30])
GENDERS = {0: "female", 1: "male", 2: "neutral"}
FLOOR_TOL_M = 0.005           # the floor report's tolerance (the contact tolerance of contact.py)


# ---------------------------------------------------------------- the mesh
def mesh_model(gender="neutral"):
    """(SMPLHBody float64, faces (13776, 3) int64) of a gender, the SMPL+H mesh itself: on the reference body only
    (body.model_name() 'smplh_reference'); on the open body it refuses (openbody.OpenBodyRefused), so that a module
    not yet ported never mixes the SMPL+H mesh with the open skeleton."""
    from .body import is_open
    if is_open():
        from .openbody import OpenBodyRefused
        raise OpenBodyRefused("viewer.mesh_model is the SMPL+H mesh: its caller reads SMPL+H arrays and runs on the "
                              "reference only (MDFM_BODY_MODEL=smplh_reference, MDFM_SMPLH_REFERENCE=1) until its wave "
                              "of docs/OPEN_BODY_PORT_DESIGN.md ports it")
    from .smplh_reference import mesh_model as smplh_mesh
    return smplh_mesh(gender)


def mesh():
    """The selected backend (body.body_model): rest_vertices_np, faces, dominant_joint, skin."""
    from .body import body_model
    return body_model()


def faces():
    """(F, 3) int64: the selected body model's triangles."""
    return np.asarray(mesh().faces, dtype=np.int64)


def rest_vertices():
    """(V, 3): the selected body model's rest mesh (the canonical T pose; read only)."""
    return mesh().rest_vertices_np()


def dominant_joint(gender="neutral"):
    """(V,) the program joint (0 to 51) with the largest skinning weight of each vertex (on the open body the owner
    of the dominant Anny bone, design D10). gender: kept for the callers; the catalogue's one shape only."""
    return mesh().dominant_joint()


def n_frames(span_s, fps=FPS):
    """Frames of a body whose knots span span_s seconds: every t_k = k / fps inside [0, span_s], that is
    floor(span_s x fps) + 1."""
    if span_s < 0:
        raise ValueError("negative span")
    return int(math.floor(float(span_s) * float(fps) + 1e-9)) + 1


def frame_times(span_s, fps=FPS):
    """t_k = k / fps for k < n_frames(span_s, fps)."""
    return np.arange(n_frames(span_s, fps), dtype=np.float64) / float(fps)


def pose_at(body, t):
    """(R_local (N, 22, 3, 3), transl (N, 3)) of a synth.body.Body at times t (N,) s: Hermite on rot6d and transl,
    rotations re-orthonormalized (module docstring)."""
    from ..body.hermite import hermite
    from ..body.rotations import rot6d_to_matrix
    tq = torch.as_tensor(np.asarray(t, dtype=np.float64))
    if float(tq.max()) > body.duration + 1e-9 or float(tq.min()) < -1e-9:
        raise ValueError(f"frame times outside the body's span [0, {body.duration}] s")
    with torch.no_grad():
        r6 = hermite(body.rot6d, tq, body.rate)
        tr = hermite(body.transl, tq, body.rate)
    return rot6d_to_matrix(r6), tr


def hands_at(body, t):
    """(N, 30, 3, 3) hand rotations of a hands.HandBody at times t (N,) s (Hermite on hand6d, as pose_at), None for
    a body without hands (task SP5)."""
    if getattr(body, "hand6d", None) is None:
        return None
    from ..body.hermite import hermite
    from ..body.rotations import rot6d_to_matrix
    tq = torch.as_tensor(np.asarray(t, dtype=np.float64))
    with torch.no_grad():
        return rot6d_to_matrix(hermite(body.hand6d, tq, body.rate))


def _shape_of(body):
    """The body's shape checked against the selected model (the open body: the catalogue's one shape)."""
    from .body import _check_open_shape, is_open
    if is_open():
        _check_open_shape(body.betas, body.gender)
    elif bool(body.betas.abs().max() > 0) or int(body.gender) != 2:
        raise ValueError("the reference body is skinned in the neutral shape only (betas 0, gender code 2)")


def vertices_at(body, t, batch=96):
    """(N, V, 3) float64 vertices of the body at times t (module docstring); a hands.HandBody is skinned with its
    hand rotations, any other body with the hands at rest: the backend's one skinning call."""
    _shape_of(body)
    R, tr = pose_at(body, t)
    Rh = hands_at(body, t)
    m = mesh()
    out = []
    for a in range(0, len(t), batch):
        b = slice(a, a + batch)
        out.append(m.skin(R[b].numpy(), tr[b].numpy(), None if Rh is None else Rh[b].numpy()))
    return np.concatenate(out, 0)


def knot_vertices(body, knots):
    """(n, V, 3) vertices of the body at the given knots, posed from the knot rotations directly (the reference of
    the frame-by-frame identity test); a hands.HandBody with its knot hand rotations."""
    _shape_of(body)
    k = list(knots)
    R = body.local_R()[k].numpy()
    tr = body.transl[k].numpy()
    Rh = body.hand_R()[k].numpy() if getattr(body, "hand6d", None) is not None else None
    return mesh().skin(R, tr, Rh)


def sole_offset_m():
    """Height (m) of the lowest foot joint (10, 11) above the lowest vertex of the rest mesh of the selected body
    model: the joint floor of the body lies this far above the sole."""
    from .body import model_name
    return _sole_offset(model_name())


@functools.lru_cache(maxsize=2)
def _sole_offset(name):
    from .body import rest_joints
    v = rest_vertices()
    r = rest_joints().numpy()
    return float(min(r[10, 1], r[11, 1]) - v[:, 1].min())


def mesh_floor_y(body_floor_y=0.0):
    """The height of a flat neutral sole on a joint floor at body_floor_y: body_floor_y minus sole_offset_m (the
    target of exam.ground_by_mesh; the scene floor itself is scene_floor's)."""
    return float(body_floor_y) - sole_offset_m()


def lowest_points(body, fps=FPS, batch=96):
    """(N,) m: the height of the lowest vertex of the mesh at every frame t_k = k / fps."""
    t = frame_times(body.duration, fps)
    return np.concatenate([vertices_at(body, t[a:a + batch])[..., 1].min(1) for a in range(0, len(t), batch)])


def floor_report(low, floor_y, tol=FLOOR_TOL_M):
    """JSON-ready residual of each frame's lowest vertex (low (N,) m) against the floor plane: min (penetration
    when negative), 2nd and 98th percentiles, median, max (float), share of frames within tol."""
    r = np.asarray(low, dtype=np.float64) - float(floor_y)
    return {"floor_y_m": float(floor_y), "frames": int(len(r)), "residual_min_m": float(r.min()),
            "residual_p2_m": float(np.percentile(r, 2)), "residual_median_m": float(np.median(r)),
            "residual_p98_m": float(np.percentile(r, 98)), "residual_max_m": float(r.max()),
            "within_tol_share": float((np.abs(r) <= tol).mean()), "tol_m": float(tol),
            "rule": "median over every frame of the lowest mesh vertex"}


def scene_floor(body, fps=FPS):
    """(floor y m, floor report) of a clip: the median over every frame of the lowest vertex of the body's mesh,
    and the residual of every frame against it (floor_report)."""
    low = lowest_points(body, fps)
    fl = float(np.median(low))
    return fl, floor_report(low, fl)


# ---------------------------------------------------------------- regions and views
REGIONS = {"body": None,
           "hands": (18, 19, 20, 21) + tuple(range(22, 52)),
           "foot_left": (4, 7, 10),
           "head": (9, 12, 13, 14, 15),
           "hand_left": (20,) + tuple(range(22, 37)),          # task SP5: the left hand of 3.4 and 3.5
           "foot_left_close": (7, 10),                          # SH11: the left foot alone (ankle and toe block)
           "feet": (7, 8, 10, 11),                              # SH11: both feet
           "toes_left": (10,)}                                  # SH11: the toe block of the left foot


def region_mask(region, gender="neutral"):
    """(V,) bool: the vertices whose dominant joint lies in the region (every vertex for 'body')."""
    j = REGIONS[region]
    dom = dominant_joint(gender)
    return np.ones(len(dom), bool) if j is None else np.isin(dom, j)


@dataclass(frozen=True)
class View:
    """A camera rule (module docstring). kind 'fixed' or 'track'; azimuth and elevation in degrees; vfov_deg;
    region of the mesh to frame; margin (share of the half image kept free)."""
    name: str
    azimuth: float = 0.0
    elevation: float = 8.0
    vfov_deg: float = 30.0
    region: str = "body"
    kind: str = "fixed"
    margin: float = 0.08
    title: str = ""


VIEWS = {
    "front": View("front", 0.0, 8.0, title="front"),
    "back": View("back", 180.0, 8.0, title="back"),
    "left": View("left", 90.0, 6.0, title="left side"),
    "right": View("right", -90.0, 6.0, title="right side"),
    "top": View("top", 0.0, 62.0, title="from above, in front"),
    "hands": View("hands", 0.0, 32.0, 26.0, "hands", margin=0.10, title="close-up, both forearms and hands"),
    "hands_top": View("hands_top", 0.0, 72.0, 26.0, "hands", margin=0.10,
                      title="close-up from above, both forearms and hands"),
    "foot_left": View("foot_left", 90.0, 8.0, 26.0, "foot_left", margin=0.10, title="close-up, left foot, side"),
    "head": View("head", 0.0, 6.0, 26.0, "head", margin=0.12, title="close-up, head and neck"),
    "track_left": View("track_left", 0.0, 5.0, 34.0, "body", kind="track", margin=0.06,
                       title="side, following the walk"),
    "examiner": View("examiner", 180.0, 10.0, 30.0, "body", margin=0.04,
                     title="examiner, behind the start of the walk"),
    "hand_left": View("hand_left", 0.0, 10.0, 24.0, "hand_left", margin=0.12, title="close-up, left hand, front"),
    "hand_left_medial": View("hand_left_medial", -60.0, 15.0, 24.0, "hand_left", margin=0.12,
                             title="close-up, left hand, from the front and the midline"),
    # SH11 (clinical review of the symptom catalogue, 2026-11-02): finger, head and foot signs
    "hand_left_side": View("hand_left_side", 90.0, 10.0, 24.0, "hand_left", margin=0.12,
                           title="close-up, left hand, left side"),
    "front_left": View("front_left", 45.0, 8.0, title="three-quarter, front and left"),
    "head_left": View("head_left", 90.0, 6.0, 26.0, "head", margin=0.12, title="close-up, head and neck, left side"),
    "foot_left_medial": View("foot_left_medial", -20.0, 15.0, 26.0, "foot_left_close", margin=0.12,
                             title="close-up, left foot, from the front and the midline"),
    "foot_back": View("foot_back", 180.0, 15.0, 26.0, "feet", margin=0.12, title="close-up, both feet, from behind"),
    "toes_left": View("toes_left", 45.0, 20.0, 12.0, "toes_left", margin=0.15,
                      title="close-up, left toes, front and left"),
}
NO_SCENE_VIEWS = ("foot_back",)   # [module] views for clips without a scene (standing): from behind, the seat block
#                                   fills the frame of a seated clip; scripts/render_mesh_video.py refuses them there


@dataclass
class CameraPath:
    """Per-frame cameras: K (3, 3); R (N, 3, 3); t (N, 3); info (how it was placed)."""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    info: dict = field(default_factory=dict)

    @property
    def fixed(self):
        return bool(np.all(self.R == self.R[:1]) and np.all(self.t == self.t[:1]))


def _axes(dirv, up=np.array([0.0, 1.0, 0.0])):
    zc = -dirv / np.linalg.norm(dirv)
    xc = np.cross(zc, up)
    xc = xc / np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return xc, yc, zc


RESERVE = (0.08, 0.14)        # shares of the image height kept free of the body: the mark (top), the caption (bottom)


def fit_distance(points, target, dirv, K, width, height, margin, reserve=RESERVE):
    """(distance, target) of a camera on target + distance dirv that sees every point (N, 3) inside the image
    window left by the reserved top and bottom bands, with a margin, the target moved so that the points' image
    extent is centred in that window (three refinements)."""
    tx = (width / 2.0) / K[0, 0] * (1.0 - margin)
    y_lo = (reserve[0] * height - K[1, 2]) / K[1, 1]
    y_hi = ((1.0 - reserve[1]) * height - K[1, 2]) / K[1, 1]
    mid, half = 0.5 * (y_lo + y_hi), 0.5 * (y_hi - y_lo) * (1.0 - margin)
    lo, hi = mid - half, mid + half
    xc, yc, zc = _axes(dirv)
    tgt = np.asarray(target, dtype=np.float64).copy()

    def dist(t):
        rel = points - t
        x, y, z = rel @ xc, rel @ yc, rel @ zc
        dy = np.where(y > 0, y / hi, y / lo) - z
        return float(max(np.max(np.abs(x) / tx - z), np.max(dy), 0.5 - z.min())), x, y, z
    for _ in range(3):
        D, x, y, z = dist(tgt)
        xn, yn = x / (z + D), y / (z + D)
        tgt = tgt + xc * (0.5 * (xn.max() + xn.min()) * D) + yc * ((0.5 * (yn.max() + yn.min()) - mid) * D)
    D, _, _, _ = dist(tgt)
    return D, tgt


def heading_deg(body, knot=None):
    """Heading (deg about +y, 0 on +z) of the root at a knot (default the centre knot)."""
    from ..body.rotations import yaw_angle
    k = body.T // 2 if knot is None else int(knot)
    return math.degrees(float(yaw_angle(body.local_R()[k:k + 1, 0])[0]))


def walk_axis(pelvis_xz):
    """(centre (2,), unit axis (2,)) of a walk: the principal direction of the pelvis path, oriented along the
    first passage (from the start toward the farthest point)."""
    p = np.asarray(pelvis_xz, dtype=np.float64)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    a = vt[0]
    far = p[np.argmax(np.linalg.norm(p - p[0], axis=1))]
    if (far - p[0]) @ a < 0:
        a = -a
    return c, a


def camera_path(view, verts_sample, pelvis, n_frames, width=SIZE[0], height=SIZE[1], heading=0.0, frame_pelvis=None,
                fps=FPS):
    """CameraPath of a View. verts_sample: (M, V, 3) vertices of a subsample of frames (for the framing);
    pelvis: (K, 3) pelvis positions at the body knots (walk views); frame_pelvis (N, 3): the pelvis at each frame
    (track view); heading: the body heading in degrees (fixed views)."""
    K = RS.intrinsics(width, height, view.vfov_deg)
    mask = region_mask(view.region)
    pts = verts_sample[:, mask].reshape(-1, 3)
    up = np.array([0.0, 1.0, 0.0])
    el = math.radians(view.elevation)
    info = {"view": view.name, "title": view.title, "kind": view.kind, "vfov_deg": view.vfov_deg,
            "region": view.region, "elevation_deg": view.elevation, "margin": view.margin}
    if view.kind == "fixed" and view.name != "examiner":
        az = math.radians(heading + view.azimuth)
        dirv = np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
        lo, hi = pts.min(0), pts.max(0)
        D, tgt = fit_distance(pts, (lo + hi) / 2, dirv, K, width, height, view.margin)
        R, t = RS.look_at(tgt + D * dirv, tgt, up)
        info.update(azimuth_deg=view.azimuth, heading_deg=heading, distance_m=D, target=tgt.tolist(),
                    eye=(tgt + D * dirv).tolist())
        return CameraPath(K, np.repeat(R[None], n_frames, 0), np.repeat(t[None], n_frames, 0), width, height, info)
    c, a = walk_axis(pelvis[:, [0, 2]])
    a3 = np.array([a[0], 0.0, a[1]])
    if view.name == "examiner":
        dirv = -a3 * math.cos(el) + up * math.sin(el)          # behind the start, looking down the track
        lo, hi = pts.min(0), pts.max(0)
        D, tgt = fit_distance(pts, (lo + hi) / 2, dirv, K, width, height, view.margin)
        R, t = RS.look_at(tgt + D * dirv, tgt, up)
        info.update(axis=a.tolist(), distance_m=D, target=tgt.tolist(), eye=(tgt + D * dirv).tolist())
        return CameraPath(K, np.repeat(R[None], n_frames, 0), np.repeat(t[None], n_frames, 0), width, height, info)
    if view.kind != "track":
        raise ValueError(f"unknown view kind {view.kind!r}")
    side = np.cross(up, a3)                                       # the body's left on the first passage
    dirv = side * math.cos(el) + up * math.sin(el)
    # framing: the body around its own pelvis, every sampled frame brought to the track origin
    s_all = (verts_sample - np.asarray(frame_pelvis)[np.linspace(0, len(frame_pelvis) - 1, len(verts_sample))
                                                     .round().astype(int)][:, None, :] * np.array([1.0, 0.0, 1.0]))
    rel = s_all[:, mask].reshape(-1, 3)
    D, tgt0 = fit_distance(rel, (rel.min(0) + rel.max(0)) / 2, dirv, K, width, height, view.margin)
    fp = np.asarray(frame_pelvis, dtype=np.float64)
    along = (fp[:, [0, 2]] - c) @ a
    n = max(1, int(round(fps)))                                   # a centred moving mean over 1 s
    pad = np.pad(along, (n // 2, n - 1 - n // 2), mode="edge")
    sm = np.convolve(pad, np.ones(n) / n, mode="valid")
    Rs, ts = [], []
    for s in sm:
        tgt = tgt0 + np.array([c[0], 0.0, c[1]]) + a3 * s
        R, t = RS.look_at(tgt + D * dirv, tgt, up)
        Rs.append(R)
        ts.append(t)
    info.update(axis=a.tolist(), distance_m=D, smoothing_s=n / fps, target_offset=tgt0.tolist())
    return CameraPath(K, np.stack(Rs), np.stack(ts), width, height, info)


# ---------------------------------------------------------------- scene: the chair
def box(lo, hi):
    """(vertices (8, 3), faces (12, 3)) of an axis-aligned box, faces wound outward."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 6, 2], [3, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int64)
    return v, f


CHAIR_POSTURES = ("sit", "sit_arms_forward")   # [module] the seated postures of base.POSTURES the chair is fitted to
CHAIR_BACK_JOINTS = (0, 3, 6, 9)               # [module] the trunk (pelvis, spine 1 to 3) kept off the backrest
CHAIR_BACK_CLEAR_M = 0.005                     # [module] SH5 (review of 2026-11-02): the backrest 5 mm behind it
CHAIR_ARM_CLEAR_M = 0.01                       # [module] SH5: the armrests 1 cm outside the widest pelvis or thigh
# The armrest tops fitted to the resting forearms (lane exam_video of 2026-11-03; handoff 13 section 1: "the forearms
# of the exam sit rest on the armrest's inner edge"): chair_mesh fit 'forearm', built and measured, off by session
# decision (decisions.md, 2026-11-03 evening: "the fitted armrests stay off and the armrest is lengthened instead").
# The armrests run on to the wrist (wave SH1c, lane chair; SYNTH_SHARED_CHANGES_DRAFT.md section 5: "the E1 rest
# posture holds the distal forearm in the air in front of the armrest's end"): chair_mesh fit 'wrist', built and off.
# MDS-UPDRS 3.17 asks that the patient "sit quietly in a chair with the hands placed on the arms of the chair" (Goetz
# et al. 2008, doi 10.1002/mds.22340; the form as reproduced in appendix IV of the protocol of ClinicalTrials.gov
# NCT03187301); under fit 'wrist' the forearm lies over the arm of the chair to the wrist, its lateral part over the
# top and above it, and the hand past its front end: this fit does not put the hands on the arms either. The default
# stays the SH5 chair, bit for bit, for two reasons, measured by the chair gate of tests/test_synth_scenes.py and
# tests/test_synth_pathology.py: (1) the arms of seated clips that move below the sit's (the 3.14 ladder's hands in
# the lap, the trunk-flexion jerks, the tonic tics, a Holmes tremor) pass more than involuntary.GUARD_TOL_M into the
# longer armrests, where the SH5 armrests leave them within it; (2) bodies of the frozen exam set change under it
# (tremor.chair_boxes moves the E1 rest tremor's centre toward flexion to clear the new armrests), which lifts the
# resting forearm further off its support. Switching it needs a scene kind of its own in exam.scene_mesh for the
# clips whose forearms rest on the armrests throughout, once the rest tremor carries its amplitude at the hand with
# the forearm on the support.
CHAIR_FIT = "sh5"                              # [module] the default fit; 'forearm' the fitted tops, 'wrist' the
#                                                armrests to the wrist
CHAIR_FITS = ("forearm", "sh5", "wrist")
CHAIR_ARM_GAP_M = 0.003                        # [module] the sit's lowest arm vertex over a top lies this far above
#                                                it: the clearance tremor.CLEAR_GAP_CM (0.3 cm) keeps from a scene
#                                                box, so that the resting forearm itself meets the tremor's clearance
CHAIR_ARM_WIDTH_M = 0.08                       # [module] the width of a top (the width of the SH5 armrest)
CHAIR_ARM_PAD_M = 0.04                         # [module] the thickness of a top, at most (thinner over a thigh)
CHAIR_ARM_POST_M = 0.02                        # [module] the least width of the side panel under a top
CHAIR_ARM_PATCH_M = 0.005                      # [module] the forearm's contact patch: its vertices within 5 mm of
#                                                its lowest one over the armrest's length
CHAIR_REST_GAP_M = 0.006                       # [module] fit 'wrist': the sit's lowest arm vertex over an armrest
#                                                lies this far above its top (twice CHAIR_ARM_GAP_M)
CHAIR_ARM_PAST_WRIST_M = 0.02                  # [module] fit 'wrist': the armrest ends this far in front of the
#                                                sit's wrist joint, the hand past its front end


def chair_mesh(floor_y=None, armrests=True, fit=None):
    """(vertices, faces) of the chair under the neutral seated postures (base.static_body of CHAIR_POSTURES, the sit
    first): a seat block under the thighs and buttocks of the sit; a backrest (5 cm thick, 55 cm high) whose front
    face lies CHAIR_BACK_CLEAR_M behind the rearmost trunk vertex (dominant joint in CHAIR_BACK_JOINTS) inside its
    height and width over CHAIR_POSTURES, the seat block (to 13 cm behind the sit's knees) starting at it; two
    armrests, each from behind the backrest's front face to CHAIR_ARM_PAST_WRIST_M in front of the wrist of the sit
    (fit 'wrist') or to 10 cm in front of its elbow (fits 'sh5' and 'forearm'). It stands on the floor plane floor_y
    (default mesh_floor_y()). armrests=False: the seat and the backrest only, identical to the first two boxes of the
    chair with arms (SH5: the AIMS position never rests on the armrests; with the hands on the thighs its forearms
    cross the front part of the 'wrist' armrests).

    fit 'sh5' (CHAIR_FIT, the default: the chair of shared wave SH5, 2026-11-02, bit for bit): two armrests 8 cm
    wide, each a side panel from the floor up to 1 cm below the lowest vertex of the proximal third of the sit's
    forearm, from 4 cm inside the elbow or CHAIR_ARM_CLEAR_M outside the widest pelvis or thigh vertex (dominant joint
    0, 1, 2) over CHAIR_POSTURES, whichever lies farther out; boxes in that order (viewer.box, 8 vertices each):
    seat, backrest, the armrest on the body's left (x > 0), the right one. Until SH5 of
    docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md (clinical review of 2026-11-02) the backrest's front face stood 2 cm
    behind the rearmost seat vertex of the sit, inside the seated spines, and the armrests were blocks from the seat
    up centred under the elbows, inside both hips (tests/test_synth_scenes.py measures both).

    fit 'forearm' (built on 2026-11-03, off by session decision): each armrest is a top and a side panel under it.
    The top, CHAIR_ARM_WIDTH_M wide, is centred on the sit's forearm contact patch (the forearm vertices, dominant
    joint the elbow, within CHAIR_ARM_PATCH_M of the lowest one between the elbow and the armrest's front end), moved
    outward until its inner face lies CHAIR_ARM_CLEAR_M outside the trunk beside it (the pelvis, spine, neck or leg
    vertices medial of the patch, inside the armrest's length and from CHAIR_ARM_CLEAR_M under the top's lower face to
    CHAIR_ARM_CLEAR_M over the patch, in CHAIR_POSTURES: the sit's elbows lie against the waist), its upper face
    CHAIR_ARM_GAP_M under the lowest arm vertex of that side (dominant joint the shoulder, the elbow, the wrist or a
    finger joint) over its footprint in CHAIR_POSTURES, CHAIR_ARM_PAD_M thick or less, so that its lower face stays
    CHAIR_ARM_CLEAR_M above the highest pelvis, spine, neck or leg vertex under it; the panel goes from the floor up
    to the top's lower face, its inner face where the SH5 armrest has it, its outer face flush with the top's
    (CHAIR_ARM_POST_M wide at least, the top widened outward to it). The forearm's underside then lies on the top,
    where the SH5 armrest leaves it hovering 1 cm above and medial of the armrest's inner edge
    (tests/test_synth_scenes.py measures both). Boxes in that order: seat, backrest, the top on the body's left, the
    right top, the left panel, the right panel; the tops are the armrests a forearm rests on (postures.armrest_boxes
    reads boxes 2 and 3: under this fit the tops, not the panels under them). ValueError when no top fits between a
    thigh and the forearm. Fits 'sh5' and 'forearm' end each armrest 10 cm in front of the sit's elbow: the distal
    forearm and the hand of the sit lie in front of it.

    fit 'wrist' (built in wave SH1c, off: the comment above CHAIR_FIT): the armrests of fit 'sh5' in width and place,
    run on to CHAIR_ARM_PAST_WRIST_M in front of the sit's wrist joint, each top CHAIR_REST_GAP_M under the lowest arm
    vertex of its side (dominant joint the shoulder, the elbow, the wrist or a finger joint) over its footprint in
    CHAIR_POSTURES. In the sit the forearm lies over the arm of the chair to the wrist and the hand past its front
    end; the forearm does not touch the top: the top is flat and the forearm's axis horizontal while the forearm
    tapers toward the wrist, so the underside's gap opens from CHAIR_REST_GAP_M at the lowest vertex to its largest
    under the wrist, and the inner face stays outside the hips, so only the forearm's lateral part lies over the top
    and its medial part beside the inner face (tests/test_synth_scenes.py measures the gap per 2 cm and the share).
    The arms of seated catalogue clips that move below the sit's pass into the front part of these armrests (the
    chair gate of tests/test_synth_scenes.py names the deepest of each family). Boxes in the order of fit 'sh5'."""
    fit = CHAIR_FIT if fit is None else str(fit)
    if fit not in CHAIR_FITS:
        raise ValueError(f"chair fit {fit!r}")
    from .body import model_name
    return _chair_mesh(round(float(mesh_floor_y() if floor_y is None else floor_y), 9), bool(armrests), fit,
                       model_name())


def _arm_ids(side_e):
    """(V,) bool: the vertices of the arm of the side whose elbow is joint side_e (dominant joint the shoulder,
    the elbow, the wrist or a finger joint of that side)."""
    from .hands import FIRST
    side = "left" if side_e == 18 else "right"
    dom = dominant_joint("neutral")
    return np.isin(dom, (side_e - 2, side_e, side_e + 2) + tuple(range(FIRST[side], FIRST[side] + 15)))


@functools.lru_cache(maxsize=8)
def _chair_mesh(floor, armrests=True, fit=CHAIR_FIT, model=None):
    from .base import static_body
    sit = static_body("sit", 0.01)
    V = knot_vertices(sit, [0])[0]
    dom = dominant_joint("neutral")
    J = sit.joints().numpy()[0]
    seat = np.isin(dom, (0, 1, 2))
    seat_top = float(V[seat, 1].min()) - 0.005
    knee_z = float(min(J[4, 2], J[5, 2]))
    x_half = float(np.abs(V[seat, 0]).max()) + 0.03
    trunk = np.isin(dom, CHAIR_BACK_JOINTS)
    back_z, wide = math.inf, 0.0
    poses = {}
    for name in CHAIR_POSTURES:
        P = V if name == "sit" else knot_vertices(static_body(name, 0.01), [0])[0]
        poses[name] = P
        inside = trunk & (P[:, 1] > seat_top) & (P[:, 1] < seat_top + 0.55) & (np.abs(P[:, 0]) < x_half)
        back_z = min(back_z, float(P[inside, 2].min()) - CHAIR_BACK_CLEAR_M)
        wide = max(wide, float(np.abs(P[seat, 0]).max()))
    parts = [box((-x_half, floor, back_z), (x_half, seat_top, knee_z - 0.13)),
             box((-x_half, seat_top, back_z - 0.05), (x_half, seat_top + 0.55, back_z))]
    panels = []
    for e, w in ((18, 20), (19, 21)) if armrests else ():
        z0, z1 = back_z - 0.05, float(J[e, 2]) + 0.10
        xc = float(J[e, 0])
        x_in = max(abs(xc) - 0.04, wide + CHAIR_ARM_CLEAR_M)
        s = 1.0 if xc > 0 else -1.0                       # the side's outward direction along x
        if fit in ("sh5", "wrist"):
            x0, x1 = (x_in, x_in + 0.08) if xc > 0 else (-x_in - 0.08, -x_in)
            if fit == "sh5":
                fore = np.isin(dom, (e,)) & (V[:, 2] < J[e, 2] + (J[w, 2] - J[e, 2]) / 3.0)
                top = float(V[fore, 1].min()) - 0.01
            else:                                         # past the wrist, CHAIR_REST_GAP_M under the arm
                z1 = float(J[w, 2]) + CHAIR_ARM_PAST_WRIST_M
                arm = _arm_ids(e)
                lows = [float(P[arm & (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 2] >= z0) & (P[:, 2] <= z1), 1].min(
                    initial=math.inf)) for P in poses.values()]
                top = min(lows) - CHAIR_REST_GAP_M
            parts.append(box((x0, floor, z0), (x1, top, z1)))
            continue
        along = (dom == e) & (V[:, 2] >= J[e, 2]) & (V[:, 2] <= z1)
        low = float(V[along, 1].min())
        patch = along & (V[:, 1] <= low + CHAIR_ARM_PATCH_M)
        c = abs(float(V[patch, 0].mean()))                # |x| of the contact patch's centre
        arm, rest = _arm_ids(e), ~np.isin(dom, tuple(range(13, 52)))   # the side's arm; pelvis, spine, neck, legs
        wall = -math.inf                                  # the trunk beside the top, medial of the patch
        for P in poses.values():
            ax = s * P[:, 0]
            band = rest & (P[:, 2] >= z0) & (P[:, 2] <= z1) & (ax < c) & (P[:, 1] <= low + CHAIR_ARM_CLEAR_M) & \
                (P[:, 1] >= low - CHAIR_ARM_GAP_M - CHAIR_ARM_PAD_M - CHAIR_ARM_CLEAR_M)
            wall = max(wall, float(ax[band].max(initial=-math.inf)))
        a_in = max(c - CHAIR_ARM_WIDTH_M / 2.0, wall + CHAIR_ARM_CLEAR_M)   # moved out, never narrowed
        a_out = max(a_in + CHAIR_ARM_WIDTH_M, x_in + CHAIR_ARM_POST_M)    # the panel CHAIR_ARM_POST_M wide at least
        overs = {n: (s * P[:, 0] >= a_in) & (s * P[:, 0] <= a_out) & (P[:, 2] >= z0) & (P[:, 2] <= z1)
                 for n, P in poses.items()}                # each posture's vertices over the top's footprint
        top = min((float(P[arm & overs[n], 1].min()) - CHAIR_ARM_GAP_M for n, P in poses.items()
                   if (arm & overs[n]).any()), default=math.inf)
        under = max((float(P[rest & overs[n] & (P[:, 1] < top), 1].max(initial=-math.inf)) for n, P in poses.items()),
                    default=-math.inf)                    # the body under the top, once the top is known
        bottom = max(top - CHAIR_ARM_PAD_M, under + CHAIR_ARM_CLEAR_M)
        if not bottom < top:
            raise ValueError(f"no armrest top fits under the forearm of joint {e} ({top:.4f} m) and over the "
                             f"body under it ({under:.4f} m)")
        lo, hi = sorted((s * a_in, s * a_out))
        parts.append(box((lo, bottom, z0), (hi, top, z1)))
        lo, hi = sorted((s * x_in, s * a_out))
        panels.append(box((lo, floor, z0), (hi, bottom, z1)))
    parts += panels
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs)


# ---------------------------------------------------------------- annotation
def caption_lines(spec):
    """(line 1, line 2, line 3) of a clip's caption: the operator and the grade; the parameters actually applied;
    the exam segment and the clip id. spec: a dict with 'operator', 'grade_text', 'params_text', 'segment_text',
    'clip_id'."""
    return (f"{spec['operator']} | {spec['grade_text']}", spec.get("params_text") or "",
            f"{spec['segment_text']} | {spec['clip_id']}")


@functools.lru_cache(maxsize=8)
def _font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def _fit_text(draw, text, font, width):
    """text cut with an ellipsis so that it fits width px."""
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


TEXT_STYLE = {"state": ((24, 64, 120), (255, 255, 255)),        # (box fill, text) of a frame_text span
              "stimulus": ((235, 140, 20), (20, 20, 20))}       # and of a stimulus mark (module values)


def frame_texts(truth):
    """[{from_s, to_s, text, kind}] of a clip's per-frame texts (module docstring): params['frame_text'] spans
    (kind 'state') and params['stimulus_marks'] (kind 'stimulus', from t_s for dur_s); [] for a clip without them."""
    p = (truth or {}).get("params") or {}
    out = [{"from_s": float(x["from_s"]), "to_s": float(x["to_s"]), "text": str(x["text"]), "kind": "state"}
           for x in (p.get("frame_text") or [])]
    out += [{"from_s": float(m["t_s"]), "to_s": float(m["t_s"]) + float(m["dur_s"]), "text": str(m["text"]),
             "kind": "stimulus"} for m in (p.get("stimulus_marks") or [])]
    return out


def active_texts(texts, t_s, end_s=None, eps=1e-9):
    """The items of texts (frame_texts) active at t_s: from_s <= t_s < to_s, a span reaching end_s (the clip's end)
    holding the last frame too."""
    t = float(t_s) + eps
    return [x for x in texts if x["from_s"] <= t and (t < x["to_s"] or (end_s is not None
                                                                        and x["to_s"] >= float(end_s) - eps))]


class Annotator:
    """Burns the SYNTHETIC mark, the view and the time (top) and the caption band (bottom, three lines) into the
    frames of one clip and view (module docstring). The static parts are drawn once into an RGBA overlay; each
    frame adds its time and the per-frame texts active at it (texts: frame_texts, end_s the clip's end). The
    framing keeps the body out of both bands (RESERVE)."""

    def __init__(self, width, height, lines, view_title, texts=None, end_s=None):
        from PIL import Image, ImageDraw
        self.w, self.h = int(width), int(height)
        s = self.h / 540.0
        self.fs = max(int(round(14 * s)), 9)
        pad = int(round(8 * s))
        self.band = int(round(3 * 1.3 * self.fs + 2 * pad))
        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        big = _font(max(int(round(22 * s)), 11))
        small = _font(max(int(round(12 * s)), 8))
        x0, y0 = pad, pad
        tw = int(d.textlength(LB.LABEL, font=big))
        self.mark_box = (x0, y0, x0 + tw + 2 * pad, y0 + int(round(30 * s)))
        d.rectangle(self.mark_box, fill=(150, 20, 20, 235))
        d.text((x0 + pad, y0 + int(round(3 * s))), LB.LABEL, fill=(255, 255, 255, 255), font=big)
        d.text((self.mark_box[2] + pad, y0 + int(round(8 * s))), "rendered body model, not a recording",
               fill=(40, 40, 40, 255), font=small)
        d.text((self.w - pad - int(d.textlength(view_title, font=small)), y0 + int(round(4 * s))), view_title,
               fill=(40, 40, 40, 255), font=small)
        d.rectangle((0, self.h - self.band, self.w, self.h), fill=(20, 20, 24, 210))
        f = _font(self.fs)
        cols = ((255, 255, 255, 255), (235, 235, 210, 255), (190, 190, 198, 255))
        self.lines = []
        for i, (txt, col) in enumerate(zip(lines, cols)):
            txt = _fit_text(d, txt, f, self.w - 2 * pad)
            self.lines.append(txt)
            d.text((pad, self.h - self.band + pad + int(i * 1.3 * self.fs)), txt, fill=col, font=f)
        a = np.asarray(ov, dtype=np.float32)
        top = int(self.mark_box[3]) + int(round(24 * s))
        self.slices = [slice(0, top), slice(self.h - self.band, self.h)]
        self.alpha = [a[sl, :, 3:] / 255.0 for sl in self.slices]
        self.rgb = [a[sl, :, :3] * al for sl, al in zip(self.slices, self.alpha)]
        self.time_xy = (self.w - pad, y0 + int(round(22 * s)))
        self.small = small
        self.texts = list(texts or [])
        self.end_s = end_s
        self.draw_rows = self.slices[0]   # the rows the per-frame texts are drawn into
        if self.texts:                    # where the per-frame texts go: between the note and the view title
            self.lab_font = _font(max(int(round(15 * s)), 9))
            self.lab_pad = pad
            self.lab_h = max(int(self.mark_box[3] - self.mark_box[1]), int(round(1.4 * self.lab_font.size)))
            self.lab_span = (self.mark_box[2] + pad + int(d.textlength("rendered body model, not a recording",
                                                                       font=small)) + 2 * pad,
                             self.w - pad - int(d.textlength(view_title, font=small)) - 2 * pad)
            self.lab_row2 = (x0, int(self.mark_box[3]) + max(int(round(3 * s)), 1))
            self.draw_rows = slice(0, min(self.h, max(top, self.lab_row2[1] + self.lab_h + 1)))

    def _boxes(self, d, items):
        """[(x0, y0, x1, y1, text, kind)] of the active texts laid out left to right: centred in the band's free
        span on the mark's row, or from the mark's left edge on the row below it when they do not fit there."""
        ws = [int(d.textlength(x["text"], font=self.lab_font)) + 2 * self.lab_pad for x in items]
        gap = self.lab_pad
        total = sum(ws) + gap * (len(ws) - 1)
        lo, hi = self.lab_span
        if total <= hi - lo:
            x, y = lo + (hi - lo - total) // 2, int(self.mark_box[1])
        else:
            x, y = self.lab_row2
        out = []
        for w, it in zip(ws, items):
            out.append((x, y, x + w, y + self.lab_h, it["text"], it["kind"]))
            x += w + gap
        return out

    def __call__(self, frame, t_s):
        """frame (H, W, 3) uint8 -> annotated copy (uint8)."""
        from PIL import Image, ImageDraw
        out = np.array(frame, dtype=np.uint8, copy=True)
        for sl, al, rgb in zip(self.slices, self.alpha, self.rgb):
            out[sl] = np.round(out[sl].astype(np.float32) * (1 - al) + rgb).astype(np.uint8)
        top = self.slices[0]
        img = Image.fromarray(out[top])
        d = ImageDraw.Draw(img)
        txt = f"t = {t_s:6.2f} s"
        d.text((self.time_xy[0] - int(d.textlength(txt, font=self.small)), self.time_xy[1]), txt, fill=(40, 40, 40),
               font=self.small)
        out[top] = np.asarray(img)
        items = active_texts(self.texts, t_s, self.end_s) if self.texts else []
        if items:                         # a frame without an active text keeps exactly the pixels above
            img = Image.fromarray(out[self.draw_rows])
            d = ImageDraw.Draw(img)
            for x0, y0, x1, y1, text, kind in self._boxes(d, items):
                fill, ink = TEXT_STYLE[kind]
                d.rectangle((x0, y0, x1, y1), fill=fill)
                ty = y0 + (y1 - y0 - int(round(1.2 * self.lab_font.size))) // 2
                d.text((x0 + self.lab_pad, ty), text, fill=ink, font=self.lab_font)
            out[self.draw_rows] = np.asarray(img)
        return out


# ---------------------------------------------------------------- rendering
@dataclass
class RenderOptions:
    width: int = SIZE[0]
    height: int = SIZE[1]
    fps: float = FPS
    shading: str = "gouraud"
    supersample: int = 1
    shadow: bool = True
    cull: bool = True


class ClipRenderer:
    """Frames of one body from one CameraPath (module docstring): render(k, verts) -> (H, W, 3) uint8."""

    def __init__(self, cam, opts, floor_y, scene=None):
        self.cam, self.o = cam, opts
        self.floor_y = float(floor_y)
        self.faces = faces()
        self.scene = scene
        self._static = None
        ss = int(opts.supersample)
        self.W, self.H = cam.width * ss, cam.height * ss
        self.K = cam.K.copy()
        self.K[:2] *= ss

    def _layer(self, k):
        """Floor, background and scene geometry for frame k (cached while the camera is fixed)."""
        if self._static is not None and self.cam.fixed:
            return self._static
        R, t = self.cam.R[k], self.cam.t[k]
        rgb, depth, hit, _ = RS.floor_layer(self.K, R, t, self.W, self.H, self.floor_y)
        rgb = rgb.astype(np.float32)
        if self.scene is not None:
            sv, sf = self.scene
            uv, z = RS.project(sv, self.K, R, t)
            fr = RS.rasterize(uv, z, sf, self.W, self.H)
            col = RS.shade(fr, sv, sf, CHAIR_ALBEDO, R, mode="flat")
            near = fr.depth < depth
            rgb = np.where(near[..., None], col, rgb)
            depth = np.where(near, fr.depth, depth)
            hit = hit & ~near
        self._static = (rgb, depth, hit)
        return self._static

    def render(self, k, verts):
        R, t = self.cam.R[k], self.cam.t[k]
        rgb, depth, floor_hit = self._layer(k)
        rgb = rgb.copy()
        if self.o.shadow:
            sh = RS.shadow_mask(verts, self.faces, self.K, R, t, self.W, self.H, self.floor_y)
            rgb[sh & floor_hit] *= 0.62
        uv, z = RS.project(verts, self.K, R, t)
        fn = RS.face_normals(verts, self.faces)
        nc = fn @ R.T
        cc = RS.to_camera(verts[self.faces].mean(1), R, t)
        fr = RS.rasterize(uv, z, self.faces, self.W, self.H, cull=self.o.cull, normals_cam=nc, centroids_cam=cc)
        col = RS.shade(fr, verts, self.faces, BODY_ALBEDO, R, mode=self.o.shading, fn=fn)
        near = fr.covered & (fr.depth < depth)
        rgb[near] = col[near]
        img = RS.to_uint8(RS.downsample(rgb, int(self.o.supersample)))
        return img


# ---------------------------------------------------------------- writing
def check_out(path, allowed_root=VIEWER_ROOT):
    """path unchanged when it lies inside allowed_root (default VIEWER_ROOT) and outside this tree, MoveAtlas
    and the gates-only store; PermissionError otherwise (mdfm.video.labels.check_out_dir)."""
    return LB.check_out_dir(os.fspath(path).replace("\\", "/"), allowed_root)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Mp4Writer:
    """Raw RGB frames piped to ffmpeg: H.264 (libx264, yuv420p, crf 18, faststart), playable in a browser's
    video tag; SYNTHETIC in the container title and comment."""

    def __init__(self, path, width, height, fps, title, crf=18):
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg is not on PATH")
        LB.check_name(path)
        self.path = path
        cmd = [ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-threads", "2",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        cmd += ["-metadata", f"title={LB.LABEL} {title}", "-metadata", f"comment={NOTICE}", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        self.proc.wait()
        if self.proc.returncode:
            raise RuntimeError(f"ffmpeg failed ({self.proc.returncode}): {err[-2000:]}")


def mp4_frame_count(path):
    """Frames decoded from an MP4 (ffprobe -count_frames)."""
    fp = shutil.which("ffprobe")
    out = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", path],
                         check=True, capture_output=True, text=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["nb_read_frames"]), int(s["width"]), int(s["height"]), s["r_frame_rate"]


def mp4_tags(path):
    """Container tags of an MP4 (ffprobe format tags)."""
    fp = shutil.which("ffprobe")
    out = subprocess.run([fp, "-v", "error", "-show_entries", "format_tags", "-of", "json", path], check=True,
                         capture_output=True, text=True)
    return json.loads(out.stdout).get("format", {}).get("tags", {})


def save_png(path, frame, extra=None):
    from PIL import Image
    LB.check_name(path)
    Image.fromarray(frame).save(path, pnginfo=LB.png_info(dict({"notice": NOTICE}, **(extra or {}))))


def render_view(body, view, out_dir, stem, caption, opts=RenderOptions(), scene=None, poster_s=None,
                png_frames=False, allowed_root=VIEWER_ROOT, sample_every=6, log=None, floor_y=None, truth=None,
                heading=None):
    """Render one body from one view into out_dir: <stem>_<view>.mp4 (every frame), <stem>_<view>.png (the poster
    frame at poster_s, default mid-clip) and, with png_frames, every frame as PNG in a folder. floor_y: the floor
    plane (default scene_floor of the body; a caller rendering several views passes it once). truth: the clip's
    truth record, whose per-frame texts (frame_texts) are burned into the frames. heading: the heading (deg) the
    view's azimuth is counted from (default heading_deg of the body, the root at the centre knot; a supine clip
    passes the couch's, ataxia.bed_view). Returns the record {files, camera, frames, floor_y_m; burned_texts when
    there are any}. out_dir must lie inside allowed_root; every name carries SYNTHETIC."""
    check_out(out_dir, allowed_root)
    os.makedirs(out_dir, exist_ok=True)
    name = LB.synthetic_name(f"{stem}_{view.name}")
    t = frame_times(body.duration, opts.fps)
    n = len(t)
    pick = np.unique(np.r_[np.arange(0, n, sample_every), n - 1])
    vs = vertices_at(body, t[pick])
    J = body.joints().numpy()
    fp = None
    if view.kind == "track":
        R, tr = pose_at(body, t)
        from ..body.joints import fk
        rest = body.rest()
        with torch.no_grad():
            fp = fk(R, tr, rest)[0][:, 0].numpy()
    cam = camera_path(view, vs, J[:, 0], n, opts.width, opts.height,
                      heading_deg(body) if heading is None else float(heading), fp, opts.fps)
    if floor_y is None:
        floor_y, _ = scene_floor(body, opts.fps)
    rend = ClipRenderer(cam, opts, floor_y, scene)
    texts = frame_texts(truth)
    ann = Annotator(opts.width, opts.height, caption, view.title, texts, float(t[-1]))
    mp4 = os.path.join(out_dir, name + ".mp4").replace("\\", "/")
    png = os.path.join(out_dir, name + ".png").replace("\\", "/")
    k_poster = int(np.clip(round((body.duration / 2 if poster_s is None else poster_s) * opts.fps), 0, n - 1))
    frames_dir = None
    if png_frames:
        frames_dir = os.path.join(out_dir, name + "_frames")
        os.makedirs(frames_dir, exist_ok=True)
    w = Mp4Writer(mp4, opts.width, opts.height, opts.fps, f"{stem} {view.name}")
    try:
        B = 48
        for a in range(0, n, B):
            V = vertices_at(body, t[a:a + B])
            for i in range(len(V)):
                k = a + i
                img = ann(rend.render(k, V[i]), float(t[k]))
                w.write(img)
                if k == k_poster:
                    save_png(png, img, {"clip": name, "frame": k, "t_s": f"{t[k]:.4f}"})
                if frames_dir:
                    save_png(os.path.join(frames_dir, f"{name}_{k:06d}.png"), img, {"clip": name, "frame": k})
            if log:
                log(f"  {name}: {min(a + B, n)}/{n} frames")
    finally:
        w.close()
    files = {"mp4": {"file": os.path.basename(mp4), "bytes": os.path.getsize(mp4), "sha256": sha256_file(mp4),
                     "codec": "H.264 libx264 crf 18, yuv420p, faststart"},
             "poster": {"file": os.path.basename(png), "bytes": os.path.getsize(png), "sha256": sha256_file(png),
                        "frame": k_poster}}
    if frames_dir:
        files["frames"] = {"folder": os.path.basename(frames_dir), "count": n}
    cam_rec = dict(cam.info, K=cam.K.tolist(), fixed=cam.fixed)
    if cam.fixed:
        cam_rec.update(R=cam.R[0].tolist(), t=cam.t[0].tolist())
    rec = {"view": view.name, "files": files, "camera": cam_rec, "n_frames": n, "fps": opts.fps,
           "width": opts.width, "height": opts.height, "duration_s": n / opts.fps, "floor_y_m": float(floor_y),
           "burned_caption": list(ann.lines)}
    if texts:
        rec["burned_texts"] = texts
    return rec
