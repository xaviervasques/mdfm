"""A numpy z-buffer rasterizer for the synthetic patient's mesh viewer (SP4 of
docs/proposals/SYNTH_PATIENT_DRAFT.md, section 9): triangles projected by a pinhole camera, covered pixels by
edge functions at pixel centres, the nearest surface by perspective-correct depth, flat or Gouraud shading.
No renderer package: numpy only, CPU.

Conventions.
- Camera: OpenCV, as mdfm.kernels.camera (x right, y down, z forward); a world point p sits at R p + t in the
  camera frame; K the intrinsics in pixels. look_at builds R and t (mdfm.kernels.camera.look_at). World: y up,
  metres, the body's floor at y = 0 (a joint floor, mdfm/body/frames.py).
- Pixels: pixel (row j, column i) covers [i, i + 1) x [j, j + 1) in image coordinates and is sampled at its
  centre (i + 0.5, j + 0.5). A pixel is covered by a triangle when its centre lies inside it or on an edge
  (all three barycentric weights >= 0; two triangles sharing an edge may both cover a pixel on it, and the
  depth test decides).
- Depth: the camera-frame z of the surface at the pixel centre, interpolated perspective-correctly (1/z is
  linear in the image). The nearest surface wins; between equal depths the first triangle in face order wins
  (a stable sort), so a frame is a deterministic function of its inputs.
- Clipping: a triangle with a vertex nearer than `near` (m, camera z) is dropped; the viewer places its
  cameras so that no body triangle is that near.
- Culling: cull=True drops the triangles whose outward normal (the right-hand normal of the vertex order,
  outward for the faces of both body models, tests/test_mesh_video.py and tests/test_openbody.py) points away from the camera.

Shading (shade): Lambert with an ambient term and directional lights given in the camera frame (so that
every view is lit alike): 'flat' takes the face normal, 'gouraud' interpolates per-vertex intensities (vertex
normals: the area-weighted mean of the adjacent face normals) with the perspective-correct barycentric
weights. Colours are linear in [0, 1] and leave through a 1/2.2 gamma.

Scene layers (floor_layer): a checkerboard floor plane intersected per pixel by the camera rays (exact, no
triangles), its depth, and a vertical background gradient where no ray meets the floor; the body is drawn
over it by depth. A drop shadow (shadow_mask) is the coverage of the body's triangles projected onto the
floor along a light direction, as seen by the camera.
"""
import math

import numpy as np

GAMMA = 1.0 / 2.2
CHUNK = 2_000_000            # candidate (triangle, pixel) pairs per pass, bounds the memory


# ---------------------------------------------------------------- camera
def intrinsics(width, height, vfov_deg):
    """(3, 3) K of a pinhole with square pixels, principal point at the image centre and a vertical field of
    view of vfov_deg."""
    f = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """(R (3, 3), t (3,)) of a camera at eye looking at target, OpenCV convention (x right, y down, z forward),
    the same rotation as mdfm.kernels.camera.look_at."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    n = np.linalg.norm(x)
    if n < 1e-9:
        raise ValueError("look_at: the view direction is parallel to up")
    x = x / n
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)
    return R, -R @ eye


def to_camera(points, R, t):
    """(..., 3) world -> (..., 3) camera frame."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(R).T + np.asarray(t)


def project(points, K, R, t):
    """(..., 3) world -> (uv (..., 2) px, z (...) m camera depth)."""
    pc = to_camera(points, R, t)
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


def camera_centre(R, t):
    return -np.asarray(R).T @ np.asarray(t)


# ---------------------------------------------------------------- geometry helpers
def face_normals(vertices, faces):
    """(F, 3) right-hand normals of the vertex order, unnormalized (length = twice the area)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])


def vertex_normals(vertices, faces, fn=None):
    """(V, 3) unit vertex normals: the area-weighted mean of the adjacent face normals."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    fn = face_normals(v, f) if fn is None else fn
    out = np.zeros_like(v)
    for c in range(3):
        for k in range(3):
            out[:, c] += np.bincount(f[:, k], weights=fn[:, c], minlength=len(v))
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-300)


# ---------------------------------------------------------------- rasterization
class Fragments:
    """Per-pixel result of rasterize: face (H, W) int64 (-1 where no triangle), bary (H, W, 3) perspective-
    correct barycentric weights, depth (H, W) camera z (inf where no triangle)."""

    def __init__(self, face, bary, depth):
        self.face, self.bary, self.depth = face, bary, depth

    @property
    def covered(self):
        return self.face >= 0


def _setup(uv, z, faces, width, height, near, cull, normals_cam=None, centroids_cam=None):
    """Per-face screen data of the faces that survive clipping, culling, degeneracy and the image bounds."""
    f = np.asarray(faces, dtype=np.int64)
    keep = (z[f] > near).all(1)
    if cull:
        keep &= (normals_cam * centroids_cam).sum(1) < 0
    idx = np.flatnonzero(keep)
    f = f[idx]
    p0, p1, p2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    A = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    lo = np.minimum(np.minimum(p0, p1), p2)
    hi = np.maximum(np.maximum(p0, p1), p2)
    x0 = np.maximum(np.ceil(lo[:, 0] - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.floor(hi[:, 0] - 0.5), width - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(lo[:, 1] - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.floor(hi[:, 1] - 0.5), height - 1).astype(np.int64)
    ok = (np.abs(A) > 1e-12) & (x0 <= x1) & (y0 <= y1)
    idx, f, p0, p1, p2, A = idx[ok], f[ok], p0[ok], p1[ok], p2[ok], A[ok]
    x0, x1, y0, y1 = x0[ok], x1[ok], y0[ok], y1[ok]
    # w_k = a_k x + b_k y + c_k (module docstring), divided by the signed area so that inside means all >= 0
    a = np.stack([p1[:, 1] - p2[:, 1], p2[:, 1] - p0[:, 1]], 1) / A[:, None]
    b = np.stack([p2[:, 0] - p1[:, 0], p0[:, 0] - p2[:, 0]], 1) / A[:, None]
    c = np.stack([p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0], p2[:, 0] * p0[:, 1] - p2[:, 1] * p0[:, 0]], 1) / A[:, None]
    iz = 1.0 / z[f]                                                     # (n, 3)
    return {"idx": idx, "a": a, "b": b, "c": c, "iz": iz, "x0": x0, "y0": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1}


def _fragments(s, W):
    """Yield, in passes of at most CHUNK row spans, the covered (triangle, pixel) pairs of the faces of _setup:
    (fi, px, py, w0, w1, w2). Each face row is cut to the span its three edge half-planes leave (widened by one
    pixel each side against rounding), then every pixel of the span is tested exactly (all weights >= 0)."""
    n = len(s["h"])
    bound = s["w"] * s["h"]
    start = 0
    while start < n:                         # one face is never split between passes
        cum = np.cumsum(bound[start:])
        stop = start + max(1, int(np.searchsorted(cum, CHUNK, side="right")))
        h = s["h"][start:stop]
        fr = np.repeat(np.arange(start, stop), h)                       # the face of each row
        py = s["y0"][fr] + (np.arange(int(h.sum())) - np.repeat(np.cumsum(h) - h, h))
        cy = py + 0.5
        a0, a1 = s["a"][fr, 0], s["a"][fr, 1]
        r0 = s["b"][fr, 0] * cy + s["c"][fr, 0]
        r1 = s["b"][fr, 1] * cy + s["c"][fr, 1]
        lo = s["x0"][fr].astype(np.float64)
        hi = (s["x0"][fr] + s["w"][fr] - 1).astype(np.float64)
        empty = np.zeros(len(fr), bool)
        for a, r in ((a0, r0), (a1, r1), (-(a0 + a1), 1.0 - r0 - r1)):
            pos, neg = a > 0, a < 0
            with np.errstate(divide="ignore", invalid="ignore"):
                xe = -r / a - 0.5                                       # pixel index where the edge crosses
            lo = np.where(pos, np.maximum(lo, np.ceil(xe) - 1), lo)
            hi = np.where(neg, np.minimum(hi, np.floor(xe) + 1), hi)
            empty |= (a == 0) & (r < 0)
        cnt = np.where(empty, 0, np.maximum(hi - lo + 1, 0)).astype(np.int64)
        tot = int(cnt.sum())
        rows = np.repeat(np.arange(len(fr)), cnt)
        px = lo.astype(np.int64)[rows] + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        fi, pyy = fr[rows], py[rows]
        cx, cyy = px + 0.5, pyy + 0.5
        w0 = s["a"][fi, 0] * cx + s["b"][fi, 0] * cyy + s["c"][fi, 0]
        w1 = s["a"][fi, 1] * cx + s["b"][fi, 1] * cyy + s["c"][fi, 1]
        w2 = 1.0 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        yield fi[ins], px[ins], pyy[ins], w0[ins], w1[ins], w2[ins]
        start = stop


def rasterize(uv, z, faces, width, height, near=0.05, cull=False, normals_cam=None, centroids_cam=None):
    """Fragments of triangles faces (F, 3) with projected vertices uv (V, 2) px and camera depths z (V,) m
    (module docstring). cull needs the camera-frame face normals and centroids (F, 3)."""
    W, H = int(width), int(height)
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    s = _setup(uv, z, faces, W, H, near, cull, normals_cam, centroids_cam)
    zbest = np.full(W * H, -np.inf)          # 1/z of the nearest surface so far (larger is nearer)
    fbest = np.full(W * H, -1, np.int64)
    bbest = np.zeros((W * H, 3))
    big = np.iinfo(np.int64).max
    for fi, px, py, w0, w1, w2 in _fragments(s, W):
        izf = s["iz"][fi]
        iz = izf[:, 0] + w1 * (izf[:, 1] - izf[:, 0]) + w2 * (izf[:, 2] - izf[:, 0])   # exact at constant depth
        pix = py * W + px
        # nearest per pixel (largest 1/z), ties to the lowest face index: deterministic (module docstring)
        zc = np.full(W * H, -np.inf)
        np.maximum.at(zc, pix, iz)
        near_ = iz == zc[pix]
        fc = np.full(W * H, big, np.int64)
        np.minimum.at(fc, pix[near_], fi[near_])
        win = np.flatnonzero(near_ & (fi == fc[pix]))
        pw, izw = pix[win], iz[win]
        better = (izw > zbest[pw]) | ((izw == zbest[pw]) & (s["idx"][fi[win]] < fbest[pw]))
        win, pw, izw = win[better], pw[better], izw[better]
        zbest[pw] = izw
        fbest[pw] = s["idx"][fi[win]]
        bbest[pw] = np.stack([w0[win] * izf[win, 0], w1[win] * izf[win, 1], w2[win] * izf[win, 2]], 1) / izw[:, None]
    depth = np.where(fbest >= 0, 1.0 / np.where(zbest > 0, zbest, 1.0), np.inf)
    return Fragments(fbest.reshape(H, W), bbest.reshape(H, W, 3), depth.reshape(H, W))


def coverage(uv, faces, width, height):
    """(H, W) bool: pixels whose centre lies inside any of the 2D triangles (no depth), e.g. a drop shadow."""
    W, H = int(width), int(height)
    s = _setup(np.asarray(uv, dtype=np.float64), np.ones(len(uv)), faces, W, H, 0.0, False)
    out = np.zeros(W * H, bool)
    for _, px, py, _, _, _ in _fragments(s, W):
        out[py * W + px] = True
    return out.reshape(H, W)


# ---------------------------------------------------------------- shading
LIGHTS_CAM = ((np.array([-0.45, -0.65, -0.6]), 0.62),     # key: from the camera's upper left, toward the scene
              (np.array([0.0, 0.0, -1.0]), 0.28),          # fill: a headlight
              (np.array([0.55, -0.35, 0.75]), 0.18))       # rim: from behind on the right
AMBIENT = 0.22


def light_dirs_world(R, lights=LIGHTS_CAM):
    """[(unit direction toward the light in the world, weight)] of camera-frame lights."""
    out = []
    for d, w in lights:
        d = np.asarray(d, dtype=np.float64)
        out.append((np.asarray(R).T @ (d / np.linalg.norm(d)), float(w)))
    return out


def intensity(normals, lights, ambient=AMBIENT):
    """(N,) Lambert intensity of unit normals (N, 3) under [(direction, weight)] plus ambient."""
    out = np.full(len(normals), float(ambient))
    for d, w in lights:
        out += w * np.clip(normals @ d, 0.0, None)
    return out


def shade(frag, vertices, faces, albedo, R, mode="gouraud", fn=None, vn=None, lights=None, ambient=AMBIENT):
    """(H, W, 3) linear colour of the covered pixels (zeros elsewhere). albedo: (3,) or per-face (F, 3)."""
    if mode not in ("flat", "gouraud"):
        raise ValueError(f"shading {mode!r}, expected 'flat' or 'gouraud'")
    lights = light_dirs_world(R) if lights is None else lights
    H, W = frag.face.shape
    out = np.zeros((H, W, 3), np.float32)
    cov = frag.covered
    if not cov.any():
        return out
    fid = frag.face[cov]
    f = np.asarray(faces)
    fn = face_normals(vertices, f) if fn is None else fn
    if mode == "flat":
        n = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
        I = intensity(n, lights, ambient)[fid]
    else:
        vn = vertex_normals(vertices, f, fn) if vn is None else vn
        Iv = intensity(vn, lights, ambient)
        I = (frag.bary[cov] * Iv[f[fid]]).sum(1)
    alb = np.asarray(albedo, dtype=np.float64)
    col = alb[fid] if alb.ndim == 2 else alb[None]
    out[cov] = col * I[:, None]
    return out


_LUT_N = 16384
_LUT = np.round(255.0 * (np.arange(_LUT_N + 1) / _LUT_N) ** GAMMA).astype(np.uint8)


def to_uint8(linear):
    """Linear colour in [0, 1] -> uint8 through the 1/2.2 gamma (a lookup table of 16,385 levels)."""
    idx = np.clip(np.asarray(linear, dtype=np.float32) * _LUT_N + 0.5, 0, _LUT_N).astype(np.int32)
    return _LUT[idx]


# ---------------------------------------------------------------- scene: floor, background, shadow
FLOOR = {"tile_m": 0.5, "light": np.array([0.80, 0.80, 0.78]), "dark": np.array([0.66, 0.67, 0.66]),
         "fade_m": (8.0, 25.0)}
SKY = (np.array([0.93, 0.94, 0.96]), np.array([0.78, 0.81, 0.86]))     # top, horizon (linear)


def background(width, height):
    """(H, W, 3) linear vertical gradient."""
    a = np.linspace(0.0, 1.0, int(height))[:, None, None]
    return np.broadcast_to(SKY[0] * (1 - a) + SKY[1] * a, (int(height), int(width), 3)).copy()


def floor_layer(K, R, t, width, height, floor_y=0.0, lights=None, floor=FLOOR):
    """(rgb (H, W, 3) linear, depth (H, W), hit (H, W) bool, world (H, W, 3)) of the floor plane y = floor_y
    over the background gradient: each pixel's ray meets the plane at camera depth s when s > 0."""
    W, H = int(width), int(height)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xc = (u - K[0, 2]) / K[0, 0]
    yc = (v - K[1, 2]) / K[1, 1]
    d = np.stack([xc, yc, np.ones_like(xc)], -1) @ np.asarray(R)        # world direction per unit camera depth
    C = camera_centre(R, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (floor_y - C[1]) / d[..., 1]
    hit = np.isfinite(s) & (s > 0)
    P = C + np.where(hit, s, 0.0)[..., None] * d
    tile = floor["tile_m"]
    par = (np.floor(P[..., 0] / tile) + np.floor(P[..., 2] / tile)).astype(np.int64) % 2
    col = np.where(par[..., None] == 0, floor["light"], floor["dark"])
    lights = light_dirs_world(R) if lights is None else lights
    lam = intensity(np.array([[0.0, 1.0, 0.0]]), lights, 0.35)[0]
    col = col * min(lam, 1.15) / 1.15
    dist = np.linalg.norm(P - C, axis=-1)
    a, b = floor["fade_m"]
    fade = np.clip((dist - a) / (b - a), 0.0, 1.0)[..., None]
    bg = background(W, H)
    rgb = np.where(hit[..., None], col * (1 - fade) + bg * fade, bg)
    depth = np.where(hit, s, np.inf)
    return rgb, depth, hit, P


def shadow_mask(vertices, faces, K, R, t, width, height, floor_y=0.0, light=(0.25, 1.0, 0.35)):
    """(H, W) bool: the body's triangles projected onto the floor plane along light (a world direction toward
    the light, pointing up), then into the camera (module docstring). Vertices below the floor are not
    projected."""
    v = np.asarray(vertices, dtype=np.float64)
    L = np.asarray(light, dtype=np.float64)
    L = L / np.linalg.norm(L)
    h = np.clip(v[:, 1] - floor_y, 0.0, None)
    P = v - (h / L[1])[:, None] * L
    P[:, 1] = floor_y
    uv, z = project(P, K, R, t)
    f = np.asarray(faces)
    ok = (z[f] > 0.05).all(1)
    return coverage(uv, f[ok], width, height)


def downsample(img, factor):
    """Mean over factor x factor blocks of (H, W, C) (H and W multiples of factor)."""
    if factor == 1:
        return img
    H, W = img.shape[:2]
    return img.reshape(H // factor, factor, W // factor, factor, -1).mean((1, 3))
