"""The synthetic reference identity of the video round trip, version 0: a mannequin rendered from the SMPL+H mesh
(proposal docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md, sections 3.1 and 6; for the pilot of section 7, step 4, no
human identity and no image model are used).

The mannequin: SMPL+H neutral, betas 0, at its rest pose (the template mesh, arms out), seen from the front (camera
on the +z side of the body, looking along -z, y up, horizontal optical axis through the centre of the mesh's
bounding box), on a plain light background, with flat shading: one grey level per triangle from an ambient and a
Lambert term (light from the upper left of the camera), no texture, so every pixel is grey (R = G = B). The template's
average face geometry (eye sockets, nose, lips, ears) shows in the shading: the mannequin has no texture, not a
smooth head. The body spans BODY_FRACTION of the image height at a horizontal field of view of HFOV_DEG. The
reference image is rendered at the pose video's size, since the generator resizes it to the call's size without
keeping its aspect (reports/video_driver/summary.md).

The face crop: a square box centred on the bounding box of the projected head vertices (those whose largest
skinning weight is on the head joint, SMPL joint 15), its side FACE_BOX_FACTOR (1.3) times the larger side of that
box, moved inside the image; cropped and resized to 512 x 512 by render.held_face_video, which holds it for every
pose frame (proposal section 3.1). Wan2.2's preprocessing takes 1.3 times the box of the 68 face keypoints; a
mannequin has none, so the head silhouette stands in for them (a [cfg] choice).

Rasterizer (numpy, CPU): per triangle, the pixel centres (integer coordinates, the OpenCV convention of
mdfm.kernels.camera.project) inside its bounding box are tested with barycentric weights (edges included, either
winding), depth is resolved by a z-buffer on 1/z (linear in screen space), and the image is rendered SUPERSAMPLE
times larger per axis and box-filtered down for anti-aliased edges. Every triangle is drawn; normals are turned
toward the camera before shading.

The SMPL+H arrays stay in memory (MPI licence, non-commercial research; mdfm/body/smplh.py): only rendered pixels
are written. Labels: SYNTHETIC in every file name, the PNG text chunks (labels.png_info) and the provenance JSON,
written only under <store>/video (labels.check_out_dir). No burned-in mark: the reference image and the face crop
are generator inputs, the reason labels.py gives for driver videos.
"""
import datetime
import hashlib
import json
import math
import os

import numpy as np
import torch

from .. import paths
from ..kernels.camera import intrinsics, look_at, project
from . import contract as C
from . import labels as LB
from . import render as R

FORMAT = "mdfm-video-identity/1"
TAG = "mannequin_neutral"
HEAD_JOINT = 15                 # SMPL joint 15, head
HFOV_DEG = 30.0                 # [cfg] narrow lens: little perspective on a front view
BODY_FRACTION = 0.85            # [cfg] share of the image height the mesh's vertical extent spans
SUPERSAMPLE = 3                 # [cfg] per axis
ALBEDO = 0.62                   # [cfg] grey of the body
AMBIENT = 0.35                  # [cfg]
DIFFUSE = 0.65                  # [cfg]
LIGHT_DIR_CAM = (-0.3, -0.5, -1.0)   # [cfg] toward the light, camera frame (x right, y down, z forward)
BACKGROUND = 240                # [cfg] plain light grey, 8-bit
FACE_BOX_FACTOR = 1.3           # Wan2.2 preprocessing face crop factor (module docstring)
MIN_DEPTH = 0.1                 # m: triangles with a vertex nearer than this are not drawn
KIND = ("mannequin: SMPL+H neutral, betas 0, rest pose, front view, flat grey shading, plain background, no texture; "
        "the template's average face geometry shows in the shading; no human identity, no image model")


def rest_mesh(gender="neutral", betas=None):
    """Vertices (6890, 3) m of the shaped rest mesh (v_template + shapedirs @ betas, betas 0 by default), faces
    (13776, 3) int64 and the head vertex ids, float64. Read from the MoveAtlas SMPL+H file, kept in memory only."""
    m = np.load(paths.smplh_model(gender))
    v = np.asarray(m["v_template"], dtype=np.float64)
    b = np.zeros(16) if betas is None else np.asarray(betas, dtype=np.float64).reshape(-1)
    v = v + np.einsum("vcs,s->vc", np.asarray(m["shapedirs"], dtype=np.float64)[:, :, :b.shape[0]], b)
    faces = np.asarray(m["f"], dtype=np.int64)
    head = np.flatnonzero(np.argmax(np.asarray(m["weights"]), axis=1) == HEAD_JOINT)
    return v, faces, head


def front_camera(vertices, width, height, hfov_deg=HFOV_DEG, body_fraction=BODY_FRACTION):
    """(K, R, t) float64 tensors and an info dict: the front camera of the module docstring, at the distance where
    the vertical extent of the mesh spans body_fraction of the image height."""
    C.check_size(width, height)
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    target = (lo + hi) / 2
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    dist = f * float(hi[1] - lo[1]) / (body_fraction * height)
    eye = target + np.array([0.0, 0.0, dist])
    Rm, t = look_at(torch.as_tensor(eye), torch.as_tensor(target))
    K = intrinsics(f, f, width / 2, height / 2)
    info = {"view": "front (+z of the body, looking along -z), horizontal optical axis", "hfov_deg": hfov_deg,
            "focal_px": f, "distance_m": float(np.linalg.norm(eye - target)), "body_fraction": body_fraction,
            "target_m": target.tolist(), "eye_m": eye.tolist()}
    return K, Rm, t, info


def face_shades(cam_vertices, faces, albedo=ALBEDO, ambient=AMBIENT, diffuse=DIFFUSE, light=LIGHT_DIR_CAM):
    """Grey level in [0, 255] of every triangle: 255 albedo (ambient + diffuse max(0, n . l)), n the unit normal
    turned toward the camera, l the unit direction toward the light (camera frame)."""
    p = np.asarray(cam_vertices, dtype=np.float64)[faces]                # (F, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    centroid = p.mean(1)
    n[(n * -centroid).sum(1) < 0] *= -1.0                                  # face the camera at the origin
    lv = np.asarray(light, dtype=np.float64)
    lv = lv / np.linalg.norm(lv)
    return np.clip(255.0 * albedo * (ambient + diffuse * np.clip(n @ lv, 0.0, None)), 0.0, 255.0)


def rasterize(uv, depth, faces, shades, width, height, background=BACKGROUND):
    """Grey image (height, width) float64 and the index of the drawn triangle per pixel (-1: background). uv (V, 2)
    px with pixel centres at integer coordinates, depth (V,) m, faces (F, 3), shades (F,) grey levels."""
    img = np.full((height, width), float(background))
    inv = np.zeros((height, width))                                        # 1/z of the nearest drawn surface
    fid = np.full((height, width), -1, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    tri = uv[faces]                                                        # (F, 3, 2)
    iz = 1.0 / np.where(depth > 0, depth, np.inf)[faces]                   # (F, 3)
    near = (depth[faces] >= MIN_DEPTH).all(1)
    x0s = np.maximum(np.ceil(tri[..., 0].min(1)), 0).astype(np.int64)
    x1s = np.minimum(np.floor(tri[..., 0].max(1)), width - 1).astype(np.int64)
    y0s = np.maximum(np.ceil(tri[..., 1].min(1)), 0).astype(np.int64)
    y1s = np.minimum(np.floor(tri[..., 1].max(1)), height - 1).astype(np.int64)
    for k in np.flatnonzero(near & (x0s <= x1s) & (y0s <= y1s)):
        (ax, ay), (bx, by), (cx, cy) = tri[k]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        x0, x1, y0, y1 = x0s[k], x1s[k], y0s[k], y1s[k]
        X, Y = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
        wa = ((bx - X) * (cy - Y) - (by - Y) * (cx - X)) / area
        wb = ((cx - X) * (ay - Y) - (cy - Y) * (ax - X)) / area
        wc = 1.0 - wa - wb
        inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
        if not inside.any():
            continue
        z = wa * iz[k, 0] + wb * iz[k, 1] + wc * iz[k, 2]
        sub = inv[y0:y1 + 1, x0:x1 + 1]
        win = inside & (z > sub)
        sub[win] = z[win]
        img[y0:y1 + 1, x0:x1 + 1][win] = shades[k]
        fid[y0:y1 + 1, x0:x1 + 1][win] = k
    return img, fid


def supersampled_intrinsics(K, ss):
    """K of an image ss times larger per axis, pixel centres kept at integer coordinates: f -> ss f,
    c -> ss (c + 0.5) - 0.5."""
    K = torch.as_tensor(K, dtype=torch.float64).clone()
    K[0, 0] *= ss
    K[1, 1] *= ss
    K[0, 2] = ss * (K[0, 2] + 0.5) - 0.5
    K[1, 2] = ss * (K[1, 2] + 0.5) - 0.5
    return K


def face_box(head_uv, width, height, factor=FACE_BOX_FACTOR):
    """(x0, y0, x1, y1) int px: the square box of the module docstring (side factor times the larger side of the
    head's bounding box, at most min(width, height), moved inside the image)."""
    uv = np.asarray(head_uv, dtype=np.float64)
    lo, hi = uv.min(0), uv.max(0)
    side = int(round(min(factor * float((hi - lo).max()), min(width, height))))
    cx, cy = (lo + hi) / 2
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = min(max(x0, 0), width - side)
    y0 = min(max(y0, 0), height - side)
    return x0, y0, x0 + side, y0 + side


def render_mannequin(width, height, gender="neutral", ss=SUPERSAMPLE):
    """The mannequin at width x height: (image (height, width, 3) uint8 RGB, info dict with the camera, the face box,
    the silhouette's pixel extent and the drawing parameters)."""
    C.check_size(width, height)
    v, faces, head = rest_mesh(gender)
    K, Rm, t, cam = front_camera(v, width, height)
    vt = torch.as_tensor(v)
    with torch.no_grad():
        uv, z = project(vt, K, Rm, t)
        uv_ss, _ = project(vt, supersampled_intrinsics(K, ss), Rm, t)
        pc = vt @ Rm.T + t
    shades = face_shades(pc.numpy(), faces)
    img_ss, fid = rasterize(uv_ss.numpy(), z.numpy(), faces, shades, width * ss, height * ss)
    grey = img_ss.reshape(height, ss, width, ss).mean((1, 3))
    image = np.repeat(np.clip(np.rint(grey), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    mask = (fid >= 0).reshape(height, ss, width, ss).any((1, 3))
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    box = face_box(uv.numpy()[head], width, height)
    info = {"camera": dict(cam, K=K.tolist(), R=Rm.tolist(), t=t.tolist()), "face_box": list(box),
            "silhouette_px": {"x0": int(cols.min()), "x1": int(cols.max()), "y0": int(rows.min()),
                              "y1": int(rows.max()), "pixels": int(mask.sum())},
            "head_vertices": int(head.size),
            "shading": {"albedo": ALBEDO, "ambient": AMBIENT, "diffuse": DIFFUSE,
                        "light_dir_camera": list(LIGHT_DIR_CAM), "background_grey": BACKGROUND,
                        "supersample": ss, "rasterizer": "numpy z-buffer, barycentric, flat shading"},
            "body": {"model": f"SMPL+H {gender}", "betas": [0.0] * 16, "pose": "rest (template mesh)",
                     "vertices": int(v.shape[0]), "faces": int(faces.shape[0]),
                     "licence": "SMPL+H arrays: MPI licence, non-commercial research"}}
    return image, info


def face_crop(image, box):
    """The 512 x 512 face crop of the reference image (render.held_face_video, one frame)."""
    from PIL import Image
    img = image if hasattr(image, "crop") else Image.fromarray(np.asarray(image, dtype=np.uint8))
    return R.held_face_video(img, box, 1)[0]


def identity_stem(width, height, tag=TAG):
    return LB.synthetic_name(f"identity_{tag}_{int(width)}x{int(height)}")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_identity(out_root, width, height, allowed_root=LB.VIDEO_ROOT, script=None, gender="neutral"):
    """Render and write the mannequin identity under out_root/<SYNTHETIC_identity_mannequin_neutral_<w>x<h>>/: the
    reference image, the 512 x 512 face crop and the provenance JSON. The folder must not exist; out_root must lie
    inside allowed_root (labels.check_out_dir). Returns the provenance record (with 'folder')."""
    from PIL import Image
    stem = identity_stem(width, height)
    folder = os.path.join(out_root, stem).replace("\\", "/")
    LB.check_out_dir(folder, allowed_root)
    if os.path.exists(folder):
        raise FileExistsError(f"{folder} exists: an identity is never rewritten")
    image, info = render_mannequin(width, height, gender)
    face = face_crop(image, info["face_box"])
    os.makedirs(folder)
    files = {}
    for key, name, img in (("reference", f"{stem}_reference.png", Image.fromarray(image)),
                           ("face", f"{stem}_face{C.FACE_SIZE}.png", face)):
        p = os.path.join(folder, LB.check_name(name))
        img.save(p, pnginfo=LB.png_info({"identity": stem, "image": key}))
        files[key] = {"file": name, "bytes": os.path.getsize(p), "sha256": _sha256_file(p),
                      "size": [int(img.size[0]), int(img.size[1])]}
    rec = {"format": FORMAT, "label": LB.LABEL, "notice": LB.NOTICE, "identity": stem, "tag": TAG,
           "kind": KIND,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "width": int(width), "height": int(height), "face_size": C.FACE_SIZE, "face_box": info["face_box"],
           "face_box_rule": f"square, {FACE_BOX_FACTOR} x the larger side of the projected head vertices' box "
                            f"(largest skinning weight on SMPL joint {HEAD_JOINT}), moved inside the image; resized "
                            f"to {C.FACE_SIZE} x {C.FACE_SIZE} bilinear (render.held_face_video)",
           "camera": info["camera"], "silhouette_px": info["silhouette_px"], "shading": info["shading"],
           "body": info["body"], "files": files, "code": dict(R.code_version(), script=script or {})}
    with open(os.path.join(folder, LB.check_name(f"{stem}_provenance.json")), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    rec["folder"] = folder
    return rec
