"""Driver pose videos: a body clip rendered into the generator's pose video (proposal section 7, step 2).

Input, BodyClip: knots of one body on the 100 Hz body clock (rot6d (K, 22, 6) with row 0 the root orientation,
transl (K, 3) m, y up, floor at y = 0), betas, gender, contact labels when known, and a source record. Two
sources here: an AMASS take of the train split of the body store (clip_from_store; the store holds no lockbox,
dev or calib person, and the train split and a lockbox person are refused here again), and the hand-built
neutral body (neutral_clip: SMPL+H neutral, betas 0, no AMASS content). A pathology operator hands over the
same BodyClip.

Frames sit at t_k = k / fps, k = 0 .. n - 1, n = round(duration x fps), on the body clock (Hermite knots,
mdfm.body.kinematics.KnotBatch.pose), at 30 or 60 fps: the 60 fps clip of a body holds the 30 fps clip in its
even frames, under the same camera, because the camera is fixed from the knots (below), not from the frames.

Framing (Framing): 'full_body', the camera of the B4.8 generic family (horizontal distance 3 to 6 m from the
clip-mean pelvis, eye height 0.8 to 1.8 m, horizontal field of view 60 to 80 degrees, drawn in that order from
the clip's seed) at a fixed view instead of a uniform azimuth (lateral: on the body's left, +x of the window
frame, 90 degrees from its heading; oblique45; frontal), then mdfm.factory.synth.keep_in_view on every drawn
point at every knot (central 0.9 of the image, depth at least 0.5 m); 'torso', the tremor framing of section
3.2: horizontal optical axis through the clip-mean torso midpoint (pelvis and neck joints), horizontal field of
view 70 degrees, distance such that the mean pelvis-neck length spans torso_fraction (one third) of the image
height at that depth.

Truth (DriverClip.truth): times, the drawn set in the world (62 points) and in pixels (float, before the
drawing truncates them), depth, score (1 drawn, 0 behind or within 0.1 m of the camera), the truncated pixels
the drawing uses, the camera, contact labels at the frames (nearest knot), the torso length in pixels (median
neck to hip-midpoint distance of the drawn set). write_clip writes the PNG frames (exact; the generator job
reads them as PIL images), the truth NPZ, the provenance JSON and, when ffmpeg is on PATH, a lossless MP4
(libx264rgb, qp 0) checked frame by frame against the PNGs; every name and metadata field carries SYNTHETIC.
"""
import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .. import paths
from ..data.canonical import Refused
from ..factory import camera_render as CR
from ..factory.synth import IN_VIEW_MARGIN, MIN_DEPTH_M, keep_in_view
from ..kernels.camera import Camera, intrinsics, look_at, project
from . import contract as C
from . import draw as D
from . import labels as LB
from . import points as P

RATE = 100.0
# camera azimuth about +y, from the heading (+z); oblique_right60: from the front and the right, toward the left hand
# from the midline (mdfm.synth.viewer's hand_left_medial azimuth, where the finger-tapping pinch opens; task VCH)
VIEWS = {"lateral": 90.0, "oblique45": 45.0, "frontal": 0.0, "oblique_right60": -60.0}
MIN_DEPTH_DRAWN = 0.1       # m: a nearer point (or one behind the camera) scores 0, as camera_render's rule
GENDERS = {0: "female", 1: "male", 2: "neutral"}
FORMAT = "mdfm-video-driver/1"


@dataclass
class BodyClip:
    """One body on the body clock (module docstring). hand6d (K, 30, 6): the hand knots of a body with posed hands
    (a synthetic patient's hand tasks, mdfm/video/exam_driver.py); None, the hands at rest."""
    rot6d: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor
    gender: str
    rate: float = RATE
    contact: np.ndarray = None
    source: dict = field(default_factory=dict)
    hand6d: torch.Tensor = None

    @property
    def n_knots(self):
        return int(self.rot6d.shape[0])

    @property
    def span_s(self):
        return (self.n_knots - 1) / self.rate


def clip_from_store(store, take, start, duration_s):
    """BodyClip of take row `take` of a BodyStore from knot `start`, duration_s long, in the window frame
    (body_store.BodyStore.windows). Refuses a store of another split than train and a lockbox person
    (canonical.Refused); ValueError when the take is shorter than the clip."""
    if getattr(store, "split", None) != "train":
        raise Refused(f"driver clips come from the train split of the body store only, not {store.split!r}")
    from ..data import amass_index          # the AMASS body store's path only: no AMASS module at the top
    row = store.takes.iloc[int(take)]
    if row["person_id"] in amass_index.LOCKBOX_PERSONS:
        raise Refused(f"{row['person_id']} is a lockbox person")
    K = int(round(duration_s * store.rate))
    bw = store.windows([int(take)], [int(start)], duration_s)
    if int(bw.lengths[0]) < K:
        raise ValueError(f"{row['take_id']}: {int(bw.lengths[0])} knots from {start}, the clip needs {K}")
    source = {"kind": "amass_body_store", "split": "train", "take_id": str(row["take_id"]),
              "archive": str(row["archive"]), "person_id": str(row["person_id"]), "start_knot": int(start),
              "n_knots": K, "store_root": str(store.root), "frame": "window frame (body_store.BodyStore.windows)",
              "licence": "AMASS: non-commercial research under the per-archive MPI EULAs, no redistribution "
                         "(MoveAtlas docs/license_matrix.csv); clips stay internal"}
    return BodyClip(bw.rot6d[0].double(), bw.transl[0].double(), bw.betas[0].double(),
                    GENDERS[int(bw.gender[0])], float(store.rate), bw.contact[0].numpy().astype(bool), source)


def neutral_clip(duration_s=2.0, rate=RATE, velocity=(0.0, 0.0, 0.0)):
    """The hand-built body: SMPL+H neutral, betas 0, rest pose, facing +z, the floor at y = 0 under its lowest
    vertex, moving at a constant velocity (m/s). No AMASS content."""
    K = int(round(duration_s * rate)) + 1
    model = P.points_model("neutral")
    betas = torch.zeros(16, dtype=torch.float64)
    v = model.body.v_template
    y0 = -float(v[:, 1].min())
    t = torch.arange(K, dtype=torch.float64)[:, None] / rate
    transl = torch.tensor([0.0, y0, 0.0], dtype=torch.float64) + t * torch.tensor(velocity, dtype=torch.float64)
    r6 = torch.zeros(K, 22, 6, dtype=torch.float64)
    r6[..., 0] = 1.0
    r6[..., 4] = 1.0                       # rot6d of the identity: first two columns (1, 0, 0), (0, 1, 0)
    source = {"kind": "hand_built", "body": "SMPL+H neutral, betas 0, rest pose", "velocity_m_s": list(velocity),
              "licence": "SMPL+H arrays: MPI licence, non-commercial research"}
    return BodyClip(r6, transl, betas, "neutral", float(rate), None, source)


@dataclass(frozen=True)
class Framing:
    """Framing rule (module docstring): rule 'full_body' or 'torso', view (VIEWS), torso_fraction and
    hfov_deg of the torso rule, margin and min_depth of keep_in_view."""
    rule: str = "full_body"
    view: str = "lateral"
    torso_fraction: float = 1.0 / 3.0
    hfov_deg: float = 70.0
    margin: float = IN_VIEW_MARGIN
    min_depth: float = MIN_DEPTH_M


def _uniform(gen, lo, hi):
    return lo + (hi - lo) * float(torch.rand((), generator=gen, dtype=torch.float64))


def frame_times(duration_s, fps):
    """t_k = k / fps for k < round(duration_s x fps), float64 s."""
    n = int(round(float(duration_s) * float(fps)))
    if n < 2:
        raise ValueError(f"{duration_s} s at {fps} fps gives {n} frames")
    return np.arange(n, dtype=np.float64) / float(fps)


def knot_points(clip):
    """The drawn set (K, 62, 3) and the SMPL joints (K, 22, 3) at the clip's knots."""
    t = np.arange(clip.n_knots, dtype=np.float64) / clip.rate
    return P.drawn_points(clip.rot6d, clip.transl, clip.betas, clip.gender, t, clip.rate, clip.hand6d)


def driver_camera(clip, framing=Framing(), seed=0, width=C.DEFAULT_SIZE[0], height=C.DEFAULT_SIZE[1], fps=30.0):
    """(camera.Camera float64, info dict) of a clip under a framing rule; the camera depends on the knots and
    the seed only, never on fps."""
    C.check_size(width, height)
    if framing.view not in VIEWS:
        raise ValueError(f"unknown view {framing.view!r}; known: {sorted(VIEWS)}")
    pts, p22 = knot_points(clip)
    az = math.radians(VIEWS[framing.view])
    side = torch.tensor([math.sin(az), 0.0, math.cos(az)], dtype=torch.float64)
    info = {"rule": framing.rule, "view": framing.view, "azimuth_deg": VIEWS[framing.view]}
    if framing.rule == "full_body":
        fam = CR.FAMILIES["generic"]
        gen = torch.Generator().manual_seed(int(seed))
        d = _uniform(gen, *fam.distance)
        h = _uniform(gen, *fam.height)
        hfov = _uniform(gen, *fam.hfov_deg)
        target = p22[:, 0].mean(0)
        eye = target + d * side
        eye[1] = h
        R, t = look_at(eye, target)
        f = (width / 2) / math.tan(math.radians(hfov) / 2)
        cam = Camera(intrinsics(f, f, width / 2, height / 2), R, t, 0.0, 0.0, int(width), int(height), float(fps))
        cam, dolly = keep_in_view(cam, pts.numpy(), framing.margin, framing.min_depth)
        info.update(distance_m=d, eye_height_m=h, hfov_deg=hfov, dolly_m=float(dolly), family="generic (B4.8)",
                    keep_in_view={"margin": framing.margin, "min_depth_m": framing.min_depth,
                                  "points": "drawn set at every knot"})
    elif framing.rule == "torso":
        hfov = float(framing.hfov_deg)
        f = (width / 2) / math.tan(math.radians(hfov) / 2)
        torso = float((p22[:, 12] - p22[:, 0]).norm(dim=-1).mean())
        target = ((p22[:, 0] + p22[:, 12]) / 2).mean(0)
        dist = f * torso / (framing.torso_fraction * height)
        R, t = look_at(target + dist * side, target)
        cam = Camera(intrinsics(f, f, width / 2, height / 2), R, t, 0.0, 0.0, int(width), int(height), float(fps))
        info.update(distance_m=dist, hfov_deg=hfov, torso_m=torso, torso_fraction=framing.torso_fraction,
                    dolly_m=0.0)
    else:
        raise ValueError(f"unknown framing rule {framing.rule!r}")
    return cam, info


@dataclass
class DriverClip:
    """A rendered clip before writing: truth (dict of arrays), camera, framing info, fps, size, seed, clip."""
    truth: dict
    camera: Camera
    framing: dict
    fps: float
    width: int
    height: int
    seed: int
    clip: BodyClip

    @property
    def n_frames(self):
        return int(self.truth["t_s"].shape[0])

    def frame(self, k):
        """Pose frame k (height, width, 3) uint8 RGB."""
        return D.draw_frame(self.width, self.height, self.truth["uv"][k], self.truth["score"][k])


def torso_px(uv):
    """Median over frames of the neck to hip-midpoint distance of the drawn set (T, 62, 2) px."""
    uv = np.asarray(uv, dtype=np.float64)
    mid = (uv[:, P.IDX["RHip"]] + uv[:, P.IDX["LHip"]]) / 2
    return float(np.median(np.linalg.norm(uv[:, P.IDX["Neck"]] - mid, axis=-1)))


def render(clip, fps, duration_s=None, framing=Framing(), seed=0, width=C.DEFAULT_SIZE[0],
           height=C.DEFAULT_SIZE[1], camera=None):
    """DriverClip of a BodyClip at fps (30 or 60 in the proposal; any positive rate) for duration_s (default the
    clip's span plus one knot); camera None draws it by driver_camera, else (Camera, info) is used as given."""
    if duration_s is None:
        duration_s = clip.n_knots / clip.rate
    t = frame_times(duration_s, fps)
    if t[-1] > clip.span_s + 1e-9:
        raise ValueError(f"{len(t)} frames at {fps} fps end at {t[-1]:.4f} s, after the clip ({clip.span_s:.4f} s)")
    cam, info = driver_camera(clip, framing, seed, width, height, fps) if camera is None else camera
    cam = cam._replace(fps=float(fps))
    pts, p22 = P.drawn_points(clip.rot6d, clip.transl, clip.betas, clip.gender, t, clip.rate, clip.hand6d)
    with torch.no_grad():
        uv, z = project(pts, cam.K, cam.R, cam.t)
    uv, z = uv.numpy(), z.numpy()
    score = (z > MIN_DEPTH_DRAWN).astype(np.float32)
    knot = np.clip(np.rint(t * clip.rate).astype(np.int64), 0, clip.n_knots - 1)
    contact = clip.contact[knot] if clip.contact is not None else np.zeros((len(t), 2), bool)
    truth = {"t_s": t, "uv": uv, "depth": z, "score": score, "drawn_xy": np.trunc(uv).astype(np.int64),
             "points_world": pts.numpy(), "pelvis_world": p22[:, 0].numpy(), "contact": contact,
             "contact_knot": knot, "torso_px": torso_px(uv), "names": np.array(P.NAMES),
             "K": cam.K.numpy(), "R": cam.R.numpy(), "t": cam.t.numpy(), "fps": float(fps),
             "width": int(width), "height": int(height)}
    return DriverClip(truth, cam, info, float(fps), int(width), int(height), int(seed), clip)


def held_face_video(image, box, n_frames):
    """The face video of the proposal (section 3.1): the reference identity's face crop held fixed, as n_frames
    copies of image (a PIL image) cropped to box (x0, y0, x1, y1) px and resized to 512 x 512 (bilinear, as the
    preprocessing's cv2.resize). The box comes from the identity image, chosen when the identity exists."""
    from PIL import Image
    face = image.convert("RGB").crop(tuple(int(v) for v in box)).resize((C.FACE_SIZE, C.FACE_SIZE), Image.BILINEAR)
    return [face] * int(n_frames)


# ---------------------------------------------------------------- writing
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def code_version():
    """Code SHA-256 of mdfm/ and of mdfm/video/ (the rule of mdfm.train.provenance.code_sha256), the commits of
    this tree and of MoveAtlas (read from their .git files)."""
    from ..train.provenance import code_sha256, moveatlas_head
    return {"mdfm_code_sha256": code_sha256(), "video_code_sha256": code_sha256(os.path.dirname(__file__)),
            "mdfm_git_head": moveatlas_head(paths.TREE), "moveatlas_head": moveatlas_head()}


def clip_stem(clip_id, fps):
    return LB.synthetic_name(f"driver_{clip_id}_{float(fps):g}fps")


def write_clip(dc, out_root, clip_id, allowed_root=LB.VIDEO_ROOT, mp4=True, script=None, extra=None):
    """Write a DriverClip under out_root/<SYNTHETIC_driver_<clip_id>_<fps>fps>/ (module docstring) and return its
    provenance record. The folder must not exist. out_root must lie inside allowed_root (labels.check_out_dir)."""
    stem = clip_stem(clip_id, dc.fps)
    folder = os.path.join(out_root, stem).replace("\\", "/")
    LB.check_out_dir(folder, allowed_root)
    if os.path.exists(folder):
        raise FileExistsError(f"{folder} exists: choose another clip id")
    frames_dir = os.path.join(folder, "frames")
    os.makedirs(frames_dir)
    from PIL import Image
    lines, fbytes = [], 0
    for k in range(dc.n_frames):
        name = LB.check_name(f"{stem}_{k:06d}.png")
        p = os.path.join(frames_dir, name)
        Image.fromarray(dc.frame(k)).save(p, pnginfo=LB.png_info({"clip": stem, "frame": k, "fps": f"{dc.fps:g}"}))
        lines.append(f"{name}\t{_sha256_file(p)}\n")
        fbytes += os.path.getsize(p)
    frames_sha = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    truth_path = os.path.join(folder, LB.check_name(f"{stem}_truth.npz"))
    np.savez_compressed(truth_path, label=LB.LABEL, **dc.truth)
    files = {"frames": {"folder": "frames", "pattern": f"{stem}_%06d.png", "count": dc.n_frames, "bytes": fbytes,
                        "list_sha256": frames_sha},
             "truth": {"file": os.path.basename(truth_path), "bytes": os.path.getsize(truth_path),
                       "sha256": _sha256_file(truth_path)}}
    ff = shutil.which("ffmpeg") if mp4 else None
    if ff:
        mp4_path = os.path.join(folder, LB.check_name(f"{stem}_pose.mp4"))
        cmd = [ff, "-v", "error", "-y", "-framerate", f"{dc.fps:g}", "-i", os.path.join(frames_dir, f"{stem}_%06d.png"),
               "-c:v", "libx264rgb", "-qp", "0", "-preset", "medium", "-pix_fmt", "rgb24"]
        cmd += LB.ffmpeg_metadata(f"driver pose video {stem}") + [mp4_path]
        subprocess.run(cmd, check=True, capture_output=True)
        files["mp4"] = {"file": os.path.basename(mp4_path), "bytes": os.path.getsize(mp4_path),
                        "sha256": _sha256_file(mp4_path), "codec": "libx264rgb qp 0 (lossless RGB)",
                        "frames_equal_png": mp4_equals_frames(ff, mp4_path, dc)}
    rec = provenance(dc, stem, files, script=script, extra=extra)
    with open(os.path.join(folder, LB.check_name(f"{stem}_provenance.json")), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    rec["folder"] = folder
    return rec


def mp4_equals_frames(ffmpeg, mp4_path, dc):
    """True when the MP4 decodes (rgb24) to the drawn frames bit for bit, frame count included."""
    n = dc.width * dc.height * 3
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-i", mp4_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    ok, k = True, 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if not buf:
                break
            if len(buf) != n or k >= dc.n_frames:
                ok = False
                break
            img = np.frombuffer(buf, np.uint8).reshape(dc.height, dc.width, 3)
            ok = ok and bool(np.array_equal(img, dc.frame(k)))
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return ok and k == dc.n_frames


def provenance(dc, stem, files, script=None, extra=None):
    """The provenance record of a written clip."""
    plan = C.segment_plan(dc.n_frames)
    stick, hand = D.stick_widths(dc.width, dc.height)
    rec = {"format": FORMAT, "label": LB.LABEL, "notice": LB.NOTICE, "clip": stem, "kind": "driver pose video",
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "body": dict(dc.clip.source, gender=dc.clip.gender, betas=[float(b) for b in dc.clip.betas],
                        rate_hz=dc.clip.rate, n_knots=dc.clip.n_knots,
                        **({} if dc.clip.hand6d is None else
                           {"hands": "posed: the 30 hand joints of the clip's hand6d (points.py opt-in path)"})),
           "seed": dc.seed, "fps": dc.fps, "n_frames": dc.n_frames, "duration_s": dc.n_frames / dc.fps,
           "width": dc.width, "height": dc.height,
           "framing": dc.framing,
           "camera": {"K": dc.truth["K"].tolist(), "R": dc.truth["R"].tolist(), "t": dc.truth["t"].tolist()},
           "torso_px": dc.truth["torso_px"],
           "drawing": {"format": "Wan2.2 draw_aapose_by_meta_new: threshold 0.5, stickwidth v2, hands drawn, "
                                 "black canvas, RGB", "backend": D.BACKEND, "stick_body_px": stick,
                       "stick_hand_px": hand, "points": list(P.NAMES)},
           "segments": plan, "latents_per_second": C.latents_per_second(dc.fps),
           "generator_contract": C.contract(dc.fps, dc.width, dc.height),
           "code": dict(code_version(), script=script or {}),
           "files": files}
    if extra:
        rec["extra"] = extra
    return rec
