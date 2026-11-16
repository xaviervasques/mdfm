"""Tests of the synthetic patient's mesh viewer (task SP4): mdfm/synth/raster.py, mdfm/synth/viewer.py,
mdfm/synth/exam.py and scripts/render_mesh_video.py.

- rasterizer geometry: a known triangle covers exactly the pixel centres inside it (checked against an
  independent per-pixel solve), perspective-correct depth, the nearer of two triangles wins whatever their
  order, ties go to the lower face index, back faces are culled, the faces of the selected body model (the open
  body by default, plan phase C wave 1) are wound outward;
- view matrices: look_at is a rotation, the body's left appears on the image's right from the front and the
  body's front on the image's left from its left side, every view of VIEWS puts its camera where its name says
  and frames its region inside the window left by the caption bands;
- the mesh equals the body truth frame by frame: at a frame on a knot the viewer's vertices are the backend's
  skinning of that knot's pose (zero error up to rounding), between knots the pose is the body's Hermite
  interpolant; on the SMPL+H reference (behind its flag) the hand skinning is SMPLHBody.vertices bit for bit;
- the tests of the exam set's AMASS references (the stand rule, the retarget) run on the reference until the
  port's wave 7 replaces those references (owner decision C8);
- frame counts: floor(span x fps) + 1 frames, decoded from the MP4 at 30 fps;
- labels: SYNTHETIC in every file name, in the MP4 title and comment, in the PNG text chunk, burned into every
  frame; the caption names the operator and the grade of every clip of the exam set;
- refusal: nothing is written outside <store>/synth_viewer (the tree, a temporary folder), names without
  SYNTHETIC are refused; the AMASS references refuse another split and a lockbox person; a clip whose truth fails
  a check (limits, contact, continuity) is refused before anything is written;
- the floor (review of 2026-10-05): the scene floor is the body's own mesh floor and its report reads every frame;
  the mesh grounding of a reference brings a floating body back onto its floor and keeps its rotations;
- the exam set holds the four families at the grades the draft lists, each with its level 0 (the grades the
  planted foot bounds are listed, not rendered), and the index page is a local file of plain video tags without
  script.
Bodies are procedural and in the neutral shape (no AMASS read, no lockbox, EVAL, CODY or care_pd record). Clips of
the tests are written under pytest's temporary folder with the allowed root moved there.
"""
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
HAVE_FF = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")


def _script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_mesh_video_script",
                                                  os.path.join(paths.TREE, "scripts", "render_mesh_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def M():
    from mdfm.synth import base, exam, raster, viewer
    return {"RS": raster, "V": viewer, "EX": exam, "base": base}


# ---------------------------------------------------------------- rasterizer geometry
def _brute_cover(uv, W, H):
    """(H, W) bool: pixel centres inside the triangle, by solving p = a + s (b - a) + t (c - a) per pixel."""
    a, b, c = (np.asarray(x, dtype=np.float64) for x in uv)
    M = np.stack([b - a, c - a], 1)
    Minv = np.linalg.inv(M)
    out = np.zeros((H, W), bool)
    for j in range(H):
        for i in range(W):
            s, t = Minv @ (np.array([i + 0.5, j + 0.5]) - a)
            out[j, i] = s >= -1e-12 and t >= -1e-12 and s + t <= 1 + 1e-12
    return out


@pytest.mark.parametrize("tri", [((3.2, 2.7), (27.9, 6.1), (11.4, 19.6)), ((25.0, 3.0), (2.5, 18.25), (29.5, 21.0)),
                                 ((0.5, 0.5), (8.5, 0.5), (0.5, 8.5))])
def test_a_known_triangle_covers_exactly_the_pixel_centres_inside_it(M, tri):
    RS = M["RS"]
    W, H = 32, 24
    for order in ((0, 1, 2), (0, 2, 1)):                       # either winding
        uv = np.array([tri[k] for k in order])
        fr = RS.rasterize(uv, np.full(3, 2.0), np.array([[0, 1, 2]]), W, H)
        want = _brute_cover(uv, W, H)
        np.testing.assert_array_equal(fr.covered, want)
        e1, e2 = uv[1] - uv[0], uv[2] - uv[0]
        area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        per = sum(np.linalg.norm(uv[k] - uv[(k + 1) % 3]) for k in range(3))
        assert abs(int(want.sum()) - area) <= per                # a sanity bound on the covered count
        np.testing.assert_allclose(fr.depth[fr.covered], 2.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fr.bary[fr.covered].sum(-1), 1.0, atol=1e-12)
    cov = RS.coverage(np.array(tri), np.array([[0, 1, 2]]), W, H)
    np.testing.assert_array_equal(cov, _brute_cover(np.array(tri), W, H))


def test_depth_is_perspective_correct(M):
    """A plane tilted in depth, seen by a pinhole: the rasterized depth equals the ray's intersection with the
    plane at every covered pixel."""
    RS = M["RS"]
    W, H = 40, 30
    K = RS.intrinsics(W, H, 50.0)
    R, t = np.eye(3), np.zeros(3)
    P = np.array([[-1.0, -0.6, 2.0], [1.2, -0.5, 4.0], [0.0, 0.9, 6.0]])     # camera frame = world
    uv, z = RS.project(P, K, R, t)
    fr = RS.rasterize(uv, z, np.array([[0, 1, 2]]), W, H)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    js, is_ = np.nonzero(fr.covered)
    assert len(js) > 50
    for j, i in zip(js, is_):
        d = np.array([(i + 0.5 - K[0, 2]) / K[0, 0], (j + 0.5 - K[1, 2]) / K[1, 1], 1.0])
        s = (n @ P[0]) / (n @ d)                                    # the ray p = s d meets the plane at depth s
        assert fr.depth[j, i] == pytest.approx(s, rel=1e-10)
        x = s * d                                                   # perspective-correct weights rebuild the point
        np.testing.assert_allclose(fr.bary[j, i] @ P, x, atol=1e-9)


def test_the_nearer_triangle_wins_whatever_the_order_and_ties_go_to_the_lower_index(M):
    RS = M["RS"]
    W, H = 30, 20
    uv = np.array([[2.0, 2.0], [26.0, 3.0], [8.0, 18.0], [4.0, 1.0], [28.0, 10.0], [3.0, 17.0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    for z_near, z_far in ((2.0, 3.0), (3.0, 2.0)):
        z = np.array([z_near] * 3 + [z_far] * 3)
        for f in (faces, faces[::-1]):
            fr = RS.rasterize(uv, z, f, W, H)
            both = RS.rasterize(uv[:3], z[:3], faces[:1], W, H).covered & \
                RS.rasterize(uv[3:], z[3:], np.array([[0, 1, 2]]), W, H).covered
            assert both.sum() > 20
            near_face = 0 if z_near < z_far else 1                 # by vertex rows: face [0, 1, 2] is at z_near
            want = np.flatnonzero((f == faces[near_face]).all(1))[0]
            assert (fr.face[both] == want).all()
            assert fr.depth[both].max() == pytest.approx(min(z_near, z_far))
    fr = RS.rasterize(uv, np.full(6, 2.5), faces, W, H)
    both = (RS.rasterize(uv[:3], np.full(3, 2.5), faces[:1], W, H).covered
            & RS.rasterize(uv[3:], np.full(3, 2.5), np.array([[0, 1, 2]]), W, H).covered)
    assert (fr.face[both] == 0).all()                               # equal depth: the lower face index


def _reference(monkeypatch):
    """The core on the SMPL+H reference, behind its flag (body.model_name)."""
    from mdfm.synth import openbody as OB
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")


@pytest.mark.parametrize("model", ["open", "smplh_reference"])
def test_back_faces_are_culled_and_the_faces_point_outward(M, model, monkeypatch):
    """The faces of both body models are wound outward (a positive signed volume), so culling the faces turned away
    from the camera leaves the image of a posed body unchanged; a single triangle is kept or culled by its
    winding."""
    RS, V = M["RS"], M["V"]
    if model != "open":
        _reference(monkeypatch)
    faces, v = V.faces(), V.rest_vertices()
    vol = np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]])).sum() / 6.0
    assert vol > 0.03                                               # m^3: an outward closed surface
    stand = M["base"].static_body("stand", 0.01)
    verts = V.knot_vertices(stand, [0])[0]
    for name in ("front", "left", "top"):
        cam = V.camera_path(V.VIEWS[name], verts[None], stand.joints().numpy()[:, 0], 1, 240, 160, 0.0)
        R, t = cam.R[0], cam.t[0]
        uv, z = RS.project(verts, cam.K, R, t)
        fn = RS.face_normals(verts, faces)
        a = RS.rasterize(uv, z, faces, 240, 160)
        b = RS.rasterize(uv, z, faces, 240, 160, cull=True, normals_cam=fn @ R.T,
                         centroids_cam=RS.to_camera(verts[faces].mean(1), R, t))
        np.testing.assert_array_equal(a.covered, b.covered)
        assert (a.face[a.covered] == b.face[b.covered]).mean() > 0.999
    # a triangle facing away from the camera is culled, the same one facing it is kept
    K = RS.intrinsics(40, 30, 50.0)
    P = np.array([[-0.5, -0.5, 3.0], [0.5, -0.5, 3.0], [0.0, 0.5, 3.0]])
    for f, kept in ((np.array([[0, 2, 1]]), True), (np.array([[0, 1, 2]]), False)):
        uv, z = RS.project(P, K, np.eye(3), np.zeros(3))
        n = RS.face_normals(P, f)
        fr = RS.rasterize(uv, z, f, 40, 30, cull=True, normals_cam=n, centroids_cam=P[f].mean(1))
        assert fr.covered.any() == kept


# ---------------------------------------------------------------- view matrices
def test_look_at_is_a_rotation_with_the_body_left_on_the_image_right_from_the_front(M):
    RS = M["RS"]
    K = RS.intrinsics(200, 100, 40.0)
    R, t = RS.look_at((0.0, 1.0, 5.0), (0.0, 1.0, 0.0))           # in front of a body facing +z
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    uv, z = RS.project(np.array([[0.0, 1.0, 0.0], [0.4, 1.0, 0.0], [0.0, 1.5, 0.0]]), K, R, t)
    np.testing.assert_allclose(uv[0], [100.0, 50.0], atol=1e-9)    # the target on the principal point
    assert z[0] == pytest.approx(5.0)
    assert uv[1, 0] > 100.0                                        # the body's left (+x) on the image's right
    assert uv[2, 1] < 50.0                                         # up in the world is up in the image
    R, t = RS.look_at((5.0, 1.0, 0.0), (0.0, 1.0, 0.0))           # on the body's left side
    uv, _ = RS.project(np.array([[0.0, 1.0, 0.4]]), K, R, t)
    assert uv[0, 0] < 100.0                                        # the body's front on the image's left
    assert RS.camera_centre(R, t) == pytest.approx(np.array([5.0, 1.0, 0.0]))


def test_every_view_places_its_camera_by_its_name_and_frames_its_region(M):
    V, RS = M["V"], M["RS"]
    stand = M["base"].static_body("stand", 0.2)
    verts = V.vertices_at(stand, V.frame_times(stand.duration))
    W, H = 320, 180
    eyes = {}
    for name in ("front", "back", "left", "right", "top", "hands", "hands_top", "foot_left", "head"):
        view = V.VIEWS[name]
        cam = V.camera_path(view, verts, stand.joints().numpy()[:, 0], len(verts), W, H, V.heading_deg(stand))
        assert cam.fixed
        eyes[name] = RS.camera_centre(cam.R[0], cam.t[0])
        d = eyes[name] - np.asarray(cam.info["target"])                 # the camera looks down by the view's elevation
        assert math.degrees(math.asin(d[1] / np.linalg.norm(d))) == pytest.approx(view.elevation, abs=1e-6)
        pts = verts[:, V.region_mask(view.region)].reshape(-1, 3)
        uv, z = RS.project(pts, cam.K, cam.R[0], cam.t[0])
        assert (z > 0).all()
        assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W
        assert uv[:, 1].min() >= V.RESERVE[0] * H - 1e-6 and uv[:, 1].max() <= (1 - V.RESERVE[1]) * H + 1e-6
    c = stand.joints().numpy()[0, 0]
    assert eyes["front"][2] > c[2] + 1.0 and eyes["back"][2] < c[2] - 1.0
    assert eyes["left"][0] > c[0] + 1.0 and eyes["right"][0] < c[0] - 1.0
    assert eyes["top"][1] > eyes["front"][1] + 1.0
    assert eyes["foot_left"][0] > c[0]
    assert V.VIEWS["hands_top"].elevation >= 60.0 > V.VIEWS["hands"].elevation   # the close-up from above (3.6)
    assert eyes["hands_top"][1] > eyes["hands"][1]


def test_the_floor_is_the_body_own_mesh_floor(M):
    """Review of 2026-10-05: the scene floor is read from the body's mesh (the median over frames of its lowest
    vertex), not from the neutral rest mesh; a body lifted by 3 cm gets a floor 3 cm higher, and the floor report
    reads the residual of every frame."""
    from dataclasses import replace
    V = M["V"]
    stand = M["base"].static_body("stand", 0.5)
    low = V.lowest_points(stand)
    assert len(low) == V.n_frames(stand.duration)
    fl, rep = V.scene_floor(stand)
    assert fl == pytest.approx(float(V.knot_vertices(stand, [0])[0][:, 1].min()), abs=1e-9)
    assert rep["within_tol_share"] == 1.0 and abs(rep["residual_max_m"]) < 1e-9
    tr = stand.transl.clone()
    tr[:, 1] += 0.03
    lifted = replace(stand.copy(), transl=tr)
    fl2, rep2 = V.scene_floor(lifted)
    assert fl2 - fl == pytest.approx(0.03, abs=1e-9)
    assert abs(fl2 - V.mesh_floor_y(0.0)) > 0.025                       # the version-1 rule would leave it floating
    r = V.floor_report(np.r_[low, low[-1] + 0.02], fl)                  # one frame 2 cm up: seen by the report
    assert r["residual_max_m"] == pytest.approx(0.02, abs=1e-9) and r["within_tol_share"] < 1.0


def test_the_mesh_grounding_puts_a_floating_reference_on_its_floor(M):
    """exam.ground_by_mesh on a procedural walk lifted by a slow 2 cm wave: the lowest skinned sole point comes back
    onto a flat neutral sole's height, rotations and the horizontal path are kept bit for bit, and the info reads
    the residual before and after."""
    from dataclasses import replace
    from mdfm.synth import contact as CT
    EX, V = M["EX"], M["V"]
    walk, _ = M["base"].walk(4.0, start_at_s=0.5)
    t = walk.times()
    tr = walk.transl.clone()
    tr[:, 1] += 0.01 + 0.01 * torch.sin(2 * math.pi * 0.4 * t)
    lifted = replace(walk.copy(), transl=tr)
    out, info = EX.ground_by_mesh(lifted)
    assert torch.equal(out.rot6d, lifted.rot6d) and torch.equal(out.transl[:, [0, 2]], lifted.transl[:, [0, 2]])
    r = CT.mesh_sole_low(out).min(1) - V.mesh_floor_y(0.0)
    assert info["residual_before"]["within_tol_share"] < 0.5 <= info["residual_after"]["within_tol_share"]
    assert info["residual_after"]["within_tol_share"] == pytest.approx(float((np.abs(r) <= CT.TOL_M).mean()))
    assert info["residual_after"]["within_tol_share"] >= 0.95 and abs(float(np.median(r))) < 0.001
    assert info["window_s"] == EX.GROUND_WINDOW_S and info["shift_max_m"] > 0.015


def test_the_track_view_follows_the_pelvis_from_the_side(M):
    V, RS = M["V"], M["RS"]
    walk, _ = M["base"].walk(6.0, start_at_s=0.5)
    t = V.frame_times(walk.duration)
    verts = V.vertices_at(walk, t[::10])
    R, tr = V.pose_at(walk, t)
    from mdfm.body.smplh import fk
    fp = fk(R, tr, walk.rest())[0][:, 0].numpy()
    cam = V.camera_path(V.VIEWS["track_left"], verts, walk.joints().numpy()[:, 0], len(t), 320, 180, 0.0, fp)
    assert not cam.fixed
    eye = np.stack([RS.camera_centre(cam.R[k], cam.t[k]) for k in range(len(t))])
    assert (eye[:, 0] > fp[:, 0] + 1.0).all()                      # on the walker's left (+x, the walk goes +z)
    assert eye[-1, 2] - eye[0, 2] > 0.5 * (fp[-1, 2] - fp[0, 2])   # it travels with the walk


# ---------------------------------------------------------------- the mesh is the body truth
def test_the_viewer_mesh_equals_the_body_at_every_frame_on_a_knot(M):
    """30 fps frames fall on the 100 Hz knots every 0.1 s: there the viewer's vertices are SMPLHBody.vertices of
    the knot pose (the frame-by-frame identity of plan SP4); between knots the pose is the Hermite interpolant."""
    from mdfm.body.hermite import hermite
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.synth.tremor import tremor
    V = M["V"]
    body, _ = tremor(M["base"].static_body("sit", 1.0), "left", grade=3, frequency_hz=7.0, seed=3)
    t = V.frame_times(body.duration)
    assert len(t) == 31
    on = [k for k in range(len(t)) if abs(t[k] * body.rate - round(t[k] * body.rate)) < 1e-9]
    assert on == list(range(0, 31, 3))
    got = V.vertices_at(body, t[on])
    ref = V.knot_vertices(body, [int(round(t[k] * body.rate)) for k in on])
    assert np.abs(got - ref).max() < 1e-9
    R, tr = V.pose_at(body, t[1:2])
    r6 = hermite(body.rot6d, torch.tensor(t[1:2]), body.rate)
    np.testing.assert_allclose(R.numpy(), rot6d_to_matrix(r6).numpy(), atol=1e-12)
    assert not np.allclose(V.vertices_at(body, t[1:2])[0], ref[0])   # the tremor moves the mesh between knots


def test_frame_counts_and_times(M):
    V = M["V"]
    assert V.n_frames(0.0) == 1 and V.n_frames(1.0) == 31 and V.n_frames(44.999) == 1350 and V.n_frames(45.0) == 1351
    t = V.frame_times(2.0, 30.0)
    np.testing.assert_allclose(t, np.arange(61) / 30.0)
    assert V.n_frames(2.0, 60.0) == 121


# ---------------------------------------------------------------- writing, labels, refusal
@pytest.mark.skipif(not HAVE_FF, reason="ffmpeg or ffprobe not on PATH")
def test_a_clip_has_its_frames_and_labels_and_stays_inside_its_root(M, tmp_path):
    V = M["V"]
    from PIL import Image
    body = M["base"].static_body("stand", 0.5)
    opts = V.RenderOptions(width=160, height=96)
    lines = ("dystonic posture, neck | TWSTRS rotation grade 2", "34.0 deg held", "E7 standing | test_clip")
    with pytest.raises(PermissionError):                            # the default root refuses a temporary folder
        V.render_view(body, V.VIEWS["front"], str(tmp_path / "x"), "test_clip", lines, opts)
    assert not os.path.exists(tmp_path / "x")
    rec = V.render_view(body, V.VIEWS["front"], str(tmp_path / "clips"), "test_clip", lines, opts,
                        allowed_root=str(tmp_path), png_frames=True)
    n = V.n_frames(body.duration)
    assert rec["n_frames"] == n == 16
    assert rec["floor_y_m"] == pytest.approx(V.scene_floor(body)[0]) and rec["burned_caption"] == list(
        V.Annotator(160, 96, lines, "front").lines)
    mp4 = tmp_path / "clips" / rec["files"]["mp4"]["file"]
    png = tmp_path / "clips" / rec["files"]["poster"]["file"]
    assert "SYNTHETIC" in mp4.name and "SYNTHETIC" in png.name
    frames, w, h, rate = V.mp4_frame_count(str(mp4))
    assert (frames, w, h, rate) == (n, 160, 96, "30/1")
    tags = V.mp4_tags(str(mp4))
    assert "SYNTHETIC" in tags["title"] and "SYNTHETIC" in tags["comment"]
    assert Image.open(png).text["SYNTHETIC"] == "1"
    folder = tmp_path / "clips" / rec["files"]["frames"]["folder"]
    pngs = sorted(p for p in os.listdir(folder) if p.endswith(".png"))
    assert len(pngs) == n and all("SYNTHETIC" in p for p in pngs)
    ann = V.Annotator(160, 96, lines, "front")
    x0, y0, x1, y1 = ann.mark_box
    for p in pngs:                                                  # the mark is burned into every frame
        a = np.asarray(Image.open(folder / p))[y0:y1, x0:x1].reshape(-1, 3).astype(int)
        red = (a[:, 0] > 120) & (a[:, 1] < 60) & (a[:, 2] < 60)
        white = (a > 200).all(1)
        assert red.mean() > 0.4 and white.any()
    # the decoded MP4 frames carry the mark too (H.264 is lossy: colour within a tolerance)
    raw = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt",
                          "rgb24", "-"], check=True, capture_output=True).stdout
    vid = np.frombuffer(raw, np.uint8).reshape(n, 96, 160, 3)
    box = vid[:, y0 + 2:y1 - 2, x0 + 2:x0 + 6].reshape(n, -1, 3).astype(int)
    assert (box[..., 0].mean(1) > 110).all() and (box[..., 1].mean(1) < 80).all()


def test_names_without_the_label_and_folders_outside_the_root_are_refused(M, tmp_path):
    V = M["V"]
    with pytest.raises(PermissionError):
        V.check_out(str(tmp_path))
    with pytest.raises(PermissionError):
        V.check_out(os.path.join(paths.TREE, "reports", "synth", "clips"))
    with pytest.raises(PermissionError):
        V.check_out("<store>/video/x")                            # the video round trip's root, not the viewer's
    assert V.check_out("<store>/synth_viewer/exam_v1/clips") == "<store>/synth_viewer/exam_v1/clips"
    with pytest.raises(ValueError):
        V.save_png(str(tmp_path / "plain.png"), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        V.Mp4Writer(str(tmp_path / "plain.mp4"), 16, 16, 30.0, "t")
    S = _script()
    with pytest.raises(PermissionError):
        S.main(["--root", str(tmp_path), "--list"])
    assert not any(tmp_path.iterdir())


def test_the_caption_names_the_operator_and_the_grade_of_every_clip(M):
    V, EX = M["V"], M["EX"]
    specs = EX.exam_set()
    for s in specs:
        l1, l2, l3 = V.caption_lines(s.caption_spec())
        assert s.operator in l1 and s.grade_text in l1 and s.clip_id in l3 and s.segment in l3
        fam = {"tremor": "tremor", "freezing": "freezing", "bradykinesia": "bradykinesia",
               "dystonia": ("dystonic posture", "none"), "reference": "healthy reference"}[s.family]
        assert any(f in s.operator for f in ((fam,) if isinstance(fam, str) else fam))
    ann = V.Annotator(960, 540, ("a" * 400, "b", "c"), "front")    # a long line is cut to the frame
    assert ann.lines[0].endswith("...") and len(ann.lines[0]) < 400


# ---------------------------------------------------------------- the exam set and its index
def test_the_exam_set_holds_the_draft_ladders_with_their_level_zero(M):
    EX, G = M["EX"], __import__("mdfm.synth.grades", fromlist=["x"])
    specs = {s.clip_id: s for s in EX.exam_set()}
    for seg, cond in (("E1", "rest"), ("E2", "postural")):
        for tag in ("0", "1b", "1a", "2", "3", "4"):
            assert f"tremor_{seg}_{cond}_g{tag}" in specs
    assert {specs[f"tremor_E1_constancy_g{g}"].grade for g in (1, 2, 3, 4)} == {1, 2, 3, 4}
    assert all(f"freezing_E8_g{g}_trembling" in specs for g in (1, 2, 3, 4)) and "freezing_E8_g2_akinesia" in specs
    for task in ("ps", "tt", "la"):
        seg = {"ps": "E4", "tt": "E5", "la": "E6"}[task]
        assert f"brady_{seg}_{task}_level0" in specs and f"brady_{seg}_{task}_arrest_g3" in specs
        for crit in ("decrement", "interruptions"):
            assert all(f"brady_{seg}_{task}_{crit}_g{g}" in specs for g in (1, 2, 3))
    assert all(f"dystonia_E7_neck_rotation_left_g{g}" in specs for g in range(1, len(G.TWSTRS_BANDS_DEG["rotation"]) + 1))
    assert all(f"dystonia_E7_neck_tilt_left_g{g}" in specs for g in range(1, len(G.TWSTRS_BANDS_DEG["laterocollis"]) + 1))
    for k in ("anterocollis", "retrocollis"):                          # TWSTRS antero- and retrocollis, thirds
        assert all(f"dystonia_E7_neck_{k}_g{g}" in specs for g in (1, 2, 3))
    assert all(f"dystonia_E7_trunk_lateral_left_g{g}" in specs for g in (1, 2, 3, 4))
    assert all(f"dystonia_E7_trunk_lateral_left_g3_duration{d}" in specs for d in (2, 3))
    for region, grades in (("proximal_arm_abduction", (1, 2, 3, 4)), ("distal_arm_elbow_flexion", (1, 2, 3, 4)),
                           ("proximal_leg_abduction", (1, 2)), ("distal_leg_knee_flexion", (1, 2))):
        assert all(f"dystonia_E7_{region}_g{g}" in specs for g in grades), region
    assert all(f"dystonia_E1_distal_leg_plantarflexion_ankle_g{g}" in specs for g in (1, 2, 3, 4))
    assert set(specs["brady_E4_ps_level0"].views) == {"front", "hands_top", "hands"}   # draft 3.6: front, top, close-up
    for region, direction, grades in EX.BOUNDED:                       # bounded grades are listed, never rendered
        for g in grades:
            assert f"dystonia_E7_{region}_{direction}_g{g}" not in specs
    assert {"DIS duration 1", "DIS in action", "DIS other directions", "standing ankle postures"} <= set(EX.NOT_RENDERED)
    for s in specs.values():
        if s.level0 is not None:
            assert specs[s.level0].level0 is None and specs[s.level0].segment == s.segment
        assert set(s.views) <= set(M["V"].VIEWS)
    fams = {f: [v for s in specs.values() if s.family == f for v in s.views] for f in
            ("tremor", "freezing", "bradykinesia", "dystonia")}
    for f, views in fams.items():                                   # frontal and lateral views in every family
        assert {"front", "examiner"} & set(views), f
        assert {"left", "right", "track_left", "foot_left"} & set(views), f
    assert {"hands"} <= set(fams["tremor"]) and {"hands"} <= set(fams["bradykinesia"])


def test_the_amass_references_refuse_another_split_and_a_lockbox_person(M, monkeypatch):
    """smplh_reference.amass_take (once exam.amass_take; wave 8), behind the reference flag."""
    import pandas as pd
    from mdfm.data import amass_index
    from mdfm.data.canonical import Refused
    from mdfm.synth import openbody as OB
    from mdfm.synth import smplh_reference as EX
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")

    class Fake:
        split, root, rate = "train", "fake", 100.0
        takes = pd.DataFrame({"take_id": ["X/1/a_poses.npz"], "person_id": [sorted(amass_index.LOCKBOX_PERSONS)[0]],
                              "archive": ["X"], "n_knots": [500]})
    with pytest.raises(Refused):
        EX.amass_take(Fake(), "X/1/a_poses.npz")
    Fake.split = "dev"
    with pytest.raises(Refused):
        EX.amass_take(Fake(), "X/1/a_poses.npz")


def test_the_stand_rule_takes_an_upright_still_stand_not_a_seated_or_gesturing_one(M, monkeypatch):
    """smplh_reference.select_stand (once exam.select_stand; wave 8) on a store of four procedural takes (review of
    2026-10-05: version 1 chose a seated pose): a seat, a stand bent forward, a stand with the arm raised and a quiet
    stand; only the quiet stand qualifies. A store of another split is refused. On the reference: the AMASS
    references leave with wave 7 (C8)."""
    _reference(monkeypatch)
    import pandas as pd
    from mdfm.data.canonical import Refused
    from mdfm.synth import smplh_reference as EX
    B = M["base"]
    from mdfm.synth.dystonia import dystonic_posture
    stand = B.static_body("stand", 16.0)
    bent, _ = dystonic_posture(stand, "trunk", "flexion", amplitude_grade=3, seed=1)
    raised, _ = dystonic_posture(stand, "proximal_arm", "abduction", amplitude_grade=4, seed=1)
    bodies = {"seat": B.static_body("sit", 16.0), "bent": bent, "raised": raised, "quiet": stand}

    class Fake:
        split, rate, root = "train", 100.0, "fake"
        takes = pd.DataFrame({"take_id": list(bodies), "person_id": [f"p{i}" for i in range(4)], "archive": ["X"] * 4,
                              "treadmill": [0] * 4, "n_knots": [b.T for b in bodies.values()]})
        betas = np.zeros((4, 16))
        gender = np.full(4, 2)

        def take(self, i):
            b = list(bodies.values())[i]
            return {"rot6d": b.rot6d.numpy().astype(np.float32), "transl": b.transl.numpy(),
                    "contact": np.ones((b.T, 2), np.uint8)}
    res = EX.select_stand(Fake())
    assert [r["take_id"] for r in res["ranked"]] == ["quiet"] and res["ranked"][0]["start_knot"] == 0
    assert res["screened"]["takes_with_a_window"] == 1
    Fake.split = "dev"
    with pytest.raises(Refused):
        EX.select_stand(Fake())


def test_the_retarget_keeps_rotations_and_puts_the_floor_at_zero(M, monkeypatch):
    """smplh_reference.retarget_neutral (once exam.retarget_neutral; wave 8) on a procedural walk given another shape:
    rotations bit for bit, neutral shape, joint floor at y = 0 by the body-store rule, pelvis path scaled by the
    pelvis-height ratio. On the reference: the AMASS references and their shapes leave with wave 7 (C8)."""
    _reference(monkeypatch)
    from dataclasses import replace
    from mdfm.body import frames
    from mdfm.synth import smplh_reference as EX
    walk, _ = M["base"].walk(4.0, start_at_s=0.5)
    betas = torch.zeros(16, dtype=torch.float64)
    betas[0] = 1.5
    other = replace(walk.copy(), betas=betas, gender=1)
    out, info = EX.retarget_neutral(other)
    assert torch.equal(out.rot6d, other.rot6d) and int(out.gender) == 2 and float(out.betas.abs().sum()) == 0.0
    assert float(frames.floor_height(out.joints()[None])[0]) == pytest.approx(0.0, abs=1e-12)
    p0, p1 = other.joints()[:, 0].numpy(), out.joints()[:, 0].numpy()
    d0, d1 = p0[-1, [0, 2]] - p0[0, [0, 2]], p1[-1, [0, 2]] - p1[0, [0, 2]]
    np.testing.assert_allclose(d1, info["ratio"] * d0, atol=1e-9)
    assert info["ratio"] != pytest.approx(1.0)


def test_the_index_page_is_a_local_page_of_video_tags(M):
    S = _script()
    rec = {"clip_id": "tremor_E1_rest_g3", "family": "tremor", "segment": "E1", "segment_title": "seated rest",
           "operator": "tremor, rest, left forearm", "grade_text": "MDS-UPDRS 3.17 grade 3", "params_text": "5 Hz",
           "level0": "tremor_E1_rest_g0", "folder": "clips/tremor",
           "views": [{"view": "front", "n_frames": 1351, "duration_s": 45.03, "camera": {"title": "front"},
                      "files": {"mp4": {"file": "SYNTHETIC_tremor_E1_rest_g3_front.mp4"},
                                "poster": {"file": "SYNTHETIC_tremor_E1_rest_g3_front.png"}}}],
           "truth": {"version": "sp4-v3", "params": {"frequency_hz": 5.0, "cm_pp": 5.48}, "validity": {"limits_new": 0}}}
    l0 = dict(rec, clip_id="tremor_E1_rest_g0", level0=None, views=[dict(rec["views"][0], files={
        "mp4": {"file": "SYNTHETIC_tremor_E1_rest_g0_front.mp4"}, "poster": {"file": "SYNTHETIC_tremor_E1_rest_g0_front.png"}})])
    page = S.index_html([l0, rec], "test", {"DIS distal leg knee flexion grade 3": "bounded by the planted foot"})
    assert "<script" not in page.lower() and "http" not in page.lower()
    assert page.count("<video") == 3                                 # two clips and the level 0 beside the graded one
    assert 'src="clips/tremor/SYNTHETIC_tremor_E1_rest_g3_front.mp4"' in page
    assert "SYNTHETIC" in page and "5.48" in page and "level 0" in page
    assert "no facial motion" in page and "no face." not in page and "bounded by the planted foot" in page


def test_a_clip_whose_truth_fails_a_check_is_refused_before_anything_is_written(M, tmp_path):
    S, EX = _script(), M["EX"]
    body = M["base"].static_body("stand", 0.2)
    truth = {"operator": "dystonic_posture", "params": {}, "validity": {
        "limits_new": 0, "contact": {"ok": True}, "continuity": {"ok": False, "max_step_deg": 37.6}}}
    assert S.failed_checks(truth) == ["continuity"]
    assert S.failed_checks({"validity": {"limits_new": 2, "contact": {"ok": False}}}) == ["limits_new", "contact"]
    assert S.failed_checks({"validity": None}) == [] and S.failed_checks(None) == []
    spec = EX.ClipSpec("test_jump", "dystonia", "E7", "DIS", "dystonic posture", 2, "DIS grade 2", "", ("front",),
                       None, None, lambda: (body, truth))
    with pytest.raises(ValueError, match="continuity"):
        S.render_clip(spec, str(tmp_path / "set"), M["V"].RenderOptions(width=64, height=36), [], log=lambda *a: None)
    assert not os.path.exists(tmp_path / "set")


# ---------------------------------------------------------------- task SP5: hands, the new views, exam_v3
def test_the_hand_skin_is_smplh_with_the_hands_at_rest_and_moves_only_the_hands(M, monkeypatch):
    """hands.skin with every hand joint at rest is SMPLHBody.vertices bit for bit; the viewer shows a HandBody's
    hands (at a frame on a knot the mesh is the knot's pose with its hand rotations, between knots the hand6d
    Hermite interpolant), and a fist moves no vertex of the trunk, legs or head. On the SMPL+H reference (the open
    body's skinning: tests/test_openbody.py and test_the_open_mesh_at_a_knot_is_the_knot_pose)."""
    _reference(monkeypatch)
    from mdfm.body.hermite import hermite
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.synth import hands as H
    from mdfm.synth import handtasks as HT
    from mdfm.synth import motions as MO
    V = M["V"]
    model, _ = V.mesh_model("neutral")
    stand = M["base"].static_body("stand", 0.3)
    R = stand.local_R()[:3]
    eye = torch.eye(3, dtype=torch.float64).expand(3, 30, 3, 3).clone()
    with torch.no_grad():
        a = model.vertices(R[:, 0], R[:, 1:], stand.transl[:3], stand.betas.expand(3, -1))
        b = H.skin(model, R[:, 0], R[:, 1:], stand.transl[:3], stand.betas.expand(3, -1), eye)
    assert torch.equal(a, b)
    t = V.frame_times(stand.duration)
    assert np.abs(V.vertices_at(H.hands_of(stand), t) - V.vertices_at(stand, t)).max() < 1e-9
    base = MO.posture_body("sit_hand_movements", 1.0)
    body, _ = HT.hand_task(base, "hand_movements", "left", n_cycles=2, decrement_onset=None, start_s=0.1, seed=1)
    t = V.frame_times(body.duration)
    on = [k for k in range(len(t)) if abs(t[k] * body.rate - round(t[k] * body.rate)) < 1e-9]
    got = V.vertices_at(body, t[on])
    ref = V.knot_vertices(body, [int(round(t[k] * body.rate)) for k in on])
    assert np.abs(got - ref).max() < 1e-9
    Rh = V.hands_at(body, t[1:2])
    ref_h = rot6d_to_matrix(hermite(body.hand6d, torch.tensor(t[1:2]), body.rate)).numpy()
    np.testing.assert_allclose(Rh.numpy(), ref_h, atol=1e-12)
    fist = V.knot_vertices(base, [0])[0]
    rest = V.knot_vertices(H.hands_of(M["base"].static_body("sit", 0.1)).with_rotations(base.local_R()[:1].expand(
        11, -1, -1, -1).clone()), [0])[0]
    far = np.isin(V.dominant_joint("neutral"), list(range(0, 18)))
    assert np.abs(fist[far] - rest[far]).max() < 1e-12 and np.abs(fist - rest).max() > 0.02


def test_the_hand_views_frame_the_left_hand(M):
    V, RS = M["V"], M["RS"]
    from mdfm.synth import motions as MO
    body = MO.posture_body("sit_finger_tapping", 0.2)
    verts = V.vertices_at(body, V.frame_times(body.duration))
    W, Hh = 320, 180
    hand = body.joints().numpy()[0, 20]
    for name in ("hand_left", "hand_left_medial"):
        view = V.VIEWS[name]
        cam = V.camera_path(view, verts, body.joints().numpy()[:, 0], len(verts), W, Hh, V.heading_deg(body))
        assert cam.fixed
        eye = RS.camera_centre(cam.R[0], cam.t[0])
        d = eye - np.asarray(cam.info["target"])
        assert math.degrees(math.asin(d[1] / np.linalg.norm(d))) == pytest.approx(view.elevation, abs=1e-6)
        pts = verts[:, V.region_mask(view.region)].reshape(-1, 3)
        uv, z = RS.project(pts, cam.K, cam.R[0], cam.t[0])
        assert (z > 0).all() and uv[:, 0].min() >= 0 and uv[:, 0].max() <= W
        assert uv[:, 1].min() >= V.RESERVE[0] * Hh - 1e-6 and uv[:, 1].max() <= (1 - V.RESERVE[1]) * Hh + 1e-6
        if name == "hand_left_medial":
            assert eye[0] < hand[0] and eye[2] > hand[2]                  # in front, on the body's midline side
        else:
            assert eye[2] > hand[2] + 0.2
    assert set(V.REGIONS["hand_left"]) == {20} | set(range(22, 37))


def test_the_exam_set_holds_the_exam_v3_motions(M):
    """exam_v3 (task SP5): finger tapping and hand movements with their level 0, decrement, interruption, arrest
    and slowing ladders; the raised 3.15 posture with level 0, grade 2 and re-emergent tremor; the arising; each
    since 'exam_v3' or, for the clips whose body changed in the review of 2026-10-05 (the hand tasks and the
    arising), 'exam_v4'; the two raise tremors, whose body changed in the shared wave of 2026-11-02, 'exam_v5'
    with the other exam clips of that wave (exam.BODY_CHANGED_V5); the arising, refolded in the wave SH1b (SH9),
    'exam_v6' (exam.BODY_CHANGED_V6); wave SH1d (exam_v7, exam.BODY_CHANGED_V7) moved the two tremor raise clips,
    added their level 0 tremor_E2_raise_long_rest_g0 and moved the hand-task interruption and arrest clips, whose body
    changed, so exam_v3 holds tremor_E2_raise_g0 alone, exam_v4 the 14 other hand clips and exam_v5 the 6 clips of
    BODY_CHANGED_V5 outside exam_v7 and exam_v8; the settling of 2026-11-10 (exam_v8, exam.BODY_CHANGED_V8) moved the
    three raise clips again (their rest back to exam.REST_BEFORE_RAISE_S) and wave SH1d part C (exam_v9,
    exam.BODY_CHANGED_V9) the freezing clips (the standing background on their holds); views in VIEWS, level 0 in its
    segment; the
    not-rendered list no longer names the finger items or the arising; the arising chair stands under the seated
    buttocks."""
    EX, V = M["EX"], M["V"]
    from mdfm.synth import motions as MO
    specs = {s.clip_id: s for s in EX.exam_set()}
    # open-body port, wave 7 (exam_v10, exam.BODY_CHANGED_V10): the references and the freezing clips (those of
    # exam_v9 among them) leave AMASS, and the freezing clips' walk is a new level 0
    v10 = {k for k, s in specs.items() if s.since == "exam_v10"}
    v9 = {k for k, s in specs.items() if s.since == "exam_v9"}
    v8 = {k for k, s in specs.items() if s.since == "exam_v8"}
    v7 = {k for k, s in specs.items() if s.since == "exam_v7"}
    assert v10 == set(EX.BODY_CHANGED_V10) and set(EX.BODY_CHANGED_V9) <= v10 and not v9
    assert v8 == set(EX.BODY_CHANGED_V8)
    assert v7 == set(EX.BODY_CHANGED_V7) - v8 - v10
    v5 = {k for k, s in specs.items() if s.since == "exam_v5"}
    assert v5 == set(EX.BODY_CHANGED_V5) - v7 - v8 and len(v5) == 6
    raises = {"tremor_E2_raise_long_rest_g0", "tremor_E2_raise_postural_g2", "tremor_E2_raise_reemergent_g2"}
    assert raises == v8 and raises <= set(EX.BODY_CHANGED_V7)
    v6 = {k for k, s in specs.items() if s.since == "exam_v6"}
    assert v6 == set(EX.BODY_CHANGED_V6) == {"ref_arise_E7"} | {
        f"dystonia_E7_shoulder_elevation_elevation_g{g}" for g in (1, 2, 3)}
    assert [k for k, s in specs.items() if s.since == "exam_v3"] == ["tremor_E2_raise_g0"]
    new = {k: s for k, s in specs.items() if k.startswith(("brady_E4a_", "brady_E4b_")) or k in raises
           or k in ("ref_arise_E7", "tremor_E2_raise_g0")}
    v4 = {k for k, s in specs.items() if s.since == "exam_v4"}
    assert v4 == {k for k in new if k.startswith(("brady_E4a_", "brady_E4b_"))} - v7 and len(v4) == 14
    for seg, short in (("E4a", "ft"), ("E4b", "hm")):
        ids = [f"brady_{seg}_{short}_level0", f"brady_{seg}_{short}_arrest_g3"]
        ids += [f"brady_{seg}_{short}_{c}_g{g}" for c in ("decrement", "interruptions", "slowing") for g in (1, 2, 3)]
        assert set(ids) <= set(new)
        assert {"front", "hand_left", "hand_left_medial"} == set(new[ids[0]].views)
    assert len(new) == 27
    assert {s.since for s in new.values()} == {"exam_v3", "exam_v4", "exam_v6", "exam_v7", "exam_v8"}
    assert all(s.since in ("exam_v2", "exam_v5", "exam_v6", "exam_v7", "exam_v10") for k, s in specs.items()
               if k not in new)
    for s in new.values():
        assert set(s.views) <= set(V.VIEWS) and s.segment in EX.SEGMENTS
        if s.level0 is not None:
            assert specs[s.level0].level0 is None and specs[s.level0].segment == s.segment
    assert "3.4, 3.5" not in EX.NOT_RENDERED and "E7 arising" not in EX.NOT_RENDERED
    assert {"E3", "E9", "3.9 grades", "slowing of 3.6 to 3.8"} <= set(EX.NOT_RENDERED)
    seated = MO.posture_body("sit_arms_folded", 0.1)
    fl = float(V.knot_vertices(seated, [0])[0][:, 1].min())
    cv, cf = EX.scene_mesh("chair_arise", seated, fl)
    low = float(V.knot_vertices(seated, [0])[0][MO._seat_vertices(), 1].min())
    assert cv[:, 1].min() == pytest.approx(fl) and cv[:8, 1].max() == pytest.approx(low - 0.005)
    assert EX.scene_mesh(None, seated, fl) is None


def test_render_clip_draws_a_supine_clip_on_the_room_floor_from_the_couch_heading(M, tmp_path, monkeypatch):
    """Shared wave of 2026-11-02 (SH7, SARA 8): a clip on the 'exam_bed' scene is rendered on the floor and from the
    heading of ataxia.bed_view (the room floor, not the couch top under its lowest vertex; heading 0, not the root's
    at the middle knot), its couch standing on that floor; a truth failing the face-touch check of the touching tics
    is refused like the other checks."""
    from mdfm.synth import ataxia as A
    EX, V = M["EX"], M["V"]
    S = _script()
    body = A.supine_body(0.2)
    truth = {"operator": "heel_shin", "params": {}, "validity": {}}
    spec = EX.ClipSpec("test_supine", "ataxia", "SARA8", "heel-shin", "heel-shin, test", 0, "level 0", "", ("left",),
                       None, "exam_bed", lambda: (body, truth), since="catalogue_v1", segment_title="heel-shin")
    seen = {}

    def rv(body_, view, out_dir, stem, caption, opts, **k):
        seen.update(k, view=view.name)
        return {"view": view.name}
    monkeypatch.setattr(V, "render_view", rv)
    rec = S.render_clip(spec, str(tmp_path / "set"), V.RenderOptions(width=96, height=54), [], log=lambda *a: None)
    bv = A.bed_view(body)
    assert seen["view"] == "left" and seen["heading"] == bv["heading_deg"] == 0.0
    assert seen["floor_y"] == bv["floor_y"] == rec["floor"]["floor_y_m"]
    assert seen["floor_y"] < V.scene_floor(body)[0] - 0.5 and "bed_view" in rec["floor"]["rule"]
    assert np.array_equal(seen["scene"][0], A.scene_mesh("exam_bed", body, bv["floor_y"])[0])
    assert S.failed_checks({"validity": {"face_touch": {"ok": False}}}) == ["face_touch"]
    for name in ("table", "backrest", "hand_trunk"):                          # the postures' checks, wave SH1c
        assert S.failed_checks({"validity": {name: {"ok": False}}}) == [name]
    assert S.failed_checks({"validity": {"face_touch": {"ok": True}, "contact": {"ok": True}}}) == []


def test_failed_checks_refuses_the_sh1_and_sh2_checks():
    """SH2 and SH1 (2026-11-02): the posture dystonias' scene checks, the shoulder coupling of the SH2 table and the
    physiological background's verdict refuse a clip; an entry without 'ok' (tremors.FaceRoom.check) is a report."""
    S = _script()
    v = {"limits_new": 0, "armrest": {"ok": False}, "hand_thigh": {"ok": True}, "hand_head": {"min_clearance_cm": 1.0},
         "coupling": {"left": {}, "right": {}, "ok": False}, "background": {"ok": False, "failed": ["accel_excess"]}}
    assert S.failed_checks({"validity": v}) == ["armrest", "coupling", "background"]
    assert S.failed_checks({"validity": dict(v, hand_head={"ok": False})}) == ["armrest", "hand_head", "coupling",
                                                                               "background"]
    # SH1b: the folded arms of the arising (motions.arise_validity 'fold', its margins a report) and the seated knees
    assert S.failed_checks({"validity": {"fold": {"ok": False, "margins_ok": True}, "knee_clearance": {"ok": False}}}) \
        == ["fold", "knee_clearance"]
    assert S.failed_checks({"validity": {"fold": {"ok": True, "margins_ok": False}}}) == []


def test_the_index_links_the_earlier_set_and_refuses_a_hand_limit_failure(M, tmp_path):
    """--with-earlier: the clips of the earlier set appear on the new set's index by relative links (their folder
    under ../<set>/), named as from that set, and are only read; a truth failing the hand joint limits is refused."""
    import json
    S = _script()
    rec = {"clip_id": "tremor_E1_rest_g3", "family": "tremor", "segment": "E1", "segment_title": "seated rest",
           "operator": "tremor, rest, left forearm", "grade_text": "MDS-UPDRS 3.17 grade 3", "params_text": "5 Hz",
           "level0": None, "folder": "clips/tremor",
           "views": [{"view": "front", "n_frames": 1351, "duration_s": 45.03, "camera": {"title": "front"},
                      "files": {"mp4": {"file": "SYNTHETIC_tremor_E1_rest_g3_front.mp4"},
                                "poster": {"file": "SYNTHETIC_tremor_E1_rest_g3_front.png"}}}],
           "truth": {"version": "sp4-v3", "params": {}, "validity": {}}}
    for name, r in (("old", rec), ("new", dict(rec, clip_id="brady_E4a_ft_level0", family="bradykinesia",
                                                    folder="clips/bradykinesia"))):
        os.makedirs(tmp_path / name / "truth")
        with open(tmp_path / name / "truth" / f"SYNTHETIC_{r['clip_id']}_truth.json", "w", encoding="utf-8") as f:
            json.dump(r, f)
    own = S.load_records(str(tmp_path / "new"))
    old = S.earlier_records(str(tmp_path / "new"), "old", {r["clip_id"] for r in own})
    assert [r["folder"] for r in old] == ["../old/clips/tremor"] and old[0]["earlier"] == "old"
    assert S.earlier_records(str(tmp_path / "new"), "old", {"tremor_E1_rest_g3"}) == []
    page = S.index_html(own + old, "new", {})
    assert 'src="../old/clips/tremor/SYNTHETIC_tremor_E1_rest_g3_front.mp4"' in page and "from old" in page
    assert "<script" not in page.lower()
    bad = {"validity": {"limits_new": 0, "hand_limits": {"ok": False, "violations": 1}}}
    assert S.failed_checks(bad) == ["hand_limits"]
    assert S.failed_checks({"validity": {"limits_new": 0, "balance": {"ok": False}}}) == ["balance"]
    many = S.earlier_records(str(tmp_path / "new"), ["old", "new"], set())      # the first set takes precedence
    assert sorted((r["clip_id"], r["earlier"]) for r in many) == [("brady_E4a_ft_level0", "new"),
                                                                  ("tremor_E1_rest_g3", "old")]


def test_a_run_plans_only_its_own_set_and_a_closed_set_is_never_rewritten(M, tmp_path, capsys):
    """Review of 2026-10-05: without --since the clips of the --set version are planned (exam_v2 its 60 since
    the shared wave of 2026-11-02 moved 7 of its 93 to exam_v5 and wave SH1d 26 to exam_v7, not the later
    ones), a set name that is no exam version needs --since; write_index refuses a set folder whose manifest exists
    and a run record that exists, before
    writing anything."""
    import json
    S, EX = _script(), M["EX"]
    specs = EX.exam_set()
    for name in ("exam_v2", "exam_v3", "exam_v4", "exam_v5", "exam_v6", "exam_v7", "exam_v8", "exam_v10"):
        S.main(["--set", name, "--list"])
        n = int(capsys.readouterr().out.strip().splitlines()[-1].split()[0])
        assert n == sum(1 for s in specs if s.since == name)
    # exam_v9's clips (the freezing clips) all changed again in exam_v10 (exam.BODY_CHANGED_V10 holds
    # BODY_CHANGED_V9): exam_v9 holds no spec now, so its name is no exam version and needs --since (wave 8, D153)
    assert not any(s.since == "exam_v9" for s in specs)
    with pytest.raises(SystemExit):
        S.main(["--set", "exam_v9", "--list"])
    # 55 since exam_v10 (open-body port, wave 7): ref_stand_E7 and ref_walk_E8 left exam_v2 (exam.BODY_CHANGED_V10);
    # 45 since plan phase E: the eight leg agility clips and the plantarflexed ankles of DIS 3 and 4 changed inside
    # exam_v10, of which no set was rendered (their seated hands kept off the open body's thighs)
    assert sum(1 for s in specs if s.since == "exam_v2") == 45
    assert sum(1 for s in specs if s.since == "exam_v5") == 6
    with pytest.raises(SystemExit):
        S.main(["--set", "dev_check_x", "--list"])
    S.main(["--set", "dev_check_x", "--since", "exam_v4", "--list"])
    assert int(capsys.readouterr().out.strip().splitlines()[-1].split()[0]) == 14     # the hand clips left in exam_v4
    out = tmp_path / "set"
    os.makedirs(out / "truth")
    with open(S.manifest_path(str(out)), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with pytest.raises(SystemExit):
        S.write_index(str(out), "set", [], specs, record_dir=None)
    assert sorted(os.listdir(out)) == sorted(["truth", os.path.basename(S.manifest_path(str(out)))])
    out2, rec = tmp_path / "set2", tmp_path / "records"
    os.makedirs(out2 / "truth")
    os.makedirs(rec)
    with open(rec / "set2.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    with pytest.raises(SystemExit):
        S.write_index(str(out2), "set2", [], specs, record_dir=str(rec))
    assert os.listdir(out2) == ["truth"] and os.listdir(rec) == ["set2.json"]


def test_the_open_mesh_at_a_knot_is_the_knot_pose_and_a_fist_moves_only_the_hand(M):
    """On the open body (the default): at a frame on a knot the viewer's mesh is the backend's skinning of the knot's
    pose with its hand rotations; a hand rotation moves only the vertices whose dominant joint is that hand's or
    the wrist's, never the trunk, the legs or the head."""
    from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
    from mdfm.synth import hands as H
    from mdfm.synth.body import model_name
    V = M["V"]
    assert model_name() == "open"
    stand = M["base"].static_body("stand", 0.3)
    hb = H.hands_of(stand)
    R = torch.eye(3, dtype=torch.float64).expand(hb.T, 30, 3, 3).clone()
    R[:, :15] = axis_angle_to_matrix(torch.tensor([0.0, 0.0, -0.9], dtype=torch.float64))   # the left hand
    hb = H.HandBody(hb.rot6d, hb.transl, hb.contact, hb.betas, hb.gender, hb.rate, matrix_to_rot6d(R))
    t = V.frame_times(hb.duration)
    on = [k for k in range(len(t)) if abs(t[k] * hb.rate - round(t[k] * hb.rate)) < 1e-9]
    got = V.vertices_at(hb, t[on])
    ref = V.knot_vertices(hb, [int(round(t[k] * hb.rate)) for k in on])
    assert np.abs(got - ref).max() < 1e-9
    rest = V.knot_vertices(stand, [0])[0]
    moved = np.linalg.norm(V.knot_vertices(hb, [0])[0] - rest, axis=1) > 1e-9
    dom = V.dominant_joint()
    assert moved.any() and set(np.unique(dom[moved])) <= {18, 20} | set(range(22, 37))
