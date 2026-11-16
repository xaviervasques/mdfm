"""Tests of the shared scene and viewer changes asked by the clinical review of the symptom catalogue (2026-11-02,
docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md: SH5, SH11, SH12, SH14), in mdfm/synth/viewer.py, motions.arise_chair,
the scene fields of mdfm/synth/involuntary.py and scripts/render_mesh_video.py.

- chair geometry (SH5, SH1c): the armless chair is the chair's seat and backrest, box for box; the chair with arms
  (fit 'sh5', the default) keeps its armrests' tops 1 cm under the proximal forearms; fit 'wrist' (wave SH1c, built
  and off) keeps the SH5 armrests' width, place and back end, runs them on to CHAIR_ARM_PAST_WRIST_M in front of the
  sit's wrists and lays their tops CHAIR_REST_GAP_M under the lowest arm vertex above them; no vertex of the seated
  postures but the forearms and hands lies inside the chair under any fit: the backrest stands CHAIR_BACK_CLEAR_M
  behind the rearmost trunk vertex of CHAIR_POSTURES and the armrests (or the side panels of fit 'forearm')
  CHAIR_ARM_CLEAR_M outside the widest pelvis or thigh (the version before held the pelvis inside both armrests and
  the spine inside the backrest, measured here);
- the forearm over the arm of the chair to the wrist (fit 'wrist', off; SYNTH_SHARED_CHANGES_DRAFT.md section 5: the
  E1 rest posture holds the distal forearm in the air in front of the armrest's end): in the sit each wrist joint
  lies CHAIR_ARM_PAST_WRIST_M behind the armrest's front end (the SH5 armrest ends more than 10 cm in front of it),
  the hand past it, the lowest arm vertex over it CHAIR_REST_GAP_M above its top, the forearm's underside over it
  between CHAIR_REST_GAP_M and 27 mm above the top per 2 cm from the elbow to the wrist (the forearm tapers, its axis
  horizontal), two fifths of the forearm over the top, no seated vertex inside any box, the trunk and thighs
  CHAIR_ARM_CLEAR_M from the armrests;
- the chair gate (wave SH1c, lane chair): the seated clips whose forearms rest on the armrests (the exam's E1 rest
  level 0 and a tremor grade, the jerk and dystonia level 0s) keep every arm vertex out of the default armrests and
  no validity check refuses them; the clips whose arms move below the sit's (the 3.14 ladder, hands in the lap; the
  trunk-flexion jerks; the tonic tics; the Holmes tremor) keep every arm vertex within involuntary.GUARD_TOL_M of the
  default armrests, no check refuses them, and they pass more than it into the armrests of fit 'wrist', which is why
  that fit is off;
- the armrest tops fitted to the forearms (fit 'forearm', built on 2026-11-03, off by session decision): in the sit
  each forearm's contact patch lies wholly over its top, the lowest arm vertex CHAIR_ARM_GAP_M above it and the
  underside within 9 mm of it along the armrest, no seated vertex inside the chair, the trunk and thighs
  CHAIR_ARM_CLEAR_M from the tops; the SH5 armrests hold a tenth of the patch, at least 13 mm under it (measured
  here);
  exam.scene_mesh draws 'chair' (the default fit) and 'chair_armless' from it; the seated involuntary clips draw the
  armless chair and their contact guard reads it, the level-0 arms lie clear of the armrests of fits 'sh5' and
  'forearm' (their hands on the thighs, the forearms cross the front part of the default armrests), and a seated
  clip built with the armrests of fit 'sh5' or 'forearm' put back into the guard is the same body; the arising
  chair's backrest stands back_clear_m behind the rearmost trunk vertex over every seated knot (the version before
  stood inside the spine, measured here), its seat meets the backrest and its arms did not move, and no pelvis,
  trunk, thigh, collar or upper arm vertex of the healthy and the failed-attempt arising enters any of its boxes;
- views (SH11): the six new close-ups put their camera at their azimuth and elevation from the body's heading and
  frame their region inside the window left by the caption bands; the earlier views are unchanged;
- per-frame texts (SH12): the truth's frame_text spans and stimulus marks are read, active on [from, to) (a span
  reaching the clip's end holds the last frame), burned into the top band at their frames only, and a clip without
  them renders exactly as before: with truth=None and with a truth without texts the frames are the same, and on
  the machine and library versions of REF_PLATFORM they are pixel for pixel those of the viewer at a768ba4
  (REF_DIGEST, a fixed two-box scene so that the chair's own changes do not enter it);
- the view from behind (foot_back) is refused on a clip with a scene;
- frame rate (SH14): render_mesh_video.fps_rule renders a clip whose truth frequency exceeds a third of 30 fps at the
  smallest multiple of 30 fps giving three frames per cycle (the orthostatic tremor of tremors.HZ at 60 fps) and
  records why: at 30 fps it would fold below the Nyquist rate, and a one-frame exposure would keep most of it.
Bodies are procedural, in the neutral shape; no lockbox, EVAL, CODY or care_pd record is read. Clips are written
under pytest's temporary folder with the allowed root moved there.
"""
import hashlib
import math
import os
import shutil

import numpy as np
import pytest
import torch

from mdfm import paths

HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
HAVE_FF = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not HAVE_MA, reason="MoveAtlas SMPL+H model not found")
torch.set_num_threads(2)

# SHA-256 of the frames (uint8 RGB, in order) of _moving_sit rendered by render_view from the 'front' view at
# 160 x 96 with the fixed scene of _fixed_scene (two boxes with literal corners, so that the chair's SH5 changes do
# not enter it): measured with the viewer of a768ba4 (before SH5, SH11, SH12, loaded from git show) by the shared
# wave's check script on 2026-11-02, equal to the frames of this viewer with truth=None and with a truth without
# texts (an earlier check found a catalogue clip without texts, tremor_E1_head_pitch_g3, 451 frames, pixel for pixel
# equal and its MP4 byte for byte). The text rasterising (Pillow's default font) and the shading depend on the
# platform and the libraries: the digest is asserted on REF_PLATFORM only (system, Pillow, numpy).
REF_DIGEST = "d39c45130ae96c5e54c34290ce4240d8f14221c0ea79341f72984f68d816b719"
REF_PLATFORM = ("Windows", "12.3.0", "2.4.6")


def _on_ref_platform():
    import platform
    import PIL
    return (platform.system(), PIL.__version__, np.__version__) == REF_PLATFORM


@pytest.fixture(scope="module")
def V():
    from mdfm.synth import viewer
    return viewer


@pytest.fixture(scope="module")
def RS():
    from mdfm.synth import raster
    return raster


def _script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_mesh_video_script",
                                                  os.path.join(paths.TREE, "scripts", "render_mesh_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _moving_sit(duration_s=0.5):
    """The static sit with the head turning 0.4 rad to the left over the clip (a body whose frames differ)."""
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.synth.base import static_body
    b = static_body("sit", duration_s)
    R = b.local_R().clone()
    a = torch.linspace(0.0, 0.4, b.T, dtype=R.dtype)
    R[:, 15] = R[:, 15] @ axis_angle_to_matrix(torch.stack([0 * a, a, 0 * a], -1))
    return b.with_rotations(R, [15])


def _frames(folder):
    """(N, H, W, 3) uint8 of the PNG frames of a render_view frames folder, in frame order."""
    from PIL import Image
    names = sorted(p for p in os.listdir(folder) if p.endswith(".png") and not p.startswith("._"))
    return np.stack([np.asarray(Image.open(os.path.join(folder, p)).convert("RGB")) for p in names])


def _fixed_scene(viewer, fl):
    """(vertices, faces) of a seat block and a backrest with literal corners (m), standing on fl."""
    v0, f0 = viewer.box((-0.25, fl, -0.21), (0.25, 0.35, 0.27))
    v1, f1 = viewer.box((-0.25, 0.35, -0.26), (0.25, 0.90, -0.21))
    return np.concatenate([v0, v1]), np.concatenate([f0, f1 + 8])


def render_digest(viewer, out_dir, allowed_root, **kw):
    """(digest, frames) of _moving_sit rendered by viewer.render_view (front, 160 x 96, the scene of _fixed_scene at
    the clip's floor, every frame as PNG); kw go to render_view (truth)."""
    body = _moving_sit()
    fl, _ = viewer.scene_floor(body)
    lines = ("none (sit) | level 0", "the static sit, the head turning", "E1 seated rest | test_unchanged")
    rec = viewer.render_view(body, viewer.VIEWS["front"], out_dir, "test_unchanged", lines,
                             viewer.RenderOptions(width=160, height=96), scene=_fixed_scene(viewer, fl),
                             allowed_root=allowed_root, png_frames=True, floor_y=fl, **kw)
    fr = _frames(os.path.join(out_dir, rec["files"]["frames"]["folder"]))
    return hashlib.sha256(np.ascontiguousarray(fr).tobytes()).hexdigest(), fr, rec


# ---------------------------------------------------------------- SH5: the chairs
def test_the_armless_chair_is_the_chair_without_its_arms(V):
    from mdfm.synth import exam as EX
    from mdfm.synth.base import static_body
    fl = V.mesh_floor_y()
    c, f = V.chair_mesh(fl)
    c0, f0 = V.chair_mesh(fl, armrests=False)
    c4, f4 = V.chair_mesh(fl, fit="forearm")
    cw, fw = V.chair_mesh(fl, fit="wrist")
    c5, f5 = V.chair_mesh(fl, fit="sh5")
    assert V.CHAIR_FIT == "sh5" and np.array_equal(c, c5) and np.array_equal(f, f5)     # the SH5 chair kept
    assert set(V.CHAIR_FITS) == {"forearm", "sh5", "wrist"}
    assert c5.shape == cw.shape == (32, 3) and c0.shape == (16, 3) and f5.shape == (48, 3) and f0.shape == (24, 3)
    assert np.array_equal(f5, fw) and c4.shape == (48, 3) and f4.shape == (72, 3)
    for cc, ff in ((c5, f5), (cw, fw), (c4, f4), (V.chair_mesh(fl, False, "forearm")),
                   (V.chair_mesh(fl, False, "wrist"))):                         # seat and backrest, box for box
        assert np.array_equal(c0, cc[:16]) and np.array_equal(f0, ff[:24])
    assert np.array_equal(V.chair_mesh()[0], V.chair_mesh(None, True)[0])
    with pytest.raises(ValueError):
        V.chair_mesh(fl, fit="other")
    sit = static_body("sit", 0.01)                                  # the SH5 armrests: 1 cm under the proximal
    Vs = V.knot_vertices(sit, [0])[0]                               # third of the forearms
    dom, J = V.dominant_joint("neutral"), sit.joints().numpy()[0]
    bw, b5 = cw.reshape(-1, 8, 3), c5.reshape(-1, 8, 3)
    poses = [Vs, V.knot_vertices(static_body("sit_arms_forward", 0.01), [0])[0]]
    assert V.CHAIR_POSTURES == ("sit", "sit_arms_forward")
    for k, (e, w) in zip((2, 3), ((18, 20), (19, 21))):
        fore = np.isin(dom, (e,)) & (Vs[:, 2] < J[e, 2] + (J[w, 2] - J[e, 2]) / 3.0)
        top = b5[k][:, 1].max()
        assert top == pytest.approx(float(Vs[fore, 1].min()) - 0.01, abs=1e-12)
        # fit 'wrist': the SH5 armrest's width, place and back end, run on to CHAIR_ARM_PAST_WRIST_M in front of the
        # sit's wrist, its top CHAIR_REST_GAP_M under the lowest arm vertex over it in CHAIR_POSTURES
        lo, hi = bw[k].min(0), bw[k].max(0)
        assert np.array_equal(lo, b5[k].min(0)) and hi[0] == b5[k][:, 0].max()
        assert hi[2] == pytest.approx(J[w, 2] + V.CHAIR_ARM_PAST_WRIST_M, abs=1e-12)
        # 16 cm longer at least on SMPL+H; on the open body 15.4 cm (Anny's wrist 13.4 cm in front of the SH5 end, its
        # forearm 23.4 cm long): re-derived from the open-body reading (wave 8 second part, D156)
        assert hi[2] > b5[k][:, 2].max() + 0.15
        over = [P[V._arm_ids(e) & (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2])]
                for P in poses]
        assert hi[1] == pytest.approx(min(float(p[:, 1].min()) for p in over if len(p)) - V.CHAIR_REST_GAP_M,
                                      abs=1e-12)
        # raised 6 to 9 mm over the SH5 top on SMPL+H; on the open body 4.0 mm (the SH5 top 1 cm under Anny's
        # proximal forearm, whose underside lies 6 mm over the 'wrist' top along the armrest): D156
        assert 0.003 < hi[1] - top < 0.005
    for kind, ref in (("chair_armless", (c0, f0)), ("chair", (c, f))):           # the scenes drawn by the viewer
        sv, sf = EX.scene_mesh(kind, None, fl)
        assert np.array_equal(sv, ref[0]) and np.array_equal(sf, ref[1])


def test_the_exam_chair_holds_no_part_of_the_seated_body_but_the_forearms(V):
    """SH5 acceptance on the seated postures: no vertex but the forearms' and hands' (dominant joint below 18) inside
    a box of the chair, under any fit; the backrest and the armrests (the panels of fit 'forearm') at their
    clearances; the chair before SH5 (rebuilt here from its formula) held the pelvis inside both armrests and the
    spine inside the backrest."""
    from mdfm.synth import motions as MO
    from mdfm.synth.base import POSTURES, static_body
    fl = V.mesh_floor_y()
    bx = V.chair_mesh(fl, fit="forearm")[0].reshape(-1, 8, 3)
    b5 = V.chair_mesh(fl)[0].reshape(-1, 8, 3)                                     # the default, fit 'sh5'
    bw = V.chair_mesh(fl, fit="wrist")[0].reshape(-1, 8, 3)                        # the armrests to the wrist
    assert np.array_equal(bx[:2], b5[:2]) and np.array_equal(bw[:2], b5[:2])
    dom = V.dominant_joint("neutral")
    keep, seatv, trunk = dom < 18, np.isin(dom, (0, 1, 2)), np.isin(dom, V.CHAIR_BACK_JOINTS)
    seat_top, x_half, front = float(bx[0][:, 1].max()), float(bx[0][:, 0].max()), float(bx[1][:, 2].max())
    assert V.CHAIR_POSTURES[0] == "sit" and set(V.CHAIR_POSTURES) == {p for p in POSTURES if p.startswith("sit")}

    def depth(P, B):                                     # > 0: inside the box, the depth to its nearest face
        return np.minimum(P - B.min(0), B.max(0) - P).min(-1)
    bodies = {n: static_body(n, 0.01) for n in V.CHAIR_POSTURES}
    bodies.update({n: MO.posture_body(n, 0.01) for n in ("sit_finger_tapping", "sit_hand_movements",
                                                         "sit_arms_folded")})
    rear, wide = math.inf, 0.0
    for name, b in bodies.items():
        P = V.knot_vertices(b, [0])[0]
        for k, B in enumerate(list(b5) + list(bx[2:]) + list(bw[2:])):
            assert depth(P[keep], B).max() <= 0.0, (name, k)
        if name in V.CHAIR_POSTURES:
            ins = trunk & (P[:, 1] > seat_top) & (P[:, 1] < seat_top + 0.55) & (np.abs(P[:, 0]) < x_half)
            rear, wide = min(rear, float(P[ins, 2].min())), max(wide, float(np.abs(P[seatv, 0]).max()))
    assert rear - front == pytest.approx(V.CHAIR_BACK_CLEAR_M, abs=1e-12)          # the backrest 5 mm behind
    assert bx[0][:, 2].min() == pytest.approx(front, abs=1e-12)                    # the seat starts at it
    sit = bodies["sit"]
    Ps, J = V.knot_vertices(sit, [0])[0], sit.joints().numpy()[0]
    for k, e, B in ((4, 18, bx), (5, 19, bx), (2, 18, b5), (3, 19, b5), (2, 18, bw), (3, 19, bw)):
        inner = float(np.abs(B[k][:, 0]).min())                     # the panels; the SH5 and 'wrist' armrests
        assert inner == pytest.approx(max(abs(J[e, 0]) - 0.04, wide + V.CHAIR_ARM_CLEAR_M), abs=1e-12)
        assert inner >= wide + V.CHAIR_ARM_CLEAR_M - 1e-12 and np.sign(B[k][:, 0].mean()) == np.sign(J[e, 0])
        assert B[k][:, 1].min() == pytest.approx(fl)
        if B is not bx:
            assert float(np.ptp(B[k][:, 0])) == pytest.approx(0.08)
        else:                       # the panel under its top, flush with its outer face, at least CHAIR_ARM_POST_M
            top = bx[k - 2]
            assert float(np.abs(B[k][:, 0]).max()) == pytest.approx(float(np.abs(top[:, 0]).max()), abs=1e-12)
            assert float(np.ptp(B[k][:, 0])) >= V.CHAIR_ARM_POST_M - 1e-12
            assert B[k][:, 1].max() == pytest.approx(top[:, 1].min(), abs=1e-12)
            assert np.allclose(B[k][:, 2].min(), top[:, 2].min()) and np.allclose(B[k][:, 2].max(), top[:, 2].max())
    old_front = float(Ps[seatv, 2].min()) - 0.02                                    # the chair before SH5
    ins = trunk & (Ps[:, 1] > seat_top) & (Ps[:, 1] < seat_top + 0.55) & (np.abs(Ps[:, 0]) < x_half)
    # on SMPL+H the chair before SH5 held the spine 0.5 to 1 cm inside its backrest and the pelvis over 2 cm inside
    # both armrests, the reason for SH5; on the open body its formula leaves the backrest 1.73 cm behind Anny's spine
    # and the pelvis 2.2 mm outside the armrests (the rest's narrower pelvis and the sit's trunk further forward):
    # the open-body readings, the SMPL+H ones the record of why SH5 was built (wave 8 second part, D156)
    assert -0.02 < old_front - float(Ps[ins, 2].min()) < -0.015
    for k, e in ((2, 18), (3, 19)):
        old = np.array([[J[e, 0] - 0.04, seat_top, old_front - 0.05], [J[e, 0] + 0.04, b5[k][:, 1].max(),
                                                                        J[e, 2] + 0.10]])
        assert -0.003 < depth(Ps[np.isin(dom, (0,))], old).max() < 0.0


def _armrest_contact(V, P, J, B, e):
    """(share, gap m, profile m) of the sit's forearm of elbow e on the box B: the share of its contact patch (the
    forearm vertices within CHAIR_ARM_PATCH_M of the lowest one between the elbow and 10 cm in front of it) whose x
    and z lie over B; the lowest vertex of the side's arm over B minus B's top; per 2 cm from the elbow to B's front
    end, the lowest forearm vertex over B minus its top."""
    dom = V.dominant_joint("neutral")
    lo, hi = B.min(0), B.max(0)
    ov = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2])
    along = (dom == e) & (P[:, 2] >= J[e, 2]) & (P[:, 2] <= J[e, 2] + 0.10)
    patch = along & (P[:, 1] <= P[along, 1].min() + V.CHAIR_ARM_PATCH_M)
    prof = [float(P[s, 1].min() - hi[1]) for z in np.arange(J[e, 2], hi[2] - 1e-9, 0.02)
            for s in [(dom == e) & ov & (P[:, 2] >= z) & (P[:, 2] < z + 0.02)] if s.any()]
    return float(ov[patch].mean()), float(P[V._arm_ids(e) & ov, 1].min() - hi[1]), np.array(prof)


def test_the_resting_forearms_lie_on_the_armrest_tops(V):
    """Fit 'forearm' (2026-11-03, off by session decision; handoff 13, section 1: the forearms of the exam sit rest on
    the armrest's inner edge): in the sit each forearm's contact patch lies wholly over its top, the lowest arm vertex
    over the top CHAIR_ARM_GAP_M above it (the clearance tremor.CLEAR_GAP_CM), the forearm's underside within 9 mm of
    the top over every 2 cm of the armrest in front of the elbow; no vertex of the seated postures, the forearms
    included, lies inside any box; the pelvis, trunk and thighs keep CHAIR_ARM_CLEAR_M from the tops (the trunk beside
    them, the thighs under them). The SH5 chair (fit 'sh5', the default, measured here): a tenth of the patch over the
    armrest, the forearm at least 13 mm above it."""
    from mdfm.synth import tremor as T
    from mdfm.synth.base import static_body
    assert V.CHAIR_ARM_GAP_M == pytest.approx(T.CLEAR_GAP_CM / 100.0)
    fl = V.mesh_floor_y()
    sit = static_body("sit", 0.01)
    P, J = V.knot_vertices(sit, [0])[0], sit.joints().numpy()[0]
    dom = V.dominant_joint("neutral")
    bx, b5 = (V.chair_mesh(fl, fit=f)[0].reshape(-1, 8, 3) for f in ("forearm", "sh5"))
    rest = ~np.isin(dom, tuple(range(13, 52)))
    for k, e in ((2, 18), (3, 19)):
        share, gap, prof = _armrest_contact(V, P, J, bx[k], e)
        assert share == 1.0 and gap == pytest.approx(V.CHAIR_ARM_GAP_M, abs=1e-9), (e, share, gap)
        # the underside within 9 mm of the top on SMPL+H; on the open body 9.16 mm at its fourth 2 cm (Anny's
        # forearm tapers more over the fitted top): re-derived to 9.5 mm (wave 8 second part, D156)
        assert len(prof) == 5 and prof.min() >= V.CHAIR_ARM_GAP_M - 1e-9 and prof.max() < 0.0095, prof
        share, gap, prof = _armrest_contact(V, P, J, b5[k], e)
        # SMPL+H: a tenth of the patch over the SH5 armrest, at least 13 mm under it; the open body: 0.87 of the patch
        # over it and the forearm 10 mm above it, the SH5 rule's own gap (its top 1 cm under the proximal forearm) (D156)
        assert 0.8 < share < 0.9 and gap == pytest.approx(0.01, abs=1e-9) and prof.min() > 0.0099, (e, share, gap)
    for name in V.CHAIR_POSTURES:
        Q = V.knot_vertices(static_body(name, 0.01), [0])[0]
        for k, B in enumerate(bx):
            out = np.maximum(B.min(0) - Q, Q - B.max(0)).max(1)               # > 0: outside the box
            assert out.min() > 0.0, (name, k)
            if k in (2, 3):
                d = np.linalg.norm(np.maximum(0.0, np.maximum(B.min(0) - Q[rest], Q[rest] - B.max(0))), axis=1)
                assert d.min() >= V.CHAIR_ARM_CLEAR_M - 1e-9, (name, k, d.min())


def test_the_resting_forearm_lies_on_the_armrest_to_the_wrist(V):
    """Fit 'wrist' (wave SH1c, built and off; SYNTH_SHARED_CHANGES_DRAFT.md section 5: the E1 rest posture holds
    the distal forearm in the air in front of the armrest's end): the forearm over the arm of the chair to the wrist,
    the hand past its front end, the forearm itself above the top, not on it. In the sit each wrist joint lies
    CHAIR_ARM_PAST_WRIST_M behind its armrest's front end, where the SH5 armrest ends more than 10 cm in front of it;
    the lowest arm vertex over the armrest lies CHAIR_REST_GAP_M above its top; per
    2 cm from the elbow to the wrist the forearm's underside over the armrest lies between CHAIR_REST_GAP_M and 27 mm
    above the top (CHAIR_REST_GAP_M at the lowest vertex, opening under the wrist: the forearm tapers and its axis is
    horizontal), and the wrist's and hand's vertices over it 15 to 21 mm above it; two fifths of the forearm's
    vertices from the elbow to the wrist lie over the top (its medial part beside the inner face, which stays outside
    the hips); no vertex of the seated postures lies inside a box; the pelvis, trunk and thighs keep
    CHAIR_ARM_CLEAR_M from the armrests."""
    from mdfm.synth.base import static_body
    fl = V.mesh_floor_y()
    sit = static_body("sit", 0.01)
    P, J = V.knot_vertices(sit, [0])[0], sit.joints().numpy()[0]
    dom = V.dominant_joint("neutral")
    bw, b5 = (V.chair_mesh(fl, fit=f)[0].reshape(-1, 8, 3) for f in ("wrist", "sh5"))
    rest = ~np.isin(dom, tuple(range(13, 52)))
    for k, e in ((2, 18), (3, 19)):
        w = e + 2
        lo, hi = bw[k].min(0), bw[k].max(0)
        assert hi[2] - J[w, 2] == pytest.approx(V.CHAIR_ARM_PAST_WRIST_M, abs=1e-12)
        assert J[w, 2] - b5[k][:, 2].max() > 0.10, (e, b5[k][:, 2].max(), J[w, 2])
        ov = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2])
        assert float(P[V._arm_ids(e) & ov, 1].min() - hi[1]) == pytest.approx(V.CHAIR_REST_GAP_M, abs=1e-9)
        prof = np.array([float(P[s, 1].min() - hi[1]) for z in np.arange(J[e, 2], J[w, 2] - 1e-9, 0.02)
                         for s in [(dom == e) & ov & (P[:, 2] >= z) & (P[:, 2] < z + 0.02)] if s.any()])
        # one reading per 2 cm of the forearm: 13 on SMPL+H, 12 on the open body (Anny's forearm 23.4 cm) (D156)
        assert len(prof) == math.ceil((J[w, 2] - J[e, 2] - 1e-9) / 0.02) == 12 and prof.max() < 0.027, prof
        assert prof.min() == pytest.approx(V.CHAIR_REST_GAP_M, abs=1e-9)
        hand = np.isin(dom, (w,) + tuple(range(22, 52))) & ov
        assert hand.sum() > 10 and 0.015 < float(P[hand, 1].min() - hi[1]) < 0.021
        along = (dom == e) & (P[:, 2] >= J[e, 2]) & (P[:, 2] <= J[w, 2])
        # two fifths of the forearm over the top on SMPL+H (0.4 to 0.45); 0.70 on the open body, Anny's forearm lying
        # further over the armrest (D156)
        assert 0.65 < float(ov[along].mean()) < 0.75, float(ov[along].mean())
    for name in V.CHAIR_POSTURES:
        Q = V.knot_vertices(static_body(name, 0.01), [0])[0]
        for k, B in enumerate(bw):
            out = np.maximum(B.min(0) - Q, Q - B.max(0)).max(1)               # > 0: outside the box
            assert out.min() > 0.0, (name, k)
            if k in (2, 3):
                d = np.linalg.norm(np.maximum(0.0, np.maximum(B.min(0) - Q[rest], Q[rest] - B.max(0))), axis=1)
                assert d.min() >= V.CHAIR_ARM_CLEAR_M - 1e-9, (name, k, d.min())


# The chair gate (wave SH1c, lane chair), raw catalogue builds on the chair with arms. CHAIR_REST_CLIPS: clips whose
# forearms rest on the armrests (the exam's E1 rest level 0 and a tremor grade whose centre clears the chair's boxes
# through tremor.chair_boxes, the jerk and dystonia level 0s). CHAIR_GATE_CLIPS: clips whose arms move below the sit's
# (the 3.14 ladder's level 0, hands in the lap; the trunk-flexion jerks: a tic, a propriospinal myoclonus, a tonic
# tic; a Holmes tremor), the deepest of each family into the armrests of fit 'wrist'.
CHAIR_REST_CLIPS = ("tremor_E1_rest_g0", "tremor_E1_rest_g3", "jerk_E1_level0", "dystonia_E1_level0")
CHAIR_GATE_CLIPS = ("park_E1_spontaneity_g0", "jerk_tics_E1_number_n3", "jerk_myoclonus_E1_propriospinal_f1_a4",
                    "jerk_tonic_tics_E1_f1_i4", "tremor_E1_holmes_g3")
CHAIR_GATE_CARRIED = ("jerk_tics_E1_number_n3", "jerk_myoclonus_E1_propriospinal_f1_a4", "jerk_tonic_tics_E1_f1_i4")
#   wave SH1d part B (review M6): the trunk jerks' arms carried over the armrests no longer drop into the 'wrist' fit
#   (the tic 2.96 mm and the propriospinal jerk 3.52 mm clear of it); the other gate clips still show why it is off


@pytest.fixture(scope="module")
def chair_gate(V):
    """{clip: {'depth': {fit: m}, 'refused': [check], 'wrist_z': (2,) m, 'T': knots}} of CHAIR_REST_CLIPS and
    CHAIR_GATE_CLIPS built raw under the default chair: per fit ('sh5', 'wrist'), the deepest arm vertex (dominant
    joint the shoulder, the elbow, the wrist or a finger joint) inside an armrest of that fit, to its nearest face
    (> 0 inside), over every second knot; the validity checks the build refuses; the wrist joints' z at knot 0."""
    from mdfm.synth import catalogue as C
    specs = {s.clip_id: s for s in C.catalogue_set(background=False)}
    fl = V.mesh_floor_y()
    boxes = {f: V.chair_mesh(fl, fit=f)[0].reshape(-1, 8, 3)[2:4] for f in ("sh5", "wrist")}
    arms = (V._arm_ids(18), V._arm_ids(19))
    out = {}
    for cid in CHAIR_REST_CLIPS + CHAIR_GATE_CLIPS:
        s = specs[cid]
        assert s.scene == "chair", cid
        body, truth = s.build()
        ks = np.arange(0, body.T, 2)
        Q = np.concatenate([V.knot_vertices(body, ks[i:i + 200]) for i in range(0, len(ks), 200)])
        depth = {f: max(float(np.minimum(Q[:, a] - B.min(0), B.max(0) - Q[:, a]).min(-1).max())
                        for a, B in zip(arms, bx)) for f, bx in boxes.items()}
        bad = [k for k, x in (truth.get("validity") or {}).items() if isinstance(x, dict) and x.get("ok") is False]
        out[cid] = {"depth": depth, "refused": bad, "wrist_z": body.joints().numpy()[0, [20, 21], 2], "T": body.T}
    return out


def test_the_resting_forearms_stay_out_of_the_default_armrests(V, chair_gate):
    """The chair gate on CHAIR_REST_CLIPS under the default chair (fit 'sh5'): no arm vertex inside an armrest at any
    second knot, no validity check refused, and at knot 0 each wrist joint in front of the default armrest's end, in
    the air, and behind the front end of the armrests of fit 'wrist' (the point left to the rest tremor's rebuild)."""
    assert V.CHAIR_FIT == "sh5"
    fl = V.mesh_floor_y()
    ends = {f: V.chair_mesh(fl, fit=f)[0].reshape(-1, 8, 3)[2:4][:, :, 2].max(1) for f in ("sh5", "wrist")}
    for cid in CHAIR_REST_CLIPS:
        g = chair_gate[cid]
        assert g["depth"]["sh5"] < 0.0, (cid, g["depth"])
        assert not g["refused"], (cid, g["refused"])
        assert np.all(g["wrist_z"] > ends["sh5"] + 0.10) and np.all(g["wrist_z"] < ends["wrist"]), (cid, g["wrist_z"])


def test_the_default_armrests_keep_the_seated_catalogue_arms_out(V, chair_gate):
    """CHAIR_GATE_CLIPS (arms that move below the sit's) under the default chair: every arm vertex within
    involuntary.GUARD_TOL_M of its armrests and no validity check refused; on SMPL+H the same bodies passed more
    than GUARD_TOL_M into the armrests of fit 'wrist', which is why that fit is off; on the open body none does."""
    from mdfm.synth import involuntary as INV
    assert V.CHAIR_FIT == "sh5"
    for cid in CHAIR_GATE_CLIPS:
        g = chair_gate[cid]
        assert g["depth"]["sh5"] <= INV.GUARD_TOL_M and not g["refused"], (cid, g)
    # wave SH1d part B: the trunk jerks' arms are carried over the armrests (jerks.arms_rule) and the Holmes tremor is
    # levelled at the hand (a smaller forearm swing), so these no longer reach into the 'wrist' fit; the 3.14 level 0
    # still does, which is why that fit stays off
    # on the open body none of them passes GUARD_TOL_M into the 'wrist' fit (Anny's seated arms lie higher over the
    # armrests): the fit's reason to stay off is the SMPL+H reading, the fit kept off by the session decision of
    # 2026-11-03 and not rebuilt on the open body (wave 8 second part, D156)
    deep = [cid for cid in CHAIR_GATE_CLIPS if chair_gate[cid]["depth"]["wrist"] > INV.GUARD_TOL_M]
    assert deep == [], deep


def test_the_seated_involuntary_clips_draw_the_armless_chair_and_keep_their_motion(monkeypatch):
    from mdfm.synth import involuntary as I
    from mdfm.synth import viewer as V
    specs = I.specs()
    seated = [s for s in specs if s.segment == "E1"]
    # 60 seated clips of the shared wave and 6 of the gap wave (peak-dose dystonic dyskinesia, thigh rubbing)
    assert len(seated) == 66 and {s.scene for s in seated} == {"chair_armless"}
    assert {s.scene for s in specs if s.segment in ("E2", "E4a")} == {"chair"}       # arms forward, tapping
    assert {s.scene for s in specs if s.segment == "E7"} == {None}
    assert I.CARRIAGE["sit"][1] == I.SCENE["sit"] == "chair_armless"   # the contact guard reads the drawn chair
    b = I.base_body("sit", 0.1)                               # the AIMS level 0: arms clear of the armrests of fits
    fl, _ = V.scene_floor(b, 10.0)                            # 'sh5' and 'forearm' (the armless chair is drawn)
    Vk = V.knot_vertices(b, [0])[0]
    probes = I._guard_geometry()[0]
    for fit in ("sh5", "forearm"):
        boxes = V.chair_mesh(fl, fit=fit)[0].reshape(-1, 8, 3)[2:]
        for side in ("left", "right"):
            p = Vk[probes[side]]
            for bx in boxes:
                gap = np.maximum(bx.min(0) - p, p - bx.max(0)).max(1)   # > 0: outside the box along some axis
                # clear by a centimetre (HEAD_CLEAR_M): 2 cm read wave 3's rest; the open-body port's wave 3 follow-up
                # rest (involuntary.THIGH_SHARE, D43) offsets the shoulders 9.5 deg outward (3.9 before), the arm
                # under 2 cm of the fit 'sh5' armrest, still clear (wave SH1d: the palms on the thighs, 0.04 until then)
                assert gap.min() > I.HEAD_CLEAR_M, (fit, side, gap.min())
    a, _ = I.chorea("sit", 4, regions=("arms",), seconds=8.0, seed=5)     # the guard never meets the armrests
    orig = I._boxes
    assert len(orig("chair_armless", a)) == 2 and len(orig("chair", a)) == 4
    monkeypatch.setattr(I, "_boxes", lambda scene, body: orig("chair" if scene == "chair_armless" else scene, body))
    for fit, n in (("sh5", 4), ("forearm", 6)):                           # the armrests put back into the guard
        monkeypatch.setattr(V, "CHAIR_FIT", fit)
        assert len(I._boxes("chair_armless", a)) == n                     # fit 'forearm': the tops and panels
        b2, _ = I.chorea("sit", 4, regions=("arms",), seconds=8.0, seed=5)
        assert torch.equal(a.rot6d, b2.rot6d) and torch.equal(a.transl, b2.transl), fit
        if getattr(a, "hand6d", None) is not None:
            assert torch.equal(a.hand6d, b2.hand6d)


@pytest.fixture(scope="module")
def arising():
    from mdfm.synth import motions as MO
    b, _ = MO.arise(standing_s=0.5)
    return b


def test_the_arising_backrest_stands_behind_the_trunk(V, arising):
    from mdfm.synth import hands as H
    from mdfm.synth import motions as MO
    b = arising
    fl = V.mesh_floor_y(0.0)
    cv, cf = MO.arise_chair(b, fl)
    bx = cv.reshape(-1, 8, 3)
    assert len(bx) == 8 and cf.shape == (96, 3)
    seat, back = bx[0], bx[1]
    ids = MO._seat_vertices()
    V0 = H.body_points(b, ids, knots=[0])[0]
    seat_top = float(V0[:, 1].min()) - 0.005
    assert seat[:, 1].max() == pytest.approx(seat_top, abs=1e-12)                   # the seat height unchanged
    front_z = float(back[:, 2].max())
    assert seat[:, 2].min() == pytest.approx(front_z, abs=1e-12)                    # the seat meets the backrest
    assert back[:, 1].min() == pytest.approx(seat_top) and back[:, 1].max() == pytest.approx(seat_top + 0.55)
    low = H.body_points(b, ids)[..., 1].min(1)                                       # the seated knots
    seated = np.flatnonzero(low <= low[0] + 0.01)
    assert 0 in seated and len(seated) < b.T
    trunk = np.flatnonzero(np.isin(H._rest()[2], (0, 3, 6, 9)))
    P = H.body_points(b, trunk, knots=seated)
    x_half = float(seat[:, 0].max())
    inside = (P[..., 1] > seat_top) & (P[..., 1] < seat_top + 0.55) & (np.abs(P[..., 0]) < x_half)
    zmin = float(np.where(inside, P[..., 2], np.inf).min())
    assert zmin - front_z == pytest.approx(0.005, abs=1e-12)                        # 5 mm clear at the closest knot
    old_front = float(V0[:, 2].min()) - 0.02                                         # the backrest before SH5
    # was 0.5 to 2 cm in the spine on SMPL+H (SH9's fold moved the back); on the open body the backrest before SH5
    # stands 2.0 cm behind the trunk's rearmost seated vertex (wave 8 second part, D156)
    assert -0.025 < old_front - zmin < -0.015
    for k in (6, 7):                                                                 # the arms did not move
        assert bx[k][:, 2].min() == pytest.approx(float(V0[:, 2].min()) - 0.02 - 0.05, abs=1e-12)
        assert bx[k][:, 1].max() == pytest.approx(seat_top + 0.22) and bx[k][:, 2].max() == seat[:, 2].max()
    assert {float(bx[6][:, 0].min()), float(bx[7][:, 0].max())} == {x_half, -x_half}
    for k in range(2, 6):                                                            # the legs under the seat
        assert bx[k][:, 1].min() == pytest.approx(fl) and bx[k][:, 1].max() == pytest.approx(seat_top - 0.05)


@pytest.mark.parametrize("clip_id", ["ref_arise_E7", "park_E7_arise_g1_attempts"])
def test_no_trunk_thigh_or_upper_arm_enters_the_arising_chair(V, clip_id):
    """Independent of arise_chair's own selection: at every 5th knot (and the last) of the healthy arising and of the
    failed attempts that sit back, no vertex of the pelvis, thighs, spine, collars, neck or upper arms (dominant
    joints 0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17) lies more than 2 mm inside any box of the drawn chair (the hands
    and forearms push on the armrests: parkinsonism.armrest_targets)."""
    from mdfm.synth import exam as EX
    from mdfm.synth import parkinsonism as PK
    spec = {s.clip_id: s for s in EX.exam_set() + PK.specs()}[clip_id]
    assert spec.scene == "chair_arise"
    body, _ = spec.build()
    fl, _ = V.scene_floor(body)
    cv, _ = EX.scene_mesh(spec.scene, body, fl)
    ks = np.unique(np.r_[np.arange(0, body.T, 5), body.T - 1])
    Vk = V.knot_vertices(body, ks)[:, np.isin(V.dominant_joint("neutral"), (0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17))]
    for k, B in enumerate(cv.reshape(-1, 8, 3)):
        d = np.minimum(Vk - B.min(0), B.max(0) - Vk).min(-1)
        assert d.max() <= 0.002, (clip_id, k, float(d.max()))


# ---------------------------------------------------------------- SH11: the views
OLD_VIEWS = {   # the views before SH11, field for field (name, azimuth, elevation, vfov, region, kind, margin, title)
    "front": (0.0, 8.0, 30.0, "body", "fixed", 0.08, "front"),
    "back": (180.0, 8.0, 30.0, "body", "fixed", 0.08, "back"),
    "left": (90.0, 6.0, 30.0, "body", "fixed", 0.08, "left side"),
    "right": (-90.0, 6.0, 30.0, "body", "fixed", 0.08, "right side"),
    "top": (0.0, 62.0, 30.0, "body", "fixed", 0.08, "from above, in front"),
    "hands": (0.0, 32.0, 26.0, "hands", "fixed", 0.10, "close-up, both forearms and hands"),
    "hands_top": (0.0, 72.0, 26.0, "hands", "fixed", 0.10, "close-up from above, both forearms and hands"),
    "foot_left": (90.0, 8.0, 26.0, "foot_left", "fixed", 0.10, "close-up, left foot, side"),
    "head": (0.0, 6.0, 26.0, "head", "fixed", 0.12, "close-up, head and neck"),
    "track_left": (0.0, 5.0, 34.0, "body", "track", 0.06, "side, following the walk"),
    "examiner": (180.0, 10.0, 30.0, "body", "fixed", 0.04, "examiner, behind the start of the walk"),
    "hand_left": (0.0, 10.0, 24.0, "hand_left", "fixed", 0.12, "close-up, left hand, front"),
    "hand_left_medial": (-60.0, 15.0, 24.0, "hand_left", "fixed", 0.12,
                         "close-up, left hand, from the front and the midline"),
}
NEW_VIEWS = {"hand_left_side": (90.0, 10.0, 24.0, "hand_left"), "front_left": (45.0, 8.0, 30.0, "body"),
             "head_left": (90.0, 6.0, 26.0, "head"), "foot_left_medial": (-20.0, 15.0, 26.0, "foot_left_close"),
             "foot_back": (180.0, 15.0, 26.0, "feet"), "toes_left": (45.0, 20.0, 12.0, "toes_left")}


def test_the_earlier_views_are_unchanged(V):
    for name, (az, el, fov, region, kind, margin, title) in OLD_VIEWS.items():
        v = V.VIEWS[name]
        assert (v.name, v.azimuth, v.elevation, v.vfov_deg, v.region, v.kind, v.margin, v.title) == \
               (name, az, el, fov, region, kind, margin, title)
    assert set(V.VIEWS) == set(OLD_VIEWS) | set(NEW_VIEWS)
    assert V.REGIONS["foot_left"] == (4, 7, 10) and V.REGIONS["head"] == (9, 12, 13, 14, 15)


def test_the_new_views_place_their_camera_and_frame_their_region(V, RS):
    from mdfm.synth import motions as MO
    from mdfm.synth.base import static_body
    stand = static_body("stand", 0.2)
    tap = MO.posture_body("sit_finger_tapping", 0.2)
    W, Hh = 320, 180
    for name, (az, el, fov, region) in NEW_VIEWS.items():
        view = V.VIEWS[name]
        assert (view.azimuth, view.elevation, view.vfov_deg, view.region, view.kind) == (az, el, fov, region, "fixed")
        assert view.title and view.title.isascii()
        body = tap if name == "hand_left_side" else stand
        verts = V.vertices_at(body, V.frame_times(body.duration))
        head = V.heading_deg(body)
        cam = V.camera_path(view, verts, body.joints().numpy()[:, 0], len(verts), W, Hh, head)
        assert cam.fixed
        eye = RS.camera_centre(cam.R[0], cam.t[0])
        d = eye - np.asarray(cam.info["target"])
        assert math.degrees(math.asin(d[1] / np.linalg.norm(d))) == pytest.approx(el, abs=1e-6)
        got = math.degrees(math.atan2(d[0], d[2])) - head                     # azimuth from the body's heading
        assert (got - az + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-6)
        mask = V.region_mask(region)
        assert 0 < mask.sum() <= len(mask) and (region == "body") == (mask.sum() == len(mask))
        pts = verts[:, mask].reshape(-1, 3)
        uv, z = RS.project(pts, cam.K, cam.R[0], cam.t[0])
        assert (z > 0).all() and uv[:, 0].min() >= 0 and uv[:, 0].max() <= W
        assert uv[:, 1].min() >= V.RESERVE[0] * Hh - 1e-6 and uv[:, 1].max() <= (1 - V.RESERVE[1]) * Hh + 1e-6
        J = body.joints().numpy()[0]
        if name == "hand_left_side":
            assert eye[0] > J[20, 0] + 0.2                                    # beside the left hand, outside
        elif name == "head_left":
            assert eye[0] > J[15, 0] + 0.3
        elif name == "foot_left_medial":
            assert eye[0] < J[10, 0] and eye[2] > J[10, 2] + 0.3              # in front, on the midline side
        elif name == "foot_back":
            assert eye[2] < min(J[7, 2], J[8, 2]) - 0.3
        elif name == "toes_left":
            assert eye[0] > J[10, 0] and eye[2] > J[10, 2]
    assert set(V.REGIONS["toes_left"]) == {10} and set(V.REGIONS["feet"]) == {7, 8, 10, 11}


# ---------------------------------------------------------------- SH12: per-frame texts
def test_the_truth_texts_are_read_and_active_on_their_spans(V):
    from mdfm.synth import ataxia as A
    from mdfm.synth import jerks as JK
    truth = {"params": {"frame_text": [dict(x) for x in A.ROMBERG_EYES],
                        "stimulus_marks": JK.stimulus_marks([5.0], ["SOUND"])}}
    tx = V.frame_texts(truth)
    assert [(x["text"], x["kind"]) for x in tx] == [("EYES OPEN", "state"), ("EYES CLOSED", "state"),
                                                    ("SOUND", "stimulus")]
    assert tx[2]["to_s"] - tx[2]["from_s"] == pytest.approx(JK.STIMULUS_MARK_S)
    assert V.frame_texts(None) == [] and V.frame_texts({"params": {}}) == [] and V.frame_texts({}) == []
    end = A.ROMBERG_S
    names = lambda t, e=end: [x["text"] for x in V.active_texts(tx, t, e)]       # noqa: E731
    assert names(A.ROMBERG_CLOSE_S - 1.0 / 30.0) == ["EYES OPEN"]
    assert names(A.ROMBERG_CLOSE_S) == ["EYES CLOSED"]                         # the boundary frame: the new state
    assert names(end) == ["EYES CLOSED"] and names(end, None) == []            # the last frame holds the last span
    assert names(5.0) == ["EYES OPEN", "SOUND"]
    for fps, n in ((30.0, 6), (60.0, 12)):                                     # a 0.2 s mark on 6 (12) frames
        t = V.frame_times(10.0, fps)
        assert sum("SOUND" in names(x) for x in t) == n


def test_the_texts_are_burned_into_the_top_band_at_their_frames_only(V):
    from PIL import Image, ImageDraw
    W, Hh = 640, 360
    lines = ("operator | grade", "parameters", "segment | clip")
    texts = [{"from_s": 1.0, "to_s": 2.0, "text": "EYES CLOSED", "kind": "state"},
             {"from_s": 1.5, "to_s": 1.7, "text": "SOUND", "kind": "stimulus"}]
    plain = V.Annotator(W, Hh, lines, "front")
    ann = V.Annotator(W, Hh, lines, "front", texts, 3.0)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (Hh, W, 3), dtype=np.uint8)
    assert np.array_equal(ann(frame, 0.5), plain(frame, 0.5))                  # nothing active: the same pixels
    assert np.array_equal(ann(frame, 2.5), plain(frame, 2.5))
    a, p = ann(frame, 1.6), plain(frame, 1.6)
    diff = np.abs(a.astype(int) - p).sum(-1) > 0
    rows = np.flatnonzero(diff.any(1))
    assert rows.size and rows.max() < ann.draw_rows.stop                       # inside the top band
    x0, y0, x1, y1 = ann.mark_box
    assert not diff[y0:y1, x0:x1].any()                                        # the SYNTHETIC mark untouched
    boxes = ann._boxes(ImageDraw.Draw(Image.new("RGB", (W, Hh))), V.active_texts(texts, 1.6, 3.0))
    assert [b[4] for b in boxes] == ["EYES CLOSED", "SOUND"]
    for x0, y0, x1, y1, text, kind in boxes:
        assert tuple(a[y0 + 1, x0 + 1]) == V.TEXT_STYLE[kind][0]               # each box in its kind's colour
        assert x0 >= ann.lab_span[0] and x1 <= ann.lab_span[1]                 # between the note and the title
        assert y0 == ann.mark_box[1]                                           # on the mark's row
    narrow = V.Annotator(400, 300, lines, "close-up, left hand, from the front and the midline", texts, 3.0)
    f2 = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    d2 = np.abs(narrow(f2, 1.6).astype(int) - V.Annotator(400, 300, lines, "close-up, left hand, from the front and "
                                                                  "the midline")(f2, 1.6)).sum(-1) > 0
    assert d2.any() and np.flatnonzero(d2.any(1)).min() >= narrow.mark_box[3]  # too narrow: the row below the mark


@pytest.mark.skipif(not HAVE_FF, reason="ffmpeg or ffprobe not on PATH")
def test_a_clip_without_texts_renders_as_before_and_one_with_texts_only_adds_them(V, tmp_path, monkeypatch):
    root = str(tmp_path)
    d0, f0, r0 = render_digest(V, str(tmp_path / "none"), root)
    d1, f1, r1 = render_digest(V, str(tmp_path / "empty"), root, truth={"params": {"frequency_hz": 5.0}})
    assert d0 == d1 and np.array_equal(f0, f1) and "burned_texts" not in r0 and "burned_texts" not in r1
    if _on_ref_platform():
        # the viewer of a768ba4 drew the SMPL+H mesh only: its frames are compared on the synthetic body's SMPL+H
        # reference backend, behind the reference flag (open-body port, wave 8 second part, D156); on the open body
        # the frames without texts equal those with a truth without texts (above)
        from mdfm.synth import openbody as OB
        with monkeypatch.context() as mp:
            mp.setenv(OB.REFERENCE_ENV, "1")
            mp.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
            dr, _, _ = render_digest(V, str(tmp_path / "reference"), root)
        assert dr == REF_DIGEST                                                # pixel for pixel the a768ba4 viewer
    truth = {"params": {"frame_text": [{"from_s": 0.2, "to_s": 0.3, "text": "EYES CLOSED"}]}}
    d2, f2, r2 = render_digest(V, str(tmp_path / "texts"), root, truth=truth)
    t = V.frame_times(0.5)
    on = (t + 1e-9 >= 0.2) & (t + 1e-9 < 0.3)
    assert on.sum() == 3 and r2["burned_texts"][0]["text"] == "EYES CLOSED"
    assert np.array_equal(f2[~on], f0[~on])                                    # the other frames untouched
    diff = np.abs(f2[on].astype(int) - f0[on]).sum(-1) > 0
    stop = V.Annotator(160, 96, ("", "", ""), "front", V.frame_texts(truth), 0.5).draw_rows.stop
    assert diff.any(axis=(1, 2)).all() and not diff[:, stop:].any()          # only the top band of those frames


# ---------------------------------------------------------------- SH14: the frame rate
def test_a_tremor_above_a_third_of_the_frame_rate_is_rendered_faster():
    from mdfm.synth import tremors as TR
    S = _script()
    f = TR.HZ["orthostatic"]
    fps, rec = S.fps_rule({"params": {"frequency_hz": f}}, 30.0)
    assert fps == 60.0 and rec["frames_per_cycle"] >= S.FRAMES_PER_CYCLE > rec["frames_per_cycle_base"]
    assert rec["alias_hz_at_base"] == pytest.approx(30.0 - f) and rec["alias_hz_at_base"] < 15.0
    assert rec["box_exposure_gain_at_base"] == pytest.approx(abs(math.sin(math.pi * f / 30) / (math.pi * f / 30)))
    assert 0.5 < rec["box_exposure_gain_at_base"] < 0.7                       # a box exposure keeps most of it
    assert S.fps_rule({"params": {"frequency_hz": 10.0}}, 30.0) == (30.0, None)   # exactly a third: 30 fps
    assert S.fps_rule({"params": {"frequency_hz": 6.0}}, 30.0) == (30.0, None)
    assert S.fps_rule({"params": {}}, 30.0) == (30.0, None) and S.fps_rule(None, 30.0) == (30.0, None)
    assert S.fps_rule({"params": {"frequency_hz": 25.0}}, 30.0)[0] == 90.0
    with pytest.raises(ValueError):
        S.fps_rule({"params": {"frequency_hz": 45.0}}, 30.0)


@pytest.mark.skipif(not HAVE_FF, reason="ffmpeg or ffprobe not on PATH")
def test_render_clip_takes_the_rate_the_scene_and_the_texts_from_the_clip(V, tmp_path, monkeypatch):
    from mdfm.synth import exam as EX
    from mdfm.synth.base import static_body
    S = _script()
    body = static_body("sit", 0.3)
    truth = {"operator": "tremor_catalogue", "params": {"frequency_hz": 15.7, "frame_text": [
        {"from_s": 0.0, "to_s": 0.3, "text": "EYES OPEN"}]}, "validity": {}}
    spec = EX.ClipSpec("test_fast", "tremor", "E1", "presence", "tremor, test", 1, "present", "15.7 Hz", ("front",),
                       None, "chair_armless", lambda: (body, truth), since="catalogue_v1")
    real = V.render_view
    seen = {}

    def rv(*a, **k):
        seen["scene"] = k.get("scene")
        return real(*a, allowed_root=str(tmp_path), **k)
    monkeypatch.setattr(V, "render_view", rv)
    rec = S.render_clip(spec, str(tmp_path / "set"), V.RenderOptions(width=96, height=54), [], log=lambda *a: None)
    v = rec["views"][0]
    assert rec["options"]["fps"] == 60.0 and rec["fps_rule"]["fps"] == 60.0 and v["fps"] == 60.0
    assert v["n_frames"] == V.n_frames(body.duration, 60.0) == 19 and v["burned_texts"][0]["text"] == "EYES OPEN"
    fl = rec["floor"]["floor_y_m"]
    assert np.array_equal(seen["scene"][0], V.chair_mesh(fl, armrests=False)[0])
    frames, w, h, rate = V.mp4_frame_count(str(tmp_path / "set" / rec["folder"] / v["files"]["mp4"]["file"]))
    assert (frames, w, h, rate) == (19, 96, 54, "60/1")


def test_the_view_from_behind_is_refused_on_a_clip_with_a_scene(V, tmp_path):
    """foot_back (viewer.NO_SCENE_VIEWS) on a seated clip would show the seat block only: render_clip refuses it
    before building the clip; a clip without a scene goes on to its build."""
    from mdfm.synth import exam as EX
    S = _script()
    assert V.NO_SCENE_VIEWS == ("foot_back",) and set(V.NO_SCENE_VIEWS) <= set(V.VIEWS)

    def build():
        raise AssertionError("built")
    for scene, err in (("chair", ValueError), (None, AssertionError)):
        spec = EX.ClipSpec("test_back", "postures", "E1", "presence", "test", 1, "present", "", ("front", "foot_back"),
                           None, scene, build, since="catalogue_v1")
        with pytest.raises(err, match="foot_back" if err is ValueError else "built"):
            S.render_clip(spec, str(tmp_path / "set"), V.RenderOptions(width=96, height=54), [], log=lambda *a: None)
