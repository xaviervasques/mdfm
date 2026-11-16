"""Build the skeleton catalogue of the synthetic patient (task SKELV): every clip of the current exam set seen as a
simple 3D skeleton beside its level 0, in one viewer page a clinician opens locally.

The set: mdfm.synth.exam.exam_set(), every spec once (the clip ids are unique there, so each id is its latest
version): the exam_v2 clips, the three raise clips of exam_v3 and the exam_v4 clips (the arising, finger tapping,
hand movements), that is the clips of the exam_v4 index page (scripts/render_mesh_video.py --set exam_v4
--index-only --with-earlier exam_v3 exam_v2). A clip whose truth fails a check (render_mesh_video.failed_checks)
is refused, as there.

Per clip: its body built by the spec (spec.build()), its points on the viewer's 30 fps clock (mdfm.synth.skeleton:
the 22 body joints, the 30 hand joints, animated when the body carries hand rotations and at rest in the wrists'
frame otherwise, and the two head landmarks, so that pronation-supination and a neck rotation show; the viewer's
interpolant and forward kinematics), and its metadata (family, segment, item, operator, grade, grade text, the
parameters text of exam.describe, the caption lines of viewer.caption_lines, the level-0 partner, the views of the
spec, the body's heading, the floor, the chair boxes of a seated clip, the walk axis of a walk) into one
gzip-compressed JSON file,
data/SYNTHETIC_<clip>_skeleton.json.gz. Positions are millimetres rounded to 1 mm, delta coded over frames
(positions_mm_delta: frame 0 absolute, then each frame minus the previous one, flat in the order frame, point, axis;
the positions are the running sum). The floor of a clip is the median over its frames of its lowest body joint, minus
viewer.sole_offset_m() (the height of the lowest foot joint above the sole of the neutral rest mesh), so that it
sits where the mesh viewer's floor sits for a flat foot. Joint positions and the two head landmarks only: no vertex,
face, shape coefficient or model array is written. Since the open-body port's wave 7 (EXPORT_FORMAT
skeleton_v7_open_body) the body is Anny's (mdfm.synth.openbody: code Apache-2.0, model data CC0 1.0), the landmarks are
read on its rest mesh (landmark_offsets, skeleton.head_landmarks), and no clip is built on a motion-capture recording:
the exam set's stand, walk and freezing clips, AMASS takes until then, are made by the program's generators.

The page, SYNTHETIC_skeleton_catalogue.html beside the data: self-contained (no network, no library: a canvas with
its own perspective projection), every clip's compressed file embedded as base64 (the page opens from file://,
where a browser blocks fetch) and decoded only when the clip is selected (DecompressionStream). English labels
(the owner's decision of 2026-11-03: everything the tree produces for a reader is in English; the French pages of
tasks SKELV to SKELV4b are superseded); the SYNTHETIC mark on the page and in every canvas.

Outputs: <store>/synth_viewer/skeleton_v1/ (outside this tree; check_out refuses any other root): the data files,
the page and SYNTHETIC_skeleton_manifest.json (every file with its bytes and SHA-256, the code record); with
--copy-to, a copy of that folder (outside this tree, MoveAtlas and the gates-only store); the run record
reports/synth/viewer/skeleton_v1.json (metadata only, refused if it exists, --no-record to skip).

The symptom catalogue (--set catalogue, tasks SKELV2 and SKELV3): mdfm.synth.catalogue.catalogue_set(modules=
--modules), the exam set then each catalogue module (default DEFAULT_MODULES: the seven of
catalogue.CATALOGUE_MODULES), into <store>/synth_viewer/skeleton_v2/ and reports/synth/viewer/skeleton_v2.json. A
clip is refused when its truth fails render_mesh_video.failed_checks or its module's own validity (module_failures:
any validity entry {'ok': False}, such as the gait generator's 'gait'); the refusals go to
SYNTHETIC_skeleton_refused.json (kept for --page-only), to the manifest (with the clips planned, shown and refused by
family and the shown clips by group) and to the page's about section, never into the page's data. The page is
PAGE_TEMPLATE with CATALOGUE_PATCHES: the list's groups in clinical order (skeleton_text.GROUP_ORDER: tremors,
finger to nose, bradykinesia, the other parkinsonian signs, atypical parkinsonism, parkinsonian postures, chorea,
ballism and athetosis, dyskinesias, stereotypies, akathisia and restless legs, mirror movements, myoclonus, negative
myoclonus, tics, startle, clonus, dystonia, paroxysmal dyskinesias, ataxia, freezing of gait, gait, healthy
references), each with the exam's scale items then the catalogue's clinical headings. The structure of the list
(groups, headings, series, chips) lives in scripts/skeleton_text.py; the operator, grade and parameter texts of a
clip are its specification's own, shown as they are. --set exam (the default) builds the exam page of task SKELV
from the skeleton_v1 data, in English too.

Task SKELV3 adds to each catalogue data file the clip's per-time texts ('overlays', clip_overlays: the Romberg eye
state of params['frame_text'] and the jerks' stimuli of params['stimulus_marks'], on the clip's clock), which the
page draws in the truth's own words at their times in the clip's own view (OVERLAY_JS), and builds the scene boxes
from the connected parts of the scene mesh (scene_parts: the chair, the arising chair and the writing table of
postures.scene_mesh; a part that is not a box is left out and listed as 'scene_left_out'). The data format is
EXPORT_FORMAT, one of the stale keys: a file written by an older export is rebuilt.

After the SKELV3 reviews: the catalogue data carry drawing landmarks beyond the 54 points (CATALOGUE_LANDMARKS: per
side the heel, the lateral forefoot, the big toe tip, the thumb and index tips and the ear, points fixed in a joint's
frame read on the neutral rest mesh, landmark_offsets), which the page draws as short lines and an ear line, so that
the moving toes, the foot's inversion, the pinch and the head's roll show; the page leaves out a symptom clip whose
points equal its level 0's on every frame and a level 0 left without a shown symptom clip (hidden_clips, named in
the about section); a clip opens on the view that shows its sign (DEFAULT_VIEW, with the page's own views
CATALOGUE_VIEWS); the fit keeps the body clear of the texts at the top and bottom of a view (CATALOGUE_JS); the
segment codes of the about section come from the index (segment_list).

Provenance of the catalogue data (--set catalogue): each data file carries the code that built it ('code': the
module of its spec's build, that module file's SHA-256, its dependencies 'deps', that is the mdfm files the module
imports, transitively and lazy imports included (code_deps, a reading of the sources), with the export code
EXPORT_CODE, the SHA-256 over them, and the synth code record of render_mesh_video.code_record). A rerun treats an
existing file as done only when its module and dependency digests equal the current files' ones; otherwise it
rebuilds it ("stale, rebuilt"), so that an edit of one family rebuilds that family only. The page is refused
while a loaded file is stale (--allow-stale overrides, and the manifest lists them); a series without a label of
skeleton_text or a clip without a heading is shown under its operator or under 'Other' and listed in the manifest
('unlabelled'), not refused. The manifest gives each file its digests and calls the code record of the page build
'page_code'. A clip whose build raises is refused as 'build_error' and the run goes on (a module still being
fixed). The catalogue modules left out of --modules
(families still being written) are never imported (BlockedModules, a guard on the import system): a delivered clip
whose build calls one is not built, and is listed as such (SYNTHETIC_skeleton_refused.json 'deferred', the manifest
and the page's about section).

Task SKELV4a (the shared wave of the clinical review): the views of viewer.VIEWS after SH11 (the exam page keeps the
views of task SKELV, skeleton_text.EXAM_VIEWS, and the regions they frame, _views_regions; the catalogue page shows
every view under the viewer's own title, a region of one joint such as the toes framed with its landmarks); the scenes
after SH5 to SH7: an oriented box of the scene mesh (the examiner's finger of the E3 clips, 'chair_target') is written
as its 8 corners and drawn by its 12 edges, an axis-aligned one stays [x0, y0, z0, x1, y1, z1]; a supine clip on the
examination couch ('exam_bed', SARA 8) takes the room floor and the couch's heading of ataxia.bed_view (not its
joints' floor nor the root's heading), is marked 'supine' (no pelvis following, the shadow on the couch top) and
opens on the page's couch views (CATALOGUE_VIEWS: the legs from the side and from above, EXTRA_VIEWS); text grades
of a series in clinical order (mild before moderate); the leg-to-leg contact check of the gaits (validity 'legs') is
written into the data file and the manifest (REPORTED_ONLY); since its calibration on normal gait (2026-11-03) it
enters gait.ok, so a failing gait clip is refused through 'gait'. EXPORT_FORMAT changes with it.

Wave SH1d (the clinical review of the English page, 2026-11-03): a clip is drawn on its level 0's floor and with its
level 0's scene boxes (page_floor, through a chain of level 0s: page_roots; the page reads them from the index), and
the catalogue export reads a clip's floor over its standing frames (clip_floor 'standing': the lowest body joint's
minimum while the feet are still); the declared jerk events, slides of a dragged foot and freezing holds of a clip's
truth are exported ('events_s', 'slides_s', 'holds_s'); the uniform checks of the skeleton (skeleton_checks:
continuity, self_contact, skate, statue, pair_floor, sinking, section 'wave SH1d') run on the shown clips' exported
positions and go to the manifest, and since 2026-11-11 refuse the clips they flag (SKELETON_REPORTED_ONLY empty;
the skate check reads the sole cloud, so that a foot turning about a point of its sole passes); the marks of
provenance of the specifications' texts ('[module]', '(D4)', 'declared map', 'estimator floor' ...) leave the
clinical caption for a provenance line under it (skeleton_text.clinical_texts); a catalogue level 0 that no clip names
as its level 0 is hidden (hidden_clips), but skeleton_text.STANDALONE_LEVEL0; a check that refuses a clip also leaves
out the clips drawn beside it as their level 0 and a level 0 it leaves alone (refusal_followers).
EXPORT_FORMAT changes with it.

The public movement gallery (2026-11-10, the owner's request: the gallery of the MoveAtlas site). Provenance is
measured at export: CaptureReads records every file a clip's build opens (and every folder it lists) under a store of
motion-capture takes (capture_roots: the AMASS body store, the AMASS caches and take index, MoveAtlas's datasets, the
gates-only and window stores) and every call of the body store or of the AMASS loader, into the data file's
'provenance'; after a build that read one, the specs are built again and the lru caches of the mdfm modules emptied, so
that the next clip built on the same take reads it again (the exam set caches its walk and stand). EXPORT_FORMAT
changes with it. --public (with --set catalogue --page-only; --data-from defaults to catalogue_open) writes
movement_gallery.html and movement_gallery_manifest.json into <store>/synth_viewer/gallery_open/: the internal
page's selection (page_selection), less every clip whose build read a capture store and every clip whose level-0
chain holds one (public_provenance; the CAPTURED table only cross-checks the measure), the level 0s that leaves alone;
the data files reduced to what the page draws (public_blob: joint and landmark positions, no mesh, shape or model
array); the page's texts for a public reader (PUBLIC_PATCHES: synthetic, not reviewed by a neurologist,
under the body model's licences and credit, the motion data under CC BY 4.0 (plan phase E); no code
comments); a scan of the final page, its decoded data and the manifest
(public_scan: local paths, file names of the tree, internal task codes and version tags, and words kept as sha256
digests) that refuses the build on a hit; the manifest in the interface of MoveAtlas's site/gen_movement_gallery.py
(public_manifest) and the run record reports/synth/viewer/gallery_open.json (plan phase E: the open build, its data
from <store>/synth_viewer/catalogue_open; gallery_public and its record are the SMPL+H gallery of 2026-11-10).

CPU only, no network. Run from <repo>:
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/render_skeleton_catalogue.py [--set catalogue]
--list prints the plan; --only keeps the clips whose id holds one of the given strings; clips whose data file
exists are skipped (an interrupted run resumes); --page-only rebuilds the page and the manifest from the data files;
with --data-from (--page-only only) the data files and the refusals are read from another folder of the viewer root,
never written there, and the page, the manifest and the refusals go to --out (a preview beside the owner's folders).
"""
import argparse
import ast
import base64
import dataclasses
import datetime
import functools
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.synth import exam as EX  # noqa: E402
from mdfm.synth import skeleton as SK  # noqa: E402
from mdfm.synth import viewer as V  # noqa: E402
from mdfm.synth.catalogue import CATALOGUE_MODULES  # noqa: E402
from mdfm.video import labels as LB  # noqa: E402


def _load_mesh_script():
    """scripts/render_mesh_video.py as a module (its failed_checks and code_record)."""
    spec = importlib.util.spec_from_file_location("render_mesh_video_for_skeleton",
                                                  os.path.join(ROOT, "scripts", "render_mesh_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RMV = _load_mesh_script()

SET = "skeleton_v1"                     # the exam set (--set exam, the default)
OUT = f"{V.VIEWER_ROOT}/{SET}"
SET_CATALOGUE = "skeleton_v2"           # the symptom catalogue (--set catalogue, tasks SKELV2 and SKELV3)
OUT_CATALOGUE = f"{V.VIEWER_ROOT}/{SET_CATALOGUE}"
CATALOGUE_SINCE = "catalogue_v1"
DEFAULT_MODULES = tuple(CATALOGUE_MODULES)   # the seven catalogue modules, all delivered on 2026-11-02 (task
#   SKELV3): no family is left out by default (BlockedModules acts only on a --modules subset)
EXPORT_FORMAT = "skeleton_v7_open_body"   # the data file's format: SKELV3 adds the per-time texts ('overlays') and,
#   after its reviews, the drawing landmarks of the feet, the thumb and index tips and the ears (CATALOGUE_LANDMARKS);
#   SKELV4a the oriented scene boxes as 8 corners and the supine clips' floor, heading and mark (supine_view); wave
#   SH1d the floor over the standing frames (clip_floor 'standing'), the declared jerk events ('events_s'), the
#   declared slides of a dragged foot ('slides_s', {} on a clip without) and the freezing holds ('holds_s', on a
#   freezing clip only); the public gallery (2026-11-10) the clip's provenance ('provenance', CaptureReads: the files
#   its build opened under a motion-capture store); the open-body port's wave 7 (the design's export format 3) the
#   open body's positions and landmarks, every clip rebuilt on it. Plan phase E adds an optional key within the same
#   format: a declared touch its module reads on the posed mesh ('touches_mesh_mm', MESH_TOUCHES, on such a clip
#   only; a file without it is read as before)
DATA = "data"
PAGE = LB.synthetic_name("skeleton_catalogue", ".html")
MANIFEST = LB.synthetic_name("skeleton_manifest", ".json")
REFUSED = LB.synthetic_name("skeleton_refused", ".json")
RECORD = os.path.join(ROOT, "reports", "synth", "viewer", f"{SET}.json")


def record_path(set_name):
    return os.path.join(ROOT, "reports", "synth", "viewer", f"{set_name}.json")


FPS = V.FPS
NOTICE = ("SYNTHETIC: joint positions of a body model of the MD_FM synthetic patient (skeleton catalogue); not a "
          "recording of any person; internal research use only")
EMBED_LIMIT_BYTES = 25_000_000          # [cfg] the page embeds every clip only under this size (task SKELV: 15 MB;
#   task SKELV4b: 25 MB for the 540 clips of the gap wave). The page is a local file opened from this machine, never
#   published as it is (the public gallery is its own build, --public)


# ---------------------------------------------------------------- export
def data_name(clip_id):
    return LB.synthetic_name(f"{clip_id}_skeleton", ".json.gz")


def data_path(out, clip_id):
    return f"{out}/{DATA}/{data_name(clip_id)}"


FOOT_JOINTS = (7, 8, 10, 11)           # the ankles and the toe-block joints of mdfm.body.smplh (the feet)
FOOT_STILL_M_S = 0.05                  # [module] wave SH1d: a foot joint is still below this speed (central differences
#   on the 30 fps clock; the 1 mm rounding of the page data adds at most 0.015 m/s), a chosen bound well above that
#   rounding and well below a walking foot
MIN_STANDING_FRAMES = 15               # [module] at least 0.5 s of standing frames on the 30 fps clock, else the median
#   rule (a clip whose feet never stand still)


def standing_frames(pos, fps=FPS):
    """(N,) bool of a skeleton's frames (positions (N, P, 3) m, the body joints first) where the four foot joints
    (FOOT_JOINTS) all move slower than FOOT_STILL_M_S (central differences, the end frames take their neighbour's):
    the body stands or sits on still feet."""
    f = np.asarray(pos, dtype=np.float64)[:, list(FOOT_JOINTS)]
    n = len(f)
    v = np.zeros(n)
    if n >= 3:
        v[1:-1] = np.linalg.norm((f[2:] - f[:-2]) / 2.0, axis=-1).max(1) * fps
        v[0], v[-1] = v[1], v[-2]
    return v < FOOT_STILL_M_S


def clip_floor(skel, rule="median"):
    """{floor_y_m, rule, lowest_joint_median_m, sole_offset_m} of a skeleton (module docstring). rule 'standing'
    (the catalogue export since wave SH1d, PLAN M5): the minimum of the lowest body joint over the standing frames
    (standing_frames: the feet still), the median rule when there are fewer than MIN_STANDING_FRAMES; the record
    adds 'standing_frames' and 'lowest_joint_standing_min_m'."""
    lowest = skel.positions[:, :SK.N_BODY, 1].min(1)
    low = float(np.median(lowest))
    off = V.sole_offset_m()
    out = {"floor_y_m": low - off, "lowest_joint_median_m": low, "sole_offset_m": off,
           "rule": "median over the frames of the lowest of the 22 body joints, minus viewer.sole_offset_m()"}
    if rule == "median":
        return out
    if rule != "standing":
        raise ValueError(f"floor rule {rule!r}")
    st = standing_frames(skel.positions[:, :SK.N_BODY], skel.fps)
    out["standing_frames"] = int(st.sum())
    if st.sum() >= MIN_STANDING_FRAMES:
        m = float(lowest[st].min())
        out.update(floor_y_m=m - off, lowest_joint_standing_min_m=m,
                   rule="minimum of the lowest of the 22 body joints over the frames where the four foot joints are "
                        "still (standing_frames), minus viewer.sole_offset_m()")
    else:
        out["rule"] += " (fewer than MIN_STANDING_FRAMES standing frames)"
    return out


def event_spans(truth):
    """[[start_s, end_s]] of the declared jerk events of a clip's truth (params['events'] of jerks.py: an onset and
    parts with their delay, rise, hold and fall, jerks.event_span) and of its ballistic intervals (params
    ['ballistic_intervals_s'] of involuntary.hemiballism, whose flings are fast by design), on the clip's clock; []
    when it declares none. The continuity check (skeleton_checks) leaves these spans out (wave SH1d)."""
    params = (truth or {}).get("params") or {}
    out = [[round(float(a), 4), round(float(b), 4)] for a, b in params.get("ballistic_intervals_s") or []]
    for e in params.get("events") or []:
        parts = e.get("parts") if isinstance(e, dict) else None
        if "onset_s" not in (e if isinstance(e, dict) else {}) or not parts or not all(
                isinstance(p, dict) and "rise_s" in p and "fall_s" in p for p in parts):
            continue
        on = float(e["onset_s"])
        a = min(on + float(p.get("delay_s", 0.0)) for p in parts)
        b = max(on + float(p.get("delay_s", 0.0)) + float(p["rise_s"]) + float(p.get("hold_s", 0.0)) +
                float(p["fall_s"]) for p in parts)
        out.append([round(a, 4), round(b, 4)])
    return out


def scene_boxes(kind, body, floor_y):
    """[[x0, y0, z0, x1, y1, z1]] m of the chair of a seated clip (exam.scene_mesh: boxes of viewer.box, 8 corners
    each), [] without a scene; scene geometry only."""
    sm = EX.scene_mesh(kind, body, floor_y)
    if sm is None:
        return []
    v = np.asarray(sm[0], dtype=np.float64).reshape(-1, 8, 3)
    return [[round(float(x), 4) for x in np.r_[b.min(0), b.max(0)]] for b in v]


def _face_parts(faces, n_vertices):
    """[[vertex indices]] of the connected parts of a mesh (vertices joined by its faces), in vertex order."""
    parent = list(range(n_vertices))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for f in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        a = find(int(f[0]))
        for b in (find(int(f[1])), find(int(f[2]))):
            if a != b:
                parent[b] = a
    parts = {}
    for i in range(n_vertices):
        parts.setdefault(find(i), []).append(i)
    return sorted(parts.values(), key=lambda p: p[0])


BOX_TOL_M = 1e-6                       # [module] the corner tolerance of an oriented box (m)
BOX_TOL_COS = 1e-6                     # [module] its right angles: the cosine between two edges at most this


def oriented_box(p, tol=BOX_TOL_M):
    """True when 8 points (8, 3) in the vertex order of viewer.box are the corners of a box of any orientation: three
    edges from corner 0 (to corners 1, 3 and 4) longer than tol and at right angles (BOX_TOL_COS), every other corner
    their sum within tol (the examiner's finger of reach.scene_mesh: viewer.box turned by a rotation)."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (8, 3):
        return False
    e = [p[1] - p[0], p[3] - p[0], p[4] - p[0]]
    want = p[0] + np.array([0 * e[0], e[0], e[0] + e[1], e[1], e[2], e[0] + e[2], e[0] + e[1] + e[2], e[1] + e[2]])
    ln = [float(np.linalg.norm(x)) for x in e]
    if min(ln) <= tol or np.abs(p - want).max() > tol:
        return False
    return all(abs(float(e[i] @ e[j])) <= BOX_TOL_COS * ln[i] * ln[j] for i, j in ((0, 1), (0, 2), (1, 2)))


def scene_parts(kind, body, floor_y):
    """(boxes, left_out) of a clip's scene for the catalogue set (task SKELV3): the mesh of exam.scene_mesh (the
    writing table of postures.scene_mesh among them) split into its connected parts; a part of 8 vertices at the
    corners of an axis-aligned box gives [x0, y0, z0, x1, y1, z1] m (rounded as scene_boxes), a part that is a box of
    another orientation (oriented_box: the examiner's finger, task SKELV4a) its 8 corners [[x, y, z]] in the vertex
    order of viewer.box (rounded the same), any other part is left out and listed ({vertices, faces}); ([], [])
    without a scene."""
    sm = EX.scene_mesh(kind, body, floor_y)
    if sm is None:
        return [], []
    v = np.asarray(sm[0], dtype=np.float64)
    f = np.asarray(sm[1], dtype=np.int64).reshape(-1, 3)
    boxes, left = [], []
    for part in _face_parts(f, len(v)):
        p = v[part]
        lo, hi = p.min(0), p.max(0)
        corners = {(a, b, c) for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])}
        if len(part) == 8 and {tuple(x) for x in p} == corners:
            boxes.append([round(float(x), 4) for x in np.r_[lo, hi]])
        elif len(part) == 8 and oriented_box(p):
            boxes.append([[round(float(x), 4) + 0.0 for x in c] for c in p])
        else:
            n_f = int(np.isin(f, part).all(1).sum())
            left.append({"vertices": len(part), "faces": n_f})
    return boxes, left


def supine_view(spec, body, floor):
    """(floor, heading_deg) of a clip: a supine clip on the examination couch (scene 'exam_bed', task SKELV4a) takes
    the room floor and the couch's heading of ataxia.bed_view, as render_mesh_video.render_clip does (its lowest
    joints lie on the couch top, and a lying root's heading is not the couch's); any other clip its joints' floor
    (clip_floor) and the root's heading at the middle knot (viewer.heading_deg)."""
    if spec.scene != "exam_bed":
        return floor, V.heading_deg(body)
    from mdfm.synth.ataxia import bed_view
    bv = bed_view(body)
    return (dict(floor, floor_y_m=float(bv["floor_y"]),
                 rule="supine on the couch: the room floor of ataxia.bed_view (the body's lowest vertex at its first "
                      "knot minus the couch height), not the joints' floor"), float(bv["heading_deg"]))


def walk_record(spec, body):
    """{centre_xz, axis_xz} of a walk clip (viewer.walk_axis on the pelvis at the knots), None for the others."""
    if "track_left" not in spec.views:
        return None
    c, a = V.walk_axis(body.joints().numpy()[:, 0, [0, 2]])
    return {"centre_xz": [float(x) for x in c], "axis_xz": [float(x) for x in a],
            "rule": "viewer.walk_axis: principal direction of the pelvis path, oriented along the first passage"}


def clip_meta(spec, truth, body, skel, set_name=SET, floor_rule="median"):
    cap = spec.caption_spec()
    cap["params_text"] = EX.describe(spec, truth)
    fl, heading = supine_view(spec, body, clip_floor(skel, floor_rule))
    return {"label": LB.LABEL, "notice": NOTICE, "catalogue": set_name, "clip_id": spec.clip_id,
            "family": spec.family, "segment": spec.segment, "segment_title": EX.segment_title(spec), "item": spec.item,
            "operator": spec.operator, "grade": spec.grade, "grade_text": spec.grade_text,
            "params_text": cap["params_text"], "caption": list(V.caption_lines(cap)), "level0": spec.level0,
            "since": spec.since, "views": list(spec.views), "scene": spec.scene,
            "heading_deg": heading, "floor": fl,
            "scene_boxes_m": scene_boxes(spec.scene, body, fl["floor_y_m"]), "walk": walk_record(spec, body),
            "body_span_s": body.duration, "body_rate_hz": body.rate,
            "toe_stance": bool(((truth or {}).get("params") or {}).get("toe_stance", False))}


def clip_overlays(truth):
    """[{kind, from_s, to_s, text}] of the per-time texts a clip's truth carries for a viewer (task SKELV3), on the
    clip's clock (t = 0 at its first frame), in the truth's own English words: params['frame_text'] ({from_s, to_s,
    text}: the Romberg eye state) as kind 'frame_text', params['stimulus_marks'] ({t_s, dur_s, text}: the stimuli of
    the jerks family) as kind 'stimulus' from t_s to t_s + dur_s; [] when it carries none."""
    p = ((truth or {}).get("params") or {})
    out = [{"kind": "frame_text", "from_s": float(x["from_s"]), "to_s": float(x["to_s"]), "text": str(x["text"])}
           for x in (p.get("frame_text") or [])]
    out += [{"kind": "stimulus", "from_s": float(x["t_s"]), "to_s": float(x["t_s"]) + float(x["dur_s"]),
             "text": str(x["text"])} for x in (p.get("stimulus_marks") or [])]
    return out


# ---------------------------------------------------------------- the drawing landmarks of the catalogue export
# Task SKELV3 review: a rotation about a joint that has no point beyond it moves no point of the skeleton, so the
# 54 points hide the moving toes (the toe block turns about the foot joint), the inversion of the foot (a roll about
# its long axis), the pinch of the thumb and index (the distal phalanges are not drawn) and the roll of the head
# (the head circle and the nose show turn and flexion). The catalogue export adds, per side, points fixed in a
# joint's frame like the head landmarks of mdfm.synth.skeleton (the joint plus its global rotation times an offset):
# kind: (joint, rule on the neutral rest mesh, the vertices whose dominant joint is the given one)
LANDMARK_RULES = {
    "toe_tip": ("foot", "the most forward vertex of the toes (dominant joint: the foot joint)"),
    "heel": ("ankle", "the most backward vertex of the foot segment (dominant joint: the ankle) more than 3 cm under "
                      "the ankle joint"),
    "forefoot_lateral": ("ankle", "the most lateral vertex of the foot segment"),
    "index_tip": ("index3", "the vertex of the distal phalanx farthest along the direction from index2 to index3"),
    "thumb_tip": ("thumb3", "the vertex of the distal phalanx farthest along the direction from thumb2 to thumb3"),
    "ear": ("head", "the most lateral vertex of the head segment"),
}
LANDMARK_SIDES = ("left", "right")
CATALOGUE_LANDMARKS = tuple(f"{s}_{k}" for s in LANDMARK_SIDES for k in LANDMARK_RULES)
LANDMARK_ROUND_M = 0.005               # [module] offsets: the mean of the side's and the mirrored other side's,
#                                        rounded to 5 mm (drawing landmarks, not mesh vertices)
LANDMARK_BONES = [                     # the lines the page draws between a landmark and a point: [from, to, style]
    *[[f"{s}_heel", f"{s}_ankle", "foot"] for s in LANDMARK_SIDES],
    *[[f"{s}_forefoot_lateral", f"{s}_heel", "foot"] for s in LANDMARK_SIDES],
    *[[f"{s}_forefoot_lateral", f"{s}_foot", "foot"] for s in LANDMARK_SIDES],
    *[[f"{s}_toe_tip", f"{s}_foot", "toe"] for s in LANDMARK_SIDES],
    *[[f"{s}_{f}_tip", f"{s}_{f}3", "finger"] for s in LANDMARK_SIDES for f in ("index", "thumb")]]
REGION_LANDMARKS = {                   # the landmarks a view's region frames (the page's fit)
    "body": list(CATALOGUE_LANDMARKS),
    "foot_left": ["left_toe_tip", "left_heel", "left_forefoot_lateral"],
    "hands": [f"{s}_{f}_tip" for s in LANDMARK_SIDES for f in ("index", "thumb")],
    "hand_left": ["left_index_tip", "left_thumb_tip"], "head": ["left_ear", "right_ear"],
    # task SKELV4a: the foot regions of viewer.VIEWS after SH11 and the legs of the couch views (CATALOGUE_REGIONS)
    "foot_left_close": ["left_toe_tip", "left_heel", "left_forefoot_lateral"],
    "feet": [f"{s}_{k}" for s in LANDMARK_SIDES for k in ("toe_tip", "heel", "forefoot_lateral")],
    "toes_left": ["left_toe_tip", "left_forefoot_lateral"],
    "legs": [f"{s}_{k}" for s in LANDMARK_SIDES for k in ("toe_tip", "heel", "forefoot_lateral")]}


def _rest_mesh():
    """(rest vertices (V, 3) m, faces (F, 3), rest joints (52, 3) m) of the selected body model (plan phase C wave 1:
    the open body by default, body.body_model; on the SMPL+H reference its template and regressor joints, as before
    the port)."""
    from mdfm.synth.body import is_open, rest_joints52
    if is_open():
        return V.rest_vertices(), V.faces(), rest_joints52().numpy()
    import torch
    model, faces = V.mesh_model("neutral")
    J = model.rest_joints(torch.zeros(1, model.shapedirs.shape[-1], dtype=torch.float64))[0].numpy()
    return model.v_template.numpy(), np.asarray(faces), J


def landmark_offsets():
    """{name: (joint index, offset (x, y, z) m in the joint's frame)} of CATALOGUE_LANDMARKS: LANDMARK_RULES read on
    the rest mesh of the selected body model (rest frame: y up, the face toward +z, the left side toward +x), each
    offset the mean of its side's and the mirrored other side's, rounded to LANDMARK_ROUND_M; vertices stay in
    memory."""
    from mdfm.synth.body import model_name
    return _landmark_offsets(model_name())


@functools.lru_cache(maxsize=2)
def _landmark_offsets(model):
    v, _, J = _rest_mesh()
    dom = V.dominant_joint("neutral")
    ix = {n: i for i, n in enumerate(SK.JOINT_NAMES)}
    raw = {}
    for side, sx in (("left", 1.0), ("right", -1.0)):
        a, f, h = ix[f"{side}_ankle"], ix[f"{side}_foot"], ix["head"]
        toes, foot, head = v[dom == f], v[dom == a], v[dom == h]
        raw[side, "toe_tip"] = (f, toes[toes[:, 2].argmax()] - J[f])
        low = foot[foot[:, 1] < J[a, 1] - 0.03]
        raw[side, "heel"] = (a, low[low[:, 2].argmin()] - J[a])
        raw[side, "forefoot_lateral"] = (a, foot[(sx * foot[:, 0]).argmax()] - J[a])
        for fin in ("index", "thumb"):
            j3, j2 = ix[f"{side}_{fin}3"], ix[f"{side}_{fin}2"]
            pts, ax = v[dom == j3], J[j3] - J[j2]
            raw[side, f"{fin}_tip"] = (j3, pts[((pts - J[j3]) @ ax).argmax()] - J[j3])
        raw[side, "ear"] = (h, head[(sx * head[:, 0]).argmax()] - J[h])
    mirror = np.array([-1.0, 1.0, 1.0])
    out = {}
    for side in LANDMARK_SIDES:
        other = "right" if side == "left" else "left"
        for kind in LANDMARK_RULES:
            j, off = raw[side, kind]
            sym = (off + raw[other, kind][1] * mirror) / 2
            out[f"{side}_{kind}"] = (int(j), tuple(round(float(x), 4) + 0.0 for x in      # no -0.0
                                                   np.round(sym / LANDMARK_ROUND_M) * LANDMARK_ROUND_M))
    return out


def catalogue_skeleton(body, fps=FPS):
    """The skeleton of a catalogue clip: mdfm.synth.skeleton.skeleton (52 joints and the two head landmarks), then
    CATALOGUE_LANDMARKS (landmark_offsets) on the same clock and forward kinematics."""
    skel = SK.skeleton(body, fps)
    P, G = SK.chain_at(body, skel.times(), True)
    if not np.allclose(P, skel.positions[:, :P.shape[1]], atol=1e-9):
        raise RuntimeError("the landmark chain differs from the skeleton's")
    off = landmark_offsets()
    extra = np.stack([P[:, j] + G[:, j] @ np.asarray(o) for j, o in (off[n] for n in CATALOGUE_LANDMARKS)], 1)
    return dataclasses.replace(skel, positions=np.concatenate([skel.positions, extra], 1),
                               names=tuple(skel.names) + CATALOGUE_LANDMARKS,
                               parents=tuple(skel.parents) + tuple(off[n][0] for n in CATALOGUE_LANDMARKS))


def catalogue_record(skel):
    """mdfm.synth.skeleton.record (1 mm, delta coded) of a catalogue skeleton: its own points, names, parents and
    bones, the landmark table with CATALOGUE_LANDMARKS (joint, offset, rule)."""
    k = skel.n_points - len(CATALOGUE_LANDMARKS)
    rec = SK.record(dataclasses.replace(skel, positions=skel.positions[:, :k], names=skel.names[:k],
                                        parents=skel.parents[:k]), mm=1, delta=True)
    q = SK.quantize_mm(skel.positions, 1)
    d = q.copy()
    d[1:] -= q[:-1]
    off = landmark_offsets()
    rec.update(n_points=skel.n_points, names=list(skel.names), parents=list(skel.parents),
               bones=[list(b) for b in skel.bones()], positions_mm_delta=d.reshape(-1).tolist())
    for n in CATALOGUE_LANDMARKS:
        j, o = off[n]
        rec["landmarks"][n] = {"joint": j, "offset_m": list(o), "rule": LANDMARK_RULES[n.split("_", 1)[1]][1]}
    return rec


# ---------------------------------------------------------------- wave SH1d: the uniform checks of the skeleton
# The clinical review of the English page (2026-11-03, PLAN.txt of the session, lane 1): defects that every family
# can make and that its own validity does not see, read on the exported positions (1 mm, 30 fps) of every shown clip:
# - continuity (M4, M9, M10): a point above ACCEL_MAX_M_S2 outside the clip's declared jerk events ('events_s'), a
#   foot point that stands on the floor jumping more than FOOT_JUMP_MM in one frame, and a segment of a symptom clip
#   (the trunk, the neck, an upper arm, an elbow's flexion) turning faster than RATE_CAPS_DEG_S and than its level 0;
# - self_contact (M6, M7, M20, M1): a hand point inside the head or a thigh, a hand within HAND_HAND_MM of the other
#   one (two fingertips within SKIN_TOL_MM), and, for a touch on the thighs the clip's own words declare
#   (DECLARED_TOUCHES), the palm farther than TOUCH_GAP_MM from the thigh. The head and the thighs are the neutral rest
#   mesh's own surfaces carried rigidly by the head and by the femur (contact_geometry), not the ellipsoid and the
#   femur-axis cylinder the plan proposes: the head of the rest mesh is no ellipsoid (the jaw, the nose and the ears
#   stand out of any one), so a fingertip on the cheek and a hand inside the jaw can read the same on an ellipsoid, and
#   the thigh's skin is no cylinder about the femur axis (its distance from the axis changes around the thigh and along
#   it): on the surface itself a point is inside or not, and the plan's 10 mm margin over the cylinder is not needed;
# - skate (M11): a foot whose least-moving point standing on the floor (the sole points and the toe tip) moves more
#   than SKATE_MM horizontally within one contact (a roll or a pivot keeps one point still);
# - statue (M3): a STATUE_WINDOW_S window in which no body joint moves by one step of the export (STATUE_MM), on a
#   clip not lying on the couch;
# - pair_floor (M5): the floor a symptom clip stands on (the sole of each foot over its standing frames on the
#   floor) against its level 0's, read at the foot that changed least;
# - sinking: a sole point more than CONTACT_TOL_MM under the floor the page draws.
# They were reported only (SKELETON_REPORTED_ONLY: written into the manifest, the clip shown) until the skate check
# had a tolerance for turning contacts (the sole cloud, 2026-11-11, reports/gpu_queue/decisions.md); since then every
# check refuses: the page hides the clips it flags (and the clips drawn beside them, refusal_followers), named in the
# about section and in SYNTHETIC_skeleton_refused.json. scripts/skeleton_page_checks.py measures them on a build of the
# catalogue set (reports/synth/viewer/page_checks.md).
SKELETON_CHECKS = ("continuity", "self_contact", "skate", "statue", "pair_floor", "sinking")
SKELETON_REPORTED_ONLY = ()
CHECKS_WORDS = ("continuity, contact between body parts, foot skate, a body frozen for a second, the floor of the "
                "pair, a foot sinking into the floor")   # the checks in the page's words, in SKELETON_CHECKS order
ACCEL_MAX_M_S2 = 100.0                 # [plan] PLAN lane 1, finding M4 (the check's own reading, 'accel_max_m_s2' in
#                                        the manifest, puts the generated walks' heel strikes well under it)
FOOT_JUMP_MM = 30.0                    # [plan] a foot point standing on the floor moving more than this in one frame
RATE_CAPS_DEG_S = {"trunk": 120.0, "neck": 200.0, "upper_arm": 250.0, "elbow": 350.0}
#   [plan] the angular speeds PLAN.txt sets as bounds: M9 (the involuntary movements' channels: about 250 deg/s at the
#   shoulder, 350 at the elbow, 200 at the neck) and M10 (the trunk of a failed rise falling back: about 120 deg/s).
#   ACCEL_MAX_M_S2 alone misses them: the failed rises reach 17 to 30 m/s2 at the head (PLAN M10), and
#   invol_chorea_E7_g3, invol_chorea_E1_g4 and invol_chorea_E1_arms_g4 stay under 100 (wave SH1d review of lane
#   page). The forearm's rotation (the plan's 350 deg/s) is not read: no two joints carry it. A reading is judged
#   against the level 0's own (rates_over), so that a level 0 that is itself fast (the healthy rise: 127 to 128 deg/s
#   at the trunk on the skeleton_v2 data, wave SH1d scan) sets the floor for its symptom clips
RATE_SEGMENTS = {"trunk": ("pelvis", "neck"), "neck": ("neck", "head"),
                 **{f"{s}_upper_arm": (f"{s}_shoulder", f"{s}_elbow") for s in ("left", "right")}}
RATE_ON_TRUNK = ("neck", "left_upper_arm", "right_upper_arm")   # the readings taken in the trunk's frame
#   (trunk_frame), so that a neck or an arm carried by a turning trunk is read on its own joint's turn, as the
#   generators bound each joint's speed on its own (wave SH1d review of lane page); the trunk is read in the room
EXPORT_STEP_MM = 1.0                   # [module] the grid of the exported positions (catalogue_record: mm=1)
RATE_EXEMPT = [    # (regex on the clip id, reason): the clips whose rate readings the continuity check does not judge
    (r"^(standing_)?tremor_",
     "a tremor: an oscillation whose peak angular speed is 2 pi times its frequency times its amplitude, graded on "
     "its amplitude by the tremor scales")]
CONTACT_TOL_MM = 5.0                   # [module] a foot point is at contact height within this of its height above the
#                                        sole on the neutral rest mesh (contact_geometry 'foot_rest_mm')
CONTACT_SPEED_M_S = 0.25               # [module] and stands (rather than skims the floor in swing) below this
#                                        horizontal speed: above the dystonic slides of 0.17 to 0.18 m/s the review
#                                        measured (PLAN M11), below a swinging foot
MIN_CONTACT_FRAMES = 3                 # [module] a contact lasts at least 0.1 s on the 30 fps clock, its first and last
#                                        frames (the landing and the lift) left out of its travel
SKATE_MM = 3.0                         # [plan] the horizontal travel of a foot within one contact
SKATE_TOUCH_MM = 2.0                   # [module] with the sole cloud, a skin point of a foot carries the contact (the
#   skate check reads its travel) on the frames it stands on the floor (_standing_on_floor) at most this above the
#   foot's lowest standing skin point: the plantar patch the foot rests on, not the points that stand within
#   CONTACT_TOL_MM of their rest height while the foot lands, lifts or rolls over them (at the push-off of a turning
#   step the ball, 1 to 4 mm above the toes the foot turns about, circles them by 3 to 5 mm: gait_E8_level0 of
#   commit e9c7749, its left turn at 9.1 to 9.9 s)
SKATE_RULE = (f"a foot whose least-moving touching point (the sole cloud, the heel and lateral sole points and the toe "
              f"tip, on the frames each one touches the floor) moves more than {SKATE_MM:g} mm plus the sole cloud's "
              f"covering radius times the foot's turn (radians) horizontally within one contact, outside the declared "
              f"slides of a dragged foot")
SKATE_POINT_SHARE = 1.0 / 3.0          # [module] a foot point the skate check reads within a contact stands on at least
#   this share of its frames, so that a point touching for a few frames (the heel at the strike, the toe tip at the
#   push-off) does not hide a glide of the foot. It was one half: a slow stance that stands flat and still for less
#   than half of its contact and then rolls onto the toes (the toe tip, the only point left on the floor, rolling
#   forward as the toes bend) then read on the toe tip alone and failed, a roll and no skate (the sound foot of
#   gait_E8_functional_dragging_left and the last step of gait_E8_level0 as the worktree builds them, wave SH1d: the
#   sole points still on 43 to 48% of the contact, the toe tip rolling 11 mm)
TURN_YAW_DEG = 20.0                    # [review] the wave SH1d review of lane page (its major 2): a contact in which
#   the foot turns by at least this (the heel sole to toe tip line, horizontally) is a turning contact, counted apart
#   ('turning'); since the sole cloud (2026-11-11) both kinds are judged by one rule, whose tolerance grows with the
#   turn by the cloud's covering radius (contact_geometry 'cover_mm'): a foot pivoting about a sole point that lies
#   between the cloud's samples moves the nearest sample by up to that radius times the turn
SLIDE_PAD_FRAMES = 1                   # [module] a declared slide (slides_s) is widened by one frame at each end, as
#                                        the jerk events (free_frames)
CAPTURED = r"^mocap_.+$"               # the clips built on a motion-capture recording would carry the prefix
#   'mocap_': none does since the open-body port's wave 7 (the exam set's stand, walk and freezing clips, AMASS takes
#   until then, are procedural); the exemption tables and the provenance cross-check keep their rule for such a clip
SKATE_EXEMPT = [   # (regex on the clip id, reason): the clips the skate check reads but does not judge (ok None)
    (CAPTURED,
     "built on a motion-capture recording: the captured feet slide within a "
     "contact, and the threshold is set for the generated feet")]
#   a clip of SKATE_EXEMPT that declares freezing holds ('holds_s', hold_spans) is judged inside them: a hold freezes
#   one captured frame, the feet kept where they stand while the knees tremble (freezing._tremble_pose), so a foot
#   sliding there is the program's own; its captured walk outside the holds stays read only. The readings of the
#   clips read and not judged go to the manifest ('skeleton_checks_not_judged'), so that the captured feet's slide
#   (the evidence for the exemption) stays in view. SKATE_MM is not calibrated on these clips: the captured feet slide
#   over most of their contacts by several times SKATE_MM (the manifest's readings), a property of the recording and
#   of its fit that the generated feet do not share
SINKING_EXEMPT = [  # (regex, reason): the clips the sinking check reads but does not judge
    (CAPTURED,
     "built on a motion-capture recording: the captured soles sit up to 8 mm "
     "under the floor of the standing frames, a property of the recording and of its fit, which no generator sets")]
#   the depths are the check's own readings ('depth_mm', in the manifest's 'skeleton_checks_not_judged'); inside a
#   freezing hold the soles are those of the captured frame it freezes
CONTACT_EXEMPT = [  # (regex, reason): the clips whose hand-to-thigh contacts the self-contact check reads, not judges
    (CAPTURED,
     "built on a motion-capture recording: the captured arm swing brushes the thigh, which no generator "
     "of the program sets")]
#   the captured walk brushes its thighs over most of its contacts and the captured stance over part of them (the
#   check's own readings, in the manifest's 'skeleton_checks_not_judged')
STATUE_WINDOW_S = 1.0                  # [plan]
STATUE_PLAN_MM = 2.0                   # [plan] the plan's limit, read beside STATUE_MM and not judged: the seconds a
#   clip spends in windows under it ('near_still_s'), a stronger reading of a layered clip than STATUE_MM, which on
#   the 1 mm grid only asks that some joint move by one step (wave SH1d, for lane physio)
STATUE_MM = 1.0                        # [module] every body joint's bounding-box diagonal over the window under this:
#   on the 1 mm grid of the export, no coordinate of any body joint changes over the window (the body frozen). The
#   plan's 2 mm also flags the slowest windows of the layered seated level 0s (dystonia_E1_level0, jerk_E1_level0,
#   invol_E1_level0), which still move a joint by one export step, while the statues of the review (the walks' still
#   ends, standing_tremor_E10_posture_g0, the freezing holds) do not move by any step (wave SH1d scan of the
#   skeleton_v2 data)
HAND_HAND_MM = 15.0                    # [plan] the nearest points of the two hands, when one of them is a joint centre
#   (inside a finger); two skin points (the index and thumb tips) are flagged within SKIN_TOL_MM only, where they
#   overlap: two fingertips resting 10 mm apart touch nothing (park_E1_spontaneity_g0, the thumb tips 9.8 mm apart
#   on the skeleton_v2 data, wave SH1d). The wings' own room (PLAN M7, at least 20 mm) is a
#   test of the tremor lane (PLAN lane 3), not this check's
TOUCH_GAP_MM = 15.0                    # [plan] the palm or fingertip of a declared touch from its target
SKIN_TOL_MM = 5.0                      # [module] a skin point (a fingertip or palm landmark) inside the head or a thigh
#                                        by more than this; a joint centre (inside the finger) is flagged once inside
PAIR_FLOOR_TOL_MM = 2.0                # [module] two 1 mm readings, each within 0.5 mm, and the 1 mm sway of a seat
FLOOR_LOW_PCT = 5.0                    # [module] a foot's floor reads its standing frames near this low percentile of
#                                        its sole height (a percentile, not the minimum, so that one frame through the
#                                        floor does not set it)
THIGH_T_MIN = 0.25                     # [module] the thigh from a quarter of the femur on (the hip crease and the
#                                        buttock deform with the hip and are left out), to the knee; a point is read
#                                        against it only inside that span of the femur (the surface is open at both
#                                        ends)
PALM_NORMAL_Y = -0.5                   # [module] the palm's skin: the hand vertices (dominant joint the wrist) whose
#                                        rest normal points palmar (down, rest palms down) by more than this
THIGH_TOL_MM = 8.0                     # [module] a skin point inside a thigh by more than this (a joint centre by more
#   than this less SKIN_TOL_MM): the rest thigh carried rigidly by the femur read up to 7.6 mm closer to a palm than
#   the posed mesh's own thigh over the same span of the femur, on clips built with the worktree's code (wave SH1d,
#   lane page fix: park_E7_gowers, park_E7_arise_g4, invol_E1_level0, gait_E8_duchenne_right, every third frame)
THIGH_FLEX_DEG = (60.0, 110.0)         # [module] the hip flexion (hip_flexion_deg) range in which a declared touch's
#   gap is read: the same comparison put the rigid thigh within 7.6 mm closer and 15 mm farther than the mesh between 60
#   and 110 deg on the seated clips and the Gowers clip (invol_E1_level0 at 80 to 82 deg: -1.6 to +3.4 mm), and 7 to
#   54 mm farther outside it (the thigh's skin bulges at deep flexion: park_E7_gowers leaning on its thighs at 120 to
#   140 deg read 20 to 35 mm off while the mesh put the palms 3 to 9 mm from the skin; a standing hip,
#   gait_E8_duchenne_right at 5 to 15 deg). One clip reads off inside the range too: park_E7_arise_g4, 29 to 50 mm
#   farther than the mesh at 73 to 105 deg. The check therefore errs toward reading a palm farther than it is: a
#   contact it flags is on the mesh too, a gap it flags is to be confirmed on the mesh before a clip is refused
HEAD_JOINT, NECK_JOINT = 15, 12
SIDES = ("left", "right")
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
HAND_JOINTS = {s: [f"{s}_wrist"] + [f"{s}_{f}{i}" for f in FINGERS for i in (1, 2, 3)] for s in SIDES}
HAND_SKIN = {s: [f"{s}_index_tip", f"{s}_thumb_tip"] for s in SIDES}
HAND_FRAME = {s: [f"{s}_{j}" for j in ("wrist", "index1", "middle1", "ring1", "pinky1")] for s in SIDES}
FOOT_POINTS = {s: [f"{s}_foot", f"{s}_heel_sole", f"{s}_lateral_sole"] for s in SIDES}
#   the points a foot rests on: the toe-block joint (the pivot of the push-off) and two points of the sole carried by
#   the foot's pose (contact_geometry 'foot_<side>'): under the heel and under the lateral border of the forefoot. The
#   drawing landmarks are not contact points: the heel landmark (the back of the heel, above the sole) swings when
#   the foot turns about the heel (the toe tapping), the tip of the big toe rolls with the toe block at each push-off
#   of gait_E8_level0, and the ankle rides the heel and ankle rockers (wave SH1d scan of the skeleton_v2 data)
FOOT_FRAME = {s: [f"{s}_ankle", f"{s}_foot", f"{s}_heel", f"{s}_forefoot_lateral"] for s in SIDES}
#   four points fixed in the foot's frame: the ankle, the toe-block joint and the two foot landmarks
SOLE_PICK_MM = 15.0                    # [module] the lateral sole point: the lowest foot vertex within this
#                                        (horizontally) of the lateral forefoot landmark
HEEL_BEHIND_MM = 20.0                  # [module] the heel sole point: the lowest foot vertex this far behind the ankle
SOLE_CLOUD_MM = 10.0                   # [module] the sole cloud of the skate check: the foot's vertices within this of
#   the sole on the neutral rest mesh, facing down (SOLE_NORMAL_Y): the plantar skin a foot can stand and pivot on,
#   the heel pad to the toe pads (the gait generator pivots a turning stance foot on its touching sole point nearest
#   the ball, gait.py PIVOT_NEAR_M, a point of the same skin)
SOLE_SUBDIV = 4                        # [module] the sole cloud sampled on the rest mesh's faces among its vertices,
#   each edge cut in this many parts (2 to 5 mm between samples, where the vertices lie 10 to 20 mm apart): a foot
#   pivoting about a point between two vertices moves each of them by its distance from the pivot times the turn
SOLE_NORMAL_Y = -0.5                   # [module] a sole cloud vertex's rest normal points down by more than this
SOLE_REGIONS = ((0.3, "heel"), (0.6, "midfoot"), (0.8, "ball"), (1.0, "toes"))
#   the name of a sole cloud point in the readings: its place along the rest sole cloud (heel 0, toes 1), up to the
#   share given (a label, not a measurement)
DECLARED_TOUCHES = [   # (regex on the clip id, target, hands, kind, the clip's own words): 'rest' holds over the clip
    #                    (the median gap of the palm), 'reach' at least once (the smallest; the module's reading on
    #                    the posed mesh when the clip carries one, MESH_TOUCHES), 'allowed' only lets the
    #                    hands touch the target (a touch its module measures on the mesh itself: the applause's
    #                    parkinsonism.CLAP_GAP_M, the geste's postures.GESTE_PALM_GAP_M, the touching tics' reach
    #                    skin gap); a level 0's 'rest' touch allows the same target to its symptom clips (a hand on the
    #                    thigh is then not flagged until inside it)
    (r"^invol_E1_level0", "thighs", SIDES, "rest", "hands resting on the thighs"),
    (r"^invol_chorea_E1_trunk_", "thighs", SIDES, "rest", "the hands staying on the thighs"),
    (r"^invol_td_stereotypy_E1_rubbing_", "thighs", SIDES, "rest", "both palms sliding along the thighs"),
    (r"^park_E7_gowers$", "thighs", SIDES, "reach", "the hands climb the thighs"),
    (r"^park_E21_applause_", "hands", SIDES, "allowed", "meeting palm to palm"),
    (r"^post_E7_cd_geste_", "head", ("right",), "allowed", "the right fingertips on the right cheek"),
    (r"^jerk_tics_E\d_touch_", "head", ("right",), "allowed", "touching the face (the right index)")]
MESH_TOUCHES = {"thighs": "hand_thigh"}   # target: the validity entry of a clip's truth whose 'touch_mm' {side: mm}
#   is its module's own reading of a declared 'reach' touch on the posed mesh (the smallest gap of the hand above the
#   target's skin; parkinsonism.gowers: the planted hands' lowest vertex above the thigh's front, measure_gowers). The
#   export carries it into the data file ('touches_mesh_mm', touch_readings) and check_self_contact judges that reading
#   against TOUCH_GAP_MM in place of the rigid thigh's (plan phase E): the rigid stand-in reads a palm farther than the
#   mesh at deep flexion (THIGH_FLEX_DEG; park_E7_gowers read 27.2 and 27.8 mm off it where the module's mesh puts the
#   planted hands 3.00 to 5.56 mm from the skin, design D115); a data file without it is read on the rigid thigh


def _vertex_normals(v, f):
    """(V, 3) unit outward normals of a mesh's vertices (the area-weighted face normals around each vertex)."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _rim(faces, mask):
    """The vertex indices on the open rim of the submesh of the vertices in mask (edges of one face only)."""
    f = faces[mask[faces].all(1)]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    return np.unique(u[c == 1])


def femur_frame(hip, knee, ankle, foot):
    """(..., 3, 3) the frame of a femur from four joint positions, columns (axis, anterior, their cross product): the
    axis hip to knee; the anterior direction the foot's (ankle to toe joint) minus the shank's (knee to ankle),
    both taken square to the axis, which holds from a straight knee (the foot forward) to a right angle (the shank
    backward)."""
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    u = unit(knee - hip)

    def perp(x):
        return x - (x * u).sum(-1, keepdims=True) * u
    a = unit(perp(unit(foot - ankle)) - perp(unit(ankle - knee)))
    return np.stack([u, a, np.cross(u, a)], -1)


def contact_geometry():
    """The body surfaces of the skeleton checks, read on the rest mesh of the selected body model (in memory, never
    written; _contact_geometry keyed by the model's name), lengths in mm:
    'head': the vertices whose dominant joint is the head or the neck, relative to the head joint, with their
    outward normals, carried by the head (the neck's part rigid with it near the head); 'y_min' the highest point of
    the submesh's open rim at the base of the neck (a point lower in the head's frame is not read); 'frame_points'
    the rest head points (the joint, the two landmarks, the two ears) that give the head's pose;
    'thigh_<side>': the vertices whose dominant joint is that hip or knee, from THIGH_T_MIN of the femur to the knee,
    in the rest femur frame (femur_frame) from the hip, with normals, and the femur's length;
    'palm_<side>': the hand's frame points (HAND_FRAME, rest) and the palm's skin (the vertices whose dominant joint
    is the wrist and whose rest normal points palmar, PALM_NORMAL_Y), carried by the hand's pose;
    'foot_<side>': the foot's frame points (FOOT_FRAME, rest) and its sole points: heel_sole the lowest vertex whose
    dominant joint is the ankle more than HEEL_BEHIND_MM behind the ankle, lateral_sole the lowest vertex of the foot
    (dominant joint the ankle or the toe block: the fifth metatarsal head) within SOLE_PICK_MM of the lateral forefoot
    landmark; carried by the foot's pose; and its sole cloud (the points a foot may pivot on, read by the skate
    check: sole_cloud): 'cloud' the vertices whose dominant joint is the ankle or the toe block, within SOLE_CLOUD_MM
    of the sole and facing down (rest normal under SOLE_NORMAL_Y), 'cloud_toe' those of the toe block, 'cloud_w' the
    barycentric weights over them of the samples of the sole (_sole_samples on the faces among them, SOLE_SUBDIV),
    'cloud_rest_mm' the samples' heights above the sole, 'cloud_region' their place along the foot (SOLE_REGIONS),
    'toe_joint' and 'toe_tip' the rest toe-block joint and toe tip landmark (the toes' bend);
    'foot_rest_mm': each foot point's height above the sole (FOOT_POINTS); 'toe_rest_mm': the toe tip landmark's
    (CATALOGUE_LANDMARKS), a point the skate check reads besides them (skate_rest_mm); 'cover_mm': the covering radius
    of the samples, the farthest point of the sampled sole (cut four times finer) from its nearest sample, over both
    feet."""
    from mdfm.synth.body import model_name
    return _contact_geometry(model_name())


@functools.lru_cache(maxsize=2)
def _contact_geometry(model):
    v, faces, J = _rest_mesh()
    v, J = v * 1000.0, J * 1000.0
    dom = V.dominant_joint("neutral")
    nrm = _vertex_normals(v, faces)
    ix = {n: i for i, n in enumerate(SK.JOINT_NAMES)}
    off = landmark_offsets()
    sel = np.isin(dom, (HEAD_JOINT, NECK_JOINT))
    head_pts = {"head": np.zeros(3), **{k: 1000.0 * np.asarray(o) for k, o in SK.head_landmarks().items()},
                **{f"{s}_ear": 1000.0 * np.asarray(off[f"{s}_ear"][1]) for s in SIDES}}
    rim = _rim(faces, sel)
    geo = {"head": {"v": v[sel] - J[HEAD_JOINT], "n": nrm[sel], "frame_points": head_pts,
                    "y_min": float((v[rim] - J[HEAD_JOINT])[:, 1].max())}}
    for s in SIDES:
        hip, knee, ankle, foot = (ix[f"{s}_{j}"] for j in ("hip", "knee", "ankle", "foot"))
        E = femur_frame(J[hip], J[knee], J[ankle], J[foot])
        L = float(np.linalg.norm(J[knee] - J[hip]))
        sel = np.isin(dom, (hip, knee))
        t = (v[sel] - J[hip]) @ E[:, 0] / L
        keep = (t >= THIGH_T_MIN) & (t <= 1.0)
        geo[f"thigh_{s}"] = {"v": ((v[sel] - J[hip]) @ E)[keep], "n": (nrm[sel] @ E)[keep], "length": L}
        fr = [ix[p] for p in HAND_FRAME[s]]
        palm = (dom == ix[f"{s}_wrist"]) & (nrm[:, 1] < PALM_NORMAL_Y)
        geo[f"palm_{s}"] = {"frame": J[fr], "v": v[palm]}
    sole = float(v[:, 1].min())
    rest, cover = {}, 0.0
    from scipy.spatial import cKDTree
    for s in SIDES:
        a = ix[f"{s}_ankle"]
        fr = {p: (J[ix[p]] if p in ix else J[off[p][0]] + 1000.0 * np.asarray(off[p][1])) for p in FOOT_FRAME[s]}
        foot, toes = dom == a, np.isin(dom, (a, ix[f"{s}_foot"]))
        pts = {}
        for name, near in (("heel_sole", foot & (v[:, 2] < J[a, 2] - HEEL_BEHIND_MM)),
                           ("lateral_sole", toes & (np.linalg.norm((v - fr[f"{s}_forefoot_lateral"])[:, [0, 2]],
                                                                   axis=1) <= SOLE_PICK_MM))):
            pts[f"{s}_{name}"] = v[np.flatnonzero(near)[v[near, 1].argmin()]]
        cl = np.flatnonzero(toes & (v[:, 1] - sole <= SOLE_CLOUD_MM) & (nrm[:, 1] < SOLE_NORMAL_Y))
        W = _sole_samples(faces, cl, SOLE_SUBDIV)
        x = W @ v[cl]
        fine = _sole_samples(faces, cl, 4 * SOLE_SUBDIV) @ v[cl]
        cover = max(cover, float(cKDTree(x).query(fine)[0].max()))   # the farthest sole point from a sample
        share = (x[:, 2] - x[:, 2].min()) / max(float(np.ptp(x[:, 2])), 1e-9)   # heel 0 to toes 1, along the foot
        geo[f"foot_{s}"] = {"frame": fr, "sole": pts, "cloud": v[cl], "cloud_w": W, "cloud_rest_mm": x[:, 1] - sole,
                            "cloud_toe": dom[cl] == ix[f"{s}_foot"],
                            # plan phase E: the toe block's sole vertices behind the toe-block joint (the plantar
                            # skin under the metatarsal heads, which Anny's weights give to the toe block) are
                            # carried by the foot, not turned with the toes: turned rigidly about a joint 25.7 mm
                            # above the sole they slid under a still foot as the toes extended (the moving toes,
                            # invol_moving_toes_E1_left_a3, read 7.99 mm of skate with every foot joint still)
                            "cloud_toe_bend": (dom[cl] == ix[f"{s}_foot"]) & (v[cl, 2] >= J[ix[f"{s}_foot"], 2]),
                            "cloud_region": [next(r for lim, r in SOLE_REGIONS if q <= lim) for q in share],
                            "toe_joint": J[ix[f"{s}_foot"]].copy(),
                            "toe_tip": J[off[f"{s}_toe_tip"][0]] + 1000.0 * np.asarray(off[f"{s}_toe_tip"][1])}
        rest[f"{s}_foot"] = float(J[ix[f"{s}_foot"], 1] - sole)
        rest.update({k: float(x[1] - sole) for k, x in pts.items()})
    geo["foot_rest_mm"] = rest
    geo["cover_mm"] = cover
    geo["toe_rest_mm"] = {f"{s}_toe_tip": float(J[off[f"{s}_toe_tip"][0], 1] + 1000.0 * off[f"{s}_toe_tip"][1][1] -
                                                sole) for s in SIDES}
    return geo


def with_sole_points(P, names, geo):
    """(P, names) of a clip with the sole points of contact_geometry 'foot_<side>' appended (carried by each foot's
    pose on its frame points, _pose); unchanged for a foot whose frame points the clip lacks."""
    ix = {nm: i for i, nm in enumerate(names)}
    extra, extra_names = [], []
    for s in SIDES:
        g = geo.get(f"foot_{s}")
        pose = None if g is None else _pose(P, ix, g["frame"])
        if pose is None:
            continue
        R, t = pose
        for k, x in g["sole"].items():
            extra.append(np.einsum("nij,j->ni", R, x) + t)
            extra_names.append(k)
    if not extra:
        return P, list(names)
    return np.concatenate([P, np.stack(extra, 1)], 1), list(names) + extra_names


def _turn_onto(u, w):
    """(N, 3, 3) the least rotation turning the directions u (N, 3) onto w (N, 3) (Rodrigues; the identity where
    either is null)."""
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12)
    w = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-12)
    k = np.cross(u, w)
    s, c = np.linalg.norm(k, axis=-1), (u * w).sum(-1)
    K = np.zeros((len(u), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(0, 2, 1)
    f = np.where(s > 1e-12, (1.0 - c) / np.maximum(s * s, 1e-24), 0.0)
    return np.eye(3)[None] + K + f[:, None, None] * (K @ K)


def sole_cloud(P, ix, geo, side):
    """(N, M, 3) mm of the sole cloud of a foot (contact_geometry 'foot_<side>' 'cloud') at every frame: the
    ankle's vertices and the toe block's behind the toe-block joint carried by the foot's pose on its frame points
    (_pose, as the sole points of with_sole_points), the toe block's ahead of that joint ('cloud_toe_bend') turned
    about it by the toes' bend (the least turn of the foot's rest toe-block joint to toe tip line onto the clip's own;
    carried by the foot's pose when the clip lacks the toe tip); None when the clip lacks the foot's frame points or
    the geometry its cloud."""
    g = (geo or {}).get(f"foot_{side}") or {}
    if "cloud" not in g:
        return None
    pose = _pose(P, ix, g["frame"])
    if pose is None:
        return None
    R, t = pose
    X = g["cloud"] @ R.transpose(0, 2, 1) + t[:, None]                  # (N, M, 3), as R @ x + t
    toe, tip, jt = g.get("cloud_toe_bend", g["cloud_toe"]), f"{side}_toe_tip", f"{side}_foot"
    if toe.any() and tip in ix and jt in ix:
        J = P[:, ix[jt]]
        Q = _turn_onto(np.einsum("nij,j->ni", R, g["toe_tip"] - g["toe_joint"]), P[:, ix[tip]] - J)
        X[:, toe] = J[:, None] + (g["cloud"][toe] - g["toe_joint"]) @ (Q @ R).transpose(0, 2, 1)
    return g["cloud_w"] @ X


def _sole_samples(faces, ids, n):
    """(S, M) the barycentric weights over the vertices ids (M,) of the points sampled on the faces whose three
    vertices are all in ids, each face cut n times along each edge (the vertices and the edge points shared by two
    faces kept once)."""
    pos = {int(k): i for i, k in enumerate(ids)}
    f = faces[np.isin(faces, ids).all(1)]
    rows = {}
    for tri in f:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                w = {}
                for vid, x in zip(tri, (i, j, n - i - j)):
                    if x:
                        w[pos[int(vid)]] = w.get(pos[int(vid)], 0) + x
                rows[tuple(sorted(w.items()))] = w
    W = np.zeros((len(rows), len(ids)))
    for r, w in enumerate(rows.values()):
        for k, x in w.items():
            W[r, k] = x / n
    return W


def _kabsch(A, B):
    """(R (N, 3, 3), t (N, 3)) with B[k] ~ R[k] @ A + t[k] for model points A (K, 3) and observed B (N, K, 3)."""
    ca, cb = A.mean(0), B.mean(1)
    H = np.einsum("ki,nkj->nij", A - ca, B - cb[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    D = np.zeros((len(B), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("nji,njk,nlk->nil", Vt, D, U)
    return R, cb - np.einsum("nij,j->ni", R, ca)


def signed_distance(points, surf):
    """(...,): the signed distance of points (..., 3) to a surface given as vertices and outward normals, in their
    unit (the nearest vertex, along its normal; negative inside). The surface keeps its search tree."""
    from scipy.spatial import cKDTree
    tree = surf.get("_tree")
    if tree is None:
        tree = surf["_tree"] = cKDTree(surf["v"])
    q = np.asarray(points, dtype=np.float64)
    _, i = tree.query(q.reshape(-1, 3))
    d = ((q.reshape(-1, 3) - surf["v"][i]) * surf["n"][i]).sum(-1)
    return d.reshape(q.shape[:-1])


def _runs(mask):
    """[(a, b)] frame ranges [a, b) of the runs of True in a (N,) bool mask."""
    m = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    d = np.flatnonzero(np.diff(m))
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def _standing_on_floor(x, h0, floor_mm, fps):
    """(N,) bool: a foot point (positions x (N, 3) mm) at contact height (CONTACT_TOL_MM of its rest height h0 above
    the floor) and moving slower than CONTACT_SPEED_M_S horizontally (central differences); (N, M) for M points
    (x (N, M, 3), h0 (M,))."""
    n = len(x)
    v = np.zeros(x.shape[:-1])
    if n >= 3:
        v[1:-1] = np.linalg.norm((x[2:] - x[:-2])[..., [0, 2]], axis=-1) / 2.0 * fps / 1000.0
        v[0], v[-1] = v[1], v[-2]
    return (x[..., 1] - floor_mm <= h0 + CONTACT_TOL_MM) & (v < CONTACT_SPEED_M_S)


def free_frames(n, fps, events):
    """(n,) bool: the frames outside the declared events (each span widened by one frame)."""
    tf = np.arange(n) / fps
    free = np.ones(n, dtype=bool)
    for a, b in events:
        free &= ~((tf >= a - 1.0 / fps) & (tf <= b + 1.0 / fps))
    return free


def rate_peaks(P, names, fps=FPS, events=()):
    """{reading: {deg_s, t_s, tol_deg_s, frame}} of a clip (positions P (N, K, 3) mm): the peak angular speed outside
    the declared events of each segment of RATE_SEGMENTS (the turn of its direction, central differences) and of each
    elbow's flexion (the angle between the upper arm and the forearm), with the most the export's rounding can add
    (tol_deg_s: each end of a segment of length L within half an EXPORT_STEP_MM per axis turns it by up to
    sqrt(3) EXPORT_STEP_MM / L per frame, so sqrt(3) EXPORT_STEP_MM fps / L rad/s on central differences; an elbow
    adds its two segments'). The readings of RATE_ON_TRUNK are taken in the trunk's frame (trunk_frame, its own
    rounding added to tol_deg_s) when the clip has the joints of that frame ('frame': 'trunk'), in the room
    otherwise ('frame': 'room'). {} for a clip of fewer than three frames or without the joints."""
    P = np.asarray(P, dtype=np.float64)
    ix = {nm: i for i, nm in enumerate(names)}
    n = len(P)
    if n < 3:
        return {}
    free = free_frames(n, fps, events)
    frame = trunk_frame(P, ix)

    def seg(a, b):
        d = P[:, ix[b]] - P[:, ix[a]]
        L = np.linalg.norm(d, axis=-1)
        return d / np.maximum(L, 1e-9)[:, None], float(np.median(L))

    def tol(L):
        return float(np.degrees(np.sqrt(3.0) * EXPORT_STEP_MM * fps / max(L, 1e-9)))

    def peak(w, tl):
        w = np.where(free, w, 0.0)
        k = int(w.argmax())
        return {"deg_s": round(float(w[k]), 1), "t_s": round(k / fps, 3), "tol_deg_s": round(tl, 1)}
    out, segs = {}, {}
    for name, (a, b) in RATE_SEGMENTS.items():
        if a not in ix or b not in ix:
            continue
        u, L = segs[name] = seg(a, b)
        tl, where = tol(L), "room"
        if frame is not None and name in RATE_ON_TRUNK:
            u = np.einsum("nij,nj->ni", frame[0], u)
            tl, where = tl + tol(frame[1]) + tol(frame[2]), "trunk"
        w = np.zeros(n)
        w[1:-1] = np.degrees(np.arccos(np.clip((u[2:] * u[:-2]).sum(-1), -1.0, 1.0))) * fps / 2.0
        out[name] = dict(peak(w, tl), frame=where)
    for s in SIDES:
        if f"{s}_upper_arm" not in segs or not all(f"{s}_{j}" in ix for j in ("elbow", "wrist")):
            continue
        (ua, La), (fa, Lf) = segs[f"{s}_upper_arm"], seg(f"{s}_elbow", f"{s}_wrist")
        ang = np.degrees(np.arccos(np.clip((ua * fa).sum(-1), -1.0, 1.0)))
        w = np.zeros(n)
        w[1:-1] = np.abs(ang[2:] - ang[:-2]) * fps / 2.0
        out[f"{s}_elbow"] = peak(w, tol(La) + tol(Lf))
    return out


def trunk_frame(P, ix):
    """(R (N, 3, 3), trunk length mm, shoulder width mm) of the trunk's frame on each frame of the positions P: the
    rows of R its axes in the room, y the pelvis to neck axis, x the right to left shoulder line made orthogonal to
    y, z = x cross y (the lengths are the medians, for the rounding of rate_peaks); None without these joints."""
    need = ("pelvis", "neck", "left_shoulder", "right_shoulder")
    if not all(k in ix for k in need):
        return None
    y = P[:, ix["neck"]] - P[:, ix["pelvis"]]
    x = P[:, ix["left_shoulder"]] - P[:, ix["right_shoulder"]]
    Lt, Ls = float(np.median(np.linalg.norm(y, axis=-1))), float(np.median(np.linalg.norm(x, axis=-1)))
    y = y / np.maximum(np.linalg.norm(y, axis=-1), 1e-9)[:, None]
    x = x - (x * y).sum(-1)[:, None] * y
    x = x / np.maximum(np.linalg.norm(x, axis=-1), 1e-9)[:, None]
    return np.stack([x, y, np.cross(x, y)], 1), Lt, Ls


def rate_kind(reading):
    """The kind of a rate reading (RATE_CAPS_DEG_S): 'left_upper_arm' is an 'upper_arm', 'trunk' a 'trunk'."""
    return reading.split("_", 1)[1] if reading.split("_", 1)[0] in SIDES else reading


def rates_over(rates, level0_rates):
    """[{reading, deg_s, level0_deg_s, cap_deg_s, t_s}] of a symptom clip's rate readings (rate_peaks) above both
    their kind's cap (RATE_CAPS_DEG_S) and the level 0's own peak of the same reading by more than the two readings'
    rounding (tol_deg_s): a movement faster than the plan's bound that its level 0 does not already make."""
    out = []
    for k, r in rates.items():
        cap, r0 = RATE_CAPS_DEG_S[rate_kind(k)], level0_rates.get(k)
        if r0 is None or r["deg_s"] <= cap or r["deg_s"] <= r0["deg_s"] + r["tol_deg_s"] + r0["tol_deg_s"]:
            continue
        out.append({"reading": k, "deg_s": r["deg_s"], "level0_deg_s": r0["deg_s"], "cap_deg_s": cap,
                    "t_s": r["t_s"]})
    return sorted(out, key=lambda x: -x["deg_s"] / x["cap_deg_s"])


def check_continuity(P, names, fps=FPS, events=(), floor_mm=None, foot_rest_mm=None, level0_rates=None,
                     clip_id="", slides=None):
    """The continuity check of a clip (positions P (N, K, 3) mm): the acceleration of every point (second
    differences) above ACCEL_MAX_M_S2 on a frame outside the declared events (each span widened by one frame), and,
    with the floor and the foot points' rest heights, a foot point at contact height on two frames in a row and
    planted on the frame before (at contact height, its step into the first frame under CONTACT_SPEED_M_S on the
    clock), that moves more than FOOT_JUMP_MM between the two, outside the events: a planted foot snapping, not a
    heel strike (not planted before) or a lift into the swing (it leaves contact height or speeds up over frames).
    The planted frame is read on its own step, not on central differences, so that a landing step just before it
    does not hide the snap that follows; nothing is asked of the frames after the snap, so that a foot that snaps
    and then skates on the floor (a pivot of a turn) is read too (wave SH1d review of lane page). With the level 0's
    rate readings (rate_peaks), a symptom clip also fails on a segment turning faster than the plan's bound and than
    its level 0 (rates_over: the ballistic chorea and dyskinesias of PLAN M9, the whiplash of the failed rises of
    M10, which stay under ACCEL_MAX_M_S2); a clip of RATE_EXEMPT is read on its rates and not judged on them. The
    foot jumps leave out the frames of a foot's declared slides (slides: slide_spans, a dragged foot)."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    out = {"ok": True, "accel_max_m_s2": 0.0, "accel_point": None, "accel_t_s": None, "accel_frames_over": 0,
           "foot_jump_max_mm": 0.0, "foot_jump_point": None, "foot_jump_t_s": None, "foot_jumps": 0,
           "events_left_out": len(events), "rates_over": [],
           "rule": f"a point above {ACCEL_MAX_M_S2:g} m/s2 outside the declared jerk events, a foot point standing "
                   f"on the floor jumping more than {FOOT_JUMP_MM:g} mm in one frame, or a segment of a symptom clip "
                   f"turning faster than its bound and than its level 0"}
    tf = np.arange(n) / fps
    free_f = free_frames(n, fps, events)                                     # the frames outside the events
    rates = rate_peaks(P, names, fps, events)
    out["rates_deg_s"] = {k: r["deg_s"] for k, r in rates.items()}
    if level0_rates is not None:
        why = skate_exempt(clip_id, RATE_EXEMPT)
        if why:
            out["rates_exempt"] = why
        else:
            out["rates_over"] = rates_over(rates, level0_rates)
    if n >= 3:
        acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1) * fps * fps / 1000.0     # (N-2, K) m/s2
        t = tf[1:-1]
        acc = np.where(free_f[1:-1, None], acc, 0.0)
        k, j = np.unravel_index(int(acc.argmax()), acc.shape)
        out.update(accel_max_m_s2=round(float(acc[k, j]), 1), accel_point=names[j], accel_t_s=round(float(t[k]), 3),
                   accel_frames_over=int((acc > ACCEL_MAX_M_S2).any(1).sum()))
    if floor_mm is not None and foot_rest_mm and n >= 3:
        ix = {nm: i for i, nm in enumerate(names)}
        best = (0.0, None, None)
        for p, h0 in foot_rest_mm.items():
            if p not in ix:
                continue
            x = P[:, ix[p]]
            free_p = free_f & ~slide_mask(n, fps, slides, p.split("_", 1)[0])
            height = x[:, 1] - floor_mm <= h0 + CONTACT_TOL_MM
            step = np.linalg.norm(np.diff(x, axis=0), axis=-1)               # step k: frame k to k + 1
            planted = height[:-1] & (step < CONTACT_SPEED_M_S * 1000.0 / fps)   # frame k at height, still into k+1
            before = np.r_[False, planted[:-1]]                              # for step k: frame k - 1 planted
            jump = np.where(height[:-1] & height[1:] & before & free_p[:-1] & free_p[1:], step, 0.0)
            out["foot_jumps"] += int((jump > FOOT_JUMP_MM).sum())
            k = int(jump.argmax())
            if jump[k] > best[0]:
                best = (float(jump[k]), p, (k + 1) / fps)
        out.update(foot_jump_max_mm=round(best[0], 1), foot_jump_point=best[1],
                   foot_jump_t_s=None if best[2] is None else round(best[2], 3))
    out["ok"] = out["accel_frames_over"] == 0 and out["foot_jumps"] == 0 and not out["rates_over"]
    return out


def skate_exempt(clip_id, table=None):
    """The reason a clip is read but not judged by the skate check (SKATE_EXEMPT, or another table of the same
    form), None for any other clip."""
    for pat, why in (SKATE_EXEMPT if table is None else table):
        if re.search(pat, clip_id or ""):
            return why
    return None


def skate_rest_mm(geo):
    """{point: rest height above the sole, mm} of the points the skate check reads (contact_geometry): the foot
    points (FOOT_POINTS) and the toe tips, the only point a plantarflexed foot touches the floor with."""
    return {**geo["foot_rest_mm"], **geo.get("toe_rest_mm", {})}


def hold_spans(truth):
    """[[start_s, end_s]] of the freezing holds a clip's truth declares (freezing.py: params['episodes'], each
    'hold_s', the span the body stands frozen between its ramps), on the clip's clock; [] for any other clip. The
    skate check judges a clip of SKATE_EXEMPT inside these spans (wave SH1d)."""
    out = []
    for e in (((truth or {}).get("params") or {}).get("episodes") or []):
        h = e.get("hold_s") if isinstance(e, dict) else None
        if isinstance(h, (list, tuple)) and len(h) == 2:
            out.append([round(float(h[0]), 4), round(float(h[1]), 4)])
    return out


def slide_spans(truth, rate):
    """{side: [[start_s, end_s]]} of the foot slides a clip's truth declares, on the clip's clock (rate: the body's
    rate, the truth's arrays are on its clock): gait.py drags the foot of each side of
    params['gait_params']['drag_sides'] on the floor over every swing of that side (a deliberate skate, read back
    under params['deliberate_skate']), the frames where truth['arrays']['stance'] holds that side out of stance (its
    columns in the order of SIDES); {} for a clip that declares none. The skate check and the foot jumps of the
    continuity check leave these spans out (wave SH1d)."""
    t = truth or {}
    sides = ((t.get("params") or {}).get("gait_params") or {}).get("drag_sides") or ()
    st = (t.get("arrays") or {}).get("stance")
    if not sides or st is None:
        return {}
    st = np.asarray(st, dtype=bool)
    if st.ndim != 2 or st.shape[1] != len(SIDES):
        return {}
    return {sd: [[round(a / float(rate), 4), round(b / float(rate), 4)] for a, b in _runs(~st[:, SIDES.index(sd)])]
            for sd in SIDES if sd in sides}


def slide_mask(n, fps, slides, side):
    """(n,) bool: the frames of a side's declared slides (slide_spans), each widened by SLIDE_PAD_FRAMES."""
    tf = np.arange(n) / fps
    m = np.zeros(n, dtype=bool)
    pad = SLIDE_PAD_FRAMES / fps
    for a, b in (slides or {}).get(side, ()):
        m |= (tf >= a - pad) & (tf <= b + pad)
    return m


def _foot_yaw_deg(P, ix, side, a, b):
    """The turn of a foot over the frames [a, b) (degrees): the range of the horizontal direction of its heel sole
    to toe tip line (the heel joint or the toe joint when the sole points are not there); None without them."""
    h = next((f"{side}_{k}" for k in ("heel_sole", "heel") if f"{side}_{k}" in ix), None)
    t = next((f"{side}_{k}" for k in ("toe_tip", "foot") if f"{side}_{k}" in ix), None)
    if h is None or t is None or b <= a:
        return None
    v = P[a:b, ix[t]][:, [0, 2]] - P[a:b, ix[h]][:, [0, 2]]
    return float(np.ptp(np.degrees(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))))


def _masked_travel(x, m):
    """(M,) the diagonal of the horizontal bounding box of each of M points (x (n, M, 3)) over its frames in m
    (n, M); inf for a point with no frame."""
    xz = x[..., [0, 2]]
    lo = np.where(m[..., None], xz, np.inf).min(0)
    hi = np.where(m[..., None], xz, -np.inf).max(0)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(hi - lo, axis=-1)
    return np.where(m.any(0), d, np.inf)


def _least_travel(Y, m, need):
    """(travel mm, index) of the point of Y (n, K, 3) that travels least (_masked_travel) over its own frames of m
    (n, K) among those with need frames at least; (inf, None) when none has."""
    t = np.where(m.sum(0) >= need, _masked_travel(Y, m), np.inf)
    k = int(t.argmin())
    return (float(t[k]), k) if np.isfinite(t[k]) else (np.inf, None)


def _skate_read(P, ix, fps, floor_mm, foot_rest_mm, k0=0, slides=None, geo=None, detail=None):
    """The contacts of check_skate on the positions P: {worst: (travel mm, point, [start_s, end_s], tolerance mm) of the
    contact that travels most, over: contacts over their tolerance, read: contacts read, turning: [read, over, largest
    travel mm] of the contacts in which the foot turns by TURN_YAW_DEG at least, straight: the same of the others,
    slide_frames: the frames of declared slides left out}; k0: the frame P starts at on the clip's clock (for the spans
    and the slides). A contact is a run of frames on which a named point of foot_rest_mm stands (_standing_on_floor),
    its first and last frames left out; it reads the point that travels least over its own frames among those that carry
    it on SKATE_POINT_SHARE of the contact at least. Without geo the named points carry on their standing frames and the
    tolerance is SKATE_MM; with geo (contact_geometry) the skin points carry (the named ones but the toe-block joint,
    and the sole cloud: sole_cloud, its points named by their place along the foot, SOLE_REGIONS) on the frames they
    touch (SKATE_TOUCH_MM), and the tolerance is SKATE_MM plus the cloud's covering radius times the foot's turn.
    detail: a list that receives one record per contact read (side, span_s, frames, yaw_deg, travel_mm, point, tol_mm,
    named_travel_mm: the reading before the sole cloud, the named points on their standing frames, and
    standing_travel_mm: the skin points on their standing frames, without SKATE_TOUCH_MM)."""
    worst, n_bad, n_all = None, 0, 0
    turn, straight, left_out = [0, 0, 0.0], [0, 0, 0.0], 0
    cover = (geo or {}).get("cover_mm")
    for s in SIDES:
        sl = slide_mask(k0 + len(P), fps, slides, s)[k0:]
        left_out += int(sl.sum())
        names = [p for p in foot_rest_mm if p.startswith(f"{s}_") and p in ix]
        if not names:
            continue
        X = np.stack([P[:, ix[p]] for p in names], 1)
        on = _standing_on_floor(X, np.array([foot_rest_mm[p] for p in names]), floor_mm, fps) & ~sl[:, None]
        C = sole_cloud(P, ix, geo, s) if geo is not None else None
        Y, Ystand, pts = X, on, list(names)
        Ycar = on
        if C is not None:
            g = geo[f"foot_{s}"]
            skin = [k for k, p in enumerate(names) if p not in SK.JOINT_NAMES]
            Y = np.concatenate([X[:, skin], C], 1)
            Ystand = np.concatenate([on[:, skin],
                                     _standing_on_floor(C, g["cloud_rest_mm"], floor_mm, fps) & ~sl[:, None]], 1)
            low = np.where(Ystand, Y[..., 1], np.inf).min(1, keepdims=True)
            Ycar = Ystand & (Y[..., 1] <= low + SKATE_TOUCH_MM)
            pts = [names[k] for k in skin] + [f"{s}_sole_{r}" for r in g["cloud_region"]]
        for a, b in _runs(on.any(1)):
            if b - a < MIN_CONTACT_FRAMES + 2:
                continue
            u, v = a + 1, b - 1                            # the landing and the lift left out
            need = max(MIN_CONTACT_FRAMES, int(np.ceil(SKATE_POINT_SHARE * (v - u))))
            trav, k = _least_travel(Y[u:v], Ycar[u:v], need)
            if k is None:
                continue
            p = pts[k]
            n_all += 1
            yaw = _foot_yaw_deg(P, ix, s, u, v)
            tol = SKATE_MM + (float(cover * np.radians(yaw)) if cover is not None and yaw is not None else 0.0)
            bad = trav > tol
            n_bad += bad
            grp = turn if yaw is not None and yaw >= TURN_YAW_DEG else straight
            grp[0], grp[1], grp[2] = grp[0] + 1, grp[1] + int(bad), max(grp[2], trav)
            span = [round((k0 + a) / fps, 3), round((k0 + b) / fps, 3)]
            if worst is None or trav > worst[0]:
                worst = (trav, p, span, tol)
            if detail is not None:
                tn, _ = _least_travel(X[u:v], on[u:v], need)
                ts, _ = _least_travel(Y[u:v], Ystand[u:v], need)
                detail.append({"side": s, "span_s": span, "frames": [k0 + a, k0 + b],
                               "yaw_deg": None if yaw is None else round(yaw, 1), "travel_mm": round(trav, 2),
                               "point": p, "tol_mm": round(float(tol), 2),
                               "named_travel_mm": round(tn, 2) if np.isfinite(tn) else None,
                               "standing_travel_mm": round(ts, 2) if np.isfinite(ts) else None})
    return {"worst": worst or (0.0, None, None, SKATE_MM), "over": int(n_bad), "read": n_all, "turning": turn,
            "straight": straight, "slide_frames": left_out}


def _split(g):
    """A group of contacts of _skate_read ([read, over, largest travel mm]) as a record."""
    return {"contacts": g[0], "contacts_over": g[1], "travel_max_mm": round(g[2], 1)}


def check_skate(P, names, fps, floor_mm, foot_rest_mm, clip_id="", holds_s=(), slides=None, geo=None):
    """The skate check, read per foot (wave SH1d review of lane page): a contact of a foot is a run of frames in which
    at least one of its points (the keys of foot_rest_mm on that side: the sole points and the toe tip, skate_rest_mm)
    stands on the floor (_standing_on_floor), MIN_CONTACT_FRAMES frames at least once its first and last frames are left
    out; each point carrying the foot on SKATE_POINT_SHARE of the contact's frames at least travels the diagonal of its
    horizontal bounding box over its own frames, and the contact reads the point that travelled least (_skate_read).
    With geo (contact_geometry, as skeleton_checks passes it) the points that carry are the skin points of the foot, its
    sole cloud among them (sole_cloud), on the frames they touch the floor (SKATE_TOUCH_MM), and a contact fails above
    SKATE_MM plus the cloud's covering radius times the foot's turn in radians (its 'tol_mm'); without geo, the named
    points on their standing frames, above SKATE_MM. A foot that rolls from the heel to the toes or turns about a point
    of its sole keeps that point still and passes; a foot that glides, slips at its pivot, or touches with one point
    only that glides, fails. A contact in which no point carries long enough is not read. The frames of a foot's
    declared slides (slides: slide_spans, a dragged foot) are no contact. The contacts are also counted apart by the
    foot's turn (TURN_YAW_DEG: 'turning' and 'straight'). 'travel_max_mm', 'point', 'span_s' and 'tol_mm' are those of
    the contact that travels most. A clip of SKATE_EXEMPT is read and not judged (ok None, its reason under 'exempt'),
    but inside its declared freezing holds (holds_s, hold_spans: each hold read as a clip of its own, 'holds')."""
    P = np.asarray(P, dtype=np.float64)
    ix = {nm: i for i, nm in enumerate(names)}
    r = _skate_read(P, ix, fps, floor_mm, foot_rest_mm, slides=slides, geo=geo)
    worst = r["worst"]
    out = {"ok": r["over"] == 0, "travel_max_mm": round(worst[0], 1), "point": worst[1], "span_s": worst[2],
           "tol_mm": round(worst[3], 1),
           "contacts": r["read"], "contacts_over": r["over"], "turning": _split(r["turning"]),
           "straight": _split(r["straight"]), "slide_frames_left_out": r["slide_frames"],
           "rule": SKATE_RULE}
    why = skate_exempt(clip_id)
    if why:
        out.update(ok=None, exempt=why)
        spans = [(int(np.ceil(a * fps)), min(len(P), int(np.floor(b * fps)))) for a, b in (holds_s or ())]
        spans = [(a, b) for a, b in spans if b > a]
        if spans:
            hw, hb, hn = None, 0, 0
            for a, b in spans:
                h = _skate_read(P[a:b], ix, fps, floor_mm, foot_rest_mm, a, slides, geo)
                hb, hn = hb + h["over"], hn + h["read"]
                if h["read"] and (hw is None or h["worst"][0] > hw[0]):
                    hw = h["worst"]
            hw = hw or (0.0, None, None, SKATE_MM)
            out.update(ok=hb == 0, holds={"spans_s": [list(x) for x in holds_s], "contacts": hn,
                                          "contacts_over": hb, "travel_max_mm": round(hw[0], 1), "point": hw[1],
                                          "span_s": hw[2], "tol_mm": round(hw[3], 1)})
            out["exempt"] = why + "; judged inside the declared freezing holds"
    return out


def skate_contact_list(P, names, fps, floor_mm, geo, slides=None):
    """[{side, span_s, frames, yaw_deg, travel_mm, point, named_travel_mm, named_point, kind}] of every contact of a
    clip (positions P mm and their names, as exported) that the skate check reads with the sole cloud
    (_skate_read with geo), the declared slides of a dragged foot read too (the check leaves them out): kind 'slide'
    for a contact that overlaps a declared slide of its foot, 'turning' for one in which the foot turns by
    TURN_YAW_DEG at least, 'straight' for the others (scripts/skeleton_page_checks.py, the skate tolerance)."""
    Pf, nf = with_sole_points(np.asarray(P, dtype=np.float64), names, geo)
    ix = {nm: i for i, nm in enumerate(nf)}
    rows = []
    _skate_read(Pf, ix, fps, floor_mm, skate_rest_mm(geo), geo=geo, detail=rows)
    for r in rows:
        a, b = r["frames"]
        y = r["yaw_deg"]
        r["kind"] = ("slide" if slide_mask(len(Pf), fps, slides, r["side"])[a:b].any() else
                     "turning" if y is not None and y >= TURN_YAW_DEG else "straight")
    return rows


def check_statue(P, fps=FPS, supine=False):
    """The statue check: the windows of STATUE_WINDOW_S in which every one of the 22 body joints stays under
    STATUE_MM (the diagonal of its bounding box over the window); a clip lying on the couch is not read."""
    P = np.asarray(P, dtype=np.float64)[:, :SK.N_BODY]
    w = int(round(STATUE_WINDOW_S * fps))
    out = {"ok": True, "statue_s": 0.0, "share": 0.0, "spans_s": [], "near_still_s": 0.0, "supine": bool(supine),
           "rule": f"a {STATUE_WINDOW_S:g} s window in which no body joint moves by one {STATUE_MM:g} mm step of "
                   f"the export (the body frozen)"}
    if supine or len(P) < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(P, w, axis=0)          # (N - w + 1, 22, 3, w)
    diag = np.linalg.norm(win.max(-1) - win.min(-1), axis=-1).max(-1)     # (N - w + 1,) the largest joint's

    def covered(still):
        cover = np.zeros(len(P), dtype=bool)
        for k in np.flatnonzero(still):
            cover[k:k + w] = True
        return cover
    still = diag < STATUE_MM
    cover = covered(still)
    out.update(ok=not still.any(), statue_s=round(float(cover.sum()) / fps, 2),
               share=round(float(cover.mean()), 3),
               spans_s=[[round(a / fps, 3), round(b / fps, 3)] for a, b in _runs(cover)],
               near_still_s=round(float(covered(diag < STATUE_PLAN_MM).sum()) / fps, 2))
    return out


def foot_floor_mm(P, names, foot_rest_mm, fps=FPS):
    """{side: mm}: the floor each foot of a clip stands on (the positions P (N, K, 3) mm with the sole points,
    with_sole_points): on each standing frame (standing_frames) a foot's sole height is the lowest of its points
    (FOOT_POINTS) less its height above the sole (foot_rest_mm); the foot reads the median over its standing frames
    that lie within CONTACT_TOL_MM of their FLOOR_LOW_PCT percentile (a tap down to the floor does not move it, and
    a foot held still in the air, the four foot joints still with the sole above the floor, is not read as
    standing: wave SH1d review of lane page). {} with fewer than MIN_STANDING_FRAMES standing frames."""
    P = np.asarray(P, dtype=np.float64)
    st = standing_frames(P[:, :SK.N_BODY] / 1000.0, fps)
    if st.sum() < MIN_STANDING_FRAMES:
        return {}
    ix = {nm: i for i, nm in enumerate(names)}
    out = {}
    for s in SIDES:
        pts = [p for p in FOOT_POINTS[s] if p in ix and p in foot_rest_mm]
        if pts:
            h = np.min(np.stack([P[st, ix[p], 1] - foot_rest_mm[p] for p in pts], 1), 1)
            out[s] = float(np.median(h[h <= np.percentile(h, FLOOR_LOW_PCT) + CONTACT_TOL_MM]))
    return out


def check_pair_floor(P, names, P_level0, names_level0, foot_rest_mm, fps=FPS):
    """The pair floor check: each foot's floor in a symptom clip (foot_floor_mm) against the same foot's in its
    level 0; the foot that changed least must lie within PAIR_FLOOR_TOL_MM, so that a foot the symptom lifts,
    plantarflexes or turns leaves the reading to the other one, and a body raised or sunk moves both. Not read (ok
    None) when either clip never stands still."""
    a, b = (foot_floor_mm(P, names, foot_rest_mm, fps), foot_floor_mm(P_level0, names_level0, foot_rest_mm, fps))
    both = [s for s in SIDES if s in a and s in b]
    if not both:
        return {"ok": None, "clip_mm": None, "level0_mm": None, "diff_mm": None,
                "rule": "not read: no standing frames in the clip or its level 0"}
    d = {s: a[s] - b[s] for s in both}
    s = min(both, key=lambda k: abs(d[k]))
    return {"ok": abs(d[s]) <= PAIR_FLOOR_TOL_MM, "foot": s, "clip_mm": round(a[s], 1), "level0_mm": round(b[s], 1),
            "diff_mm": round(d[s], 1), "diff_by_foot_mm": {k: round(v, 1) for k, v in d.items()},
            "rule": f"the sole of the foot that changed least (the median over its standing frames on the floor) "
                    f"within {PAIR_FLOOR_TOL_MM:g} mm of the same foot in the level 0"}


def touch_readings(truth):
    """{target: {side: mm}} of the declared touches a clip's module measures on the posed mesh (MESH_TOUCHES: the
    'touch_mm' of the validity entry named there, rounded to 0.01 mm); {} for a clip that carries none."""
    v = (truth or {}).get("validity") or {}
    out = {}
    for target, entry in MESH_TOUCHES.items():
        t = v.get(entry).get("touch_mm") if isinstance(v.get(entry), dict) else None
        if isinstance(t, dict):
            out[target] = {s: round(float(x), 2) for s, x in t.items() if s in SIDES and x is not None}
    return {k: x for k, x in out.items() if x}


def declared_touches(clip_id, root_id=None):
    """[(target, hands, kind, words, own)] of a clip: its own DECLARED_TOUCHES and its level 0's 'rest' touches
    (own False: allowed, not required)."""
    out = [(t, h, k, w, True) for pat, t, h, k, w in DECLARED_TOUCHES if re.search(pat, clip_id)]
    if root_id and root_id != clip_id:
        out += [(t, h, k, w, False) for pat, t, h, k, w in DECLARED_TOUCHES
                if k == "rest" and re.search(pat, root_id) and not any(o[0] == t for o in out)]
    return out


def _to_local(R, t, Q):
    """Points Q (N, M, 3) in the frames (R, t): R^T (Q - t)."""
    return np.einsum("nji,nmj->nmi", R, Q - t[:, None])


def _pose(P, ix, model_points):
    """(R, t) of a rigid part at every frame: its rest points {name: (3,)} fitted onto the observed ones (_kabsch);
    None when the clip lacks one of them."""
    if not all(k in ix for k in model_points):
        return None
    A = np.stack([model_points[k] for k in model_points])
    return _kabsch(A, np.stack([P[:, ix[k]] for k in model_points], 1))


def palm_skin(P, ix, geo, side):
    """(N, V, 3) mm: the palm's skin of a hand at every frame (contact_geometry 'palm_<side>', carried by the hand's
    pose on its frame points), None when the clip lacks them."""
    g = geo.get(f"palm_{side}")
    if g is None:
        return None
    pose = _pose(P, ix, dict(zip(HAND_FRAME[side], g["frame"])))
    if pose is None:
        return None
    R, t = pose
    return np.einsum("nij,mj->nmi", R, g["v"]) + t[:, None]


def hip_flexion_deg(P, ix, leg):
    """(N,) deg: the angle between a femur (hip to knee) and the trunk pointing down (neck to pelvis); None when the
    clip lacks one of the four joints. 0 standing, about 80 seated on the catalogue's chairs."""
    need = ("pelvis", "neck", f"{leg}_hip", f"{leg}_knee")
    if not all(k in ix for k in need):
        return None
    a = P[:, ix[f"{leg}_knee"]] - P[:, ix[f"{leg}_hip"]]
    b = P[:, ix["pelvis"]] - P[:, ix["neck"]]
    c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def check_self_contact(P, names, geo, clip_id="", root_id=None, fps=FPS, mesh_touches=None):
    """The self-contact check (positions P (N, K, 3) mm): a hand point inside the head (a joint centre inside it, a
    skin point, a fingertip landmark or the palm's skin, deeper than SKIN_TOL_MM), within HAND_HAND_MM of the other
    hand (a joint centre against a joint or a fingertip of the other hand; two fingertips within SKIN_TOL_MM, where
    they overlap; not where the clip declares a touch of the hands), inside a thigh (the same limits),
    each a 'contact'; and each touch on the thighs the clip declares ('rest' or 'reach', declared_touches
    'own') whose palm stays farther than TOUCH_GAP_MM from the thigh's skin, a 'gap'; a 'reach' touch the clip's
    module measures on the posed mesh (mesh_touches, the data file's 'touches_mesh_mm': MESH_TOUCHES) is judged on
    that reading, the rigid thigh's beside it ('rigid_gap_mm'). The surfaces are those of
    contact_geometry (geo), carried by the head's pose and the femur frames read on the positions; a point is read
    against the head above its rim and against a thigh inside its span of the femur. A thigh is passed by THIGH_TOL_MM
    (the rigid thigh's error against the posed mesh), a declared touch's gap is read on the frames with the hip
    flexed within THIGH_FLEX_DEG only ('gaps_not_read' when none is), and the thigh contacts of a clip of
    CONTACT_EXEMPT are read and not judged ('contacts_exempt')."""
    P = np.asarray(P, dtype=np.float64)
    ix = {nm: i for i, nm in enumerate(names)}
    touches = declared_touches(clip_id, root_id)
    allowed = {t for t, _, _, _, _ in touches}
    hand_j = {s: [ix[p] for p in HAND_JOINTS[s] if p in ix] for s in SIDES}
    hand_s = {s: [ix[p] for p in HAND_SKIN[s] if p in ix] for s in SIDES}
    palms = {s: palm_skin(P, ix, geo, s) for s in SIDES}
    out = {"ok": True, "contacts": [], "gaps": [], "touches": [w for _, _, _, w, own in touches if own],
           "rule": "a hand point inside the head or a thigh, within the hand-to-hand or thigh limits, or a declared "
                   "touch on the thighs that the palm does not reach"}
    found = {}

    def note(kind, val, k, what):
        if kind not in found or val < found[kind][0]:
            found[kind] = (val, k, what)

    def groups(s):          # (points (N, M, 3), names, skin) of a hand: its joints, its tip landmarks, its palm's skin
        g = [(P[:, hand_j[s]], [names[i] for i in hand_j[s]], False),
             (P[:, hand_s[s]], [names[i] for i in hand_s[s]], True)]
        if palms[s] is not None:
            g.append((palms[s], [f"{s}_palm"] * palms[s].shape[1], True))
        return [x for x in g if x[0].shape[1]]
    # the head
    pose = _pose(P, ix, geo["head"]["frame_points"])
    if pose is not None:
        R, t = pose
        for s in SIDES:
            for Q, qn, skin in groups(s):
                L = _to_local(R, t, Q)
                d = np.where(L[..., 1] >= geo["head"]["y_min"], signed_distance(L, geo["head"]), np.inf)
                k, m = np.unravel_index(int(d.argmin()), d.shape)
                if d[k, m] < (-SKIN_TOL_MM if skin else 0.0):
                    note("head", float(d[k, m]), k, qn[m])
    # the two hands
    if hand_j["left"] and hand_j["right"] and "hands" not in allowed:
        A, B = P[:, hand_j["left"] + hand_s["left"]], P[:, hand_j["right"] + hand_s["right"]]
        sa = np.r_[np.zeros(len(hand_j["left"]), bool), np.ones(len(hand_s["left"]), bool)]
        sb = np.r_[np.zeros(len(hand_j["right"]), bool), np.ones(len(hand_s["right"]), bool)]
        lim = np.where(sa[:, None] & sb[None], SKIN_TOL_MM, HAND_HAND_MM)    # two skin points: they overlap
        D = np.linalg.norm(A[:, :, None] - B[:, None], axis=-1)              # (N, a, b)
        D = np.where(D < lim[None], D, np.inf)
        k, a, b = np.unravel_index(int(D.argmin()), D.shape)
        if np.isfinite(D[k, a, b]):                 # the nearest pair of points under its limit, by name
            na = [names[i] for i in hand_j["left"] + hand_s["left"]]
            nb = [names[i] for i in hand_j["right"] + hand_s["right"]]
            note("hands", float(D[k, a, b]), k, f"{na[a]} and {nb[b]}")
    # the thighs
    thigh_gap = {s: np.full(len(P), np.inf) for s in SIDES}
    gap_read = np.zeros(len(P), dtype=bool)               # the frames a gap is read on (THIGH_FLEX_DEG)
    for leg in SIDES:
        g = geo.get(f"thigh_{leg}")
        need = [f"{leg}_{j}" for j in ("hip", "knee", "ankle", "foot")]
        if g is None or not all(x in ix for x in need):
            continue
        hip, knee, ankle, foot = (P[:, ix[x]] for x in need)
        E = femur_frame(hip, knee, ankle, foot)                                 # (N, 3, 3)
        flex = hip_flexion_deg(P, ix, leg)
        read = np.ones(len(P), dtype=bool) if flex is None else (
            (flex >= THIGH_FLEX_DEG[0]) & (flex <= THIGH_FLEX_DEG[1]))
        gap_read |= read
        for s in SIDES:
            for Q, qn, skin in groups(s):
                L = _to_local(E, hip, Q)
                span = (L[..., 0] >= THIGH_T_MIN * g["length"]) & (L[..., 0] <= g["length"])
                d = np.where(span, signed_distance(L, g), np.inf)
                lim = -THIGH_TOL_MM if skin else SKIN_TOL_MM - THIGH_TOL_MM
                k, m = np.unravel_index(int(d.argmin()), d.shape)
                if d[k, m] < lim:
                    note(f"thigh_{leg}", float(d[k, m]), k, qn[m])
                if qn[0].endswith("_palm"):
                    thigh_gap[s] = np.minimum(thigh_gap[s], np.where(read, d.min(1), np.inf))
    why = skate_exempt(clip_id, CONTACT_EXEMPT)
    for kind, (val, k, what) in sorted(found.items()):
        e = {"with": kind, "mm": round(val, 1), "t_s": round(int(k) / fps, 3), "point": what}
        if why and kind.startswith("thigh_"):              # read, not judged (CONTACT_EXEMPT)
            out.setdefault("contacts_exempt", []).append(e)
            out["exempt"] = why
        else:
            out["contacts"].append(e)
    # the declared touches on the thighs: the palm's skin within TOUCH_GAP_MM of a thigh's
    for target, hands, kind, words, own in touches:
        if not own or target != "thighs" or kind not in ("rest", "reach"):
            continue
        for s in hands:
            gap = thigh_gap[s][gap_read]
            mesh = ((mesh_touches or {}).get(target) or {}).get(s) if kind == "reach" else None
            if mesh is not None:                           # the module's own reading on the posed mesh (MESH_TOUCHES)
                rigid = float(np.min(gap)) if len(gap) else None
                out.setdefault("touches_on_mesh", []).append({
                    "target": target, "hand": s, "words": words, "mesh_gap_mm": round(float(mesh), 2),
                    "rigid_gap_mm": round(rigid, 1) if rigid is not None and np.isfinite(rigid) else None})
                if float(mesh) > TOUCH_GAP_MM:
                    out["gaps"].append({"target": target, "hand": s, "kind": kind, "words": words,
                                        "gap_mm": round(float(mesh), 2),
                                        "reading": "the smallest, read by the clip's module on the posed mesh"})
                continue
            if not len(gap):                               # no frame in the range the thigh is read in
                out.setdefault("gaps_not_read", []).append({"hand": s, "words": words})
                continue
            val = float(np.median(gap)) if kind == "rest" else float(np.min(gap))
            if val > TOUCH_GAP_MM:
                out["gaps"].append({"target": target, "hand": s, "kind": kind, "words": words,
                                    "gap_mm": round(val, 1) if np.isfinite(val) else None,
                                    "reading": (("median" if kind == "rest" else "smallest") + " over the frames "
                                                f"with the hip flexed {THIGH_FLEX_DEG[0]:g} to "
                                                f"{THIGH_FLEX_DEG[1]:g} deg" + (
                                                    "" if np.isfinite(val)
                                                    else ": the palm beside the thigh, not over it"))})
    out["ok"] = not out["contacts"] and not out["gaps"]
    return out


def page_roots(metas):
    """{clip: root} of the page's clips: a clip's level 0, then that clip's level 0, while the level 0 is on the page
    (a clip whose level 0 is not shown is its own root)."""
    by = {m["clip_id"]: m for m in metas}
    out = {}
    for c in by:
        r, seen = c, {c}
        while by[r].get("level0") in by and by[r]["level0"] not in seen:
            r = by[r]["level0"]
            seen.add(r)
        out[c] = r
    return out


def page_floor(m, root):
    """(floor_y_m, boxes) a clip is drawn with on the catalogue page (wave SH1d, PLAN M5): its root's floor and scene
    boxes (page_roots), so that a clip and its level 0 share the floor and the chair; its own when the root's scene
    kind differs."""
    if root is m or root.get("scene") != m.get("scene"):
        return m["floor"]["floor_y_m"], m["scene_boxes_m"]
    return root["floor"]["floor_y_m"], root["scene_boxes_m"]


def skeleton_checks(metas, blobs, geo=None, fps=FPS):
    """{clip: {check: result}} of SKELETON_CHECKS on the page's clips (the data files' positions, the floor the
    page draws (page_floor), the declared events 'events_s', the freezing holds 'holds_s', the level 0 through
    page_roots); geo: contact_geometry() by default."""
    geo = contact_geometry() if geo is None else geo
    by = {m["clip_id"]: m for m in metas}
    roots = page_roots(metas)
    pos, out = {}, {}

    def P(c):
        if c not in pos:
            pos[c] = blob_positions(blobs[c]).astype(np.float64)
        return pos[c]

    def names_of(m):
        return m["_names"] if "_names" in m else json.loads(gzip.decompress(blobs[m["clip_id"]]).decode("utf-8"))[
            "skeleton"]["names"]
    rates = {}

    def rates_of(c):                                       # a level 0's rate readings, read once
        if c not in rates:
            rates[c] = rate_peaks(P(c), names_of(by[c]), fps, by[c].get("events_s") or ())
        return rates[c]
    for m in metas:
        c = m["clip_id"]
        names = names_of(m)
        floor_mm = 1000.0 * page_floor(m, by[roots[c]])[0]
        Pf, nf = with_sole_points(P(c), names, geo)          # the sole points the foot checks read
        r = {"continuity": check_continuity(Pf, nf, fps, m.get("events_s") or (), floor_mm, geo["foot_rest_mm"],
                                            rates_of(roots[c]) if roots[c] != c else None, c, m.get("slides_s")),
             "self_contact": check_self_contact(P(c), names, geo, c, roots[c], fps, m.get("touches_mesh_mm")),
             "skate": check_skate(Pf, nf, fps, floor_mm, skate_rest_mm(geo), c, m.get("holds_s") or (),
                                  m.get("slides_s"), geo),
             "statue": check_statue(P(c), fps, bool(m.get("supine"))),
             "sinking": check_sinking(Pf, nf, floor_mm, fps, bool(m.get("supine")), c)}
        if roots[c] != c and not m.get("supine"):
            P0, n0 = with_sole_points(P(roots[c]), names_of(by[roots[c]]), geo)
            r["pair_floor"] = check_pair_floor(Pf, nf, P0, n0, geo["foot_rest_mm"], fps)
        for k, key in DECLARED_SPANS.items():              # data exported before the declared spans: read only
            if key not in m and r[k].get("ok") is not None:
                r[k].update(ok=None, not_judged=SPANS_UNKNOWN.format(key=key))
        out[c] = r
        if len(pos) > 8:                                   # keep the clip and its root, let the others go
            for k in [k for k in pos if k not in (c, roots[c])]:
                del pos[k]
    return out


DECLARED_SPANS = {"continuity": "events_s", "skate": "slides_s"}   # the check and the key of the declared spans
#   it leaves out; a data file without the key (exported before wave SH1d) leaves the check read and not judged,
#   since a declared jerk or a dragged foot would read as a defect (an empty list or map is a clip that declares none)
SPANS_UNKNOWN = ("the data file holds no declared spans ('{key}': exported before them), so a declared movement "
                 "would read as a defect: read, not judged")


def check_sinking(P, names, floor_mm, fps=FPS, supine=False, clip_id=""):
    """The sinking check (wave SH1d review of lane page): a sole point (the heel and lateral sole points of
    with_sole_points, vertices of the sole) more than CONTACT_TOL_MM under the floor the page draws on any frame;
    a clip lying on the couch is not read (ok None). The page floor is set from the lowest body joint (clip_floor),
    not from the soles, so a heel can sit under it while the joint stands on it. A clip of SINKING_EXEMPT is read
    and not judged (ok None, its reason under 'exempt')."""
    P = np.asarray(P, dtype=np.float64)
    pts = [i for i, nm in enumerate(names) if nm.endswith(("_heel_sole", "_lateral_sole"))]
    rule = f"a sole point more than {CONTACT_TOL_MM:g} mm under the floor the page draws"
    if supine or not pts:
        return {"ok": None, "depth_mm": None, "point": None, "t_s": None, "share": 0.0,
                "rule": "not read: " + ("a clip lying on the couch" if supine else "no sole points")}
    h = P[:, pts, 1] - floor_mm                                              # (N, M) mm above the floor
    k, j = np.unravel_index(int(h.argmin()), h.shape)
    under = (h < -CONTACT_TOL_MM).any(1)
    out = {"ok": not under.any(), "depth_mm": round(float(-h[k, j]), 1), "point": names[pts[j]],
           "t_s": round(k / fps, 3), "share": round(float(under.mean()), 3), "rule": rule}
    why = skate_exempt(clip_id, SINKING_EXEMPT)
    if why:
        out.update(ok=None, exempt=why)
    return out


def checks_failed(results, names=SKELETON_CHECKS):
    """{clip: [checks]} of the results (skeleton_checks) whose 'ok' is False, among names."""
    out = {}
    for c, r in results.items():
        bad = [k for k in names if r.get(k, {}).get("ok") is False]
        if bad:
            out[c] = bad
    return out


REPORTED_ONLY = ("legs",)   # validity entries written into the data without a refusal of their own: the leg-to-leg
#   contact of gait.validity (legcontact.validity; since the calibration of 2026-11-03 it enters gait.ok
#   (gait.LEGS_REFUSE), so a failure refuses the clip once, through 'gait')


def module_failures(truth):
    """[names] of the module's own validity checks a clip's truth fails (task SKELV2): every entry of
    truth['validity'] written as {'ok': False, ...} (the gait generator's 'gait': joint-limit table, stance slip,
    swing below the floor) but those of REPORTED_ONLY, and 'finite' when it is False."""
    v = (truth or {}).get("validity") or {}
    bad = [k for k, x in v.items() if isinstance(x, dict) and x.get("ok") is False and k not in REPORTED_ONLY]
    return bad + (["finite"] if v.get("finite") is False else [])


def reported_failures(truth):
    """[names] of the REPORTED_ONLY entries a clip's truth fails (written into its data file and the manifest)."""
    v = (truth or {}).get("validity") or {}
    return [k for k in REPORTED_ONLY if isinstance(v.get(k), dict) and v[k].get("ok") is False]


def refusal(truth, module_checks=False):
    """[names] of the checks that refuse a clip: render_mesh_video.failed_checks, and with module_checks (the
    catalogue set) the module's own validity (module_failures)."""
    bad = RMV.failed_checks(truth)
    if module_checks:
        bad = bad + [k for k in module_failures(truth) if k not in bad]
    return bad


SYNTH_DIR = os.path.join(ROOT, "mdfm", "synth")
EXPORT_CODE = ("mdfm/synth/exam.py", "mdfm/synth/skeleton.py", "mdfm/synth/viewer.py",   # read by every export;
               "mdfm/synth/catalogue.py", "mdfm/synth/physio.py")   # SH1: the background wraps every catalogue build


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _module_file(dotted):
    """The file of an mdfm module ('mdfm.synth.body': its .py, or its package's __init__.py), None otherwise."""
    parts = dotted.split(".")
    if parts[0] != "mdfm":
        return None
    base = os.path.join(ROOT, *parts)
    for p in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def _imports(path):
    """The dotted names a source file of the mdfm package imports, relative or absolute, lazy imports inside
    functions included (and 'pkg.name' for 'from pkg import name', which may be a submodule)."""
    pkg = os.path.relpath(path, ROOT).replace("\\", "/")[:-3].split("/")[:-1]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = pkg[:len(pkg) - node.level + 1] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def code_deps(path, skip=()):
    """[paths relative to the tree] of the mdfm files a module file builds with: itself, every mdfm module it imports,
    transitively, and their packages' __init__ files; the files in skip (absolute paths: the catalogue modules left
    out, which are never imported) are left out with what only they import."""
    skip = {os.path.abspath(p) for p in skip}
    seen, todo = set(), [os.path.abspath(path)]
    while todo:
        p = todo.pop()
        if p in seen or p in skip:
            continue
        seen.add(p)
        for dotted in _imports(p):
            parts = dotted.split(".")
            for i in range(1, len(parts) + 1):
                f = _module_file(".".join(parts[:i]))
                if f is not None and f not in seen:
                    todo.append(f)
    return sorted(os.path.relpath(p, ROOT).replace("\\", "/") for p in seen)


def deps_sha256(rel_paths):
    """SHA-256 over the files (relative path and bytes, in path order)."""
    h = hashlib.sha256()
    for rel in sorted(set(rel_paths)):
        h.update(rel.encode("utf-8") + b"\0")
        with open(os.path.join(ROOT, rel), "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()


def _build_fn(spec):
    fn = spec.build
    return getattr(fn, "func", fn)                         # a functools.partial builds through its function


class CodeState:
    """The digests of the code a set of specs builds with, at the time of the call (read right after the modules are
    imported for the build; read again before the page to find the files that became stale meanwhile). skip: the
    catalogue modules left out (never imported)."""

    def __init__(self, specs, skip=()):
        self.synth = RMV.code_record()["synth_code_sha256"]
        self.skip = [os.path.join(SYNTH_DIR, f"{m}.py") for m in skip]
        self._mod = {}
        for s in specs:
            self._module(inspect.getsourcefile(_build_fn(s)))

    def _module(self, path):
        if path not in self._mod:
            deps = sorted(set(code_deps(path, self.skip)) | set(EXPORT_CODE))
            self._mod[path] = (file_sha256(path), deps, deps_sha256(deps))
        return self._mod[path]

    def of(self, spec):
        """{module, module_file, module_sha256, deps, deps_sha256, synth_code_sha256} of a spec's build."""
        fn = _build_fn(spec)
        path = inspect.getsourcefile(fn)
        sha, deps, dsha = self._module(path)
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        return {"module": fn.__module__, "module_file": rel, "module_sha256": sha, "deps": deps,
                "deps_sha256": dsha, "synth_code_sha256": self.synth, "export_format": EXPORT_FORMAT,
                "rule": "module_sha256, deps_sha256 and export_format decide whether the file is stale (module "
                        "docstring)"}


STALE_KEYS = ("module", "module_sha256", "deps_sha256", "export_format")


def stale_reason(code_then, code_now):
    """None when a data file's code record matches the current code, else why it is stale."""
    if not code_then:
        return "no code record"
    why = [k for k in STALE_KEYS if code_then.get(k) != code_now[k]]
    return ", ".join(f"{k} changed" for k in why) or None


def read_code(path):
    """The code record of a data file (None without one)."""
    return read_data(path).get("code")


# ---------------------------------------------------------------- provenance: the reads of a motion-capture store
CAPTURE_EVENTS = ("open", "os.listdir", "os.scandir", "glob.glob", "shutil.copyfile", "mmap.__new__")


def capture_roots():
    """{label: normalized absolute root} of the folders that hold motion-capture takes or windows cut from them
    (the public gallery, 2026-11-10): the AMASS body store (its root and its target on F:), the AMASS caches and take
    index of this tree, MoveAtlas's datasets (every corpus, AMASS's raw archives among them), the gates-only store,
    the window stores and, since the open-body port's wave 7, the open body store (Anny fitted to open motion capture,
    mdfm.data.open_body_store). A build that opens any file or lists any folder under one of them reads captured
    motion."""
    from mdfm.data import body_store as BS
    from mdfm.data import open_body_store as OBS
    roots = {"body_store": BS.store_root(), "body_store_target": BS.STORE_TARGET,
             "open_body_store": OBS.store_root(), "open_body_store_target": OBS.STORE_TARGET,
             "amass_cache": os.path.join(paths.CACHE, "amass"),
             "amass_index": os.path.dirname(os.path.join(paths.CACHE, "amass_index", "takes.csv.gz")),
             "moveatlas_datasets": f"{paths.MOVEATLAS}/datasets", "gates_store": paths.GATES_STORE,
             "gates_store_target": paths.GATES_STORE_TARGET}
    for k in paths.WINDOW_STORE_KINDS:
        roots[f"window_store_{k}"] = paths.window_store_root(k)
    return {k: os.path.normcase(os.path.abspath(v)) for k, v in roots.items()}


class CaptureReads:
    """The files a build opens (and the folders it lists) under a motion-capture store (capture_roots), recorded
    while a clip builds (start, stop): an audit hook on the interpreter's file events (CAPTURE_EVENTS: builtins.open,
    np.memmap and np.load go through 'open'), installed once per process and silent while not recording, and
    recorders on the body store's and the AMASS loader's entry points (a store opened before the recording would
    serve its memory maps without a file event). Provenance is measured, not declared: a clip is 'captured' when its
    build read any such file, whatever its id says. The caches that could carry captured motion from one build to
    the next (the exam set's walk and stand, the series layers of physio.wrap_specs, the lru caches of the mdfm
    modules) are emptied after every build that read one (run: fresh specs, clear_mdfm_caches), so that the next
    clip built on the same take reads it again."""

    _hooked = []                                           # the instances whose hook is installed (one per process)

    def __init__(self, roots=None):
        self.roots = dict(roots if roots is not None else capture_roots())
        self.reads = None                                  # None: not recording
        self.last = []                                     # the reads of the last recording
        if not CaptureReads._hooked:
            sys.addaudithook(CaptureReads._dispatch)
        CaptureReads._hooked.append(self)
        self._patch_api()

    @staticmethod
    def _dispatch(event, args):
        if event not in CAPTURE_EVENTS:
            return
        for r in CaptureReads._hooked:
            if r.reads is not None:
                r._seen(args[0] if args else None)

    def _seen(self, path):
        try:
            if isinstance(path, int) or path is None:
                return
            p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        except (TypeError, ValueError):
            return
        for label, root in self.roots.items():
            if p == root or p.startswith(root.rstrip(os.sep) + os.sep):
                self.reads.add(f"{label}:{os.path.relpath(p, root).replace(os.sep, '/')}")

    def _patch_api(self):
        """Wrap the entry points of the body store and of the AMASS loader (once per process) so that a call while
        recording is recorded, even when it opens no file."""
        from mdfm.data import amass as AM
        from mdfm.data import body_store as BS
        for owner, names, label in ((BS.BodyStore, ("__init__", "_raw", "take", "windows", "sample"), "BodyStore"),
                                    (AM, ("load_sequence", "load_many", "list_sequences", "archive"), "amass")):
            for n in names:
                fn = getattr(owner, n, None)
                if fn is None or getattr(fn, "_capture_wrapped", False):
                    continue

                def wrapped(*a, _fn=fn, _name=f"{label}.{n}", **k):
                    for r in CaptureReads._hooked:
                        if r.reads is not None:
                            r.reads.add(f"api:{_name}")
                    return _fn(*a, **k)
                functools.update_wrapper(wrapped, fn)
                wrapped._capture_wrapped = True
                setattr(owner, n, wrapped)

    def start(self):
        self.reads = set()

    def stop(self):
        """The sorted reads since start (recording stops)."""
        out, self.reads = sorted(self.reads or ()), None
        self.last = out
        return out


def provenance_record(reads):
    """The 'provenance' block of a data file: captured when the build read under a capture store (CaptureReads)."""
    stores = sorted({r.split(":", 1)[0] for r in reads})
    return {"captured": bool(reads), "stores": stores, "n_reads": len(reads), "reads": reads[:40],
            "rule": "captured when the clip's build opened a file or listed a folder under a motion-capture store, or "
                    "called the body store or the AMASS loader (CaptureReads); a clip whose level-0 chain holds a "
                    "captured clip is left out of the public gallery as well (public_provenance)"}


def clear_mdfm_caches():
    """Empty the lru caches of the loaded mdfm modules (functions and classes' static functions)."""
    n = 0
    for name, mod in list(sys.modules.items()):
        if not (name == "mdfm" or name.startswith("mdfm.")) or mod is None:
            continue
        for v in list(vars(mod).values()):
            if callable(getattr(v, "cache_clear", None)):
                v.cache_clear()
                n += 1
    return n


def export_clip(spec, out, set_name=SET, module_checks=False, code=None, recorder=None):
    """Build one clip, write its data file; returns (meta, None) or (None, failed checks) (refusal). code: the
    clip's code record (CodeState.of), written into the file (the catalogue set). recorder: a CaptureReads, which
    records the build's reads of a motion-capture store into the file's 'provenance' (the catalogue set)."""
    if recorder is not None:
        recorder.start()
    try:
        body, truth = spec.build()
    finally:
        reads = recorder.stop() if recorder is not None else None
    bad = refusal(truth, module_checks)
    if bad:
        return None, bad
    skel = SK.skeleton(body, FPS) if code is None else catalogue_skeleton(body, FPS)
    meta = clip_meta(spec, truth, body, skel, set_name, "median" if code is None else "standing")
    if code is not None:                                  # the catalogue set: the per-time texts (task SKELV3)
        meta["overlays"] = clip_overlays(truth)
        meta["events_s"] = event_spans(truth)             # wave SH1d: the jerks the continuity check leaves out
        meta["slides_s"] = slide_spans(truth, body.rate)  # and the dragged foot's slides the foot checks leave out
        holds = hold_spans(truth)
        if holds:                                         # wave SH1d: the freezing holds the skate check judges
            meta["holds_s"] = holds
        touches = touch_readings(truth)
        if touches:                                       # plan phase E: a declared touch its module reads on the
            meta["touches_mesh_mm"] = touches             # posed mesh (MESH_TOUCHES)
        boxes, left = scene_parts(spec.scene, body, meta["floor"]["floor_y_m"])
        meta["scene_boxes_m"] = boxes                     # the same boxes as scene_boxes when the scene is boxes
        if left:
            meta["scene_left_out"] = left
        if spec.scene == "exam_bed":                      # task SKELV4a: lying on the couch (supine_view)
            meta["supine"] = True
        rep = reported_failures(truth)
        if rep:
            meta["reported_checks"] = rep
    rec = dict(meta, skeleton=SK.record(skel, mm=1, delta=True) if code is None else catalogue_record(skel))
    if code is not None:
        rec["code"] = code
    if reads is not None:
        rec["provenance"] = provenance_record(reads)
    raw = json.dumps(rec, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    p = data_path(out, spec.clip_id)
    LB.check_name(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".part"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(raw, compresslevel=9, mtime=0))
    os.replace(tmp, p)
    return meta, None


def read_data(path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def script_record(argv):
    with open(os.path.abspath(__file__), "rb") as f:
        sha = sha256_bytes(f.read())
    return {"path": "scripts/render_skeleton_catalogue.py", "sha256": sha, "argv": list(argv)}


# ---------------------------------------------------------------- the page's structure and labels
def _load_text():
    """scripts/skeleton_text.py as a module (the groups, headings, series and chips of the page, in English)."""
    spec = importlib.util.spec_from_file_location("skeleton_text", os.path.join(ROOT, "scripts", "skeleton_text.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TX = _load_text()
# the exam set's tables and functions under short names
FAMILY_NAMES, FAMILY_ORDER, ITEM_NAMES, SERIES_NAMES = TX.FAMILY_NAMES, TX.FAMILY_ORDER, TX.ITEM_NAMES, TX.SERIES_NAMES
EXAM_VIEWS = TX.EXAM_VIEWS
series_key, series_name, grade_chip, page_texts = TX.series_key, TX.series_name, TX.grade_chip, TX.page_texts


# ---------------------------------------------------------------- the page
def _clip_entry(m, size, text, family, item):
    """One clip of the page's index (page_index): its metadata and its texts ('text')."""
    sk = m["_skeleton"]
    return {"id": m["clip_id"], "family": family, "segment": m["segment"], "item": item,
            "grade": m["grade"], "level0": m["level0"], "views": m["views"], "since": m["since"],
            "heading": m["heading_deg"], "floor": m["floor"]["floor_y_m"], "boxes": m["scene_boxes_m"],
            "walk": m["walk"], "n_frames": sk["n_frames"], "n_points": sk["n_points"],
            "n_joints": sk["n_joints"], "hand_pose": sk["hand_pose"],
            "fps": sk["fps"], "duration": sk["duration_s"], "bytes": size,
            "toe_stance": m["toe_stance"], "text": text}


def view_title(title):
    """A view's title on the page: the viewer's own, first letter capitalized."""
    return title[:1].upper() + title[1:]


def _view_json(v):
    return {"azimuth": v.azimuth, "elevation": v.elevation, "vfov": v.vfov_deg, "region": v.region, "kind": v.kind,
            "margin": v.margin, "title": view_title(v.title)}


def _views_regions(catalogue=False):
    """(views, regions) of a page's index. The exam page: the views of task SKELV (EXAM_VIEWS) and the regions they
    frame, while viewer.VIEWS grows (task SKELV4a). The catalogue page: every view of viewer.VIEWS and every region.
    Each view under the viewer's own title."""
    views = {k: _view_json(v) for k, v in V.VIEWS.items() if catalogue or k in EXAM_VIEWS}
    used = {x["region"] for x in views.values()}
    regions = {k: (None if j is None else list(j)) for k, j in V.REGIONS.items() if catalogue or k in used}
    return views, regions


def page_index(metas, sizes):
    """JSON-ready index of the exam page: clips (metadata, texts, data size), families in order, views; and the
    labels that fell back (notes)."""
    clips, notes = [], {}
    for m in metas:
        text, left = page_texts(m)
        if left:
            notes[m["clip_id"]] = left
        clips.append(_clip_entry(m, sizes[m["clip_id"]], text, m["family"], m["item"]))
    views, regions = _views_regions()
    return {"label": LB.LABEL, "set": SET, "clips": clips, "family_order": list(FAMILY_ORDER),
            "families": FAMILY_NAMES, "views": views, "regions": regions}, notes


def catalogue_rows(metas):
    """[(meta, text, group, subgroup key)] of the catalogue page in the list's order (group in TX.GROUP_ORDER, then
    heading: the exam's scale items as they come, then the catalogue's SUBGROUPS in order), and the notes (labels
    that fell back). text: TX.page_texts with the series' chip and list line (TX.series_chips, TX.series_summary),
    the group and heading labels in place of the family and item, the scale item ('scale'), the page title and the
    filter's text."""
    rows, notes = [], {}
    for i, m in enumerate(metas):
        text, left = page_texts(m)
        sub, sub_lab = TX.subgroup_of(m, left)
        if left:
            notes[m["clip_id"]] = list(left)
        rows.append((i, m, text, TX.group_of(m), sub, sub_lab))
    series = {}
    for r in rows:
        if TX.is_catalogue(r[1]):
            series.setdefault(r[2]["series_key"], []).append(r)
    for key, rs in series.items():
        left = []
        ms = [r[1] for r in sorted(rs, key=lambda r: (TX.text_grade_rank(r[1]), r[0]))]   # the list's order
        chips = TX.series_chips(ms, left)
        summ = TX.series_summary(ms, chips)
        for r in rs:
            r[2]["chip"], r[2]["summary"] = chips[r[1]["clip_id"]], summ
        groups = {r[3] for r in rs}
        if len(groups) > 1:
            left.append(f"series {key} spans the groups {sorted(groups)}")
        if left:
            notes[f"series {key}"] = left
    by_id = {r[1]["clip_id"]: r for r in rows}
    for key, target in TX.SERIES_SEE_ALSO.items():          # a grade of the series shown under another heading
        if key in series and target in by_id:
            for r in series[key]:
                r[2]["summary"] += f'; see also "{by_id[target][2]["series"]}"'
    first_item, first_of = {}, {}
    for r in rows:
        if not TX.is_catalogue(r[1]):
            first_item.setdefault(r[4], len(first_item))
        first_of.setdefault(r[2]["series_key"], r[0])

    def rank(r):        # group, heading, then the series in order, its text grades in clinical order (task SKELV4a)
        g = TX.GROUP_ORDER.index(r[3]) if r[3] in TX.GROUP_ORDER else len(TX.GROUP_ORDER)
        s = (0, first_item[r[4]]) if r[4] in first_item else (1, TX.SUBGROUP_RANK.get(r[4], len(TX.SUBGROUPS)))
        return g, s, first_of[r[2]["series_key"]], TX.text_grade_rank(r[1]), r[0]

    out = []
    for i, m, text, g, sub, sub_lab in sorted(rows, key=rank):
        cat = TX.is_catalogue(m)
        raw_item = text["item"]
        title = (TX.catalogue_title(text["series"], sub_lab) if cat      # an exam series: its label, capitalized
                 else text["series"][:1].upper() + text["series"][1:])
        text, prov = TX.clinical_texts(text)          # wave SH1d: the marks of provenance out of the caption
        scale = text["item"] if cat and raw_item != sub_lab else ""      # the scale item, its marks out too
        text = dict(text, family=TX.GROUP_NAMES.get(g, g), item=sub_lab, scale=scale, title=title,
                    search=" ".join(x for x in (text["operator"], scale) if x), provenance=prov)
        out.append((m, text, g, sub))
    return out, notes


# The views of the catalogue page only (the exam page keeps the views of task SKELV). The head from the patient's
# left of the SKELV3 review is viewer.VIEWS['head_left'] since the shared wave (SH11, the same camera). A two-hand
# close-up without the elbows was tried for the mirror movements and left out: the hands lie about 60 cm apart there,
# so its frame is the one of 'hands' (the mirror clips keep their left-hand close-up first, 'hands' beside it).
# Task SKELV4a: the legs of a supine body on the examination couch (SARA 8), from its side and from above (the
# camera on the body's left, looking down: the leg's length across the view, the heel's lateral drift off the shin
# up and down it); the region 'legs' is the pelvis, hips, knees, ankles and toe blocks with the feet's landmarks.
CATALOGUE_VIEWS = {
    "legs_left": V.View("legs_left", 90.0, 20.0, 30.0, "legs", margin=0.08, title="legs, left side"),
    "legs_right": V.View("legs_right", -90.0, 20.0, 30.0, "legs", margin=0.08, title="legs, right side"),
    "legs_top": V.View("legs_top", 90.0, 72.0, 30.0, "legs", margin=0.08, title="legs, from above")}
CATALOGUE_REGIONS = {"legs": (0, 1, 2, 4, 5, 7, 8, 10, 11)}
EXTRA_VIEWS = [    # (regex on the clip id, the page's views put first, in order): the couch views of the supine clips
    (r"^atax_SARA8_heelshin_right_", ("legs_right", "legs_top")), (r"^atax_SARA8_", ("legs_left", "legs_top"))]
MORE_VIEWS = [     # (regex on the clip id, views offered after the spec's): the left hand from its side (SH11) for the
    #               hand tremors of the outstretched arms, whose hand now moves up and down at the wrist (exam_v5)
    (r"^tremor_E2_", ("hand_left_side",))]
DEFAULT_VIEW = [   # (regex on the clip id, the view a clip opens on): where the spec's first view hides the sign
    (r"^park_E4aL_sequence_ft_", "hand_left"), (r"^park_E4aL?_ulfreeze_", "hand_left"),
    (r"^atax_TITUB_", "head_left"),
    (r"^invol_lid_diphasic_E1_kicking_", "left"), (r"^invol_rls_E1_", "left"), (r"^invol_akathisia_E1_", "left"),
    (r"^post_E7_ped_excerpt$", "left"), (r"^post_E7_off_period_foot$", "left"), (r"^post_E1_fixed_foot$", "foot_left"),
    (r"^post_E7_cd_tilt_left_jerky$", "head"), (r"^post_W_", "hands"),
    (r"^park_E4L_sequence_ps_", "hands_top"), (r"^atax_SARA7_ddk_", "hands_top"),
    (r"^post_E1_fixed_fist$", "hand_left_medial"), (r"^invol_athetosis_E1_", "hands")]


def catalogue_views(clip_id, views):
    """The views of a clip on the catalogue page: its spec's, the head from the left after a head close-up, the
    page's views of EXTRA_VIEWS first and those of MORE_VIEWS last, and the view of DEFAULT_VIEW first (the one the
    page opens on)."""
    vs = list(views)
    if "head" in vs and "head_left" not in vs:
        vs.insert(vs.index("head") + 1, "head_left")
    for pat, extra in EXTRA_VIEWS:
        if re.search(pat, clip_id):
            vs = list(extra) + [x for x in vs if x not in extra]
            break
    for pat, more in MORE_VIEWS:
        if re.search(pat, clip_id):
            vs += [x for x in more if x not in vs]
    for pat, v in DEFAULT_VIEW:
        if re.search(pat, clip_id):
            return [v] + [x for x in vs if x != v]
    return vs


def page_index_catalogue(metas, sizes, set_name=SET_CATALOGUE):
    """JSON-ready index of the catalogue page (catalogue_rows): each clip under its group and heading with its
    texts and their provenance line, its views (catalogue_views), the floor and scene boxes it is drawn with (its
    level 0's, page_floor: the page reads them from the index, not from the clip's data file), every view of
    viewer.VIEWS and the catalogue's own views and regions and the landmarks the page draws (LANDMARK_BONES) and
    frames (REGION_LANDMARKS); and the notes."""
    rows, notes = catalogue_rows(metas)
    clips = [_clip_entry(m, sizes[m["clip_id"]], text, g, sub) for m, text, g, sub in rows]
    by, roots = {m["clip_id"]: m for m in metas}, page_roots(metas)
    for c, (m, _, _, _) in zip(clips, rows):
        c["views"] = catalogue_views(c["id"], c["views"])
        c["floor"], c["boxes"] = page_floor(m, by[roots[c["id"]]])   # wave SH1d: the level 0's floor and chair
        if m.get("supine"):
            c["supine"] = True
    views, regions = _views_regions(catalogue=True)
    views.update({k: _view_json(v) for k, v in CATALOGUE_VIEWS.items()})
    regions.update({k: list(j) for k, j in CATALOGUE_REGIONS.items()})
    present = [g for g in TX.GROUP_ORDER if any(c["family"] == g for c in clips)]
    return {"label": LB.LABEL, "set": set_name, "clips": clips, "family_order": present,
            "families": {g: TX.GROUP_NAMES[g] for g in present}, "views": views, "regions": regions,
            "landmark_bones": LANDMARK_BONES, "region_landmarks": REGION_LANDMARKS}, notes


def refused_label(spec):
    """The label of a refused clip (it has no data file): its operator and grade text, its id after them."""
    op, gr = spec.operator, spec.grade_text or ""
    text = op if gr.lower() in op.lower() else f"{op}, {gr}"           # the grade once ('mirror dystonia')
    return f"{text} (clip id {spec.clip_id})"


def deferred_note(deferred, labels=None):
    """The sentence on the clips not built because their build needs a family left out of --modules ('' when
    none)."""
    if not deferred:
        return ""
    labels = labels or {}
    fams = sorted({MODULE_NAMES.get(m.rsplit(".", 1)[-1], m) for v in deferred.values() for m in v})
    n = len(deferred)
    what = "is not built: its build calls" if n == 1 else "are not built: their build calls"
    return (f" {n} clip{'s' if n > 1 else ''} {what} a family of the catalogue still being written "
            f"({', '.join(fams)}), which this catalogue does not load: "
            f"{'; '.join(labels.get(c, c) for c in sorted(deferred))}.")


def segment_list(segments):
    """The segment codes of the page, from its index: the plain exam segments by number, a run of three or more
    without a gap as 'E<a> to E<b>' (task SKELV4b: E1 to E10, E19, E21, E22), then the exam's other codes (E2f,
    E4aL ...) by number, then the catalogue's own (J2, ROMB, SARA2 ...)."""
    segs = sorted(set(segments))
    plain = sorted(int(s[1:]) for s in segs if re.fullmatch(r"E\d+", s))
    out, i = [], 0
    while i < len(plain):
        j = i
        while j + 1 < len(plain) and plain[j + 1] == plain[j] + 1:
            j += 1
        out += [f"E{plain[i]} to E{plain[j]}"] if j - i >= 2 else [f"E{n}" for n in plain[i:j + 1]]
        i = j + 1
    out += sorted((s for s in segs if re.fullmatch(r"E\d+\D\w*", s)),
                  key=lambda s: (int(re.match(r"E(\d+)", s).group(1)), s.lower()))
    out += [s for s in segs if not re.match(r"E\d", s)]
    return ", ".join(out)


def _clips_sentence(ids, labels, one, many):
    """' n clip(s) <one or many>: labels.' ('' when ids is empty)."""
    if not ids:
        return ""
    n = len(ids)
    return f" {n} {one if n == 1 else many}: {'; '.join(labels.get(c, c) for c in ids)}."


def catalogue_template(refused, labels=None, deferred=None, segments=(), public=None):
    """The page template of the catalogue set: PAGE_TEMPLATE with CATALOGUE_PATCHES (each found exactly once),
    the segment codes of the page (segment_list) and the paragraph on the refused clips (labels: refused_label; the
    id otherwise) and on the clips not built (deferred_note); a clip whose generator stopped on an error
    (BUILD_ERROR), a clip whose skeleton is its level 0's (INVISIBLE), a level 0 left without a shown symptom
    clip (LONE_LEVEL0), a clip a skeleton check refuses (SKELETON_FAIL, a check out of SKELETON_REPORTED_ONLY) and a
    clip whose level 0 a check refuses (LEVEL0_REFUSED) get their own sentences. public: the public gallery's
    sentence on the clips it leaves out (public_note), with PUBLIC_PATCHES and without the code comments
    (strip_code_comments)."""
    labels = labels or {}
    errors = sorted(c for c, v in refused.items() if any(str(x).startswith(BUILD_ERROR) for x in v))
    invisible = sorted(c for c, v in refused.items() if list(v) == [INVISIBLE])
    lone = sorted(c for c, v in refused.items() if list(v) == [LONE_LEVEL0])
    checked = sorted(c for c, v in refused.items() if v and all(str(x).startswith(SKELETON_FAIL) for x in v))
    follow = sorted(c for c, v in refused.items() if list(v) == [LEVEL0_REFUSED])
    invalid = sorted(c for c in refused if c not in errors and c not in invisible and c not in lone
                     and c not in checked and c not in follow)
    if invalid:
        note = _clips_sentence(invalid, labels, "clip whose build fails the validity checks (the exam's checks and "
                               "the module's own) is not shown", "clips whose build fails the validity checks (the "
                               "exam's checks and the module's own) are not shown").lstrip()
    else:
        note = "No clip failed the validity checks (the exam's checks and the module's own)."
    note += _clips_sentence(errors, labels, "clip whose generator stopped on an error is not shown",
                            "clips whose generator stopped on an error are not shown")
    note += _clips_sentence(invisible, labels, "clip whose skeleton equals its level 0's, frame by frame (the "
                            "movement reaches no point of the skeleton), is not shown", "clips whose skeleton equals "
                            "their level 0's, frame by frame (the movement reaches no point of the skeleton), are "
                            "not shown")
    note += _clips_sentence(lone, labels, "level 0 clip without a symptom clip on the page is not shown either",
                            "level 0 clips without a symptom clip on the page are not shown either")
    note += _clips_sentence(checked, labels, f"clip whose skeleton fails the page's own checks ({CHECKS_WORDS}) is "
                            "not shown", f"clips whose skeleton fails the page's own checks ({CHECKS_WORDS}) are not "
                            "shown")
    note += _clips_sentence(follow, labels, "clip whose level 0 fails the page's own checks is not shown "
                            "either", "clips whose level 0 fails the page's own checks are not shown either")
    t = PAGE_TEMPLATE
    for old, new in CATALOGUE_PATCHES:
        if t.count(old) != 1:
            raise RuntimeError(f"catalogue patch: {old[:60]!r} found {t.count(old)} times")
        t = t.replace(old, new)
    if public is not None:
        for old, new in PUBLIC_PATCHES:
            if t.count(old) != 1:
                raise RuntimeError(f"public patch: {old[:60]!r} found {t.count(old)} times")
            t = t.replace(old, new)
        t = strip_code_comments(t).replace("__PUBLIC_NOTE__", public)
    seg = segment_list(segments)
    t = t.replace(" (__SEGMENTS__)", f" ({seg})" if seg else "")
    return t.replace("__REFUSED_NOTE__", note + deferred_note(deferred, labels))


def _page_checked(page):
    """The page's text outside its data blocks (base64) must hold no em-dash or en-dash and no word of the
    vendor, model or abbreviation lists; the base64 blocks are left out of the word checks (they can spell one)."""
    text = re.sub(r'<script type="text/plain" class="clipdata"[^>]*>[^<]*</script>', "", page)
    for bad in ("\u2014", "\u2013") + tuple(r"\b" + w + r"\b" for w in ("Cl" "aude", "Anth" "ropic", "A" "I", "I" "A")):
        if re.search(bad, text):
            raise RuntimeError(f"the page holds {bad!r}")


def build_page(out, metas, blobs, argv, mode="exam", refused=None, set_name=SET, refused_labels=None,
               deferred=None):
    """Write the page (module docstring); returns (path, bytes, embedded, notes). mode 'catalogue': the
    catalogue's index (page_index_catalogue) and template (catalogue_template). notes: the labels that fell back
    (a series without a label, a clip without a heading), listed, not a refusal."""
    sizes = {k: len(b) for k, b in blobs.items()}
    if mode == "catalogue":
        idx, notes = page_index_catalogue(metas, sizes, set_name)
        template = catalogue_template(refused or {}, refused_labels, deferred, [c["segment"] for c in idx["clips"]])
    else:
        idx, notes = page_index(metas, sizes)
        template = PAGE_TEMPLATE
    total = sum(sizes.values())
    if total * 4 / 3 > EMBED_LIMIT_BYTES:
        raise SystemExit(f"{total} bytes of data: over the embedding limit, the page would need fetch")
    blocks = "\n".join(f'<script type="text/plain" class="clipdata" data-clip="{cid}">'
                       f'{base64.b64encode(b).decode("ascii")}</script>' for cid, b in blobs.items())
    js = json.dumps(idx, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (template.replace("__INDEX_JSON__", js).replace("__DATA_BLOCKS__", blocks)
            .replace("__NOTICE__", NOTICE).replace("__N_CLIPS__", str(len(metas))))
    _page_checked(page)
    p = f"{out}/{PAGE}"
    os.makedirs(out, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    return p, os.path.getsize(p), total, notes


def load_all(out, specs):
    metas, blobs = [], {}
    for s in specs:
        p = data_path(out, s.clip_id)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            b = f.read()
        d = json.loads(gzip.decompress(b).decode("utf-8"))
        sk = d.pop("skeleton")
        d["_skeleton"] = {k: sk[k] for k in ("n_frames", "n_points", "n_joints", "fps", "duration_s", "hands",
                                             "hand_pose")}
        d["_names"] = list(sk["names"])                   # the points' names (the skeleton checks read them)
        metas.append(d)
        blobs[s.clip_id] = b
    return metas, blobs


def blob_positions(blob):
    """(N, P, 3) int32 mm of a data file's bytes (the running sum of positions_mm_delta)."""
    sk = json.loads(gzip.decompress(blob).decode("utf-8"))["skeleton"]
    q = np.asarray(sk["positions_mm_delta"], dtype=np.int32).reshape(sk["n_frames"], sk["n_points"], 3)
    return np.cumsum(q, 0, dtype=np.int32)


def hidden_clips(metas, blobs, specs):
    """{clip: [reason]} of the catalogue clips the page leaves out although their data exist (task SKELV3 review):
    a symptom clip (a level 0, a grade other than 0) whose points equal its level 0's on every common frame
    (INVISIBLE: the movement reaches no point of the skeleton), then a catalogue level 0 that is the level 0 of
    specs none of which is shown (LONE_LEVEL0: its symptom clips refused or hidden), and since wave SH1d a catalogue
    level 0 (grade 0, no level 0 of its own) that no spec names as its level 0, but those of TX.STANDALONE_LEVEL0
    (LONE_LEVEL0 too: tremor_E3_reach_right_g0, the right-arm reach whose symptom clips are all left-sided)."""
    out, level0 = {}, {}
    for m in metas:
        c, l0 = m["clip_id"], m.get("level0")
        if not TX.is_catalogue(m) or not l0 or l0 not in blobs or m["grade"] == 0:
            continue
        if l0 not in level0:
            level0[l0] = blob_positions(blobs[l0])
        a, b = blob_positions(blobs[c]), level0[l0]
        n = min(len(a), len(b))
        if a.shape[1] == b.shape[1] and np.array_equal(a[:n], b[:n]):
            out[c] = [INVISIBLE]
    shown = {m["clip_id"] for m in metas if m["clip_id"] not in out}
    out.update(lone_level0s(metas, specs, shown))
    return out


def lone_level0s(metas, specs, shown):
    """{clip: [LONE_LEVEL0]} of the catalogue level 0s among the shown clips (shown: their ids) that are the level 0
    of specs none of which is shown, or (grade 0, no level 0 of their own) that no spec names as its level 0, but
    those of TX.STANDALONE_LEVEL0 (hidden_clips, refusal_followers)."""
    out, users = {}, {}
    for s in specs:
        if s.level0:
            users.setdefault(s.level0, []).append(s.clip_id)
    for m in metas:
        c = m["clip_id"]
        if c not in shown or not TX.is_catalogue(m) or m.get("level0"):
            continue
        if users.get(c) and not any(u in shown for u in users[c]):
            out[c] = [LONE_LEVEL0]
        elif m["grade"] == 0 and not users.get(c) and c not in TX.STANDALONE_LEVEL0:
            out[c] = [LONE_LEVEL0]
    return out


def refusal_followers(metas, specs, hidden):
    """{clip: [reason]} of the clips the page leaves out after the skeleton checks' refusals (hidden: {clip: [reason]}
    of the clips left out so far, the refusals included): a clip whose level 0 a check refuses or this rule leaves
    out (LEVEL0_REFUSED: a symptom clip is shown beside its level 0 and drawn on its floor, and that level 0 is not
    shown), then a level 0 left without a shown symptom clip (lone_level0s, LONE_LEVEL0), until nothing changes."""
    out = {}
    while True:
        gone = {**hidden, **out}
        roots = {c for c, v in gone.items() if any(str(x).startswith(SKELETON_FAIL) or x == LEVEL0_REFUSED for x in v)}
        new = {m["clip_id"]: [LEVEL0_REFUSED] for m in metas
               if m["clip_id"] not in gone and m.get("level0") in roots}
        shown = {m["clip_id"] for m in metas if m["clip_id"] not in gone and m["clip_id"] not in new}
        new.update(lone_level0s(metas, specs, shown))
        if not new:
            return out
        out.update(new)


def boxes_shift_mm(own, drawn):
    """The largest horizontal difference (mm, 0.1 mm) between a clip's own scene boxes and the boxes the page draws
    for it, corner by corner (a box [x0, y0, z0, x1, y1, z1] has two corners); None when they are not the same
    number of boxes of the same form (or not boxes)."""
    try:
        a, b = np.asarray(own, dtype=np.float64), np.asarray(drawn, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] not in (3, 6):
        return None
    a, b = a.reshape(*a.shape[:-1], -1, 3), b.reshape(*b.shape[:-1], -1, 3)
    return round(1000.0 * float(np.abs(a - b)[..., [0, 2]].max()), 1) if a.size else 0.0


def floor_record(metas):
    """{clip: {root, own_floor_mm, floor_mm, boxes_shift_mm}} of the page's clips drawn on another clip's floor
    (page_floor, wave SH1d): the root it follows, its data file's floor and the floor the page draws, in mm (0.1 mm),
    and how far its own scene boxes lie horizontally from the ones drawn (boxes_shift_mm: the seat of a rise placed
    elsewhere than its level 0's, wave SH1d review of lane page)."""
    by, roots, out = {m["clip_id"]: m for m in metas}, page_roots(metas), {}
    for m in metas:
        c = m["clip_id"]
        if roots[c] != c:
            fl, bx = page_floor(m, by[roots[c]])
            out[c] = {"root": roots[c], "own_floor_mm": round(1000.0 * m["floor"]["floor_y_m"], 1),
                      "floor_mm": round(1000.0 * fl, 1), "boxes_shift_mm": boxes_shift_mm(m["scene_boxes_m"], bx)}
    return out


def checks_record(results):
    """The manifest's block of the skeleton checks: the results of every check that fails, by clip, the number of
    clips each check flags, the checks reported only, the rule, and the clips read and not judged
    ('skeleton_checks_not_judged', by check: the whole reading of a clip a table of exemptions names (SKATE_EXEMPT,
    SINKING_EXEMPT, the contacts of CONTACT_EXEMPT), so that the readings behind an exemption stay in view, and the
    ids of the clips whose data hold no declared spans (DECLARED_SPANS))."""
    failed = checks_failed(results)
    not_judged = {}
    for c, r in sorted(results.items()):
        for k in SKELETON_CHECKS:
            x = r.get(k) or {}
            if x.get("exempt") or x.get("contacts_exempt"):
                not_judged.setdefault(k, {}).setdefault("exempt", {})[c] = x
            elif x.get("not_judged"):
                not_judged.setdefault(k, {}).setdefault("spans_unknown", []).append(c)
    return {"skeleton_checks": {c: {k: results[c][k] for k in v} for c, v in sorted(failed.items())},
            "skeleton_checks_not_judged": not_judged,
            "skeleton_checks_flagged": {k: sum(k in v for v in failed.values()) for k in SKELETON_CHECKS},
            "skeleton_checks_read": len(results),
            "skeleton_checks_reported_only": list(SKELETON_REPORTED_ONLY),
            "skeleton_checks_rule": "the uniform checks of the skeleton on the exported positions (module section "
                                    "'wave SH1d'); a check in skeleton_checks_reported_only is written here and the "
                                    "clip shown, any other hides the clips it flags"}


def write_manifest(out, metas, blobs, page, page_bytes, argv, skipped, notes, set_name=SET, specs=None,
                   modules=None, stale=None, deferred=None, data_from=None, checks=None):
    """The manifest (module docstring); the catalogue set (specs given) adds its modules, the clips planned, shown and
    refused by family, the shown clips by group of the page's list, each file's code digests, the stale files and
    the page build's code record as 'page_code', and since wave SH1d the clips drawn on their level 0's floor
    (floor_record, 'page_floors') and the skeleton checks (checks: skeleton_checks of the shown clips, checks_record).
    data_from: the folder the data files were read from (--data-from), their paths relative to it."""
    files = [{"clip": m["clip_id"], "file": f"{DATA}/{data_name(m['clip_id'])}", "bytes": len(blobs[m["clip_id"]]),
              "sha256": sha256_bytes(blobs[m["clip_id"]]), "n_frames": m["_skeleton"]["n_frames"],
              "n_points": m["_skeleton"]["n_points"], "n_joints": m["_skeleton"]["n_joints"],
              "hand_pose": m["_skeleton"]["hand_pose"], "duration_s": m["_skeleton"]["duration_s"]} for m in metas]
    if specs is not None:
        for x, m in zip(files, metas):
            c = m.get("code") or {}
            x.update({k: c.get(k) for k in STALE_KEYS})
    with open(page, "rb") as f:
        page_sha = sha256_bytes(f.read())
    fams = {}
    for m in metas:
        fams[m["family"]] = fams.get(m["family"], 0) + 1
    man = {"label": LB.LABEL, "notice": NOTICE, "set": set_name, "page": PAGE, "page_bytes": page_bytes,
           "page_sha256": page_sha, "embedded": True, "clips": len(metas), "clips_by_family": fams,
           "clips_by_since": {v: sum(m["since"] == v for m in metas) for v in sorted({m["since"] for m in metas})},
           "clips_with_animated_hands": sum(m["_skeleton"]["hand_pose"] == "animated" for m in metas),
           "frames": sum(x["n_frames"] for x in files), "data_bytes": sum(x["bytes"] for x in files),
           "files": files, "refused": skipped, "unlabelled": notes,
           "format": {"file": "gzip of UTF-8 JSON", "positions": "skeleton.positions_mm_delta (module docstring)",
                      "fps": FPS, "unit": "mm, rounded to 1 mm", "up": "y"},
           "code": RMV.code_record(), "script": script_record(argv),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if data_from:
        man["data_from"] = data_from
    if specs is not None:
        fam_of = {s.clip_id: s.family for s in specs}
        planned, refused_by = {}, {}
        for s in specs:
            planned[s.family] = planned.get(s.family, 0) + 1
        deferred_by = {}
        for c in skipped:
            d = deferred_by if c in (deferred or {}) else refused_by
            d[fam_of.get(c, "?")] = d.get(fam_of.get(c, "?"), 0) + 1
        groups = {}
        for m in metas:
            g = TX.group_of(m)
            groups[g] = groups.get(g, 0) + 1
        man.update({"modules": list(modules or ()), "clips_planned": len(specs), "clips_planned_by_family": planned,
                    "clips_refused_by_family": refused_by, "deferred": dict(deferred or {}),
                    "clips_deferred_by_family": deferred_by,
                    "clips_by_group": {g: groups[g] for g in TX.GROUP_ORDER if g in groups},
                    "refusal_rule": "render_mesh_video.failed_checks and the module's own validity "
                                    "(module_failures)",
                    "stale": dict(stale or {}),
                    "scene_left_out": {m["clip_id"]: m["scene_left_out"] for m in metas if m.get("scene_left_out")},
                    "reported_checks": {m["clip_id"]: m["reported_checks"] for m in metas
                                        if m.get("reported_checks")},
                    "reported_rule": "validity entries reported without refusing the clip (REPORTED_ONLY)",
                    "clips_with_overlays": sum(bool(m.get("overlays")) for m in metas),
                    "stale_rule": "a data file is stale when its module or dependency digest differs from the "
                                  "current files' (module docstring)",
                    "page_floors": floor_record(metas),
                    "page_floor_rule": "a clip is drawn on its level 0's floor and with its level 0's scene boxes "
                                       "(page_floor); a level 0's floor is its data file's (clip_floor)"})
        if checks is not None:
            man.update(checks_record(checks))
        man["page_code"] = man.pop("code")
    p = f"{out}/{MANIFEST}"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return p, man


def write_record(man, out, path=RECORD):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: a run record is never rewritten")
    rec = {k: man[k] for k in ("label", "set", "page", "page_bytes", "page_sha256", "embedded", "clips",
                               "clips_by_family", "clips_by_since", "clips_with_animated_hands", "frames",
                               "data_bytes",
                               "refused", "unlabelled", "format", "code", "page_code", "script",
                               "created_utc") if k in man}
    rec["out"] = out
    rec["files"] = [{k: x[k] for k in ("clip", "file", "bytes", "sha256", "n_frames", "n_points", "hand_pose",
                                       "module", "module_sha256", "deps_sha256") if k in x}
                    for x in man["files"]]
    for k in ("modules", "clips_planned", "clips_planned_by_family", "clips_refused_by_family", "clips_by_group",
              "stale", "deferred", "clips_deferred_by_family"):
        if k in man:
            rec[k] = man[k]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def copy_out(out, dest):
    """A copy of the catalogue folder at dest (refused inside this tree, MoveAtlas or the gates-only store)."""
    d = os.path.abspath(dest).replace("\\", "/")
    paths.assert_not_moveatlas(d)
    paths.assert_not_gates_store(d)
    if paths._inside(d, paths.TREE):
        raise PermissionError(f"the copy is never written in the MD_FM tree: {d}")
    shutil.copytree(out, d, dirs_exist_ok=True)
    return d


def read_refused(out, key="refused"):
    """{clip: [failed checks]} of the catalogue set's refusals written by earlier runs ({} when none); key
    'deferred': {clip: [blocked modules]} of the clips not built (BlockedModules)."""
    p = f"{out}/{REFUSED}"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f).get(key, {})


def write_refused(out, refused, deferred=None):
    p = f"{out}/{REFUSED}"
    os.makedirs(out, exist_ok=True)
    rec = {"label": LB.LABEL, "notice": NOTICE, "refused": dict(sorted(refused.items())),
           "rule": "render_mesh_video.failed_checks and the module's own validity (module_failures)"}
    if deferred:
        rec.update({"deferred": dict(sorted(deferred.items())),
                    "deferred_rule": "not built: the clip's build imports a catalogue module left out of --modules "
                                     "(a family still being written), which this script never imports"})
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return p


# ---------------------------------------------------------------- the families not delivered stay unimported
MODULE_NAMES = {"tremors": "tremors", "jerks": "jerks", "involuntary": "involuntary movements", "gait": "gait",
                "postures": "dystonia and postures", "parkinsonism": "parkinsonism", "ataxia": "ataxia"}


BUILD_ERROR = "build_error"            # the refusal of a clip whose build raised (catalogue set)
INVISIBLE = "invisible_on_skeleton"    # a symptom clip whose points equal its level 0's on every frame (hidden_clips)
LONE_LEVEL0 = "level0_without_shown_symptom"   # a level 0 whose every symptom clip is refused (hidden_clips)
SKELETON_FAIL = "skeleton_check: "     # the refusal of a clip by a skeleton check out of SKELETON_REPORTED_ONLY
LEVEL0_REFUSED = "level0_refused_by_check"   # a clip whose level 0 a skeleton check refuses (refusal_followers)


class BlockedModule(ImportError):
    """The import of a catalogue module left out of --modules (BlockedModules)."""


class BlockedModules:
    """A meta path finder that refuses the import of the catalogue modules left out of --modules: a family still
    being written is never imported, even by the build of a delivered clip that calls it (such a clip is left out
    and listed as not built)."""

    def __init__(self, modules):
        self.names = {f"mdfm.synth.{m}": m for m in modules}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            raise BlockedModule(f"{fullname} is not imported (a catalogue family left out of --modules)",
                                name=fullname)
        return None


def blocked_modules(modules):
    """The catalogue modules left out of `modules`."""
    from mdfm.synth.catalogue import CATALOGUE_MODULES
    return [m for m in CATALOGUE_MODULES if m not in modules]


def set_specs(set_name, modules):
    """[ClipSpec] of --set exam (exam.exam_set) or --set catalogue (catalogue.catalogue_set of the modules)."""
    if set_name == "exam":
        return EX.exam_set()
    from mdfm.synth import catalogue as CAT
    unknown = [m for m in modules if m not in CAT.CATALOGUE_MODULES]
    if unknown:
        raise SystemExit(f"unknown catalogue modules {unknown}; known {CAT.CATALOGUE_MODULES}")
    return CAT.catalogue_set(modules=tuple(modules))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", choices=("exam", "catalogue"), default="exam",
                    help="exam: exam.exam_set() into skeleton_v1 (default); catalogue: the symptom catalogue")
    ap.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES),
                    help="the catalogue modules of --set catalogue (default: the delivered ones)")
    ap.add_argument("--out", default=None, help=f"output folder, inside {V.VIEWER_ROOT}")
    ap.add_argument("--only", nargs="*", default=None, help="export only the clips whose id holds one of these")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    ap.add_argument("--page-only", action="store_true", help="rebuild the page and manifest from the data files")
    ap.add_argument("--copy-to", default=None, help="also copy the folder there (outside this tree)")
    ap.add_argument("--no-record", action="store_true", help="no run record in the tree")
    ap.add_argument("--data-from", default=None,
                    help=f"--page-only: read the data files and refusals from this folder (inside {V.VIEWER_ROOT}, "
                         "never written) and write the page and manifest to --out")
    ap.add_argument("--allow-stale", action="store_true",
                    help="catalogue: write the page even with data files built by other code (listed in the manifest)")
    ap.add_argument("--public", action="store_true",
                    help="the public gallery (with --set catalogue --page-only): read --data-from (default "
                         f"{os.path.basename(DATA_PUBLIC)}), write {PUBLIC_PAGE} and {PUBLIC_MANIFEST} to --out "
                         f"(default {SET_PUBLIC})")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else list(argv)
    guard = BlockedModules(blocked_modules(a.modules)) if a.set == "catalogue" else None
    if guard is not None:
        sys.meta_path.insert(0, guard)
    try:
        if a.public:
            run_public(a, argv)
        else:
            run(a, argv, guard)
    finally:
        if guard is not None:
            sys.meta_path.remove(guard)


def page_selection(metas, blobs, specs):
    """(metas, blobs, hidden, checks) of the catalogue page: the loaded clips less those the page leaves out
    although their data exist (hidden_clips), then the uniform checks of the skeleton (skeleton_checks, wave SH1d) on
    the clips left, the clips a refusing check flags and their followers (refusal_followers) left out too. hidden:
    {clip: [reason]}; checks: the checks' results of the clips shown."""
    hidden = hidden_clips(metas, blobs, specs)            # data kept, left out of the page (hidden_clips)
    metas = [m for m in metas if m["clip_id"] not in hidden]
    blobs = {c: b for c, b in blobs.items() if c not in hidden}
    t0 = time.time()                                       # wave SH1d: the uniform checks of the skeleton
    checks = skeleton_checks(metas, blobs)
    refusing = [k for k in SKELETON_CHECKS if k not in SKELETON_REPORTED_ONLY]
    for c, bad in checks_failed(checks, refusing).items():
        hidden[c] = [f"{SKELETON_FAIL}{k}" for k in bad]
    hidden.update(refusal_followers(metas, specs, hidden))   # the pairs and level 0s the refusals leave alone
    metas = [m for m in metas if m["clip_id"] not in hidden]
    blobs = {c: b for c, b in blobs.items() if c not in hidden}
    flagged = checks_failed(checks)
    print(f"skeleton checks ({time.time() - t0:.0f} s, reported only: {', '.join(SKELETON_REPORTED_ONLY)}): "
          f"{ {k: sum(k in v for v in flagged.values()) for k in SKELETON_CHECKS} } clips flagged")
    checks = {c: r for c, r in checks.items() if c not in hidden}
    return metas, blobs, hidden, checks


def run(a, argv, guard=None):
    """main after its arguments (the catalogue set with the import guard of the families left out installed)."""
    cat = a.set == "catalogue"
    set_name = SET_CATALOGUE if cat else SET
    out = (a.out or (OUT_CATALOGUE if cat else OUT)).rstrip("/").replace("\\", "/")
    V.check_out(out, V.VIEWER_ROOT)
    src = out                                              # where the data files and refusals are read
    if a.data_from:
        if not a.page_only:
            raise SystemExit("--data-from reads another folder's data: it goes with --page-only only")
        src = a.data_from.rstrip("/").replace("\\", "/")
        V.check_out(src, V.VIEWER_ROOT)
        if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(out)):
            raise SystemExit("--data-from names the output folder itself")
    specs = set_specs(a.set, a.modules)
    sel = [s for s in specs if not a.only or any(k in s.clip_id for k in a.only)]
    if a.list:
        for s in sel:
            print(f"{s.clip_id:55s} {s.family:12s} {s.segment:4s} {s.since} level0 {s.level0}")
        print(f"{len(sel)} clips")
        return
    skipped = {}
    refused = read_refused(src) if cat else {}
    deferred = read_refused(src, "deferred") if cat else {}
    left_out = blocked_modules(a.modules) if cat else []
    state = CodeState(specs, left_out) if cat else None    # the code just imported for the build
    if not a.page_only:
        loaded = [n for n in (guard.names if guard else {}) if n in sys.modules]
        if loaded:
            raise SystemExit(f"{loaded} imported before the build: the families left out are never run")
        t0 = time.time()
        recorder = None
        if cat:                                            # the public gallery: each build's reads of a capture store
            recorder = CaptureReads()
            recorder.start()
            specs = set_specs(a.set, a.modules)            # the set built again while recording: it reads nothing
            pre = recorder.stop()
            if pre:
                raise SystemExit(f"the set's construction read a motion-capture store ({pre[:5]}): a clip's "
                                 "provenance could not be told from its own build")
            sel = [s for s in specs if not a.only or any(k in s.clip_id for k in a.only)]
        order = [s.clip_id for s in sel]
        current = {s.clip_id: s for s in sel}
        for cid in order:
            if recorder is not None and recorder.last:     # the last build read a capture store: no cache keeps it
                specs = set_specs(a.set, a.modules)
                current = {s.clip_id: s for s in specs if s.clip_id in current}
                print(f"  (the last build read {len(recorder.last)} capture files or calls: fresh specs, "
                      f"{clear_mdfm_caches()} lru caches emptied)")
                recorder.last = []
            s = current[cid]
            p = data_path(out, s.clip_id)
            if os.path.exists(p):
                why = stale_reason(read_code(p), state.of(s)) if cat else None
                if why is None:
                    print(f"{s.clip_id}: done before, skipped")
                    continue
                print(f"{s.clip_id}: stale ({why}), rebuilt")
                os.remove(p)
            t1 = time.time()
            try:
                _, bad = export_clip(s, out, set_name, module_checks=cat, code=state.of(s) if cat else None,
                                     recorder=recorder)
            except BlockedModule as e:
                deferred[s.clip_id] = [e.name]
                refused.pop(s.clip_id, None)
                write_refused(out, refused, deferred)
                print(f"{s.clip_id}: its build needs {e.name}, left out of --modules; not built")
                continue
            except Exception as e:                         # a module still being fixed: refuse the clip, go on
                if not cat:
                    raise
                traceback.print_exc()
                bad = [f"{BUILD_ERROR}: {type(e).__name__}: {e}"[:400]]
            deferred.pop(s.clip_id, None)
            if bad:
                skipped[s.clip_id] = bad
                print(f"{s.clip_id}: {'its build fails' if bad[0].startswith(BUILD_ERROR) else 'its truth fails'} "
                      f"{bad}; refused")
                if cat:
                    refused[s.clip_id] = bad
                    write_refused(out, refused, deferred)
                continue
            if cat and refused.pop(s.clip_id, None) is not None:
                write_refused(out, refused, deferred)
            print(f"{s.clip_id}: {os.path.getsize(data_path(out, s.clip_id))} bytes in {time.time() - t1:.1f} s "
                  f"(total {time.time() - t0:.0f} s)", flush=True)
    metas, blobs = load_all(src, specs)
    stale, hidden = {}, {}
    if cat:
        by_id = {s.clip_id: s for s in specs}
        now = CodeState(specs, left_out)                   # the code files as they are now
        stale = {m["clip_id"]: why for m in metas
                 if (why := stale_reason(m.get("code"), now.of(by_id[m["clip_id"]]))) is not None}
        metas, blobs, hidden, checks = page_selection(metas, blobs, specs)
    missing = [s.clip_id for s in specs if s.clip_id not in blobs]
    if cat:
        refused = {c: refused[c] for c in missing if c in refused}
        refused.update(hidden)
        deferred = {c: deferred[c] for c in missing if c in deferred and c not in refused}
        write_refused(out, refused, deferred)
        skipped = {**refused, **{c: [f"not built: needs {', '.join(v)}"] for c, v in deferred.items()}}
        if stale and not a.allow_stale:
            raise SystemExit(f"{len(stale)} data files built by other code than the current one (rerun without "
                             f"--page-only to rebuild them): {stale}")
        page, page_bytes, total, notes = build_page(
            out, metas, blobs, argv, mode="catalogue", refused=refused, set_name=set_name,
            refused_labels={c: refused_label(by_id[c]) for c in list(refused) + list(deferred)},
            deferred=deferred)
    else:
        page, page_bytes, total, notes = build_page(out, metas, blobs, argv)
    skipped.update({c: ["not exported"] for c in missing if c not in skipped})
    mp, man = write_manifest(out, metas, blobs, page, page_bytes, argv, skipped, notes,
                             data_from=src if src != out else None,
                             **({"set_name": set_name, "specs": specs, "modules": list(a.modules), "stale": stale,
                                 "deferred": deferred, "checks": checks} if cat else {}))
    print(f"page {page}: {page_bytes} bytes, {len(metas)} clips, data {total} bytes; missing {missing}")
    if cat:
        print(f"refused (validity): {refused}")
        print(f"not built (needs a family left out): {deferred}")
        print(f"clips with per-time texts: {man['clips_with_overlays']}; scene parts left out (not boxes): "
              f"{man['scene_left_out']}")
        print(f"validity reported, clip kept ({', '.join(REPORTED_ONLY)}): {man['reported_checks']}")
        print(f"skeleton checks flagged ({man['skeleton_checks_read']} clips read): "
              f"{man['skeleton_checks_flagged']}; clips drawn on their level 0's floor: {len(man['page_floors'])}")
        if stale:
            print(f"stale data kept (--allow-stale): {stale}")
    if notes:
        print(f"labels that fell back (listed in the manifest, 'unlabelled'): {notes}")
    if a.copy_to:
        print(f"copy: {copy_out(out, a.copy_to)}")
    unexplained = [c for c in missing if c not in refused and c not in deferred] if cat else missing
    if src != out:
        print("no run record: a page built from another folder's data (--data-from) is a preview")
    elif not a.no_record and not unexplained:
        print(f"record: {write_record(man, out, record_path(set_name))}")


# ---------------------------------------------------------------- the public gallery (--public, 2026-11-10)
# The movement gallery of the MoveAtlas site: the catalogue page rebuilt from the skeleton_v2 data (--page-only,
# --data-from) into <store>/synth_viewer/gallery_open/, for a public reader. Rules (the owner's request of
# 2026-11-10): no clip whose build read a motion-capture take, nor any clip whose level-0 chain holds one
# (public_provenance, measured by CaptureReads at export, not declared by the CAPTURED table, which is only cross-
# checked); joint positions and the drawing landmarks only (public_blob keeps the keys the page draws and refuses
# any mesh, shape or model array, PUBLIC_FORBIDDEN_KEYS); the page's texts for a public reader (PUBLIC_PATCHES) and a
# scan of the final page, its decoded data and the manifest (public_scan: local paths, file names of the tree,
# internal task codes and version tags, words naming a writing tool, its maker or its model, stored as sha256
# digests of lower-case tokens as MoveAtlas's build_public_release.py stores its lists), which refuses the build on a
# hit; a manifest in the interface of MoveAtlas's site/gen_movement_gallery.py (public_manifest).
SET_PUBLIC = "gallery_open"            # plan phase E: the open build (gallery_public: the SMPL+H gallery, 2026-11-10)
OUT_PUBLIC = f"{V.VIEWER_ROOT}/{SET_PUBLIC}"
DATA_PUBLIC = f"{V.VIEWER_ROOT}/catalogue_open"   # the catalogue data the open build reads (--set catalogue --out)
PUBLIC_PAGE = "movement_gallery.html"
PUBLIC_MANIFEST = "movement_gallery_manifest.json"
GENERATED = "generated"                # the provenance of a shown clip: every frame made by the generator
# Open-body port, wave 7 (docs/OPEN_BODY_PORT_DESIGN.md section 9): the body model is Anny, whose licences carry no
# non-commercial clause (code Apache-2.0, model data CC0 1.0). Plan phase E (section 21): every clip of the gallery is
# made by the program's generator, none from a motion-capture recording and none with an SMPL family model, so the
# gallery's motion data ship under CC BY 4.0
PUBLIC_LICENCE = ("CC BY 4.0 (Creative Commons Attribution 4.0 International, "
                  "creativecommons.org/licenses/by/4.0)")
PUBLIC_CREDIT = ("Body model: Anny 0.6.2 (NAVER LABS Europe), code under the Apache License 2.0, model data (the "
                 "MakeHuman assets it ships) under CC0 1.0; every movement made by the program's generator from "
                 "stated parameters, none from a motion-capture recording")
PUBLIC_NOTICE = ("SYNTHETIC: joint positions of a body model; movements made by a generator from stated parameters, "
                 "not recordings of any person; not reviewed by a neurologist; licence: " + PUBLIC_LICENCE)
PUBLIC_DATA_KEYS = ("label", "clip_id", "heading_deg", "scene_boxes_m", "walk", "supine", "skeleton")
PUBLIC_FORBIDDEN_KEYS = frozenset({"vertices", "verts", "faces", "betas", "shape", "shapedirs", "posedirs",
                                   "v_template", "j_regressor", "weights", "kintree_table", "mesh"})
LEFT_OUT_REASONS = {   # reason: the page's words for it, one clip and several (in this order, a clip under its first)
    "motion_capture_take": ("built on a motion-capture recording, whose licence does not allow its redistribution",
                            "built on motion-capture recordings, whose licences do not allow their redistribution"),
    "level0_chain_motion_capture": ("whose level 0 is built on a motion-capture recording",
                                    "whose level 0 is built on a motion-capture recording"),
    "provenance_unrecorded": ("whose provenance was not recorded at its build",
                              "whose provenance was not recorded at their build"),
    "level0_chain_unrecorded": ("whose level 0 has no recorded provenance", "whose level 0 has no recorded provenance"),
    "validity_check_failed": ("whose build fails the validity checks or stops on an error",
                              "whose build fails the validity checks or stops on an error"),
    "not_built": ("not built", "not built"),
    "skeleton_check_failed": ("whose skeleton, or its level 0's, fails the page's own checks",
                              "whose skeleton, or their level 0's, fails the page's own checks"),
    "invisible_on_skeleton": ("whose skeleton equals its level 0's on every frame",
                              "whose skeleton equals their level 0's on every frame"),
    "level0_without_shown_symptom": ("level 0 clip left without a symptom clip on the page",
                                     "level 0 clips left without a symptom clip on the page"),
    "no_data": ("without data", "without data"),
}

# Words naming a writing tool, its maker or its model (lower-case tokens), never in clear (the comment above):
# BLOCK in any case. The digest of a word: hashlib.sha256(word.lower().encode("utf-8")).hexdigest().
PUBLIC_WORD_SHA256 = frozenset({
    "c857d09db23e6822e3600bc06ad8d58f92ed62bc8efd81c753f77048662cb97d",
    "c70eca6b0f88f44d81a41311647e50fda1ac454ec04ffd442b0eb4743a993131",
    "7d3194f79e645c42e4396dda38be04766810ec6a00d00aced3ffc2a0a1f1a9ef",
    "60965168ce762e949600281ba6d01fee136e5b6e8257b1f216f9025ed324474c",
    "053ea4804ef1bb33d4a3d6fb024a614b6d257cebc2bc7cd915da9c9522f37ffc",
    "5d72436256ada53828b51895a94bb8489e9f1ac4fe937a8024ef1594e7045ff6",
    "3ea125d0bff386e6754b3782b300016fc79a9cf8f8669c0a5c3db64467ddb681",
    "487b91042c7cf27a19e23ea8699f5f354b1a0c3af9e418138dc6150d830f970d",
    "de2187af3fb011708ca4ff6d21e4845236d6a6d25e7e90531b26a7923d16c73f",
    "e12ce8285efc67c6d93d3a122e2589ed95089bcbb775ba5634d94e2b8385db07",
    "c9ad8f2cc1294afa0ef22fc2c019ff7243cdd272b2147ddca3f34c5036b05768",
    "5a5110ebe1544b31e6d4207a419bc1deb148735f89ae984ff0a2ab809b51f1df",
    "09cf980b5ff304ac11b7f6d2c5c263da2a867425798ef5cc5d2ebcf55c4fcd23",
    "920510199770f4d65cb8aaa2cd12bdb2b8c37f5b3907c50a734a0e3409da2823",
    "fc5a1047f5919892fcdf8aa79ea5d6bb6531b5c176939ef0110906cb225941c1",
    "a39a7ffad4a3013f29da97b84f264337f234c1cf9b3c40c7c30c677a8a18609a",
    "eea82f53ecaed277e542e8a867863be69c50efc4b94dad1f50a33944ca0c2f1f",
    "386ebc0efd04339aa982703eb666c335d8458b8657343d9783185debf0bf0c26",
    "a5ea7d719f7aba21d3f940a263d619ebb130a56a7faf669f8e333bcf59b65603",
})
# The two-letter abbreviations of the field, English and French: BLOCK when written in capitals only.
PUBLIC_ACRONYM_SHA256 = frozenset({
    "32e83e92d45d71f69dcf9d214688f0375542108631b45d344e5df2eb91c11566",
    "26138d8a757dd1dc09df6301bc4a47f0a1f7f785a725ae72574c6862127a2201",
})
# Phrases of two or three lower-case tokens joined by one space (the field's name in English and French, the model
# noun, a commit trailer, a 'made with' phrase): BLOCK.
PUBLIC_PHRASE_SHA256 = frozenset({
    "803b3eb2b6af027151950019f2d5ede506a6636a9eafd262b27c23e3f834056f",
    "5eff592cc497fccb48ebfffaca38a83f9167bc1661b4477e1fc1b66b658cd5a0",
    "442e56863b7e9cf893fd1d134d4b24b57e98acc6a3f8501d163d042259065871",
    "7c38d5864015fc722b62d2f5b8cf69be50bcea73de4cfb745c8b9780e9e8d9f5",
    "2e8aae79d9566c6391cbf216d0b40760bde9bd51a6c5b0b970a248f679cd718f",
    "dd882975dc879b308d2838ec8c35771979b0903aed6f19236a32594aff572bde",
})
# The letter prefixes of the program's internal task codes (a prefix then a digit, one token): BLOCK.
PUBLIC_TASK_PREFIX_SHA256 = frozenset({
    "11b9c0c881e29c65f3815063db28972b8ea9557c905617eeab3cfd24d7426e57",
    "89c4ec9f6b3f1086b158d8ef03dfe8155e6f79d9e66434b8f9b3432fe8720e50",
    "be18b85f77fc024db379acf19e8a1ce62307ab7bb1bca395389ecfc2dafaf741",
    "32f30ea0e83c41a331c4330213005568675f7542c25f354ebb634db78cc30d12",
})
PUBLIC_PATTERNS = [    # (regex, kind): what a public text never holds, by form (no word list here)
    (r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", "local path"),
    (r"(?i)[\\/](?:Users|Volumes|home)[\\/]", "local path"),
    (r"(?i)\b(?:mdfm|reports|scripts|docs|cache|workflows|tests|datasets)[\\/]", "file name of the tree"),
    (r"(?i)\b[\w-]+\.(?:py|md|ipynb|csv|ya?ml|npz|npy|pkl|pt|gz|rtf|docx)\b", "file name of the tree"),
    (r"\bSYNTHETIC_\w+", "file name of the tree"),
    (r"(?i)\b(?:exam|catalogue|skeleton|physio|sp\d|gallery)[_-]v\d+\b", "internal version tag"),
    (r"(?i)\bworktrees?\b|\bhandoffs?\b", "internal word"),
    (r"\b[a-z][a-z_]*\.[A-Z][A-Z0-9_]+\b", "code identifier"),
]
CODE_REF = re.compile(r"\s*\((?:[a-z_]+\.)+[A-Za-z_]+\)")   # a code identifier in brackets in a clip's text
_PUB_WORD = re.compile(r"[A-Za-z0-9]+")
_PUB_TASK = re.compile(r"^([A-Za-z]+)(\d[A-Za-z0-9]*)$")


def _sha(token):
    return hashlib.sha256(token.lower().encode("utf-8")).hexdigest()


def public_scan_text(text):
    """[(kind, excerpt)] of the hits of the public rules in one text (PUBLIC_PATTERNS and the hashed lists); the
    excerpt is the matched token or phrase, with its neighbours for a pattern."""
    hits = []
    for pat, kind in PUBLIC_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append((kind, text[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")))
    words = _PUB_WORD.findall(text)
    seen = set()
    for i, w in enumerate(words):
        d = _sha(w)
        kind = None
        if d in PUBLIC_WORD_SHA256:
            kind = "word naming a writing tool or its model"
        elif d in PUBLIC_ACRONYM_SHA256 and w.isupper():
            kind = "abbreviation of the field"
        else:
            m = _PUB_TASK.match(w)
            if m and _sha(m.group(1)) in PUBLIC_TASK_PREFIX_SHA256:
                kind = "internal task code"
        if kind and (kind, w) not in seen:
            seen.add((kind, w))
            hits.append((kind, w))
        for n in (2, 3):
            if i + n <= len(words) and _sha(" ".join(words[i:i + n])) in PUBLIC_PHRASE_SHA256:
                hits.append(("phrase naming the field or a writing tool", " ".join(words[i:i + n])))
    for bad in ("\u2014", "\u2013"):
        if bad in text:
            hits.append(("dash", repr(bad)))
    return hits


CLIPDATA = re.compile(r'<script type="text/plain" class="clipdata" data-clip="([^"]*)">([^<]*)</script>')


def public_scan(page, manifest_text):
    """{where: [(kind, excerpt)]} of the hits in the final page (its text, then each embedded data block decoded) and
    the manifest; {} when clean."""
    out = {}
    hits = public_scan_text(CLIPDATA.sub("", page))
    if hits:
        out["page"] = hits
    for m in CLIPDATA.finditer(page):
        hits = public_scan_text(gzip.decompress(base64.b64decode(m.group(2))).decode("utf-8"))
        if hits:
            out[f"data {m.group(1)}"] = hits
    hits = public_scan_text(manifest_text)
    if hits:
        out["manifest"] = hits
    return out


def _forbidden_keys(x, path=""):
    """[paths] of the keys of PUBLIC_FORBIDDEN_KEYS anywhere in a JSON value (lists read on their first items)."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() in PUBLIC_FORBIDDEN_KEYS:
                out.append(f"{path}/{k}")
            out += _forbidden_keys(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x[:4]):
            out += _forbidden_keys(v, f"{path}/{i}")
    return out


def public_blob(blob):
    """The public data file of a catalogue data file (gzip bytes): the keys the page draws (PUBLIC_DATA_KEYS, the
    floor's height alone, the landmarks' joints and offsets without their rules), the public notice; no code record,
    provenance block, texts or rules. Refused when a key of PUBLIC_FORBIDDEN_KEYS remains (a mesh, shape or model
    array)."""
    d = json.loads(gzip.decompress(blob).decode("utf-8"))
    out = {k: d[k] for k in PUBLIC_DATA_KEYS if k in d}
    out["notice"] = PUBLIC_NOTICE
    out["floor"] = {"floor_y_m": d["floor"]["floor_y_m"]}
    sk = dict(out["skeleton"])
    sk["landmarks"] = {k: {"joint": v["joint"], "offset_m": v["offset_m"]} for k, v in sk.get("landmarks", {}).items()}
    out["skeleton"] = sk
    bad = _forbidden_keys(out)
    if bad:
        raise RuntimeError(f"{d.get('clip_id')}: a mesh, shape or model key in the public data: {bad}")
    return gzip.compress(json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
                         compresslevel=9, mtime=0)


def own_provenance(meta):
    """'generated', 'motion_capture' or 'unrecorded' of a loaded clip, from its data file's 'provenance'."""
    p = meta.get("provenance")
    if not isinstance(p, dict) or not isinstance(p.get("captured"), bool):
        return "unrecorded"
    return "motion_capture" if p["captured"] else GENERATED


def public_provenance(metas, specs):
    """({clip: own provenance} of the loaded clips, {clip: reason} of those the gallery leaves out for provenance): a
    clip built on a motion-capture take ('motion_capture_take'), a clip without a provenance record
    ('provenance_unrecorded'), then every clip whose level-0 chain (its level 0, that clip's level 0, ...) holds a
    clip that is not 'generated' or has no data ('level0_chain_motion_capture', 'level0_chain_unrecorded'). The
    CAPTURED table is checked against the measure: a clip it names that measured otherwise stops the build (the
    measure missed a read)."""
    own = {m["clip_id"]: own_provenance(m) for m in metas}
    named = [c for c, p in own.items() if re.match(CAPTURED, c) and p != "motion_capture"]
    if named:
        raise SystemExit(f"provenance: {named} are built on a motion-capture take (CAPTURED) but their build "
                         "recorded no read of a capture store; the measure is not trusted")
    level0 = {s.clip_id: s.level0 for s in specs}
    out = {}
    for c, p in own.items():
        if p == "motion_capture":
            out[c] = "motion_capture_take"
        elif p == "unrecorded":
            out[c] = "provenance_unrecorded"
    for c in own:
        if c in out:
            continue
        x, seen = level0.get(c), {c}
        while x and x not in seen:
            seen.add(x)
            if own.get(x) == "motion_capture":
                out[c] = "level0_chain_motion_capture"
                break
            if own.get(x) != GENERATED:
                out[c] = "level0_chain_unrecorded"
                break
            x = level0.get(x)
    return own, out


def left_out_reasons(specs, shown, refused, deferred, hidden, provenance_out):
    """{reason: [ids]} of every clip of the set not shown, each under its first reason (LEFT_OUT_REASONS order)."""
    def reason(c):
        if c in provenance_out:
            return provenance_out[c]
        if c in refused and c not in hidden:
            return "validity_check_failed"
        if c in deferred:
            return "not_built"
        v = hidden.get(c) or refused.get(c) or []
        if any(str(x).startswith(SKELETON_FAIL) or x == LEVEL0_REFUSED for x in v):
            return "skeleton_check_failed"
        if list(v) == [INVISIBLE]:
            return "invisible_on_skeleton"
        if list(v) == [LONE_LEVEL0]:
            return "level0_without_shown_symptom"
        return "no_data"
    out = {}
    for s in specs:
        if s.clip_id not in shown:
            out.setdefault(reason(s.clip_id), []).append(s.clip_id)
    return {k: sorted(out[k]) for k in LEFT_OUT_REASONS if k in out}


def public_note(left_out):
    """The page's sentence on the catalogue's clips not in the gallery, by reason, with their numbers."""
    n = sum(len(v) for v in left_out.values())
    if not n:
        return "Every clip of the program's catalogue is shown."
    parts = [f"{len(v)} {LEFT_OUT_REASONS[k][0 if len(v) == 1 else 1]}" for k, v in left_out.items()]
    return (f"{n} clip{'s' if n != 1 else ''} of the program's catalogue {'is' if n == 1 else 'are'} not in this "
            f"gallery: {'; '.join(parts)}.")


def public_groups(idx):
    """[{name, n_clips, scales}] of the page's groups in the list's order: the group's name (TX.GROUP_NAMES), its
    shown clips, and the scales its clips' texts name (TX.scales_named, in TX.SCALE_NAMES order)."""
    out = []
    for g in idx["family_order"]:
        cs = [c for c in idx["clips"] if c["family"] == g]
        found = set()
        for c in cs:
            found.update(TX.scales_named([str(c["text"].get(k) or "")
                                          for k in ("item", "scale", "operator", "grade", "params")]))
        out.append({"name": TX.GROUP_NAMES[g], "n_clips": len(cs), "scales": [s for s in TX.SCALE_NAMES if s in found]})
    return out


def public_index(idx):
    """The catalogue index for the public page, in place: no export version per clip ('since'), the provenance line
    of each caption without the program's marks ('marks'; the meaning stays), and no code identifier in brackets in
    a text (CODE_REF: '(handtasks.FIST)' in a specification's parameter text)."""
    for c in idx["clips"]:
        c.pop("since", None)
        t = c["text"]
        t["provenance"] = [{"about": CODE_REF.sub("", e["about"]), "meaning": CODE_REF.sub("", e["meaning"])}
                           for e in t.get("provenance") or []]
        for k, v in t.items():
            if isinstance(v, str):
                t[k] = CODE_REF.sub("", v)
    return idx


def strip_code_comments(template):
    """The page template without its CSS block comments and its script's line comments (a comment line, or a comment
    after code at two spaces or more): the comments name the program's tasks. The template's strings hold no '//'."""
    t = re.sub(r"/\*.*?\*/", "", template, flags=re.S)
    t = re.sub(r"(?m)^[ \t]*//[^\n]*\n", "", t)
    return re.sub(r"(?m)(\S)[ \t]{2,}//[^\n]*$", r"\1", t)


def tree_head(root=ROOT):
    """The commit checked out in this tree (a worktree's .git file followed to its own HEAD); '' when unreadable."""
    from mdfm.train.provenance import moveatlas_head
    g = os.path.join(root, ".git")
    if not os.path.isfile(g):
        return moveatlas_head(root)
    try:
        with open(g, encoding="utf-8") as f:
            line = f.read().strip()
        d = line[len("gitdir: "):].strip() if line.startswith("gitdir: ") else ""
        with open(os.path.join(d, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
    except OSError:
        return ""
    return "" if head.startswith("ref: ") else head


def public_manifest(idx, left_out, page_bytes, page_sha, data_bytes, created=None):
    """The manifest of the public gallery, in the interface of MoveAtlas's site/gen_movement_gallery.py: public, the
    clips shown (the page's order), their provenance ('generated', each), the numbers of symptom and level 0 clips,
    the groups (public_groups), the clips left out by reason, the date, the page's size and sha256; and the
    licence, the credit, the format and the commits it was built from."""
    clips = [c["id"] for c in idx["clips"]]
    n_l0 = sum(1 for c in idx["clips"] if not c["level0"])
    return {"label": LB.LABEL, "public": True, "notice": PUBLIC_NOTICE, "clips": clips,
            "provenance": {c: GENERATED for c in clips},
            "provenance_rule": "a clip is 'generated' when its build read no motion-capture store, nor did the build "
                               "of any clip of its level-0 chain (measured at export: every file the build opened)",
            "n_symptom_clips": len(clips) - n_l0, "n_level0_clips": n_l0, "groups": public_groups(idx),
            "left_out": left_out, "n_left_out": sum(len(v) for v in left_out.values()),
            "created_utc": created or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "page": PUBLIC_PAGE, "page_bytes": page_bytes, "page_sha256": page_sha, "data_bytes": data_bytes,
            "frames": sum(c["n_frames"] for c in idx["clips"]),
            "format": {"positions": "per clip, gzip of UTF-8 JSON embedded as base64: joint and landmark positions in "
                                    "mm, rounded to 1 mm, delta coded over frames", "fps": FPS, "unit": "mm",
                       "up": "y", "points": "the 22 body joints and the 30 hand joints of the Anny body model, and "
                                            "drawing landmarks fixed in a joint's frame; no mesh, shape coefficient "
                                            "or model array"},
            "licence": PUBLIC_LICENCE, "credit": PUBLIC_CREDIT,
            "source": {"generator_commit": tree_head(), "moveatlas_commit": RMV.code_record()["moveatlas_head"]}}


def build_public_page(metas, blobs, left_out):
    """(page text, index, public blobs, notes) of the public gallery: the catalogue page (page_index_catalogue,
    catalogue_template with PUBLIC_PATCHES and without code comments) on the public data files (public_blob) and the
    public index (public_index); the page's own text checks (_page_checked)."""
    pub = {c: public_blob(b) for c, b in blobs.items()}
    sizes = {k: len(b) for k, b in pub.items()}
    idx, notes = page_index_catalogue(metas, sizes, SET_PUBLIC)
    public_index(idx)
    template = catalogue_template({}, segments=[c["segment"] for c in idx["clips"]], public=public_note(left_out))
    total = sum(sizes.values())
    if total * 4 / 3 > EMBED_LIMIT_BYTES:
        raise SystemExit(f"{total} bytes of data: over the embedding limit, the page would need fetch")
    blocks = "\n".join(f'<script type="text/plain" class="clipdata" data-clip="{cid}">'
                       f'{base64.b64encode(b).decode("ascii")}</script>' for cid, b in pub.items())
    js = json.dumps(idx, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = (template.replace("__INDEX_JSON__", js).replace("__DATA_BLOCKS__", blocks)
            .replace("__NOTICE__", PUBLIC_NOTICE).replace("__LICENCE__", PUBLIC_LICENCE)
            .replace("__N_CLIPS__", str(len(metas))))
    _page_checked(page)
    return page, idx, pub, notes


def run_public(a, argv):
    """--public: the public gallery from the catalogue data of --data-from (default catalogue_open) into --out
    (default gallery_open): the internal page's own selection (page_selection; the refusals read from the data
    folder), then the provenance rule (public_provenance) and the level 0s it leaves alone (lone_level0s); the page,
    its scan (public_scan: a hit refuses the build, nothing is written) and the manifest; the run record
    reports/synth/viewer/gallery_open.json (refused if it exists, --no-record to skip)."""
    if a.set != "catalogue" or not a.page_only:
        raise SystemExit("--public goes with --set catalogue --page-only")
    out = (a.out or OUT_PUBLIC).rstrip("/").replace("\\", "/")
    src = (a.data_from or DATA_PUBLIC).rstrip("/").replace("\\", "/")
    V.check_out(out, V.VIEWER_ROOT)
    V.check_out(src, V.VIEWER_ROOT)
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(out)):
        raise SystemExit("--data-from names the output folder itself")
    specs = set_specs(a.set, a.modules)
    by_id = {s.clip_id: s for s in specs}
    metas, blobs = load_all(src, specs)
    now = CodeState(specs, blocked_modules(a.modules))
    stale = {m["clip_id"]: why for m in metas
             if (why := stale_reason(m.get("code"), now.of(by_id[m["clip_id"]]))) is not None}
    if stale and not a.allow_stale:
        raise SystemExit(f"{len(stale)} data files built by other code than the current one: {stale}")
    refused, deferred = read_refused(src), read_refused(src, "deferred")
    own, prov_out = public_provenance(metas, specs)
    metas, blobs, hidden, _ = page_selection(metas, blobs, specs)
    metas = [m for m in metas if m["clip_id"] not in prov_out]
    blobs = {c: b for c, b in blobs.items() if c not in prov_out}
    while True:                                        # a level 0 the provenance rule leaves without a symptom clip
        lone = lone_level0s(metas, specs, {m["clip_id"] for m in metas})
        if not lone:
            break
        hidden.update(lone)
        metas = [m for m in metas if m["clip_id"] not in lone]
        blobs = {c: b for c, b in blobs.items() if c not in lone}
    shown = {m["clip_id"] for m in metas}
    bad = [c for c in shown if re.match(CAPTURED, c) or own.get(c) != GENERATED]
    if bad:
        raise SystemExit(f"provenance: {bad} would be shown")
    left_out = left_out_reasons(specs, shown, refused, deferred, hidden, prov_out)
    page, idx, pub, notes = build_public_page(metas, blobs, left_out)
    raw = page.encode("utf-8")
    man = public_manifest(idx, left_out, len(raw), sha256_bytes(raw), sum(len(b) for b in pub.values()))
    man_text = json.dumps(man, indent=1, ensure_ascii=False) + "\n"
    hits = public_scan(page, man_text)
    if hits:
        raise SystemExit(f"public scan: the page or the manifest holds what a public text never holds: "
                         f"{json.dumps(hits, ensure_ascii=False)[:6000]}")
    if sorted(man["provenance"]) != sorted(man["clips"]) or len(set(man["clips"])) != len(man["clips"]):
        raise SystemExit("manifest: the clips and their provenance disagree")
    os.makedirs(out, exist_ok=True)
    for name, b in ((PUBLIC_PAGE, raw), (PUBLIC_MANIFEST, man_text.encode("utf-8"))):
        with open(f"{out}/{name}", "wb") as f:
            f.write(b)
    print(f"public gallery {out}/{PUBLIC_PAGE}: {len(raw)} bytes, sha256 {man['page_sha256']}, {len(man['clips'])} "
          f"clips ({man['n_symptom_clips']} symptom, {man['n_level0_clips']} level 0), {len(man['groups'])} groups")
    print(f"left out: { {k: len(v) for k, v in left_out.items()} }")
    print(f"measured provenance of the loaded clips: "
          f"{ {p: sum(1 for x in own.values() if x == p) for p in sorted(set(own.values()))} }")
    if notes:
        print(f"labels that fell back: {notes}")
    if not a.no_record:
        p = record_path(SET_PUBLIC)
        if os.path.exists(p):
            raise SystemExit(f"{p} exists: a run record is never rewritten")
        rec = {k: man[k] for k in ("label", "public", "page", "page_bytes", "page_sha256", "data_bytes", "frames",
                                   "n_symptom_clips", "n_level0_clips", "groups", "left_out", "n_left_out",
                                   "created_utc", "source")}
        rec.update({"n_clips": len(man["clips"]), "out": out, "data_from": src, "stale": stale,
                    "measured_provenance": dict(sorted(own.items())),
                    "captured_reads": {m["clip_id"]: m["provenance"] for m in load_all(src, specs)[0]
                                       if own.get(m["clip_id"]) == "motion_capture"},
                    "script": script_record(argv), "page_code": RMV.code_record()})
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"record: {p}")


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SYNTHETIC Skeleton catalogue</title>
<meta name="description" content="__NOTICE__">
<style>
:root{
  --bg:#fcfcfb; --panel:#ffffff; --ink:#1d2127; --muted:#5b626d; --faint:#8b929c; --line:#e3e5e0; --line2:#d2d5cf;
  --accent:#2c5a8a; --accent-bg:#e9f0f7; --mark:#9b1717; --left:#1f63b0; --right:#c4561b; --mid:#3a3f47;
  --sym:#7a2e8f; --ref:#2f7a4a;
}
*{box-sizing:border-box}
[hidden]{display:none !important}
html,body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
body{height:100vh;display:flex;flex-direction:column;overflow:hidden}
button{font:inherit;color:inherit}
header.top{flex:none;display:flex;align-items:center;gap:10px 14px;flex-wrap:wrap;padding:8px 16px;
  border-bottom:1px solid var(--line);background:var(--bg)}
.mark{background:var(--mark);color:#fff;font-weight:700;letter-spacing:.06em;padding:3px 9px;border-radius:3px;
  font-size:13px;white-space:nowrap}
.notice{color:var(--muted);font-size:12.5px}
.apptitle{font-weight:600;margin-left:auto;font-size:13.5px;color:var(--ink)}
.apptitle small{color:var(--faint);font-weight:400}
#toggleList{display:none}
.app{flex:1;min-height:0;display:grid;grid-template-columns:310px minmax(0,1fr)}
aside{border-right:1px solid var(--line);overflow:auto;padding:10px 10px 24px}
.filter{width:100%;padding:7px 9px;border:1px solid var(--line2);border-radius:6px;background:#fff;font:inherit;
  margin-bottom:8px}
details.fam{margin:4px 0 10px}
details.fam>summary{cursor:pointer;font-weight:650;padding:5px 4px;list-style:none;display:flex;align-items:center;
  gap:6px;border-radius:5px}
details.fam>summary::-webkit-details-marker{display:none}
details.fam>summary::before{content:"\25B8";color:var(--faint);font-size:11px;transition:transform .15s}
details.fam[open]>summary::before{transform:rotate(90deg)}
details.fam>summary:hover{background:#f1f2ef}
.count{color:var(--faint);font-weight:400;font-size:12px}
.item{margin:2px 0 8px 10px}
.itemtitle{font-size:12px;color:var(--muted);padding:4px 4px 2px;font-weight:600}
.entry{display:block;width:100%;text-align:left;border:0;background:none;padding:5px 8px;border-radius:5px;
  cursor:pointer;line-height:1.3}
.entry:hover{background:#f0f2ee}
.entry.on{background:var(--accent-bg);color:var(--accent);font-weight:600}
.entry .gl{display:block;font-size:11.5px;color:var(--faint);font-weight:400;margin-top:1px}
main{overflow:auto;padding:12px 18px 28px;min-width:0}
.crumb{color:var(--muted);font-size:12.5px}
h1{font-size:19px;margin:2px 0 8px;font-weight:650;letter-spacing:-.005em}
.row{display:flex;align-items:center;flex-wrap:wrap;gap:6px 10px;margin:6px 0}
.lab{font-size:12px;color:var(--muted);min-width:44px;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{border:1px solid var(--line2);background:#fff;border-radius:999px;padding:3px 11px;cursor:pointer;
  font-size:13px;white-space:nowrap}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
.ghost{border:1px solid var(--line2);background:#fff;border-radius:6px;padding:3px 10px;cursor:pointer;font-size:13px}
.ghost:hover{border-color:var(--accent)}
label.chk{display:inline-flex;align-items:center;gap:5px;font-size:13px;color:var(--ink);cursor:pointer}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.panels.single{grid-template-columns:1fr}
figure.panel{margin:0;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;
  overflow:hidden}
.ptitle{display:flex;align-items:center;gap:8px;padding:7px 10px;border-bottom:1px solid var(--line);font-size:13px}
.tag{font-weight:650;font-size:12px;letter-spacing:.02em;padding:2px 8px;border-radius:4px}
.tag.sym{background:#f3e9f6;color:var(--sym)}
.tag.ref{background:#e7f3eb;color:var(--ref)}
.ptitle .pid{color:var(--faint);font-size:12px;margin-left:auto;font-family:ui-monospace,Consolas,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
canvas{display:block;width:100%;touch-action:none;cursor:grab;background:#fff}
canvas:active{cursor:grabbing}
figcaption{padding:8px 11px 10px;border-top:1px solid var(--line);font-size:13px}
figcaption .op{font-weight:600}
figcaption .gr{color:var(--ink)}
figcaption .pa{color:var(--muted)}
figcaption .meta{color:var(--faint);font-size:11.5px;margin-top:3px}
.transport{display:flex;align-items:center;flex-wrap:wrap;gap:8px 12px;margin:12px 0 4px;padding:8px 10px;
  border:1px solid var(--line);border-radius:8px;background:#fff}
.primary{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer;
  font-weight:600;min-width:92px}
.primary:hover{filter:brightness(1.08)}
#slider{flex:1;min-width:160px;accent-color:var(--accent)}
.time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px;min-width:120px}
.help{color:var(--faint);font-size:12px;margin:6px 2px}
.legend{display:inline-flex;gap:12px;align-items:center;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
details.about{margin-top:14px;border-top:1px solid var(--line);padding-top:8px;color:var(--muted);font-size:13px}
details.about summary{cursor:pointer;color:var(--ink);font-weight:600}
details.about p{margin:6px 0;max-width:80ch}
.err{color:var(--mark);font-weight:600}
@media (max-width:760px){
  body{height:auto;display:block;overflow:auto}
  header.top{position:sticky;top:0;z-index:5;padding:8px 16px}
  .apptitle{margin-left:0;width:100%}
  .notice{display:none}
  #toggleList{display:inline-block}
  .app{display:block}
  aside{display:none;border-right:0;border-bottom:1px solid var(--line);max-height:62vh;padding:10px 16px}
  aside.open{display:block}
  main{padding:10px 16px 24px;overflow:visible}
  .panels{grid-template-columns:1fr}
  .time{min-width:0}
  #slider{min-width:100%;order:5}
}
</style>
</head>
<body>
<header class="top">
  <span class="mark">SYNTHETIC</span>
  <span class="notice">Synthetic body rendered by a body model: not a recording of any person. Internal research
    use.</span>
  <button id="toggleList" class="ghost" type="button" aria-expanded="false">Choose a symptom</button>
  <span class="apptitle">Synthetic symptom catalogue, 3D skeleton <small>(__N_CLIPS__ clips)</small></span>
</header>
<div class="app">
  <aside id="side" aria-label="List of symptoms">
    <input id="filter" class="filter" type="search" placeholder="Filter (e.g. rest, elbow, 3.4)" aria-label="Filter">
    <div id="list"></div>
  </aside>
  <main id="main">
    <div class="crumb" id="crumb"></div>
    <h1 id="title">Loading</h1>
    <div class="row"><span class="lab">Grade</span><div id="grades" class="chips" role="group"
      aria-label="Grade"></div></div>
    <div class="row"><span class="lab">View</span><div id="views" class="chips" role="group" aria-label="View"></div>
      <label class="chk" id="followWrap" hidden><input type="checkbox" id="follow"> Follow the pelvis</label>
      <button id="resetView" class="ghost" type="button">Recentre</button></div>
    <div class="panels" id="panels">
      <figure class="panel" id="figA">
        <div class="ptitle"><span class="tag sym" id="tagA">Symptom</span><span class="pid" id="pidA"></span></div>
        <canvas id="cA" aria-label="Skeleton of the symptom"></canvas>
        <figcaption id="capA"></figcaption>
      </figure>
      <figure class="panel" id="figB">
        <div class="ptitle"><span class="tag ref">Normal (level 0)</span><span class="pid" id="pidB"></span></div>
        <canvas id="cB" aria-label="Skeleton of the level 0"></canvas>
        <figcaption id="capB"></figcaption>
      </figure>
    </div>
    <div class="transport">
      <button id="play" class="primary" type="button">Pause</button>
      <input type="range" id="slider" min="0" max="1" value="0" step="1" aria-label="Time">
      <span id="time" class="time">0.00 s</span>
      <div class="chips" id="speeds" role="group" aria-label="Speed">
        <button class="chip on" type="button" data-s="1">1x</button>
        <button class="chip" type="button" data-s="0.5">0.5x</button>
        <button class="chip" type="button" data-s="0.25">0.25x</button>
      </div>
      <span class="legend"><span><i style="background:var(--left)"></i>left side</span>
        <span><i style="background:var(--right)"></i>right side</span></span>
    </div>
    <p class="help">Drag on a view to turn it, wheel or pinch to zoom, double-click to return to the chosen view.
      Keyboard: space to play or pause, arrows to step one frame forward or back.</p>
    <details class="about">
      <summary>About this catalogue</summary>
      <p>Each clip is a synthetic body of the MD_FM synthetic patient, built by its exam specification (symptom,
        grade, parameters), then reduced to a skeleton: the 22 body joints and the 30 hand joints of the Anny
        body model, at 30 frames per second, positions rounded to the millimetre, on the same clock and the same
        interpolation as the videos of the grey mesh.</p>
      <p>Hands: animated for the finger tasks (3.4, 3.5) and the raised arms; at rest in the wrist's frame for the
        other clips, as on the mesh, which shows pronation-supination. The fingers end at their last joint (the
        distal interphalangeal joint): the distal phalanx is not drawn. Head: a circle and a nose mark, two points
        fixed in the frame of the head joint, to show the rotation and the tilt of the neck.</p>
      <p>The skeleton of the symptom and its level 0 share the same clock (t = 0 at the start of each clip) and the
        same camera; a shorter clip holds its last frame. The floor lies under a flat foot; the chair of the seated
        clips is drawn in thin lines. The symptoms are on the left side of the body (in blue).</p>
      <p>Synthetic bodies: no clip is a recording of a person. The healthy references of walking and standing are
        made by the program's generators, as every other body.</p>
    </details>
  </main>
</div>
<script type="application/json" id="index">__INDEX_JSON__</script>
__DATA_BLOCKS__
<script>
(function () {
"use strict";
var IDX = JSON.parse(document.getElementById("index").textContent);
var CLIPS = new Map();
IDX.clips.forEach(function (c, i) { c.order = i; CLIPS.set(c.id, c); });
var VIEWS = IDX.views, REGIONS = IDX.regions;
var BLOB = new Map();
document.querySelectorAll("script.clipdata").forEach(function (e) { BLOB.set(e.getAttribute("data-clip"), e); });
var D2R = Math.PI / 180, FPS = 30, NEAR = 0.03;
var COL = { L: "#1f63b0", R: "#c4561b", C: "#3a3f47" };
var $ = function (id) { return document.getElementById(id); };
function fmt(x, d) { return x.toFixed(d === undefined ? 2 : d); }
function el(tag, attrs) {
  var e = document.createElement(tag);
  for (var k in (attrs || {})) { if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]); }
  for (var i = 2; i < arguments.length; i++) {
    var c = arguments[i];
    e.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}
function norm(s) { return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase(); }

// ------------------------------------------------------------ series and list
var SERIES = new Map();
IDX.clips.forEach(function (c) {
  var k = c.text.series_key;
  if (!SERIES.has(k)) SERIES.set(k, { key: k, family: c.family, item: c.item, label: c.text.series,
    itemName: c.text.item, famName: c.text.family, clips: [] });
  SERIES.get(k).clips.push(c.id);
});
function defaultClip(s) {
  var cs = s.clips.map(function (id) { return CLIPS.get(id); });
  var g2 = cs.find(function (c) { return c.text.chip === "2"; });
  if (g2) return g2.id;
  var graded = cs.filter(function (c) { return c.level0; });
  return (graded.length ? graded[graded.length - 1] : cs[0]).id;
}
function gradeSummary(s) {
  var labs = s.clips.map(function (id) { return CLIPS.get(id).text.chip.replace(/ \(.*\)/, ""); });
  return (labs.length > 1 ? "grades " : "grade ") + labs.join(", ");
}
function buildList() {
  var list = $("list");
  list.innerHTML = "";
  IDX.family_order.forEach(function (fam) {
    var ss = Array.from(SERIES.values()).filter(function (s) { return s.family === fam; });
    if (!ss.length) return;
    var n = IDX.clips.filter(function (c) { return c.family === fam; }).length;
    var det = el("details", { "class": "fam", open: "" },
      el("summary", {}, IDX.families[fam], el("span", { "class": "count" }, String(n))));
    var items = [];
    ss.forEach(function (s) { if (items.indexOf(s.item) < 0) items.push(s.item); });
    items.forEach(function (it) {
      var grp = el("div", { "class": "item" });
      var s0 = ss.find(function (s) { return s.item === it; });
      grp.append(el("div", { "class": "itemtitle" }, s0.itemName));
      ss.filter(function (s) { return s.item === it; }).forEach(function (s) {
        var b = el("button", { "class": "entry", type: "button", "data-series": s.key }, s.label,
          el("span", { "class": "gl" }, gradeSummary(s)));
        b.dataset.text = norm([s.label, s.itemName, s.famName, s.key].join(" "));
        b.addEventListener("click", function () {
          selectClip(defaultClip(s), { resetTime: true, resetView: true });
          if (window.matchMedia("(max-width:760px)").matches) toggleList(false);
        });
        grp.append(b);
      });
      det.append(grp);
    });
    list.append(det);
  });
}
function applyFilter() {
  var q = norm($("filter").value.trim());
  document.querySelectorAll("details.fam").forEach(function (det) {
    var any = false;
    det.querySelectorAll(".item").forEach(function (grp) {
      var vis = false;
      grp.querySelectorAll(".entry").forEach(function (b) {
        var ok = !q || b.dataset.text.indexOf(q) >= 0 || norm(grp.firstChild.textContent).indexOf(q) >= 0;
        b.hidden = !ok; vis = vis || ok;
      });
      grp.hidden = !vis; any = any || vis;
    });
    det.hidden = !any;
    if (q && any) det.open = true;
  });
}
function toggleList(open) {
  var side = $("side");
  var o = open === undefined ? !side.classList.contains("open") : open;
  side.classList.toggle("open", o);
  $("toggleList").setAttribute("aria-expanded", String(o));
}

// ------------------------------------------------------------ decoding
var CACHE = new Map();
function b64bytes(s) {
  var bin = atob(s), n = bin.length, out = new Uint8Array(n);
  for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function sideOf(name) { return name.indexOf("left_") === 0 ? "L" : (name.indexOf("right_") === 0 ? "R" : "C"); }
function decodeClip(id) {
  if (CACHE.has(id)) return Promise.resolve(CACHE.get(id));
  var e = BLOB.get(id);
  if (!e) return Promise.reject(new Error("no data for " + id));
  if (typeof DecompressionStream === "undefined")
    return Promise.reject(new Error("this browser cannot decompress the data (no DecompressionStream)"));
  var stream = new Blob([b64bytes(e.textContent.trim())]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text().then(function (txt) {
    var d = JSON.parse(txt), sk = d.skeleton, n = sk.n_frames, J = sk.n_points, m = J * 3;
    var q = sk.positions_mm_delta, P = new Float32Array(n * m), acc = new Float64Array(m);
    for (var f = 0; f < n; f++) {
      var o = f * m;
      for (var k = 0; k < m; k++) { acc[k] += q[o + k]; P[o + k] = acc[k] / 1000; }
    }
    var sm = new Float32Array(n * 2), W = Math.round(sk.fps), lo = Math.floor(W / 2);
    for (var f2 = 0; f2 < n; f2++) {                     // centred moving mean of the pelvis over 1 s (edges held)
      var sx = 0, sz = 0;
      for (var w = 0; w < W; w++) {
        var g = Math.min(n - 1, Math.max(0, f2 - lo + w));
        sx += P[g * m]; sz += P[g * m + 2];
      }
      sm[2 * f2] = sx / W; sm[2 * f2 + 1] = sz / W;
    }
    var out = { id: id, n: n, J: J, nj: sk.n_joints, fps: sk.fps, duration: sk.duration_s, P: P, smooth: sm,
      names: sk.names, bones: sk.bones, side: sk.names.map(sideOf), floor: d.floor.floor_y_m,
      boxes: d.scene_boxes_m, walk: d.walk, heading: d.heading_deg, handPose: sk.hand_pose,
      head: sk.names.indexOf("head"), hc: sk.names.indexOf("head_centre"), nose: sk.names.indexOf("nose") };
    CACHE.set(id, out);
    if (CACHE.size > 16) CACHE.delete(CACHE.keys().next().value);
    return out;
  });
}

// ------------------------------------------------------------ geometry
function dirv(phi, el) {
  var p = phi * D2R, e = el * D2R;
  return [Math.sin(p) * Math.cos(e), Math.sin(e), Math.cos(p) * Math.cos(e)];
}
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function unit(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
function basis(d) {
  var zc = unit([-d[0], -d[1], -d[2]]);
  var xc = cross(zc, [0, 1, 0]);
  if (Math.hypot(xc[0], xc[1], xc[2]) < 1e-6) xc = [1, 0, 0];
  xc = unit(xc);
  return { xc: xc, yc: cross(zc, xc), zc: zc };
}
function regionJoints(region, D) {
  var all = [];
  for (var j = 0; j < D.nj; j++) all.push(j);
  var r = REGIONS[region];
  if (!r) return all;
  var sel = r.filter(function (j) { return j < D.nj; });
  return sel.length >= 2 ? sel : all;
}
function presetPhi(name, D) {
  var a = D.walk && D.walk.axis_xz;
  if (a && name === "track_left") return Math.atan2(a[1], -a[0]) / D2R;
  if (a && name === "examiner") return Math.atan2(-a[0], -a[1]) / D2R;
  return D.heading + VIEWS[name].azimuth;
}
function fitCam(pts, d, aspect, vfov, margin) {    // viewer.fit_distance without the caption bands
  var B = basis(d), ty = Math.tan(vfov * D2R / 2) * (1 - margin), tx = ty * aspect;
  var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], i, k;
  for (i = 0; i < pts.length; i += 3) for (k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], pts[i + k]); hi[k] = Math.max(hi[k], pts[i + k]);
  }
  var t = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  function proj(t) {
    var D = -Infinity, zmin = Infinity, xs = [], ys = [], zs = [];
    for (var i = 0; i < pts.length; i += 3) {
      var r0 = pts[i] - t[0], r1 = pts[i + 1] - t[1], r2 = pts[i + 2] - t[2];
      var x = r0 * B.xc[0] + r1 * B.xc[1] + r2 * B.xc[2], y = r0 * B.yc[0] + r1 * B.yc[1] + r2 * B.yc[2];
      var z = r0 * B.zc[0] + r1 * B.zc[1] + r2 * B.zc[2];
      D = Math.max(D, Math.abs(x) / tx - z, Math.abs(y) / ty - z); zmin = Math.min(zmin, z);
      xs.push(x); ys.push(y); zs.push(z);
    }
    return { D: Math.max(D, 0.45 - zmin, 0.3), xs: xs, ys: ys, zs: zs };
  }
  for (var it = 0; it < 3; it++) {
    var p = proj(t), xn0 = Infinity, xn1 = -Infinity, yn0 = Infinity, yn1 = -Infinity;
    for (i = 0; i < p.xs.length; i++) {
      var zz = p.zs[i] + p.D, xn = p.xs[i] / zz, yn = p.ys[i] / zz;
      xn0 = Math.min(xn0, xn); xn1 = Math.max(xn1, xn); yn0 = Math.min(yn0, yn); yn1 = Math.max(yn1, yn);
    }
    for (k = 0; k < 3; k++) t[k] += B.xc[k] * 0.5 * (xn0 + xn1) * p.D + B.yc[k] * 0.5 * (yn0 + yn1) * p.D;
  }
  return { target: t, D: proj(t).D };
}

// ------------------------------------------------------------ state
var S = { clip: null, partner: null, A: null, B: null, t: 0, tmax: 0, playing: true, speed: 1, preset: "front",
  dAz: 0, dEl: 0, zoom: 1, follow: false, fit: null, ready: false, msg: "Loading", token: 0, phi: 0, el: 8,
  vfov: 30 };
var dirty = true;
function isWalk(c) { return c.views.indexOf("track_left") >= 0; }
function presetsFor(c) {
  var base = isWalk(c) ? ["track_left", "examiner", "front", "left", "top"] : ["front", "left", "top"];
  c.views.forEach(function (v) { if (base.indexOf(v) < 0) base.push(v); });
  return base;
}
function captionHtml(fig, c) {
  var cap = $(fig === "A" ? "capA" : "capB");
  cap.innerHTML = "";
  cap.append(el("div", { "class": "op" }, c.text.operator), el("div", { "class": "gr" }, c.text.grade));
  if (c.text.params) cap.append(el("div", { "class": "pa" }, c.text.params));
  var hands = c.hand_pose === "animated" ? "animated hands" : "hands at rest";
  cap.append(el("div", { "class": "meta" }, c.n_frames + " frames, " + fmt(c.duration) + " s, " +
    c.n_joints + " joints (" + hands + "), " + fmt(c.bytes / 1024, 1) + " kB"));
  $(fig === "A" ? "pidA" : "pidB").textContent = c.id;
}
function buildGrades() {
  var s = SERIES.get(S.clip.text.series_key), g = $("grades");
  g.innerHTML = "";
  s.clips.forEach(function (id) {
    var c = CLIPS.get(id);
    var b = el("button", { "class": "chip" + (id === S.clip.id ? " on" : ""), type: "button", title: c.text.grade,
      "aria-pressed": String(id === S.clip.id) }, c.text.chip);
    b.addEventListener("click", function () { selectClip(id, { resetTime: false, resetView: false }); });
    g.append(b);
  });
}
function buildViews() {
  var v = $("views");
  v.innerHTML = "";
  presetsFor(S.clip).forEach(function (name) {
    var b = el("button", { "class": "chip" + (name === S.preset ? " on" : ""), type: "button",
      "aria-pressed": String(name === S.preset) }, VIEWS[name].title);
    b.addEventListener("click", function () { setPreset(name); });
    v.append(b);
  });
  $("followWrap").hidden = !isWalk(S.clip);
  $("follow").checked = S.follow;
}
function setPreset(name) {
  S.preset = name; S.dAz = 0; S.dEl = 0; S.zoom = 1;
  if (isWalk(S.clip)) S.follow = name !== "examiner";
  buildViews(); computeFit(); updateHash(); dirty = true;
}
function markList() {
  document.querySelectorAll(".entry").forEach(function (b) {
    b.classList.toggle("on", b.dataset.series === S.clip.text.series_key);
  });
}
function updateHash() {
  if (!S.clip) return;
  try { history.replaceState(null, "", "#clip=" + S.clip.id + "&view=" + S.preset); } catch (e) { /* file:// */ }
}
function selectClip(id, opt) {
  var c = CLIPS.get(id);
  if (!c) return;
  opt = opt || {};
  S.clip = c; S.partner = c.level0 ? CLIPS.get(c.level0) : null;
  var s = SERIES.get(c.text.series_key);
  $("crumb").textContent = c.text.family + " › " + c.text.item + " › " + c.segment + " " + c.text.segment;
  var ttl = s.label;
  $("title").textContent = ttl.charAt(0).toUpperCase() + ttl.slice(1);
  markList(); buildGrades();
  var pr = presetsFor(c);
  if (opt.resetView || pr.indexOf(S.preset) < 0 || (opt.preset && pr.indexOf(opt.preset) >= 0)) {
    S.preset = (opt.preset && pr.indexOf(opt.preset) >= 0) ? opt.preset : c.views[0];
    S.dAz = 0; S.dEl = 0; S.zoom = 1; S.follow = isWalk(c) && S.preset !== "examiner";
  }
  buildViews();
  captionHtml("A", c);
  $("tagA").textContent = c.level0 ? "Symptom" : "Level 0 (this clip is the reference)";
  $("figB").hidden = !S.partner;
  $("panels").classList.toggle("single", !S.partner);
  if (S.partner) captionHtml("B", S.partner);
  updateHash();
  S.ready = false; S.msg = "Loading"; dirty = true;
  var token = ++S.token;
  resize();
  Promise.all([decodeClip(c.id), S.partner ? decodeClip(S.partner.id) : Promise.resolve(null)]).then(function (r) {
    if (token !== S.token) return;
    S.A = r[0]; S.B = r[1];
    S.tmax = Math.max(S.A.duration, S.B ? S.B.duration : 0);
    $("slider").max = String(Math.round(S.tmax * FPS));
    if (opt.resetTime || S.t > S.tmax) S.t = 0;
    S.ready = true; computeFit(); dirty = true;
  }).catch(function (e) { S.msg = "Cannot read this clip: " + e.message; dirty = true; });
}

// ------------------------------------------------------------ camera
function computeFit() {
  if (!S.ready) return;
  var v = VIEWS[S.preset], p = PANELS[0], aspect = p.w / Math.max(1, p.h);
  S.phi = presetPhi(S.preset, S.A); S.el = v.elevation; S.vfov = v.vfov;
  var d = dirv(S.phi, S.el), pts = [];
  [S.A, S.B].forEach(function (D) {
    if (!D) return;
    var reg = regionJoints(v.region, D), step = Math.max(1, Math.floor(D.n / 300)), m = D.J * 3;
    var head = (v.region === "body" || v.region === "head") && D.hc >= 0;
    for (var f = 0; f < D.n; f += step) {
      var sx = S.follow ? D.smooth[2 * f] : 0, sz = S.follow ? D.smooth[2 * f + 1] : 0;
      for (var i = 0; i < reg.length; i++) {
        var o = f * m + reg[i] * 3;
        pts.push(D.P[o] - sx, D.P[o + 1], D.P[o + 2] - sz);
      }
      if (head) {                                  // the head circle (radius 0.095 m) inside the frame
        var oh = f * m + D.hc * 3;
        pts.push(D.P[oh] - sx, D.P[oh + 1] + 0.1, D.P[oh + 2] - sz, D.P[oh] - sx, D.P[oh + 1] - 0.1, D.P[oh + 2] - sz);
      }
    }
  });
  S.fit = fitCam(pts, d, aspect, v.vfov, v.margin);
  S.grid = makeGrid();
}
function makeGrid() {
  var cx, cz, hx, hz, step, major = 1;
  var walk = isWalk(S.clip);
  if (walk) {
    var lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    [S.A, S.B].forEach(function (D) {
      if (!D) return;
      for (var f = 0; f < D.n; f++) {
        lo[0] = Math.min(lo[0], D.smooth[2 * f]); hi[0] = Math.max(hi[0], D.smooth[2 * f]);
        lo[1] = Math.min(lo[1], D.smooth[2 * f + 1]); hi[1] = Math.max(hi[1], D.smooth[2 * f + 1]);
      }
    });
    cx = (lo[0] + hi[0]) / 2; cz = (lo[1] + hi[1]) / 2; hx = (hi[0] - lo[0]) / 2 + 1.5; hz = (hi[1] - lo[1]) / 2 + 1.5;
    step = 0.5;
  } else {
    cx = S.fit.target[0]; cz = S.fit.target[2]; hx = hz = 2.0; step = 0.25;
  }
  var segs = [], x0 = Math.floor((cx - hx) / step) * step, x1 = cx + hx, z0 = Math.floor((cz - hz) / step) * step,
    z1 = cz + hz;
  for (var x = x0; x <= x1 + 1e-9; x += step)
    segs.push([x, z0, x, z1, Math.abs(x / major - Math.round(x / major)) < 1e-6]);
  for (var z = z0; z <= z1 + 1e-9; z += step)
    segs.push([x0, z, x1, z, Math.abs(z / major - Math.round(z / major)) < 1e-6]);
  return segs;
}
function camera(D, k, p) {
  var el = Math.max(-8, Math.min(88, S.el + S.dEl)), d = dirv(S.phi + S.dAz, el), B = basis(d);
  var t = S.fit.target;
  if (S.follow) t = [t[0] + D.smooth[2 * k], t[1], t[2] + D.smooth[2 * k + 1]];
  var dist = S.fit.D * S.zoom;
  return { eye: [t[0] + d[0] * dist, t[1] + d[1] * dist, t[2] + d[2] * dist], xc: B.xc, yc: B.yc, zc: B.zc,
    f: (p.h / 2) / Math.tan(S.vfov * D2R / 2), cx: p.w / 2, cy: p.h / 2 };
}
function toCam(c, x, y, z) {
  var r0 = x - c.eye[0], r1 = y - c.eye[1], r2 = z - c.eye[2];
  return [r0 * c.xc[0] + r1 * c.xc[1] + r2 * c.xc[2], r0 * c.yc[0] + r1 * c.yc[1] + r2 * c.yc[2],
    r0 * c.zc[0] + r1 * c.zc[1] + r2 * c.zc[2]];
}
function scr(c, q) { return [c.cx + c.f * q[0] / q[2], c.cy + c.f * q[1] / q[2]]; }
function line3(ctx, c, a, b) {                    // a segment in world coordinates, clipped at the near plane
  var p = toCam(c, a[0], a[1], a[2]), q = toCam(c, b[0], b[1], b[2]);
  if (p[2] < NEAR && q[2] < NEAR) return;
  if (p[2] < NEAR || q[2] < NEAR) {
    var s = (NEAR - p[2]) / (q[2] - p[2]), r = [p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]), NEAR];
    if (p[2] < NEAR) p = r; else q = r;
  }
  var u = scr(c, p), v = scr(c, q);
  ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]);
}

// ------------------------------------------------------------ drawing
var PANELS = [{ id: "A", canvas: $("cA") }, { id: "B", canvas: $("cB") }];
PANELS.forEach(function (p) { p.ctx = p.canvas.getContext("2d"); p.w = 400; p.h = 400; });
function resize() {
  var dpr = Math.min(2, window.devicePixelRatio || 1);
  var narrow = window.matchMedia("(max-width:760px)").matches;
  PANELS.forEach(function (p) {
    var fig = p.canvas.parentNode;
    if (fig.hidden) return;
    var w = Math.max(200, p.canvas.clientWidth || fig.clientWidth);
    var h = narrow ? Math.round(Math.min(w * 1.0, 420))
      : Math.round(Math.max(280, Math.min(w * 0.95, window.innerHeight - 380)));
    if (p.w !== w || p.h !== h || p.dpr !== dpr) {
      p.w = w; p.h = h; p.dpr = dpr;
      p.canvas.style.height = h + "px";
      p.canvas.width = Math.round(w * dpr); p.canvas.height = Math.round(h * dpr);
    }
  });
  if (S.ready) computeFit();
  dirty = true;
}
function badge(ctx, p) {
  ctx.font = "700 12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  var w = ctx.measureText("SYNTHETIC").width + 14;
  ctx.fillStyle = "#9b1717"; ctx.fillRect(8, 8, w, 20);
  ctx.fillStyle = "#ffffff"; ctx.textBaseline = "middle"; ctx.fillText("SYNTHETIC", 15, 18.5);
  if (p.w > 300) {                                 // under the mark, clear of the view title and the time (right)
    var note = "synthetic body, not a recording";
    ctx.font = "11px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(8, 30, ctx.measureText(note).width + 8, 16);
    ctx.fillStyle = "#5b626d"; ctx.fillText(note, 12, 38.5);
  }
}
function drawPanel(p, D, isA) {
  var ctx = p.ctx;
  ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, p.w, p.h);
  if (!S.ready || !D || !S.fit) {
    badge(ctx, p);
    ctx.fillStyle = S.msg.indexOf("Cannot") === 0 ? "#9b1717" : "#8b929c";
    ctx.font = "14px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(S.msg, p.w / 2, p.h / 2); ctx.textAlign = "left";
    return;
  }
  var k = Math.min(D.n - 1, Math.round(S.t * FPS)), ended = S.t * FPS > D.n - 1 + 0.5;
  var c = camera(D, k, p), m = D.J * 3, o = k * m, P = D.P, fy = D.floor, i;
  // floor grid
  ctx.lineCap = "butt";
  [false, true].forEach(function (maj) {
    ctx.beginPath();
    S.grid.forEach(function (s) { if (s[4] === maj) line3(ctx, c, [s[0], fy, s[1]], [s[2], fy, s[3]]); });
    ctx.strokeStyle = maj ? "#cfd3cc" : "#e8eae5"; ctx.lineWidth = maj ? 1.2 : 1; ctx.stroke();
  });
  // chair
  if (D.boxes && D.boxes.length) {
    ctx.beginPath();
    D.boxes.forEach(function (b) {
      var X = [b[0], b[3]], Y = [b[1], b[4]], Z = [b[2], b[5]];
      for (var a = 0; a < 2; a++) for (var e = 0; e < 2; e++) {
        line3(ctx, c, [X[0], Y[a], Z[e]], [X[1], Y[a], Z[e]]);
        line3(ctx, c, [X[a], Y[0], Z[e]], [X[a], Y[1], Z[e]]);
        line3(ctx, c, [X[a], Y[e], Z[0]], [X[a], Y[e], Z[1]]);
      }
    });
    ctx.strokeStyle = "rgba(150,112,72,0.55)"; ctx.lineWidth = 1.2; ctx.stroke();
  }
  // camera coordinates of every point
  var cc = new Array(D.J);
  for (i = 0; i < D.J; i++) cc[i] = toCam(c, P[o + 3 * i], P[o + 3 * i + 1], P[o + 3 * i + 2]);
  // shadow on the floor
  ctx.lineCap = "round"; ctx.strokeStyle = "rgba(30,34,40,0.10)";
  D.bones.forEach(function (bn) {
    var a = bn[0], b = bn[1];
    if (a >= D.nj) return;
    var pa = toCam(c, P[o + 3 * a], fy + 0.002, P[o + 3 * a + 2]), pb = toCam(c, P[o + 3 * b], fy + 0.002,
      P[o + 3 * b + 2]);
    if (pa[2] < NEAR || pb[2] < NEAR) return;
    var u = scr(c, pa), v = scr(c, pb);
    ctx.lineWidth = Math.max(1, c.f * (a >= 22 ? 0.006 : 0.022) / ((pa[2] + pb[2]) / 2));
    ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
  });
  // the skeleton, far to near
  var items = [];
  D.bones.forEach(function (bn) {
    var a = bn[0], b = bn[1];
    if (a >= D.nj) return;
    if (cc[a][2] < NEAR || cc[b][2] < NEAR) return;
    items.push([(cc[a][2] + cc[b][2]) / 2, 0, a, b]);
  });
  for (i = 0; i < D.nj; i++) if (cc[i][2] >= NEAR) items.push([cc[i][2], 1, i]);
  if (D.hc >= 0 && cc[D.hc][2] >= NEAR) items.push([cc[D.hc][2] + 0.05, 2, D.hc]);
  items.sort(function (u, v) { return v[0] - u[0]; });
  var rest = D.handPose === "rest";
  items.forEach(function (it) {
    var a = it[2], hand = a >= 22, col = COL[D.side[a]];
    if (it[1] === 0) {
      var u = scr(c, cc[a]), v = scr(c, cc[it[3]]);
      ctx.strokeStyle = col; ctx.globalAlpha = hand && rest ? 0.7 : 1;
      ctx.lineWidth = Math.max(hand ? 0.8 : 1.6, c.f * (hand ? 0.0025 : 0.011) / it[0]);
      ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
      ctx.globalAlpha = 1;
    } else if (it[1] === 1) {
      var q = scr(c, cc[a]), r = Math.max(hand ? 1.0 : 2.2, c.f * (hand ? 0.0038 : 0.016) / it[0]);
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = col; ctx.globalAlpha = hand && rest ? 0.7 : 1; ctx.fill(); ctx.globalAlpha = 1;
      if (!hand) { ctx.lineWidth = 1; ctx.strokeStyle = "#ffffff"; ctx.stroke(); }
    } else {                                        // the head: a circle and the nose
      var hcq = scr(c, cc[D.hc]), rh = c.f * 0.095 / cc[D.hc][2];
      ctx.beginPath(); ctx.arc(hcq[0], hcq[1], rh, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(58,63,71,0.10)"; ctx.fill();
      ctx.lineWidth = Math.max(1.2, c.f * 0.006 / cc[D.hc][2]); ctx.strokeStyle = COL.C; ctx.stroke();
      if (D.nose >= 0 && cc[D.nose][2] >= NEAR) {
        var nq = scr(c, cc[D.nose]);
        ctx.beginPath(); ctx.moveTo(hcq[0], hcq[1]); ctx.lineTo(nq[0], nq[1]);
        ctx.lineWidth = Math.max(1.5, c.f * 0.008 / cc[D.nose][2]); ctx.stroke();
        ctx.beginPath(); ctx.arc(nq[0], nq[1], Math.max(2.5, c.f * 0.014 / cc[D.nose][2]), 0, 2 * Math.PI);
        ctx.fillStyle = "#b8322a"; ctx.fill();
      }
    }
  });
  // overlay
  badge(ctx, p);
  ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif"; ctx.textAlign = "right";
  ctx.fillStyle = "#5b626d"; ctx.textBaseline = "middle";
  ctx.fillText(VIEWS[S.preset].title, p.w - 10, 18.5);
  ctx.fillStyle = "#1d2127";
  ctx.fillText("t = " + fmt(Math.min(S.t, D.duration)) + " s" + (ended ? " (end of clip)" : ""), p.w - 10, 36);
  ctx.textAlign = "left";
}
function draw() {
  drawPanel(PANELS[0], S.A, true);
  if (S.partner) drawPanel(PANELS[1], S.B, false);
  $("time").textContent = fmt(S.t) + " s / " + fmt(S.tmax) + " s";
  var sl = $("slider");
  if (document.activeElement !== sl || S.playing) sl.value = String(Math.round(S.t * FPS));
}
var last = performance.now();
function tick(now) {
  var dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (S.playing && S.ready && S.tmax > 0) {
    S.t += dt * S.speed;
    if (S.t > S.tmax + 0.4) S.t = 0;              // hold the last frame briefly, then loop
    dirty = true;
  }
  if (dirty) { draw(); dirty = false; }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------ interaction
function setPlaying(on) { S.playing = on; $("play").textContent = on ? "Pause" : "Play"; dirty = true; }
$("play").addEventListener("click", function () { setPlaying(!S.playing); });
$("slider").addEventListener("input", function (e) { S.t = Number(e.target.value) / FPS; dirty = true; });
document.querySelectorAll("#speeds .chip").forEach(function (b) {
  b.addEventListener("click", function () {
    S.speed = Number(b.dataset.s);
    document.querySelectorAll("#speeds .chip").forEach(function (x) { x.classList.toggle("on", x === b); });
  });
});
$("resetView").addEventListener("click", function () { S.dAz = 0; S.dEl = 0; S.zoom = 1; dirty = true; });
$("follow").addEventListener("change", function (e) { S.follow = e.target.checked; computeFit(); dirty = true; });
$("filter").addEventListener("input", applyFilter);
$("toggleList").addEventListener("click", function () { toggleList(); });
PANELS.forEach(function (p) {
  var cv = p.canvas, ptrs = new Map(), pinch = 0;
  cv.addEventListener("pointerdown", function (e) {
    cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 2) {
      var v = Array.from(ptrs.values());
      pinch = Math.hypot(v[0][0] - v[1][0], v[0][1] - v[1][1]);
    }
  });
  cv.addEventListener("pointermove", function (e) {
    if (!ptrs.has(e.pointerId)) return;
    var prev = ptrs.get(e.pointerId);
    ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 1) {
      S.dAz -= (e.clientX - prev[0]) * 0.45;
      S.dEl = Math.max(-8 - S.el, Math.min(88 - S.el, S.dEl + (e.clientY - prev[1]) * 0.35));
    } else if (ptrs.size === 2) {
      var v = Array.from(ptrs.values()), d = Math.hypot(v[0][0] - v[1][0], v[0][1] - v[1][1]);
      if (pinch > 0) S.zoom = Math.max(0.12, Math.min(6, S.zoom * pinch / d));
      pinch = d;
    }
    dirty = true;
  });
  function up(e) { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; }
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("wheel", function (e) {
    e.preventDefault();
    S.zoom = Math.max(0.12, Math.min(6, S.zoom * Math.exp(e.deltaY * 0.0012))); dirty = true;
  }, { passive: false });
  cv.addEventListener("dblclick", function () { S.dAz = 0; S.dEl = 0; S.zoom = 1; dirty = true; });
});
document.addEventListener("keydown", function (e) {
  var tag = (document.activeElement && document.activeElement.tagName) || "";
  if (tag === "INPUT" && document.activeElement.type !== "range") return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!S.playing); }
  else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
    e.preventDefault(); setPlaying(false);
    S.t = Math.max(0, Math.min(S.tmax, Math.round(S.t * FPS + (e.code === "ArrowRight" ? 1 : -1)) / FPS));
    dirty = true;
  }
});
window.addEventListener("resize", resize);

// ------------------------------------------------------------ start
function parseHash() {
  var h = {};
  location.hash.replace(/^#/, "").split("&").forEach(function (kv) {
    var i = kv.indexOf("="); if (i > 0) h[kv.slice(0, i)] = decodeURIComponent(kv.slice(i + 1));
  });
  return h;
}
window.addEventListener("hashchange", function () {
  var h = parseHash();
  if (h.clip && CLIPS.has(h.clip) && (h.clip !== S.clip.id || (h.view && h.view !== S.preset)))
    selectClip(h.clip, { resetTime: true, resetView: true, preset: h.view });
});
buildList();
var h0 = parseHash();
var first = (h0.clip && CLIPS.has(h0.clip)) ? h0.clip
  : defaultClip(SERIES.get("tremor_E1_rest") || SERIES.values().next().value);
selectClip(first, { resetTime: true, resetView: true, preset: h0.view });
requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""


# The per-time texts of a clip (task SKELV3): c.text.overlays of the index ({kind, from, to, text}, the truth's), drawn
# at the bottom of the clip's own view while from <= t < to (a frame text whose span reaches the clip's end stays on
# its last frame); a stimulus mark on an amber band, a frame text on a dark one, stacked when several are on. The
# badge (the SYNTHETIC mark and its note) stays over the skeleton: the fit keeps the body below the rows of the
# view's texts (fitBands, CATALOGUE_JS), so that the note no longer hides the head of a standing body.
OVERLAY_JS = r"""function drawOverlays(ctx, p, D, isA) {
  var c = isA ? S.clip : S.partner, ov = c && c.text && c.text.overlays;
  if (!ov || !ov.length) return;
  var tt = Math.min(S.t, D.duration), y = p.h - 10;
  ctx.font = "700 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ov.forEach(function (o) {
    var on = tt >= o.from - 1e-6 && (tt < o.to || (o.kind === "frame_text" && o.to >= D.duration - 1e-6));
    if (!on) return;
    ctx.font = "700 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    var fs = Math.max(9, Math.min(14, 14 * (p.w - 36) / ctx.measureText(o.text).width));   // a long text fits
    ctx.font = "700 " + fs.toFixed(1) + "px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
    var w = Math.min(p.w - 16, ctx.measureText(o.text).width + 20), x = Math.max(8, (p.w - w) / 2);
    ctx.fillStyle = o.kind === "stimulus" ? "#a86400" : "rgba(29,33,39,0.85)";
    ctx.fillRect(x, y - 26, w, 26);
    ctx.fillStyle = "#ffffff"; ctx.fillText(o.text, x + 10, y - 12.5);
    y -= 32;
  });
}
"""


# The catalogue page after the SKELV3 reviews: the landmarks of the catalogue export drawn as short lines (the foot's
# heel, lateral border and big toe, the thumb and index pulps; IDX.landmark_bones) and the ear line through the head
# circle; the fit keeps the body clear of the rows at the top of a view (the SYNTHETIC mark and its note, the view's
# title, the time) and of the band of the per-time texts at the bottom (fitBands); in a close-up (a view whose
# region is not the whole body) discs and bones are capped (closeCap), as the camera stands near; the view title
# and the time sit on white boxes (labelBox), readable when a bone crosses them; the hands are drawn in the rest
# style only when both clips have their hands at rest (handRest).
CATALOGUE_JS = r"""var LANDMARK_BONES = IDX.landmark_bones || [], REGION_LM = IDX.region_landmarks || {};
var TOP_ROWS = 50, TOP_ROWS_NARROW = 64, NARROW_W = 420, BOTTOM_BAND = 44, BOTTOM_EDGE = 6;
function extraBones(names) {                    // [[point, point, style]] of the landmark lines a clip carries
  var out = [];
  LANDMARK_BONES.forEach(function (b) {
    var i = names.indexOf(b[0]), j = names.indexOf(b[1]);
    if (i >= 0 && j >= 0) out.push([i, j, b[2]]);
  });
  return out;
}
function withLandmarks(region, D, sel) {        // the landmarks of a view's region join its joints in the fit
  var add = REGION_LM[region] || [], out = sel.slice();
  add.forEach(function (n) { var i = D.names.indexOf(n); if (i >= 0) out.push(i); });
  return out;
}
function hasOverlays() {
  return [S.clip, S.partner].some(function (c) { return c && c.text && c.text.overlays && c.text.overlays.length; });
}
function fitBands(pts, d, aspect, v, p) {       // fitCam inside the rows left free by the texts of the view
  var top = (p.w < NARROW_W ? TOP_ROWS_NARROW : TOP_ROWS) / p.h;
  var bot = (hasOverlays() ? BOTTOM_BAND : BOTTOM_EDGE) / p.h;
  var k = Math.max(0.3, 1 - top - bot), B = basis(d), cu = Math.tan(v.vfov * D2R / 2) * (top - bot);
  var fit = fitCam(pts, d, aspect / k, v.vfov, 1 - (1 - v.margin) * k);   // the vertical half range times k
  for (var i = 0; i < 3; i++) fit.target[i] -= B.yc[i] * cu * fit.D;     // the body's centre between the rows
  return fit;
}
function closeCap() {                           // the largest disc radius and line width (px) of the view
  return VIEWS[S.preset].region === "body" ? { bone: Infinity, disc: Infinity } : { bone: 6, disc: 7 };
}
function handRest(D) {                          // the rest style of the hands only when both clips rest them
  return D.handPose === "rest" && !(S.A && S.B && (S.A.handPose !== "rest" || S.B.handPose !== "rest"));
}
function labelBox(ctx, text, x, y, color) {     // right-aligned text on a white box
  var w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(x - w - 4, y - 8, w + 8, 16);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}
function drawExtraBone(ctx, c, cc, it, D, cap) {   // a landmark line: foot outline, big toe, thumb or index pulp
  var a = it[2], b = it[3], st = it[4], u = scr(c, cc[a]), v = scr(c, cc[b]), fin = st === "finger";
  ctx.strokeStyle = COL[D.side[a]]; ctx.globalAlpha = fin && handRest(D) ? 0.7 : 1;
  ctx.lineWidth = Math.min(cap.bone, Math.max(fin ? 0.8 : 1.2, c.f * (fin ? 0.0025 : 0.007) / it[0]));
  ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]); ctx.stroke();
  if (st === "finger" || st === "toe") {        // a disc at the tip
    ctx.beginPath();
    ctx.arc(u[0], u[1], Math.min(cap.disc, Math.max(fin ? 1.2 : 1.6, c.f * (fin ? 0.004 : 0.006) / cc[a][2])), 0,
      2 * Math.PI);
    ctx.fillStyle = COL[D.side[a]]; ctx.fill();
  }
  ctx.globalAlpha = 1;
}
var BOX_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
function boxEdges(ctx, c, b) {                  // an oriented box: its 8 corners in the vertex order of viewer.box
  BOX_EDGES.forEach(function (e) { line3(ctx, c, b[e[0]], b[e[1]]); });
}
function shadowY(D) {                           // the shadow's plane: the couch top of a supine clip, else the floor
  if (!D.supine || !D.boxes || !D.boxes.length) return D.floor;
  var top = -Infinity;
  D.boxes.forEach(function (b) {
    top = Math.max(top, b.length === 8 ? Math.max.apply(null, b.map(function (q) { return q[1]; })) : b[4]);
  });
  return top;
}
function provenanceLine(pv) {                   // wave SH1d: the provenance of the caption's values, under a chip
  var ul = el("ul", {});
  pv.forEach(function (e) { ul.append(el("li", {}, el("span", { "class": "pva" }, e.about), ": " + e.meaning)); });
  return el("details", { "class": "prov" }, el("summary", {}, "Provenance (" + pv.length + ")"), ul);
}
function drawEars(ctx, c, cc, D, cap) {         // the ear line through the head circle: the roll of the head
  var l = D.ears[0], r = D.ears[1];
  if (l < 0 || r < 0 || cc[l][2] < NEAR || cc[r][2] < NEAR) return;
  var u = scr(c, cc[l]), v = scr(c, cc[r]);
  ctx.beginPath(); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]);
  ctx.lineWidth = Math.min(cap.bone, Math.max(1.2, c.f * 0.005 / cc[D.hc][2])); ctx.strokeStyle = COL.C; ctx.stroke();
  [[l, COL.L], [r, COL.R]].forEach(function (e) {
    var q = scr(c, cc[e[0]]);
    ctx.beginPath(); ctx.arc(q[0], q[1], Math.min(cap.disc, Math.max(2, c.f * 0.012 / cc[e[0]][2])), 0, 2 * Math.PI);
    ctx.fillStyle = e[1]; ctx.fill();
  });
}
"""


# The catalogue page (--set catalogue) is PAGE_TEMPLATE with these replacements, so that the exam page stays the
# same: groups and headings come from the index (catalogue_rows), the title and the list line of a series from its
# texts, the crumb shows the scale item, and the about section says what changes with the catalogue (hands, sides,
# order, refused clips).
CATALOGUE_PATCHES = [
    ("<title>SYNTHETIC Skeleton catalogue</title>", "<title>SYNTHETIC Symptom catalogue, skeleton</title>"),
    # wave SH1d (PLAN M5): a clip is drawn on the floor and with the scene boxes of the index (its level 0's,
    # page_floor), not its data file's own; the provenance of the caption's values folded under a chip
    ("floor: d.floor.floor_y_m,", "floor: CLIPS.get(id).floor,"),
    ("boxes: d.scene_boxes_m,", "boxes: CLIPS.get(id).boxes,"),
    ('  if (c.text.params) cap.append(el("div", { "class": "pa" }, c.text.params));\n',
     '  if (c.text.params) cap.append(el("div", { "class": "pa" }, c.text.params));\n'
     '  if (c.text.provenance && c.text.provenance.length) cap.append(provenanceLine(c.text.provenance));\n'),
    ("@media (max-width:760px){\n",
     "details.prov{margin-top:5px;font-size:12px;color:var(--muted)}\n"
     "details.prov>summary{cursor:pointer;display:inline-block;list-style:none;color:var(--faint);"
     "border:1px solid var(--line2);\n  border-radius:999px;padding:1px 9px;font-size:11.5px}\n"
     "details.prov>summary::-webkit-details-marker{display:none}\n"
     "details.prov[open]>summary{color:var(--muted);border-color:var(--accent)}\n"
     "details.prov ul{margin:5px 0 0;padding-left:18px}\n"
     "details.prov .pva{color:var(--ink)}\n"
     "@media (max-width:760px){\n"),
    ('placeholder="Filter (e.g. rest, elbow, 3.4)"', 'placeholder="Filter (e.g. rest, chorea, Romberg, steppage)"'),
    ('  var g2 = cs.find(function (c) { return c.text.chip === "2"; });',
     '  var g2 = cs.find(function (c) { return /^2(?: |$)/.test(c.text.chip); });'),
    ("function drawPanel(p, D, isA) {\n", OVERLAY_JS + CATALOGUE_JS + "function drawPanel(p, D, isA) {\n"),
    # the landmarks: their lines and the ears (decodeClip, drawPanel), their share of the fit (regionJoints)
    ('nose: sk.names.indexOf("nose") };',
     'nose: sk.names.indexOf("nose"),\n      xb: extraBones(sk.names), ears: [sk.names.indexOf("left_ear"), '
     'sk.names.indexOf("right_ear")], supine: !!d.supine };'),
    ("function regionJoints(region, D) {\n",
     "function regionJoints(region, D) { return withLandmarks(region, D, regionJoints0(region, D)); }\n"
     "function regionJoints0(region, D) {\n"),
    # task SKELV4a: a region of one joint (the toes) framed with its landmarks, not as the whole body
    ("  return sel.length >= 2 ? sel : all;\n",
     "  return sel.length + (REGION_LM[region] || []).length >= 2 ? sel : all;\n"),
    # task SKELV4a: oriented boxes by their corners; a supine clip: no pelvis following, its shadow on the couch top
    ("    D.boxes.forEach(function (b) {\n      var X = [b[0], b[3]], Y = [b[1], b[4]], Z = [b[2], b[5]];\n",
     "    D.boxes.forEach(function (b) {\n      if (b.length === 8) { boxEdges(ctx, c, b); return; }\n"
     "      var X = [b[0], b[3]], Y = [b[1], b[4]], Z = [b[2], b[5]];\n"),
    ('function isWalk(c) { return c.views.indexOf("track_left") >= 0; }',
     'function isWalk(c) { return c.views.indexOf("track_left") >= 0 && !c.supine; }'),
    ("    var pa = toCam(c, P[o + 3 * a], fy + 0.002, P[o + 3 * a + 2]), pb = toCam(c, P[o + 3 * b], fy + 0.002,\n",
     "    var pa = toCam(c, P[o + 3 * a], SY + 0.002, P[o + 3 * a + 2]), pb = toCam(c, P[o + 3 * b], SY + 0.002,\n"),
    ("  for (i = 0; i < D.nj; i++) if (cc[i][2] >= NEAR) items.push([cc[i][2], 1, i]);\n",
     "  D.xb.forEach(function (x) {\n"
     "    if (cc[x[0]][2] >= NEAR && cc[x[1]][2] >= NEAR) items.push([(cc[x[0]][2] + cc[x[1]][2]) / 2, 3, x[0], x[1], "
     "x[2]]);\n"
     "  });\n"
     "  for (i = 0; i < D.nj; i++) if (cc[i][2] >= NEAR) items.push([cc[i][2], 1, i]);\n"),
    ("    } else {                                        // the head: a circle and the nose\n",
     "    } else if (it[1] === 3) {\n      drawExtraBone(ctx, c, cc, it, D, CAP);\n"
     "    } else {                                        // the head: a circle and the nose\n"),
    ("      ctx.lineWidth = Math.max(1.2, c.f * 0.006 / cc[D.hc][2]); ctx.strokeStyle = COL.C; ctx.stroke();\n",
     "      ctx.lineWidth = Math.max(1.2, c.f * 0.006 / cc[D.hc][2]); ctx.strokeStyle = COL.C; ctx.stroke();\n"
     "      drawEars(ctx, c, cc, D, CAP);\n"),
    # close-ups: capped discs and lines; the hands' rest style only when both clips rest them
    ("  var c = camera(D, k, p), m = D.J * 3, o = k * m, P = D.P, fy = D.floor, i;\n",
     "  var c = camera(D, k, p), m = D.J * 3, o = k * m, P = D.P, fy = D.floor, i;\n  var CAP = closeCap();\n"
     "  var SY = shadowY(D);\n"),
    ("    ctx.lineWidth = Math.max(1, c.f * (a >= 22 ? 0.006 : 0.022) / ((pa[2] + pb[2]) / 2));",
     "    ctx.lineWidth = Math.min(2 * CAP.bone, Math.max(1, c.f * (a >= 22 ? 0.006 : 0.022) / ((pa[2] + pb[2]) / "
     "2)));"),
    ("      ctx.lineWidth = Math.max(hand ? 0.8 : 1.6, c.f * (hand ? 0.0025 : 0.011) / it[0]);",
     "      ctx.lineWidth = Math.min(CAP.bone, Math.max(hand ? 0.8 : 1.6, c.f * (hand ? 0.0025 : 0.011) / it[0]));"),
    ("      var q = scr(c, cc[a]), r = Math.max(hand ? 1.0 : 2.2, c.f * (hand ? 0.0038 : 0.016) / it[0]);",
     "      var q = scr(c, cc[a]), r = Math.min(CAP.disc, Math.max(hand ? 1.0 : 2.2, c.f * (hand ? 0.0038 : 0.016) / "
     "it[0]));"),
    ('  var rest = D.handPose === "rest";\n', "  var rest = handRest(D);\n"),
    # the fit inside the free rows; the view title and the time on white boxes, the time on a third row when narrow
    ("  S.fit = fitCam(pts, d, aspect, v.vfov, v.margin);\n", "  S.fit = fitBands(pts, d, aspect, v, p);\n"),
    ("  ctx.fillText(VIEWS[S.preset].title, p.w - 10, 18.5);\n  ctx.fillStyle = \"#1d2127\";\n"
     "  ctx.fillText(\"t = \" + fmt(Math.min(S.t, D.duration)) + \" s\" + (ended ? \" (end of clip)\" : \"\"), "
     "p.w - 10, 36);\n",
     "  labelBox(ctx, VIEWS[S.preset].title, p.w - 10, 18.5, \"#5b626d\");\n"
     "  labelBox(ctx, \"t = \" + fmt(Math.min(S.t, D.duration)) + \" s\" + (ended ? \" (end of clip)\" : \"\"), "
     "p.w - 10,\n"
     "    p.w < NARROW_W ? 54 : 36, \"#1d2127\");\n"),
    ("  // overlay\n  badge(ctx, p);\n", "  // overlay\n  badge(ctx, p);\n  drawOverlays(ctx, p, D, isA);\n"),
    ("    itemName: c.text.item, famName: c.text.family, clips: [] });",
     '    itemName: c.text.item, famName: c.text.family, summary: c.text.summary, search: c.text.search || "", '
     'clips: [] });'),
    ("function gradeSummary(s) {\n", "function gradeSummary(s) {\n  if (s.summary) return s.summary;\n"),
    ('b.dataset.text = norm([s.label, s.itemName, s.famName, s.key].join(" "));',
     'b.dataset.text = norm([s.label, s.itemName, s.famName, s.search].join(" "));'),
    ('<meta name="description" content="__NOTICE__">',
     '<meta name="description" content="SYNTHETIC: joint positions of a body model of the MD_FM synthetic patient '
     '(symptom catalogue, skeleton); not a recording of any person; internal research use only">'),
    ('<span class="tag ref">Normal (level 0)</span>', '<span class="tag ref" id="tagB">Normal (level 0)</span>'),
    ('  $("tagA").textContent = c.level0 ? "Symptom" : "Level 0 (this clip is the reference)";\n',
     '  $("tagA").textContent = c.level0 ? "Symptom" : (POSTURE_ONLY.test(c.id)\n'
     '    ? "Dystonic posture only (the reference)" : "Level 0 (this clip is the reference)");\n'),
    ('  if (S.partner) captionHtml("B", S.partner);\n',
     '  if (S.partner) captionHtml("B", S.partner);\n'
     '  $("tagB").textContent = S.partner && POSTURE_ONLY.test(S.partner.id)\n'
     '    ? "Dystonic posture only (without the symptom)" : "Normal (level 0)";\n'),
    ("var dirty = true;\n",
     "var dirty = true;\nvar POSTURE_ONLY = /_posture(_only)?$/;   // a level 0 that keeps the dystonic posture\n"),
    ("built by its exam specification (symptom,\n        grade, parameters)",
     "built by its specification (symptom,\n        grade, parameters)"),
    ('$("crumb").textContent = c.text.family + " › " + c.text.item + " › " + c.segment + " " + c.text.segment;',
     '$("crumb").textContent = [c.text.family, c.text.item, c.text.scale, c.segment + " " + c.text.segment]\n'
     '    .filter(function (x) { return x; }).join(" › ");'),
    ("var ttl = s.label;", "var ttl = c.text.title || s.label;"),
    ("      <p>Hands: animated for the finger tasks (3.4, 3.5) and the raised arms; at rest in the wrist's frame for the"
     "\n        other clips,",
     "      <p>Hands: animated when the clip's body carries finger rotations (each clip's caption says so: animated\n"
     "        hands or hands at rest); at rest in the wrist's frame for the other clips,"),
    ("clips is drawn in thin lines. The symptoms are on the left side of the body (in blue).</p>",
     "clips (with or without armrests), the chair of the arising, the writing table, the examiner's finger of\n"
     "        the finger-to-nose test and the examination couch of the supine clips are drawn in thin lines (boxes). A\n"
     "        symptom clip is drawn on its level 0's floor and with its level 0's chair, so that the two stand on one\n"
     "        floor; that floor lies under the level 0's feet where they stand still. A supine clip lies on the\n"
     "        couch: its floor is the room's, under the couch, and the skeleton's shadow falls on the couch. The\n"
     "        left side of the body is blue, the right side orange: most symptoms are on the left, and each clip's\n"
     "        caption names the side.</p>\n"
     "      <p>Texts on the view: the eye state of the Romberg test and of the walks with the eyes open or closed\n"
     "        (EYES OPEN, EYES CLOSED), the dual task of the functional jerks (DUAL TASK) and the stimuli of the jerks\n"
     "        (SOUND, TAP left hand or right hand) appear at the bottom of the view, in the clip's own words, at the\n"
     "        times the clip gives; each view shows those of its own clip. The neutral level 0 of a startle or of a\n"
     "        reflex myoclonus receives no stimulus and shows none.</p>"),
    ("        made by the program's generators, as every other body.</p>\n",
     "        made by the program's generators, as every other body.</p>\n"
     "      <p>Order of the list: tremors, the finger-to-nose test, bradykinesia, the other signs of parkinsonism,\n"
     "        atypical parkinsonism (PSP), parkinsonian postures, chorea, ballism and athetosis, dyskinesias,\n"
     "        stereotypies, akathisia and restless legs, mirror movements, myoclonus, negative myoclonus, tics,\n"
     "        startle, clonus, dystonia, paroxysmal dyskinesias, ataxia, freezing of gait, gait, then the healthy\n"
     "        references; in each group, the exam's items, then the catalogue's clinical entities. The texts of each\n"
     "        clip (symptom, grade, parameters) are those of its specification. The scales (MDS-UPDRS, TETRAS, SARA,\n"
     "        ICARS, UMRS, YGTSS, MRVS, SCATS, UHDRS, UDysRS, AIMS, BARS, TWSTRS, BFMDRS, WCRS, DIS, Conn, FTM\n"
     "        (Fahn-Tolosa-Marin), Woods and Teuber, PSPRS, S-FMDRS) keep their names.</p>\n"
     "      <p>Provenance: the captions give the clinical content. The Provenance chip under a caption lists, phrase\n"
     "        by phrase, where its values come from: a value the generator chose with no published source, a grading\n"
     "        of the generator's own (with no published scale) or its own reading of a published grade as a movement,\n"
     "        a value from the clip's cited sources, an amplitude read at the wrist by the program's reading rule for\n"
     "        a forearm tremor, a grade 1 placed against the detection floor of the program's amplitude estimator, a\n"
     "        starting posture of the program's video draft.</p>\n"
     "      <p>Identifiers: the code in small type at the top of each view (for example\n"
     "        gait_E8_parkinsonian_g2) is the clip's internal identifier, and the code before the segment in the\n"
     "        breadcrumb (__SEGMENTS__) the exam segment's; they find the files and are not clinical texts.</p>\n"
     "      <p>Refused clips: __REFUSED_NOTE__</p>\n"),
    ("The fingers end at their last joint (the\n        distal interphalangeal joint): the distal phalanx is not drawn. "
     "Head: a circle and a nose mark, two points\n        fixed in the frame of the head joint, to show the rotation "
     "and the tilt of the neck.</p>",
     "The fingers end at their last joint (the\n"
     "        distal interphalangeal joint), but for the thumb and the index, drawn to the pulp to show the pinch.\n"
     "        Head: a circle, a nose mark and the line of the ears, points fixed in the frame of the head joint, to\n"
     "        show the rotation, the flexion and the tilt of the neck. Feet: the heel and the lateral border of the\n"
     "        forefoot, fixed in the frame of the ankle, and the tip of the big toe, fixed in the frame of the toes, to\n"
     "        show the inversion of the foot and the movements of the toes. These marks are read on the neutral rest\n"
     "        mesh: they are drawing marks, not mesh vertices.</p>\n"
     "      <p>Views: each clip opens on the view that shows its sign best; the other views stay at hand. The body is\n"
     "        framed below the texts at the top of the view and, when the clip carries some, above the texts at the\n"
     "        bottom; in a close-up, the joints and the segments keep a moderate size.</p>"),
]

# The public gallery (--public): the catalogue page (CATALOGUE_PATCHES) with these replacements, for a public reader:
# its title and notices, the about section's sentence on the internal catalogue in public words (generated movements
# only, not reviewed by a neurologist, the body model's licences and credit, and the motion data's licence,
# PUBLIC_LICENCE, CC BY 4.0 since plan phase E), no
# reference to the program's videos or files, and the clips the gallery leaves out by reason (public_note).
PUBLIC_PATCHES = [
    ("<title>SYNTHETIC Symptom catalogue, skeleton</title>", "<title>SYNTHETIC Movement gallery</title>"),
    ('<meta name="description" content="SYNTHETIC: joint positions of a body model of the MD_FM synthetic patient '
     '(symptom catalogue, skeleton); not a recording of any person; internal research use only">',
     '<meta name="description" content="__NOTICE__">'),
    ("Synthetic body rendered by a body model: not a recording of any person. Internal research\n    use.",
     "Synthetic movements made by a body model: not a recording of any person; not reviewed by a\n"
     "    neurologist; licence: __LICENCE__."),
    ("Synthetic symptom catalogue, 3D skeleton <small>", "Movement gallery of the synthetic patient <small>"),
    ("<summary>About this catalogue</summary>", "<summary>About this gallery</summary>"),
    ("positions rounded to the millimetre, on the same clock and the same\n        interpolation as the videos of the "
     "grey mesh.</p>", "positions rounded to the millimetre.</p>"),
    ("at rest in the wrist's frame for the other clips, as on the mesh, which shows pronation-supination.",
     "at rest in the wrist's frame for the other clips, which still shows pronation-supination."),
    ("      <p>Synthetic bodies: no clip is a recording of a person. The healthy references of walking and standing are\n"
     "        made by the program's generators, as every other body.</p>\n",
     "      <p>Synthetic movements: no clip is a recording of a person, and none shows a patient. Every movement of\n"
     "        the gallery is made by the program's generator from stated parameters, the healthy movements of the\n"
     "        level 0 clips included; the clips of the program's catalogue built on a motion-capture recording are\n"
     "        left out (below). The clips have not been reviewed by a movement-disorder neurologist: they illustrate\n"
     "        how a sign moves and are not a reference for diagnosis, for training raters or for scoring a scale.</p>\n"
     "      <p>Licence and credits: the joint positions are computed with the Anny body model (version 0.6.2, NAVER\n"
     "        LABS Europe), whose code is under the Apache License 2.0 and whose model data (the MakeHuman assets it\n"
     "        ships) are under CC0 1.0. The gallery's motion data: __LICENCE__. The gallery holds no mesh, no shape\n"
     "        coefficient and no array of the model. The scales named in the captions belong to their authors and\n"
     "        publishers; the gallery names their items and grades and reproduces none of their forms.</p>\n"),
    ("is the clip's internal identifier, and the code before the segment in the\n        breadcrumb (__SEGMENTS__) the "
     "exam segment's; they find the files and are not clinical texts.",
     "is the clip's identifier, and the code before the segment in the\n        breadcrumb (__SEGMENTS__) the exam "
     "segment's; they name the clips and are not clinical texts."),
    ("      <p>Refused clips: __REFUSED_NOTE__</p>\n", "      <p>Clips not shown: __PUBLIC_NOTE__</p>\n"),
]

if __name__ == "__main__":
    main()
