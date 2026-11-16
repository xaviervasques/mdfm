"""Render the synthetic patient's exam set as mesh videos a clinician can watch (SP4 version (a) of
docs/proposals/SYNTH_PATIENT_DRAFT.md, section 9): every clip of mdfm.synth.exam.exam_set drawn as the grey SMPL+H
mesh by the numpy rasterizer of mdfm/synth/raster.py, from the views of mdfm.synth.viewer.VIEWS, at 30 frames per
second, into H.264 MP4 files and PNG posters under <store>/synth_viewer only (never in this tree), each frame
marked SYNTHETIC and captioned with the operator and the grade, with a truth JSON per clip and an index page
(SYNTHETIC_exam_index.html, a local file with plain video tags and no script).

CPU only (mdfm environment), no network, no GPU. Run from <repo>:
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    <conda>/envs/mdfm/python.exe scripts/render_mesh_video.py --set exam_v2 --shard 0/4
then, once every shard has finished, the index and the run record:
    ... scripts/render_mesh_video.py --set exam_v2 --index-only

--select-stand runs the quiet-stand selection rule of the AMASS stand reference (mdfm.synth.smplh_reference.select_stand,
once exam.select_stand; behind MDFM_SMPLH_REFERENCE=1, the module never ships) on the train body store and writes
reports/synth/viewer/stand_selection.json.

A clip is refused (nothing written) when its truth fails a check: a new joint-limit violation, the foot contact or
the leg continuity (contact.sole_report, contact.step_report). Each clip's floor is the body's own mesh floor
(viewer.scene_floor); its residual over every frame is in the clip record and the run record. exam_v1 (2026-10-05)
placed the floor from the neutral rest mesh and is superseded by exam_v2 (review of 2026-10-05).

Task SP5 (exam_v3): the exam motions the first set lacked (finger tapping 3.4, hand movements 3.5 with the SMPL+H
hand joints, the arms raised to the 3.15 posture, the arising of 3.9) are the clips whose spec says since
'exam_v3'; --since exam_v3 renders only those into a new set folder, and --index-only --with-earlier exam_v2 writes
an index page holding the new clips and, by relative links, the earlier set's (never rewritten), with --summary
reports/synth/viewer_v3.md a copy of the run record's table in the tree. A clip whose truth fails the hand joint
limits (mdfm/synth/hands.py) is refused like the others.

Review of task SP5 (2026-10-05): without --since, the clips of the set named by --set are planned (the specs whose
since is that set; a set name that is no exam version needs --since), so a run on exam_v2 no longer plans the
later clips; rendering into a set folder whose manifest exists (a set closed by --index-only) is refused, and so is
--index-only on such a folder or with a run record that exists, before anything is written. --with-earlier takes
several sets, the first taking precedence for a clip held by more than one. exam_v4 holds the exam_v3 clips whose
body changed in the review (the arising, finger tapping, hand movements); its index page links the three unchanged
raise clips of exam_v3 and the exam_v2 clips: --set exam_v4 --index-only --with-earlier exam_v3 exam_v2. A clip
whose truth fails the balance check of the arising (motions.arise_validity) is refused like the others.

--list prints the plan (clips, views, frames) without rendering. --only keeps the clips whose id holds one of the
given strings. Clips already written (their truth JSON exists) are skipped, so an interrupted run resumes. The run
record (reports/synth/viewer/<set>.json and <set>.md, written by --index-only) lists every clip with its views,
frames, file sizes and SHA-256, and the check of every written file (check_files: the MP4 decoded by ffprobe holds
the recorded frame count, size and frame rate and carries SYNTHETIC in its title and comment; the PNG poster
carries the SYNTHETIC text chunk; the SHA-256 matches); it is refused if it exists (a record is never rewritten).

Shared wave of the symptom catalogue review (2026-11-02, docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md):
- --catalogue plans the symptom catalogue (mdfm.synth.catalogue.catalogue_set: the exam set and the catalogue
  modules) instead of the exam set, so --set catalogue_v1 --catalogue renders the catalogue's clips; the index page
  lists the families outside FAMILY_ORDER after it, in the order of the plan.
- The clip's truth is passed to viewer.render_view, which burns its per-frame texts (the Romberg eye state, the
  jerks' stimuli) into the frames (SH12).
- Frame rate (SH14): a clip whose truth frequency (params['frequency_hz']) exceeds a third of the frame rate is
  rendered at the smallest multiple of that rate giving at least FRAMES_PER_CYCLE frames per cycle (fps_rule; the
  orthostatic tremor of tremors.HZ, above half of 30 fps, is rendered at 60 fps). Averaging sub-frames over a
  30 fps shutter was not chosen: a box exposure of one frame keeps most of such an oscillation and still folds it
  below the Nyquist rate, so the clip would show a slower false tremor, only dimmer; fps_rule records both numbers
  for the clip (the aliased frequency and the box exposure's gain at the base rate) beside the chosen rate.
- A supine clip (scene 'exam_bed', SARA 8 of ataxia.py) is drawn on the room floor and from the couch's heading of
  ataxia.bed_view, not on the lowest vertex of its frames (the couch top) nor from the root's heading at the middle
  knot (the pelvis turns while a heel is on the other leg).
- A clip whose truth fails the face-touch check of the touching tics (jerks.touch_readback, validity 'face_touch')
  is refused like the others.
- SH2 and SH1 of docs/proposals/SYNTH_SHARED_CHANGES_DRAFT.md (2026-11-02): a clip whose truth fails the scene checks
  of the posture dystonias (postures.py: 'armrest', 'hand_thigh', 'hand_head'), the shoulder coupling of the SH2
  joint-limit table (limits_ext.coupling_report, 'coupling') or the physiological background's own checks
  (physio.py, 'background', written by catalogue.catalogue_set) is refused too.
"""
import argparse
import dataclasses
import datetime
import hashlib
import html
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mdfm.synth import exam as EX  # noqa: E402
from mdfm.synth import viewer as V  # noqa: E402
from mdfm.synth.common import VERSION, truth_json  # noqa: E402
from mdfm.video import labels as LB  # noqa: E402

RECORD_DIR = os.path.join(ROOT, "reports", "synth", "viewer")
INDEX = "SYNTHETIC_exam_index.html"
FAMILY_ORDER = ("reference", "tremor", "freezing", "bradykinesia", "dystonia")
FAMILY_TITLE = {"reference": "Healthy references (AMASS and procedural, neutral shape)",
                "tremor": "Rest and postural tremor", "freezing": "Freezing of gait",
                "bradykinesia": "Bradykinesia (decrement, interruptions, slowing)",
                "dystonia": "Sustained (dystonic) posture"}


def script_record(argv):
    with open(os.path.abspath(__file__), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": "scripts/render_mesh_video.py", "sha256": sha, "argv": list(argv)}


def code_record():
    from mdfm.train.provenance import code_sha256, moveatlas_head
    from mdfm import paths
    return {"synth_code_sha256": code_sha256(os.path.join(ROOT, "mdfm", "synth")), "mdfm_code_sha256": code_sha256(),
            "operator_version": VERSION, "mdfm_git_head": moveatlas_head(paths.TREE),
            "moveatlas_head": moveatlas_head()}


def truth_path(out, clip_id):
    return os.path.join(out, "truth", LB.synthetic_name(f"{clip_id}_truth", ".json")).replace("\\", "/")


FRAMES_PER_CYCLE = 3          # [module] the fewest frames per cycle of a clip's periodic movement
FPS_MAX = 120.0               # [module] the highest frame rate fps_rule gives


def truth_frequency_hz(truth):
    """params['frequency_hz'] of a clip's truth (the tremor's frequency), None when absent or not positive."""
    f = ((truth or {}).get("params") or {}).get("frequency_hz")
    return float(f) if isinstance(f, (int, float)) and not isinstance(f, bool) and f > 0 else None


def alias_hz(f, fps):
    """The frequency (Hz) that f appears at when sampled at fps (folded into 0 to fps / 2)."""
    return abs(float(f) - float(fps) * round(float(f) / float(fps)))


def box_exposure_gain(f, fps):
    """|sinc|: the amplitude kept of a sinusoid of f Hz by averaging over an exposure of one frame (1 / fps s)."""
    x = math.pi * float(f) / float(fps)
    return 1.0 if x == 0 else abs(math.sin(x) / x)


def fps_rule(truth, base_fps=V.FPS):
    """(fps, record) of a clip (module docstring): base_fps, or for a truth frequency above base_fps /
    FRAMES_PER_CYCLE the smallest multiple of base_fps giving FRAMES_PER_CYCLE frames per cycle (at most FPS_MAX);
    record: the frequency, the chosen rate, frames per cycle at both rates, the frequency it would alias to at
    base_fps and the gain of a one-frame box exposure at base_fps."""
    f = truth_frequency_hz(truth)
    if f is None or f <= base_fps / FRAMES_PER_CYCLE:
        return float(base_fps), None
    fps = float(base_fps) * math.ceil(FRAMES_PER_CYCLE * f / float(base_fps) - 1e-9)
    if fps > FPS_MAX:
        raise ValueError(f"{f} Hz needs {fps} fps, above FPS_MAX {FPS_MAX}")
    return fps, {"frequency_hz": f, "base_fps": float(base_fps), "fps": fps, "frames_per_cycle_min": FRAMES_PER_CYCLE,
                 "frames_per_cycle_base": float(base_fps) / f, "frames_per_cycle": fps / f,
                 "alias_hz_at_base": alias_hz(f, base_fps), "box_exposure_gain_at_base": box_exposure_gain(f, base_fps),
                 "rule": "frequency above a third of the frame rate: rendered at the smallest multiple of the base "
                         "rate with at least FRAMES_PER_CYCLE frames per cycle (render_mesh_video.fps_rule)"}


def plan_frames(spec, body):
    return V.n_frames(body.duration)


CHECKS = ("contact", "continuity", "hand_limits", "balance", "face_touch", "armrest", "hand_thigh", "hand_head",
          "coupling", "background", "fold", "knee_clearance", "table", "backrest",
          "hand_trunk")   # the validity entries {'ok': ...} that refuse a clip
#   (failed_checks); an entry without 'ok' is a report (tremors.FaceRoom.check's 'hand_head'), not a check; 'fold'
#   (motions.arise_validity: no skin part of the folded arms inside another, nothing inside the chair) and
#   'knee_clearance' (involuntary.knee_clearance: the seated knees apart) since the wave SH1b; 'table' (postures:
#   the writing hand, the pen and the resting hands on the writing scene, read again on the layered body), 'backrest'
#   and 'hand_trunk' (postures: the arms outside the backrest and the hands outside the trunk) since the wave SH1c


def failed_checks(truth):
    """[names] of the checks a clip's truth fails: limits_new, then each of CHECKS present and not ok (module
    docstring)."""
    v = (truth or {}).get("validity") or {}
    bad = ["limits_new"] if v.get("limits_new") else []
    return bad + [k for k in CHECKS if isinstance(v.get(k), dict) and not v[k].get("ok", True)]


def render_clip(spec, out, opts, argv, png_frames=False, log=print, views=None):
    """Build, render and record one clip; returns its record (also written as the clip's truth JSON). Refuses a
    clip whose truth fails a check (failed_checks), and a view of viewer.NO_SCENE_VIEWS on a clip with a scene (from
    behind, the seat block fills the frame of a seated clip)."""
    t0 = time.time()
    blocked = [v for v in (views or spec.views) if v in V.NO_SCENE_VIEWS] if spec.scene is not None else []
    if blocked:
        raise ValueError(f"{spec.clip_id}: view {blocked} needs a clip without a scene (scene {spec.scene!r})")
    body, truth = spec.build()
    bad = failed_checks(truth)
    if bad:
        raise ValueError(f"{spec.clip_id}: its truth fails {bad}; not rendered")
    fps, fps_rec = fps_rule(truth, opts.fps)
    if fps != opts.fps:
        opts = dataclasses.replace(opts, fps=fps)
    heading = None
    if spec.scene == "exam_bed":       # supine on the couch: the room floor and the couch's heading (ataxia.bed_view)
        from mdfm.synth.ataxia import bed_view
        bv = bed_view(body)
        floor_y, heading = bv["floor_y"], bv["heading_deg"]
        floor_rep = dict(V.floor_report(V.lowest_points(body, opts.fps), floor_y),
                         rule="supine on the couch: the room floor and heading of ataxia.bed_view (the lowest mesh "
                              "vertex lies on the couch top)")
    else:
        floor_y, floor_rep = V.scene_floor(body, opts.fps)
    scene = EX.scene_mesh(spec.scene, body, floor_y)
    cap = spec.caption_spec()
    cap["params_text"] = EX.describe(spec, truth)
    lines = V.caption_lines(cap)
    folder = os.path.join(out, "clips", spec.family).replace("\\", "/")
    recs = []
    for vname in (views or spec.views):
        log(f"{spec.clip_id} / {vname}: {V.n_frames(body.duration, opts.fps)} frames at {opts.fps:g} fps")
        recs.append(V.render_view(body, V.VIEWS[vname], folder, spec.clip_id, lines, opts,
                                  scene=scene, png_frames=png_frames, floor_y=floor_y, truth=truth, heading=heading))
    rec = {"label": LB.LABEL, "notice": V.NOTICE, "clip_id": spec.clip_id, "family": spec.family,
           "segment": spec.segment, "segment_title": EX.segment_title(spec), "item": spec.item,
           "operator": spec.operator, "grade": spec.grade, "grade_text": spec.grade_text,
           "params_text": cap["params_text"], "caption": list(lines), "level0": spec.level0,
           "scene": spec.scene, "since": spec.since, "folder": f"clips/{spec.family}", "views": recs,
           "floor": floor_rep,
           "body": {"T": body.T, "rate_hz": body.rate, "span_s": body.duration, "gender": int(body.gender),
                    "betas": [float(b) for b in body.betas]},
           "truth": truth_json(truth), "code": code_record(), "script": script_record(argv),
           "options": {"width": opts.width, "height": opts.height, "fps": opts.fps, "shading": opts.shading,
                       "supersample": opts.supersample, "shadow": opts.shadow, "cull": opts.cull},
           **({"fps_rule": fps_rec} if fps_rec else {}),
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "seconds": round(time.time() - t0, 1)}
    p = truth_path(out, spec.clip_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    return rec


# ---------------------------------------------------------------- the index page
def _e(x):
    return html.escape(str(x), quote=True)


def _truth_rows(rec):
    """[(key, value)] of the truth parameters shown on the index page."""
    t = rec.get("truth") or {}
    p = t.get("params", {})
    keep = ("side", "driver", "condition", "frequency_hz", "cm_pp", "deg_pp", "share", "latency_s", "centre_shift_deg",
            "task", "item", "criterion", "amplitude0_deg", "level0_hz", "decrement_onset", "decrement_slope",
            "n_hesitations", "n_arrests", "period_s", "region", "direction", "scale", "level_deg", "applied_deg",
            "realized_peak_deg", "reachable", "foot", "plant_scale_min", "duration_grade", "n_trigger", "n_straight",
            "unit", "amplitude0", "open_amplitude", "rate_hz", "slowing_grade", "slowing_ratio",
            "closed_tip_distance_mm", "pinch_gap_mm", "seat_height_m", "rise_s", "liftoff_share",
            "realized_seat_height_m", "realized_hip_flexion_displacement_deg", "realized_hip_peak_share",
            "realized_knee_extension_deg", "realized_ankle_dorsiflexion_deg", "realized_ankle_peak_share",
            "realized_ankle_plantarflexion_deg", "realized_trunk_lean_peak_deg", "seated_shank_tilt_deg",
            "standing_shank_tilt_deg")
    rows = [(k, p[k]) for k in keep if k in p and p[k] is not None]
    for e in p.get("episodes", []) or []:
        sup = f", onset on {e['support']} support" if e.get("support") else ""
        gap = (f", soles {1000 * e['foot_gap_mesh_m']:.1f} mm apart" if e.get("foot_gap_mesh_m") is not None else "")
        rows.append(("episode", f"{e['context']}, {e['manifestation']}, {e['onset_s']:.2f} to {e['offset_s']:.2f} s"
                                f"{sup}{gap}"))
    if "source" in p:
        s = p["source"]
        rows.append(("take", f"{s.get('take_id')} from knot {s.get('start_knot')}"))
        r = s.get("retarget", {})
        if r:
            rows.append(("retarget ratio", f"{r.get('ratio'):.4f}"))
        g = s.get("ground")
        if g:
            a, b = g["residual_before"], g["residual_after"]
            rows.append(("mesh grounding", f"shift {1000 * g['shift_min_m']:+.1f} to {1000 * g['shift_max_m']:+.1f} mm; "
                                           f"lowest sole within {1000 * g['tol_m']:.0f} mm of the floor at "
                                           f"{100 * a['within_tol_share']:.1f} percent of knots before, "
                                           f"{100 * b['within_tol_share']:.1f} after"))
    a = p.get("anchor")
    if a:
        rows.append(("cited normal", f"{a['rate_hz']:.2f} Hz (SD {a['rate_sd_hz']:.2f}, {a['rate_source']}); "
                                     f"amplitude {a['amplitude']:g} {a['amplitude_unit']} ({a['amplitude_source']})"))
    bm = t.get("base_motion")
    if bm:
        gaps = bm.get("finger_gaps_mm", {}).get("left", {})
        rows.append(("base motion", f"raised from {bm['from']} to {bm['posture']} over {bm['raise_s']:g} s, held from "
                                    f"{bm['onset_s']:g} s; left finger gaps "
                                    + ", ".join(f"{k} {v:.1f} mm" for k, v in gaps.items())))
    if p.get("base") == "outstretched" and p.get("base_motion"):
        bm = p["base_motion"]
        rows.append(("base motion", f"raised from {bm['from']} to {bm['posture']} over {bm['raise_s']:g} s, held from "
                                    f"{bm['onset_s']:g} s"))
    tk, bl = p.get("trunk"), p.get("balance")
    if tk and bl:
        rows.append(("trunk bout", f"{tk['amplitude_deg']:.1f} deg of lumbar and thoracic flexion "
                                   f"({'solved for balance, module' if tk['solved'] else 'given'}); lean "
                                   f"{tk['realized_lean_seated_deg']:.1f} deg seated, {tk['realized_lean_peak_deg']:.1f} "
                                   f"deg at its peak, {tk['realized_lean_speed_peak_deg_s']:.0f} deg/s at most"))
        rows.append(("balance", f"{'ok' if bl['ok'] else 'FAILED'}: XcoM (Hof 2005) at least "
                                f"{100 * bl['min_margin_from_liftoff_m']:.1f} cm ahead of the heels from lift-off; at "
                                f"lift-off the centre of mass {100 * (bl['heel_z_m'] - bl['liftoff_com_z_m']):.1f} cm "
                                f"behind the heels at {bl['liftoff_com_vz_m_s']:.2f} m/s forward; standing, "
                                f"{100 * bl['standing_com_ahead_of_ankles_m']:.1f} cm ahead of the ankles"))
    if v_hand := ((t.get("validity") or {}).get("hand_limits")):
        rows.append(("hand joint limits", f"{'ok' if v_hand['ok'] else 'FAILED'} ({v_hand['violations']} joints "
                                          f"outside)"))
    fp = p.get("foot_pose")
    if fp:
        rows.append(("foot pose", f"heel up to {100 * fp['heel_lift_max_m']:.1f} cm, toe up to "
                                  f"{100 * fp['toe_lift_max_m']:.1f} cm, pitch {fp['pitch_change_deg']:+.1f} deg; hip "
                                  f"{fp['hip_change_deg']:.1f}, knee {fp['knee_change_deg']:.1f}, ankle "
                                  f"{fp['ankle_change_deg']:.1f} deg from level 0"))
    if p.get("grading_note"):
        rows.append(("grading note", p["grading_note"]))
    rows.append(("operator version", t.get("version")))
    v = t.get("validity") or {}
    if "limits_new" in v:
        rows.append(("new joint-limit violations", v["limits_new"]))
    if v.get("contact"):
        c = v["contact"]
        rows.append(("contact check", f"{'ok' if c['ok'] else 'FAILED'} (moved foot {1000 * c['moved_low_min_m']:+.1f} to "
                                      f"{1000 * c['moved_low_max_m']:+.1f} mm)"))
    if v.get("continuity"):
        c = v["continuity"]
        rows.append(("continuity check", f"{'ok' if c['ok'] else 'FAILED'} (largest leg step {c['max_step_deg']:.2f} "
                                         f"deg per knot, bound {c['bound_deg_s']:g} deg/s)"))
    f = rec.get("floor")
    if f:
        rows.append(("floor", f"lowest vertex {1000 * f['residual_min_m']:+.1f} to {1000 * f['residual_max_m']:+.1f} mm "
                              f"from the floor, within {1000 * f['tol_m']:.0f} mm in "
                              f"{100 * f['within_tol_share']:.1f} percent of frames"))
    return rows


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def index_html(records, set_name, not_rendered=None):
    """The index page (module docstring); not_rendered: {label: reason} (default exam.NOT_RENDERED)."""
    not_rendered = EX.NOT_RENDERED if not_rendered is None else not_rendered
    by_id = {r["clip_id"]: r for r in records}
    css = ("body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:16px;background:#f4f4f6;color:#1b1b1f}"
           ".mark{background:#961414;color:#fff;padding:6px 10px;font-weight:700;display:inline-block}"
           "section{margin:18px 0}h2{border-bottom:2px solid #ccc;padding-bottom:4px}"
           ".clip{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;margin:10px 0}"
           ".vids{display:flex;flex-wrap:wrap;gap:8px}.vids figure{margin:0}"
           "video{width:400px;max-width:100%;background:#000}figcaption{font-size:12px;color:#555}"
           "table{border-collapse:collapse;font-size:12px;margin-top:6px}td{border:1px solid #e2e2e2;padding:2px 6px}"
           ".l0 figcaption{color:#1d5e1d}.note{font-size:13px;color:#444}")
    out = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>SYNTHETIC exam viewer "
           f"{_e(set_name)}</title><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
           f"<style>{css}</style></head><body>",
           f"<div class=\"mark\">SYNTHETIC</div> <strong>Synthetic patient viewer, exam set {_e(set_name)}</strong>",
           f"<p class=\"note\">{_e(V.NOTICE)}. Every body is the shared neutral SMPL+H shape, the hands at rest except "
           "in the finger and hand tasks (3.4, 3.5) and the folded arms of the arising (3.9), no facial motion. "
           "Grades and levels are proposals of docs/proposals/SYNTH_PATIENT_DRAFT.md, not a validated "
           "calibration. Each clip shows its views; its level 0 (the same base body without the operator) plays "
           "beside it in green. Clips marked 'not for DIS grading' hold a leg posture on the toes (see their grading "
           "note). Clips from an earlier set are linked from its folder (their caption names the set). Videos load on "
           "demand.</p>"]
    families = list(FAMILY_ORDER) + [f for f in dict.fromkeys(r["family"] for r in records) if f not in FAMILY_ORDER]
    for fam in families:
        recs = [r for r in records if r["family"] == fam]
        if not recs:
            continue
        out.append(f"<section><h2>{_e(FAMILY_TITLE.get(fam, fam.replace('_', ' ')))}</h2>")
        for r in recs:
            src = f" | from {_e(r['earlier'])}" if r.get("earlier") else ""
            out.append(f"<div class=\"clip\" id=\"{_e(r['clip_id'])}\"><strong>{_e(r['segment'])} "
                       f"{_e(r['segment_title'])}: {_e(r['operator'])}</strong> | {_e(r['grade_text'])}<br>"
                       f"<span class=\"note\">{_e(r['params_text'])} | clip {_e(r['clip_id'])}{src}</span>"
                       "<div class=\"vids\">")
            for v in r["views"]:
                mp4, png = v["files"]["mp4"]["file"], v["files"]["poster"]["file"]
                out.append(f"<figure><video controls loop muted preload=\"none\" poster=\"{_e(r['folder'])}/"
                           f"{_e(png)}\" src=\"{_e(r['folder'])}/{_e(mp4)}\"></video><figcaption>"
                           f"{_e(v['camera'].get('title', v['view']))}, {v['n_frames']} frames, "
                           f"{v['duration_s']:.1f} s</figcaption></figure>")
            l0 = by_id.get(r.get("level0"))
            if l0:
                v = next((x for x in l0["views"] if x["view"] == r["views"][0]["view"]), l0["views"][0])
                out.append(f"<figure class=\"l0\"><video controls loop muted preload=\"none\" poster=\""
                           f"{_e(l0['folder'])}/{_e(v['files']['poster']['file'])}\" src=\"{_e(l0['folder'])}/"
                           f"{_e(v['files']['mp4']['file'])}\"></video><figcaption>level 0: {_e(l0['clip_id'])}, "
                           f"{_e(v['camera'].get('title', v['view']))}</figcaption></figure>")
            out.append("</div><table>")
            for k, val in _truth_rows(r):
                out.append(f"<tr><td>{_e(k)}</td><td>{_e(_fmt(val))}</td></tr>")
            out.append("</table></div>")
        out.append("</section>")
    out.append("<section><h2>Not rendered in this set</h2><ul>")
    for k, why in not_rendered.items():
        out.append(f"<li><strong>{_e(k)}</strong>: {_e(why)}</li>")
    out.append("</ul></section></body></html>\n")
    return "\n".join(out)


def load_records(out):
    folder = os.path.join(out, "truth")
    recs = []
    if not os.path.isdir(folder):
        return recs
    for f in sorted(os.listdir(folder)):
        if f.startswith("._") or not f.endswith("_truth.json"):
            continue
        with open(os.path.join(folder, f), encoding="utf-8") as fh:
            recs.append(json.load(fh))
    return recs


def check_files(out, recs):
    """{n_mp4, n_png, failures: [...]}: every MP4 and PNG of the records read back (module docstring)."""
    from PIL import Image
    fails, n_mp4, n_png = [], 0, 0
    for r in recs:
        for v in r["views"]:
            mp4 = os.path.join(out, r["folder"], v["files"]["mp4"]["file"])
            png = os.path.join(out, r["folder"], v["files"]["poster"]["file"])
            frames, w, h, rate = V.mp4_frame_count(mp4)
            tags = V.mp4_tags(mp4)
            ok = (frames == v["n_frames"] and (w, h) == (v["width"], v["height"]) and rate == f"{int(v['fps'])}/1"
                  and LB.LABEL in tags.get("title", "") and LB.LABEL in tags.get("comment", "")
                  and V.sha256_file(mp4) == v["files"]["mp4"]["sha256"])
            n_mp4 += 1
            if not ok:
                fails.append(f"{r['clip_id']}/{v['view']}: mp4 {frames} frames {w}x{h} {rate} {tags}")
            ok = (Image.open(png).text.get(LB.LABEL) == "1" and V.sha256_file(png) == v["files"]["poster"]["sha256"])
            n_png += 1
            if not ok:
                fails.append(f"{r['clip_id']}/{v['view']}: png")
    return {"n_mp4": n_mp4, "n_png": n_png, "failures": fails}


def earlier_records(out, earlier, own_ids):
    """The records of earlier sets (sibling folders of out; a name or a list, the first taking precedence) whose
    clips the set out does not hold, their folder made relative to out (../<earlier>/clips/<family>) and marked with
    the set's name; the earlier files are only read."""
    root = os.path.dirname(out.rstrip("/"))
    recs, seen = [], set(own_ids)
    for e in ([earlier] if isinstance(earlier, str) else list(earlier)):
        for r in load_records(os.path.join(root, e)):
            if r["clip_id"] not in seen:
                recs.append(dict(r, folder=f"../{e}/{r['folder']}", earlier=e))
                seen.add(r["clip_id"])
    return recs


def manifest_path(out):
    return os.path.join(out, LB.synthetic_name("manifest", ".json"))


def write_index(out, set_name, argv, specs, record_dir=RECORD_DIR, earlier=None, summary=None):
    if os.path.exists(manifest_path(out)):
        raise SystemExit(f"{manifest_path(out)} exists: the set is closed, its index is never rewritten")
    if record_dir:
        for p in (os.path.join(record_dir, f"{set_name}.json"), os.path.join(record_dir, f"{set_name}.md")):
            if os.path.exists(p):
                raise SystemExit(f"{p} exists: a run record is never rewritten")
    if summary and os.path.exists(os.path.join(ROOT, summary)):
        raise SystemExit(f"{os.path.join(ROOT, summary)} exists: a summary is never rewritten")
    own = load_records(out)
    recs = own + (earlier_records(out, earlier, {r["clip_id"] for r in own}) if earlier else [])
    order = {s.clip_id: i for i, s in enumerate(specs)}
    recs.sort(key=lambda r: order.get(r["clip_id"], 10 ** 6))
    own.sort(key=lambda r: order.get(r["clip_id"], 10 ** 6))
    page = os.path.join(out, INDEX).replace("\\", "/")
    not_rendered = dict(EX.bounded_ladders(), **EX.NOT_RENDERED)
    with open(page, "w", encoding="utf-8", newline="\n") as f:
        f.write(index_html(recs, set_name, not_rendered))
    missing = [s.clip_id for s in specs if s.clip_id not in {r["clip_id"] for r in recs}]
    files = []
    for r in own:
        for v in r["views"]:
            for kind in ("mp4", "poster"):
                x = v["files"][kind]
                files.append({"clip": r["clip_id"], "view": v["view"], "kind": kind, "file": f"{r['folder']}/{x['file']}",
                              "bytes": x["bytes"], "sha256": x["sha256"]})
    total = sum(x["bytes"] for x in files)
    manifest = {"label": LB.LABEL, "notice": V.NOTICE, "set": set_name, "index": INDEX, "clips": len(own),
                "earlier": earlier, "earlier_clips": len(recs) - len(own),
                "missing": missing, "files": files, "bytes": total, "script": script_record(argv),
                "code": code_record(),
                "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(manifest_path(out), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        rj, rm = os.path.join(record_dir, f"{set_name}.json"), os.path.join(record_dir, f"{set_name}.md")
        for p in (rj, rm):
            if os.path.exists(p):
                raise SystemExit(f"{p} exists: a run record is never rewritten")
        rows = [{"clip": r["clip_id"], "family": r["family"], "segment": r["segment"], "grade": r["grade_text"],
                 "params": r["params_text"], "level0": r["level0"],
                 "views": [{"view": v["view"], "frames": v["n_frames"], "seconds": v["duration_s"],
                            "mp4_bytes": v["files"]["mp4"]["bytes"], "mp4_sha256": v["files"]["mp4"]["sha256"],
                            "png_bytes": v["files"]["poster"]["bytes"]} for v in r["views"]],
                 "contact": (r["truth"].get("validity") or {}).get("contact"),
                 "continuity": (r["truth"].get("validity") or {}).get("continuity"),
                 "limits_new": (r["truth"].get("validity") or {}).get("limits_new"),
                 "floor": r.get("floor"), "toe_stance": (r["truth"].get("params") or {}).get("toe_stance"),
                 "caption_cut": any(x.endswith("...") for v in r["views"] for x in v.get("burned_caption", [])),
                 "render_seconds": r["seconds"]} for r in own]
        rec = {"label": LB.LABEL, "set": set_name, "out": out, "index": f"{out}/{INDEX}", "clips": rows,
               "n_clips": len(rows), "n_views": sum(len(r["views"]) for r in rows),
               "n_frames": sum(v["frames"] for r in rows for v in r["views"]), "bytes": total, "missing": missing,
               "earlier": earlier, "earlier_clips": sorted(r["clip_id"] for r in recs if r.get("earlier")),
               "not_rendered": not_rendered, "script": script_record(argv), "code": code_record(),
               "options": own[0]["options"] if own else None, "check": check_files(out, own),
               "clip_code": sorted({(r["code"]["synth_code_sha256"], r["script"]["sha256"]) for r in own}),
               "levels": level_rows(own)}
        with open(rj, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        with open(rm, "w", encoding="utf-8", newline="\n") as f:
            f.write(record_markdown(rec))
        if summary:
            sp = os.path.join(ROOT, summary)
            if os.path.exists(sp):
                raise SystemExit(f"{sp} exists: a summary is never rewritten")
            with open(sp, "w", encoding="utf-8", newline="\n") as f:
                f.write(record_markdown(rec, title=f"Synthetic patient viewer, exam set {set_name}"))
    return page, len(recs), missing, total


def level_rows(recs):
    """Rows of the levels the new exam motions realized, read from the clip truths (the hand tasks: cited normal,
    rate, amplitudes read back from the truth's cycles; the arising: cited and realized phases; the raise: posture
    onset and finger gaps)."""
    rows = []
    for r in recs:
        t = r.get("truth") or {}
        p = t.get("params") or {}
        if t.get("operator") == "bradykinesia" and p.get("unit"):
            amps = [c["amplitude"] for c in p["cycles"]]
            rows.append({"clip": r["clip_id"], "kind": "hand task", "task": p["task"], "item": p["item"],
                         "rate_hz": p["rate_hz"], "level0_hz": p["level0_hz"], "speed_gain": p["speed_gain"],
                         "amplitude0": p["amplitude0"], "unit": p["unit"], "first_amplitude": amps[0],
                         "last_amplitude": amps[-1], "n_pauses": len(p["pauses"]), "anchor": p["anchor"],
                         "closed_tip_distance_mm": p.get("closed_tip_distance_mm"),
                         "pinch_gap_mm": p.get("pinch_gap_mm"),
                         "hand_limits_ok": ((t.get("validity") or {}).get("hand_limits") or {}).get("ok")})
        elif t.get("operator") == "arise":
            rows.append({"clip": r["clip_id"], "kind": "arising", **{k: p[k] for k in p if k not in ("sources",)}})
        elif t.get("base_motion") or p.get("base") == "outstretched":
            bm = t.get("base_motion") or p.get("base_motion")
            rows.append({"clip": r["clip_id"], "kind": "raise", "onset_s": bm["onset_s"], "raise_s": bm["raise_s"],
                         "finger_gaps_mm": bm["finger_gaps_mm"], "cm_pp": p.get("cm_pp"),
                         "frequency_hz": p.get("frequency_hz"), "latency_s": p.get("latency_s"),
                         "intervals_s": p.get("intervals_s")})
    return rows


def _balance_text(x):
    """The trunk bout and balance of an arising row of level_rows (empty before the review of 2026-10-05)."""
    tk, bl = x.get("trunk"), x.get("balance")
    if not (tk and bl):
        return ""
    ref = bl.get("stand_ref_com_ahead_of_ankles_m")
    return (f" (bout {tk['amplitude_deg']:.1f} deg, peak lean {tk['realized_lean_peak_deg']:.1f} deg from the "
            f"vertical, {tk['realized_lean_speed_peak_deg_s']:.0f} deg/s at most); balance "
            f"{'ok' if bl['ok'] else 'FAILED'}: XcoM margin {100 * bl['liftoff_margin_m']:.2f} cm at lift-off, "
            f"{100 * bl['min_margin_from_liftoff_m']:.2f} cm at least after it; centre of mass at lift-off "
            f"{100 * (bl['heel_z_m'] - bl['liftoff_com_z_m']):.1f} cm behind the heels at "
            f"{bl['liftoff_com_vz_m_s']:.2f} m/s; standing {100 * bl['standing_com_ahead_of_ankles_m']:.1f} cm ahead of "
            f"the ankles" + ("" if ref is None else f" ({bl['stand_ref']}: {100 * ref:.1f})")
            + f"; shank tilt {x['seated_shank_tilt_deg']:.1f} deg seated, {x['standing_shank_tilt_deg']:.1f} standing")


def record_markdown(rec, title=None):
    """The clip table of a run record (written by this script; numbers from the record)."""
    lines = [f"# {title or 'Exam set ' + rec['set'] + ': rendered clips'}", "",
             f"Written by scripts/render_mesh_video.py --index-only (script SHA-256 {rec['script']['sha256']}); "
             f"mdfm/synth code SHA-256 {rec['code']['synth_code_sha256']}, operator version "
             f"{rec['code']['operator_version']}, MoveAtlas HEAD {rec['code']['moveatlas_head']}.", "",
             f"Index page: {rec['index']}", "",
             f"{rec['n_clips']} clips, {rec['n_views']} views, {rec['n_frames']} frames, {rec['bytes']} bytes of MP4 "
             f"and PNG files.", "",
             f"Check of the written files: {rec['check']['n_mp4']} MP4 and {rec['check']['n_png']} PNG read back, "
             f"{len(rec['check']['failures'])} failures.", "",
             "Code that rendered the clips (mdfm/synth SHA-256, script SHA-256): "
             + "; ".join(f"{a}, {b}" for a, b in rec["clip_code"]) + ".", "",
             f"Captions cut to the frame width: {sum(1 for r in rec['clips'] if r.get('caption_cut'))} clips. "
             f"Clips held on the toes (toe stance, not for DIS grading): "
             f"{sum(1 for r in rec['clips'] if r.get('toe_stance'))}.", "",
             "Floor: each clip's floor is the median over its frames of the lowest mesh vertex (viewer.scene_floor); "
             "the column gives the lowest vertex's range against it (mm) and the share of frames within "
             f"{1000 * V.FLOOR_TOL_M:g} mm.", "",
             "| clip | segment | grade | parameters | views (frames, MP4 bytes) | contact | continuity | floor |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rec["clips"]:
        views = "; ".join(f"{v['view']} ({v['frames']}, {v['mp4_bytes']})" for v in r["views"])
        c = r["contact"]
        cc = "" if not c else ("ok" if c["ok"] else "FAILED") + f" ({1000 * c['moved_low_min_m']:+.1f} to " \
                                                                  f"{1000 * c['moved_low_max_m']:+.1f} mm)"
        s = r.get("continuity")
        sc = "" if not s else ("ok" if s["ok"] else "FAILED") + f" ({s['max_step_deg']:.2f} deg/knot)"
        f = r.get("floor")
        fc = "" if not f else (f"{1000 * f['residual_min_m']:+.1f} to {1000 * f['residual_max_m']:+.1f}, "
                               f"{100 * f['within_tol_share']:.1f} percent")
        lines.append(f"| {r['clip']} | {r['segment']} | {r['grade']} | {r['params']} | {views} | {cc} | {sc} | {fc} |")
    if rec.get("earlier_clips"):
        ear = rec["earlier"] if isinstance(rec["earlier"], str) else " and ".join(rec["earlier"])
        lines += ["", f"The index page also links {len(rec['earlier_clips'])} clips of {ear} (not rendered "
                      f"again, their files unchanged)."]
    lv = rec.get("levels") or []
    if lv:
        lines += ["", "Levels of the new exam motions, read from the clip truths:", "",
                  "| clip | what | rate (Hz) | level-0 rate (Hz) | amplitude (first, last cycle) | pauses | other |",
                  "|---|---|---|---|---|---|---|"]
        for x in lv:
            if x["kind"] == "hand task":
                oth = (f"closed tip distance {x['closed_tip_distance_mm']:.1f} mm, pinch gap "
                       f"{x['pinch_gap_mm']:.1f} mm; " if x.get("closed_tip_distance_mm") is not None else "")
                oth += f"hand joint limits {'ok' if x['hand_limits_ok'] else 'FAILED'}"
                lines.append(f"| {x['clip']} | {x['task']} ({x['item']}) | {x['rate_hz']:.3f} | {x['level0_hz']:.3f} | "
                             f"{x['first_amplitude']:.2f}, {x['last_amplitude']:.2f} {x['unit']} | {x['n_pauses']} | "
                             f"{oth} |")
            elif x["kind"] == "arising":
                lines.append(f"| {x['clip']} | arising (3.9) | | | | | seat {x['realized_seat_height_m']:.4f} m (cited "
                             f"{x['seat_height_m']:g}), rise {x['rise_s']:g} s, hip flexion "
                             f"{x['realized_hip_flexion_displacement_deg']:.1f} deg (cited {x['hip_flexion_deg']:g}) "
                             f"peak at {100 * x['realized_hip_peak_share']:.1f} percent (cited "
                             f"{100 * x['hip_peak_share']:.1f}), ankle dorsiflexion "
                             f"{x['realized_ankle_dorsiflexion_deg']:.2f} deg (cited {x['ankle_dorsiflexion_deg']:g}) "
                             f"peak at {100 * x['realized_ankle_peak_share']:.1f} percent (cited "
                             f"{100 * x['ankle_peak_share']:.1f}), plantarflexion "
                             f"{x['realized_ankle_plantarflexion_deg']:.2f} deg (cited "
                             f"{x['ankle_plantarflexion_deg']:g}), knee extension "
                             f"{x['realized_knee_extension_deg']:.1f} deg, trunk lean "
                             f"{x['realized_trunk_lean_peak_deg']:.1f} deg" + _balance_text(x) + " |")
            else:
                g = x["finger_gaps_mm"]["left"]
                lines.append(f"| {x['clip']} | raise to the 3.15 posture | {x['frequency_hz'] or ''} | | "
                             f"{'' if x['cm_pp'] is None else format(x['cm_pp'], '.2f') + ' cm pp'} | | onset "
                             f"{x['onset_s']:g} s; left finger gaps "
                             + ", ".join(f"{k} {v:.1f} mm" for k, v in g.items()) + " |")
    if rec["missing"]:
        lines += ["", f"Missing (not yet rendered): {', '.join(rec['missing'])}"]
    lines += ["", "Not rendered in this set:"] + [f"- {k}: {v}" for k, v in rec["not_rendered"].items()]
    return "\n".join(lines) + "\n"


def write_stand_selection(argv, record_dir=RECORD_DIR):
    """--select-stand: run smplh_reference.select_stand (behind the reference flag) on the train body store and write
    its record (reports/synth/viewer/stand_selection.json, never rewritten); says whether exam.REF_STAND is its top
    window (never since exam_v10, whose stand is procedural)."""
    from mdfm.synth.openbody import require_reference
    require_reference("render_mesh_video.py --select-stand")
    p = os.path.join(record_dir, "stand_selection.json")
    if os.path.exists(p):
        raise SystemExit(f"{p} exists: a run record is never rewritten")
    from mdfm.data.body_store import BodyStore
    from mdfm.synth import smplh_reference as SR
    store = BodyStore(split="train")
    res = SR.select_stand(store)
    top = res["ranked"][0] if res["ranked"] else None
    agrees = bool(top and top["take_id"] == EX.REF_STAND.get("take_id")
                  and top["start_knot"] == EX.REF_STAND.get("start_knot"))
    rec = dict(res, label=LB.LABEL, ref_stand=EX.REF_STAND, ref_stand_is_top=agrees, store_split=store.split,
               script=script_record(argv), code=code_record(),
               created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    os.makedirs(record_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    return p, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", default="exam_v2", help="name of the exam set folder under the output root")
    ap.add_argument("--root", default=V.VIEWER_ROOT, help=f"output root, inside {V.VIEWER_ROOT}")
    ap.add_argument("--only", nargs="*", default=None, help="render only the clips whose id holds one of these")
    ap.add_argument("--shard", default="0/1", help="i/n: render every n-th clip from the i-th")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    ap.add_argument("--index-only", action="store_true", help="write the index page and the run record")
    ap.add_argument("--select-stand", action="store_true",
                    help="run the quiet-stand selection rule (smplh_reference.select_stand, behind "
                         "MDFM_SMPLH_REFERENCE=1) and write its record")
    ap.add_argument("--no-record", action="store_true", help="with --index-only: no record in the tree")
    ap.add_argument("--since", default=None,
                    help="keep only the clips whose spec was added in this set (default: the --set name when it is an "
                         "exam version)")
    ap.add_argument("--with-earlier", nargs="*", default=None,
                    help="with --index-only: also link the clips of these earlier set folders on the index page (the "
                         "first taking precedence)")
    ap.add_argument("--summary", default=None,
                    help="with --index-only: also write the run record's table to this new file of the tree")
    ap.add_argument("--catalogue", action="store_true",
                    help="plan the symptom catalogue (catalogue.catalogue_set) instead of the exam set")
    ap.add_argument("--png-frames", action="store_true", help="also write every frame as PNG")
    ap.add_argument("--shading", default="gouraud", choices=["gouraud", "flat"])
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--width", type=int, default=V.SIZE[0])
    ap.add_argument("--height", type=int, default=V.SIZE[1])
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else list(argv)
    out = f"{a.root.rstrip('/')}/{a.set}"
    V.check_out(out, V.VIEWER_ROOT)
    if a.select_stand:
        p, rec = write_stand_selection(argv)
        print(f"{p}: top {rec['ranked'][0] if rec['ranked'] else None}; REF_STAND is the top window: "
              f"{rec['ref_stand_is_top']}")
        return
    if a.catalogue:
        from mdfm.synth.catalogue import catalogue_set
        all_specs = catalogue_set()
    else:
        all_specs = EX.exam_set()
    versions = sorted({s.since for s in all_specs})
    since = a.since or a.set
    if since not in versions:
        raise SystemExit(f"--set {a.set} is not an exam version ({', '.join(versions)}): give --since")
    specs = [s for s in all_specs if s.since == since]
    if a.only:
        specs_sel = [s for s in specs if any(k in s.clip_id for k in a.only)]
    else:
        specs_sel = specs
    i, n = (int(x) for x in a.shard.split("/"))
    specs_sel = specs_sel[i::n]
    if a.list:
        for s in specs_sel:
            print(f"{s.clip_id:55s} {s.family:12s} {s.segment} {s.grade_text:45s} views {','.join(s.views)}")
        print(f"{len(specs_sel)} clips")
        return
    if a.index_only:
        full = all_specs if a.with_earlier else specs
        page, nrec, missing, total = write_index(out, a.set, argv, full, None if a.no_record else RECORD_DIR,
                                                 earlier=a.with_earlier, summary=a.summary)
        print(f"index: {page} ({nrec} clips, {total} bytes); missing: {missing}")
        return
    if os.path.exists(manifest_path(out)):
        raise SystemExit(f"{manifest_path(out)} exists: the set is closed, no clip is added to it")
    opts = V.RenderOptions(width=a.width, height=a.height, shading=a.shading, supersample=a.supersample)
    t0 = time.time()
    for s in specs_sel:
        if os.path.exists(truth_path(out, s.clip_id)):
            print(f"{s.clip_id}: done before, skipped")
            continue
        r = render_clip(s, out, opts, argv, png_frames=a.png_frames)
        print(f"{s.clip_id}: {len(r['views'])} views in {r['seconds']} s (total {time.time() - t0:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
