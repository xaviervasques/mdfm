"""The synthetic patient's bodies as driver pose videos (task VCH): an adapter from the trajectories of mdfm/synth to
mdfm.video.render.BodyClip, and the first exam driver set, exam_v1.

Adapter (body_clip). A synth body (mdfm.synth.body.Body: rot6d (T, 22, 6) with the root orientation in row 0, the
SMPL translation, contacts, betas, the SkeletonBank gender code, the 100 Hz clock; a mdfm.synth.hands.HandBody also
carries hand6d (T, 30, 6)) is the BodyClip of the driver renderer up to names: its knots [k0, k0 + K] become the
clip's knots bit for bit, knot k0 at t = 0 of the clip clock, the gender code becomes its name, the hand knots go to
BodyClip.hand6d (the opt-in hand path of mdfm/video/points.py; None for a body without hands). A frame k of the clip
at fps shows the body at k0 / rate + k / fps of the body clock: the interpolation reads knots i - 1 to i + 2
(Catmull-Rom, one-sided at the ends), so a window equals the whole body at every frame outside its first and last
knot interval, and a frame on a knot is that knot. The source record carries the SYNTHETIC label, the window, the
exam spec, the operator's truth (mdfm.synth.common.truth_json: its arrays left out) and the truth's event times
restated on the clip clock (clip_events: tremor on intervals, freezing episodes, the hand task's cycles, pauses and
span; a time outside [0, the window's span] lies outside the clip). The operator's arrays, per knot of the body clock,
go whole into the clip's truth NPZ as op_<name> beside op_window_start_knot (operator_arrays), so that nothing of the
truth is lost.

Exam set exam_v1 (EXAM_V1, from the specs of mdfm.synth.exam.exam_set: docs/proposals/SYNTH_PATIENT_DRAFT.md
sections 2 to 7, 13). One row per synth clip: the frame rates, the framing rule and the view, written under
EXAM_ROOT. Section 13 of the draft (one evaluation for real and synthetic): a clip that cannot carry its item at 30
fps is shown at 60 fps. Here the generator's latent clock decides what 30 fps cannot carry. The latent clock
(temporal stride 4) runs at 7.5 latents per second at 30 fps (reports/video_driver/summary.md) and 15 at 60 fps:
Nyquist frequencies of 3.75 and 7.5 Hz. The item frequencies are 5 Hz (rest tremor), 6 Hz (postural tremor), 5.5 Hz
(the freeze's trembling) and 3.67 Hz (level-0 finger tapping): all under 7.5 Hz; at 30 fps the first three lie above
3.75 Hz and the tapping 0.08 Hz under it (2.04 latents per cycle). DR-5.1 (docs/rules/DESCRIPTOR_RULES_DRAFT.md,
f_max = k x fs_native, k open in OPEN-D02) does not decide the rate: it bounds the video frame rate, not the latent
rate, and for any k above 6 / 30 = 0.2 it carries all four frequencies at 30 fps (the draft's section 3 reads it so).
So the tremor and finger-tapping clips are 60 fps, the walk and the freeze both 30 fps (the stage 1 rate of H1 and
H2) and 60 fps under one camera. Every clip is the synth body's whole length, so the durations are those of the exam
segments (E1 45 s, E2 30 s; the walk and the freeze as the exam set builds them).

Views: one camera per clip, a departure from the draft. Its sections 2 to 4 render every segment from a fixed set
(front, left side, right side, back, and a close-up per item: the front and a close-up of both forearms for E1 to E3,
the side for elbow-flexion tremor in E2, the side, front and back for E8), and section 13 asks for the same views as
for a real patient. exam_v1 is not that set; its views, measured against the other render views by the view probe
(scripts/render_driver_video.py --probe v1), are: the walk and the freeze at the full-body lateral framing of V1 (the
side view carries the knee and hip flexion, and the trembling of the knees); rest tremor at the tremor framing of
section 3.2 of the video draft ('torso') from the left side, where the elbow-flexion tremor of the forearm on the
chair arm moves the wrist up and down and forward and back; postural tremor at the torso framing from the front, not
the side the draft names for E2: the row was chosen when synth's E2 tremor turned the forearm about the vertical
elbow hinge of the arms-forward posture (mdfm/synth/base.py, sit_arms_forward), the wrist moving left and right along
the optical axis of a side camera; since exam_v5 (shared change SH4, mdfm/synth/exam.py) the 'postural' driver of
mdfm/synth/tremor.py flexes and extends the wrist, the shoulder joining above a chord (tremors.POSTURAL_SHOULDER), so
the hand moves up and down, in the image plane of the front camera as of a side one, and the row keeps its view;
finger tapping at
the torso framing from the front and the right (render.VIEWS oblique_right60, the azimuth of the synth viewer's
hand_left_medial, where the pinch opens). Not rendered: the other views of the set, every close-up (render has no
close-up framing rule) and E8 from the front and the back.

Read-backs (readback): the item's signal measured on the rendered truth, to check that the adapter kept the truth:
the tremor at the drawn point where the operator reads its level (reading_point): the fingertip landmark of the tip
where the 'rest' driver of E1 is levelled since exam_v7 (the fingertip that moves most, tremor.FINGERTIPS; the wrist
for the elbow driver of E1 until then, decision D4 of mdfm/synth), the left middle fingertip (LHandMiddle4, the
landmark vertex of points.py that is also synth's middle fingertip vertex, hands.tip_vertices) for the
'postural' driver of E2, whose level mdfm/synth reads at the middle fingertip (tremor.point_readings) since exam_v5:
there the wrist joint moves only by the shoulder's share of the drive, nothing below the chord where the shoulder
joins, so a wrist read-back missed the sign. At that point, relative to the shoulder of its side: the dominant
frequency and the image amplitude along the driven direction (the arithmetic of mdfm.video.analysis.driven_direction
and tremor_measure, which read the wrist), and the peak-to-peak 3D displacement against the operator's (point_pp_mm_3d;
the wrist's own, wrist_pp_mm_3d, is kept for every tremor); the drawn thumb-index fingertip distance per operator
cycle against the operator's amplitudes; the freezing episodes in frames. The walk has none here: its measures (H1,
H2) belong to the stage-1 round trip (analysis.py).

Use: the generator's input (scripts/video_generate.py) for the video round trip of synthetic patients. The synthetic
pathology is a test and stimulus instrument only (mdfm/synth/__init__.py).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import analysis as A
from . import contract as C
from . import labels as LB
from . import points as P
from .render import GENDERS, BodyClip

EXAM_ROOT = LB.VIDEO_ROOT + "/driver/exam_v1"
EXAM_SET = "exam_v1"
FORMAT = "mdfm-video-exam-driver/1"


# ---------------------------------------------------------------- the adapter
def _shift(t, start_s):
    return None if t is None else float(t) - float(start_s)


def clip_events(truth, start_s=0.0):
    """The event times of an operator's truth restated on the clip clock (body time minus start_s): 'tremor_on'
    (the on intervals of tremor), 'episodes' (freezing: onset, offset, hold, context, manifestation, trembling
    frequency), 'cycles', 'pauses' and 'task_s' (the hand tasks and the repeated tasks of bradykinesia). Durations
    and periods are unchanged. An empty dict for a truth without events."""
    if not truth:
        return {}
    p = truth.get("params") or {}
    op = truth.get("operator")
    out = {}
    if op == "tremor" and p.get("intervals_s"):
        out["tremor_on"] = [[_shift(a, start_s), _shift(b, start_s)] for a, b in p["intervals_s"]]
    if op == "freezing" and p.get("episodes"):
        out["episodes"] = [{"context": e.get("context"), "manifestation": e.get("manifestation"),
                            "onset_s": _shift(e["onset_s"], start_s), "offset_s": _shift(e["offset_s"], start_s),
                            "duration_s": e.get("duration_s"),
                            "hold_s": [_shift(x, start_s) for x in e.get("hold_s") or []],
                            "tremble_hz": e.get("tremble_hz")} for e in p["episodes"]]
    if op == "bradykinesia" and p.get("cycles"):
        out["cycles"] = [{"cycle": c["cycle"], "start_s": _shift(c["start_s"], start_s),
                          "peak_s": _shift(c.get("peak_s"), start_s), "period_s": c.get("period_s"),
                          "amplitude": c.get("amplitude")} for c in p["cycles"]]
        out["pauses"] = [{"kind": q.get("kind"), "after_cycle": q.get("after_cycle"),
                          "start_s": _shift(q["start_s"], start_s), "duration_s": q.get("duration_s")}
                         for q in p.get("pauses") or []]
        out["task_s"] = [_shift(p.get("start_s"), start_s), _shift(p.get("end_s"), start_s)]
    return out


def _json_truth(truth):
    from ..synth.common import truth_json
    return None if truth is None else truth_json(truth)


def body_clip(body, truth=None, start_s=0.0, duration_s=None, spec=None):
    """BodyClip of the knots [k0, k0 + round(duration_s rate)] of a synth body (module docstring; k0 = start_s rate
    must be a knot; duration_s None: to the body's last knot), with the operator's truth record in its source.
    ValueError for a window off the knots or outside the body."""
    rate = float(body.rate)
    k = float(start_s) * rate
    if abs(k - round(k)) > 1e-6:
        raise ValueError(f"a window starts on a knot: {start_s} s is {k} knots")
    k0 = int(round(k))
    K = body.T - 1 - k0 if duration_s is None else int(round(float(duration_s) * rate))
    if k0 < 0 or K < 1 or k0 + K > body.T - 1:
        raise ValueError(f"window of {K} knot intervals from knot {k0} outside a body of {body.T} knots")
    sl = slice(k0, k0 + K + 1)
    hand6d = getattr(body, "hand6d", None)
    source = {"kind": "synthetic_patient", "label": LB.LABEL,
              "body": "mdfm.synth trajectory (a procedural base or an AMASS train take retargeted to the neutral "
                      "shape, with a pathology operator)",
              "window": {"start_knot": k0, "start_s": k0 / rate, "n_knots": K + 1, "span_s": K / rate,
                         "body_knots": int(body.T), "rate_hz": rate},
              "hands": "posed (HandBody)" if hand6d is not None else "at rest",
              "spec": spec or {}, "operator_truth": _json_truth(truth),
              "events_clip_clock": clip_events(truth, k0 / rate),
              "licence": "SMPL+H arrays: MPI licence, non-commercial research; an AMASS base adds its archive's "
                         "EULA (operator_truth or spec); clips stay internal"}
    return BodyClip(body.rot6d[sl].to(torch.float64).clone(), body.transl[sl].to(torch.float64).clone(),
                    body.betas.to(torch.float64).clone(), GENDERS[int(body.gender)], rate,
                    body.contact[sl].numpy().astype(bool), source,
                    None if hand6d is None else hand6d[sl].to(torch.float64).clone())


def operator_arrays(truth, start_knot=0):
    """{op_<name>: array} of the operator truth's arrays (per knot of the body clock, whole) and
    op_window_start_knot, for the clip's truth NPZ; object arrays become strings (no pickle in the NPZ)."""
    out = {"op_window_start_knot": np.int64(start_knot)}
    for name, arr in ((truth or {}).get("arrays") or {}).items():
        a = np.asarray(arr)
        out[f"op_{name}"] = a.astype(str) if a.dtype == object else a
    return out


# ---------------------------------------------------------------- the exam set
@dataclass(frozen=True)
class ExamRow:
    """One clip of the exam driver set: the synth exam spec id, the frame rates, the framing rule and view of
    mdfm.video.render, and why."""
    spec_id: str
    fps: tuple
    framing: str
    view: str
    why: str


_FT = "finger tapping at the torso framing from the front and the right, where the pinch opens; 60 fps"
EXAM_V1 = (
    ExamRow("ref_walk_E8", (30.0, 60.0), "full_body", "lateral",
            "healthy walk (level 0 of 3.11); 30 fps for H1 and H2, 60 fps beside the freeze's; one camera for both"),
    ExamRow("freezing_E8_g2_trembling", (30.0, 60.0), "full_body", "lateral",
            "freezing of gait, 3.11 grade 2, trembling at 5.5 Hz: 60 fps carries the trembling, 30 fps the episodes"),
    ExamRow("tremor_E1_rest_g0", (60.0,), "torso", "lateral", "rest tremor level 0 (E1 seated rest, 45 s)"),
    ExamRow("tremor_E1_rest_g1b", (60.0,), "torso", "lateral", "rest tremor 3.17 grade 1 below the estimator floor"),
    ExamRow("tremor_E1_rest_g1a", (60.0,), "torso", "lateral", "rest tremor 3.17 grade 1 above the estimator floor"),
    ExamRow("tremor_E1_rest_g2", (60.0,), "torso", "lateral", "rest tremor 3.17 grade 2"),
    ExamRow("tremor_E1_rest_g3", (60.0,), "torso", "lateral", "rest tremor 3.17 grade 3"),
    ExamRow("tremor_E2_postural_g0", (60.0,), "torso", "frontal", "postural tremor level 0 (E2 arms out, 30 s)"),
    ExamRow("tremor_E2_postural_g2", (60.0,), "torso", "frontal", "postural tremor 3.15 grade 2, 6 Hz"),
    ExamRow("brady_E4a_ft_level0", (60.0,), "torso", "oblique_right60", "level 0 of 3.4; " + _FT),
    ExamRow("brady_E4a_ft_decrement_g1", (60.0,), "torso", "oblique_right60", "3.4 grade 1 (decrement); " + _FT),
    ExamRow("brady_E4a_ft_decrement_g2", (60.0,), "torso", "oblique_right60", "3.4 grade 2 (decrement); " + _FT),
    ExamRow("brady_E4a_ft_decrement_g3", (60.0,), "torso", "oblique_right60", "3.4 grade 3 (decrement); " + _FT),
)
SETS = {EXAM_SET: EXAM_V1}


def clip_id(row):
    """The driver clip id of an exam row: exam_v1_<spec id> (render.clip_stem adds SYNTHETIC_driver_ and the
    rate)."""
    return f"{EXAM_SET}_{row.spec_id}"


def exam_specs(rows=EXAM_V1, store=None):
    """{spec id: ClipSpec} of mdfm.synth.exam.exam_set for the rows (KeyError naming a missing id); store: a train
    BodyStore for the AMASS references (default opened lazily by exam_set)."""
    from ..synth.exam import exam_set
    specs = {s.clip_id: s for s in exam_set(store)}
    missing = [r.spec_id for r in rows if r.spec_id not in specs]
    if missing:
        raise KeyError(f"not in mdfm.synth.exam.exam_set: {missing}")
    return {r.spec_id: specs[r.spec_id] for r in rows}


def spec_record(spec):
    """The exam spec's description for the source record (no builder)."""
    return {"clip_id": spec.clip_id, "family": spec.family, "segment": spec.segment, "item": spec.item,
            "operator": spec.operator, "grade": spec.grade, "grade_text": spec.grade_text,
            "params_text": spec.params_text, "level0": spec.level0, "scene": spec.scene,
            "views_viewer": list(spec.views),
            "exam_set": "mdfm.synth.exam.exam_set"}


# ---------------------------------------------------------------- read-backs
POINT = {n: i for i, n in enumerate(P.NAMES)}   # the drawn set's index of each point name (points.NAMES)
READING_POINTS = {"wrist": "Wrist", "middle_tip": "HandMiddle4"}   # mdfm/synth's reading points, drawn names
# the 'rest' driver's reading point (tremor.FINGERTIPS, since exam_v7): the fingertip its truth names (level_tip, a
# finger of hands.FINGERS), drawn as that finger's tip landmark of points.py
TIP_POINTS = {"thumb": "HandThumb4", "index": "HandIndex4", "middle": "HandMiddle4", "ring": "HandRing4",
              "pinky": "HandLittle4"}
READING_TEXT = {"Wrist": "wrist", "HandMiddle4": "middle fingertip", "HandThumb4": "thumb tip",
                "HandIndex4": "index fingertip", "HandRing4": "ring fingertip", "HandLittle4": "little fingertip"}


def reading_point(op_truth, side="L"):
    """The drawn point (a name of points.NAMES) where the tremor of an operator truth is read (module docstring):
    the side's fingertip landmark of the tip the truth levels at for a truth read at 'fingertips' (since exam_v7,
    the 'rest' driver of the E1 rest tremor: the fingertip that moves most), the side's middle fingertip landmark
    for the 'postural' driver or a truth whose level is read at 'middle_tip' (since exam_v5, the E2 postural
    tremor), else the side's wrist (the elbow driver, decision D4 until exam_v7)."""
    p = (op_truth or {}).get("params") or {}
    if p.get("point") == "fingertips":
        return side + TIP_POINTS[p["level_tip"]]
    key = "middle_tip" if p.get("point") == "middle_tip" or p.get("driver") == "postural" else "wrist"
    return side + READING_POINTS[key]


def reading_text(point):
    """'wrist', 'middle fingertip' or the fingertip of a reading point's name (reading_point), for the clip
    table."""
    return READING_TEXT[point[1:]]


def point_motion_mm(points_world, fps, band, point="LWrist", ref="LShoulder"):
    """The 3D motion of a drawn point relative to another (default the left wrist and shoulder) along its first
    principal axis: {pp_mm: twice the amplitude of analysis.spectral_peak in band (the operator's peak-to-peak chord
    for a constant tremor), freq_hz: its frequency, range_mm: the largest minus the smallest displacement over the
    frames (sampled, so up to a few percent short of pp_mm at 12 frames per cycle)}."""
    w = points_world[:, POINT[point]] - points_world[:, POINT[ref]]
    d = w - w.mean(0)
    if not np.any(np.abs(d) > 1e-12):
        return {"pp_mm": 0.0, "freq_hz": None, "range_mm": 0.0}
    v = np.linalg.svd(d, full_matrices=False)[2][0]
    x = d @ v
    f, a = A.spectral_peak(x, fps, band)
    return {"pp_mm": float(2000.0 * a), "freq_hz": f, "range_mm": float(1000.0 * (x.max() - x.min()))}


def wrist_motion_mm(points_world, fps, band, side="L"):
    """point_motion_mm of the side's wrist relative to its shoulder."""
    return point_motion_mm(points_world, fps, band, f"{side}Wrist", f"{side}Shoulder")


def image_tremor(uv, fps, band, point="LWrist", ref="LShoulder", torso=None):
    """{freq_hz, amp_px, amp_torso, direction}: the image tremor of a drawn point relative to another, the arithmetic
    of analysis.driven_direction (the first principal axis of the linearly detrended difference, its x component made
    non-negative) and analysis.tremor_measure (spectral_peak along it, over torso_px), which read the wrist minus the
    shoulder; for the wrist the same numbers as theirs."""
    uv = np.asarray(uv, dtype=np.float64)
    d = uv[:, POINT[point]] - uv[:, POINT[ref]]
    tt = np.arange(len(d), dtype=np.float64)
    dd = d - np.stack([np.polyval(np.polyfit(tt, d[:, c], 1), tt) for c in range(2)], 1)
    v = np.linalg.svd(dd, full_matrices=False)[2][0]
    v = -v if v[0] < 0 or (v[0] == 0 and v[1] < 0) else v
    f, a = A.spectral_peak(d @ v, fps, band)
    tor = A.torso_px(uv) if torso is None else float(torso)
    return {"freq_hz": f, "amp_px": a, "amp_torso": a / tor, "direction": v}


def readback(truth, fps, op_truth=None, events=None):
    """The item's signal on a rendered driver truth (module docstring) -> dict of plain numbers."""
    uv, pw = np.asarray(truth["uv"]), np.asarray(truth["points_world"])
    n = uv.shape[0]
    out = {"torso_px": float(truth["torso_px"]), "n_frames": int(n)}
    op = (op_truth or {}).get("operator")
    p = (op_truth or {}).get("params") or {}
    ev = events or {}
    if op == "tremor":
        band = A.BANDS["rest"] if p.get("frequency_hz", 5.0) <= A.BANDS["rest"][1] else A.BANDS["action"]
        pt = reading_point(op_truth)
        ref = pt[0] + "Shoulder"
        m = image_tremor(uv, fps, band, pt, ref)
        p3 = point_motion_mm(pw, fps, band, pt, ref)
        w3 = wrist_motion_mm(pw, fps, band)
        out.update(tremor={"operator_hz": p.get("frequency_hz"), "operator_cm_pp": p.get("cm_pp"), "point": pt,
                           "freq_hz": m["freq_hz"], "amp_px": m["amp_px"], "amp_torso": m["amp_torso"],
                           "point_pp_mm_3d": p3["pp_mm"], "point_freq_hz_3d": p3["freq_hz"],
                           "point_range_mm_3d": p3["range_mm"],
                           "wrist_pp_mm_3d": w3["pp_mm"], "wrist_freq_hz_3d": w3["freq_hz"],
                           "wrist_range_mm_3d": w3["range_mm"], "band_hz": list(band)})
    elif op == "bradykinesia" and ev.get("cycles"):
        t = np.asarray(truth["t_s"])
        lt, li = pw[:, P.LHAND.start + P.HAND21.index("Thumb4")], pw[:, P.LHAND.start + P.HAND21.index("Index4")]
        d3 = 1000.0 * np.linalg.norm(lt - li, axis=-1)
        ut, ui = uv[:, P.LHAND.start + P.HAND21.index("Thumb4")], uv[:, P.LHAND.start + P.HAND21.index("Index4")]
        dpx = np.linalg.norm(ut - ui, axis=-1)
        cyc = []
        for c in ev["cycles"]:
            sel = (t >= c["start_s"]) & (t <= c["start_s"] + c["period_s"])
            if not sel.any():
                continue
            k = int(np.flatnonzero(sel)[np.argmax(d3[sel])])
            cyc.append({"cycle": c["cycle"], "operator_amplitude_mm": c["amplitude"], "drawn_max_mm": float(d3[k]),
                        "drawn_max_px": float(dpx[sel].max()), "peak_s": c["peak_s"], "drawn_peak_s": float(t[k]),
                        "frames": int(sel.sum())})
        out["tapping"] = {"cycles": cyc, "drawn_min_mm": float(d3.min()), "operator_rate_hz": p.get("rate_hz"),
                          "frames_per_cycle": float(fps * p["period_s"]) if p.get("period_s") else None}
    elif op == "freezing":
        out["episodes"] = [dict(e, onset_frame=int(round(e["onset_s"] * fps)),
                                offset_frame=int(round(e["offset_s"] * fps))) for e in ev.get("episodes", [])]
    return out


def segments_of(n_frames):
    """The generator's segment plan of a clip (contract.segment_plan): segments and padding."""
    plan = C.segment_plan(n_frames)
    return {"n_segments": plan["n_segments"], "pad": plan["pad"]}


def summary_markdown(record, record_path):
    """The clip table of an exam driver run record (reports/video_driver/exam_v1.md), every number read from it."""
    lines = [f"# Exam driver set {record['set']}: the clip list", "",
             f"Written by scripts/render_driver_video.py --exam {record['set'].replace('exam_', '')} from its run "
             f"record {record_path} "
             f"(script SHA-256 {record['script']['sha256'][:12]}..., mdfm/video SHA-256 "
             f"{record['code']['video_code_sha256'][:12]}..., mdfm/synth SHA-256 "
             f"{record['code']['synth_code_sha256'][:12]}"
             f"..., MD_FM commit {str(record['code']['mdfm_git_head'])[:7]} plus the working tree, MoveAtlas commit "
             f"{str(record['code']['moveatlas_head'])[:7]}). Every number below is read from that record.", "",
             f"Clips: {len(record['clips'])}, under {record['out_root']} (PNG frames, "
             f"{'lossless MP4, ' if record['totals']['mp4_equal_all'] is not None else ''}truth NPZ, provenance JSON "
             f"each); {record['width']} x {record['height']}; SYNTHETIC in every name and metadata field; "
             f"{record['seconds']} s of wall time on the CPU (no GPU).", "",
             "| clip | item | fps | frames | s | segments (pad) | framing, view | torso px | read-back |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in record["clips"]:
        rb = c["readback"]
        if "tremor" in rb:                # a record without the point keys (written before them): the wrist
            t = rb["tremor"]
            where = reading_text(t.get("point", "LWrist"))
            pp, rg = t.get("point_pp_mm_3d", t["wrist_pp_mm_3d"]), t.get("point_range_mm_3d", t["wrist_range_mm_3d"])
        if "tremor" in rb and not rb["tremor"]["operator_cm_pp"]:
            r = (f"level 0 (no tremor): {where} {rg:.3f} mm range in 3D, {t['amp_px']:.3f} px in the image band")
        elif "tremor" in rb:
            r = (f"{t['freq_hz']:.2f} Hz (operator {t['operator_hz']:g}), {t['amp_px']:.2f} px peak at the {where}, "
                 f"{where} {pp:.2f} mm pp in 3D (operator {10 * (t['operator_cm_pp'] or 0):.2f})")
        elif "tapping" in rb:
            cy = rb["tapping"]["cycles"]
            r = (f"tip distance cycle 1 {cy[0]['drawn_max_mm']:.1f} mm, {cy[0]['drawn_max_px']:.1f} px (operator "
                 f"{cy[0]['operator_amplitude_mm']:.1f}); cycle {cy[-1]['cycle']} {cy[-1]['drawn_max_mm']:.1f} mm, "
                 f"{cy[-1]['drawn_max_px']:.1f} px (operator {cy[-1]['operator_amplitude_mm']:.1f}); "
                 f"{rb['tapping']['frames_per_cycle']:.1f} frames per cycle") if cy else "no cycle in the clip"
        elif "episodes" in rb:
            r = "; ".join(f"{e['context']} {e['manifestation']}"
                          + (f" at {e['tremble_hz']:g} Hz" if e.get("tremble_hz") else "")
                          + f", frames {e['onset_frame']} to {e['offset_frame']}" for e in rb["episodes"])
        else:
            r = "none (the walk's measures, H1 and H2, are stage 1's)"
        seg = c["segments"]
        lines.append(f"| {c['clip']} | {c['item']} | {c['fps']:g} | {c['n_frames']} | {c['duration_s']:.2f} | "
                     f"{seg['n_segments']} ({seg['pad']}) | {c['framing']['rule']}, {c['framing']['view']} | "
                     f"{c['torso_px']:.1f} | {r} |")
    tot = record["totals"]
    lines += ["", f"Totals: {tot['frames']} frames, {tot['segments']} generator segments of 77 frames (one call each), "
              f"{tot['bytes']} bytes on disk; "
              + ("no MP4 written." if tot["mp4_equal_all"] is None else
                 f"every MP4 decodes to its PNG frames bit for bit: {tot['mp4_equal_all']}."), "",
              "Segments are the calls of scripts/video_generate.py (contract.segment_plan, one guidance frame; --chain "
              "wan runs them in one job). Read-backs are measured on each clip's truth (mdfm/video/exam_driver.py, "
              "readback): at the point where the operator reads the tremor (the wrist for the elbow driver of E1, the "
              "middle fingertip for the postural driver of E2), its frequency and amplitude in the image along the "
              "driven direction, and twice the spectral amplitude of its 3D displacement from the shoulder against "
              "the operator's peak-to-peak chord; the drawn "
              "thumb-index fingertip distance per tapping cycle (SMPL landmark vertices) against the operator's "
              "amplitude (synth's fingertip vertices); the freezing episodes in frames.", "", "## Rows", "", "| synth spec | fps | framing, view | why |", "|---|---|---|---|"]
    for r in record["rows"]:
        lines.append(f"| {r['spec_id']} | {', '.join(f'{f:g}' for f in r['fps'])} | {r['framing']}, {r['view']} | "
                     f"{r['why']} |")
    lines.append("")
    return "\n".join(lines)
