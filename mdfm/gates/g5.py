"""G5 harness: pathology preservation (design section 9, G5, and section 2 (d); spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 2.7 and B4.22; descriptor rules docs/rules/DESCRIPTOR_RULES_DRAFT.md
DR-19.5 and OPEN-D68; rule docs/rules/G5_RULE.md, DRAFT until the owner approves and registers it).

Question. Bodies re-estimated by the model from synthetic frontal and sagittal 2D projections of the care_pd walks
must keep what the source bodies say about gait: the UPDRS_GAIT probe and the BMCLab ON versus OFF separability.
PA-MPJPE against the care_pd fits is a diagnostic only (those fits are a monocular lifter's output).

EVAL read guard. care_pd is tier EVAL (reports/manifest_tiers/tiers_by_source.csv) and evaluation-only.
read_care_pd calls common.guard_eval_read first: while docs/rules/G5_RULE.md is not registered for G5, or was edited
since, it raises canonical.Refused before the index, the label table or any record opens. With a registered rule,
the scored persons are taken from the index rows (no record), optionally asserted disjoint from the checkpoint's
training persons, the read is logged by common.log_eval_read (reports/gates/eval_reads.csv), and only then are the
labels read and each record loaded by canonical.load(row, allow_eval=True), the only allow_eval=True of this module.
The development path (selftest, amass_dev_walks) reads non-EVAL data only: it asks the same guard for its source and
refuses an EVAL answer whatever the rule. A CODY name is refused everywhere (the guard, and check_walk).

Walks (Walk). A care_pd canonical record is root-centred per frame and in torso units, coords x scale = m (MoveAtlas
build_canonical.py; mdfm/kernels/pose.py): the source body has no root track. Every descriptor is therefore
root-free and scale-free, and the projection films a walk in place. A walk's source body is joints_m() (NaN where
the record marks a joint invalid); a re-estimated body is what the predictor returns at the walk's frames. Both go
through the same descriptor code (body_descriptors), so an identity predictor gives bitwise identical descriptors
and a zero drop.

Projections (render_view, rule section 3). The body is turned so that the coordinate axis nearest its mean
pelvis-to-neck direction becomes +y (up_vector, rotation_to_y), the floor is put ANKLE_HEIGHT_M below the 5th
percentile of the ankles' heights, and the walk's facing (the mean of the per-frame horizontal normal to the hip and
shoulder lines, body_axes) fixes two static cameras aimed at the median pelvis: frontal (in front of the walker,
looking back at it) and sagittal (on the walker's left). Distance CAMERA_DISTANCE_M, eye height
CAMERA_EYE_HEIGHT_M, horizontal field of view CAMERA_HFOV_DEG, 1280 x 720 px, CAMERA_FPS. The source frames are
read at the camera frames by linear interpolation (g4.interp_frames), held from the nearest frame for the render
only (g4.hold_fill), and rendered by camera_render.render_pose2d with the tracker process (default TrackerProcess)
from one torch generator seeded by camera_seed(person, key, view). A joint missing in the source is undetected.

Re-estimation (rule section 4). predictor(walk, renders) gets one tuple of renders per view set (VIEW_SETS: each
view alone by default) and returns the body joints (T, 15, 3) m at the walk's frames (any root, frame or yaw), or None
when its input gives no window. identity_predictor returns the source body (the identity projection of the tests).
A walk without a re-estimated body (no render, or None) stays in the lines: its descriptors are NaN, imputed by the
training fold's means like every missing value, so a failure costs information; it is counted (left_out, LEFT_OUT).

Descriptors (rule section 5; the candidate set of OPEN-D68; a parameter: names and descriptor_fn). From the body in
torso units, y up (canonical, y_up), per frame the forward and left axes (body_axes), and the fore-aft inter-ankle
signal d(t) = (left ankle - right ankle) . forward. Steps: the alternating maxima and minima of d smoothed over
SMOOTH_S with prominence STEP_PROMINENCE x its 5 to 95 percentile range, same-type extrema at least MIN_STRIDE_S
apart (step_events); a stride spans three consecutive steps. cadence_spm, 60 (n - 1) / (t_last - t_first) over n
>= MIN_STEPS steps; step_time_cv_pct, 100 SD / mean of the step intervals (at least MIN_STEPS intervals, SD with
ddof 1); step_amp_torso, the median |d| at the steps; arm_swing_deg, per arm the range over a stride of the sagittal
angle of the upper arm (shoulder to elbow) relative to the trunk axis, that is its angle from the downward direction
plus the trunk's angle (pelvis to neck) from up, both positive forward, so an arm held along a pitching trunk reads a
constant; median over strides, mean of the arms (DR-8.10); arm_swing_asym_pct, |45 - atan_deg(AS_L / AS_R)| / 90 x 100
(DR-8.11); knee_rom_deg, per leg the range over a stride of the knee flexion (180 minus the angle between thigh and
shank), median over strides, mean of the legs; foot_lift_torso, per foot the range over a stride of the ankle height
relative to the pelvis, median, mean of the feet; trunk_flex_deg, the median over frames of the trunk's sagittal
angle from up, forward positive (DR-8.12 without its 10 s windows). A value that cannot be computed is NaN.

Probe and classifier (rule section 6). Person-level folds (person_folds: N_FOLDS, persons permuted by SEED). On each
fold a ridge regression (ridge_fit: features z-scored on the training rows, missing values imputed by the training
means, intercept free, penalty RIDGE_ALPHA, each walk weighted 1 / walks of its person so a person counts once) is
fitted on the training persons of one body set and predicts the held-out persons. UPDRS_GAIT: the prediction rounded
to the nearest level and clipped to the training labels' range; ON versus OFF: the continuous score, ON the
positive class. Two readings of each drop: 'refit', a probe fitted per body set (the information kept), and
'transfer', the source probe applied to the re-estimated descriptors (the values kept). refit z-scores each
descriptor on its own training rows, so it is blind to any per-descriptor affine map of the re-estimated descriptors,
a uniform collapse toward normal gait included (the risk design section 11 assigns to this drop); transfer sees
it. reading 'both' (draft, deciding): the deciding drop is the larger of the two, and a pass needs both within the
pass line. The trivial baseline is a predictor that reads nothing: for UPDRS_GAIT the population prior of the same
folds (the training fold's weighted mean label, rounded), for ON versus OFF a constant score (50 AUROC points). Both
statistics compare predictions inside a fold only (below), so a fold-constant predictor scores exactly QWK 0 and
AUROC 50 and the folds' intercepts never rank or grade walks of different folds against each other.

Lines (rule section 1). updrs_gait_qwk_drop:<view set>: QWK(source) - QWK(re-estimated), QWK the weighted kappa with
quadratic weights over the out-of-fold walks, each walk weighted 1 / walks of its person, the chance disagreement
taken inside each fold from that fold's own margins and summed over the folds with the observed disagreement
(qwk with groups). Pass when the deciding drop is within QWK_PASS of 0 on both sides (every drop of the reading) and
the source probe is at least INFORMATIVE_QWK above the trivial baseline; kill when the deciding drop exceeds
QWK_KILL. on_off_auroc_drop:<view set>: AUROC(source) - AUROC(re-estimated) in AUROC points (x 100) on the BMCLab
persons with ON and OFF walks; unit 'walk' (draft, deciding: the out-of-fold walks with person weights, (ON, OFF)
pairs formed inside a fold, so that one fitted classifier ranks each pair and the folds' intercepts do not enter)
or 'subject' (reported: per person the AUROC of its own walks, mean over persons). Pass within AUROC_PASS points on
both sides with the source at least INFORMATIVE_AUROC points above the trivial baseline; kill above AUROC_KILL.
Both lines are within-fold comparisons, so the ON versus OFF line is unchanged by any increasing affine map of a
fold's scores: a uniform shrink of the descriptors keeps it in either reading, and only the UPDRS_GAIT transfer
reading sees it. Intervals: person-clustered percentile bootstrap (BOOT_N resamples, SEED), paired between bodies,
95 percent; under 'both' the interval of the per-resample larger drop. Verdicts by common.verdict on the point.
kill_reachable_by_trivial (extra): whether the trivial drop, what a body that carries nothing gives, is above the
kill line; a body that carries noise or inverted information can drop further. gate_verdict: kill when any line
kills, pass when every line passes (draft reading). PA-MPJPE (mm, per-frame similarity Procrustes, g4.pa_mpjpe) of
each view set is reported without kill, frame-weighted per person, mean over persons.
"""
import collections
import csv
import hashlib
import math
import os
import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from mdfm import paths
from mdfm.data import canonical
from mdfm.data.canonical import Refused
from mdfm.gates import common
from mdfm.kernels import pose as pose_kernel
from mdfm.kernels.camera import CANONICAL_15

GATE = "G5"
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G5_RULE.md")
SOURCE = "care_pd"
DEV_SOURCE = "amass"
LABELS_CSV = f"{paths.MOVEATLAS}/datasets/pose/care_pd/processed/manifest.csv"   # MoveAtlas extract_care_pd.py
LABEL_COLUMNS = ("cohort", "subject", "walk", "UPDRS_GAIT")
ON_OFF_COHORT = "BMCLab"
BMCLAB_STATE = re.compile(r"(SUB\d+)_(on|off)_walk_\d+")          # scripts/probe_medication.py, med_from_walk_name
STATES = ("ON", "OFF")
QWK_PASS, QWK_KILL = 0.05, 0.10                 # QWK [D9 G5]
AUROC_PASS, AUROC_KILL = 5.0, 10.0              # AUROC points [D9 G5]
INFORMATIVE_QWK = 0.05                          # source probe above the trivial baseline, QWK (rule section 8 item 6)
INFORMATIVE_AUROC = 5.0                         # same, AUROC points (rule section 8 item 6)
N_FOLDS = 5                                     # person-level cross-fitting [spec B4.22]
SEED = 20261005
BOOT_N = 2000
RIDGE_ALPHA = 1.0                               # [cfg], rule section 8 item 3
PROBE_READINGS = ("refit", "transfer", "both")
PROBE_READING = "both"                          # rule section 8 item 4
READING_DROPS = {"refit": ("drop",), "transfer": ("drop_transfer",), "both": ("drop", "drop_transfer")}
READING_KEY = {"refit": "drop", "transfer": "drop_transfer", "both": "drop_both"}
AUROC_UNITS = ("walk", "subject")
AUROC_UNIT = "walk"                             # OPEN-D68 (unit), rule section 8 item 2
VIEWS = ("frontal", "sagittal")
VIEW_SETS = (("frontal",), ("sagittal",))       # rule section 8 item 5
CAMERA_TAG = "G5v1"
CAMERA_FPS = 30.0                               # the G1c camera rate
CAMERA_DISTANCE_M = 4.0                         # [cfg]
CAMERA_EYE_HEIGHT_M = 1.0                       # [cfg]
CAMERA_HFOV_DEG = 70.0                          # [cfg], the middle of the G1c family's 60 to 80
CAMERA_SIZE = (1280, 720)
ANKLE_HEIGHT_M = 0.08                           # ankle joint above the floor at stance [cfg]
DESCRIPTORS = ("cadence_spm", "step_time_cv_pct", "step_amp_torso", "arm_swing_deg", "arm_swing_asym_pct",
               "knee_rom_deg", "foot_lift_torso", "trunk_flex_deg")
SMOOTH_S = 0.1                                  # s, moving mean of d before the step search [cfg]
STEP_PROMINENCE = 0.2                           # of the 5 to 95 percentile range of d [cfg]
MIN_STRIDE_S = 0.4                              # s between two maxima (or two minima) of d [cfg]
MIN_FINITE = 0.5                                # share of finite frames a signal needs [cfg]
MIN_STEPS = 3
SOURCE_BODY = "source"
LEFT_OUT = {"no_render": "the source body has no frame with both pelvis and neck: no projection",
            "no_prediction": "the predictor returns no body (its input gives no window)"}
J = {name: i for i, name in enumerate(CANONICAL_15)}
PELVIS, NECK = J["pelvis"], J["neck"]
SIDES = {"L": ("left_shoulder", "left_elbow", "left_hip", "left_knee", "left_ankle"),
         "R": ("right_shoulder", "right_elbow", "right_hip", "right_knee", "right_ankle")}
UP = np.array([0.0, 1.0, 0.0])
NAN = float("nan")


# ---------------------------------------------------------------- walks
@dataclass
class Walk:
    """One walk: canonical coords (T, 15, 3) torso units, root-centred; valid (T, 15); scale m per torso unit; fps
    the record's own rate; label UPDRS_GAIT (NaN when absent); state 'ON', 'OFF' or ''."""
    person: str
    key: str
    fps: float
    coords: np.ndarray
    valid: np.ndarray
    scale: float
    cohort: str = ""
    label: float = NAN
    state: str = ""

    @property
    def t(self):
        return np.arange(len(self.coords), dtype=np.float64) / float(self.fps)

    def joints_m(self):
        """The source body (T, 15, 3) m, root-centred, NaN where the record marks a joint invalid."""
        c = np.asarray(self.coords, dtype=np.float64) * float(self.scale)
        return np.where(np.asarray(self.valid, dtype=bool)[..., None], c, np.nan)


def walk_from_record(record, person, key, cohort="", label=NAN, state=""):
    """Walk of a canonical pose record (coords, valid_mask, scale, fps)."""
    return Walk(person=str(person), key=str(key), fps=float(record.get("fps", NAN)),
                coords=np.asarray(record["coords"]), valid=np.asarray(record["valid_mask"], dtype=bool),
                scale=float(record["scale"]), cohort=str(cohort), label=float(label), state=str(state))


def check_walk(walk):
    """Refused for a CODY name (person, key or cohort); ValueError for a walk without a rate, a scale or 15 joints."""
    if canonical.is_cody(walk.person, walk.key, walk.cohort):
        raise Refused(f"CODY is never a G5 walk: {walk.person} {walk.key}")
    c = np.asarray(walk.coords)
    if c.ndim != 3 or c.shape[1:] != (15, 3) or np.asarray(walk.valid).shape != c.shape[:2]:
        raise ValueError(f"{walk.person} {walk.key}: coords {c.shape}, valid {np.asarray(walk.valid).shape}")
    if not (math.isfinite(walk.fps) and walk.fps > 0 and math.isfinite(walk.scale) and walk.scale > 0):
        raise ValueError(f"{walk.person} {walk.key}: fps {walk.fps}, scale {walk.scale}")
    return walk


def identity_predictor(walk, renders):
    """The identity projection: the source body itself, whatever the renders."""
    return walk.joints_m()


# ---------------------------------------------------------------- geometry
def canonical_body(joints):
    """(coords (T, 15, 3) float64 torso units root-centred with NaN where invalid, valid (T, 15), scale) of a body
    (T, 15, 3) in any unit with NaN for a missing joint: the MoveAtlas 3D normalization (kernels/pose.normalize_3d)."""
    j = np.asarray(joints, dtype=np.float64)
    ok = np.isfinite(j).all(-1)
    coords, scale, valid = pose_kernel.normalize_3d(np.where(ok[..., None], j, np.nan), ok)
    c = np.asarray(coords, dtype=np.float64)
    return np.where(valid[..., None], c, np.nan), valid, scale


def up_vector(joints):
    """The coordinate axis (+-x, +-y or +-z) nearest the mean pelvis-to-neck direction over the frames where both are
    finite. ValueError when no frame has both."""
    j = np.asarray(joints, dtype=np.float64)
    v = j[:, NECK] - j[:, PELVIS]
    ok = np.isfinite(v).all(-1)
    if not ok.any():
        raise ValueError("no frame with both the pelvis and the neck")
    m = v[ok].mean(0)
    k = int(np.argmax(np.abs(m)))
    u = np.zeros(3)
    u[k] = 1.0 if m[k] > 0 else -1.0
    return u


def rotation_to_y(u):
    """Proper rotation R (3, 3) with R u = +y for a unit vector u (Rodrigues; a half turn about x when u = -y)."""
    u = np.asarray(u, dtype=np.float64) / np.linalg.norm(u)
    c = float(u @ UP)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, UP)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def y_up(joints):
    """The body turned so that its up_vector becomes +y."""
    j = np.asarray(joints, dtype=np.float64)
    return j @ rotation_to_y(up_vector(j)).T


def _horizontal_unit(v):
    h = np.array(v, dtype=np.float64)
    h[..., 1] = 0.0
    n = np.linalg.norm(h, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = h / n
    out[~(np.nan_to_num(n[..., 0]) > 1e-9)] = np.nan
    return out


def body_axes(j):
    """(forward (T, 3), left (T, 3)): unit horizontal axes of a body (T, 15, 3) y up; left from the sum of the hip and
    shoulder lines (left minus right, either alone when the other is missing), forward = left x up; NaN where both
    lines are missing."""
    hips = j[:, J["left_hip"]] - j[:, J["right_hip"]]
    shoulders = j[:, J["left_shoulder"]] - j[:, J["right_shoulder"]]
    fh, fs = np.isfinite(hips).all(-1), np.isfinite(shoulders).all(-1)
    lat = np.where(fh[:, None], hips, 0.0) + np.where(fs[:, None], shoulders, 0.0)
    lat[~(fh | fs)] = np.nan
    left = _horizontal_unit(lat)
    forward = np.stack([-left[:, 2], np.zeros(len(left)), left[:, 0]], axis=-1)
    forward[~np.isfinite(left).all(-1)] = np.nan
    return forward, left


def mean_direction(v):
    """Unit horizontal mean of unit vectors (T, 3), NaN rows ignored; +z when none is finite or they cancel."""
    ok = np.isfinite(v).all(-1)
    m = v[ok].mean(0) if ok.any() else np.array([0.0, 0.0, 1.0])
    out = _horizontal_unit(m[None])[0]
    return out if np.isfinite(out).all() else np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- descriptors
def _fill(x, min_finite=MIN_FINITE):
    """x (T,) with NaN filled by linear interpolation (held at the ends); None when under min_finite of it is finite."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if len(x) == 0 or ok.mean() < min_finite or ok.sum() < 2:
        return None
    i = np.arange(len(x))
    return np.interp(i, i[ok], x[ok])


def _smooth(x, n):
    """Centred moving mean over n samples (edges padded with the end values)."""
    if n <= 1:
        return x
    a, b = n // 2, n - 1 - n // 2
    p = np.concatenate([np.full(a, x[0]), x, np.full(b, x[-1])])
    return np.convolve(p, np.ones(n) / n, mode="valid")


def step_events(d, fps, smooth_s=SMOOTH_S, prominence=STEP_PROMINENCE, min_stride_s=MIN_STRIDE_S):
    """Frame indices (int64, sorted) of the steps of a fore-aft inter-ankle signal d (T,) (module docstring): the
    maxima and minima of the smoothed signal, alternating (of two consecutive extrema of one type, the larger |d| is
    kept). Empty when d has too few finite samples or no spread."""
    x = _fill(d)
    if x is None or len(x) < 3:
        return np.zeros(0, dtype=np.int64)
    x = _smooth(x, max(1, int(round(smooth_s * fps))))
    lo, hi = np.percentile(x, [5.0, 95.0])
    span = float(hi - lo)
    if not span > 0:
        return np.zeros(0, dtype=np.int64)
    dist = max(1, int(round(min_stride_s * fps)))
    peaks, _ = find_peaks(x, prominence=prominence * span, distance=dist)
    troughs, _ = find_peaks(-x, prominence=prominence * span, distance=dist)
    out = []
    for i, s in sorted([(int(i), 1) for i in peaks] + [(int(i), -1) for i in troughs]):
        if out and out[-1][1] == s:
            if abs(x[i]) > abs(x[out[-1][0]]):
                out[-1] = (i, s)
            continue
        out.append((i, s))
    return np.asarray([i for i, _ in out], dtype=np.int64)


def _sagittal_deg(v, forward, from_down=False):
    """Angle (deg) of vectors v (T, 3), y up, in the sagittal plane of forward (T, 3): from up (or from down),
    positive forward."""
    a = (v * forward).sum(-1)
    b = -v[:, 1] if from_down else v[:, 1]
    return np.degrees(np.arctan2(a, b))


def _flexion_deg(a, b, c):
    """180 minus the angle (deg) at b between a - b and c - b, per frame."""
    u, v = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _stride_range(x, events):
    """Median over strides (events k to k + 2) of the range of x (T,) inside the stride (strides with under
    MIN_FINITE finite samples skipped); NaN without a stride."""
    vals = []
    for a, b in zip(events[:-2], events[2:]):
        s = x[a:b + 1]
        ok = np.isfinite(s)
        if len(s) and ok.mean() >= MIN_FINITE:
            vals.append(float(s[ok].max() - s[ok].min()))
    return float(np.median(vals)) if vals else NAN


def _mean_finite(*v):
    v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else NAN


def descriptors(joints, fps, names=DESCRIPTORS):
    """{name: value} of the root-free gait descriptors (module docstring) of one body (T, 15, 3) in any unit, NaN for
    a missing joint; fps its rate. NaN where a value cannot be computed. ValueError for an unknown name."""
    unknown = [n for n in names if n not in DESCRIPTORS]
    if unknown:
        raise ValueError(f"unknown descriptors {unknown}; known: {list(DESCRIPTORS)}")
    out = dict.fromkeys(names, NAN)
    c, _, _ = canonical_body(joints)
    try:
        j = y_up(c)
    except ValueError:
        return out
    fps = float(fps)
    forward, _ = body_axes(j)
    d = ((j[:, J["left_ankle"]] - j[:, J["right_ankle"]]) * forward).sum(-1)
    ev = step_events(d, fps)
    trunk = _sagittal_deg(j[:, NECK] - j[:, PELVIS], forward)
    val = {}
    if len(ev) >= MIN_STEPS and ev[-1] > ev[0]:
        val["cadence_spm"] = 60.0 * (len(ev) - 1) / ((ev[-1] - ev[0]) / fps)
        dd = _fill(d)
        val["step_amp_torso"] = float(np.median(np.abs(dd[ev])))
    iv = np.diff(ev) / fps
    if len(iv) >= MIN_STEPS and iv.mean() > 0:
        val["step_time_cv_pct"] = 100.0 * float(iv.std(ddof=1)) / float(iv.mean())
    arms, knees, feet = {}, {}, {}
    for side, (sh, el, hip, kn, an) in SIDES.items():
        # the arm from the trunk's downward axis: an arm along a trunk leaning theta forward reads -theta from down
        arm = _sagittal_deg(j[:, J[el]] - j[:, J[sh]], forward, from_down=True) + trunk
        arms[side] = _stride_range(arm, ev)
        knees[side] = _stride_range(_flexion_deg(j[:, J[hip]], j[:, J[kn]], j[:, J[an]]), ev)
        feet[side] = _stride_range(j[:, J[an], 1] - j[:, PELVIS, 1], ev)
    val["arm_swing_deg"] = _mean_finite(arms["L"], arms["R"])
    if math.isfinite(arms["L"]) and math.isfinite(arms["R"]) and (arms["L"] > 0 or arms["R"] > 0):
        val["arm_swing_asym_pct"] = abs(45.0 - math.degrees(math.atan2(arms["L"], arms["R"]))) / 90.0 * 100.0
    val["knee_rom_deg"] = _mean_finite(knees["L"], knees["R"])
    val["foot_lift_torso"] = _mean_finite(feet["L"], feet["R"])
    if np.isfinite(trunk).mean() >= MIN_FINITE:
        val["trunk_flex_deg"] = float(np.nanmedian(trunk))
    for n in names:
        if n in val:
            out[n] = float(val[n])
    return out


def body_descriptors(joints, fps, names=DESCRIPTORS, descriptor_fn=None):
    """The descriptors of a body through descriptor_fn (default descriptors): the one path of source and
    re-estimated bodies."""
    fn = descriptors if descriptor_fn is None else descriptor_fn
    got = fn(joints, fps, names)
    return {n: float(got.get(n, NAN)) for n in names}


# ---------------------------------------------------------------- projections
def camera_seed(person, key, view, tag=CAMERA_TAG):
    """Seed of the tracker process of one walk and view: the first 63 bits of
    SHA-256('<person>\\n<key>\\n<view>\\n<tag>')."""
    h = hashlib.sha256(f"{person}\n{key}\n{view}\n{tag}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def walk_world(walk):
    """(world (T, 15, 3) m y up with the floor at 0 and NaN where invalid, facing (3,) unit horizontal) of a walk's
    source body (module docstring, Projections); None when no frame has both pelvis and neck."""
    j = walk.joints_m()
    try:
        w = y_up(j)
    except ValueError:
        return None
    forward, _ = body_axes(w)
    ank = w[:, [J["left_ankle"], J["right_ankle"]], 1]
    floor = float(np.nanpercentile(ank, 5.0)) - ANKLE_HEIGHT_M if np.isfinite(ank).any() else \
        float(np.nanmin(w[..., 1]))
    w[..., 1] -= floor
    return w, mean_direction(forward)


def view_camera(view, target, facing, distance=CAMERA_DISTANCE_M, eye_height=CAMERA_EYE_HEIGHT_M,
                hfov_deg=CAMERA_HFOV_DEG, fps=CAMERA_FPS, size=CAMERA_SIZE):
    """The fixed camera of a view (camera.Camera, float64): frontal at `distance` along the walker's facing, sagittal
    at `distance` along its left (facing turned by +90 degrees about up), eye at `eye_height` m, looking at target (3,)
    m, focal length from the horizontal field of view, principal point at the image centre, no distortion."""
    import torch

    from mdfm.kernels.camera import Camera, intrinsics, look_at
    f = np.asarray(facing, dtype=np.float64)
    left = np.array([f[2], 0.0, -f[0]])                       # up x facing
    if view == "frontal":
        direction = f
    elif view == "sagittal":
        direction = left
    else:
        raise ValueError(f"unknown view {view!r}; known: {list(VIEWS)}")
    target = np.asarray(target, dtype=np.float64)
    eye = target + distance * direction
    eye[1] = eye_height
    R, t = look_at(torch.as_tensor(eye), torch.as_tensor(target))
    w, h = size
    fpx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Camera(intrinsics(fpx, fpx, w / 2.0, h / 2.0), R, t, 0.0, 0.0, int(w), int(h), float(fps))


def render_view(walk, view, tracker=None, fps=CAMERA_FPS, tag=CAMERA_TAG):
    """The 2D input of one walk and view (module docstring): {view, coords (Q, 15, 3) float32 normalized pixels with
    z = 0 and 0 where undetected, valid (Q, 15), t (Q,) s on the walk's clock, fps, scale (px per torso unit), seed,
    camera, info {frames, detected_share, truth_missing_share, camera_KRt}}; None when the walk has no frame with
    both pelvis and neck."""
    import torch

    from mdfm.factory.camera_render import render_pose2d
    from mdfm.gates import g4
    got = walk_world(walk)
    if got is None:
        return None
    world, facing = got
    t = walk.t
    tc = t[0] + np.arange(0.0, (t[-1] - t[0]) + 1e-9, 1.0 / fps)
    xc = g4.interp_frames(world, t, tc)
    missing = ~np.isfinite(xc).all(-1)
    filled = g4.hold_fill(xc)
    if filled is None:
        return None
    pelvis = world[:, PELVIS]
    target = np.array([0.0, float(np.nanmedian(pelvis[:, 1])), 0.0])
    cam = view_camera(view, target, facing, fps=fps)
    seed = camera_seed(walk.person, walk.key, view, tag)
    gen = torch.Generator().manual_seed(int(seed))
    record, _ = render_pose2d(torch.as_tensor(filled, dtype=torch.float64), cam, gen, tracker)
    valid = np.asarray(record["valid_mask"], dtype=bool) & ~missing & ~missing[:, :1]
    coords = np.zeros(np.asarray(record["coords"]).shape, dtype=np.float32)
    coords[..., :2] = np.where(valid[..., None], np.asarray(record["coords"])[..., :2], 0.0)
    krt = torch.cat([cam.K.reshape(-1), cam.R.reshape(-1), cam.t.reshape(-1)]).tolist()
    return {"view": view, "coords": coords, "valid": valid, "t": tc, "fps": float(record["fps"]),
            "scale": float(record["scale"]), "seed": int(seed), "camera": cam,
            "info": {"frames": int(len(tc)), "detected_share": float(valid.mean()) if valid.size else 0.0,
                     "truth_missing_share": float(missing.mean()) if missing.size else 0.0, "camera_KRt": krt}}


def view_set_name(views):
    return "+".join(views)


# ---------------------------------------------------------------- per-walk table
def walk_table(walks, predictor, view_sets=VIEW_SETS, names=DESCRIPTORS, descriptor_fn=None, tracker=None,
               render=True, stats=None, left_out=None):
    """DataFrame, one row per (walk, body): walk (index), person_id, key, cohort, label, state, fps, frames, body
    ('source' or a view set name), the descriptors, left_out (bool), detected_share and pa_mpjpe_mm (NaN for the
    source). Walks are consumed one at a time (a generator is fine) and checked (check_walk). render False skips the
    projections (the predictor then gets None renders; the tests of the line arithmetic use it). left_out (a list)
    receives one dict per (walk, view set) without a body; stats counts walks and left-out bodies."""
    import pandas as pd

    from mdfm.gates import g4
    stats = collections.Counter() if stats is None else stats
    lo = [] if left_out is None else left_out
    for vs in view_sets:
        for v in vs:
            if v not in VIEWS:
                raise ValueError(f"unknown view {v!r}; known: {list(VIEWS)}")
    rows = []
    for i, walk in enumerate(walks):
        check_walk(walk)
        stats["walks"] += 1
        src_joints = walk.joints_m()
        base = {"walk": i, "person_id": walk.person, "key": walk.key, "cohort": walk.cohort,
                "label": float(walk.label), "state": walk.state, "fps": float(walk.fps), "frames": int(len(src_joints))}
        rows.append({**base, "body": SOURCE_BODY, **body_descriptors(src_joints, walk.fps, names, descriptor_fn),
                     "left_out": False, "detected_share": NAN, "pa_mpjpe_mm": NAN})
        for vs in view_sets:
            name = view_set_name(vs)
            renders = tuple(render_view(walk, v, tracker) for v in vs) if render else tuple(None for _ in vs)
            if render and any(r is None for r in renders):
                joints, reason = None, "no_render"
            else:
                joints = predictor(walk, renders)
                reason = "no_prediction" if joints is None else ""
            if joints is None:
                lo.append({"person_id": walk.person, "key": walk.key, "view_set": name, "reason": reason,
                           "meaning": LEFT_OUT[reason], "frames": int(len(src_joints)), "fps": float(walk.fps)})
                stats[f"left_out_{reason}"] += 1
                rows.append({**base, "body": name, **dict.fromkeys(names, NAN), "left_out": True,
                             "detected_share": NAN, "pa_mpjpe_mm": NAN})
                continue
            joints = np.asarray(joints, dtype=np.float64)
            if joints.shape != src_joints.shape:
                raise ValueError(f"{walk.person} {walk.key}: predicted body {joints.shape} for {src_joints.shape}")
            det = [r["info"]["detected_share"] for r in renders if r is not None]
            rows.append({**base, "body": name, **body_descriptors(joints, walk.fps, names, descriptor_fn),
                         "left_out": False, "detected_share": float(np.mean(det)) if det else NAN,
                         "pa_mpjpe_mm": g4.pa_mpjpe(joints, src_joints)})
    cols = ["walk", "person_id", "key", "cohort", "label", "state", "fps", "frames", "body", *names, "left_out",
            "detected_share", "pa_mpjpe_mm"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- folds and the probe
def person_folds(persons, k=N_FOLDS, seed=SEED):
    """{person: fold in 0..k-1}: the sorted distinct persons permuted by the seed, fold = rank mod k. ValueError with
    fewer than k persons."""
    ids = sorted({str(p) for p in persons})
    if len(ids) < k:
        raise ValueError(f"{len(ids)} persons for {k} folds")
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {ids[j]: int(r % k) for r, j in enumerate(perm)}


def person_weights(persons):
    """Weight of each row (N,): 1 / rows of its person, so that each person sums to 1."""
    p = np.asarray(persons).astype(str)
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def ridge_fit(X, y, w=None, alpha=RIDGE_ALPHA):
    """Weighted ridge regression with a free intercept on z-scored features: missing values (NaN) take the training
    column's weighted mean, a column without spread (or without any value) is dropped. Returns the model dict."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    fin = np.isfinite(X)
    ws = (w[:, None] * fin).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (w[:, None] * np.where(fin, X, 0.0)).sum(0) / ws
    mu = np.where(ws > 0, mu, 0.0)
    Xf = np.where(fin, X, mu[None])
    sd = np.sqrt((w[:, None] * (Xf - mu[None]) ** 2).sum(0) / w.sum())
    keep = sd > 1e-12
    Z = np.zeros_like(Xf)
    Z[:, keep] = (Xf[:, keep] - mu[keep]) / sd[keep]
    ybar = float((w * y).sum() / w.sum())
    A = Z.T @ (w[:, None] * Z) + alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (w * (y - ybar)))
    beta[~keep] = 0.0
    return {"mu": mu, "sd": np.where(keep, sd, 1.0), "keep": keep, "beta": beta, "ybar": ybar}


def ridge_predict(model, X):
    X = np.asarray(X, dtype=np.float64)
    Xf = np.where(np.isfinite(X), X, model["mu"][None])
    Z = np.where(model["keep"][None], (Xf - model["mu"][None]) / model["sd"][None], 0.0)
    return model["ybar"] + Z @ model["beta"]


def crossfit(features, y, persons, folds, kind, alpha=RIDGE_ALPHA, source=SOURCE_BODY):
    """Out-of-fold predictions on person-level folds (module docstring). features {body: (N, k)}, y (N,), persons (N,),
    folds {person: fold}; kind 'ordinal' (rounded to the nearest level, clipped to the training labels' range) or
    'score'. Returns {body: refit predictions}, {body: the source probe's predictions on that body}, trivial (N,):
    the population prior of each fold."""
    if kind not in ("ordinal", "score"):
        raise ValueError(f"kind {kind!r}")
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(persons).astype(str)
    w = person_weights(p)
    f = np.asarray([folds[x] for x in p])
    n = len(y)
    refit = {b: np.full(n, np.nan) for b in features}
    transfer = {b: np.full(n, np.nan) for b in features}
    trivial = np.full(n, np.nan)
    for k in sorted(set(f.tolist())):
        tr, te = f != k, f == k
        if not tr.any() or not te.any():
            continue
        models = {b: ridge_fit(np.asarray(X)[tr], y[tr], w[tr], alpha) for b, X in features.items()}
        for b, X in features.items():
            refit[b][te] = ridge_predict(models[b], np.asarray(X)[te])
            transfer[b][te] = ridge_predict(models[source], np.asarray(X)[te])
        trivial[te] = float((w[tr] * y[tr]).sum() / w[tr].sum())
        if kind == "ordinal":
            lo, hi = float(y[tr].min()), float(y[tr].max())
            for arr in (*refit.values(), *transfer.values(), trivial):
                arr[te] = np.clip(np.round(arr[te]), lo, hi)
    return refit, transfer, trivial


# ---------------------------------------------------------------- statistics
def _kappa(O, Wq):
    """Quadratic weighted kappa of weighted confusion matrices O (..., G, L, L), one per group, with disagreement
    weights Wq (L, L): the chance disagreement of each group from its own margins (an empty group adds nothing),
    observed and chance disagreements summed over the groups. NaN where the summed chance disagreement is 0."""
    tot = O.sum((-1, -2))
    with np.errstate(invalid="ignore", divide="ignore"):
        E = O.sum(-1)[..., :, None] * O.sum(-2)[..., None, :] / tot[..., None, None]
    E = np.where((tot > 0)[..., None, None], E, 0.0)
    num, den = (Wq * O).sum((-1, -2, -3)), (Wq * E).sum((-1, -2, -3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1.0 - num / den, np.nan)


def _group_index(groups, n):
    """Inverse (n,) of the groups (None: one group) and the number of groups."""
    if groups is None:
        return np.zeros(n, dtype=np.int64), 1
    g = np.unique(np.asarray(groups), return_inverse=True)[1].reshape(-1)
    if len(g) != n:
        raise ValueError(f"{len(g)} groups for {n} rows")
    return g, int(g.max()) + 1 if n else 0


def _levels(*arrays):
    vals = np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not np.allclose(vals, np.round(vals)):
        raise ValueError("QWK compares integer levels")
    return np.arange(int(vals.min()), int(vals.max()) + 1) if len(vals) else np.zeros(0, dtype=np.int64)


def qwk(y, pred, w=None, groups=None):
    """Quadratic weighted kappa of integer labels y and predictions pred (N,), rows weighted by w (default 1). groups
    (N,), optional: the chance disagreement is taken inside each group (the out-of-fold walks of one fold, graded by
    one fitted probe) and summed over the groups (_kappa), so a predictor constant inside each group scores exactly
    0. NaN when there is no chance disagreement (one rater constant over every weighted row of every group)."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    lv = _levels(y, pred)
    L = len(lv)
    if L == 0:
        return NAN
    g, G = _group_index(groups, len(y))
    O = np.zeros((G, L, L))
    np.add.at(O, (g, y.astype(np.int64) - lv[0], pred.astype(np.int64) - lv[0]), w)
    Wq = (np.arange(L)[:, None] - np.arange(L)[None]) ** 2.0
    return float(_kappa(O, Wq))


def auroc(score, positive, w=None, groups=None):
    """Weighted AUROC of a score (N,) for the positive rows (bool (N,)): the weighted share of (positive, negative)
    pairs ordered correctly, ties counted half; NaN without both classes. groups (N,), optional: pairs are formed
    inside a group only (the out-of-fold scores of one fold, ranked by one fitted classifier), the share pooled over
    the groups' pairs."""
    s = np.asarray(score, dtype=np.float64)
    pos = np.asarray(positive, dtype=bool)
    w = np.ones(len(s)) if w is None else np.asarray(w, dtype=np.float64)
    parts = _auroc_parts(s, pos, groups)
    num, den = _auroc_sum(parts, w)
    return float(num / den) if den > 0 else NAN


def _auroc_parts(s, pos, groups=None):
    """[(rows, inverse of the sorted unique scores, positive)] per group (one group when groups is None)."""
    g = np.zeros(len(s), dtype=np.int64) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    out = []
    for k in np.unique(g):
        rows = np.flatnonzero(g == k)
        out.append((rows, np.unique(s[rows], return_inverse=True)[1], pos[rows]))
    return out


def _auroc_sum(parts, w):
    """(weighted correctly ordered pairs, weighted pairs) summed over the groups of _auroc_parts."""
    num = den = 0.0
    for rows, inv, pos in parts:
        ww = w[rows]
        U = int(inv.max()) + 1 if len(inv) else 0
        wp = np.bincount(inv, weights=ww * pos, minlength=U)
        wn = np.bincount(inv, weights=ww * ~pos, minlength=U)
        num += float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum())
        den += float(wp.sum() * wn.sum())
    return num, den


def boot_counts(n_persons, n=BOOT_N, seed=SEED):
    """(n, P) int: how many times each person is drawn in each of n person-bootstrap resamples."""
    idx = np.random.default_rng(seed).integers(0, n_persons, size=(int(n), int(n_persons)))
    return np.stack([np.bincount(r, minlength=n_persons) for r in idx]) if n else np.zeros((0, n_persons), np.int64)


def interval(boot, level=0.95):
    """Percentile interval of the finite bootstrap values; (nan, nan) without any."""
    b = np.asarray(boot, dtype=np.float64)
    b = b[np.isfinite(b)]
    if not len(b):
        return (NAN, NAN)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(b, [a, 1.0 - a])
    return (float(lo), float(hi))


def qwk_boot(y, preds, persons, counts, groups=None):
    """{name: (point, boot (n,))} of the weighted QWK (person weights) of each prediction in preds {name: (N,)},
    the chance disagreement taken inside each group of groups (the folds; None: one group; qwk), the bootstrap
    reusing the same person draws counts (n, P) for every name (paired). ValueError for a person in two groups."""
    y = np.asarray(y, dtype=np.float64)
    ids, pidx = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    w = person_weights(persons)
    lv = _levels(y, *preds.values())
    L, P = len(lv), len(ids)
    g, G = _group_index(groups, len(y))
    pg = np.zeros(P, dtype=np.int64)
    pg[pidx] = g
    if not np.array_equal(pg[pidx], g):
        raise ValueError("a person sits in two groups")
    members = [np.flatnonzero(pg == k) for k in range(G)]
    counts = np.asarray(counts, dtype=np.float64)
    Wq = (np.arange(L)[:, None] - np.arange(L)[None]) ** 2.0
    out = {}
    for name, p in preds.items():
        O = np.zeros((P, L, L))
        np.add.at(O, (pidx, y.astype(np.int64) - lv[0], np.asarray(p, dtype=np.float64).astype(np.int64) - lv[0]), w)
        Op = O.reshape(P, L * L)
        point = float(_kappa(np.stack([Op[m].sum(0) for m in members]).reshape(G, L, L), Wq))
        if len(counts):
            Ob = np.stack([counts[:, m] @ Op[m] for m in members], axis=1).reshape(-1, G, L, L)
            boot = _kappa(Ob, Wq)
        else:
            boot = np.zeros(0)
        out[name] = (point, np.asarray(boot, dtype=np.float64))
    return out


def auroc_boot(scores, positive, persons, counts, unit=AUROC_UNIT, groups=None):
    """{name: (point, boot (n,))} in AUROC points (x 100) of each score in scores {name: (N,)} for the positive rows,
    the bootstrap reusing the person draws counts (n, P) (paired). unit 'walk': the walks with person weights, pairs
    formed inside a group of groups (the folds; None: one group); 'subject': per person the unweighted AUROC of its
    own walks, mean over the persons (a person without both classes is NaN and left out of the mean)."""
    if unit not in AUROC_UNITS:
        raise ValueError(f"unit {unit!r}; known: {list(AUROC_UNITS)}")
    pos = np.asarray(positive, dtype=bool)
    ids, pidx = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    w = person_weights(persons)
    out = {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        if unit == "walk":
            parts = _auroc_parts(s, pos, groups)

            def ratio(ww):
                num, den = _auroc_sum(parts, ww)
                return num / den if den > 0 else NAN

            point = ratio(w)
            boot = np.asarray([ratio(w * c[pidx]) for c in counts])
        else:
            per = np.asarray([auroc(s[pidx == k], pos[pidx == k]) for k in range(len(ids))])
            ok = np.isfinite(per)
            point = float(per[ok].mean()) if ok.any() else NAN
            c = counts[:, ok] if len(counts) else counts
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = (c @ per[ok]) / c.sum(1) if len(counts) else np.zeros(0)
        out[name] = (100.0 * point, 100.0 * np.asarray(boot, dtype=np.float64))
    return out


# ---------------------------------------------------------------- lines
def _bodies(table, view_sets):
    return [SOURCE_BODY] + [view_set_name(vs) for vs in view_sets]


def _features(table, rows, bodies, names):
    """{body: (N, k)} descriptor matrices of the walks `rows` (walk indices, in order) for each body."""
    out = {}
    for b in bodies:
        t = table[table["body"] == b].set_index("walk")
        out[b] = t.loc[rows, list(names)].to_numpy(dtype=np.float64)
    return out


def _imputed(feats):
    return {b: int((~np.isfinite(X)).sum()) for b, X in feats.items()}


def updrs_set(table):
    """Source rows with a finite label: (walk indices, persons, labels)."""
    s = table[(table["body"] == SOURCE_BODY) & np.isfinite(table["label"].astype(float))]
    return s["walk"].to_numpy(), s["person_id"].astype(str).to_numpy(), s["label"].to_numpy(dtype=np.float64)


def on_off_set(table):
    """Source rows with a state of the persons that have both ON and OFF walks: (walk indices, persons, ON (bool),
    persons with one state only)."""
    s = table[(table["body"] == SOURCE_BODY) & table["state"].isin(STATES)]
    both = s.groupby("person_id")["state"].nunique()
    keep = set(both.index[both == 2].astype(str))
    single = sorted(set(both.index.astype(str)) - keep)
    s = s[s["person_id"].astype(str).isin(keep)]
    return s["walk"].to_numpy(), s["person_id"].astype(str).to_numpy(), (s["state"] == "ON").to_numpy(), single


def summarize(table, view_sets=VIEW_SETS, names=DESCRIPTORS, n_folds=N_FOLDS, seed=SEED, n_boot=BOOT_N,
              alpha=RIDGE_ALPHA):
    """The G5 numbers of a walk table (module docstring): {updrs_gait: {view set: line}, on_off: {unit: {view set:
    line}}, pa_mpjpe: {view set: {point, interval, n_persons}}, sets: counts}. A line holds source, re-estimated,
    transfer and trivial values with intervals, drop, drop_transfer and drop_both (the larger of the two) with paired
    intervals, trivial_drop (the drop of a body that carries nothing: source minus trivial), n_persons, n_walks,
    imputed entries per body and left-out bodies; evaluated False with a reason when the set cannot carry the line."""
    bodies = _bodies(table, view_sets)
    out = {"updrs_gait": {}, "on_off": {u: {} for u in AUROC_UNITS}, "pa_mpjpe": {}, "sets": {}}
    left = {b: int(table[(table["body"] == b)]["left_out"].sum()) for b in bodies}

    # UPDRS_GAIT probe
    rows, persons, y = updrs_set(table)
    reason = _set_reason(persons, n_folds, y, "label levels")
    out["sets"]["updrs_gait"] = {"walks": int(len(rows)), "persons": int(len(set(persons.tolist())))}
    if reason:
        for vs in bodies[1:]:
            out["updrs_gait"][vs] = {"evaluated": False, "reason": reason}
    else:
        folds = person_folds(persons, n_folds, seed)
        feats = _features(table, rows, bodies, names)
        refit, transfer, trivial = crossfit(feats, y, persons, folds, "ordinal", alpha)
        counts = boot_counts(len(set(persons.tolist())), n_boot, seed)
        preds = {"source": refit[SOURCE_BODY], "trivial": trivial,     # the trivial prior is fold-constant: QWK 0
                 **{f"refit:{b}": refit[b] for b in bodies[1:]}, **{f"transfer:{b}": transfer[b] for b in bodies[1:]}}
        q = qwk_boot(y, preds, persons, counts, groups=np.asarray([folds[p] for p in persons]))
        imp = _imputed(feats)
        for b in bodies[1:]:
            out["updrs_gait"][b] = _line(q, b, persons, rows, imp, table, b)
    # BMCLab ON versus OFF
    rows, persons, on, single = on_off_set(table)
    reason = _set_reason(persons, n_folds, on.astype(float), "states")
    out["sets"]["on_off"] = {"walks": int(len(rows)), "persons": int(len(set(persons.tolist()))),
                             "persons_with_one_state": single}
    if reason:
        for unit in AUROC_UNITS:
            for b in bodies[1:]:
                out["on_off"][unit][b] = {"evaluated": False, "reason": reason}
    else:
        folds = person_folds(persons, n_folds, seed)
        feats = _features(table, rows, bodies, names)
        refit, transfer, _ = crossfit(feats, on.astype(float), persons, folds, "score", alpha)
        counts = boot_counts(len(set(persons.tolist())), n_boot, seed)
        scores = {"source": refit[SOURCE_BODY], "trivial": np.zeros(len(rows)),     # a score that carries nothing: 50
                  **{f"refit:{b}": refit[b] for b in bodies[1:]}, **{f"transfer:{b}": transfer[b] for b in bodies[1:]}}
        imp = _imputed(feats)
        fold_of_row = np.asarray([folds[p] for p in persons])
        for unit in AUROC_UNITS:
            a = auroc_boot(scores, on, persons, counts, unit, groups=fold_of_row)
            for b in bodies[1:]:
                out["on_off"][unit][b] = _line(a, b, persons, rows, imp, table, b)
    # PA-MPJPE diagnostic
    for b in bodies[1:]:
        t = table[(table["body"] == b) & np.isfinite(table["pa_mpjpe_mm"].astype(float))]
        if not len(t):
            out["pa_mpjpe"][b] = {"point": NAN, "interval": (NAN, NAN), "n_persons": 0, "n_walks": 0}
            continue
        v, p, wt = t["pa_mpjpe_mm"].to_numpy(float), t["person_id"].astype(str).to_numpy(), t["frames"].to_numpy(float)
        out["pa_mpjpe"][b] = {"point": common.person_weighted_mean(v, p, wt),
                              "interval": common.cluster_bootstrap(v, p, n=max(1, n_boot), seed=seed, weights=wt),
                              "n_persons": int(len(set(p.tolist()))), "n_walks": int(len(t))}
    out["left_out"] = left
    return out


def _set_reason(persons, n_folds, y, what):
    n = len(set(np.asarray(persons).tolist()))
    if n < n_folds:
        return f"{n} persons, fewer than the {n_folds} folds"
    if len(set(np.asarray(y).tolist())) < 2:
        return f"one value of the {what}"
    return ""


def _larger(a, b):
    """The larger of two drops, a NaN one ignored (NaN when both are)."""
    v = [float(x) for x in (a, b) if _finite(x)]
    return max(v) if v else NAN


def _line(stats, body, persons, rows, imputed, table, name):
    src, triv = stats["source"], stats["trivial"]
    ref, tra = stats[f"refit:{body}"], stats[f"transfer:{body}"]
    left = table[(table["body"] == name) & table["walk"].isin(rows)]["left_out"]
    d_ref, d_tra = src[0] - ref[0], src[0] - tra[0]
    return {"evaluated": True, "source": src[0], "reestimated": ref[0], "transfer": tra[0], "trivial": triv[0],
            "source_interval": interval(src[1]), "reestimated_interval": interval(ref[1]),
            "transfer_interval": interval(tra[1]), "trivial_interval": interval(triv[1]),
            "drop": d_ref, "drop_interval": interval(src[1] - ref[1]),
            "drop_transfer": d_tra, "drop_transfer_interval": interval(src[1] - tra[1]),
            "drop_both": _larger(d_ref, d_tra),
            "drop_both_interval": interval(np.fmax(src[1] - ref[1], src[1] - tra[1])),
            "trivial_drop": src[0] - triv[0], "informative_margin": src[0] - triv[0],
            "n_persons": int(len(set(np.asarray(persons).tolist()))), "n_walks": int(len(rows)),
            "imputed_source": imputed[SOURCE_BODY], "imputed_reestimated": imputed[body],
            "left_out": int(left.sum())}


# ---------------------------------------------------------------- results
def _finite(x):
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def gate_results(summ, rule_sha256="", checkpoint_sha256="", train_persons_sha256="", reading=PROBE_READING,
                 unit=AUROC_UNIT):
    """GateResult lines of mdfm.gates.common, one per line and view set (module docstring, Lines). reading picks
    the deciding drop: 'refit', 'transfer' or 'both' (the larger of the two; default), unit the deciding AUROC unit;
    every reading and unit stays in the summary. A pass also needs every drop of the reading at least -pass (within
    the pass line on both sides) and the source probe at least the informative margin above its trivial baseline; a
    condition that cannot be evaluated blocks the pass. extra kill_reachable_by_trivial: the trivial drop above the
    kill line (None when not finite)."""
    if reading not in PROBE_READINGS:
        raise ValueError(f"reading {reading!r}; known: {list(PROBE_READINGS)}")
    if unit not in AUROC_UNITS:
        raise ValueError(f"unit {unit!r}; known: {list(AUROC_UNITS)}")
    key = READING_KEY[reading]
    out = []
    specs = (("updrs_gait_qwk_drop", summ["updrs_gait"], QWK_PASS, QWK_KILL, INFORMATIVE_QWK, "QWK"),
             ("on_off_auroc_drop", summ["on_off"][unit], AUROC_PASS, AUROC_KILL, INFORMATIVE_AUROC, "AUROC points"))
    for line, per_view, pass_line, kill_line, margin, units in specs:
        for vs, q in per_view.items():
            if not q.get("evaluated"):
                out.append(common.GateResult(GATE, f"{line}:{vs}", NAN, None, pass_line, kill_line, "not evaluated",
                                             NAN, rule_sha256, checkpoint_sha256, train_persons_sha256, 0, unit=units,
                                             extra={"reason": q.get("reason", ""), "reading": reading,
                                                    "unit": unit if line.startswith("on_off") else "walk"}))
                continue
            drop = q[key]
            drops = [q[k] for k in READING_DROPS[reading]]
            within = all(d >= -pass_line for d in drops) if all(_finite(d) for d in drops) else None
            informative = (q["informative_margin"] >= margin) if _finite(q["informative_margin"]) else None
            v = common.verdict(drop if _finite(drop) else None, pass_line, kill_line, conditions=(within, informative))
            extra = {k: q[k] for k in ("source", "reestimated", "transfer", "trivial", "source_interval",
                                       "reestimated_interval", "drop", "drop_interval", "drop_transfer",
                                       "drop_transfer_interval", "drop_both", "drop_both_interval", "n_walks",
                                       "imputed_source", "imputed_reestimated", "left_out")}
            extra.update({"reading": reading, "unit": unit if line.startswith("on_off") else "walk",
                          "within_pass_line_both_sides": within, "informative": informative,
                          "informative_margin": q["informative_margin"], "informative_line": margin,
                          "kill_reachable_by_trivial": bool(q["trivial_drop"] > kill_line)
                          if _finite(q["trivial_drop"]) else None})
            out.append(common.GateResult(GATE, f"{line}:{vs}", float(drop), q[f"{key}_interval"], pass_line,
                                         kill_line, v, float(q["trivial_drop"]), rule_sha256, checkpoint_sha256,
                                         train_persons_sha256, int(q["n_persons"]), unit=units, extra=extra))
    return out


def gate_verdict(results):
    """The gate's verdict over its lines (draft reading, rule section 8 item 5): kill when any line kills, pass when
    every line passes, not evaluated when no line is evaluated, undetermined otherwise."""
    v = [r.verdict for r in results]
    if not v or all(x == "not evaluated" for x in v):
        return "not evaluated"
    if "kill" in v:
        return "kill"
    return "pass" if all(x == "pass" for x in v) else "undetermined"


def evaluate(walks, predictor, view_sets=VIEW_SETS, names=DESCRIPTORS, descriptor_fn=None, tracker=None, render=True,
             n_folds=N_FOLDS, seed=SEED, n_boot=BOOT_N, alpha=RIDGE_ALPHA, reading=PROBE_READING, unit=AUROC_UNIT,
             rule_sha256="", checkpoint_sha256="", train_persons_sha256="", stats=None, left_out=None):
    """walk_table, summarize and gate_results in one call: (results, table, summary)."""
    table = walk_table(walks, predictor, view_sets, names, descriptor_fn, tracker, render, stats, left_out)
    summ = summarize(table, view_sets, names, n_folds, seed, n_boot, alpha)
    return gate_results(summ, rule_sha256, checkpoint_sha256, train_persons_sha256, reading, unit), table, summ


# ---------------------------------------------------------------- development path (non-EVAL data only)
def guard_dev_source(source, rule_path=RULE_PATH, registry_path=None, tiers_path=None):
    """The guard's answer for a development source: Refused for CODY, an unlisted or EXCLUDED_V1 source and an EVAL
    source without the registered rule (common.guard_eval_read), and Refused for an EVAL source with it too: the
    development path reads non-EVAL data only."""
    g = common.guard_eval_read(GATE, source, rule_path, registry_path=registry_path or common.RULE_HASHES,
                               tiers_path=tiers_path or common.TIERS)
    if g["eval"]:
        raise Refused(f"{source} is tier EVAL: the G5 development path reads non-EVAL data only (read_care_pd reads "
                      "care_pd under the registered rule)")
    return g


def amass_dev_walks(n_persons=None, max_walks_per_person=None, rate=100.0, check_persons=None):
    """Walks of the AMASS dev body store (task walk), on the body clock (rate Hz): per take the SMPL+H joints of its
    stored body (rot6d, transl, the take's gendered rest), the canonical 15 (camera.SMPL_TO_CANONICAL_15) normalized
    as a care_pd record is (kernels/pose.normalize_3d). Persons sorted, their takes sorted by take id; at most
    n_persons persons and max_walks_per_person takes each. cohort 'amass_dev', no label, no state. The source passes
    guard_dev_source first. FileNotFoundError when the dev body store is not built. check_persons (opt-in, None
    skips it): a callable given the sorted list of the selected person ids once the take index is read and before
    any take loads; it refuses by raising (scripts/gate_g5.py: the lockbox and training-list check)."""
    import torch

    from mdfm.body.smplh import SkeletonBank
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data import body_store
    from mdfm.kernels.camera import SMPL_TO_CANONICAL_15
    guard_dev_source(DEV_SOURCE)
    if not os.path.isdir(os.path.join(body_store.store_root(), "dev")):
        raise FileNotFoundError("dev body store not built (scripts/build_body_store.py)")
    store = body_store.BodyStore(split="dev")
    tk = store.takes
    walks_idx = tk[tk["task_name"] == "walk"]
    by = collections.defaultdict(list)
    for i in walks_idx.sort_values("take_id", kind="stable").index:
        by[str(tk.at[i, "person_id"])].append(i)
    persons = sorted(by)[:n_persons] if n_persons else sorted(by)
    if check_persons is not None:
        check_persons(list(persons))
    f64 = torch.float64
    bank = SkeletonBank(f64)
    out = []
    for p in persons:
        for i in by[p][:max_walks_per_person] if max_walks_per_person else by[p]:
            a = store.take(i)
            r6 = torch.from_numpy(np.asarray(a["rot6d"], np.float64))
            rest = bank.rest(torch.from_numpy(store.betas[i:i + 1]), torch.from_numpy(store.gender[i:i + 1]))[0]
            traj = BodyTrajectory(None, r6, torch.from_numpy(np.asarray(a["transl"], np.float64)), None, rate=rate,
                                  rest=rest)
            with torch.no_grad():
                J22 = traj.pose_at(torch.arange(r6.shape[0], dtype=f64) / rate)[0].numpy()
            coords, scale, valid = pose_kernel.normalize_3d(J22[:, SMPL_TO_CANONICAL_15])
            out.append(Walk(person=p, key=str(tk.at[i, "take_id"]), fps=float(rate), coords=coords, valid=valid,
                            scale=float(scale), cohort="amass_dev"))
    return out


def assign_selftest_labels(table, label_from="cadence_spm", state_from="step_amp_torso"):
    """Known-signal labels for the self-test on non-EVAL walks (they decide nothing): label, the tertile (0, 1, 2) of
    the source body's label_from over the walks; state, per person 'ON' where the source body's state_from is at least
    the person's median, else 'OFF' (NaN values: no label, no state). Returns a copy of the table with the label and
    state of every body set from its walk's source row."""
    t = table.copy()
    src = t[t["body"] == SOURCE_BODY].set_index("walk")
    x = src[label_from].to_numpy(dtype=np.float64)
    ok = np.isfinite(x)
    label = np.full(len(x), np.nan)
    if ok.sum() >= 3:
        cuts = np.quantile(x[ok], [1.0 / 3.0, 2.0 / 3.0])
        label[ok] = np.searchsorted(cuts, x[ok], side="right")
    s = src[state_from].astype(float)
    med = s.groupby(src["person_id"]).transform("median")
    state = np.where(~np.isfinite(s.to_numpy()), "", np.where(s.to_numpy() >= med.to_numpy(), "ON", "OFF"))
    lab = dict(zip(src.index, label))
    st = dict(zip(src.index, state))
    t["label"] = t["walk"].map(lab).astype(float)
    t["state"] = t["walk"].map(st).astype(str)
    return t


def selftest(walks, predictor=identity_predictor, view_sets=VIEW_SETS, names=DESCRIPTORS, render=True, tracker=None,
             n_boot=BOOT_N, seed=SEED, source=DEV_SOURCE, registry_path=None, tiers_path=None, stats=None,
             left_out=None, rule_sha256="", checkpoint_sha256="", train_persons_sha256=""):
    """The development self-test on non-EVAL walks (the spec's identity projections of AMASS dev walks): the source
    passes guard_dev_source, the walks are rendered and re-estimated by predictor, labelled by
    assign_selftest_labels, and scored as the care_pd lines are. stats and left_out as in walk_table; the hashes go
    to the GateResult rows (scripts/gate_g5.py --selftest, dev estimates). Returns (results, table, summary)."""
    guard_dev_source(source, registry_path=registry_path, tiers_path=tiers_path)
    table = assign_selftest_labels(walk_table(walks, predictor, view_sets, names, None, tracker, render, stats,
                                              left_out))
    summ = summarize(table, view_sets, names, N_FOLDS, seed, n_boot)
    return gate_results(summ, rule_sha256, checkpoint_sha256, train_persons_sha256), table, summ


# ---------------------------------------------------------------- the care_pd read (EVAL, registered rule only)
def harness_code_files():
    """The files whose code makes the G5 inputs, descriptors and lines (the read log's code hash)."""
    from mdfm.factory import camera_render
    from mdfm.gates import g4
    from mdfm.kernels import camera as camera_kernel
    return [os.path.abspath(f) for f in (__file__, common.__file__, g4.__file__, camera_render.__file__,
                                         camera_kernel.__file__, pose_kernel.__file__, canonical.__file__)]


def split_subject_ns(subject_ns):
    """(cohort, subject) of a care_pd subject_ns 'care_pd:<cohort>:<subject>'; ValueError otherwise."""
    parts = str(subject_ns).split(":", 2)
    if len(parts) != 3 or parts[0] != SOURCE:
        raise ValueError(f"not a care_pd subject_ns: {subject_ns!r}")
    return parts[1], parts[2]


def walk_state(cohort, key):
    """'ON' or 'OFF' parsed from a BMCLab walk name (the rule of scripts/probe_medication.py), '' otherwise."""
    if cohort != ON_OFF_COHORT:
        return ""
    m = BMCLAB_STATE.search(str(key))
    return m.group(2).upper() if m else ""


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def read_labels(path=LABELS_CSV):
    """{(cohort, subject, walk): UPDRS_GAIT (NaN when empty)} of the label table. Called by read_care_pd only, after
    the logged read."""
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        missing = [c for c in LABEL_COLUMNS if c not in (r.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: columns {missing} missing")
        return {(row["cohort"], row["subject"], row["walk"]): _float(row["UPDRS_GAIT"]) for row in r}


def care_pd_rows():
    """Index rows of care_pd pose records (canonical.index; no record opens). Called by read_care_pd only, after the
    guard."""
    df = canonical.index()
    return df[(df["source"] == SOURCE) & (df["modality"] == "pose")].to_dict("records")


def care_pd_walks(rows, labels, stats):
    """Generator of the care_pd walks with a label or a BMCLab state: each record through canonical.load(row,
    allow_eval=True); a walk with neither is skipped before its record opens, a record without a rate is skipped
    (per-sample fps), each counted in stats."""
    for row in rows:
        cohort, subject = split_subject_ns(row["subject_ns"])
        key = str(row["key"])
        label = labels.get((cohort, subject, key), NAN)
        state = walk_state(cohort, key)
        if not math.isfinite(label) and not state:
            stats["walks_without_label_or_state"] += 1
            continue
        rec = canonical.load(row, allow_eval=True)
        fps = _float(rec.get("fps"))
        if not (math.isfinite(fps) and fps > 0):
            stats["walks_without_fps"] += 1
            continue
        stats["records_loaded"] += 1
        yield walk_from_record(rec, row["person_id"], key, cohort, label, state)


def read_care_pd(predictor, checkpoint_sha256="", train_persons_sha256="", train_persons=None, code_files=(),
                 view_sets=VIEW_SETS, names=DESCRIPTORS, descriptor_fn=None, tracker=None, n_boot=BOOT_N,
                 reading=PROBE_READING, unit=AUROC_UNIT, note="", rule_path=RULE_PATH, registry_path=None,
                 tiers_path=None, log_path=None, labels_path=None):
    """The G5 read on care_pd (module docstring, EVAL read guard). Steps, in order: common.guard_eval_read (Refused
    while the rule is not registered for G5: nothing opens, nothing is written); the index rows of care_pd; with
    train_persons (the checkpoint's lists) the scored persons asserted disjoint from them (g4.assert_disjoint),
    refused before the read is logged; common.log_eval_read with the checkpoint, the index rows' hash and the code
    hash (harness_code_files plus code_files); the label table; the walks (care_pd_walks) through evaluate. registry,
    tiers, log and labels paths default to the tree's. Returns (results, table, summary, info)."""
    reg = registry_path or common.RULE_HASHES
    tiers = tiers_path or common.TIERS
    g = common.guard_eval_read(GATE, SOURCE, rule_path, registry_path=reg, tiers_path=tiers)
    rows = care_pd_rows()
    persons = sorted({str(r["person_id"]) for r in rows})
    disjoint = None
    if train_persons is not None:
        from mdfm.gates import g4
        disjoint = g4.assert_disjoint({SOURCE: persons}, train_persons)
    inputs = common.sha256_json(sorted([str(r["subject_ns"]), str(r["person_id"]), str(r["key"])] for r in rows))
    code = common.code_sha256([*harness_code_files(), *code_files])
    logged = common.log_eval_read(GATE, SOURCE, rule_path, checkpoint_sha256=checkpoint_sha256, inputs_sha256=inputs,
                                  code_sha256=code, note=note, log_path=log_path or common.EVAL_LOG,
                                  registry_path=reg, tiers_path=tiers)
    lpath = labels_path or LABELS_CSV
    labels = read_labels(lpath)
    stats, left = collections.Counter(), []
    results, table, summ = evaluate(care_pd_walks(rows, labels, stats), predictor, view_sets=view_sets, names=names,
                                    descriptor_fn=descriptor_fn, tracker=tracker, render=True, n_boot=n_boot,
                                    reading=reading, unit=unit, rule_sha256=g["rule_sha256"],
                                    checkpoint_sha256=checkpoint_sha256, train_persons_sha256=train_persons_sha256,
                                    stats=stats, left_out=left)
    info = {"eval_read": logged, "index_rows": len(rows), "persons_in_index": len(persons),
            "disjoint_from_training_persons": disjoint, "labels_sha256": common.sha256_file(lpath),
            "stats": dict(stats), "left_out": left, "gate_verdict": gate_verdict(results)}
    return results, table, summ, info


# ---------------------------------------------------------------- outputs
def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if _finite(x) else "n/a"


def markdown(results, summ, info=None, title="G5 pathology preservation"):
    """The report text of a run: the lines, the PA-MPJPE diagnostic, the sets and the walks left out."""
    readings = sorted({str(r.extra.get("reading", "")) for r in results} - {""})
    L = [f"# {title}", "", f"Gate verdict (draft reading, rule section 8 item 5): {gate_verdict(results)}.",
         f"Deciding reading of the drop: {', '.join(readings) or 'n/a'} (rule section 8 item 4).", "",
         "The trivial drop is what a body that carries nothing gives; 'trivial kills' says whether it is above the "
         "kill line (a body that carries noise or inverted information can drop further).", "",
         "| line | point | 95% interval | pass | kill | verdict | trivial drop | trivial kills | source | refit | "
         "transfer | persons |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        iv = r.interval if r.interval is not None else (NAN, NAN)
        kr = r.extra.get("kill_reachable_by_trivial")
        L.append(f"| {r.line} | {_fmt(r.point)} | {_fmt(iv[0])} to {_fmt(iv[1])} | {r.pass_line:g} | {r.kill_line:g} "
                 f"| {r.verdict} | {_fmt(r.trivial_baseline)} | {'n/a' if kr is None else ('yes' if kr else 'no')} | "
                 f"{_fmt(r.extra.get('source'))} | {_fmt(r.extra.get('reestimated'))} | "
                 f"{_fmt(r.extra.get('transfer'))} | {r.n_persons} |")
    L += ["", "## PA-MPJPE against the source fits (diagnostic, no kill)", ""]
    for vs, d in summ["pa_mpjpe"].items():
        iv = d["interval"]
        L.append(f"- {vs}: {_fmt(d['point'], 1)} mm [{_fmt(iv[0], 1)}, {_fmt(iv[1], 1)}], {d['n_persons']} persons, "
                 f"{d['n_walks']} walks.")
    L += ["", "## Sets", "", f"- UPDRS_GAIT: {summ['sets']['updrs_gait']}.",
          f"- ON versus OFF: {summ['sets']['on_off']}.", f"- Bodies left out per body set: {summ.get('left_out', {})}."]
    if info:
        L += [f"- Read counts: {info.get('stats', {})}."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir, results, summ, table, info=None, name="g5", title="G5 pathology preservation"):
    """<out_dir>/<name>.json and .md (common.write) and per_walk.csv. Returns the three paths."""
    j, m = common.write({"results": [r.to_dict() for r in results], "summary": summ, "info": info or {},
                         "gate_verdict": gate_verdict(results)}, out_dir, name, markdown(results, summ, info, title))
    return j, m, common.write_csv(table, os.path.join(out_dir, "per_walk.csv"))
