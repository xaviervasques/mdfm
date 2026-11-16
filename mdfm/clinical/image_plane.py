"""S1 image-plane descriptors from raw 2D keypoints, both rest postures (DRAFT).

Study S1 of docs/MFM_DESIGN_NOTE_CLINICAL_QUESTIONS_2026-09-29.md ("the note"; section 3, priority 4; sections
4.3 and 4.5): fixed formulas on the raw 2D keypoints of modys_video, lying and sitting, no trained model.
Descriptors: TON-T, TON-DW, PHA-P1, FLOW-DII, FLOW-E, FLOW-R of docs/rules/DESCRIPTOR_RULES_DRAFT.md (DR-13,
DR-14) and the proposed FLOW-OCC (note 4.5), lengths in torso lengths.

Status: DRAFT. The descriptor rule file is not hashed (DR-0.1), the image-plane versions of the tonic and
phasic rules are not written as DR rules yet (note 4.3), and no value computed here may be reported (DR-0.5).
Every rule parameter is a keyword argument whose default sits in PARAMS with its DR or OPEN-D id and a status:
- 'draft value': a value written in a DR rule of the unhashed draft;
- 'candidate': a candidate the draft names beside an OPEN item;
- 'open': no value in the draft or the note; the default is None and the behaviour under None is stated;
- 'placeholder': a value of this module so that the code runs; it is never to be cited as a rule value;
- 'module reading': how this module reads a rule written for the 3D body on the image plane.
The module reads no file and writes none: callers pass arrays (keypoints, likelihoods, the record's rate).

Module readings (each one awaits the image-plane DR rules of note 4.3):
- R1 Image plane: the raw keypoint coordinates in pixels, image y down (note 4.3), not the projection of the
  posterior body of DR-2.10. Keypoint set: POINTS, the ordered DeepLabCut set of both modys_video zips
  (reports/modys_structure/summary.md). Pelvis and neck are the hip and shoulder midpoints (MoveAtlas
  configs/canonical_skeleton.yaml, entry modys_dlc).
- R2 Torso length (DR-2.10): the pelvis-to-neck distance; one value per record, the median over the frames where
  both hips and both shoulders are valid, unless the caller passes torso_px.
- R3 Validity: a point is valid in a frame when its coordinates are finite and its likelihood is at least
  likelihood_min (OPEN-D56). Every filter runs over each maximal run of valid samples (DR-2.4: never over a
  window); a run too short for the filter's edge padding is dropped (NaN).
- R4 Filters: DR-2.4 exactly (zero-phase Butterworth, combined -3 dB point at the named frequency through the
  corner correction c = (sqrt(2) - 1)^(1 / (2N)); band-pass = high-pass then low-pass).
- R5 Spectra: DR-2.5 exactly (scipy.signal.welch defaults apart from the segment length, trace over vector
  components, trapezoid band power over the bins with f1 <= f <= f2), computed over the valid runs of a window
  that hold at least one Welch segment, the run spectra averaged with weights equal to their segment counts.
- R6 Segment orientation: the signed angle (deg, clockwise on the image positive) of the segment vector, proximal
  to distal point, against the frame axis: image up for frame 'image' (the image-plane reading of the heading
  frame of DR-14.2), the parent segment's vector for frame 'parent' (the alternative of OPEN-D38). The 0.3 Hz
  low-pass acts on the unit direction vector, renormalized afterwards: the 2D counterpart of the 6D low-pass and
  Gram-Schmidt map of DR-14.2.
- R7 TON-T: theta(t) = |wrap(angle_lp(t) - reference)|, the rotation angle of D(t) of DR-14.2 in the plane; the
  image plane carries one rotation axis, so the per-axis components of DR-14.2 reduce to one signed in-plane
  component (in_plane_deg; DR-7.7: image-plane flexion or abduction, depending on the view). The reference is
  reference_deg (the fixed reference posture per rest protocol of note 4.5, or a within-source norm, OPEN-D57);
  without one TON-T has no value (reason no_reference, flag no_norm). orientation_deg, the circular mean of
  the low-passed direction over the window, is auxiliary and not a readout (DR-1.4).
- R8 TON-DW (DR-14.3): the dwell fraction over the window's valid samples, the dwell time = fraction / 100 x
  window length (DR-14.3 as written, also when some samples are invalid); run rules (OPEN-D39) merge gaps first,
  then drop runs shorter than dwell_min_run_s (a run of exactly that length stays, as DR-5.4 counts no event
  shorter than d_min); the episode rate (per minute of valid samples) and mean episode duration are the closure
  of OPEN-D39 that note 4.5 proposes (flag proposed). Episodes cut by the window edges count as episodes.
- R9 PHA-P1 (DR-14.4): omega(t) = d/dt of the unwrapped image-frame angle (the segment's own angular velocity,
  DR-2.2), its band power over the phasic band. The companion direction-irregularity index of DR-14.4 needs a
  vector angular velocity; the image plane gives one axis, so it is not emitted.
- R10 Velocities (no DR rule for raw keypoints): derivative 'central5', the fourth-order central difference
  inside a run and the second-order scheme of 'central3' on the two samples at each end of the run (interior
  power gain ((8 sin x - sin 2x) / (6 x))^2, x = 2 pi f / fs), or 'central3', second-order differences
  throughout, central inside and one-sided at the end samples, as numpy.gradient with edge_order 2 (power gain
  (sin x / x)^2). Both are written as differences, so that a still point has a velocity of exactly 0.
- R11 Flow (DR-13 on the image plane, DR-13.5): positions in torso lengths; FLOW-DII on the velocity band-passed
  over the flow band (DR-13.2, 2D form 2 (1 - lambda_1)); FLOW-E the band power of the velocity before the
  band-pass (DR-13.3); FLOW-R on the point minus the pelvis, low-passed at roam_lowpass_hz, on the image axes
  (the heading frame of DR-13.4 read as the image axes); FLOW-OCC, the percent of the window's valid samples
  whose band-passed speed (or, with occ_signal 'excursion', band-passed position) exceeds occ_threshold (note
  4.5; no DR rule yet; flag proposed). flow_reference turns a point's motion into its motion relative to another
  point (for example the wrist against the elbow, the joint set that
  docs/proposals/STUDY_S1_MODYS_MEASUREMENT_DRAFT.md proposes for OPEN-D36).
- R12 Regions: the held DIS sheet scores per region the trunk, the proximal arm and the proximal leg, trunk and
  arms on the sitting videos and legs on the lying ones (docs/feedback/CLINICAL_QUESTIONS_CROSSWALK.md, after
  the source README; OPEN-D53). Every row carries the body region of its target and dis_item, the item that
  scores the target directly: trunk (trunk segment, neck point), proximal_arm (upper arm, elbow), proximal_leg
  (thigh, knee), None for the forearm, shank, wrist, ankle and head, which no item of the sheet scores (a point
  carries the item of the segment whose distal end it is). scored_in_posture is True or False for a target with
  a dis_item (its region's posture in region_posture, None when that is None) and False for a target without
  one; reading a distal target against its region's proximal item is an analysis choice (OPEN-D53), not a DIS
  fact. Suffixes 1 and 2 are near and far, not left and right (MoveAtlas config; laterality unknown in every
  training record, reports/modys_structure/summary.md): every sided value carries laterality_unknown, and
  pool_sides gives the side-pooled pairs of note 4.5.
- R13 Windows (DR-2.7): window_s windows every hop_s; sample i lies at i / fs, and a window of samples a to b - 1
  is emitted at t_start_s = a / fs and at its centre time t_center_s = (a + b - 1) / (2 fs), the midpoint of its
  first and last sample times; a record shorter than one window gives no row. The two non-overlapping 10 s
  halves of a 20 s clip (S1 output 2) are window_bounds(n, fs, 10.0, 10.0); half_windows(n) gives two halves of
  floor(n / 2) samples for any span.
- R15 Reasons: a row has a value, or NaN and a reason, never both and never neither. Precedence: the record or
  call refusals (lowpass_above_nyquist, band_above_f_max, no_torso); then the window's validity
  (no_valid_samples when no sample is valid, valid_fraction_below_min below min_valid_fraction, DR-2.7); then
  the caller's missing inputs (no_reference, no_threshold, no_norm, norm_not_positive); then the computation
  (no_welch_segment, band_too_few_bins, zero_power for a dB value of a zero power, too_few_samples and no_motion
  for FLOW-DII, no_episode); not_finite is a guard that no path reaches.
- R14 Bandwidth (DR-5.1, DR-5.3, DR-4.8): f_max = min(k fs, f_lp); a band quantity is refused when its upper edge
  exceeds f_max. With k open (None) bands are checked against the Nyquist frequency only, f_max is reported as
  None and every band quantity carries bandwidth_k_open.
"""
import math
from collections import namedtuple
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

RULE = "DRAFT: docs/rules/DESCRIPTOR_RULES_DRAFT.md not hashed (DR-0.1, DR-0.5)"

POSTURES = ("lying", "sitting")
TASKS = MappingProxyType({"lying": "rest_lying", "sitting": "rest_sitting"})   # note 4.2 (names proposed there)

# reports/modys_structure/summary.md: one ordered keypoint set across both zips
POINTS = ("ankle1", "knee1", "hip1", "hip2", "knee2", "ankle2", "wrist1", "elbow1", "shoulder1", "shoulder2",
          "elbow2", "wrist2", "chin", "forehead")
DERIVED = MappingProxyType({"pelvis": ("hip1", "hip2"), "neck": ("shoulder1", "shoulder2")})

# dis_item (R12): the item of the held DIS sheet that scores the target directly (trunk, proximal arm, proximal
# leg); None for the distal segments and points and the head, which no item of that sheet scores.
DIS_ITEMS = ("trunk", "proximal_arm", "proximal_leg")
Segment = namedtuple("Segment", "proximal distal parent region base side dis_item")
SEGMENTS = MappingProxyType({
    "trunk": Segment("pelvis", "neck", None, "trunk", "trunk", None, "trunk"),
    "head": Segment("neck", "forehead", "trunk", "head", "head", None, None),
    "upper_arm_1": Segment("shoulder1", "elbow1", "trunk", "arms", "upper_arm", "1", "proximal_arm"),
    "upper_arm_2": Segment("shoulder2", "elbow2", "trunk", "arms", "upper_arm", "2", "proximal_arm"),
    "forearm_1": Segment("elbow1", "wrist1", "upper_arm_1", "arms", "forearm", "1", None),
    "forearm_2": Segment("elbow2", "wrist2", "upper_arm_2", "arms", "forearm", "2", None),
    "thigh_1": Segment("hip1", "knee1", "trunk", "legs", "thigh", "1", "proximal_leg"),
    "thigh_2": Segment("hip2", "knee2", "trunk", "legs", "thigh", "2", "proximal_leg"),
    "shank_1": Segment("knee1", "ankle1", "thigh_1", "legs", "shank", "1", None),
    "shank_2": Segment("knee2", "ankle2", "thigh_2", "legs", "shank", "2", None),
})
# a point carries the DIS item of the segment whose distal end it is (R12 reading)
FlowPoint = namedtuple("FlowPoint", "region base side dis_item")
FLOW_POINTS = MappingProxyType({
    "wrist1": FlowPoint("arms", "wrist", "1", None), "wrist2": FlowPoint("arms", "wrist", "2", None),
    "ankle1": FlowPoint("legs", "ankle", "1", None), "ankle2": FlowPoint("legs", "ankle", "2", None),
    "forehead": FlowPoint("head", "head", None, None),
    "elbow1": FlowPoint("arms", "elbow", "1", "proximal_arm"),
    "elbow2": FlowPoint("arms", "elbow", "2", "proximal_arm"),
    "knee1": FlowPoint("legs", "knee", "1", "proximal_leg"), "knee2": FlowPoint("legs", "knee", "2", "proximal_leg"),
    "neck": FlowPoint("trunk", "neck", None, "trunk"),
})

Param = namedtuple("Param", "default ids status note")
PARAMS = MappingProxyType({
    "likelihood_min": Param(0.5, "OPEN-D56", "placeholder",
                            "a point is valid at or above this DeepLabCut likelihood; 0.5 is the level the T10 "
                            "probe printed (reports/modys_structure/summary.md), not a candidate of the draft; "
                            "None applies no likelihood screen"),
    "min_valid_fraction": Param(None, "OPEN-D59", "open",
                                "None applies no minimum; valid_fraction is printed beside every value (DR-4.5)"),
    "window_s": Param(10.0, "DR-2.7", "draft value", "window length"),
    "hop_s": Param(5.0, "OPEN-D05", "candidate", "window hop"),
    "filter_order": Param(2, "OPEN-D07", "placeholder",
                          "Butterworth order N of every filter; 2 is the order of the worked example of DR-2.4"),
    "filter_padtype": Param("odd", "OPEN-D07", "placeholder",
                            "edge padding at the ends of a valid run; the scipy.signal.sosfiltfilt default"),
    "welch_seg_s": Param(4.0, "OPEN-D06", "candidate",
                         "Welch segment length for bands whose lower edge is below 1 Hz"),
    "k": Param(None, "OPEN-D02", "open",
               "bandwidth factor of DR-5.1; None checks bands against the Nyquist frequency only and flags "
               "bandwidth_k_open"),
    "tonic_lowpass_hz": Param(0.3, "DR-14.2", "draft value", "tonic low-pass"),
    "frame": Param("image", "DR-14.2, OPEN-D38", "module reading",
                   "'image': against image up (R6); 'parent': against the parent segment"),
    "reference_deg": Param(None, "OPEN-D57", "open",
                           "reference orientation per segment in the frame (a number, or a mapping segment to "
                           "number): the fixed reference posture of note 4.5 or a within-source norm; None: "
                           "TON-T and TON-DW have no value (no_norm)"),
    "dwell_threshold_deg": Param(None, "DR-14.3, OPEN-D57", "open",
                                 "theta_95 of DR-14.3 (a number, or a mapping segment to number); None: TON-DW "
                                 "has no value (no_norm)"),
    "dwell_min_run_s": Param(0.0, "OPEN-D39", "open", "0: no minimum run, DR-14.3 as written"),
    "dwell_merge_gap_s": Param(0.0, "OPEN-D39", "open", "0: no merging of runs"),
    "phasic_band_hz": Param((0.5, 3.0), "DR-14.4", "draft value", "PHA-P1 band"),
    "flow_band_hz": Param((0.5, 3.0), "DR-13.1", "draft value", "flow band"),
    "roam_lowpass_hz": Param(3.0, "DR-13.4, OPEN-D37", "candidate", "FLOW-R low-pass"),
    "segments": Param(tuple(SEGMENTS), "DR-14.1, OPEN-D53", "module reading",
                      "segments read for TON-T, TON-DW and PHA-P1: every image-plane segment of POINTS; on "
                      "modys_video the head segment (neck to forehead) has no valid sample at likelihood_min 0.5, "
                      "since forehead and chin never reach 0.5 in a training frame (reports/modys_structure/"
                      "summary.md), so its rows carry no_valid_samples; dropping the head or naming another proxy "
                      "is the owner's decision"),
    "flow_points": Param(("wrist1", "wrist2", "ankle1", "ankle2", "forehead"), "DR-13.1, OPEN-D36", "candidate",
                         "wrists, ankles and head; the head is the forehead point (any key of FLOW_POINTS may "
                         "be named); on modys_video the forehead has no valid sample at likelihood_min 0.5 "
                         "(reports/modys_structure/summary.md), so the DR-13.1 head candidate has no usable "
                         "keypoint there and its rows carry no_valid_samples; dropping it or naming another proxy "
                         "is the owner's decision"),
    "flow_reference": Param(None, "DR-13.1, OPEN-D36", "draft reading",
                            "None: each point's own image-plane velocity (DR-13.1); a mapping point to reference "
                            "point: the motion of the point relative to it (for FLOW-DII, FLOW-E and FLOW-OCC; "
                            "FLOW-R stays relative to the pelvis, DR-13.4)"),
    "occ_signal": Param("speed", "OPEN-D37", "module reading",
                        "FLOW-OCC signal: 'speed', the band-passed speed (torso lengths per second); 'excursion', "
                        "the norm of the band-passed position (torso lengths)"),
    "occ_threshold": Param(None, "OPEN-D37", "open",
                           "FLOW-OCC speed threshold in torso lengths per second (note 4.5; a number, or a "
                           "mapping point to number); None: FLOW-OCC has no value"),
    "norm_median": Param(None, "DR-2.6, OPEN-D57", "open",
                         "norm median of the raw band power, a mapping keyed (descriptor, target) or "
                         "descriptor; None: no dB re norm value (no_norm)"),
    "torso_px": Param(None, "DR-2.10", "module reading", "None: the record median of R2"),
    "derivative": Param("central5", "DR-2.2", "module reading",
                        "velocity estimator on sampled keypoints (R10); the posterior's velocity of DR-2.2 is "
                        "analytic, raw keypoints have none"),
    "region_posture": Param(MappingProxyType({"trunk": "sitting", "arms": "sitting", "legs": "lying",
                                              "head": None}), "OPEN-D53", "draft reading",
                            "posture in which DIS scores each region (R12); None: not stated in the sources read; "
                            "it sets scored_in_posture only for a target with a dis_item (trunk, proximal arm, "
                            "proximal leg); reading a distal target against its region's proximal item is an "
                            "OPEN-D53 choice of the analysis, not a DIS fact"),
    "pool": Param("mean", "OPEN-D53", "placeholder", "side pooling of the near and far values: 'mean' or 'max'"),
})
STATUSES = ("draft value", "candidate", "open", "placeholder", "module reading", "draft reading")

# keyword arguments that are not rule parameters: record facts and call options
NON_RULE_ARGS = ("likelihood", "names", "record_flags", "f_lp_hz", "windows")

ROW_KEYS = ("window", "t_start_s", "t_center_s", "posture", "task", "descriptor", "quantity", "target",
            "target_base", "side", "region", "dis_item", "scored_in_posture", "value", "unit", "reason", "flags",
            "valid_fraction", "band_hz", "f_max_hz", "resolution_s", "rule")

# reasons that refuse a quantity for the whole record or call (R14, R2, the tonic low-pass): they stand before the
# validity reasons of a window; every other reason yields to no_valid_samples and valid_fraction_below_min
_RECORD_REASONS = ("lowpass_above_nyquist", "band_above_f_max", "no_torso")

_EPS_HZ = 1e-9
_MOTION_EPS = 1e-12     # numerical zero of an RMS speed (input unit), not a rule value: below it DII is NaN
_UP = np.array([0.0, -1.0])


def _d(name):
    return PARAMS[name].default


def param_table():
    """[{name, default, ids, status, note}] of PARAMS, in order."""
    return [{"name": n, "default": dict(p.default) if isinstance(p.default, Mapping) else p.default,
             "ids": p.ids, "status": p.status, "note": p.note} for n, p in PARAMS.items()]


# ---------------------------------------------------------------- points, torso, windows
def points(xy, likelihood=None, names=POINTS, likelihood_min=_d("likelihood_min")):
    """{name: (T, 2) px} of keypoints xy (T, P, 2) in the order names, NaN where a point is not valid (R3), plus
    the derived pelvis and neck (R1) when their points are present."""
    xy = np.asarray(xy, dtype=np.float64)
    names = tuple(names)
    if xy.ndim != 3 or xy.shape[2] != 2 or xy.shape[1] != len(names):
        raise ValueError(f"xy of shape {xy.shape}, expected (T, {len(names)}, 2)")
    valid = np.isfinite(xy).all(axis=-1)
    if likelihood is not None and likelihood_min is not None:
        lk = np.asarray(likelihood, dtype=np.float64)
        if lk.shape != xy.shape[:2]:
            raise ValueError(f"likelihood of shape {lk.shape}, expected {xy.shape[:2]}")
        valid &= np.nan_to_num(lk, nan=-np.inf) >= likelihood_min
    out = {n: np.where(valid[:, i, None], xy[:, i], np.nan) for i, n in enumerate(names)}
    for name, (a, b) in DERIVED.items():
        if a in out and b in out:
            out[name] = (out[a] + out[b]) / 2.0
    return out


def torso_length(pts):
    """Median pelvis-to-neck distance (px) over the frames where both are valid (R2); NaN when none."""
    d = np.linalg.norm(pts["neck"] - pts["pelvis"], axis=-1)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else math.nan


def window_bounds(n, fs, window_s=_d("window_s"), hop_s=_d("hop_s")):
    """[(start, stop)] sample bounds of the windows of a record of n samples at fs Hz (R13)."""
    w, h = int(round(window_s * fs)), int(round(hop_s * fs))
    if w < 1 or h < 1:
        raise ValueError(f"window of {w} and hop of {h} samples")
    return [(s, s + w) for s in range(0, int(n) - w + 1, h)]


def half_windows(n):
    """[(0, n // 2), (n // 2, 2 (n // 2))]: two non-overlapping halves of floor(n / 2) samples, for records whose
    span is not a whole number of windows (pass as windows= to descriptors)."""
    h = int(n) // 2
    return [(0, h), (h, 2 * h)] if h else []


def runs(mask):
    """[(start, stop)] of the maximal runs of True in a boolean vector."""
    m = np.concatenate([[0], np.asarray(mask, dtype=bool).astype(np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _finite_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=tuple(range(1, x.ndim)))


# ---------------------------------------------------------------- filters, derivatives, spectra
def corner_hz(f_hz, fs, kind, filter_order=_d("filter_order")):
    """Single-pass design corner (Hz) that puts the -3 dB point of the forward-backward response at f_hz (DR-2.4):
    tan(pi fc / fs) = tan(pi f / fs) / c for kind 'low', x c for kind 'high', c = (sqrt(2) - 1)^(1 / (2N))."""
    if kind not in ("low", "high"):
        raise ValueError(f"kind {kind!r}")
    if not 0.0 < f_hz < fs / 2.0:
        raise ValueError(f"filter frequency {f_hz} Hz outside (0, {fs / 2.0}) at fs {fs} Hz")
    c = (math.sqrt(2.0) - 1.0) ** (1.0 / (2 * filter_order))
    t = math.tan(math.pi * f_hz / fs)
    t = t / c if kind == "low" else t * c
    return fs / math.pi * math.atan(t)


def zero_phase(x, fs, f_hz, kind, filter_order=_d("filter_order"), filter_padtype=_d("filter_padtype")):
    """DR-2.4 filter of x (T,) or (T, C) along axis 0, run by run over the valid samples (R3); samples of a run too
    short for the edge padding stay NaN."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    sos = butter(filter_order, corner_hz(f_hz, fs, kind, filter_order),
                 btype="lowpass" if kind == "low" else "highpass",
                 fs=fs, output="sos")
    for a, b in runs(_finite_rows(x)):
        try:
            out[a:b] = sosfiltfilt(sos, x[a:b], axis=0, padtype=filter_padtype)
        except ValueError:
            pass
    return out


def bandpass(x, fs, band, filter_order=_d("filter_order"), filter_padtype=_d("filter_padtype")):
    """DR-2.4 band-pass: high-pass at band[0], then low-pass at band[1]."""
    hp = zero_phase(x, fs, band[0], "high", filter_order, filter_padtype)
    return zero_phase(hp, fs, band[1], "low", filter_order, filter_padtype)


def time_derivative(x, fs, derivative=_d("derivative")):
    """Time derivative (unit per s) of x (T,) or (T, C), run by run over the valid samples (R10); runs shorter
    than 3 samples stay NaN."""
    if derivative not in ("central5", "central3"):
        raise ValueError(f"derivative {derivative!r}")
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    for a, b in runs(_finite_rows(x)):
        seg = x[a:b]
        if b - a < 3:
            continue
        d = np.empty_like(seg)
        d[1:-1] = (seg[2:] - seg[:-2]) / 2.0
        d[0] = (4.0 * (seg[1] - seg[0]) - (seg[2] - seg[0])) / 2.0
        d[-1] = (4.0 * (seg[-1] - seg[-2]) - (seg[-1] - seg[-3])) / 2.0
        if derivative == "central5" and b - a >= 5:
            d[2:-2] = (8.0 * (seg[3:-1] - seg[1:-3]) - (seg[4:] - seg[:-4])) / 12.0   # exact 0 on a constant
        out[a:b] = d * fs
    return out


def derivative_gain(f_hz, fs, derivative=_d("derivative")):
    """Amplitude gain of the interior derivative estimator at f_hz against the exact derivative."""
    x = 2.0 * math.pi * f_hz / fs
    if x == 0.0:
        return 1.0
    return (8.0 * math.sin(x) - math.sin(2.0 * x)) / (6.0 * x) if derivative == "central5" else math.sin(x) / x


def welch_psd(x, fs, welch_seg_s=_d("welch_seg_s")):
    """(f, psd, n_segments) of x (T,) or (T, C) (R5; trace over C), or (None, None, 0) when no valid run holds a
    segment of welch_seg_s."""
    from scipy.signal import welch
    x = np.asarray(x, dtype=np.float64)
    x2 = x[:, None] if x.ndim == 1 else x.reshape(len(x), -1)
    nper = int(round(welch_seg_s * fs))
    nover = nper // 2
    step = nper - nover
    total, count, f = None, 0, None
    for a, b in runs(_finite_rows(x2)):
        if b - a < nper:
            continue
        f, p = welch(x2[a:b].T, fs=fs, window="hann", nperseg=nper, noverlap=nover, detrend="constant",
                     scaling="density", return_onesided=True, axis=-1)
        m = (b - a - nper) // step + 1
        p = p.sum(axis=0) * m
        total = p if total is None else total + p
        count += m
    if not count:
        return None, None, 0
    return f, total / count, count


def band_power(f, psd, band):
    """Trapezoid integral of psd over the bins with band[0] <= f <= band[1] (DR-2.5)."""
    sel = (f >= band[0] - _EPS_HZ) & (f <= band[1] + _EPS_HZ)
    return float(np.trapezoid(psd[sel], f[sel])) if sel.sum() >= 2 else math.nan


def dii(v):
    """Direction-irregularity index of velocities v (N, p) (DR-13.2): p / (p - 1) x (1 - lambda_1) of the
    squared-speed-weighted orientation tensor; rows with a NaN are left out; NaN without motion (RMS speed at or
    below the numerical zero _MOTION_EPS)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v).all(axis=1)]
    s = float((v ** 2).sum())
    if len(v) < 2 or not math.sqrt(s / len(v)) > _MOTION_EPS:
        return math.nan
    lam1 = float(np.linalg.eigvalsh(v.T @ v / s)[-1])
    p = v.shape[1]
    return float(min(1.0, max(0.0, p / (p - 1.0) * (1.0 - lam1))))


# ---------------------------------------------------------------- angles and dwell
def wrap_deg(a):
    """Angles wrapped to [-180, 180) deg."""
    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _signed_angle(axis, u):
    cross = axis[..., 0] * u[..., 1] - axis[..., 1] * u[..., 0]
    dot = (axis * u).sum(axis=-1)
    ok = (np.linalg.norm(axis, axis=-1) > 0) & (np.linalg.norm(u, axis=-1) > 0)
    return np.where(ok, np.degrees(np.arctan2(cross, dot)), np.nan)


def segment_angle(pts, segment, frame=_d("frame")):
    """(T,) signed angle (deg, clockwise on the image positive) of a segment of SEGMENTS against the frame axis
    (R6); NaN where a point is not valid."""
    if frame not in ("image", "parent"):
        raise ValueError(f"frame {frame!r}")
    s = SEGMENTS[segment]
    u = pts[s.distal] - pts[s.proximal]
    if frame == "image" or s.parent is None:
        axis = np.broadcast_to(_UP, u.shape)
    else:
        p = SEGMENTS[s.parent]
        axis = pts[p.distal] - pts[p.proximal]
    return _signed_angle(axis, u)


def lowpass_direction(angle_deg, fs, tonic_lowpass_hz=_d("tonic_lowpass_hz"), filter_order=_d("filter_order"),
                      filter_padtype=_d("filter_padtype")):
    """Angle (deg) of the low-passed unit direction vector (R6)."""
    r = np.radians(np.asarray(angle_deg, dtype=np.float64))
    e = zero_phase(np.stack([np.cos(r), np.sin(r)], axis=1), fs, tonic_lowpass_hz, "low", filter_order,
                   filter_padtype)
    n = np.linalg.norm(e, axis=1)
    return np.where(n > 1e-6, np.degrees(np.arctan2(e[:, 1], e[:, 0])), np.nan)


def unwrap_deg(angle_deg):
    """Angle (deg) unwrapped run by run over the valid samples."""
    a = np.asarray(angle_deg, dtype=np.float64)
    out = np.full_like(a, np.nan)
    for s, e in runs(np.isfinite(a)):
        out[s:e] = np.degrees(np.unwrap(np.radians(a[s:e])))
    return out


def dwell(theta, threshold, fs, dwell_min_run_s=_d("dwell_min_run_s"), dwell_merge_gap_s=_d("dwell_merge_gap_s")):
    """TON-DW of a deviation series theta (deg) against threshold (R8): {fraction_pct, time_s, episodes,
    episode_rate_per_min, episode_mean_s}, or None without a valid sample. Fraction and time follow DR-14.3 over
    the valid samples; a merged gap counts as dwelling on its valid samples."""
    theta = np.asarray(theta, dtype=np.float64)
    valid = np.isfinite(theta)
    nv = int(valid.sum())
    if not nv:
        return None
    on = valid & (np.where(valid, theta, -np.inf) > threshold)
    r = runs(on)
    if dwell_merge_gap_s > 0:
        for (_, b1), (a2, _) in zip(r[:-1], r[1:]):
            if (a2 - b1) / fs <= dwell_merge_gap_s:
                on[b1:a2] = True
        r = runs(on)
    if dwell_min_run_s > 0:
        for a, b in r:
            if (b - a) / fs < dwell_min_run_s:
                on[a:b] = False
        r = runs(on)
    frac = 100.0 * float((on & valid).sum()) / nv
    lengths = [(b - a) / fs for a, b in r]
    return {"fraction_pct": frac, "time_s": frac / 100.0 * len(theta) / fs, "episodes": len(r),
            "episode_rate_per_min": len(r) / (nv / fs / 60.0),
            "episode_mean_s": float(np.mean(lengths)) if lengths else math.nan}


# ---------------------------------------------------------------- bandwidth and lookups
def f_max_hz(fs, k=_d("k"), f_lp_hz=None):
    """(f_max or None, admission limit) of DR-5.1: f_max = min(k fs, f_lp); with k None, f_max is None and the
    admission limit is min(fs / 2, f_lp) (R14)."""
    if k is None:
        lim = fs / 2.0 if f_lp_hz is None else min(fs / 2.0, float(f_lp_hz))
        return None, lim
    fm = k * fs if f_lp_hz is None else min(k * fs, float(f_lp_hz))
    return fm, fm


def _lookup(value, key):
    if value is None:
        return None
    if isinstance(value, Mapping):
        v = value.get(key)
        return None if v is None else float(v)
    return float(value)


def _norm(norm_median, descriptor, target):
    if norm_median is None:
        return None
    v = norm_median.get((descriptor, target), norm_median.get(descriptor))
    return None if v is None else float(v)


def db_re_norm(p, nm, reason):
    """(value, reason) of DR-2.6, 10 log10(p / nm), for a raw band power p that carries reason: the reason passes
    through; then no_norm without a norm, norm_not_positive for a norm at or below 0, zero_power for a power at or
    below 0 (a still target)."""
    if reason is not None:
        return math.nan, reason
    if nm is None:
        return math.nan, "no_norm"
    if not nm > 0:
        return math.nan, "norm_not_positive"
    if not p > 0:
        return math.nan, "zero_power"
    return 10.0 * math.log10(p / nm), None


def _band_power_reason(f, psd, nseg, band):
    """(band power, reason) of a Welch result: no_welch_segment without a segment, band_too_few_bins when fewer
    than two bins fall in the band."""
    if not nseg:
        return math.nan, "no_welch_segment"
    p = band_power(f, psd, band)
    return (p, None) if math.isfinite(p) else (math.nan, "band_too_few_bins")


# ---------------------------------------------------------------- the descriptors of one record
def descriptors(xy, fs, posture, likelihood=None, names=POINTS, record_flags=(), f_lp_hz=None, windows=None, *,
                likelihood_min=_d("likelihood_min"), min_valid_fraction=_d("min_valid_fraction"),
                window_s=_d("window_s"), hop_s=_d("hop_s"), filter_order=_d("filter_order"),
                filter_padtype=_d("filter_padtype"), welch_seg_s=_d("welch_seg_s"), k=_d("k"),
                tonic_lowpass_hz=_d("tonic_lowpass_hz"), frame=_d("frame"), reference_deg=_d("reference_deg"),
                dwell_threshold_deg=_d("dwell_threshold_deg"), dwell_min_run_s=_d("dwell_min_run_s"),
                dwell_merge_gap_s=_d("dwell_merge_gap_s"), phasic_band_hz=_d("phasic_band_hz"),
                flow_band_hz=_d("flow_band_hz"), roam_lowpass_hz=_d("roam_lowpass_hz"), segments=_d("segments"),
                flow_points=_d("flow_points"), occ_threshold=_d("occ_threshold"), norm_median=_d("norm_median"),
                torso_px=_d("torso_px"), derivative=_d("derivative"), region_posture=_d("region_posture"),
                flow_reference=_d("flow_reference"), occ_signal=_d("occ_signal")):
    """Rows (dicts with ROW_KEYS) of the S1 image-plane descriptors of one record: keypoints xy (T, P, 2) px in the
    order names, likelihoods (T, P) or None, the record's own rate fs (Hz), its posture ('lying' or 'sitting',
    OPEN-D53: the caller states it), the record's flags (record_flags, appended to every row) and the device
    low-pass f_lp_hz (DR-5.1) when the nuisance table holds one; windows, a list of (start, stop) sample bounds,
    replaces the window_s and hop_s tiling (for example half_windows). One row per window, target and quantity:
    TON-T (deviation_deg, in_plane_deg, orientation_deg), TON-DW (fraction_pct, time_s, episode_rate_per_min,
    episode_mean_s), PHA-P1 (power, db_re_norm) per segment; FLOW-DII (dii), FLOW-E (power, db_re_norm),
    FLOW-R (roaming), FLOW-OCC (occupancy_pct) per flow point. A row without a value carries NaN and a reason, a
    row with a reason carries NaN (R15)."""
    if posture not in POSTURES:
        raise ValueError(f"posture {posture!r}; expected one of {POSTURES}")
    fs = float(fs)
    if not (math.isfinite(fs) and fs > 0):
        raise ValueError(f"rate {fs}")
    pts = points(xy, likelihood, names, likelihood_min)
    need = {p for s in segments for p in (SEGMENTS[s].proximal, SEGMENTS[s].distal)}
    need |= {p for s in segments if SEGMENTS[s].parent and frame == "parent"
             for p in (SEGMENTS[SEGMENTS[s].parent].proximal, SEGMENTS[SEGMENTS[s].parent].distal)}
    if occ_signal not in ("speed", "excursion"):
        raise ValueError(f"occ_signal {occ_signal!r}")
    unknown = sorted(set(flow_points) - set(FLOW_POINTS))
    if unknown:
        raise ValueError(f"flow points without a FLOW_POINTS entry: {unknown}")
    need |= set(flow_points) | {"pelvis"} | {flow_reference[p] for p in flow_points
                                             if flow_reference is not None and flow_reference.get(p)}
    missing = sorted(n for n in need if n not in pts)
    if missing:
        raise ValueError(f"points missing from names: {missing}")
    T = len(pts["pelvis"])
    wins = window_bounds(T, fs, window_s, hop_s) if windows is None else [(int(a), int(b)) for a, b in windows]
    if any(not 0 <= a < b <= T for a, b in wins):
        raise ValueError(f"windows outside the record of {T} samples: {wins}")
    if not wins:
        return []
    fmax, admit = f_max_hz(fs, k, f_lp_hz)
    band_flags = ("bandwidth_k_open",) if k is None else ()
    base = ("image_plane",) + tuple(record_flags)
    rows = []

    def emit(w, desc, qty, target, info, value, unit, reason=None, flags=(), vf=None, band=None):
        a, b = wins[w]
        rp = region_posture.get(info.region)
        side_flags = ("laterality_unknown",) if info.side else ()
        if reason not in _RECORD_REASONS:
            if vf == 0.0:
                reason = "no_valid_samples"
            elif min_valid_fraction is not None and vf is not None and vf < min_valid_fraction:
                reason = "valid_fraction_below_min"
        value = math.nan if value is None else float(value)
        if reason is None and not math.isfinite(value):
            reason = "not_finite"            # a guard: every path above names its own reason
        if reason is not None:
            value = math.nan
        rows.append({
            "window": w, "t_start_s": a / fs, "t_center_s": (a + b - 1) / 2.0 / fs, "posture": posture,
            "task": TASKS[posture], "descriptor": desc, "quantity": qty, "target": target,
            "target_base": info.base, "side": info.side, "region": info.region, "dis_item": info.dis_item,
            "scored_in_posture": False if info.dis_item is None else (None if rp is None else rp == posture),
            "value": value, "unit": unit, "reason": reason,
            "flags": base + side_flags + tuple(flags), "valid_fraction": vf,
            "band_hz": None if band is None else tuple(float(x) for x in band),
            "f_max_hz": fmax, "resolution_s": 1.0 / fs, "rule": RULE})

    def vfrac(x, a, b):
        return float(_finite_rows(x[a:b]).mean())

    # tonic and phasic, per segment
    tonic_ok = tonic_lowpass_hz < fs / 2.0
    pha_ok = phasic_band_hz[1] <= admit + _EPS_HZ and phasic_band_hz[1] < fs / 2.0
    for seg in segments:
        info = SEGMENTS[seg]
        ang_img = segment_angle(pts, seg, "image")
        ang = ang_img if frame == "image" else segment_angle(pts, seg, "parent")
        ang_lp = lowpass_direction(ang, fs, tonic_lowpass_hz, filter_order, filter_padtype) if tonic_ok \
            else np.full(T, np.nan)
        ref = _lookup(reference_deg, seg)
        thr = _lookup(dwell_threshold_deg, seg)
        signed = wrap_deg(ang_lp - ref) if ref is not None else None
        omega = time_derivative(unwrap_deg(ang_img), fs, derivative)
        nm = _norm(norm_median, "PHA-P1", seg)
        no_norm_flag = ("no_norm",) if nm is None else ()
        tband = (0.0, tonic_lowpass_hz)
        for w, (a, b) in enumerate(wins):
            vf = vfrac(ang_lp, a, b)
            sl = ang_lp[a:b]
            fin = np.isfinite(sl)
            reason = None if tonic_ok else "lowpass_above_nyquist"
            # orientation (auxiliary)
            if reason is None and fin.any():
                r = np.radians(sl[fin])
                orient = math.degrees(math.atan2(float(np.sin(r).mean()), float(np.cos(r).mean())))
            else:
                orient = math.nan
            emit(w, "TON-T", "orientation_deg", seg, info, orient, "deg", reason, ("auxiliary",), vf, tband)
            if reason is None and signed is None:
                for q in ("deviation_deg", "in_plane_deg"):
                    emit(w, "TON-T", q, seg, info, math.nan, "deg", "no_reference", ("no_norm",), vf, tband)
            else:
                ss = None if signed is None else signed[a:b][fin]
                dev = float(np.median(np.abs(ss))) if reason is None and ss is not None and ss.size else math.nan
                inp = float(np.median(ss)) if reason is None and ss is not None and ss.size else math.nan
                emit(w, "TON-T", "deviation_deg", seg, info, dev, "deg", reason, (), vf, tband)
                emit(w, "TON-T", "in_plane_deg", seg, info, inp, "deg", reason, (), vf, tband)
            dw_q = (("fraction_pct", "percent", ()), ("time_s", "s", ()),
                    ("episode_rate_per_min", "per_min", ("proposed",)), ("episode_mean_s", "s", ("proposed",)))
            if reason is None and (signed is None or thr is None):
                rsn = "no_reference" if signed is None else "no_threshold"
                for q, u, fl in dw_q:
                    emit(w, "TON-DW", q, seg, info, math.nan, u, rsn, fl + ("no_norm",), vf, tband)
            else:
                dres = dwell(np.abs(signed[a:b]), thr, fs, dwell_min_run_s, dwell_merge_gap_s) \
                    if reason is None else None
                for q, u, fl in dw_q:
                    val = math.nan if dres is None else dres[q]
                    rsn = reason
                    if rsn is None and q == "episode_mean_s" and dres is not None and dres["episodes"] == 0:
                        rsn = "no_episode"
                    emit(w, "TON-DW", q, seg, info, val, u, rsn, fl, vf, tband)
            # phasic
            vfo = vfrac(omega, a, b)
            if not pha_ok:
                p, rsn = math.nan, "band_above_f_max"
            else:
                p, rsn = _band_power_reason(*welch_psd(omega[a:b], fs, welch_seg_s), phasic_band_hz)
            emit(w, "PHA-P1", "power", seg, info, p, "deg2/s2", rsn, band_flags, vfo, phasic_band_hz)
            db, db_rsn = db_re_norm(p, nm, rsn)
            emit(w, "PHA-P1", "db_re_norm", seg, info, db, "dB", db_rsn, band_flags + no_norm_flag, vfo,
                 phasic_band_hz)

    # flow, per point
    torso = float(torso_px) if torso_px is not None else torso_length(pts)
    flow_ok = flow_band_hz[1] <= admit + _EPS_HZ and flow_band_hz[1] < fs / 2.0
    roam_ok = roam_lowpass_hz < fs / 2.0
    for pt in flow_points:
        info = FLOW_POINTS[pt]
        ref_pt = flow_reference.get(pt) if flow_reference is not None else None
        rel_flags = (f"relative_to_{ref_pt}",) if ref_pt else ()
        if math.isfinite(torso) and torso > 0:
            pos = (pts[pt] - pts[ref_pt] if ref_pt else pts[pt]) / torso
            rel = (pts[pt] - pts["pelvis"]) / torso
            v = time_derivative(pos, fs, derivative)
            v_bp = bandpass(v, fs, flow_band_hz, filter_order, filter_padtype) if flow_ok else np.full_like(v, np.nan)
            occ_sig = v_bp if occ_signal == "speed" else (
                bandpass(pos, fs, flow_band_hz, filter_order, filter_padtype) if flow_ok else np.full_like(v, np.nan))
            y = zero_phase(rel, fs, roam_lowpass_hz, "low", filter_order, filter_padtype) if roam_ok \
                else np.full_like(rel, np.nan)
            t_reason = None
        else:
            v = v_bp = occ_sig = y = np.full((T, 2), np.nan)
            t_reason = "no_torso"
        thr = _lookup(occ_threshold, pt)
        nm = _norm(norm_median, "FLOW-E", pt)
        for w, (a, b) in enumerate(wins):
            vf_bp, vf_v, vf_y, vf_o = vfrac(v_bp, a, b), vfrac(v, a, b), vfrac(y, a, b), vfrac(occ_sig, a, b)
            band_reason = t_reason or (None if flow_ok else "band_above_f_max")
            # FLOW-DII
            if band_reason is None:
                val = dii(v_bp[a:b])
                rsn = None if math.isfinite(val) else (
                    "too_few_samples" if int(_finite_rows(v_bp[a:b]).sum()) < 2 else "no_motion")
            else:
                val, rsn = math.nan, band_reason
            emit(w, "FLOW-DII", "dii", pt, info, val, "1", rsn, band_flags + rel_flags, vf_bp, flow_band_hz)
            # FLOW-E
            if band_reason is None:
                p, rsn = _band_power_reason(*welch_psd(v[a:b], fs, welch_seg_s), flow_band_hz)
            else:
                p, rsn = math.nan, band_reason
            emit(w, "FLOW-E", "power", pt, info, p, "torso_lengths2/s2", rsn, band_flags + rel_flags, vf_v,
                 flow_band_hz)
            db, db_rsn = db_re_norm(p, nm, rsn)
            emit(w, "FLOW-E", "db_re_norm", pt, info, db, "dB", db_rsn,
                 band_flags + rel_flags + (("no_norm",) if nm is None else ()), vf_v, flow_band_hz)
            # FLOW-R
            rsn = t_reason or (None if roam_ok else "lowpass_above_nyquist")
            yw = y[a:b][_finite_rows(y[a:b])]
            val = float(np.sqrt(((yw - yw.mean(axis=0)) ** 2).sum(axis=1).mean())) if rsn is None and len(yw) \
                else math.nan
            emit(w, "FLOW-R", "roaming", pt, info, val, "torso_lengths", rsn, (), vf_y, (0.0, roam_lowpass_hz))
            # FLOW-OCC
            occ_flags = band_flags + rel_flags + ("proposed", f"occ_{occ_signal}")
            if band_reason is None and thr is None:
                emit(w, "FLOW-OCC", "occupancy_pct", pt, info, math.nan, "percent", "no_threshold", occ_flags, vf_o,
                     flow_band_hz)
            else:
                sp = np.linalg.norm(occ_sig[a:b], axis=1)
                sp = sp[np.isfinite(sp)]
                val = 100.0 * float((sp > thr).mean()) if band_reason is None and sp.size else math.nan
                emit(w, "FLOW-OCC", "occupancy_pct", pt, info, val, "percent", band_reason, occ_flags, vf_o,
                     flow_band_hz)
    return rows



def pool_sides(rows, pool=_d("pool")):
    """Side-pooled rows of the sided rows (R12; note 4.5): one row per window, descriptor, quantity and target
    base, value the mean (or max) of the near and far values when both are finite, else NaN with reason
    side_missing; target '<base>_pooled', side 'pooled', flag side_pooled."""
    if pool not in ("mean", "max"):
        raise ValueError(f"pool {pool!r}")
    groups = {}
    for r in rows:
        if r["side"] in ("1", "2"):
            groups.setdefault((r["window"], r["descriptor"], r["quantity"], r["target_base"]), {})[r["side"]] = r
    out = []
    for (_, _, _, base), d in groups.items():
        both = [d.get("1"), d.get("2")]
        tmpl = both[0] or both[1]
        vals = [x["value"] for x in both if x is not None and x["reason"] is None and math.isfinite(x["value"])]
        if len(vals) == 2:
            value, reason = (float(np.mean(vals)) if pool == "mean" else float(max(vals))), None
        else:
            value, reason = math.nan, "side_missing"
        flags = []
        for x in both:
            for fl in (x["flags"] if x is not None else ()):
                if fl not in flags:
                    flags.append(fl)
        flags.append("side_pooled")
        vfs = [x["valid_fraction"] for x in both if x is not None and x["valid_fraction"] is not None]
        new = dict(tmpl)
        new.update(target=f"{base}_pooled", side="pooled", value=value, reason=reason, flags=tuple(flags),
                   valid_fraction=min(vfs) if vfs else None)
        out.append(new)
    return out
