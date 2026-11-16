"""G4 harness: physics from the body (design section 9; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.7 and
B4.13; rule docs/rules/G4_RULE.md, DRAFT until the owner approves and registers it).

Lines [D§9]
- grf_rmse: per-foot vertical ground reaction from pose, in percent body weight (%BW), against the force
  plates of the stroke patients of stroke_gait_vancriekinge. Per person the RMSE over the scored (frame,
  foot) samples of all its trials, then the mean over persons. Pass at most 10, kill above 20.
- grf_r_gain: per person the Pearson r of prediction and plate over the same samples, minus the r of the
  Newton floor on those samples; mean over persons. At least 0.03 is a pass condition of the GRF line, and
  the design kills below it.
- metric3d (reported, no kill): MPJPE and PA-MPJPE (mm) of bodies re-estimated from one synthetic camera
  against the true 15 joints (world_joints); it needs a model, so the dry run leaves it not evaluated.
  The camera (rule section 4, decided 2026-10-05): camera_render, the G1c draw (CAMERA_FAMILY, CAMERA_FPS,
  the default tracker process) aimed at the trial-mean pelvis, from one generator seeded by
  camera_seed(person_id, recording key) (CAMERA_TAG). metric3d_trials runs a predictor on it. The MPJPE
  frame (rule sections 4 and 6, section 8 item 10, closed 2026-10-05): pelvis-relative after one yaw about
  +y per window (yaw_aligned, g1.yaw_fit), the default of metric3d_trials and always applied by
  read_lockbox; PA-MPJPE per frame.
- emg_r: gait120 level-walking trials, right tibialis anterior, gastrocnemius medialis, vastus lateralis
  and biceps femoris; per trial and muscle the Pearson r of predicted and measured log-envelopes over the
  pose frames; per person the median over its (trial, muscle) pairs, then the median over persons. Pass
  at least 0.6, kill below 0.4.
- emg_r_gain (rule revision 2026-10-05; summary and gate_results with emg_gain=True, as read_lockbox
  does): per person the model's median r minus the template's median r over the same (trial, muscle)
  pairs, median over persons; pass at least delta, kill below delta, delta = EMG_GAIN = 0.03 (rule
  sections 1 and 6, section 8 item 9, closed 2026-10-05). emg_r passes only when emg_r_gain passes.
Sets: dev, the 40 non-lockbox stroke patients (never trained in the pilot) and the gait120 dev fold
(mdfm.data.sources.folds); lockbox, the 10 lockbox stroke patients and the 24 lockbox gait120 persons, read
once through read_lockbox after the rule is registered. Statistics: one value per person, the point on
the persons; intervals by a person bootstrap (BOOT_N resamples, SEED), percentile 95 percent.
EMG trials (rule sections 2 and 6, revision 2026-10-05): a trial whose paired pose lasts under EMG_MIN_S
(1 s, infer.MIN_S) is left out for the template and the model alike and counted (split_min_s); every
predictor is scored on the same frames of the same (trial, muscle) pairs (emg_trial_table,
same_samples). Both are arguments off by default (evaluate, emg_trial_table, covered_persons), so the
development scripts keep their rows; read_lockbox applies them.
Lines closed on the evening of 2026-10-05 (rule section 8, items 3, 7, 8, 10 and 11; task GR2), which
read_lockbox applies without an argument:
- the model's flight rule (rule section 4, item 11): at a frame where both of the model's contact
  probabilities are below 0.5 it predicts 0 BW on both feet, as the Newton floor (apply_flight; the sealed
  read's GRF predictor gives the per-foot BW and the contact probabilities, flight_predictor);
- the secondary GRF row (rule section 2 (b), item 3): the primary set without the plate series (trial, foot)
  whose plate peak at the scored samples stays below LOW_PEAK_BW, with grf_rmse and grf_r_gain and the Newton
  floor recomputed on its own samples (grf_person_table with min_peak_bw; evaluate with low_peak); it decides
  nothing, and the series left out are listed;
- the training persons (rule section 2, item 7): the sealed read takes the loaded checkpoint and draws from it
  its SHA-256, its training lists per source and the training-person SHA-256 it records (checkpoint_identity);
  the lists must be non-empty and hash to that value (check_train_persons), and the scored sets are disjoint
  from them, all before the read is logged;
- the MPJPE frame (item 10, above).
The body mass (rule section 3, item 8) is grf_trials's default: see Truth. A patient dropped from the GRF lines
for want of a mass keeps its trials in the metric-3D round trip, which needs no mass (grf_trials no_mass).

Truth
- World joints (T, 15, 3) m, y up: canonical coords (stroke rotated to y up by sources.pair) x scale plus
  the raw root track of mdfm.data.mocap_raw; NaN where the joint or the root is invalid. The lab floor is
  at height 0 in both sources.
- Plates: the canonical force record (1000 Hz, column 8 per foot) through force.prepare_totals with
  coverage 'plate' (per-foot unloaded offset removed; a foot that never unloads masked; samples within 0.1 s
  of a plate contact scored), divided by m G with the subject's mass (mocap_raw.body_mass: the C3D
  parameter, then the MAT file; rule section 3, item 8: a patient with neither is dropped from the GRF
  lines and listed, reason no_body_mass; mass_from_steps is a check, used as the mass only when a caller
  asks with steps_mass, never by read_lockbox), low-passed as the window store's force band (4th-order
  zero-phase Butterworth at 40 Hz) and read at the pose frames (sample k i for pose frame i, k = force fps
  / pose fps).
- EMG: emg.log_envelope of the trial's raw EMG (20 to 450 Hz, 20 ms RMS, log with the 5th-percentile
  floor), linearly interpolated at the pose times of sources.pair; a record with non-finite samples is
  filtered per finite run (log_envelope_runs) and pose times next to a dropped sample are not scored.

Baselines (no model)
- Newton floor (newton_floor): world joints low-passed at 12 Hz (4th order, zero phase) on each run of
  frames where the 15 joints are valid, runs under MIN_RUN frames dropped; force.com15 with the recorded
  sex; a_com by central second differences; the contact probability of each foot is force.contact_weights
  of the ankle height above the trial's stance height (the STANCE_PCT percentile of both ankles' heights
  pooled over the trial) and of the ankle speed; force.newton_bw; a flight frame (both probabilities
  below 0.5) predicts 0 on both feet. Scored samples: plate-valid samples at frames where the Newton
  floor is defined, for the floor and a model alike.
- Template EMG (build_template): the step phase of each pose frame (frame f of step [a, b] has phase
  (f - a) / (b - a); a frame shared by two steps keeps the earlier step, phase 1); per muscle, the mean over
  training persons of each person's mean log-envelope in N_PHASE phase bins (bin round(100 phase)),
  evaluated by linear interpolation over the bins.
"""
import collections
import hashlib
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from mdfm import paths
from mdfm.data import canonical, mocap_raw, sources
from mdfm.data.window_store import LOWPASS_HZ, ORDER
from mdfm.kernels import emg as emg_kernel
from mdfm.kernels import force
from mdfm.kernels.camera import CANONICAL_15

GATE = "G4"
RULE_PATH = os.path.join(paths.TREE, "docs", "rules", "G4_RULE.md")
STROKE = mocap_raw.STROKE
GAIT120 = mocap_raw.GAIT120
MUSCLES = ("tibialis_anterior", "gastrocnemius_medialis", "vastus_lateralis", "biceps_femoris")
EMG_SIDE = "R"
EMG_TASK = re.compile(r"^level_walking_trial\d+$")
GRF_PASS, GRF_KILL = 10.0, 20.0             # percent BW [D§9]
R_GAIN = 0.03                               # r above the Newton floor [D§9]
EMG_PASS, EMG_KILL = 0.6, 0.4               # median r [D§9]
SEED = 20260929
BOOT_N = 2000
COM_CUTOFF_HZ = 12.0                        # [cfg]
MIN_RUN = 45                                # frames; prep.lowpass leaves shorter series unfiltered [cfg]
STANCE_PCT = 5.0                            # [cfg]
N_PHASE = 101
MIN_EMG_RUN_S = 0.25                        # s: shorter finite EMG runs are masked (log_envelope_runs) [cfg]
EMG_GAIN = 0.03                             # delta of emg_r_gain (rule sections 1 and 6; section 8 item 9, closed
                                            # 2026-10-05). None would leave the line not evaluated, blocking emg_r
EMG_MIN_S = 1.0                             # s of paired pose, the shortest EMG trial scored (rule sections 2 and
                                            # 6: infer.MIN_S, the window store's minimum)
LOW_PEAK_BW = 0.5                           # BW: the secondary GRF row leaves out a plate series whose peak at the
                                            # scored samples stays below it (rule sections 2 and 6, item 3)
CAMERA_TAG = "G4v1"                         # metric-3D camera seed tag (rule section 4)
CAMERA_FAMILY = "generic"                   # the G1c camera family (g1.PARAMS["camera_family"])
CAMERA_FPS = 30.0                           # the G1c camera rate (g1.CAMERA_FPS)
LEFT_OUT_COLUMNS = ("line", "source", "person_id", "key", "stream", "reason", "samples", "fps", "seconds",
                    "min_samples")
LEFT_OUT = {                                # reason: meaning
    "pose_shorter_than_min_s": "EMG trial whose paired pose lasts under the minimum (EMG_MIN_S in the sealed read)",
    "camera_no_pelvis": "metric-3D trial whose truth pelvis is never valid: no camera render",
    "camera_no_prediction": "metric-3D trial for which the predictor returns no joints (its input gives no window)",
    "no_body_mass": "GRF trial of a patient with neither a C3D nor a MAT body mass: dropped from the GRF lines "
                    "(rule section 3, item 8)",
}
LOW_PEAK_COLUMNS = ("person_id", "key", "foot", "samples", "peak_bw")
ANKLES = (CANONICAL_15.index("left_ankle"), CANONICAL_15.index("right_ankle"))
UP = 1


# ---------------------------------------------------------------- sets
def stroke_patients(lockbox=False):
    """Sorted subject_ns of the stroke patients (index group 'stroke') with pose rows, outside (lockbox
    False) or inside the lockbox. Reads the index only."""
    df = canonical.index()
    sel = (df["source"] == STROKE) & (df["modality"] == "pose") & (df["group"] == "stroke") & (df["lockbox"] == lockbox)
    return sorted(set(df.loc[sel, "subject_ns"]))


def gait120_persons(fold):
    """Sorted gait120 person ids of a fold: 'train' or 'dev' of sources.folds(), or 'lockbox' (index)."""
    if fold == "lockbox":
        df = canonical.index()
        return sorted(set(df.loc[(df["source"] == GAIT120) & df["lockbox"], "person_id"]))
    f = sources.folds()
    return sorted(f.loc[(f["source"] == GAIT120) & (f["fold"] == fold), "person_id"])


def _recordings(source, persons, modalities, allow_lockbox):
    """{(person_id, subject_ns, recording): [index rows]} of the persons, modalities given; rows only. A CODY
    person is refused by name before the index is read (G4 never reads CODY), and every row goes through
    canonical.check with the lockbox flag given."""
    persons = list(persons)
    cody = [p for p in persons if canonical.is_cody(source, p)]
    if cody:
        raise canonical.Refused(f"CODY is evaluation-only and never a G4 set: {cody[:3]}")
    for p in persons:
        canonical.check({"source": source, "subject_ns": p, "person_id": p}, allow_lockbox=allow_lockbox)
    df = canonical.index()
    sel = df[(df["source"] == source) & df["person_id"].isin(set(persons)) & df["modality"].isin(modalities)]
    out = collections.defaultdict(list)
    for r in sel.to_dict("records"):
        r = canonical.check(r, allow_lockbox=allow_lockbox)
        out[(r["person_id"], r["subject_ns"], sources.recording_key(source, r["modality"], r["key"]))].append(r)
    return dict(sorted(out.items()))


def min_samples(fps, min_s):
    """The fewest samples a stream at fps holds when it lasts min_s s: round(min_s fps), at least 1 (the test
    of infer.spans: a stream under it gives no window)."""
    return max(1, int(round(float(min_s) * float(fps))))


def index_pose_samples(source, pose_rows):
    """(samples, fps) of a recording's pose stream from its index rows [(n_frames, fps)], or None when the rows
    carry no usable count or rate. gait120: the pairing joins the step rows and keeps a boundary frame shared by
    two consecutive steps once, which the index cannot see, so the count is the sum of n_frames minus one frame
    per step boundary, the fewest samples the pairing can give. Other sources: one pose row per recording."""
    rows = list(pose_rows)
    if not rows:
        return None
    n = [int(r[0]) if r[0] is not None and np.isfinite(float(r[0])) else -1 for r in rows]
    fps = {float(r[1]) for r in rows}
    if min(n) <= 0 or len(fps) != 1:
        return None
    f = fps.pop()
    if not (math.isfinite(f) and f > 0):
        return None
    if source == GAIT120:
        return sum(n) - (len(n) - 1), f
    return (n[0], f) if len(n) == 1 else None


def covered_persons(source, persons, min_s=None, stats=None):
    """Sorted persons of a set with at least one recording that holds both streams of its line (stroke:
    pose and force; gait120: level-walking pose and EMG), counted from the index rows alone: no record
    opens, so it sizes a lockbox set before its read.
    min_s (s; off by default, the rule's EMG line takes EMG_MIN_S): a recording counts only when its pose
    stream holds at least min_samples(fps, min_s) samples by index_pose_samples; a recording whose rows carry
    no count or rate counts (the minimum cannot be applied there) and is counted apart. stats (a Counter,
    optional): recordings_with_both_streams, and with min_s recordings_under_min_s and
    recordings_without_sample_count."""
    persons = list(persons)
    if any(canonical.is_cody(source, p) for p in persons):
        raise canonical.Refused("CODY is evaluation-only and never a G4 set")
    stats = collections.Counter() if stats is None else stats
    need = {"pose", "force"} if source == STROKE else {"pose", "emg"}
    df = canonical.index()
    sel = df[(df["source"] == source) & df["person_id"].isin(set(persons)) & df["modality"].isin(need)]
    have = collections.defaultdict(set)
    pose_rows = collections.defaultdict(list)
    for pid, mod, key, n, fps in sel[["person_id", "modality", "key", "n_frames", "fps"]].itertuples(index=False):
        rk = sources.recording_key(source, mod, key)
        if source == STROKE or EMG_TASK.match(rk):
            have[(pid, rk)].add(mod)
            if mod == "pose":
                pose_rows[(pid, rk)].append((n, fps))
    out = set()
    for (pid, rk), mods in sorted(have.items()):
        if mods != need:
            continue
        stats["recordings_with_both_streams"] += 1
        if min_s is not None:
            got = index_pose_samples(source, pose_rows[(pid, rk)])
            if got is None:
                stats["recordings_without_sample_count"] += 1
            elif got[0] < min_samples(got[1], min_s):
                stats["recordings_under_min_s"] += 1
                continue
        out.add(pid)
    return sorted(out)


def paired_records(source, persons, allow_lockbox=False, stats=None):
    """Generator of sources.PairedRecord: stroke pose with force, gait120 level-walking pose with EMG. A
    recording without both streams, or that does not pair, is counted in stats and skipped."""
    stats = collections.Counter() if stats is None else stats
    need = ("pose", "force") if source == STROKE else ("pose", "emg")
    for (pid, sns, rk), rows in _recordings(source, persons, need, allow_lockbox).items():
        if source == GAIT120 and not EMG_TASK.match(rk):
            continue
        if {r["modality"] for r in rows} != set(need):
            stats["dropped_missing_stream"] += 1
            continue
        try:
            pr = sources.pair(source, [(r, canonical.load(r, allow_lockbox=allow_lockbox)) for r in rows])
        except sources.PairingError as e:
            stats[f"dropped_{e.reason}"] += 1
            continue
        stats["paired"] += 1
        yield pr


# ---------------------------------------------------------------- truth
def world_joints(pose, root, root_valid):
    """(T, 15, 3) m y up: coords x scale of a paired pose record (already y up) plus the root track (T, 3);
    NaN where the joint (valid_mask) or the root is invalid."""
    X = np.asarray(pose["coords"], dtype=np.float64) * float(pose["scale"]) + np.asarray(root)[:, None]
    ok = np.asarray(pose["valid_mask"], dtype=bool) & np.asarray(root_valid, dtype=bool)[:, None]
    return np.where(ok[..., None], X, np.nan)


def lowpass(x, fps, cutoff, order=ORDER):
    """Zero-phase Butterworth low-pass of x (T, ...) along axis 0."""
    b, a = butter(order, cutoff / (fps / 2.0))
    return filtfilt(b, a, x, axis=0)


def plate_bw(frec, n_pose, fps_pose, mass_kg):
    """Plate truth at the pose frames (module docstring): obs (n_pose, 2) BW and valid (n_pose, 2) bool, and
    the offset (2,) N removed. frec: a paired force record (force (T_f, 2, 9) N, sensor_mask, fps)."""
    fps_f = float(frec["fps"])
    k = int(round(fps_f / fps_pose))
    if abs(k * fps_pose - fps_f) > 1e-6:
        raise ValueError(f"force at {fps_f:g} Hz is not a multiple of the pose rate {fps_pose:g} Hz")
    total = np.asarray(frec["force"], dtype=np.float64)[:, :, 8]
    present = np.asarray(frec["sensor_mask"], dtype=bool)[:, 8]
    sv = frec.get("sample_valid")
    offset, usable, valid = force.prepare_totals(np.nan_to_num(total), present, fps_f, unit="N", coverage="plate",
                                                 valid=None if sv is None else np.asarray(sv, dtype=bool))
    y = lowpass((np.nan_to_num(total) - offset[None]) / (mass_kg * force.G), fps_f, LOWPASS_HZ)
    idx = k * np.arange(n_pose)
    if idx[-1] >= len(y):
        raise ValueError(f"{len(y)} force samples for {n_pose} pose frames at k = {k}")
    ok = valid[idx] & usable[None] & np.isfinite(total[idx])
    return np.where(ok, y[idx], 0.0), ok, offset


def _runs(ok):
    """[(start, end)] of the runs of True in a (T,) bool array."""
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def lowpass_runs(x, ok, fps, cutoff=COM_CUTOFF_HZ, min_run=MIN_RUN):
    """x (T, ...) low-passed on each run of ok frames of at least min_run frames; (y with NaN elsewhere,
    kept (T,) bool)."""
    y = np.full(x.shape, np.nan)
    kept = np.zeros(len(ok), dtype=bool)
    for s, e in _runs(np.asarray(ok, dtype=bool)):
        if e - s >= min_run:
            y[s:e] = lowpass(x[s:e], fps, cutoff)
            kept[s:e] = True
    return y, kept


def _central(y, kept, fps):
    """First and second central differences (T, ...) of y, and where both are defined (T,) bool."""
    d1 = np.full(y.shape, np.nan)
    d2 = np.full(y.shape, np.nan)
    d1[1:-1] = (y[2:] - y[:-2]) * (fps / 2.0)
    d2[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) * fps ** 2
    ok = np.zeros(len(kept), dtype=bool)
    ok[1:-1] = kept[:-2] & kept[1:-1] & kept[2:]
    return d1, d2, ok


def newton_floor(world, fps, sex, stance_pct=STANCE_PCT):
    """The model-free Newton floor (module docstring): (pred (T, 2) BW [left, right], ok (T,) bool, contact
    (T, 2) probabilities) from world joints (T, 15, 3) m y up with NaN where invalid."""
    T = world.shape[0]
    frame_ok = np.isfinite(world).all(axis=(1, 2))
    y, kept = lowpass_runs(world.reshape(T, -1), frame_ok, fps)
    y = y.reshape(T, 15, 3)
    com = force.com15(y, sex)
    _, acc, ok = _central(com, kept, fps)
    ank = y[:, ANKLES]
    vel, _, _ = _central(ank, kept, fps)
    height = ank[..., UP]
    stance = np.percentile(height[kept], stance_pct) if kept.any() else np.nan
    h = torch.as_tensor(np.nan_to_num(height - stance, nan=1.0), dtype=torch.float64)
    v = torch.as_tensor(np.nan_to_num(np.linalg.norm(vel, axis=-1), nan=10.0), dtype=torch.float64)
    p = force.contact_weights(h, v)
    a = torch.as_tensor(np.nan_to_num(acc), dtype=torch.float64)
    pred = force.newton_bw(a, p, up=UP)
    pred = torch.where(force.flight(p)[:, None], torch.zeros_like(pred), pred).numpy()
    return np.where(ok[:, None], pred, np.nan), ok, p.numpy()


def mass_from_steps(total, observed, swing, fps, band=force.BAND_N):
    """Body mass (kg) from the plates (spec question 7; since rule section 3, item 8, closed 2026-10-05, a check
    reported beside the recorded mass, never the mass of the rule's lines, grf_trials steps_mass): the
    mean over complete steps of the total vertical force divided by G. A complete step runs from a loading
    onset of one foot (its offset-removed total crossing `band` N upward on the plates) to the next onset of
    the other foot, with each foot at every sample either observed on its plates or in swing. total (T, 2)
    N, observed and swing (T, 2) bool at the force rate. NaN without a complete step."""
    total = np.asarray(total, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    covered = (observed | np.asarray(swing, dtype=bool)).all(axis=1)
    loaded = observed & (np.nan_to_num(total) > band)
    onsets = sorted((int(i) + 1, j) for j in range(2) for i in np.flatnonzero(loaded[1:, j] & ~loaded[:-1, j]))
    num, den = 0.0, 0
    for (i0, f0), (i1, f1) in zip(onsets, onsets[1:]):
        if f0 == f1 or not covered[i0:i1].all():
            continue
        num += float(np.where(observed[i0:i1], np.nan_to_num(total[i0:i1]), 0.0).sum())
        den += i1 - i0
    return num / den / force.G if den else float("nan")


@dataclass
class GRFTrial:
    """One stroke trial for the GRF line: world joints (T, 15, 3) m y up; obs (T, 2) BW and obs_valid at the
    pose frames; the subject's mass (kg) and where it came from; sex 'F', 'M' or None."""
    person: str
    key: str
    fps: float
    t: np.ndarray
    world: np.ndarray
    obs: np.ndarray
    obs_valid: np.ndarray
    mass_kg: float
    mass_from: str
    sex: str = None
    root_max_abs_m: float = float("nan")


def _sex(subject_ns, allow_lockbox):
    male = mocap_raw.mat_char(subject_ns, allow_lockbox=allow_lockbox).get("Male")
    return {1.0: "M", 0.0: "F"}.get(male)


def grf_trials(persons, allow_lockbox=False, stats=None, steps_mass=False, left_out=None, no_mass=None):
    """GRFTrial list of the stroke persons (module docstring). Body mass (rule section 3, item 8): the C3D
    parameter, else the MAT file (mocap_raw.body_mass); a trial of a patient with neither is dropped from the GRF
    lines, counted in stats (dropped_no_mass) and appended to left_out (a list, optional; LEFT_OUT_COLUMNS, line grf,
    stream force, reason no_body_mass). no_mass (a list, optional) receives each dropped trial as a GRFTrial
    without its plate truth (obs 0, obs_valid False, mass_kg NaN, mass_from ''), for the line that needs no mass:
    the metric-3D round trip scores every lockbox stroke trial (rule sections 1 and 4). steps_mass (off by
    default, never set by read_lockbox): the plate-derived mass of trial_steps_mass is taken for such a trial
    instead, mass_from 'steps' (the dropped trial is the one with no finite steps mass either)."""
    stats = collections.Counter() if stats is None else stats
    out, mass, sex = [], {}, {}
    for pr in paired_records(STROKE, persons, allow_lockbox=allow_lockbox, stats=stats):
        sns = pr.subject_ns
        if sns not in mass:
            mass[sns] = mocap_raw.body_mass(STROKE, sns, allow_lockbox=allow_lockbox)
            sex[sns] = _sex(sns, allow_lockbox)
        root, root_valid, _ = mocap_raw.root_for_paired(pr, allow_lockbox=allow_lockbox)
        pose = pr.streams["pose3d"]
        world = world_joints(pose, root, root_valid)
        fps = float(pose["fps"])
        m, how = mass[sns]
        root_max = float(np.nanmax(np.abs(root))) if root_valid.any() else float("nan")
        if not (np.isfinite(m) and m > 0):
            m, how = (trial_steps_mass(pr, world, fps), "steps") if steps_mass else (float("nan"), "")
            if not (np.isfinite(m) and m > 0):
                stats["dropped_no_mass"] += 1
                n = len(pr.t["pose3d"])
                if left_out is not None:
                    left_out.append(_left("grf", STROKE, pr.person_id, pr.key, "force", "no_body_mass", n, fps))
                if no_mass is not None:
                    no_mass.append(GRFTrial(person=pr.person_id, key=pr.key, fps=fps, t=pr.t["pose3d"], world=world,
                                            obs=np.zeros((n, 2)), obs_valid=np.zeros((n, 2), dtype=bool),
                                            mass_kg=float("nan"), mass_from="", sex=sex[sns],
                                            root_max_abs_m=root_max))
                continue
        obs, ok, _ = plate_bw(pr.streams["force"], len(pr.t["pose3d"]), fps, m)
        out.append(GRFTrial(person=pr.person_id, key=pr.key, fps=fps, t=pr.t["pose3d"], world=world, obs=obs,
                            obs_valid=ok, mass_kg=float(m), mass_from=how, sex=sex[sns], root_max_abs_m=root_max))
    return out


def trial_steps_mass(pr, world, fps):
    """mass_from_steps (kg) of one paired stroke trial: offset-removed plate totals, observed where
    prepare_totals scores them, swing where the Newton floor's contact probability is below 0.5."""
    frec = pr.streams["force"]
    fps_f = float(frec["fps"])
    k = int(round(fps_f / fps))
    total = np.nan_to_num(np.asarray(frec["force"], dtype=np.float64)[:, :, 8])
    offset, usable, valid = force.prepare_totals(total, np.asarray(frec["sensor_mask"], bool)[:, 8], fps_f,
                                                 unit="N", coverage="plate")
    _, ok, p = newton_floor(world, fps, None)
    swing = np.repeat((p < 0.5) & ok[:, None], k, axis=0)[:len(total)]
    return mass_from_steps(total - offset[None], valid & usable[None], swing, fps_f)


def newton_predict(trial):
    """Newton-floor prediction (T, 2) BW of a GRFTrial, NaN where undefined."""
    return newton_floor(trial.world, trial.fps, trial.sex)[0]


def apply_flight(bw, contact):
    """The model's flight rule (rule section 4, section 8 item 11, closed 2026-10-05): (T, 2) BW with 0 on both
    feet at a frame where both contact probabilities (T, 2) are below 0.5 (force.flight), as the Newton floor
    does; NaN on both feet at a frame whose contact probabilities are not both finite (the rule cannot be read
    there), else bw unchanged."""
    bw = np.asarray(bw, dtype=np.float64)
    p = np.asarray(contact, dtype=np.float64)
    if bw.ndim != 2 or bw.shape[1] != 2 or p.shape != bw.shape:
        raise ValueError(f"per-foot BW {bw.shape} and contact probabilities {p.shape}: both (T, 2) expected")
    known = np.isfinite(p).all(axis=1)
    fl = np.asarray(force.flight(np.where(known[:, None], p, 1.0)), dtype=bool) & known
    return np.where(~known[:, None], np.nan, np.where(fl[:, None], 0.0, bw)), fl


def flight_predictor(predictor, stats=None):
    """callable(GRFTrial) -> (T, 2) BW of a GRF predictor that returns (bw, contact), both (T, 2) at the trial's
    pose frames (per-foot vertical GRF in BW and the model's contact probabilities): apply_flight, the model's
    flight rule. ValueError for any other return. stats (a Counter, optional) counts grf_model_frames_plate, the
    frames with a valid plate sample, and grf_model_flight_frames_plate, those read in flight."""
    stats = collections.Counter() if stats is None else stats

    def f(trial):
        got = predictor(trial)
        if not (isinstance(got, tuple) and len(got) == 2):
            raise ValueError(f"{trial.person} {trial.key}: the GRF predictor returns (bw, contact), each (T, 2)")
        y, fl = apply_flight(*got)
        if y.shape != np.asarray(trial.obs).shape:
            raise ValueError(f"{trial.person} {trial.key}: predicted GRF {y.shape} for truth {trial.obs.shape}")
        plate = np.asarray(trial.obs_valid, dtype=bool).any(axis=1)
        stats["grf_model_frames_plate"] += int(plate.sum())
        stats["grf_model_flight_frames_plate"] += int((fl & plate).sum())
        return y

    return f


# ---------------------------------------------------------------- EMG truth and template
def step_phase(frames, step_frames):
    """Phase in [0, 1] of each pose frame (T,) within its step (module docstring); NaN outside every step.
    step_frames [[a, b], ...] in frame numbers."""
    frames = np.asarray(frames, dtype=np.float64)
    phase = np.full(frames.shape, np.nan)
    for a, b in step_frames:
        a, b = float(a), float(b)
        sel = np.isnan(phase) & (frames >= a) & (frames <= b)
        phase[sel] = (frames[sel] - a) / (b - a)
    return phase


def muscle_columns(channels, muscles=MUSCLES, side=EMG_SIDE):
    """Column of each muscle on the side in an EMG channel list, -1 when absent or recorded twice."""
    out = []
    for m in muscles:
        hits = [i for i, c in enumerate(channels) if c.get("muscle") == m and c.get("side") == side]
        out.append(hits[0] if len(hits) == 1 else -1)
    return np.asarray(out, dtype=np.int64)


def log_envelope_runs(sig, fps, kind="raw"):
    """(log-envelope (T, C), kept (T,) bool) of an EMG signal (T, C) with possible non-finite samples:
    emg.log_envelope when every sample is finite; otherwise emg.envelope on each run of samples finite on
    every channel (the kernel's contract; runs under MIN_EMG_RUN_S dropped), one per-channel floor (the 5th
    percentile over the kept samples), NaN outside the kept samples."""
    sig = np.asarray(sig, dtype=np.float64)
    fin = np.isfinite(sig).all(axis=1)
    if fin.all():
        return emg_kernel.log_envelope(sig, fps, kind)[0], fin
    lin = np.full(sig.shape, np.nan)
    for s, e in _runs(fin):
        if e - s >= MIN_EMG_RUN_S * fps:
            lin[s:e] = emg_kernel.envelope(sig[s:e], fps, kind)
    kept = np.isfinite(lin).all(axis=1)
    if not kept.any():
        return lin, kept
    floor = np.maximum(np.percentile(lin[kept], emg_kernel.FLOOR_PERCENTILE, axis=0), emg_kernel.FLOOR_MIN)
    return np.log(np.maximum(lin, floor[None])), kept


def measured_log_envelope(pr, muscles=MUSCLES):
    """(obs (T, M), valid (T, M)): log-envelopes of the muscles at the pose times of a gait120 PairedRecord
    (log_envelope_runs); a pose time is valid inside the EMG span with both EMG samples around it kept."""
    e = pr.streams["emg"]
    cols = muscle_columns(e["channels"], muscles)
    t_p, t_e = pr.t["pose3d"], pr.t["emg"]
    obs = np.zeros((len(t_p), len(muscles)))
    valid = np.zeros(obs.shape, dtype=bool)
    have = cols >= 0
    if have.any():
        sig = np.asarray(e["signal"], dtype=np.float64)[:, cols[have]]
        env, kept = log_envelope_runs(sig, float(e["fps"]), e.get("signal_kind") or "raw")
        j1 = np.clip(np.searchsorted(t_e, t_p), 1, len(t_e) - 1)
        inside = (t_p >= t_e[0]) & (t_p <= t_e[-1]) & kept[j1 - 1] & kept[j1]
        for j, c in enumerate(np.flatnonzero(have)):
            obs[:, c] = np.interp(t_p, t_e, np.nan_to_num(env[:, j]))
            valid[:, c] = inside
    return obs, valid


@dataclass
class EMGTrial:
    """One gait120 level-walking trial: pose times t (T,), frames (T,), phase (T,), the paired record (for a
    model), measured log-envelopes obs (T, M) and valid (T, M); fps the pose rate (None: read from t)."""
    person: str
    key: str
    t: np.ndarray
    frames: np.ndarray
    phase: np.ndarray
    obs: np.ndarray
    valid: np.ndarray
    record: object = field(default=None, repr=False)
    fps: float = None


def emg_trial(pr, keep_record=False):
    """EMGTrial of one gait120 PairedRecord."""
    frames = np.asarray(pr.pairing["pose_frames"], dtype=np.int64)
    obs, valid = measured_log_envelope(pr)
    phase = step_phase(frames, pr.pairing["step_frames"])
    fps = pr.streams["pose3d"].get("fps")
    return EMGTrial(person=pr.person_id, key=pr.key, t=pr.t["pose3d"], frames=frames, phase=phase, obs=obs,
                    valid=valid & np.isfinite(phase)[:, None], record=pr if keep_record else None,
                    fps=float(fps) if fps is not None else None)


def trial_fps(trial):
    """Pose rate (Hz) of an EMGTrial: its fps, else one over the median step of its pose times; NaN with fewer
    than 2 times."""
    if getattr(trial, "fps", None):
        return float(trial.fps)
    t = np.asarray(trial.t, dtype=np.float64)
    d = float(np.median(np.diff(t))) if len(t) >= 2 else float("nan")
    return 1.0 / d if d > 0 else float("nan")


def split_min_s(trials, min_s, source=GAIT120):
    """(kept, left_out) of EMG trials against the minimum (rule section 2): a trial whose pose holds fewer than
    min_samples(fps, min_s) samples is left out, one LEFT_OUT_COLUMNS dict each (line emg, stream pose, reason
    pose_shorter_than_min_s); a trial without a pose rate (trial_fps NaN) is left out too, its minimum
    unknown."""
    kept, out = [], []
    for tr in trials:
        n = int(len(tr.t))
        fps = trial_fps(tr)
        need = min_samples(fps, min_s) if math.isfinite(fps) else None
        if need is not None and n >= need:
            kept.append(tr)
            continue
        out.append(_left("emg", source, tr.person, tr.key, "pose", "pose_shorter_than_min_s", n, fps, need))
    return kept, out


def emg_trials(persons, allow_lockbox=False, stats=None, keep_record=False):
    """EMGTrial list of gait120 level-walking trials of the persons."""
    return [emg_trial(pr, keep_record) for pr in paired_records(GAIT120, persons, allow_lockbox=allow_lockbox,
                                                                  stats=stats)]


def build_template(trials, n_phase=N_PHASE):
    """(n_phase, M) template: per muscle, the mean over persons of each person's mean log-envelope in each
    phase bin (module docstring); NaN for a bin no person fills."""
    M = trials[0].obs.shape[1]
    per = collections.defaultdict(lambda: (np.zeros((n_phase, M)), np.zeros((n_phase, M))))
    for tr in trials:
        s, n = per[tr.person]
        b = np.round(np.nan_to_num(tr.phase) * (n_phase - 1)).astype(int)
        for m in range(M):
            ok = tr.valid[:, m]
            np.add.at(s[:, m], b[ok], tr.obs[ok, m])
            np.add.at(n[:, m], b[ok], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.stack([s / n for s, n in per.values()])
        return np.nanmean(means, axis=0) if len(means) else np.full((n_phase, M), np.nan)


def template_predict(template, phase):
    """(T, M) template log-envelopes at the phases (T,) by linear interpolation over the bins."""
    grid = np.linspace(0.0, 1.0, template.shape[0])
    out = np.full((len(phase), template.shape[1]), np.nan)
    ok = np.isfinite(phase)
    for m in range(template.shape[1]):
        good = np.isfinite(template[:, m])
        if good.sum() >= 2:
            out[ok, m] = np.interp(phase[ok], grid[good], template[good, m])
    return out


# ---------------------------------------------------------------- scoring
def pearson(a, b):
    """Pearson r of two (n,) arrays over the pairs where both are finite; NaN under 3 pairs or no spread."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def grf_samples(trial, preds, min_peak_bw=None, low_peak=None):
    """Flat scored samples of one trial: (obs (n,), {name: pred (n,)}); a sample is scored where the plate
    is valid and every prediction is finite (the Newton floor defines the set in the dry run).
    min_peak_bw (BW; off by default, the secondary row of rule section 2 (b) passes LOW_PEAK_BW): a plate series
    (the trial's foot) whose plate peak at its scored samples stays below it is left out, all its samples, and
    one LOW_PEAK_COLUMNS dict is appended to low_peak (a list, optional); a series without a scored sample is
    not counted."""
    ok = np.asarray(trial.obs_valid, dtype=bool).copy()
    for p in preds.values():
        ok &= np.isfinite(p)
    if min_peak_bw is not None:
        for j, foot in enumerate(("L", "R")):
            if not ok[:, j].any():
                continue
            peak = float(np.max(trial.obs[ok[:, j], j]))
            if not peak >= min_peak_bw:
                if low_peak is not None:
                    low_peak.append({"person_id": trial.person, "key": trial.key, "foot": foot,
                                     "samples": int(ok[:, j].sum()), "peak_bw": peak})
                ok[:, j] = False
    return trial.obs[ok], {k: np.asarray(p)[ok] for k, p in preds.items()}


def grf_person_table(trials, predictors, min_peak_bw=None, low_peak=None):
    """DataFrame per person: n samples, trials, mass and its origin, and per predictor the RMSE in %BW and
    Pearson r over the person's pooled samples. predictors: {name: callable(GRFTrial) -> (T, 2) BW}.
    min_peak_bw and low_peak (off by default): grf_samples's, the secondary row of rule section 2 (b); a person
    whose every series is left out reads samples 0 and NaN."""
    import pandas as pd
    by = collections.defaultdict(list)
    for tr in trials:
        by[tr.person].append(tr)
    rows = []
    for person, trs in sorted(by.items()):
        obs, pred = [], collections.defaultdict(list)
        for tr in trs:
            o, p = grf_samples(tr, {k: f(tr) for k, f in predictors.items()}, min_peak_bw, low_peak)
            obs.append(o)
            for k, v in p.items():
                pred[k].append(v)
        o = np.concatenate(obs)
        row = {"person_id": person, "trials": len(trs), "samples": int(o.size), "mass_kg": trs[0].mass_kg,
               "mass_from": trs[0].mass_from, "sex": trs[0].sex or ""}
        for k in predictors:
            p = np.concatenate(pred[k])
            row[f"{k}_rmse_pct_bw"] = 100.0 * float(np.sqrt(np.mean((p - o) ** 2))) if o.size else float("nan")
            row[f"{k}_r"] = pearson(p, o)
        rows.append(row)
    return pd.DataFrame(rows)


def emg_trial_table(trials, predictors, muscles=MUSCLES, min_s=None, same_samples=False, left_out=None):
    """DataFrame per (trial, muscle): person_id, key, muscle, n, and per predictor the Pearson r over the
    valid pose frames. predictors: {name: callable(EMGTrial) -> (T, M) log-envelopes}.
    min_s (s; off by default, the sealed read passes EMG_MIN_S): the trials under it (split_min_s) are left out
    before any predictor is called, for every predictor alike, and appended to left_out (a list) when given.
    same_samples (off by default, set by the sealed read): every predictor is scored on the same frames, the
    valid frames where every prediction is finite (as grf_samples does for GRF), counted in n; a pair where
    the r of any predictor is not finite is scored for none (every r NaN), and the column scored says which
    pairs count, so the person medians run over the same pairs for every predictor."""
    import pandas as pd
    if min_s is not None:
        trials, lo = split_min_s(trials, min_s)
        if left_out is not None:
            left_out.extend(lo)
    rows = []
    for tr in trials:
        preds = {k: f(tr) for k, f in predictors.items()}
        for m, name in enumerate(muscles):
            ok = tr.valid[:, m]
            if same_samples:
                ok = ok.copy()
                for p in preds.values():
                    ok &= np.isfinite(np.asarray(p, dtype=np.float64)[:, m])
            row = {"person_id": tr.person, "key": tr.key, "muscle": name, "n": int(ok.sum())}
            for k, p in preds.items():
                row[f"{k}_r"] = pearson(p[ok, m], tr.obs[ok, m]) if ok.any() else float("nan")
            if same_samples:
                row["scored"] = all(math.isfinite(row[f"{k}_r"]) for k in preds)
                if not row["scored"]:
                    for k in preds:
                        row[f"{k}_r"] = float("nan")
            rows.append(row)
    if not rows:                                # every trial left out: the columns, no row
        cols = {"person_id": object, "key": object, "muscle": object, "n": np.int64,
                **{f"{k}_r": np.float64 for k in predictors}, **({"scored": bool} if same_samples else {})}
        return pd.DataFrame({c: pd.Series(dtype=d) for c, d in cols.items()})
    return pd.DataFrame(rows)


def emg_person_table(trial_table, names):
    """Per person the median r over its (trial, muscle) pairs, per predictor name; with a column scored
    (emg_trial_table, same_samples) also pairs_scored, the pairs every predictor is scored on."""
    cols = [f"{k}_r" for k in names]
    g = trial_table.groupby("person_id", sort=True)
    out = g[cols].median().reset_index()
    out.insert(1, "pairs", g.size().values)
    if "scored" in trial_table.columns:
        out.insert(2, "pairs_scored", g["scored"].sum().astype(int).values)
    return out


def person_bootstrap(values, statistic=np.mean, n=BOOT_N, seed=SEED, level=0.95):
    """Percentile interval of a statistic over persons (one value per person, persons resampled with
    replacement n times); (nan, nan) without values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), size=(int(n), len(v)))
    boot = np.asarray([statistic(v[i]) for i in idx])
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return float(lo), float(hi)


def summary(person_grf, person_emg, grf_names, emg_names, emg_gain=False):
    """{line: {predictor: {point, interval, n_persons}}}: grf_rmse, grf_r and grf_r_gain (a predictor's r
    minus the Newton floor's r, per person) as means over persons, emg_r as the median over persons, each
    with its person-bootstrap interval. emg_gain (off by default; the sealed read sets it): also the line
    emg_r_gain of every EMG predictor but the template, per person its median r minus the template's (the
    same pairs when the person table comes from emg_trial_table with same_samples), median over persons."""
    out = {"grf_rmse_pct_bw": {}, "grf_r": {}, "grf_r_gain": {}, "emg_r": {}}

    def col(name):
        return person_grf[name].to_numpy(dtype=np.float64) if len(person_grf) else np.zeros(0)

    for k in grf_names:
        cols = [("grf_rmse_pct_bw", col(f"{k}_rmse_pct_bw")), ("grf_r", col(f"{k}_r"))]
        if k != "newton":
            cols.append(("grf_r_gain", col(f"{k}_r") - col("newton_r")))
        for line, v in cols:
            v = v[np.isfinite(v)]
            out[line][k] = {"point": float(v.mean()) if len(v) else float("nan"),
                            "interval": person_bootstrap(v, np.mean), "n_persons": int(len(v))}
    def ecol(name):
        return person_emg[name].to_numpy(dtype=np.float64) if len(person_emg) else np.zeros(0)

    for k in emg_names:
        v = ecol(f"{k}_r")
        v = v[np.isfinite(v)]
        out["emg_r"][k] = {"point": float(np.median(v)) if len(v) else float("nan"),
                           "interval": person_bootstrap(v, np.median), "n_persons": int(len(v))}
    if emg_gain:
        out["emg_r_gain"] = {}
        for k in emg_names:
            if k == "template" or "template" not in emg_names:
                continue
            v = ecol(f"{k}_r") - ecol("template_r")
            v = v[np.isfinite(v)]
            out["emg_r_gain"][k] = {"point": float(np.median(v)) if len(v) else float("nan"),
                                    "interval": person_bootstrap(v, np.median), "n_persons": int(len(v))}
    return out


# ---------------------------------------------------------------- metric 3D
def yaw_aligned(pred, true, windows):
    """Pelvis-relative predicted joints (T, J, 3) turned about +y by one yaw per window, fitted in closed form
    (mdfm.gates.g1.yaw_fit) over the entries finite in both pelvis-relative arrays; true (T, J, 3) in the lab
    frame; windows (T,) int, the window of each frame (-1: none, NaN out)."""
    from mdfm.gates import g1
    P = np.asarray(pred, dtype=np.float64) - np.asarray(pred, dtype=np.float64)[:, :1]
    Q = np.asarray(true, dtype=np.float64) - np.asarray(true, dtype=np.float64)[:, :1]
    w = np.asarray(windows, dtype=np.int64)
    if w.shape != (len(P),):
        raise ValueError(f"windows of shape {w.shape} for {len(P)} frames")
    out = np.full(P.shape, np.nan)
    for j in np.unique(w[w >= 0]):
        sel = w == j
        ok = np.isfinite(P[sel]).all(-1) & np.isfinite(Q[sel]).all(-1)
        th = g1.yaw_fit(np.where(ok[..., None], Q[sel], 0.0), np.where(ok[..., None], P[sel], 0.0))
        out[sel] = g1.turn_y(P[sel], th)
    return out


def mpjpe(pred, true, windows=None):
    """Mean per-joint position error (mm) of (T, J, 3) m arrays after removing each frame's root (joint 0),
    over the entries finite in both. windows (T,) int, optional: the window of each frame; the pelvis-relative
    prediction is then first turned by one yaw about +y per window (yaw_aligned), the rule's MPJPE frame (rule
    sections 4 and 6, section 8 item 10, closed 2026-10-05). Without windows no yaw is fitted."""
    p = pred - pred[:, :1] if windows is None else yaw_aligned(pred, true, windows)
    t = true - true[:, :1]
    d = np.linalg.norm(p - t, axis=-1)
    return 1000.0 * float(np.nanmean(d))


def pa_mpjpe(pred, true):
    """MPJPE (mm) after a per-frame similarity Procrustes alignment (rotation, translation, scale) of pred
    onto true over the joints finite in both."""
    errs = []
    for P, Q in zip(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)):
        ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        if ok.sum() < 3:
            continue
        A, B = P[ok] - P[ok].mean(0), Q[ok] - Q[ok].mean(0)
        U, S, Vt = np.linalg.svd(A.T @ B)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        s = float((S * np.diag(D)).sum() / (A * A).sum())
        errs.append(np.linalg.norm(s * A @ R - B, axis=1))
    return 1000.0 * float(np.mean(np.concatenate(errs))) if errs else float("nan")


def camera_seed(person_id, key, tag=CAMERA_TAG):
    """The metric-3D camera seed of a trial (rule section 4, decided 2026-10-05): the first 63 bits of
    SHA-256('<person_id>\\n<recording key>\\n<tag>'), keyed by person and recording since stroke recording keys
    repeat across persons."""
    h = hashlib.sha256(f"{person_id}\n{key}\n{tag}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def interp_frames(x, t, tq):
    """x (T, ...) at times t (T,) read at tq (Q,) by linear interpolation between the two nearest frames
    (clamped at the ends; a query on a frame reads that frame); NaN where a neighbour it weighs is NaN."""
    t = np.asarray(t, dtype=np.float64)
    tq = np.asarray(tq, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(t) == 1:
        return np.repeat(x[:1], len(tq), axis=0)
    i1 = np.clip(np.searchsorted(t, tq, side="right"), 1, len(t) - 1)
    i0 = i1 - 1
    a = np.clip((tq - t[i0]) / (t[i1] - t[i0]), 0.0, 1.0).reshape((-1,) + (1,) * (x.ndim - 1))
    with np.errstate(invalid="ignore"):
        out = (1.0 - a) * x[i0] + a * x[i1]
    return np.where(a == 0.0, x[i0], np.where(a == 1.0, x[i1], out))


def hold_fill(x):
    """x (Q, J, 3) with NaN rows filled per joint from the nearest frame where that joint is finite (a joint
    never finite takes joint 0's filled track); None when joint 0 is never finite."""
    x = np.array(x, dtype=np.float64)
    ok = np.isfinite(x).all(-1)
    if not ok[:, 0].any():
        return None
    for j in [0] + [j for j in range(x.shape[1]) if j != 0]:
        good = np.flatnonzero(ok[:, j])
        if not len(good):
            x[:, j] = x[:, 0]
            continue
        if len(good) > 1:
            k = np.clip(np.searchsorted(good, np.arange(len(x))), 1, len(good) - 1)
            lo, hi = good[k - 1], good[k]
            idx = np.arange(len(x))
            near = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi)
        else:
            near = np.full(len(x), good[0])
        x[~ok[:, j], j] = x[near[~ok[:, j]], j]
    return x


def camera_render(world, t, seed, fps=CAMERA_FPS, family=CAMERA_FAMILY, tracker=None):
    """The metric-3D input of one trial (rule section 4): the true joints world (T, 15, 3) m y up (NaN where
    invalid) at the pose times t (T,) s, read at camera frames fps from the first pose frame on (interp_frames;
    a joint missing at either neighbour is missing), held from its nearest present frame for the render only
    (hold_fill), projected by one static camera of camera_render.sample_camera (family) aimed at the trial-mean
    pelvis, then render_pose2d with the tracker process (default TrackerProcess), both from one torch
    generator seeded with seed (camera_seed). A missing truth joint, and every joint of a frame without its
    pelvis, is undetected. None when the pelvis is never valid; else {coords (Q, 15, 3) float32, the 2D
    record's normalized pixels with z = 0 and 0 where undetected; valid (Q, 15); t (Q,) s on the recording
    clock; fps; scale (px per torso unit); seed; family; info {frames, detected_share, truth_missing_share,
    scale_px, camera_KRt}}."""
    from mdfm.factory.camera_render import render_pose2d, sample_camera
    t = np.asarray(t, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    fin = np.isfinite(world).all(-1)
    if not len(t) or not fin[:, 0].any():
        return None
    tc = t[0] + np.arange(0.0, (t[-1] - t[0]) + 1e-9, 1.0 / fps)
    xc = interp_frames(world, t, tc)
    missing = ~np.isfinite(xc).all(-1)
    filled = hold_fill(xc)
    if filled is None:
        return None
    target = np.nanmean(world[fin[:, 0], 0], axis=0)
    gen = torch.Generator().manual_seed(int(seed))
    cam = sample_camera(gen, torch.as_tensor(target, dtype=torch.float64), family=family, fps=fps)
    record, _ = render_pose2d(torch.as_tensor(filled, dtype=torch.float64), cam, gen, tracker)
    valid = np.asarray(record["valid_mask"], dtype=bool) & ~missing & ~missing[:, :1]
    coords = np.zeros(np.asarray(record["coords"]).shape, dtype=np.float32)
    coords[..., :2] = np.where(valid[..., None], np.asarray(record["coords"])[..., :2], 0.0)
    params = torch.cat([cam.K.reshape(-1), cam.R.reshape(-1), cam.t.reshape(-1)]).tolist()
    return {"coords": coords, "valid": valid, "t": tc, "fps": float(record["fps"]), "scale": float(record["scale"]),
            "seed": int(seed), "family": family,
            "info": {"frames": int(len(tc)), "detected_share": float(valid.mean()) if valid.size else 0.0,
                     "truth_missing_share": float(missing.mean()) if missing.size else 0.0,
                     "scale_px": float(record["scale"]), "camera_KRt": params}}


def _left(line, source, person, key, stream, reason, samples, fps, min_samples_=None):
    fps = float(fps) if fps is not None else float("nan")
    return {"line": line, "source": source, "person_id": person, "key": key, "stream": stream, "reason": reason,
            "samples": int(samples), "fps": fps, "seconds": int(samples) / fps if fps > 0 else float("nan"),
            "min_samples": min_samples_}


def metric3d_trials(trials, predictor, yaw_per_window=True, left_out=None, stats=None, source=STROKE):
    """The metric-3D round trip of GRF trials (rule section 4). Per trial: camera_render of its world joints with
    the seed camera_seed(person, key); predictor(trial, camera) gives None (left out: its input gives no window),
    joints (T, 15, 3) m at the trial's pose frames, or (joints, windows) with windows (T,) int the window of each
    frame (-1: none). The prediction is taken pelvis-relative; yaw_per_window (on by default, the rule's MPJPE
    frame, section 8 item 10, closed 2026-10-05; False gives the lab-frame diagnostic) turns it by one yaw about
    +y per window (yaw_aligned), which needs windows (ValueError without them).
    A trial without a valid pelvis or without a prediction is left out, appended to left_out (LEFT_OUT_COLUMNS,
    line metric3d) and counted in stats. Returns [{person_id, key, seed, pred, truth (pelvis-relative, lab
    frame), windows, **camera info}]."""
    stats = collections.Counter() if stats is None else stats
    lo = [] if left_out is None else left_out
    out = []
    for tr in trials:
        seed = camera_seed(tr.person, tr.key)
        world = np.asarray(tr.world, dtype=np.float64)
        cam = camera_render(world, tr.t, seed)
        if cam is None:
            lo.append(_left("metric3d", source, tr.person, tr.key, "camera", "camera_no_pelvis", len(tr.t), tr.fps))
            stats["metric3d_left_out_camera_no_pelvis"] += 1
            continue
        got = predictor(tr, cam)
        if got is None:
            lo.append(_left("metric3d", source, tr.person, tr.key, "camera", "camera_no_prediction",
                            len(cam["t"]), cam["fps"]))
            stats["metric3d_left_out_camera_no_prediction"] += 1
            continue
        joints, windows = got if isinstance(got, tuple) else (got, None)
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != world.shape:
            raise ValueError(f"{tr.person} {tr.key}: predicted joints {joints.shape} for truth {world.shape}")
        if yaw_per_window:
            if windows is None:
                raise ValueError(f"{tr.person} {tr.key}: the per-window yaw needs the window of each frame")
            pred = yaw_aligned(joints, world, windows)
        else:
            pred = joints - joints[:, :1]
        stats["metric3d_trials"] += 1
        out.append({"person_id": tr.person, "key": tr.key, "seed": seed, "pred": pred, "truth": world - world[:, :1],
                    "windows": None if windows is None else np.asarray(windows, dtype=np.int64), **cam["info"]})
    return out


def metric3d_tables(entries):
    """(per trial, per person) DataFrames of metric3d_trials entries: frames scored (truth pelvis valid and a
    prediction, i.e. a joint finite in both), joint entries, MPJPE and PA-MPJPE (mm, of the aligned entries),
    detected 2D share; per person pooled over its trials' frames."""
    import pandas as pd
    trial_rows, by = [], collections.defaultdict(list)
    for e in entries:
        P, Q = e["pred"], e["truth"]
        both = np.isfinite(P).all(-1) & np.isfinite(Q).all(-1)
        frames = both.any(-1)
        trial_rows.append({"person_id": e["person_id"], "key": e["key"], "seed": str(e["seed"]),
                           "frames": int(frames.sum()), "joints": int(both.sum()), "camera_frames": e["frames"],
                           "detected_share": e["detected_share"],
                           "mpjpe_mm": mpjpe(P[frames], Q[frames]) if frames.any() else float("nan"),
                           "pa_mpjpe_mm": pa_mpjpe(P[frames], Q[frames]) if frames.any() else float("nan")})
        if frames.any():
            by[e["person_id"]].append((P[frames], Q[frames], e))
    person_rows = []
    for person, items in sorted(by.items()):
        P = np.concatenate([x[0] for x in items])
        Q = np.concatenate([x[1] for x in items])
        person_rows.append({"person_id": person, "trials": len(items), "frames": int(len(P)),
                            "detected_share": float(np.mean([x[2]["detected_share"] for x in items])),
                            "mpjpe_mm": mpjpe(P, Q), "pa_mpjpe_mm": pa_mpjpe(P, Q)})
    tcols = ["person_id", "key", "seed", "frames", "joints", "camera_frames", "detected_share", "mpjpe_mm",
             "pa_mpjpe_mm"]
    pcols = ["person_id", "trials", "frames", "detected_share", "mpjpe_mm", "pa_mpjpe_mm"]
    return pd.DataFrame(trial_rows, columns=tcols), pd.DataFrame(person_rows, columns=pcols)


def metric3d_summary(person_table):
    """{mpjpe_mm, pa_mpjpe_mm: {point, interval, n_persons}}: the mean over persons with the person bootstrap
    interval (the reported metric3d line, no pass or kill)."""
    out = {}
    for col in ("mpjpe_mm", "pa_mpjpe_mm"):
        v = person_table[col].to_numpy(dtype=np.float64) if len(person_table) else np.zeros(0)
        v = v[np.isfinite(v)]
        out[col] = {"point": float(v.mean()) if len(v) else float("nan"), "interval": person_bootstrap(v, np.mean),
                    "n_persons": int(len(v))}
    return out


# ---------------------------------------------------------------- results and the sealed read
def gate_results(summ, model, rule_sha256, checkpoint_sha256="", train_persons_sha256="", template_persons_sha256="",
                 emg_gain=False):
    """GateResult lines of mdfm.gates.common from a summary(): the model's point when `model` names a
    predictor, else not evaluated; the Newton floor and the template as trivial baselines.
    train_persons_sha256: the scored model's training persons; template_persons_sha256: the template's.
    emg_gain (off by default; the sealed read sets it, rule revision 2026-10-05): a fourth line emg_r_gain
    (the summary's, built with summary(..., emg_gain=True)), pass at least EMG_GAIN (delta, 0.03: rule section
    8 item 9, closed 2026-10-05) and kill below it, and emg_r passes only when it passes. When the summary has
    no emg_r_gain point (or EMG_GAIN were None), the line is not evaluated and emg_r cannot pass."""
    from mdfm.gates import common
    nan = float("nan")

    def get(line, name, what="point"):
        if not name:
            return nan if what == "point" else None
        return summ.get(line, {}).get(name, {}).get(what, nan if what == "point" else None)

    rm, rn = get("grf_rmse_pct_bw", model), get("grf_rmse_pct_bw", "newton")
    gain = get("grf_r_gain", model)
    if model and not math.isfinite(gain):                   # a summary without the gain line
        gain = get("grf_r", model) - get("grf_r", "newton")
    cond = (gain >= R_GAIN) if math.isfinite(gain) else None
    pm = get("emg_r", model)
    n_grf = summ["grf_rmse_pct_bw"].get("newton", {}).get("n_persons", 0)
    n_emg = summ["emg_r"].get("template", {}).get("n_persons", 0)

    def iv(line):
        v = get(line, model, "interval")
        return tuple(v) if v is not None else None

    econd, eextra = (), {}
    if emg_gain:
        egain = get("emg_r_gain", model)
        econd = ((egain >= EMG_GAIN) if (EMG_GAIN is not None and math.isfinite(egain)) else None,)
        eextra = {"r_gain_over_template": egain, "r_gain_line": EMG_GAIN}
    out = [
        common.GateResult(GATE, "grf_rmse", rm, iv("grf_rmse_pct_bw"), GRF_PASS, GRF_KILL,
                          common.verdict(rm if model else None, GRF_PASS, GRF_KILL, conditions=(cond,)), rn,
                          rule_sha256, checkpoint_sha256, train_persons_sha256, n_grf, unit="percent BW",
                          extra={"r_gain_over_newton": gain, "r_gain_line": R_GAIN}),
        common.GateResult(GATE, "grf_r_gain", gain, iv("grf_r_gain"), R_GAIN, R_GAIN,
                          common.verdict(gain if model else None, R_GAIN, R_GAIN, higher_is_better=True),
                          0.0, rule_sha256, checkpoint_sha256, train_persons_sha256, n_grf, unit="r",
                          higher_is_better=True),
        common.GateResult(GATE, "emg_r", pm, iv("emg_r"), EMG_PASS, EMG_KILL,
                          common.verdict(pm if model else None, EMG_PASS, EMG_KILL, higher_is_better=True,
                                         conditions=econd),
                          get("emg_r", "template"), rule_sha256, checkpoint_sha256, train_persons_sha256, n_emg,
                          unit="r", higher_is_better=True,
                          extra={"template_persons_sha256": template_persons_sha256, **eextra}),
    ]
    if emg_gain:
        n_gain = summ.get("emg_r_gain", {}).get(model, {}).get("n_persons", n_emg) if model else n_emg
        v = ("not evaluated" if EMG_GAIN is None else
             common.verdict(egain if model else None, EMG_GAIN, EMG_GAIN, higher_is_better=True))
        out.append(common.GateResult(GATE, "emg_r_gain", egain, iv("emg_r_gain"), EMG_GAIN, EMG_GAIN, v, 0.0,
                                     rule_sha256, checkpoint_sha256, train_persons_sha256, n_gain, unit="r",
                                     higher_is_better=True,
                                     extra={"delta": EMG_GAIN,
                                            "delta_status": ("not set: the line is not evaluated" if EMG_GAIN is None
                                                             else "docs/rules/G4_RULE.md sections 1 and 6, section 8 "
                                                                  "item 9 (closed 2026-10-05)"),
                                            "template_persons_sha256": template_persons_sha256}))
    return out


def _memo(predictors):
    """The predictors {name: callable(trial)} with each value computed once per trial object (the trial is kept
    with its value, so its id is never reused while the cache lives)."""
    cache = {}

    def once(name, f):
        def g(trial):
            k = (name, id(trial))
            if k not in cache:
                cache[k] = (trial, f(trial))
            return cache[k][1]
        return g

    return {name: once(name, f) for name, f in predictors.items()}


def evaluate(stroke_persons, emg_persons, template_persons, grf_predictors=None, emg_predictors=None,
             allow_lockbox=False, stats=None, min_s=None, same_samples=False, emg_gain=False, info=None,
             low_peak=False):
    """Run both lines on the given persons (module docstring). The Newton floor and the template (built on
    template_persons, which must not meet emg_persons) always run beside the given predictors. Returns
    (grf trials, person GRF table, EMG trial table, person EMG table, template, summary).
    min_s and same_samples (emg_trial_table) and emg_gain (summary) are off by default; read_lockbox sets them
    (min_s = EMG_MIN_S), the EMG lines of the rule revision of 2026-10-05. The template is built on every
    paired trial of template_persons. stats counts emg_left_out_under_min_s (with min_s) and
    emg_pairs_unscored (with same_samples). info (a dict, optional) receives left_out (the GRF trials of a
    patient without a body mass, grf_trials, and the EMG trials left out, LEFT_OUT_COLUMNS dicts),
    grf_persons_no_mass, grf_trials_no_mass (those GRF trials without their plate truth, grf_trials no_mass: the
    metric-3D round trip still scores them), emg_trials_all and emg_trials_kept, and with min_s
    emg_template_all_trials, the template's emg_r over every paired trial (a diagnostic that decides nothing, rule
    section 2).
    low_peak (off by default; read_lockbox sets it, rule section 2 (b), item 3; needs info): info receives
    grf_low_peak, the secondary GRF row without the plate series whose peak at the scored samples stays below
    LOW_PEAK_BW, every GRF predictor and the Newton floor recomputed on its own samples (each predictor is called
    once per trial for both rows): {min_peak_bw, person_grf, summary (grf_rmse_pct_bw, grf_r, grf_r_gain),
    series (LOW_PEAK_COLUMNS), series_left_out, samples_left_out, patients (with a series left out)}; stats
    counts grf_low_peak_series_left_out. It decides nothing."""
    import pandas as pd
    if set(template_persons) & set(emg_persons):
        raise ValueError("template persons overlap the scored persons")
    if low_peak and info is None:
        raise ValueError("the low-peak secondary row is written into info: pass info")
    stats = collections.Counter() if stats is None else stats
    grf_pred = {"newton": newton_predict, **(grf_predictors or {})}
    if low_peak:
        grf_pred = _memo(grf_pred)
    lo_grf, no_mass = [], []
    gtr = grf_trials(stroke_persons, allow_lockbox=allow_lockbox, stats=stats, left_out=lo_grf, no_mass=no_mass)
    person_grf = grf_person_table(gtr, grf_pred)
    if low_peak:
        series = []
        person_low = grf_person_table(gtr, grf_pred, min_peak_bw=LOW_PEAK_BW, low_peak=series)
        s_low = summary(person_low, pd.DataFrame(), list(grf_pred), [])
        stats["grf_low_peak_series_left_out"] += len(series)
        info["grf_low_peak"] = {"min_peak_bw": LOW_PEAK_BW, "person_grf": person_low,
                                "summary": {k: s_low[k] for k in ("grf_rmse_pct_bw", "grf_r", "grf_r_gain")},
                                "series": pd.DataFrame(series, columns=list(LOW_PEAK_COLUMNS)),
                                "series_left_out": len(series),
                                "samples_left_out": int(sum(r["samples"] for r in series)),
                                "patients": sorted({r["person_id"] for r in series})}
    template = build_template(emg_trials(template_persons, stats=collections.Counter()))
    emg_pred = {"template": lambda tr: template_predict(template, tr.phase), **(emg_predictors or {})}
    etr = emg_trials(emg_persons, allow_lockbox=allow_lockbox, stats=stats, keep_record=bool(emg_predictors))
    lo = []
    trial_emg = emg_trial_table(etr, emg_pred, min_s=min_s, same_samples=same_samples, left_out=lo)
    if min_s is not None:
        stats["emg_left_out_under_min_s"] += len(lo)
    if same_samples:
        stats["emg_pairs_unscored"] += int((~trial_emg["scored"].astype(bool)).sum()) if len(trial_emg) else 0
    person_emg = emg_person_table(trial_emg, list(emg_pred))
    if info is not None:
        info.setdefault("left_out", []).extend(lo_grf + lo)
        info["grf_persons_no_mass"] = sorted({r["person_id"] for r in lo_grf})
        info["grf_trials_no_mass"] = no_mass
        info["emg_trials_all"], info["emg_trials_kept"] = len(etr), len(etr) - len(lo)
        if min_s is not None:
            every = emg_person_table(emg_trial_table(etr, {"template": emg_pred["template"]}), ["template"])
            info["emg_template_all_trials"] = summary(pd.DataFrame(), every, [], ["template"])["emg_r"]["template"]
    return gtr, person_grf, trial_emg, person_emg, template, summary(person_grf, person_emg, list(grf_pred),
                                                                     list(emg_pred), emg_gain=emg_gain)


def truth_code_files():
    """The files whose code makes the G4 truth and baselines: this harness, the raw root tracks and body
    mass, the C3D reader, the pairing rules and the force and EMG kernels (the read log's code hash)."""
    return [os.path.abspath(f) for f in (__file__, mocap_raw.__file__, mocap_raw.C3D.__file__, sources.__file__,
                                         force.__file__, emg_kernel.__file__)]


def metric3d_code_files(yaw_per_window=True):
    """The files whose code makes the metric-3D input and frame: the camera draw and render, the projection and
    the 2D normalization, and with the per-window yaw (the rule's frame, item 10, the default as in
    metric3d_trials; False for the lab-frame diagnostic) the closed-form fit of mdfm/gates/g1.py."""
    from mdfm.factory import camera_render
    from mdfm.kernels import camera as camera_kernel
    from mdfm.kernels import pose as pose_kernel
    files = [camera_render.__file__, camera_kernel.__file__, pose_kernel.__file__]
    if yaw_per_window:
        from mdfm.gates import g1
        files.append(g1.__file__)
    return [os.path.abspath(f) for f in files]


def assert_disjoint(scored, train_persons):
    """{set name: 0} when no scored person ({set name: person ids}) is a training person of the scored model;
    canonical.Refused otherwise, and when train_persons holds no person (a check against nothing is none).
    train_persons: the checkpoint's training lists, {list name: person ids} or one iterable of person ids."""
    if isinstance(train_persons, (str, bytes)):
        raise TypeError("train_persons is a list of person ids or {list name: ids}, not a string")
    if isinstance(train_persons, dict):
        trained = set().union(*[{str(p) for p in v} for v in train_persons.values()]) if train_persons else set()
    else:
        trained = {str(p) for p in train_persons}
    if not trained:
        raise canonical.Refused("no training person given: the disjointness check needs the checkpoint's lists")
    hits = {name: sorted({str(p) for p in persons} & trained) for name, persons in scored.items()}
    hits = {k: v for k, v in hits.items() if v}
    if hits:
        raise canonical.Refused("scored persons among the checkpoint's training persons: "
                                + "; ".join(f"{k} {v[:3]}" for k, v in sorted(hits.items())))
    return {name: 0 for name in scored}


def check_train_persons(train_persons, train_persons_sha256):
    """{source: persons} of the checkpoint's training lists given to the sealed read, checked (rule section 2,
    section 8 item 7, closed 2026-10-05): a dict {source: person ids}, the lists the checkpoint stores
    (infer.load_checkpoint persons), with at least one person, whose SHA-256 by the checkpoint's own rule
    (mdfm.train.provenance.list_hashes, 'all': the sorted distinct '<source>\\t<id>' lines joined by newlines)
    equals train_persons_sha256, the value the checkpoint records (infer.train_persons_sha256).
    canonical.Refused otherwise: lists not given per source, a source whose ids are one string, no person, or
    another hash."""
    from mdfm.train import provenance
    if not isinstance(train_persons, dict):
        raise canonical.Refused("the sealed read needs the checkpoint's training lists per source, {source: ids} "
                                f"(rule section 2), not {type(train_persons).__name__}")
    for s, ids in train_persons.items():
        if isinstance(ids, (str, bytes)):
            raise canonical.Refused(f"training list {s!r} is one string, not a list of person ids")
    lists = {str(s): sorted({str(p) for p in ids}) for s, ids in train_persons.items()}
    if not any(lists.values()):
        raise canonical.Refused("no training person given: the sealed read needs the checkpoint's lists")
    got = provenance.list_hashes(lists)["all"]
    if got != str(train_persons_sha256):
        raise canonical.Refused(f"the training lists hash to {got}, not to the checkpoint's training-person "
                                f"SHA-256 {train_persons_sha256}")
    return {s: len(v) for s, v in sorted(lists.items())}


SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def checkpoint_identity(checkpoint):
    """(checkpoint SHA-256, training lists {source: ids}, training-person SHA-256) of the scored checkpoint, drawn
    from the checkpoint as mdfm.eval.infer.load_checkpoint gives it (infer.Loaded: sha256, persons, the lists the
    checkpoint stores, and persons_sha256['all'], the training-person SHA-256 it records; load_checkpoint has
    verified the stored lists against the stored hashes), so that the sealed read takes the identity and the
    lists from one object and not from strings a caller types (rule section 2, item 7). canonical.Refused when
    the object lacks one of them or a SHA-256 is not 64 lowercase hex digits. An object built by hand is not told
    apart from a loaded one, as the predictors are callables the caller gives: the trust boundary of the read."""
    sha = getattr(checkpoint, "sha256", None)
    hashes = getattr(checkpoint, "persons_sha256", None)
    psha = hashes.get("all") if isinstance(hashes, dict) else None
    if not (isinstance(sha, str) and SHA256_HEX.match(sha) and isinstance(psha, str) and SHA256_HEX.match(psha)
            and hasattr(checkpoint, "persons")):
        raise canonical.Refused("the sealed read takes the loaded checkpoint (infer.load_checkpoint): its sha256, "
                                f"persons and persons_sha256['all'], not a {type(checkpoint).__name__}")
    return sha, checkpoint.persons, psha


def read_lockbox(checkpoint, grf_predictor, emg_predictor, reread=False, reason="", code_files=(), *,
                 metric3d_predictor=None):
    """The sealed G4 read (plan task B4.21): refused unless docs/rules/G4_RULE.md is registered for G4
    (common.rule) and the read is logged once for this checkpoint and rule (common.log_lockbox_read) with
    the code hash of truth_code_files() plus code_files (and metric3d_code_files(True) with a metric-3D
    predictor), both before any lockbox record opens.
    checkpoint (rule section 2, item 7): the scored checkpoint as infer.load_checkpoint gives it; the checkpoint
    SHA-256, the training lists {source: ids} and the training-person SHA-256 the checkpoint records are drawn
    from it (checkpoint_identity), and the lists are checked against that hash (check_train_persons).
    grf_predictor: callable(GRFTrial) -> (bw, contact), the model's per-foot vertical GRF (T, 2) in BW and its
    contact probabilities (T, 2) at the trial's pose frames; the harness applies the model's flight rule
    (flight_predictor, rule section 4, item 11). emg_predictor: callable(EMGTrial) -> (T, M) log-envelopes.
    metric3d_predictor (None: not run): callable(GRFTrial, camera) -> None or (joints (T, 15, 3), windows (T,)),
    scored in the rule's frame, one yaw about +y per window (metric3d_trials, yaw_per_window True; item 10).
    Steps, in order: the rule; the checkpoint's identity and training lists (checkpoint_identity,
    check_train_persons); the scored sets from the index rows (no record) asserted disjoint from the training
    lists (assert_disjoint); all refused before the read is logged; the logged read; evaluate with the rule's
    lines (min_s = EMG_MIN_S, same_samples, emg_gain, low_peak; the body mass of grf_trials's default, no
    plate-derived mass) and gate_results with the emg_r_gain line; the metric-3D round trip (camera_seed(person,
    recording key)) on every paired stroke trial, those of a patient dropped from the GRF lines for want of a mass
    included (the line needs no mass; rule sections 1, 3 and 4), grouped by person. Returns (GateResult list,
    evaluate() outputs, info): info holds stats (with grf_model_frames_plate and grf_model_flight_frames_plate),
    left_out (GRF trials without a body mass, EMG trials under EMG_MIN_S and metric-3D trials left out),
    grf_persons_no_mass, grf_trials_no_mass, grf_low_peak (the secondary row, deciding nothing), emg_trials_all,
    emg_trials_kept, emg_template_all_trials, checkpoint_sha256, train_persons ({source: persons}),
    train_persons_sha256, disjoint_from_training_persons, template_persons_sha256, and metric3d (None when not run;
    else trials, persons, mpjpe_mm, pa_mpjpe_mm, yaw_per_window True)."""
    from mdfm.gates import common
    _, rule_sha = common.rule(RULE_PATH, gate=GATE)
    checkpoint_sha256, train_persons, train_persons_sha256 = checkpoint_identity(checkpoint)
    lists = check_train_persons(train_persons, train_persons_sha256)
    stroke, gait = stroke_patients(lockbox=True), gait120_persons("lockbox")
    template = gait120_persons("train")
    disjoint = assert_disjoint({"stroke_lockbox": stroke, "gait120_lockbox": gait}, train_persons)
    code = [*truth_code_files(), *(metric3d_code_files(True) if metric3d_predictor is not None else []),
            *code_files]
    common.log_lockbox_read(GATE, checkpoint_sha256, rule_sha, reread=reread, reason=reason,
                            code_sha256=common.code_sha256(code))
    stats, info = collections.Counter(), {"left_out": []}
    res = evaluate(stroke, gait, template, grf_predictors={"model": flight_predictor(grf_predictor, stats)},
                   emg_predictors={"model": emg_predictor}, allow_lockbox=True, stats=stats, min_s=EMG_MIN_S,
                   same_samples=True, emg_gain=True, info=info, low_peak=True)
    template_sha = sources.person_hash(template)
    results = gate_results(res[-1], "model", rule_sha, checkpoint_sha256, train_persons_sha256, template_sha,
                           emg_gain=True)
    info["metric3d"] = None
    if metric3d_predictor is not None:
        every = sorted([*res[0], *info["grf_trials_no_mass"]], key=lambda tr: tr.person)   # stable, by person
        entries = metric3d_trials(every, metric3d_predictor, yaw_per_window=True, left_out=info["left_out"],
                                  stats=stats)
        trials, persons = metric3d_tables(entries)
        info["metric3d"] = {"trials": trials, "persons": persons, **metric3d_summary(persons),
                            "yaw_per_window": True}
    info.update(stats=dict(stats), checkpoint_sha256=checkpoint_sha256, train_persons=lists,
                train_persons_sha256=train_persons_sha256, disjoint_from_training_persons=disjoint,
                template_persons_sha256=template_sha)
    return results, res, info
