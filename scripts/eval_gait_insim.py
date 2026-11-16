"""B4.19c: stride, speed and initial contacts of a checkpoint in its own synthetic domain (plan task B4.19c;
design docs/MFM_DESIGN_2026-09-24.md, section 11: sim-to-real for IMU is load-bearing).

Question: the simulator-only G3 estimate fails on the real weargait dev controls (scripts/eval_g3_dev.py). Is the
stride and speed failure already present on synthetic renders of held-out bodies (a model deficiency), or does it
appear on real sensors only (the sim-to-real gap)?

Takes: the walking takes of the AMASS dev archives held by the dev body store (data/body_store.py): task 'walk' of
docs/rules/task_vocabulary.yaml (data/windows.task_name; HDM05 names code scenes and carry no walking label),
treadmill takes left out; plus the KIT and BMLrub takes of reports/amass_walks/takes.csv whose archive is a dev
archive (amass_index.check; the splits are whole-archive and both are train, so none). Never a train, calib,
lockbox or SMPL-X take: every archive read must be a dev archive, BodyStore refuses a take or person outside its
split, and a person among the checkpoint's training persons refuses the run. Rows: 'forward' (primary), the takes
whose id carries a forward walking style of mdfm.gates.g3.WALK_STYLES (the takes of the G3 readout self-check);
'all', every walking take (backward, sneaking, shuffling and moonwalk included).

Render: each whole take in one frame (BodyStore.windows over the take) through factory/synth.SynthSampler, float32
on the GPU as in training, with the concrete xsens13 layout (13 Xsens sites, acc and gyro, magnetometer zeroed and
masked; xsens13 mount prior, per-site white noise, biases and saturation of the xsens13 device family: the renderer
and device model of training), one render seed per take (infer.seed_of(seed, 'render', take)). The raw stream is
band-copied whole (window_store.band_copy) and cut by infer.spans and infer._window into 10 s windows every 5 s, the
last one on the end, exactly as infer.record_windows cuts a real record; source 'synthetic', the take's sex and task.

Inference: infer.infer with the settings of infer.G3Predictor and scripts/eval_g3_dev.py (20 refinement steps of
256 points per stream, shape pooled per take, seed infer.seed_of(seed, take)); bodies stitched on the stream clock
(g3.stitched_body), frame f at the time of sample f.

Truth: the take's knots at the kept sample times (KnotBatch.kinematics in float64), so model and truth share every
frame; contacts by frames.contact_labels with the exact velocities and the floor at y = 0 (the rule of the
body-store labels the model was trained on).

Predictors, readouts of mdfm/readouts/gait.py, lines of mdfm/gates/g3.py:
- model: ICs from the stitched contact probabilities, strides on the stitched body, speeds over the true passes
  (the readouts of g3.body_predictions);
- model_at_true_ics: the model body read at the true ICs (event timing removed, the body geometry alone);
- gyro_on_truth (the floor): the true body read at the ICs of gait.shank_gyro_ic on the rendered raw shank_L and
  shank_R gyroscopes (device noise included); its speed is the truth's by construction and is not scored;
- constant: per take, the person-weighted mean scored true stride and pass speed of the row's other persons (the
  analogue of the G3 population baseline);
- model_legs (reports/bench/root_motion_diagnosis.md, C4 and fix 7): the model's ICs, and strides and pass speeds
  from the stance-foot no-slip speed of the decoded pose (gait.stance_speed of each window's pelvis-relative joints
  with its decoded contacts, stitched; the translation output is not read): a stride is the mean speed over it
  times its duration, a pass the mean speed over it. A diagnostic here; it becomes a G3 readout only if the owner
  registers it in docs/rules/G3_RULE.md before any lockbox read;
- bound_speed (--bound): the known-good in-sim bound of the root-motion diagnosis (the signal investigation's
  baseline.py): HistGradientBoostingRegressor speed on lower-back and shank features of 2 s segments, fitted on
  xsens13 renders of train-store windows only (fit_bound: the train walking takes and a subject-balanced
  mixture), read at the shank-gyroscope ICs (strides) and over the true passes; and per window the dead
  reckoning of that speed along the lower-back gyroscope heading (bound_pos_err_xz_m).
The reference: every true IC; true strides (successive ICs of a foot, ankle chord); true passes: the ICs of both feet
sorted, split where two lie more than gait.PASS_GAP_S apart, each run cut into consecutive segments of PASS_STEPS
steps (sharing their end ICs, a shorter remainder dropped); pass speed gait.pass_speed. Strides and passes are scored
only in steady straight walking, the analogue of the walkway's straight passes (steady_straight): the true pelvis
chord at least g3.STRAIGHT of its path over the interval, and the chord speed at least g3.STEADY times the take
speed (the median chord speed of the take's strides), the rule of g3.steady_straight on stride intervals; strides
of turns, stops and shuffles in place, a few centimetres long, would otherwise dominate a percentage error.
PASS_STEPS is 2 (one stride): the AMASS dev walks turn inside their capture volume, so segments of several strides
are rarely straight (takes.csv counts the strides and passes of each take and the scored ones). Lines:
g3.stride_lines (stride MAE cm and MAPE percent over the scored strides matched on both ends within
g3.STRIDE_MATCH_S), g3.speed_line (MAE m/s over the scored passes), g3.ic_line (median absolute IC error ms per
person over every true IC, after the predictor's offset fitted on the
row: median of predicted minus true over the nearest predictions within g3.IC_MATCH_S, the G3 dev fit; and with no
offset); per person, then the mean over persons, person-clustered bootstrap intervals; signed biases (predicted
minus true) as person-weighted means.

Root, per window: the decoded window (model window frame) against the truth over the same frames put in the
window frame of training (frames.heading_frame at knot n // 2: floor y = 0, pelvis heading +z, pelvis x and z 0):
pelvis position error (3D, horizontal, vertical; mean over knots), root orientation error (the geodesic of the
training log's root_orient_deg) and its yaw and tilt parts, the model's root yaw at the centre knot (0 for the
truth by construction), the mean absolute yaw of each body about its own centre-knot yaw (how far it turns within
the window), the walking speed of each body (mean norm of the horizontal pelvis velocity averaged over a centred
1 s window, gait.HS_SMOOTH_S, which removes the sway) and their ratio, and the local pose error (joints 1 to 21
relative to the pelvis in each body's heading frame, losses.heading_relative: the training log's joint error).

Answer rule (fixed before the first run, on the forward row): 'in_domain' (the failure is present in the synthetic
domain) when the model's stride MAPE or speed MAE lies beyond its G3 kill line (40 percent, 0.15 m/s) or is not
below the constant baseline's; 'sim_to_real' when both lie within their pass lines (25 percent, 0.08 m/s) and
below the constant's; 'partial' otherwise. The steady straight scoring of strides was added after that first full
run, in which every stride was scored and the constant baseline read a stride MAPE above 100 percent (strides of a
few centimetres in the denominator); the rule was left as it was.

Never read: a lockbox, EVAL or CODY record, a real record. The real line is read from the report of
scripts/eval_g3_dev.py for the same checkpoint (its SHA-256 checked) when it exists: --real names it, else the
first existing of reports/gates/g3/dev_<checkpoint name>/g3_dev.json (the phase B report) and
reports/gates/g3_sim/dev_<checkpoint name>/g3_sim.json (the phase A report) (resolve_real). When none exists, the
report says 'real line absent' (in summary.md, the real column included, and in insim.json, real_absent with the
paths looked for).

Sites (--sites SITE[,SITE]): the walks are rendered with only the listed sites of the xsens13 layout (site_layout:
the same layout name, mount prior and device family, the listed sites in record order), so the model reads those
sites only; default all 13, unchanged. The gyro_on_truth floor needs a shank gyroscope and is left out when no
shank is listed; --bound needs lower_back, shank_L and shank_R.

Writes <out> (default reports/gates/g3_sim/insim_<checkpoint name>/, with the suffix _sites<n> under --sites):
summary.md, insim.json, takes.csv (per take), windows.csv (per window), _provenance.json.

Run (mdfm environment):
  <conda>/envs/mdfm/python.exe scripts/eval_gait_insim.py --checkpoint <run>/ckpt_A_<step>.pt
"""
import argparse
import csv
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.body import frames  # noqa: E402
from mdfm.body.kinematics import KnotBatch  # noqa: E402
from mdfm.body.rotations import geodesic, rot6d_to_matrix, rot_y, yaw_angle  # noqa: E402
from mdfm.body.joints import fk  # noqa: E402   (the program's fk; mdfm.body.smplh re-exports the same object)
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.data import windows as W  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.eval import infer as I  # noqa: E402
from mdfm.gates import common, g3  # noqa: E402
from mdfm.readouts import gait  # noqa: E402

RATE = 100.0
SOURCE = "synthetic"            # Window.source of the renders, as in training
LAYOUT = "xsens13"
WALK_TASK = "walk"
PASS_STEPS = 2                  # [cfg] steps per true pass (3 ICs: one stride; module docstring)
KIN_CHUNK = 65536               # truth kinematics points per call
ROWS = ("forward", "all")
PRIMARY_ROW = "forward"
PREDICTORS = ("model", "model_legs", "model_at_true_ics", "gyro_on_truth", "constant", "bound_speed")
LINE_ORDER = ("stride_mae_cm", "stride_mape_pct", "stride_bias_cm", "speed_mae_m_s", "speed_bias_m_s",
              "ic_median_ms", "ic_median_ms_raw")
SHANKS = {"L": "shank_L", "R": "shank_R"}
AMASS_WALKS_CSV = os.path.join(TREE, "reports", "amass_walks", "takes.csv")
WINDOW_COLUMNS = ("root_pos_err_m", "root_pos_err_xz_m", "root_y_err_m", "root_orient_deg", "root_yaw_err_deg",
                  "root_tilt_err_deg", "root_yaw_centre_deg", "turn_true_deg", "turn_pred_deg",
                  "pelvis_speed_true_m_s", "pelvis_speed_pred_m_s", "pelvis_speed_ratio", "pose_err_mm")
BOUND_COLUMNS = ("bound_pos_err_xz_m", "bound_yaw_err_deg")


# ---------------------------------------------------------------- takes
def is_forward(take_id):
    """True when the take id carries a forward walking style of g3.WALK_STYLES (as g3.amass_dev_walks)."""
    return any(s in str(take_id) for s in g3.WALK_STYLES)



def _amass_index():
    """mdfm.data.amass_index, imported where an AMASS take is read (no AMASS module at the top: plan phase E step 2)."""
    from mdfm.data import amass_index
    return amass_index

def style_of(take_id):
    """The take's file name without its motion suffix."""
    return str(take_id).rsplit("/", 1)[-1].replace("_poses.npz", "").replace("_stageii.npz", "")


def amass_walks_rows(path=AMASS_WALKS_CSV):
    """(rows {subset, take, split} of reports/amass_walks/takes.csv, the take ids whose archive is a dev archive).
    amass_index.check refuses a CODY, unknown, SMPL-X or lockbox archive name."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    splits = {s: _amass_index().check(s) for s in sorted({r["subset"] for r in rows})}
    out = [{"subset": r["subset"], "take": r["take"], "split": splits[r["subset"]]} for r in rows]
    return out, [r["take"] for r in out if r["split"] == "dev"]


def select_takes(takes, extra=()):
    """The takes the check reads, from a dev BodyStore.takes table: task 'walk' or a take id in extra, treadmill
    takes left out (counted in attrs['treadmill']). Every archive must be a dev archive (amass_index.check: CODY,
    unknown, SMPL-X and lockbox names are refused, train and calib too), no name CODY (infer.refuse_cody). Returns
    DataFrame(i (row of takes), take_id, archive, person_id, gender, n_knots, style, forward)."""
    extra = set(extra)
    sel = takes[(takes["task_name"] == WALK_TASK) | takes["take_id"].isin(extra)]
    missing = sorted(extra - set(takes["take_id"]))
    if missing:
        raise ValueError(f"{len(missing)} dev takes of reports/amass_walks/takes.csv are not in the dev body store")
    for a in sorted(set(sel["archive"])):
        split = _amass_index().check(a)
        if split != "dev":
            raise Refused(f"{a} is a {split} archive: the in-domain check reads dev archives only")
    for r in sel.itertuples():
        I.refuse_cody(r.archive, r.take_id, r.person_id, r.subject_ns)
    tread = sel["treadmill"].astype(int) != 0
    keep = sel[~tread]
    out = pd.DataFrame({"i": keep.index.to_numpy(np.int64), "take_id": keep["take_id"].to_numpy(),
                        "archive": keep["archive"].to_numpy(), "person_id": keep["person_id"].to_numpy(),
                        "gender": keep["gender"].to_numpy(), "n_knots": keep["n_knots"].to_numpy(np.int64)})
    out["style"] = [style_of(t) for t in out["take_id"]]
    out["forward"] = [is_forward(t) for t in out["take_id"]]
    out.attrs["treadmill"] = int(tread.sum())
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- sites (--sites)
def parse_sites(text):
    """The site names of a --sites value 'SITE[,SITE]' (spaces ignored), as given; None for None. A name outside
    imu_render.SITES, a repeated name or an empty list raises ValueError."""
    from mdfm.factory.imu_render import SITES
    if text is None:
        return None
    names = [s.strip() for s in str(text).split(",") if s.strip()]
    if not names:
        raise ValueError("--sites lists no site")
    bad = [s for s in names if s not in SITES]
    if bad:
        raise ValueError(f"unknown sites {bad} (sites: {', '.join(SITES)})")
    if len(set(names)) != len(names):
        raise ValueError(f"a site is listed twice in {names}")
    return tuple(names)


def site_layout(layout, sites):
    """The concrete layout restricted to `sites` (None: the layout itself, unchanged): its IMU stream keeps the
    listed sites, in the layout's record order, with their channel mask rows; name, mount prior, device family and
    the other streams are kept, so that the renderer and the model see the same layout family with fewer sites.
    A site absent from the layout raises ValueError."""
    from dataclasses import replace
    if sites is None:
        return layout
    imu = [s for s in layout.streams if s.modality == "imu"]
    if len(imu) != 1:
        raise ValueError(f"{layout.name}: --sites needs a layout with one IMU stream")
    s = imu[0]
    missing = [x for x in sites if x not in s.sites]
    if missing:
        raise ValueError(f"sites {missing} are not in the {layout.name} layout ({', '.join(s.sites)})")
    keep = [k for k, name in enumerate(s.sites) if name in set(sites)]
    sub = replace(s, sites=tuple(s.sites[k] for k in keep), mask=tuple(s.mask[k] for k in keep))
    return replace(layout, streams=tuple(sub if x is s else x for x in layout.streams))


def default_out(name, sites=None, tree=TREE):
    """reports/gates/g3_sim/insim_<checkpoint name>, with the suffix _sites<n> when --sites lists n sites."""
    return os.path.join(tree, "reports", "gates", "g3_sim", f"insim_{name}" + (f"_sites{len(sites)}" if sites else ""))


# ---------------------------------------------------------------- render and windows
def render_take(sampler, store, i, layout, seed):
    """(BodyWindows of the whole take i in one frame, the raw rendered windows.Window): SynthSampler.render without
    the band copy (the caller copies the whole stream, as record_windows does with a record)."""
    from mdfm.factory.synth import RenderSpec
    n = int(store.takes["n_knots"].iloc[int(i)])
    bodies = store.windows([int(i)], [0], win_s=n / RATE)
    win = sampler.render(bodies, [RenderSpec(layout)], int(seed), band=False)[0]
    return bodies, win


def model_windows(raw, person_id, key, cov, source=SOURCE):
    """(windows, spans, fps, times (N,) s on the take clock of the kept samples, raw index of each kept sample) of a
    raw rendered IMU stream: band-copied whole, then cut by infer.spans and infer._window as infer.record_windows
    cuts a real record (windows start at their first sample, t = 0)."""
    t64 = float(raw.truth["t_first_s"]) + np.arange(len(raw.t), dtype=np.float64) / float(raw.fps)
    if len(t64) and np.abs(t64 - raw.t.astype(np.float64)).max() > 1e-3:
        raise ValueError(f"{key}: stream times disagree with t_first + k / fps")
    b, info = WS.band_copy(raw, return_info=True)
    tb = t64[info.idx]
    sp = I.spans(len(tb), b.fps)
    wins = [W.validate(I._window(source, person_id, key, b, tb, s, n, cov)) for s, n in sp]
    return wins, sp, float(b.fps), tb, np.asarray(info.idx)


def truth_on_clock(bodies, bank, times, chunk=KIN_CHUNK):
    """The true body of BodyWindows row 0 at times (N,) s from its first knot, float64 on the CPU: joints (N, 22, 3)
    m, global rotations (N, 22, 3, 3) and contact labels (N, 2) bool of frames.contact_labels (exact velocities,
    floor y = 0)."""
    f64 = torch.float64
    with torch.no_grad():
        rest = bank.rest(bodies.betas[:1].to(f64), bodies.gender[:1].to(torch.int64))
        kb = KnotBatch(bodies.rot6d[:1].to(f64), bodies.transl[:1].to(f64), rest, rate=RATE,
                       lengths=bodies.lengths[:1])
        t = torch.as_tensor(np.asarray(times, dtype=np.float64))
        w = torch.zeros(len(t), dtype=torch.int64)
        P, V, R = [], [], []
        for a in range(0, len(t), chunk):
            kin = kb.kinematics(w[a:a + chunk], t[a:a + chunk])
            P.append(kin.p)
            V.append(kin.v)
            R.append(kin.R)
        J, V, R = torch.cat(P), torch.cat(V), torch.cat(R)
        contact = frames.contact_labels(J[None], RATE, vel=V[None], floor=0.0)[0]
    return J.numpy(), R.numpy(), contact.numpy()


def shank_gyros(raw, idx):
    """{"L": (N, 3), "R": (N, 3)} deg/s of the raw stream's shank sites at the kept samples idx (NaN where
    invalid); a shank absent from the layout is left out."""
    from mdfm.factory.imu_render import SITES
    sites = [int(s) for s in raw.meta["site_ids"]]
    out = {}
    for foot, name in SHANKS.items():
        sid = SITES.index(name)
        if sid in sites and raw.chan_mask[sites.index(sid), 3:6].all():
            j = sites.index(sid)
            g = raw.values[idx, j, 3:6].astype(np.float64)
            g[~raw.valid[idx, j]] = np.nan
            out[foot] = g
    return out


# ---------------------------------------------------------------- passes and window errors
def interval_path(t0, t1, pelvis_xz, rate=RATE):
    """(straightness, chord speed m/s) of the horizontal pelvis path pelvis_xz (T, 2) m between times t0 < t1 (s,
    frames round(t rate)): chord over path length (NaN for a path of length 0) and chord over (t1 - t0)."""
    f0, f1 = int(gait.frame_of(t0, rate)), int(gait.frame_of(t1, rate))
    p = np.asarray(pelvis_xz[f0:f1 + 1], dtype=np.float64)
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(p[-1] - p[0]))
    return (chord / path if path > 0 else float("nan")), chord / (float(t1) - float(t0))


def passes_from_ics(ics, pelvis_xz, rate=RATE, steps=PASS_STEPS, gap_s=gait.PASS_GAP_S, straight=g3.STRAIGHT):
    """True passes (module docstring): ics {"L": times, "R": times} s, pelvis_xz (T, 2) m -> DataFrame(pass_id, t0,
    t1, n_footfalls, straightness (pelvis chord over path over the segment), straight (at least `straight`),
    chord_speed_m_s)."""
    parts = [np.asarray(v, dtype=np.float64) for v in ics.values()]
    t = np.sort(np.concatenate(parts)) if parts else np.zeros(0)
    rows, k = [], 0
    cuts = np.flatnonzero(np.diff(t) > gap_s) + 1
    for run in np.split(t, cuts):
        for a in range(0, len(run) - steps, steps):
            t0, t1 = float(run[a]), float(run[a + steps])
            ratio, v = interval_path(t0, t1, pelvis_xz, rate)
            rows.append((k, t0, t1, steps + 1, ratio, bool(np.isfinite(ratio) and ratio >= straight), v))
            k += 1
    return pd.DataFrame(rows, columns=["pass_id", "t0", "t1", "n_footfalls", "straightness", "straight",
                                       "chord_speed_m_s"])


def steady_straight(strides, passes, pelvis_xz, rate=RATE, straight=g3.STRAIGHT, steady=g3.STEADY):
    """The scored reference of one take (module docstring), in place: strides (DataFrame t0, t1) gain straightness,
    chord_speed_m_s and scored; passes (passes_from_ics) gain scored. An interval is scored when it is straight
    (pelvis chord at least `straight` of its path) and steady (chord speed at least `steady` times the take speed,
    the median chord speed of the take's strides): the rule of g3.steady_straight on stride intervals. Returns the
    take speed (m/s, NaN without a stride)."""
    fl = [interval_path(a, b, pelvis_xz, rate) for a, b in zip(strides["t0"], strides["t1"])]
    strides["straightness"] = np.array([f[0] for f in fl], dtype=np.float64)
    strides["chord_speed_m_s"] = np.array([f[1] for f in fl], dtype=np.float64)
    v = float(np.median(strides["chord_speed_m_s"])) if len(strides) else float("nan")
    for df in (strides, passes):
        ok = np.zeros(len(df), bool)
        if math.isfinite(v) and len(df):
            ok = ((df["straightness"].to_numpy() >= straight) & (df["chord_speed_m_s"].to_numpy() >= steady * v))
        df["scored"] = ok
    return v


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def window_errors(pred_joints, pred_R, true_joints, true_R, rate=RATE):
    """Root and pose errors of one decoded window (module docstring): pred_joints (n, 22, 3) m and pred_R (n, 22, 3,
    3) global rotations in the model's window frame; true_joints, true_R over the same frames in any world frame,
    put in the window frame at knot n // 2. Returns {column of WINDOW_COLUMNS: value}."""
    from mdfm.train.losses import heading_relative
    f64 = torch.float64
    Jp, Rp = torch.as_tensor(pred_joints, dtype=f64), torch.as_tensor(pred_R, dtype=f64)
    Jt, Rt = torch.as_tensor(true_joints, dtype=f64)[None], torch.as_tensor(true_R, dtype=f64)[None]
    n = Jt.shape[1]
    R_yaw, origin = frames.heading_frame(Jt, Rt[:, :, 0], n // 2)
    Jt, Rt = frames.apply_frame(Jt, R_yaw, origin)[0], frames.rotate_frame(Rt, R_yaw)[0]
    d = Jp[:, 0] - Jt[:, 0]
    c = n // 2
    yp, yt = yaw_angle(Rp[:, 0]), yaw_angle(Rt[:, 0])
    tilt = geodesic(rot_y(-yp) @ Rp[:, 0], rot_y(-yt) @ Rt[:, 0])
    vt, vp = walking_speed(Jt[:, 0].numpy(), rate), walking_speed(Jp[:, 0].numpy(), rate)
    pose = (heading_relative(Jp, Rp) - heading_relative(Jt, Rt)).norm(dim=-1).mean()
    deg = lambda x: math.degrees(float(x))  # noqa: E731
    return {"root_pos_err_m": float(d.norm(dim=-1).mean()), "root_pos_err_xz_m": float(d[:, [0, 2]].norm(dim=-1).mean()),
            "root_y_err_m": float(d[:, 1].abs().mean()),
            "root_orient_deg": deg(geodesic(Rp[:, 0], Rt[:, 0]).mean()),
            "root_yaw_err_deg": deg(_wrap(yp - yt).abs().mean()), "root_tilt_err_deg": deg(tilt.mean()),
            "root_yaw_centre_deg": deg(_wrap(yp[c]).abs()),
            "turn_true_deg": deg(_wrap(yt - yt[c]).abs().mean()), "turn_pred_deg": deg(_wrap(yp - yp[c]).abs().mean()),
            "pelvis_speed_true_m_s": vt, "pelvis_speed_pred_m_s": vp,
            "pelvis_speed_ratio": vp / vt if vt > 0 else float("nan"), "pose_err_mm": 1000.0 * float(pose)}


def walking_speed(pelvis, rate=RATE, smooth_s=gait.HS_SMOOTH_S):
    """Mean norm (m/s) of the horizontal pelvis velocity (central differences of pelvis (n, 3) m) averaged over a
    centred window of smooth_s (the direction-of-progression filter of gait.heel_strikes); NaN under 2 frames."""
    from scipy.ndimage import uniform_filter1d
    p = np.asarray(pelvis, dtype=np.float64)[:, [0, 2]]
    if len(p) < 2:
        return float("nan")
    v = np.gradient(p, 1.0 / rate, axis=0)
    v = uniform_filter1d(v, max(1, int(round(smooth_s * rate))), axis=0, mode="nearest")
    return float(np.linalg.norm(v, axis=1).mean())


def decoded_pose(body, bank):
    """Joint positions (n, 22, 3) m and global rotations (n, 22, 3, 3) of an infer.WindowBody, float64, posed on
    the SkeletonBank rest joints of its pooled shape and gender."""
    f64 = torch.float64
    with torch.no_grad():
        rest = bank.rest(torch.as_tensor(body.beta, dtype=f64)[None], torch.tensor([int(body.gender)]))[0]
        pos, Rg = fk(rot6d_to_matrix(torch.as_tensor(body.rot6d, dtype=f64)), torch.as_tensor(body.transl, dtype=f64),
                     rest)
    return pos.numpy(), Rg.numpy()


# ---------------------------------------------------------------- one take
def _ics_frame(tid, pid, ics):
    return pd.DataFrame([(tid, pid, f, float(t)) for f, ts in ics.items() for t in ts],
                        columns=["trial_id", "person_id", "foot", "t_s"])


def _strides_frame(tid, pid, st):
    return pd.DataFrame({"trial_id": tid, "person_id": pid, "foot": st["foot"].to_numpy(), "t0": st["t0"].to_numpy(),
                         "t1": st["t1"].to_numpy(), "length_m": st["length_m"].to_numpy()})


def run_take(model, sampler, store, take, layout, bank, seed=20260929, refine_steps=I.STEPS_EVAL,
             points=I.POINTS_EVAL, batch_windows=16, pool=True, bound=None):
    """Render, infer and read one take (a row of select_takes). Returns {info: dict, windows: [dict], tables:
    {ref_ics, ref_strides, ref_passes, model_ics, model_strides, model_speeds, model_true_ic_strides, gyro_ics and
    gyro_strides (when the layout has a shank gyroscope), legs_strides, legs_speeds, and with bound (fit_bound)
    bound_strides, bound_speeds}: DataFrames with trial_id and person_id}."""
    tid, pid = str(take["take_id"]), str(take["person_id"])
    I.refuse_cody(tid, pid)
    t0 = time.perf_counter()
    bodies, win = render_take(sampler, store, int(take["i"]), layout, I.seed_of(seed, "render", tid))
    raw = win.streams[0]
    wins, sp, fps, tb, idx = model_windows(raw, pid, tid, win.cov)
    if abs(fps - RATE) > 1e-9 or not np.array_equal(idx, np.arange(len(raw.t))):
        raise ValueError(f"{tid}: the band copy of the {LAYOUT} stream is not on the {RATE:g} Hz clock of the body")
    t_render = time.perf_counter() - t0
    st = {}
    wb = I.infer(model, wins, refine_steps=refine_steps, points=points, groups=[tid] * len(wins) if pool else None,
                 batch_windows=batch_windows, seed=I.seed_of(seed, tid), stats=st)
    starts = [int(round(s / fps * RATE)) for s, _ in sp]
    body = g3.stitched_body(tid, [b.joints for b in wb], [b.contact for b in wb], None, starts, RATE)
    J, R, contact = truth_on_clock(bodies, bank, tb)
    if body.contact.shape[0] != J.shape[0]:
        raise ValueError(f"{tid}: model clock of {body.contact.shape[0]} frames, truth of {J.shape[0]}")
    # readouts
    ics_t = gait.initial_contacts(contact.astype(np.float64), RATE)
    st_t = gait.strides(J, ics_t, rate=RATE)
    ps = passes_from_ics(ics_t, J[:, gait.PELVIS][:, [0, 2]])
    take_speed = steady_straight(st_t, ps, J[:, gait.PELVIS][:, [0, 2]])
    ps["speed_m_s"] = [gait.pass_speed(J, a, b, rate=RATE) for a, b in zip(ps["t0"], ps["t1"])]
    ics_m = gait.initial_contacts(body.contact, RATE)
    st_m = gait.strides(body.joints, ics_m, body.window_of_frame, RATE)
    st_mt = gait.strides(body.joints, ics_t, body.window_of_frame, RATE)
    sp_m = [gait.pass_speed(body.joints, a, b, body.window_of_frame, RATE) for a, b in zip(ps["t0"], ps["t1"])]
    gy = shank_gyros(raw, idx)
    ics_g = {f: gait.shank_gyro_ic(g, RATE) for f, g in gy.items()}
    st_g = gait.strides(J, ics_g, rate=RATE) if gy else None
    ps.insert(0, "person_id", pid)
    ps.insert(0, "trial_id", tid)
    ref_st = _strides_frame(tid, pid, st_t)
    for c in ("straightness", "chord_speed_m_s", "scored"):
        ref_st[c] = st_t[c].to_numpy()
    legs_st, legs_sp = speed_readouts(tid, pid, legs_speed(wb, starts, RATE), ics_m, ps, RATE)
    tables = {"ref_ics": _ics_frame(tid, pid, ics_t), "ref_strides": ref_st, "ref_passes": ps,
              "model_ics": _ics_frame(tid, pid, ics_m), "model_strides": _strides_frame(tid, pid, st_m),
              "model_speeds": pd.DataFrame({"trial_id": tid, "person_id": pid, "pass_id": ps["pass_id"].to_numpy(),
                                            "speed_m_s": np.asarray(sp_m, dtype=np.float64)}),
              "model_true_ic_strides": _strides_frame(tid, pid, st_mt)}
    if gy:                                      # a layout without shank gyroscope (--sites) has no gyro floor
        tables["gyro_ics"], tables["gyro_strides"] = _ics_frame(tid, pid, ics_g), _strides_frame(tid, pid, st_g)
    tables["legs_strides"], tables["legs_speeds"] = legs_st, legs_sp
    brows = None
    if bound is not None:
        bspeed, brows = bound_take(bound, raw, J, R, sp, RATE)
        tables["bound_strides"], tables["bound_speeds"] = speed_readouts(tid, pid, bspeed, ics_g, ps, RATE)
    # root and pose per window
    wrows = []
    for k, ((s, n), b) in enumerate(zip(sp, wb)):
        m = len(b.joints)
        pos, Rg = decoded_pose(b, bank)
        e = window_errors(pos, Rg, J[s:s + m], R[s:s + m])
        wrows.append({"take_id": tid, "person_id": pid, "window": k, "start_s": s / fps, "knots": m, **e,
                      **(brows[k] if brows else {})})
    info = {"take_id": tid, "person_id": pid, "archive": str(take["archive"]), "style": str(take["style"]),
            "forward": bool(take["forward"]), "n_knots": int(take["n_knots"]), "frames": int(J.shape[0]),
            "windows": len(wins), "seconds_render": t_render, "seconds_infer": st.get("seconds"),
            "refine_first": st.get("refine_first"), "refine_last": st.get("refine_last"),
            "gyro_sites": "".join(sorted(gy)), "noise_scale": raw.truth.get("noise_scale"), "take_speed_m_s": take_speed,
            "strides": int(len(ref_st)), "scored_strides": int(ref_st["scored"].sum()), "passes": int(len(ps)),
            "straight_passes": int(ps["straight"].sum()), "scored_passes": int(ps["scored"].sum())}
    return {"info": info, "windows": wrows, "tables": tables}


# ---------------------------------------------------------------- model_legs (root-motion diagnosis, fix 7)
def legs_speed(bodies, starts, rate=RATE):
    """(N,) m/s on the stream clock: gait.stance_speed of each window's pelvis-relative joints (the root translation
    is not read) and decoded contacts, each frame from the window where it is most central (gait.stitch)."""
    per = [gait.stance_speed(np.asarray(b.joints) - np.asarray(b.joints)[:, :1], b.contact, rate) for b in bodies]
    return gait.stitch(per, None, starts)[0]


def interval_mean(speed, t0, t1, rate=RATE):
    """The mean of a speed curve (N,) over the frames of [t0, t1] (NaN frames left out; NaN without one)."""
    f0, f1 = int(gait.frame_of(t0, rate)), int(gait.frame_of(t1, rate))
    seg = np.asarray(speed, dtype=np.float64)[max(f0, 0):f1 + 1]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if len(seg) else float("nan")


def speed_readouts(tid, pid, speed, ics, passes, rate=RATE):
    """(strides, speeds) DataFrames of a speed curve on the stream clock: a stride between successive ICs of one
    foot (both on the clock) has the length mean speed over it times its duration (a straight stride's pelvis
    chord); a pass the mean speed over it. The G3 readouts of a predictor that has a speed and no path."""
    n = len(speed)
    rows = []
    for foot in gait.FEET:
        t = np.sort(np.asarray(ics.get(foot, ()), dtype=np.float64))
        for a, b in zip(t[:-1], t[1:]):
            f0, f1 = int(gait.frame_of(a, rate)), int(gait.frame_of(b, rate))
            if 0 <= f0 < f1 < n:
                rows.append((tid, pid, foot, float(a), float(b), interval_mean(speed, a, b, rate) * (b - a)))
    st = pd.DataFrame(rows, columns=["trial_id", "person_id", "foot", "t0", "t1", "length_m"])
    st = st.sort_values(["t0", "foot"], kind="stable").reset_index(drop=True)
    sp = pd.DataFrame({"trial_id": tid, "person_id": pid, "pass_id": passes["pass_id"].to_numpy(),
                       "speed_m_s": np.asarray([interval_mean(speed, a, b, rate) for a, b in
                                                zip(passes["t0"], passes["t1"])], dtype=np.float64)})
    return st, sp


# ---------------------------------------------------------------- the in-sim bound (--bound)
BOUND_SEG = 200                 # samples (2 s at 100 Hz) per speed segment, hop BOUND_SEG // 2
BOUND_SITES = ("lower_back", "shank_L", "shank_R")
BOUND_WALK_MIN = 400            # knots of a train walking take read for the bound, at least
BOUND_MIX = 2500                # subject-balanced train windows added to the walking windows [cfg]
BOUND_SEED = 777


def _up_dir(acc, rate=RATE):
    from scipy.ndimage import uniform_filter1d
    m = uniform_filter1d(acc, int(rate), axis=0, mode="nearest")
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)


def _dom_freq(x, rate=RATE, lo=0.5, hi=3.5):
    x = x - x.mean()
    n = 1024
    f = np.fft.rfftfreq(n, 1 / rate)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(P[m])]) if P[m].sum() > 0 else 0.0


def bound_site_columns(site_ids):
    """Columns (lower back, left shank, right shank) of BOUND_SITES in a stream of canonical site ids."""
    from mdfm.factory.imu_render import SITES
    ids = [int(s) for s in site_ids]
    return tuple(ids.index(SITES.index(name)) for name in BOUND_SITES)


def segment_features(imu, a, b, cols, rate=RATE):
    """Features of samples [a, b) of an IMU stream (T, S, >= 6), acc g and gyro deg/s in the sensor frame: of the
    lower back the vertical and horizontal acceleration spread, the horizontal mean, the gyro RMS and peak, the
    dominant step frequency; of each shank the gyro RMS and 95th percentile, the acc norm spread, the sagittal
    angle range, the sagittal rate peak and its dominant frequency (the signal investigation's baseline.py)."""
    x = np.asarray(imu[a:b], dtype=np.float64)
    lb, sl, sr = cols
    acc, gyr = x[:, lb, :3], x[:, lb, 3:6]
    u = _up_dir(acc, rate)
    av = (acc * u).sum(1)
    ah = np.linalg.norm(acc - av[:, None] * u, axis=1)
    out = [av.std(), ah.std(), ah.mean(), np.sqrt((gyr ** 2).sum(1).mean()), np.abs(gyr).max(), _dom_freq(av, rate)]
    for s in (sl, sr):
        acc, gyr = x[:, s, :3], x[:, s, 3:6]
        gn = np.linalg.norm(gyr, axis=1)
        gc = gyr - gyr.mean(0)
        _, _, vt = np.linalg.svd(gc, full_matrices=False)
        g1 = gyr @ vt[0]
        ang = np.cumsum(g1 - g1.mean()) / rate
        out += [np.sqrt((gn ** 2).mean()), np.percentile(gn, 95), np.linalg.norm(acc, axis=1).std(),
                ang.max() - ang.min(), np.abs(g1).max(), _dom_freq(g1, rate)]
    return out


def true_speed(xz, rate=RATE, smooth_s=gait.HS_SMOOTH_S):
    """(T,) m/s: the norm of the horizontal pelvis velocity (central differences of xz (T, 2)) averaged over a
    centred window of smooth_s."""
    from scipy.ndimage import uniform_filter1d
    v = np.gradient(np.asarray(xz, dtype=np.float64), 1.0 / rate, axis=0)
    return np.linalg.norm(uniform_filter1d(v, max(1, int(round(smooth_s * rate))), axis=0, mode="nearest"), axis=1)


def stream_segments(imu, cols, n, seg=BOUND_SEG, rate=RATE):
    """(features, starts) of every BOUND_SEG-sample segment, hop BOUND_SEG // 2, of the first n samples."""
    F, S = [], []
    for a in range(0, n - seg + 1, seg // 2):
        F.append(segment_features(imu, a, a + seg, cols, rate))
        S.append(a)
    return np.asarray(F, dtype=np.float64).reshape(-1, 18), np.asarray(S, dtype=np.int64)


def speed_curve(pred, starts, n, seg=BOUND_SEG):
    """(n,) the per-sample speed of overlapping segment predictions (their mean where they overlap, linear
    between covered samples, NaN without a segment)."""
    s, c = np.zeros(n), np.zeros(n)
    for p, a in zip(pred, starts):
        s[a:a + seg] += p
        c[a:a + seg] += 1
    if not (c > 0).any():
        return np.full(n, np.nan)
    good = c > 0
    s[good] /= c[good]
    k = np.flatnonzero(good)
    s[~good] = np.interp(np.flatnonzero(~good), k, s[good])
    return s


def gyro_heading(imu, cols, n, centre, rate=RATE):
    """(n,) rad: the lower-back gyroscope projected on the up direction (1 s mean of the specific force),
    integrated by the trapezoid from the knot centre (0 there)."""
    lb = cols[0]
    acc = np.asarray(imu[:n, lb, :3], dtype=np.float64)
    gyr = np.radians(np.asarray(imu[:n, lb, 3:6], dtype=np.float64))
    r = (gyr * _up_dir(acc, rate)).sum(1)
    psi = np.concatenate([[0.0], np.cumsum((r[1:] + r[:-1]) / 2) / rate])
    return psi - psi[centre]


def dead_reckoning(speed, psi, centre, rate=RATE):
    """(n, 2) m: the horizontal displacement from the centre sample of a speed (n,) along the heading psi (n,)
    (sin psi, cos psi), trapezoid."""
    v = np.stack([speed * np.sin(psi), speed * np.cos(psi)], 1)
    p = np.concatenate([np.zeros((1, 2)), np.cumsum((v[1:] + v[:-1]) / 2, axis=0) / rate])
    return p - p[centre]


def fit_bound(store, sampler, layout, seed=BOUND_SEED, n_mix=BOUND_MIX, walk_min=BOUND_WALK_MIN, chunk=64, log=None):
    """The speed model of the in-sim bound (module docstring): HistGradientBoostingRegressor on segment_features
    of layout renders (SynthSampler.render, band copies) of train-store windows: every 10 s window every 5 s of
    the train walking takes (task 'walk', not treadmill, at least walk_min knots) plus n_mix subject-balanced
    windows (BodyStore.sample, numpy Generator seeded seed); targets the true speed (true_speed of the pelvis)
    over each segment. Only a train store is read (ValueError otherwise). Returns {model, segments, windows,
    persons, persons_sha256, split}."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from mdfm.factory.synth import RenderSpec
    if getattr(store, "split", None) != "train":
        raise ValueError(f"the in-sim bound is fitted on train-store renders only, not "
                         f"{getattr(store, 'split', None)!r}")
    t = store.takes
    rows = []
    walk = t
    if "task_name" in t.columns:
        walk = t[(t["task_name"] == WALK_TASK) & (t["treadmill"].astype(int) == 0)]
    for i in walk.index:
        n = int(t.at[i, "n_knots"])
        if n >= walk_min:
            rows += [(int(i), s) for s in range(0, max(n - 1000, 0) + 1, 500)]
    gen = np.random.default_rng(seed)
    if n_mix:
        b = store.sample(gen, int(n_mix))
        rows += [(int(i), int(s)) for i, s in zip(b.take.numpy(), b.start.numpy())]
    X, Y, persons = [], [], set()
    for c0 in range(0, len(rows), chunk):
        ch = rows[c0:c0 + chunk]
        bodies = store.windows([r[0] for r in ch], [r[1] for r in ch])
        wins = sampler.render(bodies, [RenderSpec(layout)] * len(ch), int(gen.integers(1 << 62)))
        for j, wd in enumerate(wins):
            s = wd.streams[0]
            cols = bound_site_columns(s.meta["site_ids"])
            L = int(bodies.lengths[j])
            n = min(len(s.t), L)
            if n < BOUND_SEG + 1:
                continue
            ts = true_speed(bodies.transl[j, :L][:, [0, 2]].double().numpy())
            F, S = stream_segments(s.values, cols, n)
            X.append(F)
            Y.append(np.asarray([ts[a:a + BOUND_SEG].mean() for a in S]))
            persons.add(str(bodies.person[j]))
        if log:
            log(f"bound: {min(c0 + chunk, len(rows))}/{len(rows)} windows rendered")
    X, Y = np.concatenate(X), np.concatenate(Y)
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X[ok], Y[ok])
    return {"model": model, "segments": int(ok.sum()), "windows": len(rows), "persons": sorted(persons),
            "persons_sha256": _amass_index().person_list_sha256(sorted(persons)), "split": "train"}


def bound_take(bound, raw, J, R, sp, rate=RATE):
    """The bound on one take: (speed curve (N,) on the stream clock from the fitted speed model, window rows
    {bound_pos_err_xz_m, bound_yaw_err_deg}: the lower-back gyroscope heading and the speed curve integrated from
    each window's centre (dead_reckoning) against the true pelvis and root yaw in the window frame of training
    (the root heading at the centre knot on +z, the pelvis there at 0))."""
    from mdfm.body.rotations import yaw_angle
    cols = bound_site_columns(raw.meta["site_ids"])
    n = min(len(raw.t), J.shape[0])
    F, S = stream_segments(raw.values, cols, n)
    speed = speed_curve(bound["model"].predict(F) if len(F) else np.zeros(0), S, n)
    yaw = yaw_angle(torch.as_tensor(np.asarray(R)[:n, 0], dtype=torch.float64)).numpy()
    rows = []
    for s, m in sp:
        m = min(m, n - s)
        c = m // 2
        ang = -float(yaw[s + c])
        d = np.asarray(J[s:s + m, 0][:, [0, 2]], dtype=np.float64) - np.asarray(J[s + c, 0][[0, 2]], dtype=np.float64)
        Xt = np.stack([math.cos(ang) * d[:, 0] + math.sin(ang) * d[:, 1],
                       -math.sin(ang) * d[:, 0] + math.cos(ang) * d[:, 1]], 1)
        psi = gyro_heading(raw.values[s:s + m], cols, m, c)
        Xb = dead_reckoning(speed[s:s + m], psi, c)
        yt = (yaw[s:s + m] - yaw[s + c] + math.pi) % (2 * math.pi) - math.pi
        e = (psi - yt + math.pi) % (2 * math.pi) - math.pi
        rows.append({"bound_pos_err_xz_m": float(np.linalg.norm(Xb - Xt, axis=1).mean()),
                     "bound_yaw_err_deg": math.degrees(float(np.abs(e).mean()))})
    return speed, rows


# ---------------------------------------------------------------- lines
def fit_offset(pred_ics, ref_ics, tol=g3.IC_MATCH_S):
    """(offset_s, n): median of predicted minus reference over the nearest predictions within tol (the G3 dev fit,
    here on the row's takes); g3.UNFITTED when none lies within tol."""
    d = g3.ic_differences(pred_ics, ref_ics, 0.0, tol)
    d = d[d["matched"]]
    return (float(np.median(d["diff_s"])), int(len(d))) if len(d) else g3.UNFITTED


def constant_predictions(strides, passes, name="constant"):
    """g3.Predictions of the constant baseline: each take's strides and passes predicted by the person-weighted
    mean true stride and pass speed of the other persons (leave one person out); NaN for a person alone."""
    s, p = strides.copy(), passes.copy()
    for pid in sorted(set(s["person_id"]) | set(p["person_id"])):
        so, po = strides[strides["person_id"] != pid], passes[passes["person_id"] != pid]
        vs = common.person_weighted_mean(so["length_m"], so["person_id"]) if len(so) else float("nan")
        vp = common.person_weighted_mean(po["speed_m_s"], po["person_id"]) if len(po) else float("nan")
        s.loc[s["person_id"] == pid, "length_m"] = vs
        p.loc[p["person_id"] == pid, "speed_m_s"] = vp
    return g3.Predictions(name, None, s[["trial_id", "foot", "t0", "t1", "length_m"]],
                          p[["trial_id", "pass_id", "speed_m_s"]])


def _bias(values, persons):
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    return {"point": common.person_weighted_mean(v[ok], np.asarray(persons)[ok]) if ok.any() else float("nan"),
            "n_events": int(ok.sum())}


def predictor_lines(pred, targets, ref_ics, offset=None):
    """{line: stats} and {table: DataFrame} of one predictor: stride lines (when it has strides), speed line (when it
    has speeds), IC lines after `offset` (s; a fitted one when None) and with none (when it has ICs)."""
    out, tabs = {}, {}
    if pred.strides is not None:
        lines, m = g3.stride_lines(pred, targets)
        out.update(lines)
        k = m[m["matched"]]
        out["stride_bias_cm"] = _bias(k["err_cm"], k["person_id"])
        tabs["strides"] = m
    if pred.speeds is not None:
        out["speed_mae_m_s"], m = g3.speed_line(pred, targets)
        k = m[m["matched"]]
        out["speed_bias_m_s"] = _bias(k["err_m_s"], k["person_id"])
        tabs["speeds"] = m
    if pred.ics is not None:
        off = fit_offset(pred.ics, ref_ics) if offset is None else (float(offset), 0)
        out["ic_median_ms"], tabs["ics"] = g3.ic_line(pred, ref_ics, off[0])
        out["ic_median_ms"]["offset_n"] = off[1]
        out["ic_median_ms_raw"], tabs["ics_raw"] = g3.ic_line(pred, ref_ics, 0.0)
    return out, tabs


def row_lines(tables, take_ids):
    """({predictor: {line: stats}}, {predictor: {table: DataFrame}}) over the takes of a row (module docstring)."""
    keep = set(take_ids)
    T = {k: v[v["trial_id"].isin(keep)].reset_index(drop=True) for k, v in tables.items()}
    passes = T["ref_passes"][T["ref_passes"]["scored"].astype(bool)].reset_index(drop=True)
    strides = T["ref_strides"][T["ref_strides"]["scored"].astype(bool)].reset_index(drop=True)
    persons = tuple(sorted(set(T["ref_strides"]["person_id"]) | set(T["ref_ics"]["person_id"])))
    tg = g3.Targets(persons, False, None, T["ref_ics"], None, strides, passes, None)
    scored = set(zip(passes["trial_id"], passes["pass_id"].astype(str)))
    def scored_speeds(key):
        ms = T[key]
        return ms[[(t, str(p)) in scored for t, p in zip(ms["trial_id"], ms["pass_id"])]][["trial_id", "pass_id",
                                                                                           "speed_m_s"]]
    ms = scored_speeds("model_speeds")
    cols = ["trial_id", "foot", "t0", "t1", "length_m"]
    preds = {"model": g3.Predictions("model", T["model_ics"][["trial_id", "foot", "t_s"]], T["model_strides"][cols],
                                     ms),
             "model_at_true_ics": g3.Predictions("model_at_true_ics", None, T["model_true_ic_strides"][cols])}
    if "gyro_ics" in T:                         # absent when the layout has no shank gyroscope (--sites)
        preds["gyro_on_truth"] = g3.Predictions("gyro_on_truth", T["gyro_ics"][["trial_id", "foot", "t_s"]],
                                                T["gyro_strides"][cols])
    preds["constant"] = constant_predictions(strides, passes)
    if "legs_strides" in T:
        preds["model_legs"] = g3.Predictions("model_legs", None, T["legs_strides"][cols], scored_speeds("legs_speeds"))
    if "bound_strides" in T:
        preds["bound_speed"] = g3.Predictions("bound_speed", None, T["bound_strides"][cols],
                                              scored_speeds("bound_speeds"))
    lines, tabs = {}, {}
    for name in PREDICTORS:
        if name in preds:
            lines[name], tabs[name] = predictor_lines(preds[name], tg, T["ref_ics"])
    counts = {"takes": len(keep), "persons": len(persons), "true_ics": int(len(T["ref_ics"])),
              "true_strides": int(len(T["ref_strides"])), "scored_strides": int(len(strides)),
              "true_passes": int(len(T["ref_passes"])), "scored_passes": int(len(passes)),
              "model_ics": int(len(T["model_ics"])),
              "model_strides": int(len(T["model_strides"])),
              "gyro_ics": int(len(T["gyro_ics"])) if "gyro_ics" in T else 0,
              "true_stride_m": common.person_weighted_mean(strides["length_m"], strides["person_id"])
              if len(strides) else float("nan"),
              "true_speed_m_s": common.person_weighted_mean(passes["speed_m_s"], passes["person_id"])
              if len(passes) else float("nan")}
    return lines, tabs, counts


def answer(lines, row=PRIMARY_ROW):
    """The answer rule of the module docstring on a row's lines: ('in_domain' | 'sim_to_real' | 'partial' |
    'undetermined', the four numbers read)."""
    m, c = lines[row]["model"], lines[row]["constant"]
    get = lambda d, k: float(d.get(k, {}).get("point", float("nan")))  # noqa: E731
    mape, speed = get(m, "stride_mape_pct"), get(m, "speed_mae_m_s")
    c_mape, c_speed = get(c, "stride_mape_pct"), get(c, "speed_mae_m_s")
    nums = {"model_stride_mape_pct": mape, "model_speed_mae_m_s": speed, "constant_stride_mape_pct": c_mape,
            "constant_speed_mae_m_s": c_speed}
    if not all(math.isfinite(v) for v in nums.values()):
        return "undetermined", nums
    (mp, mk), (sp, sk) = g3.LINES["stride_mape_pct"], g3.LINES["speed_mae_m_s"]
    if mape > mk or speed > sk or mape >= c_mape or speed >= c_speed:
        return "in_domain", nums
    if mape <= mp and speed <= sp:
        return "sim_to_real", nums
    return "partial", nums


def per_take(infos, tabs_all, windows):
    """takes.csv: per take, the counts; per predictor the take's matched shares, mean absolute stride error (cm),
    stride APE (percent) and speed error (m/s) over its matched scored events, and the median absolute IC error
    (ms, no offset) over its matched ICs only (an unmatched IC, infinite in the line, is left out here); the means
    of the window errors."""
    rows = []
    wdf = pd.DataFrame(windows)
    for info in infos:
        tid = info["take_id"]
        r = dict(info)
        for name in PREDICTORS:
            if name not in tabs_all:
                continue
            t = tabs_all[name]
            if "strides" in t:
                m = t["strides"][t["strides"]["trial_id"] == tid]
                k = m[m["matched"]]
                r[f"{name}_stride_matched_share"] = float(m["matched"].mean()) if len(m) else float("nan")
                r[f"{name}_stride_mae_cm"] = float(k["err_cm"].abs().mean()) if len(k) else float("nan")
                r[f"{name}_stride_mape_pct"] = float(k["ape_pct"].mean()) if len(k) else float("nan")
            if "speeds" in t:
                m = t["speeds"][t["speeds"]["trial_id"] == tid]
                k = m[m["matched"]]
                r[f"{name}_speed_mae_m_s"] = float(k["err_m_s"].abs().mean()) if len(k) else float("nan")
            if "ics_raw" in t:
                m = t["ics_raw"][t["ics_raw"]["trial_id"] == tid]
                k = m[m["matched"]]
                r[f"{name}_ic_matched_share"] = float(m["matched"].mean()) if len(m) else float("nan")
                r[f"{name}_ic_median_ms_raw"] = float(1000 * k["diff_s"].abs().median()) if len(k) else float("nan")
        s = tabs_all["model"]["strides"]
        s = s[s["trial_id"] == tid]
        p = tabs_all["model"]["speeds"]
        p = p[p["trial_id"] == tid]
        r["true_stride_m"] = float(s["length_m"].mean()) if len(s) else float("nan")      # scored strides
        r["true_speed_m_s"] = float(p["speed_m_s"].mean()) if len(p) else float("nan")
        r["model_speed_m_s"] = float(p["pred_m_s"].mean()) if len(p) else float("nan")
        w = wdf[wdf["take_id"] == tid]
        for c in WINDOW_COLUMNS + tuple(c for c in BOUND_COLUMNS if c in wdf.columns):
            r[f"win_{c}"] = float(w[c].mean()) if len(w) else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def window_summary(wdf, take_ids):
    """{column: {mean, median, p10, p90}} of the window errors of the takes."""
    w = wdf[wdf["take_id"].isin(set(take_ids))]
    out = {"windows": int(len(w))}
    for c in WINDOW_COLUMNS + tuple(c for c in BOUND_COLUMNS if c in w.columns):
        v = w[c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        out[c] = {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))} if len(v) else None
    return out


def real_lines(path, sha256):
    """The dev row of scripts/eval_g3_dev.py's g3_sim.json for the same checkpoint: {predictor: {line: stats}}, or
    None when the file is missing; a report of another checkpoint raises ValueError."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    if r.get("checkpoint", {}).get("sha256") != sha256:
        raise ValueError(f"{path} reports checkpoint {r.get('checkpoint', {}).get('sha256')}, not {sha256}")
    return {"path": common.rel(path), "lines": r["lines"]["dev"], "sets": {"dev_sha256": r["sets"]["dev_sha256"]}}


def real_candidates(name, tree=TREE):
    """The reports of scripts/eval_g3_dev.py the default --real looks for, in order: the phase B report
    reports/gates/g3/dev_<name>/g3_dev.json, then the phase A report reports/gates/g3_sim/dev_<name>/g3_sim.json."""
    return [os.path.join(tree, "reports", "gates", "g3", f"dev_{name}", "g3_dev.json"),
            os.path.join(tree, "reports", "gates", "g3_sim", f"dev_{name}", "g3_sim.json")]


def resolve_real(explicit, name, tree=TREE):
    """(the real report to read or None, the paths looked at): an explicit --real as given (real_lines reads None
    when it does not exist); else the first existing file of real_candidates."""
    if explicit:
        return explicit, [explicit]
    cands = real_candidates(name, tree)
    return next((p for p in cands if os.path.isfile(p)), None), cands


# ---------------------------------------------------------------- report
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "" if not iv else f" [{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def real_cell(st, nd=2):
    """Point and interval of a line read from g3_sim.json, where a non-finite value is stored as null: a null
    point over at least one person is the infinite value of an IC line (a person with more than half of its
    reference ICs unmatched), printed 'inf' as g3_sim.md prints it; over no person, 'n/a'."""
    p = st.get("point")
    if p is None and (st.get("n_persons") or 0) > 0:
        return "inf" + fmt_iv(st.get("interval"), nd)
    return f"{fmt(p, nd)}{fmt_iv(st.get('interval'), nd)}"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


ANSWER_TEXT = {
    "in_domain": "the stride and speed failure is already present in the model's own synthetic domain: a model "
                 "deficiency, there before any sim-to-real gap",
    "sim_to_real": "the stride and speed failure is absent in the model's own synthetic domain: it appears on real "
                   "sensors only (the sim-to-real gap of design section 11)",
    "partial": "the model is between the G3 pass and kill lines in its own synthetic domain: part of the failure is "
               "present there, the rest appears on real sensors",
    "undetermined": "undetermined: a line of the rule could not be evaluated",
}


def real_source(res):
    """Where the real line was read from, or the 'real line absent' text with the paths looked at."""
    if res.get("real"):
        return res["real"]["path"]
    looked = (res.get("real_absent") or {}).get("looked_for") or []
    return "nowhere: real line absent (" + ("no report at " + " or ".join(looked) if looked else "no report") + ")"


def sites_text(sites):
    """The layout sentence's addition under --sites ('' without)."""
    if not sites:
        return ""
    t = f" restricted to the sites {', '.join(sites)} (--sites)"
    if not any(x in sites for x in SHANKS.values()):
        t += ", without a shank gyroscope, so without the gyro_on_truth floor"
    return t


def markdown(res, raw_lines):
    ck, s = res["checkpoint"], res["sets"]
    L = ["# Gait readouts in the model's own synthetic domain (plan task B4.19c)", "",
         "Written by scripts/eval_gait_insim.py (renderer mdfm/factory/synth.py, inference mdfm/eval/infer.py, readouts "
         "mdfm/readouts/gait.py, lines mdfm/gates/g3.py); every number below is read from insim.json, takes.csv and "
         "windows.csv of this folder, the real line from " + real_source(res)
         + ". No lockbox, EVAL, CODY or real record was read.", "",
         f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} step "
         f"{ck['step']:,} of {ck['phase_steps']}. MoveAtlas HEAD {res['moveatlas_head']}.", "",
         f"Takes: {s['n_takes']} walking takes of the AMASS dev archives ({', '.join(f'{a} {n}' for a, n in s['archives'].items())}; "
         f"{s['n_persons']} persons, none among the checkpoint's training persons), {s['treadmill_left_out']} treadmill "
         f"takes left out; forward row {s['rows']['forward']['takes']} takes of {s['rows']['forward']['persons']} "
         f"persons. reports/amass_walks/takes.csv: {s['amass_walks']['rows']} takes "
         f"({', '.join(f'{k} {v}' for k, v in s['amass_walks']['by_split'].items())}), of them in a dev archive: "
         f"{s['amass_walks']['dev']}. Rendered whole in the {LAYOUT} layout"
         f"{sites_text(res['inference'].get('sites'))}, {res['inference']['windows']} windows of "
         f"{I.WIN_S:g} s every {I.HOP_S:g} s, {res['inference']['refine_steps']} refinement steps of "
         f"{res['inference']['points_per_stream']} points per stream, shape pooled "
         f"{res['inference']['shape_pooling']}.", ""]
    ans = res["answer"]
    n = ans["numbers"]
    fw = raw_lines[PRIMARY_ROW]
    L += ["## Answer", "",
          f"On the forward walks, {ANSWER_TEXT[ans['verdict']]}: rule '{ans['verdict']}' (module docstring). Model "
          f"stride MAPE {fmt(n['model_stride_mape_pct'])} percent against {fmt(n['constant_stride_mape_pct'])} for the "
          f"constant baseline, speed MAE {fmt(n['model_speed_mae_m_s'])} m/s against "
          f"{fmt(n['constant_speed_mae_m_s'])} (G3 lines: MAPE pass 25, kill 40; speed pass 0.08, kill 0.15)."]
    rl = res["real"]["lines"] if res["real"] else None
    if rl:
        L[-1] += (f" On the real dev controls the same checkpoint reads stride MAPE "
                  f"{fmt(rl['model']['stride_mape_pct']['point'])} percent and speed MAE "
                  f"{fmt(rl['model']['speed_mae_m_s']['point'])} m/s (population baseline "
                  f"{fmt(rl['population']['stride_mape_pct']['point'])} percent, "
                  f"{fmt(rl['population']['speed_mae_m_s']['point'])} m/s).")
    else:
        L[-1] += (" Real line absent: no report of scripts/eval_g3_dev.py for this checkpoint was found, so the "
                  "sim-to-real comparison is not made here.")
    ws = res["windows"][PRIMARY_ROW]
    ic = fw["model"]["ic_median_ms"]
    cf = res["counts"][PRIMARY_ROW]
    L += ["", f"Where the error sits (forward row). Model stride bias {fmt(fw['model']['stride_bias_cm']['point'])} cm "
          f"and speed bias {fmt(fw['model']['speed_bias_m_s']['point'])} m/s (predicted minus true; the scored true "
          f"strides average {fmt(cf['true_stride_m'])} m and the scored passes {fmt(cf['true_speed_m_s'])} m/s, "
          "person-weighted). Read at the true "
          f"ICs, the model body gives a stride MAPE of {fmt(fw['model_at_true_ics']['stride_mape_pct']['point'])} "
          f"percent and a stride bias of {fmt(fw['model_at_true_ics']['stride_bias_cm']['point'])} cm. Model ICs: "
          f"median error {fmt(ic['point'])} ms after an offset of {fmt(ic.get('offset_ms'), 1)} ms, "
          f"{fmt(fw['model']['ic_median_ms_raw']['point'])} ms raw, matched share {fmt(ic.get('matched_share'), 3)}. "
          f"Per window ({ws['windows']} windows): the model's walking speed over the true one has median "
          f"{fmt(ws['pelvis_speed_ratio']['median'])} (10th to 90th percentile {fmt(ws['pelvis_speed_ratio']['p10'])} "
          f"to {fmt(ws['pelvis_speed_ratio']['p90'])}); horizontal pelvis error median "
          f"{fmt(ws['root_pos_err_xz_m']['median'], 3)} m; root orientation error median "
          f"{fmt(ws['root_orient_deg']['median'], 1)} degrees (yaw {fmt(ws['root_yaw_err_deg']['median'], 1)}, "
          f"tilt {fmt(ws['root_tilt_err_deg']['median'], 1)}); model root yaw at the centre knot median "
          f"{fmt(ws['root_yaw_centre_deg']['median'], 1)} degrees; mean absolute yaw about the centre heading median "
          f"{fmt(ws['turn_true_deg']['median'], 1)} degrees for the truth and {fmt(ws['turn_pred_deg']['median'], 1)} "
          f"for the model; local pose error median {fmt(ws['pose_err_mm']['median'], 1)} mm."]
    L += ["", "## Lines", "",
          "Per person, then the mean over persons, with the person-clustered 95 percent interval; matched share in "
          "parentheses. IC lines: after the predictor's offset fitted on the row, and raw (no offset).", "",
          "| predictor | line | forward | all | real dev (eval_g3_dev) |", "|---|---|---|---|---|"]
    real_map = {"model": "model", "constant": "population", "gyro_on_truth": "shank_gyro"}
    for name in PREDICTORS:
        if name not in raw_lines[ROWS[0]]:
            continue
        for line in LINE_ORDER:
            cells = []
            for row in ROWS:
                st = raw_lines[row][name].get(line)
                if st is None:
                    cells.append("")
                    continue
                sh = st.get("matched_share")
                cells.append(f"{fmt(st['point'])}{fmt_iv(st.get('interval'))}"
                             + (f" ({fmt(sh, 3)})" if sh is not None else ""))
            if not any(cells):
                continue
            rc = ""
            if rl and name in real_map:
                key = {"ic_median_ms": "ic_median_ms_walkway"}.get(line, line)
                st = rl.get(real_map[name], {}).get(key)
                if st is not None:
                    rc = f"{real_map[name]}: {real_cell(st)}"
            elif not rl and name in real_map:
                rc = "real line absent"
            L.append(f"| {name} | {line} | {cells[0]} | {cells[1]} | {rc} |")
    L += ["", "Offsets fitted (ms, n matched): " + "; ".join(
        f"{row} {name} {fmt(raw_lines[row][name]['ic_median_ms'].get('offset_ms'), 1)} "
        f"(n {raw_lines[row][name]['ic_median_ms'].get('offset_n')})"
        for row in ROWS for name in PREDICTORS if "ic_median_ms" in raw_lines[row].get(name, {})) + ".", ""]
    L += ["Reference counts: " + "; ".join(
        f"{row}: {c['takes']} takes, {c['persons']} persons, {c['true_ics']} true ICs, {c['true_strides']} true "
        f"strides of which {c['scored_strides']} scored, {c['scored_passes']} scored passes of "
        f"{c['true_passes']}, model {c['model_ics']} ICs and "
        f"{c['model_strides']} strides, shank detector {c['gyro_ics']} ICs" for row, c in res["counts"].items()) + ".", ""]
    L += ["## Root and pose per window", "",
          "Model window frame against the truth in the same frame (floor y = 0, pelvis heading +z and pelvis x, z 0 "
          "at the centre knot); means over the knots of a window, then statistics over windows.", "",
          "| quantity | row | windows | median | mean | 10th pct | 90th pct |", "|---|---|---|---|---|---|---|"]
    for c in WINDOW_COLUMNS + BOUND_COLUMNS:
        for row in ROWS:
            w = res["windows"][row]
            st = w.get(c)
            if st:
                nd = 3 if c.endswith("_m") else 2
                L.append(f"| {c} | {row} | {w['windows']} | {fmt(st['median'], nd)} | {fmt(st['mean'], nd)} | "
                         f"{fmt(st['p10'], nd)} | {fmt(st['p90'], nd)} |")
    L += ["", "## Definitions", "",
          "- Reference: the true body of each render on the stream clock; true ICs from frames.contact_labels (the "
          "labels the model was trained on); true passes: runs of ICs (gaps up to "
          f"{gait.PASS_GAP_S:g} s) cut into segments of {PASS_STEPS} steps; strides and passes scored when steady "
          f"and straight (pelvis chord at least {g3.STRAIGHT:g} of its path, chord speed at least {g3.STEADY:g} times "
          "the take's median stride chord speed); every true IC scored.",
          "- model: the G3 readouts of the stitched model body; model_at_true_ics: the same body read at the true "
          "ICs; gyro_on_truth: the true body read at the shank-gyroscope ICs of the rendered raw gyroscopes (the "
          "floor; its speed is the truth's by construction); constant: leave-one-person-out mean scored true stride "
          "and pass speed of the row.",
          "- model_legs: the model's ICs, strides and pass speeds from the stance-foot no-slip speed of the decoded "
          "pose (the translation output not read); a diagnostic, a G3 readout only once registered in "
          "docs/rules/G3_RULE.md. bound_speed (--bound): gradient boosting speed fitted on train-store renders only, "
          "at the shank-gyroscope ICs; bound_pos_err_xz_m its dead reckoning along the lower-back gyroscope heading.",
          "- The real column reads the dev row of the G3 simulator-only report of the same checkpoint: model, "
          "population baseline and shank-gyroscope detector against the walkway (IC line: walkway reference; the "
          "real speed line scores walkway passes of several strides, this one single straight strides).",
          "- GPU inference is not bit-reproducible under the same seeds: single windows move from one run to the next, "
          "the lines above by far less than their intervals.",
          "", "Time (s): " + ", ".join(f"{k} {v}" for k, v in res["timing_s"].items()) + ".", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def _sites_arg(text):
    try:
        return parse_sites(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--refine-steps", type=int, default=I.STEPS_EVAL)
    p.add_argument("--points", type=int, default=I.POINTS_EVAL, help="kernel points per stream-window and step")
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a take")
    p.add_argument("--limit", type=int, default=None, help="the first N selected takes only (a smoke run)")
    p.add_argument("--real", default=None, help="g3_dev.json or g3_sim.json of scripts/eval_g3_dev.py for the "
                                                "checkpoint (default: reports/gates/g3/dev_<checkpoint name>/"
                                                "g3_dev.json, else reports/gates/g3_sim/dev_<checkpoint name>/"
                                                "g3_sim.json)")
    p.add_argument("--sites", type=_sites_arg, default=None, metavar="SITE[,SITE]",
                   help="render with only these sites of the xsens13 layout (default all; output suffix _sites<n>)")
    p.add_argument("--out", default=None)
    p.add_argument("--bound", action="store_true", help="add the in-sim bound (fit_bound on train-store renders)")
    p.add_argument("--bound-mix", type=int, default=BOUND_MIX, help="subject-balanced train windows of the bound")
    a = p.parse_args(argv)
    if a.bound and a.sites and not set(BOUND_SITES) <= set(a.sites):
        p.error(f"--bound needs the sites {', '.join(BOUND_SITES)} in --sites")
    return a


def main():
    a = parse()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import body_store
    from mdfm.factory import layouts as LY
    from mdfm.factory import synth as SY
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV

    t_start = time.time()
    code = [os.path.abspath(__file__), I.__file__, g3.__file__, gait.__file__, SY.__file__]
    code_sha = common.code_sha256(code)
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    out = a.out or default_out(loaded.name, a.sites)
    paths.assert_not_moveatlas(out)
    print(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), phase {loaded.phase} step {loaded.step}", flush=True)

    store = body_store.BodyStore(split="dev")
    aw_rows, aw_dev = amass_walks_rows()
    takes = select_takes(store.takes, extra=aw_dev)
    n_tread = int(takes.attrs.get("treadmill", 0))
    if a.limit:
        takes = takes.iloc[:int(a.limit)].reset_index(drop=True)
    trained = loaded.trained_persons()
    clash = sorted(set(takes["person_id"]) & trained)
    if clash:
        raise Refused(f"dev persons among the checkpoint's training persons: {clash[:3]}")
    if set(takes["person_id"]) & set(_amass_index().LOCKBOX_PERSONS):
        raise Refused("a lockbox person among the dev takes")
    layout = LY.load_layouts()[LAYOUT]
    if not layout.concrete:
        raise ValueError(f"{LAYOUT} is not a concrete layout")
    layout = site_layout(layout, a.sites)
    sampler = SY.SynthSampler(store=store, device=dev)
    bank = SkeletonBank(torch.float64)
    print(f"{len(takes)} takes ({int(takes['forward'].sum())} forward), {takes['person_id'].nunique()} persons, "
          f"{n_tread} treadmill takes left out", flush=True)
    bound, bound_info = None, None
    if a.bound:
        train = body_store.BodyStore(split="train")
        bound = fit_bound(train, SY.SynthSampler(store=train, device=dev), layout, n_mix=a.bound_mix,
                          log=lambda m: print(m, flush=True))
        clash = sorted(set(bound["persons"]) & set(takes["person_id"]))
        if clash:
            raise Refused(f"bound persons among the dev takes: {clash[:3]}")
        bound_info = {k: bound[k] for k in ("segments", "windows", "persons_sha256", "split")}
        bound_info["n_persons"] = len(bound["persons"])

    infos, windows, parts = [], [], {}
    for k, take in takes.iterrows():
        r = run_take(loaded.model, sampler, store, take, layout, bank, seed=a.seed, refine_steps=a.refine_steps,
                     points=a.points, batch_windows=a.batch_windows, pool=not a.no_pool, bound=bound)
        infos.append(r["info"])
        windows += r["windows"]
        for key, df in r["tables"].items():
            parts.setdefault(key, []).append(df)
        print(f"  [{k + 1}/{len(takes)}] {r['info']['take_id']}: {r['info']['windows']} windows, "
              f"{r['info']['seconds_infer']:.1f} s", flush=True)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    wdf = pd.DataFrame(windows)

    rows = {"forward": takes.loc[takes["forward"], "take_id"].tolist(), "all": takes["take_id"].tolist()}
    lines, tabs, counts = {}, {}, {}
    for row, ids in rows.items():
        lines[row], tabs[row], counts[row] = row_lines(tables, ids)
    verdict, nums = answer(lines)
    tk = per_take(infos, tabs["all"], windows)
    real_path, looked = resolve_real(a.real, loaded.name)
    real = real_lines(real_path, loaded.sha256)

    mem = {}
    if dev.type == "cuda":
        mem = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}
    head = PV.moveatlas_head()
    by_split = {}
    for r in aw_rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    strip = lambda d: {n: {ln: {k: v for k, v in st.items() if k != "per_person"} for ln, st in s.items()}  # noqa: E731
                       for n, s in d.items()}
    res = {
        "run": "g3_insim", "what": "gait readouts of the model on synthetic renders of AMASS dev walks, against their "
                                   "true bodies (plan task B4.19c)",
        "checkpoint": loaded.describe(),
        "inference": {"refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": I.WIN_S,
                      "hop_s": I.HOP_S, "batch_windows": a.batch_windows, "seed": a.seed,
                      "shape_pooling": "per take" if not a.no_pool else "none", "weights": a.weights,
                      "device": str(dev), "windows": int(len(wdf)), "memory": mem, "layout": LAYOUT,
                      "source": SOURCE, "render_dtype": "float32"},
        "sets": {"n_takes": int(len(takes)), "n_persons": int(takes["person_id"].nunique()),
                 "persons_sha256": _amass_index().person_list_sha256(sorted(set(takes["person_id"]))),
                 "archives": {k: int(v) for k, v in takes["archive"].value_counts().sort_index().items()},
                 "treadmill_left_out": n_tread, "limit": a.limit,
                 "rows": {row: {"takes": len(ids), "persons": counts[row]["persons"]} for row, ids in rows.items()},
                 "amass_walks": {"rows": len(aw_rows), "by_split": by_split, "dev": len(aw_dev)},
                 "trained_persons_among_takes": 0, "dev_store_persons_sha256": _amass_index().person_list_sha256(
                     sorted(set(store.takes["person_id"])))},
        "constants": {"pass_steps": PASS_STEPS, "pass_gap_s": gait.PASS_GAP_S, "straight": g3.STRAIGHT,
                      "steady": g3.STEADY, "stride_match_s": g3.STRIDE_MATCH_S, "ic_match_s": g3.IC_MATCH_S, "g3_lines": g3.LINES,
                      "walk_styles": list(g3.WALK_STYLES)},
        "answer": {"row": PRIMARY_ROW, "verdict": verdict, "text": ANSWER_TEXT[verdict], "numbers": nums},
        "lines": strip(lines),
        "per_person": {row: {n: {ln: st.get("per_person") for ln, st in s.items() if st.get("per_person")}
                             for n, s in lines[row].items()} for row in ROWS},
        "counts": counts,
        "windows": {row: window_summary(wdf, ids) for row, ids in rows.items()},
        "real": real,
        "bound": bound_info,
        "timing_s": {"all": round(time.time() - t_start, 1),
                     "render": round(float(sum(i["seconds_render"] for i in infos)), 1),
                     "infer": round(float(sum(i["seconds_infer"] or 0 for i in infos)), 1)},
        "moveatlas_head": head,
    }
    if a.sites:
        res["inference"]["sites"] = list(layout.stream("imu").sites)
    if real is None:
        res["real_absent"] = {"text": "real line absent", "looked_for": [common.rel(p) for p in looked]}
    raw_lines = lines
    res = clean(res)
    text = markdown(res, raw_lines)
    common.write_csv(tk, os.path.join(out, "takes.csv"))
    common.write_csv(wdf, os.path.join(out, "windows.csv"))
    js = json.dumps(res, indent=1, ensure_ascii=False) + "\n"
    common.write_text(os.path.join(out, "insim.json"), js)
    common.write_text(os.path.join(out, "summary.md"), text)
    prov = {"script": "scripts/eval_gait_insim.py", "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "checkpoint": res["checkpoint"]["path"], "checkpoint_sha256": loaded.sha256, "code_sha256": code_sha,
            "code_files": [common.rel(p) for p in code]}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(text)
    print(f"written {common.rel(out)}")


if __name__ == "__main__":
    main()
