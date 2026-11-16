"""F2 lever 5a (docs/proposals/F2_LEVER5_DRAFT.md section 8 item 2; task L5A, 2026-10-20): the inertial anchor of lever
5 inside the test-time refinement of the G3 predictor, label-free, off by default.

The term. In mode 'anchor', the refinement objective of each chunk of windows (model/refine.refine: -sum_w log_joint
with every model parameter frozen) gains the training term of lever 5 itself, train/losses.zupt_anchor_parts with
both of its parts, weighted by WEIGHT: per version-2 ZUPT stride of the record's own foot sensor whose two still
instants t_a and t_b lie at least EDGE_S inside a window, distance rate (S - D)^2 / T / (2 SD^2) and stillness
(v_a^2 + v_b^2) / (2 SD^2), S the ZUPT stride, D the horizontal displacement (window-frame x and z) of the decoded
ankle joint of that foot between the two instants, T = t_b - t_a, v the decoded ankle's horizontal speed at each
instant. The term is computed every step on all the chunk's targets (the kernel points of the likelihood are drawn as
without it: the term draws no random number), from one extra decode of the anchored windows at the Hermite supports
of the instants. The betas reach the term's kinematics with their gradient stopped and the decoder's FiLM reads them
detached, so the term moves the latents only; the shape follows the likelihood and the priors alone (lever 5's rule,
D7). model/refine.py is not edited: AnchoredObjective wraps model.log_joint for one chunk's refinement (refine reads
the model's log_joint at each step) and removes the wrap on exit, so no attribute of the model remains.

Where the targets come from: the record's own foot IMU, nothing else. record_strides(record) takes the band copy of
the record exactly as mdfm/eval/infer.record_windows computes it (the model's own input, windows.whitelist of the
record, 40 Hz zero-phase low-pass at 100 Hz), its foot sites (foot_L 11 and foot_R 12, each with its acc and gyro
channels: data/zupt_anchor.foot_signals, an invalid sample NaN) and runs readouts/zupt.strides (version 2 of task
SCALE, its constants unchanged), keeping the strides of at least MIN_LENGTH_M. No walkway, insole, PKMAS or other
reference value enters: the functions of this module take the record dict and the model's windows only, open no file,
and a record without a foot site carries no target (its windows are refined as without the option, bit for bit).

Constants, fixed on 2026-10-20 from the lever-5 draft and test Q of task SCALE before any 5a number existed, never
tuned on a dev value: WEIGHT 1.0 and SD 0.1 m/s (draft section 2, the walkway anchor's weight and sd; test Q's SD),
EDGE_S 0.1 s (draft section 2; test Q's EDGE_S; configs/pilot_f2l5.yaml edge_s), MIN_LENGTH_M 0.3 m (lever-5 rule D2,
configs/pilot_f2l5.yaml min_length_m), STEPS 20 (the G3 harness's evaluation refinement; test Q's arm q20). The
scripts expose no flag that changes them, and check_steps refuses an anchored refinement of another step count.

Modes (MODES): 'off', the default (nothing of this module runs: mdfm/eval/infer.py is unchanged); 'report' (the
targets and the foot report on the final bodies, the objective unchanged, so the bodies equal those of 'off' bit for
bit: the off arm's report under the same code); 'anchor' (the term in the refinement, and the report).

Foot report (foot_report; reported beside the lines of the lever-5a rule, the stance-slide report): per target, on
the final body of its window (the decoded posterior mean at every knot, the shape pooled per trial), the decoded
ankle's horizontal position at t_a and t_b (linear between knots), D over S, its horizontal speed at both instants
(central differences of the knot positions, linear between knots), and the slide over the measured still run: the
decoded ankle's horizontal path over the knots of the ZUPT still run that opens the stride (the measured foot flat
on the ground: |w| < 45 deg/s and | |f| - 1 g | < 0.12 g), and that path over S. The distance and stillness parts
on the final body beside (numpy, linear between knots: close to, not equal to, the Hermite term of the refinement).
Rows carry the trial, the person, the window and the foot; no reference value.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data import windows as W
from ..data import zupt_anchor as ZA
from ..data.window_store import band_copy
from ..readouts import zupt as Z

FORMAT = "mdfm-zupt-refine/1"
MODES = ("off", "report", "anchor")
WEIGHT = 1.0                 # the term's weight in the refinement objective (draft section 2; the walkway anchor's)
SD = 0.1                     # m/s (draft section 2; test Q's SD, reports/bench/scale/code/q_run.py)
EDGE_S = 0.1                 # s: a stride anchors a window when both instants lie this far inside it (test Q's EDGE_S)
MIN_LENGTH_M = 0.3           # m: the shortest ZUPT stride kept (lever-5 rule D2, configs/pilot_f2l5.yaml)
STEPS = 20                   # refinement steps of the anchored refinement (the G3 harness's; test Q's arm q20)
CONSTANTS = {"weight": WEIGHT, "sd_m_s": SD, "edge_s": EDGE_S, "min_length_m": MIN_LENGTH_M, "steps": STEPS}
ANKLE = {0: 7, 1: 8}         # SMPL ankle joints of the left and right foot (train/losses.ANKLE_JOINTS)
SOURCE = ("the record's own foot IMU: the band copy of its foot_L and foot_R sites, version-2 ZUPT strides "
          "(mdfm/readouts/zupt.py); no walkway, insole or other reference value")
ROW_FIELDS = ("trial_id", "person_id", "window", "foot", "t_a", "t_b", "S", "T", "D", "D_over_S", "v_a", "v_b",
              "slide_m", "slide_share", "run_inside", "distance", "still", "turn_deg")


class ZuptRefineError(ValueError):
    """A mode, a step count or an input the option refuses."""


def check_mode(mode):
    if mode not in MODES:
        raise ZuptRefineError(f"zupt mode {mode!r}, expected one of {MODES}")
    return mode


def check_steps(mode, refine_steps):
    """ZuptRefineError when the anchored refinement would run another step count than STEPS (constants section)."""
    if check_mode(mode) == "anchor" and int(refine_steps) != STEPS:
        raise ZuptRefineError(f"the anchored refinement (lever 5a) runs {STEPS} steps, its fixed constant; "
                              f"{refine_steps} asked")


def record_settings(mode):
    """The run record of the option (what the evaluation JSON states under inference.zupt_anchor)."""
    check_mode(mode)
    out = {"format": FORMAT, "mode": mode, "what": "F2 lever 5a: the inertial anchor in the refinement (task L5A)"
           if mode == "anchor" else "F2 lever 5a report only: targets and the foot report, the objective unchanged",
           "term": "train/losses.zupt_anchor_parts, distance and stillness parts, betas detached",
           "in_objective": mode == "anchor", "source": SOURCE, "zupt": {"version": Z.VERSION,
                                                                       "constants": dict(Z.CONSTANTS)}}
    out.update(CONSTANTS)
    return out


# ---------------------------------------------------------------- targets from the record alone
@dataclass(frozen=True)
class RecordStrides:
    """The ZUPT strides of one record (module docstring): t_a, t_b (n,) float64 s on the record clock (the band copy's
    sample times, as the windows' t0), length_m (n,), foot (n,) int64 (0 left, 1 right), run_t0 and run_t1 (n,) the
    first and last sample times of the still run that opens the stride, m_a, m_b (n,) int64 band-copy sample indices,
    turn_deg (n,); feet: the feet with a sensor; fps."""
    t_a: np.ndarray
    t_b: np.ndarray
    length_m: np.ndarray
    foot: np.ndarray
    run_t0: np.ndarray
    run_t1: np.ndarray
    m_a: np.ndarray
    m_b: np.ndarray
    turn_deg: np.ndarray
    feet: tuple
    fps: float

    @property
    def n(self):
        return int(len(self.t_a))

    def zstrides(self):
        """The data/zupt_anchor.Strides of these strides (its window rule, window_strides, is the one training uses)."""
        return ZA.Strides(self.m_a, self.m_b, self.t_a, self.t_b, self.length_m, self.foot, self.turn_deg)


def record_band(record, allow_eval=False):
    """(band-copied IMU Stream, its sample times on the record clock) of a canonical IMU record, computed as
    infer.record_windows computes them (whitelist, arrays, band_copy of the whole record)."""
    from .infer import refuse_cody
    refuse_cody(str(record.get("source") or ""), record.get("subject_ns"), record.get("person_id"),
                allow_eval=allow_eval)
    rec = W.whitelist(record)
    fps = W.record_fps(rec)
    values, cm, valid, meta = W.arrays("imu", rec)
    t64 = np.arange(values.shape[0], dtype=np.float64) / fps
    full = W.Stream(modality="imu", role="input", fps=fps, t=t64, values=values, chan_mask=cm, valid=valid, meta=meta)
    b, info = band_copy(full, return_info=True)
    return b, t64[info.idx]


def record_strides(record, min_length_m=MIN_LENGTH_M, allow_eval=False):
    """RecordStrides of a canonical IMU record from its own foot sites alone (module docstring)."""
    b, tb = record_band(record, allow_eval=allow_eval)
    sites = [int(s) for s in (b.meta or {}).get("site_ids", [])]
    feet = ZA.foot_signals(b.values, b.valid, sites, np.asarray(b.chan_mask, dtype=bool))
    cols = {k: [] for k in ("t_a", "t_b", "length_m", "foot", "run_t0", "run_t1", "m_a", "m_b", "turn_deg")}
    for foot, (a, g) in sorted(feet.items()):
        for s in Z.strides(a, g, float(b.fps)):
            if not s["length_m"] >= float(min_length_m):
                continue
            r0, r1 = s["run_a"]
            cols["t_a"].append(float(tb[s["m_a"]]))
            cols["t_b"].append(float(tb[s["m_b"]]))
            cols["length_m"].append(float(s["length_m"]))
            cols["foot"].append(ZA.FOOT_IDS[foot])
            cols["run_t0"].append(float(tb[r0]))
            cols["run_t1"].append(float(tb[r1 - 1]))
            cols["m_a"].append(int(s["m_a"]))
            cols["m_b"].append(int(s["m_b"]))
            cols["turn_deg"].append(float(s["turn_deg"]))
    ints = ("foot", "m_a", "m_b")
    arr = {k: np.asarray(v, dtype=np.int64 if k in ints else np.float64) for k, v in cols.items()}
    return RecordStrides(**arr, feet=tuple(sorted(feet)), fps=float(b.fps))


def window_index(strides, windows, edge_s=EDGE_S):
    """[(n_i,) int64] per window: the strides whose two instants lie at least edge_s inside it (zupt_anchor's rule)."""
    zs = strides.zstrides()
    return [ZA.window_strides(zs, float(wd.t0), float(wd.duration), edge_s) if strides.n else np.zeros(0, np.int64)
            for wd in windows]


def chunk_targets(strides, windows, index, first, count):
    """data/zupt_anchor.ZuptTargets of windows first .. first + count - 1 (window ids local to the chunk, times from
    each window's start), or None when none of them holds a stride."""
    w, ft, ta, tb, d = [], [], [], [], []
    for j in range(int(first), int(first) + int(count)):
        t0 = float(windows[j].t0)
        for i in index[j]:
            w.append(j - int(first))
            ft.append(int(strides.foot[i]))
            ta.append(float(strides.t_a[i]) - t0)
            tb.append(float(strides.t_b[i]) - t0)
            d.append(float(strides.length_m[i]))
    if not w:
        return None
    return ZA.ZuptTargets(torch.tensor(w, dtype=torch.int64), torch.tensor(ft, dtype=torch.int64),
                          torch.tensor(ta, dtype=torch.float32), torch.tensor(tb, dtype=torch.float32),
                          torch.tensor(d, dtype=torch.float32))


# ---------------------------------------------------------------- the anchored objective
class AnchoredObjective:
    """Context (module docstring): model.log_joint(z, beta, nuis, batch_k) returns log_joint minus WEIGHT times the
    term of `targets` on the chunk's full batch `batch` (the kernel-point subsample of a step does not touch the
    term); a call with a body held fixed (z None) is left as it is. calls: one record per call {term, distance,
    still} (sums over the chunk's windows) with the per-target ratio D / S."""

    def __init__(self, model, batch, targets, weight=WEIGHT, sd=SD):
        self.model, self.batch, self.targets = model, batch, targets
        self.weight, self.sd = float(weight), float(sd)
        self.calls = []

    def log_joint(self, z, beta, nuis, batch, body=None, kin=None):
        lj = self._orig(z, beta, nuis, batch, body=body, kin=kin)
        if z is None or body is not None:
            return lj
        from ..train import losses as LS
        parts = LS.zupt_anchor_parts(self.model, z, beta, self.batch, self.targets, self.sd)
        S, D = parts["S"].detach(), parts["D"].detach()
        self.calls.append({"term": float(parts["term"].detach().sum()),
                           "distance": float(parts["distance"].detach().sum()),
                           "still": float(parts["still"].detach().sum()),
                           "ratio": (D / S.clamp_min(1e-9)).double().cpu().numpy()})
        return lj - (self.weight * parts["term"]).to(lj.dtype)

    def __enter__(self):
        if "log_joint" in vars(self.model):
            raise RuntimeError("model.log_joint is already wrapped")
        self._orig = self.model.log_joint
        self.model.log_joint = self.log_joint
        return self

    def __exit__(self, *exc):
        if "log_joint" in vars(self.model):
            del self.model.log_joint
        return False


@dataclass
class TrialZupt:
    """The option on one trial (module docstring): mode, the record's strides, the model's windows, the per-window
    stride indices and the anchored refinement's per-chunk records (chunks). objective() is what infer.infer calls
    per chunk (argument zupt)."""
    mode: str
    strides: RecordStrides
    windows: list
    index: list
    weight: float = WEIGHT
    sd: float = SD
    chunks: list = field(default_factory=list)

    @staticmethod
    def of(record, windows, mode, refine_steps, allow_eval=False):
        check_steps(mode, refine_steps)
        if mode == "off":
            raise ZuptRefineError("TrialZupt of mode 'off': the option does not run")
        s = record_strides(record, allow_eval=allow_eval)
        return TrialZupt(mode, s, list(windows), window_index(s, windows))

    @property
    def n_targets(self):
        return int(sum(len(i) for i in self.index))

    def objective(self, model, batch, first, count):
        """The AnchoredObjective of the chunk's windows first .. first + count - 1, or None (mode 'report', or no
        target in the chunk: refined as without the option)."""
        if self.mode != "anchor":
            return None
        tg = chunk_targets(self.strides, self.windows, self.index, first, count)
        if tg is None:
            return None
        obj = AnchoredObjective(model, batch, tg, self.weight, self.sd)
        self.chunks.append(obj)
        return obj

    def summary(self, bodies, trial_id, person_id, rate=100.0):
        """(flat fields of the trial's log entry, report rows): the option's mode, the record's strides and feet, the
        targets and anchored windows; in mode 'anchor' the term and its parts at the first and the last call (sums
        over the chunks) and the median D / S there; the foot report's medians on the final bodies."""
        rows = foot_report(bodies, self.windows, self.strides, self.index, trial_id, person_id, rate, self.sd)
        out = {"zupt_mode": self.mode, "zupt_feet": ",".join(self.strides.feet), "zupt_record_strides": self.strides.n,
               "zupt_targets": self.n_targets, "zupt_windows_anchored": int(sum(1 for i in self.index if len(i)))}
        if self.mode == "anchor":
            used = [c for c in self.chunks if c.calls]
            for k in ("term", "distance", "still"):
                out[f"zupt_{k}_first"] = float(sum(c.calls[0][k] for c in used)) if used else None
                out[f"zupt_{k}_last"] = float(sum(c.calls[-1][k] for c in used)) if used else None
            for when, i in (("first", 0), ("last", -1)):
                r = np.concatenate([c.calls[i]["ratio"] for c in used]) if used else np.zeros(0)
                out[f"zupt_ratio_{when}"] = float(np.median(r)) if len(r) else None
            out["zupt_chunks_anchored"] = len(used)
        out.update(report_medians(rows))
        return out, rows


# ---------------------------------------------------------------- the foot report on the final bodies
def _at(x, k):
    """x (n, ...) linear between knots at the fractional knot k (clamped to the window)."""
    n = x.shape[0]
    k = min(max(float(k), 0.0), n - 1.0)
    a = int(math.floor(k))
    b = min(a + 1, n - 1)
    s = k - a
    return (1.0 - s) * x[a] + s * x[b]


def foot_report(bodies, windows, strides, index, trial_id, person_id, rate=100.0, sd=SD):
    """[row of ROW_FIELDS] per (window, target) (module docstring)."""
    rows = []
    for j, (b, wd) in enumerate(zip(bodies, windows)):
        if not len(index[j]):
            continue
        J = np.asarray(b.joints, dtype=np.float64)
        n = J.shape[0]
        t0 = float(wd.t0)
        for i in index[j]:
            foot = int(strides.foot[i])
            P = J[:, ANKLE[foot]][:, [0, 2]]
            V = np.gradient(P, 1.0 / rate, axis=0) if n > 1 else np.zeros_like(P)
            ta, tb = float(strides.t_a[i]) - t0, float(strides.t_b[i]) - t0
            S, T = float(strides.length_m[i]), max(tb - ta, 1.0 / rate)
            D = float(np.linalg.norm(_at(P, ta * rate) - _at(P, tb * rate)))
            va, vb = float(np.linalg.norm(_at(V, ta * rate))), float(np.linalg.norm(_at(V, tb * rate)))
            r0, r1 = float(strides.run_t0[i]) - t0, float(strides.run_t1[i]) - t0
            inside = bool(r0 >= -1e-9 and r1 <= (n - 1) / rate + 1e-9 and r1 > r0)
            slide = None
            if inside:
                k0, k1 = int(math.ceil(r0 * rate - 1e-6)), int(math.floor(r1 * rate + 1e-6))
                seg = P[k0:k1 + 1]
                slide = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
            rows.append({"trial_id": str(trial_id), "person_id": str(person_id), "window": j,
                         "foot": "L" if foot == 0 else "R", "t_a": ta, "t_b": tb, "S": S, "T": T, "D": D,
                         "D_over_S": D / S if S > 0 else None, "v_a": va, "v_b": vb, "slide_m": slide,
                         "slide_share": (slide / S) if slide is not None and S > 0 else None, "run_inside": inside,
                         "distance": rate * (S - D) ** 2 / T / (2.0 * sd * sd),
                         "still": (va * va + vb * vb) / (2.0 * sd * sd), "turn_deg": float(strides.turn_deg[i])})
    return rows


def report_medians(rows):
    """{zupt_report_*}: the medians of the foot report rows (D / S, the speed at the instants, the slide share over
    the rows whose still run lies in the window) and their count."""
    def med(key, sel=None):
        v = [r[key] for r in rows if r[key] is not None and (sel is None or sel(r))]
        return float(np.median(v)) if v else None
    speeds = [x for r in rows for x in (r["v_a"], r["v_b"])]
    return {"zupt_report_rows": len(rows), "zupt_report_ratio_median": med("D_over_S"),
            "zupt_report_speed_median": float(np.median(speeds)) if speeds else None,
            "zupt_report_slide_share_median": med("slide_share", lambda r: r["run_inside"]),
            "zupt_report_runs_inside": int(sum(1 for r in rows if r["run_inside"]))}


def summary_block(log, mode):
    """The evaluation JSON's zupt_anchor block from a predictor's log entries (the trials run with the option): the
    run record (record_settings), trials, targets, anchored trials, and the medians over the trials of the term at
    its first and last call and of the foot report."""
    ent = [e for e in log if e.get("zupt_mode") == mode]

    def med(k):
        v = [float(e[k]) for e in ent if e.get(k) is not None and math.isfinite(float(e[k]))]
        return float(np.median(v)) if v else None
    out = record_settings(mode)
    out.update({"trials": len(ent), "trials_with_targets": int(sum(1 for e in ent if e.get("zupt_targets"))),
                "targets": int(sum(int(e.get("zupt_targets") or 0) for e in ent)),
                "record_strides": int(sum(int(e.get("zupt_record_strides") or 0) for e in ent)),
                "report_rows": int(sum(int(e.get("zupt_report_rows") or 0) for e in ent)),
                "trial_medians": {k: med(k) for k in ("zupt_term_first", "zupt_term_last", "zupt_distance_first",
                                                      "zupt_distance_last", "zupt_still_first", "zupt_still_last",
                                                      "zupt_ratio_first", "zupt_ratio_last", "zupt_report_ratio_median",
                                                      "zupt_report_speed_median",
                                                      "zupt_report_slide_share_median")
                                  if mode == "anchor" or not k.startswith(("zupt_term", "zupt_distance", "zupt_still",
                                                                           "zupt_ratio"))}})
    return out
