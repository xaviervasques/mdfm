"""Gate G-g of reports/bench/root_motion_diagnosis.md (fix 7): the global motion of a stage-1 model state on
synthetic renders of the AMASS walks, from the scratch probes of the diagnosis (translation p1_ratio.py,
p1b_paths.py, p5_legspeed.py, p8_selfconsistency.py, p9_series.py; heading analyze_yaw.py, probe_integrate.py;
signal warp_probe.py, latent_dr.py) promoted into the tree.

State: --checkpoint a checkpoint of train/loop (eval.infer.load_checkpoint: checkpoint.load re-checks its hashes;
--weights ema or model), or 'init' (the pilot model at torch.manual_seed(cfg.seed), a smoke run).

Windows: 10 s windows every 5 s of the walking takes of a split of the AMASS body store (task 'walk' of
docs/rules/task_vocabulary.yaml, treadmill takes left out, takes of at least 10 s; the dev split by default, or
train; the calib split, which the store also holds, is never read), --max-windows at most, rendered by
factory.synth.SynthSampler in the window frame of training with the concrete layout --layout (xsens13 by
default) or, with --mixture, layouts drawn from the config's mixture. Decoded by eval.infer.infer at every knot
with z = mu_z, unrefined (--refine-steps 0, the default) or refined. No lockbox, EVAL or CODY record and no real record is read; mdfm.gates is imported only
through eval.infer, as scripts/eval_gait_insim.py does, and no gate runs.

Lines (root_lines), the decoded body against the truth, both in the window frame, per window then pooled:
- per bin of time from the centre knot L // 2 (BINS_S: 0-0.5, 0.5-1, 1-2, 2-3, 3-4, 4-5 s): speed_ratio, the sum
  of the decoded horizontal pelvis speed over the true one at the knots where the truth moves faster than MOVING
  m/s; velocity_cosine, the speed-weighted cosine between the decoded and the true horizontal velocity there;
  velocity_heading_cosine, that of the decoded velocity with the decoded heading (sin psi, cos psi), its
  self-consistency (the truth's own value is reported beside it: pelvis sway keeps it under 1); yaw_err_deg, the
  mean absolute wrapped root yaw error; heading_gain and heading_corr, the slope through 0 and the correlation of
  the decoded on the true yaw about the centre heading, both wrapped, where the true one lies within 120 deg.
  Velocities are central differences smoothed over SMOOTH_S along the time axis of each window (smooth_time);
- lateral_slope and forward_slope: slopes through 0 of the decoded on the true displacement from the centre knot,
  x (lateral to the centre heading) and z (along it);
- heading_gain by layout family (with --mixture);
- legs: the stance-foot no-slip speed of the decoded pose (readouts.gait.stance_speed of the pelvis-relative
  joints with the decoded contacts; the translation output is not read) over the true speed and the decoded root
  speed over it, per window (mean over the stance knots), medians over windows; the same estimator on the truth
  (its floor);
- time warp (warp_lines): the motion played WARPS times as fast (the take read over WARPS x 10 s and resampled to
  1,000 knots, rotations renormalized; a probe input, never training data): the decoded window speed at each factor
  over its value at 1 against the same ratio of the truth, pooled over windows. --warps F[,F] replaces WARPS (0.8,
  1.25); with other factors the warp lines of the gate are not the registered G-g read, and the report says so.
  warp_rows: one row per factor, the unwarped one (factor 1) first, with the mean decoded and true window speeds
  over the windows every factor kept and the stride error of the decoded body at the true ICs (stride_errors:
  stride_mae_cm, stride_bias_cm over every stride of those windows), written to <stem>_warps.json beside the md
  and under warp_rows in the main json (the in-sim stride error at x1.5 against x1 that menu item M1 of
  scripts/weekend_branch.py reads);
- latent dead reckoning (--latent): ridge readouts (alpha by person-grouped 4-fold validation) of the per-patch
  yaw rate and heading-frame velocity (forward, lateral), from mu_z and from the encoder state before its final
  LayerNorm, fitted on --train-windows windows of the train split, integrated over the 0.25 s patches from the
  centre patch on the probed windows; the median per window of the heading and pelvis errors at the patch
  centres (patches 2 to 37), against the decoder's own at the same knots and a body that never moves.

Gate G-g (the probed split, unrefined): speed_ratio at least 0.8 in every bin; lateral_slope at least 0.8;
velocity_cosine at least 0.9 in the bins 3-4 and 4-5 s; the warp response within 0.1 of the true ratio at 0.8 and
1.25; the decoder's heading and pelvis errors at most those of the mu_z latent dead reckoning [cfg]. References of
phaseA_fix_full (the diagnosis): speed ratio 0.53 to 0.26, lateral slope -0.009, velocity cosine about 0 at 3 to 5
s, warp x0.98 for x0.76 and x1.23, decoder 55.0 deg and 0.876 m against 36.8 deg and 0.744 m for mu_z.

Writes reports/bench/root_motion_<run folder>_<checkpoint>.md and .json (init: root_motion_init; --tag appends a
suffix) with the checkpoint SHA-256, the probed and the latent-fit persons SHA-256 and the MoveAtlas HEAD, and,
unless --no-warp, root_motion_<...>_warps.json (the warp rows). Memory:
the caching allocator is capped at --cap-gb (4.8 GB by default).

Run: <conda>/envs/mdfm/python.exe scripts/probe_root_motion.py --checkpoint <run>/ckpt_A_005000.pt
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm.readouts import gait  # noqa: E402

RATE = 100.0
K = 1000                        # knots of a 10 s window
HOP = 500
BINS_S = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.01))
MOVING = 0.3                    # m/s, true speed of a moving knot
SMOOTH_S = 0.25                 # s, velocity smoothing along the time axis
HEADING_WITHIN = math.radians(120.0)
WARPS = (0.8, 1.25)
PATCH = 25                      # knots per latent patch (0.25 s)
DR_PATCHES = (2, 38)            # patches scored by the latent dead reckoning
LINES = {"speed_ratio_min": 0.8, "lateral_slope_min": 0.8, "velocity_cosine_min": 0.9,
         "velocity_cosine_bins": ("3-4", "4-5"), "warp_tolerance": 0.1}
CAP_GB = 4.8
OUT = os.path.join(TREE, "reports", "bench")
CONFIG = os.path.join(TREE, "configs", "stage1_pilot.yaml")


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def bin_name(lo, hi):
    return f"{lo:g}-{min(hi, 5.0):g}"


# ---------------------------------------------------------------- smoothing and lines
def smooth_time(x, n):
    """x (W, T, ...) windows stacked on axis 0 -> the moving average of n samples along the time axis (axis 1) of
    each window, windows never mixed (the scratch helper filtered axis 0, across windows)."""
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"smooth_time needs (W, T, ...) arrays, got {x.shape}")
    return uniform_filter1d(x, max(1, int(n)), axis=1, mode="nearest")


def window_velocity(xz, rate=RATE, smooth_s=SMOOTH_S):
    """(T, 2) m/s: central differences of a window's horizontal positions (T, 2), smoothed along time."""
    v = np.gradient(np.asarray(xz, dtype=np.float64), 1.0 / rate, axis=0)
    return smooth_time(v[None], int(round(smooth_s * rate)))[0]


def root_lines(windows, rate=RATE):
    """The lines of the module docstring over windows, each a dict of window-frame arrays over the same T knots:
    transl_pred, transl_true (T, 3) m; yaw_pred, yaw_true (T,) rad (root yaw, window frame); layout (str); and
    optionally rel_pred, rel_true (T, 22, 3) m pelvis-relative joints and contact_pred, contact_true (T, 2)."""
    acc = {bin_name(*b): dict.fromkeys(("sp", "st", "vd", "vn", "hd", "hn", "htd", "htn", "ye", "n", "sxy", "sxx",
                                        "syy", "sx", "sy", "nh"), 0.0) for b in BINS_S}
    lx = {"lxy": 0.0, "lxx": 0.0, "fxy": 0.0, "fxx": 0.0}
    by_layout, legs = {}, []
    for wd in windows:
        tp, tt = np.asarray(wd["transl_pred"], np.float64), np.asarray(wd["transl_true"], np.float64)
        yp, yt = np.asarray(wd["yaw_pred"], np.float64), np.asarray(wd["yaw_true"], np.float64)
        T = len(tt)
        c = T // 2
        vp, vt = window_velocity(tp[:, [0, 2]], rate), window_velocity(tt[:, [0, 2]], rate)
        sp, st = np.linalg.norm(vp, axis=1), np.linalg.norm(vt, axis=1)
        tau = np.abs(np.arange(T) - c) / rate
        psi_t, psi_p = _wrap(yt - yt[c]), _wrap(yp - yp[c])
        hp = np.stack([np.sin(yp), np.cos(yp)], 1)
        ht = np.stack([np.sin(yt), np.cos(yt)], 1)
        dp, dt_ = tp[:, [0, 2]] - tp[c, [0, 2]], tt[:, [0, 2]] - tt[c, [0, 2]]
        lx["lxy"] += float((dp[:, 0] * dt_[:, 0]).sum())
        lx["lxx"] += float((dt_[:, 0] ** 2).sum())
        lx["fxy"] += float((dp[:, 1] * dt_[:, 1]).sum())
        lx["fxx"] += float((dt_[:, 1] ** 2).sum())
        lay = by_layout.setdefault(str(wd.get("layout", "")), {"sxy": 0.0, "sxx": 0.0, "n": 0})
        within = np.abs(psi_t) <= HEADING_WITHIN
        lay["sxy"] += float((psi_t * psi_p)[within].sum())
        lay["sxx"] += float((psi_t ** 2)[within].sum())
        lay["n"] += int(within.sum())
        for lo, hi in BINS_S:
            a = acc[bin_name(lo, hi)]
            m = (tau >= lo) & (tau < hi)
            mov = m & (st > MOVING)
            a["sp"] += float(sp[mov].sum())
            a["st"] += float(st[mov].sum())
            a["vd"] += float((vp[mov] * vt[mov]).sum())
            a["vn"] += float((sp[mov] * st[mov]).sum())
            a["hd"] += float((vp[mov] * hp[mov]).sum())
            a["hn"] += float(sp[mov].sum())
            a["htd"] += float((vt[mov] * ht[mov]).sum())
            a["htn"] += float(st[mov].sum())
            a["ye"] += float(np.abs(_wrap(yp[m] - yt[m])).sum())
            a["n"] += int(m.sum())
            h = m & within
            x, y = psi_t[h], psi_p[h]
            a["sxy"] += float((x * y).sum())
            a["sxx"] += float((x * x).sum())
            a["syy"] += float((y * y).sum())
            a["sx"] += float(x.sum())
            a["sy"] += float(y.sum())
            a["nh"] += int(h.sum())
        if "rel_pred" in wd:
            lp = gait.stance_speed(wd["rel_pred"], wd["contact_pred"], rate)
            lt = gait.stance_speed(wd["rel_true"], wd["contact_true"], rate)
            on = np.isfinite(lp)
            ont = np.isfinite(lt)
            if on.sum() >= 50 and st[on].mean() > MOVING:
                legs.append((float(lp[on].mean() / st[on].mean()), float(sp[on].mean() / max(lp[on].mean(), 1e-9)),
                             float(lt[ont].mean() / st[ont].mean()) if ont.sum() >= 50 else float("nan")))
    r = lambda a, b: a / b if b > 0 else None                                       # noqa: E731
    bins = {}
    for name, a in acc.items():
        corr = None
        n = a["nh"]
        if n > 1:
            vx, vy = n * a["sxx"] - a["sx"] ** 2, n * a["syy"] - a["sy"] ** 2
            if vx > 0 and vy > 0:
                corr = (n * a["sxy"] - a["sx"] * a["sy"]) / math.sqrt(vx * vy)
        bins[name] = {"speed_ratio": r(a["sp"], a["st"]), "velocity_cosine": r(a["vd"], a["vn"]),
                      "velocity_heading_cosine": r(a["hd"], a["hn"]),
                      "velocity_heading_cosine_true": r(a["htd"], a["htn"]),
                      "yaw_err_deg": math.degrees(a["ye"] / a["n"]) if a["n"] else None,
                      "heading_gain": r(a["sxy"], a["sxx"]), "heading_corr": corr, "knots": int(a["n"])}
    L = np.asarray(legs, dtype=np.float64).reshape(-1, 3)
    return {"bins": bins, "lateral_slope": r(lx["lxy"], lx["lxx"]), "forward_slope": r(lx["fxy"], lx["fxx"]),
            "heading_gain_by_layout": {k: {"gain": r(v["sxy"], v["sxx"]), "knots": v["n"]}
                                       for k, v in by_layout.items()},
            "legs": {"windows": int(len(L)),
                     "legs_over_true_median": float(np.median(L[:, 0])) if len(L) else None,
                     "root_over_legs_median": float(np.median(L[:, 1])) if len(L) else None,
                     "truth_legs_over_true_median": float(np.nanmedian(L[:, 2]))
                     if len(L) and np.isfinite(L[:, 2]).any() else None},
            "windows": len(windows)}


def window_speed(transl, rate=RATE):
    """Mean horizontal pelvis speed (m/s) of a window (T, 3), smoothed velocities."""
    return float(np.linalg.norm(window_velocity(np.asarray(transl)[:, [0, 2]], rate), axis=1).mean())


def warp_lines(base, warped):
    """{factor: {decoded_ratio, true_ratio, response_error}} of the time warp: base and warped[f] lists of windows
    (dicts with transl_pred and transl_true) in the same order; the ratios pool the window speeds."""
    s0p = sum(window_speed(w["transl_pred"]) for w in base)
    s0t = sum(window_speed(w["transl_true"]) for w in base)
    out = {}
    for f, ws in warped.items():
        sp = sum(window_speed(w["transl_pred"]) for w in ws)
        st = sum(window_speed(w["transl_true"]) for w in ws)
        dr, tr = (sp / s0p if s0p > 0 else None), (st / s0t if s0t > 0 else None)
        out[str(f)] = {"decoded_ratio": dr, "true_ratio": tr,
                       "response_error": None if dr is None or tr is None else abs(dr - tr)}
    return out


def stride_errors(win, rate=RATE):
    """(decoded minus true stride length (n,) m, true stride length (n,) m) of one window dict at its true ICs:
    gait.initial_contacts of contact_true, then gait.strides of the true joints (rel_true plus transl_true) and of
    the decoded joints_pred at those ICs (the decoded body read at the true ICs, as model_at_true_ics of
    scripts/eval_gait_insim.py); both empty without rel_true, joints_pred or contact_true."""
    if not all(k in win for k in ("rel_true", "joints_pred", "contact_true")):
        return np.zeros(0), np.zeros(0)
    jt = np.asarray(win["rel_true"], np.float64) + np.asarray(win["transl_true"], np.float64)[:, None, :]
    jp = np.asarray(win["joints_pred"], np.float64)
    n = min(len(jt), len(jp), len(win["contact_true"]))
    ics = gait.initial_contacts(np.asarray(win["contact_true"], np.float64)[:n], rate)
    st, sp = gait.strides(jt[:n], ics, rate=rate), gait.strides(jp[:n], ics, rate=rate)
    m = st.merge(sp, on=["foot", "t0", "t1"], suffixes=("_t", "_p"))
    return (m["length_m_p"] - m["length_m_t"]).to_numpy(np.float64), m["length_m_t"].to_numpy(np.float64)


def warp_rows(base, warped):
    """[row] of the time warp, one per factor, the unwarped windows (factor 1.0) first, then the factors of warped
    in their order: {factor, windows, decoded_speed_m_s, true_speed_m_s (means of window_speed over the windows),
    decoded_ratio, true_ratio, response_error (as warp_lines; 1, 1 and 0 for the unwarped row when its speeds are
    positive), strides, stride_mae_cm, stride_bias_cm, true_stride_m (stride_errors pooled over the strides of the
    windows; None without a stride)}."""
    def row(f, ws, s0p, s0t):
        sp = [window_speed(w["transl_pred"]) for w in ws]
        st = [window_speed(w["transl_true"]) for w in ws]
        dr = sum(sp) / s0p if s0p > 0 else None
        tr = sum(st) / s0t if s0t > 0 else None
        e = [stride_errors(w) for w in ws]
        d = np.concatenate([x[0] for x in e]) if e else np.zeros(0)
        lt = np.concatenate([x[1] for x in e]) if e else np.zeros(0)
        return {"factor": float(f), "windows": len(ws), "decoded_speed_m_s": float(np.mean(sp)) if sp else None,
                "true_speed_m_s": float(np.mean(st)) if st else None, "decoded_ratio": dr, "true_ratio": tr,
                "response_error": None if dr is None or tr is None else abs(dr - tr), "strides": int(len(d)),
                "stride_mae_cm": 100.0 * float(np.abs(d).mean()) if len(d) else None,
                "stride_bias_cm": 100.0 * float(d.mean()) if len(d) else None,
                "true_stride_m": float(lt.mean()) if len(lt) else None}
    s0p = sum(window_speed(w["transl_pred"]) for w in base)
    s0t = sum(window_speed(w["transl_true"]) for w in base)
    return [row(1.0, base, s0p, s0t)] + [row(f, ws, s0p, s0t) for f, ws in warped.items()]


def parse_warps(text):
    """The factors of a --warps value 'F[,F]' as floats, in the given order; None for None. A factor that is not
    a finite positive number, a repeated factor or an empty list raises ValueError."""
    if text is None:
        return None
    parts = [s.strip() for s in str(text).split(",") if s.strip()]
    if not parts:
        raise ValueError("--warps lists no factor")
    out = []
    for s in parts:
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"--warps factor {s!r} is not a number") from None
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"--warps factor {s!r} is not a positive number")
        out.append(f)
    if len(set(out)) != len(out):
        raise ValueError(f"a factor is listed twice in {parts}")
    return tuple(out)


def gate(lines, warp, latent):
    """The G-g verdicts (module docstring)."""
    b = lines["bins"]
    v = []
    for name, x in b.items():
        v.append({"line": f"speed_ratio {name} s", "value": x["speed_ratio"], "at_least": LINES["speed_ratio_min"]})
    v.append({"line": "lateral_slope", "value": lines["lateral_slope"], "at_least": LINES["lateral_slope_min"]})
    for name in LINES["velocity_cosine_bins"]:
        v.append({"line": f"velocity_cosine {name} s", "value": b[name]["velocity_cosine"],
                  "at_least": LINES["velocity_cosine_min"]})
    for f, w in (warp or {}).items():
        if not isinstance(w, dict):
            continue
        v.append({"line": f"warp response error at x{f}", "value": w["response_error"],
                  "at_most": LINES["warp_tolerance"]})
    if latent and "mu_z" in latent:
        mz = latent["mu_z"]
        v.append({"line": "heading error, decoder over mu_z dead reckoning (deg)", "value": mz["yaw_model"],
                  "at_most": mz["yaw_latent"]})
        v.append({"line": "pelvis error, decoder over mu_z dead reckoning (m)", "value": mz["pos_model"],
                  "at_most": mz["pos_latent"]})
    for x in v:
        val = x["value"]
        ok = val is not None and math.isfinite(val)
        x["pass"] = bool(ok and (val >= x["at_least"] if "at_least" in x else val <= x["at_most"]))
    return {"verdicts": v, "pass": all(x["pass"] for x in v)}


# ---------------------------------------------------------------- windows
def walk_windows(store, hop=HOP, knots=K, max_windows=None, seed=0):
    """(takes, starts) of the full windows every hop knots of the walking takes of a BodyStore (task 'walk', not
    treadmill, at least `knots` knots); at most max_windows, drawn uniformly (seed) when there are more."""
    t = store.takes
    sel = t[(t["task_name"] == "walk") & (t["treadmill"].astype(int) == 0) & (t["n_knots"] >= knots)]
    takes, starts = [], []
    for i, r in sel.iterrows():
        n = int(r["n_knots"])
        s = list(range(0, n - knots + 1, hop))
        takes += [int(i)] * len(s)
        starts += s
    takes, starts = np.asarray(takes, np.int64), np.asarray(starts, np.int64)
    if max_windows is not None and len(takes) > max_windows:
        idx = np.sort(np.random.default_rng(seed).choice(len(takes), int(max_windows), replace=False))
        takes, starts = takes[idx], starts[idx]
    return takes, starts


def warped_bodies(store, takes, starts, factor, knots=K):
    """BodyWindows of the motion played `factor` times as fast: the take read over factor x knots from each start
    (clipped to the take), resampled to `knots` knots (linear between knots, rotations renormalized by the 6D
    Gram-Schmidt, contacts nearest), in the window frame of training at the new centre knot. Windows whose take is
    too short are left out (the returned index lists the kept ones)."""
    from mdfm.body import frames
    from mdfm.body.rotations import matrix_to_rot6d, rot6d_to_matrix
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data.body_store import BodyWindows
    n_src = int(math.ceil((knots - 1) * factor)) + 2
    keep = [j for j, (i, s) in enumerate(zip(takes, starts)) if int(store.takes["n_knots"].iloc[int(i)]) >= s + n_src]
    if not keep:
        return None, []
    src = store.windows(takes[keep], starts[keep], win_s=n_src / RATE)
    q = torch.arange(knots, dtype=torch.float64) * factor
    k0 = q.floor().long()
    f = (q - k0).to(torch.float32)
    lerp = lambda x: x[:, k0] * (1 - f).reshape((1, -1) + (1,) * (x.dim() - 2)) + \
        x[:, k0 + 1] * f.reshape((1, -1) + (1,) * (x.dim() - 2))                   # noqa: E731
    r6 = matrix_to_rot6d(rot6d_to_matrix(lerp(src.rot6d)))
    tr = lerp(src.transl)
    ct = src.contact[:, torch.round(q).long().clamp_max(n_src - 1)]
    # back into the window frame at the new centre knot (the source window's frame sat at its own centre)
    bank = SkeletonBank(torch.float32)
    rest = bank.rest(src.betas, src.gender)
    from mdfm.body.kinematics import KnotBatch
    W = r6.shape[0]
    c = knots // 2
    kb = KnotBatch(r6, tr, rest, RATE, torch.full((W,), knots))
    with torch.no_grad():
        J = kb.pose(torch.arange(W), torch.full((W,), c / RATE, dtype=r6.dtype))[0]            # (W, 22, 3)
    R0 = rot6d_to_matrix(r6[:, c, 0])
    R_yaw, origin = frames.heading_frame(J[:, None], R0[:, None], 0)
    r6f, trf = frames.frame_knots(r6, tr, rest[:, 0], R_yaw, origin)
    bw = BodyWindows(r6f.float(), trf.float(), ct, torch.ones(W, knots, dtype=torch.bool), torch.full((W,), knots),
                     src.betas, src.gender, src.person, src.task, src.take, src.start)
    return bw, keep


# ---------------------------------------------------------------- latent dead reckoning
def patch_targets(yaw, xz, L, rate=RATE):
    """(P, 3) per-patch targets (mean yaw rate rad/s, forward and lateral velocity m/s in the heading frame at the
    patch centre) and (P,) patch-centre knots of a window's true yaw (T,) and pelvis xz (T, 2)."""
    y = np.unwrap(np.asarray(yaw, np.float64)[:L])
    p = np.asarray(xz, np.float64)[:L]
    vt = np.gradient(p, 1.0 / rate, axis=0)
    rt = np.gradient(y, 1.0 / rate)
    P = min(int(math.ceil((L - 1) / PATCH - 1e-9)), 40)
    out, ks = [], []
    for j in range(P):
        a, b = j * PATCH, min((j + 1) * PATCH, L - 1)
        k = min(int(round((j + 0.5) * PATCH)), L - 1)
        v = vt[a:b + 1].mean(0)
        ps = y[k]
        out.append((rt[a:b + 1].mean(), v[0] * math.sin(ps) + v[1] * math.cos(ps),
                    v[0] * math.cos(ps) - v[1] * math.sin(ps)))
        ks.append(k)
    return np.asarray(out), np.asarray(ks)


def integrate_patches(rate_, fwd, lat, ks, c, rate=RATE):
    """(heading (P,), position (P, 2)) at the patch centres of per-patch yaw rates and heading-frame velocities,
    integrated over the 0.25 s patches (midpoint values of the cumulative sums) from the patch nearest the centre
    knot c, carried from that patch's centre to c at its own rates, so that the heading and the position are 0 at c
    (the diagnosis' latent_dr, whose centre correction had the wrong sign and left the position uncorrected)."""
    pc = int(np.argmin(np.abs(ks - c)))
    off = (ks[pc] - c) / rate                                   # s from c to the centre patch's centre
    dpsi = rate_ * PATCH / rate
    head = np.cumsum(dpsi) - dpsi / 2
    head = head - head[pc] + off * rate_[pc]
    vx = fwd * np.sin(head) + lat * np.cos(head)
    vz = fwd * np.cos(head) - lat * np.sin(head)
    h = PATCH / rate
    px, pz = np.cumsum(vx * h) - vx * h / 2, np.cumsum(vz * h) - vz * h / 2
    return head, np.stack([px - px[pc] + off * vx[pc], pz - pz[pc] + off * vz[pc]], 1)


def fit_ridge(X, y, groups, alphas=np.logspace(-2, 4, 13)):
    """A ridge readout of standardized features, alpha by person-grouped 4-fold validation (R^2)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    best, ba = -np.inf, alphas[0]
    n_groups = len(set(groups))
    if n_groups >= 4:
        for a in alphas:
            sc = []
            for tr, te in GroupKFold(n_splits=4).split(Z, y, groups):
                p = Ridge(alpha=a).fit(Z[tr], y[tr]).predict(Z[te])
                sc.append(1 - ((y[te] - p) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12))
            if np.mean(sc) > best:
                best, ba = float(np.mean(sc)), a
    m = Ridge(alpha=ba).fit(Z, y)
    return lambda Xn: m.predict((Xn - mu) / sd), float(ba)


@torch.no_grad()
def latent_features(model, windows, device, amp=True, batch_windows=8):
    """[(mu_z (P, 32), pre-LN encoder state (P, d))] per window at its latent positions (the posterior means of
    model.encode; a forward pre-hook on the encoder's final LayerNorm)."""
    from mdfm.model import mfm as M
    out = []
    grab = []
    hook = model.encoder.norm.register_forward_pre_hook(lambda mod, inp: grab.append(inp[0].detach()))
    try:
        for a in range(0, len(windows), batch_windows):
            ch = windows[a:a + batch_windows]
            batch = M.batch_from_windows(ch, sources=model.cfg.sources, families=model.heads.families,
                                         rate=model.rate, gen=0).to(device)
            grab.clear()
            on = bool(amp and device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on):
                post = model.encode(batch)
            h = grab[0].float()
            R, Lr, d = h.shape
            lat = batch.tokens.lat_idx.to(device)
            pre = h.reshape(R * Lr, d)[lat.clamp_min(0)]
            for w in range(len(ch)):
                ok = (lat[w] >= 0).cpu().numpy()
                out.append((post.mu_z[w].float().cpu().numpy()[ok], pre[w].cpu().numpy()[ok]))
    finally:
        hook.remove()
    return out


def latent_rows(feats, wins, decoded=None):
    """Per-window patch tables: (X_mu (P, 32), X_pre (P, d), targets (P, 3), ks, L, true yaw at ks about c, true
    position at ks from c, decoded yaw and position at ks (or None), person)."""
    rows = []
    for j, ((mu, pre), wd) in enumerate(zip(feats, wins)):
        body = wd.body
        L = int(body["lengths"])
        from mdfm.body.rotations import rot6d_to_matrix, yaw_angle
        yaw = yaw_angle(rot6d_to_matrix(torch.as_tensor(np.asarray(body["rot6d"])[:L, 0], dtype=torch.float64))).numpy()
        xz = np.asarray(body["transl"], np.float64)[:L][:, [0, 2]]
        Y, ks = patch_targets(yaw, xz, L)
        P = min(len(ks), mu.shape[0], pre.shape[0])
        c = L // 2
        yt = _wrap(yaw - yaw[c])
        dec = None
        if decoded is not None:
            d = decoded[j]
            yp = _wrap(d["yaw_pred"] - d["yaw_pred"][c])
            pp = d["transl_pred"][:, [0, 2]] - d["transl_pred"][c, [0, 2]]
            dec = (yp[ks[:P]], pp[ks[:P]])
        rows.append({"mu": mu[:P], "pre": pre[:P], "Y": Y[:P], "ks": ks[:P], "L": L, "yaw": yt[ks[:P]],
                     "pos": (xz - xz[c])[ks[:P]], "dec": dec, "person": str(wd.person_id)})
    return rows


def latent_dead_reckoning(train_rows, rows):
    """{mu_z, pre}: {yaw_latent, yaw_model, yaw_zero (deg), pos_latent, pos_model, pos_zero (m), r2 of the three
    readouts on the probed windows, alphas, windows}: medians over windows of the per-window means over the scored
    patches (module docstring)."""
    out = {}
    groups = np.concatenate([[r["person"]] * len(r["ks"]) for r in train_rows])
    for feat in ("mu", "pre"):
        Xtr = np.concatenate([r[feat] for r in train_rows])
        Ytr = np.concatenate([r["Y"] for r in train_rows])
        fits = [fit_ridge(Xtr, Ytr[:, j], groups) for j in range(3)]
        ye, pe, me, mpe, ze, zpe, Yall, Pall = [], [], [], [], [], [], [], []
        for r in rows:
            pr = np.stack([f(r[feat]) for f, _ in fits], 1)
            Yall.append(r["Y"])
            Pall.append(pr)
            head, pos = integrate_patches(pr[:, 0], pr[:, 1], pr[:, 2], r["ks"], r["L"] // 2)
            s = slice(DR_PATCHES[0], min(DR_PATCHES[1], len(r["ks"])))
            ye.append(math.degrees(np.abs(_wrap(head[s] - r["yaw"][s])).mean()))
            pe.append(float(np.linalg.norm(pos[s] - r["pos"][s], axis=1).mean()))
            ze.append(math.degrees(np.abs(r["yaw"][s]).mean()))
            zpe.append(float(np.linalg.norm(r["pos"][s], axis=1).mean()))
            if r["dec"] is not None:
                me.append(math.degrees(np.abs(_wrap(r["dec"][0][s] - r["yaw"][s])).mean()))
                mpe.append(float(np.linalg.norm(r["dec"][1][s] - r["pos"][s], axis=1).mean()))
        Yall, Pall = np.concatenate(Yall), np.concatenate(Pall)
        r2 = [float(1 - ((Yall[:, j] - Pall[:, j]) ** 2).sum() / max(((Yall[:, j] - Yall[:, j].mean()) ** 2).sum(),
                                                                       1e-12)) for j in range(3)]
        med = lambda x: float(np.median(x)) if x else None                           # noqa: E731
        out["mu_z" if feat == "mu" else "pre_ln"] = {
            "yaw_latent": med(ye), "yaw_model": med(me), "yaw_zero": med(ze), "pos_latent": med(pe),
            "pos_model": med(mpe), "pos_zero": med(zpe), "r2_yaw_rate": r2[0], "r2_forward": r2[1], "r2_lateral": r2[2],
            "alphas": [a for _, a in fits], "windows": len(rows)}
    return out


# ---------------------------------------------------------------- the run
def render(store, takes, starts, sampler, specs_fn, seed, chunk=64, bodies=None):
    """Rendered windows.Window list of store windows (or of given BodyWindows), specs from specs_fn(n, gen)."""
    from mdfm.factory.synth import select_bodies
    gen = np.random.default_rng(seed)
    out = []
    n = len(takes) if bodies is None else int(bodies.lengths.shape[0])
    for a in range(0, n, chunk):
        idx = np.arange(a, min(a + chunk, n))
        b = store.windows(takes[idx], starts[idx]) if bodies is None else select_bodies(bodies, idx)
        out += sampler.render(b, specs_fn(len(idx), gen), int(gen.integers(1 << 62)))
    return out


def decode_windows(model, wins, refine_steps, seed, batch_windows=8):
    """Window dicts for root_lines of rendered windows decoded by eval.infer.infer (z = mu_z)."""
    from mdfm.body.rotations import rot6d_to_matrix, yaw_angle
    from mdfm.eval import infer as I
    bodies = I.infer(model, wins, refine_steps=refine_steps, points=I.POINTS_EVAL, batch_windows=batch_windows,
                     seed=seed)
    out = []
    for wd, b in zip(wins, bodies):
        n = len(b.transl)
        body = wd.body
        r6t = torch.as_tensor(np.asarray(body["rot6d"])[:n], dtype=torch.float64)
        yt = yaw_angle(rot6d_to_matrix(r6t[:, 0])).numpy()
        yp = yaw_angle(rot6d_to_matrix(torch.as_tensor(b.rot6d[:, 0], dtype=torch.float64))).numpy()
        out.append({"transl_pred": np.asarray(b.transl, np.float64),
                    "transl_true": np.asarray(body["transl"], np.float64)[:n],
                    "yaw_pred": yp, "yaw_true": yt, "joints_pred": np.asarray(b.joints, np.float64),
                    "rel_pred": np.asarray(b.joints, np.float64) - np.asarray(b.joints, np.float64)[:, :1],
                    "contact_pred": np.asarray(b.contact, np.float64), "beta": b.beta, "gender": b.gender,
                    "layout": str((wd.streams[0].meta or {}).get("layout")) if wd.streams else "",
                    "contact_true": np.asarray(body["contact"], np.float64)[:n], "rot6d_true": r6t.numpy()})
    return out


def add_true_rel(dws, bank):
    """rel_true (T, 22, 3): the truth's pelvis-relative joints (forward kinematics on the take's skeleton)."""
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.body.smplh import fk
    for d in dws:
        rest = bank.rest(torch.as_tensor(d["beta_true"], dtype=torch.float64)[None],
                         torch.tensor([int(d["gender_true"])]))[0]
        n = len(d["transl_true"])
        pos, _ = fk(rot6d_to_matrix(torch.as_tensor(d["rot6d_true"], dtype=torch.float64)),
                    torch.zeros(n, 3, dtype=torch.float64), rest)
        d["rel_true"] = (pos - pos[:, :1]).numpy()


def load_model(checkpoint, weights, device, cfg):
    """(model, describe dict, name): eval.infer.load_checkpoint, or 'init' (the pilot model at cfg.seed)."""
    from mdfm.eval import infer as I
    from mdfm.model import mfm as M
    if checkpoint in (None, "init"):
        torch.manual_seed(int(cfg.seed))
        model = M.MFM(cfg.model.mfm_config()).to(device).eval()
        model.requires_grad_(False)
        return model, {"path": "init", "sha256": None}, "init", frozenset()
    loaded = I.load_checkpoint(checkpoint, weights=weights, device=device, cfg=cfg)
    return loaded.model, loaded.describe(), loaded.name, loaded.trained_persons()


def _warps_arg(text):
    try:
        return parse_warps(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse(argv=None):
    """The command line (module docstring); --warps defaults to WARPS."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default="init")
    ap.add_argument("--weights", default="ema", choices=("ema", "model"))
    ap.add_argument("--split", default="dev", choices=("dev", "train"))
    ap.add_argument("--layout", default="xsens13")
    ap.add_argument("--mixture", action="store_true", help="layouts drawn from the config's mixture")
    ap.add_argument("--max-windows", type=int, default=480)
    ap.add_argument("--refine-steps", type=int, default=0)
    ap.add_argument("--no-warp", action="store_true")
    ap.add_argument("--warps", type=_warps_arg, default=WARPS, metavar="F[,F]",
                    help="time-warp factors (default " + ",".join(f"{f:g}" for f in WARPS) + ")")
    ap.add_argument("--latent", action="store_true", help="the latent dead reckoning (fits on train renders)")
    ap.add_argument("--train-windows", type=int, default=480)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap-gb", type=float, default=CAP_GB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    a.warps = tuple(a.warps)
    return a


def main():
    a = parse()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import amass_index
    from mdfm.data import body_store as BS
    from mdfm.factory import layouts as LY
    from mdfm.factory import synth as SY
    from mdfm.train import config as CF
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV
    t0 = time.perf_counter()
    dev = torch.device(a.device)
    LP.cap_memory(dev, a.cap_gb)
    cfg = CF.load(a.config)
    model, desc, name, trained = load_model(a.checkpoint, a.weights, dev, cfg)
    store = BS.BodyStore(split=a.split)
    takes, starts = walk_windows(store, max_windows=a.max_windows, seed=a.seed)
    persons = sorted(set(store.takes["person_id"].iloc[takes].astype(str)))
    if a.split != "train" and set(persons) & set(trained):
        clash = sorted(set(persons) & set(trained))
        raise ValueError(f"probed persons among the checkpoint's training persons: {clash[:3]}")
    lays = LY.load_layouts()
    mixture = dict(cfg.mixture.layouts)
    specs_fn = (lambda n, g: [SY.RenderSpec(LY.sample_layout(g, mixture)) for _ in range(n)]) if a.mixture else \
        (lambda n, g: [SY.RenderSpec(lays[a.layout])] * n)
    sampler = SY.SynthSampler(store=store, device=dev)
    print(f"{name}: {len(takes)} {a.split} walking windows of {len(persons)} persons", flush=True)
    wins = render(store, takes, starts, sampler, specs_fn, a.seed)
    dws = decode_windows(model, wins, a.refine_steps, a.seed)
    bank = SkeletonBank(torch.float64)
    for d, wd in zip(dws, wins):
        d["beta_true"], d["gender_true"] = np.asarray(wd.body["betas"]), int(wd.body["gender"])
    add_true_rel(dws, bank)
    lines = root_lines(dws)
    print(json.dumps({"bins": {k: v["speed_ratio"] for k, v in lines["bins"].items()},
                      "lateral_slope": lines["lateral_slope"]}), flush=True)
    warp, w_rows = None, None                   # w_rows: the warp rows (rows is the latent block's own name)
    if not a.no_warp:
        warped, base_idx = {}, None
        for f in a.warps:
            bw, keep = warped_bodies(store, takes, starts, f)
            base_idx = keep if base_idx is None else [k for k in base_idx if k in set(keep)]
            warped[f] = (bw, keep)
        base_idx = sorted(base_idx or [])
        wd_f = {}
        for f, (bw, keep) in warped.items():
            if bw is None or not base_idx:
                continue
            pos = [keep.index(k) for k in base_idx]
            from mdfm.factory.synth import select_bodies
            sub = select_bodies(bw, np.asarray(pos))
            ws = render(store, takes, starts, sampler, specs_fn, a.seed, bodies=sub)
            wd_f[f] = decode_windows(model, ws, a.refine_steps, a.seed)
            for d, wd in zip(wd_f[f], ws):                  # the true joints of the warp rows' stride errors
                d["beta_true"], d["gender_true"] = np.asarray(wd.body["betas"]), int(wd.body["gender"])
            add_true_rel(wd_f[f], bank)
        if wd_f:
            warp = warp_lines([dws[k] for k in base_idx], wd_f)
            warp["windows"] = len(base_idx)
            w_rows = warp_rows([dws[k] for k in base_idx], wd_f)
    latent, train_sha = None, None
    if a.latent:
        tstore = BS.BodyStore(split="train")
        tt, ts = walk_windows(tstore, max_windows=a.train_windows, seed=a.seed + 1)
        tsampler = SY.SynthSampler(store=tstore, device=dev)
        twins = render(tstore, tt, ts, tsampler, specs_fn, a.seed + 1)
        tp = sorted(set(str(w.person_id) for w in twins))
        if set(tp) & set(persons):
            raise ValueError("latent-fit persons among the probed persons")
        train_sha = amass_index.person_list_sha256(tp)
        train_rows = latent_rows(latent_features(model, twins, dev), twins)
        rows = latent_rows(latent_features(model, wins, dev), wins, decoded=dws)
        latent = latent_dead_reckoning(train_rows, rows)
        del tsampler
    verdict = gate(lines, warp, latent)
    tag = f"_{a.tag}" if a.tag else ""
    stem = f"root_motion_{name}{tag}"
    doc = {"script": "scripts/probe_root_motion.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "checkpoint": desc, "weights": a.weights, "split": a.split,
           "layout": "mixture" if a.mixture else a.layout, "refine_steps": a.refine_steps, "seed": a.seed,
           "windows": len(wins), "persons": len(persons), "persons_sha256": amass_index.person_list_sha256(persons),
           "latent_fit_persons_sha256": train_sha, "moveatlas_head": PV.moveatlas_head(),
           "code_sha256": PV.code_sha256(), "config_sha256": CF.sha256(cfg), "lines": lines, "warp": warp,
           "latent": latent, "gate": verdict, "gate_lines": LINES, "seconds": time.perf_counter() - t0}
    if w_rows is not None:                      # also in the main json: the file scripts/weekend_branch.py reads (M1)
        doc["warp_rows"] = w_rows
    if not a.no_warp and tuple(a.warps) != tuple(WARPS):
        doc["warps"] = list(a.warps)
        doc["gate_note"] = warp_note(a.warps)
    os.makedirs(a.out, exist_ok=True)
    js, md = os.path.join(a.out, stem + ".json"), os.path.join(a.out, stem + ".md")
    with open(js, "w", encoding="utf-8", newline="\n") as f:
        json.dump(LP._finite(doc), f, indent=1)
        f.write("\n")
    wj = None
    if w_rows is not None:
        wj = os.path.join(a.out, stem + "_warps.json")
        with open(wj, "w", encoding="utf-8", newline="\n") as f:
            json.dump(LP._finite(warps_doc(doc, w_rows)), f, indent=1)
            f.write("\n")
    write_md(md, js, doc, wj)
    print(json.dumps(LP._finite(verdict), indent=1))
    print(f"wrote {md} and {js}" + (f" and {wj}" if wj else ""))


def warp_note(warps):
    """The report's note when --warps differs from the registered WARPS."""
    return ("time-warp factors " + ", ".join(f"x{f:g}" for f in warps) + " instead of the registered "
            + ", ".join(f"x{f:g}" for f in WARPS) + ": the warp lines of this gate are not the registered G-g read")


def warps_doc(doc, rows):
    """The <stem>_warps.json document: the run's identity and the warp rows (warp_rows)."""
    keys = ("script", "run_at", "checkpoint", "weights", "split", "layout", "refine_steps", "seed", "persons_sha256",
            "moveatlas_head", "code_sha256", "config_sha256")
    return {**{k: doc[k] for k in keys}, "warps": [r["factor"] for r in rows[1:]], "rows": rows}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_md(md, js, doc, warps_js=None):
    note = f" Note: {doc['gate_note']}." if doc.get("gate_note") else ""
    L = [f"# Root motion of {doc['checkpoint'].get('path')} (gate G-g)", "",
         f"Written by scripts/probe_root_motion.py on {doc['run_at']} (reports/bench/root_motion_diagnosis.md, fix 7). "
         f"Checkpoint SHA-256 {doc['checkpoint'].get('sha256')}, {doc['weights']} weights; {doc['windows']} windows of "
         f"{doc['persons']} {doc['split']} persons (SHA-256 {doc['persons_sha256'][:12]}), layout {doc['layout']}, "
         f"{doc['refine_steps']} refinement steps; MoveAtlas {doc['moveatlas_head'][:7]}, code SHA-256 "
         f"{doc['code_sha256'][:12]}, config SHA-256 {doc['config_sha256'][:12]}.", "",
         f"Gate G-g: {'PASS' if doc['gate']['pass'] else 'FAIL'}.{note}", "", "| line | value | bound | pass |",
         "|---|---|---|---|"]
    for v in doc["gate"]["verdicts"]:
        bound = f"at least {fmt(v['at_least'])}" if "at_least" in v else f"at most {fmt(v['at_most'])}"
        L.append(f"| {v['line']} | {fmt(v['value'])} | {bound} | {'yes' if v['pass'] else 'no'} |")
    L += ["", "## By time from the centre knot", "",
          "| bin s | speed ratio | velocity cosine | velocity-heading cosine (truth) | yaw error deg | heading gain | "
          "heading corr | knots |", "|---|---|---|---|---|---|---|---|"]
    for name, b in doc["lines"]["bins"].items():
        L.append(f"| {name} | {fmt(b['speed_ratio'])} | {fmt(b['velocity_cosine'])} | "
                 f"{fmt(b['velocity_heading_cosine'])} ({fmt(b['velocity_heading_cosine_true'])}) | "
                 f"{fmt(b['yaw_err_deg'], 1)} | {fmt(b['heading_gain'])} | {fmt(b['heading_corr'])} | {b['knots']} |")
    lg = doc["lines"]["legs"]
    L += ["", f"Displacement slopes: lateral {fmt(doc['lines']['lateral_slope'])}, along the centre heading "
          f"{fmt(doc['lines']['forward_slope'])}. Legs ({lg['windows']} windows): stance-foot no-slip speed of the "
          f"decoded pose over the true speed {fmt(lg['legs_over_true_median'])} (the truth's own "
          f"{fmt(lg['truth_legs_over_true_median'])}), decoded root speed over it {fmt(lg['root_over_legs_median'])}.",
          "", "Heading gain by layout: " + "; ".join(f"{k} {fmt(v['gain'])} ({v['knots']} knots)"
                                                     for k, v in sorted(doc["lines"]["heading_gain_by_layout"].items()))
          + "."]
    if doc["warp"]:
        L += ["", "Time warp (" + str(doc["warp"].get("windows")) + " windows): " + "; ".join(
            f"x{f}: decoded {fmt(w['decoded_ratio'])} against true {fmt(w['true_ratio'])}"
            for f, w in doc["warp"].items() if f != "windows") + "."
            + (f" One row per factor, the unwarped one included: {os.path.basename(warps_js)}." if warps_js else "")]
        sr = [r for r in doc.get("warp_rows") or [] if r.get("stride_mae_cm") is not None]
        if sr:
            L[-1] += (" Stride MAE of the decoded body at the true ICs: " + "; ".join(
                f"x{r['factor']:g} {fmt(r['stride_mae_cm'], 2)} cm (bias {fmt(r['stride_bias_cm'], 2)} cm, "
                f"{r['strides']} strides)" for r in sr) + ".")
    if doc["latent"]:
        L += ["", "## Latent dead reckoning (medians over windows)", "",
              "| features | yaw latent deg | yaw decoder deg | yaw zero deg | pelvis latent m | pelvis decoder m | "
              "pelvis zero m | R^2 yaw rate, forward, lateral |", "|---|---|---|---|---|---|---|---|"]
        for k, v in doc["latent"].items():
            L.append(f"| {k} | {fmt(v['yaw_latent'], 1)} | {fmt(v['yaw_model'], 1)} | {fmt(v['yaw_zero'], 1)} | "
                     f"{fmt(v['pos_latent'])} | {fmt(v['pos_model'])} | {fmt(v['pos_zero'])} | "
                     f"{fmt(v['r2_yaw_rate'], 2)}, {fmt(v['r2_forward'], 2)}, {fmt(v['r2_lateral'], 2)} |")
    L += ["", f"Every number above comes from reports/bench/{os.path.basename(js)}.", ""]
    with open(md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
