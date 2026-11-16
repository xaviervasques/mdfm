"""Seam measures of a chained generated clip on its estimated keypoints (task VCH): what a seam between two
generator segments does to the measures of section 3.4 of docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md.

scripts/video_generate.py chains the segments of a clip longer than 77 frames by Wan2.2's scheme (one or five
guidance frames from the previous segment) and records a pixel measure per seam; the measures here need the
keypoints an estimator reads from the generated clip (the stage-1 round trip), in the drawn order
(points.wholebody133_to_drawn), beside the driver's truth. The truth is continuous across a seam, so whatever the
estimate does there and the truth does not is the seam's:

- phase (seam_phases): for a tremor clip, the phase jump at each seam of a sinusoid at the driven frequency fitted
  one window before and one after the seam (analysis.seam_phase_jumps) on the tremor signal (the wrist minus the
  shoulder along the truth's driven direction), estimate minus truth, wrapped to [-180, 180) degrees;
- jerk onsets (seam_jerks): per joint angle (analysis.joint_angles after the low-pass of analysis.py), the jerk onsets
  of the estimate (analysis.jerk_onsets) within tol_s of a seam that no truth onset matches within tol_s
  (analysis.match_events): spurious onsets at the seam;
- control windows: the same counts at points halfway between seams (and the clip's ends), the reference a seam
  count is read against.

No threshold is set here: the rule that reads these numbers belongs to the stage-1 rule file (proposal section 8).
"""
import numpy as np

from . import analysis as A

TOL_S = 0.25                      # [cfg] seam window for spurious onsets (analysis.MATCH_TOL_S)
ANGLES = ("trunk", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_hip", "r_hip", "l_knee", "r_knee")


def seam_phases(uv_est, uv_truth, fps, freq_hz, seams, side="l", win_s=1.0):
    """[{seam, jump_est_deg, jump_truth_deg, jump_deg (estimate minus truth, wrapped), amp_before_px,
    amp_after_px}] of a tremor clip at each seam frame (module docstring)."""
    d = A.driven_direction(uv_truth, side)
    xe = A.tremor_difference(uv_est, side) @ d
    xt = A.tremor_difference(uv_truth, side) @ d
    je = A.seam_phase_jumps(xe, fps, freq_hz, seams, win_s)
    jt = A.seam_phase_jumps(xt, fps, freq_hz, seams, win_s)
    out = []
    for e, t in zip(je, jt):
        out.append({"seam": e["seam"], "jump_est_deg": e["jump_deg"], "jump_truth_deg": t["jump_deg"],
                    "jump_deg": float((e["jump_deg"] - t["jump_deg"] + 180.0) % 360.0 - 180.0),
                    "amp_before_px": e["amp_before"], "amp_after_px": e["amp_after"]})
    return out


def control_points(seams, n_frames):
    """Frames halfway between consecutive seams and between the clip's ends and its first and last seam."""
    pts = [0] + sorted(int(s) for s in seams) + [int(n_frames) - 1]
    return [int((a + b) // 2) for a, b in zip(pts[:-1], pts[1:])]


def _spurious(est, truth, at, tol):
    """Estimate onsets within tol of a frame of at that no truth onset matches within tol: {frame: count}."""
    est = np.asarray(est, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    out = {}
    for s in at:
        near = est[np.abs(est - s) <= tol]
        tr = truth[np.abs(truth - s) <= 2 * tol]
        _, _, un_e = A.match_events(tr, near, tol)
        out[int(s)] = int(un_e)
    return out


def seam_jerks(uv_est, uv_truth, fps, seams, tol_s=TOL_S, angles=ANGLES, cutoff_hz=A.LOWPASS_HZ):
    """{per_angle: {angle: {seam frame: spurious onsets}}, seams: {seam: total}, controls: {frame: total},
    tol_frames} (module docstring); keypoints (T, >= 20, 2) in the drawn order."""
    from . import points as P
    ue = A.lowpass(np.asarray(uv_est, dtype=np.float64)[:, :P.N_BODY20], fps, cutoff_hz)
    ut = A.lowpass(np.asarray(uv_truth, dtype=np.float64)[:, :P.N_BODY20], fps, cutoff_hz)
    ae, at = A.joint_angles(ue), A.joint_angles(ut)
    tol = tol_s * fps
    ctrl = control_points(seams, len(ue))
    per, tot_s, tot_c = {}, {int(s): 0 for s in seams}, {c: 0 for c in ctrl}
    for name in angles:
        oe, ot = A.jerk_onsets(ae[name], fps), A.jerk_onsets(at[name], fps)
        s_counts = _spurious(oe, ot, seams, tol)
        c_counts = _spurious(oe, ot, ctrl, tol)
        per[name] = s_counts
        for k, v in s_counts.items():
            tot_s[k] += v
        for k, v in c_counts.items():
            tot_c[k] += v
    return {"per_angle": per, "seams": tot_s, "controls": tot_c, "tol_frames": tol}
