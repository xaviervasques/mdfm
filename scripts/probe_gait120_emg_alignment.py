"""gait120 EMG to pose alignment on every non-lockbox trial (plan task B4.5; spec
docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, section 2.2 and risk 11).

The rule of mdfm.data.sources: pose step k of a trial covers frames step_target_frames_100hz[k] of the
EMG record of the same trial (n_frames = last - first + 1), consecutive steps share their boundary
frame, EMG sample 0 is frame total_frame_100hz[0] and the trial holds (last - first + 1) x 20 samples
at 2,000 Hz. Per trial (every admitted, that is non-lockbox, gait120 trial of the record index):
- structure: the status of sources.pair (ok or the reason it refuses), steps, contiguity, the largest
  distance between the two copies of a shared boundary frame (mm), pose frames inside the EMG frames;
- kinematic event (level_walking, stair_*, slope_*): per step, the frame of the most anterior right
  ankle relative to the pelvis (terminal swing, just before right initial contact) among the 25 frames
  that end the step; lag = step end minus that frame, in ms. The forward axis is up x (left hip minus
  right hip), up = +y (gait120 records are y-up), hips being the ASIS proxies;
- EMG against the kinematic event: linear envelopes (4th-order zero-phase Butterworth high-pass at
  20 Hz, full-wave rectification, 4th-order zero-phase low-pass at 6 Hz) of tibialis_anterior and
  gastrocnemius_medialis, read at each pose frame by the rule; per step the phase (percent of the step)
  of each envelope peak; the same under the rule shifted by +100 ms (EMG sample 0 at frame
  total_frame_100hz[0] + 10), which shows what the check resolves.
Bands fixed before the first run, level walking only (healthy gait, right-leg cycle from right initial
contact; Perry and Burnfield, Gait Analysis, 2nd ed., 2010): tibialis anterior peak within 15 percent of
a step end (loading response or terminal swing), gastrocnemius medialis peak between 25 and 60 percent
(terminal stance). sit_to_stand and stand_to_sit get the structural checks only: their records are
root-centred, so no pelvis event exists.
Writes reports/stage1_sources/gait120_alignment.csv (one row per trial), gait120_alignment.json and
gait120_alignment.md. Run: <conda>/envs/mdfm/python.exe
scripts/probe_gait120_emg_alignment.py [--limit N]
"""
import argparse
import collections
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))
from mdfm import paths  # noqa: E402
from mdfm.data import sources as S  # noqa: E402
from _common import moveatlas_head  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
WALK = ("level_walking", "stair_ascent", "stair_descent", "slope_ascent", "slope_descent")
MUSCLES = {"ta": "tibialis_anterior", "gas": "gastrocnemius_medialis"}
SHIFT = 10                     # frames, 100 ms
SEARCH = 25                    # frames before a step end
TA_BAND = 15.0                 # percent of the step from either end
GAS_BAND = (25.0, 60.0)        # percent


def envelope(x, fps):
    b, a = butter(4, 20.0 / (fps / 2), "high")
    y = np.abs(filtfilt(b, a, x, axis=0))
    b, a = butter(4, 6.0 / (fps / 2), "low")
    return filtfilt(b, a, y, axis=0)


def task_of(trial):
    return trial.rsplit("_trial", 1)[0]


def analyse(pr, task):
    out = {}
    info = pr.pairing
    frames = np.asarray(info["pose_frames"])
    f0, f1 = info["emg_frames"]
    steps = info["step_frames"]
    out["steps"] = len(steps)
    out["contiguous"] = all(steps[k][1] == steps[k + 1][0] for k in range(len(steps) - 1))
    out["boundary_diff_mm"] = round(1000 * max(info["boundary_diff_m"]), 4) if info["boundary_diff_m"] else ""
    out["pose_inside_emg"] = bool(frames[0] >= f0 and frames[-1] <= f1)
    out["emg_samples"] = int(len(pr.t["emg"]))
    out["emg_samples_rule"] = int((f1 - f0 + 1) * 20)
    out["emg_frames"] = f"{f0}-{f1}"
    out["pose_frames"] = f"{int(frames[0])}-{int(frames[-1])}"
    out["pose_frame_count"] = int(len(frames))
    out["pose_frames_unique"] = bool(len(np.unique(frames)) == len(frames))
    out["shared_frames"] = "|".join(str(steps[k][1]) for k in range(len(steps) - 1) if steps[k][1] == steps[k + 1][0])
    out["step_scales"] = "|".join(f"{x:.4f}" for x in info["step_scales"])
    out["emg_offset_s"] = round(float(pr.offsets_s["emg"]), 4)
    if task not in WALK:
        return out
    pose = pr.streams["pose3d"]
    X = np.asarray(pose["coords"], np.float64) * float(pose["scale"])
    lat = X[:, 9] - X[:, 10]
    fwd = np.cross(lat, [0.0, 1.0, 0.0])
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    ant = np.einsum("ij,ij->i", X[:, 14] - X[:, 0], fwd)
    emg = pr.streams["emg"]
    names = [c.get("muscle") for c in emg["channels"]]
    env = {k: envelope(np.asarray(emg["signal"], np.float64)[:, names.index(m)], float(emg["fps"]))
           for k, m in MUSCLES.items()}
    lags, ph = [], collections.defaultdict(list)
    pos = {int(f): i for i, f in enumerate(frames)}
    for a, b in steps:
        seg = [pos[f] for f in range(max(a, b - SEARCH), b + 1) if f in pos]
        if seg:
            lags.append(10.0 * (b - frames[seg[int(np.argmax(ant[seg]))]]))
        fr = np.arange(a, b + 1)
        for shift, tag in ((0, ""), (SHIFT, "_shifted")):
            idx = (fr - f0 - shift) * 20
            if idx.min() < 0 or idx.max() >= len(env["ta"]):
                continue
            for k in MUSCLES:
                ph[(k, tag)].append(100.0 * int(np.argmax(env[k][idx])) / (b - a))
    out["kin_lag_ms"] = round(float(np.mean(lags)), 1) if lags else ""
    for (k, tag), v in ph.items():             # TA: distance to the nearest step end; GAS: phase
        out[f"{k}_phase{tag}"] = round(float(np.mean(v)), 2) if k == "gas" else \
            round(float(np.mean([min(p, 100 - p) for p in v])), 2)
    return out


def stats(v):
    v = np.asarray([x for x in v if x != "" and x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 2), "p10": round(float(np.percentile(v, 10)), 2),
            "p90": round(float(np.percentile(v, 90)), 2)}


def normalize(obj):
    """numpy scalars to Python, integral floats (and keys such as '2.0') to int, for the JSON and the page."""
    if isinstance(obj, dict):
        return {normalize(_key(k)): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and np.isfinite(obj) and obj.is_integer() and abs(obj) < 2 ** 53:
        return int(obj)
    return obj


def _key(k):
    try:
        f = float(k)
    except (TypeError, ValueError):
        return k
    return int(f) if f.is_integer() else k


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(TREE, "reports", "stage1_sources"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--md-only", action="store_true", help="rewrite the page (and normalize the JSON) of the last run")
    a = ap.parse_args()
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    if a.md_only:
        p = os.path.join(a.out, "gait120_alignment.json")
        with open(p, encoding="utf-8") as f:
            summ = normalize(json.load(f))
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summ, f, indent=1, default=str)
        write_md(a.out, summ)
        return
    t_start = time.time()
    adm = S.rows("gait120")
    groups = list(adm.groupby(["person_id", "subject_ns", "recording"], sort=True))
    if a.limit:
        groups = groups[:: max(1, len(groups) // a.limit)][: a.limit]
    rows = []
    for n, ((pid, sns, trial), g) in enumerate(groups):
        task = task_of(trial)
        r = {"subject_ns": sns, "trial": trial, "task": task, "pose_steps": int((g["stream"] == "pose3d").sum()),
             "emg_record": bool((g["stream"] == "emg").any())}
        try:
            pr = S.pair("gait120", S.load(g.to_dict("records")))
            r["status"] = "ok"
            r.update(analyse(pr, task))
        except S.PairingError as e:
            r["status"] = e.reason
        rows.append(r)
        if n % 500 == 0:
            print(f"{n} / {len(groups)} trials, {time.time() - t_start:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    for k in ("kin_lag_ms", "ta_phase", "gas_phase", "ta_phase_shifted", "gas_phase_shifted"):
        if k not in df:
            df[k] = np.nan
    df.to_csv(os.path.join(a.out, "gait120_alignment.csv"), index=False, lineterminator="\n")
    ok = df[df["status"] == "ok"]
    summ = {"moveatlas_head": moveatlas_head(paths.MOVEATLAS), "run_at": dt.datetime.now().isoformat(timespec="seconds"),
            "script": os.path.basename(__file__), "limit": a.limit,
            "trials": len(df), "trials_in_index": int(adm.groupby(["subject_ns", "recording"]).ngroups),
            "persons": int(df["subject_ns"].nunique()), "status": df["status"].value_counts().to_dict(),
            "ok": {"emg_samples_equal_rule": int((ok["emg_samples"] == ok["emg_samples_rule"]).sum()),
                   "pose_inside_emg": int(ok["pose_inside_emg"].sum()),
                   "contiguous": int(ok["contiguous"].sum()),
                   "steps": {int(k): int(v) for k, v in ok["steps"].value_counts().items()},
                   "boundary_diff_mm_max": float(pd.to_numeric(ok["boundary_diff_mm"], errors="coerce").max()),
                   "pose_frames_unique": int(ok["pose_frames_unique"].sum())},
            "per_task": {}}
    s002 = df[(df["subject_ns"] == "gait120:S002") & (df["trial"] == "level_walking_trial01")]
    if len(s002):
        r = s002.iloc[0]
        summ["s002_level_walking_trial01"] = {k: r[k] for k in ("status", "emg_frames", "pose_frames", "pose_frame_count",
                                                                 "shared_frames", "step_scales", "emg_samples",
                                                                 "emg_offset_s")}
    summ = normalize(summ)
    for task, g in ok.groupby("task"):
        t = {"trials": len(g), "steps": int(g["steps"].sum())}
        if task in WALK:
            t["kin_lag_ms"] = stats(pd.to_numeric(g["kin_lag_ms"], errors="coerce"))
            for k in ("ta_phase", "gas_phase", "ta_phase_shifted", "gas_phase_shifted"):
                t[k] = stats(pd.to_numeric(g[k], errors="coerce"))
            if task == "level_walking":
                for tag in ("", "_shifted"):
                    ta = pd.to_numeric(g[f"ta_phase{tag}"], errors="coerce")
                    gas = pd.to_numeric(g[f"gas_phase{tag}"], errors="coerce")
                    t[f"in_band{tag}"] = {"ta_within_15_of_a_step_end": int((ta <= TA_BAND).sum()),
                                          "gas_in_25_60": int(gas.between(*GAS_BAND).sum()),
                                          "both": int(((ta <= TA_BAND) & gas.between(*GAS_BAND)).sum()),
                                          "scored": int((ta.notna() & gas.notna()).sum())}
        summ["per_task"][task] = t
    summ["runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(a.out, "gait120_alignment.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summ, f, indent=1, default=str)
    write_md(a.out, summ)
    print(json.dumps(summ["status"]), f"done in {summ['runtime_s']} s", flush=True)


def write_md(out, s):
    o = s["ok"]
    L = ["# gait120 EMG to pose alignment (plan task B4.5)", "",
         "Generated by scripts/probe_gait120_emg_alignment.py; every number below is read from gait120_alignment.json "
         "and gait120_alignment.csv. MoveAtlas commit: " + s["moveatlas_head"] + ". Rule, event and bands in the "
         "script docstring." + (f" LIMITED RUN ({s['limit']} trials)." if s["limit"] else ""), "",
         f"Trials read: {s['trials']} of the {s['trials_in_index']} non-lockbox trials of the index, {s['persons']} "
         f"persons. Status: {s['status']}.", "",
         f"Paired trials: EMG sample count equal to (last - first + 1) x 20: {o['emg_samples_equal_rule']}; pose frames "
         f"inside the EMG frames: {o['pose_inside_emg']}; contiguous steps: {o['contiguous']}; no frame twice in the "
         f"joined pose: {o['pose_frames_unique']}; steps per trial {o['steps']}; largest distance between the two "
         f"copies of a shared boundary frame: {o['boundary_diff_mm_max']} mm.", "",
         "S002 level_walking_trial01 (the example of the spec): " + (", ".join(
             f"{k} {v}" for k, v in s["s002_level_walking_trial01"].items()) if "s002_level_walking_trial01" in s
             else "not read in this run") + ".",
         "", "| task | trials | steps | kinematic lag ms (n, median, p10, p90) | TA peak, percent from the nearest step end | "
         "GAS medialis peak, percent | TA, rule + 100 ms | GAS, rule + 100 ms |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda d: "" if not d else f"{d['n']}, {d['median']}, {d['p10']}, {d['p90']}"
    for task, t in s["per_task"].items():
        L.append(f"| {task} | {t['trials']} | {t['steps']} | {fmt(t.get('kin_lag_ms'))} | {fmt(t.get('ta_phase'))} | "
                 f"{fmt(t.get('gas_phase'))} | {fmt(t.get('ta_phase_shifted'))} | {fmt(t.get('gas_phase_shifted'))} |")
    lw = s["per_task"].get("level_walking", {})
    if "in_band" in lw:
        b, b2 = lw["in_band"], lw["in_band_shifted"]
        L += ["", f"Level walking, bands fixed before the run: TA peak within {TA_BAND:g} percent of a step end on "
              f"{b['ta_within_15_of_a_step_end']} of {b['scored']} trials, GAS medialis peak in {GAS_BAND[0]:g} to "
              f"{GAS_BAND[1]:g} percent on {b['gas_in_25_60']}, both on {b['both']}. Under the rule shifted by 100 ms: "
              f"{b2['ta_within_15_of_a_step_end']}, {b2['gas_in_25_60']} and {b2['both']} of {b2['scored']}."]
    L += ["", f"Runtime: {s['runtime_s']} s.", ""]
    text = "\n".join(L)
    assert "\u2014" not in text
    with open(os.path.join(out, "gait120_alignment.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
