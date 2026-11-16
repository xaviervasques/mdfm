"""Device noise of the weargait_pd Xsens sensors, measured on quiet standing, for the G2 renderer.

Data: the ALIGN half of the non-lockbox weargait_pd controls of scripts/g2_renderer_realism.py
(same person split; the TEST half is never read here), samples labelled Standing in the raw
GeneralEvent column, runs of at least 4 s. Per site and channel: the Welch power spectral density
(256-sample segments, linear detrend) averaged over runs; the white-noise level is the median
density over 30 to 45 Hz, where standing body motion carries little power, converted to a
per-sample standard deviation sigma = sqrt(density * fs / 2). Gyroscope bias spread: the standard
deviation, across runs, of the mean angular rate of a run, per site and axis (a still sensor's mean
rate is its bias plus sway, which averages near zero over 4 s).

The renderer takes one value per quantity: the median over the 13 sites and three axes.
Writes reports/g2_renderer/device_noise.json and device_noise.md. Needs the mdfm environment.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import welch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)
import g2_renderer_realism as g2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(TREE, "reports", "g2_renderer")
MIN_RUN = 400


def runs(mask):
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= MIN_RUN:
                out.append((start, i))
            start = None
    return out


def main():
    rng = np.random.default_rng(20260928)
    walk = g2.real_records()
    persons = sorted({r["person"] for r in walk})
    perm = rng.permutation(len(persons))
    align_p = {persons[i] for i in perm[: len(persons) // 2]}
    stand = [r for r in g2.real_records(phase="Standing") if r["person"] in align_p]
    psd = [[[] for _ in range(6)] for _ in range(13)]
    means = [[[] for _ in range(3)] for _ in range(13)]
    n_runs = 0
    for r in stand:
        x = g2.to_sites13(r["signal"], r["loc"])
        for s in range(13):
            ok = np.isfinite(x[:, s, :]).all(1)
            for a, b in runs(ok):
                seg = x[a:b, s, :].astype(np.float64)
                n_runs += 1
                for c in range(6):
                    f, p = welch(seg[:, c], fs=g2.RATE, nperseg=256, detrend="linear")
                    psd[s][c].append(p)
                for c in range(3):
                    means[s][c].append(seg[:, 3 + c].mean())
    band = (f >= 30) & (f <= 45)
    per_site = []
    for s, name in enumerate(g2.SITES13):
        sig = [float(np.sqrt(np.median(np.mean(psd[s][c], 0)[band]) * g2.RATE / 2)) for c in range(6)]
        bias = [float(np.std(means[s][c])) for c in range(3)]
        per_site.append({"site": name, "runs": len(means[s][0]), "acc_sigma_g": [round(v, 5) for v in sig[:3]],
                         "gyro_sigma_dps": [round(v, 4) for v in sig[3:]], "gyro_bias_sd_dps": [round(v, 4) for v in bias]})
    acc = float(np.median([v for p in per_site for v in p["acc_sigma_g"]]))
    gyr = float(np.median([v for p in per_site for v in p["gyro_sigma_dps"]]))
    gb = float(np.median([v for p in per_site for v in p["gyro_bias_sd_dps"]]))
    res = {"acc_noise_g": round(acc, 5), "gyro_noise_dps": round(gyr, 4), "gyro_bias_dps": round(gb, 4),
           "align_persons": len(align_p), "standing_trials": len(stand), "runs_site_level": n_runs,
           "band_hz": [30, 45], "per_site": per_site}
    json.dump(res, open(os.path.join(OUT, "device_noise.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    md = ["# Device noise of the weargait_pd sensors (quiet standing, ALIGN half)", "",
          "Measured by scripts/g2_device_noise.py (method in its docstring).", "",
          f"Renderer values (median over sites and axes): acc white noise {res['acc_noise_g']} g per sample, "
          f"gyro white noise {res['gyro_noise_dps']} deg/s per sample, gyro bias spread {res['gyro_bias_dps']} deg/s. "
          f"The factory defaults were 0.005 g, 0.3 deg/s and 1.0 deg/s.", "",
          f"ALIGN persons {len(align_p)}, standing trials {len(stand)}, site-level runs of at least 4 s: {n_runs}.", "",
          "| site | runs | acc sigma g (x, y, z) | gyro sigma deg/s (x, y, z) | gyro bias SD deg/s (x, y, z) |", "|---|---|---|---|---|"]
    md += [f"| {p['site']} | {p['runs']} | {p['acc_sigma_g']} | {p['gyro_sigma_dps']} | {p['gyro_bias_sd_dps']} |" for p in per_site]
    text = "\n".join(md) + "\n"
    assert chr(0x2014) not in text
    open(os.path.join(OUT, "device_noise.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
