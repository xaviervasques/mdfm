"""G2 sanity comparison: per site, simple statistics of real weargait_pd control walking and of
rendered AMASS walking (the data of scripts/g2_renderer_realism.prepare(), rendered takes without the residual spectrum), to
find systematic errors (units, axes, scale) before any realism work.

Per site: median norm of the specific force (g), its standard deviation within trials, the
root mean square of the angular rate (deg/s), the 99th percentile of the angular rate, and
the mean specific-force direction (after the population mounting fit) with its angle between
real and rendered. Writes reports/g2_renderer/compare_stats.csv and compare_stats.md.
"""
import csv
import os
import sys

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)
import g2_renderer_realism as g2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(TREE, "reports", "g2_renderer")


def stats(xs, s):
    acc = np.concatenate([x[:, s, 0:3] for x in xs])
    gyr = np.concatenate([x[:, s, 3:6] for x in xs])
    ok = np.isfinite(acc).all(1) & np.isfinite(gyr).all(1)
    acc, gyr = acc[ok], gyr[ok]
    an = np.linalg.norm(acc, axis=1)
    gn = np.linalg.norm(gyr, axis=1)
    within = np.nanmedian([np.nanstd(np.linalg.norm(x[:, s, 0:3], axis=1)) for x in xs])
    m = acc.mean(0)
    return {"acc_norm_median_g": round(float(np.median(an)), 3), "acc_norm_sd_within_g": round(float(within), 3),
            "gyro_rms_dps": round(float(np.sqrt((gn ** 2).mean())), 1), "gyro_p99_dps": round(float(np.percentile(gn, 99)), 1),
            "mean_dir": (m / np.linalg.norm(m)).round(3).tolist(), "axis_sd": acc.std(0).round(3).tolist()}


def main():
    p = g2.prepare()
    real = p["real_align"] + p["real_test"]
    rend, resid = p["rend"], p["resid"]
    rows = []
    for s, name in enumerate(g2.SITES13):
        a = stats([r["x"] for r in real], s)
        b = stats([r["x"] for r in rend], s)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(a["mean_dir"], b["mean_dir"]), -1, 1))))
        rows.append({"site": name, **{f"real_{k}": v for k, v in a.items()}, **{f"rend_{k}": v for k, v in b.items()},
                     "mean_dir_angle_deg": round(ang, 1), "mounting_residual_deg": resid[name]})
        print(name, a, b, round(ang, 1), flush=True)
    with open(os.path.join(OUT, "compare_stats.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    md = ["# G2 sanity comparison, real against rendered", "", "Measured by scripts/g2_compare_stats.py.", "",
          "| site | acc norm median g, real / rendered | within-trial SD of acc norm, real / rendered | gyro RMS deg/s, real / rendered | mean direction angle deg |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['site']} | {r['real_acc_norm_median_g']} / {r['rend_acc_norm_median_g']} | "
                  f"{r['real_acc_norm_sd_within_g']} / {r['rend_acc_norm_sd_within_g']} | "
                  f"{r['real_gyro_rms_dps']} / {r['rend_gyro_rms_dps']} | {r['mean_dir_angle_deg']} |")
    text = "\n".join(md) + "\n"
    open(os.path.join(OUT, "compare_stats.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
