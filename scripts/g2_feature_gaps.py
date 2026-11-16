"""G2 diagnostic: which window features separate real from rendered walking, ranked by effect size.

The data of scripts/g2_renderer_realism.prepare() (its current version: Walk phases on the real
side, steady walking on the rendered side, lockbox persons excluded, rendered takes without the
residual spectrum, which version 4 did not adopt). For every feature of
g2_renderer_realism.features (per site and channel: mean, standard deviation, 10th, 50th and
90th percentiles, log power in 0.5 to 3, 3 to 8 and 8 to 20 Hz), the standardized mean
difference (real minus rendered, pooled standard deviation) over windows and the fraction of
windows a single threshold on that feature classifies correctly (best threshold, both
directions). A single feature near 1.0 points at a systematic error rather than a realism gap.

Needs the mdfm environment. Writes reports/g2_renderer/feature_gaps.csv and feature_gaps.md.
"""
import csv
import os
import sys

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)
import g2_diagnostics as dg  # noqa: E402
import g2_renderer_realism as g2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(TREE, "reports", "g2_renderer")
CH = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
STATS = ["mean", "sd", "p10", "p50", "p90", "pow_0.5_3", "pow_3_8", "pow_8_20"]


def names():
    out = []
    for s in g2.SITES13:
        out += [f"{s}:{c}:mean" for c in CH] + [f"{s}:{c}:sd" for c in CH]
        out += [f"{s}:{c}:{q}" for q in ("p10", "p50", "p90") for c in CH]
        out += [f"{s}:{c}:{b}" for b in ("pow_0.5_3", "pow_3_8", "pow_8_20") for c in CH]
    return out


def best_threshold_accuracy(a, b):
    v = np.concatenate([a, b])
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    o = np.argsort(v)
    ys = y[o]
    # balanced accuracy of "real if value > t" for every cut, and of the reverse
    tp = len(a) - np.cumsum(ys)          # real above the cut
    tn = np.cumsum(1 - ys)               # rendered at or below the cut
    ba = 0.5 * (tp / len(a) + tn / len(b))
    return float(max(ba.max(), (1 - ba).max()))


def main():
    p = g2.prepare()
    real_test = p["real_test"]
    matched = [p["rend"][i] for i in p["chosen"]]
    Xr = np.array([g2.features(w) for r in real_test for w in g2.windows(r["x"])])
    Xm = np.array([g2.features(w) for r in matched for w in g2.windows(r["x"])])
    nm = names()
    assert len(nm) == Xr.shape[1], (len(nm), Xr.shape)
    rows = []
    for j, n in enumerate(nm):
        a, b = Xr[:, j], Xm[:, j]
        sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
        rows.append({"feature": n, "real_mean": round(float(a.mean()), 4), "rendered_mean": round(float(b.mean()), 4),
                     "smd": round(float((a.mean() - b.mean()) / sd), 3),
                     "single_threshold_bacc": round(best_threshold_accuracy(a, b), 4)})
    rows.sort(key=lambda r: -r["single_threshold_bacc"])
    with open(os.path.join(OUT, "feature_gaps.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    md = ["# G2 feature gaps, real against rendered", "", "Measured by scripts/g2_feature_gaps.py.", "",
          f"Windows: real {len(Xr)}, rendered {len(Xm)}. Top 40 features by single-threshold balanced accuracy.", "",
          "| feature | real mean | rendered mean | standardized difference | single-threshold balanced accuracy |",
          "|---|---|---|---|---|"]
    md += [f"| {r['feature']} | {r['real_mean']} | {r['rendered_mean']} | {r['smd']} | {r['single_threshold_bacc']} |" for r in rows[:40]]
    by_stat = {}
    for r in rows:
        by_stat.setdefault(r["feature"].split(":")[2], []).append(r["single_threshold_bacc"])
    md += ["", "| statistic | features | median single-threshold balanced accuracy | max |", "|---|---|---|---|"]
    md += [f"| {k} | {len(v)} | {round(float(np.median(v)), 4)} | {max(v)} |" for k, v in sorted(by_stat.items())]
    text = "\n".join(md) + "\n"
    assert chr(0x2014) not in text
    open(os.path.join(OUT, "feature_gaps.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
