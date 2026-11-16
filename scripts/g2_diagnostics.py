"""G2 diagnostics: what lets the discriminator of scripts/g2_renderer_realism.py tell rendered
walking from real walking? Same data, same split, same cadence matching; one change at a time.

Variants (all sites, balanced accuracy with a person-bootstrap interval):
- baseline (all features);
- feature groups alone: orientation (per-channel means), dynamics (standard deviation and
  centred 10th, 50th, 90th percentiles), spectrum (log power in 0.5 to 3, 3 to 8, 8 to 20 Hz);
- rendered source alone: KIT (overground) or BMLrub (treadmill), each cadence-matched;
- per-take mounting jitter: each rendered take gets, per site, a random rotation of about 7
  degrees (the spread of the real mountings measured in reports/weargait_frames/) on top of
  the fitted mounting.
A variant that drops the accuracy far below baseline points at the realism gap to fix.

Needs the mdfm environment. Writes reports/g2_renderer/diagnostics.json and diagnostics.md.
"""
import json
import os
import sys

import numpy as np
import torch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)
import g2_renderer_realism as g2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(TREE, "reports", "g2_renderer")
F = np.fft.rfftfreq(g2.WIN, 1 / g2.RATE)


def feat_orientation(w):
    return list(w.mean(0).ravel())


def feat_dynamics(w):
    c = w - w.mean(0)
    return list(c.std(0).ravel()) + list(np.percentile(c, [10, 50, 90], axis=0).ravel())


def feat_spectrum(w):
    c = w - w.mean(0)
    p = np.abs(np.fft.rfft(c, axis=0)) ** 2
    return [float(v) for lo, hi in g2.BANDS for v in np.log(p[(F >= lo) & (F < hi)].sum(0) + 1e-9).ravel()]


def feat_all(w):
    return g2.features(w)


def discriminate(real, rend, featfn, folds=5, seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X, y, grp = [], [], []
    for item, label in [(r, 1) for r in real] + [(r, 0) for r in rend]:
        for w in g2.windows(item["x"]):
            X.append(featfn(w))
            y.append(label)
            grp.append(item["person"])
    X, y, grp = np.asarray(X, np.float32), np.asarray(y), np.asarray(grp)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, grp):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    rng = np.random.default_rng(seed)

    def bacc(idx):
        yy, pp = y[idx], pred[idx]
        return 0.5 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean())

    persons = np.unique(grp)
    by = {p: np.where(grp == p)[0] for p in persons}
    boot = []
    for _ in range(500):
        idx = np.concatenate([by[p] for p in rng.choice(persons, len(persons), replace=True)])
        if len(np.unique(y[idx])) == 2:
            boot.append(bacc(idx))
    return {"balanced_accuracy": round(float(bacc(np.arange(len(y)))), 4),
            "ci95": [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])],
            "windows_real": int((y == 1).sum()), "windows_rendered": int((y == 0).sum())}


def cadence_match(real_test, rend, rng):
    bins = np.arange(60, 160, 5)
    hr, _ = np.histogram([r["cadence"] for r in real_test], bins)
    which = np.digitize([r["cadence"] for r in rend], bins) - 1
    chosen = []
    for b, n_real in enumerate(hr):
        pool = [i for i in range(len(rend)) if which[i] == b]
        k = int(round(n_real / max(hr.sum(), 1) * len(rend)))
        if pool and k:
            chosen += list(rng.choice(pool, k, replace=len(pool) < k))
    return [rend[i] for i in chosen]


def random_rotation(angle_sd_deg, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.normal(0, angle_sd_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def main():
    rng = np.random.default_rng(20260928)
    real = g2.real_records()
    for r in real:
        r["x"] = g2.to_sites13(r["signal"], r["loc"])
        r["cadence"] = g2.cadence_from_lower_back(r["x"])
    persons = sorted({r["person"] for r in real})
    perm = rng.permutation(len(persons))
    align_p = {persons[i] for i in perm[: len(persons) // 2]}
    real_align = [r for r in real if r["person"] in align_p]
    real_test = [r for r in real if r["person"] not in align_p]
    rend0 = g2.rendered_takes(mount=None)
    mount, _ = g2.fit_mounting([r["x"] for r in real_align], [r["x"] for r in rend0])
    rend = g2.rendered_takes(mount=mount)
    matched = cadence_match(real_test, rend, rng)
    res = {"baseline": discriminate(real_test, matched, feat_all)}
    print("baseline", res["baseline"], flush=True)
    for name, fn in (("orientation_only", feat_orientation), ("dynamics_only", feat_dynamics),
                     ("spectrum_only", feat_spectrum)):
        res[name] = discriminate(real_test, matched, fn)
        print(name, res[name], flush=True)
    for subset in ("KIT", "BMLrub"):
        sub = cadence_match(real_test, [r for r in rend if f":{subset}:" in r["person"]], rng)
        res[f"rendered_{subset}_only"] = discriminate(real_test, sub, feat_all)
        print(subset, res[f"rendered_{subset}_only"], flush=True)
    # per-take mounting jitter: re-render each take with its own perturbed mounting
    import csv
    rows = list(csv.DictReader(open(os.path.join(TREE, "reports", "amass_walks", "takes.csv"), encoding="utf-8")))
    jit = []
    if True:
        # render each take with its own perturbed mounting
        from mdfm.body.smplh import SMPLHBody
        from mdfm.body.trajectory import BodyTrajectory
        from mdfm.data.amass import load_sequence
        from mdfm.data.prep import amass_to_body_clock
        from mdfm.factory.imu_render import SITES, Device, render
        import zlib
        bodies = {}
        for r in rows:
            seq = load_sequence(r["subset"], r["take"])
            body = bodies.setdefault(r["gender"], SMPLHBody(r["gender"], dtype=torch.float64))
            r6, tr, b = amass_to_body_clock(seq, rate=g2.RATE, cutoff=12.0, body=body)
            traj = BodyTrajectory(body, r6, tr, b, rate=g2.RATE)
            t = torch.arange(0.5, (r6.shape[0] - 1) / g2.RATE - 0.5, 1 / g2.RATE, dtype=torch.float64)
            if t.numel() < g2.WIN:
                continue
            m = {s: torch.as_tensor(random_rotation(7.0, rng)) @ mount[s] for s in g2.SITES13}
            gen = torch.Generator().manual_seed(zlib.crc32(r["take"].encode()))
            with torch.no_grad():
                sig, _, _ = render(traj, t, sites=g2.SITES13, mount=m, device=Device(), generator=gen)
            x = sig[:, [SITES.index(s) for s in g2.SITES13], :6].numpy()
            parts = r["take"].split("/")
            jit.append({"person": f"amass:{r['subset']}:{parts[1] if len(parts) > 2 else parts[0]}",
                        "take": r["take"], "x": x, "cadence": g2.cadence_from_lower_back(x)})
    res["mounting_jitter_7deg"] = discriminate(real_test, cadence_match(real_test, jit, rng), feat_all)
    print("jitter", res["mounting_jitter_7deg"], flush=True)
    json.dump(res, open(os.path.join(OUT, "diagnostics.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    md = ["# G2 diagnostics", "", "Measured by scripts/g2_diagnostics.py (variants in its docstring).", "",
          "| variant | balanced accuracy | 95 percent interval | real windows | rendered windows |", "|---|---|---|---|---|"]
    md += [f"| {k} | {v['balanced_accuracy']} | {v['ci95']} | {v['windows_real']} | {v['windows_rendered']} |" for k, v in res.items()]
    text = "\n".join(md) + "\n"
    assert chr(0x2014) not in text
    open(os.path.join(OUT, "diagnostics.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
