"""G2 with a weaker, fixed discriminator: L2-regularized logistic regression on standardized
window features, same data as
scripts/g2_renderer_realism.py (its prepare(), with the residual spectrum). Informs the choice of the pre-registered G2 discriminator:
if a linear probe also separates rendered from real walking, the gap is large; if only
gradient boosting does, the gap is subtle.

Needs the mdfm environment. Writes reports/g2_renderer/linear_probe.json and .md.
"""
import json
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


def linear(real, rend, featfn, folds=5, seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, grp = [], [], []
    for item, label in [(r, 1) for r in real] + [(r, 0) for r in rend]:
        for w in g2.windows(item["x"]):
            X.append(featfn(w))
            y.append(label)
            grp.append(item["person"])
    X, y, grp = np.asarray(X, np.float64), np.asarray(y), np.asarray(grp)
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, grp):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    bacc = 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())
    return {"balanced_accuracy": round(float(bacc), 4), "windows_real": int((y == 1).sum()), "windows_rendered": int((y == 0).sum())}


def main():
    p = g2.prepare()
    real_test = p["real_test"]
    matched = [p["rend"][i] for i in p["chosen"]]
    res = {"logistic_all_features": linear(real_test, matched, dg.feat_all),
           "logistic_all_features_with_residual_not_adopted": linear(real_test, [p["rend_res"][i] for i in p["chosen"]], dg.feat_all)}
    print(res, flush=True)
    for name, fn in (("logistic_orientation", dg.feat_orientation), ("logistic_dynamics", dg.feat_dynamics),
                     ("logistic_spectrum", dg.feat_spectrum)):
        res[name] = linear(real_test, matched, fn)
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(OUT, "linear_probe.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    md = ["# G2 with a linear discriminator", "", "Measured by scripts/g2_linear_probe.py.", "",
          "| discriminator and features | balanced accuracy |", "|---|---|"]
    md += [f"| {k} | {v['balanced_accuracy']} |" for k, v in res.items()]
    open(os.path.join(OUT, "linear_probe.md"), "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
