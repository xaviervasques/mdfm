"""How much rest height one unit of the shape vector moves, on the open body and on v1 (plan docs/MODEL_OPEN_PORT_PLAN.md,
step M4; read with the learnability gate of reports/bench_open/learnability.md, where the open model's mean absolute
shape error of 0.375 left a rest height error of 4.2 cm and v1's 0.345 left 0.8 cm).

Rest height is the measure of train/loop.health: the largest minus the smallest y of the 22 rest joints. Open body: the
rest skeleton of the P3 basis on the phenotype latent u (mdfm/body/openbody.LatentSkeletonBank), its gradient with
respect to u at the pooled phenotype prior's mean and at the 397 store persons' own latents (the train view of
mdfm/data/open_body_store.OpenTrainStore, one row per person: the mean of its takes' latents), and the height change of
one prior standard deviation per dimension. v1 (with --v1-reference, through mdfm/body/backend.py behind the reference
flag, comparison only, no array written): the same on SMPL+H's SkeletonBank at the neutral beta prior's mean, gender
neutral.

Writes reports/bench_open/shape_sensitivity.json and .md. CPU, about a minute.
"""
import datetime
import json
import os
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm.body import openbody as BO  # noqa: E402
from mdfm.model import mfm as M  # noqa: E402

OUT = os.path.join(TREE, "reports", "bench_open")


def head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git rev-parse failed: {r.stderr.strip()}")
    return r.stdout.strip()


def height(r):
    return r[..., 1].amax(-1) - r[..., 1].amin(-1)


def jacobian(rest_fn, x):
    x = x.clone().double().requires_grad_(True)
    h = height(rest_fn(x[None]))[0]
    (g,) = torch.autograd.grad(h, x)
    return float(h), g.numpy()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mdfm_head = head()
    bank = BO.LatentSkeletonBank(torch.float64)
    prior = M.open_shape_prior()
    mean, sd = prior.mean[2].double(), torch.sqrt(torch.diagonal(prior.cov[2])).double()
    h0, g0 = jacobian(lambda u: bank.rest(u), mean)
    from mdfm.data.open_body_store import OpenTrainStore
    st = OpenTrainStore(check=False)
    persons = st.takes.groupby("person_id").indices
    U = np.stack([st.betas[ix].mean(0) for ix in persons.values()])
    G = np.stack([jacobian(lambda u: bank.rest(u), torch.as_tensor(u))[1] for u in U])
    out = {"script": "scripts/open_shape_sensitivity.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_head": mdfm_head,
           "open": {"keys": list(BO.PHENOTYPE_KEYS), "height_at_prior_mean_m": h0,
                    "dheight_du_at_prior_mean_m": g0.tolist(), "prior_sd_u": sd.numpy().tolist(),
                    "height_per_prior_sd_m": (np.abs(g0) * sd.numpy()).tolist(),
                    "persons": int(len(U)), "abs_dheight_du_median_over_persons_m": np.median(np.abs(G), 0).tolist(),
                    "gradient_norm_median_m": float(np.median(np.linalg.norm(G, axis=1)))}}
    if "--v1-reference" in sys.argv:
        from mdfm.body import backend as BK
        from mdfm.model.prior import ShapePrior
        sk = BK.skeleton_bank(BK.SMPLH, allow_reference=True, dtype=torch.float64)
        sp = ShapePrior()
        bm, bsd = sp.mean[2].double(), torch.sqrt(torch.diagonal(sp.cov[2])).double()
        g = torch.tensor([2])
        h1, g1 = jacobian(lambda b: sk.rest(b, g), bm)
        out["v1"] = {"height_at_prior_mean_m": h1, "dheight_dbeta_at_prior_mean_m": g1.tolist(),
                     "prior_sd_beta": bsd.numpy().tolist(), "height_per_prior_sd_m": (np.abs(g1) * bsd.numpy()).tolist(),
                     "gradient_norm_m": float(np.linalg.norm(g1))}
    out["open"]["gradient_norm_at_prior_mean_m"] = float(np.linalg.norm(g0))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "shape_sensitivity.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    o = out["open"]
    L = ["# Rest height against the shape vector, open body and v1", "",
         f"Written by scripts/open_shape_sensitivity.py on {out['run_at']} at MD_FM {out['mdfm_head']}. Rest height: the "
         "largest minus the smallest y of the 22 rest joints (train/loop.health).", "",
         "| open phenotype latent | dh/du at the prior mean (cm per unit) | median over the persons | prior sd of u | "
         "cm per prior sd |", "|---|---|---|---|---|"]
    for i, k in enumerate(o["keys"]):
        L.append(f"| {k} | {100 * o['dheight_du_at_prior_mean_m'][i]:.2f} | "
                 f"{100 * o['abs_dheight_du_median_over_persons_m'][i]:.2f} | {o['prior_sd_u'][i]:.3f} | "
                 f"{100 * o['height_per_prior_sd_m'][i]:.2f} |")
    L += ["", f"Open: gradient norm {100 * o['gradient_norm_at_prior_mean_m']:.2f} cm per unit of u at the prior mean "
          f"(median over the {o['persons']} persons {100 * o['gradient_norm_median_m']:.2f}); rest height there "
          f"{o['height_at_prior_mean_m']:.3f} m."]
    if "v1" in out:
        v = out["v1"]
        L += [f"v1 (SMPL+H, neutral): gradient norm {100 * v['gradient_norm_m']:.2f} cm per unit of beta at the prior "
              f"mean; the largest change per prior sd {100 * max(v['height_per_prior_sd_m']):.2f} cm (beta "
              f"{int(np.argmax(v['height_per_prior_sd_m']))}); rest height {v['height_at_prior_mean_m']:.3f} m."]
    L += ["", "Every number above comes from reports/bench_open/shape_sensitivity.json.", ""]
    with open(os.path.join(OUT, "shape_sensitivity.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
