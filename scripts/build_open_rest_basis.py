"""The open body's joint-space basis of the rest skeleton (plan docs/MODEL_OPEN_PORT_PLAN.md, decision P3, step M2).

Evaluates mdfm.body.openbody.rest_joints22 (the adapter mdfm/synth/openbody.py, Anny behind its guard) at every node
of the knot grid openbody.KNOTS and writes the node values to reports/open_body_rest/rest_basis_values.json (metres to 0.1 micrometre). Then measures the
multilinear interpolation (openbody.RestBasis) against the adapter on two held-out sets: the centres of every grid
cell, and N_RANDOM phenotypes drawn uniformly in openbody.DOMAIN (numpy default_rng seeded by SEED). P3: the basis
serves when its largest joint error stays under openbody.BASIS_TOL_M (1 mm); otherwise the model uses Anny's own
forward. Output: rest_basis_values.json, rest_basis.json (knots, the values file's SHA-256, the errors per joint and set, the time per
adapter call, the joint map SHA-256, the licence record) and _provenance.json. No record is read; no network.

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_rest_basis.py
"""
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
from mdfm.body import openbody as BO  # noqa: E402
from mdfm.body.joints import BODY_JOINTS  # noqa: E402

SEED = 20261112
N_RANDOM = 200


def git_head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def errors(basis, P):
    t0 = time.time()
    truth = np.stack([BO.rest_joints22(p).numpy() for p in P])
    dt = (time.time() - t0) / len(P)
    with torch.no_grad():
        est = basis.rest(torch.as_tensor(P)).numpy()
    e = np.linalg.norm(est - truth, axis=-1)            # (n, 22) m
    return e, dt


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(BO.BASIS_DIR, exist_ok=True)
    BO.check_joint_table()
    t0 = time.time()
    values = BO.basis_values(progress=lambda i, n: print(f"node {i}/{n}", flush=True) if i % 24 == 0 else None)
    t_nodes = time.time() - t0
    BO.write_values(values)
    values = BO.read_values()
    basis = BO.RestBasis(BO.KNOTS, values, torch.float64)
    nodes = BO.grid_nodes()
    with torch.no_grad():
        node_err = float(np.abs(basis.rest(torch.as_tensor(nodes)).numpy() - values.reshape(len(nodes), 22, 3)).max())
    axes = [np.asarray(BO.KNOTS[k]) for k in BO.PHENOTYPE_KEYS]
    mids = [0.5 * (a[1:] + a[:-1]) for a in axes]
    centres = np.stack([m.reshape(-1) for m in np.meshgrid(*mids, indexing="ij")], -1)
    rng = np.random.default_rng(SEED)
    lo = np.array([BO.DOMAIN[k][0] for k in BO.PHENOTYPE_KEYS])
    hi = np.array([BO.DOMAIN[k][1] for k in BO.PHENOTYPE_KEYS])
    rand = lo + (hi - lo) * rng.random((N_RANDOM, BO.N_PHENO))
    e_c, _ = errors(basis, centres)
    e_r, dt = errors(basis, rand)
    e_all = np.concatenate([e_c, e_r])

    def summary(e):
        return {"n": int(e.shape[0]), "max_joint_error_m": float(e.max()), "mean_joint_error_m": float(e.mean()),
                "p99_joint_error_m": float(np.percentile(e, 99)),
                "max_by_joint_m": {j: float(x) for j, x in zip(BODY_JOINTS, e.max(0))}}
    rec = {"script": "scripts/build_open_rest_basis.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_head": git_head(), "knots": {k: list(map(float, v)) for k, v in BO.KNOTS.items()},
           "keys": list(BO.PHENOTYPE_KEYS), "domain": {k: list(v) for k, v in BO.DOMAIN.items()},
           "n_nodes": int(len(nodes)), "node_reproduction_max_abs_m": node_err,
           "values": "reports/open_body_rest/rest_basis_values.json", "values_sha256": BO.file_sha256(BO.BASIS_VALUES),
           "seconds_nodes": round(t_nodes, 1), "seconds_per_adapter_rest": round(dt, 4),
           "held_out": summary(e_all), "held_out_cell_centres": summary(e_c), "held_out_random": summary(e_r),
           "random_seed": SEED, "tolerance_m": BO.BASIS_TOL_M,
           "p3_decision": "basis" if float(e_all.max()) < BO.BASIS_TOL_M else "anny_forward",
           "joint_map_sha256": BO.joint_map_sha256(), "licence": BO.licence_record()}
    with open(BO.BASIS_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    prov = {"script": "scripts/build_open_rest_basis.py", "mdfm_head": rec["mdfm_head"], "run_at": rec["run_at"],
            "python": sys.version.split()[0], "torch": torch.__version__, "reads": "the Anny package through "
            "mdfm/synth/openbody.py (no record, no network)",
            "outputs": ["rest_basis_values.json", "rest_basis.json"]}
    with open(os.path.join(BO.BASIS_DIR, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("n_nodes", "node_reproduction_max_abs_m", "seconds_per_adapter_rest",
                                          "p3_decision")}, indent=1))
    print("held out max", rec["held_out"]["max_joint_error_m"], "mean", rec["held_out"]["mean_joint_error_m"])


if __name__ == "__main__":
    main()
