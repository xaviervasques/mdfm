"""The plan's short GPU check of the open body store (docs/MODEL_OPEN_PORT_PLAN.md, step M3, "then a GPU check"): the
store's bodies through the model's kinematics on the GPU.

Draws N_WINDOWS windows of WIN_S s by person (mdfm.data.open_body_store, numpy default_rng(SEED)), their rest joints
from the P3 basis (mdfm.body.openbody.OpenSkeletonBank) on the GPU in float32, and evaluates
mdfm.body.kinematics.KnotBatch.kinematics at every knot on the GPU in float32 (positions, velocities, accelerations
by the two nested jvp) against the CPU in float64 (KnotBatch on the basis' float64 rest), and the positions against
the forward kinematics of the stored rotations (mdfm.body.joints.fk, float64). Reports the largest differences and
the GPU time per call. Nothing is trained; the GPU queue is not used.

Writes reports/open_body_store/gpu_check.json.
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/open_body_gpu_check.py
"""
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import torch  # noqa: E402

N_WINDOWS = 64
WIN_S = 10.0
SEED = 20261113


def main():
    from mdfm.body import openbody as BO
    from mdfm.body.joints import fk
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.data.open_body_store import OpenBodyStore
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    dev = torch.device("cuda")
    st = OpenBodyStore(check=False)
    persons = st.persons()
    keys = sorted(persons)
    gen = np.random.default_rng(SEED)
    K = int(round(WIN_S * 100.0))
    r6s, trs, phs, lens = [], [], [], []
    for _ in range(N_WINDOWS):
        p = keys[int(gen.integers(len(keys)))]
        cand = persons[p]
        kn = np.array([st.takes[i]["n_knots"] for i in cand], dtype=np.float64)
        i = cand[int(gen.choice(len(cand), p=kn / kn.sum()))]
        s0 = int(gen.integers(max(st.takes[i]["n_knots"] - K, 0) + 1))
        tk = st.take(i)
        r6 = np.asarray(tk["rot6d"][s0:s0 + K], dtype=np.float64)
        tr = tk["transl"][s0:s0 + K]
        n = r6.shape[0]
        pad = K - n
        r6s.append(np.concatenate([r6, np.repeat(r6[-1:], pad, 0)]))
        trs.append(np.concatenate([tr, np.repeat(tr[-1:], pad, 0)]))
        phs.append(st.phenotype(i))
        lens.append(n)
    r6 = torch.as_tensor(np.stack(r6s))
    tr = torch.as_tensor(np.stack(trs))
    ph = torch.as_tensor(np.stack(phs))
    L = torch.as_tensor(lens, dtype=torch.int64)
    w = torch.cat([torch.full((n,), k, dtype=torch.int64) for k, n in enumerate(lens)])
    t = torch.cat([torch.arange(n, dtype=torch.float64) / 100.0 for n in lens])
    bank64 = BO.OpenSkeletonBank(torch.float64)
    bank32 = BO.OpenSkeletonBank(torch.float32).to(dev)
    rest64 = bank64.rest(ph)
    rest32 = bank32.rest(ph.float().to(dev))
    rest_diff = float((rest32.double().cpu() - rest64).abs().max())
    kb64 = KnotBatch(r6, tr, rest64, rate=100.0, lengths=L)
    kb32 = KnotBatch(r6.float().to(dev), tr.float().to(dev), rest32, rate=100.0, lengths=L.to(dev))
    with torch.no_grad():
        k64 = kb64.kinematics(w, t)
        torch.cuda.synchronize()
        c0 = time.time()
        k32 = kb32.kinematics(w.to(dev), t.float().to(dev))
        torch.cuda.synchronize()
        gpu_s = time.time() - c0
        c0 = time.time()
        k32 = kb32.kinematics(w.to(dev), t.float().to(dev))
        torch.cuda.synchronize()
        gpu_s2 = time.time() - c0
        # positions against the forward kinematics of the stored rotations
        R = rot6d_to_matrix(r6[w, (t * 100.0).round().long()])
        P, _ = fk(R, tr[w, (t * 100.0).round().long()], rest64[w])
    d = {k: float((getattr(k32, k).double().cpu() - getattr(k64, k)).abs().max()) for k in ("p", "v", "a")}
    rel_a = float(((k32.a.double().cpu() - k64.a).norm(dim=-1)).median() /
                  k64.a.norm(dim=-1).median().clamp_min(1e-9))
    fk_diff = float((k64.p - P).abs().max())
    head = subprocess.run(["git", "-C", TREE, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    out = {"script": "scripts/open_body_gpu_check.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_head": head, "device": torch.cuda.get_device_name(0), "windows": N_WINDOWS, "win_s": WIN_S,
           "knots": int(len(w)), "seed": SEED, "rest_float32_gpu_vs_float64_m": rest_diff,
           "max_abs_diff_gpu32_vs_cpu64": {"p_m": d["p"], "v_m_s": d["v"], "a_m_s2": d["a"]},
           "median_relative_acc_diff": rel_a, "positions_vs_stored_fk_m": fk_diff,
           "gpu_seconds_first_call": gpu_s, "gpu_seconds_second_call": gpu_s2}
    with open(os.path.join(TREE, "reports", "open_body_store", "gpu_check.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
