"""The contact rule read back on the open body store against force plates (plan docs/MODEL_OPEN_PORT_PLAN.md, section
3.1 Feet, step M4).

The store's contact labels follow the v1 rule (mdfm.body.frames.contact_labels: the ankle or the foot joint within
H0 = 0.05 m of the floor and slower than V0 = 0.3 m/s), but the open body's foot joint is Anny's third-toe MTP head,
another landmark than SMPL+H's. The truth here: the force plates of stroke_gait_vancriekinge able-bodied (the
canonical force records of the same C3D trials, per-foot total vertical force at 1000 Hz, read through
mdfm.data.canonical.load; train-fold persons of the store only). A plate sees a foot only while that foot is on it,
so the read is per plate stance: a stance is a run of samples whose force exceeds FORCE_N; recall is the share of
the stance's knots the rule labels contact; specificity the share of the knots of the swing next to it (from
SWING_FAR_S to SWING_NEAR_S before the stance's onset and after its offset) the rule labels free. Balanced accuracy
is their mean. The onset and offset delays (rule minus plate, s) are read on each stance.

The rule is read again on a grid of H0 and V0 from the stored bodies (forward kinematics of the stored rotations on
the person's rest through the P3 basis). Stated rule: the thresholds are kept unless a grid point raises the
balanced accuracy by at least KEEP_MARGIN; then the best point is reported for M4 to adopt, unless it lies on the
grid's edge (the truth then does not bound the thresholds, and the decision is left open).

Writes reports/open_body_store/contact_check.json and contact_check.md.
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/open_body_contact_check.py
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
import torch  # noqa: E402

OUT = os.path.join(TREE, "reports", "open_body_store")
SOURCE = "stroke_gait_vancriekinge"
FORCE_N = 20.0              # [design] a plate stance: vertical force above 20 N
SWING_NEAR_S, SWING_FAR_S = 0.05, 0.25
MIN_STANCE_S = 0.2
H0_GRID = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15)
V0_GRID = (0.2, 0.3, 0.5, 0.7, 0.9, 1.2)
KEEP_MARGIN = 0.02


def runs(mask):
    """[(start, stop)) of the True runs of a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm.body import frames
    from mdfm.body.joints import fk
    from mdfm.body.rotations import rot6d_to_matrix
    from mdfm.data import canonical
    from mdfm.data.open_body_store import OpenBodyStore
    st = OpenBodyStore()
    takes = [i for i, t in enumerate(st.takes) if t["source"] == SOURCE]
    idx = canonical.index()
    force_rows = {(r["subject_ns"], r["key"]): r for r in
                  idx[(idx["source"] == SOURCE) & (idx["modality"] == "force")].to_dict("records")}
    data = []
    for i in takes:
        t = st.takes[i]
        r = force_rows.get((t["subject_ns"], t["key"]))
        if r is None:
            continue
        rec = canonical.load(r)
        if float(rec["fps"]) != 1000.0:
            continue
        F = np.asarray(rec["force"], dtype=np.float64)[:, :, 8]           # per-foot total
        tk = st.take(i)
        T = tk["rot6d"].shape[0]
        n = min(T, F.shape[0] // 10)
        Fk = F[:n * 10].reshape(n, 10, 2).mean(1)
        R = rot6d_to_matrix(torch.from_numpy(np.asarray(tk["rot6d"][:n], dtype=np.float64)))
        P, _ = fk(R, torch.from_numpy(tk["transl"][:n]), torch.from_numpy(st.rest(i)))
        data.append({"take": t["take_id"], "person": t["person_id"], "J": P, "F": Fk, "stored": tk["contact"][:n]})

    def evaluate(labels_of):
        rec_n = rec_d = sp_n = sp_d = 0
        on, off = [], []
        n_st = 0
        for d in data:
            lab = labels_of(d)
            for foot in (0, 1):
                f = d["F"][:, foot] > FORCE_N
                for a, b in runs(f):
                    if (b - a) / 100.0 < MIN_STANCE_S:
                        continue
                    n_st += 1
                    rec_n += int(lab[a:b, foot].sum())
                    rec_d += b - a
                    near, far = int(SWING_NEAR_S * 100), int(SWING_FAR_S * 100)
                    for lo, hi in ((a - far, a - near), (b + near, b + far)):
                        lo, hi = max(lo, 0), min(hi, len(lab))
                        if hi > lo and not f[lo:hi].any():
                            sp_n += int((~lab[lo:hi, foot].astype(bool)).sum())
                            sp_d += hi - lo
                    c = lab[:, foot].astype(bool)
                    rr = [x for x in runs(c) if x[0] < b and x[1] > a]
                    if rr:
                        on.append((rr[0][0] - a) / 100.0)
                        off.append((rr[-1][1] - b) / 100.0)
        recall = rec_n / max(rec_d, 1)
        spec = sp_n / max(sp_d, 1)
        return {"stances": n_st, "recall": recall, "specificity": spec, "balanced": 0.5 * (recall + spec),
                "onset_delay_s_median": float(np.median(on)) if on else None,
                "offset_delay_s_median": float(np.median(off)) if off else None}

    stored = evaluate(lambda d: d["stored"])
    grid = []
    for h0 in H0_GRID:
        for v0 in V0_GRID:
            res = evaluate(lambda d: frames.contact_labels(d["J"][None], 100.0, h0=h0, v0=v0, floor=0.0)[0].numpy())
            grid.append({"h0_m": h0, "v0_m_s": v0, **res})
    cur = next(g for g in grid if g["h0_m"] == frames.H0 and g["v0_m_s"] == frames.V0)
    best = max(grid, key=lambda g: g["balanced"])
    edge = best["h0_m"] in (H0_GRID[0], H0_GRID[-1]) or best["v0_m_s"] in (V0_GRID[0], V0_GRID[-1])
    if best["balanced"] < cur["balanced"] + KEEP_MARGIN:
        decision = "keep"
    elif edge:
        decision = "undecided: the best point lies on the grid's edge, so this truth does not bound the thresholds"
    else:
        decision = "move"
    head = subprocess.run(["git", "-C", TREE, "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    out = {"script": "scripts/open_body_contact_check.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "mdfm_head": head, "source": SOURCE, "takes": len(data), "persons": len({d["person"] for d in data}),
           "force_n": FORCE_N, "swing_window_s": [SWING_NEAR_S, SWING_FAR_S], "min_stance_s": MIN_STANCE_S,
           "stored_labels": stored, "current_rule": cur, "best": best, "keep_margin": KEEP_MARGIN,
           "decision": decision, "grid": grid}
    with open(os.path.join(OUT, "contact_check.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    L = ["# The contact rule read back on the open body store (step M4)", "",
         f"scripts/open_body_contact_check.py, MD_FM {head}: {len(data)} takes of {out['persons']} persons of "
         f"{SOURCE} able-bodied, the plates' per-foot vertical force above {FORCE_N:.0f} N as the truth "
         f"({stored['stances']} plate stances of at least {MIN_STANCE_S} s); the swing read from {SWING_NEAR_S} to "
         f"{SWING_FAR_S} s on each side of a stance.", "",
         "| labels | recall in stance | specificity in the swing next to it | balanced | onset delay s | offset delay s |",
         "|---|---|---|---|---|---|"]
    for name, g in (("stored (H0 0.05 m, V0 0.3 m/s)", stored), (f"best of the grid (H0 {best['h0_m']} m, V0 "
                                                                  f"{best['v0_m_s']} m/s)", best)):
        L.append(f"| {name} | {g['recall']:.3f} | {g['specificity']:.3f} | {g['balanced']:.3f} | "
                 f"{g['onset_delay_s_median']} | {g['offset_delay_s_median']} |")
    L += ["", f"Decision by the stated rule (keep unless a grid point gains {KEEP_MARGIN} of balanced accuracy): "
          f"{decision}.", ""]
    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "contact_check.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
