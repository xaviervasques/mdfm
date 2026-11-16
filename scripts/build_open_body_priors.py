"""The open body's priors from the open body store (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4): the phenotype prior,
the mean pose, the window priors and, for information, the joint-limit percentiles of the store.

Reads the store (mdfm.data.open_body_store, hash-checked; train folds only by construction) and, for the persons'
sex and stature, the sources' own tables: gait120 (mdfm.data.sources.demographic: sex), stroke_gait_vancriekinge
able-bodied (mdfm.data.mocap_raw.mat_char: Male, Height mm), ceti_age_kinematics (participants.tsv of the raw folder:
sex, a01 the body height in cm). Every other person counts as sex unknown.

- Phenotype prior (mdfm.model.prior.PhenotypePrior.fit): one latent vector per person (openbody.to_latent of the
  person's fitted phenotype), rows female, male (the persons of known sex) and unknown (all persons pooled).
- The gender value against sex: per sex the persons' fitted gender values (Anny's male 0, female 1), and the
  area under the curve of the gender value separating women from men; the fitted body's stature (the rest mesh's
  height, synth.openbody.anthropometry) against the measured stature where a source gives it.
- Mean pose: the weighted mean 6D value per joint over every STRIDE-th knot, each knot weighted by person
  (OpenBodyStore.knot_weights: a person's knots sum to one person's weight), the root in each knot's own heading
  frame (yaw removed), as the v1 store's priors.json holds it.
- Window priors: N_WINDOWS windows of WIN_S s drawn by person (a person uniformly, one of its takes in proportion
  to its knots, a start uniformly), from numpy default_rng(SEED), in the window frame of mdfm.body.frames (floor at
  y = 0; at the centre knot the pelvis heading +z and the pelvis x and z at 0): the mean and standard deviation of
  the root translation over the windows' knots, and at the centre knot, as reports/body_store/window_priors.json.
- Joint-limit percentiles: the 0.1 and 99.9 percentiles of each axis-angle component over the same knots (for
  information; the model's limits stay the anatomical table of reports/openbody/joint_limits.json).

Writes reports/open_body_priors/priors.json, window_priors.json, summary.md and _provenance.json.
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_body_priors.py
"""
import csv
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

OUT = os.path.join(TREE, "reports", "open_body_priors")
STRIDE = 10
N_WINDOWS = 20000
WIN_S = 10.0
SEED = 20261113
CETI_PARTICIPANTS = "datasets/pose/ceti_age_kinematics/raw/ceti-age-kinematics/ceti-age-kinematics/participants.tsv"


def git_head(root=TREE, extra=()):
    r = subprocess.run(["git", *extra, "-C", root, "log", "-1", "--format=%h"], capture_output=True, text=True)
    return r.stdout.strip()


def sex_and_stature(source, subject_ns):
    """(sex 'F', 'M' or None, stature m or None) from the source's own table."""
    from mdfm import paths
    try:
        if source == "gait120":
            from mdfm.data import sources as S
            d = S.demographic("gait120", subject_ns)
            return (d.get("sex") or None), None
        if source == "stroke_gait_vancriekinge":
            from mdfm.data import mocap_raw as MR
            c = MR.mat_char(subject_ns)
            sex = {1.0: "M", 0.0: "F"}.get(float(c.get("Male", np.nan)))
            h = c.get("Height")
            return sex, (float(h) / 1000.0 if h is not None and np.isfinite(h) and h > 0 else None)
        if source == "ceti_age_kinematics":
            rows = _ceti(paths.MOVEATLAS)
            r = rows.get(subject_ns.split(":", 1)[1])
            if r is None:
                return None, None
            sex = {"m": "M", "f": "F"}.get(r["sex"].strip().lower())
            try:
                h = float(r["a01"]) / 100.0
            except ValueError:
                h = None
            return sex, h
    except (KeyError, LookupError, ValueError):
        return None, None
    return None, None


_CETI = {}


def _ceti(ma):
    if not _CETI:
        with open(os.path.join(ma, CETI_PARTICIPANTS), encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                _CETI[r["participant_id"]] = r
    return _CETI


def auc(pos, neg):
    """Probability that a value of pos exceeds one of neg (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None]).mean()
    eq = (pos[:, None] == neg[None]).mean()
    return float(gt + 0.5 * eq)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm.body import openbody as BO
    from mdfm.body.rotations import matrix_to_axis_angle, rot6d_to_matrix, rot_y, yaw_angle
    from mdfm.data.open_body_store import OpenBodyStore
    from mdfm.model.prior import PhenotypePrior
    from mdfm.synth import openbody as OB
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    st = OpenBodyStore()
    persons = st.persons()

    # ------------------------------------------------------------ phenotype per person, sex, stature
    rows = []
    for (src, pid), idx in sorted(persons.items()):
        ph = st.phenotype(idx[0])
        t = st.takes[idx[0]]
        sex, h = sex_and_stature(src, t["subject_ns"])
        rows.append({"source": src, "person_id": pid, "phenotype": ph, "sex": sex, "stature_m": h})
    m = OB.model()
    for r in rows:
        r["fitted_stature_m"] = OB.anthropometry(BO.as_phenotype(r["phenotype"]), m)["height_m"]
    U = BO.to_latent(torch.as_tensor(np.stack([np.clip(r["phenotype"], [BO.DOMAIN[k][0] + 1e-4 for k in
                                                                         BO.PHENOTYPE_KEYS], 1 - 1e-4)
                                               for r in rows])))
    code = torch.tensor([{"F": 0, "M": 1}.get(r["sex"], 2) for r in rows])
    prior = PhenotypePrior.fit(U, code)
    names = ("female", "male", "neutral")
    pp = {g: {"n_persons": int((code == i).sum()) if i < 2 else len(rows), "mean": prior.mean[i].double().tolist(),
              "cov": prior.cov[i].double().tolist(),
              "phenotype_at_mean": dict(zip(BO.PHENOTYPE_KEYS, BO.to_phenotype(prior.mean[i].double()).tolist()))}
          for i, g in enumerate(names)}
    gi = BO.PHENOTYPE_KEYS.index("gender")
    gF = [r["phenotype"][gi] for r in rows if r["sex"] == "F"]
    gM = [r["phenotype"][gi] for r in rows if r["sex"] == "M"]
    g_all = np.array([r["phenotype"][gi] for r in rows])
    stat = [(r["stature_m"], r["fitted_stature_m"], r["source"]) for r in rows if r["stature_m"]]
    st_err = np.array([b - a for a, b, _ in stat])
    gender = {"all_persons": {"n": len(g_all), "median": float(np.median(g_all)), "q10": float(np.percentile(g_all, 10)),
                              "q90": float(np.percentile(g_all, 90)), "share_over_0.75": float((g_all > 0.75).mean())},
              "women": {"n": len(gF), "median": float(np.median(gF)) if gF else None},
              "men": {"n": len(gM), "median": float(np.median(gM)) if gM else None},
              "auc_women_over_men": auc(gF, gM),
              "by_source_median": {s: float(np.median([r["phenotype"][gi] for r in rows if r["source"] == s]))
                                   for s in sorted({r["source"] for r in rows})},
              "bounds_share": {k: float(np.mean([(r["phenotype"][j] < BO.DOMAIN[k][0] + 0.02) |
                                                 (r["phenotype"][j] > BO.DOMAIN[k][1] - 0.02) for r in rows]))
                               for j, k in enumerate(BO.PHENOTYPE_KEYS)}}
    stature = {"n": len(stat), "fitted_minus_measured_m_median": float(np.median(st_err)) if len(stat) else None,
               "abs_error_m_median": float(np.median(np.abs(st_err))) if len(stat) else None,
               "corr": float(np.corrcoef([a for a, _, _ in stat], [b for _, b, _ in stat])[0, 1]) if len(stat) > 2 else None,
               "by_source": {s: {"n": int(sum(1 for *_, x in stat if x == s)),
                                 "fitted_minus_measured_m_median": float(np.median([b - a for a, b, x in stat if x == s]))}
                             for s in sorted({x for *_, x in stat})}}

    # ------------------------------------------------------------ mean pose and limit percentiles
    tot, wsum, samples = np.zeros((22, 6)), 0.0, []
    for i, t in enumerate(st.takes):
        a = st.arrays(t["source"])
        x = torch.from_numpy(np.asarray(a["rot6d"][t["offset"]:t["offset"] + t["n_knots"]:STRIDE], dtype=np.float64))
        R0 = rot6d_to_matrix(x[:, 0])
        root = rot_y(-yaw_angle(R0)) @ R0
        x = torch.cat([torch.cat([root[:, :, 0], root[:, :, 1]], -1)[:, None], x[:, 1:]], 1)
        w = st.knot_weights(i) * STRIDE
        tot += w * x.sum(0).numpy()
        wsum += w * x.shape[0]
        samples.append(matrix_to_axis_angle(rot6d_to_matrix(x)).numpy().astype(np.float32))
    mean6 = tot / wsum
    aa = np.concatenate(samples)
    lo_hi = np.percentile(aa, [0.1, 99.9], axis=0)

    # ------------------------------------------------------------ window priors
    gen = np.random.default_rng(SEED)
    K = int(round(WIN_S * 100.0))
    keys = sorted(persons)
    sum1, sum2, nk, c1, c2 = np.zeros(3), np.zeros(3), 0, np.zeros(3), np.zeros(3)
    rest_cache = {}
    drawn = set()
    for _ in range(N_WINDOWS):
        p = keys[int(gen.integers(len(keys)))]
        cand = persons[p]
        kn = np.array([st.takes[i]["n_knots"] for i in cand], dtype=np.float64)
        i = cand[int(gen.choice(len(cand), p=kn / kn.sum()))]
        n = st.takes[i]["n_knots"]
        s0 = int(gen.integers(max(n - K, 0) + 1))
        tk = st.take(i)
        r6 = torch.from_numpy(np.asarray(tk["rot6d"][s0:s0 + K], dtype=np.float64))
        tr = np.asarray(tk["transl"][s0:s0 + K], dtype=np.float64)
        L = r6.shape[0]
        c = L // 2
        if p not in rest_cache:
            rest_cache[p] = st.rest(i)[0]
        rp = rest_cache[p]
        R0 = rot6d_to_matrix(r6[c, 0])
        Ry = rot_y(-yaw_angle(R0)).numpy()
        pel = tr[c] + rp
        origin = np.array([pel[0], 0.0, pel[2]])
        tw = (tr + rp - origin) @ Ry.T - rp
        sum1 += tw.sum(0)
        sum2 += (tw ** 2).sum(0)
        nk += L
        c1 += tw[c]
        c2 += tw[c] ** 2
        drawn.add(p)
    mean = sum1 / nk
    sd = np.sqrt(sum2 / nk - mean ** 2)
    cm = c1 / N_WINDOWS
    csd = np.sqrt(c2 / N_WINDOWS - cm ** 2)

    head = git_head()
    priors = {"version": "bodies_open_v1", "status": "from the persons of the open body store (train folds)",
              "store_hashes_sha256": {k: v for k, v in st.hashes["files"].items() if k.endswith("takes.csv")},
              "person_lists_sha256": st.hashes["person_lists_sha256"], "n_persons": len(rows),
              "phenotype_prior": {"note": "Gaussian over the phenotype latent u of mdfm/body/openbody.py, one vector "
                                  "per person; neutral pools every person (sex unknown); PhenotypePrior(path=...)",
                                  **pp},
              "mean_pose": {"note": "person-weighted mean 6D value per joint over every 10th knot, row 0 the root "
                            "orientation in each knot's own heading frame", "rot6d": mean6.tolist(),
                            "n_knots_sampled": int(len(aa))},
              "joint_limits_store": {"note": "for information: the model's limits are the anatomical table of "
                                     "reports/openbody/joint_limits.json", "percentiles": [0.1, 99.9],
                                     "lo": lo_hi[0].tolist(), "hi": lo_hi[1].tolist()},
              "gender_value": gender, "stature": stature, "script": "scripts/build_open_body_priors.py",
              "mdfm_head": head, "run_at": datetime.datetime.now().isoformat(timespec="seconds")}
    wp = {"note": "mean root translation (m) of windows of the open body store in the window frame, drawn by person; "
          "transl_centre at the centre knot L // 2 (as reports/body_store/window_priors.json)",
          "transl_mean": mean.tolist(), "transl_sd": sd.tolist(), "transl_centre_mean": cm.tolist(),
          "transl_centre_sd": csd.tolist(), "n_windows": N_WINDOWS, "n_knots": int(nk), "n_persons_drawn": len(drawn),
          "seed": SEED, "win_s": WIN_S, "script": "scripts/build_open_body_priors.py", "mdfm_head": head,
          "run_at": priors["run_at"]}
    with open(os.path.join(OUT, "priors.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(priors, f, indent=1)
    with open(os.path.join(OUT, "window_priors.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(wp, f, indent=1)
    L = ["# Open body priors (step M4)", "",
         f"scripts/build_open_body_priors.py, MD_FM {head}, {priors['run_at']}; the open body store of "
         f"reports/open_body_store/store.json, {len(rows)} persons.", "",
         "## Phenotype prior", "",
         "| row | persons | phenotype at the mean (gender, age, muscle, weight, height, proportions) |", "|---|---|---|"]
    for g in names:
        L.append(f"| {g} | {pp[g]['n_persons']} | " + ", ".join(f"{v:.2f}" for v in pp[g]["phenotype_at_mean"].values())
                 + " |")
    L += ["", "## The gender value", "",
          f"All persons: median {gender['all_persons']['median']:.2f} (10th to 90th percentile "
          f"{gender['all_persons']['q10']:.2f} to {gender['all_persons']['q90']:.2f}), "
          f"{100 * gender['all_persons']['share_over_0.75']:.0f} percent above 0.75. Persons of known sex: women "
          f"{gender['women']['n']} (median {gender['women']['median']:.2f}), men {gender['men']['n']} (median "
          f"{gender['men']['median']:.2f}); AUC of the gender value separating women from men "
          f"{gender['auc_women_over_men']:.2f} (0.5: no separation).", "",
          "By source (median): " + ", ".join(f"{s} {v:.2f}" for s, v in gender["by_source_median"].items()) + ".", "",
          "Share of persons within 0.02 of a bound of the box: " + ", ".join(
              f"{k} {100 * v:.0f} %" for k, v in gender["bounds_share"].items()) + ".", "",
          f"Stature (the fitted rest mesh against the source's measure, {stature['n']} persons): fitted minus measured "
          f"median {stature['fitted_minus_measured_m_median']:.3f} m, absolute error median "
          f"{stature['abs_error_m_median']:.3f} m, correlation {stature['corr']:.2f}; by source " + ", ".join(
              f"{s} {v['fitted_minus_measured_m_median']:.3f} m ({v['n']})" for s, v in stature["by_source"].items()) + ".",
          "", "## Window priors", "",
          f"{N_WINDOWS} windows of {WIN_S:.0f} s drawn by person ({len(drawn)} persons drawn): root translation mean "
          f"{np.round(mean, 3).tolist()} m (sd {np.round(sd, 3).tolist()}), at the centre knot "
          f"{np.round(cm, 3).tolist()} m (sd {np.round(csd, 3).tolist()}).", ""]
    text = "\n".join(L) + "\n"
    if "\u2014" in text:
        sys.exit("em-dash in summary.md")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    prov = {"script": "scripts/build_open_body_priors.py", "mdfm_head": head, "run_at": priors["run_at"],
            "store_root": st.root, "seconds": round(time.time() - t0, 1),
            "outputs": ["priors.json", "window_priors.json", "summary.md"]}
    with open(os.path.join(OUT, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
