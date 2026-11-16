"""The layout families of the synthetic factory on the open body (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4):
reports/open_layouts/layouts.json, in the schema of reports/layouts/layouts.json (scripts/build_layouts.py; read by
mdfm.factory.layouts.load_layouts(path)).

What in v1's layouts depends on the body. The families themselves do not: their sites, rates, channel masks,
acc_content and force units are read from real records (layouts.read_real, build_families, check_families, the
same calls as scripts/build_layouts.py), and no SMPL+H body enters them. Two things do:
- the mount prior file the families' mount names refer to: here reports/open_mount_priors/priors.json
  (scripts/build_open_mount_priors.py), in the 'mount_priors' key;
- v1's bench (bench.json: the factory's IMU against the reference renderer, throughput, memory), which draws bodies
  from the v1 body store (AMASS through SMPL+H) and builds SMPLHBody references. It is not run here: the factory's
  sampler (mdfm/factory/synth.SynthSampler) still draws from the v1 store with the SMPL+H SkeletonBank, and its
  port to the open body store is part of the model's port (MFM on the open body, step M4). summary.md says so.

What the families carry at run time from the body is the IMU site placement: imu_render.site_offsets on the
subject's rest joints (fractions of the rest bones plus the constant surface offsets of SITE_RULES, in the segment
frames x left, y up, z forward that both bodies share). This script measures it on the open body:
- the site offsets of the pooled phenotype prior's mean person (reports/open_body_priors/priors.json, row
  neutral) through the P3 basis, and their spread over the open body store's 397 persons (each person's
  phenotype, P3 basis);
- the surface part against Anny's skin (SKIN rule): from the site's anchor (the joint, or the point at the bone
  fraction) along the direction of the SITE_RULES surface offset, the distance to the first crossing of the skin's
  triangles (the canonical rest mesh of the same person, the faces of its largest connected part, so the mouth
  cavity and the eyes are left out); the number of crossings is written (odd: the anchor lies inside the skin). SITE_RULES' surface constants were anatomical guesses, not measured on the
  SMPL+H surface, so nothing of v1 is ported from a surface; the renderer keeps them, and the table shows where
  they lie against Anny's skin and against the placement residual bound of the IMU nuisance (OFFSET_MAX_M, 3 cm).
- with --v1-reference (the SMPL+H body behind the reference flag of mdfm/body/backend.py, for the comparison only;
  the model file is read in memory and no array is written): the same offsets on v1's mean person (SMPL+H neutral,
  the neutral mean betas of reports/body_store/priors.json); only the differences and the norms are written.

Writes reports/open_layouts/: layouts.json, checks.csv, site_offsets.csv, summary.md, _provenance.json.
    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/build_open_layouts.py
    [--v1-reference]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TREE, "scripts"))
sys.path.insert(0, TREE)

OUT = os.path.join(TREE, "reports", "open_layouts")
V1_LAYOUTS = os.path.join(TREE, "reports", "layouts", "layouts.json")
OPEN_MOUNT_PRIORS = "reports/open_mount_priors/priors.json"
OPEN_BODY_PRIORS = os.path.join(TREE, "reports", "open_body_priors", "priors.json")
V1_BODY_PRIORS = os.path.join(TREE, "reports", "body_store", "priors.json")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    import subprocess
    r = subprocess.run(["git", "-C", TREE, "log", "-1", "--format=%h"], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------- the families
def build_doc(head):
    """The families from the real records (scripts/build_layouts.py's calls) and the document with the open mount
    prior: (doc, checks). Raises when a family stream differs from its record."""
    from mdfm.factory import camera_render as CR
    from mdfm.factory import layouts as L
    real = L.read_real()
    layouts = L.build_families(real, CR.FAMILIES["generic"].fps)
    checks = L.check_families(layouts, real)
    bad = [r for r in checks if r["verdict"] != "equal"]
    if bad:
        raise ValueError(f"layouts differ from their records: {bad}")
    doc = {"format": L.FORMAT, "moveatlas_head": head, "device_noise": "reports/g2_renderer/device_noise.json",
           "mount_priors": OPEN_MOUNT_PRIORS, "camera_family": "generic",
           "families": {n: L.layout_to_json(x) for n, x in layouts.items()}}
    return doc, checks


def compare_families(doc, v1):
    """Per family: 'equal' or the differing stream fields against v1's layouts.json."""
    out = {}
    for name in sorted(set(doc["families"]) | set(v1["families"])):
        a, b = doc["families"].get(name), v1["families"].get(name)
        if a is None or b is None:
            out[name] = "only in " + ("v1" if a is None else "open")
            continue
        if a == b:
            out[name] = "equal"
            continue
        diffs = [k for k in a if k != "streams" and a[k] != b.get(k)]
        for i, (sa, sb) in enumerate(zip(a["streams"], b["streams"])):
            diffs += [f"streams[{i}].{k}" for k in sa if sa[k] != sb.get(k)]
        if len(a["streams"]) != len(b["streams"]):
            diffs.append("stream count")
        out[name] = "differs: " + ", ".join(diffs)
    return out


# ---------------------------------------------------------------- the open site offsets
def mean_phenotype():
    """The pooled phenotype prior's mean person (reports/open_body_priors/priors.json, row neutral), (6,)."""
    from mdfm.body import openbody as BO
    with open(OPEN_BODY_PRIORS, encoding="utf-8") as f:
        p = json.load(f)["phenotype_prior"]["neutral"]["phenotype_at_mean"]
    return np.array([p[k] for k in BO.PHENOTYPE_KEYS], dtype=np.float64)


def open_site_offsets(phenotypes):
    """Site offsets (B, 14, 3) m and joints (14,) of the open rest skeletons of phenotypes (B, 6), P3 basis."""
    import torch
    from mdfm.body.openbody import OpenSkeletonBank
    from mdfm.factory.imu_render import SITES, site_offsets
    bank = OpenSkeletonBank(torch.float64)
    with torch.no_grad():
        rest = bank.rest(torch.as_tensor(phenotypes, dtype=torch.float64))
        joints, off = site_offsets(rest, SITES)
    return joints.numpy(), off.numpy(), rest.numpy()


def store_phenotypes():
    """One phenotype per person of the open body store (397), in sorted person order."""
    from mdfm.data.open_body_store import OpenBodyStore
    st = OpenBodyStore()
    return np.stack([st.phenotype(idx[0]) for _, idx in sorted(st.persons().items())]), st


def ray_hits(origin, direction, V, F):
    """Distances t > 0 at which the ray origin + t direction crosses the triangles F of the vertices V
    (Moller-Trumbore), sorted."""
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    pv = np.cross(direction, e2)
    det = (e1 * pv).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = origin - v0
    u = (tv * pv).sum(1) * inv
    qv = np.cross(tv, e1)
    v = (qv * direction).sum(1) * inv
    t = (qv * e2).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.sort(t[hit])


def skin_distances(phenotype):
    """SKIN rule (module docstring) on Anny's canonical rest mesh of one phenotype: {site: (surface constant m,
    distance from the anchor to the first skin crossing along its direction m or None, crossings)}; an odd number
    of crossings puts the anchor inside the skin."""
    import torch
    from mdfm.body import openbody as BO
    from mdfm.factory.imu_render import SITE_RULES, SITES
    from mdfm.synth import openbody as OB
    ob = OB.OpenBody(BO.as_phenotype(phenotype), OB.model())
    with torch.no_grad():
        J = ob.rest_joints().cpu().numpy()
    V = np.asarray(ob.rest_vertices_np(), dtype=np.float64)
    surf = np.asarray(ob.surface_mask())
    F = np.asarray(ob.faces)
    F = F[surf[F].all(1)]
    out = {}
    for s in SITES:
        k, bone, frac, surface = SITE_RULES[s]
        surface = np.asarray(surface, dtype=np.float64)
        anchor = J[k] if bone is None else J[k] + frac * (J[bone[1]] - J[bone[0]])
        t = ray_hits(anchor, surface / np.linalg.norm(surface), V, F)
        out[s] = (float(np.linalg.norm(surface)), float(t[0]) if len(t) else None, int(len(t)))
    return out


def v1_reference_offsets():
    """Site offsets (14, 3) of v1's mean person: SMPL+H neutral with the neutral mean betas of
    reports/body_store/priors.json, through mdfm/body/backend.py behind the reference flag. Comparison only."""
    import torch
    from mdfm.body import backend
    from mdfm.factory.imu_render import SITES, site_offsets
    with open(V1_BODY_PRIORS, encoding="utf-8") as f:
        betas = json.load(f)["shape_prior"]["neutral"]["mean"]
    bank = backend.skeleton_bank("smplh", allow_reference=True, dtype=torch.float64)
    with torch.no_grad():
        rest = bank.rest(torch.tensor([betas], dtype=torch.float64), torch.tensor([2]))
        _, off = site_offsets(rest, SITES)
    stature = float(rest[0, 15, 1] - rest[0, [10, 11], 1].min())
    return off[0].numpy(), stature


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"))
    p.add_argument("--out", default=OUT)
    p.add_argument("--v1-reference", action="store_true",
                   help="also the site offsets of v1's mean person (SMPL+H behind the reference flag; comparison only)")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    if not os.path.isfile(os.path.join(a.moveatlas, "INDEX.md")):
        sys.exit(f"MoveAtlas tree not found at {a.moveatlas} (no INDEX.md)")
    os.environ["MOVEATLAS_ROOT"] = a.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    from _common import moveatlas_head
    from mdfm import paths
    from mdfm.factory.imu_render import SITE_RULES, SITES
    from mdfm.factory import layouts as L
    from mdfm.kernels.nuisance import OFFSET_MAX_M
    paths.assert_not_moveatlas(a.out)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    head = moveatlas_head(a.moveatlas)

    doc, checks = build_doc(head)
    with open(os.path.join(a.out, "layouts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, default=float)
        f.write("\n")
    write_csv(os.path.join(a.out, "checks.csv"), checks)
    back = L.load_layouts(os.path.join(a.out, "layouts.json"))
    with open(V1_LAYOUTS, encoding="utf-8") as f:
        v1 = json.load(f)
    fam_cmp = compare_families(doc, v1)
    print(f"layouts: {len(doc['families'])} families, {len(checks)} checks equal ({time.time() - t0:.0f} s)", flush=True)

    # ------------------------------------------------ site offsets on the open body
    ph0 = mean_phenotype()
    joints, off0, rest0 = open_site_offsets(ph0[None])
    off0, rest0 = off0[0], rest0[0]
    stature0 = float(rest0[15, 1] - rest0[[10, 11], 1].min())
    phs, store = store_phenotypes()
    _, offs, _ = open_site_offsets(phs)
    norms = np.linalg.norm(offs, axis=-1)
    skin = skin_distances(ph0)
    v1off, v1stature = (None, None)
    if a.v1_reference:
        v1off, v1stature = v1_reference_offsets()
    rows = []
    for k, s in enumerate(SITES):
        jk, bone, frac, surface = SITE_RULES[s]
        sc, depth, nhit = skin[s]
        r = {"site": s, "joint": int(joints[k]), "bone_fraction": frac if bone is not None else "",
             "offset_x_m": round(float(off0[k, 0]), 4), "offset_y_m": round(float(off0[k, 1]), 4),
             "offset_z_m": round(float(off0[k, 2]), 4), "norm_m": round(float(np.linalg.norm(off0[k])), 4),
             "store_norm_q10_m": round(float(np.percentile(norms[:, k], 10)), 4),
             "store_norm_median_m": round(float(np.median(norms[:, k])), 4),
             "store_norm_q90_m": round(float(np.percentile(norms[:, k], 90)), 4),
             "surface_constant_m": round(sc, 4), "skin_distance_m": round(depth, 4) if depth is not None else "",
             "surface_minus_skin_m": round(sc - depth, 4) if depth is not None else "", "skin_crossings": nhit,
             "beyond_offset_bound": (abs(sc - depth) > OFFSET_MAX_M) if depth is not None else ""}
        if v1off is not None:
            r.update({"v1_norm_m": round(float(np.linalg.norm(v1off[k])), 4),
                      "open_minus_v1_norm_m": round(float(np.linalg.norm(off0[k] - v1off[k])), 4)})
        rows.append(r)
    write_csv(os.path.join(a.out, "site_offsets.csv"), rows)

    prov = {"script": "scripts/build_open_layouts.py", "moveatlas_root": a.moveatlas, "moveatlas_head": head,
            "mdfm_head": git_head(), "body": "open (Anny, P3 basis)", "store_root": store.root,
            "open_body_priors_sha256": file_sha256(OPEN_BODY_PRIORS),
            "open_mount_priors": OPEN_MOUNT_PRIORS,
            "open_mount_priors_sha256": (file_sha256(os.path.join(TREE, OPEN_MOUNT_PRIORS))
                                         if os.path.exists(os.path.join(TREE, OPEN_MOUNT_PRIORS)) else None),
            "v1_layouts_sha256": file_sha256(V1_LAYOUTS),
            "layouts_sha256": file_sha256(os.path.join(a.out, "layouts.json")),
            "v1_reference": bool(a.v1_reference),
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "runtime_s": round(time.time() - t0, 1),
            "outputs": ["_provenance.json", "checks.csv", "layouts.json", "site_offsets.csv", "summary.md"]}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")

    # ------------------------------------------------ summary
    eq = [n for n, v in fam_cmp.items() if v == "equal"]
    S = ["# Layout families on the open body (step M4)", "",
         f"Generated by scripts/build_open_layouts.py (method in its docstring), MD_FM {prov['mdfm_head']}, MoveAtlas "
         f"{head.split(' ')[0]}; v1 is reports/layouts/ (scripts/build_layouts.py). Same schema as v1 (format "
         f"{doc['format']}, the same top-level keys); loads with layouts.load_layouts: {len(back)} families.", "",
         "## Families, open against v1", "",
         f"{sum(r['verdict'] == 'equal' for r in checks)} of {len(checks)} family streams equal their records in "
         f"sites, rate and channel mask (checks.csv). Against v1's layouts.json: {len(eq)} of {len(fam_cmp)} families "
         "identical" + ("" if len(eq) == len(fam_cmp) else " (" + "; ".join(
             f"{n}: {v}" for n, v in fam_cmp.items() if v != "equal") + ")") + ". The families read no body: the only "
         f"body-dependent key is mount_priors, {OPEN_MOUNT_PRIORS} in place of {v1['mount_priors']}; moveatlas_head "
         f"is this run's ({head.split(' ')[0]}; v1 {v1['moveatlas_head'].split(' ')[0]}).", "",
         "Bench (v1's bench.json): not run on the open body. It draws windows through the factory's sampler "
         "(mdfm/factory/synth.SynthSampler), which still reads the v1 body store with the SMPL+H SkeletonBank, and "
         "checks them against SMPLHBody references; it is rerun once the sampler draws from the open body store "
         "(MFM on the open body).", "",
         "## IMU site offsets on the open body", "",
         f"imu_render.site_offsets on the rest joints of the pooled phenotype prior's mean person (phenotype "
         f"{np.round(ph0, 2).tolist()}, rest stature {stature0:.3f} m from the head joint to the lower foot joint) and "
         f"the spread of the offset norm over the store's {len(phs)} persons; the surface constants of SITE_RULES "
         f"against Anny's skin (SKIN rule, first crossing of the skin's triangles)"
         + (f"; v1's mean person (SMPL+H neutral, mean betas; rest stature {v1stature:.3f} m by the same joints): "
            "the norm of the open offset minus the v1 offset." if v1off is not None else
            "; the v1 comparison was not run (--v1-reference)."), "",
         "| site | joint | offset norm (m) | store 10th / 50th / 90th (m) | surface constant (m) | skin distance (m) | "
         "constant minus skin (m) | crossings | beyond the 3 cm placement bound |" + (" v1 norm (m) | open minus v1 (m) |"
                                                                          if v1off is not None else ""),
         "|---|---|---|---|---|---|---|---|---|" + ("---|---|" if v1off is not None else "")]
    for r in rows:
        S.append(f"| {r['site']} | {r['joint']} | {r['norm_m']} | {r['store_norm_q10_m']} / {r['store_norm_median_m']} / "
                 f"{r['store_norm_q90_m']} | {r['surface_constant_m']} | {r['skin_distance_m']} | "
                 f"{r['surface_minus_skin_m']} | {r['skin_crossings']} | {r['beyond_offset_bound']} |"
                 + (f" {r['v1_norm_m']} | {r['open_minus_v1_norm_m']} |" if v1off is not None else ""))
    beyond = [r["site"] for r in rows if r["beyond_offset_bound"] is True]
    S += ["", "Rule kept for the surface part: the renderer and the factory keep SITE_RULES' constants (they were "
          "anatomical guesses, not SMPL+H surface measurements, and they set the prior centre of the IMU nuisance's "
          f"placement, whose residual reaches {OFFSET_MAX_M} m). Sites where the constant and Anny's skin differ by "
          f"more than that bound: {', '.join(beyond) if beyond else 'none'}.", "",
          f"layouts.json SHA-256 {prov['layouts_sha256']}.", ""]
    text = "\n".join(S)
    assert chr(0x2014) not in text
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
