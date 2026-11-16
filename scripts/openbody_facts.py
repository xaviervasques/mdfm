"""Measured facts of the open body for the port (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md sections 3 to 6).

Reads Anny through mdfm.synth.openbody and writes reports/openbody/anny_facts.json and reports/openbody/
vertex_sets.json:
- the rig: the 104 bones of the 'anny' rig with their parents, the bones per hand, the bones of the full MakeHuman
  rig (rig.default.json of anny/data/mpfb2) that the 'anny' rig leaves out; the joint map and the owner of every bone;
- the mesh: vertices, triangles, closed or not (trimesh), connected parts;
- the rest pose: the angles that say A pose or T pose (upper arm below the horizontal, elbow flexion, leg
  abduction, knee flexion), the palm's direction, and the angle of each canonical correction A_b of the mapped bones;
- the skinning: the method, the bones per vertex, the absence of pose blend shapes;
- the phenotypes: their labels and anchors, and the adult reach (stature over the height value, BMI over the weight
  value at four muscle values, both sexes, ages 25 to 85);
- the 14 identities solved to their stature and BMI targets (openbody.IDENTITY_SPECS), with the measured values;
- the speed of the adapter's posing (joints and vertices) for a 30 s clip at the export clock (30 fps, 900 frames)
  on the CPU (float64 and float32) and on the GPU (float32, and float64 for reference), and of Anny's own forward;
- the differentiability: torch.autograd.gradcheck of the posed joints and of a subset of vertices with respect to
  axis-angle rotations (the inverse kinematics the port may need).
vertex_sets.json: the vertex ids of the mesh sets of openbody.vertex_sets (rules on the skinning and the geometry).

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/openbody_facts.py
"""
import datetime as dt
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, "reports", "openbody")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm.synth import openbody as OB  # noqa: E402
from mdfm.body.rotations import axis_angle_to_matrix  # noqa: E402

F64 = torch.float64
CLIP_S, FPS = 30.0, 30.0


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(R)) - 1) / 2, -1, 1))))


def rig_facts(m, ob):
    import anny
    from anny.models import model_data as MD
    labels = list(m.bone_labels)
    rig_json = os.path.join(os.path.dirname(anny.__file__), "data", "mpfb2", "rigs", "standard", "rig.default.json")
    with open(rig_json, encoding="utf-8") as f:
        full = sorted(json.load(f))
    left_out = sorted(set(full) - set(labels))
    hand = [b for b in labels if b.endswith(".L") and (b.startswith("finger") or b.startswith("metacarpal"))]
    removed_by = {k: sorted(set(getattr(MD, n)) & set(left_out)) for k, n in
                  (("tongue", "_tongue_bone_labels"), ("breast", "_breast_bone_labels"),
                   ("facial expression", "_facial_expression_bone_labels"), ("zero weight", "_zero_weight_bone_labels"))}
    owner = OB.owner_table(m)
    return {"rig": "anny", "bones": len(labels),
            "bone_list": [{"i": i, "bone": b, "parent": (labels[p] if p >= 0 else None),
                           "program_joint_owner": OB.JOINT_NAMES[int(owner[i])]}
                          for i, (b, p) in enumerate(zip(labels, m.bone_parents))],
            "hand_bones_per_hand": len(hand), "hand_bones_left": hand,
            "finger_joints_per_hand": {"thumb": 3, "index": 4, "middle": 4, "ring": 4, "pinky": 4,
                                       "note": "the four fingers carry a metacarpal bone and three phalanges, the "
                                               "thumb three bones (finger1-1 at the carpometacarpal joint)"},
            "makehuman_rig_bones": len(full), "left_out_of_the_anny_rig": len(left_out),
            "left_out_by_group": {k: len(v) for k, v in removed_by.items()},
            "left_out_other": sorted(set(left_out) - set().union(*removed_by.values())),
            "joint_map": [{"program_joint": n, "anny_bone": b} for n, b in OB.JOINT_MAP],
            "mapped": len(OB.JOINT_MAP), "unmapped_carried_by_an_owner": len(labels) - len(OB.JOINT_MAP),
            "pose_parameterizations_of_anny": ["world", "local-bone-world", "local-bone", "local-ref", "world-orient"],
            "bone_orientation": m.bone_orientation}


def rest_facts(ob):
    C = ob.C.numpy()
    h = (ob.rest_heads.numpy() @ C.T)                                  # Anny's rest in the program's frame
    idx = OB.bone_index(ob.m)
    H = lambda b: h[idx[b]]                                            # noqa: E731
    out = {}
    for side, s in (("left", "L"), ("right", "R")):
        up = H(f"lowerarm01.{s}") - H(f"upperarm01.{s}")
        fore = H(f"wrist.{s}") - H(f"lowerarm01.{s}")
        thigh = H(f"lowerleg01.{s}") - H(f"upperleg01.{s}")
        shank = H(f"foot.{s}") - H(f"lowerleg01.{s}")
        n = C @ OB.palm_normal(side, ob.rest_heads, ob.m).numpy()
        out[side] = {"upper_arm_below_horizontal_deg": float(np.degrees(np.arcsin(-up[1] / np.linalg.norm(up)))),
                     "elbow_flexion_deg": angle_deg(up, fore),
                     "thigh_from_vertical_deg": angle_deg(thigh, [0, -1, 0]),
                     "knee_flexion_deg": angle_deg(thigh, shank),
                     "palm_normal_rest_ours": [round(float(x), 3) for x in n]}
    A = ob.A.numpy()
    corr = {n: rot_angle_deg(A[idx[b]]) for n, b in OB.JOINT_MAP}
    v = ob.rest_vertices.numpy() @ C.T
    vc = ob.canonical_vertices().numpy()
    return {"rest_pose": "A pose (the arms about 45 deg below the horizontal, the elbows slightly flexed)",
            "per_side": out, "canonical_correction_deg": corr,
            "canonical_corrected_joints": sorted(k for k, a in corr.items() if a > 1e-9),
            "rest_extent_m": {"x": float(np.ptp(v[:, 0])), "y": float(np.ptp(v[:, 1])), "z": float(np.ptp(v[:, 2]))},
            "canonical_extent_m": {"x": float(np.ptp(vc[:, 0])), "y": float(np.ptp(vc[:, 1])),
                                   "z": float(np.ptp(vc[:, 2]))}}


def mesh_facts(m, ob):
    import trimesh
    tm = trimesh.Trimesh(ob.rest_vertices.numpy(), ob.faces, process=False)
    parts = tm.split(only_watertight=False)
    return {"topology": "anny", "vertices": int(ob.rest_vertices.shape[0]), "triangles": int(ob.faces.shape[0]),
            "watertight": bool(tm.is_watertight), "connected_parts": len(parts),
            "part_vertices": sorted((int(len(p.vertices)) for p in parts), reverse=True),
            "skinning": {"method": "linear blend skinning (dense torch, skinning_method='lbs')",
                         "bones_per_vertex_max": int(m.vertex_bone_weights.shape[1]),
                         "pose_blend_shapes": False,
                         "anny_methods": ["lbs", "dqs", "warp_lbs"]}}


def phenotype_facts(m):
    from anny.models.phenotype import make_phenotype_anchors
    from anny.models.model_data import PHENOTYPE_VARIATIONS
    anchors = {k: [round(float(x), 4) for x in v] for k, v in make_phenotype_anchors(F64).items()
               if k in m.phenotype_labels}
    reach = []
    for sex in ("female", "male"):
        for age in (25, 45, 65, 85):
            row = {"sex": sex, "age_years": age, "age_value": OB.age_phenotype(age)}
            row["stature_m_at_height_value"] = {
                str(hv): round(OB.anthropometry(OB.Phenotype(gender=OB.SEX_GENDER[sex], age=OB.age_phenotype(age),
                                                             height=hv), m)["height_m"], 3)
                for hv in (0.0, 0.25, 0.5, 0.75, 1.0)}
            row["bmi_at_weight_value"] = {
                f"muscle {mu}": {str(wv): round(OB.anthropometry(OB.Phenotype(
                    gender=OB.SEX_GENDER[sex], age=OB.age_phenotype(age), weight=wv, muscle=mu), m)["bmi"], 2)
                    for wv in (0.0, 0.25, 0.5, 0.75, 1.0)} for mu in (0.1, 0.25, 0.35, 0.5)}
            reach.append(row)
    return {"labels": list(m.phenotype_labels), "anchors": anchors,
            "variations": {k: PHENOTYPE_VARIATIONS[k] for k in m.phenotype_labels},
            "age_rule": "age value = 2/3 + (years - 25) / 65 / 3 (MakeHuman's young target 25 years, old 90 years; "
                        "openbody.age_phenotype)",
            "mass_rule": f"mesh volume x {OB.BODY_DENSITY} kg/m3 (anny.Anthropometry's density)",
            "adult_reach": reach}


def identity_facts(m):
    out = []
    for name, *_ in OB.IDENTITY_SPECS:
        t = OB.identity_targets(name)
        ph = OB.solve_phenotype(t["sex"], t["age_years"], t["height_m"], t["bmi"], muscle=t["muscle"], m=m)
        a = OB.anthropometry(ph, m)
        out.append({"name": name, **t, "phenotype": ph.as_dict(),
                    "measured": {k: round(v, 4) for k, v in a.items()}})
    return out


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def speed_facts(ob):
    n = int(CLIP_S * FPS)
    torch.manual_seed(0)
    R = axis_angle_to_matrix(0.3 * torch.randn(n, 52, 3, dtype=F64))
    tr = 0.1 * torch.randn(n, 3, dtype=F64)
    rows = []
    cases = [("cpu", F64), ("cpu", torch.float32)]
    if torch.cuda.is_available():
        cases += [("cuda", torch.float32), ("cuda", F64)]
    for dev_name, dt_ in cases:
        dev = torch.device(dev_name)
        b = ob.to(dev, dt_)
        Rd, td = R.to(dev, dt_), tr.to(dev, dt_)
        with torch.no_grad():
            b.pose(Rd[:8, :22], td[:8], Rd[:8, 22:])                     # warm-up
            _sync(dev)
            res = {}
            for label, verts in (("joints only", False), ("joints and vertices", True)):
                times = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    b.pose(Rd[:, :22], td, Rd[:, 22:], vertices=verts, chunk=128 if dev.type == "cuda" else 64)
                    _sync(dev)
                    times.append(time.perf_counter() - t0)
                res[label] = round(min(times), 4)
        # float32 against float64 on the vertices (the precision a GPU path would keep)
        rows.append({"device": dev_name, "dtype": str(dt_).replace("torch.", ""), "frames": n,
                     "seconds_min_of_3": res})
    with torch.no_grad():
        J64, V64 = ob.pose(R[:60, :22], tr[:60], R[:60, 22:])
        J32, V32 = ob.to("cpu", torch.float32).pose(R[:60, :22], tr[:60], R[:60, 22:])
    m = ob.m
    anny_rows = []
    for dev_name, dt_ in cases:
        mm = m.to(device=dev_name, dtype=dt_)
        poses = torch.eye(4, dtype=dt_, device=dev_name)[None, None].repeat(n, mm.bone_count, 1, 1)
        poses[:, :, :3, :3] = axis_angle_to_matrix(0.2 * torch.randn(n, mm.bone_count, 3, dtype=F64)).to(dev_name, dt_)
        with torch.no_grad():
            mm(pose_parameters=poses[:4])
            _sync(torch.device(dev_name))
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                for a in range(0, n, 128):
                    mm(pose_parameters=poses[a:a + 128])
                _sync(torch.device(dev_name))
                times.append(time.perf_counter() - t0)
        anny_rows.append({"device": dev_name, "dtype": str(dt_).replace("torch.", ""), "frames": n,
                          "seconds_min_of_3": round(min(times), 4)})
    m.to(device="cpu", dtype=F64)
    return {"clip": f"{CLIP_S:g} s at {FPS:g} fps", "adapter": rows,
            "float32_vs_float64_max_abs_m": {"joints": float((J64 - J32.double()).abs().max()),
                                             "vertices": float((V64 - V32.double()).abs().max())},
            "anny_forward_local_ref_batches_of_128": anny_rows,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads()}


def grad_facts(ob):
    torch.manual_seed(1)
    aa0 = 0.3 * torch.randn(2, 52, 3, dtype=F64)
    tr = torch.zeros(2, 3, dtype=F64)
    sub = torch.arange(0, ob.rest_vertices.shape[0], 997)

    def joints(aa):
        R = axis_angle_to_matrix(aa)
        return ob.pose(R[:, :22], tr, R[:, 22:], vertices=False)[0]

    def verts(aa):
        R = axis_angle_to_matrix(aa)
        return ob.pose(R[:, :22], tr, R[:, 22:])[1][:, sub]
    ok_j = torch.autograd.gradcheck(joints, (aa0.clone().requires_grad_(True),), eps=1e-6, atol=1e-6)
    ok_v = torch.autograd.gradcheck(verts, (aa0[:1].clone().requires_grad_(True),), eps=1e-6, atol=1e-6)
    return {"gradcheck_joints_wrt_axis_angle": bool(ok_j), "gradcheck_vertices_wrt_axis_angle": bool(ok_v),
            "vertices_checked": int(sub.numel()),
            "note": "the posing is torch end to end (no in-place step on the graph): inverse kinematics can run by "
                    "autograd on the joints and on chosen vertices; the rest model (phenotypes) is evaluated without "
                    "gradient in openbody.rest_state"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    m = OB.model()
    ob = OB.OpenBody(OB.NEUTRAL_ADULT, m)
    import anny
    facts = {"script": "scripts/openbody_facts.py", "run_at": dt.datetime.now().isoformat(timespec="seconds"),
             "anny_version": anny.__version__, "body": "openbody.NEUTRAL_ADULT " + json.dumps(OB.NEUTRAL_ADULT.as_dict()),
             "frames": {"anny": "z up, the subject's left +x, the face toward -y (m)",
                        "program": "y up, the subject's left +x, the face toward +z (m); ours = TO_OURS anny",
                        "TO_OURS": OB.TO_OURS}}
    facts["rig"] = rig_facts(m, ob)
    facts["mesh"] = mesh_facts(m, ob)
    facts["rest"] = rest_facts(ob)
    print("rig, mesh, rest done")
    facts["phenotypes"] = phenotype_facts(m)
    print("phenotypes done")
    facts["identities"] = identity_facts(m)
    print("identities done")
    facts["speed"] = speed_facts(ob)
    print("speed done")
    facts["differentiability"] = grad_facts(ob)
    sets = OB.vertex_sets(ob)
    facts["vertex_sets"] = {k: int(len(v)) for k, v in sets.items()}
    facts["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "anny_facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1)
    with open(os.path.join(OUT_DIR, "vertex_sets.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"script": "scripts/openbody_facts.py", "body": facts["body"], "topology": "anny",
                   "rules": OB.vertex_sets.__doc__.split("\n\n")[0].replace("\n", " "),
                   "sets": {k: v.tolist() for k, v in sets.items()}}, f, indent=0)
    print(json.dumps({"rest": facts["rest"]["per_side"], "speed": facts["speed"]["adapter"],
                      "grad": facts["differentiability"]}, indent=1))
    print("wrote reports/openbody/anny_facts.json and vertex_sets.json in", facts["seconds"], "s")


if __name__ == "__main__":
    main()
