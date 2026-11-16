"""First clip-by-clip comparison of the open body with the SMPL+H body (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md
section 8): the same program rotations posed on both bodies, on a handful of clips.

Clips (CLIPS): a seated level 0, a standing level 0, the level-0 walk, a reach with kinetic tremor, a dystonic
posture with the hands (the writer's cramp), since wave 3 a seated chorea and the seated touching tics, since wave 4
the geste antagoniste (the hand on the face), since wave 5 the parkinsonian gait and the freezing by shuffling, and
since wave 6 (WAVE6_CLIPS) the arising from a chair, the 3.14 seated rest and the tandem stance, each built
by its catalogue module without the background layer
(module_specs, the bodies of the modules). Frames on the viewer's clock (30 fps, viewer.pose_at and hands_at).

Bodies: the SMPL+H neutral reference (mdfm/synth/smplh_reference.py, reached with allow_reference=True; arrays in
memory only) and the open body (openbody.OpenBody) of the comparison identity: Anny's neutral adult (gender 0.5, age
40 years) at the stature of the SMPL+H neutral rest mesh (the height value solved by bisection), every other value
0.5. The translation is retargeted as exam.retarget_neutral does for the AMASS references: the pelvis path scaled by
the ratio of the rest pelvis heights above the sole (k), then the open body shifted so that the floors of the two
meshes coincide (each floor the median over frames of the lowest vertex, viewer.scene_floor's rule).

Measures, per clip:
- joint paths after a bone-length normalisation: the open body's posed skeleton rebuilt from its bone directions with
  the SMPL+H bone lengths from the SMPL+H pelvis path; per joint the distance to the SMPL+H joint (RMS and max over
  frames), per bone the angle between the two bodies' bone directions (mean and max), so that what differs is the
  direction of the rest segments only;
- motion paths: each joint's path relative to the pelvis, centred on its clip mean (the constant rest offsets out),
  the open body's scaled to the SMPL+H stature; the RMS distance of the two paths, and the ratio of their RMS
  amplitudes (open over SMPL+H), the read-back of a displacement amplitude;
- foot contact: the body-store contact rule (frames.contact_labels at 30 fps on each body's joints, its lowest foot
  joint over the clip at 0) and the agreement of the labels; the skate of each body (body.skate on its own labels);
  the sole cloud's height above each mesh floor on the frames both bodies call a contact;
- palm distances: the least distance from each palm's skin to the same side's thigh skin per frame (the SMPL+H sets
  of involuntary.palm_vertices and of the hip-dominated vertices; the open body's of openbody.vertex_sets).
Static: the rest skeletons compared (each joint's offset from the pelvis over the stature, the bone length ratios,
the spine's heights over the pelvis-to-neck height that chose the joint map's spine bones).

Output: reports/openbody/first_comparison.json (numbers only: no array of either body is written); --out names
another file under reports/openbody/ (wave 2's rerun on a committed tree: comparison_wave2.json; wave 3's:
comparison_wave3.json; wave 4's: comparison_wave4.json; wave 5's: comparison_wave5.json; wave 6's:
comparison_wave6.json).

Every clip (wave 7, section 8's extension): scripts/openbody_limits_catalogue.py --build --bodies stores each clip's
raw body (its program rotations, translation and contact, float32; no mesh, no shape, no model array) beside its
joint-limit components; --from-cache <cache> --parts n --part i poses one part of those clips on both bodies, in
blocks of BLOCK frames, and writes one compact line per clip into <cache>/compare_<i>of<n>.jsonl (the groups, the
bone directions' largest spread and largest constant angles, the contact, the palm gaps, the lowest vertex;
resumable); --merge <cache> --out comparison_wave7.json writes the report and its .md: the per-clip lines and the
summary over every clip (the largest spread against SPREAD_BOUND_DEG, the distributions over clips of the joint-path
distances, of the motion amplitude ratios of the groups that move at least MOVING_CM, of the label agreement, of the
skate and of the palm gaps).

    PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/openbody_compare.py [--out NAME]
    ... scripts/openbody_compare.py --from-cache <store>/openbody_limits_cache_wave7 --parts 4 --part 0
    ... scripts/openbody_compare.py --merge <store>/openbody_limits_cache_wave7 --out comparison_wave7.json
"""
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "first_comparison.json")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdfm import paths  # noqa: E402
from mdfm.body import frames  # noqa: E402
from mdfm.synth import openbody as OB  # noqa: E402

CLIPS = (("jerks", "jerk_E1_level0", "seated level 0"),
         ("jerks", "jerk_E7_level0", "standing level 0"),
         ("gait", "gait_E8_level0", "walk level 0"),
         ("tremors", "tremor_E3_kinetic_g2", "reach with kinetic tremor, grade 2"),
         ("postures", "post_W_writers_cramp_g2", "dystonic posture with the hands, writer's cramp grade 2"),
         ("involuntary", "invol_chorea_E1_g2", "seated generalized chorea, grade 2 (wave 3)"),
         ("jerks", "jerk_tics_E1_touch_c2", "seated touching tics (wave 3)"),
         ("postures", "post_E7_cd_geste_C1", "cervical dystonia with the geste antagoniste, TWSTRS C 1 (wave 4)"),
         ("gait", "gait_E8_parkinsonian_g2", "parkinsonian gait, MDS-UPDRS 3.10 grade 2 (wave 5)"),
         ("gait", "gait_E8_fog_shuffling_g3", "freezing of gait by shuffling forward, MDS-UPDRS 3.11 grade 3 (wave 5)"))
FPS = 30.0
GROUPS = {"trunk_head": (0, 3, 6, 9, 12, 15), "arms": (13, 14, 16, 17, 18, 19, 20, 21), "legs": (1, 2, 4, 5, 7, 8, 10, 11),
          "hands": tuple(range(22, 52))}
FOOT_JOINTS = (7, 8, 10, 11)


def git_head(root):
    import subprocess
    r = subprocess.run(["git", "-c", f"safe.directory={root}", "-C", root, "log", "-1", "--format=%h %ad %s",
                        "--date=short"], capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def spec_of(module, clip):
    from mdfm.synth import catalogue as C
    for s in C.module_specs(module, None):
        if s.clip_id == clip:
            return s
    raise KeyError(clip)


def normalized(Ja, Js):
    """Ja rebuilt from its bone directions with Js's bone lengths from Js's root (module docstring)."""
    out = torch.empty_like(Js)
    out[:, 0] = Js[:, 0]
    for c in range(1, Js.shape[1]):
        p = OB.PARENTS52[c]
        d = Ja[:, c] - Ja[:, p]
        L = (Js[:, c] - Js[:, p]).norm(dim=-1, keepdim=True)
        out[:, c] = out[:, p] + L * d / d.norm(dim=-1, keepdim=True)
    return out


def bone_angles(Ja, Js):
    a = torch.stack([Ja[:, c] - Ja[:, OB.PARENTS52[c]] for c in range(1, Js.shape[1])], 1)
    b = torch.stack([Js[:, c] - Js[:, OB.PARENTS52[c]] for c in range(1, Js.shape[1])], 1)
    cos = (a * b).sum(-1) / a.norm(dim=-1) / b.norm(dim=-1)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))           # (N, 51)


def min_dist(V, a_ids, b_ids, chunk=128):
    out = []
    for s in range(0, V.shape[0], chunk):
        A, B = V[s:s + chunk, a_ids], V[s:s + chunk, b_ids]
        out.append(torch.cdist(A, B).amin(dim=(1, 2)))
    return torch.cat(out)


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}


def foot_block(J):
    J = J.clone()
    J[..., 1] -= J[:, list(FOOT_JOINTS), 1].amin()
    lab = frames.contact_labels(J[:, :22], FPS, floor=0.0)
    from mdfm.synth.body import skate
    sk = skate(J[:, :22], lab, FPS)
    return lab, sk


BLOCK = 240               # frames posed at once (a 60 s clip's vertices are never held at once)
SPREAD_BOUND_DEG = 0.01   # section 10: the bone-direction spread of a replayed clip (the adapter adds nothing)
MOVING_CM = 0.5           # [module] a group's motion amplitude ratio is read where its largest joint moves this much
#                           (RMS about its clip mean, on SMPL+H): a still group's ratio is a ratio of rounding


WAVE6_CLIPS = (("parkinsonism", "park_E7_arise_g0", "arising from a chair, MDS-UPDRS 3.9 level 0 (wave 6)"),
               ("parkinsonism", "park_E1_spontaneity_g0", "seated rest of 3.14, the hands on the thighs (wave 6)"),
               ("ataxia", "atax_SARA2_tandem_g0", "tandem stance, SARA 2 level 0 (wave 6)"))
#   wave 6: the arising with the folded arms, the seated hands on the thighs, an ataxic stance on its own legs


def context():
    """The comparison's bodies and their sets (module docstring): the SMPL+H reference, the open body of the
    comparison identity, the translation scale k, the rest comparison and the mesh sets of both bodies."""
    from mdfm.synth import viewer as V
    from mdfm.synth import contact as CT
    from mdfm.synth import involuntary as INV
    ref = OB.backend(OB.SMPLH_REFERENCE, allow_reference=True)
    m = OB.model()
    # the comparison identity: Anny's neutral adult at the stature of the SMPL+H neutral rest mesh
    vs_rest = ref.model.v_template.numpy()
    stature_s = float(vs_rest[:, 1].max() - vs_rest[:, 1].min())
    base = OB.NEUTRAL_ADULT.as_dict()
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        h = OB.anthropometry(OB.Phenotype(**{**base, "height": mid}), m)["height_m"]
        lo, hi = (mid, hi) if h < stature_s else (lo, mid)
    phen = OB.Phenotype(**{**base, "height": 0.5 * (lo + hi)})
    ob = OB.OpenBody(phen, m)
    rest_a, rest_s = ob.rest_joints52(), ref.rest_joints52()
    va_can = ob.canonical_vertices()
    sole_a_rest, sole_s_rest = float(va_can[:, 1].min()), float(vs_rest[:, 1].min())
    k = float((rest_a[0, 1] - sole_a_rest) / (rest_s[0, 1] - sole_s_rest))
    # static rest comparison
    st_a = float(va_can[:, 1].max() - va_can[:, 1].min())
    off_a = (rest_a - rest_a[0]) / st_a
    off_s = (rest_s - rest_s[0]) / stature_s
    rest_cmp = {n: round(float((off_a[j] - off_s[j]).norm()) * 100, 2) for j, n in enumerate(OB.JOINT_NAMES)}
    bone_ratio = {OB.JOINT_NAMES[c]: round(float((rest_a[c] - rest_a[OB.PARENTS52[c]]).norm() /
                                                 (rest_s[c] - rest_s[OB.PARENTS52[c]]).norm()), 3)
                  for c in range(1, OB.N_JOINTS)}
    spine = {}
    for name, r in (("open", rest_a), ("smplh", rest_s)):
        h0, h1 = float(r[0, 1]), float(r[12, 1])
        spine[name] = {OB.JOINT_NAMES[j]: round((float(r[j, 1]) - h0) / (h1 - h0), 3) for j in (3, 6, 9, 12, 15)}
    # the joint map's spine rule: for spine1 and spine2, the Anny spine bone whose head height over the pelvis-to-neck
    # height lies nearest the old joint's (spine3 = spine01 and neck = neck01 are fixed by the tree)
    a_idx = OB.bone_index(m)
    from mdfm.synth.openbody import rest_state
    _, heads_a, _ = rest_state(phen, m)
    anny_spine = {b: round((float(heads_a[a_idx[b], 2]) - float(heads_a[a_idx["root"], 2])) /
                           (float(heads_a[a_idx["neck01"], 2]) - float(heads_a[a_idx["root"], 2])), 3)
                  for b in ("spine05", "spine04", "spine03", "spine02", "spine01", "neck01", "neck02", "neck03", "head")}
    spine_rule = {j: min(("spine04", "spine03", "spine02"), key=lambda b: abs(anny_spine[b] - spine["smplh"][j]))
                  for j in ("spine1", "spine2")}
    spine_rule["map"] = {j: dict(OB.JOINT_MAP)[j] for j in ("spine1", "spine2", "spine3", "neck")}
    # mesh sets
    sets_a = OB.vertex_sets(ob)
    # the SMPL+H sets by the same rules on the reference's rest mesh: since wave 1 the helpers read the selected body
    # model, so the reference is selected for them here (behind its flag) and the selection restored after
    saved = {key: os.environ.get(key) for key in (OB.BACKEND_ENV, OB.REFERENCE_ENV)}
    os.environ[OB.BACKEND_ENV], os.environ[OB.REFERENCE_ENV] = OB.SMPLH_REFERENCE, "1"
    try:
        dom_s = V.dominant_joint("neutral")
        palm_s = {s: INV.palm_vertices()[s][0] for s in ("left", "right")}
        thigh_s = {"left": np.flatnonzero(dom_s == 1), "right": np.flatnonzero(dom_s == 2)}
        sole_s = CT.sole_vertices(2)
    finally:
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    if max(int(np.max(x)) for x in list(palm_s.values()) + list(thigh_s.values()) + list(sole_s.values())) >= 6890:
        raise RuntimeError("an SMPL+H vertex set holds an index past the SMPL+H mesh")
    rest = {"offset_from_pelvis_over_stature_difference_cm_per_m": rest_cmp,
            "bone_length_ratio_open_over_smplh": bone_ratio,
            "spine_heights_over_pelvis_to_neck": spine, "anny_spine_bone_heads_over_root_to_neck01": anny_spine,
            "spine_rule_nearest": spine_rule}
    return {"ref": ref, "ob": ob, "phen": phen, "k": k, "rest_a": rest_a, "rest_s": rest_s, "st_a": st_a,
            "stature_s": stature_s, "sets_a": sets_a, "palm_s": palm_s, "thigh_s": thigh_s, "sole_s": sole_s,
            "rest": rest}


def _frames(model, R, tr, Rh, soles, palms, thighs, block=BLOCK):
    """Joints (N, 52, 3) and, per frame, the lowest vertex's height, each sole's lowest height and each palm's least
    distance to the same side's thigh, posed in blocks of frames (the vertices of a long clip never held at once)."""
    J, low = [], []
    sole = {s: [] for s in ("left", "right")}
    palm = {s: [] for s in ("left", "right")}
    with torch.no_grad():
        for a in range(0, R.shape[0], block):
            sl = slice(a, a + block)
            Jb, Vb = model.pose(R[sl], tr[sl], None if Rh is None else Rh[sl])
            J.append(Jb)
            low.append(Vb[..., 1].amin(1))
            for s in ("left", "right"):
                sole[s].append(Vb[:, soles[s], 1].amin(1))
                palm[s].append(min_dist(Vb, palms[s], thighs[s]))
    return (torch.cat(J), torch.cat(low), {s: torch.cat(v) for s, v in sole.items()},
            {s: torch.cat(v) for s, v in palm.items()})


def measure(body, C, detail=True):
    """The comparison of one body (module docstring, measures per clip); detail False keeps the groups, the bone
    directions' largest spread and their largest constant angles, without the per-joint tables."""
    from mdfm.synth import viewer as V
    ref, ob, k, rest_a, rest_s = C["ref"], C["ob"], C["k"], C["rest_a"], C["rest_s"]
    sets_a = C["sets_a"]
    t = V.frame_times(body.duration, FPS)
    R, tr = V.pose_at(body, t)
    Rh = V.hands_at(body, t)
    Js, low_s, sole_s, palm_s = _frames(ref, R, tr, Rh, C["sole_s"], C["palm_s"], C["thigh_s"])
    pel_s = tr + rest_s[0]
    floor_s = float(np.median(low_s.numpy()))
    p = pel_s.clone()
    p[:, 1] -= floor_s
    tr_a = k * p - rest_a[0]
    Ja, low_a, sole_a, palm_a = _frames(ob, R, tr_a, Rh, {s: sets_a[f"sole_{s}"] for s in ("left", "right")},
                                        {s: sets_a[f"palm_{s}"] for s in ("left", "right")},
                                        {s: sets_a[f"thigh_{s}"] for s in ("left", "right")})
    floor_a = float(np.median(low_a.numpy()))
    dy = floor_s - floor_a
    Ja[..., 1] += dy
    low_a = low_a + dy
    sole_a = {s: v + dy for s, v in sole_a.items()}
    Jn = normalized(Ja, Js)
    d = (Jn - Js).norm(dim=-1) * 100.0                               # cm
    ang = bone_angles(Ja, Js)
    per_joint = {OB.JOINT_NAMES[j]: {"rms_cm": round(float(d[:, j].pow(2).mean().sqrt()), 3),
                                     "max_cm": round(float(d[:, j].max()), 3)} for j in range(1, OB.N_JOINTS)}
    groups = {g: {"rms_cm_median_over_joints": round(float(np.median([per_joint[OB.JOINT_NAMES[j]]["rms_cm"]
                                                                      for j in js if j])), 3),
                  "max_cm": round(max(per_joint[OB.JOINT_NAMES[j]]["max_cm"] for j in js if j), 3)}
              for g, js in GROUPS.items()}
    bones = {OB.JOINT_NAMES[c]: {"mean_deg": round(float(ang[:, c - 1].mean()), 2),
                                 "max_deg": round(float(ang[:, c - 1].max()), 2),
                                 "spread_deg": round(float(ang[:, c - 1].max() - ang[:, c - 1].min()), 4)}
             for c in range(1, OB.N_JOINTS)}
    # motion paths: each joint relative to the pelvis, centred on its clip mean (the constant rest offsets out),
    # the open body's scaled to the SMPL+H stature; the RMS distance of the two paths and the ratio of their RMS
    # amplitudes (open over SMPL+H), the read-back of an amplitude
    ra, rs = Ja - Ja[:, :1], Js - Js[:, :1]
    ca = (ra - ra.mean(0)) * (C["stature_s"] / C["st_a"])
    cs = rs - rs.mean(0)
    dm = (ca - cs).norm(dim=-1).pow(2).mean(0).sqrt() * 100.0             # (52,) cm
    amp_a, amp_s = ca.norm(dim=-1).pow(2).mean(0).sqrt(), cs.norm(dim=-1).pow(2).mean(0).sqrt()
    motion = {OB.JOINT_NAMES[j]: {"rms_cm": round(float(dm[j]), 3),
                                  "amplitude_smplh_cm": round(float(amp_s[j]) * 100, 3),
                                  "amplitude_ratio": round(float(amp_a[j] / amp_s[j]), 4)
                                  if float(amp_s[j]) > 1e-4 else None} for j in range(1, OB.N_JOINTS)}
    motion_groups = {g: {"rms_cm_median_over_joints": round(float(np.median([float(dm[j]) for j in js if j])), 3),
                         "rms_cm_max": round(max(float(dm[j]) for j in js if j), 3),
                         "amplitude_ratio_median": round(float(np.median(
                             [motion[OB.JOINT_NAMES[j]]["amplitude_ratio"] for j in js
                              if j and motion[OB.JOINT_NAMES[j]]["amplitude_ratio"] is not None] or [np.nan])), 4),
                         "amplitude_smplh_cm_max": round(max(motion[OB.JOINT_NAMES[j]]["amplitude_smplh_cm"]
                                                             for j in js if j), 3)}
                     for g, js in GROUPS.items()}
    # contact
    lab_s, sk_s = foot_block(Js)
    lab_a, sk_a = foot_block(Ja)
    both = lab_s & lab_a
    foot = {"contact_share_smplh": [float(x) for x in lab_s.double().mean(0)],
            "contact_share_open": [float(x) for x in lab_a.double().mean(0)],
            "label_agreement": float((lab_s == lab_a).double().mean()),
            "skate_p95_m_s": {"smplh": sk_s["p95"], "open": sk_a["p95"]}}
    gaps = {}
    for i, side in enumerate(("left", "right")):
        sel = both[:, i]
        if int(sel.sum()) == 0:
            gaps[side] = None
            continue
        gs = sole_s[side][sel] - floor_s
        ga = sole_a[side][sel] - floor_s
        gaps[side] = {"frames": int(sel.sum()), "smplh_sole_above_floor_cm": summary(gs.numpy() * 100),
                      "open_sole_above_floor_cm": summary(ga.numpy() * 100)}
    foot["sole_gap_on_shared_contact"] = gaps
    palms = {}
    for side in ("left", "right"):
        ds, da = palm_s[side] * 100, palm_a[side] * 100
        palms[side] = {"smplh_cm": summary(ds.numpy()), "open_cm": summary(da.numpy()),
                       "abs_difference_cm": summary((da - ds).abs().numpy())}
    lows = {"smplh": summary((low_s - floor_s).numpy() * 100), "open": summary((low_a - floor_s).numpy() * 100)}
    out = {"frames": int(len(t)), "duration_s": float(body.duration), "hands_animated": Rh is not None,
           "foot": foot, "palm_to_thigh": palms, "lowest_vertex_above_floor_cm": lows, "vertical_shift_m": dy}
    if detail:
        out.update(joint_paths_normalized={"groups": groups, "per_joint": per_joint},
                   motion_paths={"groups": motion_groups, "per_joint": motion}, bone_direction_angle=bones)
    else:
        spread = max(bones.items(), key=lambda kv: kv[1]["spread_deg"])
        top = sorted(bones.items(), key=lambda kv: -kv[1]["mean_deg"])[:3]
        out.update(joint_paths_normalized={"groups": groups}, motion_paths={"groups": motion_groups},
                   bone_direction={"max_spread_deg": spread[1]["spread_deg"], "max_spread_bone": spread[0],
                                   "largest_mean_deg": [[n, b["mean_deg"]] for n, b in top]})
    return out


def body_from_cache(arrs):
    """The Body (a HandBody when the cache holds hand rotations) of a clip stored by
    scripts/openbody_limits_catalogue.py --build --bodies (the program's rotations, translation and contact; float32)."""
    from mdfm.synth.body import Body
    from mdfm.synth.hands import hands_of
    b = Body(torch.from_numpy(arrs["rot6d"].astype(np.float64)), torch.from_numpy(arrs["transl"].astype(np.float64)),
             torch.from_numpy(arrs["contact"].astype(bool)), rate=float(arrs["rate"]))
    if "hand6d" in arrs.files:
        b = hands_of(b, torch.from_numpy(arrs["hand6d"].astype(np.float64)))
    return b


def run_cache(a):
    """--from-cache: every clip of the catalogue measure's cache that holds its body (module docstring), one part of
    --parts, one JSON line per clip in <cache>/compare_<part>of<parts>.jsonl (resumable)."""
    import glob
    torch.set_num_threads(a.threads)
    C = context()
    files = sorted(glob.glob(os.path.join(a.from_cache, "*.npz")))
    mine = [f for i, f in enumerate(files) if i % a.parts == a.part]
    log = os.path.join(a.from_cache, f"compare_{a.part}of{a.parts}.jsonl")
    done = set()
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            done = {json.loads(line)["clip"] for line in f if line.strip()}
    for f in mine:
        cid = os.path.basename(f)[:-4]
        if cid in done:
            continue
        tc = time.time()
        arrs = np.load(f)
        if "rot6d" not in arrs.files:
            rec = {"clip": cid, "not_measured": "no body in the cache (built without --bodies)"}
        else:
            try:
                rec = dict({"clip": cid}, **measure(body_from_cache(arrs), C, detail=False))
            except Exception as e:  # noqa: BLE001
                rec = {"clip": cid, "error": f"{type(e).__name__}: {e}"[:300]}
        rec["seconds"] = round(time.time() - tc, 1)
        with open(log, "a", encoding="utf-8", newline="\n") as fo:
            fo.write(json.dumps(rec) + "\n")
        print(f"[{a.part}/{a.parts}] {cid} {rec['seconds']}", flush=True)


def merge(a, out_path):
    """--merge: the parts' lines into one report (out_path, and its .md beside it): per clip the compact record, and
    the summary over every clip (module docstring)."""
    import glob
    recs = {}
    for p in sorted(glob.glob(os.path.join(a.merge, "compare_*of*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["clip"]] = r
    meta = []
    for p in sorted(glob.glob(os.path.join(a.merge, "meta_*.json"))):
        with open(p, encoding="utf-8") as f:
            meta.append(json.load(f))
    ok = {c: r for c, r in recs.items() if "frames" in r}
    bad = {c: r.get("error") or r.get("not_measured") for c, r in recs.items() if c not in ok}
    spread = max(ok.items(), key=lambda kv: kv[1]["bone_direction"]["max_spread_deg"]) if ok else None
    over = sorted(c for c, r in ok.items() if r["bone_direction"]["max_spread_deg"] > SPREAD_BOUND_DEG)

    def dist(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
        if not len(v):
            return None
        return {"n": int(len(v)), "min": float(v.min()), "p5": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    moving = {g: [r["motion_paths"]["groups"][g]["amplitude_ratio_median"] for r in ok.values()
                  if r["motion_paths"]["groups"][g]["amplitude_smplh_cm_max"] >= MOVING_CM] for g in GROUPS}
    summ = {
        "n_clips": len(recs), "n_measured": len(ok), "not_measured": bad,
        "bone_direction_spread": {"bound_deg": SPREAD_BOUND_DEG,
                                  "max_deg": spread[1]["bone_direction"]["max_spread_deg"] if spread else None,
                                  "max_clip": spread[0] if spread else None,
                                  "max_bone": spread[1]["bone_direction"]["max_spread_bone"] if spread else None,
                                  "clips_over_bound": over},
        "joint_paths_rms_cm_median_over_joints": {g: dist(r["joint_paths_normalized"]["groups"][g][
            "rms_cm_median_over_joints"] for r in ok.values()) for g in GROUPS},
        "motion_amplitude_ratio_median": {"rule": f"the clips whose group moves at least {MOVING_CM} cm (RMS, the "
                                                  "SMPL+H amplitude of its largest joint)",
                                          **{g: dist(v) for g, v in moving.items()}},
        "label_agreement": dist(r["foot"]["label_agreement"] for r in ok.values()),
        "skate_p95_m_s": {"smplh": dist(r["foot"]["skate_p95_m_s"]["smplh"] for r in ok.values()),
                          "open": dist(r["foot"]["skate_p95_m_s"]["open"] for r in ok.values())},
        "palm_to_thigh_median_abs_difference_cm": dist(
            max(r["palm_to_thigh"][s]["abs_difference_cm"]["median"] for s in ("left", "right")) for r in ok.values()),
        "lowest_vertex_open_min_cm": dist(r["lowest_vertex_above_floor_cm"]["open"]["min"] for r in ok.values()),
        "seconds": sum(r.get("seconds", 0.0) for r in recs.values())}
    out = {"script": "scripts/openbody_compare.py --merge", "run_at": dt.datetime.now().isoformat(timespec="seconds"),
           "md_fm_head": git_head(ROOT), "moveatlas_head": git_head(paths.MOVEATLAS),
           "bodies_built": meta, "fps": FPS, "summary": summ, "clips": dict(sorted(ok.items()))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    s = summ
    lines = ["# The open body against the SMPL+H reference on every clip (open-body port, wave 7)", "",
             f"scripts/openbody_compare.py --from-cache and --merge, {out['run_at']}; the bodies of the catalogue "
             f"measure's cache (scripts/openbody_limits_catalogue.py --build --bodies"
             + (f", MD_FM {meta[0].get('mdfm_commit')}" if meta else "") + "), raw bodies (no background layer), the "
             "same program rotations posed on both bodies at 30 fps (section 8 of docs/OPEN_BODY_PORT_DESIGN.md).", "",
             f"Clips: {s['n_clips']}, measured {s['n_measured']}, not measured {len(s['not_measured'])}.", "",
             f"- bone-direction spread (the adapter adds nothing when it stays under {SPREAD_BOUND_DEG} deg on a "
             f"replayed clip): largest {s['bone_direction_spread']['max_deg']} deg "
             f"({s['bone_direction_spread']['max_clip']}, "
             f"{s['bone_direction_spread']['max_bone']}); clips over the bound: "
             f"{len(s['bone_direction_spread']['clips_over_bound'])}"]
    for g in GROUPS:
        dj = s["joint_paths_rms_cm_median_over_joints"][g]
        dm = s["motion_amplitude_ratio_median"][g]
        lines.append(f"- {g}: joint paths after the bone-length normalisation, the median over joints of the RMS "
                     f"distance, over clips median {dj['median']:.2f} cm (max {dj['max']:.2f}); motion amplitude ratio "
                     f"(open over SMPL+H) " + ("no clip moves it" if not dm else
                                               f"median {dm['median']:.3f} (5th to 95th percentile {dm['p5']:.3f} to "
                                               f"{dm['p95']:.3f}, {dm['n']} clips)"))
    la = s["label_agreement"]
    lines += [f"- foot contact labels agree on a median {la['median']:.3f} of the frames (least {la['min']:.3f}); "
              f"skate "
              f"95th percentile, median over clips: SMPL+H {s['skate_p95_m_s']['smplh']['median']:.3f}, open "
              f"{s['skate_p95_m_s']['open']['median']:.3f} m/s",
              f"- palm to thigh, the larger side's median absolute difference: median over clips "
              f"{s['palm_to_thigh_median_abs_difference_cm']['median']:.2f} cm (max "
              f"{s['palm_to_thigh_median_abs_difference_cm']['max']:.2f})", ""]
    if s["not_measured"]:
        lines += ["## Not measured", ""] + [f"- {c}: {w}" for c, w in sorted(s["not_measured"].items())] + [""]
    if s["bone_direction_spread"]["clips_over_bound"]:
        lines += ["## Bone-direction spread over the bound", ""] + [
            f"- {c}: {ok[c]['bone_direction']['max_spread_deg']} deg ({ok[c]['bone_direction']['max_spread_bone']})"
            for c in s["bone_direction_spread"]["clips_over_bound"]] + [""]
    with open(out_path[:-5] + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(out_path, ROOT), "and its .md:", s["n_measured"], "clips measured")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="the open body against the SMPL+H reference")
    ap.add_argument("--out", default=os.path.basename(OUT), help="the report's file name under reports/openbody/")
    ap.add_argument("--from-cache", default=None, help="measure every clip whose body the catalogue cache holds")
    ap.add_argument("--merge", default=None, help="merge the parts' lines of this cache into --out")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out_path = os.path.join(os.path.dirname(OUT), os.path.basename(a.out))
    sys.stdout.reconfigure(encoding="utf-8")
    if a.from_cache:
        return run_cache(a)
    if a.merge:
        return merge(a, out_path)
    t0 = time.time()
    C = context()
    clips = []
    for module, clip, what in CLIPS + WAVE6_CLIPS:
        tc = time.time()
        try:
            body, truth = spec_of(module, clip).build()
        except Exception as e:  # noqa: BLE001  (a module of a later wave not yet ported to the open body)
            clips.append({"clip": clip, "module": module, "what": what, "not_built": f"{type(e).__name__}: {e}"[:300]})
            print(clip, "not built:", type(e).__name__)
            continue
        rec = dict({"clip": clip, "module": module, "what": what}, **measure(body, C, detail=True),
                   seconds=round(time.time() - tc, 1))
        clips.append(rec)
        groups, mg = rec["joint_paths_normalized"]["groups"], rec["motion_paths"]["groups"]
        print(clip, rec["frames"], "frames", {g: v["rms_cm_median_over_joints"] for g, v in groups.items()},
              "motion", {g: (v["rms_cm_median_over_joints"], v["amplitude_ratio_median"]) for g, v in mg.items()},
              "agree", round(rec["foot"]["label_agreement"], 3), "s", rec["seconds"])
    out = {"script": "scripts/openbody_compare.py", "run_at": dt.datetime.now().isoformat(timespec="seconds"),
           "md_fm_head": git_head(ROOT), "moveatlas_head": git_head(paths.MOVEATLAS),
           "moveatlas_use": "the SMPL+H neutral model file read in memory (body_models/smplh), as every synth run",
           "open_body": {"phenotype": C["phen"].as_dict(), "stature_m": C["st_a"], "anny": "0.6.2"},
           "smplh": {"stature_m": C["stature_s"], "shape": "neutral, betas 0"},
           "translation_scale_k": C["k"], "fps": FPS, "rest": C["rest"], "clips": clips,
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.relpath(out_path, ROOT), "in", out["seconds"], "s")


if __name__ == "__main__":
    main()
