"""The anatomical joint-limit table of the synthetic patient on the open body (plan phase C, wave 1;
docs/OPEN_BODY_PORT_DESIGN.md section 5, decision D13): every bound typed once here with its source key, mapped onto
the axis-angle components of the program's local joint rotations, written to reports/openbody/joint_limits.json,
the table body.joint_limits reads when the open body is selected. The AMASS percentiles of
reports/body_store/priors.json stay the table of the model track (mdfm/train) and of the SMPL+H reference.

Sources:
- jarm2022: the measurement table of the Japanese Association of Rehabilitation Medicine, the Japanese Orthopaedic
  Association and the Japanese Society for Surgery of the Foot (revision of April 2022; the document read is
  jarm.or.jp member/document/kadou/03.pdf, read on 2026-11-12, its SHA-256 in JARM_READ). Its reference ranges
  (sanko kadoiki kakudo) are typed in JARM below, every row of the body, the hand and the foot.
- bk2024 and boike2025 (grades.SOURCES): the finger and thumb flexion and extension of synth/hands.py
  (FLEX_LIMITS_DEG), kept as the hand table; jarm2022's finger rows are recorded beside them.

Mapping (the program's axes, synth/base.py: x the subject's left, y up, z forward, local rotations about the
parent's axes from the canonical T pose, arms horizontal, palms down, thumbs forward, legs vertical): a component of
the table is the axis-angle vector of a local rotation in degrees. Anatomical signs (base.py and the operators):
hip flexion -x, abduction of the left hip +z, external rotation of the left hip +y; knee flexion +x; ankle
dorsiflexion -x, inversion of the left foot -z, abduction of the left foot +y; toe flexion +x; trunk and neck
flexion +x, axial rotation to the left +y, lateral bending to the right +z; collar elevation of the left +z and
protraction (girdle flexion) of the left -y; shoulder abduction of the left arm +z, horizontal flexion of the left
-y, internal rotation +x (both sides, the forearm forward turning down); elbow flexion of the left -y; supination
-x (both sides, body.SUPINATION_SIGN about the bone); wrist palmar flexion of the left -z, radial deviation of the
left -y. The right side mirrors: a component about y or z changes sign, about x keeps it.

Reference positions: jarm2022 measures the shoulder's abduction and flexion from the hanging arm and its horizontal
flexion, extension and its rotations at 90 deg of abduction (section VI). The canonical rest holds the arm at 90 deg
of abduction, so the horizontal and rotation rows read directly (the shoulder's x is the humerus' axial rotation in
the T pose: internal 70, external 90 deg at 90 deg abduction), and the abduction row is shifted by the 90 deg between
the two references (180 - 90 above the T pose, 0 + 90 below it, the hanging arm). The forearm's mid position (the
thumb up, jarm2022's 0) is the T pose's palm down with the thumb forward.

Shares [design] (the program places one anatomical range on several joints; each share is stated once):
- thoracolumbar (spine1, spine2, spine3): flexion and extension 0.5, 0.25, 0.25 (spine1 is the lumbar joint of the
  three and the lumbar spine carries most of the sagittal range; equal thirds would put the seated posture's 20 deg of
  lumbar flexion, base.POSTURES 'sit', outside 15 deg), rotation and lateral bending equal thirds;
- cervical (neck, head): halves;
- forearm pronation and supination (elbow and wrist x, base.POSTURES 'sit' turns 40 deg at each): halves;
- the toes (one program joint turns the five toes): the smaller of the first toe's and the lesser toes' range.
Unnamed components (no row of jarm2022 names them: knee y and z, elbow z, collar x, toe y and z): the larger of a
module bound (MODULE_DEG) and the largest value the joint's named motions produce in that component when composed
within their own bounds (CROSS_STEPS points per named axis, both orders, about the parent's fixed axes as
base.rotation composes them): the axis-angle reading of a combined rotation carries a cross term that no motion of the
joint names (the seated elbow, 40 deg of pronation then 90 of flexion, reads 31 deg about z).

Also written: the comparison with the AMASS table (every component where the anatomical bound is tighter or wider,
by how much), the hand table (per hand joint: flexion bounds, abduction bound, source) and the procedural bases of
synth/base.py read against the new table on the selected body model (base_bodies).
"""
import datetime
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "joint_limits.json")
PRIORS = os.path.join(ROOT, "reports", "body_store", "priors.json")

JARM_READ = {"url": "https://www.jarm.or.jp/member/document/kadou/03.pdf", "read": "2026-11-12",
             "sha256": "095201e71baf698c23053cbce51aff6e9f9a5709fe892105f3bba42f62077668",
             "title": "Kansetsu kadoiki hyoji narabini sokuteiho (2022 nen 4 gatsu kaitei), the measurement table: "
                      "II upper limb, III fingers, IV lower limb, V trunk, VI other tests"}
SOURCES = {
    "jarm2022": "Japanese Association of Rehabilitation Medicine, Japanese Orthopaedic Association and Japanese Society "
                "for Surgery of the Foot. Joint range of motion: display and measurement (Kansetsu kadoiki hyoji "
                "narabini sokuteiho), revision of April 2022, Jpn J Rehabil Med 2021; 58: 1188-1200, J Jpn Soc Surg "
                "Foot 2021; 42: S372-S385, J Jpn Orthop Assoc 2022; 96: 75-86 (the measurement table, "
                "jarm.or.jp member/document/kadou/03.pdf, read 2026-11-12): the reference ranges of every joint",
    "bk2024": "B.K. MI et al. 2024, Indian J Plast Surg 57(4): 248-255, doi 10.1055/s-0044-1788593 (abstract): active "
              "range of motion of the finger joints, 390 hands of healthy adults (grades.SOURCES)",
    "boike2025": "Boike S et al. 2025, J Hand Surg Glob Online 7: 100792, doi 10.1016/j.jhsg.2025.100792 (full text, "
                 "Table 2): active thumb MCP and IP flexion, 162 adults (grades.SOURCES)",
    "design": "a share of a regional range over the program joints that carry it (stated in this script)",
    "module": "a value of this script, not taken from a source (stated where used)",
}

# jarm2022, the reference ranges (deg), as read: section, motion -> degrees
JARM = {
    "shoulder_girdle": {"flexion": 20, "extension": 20, "elevation": 20, "depression": 10},
    "shoulder": {"forward_flexion": 180, "backward_extension": 50, "abduction": 180, "adduction": 0,
                 "external_rotation": 60, "internal_rotation": 80, "horizontal_flexion": 135,
                 "horizontal_extension": 30,
                 # section VI: rotations at 90 deg of abduction (elbow 90 deg), adduction at 20 or 45 deg of flexion
                 "external_rotation_abducted_90": 90, "internal_rotation_abducted_90": 70, "adduction_flexed": 75},
    "elbow": {"flexion": 145, "extension": 5},
    "forearm": {"pronation": 90, "supination": 90},
    "wrist": {"flexion": 90, "extension": 70, "radial_deviation": 25, "ulnar_deviation": 55},
    "thumb": {"radial_abduction": 60, "ulnar_adduction": 0, "palmar_abduction": 90, "palmar_adduction": 0,
              "mcp_flexion": 60, "mcp_extension": 10, "ip_flexion": 80, "ip_extension": 10},
    "finger": {"mcp_flexion": 90, "mcp_extension": 45, "pip_flexion": 100, "pip_extension": 0, "dip_flexion": 80,
               "dip_extension": 0},
    "hip": {"flexion": 125, "extension": 15, "abduction": 45, "adduction": 20, "external_rotation": 45,
            "internal_rotation": 45},
    "knee": {"flexion": 130, "extension": 0},
    "foot_and_ankle": {"abduction": 10, "adduction": 20, "dorsiflexion": 20, "plantar_flexion": 45, "inversion": 30,
                       "eversion": 20},
    "great_toe": {"mtp_flexion": 35, "mtp_extension": 60, "ip_flexion": 60, "ip_extension": 0},
    "lesser_toe": {"mtp_flexion": 35, "mtp_extension": 40, "pip_flexion": 35, "pip_extension": 0, "dip_flexion": 50,
                   "dip_extension": 0},
    "cervical": {"flexion": 60, "extension": 50, "rotation_left": 60, "rotation_right": 60,
                 "lateral_bending_left": 50, "lateral_bending_right": 50},
    "thoracolumbar": {"flexion": 45, "extension": 30, "rotation_left": 40, "rotation_right": 40,
                      "lateral_bending_left": 50, "lateral_bending_right": 50},
}
SHARES = {  # [design] (module docstring)
    "thoracolumbar_sagittal": {"spine1": 0.5, "spine2": 0.25, "spine3": 0.25},
    "thoracolumbar_rotation_bending": {"spine1": 1 / 3, "spine2": 1 / 3, "spine3": 1 / 3},
    "cervical": {"neck": 0.5, "head": 0.5},
    "forearm": {"elbow": 0.5, "wrist": 0.5},
}
MODULE_DEG = 10.0         # [module] the least bound of a component no row names (module docstring)
CROSS_STEPS = 13          # points per named axis of the cross-term grid
ARM_REFERENCE_SHIFT_DEG = 90.0   # the T pose's abduction over jarm2022's hanging arm

JOINTS = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
          "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")


def J(section, motion, share=1.0):
    return share * JARM[section][motion]


def rows():
    """[(joint, component, lo_deg, hi_deg, source, motion at lo, motion at hi, rule)] of the named components."""
    out = []

    def add(j, c, lo, hi, src, mlo, mhi, rule=""):
        out.append((j, c, float(lo) + 0.0, float(hi) + 0.0, src, mlo, mhi, rule))   # + 0.0: no negative zero
    for side, s in (("left", 1.0), ("right", -1.0)):
        hip, knee, ankle, foot = (f"{side}_{n}" for n in ("hip", "knee", "ankle", "foot"))
        add(hip, "x", -J("hip", "flexion"), J("hip", "extension"), "jarm2022", "flexion", "extension")
        er, ir = J("hip", "external_rotation"), J("hip", "internal_rotation")
        add(hip, "y", -(ir if s > 0 else er), (er if s > 0 else ir), "jarm2022",
            "internal rotation" if s > 0 else "external rotation", "external rotation" if s > 0 else "internal rotation")
        ab, ad = J("hip", "abduction"), J("hip", "adduction")
        add(hip, "z", -(ad if s > 0 else ab), (ab if s > 0 else ad), "jarm2022",
            "adduction" if s > 0 else "abduction", "abduction" if s > 0 else "adduction")
        add(knee, "x", -J("knee", "extension"), J("knee", "flexion"), "jarm2022", "extension", "flexion")
        add(ankle, "x", -J("foot_and_ankle", "dorsiflexion"), J("foot_and_ankle", "plantar_flexion"), "jarm2022",
            "dorsiflexion", "plantar flexion")
        fab, fad = J("foot_and_ankle", "abduction"), J("foot_and_ankle", "adduction")
        add(ankle, "y", -(fad if s > 0 else fab), (fab if s > 0 else fad), "jarm2022",
            "adduction" if s > 0 else "abduction", "abduction" if s > 0 else "adduction")
        inv, ev = J("foot_and_ankle", "inversion"), J("foot_and_ankle", "eversion")
        add(ankle, "z", -(inv if s > 0 else ev), (ev if s > 0 else inv), "jarm2022",
            "inversion" if s > 0 else "eversion", "eversion" if s > 0 else "inversion")
        ext = min(J("great_toe", "mtp_extension"), J("lesser_toe", "mtp_extension"))
        fl = min(J("great_toe", "mtp_flexion"), J("lesser_toe", "mtp_flexion"))
        add(foot, "x", -ext, fl, "jarm2022", "MTP extension", "MTP flexion",
            "one joint turns the five toes: the smaller of the first toe's and the lesser toes' range")
        collar, shoulder, elbow, wrist = (f"{side}_{n}" for n in ("collar", "shoulder", "elbow", "wrist"))
        add(collar, "y", -J("shoulder_girdle", "flexion" if s > 0 else "extension"),
            J("shoulder_girdle", "extension" if s > 0 else "flexion"), "jarm2022",
            "girdle flexion" if s > 0 else "girdle extension", "girdle extension" if s > 0 else "girdle flexion")
        el, de = J("shoulder_girdle", "elevation"), J("shoulder_girdle", "depression")
        add(collar, "z", -(de if s > 0 else el), (el if s > 0 else de), "jarm2022",
            "depression" if s > 0 else "elevation", "elevation" if s > 0 else "depression")
        add(shoulder, "x", -J("shoulder", "external_rotation_abducted_90"), J("shoulder", "internal_rotation_abducted_90"),
            "jarm2022", "external rotation at 90 deg abduction", "internal rotation at 90 deg abduction",
            "the humerus' axial rotation in the T pose (section VI rows)")
        hf, he = J("shoulder", "horizontal_flexion"), J("shoulder", "horizontal_extension")
        add(shoulder, "y", -(hf if s > 0 else he), (he if s > 0 else hf), "jarm2022",
            "horizontal flexion" if s > 0 else "horizontal extension",
            "horizontal extension" if s > 0 else "horizontal flexion", "measured from 90 deg of abduction, the T pose")
        up = J("shoulder", "abduction") - ARM_REFERENCE_SHIFT_DEG
        down = J("shoulder", "adduction") + ARM_REFERENCE_SHIFT_DEG
        add(shoulder, "z", -down, up, "jarm2022", "adduction to the hanging arm" if s > 0 else "abduction",
            "abduction" if s > 0 else "adduction to the hanging arm",
            "jarm2022 reads abduction from the hanging arm: shifted by the 90 deg of the T pose")
        add(elbow, "y", -J("elbow", "flexion" if s > 0 else "extension"), J("elbow", "extension" if s > 0 else "flexion"),
            "jarm2022", "flexion" if s > 0 else "extension", "extension" if s > 0 else "flexion")
        for j, k in ((elbow, "elbow"), (wrist, "wrist")):
            sh = SHARES["forearm"][k]
            add(j, "x", -J("forearm", "supination", sh), J("forearm", "pronation", sh), "jarm2022+design",
                "supination", "pronation", f"forearm range, share {sh:g} at the {k}")
        add(wrist, "z", -J("wrist", "flexion" if s > 0 else "extension"), J("wrist", "extension" if s > 0 else "flexion"),
            "jarm2022", "palmar flexion" if s > 0 else "dorsiflexion", "dorsiflexion" if s > 0 else "palmar flexion")
        add(wrist, "y", -J("wrist", "radial_deviation" if s > 0 else "ulnar_deviation"),
            J("wrist", "ulnar_deviation" if s > 0 else "radial_deviation"), "jarm2022",
            "radial deviation" if s > 0 else "ulnar deviation", "ulnar deviation" if s > 0 else "radial deviation")
    for j in ("spine1", "spine2", "spine3"):
        sh = SHARES["thoracolumbar_sagittal"][j]
        add(j, "x", -J("thoracolumbar", "extension", sh), J("thoracolumbar", "flexion", sh), "jarm2022+design",
            "extension", "flexion", f"thoracolumbar range, share {sh:g}")
        sh = SHARES["thoracolumbar_rotation_bending"][j]
        add(j, "y", -J("thoracolumbar", "rotation_right", sh), J("thoracolumbar", "rotation_left", sh),
            "jarm2022+design", "rotation to the right", "rotation to the left", f"thoracolumbar range, share {sh:.4g}")
        add(j, "z", -J("thoracolumbar", "lateral_bending_left", sh), J("thoracolumbar", "lateral_bending_right", sh),
            "jarm2022+design", "lateral bending to the left", "lateral bending to the right",
            f"thoracolumbar range, share {sh:.4g}")
    for j in ("neck", "head"):
        sh = SHARES["cervical"][j]
        add(j, "x", -J("cervical", "extension", sh), J("cervical", "flexion", sh), "jarm2022+design", "extension",
            "flexion", f"cervical range, share {sh:g}")
        add(j, "y", -J("cervical", "rotation_right", sh), J("cervical", "rotation_left", sh), "jarm2022+design",
            "rotation to the right", "rotation to the left", f"cervical range, share {sh:g}")
        add(j, "z", -J("cervical", "lateral_bending_left", sh), J("cervical", "lateral_bending_right", sh),
            "jarm2022+design", "lateral bending to the left", "lateral bending to the right",
            f"cervical range, share {sh:g}")
    return out


def _rot(axis, deg):
    import torch
    from mdfm.body.rotations import axis_angle_to_matrix
    v = torch.zeros(len(deg), 3, dtype=torch.float64)
    v[:, "xyz".index(axis)] = torch.as_tensor(np.radians(deg))
    return axis_angle_to_matrix(v)


def cross_term(named, comp):
    """The largest |component comp| (deg) of the rotations composed from the named components' grids (named:
    {axis: (lo, hi)}), each order of the named axes, about fixed parent axes (base.rotation)."""
    import torch
    from mdfm.body.rotations import matrix_to_axis_angle
    axes = sorted(named)
    grids = [np.linspace(named[a][0], named[a][1], CROSS_STEPS) for a in axes]
    pts = np.array(list(itertools.product(*grids)))
    best = 0.0
    for order in itertools.permutations(range(len(axes))):
        M = torch.eye(3, dtype=torch.float64).expand(len(pts), 3, 3)
        for i in order:
            M = _rot(axes[i], pts[:, i]) @ M
        c = np.degrees(matrix_to_axis_angle(M).numpy())[:, "xyz".index(comp)]
        best = max(best, float(np.abs(c).max()))
    return best


def hand_table():
    from mdfm.synth import hands as H
    joints, names = {}, {}
    for side in H.SIDES:
        for f in H.FINGERS:
            for lv in range(3):
                j = H.hand_joint(side, f, lv)
                lvname = (H.THUMB_LEVELS if f == "thumb" else H.LEVELS)[lv]
                name = f"{side}_{f}_{lvname}"
                names[str(j)] = name
                if f == "thumb" and lvname == "cmc":
                    lo, hi, src = -H.CMC_LIMIT_DEG, H.CMC_LIMIT_DEG, "module"
                    abd = H.CMC_LIMIT_DEG
                else:
                    lo, hi = H.FLEX_LIMITS_DEG[f][lvname]
                    src = "boike2025" if f == "thumb" else "bk2024"
                    abd = H.ABDUCTION_LIMIT_DEG if (lv == 0 and f not in ("middle", "thumb")) else 0.0
                jk = "thumb" if f == "thumb" else "finger"
                lvk = {"mcp": "mcp", "pip": "pip", "dip": "dip", "ip": "ip", "cmc": None}[lvname]
                jarm = None if lvk is None else [-JARM[jk][f"{lvk}_extension"], JARM[jk][f"{lvk}_flexion"]]
                joints[name] = {"joint": j, "flex_lo_deg": float(lo), "flex_hi_deg": float(hi),
                                "abd_deg": float(abd),
                                "source": src, "abd_source": "module", "jarm2022_deg": jarm}
    return {"joints": joints, "joint_names": names,
            "rule": "synth/hands.py FLEX_LIMITS_DEG (bk2024, boike2025) kept as the hand table, the thumb CMC and the "
                    "knuckles' abduction module bounds (ABDUCTION_LIMIT_DEG, CMC_LIMIT_DEG); jarm2022's finger and "
                    "thumb rows recorded beside them (jarm2022_deg: extension, flexion)",
            "abduction_rule": "hand_limit_report reads |abduction| against the knuckle's bound; the middle finger "
                              "and the thumb's MCP and IP have no abduction axis (hands.hand_frame)"}


def base_bodies(lo, hi):
    """{name: [{component, read_deg, bound_deg, excess_deg}]}: the procedural bases of synth/base.py (its POSTURES and
    an 8 s walk) on the selected body model (the open body by default) against the table."""
    from mdfm.synth import base
    from mdfm.synth.body import components_deg, model_name
    bodies = {p: base.static_body(p, 0.1) for p in base.POSTURES}
    bodies["walk_8s"] = base.walk(8.0)[0]
    out = {"body_model": model_name()}
    for name, b in bodies.items():
        c = components_deg(b.rot6d)
        ex = np.maximum(lo[None] - c, c - hi[None])[:, 1:]
        w = ex.max(0)
        out[name] = [{"component": f"{JOINTS[j + 1]}.{'xyz'[k]}",
                      "read_deg": [round(float(c[:, j + 1, k].min()), 2), round(float(c[:, j + 1, k].max()), 2)],
                      "bound_deg": [float(lo[j + 1, k]), float(hi[j + 1, k])], "excess_deg": round(float(w[j, k]), 2)}
                     for j, k in zip(*np.where(w > 1e-6))]
    return out


def main():
    import torch  # noqa: F401  (the cross-term grid)
    named = {}
    lo = np.full((22, 3), np.nan)
    hi = np.full((22, 3), np.nan)
    table_rows = []
    for j, c, a, b, src, mlo, mhi, rule in rows():
        ji, ci = JOINTS.index(j), "xyz".index(c)
        if not np.isnan(lo[ji, ci]):
            raise RuntimeError(f"{j}.{c} typed twice")
        lo[ji, ci], hi[ji, ci] = a, b
        named.setdefault(ji, {})[c] = (a, b)
        table_rows.append({"joint": j, "component": c, "lo_deg": a, "hi_deg": b, "source": src, "motion_lo": mlo,
                           "motion_hi": mhi, "rule": rule})
    lo[0], hi[0] = -180.0, 180.0                     # the root: a world orientation, not checked
    for ji in range(1, 22):
        for ci, c in enumerate("xyz"):
            if np.isnan(lo[ji, ci]):
                nm = named.get(ji, {})
                cr = cross_term(nm, c) if len(nm) >= 2 else 0.0
                bound = max(MODULE_DEG, math.ceil(cr * 10.0) / 10.0)
                lo[ji, ci], hi[ji, ci] = -bound, bound
                table_rows.append({"joint": JOINTS[ji], "component": c, "lo_deg": -bound, "hi_deg": bound,
                                   "source": "module", "motion_lo": None, "motion_hi": None,
                                   "rule": f"no jarm2022 row names it: max(MODULE_DEG {MODULE_DEG:g}, the cross term "
                                           f"of the named motions {cr:.2f} deg rounded up to 0.1)",
                                   "cross_term_deg": round(cr, 3)})
    assert not np.isnan(lo).any() and (lo <= hi).all()
    with open(PRIORS, encoding="utf-8") as f:
        jl = json.load(f)["joint_limits"]
    alo, ahi = np.degrees(np.asarray(jl["lo"])), np.degrees(np.asarray(jl["hi"]))
    comparison = []
    for ji in range(1, 22):
        for ci, c in enumerate("xyz"):
            comparison.append({"component": f"{JOINTS[ji]}.{c}", "anatomical_deg": [lo[ji, ci], hi[ji, ci]],
                               "amass_deg": [round(float(alo[ji, ci]), 2), round(float(ahi[ji, ci]), 2)],
                               "lo_tighter_by_deg": round(float(lo[ji, ci] - alo[ji, ci]), 2),
                               "hi_tighter_by_deg": round(float(ahi[ji, ci] - hi[ji, ci]), 2)})
    tighter = [r for r in comparison if r["lo_tighter_by_deg"] > 0 or r["hi_tighter_by_deg"] > 0]
    table_rows.sort(key=lambda r: (JOINTS.index(r["joint"]), r["component"]))
    out = {"script": "scripts/openbody_limits.py", "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "units": "deg, axis-angle components of the local rotations in the canonical T pose (module docstring)",
           "sources": SOURCES, "jarm2022_read": JARM_READ, "jarm2022": JARM, "shares": SHARES,
           "module_deg": MODULE_DEG, "cross_steps": CROSS_STEPS, "arm_reference_shift_deg": ARM_REFERENCE_SHIFT_DEG,
           "body": {"joints": list(JOINTS), "lo_deg": lo.round(6).tolist(), "hi_deg": hi.round(6).tolist(),
                    "root": "row 0 is a world orientation and is not checked (-180 to 180)"},
           "rows": table_rows, "hands": hand_table(),
           "against_amass": {"table": "reports/body_store/priors.json joint_limits (0.1 and 99.9 percentiles)",
                             "n_components": len(comparison), "n_tighter": len(tighter),
                             "n_lo_tighter": sum(r["lo_tighter_by_deg"] > 0 for r in comparison),
                             "n_hi_tighter": sum(r["hi_tighter_by_deg"] > 0 for r in comparison),
                             "components": comparison}}
    out["base_bodies"] = base_bodies(lo, hi)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    for ji in range(1, 22):
        print(f"{JOINTS[ji]:15s}", "  ".join(f"{'xyz'[c]} [{lo[ji, c]:7.2f}, {hi[ji, c]:7.2f}] "
                                               f"(AMASS [{alo[ji, c]:7.2f}, {ahi[ji, c]:7.2f}])" for c in range(3)))
    print(f"{len(tighter)} of {len(comparison)} components tighter than the AMASS table on at least one side")


if __name__ == "__main__":
    main()
