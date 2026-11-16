"""The catalogue's one body on the open body model (plan phase C, wave 1; docs/OPEN_BODY_PORT_DESIGN.md decision D8):
solves openbody.CATALOGUE_TARGETS (a neutral adult of 40 years at the stature midway between the women's and the
men's 45-60 band, the 'average' build's BMI and muscle) to an Anny phenotype (openbody.solve_phenotype) and writes
reports/openbody/catalogue_body.json, which openbody.catalogue_phenotype reads: the phenotype, its targets, the
measured stature, mass and BMI of its rest mesh, and the canonical rest skeleton's segment lengths. Anny's model
data stay in memory and in its own cache; no vertex array is written.
"""
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "openbody", "catalogue_body.json")
SEGMENTS = (("pelvis", "spine3"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("spine3", "neck"), ("neck", "head"))


def main():
    from mdfm.synth import openbody as OB
    m = OB.model()
    ph = OB.solve_catalogue_phenotype(m)
    a = OB.anthropometry(ph, m)
    ob = OB.OpenBody(ph, m)
    J = dict(zip(OB.JOINT_NAMES, ob.rest_joints52()))
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    out = {"script": "scripts/openbody_catalogue_body.py",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "mdfm_commit": head.strip(),
           "anny_version": OB.ANNY_VERSION, "targets": OB.CATALOGUE_TARGETS, "phenotype": ph.as_dict(),
           "measured": {k: round(float(v), 6) for k, v in a.items()},
           "segment_lengths_m": {f"{p}-{c}": round(float((J[c] - J[p]).norm()), 5) for p, c in SEGMENTS},
           "rest_pelvis_height_m": round(float(J["pelvis"][1]), 5)}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
