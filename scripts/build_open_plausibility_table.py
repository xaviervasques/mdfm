"""The joint-limit table of the open model's plausibility term (plan docs/MODEL_OPEN_PORT_PLAN.md, section 3.1 and step
M4): the anatomical table of the synthetic patient's wave 1 (reports/openbody/joint_limits.json 'body', built by
scripts/openbody_limits.py from jarm2022 and the design's rules) written in the form the plausibility term reads
(mdfm/train/losses.py _limits: a 'joint_limits' block with the 22 program joints and lo, hi (22, 3) rad of each
axis-angle component of the local rotations), so that no AMASS percentile is left in the model.

The anatomical table states its components in the canonical T pose's local frames, the frames of the open body's
local rotations (mdfm/body/openbody.py); its row 0 (the root, a world orientation) is -180 to 180 degrees and so never
binds, where v1's table bounded the root in its heading frame by the AMASS percentiles: the root's tilt is left to the
data and the other terms.

Output: reports/open_body_priors/plausibility_table.json and its entry in reports/open_body_priors/_provenance.json
(the 'plausibility_table' key; the other keys of that file are left as they are).
"""
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
from mdfm.body.joints import BODY_JOINTS  # noqa: E402

SRC = os.path.join(TREE, "reports", "openbody", "joint_limits.json")
OUT_DIR = os.path.join(TREE, "reports", "open_body_priors")
OUT = os.path.join(OUT_DIR, "plausibility_table.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def table(src=SRC):
    with open(src, encoding="utf-8") as f:
        body = json.load(f)["body"]
    if list(body["joints"]) != list(BODY_JOINTS):
        raise ValueError(f"{src}: body joints differ from mdfm/body/joints.py")
    d = math.pi / 180.0
    lo = [[float(x) * d for x in row] for row in body["lo_deg"]]
    hi = [[float(x) * d for x in row] for row in body["hi_deg"]]
    if any(a > b for ra, rb in zip(lo, hi) for a, b in zip(ra, rb)):
        raise ValueError(f"{src}: a lower limit above its upper limit")
    return {"note": "the anatomical table of reports/openbody/joint_limits.json 'body' in radians: each axis-angle "
                    "component of the local rotations in the canonical T pose; row 0 (the root) "
                    + str(body.get("root")),
            "joints": list(BODY_JOINTS), "percentiles": None, "lo": lo, "hi": hi,
            "source": "reports/openbody/joint_limits.json", "source_sha256": sha256_file(src)}


def main():
    doc = {"version": "open_plausibility_v1", "joint_limits": table()}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    prov_path = os.path.join(OUT_DIR, "_provenance.json")
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
    head = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    prov["plausibility_table"] = {"script": "scripts/build_open_plausibility_table.py", "mdfm_head": head.strip(),
                                  "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                  "inputs": {"reports/openbody/joint_limits.json": sha256_file(SRC)},
                                  "outputs": {"reports/open_body_priors/plausibility_table.json": sha256_file(OUT)}}
    with open(prov_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    print(json.dumps(prov["plausibility_table"], indent=1))


if __name__ == "__main__":
    main()
