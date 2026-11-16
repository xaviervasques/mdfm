"""Rigidity of the MoveAtlas AMASS canonical records (read-only).

A record built by forward kinematics of SMPL+H is a rigid skeleton: every bone keeps its
length across frames. This script measures, for every record of
datasets/socle/amass/processed/canonical/amass_<subset>.npz, the coefficient of variation over
frames of the nine segments of the canonical 15-joint layout that are single bones (neck to
head, both upper arms, forearms, thighs and shanks) and flags a record as corrupted when the
largest exceeds 0.05. The pelvis-to-neck distance is reported but not used: it spans the three
spine joints, which bend, so it varies in a correct record (version 1 of this script used it
and flagged 1,439 records; 1,035 of them are reproduced exactly by forward kinematics,
reports/nuisance/motionx_amass_summary.md). MD_FM does not use these records (it
recomputes joints from the raw SMPL+H parameters, tests/test_body.py); the list is for the
owner's MoveAtlas corrections.

Writes reports/probe_amass_canonical_rigidity/records.csv, summary.md and _provenance.json.
Run: python scripts/probe_amass_canonical_rigidity.py --moveatlas <moveatlas>
"""
import csv
import glob
import os

import numpy as np

from _common import parse_args, write_provenance

ARGS = parse_args(__doc__.splitlines()[0])
write_provenance(ARGS)
root = f"{ARGS.moveatlas}/datasets/socle/amass/processed"
takes = {}
with open(f"{root}/canonical_manifest.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        takes.setdefault(r["take"].split("/")[0], []).append(r)

# canonical 15: pelvis, neck, head, shoulders L R, elbows L R, wrists L R, hips L R, knees L R, ankles L R
BONES = [(1, 2), (3, 5), (4, 6), (5, 7), (6, 8), (9, 11), (10, 12), (11, 13), (12, 14)]
rows = []
for path in sorted(glob.glob(f"{root}/canonical/amass_*.npz")):
    if os.path.basename(path).startswith("._"):
        continue
    subset = os.path.basename(path)[len("amass_"):-len(".npz")]
    z = np.load(path)
    keys = sorted(k[:-len("/coords")] for k in z.files if k.endswith("/coords"))
    man = takes.get(subset, [])
    for i, k in enumerate(keys):
        c = z[k + "/coords"].astype(np.float64)
        cvs = []
        for a, b in BONES:
            d = np.linalg.norm(c[:, a] - c[:, b], axis=-1)
            cvs.append(float(d.std() / max(d.mean(), 1e-12)))
        cv_t = cvs[BONES.index((9, 11))]
        cv_b = max(cvs)
        torso = np.linalg.norm(c[:, 1] - c[:, 0], axis=-1)
        cv_o = float(torso.std() / max(torso.mean(), 1e-12))
        take = man[i]["take"] if i < len(man) and int(man[i]["n_frames"]) == c.shape[0] else ""
        rows.append({"subset": subset, "record": k, "take": take, "frames": c.shape[0],
                     "thigh_cv": round(cv_t, 6), "max_bone_cv": round(cv_b, 6), "torso_cv": round(cv_o, 6),
                     "corrupted": cv_b > 0.05})

with open(os.path.join(ARGS.out, "records.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

bad = [r for r in rows if r["corrupted"]]
fr = np.array([r["frames"] for r in rows])
isbad = np.array([r["corrupted"] for r in rows])
md = ["# MoveAtlas AMASS canonical records: rigidity", "",
      "Measured by scripts/probe_amass_canonical_rigidity.py (provenance in _provenance.json).", "",
      f"Records: {len(rows)}; corrupted (largest bone-length variation above 5 percent): {len(bad)}; "
      f"left thigh alone above 5 percent: {sum(r['thigh_cv'] > 0.05 for r in rows)}; "
      f"pelvis-to-neck above 5 percent (not a bone, not used): {sum(r['torso_cv'] > 0.05 for r in rows)}; "
      f"records whose take was matched through the manifest order: {sum(1 for r in rows if r['take'])}.", "",
      "| subset | records | corrupted |", "|---|---|---|"]
for s in sorted({r["subset"] for r in rows}):
    sel = [r for r in rows if r["subset"] == s]
    md.append(f"| {s} | {len(sel)} | {sum(r['corrupted'] for r in sel)} |")
md += ["", "| frames | records | corrupted |", "|---|---|---|"]
for lo, hi in ((0, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000), (5000, 10 ** 7)):
    m = (fr >= lo) & (fr < hi)
    md.append(f"| {lo} to {hi - 1} | {int(m.sum())} | {int(isbad[m].sum())} |")
md += ["", f"Shortest corrupted record: {int(fr[isbad].min()) if isbad.any() else 'none'} frames; "
       f"longest rigid record: {int(fr[~isbad].max())} frames."]
text = "\n".join(md) + "\n"
assert chr(0x2014) not in text
open(os.path.join(ARGS.out, "summary.md"), "w", encoding="utf-8", newline="\n").write(text)
print(text)
