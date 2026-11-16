"""The open body of the synthetic patient (plan phase C, docs/OPEN_BODY_PORT_DESIGN.md): the Anny body model
(NAVER LABS Europe; code Apache-2.0, model data CC0 under anny/data/mpfb2) behind an allow list, posed from the
program's per-joint rotations through a joint map, so that the generator can leave SMPL+H.

Licence hygiene (design section 2, the audit of scripts/openbody_audit.py):
- the loader accepts only the topology and rig names of ALLOWED_TOPOLOGIES and ALLOWED_RIGS ('anny' both): the
  'smplx' and 'smpl' topologies and the 'smpl' wrappers need a non-commercial download and never load here;
- guard() runs before anny is imported: ANNY_CACHE_DIR is pinned to CACHE_DIR (<store>/anny_cache unless the
  environment variable MDFM_ANNY_CACHE names another folder), the modules that reach a non-free asset
  (BLOCKED_MODULES: anny.models.smpl, anny.models.retopology, smplx, chumpy) refuse to import, and the anny
  functions that download or locate the non-commercial archive raise OpenBodyRefused;
- the cache is scanned before every load (cache_audit): a file or folder whose name carries a NONFREE_MARKS word
  refuses the load.

Frames. Anny's world is z up, the subject's left +x and the face toward -y, in metres. The program's world is y up,
the subject's left +x and the face toward +z (synth/body.py): a point maps by ours = C anny with C the rotation of
TO_OURS (a quarter turn about x). Every array returned by this module is in the program's frame.

Canonical rest (design section 3.2). The program's rotations are local rotations about world-aligned axes from a
T pose (synth/base.py: x the subject's left, y up, z forward; the arms horizontal). Anny's rest is an A pose. Each
Anny bone b gets a world rotation A_b (canonical_corrections) that takes its rest segment to the canonical one:
- trunk, pelvis, collars, neck, head, feet and toes keep Anny's rest (A = I);
- thigh and shank: the minimal rotation of the segment (hip to knee, knee to ankle) onto the vertical;
- upper arm: the segment (shoulder to elbow) onto the horizontal, the rest elbow's flexion direction forward (+z),
  so that a rotation about y flexes the elbow toward its crease as on the old body;
- forearm and hand: the segment (elbow to wrist, wrist to the middle knuckle) onto the horizontal, the palmar
  normal down (-y), the palms facing the floor as on the old body;
- the phalanges of the four fingers: their segment onto the hand's axis, the palmar normal down (straight, flat
  fingers); the thumb keeps its rest relative to the hand (A = A_wrist);
- a bone the joint map leaves out takes the A of the program joint that owns it (OWNER), so that it moves rigidly
  with that segment.

Posing (OpenBody.pose). With Z_j the program's global rotation of joint j (the chain of the 52 local rotations,
PARENTS52) expressed in Anny's axes, every Anny bone turns by G_b = Z_owner(b) A_b and its head follows its parent,
p_b = p_parent + G_parent (h_b - h_parent), the root head at its rest plus the translation (the program's pelvis is
transl plus the rest pelvis, as synth/body.py). The mesh is the linear blend skinning of Anny's rest vertices with
the bone transforms [G_b | p_b - G_b h_b] (Anny has no pose blend shapes). The 52 program joints are the heads of
the mapped bones (JOINT_MAP), so OpenBody.rest_joints52() fed to the program's forward kinematics gives the posed
heads to rounding (tests/test_openbody.py).

Selector (backend): the open body by default; the SMPL+H reference (mdfm/synth/smplh_reference.py) only when the
caller passes allow_reference=True or the environment sets MDFM_SMPLH_REFERENCE=1, for the clip-by-clip comparison
of the port; it never ships (design section 9).

Arrays: the Anny model data stay in memory and in Anny's own cache folder; nothing of this module writes a vertex
array to a file.
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

F64 = torch.float64
ANNY_VERSION = "0.6.2"
CACHE_DIR = os.environ.get("MDFM_ANNY_CACHE", "<store>/anny_cache").replace("\\", "/")
ALLOWED_TOPOLOGIES = ("anny",)
ALLOWED_RIGS = ("anny",)
# modules that reach a non-free asset (the SMPL and SMPL-X wrappers and topologies of anny, the smplx package)
BLOCKED_MODULES = ("anny.models.smpl", "anny.models.retopology", "smplx", "chumpy")
# a cache entry whose lower-case name holds one of these words refuses the load
NONFREE_MARKS = ("noncommercial", "smpl", "anny2smpl")
REFERENCE_ENV = "MDFM_SMPLH_REFERENCE"
BACKEND_ENV = "MDFM_BODY_MODEL"
OPEN, SMPLH_REFERENCE = "open", "smplh_reference"

# anny (z up, face -y) to the program's frame (y up, face +z): ours = TO_OURS @ anny
TO_OURS = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))

# ---------------------------------------------------------------- the joint map
BODY_JOINTS = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
               "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
PARENTS22 = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
FINGERS = ("index", "middle", "pinky", "ring", "thumb")        # the program's hand order (synth/hands.py)
ANNY_FINGER = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
SIDE_SUFFIX = {"left": "L", "right": "R"}
_BODY_BONES = ("root", "upperleg01.L", "upperleg01.R", "spine04", "lowerleg01.L", "lowerleg01.R", "spine02",
               "foot.L", "foot.R", "spine01", "toe3-1.L", "toe3-1.R", "neck01", "clavicle.L", "clavicle.R", "head",
               "upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R", "wrist.L", "wrist.R")


def _hand_map():
    names, bones, parents = [], [], []
    for side, wrist in (("left", 20), ("right", 21)):
        s = SIDE_SUFFIX[side]
        for finger in FINGERS:
            first = len(BODY_JOINTS) + len(names)
            for k in (1, 2, 3):
                names.append(f"{side}_{finger}{k}")
                bones.append(f"finger{ANNY_FINGER[finger]}-{k}.{s}")
                parents.append(wrist if k == 1 else first + k - 2)
    return tuple(names), tuple(bones), tuple(parents)


HAND_JOINTS, _HAND_BONES, _HAND_PARENTS = _hand_map()
JOINT_NAMES = BODY_JOINTS + HAND_JOINTS                         # the 52 program joints (synth/skeleton.py order)
PARENTS52 = PARENTS22 + _HAND_PARENTS
JOINT_MAP = tuple(zip(JOINT_NAMES, _BODY_BONES + _HAND_BONES))  # program joint -> Anny bone whose head it is
N_BODY, N_JOINTS = len(BODY_JOINTS), len(JOINT_NAMES)
# bones outside the map, by the program joint whose segment carries them (rigid with it)
_UNMAPPED_OWNER = {"pelvis.L": 0, "pelvis.R": 0, "spine05": 0, "spine03": 3, "upperleg02.L": 1, "upperleg02.R": 2,
                   "lowerleg02.L": 4, "lowerleg02.R": 5, "neck02": 12, "neck03": 12, "eye.L": 15, "eye.R": 15,
                   "shoulder01.L": 13, "shoulder01.R": 14, "upperarm02.L": 16, "upperarm02.R": 17,
                   "lowerarm02.L": 18, "lowerarm02.R": 19}
for _s, _j in (("L", 10), ("R", 11)):
    for _t, _n in ((1, 2), (2, 3), (3, 3), (4, 3), (5, 3)):
        for _k in range(1, _n + 1):
            if (_t, _k) != (3, 1):
                _UNMAPPED_OWNER[f"toe{_t}-{_k}.{_s}"] = _j     # the five toes move with the program's foot joint
for _s, _j in (("L", 20), ("R", 21)):
    for _m in (1, 2, 3, 4):
        _UNMAPPED_OWNER[f"metacarpal{_m}.{_s}"] = _j            # the metacarpals are rigid with the hand


class OpenBodyRefused(RuntimeError):
    """A request that would reach a non-free asset, or a load outside the allow list."""


# ---------------------------------------------------------------- licence guard
class _BlockFinder(importlib.abc.MetaPathFinder):
    """Refuses to import the modules of BLOCKED_MODULES (and their submodules)."""

    def find_spec(self, fullname, path=None, target=None):
        for name in BLOCKED_MODULES:
            if fullname == name or fullname.startswith(name + "."):
                raise ImportError(f"{fullname}: blocked by mdfm.synth.openbody (it reaches a non-free asset)")
        return None


def _refuse(*_a, **_k):
    raise OpenBodyRefused("the non-commercial Anny assets (SMPL and SMPL-X topologies) are refused by "
                          "mdfm.synth.openbody")


def guard():
    """Pin Anny's cache to CACHE_DIR, block BLOCKED_MODULES and disarm Anny's non-commercial download (module
    docstring). Idempotent; returns the cache folder."""
    os.environ["ANNY_CACHE_DIR"] = CACHE_DIR
    if not any(isinstance(f, _BlockFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockFinder())
    loaded = [m for m in BLOCKED_MODULES if m in sys.modules]
    if loaded:
        raise OpenBodyRefused(f"blocked modules already imported in this process: {loaded}")
    import anny.paths as AP
    AP.set_anny_cache_path(CACHE_DIR)
    AP.download_noncommercial_data = _refuse
    AP.get_anny2smpl_data_path = _refuse
    AP.get_anny2smplx_data_path = _refuse
    return CACHE_DIR


def cache_audit(cache_dir=None):
    """[(relative path, bytes)] of every file under the cache folder; raises OpenBodyRefused when a file or folder
    name holds a NONFREE_MARKS word."""
    root = CACHE_DIR if cache_dir is None else cache_dir
    out = []
    if not os.path.isdir(root):
        return out
    for d, dirs, files in os.walk(root):
        for name in dirs + files:
            if any(m in name.lower() for m in NONFREE_MARKS):
                raise OpenBodyRefused(f"non-free entry in the Anny cache: {os.path.join(d, name)}")
        for name in files:
            p = os.path.join(d, name)
            out.append((os.path.relpath(p, root).replace("\\", "/"), os.path.getsize(p)))
    return sorted(out)


def nonfree_packages():
    """Names of the non-free packages (smplx, chumpy) importable in this environment (an empty list here)."""
    # the path finder reads the file system directly, past the guard's finder
    return [name for name in ("smplx", "chumpy") if importlib.machinery.PathFinder.find_spec(name) is not None]


def check_names(rig, topology):
    if topology not in ALLOWED_TOPOLOGIES:
        raise OpenBodyRefused(f"topology {topology!r} is outside the allow list {ALLOWED_TOPOLOGIES}")
    if rig not in ALLOWED_RIGS:
        raise OpenBodyRefused(f"rig {rig!r} is outside the allow list {ALLOWED_RIGS}")


@lru_cache(maxsize=2)
def model(rig="anny", topology="anny"):
    """The Anny model (float64, CPU, linear blend skinning in torch), through the allow list and the guard."""
    check_names(rig, topology)
    guard()
    cache_audit()
    if nonfree_packages():
        raise OpenBodyRefused(f"non-free packages installed: {nonfree_packages()}")
    import anny
    if anny.__version__ != ANNY_VERSION:
        raise OpenBodyRefused(f"anny {anny.__version__} installed, the port is pinned to {ANNY_VERSION}")
    m = anny.Anny(rig=rig, topology=topology, local_changes="none", facial_actions="none", phenotypes="default",
                  skinning_method="lbs")
    cache_audit()
    return m.to(dtype=F64)


def bone_index(m=None):
    m = model() if m is None else m
    return {b: i for i, b in enumerate(m.bone_labels)}


def owner_table(m=None):
    """(n_bones,) int64: the program joint (0 to 51) that carries each Anny bone (module docstring)."""
    m = model() if m is None else m
    idx = bone_index(m)
    owner = np.full(len(m.bone_labels), -1, np.int64)
    for j, (_, b) in enumerate(JOINT_MAP):
        owner[idx[b]] = j
    for b, j in _UNMAPPED_OWNER.items():
        owner[idx[b]] = j
    if (owner < 0).any():
        raise RuntimeError(f"bones without an owner: {[m.bone_labels[i] for i in np.where(owner < 0)[0]]}")
    return owner


# ---------------------------------------------------------------- identities (phenotypes)
SEX_GENDER = {"male": 0.0, "female": 1.0, "neutral": 0.5}       # Anny's gender anchors: male 0, female 1
MH_YOUNG_YEARS, MH_OLD_YEARS = 25.0, 90.0                        # MakeHuman's age targets (design section 4)


def age_phenotype(years):
    """Anny's age value of an adult age (years 25 to 90): linear between the 'young' anchor (2/3, read as 25
    years) and the 'old' anchor (1, read as 90 years), the MakeHuman convention (an uncertainty: Anny's code gives
    no age in years; design section 4)."""
    y = float(years)
    if not MH_YOUNG_YEARS <= y <= MH_OLD_YEARS:
        raise ValueError(f"adult age {y} outside [{MH_YOUNG_YEARS}, {MH_OLD_YEARS}] years")
    return 2.0 / 3.0 + (y - MH_YOUNG_YEARS) / (MH_OLD_YEARS - MH_YOUNG_YEARS) / 3.0


@dataclass(frozen=True)
class Phenotype:
    """Anny's six default phenotype values, each in [0, 1]."""
    gender: float = 0.5
    age: float = 2.0 / 3.0
    muscle: float = 0.5
    weight: float = 0.5
    height: float = 0.5
    proportions: float = 0.5

    def as_dict(self):
        return {k: float(getattr(self, k)) for k in ("gender", "age", "muscle", "weight", "height", "proportions")}


NEUTRAL_ADULT = Phenotype(gender=0.5, age=age_phenotype(40.0))   # wave 0's comparison body (design section 8)

# The 14 adult identities (design section 4): the two V4 references and the 12-identity panel of
# reports/video_identity (sex, age and build from their prompt tables; woman_a and man_a carry no build word and
# take 'average'). Stature and BMI are design targets, not a sourced population table: a stature per sex and age band
# (BAND_HEIGHT_M, lower after 70 for the stature loss of ageing) and a BMI per build (BUILD_BMI, inside the WHO adult
# classes: slim and average in 18.5 to 24.9, heavier in 25 to 29.9); muscle per build (BUILD_MUSCLE), because the
# heavier BMI is reachable only at low muscle values on Anny (the grid of reports/openbody/anny_facts.json).
BUILD_BMI = {"slim": 20.5, "average": 24.0, "heavier": 29.0}
BUILD_MUSCLE = {"slim": 0.35, "average": 0.25, "heavier": 0.10}
BAND_HEIGHT_M = {("female", "25-35"): 1.64, ("female", "45-60"): 1.63, ("female", "60-70"): 1.61,
                 ("female", "70-85"): 1.59, ("male", "25-35"): 1.77, ("male", "45-60"): 1.76,
                 ("male", "60-70"): 1.74, ("male", "70-85"): 1.72}
IDENTITY_SPECS = (  # (name, sex, age years, build, age band)
    ("woman_a", "female", 65, "average", "60-70"), ("man_a", "male", 65, "average", "60-70"),
    ("panel_woman_28", "female", 28, "slim", "25-35"), ("panel_woman_33", "female", 33, "average", "25-35"),
    ("panel_man_29", "male", 29, "average", "25-35"), ("panel_man_34", "male", 34, "slim", "25-35"),
    ("panel_woman_48", "female", 48, "heavier", "45-60"), ("panel_woman_56", "female", 56, "slim", "45-60"),
    ("panel_man_47", "male", 47, "heavier", "45-60"), ("panel_man_57", "male", 57, "average", "45-60"),
    ("panel_woman_74", "female", 74, "average", "70-85"), ("panel_woman_81", "female", 81, "heavier", "70-85"),
    ("panel_man_73", "male", 73, "heavier", "70-85"), ("panel_man_82", "male", 82, "slim", "70-85"))


# The catalogue's one body (design decision D8, wave 1): a neutral adult of 40 years (gender 0.5), at the stature
# midway between the women's and the men's 45-60 band (BAND_HEIGHT_M) and the 'average' build's BMI and muscle. Its
# solved phenotype is recorded by scripts/openbody_catalogue_body.py in CATALOGUE_BODY_PATH, which catalogue_phenotype
# reads (solving at every import would cost a few hundred rest evaluations).
CATALOGUE_TARGETS = {"sex": "neutral", "age_years": 40.0,
                     "height_m": round(0.5 * (BAND_HEIGHT_M[("female", "45-60")] + BAND_HEIGHT_M[("male", "45-60")]), 4),
                     "bmi": BUILD_BMI["average"], "muscle": BUILD_MUSCLE["average"], "proportions": 0.5}
CATALOGUE_BODY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   "reports", "openbody", "catalogue_body.json")


def solve_catalogue_phenotype(m=None):
    """The catalogue body's phenotype solved from CATALOGUE_TARGETS (solve_phenotype)."""
    t = CATALOGUE_TARGETS
    return solve_phenotype(t["sex"], t["age_years"], t["height_m"], t["bmi"], muscle=t["muscle"],
                           proportions=t["proportions"], m=m)


@lru_cache(maxsize=1)
def catalogue_phenotype():
    """The catalogue body's Phenotype, read from CATALOGUE_BODY_PATH (scripts/openbody_catalogue_body.py); its
    targets must be CATALOGUE_TARGETS (a changed target makes the file stale and refuses)."""
    with open(CATALOGUE_BODY_PATH, encoding="utf-8") as f:
        d = json.load(f)
    if d["targets"] != CATALOGUE_TARGETS:
        raise RuntimeError(f"{CATALOGUE_BODY_PATH} was solved for other targets: rerun "
                           "scripts/openbody_catalogue_body.py")
    return Phenotype(**d["phenotype"])


def identity_targets(name):
    """{'sex', 'age_years', 'build', 'height_m', 'bmi', 'muscle'} of one of IDENTITY_SPECS."""
    for n, sex, age, build, band in IDENTITY_SPECS:
        if n == name:
            return {"sex": sex, "age_years": float(age), "build": build, "height_m": BAND_HEIGHT_M[(sex, band)],
                    "bmi": BUILD_BMI[build], "muscle": BUILD_MUSCLE[build]}
    raise KeyError(name)


def rest_state(phen=NEUTRAL_ADULT, m=None):
    """Rest of a phenotype in Anny's frame: (vertices (V, 3), bone heads (B, 3), bone rest rotations (B, 3, 3))."""
    m = model() if m is None else m
    with torch.no_grad():
        out = m(pose_parameters=None, phenotype_kwargs={k: torch.tensor([v], dtype=F64)
                                                        for k, v in phen.as_dict().items()},
                pose_parameterization="local-ref")
    P = out["rest_bone_poses"][0]
    return out["rest_vertices"][0].clone(), P[:, :3, 3].clone(), P[:, :3, :3].clone()


def stature_m(vertices):
    """Height (m) of rest vertices in Anny's frame (z extent), the measure of anny.Anthropometry."""
    return float(vertices[:, 2].max() - vertices[:, 2].min())


def volume_m3(vertices, faces):
    """Enclosed volume (m3) of a closed triangle mesh (signed tetrahedra)."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(torch.abs((torch.cross(v0, v1, dim=-1) * v2).sum() / 6.0))


BODY_DENSITY = 980.0      # kg/m3, the density anny.Anthropometry uses for the mass


def anthropometry(phen=NEUTRAL_ADULT, m=None):
    """{'height_m', 'mass_kg', 'bmi'} of a phenotype's rest mesh (mass = volume x BODY_DENSITY)."""
    m = model() if m is None else m
    v, _, _ = rest_state(phen, m)
    h = stature_m(v)
    mass = volume_m3(v, m.faces) * BODY_DENSITY
    return {"height_m": h, "mass_kg": mass, "bmi": mass / h ** 2}


def solve_phenotype(sex, age_years, height_m, bmi, muscle=0.5, proportions=0.5, rounds=6, m=None):
    """Phenotype whose rest mesh has the target stature and BMI (bisection on Anny's height value, then on its
    weight value, alternated `rounds` times). Raises ValueError when a target lies outside the reach of [0, 1]."""
    m = model() if m is None else m
    base = dict(gender=SEX_GENDER[sex], age=age_phenotype(age_years), muscle=muscle, proportions=proportions)

    def bisect(key, target, measure, cur):
        lo, hi = 0.0, 1.0
        f = lambda x: anthropometry(Phenotype(**{**base, **cur, key: x}), m)[measure] - target  # noqa: E731
        flo, fhi = f(lo), f(hi)
        if flo > 0 or fhi < 0:
            raise ValueError(f"{measure} {target} outside the reach of {key} in [0, 1] ({flo + target:.3f} to "
                             f"{fhi + target:.3f})")
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
        return 0.5 * (lo + hi)
    cur = {"height": 0.5, "weight": 0.5}
    for _ in range(rounds):
        cur["height"] = bisect("height", height_m, "height_m", cur)
        cur["weight"] = bisect("weight", bmi, "bmi", cur)
    return Phenotype(**{**base, **cur})


# ---------------------------------------------------------------- canonical rest
def _rot_between(u, t):
    """(3, 3) the minimal rotation taking the direction u onto t."""
    u = u / u.norm()
    t = t / t.norm()
    v = torch.linalg.cross(u, t)
    c = float(u @ t)
    s = float(v.norm())
    if s < 1e-12:
        if c > 0:
            return torch.eye(3, dtype=F64)
        raise ValueError("opposite directions: the minimal rotation is not unique")
    k = v / s
    K = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=F64)
    return torch.eye(3, dtype=F64) + s * K + (1 - c) * (K @ K)


def largest_part(faces, n_vertices):
    """(n_vertices,) bool: the vertices of the largest connected part of a triangle mesh (union-find on its edges)."""
    parent = np.arange(n_vertices)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    f = np.asarray(faces, dtype=np.int64)
    for a, b in np.concatenate([f[:, [0, 1]], f[:, [1, 2]]]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n_vertices)])
    big = np.bincount(roots).argmax()
    return roots == big


def _axis_matrix(u, angle):
    """(N, 3, 3) rotations about the unit axis u (3,) by the angles (N,) rad (Rodrigues)."""
    K = torch.tensor([[0.0, -float(u[2]), float(u[1])], [float(u[2]), 0.0, -float(u[0])],
                      [-float(u[1]), float(u[0]), 0.0]], dtype=angle.dtype, device=angle.device)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + s * K + (1.0 - c) * (K @ K)


def _frame(a, b):
    """(3, 3) orthonormal frame (columns): a, the part of b normal to a, their cross product."""
    e1 = a / a.norm()
    e2 = b - (b @ e1) * e1
    if float(e2.norm()) < 1e-6:
        raise ValueError("degenerate secondary direction")
    e2 = e2 / e2.norm()
    return torch.stack([e1, e2, torch.linalg.cross(e1, e2)], 1)


def _align(u, f, ut, ft):
    """(3, 3) the rotation taking the direction u onto ut and the direction f (made normal to u) onto ft."""
    return _frame(ut, ft) @ _frame(u, f).T


def dominant_bone(m=None):
    """(V,) the bone with the largest skinning weight of each vertex."""
    m = model() if m is None else m
    w, i = m.vertex_bone_weights, m.vertex_bone_indices
    return torch.gather(i, 1, w.argmax(1, keepdim=True))[:, 0].long()


def fingertip_vertex(bone, vertices, heads, m=None):
    """The vertex whose dominant bone is `bone` (a distal phalanx) lying farthest along the phalanx's rest
    direction (from its parent's head to its head)."""
    m = model() if m is None else m
    idx = bone_index(m)
    b = idx[bone]
    d = heads[b] - heads[m.bone_parents[b]]
    sel = torch.where(dominant_bone(m) == b)[0]
    return int(sel[torch.argmax((vertices[sel] - heads[b]) @ d)])


def palm_normal(side, heads, m=None):
    """Anny-frame unit palmar normal of a hand from bone heads: the normal of the plane through the wrist and the
    index and pinky knuckles, oriented by handedness (n = s (a x r), a the hand's axis toward the middle knuckle,
    r the direction from the pinky to the index knuckle, s +1 on the left and -1 on the right; down in a T pose)."""
    m = model() if m is None else m
    idx = bone_index(m)
    s = SIDE_SUFFIX[side]
    w, i2, i5, i3 = (heads[idx[n]] for n in (f"wrist.{s}", f"finger2-1.{s}", f"finger5-1.{s}", f"finger3-1.{s}"))
    a = i3 - w
    r = i2 - i5
    n = torch.linalg.cross(a, r) * (1.0 if side == "left" else -1.0)
    # the plane through wrist, index and pinky knuckles carries the normal; the handedness gives its sign
    n_plane = torch.linalg.cross(i2 - w, i5 - w)
    n_plane = n_plane * torch.sign(n_plane @ n)
    return n_plane / n_plane.norm()


def canonical_corrections(heads, vertices, m=None):
    """(B, 3, 3) the world rotations A_b of the canonical rest (module docstring), in Anny's frame."""
    m = model() if m is None else m
    idx = bone_index(m)
    C = torch.tensor(TO_OURS, dtype=F64)
    ours = lambda v: C.T @ torch.tensor(v, dtype=F64)               # noqa: E731  a direction of ours in Anny's
    A = torch.eye(3, dtype=F64).repeat(len(m.bone_labels), 1, 1)
    down = below = ours((0, -1, 0))
    fwd = ours((0, 0, 1))
    for side, sx in (("left", 1.0), ("right", -1.0)):
        s = SIDE_SUFFIX[side]
        h = lambda n: heads[idx[f"{n}.{s}"]]                       # noqa: E731
        A[idx[f"upperleg01.{s}"]] = _rot_between(h("lowerleg01") - h("upperleg01"), down)
        A[idx[f"lowerleg01.{s}"]] = _rot_between(h("foot") - h("lowerleg01"), down)
        lateral = ours((sx, 0, 0))
        upper, fore = h("lowerarm01") - h("upperarm01"), h("wrist") - h("lowerarm01")
        A[idx[f"upperarm01.{s}"]] = _align(upper, fore, lateral, fwd)
        n = palm_normal(side, heads, m)
        A[idx[f"lowerarm01.{s}"]] = _align(fore, n, lateral, below)
        A[idx[f"wrist.{s}"]] = _align(h("finger3-1") - h("wrist"), n, lateral, below)
        for f in (2, 3, 4, 5):
            for k in (1, 2):
                A[idx[f"finger{f}-{k}.{s}"]] = _align(h(f"finger{f}-{k + 1}") - h(f"finger{f}-{k}"), n, lateral,
                                                      below)
            tip = vertices[fingertip_vertex(f"finger{f}-3.{s}", vertices, heads, m)]
            A[idx[f"finger{f}-3.{s}"]] = _align(tip - h(f"finger{f}-3"), n, lateral, below)
        for k in (1, 2, 3):
            A[idx[f"finger1-{k}.{s}"]] = A[idx[f"wrist.{s}"]]
    owner = owner_table(m)
    mapped = {idx[b]: j for j, (_, b) in enumerate(JOINT_MAP)}
    joint_bone = {j: b for b, j in mapped.items()}
    for b, j in enumerate(owner):
        if b not in mapped:
            A[b] = A[joint_bone[int(j)]]
    return A


# ---------------------------------------------------------------- the posed body
class OpenBody:
    """One identity on the open body: its rest, its canonical corrections and the posing of module docstring."""

    def __init__(self, phen=NEUTRAL_ADULT, m=None, twist_split=True):
        self.m = model() if m is None else m
        self.twist_split = bool(twist_split)
        self._shares = self._rest_np = self._dom = self._surface = None
        self.phenotype = phen
        self.rest_vertices, self.rest_heads, self.rest_R = rest_state(phen, self.m)
        self.A = canonical_corrections(self.rest_heads, self.rest_vertices, self.m)
        self.owner = torch.as_tensor(owner_table(self.m))
        self.parents = [int(p) for p in self.m.bone_parents]
        if any(p >= b for b, p in enumerate(self.parents) if p >= 0):
            raise RuntimeError("Anny's bones are expected in parent-first order")
        idx = bone_index(self.m)
        self.mapped = torch.tensor([idx[b] for _, b in JOINT_MAP], dtype=torch.int64)
        self.C = torch.tensor(TO_OURS, dtype=F64)
        self.faces = self.m.faces.detach().cpu().numpy().astype(np.int64)
        from anny.skinning.skinning import dense_bone_weights
        self.weights = dense_bone_weights(self.m.vertex_bone_weights, self.m.vertex_bone_indices,
                                          len(self.parents)).to(F64)
        eye = torch.eye(3, dtype=F64).expand(1, N_JOINTS, 3, 3)
        heads, _ = self._heads(self._globals(eye), torch.zeros(1, 3, dtype=F64))
        self.canonical_heads = heads[0]                             # Anny's frame

    # program globals of the 52 joints (ours)
    @staticmethod
    def _chain(R52):
        Z = [R52[:, 0]]
        for j in range(1, N_JOINTS):
            Z.append(Z[PARENTS52[j]] @ R52[:, j])
        return torch.stack(Z, 1)

    def _globals(self, R52):
        """(N, B, 3, 3) bone rotations G_b = Z_owner(b) A_b in Anny's frame, from local rotations R52 (N, 52, 3, 3)."""
        Z = self._chain(R52)
        Za = self.C.T @ Z @ self.C
        return Za[:, self.owner] @ self.A

    def _heads(self, G, transl):
        """(posed heads (N, B, 3) in Anny's frame, G) for bone rotations G and the program's translation (ours)."""
        h = self.rest_heads
        p = [h[0].expand(G.shape[0], 3) + transl @ self.C]          # transl (ours) in Anny's frame: C^T t
        for b in range(1, len(self.parents)):
            q = self.parents[b]
            p.append(p[q] + torch.einsum("nij,j->ni", G[:, q], h[b] - h[q]))
        return torch.stack(p, 1), G

    def to(self, device=None, dtype=F64):
        """A shallow copy whose arrays live on device in dtype (the speed measure of scripts/openbody_facts.py)."""
        out = object.__new__(OpenBody)
        out.__dict__.update(self.__dict__)
        for k in ("rest_vertices", "rest_heads", "rest_R", "A", "C", "weights", "canonical_heads"):
            setattr(out, k, getattr(self, k).to(device=device, dtype=dtype))
        out.owner, out.mapped = self.owner.to(device), self.mapped.to(device)
        return out

    def rest_joints52(self):
        """(52, 3) m: the program's rest joints (canonical heads of the mapped bones, ours)."""
        return self.canonical_heads[self.mapped] @ self.C.T

    def rest_joints(self):
        """(22, 3) m: the body joints of rest_joints52."""
        return self.rest_joints52()[:N_BODY]

    def canonical_vertices(self):
        """(V, 3) m, ours: the mesh in the canonical rest (every program rotation the identity)."""
        return self.pose(torch.eye(3, dtype=F64).expand(1, N_BODY, 3, 3), torch.zeros(1, 3, dtype=F64))[1][0]

    # ------------------------------------------------------------ the forearm's twist split (design D4, wave 1)
    def twist_shares(self):
        """{bone index: share} of the forearm bones (lowerarm01, lowerarm02 of each side): the fraction of the
        elbow-to-wrist length at the midpoint of the bone's segment (lowerarm01 from the elbow to lowerarm02's head,
        lowerarm02 from there to the wrist), measured on the canonical rest (module docstring, twist split)."""
        if getattr(self, "_shares", None) is None:
            idx = bone_index(self.m)
            h = self.canonical_heads
            out = {}
            for s in ("L", "R"):
                e, w, mid = (h[idx[f"{n}.{s}"]] for n in ("lowerarm01", "wrist", "lowerarm02"))
                L = float((w - e).norm())
                f = float((mid - e) @ (w - e)) / L ** 2
                out[idx[f"lowerarm01.{s}"]] = 0.5 * f
                out[idx[f"lowerarm02.{s}"]] = 0.5 * (f + 1.0)
            self._shares = out
        return self._shares

    def _twist_split(self, G, heads, R52):
        """Bone transforms (rotations (N, B, 3, 3), translations (N, B, 3), Anny's frame) for the skinning, with the
        elbow's axial rotation spread over the forearm bones: a forearm bone of share s turns by the elbow's global
        rotation times a turn of (s - 1) theta about the forearm's rest axis through the elbow, theta the twist of the
        elbow's local rotation about that axis (swing-twist), so that the wrist and the hand keep the full rotation
        and every bone head and program joint stays where the rigid chain puts it."""
        t = heads - torch.einsum("nbij,bj->nbi", G, self.rest_heads)
        if not self.twist_split:
            return G, t
        idx = bone_index(self.m)
        Gs, ts = G.clone(), t.clone()
        hc = self.canonical_heads
        for side, s, elbow in (("left", "L", 18), ("right", "R", 19)):
            e = idx[f"lowerarm01.{s}"]
            u_a = hc[idx[f"wrist.{s}"]] - hc[e]
            u_a = u_a / u_a.norm()
            u = self.C @ u_a                                          # the forearm's rest axis, ours
            R = R52[:, elbow]
            qw = 0.5 * torch.sqrt((1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-30))
            qv = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1) / 4.0
            # (qv / qw) . u is tan(theta / 2) of the twist; written without dividing by qw near a half turn
            theta = 2.0 * torch.atan2(qv @ u, qw)
            for b, share in self.twist_shares().items():
                if self.m.bone_labels[b].endswith(f".{s}"):
                    W = _axis_matrix(u_a, (share - 1.0) * theta)          # (N, 3, 3) Anny's frame
                    Za = G[:, b] @ self.A[b].T                            # C^T Z_elbow C
                    Gs[:, b] = Za @ W @ self.A[b]
                    ts[:, b] = heads[:, e] + torch.einsum("nij,j->ni", Za @ W, hc[b] - hc[e]) - \
                        torch.einsum("nij,j->ni", Gs[:, b], self.rest_heads[b])
        return Gs, ts

    def _skin(self, G, t, w, rest):
        blend_R = torch.einsum("vb,nbij->nvij", w, G)
        return torch.einsum("nvij,vj->nvi", blend_R, rest) + torch.einsum("vb,nbi->nvi", w, t)

    def pose(self, R_body, transl, R_hand=None, vertices=True, chunk=64, ids=None):
        """Joints (N, 52, 3) and vertices (N, V, 3) (None without vertices; with ids (M,), those vertices only,
        (N, M, 3)) in the program's frame, from the local rotations R_body (N, 22, 3, 3) (row 0 the root
        orientation), the translation transl (N, 3) and the hand rotations R_hand (N, 30, 3, 3) (None: at rest).
        The mesh carries the forearm's twist split when self.twist_split (the default)."""
        dt, dev = self.rest_heads.dtype, self.rest_heads.device
        R_body = torch.as_tensor(R_body).to(device=dev, dtype=dt)
        N = R_body.shape[0]
        Rh = torch.eye(3, dtype=dt, device=dev).expand(N, N_JOINTS - N_BODY, 3, 3) if R_hand is None else \
            torch.as_tensor(R_hand).to(device=dev, dtype=dt)
        R52 = torch.cat([R_body, Rh], 1)
        G = self._globals(R52)
        heads, _ = self._heads(G, torch.as_tensor(transl).to(device=dev, dtype=dt))
        joints = heads[:, self.mapped] @ self.C.T
        if not vertices:
            return joints, None
        if ids is None:
            w, rest = self.weights, self.rest_vertices
        else:
            sel = torch.as_tensor(np.asarray(ids, dtype=np.int64).reshape(-1), device=dev)
            w, rest = self.weights[sel], self.rest_vertices[sel]
        out = []
        for a in range(0, N, chunk):
            Gs, ts = self._twist_split(G[a:a + chunk], heads[a:a + chunk], R52[a:a + chunk])
            out.append(self._skin(Gs, ts, w, rest) @ self.C.T)
        return joints, torch.cat(out, 0)

    # ------------------------------------------------------------ the mesh interface of the core (wave 1)
    name = OPEN

    def skin(self, R_body, transl, R_hand=None, ids=None, chunk=64):
        """(N, M, 3) float64 numpy: the vertices ids (None: all) for local rotations R_body (N, 22, 3, 3),
        translations (N, 3) and hand rotations R_hand (N, 30, 3, 3) or None: the package's one skinning call."""
        with torch.no_grad():
            return self.pose(R_body, transl, R_hand, vertices=True, chunk=chunk, ids=ids)[1].cpu().numpy()

    def rest_vertices_np(self):
        """(V, 3) the canonical rest mesh (ours, numpy, cached; read only)."""
        if getattr(self, "_rest_np", None) is None:
            with torch.no_grad():
                v = self.canonical_vertices().cpu().numpy()
            v.setflags(write=False)
            self._rest_np = v
        return self._rest_np

    def surface_mask(self):
        """(V,) bool: the vertices of the mesh's largest connected part, the body's skin (Anny's mesh also holds
        three small inner parts, the mouth cavity and the eyes, design section 6)."""
        if getattr(self, "_surface", None) is None:
            self._surface = largest_part(self.faces, self.n_vertices)
        return self._surface

    def dominant_joint(self):
        """(V,) int64: the program joint (0 to 51) owning each vertex's dominant Anny bone (design D10); -1 on the
        inner parts (surface_mask false), so that no mesh rule of the package selects them."""
        if getattr(self, "_dom", None) is None:
            d = self.owner.cpu().numpy()[dominant_bone(self.m).cpu().numpy()].copy()
            d[~self.surface_mask()] = -1
            d.setflags(write=False)
            self._dom = d
        return self._dom

    @property
    def n_vertices(self):
        return int(self.rest_vertices.shape[0])

    def parents52(self):
        return list(PARENTS52)


# ---------------------------------------------------------------- mesh sets (design section 6)
SOLE_BAND_M = 0.05        # as contact.SOLE_BAND_M: the sole cloud's rest height band above the foot's lowest vertex
HEEL_BAND_M = 0.03        # as contact.HEEL_BAND_M: heel and toe sets within this of the cloud's back or front end
FOOT_BONES = ("foot",) + tuple(f"toe{t}-{k}" for t, n in ((1, 2), (2, 3), (3, 3), (4, 3), (5, 3))
                               for k in range(1, n + 1))


def vertex_sets(ob):
    """{name: vertex ids} on the canonical rest mesh of an OpenBody, every set by a rule on the skinning and the
    geometry (never a typed id): per side the sole cloud, its heel and toe ends, the palmar skin, the thigh skin and
    the five fingertips; the seat; the head's centre and its most forward vertex (the nose)."""
    v = ob.canonical_vertices().numpy()
    dom = dominant_bone(ob.m).numpy()
    idx = bone_index(ob.m)
    J = dict(zip(JOINT_NAMES, ob.rest_joints52().numpy()))
    out = {}
    for side in ("left", "right"):
        s = SIDE_SUFFIX[side]
        ids = np.flatnonzero(np.isin(dom, [idx[f"{b}.{s}"] for b in FOOT_BONES]))
        sole = ids[v[ids, 1] <= v[ids, 1].min() + SOLE_BAND_M]
        out[f"sole_{side}"] = sole
        out[f"heel_{side}"] = sole[v[sole, 2] <= v[sole, 2].min() + HEEL_BAND_M]
        out[f"toe_{side}"] = sole[v[sole, 2] >= v[sole, 2].max() - HEEL_BAND_M]
        hand = np.flatnonzero(np.isin(dom, [idx[f"wrist.{s}"]] + [idx[f"metacarpal{k}.{s}"] for k in (1, 2, 3, 4)]))
        # palm side: below the plane of the wrist and the index and pinky knuckles (palms down in the canonical rest)
        w, i2, i5 = J[f"{side}_wrist"], J[f"{side}_index1"], J[f"{side}_pinky1"]
        n = np.cross(i2 - w, i5 - w)
        n = -np.sign(n[1]) * n / np.linalg.norm(n)              # unit, pointing down (-y)
        out[f"palm_{side}"] = hand[(v[hand] - w) @ n > 0.0]
        out[f"thigh_{side}"] = np.flatnonzero(np.isin(dom, [idx[f"upperleg01.{s}"], idx[f"upperleg02.{s}"]]))
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            out[f"tip_{side}_{f}"] = np.array([fingertip_vertex(f"finger{ANNY_FINGER[f]}-3.{s}",
                                                                ob.rest_vertices, ob.rest_heads, ob.m)])
    seat = np.flatnonzero(np.isin(dom, [idx[b] for b in ("root", "pelvis.L", "pelvis.R", "upperleg01.L",
                                                         "upperleg01.R", "upperleg02.L", "upperleg02.R")]))
    # seated, the back of the thighs and the buttocks bear: the posterior skin (z below the hip joints') of the set
    seat = seat[v[seat, 2] < J["left_hip"][2]]
    out["seat_posterior"] = seat
    head = np.flatnonzero(dom == idx["head"])
    out["head"] = head
    out["nose"] = np.array([head[np.argmax(v[head, 2])]])
    return {k: np.asarray(x, dtype=np.int64) for k, x in out.items()}


# ---------------------------------------------------------------- selector
def reference_allowed(allow_reference=False):
    """True when the caller passes allow_reference=True or the environment sets MDFM_SMPLH_REFERENCE=1."""
    return bool(allow_reference) or os.environ.get(REFERENCE_ENV) == "1"


def require_reference(what, allow_reference=False):
    """Raise OpenBodyRefused unless the reference flag is set (reference_allowed): the SMPL+H and AMASS code of the
    synthetic patient (smplh_reference.py, which never ships) is for the port's comparison only."""
    if not reference_allowed(allow_reference):
        raise OpenBodyRefused(f"{what} reads SMPL+H or AMASS, for the port's comparison only: pass "
                              f"allow_reference=True or set {REFERENCE_ENV}=1")


def backend(name=None, allow_reference=False, phen=NEUTRAL_ADULT):
    """The body backend: OpenBody (the default, or name 'open'); the SMPL+H reference (name 'smplh_reference')
    only with allow_reference=True or MDFM_SMPLH_REFERENCE=1, for the comparison of the port (module docstring)."""
    name = name or os.environ.get(BACKEND_ENV, OPEN)
    if name == OPEN:
        return OpenBody(phen)
    if name == SMPLH_REFERENCE:
        if not (allow_reference or os.environ.get(REFERENCE_ENV) == "1"):
            raise OpenBodyRefused("the SMPL+H reference is for the port's comparison only: pass allow_reference=True "
                                  f"or set {REFERENCE_ENV}=1")
        from .smplh_reference import SMPLHReference
        return SMPLHReference()
    raise ValueError(f"body backend {name!r}, expected {OPEN!r} or {SMPLH_REFERENCE!r}")
