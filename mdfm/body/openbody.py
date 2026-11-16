"""The open body of the model track (docs/MODEL_OPEN_PORT_PLAN.md, step M2): Anny's rest skeleton for the program's
22 joints, its phenotype space, and the open body's licence record.

The adapter is shared, not copied: every rest skeleton here comes from mdfm.synth.openbody.OpenBody (the joint map,
the canonical T-pose rest of docs/OPEN_BODY_PORT_DESIGN.md section 3.2, Anny behind its allow list and its licence
guard). This module only reads it.

Phenotype. Anny's six values (PHENOTYPE_KEYS, the order of synth.openbody.Phenotype.as_dict) in [0, 1]: gender
(male 0, female 1), age, muscle, weight, height, proportions. The model reads them inside the box DOMAIN: every
value in [0, 1] but age in [AGE_MIN, 1]. Anny interpolates its age targets between anchors at -1/3, 0, 1/3, 2/3 and
1 (newborn, baby, child, young, old); the design reads 'young' (2/3) as 25 years and 'old' (1) as 90 (design section
4), so AGE_MIN = 1/2, halfway between the child and young anchors, is about 18 years under a linear reading between
them [design: the store's persons are adults]. The model's shape latent is u in R^6 with phenotype =
lo + (hi - lo) sigmoid(u) (to_phenotype, to_latent): a Gaussian on u is the logit-normal of the plan (section 3.1),
and the prior and the posterior head (mdfm/model/prior.PhenotypePrior, posterior.PhenotypeHead) live on u.

Rest skeleton (plan decision P3). rest_joints22(phenotype) is the adapter's OpenBody(phenotype).rest_joints(), one
Anny rest per call (about 0.4 s on the CPU), not differentiable in the phenotype. RestBasis is the joint-space
basis of P3: the multilinear interpolation of rest_joints22 on a grid of phenotype knots (KNOTS), whose node values
scripts/build_open_rest_basis.py computes through the adapter and writes to BASIS_VALUES with its error on a held-out
set of random phenotypes in BASIS_JSON. Anny's rest bone heads are a linear blend of targets whose weights are
piecewise multilinear in the phenotype with kinks at its anchors only, so the interpolation reproduces the bone
heads exactly when the knots hold every anchor inside the domain; the canonical rest straightens the limb segments
(segment lengths are norms), which the held-out error measures. The basis serves when that error stays under
BASIS_TOL_M on every joint (P3); load_basis refuses otherwise, and refuses a file whose SHA-256 differs from its
record. The basis is differentiable in the phenotype (OpenSkeletonBank.rest), and CC0-derived: it may ship.

Licence record (licence_record, check_licence): the three SHA-256 of the licence texts of the installed Anny,
computed as scripts/openbody_audit.py computes them (the wheel's LICENSE, anny/data/mpfb2/LICENSE.md, the README
licence section of the METADATA) and compared with reports/openbody/licence_audit.json, with the version, the cache
data version, the topology and rig names and the SHA-256 of the joint map. The provenance audit of the open model
(mdfm/train/provenance.audit) checks this record in place of the SMPL+H array screen of v1.
"""
import functools
import hashlib
import json
import os

import numpy as np
import torch

from .joints import BODY_JOINTS, N_BODY, PARENTS, fk

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHENOTYPE_KEYS = ("gender", "age", "muscle", "weight", "height", "proportions")
N_PHENO = len(PHENOTYPE_KEYS)
AGE_MIN = 0.5
DOMAIN = {k: ((AGE_MIN, 1.0) if k == "age" else (0.0, 1.0)) for k in PHENOTYPE_KEYS}
# basis knots: every Anny anchor inside DOMAIN (age 2/3; muscle and weight 1/2) and the domain's ends, plus the
# midpoints of gender and height, the two keys whose refinement lowered the held-out error most (a probe of
# 2026-11-12 refining one key at a time from the anchors-only grid, 1.26 mm: height 0.99, gender 1.08, the others
# 1.16 to 1.23 mm)
KNOTS = {"gender": (0.0, 0.5, 1.0), "age": (AGE_MIN, 2.0 / 3.0, 1.0), "muscle": (0.0, 0.5, 1.0),
         "weight": (0.0, 0.5, 1.0), "height": (0.0, 0.5, 1.0), "proportions": (0.0, 1.0)}
BASIS_TOL_M = 1e-3
BASIS_DIR = os.path.join(TREE, "reports", "open_body_rest")
BASIS_VALUES = os.path.join(BASIS_DIR, "rest_basis_values.json")
VALUE_DECIMALS = 7            # the node values written in metres to 0.1 micrometre (the tree versions no .npz)
BASIS_JSON = os.path.join(BASIS_DIR, "rest_basis.json")
LICENCE_AUDIT_JSON = os.path.join(TREE, "reports", "openbody", "licence_audit.json")
BACKEND = "anny"
TOPOLOGY = RIG = "anny"
LOGIT_EPS = 1e-6


class OpenBodyError(RuntimeError):
    """A licence record, a basis file or a phenotype that does not match what the open body requires."""


def _ob():
    from ..synth import openbody as OB
    return OB


# ---------------------------------------------------------------- the joint table on the adapter
def check_joint_table():
    """Raise OpenBodyError unless the adapter's 22 body joints and their parents equal the program's (joints.py)."""
    OB = _ob()
    if tuple(OB.BODY_JOINTS) != tuple(BODY_JOINTS) or tuple(OB.PARENTS22) != tuple(PARENTS):
        raise OpenBodyError("the adapter's body joints or parents differ from mdfm/body/joints.py")
    return True


def joint_map_sha256():
    """SHA-256 of the adapter's joint map (program joint -> Anny bone) and of its owner table of the unmapped bones,
    as canonical JSON."""
    OB = _ob()
    doc = {"joint_map": [list(x) for x in OB.JOINT_MAP], "unmapped_owner": dict(sorted(OB._UNMAPPED_OWNER.items()))}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- phenotype space
def phenotype_array(phen):
    """(6,) float64 in PHENOTYPE_KEYS order from a synth Phenotype, a dict or a sequence."""
    if hasattr(phen, "as_dict"):
        phen = phen.as_dict()
    if isinstance(phen, dict):
        return np.array([float(phen[k]) for k in PHENOTYPE_KEYS], dtype=np.float64)
    a = np.asarray(phen, dtype=np.float64).reshape(-1)
    if a.shape != (N_PHENO,):
        raise ValueError(f"a phenotype has {N_PHENO} values, got {a.shape}")
    return a


def as_phenotype(values):
    """The synth Phenotype of six values in PHENOTYPE_KEYS order."""
    a = phenotype_array(values)
    return _ob().Phenotype(**{k: float(v) for k, v in zip(PHENOTYPE_KEYS, a)})


def _bounds(like):
    lo = torch.tensor([DOMAIN[k][0] for k in PHENOTYPE_KEYS], dtype=like.dtype, device=like.device)
    hi = torch.tensor([DOMAIN[k][1] for k in PHENOTYPE_KEYS], dtype=like.dtype, device=like.device)
    return lo, hi


def to_phenotype(u):
    """(..., 6) latent u -> phenotype lo + (hi - lo) sigmoid(u) inside DOMAIN (module docstring)."""
    lo, hi = _bounds(u)
    return lo + (hi - lo) * torch.sigmoid(u)


def to_latent(p, eps=LOGIT_EPS):
    """(..., 6) phenotype inside DOMAIN -> latent u (the inverse of to_phenotype), clamped eps inside the box.
    A value outside the box raises ValueError."""
    p = torch.as_tensor(p)
    lo, hi = _bounds(p)
    if bool(((p < lo - 1e-9) | (p > hi + 1e-9)).any()):
        raise ValueError(f"phenotype outside the model's domain {DOMAIN}")
    s = ((p - lo) / (hi - lo)).clamp(eps, 1.0 - eps)
    return torch.log(s) - torch.log1p(-s)


# ---------------------------------------------------------------- rest skeleton through the adapter
@functools.lru_cache(maxsize=256)
def _rest_cached(key):
    OB = _ob()
    with torch.no_grad():
        return OB.OpenBody(as_phenotype(key), OB.model()).rest_joints().clone()


def rest_joints22(phen):
    """(22, 3) float64 m: the program's rest joints of one phenotype (synth Phenotype, dict or six values), from the
    adapter (synth.openbody.OpenBody.rest_joints: canonical T-pose rest, program frame y up, face +z). A (B, 6)
    array gives (B, 22, 3). Cached per phenotype rounded to 1e-12."""
    a = phen.as_dict() if hasattr(phen, "as_dict") else phen
    if not isinstance(a, dict):
        arr = np.asarray(torch.as_tensor(a, dtype=torch.float64).detach().cpu().numpy() if torch.is_tensor(a) else a,
                         dtype=np.float64)
        if arr.ndim == 2:
            return torch.stack([rest_joints22(row) for row in arr])
    key = tuple(round(float(x), 12) for x in phenotype_array(a))
    return _rest_cached(key).clone()


# ---------------------------------------------------------------- the joint-space basis (P3)
def grid_nodes(knots=None):
    """(n, 6) float64: every node of the knot grid, the last key varying fastest (the order of RestBasis.values)."""
    knots = KNOTS if knots is None else knots
    axes = [np.asarray(knots[k], dtype=np.float64) for k in PHENOTYPE_KEYS]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], -1)


class RestBasis(torch.nn.Module):
    """Multilinear interpolation of the rest joints on a phenotype knot grid (module docstring). Buffers, all
    non-persistent (the basis is a run-time input with its own hash, not a weight): one knot vector per key and
    values (n1, ..., n6, 22, 3) m. rest(p) for p (..., 6) inside DOMAIN (a value outside is clamped to it)."""

    def __init__(self, knots, values, dtype=torch.float32):
        super().__init__()
        shape = tuple(len(knots[k]) for k in PHENOTYPE_KEYS)
        values = torch.as_tensor(np.asarray(values), dtype=dtype)
        if tuple(values.shape) != shape + (N_BODY, 3):
            raise ValueError(f"basis values {tuple(values.shape)}, expected {shape + (N_BODY, 3)}")
        self.keys = PHENOTYPE_KEYS
        for k in PHENOTYPE_KEYS:
            t = torch.as_tensor(knots[k], dtype=dtype)
            if t.numel() < 2 or bool((t[1:] <= t[:-1]).any()):
                raise ValueError(f"knots of {k} must increase, at least two")
            self.register_buffer(f"knots_{k}", t, persistent=False)
        self.register_buffer("values", values.reshape(-1, N_BODY * 3), persistent=False)
        self.shape = shape
        strides, s = [], 1
        for n in reversed(shape):
            strides.append(s)
            s *= n
        self.strides = tuple(reversed(strides))

    def rest(self, p):
        """(..., 6) phenotypes -> (..., 22, 3) m rest joints, differentiable in p."""
        p = torch.as_tensor(p, dtype=self.values.dtype, device=self.values.device)
        lead = p.shape[:-1]
        p = p.reshape(-1, N_PHENO)
        idx, frac = [], []
        for d, k in enumerate(PHENOTYPE_KEYS):
            t = getattr(self, f"knots_{k}")
            x = p[:, d].clamp(t[0], t[-1])
            i = (torch.searchsorted(t, x.detach().contiguous(), right=True) - 1).clamp(0, t.numel() - 2)
            idx.append(i)
            frac.append((x - t[i]) / (t[i + 1] - t[i]))
        out = 0.0
        for corner in range(1 << N_PHENO):
            w = 1.0
            flat = 0
            for d in range(N_PHENO):
                bit = (corner >> d) & 1
                w = w * (frac[d] if bit else 1.0 - frac[d])
                flat = flat + (idx[d] + bit) * self.strides[d]
            out = out + w[:, None] * self.values[flat]
        return out.reshape(lead + (N_BODY, 3))

    forward = rest


def basis_values(knots=None, progress=None):
    """(n1, ..., n6, 22, 3) float64: rest_joints22 at every node of the knot grid, through the adapter."""
    knots = KNOTS if knots is None else knots
    nodes = grid_nodes(knots)
    out = np.empty((len(nodes), N_BODY, 3))
    for i, row in enumerate(nodes):
        out[i] = rest_joints22(row).numpy()
        if progress is not None:
            progress(i + 1, len(nodes))
    return out.reshape(tuple(len(knots[k]) for k in PHENOTYPE_KEYS) + (N_BODY, 3))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_values(values, path=BASIS_VALUES):
    """Write the node values (n1, ..., n6, 22, 3) m as JSON (shape and the values rounded to VALUE_DECIMALS)."""
    v = np.round(np.asarray(values, dtype=np.float64), VALUE_DECIMALS)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"shape": list(v.shape), "decimals": VALUE_DECIMALS, "values": v.reshape(-1).tolist()}, f)


def read_values(path=BASIS_VALUES):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.asarray(d["values"], dtype=np.float64).reshape(d["shape"])


def load_basis(dtype=torch.float32, npz=BASIS_VALUES, record=BASIS_JSON, check=True):
    """The RestBasis of scripts/build_open_rest_basis.py. With check (the default), refused (OpenBodyError) when
    the file's SHA-256 differs from its record, the record's held-out error reaches BASIS_TOL_M on a joint, its
    knots differ from KNOTS or its joint map from the adapter's."""
    with open(record, encoding="utf-8") as f:
        rec = json.load(f)
    if check:
        if file_sha256(npz) != rec["values_sha256"]:
            raise OpenBodyError(f"{npz}: SHA-256 differs from {record}")
        if not rec["held_out"]["max_joint_error_m"] < BASIS_TOL_M:
            raise OpenBodyError(f"the basis error {rec['held_out']['max_joint_error_m']} m reaches the P3 bound "
                                f"{BASIS_TOL_M} m: use rest_joints22 (Anny's forward)")
        if {k: [float(x) for x in v] for k, v in rec["knots"].items()} != \
                {k: [float(x) for x in v] for k, v in KNOTS.items()}:
            raise OpenBodyError(f"{record}: knots differ from openbody.KNOTS")
        if rec["joint_map_sha256"] != joint_map_sha256():
            raise OpenBodyError(f"{record}: joint map differs from the adapter's")
    return RestBasis(rec["knots"], read_values(npz), dtype)


class OpenSkeletonBank(torch.nn.Module):
    """The open counterpart of smplh.SkeletonBank: rest(phenotype (B, 6)) -> (B, 22, 3) m through the P3 basis
    (differentiable), fk as joints.fk. Its buffers are non-persistent, so its state_dict is empty."""

    def __init__(self, dtype=torch.float32, basis=None):
        super().__init__()
        self.basis = load_basis(dtype) if basis is None else basis
        self.register_buffer("parents", torch.tensor(PARENTS, dtype=torch.int64), persistent=False)

    def rest(self, phenotype):
        return self.basis.rest(phenotype)

    def fk(self, R_local, transl, rest):
        """Forward kinematics (joints.fk), shapes as there."""
        return fk(R_local, transl, rest)


class LatentSkeletonBank(OpenSkeletonBank):
    """The open skeleton in the model's shape coordinates (step M4): rest(u (B, 6), gender=None) -> (B, 22, 3) m,
    u the phenotype latent (to_phenotype maps it into DOMAIN), so that the model, the losses and the synthetic
    factory call it as they call smplh.SkeletonBank.rest(betas, gender). The open body has no gender code: the
    argument is accepted and not read (Anny's gender value is one of the six phenotype values). J_template: the
    basis node values, for the dtype and device checks of the callers."""

    def rest(self, u, gender=None):
        return self.basis.rest(to_phenotype(torch.as_tensor(u, dtype=self.basis.values.dtype,
                                                            device=self.basis.values.device)))

    def rest_phenotype(self, phenotype):
        return self.basis.rest(phenotype)

    @property
    def J_template(self):
        return self.basis.values


# ---------------------------------------------------------------- the licence record
def licence_record():
    """The open body's record (module docstring): backend, version, data version, topology and rig, the three
    licence SHA-256 of the installed Anny (keys of reports/openbody/licence_audit.json 'licences') and the joint
    map SHA-256. Imports anny through the adapter's guard."""
    OB = _ob()
    OB.guard()
    import anny
    from anny.models import model_data
    pkg = os.path.dirname(anny.__file__)
    site = os.path.dirname(pkg)
    info = os.path.join(site, f"anny-{anny.__version__}.dist-info")
    with open(os.path.join(info, "METADATA"), encoding="utf-8") as f:
        meta = f.read()
    section = meta[meta.index("## License"):meta.index("## Citation")]
    names = _audit_licence_names()
    lic = {names[0]: file_sha256(os.path.join(info, "licenses", "LICENSE")),
           names[1]: file_sha256(os.path.join(pkg, "data", "mpfb2", "LICENSE.md")),
           names[2]: hashlib.sha256(section.encode("utf-8")).hexdigest()}
    return {"backend": BACKEND, "anny_version": anny.__version__, "data_version": int(model_data.CURRENT_DATA_VERSION),
            "topology": TOPOLOGY, "rig": RIG, "licences_sha256": lic, "joint_map_sha256": joint_map_sha256()}


def _audit_licences(path=LICENCE_AUDIT_JSON):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["anny_version"], {k: v["sha256"] for k, v in d["licences"].items()}


def _audit_licence_names(path=LICENCE_AUDIT_JSON):
    names = list(_audit_licences(path)[1])
    if len(names) != 3:
        raise OpenBodyError(f"{path}: expected three licence texts, found {len(names)}")
    return names


def check_licence(record=None, path=LICENCE_AUDIT_JSON):
    """Problems (list of str, empty when clean) of a licence record (default: licence_record() of the installed
    Anny) against reports/openbody/licence_audit.json and the adapter's allow list and pin."""
    OB = _ob()
    rec = licence_record() if record is None else record
    version, expected = _audit_licences(path)
    problems = []
    if rec.get("backend") != BACKEND:
        problems.append(f"backend {rec.get('backend')!r}, expected {BACKEND!r}")
    if rec.get("anny_version") != version or rec.get("anny_version") != OB.ANNY_VERSION:
        problems.append(f"anny {rec.get('anny_version')!r}, the audit and the pin say {version!r}, "
                        f"{OB.ANNY_VERSION!r}")
    if rec.get("topology") not in OB.ALLOWED_TOPOLOGIES or rec.get("rig") not in OB.ALLOWED_RIGS:
        problems.append(f"topology {rec.get('topology')!r} or rig {rec.get('rig')!r} outside the allow list")
    got = rec.get("licences_sha256") or {}
    for k, v in expected.items():
        if got.get(k) != v:
            problems.append(f"licence text {k!r}: SHA-256 {got.get(k)!r}, the audit recorded {v!r}")
    for k in sorted(set(got) - set(expected)):
        problems.append(f"licence text {k!r} absent from the audit")
    if rec.get("joint_map_sha256") != joint_map_sha256():
        problems.append("joint map SHA-256 differs from the adapter's")
    return problems
