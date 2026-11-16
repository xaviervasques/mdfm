"""The model track's body selector (plan docs/MODEL_OPEN_PORT_PLAN.md step M2; the synthetic patient's design D12).

The open body (mdfm/body/openbody.py, Anny) is the default. The SMPL+H body of v1 (mdfm/body/smplh.py) is reached
through this module only behind the reference flag, for the comparison with v1: the caller passes
allow_reference=True, or the environment sets MDFM_SMPLH_REFERENCE=1 (synth.openbody.REFERENCE_ENV, the one flag
of both tracks). MDFM_MODEL_BODY names the body ('open' or 'smplh'; default 'open').

What reads SMPL+H through here: skeleton_bank('smplh') and the v1 state-dict audit (mdfm/train/provenance.py,
mdfm/model/mfm.smplh_arrays). The v1 model itself (mdfm/model/mfm.MFM with its SkeletonBank) stays as born until
step M4 gives the model the open body and its priors.
"""
import os

OPEN = "open"
SMPLH = "smplh"
BODIES = (OPEN, SMPLH)
MODEL_BODY_ENV = "MDFM_MODEL_BODY"
REFERENCE_ENV = "MDFM_SMPLH_REFERENCE"     # equal to mdfm.synth.openbody.REFERENCE_ENV (tests/test_body_open.py)


class ReferenceRefused(RuntimeError):
    """SMPL+H asked without the reference flag."""


def reference_allowed(allow_reference=False):
    return bool(allow_reference) or os.environ.get(REFERENCE_ENV) == "1"


def require_reference(what, allow_reference=False):
    """Raise ReferenceRefused unless the reference flag is set (module docstring)."""
    if not reference_allowed(allow_reference):
        raise ReferenceRefused(f"{what} reads SMPL+H, which the open model reaches only for the v1 comparison: pass "
                               f"allow_reference=True or set {REFERENCE_ENV}=1")


def model_body(name=None, allow_reference=False):
    """'open' (the default) or 'smplh' (behind the flag); name None reads MDFM_MODEL_BODY."""
    name = name or os.environ.get(MODEL_BODY_ENV) or OPEN
    if name not in BODIES:
        raise ValueError(f"body {name!r}, expected one of {BODIES}")
    if name == SMPLH:
        require_reference("the SMPL+H body", allow_reference)
    return name


def shape_dim(name=None, allow_reference=False):
    """Size of the shape vector of the body: 6 phenotype values (open) or 16 betas (smplh)."""
    if model_body(name, allow_reference) == OPEN:
        from .openbody import N_PHENO
        return N_PHENO
    from .smplh import N_BETAS
    return N_BETAS


def skeleton_bank(name=None, allow_reference=False, dtype=None):
    """The rest skeleton bank of the body: openbody.OpenSkeletonBank (rest(phenotype (B, 6))) or, behind the flag,
    smplh.SkeletonBank (rest(betas, gender))."""
    import torch
    dtype = torch.float32 if dtype is None else dtype
    if model_body(name, allow_reference) == OPEN:
        from .openbody import OpenSkeletonBank
        return OpenSkeletonBank(dtype)
    from .smplh import SkeletonBank
    return SkeletonBank(dtype)
