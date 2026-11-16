"""The body of a run (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4): which body model a run configuration trains, and
the run-time inputs that follow from it.

A run configuration names its body in model.overrides.body: 'open' (Anny, mdfm/body/openbody.py, the open
configuration configs/stage1_open.yaml) or, when absent, 'smplh' (the v1 configurations as born, SMPL+H and the
AMASS body store, reachable only behind the reference flag of mdfm/body/backend.py, which MFM checks when it builds
the SMPL+H skeleton).

For each body, OPEN_INPUTS and V1_INPUTS name the run-time inputs (tree-relative): the shape and mean-pose priors,
the window priors, the mount priors, the layouts and the joint-limit table of the plausibility term. The open ones
are all built from the open body store and the open body (reports/open_body_priors, reports/open_mount_priors,
reports/open_layouts); none is derived from AMASS or SMPL+H.

train_body_store(cfg) opens the train body store of the run (OpenTrainStore of mdfm/data/open_body_store.py, or the
v1 BodyStore), synth_sampler(cfg, store, device) the synthetic renderer on that body (the open skeleton in the
phenotype latent, the open mount priors and layouts), mount_priors(cfg) the priors the nuisance heads load.
"""
import os

from .. import paths
from ..body import backend as BK

OPEN, SMPLH = BK.OPEN, BK.SMPLH
OPEN_INPUTS = {"priors": "reports/open_body_priors/priors.json",
               "window_priors": "reports/open_body_priors/window_priors.json",
               "mount_priors": "reports/open_mount_priors/priors.json",
               "layouts": "reports/open_layouts/layouts.json",
               "plausibility_table": "reports/open_body_priors/plausibility_table.json"}
V1_INPUTS = {"priors": "reports/body_store/priors.json",
             "window_priors": "reports/body_store/window_priors.json",
             "mount_priors": "reports/mount_priors/priors.json",
             "layouts": "reports/layouts/layouts.json",
             "plausibility_table": "reports/body_store/priors.json"}


def tree_path(rel):
    return os.path.join(paths.TREE, *str(rel).split("/"))


def body_of(cfg):
    """'open' or 'smplh': the body of a run configuration (train/config.RunConfig) or of an MFMConfig."""
    mc = getattr(cfg, "model", None)
    if mc is not None and hasattr(mc, "overrides"):
        return str(mc.overrides.get("body", SMPLH))
    return str(getattr(cfg, "body", SMPLH))


def inputs(body):
    """{name: absolute path} of the run-time inputs of a body (module docstring)."""
    table = OPEN_INPUTS if body == OPEN else V1_INPUTS
    return {k: tree_path(v) for k, v in table.items()}


def mount_priors(body, dtype=None, device=None):
    """The kernels.nuisance.MountPriors of a body."""
    import torch
    from ..kernels.nuisance import MountPriors
    return MountPriors.load(inputs(body)["mount_priors"], dtype=torch.float64 if dtype is None else dtype,
                            device=device)


def layouts(body):
    """The factory layouts ({name: Layout}) of a body."""
    from ..factory import layouts as LY
    return LY.load_layouts(inputs(body)["layouts"])


def train_body_store(cfg, root=None):
    """The train body store of a run: OpenTrainStore for the open body, the v1 BodyStore otherwise."""
    if body_of(cfg) == OPEN:
        from ..data.open_body_store import OpenTrainStore
        return OpenTrainStore(root)
    from ..data import body_store as BS
    return BS.BodyStore(split="train")


def dev_body_store(cfg, root=None):
    """The dev body store of a run's dev hook: OpenTrainStore(split='dev') (the open store's dev folds,
    scripts/build_open_body_store.py --split dev) for the open body, the v1 BodyStore's dev split otherwise."""
    if body_of(cfg) == OPEN:
        from ..data.open_body_store import OpenTrainStore
        return OpenTrainStore(root, split="dev")
    from ..data import body_store as BS
    return BS.BodyStore(split="dev")


def skeleton(body, dtype=None):
    """The skeleton bank the synthetic factory renders with: openbody.LatentSkeletonBank (rest(u, gender)) for the
    open body, smplh.SkeletonBank behind the reference flag."""
    import torch
    dtype = torch.float32 if dtype is None else dtype
    if body == OPEN:
        from ..body.openbody import LatentSkeletonBank
        return LatentSkeletonBank(dtype)
    return BK.skeleton_bank(SMPLH, dtype=dtype)


def synth_sampler(cfg, store, device=None, layouts_=None, **kw):
    """factory.synth.SynthSampler on the run's body: for the open body the latent skeleton, the open mount priors and
    the open layouts (unless layouts_ is given); for v1 the sampler's own defaults."""
    from ..factory import synth as SY
    if body_of(cfg) != OPEN:
        return SY.SynthSampler(store, device=device, layouts=layouts_, **kw)
    return SY.SynthSampler(store, skel=skeleton(OPEN), layouts=layouts(OPEN) if layouts_ is None else layouts_,
                           priors=mount_priors(OPEN), device=device, **kw)
