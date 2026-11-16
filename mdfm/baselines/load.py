"""Reading a G12 baseline checkpoint for its readouts (docs/rules/G12_RULE.md, DRAFT, section 6; task T25).

mdfm/eval/infer.load_checkpoint builds the body model from a checkpoint's run configuration and refuses the combined
configuration a baseline checkpoint stores (config.combined: {'g12', 'run'}). load_frozen reads a baseline
checkpoint through train/checkpoint.load (file, configuration, person-list, records and lockbox hashes, the person
guard, the state_dict audit), rebuilds the G12Model of its configuration and loads the weights asked for: 'ema'
(the default, as J0 reads the body) or 'model'. The JEPA module (target copy and predictor) is left out of the
returned model, as the model EMA leaves it out: a readout reads the tokenizer and the encoder only. The model is
returned in eval mode without gradients. A checkpoint without a G12 configuration (a body checkpoint) is refused
with ValueError.
"""
import json
import os
from dataclasses import dataclass, field

from ..train import checkpoint as CK
from ..train import config as CF
from ..train import provenance as PV
from . import config as GC
from . import model as BM

WEIGHTS = ("ema", "model")


@dataclass
class Frozen:
    """A loaded baseline: model (G12Model, eval mode, no gradient, no JEPA module), path, sha256 (its .sha256 file
    or the file's), weights, arm, step, phase, config (the combined configuration), config_sha256, persons {source:
    ids}, drift (checkpoint.load's)."""
    model: object
    path: str
    sha256: str
    weights: str
    arm: str
    step: int
    phase: str
    config: dict
    config_sha256: str
    persons: dict = field(default_factory=dict)
    drift: list = field(default_factory=list)


def load_frozen(path, weights="ema", device="cpu", records_state=None):
    """Frozen of a G12 checkpoint (module docstring); records_state: the records_sha256 of an archived records state
    the checkpoint is read under (checkpoint.load; the v1 baselines for the port's comparison)."""
    if weights not in WEIGHTS:
        raise ValueError(f"weights {weights!r}, expected one of {WEIGHTS}")
    p = os.path.abspath(str(path))
    ck = CK.load(p, map_location="cpu", records_state=records_state)
    cfg = json.loads(ck["config"])
    if not (isinstance(cfg, dict) and set(cfg) == {"g12", "run"}):
        raise ValueError(f"{p} is not a G12 baseline checkpoint (its configuration has no g12 section)")
    g12 = GC.check(GC.from_dict(cfg["g12"]))
    run = CF.from_dict(cfg["run"])
    model = BM.G12Model.of(g12.arm, run.model.mfm_config(), g12.head)
    state = ck[weights]
    if state is None:
        raise ValueError(f"{p} holds no {weights} state")
    if model.jepa is not None:
        model.jepa = None
        state = {k: v for k, v in state.items() if not k.startswith("jepa.")}
    model.load_state_dict(state, strict=True)
    PV.audit(model)
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    side = p + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            sha = f.read().split()[0]
    else:
        sha = PV.sha256_file(p)
    sched = ck.get("sched") or {}
    return Frozen(model=model, path=p, sha256=sha, weights=weights, arm=g12.arm, step=int(ck["step"]),
                  phase=str(sched.get("in_phase", "")), config=cfg, config_sha256=ck["config_sha256"],
                  persons={s: list(v) for s, v in ck["provenance"]["persons"].items()}, drift=list(ck["drift"]))
