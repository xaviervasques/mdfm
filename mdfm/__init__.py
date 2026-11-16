"""MD_FM model code: the measured body and its sensor kernels (design docs/MFM_DESIGN_2026-09-24.md).

Licence note: the internal body model is SMPL+H, loaded at run time from the MoveAtlas tree
(body_models/smplh, MPI licence: non-commercial research on our own machines). No SMPL+H
array is ever copied into this tree, cached, or saved into a checkpoint meant for release
(design section 2).
"""
import os

# torch._dynamo, imported by torch optimizers, names its cache folder with getpass.getuser(),
# which needs the Unix module pwd when no user variable is set (the session's Git Bash).
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "torchinductor"))

__version__ = "0.1.0"
