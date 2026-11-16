"""Paths of the program. MoveAtlas is read-only; caches live in the gitignored cache/ of this tree.

The gates-only store holds what only mdfm/gates may read (the cached AMASS lockbox takes, and
the SMPL-X takes that no code reads). Its root is MDFM_GATES_STORE when set, else cache/gates_only/
of this tree (gitignored). Once the owner creates <store> (spec question 12) the root moves to
GATES_STORE_TARGET by setting MDFM_GATES_STORE.

The window stores have three roots (window_store_root, task S2.5): stage1 (MDFM_WINDOW_STORE, else
cache/window_store/), s2 (RELEASE stage-2 windows: MDFM_WINDOW_STORE_S2, else cache/window_store_s2/) and internal
(INTERNAL windows: MDFM_WINDOW_STORE_INTERNAL, else cache/window_store_internal/); mdfm.data.window_store keeps
them apart and marks the tier of the last two.
"""
import os

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVEATLAS = os.environ.get("MOVEATLAS_ROOT", "<moveatlas>").replace("\\", "/").rstrip("/")
CACHE = os.path.join(TREE, "cache")
GATES_STORE_ENV = "MDFM_GATES_STORE"
GATES_STORE_TARGET = "<store>/store/lockbox"


def gates_store_root(environ=None):
    """Root of the gates-only store: MDFM_GATES_STORE of `environ` (default os.environ), else
    cache/gates_only/ of this tree; forward slashes, no trailing slash."""
    env = os.environ if environ is None else environ
    root = env.get(GATES_STORE_ENV) or os.path.join(TREE, "cache", "gates_only")
    return root.replace("\\", "/").rstrip("/")


GATES_STORE = gates_store_root()

# Window store roots (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.4, task S2.5): the stage-1 root, never
# rebuilt for stage 2 (its data SHA-256 sits in the sampler state of every stage-1 checkpoint); the RELEASE stage-2
# root; the INTERNAL root. Each one is moved by its variable, else a gitignored folder of cache/.
WINDOW_STORE_KINDS = ("stage1", "s2", "internal")
WINDOW_STORE_ENV = {"stage1": "MDFM_WINDOW_STORE", "s2": "MDFM_WINDOW_STORE_S2",
                    "internal": "MDFM_WINDOW_STORE_INTERNAL"}
WINDOW_STORE_DEFAULT = {"stage1": "window_store", "s2": "window_store_s2", "internal": "window_store_internal"}
WINDOW_STORE_TIER = {"stage1": "RELEASE", "s2": "RELEASE", "internal": "INTERNAL"}


def window_store_root(kind="stage1", environ=None):
    """Root of the window store of `kind` (WINDOW_STORE_KINDS): its variable of `environ` (default os.environ), else
    cache/<WINDOW_STORE_DEFAULT[kind]> of this tree; forward slashes, no trailing slash, nothing created."""
    if kind not in WINDOW_STORE_KINDS:
        raise ValueError(f"window store kind {kind!r}, expected one of {WINDOW_STORE_KINDS}")
    env = os.environ if environ is None else environ
    root = env.get(WINDOW_STORE_ENV[kind]) or os.path.join(CACHE, WINDOW_STORE_DEFAULT[kind])
    return root.replace("\\", "/").rstrip("/")


def smplh_model(gender):
    """Path of the SMPL+H model file for 'male', 'female' or 'neutral' (read-only)."""
    if gender not in ("male", "female", "neutral"):
        raise ValueError(f"unknown gender {gender!r}")
    return f"{MOVEATLAS}/body_models/smplh/{gender}/model.npz"


def cache_dir(*parts):
    p = os.path.join(CACHE, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def _norm(path):
    return os.path.abspath(path).replace("\\", "/").rstrip("/").lower()


def _inside(path, root):
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + "/")


def assert_not_moveatlas(path):
    """Refuse any output path inside the MoveAtlas tree."""
    a = os.path.abspath(path).replace("\\", "/").lower()
    if a.startswith(os.path.abspath(MOVEATLAS).replace("\\", "/").lower()):
        raise PermissionError(f"refusing to write under MoveAtlas: {path}")
    return path


def gates_store_dir(*parts):
    """A folder of the gates-only store, created on demand. Refuses a root inside MoveAtlas, inside
    cache/amass (the training cache) or inside this tree outside cache/ (a versioned folder)."""
    root = GATES_STORE
    assert_not_moveatlas(root)
    if _inside(root, os.path.join(CACHE, "amass")):
        raise PermissionError(f"the gates-only store may not sit inside the training cache: {root}")
    if _inside(root, TREE) and not _inside(root, CACHE):
        raise PermissionError(f"the gates-only store may sit in this tree only under cache/ (gitignored): {root}")
    p = os.path.join(root, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def in_gates_store(path):
    """True when `path` lies inside the gates-only store (the stores of B4.4 and B4.6 refuse it)."""
    return _inside(path, GATES_STORE)


def assert_not_gates_store(path):
    """Refuse a path inside the gates-only store; return it otherwise."""
    if in_gates_store(path):
        raise PermissionError(f"refusing a path inside the gates-only store: {path}")
    return path
