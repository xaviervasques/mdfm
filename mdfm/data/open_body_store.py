"""Reader of the open body store (plan docs/MODEL_OPEN_PORT_PLAN.md, steps M3 and M4; built by
scripts/build_open_body_store.py --pack).

Layout: <root>/train/<source>/rot6d.f32 (N, 22, 6) with row 0 the root orientation, transl.f32 (N, 3) m (the pelvis
is transl plus the person's rest pelvis), contact.u1 (N, 2) [left, right], takes.csv (one row per take: offset and
n_knots into the arrays, the person's six phenotype values, person_weight 1 / n_persons and take_share, the take's
knots over its person's), shard.json; <root>/hashes.json. World: y up, metres, floor at y = 0 per take; a treadmill
take is stored in the lab frame and served in the belt frame (transl - v k / rate, v = (belt_x, 0, belt_z)), as the
v1 store does.

Rest joints: the phenotype of a take through the P3 basis (mdfm.body.openbody.load_basis), so that the kinematics of
a take are those of the fit.

Root: MDFM_OPEN_BODY_STORE, else <store>/bodies_open_v1. The reader checks every array and table against
hashes.json when opened with check=True (the default). The dev store (split 'dev', scripts/build_open_body_store.py
--split dev: the dev folds of the same sources, step M4) lies at MDFM_OPEN_BODY_STORE_DEV, else
<store>/bodies_open_v1_dev, laid out as <root>/dev/<source>/...; it serves the training's dev hook only. The G1
store (split 'g1', --split g1: the stroke patients outside the lockbox, G1's open sets) lies under
cache/g1/g1_open_v1/bodies; only the G1 harness reads it (mdfm/gates/g1_open.py).
"""
import csv
import hashlib
import json
import os

import numpy as np

from ..body.joints import N_BODY

STORE_ENV = "MDFM_OPEN_BODY_STORE"
STORE_TARGET = "<store>/bodies_open_v1"
DEV_STORE_TARGET = "<store>/bodies_open_v1_dev"
G1_STORE_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache",
                               "g1", "g1_open_v1", "bodies").replace(chr(92), "/")
SPLITS = ("train", "dev", "g1")
RATE = 100.0
ARRAYS = {"rot6d": ("rot6d.f32", "<f4", (N_BODY, 6)), "transl": ("transl.f32", "<f4", (3,)),
          "contact": ("contact.u1", "u1", (2,))}
PHENOTYPE_KEYS = ("gender", "age", "muscle", "weight", "height", "proportions")


def store_root(root=None, split="train"):
    if split not in SPLITS:
        raise ValueError(f"split {split!r}, expected one of {SPLITS}")
    env = os.environ.get({"train": STORE_ENV, "dev": STORE_ENV + "_DEV", "g1": STORE_ENV + "_G1"}[split])
    default = {"train": STORE_TARGET, "dev": DEV_STORE_TARGET, "g1": G1_STORE_TARGET}[split]
    return (root or env or default).replace("\\", "/").rstrip("/")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class OpenBodyStore:
    """One split of the open body store (train, or dev at its own root): takes (list of dicts, one per takes.csv row, with 'source'),
    arrays(source) the memmaps, take(i) one take's arrays, rest(i) its rest joints, persons and weights."""

    def __init__(self, root=None, check=True, split="train"):
        self.split = split
        self.root = store_root(root, split)
        with open(os.path.join(self.root, "hashes.json"), encoding="utf-8") as f:
            self.hashes = json.load(f)
        if self.hashes.get("split", "train") != split:
            raise ValueError(f"{self.root}: a store of split {self.hashes.get('split')!r}, asked {split!r}")
        self.sources = sorted({k.split("/")[1] for k in self.hashes["files"] if k.startswith(split + "/")})
        if check:
            bad = [k for k, v in self.hashes["files"].items() if _sha(os.path.join(self.root, k)) != v]
            if bad:
                raise ValueError(f"{self.root}: files differ from hashes.json: {bad}")
        self.takes = []
        for s in self.sources:
            with open(os.path.join(self.root, split, s, "takes.csv"), encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    r = dict(r)
                    for k in ("offset", "n_knots", "treadmill", "n_native"):
                        r[k] = int(r[k])
                    for k in PHENOTYPE_KEYS + ("person_weight", "take_share", "belt_x", "belt_z", "floor_m",
                                               "rms_m", "contact_l", "contact_r"):
                        r[k] = float(r[k])
                    self.takes.append(r)
        self._mm = {}
        self._basis = None

    def arrays(self, source):
        if source not in self._mm:
            n = sum(t["n_knots"] for t in self.takes if t["source"] == source)
            out = {}
            for name, (fn, dt, shape) in ARRAYS.items():
                out[name] = np.memmap(os.path.join(self.root, self.split, source, fn), dtype=dt, mode="r",
                                      shape=(n,) + shape)
            self._mm[source] = out
        return self._mm[source]

    def phenotype(self, i):
        t = self.takes[i]
        return np.array([t[k] for k in PHENOTYPE_KEYS], dtype=np.float64)

    def rest(self, i):
        """(22, 3) float64 m: the take's rest joints through the P3 basis."""
        import torch
        from ..body import openbody as BO
        if self._basis is None:
            self._basis = BO.load_basis(torch.float64)
        with torch.no_grad():
            return self._basis.rest(torch.as_tensor(self.phenotype(i))[None])[0].numpy()

    def take(self, i, belt_frame=True):
        """{'rot6d' (T, 22, 6) float32, 'transl' (T, 3) float64 (belt frame for a treadmill take), 'contact' (T, 2)
        uint8, 'row'}."""
        t = self.takes[i]
        a = self.arrays(t["source"])
        sl = slice(t["offset"], t["offset"] + t["n_knots"])
        tr = np.asarray(a["transl"][sl], dtype=np.float64)
        if belt_frame and t["treadmill"]:
            k = np.arange(t["n_knots"], dtype=np.float64)[:, None]
            tr = tr - k / RATE * np.array([t["belt_x"], 0.0, t["belt_z"]])
        return {"rot6d": np.asarray(a["rot6d"][sl]), "transl": tr, "contact": np.asarray(a["contact"][sl]),
                "row": t}

    def persons(self):
        """{(source, person_id): [take indices]}."""
        out = {}
        for i, t in enumerate(self.takes):
            out.setdefault((t["source"], t["person_id"]), []).append(i)
        return out

    def knot_weights(self, i):
        """The weight of one knot of take i when the store is drawn by person: person_weight x take_share / knots."""
        t = self.takes[i]
        return t["person_weight"] * t["take_share"] / t["n_knots"]


# ---------------------------------------------------------------- the training view (step M4)
FIT_MEASURE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "reports",
                           "open_body_store", "fit_measure.json")
SEX_UNKNOWN = 2


def residual_bound_m(path=FIT_MEASURE):
    """The store's residual bound (m) of reports/open_body_store/fit_measure.json (16.7 mm, kept on 2026-11-13)."""
    with open(path, encoding="utf-8") as f:
        return float(json.load(f)["residual_bound_m"])


class OpenTrainStore:
    """The open body store as the v1 BodyStore serves the synthetic factory and the mixture sampler
    (data/body_store.BodyStore: takes, persons, take, windows, sample; split 'train', or 'dev' for the training's dev
    hook, the dev store at its own root), the takes whose RMS residual
    exceeds the store's bound refused (decided on 2026-11-13: the reader excludes them, they are counted in
    n_refused_bound). The shape vector of a window is the phenotype latent u (6,) of mdfm/body/openbody.py
    (to_latent of the take's fitted phenotype), in BodyWindows.betas; its gender column is the sex code, 2
    (unknown) for every take: the phenotype prior is the pooled row and stage 1 anchors no sex (plan section 4.2,
    M4). Rest skeletons through the P3 basis, as the fit. Task ids under the vocabulary (windows.task_id of the
    take's source and key; the vocabulary's default where no rule of the source matches)."""

    def __init__(self, root=None, check=True, bound_m=None, rate=RATE, split="train"):
        import pandas as pd
        import torch
        from ..body import openbody as BO
        from .windows import task_id
        self.split = split
        self.store = OpenBodyStore(root, check=check, split=split)
        self.root, self.rate = self.store.root, float(rate)
        self.bound_m = residual_bound_m() if bound_m is None else float(bound_m)
        keep = [i for i, t in enumerate(self.store.takes) if t["rms_m"] <= self.bound_m]
        self.n_refused_bound = len(self.store.takes) - len(keep)
        self.index = np.asarray(keep, dtype=np.int64)
        rows = [self.store.takes[i] for i in keep]
        t = pd.DataFrame(rows)
        t["task"] = [task_id(r["source"], r["key"]) for r in rows]
        self.takes = t.reset_index(drop=True)
        phen = np.array([[r[k] for k in PHENOTYPE_KEYS] for r in rows], dtype=np.float64)
        self.phenotype = phen
        self.betas = BO.to_latent(torch.as_tensor(phen)).numpy()
        self.gender = np.full(len(rows), SEX_UNKNOWN, np.int64)
        basis = BO.load_basis(torch.float64)
        with torch.no_grad():
            self.rest_joints = basis.rest(torch.as_tensor(phen)).numpy()
        self.rest_root = self.rest_joints[:, 0]
        bx, bz = t["belt_x"].to_numpy(np.float64), t["belt_z"].to_numpy(np.float64)
        self.belt = np.stack([bx, np.zeros_like(bx), bz], 1) * (t["treadmill"].to_numpy()[:, None] > 0)
        self.persons = np.array(sorted(self.takes["person_id"].unique()))
        by = self.takes.groupby("person_id").indices
        self._person_takes = [np.asarray(by[p]) for p in self.persons]

    def __len__(self):
        return len(self.takes)

    def _raw(self, i):
        r = self.takes.iloc[int(i)]
        a = self.store.arrays(r["source"])
        o, n = int(r["offset"]), int(r["n_knots"])
        return {k: v[o:o + n] for k, v in a.items()}

    def take(self, i):
        """Take i (floor at y = 0, belt frame for a treadmill take): rot6d (T, 22, 6), contact (T, 2), transl (T, 3)
        float64 m, belt (3,)."""
        a = self._raw(i)
        v = self.belt[int(i)]
        k = np.arange(len(a["transl"]), dtype=np.float64)[:, None] / self.rate
        return {"rot6d": a["rot6d"], "transl": np.asarray(a["transl"], dtype=np.float64) - k * v,
                "contact": a["contact"], "belt": v.copy()}

    def windows(self, takes, starts, win_s=10.0):
        """data.body_store.BodyWindows of the given take rows and first knots in the window frame, as
        BodyStore.windows; betas (n, 6) the phenotype latent, gender (n,) the sex code."""
        import torch
        from ..body import frames
        from ..body.rotations import rot6d_to_matrix
        from .body_store import BodyWindows
        K = int(round(win_s * self.rate))
        takes = np.asarray(takes, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        n = len(takes)
        r6 = np.empty((n, K, N_BODY, 6), np.float32)
        tr = np.empty((n, K, 3), np.float64)
        ct = np.zeros((n, K, 2), bool)
        L = np.empty(n, np.int64)
        for b, (i, s) in enumerate(zip(takes, starts)):
            a = self._raw(i)
            m = len(a["transl"])
            if not 0 <= s < m:
                raise ValueError(f"start {s} outside take {i} of {m} knots")
            k = min(K, m - s)
            L[b] = k
            r6[b, :k], tr[b, :k], ct[b, :k] = a["rot6d"][s:s + k], a["transl"][s:s + k], a["contact"][s:s + k] > 0
            tr[b, :k] -= (np.arange(s, s + k, dtype=np.float64)[:, None] / self.rate) * self.belt[i]
            r6[b, k:], tr[b, k:] = r6[b, k - 1], tr[b, k - 1]
        rot6d, transl = torch.from_numpy(r6).double(), torch.from_numpy(tr)
        rest_root = torch.from_numpy(self.rest_root[takes])
        c = torch.from_numpy(L // 2)
        rows = torch.arange(n)
        R_c = rot6d_to_matrix(rot6d[rows, c, 0])
        pelvis = transl[rows, c] + rest_root
        R_yaw, origin = frames.heading_frame(pelvis[:, None, None], R_c[:, None], 0)
        rot6d, transl = frames.frame_knots(rot6d, transl, rest_root, R_yaw, origin)
        valid = torch.arange(K)[None] < torch.from_numpy(L)[:, None]
        t = self.takes.iloc[takes]
        return BodyWindows(rot6d.float(), transl.float(), torch.from_numpy(ct), valid, torch.from_numpy(L),
                           torch.from_numpy(self.betas[takes]).float(), torch.from_numpy(self.gender[takes]),
                           t["person_id"].to_numpy(), torch.tensor(t["task"].to_numpy(np.int64)),
                           torch.from_numpy(takes.copy()), torch.from_numpy(starts.copy()))

    def sample(self, gen, n, win_s=10.0):
        """n windows drawn by person (a person uniformly, a take with probability proportional to its knots, a first
        knot uniformly among those that fit a whole window), as BodyStore.sample."""
        gen = np.random.default_rng(gen) if not isinstance(gen, np.random.Generator) else gen
        K = int(round(win_s * self.rate))
        nk = self.takes["n_knots"].to_numpy(np.int64)
        takes, starts = np.empty(n, np.int64), np.empty(n, np.int64)
        for b, p in enumerate(gen.integers(len(self.persons), size=n)):
            cand = self._person_takes[p]
            i = cand[gen.choice(len(cand), p=nk[cand] / nk[cand].sum())]
            takes[b], starts[b] = i, gen.integers(max(nk[i] - K, 0) + 1)
        return self.windows(takes, starts, win_s)

    def person_list(self):
        """Sorted person ids of the served takes."""
        return [str(p) for p in self.persons]

    def hashes(self):
        """The store's hashes.json SHA-256 and the served person list's SHA-256 (provenance)."""
        doc = json.dumps(self.store.hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        persons = "\n".join(self.person_list()).encode("utf-8")
        return {"store_hashes_sha256": hashlib.sha256(doc).hexdigest(),
                "train_persons_sha256": hashlib.sha256(persons).hexdigest(), "bound_m": self.bound_m,
                "n_takes": int(len(self.takes)), "n_refused_bound": int(self.n_refused_bound)}
