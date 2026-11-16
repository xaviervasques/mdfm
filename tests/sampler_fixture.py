"""Fake stores and a fake renderer for the sampler, collate and checkpoint tests (B4.17): no record opened.

- build_fake_store(root): a window store built by data/window_store.build_source from fake recordings of five
  pilot sources (IMU of weargait_pd, nonan_gaitprint, nonan_gaitprint_young and sixmwt_lifespan with their
  sites, channels and rates; pose3d of gait120), persons in the train fold and one held-out person per source
  (dev or val); person ids in the namespace 'fake', absent from the index and the lockbox.
- FakeBodyStore: the body-store interface the sampler reads (takes with person_id and n_knots, split,
  windows(takes, starts, win_s) -> BodyWindows): a smooth swinging body walking along +z, deterministic in the
  take and the first knot.
- FakeRenderer: render(bodies, layouts, gen) -> synthetic windows.Window objects with the IMU streams of each
  layout at its band-copy rate (layouts with IMU streams only) and the body truth, deterministic in gen.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from mdfm.body.rotations import axis_angle_to_matrix, matrix_to_rot6d
from mdfm.data import sources as S
from mdfm.data import window_store as WS
from mdfm.data import windows as W
from mdfm.data.body_store import BodyWindows

IMU_SOURCES = {"weargait_pd": (list(range(13)), True, 100.0, "xsens13"),
               "nonan_gaitprint": ([0, 2, 1], False, 200.0, "nonan3"),
               "nonan_gaitprint_young": ([0, 2, 1], False, 200.0, "nonan3"),
               "sixmwt_lifespan": ([2], True, 100.0, "lower_back")}
# person count, recordings per person, seconds per recording
PLAN = {"weargait_pd": (4, 2, 14.0), "nonan_gaitprint": (4, 2, 30.0), "nonan_gaitprint_young": (4, 2, 25.0),
        "sixmwt_lifespan": (4, 1, 40.0), "gait120": (5, 3, 2.3)}
HELD = {"weargait_pd": "dev", "gait120": "dev", "nonan_gaitprint": "val", "nonan_gaitprint_young": "val",
        "sixmwt_lifespan": "val"}
IMU_ONLY = {"lower_back": 1.0, "wrist_L": 1.0, "nonan3": 1.0}


def person(source, i):
    return f"fake:{source}:P{i:02d}"


def _imu_record(source, pid, seconds, seed):
    sites, gyro, fps, _ = IMU_SOURCES[source]
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    x = np.zeros((n, len(sites), 9), np.float32)
    for g in range(len(sites)):
        x[:, g, 0:3] = 0.2 * np.sin(2 * np.pi * (1.0 + 0.1 * g) * t[:, None] + rng.uniform(0, 6, 3))
        x[:, g, 1] += 1.0
        if gyro:
            x[:, g, 3:6] = 40.0 * np.sin(2 * np.pi * 0.9 * t[:, None] + rng.uniform(0, 6, 3))
    cm = np.zeros((len(sites), 9), bool)
    cm[:, 0:3] = True
    cm[:, 3:6] = gyro
    x[:, ~cm] = 0.0
    return {"signal": x, "channel_mask": cm, "location_ids": np.array(sites), "fps": fps,
            "acc_content": "with_gravity", "source": source, "subject_ns": pid}, t


def _paired(source, pid, key, seconds, seed):
    if source in IMU_SOURCES:
        rec, t = _imu_record(source, pid, seconds, seed)
        return S.PairedRecord(source=source, person_id=pid, subject_ns=pid, key=key, streams={"imu": rec},
                              t={"imu": t}, roles={"imu": "input"}, meta={"imu": {"layout": IMU_SOURCES[source][3]}})
    rng = np.random.default_rng(seed)
    n = int(seconds * 100)
    coords = (0.3 * rng.normal(size=(1, 15, 3)) + 0.02 * rng.normal(size=(n, 15, 3))).astype(np.float32)
    coords[:, 0] = 0.0
    pose = {"coords": coords, "joint_mask": np.ones(15, bool), "coord_mask": np.ones(3, bool),
            "valid_mask": np.ones((n, 15), bool), "scale": 0.5, "dim": 3, "modality": "3d_meter", "fps": 100.0,
            "source": source, "subject_ns": pid, "task": key}
    return S.PairedRecord(source=source, person_id=pid, subject_ns=pid, key=key, streams={"pose3d": pose},
                          t={"pose3d": np.arange(n) / 100.0}, roles={"pose3d": "input"},
                          meta={"pose3d": {"layout": "mocap3d", "unit_to_si": 1.0}})


def build_fake_store(root):
    """WindowStore of the fake recordings under root (module docstring)."""
    root = str(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "demographic", lambda source, subject_ns: {})
        for k, (source, (n_p, n_r, sec)) in enumerate(PLAN.items()):
            prs, folds = [], {}
            for i in range(n_p + 1):
                pid = person(source, i)
                folds[pid] = HELD[source] if i == n_p else "train"
                for r in range(n_r):
                    prs.append(_paired(source, pid, f"trial{r:02d}", sec * (1.0 + 0.1 * r), 1000 * k + 10 * i + r))
            WS.build_source(source, out=root, paired=prs, folds=folds)
    return WS.WindowStore(root)


# ---------------------------------------------------------------- bodies
class FakeBodyStore:
    """Body-store interface of the sampler (module docstring)."""
    split = "train"

    def __init__(self, persons=6, takes=2, knots=(240, 700), seed=0):
        rng = np.random.default_rng(seed)
        rows = [{"take_id": f"fake/P{p:02d}/take{k}.npz", "person_id": f"fake:amass:P{p:02d}",
                 "n_knots": int(rng.integers(*knots))} for p in range(persons) for k in range(takes)]
        self.takes = pd.DataFrame(rows)

    def windows(self, takes, starts, win_s=10.0):
        K = int(round(win_s * 100))
        n = len(takes)
        rot = np.zeros((n, K, 22, 3))
        tr = np.zeros((n, K, 3))
        ct = np.zeros((n, K, 2), bool)
        L = np.zeros(n, np.int64)
        for b, (i, s) in enumerate(zip(takes, starts)):
            m = int(self.takes["n_knots"].iloc[int(i)])
            k = min(K, m - int(s))
            L[b] = k
            t = (int(s) + np.arange(K)) / 100.0
            ph = 0.3 * int(i)
            rot[b, :, 1:, 0] = 0.3 * np.sin(2 * np.pi * 1.0 * t + ph)[:, None]
            rot[b, :, 0, 1] = 0.1 * np.sin(2 * np.pi * 0.2 * t)
            tr[b, :, 2] = 1.0 * (t - t[k // 2])
            tr[b, :, 1] = 0.9 + 0.02 * np.sin(2 * np.pi * 2.0 * t)
            ct[b, :, 0] = np.sin(2 * np.pi * 1.0 * t + ph) > 0
            ct[b, :, 1] = ~ct[b, :, 0]
            rot[b, k:], tr[b, k:], ct[b, k:] = rot[b, k - 1], tr[b, k - 1], ct[b, k - 1]
        r6 = matrix_to_rot6d(axis_angle_to_matrix(torch.from_numpy(rot))).float()
        p = self.takes["person_id"].to_numpy()[np.asarray(takes, np.int64)]
        g = torch.tensor([int(x[-2:]) % 2 for x in p])
        return BodyWindows(r6, torch.from_numpy(tr).float(), torch.from_numpy(ct),
                           torch.arange(K)[None] < torch.from_numpy(L)[:, None], torch.from_numpy(L),
                           torch.zeros(n, 16), g, p, torch.zeros(n, dtype=torch.int64),
                           torch.as_tensor(np.asarray(takes, np.int64)), torch.as_tensor(np.asarray(starts, np.int64)))


class FakeRenderer:
    """render(bodies, layouts, gen) (module docstring); IMU layouts only."""

    def render(self, bodies, specs, gen):
        gen = gen if isinstance(gen, np.random.Generator) else np.random.default_rng(gen)
        seeds = gen.integers(0, 2 ** 62, size=len(specs))
        out = []
        for i, lay in enumerate(specs):
            lay = getattr(lay, "layout", lay)
            rng = np.random.default_rng(int(seeds[i]))
            Lk = int(bodies.lengths[i])
            dur = (Lk - 1) / 100.0
            streams = []
            for s in lay.streams:
                if s.modality != "imu":
                    raise ValueError(f"FakeRenderer renders IMU layouts only, not {lay.name}")
                fps = float(s.fps[0])
                k = max(1, int(fps // 100))
                fo = fps / k
                t0 = float(rng.uniform(0, 1 / fps))
                t = t0 + np.arange(int(np.floor((dur - t0) * fo)) + 1) / fo
                t = t[t < dur].astype(np.float32)
                cm = np.zeros((len(s.sites), 9), bool)
                cm[:, :6] = s.chan_mask[:, :6]
                v = (0.1 * rng.standard_normal((len(t), len(s.sites), 9))).astype(np.float32)
                v[..., 1] += 1.0
                v[..., 3:6] *= 100.0
                v[:, ~cm] = 0.0
                S_ = len(s.sites)
                truth = {"mount": np.tile(np.eye(3, dtype=np.float32), (S_, 1, 1)),
                         "bias_acc": np.zeros((S_, 3), np.float32), "bias_gyro": np.zeros((S_, 3), np.float32)}
                streams.append(W.Stream("imu", "input", fo, t, v, cm, np.ones((len(t), S_), bool),
                                        meta={"site_ids": list(s.site_ids), "acc_content": "with_gravity",
                                              "layout": lay.name, "device": lay.device}, truth=truth))
            take = int(bodies.take[i])
            body = {"rot6d": bodies.rot6d[i].numpy(), "transl": bodies.transl[i].numpy(),
                    "contact": bodies.contact[i].numpy(), "valid": bodies.valid[i].numpy(), "lengths": Lk,
                    "betas": bodies.betas[i].numpy(), "gender": int(bodies.gender[i]), "take": take,
                    "take_id": f"take{take}", "start": int(bodies.start[i]), "rate": 100.0, "seed": int(seeds[i])}
            out.append(W.Window("synthetic", str(bodies.person[i]), f"take{take}@{int(bodies.start[i])}", True, 0.0,
                                float(dur), streams, {"sex": int(bodies.gender[i]), "age_band": W.AGE_UNKNOWN,
                                                      "task": int(bodies.task[i])}, body))
        return out
