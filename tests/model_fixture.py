"""Hand-made windows for the model tests (B4.15): every stream kind the model reads or scores, with finite
values in contract units, no record opened. make_window(duration) holds an IMU stream (three sites, acc and
gyro, 100 Hz, xsens13 layout), a canonical 3D pose stream (100 Hz, a standing skeleton in torso units with a
0.5 m scale), a canonical 2D pose stream (30 fps, 200 px per torso unit), per-foot force totals (100 Hz, N)
and two EMG log-envelope channels (100 Hz); make_pose_window(duration) the 3D pose stream alone.
"""
import numpy as np

from mdfm.data import windows as W

# a standing canonical skeleton in torso units (pelvis to neck = 1), y up, facing +z
SKELETON15 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.35, 0.05], [0.35, 0.95, 0.0], [-0.35, 0.95, 0.0],
                       [0.4, 0.45, 0.0], [-0.4, 0.45, 0.0], [0.42, 0.0, 0.05], [-0.42, 0.0, 0.05], [0.18, -0.05, 0.0],
                       [-0.18, -0.05, 0.0], [0.2, -0.95, 0.03], [-0.2, -0.95, 0.03], [0.2, -1.85, 0.0],
                       [-0.2, -1.85, 0.0]], np.float32)


def _times(fps, duration, phase=0.0):
    t = phase + np.arange(int(duration * fps) + 1) / fps
    return t[t < duration].astype(np.float32)


def imu_stream(duration, fps=100.0, sites=(2, 7, 8), seed=0, layout="xsens13"):
    rng = np.random.default_rng(seed)
    t = _times(fps, duration, rng.uniform(0, 1 / fps))
    v = np.zeros((len(t), len(sites), 9), np.float32)
    for g in range(len(sites)):
        for c in range(6):
            v[:, g, c] = (1.0 if c == 1 else 0.0) + 0.2 * np.sin(2 * np.pi * (0.9 + 0.3 * c + g) * t + c)
    v[..., 3:6] *= 50.0
    cm = np.zeros((len(sites), 9), bool)
    cm[:, :6] = True
    return W.Stream("imu", "input", float(fps), t, v, cm, np.ones((len(t), len(sites)), bool),
                    meta={"site_ids": list(sites), "acc_content": "with_gravity", "layout": layout,
                          "device": "xsens13"})


def pose3d_stream(duration, fps=100.0, seed=1):
    rng = np.random.default_rng(seed)
    t = _times(fps, duration)
    x = SKELETON15[None] + 0.02 * rng.standard_normal((len(t), 15, 3)).astype(np.float32)
    x[:, 0] = 0.0
    return W.Stream("pose3d", "input", float(fps), t, x.astype(np.float32), np.ones((15, 3), bool),
                    np.ones((len(t), 15), bool), meta={"scale": 0.5, "unit_to_si": 1.0, "dim": 3})


def pose2d_stream(duration, fps=30.0, seed=2):
    rng = np.random.default_rng(seed)
    t = _times(fps, duration)
    x = SKELETON15[None].copy().repeat(len(t), 0)
    x = x + 0.02 * rng.standard_normal(x.shape).astype(np.float32)
    x[..., 2] = 0.0
    x[:, 0] = 0.0
    cm = np.ones((15, 3), bool)
    cm[:, 2] = False
    return W.Stream("pose2d", "input", float(fps), t, x.astype(np.float32), cm, np.ones((len(t), 15), bool),
                    meta={"scale": 200.0, "dim": 2, "camera": "generic"})


def force_stream(duration, fps=100.0):
    t = _times(fps, duration)
    v = np.zeros((len(t), 2, 9), np.float32)
    v[:, 0, 8] = 350.0 + 50.0 * np.sin(2 * np.pi * t)
    v[:, 1, 8] = 350.0 - 50.0 * np.sin(2 * np.pi * t)
    cm = np.zeros((2, 9), bool)
    cm[:, 8] = True
    return W.Stream("force", "target", float(fps), t, v, cm, np.ones((len(t), 2), bool), meta={"unit": "N"})


def emg_stream(duration, fps=100.0):
    t = _times(fps, duration)
    v = np.zeros((len(t), 2, 1), np.float32)
    v[:, 0, 0] = -2.0 + 0.5 * np.sin(2 * np.pi * t)
    v[:, 1, 0] = -1.5 + 0.5 * np.cos(2 * np.pi * t)
    return W.Stream("emg", "target", float(fps), t, v, np.ones((2, 1), bool), np.ones((len(t), 2), bool),
                    meta={"channels": [{"muscle": "tibialis_anterior", "side": "L"},
                                       {"muscle": "soleus", "side": "R"}], "unit": "mV"})


def make_window(duration=2.0, seed=0, person="p0", source="synthetic", sex=1, task=0):
    streams = [imu_stream(duration, seed=seed), pose3d_stream(duration, seed=seed + 1),
               pose2d_stream(duration, seed=seed + 2), force_stream(duration), emg_stream(duration)]
    return W.Window(source, person, f"key{seed}", source == "synthetic", 0.0, float(duration), streams,
                    {"sex": sex, "age_band": W.AGE_UNKNOWN, "task": task})


def make_pose_window(duration=2.0, seed=0, person="p0", source="gait120", sex=0, task=0):
    return W.Window(source, person, f"pose{seed}", False, 0.0, float(duration), [pose3d_stream(duration, seed=seed)],
                    {"sex": sex, "age_band": W.AGE_UNKNOWN, "task": task})
