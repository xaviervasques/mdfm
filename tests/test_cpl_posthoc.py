"""Tests of scripts/cpl_posthoc.py on hand-made signals (no store, no record, no person)."""
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "cpl_posthoc.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t_cpl_posthoc", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_quaternion_product_identity_and_composition():
    q = np.array([math.cos(0.3), math.sin(0.3), 0, 0])
    one = np.array([1.0, 0, 0, 0])
    assert np.allclose(M.quat_mul(q, one), q)
    qq = M.quat_mul(q, q)
    assert np.allclose(qq, [math.cos(0.6), math.sin(0.6), 0, 0])


def test_excursion_planar_and_tilted_axis():
    fps, f0, A = 100.0, 0.9, 35.0
    t = np.arange(0, 1 / f0 + 1e-9, 1 / fps)
    w = A * 2 * math.pi * f0 * np.cos(2 * math.pi * f0 * t)
    for axis in ([0, 0, 1], [0.3, 0.8, -0.5]):
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        g = w[:, None] * ax[None]
        assert M.excursion3d(g, fps) == pytest.approx(2 * A, rel=0.01)
    g = w[:, None] * np.array([0, 0, 1.0])[None]
    g[5, 0] = np.nan
    assert math.isnan(M.excursion3d(g, fps))


def test_excursion_of_a_ramp_and_of_two_axes():
    fps = 100.0
    t = np.arange(0, 1.0 + 1e-9, 1 / fps)
    ax = np.array([0.6, 0.0, 0.8])
    g = 40.0 * np.ones((len(t), 1)) * ax[None]                   # 40 deg/s for 1 s about a fixed axis
    assert M.excursion3d(g, fps) == pytest.approx(40.0, rel=1e-3)
    g2 = np.zeros((len(t), 3))
    g2[: len(t) // 2, 2] = 60.0                                  # 30 deg about z, then 30 deg about x
    g2[len(t) // 2:, 0] = 60.0
    e = M.excursion3d(g2, fps)
    assert 30.0 < e < 60.0                                       # the composed rotation, about 41 deg


def test_window_measures_axes():
    rng = np.random.default_rng(3)
    fps = 100.0
    t = np.arange(0, 4, 1 / fps)
    w = 150 * np.sin(2 * math.pi * 0.9 * t)
    g = np.stack([rng.normal(0, 1, len(t)), rng.normal(0, 1, len(t)), w], 1)
    win = [(0, 111), (111, 222)]
    rom, sax, exc = M.window_measures(g, win, fps)
    assert np.allclose(rom, sax, rtol=0.02)
    assert np.all(exc >= rom * 0.98)


def test_creep_sign():
    n = 120
    J = np.zeros((n, 22, 3))
    V = np.zeros((n, 22, 3))
    contact = np.zeros((n, 2), bool)
    contact[0:60, 0] = True                                     # left stance 0..59
    J[:, 7, 2] = np.linspace(0, 1.2, n)                         # left ankle chord along +z
    J[110:, 7, 2] = 1.2
    V[:, 7, 2] = 0.0
    V[0:60, 7, 2] = -0.05                                       # stance foot creeping backward
    V[0:60, 10, 2] = -0.05
    ca, cf = M.creep(J, V, contact, "L", 0, 110, 1.0)
    assert ca == pytest.approx(-0.05) and cf == pytest.approx(-0.05)
    V[0:60, 7, 2] = 0.0
    V[0:60, 10, 2] = 0.0
    assert M.creep(J, V, contact, "L", 0, 110, 1.0) == (0.0, 0.0)
    contact[:, 0] = False
    assert all(math.isnan(x) for x in M.creep(J, V, contact, "L", 0, 110, 1.0))


def test_swap_columns():
    d = pd.DataFrame({f"{c}_{s}": [1.0 if c == "rom" else 2.0] for c in ("rom", "sax", "exc")
                      for s in ("thigh", "shank", "ankle", "foot")})
    e = M.swap(d, "exc")
    assert (e[[f"rom_{s}" for s in ("thigh", "shank", "ankle", "foot")]] == 2.0).all().all()
    assert (d["rom_thigh"] == 1.0).all()


def test_no_dashes_in_script():
    with open(SCRIPT, encoding="utf-8") as fh:
        text = fh.read()
    assert chr(0x2013) not in text and chr(0x2014) not in text
