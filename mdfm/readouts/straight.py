"""The straight-path mask of the G2 walks on the open body (scripts/g2_open_render.straight_mask, step M4 of
docs/MODEL_OPEN_PORT_PLAN.md), on pelvis positions, for the readers that must not import a script or mdfm.gates: the
G3 self-check's draw of the open dev walks (mdfm/gates/g3.open_dev_walks) and the open turning batch of
scripts/check_learnability.py (its inverse: the frames where the walking direction turns).

straight_frames(p_xz, rate): the walking direction d(t), the pelvis's horizontal displacement from t - BASE_S to
t + BASE_S (two strides, so the steps' sway cancels; clipped to the take near its ends), turns by less than DEG_S
between t - 0.5 s and t + 0.5 s (clipped likewise), and the frames within MARGIN_S on both sides do too.
turn_deg(p_xz, rate): that turn (deg) at every frame. tests/test_open_port_followup.py asserts equality with the
script's mask.
"""
import numpy as np

DEG_S = 15.0
BASE_S = 1.0
MARGIN_S = 0.5


def turn_deg(p_xz, rate=100.0):
    """(n,) deg: the turn of the walking direction over the second around each frame (module docstring)."""
    p = np.asarray(p_xz, dtype=np.float64)
    n, b, h = len(p), int(round(BASE_S * rate)), int(round(0.5 * rate))
    k = np.arange(n)
    d = p[np.minimum(k + b, n - 1)] - p[np.maximum(k - b, 0)]
    ang = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    return np.degrees(np.abs(ang[np.minimum(k + h, n - 1)] - ang[np.maximum(k - h, 0)]))


def straight_frames(p_xz, rate=100.0):
    """(n,) bool: the frames on a straight path (module docstring)."""
    from scipy.ndimage import binary_erosion
    m = int(round(MARGIN_S * rate))
    return binary_erosion(turn_deg(p_xz, rate) < DEG_S, structure=np.ones(2 * m + 1, bool), border_value=1)
