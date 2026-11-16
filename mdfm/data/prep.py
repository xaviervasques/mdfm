"""From native motion to the 100 Hz body clock (design section 1).

Differentiating mocap fits twice amplifies their frame-to-frame jitter (at 120 Hz, 1 mm of
jitter becomes tens of m/s^2), so virtual sensors are rendered from a low-passed body:
rotations in 6D form and the root translation are filtered with a zero-phase Butterworth
filter (default 12 Hz, the usual cutoff for gait kinematics, D. A. Winter, Biomechanics and
Motor Control of Human Movement), then resampled onto the 100 Hz clock by cubic Hermite.
The cutoff is a nuisance parameter of the synthetic factory, recorded with every render.
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from ..body.hermite import hermite
from ..body.rotations import axis_angle_to_matrix, matrix_to_rot6d, Z_UP_TO_Y_UP


def lowpass(x, fps, cutoff, order=4):
    """Zero-phase Butterworth low-pass along axis 0; returned unchanged if too short."""
    if cutoff is None or cutoff >= fps / 2 or x.shape[0] < 3 * (order + 1) * 3:
        return x
    b, a = butter(order, cutoff / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def amass_to_body_clock(seq, rate=100.0, cutoff=12.0, y_up=True, dtype=torch.float64, body=None):
    """AMASS sequence dict (mdfm.data.amass.load_sequence) -> rot6d (T, 22, 6), transl (T, 3),
    betas (16,), on a uniform clock at `rate`, optionally rotated to y-up. The world rotation
    C is applied to the root orientation; with `body` (an SMPLHBody) the translation is
    corrected so that every joint position is exactly rotated, transl' = C (J0 + transl) - J0
    with J0 the shaped rest pelvis; without it the positions carry a constant offset, which no
    derivative-based kernel (IMU) sees but a camera kernel would."""
    fps = seq["fps"]
    aa = torch.as_tensor(seq["poses"][:, :66], dtype=dtype).reshape(-1, 22, 3)
    R = axis_angle_to_matrix(aa)
    tr = torch.as_tensor(seq["trans"], dtype=dtype)
    if y_up:
        C = Z_UP_TO_Y_UP.to(dtype)
        R = torch.cat([(C @ R[:, 0])[:, None], R[:, 1:]], 1)
        if body is None:
            tr = tr @ C.T
        else:
            J0 = body.rest_joints(torch.as_tensor(seq["betas"], dtype=dtype)[None])[0, 0].to(dtype)
            tr = (tr + J0) @ C.T - J0
    r6 = matrix_to_rot6d(R).numpy()
    r6 = lowpass(r6, fps, cutoff)
    trn = lowpass(tr.numpy(), fps, cutoff)
    T_native = r6.shape[0]
    t_end = (T_native - 1) / fps
    t = torch.arange(0, t_end + 1e-9, 1.0 / rate, dtype=dtype)
    r6_c = hermite(torch.as_tensor(r6.copy(), dtype=dtype), t, fps)
    tr_c = hermite(torch.as_tensor(trn.copy(), dtype=dtype), t, fps)
    return r6_c, tr_c, torch.as_tensor(seq["betas"], dtype=dtype)
