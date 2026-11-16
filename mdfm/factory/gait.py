"""Walking speed and cadence of a body trajectory, for matching rendered walks to real ones.

Speed: horizontal speed of the pelvis relative to the stance foot, valid overground and on a
treadmill (see speed_and_cadence). Cadence: steps per minute from the dominant
frequency of the vertical velocity of the pelvis (two peaks per stride, one per step), in
the band 1.2 to 3.0 Hz, on the body clock. Both are fixed formulas, not learned.
"""
import numpy as np
import torch


def speed_and_cadence(traj, rate=100.0, up=1):
    """(speed m/s, cadence steps/min). Speed is the horizontal speed of the pelvis relative to
    the stance foot (the lower ankle), which equals the walking speed overground (the stance
    foot is still) and the belt speed on a treadmill (the pelvis is still, the stance foot
    moves backward); median over the steady phase (frames above half the 95th percentile)."""
    T = traj.rot6d.shape[0]
    t = torch.arange(0, (T - 1) / rate, 1.0 / rate, dtype=traj.rot6d.dtype)
    ones = torch.ones_like(t)
    (pos, _), (vel, _) = torch.func.jvp(lambda tt: traj.pose_at(tt), (t,), (ones,))
    pos = pos.detach().numpy()
    vel = vel.detach().numpy()
    ank = [7, 8]
    stance = np.where(pos[:, ank[0], up] <= pos[:, ank[1], up], ank[0], ank[1])
    rel = vel[:, 0] - vel[np.arange(len(stance)), stance]
    horiz = np.linalg.norm(np.delete(rel, up, axis=1), axis=1)
    steady = horiz > 0.5 * np.percentile(horiz, 95)
    speed = float(np.median(horiz[steady])) if steady.any() else float(np.median(horiz))
    vz = vel[:, 0, up] - vel[:, 0, up].mean()
    spec = np.abs(np.fft.rfft(vz * np.hanning(len(vz)))) ** 2
    freqs = np.fft.rfftfreq(len(vz), 1.0 / rate)
    band = (freqs >= 1.2) & (freqs <= 3.0)
    step_hz = float(freqs[band][np.argmax(spec[band])]) if band.any() else float("nan")
    return speed, 60.0 * step_hz
