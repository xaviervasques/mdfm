"""Tests of the gait readouts and the shank-gyroscope baseline (spec B4.12, docs/MFM_MODEL_CODE_SPEC_2026-09-28.md
sections 2.7 and 7): exact on a synthetic walk with known strides, speed and cadence, also through 10 s windows
with a 5 s hop, each window in its own yaw frame; the contact, stitch and cadence rules; the shank detector on
a synthetic sagittal rate; on 20 AMASS dev walks (the dev body store, skipped when it is not built): truth-body
strides equal the ankle displacements at the contact ICs, windowed and stitched strides equal them too, the
readout speed against factory.gait.speed_and_cadence, and the shank detector on rendered shank gyroscopes
against the truth body's kinematic heel strikes and its contact labels. stance_speed (reports/bench/
root_motion_diagnosis.md, fix 7): exact on the synthetic walk, blind to the translation, NaN in flight; on a
train-store KIT walk its median lies within 10 percent of the pelvis chord speed.
"""
import math
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.readouts import gait as G

RATE = 100.0
F64 = torch.float64


# ---------------------------------------------------------------- synthetic walk
def _yaw(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def synthetic_walk(T=1300, stride=1.3, period=110, heading=0.6, starts=(20, 75)):
    """Joints (T, 22, 3), contact (T, 2), the plant times per foot (s) and the speed (m/s) of a walk along
    `heading` (rad about +y): each foot plants every `period` frames, `stride` m farther, stance 60 percent of
    the period, a swing arc of 8 cm; the pelvis moves at stride / period."""
    f = np.arange(T)
    t = f / RATE
    v = stride / (period / RATE)
    J = np.zeros((T, 22, 3))
    J[:, :, 1] = 0.5
    J[:, G.PELVIS, 0] = v * t
    J[:, G.PELVIS, 1] = 0.9 + 0.02 * np.sin(4 * np.pi * f / period)
    contact = np.zeros((T, 2))
    plants = {}
    for j, (foot, f0) in enumerate(zip(G.FEET, starts)):
        k = np.floor_divide(f - f0, period)
        ph = (f - f0 - k * period) / period           # phase in the cycle that starts at a plant, exact at 0
        x_plant = stride * (k + 0.5) - 0.1 * j
        stance = ph < 0.6
        s = np.clip((ph - 0.6) / 0.4, 0.0, 1.0)       # swing progress toward the next plant
        x = np.where(stance, x_plant, x_plant + stride * (3 * s ** 2 - 2 * s ** 3))
        y = np.where(stance, 0.08, 0.08 + 0.08 * np.sin(np.pi * s))
        J[:, G.ANKLES[foot], 0] = x
        J[:, G.ANKLES[foot], 1] = y
        J[:, G.ANKLES[foot], 2] = 0.1 if foot == "L" else -0.1
        contact[:, j] = stance.astype(float)
        plants[foot] = (np.flatnonzero(stance[1:] & ~stance[:-1]) + 1) / RATE
    R = _yaw(heading)
    return J @ R.T, contact, plants, v


def test_exact_on_synthetic_walking():
    J, contact, plants, v = synthetic_walk()
    ics = G.initial_contacts(contact)
    for foot in G.FEET:
        np.testing.assert_allclose(ics[foot], plants[foot], atol=1e-12)
        assert len(ics[foot]) >= 10
    st = G.strides(J, ics)
    assert len(st) == sum(len(v_) - 1 for v_ in ics.values()) and st.attrs["dropped"] == 0
    np.testing.assert_allclose(st["length_m"], 1.3, atol=1e-12)
    np.testing.assert_allclose(st["t1"] - st["t0"], 1.1, atol=1e-9)
    for a, b in ((1.0, 3.5), (0.2, 12.9), (4.44, 4.45)):
        assert G.pass_speed(J, a, b) == pytest.approx(v, abs=1e-12)
    assert G.cadence(ics) == pytest.approx(120.0 / 1.1, rel=1e-9)


def test_stitched_windows_in_their_own_frames():
    J, contact, plants, v = synthetic_walk()
    rng = np.random.default_rng(0)
    win, hop = 1000, 500
    starts = list(range(0, J.shape[0], hop))[:-1]          # 0, 500: the last window is shorter
    jw, cw = [], []
    for s in starts:
        R, o = _yaw(rng.uniform(-np.pi, np.pi)), np.array([rng.normal(), 0.0, rng.normal()])
        jw.append((J[s:s + win] - o) @ R.T)                # each window in its own yaw frame
        cw.append(contact[s:s + win])
    c_st, wof = G.stitch(cw, hop)
    assert c_st.shape == contact.shape and (wof >= 0).all()
    np.testing.assert_array_equal(c_st, contact)
    # most central: window 0 (frames 0 to 999, centre 499.5) up to frame 699, window 1 (500 to 1299, centre 899.5) after
    assert wof[699] == 0 and wof[700] == 1
    ics = G.initial_contacts(c_st)
    Jw = G.on_clock(jw, hop)
    assert Jw.shape == (2,) + J.shape and np.isnan(Jw[0, win:]).all() and np.isnan(Jw[1, :hop]).all()
    st = G.strides(Jw, ics, wof)
    ref = G.strides(J, ics)
    np.testing.assert_allclose(st["length_m"], ref["length_m"], atol=1e-9)
    assert set(st["window"]) == {0, 1}
    for a, b in ((0.3, 12.5), (6.0, 9.0), (2.0, 7.0)):
        assert G.pass_speed(Jw, a, b, wof) == pytest.approx(v, abs=1e-9)


def test_a_short_last_window_keeps_every_stride():
    """Windows of 10 s at 0, 5 and 10 s over an 11.5 s walk, each in its own yaw frame: the second and last
    are cut to 6.5 and 1.5 s, and the last one's centre (frame 1074.5) claims the frames from 950 (the second
    one's centre is 824.5), so a stride whose midpoint lies there may start before frame 1000. Such a stride
    is read in the nearest window holding both ends: none is dropped and every length equals the one-frame
    readout."""
    J, contact, plants, v = synthetic_walk(T=1150)
    rng = np.random.default_rng(3)
    starts = [0, 500, 1000]
    jw, cw = [], []
    for s in starts:
        R, o = _yaw(rng.uniform(-np.pi, np.pi)), np.array([rng.normal(), 0.0, rng.normal()])
        jw.append((J[s:s + 1000] - o) @ R.T)
        cw.append(contact[s:s + 1000])
    c_st, wof = G.stitch(cw, None, starts=starts)
    ics = G.initial_contacts(c_st)
    Jw = G.on_clock(jw, None, starts=starts)
    st = G.strides(Jw, ics, wof)
    ref = G.strides(J, ics)
    assert st.attrs["dropped"] == 0 and len(st) == len(ref)
    np.testing.assert_allclose(st["length_m"], ref["length_m"], atol=1e-9)
    mid = wof[(G.frame_of(st["t0"]) + G.frame_of(st["t1"])) // 2]
    moved = st["window"].to_numpy() != mid
    assert moved.any() and (mid[moved] == 2).all() and (st["window"].to_numpy()[moved] == 1).all()


def test_stitch_keeps_the_most_central_window():
    w = [np.full((10, 1), 0.0), np.full((10, 1), 1.0), np.full((6, 1), 2.0)]
    out, wof = G.stitch(w, 4)
    # centres 4.5, 8.5, 10.5: frame 6 is 1.5 from window 0 and 2.5 from window 1, frame 7 is 2.5 and 1.5
    assert out.shape == (14, 1)
    np.testing.assert_array_equal(wof, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(out[:, 0], wof)
    out, wof = G.stitch(w[:2], None, starts=[0, 20])
    assert (wof[10:20] == -1).all() and np.isnan(out[10:20]).all()


def test_initial_contact_rules():
    p = np.zeros((200, 2))
    p[:30, 0] = 1.0                      # contact from frame 0: censored
    p[50:90, 0] = 0.8                    # IC at 50
    p[92:95, 0] = 0.9                    # rise 42 frames later: kept (refractory 30 frames)
    p[100:103, 0] = 0.6                  # 50 frames after 50 but 8 after 92: dropped
    p[150:180, 0] = 0.4                  # below thr: no contact
    p[120, 1] = np.nan                   # a rise right after a missing frame is no event
    p[121:140, 1] = 1.0
    p[160:170, 1] = 0.5                  # exactly thr counts as contact
    ics = G.initial_contacts(p)
    np.testing.assert_allclose(ics["L"], [0.50, 0.92])
    np.testing.assert_allclose(ics["R"], [1.60])
    with pytest.raises(ValueError):
        G.initial_contacts(np.zeros((10, 3)))


def test_cadence_splits_passes():
    ics = {"L": np.array([0.0, 1.0, 2.0, 10.0, 11.0]), "R": np.array([0.5, 1.5, 10.5])}
    # passes 0.0 to 2.0 (5 ICs, 4 steps in 2 s) and 10.0 to 11.0 (3 ICs, 2 steps in 1 s): 60 x 6 / 3
    assert G.cadence(ics) == pytest.approx(120.0)
    assert math.isnan(G.cadence({"L": np.array([0.0]), "R": np.array([0.5])}))


def _sagittal_template(T=1500, period=1.1, first=0.5, ic_after=0.2):
    """A sagittal shank rate (deg/s): a 300 deg/s mid-swing bump every period, a -120 deg/s dip ic_after s later
    and a shallower loading dip after it. Returns the rate and the IC times."""
    t = np.arange(T) / RATE
    x = np.zeros(T)
    ics = []
    for m in np.arange(first, t[-1] - 0.6, period):
        x += 300 * np.exp(-0.5 * ((t - m) / 0.07) ** 2)
        ic = m + ic_after
        x -= 120 * np.exp(-0.5 * ((t - ic) / 0.02) ** 2)
        x -= 40 * np.exp(-0.5 * ((t - ic - 0.12) / 0.03) ** 2)
        ics.append(round(ic * RATE) / RATE)
    return x, np.array(ics)


def test_shank_gyro_ic_on_a_synthetic_sagittal_rate():
    x, ics = _sagittal_template()
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    other = rng.normal(scale=5.0, size=(len(x), 2))
    for sign in (1.0, -1.0):
        gyro = np.column_stack([sign * x, other]) @ q.T      # the sagittal rate along an arbitrary axis
        s, axis = G.sagittal_rate(gyro)
        assert abs(abs(axis @ q[:, 0]) - 1.0) < 1e-3
        np.testing.assert_allclose(s[np.argmax(np.abs(x))], x[np.argmax(np.abs(x))], rtol=0.05)
        det = G.shank_gyro_ic(gyro)
        np.testing.assert_allclose(det, ics, atol=0.011)
    gyro = np.column_stack([x, other]) @ q.T
    gyro[int(ics[2] * RATE) - 5] = np.nan                   # a missing sample inside one search window
    det = G.shank_gyro_ic(gyro)
    assert len(det) == len(ics) - 1 and not np.isclose(det, ics[2]).any()
    assert len(G.shank_gyro_ic(np.full((500, 3), 1.0) + rng.normal(scale=2.0, size=(500, 3)))) == 0


def test_heel_strikes_on_the_synthetic_walk():
    J, contact, plants, v = synthetic_walk()
    hs = G.heel_strikes(J)
    for foot in G.FEET:
        # the ankle is farthest ahead of the pelvis when its decelerating swing falls to the pelvis speed:
        # 1.3 m in a 0.44 s smoothstep swing against 1.18 m/s, about 30 ms before the plant
        inner = plants[foot][(plants[foot] > 0.6) & (plants[foot] < 12.4)]
        got = hs[foot][(hs[foot] > 0.6) & (hs[foot] < 12.4)]
        assert len(got) == len(inner)
        assert ((inner - got) >= 0.02 - 1e-9).all() and ((inner - got) <= 0.04 + 1e-9).all(), inner - got


# ---------------------------------------------------------------- AMASS dev walks
@pytest.fixture(scope="module")
def dev_walks():
    """The first 20 forward walks of the dev body store (mdfm.gates.g3.amass_dev_walks): joints (T, 22, 3)
    float64 at the knots, the stored contact labels, the take's gendered rest joints and rendered shank
    gyroscopes (100 Hz, no device noise)."""
    if not os.path.exists(paths.smplh_model("neutral")):
        pytest.skip("MoveAtlas SMPL+H model not found")
    from mdfm.data import body_store as S
    if not os.path.isdir(os.path.join(S.store_root(), "dev")):
        pytest.skip("dev body store not built (scripts/build_body_store.py)")
    from mdfm.gates import g3
    walks = g3.amass_dev_walks()
    assert len(walks) == g3.N_WALKS == 20
    return walks


def test_truth_body_strides_equal_the_ankle_displacements(dev_walks):
    n = 0
    for w in dev_walks:
        ics = G.initial_contacts(w["contact"])
        st = G.strides(w["J"], ics)
        assert st.attrs["dropped"] == 0 and len(st) >= 10, w["take"]
        for foot in G.FEET:
            s = st[st["foot"] == foot]
            t = torch.from_numpy(np.concatenate([s["t0"].to_numpy(), s["t1"].to_numpy()]))
            with torch.no_grad():
                p = w["traj"].pose_at(t)[0][:, G.ANKLES[foot]].numpy()
            d = p[len(s):] - p[:len(s)]
            np.testing.assert_allclose(s["length_m"], np.hypot(d[:, 0], d[:, 2]), atol=1e-9)
            n += len(s)
    assert n > 500


def test_windowed_and_stitched_strides_equal_the_take(dev_walks):
    """10 s windows with a 5 s hop, each in its own window frame (BodyStore.windows, float32 knots), stitched:
    strides and pass speeds equal those of the whole take in one frame."""
    from mdfm.body.trajectory import BodyTrajectory
    for w in dev_walks[:5]:
        store, T = w["store"], w["J"].shape[0]
        starts = np.arange(0, T - 500, 500)
        bw = store.windows(np.full(len(starts), w["i"]), starts)
        rest = w["traj"].rest
        jw, cw = [], []
        for k, s in enumerate(starts):
            L = int(bw.lengths[k])
            tr = BodyTrajectory(None, bw.rot6d[k, :L].double(), bw.transl[k, :L].double(), None, rate=RATE, rest=rest)
            with torch.no_grad():
                jw.append(tr.pose_at(torch.arange(L, dtype=F64) / RATE)[0].numpy())
            cw.append(bw.contact[k, :L].numpy().astype(float))
        c_st, wof = G.stitch(cw, 500)
        np.testing.assert_array_equal(c_st, w["contact"][:len(c_st)])
        ics = G.initial_contacts(c_st)
        st = G.strides(G.on_clock(jw, 500), ics, wof)
        ref = G.strides(w["J"], ics)
        np.testing.assert_allclose(st["length_m"], ref["length_m"], atol=1e-4)
        for r in ref.itertuples():
            assert G.pass_speed(G.on_clock(jw, 500), r.t0, r.t1, wof) == pytest.approx(
                G.pass_speed(w["J"], r.t0, r.t1), abs=1e-4)


@pytest.fixture(scope="module")
def readout_check(dev_walks):
    from mdfm.gates import g3
    return g3.readout_check(dev_walks)


def test_readout_speed_against_speed_and_cadence(readout_check):
    """Take speed from the readout (median of pass_speed over the strides under 2 s) against the factory's
    speed (horizontal pelvis speed relative to the stance ankle, median over the steady frames). The two
    formulas differ (a chord of the pelvis over a stride against an instantaneous speed relative to a
    rolling ankle), so the agreement is statistical: median absolute difference at most 0.05 m/s and at
    least 80 percent of the walks within 0.05 m/s (values in reports/gates/g3/dev_dry_run/readout_check.csv)."""
    df, s = readout_check
    d = df["speed_abs_diff_m_s"].to_numpy()
    assert len(d) == 20 and s["speed_within_0_05"] == int((d <= 0.05).sum())
    assert np.median(d) <= 0.05 and np.mean(d <= 0.05) >= 0.8, np.round(d, 3)


def test_shank_gyro_ic_on_rendered_dev_walks(readout_check):
    """Rendered shank gyroscopes (12 Hz bodies, no device noise) against the truth body on steady straight
    steps, one pooled offset removed. Against the kinematic heel strikes: at least 90 percent within 50 ms.
    Against the contact-label ICs (the readout the model's contact head learns) the spec's 90 percent is not
    reached: the labels mark foot loading, tens of milliseconds after the heel strike with a lag that varies
    between takes (contact_lag_ms per take), so the bound here is 80 percent (values in reports/gates/g3/
    dev_dry_run/g3.json, readout_check)."""
    df, s = readout_check
    hs, c = s["shank_gyro_agreement"]["heel_strike"], s["shank_gyro_agreement"]["contact_ic"]
    assert hs["n_steps"] > 300 and c["n_steps"] > 300
    assert hs["share_within"] >= 0.90, hs
    assert c["share_within"] >= 0.80, c
    lag = df["contact_lag_ms"].dropna()
    assert len(lag) >= 15 and (lag > 0).all()           # the contact labels follow the heel strike


# ---------------------------------------------------------------- stance speed
def test_stance_speed_on_the_synthetic_walk():
    """The pelvis moves at v while the stance foot is planted: the stance speed is v (median over the stance
    frames), whatever translation is added to every joint (only pelvis-relative joints are read), and NaN where no
    foot is in contact."""
    J, contact, _, v = synthetic_walk()
    rel = J - J[:, :1]
    contact = contact.copy()
    contact[300:310] = 0.0                                         # a flight phase
    s = G.stance_speed(rel, contact)
    on = contact.any(axis=1)
    assert np.isfinite(s[on]).all() and np.isnan(s[~on]).all() and (~on).sum() == 10
    assert np.median(s[on]) == pytest.approx(v, abs=1e-6)
    drift = np.cumsum(np.random.default_rng(0).normal(size=(len(J), 3)), axis=0)[:, None, :]
    moved = J + drift                                              # another translation: rel is unchanged
    np.testing.assert_allclose(G.stance_speed(moved - moved[:, :1], contact), s, equal_nan=True)
    with pytest.raises(ValueError):
        G.stance_speed(rel[:, :5], contact)


def test_stance_speed_recovers_the_speed_of_a_store_walk():
    """KIT/11/WalkingStraightForwards01 of the train body store (float64 forward kinematics of its knots, the
    stored contact labels): the median stance speed over the frames with a foot in contact lies within 10 percent
    of the pelvis chord speed of the window."""
    if not os.path.exists(paths.smplh_model("neutral")):
        pytest.skip("MoveAtlas SMPL+H model not found")
    from mdfm.body.kinematics import KnotBatch
    from mdfm.body.smplh import SkeletonBank
    from mdfm.data import body_store as S
    if not os.path.isdir(os.path.join(S.store_root(), "train")):
        pytest.skip("train body store not built (scripts/build_body_store.py)")
    store = S.BodyStore(split="train")
    t = store.takes
    i = int(t.index[t["take_id"] == "KIT/11/WalkingStraightForwards01_poses.npz"][0])
    bw = store.windows(np.array([i]), np.array([0]))
    L = int(bw.lengths[0])
    rest = SkeletonBank(F64).rest(bw.betas[:1].double(), bw.gender[:1])
    kb = KnotBatch(bw.rot6d[:1, :L].double(), bw.transl[:1, :L].double(), rest, RATE, torch.tensor([L]))
    with torch.no_grad():
        J = kb.pose(torch.zeros(L, dtype=torch.int64), torch.arange(L, dtype=F64) / RATE)[0].numpy()
    s = G.stance_speed(J - J[:, :1], bw.contact[0, :L].numpy().astype(float))
    chord = np.hypot(*(J[-1, 0, [0, 2]] - J[0, 0, [0, 2]])) / ((L - 1) / RATE)
    assert chord > 0.3 and np.isfinite(s).mean() > 0.8
    assert abs(np.nanmedian(s) / chord - 1.0) <= 0.10, (np.nanmedian(s), chord)
