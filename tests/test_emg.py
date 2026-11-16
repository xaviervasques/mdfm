"""Tests of the EMG kernel (spec B4.9, docs/MFM_MODEL_CODE_SPEC_2026-09-28.md sections 2.3 and 7): the muscle
action table (one row per muscle of the MoveAtlas vocabulary, every row citing its source); the decoder is
monotone in |tau|; emg_nll is invariant to a per-record, per-channel log gain; unknown muscles are masked;
channels are indexed by column, duplicated muscles included; the log-envelope band copy; the body enters
detached; real gait120 and stroke able-bodied records map channel by channel; the reader refuses CODY.
"""
import os
import re

import numpy as np
import pytest
import torch
import yaml

from mdfm import paths
from mdfm.kernels import emg as E
from mdfm.kernels.dynamics import MOMENT_CHANNELS

F64 = torch.float64
VOCAB = os.path.join(paths.MOVEATLAS, "configs", "canonical_emg.yaml")
LEG = {"tibialis_anterior", "gastrocnemius", "gastrocnemius_lateralis", "gastrocnemius_medialis", "soleus",
       "soleus_medialis", "soleus_lateralis", "peroneus_longus", "peroneus_brevis", "rectus_femoris",
       "vastus_lateralis", "vastus_medialis", "biceps_femoris", "semitendinosus", "hamstrings", "gluteus_medius"}


def _vocabulary():
    """The muscles list of canonical_emg.yaml (read-only; the flow sequence alone, since other blocks of the
    contract are not strict YAML)."""
    with open(VOCAB, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^muscles:\s*\[(.*?)\]", text, re.S | re.M)
    return yaml.safe_load("[" + m.group(1) + "]")


# ---------------------------------------------------------------- the action table
@pytest.mark.skipif(not os.path.exists(VOCAB), reason="MoveAtlas canonical_emg.yaml not found")
def test_table_rows_equal_the_contract_vocabulary():
    assert list(E.load_table().names) == _vocabulary()


def test_every_row_cites_its_source_and_actions_are_well_formed():
    doc = E.read_actions()
    refs = doc["references"]
    assert all(isinstance(v, str) and len(v) > 20 for v in refs.values())
    used = set()
    for row in doc["muscles"]:
        assert row["sources"], row["muscle"]
        assert set(row["sources"]) <= set(refs), row["muscle"]
        used |= set(row["sources"])
        for act in row["actions"]:
            assert len(act.split()) == 2, (row["muscle"], act)
    assert used == set(refs)
    with open(E.ACTIONS_PATH, encoding="utf-8") as f:
        assert chr(0x2014) not in f.read()
    table = E.load_table()
    assert {n for n, m in zip(table.names, table.modelled) if m} == LEG
    a = dict(zip(table.names, table.action))
    # agonist signs on (hip flexion, knee flexion, ankle dorsiflexion, hip abduction)
    assert a["tibialis_anterior"].tolist() == [0, 0, 1, 0]
    assert a["gastrocnemius_medialis"].tolist() == [0, 1, -1, 0]
    assert a["rectus_femoris"].tolist() == [1, -1, 0, 0]
    assert a["biceps_femoris"].tolist() == [-1, 1, 0, 0]
    assert a["vastus_lateralis"].tolist() == [0, -1, 0, 0] and a["soleus"].tolist() == [0, 0, -1, 0]
    assert a["gluteus_medius"].tolist() == [0, 0, 0, 1] and not a["erector_spinae"].any()


def test_leg_action_matrix_and_misspelled_actions_refused():
    table = E.load_table()
    got = {n: table.action[i].tolist() for i, n in enumerate(table.names) if table.modelled[i]}
    # (hip flexion, knee flexion, ankle dorsiflexion, hip abduction) of every modelled muscle
    plantar, biarticular_hamstring = [0, 0, -1, 0], [-1, 1, 0, 0]
    want = {"tibialis_anterior": [0, 0, 1, 0], "gastrocnemius": [0, 1, -1, 0], "gastrocnemius_lateralis": [0, 1, -1, 0],
            "gastrocnemius_medialis": [0, 1, -1, 0], "soleus": plantar, "soleus_medialis": plantar,
            "soleus_lateralis": plantar, "peroneus_longus": plantar, "peroneus_brevis": plantar,
            "rectus_femoris": [1, -1, 0, 0], "vastus_lateralis": [0, -1, 0, 0], "vastus_medialis": [0, -1, 0, 0],
            "biceps_femoris": biarticular_hamstring, "semitendinosus": biarticular_hamstring,
            "hamstrings": biarticular_hamstring, "gluteus_medius": [0, 0, 0, 1]}
    assert got == want
    # a misspelled hip, knee or ankle action raises instead of dropping out of the decoder
    with pytest.raises(ValueError):
        E.MuscleTable([{"muscle": "soleus", "actions": ["ankle plantarflexon"]}])
    with pytest.raises(ValueError):
        E.MuscleTable([{"muscle": "rectus_femoris", "actions": ["hip flexion", "knee extention"]}])
    with pytest.raises(ValueError):
        E.MuscleTable([{"muscle": "m", "actions": ["knee flexion", "knee extension"]}])
    # a listed unmodelled leg action and any non-leg action are recorded without a channel
    t = E.MuscleTable([{"muscle": "biceps_femoris", "actions": ["hip extension", "hip external_rotation"]},
                       {"muscle": "tibialis_anterior", "actions": ["ankle dorsiflexion", "subtalar inversion"]}])
    assert t.action.tolist() == [[-1, 0, 0, 0], [0, 0, 1, 0]]


# ---------------------------------------------------------------- decoder
def _inputs(n, seed, dtype=F64):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 2, 4, generator=g, dtype=dtype) * 1.5, torch.randn(n, 2, 4, generator=g, dtype=dtype) * 0.5,
            torch.randn(n, 2, 4, generator=g, dtype=dtype) * 3)


def _perturbed_decoder(seed):
    torch.manual_seed(seed)
    dec = E.EMGDecoder().double()
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(torch.randn_like(p))
    return dec


def test_decoder_is_monotone_in_the_moment_magnitude():
    dec = _perturbed_decoder(0)
    names = E.load_table().names
    C = len(names)
    muscle = torch.arange(C)
    for side in (0, 1):
        mom, ang, rate = _inputs(400, 1 + side)
        mom.requires_grad_(True)
        out, mask = dec(mom, ang, rate, muscle, torch.full((C,), side), detach_body=False)
        env = torch.where(mask, out.exp(), torch.zeros_like(out))
        for c in range(C):
            if not mask[0, c]:
                continue
            (gr,) = torch.autograd.grad(env[:, c].sum(), mom, retain_graph=True)
            # d env / d |tau_k| = sign(tau_k) d env / d tau_k >= 0 on every channel of the muscle's leg
            assert (torch.sign(mom[:, side]) * gr[:, side] >= -1e-12).all(), names[c]
            assert gr[:, 1 - side].abs().max() == 0                       # the other leg does not drive it
    # a larger |tau| with the same sign never lowers any envelope
    mom, ang, rate = _inputs(300, 5)
    out0, mask = dec(mom, ang, rate, muscle.expand(300, C), torch.zeros(300, C, dtype=torch.int64))
    out1, _ = dec(mom * 1.7, ang, rate, muscle.expand(300, C), torch.zeros(300, C, dtype=torch.int64))
    assert (out1[mask] >= out0[mask] - 1e-12).all()


def test_decoder_parameters_and_the_detached_body():
    dec = E.EMGDecoder()
    M = len(E.load_table().names)
    assert sum(p.numel() for p in dec.parameters()) == M * (len(MOMENT_CHANNELS) + 2) + (8 * 64 + 64) \
        + (64 * 64 + 64) + (64 * M + M)
    assert dec.state_dict().keys() == {"a_raw", "c_raw", "b_raw", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight",
                                       "mlp.2.bias", "mlp.4.weight", "mlp.4.bias"}
    mom, ang, rate = (x.float().requires_grad_(True) for x in _inputs(50, 7))
    ch = E.channel_table([{"muscle": "tibialis_anterior", "side": "R"}, {"muscle": "soleus", "side": "L"}])
    out, mask = dec(mom, ang, rate, torch.as_tensor(ch[0]), torch.as_tensor(ch[1]))
    out.sum().backward()
    assert mom.grad is None and dec.a_raw.grad.abs().sum() > 0          # body detached, decoder trained
    out, _ = dec(mom, ang, rate, torch.as_tensor(ch[0]), torch.as_tensor(ch[1]), detach_body=False)
    out.sum().backward()
    assert mom.grad.abs().sum() > 0
    with pytest.raises(TypeError):
        dec(mom.detach().half(), ang.detach().half(), rate.detach().half(), torch.as_tensor(ch[0]),
            torch.as_tensor(ch[1]))


def test_channels_are_indexed_by_column_with_duplicates_and_unknowns():
    chans = [{"muscle": "biceps_femoris", "side": "L"}, {"muscle": "tibialis_anterior", "side": "R"},
             {"muscle": "biceps_femoris", "side": "L"}, {"muscle": "erector_spinae", "side": "L"},
             {"muscle": "unknown", "side": "R"}, {"muscle": "deltoid_anterior", "side": "L"},
             {"muscle": "not_a_muscle", "side": "L"}, {"muscle": "gastrocnemius", "side": "midline"}]
    mus, side, modelled = E.channel_table(chans)
    names = E.load_table().names
    assert mus[0] == mus[2] == names.index("biceps_femoris") and mus[6] == -1 and side[7] == -1
    assert modelled.tolist() == [True, True, True, False, False, False, False, False]
    # a record column of another contract kind, or unresolved, is not modelled even under a leg muscle name
    rec_chans = [{"kind": "emg", "muscle": "soleus", "side": "L", "resolved": True},
                 {"kind": "unresolved_emg_ecg", "muscle": "soleus", "side": "L", "resolved": True},
                 {"kind": "aux", "muscle": "soleus", "side": "R"},
                 {"kind": "emg", "muscle": "soleus", "side": "R", "resolved": np.False_}]
    assert E.channel_table(rec_chans)[2].tolist() == [True, False, False, False]
    dec = _perturbed_decoder(3)
    mom, ang, rate = _inputs(120, 8)
    pred, mask = dec(mom, ang, rate, torch.as_tensor(mus), torch.as_tensor(side))
    assert torch.equal(mask[0], torch.as_tensor(modelled)) and torch.equal(pred[:, 0], pred[:, 2])
    assert (pred[:, ~torch.as_tensor(modelled)] == 0).all()
    # two electrodes on one muscle keep their own gains; masked columns never enter the likelihood
    g = torch.Generator().manual_seed(9)
    obs = pred + torch.randn(120, 8, generator=g, dtype=F64) * 0.3
    w = torch.arange(120) % 3
    base = E.emg_nll(pred, obs, mask, w, -1.0, n_windows=3)
    obs2 = obs.clone()
    obs2[:, 2] += 4.0                                   # the second biceps electrode reads e^4 times larger
    obs2[:, 3:] = torch.randn(120, 5, generator=g, dtype=F64) * 50
    assert torch.allclose(E.emg_nll(pred, obs2, mask, w, -1.0, n_windows=3), base, rtol=1e-12, atol=1e-10)


def test_emg_nll_invariant_to_a_per_record_channel_log_gain():
    g = torch.Generator().manual_seed(11)
    N, C, W = 600, 5, 6
    pred = torch.randn(N, C, generator=g, dtype=F64)
    obs = pred + torch.randn(N, C, generator=g, dtype=F64) * 0.4
    w = torch.randint(0, W, (N,), generator=g)
    valid = torch.rand(N, C, generator=g) > 0.1
    rec = torch.tensor([0, 0, 1, 2, 2, 2])
    ls = torch.randn(W, C, generator=g, dtype=F64) * 0.2 - 1.0
    base = E.emg_nll(pred, obs, valid, w, ls, record=rec)
    shift = torch.randn(3, C, generator=g, dtype=F64) * 5
    got = E.emg_nll(pred, obs + shift[rec[w]], valid, w, ls, record=rec)
    assert torch.allclose(got, base, rtol=1e-12, atol=1e-9)
    # a time-varying change is not absorbed; the prediction's own constant is absorbed too
    assert not torch.allclose(E.emg_nll(pred, obs + torch.linspace(0, 3, N, dtype=F64)[:, None], valid, w, ls,
                                        record=rec), base)
    assert torch.allclose(E.emg_nll(pred - 2.5, obs, valid, w, ls, record=rec), base, rtol=1e-12, atol=1e-9)
    # the truth plus a gain beats a shuffled prediction; weights scale the per-window sums
    assert E.emg_nll(pred, pred + 1.0, valid, w, ls, record=rec).sum() < \
        E.emg_nll(pred[torch.randperm(N, generator=g)], pred + 1.0, valid, w, ls, record=rec).sum()
    assert torch.allclose(E.emg_nll(pred, obs, valid, w, ls, weight=torch.full((N,), 3.0, dtype=F64), record=rec),
                          3 * base)


def test_emg_nll_gradcheck_and_observations_on_another_device():
    g = torch.Generator().manual_seed(13)
    N, C, W = 90, 3, 4
    pred = torch.randn(N, C, generator=g, dtype=F64).requires_grad_(True)
    obs = pred.detach() + 0.3 * torch.randn(N, C, generator=g, dtype=F64)
    valid = torch.rand(N, C, generator=g) > 0.2
    w = torch.randint(0, W, (N,), generator=g)
    rec = torch.tensor([0, 0, 1, 2])
    ls = (torch.randn(W, C, generator=g, dtype=F64) * 0.3 - 1.0).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: E.emg_nll(a, obs, valid, w, b, record=rec), (pred, ls), eps=1e-6,
                                    atol=1e-5, rtol=1e-4)
    if torch.cuda.is_available():
        ref = E.emg_nll(pred.detach(), obs, valid, w, -1.0, record=rec)
        got = E.emg_nll(pred.detach().cuda(), obs, valid, w, -1.0, record=rec)
        assert torch.allclose(got.cpu(), ref, rtol=1e-10, atol=1e-9)


def test_emg_nll_log_sigma_shapes_and_the_noise_floor():
    g = torch.Generator().manual_seed(12)
    N, C, W = 300, 4, 5
    pred = torch.randn(N, C, generator=g, dtype=F64)
    obs = pred + torch.randn(N, C, generator=g, dtype=F64) * 0.2
    w = torch.arange(N) % W
    valid = torch.ones(N, C, dtype=torch.bool)
    ls = torch.randn(W, generator=g, dtype=F64)
    # one scale per window (W,) equals the same scale on every column (W, C); a scalar equals a constant (W, C)
    assert torch.equal(E.emg_nll(pred, obs, valid, w, ls), E.emg_nll(pred, obs, valid, w, ls[:, None].expand(W, C)))
    assert torch.equal(E.emg_nll(pred, obs, valid, w, -1.0), E.emg_nll(pred, obs, valid, w, torch.full((W, C), -1.0)))
    # the scale never drops below the floor: a very negative log_sigma reads as the floor alone
    at_floor = E.emg_nll(pred, obs, valid, w, -50.0)
    assert torch.allclose(at_floor, E.emg_nll(pred, obs, valid, w, -60.0))
    assert torch.allclose(at_floor, E.emg_nll(pred, obs, valid, w, -50.0, floor=E.FLOOR_LOG_SIGMA))


# ---------------------------------------------------------------- observation side
def test_log_envelope_band_rms_and_floor():
    fps = 1000.0
    t = np.arange(2000) / fps
    rng = np.random.default_rng(0)
    x = np.stack([rng.normal(0, 0.01, 2000), 5.0 * np.sin(2 * np.pi * 5 * t), 1.0 * np.sin(2 * np.pi * 100 * t)], 1)
    burst = (t >= 0.5) & (t < 0.55)
    x[burst, 0] += rng.normal(0, 1.0, burst.sum())
    le, floor = E.log_envelope(x, fps, "raw")
    assert le.shape == x.shape and np.isfinite(le).all()
    peak = np.argmax(le[:, 0])
    assert 0.5 <= t[peak] < 0.55                                   # the burst peaks inside itself
    mid = slice(500, 1500)
    env = np.exp(le[mid])
    assert env[:, 1].max() < 0.01 * env[:, 2].min()                # 5 Hz motion removed, 100 Hz kept
    assert np.allclose(env[:, 2], 1 / np.sqrt(2), rtol=0.03)       # RMS of a unit sine inside the band
    raw_env = E.envelope(x, fps, "raw")
    assert np.allclose(floor, np.percentile(raw_env, 5, axis=0)) and (le >= np.log(floor)[None] - 1e-12).all()
    # a rectified or band-passed input skips the filter; an envelope is only floored and logged
    assert np.allclose(E.envelope(np.abs(x[:, 2:]), fps, "rectified"), E.envelope(x[:, 2:], fps, "bandpassed"))
    le2, _ = E.log_envelope(np.abs(x), fps, "envelope", floor=np.full(3, 1e-3))
    assert np.allclose(le2, np.log(np.maximum(np.abs(x), 1e-3)))
    for bad in (dict(signal_kind="filtered"), dict(fps=60.0)):
        with pytest.raises(ValueError):
            E.log_envelope(x, **{"fps": fps, "signal_kind": "raw", **bad})
    y = x.copy()
    y[3, 0] = np.nan
    with pytest.raises(ValueError):
        E.log_envelope(y, fps, "raw")


def test_log_envelope_equals_the_window_store_band_copy():
    from mdfm.data import window_store as WS
    from mdfm.data.windows import Stream
    for fps in (1000.0, 2000.0):
        T = int(1.5 * fps)
        rng = np.random.default_rng(int(fps))
        x = rng.normal(0, 0.05, (T, 4)) * (1 + 3 * (np.arange(T) / T))[:, None]
        st = Stream(modality="emg", role="target", fps=fps, t=np.arange(T) / fps, values=x[:, :, None],
                    chan_mask=np.ones((4, 1), bool), valid=np.ones((T, 4), bool))
        out = WS.band_copy(st)
        le, _ = E.log_envelope(x, fps, "raw")
        k = max(1, int(fps // 100))
        assert out.values.shape == (len(le[::k]), 4, 1)
        # the store keeps float32: equal to its rounding
        assert np.abs(le[::k] - out.values[:, :, 0]).max() < 1e-5


# ---------------------------------------------------------------- real records
@pytest.fixture(scope="module")
def index():
    from mdfm.data import canonical as C
    if not os.path.exists(C.index_path()):
        pytest.skip("record index not built")
    return C.index()


def test_real_records_map_channel_by_channel(index):
    from mdfm.data import canonical as C
    g = index[(index["source"] == "gait120") & (index["modality"] == "emg") & (index["subject_ns"] == "gait120:S002")
              & index["key"].str.endswith("__level_walking_trial01.npz")]
    s = index[(index["source"] == "stroke_gait_vancriekinge") & (index["modality"] == "emg")
              & (index["subject_ns"] == "stroke_gait:able_bodied:S001") & index["key"].str.endswith("_SUBJ1_1.npz")]
    assert len(g) == 1 and len(s) == 1
    for row, n_modelled in ((g.iloc[0], 12), (s.iloc[0], 12)):
        assert not row["lockbox"] and row["tier"] == "RELEASE"
        rec = C.load(row)
        x = np.asarray(rec["signal"], np.float64)
        le, floor = E.log_envelope(x, float(rec["fps"]), rec["signal_kind"])
        assert le.shape == x.shape and np.isfinite(le).all() and (floor > 0).all()
        mus, side, modelled = E.channel_table(rec["channels"])
        assert (mus >= 0).all() and (side >= 0).all() and modelled.sum() == n_modelled
        names = E.load_table().names
        for c, ch in enumerate(rec["channels"]):
            assert names[mus[c]] == ch["muscle"] and "LR"[side[c]] == ch["side"]
            assert modelled[c] == (ch["muscle"] != "erector_spinae")


def test_the_reader_refuses_cody_emg_rows(monkeypatch):
    from mdfm.data import canonical as C
    import zipfile

    def no_open(*a, **k):
        raise AssertionError("a container was opened")

    monkeypatch.setattr(zipfile, "ZipFile", no_open)
    for src, ns in (("cody_1", "cody_1:P999"), ("cody_3", "cody_external:P999"), ("cody_2", "ext_ds:P999")):
        row = {"source": src, "manifest_source": src, "modality": "emg", "subject_ns": ns, "key": "k",
               "person_id": ns, "tier": "RELEASE", "lockbox": False, "container": "x.npz", "record": 0}
        with pytest.raises(C.Refused):
            C.load(row)
