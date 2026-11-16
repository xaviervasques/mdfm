"""Tests of scripts/f2_precision.py (task F2L1, lever 3: inference precision) on the CPU, with a tiny model and
synthetic events, without a weargait file, a real checkpoint or the GPU:

- the pre-registration: the text, its hash file, the script constant and the task report agree; a changed text stops
  the analysis and --decode before any file of values is opened;
- decode_trial with amp False equals mdfm.eval.infer.G3Predictor on the CPU (the same stitched body);
- the refinement draws do not depend on the precision: under amp True (the bf16 autocast, here on the CPU) and amp
  False, the generator state seen by every Batch.subsample call is the same sequence, while the bodies differ;
- float32_backends turns TF32 off inside the block and restores the flags after, also on an error;
- the analysis: a constant +2 cm fp32 offset reads PRECISION LEVER on the stride line; a 0.1 cm jitter reads NOT A
  LEVER; a large rerun noise raises the threshold and sets the noise note; only units matched in every pass are
  compared; the per-pass lines equal g3_gain_check.lines_for on each pass;
- the reading at its edges (|D| equal to T counts; an interval touching 0 does not);
- a lockbox person or a person outside dev12 in the events file is refused before any value is read;
- --decode with a tiny model and fakes writes the three passes in the events.csv columns, then the analysis (on the
  CPU every pass runs in float32, so D and D0 are 0 and the reading is NOT A LEVER); a spy on decode_trial, with TF32
  turned on before the job, shows amp None, False, None for the three passes and TF32 off only during the fp32 pass,
  restored after, and trials.csv and decode.json record the same;
- the paired S(12) bf16 minus fp32 has the sign of the construction (fp32 2 cm closer to the walkway: positive);
- the job command passes scripts/queue_job.py's checks (dry run, nothing written).
"""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from mdfm import paths
from mdfm.data.canonical import Refused
from mdfm.gates import g3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EM_DASH = chr(0x2014)
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))


def _load(path, name):
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old
    return mod


P = _load(os.path.join(ROOT, "scripts", "f2_precision.py"), "f2_precision_tested")
GG = P.G()
COLS = ["row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "t0_s", "t1_s", "reference", "prediction",
        "signed_error", "unit", "matched"]
DEV12 = [f"weargait:DV{i:02d}" for i in range(1, 7)]
LOCK = ["weargait:LB01"]


@pytest.fixture(autouse=True)
def keep_alloc_conf():
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    yield
    if old is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old


# ---------------------------------------------------------------- the pre-registration
def test_prediction_text_hash_constant_and_report_agree():
    with open(P.PREDICTION_PATH, "rb") as f:
        raw = f.read()
    assert hashlib.sha256(raw).hexdigest() == P.PREDICTION_SHA256 == P.check_prediction()
    text = raw.decode("ascii")
    assert "\r" not in text and EM_DASH not in text and chr(0x2013) not in text
    for phrase in ("T_S = max(1.0 cm, 2 x A_S0)", "PRECISION LEVER", "NOT A LEVER", "INCONCLUSIVE", "bf16_rerun",
                   "seed_of(20260929, trial_id)", "20261005"):
        assert phrase in text, phrase
    assert P.THRESHOLD == {"stride": 1.0, "speed": 0.01} and P.NOISE_FACTOR == 2.0 and P.SEED == 20261005


def test_changed_prediction_stops_analysis_and_decode(tmp_path, monkeypatch):
    p = tmp_path / "prediction.txt"
    p.write_text("another prediction\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(P, "PREDICTION_PATH", str(p))
    with pytest.raises(AssertionError, match="prediction hash mismatch"):
        P.analyse(str(tmp_path / "no_such_folder"), DEV12, LOCK)               # the events file is never opened
    with pytest.raises(AssertionError, match="prediction hash mismatch"):
        P.decode(P.parse(["--decode", "--device", "cpu", "--out", str(tmp_path / "o")]))


# ---------------------------------------------------------------- the reading
def test_precision_reading_edges():
    A0 = {"point": 0.2}
    assert P.precision_reading({"point": 1.0, "interval": [0.5, 1.5]}, A0, 1.0)["reading"] == "PRECISION LEVER"
    assert P.precision_reading({"point": -1.2, "interval": [-1.6, -0.8]}, A0, 1.0)["reading"] == "PRECISION LEVER"
    assert P.precision_reading({"point": 1.0, "interval": [0.0, 2.0]}, A0, 1.0)["reading"] == "INCONCLUSIVE"
    assert P.precision_reading({"point": 0.2, "interval": [-0.5, 0.9]}, A0, 1.0)["reading"] == "NOT A LEVER"
    assert P.precision_reading({"point": 0.2, "interval": [-0.5, 1.0]}, A0, 1.0)["reading"] == "INCONCLUSIVE"
    big = P.precision_reading({"point": 1.5, "interval": [1.2, 1.7]}, {"point": 0.9}, 1.0)
    assert big["threshold"] == pytest.approx(1.8) and big["reading"] == "NOT A LEVER"   # the noise floor governs
    edge = P.precision_reading({"point": 1.5, "interval": [1.2, 1.8]}, {"point": 0.9}, 1.0)
    assert edge["reading"] == "INCONCLUSIVE"                                             # inside means strictly
    assert P.precision_reading({"point": 1.5, "interval": None}, A0, 1.0)["reading"] == "INCONCLUSIVE"
    assert "restated" in P.consequence({"reading": "PRECISION LEVER"}, {"reading": "NOT A LEVER"})
    assert "speed-only" in P.consequence({"reading": "NOT A LEVER"}, {"reading": "PRECISION LEVER"})
    assert "stand" in P.consequence({"reading": "NOT A LEVER"}, {"reading": "NOT A LEVER"})
    assert "second job" in P.consequence({"reading": "INCONCLUSIVE"}, {"reading": "NOT A LEVER"})


def pass_events(persons, test_offset=0.0, test_sd=0.0, rerun_sd=0.0, seed=0, drop_test=0):
    """events.csv rows of three passes: per person and task, 3 passes of 4 matched strides and one matched pass speed;
    bf16 decoded = 0.85 walkway + noise, fp32 = bf16 + test_offset + N(0, test_sd), rerun = bf16 + N(0, rerun_sd)
    (speeds get the same in m/s / 100). drop_test strides of the first person are missing from the fp32 pass, and
    one stride matched in the bf16 pass only is added."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, p in enumerate(persons):
        for task in ("SelfPace", "HurriedPace"):
            tid = f"{p.split(':')[1]}_{task}"
            for k in range(3):
                t = 2.0 + 12.0 * k
                for j in range(4):
                    x = 115.0 + 3.0 * i + rng.normal(0.0, 3.0)
                    y = 0.85 * x + rng.normal(0.0, 2.0)
                    vals = {"bf16": y, "fp32": y + test_offset + rng.normal(0.0, test_sd) if test_sd else y + test_offset,
                            "bf16_rerun": y + (rng.normal(0.0, rerun_sd) if rerun_sd else 0.0)}
                    for name, v in vals.items():
                        if name == "fp32" and i == 0 and task == "SelfPace" and k == 0 and j < drop_test:
                            continue
                        rows.append((name, "stride", tid, p, task, "LR"[j % 2], str(k), t + j, t + j + 1.1, x, v,
                                     v - x, "cm", True))
                v0 = 1.1 + 0.01 * i + rng.normal(0.0, 0.03)
                w = 0.84 * v0
                for name, v in {"bf16": w, "fp32": w + test_offset / 100.0, "bf16_rerun": w}.items():
                    rows.append((name, "speed", tid, p, task, "", str(k), t, t + 4.0, v0, v, v - v0, "m/s", True))
        rows.append(("bf16", "stride", f"{p.split(':')[1]}_SelfPace", p, "SelfPace", "L", "9", 90.0, 91.1, 120.0,
                     100.0, -20.0, "cm", True))
        rows.append(("fp32", "stride", f"{p.split(':')[1]}_SelfPace", p, "SelfPace", "L", "8", 80.0, 81.1, 120.0,
                     10.0, -110.0, "cm", False))
    return pd.DataFrame(rows, columns=COLS)


def write_events(tmp, ev):
    out = str(tmp / "prec")
    os.makedirs(out, exist_ok=True)
    ev.to_csv(os.path.join(out, "events.csv"), index=False, lineterminator="\n", encoding="utf-8")
    return out


def test_a_constant_fp32_offset_is_a_lever(tmp_path):
    out = write_events(tmp_path, pass_events(DEV12, test_offset=2.0, drop_test=2))
    res, ps, pp = P.analyse(out, DEV12, LOCK, n_boot=300)
    s = res["differences"]["stride"]
    assert s["D_test"]["point"] == pytest.approx(2.0) and s["A_test"]["point"] == pytest.approx(2.0)
    assert s["D_rerun"]["point"] == 0.0 and s["A_rerun"]["point"] == 0.0
    assert res["readings"]["stride"]["reading"] == "PRECISION LEVER"
    assert res["readings"]["speed"]["reading"] == "PRECISION LEVER"            # 0.02 m/s against T_V 0.01
    # only units matched in every pass: 6 persons x 2 tasks x 3 x 4, minus the two strides fp32 lacks
    assert res["counts"]["stride"]["common_units"] == 6 * 2 * 3 * 4 - 2 and len(ps) == 6 * 2 * 3 * 4 - 2
    assert res["counts"]["stride"]["matched_bf16"] == 6 * 2 * 3 * 4 + 6
    assert np.allclose(ps["d_test"], 2.0) and np.allclose(ps["d_rerun"], 0.0)
    assert set(pp["line"]) == {"stride", "speed"} and len(pp) == 12
    # bf16 decodes are 0.85 x walkway, fp32 2 cm longer, so closer to the walkway: S(12) bf16 minus fp32 is positive
    # and equals the difference of the two pass lines (the same six persons on both sides)
    pr = res["paired_ref_minus_test"]["S12"]
    assert pr["point"] > 1.5 and pr["interval"][0] > 0 and pr["persons_dropped"] == []
    assert pr["point"] == pytest.approx(res["lines"]["bf16"]["S12"]["point"] - res["lines"]["fp32"]["S12"]["point"],
                                        abs=1e-12)
    assert "restated" in res["consequence"] and res["diag_check"]["evaluated"] is False
    # per-pass lines equal g3_gain_check.lines_for on each pass's own matched units (the unmatched-only extra included)
    frames = P.read_passes(os.path.join(out, "events.csv"), DEV12, LOCK)
    ref = GG.lines_for(frames["fp32"][0], frames["fp32"][1], {"dev12": DEV12}, 1.0, 1.0,
                       {"S12": ("stride", "dev12"), "V12": ("speed", "dev12")}, 300, P.SEED)
    assert res["lines"]["fp32"]["S12"]["point"] == pytest.approx(ref["S12"]["point"], abs=1e-12)
    text = P.summary_md(GG.clean(res))
    assert P.LABEL in text and "PRECISION LEVER" in text and EM_DASH not in text
    json.dumps(GG.clean(res), allow_nan=False)


def test_a_small_jitter_is_not_a_lever_and_a_large_rerun_noise_is_reported(tmp_path):
    out = write_events(tmp_path, pass_events(DEV12, test_sd=0.1, seed=1))
    res, _, _ = P.analyse(out, DEV12, LOCK, n_boot=300)
    assert res["readings"]["stride"]["reading"] == "NOT A LEVER"
    assert res["readings"]["stride"]["threshold"] == pytest.approx(1.0)
    assert not res["readings"]["stride"]["rerun_noise_above_note"] and "stand" in res["consequence"]
    out2 = write_events(tmp_path / "b", pass_events(DEV12, test_offset=1.5, rerun_sd=3.0, seed=2))
    res2, _, _ = P.analyse(out2, DEV12, LOCK, n_boot=300)
    rd = res2["readings"]["stride"]
    assert rd["threshold"] == pytest.approx(2.0 * res2["differences"]["stride"]["A_rerun"]["point"]) and rd["threshold"] > 2.0
    assert rd["rerun_noise_above_note"] and rd["reading"] != "PRECISION LEVER"
    assert "noise floor is a finding" in P.summary_md(GG.clean(res2))


def test_persons_are_guarded_before_any_value(tmp_path):
    out = write_events(tmp_path, pass_events(DEV12 + LOCK))
    with pytest.raises(Refused):
        P.analyse(out, DEV12, LOCK, n_boot=50)
    out2 = write_events(tmp_path / "b", pass_events(DEV12 + ["weargait:WHC003"]))
    with pytest.raises(AssertionError, match="outside the expected set"):
        P.analyse(out2, DEV12, LOCK, n_boot=50)


# ---------------------------------------------------------------- precision and draws on a tiny model
def _record(person, task, seed, lockbox=False):
    import model_fixture as MF
    st = MF.imu_stream(12.0, sites=tuple(range(13)), seed=seed)
    sig = np.array(st.values, copy=True)
    sig[..., 6:9] = 0.3
    T = sig.shape[0]
    return {"source": "weargait_pd", "subject_ns": person.replace("weargait:", "weargait_pd:"), "person_id": person,
            "key": task, "fps": 100.0, "signal": sig, "channel_mask": np.ones((13, 9), bool),
            "location_ids": np.arange(13), "sample_valid": np.ones((T, 13), bool), "acc_content": "with_gravity",
            "lockbox": lockbox, "tier": "DEV"}


@pytest.fixture()
def tiny_loaded(tmp_path, monkeypatch):
    if not HAVE_MA:
        pytest.skip("SMPL+H model not found")
    import torch

    from mdfm.data import sources
    from mdfm.eval import infer as I
    from mdfm.model import mfm as M
    torch.manual_seed(11)
    model = M.MFM(M.TINY)
    model.eval()
    model.requires_grad_(False)
    monkeypatch.setattr(sources, "demographic", lambda source, ns: {"sex": "F", "age": 64.0})
    return I.Loaded(model=model, path=str(tmp_path / "run_tiny" / "ckpt_B_000010.pt"), name="run_tiny_ckpt_B_000010",
                    sha256="d" * 64, weights="ema", step=10, phase="B", phase_steps=(5, 5), config_json="{}",
                    config_sha256="e" * 64, persons={"weargait_pd": ["weargait:TR01"]},
                    persons_sha256={"all": "f" * 64}, hashes={}, drift=[])


def test_decode_trial_equals_the_g3_predictor_on_the_cpu(tiny_loaded):
    from mdfm.eval import infer as I
    rec = g3.model_input(_record("weargait:DV01", "SelfPace", 0))
    pred = I.G3Predictor(tiny_loaded, refine_steps=2, points=16, batch_windows=4, seed=20260929)
    a = pred("DV01_SelfPace", rec)
    b, log = P.decode_trial(tiny_loaded, "DV01_SelfPace", rec, False, 2, 16, 4, 20260929)
    np.testing.assert_array_equal(a.joints, b.joints)
    np.testing.assert_array_equal(a.contact, b.contact)
    assert log["windows"] == pred.log[-1]["windows"] == 2 and log["refine_first"] == pred.log[-1]["refine_first"]
    assert log["amp"] == "False" and log["autocast_bf16"] is False
    _, log_on = P.decode_trial(tiny_loaded, "DV01_SelfPace", rec, True, 1, 16, 4, 20260929)
    assert log_on["amp"] == "True" and log_on["autocast_bf16"] is True                  # amp True turns it on


def test_refinement_draws_do_not_depend_on_the_precision(tiny_loaded, monkeypatch):
    import torch

    from mdfm.model import mfm as M
    seen = []
    orig = M.Batch.subsample

    def spy(self, per_stream, generator=None):
        seen.append(hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest())
        return orig(self, per_stream, generator)
    monkeypatch.setattr(M.Batch, "subsample", spy)
    rec = g3.model_input(_record("weargait:DV02", "HurriedPace", 1))
    runs = {}
    for amp in (True, False):
        seen.clear()
        tb, _ = P.decode_trial(tiny_loaded, "DV02_HurriedPace", rec, amp, 3, 16, 4, 20260929)
        runs[amp] = (list(seen), tb)
    assert len(runs[True][0]) == 3 and runs[True][0] == runs[False][0]          # one draw per step, same states
    ja, jb = runs[True][1].joints, runs[False][1].joints
    ok = np.isfinite(ja) & np.isfinite(jb)
    assert ok.any() and not np.array_equal(ja[ok], jb[ok])                       # the arithmetic differs
    assert P.autocast_on("cpu", None) is False and P.autocast_on("cuda", None) is True
    assert P.autocast_on("cpu", True) is True and P.autocast_on("cuda", False) is False
    assert torch.get_float32_matmul_precision() in ("highest", "high", "medium")


def test_float32_backends_restores_the_flags():
    import torch
    before = P.backend_flags(torch)
    torch.backends.cudnn.allow_tf32 = True
    try:
        with P.float32_backends(torch) as flags:
            assert flags == {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                             "float32_matmul_precision": "highest"}
        assert torch.backends.cudnn.allow_tf32 is True
        with pytest.raises(RuntimeError):
            with P.float32_backends(torch):
                raise RuntimeError("inside")
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])


def _targets(persons):
    tr, ics, st, ps = [], [], [], []
    for p in persons:
        for task in ("SelfPace", "HurriedPace"):
            tid = f"{p.split(':')[1]}_{task}"
            tr.append((tid, p, task, 1200))
            L, R = [2.0, 3.1, 4.2, 5.3, 6.4], [2.55, 3.65, 4.75, 5.85]
            ics += [(tid, p, task, "L", x) for x in L] + [(tid, p, task, "R", x) for x in R]
            st += [(tid, p, task, "0", "L", a, b, 1.2) for a, b in zip(L[:-1], L[1:])]
            st += [(tid, p, task, "0", "R", a, b, 1.2) for a, b in zip(R[:-1], R[1:])]
            ps.append((tid, p, task, "0", 2.0, 6.4, 9, 1.1))
    walk = pd.DataFrame(ics, columns=["trial_id", "person_id", "task", "foot", "t_s"])
    return g3.Targets(tuple(sorted(persons)), False,
                      pd.DataFrame(tr, columns=["trial_id", "person_id", "task", "samples"]), walk,
                      walk.assign(t_raw_s=walk["t_s"], offset_ms=0.0),
                      pd.DataFrame(st, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1",
                                                "length_m"]),
                      pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                                "speed_m_s"]),
                      pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"]), {"lockbox_rows": 0})


def test_decode_with_a_tiny_model_then_the_analysis(tiny_loaded, tmp_path, monkeypatch):
    from mdfm.eval import infer as I
    dev = ["weargait:DV01", "weargait:DV02", "weargait:WHC003", "weargait:WHC026"]
    dev12 = ["weargait:DV01", "weargait:DV02"]
    state = {"read": []}

    def load_targets(persons, allow_lockbox=False, read_row=None, tasks=g3.TASKS):
        assert sorted(persons) == dev12 and not allow_lockbox
        state["read"].append("targets")
        return _targets(persons)

    def imu_records(persons, trials, allow_lockbox=False, read_row=None, tasks=g3.TASKS):
        assert sorted(persons) == dev12 and not allow_lockbox
        for i, r in enumerate(trials.itertuples()):
            yield r.trial_id, _record(r.person_id, r.task, i)

    real = g3.controls
    monkeypatch.setattr(g3, "controls", lambda fold: dev if fold == "dev" else ["weargait:TR01"] if fold == "train"
                        else real(fold))
    monkeypatch.setattr(g3, "lockbox_persons", lambda: LOCK)
    monkeypatch.setattr(g3, "load_targets", load_targets)
    monkeypatch.setattr(g3, "imu_records", imu_records)
    monkeypatch.setattr(I, "load_checkpoint", lambda path, weights="ema", device="cpu", cfg=None: tiny_loaded)
    monkeypatch.setattr(GG, "stamp_moveatlas", lambda: {"head": "test"})
    import torch
    calls = []
    real_decode_trial = P.decode_trial

    def spy(loaded, trial_id, record, amp, *args, **kw):
        """What each decode_trial call receives: amp and the TF32 flags in force at the call."""
        calls.append((amp, bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32),
                      torch.get_float32_matmul_precision()))
        return real_decode_trial(loaded, trial_id, record, amp, *args, **kw)
    monkeypatch.setattr(P, "decode_trial", spy)
    before = P.backend_flags(torch)
    out = str(tmp_path / "prec")
    a = P.parse(["--decode", "--device", "cpu", "--refine-steps", "1", "--points", "16", "--batch-windows", "4",
                 "--out", out])
    try:
        # TF32 on before the job, as on a GPU where it is enabled: only the fp32 pass may turn it off
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        assert P.decode(a, log=lambda *x: None) == out and state["read"] == ["targets"]
        after = P.backend_flags(torch)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(before["float32_matmul_precision"])
    n = 4                                                   # trials per pass, in the order of PASSES
    assert [c[0] for c in calls] == [None] * n + [False] * n + [None] * n
    on, off = (None, True, True, "high"), (False, False, False, "highest")
    assert calls[:n] == [on] * n and calls[n:2 * n] == [off] * n and calls[2 * n:] == [on] * n
    assert after == {"cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True, "float32_matmul_precision": "high"}
    ev = pd.read_csv(os.path.join(out, "events.csv"), encoding="utf-8")
    assert list(ev.columns) == COLS and set(ev["row"]) == {"bf16", "fp32", "bf16_rerun"}
    assert set(ev["person_id"]) <= set(dev12) and set(ev["kind"]) <= {"stride", "speed"}
    with open(os.path.join(out, "decode.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["prediction"]["sha256"] == P.PREDICTION_SHA256 and doc["trials"] == 4
    assert doc["passes"]["fp32"]["backend_flags"]["float32_matmul_precision"] == "highest"
    assert doc["passes"]["fp32"]["amp"] is False and doc["passes"]["bf16"]["autocast_bf16"] is False   # CPU
    seen = {k: v["seen_by_decode_trial"] for k, v in doc["passes"].items()}
    assert seen["fp32"] == {"amp": ["False"], "autocast_bf16": [False], "cuda_matmul_allow_tf32": [False],
                            "cudnn_allow_tf32": [False], "float32_matmul_precision": ["highest"]}
    for k in ("bf16", "bf16_rerun"):
        assert seen[k] == {"amp": ["None"], "autocast_bf16": [False], "cuda_matmul_allow_tf32": [True],
                           "cudnn_allow_tf32": [True], "float32_matmul_precision": ["high"]}
    trials = pd.read_csv(os.path.join(out, "trials.csv"), encoding="utf-8", dtype={"amp": str},
                         keep_default_na=False)
    assert len(trials) == 3 * 4 and set(trials["pass"]) == {"bf16", "fp32", "bf16_rerun"}
    assert trials.groupby("pass")["amp"].agg(lambda s: sorted(set(s))).to_dict() == {
        "bf16": ["None"], "fp32": ["False"], "bf16_rerun": ["None"]}
    assert trials.groupby("pass")["cuda_matmul_allow_tf32"].agg(lambda s: sorted(set(s))).to_dict() == {
        "bf16": [True], "fp32": [False], "bf16_rerun": [True]}
    res, ps, pp = P.analyse(out, dev12, LOCK, n_boot=50)
    if res["counts"]["stride"]["common_units"]:
        assert res["differences"]["stride"]["D_test"]["point"] == 0.0
        assert res["readings"]["stride"]["reading"] == "NOT A LEVER"
    P.write_analysis(out, res, ps, pp)
    for f in ("results.json", "summary.md", "per_stride.csv", "per_person.csv", "_provenance_decode.json"):
        assert os.path.exists(os.path.join(out, f)), f


def test_job_command_passes_queue_job_checks(capsys):
    qj = _load(os.path.join(paths.TREE, "scripts", "queue_job.py"), "queue_job_f2prec_tested")
    c = P.job_command()
    i = c.index("--")
    cmd = c[i + 1:]
    assert cmd[:2] == ["scripts/f2_precision.py", "--decode"]
    assert cmd[cmd.index("--checkpoint") + 1] == P.CHECKPOINT and cmd[cmd.index("--memory-gb") + 1] == "12"
    assert cmd[cmd.index("--batch-windows") + 1] == "32"
    assert cmd[cmd.index("--out") + 1] == "reports/bench/f2_lever1/precision_phaseA_root_full_ckpt_B_032044"
    assert c[c.index("--cls") + 1] == "1" and c[c.index("--group") + 1] == "F2L1"
    assert qj.main(c[1:i] + ["--dry-run"] + c[i:]) == 0
    assert "would write 1_071_f2l1_precision_v1_B" in capsys.readouterr().out
