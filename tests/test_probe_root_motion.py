"""Tests of scripts/probe_root_motion.py (gate G-g of reports/bench/root_motion_diagnosis.md, fix 7), without a
model: the truth taken as the decoded body scores 1 on every ratio, slope, cosine and gain line and 0 on the yaw
error, its time-warp response equals the true ratio and the gate passes; smoothing runs along the time axis of
each window, never across windows (the scratch helper of the diagnosis filtered axis 0 of stacked arrays); the
per-patch targets of a circle walk integrated from the centre patch give back its heading and path (the latent
dead reckoning with an oracle readout).
Added with --warps and the warp rows (weekend 2026-10-01, T07): the command line keeps WARPS (0.8, 1.25) and an
empty tag by default, --warps parses factors in order and refuses a non-positive, non-numeric or repeated one;
warp_rows gives one row per factor, the unwarped one first (ratios 1, error 0), its ratios those of warp_lines;
the report and the _warps.json beside it, with the note when the factors are not the registered ones.
Added in review: main end to end with the store, renderer, model and latent fits stubbed, with and without
--latent, --warps and --no-warp: md, json and the warp rows file are written, the latent rows never reach it.
"""
import math
import os

import numpy as np
import pytest

from mdfm import paths


def _script():
    import importlib.util
    old = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:                                                            # the script sets the variable at import
        spec = importlib.util.spec_from_file_location("probe_root_motion_script",
                                                      os.path.join(paths.TREE, "scripts", "probe_root_motion.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = old
    return mod


S = _script()


def _walk(omega, v=1.2, T=1000, sway=0.03, speed_scale=1.0):
    """A window-frame walk: yaw omega (t - t_c) rad plus a 1 Hz pelvis yaw sway, the pelvis on the circle of speed v
    (a line for omega 0) plus a 2 Hz lateral sway; transl (T, 3) m, yaw (T,) rad."""
    t = np.arange(T) / 100.0
    tau = t - t[T // 2]
    psi = omega * tau
    X = np.stack([v / omega * (1 - np.cos(psi)), v / omega * np.sin(psi)], 1) if omega else \
        np.stack([np.zeros_like(tau), v * tau], 1)
    X = X * speed_scale
    X[:, 0] += sway * np.sin(2 * math.pi * 2.0 * t)
    X -= X[T // 2]
    yaw = psi + 0.05 * np.sin(2 * math.pi * 1.0 * t)
    yaw = yaw - yaw[T // 2]
    return np.stack([X[:, 0], np.full(T, 0.95), X[:, 1]], 1), yaw


def _truth_windows(speed_scale=1.0):
    out = []
    for omega, lay in ((0.4, "xsens13"), (0.0, "xsens13"), (-0.7, "camera1")):
        tr, yaw = _walk(omega, speed_scale=speed_scale)
        out.append({"transl_pred": tr, "transl_true": tr, "yaw_pred": yaw, "yaw_true": yaw, "layout": lay})
    return out


def test_the_truth_scores_one_on_every_line():
    wins = _truth_windows()
    lines = S.root_lines(wins)
    for name, b in lines["bins"].items():
        for key in ("speed_ratio", "velocity_cosine", "heading_gain", "heading_corr"):
            assert b[key] == pytest.approx(1.0, abs=1e-9), (name, key)
        assert b["yaw_err_deg"] == pytest.approx(0.0, abs=1e-9) and b["knots"] > 0
        assert b["velocity_heading_cosine"] == pytest.approx(b["velocity_heading_cosine_true"], abs=1e-12)
    assert lines["lateral_slope"] == pytest.approx(1.0, abs=1e-9)
    assert lines["forward_slope"] == pytest.approx(1.0, abs=1e-9)
    assert set(lines["heading_gain_by_layout"]) == {"xsens13", "camera1"}
    assert all(v["gain"] == pytest.approx(1.0, abs=1e-9) for v in lines["heading_gain_by_layout"].values())
    # the time warp: the same windows played 0.8 and 1.25 times as fast, decoded as the truth
    warped = {f: _truth_windows(speed_scale=f) for f in S.WARPS}
    warp = S.warp_lines(wins, warped)
    for f in S.WARPS:
        w = warp[str(f)]
        assert w["decoded_ratio"] == pytest.approx(w["true_ratio"], abs=1e-12) and w["response_error"] < 1e-12
        assert w["true_ratio"] == pytest.approx(f, rel=0.05)
    verdict = S.gate(lines, warp, None)
    assert verdict["pass"] and len(verdict["verdicts"]) == len(S.BINS_S) + 1 + 2 + 2
    # a body that stands still fails the speed and slope lines
    still = [dict(w, transl_pred=np.tile(w["transl_true"][len(w["transl_true"]) // 2], (1000, 1)),
                  yaw_pred=np.zeros(1000)) for w in wins]
    bad = S.root_lines(still)
    assert all(b["speed_ratio"] == 0.0 for b in bad["bins"].values()) and bad["lateral_slope"] == 0.0
    assert not S.gate(bad, warp, None)["pass"]


def test_smoothing_runs_along_the_time_axis():
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 200, 2))
    y = S.smooth_time(x, 25)
    for w in range(5):
        np.testing.assert_allclose(y[w], uniform_filter1d(x[w], 25, axis=0, mode="nearest"), atol=1e-12)
    const = np.repeat(np.arange(5.0)[:, None, None], 200, axis=1).repeat(2, axis=2)   # each window constant in time
    np.testing.assert_allclose(S.smooth_time(const, 25), const, atol=1e-12)          # never mixed across windows
    with pytest.raises(ValueError):
        S.smooth_time(np.zeros(10), 5)


def test_the_true_patch_rates_integrate_to_the_path():
    """The latent dead reckoning with an oracle readout: the per-patch targets of a circle walk integrated over the
    patches from the centre patch give its heading within 1 deg and its pelvis within 3 cm at the patch centres."""
    tr, yaw = _walk(0.4, sway=0.0)
    L = len(yaw)
    Y, ks = S.patch_targets(yaw, tr[:, [0, 2]], L)
    head, pos = S.integrate_patches(Y[:, 0], Y[:, 1], Y[:, 2], ks, L // 2)
    c = L // 2
    s = slice(*S.DR_PATCHES)
    assert np.degrees(np.abs(S._wrap(head - (yaw[ks] - yaw[c])))[s]).max() < 1.0
    assert np.linalg.norm(pos - (tr[ks][:, [0, 2]] - tr[c, [0, 2]]), axis=1)[s].max() < 0.03
    assert len(ks) == 40 and Y[:, 1].mean() == pytest.approx(1.2, rel=0.02)


# ---------------------------------------------------------------- --warps and the warp rows (T07)
def test_the_command_line_and_warps():
    a = S.parse([])
    assert a.warps == S.WARPS == (0.8, 1.25) and a.tag == "" and not a.no_warp and a.checkpoint == "init"
    a = S.parse(["--checkpoint", "c.pt", "--config", "configs/stage1_pilot.yaml", "--warps", "1.25,1.5", "--tag",
                 "warp15"])
    assert a.warps == (1.25, 1.5) and a.tag == "warp15"
    assert S.parse(["--warps", " 2 "]).warps == (2.0,) and S.parse_warps(None) is None
    for bad in ("0", "-1.5", "1.25,x", "1.5,1.50", "nan", " , "):
        with pytest.raises(ValueError):
            S.parse_warps(bad)
        with pytest.raises(SystemExit):
            S.parse(["--warps", bad])


def test_warp_rows_hold_the_unwarped_row_and_the_ratios_of_warp_lines():
    wins = _truth_windows()
    factors = (1.25, 1.5)
    warped = {f: _truth_windows(speed_scale=f) for f in factors}
    rows = S.warp_rows(wins, warped)
    warp = S.warp_lines(wins, warped)
    assert [r["factor"] for r in rows] == [1.0, 1.25, 1.5] and all(r["windows"] == 3 for r in rows)
    r0 = rows[0]
    assert r0["decoded_ratio"] == pytest.approx(1.0) and r0["true_ratio"] == pytest.approx(1.0)
    assert r0["response_error"] == pytest.approx(0.0, abs=1e-12)
    assert r0["true_speed_m_s"] == pytest.approx(np.mean([S.window_speed(w["transl_true"]) for w in wins]))
    for r in rows[1:]:
        w = warp[str(r["factor"])]
        for k in ("decoded_ratio", "true_ratio", "response_error"):
            assert r[k] == pytest.approx(w[k], abs=1e-12), k
        assert r["true_speed_m_s"] == pytest.approx(r["factor"] * r0["true_speed_m_s"], rel=0.05)
        assert r["decoded_speed_m_s"] == pytest.approx(r["true_speed_m_s"], abs=1e-12)
    # the gate scores the warps it is given; with others than WARPS the report says it is not the registered read
    verdict = S.gate(S.root_lines(wins), warp, None)
    assert [v["line"] for v in verdict["verdicts"] if v["line"].startswith("warp")] == [
        "warp response error at x1.25", "warp response error at x1.5"]
    assert S.warp_note(factors) == ("time-warp factors x1.25, x1.5 instead of the registered x0.8, x1.25: the warp "
                                    "lines of this gate are not the registered G-g read")


def _doc(warp, note=None):
    doc = {"script": "scripts/probe_root_motion.py", "run_at": "2026-10-01T21:00:00",
           "checkpoint": {"path": "cache/runs/r/ckpt_B_028000.pt", "sha256": "a" * 64}, "weights": "ema",
           "split": "dev", "layout": "xsens13", "refine_steps": 0, "seed": 20260930, "windows": 3, "persons": 2,
           "persons_sha256": "b" * 64, "latent_fit_persons_sha256": None, "moveatlas_head": "c" * 40,
           "code_sha256": "d" * 64, "config_sha256": "e" * 64, "lines": S.root_lines(_truth_windows()),
           "warp": warp, "latent": None, "gate": S.gate(S.root_lines(_truth_windows()), warp, None),
           "gate_lines": S.LINES, "seconds": 1.0}
    if note:
        doc["gate_note"] = note
    return doc


def test_the_report_and_the_warp_rows_file(tmp_path):
    import json
    wins = _truth_windows()
    for factors, note in ((S.WARPS, None), ((1.25, 1.5), S.warp_note((1.25, 1.5)))):
        warped = {f: _truth_windows(speed_scale=f) for f in factors}
        warp = S.warp_lines(wins, warped)
        warp["windows"] = len(wins)
        doc = _doc(warp, note)
        wd = S.warps_doc(doc, S.warp_rows(wins, warped))
        assert wd["warps"] == list(factors) and [r["factor"] for r in wd["rows"]] == [1.0, *factors]
        assert wd["checkpoint"]["sha256"] == "a" * 64 and wd["moveatlas_head"] == "c" * 40
        stem = tmp_path / f"root_motion_x_{len(factors)}_{factors[-1]:g}"
        md, js, wj = (str(stem) + ".md", str(stem) + ".json", str(stem) + "_warps.json")
        with open(wj, "w", encoding="utf-8") as f:
            json.dump(wd, f)
        S.write_md(md, js, doc, wj)
        text = open(md, encoding="utf-8").read()
        assert f"One row per factor, the unwarped one included: {os.path.basename(wj)}." in text
        assert ("Note: time-warp factors" in text) == (note is not None) and "\u2014" not in text
        assert json.load(open(wj, encoding="utf-8"))["rows"][0]["factor"] == 1.0
    S.write_md(md, js, _doc(None))                                       # --no-warp: no warp line, no rows file
    assert "Time warp" not in open(md, encoding="utf-8").read()


# ---------------------------------------------------------------- stride errors of the warp rows (review)
def _contacts(T=1000):
    """(T, 2) contact labels: the left foot down from 1 s every 1 s for 0.6 s, the right foot from 1.5 s."""
    c = np.zeros((T, 2))
    for k in range(1, 10):
        c[100 * k:100 * k + 60, 0] = 1.0
        c[100 * k + 50:100 * k + 110, 1] = 1.0
    return c


def _gait_window(v, scale, T=1000):
    """A straight walk at v m/s whose joints all sit on the pelvis path; the decoded body walks scale times as far."""
    tr = np.stack([np.zeros(T), np.full(T, 0.95), v * np.arange(T) / 100.0], 1)
    return {"transl_pred": tr * [1.0, 1.0, scale], "transl_true": tr, "contact_true": _contacts(T),
            "rel_true": np.zeros((T, 22, 3)), "joints_pred": np.repeat((tr * [1.0, 1.0, scale])[:, None], 22, axis=1)}


def test_stride_errors_read_the_decoded_body_at_the_true_ics():
    d, lt = S.stride_errors(_gait_window(1.2, 0.9))
    assert len(d) == 16 and lt == pytest.approx(np.full(16, 1.2)) and d == pytest.approx(np.full(16, -0.12))
    assert [len(x) for x in S.stride_errors({"transl_true": np.zeros((5, 3))})] == [0, 0]   # no truth joints
    base = [_gait_window(1.2, 0.95), _gait_window(1.0, 0.95)]
    rows = S.warp_rows(base, {1.5: [_gait_window(1.8, 0.8), _gait_window(1.5, 0.8)]})
    r1, r15 = rows
    assert (r1["factor"], r15["factor"], r1["strides"], r15["strides"]) == (1.0, 1.5, 32, 32)
    assert r1["stride_mae_cm"] == pytest.approx(5.5) and r1["stride_bias_cm"] == pytest.approx(-5.5)
    assert r15["stride_mae_cm"] == pytest.approx(33.0) and r15["true_stride_m"] == pytest.approx(1.65)
    assert S.warp_rows(_truth_windows(), {1.5: _truth_windows(1.5)})[1]["stride_mae_cm"] is None   # no stride


# ---------------------------------------------------------------- main end to end, heavy parts stubbed (review)
class _Win:
    """A rendered window stand-in: the speed factor it was played at, its person and body."""
    def __init__(self, factor, person):
        self.factor, self.person_id = factor, person
        self.body = {"betas": np.zeros(16), "gender": 0}
        self.streams = []


def _stub_main(monkeypatch):
    """main() on the CPU without a store, a renderer or a model: three windows per split, decoded as the truth
    played at the window's factor; the latent block returns rows of its own shape (no 'factor')."""
    import types

    import pandas as pd

    from mdfm.data import body_store as BS
    from mdfm.factory import synth as SY
    from mdfm.train import provenance as PV

    class Store:
        def __init__(self, split):
            self.split = split
            self.takes = pd.DataFrame({"person_id": [f"p:{split}{k}" for k in range(3)]})

    def render(store, takes, starts, sampler, specs_fn, seed, chunk=64, bodies=None):
        if bodies is None:
            return [_Win(1.0, store.takes["person_id"].iloc[int(i)]) for i in takes]
        return [_Win(bodies[0], "p:dev0") for _ in bodies[1]]

    def decode(model, wins, refine_steps, seed, batch_windows=8):
        out = []
        for k, w in enumerate(wins):
            tr, yaw = _walk((0.4, 0.0, -0.7)[k % 3], speed_scale=w.factor)
            out.append({"transl_pred": tr, "transl_true": tr, "yaw_pred": yaw, "yaw_true": yaw, "layout": "xsens13",
                        "joints_pred": np.repeat(tr[:, None, :], 22, axis=1), "rel_true": np.zeros((len(tr), 22, 3)),
                        "contact_true": _contacts(len(tr))})
        return out

    mz = {"yaw_latent": 30.0, "yaw_model": 40.0, "yaw_zero": 50.0, "pos_latent": 0.5, "pos_model": 0.6,
          "pos_zero": 1.0, "r2_yaw_rate": 0.5, "r2_forward": 0.5, "r2_lateral": 0.5, "alphas": [1.0] * 3, "windows": 3}
    monkeypatch.setattr(BS, "BodyStore", Store)
    monkeypatch.setattr(SY, "SynthSampler", lambda store, device: types.SimpleNamespace(store=store))
    monkeypatch.setattr(SY, "select_bodies", lambda bw, pos: (bw[1], list(pos)))
    monkeypatch.setattr(PV, "moveatlas_head", lambda *a, **k: "c" * 40)
    monkeypatch.setattr(PV, "code_sha256", lambda *a, **k: "d" * 64)
    monkeypatch.setattr(S, "load_model", lambda c, w, d, cfg: (None, {"path": "init", "sha256": None}, "init",
                                                               frozenset()))
    monkeypatch.setattr(S, "walk_windows", lambda store, max_windows=None, seed=0: (np.arange(3), np.zeros(3, int)))
    monkeypatch.setattr(S, "render", render)
    monkeypatch.setattr(S, "warped_bodies", lambda store, takes, starts, f: (("warp", f), list(range(len(takes)))))
    monkeypatch.setattr(S, "decode_windows", decode)
    monkeypatch.setattr(S, "add_true_rel", lambda dws, bank: None)
    monkeypatch.setattr(S, "latent_features", lambda model, wins, dev: [None] * len(wins))
    monkeypatch.setattr(S, "latent_rows", lambda feats, wins, decoded=None: [
        {"mu": np.zeros((3, 2)), "person": str(w.person_id)} for w in wins])
    monkeypatch.setattr(S, "latent_dead_reckoning", lambda tr, rows: {"mu_z": dict(mz), "pre_ln": dict(mz)})


@pytest.mark.parametrize("extra, factors, tag", [
    ([], (0.8, 1.25), ""),
    (["--mixture", "--latent", "--tag", "mix_latent"], (0.8, 1.25), "_mix_latent"),      # the J2 mix_latent jobs
    (["--warps", "1.25,1.5", "--tag", "warp15"], (1.25, 1.5), "_warp15"),               # warp15_B
    (["--warps", "1.25,1.5", "--latent", "--tag", "w"], (1.25, 1.5), "_w"),
    (["--no-warp", "--latent", "--warps", "1.5"], None, "")])
def test_main_writes_the_report_the_json_and_the_warp_rows(monkeypatch, tmp_path, extra, factors, tag):
    """The latent block's rows never reach the warp rows file (they once shared a variable, and a --latent run
    stopped on KeyError 'factor' after the probe); every run writes md and json, the warp rows when warped."""
    import json
    _stub_main(monkeypatch)
    monkeypatch.setattr("sys.argv", ["probe_root_motion.py", "--device", "cpu", "--out", str(tmp_path), *extra])
    S.main()
    stem = tmp_path / f"root_motion_init{tag}"
    doc = json.load(open(str(stem) + ".json", encoding="utf-8"))
    md = open(str(stem) + ".md", encoding="utf-8").read()
    assert ("--latent" in extra) == (doc["latent"] is not None)
    wj = str(stem) + "_warps.json"
    if factors is None:
        assert not os.path.exists(wj) and doc["warp"] is None and "warps" not in doc and "Time warp" not in md
        assert "warp_rows" not in doc
        return
    rows = json.load(open(wj, encoding="utf-8"))
    assert rows["warps"] == list(factors) and [r["factor"] for r in rows["rows"]] == [1.0, *factors]
    assert all(r["windows"] == 3 for r in rows["rows"]) and rows["checkpoint"] == {"path": "init", "sha256": None}
    for r in rows["rows"][1:]:
        assert r["response_error"] == pytest.approx(doc["warp"][str(r["factor"])]["response_error"], abs=1e-12)
    assert f"the unwarped one included: {os.path.basename(wj)}." in md
    # the main json holds the same rows, with a stride error at 1.0 and at each factor: what menu item M1 of
    # scripts/weekend_branch.py reads from root_motion_<name>_warp15.json (a list of rows with factor under rows,
    # warp_rows or warps; stride_mae_cm in the rows at 1.0 and 1.5)
    assert doc["warp_rows"] == rows["rows"]
    for r in doc["warp_rows"]:
        assert r["strides"] == 48 and r["stride_mae_cm"] == pytest.approx(0.0, abs=1e-9)
    assert "Stride MAE of the decoded body at the true ICs: x1 0.00 cm" in md
    custom = tuple(factors) != tuple(S.WARPS)
    assert ("gate_note" in doc) == custom == ("warps" in doc) == ("Note: time-warp factors" in md)
