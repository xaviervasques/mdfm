"""scripts/jp0_read.py (task JP0-READ) on synthetic inputs: the section 3a hash refusal, the stop on a missing F1
results file before any J0 output is opened, the refit tolerance stop, the direction of F1's gain, the margin rule,
the pass set of a chain, and routing rule 3 for every reading of R. No J0 output, target or record is read."""
import hashlib
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("jp0_read")

DOC = "# t\n\n## 3. a\nx\n\n## 3a. Addendum\n\nbody line\n- item\n\n## 4. next\ny\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def synthetic_tree(tmp_path, doc=DOC, record_sha=None):
    """A tree holding a proposal and its section 3a record; returns (root, sha of section 3a of doc)."""
    root = str(tmp_path)
    write(os.path.join(root, M.PROPOSAL_REL), doc)
    sha = M.addendum_sha(doc)
    write(os.path.join(root, M.RECORD_REL), f"{record_sha or sha}  {M.PROPOSAL_REL} section 3a\n")
    return root, sha


# ---------------------------------------------------------------- hash
def test_section_hash_rule():
    assert M.addendum_sha(DOC) == hashlib.sha256(b"\nbody line\n- item\n\n").hexdigest()
    assert M.addendum_sha("no section\n") is None
    assert M.addendum_sha(DOC.replace("\n", "\r\n")) != M.addendum_sha(DOC)


def test_hash_refusal(tmp_path):
    root, sha = synthetic_tree(tmp_path)
    assert M.check_addendum(root, expected=sha) == sha
    with pytest.raises(M.Stop) as e:                       # the script constant differs from the section
        M.check_addendum(root)
    assert e.value.code == M.EXIT_HASH
    write(os.path.join(root, M.PROPOSAL_REL), DOC.replace("body line", "body  line"))
    with pytest.raises(M.Stop) as e:                       # section edited after its hash was recorded
        M.check_addendum(root, expected=sha)
    assert e.value.code == M.EXIT_HASH
    write(os.path.join(root, M.PROPOSAL_REL), "# no section 3a\n")
    with pytest.raises(M.Stop) as e:
        M.check_addendum(root, expected=sha)
    assert e.value.code == M.EXIT_HASH


def test_hash_refusal_through_main_writes_nothing(tmp_path):
    root, _ = synthetic_tree(tmp_path)
    out = os.path.join(root, "out")
    assert M.main(["--tree", root, "--out", out]) == M.EXIT_HASH
    assert not os.path.exists(out)


def test_recorded_hash_is_the_script_constant():
    with open(os.path.join(ROOT, M.RECORD_REL), encoding="utf-8") as f:
        assert f.read().split()[0] == M.ADDENDUM_SHA256


# ---------------------------------------------------------------- F1 missing
def test_missing_f1_stops_before_any_j0_output(tmp_path, monkeypatch):
    root, sha = synthetic_tree(tmp_path)
    monkeypatch.setattr(M, "ADDENDUM_SHA256", sha)

    def never(*a, **k):
        raise AssertionError("a J0 output was opened")

    monkeypatch.setattr(M, "load_checkpoint_inputs", never)
    monkeypatch.setattr(M, "analyse_checkpoint", never)
    out = os.path.join(root, "reports", "bench", "jp0_reading")
    assert M.main(["--tree", root, "--out", out]) == M.EXIT_F1
    assert not os.path.exists(out)
    with pytest.raises(M.Stop) as e:
        M.read_f1(root)
    assert e.value.code == M.EXIT_F1 and "does not exist" in e.value.message


def f1_results(root, rule_text="rule v1\n", reading="not GAIN", gains=None, rule_sha=None):
    write(os.path.join(root, M.F1_RULE_REL), rule_text)
    sha = rule_sha or hashlib.sha256(rule_text.encode("utf-8")).hexdigest()
    ends = {k: {"name": n, "checkpoint": {"sha256": f"sha_{k}"}, "gain": g} for k, (n, g) in (gains or {}).items()}
    write(os.path.join(root, M.F1_RESULTS_REL), json.dumps({"rule": {"sha256": sha}, "reading": {"reading": reading},
                                                            "ends": ends}))


def test_read_f1_gains_and_rule_check(tmp_path):
    root = str(tmp_path)
    f1_results(root, gains={"v1_B": ("phaseA_root_full_ckpt_B_032044",
                                     {"stride": {"gain": 0.9}, "speed": {"gain": 0.8}, "fitted_on": "train54"}),
                            "v1_A": ("phaseA_root_full_ckpt_A_010681", None)})
    f = M.read_f1(root)
    assert f["reading"] == "not GAIN"
    assert f["gains"]["phaseA_root_full_ckpt_B_032044"]["stride"] == 0.9
    assert f["gains"]["phaseA_root_full_ckpt_A_010681"]["stride"] is None
    f1_results(root, rule_sha="0" * 64)
    with pytest.raises(M.Stop) as e:
        M.read_f1(root)
    assert e.value.code == M.EXIT_INPUT
    f1_results(root, reading="maybe")
    with pytest.raises(M.Stop) as e:
        M.read_f1(root)
    assert e.value.code == M.EXIT_INPUT


# ---------------------------------------------------------------- refit check
def test_refit_tolerance_stop():
    ok = M.check_refit({"dev14": 19.0, "dev12": 18.0}, {"dev14": 19.009, "dev12": 17.991})
    assert abs(ok["dev14"]["difference"] - 0.009) < 1e-9
    for refit in ({"dev14": 19.011, "dev12": 18.0}, {"dev14": 19.0, "dev12": float("nan")}, {"dev14": 19.0}):
        with pytest.raises(M.Stop) as e:
            M.check_refit({"dev14": 19.0, "dev12": 18.0}, refit)
        assert e.value.code == M.EXIT_REFIT
    assert M.REFIT_TOL_CM == 0.01


# ---------------------------------------------------------------- direction of F1's gain
def test_gain_divides_the_decoded_value():
    F1 = _load("g3_gain_check")
    walkway = np.array([100.0, 120.0, 140.0])
    decoded = 0.9 * walkway
    g = F1.median_ratio(walkway, decoded)                   # F1's gain: median of decoded over walkway
    assert abs(g - 0.9) < 1e-12
    rec = M.recalibrate(decoded, g)
    assert np.allclose(rec, walkway)                         # dividing moves the decoded row onto the walkway
    assert np.abs(rec - walkway).mean() < np.abs(decoded - walkway).mean() < np.abs(decoded * g - walkway).mean()
    for bad in (0.0, -1.0, float("nan"), None):
        with pytest.raises(ValueError):
            M.recalibrate(decoded, bad)


# ---------------------------------------------------------------- pass unit and margin rule
def test_score_chain_scores_every_row_on_one_pass_set():
    persons = np.array(["a", "a", "b", "b", "c"])
    ref = np.array([100.0, 110.0, 120.0, np.nan, 130.0])
    vals = {"stem": np.array([101.0, 112.0, 123.0, 1.0, 134.0]),
            "latent": np.array([100.0, 111.0, 121.0, 1.0, 131.0]),
            "decoded": np.array([90.0, np.nan, 110.0, 1.0, 120.0]),
            "supervised": np.array([100.0, 100.0, np.nan, 1.0, 130.0])}
    lines, counts, keep = M.score_chain(persons, ref, vals, {"all": ["a", "b", "c"], "ab": ["a", "b"]},
                                        ("stem", "latent", "decoded"), n_boot=200)
    assert keep.tolist() == [True, False, True, False, True]
    assert lines["all"]["latent"]["per_person"] == {"a": 0.0, "b": 1.0, "c": 1.0}
    assert lines["all"]["decoded"]["per_person"] == {"a": 10.0, "b": 10.0, "c": 10.0}
    assert lines["all"]["supervised"]["per_person"] == {"a": 0.0, "c": 0.0}
    assert counts["all"]["with_reference"] == 4 and counts["all"]["in_chain"] == 3 and counts["all"]["left_out"] == 1
    assert counts["all"]["left_out_by_row"]["decoded"] == 1
    assert lines["ab"]["latent"]["n_persons"] == 2


def line(pp):
    return {"point": float(np.mean(list(pp.values()))), "per_person": pp}


def test_relation_margin_and_readings():
    rng = np.random.default_rng(3)
    ids = [f"p{i:02d}" for i in range(12)]
    lat = {e: line({p: 10.0 + rng.normal(0, 0.2) for p in ids}) for e in M.B_ENDS}
    far = {e: line({p: lat[e]["per_person"][p] + 5.0 + rng.normal(0, 0.2) for p in ids}) for e in M.B_ENDS}
    near = {e: line({p: lat[e]["per_person"][p] + 0.3 for p in ids}) for e in M.B_ENDS}
    r = M.relation(lat, far, n_boot=500)
    assert r["relation"] == "holds" and M.reading_of(r["relation"]) == M.CARRY
    d = max(abs(lat["v1_B"]["point"] - lat["s2_B"]["point"]), abs(far["v1_B"]["point"] - far["s2_B"]["point"]))
    assert abs(r["m"] - max(0.10 * far["v1_B"]["point"], 2 * d)) < 1e-12
    r = M.relation(lat, near, n_boot=500)
    assert r["relation"] == "fails" and M.reading_of(r["relation"]) == M.BOTTLENECK
    mixed = {"v1_B": far["v1_B"], "s2_B": line({p: lat["s2_B"]["per_person"][p] + 0.3 for p in ids})}
    assert M.relation(lat, mixed, n_boot=500)["relation"] in ("split", "fails")
    split = M.relation({"v1_B": lat["v1_B"], "s2_B": lat["v1_B"]}, {"v1_B": far["v1_B"],
                       "s2_B": line({p: lat["v1_B"]["per_person"][p] + 0.3 for p in ids})}, n_boot=500)
    assert split["d"] > 0
    assert M.relation(lat, {"v1_B": None, "s2_B": None})["relation"] == "not evaluated"
    assert M.reading_of("not evaluated") == M.PENDING


def test_split_relation():
    ids = [f"p{i:02d}" for i in range(12)]
    x = {e: line({p: 10.0 for p in ids}) for e in M.B_ENDS}
    y = {"v1_B": line({p: 13.0 + 0.01 * i for i, p in enumerate(ids)}),
         "s2_B": line({p: 10.5 + 0.01 * i for i, p in enumerate(ids)})}
    r = M.relation(x, y, n_boot=500)                        # m = max(1.3, 2 x 2.5) = 5.0: neither end clears it
    assert r["relation"] == "fails"
    y["s2_B"] = line({p: 13.0 + 0.01 * i for i, p in enumerate(ids)})
    y["v1_B"] = line({p: 14.0 + 0.01 * i for i, p in enumerate(ids)})
    r = M.relation(x, y, n_boot=500)                        # d 1.0, m max(1.4, 2.0) = 2.0: G 4.0 and 3.0
    assert r["relation"] == "holds"
    y["s2_B"] = line({p: 11.5 + 0.01 * i for i, p in enumerate(ids)})
    y["v1_B"] = line({p: 13.0 + 0.01 * i for i, p in enumerate(ids)})
    r = M.relation(x, y, n_boot=500)                        # d 1.5, m 3.0: G 3.0 at v1 (holds), 1.5 at s2
    assert r["ends"]["v1_B"]["better"] and not r["ends"]["s2_B"]["better"]
    assert r["relation"] == "split" and M.reading_of(r["relation"]) == M.SPLIT


# ---------------------------------------------------------------- routing
@pytest.mark.parametrize("R", [M.PENDING, M.BOTTLENECK, M.SPLIT, M.CARRY])
def test_routing_rule_3_for_every_R(R):
    rt = M.route("not GAIN", M.BOTTLENECK, R)
    assert rt["rule"] == 3
    assert rt["e1_screen"].startswith("eligible")


def test_routing_other_rules():
    assert M.route("not GAIN", M.CARRY, M.CARRY)["rule"] == 2
    assert M.route("not GAIN", M.CARRY, M.PENDING)["rule"] == 2
    assert M.route("not GAIN", M.CARRY, M.SPLIT)["rule"] == 2
    assert M.route("not GAIN", M.CARRY, M.BOTTLENECK)["rule"] == "2R"
    assert M.route("not GAIN", M.CARRY, M.BOTTLENECK)["e1_screen"] == "does not run"
    for R in (M.PENDING, M.BOTTLENECK, M.SPLIT, M.CARRY):
        assert M.route("not GAIN", M.SPLIT, R)["rule"] == 4
    g = M.route("GAIN", M.BOTTLENECK, M.CARRY)
    assert g["rule"] == 1 and g["after_gain_explained"]["rule"] == 2
    assert M.route("GAIN", M.CARRY, M.BOTTLENECK)["after_gain_explained"]["rule"] == 3
    assert M.route("not evaluated", M.BOTTLENECK, M.PENDING)["rule"] is None


def test_sub_readings_and_section3():
    s = M.sub_readings(M.BOTTLENECK, "holds", "fails")
    assert s["LM"] is True and s["SL"] is True and len(s["implications"]) == 3
    s = M.sub_readings(M.CARRY, "fails", "split")
    assert s["LM"] is False and s["SL"] == "split"
    r = M.section3_reading(19.51, 15.0, 15.5)                # margin max(1.951, 1.0): 4.51 >= 1.951
    assert r["reading"] == M.CARRY and abs(r["margin"] - 1.951) < 1e-12
    assert M.section3_reading(19.51, 18.0, 18.1)["reading"] == M.BOTTLENECK
    assert M.section3_reading(None, 1.0, 1.0)["reading"] == "not evaluated"


def test_script_text_style():
    raw = open(os.path.join(ROOT, "scripts", "jp0_read.py"), "rb").read()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_real_tree_copy_stops_on_missing_f1(tmp_path):
    """The proposal and record of this tree, copied: the hash passes, and without F1 the run stops with EXIT_F1."""
    root = str(tmp_path)
    for r in (M.PROPOSAL_REL, M.RECORD_REL):
        src = os.path.join(ROOT, r)
        if not os.path.exists(src):
            pytest.skip("no proposal or record in this tree")
        os.makedirs(os.path.dirname(os.path.join(root, r)), exist_ok=True)
        shutil.copyfile(src, os.path.join(root, r))
    assert M.main(["--tree", root, "--out", os.path.join(root, "out")]) == M.EXIT_F1
    assert not os.path.exists(os.path.join(root, "out"))


# ---------------------------------------------------------------- end to end on a synthetic tree
def _world(root, rng):
    """Synthetic controls, targets, J0 outputs, diag events, supervised baseline and F1 results under root. Returns
    (train, dev, targets by set)."""
    import pandas as pd

    from mdfm.eval import probes
    from mdfm.gates import g3
    PFT = M._sibling("probe_frozen_tokens")
    train = [f"weargait:HC{i:03d}" for i in range(54)]
    dev = sorted([f"weargait:WHC{i:03d}" for i in range(100, 112)] + list(M.EXCLUDED_FROM_12))

    def targets(persons):
        tr, st, ps = [], [], []
        for p in persons:
            base = 1.15 + 0.25 * rng.random()
            for task, f in (("SelfPace", 1.0), ("HurriedPace", 1.15)):
                tid = f"{p.split(':')[1]}_{task}"
                tr.append((tid, p, task, 6000))
                for k in range(2):
                    t = 10.0 * k
                    for j in range(4):
                        st.append((tid, p, task, str(k + 1), "LR"[j % 2], t + j, t + j + 1.0,
                                   base * f + 0.03 * rng.normal()))
                    ps.append((tid, p, task, str(k + 1), t, t + 5.0, 6, base * f / 1.1 + 0.02 * rng.normal()))
        trials = pd.DataFrame(tr, columns=["trial_id", "person_id", "task", "samples"])
        strides = pd.DataFrame(st, columns=["trial_id", "person_id", "task", "pass_id", "foot", "t0", "t1", "length_m"])
        passes = pd.DataFrame(ps, columns=["trial_id", "person_id", "task", "pass_id", "t0", "t1", "n_footfalls",
                                           "speed_m_s"])
        ic = pd.DataFrame(columns=["trial_id", "person_id", "task", "foot", "t_s"])
        off = pd.DataFrame(columns=["trial_id", "offset_ms", "n_matched"])
        return g3.Targets(tuple(sorted(persons)), False, trials, ic, ic.copy(), strides, passes, off,
                          {"strides": len(strides), "passes": len(passes), "lockbox_rows": 0})

    tg = {"train": targets(train), "dev": targets(dev)}
    u = {s: probes.pass_units(tg[s]) for s in ("train", "dev")}
    sets = {"train": train, "dev14": dev, "dev12": [p for p in dev if p not in M.EXCLUDED_FROM_12]}
    tgs = {"dev14": tg["dev"], "dev12": probes.subset_targets(tg["dev"], sets["dev12"])}
    names = [n for _, n, _ in M.CHECKPOINTS]
    for i, name in enumerate(names):
        noise = {"stem": 0.08, "measurement": 0.06, "latent": 0.02 + 0.002 * i, "mu_z": 0.05, "subject": 0.1}
        X = {}
        for s in ("train", "dev"):
            X[s] = {}
            n = len(u[s])
            for v, sd in noise.items():
                cols = [u[s]["stride_m"].to_numpy() + sd * rng.normal(size=n),
                        u[s]["speed_m_s"].to_numpy() + sd * rng.normal(size=n)] + [rng.normal(size=n) for _ in range(4)]
                X[s][v] = np.stack(cols, axis=1)
        X["dev"]["stem"][3] = np.nan                       # one dev pass without a stem row: left out of the chain
        data = {"units": u}
        for s in ("train", "dev"):
            data[s] = {"X": X[s], "n_rows": np.full(len(u[s]), 20), "sex": np.zeros(len(u[s]), np.int64),
                       "counts": {"passes": len(u[s])},
                       "tokens": {"trial_id": np.zeros(0, str), "person_id": np.zeros(0, str), "start": np.zeros(0),
                                  "latent": np.zeros((0, 2), np.float16), "mu_z": np.zeros((0, 2), np.float32)}}
        feat_rel = f"cache/runs/frozen_tokens/{name}/features.npz"
        PFT.save_features(os.path.join(root, feat_rel), data, {"checkpoint": {"name": name}})
        loaded, _ = PFT.load_features(os.path.join(root, feat_rel))
        loaded["units"] = u
        preds = M.refit_probes(loaded, u["train"], u["dev"])
        lines = M.per_stride_lines(u["dev"], preds, tgs)
        latent_pp = probes.score(tgs["dev12"], [probes.pass_predictions(u["dev"], preds["latent"], tgs["dev12"], "x")])
        res = {"partial": False, "checkpoint": {"name": name, "sha256": f"sha_{name}", "path": f"cache/{name}.pt"},
               "sets": sets, "features": {"path": feat_rel, "sha256": M.sha256_file(os.path.join(root, feat_rel))},
               "passes": {v: {"lines": {s: {ln: {"point": lines[v][s][ln]} for ln in ("stride_mae_cm", "speed_mae_m_s")}
                                        for s in ("dev14", "dev12")}} for v in M.PROBES}}
        res["passes"]["latent"]["per_person"] = {"dev12": {"stride_mae_cm":
                                                           latent_pp["x"]["stride_mae_cm"]["per_person"]}}
        d = M.j0_dir(root, name)
        write(os.path.join(d, "results.json"), json.dumps(res))
        pp = u["dev"].assign(n_rows=20, pred_stride_m=preds["latent"]["stride_m"],
                             pred_speed_m_s=preds["latent"]["speed_m_s"])
        pp.to_csv(os.path.join(d, "per_pass.csv"), index=False, lineterminator="\n")
        if "004000" in name:
            continue                                         # no decoded row at ckpt_B_004000
        s_ = tg["dev"].strides
        ref = 100.0 * s_["length_m"].to_numpy()
        pred = 0.9 * ref + rng.normal(0, 2.0, size=len(ref))
        matched = np.ones(len(ref), bool)
        matched[5] = False
        ev = pd.DataFrame({"row": "base", "kind": "stride", "trial_id": s_["trial_id"], "person_id": s_["person_id"],
                           "task": s_["task"], "foot": s_["foot"], "pass_id": s_["pass_id"], "reference": ref,
                           "prediction": pred, "unit": "cm", "matched": matched})
        p_ = tg["dev"].passes
        ev2 = pd.DataFrame({"row": "base", "kind": "speed", "trial_id": p_["trial_id"], "person_id": p_["person_id"],
                            "task": p_["task"], "foot": "", "pass_id": p_["pass_id"], "reference": p_["speed_m_s"],
                            "prediction": 0.95 * p_["speed_m_s"], "unit": "m/s", "matched": True})
        ev = pd.concat([ev, ev2], ignore_index=True)
        dd = M.diag_dir(root, name)
        os.makedirs(dd, exist_ok=True)
        ev.to_csv(os.path.join(dd, "events.csv"), index=False, lineterminator="\n")
        k = ev[(ev["kind"] == "stride") & ev["matched"].astype(bool)]
        base = {}
        for sub, members in (("all14", sets["dev14"]), ("no_whc003_whc026", sets["dev12"])):
            kk = k[k["person_id"].isin(members)]
            per = (kk["prediction"] - kk["reference"]).abs().groupby(kk["person_id"]).mean()
            base[sub] = {"stride_mae_cm": float(per.mean()),
                         "per_person": {p: {"stride_cm": float(v)} for p, v in per.items()}}
        write(os.path.join(dd, "diag.json"), json.dumps({"checkpoint": {"sha256": f"sha_{name}", "name": name},
                                                         "rows": {"base": base}}))
    sup = u["dev"].assign(pred_stride_m=u["dev"]["stride_m"] + 0.05, pred_speed_m_s=u["dev"]["speed_m_s"])
    lines_sup = {s: {"supervised": {"stride_mae_cm": {"point": 5.0}}} for s in ("dev12", "dev14")}
    write(os.path.join(root, M.SUPERVISED_REL), json.dumps({"per_pass_dev": sup.to_dict("records"), "lines": lines_sup}))
    gain = {"stride": {"gain": 0.9}, "speed": {"gain": 0.95}}
    f1_results(root, gains={"v1_B": (names[1], gain), "s2_B": (names[2], gain), "v1_A": (names[0], None)})
    with open(os.path.join(root, M.F1_RESULTS_REL), encoding="utf-8") as f:
        doc = json.load(f)
    for k_, n_ in (("v1_B", names[1]), ("s2_B", names[2]), ("v1_A", names[0])):
        doc["ends"][k_]["checkpoint"]["sha256"] = f"sha_{n_}"
    write(os.path.join(root, M.F1_RESULTS_REL), json.dumps(doc))
    return train, dev, tg


@pytest.fixture
def world(tmp_path, monkeypatch):
    import sys
    import types

    from mdfm.gates import g3
    root, sha = synthetic_tree(tmp_path)
    monkeypatch.setattr(M, "ADDENDUM_SHA256", sha)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    train, dev, tg = _world(root, np.random.default_rng(11))
    monkeypatch.setattr(g3, "controls", lambda fold: list(train if fold == "train" else dev))
    monkeypatch.setattr(g3, "lockbox_persons", lambda: ["weargait:LOCKED"])
    monkeypatch.setattr(g3, "load_targets", lambda persons: tg["train" if set(persons) == set(train) else "dev"])
    tf = os.path.join(root, "targets.csv")
    write(tf, "synthetic\n")
    monkeypatch.setattr(g3, "TARGET_FILES", (tf,))
    monkeypatch.setattr(g3, "SUMMARY_JSON", tf)
    monkeypatch.setitem(sys.modules, "_jp0_moveatlas_stamp", types.SimpleNamespace(stamp=lambda: {"head": "test"}))
    return root


def test_end_to_end_on_a_synthetic_tree(world):
    out = os.path.join(world, "out")
    assert M.main(["--tree", world, "--out", out]) == 0
    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        r = json.load(f)
    assert set(r["checkpoints"]) == {"v1_A", "v1_B", "s2_B", "v1_B4000", "s2_B4000"}
    assert r["readings"]["GAIN"] == "not GAIN"
    assert r["readings"]["D"] in (M.CARRY, M.BOTTLENECK, M.SPLIT)
    assert r["readings"]["R"] in (M.CARRY, M.BOTTLENECK, M.SPLIT)
    assert r["route"]["rule"] in (2, "2R", 3, 4)
    v1 = r["checkpoints"]["v1_B"]
    lines = v1["stride"]["lines"]["dev12"]
    assert lines["recalibrated"]["point"] < lines["decoded"]["point"]          # dividing by 0.9 repairs the scale
    assert v1["stride"]["pass_set"]["dev14"]["left_out_by_row"]["stem"] >= 1
    assert not r["checkpoints"]["v1_B4000"]["has_decoded"] and r["jp2_go_line"]["m"] > 0
    assert r["checkpoints"]["v1_A"]["has_recalibrated"] is False
    for c in r["checkpoints"].values():
        assert all(abs(v["difference"]) <= M.REFIT_TOL_CM for v in c["refit"]["check"].values())
    with open(os.path.join(out, "_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["addendum_sha256"] == M.ADDENDUM_SHA256 and prov["moveatlas_head"] == "test"
    text = open(os.path.join(out, "summary.md"), "rb").read()
    assert b"\r" not in text and chr(0x2014).encode("utf-8") not in text


def test_end_to_end_refit_stop(world):
    p = os.path.join(M.j0_dir(world, "phaseA_root_full_ckpt_B_032044"), "results.json")
    with open(p, encoding="utf-8") as f:
        res = json.load(f)
    res["passes"]["latent"]["lines"]["dev12"]["stride_mae_cm"]["point"] += 0.02
    write(p, json.dumps(res))
    out = os.path.join(world, "out")
    assert M.main(["--tree", world, "--out", out]) == M.EXIT_REFIT
    assert not os.path.exists(out)
