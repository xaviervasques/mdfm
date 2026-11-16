"""Tests of scripts/gate_g5.py (task G5S): the G5 command line on synthetic stand-ins only. The care_pd read is
refused before the checkpoint, the index, the label table or any record opens while the rule is unregistered
(the tree's registry while it holds no G5 row, and temporary registries in every state); the allowed read runs on
synthetic rows and records under a temporary registered rule, registry, tier table, label table and EVAL log, and
logs the read before the first record. The self-test on synthetic walkers reads no data and gives a zero drop for
the identity arm and a kill for the body that carries nothing. On AMASS-dev-like stand-ins (DevStore in place of
g5.amass_dev_walks) the persons are checked against the lockbox and the training lists before any take loads, and a
checkpoint without the model arm is refused. The model predictor runs on the tiny model of mdfm/model/mfm.py on the
CPU (when the SMPL+H files are present): camera windows of the render, inference, the 15 joints at the walk's frames
at 30, 100 and 25 Hz, and the model arm of the self-test with its hashes. Rule registration is tested on temporary
files only."""
import collections
import csv
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

from mdfm import paths
from mdfm.data import canonical
from mdfm.data.canonical import Refused
from mdfm.eval import infer as I
from mdfm.gates import common, g5
from mdfm.model import mfm as M

TREE = paths.TREE
SCRIPT = os.path.join(TREE, "scripts", "gate_g5.py")
HAVE_MA = os.path.exists(paths.smplh_model("neutral"))
needs_ma = pytest.mark.skipif(not HAVE_MA, reason="SMPL+H model not found")


def _script():
    spec = importlib.util.spec_from_file_location("gate_g5_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


def _g5_registered():
    """True when the tree's registry holds a G5 row (the owner's act, never a session's)."""
    return any(r["gate"] == "G5" for r in common.registry())


# ---------------------------------------------------------------- stand-ins
def care_pd_like(n_persons=10, seconds=4.0):
    """Synthetic walks named like care_pd BMCLab walks (labels and states built in), for the read path."""
    out = []
    for w in S.synthetic_walks(n_persons, seconds):
        p = int(w.person.rsplit("P", 1)[1])
        person = f"care_pd:BMCLab:SUB{p:02d}"
        key = w.key.replace(f"P{p:02d}_", f"SUB{p:02d}_")
        out.append(g5.Walk(person=person, key=key, fps=w.fps, coords=w.coords, valid=w.valid, scale=w.scale,
                           cohort="BMCLab", label=w.label, state=w.state))
    return out


class Readers:
    """Counting stand-ins of canonical.index and canonical.load serving synthetic care_pd-like walks."""

    def __init__(self, walks, log=None):
        self.calls = collections.Counter()
        self.walks = {w.key: w for w in walks}
        self.log = log
        self.logged_rows_at_first_load = None

    def index(self):
        import pandas as pd
        self.calls["index"] += 1
        return pd.DataFrame([{"source": "care_pd", "modality": "pose", "subject_ns": w.person, "person_id": w.person,
                              "key": w.key, "tier": "EVAL", "lockbox": False} for w in self.walks.values()])

    def load(self, row, allow_eval=False, allow_lockbox=False):
        self.calls["load"] += 1
        assert allow_eval is True and allow_lockbox is False
        if self.logged_rows_at_first_load is None:
            self.logged_rows_at_first_load = len(common.eval_read_log(self.log)) if self.log else -1
        w = self.walks[row["key"]]
        return {"coords": w.coords, "valid_mask": w.valid, "scale": w.scale, "fps": w.fps}


def _counting(monkeypatch, calls):
    """Every reader of the read path replaced by a counter that fails the test when called."""
    def boom(name):
        def f(*a, **k):
            calls[name] += 1
            raise AssertionError(f"{name} called")
        return f
    monkeypatch.setattr(canonical, "index", boom("index"))
    monkeypatch.setattr(canonical, "load", boom("load"))
    monkeypatch.setattr(g5, "read_labels", boom("read_labels"))
    monkeypatch.setattr(I, "load_checkpoint", boom("load_checkpoint"))
    monkeypatch.setattr(S, "load", boom("script_load"))


def _tiers(tmp_path):
    p = tmp_path / "tiers_by_source.csv"
    p.write_text("source,modality,tier,lane,records,persons,lockbox_records,lockbox_persons\n"
                 "care_pd,pose,EVAL,body,40,10,0,0\namass,pose,RELEASE,body,10,5,2,1\n", encoding="utf-8",
                 newline="\n")
    return str(p)


def _rule(tmp_path, status, name="G5_RULE.md", extra=""):
    p = tmp_path / name
    p.write_text(f"# G5 rule\n\nStatus: {status}\n\nA rule for the tests.\n{extra}", encoding="utf-8", newline="\n")
    return str(p)


def _label_csv(tmp_path, walks):
    """A stand-in label table in the layout of MoveAtlas extract_care_pd.py."""
    p = tmp_path / "labels.csv"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cohort", "subject", "walk", "n_frames", "fps", "UPDRS_GAIT"])
        for x in walks:
            cohort, subject = g5.split_subject_ns(x.person)
            w.writerow([cohort, subject, x.key, len(x.coords), x.fps, int(x.label)])
    return str(p)


def _checkpoint(tmp_path, amass=("amass:person:X:Y",)):
    """An empty checkpoint file with its persons/ lists (eval_g4_dev.person_lists reads them)."""
    run = tmp_path / "run"
    (run / "persons").mkdir(parents=True)
    (run / "persons" / "amass.txt").write_text("\n".join(amass) + "\n", encoding="utf-8", newline="\n")
    ck = run / "ckpt_B_000010.pt"
    ck.write_bytes(b"")
    return str(ck)


def _loaded(path, persons=None, model=None):
    return I.Loaded(model=torch.nn.Linear(2, 2) if model is None else model, path=path,
                    name=I.checkpoint_name(path), sha256="c" * 64,
                    weights="ema", step=10, phase="B", phase_steps=(5, 5), config_json="{}",
                    config_sha256="d" * 64, persons=persons or {"amass": ["amass:person:X:Y"]},
                    persons_sha256={"all": "e" * 64}, hashes={}, drift=[])


class IdentityModel:
    """The identity projection in the shape of G5Predictor (log, left_out)."""

    def __init__(self):
        self.log, self.left_out = [], []

    def __call__(self, walk, renders):
        assert renders is not None and len(renders) == 1 and renders[0] is not None
        self.log.append({"person_id": walk.person, "key": walk.key, "view": renders[0]["view"], "windows": 1,
                         "seconds": 0.0, "uncovered_frames": 0})
        return walk.joints_m()


def amass_like(n_persons=5, seconds=2.0, fps=100.0):
    """Synthetic walks named like AMASS dev takes at the body-store rate, with no label and no state (g5.selftest
    assigns them): the stand-ins served by DevStore."""
    out = []
    for w in S.synthetic_walks(n_persons, seconds, fps=fps):
        p = int(w.person.rsplit("P", 1)[1])
        out.append(g5.Walk(person=f"amass:person:DEV:{p:02d}", key=f"dev_{w.key}", fps=w.fps, coords=w.coords,
                           valid=w.valid, scale=w.scale, cohort="amass_dev"))
    return out


class DevStore:
    """A stand-in of g5.amass_dev_walks under its check_persons contract: the check gets the person ids before any
    take loads (seen: the lists it got); loads counts the takes served."""

    def __init__(self, walks):
        self.walks, self.loads, self.seen = walks, 0, []

    def __call__(self, n_persons=None, max_walks_per_person=None, rate=100.0, check_persons=None):
        persons = sorted({w.person for w in self.walks})
        persons = persons[:n_persons] if n_persons else persons
        if check_persons is not None:
            self.seen.append(list(persons))
            check_persons(list(persons))
        out = [w for w in self.walks if w.person in persons]
        self.loads += len(out)
        return out


def _no_text_dash(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return common.EM_DASH not in text and chr(0x2013) not in text


# ---------------------------------------------------------------- the read path
def test_the_read_path_refuses_while_the_rule_is_unregistered(tmp_path, monkeypatch):
    calls = collections.Counter()
    _counting(monkeypatch, calls)
    log = str(tmp_path / "eval_reads.csv")
    monkeypatch.setattr(common, "EVAL_LOG", log)
    ck = _checkpoint(tmp_path)
    out = str(tmp_path / "out")
    argv = ["--read", "--checkpoint", ck, "--out", out, "--device", "cpu"]
    # the tree's own registry, rule and tier table (read only), while the shipped rule is unregistered
    if not _g5_registered():
        with pytest.raises(Refused, match="registered rule"):
            S.main(argv)
    # temporary registries: empty, another gate's rule, a draft, a registered rule edited since
    reg, lock = str(tmp_path / "hashes.csv"), str(tmp_path / "lockbox.csv")
    monkeypatch.setattr(common, "RULE_HASHES", reg)
    monkeypatch.setattr(common, "READ_LOG", lock)
    monkeypatch.setattr(common, "TIERS", _tiers(tmp_path))
    with pytest.raises(Refused):
        S.main(argv)
    other = _rule(tmp_path, "APPROVED by the owner on 2026-10-05.", name="other.md", extra="Another gate.\n")
    common.register_rule("G4", other, "2026-10-05", registry_path=reg, log_path=lock)
    monkeypatch.setattr(g5, "RULE_PATH", other)
    with pytest.raises(Refused):
        S.main(argv)
    monkeypatch.setattr(g5, "RULE_PATH", _rule(tmp_path, "DRAFT for the owner, not approved, not hashed.",
                                               name="draft.md"))
    with pytest.raises(Refused):
        S.main(argv)
    approved = _rule(tmp_path, "APPROVED by the owner on 2026-10-05.", name="approved.md")
    common.register_rule("G5", approved, "2026-10-05", registry_path=reg, log_path=lock)
    with open(approved, "a", encoding="utf-8", newline="\n") as f:
        f.write("An edit after registration.\n")
    monkeypatch.setattr(g5, "RULE_PATH", approved)
    with pytest.raises(Refused, match="changed since its registration"):
        S.main(argv)
    assert calls == collections.Counter()                           # no checkpoint, index row, label or record
    assert not os.path.exists(log) and not os.path.exists(out)


def test_the_read_path_under_a_registered_rule_logs_before_the_first_record(tmp_path, monkeypatch):
    reg, lock = str(tmp_path / "hashes.csv"), str(tmp_path / "lockbox.csv")
    log = str(tmp_path / "g" / "eval_reads.csv")
    rule = _rule(tmp_path, "APPROVED by the owner on 2026-10-05.")
    rule_sha = common.register_rule("G5", rule, "2026-10-05", registry_path=reg, log_path=lock)["sha256"]
    walks = care_pd_like()
    readers = Readers(walks, log=log)
    for name, value in (("RULE_HASHES", reg), ("READ_LOG", lock), ("EVAL_LOG", log), ("TIERS", _tiers(tmp_path))):
        monkeypatch.setattr(common, name, value)
    monkeypatch.setattr(g5, "RULE_PATH", rule)
    monkeypatch.setattr(g5, "LABELS_CSV", _label_csv(tmp_path, walks))
    monkeypatch.setattr(canonical, "index", readers.index)
    monkeypatch.setattr(canonical, "load", readers.load)
    monkeypatch.setattr(canonical, "cody_namespaces", lambda: frozenset({"adult", "cody_external"}))
    ck = _checkpoint(tmp_path)
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck), torch.device("cpu"), None))
    monkeypatch.setattr(S, "model_predictor", lambda loaded, a, echo=None: IdentityModel())
    out = str(tmp_path / "out")
    # a checkpoint whose stored lists hold a scored person is refused before the read is logged
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck, {"care_pd": [walks[0].person]}),
                                              torch.device("cpu"), None))
    with pytest.raises(Refused):
        S.main(["--read", "--checkpoint", ck, "--out", out, "--device", "cpu", "--n-boot", "40"])
    assert not os.path.exists(log) and readers.calls["load"] == 0 and not os.path.exists(out)
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck), torch.device("cpu"), None))
    got_out, results = S.main(["--read", "--checkpoint", ck, "--out", out, "--device", "cpu", "--n-boot", "40"])
    assert got_out == out and readers.logged_rows_at_first_load == 1      # logged before the first record opened
    rows = common.eval_read_log(log)
    assert len(rows) == 1 and rows[0]["gate"] == "G5" and rows[0]["source"] == "care_pd"
    assert rows[0]["rule_sha256"] == rule_sha and rows[0]["checkpoint_sha256"] == "c" * 64
    assert rows[0]["code_sha256"] and "gate_g5.py --read" in rows[0]["note"]
    assert readers.calls["load"] == 40
    assert all(r.point == 0.0 and r.verdict == "pass" and r.rule_sha256 == rule_sha for r in results)
    for fn in ("g5.json", "g5.md", "per_walk.csv", "trials.csv", "left_out.csv", "_provenance.json"):
        assert os.path.exists(os.path.join(out, fn)), fn
        assert _no_text_dash(os.path.join(out, fn)), fn
    with open(os.path.join(out, "g5.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["gate_verdict"] == "pass" and doc["info"]["checkpoint"]["sha256"] == "c" * 64
    assert doc["info"]["disjoint_from_training_persons"] == {"care_pd": 0}
    assert doc["info"]["inference"]["calls"] == 80                     # 40 walks x 2 views


# ---------------------------------------------------------------- the self-test
def test_the_selftest_on_synthetic_walkers_reads_no_data(tmp_path, monkeypatch):
    calls = collections.Counter()
    _counting(monkeypatch, calls)
    monkeypatch.setattr(g5, "amass_dev_walks", lambda *a, **k: (_ for _ in ()).throw(AssertionError("amass read")))
    monkeypatch.setattr(g5, "read_care_pd", lambda *a, **k: (_ for _ in ()).throw(AssertionError("care_pd read")))
    out = str(tmp_path / "selftest")
    got_out, doc = S.main(["--selftest", "--sets", "synthetic", "--quick", "--out", out, "--n-boot", "40"])
    assert got_out == out and calls == collections.Counter()
    assert doc["arms"] == ["identity", "static"] and doc["checkpoint"] is None and doc["quick"] is True
    lines = collections.defaultdict(list)
    for r in doc["lines"]:
        lines[r["arm"]].append(r)
    assert len(lines["identity"]) == len(lines["static"]) == 4
    for r in lines["identity"]:
        assert r["point"] == 0.0 and r["interval"] == [0.0, 0.0] and r["verdict"] == "pass", r["line"]
    for r in lines["static"]:                                          # a body that carries nothing: the trivial drop
        assert r["point"] == pytest.approx(r["trivial_drop"]) and r["verdict"] == "kill", r["line"]
    assert doc["sets"]["synthetic"]["walks"] == 40 and doc["sets"]["synthetic"]["persons"] == 10
    for fn in ("g5_selftest.json", "g5_selftest.md", "trials.csv", "left_out.csv", "_provenance.json",
               "synthetic_identity/g5.json", "synthetic_identity/per_walk.csv", "synthetic_static/g5.md"):
        assert os.path.exists(os.path.join(out, fn)), fn
        assert _no_text_dash(os.path.join(out, fn)), fn
    with open(os.path.join(out, "_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["checkpoint_sha256"] is None and "scripts/gate_g5.py" in prov["code_files"]


def test_the_command_line_defaults_and_refusals(tmp_path):
    a = S.parse(["--selftest"])
    assert a.n_boot == g5.BOOT_N and a.synthetic_persons == 20 and a.n_persons is None and a.sets == list(S.SETS)
    q = S.parse(["--selftest", "--quick", "--out", str(tmp_path)])
    assert q.n_boot == 200 and q.synthetic_persons == 10 and q.n_persons == 5 and q.max_walks_per_person == 2
    with pytest.raises(SystemExit):
        S.parse(["--selftest", "--read"])                              # one command at a time
    with pytest.raises(SystemExit, match="scratch folder"):
        S.main(["--selftest", "--quick", "--sets", "synthetic"])
    with pytest.raises(SystemExit, match="checkpoint"):
        S.main(["--selftest", "--arms", "model", "--sets", "synthetic", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="checkpoint"):
        S.main(["--read"])
    assert S.default_out(None, "selftest").endswith(os.path.join("reports", "gates", "g5", "selftest_harness"))
    ld = _loaded(_checkpoint(tmp_path))
    assert S.default_out(ld, "selftest").endswith(os.path.join("g5", "selftest_run_ckpt_B_000010"))
    assert S.default_out(ld, "read").endswith(os.path.join("g5", "run_ckpt_B_000010"))


def test_the_amass_dev_persons_are_checked_against_the_lockbox_and_the_training_lists(monkeypatch):
    monkeypatch.setattr(S.WS, "lockbox_persons", lambda: frozenset({"amass:person:L:1"}))
    assert S.check_dev_persons(["amass:person:A:1", "amass:person:B:2"]) == {"lockbox": 0, "training": None}
    assert S.check_dev_persons(["amass:person:A:1"], {"amass": ["amass:person:Z:9"]}) == {"lockbox": 0,
                                                                                          "training": 0}
    with pytest.raises(Refused, match="lockbox"):
        S.check_dev_persons(["amass:person:A:1", "amass:person:L:1"])
    with pytest.raises(Refused, match="training"):
        S.check_dev_persons(["amass:person:A:1"], {"amass": ["amass:person:A:1"]})


def test_the_selftest_checks_the_amass_dev_persons_before_any_take_loads(tmp_path, monkeypatch):
    """The AMASS dev branch of --selftest: a lockbox person, a person of the lists the checkpoint stores and a person
    of persons/*.txt beside it are each refused from the take index, before any take loads and before any output;
    a checkpoint without the model arm is refused before it loads."""
    calls = collections.Counter()
    _counting(monkeypatch, calls)
    monkeypatch.setattr(g5, "read_care_pd", lambda *a, **k: (_ for _ in ()).throw(AssertionError("care_pd read")))
    walks = amass_like()
    store = DevStore(walks)
    monkeypatch.setattr(g5, "amass_dev_walks", store)
    ck = _checkpoint(tmp_path / "a")
    out = str(tmp_path / "selftest")
    argv = ["--selftest", "--sets", "amass_dev", "--checkpoint", ck, "--device", "cpu", "--out", out, "--n-boot", "40"]
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck), torch.device("cpu"), None))
    monkeypatch.setattr(S.WS, "lockbox_persons", lambda: frozenset({walks[0].person}))
    with pytest.raises(Refused, match="lockbox"):
        S.main(argv)
    monkeypatch.setattr(S.WS, "lockbox_persons", lambda: frozenset())
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck, {"amass": [walks[-1].person]}), torch.device("cpu"), None))
    with pytest.raises(Refused, match="training"):                    # a list the checkpoint stores
        S.main(argv)
    ck_b = _checkpoint(tmp_path / "b", amass=(walks[5].person,))
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck_b), torch.device("cpu"), None))
    with pytest.raises(Refused, match="training"):                    # persons/amass.txt beside the checkpoint
        S.main(["--selftest", "--sets", "amass_dev", "--checkpoint", ck_b, "--device", "cpu", "--out", out])
    persons = sorted({w.person for w in walks})
    assert store.seen == [persons] * 3 and store.loads == 0 and not os.path.exists(out)
    monkeypatch.setattr(S, "load", lambda a: (_ for _ in ()).throw(AssertionError("checkpoint loaded")))
    with pytest.raises(SystemExit, match="without the model arm"):     # never silently ignored
        S.main([*argv, "--arms", "identity", "static"])
    assert calls == collections.Counter() and store.loads == 0 and not os.path.exists(out)


def test_rule_registration_is_the_owners_command_on_temporary_files(tmp_path, monkeypatch):
    reg, lock = str(tmp_path / "hashes.csv"), str(tmp_path / "lockbox.csv")
    monkeypatch.setattr(common, "RULE_HASHES", reg)
    monkeypatch.setattr(common, "READ_LOG", lock)
    monkeypatch.setattr(g5, "RULE_PATH", _rule(tmp_path, "DRAFT for the owner, not approved, not hashed."))
    with pytest.raises(SystemExit, match="approved-on"):
        S.main(["--register-rule"])
    with pytest.raises(Refused):                                     # a draft is never registered
        S.main(["--register-rule", "--approved-on", "2026-10-05"])
    assert not os.path.exists(reg)
    approved = _rule(tmp_path, "APPROVED by the owner on 2026-10-05.", name="approved.md")
    monkeypatch.setattr(g5, "RULE_PATH", approved)
    row = S.main(["--register-rule", "--approved-on", "2026-10-05"])
    assert row["gate"] == "G5" and row["sha256"] == common.sha256_file(approved)
    assert [r["gate"] for r in common.registry(reg)] == ["G5"]


# ---------------------------------------------------------------- the model predictor on the CPU
@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(5)
    model = M.MFM(M.TINY)
    model.eval()
    model.requires_grad_(False)
    return model


@needs_ma
def test_the_model_predictor_reads_camera_windows_and_returns_15_joints_at_the_walks_frames(tiny):
    pred = S.G5Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=1)
    walk = S.as_walk(S.walker(seconds=12.0), "synthetic:walker:P01", "P01_on_walk_1")
    render = g5.render_view(walk, "frontal")
    wins, sp, tc = pred.windows(walk, render)
    assert sp == [(0, 300), (60, 300)] and len(tc) == 360       # 12 s at 30 fps: 10 s windows, the last on the end
    st = wins[0].streams[0]
    assert wins[0].source == "synthetic" and st.modality == "pose2d" and st.meta["layout"] == "camera1"
    assert st.meta["camera"] == "generic" and st.meta["source"] == "synthetic" and st.fps == 30.0
    assert wins[0].cov == {"sex": 2, "age_band": 8, "task": 0}            # sex and age unknown, task walk
    joints = pred(walk, (render,))
    assert joints.shape == (len(walk.coords), 15, 3) and np.isfinite(joints).all()
    row = pred.log[-1]
    assert row["view"] == "frontal" and row["windows"] == 2 and row["uncovered_frames"] == 0
    assert row["kind"] == "g5_frontal" and row["trial_id"] == f"{walk.person}/{walk.key}"
    again = S.G5Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=1)(walk, (render,))
    np.testing.assert_array_equal(again, joints)                          # deterministic under its seed
    # a walk under 1 s gives no window: left out and counted, the harness imputes it
    short = S.as_walk(S.walker(seconds=0.8), "synthetic:walker:P02", "P02_on_walk_1")
    assert pred(short, (g5.render_view(short, "sagittal"),)) is None
    assert pred.left_out[-1]["reason"] == "camera_shorter_than_a_window" and pred.left_out[-1]["view"] == "sagittal"
    with pytest.raises(ValueError, match="one view"):
        pred(walk, (render, g5.render_view(walk, "sagittal")))
    with pytest.raises(ValueError, match="render"):
        pred(walk, (None,))
    # through the harness: one row per walk and body, PA-MPJPE finite, the short walk imputed and counted
    left = []
    table = g5.walk_table([walk, short], pred, view_sets=(("sagittal",),), left_out=left)
    assert len(table) == 4 and [x["reason"] for x in left] == ["no_prediction"]
    sag = table[table.body == "sagittal"]
    assert np.isfinite(sag["pa_mpjpe_mm"].iloc[0]) and bool(sag["left_out"].iloc[1])
    assert pred.left_out[-1]["meaning"] == S.LEFT_OUT["camera_shorter_than_a_window"]


@needs_ma
@pytest.mark.parametrize("fps", [100.0, 25.0])
def test_the_model_predictor_maps_the_30_hz_camera_onto_a_walk_at_another_rate(tiny, fps):
    """A walk at 100 Hz (the AMASS dev rate) or 25 Hz under the 30 fps camera: the joints come at the walk's own
    frames, read at walk.t (not at the camera times), every frame covered."""
    pred = S.G5Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=1)
    walk = S.as_walk(S.walker(seconds=12.0, fps=fps), "synthetic:walker:P01", "P01_on_walk_1", fps=fps)
    render = g5.render_view(walk, "frontal")
    assert render["fps"] == 30.0 and len(render["t"]) != len(walk.coords)
    seen, frames = [], pred.core.frames

    def spy(wins, bodies, t_frames, outputs):
        seen.append(np.array(t_frames, dtype=np.float64))
        return frames(wins, bodies, t_frames, outputs)

    pred.core.frames = spy
    joints = pred(walk, (render,))
    assert joints.shape == (len(walk.coords), 15, 3) and np.isfinite(joints).all()
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], walk.t)                       # the walk's clock, not the camera's
    row = pred.log[-1]
    assert row["frames"] == len(walk.coords) and row["uncovered_frames"] == 0 and row["windows"] == 2
    # the same motion at 30 fps, read at the instants both clocks share, when both renders hold the same camera
    # grid (100 Hz: 360 samples as at 30 fps; 25 Hz ends one sample earlier, so its windows differ and no equality
    # is expected). The tiny model's joints barely move over time: a sanity check of the values, the spy above is
    # the check of the clock.
    w30 = S.as_walk(S.walker(seconds=12.0), "synthetic:walker:P01", "P01_on_walk_1")
    r30 = g5.render_view(w30, "frontal")
    j30 = S.G5Predictor(tiny, refine_steps=1, points=32, batch_windows=2, seed=1)(w30, (r30,))
    if len(r30["t"]) == len(render["t"]):
        g = int(np.gcd(int(fps), 30))
        a, b = joints[::int(fps) // g], j30[::30 // g]
        n = min(len(a), len(b))
        np.testing.assert_allclose(walk.t[::int(fps) // g][:n], w30.t[::30 // g][:n], atol=1e-9)
        np.testing.assert_allclose(a[:n], b[:n], atol=1e-5)
    else:
        assert fps == 25.0 and len(render["t"]) == len(r30["t"]) - 1


@needs_ma
def test_the_model_arm_on_amass_dev_walks_carries_the_checkpoint_hashes_alone(tmp_path, monkeypatch, tiny):
    """The model arm of --selftest on AMASS-dev-like walks at 100 Hz (tiny model, CPU): the persons checked first
    (training 0), the checkpoint and training-person hashes on the model arm's lines only, one inference call per
    walk and view, every walk frame covered."""
    calls = collections.Counter()
    _counting(monkeypatch, calls)
    monkeypatch.setattr(g5, "read_care_pd", lambda *a, **k: (_ for _ in ()).throw(AssertionError("care_pd read")))
    monkeypatch.setattr(S.WS, "lockbox_persons", lambda: frozenset())
    walks = amass_like()
    store = DevStore(walks)
    monkeypatch.setattr(g5, "amass_dev_walks", store)
    ck = _checkpoint(tmp_path)
    monkeypatch.setattr(S, "load", lambda a: (_loaded(ck, model=tiny), torch.device("cpu"), None))
    out = str(tmp_path / "selftest")
    got_out, doc = S.main(["--selftest", "--sets", "amass_dev", "--checkpoint", ck, "--device", "cpu", "--out", out,
                           "--n-boot", "40", "--refine-steps", "1", "--points", "32", "--batch-windows", "2"])
    assert got_out == out and calls == collections.Counter()
    assert store.seen == [sorted({w.person for w in walks})] and store.loads == len(walks)
    assert doc["arms"] == ["identity", "static", "model"]
    assert doc["checks"]["amass_dev"] == {"lockbox": 0, "training": 0}
    for arm in doc["arms"]:
        with open(os.path.join(out, f"amass_dev_{arm}", "g5.json"), encoding="utf-8") as f:
            res = json.load(f)["results"]
        want = ("c" * 64, "e" * 64) if arm == "model" else ("", "")
        assert len(res) == 4 and {(r["checkpoint_sha256"], r["train_persons_sha256"]) for r in res} == {want}, arm
        assert {r["rule_sha256"] for r in res} == {doc["rule"]["sha256"]}, arm
    inf = doc["inference"]
    n_calls = 2 * len(walks)                                               # one per walk and view
    assert inf["calls"] == n_calls and inf["windows"] == n_calls and inf["uncovered_frames"] == 0
    assert inf["device"] == "cpu" and inf["window"]["layout"] == "camera1" and inf["refine_steps"] == 1
    assert doc["runs"]["amass_dev_model"]["inference_calls"] == n_calls
    assert doc["runs"]["amass_dev_identity"]["inference_calls"] == 0
    with open(os.path.join(out, "trials.csv"), encoding="utf-8") as f:
        trials = list(csv.DictReader(f))
    assert len(trials) == n_calls and {int(r["frames"]) for r in trials} == {len(walks[0].coords)}   # 100 Hz frames
    with open(os.path.join(out, "_provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["checkpoint_sha256"] == "c" * 64 and len(prov["inputs"]) == 2
    for fn in ("g5_selftest.json", "g5_selftest.md", "trials.csv", "left_out.csv", "amass_dev_model/g5.md"):
        assert _no_text_dash(os.path.join(out, fn)), fn


def test_the_script_and_its_tests_hold_no_dash():
    for path in (SCRIPT, os.path.abspath(__file__)):
        assert _no_text_dash(path), path
