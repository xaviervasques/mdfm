"""G5 command line (plan task B4.22; rule docs/rules/G5_RULE.md, DRAFT; harness mdfm/gates/g5.py; inference
mdfm/eval/infer.py through the camera-window pieces of scripts/eval_g4_dev.py): the model as the G5 predictor, the
development self-test on synthetic stand-ins and AMASS dev walks, and the care_pd read under the registered rule.

Model predictor (G5Predictor, rule section 4). predictor(walk, renders) gets the tuple of renders of one view set
(mdfm.gates.g5.render_view: a fixed frontal or sagittal camera at 30 fps, the default tracker process). The render
becomes one pose2d input stream as the G1c camera input: canonical 2D coords in normalized pixels with their scale,
layout camera1, camera family 'generic', source 'synthetic' (care_pd never trains, so it has no per-source row),
covariates sex unknown, age unknown and task 'walk' for every walk (no demographic table is read); then
window_store.band_copy. The stream is cut into windows of infer.WIN_S s every infer.HOP_S s from its first sample,
the last aligned on its end (eval_g4_dev.cut_windows, infer.spans); a render under infer.MIN_S s gives no window and
the walk is left out (recorded here, reason camera_shorter_than_a_window with its meaning from LEFT_OUT; the
harness then imputes its descriptors and counts it, reason no_prediction). infer.infer runs the walk's windows
with infer.STEPS_EVAL refinement steps of infer.POINTS_EVAL points per stream, the shape pooled over the walk's
windows, seed infer.seed_of(seed, 'g5_<view>', '<person>/<key>'). The pose3d branch of MFM.render on the point
body (decoded posterior means) gives the 15 canonical joints (m) at the walk's own frames (walk.t, the record's rate), each frame read from the window whose knot
span holds it nearest its centre (eval_g4_dev.G4Predictor.frames); a frame no window covers reads NaN and is
counted (uncovered_frames). The joints stay in each window's frame: every G5 descriptor is root-free, scale-free and
per-frame yaw-free, and PA-MPJPE aligns each frame. One view per call: a view set of two views is refused
(ValueError), since a window holds one pose2d stream (mfm.batch_from_windows).

Self-test (--selftest, rule section 7 and section 8 item 10). Non-EVAL data only, every line computed as the care_pd
lines are (mdfm.gates.g5: walk_table, summarize, gate_results, the deciding reading 'both'):
- sets: 'synthetic', parametric walkers in place (synthetic_walks: a severity level 0 to 3 per person, the
  UPDRS_GAIT stand-in, slows the cadence and shortens steps and arm swing; two OFF and two ON walks per person, ON
  with longer steps and more arm swing); 'amass_dev', the walks of the AMASS dev body store (g5.amass_dev_walks,
  the guard asked for the source first), labelled by g5.assign_selftest_labels (tertiles of the source cadence; ON
  where the source step amplitude is at least the person's median). The AMASS dev persons are refused when one is a
  lockbox person (reports/lockbox/lockbox_persons.csv, names only) or, with a checkpoint, a training person of the
  checkpoint (persons/*.txt beside it and the lists stored in it); the check reads the person ids of the take index
  and runs before any take loads (g5.amass_dev_walks check_persons).
- arms: 'identity' (the source body itself: a drop of exactly 0, the pass check), 'static' (one fixed standing pose
  for every walk, a body that carries nothing: the trivial drop, the kill check), and 'model' (G5Predictor), present
  exactly when a checkpoint is given (a checkpoint without the model arm is refused, so that it is never ignored).
The labels are known-signal labels of the self-test: they decide nothing. Writes <out> (default
reports/gates/g5/selftest_<checkpoint name>, suffix _model for the raw weights, or reports/gates/g5/selftest_harness
without a checkpoint): g5_selftest.json and g5_selftest.md (every line of every set and arm, the PA-MPJPE
diagnostic, the sets, the walks left out, the inference log), one folder per set and arm <set>_<arm>/ (g5.json,
g5.md, per_walk.csv of g5.write_outputs), trials.csv (one row per inference call of the model arm), left_out.csv,
_provenance.json.

Registered read (--read). The EVAL read guard first (common.guard_eval_read for G5 on care_pd), before the
checkpoint loads: while docs/rules/G5_RULE.md is not registered for G5, or was edited since its registration, the
command is refused (canonical.Refused) and nothing is opened or written. With the rule registered: the checkpoint
loads, its training lists are read, and g5.read_care_pd runs with the model predictor (the guard again, the index
rows, the disjointness of the scored persons from the training lists, the logged read in reports/gates/eval_reads.csv
before the label table and the first record, then the lines). Writes <out> (default reports/gates/g5/<checkpoint
name>, suffix _model for the raw weights): g5.json, g5.md, per_walk.csv, trials.csv, left_out.csv, _provenance.json.

Owner only, after approving the rule: --register-rule --approved-on YYYY-MM-DD (common.register_rule; a session
never runs it).

Run (mdfm environment; evaluation contract --checkpoint, --weights, --device, --out, --memory-gb):
  <conda>/envs/mdfm/python.exe scripts/gate_g5.py --selftest [--checkpoint <run>/ckpt_B_<step>.pt]
  ... --selftest --quick --out <scratch folder>   (synthetic 10 persons of 4 s, AMASS dev 5 persons x 2 walks,
      200 bootstrap resamples)
  ... --read --checkpoint <ckpt>                  (refused while the rule is not registered)
"""
import argparse
import collections
import json
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (TREE, os.path.join(TREE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_g4_dev as G4D  # noqa: E402  (camera windows, inference and per-frame joints of the model)
from mdfm import paths  # noqa: E402
from mdfm.data import window_store as WS  # noqa: E402
from mdfm.data import windows as W  # noqa: E402
from mdfm.data.canonical import Refused  # noqa: E402
from mdfm.eval import infer as I  # noqa: E402
from mdfm.gates import common, g4, g5  # noqa: E402
from mdfm.model import mfm as M  # noqa: E402

WHAT_SELFTEST = "G5 development self-test, not a gate read"
WINDOW_SOURCE = "synthetic"                   # rule section 4: the G1c camera input
WINDOW_LAYOUT = "camera1"
WINDOW_CAMERA = G4D.CAMERA_FAMILY             # 'generic', the G1c camera family
WINDOW_TASK = "walk"
SETS = ("synthetic", "amass_dev")
ARMS = ("identity", "static", "model")
SYNTHETIC = {"persons": 20, "seconds": 6.0, "fps": 30.0, "seed": 0}
QUICK = {"synthetic_persons": 10, "synthetic_seconds": 4.0, "amass_persons": 5, "amass_walks": 2, "n_boot": 200}
LEFT_OUT = {"camera_shorter_than_a_window": f"camera render under {I.MIN_S:g} s (infer.MIN_S): no window"}
LEFT_OUT_COLUMNS = ("set", "arm", "person_id", "key", "view_set", "reason", "meaning", "frames", "fps")
SEED = g5.SEED


# ---------------------------------------------------------------- synthetic stand-ins
def walker(cadence=110.0, fps=30.0, seconds=5.0, step_amp=0.45, arm_l=0.35, arm_r=0.35, trunk_deg=5.0, knee=1.0,
           phase=0.0):
    """A parametric walker in place (T, 15, 3) m, y up, facing +z, left +x, pelvis at the origin: legs swing
    fore-aft at cadence / 120 strides per second (two steps per stride), the knee flexes from 0 to `knee` rad, each
    upper arm swings opposite to its leg with amplitude arm_l or arm_r rad, the trunk leans trunk_deg forward (the
    stand-in of tests/test_g5.py)."""
    t = np.arange(int(round(seconds * fps))) / fps
    ph = 2 * np.pi * cadence / 120.0 * t + phase
    th = np.radians(trunk_deg)
    T = len(t)
    up = np.stack([np.zeros(T), np.full(T, np.cos(th)), np.full(T, np.sin(th))], -1)
    X = np.zeros((T, 15, 3))
    X[:, 1] = 0.5 * up
    X[:, 2] = X[:, 1] + 0.2 * up
    for sgn, off, B, (sh, el, wr, hip, kn, an) in ((1.0, 0.0, arm_l, (3, 5, 7, 9, 11, 13)),
                                                     (-1.0, np.pi, arm_r, (4, 6, 8, 10, 12, 14))):
        X[:, sh] = X[:, 1] + np.array([sgn * 0.18, -0.05, 0.0])
        X[:, hip] = np.array([sgn * 0.1, -0.05, 0.0])
        a = step_amp * np.sin(ph + off)
        k = knee * (1 + np.sin(ph + off + np.pi / 2)) / 2
        X[:, kn] = X[:, hip] + 0.45 * np.stack([np.zeros(T), -np.cos(a), np.sin(a)], -1)
        X[:, an] = X[:, kn] + 0.42 * np.stack([np.zeros(T), -np.cos(a - k), np.sin(a - k)], -1)
        b = B * np.sin(ph + off + np.pi)
        X[:, el] = X[:, sh] + 0.3 * np.stack([np.zeros(T), -np.cos(b), np.sin(b)], -1)
        X[:, wr] = X[:, el] + 0.25 * np.stack([np.zeros(T), -np.cos(b + 0.3), np.sin(b + 0.3)], -1)
    return X


def as_walk(X, person, key, cohort="synthetic", label=float("nan"), state="", fps=30.0):
    """g5.Walk of a body X (T, 15, 3) m, normalized as a care_pd record is (kernels/pose.normalize_3d)."""
    coords, scale, valid = g5.pose_kernel.normalize_3d(X)
    return g5.Walk(person=person, key=key, fps=float(fps), coords=coords, valid=valid, scale=float(scale),
                   cohort=cohort, label=float(label), state=state)


def synthetic_walks(n_persons=SYNTHETIC["persons"], seconds=SYNTHETIC["seconds"], fps=SYNTHETIC["fps"],
                    seed=SYNTHETIC["seed"]):
    """Synthetic walks (module docstring, Self-test): person p has severity level p mod 4 (the UPDRS_GAIT stand-in),
    two OFF and two ON walks; persons 'synthetic:walker:Pnn', keys 'Pnn_<state>_walk_<k>', cohort 'synthetic'."""
    rng = np.random.default_rng(seed)
    out = []
    for p in range(int(n_persons)):
        level = p % 4
        person = f"synthetic:walker:P{p + 1:02d}"
        for state in ("OFF", "ON"):
            for w in (1, 2):
                on = state == "ON"
                X = walker(cadence=125 - 12 * level + rng.normal(0, 2), step_amp=0.55 - 0.08 * level + 0.08 * on,
                           arm_l=0.45 - 0.08 * level + 0.08 * on + rng.normal(0, 0.01), arm_r=0.45 - 0.1 * level,
                           trunk_deg=3 + 4 * level + rng.normal(0, 0.5), knee=1.1 - 0.12 * level,
                           phase=rng.uniform(0, 2 * np.pi), seconds=seconds, fps=fps)
                out.append(as_walk(X, person, f"P{p + 1:02d}_{state.lower()}_walk_{w}", label=level, state=state,
                                   fps=fps))
    return out


REST = walker(seconds=1.0)[0]


def static_predictor(walk, renders):
    """A body that carries nothing about the walk: one fixed standing pose, the same for every walk, held still."""
    return np.repeat(REST[None], len(walk.coords), axis=0)


# ---------------------------------------------------------------- the model as the G5 predictor
class G5Predictor:
    """The model as the predictor of mdfm.gates.g5 (module docstring, Model predictor): __call__(walk, renders) ->
    joints (T, 15, 3) m at the walk's frames, or None when the render gives no window. log: one dict per inference
    call (the rows of eval_g4_dev.G4Predictor.infer plus view, key, frames, uncovered_frames, detected_share);
    left_out: one dict per walk and view without a window; echo: a callable given one line per call (None:
    silent)."""

    def __init__(self, model, refine_steps=I.STEPS_EVAL, points=I.POINTS_EVAL, win_s=I.WIN_S, hop_s=I.HOP_S,
                 min_s=I.MIN_S, batch_windows=I.BATCH_WINDOWS, seed=SEED, pool=True, echo=None):
        self.core = G4D.G4Predictor(model, refine_steps=refine_steps, points=points, win_s=win_s, hop_s=hop_s,
                                    min_s=min_s, batch_windows=batch_windows, seed=seed, pool=pool, echo=echo)
        self.echo = echo
        self.left_out = []
        self.cov = {"sex": W.SEX_UNKNOWN, "age_band": W.AGE_UNKNOWN,
                    "task": int(W.task_vocabulary()["ids"][WINDOW_TASK])}

    @property
    def model(self):
        return self.core.model

    @property
    def log(self):
        return self.core.log

    @staticmethod
    def stream(render):
        """(band copy pose2d windows.Stream, times float64 (Q,) on the walk's clock) of one render (module
        docstring)."""
        vals = np.asarray(render["coords"], dtype=np.float32)
        valid = np.asarray(render["valid"], dtype=bool)
        tc = np.asarray(render["t"], dtype=np.float64)
        if vals.shape != (len(tc), 15, 3) or valid.shape != (len(tc), 15):
            raise ValueError(f"render coords {vals.shape}, valid {valid.shape} for {len(tc)} times")
        cm = np.zeros((15, 3), dtype=bool)
        cm[:, :2] = True
        meta = {"dim": 2, "scale": float(render["scale"]), "unit_to_si": None, "camera": WINDOW_CAMERA,
                "layout": WINDOW_LAYOUT, "source": WINDOW_SOURCE}
        st = W.Stream("pose2d", "input", float(render["fps"]), tc - tc[0] if len(tc) else tc, vals, cm, valid, meta)
        b, info = WS.band_copy(st, return_info=True)
        return b, tc[info.idx]

    def windows(self, walk, render):
        """(windows, spans, camera times) of one walk and render: the band copy cut into infer.spans."""
        stream, tc = self.stream(render)
        wins, sp = G4D.cut_windows(stream, tc, WINDOW_SOURCE, walk.person, walk.key, self.cov, self.core.win_s,
                                   self.core.hop_s, self.core.min_s)
        return wins, sp, tc

    def __call__(self, walk, renders):
        if renders is None or len(renders) != 1:
            n = 0 if renders is None else len(renders)
            raise ValueError(f"the model predictor reads one view per call, got {n} renders (a window holds one "
                             "pose2d stream)")
        render = renders[0]
        if render is None:
            raise ValueError(f"{walk.person} {walk.key}: the model predictor needs the render (walk_table render=True)")
        view = str(render.get("view", ""))
        wins, _, tc = self.windows(walk, render)
        if not wins:
            reason = "camera_shorter_than_a_window"
            row = {"person_id": walk.person, "key": walk.key, "view": view, "reason": reason,
                   "meaning": LEFT_OUT[reason], "samples": int(len(tc)), "fps": float(render["fps"]),
                   "min_samples": max(1, int(round(self.core.min_s * float(render["fps"]))))}
            self.left_out.append(row)
            if self.echo:
                self.echo(f"  g5_{view} {walk.person}/{walk.key}: left out, {row['reason']} ({row['samples']} samples)")
            return None
        bodies = self.core.infer(wins, f"{walk.person}/{walk.key}", f"g5_{view}")
        res, win_of = self.core.frames(wins, bodies, walk.t, ("joints15",))
        joints = np.asarray(res["joints15"], dtype=np.float64)
        self.core.log[-1].update({"view": view, "key": walk.key, "frames": int(len(win_of)),
                                  "uncovered_frames": int((win_of < 0).sum()),
                                  "detected_share": float(render["info"]["detected_share"])})
        return joints


# ---------------------------------------------------------------- checks
def check_dev_persons(persons, lists=None):
    """Refused when an AMASS dev person is a lockbox person (names only) or, with lists ({list name: ids}), a
    training person of the checkpoint; otherwise {lockbox: 0, training: 0 or None}."""
    persons = sorted({str(p) for p in persons})
    bad = sorted(set(persons) & set(WS.lockbox_persons()))
    if bad:
        raise Refused(f"lockbox persons among the AMASS dev walks: {bad[:3]}; the self-test never reads the lockbox")
    out = {"lockbox": 0, "training": None}
    if lists is not None:
        out["training"] = g4.assert_disjoint({"amass_dev": persons}, lists)["amass_dev"]
    return out


def checkpoint_lists(checkpoint, loaded):
    """{list name: ids}: persons/*.txt beside the checkpoint (eval_g4_dev.person_lists) and the lists it stores."""
    lists = G4D.person_lists(checkpoint)
    stored = {f"checkpoint:{s}": frozenset(v) for s, v in loaded.persons.items()}
    return {**lists, **stored}


def rule_info():
    """{path, sha256, status, registered} of the G5 rule file (read only)."""
    with open(g5.RULE_PATH, "rb") as f:
        raw = f.read()
    sha = common.sha256_bytes(raw)
    return {"path": common.rel(g5.RULE_PATH), "sha256": sha, "status": common.rule_status(raw.decode("utf-8"))[0],
            "registered": common.registered(g5.GATE, sha, common.RULE_HASHES)}


def code_files():
    """The harness's code files, the inference path and this script (the code hash of the outputs)."""
    return [*g5.harness_code_files(), os.path.abspath(I.__file__), os.path.abspath(M.__file__),
            os.path.abspath(G4D.__file__), os.path.abspath(__file__)]


# ---------------------------------------------------------------- the self-test
def selftest_walks(sets, a, check_dev=None):
    """{set: [Walk]} of the self-test sets (module docstring). check_dev: given the AMASS dev person ids after the
    take index is read and before any take loads (g5.amass_dev_walks check_persons); it refuses by raising."""
    out = {}
    for s in sets:
        if s == "synthetic":
            out[s] = synthetic_walks(a.synthetic_persons, a.synthetic_seconds)
        elif s == "amass_dev":
            out[s] = g5.amass_dev_walks(n_persons=a.n_persons, max_walks_per_person=a.max_walks_per_person,
                                        check_persons=check_dev)
        else:
            raise ValueError(f"unknown set {s!r}; known: {list(SETS)}")
    return out


def run_arm(set_name, walks, arm, predictor, n_boot, hashes):
    """(results, table, summary, stats, left_out) of one set and arm: g5.selftest (the guard asked first) for the
    AMASS dev walks, g5.evaluate with the walks' own labels for the synthetic ones."""
    stats, left = collections.Counter(), []
    if set_name == "amass_dev":
        res, table, summ = g5.selftest(walks, predictor, n_boot=n_boot, stats=stats, left_out=left, **hashes)
    else:
        res, table, summ = g5.evaluate(walks, predictor, n_boot=n_boot, stats=stats, left_out=left, **hashes)
    return res, table, summ, stats, left


def line_rows(set_name, arm, results):
    rows = []
    for r in results:
        x = r.extra
        rows.append({"set": set_name, "arm": arm, "line": r.line, "unit": r.unit, "point": r.point,
                     "interval": r.interval, "drop_refit": x.get("drop"), "drop_transfer": x.get("drop_transfer"),
                     "source": x.get("source"), "refit": x.get("reestimated"), "transfer": x.get("transfer"),
                     "trivial_drop": r.trivial_baseline,
                     "kill_reachable_by_trivial": x.get("kill_reachable_by_trivial"),
                     "informative": x.get("informative"), "pass_line": r.pass_line, "kill_line": r.kill_line,
                     "verdict": r.verdict, "persons": r.n_persons, "walks": x.get("n_walks"),
                     "left_out": x.get("left_out"), "reason": x.get("reason")})
    return rows


def default_out(loaded, kind, tree=TREE):
    """reports/gates/g5/<kind>_<checkpoint name> (selftest) or <checkpoint name> (read), suffix _model for the raw
    weights; reports/gates/g5/selftest_harness for a self-test without a checkpoint."""
    base = os.path.join(tree, "reports", "gates", "g5")
    if loaded is None:
        return os.path.join(base, "selftest_harness")
    name = loaded.name + ("_model" if loaded.weights == "model" else "")
    return os.path.join(base, f"selftest_{name}" if kind == "selftest" else name)


def load(a):
    """(Loaded or None, device, cap): the checkpoint for inference, the CUDA allocator capped at --memory-gb."""
    if not a.checkpoint:
        return None, None, None
    from mdfm.train import loop as LP
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    return I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev), dev, cap


def model_predictor(loaded, a, echo=None):
    """The G5Predictor of a loaded checkpoint with the command's inference options."""
    return G5Predictor(loaded.model, refine_steps=a.refine_steps, points=a.points, batch_windows=a.batch_windows,
                       seed=a.seed, pool=not a.no_pool, echo=echo)


def memory(dev, cap):
    if dev is None or dev.type != "cuda":
        return {}
    return {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
            "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9, "cap_gb": cap}


def run_selftest(a, log=print):
    t_start = time.time()
    from mdfm.train import provenance as PV
    arms = list(a.arms) if a.arms else ["identity", "static"] + (["model"] if a.checkpoint else [])
    if "model" in arms and not a.checkpoint:
        raise SystemExit("the model arm needs --checkpoint")
    if a.checkpoint and "model" not in arms:
        raise SystemExit("--checkpoint without the model arm: add model to --arms or drop --checkpoint (the "
                         "checkpoint would be ignored and the run written as the harness self-test)")
    loaded, dev, cap = load(a) if "model" in arms else (None, None, None)
    out = a.out or default_out(loaded, "selftest")
    if a.quick and (a.out is None or os.path.abspath(out) == os.path.abspath(default_out(loaded, "selftest"))):
        raise SystemExit("--quick writes to a scratch folder: pass --out")
    paths.assert_not_moveatlas(out)
    rinfo = rule_info()
    lists = checkpoint_lists(a.checkpoint, loaded) if loaded is not None else None
    checks, checked = {}, set()

    def check_dev(persons):
        """The lockbox and training-list check of the AMASS dev persons, before any of their takes loads."""
        checks["amass_dev"] = check_dev_persons(persons, lists)
        checked.update(str(p) for p in persons)

    walks = selftest_walks(a.sets, a, check_dev)
    if "amass_dev" in walks and ("amass_dev" not in checks
                                 or {w.person for w in walks["amass_dev"]} - checked):
        raise Refused("AMASS dev walks of persons not checked against the lockbox and the training lists before "
                      "their takes loaded")
    ck_sha = loaded.sha256 if loaded is not None else ""
    tp_sha = I.train_persons_sha256(loaded) if loaded is not None else ""
    pred = model_predictor(loaded, a, echo=(lambda s: log(s)) if a.verbose else None) if loaded is not None else None
    if loaded is not None:
        log(f"checkpoint {loaded.name} ({loaded.sha256[:12]}), {loaded.weights}, phase {loaded.phase} step "
            f"{loaded.step}")
    predictors = {"identity": g5.identity_predictor, "static": static_predictor, "model": pred}
    lines, pa, sets_doc, left_rows, runs = [], [], {}, [], {}
    for set_name, ws in walks.items():
        persons = sorted({w.person for w in ws})
        sets_doc[set_name] = {"walks": len(ws), "persons": len(persons), "persons_sha256": common.sha256_json(persons),
                              "walks_sha256": common.sha256_json(sorted([w.person, w.key] for w in ws)),
                              "fps": sorted({float(w.fps) for w in ws}),
                              "seconds_total": round(float(sum(len(w.coords) / w.fps for w in ws)), 2)}
        for arm in arms:
            t0 = time.time()
            is_model = arm == "model" and pred is not None
            n_calls, n_left = (len(pred.log), len(pred.left_out)) if is_model else (0, 0)
            hashes = {"rule_sha256": rinfo["sha256"], "checkpoint_sha256": ck_sha if arm == "model" else "",
                      "train_persons_sha256": tp_sha if arm == "model" else ""}
            res, table, summ, stats, left = run_arm(set_name, ws, arm, predictors[arm], a.n_boot, hashes)
            pl = list(pred.left_out[n_left:]) if is_model else []
            info = {"set": set_name, "arm": arm, "stats": dict(stats), "left_out": left, "predictor_left_out": pl,
                    "gate_verdict": g5.gate_verdict(res), "seconds": round(time.time() - t0, 1),
                    "inference_calls": (len(pred.log) - n_calls) if is_model else 0}
            g5.write_outputs(os.path.join(out, f"{set_name}_{arm}"), res, summ, table, info,
                             title=f"G5 self-test: {set_name} walks, {arm} arm (dev estimate, not a gate read)")
            lines += line_rows(set_name, arm, res)
            for vs, d in summ["pa_mpjpe"].items():
                pa.append({"set": set_name, "arm": arm, "view_set": vs, "point": d["point"], "interval": d["interval"],
                           "persons": d["n_persons"], "walks": d["n_walks"]})
            for x in left:
                left_rows.append({"set": set_name, "arm": arm, **{k: x.get(k) for k in LEFT_OUT_COLUMNS[2:]}})
            sets_doc[set_name][f"labels_{arm}"] = summ["sets"]
            runs[f"{set_name}_{arm}"] = {"gate_verdict": info["gate_verdict"], "stats": info["stats"],
                                         "left_out": summ.get("left_out"), "seconds": info["seconds"],
                                         "inference_calls": info["inference_calls"], "predictor_left_out": len(pl)}
            log(f"{set_name} {arm}: {len(ws)} walks, verdict {info['gate_verdict']}, {info['seconds']:.0f} s")
    import pandas as pd
    trials = pd.DataFrame(pred.log) if pred is not None else pd.DataFrame()
    doc = {"run": "g5_selftest", "gate": g5.GATE, "what": WHAT_SELFTEST, "quick": bool(a.quick),
           "checkpoint": loaded.describe() if loaded is not None else None,
           "rule": rinfo, "arms": arms, "sets": sets_doc, "checks": checks, "runs": runs, "lines": lines,
           "pa_mpjpe": pa, "left_out": {"rows": len(left_rows),
                                        "predictor": list(pred.left_out) if pred is not None else []},
           "labels": {"synthetic": "built in: level p mod 4 per person (UPDRS_GAIT stand-in), two OFF and two ON walks "
                                   "per person",
                      "amass_dev": "g5.assign_selftest_labels: tertiles of the source cadence; ON where the source "
                                   "step amplitude is at least the person's median"},
           "inference": None if pred is None else {
               "refine_steps": a.refine_steps, "points_per_stream": a.points, "window_s": pred.core.win_s,
               "hop_s": pred.core.hop_s, "min_s": pred.core.min_s, "batch_windows": a.batch_windows, "seed": a.seed,
               "shape_pooling": "none" if a.no_pool else "per walk and view", "weights": a.weights,
               "device": str(dev), "amp_bf16": dev.type == "cuda", "memory": memory(dev, cap),
               "window": {"source": WINDOW_SOURCE, "layout": WINDOW_LAYOUT, "camera": WINDOW_CAMERA,
                          "covariates": pred.cov},
               "calls": int(len(trials)), "windows": int(trials["windows"].sum()) if len(trials) else 0,
               "seconds_in_model": float(trials["seconds"].sum()) if len(trials) else 0.0,
               "uncovered_frames": int(trials["uncovered_frames"].sum()) if len(trials) else 0},
           "config": {"n_boot": a.n_boot, "seed": g5.SEED, "folds": g5.N_FOLDS, "reading": g5.PROBE_READING,
                      "auroc_unit": g5.AUROC_UNIT, "view_sets": [list(v) for v in g5.VIEW_SETS],
                      "synthetic_persons": a.synthetic_persons, "synthetic_seconds": a.synthetic_seconds,
                      "amass_persons": a.n_persons, "amass_walks_per_person": a.max_walks_per_person},
           "timing_s": round(time.time() - t_start, 1), "moveatlas_head": PV.moveatlas_head()}
    clean = G4D.clean(doc)
    text = selftest_report(clean)
    common.write(clean, out, "g5_selftest", text)
    common.write_csv(trials, os.path.join(out, "trials.csv"))
    common.write_csv(pd.DataFrame(left_rows, columns=list(LEFT_OUT_COLUMNS)), os.path.join(out, "left_out.csv"))
    files = code_files()
    prov = {"script": "scripts/gate_g5.py --selftest", "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_head": doc["moveatlas_head"], "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "checkpoint": clean["checkpoint"]["path"] if loaded is not None else None,
            "checkpoint_sha256": ck_sha or None, "quick": bool(a.quick), "code_sha256": common.code_sha256(files),
            "code_files": [common.rel(f) for f in files],
            "inputs": ["synthetic parametric walkers (scripts/gate_g5.py synthetic_walks), no data read"]
            + (["the AMASS dev body store (mdfm.data.body_store, split dev, task walk; tier RELEASE)"]
               if "amass_dev" in walks else [])}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    log(text)
    log(f"written {common.rel(out)}")
    return out, clean


def fmt(x, nd=4):
    return G4D.fmt(x, nd)


def fmt_iv(iv, nd=4):
    return G4D.fmt_iv(iv, nd)


def selftest_report(doc):
    """g5_selftest.md from the cleaned result dict."""
    rule, ck = doc["rule"], doc["checkpoint"]
    L = [f"# {WHAT_SELFTEST}", "",
         "Written by scripts/gate_g5.py --selftest (harness mdfm/gates/g5.py, rule docs/rules/G5_RULE.md sections 7 "
         "and 8 item 10); every number below is read from g5_selftest.json and the run folders <set>_<arm>/ of this "
         "folder. Non-EVAL data only: synthetic parametric walkers and AMASS dev walks (tier RELEASE); no care_pd "
         "index row, label or record was read and mdfm.gates.g5.read_care_pd was not called. The labels are "
         "known-signal labels of the self-test and decide nothing." + (" QUICK RUN (subsets)." if doc["quick"] else ""),
         ""]
    if ck:
        L.append(f"Checkpoint: {ck['path']} (SHA-256 {ck['sha256']}), {ck['weights']} weights, phase {ck['phase']} "
                 f"step {ck['step']:,}; hashes moved since the save: {', '.join(ck['drift']) or 'none'}.")
    else:
        L.append("No checkpoint: the harness arms only (identity and static).")
    L += ["", f"Rule {rule['path']}: status '{rule['status']}', SHA-256 {rule['sha256']} (registered: "
              f"{rule['registered']}). MoveAtlas HEAD {doc['moveatlas_head']}. Arms: {', '.join(doc['arms'])}.", "",
          "## Lines (dev estimates, not gate reads)", "",
          "Deciding reading 'both': the drop is the larger of the refit and transfer drops (rule section 1). QWK "
          "lines in QWK, ON versus OFF lines in AUROC points; the trivial drop is what a body that carries nothing "
          "gives.", "",
          "| set | arm | line | drop | 95% interval | refit drop | transfer drop | source | trivial drop | "
          "trivial kills | verdict | persons | walks |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["lines"]:
        kr = r["kill_reachable_by_trivial"]
        L.append(f"| {r['set']} | {r['arm']} | {r['line']} | {fmt(r['point'])} | {fmt_iv(r['interval'])} | "
                 f"{fmt(r['drop_refit'])} | {fmt(r['drop_transfer'])} | {fmt(r['source'])} | "
                 f"{fmt(r['trivial_drop'])} | "
                 f"{'n/a' if kr is None else ('yes' if kr else 'no')} | {r['verdict']} | {r['persons']} | "
                 f"{r['walks'] if r['walks'] is not None else 'n/a'} |")
    model_rows = [r for r in doc["lines"] if r["arm"] == "model"]
    if model_rows:
        L += ["", "## The model arm, line by line (rule section 7, attainability)", ""]
        for s in sorted({r["set"] for r in model_rows}):
            rows = [r for r in model_rows if r["set"] == s]
            counts = collections.Counter(r["verdict"] for r in rows)
            L.append(f"- {s}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" of {len(rows)} "
                     "lines; " + "; ".join(f"{r['line']} drop {fmt(r['point'])} {fmt_iv(r['interval'])}"
                                           for r in rows) + ".")
    L += ["", "## PA-MPJPE against the source bodies (diagnostic, no kill)", "",
          "| set | arm | view set | mm | 95% interval | persons | walks |", "|---|---|---|---|---|---|---|"]
    for p in doc["pa_mpjpe"]:
        L.append(f"| {p['set']} | {p['arm']} | {p['view_set']} | {fmt(p['point'], 1)} | {fmt_iv(p['interval'], 1)} | "
                 f"{p['persons']} | {p['walks']} |")
    L += ["", "## Sets", ""]
    for s, d in doc["sets"].items():
        lab = d.get(f"labels_{doc['arms'][0]}", {})
        on_off = dict(lab.get("on_off") or {})
        single = on_off.pop("persons_with_one_state", None) or []
        L.append(f"- {s}: {d['walks']} walks of {d['persons']} persons (persons SHA-256 {d['persons_sha256']}), "
                 f"{d['seconds_total']} s in all, rates {d['fps']} Hz. Labels: {doc['labels'].get(s, '')}. "
                 f"UPDRS_GAIT stand-in set {lab.get('updrs_gait')}; ON versus OFF set {on_off}, persons with one "
                 f"state (listed, not scored): {len(single)}.")
    for s, c in doc["checks"].items():
        L.append(f"- {s} persons: lockbox persons {c['lockbox']}; training persons of the checkpoint "
                 f"{'not checked (no checkpoint)' if c['training'] is None else c['training']}.")
    L += ["", "## Walks left out", "",
          "A walk without a re-estimated body keeps its place with imputed descriptors (rule section 4); counted "
          "per set, arm and body set:", ""]
    for k, r in doc["runs"].items():
        L.append(f"- {k}: {r['left_out']}; predictor left-outs {r['predictor_left_out']}; verdict "
                 f"{r['gate_verdict']}; {r['seconds']} s.")
    inf = doc["inference"]
    if inf:
        mem = inf["memory"] or {}
        L += ["", "## Inference (model arm)", "",
              f"Window input: source {inf['window']['source']}, layout {inf['window']['layout']}, camera "
              f"{inf['window']['camera']}, covariates {inf['window']['covariates']}; {inf['window_s']:g} s windows "
              f"every {inf['hop_s']:g} s (none under {inf['min_s']:g} s), {inf['refine_steps']} refinement steps of "
              f"{inf['points_per_stream']} points per stream, shape pooled {inf['shape_pooling']}, device "
              f"{inf['device']}"
              + (f", peak allocated {mem['max_allocated_gb']:.2f} GB, reserved {mem['max_reserved_gb']:.2f} GB under "
                 f"a {mem['cap_gb']} GB cap" if mem else "")
              + f". Calls {inf['calls']}, windows {inf['windows']}, {inf['seconds_in_model']:.0f} s in the model; "
              f"walk frames no window covers: {inf['uncovered_frames']} (trials.csv)."]
    L += ["", f"Time: {doc['timing_s']} s.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- the registered read
def run_read(a, log=print):
    """The care_pd read (module docstring, Registered read). Refused before anything loads while the rule is not
    registered."""
    if not a.checkpoint:
        raise SystemExit("--read needs --checkpoint")
    g = common.guard_eval_read(g5.GATE, g5.SOURCE, g5.RULE_PATH, registry_path=common.RULE_HASHES,
                               tiers_path=common.TIERS)
    t_start = time.time()
    from mdfm.train import provenance as PV
    loaded, dev, cap = load(a)
    out = a.out or default_out(loaded, "read")
    paths.assert_not_moveatlas(out)
    lists = checkpoint_lists(a.checkpoint, loaded)
    pred = model_predictor(loaded, a, echo=(lambda s: log(s)) if a.verbose else None)
    files = code_files()
    results, table, summ, info = g5.read_care_pd(
        pred, checkpoint_sha256=loaded.sha256, train_persons_sha256=I.train_persons_sha256(loaded),
        train_persons=lists, code_files=files[len(g5.harness_code_files()):], n_boot=a.n_boot,
        note=f"scripts/gate_g5.py --read, checkpoint {loaded.name}", rule_path=g5.RULE_PATH,
        registry_path=common.RULE_HASHES, tiers_path=common.TIERS, log_path=common.EVAL_LOG)
    import pandas as pd
    trials = pd.DataFrame(pred.log)
    info = dict(info, rule={"path": g["rule_path"], "sha256": g["rule_sha256"]}, checkpoint=loaded.describe(),
                predictor_left_out=list(pred.left_out), moveatlas_head=PV.moveatlas_head(),
                inference={"refine_steps": a.refine_steps, "points_per_stream": a.points, "seed": a.seed,
                           "shape_pooling": "none" if a.no_pool else "per walk and view", "weights": a.weights,
                           "device": str(dev), "memory": memory(dev, cap), "calls": int(len(trials)),
                           "uncovered_frames": int(trials["uncovered_frames"].sum()) if len(trials) else 0},
                timing_s=round(time.time() - t_start, 1))
    g5.write_outputs(out, results, summ, table, G4D.clean(info),
                     title=f"G5 pathology preservation, checkpoint {loaded.name}")
    common.write_csv(trials, os.path.join(out, "trials.csv"))
    left = [{"set": "care_pd", "arm": "model", **{k: x.get(k) for k in LEFT_OUT_COLUMNS[2:]}} for x in info["left_out"]]
    common.write_csv(pd.DataFrame(left, columns=list(LEFT_OUT_COLUMNS)), os.path.join(out, "left_out.csv"))
    prov = {"script": "scripts/gate_g5.py --read", "moveatlas_root": paths.MOVEATLAS,
            "moveatlas_head": info["moveatlas_head"], "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "checkpoint_sha256": loaded.sha256,
            "rule_sha256": g["rule_sha256"], "eval_read": info["eval_read"], "code_sha256": common.code_sha256(files),
            "code_files": [common.rel(f) for f in files],
            "inputs": ["MoveAtlas canonical pose records of care_pd (EVAL, canonical.load allow_eval in "
                       "mdfm.gates.g5.care_pd_walks only)", f"the label table {g5.LABELS_CSV}"]}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(G4D.clean(prov), indent=1) + "\n")
    log(f"G5 read: gate verdict {info['gate_verdict']}; written {common.rel(out)}")
    return out, results


# ---------------------------------------------------------------- command line
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cmd = p.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--selftest", action="store_true", help="the development self-test (non-EVAL data only)")
    cmd.add_argument("--read", action="store_true", help="the care_pd read; refused while the rule is not registered")
    cmd.add_argument("--register-rule", action="store_true", help="owner only: register the approved G5 rule")
    p.add_argument("--approved-on", default=None)
    p.add_argument("--checkpoint", default=None, help="a checkpoint of mdfm/train/checkpoint.py")
    p.add_argument("--weights", default="ema", choices=I.WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=4.0, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--out", default=None)
    p.add_argument("--refine-steps", type=int, default=I.STEPS_EVAL)
    p.add_argument("--points", type=int, default=I.POINTS_EVAL, help="kernel points per stream-window and step")
    p.add_argument("--batch-windows", type=int, default=I.BATCH_WINDOWS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a walk")
    p.add_argument("--n-boot", type=int, default=None, help=f"bootstrap resamples (default {g5.BOOT_N}, quick "
                                                            f"{QUICK['n_boot']})")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    p.add_argument("--arms", nargs="+", default=None, choices=ARMS,
                   help="default: identity and static, plus model with --checkpoint")
    p.add_argument("--synthetic-persons", type=int, default=None)
    p.add_argument("--synthetic-seconds", type=float, default=None)
    p.add_argument("--n-persons", type=int, default=None, help="AMASS dev persons (default all)")
    p.add_argument("--max-walks-per-person", type=int, default=None, help="AMASS dev walks per person (default all)")
    p.add_argument("--quick", action="store_true", help="subsets (QUICK), written to the scratch folder of --out")
    p.add_argument("--verbose", action="store_true", help="one line per inference call")
    a = p.parse_args(argv)
    q = a.quick
    if a.n_boot is None:
        a.n_boot = QUICK["n_boot"] if q else g5.BOOT_N
    if a.synthetic_persons is None:
        a.synthetic_persons = QUICK["synthetic_persons"] if q else SYNTHETIC["persons"]
    if a.synthetic_seconds is None:
        a.synthetic_seconds = QUICK["synthetic_seconds"] if q else SYNTHETIC["seconds"]
    if a.n_persons is None and q:
        a.n_persons = QUICK["amass_persons"]
    if a.max_walks_per_person is None and q:
        a.max_walks_per_person = QUICK["amass_walks"]
    return a


def main(argv=None):
    a = parse(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.register_rule:
        if not a.approved_on:
            raise SystemExit("--register-rule needs --approved-on YYYY-MM-DD")
        row = common.register_rule(g5.GATE, g5.RULE_PATH, a.approved_on, registry_path=common.RULE_HASHES,
                                   log_path=common.READ_LOG)
        log(json.dumps(row, indent=1))
        return row
    if a.read:
        return run_read(a, log)
    return run_selftest(a, log)


if __name__ == "__main__":
    try:
        main()
    except Refused as e:
        sys.exit(f"REFUSED: {e}")
