"""JP0-READ: the reading of J0 (frozen-token probes) in one unit, the pass mean, as section 3a of
docs/proposals/JEPA_EXPLORATION_DRAFT.md fixes it. Every output is labelled LABEL.

Order of the run, each step before the next reads anything:
1. Hash of section 3a (check_addendum): the SHA-256 of the UTF-8 bytes of the lines strictly between the line starting
   '## 3a. ' and the next heading line (one to six '#' then a space), each with its LF ending, must equal the first word
   of reports/bench/jepa_jp0_addendum.sha256 and ADDENDUM_SHA256 here. Otherwise the run stops with EXIT_HASH (2)
   before any other file is read.
2. F1 (read_f1): reports/bench/g3_gain_check/results.json of scripts/g3_gain_check.py, with its rule hash checked
   against reports/bench/g3_gain_check/rule.txt. Missing: the run stops with EXIT_F1 (3) before any J0 output is
   opened and before any reading is written. From it: the GAIN reading (decided by F1 alone) and the stride and speed
   gains of each end, never refitted here.
3. J0 outputs of each checkpoint of CHECKPOINTS (reports/bench/frozen_tokens_<name>/results.json, per_pass.csv and the
   feature file results.json names, whose SHA-256 must equal the one J0 recorded). The probes stem, measurement,
   latent, mu_z and subject are refitted from the feature file with mdfm.eval.probes.fit_probe, as
   scripts/probe_frozen_tokens.py fits them (54 training-fold controls, penalty by person folds). Refit check
   (check_refit): the refitted per-stride latent stride line (G3 stride MAE, dev14 and dev12) must agree with J0's
   results.json within REFIT_TOL_CM, 0.01 cm, or the run stops with EXIT_REFIT (4).
4. Pass unit (section 3a, Unit): per walkway pass of the dev controls (probes.pass_units), the reference is the mean
   walkway stride; each row's pass stride: a probe, its prediction; decoded, the mean of the base-row decoded strides
   matched to the pass's walkway strides (reports/gates/g3/diag_<name>/events.csv: row base, kind stride, matched);
   recalibrated, the decoded pass stride DIVIDED by F1's stride gain of that checkpoint (F1's gain is the median of
   decoded over walkway); supervised, per_pass_dev of reports/bench/g3_supervised_baseline/results.json; population,
   the per-task population baseline of mdfm.gates.g3 from the training fold. Error: absolute difference, mean per
   person, mean over persons: S_pass(12) and S_pass(14). A pass enters the chain of a checkpoint when the stem, latent
   and mu_z rows, and the decoded row where the checkpoint has one, all hold a finite value for it; the passes left
   out are counted. Speed (V_pass) the same way, reported.
5. Margin rule (relation): X better than Y by the margin at a B end when G = S_pass(12) of Y minus that of X is at
   least m = max(0.10 Y1, 2 d) (Y1 the S_pass(12) of Y at v1's B end, d the larger of abs(X1 - X2) and abs(Y1 - Y2)
   between v1's and seed 2's B ends) and the lower bound of the paired person bootstrap of Y minus X
   (mdfm.gates.common.cluster_bootstrap, 2,000 resamples, seed 20260929, 95 percent) is above 0; holds at both ends:
   holds, fails at both: fails, otherwise split.
6. Readings: GAIN from F1; D (X latent, Y decoded) and R (X latent, Y recalibrated): CARRY, BOTTLENECK or SPLIT, R
   pending without F1's gain at both B ends; LM (latent better than mu_z holds); SL (latent better than stem fails).
   Routing rules 1 to 4 of section 3a (route). Reported, not read: the A end's chain, the 14, speed, J0's per-stride
   lines and section 3's reading in the per-stride unit, the supervised and population rows, the paired intervals
   between adjacent rows of the chain, and the JP2 go-line margin at ckpt_B_004000 when both J0 jobs there exist.

Writes <out> (default reports/bench/jp0_reading): results.json, summary.md, _provenance.json (MoveAtlas stamp of
scripts/moveatlas_stamp.py, the section 3a hash, the SHA-256 of every input). CPU only (CUDA_VISIBLE_DEVICES=-1); no
lockbox, EVAL or CODY record; no gain fitted.

Run (mdfm environment):
  CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/jp0_read.py
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # never the GPU

import numpy as np  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

LABEL = "dev diagnostic, not a G3 read; reading of J0 in the pass unit (section 3a)"
TASK = "JP0-READ"
ADDENDUM_SHA256 = "6083e31e7d765797acd5062b70b622e46d45fe09521b260b2e5d17c107567c94"
PROPOSAL_REL = "docs/proposals/JEPA_EXPLORATION_DRAFT.md"
RECORD_REL = "reports/bench/jepa_jp0_addendum.sha256"
F1_RESULTS_REL = "reports/bench/g3_gain_check/results.json"
F1_RULE_REL = "reports/bench/g3_gain_check/rule.txt"
SUPERVISED_REL = "reports/bench/g3_supervised_baseline/results.json"
OUT_REL = "reports/bench/jp0_reading"
HEADING = re.compile(r"#{1,6} ")

EXIT_HASH, EXIT_F1, EXIT_REFIT, EXIT_INPUT = 2, 3, 4, 5
REFIT_TOL_CM = 0.01
DECODED_TOL_CM = 1e-6               # S(12) and S(14) per stride from events.csv against diag.json (as F1)
SUPERVISED_REF_TOL_M = 1e-6
BOOT_N = 2000
SEED = 20260929
LEVEL = 0.95
MARGIN_FRAC = 0.10
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")
SETS = ("dev12", "dev14")
F1_READINGS = ("GAIN", "not GAIN", "not evaluated")

# (key, checkpoint name, role); the last two are read only when both J0 jobs at ckpt_B_004000 exist
CHECKPOINTS = (("v1_A", "phaseA_root_full_ckpt_A_010681", "v1 A end"),
               ("v1_B", "phaseA_root_full_ckpt_B_032044", "v1 B end"),
               ("s2_B", "pilot_v1_s2_ckpt_B_032044", "seed 2 B end"),
               ("v1_B4000", "phaseA_root_full_ckpt_B_004000", "v1 ckpt_B_004000"),
               ("s2_B4000", "pilot_v1_s2_ckpt_B_004000", "seed 2 ckpt_B_004000"))
REQUIRED = ("v1_A", "v1_B", "s2_B")
OPTIONAL_PAIR = ("v1_B4000", "s2_B4000")
B_ENDS = ("v1_B", "s2_B")
PROBES = ("stem", "measurement", "latent", "mu_z", "subject")     # scripts/probe_frozen_tokens.VARIANTS
CHAIN = ("stem", "latent", "mu_z", "decoded", "recalibrated")
NEEDED = ("stem", "latent", "mu_z")                                # plus decoded where the checkpoint has one
BESIDE = ("measurement", "subject", "supervised", "population")
ROWS = ("stem", "measurement", "latent", "mu_z", "subject", "decoded", "recalibrated", "supervised", "population")
ADJACENT = (("stem", "latent"), ("latent", "mu_z"), ("mu_z", "decoded"), ("decoded", "recalibrated"))
CARRY, BOTTLENECK, SPLIT, PENDING = "CARRY", "BOTTLENECK", "SPLIT", "pending"


class Stop(Exception):
    """A stop of the run with its exit code (main prints the message and returns the code)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = int(code)
        self.message = message


# ---------------------------------------------------------------- small helpers
def fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv or any(fin(v) is None for v in iv) else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def clean(o):
    """JSON-ready: non-finite floats as null, numpy scalars and arrays as Python values, tuple keys joined."""
    if isinstance(o, dict):
        return {(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(path, tree=None):
    try:
        r = os.path.relpath(os.path.abspath(path), tree or TREE)
    except ValueError:
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sibling(name):
    """The module of scripts/<name>.py of this tree (loaded once, under a private name)."""
    key = f"_jp0_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(TREE, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 1. the hash of section 3a
def section_3a(text):
    """The body of section 3a (lines strictly between its heading and the next heading, each with LF), or None."""
    lines = text.split("\n")
    starts = [k for k, line in enumerate(lines) if line.startswith("## 3a. ")]
    if len(starts) != 1:
        return None
    i = starts[0]
    j = next((k for k in range(i + 1, len(lines)) if HEADING.match(lines[k])), None)
    if j is None:
        return None
    return "".join(line + "\n" for line in lines[i + 1:j])


def addendum_sha(text):
    body = section_3a(text)
    return None if body is None else hashlib.sha256(body.encode("utf-8")).hexdigest()


def check_addendum(tree=None, expected=None):
    """The SHA-256 of section 3a of the proposal under `tree`, asserted equal to the first word of the record file and
    to `expected` (default ADDENDUM_SHA256); Stop(EXIT_HASH) otherwise. Reads nothing else."""
    tree = tree or TREE
    expected = ADDENDUM_SHA256 if expected is None else expected
    prop, rec = os.path.join(tree, PROPOSAL_REL), os.path.join(tree, RECORD_REL)
    for p in (prop, rec):
        if not os.path.exists(p):
            raise Stop(EXIT_HASH, f"refused: {rel(p, tree)} is missing, the hash of section 3a cannot be checked")
    with open(prop, encoding="utf-8", newline="") as f:
        sha = addendum_sha(f.read())
    with open(rec, encoding="utf-8", newline="") as f:
        words = f.read().split()
    stored = words[0] if words else ""
    if sha is None or not (sha == stored == expected):
        raise Stop(EXIT_HASH, f"refused: section 3a of {PROPOSAL_REL} hashes to {sha}, the record holds {stored or 'nothing'}"
                              f", this script expects {expected}: the reading text is not the one hashed before J0")
    return sha


# ---------------------------------------------------------------- 2. F1
def read_f1(tree=None):
    """F1's GAIN reading and gains from its results.json: {'path', 'sha256', 'reading', 'rule_sha256', 'gains': {checkpoint
    name: {'key', 'stride', 'speed', 'checkpoint_sha256'}}}. Stop(EXIT_F1) when the file is missing, Stop(EXIT_INPUT)
    when its rule hash is not that of rule.txt now or its reading is not one of F1_READINGS."""
    tree = tree or TREE
    path = os.path.join(tree, F1_RESULTS_REL)
    if not os.path.exists(path):
        raise Stop(EXIT_F1, f"stopped: {F1_RESULTS_REL} does not exist (F1's GPU decode or its check has not finished). "
                            "GAIN and reading R need it; no J0 output was opened and no reading was written. Rerun "
                            "this script once F1 has written its results.")
    doc = read_json(path)
    rule = os.path.join(tree, F1_RULE_REL)
    rule_sha = sha256_file(rule) if os.path.exists(rule) else None
    if (doc.get("rule") or {}).get("sha256") != rule_sha or rule_sha is None:
        raise Stop(EXIT_INPUT, f"{F1_RESULTS_REL} names rule {(doc.get('rule') or {}).get('sha256')}, {F1_RULE_REL} "
                               f"hashes to {rule_sha}")
    reading = (doc.get("reading") or {}).get("reading")
    if reading not in F1_READINGS:
        raise Stop(EXIT_INPUT, f"{F1_RESULTS_REL}: reading {reading!r} is not one of {F1_READINGS}")
    gains = {}
    for key, e in (doc.get("ends") or {}).items():
        g = e.get("gain") or {}
        gains[e.get("name")] = {"key": key, "checkpoint_sha256": (e.get("checkpoint") or {}).get("sha256"),
                                "stride": fin((g.get("stride") or {}).get("gain")),
                                "speed": fin((g.get("speed") or {}).get("gain")),
                                "fitted_on": g.get("fitted_on")}
    return {"path": rel(path, tree), "sha256": sha256_file(path), "reading": reading, "rule_sha256": rule_sha,
            "reading_detail": doc.get("reading"), "gains": gains}


def usable_gain(g):
    v = fin(g)
    return v is not None and v > 0


def recalibrate(decoded, gain):
    """Decoded values divided by F1's gain (the median of decoded over walkway): a gain below 1 moves a decoded value
    that is too small up toward the walkway. ValueError for a gain that is not a finite positive number."""
    if not usable_gain(gain):
        raise ValueError(f"gain {gain} is not a finite positive number")
    return np.asarray(decoded, dtype=np.float64) / float(gain)


# ---------------------------------------------------------------- 3. refit check
def check_refit(j0_points, refit_points, tol=REFIT_TOL_CM, what="latent stride MAE (cm)"):
    """{set: {'j0', 'refit', 'difference'}} of the per-stride lines; Stop(EXIT_REFIT) when a set is missing on either
    side, a value is not finite, or a difference exceeds tol."""
    out, bad = {}, []
    for s in sorted(set(j0_points) | set(refit_points)):
        a, b = fin(j0_points.get(s)), fin(refit_points.get(s))
        d = None if a is None or b is None else b - a
        out[s] = {"j0": a, "refit": b, "difference": d}
        if d is None or abs(d) > tol:
            bad.append(s)
    if bad or not out:
        raise Stop(EXIT_REFIT, f"refit check failed: the refitted {what} differs from J0's results.json beyond {tol} "
                               f"on {bad or 'every set'} ({out}); nothing is read")
    return out


# ---------------------------------------------------------------- 4. pass unit
def line_of(per_person, n_boot=BOOT_N, seed=SEED):
    """{'point' mean over persons, 'interval' person bootstrap (None under 2 persons), 'n_persons', 'per_person'}."""
    from mdfm.gates import common
    ids = sorted(per_person)
    v = np.array([per_person[i] for i in ids], np.float64)
    if not len(ids):
        return {"point": float("nan"), "interval": None, "n_persons": 0, "per_person": {}}
    iv = list(common.cluster_bootstrap(v, np.array(ids), n=n_boot, seed=seed, level=LEVEL)) if len(ids) >= 2 else None
    return {"point": float(v.mean()), "interval": iv, "n_persons": int(len(ids)),
            "per_person": {i: float(x) for i, x in zip(ids, v)}}


def person_errors(persons, ref, pred, mask):
    """{person: mean abs(pred - ref)} over the units of mask whose pred and ref are finite."""
    p = np.asarray(persons).astype(str)
    r, y = np.asarray(ref, np.float64), np.asarray(pred, np.float64)
    ok = np.asarray(mask, bool) & np.isfinite(r) & np.isfinite(y)
    out = {}
    for q in np.unique(p[ok]):
        m = ok & (p == q)
        out[str(q)] = float(np.mean(np.abs(y[m] - r[m])))
    return out


def score_chain(persons, ref, values, sets, needed, n_boot=BOOT_N, seed=SEED):
    """The rows of one checkpoint in the pass unit on one pass set (section 3a, Unit): a unit enters when its reference
    and every row of `needed` are finite. Returns ({set: {row: line_of}}, {set: counts}, mask). A row outside `needed`
    (beside rows) is scored on the units of the set where it is finite, counted."""
    p = np.asarray(persons).astype(str)
    r = np.asarray(ref, np.float64)
    has_ref = np.isfinite(r)
    keep = has_ref.copy()
    for k in needed:
        keep &= np.isfinite(np.asarray(values[k], np.float64))
    lines, counts = {}, {}
    for s, members in sets.items():
        ins = np.isin(p, list(members))
        lines[s] = {k: line_of(person_errors(p, r, v, keep & ins), n_boot, seed) for k, v in values.items()}
        counts[s] = {"passes": int(ins.sum()), "with_reference": int((ins & has_ref).sum()),
                     "in_chain": int((ins & keep).sum()), "left_out": int((ins & has_ref & ~keep).sum()),
                     "left_out_by_row": {k: int((ins & has_ref & ~np.isfinite(np.asarray(values[k], np.float64))).sum())
                                         for k in needed},
                     "beside_rows_scored_passes": {k: int((ins & keep & np.isfinite(np.asarray(v, np.float64))).sum())
                                                   for k, v in values.items() if k not in needed},
                     "persons_in_chain": int(len(set(p[ins & keep])))}
    return lines, counts, keep


def paired(y_pp, x_pp, n_boot=BOOT_N, seed=SEED):
    """Y minus X per person over the persons both hold: {'mean', 'interval', 'n_persons', 'lower_above_0'}."""
    from mdfm.gates import common
    ids = sorted(k for k in set(y_pp) & set(x_pp) if fin(y_pp[k]) is not None and fin(x_pp[k]) is not None)
    if len(ids) < 2:
        return {"mean": None, "interval": None, "n_persons": len(ids), "lower_above_0": False}
    d = np.array([y_pp[k] - x_pp[k] for k in ids], np.float64)
    iv = list(common.cluster_bootstrap(d, np.array(ids), n=n_boot, seed=seed, level=LEVEL))
    return {"mean": float(d.mean()), "interval": iv, "n_persons": len(ids), "lower_above_0": bool(iv[0] > 0)}


# ---------------------------------------------------------------- 5. margin rule and readings
def relation(x_by_end, y_by_end, ends=B_ENDS, n_boot=BOOT_N, seed=SEED):
    """'X better than Y by the margin' (section 3a, Margin rule). x_by_end, y_by_end: {end: line_of of S_pass(12)}.
    Returns {'relation': 'holds' | 'fails' | 'split' | 'not evaluated', 'm', 'd', 'Y1', 'ends': {end: {'G',
    'paired', 'better'}}}."""
    e1, e2 = ends
    vals = [fin((d.get(e) or {}).get("point")) for d in (x_by_end, y_by_end) for e in ends]
    if any(v is None for v in vals):
        return {"relation": "not evaluated", "m": None, "d": None, "Y1": None, "ends": {}}
    X1, X2, Y1, Y2 = vals
    d = max(abs(X1 - X2), abs(Y1 - Y2))
    m = max(MARGIN_FRAC * Y1, 2.0 * d)
    per = {}
    for e in ends:
        G = float(y_by_end[e]["point"] - x_by_end[e]["point"])
        pr = paired(y_by_end[e]["per_person"], x_by_end[e]["per_person"], n_boot, seed)
        per[e] = {"G": G, "X": float(x_by_end[e]["point"]), "Y": float(y_by_end[e]["point"]), "paired": pr,
                  "G_at_least_m": bool(G >= m), "better": bool(G >= m and pr["lower_above_0"])}
    b = [per[e]["better"] for e in ends]
    rel_ = "holds" if all(b) else "fails" if not any(b) else "split"
    return {"relation": rel_, "m": m, "d": d, "Y1": Y1, "X1": X1, "X2": X2, "Y2": Y2, "ends": per}


def reading_of(rel_):
    """CARRY, BOTTLENECK or SPLIT of a relation (latent against a decoded row); PENDING when not evaluated."""
    return {"holds": CARRY, "fails": BOTTLENECK, "split": SPLIT}.get(rel_, PENDING)


RULE_TEXT = {
    1: ("GAIN: F2 works on the scale sources first (shape and leg length under IMU-only input, the decoder's "
        "root-velocity path, root integration; plan F2); the E1 screen is parked until the gain is explained (plan "
        "JP0). Once it is, reading R routes what remains, by rules 2 to 4 with R in place of D."),
    2: ("Not GAIN, CARRY at D: F2 goes to decoding first; the E1 screen runs only as a test of its predicted null (no "
        "go), after F2's decoding candidates and on the owner's word."),
    "2R": ("Not GAIN, CARRY at D, R BOTTLENECK: the scale correction closes the probe's advantage although F1's line "
           "was missed: F2 takes the scale sources before decoding, and the screen does not run."),
    3: ("Not GAIN, BOTTLENECK at D (R is reported and D routes): the representation is the lever; the E1 screen "
        "becomes eligible under the plan's other conditions (JP1, the minimum detectable effect), and F2 keeps its "
        "inference-only decoding candidates."),
    4: ("Not GAIN, SPLIT at D: JP0 chooses no lever; F2 runs its inference-only candidates on both sides, and the E1 "
        "screen is not eligible."),
}
SCREEN = {1: "parked until the gain is explained", 2: "null test only, after F2's decoding candidates, on the owner's "
          "word", "2R": "does not run", 3: "eligible under JP1 (minimum detectable effect)", 4: "not eligible"}


def route(gain, D, R):
    """The routing of section 3a, in its order. gain: F1's reading; D, R: readings. Returns {'rule', 'text',
    'e1_screen', 'routed_by', 'after_gain_explained'}."""
    if gain not in ("GAIN", "not GAIN"):
        return {"rule": None, "text": f"pending: F1's GAIN reading is {gain!r}, so rule 1 cannot be ruled in or out",
                "e1_screen": "pending", "routed_by": None, "after_gain_explained": None}
    if gain == "GAIN":
        after = {CARRY: 2, BOTTLENECK: 3, SPLIT: 4}.get(R)
        return {"rule": 1, "text": RULE_TEXT[1], "e1_screen": SCREEN[1], "routed_by": "GAIN",
                "after_gain_explained": {"reading": "R", "value": R, "rule": after,
                                         "text": RULE_TEXT[after] if after else "R pending"}}
    if D == CARRY:
        k = "2R" if R == BOTTLENECK else 2
        return {"rule": k, "text": RULE_TEXT[k], "e1_screen": SCREEN[k], "routed_by": "D, and R under rule 2",
                "after_gain_explained": None}
    if D == BOTTLENECK:
        return {"rule": 3, "text": RULE_TEXT[3], "e1_screen": SCREEN[3], "routed_by": "D (R reported)",
                "after_gain_explained": None}
    if D == SPLIT:
        return {"rule": 4, "text": RULE_TEXT[4], "e1_screen": SCREEN[4], "routed_by": "D", "after_gain_explained": None}
    return {"rule": None, "text": f"pending: reading D is {D!r}", "e1_screen": "pending", "routed_by": None,
            "after_gain_explained": None}


def sub_readings(main, lm_rel, sl_rel):
    """LM (latent better than mu_z holds) and SL (latent better than stem fails) with their implications under the
    main reading (section 3a, Sub-readings)."""
    LM = {"holds": True, "fails": False, "split": "split"}.get(lm_rel, PENDING)
    SL = {"fails": True, "holds": False, "split": "split"}.get(sl_rel, PENDING)
    imp = []
    if LM is True:
        imp.append("LM puts the latent head on F2's list (the width of z and LatentHead, a spec change for the owner).")
        if main == CARRY:
            imp.append("Under CARRY, LM names the head as the decoding lever.")
        if main == BOTTLENECK:
            imp.append("Under BOTTLENECK with LM the screen stays eligible, and its go line stays on the latent probe.")
    elif LM is False and main == CARRY:
        imp.append("Under CARRY without LM, the lever is the path after mu_z (refinement, decoder, root integration, "
                   "contact readout under OD12).")
    if main == BOTTLENECK:
        if SL is True:
            imp.append("SL under BOTTLENECK: the screen is the matching test, and its report adds E1's stem row.")
        elif SL is False:
            imp.append("SL failing leaves the screen eligible; section 3's prediction under BOTTLENECK applies.")
        elif SL == "split":
            imp.append("SL is split, reported as split.")
    elif main == CARRY:
        imp.append("Under CARRY, SL is reported only.")
    return {"LM": LM, "LM_relation": lm_rel, "SL": SL, "SL_relation": sl_rel, "main_reading": main,
            "implications": imp}


def section3_reading(D, P1, P2, d_pp=None, p1_pp=None, n_boot=BOOT_N, seed=SEED):
    """Section 3's reading in the G3 per-stride unit: CARRY when D - P1 >= max(0.10 D, 2 dP), BOTTLENECK otherwise;
    with the paired person interval of D minus P1 when per-person values are given."""
    vals = [fin(v) for v in (D, P1, P2)]
    if any(v is None for v in vals):
        return {"reading": "not evaluated"}
    D, P1, P2 = vals
    dP = abs(P1 - P2)
    m = max(MARGIN_FRAC * D, 2.0 * dP)
    out = {"D": D, "P1": P1, "P2": P2, "dP": dP, "margin": m, "D_minus_P1": D - P1,
           "reading": CARRY if D - P1 >= m else BOTTLENECK}
    if d_pp and p1_pp:
        out["paired_D_minus_P1"] = paired(d_pp, p1_pp, n_boot, seed)
    return out


# ---------------------------------------------------------------- J0 and decoded inputs
def j0_dir(tree, name):
    return os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}")


def diag_dir(tree, name):
    return os.path.join(tree, "reports", "gates", "g3", f"diag_{name}")


def key_of(df):
    return list(zip(df["trial_id"].astype(str), df["pass_id"].astype(str)))


def decoded_passes(events_path, diag_path, sets, locked):
    """Per pass (trial_id, pass_id) the mean of the matched base-row decoded strides (cm) and the matched decoded pass
    speed (m/s) of an events.csv, with its persons guarded (dev14 only, no lockbox person) and S(12), S(14) per stride
    recomputed and asserted equal to diag.json's base row within DECODED_TOL_CM. Returns (stride {key: cm}, speed
    {key: m/s}, consistency, diag doc)."""
    import pandas as pd
    who = pd.read_csv(events_path, encoding="utf-8", usecols=["person_id"], dtype=str)
    persons = set(who["person_id"].dropna())
    if persons & set(locked):
        raise Stop(EXIT_INPUT, f"{events_path}: a lockbox person")
    if not persons <= set(sets["dev14"]):
        raise Stop(EXIT_INPUT, f"{events_path}: persons outside the dev controls")
    text = {c: str for c in ("row", "kind", "trial_id", "person_id", "task", "foot", "pass_id", "unit", "matched")}
    ev = pd.read_csv(events_path, encoding="utf-8", dtype=text, keep_default_na=False)
    ev = ev[(ev["row"] == "base") & (ev["matched"].str.strip().str.lower() == "true")]
    st = ev[ev["kind"] == "stride"].copy()
    sp = ev[ev["kind"] == "speed"].copy()
    if (st["unit"] != "cm").any() or (sp["unit"] != "m/s").any():
        raise Stop(EXIT_INPUT, f"{events_path}: unexpected units")
    for d in (st, sp):
        d["x"] = pd.to_numeric(d["reference"]).astype(float)
        d["y"] = pd.to_numeric(d["prediction"]).astype(float)
    diag = read_json(diag_path)
    base = diag["rows"]["base"]
    cons = {}
    for s, sub in (("dev14", "all14"), ("dev12", "no_whc003_whc026")):
        k = st[st["person_id"].isin(set(sets[s]))]
        pp = k.assign(e=(k["y"] - k["x"]).abs()).groupby("person_id")["e"].mean()
        mine = float(pp.mean()) if len(pp) else float("nan")
        ref = fin(base[sub].get("stride_mae_cm"))
        ok = ref is not None and math.isfinite(mine) and abs(mine - ref) <= DECODED_TOL_CM
        cons[s] = {"recomputed_cm": mine, "diag_cm": ref, "ok": ok}
        if not ok:
            raise Stop(EXIT_INPUT, f"{events_path}: S per stride {mine} against diag.json {ref} on {s}")
    stride = {kk: float(v) for kk, v in st.groupby(["trial_id", "pass_id"])["y"].mean().items()}
    speed = {}
    for r in sp.itertuples():
        kk = (r.trial_id, r.pass_id)
        if kk in speed:
            raise Stop(EXIT_INPUT, f"{events_path}: two speed rows for pass {kk}")
        speed[kk] = float(r.y)
    return stride, speed, cons, diag


def refit_probes(data, u_tr, u_dev):
    """{variant: {'stride_m': dev predictions, 'speed_m_s': ..., 'alpha': {...}}}: the ridge pass probes of
    scripts/probe_frozen_tokens.pass_probes (probes.fit_probe on the 54, applied to the dev passes)."""
    from mdfm.eval import probes
    p_tr = u_tr["person_id"].to_numpy().astype(str)
    probes.check_disjoint(p_tr, u_dev["person_id"])
    out = {}
    for v in PROBES:
        Xtr, Xdev = data["train"]["X"][v], data["dev"]["X"][v]
        out[v] = {"alpha": {}}
        for tgt in ("stride_m", "speed_m_s"):
            m, s = probes.fit_probe(Xtr, u_tr[tgt].to_numpy(np.float64), p_tr)
            out[v][tgt] = m.predict(Xdev)
            out[v]["alpha"][tgt] = s["alpha"]
    return out


def per_stride_lines(u_dev, preds, tgs):
    """{variant: {set: {'stride_mae_cm', 'speed_mae_m_s'}}} of the refitted probes in the G3 per-stride unit
    (probes.score of probes.pass_predictions, as J0 scores them)."""
    from mdfm.eval import probes
    out = {}
    for v, pr in preds.items():
        out[v] = {}
        for s, tg in tgs.items():
            sc = probes.score(tg, [probes.pass_predictions(u_dev, {"stride_m": pr["stride_m"],
                                                                    "speed_m_s": pr["speed_m_s"]}, tg, v)])[v]
            out[v][s] = {"stride_mae_cm": sc["stride_mae_cm"]["point"], "speed_mae_m_s": sc["speed_mae_m_s"]["point"]}
    return out


def same_units(a, b):
    return len(a) == len(b) and key_of(a) == key_of(b)


def load_checkpoint_inputs(tree, name, u_tr, u_dev, sets):
    """J0's results.json, per_pass.csv and feature file of one checkpoint, checked (features SHA-256 as J0 recorded
    it, checkpoint name, full run, sets, pass order). Returns (results, per_pass frame, data, inputs {label: path})."""
    import pandas as pd
    PFT = _sibling("probe_frozen_tokens")
    d = j0_dir(tree, name)
    res_p, pp_p = os.path.join(d, "results.json"), os.path.join(d, "per_pass.csv")
    for p in (res_p, pp_p):
        if not os.path.exists(p):
            raise Stop(EXIT_INPUT, f"J0 output missing: {rel(p, tree)}")
    res = read_json(res_p)
    if res.get("partial") or (res.get("checkpoint") or {}).get("name") != name:
        raise Stop(EXIT_INPUT, f"{rel(res_p, tree)}: partial run or another checkpoint")
    for s in ("dev14", "dev12"):
        if sorted(res["sets"][s]) != sorted(sets[s]):
            raise Stop(EXIT_INPUT, f"{rel(res_p, tree)}: set {s} differs from the controls read now")
    if sorted(res["sets"]["train"]) != sorted(sets["train"]):
        raise Stop(EXIT_INPUT, f"{rel(res_p, tree)}: training-fold set differs from the controls read now")
    feat = os.path.join(tree, res["features"]["path"])
    if not os.path.exists(feat):
        raise Stop(EXIT_INPUT, f"feature file missing: {res['features']['path']}")
    fsha = sha256_file(feat)
    if fsha != res["features"]["sha256"]:
        raise Stop(EXIT_INPUT, f"{res['features']['path']}: SHA-256 {fsha}, J0 recorded {res['features']['sha256']}")
    data, meta = PFT.load_features(feat)
    PFT.check_features_meta(meta, name, res["features"]["path"])
    for s, u in (("train", u_tr), ("dev", u_dev)):
        if not same_units(data["units"][s], u):
            raise Stop(EXIT_INPUT, f"{res['features']['path']}: {s} passes differ from the targets read now")
        data["units"][s] = u
    pp = pd.read_csv(pp_p, encoding="utf-8", dtype={"trial_id": str, "pass_id": str, "person_id": str})
    if not same_units(pp, u_dev):
        raise Stop(EXIT_INPUT, f"{rel(pp_p, tree)}: passes differ from the targets read now")
    return res, pp, data, {"results": res_p, "per_pass": pp_p, "features": feat}


def supervised_passes(tree, u_dev):
    """{key: predicted stride (m), predicted speed} of per_pass_dev of the supervised baseline, with its reference
    stride checked against the dev units (SUPERVISED_REF_TOL_M), the path and the 12-person per-stride line."""
    path = os.path.join(tree, SUPERVISED_REL)
    doc = read_json(path)
    ref = dict(zip(key_of(u_dev), u_dev["stride_m"].to_numpy(np.float64)))
    st, sp, worst = {}, {}, 0.0
    for r in doc["per_pass_dev"]:
        k = (str(r["trial_id"]), str(r["pass_id"]))
        st[k], sp[k] = fin(r.get("pred_stride_m")), fin(r.get("pred_speed_m_s"))
        a, b = fin(r.get("stride_m")), fin(ref.get(k))
        if (a is None) != (b is None) or (a is not None and abs(a - b) > SUPERVISED_REF_TOL_M):
            raise Stop(EXIT_INPUT, f"{SUPERVISED_REL}: reference stride of pass {k} {a} differs from the targets {b}")
        if a is not None:
            worst = max(worst, abs(a - b))
    per_stride = {s: fin(((doc.get("lines") or {}).get(s) or {}).get("supervised", {}).get("stride_mae_cm", {}).get("point"))
                  for s in SETS}
    return st, sp, {"path": path, "per_stride_stride_mae_cm": per_stride, "max_reference_difference_m": worst,
                    "passes": len(doc["per_pass_dev"])}


def as_array(u_dev, mapping, scale=1.0):
    return np.array([np.nan if mapping.get(k) is None else scale * float(mapping[k]) for k in key_of(u_dev)],
                    np.float64)


# ---------------------------------------------------------------- one checkpoint
def analyse_checkpoint(tree, key, name, ctx, f1):
    """The chain of one checkpoint (section 3a, Unit and Chain): refit, refit check, rows, lines on the 12 and the
    14, stride and speed, adjacent intervals. ctx: targets, units, sets, lockbox list, population, supervised."""
    u_tr, u_dev, sets, tgs = ctx["u_tr"], ctx["u_dev"], ctx["sets"], ctx["tgs"]
    res, pp, data, inputs = load_checkpoint_inputs(tree, name, u_tr, u_dev, sets)
    preds = refit_probes(data, u_tr, u_dev)
    lines_ps = per_stride_lines(u_dev, preds, {s: tgs[s] for s in ("dev14", "dev12")})
    check = check_refit({s: res["passes"]["latent"]["lines"][s]["stride_mae_cm"]["point"] for s in ("dev14", "dev12")},
                        {s: lines_ps["latent"][s]["stride_mae_cm"] for s in ("dev14", "dev12")})
    j0_vs_refit = {v: {s: {ln: {"j0": fin(res["passes"][v]["lines"][s][ln]["point"]), "refit": fin(lines_ps[v][s][ln])}
                           for ln in ("stride_mae_cm", "speed_mae_m_s")} for s in ("dev14", "dev12")} for v in PROBES}
    pp_diff = np.abs(pp["pred_stride_m"].to_numpy(np.float64) - preds["latent"]["stride_m"]) * 100.0
    ck = res["checkpoint"]
    stride = {v: 100.0 * np.asarray(preds[v]["stride_m"], np.float64) for v in PROBES}
    speed = {v: np.asarray(preds[v]["speed_m_s"], np.float64) for v in PROBES}
    dd = diag_dir(tree, name)
    has_diag = os.path.exists(os.path.join(dd, "events.csv"))
    decoded_info, gain = None, (f1["gains"].get(name) or {})
    if has_diag:
        ds, dv, cons, diag = decoded_passes(os.path.join(dd, "events.csv"), os.path.join(dd, "diag.json"), sets,
                                            ctx["locked"])
        if (diag.get("checkpoint") or {}).get("sha256") != ck.get("sha256"):
            raise Stop(EXIT_INPUT, f"{rel(dd, tree)} is of checkpoint {(diag.get('checkpoint') or {}).get('sha256')}, "
                                   f"J0 read {ck.get('sha256')}")
        stray = sorted((set(ds) | set(dv)) - set(key_of(u_dev)))
        if stray:
            raise Stop(EXIT_INPUT, f"{rel(dd, tree)}/events.csv: {len(stray)} decoded passes not among the dev passes, "
                                   f"e.g. {stray[:3]}")
        stride["decoded"], speed["decoded"] = as_array(u_dev, ds), as_array(u_dev, dv)
        inputs.update(events=os.path.join(dd, "events.csv"), diag=os.path.join(dd, "diag.json"))
        decoded_info = {"consistency": cons, "diag": rel(dd, tree)}
        if gain and gain.get("checkpoint_sha256") not in (None, ck.get("sha256")):
            raise Stop(EXIT_INPUT, f"F1's end {gain.get('key')} is of checkpoint {gain.get('checkpoint_sha256')}, J0 read "
                                   f"{ck.get('sha256')}")
        if usable_gain(gain.get("stride")):
            stride["recalibrated"] = recalibrate(stride["decoded"], gain["stride"])
        if usable_gain(gain.get("speed")):
            speed["recalibrated"] = recalibrate(speed["decoded"], gain["speed"])
    sup_s, sup_v = ctx["supervised"]
    stride["supervised"], speed["supervised"] = as_array(u_dev, sup_s, 100.0), as_array(u_dev, sup_v)
    pop = ctx["population"]
    stride["population"] = np.array([100.0 * pop[t][0] for t in u_dev["task"]], np.float64)
    speed["population"] = np.array([pop[t][1] for t in u_dev["task"]], np.float64)
    needed = NEEDED + (("decoded",) if has_diag else ())
    persons = u_dev["person_id"].to_numpy().astype(str)
    out = {"key": key, "name": name, "checkpoint": {k: ck.get(k) for k in ("path", "sha256", "weights", "phase", "step")},
           "has_decoded": has_diag, "has_recalibrated": "recalibrated" in stride, "needed_rows": list(needed),
           "f1_gain": gain or None, "decoded": decoded_info,
           "refit": {"check": check, "tolerance_cm": REFIT_TOL_CM, "per_stride_j0_vs_refit": j0_vs_refit,
                     "alphas": {v: preds[v]["alpha"] for v in PROBES},
                     "latent_per_pass_max_abs_difference_cm_against_per_pass_csv": float(np.nanmax(pp_diff))
                     if np.isfinite(pp_diff).any() else None},
           "j0_per_stride": {v: {s: {ln: fin(res["passes"][v]["lines"][s][ln]["point"])
                                     for ln in ("stride_mae_cm", "speed_mae_m_s")} for s in SETS} for v in PROBES},
           "j0_decoded_per_stride": {s: fin(((res.get("comparison") or {}).get("decoded") or {}).get(s, {})
                                            .get("stride_mae_cm", {}).get("point")) for s in SETS},
           "j0_latent_per_person_dev12": res["passes"]["latent"]["per_person"]["dev12"].get("stride_mae_cm", {}),
           "inputs": {k: rel(v, tree) for k, v in inputs.items()},
           "inputs_sha256": {k: sha256_file(v) for k, v in inputs.items()}}
    for q, vals, refcol in (("stride", stride, "stride_m"), ("speed", speed, "speed_m_s")):
        ref = u_dev[refcol].to_numpy(np.float64) * (100.0 if q == "stride" else 1.0)
        lines, counts, _ = score_chain(persons, ref, vals, {s: sets[s] for s in SETS}, needed)
        out[q] = {"unit": "cm" if q == "stride" else "m/s", "lines": lines, "pass_set": counts}
        out[q]["adjacent"] = {s: {f"{a} to {b}": paired(lines[s][b]["per_person"], lines[s][a]["per_person"])
                                  for a, b in ADJACENT if a in lines[s] and b in lines[s]} for s in SETS}
    return out


# ---------------------------------------------------------------- report
ROW_NOTE = {"stem": "probe, tokenizer stems", "measurement": "probe, reported", "latent": "probe, primary",
            "mu_z": "probe, latent head mean", "subject": "probe, reported", "decoded": "decoded body",
            "recalibrated": "decoded / F1 stride gain", "supervised": "supervised baseline, beside",
            "population": "population baseline, beside"}


def summary_md(r):
    rd, rt = r["readings"], r["route"]
    L = [f"# JP0 reading of J0 ({LABEL})", "",
         "Written by scripts/jp0_read.py (task JP0-READ) under section 3a of docs/proposals/JEPA_EXPLORATION_DRAFT.md, "
         f"SHA-256 {r['addendum']['sha256']}, asserted before any other file was read. Every number below is read from "
         "results.json of this folder. Dev only; no lockbox, EVAL or CODY record; no gain fitted here.", "",
         "## Readings at the B ends (stride, pass unit, the 12)", "",
         f"- GAIN (F1 alone, {r['f1']['path']}): **{rd['GAIN']}**",
         f"- D, latent probe against the decoded row: **{rd['D']}** (relation {r['relations']['D']['relation']})",
         f"- R, latent probe against the recalibrated decoded row: **{rd['R']}** (relation "
         f"{r['relations']['R']['relation']})",
         f"- LM, latent better than mu_z: **{rd['LM']}** (relation {r['relations']['LM']['relation']})",
         f"- SL, stem within the margin of latent: **{rd['SL']}** (relation latent better than stem: "
         f"{r['relations']['SL']['relation']})", "",
         f"Route: rule {rt['rule']}. {rt['text']} E1 screen: {rt['e1_screen']}."]
    if rt.get("after_gain_explained"):
        a = rt["after_gain_explained"]
        L.append(f"Once the gain is explained: R {a['value']} routes what remains by rule {a['rule']}, read with R in "
                 f"place of D: {a['text']}")
    if r["sub_readings"]["implications"]:
        L += ["", "Sub-readings: " + " ".join(r["sub_readings"]["implications"])]
    s3 = r["section3"]
    sup = r["supervised"]["per_stride_stride_mae_cm"]
    L += ["", "Beside the readings (reported, not read): section 3's reading in the G3 per-stride unit "
          f"**{s3.get('reading')}** (D {fmt(s3.get('D'))} cm, P1 {fmt(s3.get('P1'))}, P2 {fmt(s3.get('P2'))}, D minus P1 "
          f"{fmt(s3.get('D_minus_P1'))} against the margin {fmt(s3.get('margin'))}; paired D minus P1 "
          f"{fmt((s3.get('paired_D_minus_P1') or {}).get('mean'))} {fmt_iv((s3.get('paired_D_minus_P1') or {}).get('interval'))});"
          f" {r['agreement']}. Supervised baseline per stride on the 12: {fmt(sup.get('dev12'))} cm "
          f"({r['supervised']['path']}).", "",
          "## Margin relations (S_pass(12), cm)", "",
          "| relation | Y1 | d | m | end | X | Y | G | paired Y minus X | 95 % interval | better |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rl in r["relations"].items():
        if not rl.get("ends"):
            L.append(f"| {name} ({rl['x']} vs {rl['y']}) | n/a | n/a | n/a | n/a | | | | | | {rl['relation']} |")
            continue
        for e, v in rl["ends"].items():
            L.append(f"| {name} ({rl['x']} vs {rl['y']}) | {fmt(rl['Y1'])} | {fmt(rl['d'])} | {fmt(rl['m'])} | {e} | "
                     f"{fmt(v['X'])} | {fmt(v['Y'])} | {fmt(v['G'])} | {fmt(v['paired']['mean'])} | "
                     f"{fmt_iv(v['paired']['interval'])} | {v['better']} |")
    for key, c in r["checkpoints"].items():
        L += ["", f"## Chain, {c['name']} ({key})", "",
              "| row | what | S_pass(12) cm | 95 % | S_pass(14) cm | V_pass(12) m/s | V_pass(14) m/s | J0 per stride, 12 (cm) |",
              "|---|---|---|---|---|---|---|---|"]
        for row in ROWS:
            a = c["stride"]["lines"]["dev12"].get(row)
            if a is None:
                continue
            b = c["stride"]["lines"]["dev14"][row]
            v12 = (c["speed"]["lines"]["dev12"].get(row) or {}).get("point")
            v14 = (c["speed"]["lines"]["dev14"].get(row) or {}).get("point")
            j0 = (c["j0_per_stride"].get(row) or {}).get("dev12", {}).get("stride_mae_cm") if row in PROBES else \
                c["j0_decoded_per_stride"]["dev12"] if row == "decoded" else None
            L.append(f"| {row} | {ROW_NOTE[row]} | {fmt(a['point'])} | {fmt_iv(a['interval'])} | {fmt(b['point'])} | "
                     f"{fmt(v12, 3)} | {fmt(v14, 3)} | {fmt(j0)} |")
        ps = c["stride"]["pass_set"]
        L += ["", "Pass set (stride): " + "; ".join(
            f"{s}: {ps[s]['in_chain']} of {ps[s]['with_reference']} passes with a reference in the chain "
            f"({ps[s]['persons_in_chain']} persons), {ps[s]['left_out']} left out "
            f"({', '.join(f'{k} {n}' for k, n in ps[s]['left_out_by_row'].items())})" for s in SETS) + "."]
        ck = c["refit"]["check"]
        L.append("Refit check, latent per-stride stride MAE, refit minus J0: " + ", ".join(
            f"{s} {fmt(v['difference'], 6)} cm" for s, v in ck.items()) + f" (tolerance {c['refit']['tolerance_cm']} cm); "
            f"largest per-pass difference against per_pass.csv "
            f"{fmt(c['refit']['latent_per_pass_max_abs_difference_cm_against_per_pass_csv'], 6)} cm.")
        if c.get("f1_gain"):
            L.append(f"F1 gains of this checkpoint: stride {fmt(c['f1_gain'].get('stride'), 4)}, speed "
                     f"{fmt(c['f1_gain'].get('speed'), 4)} (recalibrated = decoded / gain).")
        L += ["", "Adjacent rows, paired later minus earlier on the 12 (cm): " + "; ".join(
            f"{k} {fmt(v['mean'])} {fmt_iv(v['interval'])}" for k, v in c["stride"]["adjacent"]["dev12"].items()) + "."]
    jp = r.get("jp2_go_line")
    if jp:
        L += ["", "## JP2 go line at ckpt_B_004000 (reported)", "",
              f"v1 latent S_pass(12) {fmt(jp['Y1'])} cm, seed 2 {fmt(jp['latent_s2'])} cm, d {fmt(jp['d'])}, margin "
              f"{fmt(jp['m'])}: E1's latent S_pass(12) at ckpt_B_004000 at most {fmt(jp['threshold'])} cm, with the paired "
              "lower bound above 0."]
    L += ["", "## Inputs", ""] + [f"- {k}: {v['path']} (SHA-256 {v['sha256']})" for k, v in r["inputs"].items()]
    L += ["", f"MoveAtlas: HEAD {r.get('moveatlas_head') or 'n/a'}. Run started {r['run_started']}, "
          f"{r['timing_s']} s.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="root holding the inputs (default this tree)")
    p.add_argument("--out", default=None, help=f"default <tree>/{OUT_REL}")
    return p.parse_args(argv)


def run(tree, out):
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    sha = check_addendum(tree)                                     # 1: before anything else is read
    f1 = read_f1(tree)                                             # 2: before any J0 output is opened
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise Stop(EXIT_INPUT, "the reading runs on the CPU: set CUDA_VISIBLE_DEVICES=-1")
    import mdfm  # noqa: F401
    from mdfm import paths
    from mdfm.eval import probes
    from mdfm.gates import g3
    names = dict((k, n) for k, n, _ in CHECKPOINTS)
    present = {k: os.path.exists(os.path.join(j0_dir(tree, n), "results.json")) for k, n in names.items()}
    missing = [k for k in REQUIRED if not present[k]]
    if missing:
        raise Stop(EXIT_INPUT, f"J0 outputs missing for {missing}")
    keys = list(REQUIRED) + (list(OPTIONAL_PAIR) if all(present[k] for k in OPTIONAL_PAIR) else [])
    train, dev = list(g3.controls("train")), list(g3.controls("dev"))
    locked = g3.lockbox_persons()
    if len(train) != 54 or len(dev) != 14 or set(train) & set(dev) or (set(train) | set(dev)) & set(locked):
        raise Stop(EXIT_INPUT, "controls: not 54 and 14 disjoint non-lockbox persons")
    sets = {"train": train, "dev14": dev, "dev12": [p for p in dev if p not in EXCLUDED_FROM_12]}
    tg_train, tg_dev = g3.load_targets(train), g3.load_targets(dev)
    if tg_train.counts["lockbox_rows"] or tg_dev.counts["lockbox_rows"]:
        raise Stop(EXIT_INPUT, "a lockbox target row was kept")
    u_tr, u_dev = probes.pass_units(tg_train), probes.pass_units(tg_dev)
    tgs = {"dev14": tg_dev, "dev12": probes.subset_targets(tg_dev, sets["dev12"])}
    sup_s, sup_v, sup_info = supervised_passes(tree, u_dev)
    ctx = {"u_tr": u_tr, "u_dev": u_dev, "sets": sets, "tgs": tgs, "locked": locked,
           "population": g3.population_baseline(tg_train), "supervised": (sup_s, sup_v)}
    cks = {}
    for k in keys:
        print(f"{k}: {names[k]}", flush=True)
        cks[k] = analyse_checkpoint(tree, k, names[k], ctx, f1)

    def s12(row, k):
        return (cks[k]["stride"]["lines"]["dev12"].get(row)) if k in cks else None

    def rel_of(x, y):
        rr = relation({e: s12(x, e) for e in B_ENDS}, {e: s12(y, e) for e in B_ENDS})
        rr.update(x=x, y=y)
        return rr

    rels = {"D": rel_of("latent", "decoded"), "R": rel_of("latent", "recalibrated"), "LM": rel_of("latent", "mu_z"),
            "SL": rel_of("latent", "stem")}
    D = reading_of(rels["D"]["relation"])
    R = reading_of(rels["R"]["relation"])
    main_reading = R if f1["reading"] == "GAIN" else D
    subs = sub_readings(main_reading, rels["LM"]["relation"], rels["SL"]["relation"])
    readings = {"GAIN": f1["reading"], "D": D, "R": R, "LM": subs["LM"], "SL": subs["SL"]}
    rt = route(f1["reading"], D, R)
    v1 = cks["v1_B"]
    diag_v1 = read_json(os.path.join(diag_dir(tree, names["v1_B"]), "diag.json"))
    base12 = diag_v1["rows"]["base"]["no_whc003_whc026"]
    d_pp = {p: fin(v.get("stride_cm")) for p, v in (base12.get("per_person") or {}).items()}
    s3 = section3_reading(base12.get("stride_mae_cm"), v1["j0_per_stride"]["latent"]["dev12"]["stride_mae_cm"],
                          cks["s2_B"]["j0_per_stride"]["latent"]["dev12"]["stride_mae_cm"],
                          {p: v for p, v in d_pp.items() if v is not None}, v1["j0_latent_per_person_dev12"])
    agree = ("section 3's per-stride reading and reading D agree" if s3.get("reading") == D else
             f"DISAGREEMENT: section 3's per-stride reading is {s3.get('reading')}, reading D is {D}; section 3a's "
             "reading routes")
    jp2 = None
    if "v1_B4000" in cks and "s2_B4000" in cks:
        a, b = s12("latent", "v1_B4000")["point"], s12("latent", "s2_B4000")["point"]
        d = abs(a - b)
        m = max(MARGIN_FRAC * a, 2.0 * d)
        jp2 = {"Y1": a, "latent_s2": b, "d": d, "m": m, "threshold": a - m}
    inputs = {"proposal": os.path.join(tree, PROPOSAL_REL), "addendum_record": os.path.join(tree, RECORD_REL),
              "f1_results": os.path.join(tree, F1_RESULTS_REL), "f1_rule": os.path.join(tree, F1_RULE_REL),
              "supervised": sup_info["path"]}
    for k, c in cks.items():
        for lab, pth in c["inputs"].items():
            inputs[f"{k}_{lab}"] = os.path.join(tree, pth)
    for p in g3.TARGET_FILES + (g3.SUMMARY_JSON,):
        inputs[f"targets_{os.path.basename(p)}"] = p
    stamp = _sibling("moveatlas_stamp").stamp()
    res = {"what": LABEL, "task": TASK, "run_started": run_started,
           "addendum": {"sha256": sha, "proposal": PROPOSAL_REL, "record": RECORD_REL, "asserted_first": True},
           "f1": {k: f1[k] for k in ("path", "sha256", "reading", "rule_sha256", "reading_detail", "gains")},
           "readings": readings, "relations": rels, "route": rt, "sub_readings": subs, "section3": s3,
           "agreement": agree, "jp2_go_line": jp2,
           "supervised": {"path": rel(sup_info["path"], tree), "per_stride_stride_mae_cm": sup_info["per_stride_stride_mae_cm"],
                          "max_reference_difference_m": sup_info["max_reference_difference_m"]},
           "sets": {"dev14": dev, "dev12": sets["dev12"], "excluded_from_12": list(EXCLUDED_FROM_12), "n_train": len(train)},
           "lockbox": {"persons_in_lockbox": 0, "target_lockbox_rows": 0},
           "checkpoints": cks,
           "constants": {"boot_n": BOOT_N, "seed": SEED, "level": LEVEL, "margin_frac": MARGIN_FRAC,
                         "refit_tol_cm": REFIT_TOL_CM, "decoded_tol_cm": DECODED_TOL_CM, "probe_seed": probes.SEED,
                         "alphas": list(probes.ALPHAS), "folds": probes.FOLDS},
           "inputs": {k: {"path": rel(v, tree), "sha256": sha256_file(v)} for k, v in inputs.items()},
           "moveatlas_head": stamp.get("head"), "timing_s": round(time.time() - t0, 1)}
    res = clean(res)
    text = summary_md(res)
    for t, w in ((text, "summary.md"), (json.dumps(res), "results.json")):
        if chr(0x2014) in t or chr(0x2013) in t:
            raise Stop(EXIT_INPUT, f"{w} would hold a long dash")
    paths.assert_not_moveatlas(out)
    os.makedirs(out, exist_ok=True)
    code = [os.path.abspath(__file__), probes.__file__, g3.__file__, os.path.join(TREE, "scripts", "probe_frozen_tokens.py")]
    prov = {"script": "scripts/jp0_read.py", "task": TASK, "label": LABEL, "run_at": run_started,
            "addendum_sha256": sha, "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": stamp.get("head"),
            "moveatlas_stamp": stamp, "python": sys.version.split()[0], "numpy": np.__version__,
            "code_sha256": {rel(p): sha256_file(p) for p in code}, "inputs": res["inputs"],
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    for name_, body in (("results.json", json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"),
                        ("summary.md", text),
                        ("_provenance.json", json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")):
        with open(os.path.join(out, name_), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
    print(text)
    print(f"written {rel(out)}")
    return 0


def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    tree = os.path.abspath(a.tree or TREE)
    out = os.path.abspath(a.out or os.path.join(tree, OUT_REL))
    try:
        return run(tree, out)
    except Stop as s:
        print(s.message, file=sys.stderr)
        return s.code


if __name__ == "__main__":
    sys.exit(main())
