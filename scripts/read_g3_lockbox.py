"""B4.21, G3 healthy half: the sealed lockbox read of one checkpoint and its outputs (docs/rules/G3_RULE.md section 9;
harness mdfm/gates/g3.py read_lockbox, task G3H; inference mdfm/eval/infer.py; task G3W).

mdfm.gates.g3.read_lockbox returns a sealed read (a LockboxRead: the GateResult rows, the summary, the matched tables,
and the input-validity table, its report and the two sets as attributes) and writes nothing. This script runs it on one
checkpoint and writes the outputs of rule section 9 under reports/gates/g3/<checkpoint SHA-256>/. A run logs a lockbox
read in reports/gates/lockbox_reads.csv: it is the owner's act, after the owner has approved and registered the rule
(scripts/gate_g3.py --register-rule). A session never runs it.

Order of a run (nothing before read_lockbox opens a lockbox target row or a lockbox IMU record; the script itself never
calls a lockbox reader, read_lockbox is the only one):
1. Arguments: --checkpoint (a checkpoint of mdfm/train/checkpoint.py) and --reason (non-empty; logged with the read,
   a first read included) are required. --reread asks for a second read of the same checkpoint and rule, as
   read_lockbox defines it (a read is keyed on its checkpoint and its rule SHA-256): a reread without a logged read
   of this checkpoint under this rule is refused, and so is a second read under the same rule without --reread. The
   script checks both on the read log (step 4, exit 2) before the read; read_lockbox checks them again.
2. Rule: common.rule(docs/rules/G3_RULE.md, gate G3), the check read_lockbox makes first, made once more here before the
   checkpoint loads, so that an unregistered or edited rule costs no model load; refused: exit 2, nothing written.
   read_lockbox makes it again and stays the authority.
3. Checkpoint: mdfm.eval.infer.load_checkpoint (its file, config, person-list, records and lockbox-list hashes, the
   person guard). Refused (exit 2) when a weargait_pd lockbox person or a G3 dev control (names of
   reports/lockbox/lockbox_persons.csv and of mdfm.data.sources.folds, no lockbox row) is among its training persons.
4. Read log and output folder (the log rows only): the reread semantics of step 1, then the folder, keyed like the
   read on the checkpoint and the rule. The rule folder is reports/gates/g3/<checkpoint SHA-256>/ for the rule of
   the checkpoint's first logged read (or the current rule when the log holds no read of the checkpoint), and
   <checkpoint SHA-256>/rule_<rule SHA-256, 12 characters>/ for any other rule; a first read under a rule writes its
   rule folder, a reread after k logged reads of the same checkpoint and rule writes <rule folder>/reread_<k>/. The
   folder must not exist yet: a read never writes a file into an existing folder; refused (exit 2) before the read
   is logged, the message saying under which rule the existing folder's _provenance.json was written.
5. read_lockbox(checkpoint SHA-256, the model, reread, reason, code_files): its own checks first (the rule, the
   arguments, the read log, the target files, the lockbox control list, the code hash), the dev run with the same model
   (offsets fitted on the 14 dev controls), the log row, and only then the lockbox target rows and records, the
   input-validity check of each record inside the read (rule section 2) and the lines of the two sets.
   The model: infer.G3Predictor with the inference of rule section 3 (10 s windows every 5 s, 20 refinement steps of
   256 points per stream), seed g3.SEED; the weights (--weights, default ema), the shape pooling per trial (default on,
   --no-pool) and the batch windows (--batch-windows, default 16, the default of scripts/eval_g3_dev.py) are not fixed
   by the rule: they are recorded in g3.json and _provenance.json.
6. Outputs, rendered in memory, then written in exclusive-create mode (open mode "x": an existing file is never
   overwritten, the write raises): the folder is created exclusively (when it appeared between step 4 and the end of
   the read, the outputs go to <folder>_<read time> and _provenance.json says so), then validity.csv and g3.json,
   per_person.csv, offsets.csv and g3.md, and _provenance.json last (the read row of the log, the checkpoint, the
   rule, the reads of the checkpoint the log held before the read, the code hash, the inference settings, the
   MoveAtlas HEAD and the SHA-256 of every file written; a file that failed to render, or a read whose log row is not
   under the rule checked here, is listed under errors and the run exits 1).

Outputs (rule section 9):
- g3.json: the GateResult rows (the primary rows on the trials of valid records, set valid_inputs, then the all-trials
  rows, set all_trials, extra decides False), the deciding rows (g3.deciding), the input-validity report, the stats of
  every predictor per set (no per-person values), the offsets, the checkpoint, the rule, the read row and the inference
  settings. Non-finite numbers are written as null (an infinite IC value reads kill in its verdict).
- g3.md: the same, as tables.
- per_person.csv: set, predictor, line, person_id, value.
- offsets.csv: predictor, reference, offset_ms, offset_fitted (the offsets fitted on the dev controls, used in the
  read).
- validity.csv: the input-validity table of every record the read opened (g3.VALIDITY_COLUMNS).
- _provenance.json.

Exit codes: 0 written; 1 failed (a run failure, or a file that could not be rendered or written); 2 refused.

Run (owner, after the registration of docs/rules/G3_RULE.md; mdfm environment):
  <conda>/envs/mdfm/python.exe scripts/read_g3_lockbox.py --checkpoint <run>/ckpt_B_<step>.pt
      --reason "<why this checkpoint is read>"
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

SCRIPT_REL = "scripts/read_g3_lockbox.py"
DEFAULT_ROOT = os.path.join(TREE, "reports", "gates", "g3")
FILES = ("validity.csv", "g3.json", "per_person.csv", "offsets.csv", "g3.md")     # rule section 9, in write order
PROVENANCE = "_provenance.json"
PREDICTORS = ("population", "shank_gyro", "model")
LINE_ORDER = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "ic_median_ms_insole")
BATCH_WINDOWS = 16
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="the checkpoint read (mdfm/train/checkpoint.py)")
    p.add_argument("--reason", required=True, help="why this read is made; logged with the read (non-empty)")
    p.add_argument("--reread", action="store_true",
                   help="a second read of the same checkpoint and rule (read_lockbox refuses it without a first one)")
    p.add_argument("--weights", default="ema", choices=("ema", "model"))
    p.add_argument("--no-pool", action="store_true", help="no shape pooling over the windows of a trial")
    p.add_argument("--batch-windows", type=int, default=BATCH_WINDOWS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-gb", type=float, default=2.4, help="CUDA caching-allocator cap (GB, 1e9 bytes)")
    p.add_argument("--out-root", default=DEFAULT_ROOT,
                   help="the folder that holds <checkpoint SHA-256>/ (default reports/gates/g3)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- formatting
def fin(x):
    """None for a missing or non-finite number, else a float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_iv(iv, nd=2):
    return "n/a" if not iv else f"[{fmt(iv[0], nd)}, {fmt(iv[1], nd)}]"


def yes_no(x):
    return {True: "yes", False: "no"}.get(x, "n/a")


def clean(o):
    """JSON-ready copy: non-finite floats to None, tuples to lists, sets to sorted lists, numpy scalars to Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, (int, str)):
        return o
    if isinstance(o, float):
        return fin(o)
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o


# ---------------------------------------------------------------- folders and exclusive writes
def out_folder(root, checkpoint_sha256, rule_sha256, first_rule_sha256, reread, n_prior):
    """The folder of a read, keyed like the read on (checkpoint, rule). The rule folder: <root>/<checkpoint SHA-256>/
    when the rule is that of the checkpoint's first logged read (first_rule_sha256) or the log holds no read of the
    checkpoint (None), else <root>/<checkpoint SHA-256>/rule_<rule SHA-256[:12]>/. A first read under the rule writes
    the rule folder, a reread after n_prior logged reads of the same checkpoint and rule writes
    <rule folder>/reread_<n_prior>/. Two rules never share a folder, and neither do two reads under one rule."""
    base = os.path.join(root, checkpoint_sha256)
    if first_rule_sha256 not in (None, rule_sha256):
        base = os.path.join(base, f"rule_{rule_sha256[:12]}")
    return os.path.join(base, f"reread_{n_prior}") if reread else base


def folder_rule(folder):
    """The rule SHA-256 that <folder>/_provenance.json records (a read's outputs), or None when it has none."""
    try:
        with open(os.path.join(folder, PROVENANCE), encoding="utf-8") as f:
            r = (json.loads(f.read()).get("rule") or {}).get("sha256")
    except (OSError, ValueError, AttributeError):
        return None
    return r if isinstance(r, str) and r else None


def make_folder(out, stamp):
    """Create `out` exclusively (its parents on demand); when it exists by then, <out>_<stamp> (then _2, _3, ...).
    Returns (folder, the fallback reason or None). Never writes into an existing folder."""
    from mdfm import paths
    paths.assert_not_moveatlas(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for k, c in enumerate([out, f"{out}_{stamp}"] + [f"{out}_{stamp}_{i}" for i in range(2, 100)]):
        try:
            os.mkdir(c)
        except FileExistsError:
            continue
        return c, (None if k == 0 else f"{out} existed when the read ended; the outputs went to {c}")
    raise FileExistsError(f"no free output folder next to {out}")


def write_new(path, text):
    """Write UTF-8 text with LF line ends in exclusive-create mode (FileExistsError when the file exists: never an
    overwrite), after the em-dash check and the MoveAtlas guard. Returns the file's SHA-256."""
    from mdfm import paths
    from mdfm.gates import common
    common.assert_no_em_dash(text, os.path.basename(path))
    paths.assert_not_moveatlas(path)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return common.sha256_file(path)


def stamp_of(read_at):
    """A folder suffix from an ISO time: '2026-10-12T10:15:00' -> '20261012T101500'."""
    return "".join(ch for ch in str(read_at) if ch.isalnum()) or time.strftime("%Y%m%dT%H%M%S")


# ---------------------------------------------------------------- read log
def checkpoint_reads(rows, gate, checkpoint_sha256, rule_sha256):
    """From the lockbox read log rows (the log only, no lockbox data): (the rows of this gate, checkpoint and rule,
    the number of rows of this gate and checkpoint under another rule, the rule SHA-256 of the checkpoint's first
    logged read or None)."""
    mine = [r for r in rows if (r["gate"], r["checkpoint_sha256"]) == (gate, checkpoint_sha256)]
    same = [r for r in mine if r["rule_sha256"] == rule_sha256]
    return same, len(mine) - len(same), (mine[0]["rule_sha256"] if mine else None)


def logged_row(common, n_before, gate, checkpoint_sha256):
    """The row the read appended to the log (the last row of this gate and checkpoint after the first n_before rows,
    whatever its rule: the caller compares the rule), or None."""
    rows = common.read_log()[n_before:]
    new = [r for r in rows if (r["gate"], r["checkpoint_sha256"]) == (gate, checkpoint_sha256)]
    return new[-1] if new else None


# ---------------------------------------------------------------- tables of a read
def set_lines(summ, name):
    """{predictor: {line: the line's stats on set `name`, without per-person values and trial lists}}."""
    out = {}
    for pred, lines in summ.items():
        out[pred] = {}
        for line, st in lines.items():
            s = (st.get("sets") or {}).get(name)
            if s is not None:
                out[pred][line] = {k: v for k, v in s.items() if k not in ("per_person", "trials")}
    return out


def per_person_frame(summ, set_names):
    import pandas as pd
    rows = []
    for name in set_names:
        for pred, lines in summ.items():
            for line, st in lines.items():
                s = (st.get("sets") or {}).get(name)
                for pid, v in sorted(((s or {}).get("per_person") or {}).items()):
                    rows.append((name, pred, line, pid, fin(v)))
    return pd.DataFrame(rows, columns=["set", "predictor", "line", "person_id", "value"])


def offsets_frame(summ, references):
    """The offsets the read subtracted (fitted on the dev controls), from the IC stats of each predictor."""
    import pandas as pd
    rows = []
    for pred, lines in summ.items():
        for ref in references:
            st = lines.get(f"ic_median_ms_{ref}")
            if st is not None:
                rows.append((pred, ref, fin(st.get("offset_ms")), bool(st.get("offset_fitted"))))
    return pd.DataFrame(rows, columns=["predictor", "reference", "offset_ms", "offset_fitted"])


def results_json(g3, read, loaded_info, rule, row, inference, head, timing, train_persons_sha256):
    """The g3.json document of a read (module docstring)."""
    results, summ, _ = read
    names = (g3.SET_VALID, g3.SET_ALL)
    doc = {
        "gate": g3.GATE, "run": "lockbox_read",
        "what": "sealed G3 lockbox read of the weargait_pd lockbox controls (docs/rules/G3_RULE.md section 9)",
        "checkpoint": loaded_info, "checkpoint_train_persons_sha256": train_persons_sha256,
        "rule": rule, "read": row, "inference": inference,
        "sets": {g3.SET_VALID: {"role": "primary row: the trials of valid records (rule section 2); decides",
                                "trials": sorted(read.sets[g3.SET_VALID])},
                 g3.SET_ALL: {"role": "all-trials row: every trial read; reported beside, decides nothing",
                              "trials": sorted(read.sets[g3.SET_ALL])}},
        "input_validity": read.report,
        "results": [r.to_dict() for r in results],
        "deciding": [{"set": r.extra.get("set"), "line": r.line, "verdict": r.verdict} for r in g3.deciding(results)],
        "lines": {name: set_lines(summ, name) for name in names},
        "offsets": offsets_frame(summ, g3.REFERENCES).to_dict("records"),
        "constants": {"stride_match_s": g3.STRIDE_MATCH_S, "ic_match_s": g3.IC_MATCH_S, "ic_margin": g3.IC_MARGIN,
                      "matched_share_floor": g3.MATCHED_SHARE_FLOOR, "seed": g3.SEED, "boot_n": g3.BOOT_N,
                      "primary_ic_reference": g3.PRIMARY_REFERENCE, "lines": g3.LINES},
        "moveatlas_head": head, "timing_s": timing,
    }
    return json.dumps(clean(doc), indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def markdown(g3, read, loaded_info, rule, row, inference, head, folder_note):
    """g3.md of a read: every number is one of g3.json, per_person.csv, offsets.csv and validity.csv."""
    results, summ, _ = read
    rep = read.report
    v = read.validity
    ck = loaded_info
    L = ["# G3 lockbox read (docs/rules/G3_RULE.md section 9)", "",
         "Written by scripts/read_g3_lockbox.py from mdfm.gates.g3.read_lockbox; every number below is read from "
         "g3.json, per_person.csv, offsets.csv and validity.csv of this folder. A sealed read of the weargait_pd "
         f"lockbox controls, logged in reports/gates/lockbox_reads.csv at {(row or {}).get('read_at', 'n/a')} "
         f"(reread: {'yes' if (row or {}).get('reread') == '1' else 'no'}; reason: "
         f"{(row or {}).get('reason', 'n/a')}).", "",
         f"Checkpoint: {ck.get('path')} (SHA-256 {ck.get('sha256')}), {ck.get('weights')} weights, phase "
         f"{ck.get('phase')} step {ck.get('step')}. Rule docs/rules/G3_RULE.md: status '{rule['status']}', SHA-256 "
         f"{rule['sha256']} (registered). Code SHA-256 of the read: {(row or {}).get('code_sha256', 'n/a')}. "
         f"MoveAtlas HEAD {head or 'n/a'}.", "",
         f"Inference: 13 IMUs (acc and gyro, magnetometer masked), {inference['window_s']:g} s windows every "
         f"{inference['hop_s']:g} s, {inference['refine_steps']} refinement steps of {inference['points_per_stream']} "
         f"points per stream, seed {inference['seed']}, shape pooling {inference['shape_pooling']}, "
         f"{inference['batch_windows']} batch windows, device {inference['device']}. Offsets fitted on the dev "
         "controls by the dev run inside the read (offsets.csv).", ""]
    if folder_note:
        L += [f"Folder: {folder_note}.", ""]
    persons = sorted(set(v["person_id"])) if len(v) else []
    L += ["## Sets and input validity", "",
          f"- Records read: {rep['n_records']} of {rep['n_trials']} trials, {len(persons)} persons with a record.",
          f"- {g3.SET_VALID} (the primary rows, which decide): {len(read.sets[g3.SET_VALID])} trials of valid records.",
          f"- {g3.SET_ALL} (reported beside, decides nothing): {len(read.sets[g3.SET_ALL])} trials.",
          f"- Invalid records: {rep['n_invalid']}" + (": " + "; ".join(f"{x['trial_id']} ({x['reasons']})"
                                                          for x in rep["invalid"]) if rep["invalid"] else "") + ".",
          f"- Trials without a record (out of the primary row): {', '.join(rep['trials_without_record']) or 'none'}.",
          f"- Persons left without a valid trial: {', '.join(rep['persons_without_valid_trial']) or 'none'}.", "",
          "## GateResult rows", "",
          "| set | line | unit | pass | kill | model | 95 % interval | persons | matched share | floor ok | trivial "
          "baseline | margin model / detector (ms) | verdict | decides |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        e = r.extra
        mg = (f"{fmt(e.get('margin_model_ms'))} / {fmt(e.get('margin_detector_ms'))}"
              if "margin_model_ms" in e else "")
        L.append(f"| {e.get('set')} | {r.line} | {r.unit} | {fmt(r.pass_line)} | {fmt(r.kill_line)} | {fmt(r.point)} "
                 f"| {fmt_iv(r.interval)} | {r.n_persons} | {fmt(e.get('matched_share'), 3)} | "
                 f"{yes_no(e.get('floor_ok')) if 'floor_ok' in e else ''} | {fmt(r.trivial_baseline)} | {mg} | "
                 f"{r.verdict} | {yes_no(e.get('decides', True))} |")
    dec = g3.deciding(results)
    roles = sorted({f"{r.line}: {r.extra['role']}" for r in results if r.extra.get("role")})
    L += ["", "Rows that decide (g3.deciding): " + "; ".join(f"{r.line} {r.verdict}" for r in dec) + ".",
          f"The rows of {g3.SET_ALL} and the insole IC row decide nothing (rule sections 2 and 6). Roles: "
          + "; ".join(roles) + ". The matched-share floor "
          f"({g3.MATCHED_SHARE_FLOOR}) conditions the pass of the stride, speed and MAPE rows (rule section 6); the "
          f"IC rows carry the margin ({g3.IC_MARGIN} times the shank-gyroscope detector on the ICs both match). This "
          "script computes no aggregate verdict.", ""]
    for name in (g3.SET_VALID, g3.SET_ALL):
        lines = set_lines(summ, name)
        L += [f"## Every predictor, set {name}", "",
              "| predictor | line | point | 95 % interval | matched share | reference events | persons | offset (ms) |",
              "|---|---|---|---|---|---|---|---|"]
        for pred in PREDICTORS:
            for line in LINE_ORDER:
                st = lines.get(pred, {}).get(line)
                if st is None:
                    continue
                L.append(f"| {pred} | {line} | {fmt(st['point'])} | {fmt_iv(st['interval'])} | "
                         f"{fmt(st['matched_share'], 3)} | {st['n_events']} | {st['n_persons']} | "
                         f"{fmt(st.get('offset_ms'), 1) if 'offset_ms' in st else ''} |")
        L.append("")
    off = offsets_frame(summ, g3.REFERENCES)
    L += ["Offsets subtracted in the read (fitted on the dev controls by the dev run of the read): " + "; ".join(
        f"{r.predictor} against {r.reference} {fmt(r.offset_ms, 1)} ms" + ("" if r.offset_fitted else " (not fitted)")
        for r in off.itertuples()) + ".", "",
        "Files: g3.json (the GateResult rows, the deciding rows, the input-validity report, the stats per set, the "
        "offsets, the checkpoint, the rule, the read row, the inference settings), per_person.csv, offsets.csv, "
        "validity.csv (rule section 2), _provenance.json (the SHA-256 of each file).", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if not str(a.reason).strip():
        print("refused: --reason must be non-empty (it is logged with the read)")
        return EXIT_REFUSED
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t_start = time.time()
    import mdfm  # noqa: F401  (sets the torch cache folder before torch is imported)
    from mdfm import paths
    from mdfm.data.canonical import Refused
    from mdfm.gates import common, g3

    # 2. the rule (read_lockbox checks it again, first)
    try:
        rule_text, rule_sha = common.rule(g3.RULE_PATH, gate=g3.GATE)
    except Refused as e:
        print(f"refused: {e}")
        return EXIT_REFUSED
    rule = {"path": common.rel(g3.RULE_PATH), "sha256": rule_sha, "status": common.rule_status(rule_text)[0],
            "registered": True}

    # 3. the checkpoint
    import torch

    from mdfm.eval import infer as I
    from mdfm.train import loop as LP
    from mdfm.train import provenance as PV
    if not os.path.isfile(a.checkpoint):
        print(f"refused: no checkpoint file {a.checkpoint}")
        return EXIT_REFUSED
    dev = torch.device(a.device)
    cap = LP.cap_memory(dev, a.memory_gb)
    try:
        loaded = I.load_checkpoint(a.checkpoint, weights=a.weights, device=dev)
    except (Refused, PV.ProvenanceError) as e:
        print(f"refused: the checkpoint does not load: {e}")
        return EXIT_REFUSED
    ck_sha = str(loaded.sha256)
    if not g3.HEX64.fullmatch(ck_sha):
        print(f"refused: checkpoint SHA-256 {ck_sha!r} is not a lowercase hex SHA-256")
        return EXIT_REFUSED
    trained = set(loaded.trained_persons())
    bad = sorted(trained & set(g3.lockbox_persons())), sorted(trained & set(g3.controls("dev")))
    if bad[0] or bad[1]:
        print(f"refused: the checkpoint trained on {len(bad[0])} weargait_pd lockbox persons and {len(bad[1])} G3 dev "
              "controls")
        return EXIT_REFUSED

    # 4. the read log (a read is keyed on the checkpoint and the rule, as in read_lockbox) and the output folder
    root = os.path.abspath(a.out_root)
    paths.assert_not_moveatlas(root)
    same, n_other, first_rule = checkpoint_reads(common.read_log(), g3.GATE, ck_sha, rule_sha)
    n_prior = len(same)
    held = (f"the log holds {n_prior} read(s) of checkpoint {ck_sha[:12]} under rule {rule_sha[:12]} and {n_other} "
            "under other rules")
    if a.reread and not n_prior:
        print(f"refused: --reread asks for a second read, and {held}; a first read under this rule runs without "
              "--reread")
        return EXIT_REFUSED
    if n_prior and not a.reread:
        print(f"refused: {held} (first on {same[0]['read_at']}); a second read under this rule needs --reread and "
              "a reason")
        return EXIT_REFUSED
    out = out_folder(root, ck_sha, rule_sha, first_rule, a.reread, n_prior)
    if os.path.exists(out):
        owner = folder_rule(out)
        whose = ("it holds no _provenance.json of a read" if owner is None else
                 f"its _provenance.json is of a read under {'this rule' if owner == rule_sha else 'another rule'} "
                 f"({owner[:12]})")
        print(f"refused: {out} exists ({whose}); a lockbox read never writes into an existing folder; {held}")
        return EXIT_REFUSED

    # 5. the read
    inference = {"window_s": I.WIN_S, "hop_s": I.HOP_S, "refine_steps": I.STEPS_EVAL,
                 "points_per_stream": I.POINTS_EVAL,
                 "seed": g3.SEED, "shape_pooling": "none" if a.no_pool else "per trial", "weights": a.weights,
                 "batch_windows": a.batch_windows, "device": str(dev), "amp_bf16": dev.type == "cuda",
                 "memory_cap_gb": cap}
    pred = I.G3Predictor(loaded, refine_steps=I.STEPS_EVAL, points=I.POINTS_EVAL, batch_windows=a.batch_windows,
                         seed=g3.SEED, pool=not a.no_pool)
    code_files = (I.__file__, os.path.abspath(__file__))
    n_before = len(common.read_log())
    print(f"checkpoint {loaded.name} ({ck_sha[:12]}), rule {rule_sha[:12]}; reading the G3 lockbox"
          + (f" (reread after {n_prior})" if a.reread else ""), flush=True)
    try:
        read = g3.read_lockbox(ck_sha, pred, reread=a.reread, reason=a.reason, code_files=code_files)
    except Exception as e:  # noqa: BLE001
        row = logged_row(common, n_before, g3.GATE, ck_sha)
        if row is None:
            print(f"{'refused' if isinstance(e, Refused) else 'failed'} before the read was logged: "
                  f"{type(e).__name__}: {e}")
            if not isinstance(e, Refused):
                traceback.print_exc()
            return EXIT_REFUSED if isinstance(e, Refused) else EXIT_FAILED
        print(f"failed after the read was logged ({row['read_at']}, rule {row['rule_sha256'][:12]}): "
              f"{type(e).__name__}: {e}; no output written; another read of this checkpoint under this rule needs "
              "--reread with a reason")
        traceback.print_exc()
        return EXIT_FAILED
    row = logged_row(common, n_before, g3.GATE, ck_sha)
    texts, errors = {}, {}
    if row is None or row["rule_sha256"] != rule_sha:     # read_lockbox logs before it returns, under its own check
        where = "missing" if row is None else f"under rule {row['rule_sha256'][:12]}"
        errors["read"] = (f"the log row of the read is {where}, not under rule {rule_sha[:12]}: the rule changed "
                          "between this script's check and read_lockbox's; the folder and the rule named in g3.md are "
                          "those of the script's check, the GateResult rows carry the rule of the read")
    timing = {"all": round(time.time() - t_start, 1)}
    if dev.type == "cuda":
        inference["memory"] = {"max_allocated_gb": torch.cuda.max_memory_allocated(dev) / 1e9,
                               "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / 1e9}
    inference["trials_run"] = len(getattr(pred, "log", []) or [])

    # 6. the outputs: rendered first, then written in exclusive-create mode, the provenance last
    head = PV.moveatlas_head()
    info = loaded.describe()
    renderers = {
        "validity.csv": lambda: read.validity.to_csv(index=False, lineterminator="\n"),
        "g3.json": lambda: results_json(g3, read, info, rule, row, inference, head, timing,
                                        I.train_persons_sha256(loaded)),
        "per_person.csv": lambda: per_person_frame(read[1], (g3.SET_VALID, g3.SET_ALL)).to_csv(
            index=False, lineterminator="\n"),
        "offsets.csv": lambda: offsets_frame(read[1], g3.REFERENCES).to_csv(index=False, lineterminator="\n"),
    }
    for name, render in renderers.items():
        try:
            texts[name] = render()
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    try:
        folder, note = make_folder(out, stamp_of((row or {}).get("read_at", "")))
    except Exception as e:  # noqa: BLE001
        print(f"failed after the read was logged ({(row or {}).get('read_at')}): no output folder: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return EXIT_FAILED
    try:
        texts["g3.md"] = markdown(g3, read, info, rule, row, inference, head, note)
    except Exception as e:  # noqa: BLE001
        errors["g3.md"] = f"{type(e).__name__}: {e}"
    written = {}
    for name in FILES:
        if name not in texts:
            continue
        try:
            written[name] = write_new(os.path.join(folder, name), texts[name])
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"
    prov = {"script": SCRIPT_REL, "script_sha256": common.sha256_file(os.path.abspath(__file__)),
            "started_at": started, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arguments": {"checkpoint": common.rel(a.checkpoint), "reason": a.reason, "reread": bool(a.reread),
                          "weights": a.weights, "no_pool": bool(a.no_pool), "batch_windows": a.batch_windows,
                          "device": a.device, "memory_gb": a.memory_gb, "out_root": common.rel(root)},
            "checkpoint_sha256": ck_sha, "rule": rule, "read": row,
            "log_before_read": {"same_rule": n_prior, "other_rules": n_other, "first_rule_sha256": first_rule},
            "code_files": [common.rel(p) for p in (g3.__file__, g3.gait.__file__, *code_files)],
            "inference": inference, "moveatlas_root": paths.MOVEATLAS, "moveatlas_head": head,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "folder": common.rel(folder), "folder_fallback": note, "files": written, "errors": errors}
    try:
        write_new(os.path.join(folder, PROVENANCE), json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        errors[PROVENANCE] = f"{type(e).__name__}: {e}"
        print(f"failed to write {PROVENANCE}: {errors[PROVENANCE]}")
    for r in g3.deciding(read[0]):
        print(f"{r.extra.get('set')} {r.line}: {fmt(r.point)} {r.unit} -> {r.verdict}")
    print(f"written {common.rel(folder)}: {', '.join(written) or 'nothing'}"
          + (f"; errors: {errors}" if errors else ""))
    return EXIT_FAILED if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
