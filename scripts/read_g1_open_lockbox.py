"""G1 on the open sets, the sealed lockbox read of one declared checkpoint (docs/rules/G1_RULE_OPEN_2026-11-13.md,
registered as G1 version 2 on 2026-11-14; harness mdfm/gates/g1_open.py read_lockbox). A run logs a lockbox read in
reports/gates/lockbox_reads.csv: it is the owner's act (plan docs/MODEL_OPEN_PORT_PLAN.md, step M7). A session never
runs it.

Order of a run (the script never opens a lockbox row or record; read_lockbox is the only code that does, after the
log):
1. --declaration (a read declaration, format g1_open.DECLARATION_FORMAT, its fields g1_open.DECLARATION_KEYS) and a
   non-empty --reason are required; --reread asks for a second read of the same checkpoint and rule.
2. The rule the declaration chooses (g1_open.declared_rule): refused (exit 2) unless the corpus holds no AMASS and the
   open rule, the 2026-11-12 version and the draft it adopts are registered or found at their SHA-256; before any
   checkpoint loads.
3. The output folder reports/gates/g1/open_lockbox/<checkpoint SHA-256>/ (a reread: <that folder>/reread_<k>/, k
   the reads of the checkpoint and rule the log holds): refused (exit 2) when it exists or when the read log does not
   allow the read (a second read without --reread, a reread without a logged read).
4. The checkpoint (mdfm.eval.infer.load_checkpoint: the EMA weights; --records-state for a checkpoint born on an
   archived records state) as infer.G1Predictor under the mode ema_s20_nopool (20 refinement steps of 256 points per
   stream, no shape pooling, batches of 16 windows, 16 latent samples x 4 decoder draws; seed --seed, default
   20260929, the default of scripts/eval_g1_dev.py, recorded); the trivial baseline, the population prior of the
   open train store (g1_open.open_population_prior on mdfm.data.open_body_store.OpenTrainStore()).
5. read_lockbox: its checks, the conformal factors and the dev rows on the halves, the logged read, the lockbox
   fits and inputs in the gates-only store, the lines.
6. The outputs, rendered in memory, then written in exclusive-create mode: g1.json (the lines of the model and of
   the prior on the lockbox, the dev rows of both, the secondary rows, the fit report, the plan of the read and its
   SHA-256, the read row), per_source.csv, per_person.csv, g1.md, and _provenance.json last (the SHA-256 of each
   file, the checkpoint, the MD_FM HEAD).

Exit codes: 0 written; 1 failed (after the log the read is spent and the message says so); 2 refused.

Run (owner, mdfm environment, after the session has written the declaration):
  PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/read_g1_open_lockbox.py
      --declaration <declaration.json> --reason "<why this checkpoint is read>"
"""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import traceback

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

SCRIPT_REL = "scripts/read_g1_open_lockbox.py"
DEFAULT_ROOT = os.path.join(TREE, "reports", "gates", "g1", "open_lockbox")
FILES = ("g1.json", "per_source.csv", "per_person.csv", "g1.md")      # in write order, _provenance.json last
PROVENANCE = "_provenance.json"
SEED = 20260929


def head():
    r = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def out_dir(base, checkpoint_sha256, rule_sha256, reread, log_path):
    """The folder of this read (module docstring, step 3) and the number of logged reads of the checkpoint and rule;
    SystemExit(2) when the log does not allow the read or the folder exists."""
    from mdfm.gates import common
    n = sum((r["gate"], r["checkpoint_sha256"], r["rule_sha256"]) == ("G1", checkpoint_sha256, rule_sha256)
            for r in common.read_log(log_path))
    if n and not reread:
        raise SystemExit(f"refused: checkpoint {checkpoint_sha256[:12]} was read under rule {rule_sha256[:12]}; "
                         "a second read needs --reread")
    if reread and not n:
        raise SystemExit("refused: --reread without a logged read of this checkpoint and rule")
    folder = os.path.join(base, checkpoint_sha256)
    if reread:
        folder = os.path.join(folder, f"reread_{n}")
    if os.path.exists(folder):
        raise SystemExit(f"refused: {folder} exists; a read never writes into an existing folder")
    return folder, n


def _num(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def render(read, provenance_extra=None):
    """{file name: text} of a read's outputs (module docstring, step 6), _provenance.json excluded."""
    from mdfm.gates import common
    rows = lambda rs: [{k: _num(v) for k, v in r.to_dict().items()} for r in rs]  # noqa: E731
    doc = {"script": SCRIPT_REL, "gate": "G1", "rule": read.rule, "declaration_sha256": read.declaration_sha256,
           "read_row": read.read_row, "plan_sha256": read.plan_sha256, "plan": read.plan,
           "lines": rows(read.results), "prior_lines": rows(read.prior_results), "dev_lines": rows(read.dev_results),
           "dev_prior_lines": rows(read.dev_prior_results), "secondary": read.secondary, "fits": read.fits,
           "per_source": read.per_source.to_dict("records"), "prior_per_source": read.prior_per_source.to_dict("records"),
           "dev_per_source": read.extra["dev_per_source"].to_dict("records"),
           "window_counts": read.extra["window_counts"], "manifest_sha256":
           read.manifest["manifest_sha256"], **(provenance_extra or {})}
    out = {"g1.json": json.dumps(doc, indent=1, default=common._default, allow_nan=True) + "\n",
           "per_source.csv": read.per_source.to_csv(index=False, lineterminator="\n"),
           "per_person.csv": read.per_person.to_csv(index=False, lineterminator="\n")}
    L = ["# G1 on the open sets: the lockbox read", "",
         f"Rule {read.rule['path']} (SHA-256 {read.rule['sha256'][:12]}), declaration SHA-256 "
         f"{read.declaration_sha256[:12]}, checkpoint {read.read_row['checkpoint_sha256'][:12]}, read logged at "
         f"{read.read_row['read_at']}{' (reread)' if read.read_row['reread'] == '1' else ''}. Plan SHA-256 "
         f"{read.plan_sha256[:12]} (the logged input hash).", "",
         "| set | predictor | line | point | 95% interval | pass | kill | verdict |", "|---|---|---|---|---|---|---|---|"]
    for name, who, rs in (("lockbox", "model", read.results), ("lockbox", "population prior", read.prior_results),
                          ("development half", "model", read.dev_results),
                          ("development half", "population prior", read.dev_prior_results)):
        for r in rs:
            nd = 3 if r.unit == "" else 1
            L.append(f"| {name} | {who} | {r.line} | {r.point:.{nd}f} {r.unit} | {r.interval[0]:.{nd}f} to "
                     f"{r.interval[1]:.{nd}f} | {r.pass_line} | {r.kill_line} | {r.verdict} |")
    L += ["", "| source | persons | windows | G1a mm | G1b coverage | G1c mm |", "|---|---|---|---|---|---|"]
    for r in read.per_source.to_dict("records"):
        L.append(f"| {r['archive']} | {r['n_persons']} | {r['n_windows']} | {r['g1a_mpjpe_mm']:.1f} | "
                 f"{r['g1b_coverage']:.3f} | {r['g1c_pa_mpjpe_mm']:.1f} |")
    L += ["", "| source | lockbox persons | takes fitted | over the bound | fit errors | short takes | unscored |",
          "|---|---|---|---|---|---|---|"]
    for s, f in read.fits.items():
        L.append(f"| {s} | {f['persons']} | {f['takes_fitted']} | {f['over_bound']} | {f['fit_errors']} | "
                 f"{f['dropped_short']} | {', '.join(f['unscored']) or 'none'} |")
    L += ["", "Every number above comes from g1.json in this folder.", ""]
    out["g1.md"] = "\n".join(L)
    for name, text in out.items():
        common.assert_no_em_dash(text, name)
    return out


def write_outputs(folder, texts, provenance):
    """The rendered files in exclusive-create mode (FILES order), then _provenance.json with their SHA-256."""
    from mdfm.gates import common
    os.makedirs(folder)
    written = {}
    for name in FILES:
        p = os.path.join(folder, name)
        with open(p, "x", encoding="utf-8", newline="\n") as f:
            f.write(texts[name])
        written[name] = common.sha256_file(p)
    prov = dict(provenance, outputs=written)
    with open(os.path.join(folder, PROVENANCE), "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(prov, indent=1, default=common._default) + "\n")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--declaration", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--reread", action="store_true")
    p.add_argument("--records-state", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=DEFAULT_ROOT)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    from mdfm.data.canonical import Refused
    from mdfm.gates import common
    from mdfm.gates import g1_open as GO
    if not a.reason.strip():
        print("refused: --reason is empty", file=sys.stderr)
        return 2
    try:
        decl, decl_sha = GO.load_declaration(a.declaration)
        if GO.declaration_kind(decl) != "open":
            raise Refused("the declared corpus holds AMASS: the 2026-11-12 version, not the open path")
        rl = GO.declared_rule(decl)
        folder, n_prior = out_dir(a.out, decl["checkpoint"]["sha256"], rl["sha256"], a.reread, common.READ_LOG)
    except (Refused, SystemExit) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    from mdfm.data.open_body_store import OpenTrainStore
    from mdfm.eval import infer as I
    ck = decl["checkpoint"]["file"]
    loaded = I.load_checkpoint(ck if os.path.isabs(ck) else os.path.join(TREE, ck), weights="ema", device=a.device,
                               records_state=a.records_state)
    pred = I.G1Predictor(loaded, refine_steps=GO.MODE_SETTINGS["refine_steps"],
                         points=GO.MODE_SETTINGS["points_per_stream"], samples=GO.MODE_SETTINGS["latent_samples"],
                         decoder_draws=GO.MODE_SETTINGS["decoder_draws"], batch_windows=GO.MODE_BATCH_WINDOWS,
                         seed=a.seed, pool=False, configs=I.G1_CONFIGS)
    prior = GO.open_population_prior(OpenTrainStore())
    logged_before = len(common.read_log())
    try:
        read = GO.read_lockbox(a.declaration, pred, prior, reason=a.reason, reread=a.reread, log=print)
    except Refused as e:
        spent = len(common.read_log()) > logged_before
        print(f"{'failed after the read was logged' if spent else 'refused'}: {e}", file=sys.stderr)
        return 1 if spent else 2
    except Exception:  # noqa: BLE001
        spent = len(common.read_log()) > logged_before
        traceback.print_exc()
        print(f"failed{' after the read was logged: the read is spent' if spent else ''}", file=sys.stderr)
        return 1
    extra = {"checkpoint": loaded.describe(), "seed": a.seed, "reads_before": n_prior}
    try:
        texts = render(read, extra)
        if os.path.exists(folder):
            folder = f"{folder}_{read.read_row['read_at'].replace(':', '')}"
        write_outputs(folder, texts, {"script": SCRIPT_REL, "mdfm_head": head(), "written_at":
                                      datetime.datetime.now().isoformat(timespec="seconds"),
                                      "read_row": read.read_row, "declaration": common.rel(a.declaration),
                                      "declaration_sha256": decl_sha, **extra})
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("failed after the read was logged: outputs not written", file=sys.stderr)
        return 1
    print(texts["g1.md"])
    print(f"written: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
