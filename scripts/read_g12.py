"""Reading of the queued G12 baseline jobs (task G12Q; docs/rules/G12_RULE_REVISION_2026-10-08.md): the rule hash,
the GPU smoke and the training run of one arm and seed. Reads files only (CPU, no model, no store, no record).

- rule: reports/bench/g12/rule.sha256 against the SHA-256 of the file it names, recomputed: MATCH or MISMATCH;
  NOT EVALUATED while the hash file is absent.
- smoke: the JSON that scripts/train_g12.py --bench --smoke --out writes (default reports/bench/g12/smoke_<arm>.json):
  PASS or FAIL as its smoke block says, with per phase the seconds per optimizer step, the peak reserved memory
  against the bound and the allocator cap, the first and last losses, the non-finite and skipped counts, and the
  rate written into the configuration. While the file is absent: ENDED WITHOUT OUTPUT when a row of
  reports/gpu_queue/job_log.csv ran with that --out path (its state, exit code and end are reported: with the
  allocator cap equal to the bound, a peak above the bound ends the job out of memory before any JSON), NOT
  EVALUATED otherwise.
- train: the run folder's log.jsonl (default cache/runs/g12_<arm>_s<seed>): NOT EVALUATED while it is absent;
  STOPPED (a 'compute_overrun' or 'alarm' event, or a 'compute' record with finished false, in the last call);
  FINISHED (a 'compute' record with finished true: the realized clock, its ratio to the budget, the optimizer
  steps and tokens, and the 'g12_verify' event: checkpoint path, SHA-256, step, drift). With neither: IN PROGRESS
  (the last 'log' record: phase, step, clock so far and its ratio to the budget, loss) while a python process whose
  command line holds train_g12 and the run folder's name runs (psutil), or when that check is unavailable and no
  queue row exists; ENDED WITHOUT COMPUTE RECORD otherwise (a queue timeout, a crash, a resume still pending), with
  the queue rows of job_log.csv whose argv names the run folder and whether cache/runs/queue/state.json holds a
  pending resume for it. A finished seed is read whatever its ratio (rule section 5); the ratio is reported beside
  it.
- rates (once the run has a log): the seconds per optimizer step of the smoke's bench (the rate that fixed the
  step count; the 'start' record's plan carries it), the realized clock per optimizer step (the compute record's
  wall_s over its steps, every interval included) and their ratio, and per phase the realized rate of the 'log'
  records against the smoke's. The bench leaves out checkpoint writes and ran under another host state, so the
  ratio says how far the realized clock may land from the budget.

No G12 comparison is computed here: the comparison of rule section 8 needs the readouts and the body's numbers,
and the rule's G8 and G10 lines need their own rules first (revision section 3).

Writes, with --write: reports/bench/g12/read_<arm>_s<seed>.json and .md. Prints the JSON.

Usage (mdfm or base environment, from the tree root):
  python scripts/read_g12.py [--arm mfr] [--seed 20260928] [--write]
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(TREE, "reports", "bench", "g12")
RULE_HASH = os.path.join(OUT, "rule.sha256")
NOT_EVALUATED = "NOT EVALUATED"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                  # a line cut by a crash
    return out


def last_call(records):
    """The records of the last call of a run folder (from its last 'start' event on)."""
    idx = [i for i, r in enumerate(records) if r.get("event") == "start"]
    return records[idx[-1]:] if idx else list(records)


def _names(argv, target, tree):
    """True when an item of argv (a list, paths relative to the tree root) resolves to the path target."""
    want = os.path.normcase(os.path.abspath(target if os.path.isabs(target) else os.path.join(tree, target)))
    for x in argv or []:
        if isinstance(x, str) and x and not x.startswith("-"):
            p = x if os.path.isabs(x) else os.path.join(tree, x)
            if os.path.normcase(os.path.abspath(p)) == want:
                return True
    return False


def queue_rows(target, tree=TREE):
    """The rows of reports/gpu_queue/job_log.csv whose argv names the path target: [{name, state, exit_code, start,
    end, note}] ([] when the file is absent or cannot be read)."""
    path = os.path.join(tree, "reports", "gpu_queue", "job_log.csv")
    out = []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    argv = json.loads(row.get("argv") or "[]")
                except ValueError:
                    continue
                if _names(argv, target, tree):
                    out.append({k: row.get(k) for k in ("name", "state", "exit_code", "start", "end", "note")})
    except OSError:
        return []
    return out


def queue_resume_pending(target, tree=TREE):
    """True when cache/runs/queue/state.json holds a pending resume whose argv names the path target; None when the
    file is absent or unreadable."""
    path = os.path.join(tree, "cache", "runs", "queue", "state.json")
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        return None
    return any(_names((v or {}).get("argv"), target, tree) for v in (st.get("resume") or {}).values())


def live_cmdlines():
    """The command lines of the running processes (psutil), or None when psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    out = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info.get("cmdline")
        except Exception:  # noqa: BLE001  (a process that ended or is denied)
            continue
        if cl:
            out.append(" ".join(str(x) for x in cl))
    return out


def trains_on(cmdlines, run_dir):
    """True when a command line holds train_g12 and the run folder's name; None when cmdlines is None."""
    if cmdlines is None:
        return None
    name = os.path.basename(os.path.normpath(run_dir))
    return any("train_g12" in c and name in re.split(r"[\s/\\\"']+", c) for c in cmdlines)


def read_rule(path=RULE_HASH, tree=TREE):
    if not os.path.exists(path):
        return {"verdict": NOT_EVALUATED, "path": os.path.relpath(path, tree).replace(os.sep, "/")}
    with open(path, encoding="utf-8") as f:
        parts = f.read().split()
    sha, rel = parts[0], parts[1].lstrip("*")
    target = os.path.join(tree, *rel.split("/"))
    now = sha256_file(target) if os.path.exists(target) else None
    return {"verdict": "MATCH" if now == sha else "MISMATCH", "file": rel, "sha256": sha, "sha256_now": now}


def read_smoke(path, tree=TREE):
    rel = os.path.relpath(path, tree).replace(os.sep, "/")
    if not os.path.exists(path):
        rows = queue_rows(path, tree)
        if rows:
            return {"verdict": "ENDED WITHOUT OUTPUT", "path": rel, "queue_rows": rows}
        return {"verdict": NOT_EVALUATED, "path": rel}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sm = d.get("smoke") or {}
    phases = {}
    for ph, r in (d.get("phases") or {}).items():
        s = (sm.get("phases") or {}).get(ph, {})
        phases[ph] = {"seconds_per_step": r.get("seconds_per_step"), "peak_reserved_gb": r.get("peak_reserved_gb"),
                      "loss_first": s.get("loss_first"), "loss_last": s.get("loss_last"),
                      "nonfinite_steps": s.get("nonfinite_steps"), "nonfinite_micro": s.get("nonfinite_micro"),
                      "skipped": s.get("skipped"), "empty": s.get("empty"), "steps_recorded": s.get("steps_recorded")}
    verdict = NOT_EVALUATED if not sm else ("PASS" if sm.get("ok") else "FAIL")
    return {"verdict": verdict, "path": rel, "sha256": sha256_file(path), "arm": d.get("arm"), "time": d.get("time"),
            "gpu": d.get("gpu"), "seconds_per_step": d.get("seconds_per_step"), "bound_gb": sm.get("bound_gb"),
            "memory_cap_gb": d.get("memory_cap_gb"), "max_reserved_gb": d.get("max_reserved_gb"),
            "reasons": sm.get("reasons", []), "phases": phases, "written": d.get("written"),
            "config_sha256_after": d.get("config_sha256_after"), "moveatlas_head": d.get("moveatlas_head"),
            "script_sha256": d.get("script_sha256")}


def _num(x):
    return x if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


def read_train(run_dir, tree=TREE, cmdlines="live"):
    """cmdlines: the running processes' command lines (a list), None for unknown, or 'live' to read them."""
    rel = os.path.relpath(run_dir, tree).replace(os.sep, "/")
    log = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log):
        return {"verdict": NOT_EVALUATED, "run_dir": rel}
    recs = read_jsonl(log)
    seg = last_call(recs)
    start = next((r for r in recs if r.get("event") == "start"), {})
    g12 = start.get("g12") or {}
    logs = [r for r in recs if r.get("event") == "log"]
    last = logs[-1] if logs else {}
    comp = next((r for r in reversed(seg) if r.get("event") == "compute"), None)
    stops = [r for r in seg if r.get("event") in ("compute_overrun", "alarm", "alarm_stop")]
    verify = next((r for r in reversed(seg) if r.get("event") == "g12_verify"), None)
    out = {"run_dir": rel, "log_sha256": sha256_file(log), "records": len(recs), "calls": sum(
        1 for r in recs if r.get("event") == "start"), "arm": g12.get("arm"), "config_sha256": g12.get("config_sha256"),
        "planned_steps_A_B": g12.get("steps"), "plan": g12.get("plan"),
        "last_log": {k: _num(last.get(k)) if k != "phase" else last.get(k) for k in ("phase", "step", "loss")} if last
        else None,
        "compute_so_far": (last.get("compute") if last else None)}
    acc = {}
    for r in logs:
        w, n = _num(r.get("wall_s")), r.get("optimizer_steps")
        if w is not None and isinstance(n, int) and n > 0:
            a = acc.setdefault(r.get("phase"), [0.0, 0])
            a[0] += w
            a[1] += n
    out["realized_phases"] = {ph: {"wall_s": w, "steps": n, "seconds_per_step": w / n} for ph, (w, n) in acc.items()}
    if stops:
        out["verdict"] = "STOPPED"
        out["stops"] = [{k: r.get(k) for k in ("event", "phase", "step", "ratio", "wall_s")} for r in stops]
    elif comp is not None and comp.get("finished"):
        out["verdict"] = "FINISHED"
    elif comp is not None:
        out["verdict"] = "STOPPED"
    else:
        live = trains_on(live_cmdlines() if cmdlines == "live" else cmdlines, run_dir)
        rows = queue_rows(run_dir, tree)
        out["process_running"] = live
        out["queue_rows"] = rows
        out["queue_resume_pending"] = queue_resume_pending(run_dir, tree)
        out["verdict"] = "IN PROGRESS" if live or (live is None and not rows) else "ENDED WITHOUT COMPUTE RECORD"
    if comp is not None:
        out["compute"] = {k: comp.get(k) for k in ("mode", "wall_s", "budget_wall_s", "ratio", "tolerance", "steps",
                                                  "planned_steps", "tokens", "finished", "within_tolerance")}
        if _num(comp.get("wall_s")) is not None:
            out["compute"]["hours"] = comp["wall_s"] / 3600.0
    if verify is not None:
        out["verify"] = {k: verify.get(k) for k in ("path", "sha256", "step", "phase", "drift", "persons")}
    return out


def _ratio(x, y):
    x, y = _num(x), _num(y)
    return x / y if x is not None and y else None


def rates(train, smoke):
    """The realized seconds per optimizer step against the smoke's bench (module docstring, rates)."""
    if train["verdict"] == NOT_EVALUATED:
        return {"verdict": NOT_EVALUATED}
    sm = smoke.get("seconds_per_step") if smoke["verdict"] in ("PASS", "FAIL") else None
    c = train.get("compute") or train.get("compute_so_far") or {}
    real = _ratio(c.get("wall_s"), c.get("steps")) if isinstance(c.get("steps"), int) and c.get("steps") > 0 else None
    phases = {}
    for ph, r in sorted((train.get("realized_phases") or {}).items()):
        s = ((smoke.get("phases") or {}).get(ph) or {}).get("seconds_per_step")
        phases[ph] = {"realized": r["seconds_per_step"], "smoke": s, "realized_over_smoke": _ratio(r["seconds_per_step"], s)}
    return {"smoke_seconds_per_step": sm, "planned_seconds_per_step": (train.get("plan") or {}).get("seconds_per_step"),
            "realized_seconds_per_step": real, "realized_over_smoke": _ratio(real, sm), "phases": phases,
            "final": bool(train.get("compute"))}


def read(arm="mfr", seed=20260928, tree=TREE, smoke=None, run_dir=None, rule=None, cmdlines="live"):
    smoke = smoke or os.path.join(tree, "reports", "bench", "g12", f"smoke_{arm}.json")
    run_dir = run_dir or os.path.join(tree, "cache", "runs", f"g12_{arm}_s{seed}")
    rule = rule or os.path.join(tree, "reports", "bench", "g12", "rule.sha256")
    sys.path.insert(0, tree)
    try:
        from mdfm.train import provenance as PV
        head = PV.moveatlas_head()
    except Exception:  # noqa: BLE001  (the base environment may lack torch; the head is then left out)
        head = None
    sm, tr = read_smoke(smoke, tree), read_train(run_dir, tree, cmdlines)
    return {"arm": arm, "seed": int(seed), "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "script_sha256": sha256_file(os.path.abspath(__file__)), "moveatlas_head_now": head,
            "rule": read_rule(rule, tree), "smoke": sm, "train": tr, "rates": rates(tr, sm),
            "comparison": NOT_EVALUATED + " (rule section 8 needs the readouts and the body's numbers)"}


def _f(x, fmt):
    return format(x, fmt) if isinstance(x, (int, float)) else "n/a"


def markdown(res):
    L = [f"# G12 reading: arm {res['arm']}, seed {res['seed']}", "",
         f"Written {res['time']} by scripts/read_g12.py (SHA-256 {res['script_sha256']}); MoveAtlas HEAD at the read "
         f"{res['moveatlas_head_now'] or 'not read'}.", ""]
    r = res["rule"]
    L += ["## Rule", "", f"- {r['verdict']}" + (f": {r['file']} SHA-256 {r['sha256']}" if "file" in r else ""), ""]
    s = res["smoke"]
    L += ["## Smoke", "", f"- {s['verdict']} ({s['path']})"]
    for q in s.get("queue_rows") or []:
        L.append(f"- queue row: {q}")
    if s["verdict"] in ("PASS", "FAIL"):
        L.append(f"- GPU {s.get('gpu')}; seconds per optimizer step {_f(s.get('seconds_per_step'), '.4f')} (share_a "
                 f"weighted); peak bound {s.get('bound_gb')} GB, allocator cap {s.get('memory_cap_gb')} GB; written "
                 f"{s.get('written')}")
        if _num(s.get("memory_cap_gb")) is not None and _num(s.get("bound_gb")) is not None and \
                s["memory_cap_gb"] <= s["bound_gb"]:
            L.append("- the allocator cap is at most the bound, so the memory criterion cannot fail by itself: a "
                     "larger demand ends the job out of memory before any JSON (smoke ENDED WITHOUT OUTPUT)")
        for ph, p in sorted(s["phases"].items()):
            L.append(f"- phase {ph}: {_f(p['seconds_per_step'], '.4f')} s per step, peak reserved "
                     f"{_f(p['peak_reserved_gb'], '.3f')} GB (bound {s.get('bound_gb')}, cap {s.get('memory_cap_gb')}), "
                     f"loss {_f(p['loss_first'], '.4f')} to {_f(p['loss_last'], '.4f')}, non-finite steps "
                     f"{p['nonfinite_steps']}, skipped {p['skipped']}, empty {p['empty']}")
        for x in s.get("reasons") or []:
            L.append(f"- reason: {x}")
    L.append("")
    t = res["train"]
    L += ["## Training", "", f"- {t['verdict']} ({t['run_dir']})"]
    if "process_running" in t:
        L.append(f"- train_g12 process on the run folder: {t['process_running']}; pending queue resume: "
                 f"{t.get('queue_resume_pending')}")
        for q in t.get("queue_rows") or []:
            L.append(f"- queue row: {q}")
    if t["verdict"] != NOT_EVALUATED:
        L.append(f"- planned steps A, B {t.get('planned_steps_A_B')}; calls {t.get('calls')}; last log "
                 f"{t.get('last_log')}")
        c = t.get("compute") or t.get("compute_so_far") or {}
        L.append(f"- clock {_f(c.get('wall_s'), '.0f')} s against the budget {_f(c.get('budget_wall_s'), '.0f')} s, "
                 f"ratio {_f(c.get('ratio'), '.4f')}, optimizer steps {c.get('steps')}, tokens {c.get('tokens')}")
        if t.get("verify"):
            v = t["verify"]
            L.append(f"- verified checkpoint {v.get('path')} SHA-256 {v.get('sha256')} (step {v.get('step')}, drift "
                     f"{v.get('drift')})")
        for x in t.get("stops") or []:
            L.append(f"- stop: {x}")
    rt = res.get("rates") or {}
    if rt.get("verdict") != NOT_EVALUATED and rt:
        L += ["", "## Rates", "",
              f"- seconds per optimizer step: smoke bench {_f(rt.get('smoke_seconds_per_step'), '.4f')}, plan "
              f"{_f(rt.get('planned_seconds_per_step'), '.4f')}, realized clock per step "
              f"{_f(rt.get('realized_seconds_per_step'), '.4f')} ({'final' if rt.get('final') else 'so far'}); "
              f"realized over smoke {_f(rt.get('realized_over_smoke'), '.4f')}"]
        for ph, p in sorted((rt.get("phases") or {}).items()):
            L.append(f"- phase {ph}: realized {_f(p['realized'], '.4f')} s per step (log records), smoke "
                     f"{_f(p['smoke'], '.4f')}, ratio {_f(p['realized_over_smoke'], '.4f')}")
    L += ["", "## Comparison", "", f"- {res['comparison']}", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="mfr", choices=("mfr", "jepa"))
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    res = read(a.arm, a.seed)
    if a.write:
        os.makedirs(OUT, exist_ok=True)
        stem = os.path.join(OUT, f"read_{a.arm}_s{a.seed}")
        with open(stem + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1, sort_keys=True, default=str)
            f.write("\n")
        with open(stem + ".md", "w", encoding="utf-8", newline="\n") as f:
            f.write(markdown(res))
    print(json.dumps(res, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
