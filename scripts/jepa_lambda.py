"""J2b: lambda of experiment E1 from the JEPA smoke run, with the smoke's checks and its speed
(docs/proposals/JEPA_EXPLORATION_DRAFT.md, sections 2.7, 2.8 and 6).

Reads the log.jsonl of the smoke's run folder (default cache/runs/jepa_smoke, written by scripts/train_stage1.py
--phase B --init-from cache/runs/phaseA_root_full/ckpt_A_010681.pt --config configs/stage1_pilot_jepa.yaml
--max-steps 200 --jepa-grad-steps 50 at jepa.weight 1.0) and its provenance.json, and, for the speed comparison,
the log.jsonl of reference runs (default v1, cache/runs/phaseA_root_full, and seed 2, cache/runs/pilot_v1_s2). Reads
no record, no checkpoint and no store; CPU only. The smoke's call: its log's records from the last 'start' event on.

lambda (proposal 2.7). Over the 'jepa_grad' events of phase B steps 0 to N - 1 (N = --steps, default 50): g_J, the
norm over the encoder's parameters of the gradient of the step's JEPA contribution at weight 1.0, and g_R, the same
for every other term, both before clipping (mdfm/train/loop.py). A step whose g_J or g_R is null (loop.jsonl writes
a non-finite float as null) or not finite is left out and counted; a step with g_J = 0 (no window carried the term)
is kept, as the rule is written, and counted. lambda = mean(g_R) / mean(g_J) over the steps kept (a ratio of means).
Every step from 0 to N - 1 must appear exactly once.

Checks of the smoke (every one must hold for --write-config and --e1-job): the 'start' event names the jepa section
enabled at weight 1.0 and grad_steps >= N; an 'init_from' event; none of the events alarm, nonfinite,
memory_exceeded, synthetic_cap; an 'end' event at phase B step --expect-steps (default 200) and a 'verify' event of
phase B; each phase B 'log' event with a finite loss, a finite jepa loss when windows carried the term, and a
collapse ratio, where read, at or above loop.ALARMS['jepa_collapse'] (0.01); the peak reserved memory (the largest
memory_gb.max_reserved of the call's 'log' and 'end' events) below --max-reserved-gb (default 14.0 GB); lambda
computed from all N steps (none missing or repeated) with at least one step kept.

Speed (proposal section 6). Seconds per optimizer step = the wall_s of phase B 'log' events summed over their
optimizer_steps (the last record of each logged step, so that a step logged by two calls counts once). Smoke: all
its intervals, and the intervals that start at step N or later (no gradient-norm logging: the rate E1 runs at).
Each reference run: the same steps (its phase B intervals starting at or after step N and ending at or before the
smoke's last logged step) and its whole phase B (every phase B interval: the clock rate, checkpoints and dev numbers
included). ratio = smoke rate after step N / reference rate on the same steps. E1's projected phase B hours per
reference: reference whole-phase rate x ratio x the phase B steps of the smoke's config (its 'start' event) / 3600;
the smoke's own rate x those steps / 3600 is given beside (no checkpoint or dev time).

Outputs (--out, default reports/bench/jepa_smoke): lambda.json and lambda.md (the smoke log's path, size and SHA-256,
its config SHA-256 and name, the MoveAtlas commit and code SHA-256 of its provenance.json, lambda with its per-step
table, the checks, the speed). --write-config CONFIG, only when every check holds: CONFIG must be the configuration
the smoke ran (config SHA-256 of the start event); the value of the weight line of its jepa section becomes lambda
rounded to 6 significant digits, the new text must parse (mdfm.train.config.load) to the old configuration with
jepa.weight replaced and nothing else, and it is written in place (a temporary file, then a rename); the config
SHA-256 before and after go to lambda.json and lambda.md. A CONFIG that already holds that value is left as it is.
lambda.json and lambda.md are written by every call; a call on the same smoke log (same SHA-256) as the lambda.json
already in --out keeps that file's records: the config record of the write of lambda (with its time, 'at'), unless
this call writes the config or finds the value in another file or in a file changed since that write, and the list
e1_jobs of every E1 job file written from that log (by path, the latest of a path). A changed smoke log starts the
records afresh.
--e1-job PATH, only when every check holds and CONFIG (--write-config, or --config) holds lambda: the job file of
E1 for scripts/queue_job.py --file (written, not queued): cls 2, training, train_stage1.py --phase B --init-from (the
smoke's source) --config CONFIG --run-dir --e1-run-dir (default cache/runs/pilot_e1_jepa) --prefetch 2
--memory-cap-gb 14.5, without --jepa-grad-steps; requires the source checkpoint and lambda.json; s_per_step the
largest projected rate (per reference, and the smoke's own), est_hours that rate times the steps of the call,
timeout_hours twice that. The full phase B (job e1_jepa_train) by default; with --e1-max-steps N, the JP2 screen of
section 3a of the proposal (job e1_jepa_screen: --max-steps N, train_steps N, its output ckpt_B_<N>.pt), which a
later full run continues with --resume of that checkpoint.

Usage (mdfm environment, from the tree root; CUDA_VISIBLE_DEVICES=-1):
  python scripts/jepa_lambda.py
  python scripts/jepa_lambda.py --write-config configs/stage1_pilot_jepa.yaml
      --e1-job reports/bench/jepa_smoke/e1_jepa_train.job.json
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
sys.path.insert(0, TREE)
sys.path.insert(0, HERE)

FORMAT = "mdfm-jepa-lambda/1"
RUN = "cache/runs/jepa_smoke"
REFERENCES = ("cache/runs/phaseA_root_full", "cache/runs/pilot_v1_s2")
OUT = "reports/bench/jepa_smoke"
E1_RUN = "cache/runs/pilot_e1_jepa"
STEPS = 50
EXPECT_STEPS = 200
MAX_RESERVED_GB = 14.0
COLLAPSE_LINE = 0.01            # mdfm/train/loop.py ALARMS['jepa_collapse'] (checked equal in the tests)
SMOKE_WEIGHT = 1.0
BAD_EVENTS = ("alarm", "nonfinite", "memory_exceeded", "synthetic_cap")
DIGITS = 6
WEIGHT_COMMENT = "# lambda: scripts/jepa_lambda.py, reports/bench/jepa_smoke/lambda.md (proposal 2.7)"


# ---------------------------------------------------------------- reading
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    """(records, unparsed line count) of a JSON-lines file."""
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                bad += 1
    return recs, bad


def last_call(recs):
    """The records from the last 'start' event on (all of them without one)."""
    idx = [i for i, r in enumerate(recs) if r.get("event") == "start"]
    return recs[idx[-1]:] if idx else list(recs)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ---------------------------------------------------------------- lambda
def lambda_of(call, n=STEPS):
    """lambda of the module docstring from the call's 'jepa_grad' events: {lambda, steps: [{step, g_J, g_R, kept}],
    kept, left_out, zero_g_J, missing, repeated, mean_g_J, mean_g_R}."""
    seen, repeated = {}, []
    for r in call:
        if r.get("event") != "jepa_grad" or r.get("phase") != "B":
            continue
        s = int(r.get("step", -1))
        if not 0 <= s < n:
            continue
        if s in seen:
            repeated.append(s)
        seen[s] = r
    rows = []
    for s in range(n):
        if s not in seen:
            continue
        gj, gr = seen[s].get("g_J"), seen[s].get("g_R")
        rows.append({"step": s, "g_J": gj, "g_R": gr, "kept": finite(gj) and finite(gr)})
    kept = [r for r in rows if r["kept"]]
    mj = sum(float(r["g_J"]) for r in kept) / len(kept) if kept else None
    mr = sum(float(r["g_R"]) for r in kept) / len(kept) if kept else None
    lam = mr / mj if kept and mj and mj > 0 else None
    return {"lambda": lam, "n": n, "steps": rows, "kept": len(kept),
            "left_out": [r["step"] for r in rows if not r["kept"]],
            "zero_g_J": [r["step"] for r in kept if float(r["g_J"]) == 0.0],
            "missing": [s for s in range(n) if s not in seen], "repeated": sorted(set(repeated)),
            "mean_g_J": mj, "mean_g_R": mr}


def rounded(x, digits=DIGITS):
    """x rounded to `digits` significant digits."""
    return float(f"{float(x):.{int(digits)}g}")


# ---------------------------------------------------------------- checks
def peak_reserved(call):
    """The largest memory_gb.max_reserved of the call's 'log' and 'end' events (None without a CUDA record)."""
    vals = [r["memory_gb"]["max_reserved"] for r in call if r.get("event") in ("log", "end")
            and isinstance(r.get("memory_gb"), dict) and finite(r["memory_gb"].get("max_reserved"))]
    return max(vals) if vals else None


def checks(call, lam, n=STEPS, expect_steps=EXPECT_STEPS, max_gb=MAX_RESERVED_GB):
    """[{name, ok, detail}] of the module docstring's checks."""
    out = []

    def add(name, ok, detail):
        out.append({"name": name, "ok": bool(ok), "detail": detail})
    start = next((r for r in call if r.get("event") == "start"), None)
    jp = (start or {}).get("jepa") or {}
    sec = jp.get("section") or {}
    add("start", start is not None and sec.get("enabled") is True and sec.get("weight") == SMOKE_WEIGHT
        and int(jp.get("grad_steps") or 0) >= n,
        f"jepa enabled {sec.get('enabled')}, weight {sec.get('weight')}, grad_steps {jp.get('grad_steps')}")
    init = [r for r in call if r.get("event") == "init_from"]
    add("init_from", len(init) == 1, f"{len(init)} init_from event(s)"
        + (f": {init[0].get('source', {}).get('path')} (SHA-256 {init[0].get('source', {}).get('sha256')})"
           if init else ""))
    bad = {e: sum(r.get("event") == e for r in call) for e in BAD_EVENTS}
    add("no_bad_events", not any(bad.values()), ", ".join(f"{e} {k}" for e, k in bad.items()))
    end = [r for r in call if r.get("event") == "end"]
    e = end[-1] if end else {}
    add("end", bool(end) and e.get("phase") == "B" and e.get("step") == expect_steps,
        f"end at phase {e.get('phase')} step {e.get('step')} (expected B {expect_steps})")
    ver = [r for r in call if r.get("event") == "verify" and r.get("phase") == "B"]
    add("verify", bool(ver), f"{len(ver)} verify event(s) of phase B"
        + (f": {ver[-1].get('path')}, drift {ver[-1].get('drift')}" if ver else ""))
    logs = [r for r in call if r.get("event") == "log" and r.get("phase") == "B"]
    bad_logs = []
    ratios = []
    for r in logs:
        j = r.get("jepa") or {}
        if not finite(r.get("loss")):
            bad_logs.append(f"step {r.get('step')}: loss {r.get('loss')}")
        if (j.get("windows") or 0) > 0 and not finite(j.get("loss")):
            bad_logs.append(f"step {r.get('step')}: jepa loss {j.get('loss')}")
        cr = j.get("collapse_ratio")
        if cr is not None:
            ratios.append(float(cr))
            if not finite(cr) or float(cr) < COLLAPSE_LINE:
                bad_logs.append(f"step {r.get('step')}: collapse ratio {cr}")
    add("logs", bool(logs) and not bad_logs and all("jepa" in r for r in logs),
        f"{len(logs)} phase B log events; collapse ratio min {min(ratios) if ratios else None}"
        + (f"; problems: {'; '.join(bad_logs)}" if bad_logs else ""))
    peak = peak_reserved(call)
    add("peak_reserved", peak is not None and peak < float(max_gb),
        f"peak reserved {peak} GB (line {max_gb} GB)")
    add("lambda", lam["lambda"] is not None and not lam["missing"] and not lam["repeated"],
        f"{lam['kept']} of {n} steps kept, left out {lam['left_out']}, missing {lam['missing']}, "
        f"repeated {lam['repeated']}")
    return out


# ---------------------------------------------------------------- speed
def intervals(recs, phase="B"):
    """{end step: (optimizer steps, wall_s)} of the phase's 'log' events, the last record of each step."""
    out = {}
    for r in recs:
        if r.get("event") == "log" and r.get("phase") == phase and finite(r.get("wall_s")) and \
                int(r.get("optimizer_steps") or 0) > 0:
            out[int(r["step"])] = (int(r["optimizer_steps"]), float(r["wall_s"]))
    return out


def rate(iv, lo=None, hi=None):
    """(seconds per step, steps) over the intervals that start at or after lo and end at or before hi."""
    steps = wall = 0.0
    for end, (n, w) in iv.items():
        if (lo is None or end - n >= lo) and (hi is None or end <= hi):
            steps += n
            wall += w
    return (wall / steps if steps else None), int(steps)


def speed(call, refs, n=STEPS):
    """The speed section of the module docstring; refs: {name: records of the reference log}."""
    iv = intervals(call)
    last = max(iv) if iv else None
    start = next((r for r in call if r.get("event") == "start"), {})
    n_b = int((start.get("steps") or [0, 0])[1])
    all_rate, all_steps = rate(iv)
    after, after_steps = rate(iv, lo=n)
    out = {"phase_b_steps": n_b, "smoke": {"all": {"s_per_step": all_rate, "steps": all_steps},
                                           "after_grad_logging": {"s_per_step": after, "steps": after_steps,
                                                                  "from_step": n, "to_step": last}},
           "references": {}, "projection_h": {}}
    if after is not None and n_b:
        out["projection_h"]["smoke_rate"] = after * n_b / 3600.0
    for name, recs in refs.items():
        riv = intervals(recs)
        same, same_steps = rate(riv, lo=n, hi=last)
        whole, whole_steps = rate(riv)
        ratio = after / same if after is not None and same else None
        out["references"][name] = {"same_steps": {"s_per_step": same, "steps": same_steps},
                                   "whole_phase_b": {"s_per_step": whole, "steps": whole_steps},
                                   "ratio": ratio,
                                   "projected_s_per_step": whole * ratio if whole and ratio else None}
        if whole and ratio and n_b:
            out["projection_h"][name] = whole * ratio * n_b / 3600.0
    return out


# ---------------------------------------------------------------- the config
def weight_line(text):
    """(line index, indent) of the weight line of the top-level jepa section of a YAML text; ValueError when there
    is not exactly one."""
    lines = text.split("\n")
    try:
        i0 = next(i for i, ln in enumerate(lines) if re.match(r"^jepa:\s*(#.*)?$", ln))
    except StopIteration:
        raise ValueError("no top-level jepa section") from None
    hits = []
    for i in range(i0 + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln[0].isspace():
            break
        m = re.match(r"^(\s+)weight:\s*[^\s#]+\s*(#.*)?$", ln)
        if m:
            hits.append((i, m.group(1)))
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} weight lines in the jepa section")
    return hits[0]


def write_config(path, value, smoke_sha):
    """Write `value` as jepa.weight of the YAML config `path` (module docstring); returns {path, sha256_before,
    sha256_after, file_sha256_before, file_sha256_after, written}."""
    from mdfm.train import config as CF
    old = CF.load(path)
    before = CF.sha256(old)
    info = {"path": path.replace("\\", "/"), "sha256_before": before, "file_sha256_before": sha256_file(path),
            "weight_before": float(old.jepa.weight), "weight_after": float(value)}
    if float(old.jepa.weight) == float(value) and before != smoke_sha:
        return dict(info, sha256_after=before, file_sha256_after=info["file_sha256_before"], written=False)
    if before != smoke_sha:
        raise ValueError(f"{path} (config SHA-256 {before}) is not the configuration the smoke ran ({smoke_sha})")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    i, indent = weight_line(text)
    lines = text.split("\n")
    left = f"{indent}weight: {float(value)!r}"
    lines[i] = left + " " * max(1, 35 - len(left)) + WEIGHT_COMMENT
    new = "\n".join(lines)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    try:
        got = CF.load(tmp)
        want = replace(old, jepa=replace(old.jepa, weight=float(value)))
        if CF.to_dict(got) != CF.to_dict(want):
            raise ValueError("the rewritten config differs from the old one by more than jepa.weight")
    except Exception:
        os.remove(tmp)
        raise
    os.replace(tmp, path)
    return dict(info, sha256_after=CF.sha256(got), file_sha256_after=sha256_file(path), written=True)


def e1_job(source, config, run_dir, n_b, s_per_step, lam_value, cfg_sha, out=OUT, max_steps=None):
    """The E1 job dict (module docstring), checked by queue_job.normalize: n_b phase B steps of the config;
    max_steps: the JP2 screen's call length (None: the full phase B); out: the folder of lambda.json."""
    import queue_job as QJ
    n = int(n_b) if max_steps is None else int(max_steps)
    hours = float(s_per_step) * n / 3600.0
    argv = [QJ.PYTHON, "scripts/train_stage1.py", "--phase", "B", "--init-from", source, "--config", config,
            "--run-dir", run_dir, "--prefetch", "2", "--memory-cap-gb", "14.5"]
    what = "phase B"
    if max_steps is not None:
        argv += ["--max-steps", str(n)]
        what = f"the JP2 screen (section 3a), {n:,} phase B steps,"
    job = {"name": "e1_jepa_train" if max_steps is None else "e1_jepa_screen", "group": "E1", "cls": 2,
           "priority": 1, "gpu": True, "training": True, "run_dir": run_dir, "argv": argv,
           "requires": [source, f"{out}/lambda.json"], "train_phase": "B", "train_steps": n,
           "s_per_step": round(float(s_per_step), 4), "est_hours": round(hours, 2),
           "timeout_hours": round(2.0 * hours, 1), "outputs": [f"{run_dir}/ckpt_B_{n:06d}.pt"],
           "note": f"E1 of docs/proposals/JEPA_EXPLORATION_DRAFT.md (section 2): {what} from {source} at jepa.weight "
                   f"{lam_value!r} ({out}/lambda.md), config SHA-256 {cfg_sha}"}
    return QJ.normalize(job)


# ---------------------------------------------------------------- report
def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}g}"


def markdown(res):
    lam = res["lambda"]
    L = ["# JEPA smoke: lambda of E1, checks and speed", "",
         f"Written {res['written']} by scripts/jepa_lambda.py (task J2b; docs/proposals/JEPA_EXPLORATION_DRAFT.md, "
         f"sections 2.7, 2.8 and 6). Every number below is in lambda.json beside this file.", "",
         "## Smoke run", "",
         f"- log: {res['log']['path']} ({res['log']['records']} records, {res['log']['bytes']:,} bytes, SHA-256 "
         f"{res['log']['sha256']}); call from its last start event, {res['log']['call_records']} records",
         f"- config: {res['smoke']['name']}, config SHA-256 {res['smoke']['config_sha256']}",
         f"- MoveAtlas commit {res['smoke']['moveatlas_head']}, code SHA-256 {res['smoke']['code_sha256']} "
         f"(the run folder's provenance.json)",
         f"- source: {res['smoke']['source']}", "",
         "## Checks", "", "| check | holds | detail |", "|---|---|---|"]
    for c in res["checks"]:
        L.append(f"| {c['name']} | {'yes' if c['ok'] else 'NO'} | {c['detail']} |")
    L += ["", f"All checks hold: {'yes' if res['passed'] else 'NO'}.", "",
          "## lambda (proposal 2.7)", "",
          f"Steps 0 to {lam['n'] - 1} of phase B: {lam['kept']} kept, left out {lam['left_out']} (null or not "
          f"finite), g_J = 0 at {lam['zero_g_J']}, missing {lam['missing']}, repeated {lam['repeated']}.",
          f"mean g_R {_f(lam['mean_g_R'], 6)}, mean g_J {_f(lam['mean_g_J'], 6)}: lambda = mean(g_R) / mean(g_J) = "
          f"{_f(lam['lambda'], 10)}; rounded to {DIGITS} significant digits: {_f(res['lambda_config'], DIGITS)}.", ""]
    if res.get("config"):
        c = res["config"]
        L += [f"Config {c['path']}: jepa.weight {c['weight_before']} -> {c['weight_after']} "
              f"({'written' if c['written'] else 'already held'} at {c.get('at')}); config SHA-256 "
              f"{c['sha256_before']} -> {c['sha256_after']}; file SHA-256 {c['file_sha256_before']} -> "
              f"{c['file_sha256_after']}.", ""]
    for j in res.get("e1_jobs") or []:
        L += [f"E1 job file (not queued), written at {j.get('at')}: {j['path']}, job {j['name']}, "
              f"{j['train_steps']:,} phase B steps at {_f(j['s_per_step'])} s per step, {_f(j['est_hours'])} h; "
              f"queue it with `python scripts/queue_job.py --file {j['path']}`.", ""]
    L += ["| step | g_J | g_R | kept |", "|---|---|---|---|"]
    for r in lam["steps"]:
        L.append(f"| {r['step']} | {_f(r['g_J'], 6)} | {_f(r['g_R'], 6)} | {'yes' if r['kept'] else 'no'} |")
    sp = res["speed"]
    sm = sp["smoke"]
    L += ["", "## Speed (proposal section 6)", "",
          f"- smoke, all intervals: {_f(sm['all']['s_per_step'])} s per step over {sm['all']['steps']} steps",
          f"- smoke, steps {sm['after_grad_logging']['from_step']} to {sm['after_grad_logging']['to_step']} (no "
          f"gradient-norm logging): {_f(sm['after_grad_logging']['s_per_step'])} s per step over "
          f"{sm['after_grad_logging']['steps']} steps"]
    for name, r in sp["references"].items():
        L.append(f"- {name}: {_f(r['same_steps']['s_per_step'])} s per step on the same steps "
                 f"({r['same_steps']['steps']}), {_f(r['whole_phase_b']['s_per_step'])} over its whole phase B "
                 f"({r['whole_phase_b']['steps']} steps); smoke / reference {_f(r['ratio'])}; E1 projected "
                 f"{_f(r['projected_s_per_step'])} s per step, {_f(sp['projection_h'].get(name))} h for "
                 f"{sp['phase_b_steps']:,} phase B steps")
    L.append(f"- the smoke's own rate over {sp['phase_b_steps']:,} steps: {_f(sp['projection_h'].get('smoke_rate'))} h "
             f"(no checkpoint or dev time)")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main
def absolute(p):
    return p if os.path.isabs(p) else os.path.join(TREE, p)


def tree_rel(p):
    a = os.path.abspath(absolute(p))
    try:
        r = os.path.relpath(a, TREE)
    except ValueError:
        return a.replace("\\", "/")
    return a.replace("\\", "/") if r.startswith("..") else r.replace("\\", "/")


def run(args):
    log = absolute(os.path.join(args.run, "log.jsonl"))
    recs, bad = read_jsonl(log)
    call = last_call(recs)
    lam = lambda_of(call, args.steps)
    chk = checks(call, lam, args.steps, args.expect_steps, args.max_reserved_gb)
    passed = all(c["ok"] for c in chk)
    refs = {}
    for r in args.reference:
        p = absolute(os.path.join(r, "log.jsonl"))
        if os.path.exists(p):
            refs[tree_rel(r)] = read_jsonl(p)[0]
    sp = speed(call, refs, args.steps)
    start = next((r for r in call if r.get("event") == "start"), {})
    init = next((r for r in call if r.get("event") == "init_from"), {})
    prov_path = absolute(os.path.join(args.run, "provenance.json"))
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as f:
            prov = json.load(f)
    hashes = prov.get("hashes") or {}
    src = init.get("source") or {}
    res = {"format": FORMAT, "written": datetime.datetime.now().isoformat(timespec="seconds"),
           "log": {"path": tree_rel(log), "sha256": sha256_file(log), "bytes": os.path.getsize(log),
                   "records": len(recs), "unparsed_lines": bad, "call_records": len(call)},
           "smoke": {"name": start.get("name"), "config_sha256": start.get("config_sha256"),
                     "steps": start.get("steps"), "jepa": start.get("jepa"), "source": src.get("path"),
                     "source_sha256": src.get("sha256"), "moveatlas_head": hashes.get("moveatlas_head"),
                     "code_sha256": hashes.get("code_sha256")},
           "steps": args.steps, "checks": chk, "passed": passed, "lambda": lam,
           "lambda_config": rounded(lam["lambda"]) if lam["lambda"] is not None else None, "speed": sp}
    if passed and args.write_config:
        res["config"] = write_config(absolute(args.write_config), res["lambda_config"], start.get("config_sha256"))
        res["config"].update(path=tree_rel(args.write_config), at=res["written"])
    if passed and args.e1_job:
        res["e1_job"] = dict(write_e1_job(args, res), at=res["written"])
    out = absolute(args.out)
    carry_over(res, previous(os.path.join(out, "lambda.json"), res["log"]["sha256"]))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "lambda.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    with open(os.path.join(out, "lambda.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown(res))
    if (args.write_config or args.e1_job) and not passed:
        raise SystemExit("jepa_lambda: the smoke's checks do not all hold, nothing written but lambda.json and "
                         "lambda.md: " + "; ".join(f"{c['name']}: {c['detail']}" for c in chk if not c["ok"]))
    return res


def previous(path, log_sha):
    """The lambda.json at `path` when this script wrote it from the same smoke log (log SHA-256 log_sha), else
    None."""
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(prev, dict) or prev.get("format") != FORMAT or (prev.get("log") or {}).get("sha256") != log_sha:
        return None
    return prev


def carry_over(res, prev):
    """Keep in res what the earlier calls on the same smoke log recorded (module docstring, Outputs): e1_jobs, every
    E1 job file written (by path, the latest record of a path), and the config record of the write of lambda, which
    a call that writes nothing to the config (no --write-config, or the value already held by the file the earlier
    call wrote) does not replace."""
    prev = prev or {}
    jobs = {}
    for j in list(prev.get("e1_jobs") or []) + [prev.get("e1_job"), res.get("e1_job")]:
        if j:
            jobs[j["path"]] = j
    if jobs:
        res["e1_jobs"] = [jobs[p] for p in sorted(jobs)]
    old, cur = prev.get("config"), res.get("config")
    if old and (cur is None or (not cur["written"] and old.get("written") and old.get("path") == cur["path"]
                                and old.get("sha256_after") == cur["sha256_after"])):
        res["config"] = old
    return res


def write_e1_job(args, res):
    """Write E1's job file (module docstring); returns {path, s_per_step, est_hours}."""
    from mdfm.train import config as CF
    cfg_path = args.write_config or args.config
    cfg = CF.load(absolute(cfg_path))
    if float(cfg.jepa.weight) != res["lambda_config"]:
        raise SystemExit(f"jepa_lambda: {cfg_path} holds jepa.weight {cfg.jepa.weight}, not lambda "
                         f"{res['lambda_config']}: give --write-config")
    sp = res["speed"]
    rates = [r["projected_s_per_step"] for r in sp["references"].values() if r.get("projected_s_per_step")]
    own = sp["smoke"]["after_grad_logging"]["s_per_step"]
    rates += [own] if own else []
    if not rates or not sp["phase_b_steps"]:
        raise SystemExit("jepa_lambda: no rate to estimate E1's hours")
    job = e1_job(tree_rel(res["smoke"]["source"]), tree_rel(cfg_path), args.e1_run_dir, sp["phase_b_steps"],
                 max(rates), res["lambda_config"], CF.sha256(cfg), out=tree_rel(args.out), max_steps=args.e1_max_steps)
    path = absolute(args.e1_job)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(job, f, indent=1)
        f.write("\n")
    return {"path": tree_rel(path), "name": job["name"], "train_steps": job["train_steps"],
            "s_per_step": job["s_per_step"], "est_hours": job["est_hours"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=RUN, help="the smoke's run folder (its log.jsonl and provenance.json)")
    ap.add_argument("--reference", nargs="*", default=list(REFERENCES), help="reference run folders (speed)")
    ap.add_argument("--steps", type=int, default=STEPS, help="phase B steps of the lambda rule")
    ap.add_argument("--expect-steps", type=int, default=EXPECT_STEPS, help="the smoke's last phase B step")
    ap.add_argument("--max-reserved-gb", type=float, default=MAX_RESERVED_GB)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--write-config", default=None, help="the config to receive lambda as jepa.weight")
    ap.add_argument("--config", default="configs/stage1_pilot_jepa.yaml",
                    help="the E1 config read by --e1-job when --write-config is not given")
    ap.add_argument("--e1-job", default=None, help="write E1's job file (not queued)")
    ap.add_argument("--e1-run-dir", default=E1_RUN)
    ap.add_argument("--e1-max-steps", type=int, default=None, help="the JP2 screen: phase B steps of the E1 call")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    res = run(a)
    print(f"lambda {res['lambda']['lambda']} (config value {res['lambda_config']}); checks "
          f"{'all hold' if res['passed'] else 'FAILED: ' + ', '.join(c['name'] for c in res['checks'] if not c['ok'])}")
    print(f"wrote {tree_rel(os.path.join(absolute(a.out), 'lambda.json'))} and lambda.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
