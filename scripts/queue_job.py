"""Job files of the GPU queue that scripts/gpu_queue.py runs: write_job(dict), write_template(dict) and a CLI.

Evaluation contract: every evaluation script takes --checkpoint, --weights {ema,model}, --device, --out and
--memory-gb. (scripts/probe_root_motion.py, a probe, takes --cap-gb in place of --memory-gb, and its --out is the
folder of its root_motion_<name><tag> files.)

Job file: cache/runs/queue/jobs/<cls>_<priority:03d>_<name>.json, written atomically (a temporary file in the
same folder, then os.replace), format mdfm-queue-job/1. Fields:
- name: unique over the jobs folder ([A-Za-z0-9_.-], at most 120 characters);
- argv: the command; its first item is the mdfm python (PYTHON); paths are relative to the tree root, which is
  the runner's working folder;
- env: extra environment variables {str: str}; the runner adds CUDA_VISIBLE_DEVICES=0 to a GPU job unless env
  names it, CUDA_VISIBLE_DEVICES=-1 to a CPU job, and PYTHONIOENCODING=utf-8;
- cls: 1 branch inputs and post-train comparators, 2 training, 3 other evaluations, 4 reference-checkpoint jobs,
  5 extension;
- priority: 0 to 999, the lower first within a class;
- gpu: false for a CPU-only step; training: true for a scripts/train_stage1.py run; run_dir: its run folder;
- uses: the entry script and the config (tree-relative), read by the lock and green-marker checks (default:
  argv[1] and the value of --config);
- requires: paths that must exist;
- after: job names that must have exited 0 (the trigger phase_b_done counts as a job that exited 0 once it
  has fired);
- soft_after with soft_deadline: names waited for (finished in any state) until soft_deadline, then ignored;
- not_before: no launch before this time;
- not_if: name globs; the job is blocked while a matching job is pending or running;
- not_if_any: name globs; the job is blocked while a matching job exists in any state;
- est_hours (required); timeout_hours (default 2 x est_hours for an evaluation, none for training); outputs;
- optional: group (J1, J2, J3, seed2, ...), train_phase and train_steps (training: the phase this call runs and
  its step count, read by the deadline guard), s_per_step (training: the measured rate of that estimate),
  max_resumes (training: resumes after a crash, default 1), note.
Times are local times of this PC in ISO 8601 without a zone ('2026-10-02T03:30').

Placeholders in argv, requires and outputs, resolved by the runner at launch (a job waits while one is
unresolved): {phase_b_ckpt} and {phase_b_name} (the trigger phase_b_done of scripts/gpu_queue.py),
{ref_ckpt} and {ref_name} (scripts/weekend_branch.py --stage reference), {ext_ckpt} and {ext_steps}
(scripts/weekend_branch.py --stage extension, the file cache/runs/queue/EXTEND).

Post-train templates: cache/runs/queue/post_train/<name>.json, format mdfm-queue-template/1, with name, cls,
runs (fnmatch globs of run folder names; a leading '!' excludes) and jobs (job dicts; cls comes from the
template). When a training run of the queue logs its phase B 'verify' event, the runner writes one job file per
template job with these placeholders replaced: {ckpt} (the verified checkpoint), {name} (its report name,
<run folder>_<file stem>, as mdfm.eval.infer.checkpoint_name), {run_dir}, {run} (the run folder name),
{config} (the run's --config), {ckpt_a4000} and {ckpt_b4000} (the run's ckpt_A_004000.pt and ckpt_B_004000.pt).

Usage (mdfm environment):
  python scripts/queue_job.py --name N --cls 1 --priority 10 --est-hours 0.5 [options] -- SCRIPT ARGS...
      (the mdfm python is put in front of SCRIPT)
  python scripts/queue_job.py --file job.json [--replace]
  python scripts/queue_job.py --template template.json [--replace]
  python scripts/queue_job.py --seed-weekend [--dry-run]
      the J1, J2 and seed-2 jobs and the post-train templates of the weekend 2026-10-01 (seed_weekend)
"""
import argparse
import datetime
import fnmatch
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
PYTHON = "<conda>/envs/mdfm/python.exe"
# The video environment of env/VIDEO_ENV.md (owner approval 2026-10-05): its jobs (scripts/video_generate.py) run
# under their own interpreter, never under mdfm's. No other interpreter is accepted.
VIDEO_PYTHON = "<store>/envs/mdfm_video/python.exe"
PYTHONS = (PYTHON, VIDEO_PYTHON)
JOB_FORMAT = "mdfm-queue-job/1"
TEMPLATE_FORMAT = "mdfm-queue-template/1"
CLASSES = {1: "branch inputs and post-train comparators", 2: "training", 3: "other evaluations",
           4: "reference-checkpoint jobs", 5: "extension"}
NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")
LIST_FIELDS = ("uses", "requires", "after", "soft_after", "not_if", "not_if_any", "outputs")
TIME_FIELDS = ("soft_deadline", "not_before")
QUEUE_PLACEHOLDERS = ("phase_b_ckpt", "phase_b_name", "ref_ckpt", "ref_name", "ext_ckpt", "ext_steps")
TEMPLATE_PLACEHOLDERS = ("ckpt", "name", "run_dir", "run", "config", "ckpt_a4000", "ckpt_b4000")
TRIGGERS = ("phase_b_done",)


class JobError(ValueError):
    """A job or template that does not meet the format of the module docstring."""


def queue_dir(tree=TREE):
    return os.path.join(tree, "cache", "runs", "queue")


def jobs_dir(tree=TREE):
    return os.path.join(queue_dir(tree), "jobs")


def templates_dir(tree=TREE):
    return os.path.join(queue_dir(tree), "post_train")


def same_python(a, b=PYTHON):
    """True when two interpreter paths name the same file (case and slashes ignored)."""
    norm = lambda p: os.path.normcase(os.path.normpath(str(p).replace("/", os.sep)))  # noqa: E731
    return norm(a) == norm(b)


def parse_time(text):
    """A naive local datetime from ISO 8601 text (a trailing zone is refused: times are local)."""
    t = datetime.datetime.fromisoformat(str(text))
    if t.tzinfo is not None:
        raise JobError(f"time {text!r} carries a zone; queue times are local")
    return t


def tree_rel(path, tree=TREE):
    """A tree-relative path with forward slashes (absolute paths inside the tree are made relative)."""
    p = str(path).replace("\\", "/")
    if os.path.isabs(p):
        try:
            r = os.path.relpath(p, tree)
        except ValueError:            # another drive
            return p
        if not r.startswith(".."):
            return r.replace("\\", "/")
    return p


def checkpoint_name(path):
    """'<run folder>_<file stem>' of a checkpoint path (mdfm.eval.infer.checkpoint_name, without the import)."""
    p = os.path.abspath(str(path).replace("/", os.sep))
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"


def default_uses(argv):
    """argv[1] when it is a .py file, and the value of --config."""
    out = []
    if len(argv) > 1 and str(argv[1]).endswith(".py"):
        out.append(tree_rel(argv[1]))
    for i, a in enumerate(argv[:-1]):
        if a == "--config":
            out.append(tree_rel(argv[i + 1]))
    return out


def normalize(job):
    """The job with its defaults filled in and its fields checked (JobError otherwise)."""
    if not isinstance(job, dict):
        raise JobError("a job is a dict")
    j = dict(job)
    name = j.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise JobError(f"name {name!r}: [A-Za-z0-9_.-], 1 to 120 characters")
    argv = j.get("argv")
    if not isinstance(argv, list) or len(argv) < 2 or not all(isinstance(a, str) for a in argv):
        raise JobError(f"{name}: argv must be a list of at least two strings")
    if not any(same_python(argv[0], p) for p in PYTHONS):
        raise JobError(f"{name}: argv[0] {argv[0]!r} is not the mdfm python {PYTHON} nor the video python "
                       f"{VIDEO_PYTHON}")
    if same_python(argv[0], VIDEO_PYTHON) and not os.path.basename(argv[1]).startswith("video_"):
        raise JobError(f"{name}: the video python runs only scripts/video_*.py, not {argv[1]!r}")
    env = j.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise JobError(f"{name}: env must map str to str")
    j["env"] = dict(env)
    cls = j.get("cls")
    if cls not in CLASSES:
        raise JobError(f"{name}: cls {cls!r}, expected one of {sorted(CLASSES)}")
    pri = j.get("priority")
    if not isinstance(pri, int) or isinstance(pri, bool) or not 0 <= pri <= 999:
        raise JobError(f"{name}: priority {pri!r}, expected an int from 0 to 999")
    j["gpu"] = bool(j.get("gpu", True))
    j["training"] = bool(j.get("training", False))
    if j["training"] and not j.get("run_dir"):
        raise JobError(f"{name}: a training job needs run_dir")
    if j.get("run_dir") is not None:
        j["run_dir"] = tree_rel(j["run_dir"])
    for f in LIST_FIELDS:
        v = j.get(f)
        v = [] if v is None else v
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise JobError(f"{name}: {f} must be a list of strings")
        j[f] = list(v)
    if not j["uses"]:
        j["uses"] = default_uses(argv)
    if not j["uses"]:
        raise JobError(f"{name}: uses is empty and argv names no script")
    j["uses"] = [tree_rel(u) for u in j["uses"]]
    for f in TIME_FIELDS:
        if j.get(f) is not None:
            parse_time(j[f])
    if j["soft_after"] and not j.get("soft_deadline"):
        raise JobError(f"{name}: soft_after needs soft_deadline")
    est = j.get("est_hours")
    if not isinstance(est, (int, float)) or isinstance(est, bool) or not est > 0:
        raise JobError(f"{name}: est_hours {est!r}, expected a positive number")
    j["est_hours"] = float(est)
    if j.get("timeout_hours") is None:
        j["timeout_hours"] = None if j["training"] else round(2.0 * j["est_hours"], 4)
    elif not isinstance(j["timeout_hours"], (int, float)) or not j["timeout_hours"] > 0:
        raise JobError(f"{name}: timeout_hours {j['timeout_hours']!r}")
    if j["training"]:
        j["max_resumes"] = int(j.get("max_resumes", 1))
    j["format"] = JOB_FORMAT
    return j


def job_filename(job):
    return f"{int(job['cls'])}_{int(job['priority']):03d}_{job['name']}.json"


def read_json(path):
    """The parsed file, or None when it is missing or not JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def existing_names(folder):
    """{job name: file path} of the job files of a folder (AppleDouble sidecars left out)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.json"))):
        if os.path.basename(p).startswith(("._", ".")):
            continue
        d = read_json(p)
        if isinstance(d, dict) and isinstance(d.get("name"), str):
            out[d["name"]] = p
    return out


def _atomic_json(path, obj):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return path


def write_job(job, tree=TREE, replace=False, folder=None):
    """Check the job (normalize), refuse a name already queued (unless replace, which removes the old file) and
    write it atomically; returns the path."""
    j = normalize(job)
    folder = jobs_dir(tree) if folder is None else folder
    names = existing_names(folder)
    path = os.path.join(folder, job_filename(j))
    if j["name"] in names:
        if not replace:
            raise JobError(f"{j['name']}: already queued as {names[j['name']]}")
        if os.path.abspath(names[j["name"]]) != os.path.abspath(path):
            os.remove(names[j["name"]])
    j.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    return _atomic_json(path, j)


def normalize_template(tpl):
    """The template checked: name, cls, runs and jobs (each job checked with the template's cls)."""
    if not isinstance(tpl, dict):
        raise JobError("a template is a dict")
    t = dict(tpl)
    if not isinstance(t.get("name"), str) or not NAME.match(t["name"]):
        raise JobError(f"template name {t.get('name')!r}")
    if t.get("cls") not in CLASSES:
        raise JobError(f"template {t['name']}: cls {t.get('cls')!r}")
    runs = t.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(r, str) for r in runs):
        raise JobError(f"template {t['name']}: runs must be a non-empty list of globs")
    jobs = t.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise JobError(f"template {t['name']}: jobs must be a non-empty list")
    probe = {k: "x" for k in TEMPLATE_PLACEHOLDERS}
    for j in jobs:
        normalize(substitute(dict(j, cls=t["cls"]), probe))
    t["format"] = TEMPLATE_FORMAT
    return t


def write_template(tpl, tree=TREE, replace=False):
    """Check and write cache/runs/queue/post_train/<name>.json atomically; returns the path."""
    t = normalize_template(tpl)
    path = os.path.join(templates_dir(tree), f"{t['name']}.json")
    if os.path.exists(path) and not replace:
        raise JobError(f"template {t['name']} exists: {path}")
    t.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    return _atomic_json(path, t)


def runs_match(patterns, run):
    """True when the run folder name matches an include glob and no '!' exclude glob."""
    inc = [p for p in patterns if not p.startswith("!")]
    exc = [p[1:] for p in patterns if p.startswith("!")]
    return any(fnmatch.fnmatchcase(run, p) for p in inc) and not any(fnmatch.fnmatchcase(run, p) for p in exc)


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute(obj, values):
    """obj with every {key} of `values` replaced in its strings (lists and dicts walked); other braces kept."""
    if isinstance(obj, str):
        return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), obj)
    if isinstance(obj, list):
        return [substitute(x, values) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, values) for k, v in obj.items()}
    return obj


def placeholders(obj):
    """The set of {key} names left in the strings of obj."""
    if isinstance(obj, str):
        return set(PLACEHOLDER.findall(obj))
    if isinstance(obj, list):
        return set().union(*[placeholders(x) for x in obj]) if obj else set()
    if isinstance(obj, dict):
        return set().union(*[placeholders(v) for v in obj.values()]) if obj else set()
    return set()


# ---------------------------------------------------------------- the weekend of 2026-10-01
RUN1 = "cache/runs/phaseA_root_full"
RUN2 = "cache/runs/pilot_v1_s2"
CONFIG1 = "configs/stage1_pilot.yaml"
CONFIG2 = "configs/stage1_pilot_s2.yaml"
J3_SOFT = ["diag_A", "diag_B", "ablations_B", "insim_sites6_B", "warp15_B", "g1_decompose_B"]
SEED2_SOFT_DEADLINE = "2026-10-02T03:30"
# hours of one run, from the times of the phase A reports of 2026-09-30 (gpu_queue.measured_eval_minutes):
# eval_g3_dev.py with its secondary rows, eval_gait_insim.py and eval_g1_dev.py ran in at most 9, 7 and 35 min
EST = {"g3": 0.3, "g3_nosec": 0.25, "insim": 0.25, "g1": 0.75, "g1_lt2": 0.4, "rm_xsens13": 0.4,
       "rm_mix_latent": 0.6}


def _py(script, *args):
    return [PYTHON, f"scripts/{script}", *args]


def _ck1(stem):
    return f"{RUN1}/{stem}.pt"


def _nm1(stem):
    return f"phaseA_root_full_{stem}"


def weekend_jobs(rates):
    """The job dicts of J1, J2 and the two seed-2 training jobs (task T01, step 5). rates: {'A': s/step,
    'B': s/step} measured on seed 1 (gpu_queue.seed_rates)."""
    J = []
    # J1: the phase B end of seed 1, after the trigger phase_b_done
    d = "reports/gates/g3/dev_{phase_b_name}"
    J += [
        dict(name="j1_g3_dev_B", group="J1", priority=1, est_hours=EST["g3"], after=["phase_b_done"],
             argv=_py("eval_g3_dev.py", "--checkpoint", "{phase_b_ckpt}", "--out", d), outputs=[d]),
        dict(name="j1_g3_dev_B_model", group="J1", priority=2, est_hours=EST["g3_nosec"], after=["phase_b_done"],
             argv=_py("eval_g3_dev.py", "--checkpoint", "{phase_b_ckpt}", "--weights", "model", "--no-secondary",
                      "--out", d + "_model"), outputs=[d + "_model"]),
        dict(name="j1_insim_B", group="J1", priority=3, est_hours=EST["insim"], after=["phase_b_done", "j1_g3_dev_B"],
             argv=_py("eval_gait_insim.py", "--checkpoint", "{phase_b_ckpt}", "--real", d + "/g3_dev.json",
                      "--out", "reports/gates/g3_sim/insim_{phase_b_name}"),
             outputs=["reports/gates/g3_sim/insim_{phase_b_name}"]),
        dict(name="j1_g1_dev_B", group="J1", priority=4, est_hours=EST["g1"], after=["phase_b_done"],
             argv=_py("eval_g1_dev.py", "--checkpoint", "{phase_b_ckpt}", "--out", "reports/gates/g1/dev_{phase_b_name}"),
             outputs=["reports/gates/g1/dev_{phase_b_name}"]),
    ]
    j1 = [j["name"] for j in J]
    J2 = []
    pri = 10
    for s in ("004000", "008000", "012000", "016000", "020000", "024000", "028000"):
        out = f"reports/gates/g3/curve_{_nm1('ckpt_B_' + s)}"
        J2.append(dict(name=f"j2_g3_curve_B_{s}", priority=pri, est_hours=EST["g3_nosec"],
                       argv=_py("eval_g3_dev.py", "--checkpoint", _ck1("ckpt_B_" + s), "--no-secondary", "--out", out),
                       requires=[_ck1("ckpt_B_" + s)], outputs=[out]))
        pri += 1
    for stem in ("ckpt_A_004000", "ckpt_B_004000"):
        tag = stem[5:]
        out = f"reports/gates/g1/dev_{_nm1(stem)}_lt2"
        J2.append(dict(name=f"j2_g1_lt2_{tag}", priority=pri, est_hours=EST["g1_lt2"],
                       argv=_py("eval_g1_dev.py", "--checkpoint", _ck1(stem), "--limit-takes", "2", "--out", out),
                       requires=[_ck1(stem)], outputs=[out]))
        out = f"reports/gates/g3_sim/insim_{_nm1(stem)}"
        J2.append(dict(name=f"j2_insim_{tag}", priority=pri + 1, est_hours=EST["insim"],
                       argv=_py("eval_gait_insim.py", "--checkpoint", _ck1(stem), "--out", out),
                       requires=[_ck1(stem)], outputs=[out]))
        pri += 2
    out = f"reports/gates/g3/curve_{_nm1('ckpt_A_004000')}"
    J2.append(dict(name="j2_g3_curve_A_004000", priority=pri, est_hours=EST["g3_nosec"],
                   argv=_py("eval_g3_dev.py", "--checkpoint", _ck1("ckpt_A_004000"), "--no-secondary", "--out", out),
                   requires=[_ck1("ckpt_A_004000")], outputs=[out]))
    pri += 1
    for stem in ("ckpt_A_002000", "ckpt_A_006000", "ckpt_A_010681", "ckpt_B_008000", "ckpt_B_016000",
                 "ckpt_B_024000", None):
        if stem is None:
            ck, nm, tag, req = "{phase_b_ckpt}", "{phase_b_name}", "B_end", ["{phase_b_ckpt}"]
        else:
            ck, nm, tag, req = _ck1(stem), _nm1(stem), stem[5:], [_ck1(stem)]
        out = f"reports/gates/g1/dev_{nm}_lt2"
        J2.append(dict(name=f"j2_g1_lt2_{tag}", priority=pri, est_hours=EST["g1_lt2"],
                       argv=_py("eval_g1_dev.py", "--checkpoint", ck, "--limit-takes", "2", "--out", out),
                       requires=req, outputs=[out]))
        pri += 1
    for stem in ("ckpt_A_010681", None):
        ck, nm, tag = (_ck1(stem), _nm1(stem), "A_010681") if stem else ("{phase_b_ckpt}", "{phase_b_name}", "B_end")
        for suffix, extra, est in (("xsens13", [], EST["rm_xsens13"]),
                                   ("mix_latent", ["--mixture", "--latent"], EST["rm_mix_latent"])):
            J2.append(dict(name=f"j2_root_motion_{tag}_{suffix}", priority=pri, est_hours=est,
                           argv=_py("probe_root_motion.py", "--checkpoint", ck, "--config", CONFIG1, *extra,
                                    "--tag", suffix, "--out", "reports/bench"),
                           requires=[ck],
                           outputs=[f"reports/bench/root_motion_{nm}_{suffix}.json"]))
            pri += 1
    for j in J2:
        j.update(group="J2", after=list(j1))
    J += J2
    j12 = [j["name"] for j in J]
    for j in J:
        j.update(cls=1, gpu=True)
    train = [PYTHON, "scripts/train_stage1.py"]
    est_a = round(10681 * rates["A"] / 3600.0 + 0.15, 2)
    est_b = round(32044 * rates["B"] / 3600.0 + 0.25, 2)
    J += [
        dict(name="s2_train_A", group="seed2", cls=2, priority=1, gpu=True, training=True, run_dir=RUN2,
             argv=train + ["--phase", "A", "--config", CONFIG2, "--run-dir", RUN2, "--prefetch", "2",
                           "--memory-cap-gb", "14.5"],
             after=j12, soft_after=list(J3_SOFT), soft_deadline=SEED2_SOFT_DEADLINE,
             train_phase="A", train_steps=10681, s_per_step=rates["A"], est_hours=est_a,
             timeout_hours=round(2.0 * est_a, 1), outputs=[f"{RUN2}/ckpt_A_010681.pt"],
             note="seed 2 phase A (seed 20261001); departures from seed 1 in reports/weekend_2026-10-01/tasks/T01.md"),
        dict(name="s2_train_B", group="seed2", cls=2, priority=2, gpu=True, training=True, run_dir=RUN2,
             argv=train + ["--phase", "B", "--config", CONFIG2, "--run-dir", RUN2, "--resume",
                           f"{RUN2}/ckpt_A_010681.pt", "--prefetch", "2", "--memory-cap-gb", "14.5"],
             after=["s2_train_A"], requires=[f"{RUN2}/ckpt_A_010681.pt"],
             train_phase="B", train_steps=32044, s_per_step=rates["B"], est_hours=est_b,
             timeout_hours=round(2.0 * est_b, 1), outputs=[f"{RUN2}/ckpt_B_032044.pt"],
             note="seed 2 phase B from its own phase A end"),
    ]
    return J


def weekend_templates():
    """The cls 1 post-train templates of every training run of the queue (task T01, step 5)."""
    runs = ["*", "!phaseA_root_full"]
    d = "reports/gates/g3/dev_{name}"
    j1 = dict(name="post_j1", cls=1, runs=runs, jobs=[
        dict(name="pt_g3_dev_{run}", priority=1, est_hours=EST["g3"],
             argv=_py("eval_g3_dev.py", "--checkpoint", "{ckpt}", "--out", d), outputs=[d]),
        dict(name="pt_g3_dev_model_{run}", priority=2, est_hours=EST["g3_nosec"],
             argv=_py("eval_g3_dev.py", "--checkpoint", "{ckpt}", "--weights", "model", "--no-secondary",
                      "--out", d + "_model"), outputs=[d + "_model"]),
        dict(name="pt_insim_{run}", priority=3, est_hours=EST["insim"], after=["pt_g3_dev_{run}"],
             argv=_py("eval_gait_insim.py", "--checkpoint", "{ckpt}", "--real", d + "/g3_dev.json",
                      "--out", "reports/gates/g3_sim/insim_{name}"), outputs=["reports/gates/g3_sim/insim_{name}"]),
        dict(name="pt_g1_dev_{run}", priority=4, est_hours=EST["g1"],
             argv=_py("eval_g1_dev.py", "--checkpoint", "{ckpt}", "--out", "reports/gates/g1/dev_{name}"),
             outputs=["reports/gates/g1/dev_{name}"])])
    curve = dict(name="post_curve", cls=1, runs=runs, jobs=[])
    for i, s in enumerate(("004000", "008000", "016000", "024000")):
        ck, out = "{run_dir}/ckpt_B_" + s + ".pt", "reports/gates/g3/curve_{run}_ckpt_B_" + s
        curve["jobs"].append(dict(name="pt_g3_curve_{run}_B_" + s, priority=10 + i, est_hours=EST["g3_nosec"],
                                  argv=_py("eval_g3_dev.py", "--checkpoint", ck, "--no-secondary", "--out", out),
                                  requires=[ck], outputs=[out]))
    comp = dict(name="post_comparators", cls=1, runs=runs, jobs=[])
    pri = 20
    for key, stem in (("ckpt_a4000", "ckpt_A_004000"), ("ckpt_b4000", "ckpt_B_004000")):
        ck, tag = "{" + key + "}", stem[5:]
        out = "reports/gates/g1/dev_{run}_" + stem + "_lt2"
        comp["jobs"].append(dict(name="pt_g1_lt2_{run}_" + tag, priority=pri, est_hours=EST["g1_lt2"],
                                 argv=_py("eval_g1_dev.py", "--checkpoint", ck, "--limit-takes", "2", "--out", out),
                                 requires=[ck], outputs=[out]))
        out = "reports/gates/g3_sim/insim_{run}_" + stem
        comp["jobs"].append(dict(name="pt_insim_{run}_" + tag, priority=pri + 1, est_hours=EST["insim"],
                                 argv=_py("eval_gait_insim.py", "--checkpoint", ck, "--out", out),
                                 requires=[ck], outputs=[out]))
        pri += 2
    out = "reports/gates/g3/curve_{run}_ckpt_A_004000"
    comp["jobs"].append(dict(name="pt_g3_curve_{run}_A_004000", priority=pri, est_hours=EST["g3_nosec"],
                             argv=_py("eval_g3_dev.py", "--checkpoint", "{ckpt_a4000}", "--no-secondary",
                                      "--out", out), requires=["{ckpt_a4000}"], outputs=[out]))
    for t in (j1, curve, comp):
        for j in t["jobs"]:
            j.setdefault("gpu", True)
            j["group"] = "post_{run}"
    return [j1, curve, comp]


def seed_weekend(tree=TREE, dry_run=False, rates=None, out=None):
    """Write the weekend jobs and templates (jobs already queued are reported and left as they are)."""
    out = sys.stdout if out is None else out
    if rates is None:
        import gpu_queue as GQ              # noqa: E402  (lazy: gpu_queue imports this module)
        rates = GQ.seed_rates(tree)
    jobs, tpls = weekend_jobs(rates), weekend_templates()
    have = existing_names(jobs_dir(tree))
    written = []
    for j in jobs:
        normalize(j)
        if j["name"] in have:
            print(f"queue_job: {j['name']} already queued ({os.path.basename(have[j['name']])}), left as it is", file=out)
            continue
        if dry_run:
            print(f"queue_job: would write {job_filename(normalize(j))}", file=out)
            continue
        written.append(write_job(j, tree=tree))
        print(f"queue_job: wrote {tree_rel(written[-1], tree)}", file=out)
    for t in tpls:
        path = os.path.join(templates_dir(tree), f"{t['name']}.json")
        if os.path.exists(path):
            print(f"queue_job: template {t['name']} exists, left as it is", file=out)
            continue
        if dry_run:
            normalize_template(t)
            print(f"queue_job: would write template {t['name']} ({len(t['jobs'])} jobs)", file=out)
            continue
        written.append(write_template(t, tree=tree))
        print(f"queue_job: wrote {tree_rel(written[-1], tree)}", file=out)
    return written


# ---------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--file", help="a job file to check and queue")
    p.add_argument("--template", help="a post-train template file to check and write")
    p.add_argument("--seed-weekend", action="store_true", help="the J1, J2, seed-2 jobs and post-train templates")
    p.add_argument("--dry-run", action="store_true", help="check and print, write nothing")
    p.add_argument("--replace", action="store_true", help="replace a job or template of the same name")
    p.add_argument("--name")
    p.add_argument("--cls", type=int)
    p.add_argument("--priority", type=int)
    p.add_argument("--cpu", action="store_true", help="a CPU-only step (gpu false)")
    p.add_argument("--training", action="store_true")
    p.add_argument("--run-dir")
    p.add_argument("--group")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    for f in LIST_FIELDS:
        p.add_argument("--" + f.replace("_", "-"), nargs="+", default=None)
    p.add_argument("--soft-deadline")
    p.add_argument("--not-before")
    p.add_argument("--est-hours", type=float)
    p.add_argument("--timeout-hours", type=float)
    p.add_argument("--note")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- SCRIPT ARGS... (after a '--')")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        if a.seed_weekend:
            seed_weekend(dry_run=a.dry_run)
            return 0
        if a.template:
            t = read_json(a.template)
            if a.dry_run:
                normalize_template(t)
                print(f"queue_job: template {t['name']} is valid")
                return 0
            print(f"queue_job: wrote {write_template(t, replace=a.replace)}")
            return 0
        if a.file:
            job = read_json(a.file)
            if job is None:
                raise JobError(f"{a.file}: not a JSON file")
        else:
            cmd = list(a.command)
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            if not cmd:
                raise JobError("no command: give -- SCRIPT ARGS...")
            job = {"name": a.name, "cls": a.cls, "priority": a.priority, "gpu": not a.cpu, "training": a.training,
                   "run_dir": a.run_dir, "argv": [PYTHON] + cmd, "est_hours": a.est_hours,
                   "env": dict(kv.split("=", 1) for kv in a.env)}
            for f in LIST_FIELDS:
                if getattr(a, f) is not None:
                    job[f] = getattr(a, f)
            for f in ("soft_deadline", "not_before", "timeout_hours", "group", "note"):
                if getattr(a, f) is not None:
                    job[f] = getattr(a, f)
        if a.dry_run:
            j = normalize(job)
            print(json.dumps(j, indent=1))
            print(f"queue_job: valid, would write {job_filename(j)}")
            return 0
        print(f"queue_job: wrote {write_job(job, replace=a.replace)}")
        return 0
    except JobError as e:
        print(f"queue_job: refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
