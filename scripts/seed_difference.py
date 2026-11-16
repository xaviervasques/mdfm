"""SEED (2026-10-06): the seed differences of the lines that the F2 lever-2 rule and the JEPA proposal read, at
matched evaluation settings, beside the mismatched ones those documents used. Label of every output: LABEL.

Why. Spec departure 23 (docs/MFM_MODEL_CODE_SPEC_2026-09-28.md): seed 1's G3 evaluations (cache/runs/phaseA_root_full)
ran with --batch-windows 32 under a 12 GB cap, seed 2's (cache/runs/pilot_v1_s2) at the script defaults, batch_windows
16 under 2.4 GB. batch_windows changes the refinement point draws (reports/weekend_2026-10-01/tasks/T04.md, finding 4),
so |seed 1 minus seed 2| of a G3 line mixes the seed with the evaluation settings. The F2 lever-2 rule
(reports/bench/f2_lever2/rule.txt and rule_amendment_1.txt, hashed) and the JEPA proposal
(docs/proposals/JEPA_EXPLORATION_DRAFT.md, section 3 and the hashed addendum 3a) build margins from those differences.
Task SEED queued seed 2's G3 dev, diag and curve evaluations again at seed 1's settings into new folders (suffix
MATCHED, '_bw32'), and the G4 dev evaluation of both B ends under the harness of task G4H (suffix G4H, '_g4h'), queue
group SEED (RERUNS). Three code controls of the same group (CONTROLS, suffix CONTROL, '_bw32_now') repeat seed 1's own
B-end diag and dev and its ckpt_B_004000 curve point at seed 1's own argv under the evaluation code of the reruns:
seed 1's folders hold the code of 2026-10-01/02, the reruns run today's. This script reads them beside the folders the
documents read.

The hashed F2 lever-2 rule keeps its margins: the matched differences are reported beside its reading, never
substituted. The same holds for the JP0 reading of the JEPA addendum (hashed) and for the margins that section 3 of
the JEPA proposal prints.

Lines (LINES), each with seed 1, seed 2 as the documents read it (mismatched), seed 2 at seed 1's settings (matched)
and, where a control exists, seed 1 under today's code (s1_now); d = |seed 1 minus seed 2| (d_mismatched,
d_matched), d_same_code = |s1_now minus seed 2 matched| (both seeds at seed 1's settings under today's code; None
unless the control and the rerun recorded one code_sha256), the effect of the rerun, seed 2 matched minus seed 2
mismatched, and the code effect on seed 1, s1_now minus seed 1. The rerun's effect holds the settings alone
('settings') when the rerun recorded the code_sha256 of the folder it repeats or when the control, run under the
rerun's code, reproduces seed 1 (TOL); otherwise it holds 'settings and code', and summary.md says so:
- B end (phaseA_root_full_ckpt_B_032044 against pilot_v1_s2_ckpt_B_032044): S(12), V(12), S(14), V(14) from the base
  row of diag.json (subsets no_whc003_whc026 and all14, as the seeds file); the walkway IC, stride_mape_pct and the
  insole IC from results_dev of g3_dev.json (point); G1a of the full G1 dev runs.
- ckpt_B_004000: S(12), V(12), S(14), V(14) from per_person.csv of the curve folders (scripts/weekend_branch.py
  g3set_from_folder, as the seeds file); the walkway and insole IC from results_dev of their g3_dev.json; G1a of the
  --limit-takes 2 G1 runs.
- G1a needs no rerun: both seeds' G1 runs used --memory-gb 12 --batch-windows 64 --block-windows 64 (jobs
  j1_g1_dev_B and pt_g1_dev_pilot_v1_s2_r2, and the lt2 pair). The script checks the inference blocks and reports
  the line as matched already; its d_same_code is d_matched when both G1 folders recorded one code_sha256.
Uses (USES): every margin of those documents that a seed difference enters, max(fraction x seed 1's line, 2 x d),
computed with the mismatched d (the documents' value), with the matched d and with the same-code d, beside each other,
with the threshold the document prints (seed 1's line minus the margin for a success line, plus it for a
non-inferiority line).
JP0 (the hashed addendum 3a of the JEPA proposal): relations D (latent probe against the decoded row) and R (latent
against the decoded row divided by F1's stride gain) in the pass unit; d = max(|X1 - X2|, |Y1 - Y2|),
m = max(0.10 Y1, 2 d). The decoded row of seed 2 is recomputed from events.csv of the matched diag with the functions
of scripts/jp0_read.py (decoded_passes, line_of, relation); the latent rows come from per_pass.csv of the J0 folders
(the probes' settings are matched already: the three J0 jobs share one argv apart from the checkpoint). R stays
partly matched: F1's stride gain of seed 2 was fitted on a --batch-windows 16 train54 decode (job
f1_gain_decode_s2_B_r2), which no SEED job reruns. The same-code relations take seed 1's decoded row from the control
diag as well.
G4 (reported, no document margin): grf_rmse, grf_r_gain, emg_r, emg_r_gain, the metric-3D MPJPE and PA-MPJPE and the
GRF row without the low-peak series, of both B ends before task G4H (the folders of 2026-10-04) and after it (suffix
_g4h): the seed difference under each harness and the harness effect per seed.
Reported only: the other curve points (B_008000, B_016000, B_024000, A_004000) and the raw-weight dev runs.
Stated beside the numbers, never a stop: the MoveAtlas HEAD and the records drift of each rerun and control folder
against the folders it pairs with (drift_of: checkpoint.drift of its report json and, when it names records_sha256,
the drift_log.jsonl entry that its checkpoint load appended beside the checkpoint, mdfm/train/checkpoint.py: the
changed sources and columns, and which of them the gate reads, GATE_SOURCES), and the evaluation code then and now
(code_block: the code_sha256 each folder's _provenance.json recorded over its code_files, and the same hash over those
files today, the code a rerun runs). A record or code change between the folders would enter the settings effect.
Beside the F2 lever-2 reading: when reports/bench/f2_lever2/results.json exists, each line of R1, R2 and the early
read with its recorded status under the hashed margin and, beside it, the status under the margin built from the
matched d and from the same-code d (R1 and early: point at or above the margin and the interval's lower bound above
0; R2: difference at or below the margin); a status that would change is named, and the rule's reading stands.

Checks (exit EXIT_INPUT): the dev sets of reports/bench/g3_gain_check/results.json hash to the values that the F2
lever-2 rule quotes (section 3), and every person of a G3 row read lies in them; the mismatched S and V differences
equal those of reports/gpu_queue/branch_seeds_20261002_211426.json (TOL); the JP0 values recomputed from the
mismatched folders equal reports/bench/jp0_reading/results.json (JP0_TOL_Y on the decoded rows, JP0_TOL_X on the
latent rows, whose per-pass values come from per_pass.csv and differ from JP0's refit by under 1e-5 cm). Each rerun
folder (exit EXIT_SETTINGS otherwise): of the same checkpoint (name and SHA-256) and weights as the folder it
repeats; a G3 rerun at seed 1's settings (SETTING_KEYS of the inference block and memory.cap_gb equal to seed 1's
folder, batch_windows 32 and cap 12 GB); a G4 rerun with the settings of the folder it repeats and the harness of
G4H (config keys G4H_CONFIG_KEYS); a control of seed 1's checkpoint at seed 1's settings, the same checks.

Exit codes: EXIT_OK (0) every rerun and control folder present and matched, the outputs written (or printed with
--dry-run); EXIT_INPUT (2) an input the documents read is missing or inconsistent; EXIT_WAITING (3) a folder of RERUNS
or CONTROLS does not exist yet (each of its files is needed, _provenance.json included), listed (nothing written;
--partial prints what exists); EXIT_SETTINGS (4) a rerun or control folder exists but is not what its job asked;
EXIT_EXISTS (5) <out>/results.json exists (pass --out for a new folder).

Writes <out> (default reports/bench/seed_difference/): results.json, summary.md, _provenance.json (the MoveAtlas stamp
of scripts/moveatlas_stamp.py, the SHA-256 of every input and of the code). The README.md of that folder is not
written by this script. No lockbox, EVAL, CODY or care_pd record; CPU only (CUDA_VISIBLE_DEVICES=-1 before numpy
loads); no GPU job is run or queued here.

Run (mdfm environment):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 \\
    <conda>/envs/mdfm/python.exe scripts/seed_difference.py [--status | --partial | --dry-run]
    [--out DIR]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # never the GPU

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
if TREE not in sys.path:
    sys.path.insert(0, TREE)
LABEL = ("dev diagnostic, not a gate read: seed differences at matched evaluation settings, reported beside the "
         "documents' margins, never substituted")
TASK = "SEED"
OUT_REL = "reports/bench/seed_difference"
EXIT_OK, EXIT_INPUT, EXIT_WAITING, EXIT_SETTINGS, EXIT_EXISTS = 0, 2, 3, 4, 5
TOL = 1e-9
JP0_TOL_Y = 1e-6                    # cm, the decoded and recalibrated rows (same file, same arithmetic)
JP0_TOL_X = 1e-4                    # cm, the latent row (per_pass.csv against JP0's refit)
MOVEATLAS_DEFAULT = "<moveatlas>"

S1_RUN, S2_RUN = "phaseA_root_full", "pilot_v1_s2"
S1_B, S2_B = f"{S1_RUN}_ckpt_B_032044", f"{S2_RUN}_ckpt_B_032044"
CURVE_STEPS = ("ckpt_B_004000", "ckpt_B_008000", "ckpt_B_016000", "ckpt_B_024000", "ckpt_A_004000")
EARLY = "ckpt_B_004000"
MATCHED, G4H, CONTROL = "_bw32", "_g4h", "_bw32_now"
SEED1_SETTINGS = {"batch_windows": 32, "cap_gb": 12.0}       # argv of j1_g3_dev_B, diag_B and j2_g3_curve_* (job_log)
SETTING_KEYS = ("refine_steps", "points_per_stream", "window_s", "hop_s", "batch_windows", "seed", "shape_pooling",
                "weights", "amp_bf16")
G1_SETTING_KEYS = ("refine_steps", "points_per_stream", "latent_samples", "decoder_draws", "batch_windows",
                   "block_windows", "seed", "shape_pooling", "weights", "amp_bf16")
G4H_CONFIG_KEYS = ("emg_gain", "emg_min_s", "low_peak_bw")    # g4.json config keys the G4H harness writes
# the record sources each gate reads (mdfm/gates/g3.py SOURCE; mdfm/gates/g4.py STROKE and GAIT120)
GATE_SOURCES = {"g3": ("weargait_pd",), "g4": ("stroke_gait_vancriekinge", "gait120")}
DRIFT_LOG = "drift_log.jsonl"                                   # beside the checkpoint (mdfm/train/checkpoint.py)
TAGS = ("matched", "same_code")                                 # the d set beside the documents' (mismatched) one
SUB12, SUB14 = "no_whc003_whc026", "all14"
EXCLUDED_FROM_12 = ("weargait:WHC003", "weargait:WHC026")
DEV14_SHA256 = "31cd3cd41c548cc467275bc1556b0a9f8ba11fe6dc5a3fa3c4acd2bb7acd4267"   # F2 lever-2 rule, section 3
DEV12_SHA256 = "09f1173e34474ae9053a2e74efa1680c07ca38c664fbd30e7c0f01481355338a"
SEEDS_REL = "reports/gpu_queue/branch_seeds_20261002_211426.json"
F1_RESULTS_REL = "reports/bench/g3_gain_check/results.json"
JP0_RESULTS_REL = "reports/bench/jp0_reading/results.json"
F2L2_RESULTS_REL = "reports/bench/f2_lever2/results.json"
JOB_LOG_REL = "reports/gpu_queue/job_log.csv"
G3_REL, G4_REL = "reports/gates/g3", "reports/gates/g4"

# (job name, folder, files that must exist, kind, folder it repeats, seed 1's folder of the same settings); the ten
# jobs of queue group SEED. _provenance.json is the last file each evaluation script writes: it marks a finished run.
G3_DEV_FILES = ("g3_dev.json", "per_person.csv", "_provenance.json")
G3_SIM_FILES = ("g3_sim.json", "per_person.csv", "_provenance.json")      # a phase A checkpoint
RERUNS = (
    ("seed_s2_g3_diag_bw32", f"{G3_REL}/diag_{S2_B}{MATCHED}", ("diag.json", "events.csv", "_provenance.json"),
     "g3", f"{G3_REL}/diag_{S2_B}", f"{G3_REL}/diag_{S1_B}"),
    ("seed_s2_g3_dev_bw32", f"{G3_REL}/dev_{S2_B}{MATCHED}", G3_DEV_FILES, "g3", f"{G3_REL}/dev_{S2_B}",
     f"{G3_REL}/dev_{S1_B}"),
    ("seed_s2_g3_curve_B_004000_bw32", f"{G3_REL}/curve_{S2_RUN}_ckpt_B_004000{MATCHED}", G3_DEV_FILES, "g3",
     f"{G3_REL}/curve_{S2_RUN}_ckpt_B_004000", f"{G3_REL}/curve_{S1_RUN}_ckpt_B_004000"),
    ("seed_v1_g4_dev_g4h", f"{G4_REL}/dev_{S1_B}{G4H}", ("g4.json", "_provenance.json"), "g4", f"{G4_REL}/dev_{S1_B}",
     None),
    ("seed_s2_g4_dev_g4h", f"{G4_REL}/dev_{S2_B}{G4H}", ("g4.json", "_provenance.json"), "g4", f"{G4_REL}/dev_{S2_B}",
     None),
) + tuple(
    (f"seed_s2_g3_curve_{s[5:]}_bw32", f"{G3_REL}/curve_{S2_RUN}_{s}{MATCHED}",
     G3_SIM_FILES if s.startswith("ckpt_A") else G3_DEV_FILES, "g3",
     f"{G3_REL}/curve_{S2_RUN}_{s}", f"{G3_REL}/curve_{S1_RUN}_{s}") for s in CURVE_STEPS[1:]
) + (
    ("seed_s2_g3_dev_model_bw32", f"{G3_REL}/dev_{S2_B}_model{MATCHED}", G3_DEV_FILES, "g3",
     f"{G3_REL}/dev_{S2_B}_model", f"{G3_REL}/dev_{S1_B}_model"),
)
# the code controls of queue group SEED, same layout: seed 1's own evaluation at its own argv under today's code; the
# folder it repeats is seed 1's folder, which also sets the settings
CONTROLS = (
    ("seed_v1_g3_diag_now", f"{G3_REL}/diag_{S1_B}{CONTROL}", ("diag.json", "events.csv", "_provenance.json"), "g3",
     f"{G3_REL}/diag_{S1_B}", f"{G3_REL}/diag_{S1_B}"),
    ("seed_v1_g3_dev_now", f"{G3_REL}/dev_{S1_B}{CONTROL}", G3_DEV_FILES, "g3", f"{G3_REL}/dev_{S1_B}",
     f"{G3_REL}/dev_{S1_B}"),
    ("seed_v1_g3_curve_B_004000_now", f"{G3_REL}/curve_{S1_RUN}_{EARLY}{CONTROL}", G3_DEV_FILES, "g3",
     f"{G3_REL}/curve_{S1_RUN}_{EARLY}", f"{G3_REL}/curve_{S1_RUN}_{EARLY}"),
)
JOBS = RERUNS + CONTROLS

# id: (label, end, reader, argument, unit, decimals)
LINES = {
    "S12_B": ("S(12), stride MAE on the 12", "B end", "diag", ("S", SUB12), "cm", 2),
    "V12_B": ("V(12), speed MAE on the 12", "B end", "diag", ("V", SUB12), "m/s", 4),
    "S14_B": ("S(14), stride MAE on the 14", "B end", "diag", ("S", SUB14), "cm", 2),
    "V14_B": ("V(14), speed MAE on the 14", "B end", "diag", ("V", SUB14), "m/s", 4),
    "IC_B": ("walkway IC median", "B end", "dev", "ic_median_ms_walkway", "ms", 2),
    "MAPE_B": ("stride_mape_pct", "B end", "dev", "stride_mape_pct", "percent", 2),
    "ICins_B": ("insole IC median", "B end", "dev", "ic_median_ms_insole", "ms", 2),
    "G1a_B": ("G1a, full G1 dev run", "B end", "g1", "full", "mm", 2),
    "S12_4000": ("S(12), stride MAE on the 12", EARLY, "curve", ("S", SUB12), "cm", 2),
    "V12_4000": ("V(12), speed MAE on the 12", EARLY, "curve", ("V", SUB12), "m/s", 4),
    "S14_4000": ("S(14), stride MAE on the 14", EARLY, "curve", ("S", SUB14), "cm", 2),
    "V14_4000": ("V(14), speed MAE on the 14", EARLY, "curve", ("V", SUB14), "m/s", 4),
    "IC_4000": ("walkway IC median", EARLY, "curve_dev", "ic_median_ms_walkway", "ms", 2),
    "ICins_4000": ("insole IC median", EARLY, "curve_dev", "ic_median_ms_insole", "ms", 2),
    "G1a_4000": ("G1a, --limit-takes 2 G1 run", EARLY, "g1", "lt2", "mm", 2),
}
# seeds file keys of the lines it holds: {line id: (section, key)}
SEEDS_KEYS = {"S12_B": ("B_end", "S12"), "V12_B": ("B_end", "V12"), "S14_B": ("B_end", "S14"),
              "V14_B": ("B_end", "V14"), "S12_4000": ("ckpt_B_004000", "S12"), "V12_4000": ("ckpt_B_004000", "V12"),
              "S14_4000": ("ckpt_B_004000", "S14"), "V14_4000": ("ckpt_B_004000", "V14")}

F2L2_DOC = ("F2 lever-2 rule (reports/bench/f2_lever2/rule.txt with rule_amendment_1.txt, hashed): keeps its margins; "
            "the matched values are reported beside its reading, never substituted")
JEPA_DOC = ("JEPA proposal, section 3 (docs/proposals/JEPA_EXPLORATION_DRAFT.md, DRAFT, E1 not run): the margins it "
            "prints stand until the owner changes them; the matched values are reported beside")
# (document, item, line id, fraction, kind): kind 'success' (threshold line minus margin), 'noninf' (line plus
# margin) or 'reported' (d beside, no margin)
USES = (
    ("F2L2", "R1 success margin m_S (section 7)", "S12_B", 0.10, "success"),
    ("F2L2", "R1 success margin m_V (section 7)", "V12_B", 0.10, "success"),
    ("F2L2", "R2 non-inferiority S(14) (section 7)", "S14_B", 0.05, "noninf"),
    ("F2L2", "R2 non-inferiority V(14) (section 7)", "V14_B", 0.05, "noninf"),
    ("F2L2", "R2 non-inferiority walkway IC (section 7)", "IC_B", 0.05, "noninf"),
    ("F2L2", "R2 non-inferiority G1a (section 7)", "G1a_B", 0.05, "noninf"),
    ("F2L2", "R2 non-inferiority stride_mape_pct (amendment A1)", "MAPE_B", 0.05, "noninf"),
    ("F2L2", "insole IC, reported beside its d (amendment A1)", "ICins_B", None, "reported"),
    ("F2L2", "early read S(12), reported only (section 7)", "S12_4000", 0.10, "success"),
    ("F2L2", "early read V(12), reported only (section 7)", "V12_4000", 0.10, "success"),
    ("JEPA", "E1 success S(12)", "S12_B", 0.10, "success"),
    ("JEPA", "E1 not worse V(12)", "V12_B", 0.05, "noninf"),
    ("JEPA", "E1 not worse S(14)", "S14_B", 0.05, "noninf"),
    ("JEPA", "E1 not worse V(14)", "V14_B", 0.05, "noninf"),
    ("JEPA", "E1 not worse walkway IC", "IC_B", 0.05, "noninf"),
    ("JEPA", "E1 not worse insole IC", "ICins_B", 0.05, "noninf"),
    ("JEPA", "E1 not worse G1a", "G1a_B", 0.05, "noninf"),
    ("JEPA", "early check S(12)", "S12_4000", 0.10, "success"),
    ("JEPA", "early check V(12)", "V12_4000", 0.05, "noninf"),
    ("JEPA", "early check S(14)", "S14_4000", 0.05, "noninf"),
    ("JEPA", "early check V(14)", "V14_4000", 0.05, "noninf"),
    ("JEPA", "early check walkway IC", "IC_4000", 0.05, "noninf"),
    ("JEPA", "early check insole IC", "ICins_4000", 0.05, "noninf"),
    ("JEPA", "early check G1a (--limit-takes 2)", "G1a_4000", 0.05, "noninf"),
)
# F2 lever-2 results.json: (block, key in it, line id, fraction)
F2L2_BESIDE = (("R1", "S12", "S12_B", 0.10), ("R1", "V12", "V12_B", 0.10), ("R2", "S14", "S14_B", 0.05),
               ("R2", "V14", "V14_B", 0.05), ("R2", "IC", "IC_B", 0.05), ("R2", "G1a", "G1a_B", 0.05),
               ("R2", "MAPE", "MAPE_B", 0.05), ("early", "S12", "S12_4000", 0.10), ("early", "V12", "V12_4000", 0.10))
G4_LINES = ("grf_rmse", "grf_r_gain", "emg_r", "emg_r_gain")
G4_UNITS = {"grf_rmse": "percent BW", "grf_r_gain": "r", "emg_r": "r", "emg_r_gain": "r", "mpjpe_mm": "mm",
            "pa_mpjpe_mm": "mm", "grf_rmse_no_low_peak": "percent BW"}
REPORTED_DEV = ("stride_mae_cm", "speed_mae_m_s", "stride_mape_pct", "ic_median_ms_walkway", "ic_median_ms_insole")


class Exit(Exception):
    """A stop of the run with its exit code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = int(code)
        self.message = message


# ---------------------------------------------------------------- small helpers
def fin(x):
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2):
    v = fin(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def clean(o):
    """JSON-ready: non-finite floats as null, numpy scalars as Python values, tuple keys joined."""
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


def rel(path, tree):
    try:
        r = os.path.relpath(os.path.abspath(path), tree)
    except ValueError:                                # another drive
        return str(path).replace("\\", "/")
    return r.replace("\\", "/")


def read_json(path):
    """The parsed file, or None when it is absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def person_hash(ids):
    """SHA-256 of the sorted ids joined by newlines (mdfm.data.sources.person_hash)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _sibling(name):
    key = f"_seed_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def WB():
    """scripts/weekend_branch.py: the G3 and G1 folder readers of the seeds file."""
    return _sibling("weekend_branch")


def JP():
    """scripts/jp0_read.py: decoded_passes, line_of, relation of the JP0 reading."""
    return _sibling("jp0_read")


# ---------------------------------------------------------------- sets and guards
def load_sets(tree):
    """{dev14, dev12} of F1's results.json, hashed to the values that the F2 lever-2 rule quotes (section 3), dev12
    equal to dev14 without WHC003 and WHC026."""
    doc = read_json(os.path.join(tree, F1_RESULTS_REL))
    if doc is None:
        raise Exit(EXIT_INPUT, f"missing: {F1_RESULTS_REL}")
    s = doc.get("sets") or {}
    sets = {"dev14": sorted(s.get("dev14") or []), "dev12": sorted(s.get("dev12") or [])}
    for k, want in (("dev14", DEV14_SHA256), ("dev12", DEV12_SHA256)):
        if person_hash(sets[k]) != want:
            raise Exit(EXIT_INPUT, f"{F1_RESULTS_REL}: set {k} does not hash to the value of the F2 lever-2 rule")
    if set(sets["dev12"]) != set(sets["dev14"]) - set(EXCLUDED_FROM_12):
        raise Exit(EXIT_INPUT, f"{F1_RESULTS_REL}: dev12 is not dev14 without WHC003 and WHC026")
    return sets


def guard_persons(gset, sets, what):
    """Every person of a G3 set (weekend_branch layout) lies in dev14, those of the 12-subset in dev12."""
    if gset is None:
        return
    for sub, allowed in ((SUB14, sets["dev14"]), (SUB12, sets["dev12"])):
        s = (gset.get("subsets") or {}).get(sub) or {}
        persons = set(s.get("S") or {}) | set(s.get("V") or {})
        out = sorted(persons - set(allowed))
        if out:
            raise Exit(EXIT_INPUT, f"{what}: {len(out)} persons of the {sub} rows outside the dev controls")


# ---------------------------------------------------------------- folders and settings
def folder(tree, kind, name, suffix=""):
    """reports/gates/g3/<kind>_<name><suffix> under tree."""
    return os.path.join(tree, G3_REL, f"{kind}_{name}{suffix}")


def json_of(path):
    """The report json of a G3 or G4 folder (diag.json, g3_dev.json, g4.json), or None."""
    for n in ("diag.json", "g3_dev.json", "g3_sim.json", "g4.json"):
        p = os.path.join(path, n)
        if os.path.exists(p):
            return p
    return None


def finished(path):
    """A folder whose evaluation finished: its _provenance.json, the last file the scripts write, exists."""
    return os.path.exists(os.path.join(path, "_provenance.json"))


def head_of(path):
    """The MoveAtlas HEAD a folder was measured against: moveatlas_head of its _provenance.json, else the head of its
    moveatlas_stamp, else moveatlas_head of its report json; None when none is recorded."""
    prov = read_json(os.path.join(path, "_provenance.json")) or {}
    head = prov.get("moveatlas_head") or (prov.get("moveatlas_stamp") or {}).get("head")
    return head or (doc_of(path) or {}).get("moveatlas_head")


def code_of(path):
    """code_sha256 that a folder's _provenance.json records (the evaluation code), or None."""
    return (read_json(os.path.join(path, "_provenance.json")) or {}).get("code_sha256")


def drift_of(tree, path, doc, kind):
    """The records drift of a folder's checkpoint load: {'drift': checkpoint.drift of its report json, 'records': None
    when that list does not name records_sha256, else the evidence of the drift_log.jsonl entry beside the checkpoint
    (the last records_drift entry of that checkpoint at or before the folder's run_at): the records hashes, the
    changed sources with their columns, the run's sources and the changed sources that the gate reads
    (GATE_SOURCES[kind]); {'reason': ...} when no entry is found}."""
    ck = (doc or {}).get("checkpoint") or {}
    names = [str(x) for x in ck.get("drift") or []]
    out = {"drift": names, "records": None}
    if "records_sha256" not in names:
        return out
    cp = str(ck.get("path") or "").replace("\\", "/").strip("/")
    log = os.path.join(tree, os.path.dirname(cp), DRIFT_LOG) if cp else None
    run_at = str((read_json(os.path.join(path, "_provenance.json")) or {}).get("run_at") or "")
    best = None
    if log and os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("event") != "records_drift":
                    continue
                if not str(e.get("checkpoint") or "").replace("\\", "/").endswith("/" + cp):
                    continue
                if run_at and str(e.get("time") or "") > run_at:
                    continue
                best = e
    if best is None:
        out["records"] = {"reason": f"no records_drift entry of {cp or 'an unnamed checkpoint'} in "
                                    f"{rel(log, tree) if log else 'no log'} at or before {run_at or 'the run'}"}
        return out
    ev = best.get("evidence") or {}
    changes = [{"source": c.get("source"), "kind": c.get("kind"), "columns": list(c.get("columns") or []),
                "rows": c.get("rows")} for c in ev.get("changes") or []]
    changed = sorted({str(c["source"]) for c in changes if c["source"]})
    out["records"] = {"log": rel(log, tree), "time": best.get("time"), "records_sha256": ev.get("records_sha256"),
                      "policy": ev.get("policy"), "run_sources": ev.get("run_sources"), "changes": changes,
                      "changed_sources": changed, "gate_sources": list(GATE_SOURCES[kind]),
                      "gate_sources_changed": sorted(set(changed) & set(GATE_SOURCES[kind]))}
    return out


def doc_of(path):
    """The parsed report json of a folder, or None."""
    js = json_of(path)
    return None if js is None else read_json(js)


def settings_of(doc, keys=SETTING_KEYS):
    """{key: value} of the inference block, with cap_gb (memory.cap_gb)."""
    inf = (doc or {}).get("inference") or {}
    out = {k: inf.get(k) for k in keys}
    out["cap_gb"] = fin((inf.get("memory") or {}).get("cap_gb"))
    return out


def ckpt_of(doc):
    ck = (doc or {}).get("checkpoint") or {}
    return {"name": ck.get("name"), "sha256": ck.get("sha256"),
            "weights": ck.get("weights") or ((doc or {}).get("inference") or {}).get("weights")}


def settings_differ(a, b):
    """['key a against b'] where two settings dicts differ."""
    return [f"{k} {a.get(k)} against {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def job_rows(tree):
    """{job name: last job_log.csv row} (empty when the log is absent)."""
    p = os.path.join(tree, JOB_LOG_REL)
    out = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[r.get("name")] = r
    return out


def rerun_status(tree):
    """One row per job of JOBS (RERUNS, then CONTROLS): present (every file exists), problems (EXIT_SETTINGS), the
    settings read, the MoveAtlas head, records drift and code hash beside those of the folders it pairs with."""
    jobs = job_rows(tree)
    rows = []
    for job, fold, files, kind, original, seed1 in JOBS:
        p = os.path.join(tree, fold)
        present = all(os.path.exists(os.path.join(p, f)) for f in files)
        row = {"job": job, "role": "control" if fold.endswith(CONTROL) else "rerun", "kind": kind, "folder": fold,
               "files": list(files), "present": present, "problems": [], "repeats": original,
               "seed1_counterpart": seed1,
               "job_log": {k: (jobs.get(job) or {}).get(k) for k in ("state", "exit_code", "start", "end")}
               if job in jobs else None}
        if not present:
            row["missing"] = [f for f in files if not os.path.exists(os.path.join(p, f))]
            rows.append(row)
            continue
        doc = doc_of(p)
        orig = doc_of(os.path.join(tree, original))
        if orig is None:
            raise Exit(EXIT_INPUT, f"missing: the folder {original} that {job} repeats")
        a, b = ckpt_of(doc), ckpt_of(orig)
        for k in ("name", "sha256", "weights"):
            if a[k] != b[k]:
                row["problems"].append(f"checkpoint {k} {a[k]} against {b[k]} in {original}")
        row["checkpoint"] = a
        s1doc = doc_of(os.path.join(tree, seed1)) if seed1 else None
        # the MoveAtlas state each folder was measured against and the records drift its checkpoint load logged:
        # stated beside the numbers (a changed record set would enter the settings effect), never a stop
        row["moveatlas_head"] = {"rerun": head_of(p), "repeats": head_of(os.path.join(tree, original)),
                                 "seed1": head_of(os.path.join(tree, seed1)) if seed1 else None}
        row["drift"] = {"rerun": ((doc or {}).get("checkpoint") or {}).get("drift"),
                        "repeats": (orig.get("checkpoint") or {}).get("drift")}
        row["records_drift"] = {"rerun": drift_of(tree, p, doc, kind)["records"],
                                "repeats": drift_of(tree, os.path.join(tree, original), orig, kind)["records"]}
        heads = {v for v in row["moveatlas_head"].values() if v}
        row["same_moveatlas_head"] = len(heads) <= 1
        # the evaluation code hash each folder recorded (_provenance.json code_sha256): code changed between the
        # folders enters the settings effect as well, stated beside the numbers, never a stop
        row["code_sha256"] = {"rerun": code_of(p), "repeats": code_of(os.path.join(tree, original)),
                              "seed1": code_of(os.path.join(tree, seed1)) if seed1 else None}
        codes = {v for v in row["code_sha256"].values() if v}
        row["same_code"] = len(codes) <= 1
        if kind == "g3":
            if s1doc is None:
                raise Exit(EXIT_INPUT, f"missing: seed 1's folder {seed1}")
            mine, s1 = settings_of(doc), settings_of(s1doc)
            row["settings"], row["seed1_settings"], row["mismatched_settings"] = mine, s1, settings_of(orig)
            row["problems"] += [f"settings: {x} (seed 1)" for x in settings_differ(mine, s1)]
            for k, v in SEED1_SETTINGS.items():
                if s1.get(k) != v:
                    row["problems"].append(f"seed 1's folder {seed1}: {k} {s1.get(k)}, expected {v}")
        else:
            mine, old = settings_of(doc), settings_of(orig)
            row["settings"], row["mismatched_settings"] = mine, old
            row["problems"] += [f"settings: {x} (folder repeated)" for x in settings_differ(mine, old)]
            conf = (doc or {}).get("config") or {}
            miss = [k for k in G4H_CONFIG_KEYS if k not in conf]
            if miss:
                row["problems"].append(f"not the G4H harness: config without {', '.join(miss)}")
            if (doc or {}).get("quick"):
                row["problems"].append("a --quick run (subsets)")
        rows.append(row)
    return rows


# ---------------------------------------------------------------- G3 and G1 lines
def diag_set(tree, path, sets):
    """(G3 set of the diag base row, checkpoint, settings) of a diag folder; Exit(EXIT_INPUT) when unreadable."""
    p = os.path.join(path, "diag.json")
    doc = read_json(p)
    if doc is None:
        raise Exit(EXIT_INPUT, f"missing: {rel(p, tree)}")
    s = WB().g3set_from_diag_row((doc.get("rows") or {}).get("base"), p)
    if s is None:
        raise Exit(EXIT_INPUT, f"no base row in {rel(p, tree)}")
    guard_persons(s, sets, rel(p, tree))
    return s, ckpt_of(doc), settings_of(doc), doc.get("moveatlas_head")


def dev_doc(tree, path):
    """g3_dev.json of a G3 dev folder, g3_sim.json for a phase A checkpoint (scripts/eval_g3_dev.py)."""
    for n in ("g3_dev.json", "g3_sim.json"):
        doc = read_json(os.path.join(path, n))
        if doc is not None:
            return doc
    raise Exit(EXIT_INPUT, f"missing: {rel(os.path.join(path, 'g3_dev.json'), tree)} (nor g3_sim.json)")


def dev_line(doc, line):
    for r in (doc or {}).get("results_dev") or []:
        if isinstance(r, dict) and r.get("line") == line:
            return fin(r.get("point"))
    return None


def curve_set(tree, path, sets):
    s = WB().g3set_from_folder(path)
    if s is None:
        raise Exit(EXIT_INPUT, f"no per_person.csv beside the G3 json of {rel(path, tree)}")
    guard_persons(s, sets, rel(path, tree))
    return s


def g1_of(tree, name, variant):
    """(point, settings, checkpoint, json path) of the G1 dev run of a checkpoint name in a variant."""
    g = WB().g1_reads(tree, name).get(variant)
    if g is None or g.get("point") is None:
        raise Exit(EXIT_INPUT, f"no G1 dev run ({variant}) of {name} under reports/gates/g1/")
    doc = read_json(g["json"])
    return g["point"], settings_of(doc, G1_SETTING_KEYS), ckpt_of(doc), g["json"]


def value_of(tree, lid, who, sets, cache):
    """(value, source path, settings, checkpoint) of a line for who in ('s1', 's2', 's2m', 's1now'); value None with
    source None when the matched or control folder does not exist (or, for 's1now', a G1 line: no control). The source
    is the folder (G3) or the g1.json (G1), tree-relative."""
    label, end, reader, arg, unit, nd = LINES[lid]
    run = S1_RUN if who in ("s1", "s1now") else S2_RUN
    name = f"{run}_ckpt_B_032044" if end == "B end" else f"{run}_{EARLY}"
    suffix = {"s2m": MATCHED, "s1now": CONTROL}.get(who, "")
    if who == "s1now" and reader == "g1":
        return None, None, None, None
    if reader == "g1":
        key = ("g1", name, arg)
        if key not in cache:
            cache[key] = g1_of(tree, name, arg)
        v, st, ck, js = cache[key]
        return v, rel(js, tree), st, ck
    kind = {"diag": "diag", "dev": "dev"}.get(reader, "curve")
    path = folder(tree, kind, name, suffix)
    if who in ("s2m", "s1now") and not finished(path):
        return None, None, None, None
    key = (reader, path)
    if key not in cache:
        if reader == "diag":
            s, ck, st, _ = diag_set(tree, path, sets)
            cache[key] = (s, ck, st)
        else:
            doc = dev_doc(tree, path)
            obj = curve_set(tree, path, sets) if reader == "curve" else doc
            cache[key] = (obj, ckpt_of(doc), settings_of(doc))
    obj, ck, st = cache[key]
    if reader in ("diag", "curve"):
        k, sub = arg
        v = ((obj["subsets"].get(sub) or {}).get(f"{k}_point"))
    else:
        v = dev_line(obj, arg)
    if fin(v) is None:
        raise Exit(EXIT_INPUT, f"{lid}: no finite value in {rel(path, tree)}")
    return fin(v), rel(path, tree), st, ck


def folder_code(tree, src):
    """code_sha256 of the folder of a tree-relative source (a folder, or the g1.json in one); None when absent."""
    if not src:
        return None
    p = os.path.join(tree, src)
    return code_of(os.path.dirname(p) if src.endswith(".json") else p)


def g3_lines(tree, sets):
    """{line id: row} of LINES: s1, s2 (mismatched), s2m (matched), s1_now (control), d_mismatched, d_matched,
    d_same_code, settings_effect_s2 (seed 2 matched minus mismatched), code_effect_s1 (seed 1 now minus then) and
    effect_holds ('settings', 'settings and code', None without a matched value)."""
    cache, out = {}, {}
    for lid, (label, end, reader, arg, unit, nd) in LINES.items():
        v1, p1, st1, ck1 = value_of(tree, lid, "s1", sets, cache)
        v2, p2, st2, ck2 = value_of(tree, lid, "s2", sets, cache)
        row = {"label": label, "end": end, "unit": unit, "decimals": nd, "s1": v1, "s2_mismatched": v2,
               "sources": {"s1": p1, "s2": p2},
               "settings": {"s1": st1, "s2": st2}, "checkpoints": {"s1": ck1, "s2": ck2},
               "d_mismatched": abs(v1 - v2)}
        v4 = None
        if reader == "g1":
            diff = settings_differ(st1, st2)
            row["matched_already"] = not diff
            row["settings_note"] = ("matched already: both G1 runs share their settings" if not diff else
                                    "NOT matched and no rerun queued: " + "; ".join(diff))
            v3 = v2 if not diff else None
            row["sources"]["s2m"] = row["sources"]["s2"] if not diff else None
        else:
            row["matched_already"] = False
            row["settings_note"] = "mismatched settings: " + "; ".join(settings_differ(st1, st2)) \
                if settings_differ(st1, st2) else "the two folders already share their settings"
            v3, p3, st3, ck3 = value_of(tree, lid, "s2m", sets, cache)
            row["sources"]["s2m"] = p3
            row["settings"]["s2m"] = st3
            row["checkpoints"]["s2m"] = ck3
            v4, p4, st4, ck4 = value_of(tree, lid, "s1now", sets, cache)
            row["sources"]["s1now"] = p4
            row["settings"]["s1now"] = st4
            row["checkpoints"]["s1now"] = ck4
        codes = {k: folder_code(tree, p) for k, p in row["sources"].items()}
        row["code_sha256"] = codes
        row["s2_matched"] = v3
        row["s1_now"] = v4
        row["d_matched"] = None if v3 is None else abs(v1 - v3)
        row["settings_effect_s2"] = None if v3 is None else v3 - v2
        row["code_effect_s1"] = None if v4 is None else v4 - v1
        if reader == "g1":
            # no rerun and no control: the G1 runs' own code hashes say whether d_matched is a same-code d
            same = row["matched_already"] and codes.get("s1") is not None and codes.get("s1") == codes.get("s2")
            row["d_same_code"] = row["d_matched"] if same else None
            row["effect_holds"] = None
            row["code_note"] = ("both G1 runs recorded one code_sha256" if same else
                                "the G1 runs recorded different or no code_sha256: no same-code d")
        else:
            # the control and the rerun must have run one evaluation code for a same-code d or a reproduction to count
            pair = codes.get("s1now") is not None and codes.get("s1now") == codes.get("s2m")
            row["control_same_code_as_rerun"] = None if v3 is None or v4 is None else pair
            row["d_same_code"] = None if v3 is None or v4 is None or not pair else abs(v4 - v3)
            same_code = codes.get("s2m") is not None and codes.get("s2m") == codes.get("s2")
            reproduces = v4 is not None and pair and abs(v4 - v1) <= TOL
            row["control_reproduces_s1"] = None if v4 is None else reproduces
            row["effect_holds"] = None if v3 is None else ("settings" if same_code or reproduces else
                                                           "settings and code")
        out[lid] = row
    return out


def check_seeds_file(tree, lines):
    """The mismatched S and V values and differences equal those of the seeds file (TOL); returns the file's d."""
    doc = read_json(os.path.join(tree, SEEDS_REL))
    if doc is None:
        raise Exit(EXIT_INPUT, f"missing: {SEEDS_REL}")
    out = {}
    for lid, (sec, key) in SEEDS_KEYS.items():
        e = ((doc.get("seed_diff") or {}).get(sec) or {}).get(key) or {}
        row = lines[lid]
        for mine, theirs in ((row["s1"], e.get("seed1")), (row["s2_mismatched"], e.get("seed2")),
                             (row["d_mismatched"], e.get("abs_diff"))):
            if fin(theirs) is None or abs(mine - float(theirs)) > TOL:
                raise Exit(EXIT_INPUT, f"{lid}: {mine} recomputed from the folders, {theirs} in {SEEDS_REL}")
        out[lid] = float(e["abs_diff"])
    return out


def margin_row(lines, doc, item, lid, frac, kind):
    row = lines[lid]
    out = {"document": doc, "item": item, "line": lid, "label": row["label"], "end": row["end"], "unit": row["unit"],
           "decimals": row["decimals"], "fraction": frac, "kind": kind, "s1": row["s1"],
           "d_mismatched": row["d_mismatched"], "d_matched": row["d_matched"], "d_same_code": row["d_same_code"],
           "matched_already": row["matched_already"]}
    if kind == "reported":
        return out
    for tag in ("mismatched",) + TAGS:
        d = row[f"d_{tag}"]
        if d is None:
            out[f"margin_{tag}"] = out[f"threshold_{tag}"] = out[f"term_{tag}"] = None
            continue
        a, b = frac * row["s1"], 2.0 * d
        m = max(a, b)
        out[f"margin_{tag}"] = m
        out[f"term_{tag}"] = "fraction" if a >= b else "2 x d"
        out[f"threshold_{tag}"] = row["s1"] - m if kind == "success" else row["s1"] + m
    if out.get("margin_matched") is not None:
        out["margin_changes"] = abs(out["margin_matched"] - out["margin_mismatched"]) > TOL
    if out.get("margin_same_code") is not None:
        out["margin_changes_same_code"] = abs(out["margin_same_code"] - out["margin_mismatched"]) > TOL
    return out


def uses_rows(lines):
    return [margin_row(lines, *u) for u in USES]


# ---------------------------------------------------------------- beside the F2 lever-2 reading
TAG_WORDS = {"matched": "the matched margin", "same_code": "the same-code margin"}


def f2l2_beside(tree, lines):
    """Each line of the F2 lever-2 reading with its recorded status and, beside it, the status under the margin built
    from the matched d and from the same-code d (labelled, never a reading). R1 and early: the point at or above the
    margin and the interval's lower bound above 0; R2: the difference at or below the margin. {'present': False}
    until the reading exists."""
    p = os.path.join(tree, F2L2_RESULTS_REL)
    doc = read_json(p)
    if doc is None:
        return {"present": False, "path": F2L2_RESULTS_REL}
    out = {"present": True, "path": F2L2_RESULTS_REL, "sha256": sha256_file(p),
           "verdict": (doc.get("verdict") or {}).get("rule"), "rows": [], "changes": [],
           "label": "beside the reading, never substituted: the rule's verdict stands"}
    for block, key, lid, frac in F2L2_BESIDE:
        src = ((doc.get("early") or {}).get("lines") or {}).get(key) if block == "early" else \
            (doc.get(block) or {}).get(key)
        src = src or {}
        line = lines[lid]
        r = {"block": block, "line": key, "line_id": lid, "margin_rule": fin(src.get("margin"))}
        if block == "R2":
            diff = fin(src.get("difference"))
            r.update(read=bool(src.get("read")), difference=diff, status_rule=src.get("holds"))
        else:
            imp = src.get("improvement") or {}
            pt, iv = fin(imp.get("point")), imp.get("interval")
            low = fin(iv[0]) if isinstance(iv, (list, tuple)) and iv else None
            status = src.get("meets") if block == "R1" else src.get("would_meet")
            r.update(read=pt is not None, improvement=pt, lower=low, status_rule=status)
        for tag in TAGS:
            d = line[f"d_{tag}"]
            mm = None if d is None else max(frac * line["s1"], 2.0 * d)
            if block == "R2":
                st = None if r["difference"] is None or mm is None else bool(r["difference"] <= mm + TOL)
            else:
                st = None if r["improvement"] is None or r["lower"] is None or mm is None else \
                    bool(r["improvement"] >= mm - TOL and r["lower"] > TOL)
            r[f"margin_{tag}"], r[f"status_{tag}"] = mm, st
            if r["status_rule"] is not None and st is not None and r["status_rule"] != st:
                out["changes"].append(f"{block} {key}: {r['status_rule']} under the hashed margin, {st} under "
                                      f"{TAG_WORDS[tag]}")
        out["rows"].append(r)
    return out


# ---------------------------------------------------------------- JP0 (JEPA addendum 3a)
def jp0_units(tree, name, sets):
    """The dev pass units of J0 (per_pass.csv of reports/bench/frozen_tokens_<name>/): keys, persons, reference
    stride (cm) and the latent probe's stride (cm)."""
    import pandas as pd
    p = os.path.join(tree, "reports", "bench", f"frozen_tokens_{name}", "per_pass.csv")
    if not os.path.exists(p):
        raise Exit(EXIT_INPUT, f"missing: {rel(p, tree)}")
    u = pd.read_csv(p, encoding="utf-8", dtype={"trial_id": str, "pass_id": str, "person_id": str})
    if not set(u["person_id"]) <= set(sets["dev14"]):
        raise Exit(EXIT_INPUT, f"{rel(p, tree)}: persons outside the dev controls")
    return {"keys": list(zip(u["trial_id"].astype(str), u["pass_id"].astype(str))),
            "persons": u["person_id"].to_numpy().astype(str),
            "ref": u["stride_m"].to_numpy(np.float64) * 100.0,
            "latent": u["pred_stride_m"].to_numpy(np.float64) * 100.0, "path": p}


def jp0_rows(tree, units, diag_path, gain, sets):
    """{latent, decoded, recalibrated: line_of on dev12} on the chain (reference, latent and decoded finite), and the
    chain count."""
    jp = JP()
    try:
        ds, _, cons, _ = jp.decoded_passes(os.path.join(diag_path, "events.csv"), os.path.join(diag_path, "diag.json"),
                                           sets, ())
    except jp.Stop as e:
        raise Exit(EXIT_INPUT, f"JP0 decoded passes of {rel(diag_path, tree)}: {e.message}")
    dec = np.array([np.nan if ds.get(k) is None else float(ds[k]) for k in units["keys"]], np.float64)
    keep = np.isfinite(units["ref"]) & np.isfinite(units["latent"]) & np.isfinite(dec)
    ins = np.isin(units["persons"], list(sets["dev12"]))
    rows = {"latent": units["latent"], "decoded": dec}
    if fin(gain) is not None and gain > 0:
        rows["recalibrated"] = dec / float(gain)
    out = {k: jp.line_of(jp.person_errors(units["persons"], units["ref"], v, keep & ins)) for k, v in rows.items()}
    return out, {"passes_dev12": int(ins.sum()), "in_chain_dev12": int((ins & keep).sum()), "consistency": cons}


def jp0_block(tree, sets):
    """Relations D and R of the JP0 reading with seed 2's decoded row from the mismatched diag (reproducing the
    reading, checked against JP0's results.json) and from the matched diag (beside, once it exists)."""
    jp = JP()
    res = read_json(os.path.join(tree, JP0_RESULTS_REL))
    if res is None:
        raise Exit(EXIT_INPUT, f"missing: {JP0_RESULTS_REL}")
    gains = (res.get("f1") or {}).get("gains") or {}
    ends = {"v1_B": S1_B, "s2_B": S2_B}
    units = {e: jp0_units(tree, n, sets) for e, n in ends.items()}
    rows = {}
    for e, n in ends.items():
        rows[e] = jp0_rows(tree, units[e], folder(tree, "diag", n), (gains.get(n) or {}).get("stride"), sets)
    m2 = folder(tree, "diag", S2_B, MATCHED)
    matched = None
    if finished(m2) and os.path.exists(os.path.join(m2, "events.csv")):
        matched = jp0_rows(tree, units["s2_B"], m2, (gains.get(S2_B) or {}).get("stride"), sets)
    c1 = folder(tree, "diag", S1_B, CONTROL)
    control = None
    # the control diag counts for the same-code relations only under the matched diag's code_sha256
    pair = matched is not None and code_of(c1) is not None and code_of(c1) == code_of(m2)
    if pair and finished(c1) and os.path.exists(os.path.join(c1, "events.csv")):
        control = jp0_rows(tree, units["v1_B"], c1, (gains.get(S1_B) or {}).get("stride"), sets)
    # the reproduction check against JP0's results.json
    for e in ends:
        lines = ((((res.get("checkpoints") or {}).get(e) or {}).get("stride") or {}).get("lines") or {}).get("dev12")
        lines = lines or {}
        for row, tol in (("latent", JP0_TOL_X), ("decoded", JP0_TOL_Y), ("recalibrated", JP0_TOL_Y)):
            if row not in rows[e][0]:
                continue
            want = fin((lines.get(row) or {}).get("point"))
            got = fin(rows[e][0][row]["point"])
            if want is None or got is None or abs(want - got) > tol:
                raise Exit(EXIT_INPUT, f"JP0 {e} {row}: {got} recomputed, {want} in {JP0_RESULTS_REL}")
    out = {"present_matched": matched is not None, "present_control": finished(c1),
           "control_same_code_as_matched": pair if matched is not None and finished(c1) else None, "relations": {},
           "chain": {"v1_B": rows["v1_B"][1], "s2_B_mismatched": rows["s2_B"][1],
                     "s2_B_matched": None if matched is None else matched[1],
                     "v1_B_now": None if control is None else control[1]},
           "f1_gains": {n: (gains.get(n) or {}).get("stride") for n in ends.values()},
           "label": ("beside the JP0 reading (hashed addendum 3a), never substituted; R stays partly matched: F1's "
                     "seed-2 stride gain comes from a --batch-windows 16 train54 decode (job f1_gain_decode_s2_B_r2); "
                     "same code: seed 1's decoded row from the control diag, seed 2's from the matched diag")}
    for rid, y in (("D", "decoded"), ("R", "recalibrated")):
        x_by = {"v1_B": rows["v1_B"][0]["latent"], "s2_B": rows["s2_B"][0]["latent"]}
        if y not in rows["v1_B"][0] or y not in rows["s2_B"][0]:
            out["relations"][rid] = {"reason": f"no {y} row (F1 gain missing)"}
            continue
        mis = jp.relation(x_by, {"v1_B": rows["v1_B"][0][y], "s2_B": rows["s2_B"][0][y]})
        want = (res.get("relations") or {}).get(rid) or {}
        for k in ("d", "m"):
            if fin(want.get(k)) is None or abs(float(want[k]) - mis[k]) > JP0_TOL_X:
                raise Exit(EXIT_INPUT, f"JP0 relation {rid}: {k} {mis[k]} recomputed, {want.get(k)} in "
                                       f"{JP0_RESULTS_REL}")
        r = {"x": "latent", "y": y, "recorded": {k: want.get(k) for k in ("relation", "d", "m")},
             "mismatched": _rel_view(mis)}
        if matched is not None and y in matched[0]:
            x2 = dict(x_by, s2_B=matched[0]["latent"])
            mat = jp.relation(x2, {"v1_B": rows["v1_B"][0][y], "s2_B": matched[0][y]})
            r["matched"] = _rel_view(mat)
            r["changes"] = mat["relation"] != mis["relation"]
            if control is not None and y in control[0]:
                x3 = {"v1_B": control[0]["latent"], "s2_B": matched[0]["latent"]}
                same = jp.relation(x3, {"v1_B": control[0][y], "s2_B": matched[0][y]})
                r["same_code"] = _rel_view(same)
                r["changes_same_code"] = same["relation"] != mis["relation"]
        out["relations"][rid] = r
    return out


def _rel_view(r):
    ends = {e: {"G": v.get("G"), "lower": (v.get("paired") or {}).get("interval", [None])[0]
                if (v.get("paired") or {}).get("interval") else None, "better": v.get("better")}
            for e, v in (r.get("ends") or {}).items()}
    return {"relation": r.get("relation"), "d": r.get("d"), "m": r.get("m"), "X1": r.get("X1"), "X2": r.get("X2"),
            "Y1": r.get("Y1"), "Y2": r.get("Y2"), "ends": ends}


# ---------------------------------------------------------------- G4 (reported)
def g4_values(doc):
    """{line: (point, verdict)} of a g4.json: the gate lines, the metric-3D MPJPE and PA-MPJPE, the GRF row without
    the low-peak series."""
    out = {}
    for r in (doc or {}).get("lines") or []:
        if isinstance(r, dict) and r.get("line") in G4_LINES:
            out[r["line"]] = (fin(r.get("point")), r.get("verdict"))
    m3 = (doc or {}).get("metric3d") or {}
    for k in ("mpjpe_mm", "pa_mpjpe_mm"):
        out[k] = (fin((m3.get(k) or {}).get("point")), None)
    nl = (((doc or {}).get("summary") or {}).get("no_low_peak") or {}).get("grf_rmse_pct_bw") or {}
    out["grf_rmse_no_low_peak"] = (fin((nl.get("model") or {}).get("point")), None)
    return out


def g4_block(tree):
    """The G4 seed difference before and after task G4H, and the harness effect per seed (reported only)."""
    paths = {"before": (os.path.join(tree, G4_REL, f"dev_{S1_B}"), os.path.join(tree, G4_REL, f"dev_{S2_B}")),
             "after": (os.path.join(tree, G4_REL, f"dev_{S1_B}{G4H}"), os.path.join(tree, G4_REL, f"dev_{S2_B}{G4H}"))}
    vals, src = {}, {}
    for h, (a, b) in paths.items():
        for who, p in (("s1", a), ("s2", b)):
            doc = read_json(os.path.join(p, "g4.json")) if (h == "before" or finished(p)) else None
            if doc is None and h == "before":
                raise Exit(EXIT_INPUT, f"missing: {rel(os.path.join(p, 'g4.json'), tree)}")
            vals[(h, who)] = g4_values(doc) if doc is not None else None
            src[f"{h} {who}"] = rel(p, tree) if doc is not None else None
    rows = {}
    for k in G4_LINES + ("mpjpe_mm", "pa_mpjpe_mm", "grf_rmse_no_low_peak"):
        r = {"unit": G4_UNITS[k]}
        for h in ("before", "after"):
            a = (vals[(h, "s1")] or {}).get(k, (None, None))
            b = (vals[(h, "s2")] or {}).get(k, (None, None))
            r[h] = {"s1": a[0], "s2": b[0], "verdict_s1": a[1], "verdict_s2": b[1],
                    "d": None if a[0] is None or b[0] is None else abs(a[0] - b[0])}
        for who in ("s1", "s2"):
            x, y = r["before"][who], r["after"][who]
            r[f"harness_effect_{who}"] = None if x is None or y is None else y - x
        rows[k] = r
    return {"label": "reported only: no document margin reads a G4 seed difference", "sources": src, "lines": rows}


# ---------------------------------------------------------------- the evaluation code then and now
CODE_FOLDERS = (f"{G3_REL}/dev_{S1_B}", f"{G3_REL}/dev_{S2_B}", f"{G3_REL}/diag_{S1_B}", f"{G3_REL}/diag_{S2_B}",
                f"{G3_REL}/curve_{S1_RUN}_{EARLY}", f"{G3_REL}/curve_{S2_RUN}_{EARLY}", f"{G4_REL}/dev_{S1_B}",
                f"{G4_REL}/dev_{S2_B}")


def code_hash(tree, files):
    """mdfm.gates.common.code_sha256 of tree-relative files: SHA-256 over the sorted '<path> TAB <file SHA-256> LF'
    lines; None when a file is missing."""
    rows = []
    for f in files:
        full = os.path.join(tree, f)
        if not os.path.exists(full):
            return None
        rows.append((str(f).replace("\\", "/"), sha256_file(full)))
    return hashlib.sha256("".join(f"{r}\t{h}\n" for r, h in sorted(rows)).encode("utf-8")).hexdigest()


def code_block(tree):
    """Per folder the documents read: the evaluation code SHA-256 its _provenance.json recorded over its code_files,
    and the same hash over those files now. A rerun runs the code now, so a difference enters its settings effect
    unless the changed code leaves the numbers alone; stated, never a stop."""
    out = {}
    for f in CODE_FOLDERS:
        prov = read_json(os.path.join(tree, f, "_provenance.json")) or {}
        files = prov.get("code_files") or []
        now = code_hash(tree, files) if files else None
        out[f] = {"code_files": files, "recorded": prov.get("code_sha256"), "now": now,
                  "same": None if now is None or not prov.get("code_sha256") else now == prov.get("code_sha256"),
                  "moveatlas_head": head_of(os.path.join(tree, f))}
    return out


# ---------------------------------------------------------------- reported only
def reported_block(tree, sets):
    """The other curve points (results_dev and the 12-subset S and V) and the raw-weight dev runs."""
    out = {"curve": {}, "raw_weights": {}}
    for step in CURVE_STEPS[1:]:
        trip = {"s1": folder(tree, "curve", S1_RUN, f"_{step}"), "s2": folder(tree, "curve", S2_RUN, f"_{step}"),
                "s2m": folder(tree, "curve", S2_RUN, f"_{step}{MATCHED}")}
        out["curve"][step] = _reported_rows(tree, trip, sets, with_sets=True)
    trip = {"s1": folder(tree, "dev", S1_B, "_model"), "s2": folder(tree, "dev", S2_B, "_model"),
            "s2m": folder(tree, "dev", S2_B, f"_model{MATCHED}")}
    out["raw_weights"] = _reported_rows(tree, trip, sets, with_sets=False)
    return out


def _reported_rows(tree, trip, sets, with_sets):
    docs = {}
    for who, p in trip.items():
        if who == "s2m" and not finished(p):
            docs[who] = None
            continue
        docs[who] = dev_doc(tree, p)
    rows = {}
    for line in REPORTED_DEV:
        rows[line] = {who: (None if d is None else dev_line(d, line)) for who, d in docs.items()}
    if with_sets:
        for who, p in trip.items():
            if docs.get(who) is None:
                continue
            s = curve_set(tree, p, sets)
            for k, key in (("S12", "S"), ("V12", "V")):
                rows.setdefault(k, {})[who] = (s["subsets"].get(SUB12) or {}).get(f"{key}_point")
    for r in rows.values():
        a, b, c = r.get("s1"), r.get("s2"), r.get("s2m")
        r["d_mismatched"] = None if a is None or b is None else abs(a - b)
        r["d_matched"] = None if a is None or c is None else abs(a - c)
    return {"sources": {who: rel(p, tree) for who, p in trip.items()},
            "settings": {who: (None if d is None else settings_of(d)) for who, d in docs.items()}, "lines": rows}


# ---------------------------------------------------------------- the run
def run(tree=None, sets=None, partial=False, log=print):
    """Every block; returns (results, exit code). Writes nothing."""
    t0 = time.time()
    tree = os.path.abspath(tree or TREE)
    sets = sets if sets is not None else load_sets(tree)
    status = rerun_status(tree)
    bad = [r for r in status if r["present"] and r["problems"]]
    if bad:
        raise Exit(EXIT_SETTINGS, "rerun folders not as their jobs asked: " + "; ".join(
            f"{r['folder']}: {', '.join(r['problems'])}" for r in bad))
    missing = [r for r in status if not r["present"]]
    if missing and not partial:
        raise Exit(EXIT_WAITING, "waiting for " + str(len(missing)) + " rerun and control folders: " + "; ".join(
            f"{r['folder']} (job {r['job']})" for r in missing))
    lines = g3_lines(tree, sets)
    res = {"what": LABEL, "task": TASK, "complete": not missing, "waiting_for": [r["folder"] for r in missing],
           "statement": ("The hashed F2 lever-2 rule keeps its margins; the matched differences are reported beside "
                         "its reading, never substituted. The JP0 reading (hashed addendum 3a of the JEPA proposal) "
                         "and the margins that section 3 of the JEPA proposal prints stand in the same way."),
           "documents": {"F2L2": F2L2_DOC, "JEPA": JEPA_DOC},
           "reruns": status, "lines": lines, "seeds_file_check": check_seeds_file(tree, lines),
           "uses": uses_rows(lines), "f2l2_beside": f2l2_beside(tree, lines), "jp0": jp0_block(tree, sets),
           "g4": g4_block(tree), "reported": reported_block(tree, sets), "code": code_block(tree),
           "sets": {"dev14_sha256": person_hash(sets["dev14"]), "dev12_sha256": person_hash(sets["dev12"])}}
    res["timing_s"] = round(time.time() - t0, 1)
    log(f"lines read: {len(lines)}; rerun and control folders present: {len(status) - len(missing)} of "
        f"{len(status)}")
    return res, (EXIT_WAITING if missing else EXIT_OK)


REPORT_FILES = ("diag.json", "events.csv", "g3_dev.json", "g3_sim.json", "per_person.csv", "g4.json")


def input_files(tree, res):
    """{tree-relative path: SHA-256} of every file read (report files of every folder named, G1 jsons, the seeds
    file, F1's and JP0's results, J0's per_pass.csv, the F2 lever-2 results when present)."""
    paths = [SEEDS_REL, F1_RESULTS_REL, JP0_RESULTS_REL]
    folders = []
    for row in res["lines"].values():
        for p in row["sources"].values():
            if p and p.endswith(".json"):
                paths.append(p)
            elif p:
                folders.append(p)
    for name in (S1_B, S2_B):
        paths.append(f"reports/bench/frozen_tokens_{name}/per_pass.csv")
        folders.append(f"{G3_REL}/diag_{name}")
    folders += [f"{G3_REL}/diag_{S2_B}{MATCHED}", f"{G3_REL}/diag_{S1_B}{CONTROL}"]
    for x in res["reruns"]:
        for d in (x.get("records_drift") or {}).values():
            if (d or {}).get("log"):
                paths.append(d["log"])
    folders += [p for p in res["g4"]["sources"].values() if p]
    for b in list(res["reported"]["curve"].values()) + [res["reported"]["raw_weights"]]:
        folders += [p for p in b["sources"].values() if p]
    if res["f2l2_beside"].get("present"):
        paths.append(F2L2_RESULTS_REL)
    for f in folders:
        paths += [f"{f}/{n}" for n in REPORT_FILES]
    files = {}
    for p in paths:
        full = os.path.join(tree, p)
        if p not in files and os.path.exists(full):
            files[p.replace("\\", "/")] = sha256_file(full)
    return dict(sorted(files.items()))


# ---------------------------------------------------------------- summary
def _drift_text(x):
    """The records drift of a rerun's load as text: 'none', the changed sources with their columns, or the reason
    no evidence was found; '' for a folder not present."""
    if not x.get("present"):
        return ""
    d = ((x.get("records_drift") or {}).get("rerun"))
    if d is None:
        return "none"
    if d.get("reason"):
        return d["reason"]
    parts = [f"{c.get('source')} ({', '.join(c.get('columns') or []) or c.get('kind')})"
             for c in d.get("changes") or []]
    return "records " + str((d.get("records_sha256") or ["-"])[0])[:8] + " to " \
        + str((d.get("records_sha256") or ["-", "-"])[-1])[:8] + ": " + ("; ".join(parts) or "no row changed")


def _gate_changed(x):
    """'no', the changed sources the gate reads, or 'n/a' (not present, or no evidence found)."""
    if not x.get("present"):
        return "n/a"
    d = ((x.get("records_drift") or {}).get("rerun"))
    if d is None:
        return "no"
    if d.get("reason"):
        return "n/a"
    return ", ".join(d.get("gate_sources_changed") or []) or "no"


def summary_md(r):
    def nd(row):
        return row.get("decimals", 2)

    L = ["# Seed differences at matched evaluation settings (task SEED)", "",
         f"Label: {r['what']}.", "",
         r["statement"], "",
         "Written by scripts/seed_difference.py. Seed 1: cache/runs/phaseA_root_full; seed 2: cache/runs/pilot_v1_s2. "
         "Mismatched: seed 2's folders as the documents read them (G3 at batch_windows 16 under a 2.4 GB cap). "
         "Matched: seed 2 at seed 1's settings (batch_windows 32 under 12 GB), the folders of queue group SEED "
         "(suffix _bw32). Seed 1 now: the code controls of the same group, seed 1's own evaluations at its own "
         "settings under the code the reruns run (suffix _bw32_now). d = |seed 1 minus seed 2|; d same code = "
         "|seed 1 now minus seed 2 matched|, both seeds at seed 1's settings under one code; effect of the rerun = "
         "seed 2 matched minus seed 2 mismatched; code effect = seed 1 now minus seed 1.", ""]
    if not r.get("complete"):
        L += [f"PARTIAL: {len(r['waiting_for'])} rerun and control folders do not exist yet: "
              + ", ".join(r["waiting_for"]) + ".", ""]
    if r.get("moveatlas_head"):
        L += [f"MoveAtlas HEAD at the run: {r['moveatlas_head']}.", ""]
    holds = [row.get("effect_holds") for row in r["lines"].values() if row.get("effect_holds") is not None]
    effect = "settings effect" if holds and all(h == "settings" for h in holds) else "settings and code effect"
    L += ["## Lines", "", "| line | end | unit | seed 1 | seed 2 mismatched | seed 2 matched | seed 1 now | "
          f"d mismatched | d matched | d same code | {effect} | code effect seed 1 | holds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lid, row in r["lines"].items():
        k = nd(row)
        L.append(f"| {row['label']} | {row['end']} | {row['unit']} | {fmt(row['s1'], k)} | "
                 f"{fmt(row['s2_mismatched'], k)} | {fmt(row['s2_matched'], k)} | {fmt(row.get('s1_now'), k)} | "
                 f"{fmt(row['d_mismatched'], k + 1)} | {fmt(row['d_matched'], k + 1)} | "
                 f"{fmt(row.get('d_same_code'), k + 1)} | {fmt(row['settings_effect_s2'], k + 1)} | "
                 f"{fmt(row.get('code_effect_s1'), k + 1)} | {row.get('effect_holds') or 'n/a'} |")
    pairs = [lid for lid, row in r["lines"].items() if row.get("control_same_code_as_rerun") is False]
    L += ["", f"The {effect} column: seed 2 matched minus seed 2 mismatched. It holds the settings alone ('settings' "
          "under holds) where the rerun recorded the code_sha256 of the folder it repeats, or where the control, run "
          f"under the rerun's code, reproduces seed 1 (code effect within {TOL:g}). Otherwise the code change enters "
          "it ('settings and code'): read it against the code effect on seed 1, which also holds any run-to-run "
          "variation, and prefer d same code to d matched. Lines whose control and rerun recorded different "
          "code_sha256 (no d same code): " + (", ".join(pairs) or "none") + ".", "",
          "G1a rows: " + "; ".join(sorted({row["settings_note"] for row in r["lines"].values()
                                         if row.get("matched_already") or "G1a" in row["label"]}))
          + "; " + "; ".join(sorted({row["code_note"] for row in r["lines"].values() if row.get("code_note")})) + ".",
          ""]
    L += ["## Margins that read a seed difference", "",
          "Margin = max(fraction x seed 1's line, 2 x d), with its deciding term, then the threshold: seed 1's line "
          "minus the margin (success) or plus it (not worse). The mismatched columns are the documents' values; the "
          "matched and same-code columns are beside them.", "",
          "| document | item | fraction | d mismatched | margin, threshold mismatched | d matched | "
          "margin, threshold matched | d same code | margin, threshold same code | margin changes (matched, same "
          "code) |", "|---|---|---|---|---|---|---|---|---|---|"]

    def mt(u, tag, k):
        m = u.get(f"margin_{tag}")
        return "n/a" if m is None else (f"{fmt(m, k + 1)} ({u.get(f'term_{tag}')}), "
                                        f"{fmt(u.get(f'threshold_{tag}'), k)}")

    def yn(x):
        return "n/a" if x is None else ("yes" if x else "no")

    for u in r["uses"]:
        k = u["decimals"]
        if u["kind"] == "reported":
            L.append(f"| {u['document']} | {u['item']} | reported | {fmt(u['d_mismatched'], k + 1)} | | "
                     f"{fmt(u['d_matched'], k + 1)} | | {fmt(u.get('d_same_code'), k + 1)} | | |")
            continue
        L.append(f"| {u['document']} | {u['item']} | {u['fraction']:.2f} | {fmt(u['d_mismatched'], k + 1)} | "
                 f"{mt(u, 'mismatched', k)} | {fmt(u['d_matched'], k + 1)} | {mt(u, 'matched', k)} | "
                 f"{fmt(u.get('d_same_code'), k + 1)} | {mt(u, 'same_code', k)} | "
                 f"{yn(u.get('margin_changes'))}, {yn(u.get('margin_changes_same_code'))} |")
    L += ["", f"- {r['documents']['F2L2']}.", f"- {r['documents']['JEPA']}.", ""]
    fb = r["f2l2_beside"]
    L += ["## Beside the F2 lever-2 reading", ""]
    if not fb.get("present"):
        L += [f"{fb['path']} does not exist yet: nothing to set beside.", ""]
    else:
        L += [f"Verdict of the rule: {fb.get('verdict')} ({fb['label']}).", "",
              "| block | line | point | lower bound | margin (hashed) | status (rule) | margin (matched d) | "
              "status (matched d) | margin (same-code d) | status (same-code d) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for x in fb["rows"]:
            pt = x.get("difference") if x["block"] == "R2" else x.get("improvement")
            L.append(f"| {x['block']} | {x['line']} | {fmt(pt, 4)} | {fmt(x.get('lower'), 4)} | "
                     f"{fmt(x['margin_rule'], 4)} | {x['status_rule']} | {fmt(x.get('margin_matched'), 4)} | "
                     f"{x.get('status_matched')} | {fmt(x.get('margin_same_code'), 4)} | {x.get('status_same_code')} |")
        L += ["", "Statuses the matched or same-code margins would change: " + ("; ".join(fb["changes"]) or "none")
              + ".", ""]
    jp = r["jp0"]
    L += ["## JP0 relations (JEPA addendum 3a)", "", jp["label"] + ".", "",
          "| relation | reading | d | m | relation | d (matched) | m (matched) | relation (matched) | d (same code) | "
          "m (same code) | relation (same code) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rid, x in jp["relations"].items():
        if "mismatched" not in x:
            L.append(f"| {rid} | {x.get('reason')} | | | | | | | | | |")
            continue
        mi, ma, sc = x["mismatched"], x.get("matched") or {}, x.get("same_code") or {}
        L.append(f"| {rid} | latent against {x['y']} | {fmt(mi['d'], 3)} | {fmt(mi['m'], 3)} | {mi['relation']} | "
                 f"{fmt(ma.get('d'), 3)} | {fmt(ma.get('m'), 3)} | {ma.get('relation', 'n/a')} | "
                 f"{fmt(sc.get('d'), 3)} | {fmt(sc.get('m'), 3)} | {sc.get('relation', 'n/a')} |")
    ch = jp["chain"]

    def chain(x):
        return "n/a" if x is None else str(x["in_chain_dev12"])

    L += ["", "Passes of the 12 in the chain: v1 " + chain(ch["v1_B"]) + ", seed 2 mismatched "
          + chain(ch["s2_B_mismatched"]) + ", seed 2 matched " + chain(ch["s2_B_matched"]) + ", v1 now "
          + chain(ch.get("v1_B_now")) + ".", ""]
    if jp.get("control_same_code_as_matched") is False:
        L += ["The control diag recorded another code_sha256 than the matched diag: no same-code relation.", ""]
    g4 = r["g4"]
    L += ["## G4 (reported only)", "", g4["label"] + ".", "",
          "| line | unit | before G4H: seed 1 | seed 2 | d | after G4H: seed 1 | seed 2 | d | harness effect seed 1 "
          "| harness effect seed 2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in g4["lines"].items():
        b, a = x["before"], x["after"]
        L.append(f"| {k} | {x['unit']} | {fmt(b['s1'], 4)} ({b['verdict_s1'] or '-'}) | {fmt(b['s2'], 4)} "
                 f"({b['verdict_s2'] or '-'}) | {fmt(b['d'], 4)} | {fmt(a['s1'], 4)} ({a['verdict_s1'] or '-'}) | "
                 f"{fmt(a['s2'], 4)} ({a['verdict_s2'] or '-'}) | {fmt(a['d'], 4)} | {fmt(x['harness_effect_s1'], 4)} "
                 f"| {fmt(x['harness_effect_s2'], 4)} |")
    L += ["", "## Reported only: the other curve points and the raw weights", "",
          "| folder | line | seed 1 | seed 2 mismatched | seed 2 matched | d mismatched | d matched |",
          "|---|---|---|---|---|---|---|"]
    blocks = [(f"curve {s}", b) for s, b in r["reported"]["curve"].items()] + [("dev, raw weights",
                                                                                r["reported"]["raw_weights"])]
    for what, b in blocks:
        for line, x in b["lines"].items():
            L.append(f"| {what} | {line} | {fmt(x.get('s1'), 4)} | {fmt(x.get('s2'), 4)} | {fmt(x.get('s2m'), 4)} | "
                     f"{fmt(x.get('d_mismatched'), 4)} | {fmt(x.get('d_matched'), 4)} |")
    L += ["", "## Rerun and control folders", "",
          "MoveAtlas HEAD of each folder (the rerun, the folder it repeats, seed 1's folder), and the records drift "
          "that the checkpoint load of the rerun logged (drift_log.jsonl beside the checkpoint): the changed sources "
          "with their columns, and whether one of them is a source the gate reads (G3: "
          + ", ".join(GATE_SOURCES["g3"]) + "; G4: " + ", ".join(GATE_SOURCES["g4"]) + "). A rerun measured against "
          "another MoveAtlas state, or run under another evaluation code, than the folders it pairs with carries any "
          "record or code change into its effect.", "",
          "| job | role | folder | present | job_log state | MoveAtlas HEAD rerun / repeats / seed 1 | records drift "
          "of the rerun's load | a source the gate reads changed |",
          "|---|---|---|---|---|---|---|---|"]
    for x in r["reruns"]:
        st = (x.get("job_log") or {}).get("state") or "no row"
        hd = x.get("moveatlas_head") or {}
        heads = " / ".join(str(hd.get(k) or "-")[:12] for k in ("rerun", "repeats", "seed1")) if hd else ""
        L.append(f"| {x['job']} | {x.get('role', 'rerun')} | {x['folder']} | {'yes' if x['present'] else 'no'} | "
                 f"{st} | {heads} | {_drift_text(x)} | {_gate_changed(x)} |")
    L += ["", "## Evaluation code then and now", "",
          "code_sha256 that each folder's _provenance.json recorded over its code_files, and the same hash over those "
          "files now (the code a rerun runs), and the MoveAtlas HEAD the folder was measured against.", "",
          "| folder | recorded | now | same | MoveAtlas HEAD |", "|---|---|---|---|---|"]
    for f, x in r.get("code", {}).items():
        L.append(f"| {f} | {str(x.get('recorded') or '-')[:16]} | {str(x.get('now') or '-')[:16]} | "
                 f"{'n/a' if x.get('same') is None else ('yes' if x['same'] else 'no')} | "
                 f"{str(x.get('moveatlas_head') or '-')[:12]} |")
    diff_heads = [x["job"] for x in r["reruns"] if x.get("present") and not x.get("same_moveatlas_head", True)]
    diff_code = [x["job"] for x in r["reruns"] if x.get("present") and not x.get("same_code", True)]
    gate = [x["job"] for x in r["reruns"] if x.get("present") and _gate_changed(x) not in ("no", "n/a")]
    unknown = [x["job"] for x in r["reruns"] if x.get("present") and _gate_changed(x) == "n/a"
               and ((x.get("records_drift") or {}).get("rerun") or {}).get("reason")]
    L += ["", "Reruns measured against another MoveAtlas HEAD than the folders they pair with: "
          + (", ".join(diff_heads) or "none") + ".",
          "Of the reruns whose load logged a records drift, those where it touches a source the gate reads: "
          + (", ".join(gate) or "none") + "; those whose drift entry was not found: " + (", ".join(unknown) or "none")
          + ".",
          "Reruns run under another evaluation code SHA-256 (code_sha256 of _provenance.json) than the folders they "
          "pair with: " + (", ".join(diff_code) or "none") + ".", ""]
    text = "\n".join(L) + "\n"
    for bad in (chr(0x2014), chr(0x2013)):
        if bad in text:
            raise Exit(EXIT_INPUT, "summary.md would hold a long dash")
    return text


# ---------------------------------------------------------------- outputs
def write_outputs(tree, out, res, stamp, started):
    moveatlas = os.environ.get("MOVEATLAS_ROOT", MOVEATLAS_DEFAULT).replace("\\", "/").rstrip("/").lower()
    o = os.path.abspath(out).replace("\\", "/").rstrip("/").lower()
    if o == moveatlas or o.startswith(moveatlas + "/"):
        raise Exit(EXIT_INPUT, f"--out {out} lies inside the MoveAtlas tree")
    res = dict(res, run_started=started, moveatlas_head=(stamp or {}).get("head"))
    files = input_files(tree, res)
    res["inputs"] = files
    res = clean(res)
    text = summary_md(res)
    body = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if chr(0x2014) in body or chr(0x2013) in body:
        raise Exit(EXIT_INPUT, "results.json would hold a long dash")
    code = [os.path.abspath(__file__), os.path.join(HERE, "weekend_branch.py"), os.path.join(HERE, "jp0_read.py")]
    prov = {"script": "scripts/seed_difference.py", "task": TASK, "label": LABEL, "run_at": started,
            "moveatlas_head": (stamp or {}).get("head"), "moveatlas_stamp": stamp, "python": sys.version.split()[0],
            "numpy": np.__version__, "code_sha256": {rel(p, tree): sha256_file(p) for p in code},
            "inputs": files, "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}}
    os.makedirs(out, exist_ok=True)
    for name, data in (("results.json", body), ("summary.md", text),
                       ("_provenance.json", json.dumps(clean(prov), indent=1, ensure_ascii=False) + "\n")):
        tmp = os.path.join(out, f".{name}.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
        os.replace(tmp, os.path.join(out, name))
    return text


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tree", default=None, help="root holding the inputs (default this tree)")
    p.add_argument("--out", default=None, help=f"default <tree>/{OUT_REL}")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true", help="the rerun folders only; writes nothing")
    g.add_argument("--partial", action="store_true", help="read what exists, print, write nothing")
    g.add_argument("--dry-run", action="store_true", help="every folder needed; print, write nothing")
    return p.parse_args(argv)


def status_text(rows):
    L = []
    for r in rows:
        what = "present" if r["present"] else "missing " + ", ".join(r.get("missing") or [])
        if r["problems"]:
            what += "; PROBLEMS: " + "; ".join(r["problems"])
        L.append(f"{r['job']}: {r['folder']}: {what}")
    return "\n".join(L)


def main(argv=None, sets=None, stamp=None, log=print):
    a = parse(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    tree = os.path.abspath(a.tree or TREE)
    out = os.path.abspath(a.out or os.path.join(tree, OUT_REL))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        if a.status:
            rows = rerun_status(tree)
            log(status_text(rows))
            if any(r["present"] and r["problems"] for r in rows):
                return EXIT_SETTINGS
            return EXIT_WAITING if any(not r["present"] for r in rows) else EXIT_OK
        writing = not (a.partial or a.dry_run)
        if writing and os.path.exists(os.path.join(out, "results.json")):
            raise Exit(EXIT_EXISTS, f"{rel(os.path.join(out, 'results.json'), tree)} exists; pass --out for a new "
                                    "folder")
        res, code = run(tree, sets=sets, partial=a.partial, log=log)
        if not writing:
            log(summary_md(clean(dict(res, run_started=started))))
            return code
        if stamp is None:
            stamp = _sibling("moveatlas_stamp").stamp()
        text = write_outputs(tree, out, res, stamp, started)
        log(text)
        log(f"written {rel(out, tree)}")
        return EXIT_OK
    except Exit as e:
        log(e.message)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
