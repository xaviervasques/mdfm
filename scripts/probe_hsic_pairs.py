"""S2.8 review: where the stratified HSIC term can act in S2a (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7 and
section 9 item 10).

losses.hsic is active in a (task, HSIC stratum) pair of a micro-step only when the pair holds windows of at least two
source codes (synthetic windows carry the code 'synthetic'; the replay's sources keep their names, so the replay's
weargait_pd controls and the weargait_pd unit share one code). This probe lists, over the windows phase C can draw,
the source codes each pair can ever hold, and the stage-2 training persons none of whose recordings sits in a pair of
two codes or more: the persons on whose windows the term never acts, whatever the draw. It does so under the strata
table (task, stratum) and under the stage-1 stratification (task only), so that the cost of the strata reads apart
from the cost of the tasks. An upper bound of where the term acts: a micro-step holds a draw of these windows.

Inputs: the strata table (cache/hsic_strata/strata.csv under the SHA-256 of reports/hsic_strata/strata.sha256); the
records.csv tables of the stage-2 store (train fold, the units) and of the stage-1 store (train fold, the replay,
stratum 0); the takes.csv tables of the AMASS body store's train split (the synthetic windows, stratum 0, their task
from windows.task_id('amass', take_id): the tasks a synthetic window can carry). No record is opened; no lockbox,
CODY or care_pd row is read (the stores and the body store hold none).
Writes reports/hsic_strata/pairs.md and pairs.json.

Usage (mdfm environment, CPU):
  <conda>/envs/mdfm/python.exe scripts/probe_hsic_pairs.py
"""
import collections
import csv
import datetime
import hashlib
import json
import os
import sys

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import moveatlas_head  # noqa: E402

SYNTHETIC = "synthetic"         # losses.SYNTHETIC, the source code of the synthetic windows
OUT = os.path.join(TREE, "reports", "hsic_strata")


def pair_sources(records, strata):
    """{(task, stratum): set of source codes} of records [(source, person_id, task)]; a person absent from strata
    (the replay, the synthetic windows) is in stratum 0."""
    out = collections.defaultdict(set)
    for src, pid, task in records:
        out[(int(task), int(strata.get(pid, 0)))].add(src)
    return dict(out)


def never_active(records, strata, persons):
    """Sorted persons of `persons` none of whose records sits in a (task, stratum) pair of two source codes or more
    (pair_sources over every record)."""
    pairs = pair_sources(records, strata)
    act = {p: False for p in persons}
    for src, pid, task in records:
        if pid in act and len(pairs[(int(task), int(strata.get(pid, 0)))]) >= 2:
            act[pid] = True
    return sorted(p for p, a in act.items() if not a)


def train_records(root):
    """[(source, person_id, task)] of the train fold of a window store's records.csv tables, and their SHA-256."""
    out, shas = [], {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "records.csv")
        if name.startswith("._") or not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            shas[name] = hashlib.sha256(f.read()).hexdigest()
        with open(path, encoding="utf-8", newline="") as f:
            out += [(r["source"], r["person_id"], int(r["task"])) for r in csv.DictReader(f) if r["fold"] == "train"]
    return out, shas


def synthetic_tasks(root):
    """{task id: takes} of the body store's train takes (takes.csv of each archive), and the SHA-256 of the sorted
    take ids."""
    from mdfm.data import windows as W
    tasks, ids = collections.Counter(), []
    split = os.path.join(root, "train")
    for name in sorted(os.listdir(split)):
        path = os.path.join(split, name, "takes.csv")
        if name.startswith("._") or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ids.append(r["take_id"])
                tasks[W.task_id("amass", r["take_id"])] += 1
    return dict(tasks), hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from mdfm import paths
    from mdfm.data import body_store as BSt
    from mdfm.data import strata as ST
    from mdfm.data import window_store as WS
    from mdfm.data import windows as W
    t0 = datetime.datetime.now()
    with open(os.path.join(OUT, "strata.sha256"), encoding="utf-8") as f:
        sha = f.read().split()[0]
    tab = ST.load(ST.TABLE, sha)
    s2_root, s1_root, body_root = WS.store_root(kind="s2"), WS.store_root(), BSt.store_root()
    units, s2_shas = train_records(s2_root)
    replay, s1_shas = train_records(s1_root)
    syn, takes_sha = synthetic_tasks(body_root)
    for src, pid, _ in replay:
        if pid in tab:
            raise ValueError(f"replay person {pid} in the strata table")
    records = units + replay + [(SYNTHETIC, None, t) for t in syn]
    persons = sorted({pid for _, pid, _ in units})
    src_of = {pid: src for src, pid, _ in units}
    names = W.task_vocabulary()["tasks"]
    res = {}
    for key, strata in (("task_stratum", tab.strata), ("task_only", {})):
        never = never_active(records, strata, persons)
        res[key] = {"persons": len(persons), "never_active": len(never),
                    "never_active_by_source": dict(sorted(collections.Counter(src_of[p] for p in never).items())),
                    "never_active_by_stratum": {str(k): v for k, v in sorted(collections.Counter(
                        tab.strata[p] for p in never).items())},
                    "never_active_ids": never}
    lost = sorted(set(res["task_stratum"]["never_active_ids"]) - set(res["task_only"]["never_active_ids"]))
    pairs = pair_sources(records, tab.strata)
    unit_pairs = sorted({(t, tab.strata[p]) for _, p, t in units}, key=lambda k: (k[1], k[0]))
    head = moveatlas_head(paths.MOVEATLAS)
    doc = {"script": "scripts/probe_hsic_pairs.py", "run_at": t0.isoformat(timespec="seconds"),
           "moveatlas_head": head, "strata_sha256": tab.sha256, "stage2_records_sha256": s2_shas,
           "stage1_records_sha256": s1_shas, "body_store_train_take_ids_sha256": takes_sha,
           "synthetic_takes_per_task": {names[k]: v for k, v in sorted(syn.items())},
           "unit_pairs": [{"task": names[t], "stratum": s, "sources": sorted(pairs[(t, s)])} for t, s in unit_pairs],
           "task_stratum": {k: v for k, v in res["task_stratum"].items() if k != "never_active_ids"},
           "task_only": {k: v for k, v in res["task_only"].items() if k != "never_active_ids"},
           "taken_out_by_the_strata": {"persons": len(lost),
                                       "by_source": dict(sorted(collections.Counter(src_of[p] for p in lost).items())),
                                       "by_stratum": {str(k): v for k, v in sorted(collections.Counter(
                                           tab.strata[p] for p in lost).items())}}}
    with open(os.path.join(OUT, "pairs.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
    ts, to = doc["task_stratum"], doc["task_only"]
    L = ["# Where the stratified HSIC term can act in S2a (S2.8 review)", "",
         f"Generated by scripts/probe_hsic_pairs.py at {doc['run_at']}; MoveAtlas commit {head}. Strata table SHA-256 "
         f"{tab.sha256}. Windows phase C can draw: the stage-2 store's train fold ({len(units)} recordings), the "
         f"stage-1 store's train fold (the replay, {len(replay)} recordings, stratum 0) and the synthetic windows "
         f"(stratum 0, source code '{SYNTHETIC}', the tasks of {sum(syn.values())} body-store train takes). The term "
         "is active in a (task, stratum) pair holding two source codes or more; the counts below are an upper bound "
         "of where it acts, since a micro-step holds a draw of these windows.", "",
         "## Stage-2 training persons on whose windows the term never acts", "",
         "| stratification | persons | never in an active pair | by source | by stratum |", "|---|---|---|---|---|",
         f"| (task, stratum), S2a | {ts['persons']} | {ts['never_active']} | {ts['never_active_by_source']} | "
         f"{ts['never_active_by_stratum']} |",
         f"| task only, stage 1 | {to['persons']} | {to['never_active']} | {to['never_active_by_source']} | "
         f"{to['never_active_by_stratum']} |", "",
         f"Taken out of every active pair by the strata: {len(lost)} persons, by source "
         f"{doc['taken_out_by_the_strata']['by_source']}, by stratum {doc['taken_out_by_the_strata']['by_stratum']} "
         "(the integers of strata.csv).", "",
         "## The (task, stratum) pairs of the stage-2 training recordings", "",
         "| task | stratum | source codes | active |", "|---|---|---|---|"]
    for e in doc["unit_pairs"]:
        L.append(f"| {e['task']} | {e['stratum']} | {', '.join(e['sources'])} | "
                 f"{'yes' if len(e['sources']) >= 2 else 'no'} |")
    L += ["", "Synthetic takes per task (body-store train split): " + ", ".join(
        f"{k} {v}" for k, v in doc["synthetic_takes_per_task"].items()) + ".", ""]
    with open(os.path.join(OUT, "pairs.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print(f"(task, stratum): {ts['never_active']} of {ts['persons']} never active {ts['never_active_by_source']}; "
          f"task only: {to['never_active']} {to['never_active_by_source']}; taken out by the strata {len(lost)}")


if __name__ == "__main__":
    main()
