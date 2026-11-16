"""Structure census for the three G8X rule drafts (task G8X, wave 18, 2026-10-16): counts only.

Drafts served: docs/rules/G8_MODYS_DIS_RULE_DRAFT.md, docs/rules/G8_HUBU_FINGERTAP_RULE_DRAFT.md and
docs/rules/CP_ENDPOINT_PREMIUM_RULE_DRAFT.md. Written before any dev number of those lines exists: this script
converts a DIS item cell to a number only to class it and keeps the class, never the value; it runs no model,
descriptor or probe, and computes no statistic that involves a DIS item value or a rating.

What it reads, and how (guards asserted in the code):
- cache/manifest/manifest_records.csv.gz (gzip): the modys_video records with their person and lockbox flag, and
  the rows of hubu_fis_fingertap (none expected). reports/lockbox/lockbox_persons.csv: the 7 lockbox children,
  compared with the manifest column. reports/manifest_tiers/tiers_by_source.csv: the tier rows of both sources.
- MoveAtlas datasets/pose/modys_video/raw/data_clinical_scoring.xlsx, with openpyxl in read-only mode: the header
  row must equal HEADER (64 columns) before any data row is read. In every data row the ID cell is read before
  any other cell (scan_rows); a row of a lockbox child is counted and skipped, and no other cell of it is indexed,
  iterated or tested by this script (the xlsx reader parses every cell of a row as it yields it); a row with no ID
  counts as empty only when every cell is empty, and is refused otherwise. In a training child's row the video,
  time and scorer cells are read, and each of the 20 DIS item cells is classed as
  'scored' (an integer from 0 to 4), 'not_scored' (999, the sheet's code), 'blank' or 'other': the class is kept,
  never the value. The _max and _pscore columns are not read. No coordinate zip is opened.
- reports/modys_structure/scorer_by_video.csv: the scorer structure of task T10, checked against the sheet.
- MoveAtlas datasets/pose/hubu_fis_fingertap/datacard.md (documentation): its status line, verbatim. No file of the
  hubu_fis_fingertap raw folder is opened or listed: its ratings are EVAL (design section 5).
- scripts/build_tiers_and_lockbox.py and mdfm/model/tokenize.py, parsed with ast (no import of torch): the EVAL
  set and the tokenizer's input stems.
- mdfm.gates.common.guard_eval_read for both sources with no rule: what the EVAL read guard says today.

Computes: the 5 child folds of the 27 training children (mdfm.eval.probes.person_folds, k 5, seed 20261004,
the outer folds mdfm.eval.probes.cross_fit deals with that seed), written with their SHA-256 before any dev read;
the 95 percent interval of a Spearman correlation from counts only, at planted values (Fisher z with the
variance 1.06 / (n - 3) of Fieller, Hartley and Pearson 1957, Biometrika 44, 470-481; cited from the S1 draft,
checked against the source before any hash).

Writes: reports/census_g8x_lines/ (summary.md, facts.json, modys_child_folds.csv, modys_child_folds.sha256,
run.log, _provenance.json). Nothing in MoveAtlas.
Run: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python scripts/census_g8x_lines.py
     --moveatlas <moveatlas>
"""
import ast
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

REGIONS = ("TRUNK", "RAP", "LAP", "RLP", "LLP")      # trunk, right and left proximal arm, right and left proximal leg
COMPONENTS = ("D", "CA")                             # dystonia, choreoathetosis
FACETS = ("tD", "tA")                                # duration, amplitude
ITEMS = tuple(f"{c}_{r}_R_{f}" for r in REGIONS for c in COMPONENTS for f in FACETS)
HEADER = ("ID", "video", "time", "scorer") + tuple(x for i in ITEMS for x in (i, i + "_max", i + "_pscore"))
TIMES = ("T0", "T3", "T12")
CLASSES = ("scored", "not_scored", "blank", "other")
NOT_SCORED = 999
FOLDS = 5
FOLD_SEED = 20261004                                 # mdfm.eval.probes.SEED
SPEARMAN_VAR = 1.06                                  # var(atanh(r_s)) ~ 1.06 / (n - 3), Fieller et al. 1957
Z95 = 1.959963984540054
PLANTED = (0.35, 0.50)                               # the design's kill and pass lines of the modys_video line
CHILD_COUNTS = {"lockbox": 7, "training": 27, "all": 34}
HUBU = "hubu_fis_fingertap"
MODYS = "modys_video"

assert len(ITEMS) == 20 and len(HEADER) == 64


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)
        print(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def child_of_id(v):
    """The person id of a sheet ID cell: 'modys:child' + the ID on two digits (reports/modys_subjects)."""
    return "modys:child" + str(int(v)).zfill(2)


def classify(v):
    """Class of one DIS item cell: 'scored' (an integer 0 to 4), 'not_scored' (999), 'blank' or 'other'."""
    if v is None or isinstance(v, bool):
        return "blank" if v is None else "other"
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return "blank"
        try:
            v = float(s)
        except ValueError:
            return "other"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "other"
    if not math.isfinite(x) or x != int(x):
        return "other"
    if int(x) == NOT_SCORED:
        return "not_scored"
    return "scored" if 0 <= int(x) <= 4 else "other"


def read_sheet(path, lockbox, training, log=print):
    """Census of the scoring sheet (module docstring): (counts, per_video, n_lockbox_rows). counts[item][time][class]
    and counts[item]['all'][class]; per_video {video: {'person_ns', 'time', 'rows', 'scorers', 'scored_rows'}}, where
    scored_rows counts the rows holding at least one scored item. Raises on a header other than HEADER, on a child that
    is neither lockbox nor training, and on a training video seen with two children or two times."""
    import openpyxl
    lockbox, training = set(lockbox), set(training)
    assert not lockbox & training, "a child is both lockbox and training"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        assert len(wb.worksheets) == 1, f"{len(wb.worksheets)} sheets, expected 1"
        rows = wb.worksheets[0].iter_rows(values_only=True)
        header = tuple(next(rows))
        head, tail = header[:len(HEADER)], header[len(HEADER):]
        if head != HEADER or any(c is not None for c in tail):
            raise ValueError(f"header differs from the expected 64 columns: {header[:8]}...")
        log("header: the 64 expected columns (ID, video, time, scorer, 20 DIS items with _max and _pscore)")
        counts, per_video, n_lock, n_empty = scan_rows(rows, lockbox, training)
        log(f"sheet rows: {sum(v['rows'] for v in per_video.values())} of training children read, {n_lock} of "
            f"lockbox children skipped on the ID cell, {n_empty} empty")
        return counts, per_video, n_lock
    finally:
        wb.close()


def scan_rows(rows, lockbox, training):
    """The data rows of the sheet, after the header check: (counts, per_video, n_lockbox_rows, n_empty_rows). The ID
    cell (column 0) of each row is read before any other cell. A lockbox child's row is counted and skipped: no other
    cell of it is indexed, iterated or tested. A row whose ID cell is empty is counted as empty when every cell is
    empty and refused otherwise (a row with data and no child cannot be told from a lockbox row)."""
    counts = {i: {t: collections.Counter() for t in TIMES + ("all",)} for i in ITEMS}
    per_video = {}
    n_lock = n_empty = 0
    col = {name: k for k, name in enumerate(HEADER)}
    assert col["ID"] == 0
    for row in rows:
        idv = row[0] if len(row) else None               # the ID cell first, before any other cell
        if idv is not None:
            child = child_of_id(idv)
            if child in lockbox:
                n_lock += 1                                # no other cell of a lockbox row is touched
                continue
        elif any(c is not None for c in row):
            raise ValueError("a sheet row holds data and no ID")
        else:
            n_empty += 1
            continue
        if child not in training:
            raise ValueError(f"sheet ID {child} is neither a training nor a lockbox child")
        row = tuple(row) + (None,) * max(0, len(HEADER) - len(row))
        video, time, scorer = int(row[col["video"]]), str(row[col["time"]]).strip(), int(row[col["scorer"]])
        if time not in TIMES:
            raise ValueError(f"video {video:03d}: time {time!r} outside {TIMES}")
        v = per_video.setdefault(video, {"person_ns": child, "time": time, "rows": 0, "scorers": set(),
                                         "scored_rows": 0})
        if v["person_ns"] != child or v["time"] != time:
            raise ValueError(f"video {video:03d} seen with two children or two times")
        v["rows"] += 1
        v["scorers"].add(scorer)
        any_scored = False
        for item in ITEMS:
            c = classify(row[col[item]])
            counts[item][time][c] += 1
            counts[item]["all"][c] += 1
            any_scored |= c == "scored"
        v["scored_rows"] += int(any_scored)
    return counts, per_video, n_lock, n_empty


def child_folds(children, k=FOLDS, seed=FOLD_SEED):
    """{child: fold} of mdfm.eval.probes.person_folds over the sorted unique children."""
    from mdfm.eval import probes
    ids = np.array(sorted(set(children)))
    f = probes.person_folds(ids, k, seed)
    return {str(c): int(x) for c, x in zip(ids, f)}


def spearman_interval(rho, n, var=SPEARMAN_VAR, z=Z95):
    """(low, high, half width in z units) of the Fisher-z 95 percent interval of a Spearman correlation rho on n units."""
    if n <= 3:
        raise ValueError(f"n = {n}: the Fisher-z interval needs n > 3")
    se = math.sqrt(var / (n - 3))
    c = math.atanh(rho)
    return math.tanh(c - z * se), math.tanh(c + z * se), z * se


def ast_constant(path, name):
    """The literal value of the module-level assignment `name` in a Python file, parsed with ast."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not assigned at module level in {path}")


def manifest_rows(path, sources):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r["source"] in sources]


def main():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    from _common import parse_args  # noqa: E402
    import moveatlas_stamp  # noqa: E402
    a = parse_args(__doc__.splitlines()[0])
    log = Log()
    root = a.moveatlas
    facts = {"task": "G8X", "note": "counts only; each item cell of a training row converted to a number only to class "
                                    "it, the class kept and never the value; no rating, model output or statistic of "
                                    "a DIS item value; no cell of a lockbox row other than its ID"}

    # 1. manifest, lockbox and tiers
    man_path = os.path.join(HERE, "cache", "manifest", "manifest_records.csv.gz")
    rows = manifest_rows(man_path, {MODYS, HUBU})
    mod = [r for r in rows if r["source"] == MODYS]
    hub = [r for r in rows if r["source"] == HUBU]
    lock = sorted({r["person_id"] for r in mod if r["lockbox"] == "True"})
    train = sorted({r["person_id"] for r in mod if r["lockbox"] == "False"})
    assert len(lock) == 7 and len(train) == 27 and not set(lock) & set(train), (len(lock), len(train))
    with open(os.path.join(HERE, "reports", "lockbox", "lockbox_persons.csv"), encoding="utf-8", newline="") as f:
        lb = list(csv.DictReader(f))
    assert sorted(r["person_id"] for r in lb if r["source"] == MODYS) == lock, "manifest and lockbox file differ"
    with open(os.path.join(HERE, "reports", "manifest_tiers", "tiers_by_source.csv"), encoding="utf-8",
              newline="") as f:
        tiers = [r for r in csv.DictReader(f) if r["source"] in (MODYS, HUBU)]
    facts["modys_manifest"] = {"records": len(mod), "records_lockbox": sum(r["lockbox"] == "True" for r in mod),
                               "children_training": len(train), "children_lockbox": len(lock),
                               "records_with_fps": sum(bool(r["fps"].strip()) for r in mod),
                               "modalities": sorted({r["modality"] for r in mod}),
                               "tiers": sorted({r["tier"] for r in mod})}
    facts["hubu_manifest"] = {"records": len(hub), "lockbox_rows": sum(r["source"] == HUBU for r in lb),
                              "tier_table_rows": [dict(r) for r in tiers if r["source"] == HUBU]}
    facts["modys_tier_rows"] = [dict(r) for r in tiers if r["source"] == MODYS]
    log(f"manifest: {len(mod)} modys_video records ({facts['modys_manifest']['records_lockbox']} lockbox), "
        f"{len(train)} training and {len(lock)} lockbox children; {len(hub)} hubu_fis_fingertap records; "
        f"tier rows hubu {len(facts['hubu_manifest']['tier_table_rows'])}")

    # 2. the scoring sheet: classes only
    sheet = os.path.join(root, "datasets", "pose", MODYS, "raw", "data_clinical_scoring.xlsx")
    counts, per_video, n_lock = read_sheet(sheet, lock, train, log)
    with open(os.path.join(HERE, "reports", "modys_structure", "scorer_by_video.csv"), encoding="utf-8",
              newline="") as f:
        sbv = {int(r["video"]): r for r in csv.DictReader(f)}
    assert set(sbv) == set(per_video), "training videos differ from reports/modys_structure/scorer_by_video.csv"
    for v, d in per_video.items():
        r = sbv[v]
        assert r["person_ns"] == d["person_ns"] and r["time"] == d["time"] and int(r["n_scoring_rows"]) == d["rows"]
        assert r["scorers"] == "+".join(str(s) for s in sorted(d["scorers"]))
    log(f"training videos {len(per_video)}: equal to reports/modys_structure/scorer_by_video.csv (child, time, rows, "
        "scorers)")
    both = {v for v, r in sbv.items() if r["in_zip_named_lying"] == "True" and r["in_zip_named_sitting"] == "True"}
    facts["sheet"] = {
        "training_rows": sum(d["rows"] for d in per_video.values()), "lockbox_rows_skipped": n_lock,
        "training_videos": len(per_video),
        "videos_with_a_scored_row": sum(d["scored_rows"] > 0 for d in per_video.values()),
        "videos_with_a_scored_row_by_time": {t: sum(d["scored_rows"] > 0 and d["time"] == t
                                                    for d in per_video.values()) for t in TIMES},
        "videos_with_a_scored_row_in_both_zips": sum(d["scored_rows"] > 0 and v in both for v, d in per_video.items()),
        "rows_without_any_scored_item": sum(d["rows"] - d["scored_rows"] for d in per_video.values()),
        "item_classes": {i: {t: {c: int(counts[i][t][c]) for c in CLASSES} for t in TIMES + ("all",)}
                         for i in ITEMS}}
    log("item classes (all training rows): " + "; ".join(
        f"{i} " + "/".join(str(counts[i]["all"][c]) for c in CLASSES) for i in ITEMS))

    # 3. child folds, fixed before any dev read
    folds = child_folds(train)
    vids = collections.defaultdict(list)
    for v, d in per_video.items():
        vids[d["person_ns"]].append(v)
    fold_path = os.path.join(a.out, "modys_child_folds.csv")
    with open(fold_path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["person_ns", "fold", "videos_on_sheet", "videos_in_both_zips"])
        for c in sorted(folds):
            w.writerow([c, folds[c], len(vids[c]), sum(v in both for v in vids[c])])
    fold_sha = sha256_file(fold_path)
    with open(os.path.join(a.out, "modys_child_folds.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{fold_sha}  modys_child_folds.csv\n")
    facts["folds"] = {"k": FOLDS, "seed": FOLD_SEED, "function": "mdfm.eval.probes.person_folds",
                      "children_per_fold": {str(k): sum(x == k for x in folds.values()) for k in range(FOLDS)},
                      "videos_per_fold": {str(k): sum(len(vids[c]) for c in folds if folds[c] == k)
                                          for k in range(FOLDS)},
                      "sha256": fold_sha}
    log(f"child folds: {facts['folds']['children_per_fold']} children, {facts['folds']['videos_per_fold']} videos; "
        f"SHA-256 {fold_sha}")

    # 4. precision from counts only (planted values)
    facts["spearman_interval_from_counts"] = {
        name: {str(rho): dict(zip(("low", "high", "half_width_z"), (round(x, 4) for x in spearman_interval(rho, n))))
               for rho in PLANTED} | {"n": n}
        for name, n in CHILD_COUNTS.items()}

    # 5. hubu: documentation and code only
    card = os.path.join(root, "datasets", "pose", HUBU, "datacard.md")
    with open(card, encoding="utf-8") as f:
        card_lines = f.read().splitlines()
    status = [ln for ln in card_lines if ln.startswith("- status")]
    assert len(status) == 1
    eval_set = ast_constant(os.path.join(HERE, "scripts", "build_tiers_and_lockbox.py"), "EVAL")
    stems = ast_constant(os.path.join(HERE, "mdfm", "model", "tokenize.py"), "STEMS")
    inputs = ast_constant(os.path.join(HERE, "mdfm", "model", "tokenize.py"), "INPUT_MODALITIES")
    from mdfm.data.canonical import Refused
    from mdfm.gates import common
    guard = {}
    for src in (HUBU, MODYS):
        try:
            g = common.guard_eval_read("G8", src, None)
            guard[src] = f"admitted: tier {g['tier']}, eval {g['eval']}"
        except Refused as e:
            guard[src] = f"refused: {e}"
    facts["hubu"] = {"datacard_sha256": sha256_file(card), "datacard_status_line": status[0],
                     "in_eval_set_of_build_tiers": HUBU in eval_set, "tokenizer_stems": list(stems),
                     "tokenizer_input_modalities": list(inputs),
                     "a_hand_stem": any("hand" in s or "finger" in s for s in stems)}
    facts["eval_guard_today"] = guard
    log(f"hubu: EVAL set {HUBU in eval_set}; tokenizer stems {list(stems)}; guard: {guard[HUBU]}")

    # 6. outputs
    with open(os.path.join(a.out, "facts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1)
        f.write("\n")
    write_summary(a.out, facts)
    with open(os.path.join(a.out, "run.log"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(log.lines) + "\n")
    outputs = ["summary.md", "facts.json", "modys_child_folds.csv", "modys_child_folds.sha256", "run.log"]
    prov = {"script": "scripts/census_g8x_lines.py", "moveatlas_root": root,
            "moveatlas_stamp": moveatlas_stamp.stamp(root),
            "inputs": {}, "outputs_sha256": {o: sha256_file(os.path.join(a.out, o)) for o in outputs},
            "python": sys.version.split()[0], "numpy": np.__version__}
    for rel in (f"datasets/pose/{MODYS}/raw/data_clinical_scoring.xlsx", f"datasets/pose/{HUBU}/datacard.md"):
        prov["inputs"][rel] = {"bytes": os.path.getsize(os.path.join(root, rel)),
                               "sha256": sha256_file(os.path.join(root, rel))}
    for rel in ("cache/manifest/manifest_records.csv.gz", "reports/lockbox/lockbox_persons.csv",
                "reports/manifest_tiers/tiers_by_source.csv", "reports/modys_structure/scorer_by_video.csv",
                "scripts/build_tiers_and_lockbox.py", "mdfm/model/tokenize.py", "mdfm/eval/probes.py"):
        prov["inputs"][rel] = {"bytes": os.path.getsize(os.path.join(HERE, rel)),
                               "sha256": sha256_file(os.path.join(HERE, rel))}
    with open(os.path.join(a.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    for o in outputs + ["_provenance.json"]:
        print(f"{sha256_file(os.path.join(a.out, o))}  {o}")


def write_summary(out, facts):
    L = []
    add = L.append
    m, s, fo, h = facts["modys_manifest"], facts["sheet"], facts["folds"], facts["hubu"]
    add("# G8X structure census (counts only)")
    add("")
    add("scripts/census_g8x_lines.py; MoveAtlas stamp in _provenance.json. No statistic of a DIS item value, no rating "
        "and no model output was computed: each item cell of a training row was converted to a number only to class it "
        "(scored 0 to 4, not scored 999, blank, other) and only the class kept; of a lockbox row only the ID cell was "
        "read. No coordinate zip and no file of the hubu_fis_fingertap raw folder was opened.")
    add("")
    add("## modys_video")
    add("")
    add(f"- Manifest: {m['records']} records ({m['records_lockbox']} lockbox), {m['children_training']} training and "
        f"{m['children_lockbox']} lockbox children, records with an fps value {m['records_with_fps']}, modality "
        f"{', '.join(m['modalities'])}, tier {', '.join(m['tiers'])}.")
    add(f"- Scoring sheet, training children: {s['training_rows']} rows of {s['training_videos']} videos; "
        f"{s['lockbox_rows_skipped']} lockbox rows skipped on the ID cell. Videos with at least one scored item: "
        f"{s['videos_with_a_scored_row']} (by visit "
        + ", ".join(f"{t} {n}" for t, n in s["videos_with_a_scored_row_by_time"].items())
        + f"; held in both zips {s['videos_with_a_scored_row_in_both_zips']}); rows without any scored item: "
          f"{s['rows_without_any_scored_item']}.")
    add("")
    add("Item classes over the training rows (scored / not scored / blank / other):")
    add("")
    add("| item | all | T0 | T3 | T12 |")
    add("|---|---|---|---|---|")
    for i, d in s["item_classes"].items():
        add(f"| {i} | " + " | ".join("/".join(str(d[t][c]) for c in CLASSES) for t in ("all",) + TIMES) + " |")
    add("")
    add(f"- Child folds ({fo['function']}, k {fo['k']}, seed {fo['seed']}): children per fold "
        f"{fo['children_per_fold']}, videos per fold {fo['videos_per_fold']}; modys_child_folds.csv, SHA-256 "
        f"{fo['sha256']}.")
    add("")
    add("Spearman 95 percent interval from counts only (Fisher z, variance 1.06 / (n - 3)), at planted values:")
    add("")
    add("| children | n | planted 0.35 | planted 0.50 | half width (z units) |")
    add("|---|---|---|---|---|")
    for name, d in facts["spearman_interval_from_counts"].items():
        a, b = d["0.35"], d["0.5"]
        add(f"| {name} | {d['n']} | [{a['low']}, {a['high']}] | [{b['low']}, {b['high']}] | {a['half_width_z']} |")
    add("")
    add("## hubu_fis_fingertap")
    add("")
    add(f"- Manifest records {facts['hubu_manifest']['records']}, lockbox rows {facts['hubu_manifest']['lockbox_rows']}, "
        f"tier-table rows {len(facts['hubu_manifest']['tier_table_rows'])}; in the EVAL set of "
        f"scripts/build_tiers_and_lockbox.py: {h['in_eval_set_of_build_tiers']}.")
    add(f"- EVAL read guard today (mdfm.gates.common.guard_eval_read, gate G8, no rule): "
        f"{facts['eval_guard_today'][HUBU]}")
    add(f"- Tokenizer input stems (mdfm/model/tokenize.py STEMS): {', '.join(h['tokenizer_stems'])}; a hand or finger "
        f"stem: {h['a_hand_stem']}.")
    add(f"- Datacard status line (MoveAtlas documentation, SHA-256 {h['datacard_sha256']}), verbatim:")
    add("")
    add("  " + h["datacard_status_line"])
    add("")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
