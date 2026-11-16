"""scripts/e1_read.py (task E1R): the reading of E1 against pilot v1 under sections 3 and 3a of
docs/proposals/JEPA_EXPLORATION_DRAFT.md. CPU only; no lockbox, EVAL, CODY or care_pd record.

v1's and seed 2's files are read from this tree (the margins are recomputed from them and checked against the numbers
that section 3 prints); every E1 input is synthetic, written into a temporary tree: the run log (three calls, two
resumes, as the record of decisions.md 2026-10-07 21:00 describes), the job_log.csv rows, the B-end reports made from
v1's own report files with E1's checkpoint name and SHA-256 and shifted values, J0's results.json, and the early
check folders (v1's ckpt_B_004000 files under E1's name and SHA-256). The pass unit is replaced by a reader that
returns v1's and seed 2's S_pass(12) rows as the JP0 reading holds them and synthetic rows for E1 (the default reader,
scripts/jp0_read.py analyse_checkpoint, is exercised by `scripts/e1_read.py --margins` on v1's and seed 2's J0
outputs). No real E1 output is opened.

Outcomes covered: MET (success); NOT MET by the main line (its point, and its point passing with the paired lower
bound at or below 0); NOT MET by each non-inferiority line; NOT MET by R0 (an alarm, a report mismatch, an exit 1, a
verify event at another step); an inconsistent resume lineage (exit 4, nothing written); a crash fragment in the log
(recovered, reported, not a condition); NOT EVALUATED (exit 3, the missing inputs named, nothing written); the section
3a hash refusal (exit 2, nothing read or written); an existing results.json (exit 5); a printed margin that differs or
sits in the wrong part of section 3 (exit 4); the G10 caution at 5.0 and 4.9 points; the route under NOT MET and under
the G10 caution; the early check that never stops the reading; and the unit pieces: lineage, R0, predictions, the
route, the one-end relation, m_P, the log parser, the seed differences beside.
"""
import copy
import csv
import json
import os
import shutil
import sys

import numpy as np
import pytest

from mdfm import paths

sys.path.insert(0, os.path.join(paths.TREE, "scripts"))
import e1_read as E  # noqa: E402

TREE = paths.TREE
V1_B, S2_B = E.NAMES["v1_B"], E.NAMES["s2_B"]
NAME = E.E1_B_DEFAULT
SHA_B = "e1" * 32
SHA_4000 = "e4" * 32
HAVE_REPORTS = all(os.path.exists(os.path.join(TREE, p)) for p in (
    f"reports/gates/g3/diag_{V1_B}/diag.json", f"reports/gates/g3/diag_{S2_B}/diag.json",
    f"reports/gates/g3/dev_{V1_B}/g3_dev.json", f"reports/gates/g1/dev_{V1_B}/g1.json",
    f"reports/gates/g3/curve_{E.NAMES['v1_4000']}/g3_dev.json", f"reports/bench/frozen_tokens_{V1_B}/results.json",
    E.JP0_RESULTS_REL, E.F1_RESULTS_REL, E.SEEDS_REL))
needs_reports = pytest.mark.skipif(not HAVE_REPORTS, reason="v1, seed 2, J0 or JP0 reports not found")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, doc):
    write(path, json.dumps(doc))


@pytest.fixture(scope="module")
def sets_locked():
    if not HAVE_REPORTS:
        pytest.skip("reports not found")
    return E.default_sets(TREE)


@pytest.fixture(scope="module")
def jp0doc():
    if not HAVE_REPORTS:
        pytest.skip("reports not found")
    return load(os.path.join(TREE, E.JP0_RESULTS_REL))


# ---------------------------------------------------------------- the synthetic E1 inputs
def log_records(alarm=False, mismatch=None, resume2=f"{E.E1_RUN}/ckpt_B_024000.pt", final_sha=SHA_B,
                verify_sha=None, verify_step=32044, resume1_step=22000):
    """Three calls: init_from to a crash after 22,950; resume from 22,000 to a crash after 24,500; resume from
    24,000 to the end, the verify event at 32,044."""
    run = f"cache/runs/{E.E1_RUN}"
    start = {"event": "start", "config_sha256": E.E1_CONFIG_SHA256, "steps": [10681, 32044],
             "report_mismatches": mismatch or [], "run_dir": run}

    def logs(a, b):
        return [{"event": "log", "phase": "B", "step": s, "time": "2026-10-07T00:00:00"} for s in range(a, b + 1, 50)]

    def ck(s, sha=None):
        return {"event": "checkpoint", "phase": "B", "step": s, "path": f"{run}\\ckpt_B_{s:06d}.pt",
                "sha256": sha or f"{s:064d}"}

    c1 = [dict(start), {"event": "init_from", "phase": "B", "step": 0,
                        "source": {"path": "<repo>/" + E.SOURCE_PATH, "sha256": E.SOURCE_SHA256}}]
    c1 += logs(50, 4000) + [ck(4000, SHA_4000)] + logs(4050, 22000) + [ck(22000)] + logs(22050, 22950)
    c2 = [dict(start), {"event": "resume", "phase": "B", "step": resume1_step, "drift": [],
                        "path": f"<repo>\\cache\\runs\\{E.E1_RUN}\\ckpt_B_022000.pt"}]
    c2 += logs(22050, 24000) + [ck(24000)] + logs(24050, 24500)
    rs = int(resume2.split("_")[-1].split(".")[0])
    c3 = [dict(start), {"event": "resume", "phase": "B", "step": rs, "drift": [],
                        "path": "<repo>\\cache\\runs\\" + resume2.replace("/", "\\")}]
    c3 += logs(rs + 50, 32000)
    if alarm:
        c3.append({"event": "alarm", "phase": "B", "step": 30000, "source": "log", "alarms": ["x"]})
    c3 += logs(32044, 32044) + [{"event": "phase_end", "phase": "B", "step": 32044}, ck(32044, final_sha),
                                {"event": "end", "phase": "B", "step": 32044}]
    c3.append({"event": "verify", "phase": "B", "step": verify_step, "path": f"{run}\\ckpt_B_032044.pt",
               "sha256": verify_sha or final_sha, "report_mismatches": [], "drift": []})
    return c1 + c2 + c3


def crash_log_text():
    """The log of log_records as text, with the last record of call 1 cut to a fragment that the start event of call
    2 joins (no newline), and NUL padding before the start event of call 3."""
    lines = [json.dumps(r) for r in log_records()]
    starts = [i for i, t in enumerate(lines) if '"event": "start"' in t]
    k2, k3 = starts[1], starts[2]
    lines[k3] = "\x00" * 16 + lines[k3]
    lines[k2 - 1] = lines[k2 - 1][:25] + lines[k2]
    del lines[k2]
    return "\n".join(lines) + "\n"


def job_log_rows(root, final_exit=0):
    head = ["name", "cls", "priority", "state", "argv", "exit_code", "start", "end"]
    base = ["python", "scripts/train_stage1.py", "--phase", "B", "--init-from", E.SOURCE_PATH, "--config",
            "configs/stage1_pilot_jepa.yaml", "--run-dir", E.E1_RUN_DIR]
    rows = [["e1_jepa_train", 2, 1, "r0", json.dumps(base + ["--resume", f"{E.E1_RUN_DIR}/ckpt_B_022000.pt"]), 1,
             "", ""],
            ["e1_jepa_train_r3", 2, 1, "done" if final_exit == 0 else "failed",
             json.dumps(base + ["--resume", f"{E.E1_RUN_DIR}/ckpt_B_024000.pt"]), final_exit, "", ""]]
    p = os.path.join(root, E.JOB_LOG_REL)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)
    return p


def shift_diag(doc, ds, point_add=None):
    """Lower each person's stride by ds (a number, or a function (person, stride) -> shift) in every subset of the
    base row, the subset's stride MAE by the mean shift; then add point_add {(subset, field): value} to a subset's
    point."""
    f = ds if callable(ds) else (lambda p, v: ds)
    for sub in doc["rows"]["base"].values():
        if not isinstance(sub, dict) or "per_person" not in sub:
            continue
        applied = []
        for p, v in sub["per_person"].items():
            if isinstance(v, dict) and v.get("stride_cm") is not None:
                s = float(f(p, v["stride_cm"]))
                v["stride_cm"] -= s
                applied.append(s)
        if sub.get("stride_mae_cm") is not None and applied:
            sub["stride_mae_cm"] -= float(np.mean(applied))
    for (sub, field), add in (point_add or {}).items():
        doc["rows"]["base"][sub][field] += add
    return doc


def e1_tree(tmp_path, ds=3.0, g1a_add=0.0, insole_add=0.0, walkway_add=0.0, point_add=None, log=None, log_text=None,
            missing=(), early=True, early_sha=SHA_4000, leak_add=0.0, p_shift=0.0):
    """A temporary tree holding E1's inputs (module docstring); ds lowers E1's stride MAE per person against v1's,
    p_shift lowers E1's per-stride probe line, early_sha is the SHA-256 the early reports carry."""
    root = str(tmp_path / "e1tree")
    if log_text is None:
        log_text = "".join(json.dumps(r) + "\n" for r in (log if log is not None else log_records()))
    write(os.path.join(root, E.E1_RUN_DIR, "log.jsonl"), log_text)
    job_log_rows(root)
    g3, g1, bench = (os.path.join(root, "reports", x) for x in ("gates/g3", "gates/g1", "bench"))
    ck = {"name": NAME, "sha256": SHA_B, "weights": "ema"}
    d = shift_diag(load(os.path.join(TREE, f"reports/gates/g3/diag_{V1_B}/diag.json")), ds, point_add)
    d["checkpoint"] = dict(d["checkpoint"], **ck)
    dump(os.path.join(g3, f"diag_{NAME}", "diag.json"), d)
    write(os.path.join(g3, f"diag_{NAME}", "events.csv"), "row,kind\n")
    g = load(os.path.join(TREE, f"reports/gates/g3/dev_{V1_B}/g3_dev.json"))
    g["checkpoint"] = dict(g["checkpoint"], **ck)
    for r in g["results_dev"]:
        if r["line"] == "ic_median_ms_insole":
            r["point"] += insole_add
        if r["line"] == "ic_median_ms_walkway":
            r["point"] += walkway_add
    dump(os.path.join(g3, f"dev_{NAME}", "g3_dev.json"), g)
    h = load(os.path.join(TREE, f"reports/gates/g1/dev_{V1_B}/g1.json"))
    h["checkpoint"] = dict(h["checkpoint"], **ck)
    for x in h["lines"]:
        if x["line"].startswith("G1a"):
            x["point"] += g1a_add
    dump(os.path.join(g1, f"dev_{NAME}", "g1.json"), h)
    j0v1 = load(os.path.join(TREE, f"reports/bench/frozen_tokens_{V1_B}/results.json"))
    p1 = j0v1["passes"]["latent"]["lines"]["dev12"]["stride_mae_cm"]["point"]
    dump(os.path.join(bench, f"frozen_tokens_{NAME}", "results.json"),
         {"partial": False, "checkpoint": ck,
          "passes": {"latent": {"lines": {"dev12": {"stride_mae_cm": {"point": p1 - p_shift}}}}},
          "leakage": {"leakage_points": j0v1["leakage"]["leakage_points"] + leak_add}})
    write(os.path.join(bench, f"frozen_tokens_{NAME}", "per_pass.csv"), "trial_id,pass_id\n")
    if early:
        v4 = E.NAMES["v1_4000"]
        ck4 = {"name": E.E1_4000, "sha256": early_sha, "weights": "ema"}
        cur = os.path.join(g3, f"curve_{E.E1_4000}")
        os.makedirs(cur, exist_ok=True)
        c = load(os.path.join(TREE, f"reports/gates/g3/curve_{v4}/g3_dev.json"))
        c["checkpoint"] = dict(c["checkpoint"], **ck4)
        dump(os.path.join(cur, "g3_dev.json"), c)
        shutil.copyfile(os.path.join(TREE, f"reports/gates/g3/curve_{v4}/per_person.csv"),
                        os.path.join(cur, "per_person.csv"))
        h4 = load(os.path.join(TREE, f"reports/gates/g1/dev_{v4}_lt2/g1.json"))
        h4["checkpoint"] = dict(h4["checkpoint"], **ck4)
        dump(os.path.join(g1, f"dev_{E.E1_4000}_lt2", "g1.json"), h4)
        dump(os.path.join(bench, f"frozen_tokens_{E.E1_4000}", "results.json"), {"partial": False})
        write(os.path.join(bench, f"frozen_tokens_{E.E1_4000}", "per_pass.csv"), "trial_id,pass_id\n")
    for m in missing:
        os.remove(os.path.join(root, m))
    return root


def line(per_person):
    v = np.array(list(per_person.values()), np.float64)
    return {"point": float(v.mean()), "interval": None, "n_persons": len(v), "per_person": dict(per_person)}


class FakeReader:
    """v1's and seed 2's S_pass(12) rows as the JP0 reading holds them; E1's: v1's rows with the latent lowered by
    lat_shift and the decoded lowered by dec_shift per person; at ckpt_B_004000 the rows of v1's B end."""

    def __init__(self, jp0doc, lat_shift=0.0, dec_shift=3.0):
        self.doc, self.lat, self.dec = jp0doc, lat_shift, dec_shift
        self.calls = []

    def rows_of(self, key):
        lines = self.doc["checkpoints"][key]["stride"]["lines"]["dev12"]
        return {r: {k: lines[r].get(k) for k in ("point", "interval", "n_persons", "per_person")}
                for r in E.CHAIN if r in lines}

    def __call__(self, name, where):
        self.calls.append(name)
        if name == V1_B:
            return {"rows": self.rows_of("v1_B"), "checkpoint_sha256": None}
        if name == S2_B:
            return {"rows": self.rows_of("s2_B"), "checkpoint_sha256": None}
        base = self.rows_of("v1_B")
        if name in (E.NAMES["v1_4000"], E.NAMES["s2_4000"]):
            return {"rows": base, "checkpoint_sha256": None}
        rows = {r: line({p: v - (self.lat if r == "latent" else self.dec if r == "decoded" else 0.0)
                         for p, v in base[r]["per_person"].items()}) for r in ("stem", "latent", "mu_z", "decoded")}
        return {"rows": rows, "checkpoint_sha256": SHA_B if name == NAME else SHA_4000}


class FailingEarlyReader(FakeReader):
    def __call__(self, name, where):
        if name == E.E1_4000:
            raise E.Stop(E.EXIT_HARNESS, "refit check failed")
        return super().__call__(name, where)


def run(tmp_path, root, sets_locked, reader, **kw):
    sets, locked = sets_locked
    out = str(tmp_path / "out")
    res = E.run(TREE, out, e1_tree=root, sets=sets, locked=locked, pass_reader=reader,
                job_log=os.path.join(root, E.JOB_LOG_REL), seed_difference=str(tmp_path / "no_seed_difference.json"),
                stamp={"head": "test"}, **kw)
    return res, out


def stop_code(fn):
    with pytest.raises(E.Stop) as e:
        fn()
    return e.value.code, e.value.message


def v1_per_person_12():
    d = load(os.path.join(TREE, f"reports/gates/g3/diag_{V1_B}/diag.json"))
    pp = d["rows"]["base"][E.WB().SUB12]["per_person"]
    return {p: v["stride_cm"] for p, v in pp.items() if isinstance(v, dict) and v.get("stride_cm") is not None}


# ---------------------------------------------------------------- outcomes
@needs_reports
def test_success_writes_the_reading_and_never_rewrites_it(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, ds=3.0, insole_add=3.5)
    res, out = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET and res["failed"] == []
    assert res["r0"]["holds"] and res["r0"]["exit_from"].startswith("job_log.csv row e1_jepa_train_r3")
    assert set(res["r0"]["checks"]) == {"verify_B_32044", "report_mismatches_empty", "no_alarm", "exit_0"}
    s = res["b_end"]["lines"]["S12"]
    assert abs(s["at_most"] - (s["line_v1"] - s["margin"])) < 1e-12 and f"{s['at_most']:.2f}" == "17.56"
    assert abs(s["paired_v1_minus_e1"]["mean"] - 3.0) < 1e-9 and s["paired_v1_minus_e1"]["n_persons"] == 12
    assert all(res["b_end"]["lines"][k]["holds"] for k in E.LINES)
    # the insole IC line is 2 x the seed difference (19.64 ms), not 5 percent of v1's 12.50 ms (13.13 ms)
    ins = res["b_end"]["lines"]["IC_insole"]
    assert 1.05 * ins["line_v1"] < ins["value"] < ins["at_most"] and f"{ins['at_most']:.2f}" == "19.64"
    # predictions: J0 per stride CARRY (the probe within its band) fails by success; 3a BOTTLENECK holds
    assert res["j0_section3"]["reading"] == E.CARRY and res["jp0_reading"]["main_reading"] == E.BOTTLENECK
    assert res["predictions"]["section3_per_stride"]["holds"] is False
    assert res["predictions"]["section3a_pass_unit"]["holds"] is True
    assert res["route_parts"]["prediction_next"] == E.NEXT["bottleneck_holds"] and res["predictions"]["agree"] is False
    assert res["route_parts"]["candidate"] is True and res["route"].endswith(E.CANDIDATE)
    # the JP0 reading is GAIN: routing rule 1 is stated first
    assert res["jp0_reading"]["readings"]["GAIN"] == "GAIN" and res["route"].startswith(E.GAIN_RULE_1)
    assert res["pass_unit"]["relations"]["probe_latent"]["better"] is True
    assert res["pass_unit"]["e1_readings"]["R"] == E.PENDING and res["g10"]["caution"] is False
    assert res["lineage"]["consistent"] and res["lineage"]["agrees_with_record"]
    assert res["lineage"]["resumes"] == [f"{E.E1_RUN}/ckpt_B_022000.pt", f"{E.E1_RUN}/ckpt_B_024000.pt"]
    assert res["jp2_go_line"]["go"] is not None and res["early"]["missing"] == [] and res["early"]["error"] is None
    assert all(res["early"]["lines"][k]["evaluated"] for k in E.LINES)
    assert res["early"]["checkpoint_sha256_from_log"] == SHA_4000
    for f in ("results.json", "summary.md", "_provenance.json"):
        with open(os.path.join(out, f), encoding="utf-8", newline="") as fh:
            t = fh.read()
        assert "\r" not in t and chr(0x2014) not in t and chr(0x2013) not in t
    assert load(os.path.join(out, "results.json"))["addendum"]["sha256"] == E.ADDENDUM_SHA256
    code, _ = stop_code(lambda: run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5)))
    assert code == E.EXIT_EXISTS


@needs_reports
def test_the_main_line_fails(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, ds=1.0)
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=0.0, dec_shift=1.0))
    assert res["verdict"] == E.NOT_MET and res["failed"] == ["success line S12"]
    assert res["b_end"]["lines"]["S12"]["point_holds"] is False and res["r0"]["holds"]
    # per stride: CARRY with the probe unchanged holds; pass unit: BOTTLENECK, neither improved
    assert res["predictions"]["section3_per_stride"]["holds"] is True
    assert res["predictions"]["section3a_pass_unit"]["next"] == E.NEXT["bottleneck_neither"]
    assert res["route_parts"]["prediction_next"] == E.NEXT["bottleneck_neither"]
    assert res["route"].endswith(E.NEXT["bottleneck_neither"]) and not res["route_parts"]["candidate"]


@needs_reports
def test_the_point_passes_but_the_paired_lower_bound_does_not(tmp_path, monkeypatch, sets_locked, jp0doc):
    """Two persons carry the whole gain (E1 at 2 cm for them), the ten others are 0.5 cm worse: the point is under
    17.56 cm, the bootstrap draws none of the two often enough for the lower bound to sit at -0.5 cm."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    pp = v1_per_person_12()
    top2 = sorted(pp, key=pp.get)[-2:]
    root = e1_tree(tmp_path, ds=lambda p, v: (v - 2.0) if p in top2 else -0.5)
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    s = res["b_end"]["lines"]["S12"]
    assert s["point_holds"] is True and s["value"] <= s["at_most"]
    assert s["paired_v1_minus_e1"]["interval"][0] <= 0 and s["paired_v1_minus_e1"]["lower_above_0"] is False
    assert s["holds"] is False and res["verdict"] == E.NOT_MET and res["failed"] == ["success line S12"]


@needs_reports
@pytest.mark.parametrize("case", ["V12", "S14", "V14", "IC", "IC_insole_G1a"])
def test_a_noninferiority_line_fails(tmp_path, monkeypatch, sets_locked, jp0doc, case):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    sub12, sub14 = E.WB().SUB12, E.WB().SUB14
    kw = {"V12": {"point_add": {(sub12, "speed_mae_ms"): 0.02}},
          "S14": {"point_add": {(sub14, "stride_mae_cm"): 5.0}},
          "V14": {"point_add": {(sub14, "speed_mae_ms"): 0.04}},
          "IC": {"walkway_add": 1.0},
          "IC_insole_G1a": {"g1a_add": 5.0, "insole_add": 7.5}}[case]
    want = {"V12": ["V12"], "S14": ["S14"], "V14": ["V14"], "IC": ["IC"], "IC_insole_G1a": ["IC_insole", "G1a"]}[case]
    root = e1_tree(tmp_path, ds=3.0, **kw)
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.NOT_MET
    assert res["failed"] == [f"non-inferiority {k}" for k in want]
    assert res["b_end"]["lines"]["S12"]["holds"] is True
    # the prediction held, but a failed line blocks adoption and the route names it
    assert res["route_parts"]["prediction_next"] == E.NEXT["bottleneck_holds"]
    assert res["route_parts"]["candidate"] is False and "not a candidate" in res["route"]
    assert all(f"non-inferiority {k}" in res["route"] for k in want)


@needs_reports
@pytest.mark.parametrize("case", ["alarm", "mismatch", "exit"])
def test_r0_fails(tmp_path, monkeypatch, sets_locked, jp0doc, case):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    log = {"alarm": log_records(alarm=True), "mismatch": log_records(mismatch=["records_sha256"]),
           "exit": log_records()}[case]
    root = e1_tree(tmp_path, ds=3.0, log=log)
    if case == "exit":
        job_log_rows(root, final_exit=1)
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    want = {"alarm": "no_alarm", "mismatch": "report_mismatches_empty", "exit": "exit_0"}[case]
    assert res["verdict"] == E.NOT_MET and res["r0"]["failed"] == [want]
    assert res["failed"] == [f"R0: {want}"]
    assert res["route_parts"]["candidate"] is False and f"R0: {want}" in res["route"]
    assert "not a candidate" in res["route"]


@needs_reports
@pytest.mark.parametrize("add,caution", [(5.0, True), (4.9, False)])
def test_the_g10_caution_blocks_adoption_from_5_points(tmp_path, monkeypatch, sets_locked, jp0doc, add, caution):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, ds=3.0, leak_add=add)
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET and res["g10"]["caution"] is caution
    assert abs(res["g10"]["rise"] - add) < 1e-9
    assert res["route_parts"]["candidate"] is (not caution)
    assert ("G10 caution applies" in res["route"]) is caution


@needs_reports
def test_an_inconsistent_lineage_stops_and_writes_nothing(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, log=log_records(resume2=f"{E.E1_RUN}/ckpt_B_026000.pt"))
    code, msg = stop_code(lambda: run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5)))
    assert code == E.EXIT_HARNESS and "integrity question" in msg and "ckpt_B_026000" in msg
    assert not os.path.exists(str(tmp_path / "out"))


@needs_reports
def test_a_crash_fragment_in_the_log_is_reported_not_a_condition(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, log_text=crash_log_text())
    res, _ = run(tmp_path, root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET and res["r0"]["holds"] and res["lineage"]["consistent"]
    assert res["r0"]["unparsed_log_lines"] == 1 and len(res["r0"]["recovered_line_numbers"]) == 2
    assert [c["how"] for c in res["lineage"]["calls"]] == ["init_from", "resume", "resume"]
    assert res["lineage"]["calls"][1]["steps_trained_again"] == 900


@needs_reports
def test_not_evaluated_names_the_missing_inputs_and_writes_nothing(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = e1_tree(tmp_path, missing=(f"reports/gates/g3/diag_{NAME}/diag.json",
                                      f"reports/bench/frozen_tokens_{NAME}/results.json"))
    code, msg = stop_code(lambda: run(tmp_path, root, sets_locked, FakeReader(jp0doc)))
    assert code == E.EXIT_MISSING and "NOT EVALUATED" in msg
    assert f"diag_{NAME}/diag.json" in msg and f"frozen_tokens_{NAME}/results.json" in msg
    assert not os.path.exists(str(tmp_path / "out"))
    # a run that has not ended: no log at all
    shutil.rmtree(os.path.join(root, E.E1_RUN_DIR))
    code, msg = stop_code(lambda: run(tmp_path, root, sets_locked, FakeReader(jp0doc)))
    assert code == E.EXIT_MISSING and "log.jsonl" in msg and not os.path.exists(str(tmp_path / "out"))
    # the early inputs are reported, never required
    root2 = e1_tree(tmp_path / "b", early=False)
    res, _ = run(tmp_path / "b", root2, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert len(res["early"]["missing"]) == 5 and res["verdict"] == E.MET
    assert res["jp2_go_line"]["go"] is None and res["jp2_go_line"]["reason"].startswith("not evaluated")


@needs_reports
def test_the_early_check_never_stops_the_reading(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    # early reports of another checkpoint than the log's ckpt_B_004000: not evaluated, with the reason
    root = e1_tree(tmp_path / "a", early_sha="ab" * 32)
    res, _ = run(tmp_path / "a", root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET
    assert not any(res["early"]["lines"][k]["evaluated"] for k in E.LINES)
    assert all("SHA-256" in res["early"]["why"][k] for k in E.LINES)
    # the pass unit of ckpt_B_004000 fails: the go line is not evaluated
    root = e1_tree(tmp_path / "b")
    res, _ = run(tmp_path / "b", root, sets_locked, FailingEarlyReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET and res["jp2_go_line"]["go"] is None
    assert "refit check failed" in res["jp2_go_line"]["reason"]
    # a person guard of the curve rows: the early check is not evaluated
    go = E.GO()
    orig = go.values_at

    def guarded(tree, name, step, sets, locked):
        if name == E.E1_4000:
            raise AssertionError("a lockbox person in the curve rows")
        return orig(tree, name, step, sets, locked)

    monkeypatch.setattr(go, "values_at", guarded)
    root = e1_tree(tmp_path / "c")
    res, _ = run(tmp_path / "c", root, sets_locked, FakeReader(jp0doc, lat_shift=2.5))
    assert res["verdict"] == E.MET and "lockbox person" in res["early"]["error"]
    assert not any(res["early"]["lines"][k]["evaluated"] for k in E.LINES)


@needs_reports
def test_a_report_of_another_checkpoint_is_a_harness_defect(tmp_path, monkeypatch, sets_locked, jp0doc):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    # a consistent lineage whose verified checkpoint is not the one the B-end reports were made of
    root = e1_tree(tmp_path, log=log_records(final_sha="ab" * 32))
    code, msg = stop_code(lambda: run(tmp_path, root, sets_locked, FakeReader(jp0doc)))
    assert code == E.EXIT_HARNESS and "SHA-256" in msg and "integrity" not in msg
    assert not os.path.exists(str(tmp_path / "out"))


def test_the_hash_refusal_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = str(tmp_path / "tree")
    for r in (E.PROPOSAL_REL, "reports/bench/jepa_jp0_addendum.sha256"):
        os.makedirs(os.path.join(root, os.path.dirname(r)), exist_ok=True)
        shutil.copyfile(os.path.join(TREE, r), os.path.join(root, r))
    assert E.check_addendum(root) == E.ADDENDUM_SHA256
    p = os.path.join(root, E.PROPOSAL_REL)
    with open(p, encoding="utf-8", newline="") as f:
        t = f.read()
    write(p, t.replace("Readings at the B ends, on stride:", "Readings at the B ends, on stride (edited):"))

    def never(*a, **k):
        raise AssertionError("read after a hash refusal")

    monkeypatch.setattr(E, "prepare", never)
    monkeypatch.setattr(E, "read_log", never)
    out = str(tmp_path / "out")
    code, msg = stop_code(lambda: E.run(root, out, pass_reader=never, stamp={}))
    assert code == E.EXIT_HASH and "refused" in msg and not os.path.exists(out)
    assert E.main(["--tree", root, "--out", out]) == E.EXIT_HASH and not os.path.exists(out)


@needs_reports
def test_a_printed_margin_that_differs_is_a_harness_defect(sets_locked):
    m = E.margins(TREE, *sets_locked)
    text = E.section_3_text(TREE)
    assert len(E.check_printed(m, text)) == 16
    code, msg = stop_code(lambda: E.check_printed(m, text.replace("at most 17.56 cm", "at most 17.57 cm")))
    assert code == E.EXIT_HARNESS and "17.56" in msg
    m2 = copy.deepcopy(m)
    m2["b_end"]["G1a"]["at_most"] += 0.1
    assert stop_code(lambda: E.check_printed(m2, text))[0] == E.EXIT_HARNESS
    # the B-end and early S(14) lines swapped: each string still exists in the section, but in the other part
    swapped = (text.replace("S(14) at most 28.15 cm", "S(14) at most @@").replace("S(14) at most 27.58 cm",
                                                                                 "S(14) at most 28.15 cm")
               .replace("S(14) at most @@", "S(14) at most 27.58 cm"))
    assert "S(14) at most 28.15 cm" in swapped and "S(14) at most 27.58 cm" in swapped
    code, msg = stop_code(lambda: E.check_printed(m, swapped))
    assert code == E.EXIT_HARNESS and "S(14) at most 28.15 cm (in the B-end reading)" in msg
    assert "S(14) at most 27.58 cm (in the Early check bullet)" in msg


@needs_reports
def test_m_p_is_the_margin_rule_of_the_latent_row(jp0doc):
    fr = FakeReader(jp0doc)
    v1, s2 = {"rows": fr.rows_of("v1_B")}, {"rows": fr.rows_of("s2_B")}
    pm = E.pass_margins(v1, s2, jp0doc)
    Y1, Y2 = v1["rows"]["latent"]["point"], s2["rows"]["latent"]["point"]
    assert pm["P"]["Y1"] == Y1 and abs(pm["P"]["d"] - abs(Y1 - Y2)) < 1e-12
    assert pm["P"]["m"] == max(0.10 * Y1, 2.0 * abs(Y1 - Y2))
    assert 2.0 * abs(Y1 - Y2) > 0.10 * Y1          # here the seed term binds


# ---------------------------------------------------------------- units
def test_lineage_reports_the_calls_and_finds_the_breaks():
    lin = E.lineage(log_records())
    assert lin["consistent"] and lin["problems"] == [] and lin["agrees_with_record"]
    assert [c["how"] for c in lin["calls"]] == ["init_from", "resume", "resume"]
    assert [c["from_step"] for c in lin["calls"][1:]] == [22000, 24000]
    assert [c.get("steps_trained_again") for c in lin["calls"][1:]] == [950, 500]
    assert lin["final_verify"]["path"] == f"{E.E1_RUN}/ckpt_B_032044.pt"
    # a resume from a checkpoint no call logged, a resume beyond the last step reached, another config, a verify
    # SHA-256 other than the logged checkpoint's
    assert not E.lineage(log_records(resume2=f"{E.E1_RUN}/ckpt_B_026000.pt"))["consistent"]
    recs = log_records()
    for r in recs:
        if r.get("event") == "start":
            r["config_sha256"] = "0" * 64
            break
    assert any("config" in p for p in E.lineage(recs)["problems"])
    assert not E.lineage(log_records(verify_sha="ab" * 32))["consistent"]
    recs = [r for r in log_records() if r.get("event") != "init_from"]
    assert any("init_from" in p for p in E.lineage(recs)["problems"])
    # a resume event whose step is not the step at which its checkpoint was logged (21,000 for ckpt_B_022000)
    lin = E.lineage(log_records(resume1_step=21000))
    assert lin["problems"] == [f"call 2: resumes {E.E1_RUN}/ckpt_B_022000.pt at step 21000, logged at step 22000"]
    # one resume only: consistent, but not the record
    calls, _ = E.segments(log_records(resume2=f"{E.E1_RUN}/ckpt_B_022000.pt"))
    lin1 = E.lineage([r for c in calls[:1] + calls[2:] for r in c])
    assert lin1["consistent"], lin1["problems"]
    assert lin1["log_says"]["resume_events"] == 1 and not lin1["agrees_with_record"]
    # a call that crashed before its resume event and trained nothing is reported, not a break
    empty = log_records()
    i = next(k for k, r in enumerate(empty) if r.get("event") == "resume")
    empty.insert(i - 1, {"event": "start", "config_sha256": E.E1_CONFIG_SHA256, "steps": [10681, 32044],
                         "report_mismatches": []})
    assert E.lineage(empty)["consistent"]


def test_r0_reads_the_four_conditions_of_the_branch_rule():
    recs = log_records()
    lin = E.lineage(recs)
    r = E.r0_reading(recs, 0, lin, [])
    assert r["holds"] and r["exit_from"].startswith("inferred")
    assert set(r["checks"]) == {"verify_B_32044", "report_mismatches_empty", "no_alarm", "exit_0"}
    rows = [{"name": "e1_jepa_train_r3", "state": "failed", "exit_code": 1, "resume": f"{E.E1_RUN}/ckpt_B_024000.pt"}]
    assert E.r0_reading(recs, 0, lin, rows)["failed"] == ["exit_0"]
    nov = [x for x in recs if x.get("event") != "verify"]
    r = E.r0_reading(nov, 0, E.lineage(nov), [])
    assert not r["holds"] and "verify_B_32044" in r["failed"]
    # a verify event past the phase B end at another step (the lineage stays consistent)
    late = log_records(verify_step=32100)
    lin = E.lineage(late)
    assert lin["consistent"]
    assert E.r0_reading(late, 0, lin, [])["failed"] == ["verify_B_32044"]
    # unparsed lines are reported, not a condition
    r = E.r0_reading(recs, {"unparsed": 2, "unparsed_lines": [7, 9], "recovered_lines": [9]}, E.lineage(recs), [])
    assert r["holds"] and r["unparsed_log_lines"] == 2 and r["unparsed_line_numbers"] == [7, 9]


def test_parse_log_recovers_records_after_padding_or_a_fragment(tmp_path):
    p = str(tmp_path / "log.jsonl")
    write(p, json.dumps({"event": "a"}) + "\n" + "\x00" * 8 + json.dumps({"event": "b", "x": {"y": 1}}) + "\n"
          + '{"event": "log", "st' + json.dumps({"event": "c", "x": {"y": 2}}) + "\n" + "garbage\n\n")
    recs, info = E.parse_log(p)
    assert [r["event"] for r in recs] == ["a", "b", "c"] and recs[2]["x"] == {"y": 2}
    assert info == {"unparsed": 2, "unparsed_lines": [3, 4], "recovered_lines": [2, 3]}
    assert E.parse_log(str(tmp_path / "absent.jsonl"))[0] is None


def test_predictions_follow_section_3():
    P = E.predict
    assert P(E.CARRY, False, False, True)["holds"] is True
    assert P(E.CARRY, False, False, True)["next"] == E.NEXT["carry_holds"]
    assert P(E.CARRY, True, False, True)["holds"] is False
    assert P(E.CARRY, False, True, False)["next"] == E.NEXT["carry_probe_moved"]
    assert P(E.BOTTLENECK, True, True, False)["next"] == E.NEXT["bottleneck_holds"]
    assert P(E.BOTTLENECK, False, True, False)["next"] == E.NEXT["bottleneck_both"]
    assert P(E.BOTTLENECK, False, False, True)["next"] == E.NEXT["bottleneck_neither"]
    assert P(E.BOTTLENECK, True, False, True)["next"] == E.NEXT["bottleneck_failed_success"]
    for r in (E.SPLIT, E.PENDING, None):
        assert P(r, True, True, True)["holds"] is None


def test_the_route_names_what_blocks_adoption():
    held = E.predict(E.BOTTLENECK, True, True, False)
    ok = {"evaluated": True, "caution": False, "e1": 26.0, "v1": 25.3, "rise": 0.7}
    text, parts = E.route_of(held, E.MET, [], ok, {"GAIN": "NO GAIN"})
    assert parts["candidate"] and text == E.NEXT["bottleneck_holds"] + ": " + E.CANDIDATE
    assert parts["gain_rule_1"] is None and parts["blocks"] == []
    text, parts = E.route_of(held, E.NOT_MET, ["R0: no_alarm", "non-inferiority G1a"], ok, {})
    assert not parts["candidate"] and "R0: no_alarm" in text and "non-inferiority G1a" in text
    hot = dict(ok, caution=True, e1=31.0, rise=5.7)
    text, parts = E.route_of(held, E.MET, [], hot, {})
    assert not parts["candidate"] and "G10 caution applies" in text and "5.70" in text
    text, parts = E.route_of(held, E.MET, [], {"evaluated": False, "caution": None}, {})
    assert not parts["candidate"] and "not evaluated" in text
    neither = E.predict(E.BOTTLENECK, False, False, True)
    text, parts = E.route_of(neither, E.NOT_MET, ["success line S12"], ok, {"GAIN": "GAIN"})
    assert text == E.GAIN_RULE_1 + ". Under it: " + E.NEXT["bottleneck_neither"]
    assert not parts["adoption_possible"] and not parts["candidate"]


def test_one_end_relation_and_the_per_stride_band():
    ids = [f"p{i:02d}" for i in range(12)]
    rng = np.random.default_rng(5)
    y = line({p: float(v) for p, v in zip(ids, rng.normal(10, 2, 12))})
    x = line({p: v - 3.0 for p, v in y["per_person"].items()})
    r = E.one_end(x, y, 2.0)
    assert r["better"] and not r["worse"] and not r["within"] and abs(r["G"] - 3.0) < 1e-12
    assert E.one_end(y, x, 2.0)["worse"] and not E.one_end(x, y, 3.5)["better"]
    assert E.one_end(x, y, 3.5)["within"] and E.one_end(None, y, 1.0) == {"evaluated": False}
    assert E.e1_carry(r) == E.CARRY and E.e1_carry(E.one_end(x, y, 3.5)) == E.BOTTLENECK
    # heterogeneous gains: two persons 15 cm better, ten 0.3 cm worse; G 2.25 >= m 2.0, the lower bound below 0
    xh = line({p: v - (15.0 if i < 2 else -0.3) for i, (p, v) in enumerate(y["per_person"].items())})
    rh = E.one_end(xh, y, 2.0)
    assert rh["G"] >= 2.0 and rh["paired"]["interval"][0] <= 0 and not rh["better"] and not rh["within"]
    ps = E.per_stride_probe({"P1": 7.0, "dP": 0.6}, 5.7)
    assert abs(ps["band"] - 1.2) < 1e-12 and ps["improved"] and not ps["within"]
    assert E.per_stride_probe({"P1": 7.0, "dP": 0.6}, 7.5)["within"]


def test_the_matched_seed_differences_stay_beside(tmp_path):
    reads = {"S12": {"value": 17.5, "at_most": 17.56, "paired_v1_minus_e1": {"lower_above_0": True}},
             "G1a": {"value": 85.0, "at_most": 85.88}}
    doc = {"complete": True, "uses": [
        {"document": "JEPA", "item": "E1 success S(12)", "line": "S12_B", "kind": "success",
         "threshold_mismatched": 17.56, "threshold_matched": 17.40, "threshold_same_code": None, "d_matched": 0.1},
        {"document": "JEPA", "item": "E1 not worse G1a", "line": "G1a_B", "kind": "noninf",
         "threshold_mismatched": 85.88, "threshold_matched": 86.0, "threshold_same_code": 84.9},
        {"document": "F2L2", "item": "R1", "line": "S12_B"}]}
    p = str(tmp_path / "sd.json")
    dump(p, doc)
    b = E.seed_difference_beside(p, reads, {})
    assert b["present"] and len(b["rows"]) == 2 and "never substituted" in b["label"]
    s, g = b["rows"]
    assert s["e1_holds_mismatched"] is True and s["e1_holds_matched"] is False and s["e1_holds_same_code"] is None
    assert s["mismatched_equals_here"] is True and g["e1_holds_same_code"] is False
    assert b["changes"] == ["E1 success S(12)"]
    assert E.seed_difference_beside(str(tmp_path / "absent.json"), reads, {})["present"] is False


def test_the_script_text_follows_the_style():
    with open(os.path.join(TREE, "scripts", "e1_read.py"), encoding="utf-8", newline="") as f:
        t = f.read()
    assert "\r" not in t and chr(0x2014) not in t and chr(0x2013) not in t
    assert E.ADDENDUM_SHA256.startswith("6083e31e")
