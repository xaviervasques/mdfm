"""G1D-SYN: synthesis of the G1 diagnosis (Hstand, Hprior, Hlayout) and the attainability numbers of the G1 levers.

Stages (each writes under --out, default reports/bench/g1_diag/SYN):
- attain: counts only, no error value. From the per_take rows of the decomposition
  (reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.json): windows per take and per person,
  and the persons eligible for the pooling check (at least one take with at most FEW windows and one with at
  least MANY). Writes attain.json.
- table: the lever ceilings already measured by other scripts, read from their outputs (decompose.json,
  reports/bench/g1_diag/Hlayout/results.json, reports/bench/g1_diag/Hstand/analysis.json), each set against the
  G1a point of the same file: best case = G1a minus the ceiling, and its distance to the kill line (70 mm) and the
  pass line (45 mm). No new error value: arithmetic on values those files hold. Writes levers.json, levers.md.
- pooling: the pre-registered check of reports/weekend_2026-10-01/tasks/G1D-SYN.md, section Prediction. Per take,
  the shape gain s = g1a_mm - true_betas_mm (the take's knot-weighted G1a minus the same with the true betas).
  Shape is pooled per take at inference, so a take with more windows pools more. Within person, s of the takes
  with at most FEW windows minus s of the takes with at least MANY (each a knot-weighted mean), averaged over the
  eligible persons; and the within-person slope of s on log2(windows) (person-demeaned, knot-weighted least
  squares over every person whose takes differ in window count). Person-cluster bootstrap, B resamples, seed
  SEED. Writes pooling.json, pooling.md.
- margins: the margins and attainability numbers of docs/proposals/G1_LEVER1_RULE_DRAFT.md section 7, from the
  G1 dev runs of v1 and seed 2 at the B end (g1.json lines), decompose.json and pooling.json. Writes margins.json.

Data: decompose.json and the G1D output files only (dev windows, already scored); no model, no GPU, no lockbox,
EVAL, CODY or care_pd record. Run (CPU):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 <conda>/envs/mdfm/python.exe \
      scripts/g1d_SYN.py --stage attain
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(TREE, "reports", "bench", "g1_diag", "SYN")
DECOMPOSE = "reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.json"
HLAYOUT = "reports/bench/g1_diag/Hlayout/results.json"
HSTAND = "reports/bench/g1_diag/Hstand/analysis.json"
KILL_MM, PASS_MM = 70.0, 45.0
FEW, MANY = 2, 5          # take classes of the pooling check (windows per take)
MARGIN_MM = 2.10          # 2 x the G1a seed difference at the B end (reports/gates/inference_mode/criterion.json)
B, SEED = 2000, 20261006
DASHES = (chr(0x2013), chr(0x2014))


def _load(rel):
    with open(os.path.join(TREE, rel), encoding="utf-8") as f:
        return json.load(f)


def _sha(rel):
    with open(os.path.join(TREE, rel), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write(path, text):
    for d in DASHES:
        if d in text:
            raise ValueError(f"dash character in {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, obj):
    _write(path, json.dumps(obj, indent=1, sort_keys=False) + "\n")


def _moveatlas_head():
    root = os.environ.get("MOVEATLAS_ROOT", "<moveatlas>")
    try:
        r = subprocess.run(["git", "-c", "safe.directory=<moveatlas>", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def _provenance(stage, inputs):
    return {"script": "scripts/g1d_SYN.py", "stage": stage, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script_sha256": _sha("scripts/g1d_SYN.py"), "inputs": {p: _sha(p) for p in inputs},
            "moveatlas_head": _moveatlas_head()}


# ---------------------------------------------------------------- pure functions (tested)
def eligible_persons(takes, few=FEW, many=MANY):
    """Persons with at least one take of at most `few` windows and one of at least `many`. takes: dicts with
    person_id, windows. Returns a sorted list."""
    has_few, has_many = set(), set()
    for t in takes:
        if t["windows"] <= few:
            has_few.add(t["person_id"])
        if t["windows"] >= many:
            has_many.add(t["person_id"])
    return sorted(has_few & has_many)


def few_minus_many(takes, persons, few=FEW, many=MANY, key="s"):
    """Mean over `persons` of (knot-weighted mean of key over the person's takes with at most `few` windows) minus
    (the same over the takes with at least `many`). Persons lacking a class are skipped; NaN if none is left."""
    diffs = []
    for p in persons:
        a = [(t[key], t["knots"]) for t in takes if t["person_id"] == p and t["windows"] <= few]
        b = [(t[key], t["knots"]) for t in takes if t["person_id"] == p and t["windows"] >= many]
        if not a or not b:
            continue
        ma = sum(v * w for v, w in a) / sum(w for _, w in a)
        mb = sum(v * w for v, w in b) / sum(w for _, w in b)
        diffs.append(ma - mb)
    return float(np.mean(diffs)) if diffs else float("nan")


def within_slope(takes, persons, key="s"):
    """Knot-weighted least-squares slope of key on log2(windows) after removing each person's weighted mean of
    both, over the takes of `persons`. NaN when log2(windows) has no within-person variance."""
    xs, ys, ws = [], [], []
    for p in persons:
        rows = [t for t in takes if t["person_id"] == p]
        w = np.array([t["knots"] for t in rows], dtype=float)
        x = np.log2(np.array([t["windows"] for t in rows], dtype=float))
        y = np.array([t[key] for t in rows], dtype=float)
        xs.append(x - np.sum(w * x) / np.sum(w))
        ys.append(y - np.sum(w * y) / np.sum(w))
        ws.append(w)
    if not xs:
        return float("nan")
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    sxx = float(np.sum(w * x * x))
    return float(np.sum(w * x * y) / sxx) if sxx > 0 else float("nan")


def person_bootstrap(takes, persons, stat, b=B, seed=SEED):
    """Percentile 95 percent interval of stat(takes_resampled, persons_resampled) over b person resamples (a
    resampled person's takes get a new person key, so a person drawn twice counts twice)."""
    rng = np.random.default_rng(seed)
    by = {p: [t for t in takes if t["person_id"] == p] for p in persons}
    vals = []
    for _ in range(int(b)):
        pick = rng.integers(0, len(persons), len(persons))
        tk, ps = [], []
        for j, i in enumerate(pick):
            key = f"{persons[i]}#{j}"
            ps.append(key)
            tk += [dict(t, person_id=key) for t in by[persons[i]]]
        v = stat(tk, ps)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reading(point, ci, slope_ci, margin=MARGIN_MM):
    """The pre-registered reading: HELPS when point >= margin, its lower bound > 0 and the slope interval lies
    below 0; NO_POOLING_GAIN when point <= 0 or its upper bound < margin; UNCLEAR otherwise."""
    if point >= margin and ci[0] > 0 and slope_ci[1] < 0:
        return "HELPS"
    if point <= 0 or ci[1] < margin:
        return "NO_POOLING_GAIN"
    return "UNCLEAR"


def best_case(g1a, ceiling):
    """Best-case G1a after a lever of the given ceiling, with its distances to the kill and pass lines."""
    b = g1a - ceiling
    return {"best_case_mm": b, "above_kill_mm": b - KILL_MM, "above_pass_mm": b - PASS_MM}


# ---------------------------------------------------------------- stages
def stage_attain(out):
    d = _load(DECOMPOSE)
    takes = d["per_take"]
    el = eligible_persons(takes)
    varied = sorted({t["person_id"] for t in takes}
                    - {p for p in {t["person_id"] for t in takes}
                       if len({t["windows"] for t in takes if t["person_id"] == p}) < 2})
    res = {"takes": len(takes), "persons": len({t["person_id"] for t in takes}),
           "windows_per_take_counts": {str(k): sum(1 for t in takes if t["windows"] == k)
                                       for k in sorted({t["windows"] for t in takes})},
           "few": FEW, "many": MANY, "eligible_persons": el,
           "eligible_takes_few": sum(1 for t in takes if t["person_id"] in el and t["windows"] <= FEW),
           "eligible_takes_many": sum(1 for t in takes if t["person_id"] in el and t["windows"] >= MANY),
           "persons_with_varied_window_counts": len(varied),
           "takes_of_varied_persons": sum(1 for t in takes if t["person_id"] in varied),
           "windows_per_person": {p: sum(t["windows"] for t in takes if t["person_id"] == p)
                                  for p in sorted({t["person_id"] for t in takes})},
           "provenance": _provenance("attain", [DECOMPOSE])}
    wp = list(res["windows_per_person"].values())
    res["windows_per_person_median"] = float(np.median(wp))
    res["windows_per_take_max"] = max(t["windows"] for t in takes)
    _write_json(os.path.join(out, "attain.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("windows_per_person", "provenance")}, indent=1))


def stage_table(out):
    d = _load(DECOMPOSE)
    lay = _load(HLAYOUT)
    st = _load(HSTAND)
    g1a = d["rows"]["g1a"]["point_mm"]
    rows = []

    def add(lever, kind, what, ceiling, ci, source, note=""):
        r = {"lever": lever, "kind": kind, "ceiling_what": what, "ceiling_mm": ceiling, "ceiling_ci_mm": ci,
             "source": source, "note": note}
        r.update(best_case(g1a, ceiling))
        rows.append(r)

    R = d["rows"]
    G = d["by_group"]
    add("shape", "inference or retrain", "true betas (decoder and skeleton)", R["true_betas"]["delta_mm"],
        R["true_betas"]["delta_interval_mm"], DECOMPOSE + " rows.true_betas")
    add("arms, global orientation", "retrain or input", "true global orientations of the arm segments",
        G["arms"]["gain_mm"], G["arms"]["gain_interval_mm"], DECOMPOSE + " by_group.arms")
    add("arms, local rotations (elbow bias inside)", "inference or retrain", "true local rotations of the arm joints",
        G["arms"]["gain_local_mm"], G["arms"]["gain_local_interval_mm"], DECOMPOSE + " by_group.arms")
    add("legs, global orientation", "retrain or input", "true global orientations of the leg segments",
        G["legs"]["gain_mm"], G["legs"]["gain_interval_mm"], DECOMPOSE + " by_group.legs")
    add("spine and head, global orientation", "retrain or input", "true global orientations of spine and head",
        G["spine_head"]["gain_mm"], G["spine_head"]["gain_interval_mm"], DECOMPOSE + " by_group.spine_head")
    add("root orientation", "retrain or input", "true global orientation of the root",
        R["true_root_orient"]["delta_mm"], R["true_root_orient"]["delta_interval_mm"],
        DECOMPOSE + " rows.true_root_orient")
    add("global part (all segments' global orientation and scale)", "retrain or input",
        "G1a minus PA-MPJPE (22 joints)", R["pa"]["delta_mm"], R["pa"]["delta_interval_mm"], DECOMPOSE + " rows.pa")
    add("within-window yaw drift", "retrain or input", "G1a minus per-knot heading-frame MPJPE",
        R["heading"]["delta_mm"], R["heading"]["delta_interval_mm"], DECOMPOSE + " rows.heading")
    add("refinement steps 20 to 0", "inference", "G1a minus the 0-step G1a", R["refine0"]["delta_mm"],
        R["refine0"]["delta_interval_mm"], DECOMPOSE + " rows.refine0", "negative: 0 steps is worse")
    add("best of 16 draws (oracle, not a lever)", "oracle", "G1a minus best of 16", R["best_of_16"]["delta_mm"],
        R["best_of_16"]["delta_interval_mm"], DECOMPOSE + " rows.best_of_16")
    cf = st["runs"]["v1_B_end"]["counterfactual"]["remove_q5"]
    add("fastest fifth removed (a reweighting, not a lever)", "analysis", "G1 statistic minus the same without Q5",
        -cf["change_mm"], [-cf["interval_mm"][1], -cf["interval_mm"][0]], HSTAND + " runs.v1_B_end.counterfactual",
        "two persons drop out")
    ex = lay["archive"]["v1"].get("extrapolation_to_d1")
    if isinstance(ex, dict):
        exv = ex.get("drop_mm")
    else:
        exv = ex
    if isinstance(exv, (int, float)):
        add("training motion coverage (extrapolation, not a criterion)", "retrain (data)",
            "fitted d coefficient times (dev mean d minus 1)", float(exv), None, HLAYOUT + " archive.v1",
            "outside the dev range of d")

    a = lay["arms"]
    own = {"set": "Hlayout CPU set", "windows": a["windows"], "persons": a["persons"],
           "six_protocol_mm": a["six_protocol_g1a_mm"], "six_factory_mm": a["six_factory_g1a_mm"],
           "xs13_factory_mm": a["xs13_factory_g1a_mm"], "layout_diff_mm": a["layout_g1a"]["diff_mm"],
           "layout_ci_mm": a["layout_g1a"]["ci_mm"], "gate_same_windows_mm": a["harness_gate_same_windows_mm"],
           "xs13_above_kill_mm": a["xs13_factory_g1a_mm"] - KILL_MM,
           "xs13_above_pass_mm": a["xs13_factory_g1a_mm"] - PASS_MM,
           "six_protocol_above_kill_mm": a["six_protocol_g1a_mm"] - KILL_MM}
    align = {"what": "literature-style root alignment approximated by the true root rotation with the decoded "
                     "local rotations kept (row true_root_orient_local)",
             "point_mm": R["true_root_orient_local"]["point_mm"],
             "interval_mm": R["true_root_orient_local"]["interval_mm"],
             "delta_mm": R["true_root_orient_local"]["delta_mm"],
             "delta_interval_mm": R["true_root_orient_local"]["delta_interval_mm"],
             "heading_point_mm": R["heading"]["point_mm"], "pa_point_mm": R["pa"]["point_mm"],
             "floor_point_mm": R["floor"]["point_mm"]}
    by_arch = {k: {"g1a_mm": v["g1a_mm"], "true_betas_mm": v["true_betas_mm"],
                   "shape_gain_mm": v["g1a_mm"] - v["true_betas_mm"]} for k, v in d["by_archive"].items()}
    res = {"g1a_mm": g1a, "g1a_interval_mm": R["g1a"]["interval_mm"], "kill_mm": KILL_MM, "pass_mm": PASS_MM,
           "gap_to_kill_mm": g1a - KILL_MM, "gap_to_pass_mm": g1a - PASS_MM, "levers": rows,
           "layout_cpu_set": own, "alignment": align, "by_archive_knot_weighted": by_arch,
           "provenance": _provenance("table", [DECOMPOSE, HLAYOUT, HSTAND])}
    _write_json(os.path.join(out, "levers.json"), res)
    L = [f"# G1 lever ceilings against the lines (scripts/g1d_SYN.py --stage table)", "",
         f"G1a {g1a:.2f} mm ({DECOMPOSE}, row g1a); above the kill line by {g1a - KILL_MM:.2f} mm, above the pass "
         f"line by {g1a - PASS_MM:.2f} mm. Best case = G1a minus the ceiling (a measured upper bound of the lever "
         f"on the same windows, not its expected effect); ceilings do not add.", "",
         "| lever | kind | ceiling | ceiling mm | interval | best case mm | above 70 | above 45 | source |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "n/a" if r["ceiling_ci_mm"] is None else f"{r['ceiling_ci_mm'][0]:.2f} to {r['ceiling_ci_mm'][1]:.2f}"
        L.append(f"| {r['lever']} | {r['kind']} | {r['ceiling_what']} | {r['ceiling_mm']:.2f} | {ci} | "
                 f"{r['best_case_mm']:.2f} | {r['above_kill_mm']:.2f} | {r['above_pass_mm']:.2f} | {r['source']} |")
    L += ["", f"Layout (Hlayout CPU set, {own['windows']} windows, {own['persons']} persons; its own level, not the "
          f"gate's): six_protocol {own['six_protocol_mm']:.2f} mm, xs13_factory {own['xs13_factory_mm']:.2f} mm "
          f"(above 70 by {own['xs13_above_kill_mm']:.2f}, above 45 by {own['xs13_above_pass_mm']:.2f}); layout "
          f"difference {own['layout_diff_mm']:.2f} mm ({own['layout_ci_mm'][0]:.2f} to {own['layout_ci_mm'][1]:.2f}).",
          "", f"Alignment: true root rotation with decoded local rotations {align['point_mm']:.2f} mm "
          f"({align['interval_mm'][0]:.2f} to {align['interval_mm'][1]:.2f}), change {align['delta_mm']:.2f} mm; "
          f"per-knot heading frame {align['heading_point_mm']:.2f} mm; PA-MPJPE {align['pa_point_mm']:.2f} mm; "
          f"floor {align['floor_point_mm']:.2f} mm.", "", "By archive (knot-weighted): " + "; ".join(
              f"{k} G1a {v['g1a_mm']:.2f}, true betas {v['true_betas_mm']:.2f}, shape gain {v['shape_gain_mm']:.2f}"
              for k, v in by_arch.items()) + " mm.", ""]
    _write(os.path.join(out, "levers.md"), "\n".join(L))
    print("\n".join(L))


def stage_pooling(out):
    d = _load(DECOMPOSE)
    takes = [{"person_id": t["person_id"], "archive": t["archive"], "windows": int(t["windows"]),
              "knots": int(t["knots"]), "g1a": float(t["g1a_mm"]), "tb": float(t["true_betas_mm"]),
              "s": float(t["g1a_mm"]) - float(t["true_betas_mm"])} for t in d["per_take"]]
    el = eligible_persons(takes)
    point = few_minus_many(takes, el)
    ci = person_bootstrap(takes, el, lambda tk, ps: few_minus_many(tk, ps))
    varied = sorted(p for p in {t["person_id"] for t in takes}
                    if len({t["windows"] for t in takes if t["person_id"] == p}) >= 2)
    slope = within_slope(takes, varied)
    slope_ci = person_bootstrap(takes, varied, lambda tk, ps: within_slope(tk, ps))
    per = {}
    for p in el:
        a = [t for t in takes if t["person_id"] == p and t["windows"] <= FEW]
        b = [t for t in takes if t["person_id"] == p and t["windows"] >= MANY]
        wm = lambda rows, k: sum(t[k] * t["knots"] for t in rows) / sum(t["knots"] for t in rows)  # noqa: E731
        per[p] = {"takes_few": len(a), "takes_many": len(b), "s_few_mm": wm(a, "s"), "s_many_mm": wm(b, "s"),
                  "g1a_few_mm": wm(a, "g1a"), "g1a_many_mm": wm(b, "g1a"), "tb_few_mm": wm(a, "tb"),
                  "tb_many_mm": wm(b, "tb")}
    g1a_diff = few_minus_many(takes, el, key="g1a")
    tb_diff = few_minus_many(takes, el, key="tb")
    read = reading(point, ci, slope_ci)
    res = {"few": FEW, "many": MANY, "margin_mm": MARGIN_MM, "eligible_persons": el,
           "few_minus_many_shape_gain_mm": point, "interval_mm": ci, "varied_persons": len(varied),
           "within_slope_mm_per_doubling": slope, "slope_interval": slope_ci, "reading": read,
           "reported_g1a_few_minus_many_mm": g1a_diff, "reported_true_betas_few_minus_many_mm": tb_diff,
           "per_person": per, "bootstrap": {"resamples": B, "seed": SEED},
           "provenance": _provenance("pooling", [DECOMPOSE])}
    _write_json(os.path.join(out, "pooling.json"), res)
    L = ["# Pooling check (scripts/g1d_SYN.py --stage pooling)", "",
         f"Shape gain s = take G1a minus take G1a with true betas ({DECOMPOSE}, per_take). Takes with at most {FEW} "
         f"windows against takes with at least {MANY}, within person, {len(el)} eligible persons.", "",
         f"- few minus many, mean over persons: {point:.2f} mm, person bootstrap {ci[0]:.2f} to {ci[1]:.2f} "
         f"({B} resamples, seed {SEED}).",
         f"- within-person slope of s on log2(windows), {len(varied)} persons: {slope:.2f} mm per doubling, "
         f"{slope_ci[0]:.2f} to {slope_ci[1]:.2f}.",
         f"- reported, not a criterion: G1a few minus many {g1a_diff:.2f} mm; G1a with true betas few minus many "
         f"{tb_diff:.2f} mm.", f"- reading: {read}.", "", "| person | takes few | takes many | s few | s many | "
         "G1a few | G1a many |", "|---|---|---|---|---|---|---|"]
    for p, v in per.items():
        L.append(f"| {p} | {v['takes_few']} | {v['takes_many']} | {v['s_few_mm']:.2f} | {v['s_many_mm']:.2f} | "
                 f"{v['g1a_few_mm']:.2f} | {v['g1a_many_mm']:.2f} |")
    _write(os.path.join(out, "pooling.md"), "\n".join(L) + "\n")
    print("\n".join(L))


G1_V1 = "reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/g1.json"
G1_S2 = "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_032044/g1.json"


def ni_margin(v1, s2, frac=0.05):
    """Non-inferiority margin of the v2check form: max(frac x |v1's line|, 2 x |v1 - s2|)."""
    return max(frac * abs(v1), 2.0 * abs(v1 - s2))


def stage_margins(out):
    """Margins and attainability numbers of docs/proposals/G1_LEVER1_RULE_DRAFT.md, section 7, from the two G1 dev
    runs, the decomposition and pooling.json (values those files hold; no new error value)."""
    v1 = {r["line"]: r for r in _load(G1_V1)["lines"]}
    s2 = {r["line"]: r for r in _load(G1_S2)["lines"]}
    a, b, c = "G1a six IMUs MPJPE", "G1b single wrist coverage", "G1c single camera PA-MPJPE"
    d = _load(DECOMPOSE)
    tb = d["rows"]["true_betas"]
    pool = None
    if os.path.exists(os.path.join(out, "pooling.json")):
        with open(os.path.join(out, "pooling.json"), encoding="utf-8") as f:
            pool = json.load(f)
    seed_d = abs(v1[a]["point"] - s2[a]["point"])
    m_band = 2.0 * seed_d
    m_f2 = max(0.10 * v1[a]["point"], m_band)
    res = {"g1a_v1_mm": v1[a]["point"], "g1a_s2_mm": s2[a]["point"], "g1a_seed_difference_mm": seed_d,
           "success_margin_mm": m_band, "success_margin_f2_form_mm": m_f2,
           "ceiling_true_betas_mm": tb["delta_mm"], "ceiling_interval_mm": tb["delta_interval_mm"],
           "ceiling_minus_margin_mm": tb["delta_mm"] - m_band,
           "margin_over_ceiling": m_band / tb["delta_mm"],
           "f2_form_attainable": m_f2 <= tb["delta_mm"],
           "band_attainable_point": m_band <= tb["delta_mm"],
           "band_below_ceiling_lower_bound": m_band < tb["delta_interval_mm"][0],
           "g1a_at_ceiling_mm": v1[a]["point"] - tb["delta_mm"],
           "g1a_at_margin_mm": v1[a]["point"] - m_band,
           "g1a_at_ceiling_above_kill_mm": v1[a]["point"] - tb["delta_mm"] - KILL_MM,
           "g1a_at_ceiling_above_pass_mm": v1[a]["point"] - tb["delta_mm"] - PASS_MM,
           "g1b_coverage_v1": v1[b]["point"], "g1b_coverage_s2": s2[b]["point"],
           "g1b_coverage_margin": ni_margin(v1[b]["point"], s2[b]["point"]),
           "g1b_coverage_v1_minus_margin": v1[b]["point"] - ni_margin(v1[b]["point"], s2[b]["point"]),
           "g1b_pass_line": v1[b]["pass_line"],
           "g1c_v1_mm": v1[c]["point"], "g1c_s2_mm": s2[c]["point"],
           "g1c_margin_mm": ni_margin(v1[c]["point"], s2[c]["point"]),
           "g1c_pass_line_mm": v1[c]["pass_line"],
           "harness_check_tolerance_mm": 2.0,
           "pooling_check": None if pool is None else {k: pool[k] for k in (
               "few_minus_many_shape_gain_mm", "interval_mm", "within_slope_mm_per_doubling", "slope_interval",
               "reading")},
           "provenance": _provenance("margins", [G1_V1, G1_S2, DECOMPOSE])}
    _write_json(os.path.join(out, "margins.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=("attain", "table", "pooling", "margins"))
    p.add_argument("--out", default=OUT)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    {"attain": stage_attain, "table": stage_table, "pooling": stage_pooling,
     "margins": stage_margins}[a.stage](a.out)


if __name__ == "__main__":
    main()
