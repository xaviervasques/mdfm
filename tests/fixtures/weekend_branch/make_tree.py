"""Fixture tree of tests/test_weekend_branch.py: the report folders, run logs and job log that
scripts/weekend_branch.py reads, with synthetic values (no number of a real run).

build(root, **over) writes the base scenario under root and returns a dict of the values used. In the base
scenario the branch stage reads: R0 holds; R1 clear of the kill lines (G1a 62 mm, S 8 cm, V 0.12 m/s on the 12);
the IC margin not met; R-insim above 5 cm; R-AB 'phase B improved'; M1 does not fire (hurried speed bias -0.05
m/s); M2 fires (2 cm of a 3 cm gap); M4 fires (6 mm of a 17 mm gap); M6 does not fire (no consistent effect);
v2 is M2; R2a proposes refine60 (G1a) and lowpass12 (S). The other stages: seed differences at ckpt_B_004000
and at the B end, a v2 check that goes, a reference that stays v1's B end, an extension that extends.

Keyword overrides (all optional) replace pieces of the scenario; the tests use them to flip each row.
"""
import csv
import json
import os

IDS = ["weargait:HC103", "weargait:HC113", "weargait:HC128", "weargait:HC901", "weargait:HC130", "weargait:HC133",
       "weargait:HC136", "weargait:NLS901 (control)", "weargait:WHC002", "weargait:WHC003", "weargait:WHC014",
       "weargait:WHC901", "weargait:WHC026", "weargait:WHC033"]
OUT = ("weargait:WHC003", "weargait:WHC026")
TWELVE = [i for i in IDS if i not in OUT]
# S on the 12 at the B end: mean 8.0 cm; the two excluded persons far above
S_B12 = [7.0, 8.5, 9.0, 6.5, 8.0, 7.5, 9.5, 8.2, 7.8, 8.8, 7.2, 8.0]
V_B12 = [0.10, 0.13, 0.12, 0.11, 0.14, 0.12, 0.13, 0.10, 0.12, 0.13, 0.11, 0.13]
JIT = [0.10, -0.05, 0.08, -0.12, 0.03, 0.06, -0.09, 0.02, -0.04, 0.11, -0.07, 0.05]   # mean 0.0067
ALT = [0.5, -0.5, 0.6, -0.6, 0.4, -0.4, 0.55, -0.55, 0.45, -0.45, 0.5, -0.5]           # mean 0


def person_map(v12, whc003, whc026):
    m = dict(zip(TWELVE, v12))
    m["weargait:WHC003"], m["weargait:WHC026"] = whc003, whc026
    return {i: m[i] for i in IDS}


def shift(m, d, jit=None):
    out = {}
    for k, (i, v) in enumerate(m.items()):
        out[i] = None if v is None else v + d + (jit[k % len(jit)] if jit else 0.0)
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def subset_of(m, name):
    drop = {"all14": (), "no_whc003": ("weargait:WHC003",), "no_whc003_whc026": OUT}[name]
    return {k: v for k, v in m.items() if k not in drop}


def diag_row(S, V, selfpace=None, extra=None):
    row = {}
    for name in ("all14", "no_whc003", "no_whc003_whc026"):
        s, v = subset_of(S, name), subset_of(V, name)
        pp = {k: {"stride_cm": s[k], "speed_ms": v[k]} for k in s}
        row[name] = {"stride_mae_cm": mean(s.values()), "speed_mae_ms": mean(v.values()),
                     "stride_mape_pct": None if mean(s.values()) is None else mean(s.values()) * 7.0,
                     "n_persons": len(s), "per_person": pp}
    if selfpace is not None:
        row["selfpace_stride_mae_cm"] = {
            name: {"point": mean(subset_of(selfpace, name).values()), "per_person": subset_of(selfpace, name)}
            for name in ("all14", "no_whc003", "no_whc003_whc026")}
    row.update(extra or {})
    return row


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(path, doc):
    write(path, json.dumps(doc, indent=1) + "\n")


def g3_folder(root, rel, S, V, name="g3_dev.json", ic=22.0, margin=(19.17, 13.33), detector=None, weights="ema"):
    folder = os.path.join(root, rel)
    doc = {"run": "fixture", "checkpoint": {"weights": weights},
           "lines": {"dev": {"model": {"stride_mae_cm": {"point": mean(S.values())},
                                       "speed_mae_m_s": {"point": mean(V.values())},
                                       "ic_median_ms_walkway": {"point": ic}},
                             "shank_gyro": {"ic_median_ms_walkway": {"point": detector}}}},
           "results_dev": [{"line": "stride_mae_cm", "point": mean(S.values())},
                           {"line": "ic_median_ms_walkway", "point": ic,
                            "extra": {"margin_model_ms": margin[0], "margin_detector_ms": margin[1],
                                      "margin": 0.8}}]}
    write_json(os.path.join(folder, name), doc)
    rows = [["row", "predictor", "line", "person_id", "value"]]
    for line, m in (("stride_mae_cm", S), ("speed_mae_m_s", V)):
        for k, v in m.items():
            rows.append(["dev", "model", line, k, "" if v is None else repr(v)])
            rows.append(["dev", "walkway", line, k, "0.0"])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "per_person.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def takes(n, base, step=0.37, jit=None):
    """{take id: mm} of n takes around base."""
    out = {}
    for t in range(n):
        out[f"Archive/p{t % 5}/take_{t:03d}.npz"] = base + ((t * step) % 3.0 - 1.5) + (jit[t % len(jit)] if jit else 0.0)
    return out


def knots(t):
    return 1000 if int(t[-7:-4]) % 3 else 400


def g1_folder(root, rel, point, tk, limit_takes=None, weights="ema"):
    folder = os.path.join(root, rel)
    write_json(os.path.join(folder, "g1.json"), {
        "gate": "G1", "partial_run": {"limit_takes": limit_takes} if limit_takes else None,
        "inference": {"weights": weights},
        "lines": [{"line": "G1a six IMUs MPJPE", "point": point, "interval": [point - 4.0, point + 4.0],
                   "pass_line": 45.0, "kill_line": 70.0, "n_persons": 23}]})
    rows = [["window", "take_id", "archive", "person_id", "task", "n", "g1a_m"]]
    for w, (t, v) in enumerate(tk.items()):
        rows.append([w, t, "Archive", "amass:person:Archive:" + t.split("/")[1], "walk", knots(t), v / 1000.0])
    with open(os.path.join(folder, "windows.csv"), "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def run_log(root, run, phase_b_end=True, verify_b_step=32044, alarm=False, mismatches=(), nonfinite=(),
            max_reserved=11.75, b_steps=32044, ckpts=range(2000, 32001, 2000), s_per_step=2.4):
    import datetime
    t0 = datetime.datetime(2026, 9, 30, 22, 17, 29)
    rel = f"cache/runs/{run}"
    ev = [{"event": "start", "phase": "A", "steps": [10681, b_steps], "report_mismatches": [], "seed": 1,
           "time": "2026-09-30T16:45:00"},
          {"event": "verify", "phase": "A", "step": 10681, "path": rel + "\\ckpt_A_010681.pt", "sha256": "a" * 64,
           "report_mismatches": [], "drift": []},
          {"event": "start", "phase": "B", "steps": [10681, b_steps], "report_mismatches": list(mismatches),
           "time": t0.isoformat()}]
    last = verify_b_step if phase_b_end else min(verify_b_step, 6000)
    for s in range(50, last + 1, 50):
        ev.append({"event": "log", "phase": "B", "step": s,
                   "time": (t0 + datetime.timedelta(seconds=s * s_per_step)).isoformat(timespec="seconds"),
                   "memory_gb": {"max_reserved": max_reserved, "reserved": max_reserved - 0.5}})
        if s in set(ckpts):
            ev.append({"event": "checkpoint", "phase": "B", "step": s, "path": f"{rel}\\ckpt_B_{s:06d}.pt"})
        if s % 1000 == 0:
            ev.append({"event": "dev", "phase": "B", "step": s, "trigger": "interval"})
    for s in nonfinite:
        ev.append({"event": "nonfinite", "phase": "B", "step": s, "grad_norm": float("nan")})
    if alarm:
        ev.append({"event": "alarm", "phase": "B", "step": 5000, "source": "dev", "alarms": [{"name": "x", "value": 0.1}]})
    if phase_b_end:
        ev.append({"event": "end", "phase": "B", "step": verify_b_step,
                   "memory_gb": {"max_reserved": max_reserved}})
        ev.append({"event": "verify", "phase": "B", "step": verify_b_step,
                   "path": f"{rel}\\ckpt_B_{verify_b_step:06d}.pt", "sha256": "b" * 64,
                   "report_mismatches": list(mismatches), "drift": []})
    write(os.path.join(root, rel, "log.jsonl"), "".join(json.dumps(e) + "\n" for e in ev))


def build(root, **over):
    """The base scenario (module docstring); returns the values used."""
    o = dict(over)
    v = {}
    # ------------------------------------------------ B end, seed 1 (phaseA_root_full_ckpt_B_032044)
    SB = person_map(S_B12, 40.0, 20.0)
    VB = person_map(V_B12, 1.0, 0.4)
    SA = o.get("S_A", shift(SB, 15.0, JIT))                  # the A end, 15 cm worse: phase B improved
    VA = o.get("V_A", shift(VB, 0.15, [x / 10 for x in JIT]))
    sp = o.get("selfpace", shift(SB, -1.2, JIT))             # M1 effect 1.2 cm, 0.4 of the gap
    base_extra = {"hurried_speed_bias_ms": o.get("hurried_bias", -0.05),
                  "spearman_speed_error": o.get("spearman", -0.6),
                  "heel_strike_ic": {"median_ms": 18.0, "offset_ms": 40.0, "margin_ratio": 1.2}}
    rows = {"base": diag_row(SB, VB, selfpace=sp, extra=base_extra),
            "lowpass12": diag_row(o.get("S_lowpass12", shift(SB, -2.0, JIT)), VB),
            "refine60": diag_row(o.get("S_refine60", shift(SB, -0.3, ALT)), VB),
            "model_legs": diag_row(o.get("S_model_legs", shift(SB, 0.0, ALT)), VB),
            "refine0": diag_row(shift(SB, 1.0), VB), "sites6": diag_row(shift(SB, 3.0), VB),
            "swap_whc026": diag_row(SB, VB), "family_lower_back": diag_row(shift(SB, 4.0), VB),
            "family_wrists": diag_row(shift(SB, 6.0), VB), "family_legs": diag_row(shift(SB, 1.0), VB)}
    for k in o.get("drop_rows", ()):
        rows.pop(k, None)
    if o.get("drop_base_keys"):
        for k in o["drop_base_keys"]:
            rows["base"].pop(k, None)
    nB, nA = "phaseA_root_full_ckpt_B_032044", "phaseA_root_full_ckpt_A_010681"
    if o.get("diag_b", True):
        write_json(os.path.join(root, "reports/gates/g3", f"diag_{nB}", "diag.json"), {"rows": rows})
    if o.get("diag_a", True):
        write_json(os.path.join(root, "reports/gates/g3", f"diag_{nA}", "diag.json"),
                   {"rows": {"base": diag_row(SA, VA)}})
    g3_folder(root, f"reports/gates/g3/dev_{nB}", SB, VB, name=o.get("g3_b_name", "g3_dev.json"),
              ic=o.get("ic", 22.0), margin=o.get("margin", (19.17, 13.33)))
    g3_folder(root, f"reports/gates/g3_sim/dev_{nA}", SA, VA, name=o.get("g3_a_name", "g3_sim.json"))
    g1_folder(root, f"reports/gates/g1/dev_{nB}", o.get("g1a", 62.0), takes(20, 62.0))
    write_json(os.path.join(root, f"reports/gates/g3_sim/insim_{nB}", "insim.json"),
               {"lines": {"forward": {"model": {"stride_mae_cm": {"point": o.get("insim", 8.8),
                                                                  "interval": [7.0, 10.8]}}}}})
    # decompose: per take g1a 62, true root orientation 6 mm lower; refine20 64, refine60 52, abs 70 (partial)
    g1a_t = takes(30, 62.0)
    tro_t = {t: x - o.get("m4_effect", 6.0) + (0.4 if i % 2 else -0.4) for i, (t, x) in enumerate(g1a_t.items())}
    r20 = takes(10, 64.0)
    pt = []
    for t in g1a_t:
        rec = {"take": t, "knots": knots(t), "g1a_mm": g1a_t[t], "true_root_orient_mm": tro_t[t]}
        if t in r20:
            rec.update(refine20_mm=r20[t], refine60_mm=r20[t] - o.get("r60_gain", 12.0) + (0.5 if knots(t) == 400 else -0.3),
                       abs_mm=r20[t] + 6.0)
        pt.append(rec)
    dec = {"g1a_mm": o.get("dec_g1a", 62.0), "true_root_orient_mm": 56.0, "true_betas_mm": 60.0, "refine0_mm": 70.0,
           "refine20_mm": 64.0, "refine60_mm": 52.0, "abs_mm": 70.0, "best_of_16_mm": 50.0, "floor_mm": 3.0,
           "by_task": {}, "by_archive": {}, "by_group": {}, "lockbox_mix_mm": 65.0, "per_take": pt}
    for k in o.get("drop_dec_keys", ()):
        dec.pop(k, None)
    if o.get("decompose", True):
        write_json(os.path.join(root, f"reports/bench/g1_decompose_{nB}", "decompose.json"), dec)
    warp = o.get("warp", {"rows": [{"factor": 1.0, "stride_mae_cm": 8.8}, {"factor": 1.25, "stride_mae_cm": 10.0},
                                   {"factor": 1.5, "stride_mae_cm": 13.5}]})
    if warp is not None:
        write_json(os.path.join(root, f"reports/bench/root_motion_{nB}_warp15.json"), warp)
    run_log(root, "phaseA_root_full", phase_b_end=o.get("phase_b_end", True), alarm=o.get("alarm", False),
            mismatches=o.get("mismatches", ()))
    # ------------------------------------------------ other stages
    S4_1 = shift(SB, 4.0)                    # v1 ckpt_B_004000: S 12 cm on the 12
    S4_2 = shift(SB, 4.5, JIT)               # seed 2 ckpt_B_004000: seed difference about 0.5
    S4_v2 = o.get("S_v2", shift(SB, 1.0, JIT))  # v2 ckpt_B_004000: 9 cm, 3 cm better
    g3_folder(root, "reports/gates/g3/curve_phaseA_root_full_ckpt_B_004000", S4_1, shift(VB, 0.03))
    g3_folder(root, "reports/gates/g3/curve_pilot_v1_s2_ckpt_B_004000", S4_2, shift(VB, 0.035))
    g3_folder(root, "reports/gates/g3/dev_pilot_v2_ckpt_B_004000", S4_v2, o.get("V_v2", shift(VB, 0.031)))
    g3_folder(root, "reports/gates/g3/curve_phaseA_root_full_ckpt_B_024000", o.get("S_75", shift(SB, 1.0, JIT)),
              VB)
    g3_folder(root, "reports/gates/g3/dev_pilot_v1_s2_ckpt_B_032044", o.get("S_s2_end", shift(SB, 0.3, JIT)),
              shift(VB, 0.005))
    g1_folder(root, "reports/gates/g1/dev_phaseA_root_full_ckpt_B_004000_lt2", 70.0, takes(10, 70.0), 2)
    g1_folder(root, "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_004000_lt2", 71.0, takes(10, 71.0), 2)
    g1_folder(root, "reports/gates/g1/dev_pilot_v2_ckpt_B_004000_lt2", o.get("g1a_v2", 70.5),
              takes(10, o.get("g1a_v2", 70.5)), 2)
    g1_folder(root, "reports/gates/g1/dev_pilot_v1_s2_ckpt_B_032044", o.get("g1a_s2_end", 63.0),
              takes(20, o.get("g1a_s2_end", 63.0)))
    for step, val in ((8000, 68.0), (16000, 66.0), (24000, 64.0), (32044, o.get("g1a_lt2_end", 63.0))):
        g1_folder(root, f"reports/gates/g1/dev_phaseA_root_full_ckpt_B_{step:06d}_lt2", val, takes(10, val), 2)
    run_log(root, "pilot_v1_s2", phase_b_end=o.get("s2_end", True))
    run_log(root, "pilot_v2", verify_b_step=4000, alarm=o.get("v2_alarm", False),
            nonfinite=o.get("v2_nonfinite", ()), max_reserved=o.get("v2_reserved", 12.5), ckpts=(2000, 4000))
    v.update(nB=nB, nA=nA)
    return v
