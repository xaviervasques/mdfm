"""Admissible-null counts of Q10 (pallidal_units_dystonia) and the Holm feasibility of the confirmatory
primaries, from counts only.

Task T12 of the weekend plan 2026-10-01. Rules read: docs/rules/NEURAL_RULES_DRAFT.md NR-1.7, NR-1.8,
NR-6.10.3, NR-7.2 to NR-7.5 (draft, not hashed).

Q10 (slot 7): gene labels permuted across the TOR1A and SGCE patients with group sizes kept, so the number
of non-identity assignments is k = C(n_TOR1A + n_SGCE, n_SGCE) - 1 and the smallest attainable p-value is
1 / (k + 1) (NR-1.7). Holm over m = 6 primaries (NR-7.2): at step j the threshold is alpha / (m - j + 1), so
a primary whose smallest p is p_min can be rejected at step j only when alpha >= (m - j + 1) * p_min. Step 1
(alpha >= m * p_min) is the NR-7.5 check: rejection whatever the other primaries give. Step j > 1 needs the
j - 1 primaries with smaller p-values rejected first.

The patient counts come from reports/pallidal_units/patients_per_gene.csv (scripts/probe_pallidal_units.py,
identifier columns only). When reports/admissible_nulls/oxford_q1.csv (task T11,
scripts/probe_oxford_structure.py) is present, its rows of scope q1 (and only those) are merged into
summary.md, with the MoveAtlas commit of _provenance_oxford_q1.json; a file without a q1 row is reported as
not merged. Whether the merged nulls are drawn or enumerated is read from its enumeration_feasible_k_le_1e6
column: drawn nulls give a smallest p of 1 / (B + 1) with B the draw count of OPEN-33, and the summary gives
the smallest B that passes the NR-7.5 check at each reference alpha.

Outputs:
  reports/admissible_nulls/pallidal_q10.csv              Q10 primary and the exploratory pallidal contrasts
  reports/admissible_nulls/summary.md                     all slots of NR-7.1, Q10 and Q1 counts, Holm steps
  reports/admissible_nulls/_provenance_pallidal_q10.json  stamp, inputs and output SHA-256
Alpha is OPEN-01: the reference levels ALPHAS are shown for reading only and fill no rule value (NR-0.5).

Run:
  CUDA_VISIBLE_DEVICES=-1 python scripts/build_admissible_nulls.py [--moveatlas <moveatlas>]
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moveatlas_stamp import default_root, stamp  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENES_CSV = os.path.join(TREE, "reports", "pallidal_units", "patients_per_gene.csv")
NULLS = os.path.join(TREE, "reports", "admissible_nulls")
OXFORD_Q1 = os.path.join(NULLS, "oxford_q1.csv")
OXFORD_Q1_PROV = os.path.join(NULLS, "_provenance_oxford_q1.json")
Q1_SCOPE = "q1"

M = 6                                     # NR-7.2: slots 1, 2, 4, 5, 6, 7
ENUM_MAX = 10 ** 6                        # enumeration limit, the convention of probe_oxford_structure.py (T11)
K_MIN = 20                                # NR-1.8
ALPHAS = (Fraction(5, 100), Fraction(1, 100))   # reference levels for reading only; alpha is OPEN-01
TITLE = "gene or its surgical correlates"
TITLE_CLASS = "gene class or its surgical correlates"

# (name, role, group a genes, group b genes, source)
CONTRASTS = (
    ("Q10 TOR1A against SGCE", "primary, slot 7", ("TOR1A",), ("SGCE",), "NR-6.10.2, NR-6.10.3, NR-7.1 slot 7"),
    ("TOR1A against VPS16", "exploratory", ("TOR1A",), ("VPS16",), "NR-6.10.4"),
    ("SGCE against VPS16", "exploratory", ("SGCE",), ("VPS16",), "NR-6.10.4"),
    ("brain iron (PANK2, PLA2G6) against TOR1A, THAP1, GNAL", "exploratory", ("PANK2", "PLA2G6"),
     ("TOR1A", "THAP1", "GNAL"), "design note 2026-09-29 priority 7 (b), explicit gene lists"),
)

SLOTS = (
    (1, "oxford_dystonia_lfp", "Q1", "level labels permuted across whole files within hemisphere"),
    (2, "pd_meg_stn_lfp_ds004998", "F1", "within-subject circular shift per record and held-out subject"),
    (4, "merk_neumann_ecog_lfp", "F4", "within-subject circular shift per record and held-out subject"),
    (5, "seizeit2", "Q8", "p-value OPEN-49"),
    (6, "modys_video", "Q9", "p-value OPEN-49"),
    (7, "pallidal_units_dystonia", "Q10", "gene labels permuted across patients"),
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def admissible_k(n_a, n_b):
    """Non-identity assignments of the labels of two groups with sizes kept."""
    return math.comb(n_a + n_b, n_b) - 1


def holm_alpha_min(p_min, m=M):
    """Smallest alpha at which a p-value p_min is rejected at Holm step j, for j = 1..m (exact fractions)."""
    return [Fraction(m - j + 1) * p_min for j in range(1, m + 1)]


def min_draws(alpha, m=M):
    """Smallest draw count B with 1 / (B + 1) <= alpha / m (NR-7.5 at step 1 for drawn nulls)."""
    return math.ceil(Fraction(m) / alpha) - 1


def fmt(x):
    if isinstance(x, Fraction):
        x = float(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def alpha_label(a):
    return fmt(a).replace("0.", "0p")


def stamp_line(s):
    """The MoveAtlas commit line of a summary, from a stamp of scripts/moveatlas_stamp.py (no time field)."""
    return (f"- MoveAtlas measured against: HEAD {s['head']}, uncommitted diff SHA-256 {s['diff_sha256']} "
            f"({s.get('diff_bytes', '?')} bytes), from scripts/moveatlas_stamp.py; the full stamp is in "
            "_provenance_pallidal_q10.json.")


def select_q1_rows(all_rows):
    """(rows of scope q1, rows of any other scope) of oxford_q1.csv; no fallback to the other scopes."""
    q1 = [r for r in all_rows if r.get("scope") == Q1_SCOPE]
    other = [r for r in all_rows if r.get("scope") != Q1_SCOPE]
    return q1, other


def null_mode(r):
    """drawn, enumerated or unknown, from the enumeration_feasible_k_le_1e6 column of oxford_q1.csv."""
    v = str(r.get("enumeration_feasible_k_le_1e6", "")).strip().lower()
    return {"false": "drawn", "true": "enumerated"}.get(v, "unknown")


def q1_modes(q1_rows):
    """The set of null modes of the merged Q1 rows."""
    return {null_mode(r) for r in q1_rows}


def contrast_rows(genes):
    rows = []
    for name, role, ga, gb, source in CONTRASTS:
        missing = [g for g in ga + gb if g not in genes]
        n_a = sum(genes.get(g, 0) for g in ga)
        n_b = sum(genes.get(g, 0) for g in gb)
        k = admissible_k(n_a, n_b) if n_a and n_b else 0
        p_min = Fraction(1, k + 1) if k else None
        row = {"contrast": name, "role": role, "group_a": " ".join(ga), "n_a": n_a, "group_b": " ".join(gb),
               "n_b": n_b, "unit": "patient", "k": k, "smallest_p": fmt(p_min) if p_min else "",
               "pass_decision_possible_k_ge_20": k >= K_MIN, "enumerated_k_le_1e6": 0 < k <= ENUM_MAX,
               "genes_missing": " ".join(missing), "source": source}
        primary = role.startswith("primary")
        steps = holm_alpha_min(p_min) if (primary and p_min) else [None] * M
        for j, a in enumerate(steps, 1):
            row[f"holm_alpha_min_step_{j}"] = fmt(a) if a is not None else "outside the Holm family"
        for a in ALPHAS:
            key = f"nr_7_5_ok_at_alpha_{alpha_label(a)}"
            row[key] = (p_min <= a / M) if (primary and p_min) else "outside the Holm family"
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def q1_section(q1_rows, q1_other=(), q1_stamp=None):
    L = []
    L.append("## Slot 1, Q1 (oxford_dystonia_lfp), merged from oxford_q1.csv")
    L.append("")
    if q1_stamp:
        src = (f"measured against MoveAtlas HEAD {q1_stamp.get('head', '?')}, uncommitted diff SHA-256 "
               f"{q1_stamp.get('diff_sha256', '?')} (_provenance_oxford_q1.json)")
    else:
        src = "no MoveAtlas stamp found in _provenance_oxford_q1.json"
    drawn = [r for r in q1_rows if null_mode(r) == "drawn"]
    enum = [r for r in q1_rows if null_mode(r) == "enumerated"]
    unknown = len(q1_rows) - len(drawn) - len(enum)
    if len(drawn) == len(q1_rows):
        mode = ("Every merged row has k above the enumeration limit, so the nulls are drawn without replacement "
                "(NR-1.7) and the smallest attainable p is 1 / (B + 1), with B the draw count of OPEN-33.")
    elif len(enum) == len(q1_rows):
        mode = ("Every merged row has k within the enumeration limit, so the nulls are enumerated and the smallest "
                "attainable p is 1 / (k + 1).")
    else:
        mode = (f"{len(drawn)} of {len(q1_rows)} merged rows have k above the enumeration limit (drawn nulls, "
                f"smallest p 1 / (B + 1), B the draw count of OPEN-33), {len(enum)} are enumerable (smallest p "
                f"1 / (k + 1)) and {unknown} carry no enumeration flag.")
    L.append(f"Rows of scope {Q1_SCOPE} of reports/admissible_nulls/oxford_q1.csv (task T11, "
             f"scripts/probe_oxford_structure.py, {src}). k is the number of distinct non-identity level "
             "assignments (NR-6.1.7); smallest_p there is 1 / (k + 1), the bound under full enumeration. " + mode)
    L.append("")
    L.append("| scope | files | OPEN-06 | patients | hemispheres | n files | k | log10 k | 1 / (k + 1) | k >= 20 | "
             "enumerable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in q1_rows:
        L.append(f"| {r.get('scope', '')} | {r.get('files', '')} | {r.get('open06', '')} | "
                 f"{r.get('n_patients', '')} | {r.get('n_hemispheres', '')} | {r.get('n_files', '')} | "
                 f"{r.get('k', '')} | {r.get('log10_k', '')} | {r.get('smallest_p', '')} | "
                 f"{r.get('pass_decision_possible_k_ge_20', '')} | {r.get('enumeration_feasible_k_le_1e6', '')} |")
    L.append("")
    L.append(f"- Rows with drawn nulls: {len(drawn)} of {len(q1_rows)}.")
    if q1_other:
        scopes = {}
        for r in q1_other:
            scopes.setdefault(r.get("scope", ""), []).append(r)
        parts = []
        for sc in sorted(scopes):
            hem = sorted({str(r.get("n_hemispheres", "")) for r in scopes[sc]})
            parts.append(f"{sc or '(empty)'} ({len(scopes[sc])} rows, {'/'.join(hem)} hemispheres)")
        L.append(f"- Only the rows of scope {Q1_SCOPE} are merged; the rows of the other scopes stay in "
                 "oxford_q1.csv: " + ", ".join(parts) + ".")
    else:
        L.append(f"- oxford_q1.csv has no row of a scope other than {Q1_SCOPE}.")
    if drawn:
        for a in ALPHAS:
            L.append(f"- NR-7.5 at step 1 for drawn nulls: 1 / (B + 1) <= {fmt(a)} / {M} needs B >= "
                     f"{min_draws(a)} (reference alpha {fmt(a)}; alpha is OPEN-01).")
        L.append(f"- At Holm step j the bound is B >= (m - j + 1) / alpha - 1; at step {M} it is "
                 "B >= 1 / alpha - 1.")
    L.append("")
    return L


def q1_slot_cells(q1_rows, q1_present):
    """(null mode text, k, smallest p, alpha min at step 1, alpha min at step m) of the Q1 row of the slot table."""
    if not q1_present:
        return "", "oxford_q1.csv absent", "not merged", "not computed", "not computed"
    if not q1_rows:
        return "", f"oxford_q1.csv has no row of scope {Q1_SCOPE}", "not merged", "not computed", "not computed"
    kk = ", ".join(sorted({r.get("k", "") for r in q1_rows}, key=lambda s: (len(s), s))) + " (by scope)"
    modes = q1_modes(q1_rows)
    if modes == {"drawn"}:
        return "; drawn", kk, "1 / (B + 1), B = OPEN-33", f"{M} / (B + 1)", "1 / (B + 1)"
    if modes == {"enumerated"}:
        return "; enumerated", kk, "1 / (k + 1) by scope", f"{M} / (k + 1)", "1 / (k + 1)"
    return ("; drawn or enumerated by scope", kk, "1 / (B + 1) where drawn, 1 / (k + 1) where enumerated",
            "see the Q1 section", "see the Q1 section")


def summary_md(rows, genes, q1_rows, q1_present, ma_stamp, q1_other=(), q1_stamp=None):
    q10 = rows[0]
    k = q10["k"]
    steps = [q10[f"holm_alpha_min_step_{j}"] for j in range(1, M + 1)]
    q10_mode = "enumerated" if q10["enumerated_k_le_1e6"] else "drawn"
    L = []
    L.append("# Admissible nulls and Holm feasibility of the confirmatory primaries (counts only)")
    L.append("")
    L.append("Status: DRAFT. Counts for the hashing script of docs/rules/NEURAL_RULES_DRAFT.md (NR-7.5); no "
             "outcome, firing value or test statistic was read. Written by scripts/build_admissible_nulls.py; the "
             "patient counts come from reports/pallidal_units/patients_per_gene.csv.")
    L.append("")
    L.append(stamp_line(ma_stamp))
    L.append("")
    L.append("## Holm over m = 6 (NR-7.2, NR-7.3, NR-7.5)")
    L.append("")
    L.append(f"- Holm step j compares the j-th smallest p-value with alpha / (m - j + 1), m = {M}. A primary whose "
             "smallest attainable p is p_min can be rejected at step j only when alpha >= (m - j + 1) * p_min.")
    L.append("- Step 1 (alpha >= m * p_min) is the NR-7.5 check: the primary can be rejected whatever the other "
             "primaries give. Step j > 1 needs the j - 1 primaries with smaller p-values rejected first; a primary "
             "not run enters with p = 1 (NR-7.4) and is never rejected, so with r primaries not run the last "
             f"reachable step is {M} - r.")
    L.append("- The smallest p is attained only when no non-identity null reaches the observed statistic (NR-1.7: "
             "p = (1 + b) / (k + 1)).")
    L.append("")
    L.append("| slot | dataset | primary | null | k | smallest p | alpha min at step 1 | alpha min at step 6 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for slot, ds, q, null in SLOTS:
        if q == "Q10":
            L.append(f"| {slot} | {ds} | {q} | {null}; {q10_mode} | {k} | {q10['smallest_p']} | {steps[0]} | "
                     f"{steps[-1]} |")
        elif q == "Q1":
            mode, kk, p1, a1, am = q1_slot_cells(q1_rows, q1_present)
            L.append(f"| {slot} | {ds} | {q} | {null}{mode} | {kk} | {p1} | {a1} | {am} |")
        elif q in ("F1", "F4"):
            L.append(f"| {slot} | {ds} | {q} | {null} | per record and subject, k >= 20 (NR-1.8) | not from counts: "
                     "designated test and family-level p are OPEN-57 | not computed | not computed |")
        else:
            L.append(f"| {slot} | {ds} | {q} | {null} | none unless null-based | not from counts: OPEN-49 | "
                     "not computed | not computed |")
    L.append("")
    L.append(f"## Slot 7, Q10 ({TITLE})")
    L.append("")
    enum_txt = (f"enumerated (k <= {ENUM_MAX})" if q10["enumerated_k_le_1e6"] else
                f"above the enumeration limit {ENUM_MAX}, drawn (the smallest p below holds under full "
                "enumeration only)")
    L.append(f"- Patients: TOR1A {genes.get('TOR1A', 0)}, SGCE {genes.get('SGCE', 0)} (patients_per_gene.csv). "
             f"k = C({q10['n_a']} + {q10['n_b']}, {q10['n_b']}) - 1 = {k} non-identity assignments, {enum_txt}; "
             f"smallest p = 1 / {k + 1} = {q10['smallest_p']}; k >= {K_MIN}: "
             f"{q10['pass_decision_possible_k_ge_20']} (NR-1.8).")
    L.append("- Smallest alpha at which Q10 can be rejected, by Holm step: "
             + "; ".join(f"step {j} {s}" for j, s in enumerate(steps, 1)) + ".")
    for a in ALPHAS:
        ok = q10[f"nr_7_5_ok_at_alpha_{alpha_label(a)}"]
        L.append(f"- NR-7.5 at reference alpha {fmt(a)}: 1 / {k + 1} <= {fmt(a)} / {M} is {ok}.")
    L.append("")
    if q1_present and q1_rows:
        L.extend(q1_section(q1_rows, q1_other, q1_stamp))
    elif q1_present:
        found = ", ".join(sorted({r.get("scope", "") or "(empty)" for r in q1_other})) or "none"
        L.append("## Slot 1, Q1")
        L.append("")
        L.append(f"- reports/admissible_nulls/oxford_q1.csv has no row of scope {Q1_SCOPE} (scopes found: {found}); "
                 "Q1 is not merged.")
        L.append("")
    else:
        L.append("## Slot 1, Q1")
        L.append("")
        L.append("- reports/admissible_nulls/oxford_q1.csv is absent; Q1 is not merged.")
        L.append("")
    L.append("## Exploratory pallidal contrasts (outside the Holm family)")
    L.append("")
    L.append("| contrast | group a | n a | group b | n b | k | smallest p | source |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows[1:]:
        L.append(f"| {r['contrast']} | {r['group_a']} | {r['n_a']} | {r['group_b']} | {r['n_b']} | {r['k']} | "
                 f"{r['smallest_p']} | {r['source']} |")
    L.append("")
    balanced = [r["contrast"] for r in rows if r["n_a"] == r["n_b"] and r["k"]]
    if balanced:
        L.append(f"- Equal group sizes ({'; '.join(balanced)}): the full label swap is one of the k nulls, and a "
                 "two-sided statistic that is symmetric under that swap ties the observed value, so the smallest "
                 "two-sided p is then 2 / (k + 1).")
    L.append(f"- The contrast on gene lists is titled \"{TITLE_CLASS}\"; the isolated against combined contrast "
             "depends on the classes of reports/gene_classes/ and its count is printed there.")
    L.append("")
    L.append("## Reading")
    L.append("")
    L.append("- NR-7.5 says a primary failing its check \"can never be rejected\". Under Holm that holds at step 1 "
             "only: a primary whose p_min exceeds alpha / m can still be rejected at a later step when other "
             "primaries are rejected first. The check is the condition for rejection whatever the other "
             "primaries give.")
    L.append("")
    return "\n".join(L)


def run(moveatlas):
    os.makedirs(NULLS, exist_ok=True)
    genes = {r["gene"]: int(r["n_patients"]) for r in read_rows(GENES_CSV)}
    rows = contrast_rows(genes)
    q1_present = os.path.isfile(OXFORD_Q1)
    q1_rows, q1_other = select_q1_rows(read_rows(OXFORD_Q1)) if q1_present else ([], [])
    q1_stamp = None
    if q1_present and os.path.isfile(OXFORD_Q1_PROV):
        with open(OXFORD_Q1_PROV, encoding="utf-8") as f:
            q1_stamp = json.load(f).get("moveatlas_stamp")
    ma_stamp = stamp(moveatlas)
    out_csv = os.path.join(NULLS, "pallidal_q10.csv")
    write_csv(out_csv, rows)
    text = summary_md(rows, genes, q1_rows, q1_present, ma_stamp, q1_other, q1_stamp)
    assert "\u2014" not in text and "\u2013" not in text
    out_md = os.path.join(NULLS, "summary.md")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    inputs = {"reports/pallidal_units/patients_per_gene.csv": sha256_file(GENES_CSV)}
    if q1_present:
        inputs["reports/admissible_nulls/oxford_q1.csv"] = sha256_file(OXFORD_Q1)
        if os.path.isfile(OXFORD_Q1_PROV):
            inputs["reports/admissible_nulls/_provenance_oxford_q1.json"] = sha256_file(OXFORD_Q1_PROV)
    pp = os.path.join(TREE, "reports", "pallidal_units", "_provenance.json")
    if os.path.isfile(pp):
        inputs["reports/pallidal_units/_provenance.json"] = sha256_file(pp)
    prov = {
        "script": "scripts/build_admissible_nulls.py",
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "moveatlas_stamp": ma_stamp,
        "q1_rows_merged": len(q1_rows),
        "q1_rows_of_other_scopes_not_merged": len(q1_other),
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "inputs": inputs,
        "parameters": {"M": M, "ENUM_MAX": ENUM_MAX, "K_MIN": K_MIN,
                       "ALPHAS (reference only, OPEN-01)": [str(a) for a in ALPHAS]},
        "outputs": {"reports/admissible_nulls/pallidal_q10.csv": sha256_file(out_csv),
                    "reports/admissible_nulls/summary.md": sha256_file(out_md)},
        "note": "counts only; k per NR-6.10.3, p per NR-1.7, Holm per NR-7.3, check per NR-7.5",
    }
    with open(os.path.join(NULLS, "_provenance_pallidal_q10.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    return rows, q1_present and bool(q1_rows), prov["outputs"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--moveatlas", default=default_root())
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    rows, q1_present, outs = run(a.moveatlas.replace("\\", "/").rstrip("/"))
    for r in rows:
        print(f"{r['contrast']}: n {r['n_a']} and {r['n_b']}, k {r['k']}, smallest p {r['smallest_p']}")
    print("Q10 alpha min by Holm step:", [rows[0][f"holm_alpha_min_step_{j}"] for j in range(1, M + 1)])
    print("oxford_q1.csv merged:", q1_present)
    for k, v in outs.items():
        print(f"  {k} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
