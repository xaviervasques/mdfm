"""Study S2: Q9 at the facet level, set beside the scorer floor (modys_video, Holm slot 6).

Rules: docs/rules/STUDY_FAMILY_2026-11-12.md and docs/rules/STUDY_S2_2026-11-12.md (registered on 2026-11-12),
with docs/rules/STUDY_S1_2026-11-12.md, which S2 reads entire.

S2 section 3, item 2: S2 does not run while S1 section 2 leaves the posture of the two zips open; slot 6 then reads
"not run" and enters the Holm family with p = 1 (NR-7.4, family file section 2, item 3). This script reads the
posture that scripts/study_s1.py, stage posture, wrote (reports/modys_structure/posture_v2.json) and, while it is
open, writes the slot 6 record and reads nothing else: no DIS label, no descriptor, no feasibility table and no
within-source replication. When the posture is settled the script stops: the primary run is written and tested
before it runs.

Writes: reports/studies/s2/slot6.md and slot6.json.
Run: PYTHONDONTWRITEBYTECODE=1 <conda>/envs/mdfm/python.exe scripts/study_s2.py
     --moveatlas <moveatlas>  (or MOVEATLAS_ROOT)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import study_s1 as S1  # noqa: E402

NR_DRAFT = "docs/rules/NEURAL_RULES_DRAFT.md"
NR_SHA256 = "9e629754cca7e6511eead98048bfd8f079b945134424c2cfaf8700aa45d4174e"
M_FAMILY = 6          # NR-7.2, family file section 1
ALPHA = 0.05          # OPEN-01, family file section 2, item 2
OUT_S2 = os.path.join(HERE, "reports", "studies", "s2")


def holm_adjusted(pvals):
    """Holm adjusted p-values in input order (NR-7.3): p_adj(i) = max over l <= i of min(1, (m - l + 1) p(l))."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def slot6_not_run(posture, rules, moveatlas):
    """The slot 6 record when the posture is open: p = 1, final whatever the other five slots give."""
    p = 1.0
    worst = holm_adjusted([0.0] * (M_FAMILY - 1) + [p])[-1]      # the smallest adjusted p slot 6 can reach
    return {"dataset": "modys_video", "slot": 6, "primary": "Q9 at the facet level (S2 section 3)",
            "outcome": "not run", "reason": "S1 section 2 leaves the posture of the two zips open "
            "(reports/modys_structure/posture_v2.md): S2 section 3, item 2", "p_raw": p,
            "holm_adjusted_p": worst, "holm_note": "a p of 1 has a Holm-adjusted p of 1 whatever the other five "
            "slots give (NR-7.3), so the slot reads not run, final (family file section 2, item 3)",
            "k": None, "smallest_p": None, "interval": None, "units_of_independence": "27 training children "
            "(none read by S2)", "m": M_FAMILY, "alpha": ALPHA, "rules": rules, "moveatlas_head": moveatlas["head"],
            "posture": posture, "read": "nothing: no DIS label, no descriptor value, no feasibility table, no li_taati "
            "or hd_accel_jnet record"}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--moveatlas", default=None)
    args = ap.parse_args(argv)
    root = S1.root_of(args)
    rules = S1.check_rules((S1.FAMILY, S1.S2_RULE, S1.S1_RULE))
    nr = S1.sha256_file(os.path.join(HERE, NR_DRAFT))
    assert nr == NR_SHA256, "the neural rule draft differs from the SHA-256 S2 adopts"
    rules[NR_DRAFT] = nr
    posture = S1.load_posture()
    assert posture["probe_v2_json_sha256"] == S1.sha256_file(os.path.join(S1.OUT_STRUCT, "probe_v2.json"))
    if posture["settled"]:
        sys.exit("the posture is settled: the S2 primary run is to be written and tested before it runs")
    hdr, ma = S1.header_lines(root, rules)
    rec = slot6_not_run(posture, rules, ma)
    os.makedirs(OUT_S2, exist_ok=True)
    with open(os.path.join(OUT_S2, "slot6.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write("\n")
    md = ["# Study S2, Holm slot 6 (modys_video): not run", ""] + hdr + [
          "", "| item (NR-7.9) | value |", "|---|---|",
          f"| dataset | {rec['dataset']} |", f"| primary | {rec['primary']} |",
          f"| outcome | {rec['outcome']} |", f"| reason | {rec['reason']} |",
          "| k and 1 / (k + 1) | none: no null drawn |", f"| raw p-value | {rec['p_raw']:.0f} (NR-7.4) |",
          f"| Holm-adjusted p-value | {rec['holm_adjusted_p']:.0f} ({rec['holm_note']}) |",
          "| interval [L, U] | none |", f"| units of independence | {rec['units_of_independence']} |",
          f"| family | m = {rec['m']}, alpha = {rec['alpha']} (family file sections 1 and 2) |", "",
          "Posture record read: reports/modys_structure/posture_v2.json (step 1 settled: "
          f"{'yes' if posture['step1_settled'] else 'no'}; step 1 reading {posture['step1_reading']}; step 2 viewing "
          f"{posture['step2_viewing']}).", "",
          f"Read by S2: {rec['read']}.", ""]
    S1.write_text(os.path.join(OUT_S2, "slot6.md"), "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
