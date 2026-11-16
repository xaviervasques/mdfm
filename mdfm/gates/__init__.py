"""Gate harnesses (design section 9; spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, sections 2.7 and 6).

common.py: person-weighted statistics, person-clustered bootstrap, verdicts, rule files and their
registered hashes (docs/rules/gate_rule_hashes.csv), the lockbox read log
(reports/gates/lockbox_reads.csv), result writers. g1.py: the G1 harness (inputs, truth, scoring, the
population-prior trivial baseline).

Only code in this package passes allow_lockbox=True or allow_eval=True to the readers (a grep test in
tests/test_amass_index.py), and no module of mdfm/train or mdfm/model imports it (an import-graph test
in tests/test_gates_common.py). A lockbox read needs the gate's rule file approved by the owner and
registered with its hash, and is logged once per gate, checkpoint and rule before any lockbox array is
read.
"""
