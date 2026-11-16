# The confirmatory family under modular pre-registration, dated version of 2026-11-12

Status: APPROVED by the owner on 2026-11-12.

Adopts docs/proposals/CONFIRMATORY_FAMILY_DRAFT.md, SHA-256
f4f38c51f6a2debb58f2d58d775b7f77e093ad573bc29b90057017f8d470d962, as committed at MD_FM 14214e2 ("the draft"), and
through it NR-7.1 to NR-7.10 of docs/rules/NEURAL_RULES_DRAFT.md (SHA-256
9e629754cca7e6511eead98048bfd8f079b945134424c2cfaf8700aa45d4174e). Every section of the draft stands except what
sections 1 to 5 below replace or add; where they disagree, this file governs. Written by the session on 2026-11-12
under the owner's delegation of 2026-11-03 (decisions A14 and A21 of docs/OWNER_DECISIONS_2026-11-12.md, and the
choices of section 2 logged in reports/gpu_queue/decisions.md, entry of 2026-11-12 for the dated rules) and the owner's
answers C2 and C8 of 2026-11-12. English, no em-dashes.

What changed since the draft: modular registration is adopted (the note's question 6, decision A14); OPEN-01 and
OPEN-56 take the draft's defaults; slots 2, 4 and 5 count as not run at closing; the closing date is fixed; the
rule-draft defects of task T22 are corrected for every file of the family (section 4); no study reads a model
(section 5).

How it becomes binding. The owner reads this file in chat and answers "oui famille"; the session then runs, from
D:\MD_FM\mdfm with PYTHONDONTWRITEBYTECODE=1, `C:/Users/vasqu/anaconda3/envs/mdfm/python.exe
scripts/register_dated_rule.py --gate FAMILY --path docs/rules/STUDY_FAMILY_2026-11-12.md --expect <the SHA-256
prefix given with the text> --approved-on YYYY-MM-DD`, which changes only the status line above to `Status: APPROVED
by the owner on YYYY-MM-DD.` and records the file's SHA-256 through mdfm.gates.common.register_rule. This file is
registered before any study file; a study file names it by path and its registered SHA-256 is printed on every
confirmatory output.

## 1. The files of the family and their order (decision A14)

| slot | dataset | primary | carried by |
|---|---|---|---|
| 1 | oxford_dystonia_lfp | Q1, level-by-amplitude interaction | docs/rules/STUDY_S3_2026-11-12.md |
| 2 | pd_meg_stn_lfp_ds004998 | family F1 | the neural draft; not run (section 2) |
| 3 | physionet_multimodal_gait | F3, not run, not counted (OD10, Holm over six) | none |
| 4 | merk_neumann_ecog_lfp | family F4 | the neural draft; not run (section 2) |
| 5 | seizeit2 | Q8 | the neural draft; not run (section 2) |
| 6 | modys_video | Q9 at the facet level | docs/rules/STUDY_S2_2026-11-12.md |
| 7 | pallidal_units_dystonia | Q10, TOR1A against SGCE | docs/rules/STUDY_S4_2026-11-12.md |

docs/rules/STUDY_S1_2026-11-12.md is an estimation study and holds no slot. Order of registration: this file; then S1
and S2 together; S3 and S4 in any order after this file. Each study file is registered before any datum of its study
is read; m stays 6 and no primary enters, leaves or changes slot after this file is registered (NR-7.2).

## 2. The draft's decisions for the owner (made under the delegation)

1. The note's question 6: modular pre-registration, this file first (decision A14).
2. OPEN-01: family-wise alpha 0.05 for the Holm family; 95 percent two-sided intervals for the [L, U] of every
   confirmatory slot. The 90 percent conformal readout intervals of DR-3.3 are another object and are not changed.
3. OPEN-56: NR-7.4 confirmed (a primary not run enters with p = 1, m stays 6), with the modular reading of draft section
   5: a slot read before the others is final at once when it reads absent or undetermined, or when it reads pass by its
   own rule with m x p <= alpha; otherwise it reads "pass pending the family" until closing.
4. Slots 2, 4 and 5 get no study file: they stay in the neural draft, which no phase of docs/PLAN_2026-11-12.md reads
   (decisions B14 to B16). They count as not run at closing, p = 1.
5. Closing: when every slot has a p-value or a not-run p = 1, or on 2027-09-23, one year after the program's start of
   2026-09-23 (the year 1 of NR-7.1), whichever comes first. Holm then runs once over the six p-values and every
   pending slot reads its final outcome.

## 3. Shared values

Each study file restates the DR and NR values it reads unchanged and names its departures. Before each study file is
given to the owner, scripts/check_rule_drafts.py runs over docs/rules and docs/proposals with its output in
reports/rule_checks/dated_2026-11-12/, and a shared-value mismatch it reports on a value the study file states is
corrected first.

## 4. Corrections of the drafts the family cites (task T22; decision A21)

The rule-draft checker (reports/rule_checks/summary.md, scripts/check_rule_drafts.py, 2026-10-01; task
reports/weekend_2026-10-01/tasks/T22.md) found 6 stale lockbox references, 5 sentences that call an existing file
absent and 4 paired OPEN items that close at different points. The drafts stay unedited; for every file of this family
the following read in their place:
- Lockbox version, 6 places: lockbox v1.2 (reports/lockbox/lockbox_persons.csv, file SHA-256
  619dfdacef464387ab5447299534e98b152507dea9bca91c95f68e3324762162, rebuilt on 2026-10-05 with the same SHA-256) where
  NR-4.6, NR-6.9.1, NR-6.10.6, NR-12.1 and NR-12.5 write v1.1, in the words of items C1 to C5 of
  docs/rules/NEURAL_RULES_DRAFT_REVISION_2026-10-06.md (SHA-256
  dce7a0e3af2a6ca0f42d77b8cefcef8f0c1db8ca4d35bf2355b3ef520fe47ea5); and in DR-6.2 of the descriptor draft, the norm
  sources "each minus its lockbox v1.2 persons" (v1.2 changed weargait_pd only, reports/lockbox/v1_2_changes.csv).
- Files that exist, 5 sentences: docs/rules/DESCRIPTOR_RULES_DRAFT.md exists as a draft, not hashed, in NR-5.6 (c), the
  preamble of neural section 9 and OPEN-79 (items C6 to C8 of the same revision), and in gap T4 of
  docs/rules/CODY_CONSENSUS_RULE_DRAFT.md; in its gap T3, mdfm/gates/ exists while scripts/check_cody_exclusion.py,
  scripts/check_cody_rule_file.py and scripts/cody_label_census.py are still absent (checked 2026-11-12). No study file
  reads the CODY draft (decision B18). NR-11.1 reads as item D1 of the same revision (reports/neural_lockbox/mdd/
  exists since 2026-10-06).
- Paired OPEN items, 4 pairs: OPEN-D53 with OPEN-51 and OPEN-D57 with OPEN-67 are closed together, by one text each, in
  docs/rules/STUDY_S1_2026-11-12.md (sections 2 to 4), which S2 reads. OPEN-D27 with OPEN-16 (the tremor band) and
  OPEN-D45 with OPEN-21 (the coherence segment settings) are read by no study of the family; whichever later file
  closes one member closes the other with the same text, the earlier closure governing both (item C10 of the same
  revision), and the descriptor item takes the class of its neural pair.

## 5. No study reads a model (owner decision C8)

S1 to S4 read raw sensor streams, keypoints and tables, never a checkpoint or a body: the port of the model to the open
body changes no slot. A replication on body posteriors, where a study names one, reads the checkpoint its own
declaration names, sits outside Holm and decides nothing.
