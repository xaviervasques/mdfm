# Addendum 1, dated revision of 2026-10-06: the neural census corrections

Date: 2026-10-06. Status: DRAFT. The owner approved on 2026-10-05 that the five corrections found by the neural
census (weekend task T23) be applied in a dated revision of addendum 1 (docs/OWNER_ACTIONS.md section 2: the
neural census item, and the answers of 2026-10-05, "the neural census corrections go into a dated addendum
revision"). This file is that revision; the owner has not yet read its text.

It amends docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md, called "the addendum" below, which is not edited. From
this date, where this file and the addendum disagree, this file holds. The companion revision of the neural
rule draft is docs/rules/NEURAL_RULES_DRAFT_REVISION_2026-10-06.md.

Every number below is quoted from a script output named beside it. The census reports (reports/merk/,
reports/neural_manifests/, reports/ataxie_diagnoses/, reports/ds004998/) were measured on 2026-10-01 on the
MoveAtlas working tree at HEAD f2e6f99 with the then uncommitted correction batch 0002; the stamp in each
_provenance.json holds the hash of that diff. The owner committed the batch as MoveAtlas 05a9c10 on 2026-10-05
(docs/OWNER_ACTIONS.md section 1). This task did not check that the committed tree equals the measured one and
did not rerun the four probes.

How to read each item: the addendum section, the text it replaces (quoted verbatim), the new text, the evidence.

## Summary

| id | addendum sections | change | evidence |
|---|---|---|---|
| R1 | 2.2 (N-LFP) | merk carries MNI coordinates for its STN and ECoG contacts; the statement "no imaging localizes contacts" no longer covers every in-scope set; the STN runs of sub-003 lie off the cluster of the other STN runs | reports/merk/summary.md, facts.json (mni), checks MRK-08, MRK-09, MRK-10 |
| R2 | 1.1 | physionet_multimodal_gait EEG: 20 channels (18 scalp sites, A1, A2), every one referenced to Pz, which is not a data column | reports/neural_manifests/summary.md, check EEG-10 |
| R3 | 1.1 | multimodal_fog: the five channels are 3 EMG, ECG and EOG | reports/neural_manifests/summary.md, check EMG-02 |
| R4 | 1.1, 1.6 item 12, 2.5, 3 (Q7) | merk grip force is centred per run but has unit variance in 59 of 146 run-channels only; no run lasts 10 minutes | reports/merk/summary.md, facts.json (force_main), checks MRK-03, MRK-04, MRK-05 |
| R5 | 4.7 | "No genotyped body data is in scope" is qualified by ataxie_kinect_dib's genetic-ataxia labels | reports/ataxie_diagnoses/summary.md, checks ATX-01, ATX-03, ATX-04 |
| R6 | 1.5, 3 (Q4) | ds004998 counts move from BIDS runs to recordings, as in the neural rule revision; the order-null design stays open | reports/ds004998/summary.md sections 2, 3 and 4, order_null_feasibility.csv |

R1 to R5 are the five approved corrections. R6 is not one of them: it carries into the addendum the change
"from runs to recordings" that the neural rule revision of the same day makes (weekend task T24; the owner's
sheet of 2026-10-05 proposed rewording both texts), so that the two documents state the same counts. It changes
counts only. The choice among the three order-null designs stays the owner's, with no default.

## R1. Contact localization: merk has MNI coordinates (section 2.2, N-LFP)

Replaces (2.2, bullet N-LFP, third sentence):

> No imaging localizes contacts for any in-scope cohort, so the token identity is lead level and montage, not a
> nucleus, and region-level statements ("zona incerta tracks the phasic change") are forbidden for every set
> without contact localization (RT).

New text:

> One in-scope set localizes its contacts: merk_neumann_ecog_lfp carries MNI coordinates for every STN channel
> of its 20 STN runs (60 rows, 12 subjects) and for every ECoG channel of its 73 runs (871 rows). The run files
> are named space-mni while their session coordinate files declare "Other" in metres; the per-subject
> electrodes files declare MNI in millimetres (22 files) or "Others" in millimetres (1 file), and the run rows,
> converted to millimetres, equal the per-subject rows within 0.0 mm. The STN rows of 18 runs lie within 10 mm
> of the median STN centroid (|x| 13.337, y -13.802, z -5.709 mm); both STN runs of sub-003 lie farther (23.8
> and 34.1 mm), so sub-003 enters no location-dependent use until its coordinates are checked. A
> location-dependent use is any output that selects, groups, weights or labels a contact by its coordinates or
> by a region. On merk a contact may be identified by its MNI coordinate as the source gives it; the
> coordinates are not checked against an atlas, so a region label for a merk contact needs an atlas and an
> assignment rule written in the neural rule file first. The montage of the merk STN channels rests on the
> datacard (3 bipolar channels): the sidecars declare no reference (iEEGReference n/a), and the 3 STN rows of
> every STN run equal contacts 0, 1 and 2 of the 4-contact lead of the per-subject file. For every other
> in-scope set the statement of 2.2 stands: no imaging localizes contacts, so the token identity is lead level
> and montage, not a nucleus, and region-level statements ("zona incerta tracks the phasic change") are
> forbidden for every set without contact localization (RT).

Evidence: reports/merk/summary.md ("The MNI fact"), facts.json keys mni.runs_stn_all_finite (20),
mni.seeg_rows_finite (60), mni.ecog_rows_finite (871), mni.subject_coordsystems, mni.max_abs_diff_run_vs_subject_mm
(0.0), mni.stn_near_runs (18), mni.stn_median_centroid_mm, mni.stn_far_runs; addendum_checks.csv rows MRK-08
(corrected), MRK-09 and MRK-10 (qualified); the 10 mm threshold is STN_FAR_MM of scripts/probe_merk.py. The
census checked merk only; item 1 of the appendix of docs/feedback/CLINICAL_QUESTIONS_CROSSWALK.md keeps the
statement for the in-scope dystonia sets.

Unchanged and consistent with R1: section 8, "no nucleus label on a contact without imaging"; 2.2 N-ECoG.

## R2. physionet_multimodal_gait EEG channels (section 1.1)

Replaces (1.1, row physionet_multimodal_gait, column "neural signal"):

> scalp EEG 19 channels, 300 Hz, reference Pz, 176 records, 2.93 h; every record 59.87 s (EEGm, S)

New text:

> scalp EEG 20 channels (18 scalp sites plus A1 and A2), 300 Hz, every channel referenced to Pz, which is not a
> data column, 176 records, 2.93 h; every record 59.87 s (EEGm, S; reports/neural_manifests/, EEG-10)

Evidence: reports/neural_manifests/summary.md, check EEG-10 (corrected): 20 channels in every record, one site
list over the 176 records (C3 C4 CZ F3 F4 F7 F8 FP1 FP2 FZ O1 O2 P3 P4 P7 P8 T7 T8, A1, A2); every channel
referenced to PZ in 176 of 176 record metas; "Reference location" Pz in 177 of 177 raw headers. The "19
channels" of the addendum counted the 18 scalp data channels plus the Pz reference (section "The
physionet_multimodal_gait montage" of the same summary). Unchanged: 2.2 N-EEG, "Declared reference (Pz in
ds004584 and physionet_multimodal_gait, ...)" (EEG-10, EEG-11).

## R3. multimodal_fog channels (section 1.1)

Replaces (1.1, row multimodal_fog, column "movement on the same clock", first clause):

> 5 EMG, ECG and EOG channels;

New text:

> 5 channels: 3 EMG (tibialis anterior left and right, gastrocnemius right, each in all 56 records), ECG and EOG;

Evidence: reports/neural_manifests/summary.md, check EMG-02 (qualified): 5 channels per record, of which EMG 3,
ECG 1 and EOG 1 (record meta); EMG muscle sites over the 56 records: gastrocnemius R in 56, tibialis_anterior L
in 56, tibialis_anterior R in 56; 500 Hz. Unchanged: 2.2 N-EMG drive band ("multimodal_fog 500").

## R4. merk grip force: centred, not unit variance (sections 1.1, 1.6 item 12, 2.5, 3 Q7)

R4a. Replaces (1.1, row merk_neumann_ecog_lfp, column "movement on the same clock"):

> grip force left and right in all 73 runs, z-scored, so unitless (DC, S); run duration 39 to 569 s, median
> 288 s (S)

New text:

> grip force left and right (MOV_LEFT, MOV_RIGHT) in all 73 runs, centred on each run (146 of 146
> run-channels) but with unit variance in only 59 of 146 (standard deviation 0.713 to 1.742, median 1.063), so
> standardized by a factor other than the run's own standard deviation: unitless, and not on one scale across
> runs; the files label these channels microvolt, a unit the values do not carry; the MOV_*_CLEAN channels are
> rescaled traces (minimum 0, maximum 1, zero in 0.836 to 0.977 of the samples), not force; run duration 39 to
> 569 s, median 288 s (reports/merk/, MRK-03, MRK-04)

R4b. Replaces (1.6, item 12):

> merk_neumann_ecog_lfp: grip force is z-scored, so it carries no physical unit (DC), and run durations are 39 to
> 569 s, median 288 s (S), against the datacard's "10-25 min/run".

New text:

> merk_neumann_ecog_lfp: grip force carries no physical unit; the datacard calls it z-scored, but as stored it is
> centred on each run and has unit variance in only 59 of 146 run-channels (MRK-03). Run durations are 39 to
> 569 s, median 288 s, and no run lasts 10 minutes (the longest 9.5 min), against the datacard's "10-25 min/run"
> (MRK-04, MRK-05). Both datacard statements are MoveAtlas correction candidates.

R4c. Replaces (2.5, bullet "Outputs", parenthesis):

> (standardized and flagged unitless where the movement channel is z-scored, as in merk)

New text:

> (standardized and flagged unitless where the movement channel carries no unit, as merk grip force, centred
> per run on a scale that differs between runs)

R4d. Replaces (3, Q7, bullet "Data"):

> merk (grip force in all 73 runs, z-scored, ECoG in 73, STN in 20; runs 39 to 569 s)

New text:

> merk (grip force in all 73 runs, centred per run and unitless, ECoG in 73, STN in 20; runs 39 to 569 s)

Evidence: reports/merk/facts.json, force_main: runs_channels 146, centred 146, unit_variance 59, std_min 0.7130,
std_median 1.0632, std_max 1.7418, header and channels.tsv unit microvolt; force_clean: min_equal_zero 146,
max_equal_one 146, share_zero_min 0.8355, share_zero_max 0.9771; duration_s min 39.001, median 288.0, max 569.0;
addendum_checks.csv rows MRK-03 (qualified), MRK-04 (confirmed), MRK-05 (corrected: 0 of 73 runs last 10 to 25
min, the longest 9.5 min). Fixed definitions (ZS_MEAN_TOL 0.01, ZS_STD_TOL 0.05) in scripts/probe_merk.py.

Unchanged: section 6 G19, F4 "gains standardized"; section 8, "merk force" among the undeclared units.

## R5. Genotype and phenotype, qualified by ataxie_kinect_dib (section 4.7)

Replaces (4.7, first bullet):

> In-scope data: pallidal_units_dystonia only (Q10). No genotyped body data is in scope.

New text:

> In-scope data: pallidal_units_dystonia (Q10). No body data with variants is in scope. ataxie_kinect_dib
> (admitted on 2026-09-24, 1.4) comes closest: its one table pairs, for each of 51 participants (15 with
> progressive ataxia, 16 with chronic ataxia, 20 healthy), a diagnosis label with SARA and 9 spatio-temporal
> gait parameters, and 26 of its 31 patients carry the label of a genetic ataxia (frda 10, adck3 6, joubert s.
> 4, arsacs 2, at 1, itpr1 1, sca 1, sca2 1), 21 of them a gene symbol or a single-gene disorder. These are
> diagnosis labels, not variants, and per-participant summaries computed by the authors, not a body signal: the
> set supports at most a descriptive label-to-gait table, outside the hyperkinetic program, and no gate or
> question of this addendum reads it.

Evidence: reports/ataxie_diagnoses/summary.md, addendum_checks.csv rows ATX-01 (qualified: 9 spatio-temporal
gait parameters and SARA per participant; 26 of 31 patients with a genetic-ataxia label, 21 of them a gene
symbol or a single-gene disorder; 0 columns resembling a genotype), ATX-03 (51 participants: CA 16, H 20, PA
15), ATX-04 (class counts). The label expansions come from Table 1 of the article (Summa et al. 2020, Data in
Brief, PMC7498847), held as a constant in scripts/probe_ataxie_diagnoses.py.

Unchanged: 1.4 ("holds a supplementary table only, no skeleton data", ATX-02 confirmed); the other rules of 4.7;
appendix A3, "Genotype to body phenotype mapping, stage 2".

## R6. ds004998: from BIDS runs to recordings (sections 1.5 and 3 Q4)

The 108 task runs of ds004998 are 76 recordings: 32 run-2 files repeat the end of their run-1 (same meas_date,
sample range inside run-1), and the one run-2 that is a separate recording is sub-BYJoWR MoveR MedOff
(reports/ds004998/summary.md section 2).

R6a. Replaces (1.5, bullet "ds004998 medication order is a verified confound", from its first clause to the
sentence on sub-iDpl28):

> in 18 of 20 subjects every MedOff run precedes every MedOn run, with 21 to 77 min between blocks;
> sub-AbzsOg and sub-BYJoWR interleave, and in sub-BYJoWR one MedOff run at 09:20 follows the only MedOn run at
> 08:24, a possible label error for the owner (S, from the scans.tsv acq_time columns). Runs by state and task:
> MedOff HoldL 16, HoldR 9, MoveL 20, MoveR 8, Rest 3; MedOn HoldL 17, HoldR 9, MoveL 16, MoveR 6, Rest 4. An
> order-only null is feasible: 11 of 20 subjects have at least two MedOff runs of the same task, and 19 of 20
> have at least two MedOff runs of any task (S). sub-iDpl28 has MoveL runs only in MedOff and MoveR runs only in
> MedOn (S), so its move runs cannot be task-matched.

New text:

> in 18 of 20 subjects every MedOff recording starts before every MedOn recording, 21.2 to 76.6 min from the
> acq_time of the last MedOff recording to that of the first MedOn one (12.7 to 66.0 min from the end of the
> one to the start of the other); sub-AbzsOg and sub-BYJoWR interleave, and in sub-BYJoWR one MedOff recording
> at 09:20 follows the only MedOn recording at 08:24, a possible label error for the owner. BIDS runs by state
> and task: MedOff HoldL 16, HoldR 9, MoveL 20, MoveR 8, Rest 3; MedOn HoldL 17, HoldR 9, MoveL 16, MoveR 6,
> Rest 4; as recordings: MedOff HoldL 12, HoldR 6, MoveL 13, MoveR 6, Rest 2; MedOn HoldL 12, HoldR 6, MoveL 12,
> MoveR 5, Rest 2. Counted in recordings with distinct acquisition times, a task-matched order-only null is
> feasible in 1 of 20 subjects (sub-BYJoWR) under either reading of task matching, and 18 of 20 subjects have two
> MedOff recordings of distinct times over all tasks, a hold and a move recording; the "11 of 20" and "19 of
> 20" first written here counted BIDS runs. The order-null design (between the hold and the move MedOff
> recordings, early against late inside one MedOff recording, or none, with every medication result reported
> as "state or time") is the owner's decision, with no default. sub-iDpl28 has MoveL recordings only in MedOff
> and MoveR recordings only in MedOn, so its move recordings cannot be task-matched under the side-specific
> reading.

R6b. Replaces (3, Q4, bullet "The confound, measured"):

> in 18 of 20 subjects every MedOff run precedes every MedOn run by 21 to 77 min (1.5).

New text:

> in 18 of 20 subjects every MedOff recording starts before every MedOn recording, 21.2 to 76.6 min apart (1.5).

R6c. Replaces (3, Q4, bullet "Test", second sentence):

> An order-only null is trained first with the same pipeline to separate earlier from later task-matched MedOff
> runs, which 11 of 20 subjects support.

New text:

> An order-only null is trained first with the same pipeline to separate earlier from later task-matched MedOff
> recordings, which 1 of 20 subjects supports once the run-2 copies are set aside (1.5); while no order-null
> design is chosen, the order-only AUROC cannot be formed, so no medication result can read as a medication
> effect under the next sentence.

Evidence: reports/ds004998/summary.md sections 2 (run-2 copies), 3 (the state by task table, columns "BIDS
runs" and "recordings"), 4 (order census: 18 of 20, 21.2 to 76.6 min and 12.7 to 66.0 min, the interleaved
sequences, the feasibility table: BIDS runs with ties 11, recordings with distinct acq_time 1, any task with
distinct acq_time 18); order_null_feasibility.csv; addendum_checks.csv rows "A1.5 runs by state and task" (holds
as a count of BIDS runs, does not hold as a count of recordings), "A1.5 and NR-6.4.4 order null" (does not hold),
"A1.5 any task" (18 of 20 as recordings), "A1.5 sub-iDpl28" (holds). The consequence for G21 is computed in
reports/neural_lockbox/mdd/summary.md section 5 (scripts/neural_mdd.py: the task-matched order null feasible in 1
of 20 subjects counted in recordings, 11 of 20 counted in BIDS runs with ties).

## Records of the 2026-09-24 pass that these corrections supersede (section 9, not rewritten)

Section 9 lists what the pass of 2026-09-24 checked; it stays as a record. Two of its lines are superseded:

| section 9 line | superseded by |
|---|---|
| "merk_neumann_ecog_lfp: 73 runs, durations 39 to 569 s with a median of 288 s; force z-scored per the datacard." | R4 (reports/merk/, MRK-03, MRK-04) |
| "ds004998: ... and the run table by state and task (11 of 20 subjects with at least two MedOff runs of the same task)." | R6 (reports/ds004998/summary.md sections 3 and 4) |

## Seen while revising, not applied (outside the approval of 2026-10-05)

- 1.1, row merk_neumann_ecog_lfp, licence column "CC0 (DC; no LM row)": the licence matrix row exists since
  2026-09-24 (1.6 item 9, 2.7). Proposed text: "CC0 (DC and LM)".
- 1.1, row pd_meg_stn_lfp_ds004998, "4 EMG at 2000 Hz, muscles not named": 4 task runs carry a fifth EMG channel,
  EMG096, all of sub-AbzsOg (reports/ds004998/addendum_checks.csv, row "A1.1 EMG; A1.7", holds with a correction).
- 2.9, stage N0 list, "the minimum detectable difference published per G16 and G21 target": computed on
  2026-10-06 by scripts/neural_mdd.py (reports/neural_lockbox/mdd/, plan N0.4) as a DRAFT input to NR-11; the
  standard error it needs is still OPEN-73 of the neural rule draft.
- The four census reports predate the MoveAtlas commit 05a9c10 (header of this file); the program's rules ask
  for a rerun of the dependent reports when a record changes. scripts/probe_merk.py, probe_neural_manifests.py,
  probe_ataxie_diagnoses.py and probe_ds004998.py were not rerun by this task; a rerun before the neural hash
  would let this revision cite 05a9c10.
