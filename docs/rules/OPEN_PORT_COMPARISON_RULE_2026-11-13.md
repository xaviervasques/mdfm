# Open port comparison rule, dated version of 2026-11-13: the open model against v1, seed by seed, on dev

Status: APPROVED by the owner on 2026-11-14.

Written by the session on 2026-11-13 for step M5 of docs/MODEL_OPEN_PORT_PLAN.md (decisions P1, P8 and P9 there),
under the owner's delegation of 2026-11-03 and the owner's answers C2 and C8 of 2026-11-12. English, no em-dashes.

What it decides, on development sets only: whether the open model (the open body Anny, the open body store, the v1
recipe otherwise, decision P1) is non-inferior to v1 (SMPL+H, AMASS body truth), line by line and seed by seed; and
when the result on seed 20260928 stops step M6 before seeds 20261001 and 20261013 and the six G12 baseline runs are
spent (decision P9). It reads no lockbox, EVAL or CODY record. It decides no gate line and no lockbox read (each gate
decides by its own registered rule), and it does not reopen C8: its verdict is the measured cost of the port.

Numbers. Every value of section 4 is read from the report it names. The arithmetic of section 4 (ranges, products,
maxima) and every comparison are computed by scripts/open_port_comparison.py, to be written before the first
comparison; its test asserts that the script's margins equal the values of section 4 at the precision printed, as
tests/test_f2_lever2_rule.py does for reports/bench/f2_lever2/rule.txt.

How it becomes binding. Section 1 names the SHA-256 of configs/stage1_open.yaml and of its two seed copies as they
stand when this file is read (checked current on 2026-11-14). The owner reads this file in chat and answers
"oui OPEN_PORT". The session runs, from D:\MD_FM\mdfm with PYTHONDONTWRITEBYTECODE=1,
`C:/Users/vasqu/anaconda3/envs/mdfm/python.exe scripts/register_dated_rule.py --gate OPEN_PORT --path
docs/rules/OPEN_PORT_COMPARISON_RULE_2026-11-13.md --expect <the SHA-256 prefix given with the text> --approved-on
YYYY-MM-DD`, which changes only the status line above and records the file's SHA-256 in docs/rules/gate_rule_hashes.csv
(scripts/gate_table.py lists the gates of its GATES table, which does not name OPEN_PORT; the registry row is this
rule's record). It is registered together with docs/rules/G1_RULE_OPEN_2026-11-13.md (G1 version 2), whose halves the
G1 lines read, before the first open run starts. No comparison number is computed before both are registered. A later
version of this rule may be registered until the first open number exists (the phase A read of section 6); after
that, none: the margins and the stop condition are fixed before the open model has been measured.

## 1. The two models, the seeds and the declaration

- Open model: seeds 20260928, 20261001 and 20261013 of configs/stage1_open.yaml, configs/stage1_open_s2.yaml and
  configs/stage1_open_s3.yaml (equal but for their name and seed, as v1's three), read on 2026-11-14 after the
  rest-joint term entered them (losses.sup_rest [10, 10], the weight chosen by the learnability scan of
  reports/bench_open/rest_scan/; plan step M4): configuration SHA-256 (train/config.sha256, the config_sha256 every
  checkpoint records)
  20260928: e81486e7f7c8d53277d0cddbc8eb4831b76cc4a7e5a87e259ada5e4253f00d91 (file 9d59efdcc2e4ee1e...);
  20261001: 66ddadda585a0f38721ad6952c129809d4a96d2a843b8261113befe6b677ed03 (file 1ff07da3cc2d6274...);
  20261013: 6e42d3f39071e0e3fec43d29c1dc872f8b7c08e45448340e325344f672b19388 (file 839f3fe5a5dcb93b...).
  The open recipe departs from v1's in the body, its data and this one term (v1 has no rest-joint term: on SMPL+H
  the shape NLL alone held the rest height within 1 cm on the learnability batch, reports/bench/learnability.md;
  on Anny it did not, reports/bench_open/shape_sensitivity.md), which the comparison measures with the rest of the
  port.
  A checkpoint whose config_sha256 differs from its seed's is not an arm of this comparison. The phase B end of each
  (step 32,044 of the v1 recipe, plan section 4.1), EMA weights.
- v1: the phase B ends ckpt_B_032044 of phaseA_root_full (seed 20260928, SHA-256
  cffccdac4109dfbfb921ed4ebcb690ce27dd2d35045a3afdaaca7d38f321a677), pilot_v1_s2 (20261001,
  5e97c4fe3b208f8f11850a5318ed827d26316f1e2fc867355d74f50a3d96e023) and pilot_v1_s3 (20261013,
  8211e469e26af21d6dc04f57a476fa774c309f20e2dcd4218d071854edb74f17) (checkpoint blocks of
  reports/gates/g1/dev_<run>_ckpt_B_032044/g1.json), EMA weights, each loaded under the archived records state it was
  born on (plan step M7: records_state ca5a2d9eaada02b70f834d20ebc77f19de7b89719a0aed32e352faed0abdb8d8 for the first
  two, 0a453d1ce2335cee32374f410987446ab84f5d146dddc07349eccfa30bc680bf for the third,
  reports/provenance_history/records_archive.json), with the SMPL+H body reached only behind MDFM_SMPLH_REFERENCE=1
  (plan step M2).
- Seed matching: the open seed s is compared with the v1 seed s, for s in the three seeds; never across seeds.
- Same code: both models' rows of a line come from one evaluation code SHA-256, one MoveAtlas commit and the settings
  of section 3. v1 is rerun for this comparison; a v1 number of an earlier report is never a comparison row (those
  numbers set the margins of section 4 only, since they were measured under other settings: G1 with shape pooling per
  take and 64-window batches, G3 with pooling per trial and 32-window batches).
- Declaration: a JSON file written before any comparison number is computed, its SHA-256 carried by every output. It
  names this rule's SHA-256; for each model and seed the checkpoint path and SHA-256, its config_sha256, its
  training-person list and that list's SHA-256, its body backend (name, version, joint map SHA-256) and its records
  state; the person lists of every set of section 3 with their SHA-256; the settings of section 3; the code SHA-256 of
  each evaluation script and of scripts/open_port_comparison.py; the MoveAtlas commit. Each model's training persons
  are asserted disjoint from every scored set before any number is computed. Outputs: reports/open_port_comparison/
  (seed1_phaseA/, seed1/, three_seeds/), each with its _provenance.json.

## 2. Training alarms and run integrity (R0)

The alarms the rule relies on are those of mdfm/train/loop.py (ALARMS and the module docstring, read at MD_FM 4769987),
with schedule.stop_on_alarm true and schedule.alarm_consecutive empty (every alarm on one read), as in
configs/stage1_pilot.yaml (stop_on_alarm set true, alarm_consecutive left at its empty default):
- at a log record: attention entropy below 0.5 nat in any decoder block from step 500 of the phase; mean absolute beta
  error above 2 from step 1,000; mean logvar_z below -9 from step 1,000;
- at a dev event: spread_ratio below 0.3 and z0_margin_mm below 1 mm from step 2,000; root_speed_ratio below 0.5 and
  heading_gain below 0.3 from step 5,000;
- jepa_collapse and unit_nll_rise belong to JEPA and stage-2 runs and do not apply.
The beta alarm reads the 16 SMPL+H betas, which the open body does not have. Its open counterpart is the same alarm
(beta_error, the mean absolute error of the posterior mean against the truth on truth windows, line 2.0, from step
1,000) on the open model's shape vector, the six phenotype latents u of mdfm/body/openbody.py (mdfm/model/mfm.py, Open
body): the code computes it on whatever shape vector the model carries. Reason for keeping the line: the per-dimension
standard deviations of the open phenotype prior on u (reports/open_body_priors/priors.json, pooled row, 397 persons)
run from 0.38 to 1.39, those of v1's beta prior (reports/body_store/priors.json, neutral row) from 0.82 to 1.59, so a
mean error above 2 lies beyond the prior's spread on both bodies. The dev-event alarms read the dev hook's numbers,
which an open run computes on the dev folds of the open body store (F:/MD_FM/bodies_open_v1_dev, built by
scripts/build_open_body_store.py --split dev, reports/open_body_store/store_dev.json; plan step M4); an open run
without its dev hook (--no-dev) fails R0. Besides the alarms, the loop stops on
MemoryBudgetError (schedule.max_reserved_gb, 14.5 GB in configs/stage1_pilot.yaml), SyntheticCapError
(mixture.synthetic_cap 0.60) and FloatingPointError after MAX_NONFINITE = 3 skipped optimizer steps in a row.

R0 holds for a run when its log.jsonl and reports/gpu_queue/job_log.csv show: every 'start' event at the declared
config SHA-256; no 'alarm' event in phase A or phase B; no stop of the three kinds above; a phase B 'verify' event at
step 32,044; and the last job_log row of its training job exited 0 (a crash resumed by the queue is allowed when the
rest holds). R0 for phase A alone reads the same up to the phase A end.

## 3. The lines, their sets and settings

Each line is computed for both models of a seed on the same persons, windows and inputs. Direction: the orientation
of the difference d = open minus v1 for a lower-is-better metric, v1 minus open for a higher-is-better one, so that a
positive d means the open model is worse.

| id | line | rule and set | metric | better | role |
|---|---|---|---|---|---|
| L1 | G1b coverage | G1 open version, development half (gait120 10, stroke patients 20) | coverage of the left wrist at 0.90, offset-corrected (section 5) | higher | deciding |
| L2 | G1b energy score | same | energy score of the 64 wrist draws (mm), offset-corrected | lower | deciding |
| L3 | G1c | same | PA-MPJPE, 15 canonical joints (mm), offset-corrected | lower | deciding |
| L4 | G3 stride | G3 2026-11-12, the 12 valid dev controls (primary row) | stride_mae_cm | lower | deciding |
| L5 | G3 speed | same | speed_mae_m_s | lower | deciding |
| L6 | G3 IC | same | ic_median_ms, walkway reference | lower | deciding |
| L7 | G4 GRF | G4 2026-11-12, the 38 scored dev stroke patients | grf_r_gain | higher | deciding |
| L8 | G4 EMG | same rule, the 18 scored gait120 dev persons | emg_r | higher | deciding |
| L9 | G4 metric3d | same rule, the 38 dev stroke patients | PA-MPJPE (mm) | lower | deciding |
| L10 | G8 | G8 2026-11-12, pd_dev (14 PD persons), fitted on pd_train | mean QWK, round, chosen variant, the probe on frozen tokens | higher | deciding |
| L11 | G12 readout, stride | G12 2026-11-12 representation row, dev12 | stride_mae_cm of the token probe, chosen variant | lower | deciding |
| L12 | G12 readout, speed | same | speed_mae_m_s of the token probe | lower | deciding |
| L13 | G12 readout, IC | same | ic_median_ms of the token probe | lower | deciding |

Reported beside, deciding nothing: G1a and the G1b width ratio on the development half; L1 to L3 without the offset
correction; the G1 per-source rows; G3 on the 14 (all trials), the ZUPT rows (no model) and stride_mape_pct; G4
grf_rmse, emg_r_gain and metric3d MPJPE; the G8 qmap, all_dev and pd_pool rows; the G12 readouts on the 14; and, once
seeds 2 and 3 exist, the open G12 baselines against v1's seed by seed (their training data changed with the body
store, so they measure the port of the arms, not a line of this rule).

Why these and not others. grf_r_gain decides and grf_r does not: the Newton floor is common to both models, so the
difference of the gains equals the difference of the mean correlations, and grf_r_gain is the G4 claim (G4 dated rule
section 2). emg_r decides and emg_r_gain is reported: the template is common to both models, so the gains differ
from emg_r only through the median. grf_rmse kills for every model and is not a claim of the program (same section).

Settings, identical for both models:
- G1 and G3: the mode ema_s20_nopool (section 2 of the G1 and G3 dated versions of 2026-11-12); G1 inputs, truth and
  conformal factors as docs/rules/G1_RULE_OPEN_2026-11-13.md sections 2 to 4, the factors fitted per model on the
  calibration half.
- G4: scripts/eval_g4_dev.py at the settings the v1 reports of section 4 record (EMA weights, 20 refinement steps of
  256 points, 10 s windows every 5 s, shape pooling per recording, 8-window batches, seed 20260929).
- G8 and the G12 readouts: tokenizer and encoder only, EMA weights, the token pass on the GPU with bf16 autocast (G8
  dated rule section 2, item 9); the variant chosen on the fitting set for each model (G8 draft section 4; G12 revision
  item 8).

## 4. Margins

The form, for every deciding line: M = max(0.05 x |V|, 2 x R), V v1's value of the line on the comparison set (v1
seed 20260928's at the seed-1 reading, the mean of v1's three seeds at the three-seed reading), R the range of v1's
measured per-seed values of the line. It is the non-inferiority form of reports/bench/f2_lever2/rule.txt section 7
(max(0.05 x v1's line, 2 x d), hashed before its run), with the range over v1's seeds in place of a two-seed
difference. The 2 x R term keeps a reseed of v1 from failing the line; the 5 percent floor holds where v1's seeds agree
more closely than any change of body could. For the G1 lines the sets change from AMASS to the open sets, so R is
taken relative to v1's mean on AMASS and applied to V (M = max(0.05, 2 x R / mean) x |V|); for the other lines the sets
are the ones measured, so R is absolute. A comparison of a point with a margin uses a tolerance of 1e-9 in the line's
unit.

L1, G1b coverage. Measured: 0.92927, 0.90557, 0.92382 (seeds 20260928, 20261001, 20261013; line "G1b single wrist
coverage" of reports/gates/g1/dev_phaseA_root_full_ckpt_B_032044/g1.json, dev_pilot_v1_s2_ckpt_B_032044/g1.json,
dev_pilot_v1_s3_ckpt_B_032044/g1.json; AMASS dev, 23 persons). Range 0.02370, mean 0.91955, 2 x range / mean 0.0516.
Margin: M = 0.0516 x V.

L2, G1b energy score. Measured: 125.215, 124.502, 122.538 mm (secondary row "G1b energy score", all windows, same
three files). Range 2.677 mm, mean 124.085, 2 x range / mean 0.0431, under the floor. Margin: M = 0.05 x V.

L3, G1c. Measured: 42.092, 41.637, 40.856 mm (line "G1c single camera PA-MPJPE", same three files). Range 1.236 mm,
mean 41.528, 2 x range / mean 0.0595. Margin: M = 0.0595 x V.

L4, G3 stride. Measured on the 12: 19.6878, 19.6925, 19.6663 cm (reports/bench/g12/body_seed_points.json, from
reports/gates/g3/dev_phaseA_root_full_ckpt_B_032044_bw32_now, dev_pilot_v1_s2_ckpt_B_032044_bw32 and
dev_pilot_v1_s3_ckpt_B_032044). Range 0.0262 cm, 2 x range 0.0524; 0.05 x mean 0.984. Margin: M = 0.05 x V (0.98 cm
at v1's measured mean). The 14's range is wider (26.51, 27.51, 25.38 cm, same file), from the two controls whose
inputs fail the G3 validity check; the 14 are reported only.

L5, G3 speed. Measured on the 12: 0.192856, 0.195439, 0.195607 m/s (same file). Range 0.00275, 2 x range 0.00550;
0.05 x mean 0.00973. Margin: M = 0.05 x V (0.0097 m/s at v1's measured mean).

L6, G3 IC. Measured on the 12: 13.333 and 12.500 ms for seeds 20260928 and 20261001 (rows dev12 ic_median_ms of the
body, reports/bench/g12/compare_g3_dev/g12.json and g12.md). No report holds seed 20261013's IC on the 12; on the 14
the three seeds read 14.286, 14.286 and 13.571 ms (ic_median_ms_walkway of each g3_dev.json named under L4), range
0.714. R taken: 0.833 ms, the larger of the two-seed range on the 12 and the three-seed range on the 14. The line is
the mean over the 12 persons of per-person medians on a 10 ms grid, so 0.833 ms is one person's median moving one
step. Margin: M = max(0.05 x V, 1.667 ms), that is 1.667 ms at v1's measured values (two persons, one step each).

L7, G4 GRF. Measured: 0.061869, 0.051941, 0.066476 (line grf_r_gain of reports/gates/g4/
dev_phaseA_root_full_ckpt_B_032044_g4h/g4.json, dev_pilot_v1_s2_ckpt_B_032044_g4h/g4.json and
dev_pilot_v1_s3_ckpt_B_032044/g4.json, all three under the G4H harness: the first two rerun under it on 2026-10-07,
the third run on 2026-10-18). Range 0.01454, 2 x range 0.02907; 0.05 x mean 0.0030. Margin: M = 0.0291. Holding this
line does not hold the G4 claim: the open model's own grf_r_gain against the 0.03 line is the G4 rule's business.

L8, G4 EMG. Measured: 0.722662, 0.720697, 0.714921 (line emg_r, same three files). Range 0.00774, 2 x range 0.01548;
0.05 x mean 0.0360. Margin: M = 0.05 x V (0.036 at v1's measured mean).

L9, G4 metric3d. Measured: 32.106, 33.811, 32.533 mm (metric3d pa_mpjpe_mm, same three files). Range 1.705 mm, 2 x
range 3.409; 0.05 x mean 1.641. Margin: M = 3.41 mm (unless 0.05 x V exceeds it). This line's truth is the 15
canonical joints of the marker records, neither body's joints: it is the one pose line of the comparison that no body
definition favours.

L10, G8. Measured: the body 0.221834 (seed 20260928, variant measurement) and 0.238696 (seed 20261001, variant latent)
(reports/gates/g8/dev_phaseA_root_full_ckpt_B_032044_gpu/g8_dev.json and dev_pilot_v1_s2_ckpt_B_032044_gpu/g8_dev.json,
lines pd_dev round, chosen); no third-seed G8 readout exists. The body's range is 0.01686 from two seeds. On the same
line, persons and probe, the two G12 arms read MFR 0.196964 and 0.150194 (range 0.04677) and JEPA 0.115385 and 0.224831
(range 0.10945) (reports/gates/g8/dev_g12_mfr_s20260928_ckpt_B_056774_gpu, dev_g12_mfr_s20261001_ckpt_B_056774_gpu,
dev_g12_jepa_s20260928_ckpt_B_047617_gpu, dev_g12_jepa_s20261001_ckpt_B_047617_gpu). Fallback: R = 0.04677, the
median of the three arms' two-seed ranges. Reason: one two-seed range is a single draw, the chosen variant changes
between the body's two seeds, and the other arms show three to six times the body's range on these 14 persons; a
margin from the body's range alone (0.034) would sit inside the seed noise of a difference of models. Margin: M =
max(0.05 x V, 0.0935), that is 0.0935 at v1's measured values. On dev this line detects a collapse only; the G8
lockbox read on the open model decides G8.

L11, G12 readout, stride. Measured on the 12: 7.94822, 8.55378, 6.96329 cm (lines dev12 stride_mae_cm of
reports/bench/g12/readouts/phaseA_root_full_ckpt_B_032044/results.json, pilot_v1_s2_ckpt_B_032044/results.json,
pilot_v1_s3_ckpt_B_032044/results.json; the representation row of reports/bench/g12/compare_g3_dev/g12.md). Range
1.5905, 2 x range 3.181; 0.05 x mean 0.391. Margin: M = 3.18 cm. The MFR arm's three seeds spread alike on the same
probe (6.43, 7.92, 7.17 cm, same g12.md).

L12, G12 readout, speed. Measured: 0.056425, 0.072454, 0.058287 m/s (same files). Range 0.016029, 2 x range 0.03206;
0.05 x mean 0.0031. Margin: M = 0.0321 m/s.

L13, G12 readout, IC. Measured: 24.167, 22.500, 21.667 ms (same files, dev12 ic_median_ms). Range 2.500, 2 x range
5.000; 0.05 x mean 1.139. Margin: M = 5.0 ms.

## 5. The offset correction of the G1 lines

The G1 truth on the open sets is Anny's joints (G1 open version, section 2), the open model's own joint definitions;
v1 predicts SMPL+H's, which differ by up to 5.48 cm per metre of stature at a rest joint
(reports/openbody/first_comparison.json, right_collar). Scored raw, v1 would lose on definitions and the comparison
would favour the open model. So L1 to L3 are scored after a correction fitted on the calibration half, the same for
both models: for model m and joint j of the line (the 15 canonical joints for L3, the left wrist, program joint 20,
for L1 and L2), b_mj is the person-weighted mean over the calibration half's knots of truth minus prediction (for the
wrist, minus the mean of the 64 draws), in the frame the line scores (pelvis-relative, the truth's pelvis-heading frame
at the knot; for the wrist, each draw in its own body's frame, draft section 5). On the development half b_mj is added
to every predicted position of joint j (to every draw for the wrist) before scoring; the conformal factor of L1 is
fitted after the correction. The raw rows are reported beside.

A second tilt is not removed: the inputs are rendered from Anny bodies at the open body's sensor sites, the open
model's training domain and not v1's. A G1 line that holds therefore reads "holds, tilted toward the open model"; a G1
line that fails stands as a loss.

## 6. The decision on seed 1 alone, and the stop condition

Pre-flight result known before this rule is registered (it decides nothing here and moves no margin): the
learnability gate's turning batch on configs/stage1_open.yaml failed its root line at its last step, 0.1295 m against
0.10 (root yaw 2.77 deg against 5, passed), with the root's median over the last six readings 0.143 m against v1's
0.136 m and the yaw's 3.12 deg against v1's 3.34 deg, on 14 windows of 14 persons
(reports/bench_open/learnability_turns_walks_rest10.md, reports/bench_open/learnability_reading.md). M6 seed 1 starts
with that line recorded as failed; this section's stop condition is the binding check.

Early read at the phase A end of seed 20260928 (checkpoint ckpt_A_010681): R0 for phase A, and the G3 simulator-only
dev estimate (stride_mape_pct of the phase A model on the 14 dev controls, G3 draft section 8). STOP when phase A
fails R0 or the estimate exceeds 40 percent, the value at which G3 draft section 8 stops the program on the lockbox
(v1's phase A end read 21.07 percent, G3 dated rule section 5). Otherwise phase B runs.

Seed-1 reading, after the phase B end of seed 20260928 and its probes, with v1 seed 20260928 rerun under section 3. For
each deciding line, d_1 and its paired interval: persons resampled with replacement, the same persons for both models
in each resample, each model's line recomputed on the resample by the gate's own statistic, the oriented difference
taken; 2,000 resamples; the 2.5 and 97.5 percentiles; numpy.random.default_rng seeded with the gate's bootstrap seed
(20260929 for G1, G3, G4 and the G12 readouts; 20261015 for G8). For L1 to L3 the conformal factor and the offsets
stay those fitted on the calibration half. Status of the line:
- HOLDS when d_1 is at most M;
- EXCEEDS when d_1 is above M;
- CLEAR LOSS when d_1 is above 2 x M and the lower bound of its interval is above M.

STOP: M6 holds seeds 20261001 and 20261013 and the six G12 baseline runs when any of these holds:
- S1: the seed-1 run fails R0 (a training alarm or another stop of section 2);
- S2: a deciding line reads CLEAR LOSS;
- S3: seven or more of the 13 deciding lines read EXCEEDS;
- S4: a deciding line that v1 reads cannot be read on the open model because the open model lacks the head or output
  it needs (a capability the port lost).
GO otherwise: the two other seeds and the six baseline runs are queued (decision P9). A line that cannot be computed
for another reason (a failed job, a missing input) leaves the reading NOT EVALUATED, and nothing further is queued
until it is computed.

After a STOP: the reading is written with its cause; the run and its numbers stay in the record and are reported with
any later result. A change enters only through a dated amendment of configs/stage1_open.yaml or of the code, written
with its reason before a new seed-1 run, which this rule reads again unchanged (same margins, same stop). The
attempts are counted; the three-seed reading uses the last attempt's seed 1 and prints the number of attempts beside
its verdict, since each attempt has looked at the development sets.

## 7. The decision on three seeds

For each deciding line, D = the mean over the three seeds of d_s, V the mean of v1's three seeds. The line HOLDS when
D is at most M and FAILS otherwise. Reported beside: each seed's d_s; the paired interval of D (as in section 6, each
resample drawing the same persons for all six checkpoints); "better" when the interval's upper bound is below 0.

Verdict:
- NON-INFERIOR: every deciding line holds;
- INFERIOR ON <lines>: one or more deciding lines fail, each named with its D, M and interval;
- NOT EVALUATED: a deciding line lacks a seed's row on either model; the output names it.
G1 lines that hold carry the label of section 5.

What the verdict does: it is the port's measured cost on development data, written to the handoff and beside any text
that reports the open model, with every per-seed row. It decides no gate line and no lockbox read, and the lockbox
reads stay on the open model only (question Q2 of the plan, recommended yes). An INFERIOR line is reported as a loss of
the port on that line, not explained away.

## 8. What this rule never writes

A gate pass or kill from a dev number; a v1 number of an earlier report as a comparison row; a margin, line or set
moved after an open number exists; a comparison across seeds; a lockbox read of v1.

## 9. Cost on the GPU

Seed 1 alone, phase A, phase B and its probes: 25.85 h projected, against 209.76 h for the whole of M6
(reports/model_open_port/summary.md and inventory.json, gpu_projection, on v1's measured step cost; the bench on the
open body replaces it before the queue is filled, plan section 4.1). The v1 reruns of section 3 add one probe set per
v1 seed, 1.384 h at the measured mean (same file). The G1 fits of the development and calibration halves run on the
CPU.

## 10. Note for the G12 declaration of the port

docs/rules/G12_RULE_2026-11-12.md (registered as G12 version 1) admits the port's retrained body and baselines as a
second declaration (its section 1) and names the baselines' configuration as configs/g12_baseline.yaml with
base_config_sha256 equal to the declared body configuration's SHA-256. The port's baselines therefore train from that
path with base_config set to configs/stage1_open.yaml and base_config_sha256 to its SHA-256 (section 1 above); the v1
arms keep, in their checkpoints and in git history, the SHA-256 of the version they trained under
(base_config_sha256 7f761b1ac425a2dfefc4e546908fa766e2545d915f9c2a88dcb8d199f140b2b5 in the file as of MD_FM 4769987).
No new G12 version is needed for this.
