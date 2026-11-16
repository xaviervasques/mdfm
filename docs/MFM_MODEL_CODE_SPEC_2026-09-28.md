# MD_FM stage-1 model code: the 60M pilot (DRAFT)

Date: 2026-09-28. Status: DRAFT for the owner and the builders, reviewed the same day against
the code on disk, the reports and the MoveAtlas contracts (section "Review against the code").
Nothing in this document is built yet. It specifies plan task B4 (docs/EXECUTION_PLAN.md,
section 3.3) on top of the code that exists under mdfm/ (the B1 core and the B4.0 canonical
reader). English, no em-dashes.

Read for this spec: docs/MFM_DESIGN_2026-09-24.md sections 0 to 5, 9, 11, 12 and 13;
docs/MFM_DESIGN_NOTE_ARCHITECTURE_2026-09-24.md; docs/EXECUTION_PLAN.md; mdfm/ and
scripts/bench_encoder.py; reports/bench, amass_walks, g2_renderer, weargait_targets,
weargait_frames, record_index, lockbox, manifest_tiers and person_ids; the MoveAtlas contracts
canonical_imu, canonical_skeleton, canonical_force and canonical_emg (read-only); three
candidate designs and the scores of three judges.

## How numbers are labelled

- [D§n], [A§n], [P]: the design, the architecture note, the execution plan.
- [R path]: a report on disk under reports/.
- [probe B4.n]: counted read-only while drafting or reviewing (2026-09-28) from files on disk:
  the CSV reports, the manifest index, record headers and a few records opened through
  mdfm.data.canonical. It is not yet in reports/. Task B4.n writes it there; until then it is
  a planning figure.
- [scratch B4.3]: a GPU measurement made in a scratch run during the design and reproduced by
  two judges. It is not on disk; task B4.3 writes reports/bench/bench_kinematics.md.
- [est]: planning estimate. The task named beside it replaces it with a measurement.
- [cfg]: configuration value (loss weight, noise floor, mixture share). It is an initial value,
  changed on dev folds only, and it enters the config hash.

## How this spec was made

Three designs were scored by three judges on fidelity, feasibility, testability and simplicity
(0 to 10 each). Totals over the judges: minimal 88.5 (28.0, 30.5, 30.0), throughput 86.0
(26.5, 31.0, 28.5), faithful 77.0 (28.5, 24.0, 24.5). This spec takes "minimal" as the base:
harnesses and trivial baselines first, the smallest code that can read the pilot gates. It
grafts the ideas the judges named in the other two and fixes the flaws they found.

| from | ideas kept |
|---|---|
| minimal (base) | gate harnesses dry-run on dev with trivial baselines before any model code; joint-space shape basis; treadmill belt frame; AMASS take index; gait120 steps 01 and 02 concatenated, with the EMG clock mapped exactly through gait_events; simulator-only G3 dev estimate on the non-lockbox controls; constant offsets fitted on dev persons only; split conformal on whole-archive calibration folds; per-batch guard; closed-form per-record EMG log gain; heading frame at the window centre |
| faithful | lockbox archives derived from the CSVs and refused inside mdfm/data/amass.py before tarfile.open; *_stageii.npz members listed; CNRS and WEIZMANN excluded as SMPL-X fits; the four test files that read ACCAD Female1 moved; allow_lockbox=True confined to mdfm/gates by a grep test; gender-routed skeleton; time-query decoder without query-query attention, float32 output path, velocity and acceleration supervision; likelihood scaled so that likelihood and KL keep ELBO proportions; autoregressive latent prior with free bits; stopped-gradient refinement with straight-through scoring and an amortization term; nuisance consistency on paired renders; HSIC in strata; raw root tracks for G4; mount priors from measured frames; label strip; the G1 protocol; cumulative synthetic-share accounting; EMG kernel with inverse-dynamics torques and monotone moment-arm maps |
| throughput | rsqrt Gram-Schmidt, so that backward through nested jvp works; rest joints once per subject; one second-order jvp per micro-step over sampled contiguous segments with importance weights; no-grad GPU rendering in macro-batches; physionet_multimodal_gait left out; AMASS rows selected from canonical.index only; per-source unit probe; exact replica of the MoveAtlas pose normalization; kernels refuse half precision; G1b scored in each body's own pelvis-heading frame; analysis-by-synthesis reference row; import-graph test; second lockbox read refused; CUDA graphs as a measured contingency |

| flaw found by the judges | in | fixed here |
|---|---|---|
| trains on CNRS and WEIZMANN, SMPL-X fits (165-dim poses) posed with SMPL+H | minimal, throughput | excluded by surface_model (reports/person_ids/amass_folders.csv) |
| guard only in an index module; amass.py unguarded; tests read a lockbox person | minimal, throughput | refusal inside amass.py; all four test files moved (B4.1) |
| typed mask, nuisance tokens, continuous-time positions, native-rate stems dropped | minimal | kept (B4.14) |
| no motion prior or KL, while single-sensor sources train the kernel likelihood | minimal, throughput | latent autoregressive prior with KL (B4.15); capture24 and ltmm deferred |
| 25 knots per token from an MLP: jumps at patch borders, amplified 1e4 | throughput | decoder is one smooth function of t, plus velocity and acceleration supervision |
| no refinement during training (architecture note section 4 misread) | minimal, throughput | stopped-gradient refinement in phase B, cadence set by the bench |
| backward through nested jvp untested until late | faithful | gradcheck in B4.2, the second task |
| physionet_multimodal_gait fed to the sensor-frame kernel | faithful | left out |
| no treadmill handling | faithful, throughput | belt frame in B4.4 |
| 2 s minimum window drops most gait120 and stroke records | faithful | 1 s minimum; gait120 steps concatenated |
| no conformal step in the pilot | faithful | split conformal on calibration archives (D§4 heads) |
| scope of about 40 modules and 29 steps | faithful | G12 moved to B5; soft-tissue filter, multistart mounts, joint status, capture24 and ltmm out; force and EMG are targets only |
| force kernel absent from training; stroke G4 and G5 moved out of the pilot | minimal | force and EMG kernels in phase B; G4 and G5 are B4 tasks |
| IMU likelihood band 20 Hz | minimal | 40 Hz, as D§3 |
| EMG from a black-box convolution on kinematics | minimal | D§3 kernel |
| lockbox hash f94df2ff (v1.1) typed into code | faithful, throughput | hashes read from reports at run time; lockbox v1.2 (619dfdac...) |
| G2 quoted as 0.990 or 0.998 | faithful, throughput | version 4: 0.984 on all 13 sites, 0.9995 with the residual spectrum [R g2_renderer/results.json] |
| val and calibration folds carved by id | throughput | whole archives |
| time-warp augmentation breaks gravity-consistent dynamics | throughput | no time warp |
| truth knots stored in float16 (found while drafting) | faithful | float32 (section 2.3) |
| about 45M parameters, not the 60M pilot | minimal | about 59M [est] |

## Review against the code (2026-09-28)

Checked read-only: every API the spec calls in mdfm/ (names, signatures, units, frames), the
numbers against reports/ and the manifest index, the rules against the data flow, the memory
budget against reports/bench/, and each build task for a test that runs alone. Confirmed
without change: the AMASS split counts and hours of section 1.3, the cache contents of
B4.1, the fps histogram of section 3, the S002 gait120 frames, the token counts per layout,
the canary cody_canary:mdfm-v1, the parameter sum, and the in-place failure of the current
rot6d_to_matrix under nested jvp (reproduced on CPU with torch 2.11.0+cu128; the rsqrt form
passes gradcheck in float64). Corrected here:

| finding | evidence | fix, section |
|---|---|---|
| G2 moved to version 4 after drafting; summary.md prints the with-residual value on the line of the without-residual run | results.json: all_sites 0.984 [0.9607, 0.9994], all_sites_with_residual 0.9995 [0.9985, 1.0]; scripts/g2_renderer_realism.py writes `ba` (with residual) on both lines | numbers updated; the report bug is an open item (1.1, 8) |
| stroke canonical 3D records are z-up; gait120 and amass are y-up | the median neck-minus-ankle vector points along +z on 30 of 30 random non-lockbox stroke records (23 able-bodied, 7 stroke); median neck y 0.99 torso units on a gait120 and a KIT record | fixed per-source up rotation in sources.py; the yaw profile of pose3d_nll cannot absorb it (2.0, 2.2, B4.5) |
| gait120 steps share a frame and carry their own scale | S002 trial 1: step01 108 frames (468 to 575), step02 105 (575 to 679), scales 0.5071 and 0.5056 | un-normalize each step, drop the duplicate frame (2.2) |
| loaded records carry manifest fields (group, tier, lockbox) and a full "manifest" dict; stroke pose adds label, group, analog_labels | mdfm.data.canonical._finish; records opened | windows.cut keeps a whitelist (2.2) |
| sdpa_kernel ignores the list order unless set_priority=True | signature on torch 2.11: (backends, set_priority=False) | set_priority=True and a profiler check (2.5) |
| grep tests would trip on tests/ | tests/test_canonical.py and tests/test_tiers.py pass allow_lockbox=True or allow_eval=True inside refusal and reader tests | grep scope is mdfm/ and scripts/ (B4.1, B4.22) |
| test_body.py opens the lockbox container amass_ACCAD.npz with np.load, outside the reader | tests/test_body.py line 68 | B4.1 replaces it with canonical.load on a KIT record (B4.1, risk 6) |
| amass cache files do not record the body model; weargait target CSVs hold lockbox rows | data/amass.py _store keeps poses, trans, betas, gender, fps; events.csv and insole_events.csv carry a lockbox column | _store writes surface_model; the G3 harness filters on the column (2.2, 2.7, risk 6) |
| short real records could pack about 1,600 one-second windows into a micro-step, beyond the point and knot budgets, and a window could lose all its kernel segments | 1 s windows make about 10 tokens; weights n_valid / n_sampled | at most 512 windows per micro-step; one segment per stream-window always kept (2.2, 5) |
| stroke force totals: plate-to-foot mapping and unloaded offset unmeasured | one able-bodied force record reads above 1 N on foot_L in every sample, peak 544 N | B4.5 probe item (2.2, risk 11) |
| per-source cap of 15 percent of real tokens is infeasible with six real sources | 6 x 15 = 90 percent | 15 percent of all tokens per real source, as D§5 (2.2) |
| phase A is 100 percent synthetic, so the 60 percent cap holds only cumulatively | section 5 arithmetic | stated; question 1 (5, 9) |
| refinement cost was amortized per micro-step, not per optimizer step | 2 micro-steps per optimizer step | phase B rate and GPU-hours recomputed (5) |
| encoder activations were read from the bench peak without removing what the bench kept resident | bench_mfu keeps twelve 4 x 4,096^2 bool masks (0.81 GB) and 37.8M-parameter AdamW states (0.61 GB) on the GPU | about 5.4 GB; card is 16,303 MiB with the display under WDDM (5) |
| stems and synthetic streams: the pilot's inputs are band copies, synthetic renders were not | window_store decimates to about 100 Hz; synth rendered at the layout rate | one band_copy function for real, synthetic and G1 inputs (2.2, 2.4, 6) |
| pose2d_nll mixed a 3 px sigma with torso units | kernels/camera.keypoint_nll is in pixels; the 2D contract has no confidence field | residual in pixels against obs x scale (2.3) |
| G1 self-test expected 47.9 mm for 30 mm joint noise under a pelvis-relative score | pelvis-relative error of 21 joints | 45.7 mm when noise is added to pelvis-relative positions (6) |
| refinement and Newton-floor tests needed a trained model | B4.13 and B4.15 run before any training | body held at truth; com15 on canonical joints (2.3, 2.5, 7) |
| G3-sim dev estimate "on controls none seen by phase A" | mount priors and G2 device noise come from non-lockbox controls | clean estimate on the 14 dev controls, drawn from the G2 TEST half (2.2, B4.19) |
| stroke EMG flag quoted as one value | 513 of 529 non-lockbox stroke EMG records carry the flag, record p99 14.5 to 991 mV, median 258 | numbers replaced (1.5, 8) |
| vidimu scale quoted from the judges | 40 random vidimu records: scale 437.5 to 544.0, median 516.6, modality 3d_meter | replaced (1.3) |

## 1. Scope of the pilot and deferrals

### 1.1 Goal and pass set

D§13, days 30 to 60: stage 1 as the 60M pilot on the local RTX 5080; pass G1, G4, the healthy
half of G3 and the G5 preservation check before any cluster time; start the G12 baseline. The
plan splits this into B4 (this spec) and B5 (the G12 baseline and the backbone ablation, which
reuse the pilot's sampler, tokenizer, encoder and measured step so that "equal compute" is
exact). The pilot's G1 MPJPE and G3 stride MAE are also the 60M reference of G13: the 200M
model must improve them by 10 percent and 1 cm, or the program stays at 60M [D§9].

| gate | set | lines [D§9] |
|---|---|---|
| G1 | 53 amass lockbox persons, whole archives ACCAD, DFaust, GRAB, MoSh, PosePrior, SOMA, SSM, TCDHands, TotalCapture, Transitions: 74 subject ids, 2,141 motions [probe B4.1]; by group ACCAD 9, GRAB 10, MPI 25, SOMA 2, TCDHands 1, TotalCapture 5, Transitions 1 [R lockbox/summary.md] | six IMUs MPJPE at most 45 mm, kill above 70; single wrist 90 percent coverage at least 0.85 with the sharpness bound, kill below 0.70; single camera PA-MPJPE at most 60 mm, kill above 90 |
| G3, healthy half | weargait_pd lockbox controls: 17 in lockbox v1.2 [R lockbox/summary.md]; 13 IMUs only | stride MAE at most 5 cm, kill above 10; speed MAE at most 0.08 m/s, kill above 0.15; initial contact median error at most 30 ms, kill above 50 or if not 20 percent better than a shank-gyroscope detector. Simulator-only variant first: stride MAPE at most 25 percent, kill above 40, which stops everything |
| G4 | stroke lockbox, 10 stroke persons [R]; gait120 lockbox, 24 persons [R] | per-foot vertical GRF from pose only, RMSE at most 10 percent body weight, kill above 20, and at least 0.03 above the Newton baseline in correlation; stroke metric-3D round trip reported; gait120 median log-envelope correlation at least 0.6 over four leg muscles, kill below 0.4 |
| G5 | care_pd, EVAL, read inside the harness only | UPDRS_GAIT probe within 0.05 QWK of the probe on source bodies, kill if the drop exceeds 0.10; BMCLab ON versus OFF separability within 5 AUROC points, kill above 10; PA-MPJPE as a diagnostic |

G2 is a precondition, not a pilot gate. D§9 places it "before any pretraining". Its first half,
version 4 (lockbox persons excluded, measured device noise), reads balanced accuracy 0.984
(95 percent interval 0.9607 to 0.9994) on all 13 sites and 0.9995 with the 5 to 50 Hz
residual spectrum added, against a kill line of 0.85 [R g2_renderer/results.json]; the second
line of summary.md prints 0.9995 for the run without the residual spectrum, a report bug
(the script writes the with-residual value on both lines). Version 3 read 0.9927
[R g2_renderer/v3_sign_fix/summary.md]. The discriminator decision OD9 is open [P]. This spec
builds everything that does not train (B4.1 to B4.18) now; the phase A run (B4.19) waits on
the owner's answer to question 1 of section 9.

### 1.2 The pilot in one paragraph

Each 10 s window of native samples becomes one token sequence: 40 latent body tokens, one
subject token, one measurement token per channel group per 0.25 s patch, and one nuisance
token per IMU site, camera or mocap stream. A typed mask keeps body positions from attending to
nuisance tokens. A 12 x 512 dense encoder (the configuration measured in reports/bench/) turns
the latent tokens into a Gaussian posterior over a 32-dimensional latent per patch, and the
subject token into a posterior over the 16 shape coefficients. An autoregressive Gaussian prior
over the latent patches (2 s span), conditioned on sex, task and shape, is the healthy motion
prior of stage 1. A time-query decoder turns a latent sample into knots of the 100 Hz body clock
(D§1). Cubic Hermite interpolation of the knots gives pose, velocity and acceleration at any
native timestamp, and the kernels of D§3 turn them into predicted IMU, 2D pose, 3D pose, force
and EMG. Loss: negative log-likelihood of every observed stream through its kernel, direct
supervision where the body is given (amass), KL to the prior, nuisance priors and consistency,
plausibility, subject consistency and HSIC. Inference: the amortized posterior refined by a few
stopped-gradient steps through the kernels (5 in training, 20 at evaluation [D§4]).

| part | design size [D§4] | pilot size | label |
|---|---|---|---|
| tokenizers (stems, embeddings) | about 15M | about 3.5M | [est] |
| local encoder | 85M (12 x 768) | 37,829,632 (12 x 512, 8 heads) | [R bench/bench_encoder.md] |
| body decoder | about 40M with the flow head | about 12.7M (4 cross-attention layers, width 512) | [est] |
| flow head (latent prior) | inside the decoder budget | about 3.2M (4 causal layers, width 256; an autoregressive Gaussian, departure 6) | [est] |
| session aggregator | about 25M | not built (stage 4) | |
| heads | about 10M | about 2M: posterior and shape heads, nuisance heads, ForceHead, EMG decoder, source offsets | [est] |
| total | about 200M | about 59M; B4.15 asserts 57M to 62M | [est] |

### 1.3 Data in the pilot

RELEASE tier only, so that G13 compares the 60M and 200M RELEASE models on the same tier (the
INTERNAL 60M run is B6 [P]). EVAL, EXCLUDED_V1, CODY and lockbox rows are refused by the
readers before any container opens. Healthy persons only.

Bodies with truth: AMASS SMPL+H fits, joints recomputed from the raw npz and never read from
the canonical records [D§5]. Splits by whole archive:

| split | archives | persons | motions | hours |
|---|---|---|---|---|
| lockbox, read by gates only | the 10 archives of section 1.1 | 53 | 2,141 | fps of GRAB and SOMA read from the raw npz at B4.21 |
| excluded, SMPL-X fits | CNRS, WEIZMANN | 7 | 1,505 | manifest fps blank |
| dev (proposal) | EyesJapanDataset, HDM05, SFU | 23 | 1,009 | 8.72 |
| calibration (proposal) | DanceDB, EKUT, HUMAN4D, HumanEva | 27 | 674 | 5.33 |
| train | BMLhandball, BMLmovi, BMLrub, CMU, KIT | 360 | 11,825 | 33.45 |

All five rows are [probe B4.1], from reports/person_ids/amass_folders.csv,
reports/lockbox/lockbox_persons.csv and the manifest index. No SMPL-X model is on disk
(body_models holds smpl and smplh). BMLrub holds 412 takes named "treadmill" [probe B4.4]. The
four near-identical take pairs of T0.7c (CMU_105 and CMU_91, CMU_122 and CMU_83, KIT_1717 and
KIT_1747, KIT_575 and KIT_576) [P] all sit inside train archives.

Real healthy partial-state sources, non-lockbox:

| source | streams | rate | persons | hours | why it is in |
|---|---|---|---|---|---|
| weargait_pd controls | 13 Xsens IMUs, acc and gyro (magnetometer masked: normalized units) as input; insole totals as target | 100 Hz | 68, of which 14 held out as the G3 dev fold [R weargait_targets/summary.md] | 18.43 IMU hours over controls and PD [probe B4.5] | G3 |
| gait120 | 15 canonical joints (hips are ASIS proxies) as input; 12 right-leg EMG channels as target | 100 Hz; EMG 2,000 Hz | 96 [D§5, R lockbox] | pose 2.00, EMG 3.14 [probe B4.5] | G4 EMG |
| stroke_gait_vancriekinge, able-bodied | 15 canonical joints as input (z-up records, rotated to y-up); plate force totals and EMG as target | 100 Hz; force and EMG 1,000 Hz | 138 [D§5] (117 with EMG [probe B4.5]) | 0.65 of pose over all 178 non-lockbox persons, 0.43 for the 138 able-bodied [probe B4.5] | G4 GRF |
| nonan_gaitprint, nonan_gaitprint_young | head, trunk, lower_back, acc only, with gravity | 200 Hz | 41, 35 [D§5] | 47.70, 41.32 [probe B4.5] | healthy gait prior |
| sixmwt_lifespan | lower_back, acc and gyro | 100 Hz | 60 [D§5] | 6.17 [probe B4.5] | healthy gait prior |

Healthy filters: weargait controls from the source CONTROLS table, as
scripts/build_tiers_and_lockbox.py reads them (85 controls, 17 in the lockbox) [R] and writes
them into the index column group (control); stroke by the able_bodied namespace (index group
able_bodied); the other three sources are healthy by construction. The filter selects rows in
sources.rows() only: group never reaches a window, a batch or a loss (windows.cut, 2.2). The 40
non-lockbox stroke patients are never trained in the pilot (stage 2 trains them); they serve as
G4 development data.

Left out of the pilot, with the fact that decided it:

| source | fact |
|---|---|
| capture24, ltmm | free-living single sensors, 3,883 and 4,874 hours [probe B4.5]; no pilot gate reads them; they are the G9 null controls read at B6 |
| vidimu | its pose scale fields read 437.5 to 544.0 (median 516.6 over 40 random records) under a 3d_meter label, so the unit is not metres [probe B4.5]; the IMU rerun is B3, todo [P] |
| motion_x | no subject ids [D§5]; noisy video 3D that no pilot gate reads |
| physionet_multimodal_gait | gravity-free MVN acceleration in a global frame, no gyroscope [probe B4.5]; fails G15 [P N0.3] |
| ntu_rgbd, koa_pd_nm | INTERNAL tier [D§5] |
| gaitrec, physionet_gaitpdb, gaitndd controls | force only, and the pilot has no force input stem |
| drummer_tremor_model, physionet_semg_walking | C3D ingestion is B3, todo [P]; EMG only |
| every pathological training portion | stage 2 |

### 1.4 Design elements not built in the pilot

| element | enters at | fact that decides |
|---|---|---|
| session aggregator, session residual, intervention covariates | stage 4 (B7) | stage 1 has no within-subject contrast |
| deviation embedding r, descriptor-similarity loss, healthy norm tables | B6 and B7 | need pathology and the T0.6 rule file |
| descriptor families beyond the gait formulas, isotonic calibrators, joint status | B7 | stage 3; the pilot's G3 and G5 formulas are pre-registered in their rule files |
| silhouette kernel | B6 and B8 | only koa_pd_nm (INTERNAL) and cody_3 (EVAL) carry silhouettes |
| force and EMG input stems | B6 | no pilot gate reads force or EMG as input: G3 reads IMU only, G4 reads pose only |
| IMU 20 to 250 Hz measurement tokens, EMG burst tokens | stage 2 | brief pathological events; the mask reserves their type id |
| magnetometer heading kernel | B6 | weargait magnetometers are in normalized units [R weargait_frames/summary.md] |
| soft-tissue filter | the B2 renderer loop (OD9), then the pilot if B2 adopts it | G2 is a kill and its diagnostics have not located the gap; the IMU kernel takes an optional per-site FIR, identity by default |
| missing-not-at-random tracker model fitted per real tracker | stage 2 | needs real 2D validity masks, all in pathological or EVAL sources; the pilot uses a parametric synthetic tracker |
| 8-position insole head, momentum balance beyond the Newton floor | B6 | gaitpdb is left out |
| MA-22 open rig | before any release | release concern |
| 30 s context | with the session aggregator | pilot windows are independent |
| marker-level SMPL+H fits of gait120 and stroke | B6 | the canonical records hold 15 joints; the pilot reads them through the 3D pose kernel with learned per-source joint offsets |
| unrolled refinement | ablation | A§4 item 3 |
| torch.compile, Triton, FlexAttention, FP8 | not planned | none can be verified on this Windows build [R bench] |
| G12 baseline, backbone ablation | B5 | plan order |

### 1.5 Departures from the design text (the owner can overturn each)

1. Force and EMG are kernel targets, never encoder inputs, in the pilot (section 1.4).
2. The EMG likelihood trains the EMG decoder on a stopped-gradient body: 513 of the 529
   non-lockbox stroke EMG records carry an amplitude flag in their notes (record p99 from 14.5
   to 991 mV, median 258; electrode-referred scale unresolved [probe B4.5]) and the gait120
   pose to EMG mapping is verified on one record so far. G4 tests body to EMG, which needs the
   decoder, not EMG to body. The full-gradient variant is a B6 ablation.
3. The pilot's motion prior is conditioned on sex, task and shape, not on age band: AMASS has
   no age, and among the pilot's real sources age nearly identifies the source
   (nonan_gaitprint_young, sixmwt_lifespan), which HSIC does not reach inside the prior.
4. gait120 and stroke enter through the 3D pose kernel with per-source joint-definition
   offsets, not as fitted bodies (D§5 lists them as fitted bodies).
5. The kernels use the gender-routed SMPL+H skeleton of the recorded sex (neutral when unknown).
   Sex never enters the encoder.
6. The latent prior is an autoregressive Gaussian over 0.25 s latent patches with a 2 s span,
   not a normalizing flow over 2 s segments (D§4 "flow head"). It gives an exact KL per patch
   and a causality test; a flow head is an ablation once the pilot runs.
7. Refinement during training (D§4: 5 steps) runs in phase B only, every N-th optimizer step
   (N = 4 by default, set by the B4.18 bench), because each refined micro-step costs about
   1.2 to 1.6 s (section 5). Evaluation always refines 20 steps.

Departures from this spec's own first text, made after the phase A diagnosis of 2026-09-29
(reports/bench/phaseA_diagnosis.md, causes C1 to C5; the owner can overturn each):

8. Precision: the whole decoder runs in float32, not only its output projection (section 2.0).
   bf16 rounding in the trunk became knot jitter that the Hermite second derivative amplifies
   about 1e4 times (C1).
9. Rotation head: the root is a yaw and a two-component tilt around the mean root, each other
   joint an axis-angle residual around its mean rotation, instead of a 6D residual added to the
   mean pose. The 6D form had a free scale that grew to about 10 and stopped the root
   orientation from learning (C3).
10. Decoder cross-attention: unit queries and keys with a per-head content bound of at most 5
    (cosine attention), a fixed per-head distance bias, rotary periods from 0.5 s. The
    unbounded attention saturated on one patch per knot (C2).
11. Shape: the decoder's FiLM reads sg(beta), sup_pos poses the decoded and the true knots on
    the true skeleton, and on windows with a body truth the kinematics pose the draw on sg(beta).
    There the pose terms and the kernels no longer move the shape posterior, which sup_beta,
    kl_beta, subject and the motion prior's condition train (C4).
12. The decoder's initial root translation transl0 is the mean root translation of train
    windows (reports/body_store/window_priors.json), not 0 (C5a).
13. sup_root scores the horizontal root with sigma 0.20 m and the vertical with 0.02 m; every
    supervision sigma is a config value (losses.sigmas) (C5b).
14. Schedule: peak lr 1e-4 and floor 1e-5 (3e-4 and 3e-5 before), dev numbers every 1,000
    steps, training alarms that checkpoint and stop a run, and the gates G-a to G-f of
    reports/bench/phaseA_diagnosis.md section 5 as the entry condition of a phase A run (C5d).

Departures made after the root-motion diagnosis of 2026-09-30 (reports/bench/root_motion_diagnosis.md,
causes C1 to C5; the fixed phase A run learned the local pose and not the global motion; the owner
can overturn each):

15. Root path: the decoder outputs a yaw rate and a heading-frame pelvis velocity on a grid anchored
    at the centre knot L // 2, integrated by a fixed cumulative trapezoid and read at the knots by a
    cubic Hermite, plus a per-knot heading-frame residual bounded to 5 cm; the root yaw is no longer
    an output; 95 outputs (93 before); the anchor is the mean window-frame root at the centre knot
    (reports/body_store/window_priors.json, scripts/build_window_priors.py) (C1). The grid is 0.1 s,
    not the 0.25 s of the diagnosis, and the rates are the means over the grid interval, not central
    differences: the 0.25 s grid failed the diagnosis' own discretization check with either rate,
    central differences drift at both steps, and the diagnosis' fallback, 0.1 s, with the interval
    means passes it on the KIT walks (the check's own walk) and keeps the pelvis within 5 cm on every
    window checked, while its yaw still exceeds 2 deg at some knot on 2 of the 3 circle walks and on
    27 percent of 400 sampled train windows, 99th percentile 2.50 deg (reports/bench/root_grid.md,
    scripts/check_root_grid.py).
16. sup_root_rate: direct supervision of the heading-frame pelvis velocity (sigma 0.1 m/s) and of
    the yaw rate (0.05 rad/s) at the grid points, weight 1 in both phases, on windows with a body
    truth only (C2, C3), where the truth's heading is defined (the root's forward axis more than 26
    deg from the vertical; a lying, crawling or swimming root has none, and its yaw rate is atan2
    noise). The phase A diagnosis kept "no term added"; this one is added in the family of sup_deriv.
17. sup_deriv scores each body's joint velocities and accelerations in its own heading frame at the
    point, in the window frame where the truth's heading is undefined (C2).
18. sup_root_xz sigma 0.50 m (0.20 before), a weak window-frame consistency (C2).
19. Root-motion dev numbers (speed ratio, heading gain, lateral slope at the grid points of the
    heading-observable truth windows), the alarms root_speed_ratio and heading_gain from step
    5,000, and gate G-g in the entry conditions of section 5 (C1, C3).

Departure made at the start of phase B (2026-09-30; the owner can overturn it):

20. Decoder memory: in training, each decoder block runs through torch.utils.checkpoint
    (BodyDecoder.checkpoint_blocks, set by train/loop.Trainer, decoder_checkpoint; the training
    script's --no-decoder-checkpoint turns it off): the block's activations are recomputed in the
    backward pass instead of being stored, with the same values and gradients
    (tests/test_decoder.py). The first phase B call stopped at its first optimizer step with a
    CUDA out-of-memory error under the 14.5 GB cap: the float32 decoder of departure 8 stores the
    activations of every padded query. On the first 8 phase B micro-steps the peak allocated
    memory falls from 12.66 to 8.47 GB at the same seconds per unrefined micro-step
    (reports/bench/phaseB_memory.md, scripts/bench_phaseB_memory.py).

Departures made for the weekend of 2026-10-01 to 2026-10-05, each one changing what a run or an
evaluation computes (written 2026-10-04; the owner can overturn each):

21. Seed 2 of the pilot (cache/runs/pilot_v1_s2): configs/stage1_pilot_s2.yaml is
    configs/stage1_pilot.yaml with name stage1_pilot_s2 and seed 20261001, no other line changed
    (config SHA-256 a81f22e914d39dfd9cacee422dcfd3cd2f05cf36d9df7741abf47e8500c7470c against seed
    1's 7f761b1ac425a2dfefc4e546908fa766e2545d915f9c2a88dcb8d199f140b2b5). So the model
    initialization differs and the sampler streams are 20261001 (phase A) and 20261002 (phase B),
    seed 1's being 20260928 and 20260929 (loop.phase_seed is the seed plus the phase index); the dev
    windows do not move (DEV_SEED 20260929 in scripts/train_stage1.py). Step counts, the 14.5 GB
    cap and the prefetch of 2 are seed 1's. Its phase A recomputes the decoder blocks in the
    backward pass (departure 20; decoder_checkpoint true in both start events of its log.jsonl),
    which seed 1's phase A, run before departure 20, did not; the values and gradients are the
    same (tests/test_decoder.py). Both phases ran under code SHA-256 a43f893b8eab..., the code of
    seed 1's phase B (seed 1's phase A checkpoint carries 4d405d8726d0...). The queue runner
    launched both phases: phase A 2026-10-01 22:27 to 2026-10-02 02:18, phase B 02:18 to 20:10,
    exit 0 (reports/gpu_queue/job_log.csv, s2_train_A and s2_train_B). Source:
    reports/weekend_2026-10-01/tasks/T01.md, section 'Seed-2 departure text'.
22. scripts/eval_g3_dev.py on phase B checkpoints (weekend task T02): a phase B checkpoint writes
    g3_dev.json and g3_dev.md into reports/gates/g3/dev_<checkpoint name>/ (suffix _model for the
    raw weights), labelled 'G3 healthy-half dev estimate (phase B checkpoint), not a gate read';
    the simulator-only stop rule is not read on it (stops_program false, stop_rule starting 'not
    applicable'), since docs/rules/G3_RULE.md section 8 reads the stop on the phase A checkpoint
    only, while the former script would have printed 'stops the program: yes' for a phase B
    checkpoint above 40 percent stride MAPE. A phase A checkpoint keeps g3_sim.json and g3_sim.md
    in reports/gates/g3_sim/, its labels and its stop rule word for word, except that --weights
    model without --out now writes dev_<name>_model instead of overwriting the EMA folder. The
    json keys, the lines and the GateResult rows are unchanged; mdfm/gates/g3.py still writes the
    role text 'simulator-only line (phase A model)' on the MAPE row of a phase B json. Source:
    reports/weekend_2026-10-01/tasks/T02.md.
23. Evaluation memory caps and batch sizes of the weekend jobs (argv in
    reports/gpu_queue/job_log.csv; the settings each run used are in the inference block of its
    json):
    - seed 1 (cache/runs/phaseA_root_full): scripts/eval_g3_dev.py and scripts/eval_g3_diag.py ran
      with --memory-gb 12 --batch-windows 32 (the J1 dev reports, the learning curve, diag_A, diag_B,
      ablations_B), as the phase A end reference reports/gates/g3_sim/dev_phaseA_root_full_ckpt_A_010681
      of 2026-09-30 did; scripts/eval_g1_dev.py with --memory-gb 12 --batch-windows 64
      --block-windows 64 (the full runs and the --limit-takes 2 curve); scripts/eval_gait_insim.py
      and scripts/eval_g4_dev.py with --memory-gb 12; scripts/probe_g1_decompose.py with
      --memory-gb 8 and its own batch and block windows of 32 (it reproduces the J1 G1a within
      0.01 mm, reports/bench/g1_decompose_phaseA_root_full_ckpt_B_032044/decompose.md).
    - seed 2 (cache/runs/pilot_v1_s2), from the post-train templates: scripts/eval_g3_dev.py,
      scripts/eval_g3_diag.py and scripts/eval_gait_insim.py at the script defaults (G3:
      batch_windows 16 under a 2.4 GB cap, recorded in each g3_dev.json and diag.json); the three
      scripts/eval_g1_dev.py jobs at the default cap stopped on CUDA out of memory on 2026-10-02
      (cache/runs/queue/logs/pt_g1_dev_pilot_v1_s2.log and the two _lt2 logs) and were rerun on
      2026-10-04 with seed 1's settings (jobs ending in _r2); scripts/eval_g4_dev.py with
      --memory-gb 12.
    batch_windows changes the refinement point draws (reports/weekend_2026-10-01/tasks/T04.md,
    finding 4): the G1 numbers of the two seeds share their settings, while the G3 numbers of seed 2
    differ from seed 1's by those draws as well as by the seed, and so does the seed difference of
    reports/gpu_queue/branch_seeds_20261002_211426.md. The cap itself is the CUDA caching-allocator
    limit (--memory-gb of scripts/eval_g3_dev.py, default 2.4 GB), which decides whether a run
    fits, as the seed-2 G1 failures show.
24. JEPA auxiliary loss on the encoder, experiment E1 of docs/proposals/JEPA_EXPLORATION_DRAFT.md (section 2),
    code of task J2a (reports/weekend_2026-10-01/tasks/J2a.md), behind the config section jepa, off by
    default: configs/stage1_pilot.yaml and configs/stage1_pilot_s2.yaml keep their config SHA-256
    (canonical_json leaves the section out at its defaults) and a run without the section computes what it
    computed before, bit for bit (losses, gradients and weights of a tiny CPU run compared before and after
    the change). With jepa.enabled, in phase B only, real windows of at least 14 patches carry a term, pose3d
    windows excepted (the tokenizer turns every pose3d token of a window into the hip-heading frame of the
    sample nearest its centre, which a masked block can hold). Per window, 1 to 4 blocks of 4 to 8 patches,
    15 to 30 percent of its patches, hide the latent token and every measurement token of each masked patch;
    the masks come from a numpy generator seeded by the micro-step seed, so the data order, the renders and
    the refinement draws do not move. The online tokenizer and encoder read the window under the typed mask
    with every masked key removed (the context, its activations recomputed in the backward pass,
    jepa.recompute); a predictor of 4 pre-norm blocks of width 256 with 4 heads and the encoder's rotary times
    (3,422,720 parameters, output layer not zero-initialized) predicts, at each masked patch's latent position,
    the layer-normed final hidden state of an EMA target copy of the tokenizer and encoder (no gradient; decay
    0.996 rising to 1.0 by a cosine over phase B; started from the raw weights of the phase A end). The term,
    the smooth L1 (beta 1) averaged over the 512 features and the window's masked patches, enters per_window
    with the weight jepa.weight (lambda), fixed before E1 from the gradient-norm ratio of J2b's smoke run. Its
    gradient reaches the encoder, the tokenizer and the predictor, never the decoder, the prior, the heads or
    the kernels, and is taken before the main pass, so the context activations are freed before the decoder's
    are made. Alarm jepa_collapse: the spread of the target vectors or of the predictions over a micro-step's
    masked patches below 0.01 of its step-0 value. The target and the predictor ride in the model's state dict
    (jepa.*); the model EMA, which the evaluations read, leaves them out, and a model without the module loads
    a JEPA run's weights. This adds a backbone loss beside the masked reconstruction that design section 4
    and section 4 of this spec exclude: it predicts encoder representations, never sensor values, reads no
    label and uses no identity contrast. Its adoption into the main recipe is the owner's decision after E1
    (proposal sections 3 and 7).
25. Walkway anchor on the training-fold controls, F2 lever 4 (docs/proposals/F2_LEVER4_WALKWAY_ANCHOR_DRAFT.md; rule
    reports/bench/f2_lever4/rule.txt; code of task L4C, reports/weekend_2026-10-01/tasks/L4C.md; feasibility probe
    scripts/probe_walkway_anchor.py, reports/walkway_anchor/), behind the config section walkway_anchor, off by
    default: every existing stage-1 configuration keeps its config SHA-256 (canonical_json leaves the section out at
    its defaults) and a run without the section computes what it computed before, bit for bit (losses, gradients,
    weights and log record keys of a tiny CPU run compared before and after the change). With
    walkway_anchor.enabled, the trainer builds the anchor table (mdfm/data/walkway_anchor.py) from the window store's
    weargait_pd training fold, whose SHA-256 must equal walkway_anchor.persons_sha256 (train54,
    293f98d02adc48b0fa266c2125be40c23201066403fcb110543d8c882c9c77d6 in configs/stage1_pilot_f2l4.yaml). In phase B
    only, a real weargait_pd window of those persons that holds at least one stride of a straight walkway pass of a
    recording whose key is in walkway_anchor.tasks (SelfPace, HurriedPace, SelfPace_mat, HurriedPace_mat,
    SelfPace_matTURN) carries a term: per pass segment (the first and the last full footfall of one foot inside the
    window), the decoded forward distance, the integral of the forward component of the decoder's heading-frame pelvis
    velocity on the root grid between the segment's ends (linear between grid points), against the walkway distance
    between the same heels, as the squared mean-speed error over 2 x sd^2 counted at each of the segment's knots
    (sd 0.1 m/s, weight 1.0: the scale and weight of sup_root_rate); in phase A it is computed and logged only. The
    targets come from reports/weargait_targets/events.csv through a column whitelist, never from a group, phase or
    demographic column; the heel cells become metres with a cell size refit on the SelfPace trials of the
    training-fold controls only (1.27136 cm on 42 trials; the 1.27091 cm of the G3 targets is fitted with dev and PD
    persons and is not read); trials whose shank-gyroscope contacts sit more than 50 ms from the pooled offset of the
    probe's alignment check are left out (one, WHC027_HurriedPace_mat). The anchor table is refused when it holds a
    person outside that fold, a held-out (dev) person, a person of the lockbox list or of the deny lists (lockbox,
    EVAL, CODY, EXCLUDED_V1) or of the proposed PD folds (reports/stage2_sources/reviewed_ecf7bbf_2026-10-06/
    folds_proposed.csv), when its cell size was fitted on any other person or differs from the refit of its recorded
    pairs, or when its rows carry a person column or a label key; the per-micro-step targets are keyed by the window's
    index and pass assert_label_free. The sampler, the collate, the renders and every random draw are unchanged, so a
    run draws the windows it drew without the section. The gradient reaches the decoder, the latent head, the encoder
    and the tokenizer; never the shape posterior (the decoder's FiLM reads the betas detached), the source offsets,
    the nuisance heads, the prior or the kernels. This adds a real-domain supervision that section 4 of this spec
    does not have: a body measurement of the training persons (no diagnosis, severity, group or person key) and the
    very quantity that G3 reads, so the G3 rows of an anchored body are labelled 'walkway-anchored body' and are not
    label-free stage-1 rows. Its adoption into the main recipe is the owner's decision after the reading of its rule,
    whose GO is qualified by transfer lines to unseen persons, PD persons, fewer sites and simulation.

## 2. Modules with APIs and shapes

### 2.0 Conventions

- Shape letters: W windows in a micro-step; P patches of 0.25 s per window (at most 40); T
  knots per window on the 100 Hz clock (1,000 for 10 s); M decoded knots in a micro-step (flat);
  N kernel points in a micro-step (flat); S IMU sites (at most 14); G channel groups; C channels;
  R packed rows; L row length; D = 512.
- World: y up, metres; gravity (0, -9.81, 0) m/s^2 as kernels/imu.G (a standing body reads
  9.81 / 9.80665 g). Contract units at the edges: acc in g with G0 = 9.80665
  (canonical_imu.yaml, factory/imu_render.G0), gyro in deg/s (the kernel kernels/imu.imu
  returns m/s^2 and rad/s; imu_render.render and the new imu_predict convert), pose in
  canonical torso units (coords x scale gives the source unit, metres for gait120, stroke and
  amass), force in N or unit "unknown" (weargait insoles), EMG in mV with unit_scale recorded.
- Up axis of canonical 3D pose: y for amass and gait120, z for stroke_gait_vancriekinge
  (pig_vancriekinge is z-up in canonical_skeleton.yaml and its records stay z-up [probe
  B4.5]). data/sources.py applies a fixed per-source rotation to y-up before any kernel; the
  per-window yaw of pose3d_nll only absorbs heading.
- IMU records are compact: signal (T, L, 9) with location_ids (L,) into the 14-site taxonomy
  and channel_mask (L, 9), sample_valid (T, L) where the source marks gaps
  (MANIFEST_SCHEMA.md section 2); the (T, 14, 9) block exists only in imu_render.render.
- Window frame: floor at y = 0; at the window's centre knot the pelvis heading (SMPL rest +z
  projected on the floor) is +z and the pelvis x and z are 0.
- Time: seconds from the window start. Absolute times stay float64 until the window start is
  subtracted (float32 loses milliseconds on long records), float32 after.
- Precision: bf16 autocast in stems, encoder and prior; float32 in the whole decoder (it
  disables the caller's autocast; departure 8), Hermite interpolation, forward kinematics, jvp,
  kernels, likelihoods and refinement. TF32 stays off (the torch default). The kinematics and
  kernels raise on half-precision input or active autocast; MFM.render calls them inside
  torch.autocast("cuda", enabled=False).
- SMPL+H arrays and every array derived from them (J_template, J_dirs) are non-persistent
  buffers, never in a state_dict, and every checkpoint is audited for them.
- Refusals use mdfm.data.canonical.Refused. In mdfm/ and scripts/, only code under mdfm/gates/
  passes allow_lockbox=True or allow_eval=True (scripts/gate_*.py and build_g1_inputs.py call
  mdfm.gates, never the readers). tests/ may pass the flags only in refusal tests and in the
  existing reader tests of tests/test_canonical.py and tests/test_tiers.py, which read index
  rows, fake rows or one care_pd record and never train.

### 2.1 mdfm/body

**body/rotations.py (edit).** rot6d_to_matrix(x (..., 6)) -> (..., 3, 3) by Gram-Schmidt with
b = a * rsqrt(sum(a * a) + eps). The current a / (a.norm() + eps) makes backward through two
nested torch.func.jvp calls fail with an in-place error on the (N, 22, 1) norm output; the
failure and the rsqrt fix were reproduced on CPU and CUDA with torch 2.11.0+cu128 while
drafting. Adds yaw_angle(R (..., 3, 3)) -> (...) rad about +y, rot_y(theta) -> (..., 3, 3),
geodesic(R1, R2) -> (...) rad. Tests: nested-jvp backward matches float64 finite differences
(gradcheck on 4 knots); existing round trips pass; yaw_angle(rot_y(th)) == th. About +30 lines.

**body/smplh.py (edit).** SMPLHBody gains non-persistent buffers J_template (22, 3) and J_dirs
(22, 3, 16): the first 22 rows of J_regressor applied to v_template and shapedirs.
rest_joints22(betas (B, 16)) -> (B, 22, 3) m. fk(R_local (..., 22, 3, 3) with row 0 the root
orientation, transl (..., 3) m, rest (..., 22, 3)) -> pos (..., 22, 3) m and R_glob (..., 22,
3, 3). SkeletonBank(dtype) stacks the three genders: J_template (3, 22, 3), J_dirs (3, 22, 3,
16), parents (22,), all non-persistent; rest(betas (B, 16), gender (B,) int64 with 0 female, 1
male, 2 neutral) -> (B, 22, 3). forward() and vertices() keep their positional signatures;
forward() gains an optional rest=None (B, 22, 3), since today it calls rest_joints(betas) over
the 6,890 vertices for every row. BodyTrajectory caches rest once per trajectory and calls
fk(), so kernels/force.py (through BodyTrajectory.pose_at) and factory/imu_render.py
(site_kinematics) compute rest joints once per subject instead of once per timestamp (forward
kinematics with backward, no jvp: 168.7 KB per timestamp today against 4.8 KB [scratch
B4.3]). Tests: rest from the basis equals the vertex path to 1e-9 in float64 for random betas
and all genders; a mixed-gender batch equals per-gender calls; state_dict() is empty; the FK
check against a MoveAtlas canonical record, read through canonical.load, passes on a KIT take
(KIT has no non-rigid canonical record [R probe_amass_canonical_rigidity]); on the GPU,
forward kinematics of 65,536 timestamps with backward (no jvp) stays under 0.5 GB. About +70
lines, 0 parameters.

**body/hermite.py (add).** hermite_gather(y (W, T, D), w (N,) int64, t (N,) s, rate=100.0) ->
(N, D), with the Catmull-Rom tangents and one-sided ends of hermite(); it reads knots i-1 to i+2
of window w only. support(t (N,), T, rate) -> (N, 4) knot ids, used by collate to decide which
knots to decode. Tests: equals hermite() per window at 10,000 random times to 1e-12 in float64,
first and last segments included. About +40 lines.

**body/kinematics.py (new).** KnotBatch(rot6d (W, T, 22, 6), transl (W, T, 3) m, rest (W, 22,
3) m, rate=100.0), float32 or float64. kinematics(w (N,), t (N,)) -> BodyKin with p, v, a (N,
22, 3) in m, m/s, m/s^2; R (N, 22, 3, 3); omega, alpha (N, 22, 3) in rad/s, rad/s^2 (world),
all from one second-order nested jvp for every point of the micro-step. pose(w, t) -> p, R
(primal only, for supervision). point_kin(kin, joint (S,), offset (N, S, 3) in the segment
frame) -> p, a (N, S, 3), R (N, S, 3, 3), omega (N, S, 3) by a_k + alpha x r + omega x (omega x
r). Raises on half precision or active autocast. Rest joints come from SkeletonBank.rest(betas,
gender) per window; the SMPL+H parents are the same for the three genders. Cost, measured by
B4.3 (reports/bench/bench_kinematics.md): 20 KB per point held after the forward and 28.7 KB
per point at peak with backward (1.89 GB at 65,536 points), about 50 ms forward and backward
up to 16,384 points, 83 ms at 65,536 and 171 ms at 131,072 [R bench_kinematics]. Tests: equals BodyTrajectory.point_kinematics on a KIT take to 1e-6 in
float64; batched equals a per-window loop; point_kin equals the jvp of the attached point;
backward passes gradcheck (float64, 2 windows, 8 times); bf16 input raises. About 180 lines.

**body/frames.py (new).** heading_frame(joints (B, T, 22, 3), R_root (B, T, 3, 3), centre int)
-> R_yaw (B, 3, 3), origin (B, 3); apply_frame(x, R_yaw, origin); hips_heading(coords15 (T,
15, 3), valid (T, 15)) -> rad; floor_height(joints (T, 22, 3)) -> m, the 2nd percentile of the
lowest foot joint (SMPL joints 10 and 11 sit inside the foot, so the floor is a joint floor, the
same one the plausibility term and the contact rule use); contact_labels(joints (B, T, 22, 3),
rate=100, h0=0.05, v0=0.3) -> (B, T, 2)
bool [left, right]: ankle (7, 8) or foot (10, 11) joint within h0 of the take's floor and slower
than v0, the thresholds of kernels/force.contact_weights; is_treadmill(joints, rate) -> bool
(steady walking with median stance-foot horizontal speed above 0.3 m/s while the pelvis moves
below 0.2 m/s [cfg]); belt_frame(transl (T, 3), joints, rate) -> transl, a constant-velocity
Galilean shift that removes the mean stance-foot velocity (IMU signals are invariant to it).
Tests: a world yaw leaves heading-frame joints unchanged (1e-6); contacts all true on a
standing take and 0.5 to 0.75 per foot in the steady part of a KIT walk; belt_frame leaves
rendered IMU unchanged (1e-5) and brings the stance-foot speed of a BMLrub treadmill take below
0.1 m/s. About 180 lines.

### 2.2 mdfm/data

**data/amass_index.py (new) and data/amass.py (edit).** LOCKBOX_ARCHIVES is derived at import
from reports/lockbox/lockbox_persons.csv joined with reports/person_ids/amass_folders.csv: an
archive is locked when any of its subject ids maps to a lockbox person (10 archives, none mixed
[probe B4.1]). EXCLUDED_ARCHIVES: surface_model other than smplh in amass_folders.csv (CNRS,
WEIZMANN). DEV_ARCHIVES and CALIB_ARCHIVES come from the config (proposal in section 1.3).
build(out='cache/amass_index/takes.csv.gz'), run by scripts/build_amass_index.py: one row per
manifest row of source amass with take_id (the manifest key, which is the archive member path),
archive (through subject_ns; the member prefix is not the archive name: BMLrub members start
with BioMotionLab_NTroje/), subject_ns, person_id, gender, surface_model, fps, n_frames,
split in {train, dev, calib, lockbox, excluded_smplx}, tie_group. It never opens a lockbox
archive: every train, dev and calib row has a manifest fps; the blank fps of GRAB and SOMA (2,914
amass rows without fps together with CNRS and WEIZMANN [P, task B4.0]) is read from the raw npz
by the G1 input path only. check(archive, allow_lockbox=False) raises Refused for a locked
archive and always for an excluded one. takes(split) -> DataFrame. In amass.py:
load_sequence(subset, member, allow_lockbox=False) and load_many(...) call check before any
cache read and before tarfile.open; list_sequences returns *_poses.npz and *_stageii.npz (GRAB
and SOMA are all *_stageii.npz: 1,409 of the 2,141 lockbox motions [probe B4.1]); a member
whose surface_model_type is not smplh raises in _store, before anything is cached, and _store
writes surface_model into the cache file (today's cache files hold poses, trans, betas,
gender and fps only, so _read_cache refuses a file without the field and it is rebuilt).
cache/amass today holds lockbox takes (GRAB
1,340, SOMA 69, ACCAD 69, Transitions 16, MoSh 6, TotalCapture 4) and SMPL-X takes (CNRS 79,
WEIZMANN 1,426) [probe B4.1]: B4.1 moves them to the gates-only store
<store>/store/lockbox/amass_raw (blocked on question 12: <store> does not exist; until then
the refusal in amass.py makes the cached files unreachable), and training code selects takes
from the index, never by globbing a cache folder. Tests: a locked member is refused before
tarfile.open (monkeypatched to raise) and before a cache read; the 17,154 manifest rows join
one to one; split person sets are disjoint and their counts equal section 1.3; both suffixes
are listed; tie pairs sit on one side; a grep test over mdfm/ and scripts/ finds
allow_lockbox=True only under mdfm/gates/. About 180 lines plus a 60-line script.

**data/body_store.py (new).** build(split, out='<store>/store/bodies_v1', cutoff_hz=12.0,
rate=100.0), run by scripts/build_body_store.py: amass.load_many, then prep.amass_to_body_clock
with the take's gendered body, then frames.belt_frame for treadmill takes, floor moved to y = 0,
contact labels. Per archive, float32 memmaps rot6d (N, 22, 6), transl (N, 3) m and uint8 contact
(N, 2); takes.csv with take_id, archive, person_id, gender, betas (16), fps_native (from the raw
npz, asserted equal to the manifest at the manifest's printed precision: 17 rows read
120.000046 and 13 read 59.999989 [probe B4.1]), offset, n_knots, treadmill, floor_m, task
id. float32 and not float16: a 6D value near 1 rounds by up to about 5e-4 in float16, about 0.5 mm at a 1 m
lever arm, which the Hermite second derivative (1/h^2 = 1e4 at h = 0.01 s) turns into up to
about 5 m/s^2 of rendering noise [est]. Writes reports/body_store/: hours per archive and split,
treadmill count, index SHA-256, a Gaussian shape prior per gender (mean (16,), covariance (16,
16); statistics of AMASS fits, not SMPL+H arrays), the mean pose (22, 6) in the heading frame,
and the 0.1 and 99.9 percentiles of each axis-angle component per joint, all from train
persons only. BodyStore(path, split).sample(gen, n, win_s=10.0) -> rot6d (n, 1000, 22, 6),
transl (n, 1000, 3), contact (n, 1000, 2), valid (n, 1000), betas (n, 16), gender (n,), person
(n,), task (n,), in the window frame. Takes shorter than 1 s are skipped; shorter windows are
masked, never padded with motion. Size: 47.5 hours of train, dev and calib bodies, about 17.1M
knots, about 9.2 GB, which fits the 47.9 GiB of RAM as page cache [est; RAM probe B4.4].
The shape prior of the neutral skeleton (sex unknown) is the pooled prior. Tests, on a
temporary store of three train takes so that they run before the full build (which waits on
question 12): a stored window equals prep output (1e-6); no lockbox, dev or calib person in
the train shards; short takes masked. About 220 lines plus a 90-line script.

**data/sources.py (new).** SOURCES: source -> SourceSpec(streams with their role, input or
target; channels kept, never the magnetometer; person filter; pairing rule; fold rule; cap;
up_axis of its 3D pose). rows() -> admitted index rows (tier RELEASE, lockbox False, lane
body, healthy filter on the index column group), read with canonical.iterate(tier="RELEASE",
lane="body", sources=...) and flags left at False. pair(source, rows) ->
PairedRecord(streams on one clock, offsets_s):
- weargait_pd: IMU plus insole totals, same CSV and same 100 Hz clock, offset 0;
- stroke_gait_vancriekinge: pose (100 Hz), force and EMG (1,000 Hz) of one C3D trial; pose
  rotated from z-up to y-up; the probe records which plate the canonical force record maps to
  each foot and its unloaded offset, against the ankle over the plate (a record opened in
  review reads above 1 N on foot_L in every sample, peak 544 N);
- gait120: steps 01 and 02 of a trial concatenated when contiguous. Pose step records carry no
  frame field; step k maps to gait_events.step_target_frames_100hz[k] of the EMG record of the
  same trial, asserted by n_frames (on S002 trial 1: 108 frames for 468 to 575, 105 for 575 to
  679). Consecutive steps share their boundary frame (575), which is kept once, and each step
  is un-normalized by its own scale (0.5071 and 0.5056 on S002) before the join and
  re-normalized once. The EMG trial maps through gait_events: EMG sample 0 is frame
  total_frame_100hz[0] (S002: 5,900 samples at 2,000 Hz map to frames 456 to 750) [probe B4.5].
folds(seed) -> DataFrame(source, person_id, fold) with the SHA-256 of each sorted list:
weargait controls dev 14 (20 percent of 68), drawn among the TEST half of
scripts/g2_renderer_realism.py (its prepare() gives the halves), so that no mounting,
device-noise or residual-spectrum statistic of the renderer saw them; gait120 and stroke
able-bodied dev 20 percent, nonan and sixmwt val 10 percent [cfg]. covariates() reads sex
from the source demographic table where one exists (the probe names the file per source),
unknown otherwise. scripts/probe_stage1_sources.py writes reports/stage1_sources/: persons,
hours, fps, channel content, acc_content, up axis of each 3D pose source (median neck minus
ankle direction), torso length in metres (coords x scale), plate-to-foot mapping, pair
counts, windows dropped. scripts/probe_gait120_emg_alignment.py checks the gait_events rule on
every non-lockbox gait120 trial against a kinematic event. About 240 lines plus two scripts of
about 80 lines.

**data/windows.py (new).** Stream: modality in {imu, pose2d, pose3d, force, emg}, role in
{input, target}, fps, t (T,) s from the window start, values (T, G, C) float32 in contract
units, chan_mask (G, C), valid (T, G), meta (site ids, acc_content, dim, scale, unit_to_si,
layout family, device family, camera family, source, muscle table, optional root track (T, 3)
m), truth (synthetic only: mounts (G, 3, 3), biases, camera, joint offsets). Window: source,
person_id, key, synthetic, t0, duration, streams, cov (sex, age_band, task), body truth
(synthetic only: knots and the store fields). cut(paired record, win_s=10.0, stride_s=10.0,
min_s=1.0) -> list[Window] at native samples, no resampling. It copies a whitelist of record
fields (the main array and its masks, fps, source, subject_ns, person_id, key, task, dim,
modality, scale, acc_content, mag_unit, unit, unit_scale, signal_kind, channels, location_ids,
and gait_events for pairing only) and drops everything else: canonical.load also returns
tier, lockbox, scope, lane and the full index row under "manifest" (with its group column),
and stroke pose records carry label, group, analog_labels and label_prefix_stripped, stroke
EMG records label [probe B4.5]. It refuses a record without fps. covariates(record, source) ->
sex (0 female, 1 male, 2 unknown), age_band (0 to 7 by decade, 8 unknown; logged, never a
condition in the pilot), task id from
docs/rules/task_vocabulary.yaml (hashed; walk, turn, stand, sit_stand, stairs, slope, transfer,
free_living, adl, sport_dance, other, unknown). validate(window) raises on any contract
violation. Tests: sample times equal native times; no field outside the whitelist survives
(property test, one record per source, the manifest dict and group included); a record
without fps is refused; tails under 1 s dropped and counted. About 220 lines.

**data/window_store.py (new).** build_source(source, out='<store>/store/windows_v1'), run by
scripts/build_window_store.py: observation copies on the band the body carries [D§3]. IMU:
4th-order zero-phase Butterworth at 40 Hz when fps is above 80 Hz, then every k-th native sample
with k = max(1, floor(fps / 100)), so timestamps stay native. Force: the same filter and
decimation. EMG: band-pass 20 to 450 Hz, rectify, 20 ms moving RMS, log with a per-channel floor
at the 5th percentile, decimated the same way. Pose: as recorded. The filters run per
contiguous valid run (sample_valid, NaN in the weargait insole totals); a run shorter than the
filter's padding is masked. band_copy(stream) is one function, used by the store, by
factory/synth.py for every synthetic stream and by the G1 input builder, so that real,
synthetic and gate inputs share one sample grid and one band. float16 memmaps
values.f16 (N, G, C) and valid.u1 (N, G), records.csv (keys, person_id, fps_native, k, offset,
n, flags, fold), and the SHA-256 of each sorted person list in reports/window_store/. Any path
under <store>/store/lockbox is refused. WindowStore(path)[record, t0, dur] -> Stream. Tests: at
500 Hz a 10 Hz sinusoid passes and a 60 Hz one is removed; a NaN run does not spread past its
own edges; kept timestamps are native; float16
round trip under 1 mg and 0.1 deg/s; a synthetic 50 ms EMG burst peaks inside the burst; a
read-back of every shard finds no lockbox, EVAL or CODY person; person-list hashes are stable
across two builds. About 220 lines plus a 60-line script.

**data/mocap_raw.py (new).** root_track(source, subject_ns, take, allow_lockbox=False) -> t
(T,) s, pelvis (T, 3) m y-up, fps. gait120: PELV_C, the centroid of LASI, RASI, LPSI and RPSI
from the raw TRC in the MoveAtlas zips; stroke: PELO from the Plug-in Gait C3D, plus the C3D
body-mass parameter when present. Refusals are applied before any zip opens. The C3D reader is
an owner decision (question 6). The track is attached to the pose3d Stream as meta['root'];
the pose3d shards of the window store (B4.6) are rebuilt with it at B4.13, which is what
root_nll of phase B reads.
Tests: root-centring the raw joints reproduces canonical coords x scale to 1 mm; lockbox rows
refused without the flag. About 150 lines (plus about 150 for an in-tree C3D reader).

**data/sampler.py (new).** MixtureSampler(body_store, window_store, synth_spec, phase, seed):
subject-balanced; real sources weighted by sqrt(subject-hours) with a cap of 15 percent of all
tokens of a micro-step per real source [D§5 "15 percent per-source batch cap"] (a cap of 15
percent of the real tokens could not be met by the six real sources: 6 x 15 = 90 percent);
synthetic share 1.0 of tokens in phase A and 0.45 in phase B [cfg];
two windows per person where possible; 25 percent of synthetic windows rendered twice [cfg];
each visit draws a random 10 s span of a real record (the whole record when shorter, 1 s
minimum). BucketBatcher(rows, tokens_per_micro=16384, max_windows=512) yields micro-batches of
one bucket, windows packed first-fit decreasing into rows; paired windows in the same
micro-step. The window cap [cfg] matters for short records (gait120 step pairs last about 2 s
and make 10 to 20 tokens per stream-window), since kernel points and decoded knots are
budgeted per micro-step (collate).
state_dict() and load_state_dict() for resume. guard(meta) runs on every host batch before
transfer (provenance.assert_clean). Tests: over 1,000 simulated steps the source cap, synthetic
share and pairing hold; output deterministic and resumable. About 220 lines.

**data/collate.py (new).** collate(windows, bucket) -> TokenBatch:
- stem inputs per stem type: vals (N_tok, n_max, C) float32, tau (N_tok, n_max) s from the
  patch centre, smask (N_tok, n_max), nyq (N_tok,) Hz, chan_id, grp_id, dest (row, pos);
- rows: t_center (R, L) s, seg (R, L) window index or -1, ttype (R, L) int8 (0 body: latent,
  subject and the kinematic-band measurement tokens; 1 nuisance; 2 neural, reserved; 3 the
  20 to 250 Hz measurement tokens of stage 2, reserved), kind (R, L) (latent, subject,
  measurement, nuisance, padding);
- gathers: lat_idx (W, P), subj_idx (W,), nuis_idx per stream and site;
- kernel points: 4 segments of 16 consecutive band-copy samples per stream per window [cfg],
  as flat w (N,), t (N,), stream_id (N,), obs, valid and a weight per point equal to n_valid /
  n_sampled of its stream; at most 32,768 points per micro-step [cfg], segments dropped (never
  windows, and never the last segment of a stream-window, so every weight stays finite) to
  fit; with max_windows = 512 one segment per stream-window always fits (512 x 16 = 8,192);
- knots: the union of hermite supports of the points plus 64 random supervision knots per
  amass window, as (w, k) pairs; at most 32,768 per micro-step [cfg] with the decoder's root-grid
  points counted (101 per 10 s window, departure 15; the supervision knots share what the supports
  and the grid leave) (one 16-sample segment at 100 Hz needs 19 knots); dense buffers for
  hermite_gather are sized (W, T_max of the bucket);
- truth (synthetic): rot6d, transl, contact at every decoded knot of an amass window (supports
  and supervision knots), the joint velocities and accelerations of the render at the kernel
  points (so sup_deriv needs no second kinematics call), betas, gender, nuisance;
- outside the encoder input: sex, task, person_hash, source_id, gender.
Token layout per window: P latent tokens, 1 subject token, the measurement tokens, one nuisance
token per IMU site, camera or pose3d stream. Counts for 10 s windows: wrist acc only 82; one
site acc and gyro 122; nonan3 (acc only) 164; camera or mocap3d 82; six IMUs 527; xsens13
1,094. Tests: every token lands in its own row and window; rows never exceed the bucket; knots
cover the support of every point; the encoder input dataclass has no covariate, label or source
field. About 240 lines.

### 2.3 mdfm/kernels

**kernels/imu.py (extend).** imu_predict(kin, site_joint (S,), offset (N, S, 3) m, mount (N, S,
3, 3) sensor to segment, bias_acc (N, S, 3) g, bias_gyro (N, S, 3) deg/s, fir=None) -> acc (N,
S, 3) g, gyro (N, S, 3) deg/s in the sensor frame; fir is the optional per-site FIR of section
1.4, identity by default. imu_nll(pred, obs, valid (N, S, 6), log_scale (N, S, 2), weight (N,),
nu=4.0, floors 0.01 g and 1 deg/s [cfg]) -> (W,): Student-t over valid channels, times the point
weights. Tests: equals factory.imu_render.render per site to 1e-5 on a KIT take; a standing body
reads 1 g (up to 9.81 / 9.80665) and zero gyro; masked channels contribute zero. About +120
lines.

**kernels/nuisance.py (new) and scripts/build_mount_priors.py.** IMUNuisance: mount_res6d (W,
S, 6) on the prior R0[family, site]; offset_res (W, S, 3) tanh-bounded to 3 cm; bias_acc,
bias_gyro; log_scale (W, S, 2). CameraNuisance: rot6d (W, 6), log distance, log focal, log
sigma_px. SourceOffsets: nn.Parameter (n_sources, 15, 3) m, the synthetic-truth row frozen at
zero. MountPriors.load('reports/mount_priors/priors.json') -> R0 (n_families, 14, 3, 3) and the
angular spread; log_prior(nuis, family) -> (W,). The script builds the xsens13 prior from the
"fit" set of reports/weargait_frames (controls outside the lockbox) minus the 14 G3 dev
controls, reusing the population mounting fit of scripts/g2_renderer_realism.py (which leaves
lockbox persons out since its version 4); other sites get a generic family centred on identity
with 10 degrees of spread [cfg]. Tests: a zero residual gives the prior rotation; the offset residual
never exceeds 3 cm; the per-site vertical axis of the xsens13 prior matches
reports/weargait_frames (for example shank +X up) within the reported spread; log_prior is
finite on every family; the script asserts that no lockbox or dev person entered. About 160
lines plus a 150-line script.

**kernels/pose.py (new) and kernels/camera.py (reduce argument).** normalize_3d(joints (T, 15,
3) m) and normalize_2d(uv (T, 15, 2) px, valid) replicate the MoveAtlas normalization exactly
(canonical_skeleton.yaml: root-centre per frame on the pelvis, image y flipped up in 2D, divide
by the per-record median pelvis-neck length) and return coords and scale. to_canonical15(pos
(..., 22, 3), R_glob, offsets (B, 15, 3) or None) through camera.SMPL_TO_CANONICAL_15 plus the
per-source offsets in the parent segment frame (gait120 hips are ASIS proxies). pose3d_nll(pred
(N, 15, 3) m, obs (N, 15, 3) canonical, scale, unit_to_si, valid, log_sigma, nu=4) -> (W,):
compares root-centred prediction with obs x scale x unit_to_si after one yaw per window
profiled in closed form (obs first rotated to y-up for z-up sources); sigma per source
learned, at least 1 cm [cfg]. pose2d_nll(pred, camera nuisance, obs (N, 15, 3) with z masked,
scale_px, valid, conf=None, sigma0=3 px [cfg]) -> (W,): project with camera.project, root-centre
each frame on the projected pelvis and flip image y up, then camera.keypoint_nll(...,
reduce='none') in pixels against obs x scale_px. Comparing in pixels keeps sigma0 in its unit
and avoids normalizing the prediction by a window median while the record is normalized by a
record median. The canonical 2D contract has no confidence field (valid_mask only,
MANIFEST_SCHEMA.md section 1), so conf is 1 for every contract stream, synthetic ones
included. root_nll(pelvis_pred
(N, 3), root_obs (N, 3), valid) -> (W,) relative to the window mean. Tests: normalize_3d applied
to a non-lockbox AMASS take reproduces its MoveAtlas canonical record to 1e-4; zero residual at
truth; pose3d invariant to root translation and a common yaw; the closed-form yaw equals a
brute-force search to 0.1 degree; missing joints contribute zero. About 200 lines.

**kernels/force.py (extend).** com_acc(acc (N, 22, 3), sex (N,)) -> (N, 3): com() is linear in
joint positions, so the COM acceleration is com() of joint accelerations (com() takes sex as a
string today; the batched form takes a tensor). com15(joints15 (..., 15, 3) m, sex): the de
Leva segments on the 15 canonical joints (trunk from pelvis to neck, feet at the ankles), used
by the model-free Newton baseline of the G4 dry run (B4.13) on canonical pose x scale plus the
raw root track; its difference to com() on the same SMPL bodies is reported. newton_bw(com_acc,
contact_prob (N, 2)) -> (N, 2) in body weights, (1 + a_com_y / g) x share. ForceHead(hidden=64):
share = softmax over feet of the decoder contact logits plus a zero-initialized residual MLP of
foot heights and velocities; flight when both contact probabilities are below 0.5.
force_nll(pred_bw, obs (N, 2) native units, sensor_mask (W, 2, 9), weight, log_sigma, nu=4) ->
(W,) on the per-foot totals (column 8), with the per-record gain (native units per body weight)
profiled in closed form on stance samples under a log-normal prior centred on m g when the unit
is N and the mass is known, flat otherwise (weargait insoles: unit not stated
[R weargait_targets/summary.md]). Working in body weights keeps mass out of the kernel. Tests:
standing reads 0.5 BW per foot; ForceHead equals newton_bw at initialization; the NLL is
invariant to a unit change of obs; flight is masked; totals-only masks are honoured; com15
and com() agree within 3 cm [cfg] on a KIT walk mapped to 15 joints. With the gain profiled and no
known mass (weargait unit unknown, stroke mass pending question 7), force_nll constrains
timing and shape of the per-foot totals; the absolute scale in body weights comes from
newton_bw. About +140 lines, about 5k parameters.

**kernels/dynamics.py (new).** leg_torques(kin, grf_bw (N, 2), cop (N, 2, 3) m defaulting to
the ankle-foot midpoint on the floor, sex (N,)) -> tau (N, 2 legs, 3 joints hip, knee, ankle, 3)
in N m per kg, parent segment frame; bottom-up Newton-Euler with the de Leva masses and COM
fractions already in force.py plus radii of gyration added with their citation. Tests: quiet
standing gives the gravity moment only; a free-swinging leg with zero GRF matches an analytic
pendulum; the sign convention holds on a sagittal flexion. About 150 lines.

**kernels/emg.py (new) and kernels/muscle_actions.yaml.** log_envelope(signal (T, C), fps,
signal_kind) -> (T, C), the window-store formula. muscle_actions.yaml: per muscle of the
canonical_emg.yaml vocabulary, the joint, the flexor or extensor sign and the citation.
EMGDecoder(n_muscles, hidden=64): log_env_hat (N, C) = log(sum_j softplus(a_mj) relu(sign_mj
tau_j) + softplus(c_m) cocontraction + softplus(b_m) + a small non-negative MLP of joint angles
and velocities). emg_nll(pred, obs (N, C), valid, log_sigma) -> (W,), Gaussian on log-envelopes
after the per-record, per-channel log gain removed in closed form (difference of window means).
The body enters detached in the pilot (section 1.5). Tests: d env / d |tau| >= 0 on random
inputs; the NLL is invariant to a per-record, per-channel constant; unknown muscles are masked.
About 180 lines plus the YAML, about 0.1M parameters.

### 2.4 mdfm/factory

**factory/layouts.py (new) and scripts/build_layouts.py.** Layout(name, sites, fps list,
channels 'acc' or 'acc+gyro', acc_content, device ranges for noise, bias and saturation, mount
family). Sites, rates and channel masks are read by the script from one non-lockbox record per
source through canonical.load and written to reports/layouts/layouts.json, never typed.
Families: xsens13 (weargait_pd), nonan3 (head, trunk, lower_back, acc only, 200 Hz), lower_back
acc and gyro (sixmwt_lifespan), g1_six (head, lower_back, wrist_L, wrist_R, shank_L, shank_R),
wrist_L acc and gyro, wrist acc only, random k-site subsets (k = 1 to 14), camera1, mocap3d,
xsens13 plus insole totals, mocap3d plus one plate. sample_layout(gen, mixture) -> Layout.
Tests: each layout equals a real record of its source in sites, rate and channel mask; draws
match the mixture within 2 percent over 10,000 draws. About 100 lines plus a 70-line script.

**factory/camera_render.py (new).** sample_camera(gen, family='generic') -> Camera(K, R, t, fps
in {25, 30, 50, 60}, 1280 x 720): distance 3 to 6 m, height 0.8 to 1.8 m, uniform azimuth,
looking at the window-mean pelvis [cfg]. TrackerProcess: Student-t pixel noise (3 px, nu 4),
dropouts written into valid_mask that rise with joint image speed and a self-occlusion proxy
(depth order), frame-time jitter; a per-joint confidence is kept as nuisance truth only, since
the 2D contract has no confidence field. render_pose2d(pos15 (T, 15, 3), cam, gen) -> the
canonical 2D contract through kernels/pose.normalize_2d; render_pose3d(pos15, offsets, gen) -> the canonical 3D
contract. Tests: a noise-free render scored by pose2d_nll gives zero residual; dropouts rise
with joint speed; output is deterministic under a seed. About 160 lines.

**factory/synth.py (new) and a batched site_offsets in factory/imu_render.py.**
SynthSampler(store, skel, layouts, priors, device='cuda').render(bodies, specs, gen) -> list
of Windows with 1 to 3 streams each, rendered without gradients in macro-batches of about 4
micro-steps: one kinematics call for all timestamps, in chunks of 131,072 points (262,144
timestamps in 131.9 ms and 2.79 GB at peak [R bench_kinematics]). IMU at the layout rate at t_k = phi + k /
fps with phi uniform within one sample period; mounts from the priors; device bias, noise and
saturation, for the xsens13 family from the per-site levels measured on weargait quiet
standing (reports/g2_renderer/device_noise.json: median white noise 0.00363 g and 0.2066
deg/s, gyro bias spread 0.5145 deg/s, against the Device defaults 0.005 g, 0.3 and 1.0
deg/s); the 5 to 50 Hz residual spectrum of G2 version 4 enters only if B2 adopts it (OD9);
magnetometer masked. Every rendered stream then passes through window_store.band_copy, as
real streams do. pose2d through camera_render; pose3d with per-source offsets and noise; force
totals = newton_bw x a random gain. Truth kept: knots, contacts, betas, gender, nuisance, and
joint velocity and acceleration at the rendered times. render_pair renders one body twice
under resampled nuisance and device noise.
imu_render.render stays the reference renderer of the G1 harness and of the tests. Tests: IMU
before band_copy equals imu_render.render to 1e-4 g and 1e-2 deg/s for the same body and mount,
noise off; a synthetic and a real stream of the same layout share rate, filter and decimation factor
after band_copy (nonan3: 200 Hz, k = 2); every Window
passes windows.validate; token counts per layout equal section 2.2; 262,144 timestamps render
under 4 GB. About 240 lines.

### 2.5 mdfm/model

**model/rope.py and model/tokenize.py (new).** patchify(stream, patch_s=0.25) groups channels as
IMU site x {acc, gyro} and one group per pose stream (15 joints; limb groups are an ablation).
ContinuousStem(c_in, d=512, freqs_hz=(0, 4, ..., 40), hidden=1024): per channel, Hann-weighted
projections sum_i w(tau_i) x_i exp(-2 pi j f_k tau_i) / sum_i w(tau_i) over the native samples of
a 0.5 s Hann window centred on the patch (half overlap), plus the masked mean, standard
deviation and slope; frequencies above 0.8 x the stream's Nyquist are zeroed and flagged; then
Linear, GELU, Linear. The stems read band copies (100 to 199 Hz after decimation, so at most
100 samples per span; 40 Hz is exactly 0.8 x the 50 Hz Nyquist of a 100 Hz copy), never the
20 to 250 Hz band (stage 2). Rate-independent by construction, so per-sample fps needs no
resampling.
Stems: imu_acc, imu_gyro, pose2d, pose3d. Fixed input scalings: acc / 1 g, gyro / 200 deg/s,
pose in torso units in the hip-heading frame at the window centre. Embeddings: modality, site
(14), sensor type, dim; learned latent and subject token vectors; NuisanceEmbed(12 -> 512) of
the descriptor (layout family, device family, rate bucket, acc_content, dim, camera family,
site). LayerNorm or GroupNorm only, never BatchNorm [A§5]. rope.apply(q, k, t (R, L) s) with
periods 0.25 to 64 s; latent and measurement tokens at patch centres, subject and nuisance
tokens at the window centre. Tests: one analytic signal sampled at 100 and 128 Hz gives tokens
within 2 percent; content above the Nyquist is masked; a constant time shift leaves attention
scores unchanged; absent channels give the same tokens as zeros; no nn.BatchNorm in mdfm.model.
About 280 lines, about 3.5M parameters [est].

**model/masks.py (new).** build_mask(seg (R, L), ttype (R, L)) -> (R, 1, L, L) bool on the GPU:
attention stays within one window; body queries see body keys only; nuisance queries see body
and nuisance keys; reserved types see body keys and their own type; every position sees itself
(padding sees only itself). one_way_grad(model, batch) -> max |d body outputs / d nuisance
inputs|. Tests: the zero-gradient test of scripts/bench_encoder.py moves into tests/ (exactly
0.0 with the mask, non-zero without it; the bench measured 0.0 against 1.41e-07
[R bench/bench_encoder.md]); no NaN from padding. About 80 lines.

**model/encoder.py (new).** Encoder(d=512, layers=12, heads=8, mlp_ratio=4): the pre-norm
blocks of scripts/bench_encoder.py with rope on q and k. forward(x (R, L, 512) bf16, t (R, L),
mask (R, 1, L, L)) -> (R, L, 512). Attention under sdpa_kernel([CUDNN_ATTENTION,
EFFICIENT_ATTENTION, MATH], set_priority=True): without set_priority the list only enables
backends and torch's default order picks among them (signature on torch 2.11:
sdpa_kernel(backends, set_priority=False)). cuDNN 11.69 ms against 17.86 for the efficient
backend and 88.81 for math, flash refuses the mask, FlexAttention does not import on this
build [R bench; the bench forced one backend per timing]. Tests: 37,829,632 parameters at
12 x 512; the one-way zero gradient holds through 12 layers; backend parity within 2e-2 in
bf16 (GPU); a torch.profiler trace of one GPU step names the cuDNN attention kernel; a 2 x 64
configuration runs a CPU forward in under 1 s. About 100 lines.

**model/posterior.py (new).** LatentHead(512 -> 2 x 32) at the latent positions -> mu_z, logvar_z
(W, P, 32); ShapeHead(512 -> 2 x 16) at the subject position -> mu_b, logvar_b (W, 16);
logvars clamped to [-10, 4]. subject_pool(mu_b, logvar_b, person_hash) -> the precision-weighted
mean per person in the micro-step. sample(mu, logvar, n), reparameterized. Tests: pooling of a
lone window is the identity; gradients reach mu and logvar. About 90 lines.

**model/prior.py (new).** MotionPrior(d_z=32, d=256, layers=4, span=8 patches = 2 s): a causal
sliding-window transformer over z (W, P, 32) giving mu_p and log sigma_p of patch p from z_(p-8)
to z_(p-1); conditions sex (3 levels), task (vocabulary plus unknown), beta (16) and a zero
session slot reserved for stage 4; 20 percent condition dropout to unknown [cfg]. log_prob(z,
cond) -> (W,); kl(mu_q, logvar_q, z_sample, cond, free_bits=0.02) -> (W,), the analytic Gaussian
KL per patch given the sampled past, with free bits per dimension; sample(cond, n). ShapePrior:
the per-gender Gaussian of reports/body_store/. Tests: changing z_p leaves the prior of patches
up to p unchanged; log_prob equals the brute-force Gaussian product; sample then log_prob is
finite. About 150 lines, about 3.2M parameters [est].

**model/decoder.py (new; amended 2026-09-29, departures 8 to 12, and 2026-09-30, departure 15).** BodyDecoder(d=512, layers=4,
heads=8, d_z=32, n_fourier=32 up to 20 Hz). Keys and values: the P latent samples projected to
512, rope at patch centres. Queries: one per requested knot, Fourier features of the knot time
plus a learned vector, FiLM-modulated by sg(beta) (FiLM conditions the decoder on the shape and
never trains it); cross-attention, then an MLP. No attention among queries, so a sparse set of
knots equals the same knots decoded densely. Cross-attention: queries and keys normalized per
head before rope (rotary periods from DEC_ROPE_MIN_PERIOD_S = 0.5 s, twice the patch spacing),
the queries times g_h = 5 sigmoid(s_h) (2 at initialization; 8 learned scalars per block), so
every content logit lies within +-5 whatever the weights; a fixed distance bias -|t_query -
t_patch| / tau_h with tau = 0.125, 0.25, 0.5, 1, 2, 4, 8 s and one head without bias, passed as
the float mask of scaled_dot_product_attention (-inf at padding patches). attention_stats()
recomputes entropy, logit gap, mass within 0.5 s of the query and the residual norm per block.
forward(z (W, P, 32), beta (W, 16), knots (M,) pairs (w, k), n_patches, lengths (W,)) -> rot6d
(M, 22, 6) with the root orientation first, transl (M, 3) m in the window frame, contact_logit
(M, 2), pos_logvar (M, 22) log m^2 (isotropic, pelvis-relative joint positions in the heading
frame) and root, the RootGrid of the windows, from a zero-init output of 95 values. Integrated
root (departure 15): the root grid of a window holds the knots L // 2 + 10 i (0.1 s, 101 points
for a 10 s window), decoded in the same trunk call as the requested knots; there the outputs are
the yaw rate omega and the heading-frame pelvis velocity u (lateral, forward), both means over the
grid interval; a fixed cumulative trapezoid from the centre point integrates omega into psi (the
yaw about the centre heading) and rot_y(psi) u into X (the horizontal displacement from the centre
knot), and a cubic Hermite with the tangents omega and rot_y(psi) u reads psi and X at the knots.
The lengths are required (ValueError without). Per knot: the root rot_y(psi) R_mean,0 exp([a, 0,
c]) (a and c the tilt; R_mean,0 the mean root without its own yaw), each joint j R_mean,j
exp(omega_j), returned as 6D with orthonormal columns (no output has a free scale), R_mean the
rotations of the training mean pose (a buffer not saved, rebuilt from a loaded mean_pose that
differs); the horizontal root anchor + X + rot_y(psi) r - r_c, r a heading-frame residual bounded
to 5 cm per component (tanh) and r_c its value at the centre point, so the centre knot sits on the
anchor, the mean window-frame root at the centre knot of train windows
(reports/body_store/window_priors.json transl_centre_mean, scripts/build_window_priors.py, which
asserts an x and z spread under 2 cm); the vertical root transl0 plus a residual, transl0 the mean
root translation of train windows in the window frame (y = 1.108 m), so the initial body stands on
the floor. The whole decoder runs in float32 whatever autocast the caller holds: a bf16 value near
1 m rounds by about 4 mm, which the second derivative turns into tens of m/s^2. scatter_dense(out,
W, T) places decoded knots into (W, T, ...) buffers for hermite_gather. Tests: knots decoded one
at a time equal knots decoded together to 1e-6, the grid too; at initialization the output is the
mean pose (rotation matrices to 1e-6) standing still at the anchor and transl0's height, and the
ankle and foot joints lie within -0.05 to 0.20 m of the floor; outputs under a bf16 autocast equal
float32 decoding to 1e-6; the rotation outputs stay orthonormal with the output weights times
1e3; the content logits stay within +-5 with the q and k weights times 1e3; the 0.125 s head puts
at least 0.9 of its mass within 0.5 s of the query at initialization; FiLM passes no gradient to
beta; a loaded mean_pose rebuilds R_mean; constant rates decode a line and a circle of radius v /
omega (1 cm), the yaw exact; the centre knot sits on the anchor with yaw 0 whatever the weights;
the residual stays within 2 sqrt(2) x 5 cm of the path; the true grid rates of a train KIT walk
through the operator give the pelvis within 5 cm and the yaw within 2 deg at every knot. About 500
lines, about 12.7M parameters [est].

**model/heads.py (new).** NuisanceHeads(512): imu(h (W, S, 512), family) -> IMUNuisance;
camera(h (W, 512)) -> CameraNuisance; every residual zero-initialized; scales floored. Heads read
nuisance-token outputs only. Tests: at initialization the heads return the prior centres; bounds
hold on extreme inputs; no head indexes a body-typed position. About 120 lines, about 1.5M
parameters [est].

**model/mfm.py (new).** MFM(cfg) holds tokenize, encoder, posterior heads, prior, decoder,
nuisance heads, ForceHead, EMGDecoder, SourceOffsets and a non-persistent SkeletonBank.
encode(batch) -> Posterior(mu_z, logvar_z, mu_b, logvar_b, nuisance per stream). decode(z,
beta, knots, n_patches, lengths) -> BodyOut (the lengths required, departure 15). render(body, batch, nuis) -> predictions per stream at the kernel
points (IMU (N, S, 3) x 2; pose (N, 15, 2 or 3); force (N, 2) BW; EMG (N, C)). log_lik(pred,
batch) -> (W,). log_joint(z, beta, nuis, batch) -> (W,) = log_lik + log p(z | c) + log p(beta |
sex) + log p(nuis | family). posterior_samples(batch, n=16, refine_steps=20) -> bodies at all
knots with pos_logvar. Tests: a tiny configuration runs forward and backward on CPU in under 5
s; provenance.audit_state_dict passes; the pilot configuration has 57M to 62M parameters.
About 220 lines.

**model/refine.py (new).** refine(model, post, batch, steps, lr=0.05, points_per_stream) ->
Posterior with refined means; Adam on z, beta and the nuisances to minimize -log_joint with
every model parameter frozen, detached result, encoder variances kept. steps 5 in training and
20 at evaluation [D§4]; 64 points per stream in training, 256 at evaluation [cfg]. Training
scores the decoder and kernels at z_st = z0 + sg(zK - z0). refine(..., body=None): with a body
given (truth knots), z and beta are skipped and only the nuisances move, which is what the
tests use before any training. Tests: the objective decreases on a toy batch; no model
parameter receives a gradient; with the body held at its truth knots, a synthetic xsens13
window with every mount 20 degrees off is refined to under 5 degrees; steps = 0 is the
identity. About 120 lines.

### 2.6 mdfm/train

**train/losses.py (new).** stage1_losses(model, batch, post, body, pred, phase) -> dict of (W,)
terms and the total; each term also callable alone (section 4). assert_label_free(batch) raises
on any key named label*, group, cohort, severity, diagnosis, phase or manifest, at any depth of
the batch and of the window meta. Tests: every term is zero
or minimal at truth on a synthetic batch; HSIC is zero for independent summaries, positive for
source-coded ones, inactive in strata with fewer than 2 sources; the plausibility hinge is zero
inside the limits; the label check trips on an injected key; a grep test shows no loss reads a
label, group, cohort, severity or phase field. About 280 lines.

**train/provenance.py (new).** person_lists(store, window_store) -> {source: sorted ids}.
hashes() reads at run time, never typed: records and lockbox SHA-256 from
reports/manifest_tiers/hashes.json and reports/lockbox/lockbox.sha256 (today v1.2, lockbox
619dfdac..., records ca5a2d9e...); the MoveAtlas HEAD (f2e6f99 expected [P]); body-store and
window-store index hashes; rule-file hashes; config SHA-256; a code SHA-256 over mdfm/**/*.py
(the tree has no commit). assert_clean(persons) raises on any lockbox, EVAL or CODY person, on
a CODY namespace of reports/manifest_tiers/cody_denylist.json, and checks that the loader
refuses the canary cody_canary:mdfm-v1 (reports/manifest_tiers/cody_denylist.json,
canary_subject_ns). audit_state_dict(sd) compares the SHA-256 of every tensor with every
SMPL+H array as SMPLHBody stores it (shapedirs cut to 16 betas) and every derived array
(J_template, J_dirs) of the three genders, compares small tensors with them by allclose, and
screens keys and shapes (any dimension equal to 6890 or 52). It runs on the model, the EMA and
every tensor of the optimizer state. A hash cannot catch an arbitrary sub-slice; the shape
screen and the non-persistent buffers are the guard there. Tests: one injected lockbox person,
one EVAL person and the canary each abort; the audit trips on an injected shapedirs buffer,
J_template or J_dirs, and on a (6890, 3) tensor, and passes on the pilot model. About 170
lines.

**train/config.py and train/checkpoint.py (new).** Config dataclasses (model, mixture, schedule,
loss weights) hashed as canonical JSON; run configs in configs/stage1_pilot.yaml.
save(path, model, ema, opt, sched, sampler_state, rng, cfg, provenance) writes only after
audit_state_dict and assert_clean pass; load(path) uses torch.load(weights_only=True) and
re-checks the hashes. Tests: round trip; a failed audit blocks the write; resume reproduces the
next 10 losses exactly on the tiny CPU configuration. About 150 lines.

**train/loop.py, scripts/train_stage1.py, scripts/bench_stage1.py (new).** train(cfg) runs phase
A then phase B with the dev hooks of section 5. bench(cfg, steps=200) writes
reports/bench/stage1_pilot.md: useful tokens/s per bucket, the step-time split (render, stems,
encoder, decoder, kinematics, kernels, refinement, optimizer) and peak memory per bucket. The
health fields and the alarms of section 5 (departure 14). Tests: a 20-step CPU smoke run on a
tiny configuration with fake data lowers the loss; a saturated attention after step 500 logs an
alarm, writes a checkpoint and stops the run. About 280 lines plus two scripts.

### 2.7 mdfm/eval, mdfm/readouts, mdfm/gates

**eval/conformal.py (new).** fit(scores (n,), level=0.90) -> the finite-sample quantile;
ellipsoid_factor(radius (n,), dof=3, level=0.90) -> c such that the chi-square 3-dof ellipsoid
of the moment-matched predictive, scaled by c, covers 90 percent of the calibration truths;
energy_score(draws (n, 3), y (3,)). Tests: coverage 0.90 plus or minus 0.01 on 100,000
exchangeable draws; the energy score of a Gaussian matches its closed form. About 70 lines.

**readouts/gait.py (new).** initial_contacts(contact_prob (T, 2), rate=100, thr=0.5,
refractory_s=0.3) -> times per foot. strides(joints (T, 22, 3) m, ics, window_of_frame) ->
DataFrame(foot, t0, t1, length_m): horizontal ankle-joint distance between successive ICs of
one foot inside one window frame. pass_speed(joints, t0, t1) -> m/s from the horizontal pelvis
displacement. cadence(ics) -> steps/min. shank_gyro_ic(gyro (T, 3) deg/s, rate) -> times:
sagittal axis as the first principal axis of the walking gyroscope, mid-swing peaks above 50
deg/s, IC at the next negative peak within 0.4 s (Salarian et al. 2004), fixed before any read.
stitch(windows, hop) keeps each frame from the window where it is most central. Tests: exact on
synthetic walking with known strides; on 20 dev walks truth-body strides equal the ankle
displacements and speeds match factory.gait.speed_and_cadence within 0.05 m/s; shank_gyro_ic
on rendered shank gyroscopes finds the truth-body contacts within 50 ms in 90 percent of steps.
About 200 lines.

**gates/common.py (new).** person_weighted_mean(values, persons); cluster_bootstrap(values,
persons, n=2000, seed) -> (lo, hi); GateResult(gate, line, point, interval, pass_line,
kill_line, verdict in {pass, kill, undetermined, not evaluated}, trivial_baseline, rule_sha256,
checkpoint_sha256, train_persons_sha256, n_persons); rule(path) -> text and SHA-256, refused
unless registered in docs/rules/gate_rule_hashes.csv; log_lockbox_read(gate, ckpt, rule_hash)
appends to reports/gates/lockbox_reads.csv and refuses a second read of the same gate, checkpoint
and rule without --reread and a logged reason; write(results, dir) writes JSON and MD and asserts
no em-dash. Tests: the interval brackets the point; a person with 1,000 windows counts once;
verdict directions; an edited registered rule refuses. About 160 lines.

**gates/g1.py, scripts/gate_g1.py, docs/rules/G1_RULE.md (new).** Section 6. About 320 lines.

**gates/g3.py, scripts/gate_g3.py, docs/rules/G3_RULE.md (new).** Set: weargait_pd lockbox
controls (17), tasks SelfPace and HurriedPace, 13 IMUs only (acc and gyro, magnetometer
masked); walkway, insole and PKMAS columns are targets only (reports/weargait_targets/events.csv,
reports/weargait_targets/cache/insole_events.csv; both hold the 252 lockbox trial rows of 36
persons flagged in a lockbox column [R weargait_targets/summary.md], which the harness reads
only in a logged lockbox read and filters out in every dev run). Inference: 10 s windows
with a 5 s hop, 20 refinement steps, stitched.
Lines: stride MAE in cm on strides matched to walkway strides within 0.15 s, per person then
over persons; gait speed MAE per pass against the walkway; initial-contact median absolute
error against the reference that OD12 fixes (proposal: walkway contact; insole with the
per-trial offset removed as a second reference), after one constant offset per detector fitted
on the 14 dev controls only; the shank-gyroscope detector on the same events and the 20 percent
margin; matched shares reported. Simulator-only variant: stride MAPE of the phase A model.
Trivial baseline: training-fold mean stride and speed per task. Tests: walkway ICs fed as model
ICs give zero stride error; the offset fit asserts dev person ids; unmatched events are counted.
About 300 lines.

**gates/g4.py, scripts/gate_g4.py, docs/rules/G4_RULE.md (new).** GRF: stroke lockbox (10),
canonical pose x scale plus the raw pelvis track, 20 refinement steps, ForceHead per-foot GRF in
body weights against the plate (percent body weight per the rule, question 7); Newton floor on
the same bodies; RMSE and correlation. Metric-3D round trip: the true 15 joints (metres, root
track restored) projected into one synthetic camera, bodies re-estimated, MPJPE and PA-MPJPE in
mm, reported without kill. EMG: gait120 lockbox (24), level-walking trials, pose to body to EMG
decoder, Pearson r of log-envelopes over the pose frames per trial and muscle for tibialis
anterior, gastrocnemius medialis, vastus lateralis and biceps femoris (all four are in the
gait120 channel table [probe B4.5]); per person median, then the median over persons. Template
baseline: mean log-envelope per muscle over the step cycle of training persons. Development
data: the 40 non-lockbox stroke patients (not trained in the pilot) and the gait120 dev fold.
Tests: on a synthetic body with known GRF the Newton floor reproduces it; the measured envelope
as prediction gives r = 1; step normalization is exact on S002. About 240 lines.

**gates/g5.py, scripts/gate_g5.py, docs/rules/G5_RULE.md (new).** care_pd is EVAL:
canonical.load(..., allow_eval=True) is called here only. Frontal and sagittal synthetic
projections of the care_pd walks through the camera_render tracker process; bodies
re-estimated; the UPDRS_GAIT probe (5-fold person-level cross-fitting inside care_pd, an
internal instrument) run on root-free gait descriptors of source bodies and of re-estimated
bodies, QWK drop; BMCLab ON versus OFF AUROC of the same descriptors (19 subjects [D§5]),
difference; PA-MPJPE as a diagnostic. Tests: on amass dev walks, identity projections give zero
drop. About 220 lines.

## 3. Data flow

1. AMASS take to body store (offline, B4.4).
   amass_index row (train, dev or calib) -> amass.load_many (guarded) -> poses (T_n, 156)
   axis-angle rad, trans (T_n, 3) m z-up, betas (16,), gender, fps_native (120 for 8,880 rows,
   100 for 4,620, 60 for 674, 250 for 57, 150 for 9 among all amass rows with a manifest fps,
   lockbox included [probe B4.1]) -> prep.amass_to_body_clock(rate 100, cutoff 12 Hz, y-up,
   gendered SMPLHBody) ->
   rot6d (T, 22, 6), transl (T, 3) m -> belt frame if treadmill -> floor to y = 0 -> contacts
   (T, 2) -> float32 shards on <store>/store/bodies_v1.

2. Synthetic window (every step, GPU, no gradient).
   BodyStore.sample -> rot6d (1000, 22, 6), transl (1000, 3), contact (1000, 2), betas (16,),
   gender, person, task, in the window frame. Layout draw -> device, mounts from the priors,
   camera. synth.render, one kinematics call per macro-batch -> Streams:
   - IMU (T_dev, S, 9): acc g, gyro deg/s, magnetometer masked, at t = phi + k / fps;
   - pose2d (T_cam, 15, 3): canonical 2D, z masked, tracker dropouts in valid_mask;
   - pose3d (T, 15, 3): canonical 3D with source offsets and noise;
   - force totals (T_f, 2, 9) = newton_bw x gain (target only).
   Every stream -> window_store.band_copy. Truth kept. 25 percent of windows rendered twice
   [cfg].

3. Real window (offline, B4.5 and B4.6; sliced per step).
   canonical.iterate(tier RELEASE, lane body, sources = pilot, allow_eval False, allow_lockbox
   False) -> healthy filter on the index column group -> sources.pair (stroke pose rotated to
   y-up, gait120 steps joined) -> windows.cut (field whitelist) -> window_store band copies
   (IMU and force at 40 Hz decimated on native timestamps, EMG 20 ms log-envelopes, pose as
   recorded; float16) -> per step the sampler draws a record, t0 and a duration of at most 10 s
   -> Streams with the same structure as synthetic ones. Paired families arrive as one window
   with several streams on one clock.

4. Collate (workers, then GPU).
   Stem inputs per token (native samples of the 0.5 s Hann span, offsets, masks, Nyquist);
   rows with seg, ttype and t_center; the typed block-diagonal mask; kernel points (4 segments
   of 16 samples per stream per window, weights n_valid / n_sampled); decoded-knot list
   (supports plus 64 supervision knots per amass window); truth; covariates kept outside the
   encoder input.

5. Model.
   Stems -> x (R, L, 512) bf16 -> encoder (12 x 512, rope, cuDNN SDPA) -> h (R, L, 512) ->
   gathers: mu_z, logvar_z (W, P, 32); mu_b, logvar_b (W, 16), pooled per person; nuisance per
   IMU site and camera -> z sample (every N-th optimizer step in phase B: 5 stopped-gradient
   refinement steps first) -> decoder at the M knots -> rot6d (M, 22, 6), transl (M, 3),
   contact_logit (M, 2), pos_logvar (M, 22), float32 -> dense (W, T_max, ...) buffers ->
   KnotBatch.kinematics at the N points -> BodyKin -> kernels: IMU (N, S, 3) x 2 in g and deg/s;
   pose3d (N, 15, 3) m; pose2d (N, 15, 2) px, root-centred; root (N, 3); force (N, 2) BW; leg
   torques (N, 2, 3, 3) N m/kg and EMG log-envelopes (N, C) on the detached body.

6. Losses (section 4) -> backward -> AdamW step after 2 micro-steps.

7. Evaluation path: encode -> refine 20 steps (256 points per stream per step) -> decode all
   1,000 knots per window for 16 latent samples -> joints (16, W, 1000, 22, 3) with pos_logvar
   -> readouts or gate metrics.

## 4. Losses

Scale convention: every term is a per-window sum estimated from samples (the sampled sum times
n_total / n_sampled), so likelihood, supervision and KL keep ELBO proportions; the total is the
mean over the windows of the optimizer step. Weights and fixed scales are [cfg]. No label
enters any term: assert_label_free runs on every batch and windows.cut keeps a field whitelist
at the source. Control-group filters select training persons; they are never a target.

| term | windows | definition | weight, phase |
|---|---|---|---|
| imu_nll | every IMU stream | Student-t, nu 4, on acc (g) and gyro (deg/s) in the sensor frame at native band-copy timestamps (band limited at 40 Hz), scales from the nuisance heads with floors 0.01 g and 1 deg/s | 1, A and B |
| pose3d_nll | gait120, stroke able-bodied, synthetic mocap3d | Student-t, nu 4, on root-centred metres after source offsets and a closed-form yaw per window; sigma per source at least 1 cm | 1, A and B |
| pose2d_nll | synthetic camera streams | Student-t, nu 4, sigma 3 px (confidence 1: the 2D contract has none), in pixels against obs x scale_px, through the camera nuisance | 1, A and B |
| root_nll | pose3d streams with a raw root track | Student-t on the pelvis relative to its window mean | 1, B |
| force_nll | weargait insoles, stroke plates, synthetic force | Student-t on per-foot totals in body weights, per-record gain profiled in closed form, flight masked | 1, B (synthetic force from A) |
| emg_nll | gait120 and stroke able-bodied EMG | Gaussian on 20 ms log-envelopes after the closed-form per-record, per-channel log gain; body detached | 1, B |
| sup_rot | amass windows | squared geodesic angle of the 22 joint rotations (root in the window frame) over 2 x (0.1 rad)^2 at the supervision knots | 1, A and B |
| sup_pos | amass windows | pelvis-relative joint positions in the heading frame, the decoded and the true knots both posed on the true skeleton (the term scores the pose, never the shape; departure 11): squared error over 2 x (0.02 m)^2, plus the Gaussian NLL of pos_logvar against the detached mean (bands calibrate without pulling the mean) | 1, A and B |
| sup_root | amass windows | root translation in the window frame: horizontal squared error over 2 x (0.50 m)^2 (a weak window-frame consistency; 0.20 m before departure 18) plus vertical over 2 x (0.02 m)^2, logged apart as sup_root_xz and sup_root_y (departures 13, 18) | 1, A and B |
| sup_root_rate | amass windows | at the root-grid points inside the window where the truth's heading is defined: the decoded heading-frame pelvis velocity and yaw rate against the truth's grid-interval means (the heading-frame velocity in the truth's own integrated yaw), over 2 x (0.1 m/s)^2 and 2 x (0.05 rad/s)^2, logged apart as sup_root_rate_u and sup_root_rate_yaw; local quantities that a heading error elsewhere does not enter (departure 16) | 1, A and B |
| sup_deriv | amass windows | joint velocity and acceleration at the kernel points, predicted from the decoded knots through KnotBatch.kinematics and true from the render's kinematics of the true knots (the same Hermite, kept by synth), each body's vectors in its own heading frame at the point, the window frame where the truth's heading is undefined (departure 17), over 2 x (0.1 m/s)^2 and 2 x (2 m/s^2)^2; keeps the decoder smooth under the 1e4 amplification | 0.1, A and B |
| sup_contact | amass windows | BCE of contact logits against the truth contacts | 1, A and B |
| sup_beta | amass windows | Gaussian NLL of the true betas under the shape posterior | 1, A and B |
| kl_z | every window | KL of q(z) against p(z given sex, task, beta), summed over patches, free bits 0.02 nats per dimension | annealed 0 to 1 over the first 10 percent of phase A |
| kl_beta | every window | KL of q(beta) against the per-gender shape prior | same schedule |
| nuis_prior | every nuisance | -log p(nuisance given layout family): mount angle against the prior spread, bias priors of the device family | 1, A and B |
| nuis_aux | synthetic windows | squared mount geodesic and bias errors in prior-sd units, camera pose error, against the rendered nuisance | 0.1, A only |
| consistency | paired synthetic windows | symmetric KL between q(z given x_a) and q(z given x_b), one body under two nuisance draws | 0.5, A and B |
| plausibility | every window | hinge outside the joint-limit percentiles; relu(-y) of the foot joints; contact probability x horizontal foot speed squared | 0.1, A and B |
| subject | persons with two windows in the micro-step | KL of q(beta given window) against the pooled q(beta given person) | 0.1, A and B |
| hsic | every window | Gaussian-kernel HSIC between mean_p(mu_z) (W, 32) and a one-hot source code, only in task strata (docs/rules/task_vocabulary.yaml) that hold at least two sources in the micro-step; synthetic windows carry one code, synthetic [cfg; the alternative leaves them out of the term]; always logged | 0.1, B |
| amortization | refined steps | squared distance of mu_z to sg(z_K) over sg(sigma_z^2) | 0.1, B |

The fixed scales of the supervision rows (rot 0.1 rad, pos 0.02 m, root_xz 0.50 m, root_y
0.02 m, vel 0.1 m/s, acc 2 m/s^2, root_vel 0.1 m/s, root_yaw_rate 0.05 rad/s) are config values
(losses.sigmas of configs/stage1_pilot.yaml; dev choices: root_xz in 0.2, 0.5 and 1.0 m,
root_vel in 0.05 and 0.1 m/s, root_yaw_rate in 0.05, 0.1 and 0.2 rad/s; 0.05 because 0.1 left the turning batch
of scripts/check_learnability.py --batch turns at a turn slope of 0.85, reports/bench/learnability_turns_yawrate010.md). On windows with a body truth no pose term and no
kernel moves the shape posterior: the decoder's FiLM reads sg(beta), sup_pos poses both bodies
on the true skeleton and the kinematics pose the draw on sg(beta), so the shape learns there from
sup_beta, kl_beta, subject and the motion prior's condition; on real windows the kernel
likelihood still reaches beta through the skeleton (departure 11).

Bone-length constancy needs no term: one
beta per window through SkeletonBank holds it by construction. Momentum balance enters through
force_nll wherever force exists. Absent by design [D§4]: adversarial site heads, masked
reconstruction of raw sensors, the descriptor-similarity loss (stage 2), any direct body
supervision on real data.

## 5. Training loop and memory budget

Phases.
- Phase A, synthetic only (train archives through the factory), 25 percent of the optimizer
  steps [cfg]. KL annealing, nuis_aux on, refinement off. Its last checkpoint is the
  simulator-only model of G3.
- Phase B, synthetic plus the real healthy windows, 75 percent of the steps [cfg]. Synthetic
  share 45 percent of tokens, so the cumulative share is 0.25 x 1.0 + 0.75 x 0.45 = 0.5875, under
  the 60 percent cap of D§5 read cumulatively. Read per step, the cap would forbid phase A,
  which is 100 percent synthetic; phase B alone (0.45) meets it either way. The reading is
  part of question 1. nuis_aux off; refinement on (5 stopped-gradient steps every N-th
  optimizer step, N = 4 by default, set by B4.18); HSIC on; EMG decoder trained on the
  detached body; 500-step learning-rate re-warmup.
- Token budget: 1.0 to 1.5 billion tokens in total [est], fixed by B4.18 from the measured
  end-to-end rate.

Batching.
- 16,384 tokens per micro-step, 2 micro-steps per optimizer step [cfg]. One bucket per
  micro-step; rows start at the measured 128, 640 and 1,536 tokens [R bench] and B4.18 re-tunes
  them from the pilot's window-length histogram. Windows packed first-fit decreasing; typed
  block-diagonal mask. Long packed rows are not used: 12 x 768 ran at 22 percent of nominal peak
  packed against 32 percent bucketed, 12 x 512 at 17 against 25 percent [R bench].
- At most 32,768 kernel points, 32,768 decoded knots and 512 windows per micro-step [cfg].

Precision and optimizer.
- bf16 autocast as in section 2.0; fp32 master weights; no GradScaler.
- AdamW: betas 0.9 and 0.95, eps 1e-8, weight decay 0.05 except norms, biases, embeddings and
  source offsets; fused=True if the smoke test passes on this build, foreach otherwise; peak lr
  1e-4, 2,000 warmup steps, cosine to 1e-5 within each phase (3e-4 and 3e-5 before the phase A
  diagnosis, where one update at 3e-4 raised the held-out loss; departure 14); gradient clip
  1.0; EMA of weights, decay 0.999, saved beside the raw weights [cfg].
- No torch.compile, Triton or FlexAttention. If the B4.18 profile shows kinematics and rendering
  above 40 percent of the step, they are captured as CUDA graphs with static padded shapes
  (native CUDA, no Triton), one graph per bucket.

Data loading on Windows.
- DataLoader with spawn, 4 to 6 persistent workers, pinned memory; workers only slice memmaps
  opened lazily per worker on F: (NTFS, 2,540 GiB free [probe B4.4]), never zips on the exFAT D:
  disk. Synthetic rendering runs on the GPU in the main process. TORCHINDUCTOR_CACHE_DIR is set
  by mdfm/__init__.py.
- The guard runs on every host batch before transfer.

Checkpoints.
- Every 2,000 optimizer steps and at each phase end, to
  <store>/runs/stage1_pilot_<date>_s<seed>/, never into the tree: state_dict, EMA and
  optimizer state (all three audited), scheduler, sampler state, RNG states, config JSON and
  its SHA-256, provenance.json (section 2.6) with the sorted person lists per source and fold as files in the run folder.
- save() refuses to write unless assert_clean and audit_state_dict pass; load() uses
  weights_only=True.

Logging and dev evaluation.
- log.jsonl every 50 steps: every loss term, KL per dimension and the number of active latent
  dimensions, tokens/s, step-time split, peak memory, gradient norm, per-source and synthetic
  token shares, nuisance errors on synthetic windows, refinement objective drop, HSIC; and the
  health of the first micro-step of each interval, without gradient: per decoder block the
  attention entropy, logit gap, mass near the query and residual norm; the RMS of the raw
  rotation outputs; logvar_z; the spread of mu_z between and within windows; on windows with a
  body truth the mean absolute beta error and the rest-height ratio.
- Training dev numbers every 1,000 steps (loop.evaluate on 96 dev windows): the heading-frame
  joint error, also on the true skeleton and with z = 0, the root orientation error, the
  within-window spread ratio, and the constant mean pose on the prior-mean skeleton on the same
  windows; the root motion at the valid root-grid points of the windows whose input observes the
  heading (losses.heading_observable: a gyroscope channel or a pose stream): root_speed_ratio
  (decoded over true pelvis speed where the truth moves faster than 0.3 m/s), the same beyond 3 s
  from the centre, heading_gain and heading_corr (slope and correlation of the decoded on the true
  yaw about the centre, within 120 deg), the mean yaw error and that of a body that never turns,
  and lateral_slope (departure 19).
- Alarms (schedule.stop_on_alarm, default true; departures 14 and 19): an attention entropy under
  0.5 nat in any decoder block after step 500, a spread ratio under 0.3 after step 2,000, a beta
  error above 2 or a mean logvar_z under -9 after step 1,000, a z = 0 dev error within 1 mm of the
  dev error after step 2,000, or a dev root_speed_ratio under 0.5 or heading_gain under 0.3 from
  step 5,000 logs 'alarm', writes a checkpoint and stops the run. The phase A run of 2026-09-29
  would have stopped at step 2,000, that of 2026-09-30 (phaseA_fix_full) at step 5,000.
- Every 5,000 steps: G1 protocol on 300 fixed dev-archive windows with 5 refinement steps; in
  phase B also G3 dev lines on the 14 dev controls, gait120 dev EMG correlation and stroke dev
  GRF. Full dev evaluation with 20 refinement steps at each phase end [cfg].
- Training writes nothing under reports/; gate scripts do. The loop never touches the lockbox.

Entry gates of a phase A run (reports/bench/phaseA_diagnosis.md section 5 and
root_motion_diagnosis.md section 6; departures 14 and 19): the test suite (G-a);
scripts/check_learnability.py on one fixed batch of 20 windows, 1,500 steps from initialization:
joint error under 10 mm, sup_deriv under 1e4, root orientation under 3 deg, beta error under 1,
rest height within 1 cm, attention entropy above 1 nat in every block, and with --batch turns (17
train windows turning more than 60 deg about the centre, with a defined heading) a horizontal root
RMS error under 0.10 m and a root yaw error under 5 deg (G-b); scripts/bench_stage1.py at 16,384 tokens under 14.5 GB
reserved, else 12,288 tokens with 3 micro-steps (G-c); scripts/probe_step_size.py at step 2,000
of a 5,000-step run: one update at the peak lr lowers the held-out loss, no term holds more than
half of its first-order gain, and the root terms (sup_root_xz, sup_root_rate_u,
sup_root_rate_yaw, sup_rot_root) together hold at least 5 percent of it (G-d); that run at step
5,000: dev error at least 10 mm under the constant-pose prior, spread ratio above 0.5, entropy
above 1 nat, the z = 0 error at least 10 mm above the dev error, dev root_speed_ratio at least 0.7
and heading_gain at least 0.6, no alarm (G-e); scripts/probe_root_motion.py on that checkpoint,
dev walks rendered in xsens13, unrefined: speed ratio at least 0.8 in every time bin to 5 s,
lateral slope at least 0.8, velocity cosine (decoded against true) at least 0.9 at 3 to 5 s, time-warp response
within 0.1 of the true ratio at 0.8 and 1.25, heading and pelvis errors at most those of the mu_z
latent dead reckoning (G-g); then the rest of phase A and the G1 and G3 simulator-only dev runs,
with scripts/eval_gait_insim.py --bound (G-f).

Seeds: one for the pilot; three later for G12 and G13.

Memory budget. The card holds 16,303 MiB, 17.1 GB in the 1e9-byte units torch and the bench
report, and drives the Windows display under WDDM (773 MiB in use at idle, nvidia-smi on
2026-09-28). Anchor: the bench's 12 x 512 encoder at 4 rows of 4,096 packed tokens (16,384)
peaked at 6.81 GB [R bench]. That peak holds the 37.8M-parameter AdamW states and gradients
(0.61 GB) and the twelve 4 x 4,096^2 bool masks bench_mfu keeps on the GPU (0.81 GB), so the
encoder activations are about 5.4 GB. Everything else [est]:

| item | GB |
|---|---|
| parameters, gradients and AdamW states, about 59M x 16 bytes | 0.95 |
| EMA | 0.24 |
| encoder activations at 16,384 tokens (bench peak minus optimizer states and resident masks) | about 5.4 |
| decoder activations, 32,768 knots x 4 layers | 2 to 3 |
| stems, prior, heads, dense knot buffers (at most 512 windows x 1,000 knots x 135 floats, with gradients) and kernel outputs | 0.5 to 1 |
| kinematics with backward, 32,768 points x 28.7 KB at peak [R bench_kinematics] | 0.94 |
| supervision forward kinematics (4.8 KB per knot) | under 0.1 |
| masks, 10 rows of 1,536 squared, bool | 0.02 |
| allocations, sum | about 10.2 to 11.7 |
| CUDA context and allocator reserve (outside max_memory_allocated) | about 1.5 |
| total | about 11.7 to 13.2 |

The render transient (2.68 GB at peak for a 131,072-point chunk [R bench_kinematics]) and the refinement passes run
before the forward pass and are freed before it. The loop asserts
torch.cuda.max_memory_reserved() under 14.5 GB, logs allocated and reserved every 50 steps and
records torch.cuda.mem_get_info() at start; if exceeded, 12,288 tokens per micro-step with 3
micro-steps.

Step time [est; B4.18 replaces it]: encoder about 140 ms per 16,384 useful tokens (116,938
useful tokens/s, measured with 4,096-token bucketed steps [R bench]); kinematics about 50 to 85 ms
[R bench_kinematics]; decoder about 70 ms; rendering about 75 ms amortized;
stems, heads and losses about 50 ms: about 0.5 to 0.6 s per micro-step, about 27,000 to 33,000
useful tokens/s in phase A. A refined micro-step adds 5 decoder and kinematics passes, about
1.2 to 1.6 s. An optimizer step holds 2 micro-steps, so at N = 4 refinement adds 2 x (1.2 to
1.6) / 4 = 0.6 to 0.8 s to the 1.0 to 1.2 s of an optimizer step on average: about 16,000 to
20,500 useful tokens/s in phase B, which is why B4.18 sets N. At these rates, with 25 percent
of the tokens in phase A, 1.0 to 1.5 billion tokens take about 12 to 23 GPU-hours; D§12 budgets
about ten days of calendar for the pilot, which leaves room for dev iterations.

## 6. G1 harness and trivial baseline

Data.
- The 53 lockbox persons of reports/lockbox/lockbox_persons.csv, selected by person id and
  held out by whole archive (section 1.1). The lockbox list is verified against
  reports/lockbox/lockbox.sha256 before anything opens.
- Only mdfm/gates/g1.py passes allow_lockbox=True to amass.load_many (scripts/build_g1_inputs.py
  calls it). GRAB and SOMA members are *_stageii.npz and have no manifest fps: their
  fps comes from the raw npz. Inputs and truth are cached under <store>/store/lockbox/g1_v1
  with a manifest SHA-256; the window store and sampler refuse that path.
- The identical code runs on the dev archives for development. A lockbox read needs
  docs/rules/G1_RULE.md read by the owner and registered with its hash.

Truth.
- prep.amass_to_body_clock (100 Hz, 12 Hz cutoff, y-up, the take's gendered SMPLHBody), belt
  frame for treadmill takes (none in the lockbox archives), forward kinematics of the 22 body
  joints with the true betas and gender; the 15 canonical joints through SMPL_TO_CANONICAL_15.
- Consecutive non-overlapping 10 s windows from t = 0; a last window is kept if it lasts at least
  2 s; takes under 2 s are counted and dropped. Each window in its own window frame.

Inputs, rendered once with seed SHA-256(take_id, window index, config, 'G1v1') by the reference
renderer factory/imu_render.render and camera_render, independent of the training renderer
(which B4.10 tests against it):
- G1a, six IMUs: head, lower_back, wrist_L, wrist_R, shank_L, shank_R; acc and gyro at 100 Hz;
  device model: either imu_render.Device defaults (noise 0.005 g and 0.3 deg/s, bias 0.02 g
  and 1 deg/s) or the per-site levels of reports/g2_renderer/device_noise.json that the
  training renderer uses for xsens13 (question 4); mounts at the xsens13 prior plus 5 degrees
  of isotropic jitter; magnetometer masked; then window_store.band_copy, as for every stream
  the model reads [cfg, rule file].
- G1b, single wrist: wrist_L, the same device and mount draw.
- G1c, single camera: one camera per window from sample_camera (1280 x 720, 30 fps), the default
  tracker process, the canonical 2D normalization.

Inference: encode, refine 20 steps (256 points per stream per step, stopped gradients), decode
all knots. The model uses the take's recorded gender for its skeleton (sex is a covariate,
never an encoder input). G1b predictive: 16 latent samples x 4 decoder draws from pos_logvar =
64 bodies.

Scoring. Each window is scored; windows are averaged per person, frame-weighted; the statistic
is the mean over the 53 persons, so GRAB's 1,340 motions count as 10 persons. Each line reports
its point estimate with a person-clustered bootstrap 95 percent interval (2,000 resamples, seed
in the rule file); the verdict fires on the point estimate [D§9].
- G1a: MPJPE in mm over the 22 body joints at 100 Hz, pelvis-relative, after one rotation about
  the vertical axis per window fitted in closed form by least squares over all joints and frames
  (absolute heading is unobservable without a magnetometer; pitch, roll and relative yaw within
  the window stay scored). Pass at most 45, kill above 70.
- G1b: the true wrist_L position relative to the pelvis, in the pelvis-heading frame of the same
  body at the same frame (the truth in its own frame, each draw in its own frame, so the truth
  never aligns the prediction). Coverage: share of frames whose truth lies inside the 90 percent
  chi-square 3-dof ellipsoid of the moment-matched 64-draw predictive, scaled by the conformal
  factor fitted on the calibration archives. Pass needs coverage at least 0.85 and both
  sharpness bounds of D§9: band width (geometric mean of the three per-axis 90 percent widths)
  at most 1.5 times the six-IMU predictive width of the same wrist on the same windows; energy
  score of the 64 draws at least 20 percent below the population prior's. Kill below 0.70.
- G1c: PA-MPJPE in mm on the 15 canonical joints (the joints the camera sees), per-frame
  similarity Procrustes with scale. Pass at most 60, kill above 90.
- Secondary rows without kill: the other joint set of each line; per archive; walking windows
  only; zero refinement steps (the amortization gap); acc-only wrist; the representation check
  of truth re-rendered and re-scored.

Trivial baseline (stage 0 number, same code, no model), from the train archives only, never
from lockbox, dev or calibration statistics:
- G1a: the mean pelvis-relative pose of the train body store in the heading frame as a constant
  pose, after the same yaw alignment.
- G1c: the same mean pose after per-frame Procrustes.
- G1b: 64 wrist positions drawn once from train windows (pelvis-relative, pelvis-heading frame)
  as the predictive, with its own conformal factor: its coverage, width and energy score; the
  last is the reference of the 20 percent bound.
- A non-trivial reference without kill: analysis by synthesis alone, 200 refinement steps from
  the prior mean with the encoder bypassed, which separates what the kernels identify from what
  the encoder learned.

Protocol and outputs. docs/rules/G1_RULE.md fixes every choice above (sets, seeds, sites,
device, mounts, camera, joint sets, alignment, frames, conformal fit, statistics, inference
mode) before the first read; the hash goes into docs/rules/gate_rule_hashes.csv. One lockbox
read per registered checkpoint, logged in reports/gates/lockbox_reads.csv. Outputs:
reports/gates/g1/<checkpoint_sha256>/g1.json, g1.md, per_person.csv, per_archive.csv, with the
rule, checkpoint, training person-list (asserted disjoint from the 53), input-cache and code
hashes. These are the 60M reference numbers of G13, so harness and rule are frozen after the
first lockbox read.

Harness self-tests (dev archives or analytic cases, never the lockbox): a pure yaw offset gives
MPJPE 0; a pure scale gives PA-MPJPE 0; 30 mm isotropic noise added to the pelvis-relative
positions of the 21 non-root joints gives about 45.7 mm (21 / 22 x 30 x sqrt(8 / pi); the
pelvis-relative score removes pelvis noise, so noise added to world joints would give about
64.6 mm, sqrt(2) x 45.7) and coverage 0.90 plus or minus 0.02 with the right sigma; the
closed-form yaw equals a brute-force search to 0.1 degree; the truth-as-prediction oracle
scores MPJPE 0, PA-MPJPE 0 and energy score 0, and its zero-covariance predictive is reported
as degenerate, never as covered; the population prior runs end to end on 3 dev motions; a
person with 1,000 windows counts once.

## 7. Build order

Each task ends with its tests passing on the full suite (<conda>/envs/mdfm/
python.exe -m pytest -q -p no:cacheprovider tests). Harnesses come before the model (D§5 stage
0); the body path is measured before anything depends on it. Unit tests read at most a few
non-lockbox records or a temporary store, so each task's tests run without the full body or
window store; the full builds of B4.1 (the cache move), B4.4, B4.6 and every run from B4.18 on
need <store> (question 12). The grep tests of B4.1 and B4.22 scan mdfm/ and scripts/; tests/
keeps its refusal and reader tests (section 2.0).

| id | task | files | done when |
|---|---|---|---|
| B4.1 | AMASS index, splits and lockbox guard; move the four test fixtures off ACCAD Female1; move cached lockbox and SMPL-X takes to the gates-only store | mdfm/data/amass_index.py, mdfm/data/amass.py, scripts/build_amass_index.py, tests/test_amass_index.py, tests/test_body.py, tests/test_imu_kernel.py, tests/test_camera_force_kernels.py, tests/test_factory.py, reports/amass_split/ | a locked member is refused before tarfile.open and before a cache read; all 17,154 manifest rows join one to one with both suffixes; reports/amass_split/summary.md lists train, dev, calib, lockbox (53 persons, 74 ids, 2,141 motions) and excluded (7 persons, 1,505 motions) with a SHA-256 per person list; tie pairs on one side; the fixture take of the four test files is asserted to sit in a train archive (KIT), and test_body.py reads its FK reference through canonical.load on a KIT record instead of np.load on the lockbox container amass_ACCAD.npz; lockbox names in tests/ appear only inside refusal tests; a grep test over mdfm/ and scripts/ finds no allow_lockbox=True outside mdfm/gates/; the cache move waits on question 12 |
| B4.2 | Trainable body core: rsqrt Gram-Schmidt, joint-space shape basis, SkeletonBank, rest joints once per subject | mdfm/body/rotations.py, mdfm/body/smplh.py, mdfm/body/trajectory.py, mdfm/factory/imu_render.py, mdfm/kernels/force.py, tests/test_body.py, tests/test_skeleton.py | backward through two nested jvp of rot6d_to_matrix(hermite(y, t)) passes gradcheck in float64; rest from the basis equals the vertex path to 1e-9 for all genders; mixed-gender batch equals per-gender calls; state_dict empty; FK against the MoveAtlas canonical record passes on a KIT take; forward kinematics of 65,536 timestamps with backward (no jvp) under 0.5 GB on the GPU |
| B4.3 | Batched kinematics and its bench | mdfm/body/hermite.py, mdfm/body/kinematics.py, scripts/bench_kinematics.py, reports/bench/bench_kinematics.md, tests/test_kinematics.py | equals BodyTrajectory.point_kinematics to 1e-6 in float64 on a KIT take; gradcheck of backward passes; bf16 raises; the report gives KB per point and forward-plus-backward ms at 4k, 16k, 65k and 131k points, the old-path cost and the no-grad render cost, with torch version and GPU |
| B4.4 | Frames and body store on F: | mdfm/body/frames.py, mdfm/data/body_store.py, scripts/build_body_store.py, tests/test_frames.py, tests/test_body_store.py, reports/body_store/ | yaw-invariance, contact and belt-frame tests pass (treadmill stance speed below 0.1 m/s, IMU unchanged to 1e-5); the store holds float32 shards with no lockbox, dev or calib person in train; reports/body_store/ holds hours per archive and split, the treadmill count, the index hash, shape priors, mean pose and joint-limit percentiles from train persons only |
| B4.5 | Real-source table, pairing, windows, task vocabulary, folds and probes | mdfm/data/sources.py, mdfm/data/windows.py, docs/rules/task_vocabulary.yaml, scripts/probe_stage1_sources.py, scripts/probe_gait120_emg_alignment.py, reports/stage1_sources/, tests/test_sources.py, tests/test_windows.py | every admitted row passes canonical.check without flags; 68 weargait controls and 138 stroke able-bodied persons admitted; torso length coords x scale in 0.35 to 0.65 m on 50 records per pose source; the up axis of each 3D pose source is measured and stroke is rotated to y-up; weargait IMU and insole pairs share n at 100 Hz; the S002 gait120 mapping reproduces frames 456 to 750 and 468 to 679 with frame 575 kept once and each step un-normalized by its own scale; the alignment probe covers every non-lockbox gait120 trial; the stroke plate-to-foot mapping is tabled; no field outside the whitelist survives cut; the 14 weargait dev controls lie in the G2 TEST half; folds hashed per source |
| B4.6 | Window store with band copies | mdfm/data/window_store.py, scripts/build_window_store.py, tests/test_window_store.py, reports/window_store/ | filter, native-timestamp, float16 and EMG-burst tests pass; a read-back of every shard finds no lockbox, EVAL or CODY person; person-list hashes stable across two builds; a lockbox-store path is refused |
| B4.7 | Batched IMU kernel and NLL, nuisance parameterization, mount priors | mdfm/kernels/imu.py, mdfm/kernels/nuisance.py, scripts/build_mount_priors.py, reports/mount_priors/, tests/test_imu_batch.py, tests/test_nuisance.py | imu_predict equals imu_render.render per site to 1e-5; standing reads 1 g and zero gyro; priors built from non-lockbox, non-dev controls only (asserted); per-site vertical axes match reports/weargait_frames within the spread |
| B4.8 | Pose kernels, canonical normalization, camera sampler and tracker | mdfm/kernels/pose.py, mdfm/kernels/camera.py, mdfm/factory/camera_render.py, tests/test_pose_kernels.py, tests/test_camera_render.py | normalize_3d reproduces a non-lockbox MoveAtlas AMASS canonical record to 1e-4; zero residual at truth; closed-form yaw equals brute force to 0.1 degree; dropouts rise with joint speed; seeded determinism |
| B4.9 | Force (Newton floor in body weights, ForceHead), leg inverse dynamics, EMG envelope and monotone decoder | mdfm/kernels/force.py, mdfm/kernels/dynamics.py, mdfm/kernels/emg.py, mdfm/kernels/muscle_actions.yaml, tests/test_force_batch.py, tests/test_dynamics.py, tests/test_emg.py | standing 0.5 BW per foot; ForceHead equals the floor at init; gain invariance; com15 within 3 cm of com() on a KIT walk; standing and pendulum torque tests; EMG monotonicity and gain invariance; every muscle row cites its source |
| B4.10 | Layout families and the GPU synthetic renderer with paired renders | mdfm/factory/layouts.py, mdfm/factory/synth.py, mdfm/factory/imu_render.py, scripts/build_layouts.py, reports/layouts/, tests/test_synth.py | layouts equal real records of their sources; IMU before band_copy equals the reference renderer to 1e-4 g and 1e-2 deg/s, noise off; synthetic and real streams of a layout share rate and decimation after band_copy; every Window validates; token counts 82, 122, 164, 527, 1,094; 262,144 timestamps under 4 GB; the rendered windows per second per layout are recorded |
| B4.11 | Gate commons, conformal, G1 harness dry run on dev with the population prior; G1 rule draft for the owner | mdfm/gates/__init__.py, mdfm/gates/common.py, mdfm/gates/g1.py, mdfm/eval/conformal.py, scripts/gate_g1.py, scripts/build_g1_inputs.py, docs/rules/G1_RULE.md, docs/rules/gate_rule_hashes.csv, tests/test_gates_common.py, tests/test_g1.py | the self-tests of section 6 pass; reports/gates/g1/population_prior_dev/ holds the trivial-baseline table on the dev archives; reports/gates/lockbox_reads.csv exists with no lockbox read; a second read of a registered version is refused; an import-graph test shows mdfm/train and mdfm/model never import mdfm.gates |
| B4.12 | Gait readouts, shank-gyroscope detector and G3 harness dry run on the 14 dev controls; G3 rule draft (OD12) | mdfm/readouts/gait.py, mdfm/gates/g3.py, scripts/gate_g3.py, docs/rules/G3_RULE.md, tests/test_gait_readouts.py, tests/test_g3.py | exact on synthetic strides; truth-body strides equal ankle displacements on dev walks; walkway ICs as model ICs give zero stride error; shank-gyroscope and population-baseline numbers written for the dev controls; both IC references implemented; the offset fit asserts dev person ids; a dev run never reads a lockbox row of events.csv or insole_events.csv (asserted on their lockbox column) |
| B4.13 | Root tracks, G4 harness, Newton and template baselines on dev data; G4 rule draft | mdfm/data/mocap_raw.py, mdfm/gates/g4.py, scripts/gate_g4.py, docs/rules/G4_RULE.md, tests/test_mocap_raw.py, tests/test_g4.py | gait120 TRC root tracks reproduce canonical coords x scale to 1 mm and the pose3d shards of the window store are rebuilt with them; the stroke C3D path works or is marked pending question 6; model-free Newton-floor GRF numbers (force.com15 on canonical pose x scale plus the root track) on the 40 non-lockbox stroke patients and template EMG correlations on the gait120 dev fold are written; the measured envelope as prediction gives r = 1 |
| B4.14 | Tokenizer, rope, typed masks, encoder; the one-way test moves from the bench into tests | mdfm/model/rope.py, mdfm/model/tokenize.py, mdfm/model/masks.py, mdfm/model/encoder.py, tests/test_tokenize.py, tests/test_encoder.py | 100 and 128 Hz tokens within 2 percent; Nyquist mask; relative-time invariance; one-way gradient exactly 0.0 with the mask and non-zero without through 12 layers; 37,829,632 encoder parameters; backend parity within 2e-2 on the GPU; the profiler names the cuDNN attention kernel under set_priority=True |
| B4.15 | Posterior heads, motion and shape priors, time-query decoder, nuisance heads, assembled model, refinement | mdfm/model/posterior.py, mdfm/model/prior.py, mdfm/model/decoder.py, mdfm/model/heads.py, mdfm/model/mfm.py, mdfm/model/refine.py, tests/test_prior.py, tests/test_decoder.py, tests/test_mfm.py, tests/test_refine.py | prior causality and brute-force likelihood tests pass; sparse knots equal dense to 1e-6; init equals the mean pose; outputs float32 under bf16; tiny CPU forward and backward under 5 s; with the body held at its truth knots, a 20 degree mount error refined to under 5 degrees; 57M to 62M parameters; the state_dict audit passes |
| B4.16 | Loss terms and the label-free check | mdfm/train/losses.py, tests/test_losses.py | every term minimal at truth; HSIC activation rule holds; an injected label key trips the check; the no-label grep test passes |
| B4.17 | Sampler, collate, provenance, config and checkpoint | mdfm/data/sampler.py, mdfm/data/collate.py, mdfm/train/provenance.py, mdfm/train/config.py, mdfm/train/checkpoint.py, configs/stage1_pilot.yaml, tests/test_sampler.py, tests/test_collate.py, tests/test_provenance.py | caps (15 percent of all tokens per real source, 512 windows), shares and pairing hold over 1,000 simulated steps; packing and knot coverage exact; every stream-window keeps at least one kernel segment; the encoder input carries no covariate; injected lockbox, EVAL, CODY and canary ids abort; the audit trips on shapedirs, J_template and J_dirs in the model, the EMA or the optimizer state; hashes are read from reports; resume reproduces the next 10 losses on the tiny configuration |
| B4.18 | Training loop, end-to-end bench and GPU smoke run | mdfm/train/loop.py, scripts/train_stage1.py, scripts/bench_stage1.py, reports/bench/stage1_pilot.md, tests/test_loop_smoke.py | the 20-step CPU smoke run lowers the loss; reports/bench/stage1_pilot.md gives tokens/s, time split and peak memory per bucket (target at most 12 GB); row lengths, micro-batch, token budget, step counts and the refinement cadence N are written into configs/stage1_pilot.yaml from it; a 1,000-step GPU run of phase A lowers the loss and writes a clean checkpoint |
| B4.19 | Phase A run (after question 1), G1 dev table, simulator-only G3 dev estimate, then one sealed simulator-only G3 read | configs/stage1_pilot.yaml, <store>/runs/, reports/gates/g1/, reports/gates/g3_sim/ | the phase A checkpoint passes provenance; the G1 dev table is written; the G3-sim dev estimate is logged on the 14 dev controls (no body, signal or renderer statistic of theirs entered phase A) and, as a secondary row, on all 68 non-lockbox controls (54 of them contributed mount-prior statistics and some device-noise and residual statistics, never a window); after the owner confirms and the G3 rule is hashed, one sealed read on the 17 lockbox controls is logged; stride MAPE above 40 percent stops the program and is reported |
| B4.20 | Phase B run, conformal factors on the calibration archives, inference mode chosen on dev, rules hashed | <store>/runs/, docs/rules/gate_rule_hashes.csv | the pilot checkpoint carries hashed person lists; conformal tables are saved with it; G1 dev, G3 dev (14 controls), gait120 dev EMG and stroke dev GRF lines are recorded; G1, G3 and G4 rule hashes are registered |
| B4.21 | Lockbox reads of G1, the G3 healthy half and G4 | reports/gates/g1/, reports/gates/g3/, reports/gates/g4/, reports/gates/lockbox_reads.csv | each line shows point, interval, trivial baseline and verdict (pass, kill, undetermined or not evaluated); each read is logged once; the G1 and G3 numbers are recorded as the 60M reference of G13 |
| B4.22 | G5 harness and read on care_pd (EVAL, harness only); plan and handoff update | mdfm/gates/g5.py, scripts/gate_g5.py, docs/rules/G5_RULE.md, tests/test_g5.py, docs/EXECUTION_PLAN.md, docs/handoffs/ | in mdfm/ and scripts/, allow_eval=True appears only under mdfm/gates (grep test; tests/test_canonical.py keeps its reader tests); identity projections of amass dev walks give zero drop; the QWK drop and the BMCLab separability difference are reported with intervals; B4 status and the next handoff are written |

## 8. Risks

1. Sim-to-real is load-bearing. G2 version 4 reads 0.984 on all 13 sites and 0.9995 with the
   residual spectrum, against a kill line of 0.85 [R g2_renderer/results.json]. No single
   window feature separates the sets beyond 0.88; the top one is ankle_L gyro_x power in 8 to
   20 Hz (single-threshold balanced accuracy 0.8783), then foot and thigh acceleration
   percentiles [R g2_renderer/feature_gaps.md]; in version 4 the rendered 8 to 20 Hz power
   exceeds the real one on those channels, while in version 3 it fell short (shank_L gyro_y,
   0.885 [R g2_renderer/v3_sign_fix/feature_gaps.md]). A simulator-only G3 above 40 percent
   MAPE stops everything [D§9]. Early indicator: the G3-sim dev estimate right after phase A.
   Phase B's real windows and learned nuisances are a mitigation, not a guarantee.
2. The body kernel could set the pace: one batched nested-jvp call costs about 50 ms forward
   and backward up to 16,384 points [R bench_kinematics], against about 140 ms of encoder per
   micro-step; refinement multiplies calls. Mitigation: one call per micro-step, sampled segments, macro-batched
   rendering, a measured refinement cadence, CUDA graphs if B4.18 demands them.
3. Hermite amplification: any bf16 or float16 in the knot path corrupts accelerations by the
   1/h^2 = 1e4 factor. Float32 output path and store, kernels that raise on half precision,
   velocity and acceleration supervision; the IMU NLL of decoded against true bodies on
   synthetic windows is logged.
4. Canonical pose records are root-centred per frame and divided by a torso scale (pelvis
   exactly 0 in gait120 and stroke records [probe B4.5]). The vertical root is inferable from
   contacts only; G4 GRF needs the raw pelvis tracks and so a C3D reader for stroke (question
   6). Without it the G4 GRF line runs on root-centred pose with a contact-inferred root and is
   flagged as that variant.
5. Heading is unobservable from IMUs without a magnetometer. G1 alignment and the G1b frame
   are pre-registered in the rule file before the first read.
6. Lockbox hygiene in code: amass.py has no refusal today; four test files read the lockbox
   person <lockbox person>, and tests/test_body.py also opens the lockbox container
   datasets/socle/amass/processed/canonical/amass_ACCAD.npz with np.load, outside the reader;
   list_sequences misses *_stageii.npz (1,409 of the 2,141 lockbox motions); cache/amass holds
   lockbox takes; the amass cache files do not record surface_model; the mount prior must also
   leave out the 14 G3 dev controls, not only the lockbox; reports/weargait_targets/events.csv
   holds the weargait lockbox trials flagged in a column. B4.1, B4.7 and B4.12 fix all seven,
   and grep, import-graph and audit tests keep them fixed.
7. Treadmill takes (412 in BMLrub [probe B4.4]) teach near-zero pelvis translation under
   walking IMU patterns unless moved to the belt frame (B4.4). Indicator: synthetic stride error
   on dev walks.
8. Initial contact: the 12 Hz bodies carry no heel-strike impact, and walkway against insole
   contacts differ by a median 30 ms, p90 80 ms (10 and 40 ms after a per-trial offset)
   [R weargait_targets/summary.md], the size of the pass line itself (OD12). Indicator: dev IC
   error against the shank-gyroscope detector after phase B.
9. Stride at 5 cm is about 5 percent of the SelfPace control median stride of 106.8 cm
   [R weargait_targets/summary.md]. Indicator: synthetic stride MAE in the xsens13 layout must
   sit well below 5 cm before any real read.
10. Calibration and coverage shift: G1b rests on the posterior plus a conformal factor fitted on
    calibration archives, while the lockbox is rich in hand-object motion (GRAB 1,340 and
    TCDHands 62 of its 2,141 motions [probe B4.1]); exchangeability does not hold exactly.
    Indicators: dev coverage with factors fitted on calibration, the number of active latent
    dimensions, per-archive rows.
11. Units, frames and pairing: 513 of 529 non-lockbox stroke EMG records carry an amplitude
    flag (record p99 14.5 to 991 mV [probe B4.5]); the weargait insole unit is not stated; the
    stroke canonical 3D pose is z-up while gait120 and amass are y-up; the stroke
    plate-to-foot mapping and unloaded offset are unmeasured (one record read in review stays
    above 1 N on foot_L throughout); the gait120 pose to EMG mapping is verified on one trial
    until the B4.5 probe covers all. Gains absorb scale, not timing or axes.
12. Real data are thinner than synthetic: gait120 and stroke able-bodied hold about 2.4 hours of
    pose together (2.00 and 0.43, non-lockbox [probe B4.5]), so their windows repeat many
    times in phase B. Indicator: dev kernel NLL rising while train NLL falls, per source.
13. The pilot may plateau: comparable encoders sit at 4M to 85M parameters and several plateau
    there [A§7]. G13 then keeps the program at 60M, an outcome and not a failure.
14. Tooling: <store> does not exist yet (F: is NTFS with 2,540 GiB free; checked in review)
    and is not in the session's allowed directories (handoffs 3 and 4); fused AdamW is
    unverified on this build; spawned workers need picklable, memory-mapped datasets; no C3D
    reader is installed in the mdfm environment (pip list: no ezc3d, no c3d); the GPU also
    drives the Windows display (WDDM), so free memory varies with the desktop.
15. Schedule: 22 tasks for days 30 to 60. If time runs short, the cut order is the
    analysis-by-synthesis reference row, then the stroke metric-3D round trip, then the G5 read
    (still before B6), never a gate's kill line.
16. The G2 report prints the with-residual balanced accuracy on its without-residual line
    (scripts/g2_renderer_realism.py writes `ba` on both); OD9 must be decided on results.json,
    and the script's summary fixed and re-run before any G2 number is quoted again.

## 9. Open questions for the owner

1. G2 comes "before any pretraining" [D§9], G2 version 4 reads 0.984 (0.9995 with the residual
   spectrum) and OD9 is open. Proposal: build B4.1 to B4.18 now; allow a phase A run whose only
   outputs are dev numbers (G1 dev and the G3-sim dev estimate on the 14 dev controls, with the
   68 non-lockbox controls as a secondary row) to inform the B2 renderer loop; hold the sealed
   G3-sim read and phase B until G2 passes or you waive it. Part of the same decision: the 60
   percent synthetic cap of D§5 read cumulatively (phase A at 100 percent, 0.5875 overall) or
   per step (no synthetic-only phase A; the pilot would start with the mixture of phase B).
   Answered by the owner on 2026-09-28 (OD9, the proposal): the first half of G2 is a reported
   diagnostic; synthetic data are admitted by utility, the simulator-only G3 line scored on
   non-lockbox controls (the 14 dev controls, the 68 as a secondary row) after phase A, and
   later by the second half of G2. Phase A may run once B4.18 passes. A simulator-only test
   needs a synthetic-only phase, so the 60 percent cap is read cumulatively; the sealed
   lockbox read of G3 stays for B4.21.
2. The simulator-only G3 variant: this spec reads it as a model trained on the simulator only
   and read on the real weargait lockbox controls (sim-to-real, D§11). Confirm before the G3
   rule is hashed.
3. OD12, the G3 initial-contact reference: walkway contact (proposal) or insole with a per-trial
   offset removed.
4. G1 rule, before hashing: the six sites; wrist_L with acc and gyro at 100 Hz; 22 joints for
   G1a and 15 canonical joints for G1c; one yaw per window; the G1b frame and the conformal
   ellipsoid; width and energy-score forms; device (the Device defaults or the weargait levels
   of device_noise.json that training uses for xsens13), mount and camera distributions;
   person weighting; the recorded gender in the model's skeleton; 20 refinement steps as the
   primary.
5. AMASS folds: dev = EyesJapanDataset, HDM05, SFU (23 persons); calibration = DanceDB, EKUT,
   HUMAN4D, HumanEva (27 persons); train keeps 360 persons and 33.45 hours [probe B4.1]. They
   must be declared before any training.
6. C3D reader for the stroke root tracks and body mass: add a PyPI package (ezc3d or c3d) to
   env/requirements.txt, or write an in-tree reader of about 150 lines? Environment changes are
   yours.
7. G4 percent-body-weight normalization on the stroke lockbox: body mass from the C3D
   parameter when present (gait120 carries mass in Anthropometric_Data_Participants.xlsx
   [probe B4.5]; the first sheets of the stroke xlsx files show none), or the mean force over
   complete steps?
8. The pilot's motion prior leaves out age band (section 1.5, item 3). Accept, or condition on
   age band with 50 percent dropout?
9. EMG trains its decoder on a detached body in the pilot (section 1.5, item 2). Accept, or let
   EMG gradients reach the body?
10. gait120 and stroke through the 3D pose kernel with per-source offsets instead of fitted
    bodies (section 1.5, item 4); marker-level fits at B6. Accept?
11. Source scope: capture24, ltmm, vidimu, motion_x and the force-only controls enter only at
    200M (section 1.3). Accept?
Adopted by the session on 2026-09-29 under the owner's carte blanche for the night, each
revisable and listed in docs/OWNER_ACTIONS.md: question 2, the spec's reading (the sealed
simulator-only read waits on a hashed G3 rule; the admission test of OD9 is the dev
estimate on non-lockbox controls); question 3, the walkway contact as the provisional
reference, OD12 staying open; question 4, the listed G1 defaults, the G1 rule unhashed;
question 5, the proposed AMASS folds; question 6, an in-tree C3D reader (no package
installed); question 7, body mass from the C3D parameter when present, else from the
stroke MAT file (sub_char Weight, found by B4.5), else the mean force over complete steps;
questions 8 to 11, the departures as written.

12. Owner action, pending since handoff 3: create <store> and add "additionalDirectories":
    ["<workdir>/data_cache", "<store>"] to the local settings of the working sessions before B4.1 moves the
    cached lockbox takes and B4.4 writes the body store.
13. The latent prior as an autoregressive Gaussian instead of a flow, and refinement during
    training only in phase B every N-th step (section 1.5, items 6 and 7). Accept?
