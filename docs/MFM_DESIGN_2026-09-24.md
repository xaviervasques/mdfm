# A measured-body foundation model of abnormal human movement: design from scratch

Date: 2026-09-24. Status: design proposal, owner review pending. English, no em-dashes.
Data scope: section 16 (owner decision, 2026-09-24): MoveAtlas data only, published by
MoveAtlas or held on disk with a right to use; anything else is outside the program.
Provenance: section 15. Data facts: the MoveAtlas corpus at commit a5d3814 as measured in
reports/ (2026-09-23) and as checked on disk by the verification passes of section 15;
facts measured on 2026-09-28 (reports/) run against MoveAtlas f2e6f99, the owner's commit
of the registry corrections, and are dated where they enter.
This document owes nothing to KINEFM_V2_SPEC.md or to the mission prompt's claims; it
keeps only the owner's standing rules (CODY evaluation-only, patient-level splits, raw
video never leaves the disk, licence discipline, MoveAtlas data only).

## 0. Thesis

The model is one anonymous 3D body, inferred as a posterior trajectory from whatever was
recorded, with every MoveAtlas contract read as a physical measurement of that body
through a differentiable sensor kernel whose only site-specific parts are per-record
nuisances (camera, placement, gain, rate, tracker). A healthy prior over the body's
motion, conditioned on age, task and shape, is learned at scale on the healthy socle and
on virtual sensors rendered from it. Pathology is not a learned latent and not a class:
it is a short list of unit-bearing descriptors computed by fixed formulas from the body
posterior against the healthy norm, organized in named families that clinicians already
use (oscillation, sustained posture, irregular flow, amplitude and speed gains with
decrement, transition failure), each with a credible interval, a measured-or-inferred
status, a bandwidth statement and an abstention rule. Interventions (levodopa, DBS) are
low-capacity, site-and-channel-masked, monotone displacements of those descriptors with
explicit kinetics, from which the same patient's counterfactual body is resampled and
re-rendered into the sensor the clinic used, and they are falsified within subject on
data the model never trains on.

What makes it extraordinary, and what each claim is tested on:

1. Devices become interchangeable by construction. A phone video, a smartwatch, an insole
   and a 13-sensor suit report the same quantities on the same body, with intervals that
   widen when the sensor is poorer. Tested on weargait_pd (13 IMUs against walkway and
   insoles, same trials), stroke gait (pose against force plates and EMG), and on
   the first real video-versus-IMU pair on patients, fog_turning_filho (Videos.zip fetched
   and verified on 2026-09-24; 71 video and IMU turning pairs of 35 PD), never trained,
   keypointed locally once the dedicated environment exists (gate G14).
2. Scales are linked through physics, not through score regression. Tremor in hertz and
   degrees, stride in metres, decrement in percent per cycle, dwell time in seconds; scale
   items are monotone maps from named descriptors with the anchor table shown, and two
   scales that share a descriptor are compared on that axis and never converted.
3. The model knows what it cannot measure. Rigidity, speech and face return not measured.
   Axial rotations from a single camera return prior-only. Millisecond events from 30 fps
   video are not claimed. Every readout carries a status and a bandwidth.
4. Counterfactual bodies with a declared locus and kinetics, refused outside support,
   falsified on never-trained within-subject data (care_pd BMCLab ON and OFF walks,
   tremordb DBS on, off and washout; pd_biostamprc21 medication windows, admitted as
   EVAL on 2026-09-24).
5. Every claim has a gate with a kill number on data on disk, including a no-body
   baseline at equal compute that demotes the body if it does not earn its keep.

What it is not: a disease classifier (diagnosis is at most a within-source differential),
a video generator (the body is anonymous, never the patient's pixels), a brain model (EEG
and LFP are not measurements of the body and stay out of version 1), and a mechanistic
controller (a latent motor controller with named perturbation parameters was designed and
judged unidentifiable from a single-visit corpus; it stays a research lane, section 10).

## 1. Representation

- Body trajectory q(t): root translation and orientation, 21 body-joint rotations in 6D
  form, foot contact probabilities, on a 100 Hz internal clock with cubic Hermite
  interpolation, so that pose, velocity and acceleration exist analytically at any native
  sensor timestamp. No hands, no face in version 1. Every joint carries a log-variance and
  a status derived from posterior contraction: measured, inferred, prior only.
- Subject: shape and anthropometry (segment lengths, stature; internally the 16 SMPL+H
  coefficients), sex and age band as covariates of the priors and of the norm, never as
  encoder inputs. Subject variables are pooled across records by a set aggregator; windows
  of one person share them through a consistency loss, never through identity contrast.
- Session: an intervention covariate vector u (medication ON, OFF or unknown; minutes since
  dose; DBS target and state) and a 16-dimensional session residual trained only on
  within-subject contrasts. Etiology never enters any encoder or prior.
- Record nuisance: camera intrinsics and pose, tracker identity, IMU placement and
  orientation on the segment, device rate and filter, gains and units, insole layout,
  clothing dilation, magnetometer field. Consumed by the sensor kernels and by nothing
  else.
- Norm and deviation: a healthy norm over descriptors and short-horizon kinematics,
  conditioned on age band, task and shape, learned on healthy sources. Deviation is
  measured against that marginal norm, never against the subject's own recent history: a
  causal predictor absorbs sustained and scaled pathology (bradykinesia, dystonic posture,
  hypokinetic gait) into its context and reports nothing. A 128-dimensional deviation
  embedding r, trained with descriptor-similarity targets in site-balanced batches, serves
  retrieval, probes and calibrators; it is never the source of a unit-bearing number.
- Clocks: native samples; 10 s windows in a 30 s context; session (minutes since dose);
  visit (weeks). A free-living day is consecutive windows sharing subject and session
  variables, never one latent.

The named descriptor families (section 6) are the design's answer to the perturbation
idea: the same physiology that a controller model would parameterize (oscillator,
tonic bias with co-contraction, stochastic flow, gains and decrement, transition hazard)
is measured as formulas on the body, with the mechanism left open.

## 2. The 3D body output

- Parameterization inside the training loop: SMPL+H (male, female, neutral, 16 shape
  parameters, 22 joints with root) for forward kinematics, skinning, silhouette
  rasterization, marker kernels and segment inertia.
- Released rig: MA-22, an open 22-joint tree whose joint-centre definitions, rest pose and
  a shape space are distilled from CC0 and CC BY marker mocap (gait120,
  stroke_gait_vancriekinge) and from silhouette proportions. MA-22 is the native
  parameterization of the released decoder: segment lengths, joint centres and a
  distilled 10-coefficient shape space are regressed directly, the MA-22 body must stay
  within 5 mm MPJPE of the internal body on G1, no SMPL+H joint regressor, blend shapes or shape basis are present
  in any released checkpoint, and a weight audit hashes every array of a release against
  the SMPL+H model arrays. A user with their own SMPL+H licence obtains a mesh through an
  adapter that runs on their model at inference. The SMPL+H LICENSE on disk forbids
  reproduction, modification and any third-party availability of the model and any
  training for commercial use; this is the only path that respects it.
- Output at any rate: joint rotations, joint centres in metres, segment lengths, contacts,
  16 posterior samples with per-joint 90 percent bands, per-part status, and the
  descriptor vector.
- Anthropometry: inferred from 3D and silhouette sources; 2D-only sources give segment
  ratios with metric scale from a reference object or the age-and-sex prior; IMU-only
  inputs receive the prior mean, and every length-dependent readout then carries the prior
  width and a shape-inferred flag. Silhouettes give proportions only (the contract
  normalizes scale per recording). Children use the adult body with a widened shape prior
  and a paediatric flag; infants are excluded in version 1 (no licensed infant body).
- What a single wrist gives: a measured wrist trajectory, an inferred arm, a prior-only
  trunk and legs. The body is still rendered, with the status shown, because the common
  coordinate system is what makes readouts comparable; but no length-dependent or
  leg-dependent readout is emitted as measured from a wrist.
- Validation without paired pathological pose and IMU on disk: (a) synthetic round trips on
  held-out amass subjects (G1); (b) real cross-sensor checks on the pairs that exist:
  weargait_pd 13 IMUs against walkway metrics and insole events, stroke gait pose against
  force plates and EMG timing, gait120 pose against EMG, vidimu orientation pairs on 16
  healthy subjects (G3, G4); (c) the only pathological metric-3D round trip on disk, the
  stroke lockbox with true marker mocap; (d) care_pd pathology preservation, label-based:
  bodies re-estimated from synthetic projections of the care_pd walks must keep the
  UPDRS_GAIT probe and the ON versus OFF separability (G5), with joint error against the
  care_pd fits reported only as a diagnostic, because those fits are themselves a
  monocular lifter's output; (e) same-subject consistency across the two CODY
  extractions of the same 25 adults and across the two walking directions of koa_pd_nm
  (G6); (f) a posterior predictive check on every record that re-renders the body into the
  recorded sensor and abstains when the residual leaves the calibration band; (g) the real
  patient pair, fog_turning_filho Videos.zip (CC BY 4.0, the video part of a published
  MoveAtlas dataset; fetched on 2026-09-24, size and MD5 matching the figshare record; 73
  videos of 35 PD, lower limbs only, 71 of them paired with an IMU turning trial):
  keypointed locally once the dedicated environment exists, video versus IMU on the same
  trials (G14).

## 3. Sensor models per modality

Physics with small learned residuals, evaluated at native timestamps, with per-record
nuisance and presence masks. Every kernel likelihood is defined on the band the body can
carry (low-pass at 40 Hz for IMU, 20 ms envelopes for EMG); the IMU 20 to 250 Hz band and
the EMG burst structure travel as measurement tokens beside the kinematic tokens so that
brief events survive the body bottleneck without pretending the body explains them.

- Pose 2D (15 canonical joints; li_taati, modys_video, remap_open 2D
  streams; CODY evaluation only): projection under a per-record camera with source-family
  priors (the CODY exam is a fixed tripod); Student-t noise scaled by tracker confidence;
  a missing-not-at-random detection model p(joint missing | joint speed, pose atypicality,
  occlusion proxy) fitted per tracker on modys_video, li_taati and
  remap_open validity masks and on synthetic re-tracking, never on CODY, so that tracker
  failure is evidence rather than a lost frame. Depth stays in the posterior; no lifting.
  Per readout, the degrees of freedom observable from one view are declared (image-plane
  flexion and abduction); axial rotations are prior-only unless two views exist.
- Pose 3D: joint likelihood after per-source joint-definition offsets learned as nuisance;
  ntu_rgbd is a noisy measurement, not truth (17,073 of its records sit outside the
  thigh-to-torso anthropometric band); motion_x is likewise a noisy joint-position
  measurement, never rotation or shape supervision (most of it is monocular mesh recovery,
  and its mocap subsets duplicate amass).
- IMU (14 canonical sites, 9 channels, 31.25 to 500 Hz): a sensor on segment k at offset r
  with orientation R_s measures acc = R_s^T R_k^T (p_ddot_k + omega_dot x r + omega x
  (omega x r) - g) + bias and gyro = R_s^T omega_k + bias; the magnetometer is a heading
  observation or masked; the acc_content flag switches gravity off for gravity-removed
  sources (pads); a device model carries rate, filter, saturation and units; placement is
  nuisance with a per-site prior; a soft-tissue filter of fixed small capacity is the only
  learned part, with a documented gain limit.
- Force: per-foot vertical total from whole-body momentum balance with contact allocation
  from foot vertex height and velocity; the Newtonian centre-of-mass floor m(g + a_com)
  split by contact is the baseline any learned term must beat; the 8 insole positions come
  from a learned head; totals-only sources read the sum; uncalibrated sources (gaitndd,
  physionet_multimodal_gait) carry a per-record affine gain and shape-only likelihoods.
- Silhouette (64-point contour plus 8 scalars; koa_pd_nm internal, cody_3 evaluation
  only): soft rasterization of the mesh under the camera nuisance with a clothing dilation
  latent, through the exporter's own normalization; Chamfer likelihood on the contour and
  Gaussian on the scalars.
- EMG (500 to 2000 Hz): no credible forward physics for raw waveforms; a learned decoder
  maps kinematics and inverse-dynamics torques through monotone moment-arm maps to
  per-muscle log-envelopes with a co-contraction term; likelihood on 20 ms envelopes and
  burst onsets; a per-record log-gain nuisance.
- Laser-Doppler velocity (tremordb): hand velocity along an unknown beam vector, which
  identifies tremor frequency and relative amplitude only, never centimetres.
- EEG and LFP: not measurements of the body, no kernel in version 1 (section 10).

## 4. Architecture

About 200M parameters after a 60M pilot passes the body gates locally.

- Tokenizers (about 15M): per-modality convolutional stems producing one token per channel
  group per 0.25 s patch (0.1 s for EMG) from the native samples inside the patch,
  continuous-time rotary positions, channel identity embeddings from the contracts,
  presence masks, and the nuisance descriptor as a separate token visible only to the
  kernel heads. Measurement tokens as in section 3.
- Local encoder (about 85M): 12 layers, width 768, 10 s windows, all modalities of a record
  in one sequence, with a typed attention mask that forbids body positions from attending
  to nuisance tokens.
- Body decoder (about 40M): time-query heads emitting q(t), variances and contacts at any
  t, plus a flow head over 2 s segments conditioned on subject, session and task variables,
  used for posterior sampling and for counterfactual bodies.
- Session aggregator (about 25M): a light recurrent module over window summaries
  producing the session residual and the effect displacement from the parametric
  intervention kinetics.
- Heads (about 10M): fixed descriptor formulas, isotonic scale-item calibrators, conformal
  intervals fitted on a calibration fold carved from the training portion, the deviation
  embedding r, the nuisance heads.
- Losses: negative log-likelihood of every observed stream under its kernel on its band;
  direct supervision of rotations and shape where a body is given (amass) or fitted
  (gait120, stroke mocap); KL to age-and-sex-conditioned priors; nuisance consistency (one
  synthetic body rendered under resampled nuisances must give one body); physical
  plausibility (joint limits, bone-length constancy, foot penetration, momentum balance
  where force exists); subject consistency; an HSIC penalty between body-state summaries
  and source identity, active only in diagnosis-and-task strata that hold at least two
  sources (listed in the nuisance table; single-source diagnoses are marked site
  invariance untestable); the descriptor-similarity metric loss for r. No adversarial
  site head, no masked reconstruction of raw sensors, no label in any backbone loss.
- Inference: an amortized posterior refined by a few gradient steps through the kernels
  (5 during training, 20 at evaluation).

## 5. Data tiers, splits and training recipe

Tiers, hashed on day one (frozen on 2026-09-28 by scripts/build_tiers_and_lockbox.py,
reports/manifest_tiers/; every canonical record carries a tier, a person id and a lane,
body or neural):
- EVAL, never trained in any form: cody_1, cody_2, cody_3 (one merged registry: the 25
  adults of cody_1 and cody_3 dataset_0 are the same people, cody_2 inference equals cody_3
  dataset_2; the denylist covers the adult, ext_ds and PEDIA namespaces), care_pd (CC
  BY-NC-ND; its probes are internal evaluation instruments, never released, never used to
  set thresholds of released heads), tremordb, the hubu_fis_fingertap ratings,
  fog_turning_filho (on disk: its shank IMU with per-sample freezing flags, which are
  specialist video labels, its session-level freezing table and, fetched and verified on
  2026-09-24, its Videos.zip; the cross-device freezing set and the G14 pairs), and,
  admitted on 2026-09-24 (docs/decisions/licence_decisions_2026-09-24.csv),
  pd_biostamprc21 (CC BY 4.0 per DataCite and the IEEE DataPort terms; medication and
  item windows ruled evaluation-only) and kaggle_fog (the Kaggle rules allow any use and
  forbid redistribution of the Kaggle copy; any redistribution goes through the sponsor's
  CC BY 4.0 copy on Zenodo, 10.5281/zenodo.10959560; its CONTRAST half trains only the
  INTERNAL comparison session head of stage 4).
- INTERNAL, trained locally only, never released and never uploaded to a rented cluster:
  ntu_rgbd (portal agreement), koa_pd_nm (research-only); admitted on 2026-09-24 for
  internal research only (no data, derived data or weights released; results reported
  with citation): goodwin_smm (no licence, citation request), asdpose (data sharing
  statement, no explicit data licence), osf_fmd_eeg and osf_fmd_kinematics (no licence on
  the OSF nodes; patient data stays local, aggregate results only); mini_rgbd (CC
  BY-NC-SA 4.0, admitted on 2026-09-24; unused in version 1 because infants are excluded;
  its SMIL model files never shipped).
- EXCLUDED_V1: infant_pose_chambers (infants are excluded in version 1, section 2).
- RELEASE: everything else except panelapp (Genomics England terms, non-commercial, all
  rights reserved), a reference only and never inside a released artefact; ninapro (CC0
  on Dryad) enters a derived release only once its held zips, which come from the project
  site, are recorded as equivalent to the Dryad copy; released weights inherit CC
  BY-NC-SA 4.0 from amass, motion_x and pads; the non-commercial admissions (pads,
  mini_rgbd, panelapp) hold while MD_FM stays non-commercial research; the BY-SA source oxford_dystonia_lfp never trains released weights; for amass and motion_x on an external cluster, written confirmation
  that derived 15-joint tensors under the licensee's control on rented compute are
  permitted, otherwise the factory's synthetic renders substitute.

Splits: patient level everywhere, keyed on a re-keyed person id (mapped on 2026-09-28 for
32,609 canonical subject ids, no person shared between two sources, reports/person_ids/):
li_taati (390 subject ids, the trial numbers of 9 persons), modys_video (94 manifest
records keyed per video for 34 children,
the distinct IDs of data_clinical_scoring.xlsx; 95 videos, T0, T3 and T12 visits for 29
children, every visit of a child on one side of every split) and monipar (46 records for
6 persons, the supervised subgroup in the canonical manifest) are collapsed before any
split, as are amass (503 ids of 470 persons) and cap_sleep (35 ids of 34 persons, two
byte-identical EDF files); both hubu_fis hands
share one id; koa_pd_nm sequences 01 and 02 never separate; motion_x has no subject ids
and enters no subject-level metric. A LOCKBOX of 20 percent of the patients of every
labelled pathology source with at least 30 patients is frozen on day one and never seen by
any backbone stage; every clinical gate is reported on it with a single split. modys_video,
at 34 children, clears that line by 4: its lockbox is 7 children with all their visits and
its training portion 27; the person map of 2026-09-28 confirms 34 children. Lockbox
v1.2, frozen on 2026-09-28 (scripts/build_tiers_and_lockbox.py, reports/lockbox/), holds
871 persons: the labelled pathology sources above, the G1 amass hold-out (53 persons in
whole subsets), the G4 gait120 hold-out (24) and the neural-lane sources cap_sleep,
pd_rest_eeg_ds004584, pd_walking_eeg_ds007526, physionet_multimodal_gait and seizeit2,
drawn per group with seeds derived from the source names and hashed with SHA-256. Sources
with fewer than 30 patients (li_taati with its 9 persons,
daphnet, multimodal_fog, sensefog, fog_star, remap_open, monipar, hd_accel_jnet with its
24 HD and 10 healthy; goodwin_smm, internal research only) stay out of the backbone, have no
lockbox, and serve stage-3 and stage-4 probes with 5-fold patient-level cross-fitting on
the frozen backbone, reported without kill. fog_turning_filho (35 PD in PDFEinfo.csv) is
EVAL, never trained: its IMU serves the cross-device freezing check and, with its
video (fetched and verified on 2026-09-24), G14 as the only real video-versus-IMU pair on patients. CODY exclusion is mechanical: the manifest builder rejects the CODY
namespaces, the training subject list is hashed into every checkpoint, the CODY loader
exists only in the evaluation harness, and a canary record is checked after every run.
No statistic of any kind (tracker noise, normalization, threshold) is computed on CODY.

Stage 0, harness and factory (no training): every gate scripted with its kill number and
its trivial-baseline number (population prior; per-source logistic regression on
handcrafted features); kernels unit-tested; virtual sensors rendered from the amass bodies
(14-site IMU under device models imitating the real layouts, 8 cameras through the
tracker-failure process, silhouettes, plate and insole totals); motion_x deduplicated
against amass by subset name and by sequence hash before any hold-out; frame rates
recovered for modys_video and li_taati (modys_video measured at 25 frames per
second on 2026-09-28 from the source's stick-figure movies, reports/nuisance/; li_taati
at 30 frames per second as stated by its data paper, PMC6219082, since no rate is
recoverable on disk); the pd_biostamprc21 group and placement
table recovered from the source (pd_biostamprc21 admitted as EVAL on 2026-09-24; groups
recovered on 2026-09-28: 17 PD and 17 controls); care_pd medication state parsed from the
raw walk names into the evaluation manifest (done on 2026-09-28: BMCLab 19 subjects with
both ON and OFF walks, E-LC 17).

Stage 1, healthy body prior and kernel calibration (about 250 H100-hours). Bodies with
truth: amass (17,154 records, 503 subject ids of 470 persons, SMPL+H parameters from the
raw npz, joints recomputed from them and never read from the canonical records, 404 of
which were not rigid skeletons until the MoveAtlas rebuild of 2026-09-29, reports/probe_amass_canonical_rigidity/before_rebuild_f2e6f99/); fitted
bodies: gait120 (120 subjects, 39 markers), stroke_gait_vancriekinge able-bodied (138).
Noisy 3D joint measurements: motion_x (after deduplication), ntu_rgbd (internal tier).
Healthy partial-state sources: vidimu (53; 208 orientation pairs on 16), the free
acceleration of physionet_multimodal_gait (57), nonan_gaitprint (41) and
nonan_gaitprint_young (35), sixmwt_lifespan (60), capture24 (151, capped at 2 h per
subject per epoch), ltmm non-fallers (42, capped), gaitrec healthy controls (force),
drummer_tremor_model (18, after ingestion of its C3D kinematics), physionet_semg_walking
(31), the control arms of weargait_pd (85), physionet_gaitpdb (72) and gaitndd (16), each
source minus its lockbox v1.2 persons (amass 53, gait120 24, physionet_multimodal_gait 12
among them). Synthetic tokens capped at 60 percent of the mixture. The age norm for gait and
posture descriptors is built from the age-bearing sources (weargait_pd controls, nonan,
sixmwt, ltmm, gaitrec healthy, stroke able-bodied, gait120, capture24 bands); amass and
motion_x carry no age and serve the kinematic prior only; deviation abstains for age bands
with fewer than 20 healthy subjects.

Stage 2, pathological analysis-by-synthesis with 30 percent healthy replay, no labels
(about 350 H100-hours), training portions after lockbox: pads (375 of 469), weargait_pd
(148 of 185), physionet_gaitpdb (132 of 165), gaitndd (51 of 64), clinical_gait_multipatho
(208 of 260), imu_inthewild_iprognosis (36 of 45),
modys_video (27 of 34 children), stroke_gait_vancriekinge stroke (40 of 50), gaitrec
orthopaedic patients (1,667 of 2,084) as musculoskeletal negatives; INTERNAL adds
koa_pd_nm (77 of 96), locally. Sampling: subject-balanced, source
weight proportional to the square root of subject-hours, 15 percent per-source batch cap.
Synthetic pathology (procedural perturbations of healthy bodies) serves kernel unit tests,
descriptor-formula tests and robustness only, never the deviation embedding or any
readout.

Stage 3, calibration heads on the frozen backbone (probes, never backpropagated), trained
on the training portions or by cross-fitting, reported on the lockbox:
modys_video DIS, weargait_pd MDS-UPDRS-III items 3.10 to 3.13 and Hoehn and Yahr, freezing
episodes (fog_star, daphnet, multimodal_fog, sensefog by cross-fitting; fog_turning_filho
never), hd_accel_jnet Huntington scores and li_taati UDysRS (9 persons) by cross-fitting,
monipar tremor labels, within-source differential on pads and gaitndd. cdystonia and
dystonia_alcohol_bfm enter no stage (section 7).

Stage 4, effect operator (under 50 free parameters plus per-subject, per-channel gains):
levodopa fitted on the 59 weargait_pd PD subjects without DBS (time since last dose; dose
amount from the levodopa-equivalent daily dose, computed on 2026-09-28 for the 69 PD
with a legible regimen, median 880 mg/day), on
li_taati take order as an ordinal within-subject dose axis. kaggle_fog, admitted as EVAL
on 2026-09-24: a CONTRAST half of kaggle_fog (patient-level split) trains only an
INTERNAL comparison session head on the frozen backbone, never the backbone and never a
released head, while the other half stays EVAL (ON versus OFF with per-visit UPDRS-III);
the released session head rests on weargait_pd and li_taati alone. DBS keeps STN and GPi
masks with a population prior and no fitted kinetics; Vim is
removed from the operator. No toxin branch (section 7). No CODY at any stage.

## 6. Clinical readouts

The clinician receives the moving body from any angle (never the patient's pixels), with
per-part status and bands, then readouts in SI units with 90 percent conformal intervals,
each with a bandwidth statement (30 fps video carries nothing above 12 Hz or shorter than
60 ms; a 31.25 Hz IMU cannot see 12 Hz tremor) and an abstention flag when the limb was
not observed, the task did not elicit the item, or the posterior predictive check failed.

- Tremor, per limb independently (no shared frequency; asymmetry is physiology):
  dominant frequency, amplitude (degrees at the joint, centimetres at the hand where
  metric scale is measured), rhythmicity, and the condition (rest, postural, kinetic)
  inferred from the body itself (limb at rest against support, held against gravity, or
  moving) with an abstain state, because free-living records carry no task token; on pads
  a load-sensitivity descriptor (StretchHold versus HoldWeight), since central tremor does
  not shift with load.
- Bradykinesia: amplitude gain and speed gain relative to the age norm, per-cycle
  decrement, hesitations per minute; the sequence effect reported separately and not
  assumed levodopa-responsive.
- Finger tapping: from locally extracted 2D hand keypoints through a dedicated thumb-index
  aperture measurement (amplitude in hand-length units, speed, decrement, halts), outside
  the body rig and declared so; tested on hubu_fis_fingertap (118 subjects, 234 clips).
- Gait and posture: speed, stride length, cadence, variability, double support, arm-swing
  amplitude and asymmetry, trunk and neck angles, turning duration, freezing episodes
  (onset, duration, halt probability).
- Irregular flow (dyskinesia and chorea-like): irregularity of joint-velocity direction over
  0.5 to 3 Hz, energy, spatial roaming; calibrated to UDysRS (li_taati) and to the DIS
  choreoathetosis subscore (modys_video).
- Sustained abnormal posture (the channel is not called dystonia until a dystonia-specific
  test passes): sustained deviation from the task norm in degrees, dwell time, overflow,
  laterality, task dependence; calibrated to the DIS dystonia subscore; marked inferred,
  not measured, when only one task is observed; a null on stroke posture and knee
  osteoarthritis contracture (G9).
- Myoclonus: event count and amplitude from any sensor; duration only from sensors at
  100 Hz or more; no training exemplar, so no gate and no claim.
- Tics, ballismus, stereotypies, athetosis: declared absent (no training source in scope
  outside CODY for tics, ballismus and athetosis; the stereotypy sources goodwin_smm and
  asdpose were admitted on 2026-09-24 for internal research only: goodwin_smm feeds an
  INTERNAL stereotypy probe by cross-fitting, asdpose (not ingested) an internal
  stereotypy analysis in the addendum, and no released readout covers stereotypy).
  Rigidity, speech, face: not measured.
- Paediatric sources: no healthy child norm exists in any tier, so deviation for
  modys_video is computed within source against its lowest-DIS quartile and flagged as a
  within-source norm; cody_2 children get no degree-valued readout in this program, because
  no healthy child norm exists in scope (shanghai_child_gait3d is not held; leuven_cp_gait,
  spastic cerebral palsy curves on disk, admitted on 2026-09-24 under CC0, is a patient
  reference, not a norm).
- Scale items: monotone isotonic maps from named descriptors with the anchor table shown;
  scales sharing a descriptor are compared on that axis, never converted.
- Differential support only within one source where diagnosis and device coincide (pads:
  PD, ET, atypical parkinsonism, MS, healthy; gaitndd: PD, HD, ALS, control), labelled
  device-bound; no cross-source etiology head, because diagnosis equals source for HD,
  CP, stroke, KOA and MS.
- Monitoring: a change is flagged only beyond a between-day minimal detectable change,
  estimated on ltmm non-fallers and, for PD, on the same-state windows of pd_biostamprc21
  (EVAL, admitted on 2026-09-24) across days as an evaluation-only number;
  descriptors without a between-day estimate ship with monitoring disabled, and no
  acquisition is planned to lift that.

## 7. Interventions and counterfactual bodies

The operator displaces descriptors: d' = d + sum over k of A_k(u_k), each A_k monotone in
its dose axis, masked to the channels and sites the intervention can touch, with
kinetics. Levodopa: an effect compartment with population onset 30 to 60 min and duration
2 to 5 h and a per-subject rate under a prior (the kinetics stay a prior in this program:
no dataset in scope carries both a dose axis and timestamps). DBS: a step with a population prior on
washout; no kinetics are fitted on tremordb (its washout series are not monotone and too
few). Masks: levodopa lowers the bradykinesia gains, tremor amplitude and freezing
propensity and raises the flow term with dose; STN acts on tremor, bradykinesia and
freezing with dyskinesia free in sign; GPi on tremor, bradykinesia, dyskinesia and
sustained posture; Vim on tremor only, as a prior. Gains are per subject and per channel,
so domain-divergent responses (tremor resistant, bradykinesia responsive, ON-freezing) are
representable. Botulinum toxin is out of version 1 entirely: no kinetics, no mask, no
prior; cdystonia and dystonia_alcohol_bfm enter no stage; no dataset in scope carries a
body signal across toxin cycles, so the branch stays out of the program.

A counterfactual body is produced by inferring subject, session, body and nuisance from the
observed session, replacing u by u', resampling the body from the flow head under the
displaced descriptors with the task fixed, and re-rendering it into the sensor the clinic
used. It is shown beside the observed body with the predicted change per descriptor and a
support score, and refused outside support. It is a hypothesis in physical units, not a
picture.

Falsification today, all never trained, each with a trivial baseline (predicted change
equals a constant times the OFF deviation) that the operator must beat:
- care_pd BMCLab: 19 PD with straight OFF and ON walks and UPDRS_GAIT per walk;
  direction and within-subject magnitude of speed, stride length and arm swing.
- care_pd E-LC: 17 subjects with OFF and post-dose Timed Up and Go in a freezer cohort,
  no UPDRS_GAIT; turn duration and freezing halt probability, segmented by the body into
  straight portion and turn; no arm-swing claim.
- pd_biostamprc21, EVAL, admitted on 2026-09-24 under CC BY 4.0: ON versus OFF
  item windows on the 12 PD with tagged OFF then ON windows
  (5 PD and all 17 controls carry untagged medication rows); Status, age and sex carried
  from the raw clinic CSV into the EVAL manifest; a per-subject paired test; the controls'
  null as a session-order permutation; forearm-in-wrist-slot placement in the nuisance
  table.
- tremordb, as it sits on disk: 13 pairable subjects, of which 4 high-amplitude ones carry a
  DBS ON versus OFF pair at medication OFF (one GPi, three STN) for the direction test,
  reported as a count without kill; the 8 low-amplitude subjects sit at the laser-Doppler
  floor (1 to 4 mm/s) and become an abstention null (the model must abstain there);
  re-emergence within 15 min is tested on the 3 subjects with that series; the levodopa
  direction is tested on the 6 high-amplitude subjects with medication ON and OFF at DBS
  OFF. No monotone-washout gate (the measured 15, 30, 45 min series are not monotone), no
  centimetre claim (unknown beam vector), no Vim (both Vim subjects have DBS OFF only).
- kaggle_fog EVAL half, admitted on 2026-09-24: subjects with ON and OFF recordings and
  per-visit UPDRS-III ON and OFF (80 such subjects in the harmonized 101), a second
  never-trained levodopa set for the session head; the Kaggle rules allow any use, so its
  results are publishable; the Kaggle copy is never redistributed, and any data release
  goes through the sponsor's CC BY 4.0 copy on Zenodo (10.5281/zenodo.10959560).
- oxford_dystonia_lfp non-LFP channels, a description, not a test of the operator: the
  62 ramp files, tabulated by a script on 2026-09-28 (reports/oxford_lfp/), hold 46
  ascending ramps (one ON block per amplitude step, alternating with OFF blocks) and 16
  single-block files, no down-ramp, at 4096 Hz with stimulation at 130 Hz; ON blocks
  last a median 65 s in P2 and 41 s in P3, so the 30 s of the one file first measured
  does not generalize; 4 patients (8 hemispheres, 6.69 h) carry a head accelerometer
  during ramps, 3 of them (6 hemispheres) also neck EMG, and P6 and P7 a second,
  undocumented triaxial accelerometer. Block onset and offset come from the LFP
  stimulation artefact and the command channel gives the amplitude only, because the
  command precedes the first artefact pulse by 200 to 260 ms (Am1 files) or 5 to 41 ms
  (Cur1 files): G15 as written fails in all 62 ramp files. No therapeutic dystonic change
  is expected within one ON block, so the ON block
  against its flanking OFF blocks, per amplitude, is reported as an acute effect or side
  effect, per patient (n = 4, hemispheres nested), without kill; the step at and the step
  below the side-effect threshold are excluded from any benefit reading. The leads sit in
  STN, ZI and VL, never GPi.

The intervention claims of this program stop at these sets. No acquisition is planned
(section 14): levodopa kinetics stay a prior, the DBS operator keeps its population
priors, and the toxin branch stays out.

## 8. Site and device control and its proof

Prevention: nuisance lives only in the kernels; the typed attention mask keeps nuisance
tokens away from body positions; the factory renders one body under many devices,
cameras, rates and trackers with a nuisance-consistency loss; a nuisance descriptor table
for the 46 harmonized sources in scope (goodwin_smm, kaggle_fog, ninapro and
pd_biostamprc21 among them since the licence decisions of 2026-09-24) (device, rate, placement convention, units and known unit errors,
acc_content, tracker, camera family) feeds only the kernels; HSIC in multi-source strata;
no gradient-reversal head.

Proof, with every version (kill numbers in G10): (a) the leakage number: balanced accuracy
of a linear source probe on posterior-mean body summaries minus that of a probe fed only
the legitimate covariates (diagnosis, task, age band, sex, sensor-availability mask,
native rate), on task-matched walking windows from weargait_pd, clinical_gait_multipatho,
fog_star, fog_turning_filho and physionet_gaitpdb, reported per stratum; (b) care_pd
leave-one-centre-out UPDRS_GAIT against pooled, with cohorts resampled to a common rate
and the centre probe stratified by score so that severity cannot masquerade as site;
(c) weargait_pd split by subject-id family (NLS and HC versus WPD and WHC, plausibly its
two sites, to be confirmed from the Synapse metadata), training gait readouts on one
family and testing on the other; (d) device swap on the same trials: weargait_pd 13
sensors versus two wrists, IMU-only versus insole-only descriptors on 175 paired subjects,
pads left versus right watch on tremor frequency only; (e) tracker swap on the same 25
CODY adults (keypoints versus silhouettes) and walking-direction swap on koa_pd_nm;
(f) cross-device freezing transfer, leave-one-source-out across daphnet, fog_star,
multimodal_fog and sensefog, and to the never-trained fog_turning_filho IMU
(session level, and episode level against its specialist video labels) and to the
kaggle_fog EVAL half (admitted on 2026-09-24); (g) a re-identification audit:
identity probes on r, and also on shape, segment lengths and the descriptor vector across
sessions (koa_pd_nm, ltmm), reported as top-1 among N; the model card states that body
parameters are biometric data and that stature and segment lengths are released only at
the resolution the norm needs.

## 9. Evaluation gates with kill numbers

A kill fires when the point estimate is below the line; the subject-clustered bootstrap
interval is reported beside it. Every CODY kill line is set at the trivial-baseline rate
(majority class or population prior, measured in stage 0) plus 0.05, never at a nominal
chance. Coverage gates carry a sharpness bound (mean 90 percent band width at most 1.5
times the multi-sensor band, CRPS at least 20 percent better than the population prior;
for G9 an interval width at most 0.8 of the population-prior interval) so that widening
the prior cannot pass them.

- G1 kernel round trip, 53 amass persons of the lockbox v1.2, held out by whole subset
  (ACCAD, GRAB, MPI, SOMA, TCDHands, TotalCapture, Transitions) as well as by id,
  synthetic sensors: six virtual IMUs MPJPE at most 45 mm, kill above 70; single wrist
  IMU: 90 percent interval coverage of the true wrist trajectory at least 85 percent with
  the sharpness bound, kill below 70; single camera PA-MPJPE at most 60 mm, kill above 90.
- G2 renderer realism, before any pretraining: a discriminator on 2 s IMU windows between
  weargait_pd control trials and amass walking rendered in the 13-Xsens layout, with the
  rendered set matched to the real set on speed and cadence quantiles, accuracy at most
  0.70, kill above 0.85; a second test on the same body: bodies inferred from weargait
  control trials, re-rendered into the 13 sensors, real versus re-rendered, same numbers;
  removing the soft-tissue filter must not move the discriminator above 0.85. Decided by the
  owner on 2026-09-28 (OD9 of docs/EXECUTION_PLAN.md), after the first half read 0.984
  against the 0.85 kill line (reports/g2_renderer/: Walk phases only, lockbox persons
  excluded, measured device noise): the first half is a pre-registered, reported diagnostic
  with its discriminator fixed as in scripts/g2_renderer_realism.py (gradient boosting on 624
  window features, 5-fold person-grouped); synthetic data enter stage 1 by utility, the
  simulator-only line of G3 scored on non-lockbox persons before pretraining, and by the
  second half once bodies can be inferred.
- G3 IMU reads the pathological body, weargait_pd lockbox (37 subjects, PD and controls
  separately), 13 IMUs only: stride length MAE at most 5 cm against the walkway, kill
  above 10; gait speed MAE at most 0.08 m/s, kill above 0.15; initial contact against
  insole events median error at most 30 ms, kill above 50 or if not 20 percent better than
  a shank-gyroscope detector. Simulator-only variant first: stride MAPE at most 25 percent,
  kill above 40, which stops everything.
- G4 physics from the body: stroke lockbox, per-foot vertical GRF from pose only, RMSE at
  most 10 percent body weight, kill above 20, and at least 0.03 above the Newton baseline
  in correlation, kill otherwise; the stroke lockbox metric-3D round trip (true marker
  mocap, 10 subjects) reported; gait120 lockbox (24 subjects), median log-envelope
  correlation at least 0.6 over four leg muscles, kill below 0.4.
- G5 pathology preservation, care_pd, evaluation only: synthetic frontal and sagittal
  projections through the tracker process; the UPDRS_GAIT probe on re-estimated bodies
  within 0.05 QWK of the probe on source bodies, kill if the drop exceeds 0.10; BMCLab ON
  versus OFF separability of gait descriptors within 5 AUROC points, kill above 10;
  PA-MPJPE against the care_pd fits reported as a diagnostic only.
- G6 consistency on the same subjects: cody_1 versus cody_3 on the 25 adults, identical
  dominant phenomenology in at least 80 percent and tremor-frequency and posture-deviation
  ICC at least 0.8 on the conditions that exist (rest, posture, action of the upper limb;
  no gait task is documented), kill below 60 percent or 0.6; koa_pd_nm shape from
  sequence 01 closer to 02 than to the between-subject median in at least 90 percent, kill
  below 75.
- G7 phenomenology on CODY, one sealed read per version, bundled with G6 and the tracker
  swap of G10 under one rule-file hash: cody_1, 25 adults (21 patients, 4 controls),
  frame-level consensus of its two raters under a pre-registered rule (a frame is positive
  when both raters score at least 1; disagreement frames excluded and their fraction
  reported), pre-registered descriptor rules plus a frozen probe: tremor AUROC at least
  0.85; sustained posture and flow AUROC at least 0.75 each; kill lines at the trivial
  baseline plus 0.05; myoclonus reported without kill; tics, ballismus, athetosis and
  stereotypy reported as absent. cody_3 external, 34 subjects from silhouettes only:
  per-label subject-level balanced accuracy rather than a dominant-phenomenology top-1
  (tremor 22 versus 12 subjects, cohort-confounded and reported as such; sustained posture
  within dataset_3, 8 versus 12), children of dataset_2 reported without kill; kill lines at
  the stage-0 trivial baseline plus 0.05.
- G8 severity, lockbox and never-trained sets: li_taati UDysRS Spearman as a
  person-level cross-fitted estimate on its 9 persons, reported without kill (fewer than 30
  patients, no lockbox; 2026-09-28); modys_video DIS at least 0.50 on its 7-child lockbox
  with a child-clustered
  interval, kill below 0.35 only if the 5-fold child-level cross-fitted estimate on the 27
  training children is also below 0.35 (7 children cannot carry a kill alone); weargait_pd items 3.10 to
  3.13 mean QWK at least 0.50, kill below 0.40; care_pd UPDRS_GAIT leave-one-cohort-out
  over its four labelled cohorts (110 subjects) Spearman at least 0.55, kill below 0.40;
  hubu_fis_fingertap QWK at least 0.55, kill below 0.40, with the asymmetry sign recovered
  in at least 70 percent of the 40 discordant subjects.
- G9 null controls and honesty: capture24 (151 healthy, 3,883 h), per activity class from
  its annotation dictionary: tremor, freezing or flow flagged in at most 2 percent of
  sleep, sitting and walking windows, kill above 5, with rhythmic manual tasks reported
  separately as a known limit of wrist-only input; ltmm non-fallers at most 8; gaitrec
  orthopaedic patients: a movement-disorder profile above the healthy 95th percentile in
  at most 10 percent, kill above 25; stroke lockbox and koa_pd_nm grades 3 and 4 flagged as
  sustained abnormal posture with task dependence in at most 10 percent, kill above 25;
  rigidity returns not measured with prior-width coverage at least 0.85, kill below 0.80;
  the rate-and-unit stress test moves at least 95 percent of corrupted records out of band,
  kill below 80; 90 percent intervals on every G8 gate have coverage between 0.85 and 0.95
  on the lockbox with calibrators fitted elsewhere, kill below 0.80.
- G10 site and device: leakage number at most 5 points per stratum, kill at 10 or more;
  care_pd leave-one-centre-out drop at most 0.10, kill at 0.20; weargait_pd site-family
  transfer stride MAE ratio at most 1.3, kill above 1.6; pads two-watch tremor frequency
  within 0.5 Hz in at least 90 percent of records where both wrists exceed threshold, kill
  below 75; cross-device freezing event-level average precision, leave-one-source-out
  across daphnet, fog_star, multimodal_fog and sensefog (each under 30 patients), reported
  with its interval without kill; on the never-trained fog_turning_filho IMU, predicted
  per-session freezing episode count and time in freezing against the session table of
  PDFEinfo.csv (35 PD, 22 with at least one episode in session 1) and, since the
  per-sample flags of its 71 turning trials carry per-episode timing, episode timing
  against those flags (specialist video labels, never IMU-derived), reported without kill;
  on kaggle_fog (admitted as EVAL on 2026-09-24), event-level average precision at least
  0.40 on its EVAL half, kill below 0.30; absolute
  correlation between per-video usable fraction and DIS error on modys_video (94 manifest
  records of 34 children, errors from the cross-fitted probe and the lockbox) at most 0.2,
  kill above 0.4.
- G11 interventions, never trained, each against the trivial proportional baseline:
  BMCLab (19): correct direction of change in speed, stride length and arm swing in at
  least 80 percent of subjects, kill below 70, within-subject magnitude Spearman at least
  0.4 and at least 0.1 above the baseline, kill below 0.2; E-LC (17): direction of turn
  duration and halt probability in at least 65 percent, kill below 50; pd_biostamprc21
  (admitted as EVAL on 2026-09-24; the 12 PD with tagged OFF then ON windows): ON
  versus OFF window AUROC from the session state at
  least 0.70, kill below 0.60, and the operator applied to OFF predicts the ON direction of
  bradykinesia and tremor descriptors in at least 80 percent, kill below 65; tremordb:
  direction of amplitude change on the 4 high-amplitude pairable subjects and the levodopa
  direction on the 6 with medication ON and OFF, both reported as counts without kill;
  abstention on the 8 subjects at the measurement floor required, kill if the model emits
  an amplitude there; kaggle_fog (admitted as EVAL on 2026-09-24): session-head ON versus
  OFF AUROC at least 0.70, kill below 0.60.
- G12 body premium: a masked multimodal transformer over the same tokens at equal compute,
  three seeds, with its readouts defined in stage 0 as linear probes on its frozen tokens
  trained on the same lockbox-excluded targets: the body model must beat it on G3, G8 and
  G10 beyond the three-seed interval; if it loses on G7 or G8 beyond the interval, the body
  is demoted to a readout layer on the baseline and reported as such; G11 is a body-only
  claim with its own null.
- G13 process and scale: at most four CODY reads per year, each logged with the hash of the
  pre-registered rule file; any number reported after an undated threshold change is
  void; scaling to 200M must improve G1 MPJPE by 10 percent and G3 stride MAE by 1 cm over
  the 60M pilot or the program stays at 60M.
- G14 video against IMU on patients, fog_turning_filho, never trained (Videos.zip fetched
  and verified on 2026-09-24; 71 video and IMU turning pairs of 35 PD; one Sony camera at
  about 29.98 frames per second filming the lower limbs; one Physilog 5 at 128 Hz on the
  shank of the more affected side; three sessions one month apart, all ON medication). The
  per-sample freezing flag and the PDFEinfo episode table are one consensus annotation of
  the video by two specialists, transferred to the IMU clock by an undocumented procedure:
  they are called specialist video labels, never IMU-derived. Nothing from this set trains,
  calibrates or picks a threshold, except out-of-fold in 5-fold subject-disjoint
  cross-validation stratified by freezer status. The video-IMU offset and drift of each
  pair are estimated from kinematics only (video shank angular velocity of the
  instrumented leg against the IMU gyroscope), never from flags or model output; a pair
  whose offset uncertainty exceeds 100 ms or whose drift exceeds 50 ms per 120 s is
  excluded and counted. Intervals are subject-cluster bootstrap; a pass needs the point
  estimate at the pass line and the interval bound on the kill side not beyond the kill
  line; if more than 20 percent of pairs are excluded or fewer than 15 subjects carry a
  matched episode, the metric is reported without kill. Halt probability is the per-frame
  probability of being in freezing now, scored on 2 s windows; turning duration is the
  time of each 360 degree turn, the reference segmentation integrating the vertical shank
  gyroscope under a rule hashed before any video is opened (written and hashed on
  2026-09-28, reports/fog_imu/turn_rule.sha256; it finds 1,500 turns, 1,282 of them
  complete and freezing-free). G14a, each sensor against the
  specialist labels (the IMU arm runs without video): halt-probability AUROC at least
  0.75, kill below 0.60, and not below the AUROC of the IMU freeze index on the same
  windows (0.580, 95 percent interval 0.434 to 0.706, measured on 2026-09-28 in
  reports/fog_imu/); median absolute onset error on matched episodes at most 1.0 s,
  kill above
  2.0 s; event F1 at 0.2 to 2 s tolerances, false alarms per minute in non-freezers and in
  standing files, and pre-freezing AUROC reported without kill; a pass licenses only
  "agrees with specialist video annotation". G14b, cross-sensor, video model against IMU
  model and an independent IMU rule: turning duration per matched freezing-free turn,
  ICC(2,1) at least 0.70, kill below 0.50; freezing onset, video arm against IMU arm on
  episodes both detect, after the kinematic offset, median absolute difference at most
  300 ms, kill above 600 ms; halt AUROC of the video arm minus the IMU arm against the
  labels at least -0.05, kill below -0.15. Until G14b passes, the cross-sensor thesis
  reads "the same quantities where cross-sensor agreement has been measured"; 2D-only
  readouts, and every lower-limb-only video readout, carry a metric-scale-from-prior flag.
  Keypointing of the video waits on the extraction of Videos.zip on the local disk (an owner action since the permission
  classifier refused it on 2026-09-24; the dedicated environment of OD4 exists).

## 10. What is deliberately excluded, with the fact that decided it

- 2D-to-3D lifting as data: fabricates depth with a healthy prior exactly where pathology
  lives; depth stays in the posterior.
- A learned pathology latent as the primary object, and a mechanistic controller with
  named perturbation parameters (loop gains, delays, noise structure): unidentifiable from
  a mostly single-visit, single-sensor corpus; readouts are formulas, and the controller
  stays out of the program (the paired prospective cohort it needed is outside the scope,
  section 14).
- A shared tremor frequency across limbs and any gate demanding bilateral symmetry:
  contradicts PD and ET physiology.
- Millisecond myoclonus readouts from 25 to 33 fps video, and MDS-UPDRS items 3.4 and 3.17
  from a body without fingers.
- Tics, ballismus, stereotypies and athetosis as channels: no training source; procedural
  caricatures judged on 21 CODY patients are indistinguishable from any large fast
  movement.
- A cross-source etiology head: diagnosis equals source for HD, CP, stroke, KOA and MS.
- A rank-one intervention operator: levodopa responses diverge by domain within a subject.
- The toxin branch: score tables only, no body observation can update or falsify it.
- Physics-in-the-loop contact simulation and neural SDE controllers: the claim collapses
  when the physics loss is relaxed, and the compute is not credible.
- EEG and LFP as body sensors, and any neural branch in version 1: merk_neumann_ecog_lfp,
  oxford_stn_lfp and the sensefog Percept LFP stay an inversion testbed on the frozen body.
- seizeit2 in the body model: 22,897 h of seizure monitoring with deltoid EMG and 25 Hz
  IMU, 57 percent of corpus hours, no body content.
- Hour-weighted sampling and identity-contrastive pretraining: a re-identification channel.
- Gradient-reversal site heads: they defeat the probe they train against.
- A billion-parameter model: about 200 h of 3D pose and thin pathology; sizes above 400M
  have nothing to learn from.
- An infant body, hands and face in version 1; the animal lane; disease progression over
  months (the only longitudinal pathology signal in scope is modys_video, T0, T3 and T12
  for 29 children, with the course between visits not stated in its files: a
  sensitivity-to-change analysis at most, no progression claim).
- Training on care_pd, pd_biostamprc21, tremordb or CODY in any tier, training on
  kaggle_fog beyond the INTERNAL comparison session head of stage 4 (its CONTRAST half),
  and any CODY use for thresholds, normalization or tracker statistics.

## 11. Risks and early indicators

- Sim-to-real for IMU is load-bearing: G2 above 0.85 or the simulator-only G3 above 40
  percent, visible by day 60.
- The hyperkinetic side of the backbone rests on modys_video (34 children), with li_taati
  (9 persons, person-level cross-fitting since 2026-09-28) and hd_accel_jnet in the probes
  only: a G7 point estimate at the kill line at the first read means the paper claims a
  hypokinetic and gait instrument with per-limb tremor, flow and sustained posture, not
  hyperkinetic coverage.
- Session state may not be identifiable from cross-sectional data: G11 BMCLab direction
  below 70 percent while the BMCLab separability of G5 passes (and G11 pd_biostamprc21
  AUROC below 0.60).
- Prior collapse toward normal gait: the G5 preservation drop, checked on the 60M pilot
  before scaling.
- Wrist-only and lower-back-only inputs are prior-dominated: G1 single-wrist coverage
  below 70 or G9 false positives above 5 percent.
- Bandwidth: li_taati runs at 30 frames per second as stated by its data paper
  (PMC6219082), a nominal rate, since none is recoverable on disk (2026-09-28);
  modys_video runs at 25 frames per second, measured that day.
- The nuisance descriptor table is the schedule risk (documented unit errors in weargait_pd
  gyro, stroke EMG, physionet_semg_walking headers, multimodal_fog counts): a G10 leakage
  number above 10 traced to one source.
- E-LC is a freezer cohort under acute levodopa (freezing persists): a G11b failure there
  is physiology before it is a model defect, which is why it has its own threshold.
- With kaggle_fog admitted (2026-09-24), a gap between the released session head and the
  INTERNAL comparison head trained on its CONTRAST half measures what the release loses to
  licence discipline; a gap above 0.10 AUROC is reported, not hidden.
- The scope itself: no new data enter. The licence decisions of 2026-09-24 keep the
  medication-window set and the kaggle_fog line of G11 and the kill line of the
  cross-device freezing check; goodwin_smm and asdpose enter for internal research only
  (results reported with citation; no data, derived data or weights released). Until the
  fog_turning_filho video, on disk since 2026-09-24, is keypointed (the dedicated
  environment of OD4 exists; the videos wait on extraction), cross-sensor agreement on
  patients rests on weargait_pd and stroke gait only. A line not evaluated is marked not evaluated in the gate table, not passed.
- Thin hyperkinetic and paediatric ground: 34 modys_video children (lockbox 7), no healthy
  child norm, no tic or myoclonus training source, levodopa kinetics that never leave their
  prior. The paper states these limits; no acquisition is planned to lift them.
- The MA-22 body drifting more than 5 mm from the internal body on G1 means the open rig
  is not carrying the model; the release is blocked until it does.
- Compute is required, not secured: the cluster application is on the critical path; the
  60M pilot and all evaluations run locally regardless.

## 12. Compute

Local RTX 5080 (16 GB): kernels and factory (silhouette renders and virtual sensors cached
on scratch), the 60M pilot (about ten days), the INTERNAL tier at 60M, every evaluation
including the sealed CODY read, inference of the 200M model, head and conformal fits, and
all local video work (hubu_fis_fingertap, koa_pd_nm, and the fog_turning_filho video,
fetched on 2026-09-24 into a local cache outside the git tree). External cluster:
8 H100 80 GB with 2 TB of fast scratch; stage 1 about 250 H100-hours, stage 2 about 350,
stages 3 and 4 under 50, the G12 baseline about 500, three seeds and four ablations about
2,000; about 3,200 H100-hours in year one for the RELEASE tier, three weeks of wall time
on 8 GPUs, gated by the G13 scaling rule. Only canonical tensors and body parameters of
RELEASE-tier sources travel to the cluster.

## 13. First 90 days

- Days 1 to 15: freeze the manifest tiers with hashes and licence classes, the lockbox, the
  re-keyed person ids, the CODY denylist (adult, ext_ds and PEDIA namespaces) and canary,
  and the harness with every gate scripted with its kill number and trivial-baseline
  number; pre-register the descriptor rule file and the CODY consensus rule with a dated
  hash; build the nuisance descriptor table; recover the frame rates of modys_video, li_taati and
  the remap_open turning episodes (3,498 records); compute a levodopa-equivalent dose from
  the weargait_pd medication text where legible; parse care_pd medication state from the
  raw walk names; deduplicate motion_x against amass; verify the ds3 rater roster of
  cody_3; recover the pd_biostamprc21 group and placement table. Done by 2026-09-28:
  tiers, person ids and lanes on every canonical record (32,609 canonical subject ids
  mapped to persons, none shared between two sources); the lockbox v1.2 (871 persons,
  frozen with a SHA-256); the frame rates of modys_video (25 frames per second, measured)
  and of the remap_open turning episodes (30 frames per second, nominal, stated by the
  source readme), li_taati's stated as 30 frames per second by its
  data paper (PMC6219082), not recoverable on disk; the weargait_pd levodopa-equivalent
  dose for 69 PD with a legible regimen (median 880 mg/day); care_pd medication state
  (BMCLab 19 subjects with ON and OFF walks, E-LC 17); the pd_biostamprc21 group table (17
  PD, 17 controls; 12 PD with tagged OFF then ON windows, 5 PD untagged).
- Days 1 to 30: implement and unit-test the kernels on amass; render the virtual sensors;
  run both halves of G2 and fix the renderer until they pass (since OD9, 2026-09-28, the
  first half is a reported diagnostic and synthetic data are admitted by utility, section 9).
- Days 10 to 40: ingest the recovered pairs: weargait_pd per-trial IMU orientations,
  walkway and insole arrays and the PKMAS walkway metrics as targets; the
  physionet_multimodal_gait vendor kinematics as weak auxiliary targets; the vidimu IMU
  rerun; drummer_tremor_model C3D kinematics; extract hand keypoints for
  hubu_fis_fingertap locally; keypoint the fog_turning_filho videos locally (fetched and
  verified on 2026-09-24) once Videos.zip is extracted (the environment of OD4 exists).
- Days 30 to 60: stage 1 as the 60M pilot on the local GPU; pass G1, G4, the healthy half of
  G3 and the G5 preservation check before any cluster time; start the G12 baseline.
- Days 45 to 75: stages 1 and 2 at 200M on the cluster for the RELEASE tier, INTERNAL at
  60M locally; report G3, G5, G6, G9, G10, G13 and G14; run the
  ablations.
- Days 60 to 90: stage 3 heads and stage 4 operator; report G8 and G11 on the lockbox and
  the never-trained sets; one sealed G7 read on cody_1 and cody_3 with the rule-file hash
  logged; write the go or no-go report with the gate table per tier and the G12 decision.
- In parallel from day 1, owner decisions within MoveAtlas only (no ethics submission, no
  portal route, no application, no fetch of data MoveAtlas does not hold). Taken on
  2026-09-24: the video part of the published fog_turning_filho was fetched on the owner's
  go and verified (G14); the 15 rights_unclear datasets were decided by the session on the
  owner's delegation (docs/decisions/licence_decisions_2026-09-24.csv, section 16.3).
  Applied by the owner on 2026-09-24 from the verified patch in docs/moveatlas_corrections/: the registry
  corrections of modys_video to 34 children and fog_turning_filho to 35 patients,
  mrc_lfp_dystonia marked a duplicate of oxford_dystonia_lfp, and a CC0
  docs/license_matrix.csv row for merk_neumann_ecog_lfp, committed as MoveAtlas f2e6f99.
  Owner decision OD4 of docs/EXECUTION_PLAN.md, the dedicated environment, is done (conda
  env mdfm, env/README.md). Open since 2026-09-28: the G2 discriminator (section 9, OD9),
  whether a primary replaces F3 (16.5, OD10) and the corrected oxford G15 rule (OD11).
  Outside data: the cluster application only.

## 14. Acquisitions: outside the scope

By owner decision (2026-09-24, section 16) the program acquires nothing: no cohort, no
application, no fetch of a dataset MoveAtlas does not hold. No gate, readout, use case or
question depends on an acquisition. What each acquisition of the earlier ranking would
have unlocked, for the record:
- In-house paired prospective cohort: pathological 3D truth with two views, hyperkinetic
  cross-sensor agreement, timed levodopa kinetics, between-visit repeats.
- mpower: the levodopa contrast at scale for the session head.
- Toxin cohort: a toxin operator.
- shanghai_child_gait3d: a healthy paediatric norm, hence degree-valued readouts for children.
- Dystonia Coalition: a focal dystonia video training source.
- PPMI sensors, WATCH-PD, PPP: progression claims.
Two former items are not acquisitions: the fog_turning_filho video belongs to a published
MoveAtlas dataset and was fetched on 2026-09-24, and leuven_cp_gait is on disk and was
admitted (CC0) on 2026-09-24 (section 16.3).

## 15. Provenance and what remains unverified

This document was produced on 2026-09-23 and 2026-09-24 by: six independent from-scratch
designs written by separate reasoning passes from six angles (analysis-by-synthesis
world model, scale-first token sequence model, latent controller dynamics, clinic-first,
data-reality-first, contrarian retrieval atlas); twelve adversarial judgements (a
machine-learning and data-feasibility lens and a clinical and motor-neuroscience lens per
design), each of which checked dataset ids, counts and licences on disk; one synthesis
(the clinic-first design won on both lenses; the measured-body frame was shared by five
of six); two red-team passes (feasibility, clinical and governance) with 37 kill attempts
of which 6 succeeded; a final revision pass of the panel that applied every fix and
re-checked tremordb, pd_biostamprc21 and kaggle_fog (both admitted as EVAL by the licence
decisions of 2026-09-24, section 16.3), care_pd and cody_3 on disk (its
corrections are folded into sections 5, 7, 9 and 11); and the session's own position
written before reading the panel. The judges' scores were modest (5 to 7 of 10 on soundness, feasibility
and clinical value; 4 to 7 on the credibility of the 3D body; 8 to 9 on falsifiability):
the field's honest verdict is that the measurement frame is a real advance for this corpus
and that the pathological 3D body is credible only where sensors constrain it, which is
what sections 2, 6 and 9 encode. On 2026-09-24 the owner restricted the program to
MoveAtlas data (section 16); a scope pass applied that rule with the categories measured
in reports/data_scope/ (scripts/data_scope.py against MoveAtlas commit a5d3814) and folded
in on-disk corrections: modys_video counts 34 children, not 63, and fog_turning_filho
holds its IMU and PDFEinfo.csv (35 PD). On the same day, the licence of each of the 15
rights_unclear datasets was read at its source, re-checked
(reports/licence_research/licence_verdicts.md) and decided by the session on the owner's
delegation (docs/decisions/licence_decisions_2026-09-24.csv), and the fog_turning_filho
Videos.zip was fetched on the owner's go, checked against the figshare size and MD5 and
inventoried (scripts/probe_fog_videos.py, reports/fog_videos/), with its data paper
(PMC8904564) read for the camera, the IMU placement and the origin of the freezing labels.

Unverified as of this date and to be settled in the day-1 to day-15 tasks: the exact
frame rates of modys_video and li_taati; what happened to the modys_video children between
T0, T3 and T12; the weargait_pd site families; the overlap between motion_x and amass; the
amass and motion_x terms on rented compute; the cody_3 rater roster; whether the three
AUX1 channels of oxford_dystonia_lfp P1 form an accelerometer; the pd_biostamprc21 group
table (17 PD and 17 controls per the raw clinic CSV; the registry says 34 PD). Settled on
2026-09-28 (reports/nuisance/, reports/person_ids/): modys_video runs at 25 frames per
second; pd_biostamprc21 holds 17 PD and 17 controls; li_taati's 390 ids are trial numbers
of 9 persons, not about 134 patients; monipar is 6 persons in the canonical manifest, not
28; the li_taati frame rate is not recoverable on disk and its data paper (PMC6219082)
states 30 frames per second. Every number in
sections 5 to 9 that names a dataset was checked against docs/datasets_index.csv, the
datacards or the processed tables by at least one verification pass, except the
modys_video child count and the fog_turning_filho patient count, where the registry was
wrong (34 distinct IDs in data_clinical_scoring.xlsx; 35 PD in PDFEinfo.csv), and the
fog_turning_filho video and sensor figures of G14, which come from reports/fog_videos/ and
its data paper (PMC8904564); every scope category comes from
reports/data_scope/data_scope.csv. The compute figures are estimates and will be replaced
by measured pilot numbers.

## 16. Data scope (owner decision, 2026-09-24)

The owner decided on 2026-09-24 that the program uses MoveAtlas data only: datasets published
by MoveAtlas (scope published) and datasets held on disk with a right to use but not
published (scope usable_not_published). This section overrides any earlier plan in this
document and in its draft addendum. Categories and counts come from scripts/data_scope.py
run against MoveAtlas commit a5d3814 with the licence decisions of
docs/decisions/licence_decisions_2026-09-24.csv (reports/data_scope/data_scope_summary.md,
data_scope.csv, _provenance.json).

### 16.1 Measured categories

| scope | meaning | datasets | on-disk GB | in the program |
|---|---|---|---|---|
| published | on the MoveAtlas Zenodo record | 54 | 508.5 | in scope |
| usable_not_published | on disk, not hosted; licence grants research use, or own data | 46 | 2046.1 | in scope; the release rules of section 5 and the conditions of 16.3 |
| rights_unclear | on disk, not hosted; no, restrictive or unverified terms | 0 | 0.0 | none since the licence decisions of 2026-09-24 (16.3) |
| not_held | catalogued only, barrier, not fetched | 71 | 0.0 | outside the scope (16.6) |
| excluded | on disk, decided out of scope | 1 | 0.0 | outside the scope (16.6) |
| tool | software, not data | 1 | 0.1 | not data |

In scope: 100 datasets (54 published, 46 usable_not_published); all 46 harmonized
datasets are in scope: the 42 in scope before the licence decisions (36 published, 6
usable_not_published) plus goodwin_smm, kaggle_fog, ninapro and pd_biostamprc21. The
licence decisions of 2026-09-24 (16.3) settled the 15 rights_unclear datasets:
kaggle_spiral is excluded, mobile_gaitlab is not held (the disk holds only its code and a
testing-only Gillette demo), and every other one is usable_not_published under the
conditions of 16.3. Before them the scope held 87 datasets (54 published, 33
usable_not_published), with 70 not held. scripts/data_scope.py reads the decisions from
docs/decisions/licence_decisions_2026-09-24.csv; no line of this design stays conditional
on an owner decision about a dataset.

### 16.2 What the design uses

Before the licence decisions this document named 56 registry ids: 35 published, 9
usable_not_published, 6 rights_unclear, 6 not_held (data_scope.csv matched against the
text); the decisions moved pd_biostamprc21, kaggle_fog, goodwin_smm, asdpose and
leuven_cp_gait to usable_not_published and mobile_gaitlab to not_held.
- usable_not_published: amass and motion_x (RELEASE, with the rented-compute rule of
  section 5), ntu_rgbd and koa_pd_nm (INTERNAL), care_pd and hubu_fis_fingertap (EVAL),
  oxford_dystonia_lfp (analysis lane, CC BY-SA 4.0), seizeit2 (outside the body model,
  section 10), merk_neumann_ecog_lfp (inversion testbed, CC0; no row in
  docs/datasets_index.csv; its CC0 docs/license_matrix.csv row was added by the owner on
  2026-09-24 from docs/moveatlas_corrections/); admitted on 2026-09-24:
  pd_biostamprc21 and kaggle_fog (EVAL), goodwin_smm and asdpose (INTERNAL, internal
  research only), leuven_cp_gait (CC0).
- not_held: oxford_stn_onoff_dbs, oxford_stn_lfp, dryad_pd_5imu_adl, mindts_tourette,
  mpower, shanghai_child_gait3d, mobile_gaitlab (the disk holds only its code and a
  testing-only Gillette demo); the Dystonia Coalition, PPMI sensors, WATCH-PD and PPP,
  named in prose, are not_held as well.

Stages 0 to 4 and every kill line of section 9 run on published and usable_not_published
data alone, under the conditions of 16.3; a not_held dataset appears only in 16.6.

Gates under the scope:
- In scope in full: G1 to G7, G9, G12, G13.
- G8: in scope; modys_video on 34 children (lockbox 7, training portion 27) with the double
  condition written in G8; li_taati (9 persons) as a person-level cross-fitted estimate
  without kill (2026-09-28).
- G10: in scope, with the kaggle_fog freezing line (kaggle_fog admitted as EVAL); the
  leave-one-source-out and fog_turning_filho lines are reported without kill.
- G11: in scope, with BMCLab (19), E-LC (17), tremordb and the pd_biostamprc21 and
  kaggle_fog lines (both admitted as EVAL on 2026-09-24).
- G14: in scope; the fog_turning_filho video was fetched and verified on 2026-09-24; its
  keypointing waits on the extraction of Videos.zip (the environment of OD4 exists); the IMU
  arm of G14a runs without video and its pre-checks are done (reports/fog_imu/).
A line that is not evaluated appears in the gate table as "not evaluated" with its
reason, never as passed.

### 16.3 Decisions within MoveAtlas (2026-09-24)

fog_turning_filho video. The dataset is published (CC BY 4.0); IMU.zip (106 recordings,
71 of them turning trials of 120 s, each with the per-sample freezing flag) and
PDFEinfo.csv are on disk. Videos.zip (figshare item 14984667 v7, CC BY 4.0, 5,011,298,150
bytes) was fetched on the owner's go on 2026-09-24 into a local cache outside the git
tree; its size and MD5 match the figshare record. Inventory (scripts/probe_fog_videos.py,
reports/fog_videos/): 73 MP4 videos of 35 patients (sessions 1, 2, 3: 35, 23, 15 videos),
1280x720 H.264, about 29.98 frames per second with no constant frame interval, an audio
track in every file, durations 112 to 123 s; 71 videos pair with the 71 IMU turning
trials (2 videos have no IMU trial); video and IMU durations differ by -7.9 to +3.0 s
(median +0.4 s, 57 of 71 within 1 s), so every pair needs its own offset. In all 71
sessions the flag runs equal the PDFEinfo episode count and the flagged time equals the
PDFEinfo total time in freezing within 1 s. The data paper (Frontiers in Neuroscience
2022, PMC8904564) states one Sony camera at 30 Hz filming only the lower limbs and one
Physilog 5 IMU at 128 Hz on the shank of the more affected side (not the lower back);
the freezing labels come from two movement disorder specialists who annotated the video
together by consensus, and the per-sample flag and the episode table are that annotation
transferred to the IMU timeline by an undocumented procedure, with no inter-rater
agreement reported. G14 and validation route (g) of section 2 use these pairs; their
keypointing waits on the extraction of Videos.zip (the environment of OD4 exists).

The 15 datasets that were rights_unclear, decided on 2026-09-24 by the session on the
owner's explicit delegation in chat after each licence was read at its source and
re-checked (docs/decisions/licence_decisions_2026-09-24.csv,
reports/licence_research/licence_verdicts.md):

| id | decision | licence and conditions | feeds |
|---|---|---|---|
| pd_biostamprc21 | admitted, EVAL, never trained | CC BY 4.0 per DataCite and the IEEE DataPort terms | G11 medication windows (12 PD); PD between-day change (section 6); stage 0 table |
| kaggle_fog | admitted, EVAL | Kaggle rules allow any use and forbid redistribution of the Kaggle copy; the sponsor's full copy on Zenodo 10.5281/zenodo.10959560 is CC BY 4.0 and is the only route for any redistribution | stage 4 INTERNAL comparison head (CONTRAST half only); G10 and G11 lines (EVAL half) |
| goodwin_smm | admitted for internal research only, INTERNAL | no licence, citation request; no data, derived data or weights released; results reported with citation | INTERNAL stereotypy probe (6 subjects); addendum Q17 |
| asdpose | admitted for internal research only, INTERNAL | data sharing statement: access for everyone, analyses for any purpose; no explicit data licence; no data, derived data or weights released; results reported with citation | stereotypy candidate (241 children, 7,352 annotated segments; not ingested) |
| leuven_cp_gait | admitted | CC0; 357 children with CP and 56 typically developing children per its source | spastic cerebral palsy reference curves |
| mobile_gaitlab | excluded, counted as not held | the disk holds only the code and a testing-only Gillette demo; the 1,026-patient set is not held | nothing in the design depends on it |
| ninapro | admitted | CC0 on Dryad; the held zips come from the project site, and their equivalence to the Dryad copy is recorded before any derived release | addendum 4.2 healthy hand EMG reference |
| osf_fmd_eeg | admitted for internal research only | no licence on the OSF node; patient data stays local, aggregate results only | addendum AUX (derived EEG, 37 participants) |
| osf_fmd_kinematics | admitted for internal research only | no licence on the OSF node; patient data stays local, aggregate results only | addendum AUX (head, eye and gaze kinematics) |
| mini_rgbd | admitted, INTERNAL | CC BY-NC-SA 4.0; its SMIL model files are never shipped | nothing in version 1 (infants are excluded) |
| ataxie_kinect_dib | admitted | CC BY 4.0 via the article; supplementary table only, no skeleton data | nothing |
| kamath_snpc_geo | admitted | NCBI GEO, no restriction | nothing |
| clinvar | admitted | NCBI policy, attribution, never for diagnosis | nothing |
| panelapp | reference only | Genomics England terms, non-commercial, all rights reserved; never inside a released artefact | nothing |
| kaggle_spiral | excluded | the CC BY-NC-ND label was set by an uploader who is not the creator | nothing |

The internal-research admissions (goodwin_smm, asdpose, osf_fmd_eeg, osf_fmd_kinematics)
release no data, derived data or weights; their results are reported with citation, and
the patient data of the two osf_fmd sets stays local, with aggregate results only. The
non-commercial admissions (pads, mini_rgbd, panelapp) hold while MD_FM stays
non-commercial research, which the release of weights under CC BY-NC-SA already assumes.

### 16.4 Corrections folded in

1. modys_video: 34 distinct children in data_clinical_scoring.xlsx, 95 videos, 161 scored
   rows (T0 68, T3 29, T12 64), 29 children at three time points; the registry, the
   datacard and the earlier design said 63; the manifest keys its 94 records per video.
   Consequences: lockbox 7 and training portion 27 (section 5), G8 and G10 recounted
   (section 9), the progression statement of section 10 corrected. The frame rate,
   absent from the manifest, was measured at 25 frames per second from the source's
   stick-figure movies on 2026-09-28 (reports/nuisance/).
2. oxford_dystonia_lfp: 68 MAT 7.3 files at 4096 Hz, not 77 (62 ramp, 5 single-pulse, 1
   paired-pulse). The 62 ramp files, tabulated on 2026-09-28 (reports/oxford_lfp/), are 46
   ascending ramps and 16 single-block files with no down-ramp, stimulated at 130 Hz; ON
   blocks last a median 65 s in P2 and 41 s in P3, not the about 30 s generalized from
   P3_L5_Testing.mat (28.4 to 32.6 s ON blocks per 0.5 mA step, 14.5 to 24.7 s OFF
   blocks); 4 patients (P2, P3, P6, P7; 8 hemispheres, 6.69 h) carry a head accelerometer
   during ramps, 3 of them (P2, P3, P6; 6 hemispheres) also neck EMG, both confirmed; P6
   and P7 also carry a second, undocumented triaxial accelerometer; P1 and P4 have no
   labelled movement channel, P5 has a single-pulse file only.
3. pd_meg_stn_lfp_ds004998: acq_time in the sub-*_ses-PeriOp_scans.tsv files puts every
   MedOff run before every MedOn run in 18 of 20 subjects (about 50 to 80 min between
   blocks); sub-AbzsOg and sub-BYJoWR interleave, and in sub-BYJoWR one MedOff run is
   timed after the only MedOn run, a possible label error for the owner. Medication state
   is confounded with time on the table.
4. fog_turning_filho: PDFEinfo.csv lists 35 PD with session-level freezing (time in
   freezing, episode count; 22 with at least one episode in session 1; session 2 and 3
   values for 23 and 13); the per-sample freezing flag of its 71 IMU turning trials
   carries per-episode timing: its runs equal the PDFEinfo episode count in all 71
   sessions and its flagged time equals the PDFEinfo time in freezing within 1 s. Flag
   and table are one consensus video annotation by two specialists (specialist video
   labels, never IMU-derived); the IMU sits on the shank of the more affected side, not
   the lower back. Measured on 2026-09-28 (reports/fog_imu/): 173 flagged runs in 42 of the
   71 trials; 26 of the 35 patients have at least one flagged episode (22, 12 and 8 in
   sessions 1, 2 and 3).

### 16.5 The neural and hyperkinetic addendum under the scope

The draft addendum (neural state, brain-body coupling, dystonia and hyperkinetic program)
obeys the same rule. In scope: sensefog, multimodal_fog, physionet_multimodal_gait,
cap_sleep, pd_rest_eeg_ds004584, pd_rest_eeg_onoff_ds002778, pd_walking_eeg_ds007526,
pallidal_units_dystonia and lance_adams_myoclonus (published); seizeit2,
pd_meg_stn_lfp_ds004998, oxford_dystonia_lfp, merk_neumann_ecog_lfp, dandi_m1_mptp_units,
dandi_pd_macaque_units and nlb_mc_maze (usable_not_published). osf_fmd_eeg is
usable_not_published since 2026-09-24, for internal research only (16.3).

Removed from the addendum, not kept as "needs acquisition": every not_held set (16.6); the
prospective dystonia DBS cohort with its hypotheses H1 to H3 and sample sizes (the red-team
fixes to them become moot), the status dystonicus arm, in-house Percept streaming, the
neural add-on to the in-house cohort; the questions that need them (Q2, Q13, Q14, Q18, Q3
in GPi, Q11 stage 2, the clinical parts of Q15 and Q17); and the acquisition-dependent
parts of U3 and U5 to U10. The addendum keeps what in-scope data can answer.

Binding fixes from the two red-team reports (every attack that did not survive), to be
written into the addendum and its stage N0 rule file:
- Q1 (oxford_dystonia_lfp): an acute contrast of each ON block against its flanking OFF
  blocks per amplitude, with a level-by-amplitude interaction (dose and time are collinear
  within a file) and nulls that permute ON and OFF labels within amplitude pairs and level
  labels across whole files within hemisphere; the patient (n = 4) is the unit, hemispheres
  nested; one body primary (1 to 12 Hz head-acceleration power) and one neural primary (4
  to 12 Hz periodic power on the bipolar pair around the stimulated level), all else
  exploratory; no therapeutic reading and no programming claim; the step at and the step
  below the side-effect threshold excluded; all 62 ramp files tabulated by a script
  (done on 2026-09-28, reports/oxford_lfp/: 46 ascending ramps and 16 single-block files,
  no down-ramp), and the second, undocumented triaxial accelerometer of P6 and P7 checked
  for a trunk site.
- G19 loses F6 (behind-the-ear EEG to deltoid EMG) and F7 (sleep stage to chin tone, a
  scoring rule); seizeit2 wake data serve as a G20 artefact positive control and as
  pretraining signal only.
- G21 loses its seizeit2 target (seizure labels are evaluation-only). Q12 trains nothing,
  runs on all 125 subjects with patient-clustered AUROC, reports no number for a contrast
  with fewer than 10 patients per class, keeps as negatives only seizures labelled
  non-motor, and calls its latency "behind-the-ear EEG change latency relative to the
  annotated onset".
- ds004998: an order-only null (earlier against later task-matched MedOff runs through the
  same pipeline) for Q6, the G21 medication target and the levodopa arm; an ON against OFF
  result counts as a medication effect only if it beats the order-only AUROC beyond the
  subject-bootstrap interval, otherwise it reads "state or time"; tasks matched within
  subject; sub-AbzsOg and sub-BYJoWR reported separately.
- G15 on sensefog: an anchor independent of the one used to align (a second artefact or
  tap edge, or the ECG shared by LFP and EEG at session start and end), with offset and
  drift estimated; with one anchor only, lags above 4 ms plus the worst-case drift are
  admitted and no direction is claimed at beta-phase scale; the Percept group delay is
  estimated from the artefact edge. Measured on 2026-09-28 (reports/sync_sensefog/):
  independent anchors exist, and the admissible-lag floors per session are tabulated there.
- G15 on oxford_dystonia_lfp (2026-09-28, reports/oxford_lfp/): the addendum's rule
  (command channel and LFP artefact within one sample) fails in all 62 ramp files; the
  command precedes the first artefact pulse by 200 to 260 ms (Am1 files) or 5 to 41 ms
  (Cur1 files), with up to 188 samples of variation within a file. The proposed
  correction (owner decision OD11 of docs/EXECUTION_PLAN.md) takes the stimulation onset
  and offset from the LFP artefact itself and uses the command channel for the amplitude
  only.
- G15 on seizeit2 (2026-09-28, reports/sync_seizeit2/): 183 of 2,850 runs pass (399.7 h),
  901 are void and 1,766 cannot be checked; its accelerometer channels are not scaled in g
  (resting norm median 0.168), which matters for the fixed-threshold body descriptors of
  Q12 (Q8 in the addendum).
- G15 on physionet_multimodal_gait (2026-09-28, reports/sync_physionet_gait/): it fails,
  since the lags between EEG, IMU and force are not constant across speeds; the set serves
  pretraining only, and F3 cannot run on it (multiplicity below).
- G19 nulls: the rule file states per set the number of admissible nulls and the smallest
  attainable p; a pass needs at least 20 per held-out subject; 60 s records
  (physionet_multimodal_gait, pretraining only since 2026-09-28) use cross-record swaps
  within subject across treadmill speeds and a stride-cycle shuffle, without the 30 s
  floor; records with fewer than 20 nulls enter no pass.
- Q4: the kinematic regressor uses body samples at or before the neural sample only;
  direction comes from conditional spectral Granger or directed coherence at the native
  4096 Hz, with the group-delay standard error stated per hemisphere and the single-pulse
  files (P5, P6, P7) as causal anchor.
- G20: a kill gate only where a head sensor exists (oxford_dystonia_lfp;
  physionet_multimodal_gait failed G15 on 2026-09-28 and serves pretraining only);
  sensefog and multimodal_fog are marked "prior-only regressor" and
  carry no gait-phase claim; an injected-artefact ground truth on sensefog sitting EEG
  scores both removal and recovery; multimodal_fog, delivered ICA-cleaned, leaves the
  harmonic criterion.
- G16: the LFP front end is selected on rest against hold or move blocks, never on MedOn
  and MedOff labels; pass and kill are read on the paired subject-clustered bootstrap
  interval (pass: lower bound above 0 on 2 tasks; kill: upper bound below 0 on 2 tasks;
  otherwise undetermined), with the minimum detectable difference per task published
  before training.
- G21: three outcomes, pass, absent (the interval excludes a 0.05 gain) and undetermined;
  only absent demotes a use case; the minimum detectable gain per set is published in N0.
- Multiplicity: one confirmatory primary per dataset in year 1 (oxford_dystonia_lfp Q1,
  ds004998 F1, physionet_multimodal_gait F3, merk_neumann_ecog_lfp F4, seizeit2 Q12), Holm
  across the family, everything else exploratory without "falsified" language. Since
  2026-09-28 F3 cannot run (physionet_multimodal_gait fails G15 and serves pretraining
  only): the confirmatory family of addendum section 3 has six primaries unless the owner
  names a replacement, and Holm runs over six.
- Q11 stage 1: titled "gene or its surgical correlates" (the CSV has no age, anaesthesia or
  centre column), restricted to pairwise contrasts of genes with at least 5 patients
  (TOR1A 9, SGCE 5, VPS16 5) against a label-permutation null; asking the depositors for
  covariates is outside the scope, and stage 2 does not start.
- Q10: within modys_video only (34 children), mixtures of factor analysers with K = 1 to 3
  after marginal Gaussianizing transforms, K > 1 tested by parametric-bootstrap likelihood
  ratio, reported as descriptive; other sources serve as within-source replications only.
Counts corrected by the same reports: PD and healthy EEG 37.9 h (not 37.8); synchronized
sets 9.3 h (not 9.2), kept out of pretraining except physionet_multimodal_gait, which
serves pretraining only since it failed G15 on 2026-09-28.

### 16.6 Appendix: outside the scope by owner decision

No gate, readout, use case or question depends on anything listed here.
- Not held, named in this design or its addendum: oxford_stn_onoff_dbs, oxford_stn_lfp,
  mrc_lfp_dystonia (a duplicate of oxford_dystonia_lfp), beijing_sleep_lfp,
  dabi_movement_ephys, dryad_stn_kinematics, dryad_vim_et_intraop, dryad_pd_5imu_adl,
  tuh_eeg, openneuro_pd_rest_eeg_768, natmeg_pd, lancaster_prehd, camcan, mass_sleep,
  nsrr_plm_psg, iceberg_icm, ebrains_rat_lfp_atlas, gin_pd_rat_dbs, open_dbs_pd,
  openneuro_cd_ctbs_174, dystonia_coalition, mindts_tourette, osf_tic_video_46vxm,
  enroll_hd_family, cis_pd, real_pd, nbia_tircon, dbsmatchmaker, mpower, ppmi_sensors,
  watch_pd, ppp_radboud, shanghai_child_gait3d, tim_tremor, mdsgene, uncommonvoice,
  meei_voice_db, mobile_gaitlab (the disk holds only its code and a testing-only Gillette
  demo).
- Excluded by the licence decisions of 2026-09-24, named in 16.3: kaggle_spiral (its
  CC BY-NC-ND label was set by an uploader who is not the creator).
- Not in the registry, named in the addendum: the EDS Spectrum Study and the EDS figshare
  videos, the ATP1A3 video archive, PEDiDBS, CHILD-DBS, PedsDBS-TT, the paediatric pallidal
  LFP cohorts, the hyperkinetic trial cohorts, the family-mediated registries, the GSTT
  paediatric videos.
- New acquisitions: the in-house paired prospective cohort and its neural add-on, the
  toxin cohort, the prospective dystonia DBS cohort with its status dystonicus arm,
  in-house Percept BrainSense streaming, and any fetch of data MoveAtlas does not hold.
