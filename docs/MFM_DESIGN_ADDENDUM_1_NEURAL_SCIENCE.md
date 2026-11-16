# Addendum 1 to the MFM design: the neural state, brain-body coupling, and the dystonia and hyperkinetic program, under the MoveAtlas-only scope

Date: 2026-09-24. Status: proposal for owner review. Amends docs/MFM_DESIGN_2026-09-24.md, called "the design" below. English, no em-dashes.
Corpus: MoveAtlas at commit a5d3814, read-only; the measurements dated 2026-09-28 were made against f2e6f99, the owner's commit of the patch of 1.6. Scope: the owner's decision of 2026-09-24, section 0.
Thresholds, window lengths and parameter budgets are pre-registration choices of this addendum, not measurements. Every count carries its source.

Source keys:
- DS: reports/data_scope/data_scope.csv and data_scope_summary.md in this tree (scripts/data_scope.py against MoveAtlas a5d3814, run 2026-09-24). Scope and on-disk size come from there.
- EEGm, EMGm, IMUm, FORCEm: datasets/_canonical_eeg_manifest.csv, _canonical_emg_manifest.csv, _canonical_imu_manifest.csv, _canonical_force_manifest.csv in the MoveAtlas tree.
- DC: the dataset's datacard.md. LM: docs/license_matrix.csv. FB: docs/feedback/FEEDBACK_PAPERS_2026-09-23.md in this tree.
- NI, HI, SI: the neural, hyperkinetic and science inventory passes of 2026-09-24.
- RT: the two red-team passes on the draft of this addendum, 2026-09-24. Every fix of every attack that the draft did not survive is applied here.
- S: re-checked on disk by this pass (section 9 lists how).

## 0. What changes and why

### 0.1 The owner's scope decision

The program uses MoveAtlas data only, in two classes measured by DS: published (54 datasets, 508.5 GB on disk) and usable_not_published, that is held on disk with a right to use but not to host (46 datasets since the licence decisions of 2026-09-24, on-disk size in DS). Together, 100 datasets, 46 of them harmonized. Three classes leave the plan:

- not_held (71 datasets, 0.0 GB on disk): barriers, catalogued-only entries and everything not fetched. They appear only in the appendix, and no gate, question or use case of this document depends on them. mobile_gaitlab counts as not held since 2026-09-24: its disk copy holds only the code and a testing-only Gillette demo, not the 1,026-patient set.
- excluded (1 dataset): kaggle_spiral, whose CC BY-NC-ND label was set by an uploader who is not the creator; nothing here uses it.
- new acquisition of any kind: no prospective cohort, no new fetch, no application, no author request. The appendix records what was proposed and dropped.

The former class rights_unclear (held, but with no written licence, restrictive terms or unverified terms; 15 datasets, 219.0 GB) is empty since 2026-09-24. Each licence was read at its source and re-checked (reports/licence_research/licence_verdicts.md), and the session decided each dataset on the owner's explicit delegation in chat (docs/decisions/licence_decisions_2026-09-24.csv). Admitted: pd_biostamprc21, kaggle_fog, ninapro, leuven_cp_gait, ataxie_kinect_dib, kamath_snpc_geo, clinvar and mini_rgbd. Admitted for internal research only (no data, derived data or weights released, results reported with citation): goodwin_smm, asdpose, osf_fmd_eeg and osf_fmd_kinematics. Reference only: panelapp. All of these are now usable_not_published. kaggle_spiral is excluded and mobile_gaitlab is not held (above). Section 1.4 gives each decision with its conditions.

One published dataset needed a fetch of a part already published: fog_turning_filho holds its IMU on disk (IMU.zip, 106 records, S), and its Videos.zip had never been downloaded. On the owner's go, Videos.zip (figshare item 14984667 v7, CC BY 4.0, 5,011,298,150 bytes) was fetched on 2026-09-24 into a local cache outside the git tree, and its size and MD5 match the figshare record. The inventory (scripts/probe_fog_videos.py, reports/fog_videos/) finds 73 videos of 35 patients, 71 of them paired with the 71 IMU turning trials. Design gate G14, the only real video-versus-IMU pair on patients, therefore has its data; keypointing of the video waits on the extraction of Videos.zip on the local disk, an owner action (the dedicated environment of OD4 exists since 2026-09-24).

### 0.2 What the addendum still adds

Version 1 of the design keeps EEG and LFP out because they do not measure the body (design sections 0, 3 and 10). That reason stands, and it is why they do not become body sensors here. This addendum adds a second measured object, the neural state, observed through its own contracts (scalp EEG, MEG, LFP, ECoG, and the drive band of EMG), read through fixed unit-bearing descriptors as the body is, and coupled to the body by a low-capacity, lagged module trained only on synchronized sessions and always judged against a within-subject null.

The dependency runs one way. The neural side reads the body posterior: as the target of coupling, as a motion-artefact regressor, and as the source of movement onsets with a timing uncertainty. The body posterior, its descriptors and every body readout are computed without neural input, so a tremor frequency from a clinic phone video is the same quantity as the one computed beside subthalamic LFP, and design gates G1 to G14 are unaffected by construction (checked by G17).

### 0.3 What the scope costs, stated plainly

- The DBS latency hypothesis of the mission prompt (claim 3: the phasic command changes within hours while tonic posture takes weeks) cannot be tested. It needs a prospective GPi-DBS cohort with a body time course, which is an acquisition. No in-scope dataset holds chronic or longitudinal DBS in dystonia (HI). The hypothesis stays written down, untested, and the design's GPi operator keeps one population prior with no fitted kinetics.
- There is no GPi recording of any kind in dystonia in scope. oxford_dystonia_lfp records STN, zona incerta and VL thalamus, in 130 Hz stimulation blocks whose ON phase lasts a median 65 s in P2 and 41 s in P3 (reports/oxford_lfp/, 2026-09-28). Every dystonia neural statement of year 1 is therefore acute, target-specific and a case series.
- No in-scope recording pairs a hyperkinetic rating scale (DIS, BFM, TWSTRS, UDysRS, YGTSS, UHDRS) with EEG or LFP (HI). Every neural biomarker validated against ordinal totals in the literature stays unrechecked against continuous body targets in dystonia.
- There is no tic signal outside CODY, which is evaluation-only. Stereotypy exists only in goodwin_smm and asdpose, admitted on 2026-09-24 for internal research only (no data, derived data or weights released), and in mmasd, where stereotypy is a proxy.
- leuven_cp_gait, admitted on 2026-09-24, reports 56 typically developing children beside 357 with CP per its source; until its curves on disk are counted and a typically developing group is found there, there is no healthy child norm in scope, and every paediatric readout stays a within-source deviation.
- Twelve of the eighteen questions of the draft lost their data-needed line. Six survive with in-scope data, plus seven rebuilt or newly scoped ones; section 3 lists thirteen. Since 2026-09-28 Q6 has no admissible data: physionet_multimodal_gait fails G15 and serves PRE only.

### 0.4 What stays, and what this replaces in the design

Unchanged: the body thesis, the tiers, patient-level splits, CODY evaluation-only in every form, licence discipline, the design's 90-day plan (the neural work runs beside it, mostly on CPU), and the refusal of a learned pathology latent.

Replaced in the design: the exclusion of "any neural branch in version 1" (section 10; the exclusion of EEG and LFP as body sensors stays); the line "EEG and LFP: no kernel in version 1" (section 3; they get neural-side contracts, never a body kernel); the acquisition list of design section 14, which the scope suspends (appendix). Corrections to design counts are in 1.6.

## 1. Data

Scope classes are DS values: published, usable_not_published, rights_unclear (empty since the licence decisions of 2026-09-24), excluded, not_held. Form is harmonized (a MoveAtlas contract exists) or raw (read by a reader in this tree). Pairing status, from the inventories: synchronized (same file or stream, or an alignment verified on disk); synchronized, not verified (declared by the source); alignable, not aligned; neural-only; table only.
Roles: PRE, neural encoder pretraining on unlabelled signal; CPL, brain-body coupling on synchronized sessions; INT, within-subject intervention contrast; EVAL, evaluation only; AUX, unit-level or animal material outside every encoder.

### 1.1 In-scope neural datasets with movement on the same clock

| id | scope | licence | form | GB | pairing | subjects | neural signal | movement on the same clock | role |
|---|---|---|---|---|---|---|---|---|---|
| sensefog | published | CC BY 4.0 | harmonized (EEG, EMG, IMU), Percept raw | 3.9 | EEG and EMG in one stream; IMU aligned by the 'M  1' marker, verified by a heel-strike check; Percept LFP alignable; G15 (2026-09-28) finds independent anchors and sets an admissible-lag floor per session (reports/sync_sensefog/) | 11 PD with raw EEG (12 in the paper) | scalp EEG 48 sites, 1000 Hz, reference undocumented, 45 records, 2.73 h (EEGm, S); STN LFP (Percept) 250 Hz, 43 sessions, 86 streams, 5.0 h, unit undeclared | leg EMG (TA, GA, bilateral) 1000 Hz, 45 records; APDM IMU 128 Hz, 43 records, sites lower_back, ankle_L, ankle_R, no head sensor (IMUm, S) | CPL (EEG, EMG, IMU now; LFP above the per-session admissible-lag floor of G15); freezing probe by cross-fitting |
| multimodal_fog | published | CC BY 4.0 | harmonized | 5.8 | one combined 500 Hz stream | 12 PD (008 recorded twice) | scalp EEG 25 channels, 500 Hz, delivered filtered and ICA-cleaned, mastoid reference, 56 records, 3.63 h (EEGm, S) | 5 EMG, ECG and EOG channels; IMU at 100 Hz native resampled to 500 Hz, sites lower_back, wrist_L, shank_L, shank_R in varying combinations, no head sensor (IMUm, S) | CPL (slow regime only); freezing probe by cross-fitting |
| physionet_multimodal_gait | published | CC BY 4.0 | harmonized | 7.6 | synchronized per source, not verified (force Sync columns all zero); fails G15 (2026-09-28): lags between EEG, IMU and force are not constant across speeds (reports/sync_physionet_gait/) | 59 healthy adults with EEG; 57 with EEG and IMU (IMUm, S) | scalp EEG 19 channels, 300 Hz, reference Pz, 176 records, 2.93 h; every record 59.87 s (EEGm, S) | Xsens free acceleration 60 Hz at 9 sites including head in all 167 records (IMUm, S); EMG as a 16.05 Hz envelope, 169 records, 58 subjects; force plates 173 records | PRE only, since it fails G15; the only in-scope walking set with a head sensor |
| pd_meg_stn_lfp_ds004998 | usable_not_published | CC0 | raw (BIDS FIF) | 173.7 | same FIF | 20 PD on disk (22 in participants.tsv) | MEG 306 channels at 2000 Hz; bilateral STN LFP from externalized leads; empty-room runs for 7 subjects | 4 EMG at 2000 Hz, muscles not named; no accelerometer | CPL; INT (levodopa, with the order null of 1.5); cross-fitting only |
| oxford_dystonia_lfp | usable_not_published | CC BY-SA 4.0 | raw (3 zips, MATLAB 7.3) | 14.7 | same file, 4096 Hz | 7 cervical dystonia, 12 hemispheres; 5 patients with a labelled movement channel (S) | 8-level leads spanning STN, zona incerta and VL thalamus; 6 to 7 scalp EEG contacts; 68 files, 9.65 h (S) | neck EMG in P2, P3, P5, P6; head accelerometers in P2, P3, P5, P6, P7; ramp files with a movement channel 6.69 h of 8.38 h of ramps (S) | CPL; INT (stimulation dose); analysis lane, never in released weights |
| merk_neumann_ecog_lfp | usable_not_published | CC0 (DC; no LM row) | raw (BIDS iEEG), not in the registry | 1.4 | same file | 16 participant ids; the paper analyses 11 PD | sensorimotor ECoG strip in 73 runs; STN LFP in 20 of 73; 1000 Hz | grip force left and right in all 73 runs, z-scored, so unitless (DC, S); run duration 39 to 569 s, median 288 s (S) | CPL; body-to-brain testbed |
| seizeit2 | usable_not_published | CC0 | harmonized (EEG, EMG), IMU aux | 198.2 | one device family, two modules (EEG module on the upper back with behind-the-ear electrodes; ECG and EMG module on the left chest), cross-module clock checked by G15 (2026-09-28): 183 of 2,850 runs pass (399.7 h), 901 are void and 1,766 cannot be checked (reports/sync_seizeit2/) | 125 focal epilepsy | behind-the-ear EEG, 2 bipolar channels, 256 Hz, 2,850 records, 11,626.25 h (EEGm, S) | deltoid EMG 256 Hz, 2,804 records, 124 subjects; acc and gyro at 25 Hz per module, accelerometer not scaled in g (resting norm median 0.168) | PRE (unlabelled signal); EVAL (seizure labels, datacard rule); artefact positive control |
| cap_sleep | published | ODC-By 1.0 | harmonized | 28.8 | same EDF | 35 harmonized ids (22 RBD, 13 normal) of 34 persons: two RBD EDF files are byte-identical (reports/person_ids/) | PSG EEG, 2 to 13 bipolar channels, 100 to 512 Hz, 331.52 h (EEGm, S) | chin EMG canonical for 3 records only (EMGm, S); limb EMG unidentified in aux | PRE; sleep-staging probe (G16). No coupling family (RT) |
| dandi_m1_mptp_units | usable_not_published | CC BY 4.0 | raw (NWB) | 6.4 | same NWB | 2 macaques | M1 units with antidromic identification, pre and post MPTP flag | elbow angle, torque, velocity and 6 arm EMG near 1 kHz in 432 of 447 files | AUX: fast-regime methods check (F8) |
| nlb_mc_maze | usable_not_published | CC BY 4.0 | raw (NWB) | 0.7 | same NWB | 1 healthy macaque | 182 M1 and PMd units | hand position and velocity at 1000 Hz, 2,295 trials | AUX: secondary methods check |

### 1.2 In-scope neural-only, table-only or signal-free

| id | scope | licence | form | GB | pairing | content | role |
|---|---|---|---|---|---|---|---|
| pd_rest_eeg_onoff_ds002778 | published | CC0 | harmonized | 1.0 | neural-only | 31 subjects (15 PD recorded ON and OFF, 16 controls), EEG 32 channels at 512 Hz, 46 records, 2.52 h (EEGm, S) | EVAL (medication contrast) |
| pd_rest_eeg_ds004584 | published | CC0 | harmonized | 5.9 | neural-only | 149 subjects (100 PD, 49 controls), 63 channels at 500 Hz, 149 records, 6.64 h | PRE; G16 probe; lockbox 30 |
| pd_walking_eeg_ds007526 | published | CC0 | harmonized | 8.5 | neural-only | 144 subjects, 60 channels at 250 Hz, 277 records, 19.49 h; amplitude off by about 1000, unresolved | PRE and probe only after the amplitude is traced; lockbox 29 |
| pallidal_units_dystonia | published | CC BY 4.0 | table | under 0.01 (0.48 MB) | table only | 1,714 GPi units from 32 patients as 38 unit-feature columns plus patient, gene, hemisphere, trajectory, depth; no age, no anaesthesia, no severity (S) | AUX (Q10) |
| dandi_pd_macaque_units | usable_not_published | CC BY 4.0 | raw (NWB) | 1049.2 | movement as events only | 2 macaques, GPi, VL and M1 units | AUX, not used in year 1 (size) |
| striatal_gait_6ohda | usable_not_published | CC BY 4.0 | raw (zip), not read | 2.2 | not verified | 33 mice, opto-tagged striatal units per DC | AUX after reading |
| lance_adams_myoclonus | published | CC BY 4.0 | raw (figures) | 0.011 | no signal | back-averaging as PDF figures; one identifiable video (INTERNAL) | none; it documents the absence of myoclonus signal |
| tremordb | published | ODC-By 1.0 | table and traces | 0.008 | no neural signal | 15 PD with DBS, hand velocity by laser Doppler | body-side EVAL in the design (G11); not used here |


### 1.3 In-scope dystonia and hyperkinetic datasets, body side

| id | scope | licence | form | GB | population and signal | scales and labels | role |
|---|---|---|---|---|---|---|---|
| modys_video | published | CC BY 4.0 | harmonized | 0.09 | dyskinetic cerebral palsy; DeepLabCut 2D, 14 points, 500 frames per clip, pixels, 25 frames per second (measured from the source's stick-figure movies, reports/nuisance/) | DIS dystonia and choreoathetosis per region, duration and amplitude; 34 distinct ids, 95 videos, 161 scored rows; T0 34 videos, T3 29, T12 32; 29 ids at three time points, 3 at two, 2 at one; 66 videos scored by 2 scorers, 29 by 1 (S) | RELEASE; backbone and DIS probe (design); Q9, 4.6 |
| li_taati | published | CC BY 4.0 | harmonized | 0.17 | PD with levodopa-induced dyskinesia; 2D, 15 points, 30 frames per second as stated by the data paper (PMC6219082), not recoverable on disk; 511 records, 258 subject ids that are trial numbers of 9 persons (sn_numbers.txt; the paper reports 9 participants; reports/person_ids/), so no lockbox and person-level cross-fitting only | UDysRS part 3 per body part, MDS-UPDRS-III items, CAPSIT | RELEASE; 4.1, 4.10 |
| hd_accel_jnet | published | CC BY 4.0 | harmonized | 0.16 | Huntington disease; wrist accelerometer 100 Hz, 35 records, 35 subject ids (IMUm, S) | UHDRS TMS, FA, TFC; 24 HD and 10 controls per DC | probe by cross-fitting (design); 4.10 |
| gaitndd | published | ODC-By 1.0 | harmonized (force) | 0.06 | 64 subjects: 20 HD, 15 PD, 13 ALS, 16 controls; foot-switch force 300 Hz, uncalibrated | severity column | backbone (design); 4.10 |
| pads | published | CC BY-NC-SA 4.0 | harmonized | 1.8 | 469 subjects, wrist IMU, PD, ET, atypical parkinsonism, MS, healthy | within-source differential | design; Q12 |
| cervical_motion_multiplanar | published | CC BY 4.0 (DC and LM) | harmonized (EMG) | 0.59 | 14 asymptomatic adults, 127 records, 4.39 h; 2 EMG channels at 1785.7 Hz plus quaternions at 74.1 Hz; sensor placement undocumented | 504 cervical movements in 3 planes | healthy neck reference for 4.1 and 4.2, with the placement caveat |
| cdystonia | published | free teaching repository, no SPDX licence | table | 0.0 (4.4 KB) | 109 cervical dystonia patients, 9 sites, 631 rows | TWSTRS at weeks 0, 2, 4, 8, 12, 16; placebo, 5000 U, 10000 U | scale-time-scale prior only (4.4); no stage, no operator (design section 7) |
| dystonia_alcohol_bfm | published | CC0 | table | 0.0 (167 KB) | on disk: a co-investigator list and one aggregate family-history table | no per-patient BFM rows (HI) | none beyond documenting the defect (1.6) |
| drummer_tremor_model | published | etalab | harmonized | 3.3 | 18 healthy subjects, 2000 Hz EMG and C3D kinematics | none | physiological tremor reference (Q12) |
| monipar | published | CC BY 4.0 | harmonized | 0.04 | 46 records of 6 persons, IMU: the canonical cohort is the supervised subgroup, and 22 more subjects are on disk, not ingested (reports/person_ids/) | tremor labels | design probe; Q12 |
| imu_inthewild_iprognosis | published | CC BY 4.0 | harmonized | 2.0 | 45 subjects, wrist IMU in the wild | PD and control | design; Q12 context |
| mendeley_et_accel | usable_not_published | CC BY 4.0 | raw | 0.003 | essential tremor accelerometry | none | Q12 secondary |
| osf_tremor_amplitude | usable_not_published | CC BY-NC-SA 4.0 | raw | 0.06 | tremor amplitude recordings | none | Q12 secondary |
| gait120 | published | CC BY 4.0 | harmonized | 18.1 | 120 healthy subjects, 39 markers, 2000 Hz EMG, 4,125 EMG records (EMGm, S) | none | healthy limb EMG reference for 4.2 |
| fog_turning_filho | published | CC BY 4.0 | harmonized (IMU); video raw, in a local cache outside the git tree | 0.18 (IMU; Videos.zip, 5.0 GB, in the cache) | 35 PD (IMU manifest), three sessions one month apart, all ON medication; one Physilog 5 at 128 Hz on the shank of the more affected side (data paper; the IMU manifest says lower back), 106 records, 71 of them turning trials of 120 s; 73 MP4 videos of 35 patients, lower limbs only, 1280x720, about 29.98 frames per second, 71 paired with a turning trial (fetched and inventoried 2026-09-24, reports/fog_videos/) | per-sample freezing flag and PDFEinfo episode table, one consensus video annotation by two specialists transferred to the IMU clock (specialist video labels, with per-episode timing); clinical table | EVAL, never trained; design G10 and G14 |
| cody_1, cody_2, cody_3 | published | own data (IRB), derived CC BY 4.0 | harmonized and raw | 0.62, 0.68, 122.2 | in-house hyperkinetic cohorts, 2D keypoints and silhouettes | per-frame and subject-level phenomenology ratings; the label census is descriptive, hashed once into the rule file, and sets no threshold of any gate or question (RT) | EVAL only, sealed reads (design G7) |

Also in scope and not used here: dystonia_mri_ds006395 (66.4 GB, CC0, imaging only), svd_spasmodic_dysphonia (0.49 GB, voice), mmasd (7.6 GB, Apache, stereotypy as a proxy), ssbd (0.002 GB, annotations only), obf_psychiatric (0.01 GB, no dyskinesia labels), chorea_acanthocytosis_mri (0.019 GB, one identifiable video).

### 1.4 Formerly rights_unclear, the ones that matter here (licence decisions of 2026-09-24, section 7)

| id | GB | decision of 2026-09-24 and its conditions | what it holds | what depends on it |
|---|---|---|---|---|
| pd_biostamprc21 | 17.2 | admitted: CC BY 4.0 per DataCite and the IEEE DataPort terms; tier EVAL, never trained | 34 subjects, medication and item windows | the design's G11 medication test; nothing in this addendum |
| kaggle_fog | 133.6 | admitted: the Kaggle rules allow any use and forbid redistribution of the Kaggle copy, so any redistribution goes only through the sponsor's full copy on Zenodo (10.5281/zenodo.10959560, CC BY 4.0); tier EVAL, its CONTRAST half training only the INTERNAL comparison session head of design stage 4 | 136 subjects, freezing; 80 of the harmonized 101 with ON and OFF visits and per-visit UPDRS-III (design section 7) | the design's session head and cross-device freezing; U8 scale |
| ninapro | 41.7 | admitted: CC0 on Dryad; the held zips come from the project site, and their equivalence to the Dryad copy is recorded before any derived release | hand EMG, 201 records, 67 subjects (EMGm) | the healthy hand EMG reference of 4.2 |
| goodwin_smm | 0.41 | internal research only (no licence, a citation request): no data, derived data or weights released, results reported with citation; tier INTERNAL | 6 subjects, stereotypy intervals | the stereotypy side of 4.9 |
| asdpose | 12.9 | internal research only (the data sharing statement gives access to everyone and allows analyses for any purpose, with no explicit data licence): no data, derived data or weights released, results reported with citation; tier INTERNAL | 241 children, 7,352 stereotypy segments | the only realistic stereotypy training source held on disk |
| osf_fmd_eeg | 0.054 | internal research only (no licence on the OSF node): no data, derived data or weights released; patient data stays local, aggregate results only, with citation | 37 subjects, derived time-frequency EEG only | any functional-disorder neural analysis |
| osf_fmd_kinematics | 0.117 | internal research only (no licence on the OSF node): no data, derived data or weights released; patient data stays local, aggregate results only, with citation | 10 participants, head, eye and gaze kinematics | the functional-disorder body side |
| leuven_cp_gait | 0.036 | admitted: CC0 | spastic CP Vicon curves; 357 children with CP and 56 typically developing children per its source | a partial paediatric kinematic reference, and the healthy child norm of 0.3 if its typically developing curves are on disk |
| mobile_gaitlab | 0.012 | excluded; counted as not held in DS | the code repository and a Gillette demo (testing only); the 1,026-subject CP set (SimTK, CC BY-NC) is not on disk (licence research 2026-09-24) | nothing: treated as not held |
| mini_rgbd | 7.3 | admitted: CC BY-NC-SA 4.0, tier INTERNAL; its SMIL model files are never shipped | 12 infant sequences | nothing here (infants excluded in version 1) |

The other five formerly rights_unclear entries touch nothing in this addendum. clinvar is admitted under the NCBI policy, with attribution, never for diagnosis. kamath_snpc_geo is admitted (NCBI GEO, no restriction). ataxie_kinect_dib is admitted (CC BY 4.0 via the article) and holds a supplementary table only, no skeleton data. panelapp is reference only (Genomics England terms, non-commercial, all rights reserved) and never enters a released artefact. kaggle_spiral is excluded: its CC BY-NC-ND label was set by an uploader who is not the creator. The non-commercial admissions (pads, mini_rgbd, panelapp) hold while MD_FM stays non-commercial research, which the design already assumes by releasing weights under CC BY-NC-SA.

### 1.5 What the in-scope counts say

- Harmonized EEG: 8 sources, 3,634 records, 566 subject ids, 11,995.7 h (EEGm, S). seizeit2 holds 11,626.25 h and cap_sleep 331.52 h; the PD and healthy EEG is 37.9 h over 406 subject ids. The three sets declared synchronized hold 9.3 h of EEG; physionet_multimodal_gait (2.93 h) fails G15 (2026-09-28).
- Invasive recordings held raw, none under a contract: ds004998, oxford_dystonia_lfp, merk_neumann_ecog_lfp, the sensefog Percept streams (NI).
- Movement-disorder patients with brain and body on one clock: 59 to 64 (sensefog 11, multimodal_fog 12, ds004998 20, merk 11 to 16 PD, oxford 5 cervical dystonia counted by labelled movement channels; RT, S).
- The dystonia pair reduces further. Ramp files with a head accelerometer: 8 hemispheres of 4 patients (P2, P3, P6, P7), 6.69 h of the 8.38 h of ramps. Of those, 6 hemispheres of 3 patients (P2, P3, P6) also carry neck EMG. P5 has EMG and two triaxial accelerometers in its single-pulse file only (0.19 h). P1 has 2 ramp files (0.28 h) whose only aux channels are three identically named AUX1 traces, unidentified. P4 has no movement channel in any of its 12 files. Single-pulse files exist for P5 left, P6 left and right, P7 left and right; one paired-pulse file for P7 left (S).
- Ramp file durations are uneven: 115 to 506 s for P6 and P7, 453 to 883 s for P3, 768 to 1,018 s per file for P2 (S).
- Stimulation during oxford ramps is at 130 Hz. On P3_L5_Testing.mat it alternates about 30 s ON blocks per 0.5 mA step with 15 to 25 s OFF blocks (RT; P6_R3_Testing.mat shows 29.4 s OFF then 83.7 s at 2.0 mA). The table of all 62 ramp files (reports/oxford_lfp/, 2026-09-28) finds 46 ascending ramps, 16 single-block files and no down-ramp; ON blocks last a median 65 s in P2 and 41 s in P3, so about 30 s does not hold for every patient. The command channel precedes the first artefact pulse by 200 to 260 ms (Am1 files) or 5 to 41 ms (Cur1 files), with up to 188 samples of variation within a file, so block onsets and offsets come from the LFP artefact and the command channel gives the amplitude only.
- ds004998 medication order is a verified confound: in 18 of 20 subjects every MedOff run precedes every MedOn run, with 21 to 77 min between blocks; sub-AbzsOg and sub-BYJoWR interleave, and in sub-BYJoWR one MedOff run at 09:20 follows the only MedOn run at 08:24, a possible label error for the owner (S, from the scans.tsv acq_time columns). Runs by state and task: MedOff HoldL 16, HoldR 9, MoveL 20, MoveR 8, Rest 3; MedOn HoldL 17, HoldR 9, MoveL 16, MoveR 6, Rest 4. An order-only null is feasible: 11 of 20 subjects have at least two MedOff runs of the same task, and 19 of 20 have at least two MedOff runs of any task (S). sub-iDpl28 has MoveL runs only in MedOff and MoveR runs only in MedOn (S), so its move runs cannot be task-matched.
- seizeit2 seizure census over all 2,850 events files: 883 seizures; hyperkinetic 270 in 30 subjects, of which the top 4 subjects hold 171 (63 percent); explicitly non-motor 324 in 51 subjects; automatisms 68 in 28; focal to bilateral 55 in 37; tonic 26 in 10; unknown or unspecified 133 in 28; 11 subjects carry both hyperkinetic and explicitly non-motor seizures (S). These are epileptic seizure types, not hyperkinetic movement disorders.
- No in-scope recording pairs a hyperkinetic scale with EEG or LFP (HI). Tics, chorea, myoclonus as raw signal, levodopa-induced dyskinesia and status dystonicus have no neural recording in scope.
- Head sensors: of the walking sets, only physionet_multimodal_gait carries a head IMU (all 167 records). sensefog has lower back and both ankles; multimodal_fog has wrist, shank and sometimes lower back; neither has a head sensor (IMUm, S). This decides where G20 can be a kill gate; since physionet_multimodal_gait fails G15 and serves PRE only (2026-09-28), that is oxford alone.
- pallidal_units_dystonia: 1,714 units, 32 patients; patients per gene TOR1A 9, SGCE 5, VPS16 5, THAP1 4, PANK2 3, GNAL 2, KMT2B 2, AOPEP 1, PLA2G6 1; units per gene TOR1A 434, SGCE 384, VPS16 345, PLA2G6 169, PANK2 119, KMT2B 94, THAP1 74, GNAL 50, AOPEP 45 (S).

### 1.6 Corrections for the owner (applied in MoveAtlas by the owner, never by a session; the not_held ids named here are registry facts, not data sources)

Items 1 and 5, the licence-row half of item 9 (a CC0 row for merk_neumann_ecog_lfp) and the fog_turning_filho participant count (35, not 50) were corrected by a verified patch against MoveAtlas a5d3814 (docs/moveatlas_corrections/), which the owner applied to the MoveAtlas working tree on 2026-09-24 and committed as MoveAtlas f2e6f99. Item 1 has a second half: the MoveAtlas canonical manifest gives modys_video one subject id per video (94 ids for 34 children), so every MD_FM split of this source uses the child map of reports/modys_subjects/ (scripts/probe_modys_subjects.py).

1. modys_video has 34 distinct ids in data_clinical_scoring.xlsx (S), not 63 as in the registry, the datacard and the design. The design's modys_video lockbox becomes 7 children and its training portion 27; the source stays above the 30-patient backbone threshold by 4; the modys_video numbers of design G8 and G10 must be recounted.
2. modys_video is longitudinal: 95 videos over T0, T3 and T12, with 29 ids at three time points and 32 with both T0 and T12 (S). The design's statement "no longitudinal pathology signal on disk" does not hold for it. What happened between time points is not stated in the files read.
3. seizeit2 EEG and EMG are at 256 Hz (EDF headers and EEGm, S), not 250 Hz as its datacard and README say; and its two modules sit on the upper back (EEG, electrodes behind the ear) and the left chest (ECG and EMG, electrodes to the left deltoid), not on the neck (README, S).
4. oxford_dystonia_lfp holds 68 .mat files (S), not 77; P5 has a single-pulse file and no ramp; P1 has two ramp files and no labelled movement channel.
5. mrc_lfp_dystonia duplicates oxford_dystonia_lfp (same portal page and cohort) and still says it needs the owner's MRC portal login.
6. dystonia_alcohol_bfm holds a co-investigator list and one aggregate family-history table, no per-patient BFM, against docs/interventions.csv and its datacard.
7. osf_fmd_kinematics holds head, eye and gaze kinematics (10 participants, 3 sessions, about 221 Hz), not limb kinematics; osf_fmd_eeg lists 37 participants, not about 56.
8. cap_sleep chin EMG is canonical for 3 records only (EMGm, S); its limb EMG channels stay unidentified in aux.
9. The inversion_testbed ids (merk_neumann_ecog_lfp, oxford_stn_lfp, lead_dbs) have no row in docs/datasets_index.csv, and merk_neumann_ecog_lfp had no row in docs/license_matrix.csv, which the program's rules name authoritative; the owner added its CC0 row on 2026-09-24 from the patch of docs/moveatlas_corrections/.
10. reports/probe_registry/license_classes.csv in this tree classes cervical_motion_multiplanar as ND against the CC BY 4.0 of its datacard and of LM: a probe error to fix in scripts/.
11. ds004998: 2 of the 22 participants of participants.tsv (sub-BgojEx, sub-zxEhes) are absent on disk; the medication order is confounded with time on the table (1.5).
12. merk_neumann_ecog_lfp: grip force is z-scored, so it carries no physical unit (DC), and run durations are 39 to 569 s, median 288 s (S), against the datacard's "10-25 min/run".
13. fog_turning_filho: the freezing flag is a column of the IMU text files inside IMU.zip (S), and it is a specialist video label, not an IMU-derived one. Per the data paper (Frontiers in Neuroscience 2022, PMC8904564), two movement disorder specialists annotated the video together by consensus, and the per-sample flag and the PDFEinfo episode table are that annotation transferred to the IMU timeline by an undocumented procedure; no inter-rater agreement is reported. In all 71 turning sessions the flag runs equal the PDFEinfo episode count and the flagged time equals the PDFEinfo total time in freezing within 1 s (reports/fog_videos/), so the IMU files carry per-episode timing. The one Physilog 5 at 128 Hz sits on the shank of the more affected side, not on the lower back that the canonical IMU manifest records. Videos.zip was fetched and verified on 2026-09-24 (0.1).

### 1.7 Not verified

- Medication state in multimodal_fog (except patient 008), merk_neumann_ecog_lfp, ds004584 and ds007526.
- The muscles of the ds004998 EMG; the identity of the second oxford accelerometer (AcS) in P6 and P7, which decides whether head-on-trunk tilt is measurable there; the task or posture during oxford ramps; the identity of the three AUX1 channels of P1.
- The seizeit2 cross-module clock in the 1,766 runs that G15 cannot check; the Percept unit, filter corner and group delay.
- The ds007526 amplitude scale; the content of striatal_gait_6ohda.
- Whether the sensefog PatientEvents markers (Off, On, Dyskinesien Anfang, in 8 files) fall inside recorded sessions; whether dyskinesia occurs in ds004998 MedOn runs.
- The care between the modys_video time points.

## 2. Architecture of the neural extension

### 2.1 Two measured objects, one clock, one direction

- The body is as designed: q(t) on the 100 Hz internal clock, descriptors d(t) by fixed formulas, EMG 20 ms envelopes as body measurements (design sections 1 to 3).
- The neural state n(t) is a posterior over short-time spectral and event structure, per recorded site, on the same clock. Sites: scalp positions (10-10), MEG sensor groups, each lead contact or montage pair, each ECoG contact, each EMG muscle for its drive band.
- A site that was not recorded is not measured. The model never estimates pallidal or subthalamic activity from scalp EEG, the neural counterpart of the design's refusal to lift depth. DBS-evoked cortical potentials in scalp EEG are reported as stimulation responses, never as subcortical state (RT).
- One direction, enforced four ways: separate parameter sets; the typed attention mask forbids body positions from attending to neural tokens; every body quantity passed to the neural side is stop-gradient; no statistic computed on neural data touches the body. G17 checks the result.
- EMG is the one signal both objects read, through different bands. Its envelope stays a body measurement. Its content above 20 Hz, its burst timing and its coherence with neural channels belong to the neural state.

### 2.2 Neural contracts

MoveAtlas has no contract for LFP, MEG, ECoG or units (rule R1; NI) and is frozen under review. The contracts are readers in <repo>/scripts that read the raw folders read-only and write canonical neural tensors to a gitignored cache. They are offered to the owner as the MoveAtlas R1 lane later, never applied there by a session.

- N-EEG, scalp and wearable. Channel identity is a 10-10 position with template coordinates; a bipolar derivation is an ordered pair. Declared reference (Pz in ds004584 and physionet_multimodal_gait, mastoids in multimodal_fog, undocumented in sensefog); unit and per-record log-gain as nuisance; native rate; delivered filters and ICA status; line frequency; validity mask. ds007526 is refused until its amplitude is traced.
- N-MEG. 204 planar gradiometers and 102 magnetometers with positions, continuous head position, empty-room covariance where present. Year 1: sensor-level descriptors per sensor triplet.
- N-LFP. Per contact: lead model, contact level, hemisphere, recording mode, montage, rate, stimulation channels as covariates, validity mask. No imaging localizes contacts for any in-scope cohort, so the token identity is lead level and montage, not a nucleus, and region-level statements ("zona incerta tracks the phasic change") are forbidden for every set without contact localization (RT). The declared target enters only as a coarse location prior in the documentation of a result.
- N-Percept. BrainSenseTimeDomain at 250 Hz rebuilt on the TicksInMses clock with lost ticks masked; shipped Gain as nuisance; absolute dates and device serials dropped at read time; an ECG artefact mask; the hardware filter corner recorded and the group delay estimated from the artefact edge before any phase-lag readout (RT). Descriptors on Percept streams are ratios in dB against a within-record baseline, never microvolts.
- N-ECoG. Contacts with strip identity and region as declared by the source; montage declared.
- N-EMG drive band. EMG at 500 Hz or more (sensefog 1000, multimodal_fog 500, ds004998 2000, oxford 4096) and seizeit2 at 256 Hz, rectified, for coherence and burst timing. The physionet_multimodal_gait EMG is a 16.05 Hz envelope and has no drive band.
- Units, aux only. pallidal_units_dystonia features and the NWB units are analysed as tables, never as tokens of an encoder.

### 2.3 Neural descriptors

Fixed formulas on the neural posterior, each with an interval, a status, a bandwidth statement and, from RT, an explicit time and frequency resolution beside it:

- periodic power in dB above the aperiodic fit, peak frequency and bandwidth, in pre-registered bands: theta-alpha 4 to 12 Hz, low beta 13 to 20 Hz, high beta 20 to 30 Hz, gamma 60 to 90 Hz. Estimation windows: at least 2 s for 4 to 12 Hz and for aperiodic fits, at least 1 s for beta, emitted at a 0.25 s hop;
- aperiodic offset and exponent, separated from the periodic part in the same fit;
- burst rate per minute and burst duration in ms, detected on native-rate envelopes, never on the 0.25 s grid;
- event-related desynchronization and synchronization in percent against a pre-registered baseline window;
- coherence and phase lag in ms from the phase slope, computed at native rate between sites and between a site and a muscle;
- stimulation responses: change per mA in dB per mA, and, with single pulses at 4096 Hz, evoked response amplitude, frequency and latency.

Bandwidth and stimulation rules: a 250 Hz stream (Percept) carries nothing above 125 Hz and the 256 Hz ear EEG nothing above 128 Hz; a gamma descriptor is refused where its band is not carried. Every gamma estimate carries the stimulation frequency; a band around half the stimulation frequency is excluded or entrainment is modelled explicitly; scalp-EEG gamma is refused in cervical dystonia and during walking unless a muscle-artefact regressor passes a pre-registered test (RT).

### 2.4 Encoders and self-supervised pretraining

- Scalp EEG: one encoder across montages, about 25M parameters (estimate): per-channel convolutional stems on 0.25 s patches, channel tokens with position embeddings, a transformer over channels and time. Objective: masked prediction of the log-spectra of masked channel-patches. No identity-contrastive objective, no label.
- Mixture, neural-only in-scope sources: seizeit2 (125), cap_sleep (35), ds004584 (149), ds007526 (144, only after its amplitude is traced): 453 subject ids and about 11,984 h (EEGm), minus lockboxes. Subject-balanced sampling over these four gives seizeit2 27.6 percent of draws against 97 percent of hours. Since 2026-09-28 physionet_multimodal_gait (59 subjects, 2.93 h) joins the mixture: it fails G15 and serves PRE only.
- Kept out of pretraining: ds002778 (EVAL); the synchronized sets sensefog and multimodal_fog (whose subjects must stay unseen by the encoder); every seizeit2 seizure label.
- Lockboxes, 20 percent of the persons of sources with at least 30, drawn per group with seeds derived from the source names and frozen with a SHA-256 in lockbox v1.2 on 2026-09-28 (scripts/build_tiers_and_lockbox.py, reports/lockbox/): ds004584 30, ds007526 29, seizeit2 25, cap_sleep 7, physionet_multimodal_gait 12.
- Leakage, stated because it cannot be fully removed (RT): design stage 1 trains the healthy body prior on the free acceleration of physionet_multimodal_gait (57 subjects), so bodies for the 12 neural-lockbox subjects would come from a model that saw their IMU. Either those 12 subjects leave design stage 1, or the exposure is printed beside every number of family F3. Since 2026-09-28 F3 does not run (the set fails G15), and lockbox v1.2 holds these 12 persons in every modality, their IMU included (reports/manifest_tiers/). Cross-fitted G16 figures on subjects the encoder saw unlabelled are labelled transductive; only the lockbox read is decisive.
- LFP and ECoG: the fixed spectral parametrization is the front end. A learned temporal model (about 3M parameters) is admitted only by G16, and its selection target is rest against move or hold blocks, never the medication labels that Q4 and G21 later report on (RT). MEG: fixed features only in year 1.
- Motion artefact: head and segment accelerations predicted by the body posterior enter as stop-gradient regressors, strictly causal (body samples at or before the neural sample). Where no head sensor exists on the recording, the regressor is prior-only and is labelled so, and no gait-phase claim is made on that set (RT).

### 2.5 Coupling module, two regimes

RT showed that one 0.25 s grid cannot serve both a 20 ms corticomuscular lag and a 10 s freezing horizon. The module has two regimes, declared per family:

- Fast regime, lags under 200 ms: phase slope, cross-correlation of rectified EMG with band-limited LFP or EEG, and conditional directed coherence, all at native rate (at least 500 Hz; 4096 Hz on oxford). Lags are reported in ms with the standard error of the group-delay estimate from coherence and bandwidth, and a direction is claimed only when that interval excludes 0. Families F1 (fast part), F5 and F8 and question Q2 use only this regime.
- Slow regime, lags of 0.5 s and more (freezing hazard, stimulation dose, state): y(t) = a + sum over b of beta_b (phi_b * x)(t) + sum over b of gamma_b (phi_b * y)(t - 0.25 s) + delta u(t) + e(t), on a 4 Hz grid, with five fixed B-spline lag kernels over -2 s to +2 s (the causal half only for predictive claims); u holds stimulation amplitude and contact, medication state and task phase; subject-level random slopes under a population prior; under 200 population parameters per family.
- Artefact and re-afference removal is strictly causal: only the part of the neural signal predicted by past body samples is removed, plus an accelerometer-locked artefact template. A two-sided kernel would remove the driving component itself and bias every direction test toward re-afference (RT).
- Nulls, all within subject: circular shift by at least three integrated autocorrelation times; cross-record swaps within subject where records are short (physionet_multimodal_gait records are all 59.87 s, multimodal_fog records have a 10th percentile of 25.5 s, so a 30 s floor with 1,000 shifts does not exist there); a stride-cycle shuffle that keeps within-stride structure. The rule file states, per set, the number of distinct admissible nulls k and the smallest attainable p of 1/(k+1). A record with fewer than 20 admissible nulls enters no pass decision (RT).
- Outputs: gain in body units per neural unit (standardized and flagged unitless where the movement channel is z-scored, as in merk), lag with interval, direction, null percentile, number of subjects, artefact status. There is no neural-to-body generative decoder.

### 2.6 Neural arm of the intervention operator

Masked, monotone displacements of neural descriptors, under 20 parameters plus per-subject gains, fitted by cross-fitting. Two arms exist in scope:

- Levodopa on STN and cortical spectral descriptors (ds004998, MedOff against MedOn within subject), reported only beside the order-only null of Q4 (feasibility in 1.5), and labelled "state or time" when it does not beat that null.
- Stimulation amplitude on low-frequency and broadband LFP power (oxford ramps, within file), block-paired ON against flanking OFF. This arm is analysis-lane only: oxford is CC BY-SA 4.0 and never enters released weights (G22).

It never displaces body descriptors; the body operator of the design is unchanged.

### 2.7 What is released and what stays internal

- RELEASE: reader code and descriptor formulas; the scalp-EEG encoder trained on the CC0, ODC-By and CC BY 4.0 mixture (seizeit2, ds004584, ds007526, cap_sleep, and physionet_multimodal_gait since it failed G15), which inherits no NC, ND or SA term and can ship as a checkpoint separate from the body model's CC BY-NC-SA weights (an owner decision); coupling-atlas tables with per-source attribution.
- Never in released weights: oxford_dystonia_lfp and the stimulation arm fitted on it (CC BY-SA 4.0, which cannot mix with the BY-NC-SA body model); every EVAL label (seizeit2 seizures, ds002778 medication states, CODY); every source admitted on 2026-09-24 for internal research only (goodwin_smm, asdpose, osf_fmd_eeg, osf_fmd_kinematics) or as reference only (panelapp).
- merk_neumann_ecog_lfp carries a CC0 row in the authoritative licence matrix since 2026-09-24; rule R1 still keeps it an evaluation target.
- Privacy: per-record neural embeddings are not released, because EEG is biometric; Percept dates and serials are dropped at read; the raw-video rule is unchanged; the ds002778 curators ask for a courtesy note before publication (owner action).

### 2.8 Compute and environment

- Readers and preprocessing on CPU. The largest input is ds004998, 173.7 GB (DS). seizeit2 at 256 Hz, 2 channels and float16 is about 43 GB of cache.
- Measured on this machine: the base interpreter has h5py 3.12.1 and scipy 1.13.1 and has neither mne nor torch (S). Stage N0 therefore starts with the oxford .mat readers through h5py and scipy on CPU, which need nothing new. The FIF readers for ds004998 and every GPU step are blocked on a dedicated MFM environment, which is owner question 4 of handoff 1.
- EEG encoder: about 100 GPU-hours on the local RTX 5080, scheduled after the design's 60M pilot passes G1, since the GPU is shared. LFP temporal model: under 10 GPU-hours. Coupling fits and nulls: CPU hours. The neural extension needs no cluster time.

### 2.9 First 12 months under the scope

- Month 0 to 1, stage N0, no training: oxford readers (68 files) in h5py and scipy; the step structure of all 62 ramp files tabulated from the Cur and Am channels, including any down-ramp segments, which break the collinearity of dose and time where they exist (RT); the seizeit2, sensefog and physionet_multimodal_gait synchronization checks of G15; the neural rule file with the questions, the families, the number of tests per family, the confirmatory primaries and the admissible-null counts, dated and hashed (RT); the minimum detectable difference published per G16 and G21 target; lockboxes frozen; every number of this addendum regenerated by scripts in scripts/ with outputs in reports/. Done by 2026-09-28: the oxford readers and the table of all 62 ramp files (46 ascending ramps, 16 single-block files, no down-ramp, so no file breaks the collinearity of dose and time; reports/oxford_lfp/); the G15 checks of oxford (the one-sample rule of section 6 fails in all 62 ramp files, so block timing comes from the LFP artefact, 1.5), seizeit2, sensefog and physionet_multimodal_gait (reports/oxford_lfp/, sync_seizeit2/, sync_sensefog/, sync_physionet_gait/); the lockboxes (lockbox v1.2, reports/lockbox/).
- Month 1 to 3, analyses that need no body model and no new environment: Q1 and Q2 on the oxford accelerometers and EMG, Q10 on the pallidal units, Q8 on seizeit2 with fixed formulas.
- Month 2 to 4: the ds004998 and merk readers once the environment exists; EEG encoder pretraining; G16.
- Month 3 to 6: coupling on body posteriors from the design's pilot; G17, G19, G20; Q3, Q5, Q7 (Q6 does not run: physionet_multimodal_gait fails G15).
- Month 6 to 12: Q1, Q2 and Q4 re-run on body posteriors; the first coupling atlas (U11); G21; the dystonia case-series paper on oxford; the modys_video longitudinal paper.

## 3. Scientific questions

Why this is scientifically interesting under the scope. Movement disorders are the field of neurology whose primary objects are defined by an expert's eye and scored on ordinal scales that mix amplitude, duration and frequency. The measured body supplies a quantitative space in physical units, independent of the device, with the limits of each sensor stated; the neural extension puts circuit measurements on the same clock. The scope removes the prospective arm, so what remains is: what can be measured and falsified on data already held, said exactly.

Error control. One confirmatory primary analysis per dataset for year 1, Holm-corrected across the primaries below (RT); since physionet_multimodal_gait failed G15 on 2026-09-28, its primary cannot run and the family holds six unless the owner names a replacement:

| dataset | confirmatory primary |
|---|---|
| oxford_dystonia_lfp | Q1, the block-paired acute phasic head-movement contrast, level by amplitude |
| pd_meg_stn_lfp_ds004998 | family F1, STN beta to EMG tremor-band power and movement-block onsets, against the order-only null |
| physionet_multimodal_gait | none since 2026-09-28: the set fails G15 and serves PRE only, so family F3 (sensorimotor EEG to gait phase) cannot run; a replacement is the owner's decision |
| merk_neumann_ecog_lfp | family F4, ECoG and STN beta and gamma to grip force |
| seizeit2 | Q8, hyperkinetic against explicitly non-motor seizures from fixed body descriptors |
| modys_video | Q9, each DIS subscale monotone in its own body axis within source |
| pallidal_units_dystonia | Q10, one pairwise gene-or-surgical-correlates contrast (TOR1A against SGCE) |

Everything else in this section is exploratory: intervals and counts are reported, and the word falsified is not used. Sets under 30 patients report intervals without kill. Gate outcomes are three-valued: pass, absent, undetermined (RT).

| rank | question | status | in-scope data |
|---|---|---|---|
| Q1 | Acute stimulation response of the dystonic neck, block by block, and which LFP change tracks it | confirmatory, case series | oxford_dystonia_lfp, 8 hemispheres of 4 patients, 6.69 h |
| Q2 | Drive or re-afference in the stimulation-free blocks | exploratory, fast regime | oxford, 6 hemispheres of 3 patients with EMG; 3 single-pulse files with EMG (P5, P6 left and right) |
| Q3 | The body posterior as a motion-artefact regressor | enabling | oxford (physionet_multimodal_gait fails G15 and serves PRE only since 2026-09-28); sensefog and multimodal_fog as prior-only |
| Q4 | Does the neural state add to the body for levodopa state | exploratory | ds004998 (20); ds002778 (15 PD, EVAL) |
| Q5 | Neural premium for freezing; freezing against voluntary stop | exploratory | multimodal_fog (12); sensefog (4 freezers, 198 s) |
| Q6 | Healthy gait brain-body reference | not run: physionet_multimodal_gait fails G15 (2026-09-28), so F3 cannot run | physionet_multimodal_gait (57 with EEG and IMU) |
| Q7 | How much of the neural state movement predicts | confirmatory (F4) | merk (73 runs); ds004998 |
| Q8 | Ictal motor semiology as a label-free test of the descriptors | confirmatory | seizeit2 (125 subjects) |
| Q9 | Continuous axes or categories in dyskinetic cerebral palsy | confirmatory, body only | modys_video (34 children) |
| Q10 | Gene, or its surgical correlates, at the pallidal cell level | confirmatory | pallidal_units_dystonia (19 patients in the 3 genes with at least 5: TOR1A 9, SGCE 5, VPS16 5) |
| Q11 | Can a kinematic trigger replace an EMG trigger for back-averaging | method, exploratory | sensefog (EMG 1000 Hz, IMU 128 Hz) |
| Q12 | Tremor separation on body descriptors, within source | exploratory | pads, drummer_tremor_model, monipar, ds004998 hold runs, oxford head |
| Q13 | Sleep staging and REM muscle tone as a descriptive probe | exploratory | cap_sleep (35 ids, 34 persons) |

Q1. Within an oxford ramp file, what changes acutely in the neck between a stimulation block (ON a median 65 s in P2 and 41 s in P3) and its flanking OFF blocks, and which LFP change accompanies it?
- Data: ramp files at contact levels 2 to 7 with a head accelerometer, 8 hemispheres of 4 patients, 6.69 h; 6 of those hemispheres also carry neck EMG (S).
- Pre-registered expectation, stated before any read: a therapeutic dystonic effect, phasic or tonic, is not expected within one ON block of STN, zona incerta or VL stimulation (a median 65 s in P2 and 41 s in P3); the addendum's own reading of the literature places dystonic benefit at hours to months (FB, SI). Any body change at this timescale is reported as an acute effect or a side effect, not as benefit. Steps at and one below the documented side-effect threshold are excluded from any benefit reading (RT).
- Test: block-paired contrast per amplitude, ON block against its flanking OFF blocks, discarding 5 s after each transition, against a null that permutes ON and OFF labels within each amplitude pair (RT). The primary is a level-by-amplitude interaction, because each hemisphere has files at six contact levels (16 of the 62 ramp files hold a single ON block, not a ramp; reports/oxford_lfp/) while time drift is shared across files and contact efficacy is not; the null permutes level labels across whole files within hemisphere (RT). One body primary, 1 to 12 Hz head-acceleration power; one neural primary, 4 to 12 Hz periodic power on the bipolar pair around the stimulated level. Every other descriptor is exploratory, and a maximum over descriptors is tested with a max-statistic bootstrap.
- Unit of independence: the patient (4), with hemispheres nested. Counts, intervals, no kill.
- What it cannot say: nothing about GPi, nothing about programming, nothing about therapeutic latency, nothing about yaw (a head accelerometer measures tilt against gravity, not rotation about the vertical, which is the commonest torticollis component), and nothing about head-on-trunk posture unless the second accelerometer of P6 and P7 proves to sit on the trunk (1.7).

Q2. In the stimulation-free blocks, does deep low-frequency activity lead neck EMG, or follow it?
- Data: oxford OFF blocks of the 6 hemispheres with EMG (P2, P3, P6), with OFF durations tabulated per file in reports/oxford_lfp/, and the single-pulse files of P5 and P6 (left and right), whose evoked EMG latencies are the causal anchor; P7 has no EMG, so its two single-pulse files give accelerometer responses only (S).
- Test: fast regime only, at 4096 Hz. Phase slope and conditional directed coherence in 3 to 12 Hz between LFP and EMG, before and after removing the part of the LFP predicted by past kinematics plus the accelerometer-locked artefact template. The circular-shift null uses the admissible-null rule. Total stimulation-free time per hemisphere is reported beside every estimate, and the first 5 s after each offset is discarded.
- Reading: a drive account predicts the LFP leading the EMG by a positive lag that survives the causal removal; a re-afference account predicts the EMG and kinematics leading, or the coupling vanishing. With both present, a phase slope is a mixture, so the single-pulse anchor decides sign where it exists.
- Framing: this is a replication in STN, zona incerta and VL of the pallidal drive analysis of Sharott et al. 2008 in at most 3 patients, not an answer to a question the field left open (RT).

Q3. Can the body posterior remove mechanical artefact from EEG and LFP without removing task-related neural modulation?
- Data: physionet_multimodal_gait (a head IMU in all 167 records) and oxford (head accelerometer); since 2026-09-28 physionet_multimodal_gait fails G15 and serves PRE only, which leaves oxford. sensefog and multimodal_fog have no head sensor, so their regressor is prior-only.
- Test and decision: G20, which is semi-synthetic. A known gait-phase-locked beta modulation and a known desynchronization are injected into recorded EEG at pre-registered depths, and the gate asks for recovery of the injected modulation as well as harmonic reduction. Without that, a regressor that removes stride harmonics can remove the real gait-phase modulation and still pass (RT). multimodal_fog arrives filtered and ICA-cleaned and is excluded from the harmonic-reduction criterion.
- Consequence: failure voids every walking or dystonic-movement coupling claim on the set concerned.

Q4. Does the neural state add to the body in estimating levodopa state?
- Data: ds004998, 20 PD with MedOff and MedOn runs in hold, move and rest, and MDS-UPDRS-III item tables; the body proxy is EMG only (no accelerometer). ds002778 (15 PD ON and OFF, rest EEG) for the neural-only side, evaluation-only.
- The confound, measured: in 18 of 20 subjects every MedOff run precedes every MedOn run by 21 to 77 min (1.5). Any within-subject classifier can learn time on the table.
- Test: within subject, task matched (hold with hold, move with move), leave-one-subject-out AUROC for STN beta alone, EMG descriptors alone, and both. An order-only null is trained first with the same pipeline to separate earlier from later task-matched MedOff runs, which 11 of 20 subjects support. The medication result is reported as a medication effect only if it exceeds the order-only AUROC beyond the subject-bootstrap interval; otherwise it is reported as "state or time" (RT). sub-AbzsOg and sub-BYJoWR are reported separately.
- Further confounds: postoperative day and the stun effect attenuate beta; hold runs carry re-emergent postural tremor, not rest tremor.
- Decision rule: pass, absent or undetermined (G21). Only "absent" demotes U7.

Q5. Do EEG or STN LFP add to the body in predicting freezing seconds ahead, and do they separate freezing from a voluntary stop?
- Data: multimodal_fog, 12 PD with dense freezing labels; sensefog, 198 s of labelled freezing in 4 subjects and 0.047 h of voluntary stops, with the Percept LFP above the per-session admissible-lag floors of G15 (reports/sync_sensefog/).
- Test: 5-fold patient-level cross-fitting, event-level average precision at a 2 s horizon, joint against body-only at equal parameters, three seeds. With 12 and 4 subjects the interval is wide by construction, so the outcome is three-valued and the minimum detectable gain is published in N0.
- Limit: no gait-phase claim on either set (no head sensor), and no direction claim on the sensefog LFP at beta-phase scale in single-anchor sessions, nor for lags under a session's admissible-lag floor (G15, reports/sync_sensefog/).

Q6. What are the normal lags and gait-phase modulation between cortex, muscle and body?
- Data: physionet_multimodal_gait, 57 subjects with EEG and IMU, 3 treadmill speeds, every record 59.87 s.
- Test: family F3, after G20 passes on this set. Since 2026-09-28 it cannot run: the set fails G15 (lags between EEG, IMU and force are not constant across speeds, reports/sync_physionet_gait/) and serves PRE only. Nulls are built from cross-record swaps within subject across speeds and from a stride-cycle shuffle, with at least 20 admissible nulls per held-out subject.
- Value: the healthy lag and modulation reference every disease coupling needs, on the only in-scope walking set where the artefact regressor is measured rather than prior.

Q7. How much of the subthalamic and cortical spectral state does measured movement predict?
- Data: merk (grip force in all 73 runs, z-scored, ECoG in 73, STN in 20; runs 39 to 569 s) and ds004998 (EMG with STN and MEG).
- Test: leave-one-patient-out prediction of beta and gamma power and burst rate from body variables against a temporal-permutation null sized by the admissible-null rule; the explained fraction is reported. Gains are standardized and flagged unitless.
- Value: a bound on what wearables can say about neural state; the unexplained part is what only sensing supplies.

Q8. Do the body descriptors mean what they claim on independent expert labels of motor events recorded with a neural channel?
- Data: seizeit2, all 125 subjects. Q8 trains nothing, so its body part does not need the encoder lockbox; if the EEG change latency is read from the encoder rather than from fixed formulas, that part is reported on the seizeit2 encoder lockbox only (RT). Classes: hyperkinetic 270 seizures in 30 subjects (top 4 subjects hold 63 percent), explicitly non-motor 324 in 51 subjects, tonic 26 in 10 subjects. The accelerometer channels are not scaled in g (resting norm median 0.168, reports/sync_seizeit2/), so a fixed-threshold body descriptor applies only after their scale is corrected.
- Test: fixed formulas on the 25 Hz IMU and the 256 Hz deltoid EMG separate hyperkinetic from explicitly non-motor seizures, with patient-clustered AUROC (cluster bootstrap, and a per-patient averaged figure beside it). Seizures labelled unknown or unspecified are not used as negatives. The tonic against hyperkinetic contrast is reported only if at least 10 patients per class carry it, which the census makes unlikely.
- Timing: the EEG-related number is reported as "behind-the-ear EEG change latency relative to the annotated onset", never as ictal onset, because two behind-the-ear channels cannot date the onset of extratemporal seizures (RT).
- Naming: epileptic hyperkinetic seizures are not hyperkinetic movement disorders. Q8 tests descriptors, not a disease.

Q9. In dyskinetic cerebral palsy, are dystonia and choreoathetosis two categories or two continuous axes?
- Data: modys_video only. Pooling it with li_taati and hd_accel_jnet would let a mixture recover source, diagnosis and device rather than phenomenology (RT); the other sources serve as within-source replications.
- Test: mixtures of factor analysers with K from 1 to 3 on the two body axes after marginal Gaussianizing transforms, K greater than 1 tested by parametric-bootstrap likelihood ratio at patient level; each DIS subscale monotone in its own axis (Spearman at least 0.5) and at most 0.3 on the other, per region. Reported as descriptive at 34 children.
- CODY: the between-centroid test happens only inside the sealed evaluation script of design G7, from rule-file formulas, and the centroids are discarded after the read.

Q10. Do monogenic dystonias carry distinct pallidal firing signatures?
- Data: pallidal_units_dystonia. The CSV holds patient, gene, hemisphere, trajectory, depth and 38 unit features, and no age, anaesthesia, centre or era column (S). Age at surgery and anaesthesia both covary with gene and both change firing, and the covariates cannot be obtained inside the scope.
- Test: pairwise leave-one-patient-out contrasts between genes with at least 5 patients (TOR1A 9, SGCE 5, VPS16 5), with depth and trajectory as covariates, against a label-permutation null. Firing-rate-free features are reported separately, since rate is the feature most sensitive to anaesthesia.
- Title of the result: "gene or its surgical correlates". Any positive result may reflect surgical context. The body-side stage is outside the scope (appendix).

Q11. Can a kinematic trigger replace an EMG trigger for jerk-locked back-averaging?
- Data: sensefog, which has 1000 Hz leg EMG and a 128 Hz ankle IMU on a verified clock, and contains voluntary steps in PD, not jerks.
- Test, by simulation on real data (RT): measure the distribution of the EMG-onset to IMU-onset delay in sensefog, convolve a 10 ms and a 20 ms template cortical transient with that jitter, and report the attenuation and latency bias. A 128 Hz IMU adds 7.8 ms quantization. The kinematic route is admissible only where the predicted attenuation is under 30 percent.
- What it is not: a myoclonus test. There is no raw myoclonus EEG and EMG in scope, and EMG stays mandatory for any clinical statement about a generator.

Q12. Do tremor types separate on body descriptors, within source?
- Data: pads (469 subjects, ET, PD, atypical parkinsonism, MS, healthy, CC BY-NC-SA), drummer_tremor_model (18 healthy, physiological reference), monipar, mendeley_et_accel and osf_tremor_amplitude as small ET references, ds004998 hold runs for PD postural tremor with STN and MEG, oxford head accelerometers for dystonic head tremor if present.
- Test: within pads, do irregularity, position dependence and load sensitivity add to dominant frequency and amplitude, at patient level; separately, in ds004998, is STN-cortex coherence present at the tremor frequency during hold epochs detected from EMG, against the shift null.
- Limit: dystonic limb tremor with EMG and EEG does not exist in scope, so the dystonic against essential contrast is not testable; ds004998 holds postural re-emergent tremor, not rest tremor, and comes from the group of the study it would replicate.

Q13. Sleep, as a probe only.
- cap_sleep serves the G16 staging probe and a descriptive report of REM muscle tone. Stage-to-chin-tone coupling is not a coupling family: sleep stage R is scored from chin atonia, so it would test a scoring rule (RT). Chin EMG is canonical in 3 records only and the limb channels are unidentified, so the REM-without-atonia question in the 22 RBD ids (two of them are one byte-identical recording) is an owner question for later, after the channels are identified.

## 4. Dystonia and hyperkinetic program

### 4.1 Tonic and phasic decomposition per body segment

- Definitions, fixed formulas on the body posterior per segment, task and 10 s window: tonic T is the median deviation of segment orientation from the task norm in degrees after a 0.3 Hz low-pass, with the dwell fraction beyond the norm's 95th percentile; phasic P is angular-velocity power in 0.5 to 3 Hz with the design's direction-irregularity index and in 3 to 12 Hz with rhythmicity, in dB against the norm, plus brief events per minute (durations only from sensors at 100 Hz or more). Each axis carries a status: yaw is prior-only from one camera, and an accelerometer alone measures tilt, not yaw.
- In-scope data: modys_video (DIS per region, 34 children, 25 frames per second from the frame-rate recovery of 2026-09-28, reports/nuisance/); li_taati (UDysRS per body part, 9 persons, 30 frames per second as stated by its data paper, PMC6219082); oxford (head only, acute); cody_1 to cody_3 for evaluation.
- Healthy reference for the neck: cervical_motion_multiplanar (14 asymptomatic adults, 127 records, 4.39 h), with its undocumented sensor placement stated beside every number.
- No healthy child norm for the modys_video postures is in scope (4.6), so paediatric deviation is computed within source against the lowest-DIS quartile and flagged as a within-source norm (design section 6).
- Test: Q9. Exploratory elsewhere.

### 4.2 Co-contraction and overflow from EMG

- Definitions: a co-contraction index per antagonist pair on 20 ms envelopes normalized to the record's 95th percentile (no maximum voluntary contraction exists in these sources), as overlap over 1 s windows; muscle overflow as envelope above the rest threshold in muscles the task does not require; kinematic overflow in degrees and dwell; EMG-EMG coherence at 4 to 12 Hz as common drive.
- In-scope data: oxford neck EMG, which is thin and patient-specific (P2 left and right splenius (LSpc, RSpc) in its right-hemisphere files and two unnamed channels (EMG1, EMG2) in its left-hemisphere files, P3 left splenius and left SCM, P6 right SCM and right trapezius, P5 four channels in its single-pulse file only). Healthy references: cervical_motion_multiplanar (2 channels, placement undocumented) and gait120 (4,125 EMG records, 120 subjects, 2000 Hz) for limb muscles. ninapro (hand EMG, 201 records, 67 subjects) is admitted under CC0 on Dryad (licence decision of 2026-09-24); the held zips come from the project site, and their equivalence to the Dryad copy is recorded before any derived release.
- What can be asked: within the oxford patients, does the co-contraction index covary with tonic deviation across blocks and levels. The between-group question (patients against controls holding the same posture) is not answerable: no in-scope dataset records the same neck task in patients and controls with comparable EMG placement.
- A capsular contraction driven by stimulation produces a sudden tonic EMG onset time-locked to an amplitude change; such epochs are flagged and excluded from T rather than read as dystonia (RT).

### 4.3 Neural biomarkers against continuous body targets

- Testable in scope: the acute stimulation response of Q1 and the direction question of Q2, both on oxford lead levels, both reported without nucleus labels.
- Not testable in scope: pallidal 4 to 12 Hz against a body descriptor rather than a scale total (no GPi recording in dystonia is held, and no in-scope recording pairs a scale with LFP); pallidal beta in status dystonicus; chronic cortical gamma against dyskinesia. These move to the appendix.
- Framing correction (RT): it is not true that every dystonia biomarker was validated against ordinal totals. Most were, including Neumann et al. 2017 and Lofredi et al. 2024; Barow et al. 2014 split phasic from tonic by clinical classification. The contribution the program can still claim is moving that split from a clinical classification to a continuous physical quantity, and in scope it can only demonstrate the measurement, not the biomarker comparison.

### 4.4 The DBS latency hypothesis under the scope

The mission prompt's claim 3 (the motor command changes within hours while the tonic component takes weeks to months) cannot be tested. It requires repeated measurements of the same patients after stimulation onset, which is an acquisition. Under the scope:

- The hypothesis stays recorded, untested, with its prediction written down so that it can be tested if the scope ever changes. The prospective protocol, its endpoints and its corrected sample sizes are in the appendix, not in the plan.
- The design's GPi operator keeps a population prior and a single time constant, and the counterfactual body refuses a tonic time course under dystonia DBS (U3).
- What in-scope data can still say about time scales. cdystonia gives the shape of a week-scale course of a scale under toxin (109 patients, 631 rows, TWSTRS at weeks 0 to 16, three arms), which bounds how fast a scale moves but says nothing about a body. oxford gives ON blocks lasting a median 65 s in P2 and 41 s in P3 (reports/oxford_lfp/), which say nothing about therapeutic latency in either compartment. modys_video gives a 12-month body-side change axis in 32 children with T0 and T12 (29 with three time points), but the care between time points is not in the files, so it estimates sensitivity to change, not a treatment effect.
- Nothing in this document depends on the hypothesis being true.

### 4.5 Status dystonicus

No in-scope data. The question and its ethics requirements (deferred consent, a no-delay-to-care clause, sedation and neuromuscular blockade as covariates, body readouts masked during paralysis, relapse and weaning rather than detection as the endpoint) are recorded in the appendix so that nothing is lost, and no gate depends on them.

### 4.6 Paediatric dyskinetic cerebral palsy

- In-scope data: modys_video, 34 children, 95 videos, 161 scored rows, DIS per region for dystonia and choreoathetosis with duration and amplitude, two postures, 500 frames per clip, 25 frames per second (measured on 2026-09-28 from the source's stick-figure movies, reports/nuisance/). cody_2 and the cody_3 children are evaluation only.
- Tests: Q9 within source; sensitivity to change from T0 to T12 within child, on the 32 children who have both, with the intervening care unknown and stated; agreement analysis on the 66 videos with two scorers against the 29 with one.
- Missing in scope: a healthy child norm for these postures (leuven_cp_gait, admitted under CC0 on 2026-09-24, holds gait curves only, of 357 children with CP and 56 typically developing children per its source; shanghai_child_gait3d is not held), two views or IMUs for yaw, and EMG for co-contraction.
- Contribution: a physical axis for the paediatric scale-linkage work, on 34 children, reported as descriptive.

### 4.7 Genotype and phenotype

- In-scope data: pallidal_units_dystonia only (Q10). No genotyped body data is in scope.
- Rules: the gene is a conditioning token, never a latent axis. Because intervention direction is gene-specific (FB), no hyperkinetic counterfactual is produced for a monogenic disorder without gene-specific within-subject data, which the scope does not hold.
- The cell-to-body mapping (a Mantel test between body-phenotype distance and unit-feature distance) is outside the scope.

### 4.8 Myoclonus

- In-scope data: no raw myoclonus EEG and EMG. lance_adams_myoclonus is figures. cody_1 records myoclonus frames in a minority of its patients, evaluation only, and no threshold of any gate or question is derived from that census (RT).
- Programme: Q11 only, the trigger-jitter simulation. Generator classification needs EMG burst duration, giant somatosensory evoked potentials and long-latency reflexes, and stays outside the model.

### 4.9 Tics and stereotypies

- Tics: no in-scope signal outside CODY. The design's exclusion of tics as a channel stands, and no question depends on them.
- Stereotypies: goodwin_smm and asdpose are admitted for internal research only (licence decision of 2026-09-24, section 7), tier INTERNAL: no data, derived data or weights are released, and results are reported with citation. mmasd (Apache, in scope) holds therapy activities where stereotypy is a proxy only and ADOS-2 ratings, which is not a stereotypy label; ssbd holds annotations without video.
- The first analysis on asdpose is rhythmicity, template stability across bouts and bout duration on its 7,352 annotated segments, within source, with no tic comparison possible.

### 4.10 Chorea and levodopa-induced dyskinesia

- In-scope data: li_taati (UDysRS, 9 persons), hd_accel_jnet (24 HD with UHDRS TMS, FA and TFC, and 10 controls), gaitndd (20 HD). No neural recording accompanies any of them.
- Tests: the irregular-flow descriptor against UDysRS (design G8, 9 persons) and against UHDRS TMS (24 patients), both by person-level cross-fitting and reported without kill. Since li_taati counts 9 persons (2026-09-28), its G8 UDysRS line (Spearman at least 0.55, kill below 0.40) becomes a cross-fitted estimate at person level without kill.
- Not testable in scope: whether cortical or subthalamic gamma leads body-measured dyskinesia, and any premanifest Huntington question. Both need data that is not held.

## 5. Revised use cases

Read against the seven use cases of docs/MISSION_PROMPT.md section 6, the design serves four (U1 to U4). This addendum keeps them with revisions and adds five that the scope supports. Each use case's evidence gate rests only on in-scope data. Maturity: today (fixed descriptors on held data, no trained model), 12 months, or not supported under this scope.

U1. Readouts from a phone video or a wearable (revised).
- User: movement-disorder clinician, trialist. Input: phone video or wearables, surface EMG when present.
- Output: design section 6 readouts with interval, status and bandwidth; added, per-segment tonic and phasic axes (4.1) and co-contraction where EMG exists (4.2).
- Evidence gate: design G3, G7, G8; G14 on the 71 fog_turning_filho video and IMU turning pairs (Videos.zip fetched and verified on 2026-09-24; keypointing waits on the extraction of Videos.zip, an owner action; the environment of OD4 exists); G17 for invariance.
- Maturity: 12 months for gait, bradykinesia and tremor; hyperkinetic channels as far as the first sealed CODY read allows.

U2. Monitoring across days (kept).
- Unchanged from design section 6. Under this scope the PD-specific between-day estimate comes from pd_biostamprc21, admitted on 2026-09-24 (CC BY 4.0, tier EVAL, never trained), beside the ltmm non-faller estimate; descriptors without a between-day estimate ship with monitoring disabled.
- Maturity: 12 months for healthy references; hyperkinetic monitoring not supported.

U3. Counterfactual body under an intervention (revised).
- Revisions: for dystonia DBS, the counterfactual states no time course at all under this scope, because no in-scope data constrains the kinetics; for monogenic hyperkinetic disorders, no counterfactual.
- Evidence gate: design G11 (levodopa in PD, on care_pd and tremordb as the design defines them).
- Maturity: levodopa in PD, 12 months; DBS in dystonia, not supported.

U4. Intervention effect as a research endpoint (kept for PD).
- Evidence gate: design G11 on care_pd and tremordb; its pd_biostamprc21 and kaggle_fog lines are firm: both sets were admitted on 2026-09-24 as tier EVAL (section 7). On 2026-09-28 the care_pd walk names give both ON and OFF walks for 19 BMCLab and 17 E-LC subjects, and pd_biostamprc21 has 12 PD with tagged OFF then ON windows (reports/nuisance/). Maturity: 12 months.

U5. Acute stimulation-response analysis on externalized recordings (new, research only).
- User: researchers analysing externalized dystonia recordings.
- Input: an externalized session with stimulation ramps, a head accelerometer and, where present, neck EMG.
- Output: per contact level and amplitude, the acute block-paired body change with interval, the acute LFP change per mA, and flagged capsular-contraction signatures, all labelled acute.
- Evidence gate: Q1 and Q2 with G19 and G20 on oxford; the numbers are counts on 4 patients.
- Explicitly not: programming support, a benefit readout, a tonic time course, or any statement about GPi. That use case is outside the scope (appendix).
- Maturity: today as an analysis, on the analysis lane only (CC BY-SA).

U6. Coupling-based target discovery in Parkinson disease (new).
- Input: synchronized neural and body sessions in scope. Output: neural features ranked by held-out lagged coupling with each body descriptor, with gain, lag and null percentile.
- Evidence gate: G19 families that pass (F1, F2 slow part, F4), Q4 and Q7.
- Maturity: 12 months in PD. The dystonia version is not supported.

U7. Medication state from wearables (new, restricted).
- Input at use: wrist and ankle IMU or phone video only. Output: a continuous state estimate with interval. The head never reads neural input at use, and it sits on the frozen body posterior with stop-gradient, so its neural-target training never changes the body (2.1, G17).
- Training with neural targets is restricted (RT): only hemispheres with a pre-registered periodic beta peak, only the bradykinesia and rigidity axis, with tremor and dyskinesia kept on body targets and performance reported separately for tremor-dominant patients.
- Evidence gate: Q4 must first show, against the order-only null, that the neural state carries state information beyond the body proxy.
- Maturity: body-only state 12 months; the neural calibration is a proof of concept on 20 subjects, not a product claim.

U8. Freezing prediction (new).
- Input: IMU at use; EEG or LFP in research. Output: freezing hazard 2 s ahead with interval, and the measured neural premium as pass, absent or undetermined.
- Evidence gate: design G10 cross-device freezing, Q5, G21. fog_turning_filho supplies never-trained freezing labels today: specialist video labels carried per sample in its IMU files, with per-episode timing (design G10); its video, fetched and verified on 2026-09-24, serves design G14. On 2026-09-28 the IMU files hold 173 flagged freezing runs in 42 of the 71 turning trials and in 26 of the 35 patients, and the IMU freeze-index baseline reaches an AUROC of 0.580 (95 percent interval 0.434 to 0.706) against these labels (reports/fog_imu/).
- Maturity: body-only 12 months; the neural premium measured within 12 months as a count.

U9. Kinematic triggering as an adjunct in myoclonus workup (new, narrowed).
- Output: IMU- or video-triggered averaging as an adjunct when EMG is missing, or to count jerks, with the measured attenuation and latency bias from Q11. Generator classification requires EMG and is outside the model.
- Evidence gate: Q11 attenuation under 30 percent. Maturity: method within 12 months; no clinical claim.

U10. Endpoints for paediatric dyskinetic cerebral palsy (new, narrowed from the draft's rare-disease version).
- Output: unit-bearing body axes per region with intervals, their change between visits, and the scorer-agreement context.
- Evidence gate: design G8 on the modys_video DIS probe, design G9 null controls, and the T0 to T12 sensitivity of 4.6 on 32 children.
- Maturity: within 12 months as a descriptive instrument on one cohort. Trial endpoints for rare hyperkinetic disorders need cohorts outside the scope and are not claimed.

U11. Brain-body coupling atlas (new).
- Input: every in-scope synchronized session. Output: per disorder, site, neural descriptor, body descriptor, task and state: gain (standardized where the movement channel is unitless), lag with interval, direction where the fast regime supports it, null percentile, number of subjects, artefact status, and rows reported as absent where G19 kills.
- Evidence gate: G15, G19, G20.
- Maturity: first edition within 12 months, covering PD hold, move and rest (ds004998), PD gait (multimodal_fog, sensefog, slow regime), grip (merk), dystonia under stimulation (oxford), and wearable wake as an artefact control (seizeit2). No sleep rows, and no healthy gait rows: physionet_multimodal_gait failed G15 on 2026-09-28 and serves PRE only.

## 6. New gates with kill numbers

Conventions: a kill fires on the point estimate, the subject-clustered bootstrap interval is reported beside it, sets under 30 patients are reported without kill, and every gate has three outcomes: pass, absent (the interval excludes the pass margin from above), undetermined (the interval contains both 0 and the margin). Only "absent" demotes a use case (RT).

- G15, synchronization and contract integrity, before any coupling.
  - sensefog: the authors' alignment is one hand-read stimulation-artefact sample per session, so checking the same artefact is circular. G15 requires an independent anchor: a second artefact or tap edge at session end where one exists, or cross-correlation of a shared physiological artefact (the ECG in LFP and EEG) at session start and end, from which offset and drift in ppm are estimated. Where only one anchor exists, only lags larger than 4 ms plus the worst-case drift over the session are admissible, and no direction claim is made at beta-phase scale. The Percept group delay is estimated from the artefact edge first. Checked on 2026-09-28: independent anchors exist, and the admissible-lag floor of each session is in reports/sync_sensefog/.
  - physionet_multimodal_gait: EEG, IMU and force lags from shared events constant within 20 ms across the 3 speeds in at least 90 percent of subjects, otherwise the set serves PRE only. On 2026-09-28 it failed (the lags are not constant across speeds, reports/sync_physionet_gait/): PRE only, and F3 does not run on it.
  - seizeit2: cross-correlate the EEG-module and ECG-EMG-module accelerometer magnitudes in 10 min windows at the start and end of every run; cross-module use of a run is void if the offset exceeds 40 ms or the drift across the run exceeds 40 ms. Runs last up to 87,192 s, so drift matters as much as offset. On 2026-09-28, 183 of the 2,850 runs pass (399.7 h), 901 are void and 1,766 cannot be checked (reports/sync_seizeit2/). Its accelerometer channels are not scaled in g (resting norm median 0.168), so a fixed-threshold body descriptor of Q8 cannot read them as g.
  - oxford (corrected rule proposed on 2026-09-28, owner decision OD11 of docs/EXECUTION_PLAN.md): the stimulation onset and offset are taken from the LFP stimulation artefact itself, and the command channel gives the amplitude only. The rule first written here (command channel and LFP artefact within one sample in every ramp file, otherwise the file is excluded) failed in all 62 ramp files on 2026-09-28: the command precedes the first artefact pulse by 200 to 260 ms in Am1 files and 5 to 41 ms in Cur1 files, with up to 188 samples of variation within a file (reports/oxford_lfp/). The ON and OFF block structure of all 62 ramp files is tabulated there: 46 ascending ramps and 16 single-block files, no down-ramp, stimulation at 130 Hz.
  - ds004998 and merk: matching sample counts per run.
  - Kill: any coupling estimate from a set or session that has not passed G15 is void.
- G16, neural encoders against baselines reproduced locally. Baselines measured in N0 on the same patient-level folds: log band power per channel with a regularized linear model; a published compact convolutional EEG network trained from scratch; a published staging network for sleep. Probes on the frozen encoder with patient-level cross-fitting, lockbox read once: cap_sleep 5-stage staging (kappa), ds004584 PD against control (AUROC), ds007526 UPDRS-III (only if its amplitude is traced). Decision on the paired, subject-clustered bootstrap interval of the encoder-minus-baseline difference: pass when the lower bound is above 0 on 2 tasks, kill when the upper bound is below 0 on 2 tasks, otherwise undetermined and the fixed descriptors serve with that label. The minimum detectable difference per task is published before training. The LFP front end is selected on rest against move or hold blocks, or nested inside each outer fold, never on the medication labels that Q4 and G21 report (RT).
- G17, the body is unchanged by the neural branch. With the branch attached, on every synchronized record and with a fixed seed, body posterior samples and every descriptor equal the body-only run (joint rotations within 1e-6 rad; descriptors identical at reported precision) with neural channels present, masked or replaced by noise; design G1 to G14 recomputed with the branch attached reproduce their numbers. DBS-evoked cortical potentials are reported as stimulation responses and never as subcortical state. Kill: any difference; the branch is detached until the leak is found.
- G18, CODY untouched. The neural manifest builder rejects the CODY namespaces; the canary check runs on neural manifests; the hashed subject list of every neural checkpoint and coupling fit contains no CODY id; no descriptor formula, threshold or normalization changes after the rule-file hash that precedes a CODY read. The CODY label census is descriptive, computed once and hashed into the rule file, and G18 scans the rule file for any number derived after that census. Kill: any CODY id in a neural manifest, checkpoint or fit, or any threshold traceable to the census.
- G19, coupling above a within-subject null on held-out subjects. Pre-registered families: F1 PD hold and move (ds004998), STN beta to EMG tremor-band power and movement-block onsets, fast and slow parts declared separately; F2 PD gait (multimodal_fog, sensefog), slow regime only, no gait-phase claim (prior-only artefact regressor); F3 healthy gait (physionet_multimodal_gait), EEG to gait phase after G20, not run since the set failed G15 on 2026-09-28 and serves PRE only; F4 grip (merk), ECoG and STN beta and gamma to grip force, gains standardized; F5 dystonia under stimulation (oxford), 3 to 12 Hz LFP to neck EMG envelope and head phasic power, fast regime, counts without kill; F8 macaque (dandi_m1_mptp_units), M1 units to EMG and elbow torque, fast regime only, as a methods check. The draft's F6 (behind-the-ear EEG to deltoid EMG) and F7 (sleep stage to chin tone) are deleted: the first is muscle and movement artefact rather than corticomuscular coupling, the second is a scoring rule (RT). seizeit2 wake data instead serves as a pre-registered artefact positive control, where coupling is expected to vanish after artefact regression. Metric: held-out gain in explained variance of the lagged neural terms over the body's own-history model, per held-out subject, against the null, with at least 20 admissible nulls per subject. Pass: gain above the null 95th percentile in at least 60 percent of held-out subjects, and a population lag interval excluding 0 wherever a direction is claimed. Kill: gain at or below the null median in at least 50 percent; the family is reported as absent and removed from U6, U8 and U11. If F8 does not recover cortical activity leading EMG in both macaques in the fast regime, direction readouts are withdrawn for every fast-regime family.
- G20, motion-artefact control, a kill gate only where a head sensor exists on the recording: physionet_multimodal_gait and oxford (RT). Since physionet_multimodal_gait failed G15 on 2026-09-28 and serves PRE only, the kill gate runs on oxford alone. Semi-synthetic ground truth: a known gait-phase-locked beta modulation and a known movement-related desynchronization are injected into recorded EEG at pre-registered depths; after regression, at least 80 percent of the injected phase-locked modulation must be recovered (kill below 60), while power at the gait fundamental and its first three harmonics falls by at least 50 percent (kill below 25). A second ground truth injects walking artefact predicted from another subject's body accelerations into seated EEG (sensefog sitting sessions) and scores both the artefact removed and the recovery of the injected clean signal (RT). multimodal_fog is excluded from the harmonic criterion (it arrives ICA-cleaned). sensefog and multimodal_fog carry the label "prior-only regressor" and support no gait-phase claim. On oxford, head-movement harmonics in LFP and EEG fall by at least 50 percent while the stimulation-induced low-frequency change keeps 80 percent. Neck EMG or ICA muscle components serve as a second regressor.
- G21, neural premium, the analogue of design G12. Joint body-and-neural against body-only models at equal parameter count and data, same folds, three seeds, for two pre-registered targets: within-subject medication state (ds004998, against the order-only null) and freezing 2 s ahead (multimodal_fog). The draft's third target (seizure motor onset on seizeit2) is removed, because training either model on seizure labels contradicts the datacard rule and this document (RT); where a seizeit2 comparison is wanted, it is zero-parameter (a fixed EEG-onset rule plus a fixed body rule against the body rule alone) and is reported as evaluation. Outcomes: pass (gain of at least 0.05 beyond the subject-bootstrap interval), absent, undetermined; the minimum detectable gain per set is published in N0. Only "absent" demotes a use case.
- G22, release audit. On every released neural artefact: the hashed training manifest contains no BY-SA source (oxford, and the stimulation arm fitted on it), no source of tier INTERNAL, EVAL or reference only (among them goodwin_smm, asdpose, osf_fmd_eeg, osf_fmd_kinematics, mini_rgbd, pd_biostamprc21, kaggle_fog and panelapp) and no EVAL label; no per-record embedding ships; Percept dates and serials are absent; merk keeps its licence-matrix row (added on 2026-09-24); the identity probe of design section 8 is extended to released neural descriptors. Kill: any violation blocks the release.

## 7. Owner decisions

Fifteen held datasets had no written licence, restrictive terms or unverified terms (DS before 2026-09-24). The licence of each was read at its source and re-checked by a second reader (reports/licence_research/licence_verdicts.md, scripts/licence_verdicts.py), and the session decided each one on 2026-09-24 on the owner's explicit delegation in chat (docs/decisions/licence_decisions_2026-09-24.csv). Each row below gives the decision, its conditions and what it unlocks here. The rows are ordered by what they change here.

Decisions of 2026-09-24. Admitted: pd_biostamprc21, kaggle_fog, ninapro, leuven_cp_gait, ataxie_kinect_dib (CC BY 4.0 via the article; supplementary table only, no skeleton data), kamath_snpc_geo (NCBI GEO, no restriction), clinvar (NCBI policy, attribution, never for diagnosis) and mini_rgbd. Admitted for internal research only, with no data, derived data or weights released and results reported with citation: goodwin_smm, asdpose, osf_fmd_eeg and osf_fmd_kinematics. Reference only: panelapp (Genomics England terms, non-commercial, all rights reserved), never inside a released artefact. Excluded: kaggle_spiral, whose CC BY-NC-ND label was set by an uploader who is not the creator, and mobile_gaitlab, counted as not held because the disk holds only its code and a testing-only Gillette demo. The non-commercial admissions (pads, mini_rgbd, panelapp) hold while MD_FM stays non-commercial research, which the design already assumes by releasing weights under CC BY-NC-SA. After the decisions, DS counts 100 datasets in scope (54 published, 46 usable_not_published), 71 not held, 1 excluded, 0 rights_unclear and 1 tool, and all 46 harmonized datasets are in scope.

| id | GB | decision of 2026-09-24 and its conditions | what it unlocks here |
|---|---|---|---|
| asdpose | 12.9 | admitted for internal research only, tier INTERNAL: the data sharing statement gives access to everyone and analyses for any purpose, with no explicit data licence; no data, derived data or weights released; results reported with citation | the only realistic stereotypy source held on disk: 241 children, 7,352 annotated segments, enough for a within-source rhythmicity and bout-duration analysis (4.9) |
| goodwin_smm | 0.41 | admitted for internal research only: no licence, a citation request; tier INTERNAL and outside the backbone, as the design already holds it | 6 subjects, 33.6 h, three labelled stereotypy classes as an internal sanity set; nothing changes in the design |
| ninapro | 41.7 | admitted: CC0 on Dryad; the held zips come from the project site, and their equivalence to the Dryad copy is recorded before any derived release | a healthy hand EMG reference (201 records, 67 subjects) for the co-contraction index of 4.2, beside gait120 limb EMG and cervical_motion_multiplanar neck EMG |
| pd_biostamprc21 | 17.2 | admitted: CC BY 4.0 per DataCite and the IEEE DataPort terms; tier EVAL, never trained | the design's G11 medication test on its 12 PD with tagged OFF then ON windows (5 PD untagged; 34 subjects in total, 17 PD and 17 controls) and the between-day minimal detectable change for PD (U2) |
| kaggle_fog | 133.6 | admitted, tier EVAL: the Kaggle rules allow any use and forbid redistribution of the Kaggle copy; the sponsor's full copy on Zenodo (10.5281/zenodo.10959560, CC BY 4.0) is the only route for any redistribution; its CONTRAST half trains only the INTERNAL comparison session head of stage 4, as the design describes | 136 subjects (80 of the harmonized 101 with ON and OFF visits, design section 7) for that session head, the kaggle_fog line of design G11 and the cross-device freezing transfer (U8); the levodopa operator stays fitted on weargait_pd and li_taati |
| osf_fmd_eeg | 0.054 | admitted for internal research only: no licence on the OSF node; patient data stays local, aggregate results only, with citation | derived time-frequency EEG in 18 FMD and 19 controls with S-FMDRS, for an internal functional-disorder neural analysis |
| osf_fmd_kinematics | 0.117 | admitted for internal research only, on the same terms as osf_fmd_eeg | head, eye and gaze kinematics in 10 participants, 3 sessions, for an internal functional-disorder body analysis |
| leuven_cp_gait | 0.036 | admitted: CC0; 357 children with CP and 56 typically developing children per its source | spastic CP Vicon curves as a partial paediatric kinematic reference (4.6) |
| mobile_gaitlab | 0.012 | excluded and counted as not held: the disk holds only the code and a testing-only Gillette demo; the 1,026-patient set is not held (licence research 2026-09-24) | nothing |
| mini_rgbd | 7.3 | admitted: CC BY-NC-SA 4.0, tier INTERNAL; its SMIL model files are never shipped | 12 infant sequences, unused in version 1 because infants are excluded |

Two further items that are not licence questions:

- fog_turning_filho video, fetched. Videos.zip (figshare item 14984667 v7, CC BY 4.0, 5,011,298,150 bytes) was fetched on the owner's go on 2026-09-24 into a local cache outside the git tree, and its size and MD5 match the figshare record. The inventory (scripts/probe_fog_videos.py, reports/fog_videos/) holds 73 MP4 videos of 35 patients at about 29.98 frames per second with no constant frame interval; 71 pair with the 71 IMU turning trials, and every pair needs its own offset. Per the data paper (PMC8904564), one Sony camera filmed the lower limbs and one Physilog 5 at 128 Hz sat on the shank of the more affected side. The per-sample freezing flag and the PDFEinfo episode table are one consensus annotation of the video by two specialists, transferred to the IMU clock by an undocumented procedure: they are specialist video labels, never IMU-derived, and they carry per-episode timing. Design gate G14, the only real video-versus-IMU pair on patients, runs on these pairs in its form corrected on 2026-09-24; keypointing of the video waits on the extraction of Videos.zip, an owner action (the environment of OD4 exists). Until G14b passes, the cross-sensor thesis reads "the same quantities where cross-sensor agreement has been measured", and 2D-only readouts, and every lower-limb-only video readout, carry a metric-scale-from-prior flag.
- Registry and licence hygiene that blocks release work: a licence-matrix row for merk_neumann_ecog_lfp, the cervical_motion_multiplanar probe class, and the mrc_lfp_dystonia duplicate (1.6). The first and the third were applied by the owner on 2026-09-24 from the verified patch in docs/moveatlas_corrections/ and committed as MoveAtlas f2e6f99; the second is a probe error to fix in scripts/ of this tree (1.6).

## 8. Risks and exclusions

Risks and early indicators:
- Small n is the defining risk. The dystonia analysis rests on 4 patients, the levodopa analysis on 20, the freezing analysis on 12 and 4. Indicator: G19 returns undetermined rather than pass or absent on F1 and F5, in which case the year-1 output is a set of intervals and a methods paper, not a finding.
- The oxford task during ramps is undocumented and the ON blocks last tens of seconds (a median 65 s in P2 and 41 s in P3, reports/oxford_lfp/), so the design can support an acute contrast and nothing more. No ramp descends (46 ascending ramps, 16 single-block files), so stimulation amplitude and time stay collinear within a file. A tonic change cannot be attributed to stimulation if posture instructions changed between steps.
- Accelerometers do not see yaw, so rotational torticollis is unmeasured in oxford.
- Artefact: stimulation artefact in EMG and accelerometers, motion artefact in LFP and EEG. G20 is a kill gate only where a head sensor exists, which is oxford alone since physionet_multimodal_gait failed G15 on 2026-09-28 and serves PRE only.
- The ds004998 medication order is confounded with time on the table in 18 of 20 subjects; without the order-only null the whole levodopa arm would be uninterpretable.
- seizeit2 seizures cluster: 4 subjects hold 63 percent of the hyperkinetic events, so every seizure-level number must be patient-clustered.
- Undeclared units and references (Percept, sensefog EEG, ds007526, merk force, seizeit2 accelerometers not scaled in g) force ratios and standardized gains, and refusals where needed.
- Multiple comparisons: six confirmatory primaries with Holm over six (F3 cannot run since physionet_multimodal_gait failed G15 on 2026-09-28; the family stays at six unless the owner names a replacement), the false discovery rate controlled within family, and no post hoc family promoted to a claim. ds004998 and oxford are each reused by several exploratory analyses, which is why only one analysis per dataset is confirmatory.
- Coupling is not causation. The only interventional leverage in scope is stimulation dose (oxford) and levodopa (ds004998); everything else is observational.
- EEG is biometric: per-record embeddings stay unreleased (G22).
- Licences: BY-SA isolation for oxford, the merk licence row (added on 2026-09-24), the four unlicensed deposits admitted for internal research only, and the non-commercial admissions, which hold only while MD_FM stays non-commercial research (section 7).
- Environment: the base interpreter has no mne and no torch, and the only sm_120 torch environment is the CODY environment the owner forbids extending. Every FIF and GPU step is blocked on owner question 4; the oxford work is not.
- Registry defects (1.6) outside the owner's MoveAtlas commit f2e6f99 propagate until the owner corrects them.

Deliberately excluded:
- Neural input to the body posterior, a joint brain-body latent, and any neural-to-body generative decoder.
- Inference of unrecorded structures: no pallidal or subthalamic state from scalp EEG, and no nucleus label on a contact without imaging.
- Spikes as encoder input; animal data in any encoder (the macaque sets are methods checks only).
- Imaging (dystonia_mri_ds006395 and the catalogued imaging sets) and fNIRS.
- seizeit2 seizure labels in any training, including in G21.
- Sleep claims beyond a staging probe and descriptive REM tone; no sleep coupling family.
- Closed-loop control, any clinical decision function, and any programming-support claim.
- Decoding of intended movement.
- Botulinum toxin (unchanged from the design): cdystonia and dystonia_alcohol_bfm enter no stage, and no toxin operator exists.
- Tics, ballismus and athetosis as trained channels; stereotypy as a released channel (asdpose, its only realistic source, is admitted for internal research only, 4.9).

## 9. Provenance and what remains unverified

Inputs: the three inventory passes of 2026-09-24 (neural, hyperkinetic, science), the two red-team passes of the same day, the design, FB, docs/MISSION_PROMPT.md section 6, the measured data scope (reports/data_scope/, scripts/data_scope.py against MoveAtlas a5d3814, run 2026-09-24), and MoveAtlas itself, read-only.

Checks made by this pass (read-only scripts in the session scratchpad, to be ported to scripts/ in stage N0):
- Manifest totals and per-source counts: 3,634 EEG records, 566 subject ids, 11,995.7 h; per-source records, subjects, hours and record durations; 37.9 h of PD and healthy EEG over 406 ids; 9.3 h in the three synchronized sets; physionet_multimodal_gait records all 59.87 s; multimodal_fog record seconds p10 25.5 and median 79.5; IMU sites per source, with a head sensor in all 167 physionet_multimodal_gait records and none in sensefog, multimodal_fog or fog_turning_filho.
- oxford_dystonia_lfp: the channel survey of all 68 files summarized per hemisphere (8 ramp hemispheres of 4 patients with an accelerometer, 6.69 h of 8.38 h; 6 hemispheres of 3 patients with EMG; 5 single-pulse files; one paired file; P1 and P4 without labelled movement channels), and per-file durations.
- ds004998: the acq_time census of all scans.tsv files (18 of 20 subjects MedOff before MedOn, gaps 21 to 77 min, 2 interleaved), and the run table by state and task (11 of 20 subjects with at least two MedOff runs of the same task).
- seizeit2: the census of all 2,850 events files (883 seizures; hyperkinetic 270 in 30 subjects, top 4 subjects 63 percent; explicitly non-motor 324 in 51; tonic 26 in 10; 11 subjects with both hyperkinetic and explicitly non-motor), and the README sensor placement.
- modys_video: 161 rows, 34 ids, 95 videos, T0 34 videos, T3 29, T12 32, 29 ids at three time points, 32 with T0 and T12, 66 videos with two scorers.
- pallidal_units_dystonia: 1,714 units, 32 patients, 38 unit-feature columns, patients and units per gene, and the absence of age, anaesthesia and centre columns.
- merk_neumann_ecog_lfp: 73 runs, durations 39 to 569 s with a median of 288 s; force z-scored per the datacard.
- fog_turning_filho: IMU.zip holds 106 text and 106 spreadsheet files, and the text header carries a per-sample freezing flag.
- Environment: h5py 3.12.1 and scipy 1.13.1 present, mne and torch absent.

Per the program's rules, every number in a document must come from a script in scripts/ with its output in reports/ and the MoveAtlas commit it was measured against. The data-scope numbers (reports/data_scope/, after the licence decisions of 2026-09-24) and the fog_turning_filho video inventory (scripts/probe_fog_videos.py, reports/fog_videos/) meet that rule already; since 2026-09-28 so does the oxford_dystonia_lfp channel survey (scripts/probe_oxford_lfp.py, reports/oxford_lfp/); the other checks above do not yet, and porting them is the first task of stage N0. Until then this addendum is a proposal, not a measured document.

Unverified at this date: the list of 1.7, the compute estimates of 2.8, and the literature statements carried from SI and FB, several of which those passes flag as recalled and not verified.

## Appendix. Outside the scope by owner decision

Everything in this appendix is excluded from the plan. No gate, no question and no use case above depends on any of it. It is recorded so that the cost of the scope is visible and so that nothing has to be rediscovered if the scope changes.

### A1. Datasets not held (71 in DS), the ones that were load-bearing in the draft

- Invasive and neural: oxford_stn_onoff_dbs (about 16 patients, 30 hemispheres ON and OFF levodopa, 26 during 130 Hz DBS, CC BY-SA), oxford_stn_lfp, dryad_stn_kinematics, dryad_vim_et_intraop, dabi_movement_ephys (the only Tourette CM and GPi recordings found, the adaptive DBS gait trials, a paediatric dystonia placeholder), beijing_sleep_lfp (121 patients across five disorders, 140 nights; the owner decided on 2026-08-25 not to send the request), tuh_eeg, openneuro_pd_rest_eeg_768, natmeg_pd, lancaster_prehd, camcan, mass_sleep, nsrr_plm_psg, iceberg_icm, ebrains_rat_lfp_atlas, gin_pd_rat_dbs, allen_cell_types, mrc_lfp_dystonia (a registry duplicate of a held deposit).
- Hyperkinetic and paediatric: dystonia_coalition (over 3,000 participants with standardized exam videos), paediatric_pallidal_lfp_cohorts, pedidbs, child_dbs, pedsdbs_tt, eds_spectrum_study (609 patients, 74 genes), eds_figshare_videos, atp1a3_video_archive (88 patients), mindts_tourette, osf_tic_video_46vxm, enroll_hd_family, cis_pd, real_pd, nbia_tircon, dbsmatchmaker, hyperkinetic_trial_cohorts, family_mediated_registries, gstt_paediatric_videos, tim_tremor (withdrawn at source), mdsgene, shanghai_child_gait3d, mpower, ppmi_sensors, watch_pd, ppp_radboud, and the remaining catalogued entries of DS.

### A2. Acquisitions proposed and dropped

- The prospective dystonia GPi-DBS cohort with sensing, video, IMU and EMG (the only route to the DBS latency hypothesis, to GPi biomarkers against body targets and to a programming use case). Its statistics, if it ever returns, must use the corrected forms: Spearman sizes rather than Pearson (33 evaluable at r = 0.5 and 51 at r = 0.4, enrolling 41 or 63 at 20 percent attrition); one pre-specified early predictor or Holm across two; recordings at 0, 30, 60 and 120 min after first activation, because a first visit at 24 h leaves the fast time constant unidentifiable; two baseline sessions so that the predictor's baseline is independent of the outcome's; a ratio test stated as greater than 1 rather than at least 5; stratification by etiology; an interval-censored kinetic model with ties and censored cases defined before enrolment; power reported across sign-test probabilities of 0.65 to 0.8; the neural-against-body predictor comparison sized by Spearman simulation with the measured test-retest error; sizing by simulation under the actual visit schedule. Governance: home monitoring by IMU only until an on-device keypoint pipeline exists and is validated, video kept on a site-managed device, child assent with parental consent, a bystander rule, a weekly burden limit, and an explicit statement that no research procedure changes the timing of stimulation or programming.
- The status dystonicus arm, with deferred consent, a no-delay-to-care clause, sedation and neuromuscular blockade as covariates, body readouts masked during paralysis, and relapse after weaning rather than detection as the endpoint.
- The toxin cohort (60 cervical dystonia at weeks 0, 4 and 12 with TWSTRS and neck EMG).
- The in-house paired prospective cohort of the design, and the neural add-on to it (EEG and EMG for jerks, sensory trick and voluntary mimics).
- Percept chronic streaming in the owner's own patients.
- Every portal, application or author request: the MRC portal, Dryad, DABI, TUH, the Dystonia Coalition, the paediatric pallidal cohorts, and requests to depositors for missing covariates such as the age and anaesthesia of the pallidal-units cohort.

### A3. Questions that left the plan with their data

- The time constants of phasic and tonic change after GPi DBS, and early prediction of the slow course (needs the prospective cohort).
- Which body component pallidal theta-alpha tracks, compared with the scale total (needs GPi sensing with a body and a same-day scale).
- Detection and relapse prediction of status dystonicus (needs paediatric chronic pallidal sensing).
- Whether cortical or subthalamic gamma leads body-measured dyskinesia (needs chronic sensing at home with wearables).
- Tics against voluntary mimics, and the premonitory interval (needs a Tourette cohort with EEG, EMG and IMU).
- The timing of the sensory trick (needs a dedicated protocol).
- Genotype to body phenotype mapping, stage 2 (needs genotyped standardized video or IMU).
- Premanifest Huntington disease, infant general movements, and any progression claim (need cohorts that are not held).
